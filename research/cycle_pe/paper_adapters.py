"""Lazy adapters for the public BREC v3 and ZINC-12K benchmarks.

Neither public dataset is downloaded by the built-in CycleCount-OOD tests.
ZINC is loaded through its official PyTorch Geometric split implementation;
BREC uses the official graph6 ``brec_v3.npy`` artifact and keeps RPC pairs
lazy so a 51,200-graph file is never tensorized all at once.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import networkx as nx
import numpy as np

from chartgat.cache import atomic_write_json
from research.cycle_pe.paper_data import (
    DatasetBundle,
    PaperGraph,
    canonical_edges,
    sha256_file,
)

BREC_SOURCE_URL = "https://github.com/GraphPKU/BREC"
BREC_RAW_URL = "https://raw.githubusercontent.com/GraphPKU/BREC/Release/BREC_data_all.zip"
ZINC_SOURCE_URL = (
    "https://pytorch-geometric.readthedocs.io/en/latest/generated/"
    "torch_geometric.datasets.ZINC.html"
)
PYG_INSTALL_URL = "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"

BREC_CATEGORIES = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}

BREC_OFFICIAL_NUM_RELABEL = 32
BREC_OFFICIAL_PAIR_COUNT = 400
BREC_OFFICIAL_RECORD_COUNT = 4 * BREC_OFFICIAL_NUM_RELABEL * BREC_OFFICIAL_PAIR_COUNT
ZINC_SPLIT_SIZES = {"train": 10_000, "validation": 1_000, "test": 1_000}

_BREC_DOWNLOAD_LIMIT = 512 * 1024 * 1024
_BREC_EXTRACT_LIMIT = 512 * 1024 * 1024
_BREC_ARCHIVE_MEMBER_LIMIT = 10_000
_BREC_ARCHIVE_TOTAL_LIMIT = 1024 * 1024 * 1024
_BREC_DOWNLOAD_HOSTS = {
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "github.com",
}


def _load_brec_records(path: Path) -> np.ndarray:
    try:
        records = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to load BREC graph6 records from {path}") from exc
    if records.ndim != 1 or len(records) < 1:
        raise RuntimeError("BREC artifact must be a non-empty one-dimensional NumPy array")
    return records


def _require_pyg_zinc() -> type:
    try:
        from torch_geometric.datasets import ZINC
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ZINC-12K is optional and requires PyTorch Geometric. Install a "
            "PyTorch build matching the target CUDA runtime, then run "
            "`python -m pip install torch-geometric`; use the wheel matrix at "
            f"{PYG_INSTALL_URL}. The `core` suite does not require PyG."
        ) from exc
    return ZINC


def _one_hot(values: np.ndarray, width: int, *, name: str) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= width):
        raise RuntimeError(f"unexpected {name} category outside [0, {width - 1}]")
    return np.eye(width, dtype=np.float64)[flat]


def _pyg_zinc_graph(data: Any, *, graph_id: str, split: str) -> PaperGraph:
    num_nodes = int(data.num_nodes)
    node_features = _one_hot(data.x.detach().cpu().numpy(), 28, name="ZINC atom")
    edge_index = data.edge_index.detach().cpu().numpy()
    raw_edge_attr = data.edge_attr.detach().cpu().numpy()
    attributes: dict[tuple[int, int], int] = {}
    for column in range(edge_index.shape[1]):
        u, v = int(edge_index[0, column]), int(edge_index[1, column])
        if u == v:
            continue
        edge = (min(u, v), max(u, v))
        category = int(np.asarray(raw_edge_attr[column]).reshape(-1)[0])
        previous = attributes.setdefault(edge, category)
        if previous != category:
            raise RuntimeError("directed copies of a ZINC bond disagree on bond type")
    edges = tuple(sorted(attributes))
    edge_features = _one_hot(np.asarray([attributes[edge] for edge in edges]), 4, name="ZINC bond")
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    if not nx.is_connected(graph):
        raise RuntimeError(f"ZINC molecule {graph_id} is unexpectedly disconnected")
    target = np.asarray(data.y.detach().cpu().numpy(), dtype=np.float64).reshape(1)
    return PaperGraph(
        graph_id=graph_id,
        split=split,
        family="zinc_molecule",
        num_nodes=num_nodes,
        edges=edges,
        node_features=node_features,
        edge_features=edge_features,
        graph_targets=target,
    )


def _zinc_cache_ready(root: Path) -> bool:
    processed = root / "subset" / "processed"
    processed_ready = all(
        (processed / f"{split}.pt").is_file() for split in ("train", "val", "test")
    )
    raw = root / "raw"
    raw_names = (
        "train.pickle",
        "val.pickle",
        "test.pickle",
        "train.index",
        "val.index",
        "test.index",
    )
    return processed_ready or all((raw / name).is_file() for name in raw_names)


def _zinc_cache_hashes(root: Path) -> dict[str, str]:
    candidates = [
        *(root / "subset" / "processed" / f"{split}.pt" for split in ("train", "val", "test")),
        *(
            root / "raw" / name
            for name in (
                "train.pickle",
                "val.pickle",
                "test.pickle",
                "train.index",
                "val.index",
                "test.index",
            )
        ),
    ]
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }


def load_zinc12k(data_root: Path, *, allow_download: bool = False) -> DatasetBundle:
    """Load PyG's official 10k/1k/1k ZINC subset partitions."""

    zinc_class = _require_pyg_zinc()
    root = data_root.expanduser().resolve() / "ZINC12K"
    if not allow_download and not _zinc_cache_ready(root):
        raise FileNotFoundError(
            f"No complete PyG ZINC-12K cache was found at {root}. Copy an existing "
            "official cache there, or explicitly permit the PyG download with "
            f"`--allow-download`. Loader documentation: {ZINC_SOURCE_URL}"
        )
    requested = {"train": "train", "validation": "val", "test": "test"}
    splits: dict[str, list[PaperGraph]] = {}
    official_sizes: dict[str, int] = {}
    try:
        for split, pyg_split in requested.items():
            dataset = zinc_class(root=str(root), subset=True, split=pyg_split)
            official_sizes[split] = len(dataset)
            if len(dataset) != ZINC_SPLIT_SIZES[split]:
                raise RuntimeError(
                    f"ZINC-12K {split} must contain {ZINC_SPLIT_SIZES[split]} graphs, "
                    f"found {len(dataset)}"
                )
            splits[split] = [
                _pyg_zinc_graph(
                    dataset[index], graph_id=f"zinc12k:{split}:{index:05d}", split=split
                )
                for index in range(len(dataset))
            ]
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to prepare ZINC-12K at {root}. Ensure outbound access for the "
            "first PyG download or copy an existing PyG ZINC cache there. Official "
            f"loader documentation: {ZINC_SOURCE_URL}. Original error: {exc}"
        ) from exc
    return DatasetBundle(
        name="ZINC-12K",
        splits=splits,
        graph_target_names=("constrained_solubility",),
        metadata={
            "adapter": "torch_geometric.datasets.ZINC(subset=True)",
            "source_url": ZINC_SOURCE_URL,
            "official_split_names": requested,
            "official_split_sizes": official_sizes,
            "loaded_split_sizes": {name: len(graphs) for name, graphs in splits.items()},
            "download_allowed": bool(allow_download),
            "cache_sha256": _zinc_cache_hashes(root),
        },
    )


def _category(pair_index: int) -> str:
    for name, (start, stop) in BREC_CATEGORIES.items():
        if start <= pair_index < stop:
            return name
    return "custom"


def _decode_graph6(record: Any) -> nx.Graph:
    value = record.item() if isinstance(record, np.ndarray) and record.ndim == 0 else record
    if isinstance(value, str):
        payload = value.encode("ascii")
    elif isinstance(value, (bytes, np.bytes_)):
        payload = bytes(value)
    else:
        raise RuntimeError(f"unsupported BREC graph6 record type: {type(value).__name__}")
    try:
        graph = nx.from_graph6_bytes(payload.strip())
    except (nx.NetworkXError, ValueError) as exc:
        raise RuntimeError("invalid graph6 record in brec_v3.npy") from exc
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
        raise RuntimeError("the static paper model currently requires connected BREC graphs")
    return graph


def _brec_graph(record: Any, *, graph_id: str, family: str) -> PaperGraph:
    graph = _decode_graph6(record)
    return PaperGraph(
        graph_id=graph_id,
        split="brec_rpc",
        family=family,
        num_nodes=graph.number_of_nodes(),
        edges=canonical_edges(graph.edges()),
    )


@dataclass(frozen=True)
class BRECPair:
    pair_index: int
    category: str
    train_test: tuple[PaperGraph, ...]
    reliability: tuple[PaperGraph, ...]


class BRECAdapter:
    """Lazy view of the official RPC layout (G/H and G/G permutation pairs)."""

    def __init__(
        self,
        path: Path,
        *,
        num_relabel: int = BREC_OFFICIAL_NUM_RELABEL,
        protocol: str = "official",
    ) -> None:
        if num_relabel < 2:
            raise ValueError("BREC RPC needs at least two relabelings")
        if protocol not in {"official", "custom"}:
            raise ValueError("BREC protocol must be 'official' or 'custom'")
        self.path = path.expanduser().resolve()
        self.num_relabel = int(num_relabel)
        self.protocol = protocol
        self._records = _load_brec_records(self.path)
        block = 4 * self.num_relabel
        if len(self._records) % block:
            raise RuntimeError(
                f"BREC artifact must contain 4*q records per pair (q={self.num_relabel})"
            )
        self.pair_count = len(self._records) // block
        if self.pair_count < 1:
            raise RuntimeError("BREC artifact contains no RPC pairs")
        if self.protocol == "official":
            if self.num_relabel != BREC_OFFICIAL_NUM_RELABEL:
                raise RuntimeError(
                    "official BREC requires q=32; use --brec-protocol custom for other q values"
                )
            if len(self._records) != BREC_OFFICIAL_RECORD_COUNT:
                raise RuntimeError(
                    "official BREC v3 requires exactly "
                    f"{BREC_OFFICIAL_RECORD_COUNT:,} records, found {len(self._records):,}"
                )
            if self.pair_count != BREC_OFFICIAL_PAIR_COUNT:
                raise RuntimeError(
                    "official BREC v3 requires exactly "
                    f"{BREC_OFFICIAL_PAIR_COUNT} pairs, found {self.pair_count}"
                )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "BREC v3 graph6/RPC",
            "source_url": BREC_SOURCE_URL,
            "raw_artifact_url": BREC_RAW_URL,
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "records": len(self._records),
            "pair_count": self.pair_count,
            "num_relabel": self.num_relabel,
            "protocol": self.protocol,
            "rpc_threshold": 72.34 if self.num_relabel == 32 else None,
            "categories": BREC_CATEGORIES,
            "official_shape_validated": self.protocol == "official",
            "official_source_hash_pinned": False,
            "hash_note": (
                "SHA-256 is recorded for provenance; GraphPKU/BREC does not publish a "
                "canonical SHA-256 in the Release runner or README."
            ),
        }

    def load_pair(self, pair_index: int) -> BRECPair:
        if not 0 <= pair_index < self.pair_count:
            raise IndexError("BREC pair index out of range")
        category = _category(pair_index)
        span = 2 * self.num_relabel
        train_start = pair_index * span
        reliability_start = (self.pair_count + pair_index) * span

        def decode_block(start: int, phase: str) -> tuple[PaperGraph, ...]:
            return tuple(
                _brec_graph(
                    self._records[start + offset],
                    graph_id=f"brec:{pair_index:03d}:{phase}:{offset:02d}",
                    family=category,
                )
                for offset in range(span)
            )

        return BRECPair(
            pair_index=pair_index,
            category=category,
            train_test=decode_block(train_start, "train_test"),
            reliability=decode_block(reliability_start, "reliability"),
        )


def validate_brec_v3(
    path: Path,
    *,
    protocol: str = "official",
    num_relabel: int = BREC_OFFICIAL_NUM_RELABEL,
) -> dict[str, Any]:
    """Parse and structurally validate a BREC artifact, returning provenance metadata."""

    return BRECAdapter(path, num_relabel=num_relabel, protocol=protocol).metadata


def _brec_candidates(data_root: Path) -> tuple[Path, ...]:
    root = data_root.expanduser().resolve()
    return (
        root / "BREC" / "Data" / "raw" / "brec_v3.npy",
        root / "Data" / "raw" / "brec_v3.npy",
        root / "brec_v3.npy",
    )


def find_brec_v3(data_root: Path) -> Path:
    candidates = _brec_candidates(data_root)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    locations = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "BREC v3 is absent and network access is fail-closed. Extract the official "
        f"BREC_data_all.zip from {BREC_SOURCE_URL}, place brec_v3.npy at one of "
        f"the paths below, or explicitly pass --allow-download:\n  - {locations}"
    )


def _validated_brec_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    infos = archive.infolist()
    if len(infos) > _BREC_ARCHIVE_MEMBER_LIMIT:
        raise RuntimeError("BREC archive has an unsafe number of members")
    if sum(info.file_size for info in infos) > _BREC_ARCHIVE_TOTAL_LIMIT:
        raise RuntimeError("BREC archive exceeds the uncompressed-size safety limit")
    matches: list[zipfile.ZipInfo] = []
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or (path.parts and ":" in path.parts[0])
        ):
            raise RuntimeError(f"unsafe path in BREC archive: {name!r}")
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise RuntimeError(f"symbolic link rejected in BREC archive: {name!r}")
        if info.file_size > _BREC_EXTRACT_LIMIT:
            raise RuntimeError(f"oversized member rejected in BREC archive: {name!r}")
        if info.compress_size and info.file_size > 1_000 * info.compress_size:
            raise RuntimeError(f"suspicious compression ratio in BREC archive: {name!r}")
        if not info.is_dir() and path.name == "brec_v3.npy":
            matches.append(info)
    if len(matches) != 1:
        raise RuntimeError("official BREC archive must contain exactly one brec_v3.npy member")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_brec_v3(data_root: Path) -> Path:
    """Explicitly download and safely extract the official BREC v3 artifact."""

    target = _brec_candidates(data_root)[0]
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_path: Path | None = None
    extracted_path: Path | None = None
    try:
        request = urllib.request.Request(
            BREC_RAW_URL,
            headers={"User-Agent": "cycle-pe-paper/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            final_url = response.geturl()
            parsed = urllib.parse.urlparse(final_url)
            if parsed.scheme != "https" or parsed.hostname not in _BREC_DOWNLOAD_HOSTS:
                raise RuntimeError(f"unsafe redirect while downloading BREC: {final_url}")
            length_header = response.headers.get("Content-Length")
            if length_header is not None and int(length_header) > _BREC_DOWNLOAD_LIMIT:
                raise RuntimeError("BREC download exceeds the compressed-size safety limit")
            with tempfile.NamedTemporaryFile(
                prefix="brec-v3-", suffix=".zip", dir=target.parent, delete=False
            ) as archive_stream:
                archive_path = Path(archive_stream.name)
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > _BREC_DOWNLOAD_LIMIT:
                        raise RuntimeError("BREC download exceeds the compressed-size safety limit")
                    archive_stream.write(chunk)
                archive_stream.flush()
                os.fsync(archive_stream.fileno())
        if archive_path is None:
            raise RuntimeError("BREC download did not create an archive")
        archive_sha256 = _sha256(archive_path)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                member = _validated_brec_member(archive)
                with (
                    tempfile.NamedTemporaryFile(
                        prefix="brec-v3-", suffix=".npy", dir=target.parent, delete=False
                    ) as destination,
                    archive.open(member) as source,
                ):
                    extracted_path = Path(destination.name)
                    extracted = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        extracted += len(chunk)
                        if extracted > _BREC_EXTRACT_LIMIT:
                            raise RuntimeError(
                                "BREC brec_v3.npy exceeds the extraction safety limit"
                            )
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
        except zipfile.BadZipFile as exc:
            raise RuntimeError("downloaded BREC artifact is not a valid ZIP file") from exc
        if extracted_path is None:
            raise RuntimeError("BREC extraction did not produce brec_v3.npy")
        with extracted_path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                raise RuntimeError("extracted brec_v3.npy has an invalid NumPy header")
        _load_brec_records(extracted_path)
        npy_sha256 = _sha256(extracted_path)
        if target.exists():
            # A concurrent successful preparation wins; never overwrite it.
            return target
        os.replace(extracted_path, target)
        extracted_path = None
        metadata = {
            "source_url": BREC_RAW_URL,
            "archive_sha256": archive_sha256,
            "brec_v3_sha256": npy_sha256,
            "bytes": target.stat().st_size,
        }
        metadata_path = target.with_name("brec_v3.download.json")
        atomic_write_json(metadata_path, metadata)
        return target
    finally:
        for temporary in (archive_path, extracted_path):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def load_brec_v3(
    data_root: Path,
    *,
    num_relabel: int = BREC_OFFICIAL_NUM_RELABEL,
    allow_download: bool = False,
    protocol: str = "official",
) -> BRECAdapter:
    root = data_root.expanduser().resolve()
    try:
        path = find_brec_v3(root)
    except FileNotFoundError:
        if not allow_download:
            raise
        path = download_brec_v3(root)
    return BRECAdapter(path, num_relabel=num_relabel, protocol=protocol)


__all__ = [
    "BRECAdapter",
    "BRECPair",
    "BREC_CATEGORIES",
    "BREC_OFFICIAL_NUM_RELABEL",
    "BREC_OFFICIAL_PAIR_COUNT",
    "BREC_OFFICIAL_RECORD_COUNT",
    "BREC_SOURCE_URL",
    "PYG_INSTALL_URL",
    "ZINC_SOURCE_URL",
    "download_brec_v3",
    "find_brec_v3",
    "load_brec_v3",
    "validate_brec_v3",
    "load_zinc12k",
]
