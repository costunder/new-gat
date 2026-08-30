"""Official PyG/OGB public adapters with verified caches and opt-in downloads.

The synthetic paper core has no optional dependencies.  Official
PascalVOC-SP and ogbg-molhiv data are touched only through this module and only
when a verified real cache exists or the caller explicitly allows downloading.
Missing public data never falls back to generated graphs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_write_json,
)

PUBLIC_SCHEMA_VERSION = 2
SOURCE_URLS = {
    "pascalvoc_sp": "https://github.com/vijaydwivedi75/lrgb",
    "ogbg_molhiv": "https://ogb.stanford.edu/docs/graphprop/",
}
OFFICIAL_SPLIT_SIZES = {
    "pascalvoc_sp": {"train": 8_498, "validation": 1_227, "test": 1_449},
    "ogbg_molhiv": {"train": 32_901, "validation": 4_113, "test": 4_113},
}


class OptionalDatasetDependencyError(RuntimeError):
    pass


class IndexedCollection(Sequence[dict[str, Any]]):
    """Lazy adapter over a PyG dataset and an official index split."""

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int] | Tensor,
        adapter: Callable[[Any, str], dict[str, Any]],
        prefix: str,
    ) -> None:
        self.dataset = dataset
        if isinstance(indices, Tensor):
            self.indices = [int(value) for value in indices.reshape(-1)]
        else:
            self.indices = [int(value) for value in indices]
        self.adapter = adapter
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        return self.adapter(self.dataset[source_index], f"{self.prefix}-{source_index}")


def deduplicate_undirected_edges(
    edge_index: Tensor, edge_features: Tensor | None, num_nodes: int
) -> tuple[Tensor, Tensor]:
    """Collapse reciprocal PyG arcs and remove incidence-zero self loops.

    Continuous directional attributes are averaged into one orientation-free
    physical-edge feature. Integer/categorical reciprocal attributes must
    agree exactly; silently selecting one category would corrupt chemistry.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if edge_features is None:
        edge_features = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)
    if edge_features.ndim == 1:
        edge_features = edge_features[:, None]
    if edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError("edge attributes and edge_index disagree")
    selected: dict[tuple[int, int], list[int]] = {}
    for column in range(edge_index.shape[1]):
        first = int(edge_index[0, column])
        second = int(edge_index[1, column])
        if first == second:
            continue
        if first < 0 or second < 0 or first >= num_nodes or second >= num_nodes:
            raise ValueError("edge endpoint outside graph")
        key = (min(first, second), max(first, second))
        selected.setdefault(key, []).append(column)
    if not selected:
        raise ValueError("graph has no non-self edges after incidence conversion")
    keys = sorted(selected)
    indices = torch.tensor(keys, dtype=torch.long).t().contiguous()
    attributes: list[Tensor] = []
    for key in keys:
        values = edge_features[selected[key]]
        if values.is_floating_point():
            attributes.append(values.mean(dim=0))
        else:
            reference = values[0]
            if not torch.equal(values, reference.expand_as(values)):
                raise ValueError(f"conflicting categorical reciprocal edge attributes for {key}")
            attributes.append(reference)
    return indices, torch.stack(attributes)


def adapt_pyg_graph(data: Any, graph_id: str, *, task: str) -> dict[str, Any]:
    x = data.x
    if x is None:
        raise ValueError(f"{graph_id} has no node features")
    x = x.detach().cpu()
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.detach().cpu()
    edges, attributes = deduplicate_undirected_edges(
        data.edge_index.detach().cpu(), edge_attr, int(x.shape[0])
    )
    y = data.y.detach().cpu()
    if task == "node":
        y = y.reshape(-1).long()
        if y.numel() != x.shape[0]:
            raise ValueError("PascalVOC-SP node labels do not match the node count")
    elif task == "graph":
        y = y.reshape(-1).float()
    else:
        raise ValueError("task must be node or graph")
    return {
        "graph_id": graph_id,
        "x": x,
        "edge_index": edges,
        "edge_features": attributes,
        "y": y,
        "task": task,
        "categorical": task == "graph",
    }


def _dependency_error() -> OptionalDatasetDependencyError:
    return OptionalDatasetDependencyError(
        "Official public suites require optional packages 'torch-geometric' and 'ogb'. "
        "Activate the dedicated Conda environment and run `bash scripts/setup_gpu.sh` "
        "from the repository root to install the exact GPU dependency pins. See "
        "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html "
        "and https://ogb.stanford.edu/docs/home/. The core S1-S4 suite does not need them."
    )


def _load_official(data_root: Path) -> dict[str, Any]:
    try:
        import torch_geometric  # noqa: F401
        from ogb.graphproppred import PygGraphPropPredDataset
        from torch_geometric.datasets import LRGBDataset
    except (ImportError, OSError) as error:
        raise _dependency_error() from error

    pyg_root = data_root / "pyg"
    pascal: dict[str, Any] = {}
    for split, official_split in (("train", "train"), ("validation", "val"), ("test", "test")):
        dataset = LRGBDataset(root=str(pyg_root), name="PascalVOC-SP", split=official_split)
        pascal[split] = IndexedCollection(
            dataset,
            range(len(dataset)),
            partial(adapt_pyg_graph, task="node"),
            f"pascal-{split}",
        )

    mol_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(data_root / "ogb"))
    split_indices = mol_dataset.get_idx_split()
    mol = {
        split: IndexedCollection(
            mol_dataset,
            split_indices[official],
            partial(adapt_pyg_graph, task="graph"),
            f"molhiv-{split}",
        )
        for split, official in (("train", "train"), ("validation", "valid"), ("test", "test"))
    }
    return {"fixture": False, "pascalvoc_sp": pascal, "ogbg_molhiv": mol}


def _processed_paths(datasets: dict[str, Any], root: Path) -> list[str]:
    paths: set[str] = set()
    for dataset_name in ("pascalvoc_sp", "ogbg_molhiv"):
        for split in ("train", "validation", "test"):
            collection = datasets[dataset_name][split]
            path_values = list(getattr(collection.dataset, "processed_paths", []))
            if not path_values and getattr(collection.dataset, "processed_dir", None):
                path_values = [collection.dataset.processed_dir]
            for path_value in path_values:
                path = Path(path_value).resolve()
                try:
                    paths.add(str(path.relative_to(root.resolve())))
                except ValueError:
                    paths.add(str(path))
    return sorted(paths)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processed_hashes(root: Path, paths: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path_value in paths:
        candidate = Path(path_value)
        resolved = candidate if candidate.is_absolute() else root / candidate
        files = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for path in files:
            if not path.is_file():
                continue
            try:
                key = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                key = str(path.resolve())
            hashes[key] = _file_sha256(path)
    return hashes


def validate_public_cache(data_root: Path | str) -> tuple[Path, dict[str, Any]]:
    """Validate the public marker and all recorded processed files without downloading."""

    public_root = Path(data_root).expanduser().resolve() / "conductance_gat" / "public"
    marker = public_root / "official-ready.json"
    if not marker.is_file():
        raise FileNotFoundError(f"conductance public cache marker is missing: {marker}")
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"invalid conductance public marker: {marker}") from error
    if manifest.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise CacheWrongRequestError(f"unsupported conductance public marker schema: {marker}")
    if manifest.get("fixture") is not False:
        raise CacheWrongRequestError(f"only official public data caches are supported: {marker}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(SOURCE_URLS):
        raise CacheCorruptError("conductance public marker has an invalid dataset set")
    for name, split_sizes in OFFICIAL_SPLIT_SIZES.items():
        if datasets[name].get("source_url") != SOURCE_URLS[name]:
            raise CacheWrongRequestError(f"conductance public source mismatch for {name}")
        if datasets[name].get("splits") != split_sizes:
            raise CacheCorruptError(
                f"conductance public split cardinalities are invalid for {name}"
            )
    required_paths = manifest.get("required_processed_paths")
    stored_hashes = manifest.get("processed_sha256")
    if not isinstance(required_paths, list) or not required_paths:
        raise CacheIncompleteError("conductance public marker has no processed-file inventory")
    if not isinstance(stored_hashes, dict) or not stored_hashes:
        raise CacheIncompleteError("conductance public marker has no processed-file checksums")
    missing = []
    for path_value in required_paths:
        path = Path(path_value)
        resolved = path if path.is_absolute() else public_root / path
        if not resolved.exists():
            missing.append(str(resolved))
    if missing:
        raise CacheIncompleteError("conductance public processed files are missing: " + missing[0])
    actual_hashes = _processed_hashes(public_root, required_paths)
    if actual_hashes != stored_hashes:
        raise CacheCorruptError("conductance public processed-file checksum mismatch")
    return marker, manifest


def prepare_public_data(
    data_root: Path | str,
    *,
    allow_download: bool = False,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    public_root = Path(data_root).expanduser().resolve() / "conductance_gat" / "public"
    marker = public_root / "official-ready.json"
    if not allow_download and not marker.exists():
        raise RuntimeError(
            "Official public data is not marked prepared. Run once with "
            "`--suite public --prepare-only --allow-download` to let the official "
            "PyG/OGB dataset classes download into --data-root. No download is "
            "attempted without that explicit flag. Generated substitutes are not supported."
        )
    if not allow_download:
        validate_public_cache(data_root)
    public_root.mkdir(parents=True, exist_ok=True)
    datasets = _load_official(public_root)
    split_sizes = {
        name: {split: len(datasets[name][split]) for split in datasets[name]}
        for name in SOURCE_URLS
    }
    if split_sizes != OFFICIAL_SPLIT_SIZES:
        raise RuntimeError(
            f"Official public split cardinalities do not match the pinned protocol: {split_sizes}"
        )
    required_paths = _processed_paths(datasets, public_root)
    manifest = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "fixture": False,
        "datasets": {
            name: {
                "source_url": SOURCE_URLS[name],
                "splits": split_sizes[name],
            }
            for name in SOURCE_URLS
        },
        "required_processed_paths": required_paths,
        "processed_sha256": _processed_hashes(public_root, required_paths),
    }
    atomic_write_json(
        marker,
        manifest,
        validator=lambda temporary: json.loads(temporary.read_text(encoding="utf-8")),
    )
    validate_public_cache(data_root)
    return datasets, marker, manifest


__all__ = [
    "IndexedCollection",
    "OptionalDatasetDependencyError",
    "SOURCE_URLS",
    "adapt_pyg_graph",
    "deduplicate_undirected_edges",
    "prepare_public_data",
    "validate_public_cache",
]
