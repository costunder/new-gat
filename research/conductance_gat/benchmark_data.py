"""Official GAT/GATv2 datasets for our conductance model; no generated fallback."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_publish,
    atomic_write_json,
)

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
CACHE_VERSION = 1
SOURCES = {
    "cora": "https://github.com/kimiyoung/planetoid/tree/master/data",
    "citeseer": "https://github.com/kimiyoung/planetoid/tree/master/data",
    "pubmed": "https://github.com/kimiyoung/planetoid/tree/master/data",
    "ppi": "https://graphsage.stanford.edu/",
    "ogbn-arxiv": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
}
EXPECTED = {
    "cora": {"nodes": 2708, "features": 1433, "classes": 7, "splits": [140, 500, 1000]},
    "citeseer": {"nodes": 3327, "features": 3703, "classes": 6, "splits": [120, 500, 1000]},
    "pubmed": {"nodes": 19717, "features": 500, "classes": 3, "splits": [60, 500, 1000]},
    "ogbn-arxiv": {
        "nodes": 169343,
        "features": 128,
        "classes": 40,
        "splits": [90941, 29799, 48603],
    },
    "ppi": {"features": 50, "classes": 121, "graphs": [20, 2, 2]},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_edges(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """One orientation per edge for B, plus a canonical adjacency representation."""
    edges = edge_index.detach().cpu().long()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be a 2 x E matrix")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError("edge endpoint is outside the graph")
    low, high = torch.minimum(edges[0], edges[1]), torch.maximum(edges[0], edges[1])
    keys = torch.unique(low[low != high] * num_nodes + high[low != high], sorted=True)
    incidence = torch.stack((keys.div(num_nodes, rounding_mode="floor"), keys % num_nodes))
    arcs = torch.cat((incidence, incidence.flip(0)), dim=1)
    # Preserve a sorted adjacency representation without materializing B.
    order = torch.argsort(arcs[0] * num_nodes + arcs[1])
    return arcs[:, order].contiguous(), incidence.contiguous()


def _graph(data: Any, *, normalize_features: bool) -> dict[str, Tensor]:
    x = data.x.detach().cpu().float().contiguous()
    if normalize_features:
        # Exactly the PyG NormalizeFeatures rule used for Planetoid datasets.
        x = x - x.min()
        x = x / x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    arcs, incidence = canonical_edges(data.edge_index, x.shape[0])
    return {
        "x": x,
        "y": data.y.detach().cpu().contiguous(),
        "edge_index": arcs,
        "incidence_edge_index": incidence,
    }


def _split_mask(indices: Tensor, num_nodes: int) -> Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices.reshape(-1).long()] = True
    return mask


def validate_payload(name: str, payload: dict[str, Any]) -> None:
    """Validate real cache tensors, including mandatory official dimensions/split sizes."""
    if name not in DATASETS or payload.get("dataset") != name:
        raise ValueError("unknown or mismatched dataset")
    splits = payload["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("all official splits are required")
    graphs = payload["graphs"]
    if not graphs:
        raise ValueError("empty benchmark cache")
    spec = EXPECTED[name]
    for graph in graphs:
        x, y = graph["x"], graph["y"]
        if x.ndim != 2 or not torch.isfinite(x).all() or y.shape[0] != x.shape[0]:
            raise ValueError("invalid node features or targets")
        if x.shape[1] != spec["features"]:
            raise ValueError("feature count differs from official dataset")
        arcs, incidence = canonical_edges(graph["edge_index"], x.shape[0])
        if not torch.equal(arcs, graph["edge_index"]):
            raise ValueError("common undirected adjacency is not canonical")
        if not torch.equal(incidence, graph["incidence_edge_index"]):
            raise ValueError("incidence and adjacency represent different graphs")
    if name == "ppi":
        flattened = [int(index) for values in splits.values() for index in values]
        if sorted(flattened) != list(range(len(graphs))):
            raise ValueError("PPI graphs must be disjoint and exhaustive across splits")
        if [len(splits[key]) for key in ("train", "validation", "test")] != spec["graphs"]:
            raise ValueError("PPI requires its official 20/2/2 graph split")
        if any(
            graph["y"].ndim != 2
            or graph["y"].shape[1] != payload["classes"]
            or not torch.all((graph["y"] == 0) | (graph["y"] == 1))
            for graph in graphs
        ):
            raise ValueError("PPI requires binary multi-label node targets")
    else:
        if len(graphs) != 1:
            raise ValueError("citation benchmark must contain exactly one graph")
        n = graphs[0]["x"].shape[0]
        masks = [splits[key] for key in ("train", "validation", "test")]
        if any(mask.dtype != torch.bool or mask.shape != (n,) or not mask.any() for mask in masks):
            raise ValueError("each node split must be a nonempty boolean mask")
        if torch.any(sum(mask.long() for mask in masks) > 1):
            raise ValueError("train, validation and test masks overlap")
        y = graphs[0]["y"]
        if (
            y.ndim != 1
            or y.dtype != torch.long
            or int(y.min()) < 0
            or int(y.max()) >= payload["classes"]
        ):
            raise ValueError("invalid node class labels")
        if n != spec["nodes"] or [int(m.sum()) for m in masks] != spec["splits"]:
            raise ValueError("node count/split sizes differ from the official protocol")
    if payload["classes"] != spec["classes"]:
        raise ValueError("class count differs from official dataset")


@contextmanager
def _pyg_safe_globals():
    """Allow only PyG data containers in old OGB processed caches on PyTorch >=2.6."""
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage

    with torch.serialization.safe_globals(
        [Data, DataEdgeAttr, DataTensorAttr, BaseStorage, EdgeStorage, GlobalStorage, NodeStorage]
    ):
        yield


def _download_official(name: str, root: Path) -> tuple[dict[str, Any], list[Path]]:
    """Called only after the user explicitly permits dataset downloads."""
    try:
        from torch_geometric.datasets import PPI, Planetoid
    except ImportError as exc:
        raise RuntimeError(
            "Install the project's Conda GPU environment (torch-geometric required)."
        ) from exc
    raw_dirs: list[Path] = []
    payload: dict[str, Any] = {"dataset": name, "classes": EXPECTED[name]["classes"]}
    with _pyg_safe_globals():
        if name in {"cora", "citeseer", "pubmed"}:
            pyg_name = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
            dataset = Planetoid(str(root / "sources"), pyg_name, split="public")
            data = dataset[0]
            graph = _graph(data, normalize_features=True)
            graph["y"] = graph["y"].reshape(-1).long()
            payload.update(
                graphs=[graph],
                splits={
                    "train": data.train_mask.cpu(),
                    "validation": data.val_mask.cpu(),
                    "test": data.test_mask.cpu(),
                },
            )
            raw_dirs.append(Path(dataset.raw_dir))
        elif name == "ppi":
            graphs: list[dict[str, Tensor]] = []
            splits: dict[str, list[int]] = {}
            for split, official in (("train", "train"), ("validation", "val"), ("test", "test")):
                dataset = PPI(str(root / "sources" / "PPI"), split=official)
                start = len(graphs)
                graphs.extend(_graph(data, normalize_features=False) for data in dataset)
                for graph in graphs[start:]:
                    graph["y"] = graph["y"].float()
                splits[split] = list(range(start, len(graphs)))
                raw_dirs.append(Path(dataset.raw_dir))
            payload.update(graphs=graphs, splits=splits)
        else:
            try:
                from ogb.nodeproppred import PygNodePropPredDataset
            except ImportError as exc:
                raise RuntimeError(
                    "ogbn-arxiv requires the project's optional 'ogb' dependency."
                ) from exc
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "sources"))
            graph = _graph(dataset[0], normalize_features=False)
            graph["y"] = graph["y"].reshape(-1).long()
            indices = dataset.get_idx_split()
            payload.update(
                graphs=[graph],
                splits={
                    key: _split_mask(indices[official], graph["x"].shape[0])
                    for key, official in (
                        ("train", "train"),
                        ("validation", "valid"),
                        ("test", "test"),
                    )
                },
            )
            raw_dirs.extend([Path(dataset.raw_dir), Path(dataset.root) / "split"])
    files = sorted(
        {path for directory in raw_dirs for path in directory.rglob("*") if path.is_file()}
    )
    if not files:
        raise RuntimeError("Official download has no raw source files to fingerprint")
    return payload, files


def load_dataset(
    name: str, data_root: Path, *, allow_download: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify cache or prepare official data; never instantiate a downloader offline."""
    if name not in DATASETS:
        raise ValueError(f"Unsupported matched dataset: {name}")
    root = data_root.expanduser().resolve() / "conductance_gat" / "matched_benchmark_v1"
    folder = root / name
    tensor_path, manifest_path = folder / "data.pt", folder / "manifest.json"
    if tensor_path.exists() or manifest_path.exists():
        if not tensor_path.is_file() or not manifest_path.is_file():
            raise CacheIncompleteError(
                f"Incomplete dataset cache: {folder}; "
                "restore the missing file or use a new data root"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise CacheCorruptError(f"Unreadable dataset manifest: {manifest_path}") from exc
        if manifest.get("schema_version") != CACHE_VERSION or manifest.get("dataset") != name:
            raise CacheWrongRequestError(f"Dataset cache protocol mismatch: {folder}")
        if sha256_file(tensor_path) != manifest.get("data_sha256"):
            raise CacheCorruptError(f"Dataset cache checksum mismatch: {tensor_path}")
        try:
            payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
            validate_payload(name, payload)
            actual_splits = {
                key: tensor_hash(value if isinstance(value, Tensor) else torch.tensor(value))
                for key, value in payload["splits"].items()
            }
            if actual_splits != manifest.get("split_sha256"):
                raise ValueError("official split fingerprint mismatch")
            if manifest.get("source_url") != SOURCES[name] or not manifest.get(
                "source_files_sha256"
            ):
                raise ValueError("official dataset provenance missing or incorrect")
        except Exception as exc:
            raise CacheCorruptError(f"Invalid dataset tensors/metadata: {folder}: {exc}") from exc
        manifest["preprocessing"]["self_loops"] = (
            "conductance residual identity; no incidence loops"
        )
        return payload, manifest
    if not allow_download:
        raise FileNotFoundError(
            f"{name} is not prepared. Run bash scripts/prepare_data.sh first. "
            "No synthetic substitute is allowed."
        )
    payload, files = _download_official(name, root)
    validate_payload(name, payload)
    atomic_publish(tensor_path, lambda path: torch.save(payload, path))
    split_hashes = {
        key: tensor_hash(value if isinstance(value, Tensor) else torch.tensor(value))
        for key, value in payload["splits"].items()
    }
    manifest = {
        "schema_version": CACHE_VERSION,
        "dataset": name,
        "source_url": SOURCES[name],
        "data_sha256": sha256_file(tensor_path),
        "split_sha256": split_hashes,
        "source_files_sha256": {str(path.relative_to(root)): sha256_file(path) for path in files},
        "split": "official_public_masks"
        if name in DATASETS[:3]
        else "official_inductive_graph_split"
        if name == "ppi"
        else "official_time_split",
        "task": "multi_label_node_classification" if name == "ppi" else "node_classification",
        "metric": "micro_f1" if name == "ppi" else "accuracy",
        "preprocessing": {
            "graph": "undirected, deduplicated arcs, self-loops removed before operators",
            "features": "PyG NormalizeFeatures equivalent"
            if name in DATASETS[:3]
            else "official unmodified features",
            "incidence": "same undirected graph, one low-to-high orientation per edge",
            "self_loops": "conductance residual identity; no incidence loops",
        },
        "graphs": [
            {
                "nodes": int(g["x"].shape[0]),
                "arcs": int(g["edge_index"].shape[1]),
                "undirected_edges": int(g["incidence_edge_index"].shape[1]),
            }
            for g in payload["graphs"]
        ],
        "split_counts": {
            key: len(value) if isinstance(value, list) else int(value.sum())
            for key, value in payload["splits"].items()
        },
    }
    atomic_write_json(manifest_path, manifest)
    return payload, manifest
