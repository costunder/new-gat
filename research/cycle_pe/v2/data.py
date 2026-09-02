"""Official molecular splits with complete, graph-local cycle basis matrices.

Only the official split adapter and source fingerprint are shared with v1.
Neither v1 graph preparation nor cycle-set statistics are used here. Cached
columns and ragged batches retain all cycle vectors without global column IDs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_publish,
    atomic_write_json,
)
from research.cycle_pe.benchmark_data import graph_fingerprint, load_official_splits
from research.cycle_pe.v2.basis import left_nullspace_basis, validate_cycle_basis

DATASETS = ("zinc12k", "peptides_struct")
SPLITS = ("train", "validation", "test")
CACHE_VERSION = "complete-left-nullspace-svd-v2-1"
CACHE_NAMESPACE = "cycle_pe_v2_benchmark"
SCHEMAS = {
    "zinc12k": {"atoms": (28,), "bonds": (4,), "targets": 1},
    "peptides_struct": {
        "atoms": (119, 4, 12, 12, 10, 6, 6, 2, 2),
        "bonds": (5, 6, 2),
        "targets": 11,
    },
}
SOURCES = {
    "zinc12k": "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html",
    "peptides_struct": "https://github.com/vijaydwivedi75/lrgb",
}


@dataclass
class Graph:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Tensor
    cycle_basis: Tensor


@dataclass
class Batch:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Tensor
    batch: Tensor
    ptr: Tensor
    cycle_bases: tuple[Tensor, ...]
    edge_ptr: Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            **{
                field.name: tuple(value.to(device, non_blocking=True) for value in current)
                if field.name == "cycle_bases"
                else current.to(device, non_blocking=True)
                for field in fields(self)
                for current in (getattr(self, field.name),)
            }
        )

    def pin_memory(self) -> Batch:
        return Batch(
            **{
                field.name: tuple(value.pin_memory() for value in current)
                if field.name == "cycle_bases"
                else current.pin_memory()
                for field in fields(self)
                for current in (getattr(self, field.name),)
            }
        )


def collate(graphs: list[Graph]) -> Batch:
    """Concatenate graph tensors, keeping every basis in its own coordinate chart."""
    if not graphs:
        raise ValueError("cannot collate an empty graph list")
    for graph in graphs:
        validate_graph(graph, check_basis=False)
    widths = {(g.x.shape[1], g.edge_attr.shape[1], g.y.numel()) for g in graphs}
    if len(widths) != 1:
        raise ValueError("cannot collate graphs with different molecular schemas")
    counts = [len(graph.x) for graph in graphs]
    ptr = torch.tensor([0, *np.cumsum(counts).tolist()], dtype=torch.long)
    edge_ptr = torch.tensor(
        [0, *np.cumsum([graph.edge_index.shape[1] for graph in graphs]).tolist()],
        dtype=torch.long,
    )
    return Batch(
        x=torch.cat([graph.x for graph in graphs]),
        edge_index=torch.cat(
            [graph.edge_index + ptr[index] for index, graph in enumerate(graphs)], dim=1
        ),
        edge_attr=torch.cat([graph.edge_attr for graph in graphs]),
        y=torch.stack([graph.y for graph in graphs]),
        batch=torch.repeat_interleave(torch.arange(len(graphs)), torch.tensor(counts)),
        ptr=ptr,
        cycle_bases=tuple(graph.cycle_basis for graph in graphs),
        edge_ptr=edge_ptr,
    )


def _integer_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f"{name} must be an integer tensor")
    value = value.detach().cpu()
    if not torch.isfinite(value).all() or (
        value.is_floating_point() and not torch.equal(value, value.round())
    ):
        raise ValueError(f"{name} must contain finite integer values")
    return value.long().contiguous()


def _canonical_inputs(data: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raw_nodes = data.num_nodes
    if isinstance(raw_nodes, bool) or not isinstance(raw_nodes, (int, np.integer)):
        raise ValueError("num_nodes must be a positive integer")
    num_nodes = int(raw_nodes)
    if num_nodes < 1:
        raise ValueError("official graph must contain at least one node")
    x = _integer_tensor(data.x, "atom features")
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2 or x.shape[0] != num_nodes or x.shape[1] < 1:
        raise ValueError("invalid official atom-feature shape")
    edge_index = _integer_tensor(data.edge_index, "edge_index")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if (edge_index < 0).any() or (edge_index >= num_nodes).any():
        raise ValueError("edge endpoint out of range")
    edge_attr = _integer_tensor(data.edge_attr, "bond features")
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(1)
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("invalid official bond-feature shape")
    if edge_attr.shape[1] < 1:
        raise ValueError("official bonds require categorical features")
    if not isinstance(data.y, Tensor) or data.y.is_complex():
        raise ValueError("official target must be a real tensor")
    y = data.y.detach().cpu().float().reshape(-1)
    if not len(y) or not torch.isfinite(y).all():
        raise ValueError("official targets must be finite and nonempty")
    pairs = list(map(tuple, edge_index.T.tolist()))
    if len(set(pairs)) != len(pairs) or any(u == v for u, v in pairs):
        raise ValueError("molecular benchmark requires simple loop-free edges")
    attributes = {edge: edge_attr[index] for index, edge in enumerate(pairs)}
    for u, v in pairs:
        if (v, u) not in attributes or not torch.equal(attributes[u, v], attributes[v, u]):
            raise ValueError("molecular bonds must have agreeing directed copies")
    canonical = sorted((u, v) for u, v in pairs if u < v)
    canonical_index = torch.tensor(canonical, dtype=torch.long).reshape(-1, 2).T.contiguous()
    canonical_attr = (
        torch.stack([attributes[pair] for pair in canonical])
        if canonical
        else edge_attr.new_empty((0, edge_attr.shape[1]))
    )
    return x, canonical_index, canonical_attr, y


def validate_graph(graph: Graph, *, dataset: str | None = None, check_basis: bool = True) -> None:
    """Validate prepared/cache schema and, on preparation/load, basis identities."""
    for field in fields(Graph):
        value = getattr(graph, field.name)
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise ValueError("prepared graph fields must be CPU tensors")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite prepared graph field: {field.name}")
    if graph.x.dtype != torch.long or graph.x.ndim != 2 or min(graph.x.shape) < 1:
        raise ValueError("invalid prepared atom-feature schema")
    if (
        graph.edge_index.dtype != torch.long
        or graph.edge_index.ndim != 2
        or graph.edge_index.shape[0] != 2
    ):
        raise ValueError("invalid prepared edge_index schema")
    edge_count = graph.edge_index.shape[1]
    if (
        graph.edge_attr.dtype != torch.long
        or graph.edge_attr.ndim != 2
        or graph.edge_attr.shape[0] != edge_count
        or graph.edge_attr.shape[1] < 1
    ):
        raise ValueError("invalid prepared bond-feature schema")
    if graph.y.dtype != torch.float32 or graph.y.ndim != 1 or not graph.y.numel():
        raise ValueError("invalid prepared target schema")
    if (
        graph.cycle_basis.dtype != torch.float32
        or graph.cycle_basis.ndim != 2
        or graph.cycle_basis.shape[0] != edge_count
    ):
        raise ValueError("invalid prepared cycle-basis schema")
    if (graph.x < 0).any() or (graph.edge_attr < 0).any():
        raise ValueError("categorical features must be nonnegative")
    if dataset is not None:
        if dataset not in DATASETS:
            raise ValueError(f"unknown cycle PE v2 dataset: {dataset}")
        schema = SCHEMAS[dataset]
        if graph.y.numel() != schema["targets"]:
            raise ValueError(f"{dataset}: unexpected target width")
        for values, name in ((graph.x, "atoms"), (graph.edge_attr, "bonds")):
            cardinalities = schema[name]
            if values.shape[1] != len(cardinalities):
                raise ValueError(f"{dataset}: unexpected {name} field count")
            if any((values[:, i] >= size).any() for i, size in enumerate(cardinalities)):
                raise ValueError(f"{dataset}: categorical {name} index out of range")
    if check_basis:
        validate_cycle_basis(len(graph.x), graph.edge_index.numpy(), graph.cycle_basis.numpy())


def prepare_graph(data: Any, *, dataset: str | None = None) -> Graph:
    """Preserve official chemistry/targets and attach the full raw SVD basis."""
    x, edge_index, edge_attr, y = _canonical_inputs(data)
    graph = Graph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        cycle_basis=torch.from_numpy(left_nullspace_basis(len(x), edge_index.numpy())),
    )
    validate_graph(graph, dataset=dataset)
    return graph


def preparation_signature(dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown cycle PE v2 dataset: {dataset}")
    directory = Path(__file__).resolve().parent
    return {
        "version": CACHE_VERSION,
        "dataset": dataset,
        "representation": "complete_orthonormal_left_nullspace_basis",
        "incidence": "B[m,n], canonical sorted u<v edges, tail -1 and head +1",
        "basis": "numpy.linalg.svd(B, full_matrices=True); all m-n+c left-null columns",
        "storage": "float32 matrices [num_edges, cycle_rank], graph-local ragged columns",
        "numpy_version": np.__version__,
        "implementation_sha256": {
            "v2/basis.py": hashlib.sha256((directory / "basis.py").read_bytes()).hexdigest(),
            "v2/data.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "official_adapter": hashlib.sha256(
                (directory.parent / "benchmark_data.py").read_bytes()
            ).hexdigest(),
        },
    }


def _validate_cached_graphs(rows: Any, official: Any, dataset: str) -> list[Graph]:
    if not isinstance(rows, list) or len(rows) != len(official):
        raise ValueError("cached graph count/schema mismatch")
    names = {field.name for field in fields(Graph)}
    graphs = []
    for row, source in zip(rows, official, strict=True):
        if not isinstance(row, dict) or set(row) != names:
            raise ValueError("cached graph field schema mismatch")
        graph = Graph(**row)
        validate_graph(graph, dataset=dataset)
        expected = _canonical_inputs(source)
        for name, source_value in zip(("x", "edge_index", "edge_attr", "y"), expected, strict=True):
            if not torch.equal(getattr(graph, name), source_value):
                raise ValueError(f"cached {name} disagrees with official graph content/order")
        graphs.append(graph)
    return graphs


def load_benchmark(
    data_root: Path,
    dataset: str,
    *,
    allow_download: bool,
    splits: tuple[str, ...] = SPLITS,
) -> tuple[dict[str, list[Graph]], dict[str, Any]]:
    """Load fixed official splits, validating immutable basis caches fail-closed."""
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(split not in SPLITS for split in splits)
    ):
        raise ValueError("splits must be a nonempty unique subset of official splits")
    signature = preparation_signature(dataset)
    official = load_official_splits(
        data_root,
        dataset,
        allow_download=allow_download,
        splits=splits,
    )
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = data_root / CACHE_NAMESPACE / dataset / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Graph]] = {}
    split_hashes = {}
    for split in splits:
        digest = hashlib.sha256()
        for data in official[split]:
            graph_fingerprint(data, digest)
        source_hash = split_hashes[split] = digest.hexdigest()
        cache, meta = cache_dir / f"{split}.pt", cache_dir / f"{split}.json"
        if cache.exists() != meta.exists():
            raise CacheIncompleteError(
                f"Incomplete cycle PE v2 cache at {cache}; no silent rebuild"
            )
        if cache.exists():
            try:
                metadata = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CacheCorruptError(f"Unreadable cycle PE v2 metadata: {meta}") from exc
            if not isinstance(metadata, dict):
                raise CacheCorruptError(f"Invalid cycle PE v2 metadata schema: {meta}")
            if (
                metadata.get("signature") != signature
                or metadata.get("source_sha256") != source_hash
                or metadata.get("split") != split
            ):
                raise CacheWrongRequestError(f"Mismatched cycle PE v2 cache: {cache}; no rebuild")
            if metadata.get("cache_sha256") != hashlib.sha256(cache.read_bytes()).hexdigest():
                raise CacheCorruptError(f"Corrupt cycle PE v2 cache payload: {cache}; no rebuild")
            try:
                rows = torch.load(cache, map_location="cpu", weights_only=True)
                graphs = _validate_cached_graphs(rows, official[split], dataset)
            except Exception as exc:
                raise CacheCorruptError(
                    f"Invalid cycle PE v2 cache content: {cache}: {exc}"
                ) from exc
        else:
            graphs = []
            for index, data in enumerate(official[split]):
                graphs.append(prepare_graph(data, dataset=dataset))
                if (index + 1) % 1000 == 0:
                    print(
                        f"{dataset}/{split}: full cycle bases {index + 1}/{len(official[split])}",
                        flush=True,
                    )
            rows = [
                {field.name: getattr(graph, field.name) for field in fields(Graph)}
                for graph in graphs
            ]
            atomic_publish(cache, lambda path, payload=rows: torch.save(payload, path))
            atomic_write_json(
                meta,
                {
                    "signature": signature,
                    "split": split,
                    "source_sha256": source_hash,
                    "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                },
            )
        result[split] = graphs
    protocol = {
        "comparison": "ours_only_on_official_benchmark_splits",
        "source_url": SOURCES[dataset],
        "official_splits": True,
        "loaded_splits": list(splits),
        "split_sizes": {split: len(result[split]) for split in splits},
        "split_content_sha256": split_hashes,
        "target_width": SCHEMAS[dataset]["targets"],
        "target_scaling": "official supplied labels, unchanged; no fitted target scaling",
        "input_features": "ZINC categorical atoms/bonds"
        if dataset == "zinc12k"
        else "OGB 9 atom / 3 bond categorical fields",
        "preparation": signature,
        "cache_directory": str(cache_dir),
        "basis_storage": "all beta=m-n+c columns per graph; no padding or truncation",
        "basis_coordinates": "SVD coordinates are not invariant to arbitrary nullspace rotations",
    }
    return result, protocol


__all__ = [
    "Batch",
    "CACHE_NAMESPACE",
    "CACHE_VERSION",
    "DATASETS",
    "Graph",
    "collate",
    "load_benchmark",
    "preparation_signature",
    "prepare_graph",
    "validate_graph",
]
