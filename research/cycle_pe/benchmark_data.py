"""Official datasets for our cycle PE; no fallback or random re-splitting.

Only adapters import PyG. Tensor preparation and invariance tests are independent
of optional download libraries. Only our cycle-set PE is precomputed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from chartgat.algebra import incidence_matrix
from chartgat.cache import atomic_publish, atomic_write_json
from chartgat.graphs import spanning_tree_indices
from research.cycle_pe.features import (
    SET_STAT_NAMES,
    cycle_set_statistics,
    static_fundamental_basis,
)

DATASETS = ("zinc12k", "peptides_struct")
CACHE_VERSION = "own-cycle-set-v2"
SPLITS = ("train", "validation", "test")
EXPECTED_SIZES = {
    "zinc12k": (10000, 1000, 1000),
    "peptides_struct": (10873, 2331, 2331),
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
    cycle_set: Tensor


@dataclass
class Batch(Graph):
    batch: Tensor
    ptr: Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            **{f.name: getattr(self, f.name).to(device, non_blocking=True) for f in fields(self)}
        )

    def pin_memory(self) -> Batch:
        return Batch(**{f.name: getattr(self, f.name).pin_memory() for f in fields(self)})


def collate(graphs: list[Graph]) -> Batch:
    counts = [len(g.x) for g in graphs]
    ptr = torch.tensor([0, *np.cumsum(counts).tolist()], dtype=torch.long)
    return Batch(
        x=torch.cat([g.x for g in graphs]),
        edge_index=torch.cat([g.edge_index + ptr[i] for i, g in enumerate(graphs)], dim=1),
        edge_attr=torch.cat([g.edge_attr for g in graphs]),
        y=torch.stack([g.y for g in graphs]),
        cycle_set=torch.cat([g.cycle_set for g in graphs]),
        batch=torch.repeat_interleave(torch.arange(len(graphs)), torch.tensor(counts)),
        ptr=ptr,
    )


def graph_fingerprint(data: Any, digest: Any) -> None:
    """Hash actual ordered topology, chemistry and labels, not just split sizes."""
    for key in ("x", "edge_index", "edge_attr", "y"):
        tensor = getattr(data, key).detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())


def cycle_statistics(num_nodes: int, edge_index: Tensor) -> Tensor:
    """Six sign/column-order invariant summaries of one fixed BFS cycle basis.

    Reuses the existing basis and set-statistics implementation, including
    disconnected graphs componentwise. No m-by-m projector is constructed.
    The chart is not invariant to recomputing BFS after arbitrary relabeling.
    """
    directed = edge_index.T.tolist()
    edges = sorted({tuple(sorted((u, v))) for u, v in directed})
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    edge_lookup = {edge: index for index, edge in enumerate(edges)}
    blocks = []
    components = sorted(nx.connected_components(graph), key=min)
    for component in components:
        nodes = sorted(component)
        local = {node: i for i, node in enumerate(nodes)}
        component_edges = [edge for edge in edges if edge[0] in component]
        local_edges = [(local[u], local[v]) for u, v in component_edges]
        incidence = incidence_matrix(len(nodes), local_edges)
        tree = spanning_tree_indices(len(nodes), local_edges, mode="bfs")
        block = static_fundamental_basis(incidence, tree)
        blocks.append((component_edges, block))
    rank = sum(block.shape[1] for _, block in blocks)
    basis = np.zeros((len(edges), rank), dtype=np.float64)
    offset = 0
    for component_edges, block in blocks:
        rows = [edge_lookup[edge] for edge in component_edges]
        basis[rows, offset : offset + block.shape[1]] = block
        offset += block.shape[1]
    values = cycle_set_statistics(basis)
    indices = [edge_lookup[tuple(sorted(edge))] for edge in directed]
    return torch.from_numpy(values[indices].reshape(len(directed), len(SET_STAT_NAMES))).float()


def prepare_graph(data: Any) -> Graph:
    """Preserve chemistry/targets and compute only the original cycle-set PE."""
    x = data.x.detach().cpu().long().reshape(int(data.num_nodes), -1)
    edge_index = data.edge_index.detach().cpu().long().contiguous()
    edge_attr = data.edge_attr.detach().cpu().long()
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(1)
    if edge_attr.ndim != 2 or len(edge_attr) != edge_index.shape[1]:
        raise ValueError("invalid official bond-feature shape")
    y = data.y.detach().cpu().float().reshape(-1)
    n = len(x)
    if n < 1 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("invalid official graph shape")
    if not torch.isfinite(y).all() or not torch.isfinite(data.x).all():
        raise ValueError("nonfinite official input/target")
    pairs = list(map(tuple, edge_index.T.tolist()))
    if len(set(pairs)) != len(pairs) or any(u == v for u, v in pairs):
        raise ValueError("molecular benchmark requires simple loop-free edges")
    attributes = {edge: edge_attr[i] for i, edge in enumerate(pairs)}
    for u, v in pairs:
        if (v, u) not in attributes or not torch.equal(attributes[u, v], attributes[v, u]):
            raise ValueError("molecular bonds must have agreeing directed copies")
    # The original cycle-PE message layer itself sends messages in both
    # directions, so retain exactly one copy per official undirected bond.
    keep = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, keep]
    edge_attr = edge_attr[keep]
    return Graph(
        x,
        edge_index,
        edge_attr,
        y,
        cycle_statistics(n, edge_index),
    )


def _ready(root: Path, dataset: str) -> bool:
    if dataset == "zinc12k":
        raw = root / "raw"
        raw_names = [
            f"{split}.{suffix}"
            for split in ("train", "val", "test")
            for suffix in ("pickle", "index")
        ]
    else:
        raw = root / "peptides-struct" / "raw"
        raw_names = [f"{split}.pt" for split in ("train", "val", "test")]
    # PyG checks raw artifacts BEFORE processed files in Dataset.__init__.
    # Processed-only caches must not trigger an implicit network download.
    return all((raw / name).is_file() for name in raw_names)


def _requested_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(split not in SPLITS for split in splits)
    ):
        raise ValueError("splits must be a nonempty unique subset of official splits")
    return splits


def load_official_splits(
    data_root: Path,
    dataset: str,
    *,
    allow_download: bool,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    splits = _requested_splits(splits)
    if dataset not in DATASETS:
        raise ValueError(f"unknown cycle PE dataset: {dataset}")
    root = data_root / ("ZINC12K" if dataset == "zinc12k" else "LRGB")
    if not allow_download and not _ready(root, dataset):
        raise FileNotFoundError(f"{dataset}: official data absent at {root}; run prepare_data.sh")
    try:
        from torch_geometric.datasets import ZINC, LRGBDataset
    except ImportError as exc:
        raise RuntimeError(
            "Cycle PE benchmarks require the project's PyG paper dependencies"
        ) from exc
    official_names = {"train": "train", "validation": "val", "test": "test"}
    expected_sizes = dict(zip(SPLITS, EXPECTED_SIZES[dataset], strict=True))
    datasets = {}
    for split in splits:
        official = official_names[split]
        datasets[split] = (
            ZINC(str(root), subset=True, split=official)
            if dataset == "zinc12k"
            else LRGBDataset(str(root), name="Peptides-struct", split=official)
        )
        actual = len(datasets[split])
        if actual != expected_sizes[split]:
            raise RuntimeError(
                f"{dataset}/{split} official split mismatch: {actual} != {expected_sizes[split]}"
            )
    return datasets


def load_benchmark(
    data_root: Path,
    dataset: str,
    *,
    allow_download: bool,
    splits: tuple[str, ...] = SPLITS,
) -> tuple[dict[str, list[Graph]], dict[str, Any]]:
    splits = _requested_splits(splits)
    official = load_official_splits(
        data_root,
        dataset,
        allow_download=allow_download,
        splits=splits,
    )
    target_width = 1 if dataset == "zinc12k" else 11
    signature = {
        "version": CACHE_VERSION,
        "dataset": dataset,
        "representation": "existing_bfs_cycle_set",
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = data_root / "cycle_pe_benchmark" / dataset / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Graph]] = {}
    split_hashes = {}
    for split in splits:
        digest = hashlib.sha256()
        for data in official[split]:
            graph_fingerprint(data, digest)
        split_hashes[split] = digest.hexdigest()
        cache = cache_dir / f"{split}.pt"
        meta = cache_dir / f"{split}.json"
        if cache.exists() and meta.exists():
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            if (
                metadata.get("source_sha256") != split_hashes[split]
                or metadata.get("signature") != signature
                or metadata.get("cache_sha256") != hashlib.sha256(cache.read_bytes()).hexdigest()
            ):
                raise RuntimeError(f"Mismatched/corrupt PE cache: {cache}; no silent rebuild")
            rows = torch.load(cache, map_location="cpu", weights_only=True)
            graphs = [Graph(**row) for row in rows]
        elif cache.exists() or meta.exists():
            raise RuntimeError(
                f"Incomplete PE cache at {cache_dir}; remove only this cache and prepare again"
            )
        else:
            graphs = []
            for index, data in enumerate(official[split]):
                graph = prepare_graph(data)
                if graph.y.numel() != target_width:
                    raise ValueError(f"{dataset}: unexpected target width")
                graphs.append(graph)
                if (index + 1) % 1000 == 0:
                    print(
                        f"{dataset}/{split}: topology PE {index + 1}/{len(official[split])}",
                        flush=True,
                    )
            rows = [
                {field.name: getattr(graph, field.name) for field in fields(graph)}
                for graph in graphs
            ]
            atomic_publish(cache, lambda path, payload=rows: torch.save(payload, path))
            atomic_write_json(
                meta,
                {
                    "signature": signature,
                    "source_sha256": split_hashes[split],
                    "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                },
            )
        if len(graphs) != len(official[split]):
            raise RuntimeError(f"{dataset}: cached graph count mismatch")
        result[split] = graphs
    protocol = {
        "comparison": "ours_only_on_official_benchmark_splits",
        "source_url": SOURCES[dataset],
        "official_splits": True,
        "loaded_splits": list(splits),
        "split_sizes": {s: len(result[s]) for s in splits},
        "split_content_sha256": split_hashes,
        "target_width": target_width,
        "target_scaling": "official supplied labels, unchanged; no fitted target scaling",
        "input_features": "ZINC categorical atoms/bonds"
        if dataset == "zinc12k"
        else "OGB 9 atom / 3 bond categorical fields",
        "preparation": signature,
        "cache_directory": str(cache_dir),
    }
    return result, protocol
