"""Datasets and spanning-tree charts for the independent paper protocol.

This module supplies graph-level downstream labels, a true uniform
spanning-tree sampler, deterministic caches, and optional PyG dataset adapters.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix, validate_spanning_tree
from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_write_bytes,
)
from chartgat.graphs import make_connected_graph, spanning_tree_indices

from .augmentation import TreeChart

IntArray = NDArray[np.int64]
DATASET_VERSION = 2
TARGET_CYCLE_LENGTHS = (3, 4, 5, 6)
ZINC_NUM_ATOM_TYPES = 28
ZINC_NUM_BOND_TYPES = 4


class OptionalDatasetError(RuntimeError):
    """A requested optional dataset cannot be imported or downloaded."""


@dataclass(frozen=True)
class GraphRecord:
    """One physical graph and a downstream label that is independent of its chart."""

    graph_id: str
    family: str
    split: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    target: tuple[float, ...]
    task_type: str = "regression"
    x: tuple[int, ...] | None = None
    edge_attr: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.x is not None:
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in self.x
            ):
                raise ValueError("categorical node x values must be non-negative integers")
            normalized_x = tuple(int(value) for value in self.x)
            if len(normalized_x) != self.num_nodes:
                raise ValueError("categorical node x must have one value per node")
            if any(value < 0 for value in normalized_x):
                raise ValueError("categorical node x values must be non-negative integers")
            object.__setattr__(self, "x", normalized_x)
        if self.edge_attr is not None:
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in self.edge_attr
            ):
                raise ValueError("categorical edge_attr values must be non-negative integers")
            normalized_edge_attr = tuple(int(value) for value in self.edge_attr)
            if len(normalized_edge_attr) != len(self.edges):
                raise ValueError(
                    "categorical edge_attr must align one-to-one with undirected edges"
                )
            if any(value < 0 for value in normalized_edge_attr):
                raise ValueError("categorical edge_attr values must be non-negative integers")
            object.__setattr__(self, "edge_attr", normalized_edge_attr)

    @property
    def beta(self) -> int:
        return len(self.edges) - self.num_nodes + 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "family": self.family,
            "split": self.split,
            "num_nodes": self.num_nodes,
            "edges": [list(edge) for edge in self.edges],
            "target": list(self.target),
            "task_type": self.task_type,
            "x": None if self.x is None else list(self.x),
            "edge_attr": None if self.edge_attr is None else list(self.edge_attr),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> GraphRecord:
        num_nodes = int(payload["num_nodes"])
        raw_edges = tuple((int(edge[0]), int(edge[1])) for edge in payload["edges"])
        edges = _canonical_edges(num_nodes, raw_edges)
        if len(edges) != len(raw_edges):
            raise ValueError(
                "cached undirected edges contain a self-loop, duplicate, or parallel edge"
            )
        raw_edge_attr = payload.get("edge_attr")
        if raw_edge_attr is None:
            edge_attr = None
        else:
            values = tuple(raw_edge_attr)
            if len(values) != len(raw_edges):
                raise ValueError(
                    "cached categorical edge_attr does not align with undirected edges"
                )
            by_edge = {
                (min(u, v), max(u, v)): value
                for (u, v), value in zip(raw_edges, values, strict=True)
            }
            edge_attr = tuple(by_edge[edge] for edge in edges)
        return cls(
            graph_id=str(payload["graph_id"]),
            family=str(payload["family"]),
            split=str(payload["split"]),
            num_nodes=num_nodes,
            edges=edges,
            target=tuple(float(value) for value in payload["target"]),
            task_type=str(payload.get("task_type", "regression")),
            x=(None if payload.get("x") is None else tuple(payload["x"])),
            edge_attr=edge_attr,
        )


@dataclass(frozen=True)
class PreparedDataset:
    """Validated records plus cache provenance."""

    suite: str
    records: tuple[GraphRecord, ...]
    data_path: Path
    manifest_path: Path
    data_sha256: str
    target_names: tuple[str, ...]
    task_type: str


def _canonical_edges(
    num_nodes: int, edges: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least two")
    canonical: set[tuple[int, int]] = set()
    for raw_u, raw_v in edges:
        u, v = int(raw_u), int(raw_v)
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError("edge endpoint lies outside [0, num_nodes)")
        if u == v:
            continue
        canonical.add((min(u, v), max(u, v)))
    result = tuple(sorted(canonical))
    if not result:
        raise ValueError("graph must contain at least one edge")
    _adjacency(num_nodes, result, require_connected=True)
    return result


def _adjacency(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    require_connected: bool,
) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    seen_edges: set[tuple[int, int]] = set()
    for edge_index, (raw_u, raw_v) in enumerate(edges):
        u, v = int(raw_u), int(raw_v)
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError("edge endpoint lies outside [0, num_nodes)")
        if u == v:
            raise ValueError("self-loops are not supported by the chart protocol")
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            raise ValueError("parallel or duplicate undirected edges are not supported")
        seen_edges.add(key)
        adjacency[u].append((v, edge_index))
        adjacency[v].append((u, edge_index))
    for neighbors in adjacency:
        neighbors.sort()
    if require_connected:
        reached = {0}
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adjacency[node]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if len(reached) != num_nodes:
            raise ValueError("graph is disconnected")
    return adjacency


def traversal_tree_indices(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    method: str,
    root: int,
) -> IntArray:
    """Return a deterministic BFS/DFS tree from an explicit root."""

    if method not in {"bfs", "dfs"}:
        raise ValueError("method must be bfs or dfs")
    if not 0 <= root < num_nodes:
        raise ValueError("root lies outside [0, num_nodes)")
    adjacency = _adjacency(num_nodes, edges, require_connected=True)
    selected: list[int] = []
    seen = {root}
    frontier: deque[int] | list[int]
    if method == "bfs":
        frontier = deque([root])
        while frontier:
            node = frontier.popleft()
            for neighbor, edge_index in adjacency[node]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                selected.append(edge_index)
                frontier.append(neighbor)
    else:
        stack = [root]
        next_neighbor = [0]
        while stack:
            node = stack[-1]
            position = next_neighbor[-1]
            if position == len(adjacency[node]):
                stack.pop()
                next_neighbor.pop()
                continue
            next_neighbor[-1] += 1
            neighbor, edge_index = adjacency[node][position]
            if neighbor in seen:
                continue
            seen.add(neighbor)
            selected.append(edge_index)
            stack.append(neighbor)
            next_neighbor.append(0)
    tree = np.asarray(sorted(selected), dtype=np.int64)
    validate_spanning_tree(incidence_matrix(num_nodes, edges), tree)
    return tree


def wilson_ust_indices(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    seed: int,
    root: int | None = None,
) -> IntArray:
    """Sample an unweighted uniform spanning tree with Wilson's algorithm.

    Loop-erased random walks generate every spanning tree with equal
    probability on a finite connected unweighted graph.  ``root`` changes the
    construction order but not the UST distribution.
    """

    adjacency = _adjacency(num_nodes, edges, require_connected=True)
    rng = np.random.default_rng(seed)
    resolved_root = int(rng.integers(num_nodes)) if root is None else int(root)
    if not 0 <= resolved_root < num_nodes:
        raise ValueError("root lies outside [0, num_nodes)")

    in_tree = np.zeros(num_nodes, dtype=np.bool_)
    in_tree[resolved_root] = True
    selected: list[int] = []
    starts = [int(node) for node in rng.permutation(num_nodes) if node != resolved_root]
    for start in starts:
        if in_tree[start]:
            continue
        path_nodes = [start]
        path_edges: list[int] = []
        positions = {start: 0}
        while not in_tree[path_nodes[-1]]:
            node = path_nodes[-1]
            choices = adjacency[node]
            neighbor, edge_index = choices[int(rng.integers(len(choices)))]
            if neighbor in positions:
                keep = positions[neighbor]
                for removed in path_nodes[keep + 1 :]:
                    positions.pop(removed)
                path_nodes = path_nodes[: keep + 1]
                path_edges = path_edges[:keep]
                continue
            path_edges.append(edge_index)
            path_nodes.append(neighbor)
            positions[neighbor] = len(path_nodes) - 1
        selected.extend(path_edges)
        in_tree[np.asarray(path_nodes, dtype=np.int64)] = True

    tree = np.asarray(sorted(selected), dtype=np.int64)
    validate_spanning_tree(incidence_matrix(num_nodes, edges), tree)
    return tree


def build_paper_chart(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    method: str,
    seed: int,
    root: int | None = None,
    name: str | None = None,
) -> TreeChart:
    """Build a full-beta chart from an explicit paper-protocol sampler."""

    normalized = method.strip().lower()
    rng = np.random.default_rng(seed ^ 0x5EED5EED)
    resolved_root = int(rng.integers(num_nodes)) if root is None else int(root)
    if normalized in {"bfs", "dfs"}:
        tree = traversal_tree_indices(num_nodes, edges, method=normalized, root=resolved_root)
    elif normalized in {"wilson", "wilson_ust", "ust"}:
        tree = wilson_ust_indices(num_nodes, edges, seed=seed, root=resolved_root)
        normalized = "wilson_ust"
    elif normalized in {"random", "random_priority", "random_priority_kruskal"}:
        tree = spanning_tree_indices(num_nodes, edges, mode="random", seed=seed)
        normalized = "random_priority_kruskal"
    else:
        raise ValueError(
            "unknown chart method; use bfs, dfs, wilson_ust, or random_priority_kruskal"
        )
    incidence = incidence_matrix(num_nodes, edges)
    basis, chords = fundamental_cycle_basis(incidence, tree, return_chords=True)
    chart_name = name or f"{normalized}:root={resolved_root}:seed={seed}"
    return TreeChart(chart_name, tree, chords, basis)


def chart_key(chart: TreeChart) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in chart.tree_edge_indices))


def sample_paper_charts(
    record: GraphRecord,
    *,
    count: int,
    methods: Sequence[str],
    seed: int,
    roots: Sequence[int] | None = None,
    exclude: Iterable[tuple[int, ...]] = (),
    require_distinct: bool = False,
) -> list[TreeChart]:
    """Sample deterministic mixed-method charts with random-root coverage."""

    if count < 1:
        raise ValueError("count must be positive")
    if not methods:
        raise ValueError("at least one chart method is required")
    if record.beta == 0:
        only_chart = build_paper_chart(
            record.num_nodes, record.edges, method="bfs", seed=seed, root=0
        )
        return [only_chart] * count
    forbidden = set(exclude)
    seen = set(forbidden)
    charts: list[TreeChart] = []
    max_attempts = max(128, count * 64)
    for attempt in range(max_attempts):
        method = methods[attempt % len(methods)]
        chart_seed = seed + attempt * 104_729
        if roots:
            root_index = (attempt // len(methods)) % len(roots)
            root = int(roots[root_index]) % record.num_nodes
        else:
            root = int(np.random.default_rng(chart_seed).integers(record.num_nodes))
        chart = build_paper_chart(
            record.num_nodes,
            record.edges,
            method=method,
            seed=chart_seed,
            root=root,
        )
        key = chart_key(chart)
        if key in seen:
            continue
        charts.append(chart)
        seen.add(key)
        if len(charts) == count:
            return charts
    if require_distinct and len(charts) < count:
        raise RuntimeError(
            f"only {len(charts)} distinct charts were available for {record.graph_id}; "
            f"requested {count}"
        )
    if not charts:
        fallback = build_paper_chart(
            record.num_nodes, record.edges, method="bfs", seed=seed, root=0
        )
        charts.append(fallback)
    while len(charts) < count:
        charts.append(charts[len(charts) % len(charts)])
    return charts


def simple_cycle_counts(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    lengths: Sequence[int] = TARGET_CYCLE_LENGTHS,
) -> tuple[int, ...]:
    """Count undirected simple cycles exactly for the requested small lengths."""

    requested = tuple(int(length) for length in lengths)
    if not requested or min(requested) < 3:
        raise ValueError("cycle lengths must all be at least three")
    adjacency = _adjacency(num_nodes, edges, require_connected=True)
    maximum = max(requested)
    cycles: set[tuple[int, ...]] = set()

    def canonical_cycle(path: Sequence[int]) -> tuple[int, ...]:
        values = tuple(path)
        rotations = []
        for orientation in (values, tuple(reversed(values))):
            rotations.extend(
                orientation[offset:] + orientation[:offset] for offset in range(len(orientation))
            )
        return min(rotations)

    for start in range(num_nodes):
        stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [
            (start, (start,), frozenset({start}))
        ]
        while stack:
            node, path, visited = stack.pop()
            for neighbor, _ in adjacency[node]:
                if neighbor == start and len(path) >= 3:
                    cycles.add(canonical_cycle(path))
                    continue
                if neighbor < start or neighbor in visited or len(path) >= maximum:
                    continue
                stack.append((neighbor, (*path, neighbor), visited | {neighbor}))
    counts = {length: 0 for length in requested}
    for cycle in cycles:
        if len(cycle) in counts:
            counts[len(cycle)] += 1
    return tuple(counts[length] for length in requested)


def _cycle_chain_graph(cycle_sizes: Sequence[int]) -> tuple[int, tuple[tuple[int, int], ...]]:
    edges: list[tuple[int, int]] = []
    anchors: list[int] = []
    offset = 0
    for raw_size in cycle_sizes:
        size = int(raw_size)
        if size < 3:
            raise ValueError("cycle size must be at least three")
        nodes = list(range(offset, offset + size))
        anchors.append(nodes[0])
        edges.extend((nodes[index], nodes[(index + 1) % size]) for index in range(size))
        offset += size
    edges.extend((anchors[index], anchors[index + 1]) for index in range(len(anchors) - 1))
    return offset, _canonical_edges(offset, edges)


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _register_unique_graph(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    buckets: dict[tuple[int, int], list[nx.Graph]],
) -> bool:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    bucket = buckets.setdefault((num_nodes, len(edges)), [])
    if any(nx.is_isomorphic(graph, previous) for previous in bucket):
        return False
    bucket.append(graph)
    return True


def build_cyclecount_records(*, seed: int) -> tuple[GraphRecord, ...]:
    """Create graph-first ID/OOD splits with chart-independent cycle-count labels."""

    counts = {"train": 128, "validation": 24, "id_test": 40, "ood_test": 40}
    records: list[GraphRecord] = []
    graph_buckets: dict[tuple[int, int], list[nx.Graph]] = {}
    for split in ("train", "validation", "id_test"):
        for index in range(counts[split]):
            for attempt in range(1_000):
                graph_seed = _stable_seed(f"{split}:{index}:{attempt}", seed)
                rng = np.random.default_rng(graph_seed)
                num_nodes = int(rng.integers(8, 13))
                extra_edges = int(rng.integers(2, min(6, num_nodes - 2)))
                edges = _canonical_edges(
                    num_nodes,
                    make_connected_graph(num_nodes, extra_edges, seed=graph_seed),
                )
                if _register_unique_graph(num_nodes, edges, graph_buckets):
                    break
            else:
                raise RuntimeError("failed to generate a unique ID graph split")
            target = simple_cycle_counts(num_nodes, edges)
            records.append(
                GraphRecord(
                    graph_id=f"id-{split}-{index:05d}",
                    family="recursive_tree_plus_chords",
                    split=split,
                    num_nodes=num_nodes,
                    edges=edges,
                    target=tuple(float(value) for value in target),
                )
            )
    for index in range(counts["ood_test"]):
        for attempt in range(1_000):
            graph_seed = _stable_seed(f"ood_test:{index}:{attempt}", seed)
            rng = np.random.default_rng(graph_seed)
            cycle_count = int(rng.integers(2, 5))
            cycle_sizes = tuple(int(value) for value in rng.integers(3, 7, size=cycle_count))
            num_nodes, edges = _cycle_chain_graph(cycle_sizes)
            if _register_unique_graph(num_nodes, edges, graph_buckets):
                break
        else:
            raise RuntimeError("failed to generate a unique OOD graph split")
        target = simple_cycle_counts(num_nodes, edges)
        records.append(
            GraphRecord(
                graph_id=f"ood-cycle-chain-{index:05d}",
                family="cactus_cycle_chain_family_ood",
                split="ood_test",
                num_nodes=num_nodes,
                edges=edges,
                target=tuple(float(value) for value in target),
            )
        )
    return tuple(records)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    def validate_json(temporary: Path) -> None:
        json.loads(temporary.read_text(encoding="utf-8"))

    atomic_write_bytes(path, content, validator=validate_json)


def _load_cached_dataset(
    *,
    suite: str,
    data_path: Path,
    manifest_path: Path,
) -> PreparedDataset:
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "dataset cache and manifest must either both exist or both be absent"
        )
    data_content = data_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256_bytes(data_content)
    if manifest.get("data_sha256") != digest:
        raise ValueError(f"dataset cache checksum mismatch: {data_path}")
    payload = json.loads(data_content)
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"unsupported dataset manifest version: {manifest_path}")
    if payload.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"unsupported dataset cache version: {data_path}")
    if manifest.get("suite") != suite or payload.get("suite") != suite:
        raise ValueError(f"dataset cache suite mismatch for {suite!r}: {data_path}")
    records = tuple(GraphRecord.from_payload(record) for record in payload["records"])
    if int(manifest.get("num_graphs", -1)) != len(records):
        raise ValueError(f"dataset cache graph count mismatch: {data_path}")
    graph_ids = [record.graph_id for record in records]
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError(f"dataset cache contains duplicate graph IDs: {data_path}")
    expected_split_ids: dict[str, list[str]] = {}
    for record in records:
        expected_split_ids.setdefault(record.split, []).append(record.graph_id)
    if manifest.get("split_graph_ids") != expected_split_ids:
        raise ValueError(f"dataset split manifest mismatch: {manifest_path}")
    task_type = str(manifest["task_type"])
    if any(record.task_type != task_type for record in records):
        raise ValueError(f"dataset cache contains conflicting task types: {data_path}")
    if not all(np.all(np.isfinite(record.target)) for record in records):
        raise ValueError(f"dataset cache contains a non-finite target: {data_path}")
    return PreparedDataset(
        suite=suite,
        records=records,
        data_path=data_path,
        manifest_path=manifest_path,
        data_sha256=digest,
        target_names=tuple(str(name) for name in manifest["target_names"]),
        task_type=task_type,
    )


def _cache_records(
    *,
    suite: str,
    records: Sequence[GraphRecord],
    data_path: Path,
    manifest_path: Path,
    target_names: Sequence[str],
    task_type: str,
    source: str,
    seed: int,
) -> PreparedDataset:
    payload = {
        "dataset_version": DATASET_VERSION,
        "suite": suite,
        "records": [record.to_payload() for record in records],
    }
    content = _json_bytes(payload)
    digest = _sha256_bytes(content)
    if data_path.exists() or manifest_path.exists():
        cached = _load_cached_dataset(suite=suite, data_path=data_path, manifest_path=manifest_path)
        if cached.data_sha256 != digest:
            raise ValueError(
                f"existing deterministic cache does not match requested seed/options: {data_path}"
            )
        return cached
    split_ids: dict[str, list[str]] = {}
    for record in records:
        split_ids.setdefault(record.split, []).append(record.graph_id)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "suite": suite,
        "source": source,
        "seed": seed,
        "profile": "full",
        "task_type": task_type,
        "target_names": list(target_names),
        "data_path": str(data_path),
        "data_sha256": digest,
        "num_graphs": len(records),
        "split_graph_ids": split_ids,
        "graph_split_before_chart_sampling": True,
        "categorical_feature_schema": {
            "x": "optional non-negative integer atom/node category per node",
            "edge_attr": (
                "optional non-negative integer bond category aligned with each "
                "canonical undirected edge"
            ),
            "missing_value": None,
        },
    }
    _atomic_write(data_path, content)
    _atomic_write(manifest_path, _json_bytes(manifest))
    return _load_cached_dataset(suite=suite, data_path=data_path, manifest_path=manifest_path)


def prepare_cyclecount_dataset(data_root: Path, *, seed: int) -> PreparedDataset:
    """Create or verify the offline CycleCount-style deterministic cache."""

    cache_dir = data_root.expanduser().resolve() / "cyclecount_ood_v2"
    stem = f"seed-{seed}-full"
    data_path = cache_dir / f"{stem}.json"
    manifest_path = cache_dir / f"{stem}.manifest.json"
    if data_path.exists() or manifest_path.exists():
        return validate_prepared_cache("core", data_root, seed=seed)
    records = build_cyclecount_records(seed=seed)
    _validate_protocol_records("core", records)
    return _cache_records(
        suite="core",
        records=records,
        data_path=data_path,
        manifest_path=manifest_path,
        target_names=tuple(f"cycles_len_{length}" for length in TARGET_CYCLE_LENGTHS),
        task_type="regression",
        source="generated://tree_augmentation/cyclecount_ood_v2",
        seed=seed,
    )


def _require_pyg(suite: str) -> tuple[Any, Any]:
    try:
        from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalDatasetError(
            f"suite {suite!r} requires the optional 'torch-geometric' package and its "
            "matching PyTorch wheels. Install it in the active Linux/CUDA environment; "
            "the core offline suite does not require PyG."
        ) from error
    return GNNBenchmarkDataset, ZINC


def _pyg_edges(data: Any) -> tuple[int, tuple[tuple[int, int], ...]]:
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().numpy()
    edges = ((int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1]))
    return num_nodes, _canonical_edges(num_nodes, edges)


def _pyg_categorical_vector(value: Any, *, expected: int, name: str) -> tuple[int, ...]:
    """Read a scalar categorical PyG feature without casting away information."""

    if value is None:
        raise ValueError(f"ZINC record is missing required categorical {name}")
    raw = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.asarray(raw)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) != expected:
        raise ValueError(f"categorical {name} must have shape [{expected}] or [{expected}, 1]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"categorical {name} must use an integer dtype")
    result = tuple(int(item) for item in array.tolist())
    if any(item < 0 for item in result):
        raise ValueError(f"categorical {name} values must be non-negative")
    return result


def zinc_record_from_pyg(
    data: Any,
    *,
    graph_id: str,
    split: str,
) -> GraphRecord:
    """Convert one PyG ZINC molecule while preserving atom and bond categories.

    PyG stores each undirected bond as directed arcs.  The cache stores each
    physical bond once, in canonical edge order, and rejects conflicting
    categories rather than silently choosing one direction.
    """

    num_nodes = int(data.num_nodes)
    x = _pyg_categorical_vector(data.x, expected=num_nodes, name="node x")
    if any(value >= ZINC_NUM_ATOM_TYPES for value in x):
        raise ValueError(f"ZINC node x category exceeds supported range [0, {ZINC_NUM_ATOM_TYPES})")
    edge_index_raw = (
        data.edge_index.detach().cpu().numpy()
        if hasattr(data.edge_index, "detach")
        else np.asarray(data.edge_index)
    )
    edge_index = np.asarray(edge_index_raw)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("PyG edge_index must have shape [2, num_directed_edges]")
    directed_attr = _pyg_categorical_vector(
        data.edge_attr,
        expected=edge_index.shape[1],
        name="bond edge_attr",
    )
    attributes_by_edge: dict[tuple[int, int], int] = {}
    directed_arcs: set[tuple[int, int]] = set()
    for index in range(edge_index.shape[1]):
        u, v = int(edge_index[0, index]), int(edge_index[1, index])
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError("ZINC edge endpoint lies outside [0, num_nodes)")
        if u == v:
            raise ValueError("ZINC self-loops are not supported by the chart protocol")
        if (u, v) in directed_arcs:
            raise ValueError(
                "parallel or duplicate directed ZINC bonds cannot be represented losslessly"
            )
        directed_arcs.add((u, v))
        category = directed_attr[index]
        if category >= ZINC_NUM_BOND_TYPES:
            raise ValueError(
                f"ZINC bond edge_attr category exceeds supported range [0, {ZINC_NUM_BOND_TYPES})"
            )
        edge = (min(u, v), max(u, v))
        previous = attributes_by_edge.setdefault(edge, category)
        if previous != category:
            raise ValueError(f"directed copies of ZINC bond {edge} have conflicting edge_attr")
    edges = _canonical_edges(num_nodes, attributes_by_edge)
    edge_attr = tuple(attributes_by_edge[edge] for edge in edges)
    target_raw = data.y.detach().cpu().numpy() if hasattr(data.y, "detach") else np.asarray(data.y)
    target_array = np.asarray(target_raw).reshape(-1)
    if target_array.size != 1 or not np.isfinite(target_array[0]):
        raise ValueError("ZINC target y must contain exactly one finite scalar")
    return GraphRecord(
        graph_id=graph_id,
        family="ZINC-12K",
        split=split,
        num_nodes=num_nodes,
        edges=edges,
        target=(float(target_array[0]),),
        x=x,
        edge_attr=edge_attr,
    )


def _prepare_csl_records(data_root: Path, *, seed: int) -> tuple[GraphRecord, ...]:
    GNNBenchmarkDataset, _ = _require_pyg("csl")
    raw_root = data_root / "pyg" / "CSL"
    try:
        dataset = GNNBenchmarkDataset(root=str(raw_root), name="CSL")
    except Exception as error:
        raise OptionalDatasetError(
            f"failed to prepare CSL under {raw_root}. Check network access, write permission, "
            "and the PyG dataset download; original error: {error}"
        ) from error
    labels = [int(dataset[index].y.reshape(-1)[0]) for index in range(len(dataset))]
    folds: dict[int, int] = {}
    for label in sorted(set(labels)):
        members = [index for index, value in enumerate(labels) if value == label]
        rng = np.random.default_rng(_stable_seed(f"csl:{label}", seed))
        for position, index in enumerate(rng.permutation(members)):
            folds[int(index)] = position % 5
    records: list[GraphRecord] = []
    for index in range(len(dataset)):
        data = dataset[index]
        num_nodes, edges = _pyg_edges(data)
        fold = folds[index]
        split = "train" if fold < 3 else "validation" if fold == 3 else "test"
        records.append(
            GraphRecord(
                graph_id=f"csl-{index:05d}",
                family="CSL",
                split=split,
                num_nodes=num_nodes,
                edges=edges,
                target=(float(labels[index]),),
                task_type="classification",
            )
        )
    return tuple(records)


def _prepare_zinc_records(data_root: Path) -> tuple[GraphRecord, ...]:
    _, ZINC = _require_pyg("zinc")
    raw_root = data_root / "pyg" / "ZINC"
    records: list[GraphRecord] = []
    for split in ("train", "val", "test"):
        try:
            dataset = ZINC(root=str(raw_root), subset=True, split=split)
        except Exception as error:
            raise OptionalDatasetError(
                f"failed to prepare ZINC-12K split {split!r} under {raw_root}. Check network "
                f"access, write permission, and the PyG dataset download; original error: {error}"
            ) from error
        normalized_split = "validation" if split == "val" else split
        for index in range(len(dataset)):
            data = dataset[index]
            records.append(
                zinc_record_from_pyg(
                    data,
                    graph_id=f"zinc-{normalized_split}-{index:05d}",
                    split=normalized_split,
                )
            )
    return tuple(records)


def prepare_optional_pyg_dataset(
    suite: str,
    data_root: Path,
    *,
    seed: int,
    allow_download: bool = False,
) -> PreparedDataset:
    """Prepare CSL or ZINC through optional PyG adapters with verified caches."""

    normalized = suite.lower()
    if normalized not in {"csl", "zinc"}:
        raise ValueError("optional PyG suite must be csl or zinc")
    cache_dir = data_root.expanduser().resolve() / f"{normalized}_pyg_v2"
    stem = f"seed-{seed}-full"
    data_path = cache_dir / f"{stem}.json"
    manifest_path = cache_dir / f"{stem}.manifest.json"
    if data_path.exists() or manifest_path.exists():
        return validate_prepared_cache(normalized, data_root, seed=seed)
    if not allow_download:
        raise OptionalDatasetError(
            f"suite {normalized!r} has no verified processed cache under {cache_dir}. "
            "Re-run the CLI with --allow-download to let the PyG adapter access its "
            "public dataset endpoint, or copy a complete cache plus manifest here."
        )
    if normalized == "csl":
        records = _prepare_csl_records(data_root.expanduser().resolve(), seed=seed)
        target_names = tuple(f"class_{index}" for index in range(10))
        task_type = "classification"
        source = "PyG:GNNBenchmarkDataset/CSL"
    else:
        records = _prepare_zinc_records(data_root.expanduser().resolve())
        target_names = ("constrained_logP",)
        task_type = "regression"
        source = "PyG:ZINC(subset=True)"
    _validate_protocol_records(normalized, records)
    return _cache_records(
        suite=normalized,
        records=records,
        data_path=data_path,
        manifest_path=manifest_path,
        target_names=target_names,
        task_type=task_type,
        source=source,
        seed=seed,
    )


def validate_prepared_cache(
    suite: str,
    data_root: Path,
    *,
    seed: int,
) -> PreparedDataset:
    """Validate one requested processed cache without generating or downloading data."""

    normalized = suite.lower()
    if normalized not in {"core", "csl", "zinc"}:
        raise ValueError("suite must be core, csl, or zinc")
    cache_name = "cyclecount_ood_v2" if normalized == "core" else f"{normalized}_pyg_v2"
    cache_dir = data_root.expanduser().resolve() / cache_name
    stem = f"seed-{seed}-full"
    data_path = cache_dir / f"{stem}.json"
    manifest_path = cache_dir / f"{stem}.manifest.json"
    present = (data_path.is_file(), manifest_path.is_file())
    if not any(present):
        raise FileNotFoundError(f"tree {normalized} cache is missing for seed={seed}: {data_path}")
    if not all(present):
        raise CacheIncompleteError(
            f"tree {normalized} data and manifest must both exist: {cache_dir}"
        )
    try:
        prepared = _load_cached_dataset(
            suite=normalized, data_path=data_path, manifest_path=manifest_path
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise CacheCorruptError(f"invalid tree {normalized} processed cache") from error
    # Existing full v2 caches used tiny=false. Accept those without rewriting
    # their records or fingerprints, but never accept a reduced legacy cache.
    if (
        manifest.get("seed") != int(seed)
        or manifest.get("tiny", False) is not False
        or manifest.get("profile", "full") != "full"
    ):
        raise CacheWrongRequestError(f"tree {normalized} cache seed/profile mismatch")
    expected_source = {
        "core": "generated://tree_augmentation/cyclecount_ood_v2",
        "csl": "PyG:GNNBenchmarkDataset/CSL",
        "zinc": "PyG:ZINC(subset=True)",
    }[normalized]
    if manifest.get("source") != expected_source:
        raise CacheWrongRequestError(f"tree {normalized} cache source mismatch")
    _validate_protocol_records(normalized, prepared.records)
    return prepared


def _validate_protocol_records(suite: str, records: Sequence[GraphRecord]) -> None:
    """Reject incomplete public splits and reduced caches before paper training."""

    expected_counts = {
        "core": {"train": 128, "validation": 24, "id_test": 40, "ood_test": 40},
        "csl": {"train": 90, "validation": 30, "test": 30},
        "zinc": {"train": 10_000, "validation": 1_000, "test": 1_000},
    }[suite]
    actual_counts: dict[str, int] = {}
    for record in records:
        actual_counts[record.split] = actual_counts.get(record.split, 0) + 1
    if actual_counts != expected_counts:
        raise CacheCorruptError(f"tree {suite} split cardinalities are invalid")
    expected_target_width = 4 if suite == "core" else 1
    for record in records:
        if len(record.target) != expected_target_width or not np.all(np.isfinite(record.target)):
            raise CacheCorruptError(f"tree {suite} target shape or value is invalid")
        if suite == "zinc":
            if record.x is None or len(record.x) != record.num_nodes:
                raise CacheCorruptError("tree ZINC atom features are missing or misaligned")
            if record.edge_attr is None or len(record.edge_attr) != len(record.edges):
                raise CacheCorruptError("tree ZINC bond features are missing or misaligned")
            if any(not 0 <= int(value) < ZINC_NUM_ATOM_TYPES for value in record.x):
                raise CacheCorruptError("tree ZINC atom category is outside the supported range")
            if any(not 0 <= int(value) < ZINC_NUM_BOND_TYPES for value in record.edge_attr):
                raise CacheCorruptError("tree ZINC bond category is outside the supported range")


__all__ = [
    "DATASET_VERSION",
    "TARGET_CYCLE_LENGTHS",
    "ZINC_NUM_ATOM_TYPES",
    "ZINC_NUM_BOND_TYPES",
    "GraphRecord",
    "OptionalDatasetError",
    "PreparedDataset",
    "build_cyclecount_records",
    "build_paper_chart",
    "chart_key",
    "prepare_cyclecount_dataset",
    "prepare_optional_pyg_dataset",
    "sample_paper_charts",
    "simple_cycle_counts",
    "traversal_tree_indices",
    "validate_prepared_cache",
    "wilson_ust_indices",
    "zinc_record_from_pyg",
]
