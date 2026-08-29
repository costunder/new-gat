"""Deterministic datasets and exact short-cycle labels for the paper path.

The built-in ``CycleCount-OOD`` suite is intentionally self contained: it does
not download public data and uses only NetworkX/NumPy.  Public benchmarks are
adapted lazily in :mod:`research.cycle_pe.paper_adapters`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from chartgat.cache import CacheCorruptError, CacheWrongRequestError, atomic_publish

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CYCLE_LENGTHS = (3, 4, 5, 6)
EDGE_TARGET_NAMES = tuple(f"edge_c{length}" for length in CYCLE_LENGTHS) + (
    "edge_shortest_cycle",
    "edge_short_cycle_congestion",
)
NODE_TARGET_NAMES = tuple(f"node_c{length}" for length in CYCLE_LENGTHS)
GRAPH_TARGET_NAMES = tuple(f"graph_c{length}" for length in CYCLE_LENGTHS)
CORE_SPLITS = ("train", "validation", "id_test", "size_ood", "family_ood")
GENERATOR_VERSION = "cycle-count-ood-v4"


@dataclass
class PaperGraph:
    """One connected undirected graph and optional supervision arrays."""

    graph_id: str
    split: str
    family: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    node_features: FloatArray | None = None
    edge_features: FloatArray | None = None
    edge_targets: FloatArray | None = None
    node_targets: FloatArray | None = None
    graph_targets: FloatArray | None = None

    @property
    def beta(self) -> int:
        return len(self.edges) - self.num_nodes + 1


@dataclass
class DatasetBundle:
    """Named graph splits and target metadata used by the common trainer."""

    name: str
    splits: dict[str, list[PaperGraph]]
    edge_target_names: tuple[str, ...] = ()
    node_target_names: tuple[str, ...] = ()
    graph_target_names: tuple[str, ...] = ()
    cache_path: Path | None = None
    cache_sha256: str | None = None
    metadata: dict[str, Any] | None = None


def canonical_edges(edges: Any) -> tuple[tuple[int, int], ...]:
    """Return a deterministic simple undirected edge tuple."""

    values = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if int(u) != int(v)}
    return tuple(sorted(values))


def enumerate_short_cycles(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
    *,
    max_length: int = 6,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate each undirected simple cycle of length at most ``max_length``.

    The smallest vertex is fixed as the start and the two orientations are
    broken by comparing the first and last vertices.  This avoids relying on
    the ordering conventions of a library cycle-basis implementation.
    """

    if max_length < 3:
        return ()
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    cycles: list[tuple[int, ...]] = []
    for start in range(num_nodes):
        for first in adjacency[start]:
            if first <= start:
                continue
            stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [
                (first, (start, first), frozenset((start, first)))
            ]
            while stack:
                current, path, seen = stack.pop()
                for neighbor in reversed(adjacency[current]):
                    if neighbor == start:
                        if len(path) >= 3 and path[1] < path[-1]:
                            cycles.append(path)
                        continue
                    if len(path) >= max_length or neighbor <= start or neighbor in seen:
                        continue
                    stack.append((neighbor, (*path, neighbor), seen | {neighbor}))
    return tuple(sorted(cycles, key=lambda item: (len(item), item)))


def exact_cycle_targets(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build edge/node/graph C3--C6 counts plus edge length/congestion labels."""

    edge_index = {edge: index for index, edge in enumerate(edges)}
    edge_counts = np.zeros((len(edges), len(CYCLE_LENGTHS)), dtype=np.float64)
    node_counts = np.zeros((num_nodes, len(CYCLE_LENGTHS)), dtype=np.float64)
    graph_counts = np.zeros(len(CYCLE_LENGTHS), dtype=np.float64)
    shortest = _shortest_cycle_lengths(num_nodes, edges)

    for cycle in enumerate_short_cycles(num_nodes, edges, max_length=max(CYCLE_LENGTHS)):
        length = len(cycle)
        if length not in CYCLE_LENGTHS:
            continue
        target_index = CYCLE_LENGTHS.index(length)
        graph_counts[target_index] += 1.0
        node_counts[np.asarray(cycle, dtype=np.int64), target_index] += 1.0
        cycle_edges = [
            (min(cycle[i], cycle[(i + 1) % length]), max(cycle[i], cycle[(i + 1) % length]))
            for i in range(length)
        ]
        for edge in cycle_edges:
            index = edge_index[edge]
            edge_counts[index, target_index] += 1.0

    congestion = edge_counts.sum(axis=1)
    edge_targets = np.concatenate((edge_counts, shortest[:, None], congestion[:, None]), axis=1)
    return edge_targets, node_counts, graph_counts


def _shortest_cycle_lengths(num_nodes: int, edges: tuple[tuple[int, int], ...]) -> FloatArray:
    """Return the exact girth-through-edge, with zero reserved for bridges.

    Removing edge ``(u, v)`` turns its shortest containing cycle into the
    shortest remaining ``u``--``v`` path plus that removed edge.  The sparse
    BFS implementation avoids enumerating long cycles.
    """

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    result = np.zeros(len(edges), dtype=np.float64)
    for edge_index, (source, target) in enumerate(edges):
        distance = [-1] * num_nodes
        distance[source] = 0
        frontier: deque[int] = deque((source,))
        while frontier and distance[target] < 0:
            node = frontier.popleft()
            for neighbor in adjacency[node]:
                if (node == source and neighbor == target) or (
                    node == target and neighbor == source
                ):
                    continue
                if distance[neighbor] >= 0:
                    continue
                distance[neighbor] = distance[node] + 1
                frontier.append(neighbor)
        if distance[target] >= 0:
            result[edge_index] = float(distance[target] + 1)
    return result


def _tree_plus_chords(num_nodes: int, beta: int, seed: int) -> tuple[tuple[int, int], ...]:
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    for node in range(1, num_nodes):
        parent = int(rng.integers(0, node))
        edges.add((parent, node))
    candidates = [
        (u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes) if (u, v) not in edges
    ]
    chosen = rng.choice(len(candidates), size=beta, replace=False)
    edges.update(candidates[int(index)] for index in np.atleast_1d(chosen))
    return tuple(sorted(edges))


def _random_regular(num_nodes: int, seed: int) -> tuple[tuple[int, int], ...]:
    # Cubic graphs give useful short-cycle variation while remaining sparse.
    if num_nodes % 2:
        num_nodes += 1
    for attempt in range(128):
        graph = nx.random_regular_graph(3, num_nodes, seed=seed + attempt)
        if nx.is_connected(graph):
            return canonical_edges(graph.edges())
    raise RuntimeError("failed to generate a connected random-regular graph")


def _small_world(num_nodes: int, seed: int) -> tuple[tuple[int, int], ...]:
    graph = nx.connected_watts_strogatz_graph(
        num_nodes,
        k=4,
        p=0.35,
        tries=256,
        seed=seed,
    )
    return canonical_edges(graph.edges())


def _local_chords(num_nodes: int, beta: int, seed: int) -> tuple[tuple[int, int], ...]:
    """Generate a path with local chords, producing overlapping cycle blocks."""

    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = {(node, node + 1) for node in range(num_nodes - 1)}
    candidates: list[tuple[int, int]] = []
    for span in (2, 3, 4, 5):
        candidates.extend((start, start + span) for start in range(num_nodes - span))
    rng.shuffle(candidates)
    for edge in candidates:
        if edge not in edges:
            edges.add(edge)
        if len(edges) == num_nodes - 1 + beta:
            break
    if len(edges) != num_nodes - 1 + beta:
        raise RuntimeError("not enough distinct local chord candidates")
    return tuple(sorted(edges))


def _graph_seed(seed: int, split_index: int, sample_index: int) -> int:
    sequence = np.random.SeedSequence([int(seed), int(split_index), int(sample_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _generate_graph(split: str, index: int, seed: int) -> PaperGraph:
    split_index = CORE_SPLITS.index(split)
    graph_seed = _graph_seed(seed, split_index, index)
    rng = np.random.default_rng(graph_seed)

    if split == "size_ood":
        num_nodes = int(rng.integers(28, 39))
        if index % 2:
            num_nodes += num_nodes % 2
            family = "random_regular"
            edges = _random_regular(num_nodes, graph_seed)
        else:
            family = "tree_plus_chords"
            beta = int(rng.integers(8, 15))
            edges = _tree_plus_chords(num_nodes, beta, graph_seed)
    elif split == "family_ood":
        num_nodes = int(rng.integers(14, 23))
        if index % 2:
            family = "small_world"
            edges = _small_world(num_nodes, graph_seed)
        else:
            family = "local_chords"
            beta = int(rng.integers(4, min(9, num_nodes - 2)))
            edges = _local_chords(num_nodes, beta, graph_seed)
    else:
        num_nodes = int(rng.integers(14, 23))
        if index % 2:
            num_nodes += num_nodes % 2
            family = "random_regular"
            edges = _random_regular(num_nodes, graph_seed)
        else:
            family = "tree_plus_chords"
            beta = int(rng.integers(3, min(9, num_nodes - 2)))
            edges = _tree_plus_chords(num_nodes, beta, graph_seed)

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edges)
    if not nx.is_connected(nx_graph):
        raise RuntimeError("CycleCount-OOD generator produced a disconnected graph")
    edge_targets, node_targets, graph_targets = exact_cycle_targets(num_nodes, edges)
    return PaperGraph(
        graph_id=f"{split}:{family}:{index:06d}:{graph_seed}",
        split=split,
        family=family,
        num_nodes=num_nodes,
        edges=edges,
        edge_targets=edge_targets,
        node_targets=node_targets,
        graph_targets=graph_targets,
    )


def cycle_count_split_sizes(tiny: bool) -> dict[str, int]:
    if tiny:
        return {
            "train": 10,
            "validation": 4,
            "id_test": 4,
            "size_ood": 4,
            "family_ood": 4,
        }
    return {
        "train": 10_000,
        "validation": 2_000,
        "id_test": 2_000,
        "size_ood": 3_000,
        "family_ood": 3_000,
    }


def _graph_to_json(graph: PaperGraph) -> dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "split": graph.split,
        "family": graph.family,
        "num_nodes": graph.num_nodes,
        "edges": [list(edge) for edge in graph.edges],
        "edge_targets": graph.edge_targets.tolist() if graph.edge_targets is not None else None,
        "node_targets": graph.node_targets.tolist() if graph.node_targets is not None else None,
        "graph_targets": (
            graph.graph_targets.tolist() if graph.graph_targets is not None else None
        ),
    }


def _graph_from_json(record: dict[str, Any]) -> PaperGraph:
    def array_or_none(name: str) -> FloatArray | None:
        value = record.get(name)
        return None if value is None else np.asarray(value, dtype=np.float64)

    return PaperGraph(
        graph_id=str(record["graph_id"]),
        split=str(record["split"]),
        family=str(record["family"]),
        num_nodes=int(record["num_nodes"]),
        edges=canonical_edges(record["edges"]),
        edge_targets=array_or_none("edge_targets"),
        node_targets=array_or_none("node_targets"),
        graph_targets=array_or_none("graph_targets"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cycle_count_specification(
    *, seed: int, tiny: bool, split_sizes: dict[str, int] | None
) -> tuple[dict[str, int], dict[str, Any], Path]:
    sizes = dict(cycle_count_split_sizes(tiny) if split_sizes is None else split_sizes)
    if set(sizes) != set(CORE_SPLITS) or any(int(value) < 1 for value in sizes.values()):
        raise ValueError(f"split_sizes must provide positive counts for {CORE_SPLITS}")
    specification = {
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "split_sizes": {name: int(sizes[name]) for name in CORE_SPLITS},
        "cycle_lengths": list(CYCLE_LENGTHS),
    }
    key = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return sizes, specification, Path(f"{GENERATOR_VERSION}-{key}.json.gz")


def _validate_cycle_count_payload(
    payload: Any, specification: dict[str, Any]
) -> dict[str, list[PaperGraph]]:
    if not isinstance(payload, dict):
        raise CacheCorruptError("CycleCount cache root must be a mapping")
    if payload.get("schema_version") != 1:
        raise CacheWrongRequestError("unsupported CycleCount cache schema")
    if payload.get("specification") != specification:
        raise CacheWrongRequestError("CycleCount cache specification mismatch")
    expected_targets = {
        "edge": list(EDGE_TARGET_NAMES),
        "node": list(NODE_TARGET_NAMES),
        "graph": list(GRAPH_TARGET_NAMES),
    }
    if payload.get("target_names") != expected_targets:
        raise CacheCorruptError("CycleCount target schema mismatch")
    records = payload.get("graphs")
    if not isinstance(records, list):
        raise CacheCorruptError("CycleCount graphs must be a list")
    expected_total = sum(int(value) for value in specification["split_sizes"].values())
    if len(records) != expected_total:
        raise CacheCorruptError("CycleCount graph count does not match the requested splits")
    splits = {name: [] for name in CORE_SPLITS}
    graph_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise CacheCorruptError("CycleCount graph record must be a mapping")
        try:
            graph = _graph_from_json(record)
        except (KeyError, TypeError, ValueError) as error:
            raise CacheCorruptError("invalid CycleCount graph record") from error
        if graph.split not in splits:
            raise CacheCorruptError(f"unknown CycleCount split {graph.split!r}")
        if not graph.graph_id or graph.graph_id in graph_ids:
            raise CacheCorruptError("CycleCount graph IDs must be non-empty and unique")
        graph_ids.add(graph.graph_id)
        edge_count = len(graph.edges)
        raw_edges = record.get("edges")
        if not isinstance(raw_edges, list) or len(raw_edges) != edge_count:
            raise CacheCorruptError("CycleCount edges are duplicated, malformed, or missing")
        if graph.num_nodes < 2 or edge_count < 1:
            raise CacheCorruptError("CycleCount graph dimensions are invalid")
        if any(u < 0 or v >= graph.num_nodes for u, v in graph.edges):
            raise CacheCorruptError("CycleCount edge endpoint lies outside the graph")
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(graph.num_nodes))
        nx_graph.add_edges_from(graph.edges)
        if not nx.is_connected(nx_graph):
            raise CacheCorruptError("CycleCount graph must be connected")
        arrays = (
            (graph.edge_targets, (edge_count, len(EDGE_TARGET_NAMES))),
            (graph.node_targets, (graph.num_nodes, len(NODE_TARGET_NAMES))),
            (graph.graph_targets, (len(GRAPH_TARGET_NAMES),)),
        )
        for array, shape in arrays:
            if array is None or array.shape != shape or not np.all(np.isfinite(array)):
                raise CacheCorruptError("CycleCount target tensor has invalid shape or values")
        splits[graph.split].append(graph)
    actual_sizes = {name: len(graphs) for name, graphs in splits.items()}
    if actual_sizes != specification["split_sizes"]:
        raise CacheCorruptError("CycleCount split cardinalities do not match the specification")
    for graphs in splits.values():
        graphs.sort(key=lambda graph: graph.graph_id)
    return splits


def validate_cycle_count_ood_cache(
    data_root: Path,
    *,
    seed: int,
    tiny: bool = False,
    split_sizes: dict[str, int] | None = None,
) -> DatasetBundle:
    """Read and fully validate a requested CycleCount-OOD cache without writing."""

    _, specification, filename = _cycle_count_specification(
        seed=seed, tiny=tiny, split_sizes=split_sizes
    )
    cache_path = data_root.expanduser().resolve() / "cycle_count_ood" / filename
    if not cache_path.is_file():
        raise FileNotFoundError(f"CycleCount cache is missing for seed={seed}: {cache_path}")
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"failed to parse CycleCount cache: {cache_path}") from error
    splits = _validate_cycle_count_payload(payload, specification)
    return DatasetBundle(
        name="CycleCount-OOD",
        splits=splits,
        edge_target_names=EDGE_TARGET_NAMES,
        node_target_names=NODE_TARGET_NAMES,
        graph_target_names=GRAPH_TARGET_NAMES,
        cache_path=cache_path,
        cache_sha256=sha256_file(cache_path),
        metadata=specification,
    )


def load_or_generate_cycle_count_ood(
    data_root: Path,
    *,
    seed: int,
    tiny: bool = False,
    split_sizes: dict[str, int] | None = None,
) -> DatasetBundle:
    """Load a content-addressed cache or deterministically build CycleCount-OOD."""

    sizes, specification, filename = _cycle_count_specification(
        seed=seed, tiny=tiny, split_sizes=split_sizes
    )
    cache_dir = data_root.expanduser().resolve() / "cycle_count_ood"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / filename

    if cache_path.exists():
        return validate_cycle_count_ood_cache(
            data_root, seed=seed, tiny=tiny, split_sizes=split_sizes
        )
    records = []
    for split in CORE_SPLITS:
        for index in range(int(sizes[split])):
            records.append(_graph_to_json(_generate_graph(split, index, seed)))
    payload = {
        "schema_version": 1,
        "specification": specification,
        "target_names": {
            "edge": list(EDGE_TARGET_NAMES),
            "node": list(NODE_TARGET_NAMES),
            "graph": list(GRAPH_TARGET_NAMES),
        },
        "graphs": records,
    }

    def write(temporary: Path) -> None:
        # mtime=0 makes the compressed cache byte-deterministic as well.
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(filename="", fileobj=raw_stream, mode="wb", mtime=0) as zipped:
                zipped.write(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )

    def validate_temporary(temporary: Path) -> None:
        with gzip.open(temporary, "rt", encoding="utf-8") as stream:
            _validate_cycle_count_payload(json.load(stream), specification)

    atomic_publish(cache_path, write, validator=validate_temporary)
    return validate_cycle_count_ood_cache(data_root, seed=seed, tiny=tiny, split_sizes=split_sizes)


def structural_input_features(graph: PaperGraph) -> tuple[FloatArray, FloatArray]:
    """Return node/edge attributes without leaking any exact cycle label."""

    if graph.node_features is not None:
        node_features = np.asarray(graph.node_features, dtype=np.float64)
    else:
        degrees = np.zeros(graph.num_nodes, dtype=np.float64)
        for u, v in graph.edges:
            degrees[u] += 1.0
            degrees[v] += 1.0
        scale = max(1.0, float(graph.num_nodes - 1))
        node_features = np.column_stack((np.ones(graph.num_nodes), degrees / scale))

    if graph.edge_features is not None:
        edge_features = np.asarray(graph.edge_features, dtype=np.float64)
    else:
        degrees = np.zeros(graph.num_nodes, dtype=np.float64)
        for u, v in graph.edges:
            degrees[u] += 1.0
            degrees[v] += 1.0
        scale = max(1.0, float(graph.num_nodes - 1))
        rows = []
        for u, v in graph.edges:
            low, high = sorted((degrees[u], degrees[v]))
            rows.append((1.0, low / scale, high / scale, abs(high - low) / scale))
        edge_features = np.asarray(rows, dtype=np.float64).reshape(len(graph.edges), 4)
    return node_features, edge_features


__all__ = [
    "CORE_SPLITS",
    "CYCLE_LENGTHS",
    "DatasetBundle",
    "EDGE_TARGET_NAMES",
    "GENERATOR_VERSION",
    "GRAPH_TARGET_NAMES",
    "NODE_TARGET_NAMES",
    "PaperGraph",
    "canonical_edges",
    "cycle_count_split_sizes",
    "enumerate_short_cycles",
    "exact_cycle_targets",
    "load_or_generate_cycle_count_ood",
    "sha256_file",
    "structural_input_features",
    "validate_cycle_count_ood_cache",
]
