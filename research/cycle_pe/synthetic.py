"""Graph-family splits for static cycle-PE structural probes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from chartgat.algebra import incidence_matrix
from chartgat.graphs import make_connected_graph, spanning_tree_indices

from .features import degree_only_edge_features, static_cycle_feature_bundle

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROBE_VARIANTS = (
    "degree_only",
    "degree_plus_raw",
    "degree_plus_cycle_set",
    "degree_plus_projector_leverage",
)


@dataclass(frozen=True)
class ProbeGraph:
    """One topology-only graph example in a graph-family split."""

    family: str
    split: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ProbeMatrix:
    """Stacked edge examples for one split and one PE variant."""

    features: FloatArray
    targets: IntArray
    graph_ids: tuple[str, ...]


def _canonical_edges(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    canonical = {tuple(sorted((int(u), int(v)))) for u, v in edges if u != v}
    return tuple(sorted(canonical))


def _cycle_edges(offset: int, size: int) -> list[tuple[int, int]]:
    if size < 3:
        raise ValueError("cycle size must be at least three")
    nodes = list(range(offset, offset + size))
    return [(nodes[index], nodes[(index + 1) % size]) for index in range(size)]


def cycle_with_tail(cycle_size: int, tail_length: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    """A single cycle with an attached bridge path."""

    if tail_length < 1:
        raise ValueError("tail_length must be positive")
    edges = _cycle_edges(0, cycle_size)
    previous = 0
    for node in range(cycle_size, cycle_size + tail_length):
        edges.append((previous, node))
        previous = node
    return cycle_size + tail_length, _canonical_edges(edges)


def double_cycle_bridge(
    left_size: int,
    right_size: int,
    bridge_length: int,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Two disjoint cycles connected by a bridge path."""

    if bridge_length < 1:
        raise ValueError("bridge_length must be positive")
    intermediate_count = bridge_length - 1
    right_offset = left_size + intermediate_count
    edges = _cycle_edges(0, left_size)
    edges.extend(_cycle_edges(right_offset, right_size))
    path = [0, *range(left_size, right_offset), right_offset]
    edges.extend(zip(path[:-1], path[1:], strict=True))
    return right_offset + right_size, _canonical_edges(edges)


def cactus_cycle_chain(
    cycle_sizes: Sequence[int],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """A chain of vertex-disjoint cycles connected by single bridge edges."""

    if len(cycle_sizes) < 2:
        raise ValueError("a cactus chain needs at least two cycles")
    edges: list[tuple[int, int]] = []
    offsets: list[int] = []
    offset = 0
    for size in cycle_sizes:
        offsets.append(offset)
        edges.extend(_cycle_edges(offset, int(size)))
        offset += int(size)
    edges.extend((offsets[index], offsets[index + 1]) for index in range(len(offsets) - 1))
    return offset, _canonical_edges(edges)


def bridge_cycle_labels(graph: ProbeGraph) -> IntArray:
    """Label an edge one iff it belongs to at least one undirected cycle."""

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))
    nx_graph.add_edges_from(graph.edges)
    if not nx.is_connected(nx_graph):
        raise ValueError("probe graphs must be connected")
    bridges = {tuple(sorted(edge)) for edge in nx.bridges(nx_graph)}
    return np.asarray([int(edge not in bridges) for edge in graph.edges], dtype=np.int64)


def make_graph_family_split(
    *,
    samples_per_family: int = 8,
    seed: int = 0,
) -> tuple[list[ProbeGraph], list[ProbeGraph]]:
    """Build train/test sets with disjoint graph-generator families.

    Training uses ``cycle_tail`` and ``random_tree_plus_chords``.  Testing uses
    only the unseen ``double_cycle_bridge`` and ``cactus_cycle_chain`` families.
    The split therefore probes family transfer rather than random edge leakage.
    """

    if samples_per_family < 1:
        raise ValueError("samples_per_family must be positive")
    rng = np.random.default_rng(seed)
    train: list[ProbeGraph] = []
    test: list[ProbeGraph] = []

    for sample in range(samples_per_family):
        cycle_size = int(rng.integers(4, 9))
        tail_length = int(rng.integers(3, 8))
        node_count, edges = cycle_with_tail(cycle_size, tail_length)
        train.append(ProbeGraph("cycle_tail", "train", node_count, edges))

        node_count = int(rng.integers(9, 16))
        extra_edges = int(rng.integers(1, min(4, node_count - 2)))
        edges = tuple(make_connected_graph(node_count, extra_edges, seed=seed * 1009 + sample))
        train.append(ProbeGraph("random_tree_plus_chords", "train", node_count, edges))

        left_size = int(rng.integers(3, 8))
        right_size = int(rng.integers(3, 8))
        bridge_length = int(rng.integers(2, 7))
        node_count, edges = double_cycle_bridge(left_size, right_size, bridge_length)
        test.append(ProbeGraph("double_cycle_bridge", "test", node_count, edges))

        sizes = tuple(int(value) for value in rng.integers(3, 8, size=3))
        node_count, edges = cactus_cycle_chain(sizes)
        test.append(ProbeGraph("cactus_cycle_chain", "test", node_count, edges))

    return train, test


def graph_feature_matrix(
    graph: ProbeGraph,
    variant: str,
    *,
    max_cycles: int,
) -> FloatArray:
    """Create one static edge-feature matrix for the requested PE variant."""

    if variant not in PROBE_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {PROBE_VARIANTS}")
    incidence = incidence_matrix(graph.num_nodes, graph.edges)
    tree = spanning_tree_indices(graph.num_nodes, graph.edges, mode="bfs")
    cycle = static_cycle_feature_bundle(incidence, tree, max_cycles=max_cycles)
    degree = degree_only_edge_features(graph.num_nodes, graph.edges)
    if variant == "degree_only":
        return degree
    if variant == "degree_plus_raw":
        return np.concatenate((degree, cycle["raw_padded"]), axis=1)
    if variant == "degree_plus_cycle_set":
        return np.concatenate((degree, cycle["cycle_set"]), axis=1)
    return np.concatenate((degree, cycle["projector_leverage"]), axis=1)


def stack_probe_graphs(
    graphs: Sequence[ProbeGraph],
    variant: str,
    *,
    max_cycles: int,
) -> ProbeMatrix:
    """Stack edges across graphs while retaining graph identifiers."""

    if not graphs:
        raise ValueError("graphs must not be empty")
    feature_parts: list[FloatArray] = []
    target_parts: list[IntArray] = []
    graph_ids: list[str] = []
    for index, graph in enumerate(graphs):
        features = graph_feature_matrix(graph, variant, max_cycles=max_cycles)
        targets = bridge_cycle_labels(graph)
        feature_parts.append(features)
        target_parts.append(targets)
        graph_ids.extend([f"{graph.split}:{graph.family}:{index}"] * len(graph.edges))
    return ProbeMatrix(
        np.concatenate(feature_parts, axis=0),
        np.concatenate(target_parts, axis=0),
        tuple(graph_ids),
    )


__all__ = [
    "PROBE_VARIANTS",
    "ProbeGraph",
    "ProbeMatrix",
    "bridge_cycle_labels",
    "cactus_cycle_chain",
    "cycle_with_tail",
    "double_cycle_bridge",
    "graph_feature_matrix",
    "make_graph_family_split",
    "stack_probe_graphs",
]
