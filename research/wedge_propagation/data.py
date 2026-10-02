"""Independent CPU graph/feature data for the fixed-operator experiment.

``full`` is the proposed 198-graph measurement contract, not a training
subset. ``debug`` is a separate, explicit test profile. Each case uses named
local random streams, so execution order and feature draws cannot change its
topology. ``make_specs`` and ``make_case`` support parallel preprocessing.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GraphCase:
    graph_id: str
    family: str
    num_nodes: int
    edges: torch.Tensor
    features: torch.Tensor
    graph_seed: int
    feature_seed: int
    base_tree_seed: int | None = None


@dataclass(frozen=True)
class GraphSpec:
    """A complete, order-independent CPU generation request."""

    graph_id: str
    family: str
    num_nodes: int
    num_features: int
    graph_seed: int
    feature_seed: int
    grid_columns: int | None = None
    base_tree_seed: int | None = None


_PROFILES = {
    "full": ((20, 30, 40, 60, 80, 100), (4, 5, 5, 6, 8, 10), 10, 16),
    "debug": ((5, 8), (1, 2), 2, 4),
}
_DETERMINISTIC_FAMILIES = ("cycle", "star", "grid")
_RANDOM_FAMILIES = ("er", "tree", "tree_chord")


def named_seed(master_seed: int, stream: str, graph_id: str) -> int:
    """Derive a stable unsigned seed without Python's process-randomized hash."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if master_seed < 0:
        raise ValueError("master_seed must be nonnegative")
    payload = f"wedge-fixed-data-v1\0{master_seed}\0{stream}\0{graph_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def make_specs(master_seed: int = 20261002, profile: str = "full") -> list[GraphSpec]:
    """Describe every case; callers can map ``make_case`` over these in workers."""

    if profile not in _PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose 'full' or 'debug'")
    # Validate even before any requests are built.
    named_seed(master_seed, "validation", profile)
    sizes, columns, random_draws, num_features = _PROFILES[profile]
    specs = []
    for num_nodes, grid_columns in zip(sizes, columns, strict=True):
        for family in _DETERMINISTIC_FAMILIES + _RANDOM_FAMILIES:
            draws = random_draws if family in _RANDOM_FAMILIES else 1
            for draw in range(draws):
                graph_id = f"{profile}-{family}-n{num_nodes}-draw{draw}"
                base_tree_seed = None
                if family == "tree_chord":
                    tree_id = f"{profile}-tree-n{num_nodes}-draw{draw}"
                    base_tree_seed = named_seed(master_seed, "graph", tree_id)
                specs.append(
                    GraphSpec(
                        graph_id=graph_id,
                        family=family,
                        num_nodes=num_nodes,
                        num_features=num_features,
                        graph_seed=named_seed(master_seed, "graph", graph_id),
                        feature_seed=named_seed(master_seed, "feature", graph_id),
                        grid_columns=grid_columns if family == "grid" else None,
                        base_tree_seed=base_tree_seed,
                    )
                )
    return specs


def _canonical_pairs(pairs: np.ndarray, num_nodes: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    if np.any(pairs < 0) or np.any(pairs >= num_nodes):
        raise ValueError("edge endpoint outside the declared node set")
    pairs = np.sort(pairs, axis=1)
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("self-loop in a physical-edge graph")
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    if len(pairs) > 1 and np.any(np.all(pairs[1:] == pairs[:-1], axis=1)):
        raise ValueError("duplicate physical edge")
    return pairs


def _tree_pairs(num_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform labeled tree via a uniform Prüfer sequence."""

    sequence = rng.integers(0, num_nodes, size=num_nodes - 2)
    degree = 1 + np.bincount(sequence, minlength=num_nodes)
    leaves = np.flatnonzero(degree == 1).tolist()
    heapq.heapify(leaves)
    pairs = []
    for parent in sequence:
        leaf = heapq.heappop(leaves)
        pairs.append((leaf, int(parent)))
        degree[leaf] -= 1
        degree[parent] -= 1
        if degree[parent] == 1:
            heapq.heappush(leaves, int(parent))
    pairs.append((heapq.heappop(leaves), heapq.heappop(leaves)))
    return _canonical_pairs(np.asarray(pairs), num_nodes)


def make_case(spec: GraphSpec) -> GraphCase:
    """Build one CPU case with no global RNG mutation or external graph code."""

    n = spec.num_nodes
    if n < 3 or spec.num_features < 1:
        raise ValueError("a case needs at least three nodes and one feature realization")
    if spec.graph_seed < 0 or spec.feature_seed < 0:
        raise ValueError("case seeds must be nonnegative")
    rng = np.random.default_rng(spec.graph_seed)
    if spec.family == "cycle":
        nodes = np.arange(n, dtype=np.int64)
        pairs = np.column_stack((nodes, np.roll(nodes, -1)))
    elif spec.family == "star":
        pairs = np.column_stack((np.zeros(n - 1, dtype=np.int64), np.arange(1, n)))
    elif spec.family == "grid":
        columns = spec.grid_columns
        if columns is None or columns < 1 or n % columns:
            raise ValueError("grid columns must divide the full declared node count")
        nodes = np.arange(n, dtype=np.int64).reshape(n // columns, columns)
        horizontal = np.column_stack((nodes[:, :-1].ravel(), nodes[:, 1:].ravel()))
        vertical = np.column_stack((nodes[:-1].ravel(), nodes[1:].ravel()))
        pairs = np.concatenate((horizontal, vertical), axis=0)
    elif spec.family == "er":
        if n < 5:
            raise ValueError("ER probability 4/(n-1) requires at least five nodes")
        left, right = np.triu_indices(n, k=1)
        selected = rng.random(len(left)) < 4 / (n - 1)
        pairs = np.column_stack((left[selected], right[selected]))
    elif spec.family == "tree":
        pairs = _tree_pairs(n, rng)
    elif spec.family == "tree_chord":
        if spec.base_tree_seed is None or spec.base_tree_seed < 0:
            raise ValueError("tree_chord needs the corresponding base tree seed")
        tree = _tree_pairs(n, np.random.default_rng(spec.base_tree_seed))
        left, right = np.triu_indices(n, k=1)
        candidates = np.column_stack((left, right))
        tree_keys = tree[:, 0] * n + tree[:, 1]
        nonedges = candidates[~np.isin(left * n + right, tree_keys)]
        selected = rng.choice(len(nonedges), size=n // 4, replace=False)
        pairs = np.concatenate((tree, nonedges[selected]), axis=0)
    else:
        raise ValueError(f"unknown graph family {spec.family!r}")

    pairs = _canonical_pairs(pairs, n)
    edges = torch.from_numpy(pairs.T.copy())
    features = torch.from_numpy(
        np.random.default_rng(spec.feature_seed).standard_normal((n, spec.num_features))
    )
    return GraphCase(
        graph_id=spec.graph_id,
        family=spec.family,
        num_nodes=n,
        edges=edges,
        features=features,
        graph_seed=spec.graph_seed,
        feature_seed=spec.feature_seed,
        base_tree_seed=spec.base_tree_seed,
    )


def make_cases(master_seed: int = 20261002, profile: str = "full") -> list[GraphCase]:
    """Generate the complete declared profile; no cap, sampling, or fallback."""

    return [make_case(spec) for spec in make_specs(master_seed, profile)]


def graph_content_hash(case: GraphCase) -> str:
    """Hash all nodes, including isolates, and canonical physical edges."""

    payload = case.num_nodes.to_bytes(8, "little")
    payload += case.edges.numpy().astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def feature_content_hash(case: GraphCase) -> str:
    """Hash shape and complete float64 features independently of graph labels."""

    shape = np.asarray(case.features.shape, dtype="<i8").tobytes()
    payload = case.features.numpy().astype("<f8", copy=False).tobytes(order="C")
    return hashlib.sha256(shape + payload).hexdigest()
