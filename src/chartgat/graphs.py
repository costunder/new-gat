"""Small deterministic graph generators used by the experiment CLIs."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

EdgeList = list[tuple[int, int]]


def make_connected_graph(
    num_nodes: int,
    extra_edges: int,
    *,
    seed: int = 0,
) -> EdgeList:
    """Generate a simple connected undirected graph with fixed orientations.

    A random recursive tree guarantees connectivity; additional edges are then
    sampled without replacement. Every undirected edge is oriented from the
    smaller to the larger node, leaving orientation randomization to explicit
    gauge tests.
    """

    if num_nodes < 2:
        raise ValueError("num_nodes must be at least two")
    max_extra = num_nodes * (num_nodes - 1) // 2 - (num_nodes - 1)
    if not 0 <= extra_edges <= max_extra:
        raise ValueError(f"extra_edges must lie in [0, {max_extra}]")

    rng = np.random.default_rng(seed)
    edge_set: set[tuple[int, int]] = set()
    for node in range(1, num_nodes):
        parent = int(rng.integers(0, node))
        edge_set.add((parent, node))

    candidates = [
        (u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes) if (u, v) not in edge_set
    ]
    if extra_edges:
        chosen = rng.choice(len(candidates), size=extra_edges, replace=False)
        edge_set.update(candidates[int(index)] for index in np.atleast_1d(chosen))
    return sorted(edge_set)


def spanning_tree_indices(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    mode: str = "bfs",
    seed: int = 0,
) -> NDArray[np.int64]:
    """Return edge indices for a BFS, DFS, or random-weight spanning tree."""

    if mode not in {"bfs", "dfs", "random"}:
        raise ValueError("mode must be one of: bfs, dfs, random")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for index, (u, v) in enumerate(edges):
        if u == v:
            continue
        adjacency[u].append((v, index))
        adjacency[v].append((u, index))

    if mode == "random":
        return _random_weight_tree(num_nodes, edges, seed=seed)

    selected: list[int] = []
    seen = {0}
    if mode == "bfs":
        frontier: deque[int] | list[int] = deque([0])
        pop = frontier.popleft
        push = frontier.append
    else:
        frontier = [0]
        pop = frontier.pop
        push = frontier.append

    while frontier:
        node = pop()
        neighbors = sorted(adjacency[node], reverse=(mode == "dfs"))
        for neighbor, edge_index in neighbors:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            selected.append(edge_index)
            push(neighbor)
    if len(seen) != num_nodes:
        raise ValueError("graph is disconnected")
    return np.asarray(selected, dtype=np.int64)


def _random_weight_tree(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    seed: int,
) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random(len(edges)))
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    selected: list[int] = []
    for edge_index in order:
        u, v = edges[int(edge_index)]
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue
        if rank[root_u] < rank[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        if rank[root_u] == rank[root_v]:
            rank[root_u] += 1
        selected.append(int(edge_index))
        if len(selected) == num_nodes - 1:
            break
    if len(selected) != num_nodes - 1:
        raise ValueError("graph is disconnected")
    return np.asarray(selected, dtype=np.int64)


__all__ = ["make_connected_graph", "spanning_tree_indices"]
