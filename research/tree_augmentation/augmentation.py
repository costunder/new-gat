"""Tree-chart resampling and lossless coordinate-change certification.

No conductance, GAT, node-potential, or flow-completion object is imported or
used in this module.  Every enabled chart keeps the full cycle rank ``beta``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chartgat.algebra import chart_transition, fundamental_cycle_basis, incidence_matrix
from chartgat.graphs import spanning_tree_indices

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TreeChart:
    """A full fundamental-cycle coordinate chart induced by one spanning tree."""

    name: str
    tree_edge_indices: IntArray
    chord_edge_indices: IntArray
    basis: FloatArray

    @property
    def beta(self) -> int:
        """Return the cycle rank represented by this chart."""

        return int(self.basis.shape[1])

    @property
    def num_edges(self) -> int:
        """Return the physical edge dimension."""

        return int(self.basis.shape[0])


def ensure_full_cycle_budget(beta: int, k: int | None = None) -> int:
    """Accept only the full-``beta`` lossless mode.

    ``k < beta`` is a distinct lossy extension.  It is intentionally disabled
    so that truncation cannot silently enter a lossless augmentation result.
    """

    if beta < 0:
        raise ValueError("beta must be non-negative")
    if k is None:
        return beta
    if k < 0 or k > beta:
        raise ValueError("k must lie in [0, beta]")
    if k < beta:
        raise NotImplementedError(
            "k < beta is a disabled lossy extension, not lossless tree augmentation"
        )
    return k


def build_tree_chart(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    method: str,
    seed: int = 0,
    name: str | None = None,
) -> TreeChart:
    """Sample a BFS, DFS, or random tree and construct its full cycle chart."""

    tree = spanning_tree_indices(num_nodes, edges, mode=method, seed=seed)
    B = incidence_matrix(num_nodes, edges)
    basis, chords = fundamental_cycle_basis(B, tree, return_chords=True)
    ensure_full_cycle_budget(int(basis.shape[1]))
    chart_name = name if name is not None else f"{method}:{seed}"
    return TreeChart(chart_name, tree, chords, basis)


def sample_tree_charts(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    include_bfs: bool = True,
    include_dfs: bool = True,
    random_count: int = 0,
    random_seed_start: int = 0,
) -> list[TreeChart]:
    """Return unique full-``beta`` BFS/DFS/random charts.

    Different samplers can occasionally return the same edge set.  Such charts
    are deduplicated because repeated coordinates are not useful augmentation.
    """

    if random_count < 0:
        raise ValueError("random_count must be non-negative")
    requests: list[tuple[str, int]] = []
    if include_bfs:
        requests.append(("bfs", 0))
    if include_dfs:
        requests.append(("dfs", 0))
    requests.extend(("random", random_seed_start + offset) for offset in range(random_count))

    charts: list[TreeChart] = []
    seen: set[tuple[int, ...]] = set()
    for method, seed in requests:
        chart = build_tree_chart(num_nodes, edges, method=method, seed=seed)
        key = tuple(sorted(int(index) for index in chart.tree_edge_indices))
        if key not in seen:
            charts.append(chart)
            seen.add(key)
    if not charts:
        raise ValueError("at least one tree sampler must be enabled")
    return charts


def find_unseen_chart(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    seen_charts: Sequence[TreeChart],
    *,
    seed_start: int = 10_000,
    max_attempts: int = 10_000,
) -> TreeChart:
    """Sample a random chart whose spanning tree was not used for training."""

    seen = {tuple(sorted(int(index) for index in chart.tree_edge_indices)) for chart in seen_charts}
    for offset in range(max_attempts):
        chart = build_tree_chart(
            num_nodes,
            edges,
            method="random",
            seed=seed_start + offset,
            name=f"unseen-random:{seed_start + offset}",
        )
        key = tuple(sorted(int(index) for index in chart.tree_edge_indices))
        if key not in seen:
            return chart
    raise RuntimeError("failed to sample an unseen spanning-tree chart")


def transport_coordinates(
    source: TreeChart,
    target: TreeChart,
    coordinates: ArrayLike,
) -> FloatArray:
    """Transport full cycle coordinates without changing the physical cycle state."""

    if source.basis.shape != target.basis.shape:
        raise ValueError("source and target charts must describe the same edge/cycle dimensions")
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] != source.beta:
        raise ValueError("coordinates must have shape (beta,) or (beta, channels)")
    transition = chart_transition(source.basis, target.basis)
    return np.asarray(transition @ values, dtype=np.float64)


def lossless_transition_error(
    charts: Sequence[TreeChart],
    coordinates: ArrayLike,
) -> float:
    """Return the largest physical reconstruction error over all chart pairs."""

    if not charts:
        raise ValueError("charts must not be empty")
    source = charts[0]
    source_coordinates = np.asarray(coordinates, dtype=np.float64)
    physical = source.basis @ source_coordinates
    maximum = 0.0
    for target in charts:
        target_coordinates = transport_coordinates(source, target, source_coordinates)
        maximum = max(maximum, float(np.linalg.norm(target.basis @ target_coordinates - physical)))
    return maximum


def transition_cocycle_error(charts: Sequence[TreeChart]) -> float:
    """Return the largest spanning-tree chart-transition cocycle residual."""

    if not charts:
        raise ValueError("charts must not be empty")
    maximum = 0.0
    for source in charts:
        for middle in charts:
            source_to_middle = chart_transition(source.basis, middle.basis)
            for target in charts:
                middle_to_target = chart_transition(middle.basis, target.basis)
                source_to_target = chart_transition(source.basis, target.basis)
                residual = middle_to_target @ source_to_middle - source_to_target
                maximum = max(maximum, float(np.linalg.norm(residual)))
    return maximum


def cycle_projector(cycle_basis: ArrayLike) -> FloatArray:
    """Return the physical orthogonal projector onto the represented cycle space."""

    basis = np.asarray(cycle_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("cycle_basis must be two-dimensional")
    beta = basis.shape[1]
    if beta == 0:
        return np.zeros((basis.shape[0], basis.shape[0]), dtype=np.float64)
    if np.linalg.matrix_rank(basis) != beta:
        raise ValueError("cycle_basis must have full column rank")
    gram_inverse = np.linalg.inv(basis.T @ basis)
    projector = basis @ gram_inverse @ basis.T
    return np.asarray((projector + projector.T) / 2.0, dtype=np.float64)


def cycle_projector_diagonal(cycle_basis: ArrayLike) -> FloatArray:
    """Return chart-independent static edge cycle leverage scores."""

    return np.diag(cycle_projector(cycle_basis)).copy()
