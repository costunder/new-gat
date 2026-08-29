"""Static edge positional encodings derived from the graph cycle space.

This module deliberately contains no sample-dependent state and no trainable
operator.  Every feature is a deterministic function of graph topology, an
incidence orientation, and (for chart-dependent variants) a spanning tree.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chartgat.algebra import fundamental_cycle_basis

FloatArray = NDArray[np.float64]

SET_STAT_NAMES = (
    "participation_fraction",
    "magnitude_rms",
    "max_magnitude",
    "mean_cycle_length_fraction",
    "min_cycle_length_fraction",
    "max_cycle_length_fraction",
)


def _as_cycle_basis(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    basis = np.asarray(cycle_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("cycle_basis must have shape (num_edges, cycle_rank)")
    if not np.all(np.isfinite(basis)):
        raise ValueError("cycle_basis must contain only finite values")
    if basis.shape[1] and np.linalg.matrix_rank(basis, tol=atol) != basis.shape[1]:
        raise ValueError("cycle_basis must have full column rank")
    return basis


def static_fundamental_basis(
    incidence: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
) -> FloatArray:
    """Return the fixed fundamental cycle basis for one graph and tree.

    The output is topology-only.  It is intentionally exposed because raw
    fundamental-basis columns are a useful diagnostic PE, even though their
    ordering and signs are chart conventions rather than intrinsic structure.
    """

    return np.asarray(fundamental_cycle_basis(incidence, tree_edge_indices), dtype=np.float64)


def raw_padded_basis_pe(cycle_basis: ArrayLike, max_cycles: int) -> FloatArray:
    """Pad raw fundamental-cycle columns to a common feature width.

    This representation is *not* invariant to cycle-column permutation, sign,
    or spanning-tree choice.  It is included as a transparent diagnostic, not
    as an intrinsic graph PE.
    """

    basis = _as_cycle_basis(cycle_basis)
    if not isinstance(max_cycles, (int, np.integer)) or max_cycles < 0:
        raise ValueError("max_cycles must be a non-negative integer")
    if basis.shape[1] > max_cycles:
        raise ValueError(f"cycle rank {basis.shape[1]} exceeds padded width {max_cycles}")
    padded = np.zeros((basis.shape[0], int(max_cycles)), dtype=np.float64)
    padded[:, : basis.shape[1]] = basis
    return padded


def cycle_set_statistics(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    """Compute per-edge statistics invariant to column signs and permutations.

    The six output columns are listed in :data:`SET_STAT_NAMES`.  These
    statistics summarize a *chosen fundamental cycle set*.  They are invariant
    to reordering or independently flipping that set's columns, but they are
    not claimed to be invariant to replacing the spanning tree.
    """

    basis = _as_cycle_basis(cycle_basis, atol=atol)
    edge_count, cycle_rank = basis.shape
    if cycle_rank == 0:
        return np.zeros((edge_count, len(SET_STAT_NAMES)), dtype=np.float64)

    magnitude = np.abs(basis)
    membership = magnitude > atol
    cycle_lengths = membership.sum(axis=0).astype(np.float64)
    participation = membership.sum(axis=1)
    nonempty = participation > 0

    features = np.zeros((edge_count, len(SET_STAT_NAMES)), dtype=np.float64)
    features[:, 0] = participation / cycle_rank
    features[:, 1] = np.sqrt(np.mean(np.square(magnitude), axis=1))
    features[:, 2] = magnitude.max(axis=1)

    weighted_lengths = membership @ cycle_lengths
    features[nonempty, 3] = (
        weighted_lengths[nonempty] / participation[nonempty] / max(1, edge_count)
    )
    for edge_index in np.flatnonzero(nonempty):
        lengths = cycle_lengths[membership[edge_index]] / max(1, edge_count)
        features[edge_index, 4] = float(lengths.min())
        features[edge_index, 5] = float(lengths.max())
    return features


def cycle_projector(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    r"""Return the orthogonal projector onto :math:`\ker(B^\top)`.

    For any full-rank cycle basis ``F``, this is

    ``P_cycle = F (F.T F)^(-1) F.T``.

    Unlike raw fundamental columns, the result is invariant to every invertible
    change of cycle basis.  Projector-based cycle PE is established prior-style
    methodology and is treated only as a baseline in this research track.
    """

    basis = _as_cycle_basis(cycle_basis, atol=atol)
    edge_count, cycle_rank = basis.shape
    if cycle_rank == 0:
        return np.zeros((edge_count, edge_count), dtype=np.float64)
    gram = basis.T @ basis
    projector = basis @ np.linalg.solve(gram, basis.T)
    # Suppress harmless asymmetry from floating-point linear solves.
    return 0.5 * (projector + projector.T)


def projector_leverage_pe(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    """Return the projector diagonal as one static feature per edge.

    The leverage is zero exactly on bridges (up to numerical precision) and is
    invariant to cycle-basis coordinates and incidence orientation.
    """

    leverage = np.diag(cycle_projector(cycle_basis, atol=atol)).copy()
    leverage[np.abs(leverage) < atol] = 0.0
    return leverage[:, None]


def degree_only_edge_features(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> FloatArray:
    """Build a small topology baseline from unordered endpoint degrees."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    degrees = np.zeros(num_nodes, dtype=np.float64)
    for u, v in edges:
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("edge endpoint out of range")
        if u == v:
            degrees[u] += 2.0
        else:
            degrees[u] += 1.0
            degrees[v] += 1.0

    rows: list[list[float]] = []
    scale = max(1.0, float(num_nodes - 1))
    for u, v in edges:
        low, high = sorted((degrees[u], degrees[v]))
        rows.append(
            [
                low / scale,
                high / scale,
                (low + high) / (2.0 * scale),
                abs(high - low) / scale,
            ]
        )
    return np.asarray(rows, dtype=np.float64).reshape(len(edges), 4)


def static_cycle_feature_bundle(
    incidence: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
    *,
    max_cycles: int,
) -> dict[str, FloatArray]:
    """Construct all static cycle-PE variants for one graph."""

    basis = static_fundamental_basis(incidence, tree_edge_indices)
    return {
        "basis": basis,
        "raw_padded": raw_padded_basis_pe(basis, max_cycles),
        "cycle_set": cycle_set_statistics(basis),
        "projector_leverage": projector_leverage_pe(basis),
    }


__all__ = [
    "SET_STAT_NAMES",
    "cycle_projector",
    "cycle_set_statistics",
    "degree_only_edge_features",
    "projector_leverage_pe",
    "raw_padded_basis_pe",
    "static_cycle_feature_bundle",
    "static_fundamental_basis",
]
