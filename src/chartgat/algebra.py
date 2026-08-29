"""Linear-algebra utilities for gradient--cycle graph coordinates.

The incidence convention throughout this module is ``B.shape == (m, n)``:
each edge is a row, with ``-1`` at its tail and ``+1`` at its head.  Thus
``B @ p`` is an edge gradient and ``ker(B.T)`` is the circulation space.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def incidence_matrix(
    num_nodes: int,
    edges: Sequence[tuple[int, int]] | Iterable[tuple[int, int]],
    *,
    dtype: np.dtype | type = np.float64,
) -> NDArray:
    """Construct an oriented edge-by-node incidence matrix.

    Parameters
    ----------
    num_nodes:
        Number of graph vertices. Vertices are indexed from ``0``.
    edges:
        Directed representatives ``(tail, head)``. For an undirected graph the
        direction is merely an orientation gauge. Parallel edges and self-loops
        are accepted; a self-loop produces a zero incidence row.
    """

    if not isinstance(num_nodes, (int, np.integer)) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")

    edge_list = list(edges)
    B = np.zeros((len(edge_list), int(num_nodes)), dtype=dtype)
    for edge_index, edge in enumerate(edge_list):
        if len(edge) != 2:
            raise ValueError(f"edge {edge_index} must contain (tail, head)")
        tail, head = edge
        if not isinstance(tail, (int, np.integer)) or not isinstance(head, (int, np.integer)):
            raise TypeError("edge endpoints must be integer node indices")
        if not (0 <= tail < num_nodes and 0 <= head < num_nodes):
            raise ValueError(f"edge {edge_index} has an endpoint out of range")
        B[edge_index, int(tail)] -= 1
        B[edge_index, int(head)] += 1
    return B


def _as_incidence(B: ArrayLike, *, atol: float) -> FloatArray:
    matrix = np.asarray(B, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("B must be a two-dimensional edge-by-node matrix")
    if matrix.shape[1] < 1:
        raise ValueError("B must have at least one node column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("B must contain only finite values")
    if not np.allclose(matrix.sum(axis=1), 0.0, atol=atol, rtol=0.0):
        raise ValueError("each incidence row must sum to zero")
    return matrix


def validate_spanning_tree(
    B: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
    *,
    atol: float = 1e-10,
) -> IntArray:
    """Validate and return the edge indices of a spanning tree.

    For an incidence matrix, selecting ``n - 1`` rows of rank ``n - 1`` is
    equivalent to selecting a spanning tree. This rank formulation also handles
    parallel edges without reconstructing an adjacency list.
    """

    matrix = _as_incidence(B, atol=atol)
    m, n = matrix.shape
    raw = np.asarray(tree_edge_indices)
    if raw.ndim != 1:
        raise ValueError("tree_edge_indices must be one-dimensional")
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.equal(raw, np.floor(raw))):
            raise TypeError("tree edge indices must be integers")
    tree = raw.astype(np.int64, copy=False)
    if tree.size != n - 1:
        raise ValueError(f"a spanning tree on {n} nodes must contain {n - 1} edges")
    if np.unique(tree).size != tree.size:
        raise ValueError("tree edge indices must be unique")
    if np.any(tree < 0) or np.any(tree >= m):
        raise ValueError("tree edge index out of range")

    if n > 1:
        singular_values = np.linalg.svd(matrix[tree], compute_uv=False)
        rank = int(np.count_nonzero(singular_values > atol * max(matrix.shape)))
        if rank != n - 1:
            raise ValueError("selected edges do not form a spanning tree")
    return tree.copy()


def fundamental_cycle_basis(
    B: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
    *,
    return_chords: bool = False,
    atol: float = 1e-10,
) -> FloatArray | tuple[FloatArray, IntArray]:
    """Build the fundamental cycle basis associated with a spanning tree.

    Chords are ordered by their original edge index. The returned ``F`` obeys

    ``B.T @ F == 0`` and ``F[chord_edge_indices, :] == I``.

    Consequently, the physical circulation ``z = F @ a`` is encoded simply by
    reading its values on the chord edges: ``a = z[chord_edge_indices]``.
    """

    matrix = _as_incidence(B, atol=atol)
    m, n = matrix.shape
    tree = validate_spanning_tree(matrix, tree_edge_indices, atol=atol)
    tree_set = set(tree.tolist())
    chords = np.asarray([i for i in range(m) if i not in tree_set], dtype=np.int64)
    beta = chords.size
    F = np.zeros((m, beta), dtype=np.float64)

    if beta:
        tree_incidence_transpose = matrix[tree].T
        for column, chord in enumerate(chords):
            tree_values, *_ = np.linalg.lstsq(tree_incidence_transpose, -matrix[chord], rcond=None)
            rounded = np.rint(tree_values)
            if np.allclose(tree_values, rounded, atol=atol, rtol=0.0):
                tree_values = rounded
            F[tree, column] = tree_values
            F[chord, column] = 1.0

    residual = matrix.T @ F
    scale = max(1.0, float(np.linalg.norm(matrix) * np.linalg.norm(F)))
    if not np.allclose(residual, 0.0, atol=atol * scale, rtol=0.0):
        raise RuntimeError("failed to construct a circulation basis")
    if beta and np.linalg.matrix_rank(F, tol=atol) != beta:
        raise RuntimeError("constructed fundamental cycles are rank deficient")

    if return_chords:
        return F, chords
    return F


def chart_transition(
    source_basis: ArrayLike,
    target_basis: ArrayLike,
    *,
    atol: float = 1e-10,
) -> FloatArray:
    """Return coordinates mapping a source cycle chart into a target chart.

    If ``z = F_source @ a_source = F_target @ a_target``, this function returns
    ``M`` such that ``a_target = M @ a_source`` and
    ``F_target @ M == F_source``. For fundamental cycle charts, ``M`` is a
    unimodular integer matrix (up to floating-point representation).
    """

    source = np.asarray(source_basis, dtype=np.float64)
    target = np.asarray(target_basis, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("cycle bases must be two-dimensional")
    if source.shape != target.shape:
        raise ValueError("source and target bases must have identical shape")
    beta = source.shape[1]
    if beta == 0:
        return np.empty((0, 0), dtype=np.float64)
    if (
        np.linalg.matrix_rank(source, tol=atol) != beta
        or np.linalg.matrix_rank(target, tol=atol) != beta
    ):
        raise ValueError("cycle bases must have full column rank")

    M, *_ = np.linalg.lstsq(target, source, rcond=None)
    scale = max(1.0, float(np.linalg.norm(source)))
    if not np.allclose(target @ M, source, atol=atol * scale, rtol=0.0):
        raise ValueError("source and target do not span the same cycle space")
    rounded = np.rint(M)
    if np.allclose(M, rounded, atol=atol * max(1, beta), rtol=0.0):
        M = rounded
    return M


def encode_edge_state(
    B: ArrayLike,
    cycle_basis: ArrayLike,
    edge_state: ArrayLike,
    *,
    atol: float = 1e-10,
) -> tuple[FloatArray, FloatArray]:
    """Encode ``e`` as ``e = B @ p + F @ a`` using the mean-zero potential.

    Vector-valued edge features are supported: ``edge_state`` may have shape
    ``(m,)`` or ``(m, d)``. Since the two subspaces are orthogonal in the
    unweighted Euclidean metric, the minimum-norm least-squares potential fixes
    the additive potential gauge automatically.
    """

    matrix = _as_incidence(B, atol=atol)
    F = np.asarray(cycle_basis, dtype=np.float64)
    edge = np.asarray(edge_state, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != matrix.shape[0]:
        raise ValueError("cycle_basis must have shape (num_edges, beta)")
    if edge.ndim not in (1, 2) or edge.shape[0] != matrix.shape[0]:
        raise ValueError("edge_state must have shape (num_edges,) or (num_edges, d)")
    if not np.allclose(matrix.T @ F, 0.0, atol=atol, rtol=0.0):
        raise ValueError("cycle_basis columns must lie in ker(B.T)")

    p, *_ = np.linalg.lstsq(matrix, edge, rcond=None)
    cycle_state = edge - matrix @ p
    if F.shape[1]:
        a, *_ = np.linalg.lstsq(F, cycle_state, rcond=None)
    else:
        trailing_shape = edge.shape[1:] if edge.ndim == 2 else ()
        a = np.empty((0, *trailing_shape), dtype=np.float64)

    reconstruction = decode_edge_state(matrix, F, p, a)
    scale = max(1.0, float(np.linalg.norm(edge)))
    if not np.allclose(reconstruction, edge, atol=atol * scale, rtol=0.0):
        raise ValueError("B and cycle_basis do not span the supplied edge state")
    return np.asarray(p), np.asarray(a)


def decode_edge_state(
    B: ArrayLike, cycle_basis: ArrayLike, potential: ArrayLike, cycle_coordinates: ArrayLike
) -> FloatArray:
    """Decode gradient--cycle coordinates as ``B @ p + F @ a``."""

    matrix = np.asarray(B, dtype=np.float64)
    F = np.asarray(cycle_basis, dtype=np.float64)
    p = np.asarray(potential, dtype=np.float64)
    a = np.asarray(cycle_coordinates, dtype=np.float64)
    if matrix.ndim != 2 or F.ndim != 2 or matrix.shape[0] != F.shape[0]:
        raise ValueError("B and cycle_basis must share their edge dimension")
    try:
        return np.asarray(matrix @ p + F @ a)
    except ValueError as exc:
        raise ValueError("incompatible coordinate shapes") from exc


def orthonormal_cycle_basis(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    """Orthonormalize a full-rank cycle basis without changing its span."""

    F = np.asarray(cycle_basis, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("cycle_basis must be two-dimensional")
    m, beta = F.shape
    if beta == 0:
        return np.empty((m, 0), dtype=np.float64)
    if np.linalg.matrix_rank(F, tol=atol) != beta:
        raise ValueError("cycle_basis must have full column rank")
    U, _ = np.linalg.qr(F, mode="reduced")
    # Remove QR's arbitrary sign choice for deterministic tests and artifacts.
    for column in range(beta):
        pivot = int(np.argmax(np.abs(U[:, column])))
        if U[pivot, column] < 0:
            U[:, column] *= -1
    return U


def _orientation_signs(signs: ArrayLike, num_edges: int) -> FloatArray:
    values = np.asarray(signs, dtype=np.float64)
    if values.shape != (num_edges,):
        raise ValueError("orientation signs must have shape (num_edges,)")
    if not np.all(np.isin(values, (-1.0, 1.0))):
        raise ValueError("orientation signs must all equal -1 or +1")
    return values


def flip_incidence(B: ArrayLike, signs: ArrayLike) -> FloatArray:
    """Apply independent edge-orientation flips, equivalently ``Q @ B``."""

    matrix = np.asarray(B, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("B must be two-dimensional")
    values = _orientation_signs(signs, matrix.shape[0])
    return values[:, None] * matrix


def flip_edge_quantity(edge_quantity: ArrayLike, signs: ArrayLike) -> FloatArray:
    """Transform an orientation-covariant edge vector or feature matrix."""

    edge = np.asarray(edge_quantity, dtype=np.float64)
    if edge.ndim not in (1, 2):
        raise ValueError("edge_quantity must be a vector or matrix")
    values = _orientation_signs(signs, edge.shape[0])
    if edge.ndim == 1:
        return values * edge
    return values[:, None] * edge


def flip_cycle_basis(cycle_basis: ArrayLike, signs: ArrayLike) -> FloatArray:
    """Transform physical cycle columns under an edge-orientation flip."""

    F = np.asarray(cycle_basis, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("cycle_basis must be two-dimensional")
    values = _orientation_signs(signs, F.shape[0])
    return values[:, None] * F
