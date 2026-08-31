"""Complete orthonormal left-nullspace bases of oriented graph incidence.

Rows correspond to the supplied, canonical undirected edges. Columns are all
``beta = m - n + c`` basis vectors of ``ker(B.T)``. No statistics, projector,
padding width or truncated spectrum replaces these vectors.

An SVD basis is coordinate-dependent: signs, column order and rotations within
the nullspace are not intrinsic graph labels. Canonical edge ordering stabilizes
one input convention but does not promise relabeling or cross-LAPACK invariance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def incidence_and_cycle_rank(
    num_nodes: int, edge_index: ArrayLike
) -> tuple[NDArray[np.float64], int]:
    """Return ``B[m,n]`` (tail -1, head +1) and exact combinatorial nullity.

    The graph must be simple and loop-free, with exactly one row per bond and
    orientation ``u < v``. Isolated vertices count as connected components.
    Input edge order is preserved, so returned basis rows remain aligned.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, (int, np.integer)):
        raise ValueError("num_nodes must be a positive integer")
    if num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if not np.issubdtype(edges.dtype, np.integer) or np.issubdtype(edges.dtype, np.bool_):
        raise ValueError("edge_index must contain integer node indices")
    if np.any(edges < 0) or np.any(edges >= num_nodes):
        raise ValueError("edge endpoint out of range")
    if np.any(edges[0] >= edges[1]):
        raise ValueError("edges must have canonical loop-free orientation u < v")
    pairs = list(map(tuple, edges.T.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate undirected edge")

    parent = list(range(num_nodes))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    components = num_nodes
    for u, v in pairs:
        left, right = find(u), find(v)
        if left != right:
            parent[right] = left
            components -= 1
    edge_count = edges.shape[1]
    incidence = np.zeros((edge_count, num_nodes), dtype=np.float64)
    rows = np.arange(edge_count)
    incidence[rows, edges[0]] = -1.0
    incidence[rows, edges[1]] = 1.0
    return incidence, edge_count - num_nodes + components


def _orthonormality_tolerance(dtype: np.dtype, edge_count: int, cycle_rank: int) -> float:
    # A rank-deficient Gram matrix has distance at least 1 from identity.
    # Never let size scaling weaken the full-rank certificate to that point.
    return min(
        0.01,
        32.0 * np.finfo(dtype).eps * max(1.0, np.sqrt(edge_count)) * max(1, cycle_rank),
    )


def validate_cycle_basis(num_nodes: int, edge_index: ArrayLike, basis: ArrayLike) -> None:
    """Reject wrong dimension, nonfinite, non-null or non-orthonormal columns.

    Float32 storage is intentional. Checks are accumulated in float64, with
    tolerances scaled by the stored precision, matrix size and input norm.
    Together, exact nullity and orthonormal columns certify a full basis rather
    than merely a collection of vectors from the nullspace.
    """
    incidence, cycle_rank = incidence_and_cycle_rank(num_nodes, edge_index)
    raw = np.asarray(basis)
    if raw.ndim != 2 or raw.shape != (len(incidence), cycle_rank):
        raise ValueError(
            f"cycle_basis must have shape ({len(incidence)}, {cycle_rank}); got {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.floating) or not np.all(np.isfinite(raw)):
        raise ValueError("cycle_basis must contain finite floating-point values")
    if raw.dtype.itemsize < 4:
        raise ValueError("cycle_basis storage requires float32 or float64 precision")
    if not cycle_rank:
        return
    values = raw.astype(np.float64, copy=False)
    epsilon = np.finfo(raw.dtype).eps
    residual = np.linalg.norm(incidence.T @ values, ord="fro")
    residual_scale = max(1.0, np.linalg.norm(incidence, ord="fro")) * max(
        1.0, np.linalg.norm(values, ord="fro")
    )
    if residual > 32.0 * epsilon * residual_scale:
        raise ValueError("cycle_basis is not in the left nullspace: B.T @ U_c != 0")
    gram_error = np.linalg.norm(values.T @ values - np.eye(cycle_rank), ord="fro")
    gram_tolerance = _orthonormality_tolerance(raw.dtype, len(values), cycle_rank)
    if gram_error > gram_tolerance:
        raise ValueError("cycle_basis columns must be orthonormal and full rank")


def left_nullspace_basis(num_nodes: int, edge_index: ArrayLike) -> NDArray[np.float32]:
    """Compute every left-nullspace vector with full SVD, preserving edge rows.

    ``B = U S V.T`` implies ``U_c = U[:, rank(B):]``. The rank is checked
    against ``n-c``; it is not chosen by a user-specified width. Forests have
    shape ``(m, 0)`` and edgeless graphs ``(0, 0)``.
    """
    incidence, cycle_rank = incidence_and_cycle_rank(num_nodes, edge_index)
    edge_count = len(incidence)
    if not cycle_rank:
        return np.empty((edge_count, 0), dtype=np.float32)
    left, singular_values, _ = np.linalg.svd(incidence, full_matrices=True)
    threshold = (
        np.finfo(np.float64).eps
        * max(incidence.shape)
        * (float(singular_values[0]) if len(singular_values) else 0.0)
    )
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    expected_rank = edge_count - cycle_rank
    if numerical_rank != expected_rank:
        raise ValueError(f"incidence SVD rank {numerical_rank} disagrees with n-c={expected_rank}")
    basis = np.ascontiguousarray(left[:, expected_rank:], dtype=np.float64)
    validate_cycle_basis(num_nodes, edge_index, basis)
    result = basis.astype(np.float32)
    validate_cycle_basis(num_nodes, edge_index, result)
    return result


__all__ = ["incidence_and_cycle_rank", "left_nullspace_basis", "validate_cycle_basis"]
