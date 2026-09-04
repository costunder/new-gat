"""Sparse DFS fundamental coordinates of the graph circulation space.

For edge-by-node incidence ``B``, the cycle space is ``ker(B.T)`` with exact
dimension ``m - n + components``.  No QR, SVD, dense incidence matrix, Gram
matrix or dense projector is needed.  The PE uses the selected cycle supports;
it is not invariant to replacing the DFS forest by a different cycle basis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

BASIS_BACKENDS = ("dfs_fundamental",)
DEFAULT_BASIS_BACKEND = "dfs_fundamental"


def _checked_edges(num_nodes: int, edge_index: ArrayLike) -> NDArray[np.int64]:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, (int, np.integer)):
        raise ValueError("num_nodes must be a positive integer")
    if num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if not np.issubdtype(edges.dtype, np.integer) or np.issubdtype(edges.dtype, np.bool_):
        raise ValueError("edge_index must contain integer node indices")
    edges = edges.astype(np.int64, copy=False)
    if np.any(edges < 0) or np.any(edges >= num_nodes):
        raise ValueError("edge endpoint out of range")
    if np.any(edges[0] == edges[1]):
        raise ValueError("self-loops are not supported")
    undirected = [tuple(sorted(pair)) for pair in edges.T.tolist()]
    if len(set(undirected)) != len(undirected):
        raise ValueError("duplicate undirected edge")
    return edges


def incidence_and_cycle_rank(
    num_nodes: int, edge_index: ArrayLike
) -> tuple[sparse.csr_matrix, int]:
    """Return sparse supplied-orientation incidence ``B[m,n]`` and nullity."""
    edges = _checked_edges(num_nodes, edge_index)
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u_raw, v_raw in edges.T:
        u, v = int(u_raw), int(v_raw)
        adjacency[u].append(v)
        adjacency[v].append(u)
    visited = np.zeros(num_nodes, dtype=np.bool_)
    components = 0
    for root in range(num_nodes):
        if visited[root]:
            continue
        components += 1
        visited[root] = True
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    edge_count = edges.shape[1]
    incidence = sparse.coo_matrix(
        (
            np.tile([-1.0, 1.0], edge_count),
            (np.repeat(np.arange(edge_count), 2), edges.T.reshape(-1)),
        ),
        shape=(edge_count, num_nodes),
        dtype=np.float64,
    ).tocsr()
    return incidence, edge_count - num_nodes + components


def sparse_left_nullspace_basis(num_nodes: int, edge_index: ArrayLike) -> sparse.csr_matrix:
    """Compatibility spelling for the complete signed DFS fundamental basis."""
    return dfs_fundamental_cycle_basis(num_nodes, edge_index)


def dfs_fundamental_cycle_basis(num_nodes: int, edge_index: ArrayLike) -> sparse.csr_matrix:
    """Return all signed DFS cycles; circular coordinates are available separately."""
    return dfs_fundamental_cycle_coordinates(num_nodes, edge_index)[0]


def dfs_fundamental_cycle_coordinates(
    num_nodes: int,
    edge_index: ArrayLike,
) -> tuple[sparse.csr_matrix, NDArray[np.int64]]:
    """Return signed DFS basis and circular edge positions aligned with CSR data.

    DFS chooses one spanning tree per connected component.  Every non-tree edge
    (chord) contributes that edge plus the unique parent path between its
    endpoints.  The chord row of its own column is one and every other chord row
    is zero, so the columns are independent without a numerical rank test.

    Discovering the forest costs ``O(num_nodes + num_edges)``.  Materializing
    the explicit basis additionally costs ``O(nnz(Z))``; that output term can be
    superlinear because one tree edge may occur in many fundamental cycles.
    Positions enumerate the actual chord plus ordered tree path, NOT CSR row
    order.  A position shift/reversal changes the origin/direction only.
    """
    edges = _checked_edges(num_nodes, edge_index)
    edge_count = edges.shape[1]
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge, (tail_raw, head_raw) in enumerate(edges.T):
        tail, head = int(tail_raw), int(head_raw)
        adjacency[tail].append((head, edge))
        adjacency[head].append((tail, edge))

    forest_parent = np.full(num_nodes, -1, dtype=np.int64)
    parent_edge = np.full(num_nodes, -1, dtype=np.int64)
    up_sign = np.zeros(num_nodes, dtype=np.int8)
    depth = np.zeros(num_nodes, dtype=np.int64)
    tree_edge = np.zeros(edge_count, dtype=np.bool_)

    # Store the next adjacency position on the stack.  This is iterative DFS
    # with explicit backtracking, rather than all-simple-cycle enumeration.
    for root in range(num_nodes):
        if forest_parent[root] != -1:
            continue
        forest_parent[root] = root
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            node, position = stack[-1]
            if position == len(adjacency[node]):
                stack.pop()
                continue
            stack[-1] = (node, position + 1)
            neighbor, edge = adjacency[node][position]
            if forest_parent[neighbor] != -1:
                continue
            forest_parent[neighbor] = node
            parent_edge[neighbor] = edge
            depth[neighbor] = depth[node] + 1
            tail, head = map(int, edges[:, edge])
            up_sign[neighbor] = 1 if (neighbor, node) == (tail, head) else -1
            tree_edge[edge] = True
            stack.append((neighbor, 0))

    chords = np.flatnonzero(~tree_edge)
    rank = len(chords)
    if rank == 0:
        return sparse.csr_matrix((edge_count, 0), dtype=np.float32), np.empty(0, dtype=np.int64)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    positions: list[int] = []

    def append_up(vertex: int, path: list[tuple[int, float]], multiplier: int) -> int:
        edge = int(parent_edge[vertex])
        if edge < 0:
            raise RuntimeError("DFS chord endpoints do not share a spanning-tree component")
        path.append((edge, float(multiplier * int(up_sign[vertex]))))
        return int(forest_parent[vertex])

    for column, chord_raw in enumerate(chords):
        chord = int(chord_raw)
        tail, head = map(int, edges[:, chord])
        # Traverse the chord tail -> head, then the parent path head -> tail.
        left, right = head, tail
        left_path: list[tuple[int, float]] = []
        right_path: list[tuple[int, float]] = []
        while depth[left] > depth[right]:
            left = append_up(left, left_path, +1)
        while depth[right] > depth[left]:
            right = append_up(right, right_path, -1)
        while left != right:
            left = append_up(left, left_path, +1)
            right = append_up(right, right_path, -1)
        ordered_path = [(chord, 1.0), *left_path, *reversed(right_path)]
        for position, (edge, sign) in enumerate(ordered_path):
            rows.append(edge)
            columns.append(column)
            values.append(sign)
            positions.append(position)

    # Canonicalize the sparse coordinates once and apply exactly the same
    # permutation to signed coefficients and positions, including position 0.
    layout = sparse.coo_matrix(
        (np.arange(len(values), dtype=np.int64), (rows, columns)),
        shape=(edge_count, rank),
        dtype=np.int64,
    ).tocsr()
    result = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32)[layout.data], layout.indices, layout.indptr),
        shape=layout.shape,
    )
    return result, np.asarray(positions, dtype=np.int64)[layout.data]


def validate_cycle_basis(num_nodes: int, edge_index: ArrayLike, basis: ArrayLike) -> None:
    """Certify a fundamental basis without a dense matrix or factorization.

    Every column must have a nonzero singleton-row witness.  Those rows form a
    nonsingular diagonal submatrix (the chord identity for constructed DFS Z),
    proving independence.  Column scaling/sign/permutation is supported; this
    deliberately does NOT certify arbitrary mixed coordinates ZR without that
    witness.  Such charts are no longer inputs to the sparse cycle-support PE.
    Null residuals are compared relative to each column, never an absolute
    rank threshold that rejects a uniformly small but valid basis.
    """
    incidence, cycle_rank = incidence_and_cycle_rank(num_nodes, edge_index)
    raw = basis if sparse.issparse(basis) else np.asarray(basis)
    if len(raw.shape) != 2 or raw.shape != (incidence.shape[0], cycle_rank):
        raise ValueError(
            f"cycle_basis must have shape ({incidence.shape[0]}, {cycle_rank}); got {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.floating) or raw.dtype.itemsize < 4:
        raise ValueError("cycle_basis storage requires float32 or float64 precision")
    values = sparse.csr_matrix(raw, dtype=np.float64)
    values.sum_duplicates()
    values.eliminate_zeros()
    if not np.all(np.isfinite(values.data)):
        raise ValueError("cycle_basis must contain finite floating-point values")
    if not cycle_rank:
        return
    singleton_rows = np.flatnonzero(np.diff(values.indptr) == 1)
    witnesses = values.indices[values.indptr[singleton_rows]]
    if not np.all(np.bincount(witnesses, minlength=cycle_rank) > 0):
        raise ValueError(
            "cycle_basis full column rank requires a fundamental singleton-row witness "
            "per column; arbitrary mixed bases are unsupported"
        )
    column_scale = np.asarray(abs(values).sum(axis=0)).reshape(-1)
    residual = (incidence.T @ values).tocoo()
    # Constructed signed unit cycles cancel exactly in float64.  Do not let a
    # long cycle's relative tolerance admit a nonzero integer residual.
    signed_unit = np.all(np.abs(values.data) == 1.0)
    threshold = 0.0 if signed_unit else 64.0 * np.finfo(raw.dtype).eps * column_scale[residual.col]
    if residual.nnz and np.any(np.abs(residual.data) > threshold):
        raise ValueError("cycle_basis is not in the left nullspace: B.T @ Z != 0")


def left_nullspace_basis(num_nodes: int, edge_index: ArrayLike) -> sparse.csr_matrix:
    """Return the complete sparse signed DFS basis; no orthogonalization."""
    return build_cycle_basis(num_nodes, edge_index)


def validate_cycle_positions(
    num_nodes: int,
    edge_index: ArrayLike,
    basis: sparse.csr_matrix,
    positions: ArrayLike,
) -> None:
    """Certify every circular position follows one complete simple physical cycle.

    The integer position metadata avoids reconstructing discrete order from
    rounded angles.  Any cyclic origin shift or reversal is allowed; arbitrary
    edge-order permutations are rejected.  No dense graph matrix is allocated.
    """
    edges = _checked_edges(num_nodes, edge_index)
    positions = np.asarray(positions)
    if (
        positions.shape != (basis.nnz,)
        or not np.issubdtype(positions.dtype, np.integer)
        or np.issubdtype(positions.dtype, np.bool_)
    ):
        raise ValueError("cycle positions must be an integer vector aligned with sparse basis data")
    columns = basis.indices
    lengths = np.bincount(columns, minlength=basis.shape[1])
    if np.any(lengths < 3) or np.any(positions < 0) or np.any(positions >= lengths[columns]):
        raise ValueError("cycle positions must cover each simple cycle's complete position range")
    if not basis.nnz:
        return
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    flat_positions = offsets[columns] + positions
    if not np.all(np.bincount(flat_positions, minlength=basis.nnz) == 1):
        raise ValueError("cycle positions must be a permutation of 0..length-1 for each cycle")
    rows = np.repeat(np.arange(basis.shape[0]), np.diff(basis.indptr))
    ordered_rows = np.empty(basis.nnz, dtype=np.int64)
    ordered_rows[flat_positions] = rows
    next_positions = offsets[columns] + (positions + 1) % lengths[columns]
    current_edges = edges[:, rows]
    next_edges = edges[:, ordered_rows[next_positions]]
    tail_matches = (current_edges[0] == next_edges[0]) | (current_edges[0] == next_edges[1])
    head_matches = (current_edges[1] == next_edges[0]) | (current_edges[1] == next_edges[1])
    if not np.all(tail_matches ^ head_matches):
        raise ValueError("cycle positions do not follow adjacent physical edges around the cycle")
    shared_vertices = np.where(tail_matches, current_edges[0], current_edges[1])
    ordered_vertices = np.empty(basis.nnz, dtype=np.int64)
    ordered_vertices[flat_positions] = shared_vertices
    for start, length in zip(offsets, lengths, strict=True):
        if len(set(ordered_vertices[start : start + length].tolist())) != length:
            raise ValueError("cycle positions must follow a simple cycle without repeated vertices")


def cycle_position_factors(
    basis: sparse.csr_matrix,
    positions: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Cosine/sine of actual circular edge positions, aligned with CSR nonzeros."""
    lengths = np.bincount(basis.indices, minlength=basis.shape[1])
    angles = 2.0 * np.pi * positions.astype(np.float64) / lengths[basis.indices]
    return np.ascontiguousarray(np.stack((np.cos(angles), np.sin(angles))), dtype=np.float32)


def build_cycle_basis(
    num_nodes: int,
    edge_index: ArrayLike,
    *,
    backend: str = DEFAULT_BASIS_BACKEND,
) -> sparse.csr_matrix:
    """Build all signed DFS cycles as sparse coordinates of ``ker(B.T)``."""
    return build_cycle_coordinates(num_nodes, edge_index, backend=backend)[0]


def build_cycle_coordinates(
    num_nodes: int,
    edge_index: ArrayLike,
    *,
    backend: str = DEFAULT_BASIS_BACKEND,
) -> tuple[sparse.csr_matrix, NDArray[np.int64]]:
    """Build and certify complete sparse DFS cycles plus their circular positions."""
    if backend not in BASIS_BACKENDS:
        raise ValueError(
            f"basis backend must be one of {BASIS_BACKENDS}; thin_q/projector PE is retired, "
            "use a new sparse-DFS run/cache rather than resuming its checkpoints"
        )
    fundamental, positions = dfs_fundamental_cycle_coordinates(num_nodes, edge_index)
    validate_cycle_basis(num_nodes, edge_index, fundamental)
    validate_cycle_positions(num_nodes, edge_index, fundamental, positions)
    return fundamental, positions


__all__ = [
    "BASIS_BACKENDS",
    "DEFAULT_BASIS_BACKEND",
    "build_cycle_basis",
    "build_cycle_coordinates",
    "cycle_position_factors",
    "dfs_fundamental_cycle_basis",
    "dfs_fundamental_cycle_coordinates",
    "incidence_and_cycle_rank",
    "left_nullspace_basis",
    "sparse_left_nullspace_basis",
    "validate_cycle_basis",
    "validate_cycle_positions",
]
