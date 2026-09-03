"""Sparse, chart-independent construction of the graph circulation space.

For edge-by-node incidence ``B``, the cycle space is ``ker(B.T)`` with exact
dimension ``m - n + components``.  We construct a sparse fundamental basis;
the model subsequently uses only its coordinate-free orthogonal projector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

BASIS_BACKENDS = ("thin_q", "dfs_fundamental")
DEFAULT_BASIS_BACKEND = "thin_q"


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
) -> tuple[NDArray[np.float64], int]:
    """Return supplied-orientation incidence ``B[m,n]`` and exact nullity."""
    edges = _checked_edges(num_nodes, edge_index)
    parent = np.arange(num_nodes, dtype=np.int64)
    size = np.ones(num_nodes, dtype=np.int64)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    components = num_nodes
    for u_raw, v_raw in edges.T:
        u, v = find(int(u_raw)), find(int(v_raw))
        if u == v:
            continue
        if size[u] < size[v]:
            u, v = v, u
        parent[v] = u
        size[u] += size[v]
        components -= 1
    edge_count = edges.shape[1]
    incidence = np.zeros((edge_count, num_nodes), dtype=np.float64)
    rows = np.arange(edge_count)
    incidence[rows, edges[0]] = -1.0
    incidence[rows, edges[1]] = 1.0
    return incidence, edge_count - num_nodes + components


def sparse_left_nullspace_basis(num_nodes: int, edge_index: ArrayLike) -> sparse.csr_matrix:
    """Build a sparse full basis of ``ker(B.T)`` without SVD/eigendecomposition.

    Union/find selects a spanning forest.  Each chord closes one fundamental
    cycle, so the chord rows form an identity matrix and prove full rank.
    Complexity is linear plus the total length of the fundamental cycles.
    """
    edges = _checked_edges(num_nodes, edge_index)
    edge_count = edges.shape[1]
    union_parent = np.arange(num_nodes, dtype=np.int64)
    union_size = np.ones(num_nodes, dtype=np.int64)

    def find(vertex: int) -> int:
        while union_parent[vertex] != vertex:
            union_parent[vertex] = union_parent[union_parent[vertex]]
            vertex = int(union_parent[vertex])
        return vertex

    tree_edges: list[int] = []
    chords: list[int] = []
    for edge, (u_raw, v_raw) in enumerate(edges.T):
        u, v = find(int(u_raw)), find(int(v_raw))
        if u == v:
            chords.append(edge)
            continue
        if union_size[u] < union_size[v]:
            u, v = v, u
        union_parent[v] = u
        union_size[u] += union_size[v]
        tree_edges.append(edge)
    rank = len(chords)
    if rank == 0:
        return sparse.csr_matrix((edge_count, 0), dtype=np.float32)

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge in tree_edges:
        tail, head = map(int, edges[:, edge])
        adjacency[tail].append((head, edge))
        adjacency[head].append((tail, edge))

    # up_sign[v] is the coefficient for traversing v -> parent[v].
    forest_parent = np.full(num_nodes, -1, dtype=np.int64)
    parent_edge = np.full(num_nodes, -1, dtype=np.int64)
    up_sign = np.zeros(num_nodes, dtype=np.int8)
    depth = np.zeros(num_nodes, dtype=np.int64)
    for root in range(num_nodes):
        if forest_parent[root] != -1:
            continue
        forest_parent[root] = root
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor, edge in adjacency[node]:
                if forest_parent[neighbor] != -1:
                    continue
                forest_parent[neighbor] = node
                parent_edge[neighbor] = edge
                depth[neighbor] = depth[node] + 1
                tail, head = map(int, edges[:, edge])
                up_sign[neighbor] = 1 if (neighbor, node) == (tail, head) else -1
                stack.append(neighbor)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def append_up(vertex: int, column: int, multiplier: int) -> int:
        rows.append(int(parent_edge[vertex]))
        columns.append(column)
        values.append(float(multiplier * int(up_sign[vertex])))
        return int(forest_parent[vertex])

    for column, chord in enumerate(chords):
        tail, head = map(int, edges[:, chord])
        # chord tail -> head is closed by tree path head -> tail.
        left, right = head, tail
        while depth[left] > depth[right]:
            left = append_up(left, column, +1)
        while depth[right] > depth[left]:
            right = append_up(right, column, -1)
        while left != right:
            left = append_up(left, column, +1)
            right = append_up(right, column, -1)
        rows.append(chord)
        columns.append(column)
        values.append(1.0)

    result = sparse.coo_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(edge_count, rank),
        dtype=np.float32,
    ).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    return result


def dfs_fundamental_cycle_basis(num_nodes: int, edge_index: ArrayLike) -> sparse.csr_matrix:
    """Return the signed fundamental-cycle basis of an iterative DFS forest.

    DFS chooses one spanning tree per connected component.  Every non-tree edge
    (chord) contributes that edge plus the unique parent path between its
    endpoints.  The chord row of its own column is one and every other chord row
    is zero, so the columns are independent without a numerical rank test.

    Discovering the forest costs ``O(num_nodes + num_edges)``.  Materializing
    the explicit basis additionally costs ``O(nnz(Z))``; that output term can be
    superlinear because one tree edge may occur in many fundamental cycles.
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
        return sparse.csr_matrix((edge_count, 0), dtype=np.float32)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def append_up(vertex: int, column: int, multiplier: int) -> int:
        edge = int(parent_edge[vertex])
        if edge < 0:
            raise RuntimeError("DFS chord endpoints do not share a spanning-tree component")
        rows.append(edge)
        columns.append(column)
        values.append(float(multiplier * int(up_sign[vertex])))
        return int(forest_parent[vertex])

    for column, chord_raw in enumerate(chords):
        chord = int(chord_raw)
        tail, head = map(int, edges[:, chord])
        # Traverse the chord tail -> head, then the parent path head -> tail.
        left, right = head, tail
        while depth[left] > depth[right]:
            left = append_up(left, column, +1)
        while depth[right] > depth[left]:
            right = append_up(right, column, -1)
        while left != right:
            left = append_up(left, column, +1)
            right = append_up(right, column, -1)
        rows.append(chord)
        columns.append(column)
        values.append(1.0)

    result = sparse.coo_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(edge_count, rank),
        dtype=np.float32,
    ).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    return result


def validate_cycle_basis(num_nodes: int, edge_index: ArrayLike, basis: ArrayLike) -> None:
    """Certify dimension, finite values, nullness and full column rank.

    Orthonormality is deliberately not required: a cycle space is intrinsic,
    while a coordinate chart is not.  The downstream PE is invariant to every
    invertible basis replacement ``Z -> ZR``.
    """
    incidence, cycle_rank = incidence_and_cycle_rank(num_nodes, edge_index)
    if sparse.issparse(basis):
        raw = basis
        shape = raw.shape
        if not np.issubdtype(raw.dtype, np.floating) or not np.all(np.isfinite(raw.data)):
            raise ValueError("cycle_basis must contain finite floating-point values")
        values = raw.astype(np.float64, copy=False)
        dense = values.toarray()
    else:
        raw = np.asarray(basis)
        shape = raw.shape
        if not np.issubdtype(raw.dtype, np.floating) or not np.all(np.isfinite(raw)):
            raise ValueError("cycle_basis must contain finite floating-point values")
        if raw.dtype.itemsize < 4:
            raise ValueError("cycle_basis storage requires float32 or float64 precision")
        dense = raw.astype(np.float64, copy=False)
        values = dense
    if len(shape) != 2 or shape != (len(incidence), cycle_rank):
        raise ValueError(
            f"cycle_basis must have shape ({len(incidence)}, {cycle_rank}); got {shape}"
        )
    if not cycle_rank:
        return
    epsilon = np.finfo(raw.dtype).eps
    residual = np.linalg.norm(incidence.T @ values, ord="fro")
    scale = max(1.0, np.linalg.norm(incidence, ord="fro")) * max(
        1.0, np.linalg.norm(dense, ord="fro")
    )
    if residual > 64.0 * epsilon * scale:
        raise ValueError("cycle_basis is not in the left nullspace: B.T @ Z != 0")
    gram = dense.T @ dense
    try:
        factor = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cycle_basis must have full column rank") from exc
    diagonal = np.diag(factor)
    if diagonal.min() <= np.sqrt(epsilon) * max(1.0, diagonal.max()):
        raise ValueError("cycle_basis must have numerically full column rank")


def left_nullspace_basis(num_nodes: int, edge_index: ArrayLike) -> NDArray[np.float32]:
    """Build sparse fundamental cycles, then cache-ready thin-QR coordinates."""
    fundamental = sparse_left_nullspace_basis(num_nodes, edge_index)
    validate_cycle_basis(num_nodes, edge_index, fundamental)
    edge_count, rank = fundamental.shape
    if rank == 0:
        return np.empty((edge_count, 0), dtype=np.float32)
    q, _ = np.linalg.qr(fundamental.toarray().astype(np.float64), mode="reduced")
    for column in range(rank):
        pivot = int(np.argmax(np.abs(q[:, column])))
        if q[pivot, column] < 0:
            q[:, column] *= -1.0
    result = np.ascontiguousarray(q, dtype=np.float32)
    validate_cycle_basis(num_nodes, edge_index, result)
    return result


def build_cycle_basis(
    num_nodes: int,
    edge_index: ArrayLike,
    *,
    backend: str = DEFAULT_BASIS_BACKEND,
) -> NDArray[np.float32]:
    """Build the selected dense cache representation of ``ker(B.T)``.

    ``thin_q`` preserves the production representation used by the fast model
    path. ``dfs_fundamental`` stores raw signed fundamental cycles selected by
    iterative DFS; the model must orthonormalize those coordinates before using
    ``Q Q.T`` as a projector.
    """
    if backend not in BASIS_BACKENDS:
        raise ValueError(f"basis backend must be one of {BASIS_BACKENDS}")
    if backend == "thin_q":
        return left_nullspace_basis(num_nodes, edge_index)
    fundamental = dfs_fundamental_cycle_basis(num_nodes, edge_index)
    validate_cycle_basis(num_nodes, edge_index, fundamental)
    return np.ascontiguousarray(fundamental.toarray(), dtype=np.float32)


__all__ = [
    "BASIS_BACKENDS",
    "DEFAULT_BASIS_BACKEND",
    "build_cycle_basis",
    "dfs_fundamental_cycle_basis",
    "incidence_and_cycle_rank",
    "left_nullspace_basis",
    "sparse_left_nullspace_basis",
    "validate_cycle_basis",
]
