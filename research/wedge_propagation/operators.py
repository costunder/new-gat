"""Independent, fixed incidence and unordered-wedge operators for phases 0/1.

An edge column ``(u, v)`` gives the incidence row ``-e_u + e_v``.
A wedge column ``(i, j, k)`` gives ``e_i - 2 e_j + e_k``; its endpoints
are unordered. Kernels act on every trailing feature dimension together,
including independent feature realizations and disjoint-union graph batches.

Indices must be ``torch.long`` and describe a simple undirected topology.
Full topology validation and static degree terms are cached by tensor identity,
node count, and in-place version. A GPU topology is copied to CPU once during
validation, before the arithmetic kernel; subsequent applications do not
synchronize or copy features to CPU. Do not mutate indices through ``.data``.
Dense constructors and ``dense_R`` are reference implementations, not the
execution path for fixed measurements.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "build_wedges",
    "dense_R",
    "dense_incidence",
    "dense_wedge",
    "fixed_wedge_apply",
    "fixed_wedge_fast_apply",
    "incidence_apply",
    "incidence_transpose",
    "laplacian_apply",
    "wedge_apply",
    "wedge_transpose",
]


@dataclass
class _Topology:
    reference: weakref.ReferenceType[Tensor]
    version: int | None
    n: int
    degrees: Tensor | None = None
    correction: Tensor | None = None


_TOPOLOGIES: dict[tuple[str, int], _Topology] = {}


def _node_count(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be a nonnegative integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    return n


def _validate_indices(indices: Tensor, n: int, kind: Literal["edges", "wedges"]) -> _Topology:
    n = _node_count(n)
    rows = 2 if kind == "edges" else 3
    if not isinstance(indices, Tensor):
        raise TypeError(f"{kind} must be a torch.Tensor")
    if indices.layout != torch.strided or indices.ndim != 2 or indices.shape[0] != rows:
        raise ValueError(f"{kind} must have shape [{rows}, count] and strided layout")
    if indices.dtype != torch.long:
        raise TypeError(f"{kind} must have dtype torch.long")

    # Inference tensors do not have a mutation version, so do not reuse a cache
    # for them. Ordinary static index tensors use a synchronization-free lookup.
    version = None if torch.is_inference(indices) else indices._version
    key = (kind, id(indices))
    cached = _TOPOLOGIES.get(key)
    if (
        version is not None
        and cached is not None
        and cached.reference() is indices
        and cached.version == version
        and cached.n == n
    ):
        return cached

    columns = indices.detach().to(device="cpu").t().tolist()
    seen: set[tuple[int, ...]] = set()
    for column in columns:
        if any(node < 0 or node >= n for node in column):
            raise ValueError(f"{kind} contain a node outside [0, n)")
        if len(set(column)) != rows:
            if kind == "edges":
                raise ValueError("edges must not contain self loops")
            raise ValueError("a wedge must contain three distinct nodes")
        if kind == "edges":
            canonical = tuple(sorted(column))
        else:
            i, j, k = column
            canonical = (j, min(i, k), max(i, k))
        if canonical in seen:
            raise ValueError(f"{kind} contain an undirected duplicate")
        seen.add(canonical)

    def discard(reference: weakref.ReferenceType[Tensor]) -> None:
        current = _TOPOLOGIES.get(key)
        if current is not None and current.reference is reference:
            del _TOPOLOGIES[key]

    topology = _Topology(weakref.ref(indices, discard), version, n)
    _TOPOLOGIES[key] = topology
    return topology


def _validate_values(values: Tensor, indices: Tensor, count: int, name: str) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.layout != torch.strided or values.ndim < 1 or values.shape[0] != count:
        raise ValueError(f"{name} must have shape [{count}, ...] and strided layout")
    if not values.is_floating_point():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if values.device != indices.device:
        raise ValueError(f"{name} and indices must be on the same device")


def _matrix_options(dtype: torch.dtype, device: torch.device | str | None, indices: Tensor):
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dense reference matrices require a real floating-point dtype")
    return {"dtype": dtype, "device": indices.device if device is None else device}


def dense_incidence(
    edges: Tensor,
    n: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the oriented reference matrix B with shape [E, N]."""
    _validate_indices(edges, n, "edges")
    matrix = torch.zeros((edges.shape[1], n), **_matrix_options(dtype, device, edges))
    index = edges.to(device=matrix.device)
    rows = torch.arange(edges.shape[1], device=matrix.device)
    matrix[rows, index[0]] = -1
    matrix[rows, index[1]] = 1
    return matrix


def build_wedges(edges: Tensor, n: int) -> Tensor:
    """Enumerate all unordered neighbor pairs as CPU long [3, P] indices.

    Rows are sorted by center, then by the two endpoint IDs. Construction is
    tensorized and never caps degree or path count. This static CPU preprocessing
    is done once per graph, rather than inside an operator application.
    """
    _validate_indices(edges, n, "edges")
    edges_cpu = edges.detach().to(device="cpu")
    centers = torch.cat((edges_cpu[0], edges_cpu[1]))
    neighbors = torch.cat((edges_cpu[1], edges_cpu[0]))
    degree = torch.bincount(centers, minlength=n)
    counts = degree * (degree - 1) // 2
    p = int(counts.sum())
    if p == 0:
        return torch.empty((3, 0), dtype=torch.long)

    # Stable lexicographic order gives sorted adjacency lists without a padded
    # [N, max_degree] representation or a Python loop over graph centers.
    order = torch.argsort(neighbors, stable=True)
    order = order[torch.argsort(centers[order], stable=True)]
    neighbors = neighbors[order]
    adjacency_start = degree.cumsum(0) - degree
    path_start = counts.cumsum(0) - counts
    center = torch.repeat_interleave(torch.arange(n), counts)
    ordinal = torch.arange(p) - path_start[center]
    d = degree[center]

    # Invert the upper-triangle row starts r*(2*d-r-1)/2. Integer comparisons
    # correct rounding at exact square roots before the final gather.
    discriminant = (2 * d - 1).to(torch.float64).square() - 8 * ordinal
    first = torch.floor(((2 * d - 1) - discriminant.sqrt()) / 2).to(torch.long)
    first = first.clamp_min(0)
    start = first * (2 * d - first - 1) // 2
    first = torch.where(start > ordinal, first - 1, first)
    next_start = (first + 1) * (2 * d - first - 2) // 2
    first = torch.where(next_start <= ordinal, first + 1, first)
    start = first * (2 * d - first - 1) // 2
    second = first + 1 + ordinal - start
    offset = adjacency_start[center]
    return torch.stack((neighbors[offset + first], center, neighbors[offset + second]))


def dense_wedge(
    wedges: Tensor,
    n: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the reference second-difference matrix A with shape [P, N]."""
    _validate_indices(wedges, n, "wedges")
    matrix = torch.zeros((wedges.shape[1], n), **_matrix_options(dtype, device, wedges))
    index = wedges.to(device=matrix.device)
    rows = torch.arange(wedges.shape[1], device=matrix.device)
    matrix[rows, index[0]] = 1
    matrix[rows, index[1]] = -2
    matrix[rows, index[2]] = 1
    return matrix


def dense_R(
    edges: Tensor,
    wedges: Tensor,
    n: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return [P, E] R satisfying A=RB, including orientation correction.

    This CPU-topology reference checks that both edges of each supplied wedge
    exist. R is signed and is generally not an ordinary line-graph incidence.
    """
    _validate_indices(edges, n, "edges")
    _validate_indices(wedges, n, "wedges")
    edge_columns = edges.detach().to(device="cpu").t().tolist()
    path_columns = wedges.detach().to(device="cpu").t().tolist()
    lookup = {tuple(sorted(edge)): (index, edge[0]) for index, edge in enumerate(edge_columns)}
    selected: list[tuple[int, int]] = []
    signs: list[tuple[int, int]] = []
    for i, j, k in path_columns:
        left = lookup.get(tuple(sorted((i, j))))
        right = lookup.get(tuple(sorted((j, k))))
        if left is None or right is None:
            raise ValueError(f"wedge ({i}, {j}, {k}) contains an edge absent from edges")
        selected.append((left[0], right[0]))
        signs.append((1 if left[1] == j else -1, 1 if right[1] == j else -1))
    matrix = torch.zeros(
        (wedges.shape[1], edges.shape[1]), **_matrix_options(dtype, device, edges)
    )
    if selected:
        columns = torch.tensor(selected, dtype=torch.long, device=matrix.device)
        rows = torch.arange(wedges.shape[1], device=matrix.device).unsqueeze(1)
        matrix[rows, columns] = torch.tensor(signs, dtype=dtype, device=matrix.device)
    return matrix


def incidence_apply(edges: Tensor, x: Tensor) -> Tensor:
    """Compute Bx for all feature axes, preserving edge orientation."""
    if not isinstance(x, Tensor) or x.ndim < 1:
        raise ValueError("x must have shape [N, ...]")
    _validate_indices(edges, x.shape[0], "edges")
    _validate_values(x, edges, x.shape[0], "x")
    return x.index_select(0, edges[1]) - x.index_select(0, edges[0])


def incidence_transpose(edges: Tensor, flows: Tensor, n: int) -> Tensor:
    """Compute B.T flows, including the differentiable E=0 zero operator."""
    _validate_indices(edges, n, "edges")
    _validate_values(flows, edges, edges.shape[1], "flows")
    result = flows.new_zeros((n, *flows.shape[1:]))
    result = result.index_add(0, edges[1], flows)
    return result.index_add(0, edges[0], -flows)


def laplacian_apply(edges: Tensor, x: Tensor) -> Tensor:
    """Compute Lx=B.T Bx without a dense or sparse matrix materialization."""
    return incidence_transpose(edges, incidence_apply(edges, x), x.shape[0])


def wedge_apply(wedges: Tensor, x: Tensor) -> Tensor:
    """Compute Ax for all unordered wedges and all trailing feature axes."""
    if not isinstance(x, Tensor) or x.ndim < 1:
        raise ValueError("x must have shape [N, ...]")
    _validate_indices(wedges, x.shape[0], "wedges")
    _validate_values(x, wedges, x.shape[0], "x")
    return (
        x.index_select(0, wedges[0])
        - 2 * x.index_select(0, wedges[1])
        + x.index_select(0, wedges[2])
    )


def wedge_transpose(wedges: Tensor, q: Tensor, n: int) -> Tensor:
    """Compute A.T q with a single batched scatter for each coefficient."""
    _validate_indices(wedges, n, "wedges")
    _validate_values(q, wedges, wedges.shape[1], "q")
    result = q.new_zeros((n, *q.shape[1:]))
    result = result.index_add(0, wedges[0], q)
    result = result.index_add(0, wedges[1], -2 * q)
    return result.index_add(0, wedges[2], q)


def fixed_wedge_apply(wedges: Tensor, x: Tensor) -> Tensor:
    """Compute Qx=A.T Ax by explicit full-wedge gather/scatter."""
    return wedge_transpose(wedges, wedge_apply(wedges, x), x.shape[0])


def fixed_wedge_fast_apply(edges: Tensor, x: Tensor) -> Tensor:
    """Compute exactly Qx in O(E times feature-count) using the identity.

    Q=L^2+B.T diag(d_u+d_v-4) B includes every unordered wedge. The edge
    correction may be negative; it must not be clipped or interpreted alone
    as a PSD metric. Static degree/correction tensors are cached per topology.
    """
    if not isinstance(x, Tensor) or x.ndim < 1:
        raise ValueError("x must have shape [N, ...]")
    n = x.shape[0]
    topology = _validate_indices(edges, n, "edges")
    _validate_values(x, edges, n, "x")
    if topology.degrees is None:
        topology.degrees = torch.bincount(edges.reshape(-1), minlength=n)
        topology.correction = topology.degrees[edges[0]] + topology.degrees[edges[1]] - 4
    assert topology.correction is not None
    differences = incidence_apply(edges, x)
    first = incidence_transpose(edges, differences, n)
    squared = laplacian_apply(edges, first)
    coefficient = topology.correction.reshape((-1,) + (1,) * (x.ndim - 1))
    return squared + incidence_transpose(edges, differences * coefficient, n)
