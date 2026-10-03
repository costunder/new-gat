"""Vectorized fixed local incidence, signed relations and full-node recovery.

Feature columns are independent fields, not independent graph/model replicas.
The two CG solves use every local and feature together. The iteration ceiling
is a numerical convergence safeguard tied to the largest complete local.
Failure raises an error; no smaller graph, CPU fallback or approximate output
is silently returned. Partial transfer diagnostics never call a partial inverse.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from numbers import Integral

import torch
from torch import Tensor

from .topology import CORRESPONDENCE_KINDS, LocalTopology


def _values(top: LocalTopology, value: Tensor, count: int, name: str) -> None:
    if not isinstance(top.local_node_global, Tensor):
        raise TypeError("call topology.to(device) before applying tensor operators")
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] != count
        or value.shape[1] < 1
    ):
        raise ValueError(f"{name} must have shape [{count},features] with positive features")
    if not value.is_floating_point() or value.device != top.local_node_global.device:
        raise ValueError(f"{name} must be floating point on the topology device")


def incidence(top: LocalTopology, h: Tensor) -> Tensor:
    """Local B_v h_v, h is flattened local-node [NL,F], not physical [N,F]."""
    _values(top, h, top.num_local_nodes, "h")
    return h.index_select(0, top.local_edge_nodes[1]) - h.index_select(0, top.local_edge_nodes[0])


def incidence_transpose(top: LocalTopology, q: Tensor) -> Tensor:
    """Disjoint local divergences B_v.T q_v, shape [NL,F]."""
    _values(top, q, top.num_local_edges, "q")
    result = q.new_zeros((top.num_local_nodes, q.shape[1]))
    result = result.index_add(0, top.local_edge_nodes[0], -q)
    return result.index_add(0, top.local_edge_nodes[1], q)


def local_weight(top: LocalTopology, mode: str, *, dtype: torch.dtype = torch.float64) -> Tensor:
    if not isinstance(top.local_degree, Tensor):
        raise TypeError("call topology.to(device) before applying tensor operators")
    if mode == "unit":
        return torch.ones(top.num_local_edges, dtype=dtype, device=top.local_degree.device)
    if mode != "local_degree":
        raise ValueError("weight mode must be unit or local_degree")
    denominator = top.local_degree[top.local_edge_nodes].sum(0).to(dtype)
    return 2.0 / denominator


def aggregate_squares(
    top: LocalTopology, values: Tensor, *, space: str = "edge", weights: Tensor | None = None
) -> Tensor:
    """Per-local sums of squares; optional weights multiply squares once."""
    if space not in ("edge", "node"):
        raise ValueError("space must be edge or node")
    count = top.num_local_edges if space == "edge" else top.num_local_nodes
    _values(top, values, count, "values")
    squared = values.square()
    if weights is not None:
        if (
            weights.shape != (count,)
            or weights.device != values.device
            or weights.dtype != values.dtype
        ):
            raise ValueError("weights must match flattened values count/device/dtype")
        squared = squared * weights[:, None]
    indices = top.local_edge_center if space == "edge" else top.local_node_center
    return values.new_zeros((top.n, values.shape[1])).index_add(0, indices, squared)


def _correspondence_sum(
    top: LocalTopology,
    left: Tensor,
    kind: str,
    *,
    right: Tensor | None = None,
    relation_batch: int | None = None,
) -> Tensor:
    """Exact pair chunks bound temporary overlap gathers, preserving all pairs."""
    if relation_batch is not None and (
        isinstance(relation_batch, bool)
        or not isinstance(relation_batch, Integral)
        or relation_batch < 1
    ):
        raise ValueError("relation_batch must be a positive directed-pair count or None")
    chunk = top.num_pairs if relation_batch is None else int(relation_batch)
    output = left.new_zeros((top.num_pairs, left.shape[1]))
    if top.num_pairs == 0:
        return output
    offsets = top.correspondence_offsets[CORRESPONDENCE_KINDS.index(kind)]
    left_indices = getattr(top, kind + "_left")
    pairs = getattr(top, kind + "_pair")
    right_indices = getattr(top, kind + "_right") if right is not None else None
    for start in range(0, top.num_pairs, chunk):
        stop = min(start + chunk, top.num_pairs)
        begin, end = offsets[start], offsets[stop]
        chosen = left.index_select(0, left_indices[begin:end])
        value = (
            chosen.square()
            if right is None
            else chosen * right.index_select(0, right_indices[begin:end])
        )
        output.index_add_(0, pairs[begin:end], value)
    return output


def relation_terms(
    top: LocalTopology,
    q_left: Tensor,
    q_right: Tensor | None = None,
    *,
    relation_batch: int | None = None,
) -> dict[str, Tensor]:
    """Signed flow bilinears for every directed adjacent center pair.

    shared_nodes equals q_v.T (B_v_global B_u_global.T) q_u.
    Subtracting twice shared_edges removes exactly the same-physical-edge terms.
    The remaining support is distinct edges sharing an original endpoint.
    These signed relationships do not assert a positive semidefinite energy.
    """
    q_right = q_left if q_right is None else q_right
    _values(top, q_left, top.num_local_edges, "q_left")
    _values(top, q_right, top.num_local_edges, "q_right")
    if q_left.shape != q_right.shape or q_left.dtype != q_right.dtype:
        raise ValueError("left/right flow fields must share feature coordinates and dtype")
    shared_edges = _correspondence_sum(
        top, q_left, "shared_edge", right=q_right, relation_batch=relation_batch
    )
    left_d = incidence_transpose(top, q_left)
    right_d = left_d if q_right is q_left else incidence_transpose(top, q_right)
    shared_nodes = _correspondence_sum(
        top, left_d, "shared_node", right=right_d, relation_batch=relation_batch
    )
    return {
        "shared_edges": shared_edges,
        "shared_nodes": shared_nodes,
        "distinct_edges": shared_nodes - 2 * shared_edges,
    }


def transfer_bookkeeping(
    top: LocalTopology, q: Tensor, *, relation_batch: int | None = None
) -> dict[str, Tensor]:
    """Exact retained/omitted sender divergence and edge-flow norm partitions.

    Receiver membership is S_u. Boundary edges have exactly one endpoint in S_u;
    omitted edges have both endpoints outside. This is observation bookkeeping,
    not inference of unavailable flow from the retained nodes.
    """
    _values(top, q, top.num_local_edges, "q")
    d = incidence_transpose(top, q)
    sender = top.pair_centers[0]

    def squares(values, kind):
        return _correspondence_sum(top, values, kind, relation_batch=relation_batch)

    def counts(pairs):
        return torch.bincount(pairs, minlength=top.num_pairs)

    return {
        "sender_node_count": (top.node_offsets[1:] - top.node_offsets[:-1])[sender],
        "retained_node_count": counts(top.shared_node_pair),
        "omitted_node_count": counts(top.omitted_node_pair),
        "sender_edge_count": (top.edge_offsets[1:] - top.edge_offsets[:-1])[sender],
        "retained_edge_count": counts(top.shared_edge_pair),
        "boundary_edge_count": counts(top.boundary_edge_pair),
        "omitted_edge_count": counts(top.omitted_edge_pair),
        "sender_divergence_norm_sq": aggregate_squares(top, d, space="node")[sender],
        "retained_divergence_norm_sq": squares(d, "shared_node"),
        "omitted_divergence_norm_sq": squares(d, "omitted_node"),
        "sender_flow_norm_sq": aggregate_squares(top, q)[sender],
        "retained_flow_norm_sq": squares(q, "shared_edge"),
        "boundary_flow_norm_sq": squares(q, "boundary_edge"),
        "omitted_flow_norm_sq": squares(q, "omitted_edge"),
    }


def _node_sum(top: LocalTopology, values: Tensor) -> Tensor:
    return values.new_zeros((top.n, values.shape[1])).index_add(0, top.local_node_center, values)


def _project(top: LocalTopology, values: Tensor) -> Tensor:
    # Every induced one-hop local is connected, including one-node isolates.
    counts = (top.node_offsets[1:] - top.node_offsets[:-1]).to(values.dtype)
    means = _node_sum(top, values) / counts[:, None]
    return values - means.index_select(0, top.local_node_center)


@torch.no_grad()
def solve_local_laplacian(
    top: LocalTopology,
    rhs: Tensor,
    weights: Tensor | None = None,
    *,
    tol: float = 1e-9,
    iteration_multiplier: int = 4,
    progress: Callable[[int, int, float], None] | None = None,
) -> tuple[Tensor, dict]:
    """Mean-zero L_C pseudoinverse via vectorized projected Jacobi CG.

    RHS must have zero local sum, as B_v.T q_v does. Max iterations equal the
    multiplier times the largest full local node count. All local/feature systems
    are masked independently; no local or feature axis is traversed in Python.
    Actual final residual is recomputed, not inferred from the CG recurrence.
    """
    _values(top, rhs, top.num_local_nodes, "rhs")
    if rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError("CG requires float32 or float64")
    if not math.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("tol must be finite and in (0,1)")
    if (
        isinstance(iteration_multiplier, bool)
        or not isinstance(iteration_multiplier, Integral)
        or iteration_multiplier < 1
    ):
        raise ValueError("iteration_multiplier must be a positive integer")
    if weights is None:
        weights = local_weight(top, "unit", dtype=rhs.dtype)
    if (
        weights.shape != (top.num_local_edges,)
        or weights.dtype != rhs.dtype
        or weights.device != rhs.device
    ):
        raise ValueError("CG weights must match topology/device/dtype")
    if (
        not bool(torch.isfinite(rhs).all())
        or not bool(torch.isfinite(weights).all())
        or not bool((weights > 0).all())
    ):
        raise ValueError("CG requires finite RHS and finite positive local weights")
    rhs_norm = _node_sum(top, rhs.square()).sqrt()
    exact_zero = _node_sum(top, rhs.abs()) == 0
    if not bool(((rhs_norm > 0) | exact_zero).all()):
        raise FloatingPointError("CG RHS norm underflow; use representable precision/scales")
    counts = (top.node_offsets[1:] - top.node_offsets[:-1]).to(rhs.dtype)
    balance = _node_sum(top, rhs).abs() / (
        counts.sqrt()[:, None] * torch.where(rhs_norm > 0, rhs_norm, torch.ones_like(rhs_norm))
    )
    if not bool((balance <= max(tol, 1e-12)).all()):
        raise ValueError("CG RHS is not in the local Laplacian range (nonzero local sum)")
    diagonal = rhs.new_zeros(top.num_local_nodes)
    diagonal = diagonal.index_add(0, top.local_edge_nodes[0], weights)
    diagonal = diagonal.index_add(0, top.local_edge_nodes[1], weights)
    inverse_diagonal = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal)).reciprocal()
    x = torch.zeros_like(rhs)
    r = _project(top, rhs)
    z = _project(top, r * inverse_diagonal[:, None])
    p = z.clone()
    rz = _node_sum(top, r * z)
    active = ~exact_zero
    iterations = torch.zeros_like(active, dtype=torch.long)
    maximum = int(iteration_multiplier) * top.max_local_nodes
    performed = 0
    safe_norm = torch.where(rhs_norm > 0, rhs_norm, torch.ones_like(rhs_norm))
    started = last_progress = time.monotonic()
    for iteration in range(1, maximum + 1):
        if not bool(active.any()):
            break
        current_time = time.monotonic()
        if progress is not None and current_time - last_progress >= 30:
            progress(iteration - 1, maximum, current_time - started)
            last_progress = current_time
        difference = incidence(top, p)
        ap = incidence_transpose(top, weights[:, None] * difference)
        denominator = aggregate_squares(top, difference, weights=weights)
        step = torch.where(
            active,
            rz / torch.where(denominator > 0, denominator, torch.ones_like(denominator)),
            torch.zeros_like(rz),
        )
        x = x + step[top.local_node_center] * p
        r = _project(top, r - step[top.local_node_center] * ap)
        relative = _node_sum(top, r.square()).sqrt() / safe_norm
        new_active = active & (relative > tol)
        iterations = torch.where(
            active & ~new_active, torch.full_like(iterations, iteration), iterations
        )
        z = _project(top, r * inverse_diagonal[:, None])
        next_rz = _node_sum(top, r * z)
        beta = torch.where(
            new_active,
            next_rz / torch.where(rz != 0, rz, torch.ones_like(rz)),
            torch.zeros_like(rz),
        )
        p = (z + beta[top.local_node_center] * p) * new_active[top.local_node_center]
        rz, active, performed = next_rz, new_active, iteration
    x = _project(top, x)
    residual = rhs - incidence_transpose(top, weights[:, None] * incidence(top, x))
    relative = _node_sum(top, residual.square()).sqrt() / safe_norm
    converged = relative <= tol
    if (
        not bool(torch.isfinite(x).all())
        or not bool(torch.isfinite(relative).all())
        or not bool(converged.all())
    ):
        worst = float(relative.max().item())
        raise RuntimeError(
            f"local CG not converged: iterations={performed}/{maximum}, "
            f"maximum relative residual={worst:.6g}, tol={tol}"
        )
    return x, {
        "iterations": iterations,
        "relative_residual": relative,
        "converged": converged,
        "global_iterations": performed,
        "max_iterations": maximum,
    }


@torch.no_grad()
def recover_flows(
    top: LocalTopology,
    q: Tensor,
    weights: Tensor,
    *,
    tol: float = 1e-9,
    iteration_multiplier: int = 4,
    progress: Callable[[str, int, int, float], None] | None = None,
) -> dict[str, Tensor | dict]:
    """Euclidean cut/cycle split and separate known-C full-divergence recovery."""
    d = incidence_transpose(top, q)
    cut_potential, cut_solver = solve_local_laplacian(
        top,
        d,
        tol=tol,
        iteration_multiplier=iteration_multiplier,
        progress=None if progress is None else lambda i, m, elapsed: progress("cut", i, m, elapsed),
    )
    weighted_potential, reconstruction_solver = solve_local_laplacian(
        top,
        d,
        weights,
        tol=tol,
        iteration_multiplier=iteration_multiplier,
        progress=None
        if progress is None
        else lambda i, m, elapsed: progress("reconstruction", i, m, elapsed),
    )
    q_cut = incidence(top, cut_potential)
    return {
        "q_cut": q_cut,
        "q_cycle": q - q_cut,
        "q_recon": weights[:, None] * incidence(top, weighted_potential),
        "cut_solver": cut_solver,
        "reconstruction_solver": reconstruction_solver,
    }


__all__ = [
    "incidence",
    "incidence_transpose",
    "local_weight",
    "aggregate_squares",
    "relation_terms",
    "transfer_bookkeeping",
    "solve_local_laplacian",
    "recover_flows",
]
