"""Sparse shared/per-head conductance diffusion and polynomial filtering for V5."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_graph_index(graph_index: Tensor, num_graphs: int) -> None:
    """Keep invalid graph IDs an error without synchronizing CUDA to the host."""

    if graph_index.ndim != 1 or graph_index.dtype != torch.long:
        raise ValueError("graph_index must be a one-dimensional int64 tensor")
    if isinstance(num_graphs, bool) or not isinstance(num_graphs, int) or num_graphs < 0:
        raise ValueError("num_graphs must be a nonnegative integer")
    torch._assert_async(
        ((graph_index >= 0) & (graph_index < num_graphs)).all(),
        "graph_index must be in [0, num_graphs)",
    )


def graph_sum(
    values: Tensor, graph_index: Tensor, num_graphs: int, *, validate_index: bool = True
) -> Tensor:
    """Sum rows by graph; one graph uses a reduction, not contended scatter.

    Internal callers may skip the index check only after validating the same
    graph structure once. Empty inputs retain a differentiable zero result.
    """

    if validate_index:
        if values.ndim < 1 or graph_index.shape != (values.shape[0],):
            raise ValueError("values and graph_index must have aligned rows")
        if values.device != graph_index.device:
            raise ValueError("values and graph_index must share a device")
        _validate_graph_index(graph_index, num_graphs)
    if num_graphs == 1:
        return values.sum(dim=0, keepdim=True, dtype=values.dtype)
    return values.new_zeros((num_graphs, *values.shape[1:])).index_add(0, graph_index, values)


def graph_broadcast(
    values: Tensor, graph_index: Tensor, num_graphs: int, *, validate_index: bool = True
) -> Tensor:
    """Expand graph rows without indexed accumulation in single-graph backward."""

    if validate_index:
        if values.ndim < 1 or values.shape[0] != num_graphs:
            raise ValueError("values must have one row per graph")
        if values.device != graph_index.device:
            raise ValueError("values and graph_index must share a device")
        _validate_graph_index(graph_index, num_graphs)
    if num_graphs == 1:
        return values.expand((graph_index.numel(), *values.shape[1:]))
    return values[graph_index]


def graph_weighted_mean(
    values: Tensor,
    graph_index: Tensor,
    num_graphs: int,
    weights: Tensor | None = None,
) -> Tensor:
    """One weighted scalar mean per graph; empty graphs map to zero."""

    if values.ndim != 1 or graph_index.shape != values.shape or graph_index.dtype != torch.long:
        raise ValueError("values and graph_index must be aligned one-dimensional tensors")
    if graph_index.device != values.device:
        raise ValueError("values and graph_index must share a device")
    if isinstance(num_graphs, bool) or not isinstance(num_graphs, int) or num_graphs < 0:
        raise ValueError("num_graphs must be a nonnegative integer")
    if weights is None:
        weights = torch.ones_like(values)
    if (
        weights.shape != values.shape
        or weights.dtype != values.dtype
        or weights.device != values.device
    ):
        raise ValueError("weights must match values")
    _validate_graph_index(graph_index, num_graphs)
    numerator = graph_sum(values * weights, graph_index, num_graphs, validate_index=False)
    denominator = graph_sum(weights, graph_index, num_graphs, validate_index=False)
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny)


def weighted_degree(
    edge_weight: Tensor,
    incidence: Tensor,
    num_nodes: int,
    *,
    edge_chunk_size: int = 65536,
) -> Tensor:
    """Weighted undirected node degree without constructing an adjacency."""

    degree = edge_weight.new_zeros((num_nodes, *edge_weight.shape[1:]))
    for start in range(0, edge_weight.shape[0], edge_chunk_size):
        stop = start + edge_chunk_size
        tail, head = incidence[:, start:stop]
        values = edge_weight[start:stop]
        degree.index_add_(0, tail, values)
        degree.index_add_(0, head, values)
    return degree


class _ChunkedUndirectedPropagation(torch.autograd.Function):
    """Exact sparse propagation with recomputed edge-feature products.

    Ordinary autograd retains every chunk's gathered edge features until
    backward. This implementation instead saves only node messages, scalar
    edge weights and incidence indices: O(N * heads * width + E) storage.
    Forward and first-order backward temporaries are bounded by the edge
    chunk size; no edges or feature channels are omitted. Higher derivatives
    are supported by differentiable backward operations, but requesting
    create_graph=True necessarily retains that additional derivative graph
    and is outside the first-order saved-memory bound.
    """

    @staticmethod
    def forward(ctx, message, edge_weight, incidence, edge_chunk_size):
        ctx.save_for_backward(message, edge_weight, incidence)
        ctx.edge_chunk_size = edge_chunk_size
        ctx.set_materialize_grads(False)
        propagated = torch.zeros_like(message)
        for start in range(0, edge_weight.numel(), edge_chunk_size):
            stop = start + edge_chunk_size
            tail, head = incidence[:, start:stop]
            weight = edge_weight[start:stop, None, None]
            propagated.index_add_(0, tail, weight * message[head])
            propagated.index_add_(0, head, weight * message[tail])
        return propagated

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        message, edge_weight, incidence = ctx.saved_tensors
        need_message, need_weight = ctx.needs_input_grad[:2]
        grad_message = torch.zeros_like(message) if need_message else None
        weight_gradients = []
        for start in range(0, edge_weight.numel(), ctx.edge_chunk_size):
            stop = start + ctx.edge_chunk_size
            tail, head = incidence[:, start:stop]
            if need_message:
                weight = edge_weight[start:stop, None, None]
                grad_message.index_add_(0, tail, weight * grad_output[head])
                grad_message.index_add_(0, head, weight * grad_output[tail])
            if need_weight:
                gradient = (grad_output[tail] * message[head]).sum(dim=(1, 2))
                gradient = gradient + (grad_output[head] * message[tail]).sum(dim=(1, 2))
                weight_gradients.append(gradient)
        grad_weight = None
        if need_weight:
            grad_weight = (
                torch.cat(weight_gradients) if weight_gradients else torch.zeros_like(edge_weight)
            )
        return grad_message, grad_weight, None, None


class _ChunkedHeadPropagation(torch.autograd.Function):
    """Two receiver-specific arc weights; save O(NHD + EH), not O(EHD)."""

    @staticmethod
    def forward(ctx, message, to_tail, to_head, incidence, edge_chunk_size):
        ctx.save_for_backward(message, to_tail, to_head, incidence)
        ctx.edge_chunk_size = edge_chunk_size
        ctx.set_materialize_grads(False)
        result = torch.zeros_like(message)
        for start in range(0, incidence.shape[1], edge_chunk_size):
            stop = start + edge_chunk_size
            tail, head = incidence[:, start:stop]
            result.index_add_(0, tail, to_tail[start:stop, :, None] * message[head])
            result.index_add_(0, head, to_head[start:stop, :, None] * message[tail])
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None
        message, to_tail, to_head, incidence = ctx.saved_tensors
        need_message, need_tail, need_head = ctx.needs_input_grad[:3]
        grad_message = torch.zeros_like(message) if need_message else None
        tail_gradients, head_gradients = [], []
        for start in range(0, incidence.shape[1], ctx.edge_chunk_size):
            stop = start + ctx.edge_chunk_size
            tail, head = incidence[:, start:stop]
            if need_message:
                grad_message.index_add_(0, head, to_tail[start:stop, :, None] * grad_output[tail])
                grad_message.index_add_(0, tail, to_head[start:stop, :, None] * grad_output[head])
            if need_tail:
                tail_gradients.append((grad_output[tail] * message[head]).sum(dim=-1))
            if need_head:
                head_gradients.append((grad_output[head] * message[tail]).sum(dim=-1))
        grad_tail = (
            (torch.cat(tail_gradients) if tail_gradients else torch.zeros_like(to_tail))
            if need_tail
            else None
        )
        grad_head = (
            (torch.cat(head_gradients) if head_gradients else torch.zeros_like(to_head))
            if need_head
            else None
        )
        return grad_message, grad_tail, grad_head, None, None


def conductance_propagation_coefficients(
    relative_c: Tensor,
    incidence: Tensor,
    num_nodes: int,
    *,
    sampling_correction: Tensor | None = None,
    normalization: str = "symmetric",
    edge_chunk_size: int = 65536,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (tail-receives, head-receives, degree), independently per C head.

    Raw mean-one C is positive and need not be <=1. Only row-normalized
    receiver coefficients are probabilities; symmetric coefficients are not
    a row-stochastic attention matrix. No C heads are averaged here.
    """
    if normalization not in {"symmetric", "row"}:
        raise ValueError("normalization must be symmetric or row")
    correction = torch.ones_like(relative_c) if sampling_correction is None else sampling_correction
    if relative_c.ndim == 2 and correction.ndim == 1:
        correction = correction[:, None]
    effective = relative_c * correction
    degree = weighted_degree(effective, incidence, num_nodes, edge_chunk_size=edge_chunk_size)
    active = degree > 0
    safe = torch.where(active, degree, torch.ones_like(degree))
    tail, head = incidence
    if normalization == "row":
        return effective / safe[tail], effective / safe[head], degree
    inverse = safe.rsqrt() * active.to(degree.dtype)
    weight = effective * inverse[tail] * inverse[head]
    return weight, weight, degree


def shared_head_diffusion(
    message: Tensor,
    relative_c: Tensor,
    incidence: Tensor,
    node_graph: Tensor,
    beta: Tensor,
    *,
    sampling_correction: Tensor | None = None,
    edge_chunk_size: int = 65536,
    propagation_normalization: str = "symmetric",
    polynomial_coefficients: Tensor | None = None,
) -> Tensor:
    """Diffuse ``N x heads x width`` messages using shared or per-head C.

    For head h this is ``V_h + beta_h(G) * (P_C V_h - V_h)`` on
    nonisolated nodes. Isolates retain V exactly. ``sampling_correction`` is a
    known importance weight, not part of learned C.
    ``row`` uses receiver degree, ``symmetric`` uses both endpoint degrees.
    Optional polynomial coefficients add a2*(P²-I)+a3*(P³-I), initially zero.
    """

    if message.ndim != 3 or not message.is_floating_point():
        raise ValueError("message must be an N x heads x width floating tensor")
    if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("incidence must be a 2 x E int64 tensor")
    if relative_c.ndim not in {1, 2} or relative_c.shape[0] != incidence.shape[1]:
        raise ValueError("relative_c must have shape E or E x heads")
    if relative_c.ndim == 2 and relative_c.shape[1] != message.shape[1]:
        raise ValueError("per-head C must have one column per message head")
    if propagation_normalization not in {"symmetric", "row"}:
        raise ValueError("propagation_normalization must be symmetric or row")
    if polynomial_coefficients is not None and polynomial_coefficients.shape != (
        message.shape[1],
        2,
    ):
        raise ValueError("polynomial_coefficients must have shape heads x 2")
    if not relative_c.is_floating_point():
        raise ValueError("relative_c must be floating point")
    if node_graph.dtype != torch.long or node_graph.shape != (message.shape[0],):
        raise ValueError("node_graph must contain one graph index per node")
    if beta.ndim != 2 or beta.shape[1] != message.shape[1]:
        raise ValueError("beta must have shape num_graphs x heads")
    if any(value.device != message.device for value in (relative_c, incidence, node_graph, beta)):
        raise ValueError("all diffusion inputs must share a device")
    if sampling_correction is None:
        sampling_correction = torch.ones_like(relative_c)
    if (
        sampling_correction.shape not in {relative_c.shape, (relative_c.shape[0],)}
        or sampling_correction.dtype != relative_c.dtype
        or sampling_correction.device != message.device
    ):
        raise ValueError("sampling_correction must match relative_c")
    if (
        isinstance(edge_chunk_size, bool)
        or not isinstance(edge_chunk_size, int)
        or edge_chunk_size < 1
    ):
        raise ValueError("edge_chunk_size must be a positive integer")

    # FP32 island for the conductance geometry.  Dense encoder/W/FFN kernels
    # may run under BF16 autocast, but centering, weighted degree and diffusion
    # do not. Cache structure is validated once before training; CUDA boolean
    # reductions here would otherwise serialize every layer and every batch.
    compute_dtype = (
        torch.float32 if message.dtype in {torch.float16, torch.bfloat16} else message.dtype
    )
    message_compute = message.to(compute_dtype)
    if (
        relative_c.ndim == 2
        or propagation_normalization != "symmetric"
        or polynomial_coefficients is not None
    ):
        to_tail, to_head, degree = conductance_propagation_coefficients(
            relative_c.to(compute_dtype),
            incidence,
            message.shape[0],
            sampling_correction=sampling_correction.to(compute_dtype),
            normalization=propagation_normalization,
            edge_chunk_size=edge_chunk_size,
        )
        if to_tail.ndim == 1:
            to_tail = to_tail[:, None].expand(-1, message.shape[1])
            to_head = to_head[:, None].expand(-1, message.shape[1])
            degree = degree[:, None].expand(-1, message.shape[1])
        active = (degree > 0).unsqueeze(-1)
        propagated = _ChunkedHeadPropagation.apply(
            message_compute, to_tail, to_head, incidence, edge_chunk_size
        )
        node_beta = graph_broadcast(beta.to(compute_dtype), node_graph, beta.shape[0]).unsqueeze(-1)
        output = message_compute + node_beta * (propagated - active * message_compute)
        if polynomial_coefficients is not None:
            # P acts as identity on isolates. Coefficients represent
            # (1-beta-a2-a3)I + beta P + a2 P² + a3 P³.
            p1 = propagated + (~active) * message_compute
            p2 = (
                _ChunkedHeadPropagation.apply(p1, to_tail, to_head, incidence, edge_chunk_size)
                + (~active) * p1
            )
            p3 = (
                _ChunkedHeadPropagation.apply(p2, to_tail, to_head, incidence, edge_chunk_size)
                + (~active) * p2
            )
            coefficients = polynomial_coefficients.to(compute_dtype)
            output = output + coefficients[None, :, 0, None] * (p2 - message_compute)
            output = output + coefficients[None, :, 1, None] * (p3 - message_compute)
        return output.to(message.dtype)
    effective = relative_c.to(compute_dtype) * sampling_correction.to(compute_dtype)
    degree = weighted_degree(
        effective, incidence, message.shape[0], edge_chunk_size=edge_chunk_size
    )
    active = degree > 0
    # Do not evaluate rsqrt(0) on isolates: its backward can otherwise create
    # 0 * inf even though the forward where() selected the zero branch.
    safe_degree = torch.where(active, degree, torch.ones_like(degree))
    inverse = safe_degree.rsqrt() * active.to(compute_dtype)
    tail, head = incidence
    weight = effective * inverse[tail] * inverse[head]
    propagated = _ChunkedUndirectedPropagation.apply(
        message_compute, weight, incidence, edge_chunk_size
    )
    node_beta = graph_broadcast(beta.to(compute_dtype), node_graph, beta.shape[0]).unsqueeze(-1)
    output = message_compute + node_beta * (propagated - active[:, None, None] * message_compute)
    return output.to(message.dtype)
