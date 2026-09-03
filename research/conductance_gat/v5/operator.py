"""Sparse shared-conductance, multi-head diffusion for V5."""

from __future__ import annotations

import torch
from torch import Tensor


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
    numerator = values.new_zeros(num_graphs).index_add(0, graph_index, values * weights)
    denominator = values.new_zeros(num_graphs).index_add(0, graph_index, weights)
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny)


def weighted_degree(
    edge_weight: Tensor,
    incidence: Tensor,
    num_nodes: int,
    *,
    edge_chunk_size: int = 65536,
) -> Tensor:
    """Weighted undirected node degree without constructing an adjacency."""

    degree = edge_weight.new_zeros(num_nodes)
    for start in range(0, edge_weight.numel(), edge_chunk_size):
        stop = start + edge_chunk_size
        tail, head = incidence[:, start:stop]
        values = edge_weight[start:stop]
        degree.index_add_(0, tail, values)
        degree.index_add_(0, head, values)
    return degree


def shared_head_diffusion(
    message: Tensor,
    relative_c: Tensor,
    incidence: Tensor,
    node_graph: Tensor,
    beta: Tensor,
    *,
    sampling_correction: Tensor | None = None,
    edge_chunk_size: int = 65536,
) -> Tensor:
    """Diffuse ``N x heads x width`` messages with one C shared by all heads.

    For head h this is ``V_h + beta_h(G) * (P_C V_h - V_h)`` on
    nonisolated nodes. Isolates retain V exactly. ``sampling_correction`` is a
    known importance weight, not part of learned C.
    """

    if message.ndim != 3 or not message.is_floating_point():
        raise ValueError("message must be an N x heads x width floating tensor")
    if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("incidence must be a 2 x E int64 tensor")
    if relative_c.ndim != 1 or relative_c.shape[0] != incidence.shape[1]:
        raise ValueError("relative_c must contain one value per physical edge")
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
        sampling_correction.shape != relative_c.shape
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
    effective = relative_c.to(compute_dtype) * sampling_correction.to(compute_dtype)
    degree = weighted_degree(
        effective, incidence, message.shape[0], edge_chunk_size=edge_chunk_size
    )
    active = degree > 0
    inverse = torch.where(active, degree.rsqrt(), torch.zeros_like(degree))
    propagated = torch.zeros_like(message_compute)
    for start in range(0, effective.numel(), edge_chunk_size):
        stop = start + edge_chunk_size
        tail, head = incidence[:, start:stop]
        weight = effective[start:stop] * inverse[tail] * inverse[head]
        propagated.index_add_(0, tail, weight[:, None, None] * message_compute[head])
        propagated.index_add_(0, head, weight[:, None, None] * message_compute[tail])
    node_beta = beta.to(compute_dtype)[node_graph].unsqueeze(-1)
    output = message_compute + node_beta * (propagated - active[:, None, None] * message_compute)
    return output.to(message.dtype)
