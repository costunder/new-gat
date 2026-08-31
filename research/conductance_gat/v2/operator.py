"""Exact first-order row-normalized propagation with bounded edge workspace.

No dense incidence/Laplacian matrix or full E-by-width activation is created.
Backward differentiates both weighted neighbor sums and C-dependent degrees;
double backward is intentionally unsupported. CUDA index_add_ can be subject to
floating-point ordering differences: chunking is not a bitwise determinism claim.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


class _ChunkedNormalizedPropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: Tensor, c: Tensor, incidence: Tensor, chunk_size: int) -> Tensor:
        degree = state.new_zeros(state.shape[0])
        neighbor_sum = torch.zeros_like(state)
        for start in range(0, c.numel(), chunk_size):
            stop = start + chunk_size
            tail, head = incidence[:, start:stop]
            weights = c[start:stop]
            degree.index_add_(0, tail, weights)
            degree.index_add_(0, head, weights)
            neighbor_sum.index_add_(0, tail, weights[:, None] * state[head])
            neighbor_sum.index_add_(0, head, weights[:, None] * state[tail])
        if not bool(torch.isfinite(degree).all()):
            raise FloatingPointError("Conductance degree overflow; C is not silently rescaled")
        nonisolated = degree > 0
        safe_degree = torch.where(nonisolated, degree, torch.ones_like(degree))
        mean = neighbor_sum / safe_degree[:, None]
        result = torch.where(nonisolated[:, None], 0.05 * state + 0.95 * mean, state)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite direct-conductance propagation")
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(state, c, incidence, safe_degree, mean, nonisolated)
        # Using neighbor means above is algebraically H - .95 D_C^dagger B.T C B H.
        # An isolate's mean is zero and it has no edge contributions in backward.
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        state, c, incidence, safe_degree, mean, nonisolated = ctx.saved_tensors
        grad_state = None
        grad_c = None
        if ctx.needs_input_grad[0]:
            grad_state = torch.where(
                nonisolated[:, None], 0.05 * grad_output, grad_output
            )
        if ctx.needs_input_grad[1]:
            grad_c = torch.empty_like(c)
        for start in range(0, c.numel(), ctx.chunk_size):
            stop = start + ctx.chunk_size
            tail, head = incidence[:, start:stop]
            weights = c[start:stop]
            scaled_tail = grad_output[tail] / safe_degree[tail, None]
            scaled_head = grad_output[head] / safe_degree[head, None]
            if grad_state is not None:
                grad_state.index_add_(0, head, 0.95 * weights[:, None] * scaled_tail)
                grad_state.index_add_(0, tail, 0.95 * weights[:, None] * scaled_head)
            if grad_c is not None:
                # d mu_i/d c_ij = (H_j - mu_i)/d_i: the -mu_i term is
                # essential and would disappear if the denominator were detached.
                grad_c[start:stop] = 0.95 * (
                    (scaled_tail * (state[head] - mean[tail])).sum(dim=1)
                    + (scaled_head * (state[tail] - mean[head])).sum(dim=1)
                )
        return grad_state, grad_c, None, None


def chunked_normalized_propagation(
    state: Tensor, c: Tensor, incidence: Tensor, *, edge_chunk_size: int = 65536
) -> Tensor:
    """Compute H - .95 D_C^dagger B.T C B H; all inputs describe one graph.

The kernel accepts either edge orientation, with exactly one C entry per supplied
physical edge. The graph-bound model separately enforces canonical identity/order.
FP64 is preserved for numerical verification; the research model uses FP32.
"""
    if state.ndim != 2 or state.dtype not in {torch.float32, torch.float64}:
        raise ValueError("state must be a two-dimensional float32/float64 tensor")
    if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("incidence must be a 2 x E int64 tensor")
    if c.ndim != 1 or c.shape[0] != incidence.shape[1] or c.dtype != state.dtype:
        raise ValueError("C must have one same-dtype scalar per incidence edge")
    if state.device != c.device or state.device != incidence.device:
        raise ValueError("state, C and incidence must share a device")
    if isinstance(edge_chunk_size, bool) or not isinstance(edge_chunk_size, int):
        raise ValueError("edge_chunk_size must be a positive integer")
    if edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be a positive integer")
    if incidence.numel() and (
        bool((incidence < 0).any()) or bool((incidence >= state.shape[0]).any())
    ):
        raise ValueError("incidence endpoint is outside the graph")
    if incidence.shape[1] and bool((incidence[0] == incidence[1]).any()):
        raise ValueError("physical incidence edges must not contain self loops")
    if not bool(torch.isfinite(c.detach()).all()) or not bool((c.detach() > 0).all()):
        raise FloatingPointError("C must remain finite and strictly positive; no clipping is used")
    if not bool(torch.isfinite(state.detach()).all()):
        raise FloatingPointError("Nonfinite input to direct-conductance propagation")
    return _ChunkedNormalizedPropagation.apply(state, c, incidence, edge_chunk_size)
