"""Exact chunked symmetric normalized diffusion, including C-dependent degrees.

The custom backward saves O(nd+m) values and uses O(chunk*d) edge workspace.
It is first-order only; no dense B, C or Laplacian is materialized. The frozen-C
linear map is symmetric, but adaptive C(H) makes the complete layer nonlinear.
Symmetric normalization does not generally preserve the constant node vector.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


class _SymmetricPropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, c, incidence, alpha, chunk_size):
        degree = state.new_zeros(state.shape[0])
        for start in range(0, c.numel(), chunk_size):
            tail, head = incidence[:, start : start + chunk_size]
            weights = c[start : start + chunk_size]
            degree.index_add_(0, tail, weights)
            degree.index_add_(0, head, weights)
        if not bool(torch.isfinite(degree).all()):
            raise FloatingPointError("Nonfinite symmetric conductance degree")
        active = degree > 0
        safe_degree = torch.where(active, degree, torch.ones_like(degree))
        inverse = safe_degree.rsqrt() * active.to(state.dtype)
        propagated = torch.zeros_like(state)
        for start in range(0, c.numel(), chunk_size):
            tail, head = incidence[:, start : start + chunk_size]
            weights = c[start : start + chunk_size] * inverse[tail] * inverse[head]
            propagated.index_add_(0, tail, weights[:, None] * state[head])
            propagated.index_add_(0, head, weights[:, None] * state[tail])
        laplacian_state = active[:, None] * state - propagated
        result = state - alpha * laplacian_state
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite symmetric propagation")
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(state, c, incidence, alpha, inverse, propagated, active)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        state, c, incidence, alpha, inverse, propagated, active = ctx.saved_tensors
        propagated_grad = torch.zeros_like(grad_output)
        need_state, need_c, _, need_alpha, _ = ctx.needs_input_grad
        if need_state or need_c:
            for start in range(0, c.numel(), ctx.chunk_size):
                tail, head = incidence[:, start : start + ctx.chunk_size]
                weights = c[start : start + ctx.chunk_size] * inverse[tail] * inverse[head]
                propagated_grad.index_add_(0, tail, weights[:, None] * grad_output[head])
                propagated_grad.index_add_(0, head, weights[:, None] * grad_output[tail])
        grad_state = (
            grad_output - alpha * (active[:, None] * grad_output - propagated_grad)
            if need_state
            else None
        )
        grad_c = None
        if need_c:
            degree_term = (
                -0.5
                * (grad_output * propagated + state * propagated_grad).sum(dim=1)
                * inverse.square()
            )
            grad_c = torch.empty_like(c)
            for start in range(0, c.numel(), ctx.chunk_size):
                tail, head = incidence[:, start : start + ctx.chunk_size]
                direct_term = (
                    inverse[tail]
                    * inverse[head]
                    * (
                        (grad_output[tail] * state[head]).sum(dim=1)
                        + (grad_output[head] * state[tail]).sum(dim=1)
                    )
                )
                grad_c[start : start + ctx.chunk_size] = alpha * (
                    direct_term + degree_term[tail] + degree_term[head]
                )
        grad_alpha = (
            -(grad_output * (active[:, None] * state - propagated)).sum().reshape_as(alpha)
            if need_alpha
            else None
        )
        return grad_state, grad_c, None, grad_alpha, None


def symmetric_propagation(
    state: Tensor,
    c: Tensor,
    incidence: Tensor,
    alpha: Tensor,
    *,
    edge_chunk_size: int = 65536,
) -> Tensor:
    """H-alpha D_C^-1/2 B.T C B D_C^-1/2 H, identity on isolates."""
    if state.ndim != 2 or state.dtype not in {torch.float32, torch.float64}:
        raise ValueError("state must be a float32/float64 node-feature matrix")
    if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("incidence must be a 2 x E int64 tensor")
    if c.ndim != 1 or c.shape[0] != incidence.shape[1] or c.dtype != state.dtype:
        raise ValueError("C must have one same-dtype scalar per physical edge")
    if alpha.ndim != 0 or alpha.dtype != state.dtype:
        raise ValueError("alpha must be a same-dtype scalar tensor")
    if any(value.device != state.device for value in (c, incidence, alpha)):
        raise ValueError("state, C, incidence and alpha must share a device")
    if (
        isinstance(edge_chunk_size, bool)
        or not isinstance(edge_chunk_size, int)
        or edge_chunk_size < 1
    ):
        raise ValueError("edge_chunk_size must be a positive integer")
    if incidence.numel() and (
        bool((incidence < 0).any()) or bool((incidence >= state.shape[0]).any())
    ):
        raise ValueError("incidence endpoint is outside the node matrix")
    if incidence.shape[1] and bool((incidence[0] == incidence[1]).any()):
        raise ValueError("physical incidence edges must not contain self loops")
    if not bool(torch.isfinite(c.detach()).all()) or not bool((c.detach() > 0).all()):
        raise FloatingPointError("C must remain finite and positive")
    if not bool(torch.isfinite(state.detach()).all()):
        raise FloatingPointError("Nonfinite state")
    checked_alpha = alpha.detach()
    if not bool(torch.isfinite(checked_alpha)) or not bool(
        (checked_alpha >= 0) & (checked_alpha <= 1)
    ):
        raise FloatingPointError("alpha must be finite and in [0, 1]")
    return _SymmetricPropagation.apply(state, c, incidence, alpha, edge_chunk_size)
