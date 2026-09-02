"""Exact chunked symmetric propagation for distinct residual and message states.

For nonisolated nodes this computes

    (1 - alpha) H + alpha D_C^-1/2 A_C D_C^-1/2 M,

where ``H`` is the residual state and ``M`` is the message state (``H W`` in
the v4 model). Isolates remain exactly ``H``. The custom backward differentiates
the direct edge weights and both C-dependent symmetric degree factors. It saves
O(nd + m) values, uses O(chunk*d) edge workspace, and intentionally supports
first-order gradients only. No dense adjacency, incidence, C, or Laplacian is
materialized.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


class _SymmetricSpatialPropagation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        residual_state: Tensor,
        message_state: Tensor,
        c: Tensor,
        incidence: Tensor,
        alpha: Tensor,
        chunk_size: int,
    ) -> Tensor:
        degree = residual_state.new_zeros(residual_state.shape[0])
        for start in range(0, c.numel(), chunk_size):
            stop = start + chunk_size
            tail, head = incidence[:, start:stop]
            weights = c[start:stop]
            degree.index_add_(0, tail, weights)
            degree.index_add_(0, head, weights)
        if not bool(torch.isfinite(degree).all()):
            raise FloatingPointError("Nonfinite symmetric conductance degree")

        active = degree > 0
        safe_degree = torch.where(active, degree, torch.ones_like(degree))
        inverse = safe_degree.rsqrt() * active.to(residual_state.dtype)
        propagated = torch.zeros_like(message_state)
        for start in range(0, c.numel(), chunk_size):
            stop = start + chunk_size
            tail, head = incidence[:, start:stop]
            weights = c[start:stop] * inverse[tail] * inverse[head]
            propagated.index_add_(0, tail, weights[:, None] * message_state[head])
            propagated.index_add_(0, head, weights[:, None] * message_state[tail])

        result = residual_state - alpha * (active[:, None] * residual_state - propagated)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite symmetric spatial propagation")

        ctx.chunk_size = chunk_size
        ctx.save_for_backward(
            residual_state,
            message_state,
            c,
            incidence,
            alpha,
            inverse,
            propagated,
            active,
        )
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        (
            residual_state,
            message_state,
            c,
            incidence,
            alpha,
            inverse,
            propagated,
            active,
        ) = ctx.saved_tensors
        (
            need_residual,
            need_message,
            need_c,
            _,
            need_alpha,
            _,
        ) = ctx.needs_input_grad

        # P_C is symmetric. P_C grad_output is needed both for the message
        # gradient and for the degree-factor part of dL/dC.
        propagated_grad = None
        if need_message or need_c:
            propagated_grad = torch.zeros_like(grad_output)
            for start in range(0, c.numel(), ctx.chunk_size):
                stop = start + ctx.chunk_size
                tail, head = incidence[:, start:stop]
                weights = c[start:stop] * inverse[tail] * inverse[head]
                propagated_grad.index_add_(0, tail, weights[:, None] * grad_output[head])
                propagated_grad.index_add_(0, head, weights[:, None] * grad_output[tail])

        grad_residual = (
            grad_output - alpha * active[:, None] * grad_output if need_residual else None
        )
        grad_message = alpha * propagated_grad if need_message else None

        grad_c = None
        if need_c:
            # The first inner product is the derivative of the row inverse
            # degree; the second is the derivative of the column inverse
            # degree. Each incident C_e contributes to both endpoint degrees.
            degree_term = (
                -0.5
                * (
                    (grad_output * propagated).sum(dim=1)
                    + (message_state * propagated_grad).sum(dim=1)
                )
                * inverse.square()
            )
            grad_c = torch.empty_like(c)
            for start in range(0, c.numel(), ctx.chunk_size):
                stop = start + ctx.chunk_size
                tail, head = incidence[:, start:stop]
                direct_term = (
                    inverse[tail]
                    * inverse[head]
                    * (
                        (grad_output[tail] * message_state[head]).sum(dim=1)
                        + (grad_output[head] * message_state[tail]).sum(dim=1)
                    )
                )
                grad_c[start:stop] = alpha * (direct_term + degree_term[tail] + degree_term[head])

        grad_alpha = (
            -(grad_output * (active[:, None] * residual_state - propagated)).sum().reshape_as(alpha)
            if need_alpha
            else None
        )
        return grad_residual, grad_message, grad_c, None, grad_alpha, None


def symmetric_spatial_propagation(
    residual_state: Tensor,
    message_state: Tensor,
    c: Tensor,
    incidence: Tensor,
    alpha: Tensor,
    *,
    edge_chunk_size: int = 65536,
) -> Tensor:
    """Apply the v4 symmetric operator with identity behavior on isolates.

    ``residual_state`` and ``message_state`` must have the same node/feature
    shape. Passing the same tensor for both exactly recovers the v3 equation.
    Physical edges are undirected and must appear exactly once; either endpoint
    orientation and any edge ordering are accepted.
    """

    allowed_dtypes = {torch.float32, torch.float64}
    if residual_state.ndim != 2 or residual_state.dtype not in allowed_dtypes:
        raise ValueError("residual_state must be a float32/float64 node-feature matrix")
    if (
        message_state.ndim != 2
        or message_state.shape != residual_state.shape
        or message_state.dtype != residual_state.dtype
    ):
        raise ValueError("message_state must match residual_state shape and dtype")
    if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("incidence must be a 2 x E int64 tensor")
    if c.ndim != 1 or c.shape[0] != incidence.shape[1] or c.dtype != residual_state.dtype:
        raise ValueError("C must have one residual-state-dtype scalar per physical edge")
    if alpha.ndim != 0 or alpha.dtype != residual_state.dtype:
        raise ValueError("alpha must be a residual-state-dtype scalar tensor")
    if any(value.device != residual_state.device for value in (message_state, c, incidence, alpha)):
        raise ValueError("all propagation inputs must share a device")
    if (
        isinstance(edge_chunk_size, bool)
        or not isinstance(edge_chunk_size, int)
        or edge_chunk_size < 1
    ):
        raise ValueError("edge_chunk_size must be a positive integer")
    if incidence.numel() and (
        bool((incidence < 0).any()) or bool((incidence >= residual_state.shape[0]).any())
    ):
        raise ValueError("incidence endpoint is outside the node matrix")
    if incidence.shape[1] and bool((incidence[0] == incidence[1]).any()):
        raise ValueError("physical incidence edges must not contain self loops")
    if not bool(torch.isfinite(c.detach()).all()) or not bool((c.detach() > 0).all()):
        raise FloatingPointError("C must remain finite and positive")
    if not bool(torch.isfinite(residual_state.detach()).all()):
        raise FloatingPointError("Nonfinite residual state")
    if not bool(torch.isfinite(message_state.detach()).all()):
        raise FloatingPointError("Nonfinite message state")
    checked_alpha = alpha.detach()
    if not bool(torch.isfinite(checked_alpha)) or not bool(
        (checked_alpha >= 0) & (checked_alpha <= 1)
    ):
        raise FloatingPointError("alpha must be finite and in [0, 1]")
    return _SymmetricSpatialPropagation.apply(
        residual_state,
        message_state,
        c,
        incidence,
        alpha,
        edge_chunk_size,
    )
