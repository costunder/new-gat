"""Exact first-order Gram autograd with one history/C gradient allocation."""

import torch
from torch.autograd.function import once_differentiable


def diagnostic_projection(history, weight, node_chunk):
    """Project every node/hop without simultaneously stacking full input history.

    Observation-only path: prediction readouts retain their declared contraction.
    Chunk over nodes for memory, vectorize all hops/heads/features within a chunk.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("streamed diagnostic projection requires no_grad")
    if node_chunk < 1:
        raise ValueError("diagnostic node chunk must be positive")
    nodes = history[0].shape[0]
    result = None
    for start in range(0, max(nodes, 1), node_chunk):
        stop = min(start + node_chunk, nodes)
        projected = torch.einsum(
            "knd,hdw->knhw", torch.stack([value[start:stop] for value in history]), weight
        )
        if result is None:
            result = projected.new_empty(len(history), nodes, *projected.shape[2:])
        result[:, start:stop].copy_(projected)
    return result


class LocalGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, history, conductance, edges, chunk, diagonal_only):
        depth, nodes, heads, _ = history.shape
        pairs = (
            torch.arange(depth, device=history.device).expand(2, -1)
            if diagonal_only
            else torch.triu_indices(depth, depth, device=history.device)
        )
        out = history.new_zeros(nodes, heads, pairs.shape[1])
        for start in range(0, edges.shape[1], chunk):
            ends = edges[:, start : start + chunk]
            delta = history[:, ends[1]] - history[:, ends[0]]
            product = (delta[pairs[0]] * delta[pairs[1]]).sum(-1).permute(1, 2, 0)
            value = product * conductance[start : start + chunk, :, None] / 2
            out.index_add_(0, ends[0], value)
            out.index_add_(0, ends[1], value)
        ctx.save_for_backward(history, conductance, edges, pairs)
        ctx.chunk = chunk
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        history, c, edges, pairs = ctx.saved_tensors
        # These buffers are allocated once, not once per gather and edge chunk.
        dh = torch.zeros_like(history) if ctx.needs_input_grad[0] else None
        dc = torch.zeros_like(c) if ctx.needs_input_grad[1] else None
        for start in range(0, edges.shape[1], ctx.chunk):
            ends = edges[:, start : start + ctx.chunk]
            delta = history[:, ends[1]] - history[:, ends[0]]
            upstream = (output_gradient[ends[0]] + output_gradient[ends[1]]) / 2
            if dc is not None:
                products = (delta[pairs[0]] * delta[pairs[1]]).sum(-1).permute(1, 2, 0)
                contribution = (upstream * products).sum(-1)
                if c.shape[1] == 1:
                    contribution = contribution.sum(1, keepdim=True)
                dc[start : start + ends.shape[1]].copy_(contribution)
            if dh is not None:
                weight = (upstream * c[start : start + ends.shape[1], :, None]).permute(2, 0, 1)
                # K x chunk x H x d, independent of the full node count.
                local = torch.zeros_like(delta)
                local.index_add_(0, pairs[0], weight[..., None] * delta[pairs[1]])
                local.index_add_(0, pairs[1], weight[..., None] * delta[pairs[0]])
                dh.index_add_(1, ends[0], -local)
                dh.index_add_(1, ends[1], local)
        return dh, dc, None, None, None


class GramReadout(torch.autograd.Function):
    """Reference-order readout with Gram recomputation in first-order backward.

    The historical edge-first contraction reordered FP32 sums before downstream
    BF16 rounding. Matching dtype alone did not match deep-model gradients.
    Keep the node-Gram-then-readout order exactly. A transient N x H x P Gram
    is allocated, but not saved across forward/backward; backward recomputes it.
    This is an explicit memory/recomputation policy, not a speedup claim.
    """

    @staticmethod
    def forward(ctx, history, c, readout, edges, chunk, diagonal_only):
        with torch.autocast(device_type=history.device.type, enabled=False):
            statistics = LocalGram.apply(history, c, edges, chunk, diagonal_only)
            out = torch.einsum("nhp,hpd->nhd", statistics, readout)
        ctx.save_for_backward(history, c, readout, edges)
        ctx.chunk, ctx.diagonal_only = chunk, diagonal_only
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        history, c, readout, edges = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        # Reuse reference autograd order in backward too. LocalGram allocates
        # each history/C gradient buffer once; no Gram survives from forward.
        values = [
            value.detach().requires_grad_(needed)
            for value, needed in zip((history, c, readout), needs, strict=True)
        ]
        with torch.enable_grad(), torch.autocast(device_type=history.device.type, enabled=False):
            statistics = LocalGram.apply(values[0], values[1], edges, ctx.chunk, ctx.diagonal_only)
            out = torch.einsum("nhp,hpd->nhd", statistics, values[2])
            gradients = iter(
                torch.autograd.grad(
                    out,
                    [value for value, needed in zip(values, needs, strict=True) if needed],
                    gradient,
                    create_graph=False,
                )
            )
        dh, dc, dr = (next(gradients) if needed else None for needed in needs)
        return dh, dc, dr, None, None, None
