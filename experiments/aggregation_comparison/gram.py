"""Exact first-order Gram autograd with one history/C gradient allocation."""

import torch
from torch.autograd.function import once_differentiable


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
    """Exact contraction without allocating node x head x pair statistics.

    Uses the same already-projected history (including AMP rounding) as LocalGram.
    Higher derivatives are deliberately unsupported, as in LocalGram.
    """

    @staticmethod
    def forward(ctx, history, c, readout, edges, chunk, diagonal_only):
        depth, nodes, heads, _ = history.shape
        pairs = (
            torch.arange(depth, device=history.device).expand(2, -1)
            if diagonal_only
            else torch.triu_indices(depth, depth, device=history.device)
        )
        out = history.new_zeros(nodes, heads, readout.shape[-1])
        for start in range(0, edges.shape[1], chunk):
            ends = edges[:, start : start + chunk]
            delta = history[:, ends[1]] - history[:, ends[0]]
            product = (delta[pairs[0]] * delta[pairs[1]]).sum(-1).permute(1, 2, 0)
            value = torch.einsum("ehp,hpd->ehd", product, readout)
            value = value * c[start : start + chunk, :, None] / 2
            out.index_add_(0, ends[0], value)
            out.index_add_(0, ends[1], value)
        ctx.save_for_backward(history, c, readout, edges, pairs)
        ctx.chunk = chunk
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        history, c, readout, edges, pairs = ctx.saved_tensors
        dh = torch.zeros_like(history) if ctx.needs_input_grad[0] else None
        dc = torch.zeros_like(c) if ctx.needs_input_grad[1] else None
        dr = torch.zeros_like(readout) if ctx.needs_input_grad[2] else None
        for start in range(0, edges.shape[1], ctx.chunk):
            ends = edges[:, start : start + ctx.chunk]
            delta = history[:, ends[1]] - history[:, ends[0]]
            product = (delta[pairs[0]] * delta[pairs[1]]).sum(-1).permute(1, 2, 0)
            upstream = (gradient[ends[0]] + gradient[ends[1]]) / 2
            weight = c[start : start + ends.shape[1], :, None]
            if dr is not None:
                dr.add_(torch.einsum("ehp,ehd->hpd", product * weight, upstream))
            pair_gradient = torch.einsum("ehd,hpd->ehp", upstream, readout)
            if dc is not None:
                value = (pair_gradient * product).sum(-1)
                if c.shape[1] == 1:
                    value = value.sum(1, keepdim=True)
                dc[start : start + ends.shape[1]].copy_(value)
            if dh is not None:
                pair_gradient = (pair_gradient * weight).permute(2, 0, 1)[..., None]
                local = torch.zeros_like(delta)
                local.index_add_(0, pairs[0], pair_gradient * delta[pairs[1]])
                local.index_add_(0, pairs[1], pair_gradient * delta[pairs[0]])
                dh.index_add_(1, ends[0], -local)
                dh.index_add_(1, ends[1], local)
        return dh, dc, dr, None, None, None
