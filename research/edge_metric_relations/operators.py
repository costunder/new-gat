"""Sparse physical-node actions of the occurrence edge metric."""
from __future__ import annotations

from numbers import Integral
import torch
from torch import Tensor

from .geometry import EdgeGeometry


def check_value(g, value):
    if not isinstance(g, EdgeGeometry) or not isinstance(value, Tensor):
        raise TypeError("expected EdgeGeometry and tensor")
    if value.ndim < 2 or value.shape[-2] != g.n or value.shape[-1] < 1:
        raise ValueError("value must be [...,all physical nodes,features]")
    if value.dtype not in (torch.float32, torch.float64) or value.device != g.c0.device or value.dtype != g.c0.dtype:
        raise ValueError("values and geometry must share float32/float64 dtype and device")


def chunk_size(value, count):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
        raise ValueError("chunk must be a positive integer or None")
    return max(1, count) if value is None else int(value)


def incidence(g, value):
    check_value(g, value)
    return value.index_select(-2, g.edges[1]) - value.index_select(-2, g.edges[0])


def divergence(g, flow):
    if flow.ndim < 2 or flow.shape[-2] != g.num_edges:
        raise ValueError("flow must be [...,physical_edges,features]")
    result = flow.new_zeros(flow.shape[:-2] + (g.n, flow.shape[-1]))
    result = result.index_add(-2, g.edges[0], -flow)
    return result.index_add(-2, g.edges[1], flow)


def diagonal_action(g, value, multiplier=None):
    grad = incidence(g, value)
    coefficient = g.cbar0 if multiplier is None else multiplier * g.cbar0
    return divergence(g, grad * coefficient.unsqueeze(-1))


def pair_physical_coefficient(g, multiplier, begin, end, pair_coefficient):
    """Ceff off-diagonal coefficient for canonical occurrence relations."""
    l, r = g.pair_left[begin:end], g.pair_right[begin:end]
    el, er = g.occ_edge[l], g.occ_edge[r]
    scale = (g.c0[l] * g.c0[r]).sqrt() * g.occ_scale[l] * g.occ_scale[r]
    learned = (multiplier.index_select(-1, el) * multiplier.index_select(-1, er)).sqrt()
    return el, er, pair_coefficient * scale * learned


def apply_metric(g, value, multiplier=None, pair_coefficient=None, *, rho=.5, pair_chunk=None):
    """Apply Bcal.T sqrt(D)(I+rho K)sqrt(D) Bcal.

    pair_coefficient contains the orientation sign and topology normalization.
    Only scalar coefficients may be supplied in full; no dense metric is built.
    """
    grad = incidence(g, value)
    if multiplier is None:
        multiplier = value.new_ones(value.shape[:-2] + (g.num_edges,))
    if multiplier.shape != value.shape[:-2] + (g.num_edges,):
        raise ValueError("multiplier must match the value prefix and physical edge count")
    flow = grad * (multiplier * g.cbar0).unsqueeze(-1)
    if pair_coefficient is not None:
        if pair_coefficient.shape != value.shape[:-2] + (g.num_pairs,):
            raise ValueError("pair coefficient must match the value prefix and eligible pair count")
        size = chunk_size(pair_chunk, g.num_pairs)
        for begin in range(0, g.num_pairs, size):
            end = min(begin + size, g.num_pairs)
            el, er, c = pair_physical_coefficient(g, multiplier, begin, end, pair_coefficient[..., begin:end])
            flow = flow.index_add(-2, el, rho * c.unsqueeze(-1) * grad.index_select(-2, er))
            flow = flow.index_add(-2, er, rho * c.unsqueeze(-1) * grad.index_select(-2, el))
    return divergence(g, flow)


def graph_sum(g, scalar, graph_index):
    """Reduce the last relation axis, retaining seed and realization axes."""
    result = scalar.new_zeros(scalar.shape[:-1] + (g.num_graphs,))
    return result.index_add(-1, graph_index, scalar)


__all__ = ["incidence", "divergence", "diagonal_action", "apply_metric", "pair_physical_coefficient", "graph_sum"]
