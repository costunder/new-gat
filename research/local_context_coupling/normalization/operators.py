"""Symmetric degree-normalized S/G actions on all original local copies.

The frozen base geometry remains the raw weighted local A and unit-link K.
Normalization coefficients are separate tensors. S and G are explicit fixed
Laplacians with weighted degree at most one half. Edge chunks retain every
edge and custom symmetric autograd avoids saving feature-sized edge flows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
import math
from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from ..operators import CouplingGeometry, merge, replicate


@dataclass(frozen=True)
class NormalizedGeometry:
    base: CouplingGeometry
    intra_policy: str
    cross_policy: str
    s_weights: Tensor
    g_weights: Tensor
    s_degree: Tensor
    g_degree: Tensor
    eta_by_local: Tensor
    metadata: dict[str, Any]

    @property
    def mode(self):
        return self.base.mode

    @property
    def topology(self):
        return self.base.topology

    @property
    def n(self):
        return self.base.n

    @property
    def num_graphs(self):
        return self.base.num_graphs

    @property
    def num_copies(self):
        return self.base.num_copies

    @property
    def num_cross_edges(self):
        return self.base.num_cross_edges

    def to(self, device, dtype=torch.float64, base: CouplingGeometry | None = None):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("normalized geometry precision must be float32/64")
        converted = self.base.to(device, dtype) if base is None else base
        requested_device = torch.device(device)
        same_device = converted.intra_weights.device.type == requested_device.type
        if requested_device.index is not None:
            same_device = same_device and converted.intra_weights.device.index == requested_device.index
        if (converted.mode != self.mode or converted.n != self.n
                or converted.num_copies != self.num_copies
                or converted.num_cross_edges != self.num_cross_edges
                or converted.intra_weights.dtype != dtype
                or not same_device):
            raise ValueError("shared converted base geometry does not match normalization")
        return replace(self, base=converted, **{
            key: getattr(self, key).to(device=device, dtype=dtype)
            for key in ("s_weights", "g_weights", "s_degree", "g_degree", "eta_by_local")
        })


def _safe_step(degree_maximum):
    positive = degree_maximum > 0
    denominator = torch.where(positive, degree_maximum, torch.ones_like(degree_maximum))
    return torch.where(positive, .5 / denominator, torch.zeros_like(degree_maximum))


def _degree(count, left, right, weights):
    return weights.new_zeros(count).index_add(0, left, weights).index_add(0, right, weights)


def prepare_normalized_geometry(base: CouplingGeometry, intra="graph", cross="graph"):
    """Cache full coefficients once on CPU; none is an off diagnostic reference.

    For each local v, local policy uses .5/max_i degree(A_v)_i. Graph policy
    repeats the existing graph-wide eta. Edge G uses the unit K's endpoint
    degrees before normalization, never an iteratively changed degree.
    """
    if not isinstance(base, CouplingGeometry) or base.intra_weights.device.type != "cpu":
        raise ValueError("prepare_normalized_geometry requires a CPU CouplingGeometry")
    if intra not in ("graph", "local") or cross not in ("none", "graph", "edge"):
        raise ValueError("unknown intra/cross normalization policy")
    if intra == "graph":
        eta = base.eta.index_select(0, base.physical_graph)
    else:
        local_max = base.intra_degree.new_zeros(base.n).scatter_reduce(
            0, base.topology.local_node_center, base.intra_degree,
            reduce="amax", include_self=True,
        )
        eta = _safe_step(local_max)
    tail, head = base.topology.local_edge_nodes
    s_weights = base.intra_weights * eta.index_select(0, base.topology.local_edge_center)
    if cross in ("none", "graph"):
        g_weights = base.gamma.index_select(0, base.cross_graph)
    else:
        maximum = torch.maximum(
            base.cross_degree.index_select(0, base.cross_left),
            base.cross_degree.index_select(0, base.cross_right),
        )
        if bool((maximum <= 0).any()):
            raise ValueError("every canonical cross link must have positive unit-K endpoint degree")
        g_weights = .5 / maximum
    s_degree = _degree(base.num_copies, tail, head, s_weights)
    g_degree = _degree(base.num_copies, base.cross_left, base.cross_right, g_weights)
    tolerance = 32 * torch.finfo(s_weights.dtype).eps
    for key, weights, degrees in (("S", s_weights, s_degree), ("G", g_weights, g_degree)):
        if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
            raise ValueError(f"{key} edge weights must be finite and positive on every retained edge")
        if not bool(torch.isfinite(degrees).all()) or bool((degrees > .5 + tolerance).any()):
            raise ValueError(f"{key} normalized weighted degree exceeds one half")
    nonzero_eta = eta[eta > 0]
    metadata = {
        **base.metadata, "intra_policy": intra, "cross_policy": cross,
        "diagnostic_cross_policy": "graph" if cross == "none" else cross,
        "energy_operator": "applied_S_G",
        "step_rule": "explicit_separate_fixed_S_G_coefficients",
        "base_step_rule": base.metadata["step_rule"],
        "intra_step_rule": "0.5/max_graph_copy_weighted_degree" if intra == "graph" else "0.5/max_ego_weighted_degree",
        "cross_rule": "0.5/max_graph_unit_K_degree" if cross in ("none", "graph") else "0.5/max_unit_K_endpoint_degree",
        "s_degree_max": float(s_degree.max()), "g_degree_max": float(g_degree.max()),
        "eta_min_nonzero": float(nonzero_eta.min()) if nonzero_eta.numel() else None,
        "eta_max": float(eta.max()), "num_locals": base.n,
        "all_canonical_cross_links_retained": True,
        "off_cross_operator_is_diagnostic_reference": cross == "none",
    }
    return NormalizedGeometry(base, intra, cross, s_weights, g_weights, s_degree, g_degree, eta, metadata)


def _check(geometry, value, nodes):
    if not isinstance(geometry, NormalizedGeometry):
        raise ValueError("NormalizedGeometry required")
    if (not isinstance(value, Tensor) or value.ndim not in (2, 3)
            or value.shape[-2] != nodes or value.shape[-1] < 1
            or value.shape[0] < 1):
        raise ValueError("values must preserve every physical/copy node with 2D or packed 3D shape")
    if value.dtype != geometry.s_weights.dtype or value.device != geometry.s_weights.device:
        raise ValueError("value and normalization coefficients must share device and dtype")


def _fields(geometry, kind):
    if kind == "S":
        left, right = geometry.topology.local_edge_nodes
        return left, right, geometry.s_weights
    if kind in ("G", "K"):
        base = geometry.base
        weights = geometry.g_weights if kind == "G" else None
        return base.cross_left, base.cross_right, weights
    raise ValueError("normalized operator kind must be S, G or raw K")


def _chunk(chunk):
    if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk < 1):
        raise ValueError("edge_chunk must be a positive integer or None")
    return None if chunk is None else int(chunk)


def _laplacian_chunks(geometry, value, kind, chunk):
    left, right, weights = _fields(geometry, kind)
    count, size = left.numel(), left.numel() if chunk is None else chunk
    result = torch.zeros_like(value)
    if count:
        for begin in range(0, count, size):
            end = min(begin + size, count)
            flow = value.index_select(-2, right[begin:end]) - value.index_select(-2, left[begin:end])
            if weights is not None:
                flow = flow * weights[begin:end].reshape((1,) * (value.ndim - 2) + (-1, 1))
            result.index_add_(-2, left[begin:end], -flow)
            result.index_add_(-2, right[begin:end], flow)
    return result


class _FixedNormalizedLaplacian(Function):
    @staticmethod
    def forward(ctx, value, geometry, kind, chunk):
        ctx.geometry, ctx.kind, ctx.chunk = geometry, kind, chunk
        return _laplacian_chunks(geometry, value, kind, chunk)

    @staticmethod
    def backward(ctx, gradient):
        return _FixedNormalizedLaplacian.apply(gradient, ctx.geometry, ctx.kind, ctx.chunk), None, None, None


def action(geometry, value, kind, edge_chunk=None):
    _check(geometry, value, geometry.num_copies)
    return _FixedNormalizedLaplacian.apply(value, geometry, kind, _chunk(edge_chunk))


def apply_s(geometry, value, edge_chunk=None):
    return action(geometry, value, "S", edge_chunk)


def apply_g(geometry, value, edge_chunk=None):
    return action(geometry, value, "G", edge_chunk)


def apply_raw_k(geometry, value, edge_chunk=None):
    return action(geometry, value, "K", edge_chunk)


def energy(geometry, value, kind, edge_chunk=None):
    """Half energy per graph and channel: [G,F] or [S,G,F]."""
    _check(geometry, value, geometry.num_copies)
    chunk = _chunk(edge_chunk)
    left, right, weights = _fields(geometry, kind)
    graph = geometry.base.copy_graph[left] if kind == "S" else geometry.base.cross_graph
    count, size = left.numel(), left.numel() if chunk is None else chunk
    result = value.new_zeros(value.shape[:-2] + (geometry.num_graphs, value.shape[-1]))
    if count:
        for begin in range(0, count, size):
            end = min(begin + size, count)
            difference = value.index_select(-2, right[begin:end]) - value.index_select(-2, left[begin:end])
            squared = difference.square()
            if weights is not None:
                squared = squared * weights[begin:end].reshape((1,) * (value.ndim - 2) + (-1, 1))
            result = result.index_add(-2, graph[begin:end], .5 * squared)
    return result


def energy_s(geometry, value, edge_chunk=None):
    return energy(geometry, value, "S", edge_chunk)


def energy_g(geometry, value, edge_chunk=None):
    return energy(geometry, value, "G", edge_chunk)


def mixed_action(geometry, value, edge_chunk=None):
    _check(geometry, value, geometry.n)
    y = replicate(geometry.base, value)
    return merge(geometry.base, apply_s(geometry, apply_g(geometry, apply_s(geometry, y, edge_chunk), edge_chunk), edge_chunk))


def _gain(value, reference):
    if isinstance(value, bool) or not isinstance(value, (Real, Tensor)):
        raise ValueError("cross gain must be scalar or packed-seed tensor")
    if isinstance(value, Real) and (not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("cross gain must lie in [0,1]")
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim and (reference.ndim != 3 or tensor.shape != (reference.shape[0],)):
        raise ValueError("cross gain must be scalar or [packed seeds]")
    if isinstance(value, Tensor):
        valid = (torch.isfinite(tensor) & (tensor >= 0) & (tensor <= 1)).all()
        if valid.device.type == "cpu":
            if not bool(valid):
                raise ValueError("cross gain must be finite and lie in [0,1]")
        else:
            torch._assert_async(valid, "cross gain must be finite and lie in [0,1]")
    return tensor.reshape(-1, 1, 1) if tensor.ndim else tensor


def sandwich(geometry, value, *, cross=True, cross_gain=1., diagnostics=False, edge_chunk=None):
    """M(I-S)(I-rho G)(I-S)R on the full fixed copy topology."""
    _check(geometry, value, geometry.n)
    if not isinstance(cross, bool) or not isinstance(diagnostics, bool):
        raise TypeError("cross and diagnostics must be boolean")
    rho = _gain(cross_gain, value)
    y0 = replicate(geometry.base, value)
    y1 = y0 - apply_s(geometry, y0, edge_chunk)
    y2 = y1 - rho * apply_g(geometry, y1, edge_chunk) if cross else y1
    y3 = y2 - apply_s(geometry, y2, edge_chunk)
    result = merge(geometry.base, y3)
    if not diagnostics:
        return result
    return result, {"replicated": y0, "after_intra_1": y1, "after_cross": y2,
                    "after_intra_2": y3, "cross_gain": rho, "energy_operator": "applied_S_G"}


def immediate(geometry, value, *, cross=True, cross_gain=1., edge_chunk=None):
    """Cancellation control: one internal step, cross, immediate copy mean."""
    _check(geometry, value, geometry.n)
    if not isinstance(cross, bool):
        raise TypeError("cross must be boolean")
    rho = _gain(cross_gain, value)
    y1 = replicate(geometry.base, value)
    y1 = y1 - apply_s(geometry, y1, edge_chunk)
    y2 = y1 - rho * apply_g(geometry, y1, edge_chunk) if cross else y1
    return merge(geometry.base, y2)
