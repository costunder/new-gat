"""Shared, seed-packed diagonal and pair generators with exact sparse actions.

Seed is the leading axis. Any intervening axes, including independent scalar
realizations, remain independent. Only the last feature axis enters RMS.
Layer-specific namespaces give independent generators with matched condition
initializations. Exact relation chunks are checkpointed, never sampled.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .operators import (
    check_value, chunk_size, divergence, graph_sum, incidence, pair_physical_coefficient,
)

CONDITIONS = ("D0", "D1", "F0", "F1", "F2", "DA")
INTERVENTIONS = ("offdiag_zero", "diagonal_one", "pair_zero")
EPSILON = 1e-4


def _features(left, right, epsilon):
    scale = ((left.square().mean(-1) + right.square().mean(-1)) * .5 + epsilon ** 2).sqrt()
    left, right = left / scale.unsqueeze(-1), right / scale.unsqueeze(-1)
    l, r = left.square().mean(-1), right.square().mean(-1)
    return torch.stack((l + r, (l - r).abs(), (left * right).mean(-1), (left - right).square().mean(-1)), -1)


def pair_features(g, z, begin=0, end=None, epsilon=EPSILON):
    check_value(g, z)
    if epsilon != EPSILON:
        raise ValueError("RMS epsilon is locked to 1e-4")
    end = g.num_pairs if end is None else end
    anchor = z.index_select(-2, g.pair_anchor[begin:end])
    left = z.index_select(-2, g.pair_other_left[begin:end]) - anchor
    right = z.index_select(-2, g.pair_other_right[begin:end]) - anchor
    dynamic = _features(left, right, epsilon)
    structural = g.pair_structure[begin:end].expand(z.shape[:-2] + (end - begin, 4))
    return torch.cat((dynamic, structural), -1)


def physical_edge_features(g, z, begin=0, end=None, epsilon=EPSILON):
    check_value(g, z)
    if epsilon != EPSILON:
        raise ValueError("RMS epsilon is locked to 1e-4")
    end = g.num_edges if end is None else end
    return _features(z.index_select(-2, g.edges[0, begin:end]), z.index_select(-2, g.edges[1, begin:end]), epsilon)


class _PackedMLP(nn.Module):
    def __init__(self, seeds, inputs, hidden, namespace):
        super().__init__()
        value = torch.empty(len(seeds), inputs, hidden)
        for i, seed in enumerate(seeds):
            key = f"edge-metric-v1|{namespace}|{seed}".encode()
            named = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1)
            generator = torch.Generator().manual_seed(named)
            value[i].uniform_(-math.sqrt(6 / (inputs + hidden)), math.sqrt(6 / (inputs + hidden)), generator=generator)
        self.w1 = nn.Parameter(value)
        self.b1 = nn.Parameter(torch.zeros(len(seeds), hidden))
        self.w2 = nn.Parameter(torch.zeros(len(seeds), hidden, 1))
        self.b2 = nn.Parameter(torch.zeros(len(seeds), 1))

    def forward(self, features):
        if features.shape[0] != self.w1.shape[0]:
            raise ValueError("MLP seed axis must match its independent parameter blocks")
        flat = features.reshape(features.shape[0], -1, features.shape[-1])
        hidden = F.silu(torch.bmm(flat, self.w1) + self.b1[:, None])
        output = torch.bmm(hidden, self.w2) + self.b2[:, None]
        return output.reshape(features.shape[:-1])


def _checkpoint(function, *arguments):
    if torch.is_grad_enabled():
        return checkpoint(function, *arguments, use_reentrant=False, preserve_rng_state=False)
    return function(*arguments)


class EdgeMetricGate(nn.Module):
    def __init__(self, condition, seeds=(0,), hidden=64, epsilon=EPSILON, namespace="layer_0"):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown edge metric condition {condition!r}")
        if isinstance(hidden, bool) or not isinstance(hidden, Integral) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if epsilon != EPSILON:
            raise ValueError("RMS epsilon is locked to 1e-4")
        if not isinstance(seeds, (tuple, list)) or not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds
        ) or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a nonempty distinct sequence of nonnegative integers")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must identify this layer's matched generators")
        self.condition, self.seeds, self.epsilon, self.hidden = condition, tuple(map(int, seeds)), epsilon, int(hidden)
        self.namespace, self.rho = namespace, .5
        self.diagonal_gate = _PackedMLP(self.seeds, 4, self.hidden, namespace + ".diagonal") if condition in ("D1", "F2", "DA") else None
        self.pair_gate = _PackedMLP(self.seeds, 8, self.hidden, namespace + ".pair") if condition in ("F1", "F2", "DA") else None

    @property
    def parameters_per_seed(self):
        return sum(p.numel() for p in self.parameters()) // len(self.seeds)

    @property
    def trainable_parameters_per_seed(self):
        return self.parameters_per_seed

    def _pack(self, g, z, value=None):
        check_value(g, z)
        bare = z.ndim == 2
        if bare:
            if len(self.seeds) != 1:
                raise ValueError("bare values can only be used with one seed")
            z = z.unsqueeze(0)
            value = None if value is None else value.unsqueeze(0)
        if z.shape[0] != len(self.seeds):
            raise ValueError("leading value axis must match packed independent seeds")
        if value is not None:
            check_value(g, value)
            if z.shape[:-1] != value.shape[:-1]:
                raise ValueError("gate input and applied value must have the same prefix/node axes")
        for parameter in self.parameters():
            if parameter.device != z.device or parameter.dtype != z.dtype:
                raise ValueError("gate parameters and inputs must share dtype and device")
        return z, value, bare

    @staticmethod
    def _override(value, z, count, bare, name, lower, upper):
        if value is None:
            return None
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if bare:
            value = value.unsqueeze(0)
        if value.shape != z.shape[:-2] + (count,) or value.dtype != z.dtype or value.device != z.device:
            raise ValueError(f"{name} must match input prefix, count, dtype and device")
        valid = torch.isfinite(value).all() & (value >= lower).all() & (value <= upper).all()
        if value.device.type == "cpu":
            if not bool(valid):
                raise ValueError(f"{name} outside its fixed finite bounds")
        else:
            torch._assert_async(valid, f"{name} outside its fixed finite bounds")
        return value

    def _raw(self, g, z, begin, end, pair_override=None, intervention=None):
        if intervention == "pair_zero":
            return z.new_zeros(z.shape[:-2] + (end - begin,))
        if pair_override is not None:
            return pair_override[..., begin:end]
        if self.condition == "F0":
            return z.new_ones(z.shape[:-2] + (end - begin,))
        if self.pair_gate is None:
            return z.new_zeros(z.shape[:-2] + (end - begin,))
        return self.pair_gate(pair_features(g, z, begin, end, self.epsilon)).tanh()

    def _pair_all(self, g, z, size, override, intervention):
        chunks = [self._raw(g, z, b, min(b + size, g.num_pairs), override, intervention)
                  for b in range(0, g.num_pairs, size)]
        return torch.cat(chunks, -1) if chunks else z.new_zeros(z.shape[:-2] + (0,))

    def _diagonal(self, g, z, size, pair_override, diagonal_override, intervention):
        if intervention == "diagonal_one":
            return z.new_ones(z.shape[:-2] + (g.num_edges,))
        if diagonal_override is not None:
            return diagonal_override
        if self.diagonal_gate is None:
            return z.new_ones(z.shape[:-2] + (g.num_edges,))
        logits = []
        for begin in range(0, g.num_edges, size):
            end = min(begin + size, g.num_edges)
            function = lambda source, b=begin, e=end: self.diagonal_gate(physical_edge_features(g, source, b, e, self.epsilon))
            logits.append(_checkpoint(function, z))
        logit = torch.cat(logits, -1) if logits else z.new_zeros(z.shape[:-2] + (0,))
        if self.condition == "DA":
            rows = z.new_zeros(z.shape[:-2] + (g.num_occurrences,))
            for begin in range(0, g.num_pairs, size):
                end = min(begin + size, g.num_pairs)
                function = lambda source, b=begin, e=end: self._raw(g, source, b, e, pair_override, intervention) * g.pair_norm[b:e]
                coefficient = _checkpoint(function, z)
                rows = rows.index_add(-1, g.pair_left[begin:end], coefficient)
                rows = rows.index_add(-1, g.pair_right[begin:end], coefficient)
            physical = z.new_zeros(z.shape[:-2] + (g.num_edges,)).index_add(-1, g.occ_edge, rows)
            # Every occurrence, including rows with zero eligible links, is in
            # this mean. There is no extra eligible-pair mean or hidden cap.
            logit = logit + physical / g.occ_count
        return (math.log(2) * logit.tanh()).exp()

    def coefficients(self, g, gate_z, *, pair_chunk=None, intervention=None, pair_override=None, diagonal_override=None):
        """Explicit scalar snapshot for fixed-operator audits; production need not store it."""
        if intervention is not None and intervention not in INTERVENTIONS:
            raise ValueError("unknown coefficient intervention")
        z, _, bare = self._pack(g, gate_z)
        po = self._override(pair_override, z, g.num_pairs, bare, "pair_override", -1., 1.)
        do = self._override(diagonal_override, z, g.num_edges, bare, "diagonal_override", .5, 2.)
        size = chunk_size(pair_chunk, max(g.num_pairs, g.num_edges))
        multiplier = self._diagonal(g, z, size, po, do, intervention)
        raw = self._pair_all(g, z, size, po, intervention)
        active = self.condition in ("F0", "F1", "F2") and intervention not in ("offdiag_zero", "pair_zero")
        coefficient = raw * g.pair_sign * g.pair_norm if active else torch.zeros_like(raw)
        result = {"diagonal_multiplier": multiplier, "pair_raw": raw, "pair_coefficient": coefficient, "cross_active": active}
        return {k: v.squeeze(0) if bare and isinstance(v, Tensor) else v for k, v in result.items()}

    def apply(self, g, gate_z, value, *, pair_chunk=None, intervention=None, diagnostics=False,
              pair_override=None, diagonal_override=None):
        if intervention is not None and intervention not in INTERVENTIONS:
            raise ValueError("unknown coefficient intervention")
        if not isinstance(diagnostics, bool):
            raise TypeError("diagnostics must be boolean")
        z, value, bare = self._pack(g, gate_z, value)
        po = self._override(pair_override, z, g.num_pairs, bare, "pair_override", -1., 1.)
        do = self._override(diagonal_override, z, g.num_edges, bare, "diagonal_override", .5, 2.)
        size = chunk_size(pair_chunk, max(g.num_pairs, g.num_edges))
        multiplier = self._diagonal(g, z, size, po, do, intervention)
        grad = incidence(g, value)
        diag_flow = grad * (multiplier * g.cbar0).unsqueeze(-1)
        cross_flow = torch.zeros_like(diag_flow)
        active = self.condition in ("F0", "F1", "F2") and intervention not in ("offdiag_zero", "pair_zero")
        if active:
            for begin in range(0, g.num_pairs, size):
                end = min(begin + size, g.num_pairs)
                el, er = g.occ_edge[g.pair_left[begin:end]], g.occ_edge[g.pair_right[begin:end]]
                def function(source, applied_grad, mult, b=begin, e=end):
                    raw = self._raw(g, source, b, e, po, intervention)
                    _, _, c = pair_physical_coefficient(g, mult, b, e, raw * g.pair_sign[b:e] * g.pair_norm[b:e])
                    return self.rho * c.unsqueeze(-1) * applied_grad.index_select(-2, g.occ_edge[g.pair_right[b:e]]), self.rho * c.unsqueeze(-1) * applied_grad.index_select(-2, g.occ_edge[g.pair_left[b:e]])
                left_flow, right_flow = _checkpoint(function, z, grad, multiplier)
                cross_flow = cross_flow.index_add(-2, el, left_flow)
                cross_flow = cross_flow.index_add(-2, er, right_flow)
        message_diag, message_cross = divergence(g, diag_flow), divergence(g, cross_flow)
        output = message_diag + message_cross
        details = {}
        if diagnostics:
            with torch.no_grad():
                intra = graph_sum(g, .5 * (grad * diag_flow).sum(-1), g.edge_graph)
                cross = graph_sum(g, .5 * (grad * cross_flow).sum(-1), g.edge_graph)
                # Reduce channels before expanding edge occurrences. This
                # avoids a feature-sized occurrence allocation in diagnostics.
                local_scalar = .5 * grad.square().sum(-1).index_select(-1, g.occ_edge) * g.c0 * multiplier.index_select(-1, g.occ_edge)
                raw_local = local_scalar.new_zeros(local_scalar.shape[:-1] + (g.n,)).index_add(-1, g.occ_center, local_scalar)
                corrected_local = local_scalar.new_zeros(local_scalar.shape[:-1] + (g.n,)).index_add(-1, g.occ_center, local_scalar * g.occ_scale.square())
                total = value.new_zeros(value.shape[:-2])
                squares, abs_total, nonzero = total.clone(), total.clone(), total.clone()
                minimum, maximum = torch.full_like(total, float("inf")), torch.full_like(total, -float("inf"))
                for begin in range(0, g.num_pairs, size):
                    end = min(begin + size, g.num_pairs)
                    raw = self._raw(g, z, begin, end, po, intervention)
                    total = total + raw.sum(-1); squares = squares + raw.square().sum(-1)
                    abs_total = abs_total + raw.abs().sum(-1); nonzero = nonzero + (raw != 0).sum(-1)
                    minimum = torch.minimum(minimum, raw.min(-1).values); maximum = torch.maximum(maximum, raw.max(-1).values)
                mean = total / max(1, g.num_pairs)
                norm = lambda t: t.square().sum((-2, -1)).sqrt()
                details = {"q_diag": diag_flow.detach(), "q_cross": cross_flow.detach(),
                           "message_diag": message_diag.detach(), "message_cross": message_cross.detach(),
                           "energy_intra": intra, "energy_cross": cross, "energy_total": intra + cross,
                           "raw_local_energy": raw_local, "corrected_local_energy": corrected_local,
                           "diag_multiplier": multiplier.detach(), "diagonal_min": multiplier.min(-1).values if g.num_edges else None,
                           "diagonal_max": multiplier.max(-1).values if g.num_edges else None,
                           "diagonal_mean": multiplier.mean(-1) if g.num_edges else None,
                           "diagonal_std": multiplier.std(-1, correction=0) if g.num_edges else None,
                           "pair_mean": mean if g.num_pairs else None,
                           "pair_std": (squares / max(1, g.num_pairs) - mean.square()).clamp_min(0).sqrt() if g.num_pairs else None,
                           "pair_abs_mean": abs_total / g.num_pairs if g.num_pairs else None,
                           "pair_min": minimum if g.num_pairs else None, "pair_max": maximum if g.num_pairs else None,
                           "pair_nonzero_count": nonzero, "eligible_pair_count": g.num_pairs,
                           "pair_statistic_space": "orientation_invariant_raw_tanh_coefficient",
                           "cross_active": active, "message_norm": norm(output.detach()),
                           "diagonal_message_norm": norm(message_diag), "cross_message_norm": norm(message_cross),
                           "energy_operator": "occurrence_corrected_edge_metric", "epsilon": self.epsilon, "rho": self.rho}
        if bare:
            output = output.squeeze(0)
            details = {k: v.squeeze(0) if isinstance(v, Tensor) else v for k, v in details.items()}
        return output, details

    forward = apply


def analytic_teacher(g, z, *, pair_chunk=None, diagnostics=False):
    """Locked Teacher3 rule; each realization keeps its scalar feature axis."""
    bare = z.ndim == 2
    packed = z.unsqueeze(0) if bare else z
    gate = EdgeMetricGate("F0", tuple(range(packed.shape[0]))).to(device=z.device, dtype=z.dtype)
    size = chunk_size(pair_chunk, g.num_pairs)
    chunks = []
    for begin in range(0, g.num_pairs, size):
        end = min(begin + size, g.num_pairs)
        feature = pair_features(g, packed, begin, end)
        chunks.append((2 * feature[..., 2] + .5 * (2 * feature[..., 7] - 1)).tanh())
    raw = torch.cat(chunks, -1) if chunks else packed.new_zeros(packed.shape[:-2] + (0,))
    if bare:
        raw = raw.squeeze(0)
    return gate.apply(g, z, z, pair_chunk=pair_chunk, diagnostics=diagnostics, pair_override=raw)


__all__ = ["CONDITIONS", "INTERVENTIONS", "EPSILON", "EdgeMetricGate", "pair_features", "physical_edge_features", "analytic_teacher"]
