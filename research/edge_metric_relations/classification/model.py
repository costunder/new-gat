"""Adaptive edge metrics and three baselines with identical projection streams."""
from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...wedge_propagation.classification.model import _matrix, _named_seed, PackedClassifier as _Dropout
from ..gates import EdgeMetricGate
from .common import parse_condition, condition_metadata

SCOPES = {"layer_0": (0,), "layer_1": (1,), "both": (0, 1)}
INTERVENTIONS = ("no_op", "offdiag_zero", "diagonal_one", "pair_zero")


def _norm(x):
    return x.detach().square().flatten(1).sum(1).sqrt()


class PackedClassifier(nn.Module):
    def __init__(self, condition, input_dim, classes, seeds, *, hidden=64, dropout=.5,
                 dataset_name="unspecified", path_chunk=None, checkpoint_paths=True):
        super().__init__()
        self.mode, self.operator_family, self.cross_policy, self.variant = parse_condition(condition)
        self.condition, self.dataset_name = condition, dataset_name
        self.input_dim, self.classes, self.hidden = int(input_dim), int(classes), int(hidden)
        if min(self.input_dim, self.hidden) < 1 or self.classes < 2:
            raise ValueError("Invalid classifier dimensions")
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("Packed seeds must be distinct nonnegative integers")
        self.seeds, self.dropout = tuple(seeds), float(dropout)
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        if path_chunk is not None and (type(path_chunk) is not int or path_chunk < 1):
            raise ValueError("Chunk must be positive")
        self.num_layers, self.path_chunk, self.checkpoint_paths = 2, path_chunk, checkpoint_paths
        self.weights = nn.ParameterList([
            _matrix(self.seeds, dataset_name, "edge-metric.layer0", input_dim, hidden),
            _matrix(self.seeds, dataset_name, "edge-metric.layer1", hidden, classes),
        ])
        self.register_buffer("dropout_keys", torch.tensor([
            [_named_seed(dataset_name, seed, f"edge-metric.dropout.{l}") & 0xFFFFFFFF for seed in self.seeds]
            for l in range(2)], dtype=torch.int64), persistent=False)
        self.gates = nn.ModuleList([
            EdgeMetricGate(self.variant, self.seeds, hidden=64, epsilon=1e-4,
                           namespace=f"{dataset_name}.edge-metric.gate{l}")
            for l in range(2)
        ]) if self.operator_family == "edge_metric" else nn.ModuleList()
        self.polynomial = nn.ParameterList([
            nn.Parameter(torch.tensor([[1., -1/3, 0.]] * len(seeds))) for _ in range(2)
        ]) if self.variant == "P2" else nn.ParameterList()

    _dropout = _Dropout._dropout

    @property
    def parameters_per_seed(self):
        return sum(p.numel() for p in self.parameters()) // len(self.seeds)

    @property
    def trainable_parameters_per_seed(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) // len(self.seeds)

    def weight_decay_groups(self, weight_decay):
        extra = [p for name, p in self.named_parameters() if not name.startswith("weights.")]
        groups = [{"params": list(self.weights), "weight_decay": weight_decay}]
        if extra:
            groups.append({"params": extra, "weight_decay": 0.})
        return groups

    def _gcn(self, graph, z):
        output = torch.zeros_like(z)
        count = graph.gcn_edges.shape[1]
        size = count if self.path_chunk is None else self.path_chunk
        for start in range(0, count, size):
            end = min(start + size, count)
            source, target = graph.gcn_edges[:, start:end]
            message = z.index_select(1, source) * graph.gcn_weights[None, start:end, None]
            output = output.index_add(1, target, message)
        return output

    def _baseline(self, graph, z, layer):
        pz = self._gcn(graph, z)
        if self.variant == "G1":
            return pz
        if self.variant == "G2":
            return self._gcn(graph, pz)
        lz = z - pz
        l2z = lz - self._gcn(graph, lz)
        a = self.polynomial[layer]
        a0, a1, a2 = a.unbind(1)
        derivative_end = a1 + 4*a2
        inside = ((a1 < 0) & (derivative_end > 0)) | ((a1 > 0) & (derivative_end < 0))
        safe = torch.where(inside, 2*a2, torch.ones_like(a2))
        critical = -torch.where(inside, a1, torch.zeros_like(a1)) / safe
        # Evaluate outside candidates at a bounded dummy point to avoid overflow
        # from tiny a2. Only critical points in [0,2] enter the supremum.
        point = torch.where(inside, critical, torch.zeros_like(critical))
        critical_value = a0 + a1 * point + a2 * point.square()
        bounds = torch.stack((torch.ones_like(a0), a0.abs(), (a0 + 2*a1 + 4*a2).abs(),
                              torch.where(inside, critical_value.abs(), torch.zeros_like(a0))), 1)
        kappa = bounds.amax(1)
        return (a0[:, None, None]*z + a1[:, None, None]*lz + a2[:, None, None]*l2z) / kappa[:, None, None]

    def _metric(self, graph, z, layer, intervention=None, diagnostics=False):
        geometry = graph.geometry_for(self.mode)
        value = geometry.n0[None, :, None] * z
        q, detail = self.gates[layer].apply(geometry, z, value, pair_chunk=self.path_chunk,
                                          intervention=intervention, diagnostics=diagnostics)
        output = z - geometry.n0[None, :, None] * q / 3
        if not diagnostics:
            return output, {}
        mapped = {**condition_metadata(self.condition), "projected_norm": _norm(z),
                  "output_norm": _norm(output)}
        for key in ("energy_intra", "energy_cross", "energy_total"):
            if key in detail:
                mapped[key] = detail[key].detach().reshape(len(self.seeds), -1).sum(1)
        for key in ("message_diag", "message_cross"):
            if key in detail:
                mapped[key + "_norm"] = _norm(geometry.n0[None, :, None] * detail[key] / 3)
        # Pure offdiag removal at the same projected Z and the same diagonal coefficients.
        off, _ = self.gates[layer].apply(geometry, z, value, pair_chunk=self.path_chunk,
                                       intervention="offdiag_zero", diagnostics=False,
                                       diagonal_override=detail["diag_multiplier"])
        off_output = z - geometry.n0[None, :, None] * off / 3
        delta, denominator = _norm(output-off_output), _norm(off_output)
        mapped.update(matched_delta_norm=delta, off_norm=denominator,
                      off_nonzero=denominator > 0,
                      matched_delta_relative=delta / torch.where(denominator > 0, denominator, torch.ones_like(denominator)))
        for key in ("diagonal_mean", "diagonal_std", "pair_mean", "pair_std"):
            if detail.get(key) is not None:
                mapped[key] = detail[key].detach().reshape(len(self.seeds), -1).mean(1)
        return output, mapped

    def forward(self, graph, *, epoch=0, diagnostics=False, intervention=None,
                intervention_layers=(0, 1)):
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        if intervention is not None and intervention not in INTERVENTIONS:
            raise ValueError("Unknown frozen intervention")
        layers = tuple(intervention_layers)
        if not layers or len(set(layers)) != len(layers) or any(type(l) is not int or l not in (0, 1) for l in layers):
            raise ValueError("Select distinct valid classifier layers")
        x = graph.x
        if x.shape != (graph.num_nodes, self.input_dim) or x.device != self.weights[0].device or x.dtype != self.weights[0].dtype:
            raise ValueError("Full graph input/model shape, device or dtype differs")
        h = x[None].expand(len(self.seeds), -1, -1)
        details = []
        for layer, weight in enumerate(self.weights):
            z = torch.bmm(self._dropout(h, int(epoch), layer), weight)
            effective = intervention if layer in layers and intervention != "no_op" else None
            if self.operator_family == "edge_metric":
                if self.checkpoint_paths and torch.is_grad_enabled() and not diagnostics:
                    h = checkpoint(lambda v, l=layer, i=effective: self._metric(graph, v, l, i)[0],
                                   z, use_reentrant=False, preserve_rng_state=False)
                    detail = {}
                else:
                    h, detail = self._metric(graph, z, layer, effective, diagnostics)
            else:
                h = self._baseline(graph, z, layer)
                detail = {**condition_metadata(self.condition), "projected_norm": _norm(z), "output_norm": _norm(h)}
            if diagnostics:
                details.append(detail)
            if layer == 0:
                h = h.relu()
        return h, details
