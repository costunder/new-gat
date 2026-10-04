"""Independent matched classifiers for graph/local intra and graph/edge G.

All twenty conditions use the same two macro layers, projections and dropout
streams. The frozen normalized S/G actions are applied directly to copy states.
One bounded rho per seed is learned only in learned variants and shared by the
two layers. No scalar energy lift or learned edge C is introduced.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..classification.model import PackedClassifier as _ReferenceClassifier
from ..classification.model import INTERVENTIONS, SCOPES, _norm
from ..operators import merge, replicate
from .common import CONDITIONS, condition_metadata, parse_condition
from .operators import NormalizedGeometry, apply_s, apply_g, apply_raw_k, energy_s, energy_g


class PackedClassifier(_ReferenceClassifier):
    def __init__(
        self, condition: str, input_dim: int, classes: int, seeds,
        hidden: int = 64, dropout: float = .5, dataset_name: str = "unspecified",
        edge_chunk: int | None = None, path_chunk: int | None = None,
        checkpoint_paths: bool = True,
    ):
        mode, intra, cross, variant = parse_condition(condition)
        super().__init__(
            f"{mode}__{variant}", input_dim, classes, seeds, hidden=hidden,
            dropout=dropout, dataset_name=dataset_name, edge_chunk=edge_chunk,
            path_chunk=path_chunk, checkpoint_paths=checkpoint_paths,
        )
        self.condition = condition
        self.intra_policy, self.cross_policy = intra, cross

    def _action(self, geometry, value, kind):
        function = {"intra": apply_s, "cross": apply_g, "raw_cross": apply_raw_k}.get(kind)
        if function is None:
            raise ValueError("normalized model action must be intra, cross or raw_cross")
        return function(geometry, value, self.edge_chunk)

    def _stages(self, geometry, z, gain, cross_active=True):
        y0 = replicate(geometry.base, z)
        y1 = y0 - apply_s(geometry, y0, self.edge_chunk)
        y2 = y1 - gain[:, None, None] * apply_g(geometry, y1, self.edge_chunk) if cross_active else y1
        y3 = y2 - apply_s(geometry, y2, self.edge_chunk)
        return merge(geometry.base, y3), (y0, y1, y2, y3)

    def _diagnostics(self, geometry, z, output, stages, gain):
        y0, y1, y2, y3 = (value.detach() for value in stages)
        off = merge(geometry.base, y1 - apply_s(geometry, y1, self.edge_chunk))
        base_norm, projected_norm = _norm(off), _norm(z)
        matched = _norm(output.detach() - off)
        off_nonzero, projected_nonzero = base_norm > 0, projected_norm > 0
        denominator = torch.where(off_nonzero, base_norm, torch.ones_like(base_norm))
        formula_denominator = torch.where(projected_nonzero, projected_norm, torch.ones_like(projected_norm))
        context = apply_g(geometry, y1, self.edge_chunk)
        # GR=MG=0 gives delta=rho M S G y1 = -rho M S G S R z.
        predicted = gain[:, None, None] * merge(geometry.base, apply_s(geometry, context, self.edge_chunk))
        residual = _norm(output.detach() - off - predicted)
        result = {
            **condition_metadata(self.condition), "energy_operator": "applied_S_G",
            "projected_norm": projected_norm, "output_norm": _norm(output), "off_norm": base_norm,
            "matched_delta_norm": matched, "matched_delta_relative": matched / denominator,
            "off_nonzero": off_nonzero, "projected_nonzero": projected_nonzero,
            "predicted_delta_norm": _norm(predicted), "formula_residual_norm": residual,
            "formula_relative_error": residual / formula_denominator,
            "context_norm": _norm(context),
            "raw_context_norm": _norm(apply_raw_k(geometry, y1, self.edge_chunk)),
            "cross_energy_before": energy_g(geometry, y1, self.edge_chunk).flatten(1).sum(1),
            "cross_energy_after": energy_g(geometry, y2, self.edge_chunk).flatten(1).sum(1),
            "intra_energy_before": energy_s(geometry, y0, self.edge_chunk).flatten(1).sum(1),
            "intra_energy_after": energy_s(geometry, y3, self.edge_chunk).flatten(1).sum(1),
            "rho": gain.detach(), "rho_original": self.cross_gain.detach(),
            "theta_available": self.theta_cross is not None,
        }
        for key in ("s_degree_max", "g_degree_max", "eta_max"):
            result[key] = gain.new_full(gain.shape, geometry.metadata[key])
        if self.theta_cross is not None:
            result["theta"] = self.theta_cross.detach()
        return result

    def forward(
        self, graph: Any, *, epoch: int = 0, diagnostics: bool = False,
        intervention: str | None = None, intervention_layers=(0, 1),
    ):
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if not isinstance(diagnostics, bool):
            raise TypeError("diagnostics must be boolean")
        if intervention is not None and intervention not in INTERVENTIONS:
            raise ValueError(f"unknown gain intervention {intervention!r}")
        layers = tuple(intervention_layers)
        if not layers or len(set(layers)) != len(layers) or any(
            isinstance(layer, bool) or not isinstance(layer, Integral) or layer not in (0, 1)
            for layer in layers
        ):
            raise ValueError("intervention_layers must select distinct layers from 0 and 1")
        if isinstance(graph, tuple) and len(graph) == 2:
            x, geometry = graph
        else:
            x = graph.x
            geometry = graph.geometry_for(self.weight_mode, self.intra_policy, self.cross_policy)
        effective_cross = "graph" if self.cross_policy == "none" else self.cross_policy
        if (not isinstance(geometry, NormalizedGeometry) or geometry.mode != self.mode
                or geometry.intra_policy != self.intra_policy
                or geometry.cross_policy not in (effective_cross, self.cross_policy)):
            raise ValueError("normalized geometry must match all declared operator policies")
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape != (geometry.n, self.input_dim):
            raise ValueError("x must be [all physical nodes,input_dim]")
        if (x.device != self.weights[0].device or x.dtype != self.weights[0].dtype
                or x.device != geometry.s_weights.device or x.dtype != geometry.s_weights.dtype):
            raise ValueError("input, coefficients and model must share device and dtype")
        h = x.unsqueeze(0).expand(len(self.seeds), -1, -1)
        original_gain, details = self.cross_gain, []
        for layer, weight in enumerate(self.weights):
            z = torch.bmm(self._dropout(h, int(epoch), layer), weight)
            gain, cross_active = original_gain, self.variant != "off"
            if intervention is not None and layer in layers:
                gain = original_gain.new_full(original_gain.shape, float(intervention == "gain1"))
                cross_active = intervention == "gain1"
            if diagnostics:
                h, stages = self._stages(geometry, z, gain, cross_active)
                with torch.no_grad():
                    details.append(self._diagnostics(geometry, z, h, stages, gain))
            elif self.checkpoint_paths and torch.is_grad_enabled():
                h = checkpoint(
                    lambda value, rho, geo=geometry, active=cross_active: self._macro(geo, value, rho, active),
                    z, gain, use_reentrant=False, preserve_rng_state=False,
                )
            else:
                h = self._macro(geometry, z, gain, cross_active)
            if layer == 0:
                h = h.relu()
        return h, details
