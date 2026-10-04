"""Matched full-graph classifiers using actual local-copy coupling updates.

The leading dimension packs independent seeds. All physical nodes, local
copies, induced local edges and shared-copy links take part in every macro
layer. Edge chunks change only temporary storage. Their custom autograd
backward applies the same symmetric fixed Laplacian rather than retaining
feature-sized edge activations. No learned edge conductance is claimed.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from ..model import PackedClassifier as _ParentClassifier
from ..operators import CouplingGeometry, merge, replicate
from ...wedge_propagation.classification.model import _integer

WEIGHTS = ("unit", "local_degree")
VARIANTS = ("off", "fixed", "learned")
CONDITIONS = tuple(f"{mode}__{variant}" for mode in WEIGHTS for variant in VARIANTS)
INTERVENTIONS = ("gain0", "gain1")
SCOPES = {"layer_0": (0,), "layer_1": (1,), "both": (0, 1)}


def parse_condition(condition: str) -> tuple[str, str]:
    if not isinstance(condition, str) or condition not in CONDITIONS:
        raise ValueError(f"unknown local-context classification condition {condition!r}")
    mode, variant = condition.split("__")
    return mode, variant


def _edge_fields(geometry: CouplingGeometry, kind: str):
    if kind == "intra":
        left, right = geometry.topology.local_edge_nodes
        return left, right, geometry.intra_weights
    if kind == "cross":
        return geometry.cross_left, geometry.cross_right, None
    raise ValueError("Laplacian kind must be intra or cross")


def _laplacian_chunks(geometry: CouplingGeometry, value: Tensor, kind: str, chunk: int | None) -> Tensor:
    left, right, weights = _edge_fields(geometry, kind)
    count = left.numel()
    size = count if chunk is None else chunk
    result = torch.zeros_like(value)
    if count:
        for begin in range(0, count, size):
            end = min(begin + size, count)
            tail, head = left[begin:end], right[begin:end]
            flow = value.index_select(1, head) - value.index_select(1, tail)
            if weights is not None:
                flow = flow * weights[None, begin:end, None]
            result.index_add_(1, tail, -flow)
            result.index_add_(1, head, flow)
    return result


class _FixedLaplacian(Function):
    @staticmethod
    def forward(ctx, value, geometry, kind, chunk):
        ctx.geometry, ctx.kind, ctx.chunk = geometry, kind, chunk
        return _laplacian_chunks(geometry, value, kind, chunk)

    @staticmethod
    def backward(ctx, grad):
        # Both A and K are symmetric; their exact transpose is the same action.
        return _FixedLaplacian.apply(grad, ctx.geometry, ctx.kind, ctx.chunk), None, None, None


def _norm(value: Tensor) -> Tensor:
    return value.detach().square().flatten(1).sum(1).sqrt()


def _energy(geometry: CouplingGeometry, value: Tensor, kind: str, chunk: int | None) -> Tensor:
    left, right, weights = _edge_fields(geometry, kind)
    result = value.new_zeros(value.shape[0])
    count, size = left.numel(), left.numel() if chunk is None else chunk
    if count:
        for begin in range(0, count, size):
            end = min(begin + size, count)
            difference = value.index_select(1, right[begin:end]) - value.index_select(1, left[begin:end])
            squared = difference.square()
            if weights is not None:
                squared = squared * weights[None, begin:end, None]
            result = result + .5 * squared.flatten(1).sum(1)
    return result


class PackedClassifier(_ParentClassifier):
    """Two macro layers, with one bounded gain shared across layers per seed.

    Projection initialization and counter dropout are inherited unchanged from
    the previous verified model. Off and fixed variants have no unused gain
    parameter. Frozen interventions replace the effective gain in selected
    layers and recompute the rest of the forward; they do not edit parameters.
    """

    def __init__(
        self, condition: str, input_dim: int, classes: int, seeds,
        hidden: int = 64, dropout: float = .5, dataset_name: str = "unspecified",
        edge_chunk: int | None = None, path_chunk: int | None = None,
        checkpoint_paths: bool = True,
    ):
        mode, variant = parse_condition(condition)
        if edge_chunk is not None and path_chunk is not None and edge_chunk != path_chunk:
            raise ValueError("edge_chunk and path_chunk aliases disagree")
        selected_chunk = edge_chunk if edge_chunk is not None else path_chunk
        selected_chunk = None if selected_chunk is None else _integer(selected_chunk, "edge_chunk")
        if not isinstance(checkpoint_paths, bool):
            raise TypeError("checkpoint_paths must be boolean")
        super().__init__(
            input_dim, classes, seeds, hidden=hidden, dropout=dropout,
            dataset_name=dataset_name, mode=mode,
            condition="cross_on" if variant == "learned" else "cross_off",
        )
        self.condition, self.variant, self.weight_mode = condition, variant, mode
        self.edge_chunk = self.path_chunk = selected_chunk
        self.checkpoint_paths = checkpoint_paths

    @property
    def cross_gain(self) -> Tensor:
        if self.variant == "learned":
            return self.theta_cross.sigmoid()
        if self.variant == "fixed":
            return self.weights[0].new_ones(len(self.seeds))
        return self.weights[0].new_zeros(len(self.seeds))

    @property
    def rho(self) -> Tensor:
        return self.cross_gain

    def _action(self, geometry, value, kind):
        return _FixedLaplacian.apply(value, geometry, kind, self.edge_chunk)

    def _stages(self, geometry, z, gain, cross_active=True):
        eta = geometry.eta.index_select(0, geometry.copy_graph)[None, :, None]
        gamma = geometry.gamma.index_select(0, geometry.copy_graph)[None, :, None]
        y0 = replicate(geometry, z)
        y1 = y0 - eta * self._action(geometry, y0, "intra")
        y2 = y1 - gain[:, None, None] * gamma * self._action(geometry, y1, "cross") if cross_active else y1
        y3 = y2 - eta * self._action(geometry, y2, "intra")
        return merge(geometry, y3), (y0, y1, y2, y3)

    def _macro(self, geometry, z, gain, cross_active=True):
        return self._stages(geometry, z, gain, cross_active)[0]

    def _diagnostics(self, geometry, z, output, stages, gain):
        y0, y1, y2, y3 = (value.detach() for value in stages)
        eta = geometry.eta.index_select(0, geometry.copy_graph)[None, :, None]
        off = merge(geometry, y1 - eta * self._action(geometry, y1, "intra"))
        base_norm = _norm(off)
        matched = _norm(output.detach() - off)
        nonzero = base_norm > 0
        denominator = torch.where(nonzero, base_norm, torch.ones_like(base_norm))
        result = {
            "projected_norm": _norm(z), "output_norm": _norm(output), "off_norm": base_norm,
            "matched_delta_norm": matched, "matched_delta_relative": matched / denominator,
            "off_nonzero": nonzero,
            "context_norm": _norm(self._action(geometry, y1, "cross")),
            "cross_energy_before": _energy(geometry, y1, "cross", self.edge_chunk),
            "cross_energy_after": _energy(geometry, y2, "cross", self.edge_chunk),
            "intra_energy_before": _energy(geometry, y0, "intra", self.edge_chunk),
            "intra_energy_after": _energy(geometry, y3, "intra", self.edge_chunk),
            "rho": gain.detach(), "rho_original": self.cross_gain.detach(),
            "theta_available": self.theta_cross is not None,
        }
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
            geometry = graph.geometry_for(self.weight_mode) if hasattr(graph, "geometry_for") else graph.geometry
        if not isinstance(geometry, CouplingGeometry) or geometry.mode != self.mode:
            raise ValueError("geometry must match the model's weight mode")
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape != (geometry.n, self.input_dim):
            raise ValueError("x must be [all physical nodes,input_dim]")
        if x.device != self.weights[0].device or x.dtype != self.weights[0].dtype:
            raise ValueError("input and model must share device and dtype")
        h = x.unsqueeze(0).expand(len(self.seeds), -1, -1)
        original_gain, details = self.cross_gain, []
        for layer, weight in enumerate(self.weights):
            z = torch.bmm(self._dropout(h, int(epoch), layer), weight)
            gain = original_gain
            cross_active = self.variant != "off"
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
                h = torch.relu(h)
        return h, details
