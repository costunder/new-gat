"""Read-only observations for symmetric relative-C, never the old row-normalized rho.

Training observations attach to the actual forward. Interventions run only after
validation checkpoint selection, with no optimizer step or test-label metric involved.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..ablation.model import state_sha256
from ..benchmark import _binary_counts, _micro_f1_from_counts


@contextmanager
def evaluation_mode(model: nn.Module):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode


def norm(parameters, *, gradient: bool = False) -> float | None:
    total, present = 0.0, False
    for parameter in parameters:
        value = parameter.grad if gradient else parameter.detach()
        if value is None:
            continue
        value = value.detach().double()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite parameter/task gradient observation")
        total += float(value.square().sum())
        present = True
    return math.sqrt(total) if present else None


def gate_parameters(operator):
    return [
        value
        for name, value in operator.estimator.named_parameters()
        if name not in {"raw_gamma", "raw_tau"}
    ]


@torch.no_grad()
def moments(value: Tensor, *, quantiles: bool = False) -> dict[str, Any]:
    flat = value.detach().flatten().double()
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("Nonfinite v3 observation")
    result = {
        "count": flat.numel(),
        "mean": None,
        "std": None,
        "cv": None,
        "min": None,
        "max": None,
    }
    if not flat.numel():
        return result
    mean, std = float(flat.mean()), float(flat.std(correction=0))
    result.update(
        mean=mean,
        std=std,
        cv=std / abs(mean) if mean else None,
        min=float(flat.min()),
        max=float(flat.max()),
    )
    if quantiles:
        # Official transductive graphs have fewer than torch.quantile's 2**24
        # element limit. Refuse to silently label a sampled median as exact.
        if flat.numel() > 2**24:
            raise ValueError("Exact diagnostic quantile exceeds supported tensor size")
        values = torch.quantile(flat, flat.new_tensor([0.1, 0.5, 0.9, 0.99])).tolist()
        result["quantiles"] = dict(zip(("p10", "p50", "p90", "p99"), values, strict=True))
        result["quantile_policy"] = "exact_population"
    return result


class ForwardObservation:
    """Observe actual v3 score/C/state without extra forward/backward or RNG use."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.records: dict[int, dict[str, Any]] = {}
        self.conductances: dict[int, Tensor] = {}

    def __enter__(self):
        try:
            for index, operator in enumerate(self.model.operators):
                self.handles.append(
                    operator.estimator.register_forward_hook(
                        lambda module, inputs, output, i=index: self._conductance(i, module, output)
                    )
                )
                self.handles.append(
                    operator.register_forward_hook(
                        lambda module, inputs, output, i=index: self._operator(
                            i, module, inputs, output
                        )
                    )
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.conductances.clear()

    @torch.no_grad()
    def _conductance(self, index: int, estimator, c: Tensor):
        value = c.detach()
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise FloatingPointError("Observed C must be finite and positive")
        scores = estimator.last_scores
        if scores is None or scores.shape != value.shape:
            raise RuntimeError("V3 estimator did not expose aligned actual-forward scores")
        self.records[index] = {
            "layer": index,
            "score": moments(scores),
            "conductance": moments(value),
            "log_conductance": moments(value.log()),
            "gamma": (
                float(estimator.gamma.detach()) if estimator.gamma is not None else None
            ),
            "tau": float(estimator.tau.detach()) if estimator.tau is not None else None,
            "estimator_trainable": any(p.requires_grad for p in estimator.parameters()),
            "estimator_parameter_count": sum(
                parameter.numel() for parameter in estimator.parameters()
            ),
            "parameter_free_fixed_control": estimator.gate_mode == "fixed_one",
        }
        self.conductances[index] = value

    @torch.no_grad()
    def _operator(self, index: int, operator, inputs, output: Tensor):
        state, incidence = inputs[:2]
        c = self.conductances.pop(index)
        tail, head = incidence
        degree = c.new_zeros(state.shape[0])
        degree.index_add_(0, tail, c)
        degree.index_add_(0, head, c)
        positive = degree > 0
        degree_stats = moments(degree, quantiles=True)
        positive_stats = moments(degree[positive], quantiles=True)
        median = degree_stats.get("quantiles", {}).get("p50")
        degree_stats.update(
            positive_count=int(positive.sum()),
            positive_quantiles=positive_stats.get("quantiles", {}),
            max_over_median=degree_stats["max"] / median if median else None,
        )
        inv = torch.where(positive, degree, torch.ones_like(degree)).rsqrt() * positive
        neighbor_sum = torch.zeros_like(degree)
        edge_weight = c * inv[tail] * inv[head]
        neighbor_sum.index_add_(0, tail, edge_weight)
        neighbor_sum.index_add_(0, head, edge_weight)
        alpha = float(operator.alpha.detach())
        before_norm = float(state.detach().double().norm())
        change = float((output.detach().double() - state.detach().double()).norm())
        self.records[index].update(
            alpha=alpha,
            weighted_degree=degree_stats,
            neighbor_weight_row_sum=moments(alpha * neighbor_sum, quantiles=True),
            relative_conv_change=change / before_norm if before_norm else None,
            gate_parameter_norm=norm(gate_parameters(operator)),
            gate_gradient_norm=None,
        )

    def summary(self, *, gradients: bool = False):
        if set(self.records) != set(range(len(self.model.operators))):
            raise RuntimeError("Missing actual-forward layer observation")
        output = []
        for index in range(len(self.model.operators)):
            record = dict(self.records[index])
            if gradients:
                record["gate_gradient_norm"] = norm(
                    gate_parameters(self.model.operators[index]), gradient=True
                )
            output.append(record)
        return output


def evaluate_validation(
    model,
    data,
    indices: Tensor | None,
    *,
    observe: bool = True,
    device: torch.device | None = None,
):
    """Evaluate one fixed graph or the complete official two-graph PPI validation split."""
    if indices is None:
        if not isinstance(data, dict) or "validation" not in data:
            raise ValueError("PPI validation loader is missing")
        loader = data["validation"]
        if len(loader) != 1:
            raise ValueError("PPI validation must be its two official graphs in one batch")
        if device is None:
            device = next(model.parameters()).device
        graph = next(iter(loader)).to(device, non_blocking=True)
        if int(getattr(graph, "num_graphs", 0)) != 2:
            raise ValueError("PPI validation must contain both official validation graphs")
        with evaluation_mode(model):
            if observe:
                with ForwardObservation(model) as observation:
                    logits = model(graph)
                layers = observation.summary()
            else:
                logits, layers = model(graph), []
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite validation logits")
            labels = graph.y
            counts = _binary_counts(logits, labels)
            result = {
                "metric": _micro_f1_from_counts(counts),
                "metric_name": "micro_f1",
                "prediction_rule": "logit_gt_zero_node_label",
                "loss": float(F.binary_cross_entropy_with_logits(logits, labels)),
                "layers": layers,
                "mode": "eval",
                "split": "validation",
                "validation_graph_count": 2,
                "label_decision_count": labels.numel(),
                "prediction_unit": "node_label_decision",
                "observation_scope": (
                    "all official PPI validation graphs; global node-label micro-F1"
                ),
            }
        return result, logits.detach().cpu()
    graph = data
    if not indices.numel():
        raise ValueError("Validation mask is empty")
    with evaluation_mode(model):
        if observe:
            with ForwardObservation(model) as observation:
                full_logits = model(graph)
            layers = observation.summary()
        else:
            full_logits, layers = model(graph), []
        logits = full_logits.index_select(0, indices)
        labels = graph.y.index_select(0, indices)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite validation logits")
        result = {
            "metric": int((logits.argmax(-1) == labels).sum()) / indices.numel(),
            "metric_name": "accuracy",
            "prediction_rule": "argmax_node_class",
            "loss": float(F.cross_entropy(logits, labels)),
            "layers": layers,
            "mode": "eval",
            "split": "validation",
            "validation_graph_count": 1,
            "label_decision_count": indices.numel(),
            "prediction_unit": "node",
            "observation_scope": "whole transductive graph states; validation labels only",
        }
    return result, logits.detach().cpu()


def changed_prediction_fraction(logits: Tensor, reference: Tensor, metric_name: str) -> float:
    if logits.shape != reference.shape or not logits.numel():
        raise ValueError("Intervention/reference logits must be nonempty and aligned")
    if metric_name == "micro_f1":
        changed = (logits > 0) != (reference > 0)
    elif metric_name == "accuracy":
        changed = logits.argmax(-1) != reference.argmax(-1)
    else:
        raise ValueError("Unsupported intervention prediction metric")
    return float(changed.double().mean())


class Intervention:
    """Temporary output hooks; the actual operator recomputes D_C after replacing C."""

    def __init__(self, model, name: str, seed: int):
        if name not in {"mean_c", "shuffled_c", "ones_c", "propagation_off"}:
            raise ValueError("Unsupported v3 intervention")
        self.model, self.name, self.seed, self.handles = model, name, seed, []

    def __enter__(self):
        try:
            for index, operator in enumerate(self.model.operators):
                if self.name == "propagation_off":
                    self.handles.append(
                        operator.register_forward_hook(lambda module, inputs, output: inputs[0])
                    )
                else:
                    self.handles.append(
                        operator.estimator.register_forward_hook(
                            lambda module, inputs, output, i=index: self.replace(inputs, output, i)
                        )
                    )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def replace(self, inputs, c: Tensor, layer: int):
        if self.name == "ones_c":
            return torch.ones_like(c)
        _, incidence, node_graph, num_graphs = inputs
        edge_graph = node_graph[incidence[0]]
        result = c.clone()
        generator = torch.Generator(device=c.device).manual_seed(self.seed + 104729 * layer)
        for graph_index in range(num_graphs):
            ids = (edge_graph == graph_index).nonzero(as_tuple=False).flatten()
            if not ids.numel():
                continue
            values = c.index_select(0, ids)
            if self.name == "mean_c":
                result[ids] = values.mean()
            else:
                permutation = torch.randperm(ids.numel(), device=c.device, generator=generator)
                result[ids] = values[permutation]
        return result


def best_checkpoint_interventions(
    model,
    data,
    indices,
    original,
    reference: Tensor,
    *,
    seed: int,
    device: torch.device | None = None,
):
    """All-layer read-only interventions, only on the selected best checkpoint."""
    before = state_sha256(model)
    modes = [module.training for module in model.modules()]
    gradients = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    rows = []
    try:
        for name in ("mean_c", "shuffled_c", "ones_c", "propagation_off"):
            with Intervention(model, name, seed):
                result, logits = evaluate_validation(
                    model, data, indices, observe=False, device=device
                )
            difference = logits.double() - reference.double()
            rows.append(
                {
                    "intervention": name,
                    "validation": result["metric"],
                    "loss": result["loss"],
                    "percentage_points": 100 * (result["metric"] - original["metric"]),
                    "score_delta": result["metric"] - original["metric"],
                    "logit_mean_absolute_delta": float(difference.abs().mean()),
                    "logit_max_absolute_delta": float(difference.abs().max()),
                    "changed_prediction_fraction": changed_prediction_fraction(
                        logits, reference, original["metric_name"]
                    ),
                    "prediction_unit": original["prediction_unit"],
                    "prediction_rule": original["prediction_rule"],
                }
            )
    finally:
        if state_sha256(model) != before or modes != [
            module.training for module in model.modules()
        ]:
            raise RuntimeError("Interventions changed model state or training modes")
        for name, parameter in model.named_parameters():
            old = gradients[name]
            if (old is None) != (parameter.grad is None) or (
                old is not None and not torch.equal(old, parameter.grad)
            ):
                raise RuntimeError("Interventions changed a parameter gradient")
    return {
        "status": "passed",
        "scope": "validation_selected_best_checkpoint_only",
        "layers": "all_layers_simultaneously",
        "original": {"validation": original["metric"], "loss": original["loss"]},
        "validation_graph_count": original["validation_graph_count"],
        "prediction_unit": original["prediction_unit"],
        "prediction_rule": original["prediction_rule"],
        "rows": rows,
        "shuffle_seed": seed,
        "normalization_recomputed": True,
        "mean_ones_note": (
            "Graph-constant positive C cancels under symmetric normalization; "
            "mean-C and C=1 are redundant up to rounding."
        ),
        "interpretation": (
            "Checkpoint reliance, not a retrained-model benefit; "
            "no optimizer step or test evaluation."
        ),
    }
