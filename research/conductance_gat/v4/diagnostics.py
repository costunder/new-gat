"""Read-only observations for the V4 conductance/spatial factorial.

Training observations attach to the actual full-graph forward.  Interventions
run only after validation checkpoint selection and never update parameters or
inspect test labels.  Replacing estimator output means that the operator's
normal symmetric-normalization path recomputes the C-dependent degrees.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..ablation.model import state_sha256


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


def _rng_snapshot():
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _same_rng_state(before) -> bool:
    if not torch.equal(before["cpu"], torch.random.get_rng_state()):
        return False
    current_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return len(before["cuda"]) == len(current_cuda) and all(
        torch.equal(old, current) for old, current in zip(before["cuda"], current_cuda, strict=True)
    )


def gate_parameters(operator):
    return [
        value
        for name, value in operator.estimator.named_parameters()
        if name not in {"raw_gamma", "raw_tau"}
    ]


def spatial_parameters(operator):
    return list(operator.message_transform.parameters())


@torch.no_grad()
def moments(value: Tensor, *, quantiles: bool = False) -> dict[str, Any]:
    flat = value.detach().flatten().double()
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("Nonfinite V4 observation")
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
        if flat.numel() > 2**24:
            raise ValueError("Exact diagnostic quantile exceeds supported tensor size")
        values = torch.quantile(flat, flat.new_tensor([0.1, 0.5, 0.9, 0.99])).tolist()
        result["quantiles"] = dict(zip(("p10", "p50", "p90", "p99"), values, strict=True))
        result["quantile_policy"] = "exact_population"
    return result


@torch.no_grad()
def spatial_weight_statistics(operator) -> dict[str, Any]:
    weight = operator.message_transform.weight.detach().double()
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("V4 message transform must be a square matrix")
    if not bool(torch.isfinite(weight).all()):
        raise FloatingPointError("Nonfinite V4 spatial message transform")
    identity = torch.eye(weight.shape[0], dtype=weight.dtype, device=weight.device)
    distance = float((weight - identity).norm())
    singular = torch.linalg.svdvals(weight)
    minimum, maximum = float(singular.min()), float(singular.max())
    return {
        "spatial_mode": operator.spatial_mode,
        "trainable": any(p.requires_grad for p in operator.message_transform.parameters()),
        "parameter_norm": norm(spatial_parameters(operator)),
        "identity_distance_frobenius": distance,
        "identity_relative_distance": distance / math.sqrt(weight.shape[0]),
        "singular_values": {
            "count": singular.numel(),
            "min": minimum,
            "max": maximum,
            "mean": float(singular.mean()),
            "std": float(singular.std(correction=0)),
            "condition_number": maximum / minimum if minimum else None,
        },
    }


class ForwardObservation:
    """Observe actual V4 C, message transform, and propagation without extra RNG use."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.records: dict[int, dict[str, Any]] = {}
        self.conductances: dict[int, Tensor] = {}
        self.messages: dict[int, Tensor] = {}

    def __enter__(self):
        try:
            for index, operator in enumerate(self.model.operators):
                self.handles.append(
                    operator.estimator.register_forward_hook(
                        lambda module, inputs, output, i=index: self._conductance(i, module, output)
                    )
                )
                self.handles.append(
                    operator.message_transform.register_forward_hook(
                        lambda module, inputs, output, i=index: self._message(i, inputs, output)
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
        self.messages.clear()

    @torch.no_grad()
    def _conductance(self, index: int, estimator, c: Tensor):
        value = c.detach()
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise FloatingPointError("Observed C must be finite and positive")
        scores = estimator.last_scores
        if scores is None or scores.shape != value.shape:
            raise RuntimeError("V4 estimator did not expose aligned actual-forward scores")
        self.records[index] = {
            "layer": index,
            "score": moments(scores),
            "conductance": moments(value),
            "log_conductance": moments(value.log()),
            "gamma": float(estimator.gamma.detach()),
            "tau": float(estimator.tau.detach()),
            "estimator_trainable": any(p.requires_grad for p in estimator.parameters()),
        }
        self.conductances[index] = value

    @torch.no_grad()
    def _message(self, index: int, inputs, output: Tensor):
        state = inputs[0]
        if output.shape != state.shape:
            raise RuntimeError("V4 spatial transform changed the node-state shape")
        if not bool(torch.isfinite(output.detach()).all()):
            raise FloatingPointError("Nonfinite V4 spatial message")
        self.messages[index] = output.detach()

    @torch.no_grad()
    def _operator(self, index: int, operator, inputs, output: Tensor):
        state, incidence = inputs[:2]
        if index not in self.records or index not in self.conductances:
            raise RuntimeError("V4 operator ran without an aligned conductance observation")
        if index not in self.messages:
            raise RuntimeError("V4 operator ran without an aligned spatial-message observation")
        c = self.conductances.pop(index)
        message = self.messages.pop(index)
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
        message_norm = float(message.double().norm())
        change = float((output.detach().double() - state.detach().double()).norm())
        message_change = float((message.double() - state.detach().double()).norm())
        self.records[index].update(
            alpha=alpha,
            weighted_degree=degree_stats,
            neighbor_weight_row_sum=moments(alpha * neighbor_sum, quantiles=True),
            state_norm=before_norm,
            message_norm=message_norm,
            relative_message_transform_change=(
                message_change / before_norm if before_norm else None
            ),
            relative_conv_change=change / before_norm if before_norm else None,
            gate_parameter_norm=norm(gate_parameters(operator)),
            gate_gradient_norm=None,
            spatial_weight=spatial_weight_statistics(operator),
            spatial_gradient_norm=None,
        )

    def summary(self, *, gradients: bool = False):
        if set(self.records) != set(range(len(self.model.operators))):
            raise RuntimeError("Missing actual-forward V4 layer observation")
        output = []
        for index in range(len(self.model.operators)):
            record = dict(self.records[index])
            if gradients:
                operator = self.model.operators[index]
                record["gate_gradient_norm"] = norm(gate_parameters(operator), gradient=True)
                record["spatial_gradient_norm"] = norm(spatial_parameters(operator), gradient=True)
            output.append(record)
        return output


def evaluate_validation(model, graph, indices: Tensor, *, observe: bool = True):
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
            "loss": float(F.cross_entropy(logits, labels)),
            "layers": layers,
            "mode": "eval",
            "split": "validation",
            "observation_scope": "whole transductive graph states; validation labels only",
        }
    return result, logits.detach().cpu()


class Intervention:
    """Temporary output hooks for selected-checkpoint read-only interventions."""

    NAMES = {
        "mean_c",
        "shuffled_c",
        "ones_c",
        "identity_w",
        "ones_c_identity_w",
        "propagation_off",
    }

    def __init__(self, model, name: str, seed: int):
        if name not in self.NAMES:
            raise ValueError("Unsupported V4 intervention")
        self.model, self.name, self.seed, self.handles = model, name, seed, []

    def __enter__(self):
        try:
            for index, operator in enumerate(self.model.operators):
                if self.name == "propagation_off":
                    self.handles.append(
                        operator.register_forward_hook(lambda module, inputs, output: inputs[0])
                    )
                    continue
                if self.name in {"mean_c", "shuffled_c", "ones_c", "ones_c_identity_w"}:
                    self.handles.append(
                        operator.estimator.register_forward_hook(
                            lambda module, inputs, output, i=index: self.replace_c(
                                inputs, output, i
                            )
                        )
                    )
                if self.name in {"identity_w", "ones_c_identity_w"}:
                    self.handles.append(
                        operator.message_transform.register_forward_hook(
                            lambda module, inputs, output: inputs[0]
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

    def replace_c(self, inputs, c: Tensor, layer: int):
        if self.name in {"ones_c", "ones_c_identity_w"}:
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


def _intervention_row(name, result, logits, original, reference):
    difference = logits.double() - reference.double()
    return {
        "intervention": name,
        "intervention_kind": "read_only_selected_checkpoint",
        "fresh_training": False,
        "validation": result["metric"],
        "loss": result["loss"],
        "percentage_points": 100 * (result["metric"] - original["metric"]),
        "score_delta": result["metric"] - original["metric"],
        "logit_mean_absolute_delta": float(difference.abs().mean()),
        "logit_max_absolute_delta": float(difference.abs().max()),
        "changed_prediction_fraction": float(
            (logits.argmax(-1) != reference.argmax(-1)).double().mean()
        ),
    }


def best_checkpoint_interventions(model, graph, indices, original, reference: Tensor, *, seed: int):
    """All-layer read-only interventions, only on the selected best checkpoint."""
    before = state_sha256(model)
    modes = [module.training for module in model.modules()]
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    rng = _rng_snapshot()
    names = (
        "mean_c",
        "shuffled_c",
        "ones_c",
        "identity_w",
        "ones_c_identity_w",
        "propagation_off",
    )
    rows, intervention_logits = [], {}
    try:
        for name in names:
            with Intervention(model, name, seed):
                result, logits = evaluate_validation(model, graph, indices, observe=False)
            intervention_logits[name] = logits
            rows.append(_intervention_row(name, result, logits, original, reference))
    finally:
        if state_sha256(model) != before or modes != [
            module.training for module in model.modules()
        ]:
            raise RuntimeError("Interventions changed model state or training modes")
        if not _same_rng_state(rng):
            raise RuntimeError("Interventions changed a global CPU/CUDA RNG state")
        for name, parameter in model.named_parameters():
            old = gradients[name]
            if (old is None) != (parameter.grad is None) or (
                old is not None and not torch.equal(old, parameter.grad)
            ):
                raise RuntimeError("Interventions changed a parameter gradient")
    mean_logits = intervention_logits["mean_c"].double()
    ones_logits = intervention_logits["ones_c"].double()
    numeric_difference = mean_logits - ones_logits
    numeric_check = {
        "comparison": "mean_c_vs_ones_c",
        "allclose_rtol": 1e-5,
        "allclose_atol": 1e-6,
        "passed": bool(torch.allclose(mean_logits, ones_logits, rtol=1e-5, atol=1e-6)),
        "logit_mean_absolute_delta": float(numeric_difference.abs().mean()),
        "logit_max_absolute_delta": float(numeric_difference.abs().max()),
        "changed_prediction_fraction": float(
            (mean_logits.argmax(-1) != ones_logits.argmax(-1)).double().mean()
        ),
    }
    return {
        "status": "passed",
        "scope": "validation_selected_best_checkpoint_only",
        "layers": "all_layers_simultaneously",
        "original": {"validation": original["metric"], "loss": original["loss"]},
        "rows": rows,
        "shuffle_seed": seed,
        "normalization_recomputed_for_c_interventions": True,
        "mean_c_numeric_check": numeric_check,
        "mean_ones_note": (
            "Graph-constant positive C cancels under symmetric normalization; mean-C and "
            "C=1 should agree up to floating-point rounding. This is a numerical check, "
            "not an independent causal intervention."
        ),
        "interpretation": (
            "Checkpoint reliance, not a retrained-model benefit; no optimizer step or test "
            "evaluation. Fresh factorial arms, not these rows, estimate learned-component "
            "contrasts."
        ),
    }
