"""Read-only observations for the V4 conductance/spatial factorial.

Training observations attach to the actual transductive full-graph forward or
PPI whole-graph minibatch. Interventions run only after validation checkpoint
selection and never update parameters or compute a test-label metric. Replacing
estimator output means that the operator's normal symmetric-normalization path
recomputes the C-dependent degrees.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
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
    if operator.spatial_mode == "fixed_identity":
        channels = operator.message_transform.in_features
        return {
            "spatial_mode": "fixed_identity",
            "trainable": False,
            "parameter_present": False,
            "parameter_norm": None,
            "identity_distance_frobenius": 0.0,
            "identity_relative_distance": 0.0,
            "singular_values": {
                "count": channels,
                "min": 1.0,
                "max": 1.0,
                "mean": 1.0,
                "std": 0.0,
                "condition_number": 1.0,
            },
        }
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
        "parameter_present": True,
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


def _prediction_tensor(logits: Tensor, prediction_rule: str) -> Tensor:
    if prediction_rule == "argmax_node_class":
        return logits.argmax(-1)
    if prediction_rule == "logit_gt_zero_node_label":
        return logits > 0
    raise ValueError("Unknown V4 prediction rule")


def _ppi_graph_count(graph) -> int:
    value = getattr(graph, "num_graphs", None)
    if value is not None:
        return int(value)
    batch = getattr(graph, "batch", None)
    return int(batch.max()) + 1 if isinstance(batch, Tensor) and batch.numel() else 1


def evaluate_validation(
    model,
    graph,
    indices: Tensor | None,
    *,
    observe: bool = True,
    device: torch.device | None = None,
):
    if indices is not None and not indices.numel():
        raise ValueError("Validation mask is empty")
    with evaluation_mode(model):
        context = ForwardObservation(model) if observe else nullcontext(None)
        with context as observation:
            if indices is not None:
                full_logits = model(graph)
                logits = full_logits.index_select(0, indices)
                labels = graph.y.index_select(0, indices)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite validation logits")
                metric = int((logits.argmax(-1) == labels).sum()) / indices.numel()
                loss = float(F.cross_entropy(logits, labels))
                graph_count = 1
                metric_name = "accuracy"
                prediction_rule = "argmax_node_class"
                observation_scope = (
                    "whole transductive graph states; validation labels only"
                )
            else:
                if not isinstance(graph, dict) or "validation" not in graph:
                    raise ValueError("PPI validation requires the official validation loader")
                batches = graph["validation"]
                if len(batches) != 1:
                    raise ValueError(
                        "PPI validation must pack its two official graphs into one batch"
                    )
                if device is None:
                    device = next(model.parameters()).device
                counts = torch.zeros(3, dtype=torch.int64, device=device)
                loss_sum = torch.zeros((), dtype=torch.float64, device=device)
                label_count = 0
                graph_count = 0
                parts = []
                for batch in batches:
                    batch = batch.to(device, non_blocking=True)
                    batch_logits = model(batch)
                    if batch_logits.shape != batch.y.shape or not bool(
                        torch.isfinite(batch_logits).all()
                    ):
                        raise FloatingPointError("Invalid or nonfinite PPI validation logits")
                    counts.add_(_binary_counts(batch_logits, batch.y))
                    loss_sum.add_(
                        F.binary_cross_entropy_with_logits(
                            batch_logits, batch.y, reduction="sum"
                        ).double()
                    )
                    label_count += batch.y.numel()
                    graph_count += _ppi_graph_count(batch)
                    parts.append(batch_logits.detach().cpu())
                if graph_count != 2 or not label_count:
                    raise ValueError("PPI validation must cover both official validation graphs")
                logits = torch.cat(parts, dim=0)
                metric = _micro_f1_from_counts(counts)
                loss = float(loss_sum / label_count)
                metric_name = "micro_f1"
                prediction_rule = "logit_gt_zero_node_label"
                observation_scope = "all two official inductive validation graphs"
        layers = observation.summary() if observation is not None else []
        result = {
            "metric": metric,
            "metric_name": metric_name,
            "prediction_rule": prediction_rule,
            "loss": loss,
            "layers": layers,
            "mode": "eval",
            "split": "validation",
            "validation_graph_count": graph_count,
            "observation_scope": observation_scope,
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
        self.c_contract_checks: list[dict[str, Any]] = []

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
            result = torch.ones_like(c)
            exact_one = bool((result == 1).all())
            if not exact_one:
                raise RuntimeError("C=1 intervention did not produce exact unit conductance")
            self.c_contract_checks.append(
                {
                    "layer": layer,
                    "edge_count": c.numel(),
                    "contract": "exact_one",
                    "satisfied": True,
                }
            )
            return result
        _, incidence, node_graph, num_graphs = inputs
        edge_graph = node_graph[incidence[0]]
        result = c.clone()
        generator = torch.Generator(device=c.device).manual_seed(self.seed + 104729 * layer)
        nonempty_graphs = 0
        for graph_index in range(num_graphs):
            ids = (edge_graph == graph_index).nonzero(as_tuple=False).flatten()
            if not ids.numel():
                continue
            nonempty_graphs += 1
            values = c.index_select(0, ids)
            if self.name == "mean_c":
                mean = values.mean()
                if not bool(torch.isfinite(mean)) or not bool(mean > 0):
                    raise FloatingPointError(
                        "Mean-C intervention requires a finite positive graph mean"
                    )
                result[ids] = mean
                if not bool((result.index_select(0, ids) == mean).all()):
                    raise RuntimeError("Mean-C intervention is not graph-constant")
            else:
                permutation = torch.randperm(ids.numel(), device=c.device, generator=generator)
                result[ids] = values[permutation]
        if self.name == "mean_c":
            if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
                raise FloatingPointError(
                    "Mean-C intervention did not remain finite and positive"
                )
            self.c_contract_checks.append(
                {
                    "layer": layer,
                    "edge_count": c.numel(),
                    "nonempty_graph_count": nonempty_graphs,
                    "contract": "graph_constant_positive",
                    "satisfied": True,
                }
            )
        return result

    def contract_summary(self, expected_layers: int) -> dict[str, Any]:
        """Summarize the directly checked C-replacement contract."""

        if self.name not in {"mean_c", "ones_c", "ones_c_identity_w"}:
            raise ValueError("This intervention has no C-replacement contract")
        expected = list(range(expected_layers))
        observed = [record["layer"] for record in self.c_contract_checks]
        satisfied = all(record["satisfied"] for record in self.c_contract_checks)
        if observed != expected or not satisfied:
            raise RuntimeError("C intervention contract checks are missing or unsatisfied")
        return {
            "contract": self.c_contract_checks[0]["contract"] if expected else None,
            "satisfied": True,
            "layers_checked": len(self.c_contract_checks),
            "edge_counts": [record["edge_count"] for record in self.c_contract_checks],
        }


def _logit_difference(left: Tensor, right: Tensor, label: str) -> Tensor:
    if not isinstance(left, Tensor) or not isinstance(right, Tensor) or left.shape != right.shape:
        raise ValueError(f"{label} logits must be aligned tensors")
    if not left.numel():
        raise ValueError(f"{label} logits must be nonempty")
    left, right = left.detach().double(), right.detach().double()
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise FloatingPointError(f"Nonfinite {label} logits")
    return left - right


def _intervention_row(name, result, logits, original, reference):
    difference = _logit_difference(logits, reference, name)
    prediction_rule = result["prediction_rule"]
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
            (
                _prediction_tensor(logits, prediction_rule)
                != _prediction_tensor(reference, prediction_rule)
            )
            .double()
            .mean()
        ),
    }


def best_checkpoint_interventions(
    model,
    graph,
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
    rows, intervention_logits, replacement_contracts = [], {}, {}
    try:
        for name in names:
            with Intervention(model, name, seed) as intervention:
                result, logits = evaluate_validation(
                    model, graph, indices, observe=False, device=device
                )
            if name in {"mean_c", "ones_c"}:
                replacement_contracts[name] = intervention.contract_summary(len(model.operators))
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
    numeric_difference = _logit_difference(mean_logits, ones_logits, "mean-C/C=1")
    mean_absolute_delta = float(numeric_difference.abs().mean())
    max_absolute_delta = float(numeric_difference.abs().max())
    if not math.isfinite(mean_absolute_delta) or not math.isfinite(max_absolute_delta):
        raise FloatingPointError("Nonfinite mean-C/C=1 numerical delta")
    if max_absolute_delta < mean_absolute_delta:
        raise RuntimeError("Mean-C/C=1 maximum numerical delta is smaller than its mean")
    numeric_check = {
        "comparison": "mean_c_vs_ones_c",
        "role": "informational_non_gating",
        "separate_full_graph_forwards": True,
        "allclose_rtol": 1e-5,
        "allclose_atol": 1e-6,
        "within_declared_tolerance": bool(
            torch.allclose(mean_logits, ones_logits, rtol=1e-5, atol=1e-6)
        ),
        "logit_mean_absolute_delta": mean_absolute_delta,
        "logit_max_absolute_delta": max_absolute_delta,
        "changed_prediction_fraction": float(
            (
                _prediction_tensor(mean_logits, original["prediction_rule"])
                != _prediction_tensor(ones_logits, original["prediction_rule"])
            )
            .double()
            .mean()
        ),
        "replacement_contracts": replacement_contracts,
    }
    return {
        "status": "passed",
        "scope": "validation_selected_best_checkpoint_only",
        "layers": "all_layers_simultaneously",
        "original": {"validation": original["metric"], "loss": original["loss"]},
        "metric_name": original["metric_name"],
        "prediction_rule": original["prediction_rule"],
        "validation_graph_count": original["validation_graph_count"],
        "rows": rows,
        "shuffle_seed": seed,
        "normalization_recomputed_for_c_interventions": True,
        "mean_c_numeric_check": numeric_check,
        "mean_ones_note": (
            "Graph-constant positive C cancellation and the replacement contracts are enforced "
            "directly. Mean-C and C=1 logits come from separate full-graph validation forwards, "
            "so their "
            "allclose result is informational and non-gating because CUDA scatter rounding need "
            "not be bitwise repeatable. This is not an independent causal intervention."
        ),
        "interpretation": (
            "Checkpoint reliance, not a retrained-model benefit; no optimizer step or test "
            "evaluation. Fresh factorial arms, not these rows, estimate learned-component "
            "contrasts."
        ),
    }
