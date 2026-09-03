"""Focused diagnostics for V5 C/beta learning and checkpoint interventions."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


def require_finite_tensor(value: Tensor, label: str) -> None:
    """Reject non-finite model outputs before they can contaminate metrics/state."""

    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"nonfinite {label}")


def tensor_moments(value: Tensor | None) -> dict[str, float | int | None]:
    if value is None or not value.numel():
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "cv": None}
    flat = value.detach().double().flatten()
    mean, std = float(flat.mean()), float(flat.std(correction=0))
    return {
        "count": flat.numel(),
        "mean": mean,
        "std": std,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "cv": std / abs(mean) if mean else None,
    }


def parameter_norm(parameters, *, gradient: bool = False) -> float | None:
    values = []
    for parameter in parameters:
        value = parameter.grad if gradient else parameter
        if value is not None:
            values.append(value.detach().double().square().sum())
    return float(torch.stack(values).sum().sqrt()) if values else None


def layer_diagnostics(model: nn.Module, *, gradients: bool = False) -> list[dict[str, Any]]:
    rows = []
    for layer, operator in enumerate(model.operators):
        estimator_parameters = list(operator.estimator.parameters())
        beta_parameters = list(operator.beta_estimator.parameters())
        rows.append(
            {
                "layer": layer,
                "conductance": tensor_moments(operator.estimator.last_c),
                "log_conductance": tensor_moments(operator.estimator.last_log_c),
                "score": tensor_moments(operator.estimator.last_scores),
                "beta": tensor_moments(operator.last_beta),
                "sampling_correction": tensor_moments(operator.last_sampling_correction),
                "conductance_parameter_norm": parameter_norm(estimator_parameters),
                "conductance_gradient_norm": (
                    parameter_norm(estimator_parameters, gradient=True) if gradients else None
                ),
                "beta_parameter_norm": parameter_norm(beta_parameters),
                "beta_gradient_norm": (
                    parameter_norm(beta_parameters, gradient=True) if gradients else None
                ),
            }
        )
    return rows


def require_first_step_conductance_gradient(model: nn.Module) -> dict[str, Any]:
    """Fail if V4-style identity initialization starves C on its first active step."""

    if model.conductance_mode != "dynamic":
        return {"applicable": False, "passed": True, "layers": []}
    rows = []
    for layer, operator in enumerate(model.operators):
        named = list(operator.estimator.named_parameters())
        total = parameter_norm((value for _, value in named), gradient=True)
        upstream = parameter_norm(
            (value for name, value in named if "score_network.4" not in name), gradient=True
        )
        passed = total is not None and total > 0 and upstream is not None and upstream > 0
        rows.append(
            {
                "layer": layer,
                "total_gradient_norm": total,
                "upstream_gradient_norm": upstream,
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in rows):
        raise RuntimeError("V5 first active-C backward has a zero conductance gradient path")
    return {"applicable": True, "passed": True, "layers": rows}


@contextmanager
def conductance_intervention(model: nn.Module, mode: str | None):
    previous = [operator.estimator.override for operator in model.operators]
    try:
        for operator in model.operators:
            operator.estimator.override = mode
        yield
    finally:
        for operator, value in zip(model.operators, previous, strict=True):
            operator.estimator.override = value


@torch.no_grad()
def evaluate(
    model: nn.Module,
    source,
    indices: Tensor | None,
    *,
    device: torch.device,
    precision: str = "fp32",
) -> dict[str, Any]:
    model.eval()
    if indices is not None:
        # PyG Data.to mutates storage. Clone so sampled training keeps its
        # canonical full graph on CPU after full-graph validation.
        graph = source.clone().to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            logits = model(graph)
        require_finite_tensor(logits, "validation logits")
        selected = logits.index_select(0, indices.to(device))
        target = graph.y.index_select(0, indices.to(device))
        metric = float((selected.argmax(dim=-1) == target).float().mean())
        count = int(target.numel())
    else:
        totals = torch.zeros(5, dtype=torch.long, device=device)
        for graph in source:
            graph._v5_num_graphs = int(graph.num_graphs)
            graph = graph.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                logits = model(graph)
            prediction = logits > 0
            target = graph.y.bool()
            totals += torch.stack(
                (
                    (prediction & target).sum(),
                    (prediction & ~target).sum(),
                    (~prediction & target).sum(),
                    target.new_tensor(target.numel(), dtype=torch.long),
                    torch.isfinite(logits).all().long(),
                )
            )
        tp, fp, fn, count, finite_batches = (int(value) for value in totals.cpu().tolist())
        if finite_batches != len(source):
            raise FloatingPointError("nonfinite validation logits")
        denominator = 2 * tp + fp + fn
        metric = 2 * tp / denominator if denominator else 0.0
    if not math.isfinite(metric):
        raise FloatingPointError("nonfinite validation metric")
    return {"metric": metric, "label_count": count, "layers": layer_diagnostics(model)}


def selected_checkpoint_interventions(
    model: nn.Module,
    source,
    indices: Tensor | None,
    *,
    device: torch.device,
    precision: str = "fp32",
) -> dict[str, Any]:
    result = {}
    for name, override in (
        ("learned", None),
        ("c_one", "ones"),
        ("mean_c", "mean"),
        ("shuffled_c", "shuffle"),
    ):
        with conductance_intervention(model, override):
            result[name] = evaluate(model, source, indices, device=device, precision=precision)
    baseline = result["learned"]["metric"]
    for value in result.values():
        value["delta_from_learned"] = value["metric"] - baseline
    return result
