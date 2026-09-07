"""Focused diagnostics for V5 C/beta learning and checkpoint interventions."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


class PreparedValidationGraph:
    """One invocation's owned, immutable full-graph device snapshot.

    The CPU sampler graph is never moved or mutated. Construction is explicit
    so the trainer/calibration can account for this resident memory allocation.
    No hidden global cache and no cache for PPI's varying graph batches.
    """

    def __init__(self, source, device: torch.device):
        self.graph = source.clone().to(device)
        self.device = self.graph.x.device
        self._tensors = {
            key: (value, value._version)
            for key, value in self.graph.items()
            if isinstance(value, Tensor)
        }
        storages = {
            (value.device, value.untyped_storage().data_ptr()): value.untyped_storage().nbytes()
            for value, _ in self._tensors.values()
        }
        self._selected_source = None
        self._selected_version = None
        self._selected = None
        self._selected_cache_version = None
        self.metadata = {
            "policy": "one owned immutable graph clone/device transfer per training invocation",
            "source_mutated": False,
            "device": str(self.device),
            "cached_graph_tensor_storage_bytes": sum(storages.values()),
            "indices_policy": "one device clone per source tensor identity/version",
            "scope": "full validation graph only; no model activations or predictions cached",
        }

    def prepare(self, indices: Tensor, device: torch.device):
        requested = torch.device(device)
        if requested.type != self.device.type or (
            requested.index is not None and requested.index != self.device.index
        ):
            raise ValueError("validation graph cache belongs to another device")
        for key, (value, version) in self._tensors.items():
            if getattr(self.graph, key) is not value or value._version != version:
                raise RuntimeError("owned validation graph cache was mutated")
        if self._selected is not None and self._selected._version != self._selected_cache_version:
            raise RuntimeError("owned validation index cache was mutated")
        if indices is not self._selected_source or indices._version != self._selected_version:
            self._selected = indices.detach().clone().to(self.device)
            self._selected_source, self._selected_version = indices, indices._version
            self._selected_cache_version = self._selected._version
        return self.graph, self._selected


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


def solver_diagnostics(estimator: nn.Module) -> dict[str, Any]:
    """Serialize the last forward's solver audit only at an existing log boundary."""

    values = getattr(estimator, "last_solver_diagnostics", None)
    if values is None:
        return {"applicable": False, "reason": "MLP backend has no inner C optimization"}

    def convert(value):
        if isinstance(value, Tensor):
            require_finite_tensor(value, "C solver diagnostic")
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return {"scope": "last forward graph or disjoint graph batch", **convert(values)}


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
                "conductance_backend": operator.conductance_backend,
                "c_optimization": solver_diagnostics(operator.estimator),
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
        optimization = operator.conductance_backend == "optimization"
        upstream = parameter_norm(
            (
                value
                for name, value in named
                if ("projection" in name if optimization else "score_network.4" not in name)
            ),
            gradient=True,
        )
        passed = total is not None and total > 0 and upstream is not None and upstream > 0
        rows.append(
            {
                "layer": layer,
                "total_gradient_norm": total,
                "upstream_gradient_norm": upstream,
                "conductance_backend": operator.conductance_backend,
                "upstream_definition": (
                    "compatibility projection parameters through unrolled C updates"
                    if optimization
                    else "MLP parameters excluding its final score weight"
                ),
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
    collect_diagnostics: bool = True,
) -> dict[str, Any]:
    model.eval()
    if indices is not None:
        # PyG Data.to mutates storage. Clone so sampled training keeps its
        # canonical full graph on CPU after full-graph validation.
        if isinstance(source, PreparedValidationGraph):
            graph, selected_indices = source.prepare(indices, device)
        else:
            graph = source.clone().to(device)
            selected_indices = indices.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            logits = model(graph)
        require_finite_tensor(logits, "validation logits")
        selected = logits.index_select(0, selected_indices)
        target = graph.y.index_select(0, selected_indices)
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
    result = {"metric": metric, "label_count": count}
    if collect_diagnostics:
        result["layers"] = layer_diagnostics(model)
    return result


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
