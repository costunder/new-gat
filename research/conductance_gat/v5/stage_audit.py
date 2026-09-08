"""Read-only, validation-only audits of the two V5 learning roles.

This measures a saved model, not a new training recipe. A finite-step reference
is never called an exact optimizer, and an executed audit is not a claim that C
improves generalization. Local comparisons freeze the baseline layer input;
whole-model higher-K evaluation deliberately has a separate interpretation.
"""

from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .diagnostics import PreparedValidationGraph, require_finite_tensor
from .distribution_audit import audit_conductance_distribution


def _iterative_reference(operator) -> bool:
    return (
        operator.conductance_backend == "optimization"
        and getattr(operator, "conductance_generator", "optimized") == "optimized"
        and operator.estimator.mode != "fixed_one"
    )


def _set_steps(estimator, steps):
    if hasattr(estimator, "solver_steps"):
        estimator.solver_steps = steps


def _diagnostic_state(module: nn.Module) -> dict[str, Any]:
    return {key: value for key, value in vars(module).items() if key.startswith("last_")}


def _restore_diagnostics(module: nn.Module, previous: dict[str, Any]) -> None:
    for key in list(vars(module)):
        if key.startswith("last_") and key not in previous:
            delattr(module, key)
    for key, value in previous.items():
        setattr(module, key, value)


@contextmanager
def _preserve_state(model: nn.Module, device: torch.device, source):
    modules = list(model.modules())
    training = [module.training for module in modules]
    diagnostics = [_diagnostic_state(module) for module in modules]
    estimators = [operator.estimator for operator in model.operators]
    settings = [
        (estimator.override, getattr(estimator, "solver_steps", None)) for estimator in estimators
    ]
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    # DataLoader generators are not necessarily the global torch generator.
    owners = [source, getattr(source, "sampler", None), getattr(source, "batch_sampler", None)]
    owners += [getattr(owner, "sampler", None) for owner in list(owners)]
    generators = {
        id(generator): (generator, generator.get_state())
        for owner in owners
        if isinstance((generator := getattr(owner, "generator", None)), torch.Generator)
    }
    try:
        model.eval()
        yield
    finally:
        for estimator, (override, steps) in zip(estimators, settings, strict=True):
            estimator.override = override
            _set_steps(estimator, steps)
        for module, mode, previous in zip(modules, training, diagnostics, strict=True):
            module.training = mode  # Preserve heterogeneous flags, not just the root flag.
            _restore_diagnostics(module, previous)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        for generator, state in generators.values():
            generator.set_state(state)


def _json(value):
    """Transfer summary leaves once per device/dtype, not once per tiny statistic.

    Keep integer and bool groups separate from floating values, so JSON counts
    retain exact int64 values. Only summary tensors are packed, never activations.
    """
    groups, seen = {}, set()

    def collect(item):
        if isinstance(item, Tensor):
            if id(item) not in seen:
                seen.add(id(item))
                groups.setdefault((item.device, item.dtype), []).append(item)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise FloatingPointError("nonfinite stage audit statistic")

    collect(value)
    converted = {}
    for leaves in groups.values():
        packed = torch.cat([leaf.detach().reshape(-1) for leaf in leaves]).cpu()
        require_finite_tensor(packed, "stage audit statistic")
        offset = 0
        for leaf in leaves:
            count = leaf.numel()
            converted[id(leaf)] = packed[offset : offset + count].reshape(leaf.shape).tolist()
            offset += count

    def rebuild(item):
        if isinstance(item, Tensor):
            return converted[id(item)]
        if isinstance(item, dict):
            return {key: rebuild(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [rebuild(child) for child in item]
        return item

    return rebuild(value)


def _difference(left: Tensor, reference: Tensor) -> dict[str, Any]:
    require_finite_tensor(left, "stage audit compared output")
    require_finite_tensor(reference, "stage audit reference output")
    delta = left.float() - reference.float()
    return {
        "squared_difference": delta.square().sum(dtype=torch.float64),
        "squared_reference": reference.float().square().sum(dtype=torch.float64),
        "max_abs": delta.abs().amax() if delta.numel() else delta.new_zeros(()),
        "elements": delta.numel(),
    }


def _finish_difference(value: dict[str, Any]) -> dict[str, Any]:
    value = _json(value)
    numerator = math.sqrt(value["squared_difference"])
    denominator = math.sqrt(value["squared_reference"])
    return {
        "relative_l2": numerator / denominator if denominator else (0.0 if not numerator else None),
        "reference_l2": denominator,
        "difference_l2": numerator,
        "rms_difference": numerator / math.sqrt(value["elements"]) if value["elements"] else None,
        "max_abs": value["max_abs"],
        "elements": value["elements"],
        "zero_reference_norm": denominator == 0,
    }


def _moments(value: Tensor) -> dict[str, Any]:
    flat = value.detach().float().flatten()
    if not flat.numel():
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": flat.numel(),
        "mean": flat.mean(),
        "std": flat.std(correction=0),
        "min": flat.amin(),
        "max": flat.amax(),
    }


def _local_hook(layer: int, batch_number: int, rows: list, reference_steps: int):
    def hook(operator, args, kwargs, baseline_output):
        # Only this layer's O(E) conductances and O(ND) output are live during
        # recomputation. No all-layer feature/edge activation cache is retained.
        estimator = operator.estimator
        modules = list(operator.modules())
        previous_diagnostics = [_diagnostic_state(module) for module in modules]
        previous_override, previous_steps = (
            estimator.override,
            getattr(estimator, "solver_steps", None),
        )
        iterative = _iterative_reference(operator)
        baseline_c = estimator.last_c
        baseline_solver = getattr(estimator, "last_solver_diagnostics", {})
        baseline_beta = operator.last_beta
        distribution = audit_conductance_distribution(
            baseline_c,
            args[1],
            args[2],
            args[3],
            heads=baseline_beta.shape[1],
            beta=baseline_beta,
            sampling_correction=kwargs.get("sampling_correction"),
            normalization=getattr(operator, "propagation_normalization", "symmetric"),
            propagation_filter=getattr(operator, "propagation_filter", "linear"),
        )
        polynomial = getattr(operator, "polynomial_delta", None)
        distribution["polynomial_delta_by_head"] = (
            None if polynomial is None else polynomial.detach().float()
        )
        try:
            estimator.override = "ones"
            # Calling forward directly avoids recursion into this audit hook.
            ones_output = operator.forward(*args, **kwargs)
            ones_comparison = _difference(ones_output, baseline_output)
            del ones_output
            estimator.override = None
            _set_steps(estimator, reference_steps if iterative else previous_steps)
            reference_output = operator.forward(*args, **kwargs) if iterative else None
            reference_c = estimator.last_c if iterative else None
            reference_solver = (
                estimator.last_solver_diagnostics
                if iterative
                else {"applicable": False, "reason": "no iterative learned C solver"}
            )
            rows.append(
                {
                    "layer": layer,
                    "batch": batch_number,
                    "scope": "same baseline H, B, omega, W and beta; before residual/FFN",
                    "nodes": args[0].shape[0],
                    "edges": args[1].shape[1],
                    "graphs": args[3],
                    "deployed_steps": previous_steps,
                    "reference_steps": reference_steps if iterative else None,
                    "iterative_reference_applicable": iterative,
                    "baseline_c": _moments(baseline_c),
                    "baseline_beta": _moments(baseline_beta),
                    "distribution": distribution,
                    "c_deployed_vs_reference": _difference(baseline_c, reference_c)
                    if iterative
                    else None,
                    "operator_deployed_vs_reference": (
                        _difference(baseline_output, reference_output) if iterative else None
                    ),
                    "operator_c_one_vs_deployed": ones_comparison,
                    "baseline_solver": baseline_solver,
                    "reference_solver": reference_solver,
                }
            )
        finally:
            estimator.override = previous_override
            _set_steps(estimator, previous_steps)
            for module, previous in zip(modules, previous_diagnostics, strict=True):
                _restore_diagnostics(module, previous)

    return hook


def _batches(source, indices: Tensor | None, device: torch.device):
    if indices is not None:
        if isinstance(source, PreparedValidationGraph):
            graph, selected = source.prepare(indices, device)
        else:
            graph, selected = source.clone().to(device), indices.to(device)
        yield graph, selected
    else:
        # The caller supplies measured disjoint-union graph batches. Never split
        # a physical batch into per-graph GPU forwards or truncate the iterable.
        for original in source:
            graph = original.clone().to(device, non_blocking=True)
            graph._v5_num_graphs = int(original.num_graphs)
            yield graph, None


def _memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "cuda_available_for_this_audit": False,
            "allocated_bytes": None,
            "reserved_bytes": None,
        }
    return {
        "cuda_available_for_this_audit": True,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_scope": "process cumulative peak; audit does not reset the caller's counters",
    }


def _task_head_gradients(model, graph, selected, baseline, *, batch_number, precision):
    """Actual validation-loss VJP in C space, not a proxy or theta-gradient claim.

    Replace only the inspected layer's C by equal-valued independent head leaves.
    Other layers remain live, including downstream C dependence on hidden states.
    autograd.grad never populates or clears parameter.grad and creates no optimizer.
    """
    rows = []
    for layer, operator in enumerate(model.operators):
        if operator.estimator.mode == "fixed_one":
            rows.append(
                {
                    "layer": layer,
                    "batch": batch_number,
                    "measured": False,
                    "reason": "fixed C has no learned stage-1 conductance",
                }
            )
            continue
        if not graph.incidence_edge_index.shape[1]:
            rows.append(
                {
                    "layer": layer,
                    "batch": batch_number,
                    "measured": False,
                    "reason": "edgeless graph has no C-space gradient",
                }
            )
            continue
        leaves = []

        def split_heads(module, args, c, *, heads=operator.value_weight.shape[0], leaves=leaves):
            leaf = (c[:, None].expand(-1, heads) if c.ndim == 1 else c).detach().clone()
            leaf.requires_grad_(True)
            leaves.append(leaf)
            return leaf

        handle = operator.estimator.register_forward_hook(split_heads)
        logits = loss = gradient = None
        try:
            with (
                torch.enable_grad(),
                torch.autocast(
                    device_type=graph.x.device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ),
            ):
                logits = model(graph)
                logits = logits if selected is None else logits.index_select(0, selected)
                target = graph.y if selected is None else graph.y.index_select(0, selected)
                if len(leaves) != 1:
                    raise RuntimeError(
                        "head gradient audit requires one selected layer call per batch"
                    )
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float())
                    if selected is None
                    else torch.nn.functional.cross_entropy(logits, target)
                )
                (gradient,) = torch.autograd.grad(loss, leaves[0])
            require_finite_tensor(gradient, "actual task head-C gradient")
            equivalence = _difference(logits.detach(), baseline)
            node_graph = getattr(graph, "batch", None)
            if node_graph is None:
                node_graph = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
            edge_graph = node_graph[graph.incidence_edge_index[0]]
            graphs = []
            for graph_id in range(int(getattr(graph, "_v5_num_graphs", 1))):
                g = gradient[edge_graph == graph_id].float()
                dot = g.T @ g
                norm = dot.diag().clamp_min(0).sqrt()
                denominator = norm[:, None] * norm[None, :]
                cosine = dot / denominator.clamp_min(torch.finfo(dot.dtype).tiny)
                graphs.append(
                    {
                        "graph_in_batch": graph_id,
                        "physical_edges": g.shape[0],
                        "head_gradient_norms": norm,
                        "head_pair_dot_matrix": dot,
                        "head_pair_cosine_matrix": cosine,
                        "cosine_undefined_zero_norm": denominator == 0,
                        "negative_dot_matrix": dot < 0,
                        "shared_c_sum_gradient_norm": g.sum(-1).norm(),
                    }
                )
            rows.append(
                {
                    "layer": layer,
                    "batch": batch_number,
                    "measured": True,
                    "task_loss": loss.detach().float(),
                    "equal_c_forward_difference": equivalence,
                    "graphs": graphs,
                }
            )
        finally:
            handle.remove()
            # autograd.grad targets C, so upstream parameter/feature branches
            # can still own saved tensors until the scalar/logit graph dies.
            leaves.clear()
            logits = loss = gradient = None
    return rows


@torch.no_grad()
def audit_stage_roles(
    model: nn.Module,
    source,
    indices: Tensor | None,
    *,
    device: torch.device,
    precision: str = "fp32",
    reference_steps: int,
    reference_tolerance: float,
    head_gradient_conflict: bool = False,
    repeat_evaluations: int = 5,
) -> dict[str, Any]:
    """Inspect all supplied validation labels without fitting any parameter.

    ``indices`` selects the complete official validation split on a full graph.
    ``None`` denotes a single-pass iterable of already-batched PPI validation
    graphs. The caller must supply validation data, never test data or a subset;
    this API cannot infer the provenance of an arbitrary tensor of indices.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must preserve the saved fp32 or bf16 recipe")
    if (
        isinstance(reference_steps, bool)
        or not isinstance(reference_steps, int)
        or reference_steps < 1
    ):
        raise ValueError("reference_steps must be a positive integer")
    if (
        isinstance(reference_tolerance, bool)
        or not isinstance(reference_tolerance, (int, float))
        or not math.isfinite(reference_tolerance)
        or reference_tolerance <= 0
    ):
        raise ValueError("reference_tolerance must be finite and positive")
    operators = list(model.operators)
    if not operators or any(
        operator.conductance_backend not in {"optimization", "mlp"} for operator in operators
    ):
        raise ValueError("stage audit requires reviewed V5 operators")
    if type(repeat_evaluations) is not int or repeat_evaluations < 5:
        raise ValueError("repeat_evaluations must be an integer at least 5")
    deployed_steps = [getattr(operator.estimator, "solver_steps", None) for operator in operators]
    iterative = any(_iterative_reference(operator) for operator in operators)
    if any(steps is not None and reference_steps < steps for steps in deployed_steps):
        raise ValueError("reference_steps cannot reduce any deployed solver budget")
    if indices is not None and (
        indices.dtype != torch.long or indices.ndim != 1 or not indices.numel()
    ):
        raise ValueError("validation indices must be a nonempty one-dimensional int64 tensor")
    if any(
        parameter.device.type != device.type
        or (device.index is not None and parameter.device.index != device.index)
        for parameter in model.parameters()
    ):
        raise ValueError("model must already reside on the requested device")
    variants = (
        ("learned", None),
        *((f"repeat_{number}", None) for number in range(2, repeat_evaluations + 1)),
        ("c_one", "ones"),
        ("mean_c", "mean"),
        ("shuffled_c", "shuffle"),
        ("mean_head_beta", None),
        *((("higher_k_full_model", None),) if iterative else ()),
    )
    totals = {name: torch.zeros(7, dtype=torch.float64, device=device) for name, _ in variants}
    differences = {name: None for name, _ in variants}
    local_rows, input_rows, gradient_rows, batch_noise_rows = [], [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory_before, start = _memory(device), time.perf_counter()
    with _preserve_state(model, device, source):
        for batch_number, (graph, selected) in enumerate(_batches(source, indices, device)):
            target = graph.y if selected is None else graph.y.index_select(0, selected)
            if (
                (selected is None and target.ndim != 2)
                or (selected is not None and target.ndim != 1)
                or not target.numel()
            ):
                raise ValueError("validation target shape does not match classification task")
            require_finite_tensor(target, "validation targets")
            input_rows.append(
                {
                    "batch": batch_number,
                    "nodes": graph.x.shape[0],
                    "edges": graph.incidence_edge_index.shape[1],
                    "input_shape": list(graph.x.shape),
                    "graphs": int(getattr(graph, "_v5_num_graphs", 1)),
                    "label_count": target.numel(),
                }
            )
            baseline = baseline_prediction = None
            batch_repeats = []
            for name, override in variants:
                for operator, steps in zip(operators, deployed_steps, strict=True):
                    operator.estimator.override = override
                    _set_steps(
                        operator.estimator,
                        reference_steps
                        if name == "higher_k_full_model" and _iterative_reference(operator)
                        else steps,
                    )
                handles = []
                try:
                    if name == "learned":
                        handles = [
                            operator.register_forward_hook(
                                _local_hook(layer, batch_number, local_rows, reference_steps),
                                with_kwargs=True,
                            )
                            for layer, operator in enumerate(operators)
                        ]
                    if name == "mean_head_beta":
                        handles = [
                            operator.beta_estimator.register_forward_hook(
                                lambda module, args, beta: beta.mean(-1, keepdim=True).expand_as(
                                    beta
                                )
                            )
                            for operator in operators
                        ]
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
                    ):
                        logits = model(graph)
                    require_finite_tensor(logits, "stage audit validation logits")
                finally:
                    for handle in handles:
                        handle.remove()
                logits = logits if selected is None else logits.index_select(0, selected)
                prediction = logits > 0 if selected is None else logits.argmax(dim=-1)
                if name == "learned":
                    baseline, baseline_prediction = logits, prediction
                difference = _difference(logits, baseline)
                if name == "learned" or name.startswith("repeat_"):
                    batch_repeats.append(
                        {
                            key: value.clone() if isinstance(value, Tensor) else value
                            for key, value in difference.items()
                        }
                    )
                previous = differences[name]
                if previous is None:
                    differences[name] = difference
                else:
                    previous["squared_difference"] += difference["squared_difference"]
                    previous["squared_reference"] += difference["squared_reference"]
                    previous["max_abs"] = torch.maximum(previous["max_abs"], difference["max_abs"])
                    previous["elements"] += difference["elements"]
                totals[name][4] += target.numel()
                totals[name][5] += (prediction != baseline_prediction).sum()
                totals[name][6] += prediction.numel()
                if selected is None:
                    positive = target.bool()
                    totals[name][:3] += torch.stack(
                        (
                            (prediction & positive).sum(),
                            (prediction & ~positive).sum(),
                            (~prediction & positive).sum(),
                        )
                    )
                else:
                    totals[name][3] += (prediction == target).sum()
            batch_noise_rows.append({"batch": batch_number, "repeats": batch_repeats})
            if head_gradient_conflict:
                for operator, steps in zip(operators, deployed_steps, strict=True):
                    operator.estimator.override = None
                    _set_steps(operator.estimator, steps)
                gradient_rows.extend(
                    _task_head_gradients(
                        model,
                        graph,
                        selected,
                        baseline,
                        batch_number=batch_number,
                        precision=precision,
                    )
                )
        if not input_rows:
            raise ValueError("validation source produced no batches")
        if len(local_rows) != len(input_rows) * len(operators):
            raise RuntimeError("not every V5 operator participated exactly once per baseline batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed, memory_after = time.perf_counter() - start, _memory(device)
    interventions = {}
    for name, _ in variants:
        tp, fp, fn, correct, count, changed, prediction_count = totals[name].cpu().tolist()
        denominator = 2 * tp + fp + fn
        metric = (
            correct / count
            if indices is not None
            else (2 * tp / denominator if denominator else 0.0)
        )
        diff = _finish_difference(differences[name])
        interventions[name] = {
            "metric": metric,
            "label_count": int(count),
            "logit_relative_l2": diff["relative_l2"],
            "logit_max_abs": diff["max_abs"],
            "logit_difference": diff,
            "prediction_changed_count": int(changed),
            "prediction_changed_fraction": changed / prediction_count,
        }
    for value in interventions.values():
        value["delta_from_learned"] = value["metric"] - interventions["learned"]["metric"]
    repeats = [
        interventions["learned"],
        *[interventions.pop(f"repeat_{number}") for number in range(2, repeat_evaluations + 1)],
    ]
    noise_l2 = max(row["logit_difference"]["difference_l2"] for row in repeats)
    noise_max = max(row["logit_max_abs"] for row in repeats)
    noise_changed = max(row["prediction_changed_fraction"] for row in repeats)
    roundoff_floor = (
        32
        * torch.finfo(torch.float32).eps
        * interventions["learned"]["logit_difference"]["reference_l2"]
    )
    sensitivity_threshold = max(2 * noise_l2, roundoff_floor)
    for name, value in interventions.items():
        value["above_observed_repeat_noise"] = (
            value["logit_difference"]["difference_l2"] > sensitivity_threshold
            if name != "learned"
            else False
        )
    fixed = all(operator.estimator.mode == "fixed_one" for operator in operators)
    contribution = (
        "not_applicable"
        if fixed
        else "observed_above_repeat_noise"
        if interventions["c_one"]["above_observed_repeat_noise"]
        else "inconclusive"
    )
    for row in batch_noise_rows:
        row["repeats"] = [_finish_difference(value) for value in row["repeats"]]
        maximum = max(value["difference_l2"] for value in row["repeats"])
        floor = 32 * torch.finfo(torch.float32).eps * row["repeats"][0]["reference_l2"]
        row["sensitivity_threshold_l2"] = max(2 * maximum, floor)
    for row in gradient_rows:
        if row["measured"]:
            row["equal_c_forward_difference"] = _finish_difference(
                row["equal_c_forward_difference"]
            )
            row["forward_equivalence_within_repeat_noise"] = (
                row["equal_c_forward_difference"]["difference_l2"]
                <= batch_noise_rows[row["batch"]]["sensitivity_threshold_l2"]
            )
    for row in local_rows:
        for key in (
            "c_deployed_vs_reference",
            "operator_deployed_vs_reference",
            "operator_c_one_vs_deployed",
        ):
            row[key] = _finish_difference(row[key]) if row[key] is not None else None
        residual = row["reference_solver"].get("projected_gradient_rms_final")
        active = row["reference_solver"].get("active_nodes")
        if residual is not None and active is not None:
            active = active.reshape(*active.shape, *((1,) * (residual.ndim - active.ndim)))
        row["reference_residual_tolerance"] = reference_tolerance
        row["reference_tolerance_reached_by_graph"] = (
            None if residual is None else ((residual <= reference_tolerance) | (active == 0))
        )
        row["reference_is_exact_optimum"] = False
    return _json(
        {
            "execution_status": "passed",
            "contribution_status": contribution,
            "contribution_interpretation": (
                "dynamic C sensitivity requires logit L2 above the numerical screen of "
                "same-checkpoint eval differences; an empirical noise screen, not a "
                "significance test, usefulness certificate or convergence proof. Fixed C "
                "contribution is not_applicable even if numerical repeated outputs differ."
            ),
            "repeat_noise_control": {
                "evaluation_count": repeat_evaluations,
                "different_training_seeds": False,
                "scope": "same checkpoint, eval mode, same full validation inputs; no retraining",
                "repeats": repeats,
                "per_physical_validation_batch": batch_noise_rows,
                "max_logit_difference_l2": noise_l2,
                "max_logit_max_abs": noise_max,
                "fp32_roundoff_guard_l2": roundoff_floor,
                "sensitivity_threshold_l2": sensitivity_threshold,
                "max_prediction_changed_fraction": noise_changed,
                "validation_min": min(row["metric"] for row in repeats),
                "validation_max": max(row["metric"] for row in repeats),
                "screen": "L2 > max(2 * observed repeat max L2, 32 * fp32 eps * baseline L2); "
                "conservative empirical guard, not a proven error bound",
                "confidence_interval": None,
            },
            "head_gradient_conflict": {
                "requested": head_gradient_conflict,
                "measured": any(row["measured"] for row in gradient_rows),
                "reason": (
                    None
                    if head_gradient_conflict
                    else "not requested; enable --head-gradient-conflict for extra validation VJPs"
                ),
                "scope": "actual validation task-loss partial gradients in C space; "
                "not theta-space",
                "method": "one equal-valued independent E x H C leaf at one layer at a time",
                "forward_equivalence_policy": "interpret only rows within observed repeat noise",
                "parameter_grads_modified": False,
                "optimizer_used": False,
                "layers": gradient_rows,
            },
            "scope": {
                "split": "validation_only",
                "test_used": False,
                "split_provenance": "caller must verify complete official validation data",
                "optimizer_used": False,
                "backward_used": any(row["measured"] for row in gradient_rows),
                "autograd_vjp_used": any(row["measured"] for row in gradient_rows),
                "parameter_backward_called": False,
                "parameters_updated": False,
                "training_recipe_changed": False,
                "model_state_restored": True,
                "full_validation_passes": len(variants),
                "additional_gradient_validation_passes": (
                    sum(operator.estimator.mode != "fixed_one" for operator in operators)
                    if head_gradient_conflict
                    else 0
                ),
                "metric": "accuracy" if indices is not None else "micro_f1",
                "conductance_role": (
                    "fixed_one_control"
                    if fixed
                    else "learned_C"
                    if any(
                        p.requires_grad
                        for operator in operators
                        for p in operator.estimator.parameters()
                    )
                    else "parameter_free_structural_C_control; sensitivity is not learned C benefit"
                ),
            },
            "reference_contract": {
                "applicable": iterative,
                "not_applicable_reason": None
                if iterative
                else "fixed, analytic or MLP generator has no iterative C reference",
                "deployed_steps_by_layer": deployed_steps,
                "reference_steps": reference_steps,
                "projected_gradient_rms_tolerance": reference_tolerance,
                "exact_optimum_claimed": False,
                "local_scope": "baseline H/B/omega/W/beta frozen; only C solver K differs",
                "full_model_scope": (
                    "all layers use higher K; upstream C changes downstream H; "
                    "not a frozen-input solver comparison"
                ),
                "sampling_fidelity_or_unbiasedness_verified": False,
            },
            "interventions": interventions,
            "local_layer_comparisons": local_rows,
            "inputs": input_rows,
            "resources": {
                "device": str(device),
                "precision": precision,
                "geometry_precision": "fp32",
                "elapsed_wall_seconds": elapsed,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_policy": (
                    "single input batch and local layer recomputations; "
                    "no all-layer activation cache"
                ),
                "gpu_utilization": None,
                "gpu_utilization_reason": "not measured by this read-only numerical audit",
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameter_count": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            },
        }
    )
