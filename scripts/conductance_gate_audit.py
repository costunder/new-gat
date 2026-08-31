#!/usr/bin/env python3
"""Train-label-only gradient audit for one restored conductance checkpoint.

This module deliberately contains no checkpoint, dataset, optimizer, or file I/O.
The owning diagnostic command restores and validates those artifacts, then calls
``audit_gate_gradients``.  Hooks observe the tensors used by the real model
forward; no diagnostic recomputation is presented as an exact task gradient.
"""

from __future__ import annotations

import bisect
import json
import math
from types import SimpleNamespace
from typing import Any

_BLOCK_ELEMENTS = 1_048_576


class _StreamingDistribution:
    """Exact scalar moments plus a bounded, deterministic systematic sample."""

    def __init__(self, expected_count: int, sample_limit: int, near_zero: float) -> None:
        import torch

        if expected_count < 0:
            raise ValueError("expected_count must be nonnegative")
        self.expected_count = int(expected_count)
        self.sample_limit = int(sample_limit)
        self.near_zero = float(near_zero)
        sample_count = min(int(sample_limit), self.expected_count)
        if sample_count:
            # Integer midpoints of equal-width bins.  The sample is deterministic,
            # bounded, duplicate-free, and spans the concatenated tensor stream.
            self.sample_positions = [
                ((2 * index + 1) * self.expected_count) // (2 * sample_count)
                for index in range(sample_count)
            ]
        else:
            self.sample_positions = []
        self.samples: list[float] = []
        self.count = 0
        self.calls = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.absolute_sum = 0.0
        self.squared_sum = 0.0
        self.zeros = 0
        self.near_zeros = 0
        self.negatives = 0
        self._torch = torch

    @staticmethod
    def _blocks(value, limit: int = _BLOCK_ELEMENTS):
        """Yield row-major flat blocks without flattening a whole strided tensor."""
        if value.numel() == 0:
            return
        if value.is_contiguous():
            flat = value.view(-1)
            for start in range(0, flat.numel(), limit):
                yield flat[start : start + limit]
            return
        if value.ndim == 0:
            yield value.reshape(1)
            return
        per_row = max(1, value[0].numel())
        rows = max(1, limit // per_row)
        for start in range(0, value.shape[0], rows):
            yield value[start : start + rows].contiguous().view(-1)

    def append(self, value) -> None:
        torch = self._torch
        value = value.detach()
        self.calls += 1
        for raw_block in self._blocks(value):
            block = raw_block.to(dtype=torch.float64)
            if not bool(torch.isfinite(block).all()):
                raise FloatingPointError("nonfinite tensor encountered in gate audit")
            size = int(block.numel())
            if size == 0:
                continue
            global_start = self.count
            global_end = global_start + size
            left = bisect.bisect_left(self.sample_positions, global_start)
            right = bisect.bisect_left(self.sample_positions, global_end)
            if right > left:
                offsets = torch.tensor(
                    [position - global_start for position in self.sample_positions[left:right]],
                    dtype=torch.long,
                    device=block.device,
                )
                self.samples.extend(float(item) for item in block[offsets].cpu().tolist())

            block_variance, block_mean = torch.var_mean(block, correction=0)
            numbers = (
                torch.stack(
                    (
                        block_mean,
                        block_variance,
                        block.min(),
                        block.max(),
                        block.abs().sum(),
                        block.square().sum(),
                        (block == 0).sum(),
                        (block.abs() <= self.near_zero).sum(),
                        (block < 0).sum(),
                    )
                )
                .cpu()
                .tolist()
            )
            other_mean, other_variance = float(numbers[0]), float(numbers[1])
            if self.count:
                delta = other_mean - self.mean
                combined = self.count + size
                self.m2 += other_variance * size + delta * delta * self.count * size / combined
                self.mean += delta * size / combined
            else:
                self.mean = other_mean
                self.m2 = other_variance * size
            self.minimum = min(self.minimum, float(numbers[2]))
            self.maximum = max(self.maximum, float(numbers[3]))
            self.absolute_sum += float(numbers[4])
            self.squared_sum += float(numbers[5])
            self.zeros += int(numbers[6])
            self.near_zeros += int(numbers[7])
            self.negatives += int(numbers[8])
            self.count += size
        if self.count > self.expected_count:
            raise RuntimeError("hook observed more elements than the declared forward stream")

    def report(self) -> dict[str, Any]:
        torch = self._torch
        if self.count == 0:
            moments = None
        else:
            moments = {
                "mean": self.mean,
                "std_population": math.sqrt(max(0.0, self.m2 / self.count)),
                "min": self.minimum,
                "max": self.maximum,
                "mean_absolute": self.absolute_sum / self.count,
                "l2_norm": math.sqrt(max(0.0, self.squared_sum)),
                "zero_fraction": self.zeros / self.count,
                "near_zero_fraction": self.near_zeros / self.count,
                "negative_fraction": self.negatives / self.count,
            }
            moments["coefficient_of_variation"] = (
                moments["std_population"] / abs(self.mean) if self.mean != 0 else None
            )
        quantiles = None
        if self.samples:
            sample = torch.tensor(self.samples, dtype=torch.float64)
            values = torch.quantile(
                sample, torch.tensor([0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
            ).tolist()
            quantiles = dict(zip(("min", "p10", "median", "p90", "max"), values, strict=True))
        return {
            "observed_elements": self.count,
            "expected_elements": self.expected_count,
            "observed_calls": self.calls,
            "near_zero_threshold": self.near_zero,
            "all_element_moments": moments,
            "quantile_sample": {
                "strategy": "deterministic_equal_width_bin_midpoints_over_concatenated_stream",
                "sample_count": len(self.samples),
                "sample_limit": self.sample_limit,
                "quantiles": quantiles,
                "note": "Quantiles are sampled; all_element_moments use every observed element.",
            },
        }


def _new_stats(expected: int, sample_limit: int, near_zero: float):
    return _StreamingDistribution(expected, sample_limit, near_zero)


def _pack_ppi_graphs(graphs: list[dict[str, Any]], indices: list[int], device):
    import torch

    features, labels, incidences, node_graph = [], [], [], []
    ptr = [0]
    for local_index, graph_index in enumerate(indices):
        raw = graphs[graph_index]
        x = raw["x"].to(device)
        y = raw["y"].to(device)
        edges = raw["incidence_edge_index"].to(device)
        features.append(x)
        labels.append(y)
        incidences.append(edges + ptr[-1])
        node_graph.append(torch.full((len(x),), local_index, dtype=torch.long, device=device))
        ptr.append(ptr[-1] + len(x))
    return (
        SimpleNamespace(
            x=torch.cat(features),
            incidence_edge_index=torch.cat(incidences, dim=1),
            batch=torch.cat(node_graph),
            ptr=torch.tensor(ptr, dtype=torch.long, device=device),
            num_graphs=len(indices),
        ),
        torch.cat(labels),
    )


def _audit_batches(payload: dict[str, Any], device, ppi_batches: int, ppi_batch_size: int):
    import torch

    dataset = str(payload.get("dataset", ""))
    graphs = payload["graphs"]
    splits = payload["splits"]
    if dataset == "ppi":
        train = [int(item) for item in splits["train"]]
        held_out = {int(item) for name in ("validation", "test") for item in splits[name]}
        if len(train) != len(set(train)) or set(train) & held_out:
            raise ValueError("PPI train graphs must be unique and disjoint from held-out graphs")
        selected = train[: ppi_batches * ppi_batch_size]
        if not selected:
            raise ValueError("PPI audit selected no training graph")
        output = []
        for start in range(0, len(selected), ppi_batch_size):
            graph_indices = selected[start : start + ppi_batch_size]
            graph, labels = _pack_ppi_graphs(graphs, graph_indices, device)
            output.append((graph, labels.float(), graph_indices))
        return output

    raw = graphs[0]
    train_mask = splits["train"]
    if train_mask.dtype != torch.bool or train_mask.shape != (len(raw["x"]),):
        raise ValueError("node benchmark train split must be a boolean node mask")
    for name in ("validation", "test"):
        other = splits[name]
        if other.shape != train_mask.shape or bool(torch.any(train_mask & other)):
            raise ValueError("node train mask overlaps or disagrees with a held-out split")
    train_indices = train_mask.nonzero(as_tuple=False).flatten().to(device)
    if train_indices.numel() == 0:
        raise ValueError("node benchmark train split is empty")
    graph = SimpleNamespace(
        x=raw["x"].to(device),
        incidence_edge_index=raw["incidence_edge_index"].to(device),
    )
    # Only train labels cross the device boundary.  Held-out labels are never read.
    labels = raw["y"][train_mask].to(device).long()
    return [(graph, (train_indices, labels), [0])]


def _layer_expected(operator, total_edges: int) -> dict[str, int]:
    linears = tuple(operator.estimator.network[index] for index in (0, 2, 4))
    return {
        "input_abs_bh": total_edges * operator.estimator.channels,
        "input_squared_bh": total_edges * operator.estimator.channels,
        "linear_0_input": total_edges * linears[0].in_features,
        "linear_0_preactivation": total_edges * linears[0].out_features,
        "silu_1_output": total_edges * linears[0].out_features,
        "linear_2_input": total_edges * linears[1].in_features,
        "linear_2_preactivation": total_edges * linears[1].out_features,
        "silu_3_output": total_edges * linears[1].out_features,
        "linear_4_input": total_edges * linears[2].in_features,
        "linear_4_preactivation": total_edges * linears[2].out_features,
        "raw_logit": total_edges,
        "conductance": total_edges,
        "raw_logit_gradient": total_edges,
    }


def _install_layer_hooks(
    model, total_edges: int, sample_limit: int, near_zero: float, handles, tensor_handles
):
    reports = []
    for layer_index, operator in enumerate(model.operators):
        estimator = operator.estimator
        if getattr(estimator, "mode", None) != "full":
            raise ValueError("gate audit requires the full |BH|, (BH)^2 conductance estimator")
        expected = _layer_expected(operator, total_edges)
        stats = {key: _new_stats(count, sample_limit, near_zero) for key, count in expected.items()}

        def linear_hook(number: int, *, _stats=stats, _channels=estimator.channels):
            def capture(_module, inputs, output):
                _stats[f"linear_{number}_input"].append(inputs[0])
                _stats[f"linear_{number}_preactivation"].append(output)
                if number == 0:
                    channels = _channels
                    _stats["input_abs_bh"].append(inputs[0][:, :channels])
                    _stats["input_squared_bh"].append(inputs[0][:, channels : 2 * channels])
                if number == 4:
                    _stats["raw_logit"].append(output)

                    def raw_gradient(gradient):
                        _stats["raw_logit_gradient"].append(gradient)

                    if output.requires_grad:
                        tensor_handles.append(output.register_hook(raw_gradient))

            return capture

        def silu_hook(number: int, *, _stats=stats):
            return lambda _module, _inputs, output: _stats[f"silu_{number}_output"].append(output)

        def estimator_hook(_module, _inputs, output, *, _stats=stats):
            _stats["conductance"].append(output)

        for module, hook in (
            (estimator.network[0], linear_hook(0)),
            (estimator.network[1], silu_hook(1)),
            (estimator.network[2], linear_hook(2)),
            (estimator.network[3], silu_hook(3)),
            (estimator.network[4], linear_hook(4)),
            (estimator, estimator_hook),
        ):
            handles.append(module.register_forward_hook(hook))
        reports.append((layer_index, stats))
    return reports


def _tensor_norm(value) -> float:
    return math.sqrt(float(value.detach().double().square().sum().cpu()))


def _parameter_report(model, accumulated: dict[int, Any], weight_decay: float, near_zero: float):
    output = {}
    epsilon = 1.0e-12
    for name, parameter in model.named_parameters():
        values = parameter.detach()
        count = values.numel()
        norm = _tensor_norm(values)
        gradient = accumulated.get(id(parameter))
        gradient_norm = None if gradient is None else _tensor_norm(gradient)
        decay_norm = weight_decay * norm
        ratio = None if gradient_norm is None else gradient_norm / max(decay_norm, epsilon)
        cosine = None
        if gradient is not None and gradient_norm > 0 and decay_norm > 0:
            cosine = float(
                (gradient.detach().double() * values.double()).sum().cpu() / (gradient_norm * norm)
            )
        output[name] = {
            "requires_grad": parameter.requires_grad,
            "parameter": {
                "elements": count,
                "l2_norm": norm,
                "max_absolute": float(values.abs().max().cpu()) if count else None,
                "zero_fraction": float((values == 0).sum().cpu()) / count if count else None,
                "near_zero_fraction": (
                    float((values.abs() <= near_zero).sum().cpu()) / count if count else None
                ),
            },
            "task_gradient": {
                "is_none": gradient is None,
                "l2_norm": gradient_norm,
                "max_absolute": (
                    float(gradient.abs().max().cpu()) if gradient is not None and count else None
                ),
            },
            "weight_decay_term_norm": decay_norm,
            "task_to_decay_norm_ratio": ratio,
            "ratio_denominator_epsilon": epsilon,
            "ratio_denominator_was_clamped": decay_norm < epsilon,
            "task_decay_cosine": cosine,
            "near_zero_threshold": near_zero,
            "cosine_note": "null when task gradient or weight-decay vector has zero norm",
        }
    return output


def audit_gate_gradients(
    model,
    payload: dict[str, Any],
    device,
    *,
    weight_decay: float,
    mode: str = "eval",
    ppi_batches: int = 1,
    ppi_batch_size: int = 2,
    rng_seed: int = 0,
    sample_limit: int = 4096,
    near_zero: float = 1e-8,
) -> dict[str, Any]:
    """Audit exact task gradients from training labels without changing model state.

    ``mode='eval'`` is the default: autograd remains enabled while dropout is off.
    ``mode='train'`` permits a controlled dropout audit under a private RNG fork.
    The result is strict-JSON-finite.  It is a local first-order audit, not an Adam
    update reconstruction and not evidence of a causal training intervention.
    """
    import torch
    from torch.nn import functional as F

    device = torch.device(device)
    if mode not in {"eval", "train"}:
        raise ValueError("mode must be eval or train")
    if (
        not math.isfinite(weight_decay)
        or weight_decay < 0
        or ppi_batches < 1
        or ppi_batch_size < 1
        or sample_limit < 1
        or near_zero < 0
        or not math.isfinite(near_zero)
    ):
        raise ValueError("invalid gradient-audit controls")
    modules = list(model.modules())
    module_modes = [module.training for module in modules]
    parameters = list(model.parameters())
    original_grads = [parameter.grad for parameter in parameters]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    handles: list[Any] = []
    tensor_handles: list[Any] = []
    batches: list[dict[str, Any]] = []
    accumulated: dict[int, Any] = {}
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    try:
        with torch.inference_mode(False), torch.random.fork_rng(devices=cuda_devices, enabled=True):
            # torch.manual_seed also seeds every CUDA device.  Seed only the CPU
            # and the explicitly forked device so unrelated GPU RNGs stay intact.
            torch.random.default_generator.manual_seed(rng_seed)
            if device.type == "cuda":
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(rng_seed)
            model.train(mode == "train")
            # Keep future PPI batches on the CPU; only the current packed batch
            # needs device memory for forward/backward.
            prepared = _audit_batches(payload, torch.device("cpu"), ppi_batches, ppi_batch_size)
            total_edges = sum(int(graph.incidence_edge_index.shape[1]) for graph, _, _ in prepared)
            layer_stats = _install_layer_hooks(
                model, total_edges, sample_limit, near_zero, handles, tensor_handles
            )
            if payload["dataset"] == "ppi":
                total_labels = sum(int(labels.numel()) for _, labels, _ in prepared)
                loss_name, reduction = "binary_cross_entropy_with_logits", "label_element_mean"
            else:
                total_labels = int(prepared[0][1][1].numel())
                loss_name, reduction = "cross_entropy", "train_node_mean"

            trainable = [parameter for parameter in parameters if parameter.requires_grad]
            if not trainable:
                raise ValueError("model has no trainable parameter")
            loss_value = 0.0
            with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
                for batch_index, (graph, targets, graph_indices) in enumerate(prepared):
                    graph = SimpleNamespace(
                        **{
                            name: value.to(device) if isinstance(value, torch.Tensor) else value
                            for name, value in vars(graph).items()
                        }
                    )
                    logits = model(graph)
                    if payload["dataset"] == "ppi":
                        targets = targets.to(device)
                        label_count = int(targets.numel())
                        batch_loss = F.binary_cross_entropy_with_logits(logits, targets)
                    else:
                        train_indices, labels = (target.to(device) for target in targets)
                        label_count = int(labels.numel())
                        batch_loss = F.cross_entropy(logits[train_indices], labels)
                    if not bool(torch.isfinite(batch_loss.detach())):
                        raise FloatingPointError("nonfinite train-only loss in gate audit")
                    objective_weight = label_count / total_labels
                    objective = batch_loss * objective_weight
                    gradients = torch.autograd.grad(objective, trainable, allow_unused=True)
                    for parameter, gradient in zip(trainable, gradients, strict=True):
                        if gradient is None:
                            continue
                        key = id(parameter)
                        if key not in accumulated:
                            accumulated[key] = gradient.detach().clone()
                        else:
                            accumulated[key].add_(gradient.detach())
                    contribution = float(objective.detach().cpu())
                    loss_value += contribution
                    batches.append(
                        {
                            "batch": batch_index,
                            "graph_indices": graph_indices,
                            "graphs": len(graph_indices),
                            "nodes": int(graph.x.shape[0]),
                            "train_nodes": (
                                int(graph.x.shape[0])
                                if payload["dataset"] == "ppi"
                                else label_count
                            ),
                            "edges": int(graph.incidence_edge_index.shape[1]),
                            "train_label_elements": label_count,
                            "batch_mean_loss": float(batch_loss.detach().cpu()),
                            "objective_weight": objective_weight,
                            "weighted_loss_contribution": contribution,
                        }
                    )

            layers = [
                {
                    "layer": layer_index,
                    "tensors": {name: statistic.report() for name, statistic in stats.items()},
                }
                for layer_index, stats in layer_stats
            ]
            report = {
                "schema_version": 1,
                "dataset": payload["dataset"],
                "mode": mode,
                "rng_seed": rng_seed,
                "label_scope": "train_only",
                "controls": {
                    "weight_decay": weight_decay,
                    "near_zero_threshold": near_zero,
                    "sample_limit": sample_limit,
                    "ppi_requested_batches": ppi_batches,
                    "ppi_batch_size": ppi_batch_size,
                    "actual_batches": len(batches),
                },
                "loss": {
                    "name": loss_name,
                    "value": loss_value,
                    "reduction": reduction,
                    "batches": len(batches),
                    "train_label_elements": total_labels,
                    "batch_aggregation": (
                        "PPI batch means weighted by label-element count; this is one combined "
                        "audit objective, not a replay of sequential optimizer steps"
                        if payload["dataset"] == "ppi"
                        else "one full-graph forward with loss restricted to training nodes"
                    ),
                },
                "batches": batches,
                "parameters": _parameter_report(model, accumulated, weight_decay, near_zero),
                "layers": layers,
                "notes": {
                    "autograd": "enabled; default eval mode disables dropout only",
                    "gradient_scope": (
                        "training labels only; full graph features remain transductive"
                    ),
                    "ppi_selection": (
                        "deterministic first requested training batches in split order"
                    ),
                    "weight_decay": (
                        "lambda_times_parameter only; no optimizer moments or update reconstructed"
                    ),
                    "interpretation": "local checkpoint gradient audit, not a causal intervention",
                },
            }
            json.dumps(report, allow_nan=False)
            return report
    finally:
        for handle in reversed(tensor_handles):
            handle.remove()
        for handle in reversed(handles):
            handle.remove()
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, training in zip(modules, module_modes, strict=True):
            module.training = training
        for parameter, gradient in zip(parameters, original_grads, strict=True):
            parameter.grad = gradient
