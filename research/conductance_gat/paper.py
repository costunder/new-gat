"""Linux/CUDA paper runner for the independent conductance-GAT track.

Examples
--------
python -m research.conductance_gat.paper --suite core --data-root ./data \
    --output-dir ./results/conductance --device cuda --seed 17
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader

from chartgat.observability import RuntimeResourceMonitor, runtime_resource_snapshot
from chartgat.seeds import SeedAxes, resolve_seed_axes

from .paper_data import nonlinear_conductance, prepare_core_cache
from .public_data import prepare_public_data
from .sparse import (
    PackedGraphBatch,
    SparseIncidenceConductanceLayer,
    edge_gradient,
    pack_graph_examples,
)

CORE_CLAIMS = {
    "s1": "Static shared conductance law generalizes to held-out graph identities.",
    "s2": "The law transfers from ER/RGG n=16..32 to larger grid/barbell graphs.",
    "s3": "State-dependent positive conductance supports stable held-graph rollout.",
    "s4": "Identification limits are mapped across contrast, excitation coverage, and SNR.",
}
TRAINING_OBJECTIVES = {"node_only", "flux_only", "joint"}
SOURCE_FILES = (
    "research/conductance_gat/paper.py",
    "research/conductance_gat/paper_data.py",
    "research/conductance_gat/public_data.py",
    "research/conductance_gat/sparse.py",
    "src/chartgat/observability.py",
    "src/chartgat/seeds.py",
)


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required implementation source is missing: {path}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _verify_implementation_unchanged(expected: Mapping[str, str]) -> dict[str, Any]:
    actual = _implementation_hashes()
    if actual != dict(expected):
        changed = sorted(
            path
            for path in set(expected) | set(actual)
            if expected.get(path) != actual.get(path)
        )
        raise RuntimeError(
            "conductance paper implementation changed during execution: "
            + ", ".join(changed)
        )
    return {
        "valid": True,
        "policy": "explicit source files hashed before training and verified before artifacts",
        "sha256": actual,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _require_true_async(predicate: Tensor, message: str) -> None:
    assertion = getattr(torch, "_assert_async", None)
    if assertion is not None:
        assertion(predicate, message)
        return
    if not bool(predicate):
        raise RuntimeError(message)


def _require_finite_async(value: Tensor, label: str) -> None:
    _require_true_async(torch.isfinite(value).all(), f"nonfinite {label}")


def _parameter_inventory(model: nn.Module) -> dict[str, Any]:
    all_parameters = list(model.named_parameters())
    trainable = [(name, parameter) for name, parameter in all_parameters if parameter.requires_grad]
    return {
        "total_parameters": sum(parameter.numel() for _, parameter in all_parameters),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_tensors": len(trainable),
        "frozen_parameters": sum(
            parameter.numel() for _, parameter in all_parameters if not parameter.requires_grad
        ),
    }


def _optimizer_integrity(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    duplicate_ids = sorted(
        parameter_id for parameter_id in set(optimizer_ids) if optimizer_ids.count(parameter_id) > 1
    )
    missing = sorted(
        name for parameter_id, name in expected.items() if parameter_id not in optimizer_ids
    )
    unexpected = [
        parameter_id for parameter_id in optimizer_ids if parameter_id not in expected
    ]
    if duplicate_ids or missing or unexpected:
        raise RuntimeError(
            "optimizer ownership mismatch: "
            f"missing={missing}, duplicate_parameter_ids={duplicate_ids}, "
            f"unexpected_parameter_ids={unexpected}"
        )
    return {
        "verified": True,
        "optimizer": type(optimizer).__name__,
        "parameter_groups": len(optimizer.param_groups),
        "owned_trainable_parameter_tensors": len(optimizer_parameters),
        "owned_trainable_parameters": sum(parameter.numel() for parameter in optimizer_parameters),
    }


def _gradient_integrity(model: nn.Module) -> dict[str, Any]:
    missing: list[str] = []
    nonfinite: list[str] = []
    zero: list[str] = []
    norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
            continue
        norm = float(gradient.detach().float().norm().cpu())
        norms[name] = norm
        if norm == 0.0:
            zero.append(name)
    if missing or nonfinite or not any(value > 0 for value in norms.values()):
        raise RuntimeError(
            "gradient connectivity failed after the actual loss backward pass: "
            f"missing={missing}, nonfinite={nonfinite}, zero={zero}"
        )
    return {
        "verified": True,
        "all_trainable_gradients_present": not missing,
        "all_present_gradients_finite": not nonfinite,
        "nonzero_gradient_parameter_tensors": sum(value > 0 for value in norms.values()),
        "zero_gradient_parameter_tensors": zero,
        "gradient_norms": norms,
    }


def _first_step_profile(
    *,
    batch_description: Mapping[str, Any],
    gradient: Mapping[str, Any],
    before: Mapping[str, Tensor],
    model: nn.Module,
    timing_seconds: Mapping[str, float],
) -> dict[str, Any]:
    changed = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not torch.equal(before[name], parameter.detach())
    ]
    if not changed:
        raise RuntimeError(
            "optimizer connectivity failed: the first finite backward pass changed no parameter"
        )
    return {
        "input_to_forward_to_loss_to_backward_verified": True,
        "optimizer_update_verified": True,
        "changed_parameter_tensors": changed,
        "batch": dict(batch_description),
        "gradient": dict(gradient),
        "timing_seconds": dict(timing_seconds),
        "timing_scope": (
            "first actual optimizer step with CUDA synchronization at stage boundaries; "
            "remaining steps run without per-stage synchronization"
        ),
    }


def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None, "total": 0}
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
        "total": sum(values),
    }


def _example_statistics(examples: Sequence[Mapping[str, Any]], *, public: bool) -> dict[str, Any]:
    node_key = "x" if public else "node_state"
    node_counts = [int(example[node_key].shape[0]) for example in examples]
    edge_counts = [int(example["edge_index"].shape[1]) for example in examples]
    first = examples[0] if examples else None
    return {
        "examples_available": len(examples),
        "examples_used": len(examples),
        "used_fraction": 1.0 if examples else None,
        "graph_nodes": _distribution(node_counts),
        "graph_edges": _distribution(edge_counts),
        "first_input_shapes": None
        if first is None
        else {
            node_key: list(first[node_key].shape),
            "edge_index": list(first["edge_index"].shape),
            "edge_features": list(first["edge_features"].shape),
            "target": list(first["y"].shape)
            if public
            else list(first["true_node_message"].shape),
        },
        "sampling_ratio": 1.0 if examples else None,
        "subset_or_fast_mode": False,
    }


def _dataset_observability(
    core: Mapping[str, Any] | None, public: Mapping[str, Any] | None
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if core is not None:
        report["core"] = {
            suite_name: {
                split: _example_statistics(examples, public=False)
                for split, examples in suite.items()
                if isinstance(examples, Sequence)
                and not isinstance(examples, (str, bytes))
                and examples
                and isinstance(examples[0], Mapping)
                and "node_state" in examples[0]
            }
            for suite_name, suite in core.items()
            if isinstance(suite, Mapping)
        }
    if public is not None:
        report["public"] = {
            dataset_name: {
                split: _example_statistics(examples, public=True)
                for split, examples in splits.items()
                if isinstance(examples, Sequence)
            }
            for dataset_name, splits in public.items()
            if dataset_name != "fixture" and isinstance(splits, Mapping)
        }
    return report


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {normalized!r} was requested but this PyTorch build cannot use CUDA"
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def runtime_metadata(
    device: torch.device, *, amp: bool, pin_memory: bool, batch_size: int
) -> dict[str, Any]:
    cuda = device.type == "cuda"
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "amp": bool(amp),
        "pin_memory": bool(pin_memory),
        "batch_size": int(batch_size),
        "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
        "visible_cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "cpu_logical_count": os.cpu_count(),
        "precision": "float16_autocast" if amp else "float32",
    }
    if cuda:
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_total_memory_bytes": int(properties.total_memory),
                "mig_detected_from_device_name": "MIG" in properties.name.upper(),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    else:
        metadata.update(
            {
                "cuda_total_memory_bytes": None,
                "mig_detected_from_device_name": None,
                "cuda_peak_allocated_bytes": None,
                "cuda_peak_reserved_bytes": None,
                "cuda_measurement_unavailable_reason": "requested device is CPU, not CUDA",
            }
        )
    return metadata


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(
    examples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=pack_graph_examples,
    )


def _normalized_loss(
    model: SparseIncidenceConductanceLayer,
    batch: PackedGraphBatch,
    *,
    objective: str,
    collect_diagnostics: bool = True,
) -> tuple[Tensor, dict[str, float | None]]:
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"unknown training objective {objective!r}")
    _, diagnostics = model(batch, return_diagnostics=True)
    flux_target = None
    if objective in {"flux_only", "joint"}:
        flux_target = batch.observed_flux if batch.observed_flux is not None else batch.true_flux
    node_target = None
    if objective in {"node_only", "joint"}:
        node_target = (
            batch.observed_node_message
            if batch.observed_node_message is not None
            else batch.true_node_message
        )
    epsilon = torch.finfo(diagnostics["edge_flux"].dtype).eps
    flux_relative = None
    if flux_target is not None:
        flux_mse = (diagnostics["edge_flux"] - flux_target).square().mean()
        flux_scale = flux_target.square().mean().clamp_min(epsilon)
        flux_relative = flux_mse / flux_scale
    node_relative = None
    if node_target is not None:
        node_mse = (diagnostics["node_message"] - node_target).square().mean()
        node_scale = node_target.square().mean().clamp_min(epsilon)
        node_relative = node_mse / node_scale
    if objective == "node_only":
        if node_relative is None:
            raise ValueError("node_only training requires a node-message target")
        loss = node_relative
    elif objective == "flux_only":
        if flux_relative is None:
            raise ValueError("flux_only training requires an edge-flux target")
        loss = flux_relative
    else:
        if flux_relative is None or node_relative is None:
            raise ValueError("joint training requires edge-flux and node-message targets")
        loss = flux_relative + node_relative
    if not collect_diagnostics:
        return loss, {}
    return loss, {
        "loss": float(loss.detach().float().cpu()),
        "flux_relative_mse": (
            None if flux_relative is None else float(flux_relative.detach().float().cpu())
        ),
        "node_relative_mse": (
            None if node_relative is None else float(node_relative.detach().float().cpu())
        ),
    }


@torch.no_grad()
def _validation_loss(
    model: SparseIncidenceConductanceLayer,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    objective: str,
    loader: DataLoader | None = None,
) -> float:
    if not examples:
        raise ValueError("validation split is empty")
    model.eval()
    total: Tensor | None = None
    count = 0
    effective_loader = loader or _loader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    for batch in effective_loader:
        batch = batch.to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            loss, _ = _normalized_loss(
                model, batch, objective=objective, collect_diagnostics=False
            )
        weighted = loss.detach().float() * batch.num_graphs
        total = weighted if total is None else total + weighted
        count += batch.num_graphs
    if count != len(examples):
        raise RuntimeError(
            f"validation loader consumed {count} graphs but {len(examples)} were required"
        )
    if total is None:
        raise RuntimeError("validation produced no loss")
    return float(total.cpu()) / count


def train_sparse_model(
    model: SparseIncidenceConductanceLayer,
    train_examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
    objective: str,
    execution_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"unknown training objective {objective!r}")
    if not train_examples or not validation_examples:
        raise ValueError("training and validation splits must both be nonempty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    parameter_inventory = _parameter_inventory(model)
    optimizer_integrity = _optimizer_integrity(model, optimizer)
    scaler = _grad_scaler(amp)
    best_validation = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    train_loader = _loader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    validation_loader = _loader(
        validation_examples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    optimizer_steps = 0
    graphs_processed = 0
    data_wait_seconds = 0.0
    validation_seconds = 0.0
    first_step: dict[str, Any] | None = None
    _synchronize(device)
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total: Tensor | None = None
        count = 0
        if train_loader.generator is None:
            raise RuntimeError("training DataLoader has no generator for deterministic shuffling")
        train_loader.generator.manual_seed(seed + epoch)
        wait_started = time.perf_counter()
        for batch in train_loader:
            data_wait_seconds += time.perf_counter() - wait_started
            profile_this_step = first_step is None
            if profile_this_step:
                _synchronize(device)
                transfer_started = time.perf_counter()
            batch = batch.to(device, non_blocking=pin_memory)
            if profile_this_step:
                _synchronize(device)
                transfer_seconds = time.perf_counter() - transfer_started
                forward_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, _ = _normalized_loss(
                    model, batch, objective=objective, collect_diagnostics=False
                )
            _require_finite_async(
                loss.detach(),
                f"{objective} training loss at epoch={epoch}, "
                f"optimizer_step={optimizer_steps + 1}",
            )
            if profile_this_step:
                _synchronize(device)
                forward_seconds = time.perf_counter() - forward_started
                backward_started = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_integrity = _gradient_integrity(model) if profile_this_step else None
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            _require_finite_async(
                gradient_norm,
                f"clipped gradient norm at epoch={epoch}, optimizer_step={optimizer_steps + 1}",
            )
            if profile_this_step:
                _synchronize(device)
                backward_seconds = time.perf_counter() - backward_started
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                optimizer_started = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            if profile_this_step:
                _synchronize(device)
                optimizer_seconds = time.perf_counter() - optimizer_started
                first_step = _first_step_profile(
                    batch_description={
                        "graphs": batch.num_graphs,
                        "nodes": batch.num_nodes,
                        "edges": batch.num_edges,
                        "node_state_shape": list(batch.node_state.shape),
                        "edge_index_shape": list(batch.edge_index.shape),
                        "edge_features_shape": list(batch.edge_features.shape),
                        "loss": float(loss.detach().float().cpu()),
                    },
                    gradient=gradient_integrity or {},
                    before=before,
                    model=model,
                    timing_seconds={
                        "host_to_device": transfer_seconds,
                        "forward_and_loss": forward_seconds,
                        "backward_and_gradient_validation": backward_seconds,
                        "optimizer": optimizer_seconds,
                    },
                )
            weighted = loss.detach().float() * batch.num_graphs
            total = weighted if total is None else total + weighted
            count += batch.num_graphs
            graphs_processed += batch.num_graphs
            wait_started = time.perf_counter()
        if count != len(train_examples):
            raise RuntimeError(
                f"training loader consumed {count} graphs but {len(train_examples)} were required"
            )
        _synchronize(device)
        validation_started = time.perf_counter()
        validation = _validation_loss(
            model,
            validation_examples,
            device=device,
            amp=amp,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            objective=objective,
            loader=validation_loader,
        )
        _synchronize(device)
        validation_seconds += time.perf_counter() - validation_started
        if not math.isfinite(validation):
            raise RuntimeError(f"nonfinite validation loss at epoch={epoch}: {validation}")
        if total is None:
            raise RuntimeError(f"training epoch {epoch} produced no loss")
        train_loss = float(total.cpu()) / count
        history.append(
            {
                "epoch": epoch,
                "training_objective": objective,
                "train_loss": train_loss,
                "validation_loss": validation,
            }
        )
        if validation < best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    _synchronize(device)
    training_seconds = time.perf_counter() - training_started
    expected_steps = epochs * math.ceil(len(train_examples) / batch_size)
    if optimizer_steps != expected_steps:
        raise RuntimeError(
            f"optimizer executed {optimizer_steps} steps but {expected_steps} were required"
        )
    if best_state is None or first_step is None:
        raise RuntimeError("training completed without a validated checkpoint or optimizer step")
    model.load_state_dict(best_state)
    if execution_report is not None:
        execution_report.clear()
        execution_report.update(
            {
                "model": {
                    **parameter_inventory,
                    "model_class": type(model).__name__,
                    "conductance_layers": 1,
                    "hidden_channels": (
                        model.estimator.network[0].out_features
                        if model.estimator is not None
                        else 1
                    ),
                    "attention_heads": None,
                    "attention_heads_reason": "sparse conductance operator has no attention heads",
                },
                "optimizer_integrity": optimizer_integrity,
                "optimization": {
                    "epochs": epochs,
                    "optimizer_steps": optimizer_steps,
                    "expected_optimizer_steps": expected_steps,
                    "learning_rate": learning_rate,
                    "weight_decay": 1.0e-5,
                    "gradient_clip_norm": 5.0,
                    "precision": "float16_autocast" if amp else "float32",
                    "best_validation_loss": best_validation,
                },
                "loader": {
                    "physical_batch_size_graphs": batch_size,
                    "gradient_accumulation_steps": 1,
                    "data_parallel_workers": 1,
                    "effective_batch_size_graphs": batch_size,
                    "num_workers": num_workers,
                    "persistent_workers": num_workers > 0,
                    "prefetch_factor": 2 if num_workers > 0 else None,
                    "pin_memory": pin_memory,
                    "non_blocking_transfer": pin_memory,
                    "collate": "pack_graph_examples_disjoint_union",
                    "sampling_ratio": 1.0,
                    "drop_last": False,
                },
                "data": {
                    "train": _example_statistics(train_examples, public=False),
                    "validation": _example_statistics(validation_examples, public=False),
                },
                "first_optimizer_step": first_step,
                "path_integrity": {
                    "input_forward_loss_backward_optimizer": True,
                    "validation_metric_evaluated": True,
                    "checkpoint_selected_by_validation_only": True,
                },
                "timing_seconds": {
                    "training_including_validation": training_seconds,
                    "data_wait_during_training": data_wait_seconds,
                    "validation": validation_seconds,
                },
                "throughput": {
                    "training_graphs_processed": graphs_processed,
                    "graphs_per_second_including_validation": (
                        graphs_processed / training_seconds if training_seconds > 0 else None
                    ),
                    "optimizer_steps_per_second_including_validation": (
                        optimizer_steps / training_seconds if training_seconds > 0 else None
                    ),
                },
                "debug_subset_fast_mode": False,
            }
        )
    return history


def _pearson(first: Tensor, second: Tensor) -> float | None:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    if first.numel() < 2:
        return None
    first = first - first.mean()
    second = second - second.mean()
    first_tolerance = 1.0e-7 * max(float(first.abs().max()), 1.0)
    second_tolerance = 1.0e-7 * max(float(second.abs().max()), 1.0)
    if float(first.norm()) <= first_tolerance or float(second.norm()) <= second_tolerance:
        return None
    denominator = first.norm() * second.norm()
    if float(denominator) <= torch.finfo(torch.float32).eps:
        return None
    return float(torch.dot(first, second) / denominator)


def _rank(values: Tensor) -> Tensor:
    # Synthetic conductances are continuous; ties are vanishingly rare.  The
    # deterministic stable ordering is sufficient for this diagnostic.
    order = torch.argsort(values.reshape(-1), stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
    return ranks


def _segment_sum(values: Tensor, index: Tensor, groups: int) -> Tensor:
    shape = (groups, *values.shape[1:])
    result = values.new_zeros(shape)
    result.index_add_(0, index, values)
    return result


def _segment_mean(values: Tensor, index: Tensor, groups: int) -> Tensor:
    total = _segment_sum(values, index, groups)
    counts = torch.bincount(index, minlength=groups).to(values)
    if bool(torch.any(counts == 0)):
        raise ValueError("segmented metric received an empty graph")
    return total / counts.reshape(-1, *([1] * (values.ndim - 1)))


def _segmented_relative_l2(
    prediction: Tensor, target: Tensor, index: Tensor, groups: int
) -> Tensor:
    numerator = _segment_sum((prediction - target).square(), index, groups).flatten(1).sum(1)
    denominator = _segment_sum(target.square(), index, groups).flatten(1).sum(1)
    epsilon = torch.finfo(prediction.dtype).eps
    return numerator.sqrt() / denominator.clamp_min(epsilon).sqrt()


def _segmented_pearson(
    first: Tensor, second: Tensor, index: Tensor, groups: int
) -> Tensor:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    counts = torch.bincount(index, minlength=groups).to(first)
    first_sum = _segment_sum(first, index, groups)
    second_sum = _segment_sum(second, index, groups)
    product_sum = _segment_sum(first * second, index, groups)
    first_square_sum = _segment_sum(first.square(), index, groups)
    second_square_sum = _segment_sum(second.square(), index, groups)
    covariance = product_sum - first_sum * second_sum / counts.clamp_min(1)
    first_variance = (first_square_sum - first_sum.square() / counts.clamp_min(1)).clamp_min(0)
    second_variance = (
        second_square_sum - second_sum.square() / counts.clamp_min(1)
    ).clamp_min(0)
    denominator = (first_variance * second_variance).sqrt()
    defined = (counts >= 2) & (denominator > torch.finfo(first.dtype).eps)
    result = first.new_full((groups,), torch.nan)
    result[defined] = covariance[defined] / denominator[defined]
    return result


def _within_group_ranks(values: Tensor, index: Tensor, groups: int) -> Tensor:
    value_order = torch.argsort(values.reshape(-1), stable=True)
    grouped_order = value_order[torch.argsort(index[value_order], stable=True)]
    counts = torch.bincount(index, minlength=groups)
    if bool(torch.any(counts == 0)):
        raise ValueError("rank correlation received an empty graph")
    starts = counts.cumsum(0) - counts
    positions = torch.arange(values.numel(), device=values.device) - torch.repeat_interleave(
        starts, counts
    )
    ranks = values.new_empty(values.numel(), dtype=torch.float32)
    ranks[grouped_order] = positions.to(torch.float32)
    return ranks


def _finite_mean_or_none(values: Tensor) -> float | None:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return None
    return float(values[finite].mean())


def _mean(values: Iterable[float | None]) -> float | None:
    selected = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    return sum(selected) / len(selected) if selected else None


@torch.no_grad()
def evaluate_sparse_model(
    model: SparseIncidenceConductanceLayer,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    oracle: bool = False,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("sparse evaluation split is empty")
    model.eval()
    metric_batches: list[Tensor] = []
    cap_batches: list[Tensor] = []
    graph_id_to_index = {
        graph_id: index
        for index, graph_id in enumerate(sorted({str(example["graph_id"]) for example in examples}))
    }
    maximum_edges = max(int(example["edge_index"].shape[1]) for example in examples)
    if maximum_edges < 1:
        raise ValueError("sparse evaluation requires at least one edge per graph")
    state_keys: list[Tensor] = []
    state_values: list[Tensor] = []
    loader = _loader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=pin_memory)
        override = batch.true_conductance if oracle else None
        with _autocast(device, amp):
            predicted_next, diagnostics = model(
                batch, conductance_override=override, return_diagnostics=True
            )
        if batch.true_flux is None or batch.true_conductance is None:
            raise ValueError("sparse evaluation requires flux and conductance targets")
        if batch.true_node_message is None or batch.true_next_state is None:
            raise ValueError("sparse evaluation requires node-message and next-state targets")
        if batch.true_gradient is None:
            raise ValueError("sparse evaluation requires gradient targets")
        predicted_flux = diagnostics["edge_flux"].float()
        predicted_c = diagnostics["conductance"].float()
        true_flux = batch.true_flux.float()
        true_c = batch.true_conductance.float()
        predicted_message = diagnostics["node_message"].float()
        true_message = batch.true_node_message.float()
        current_next = predicted_next.float()
        true_next = batch.true_next_state.float()
        log_square = (
            predicted_c.clamp_min(1e-8).log() - true_c.clamp_min(1e-8).log()
        ).square()
        conductance_pearson = _segmented_pearson(
            predicted_c, true_c, batch.edge_graph, batch.num_graphs
        )
        predicted_ranks = _within_group_ranks(
            predicted_c, batch.edge_graph, batch.num_graphs
        )
        true_ranks = _within_group_ranks(true_c, batch.edge_graph, batch.num_graphs)
        rank_pearson = _segmented_pearson(
            predicted_ranks, true_ranks, batch.edge_graph, batch.num_graphs
        )
        metric_batches.append(
            torch.stack(
                (
                    _segmented_relative_l2(
                        predicted_flux, true_flux, batch.edge_graph, batch.num_graphs
                    ),
                    _segmented_relative_l2(
                        predicted_message,
                        true_message,
                        batch.node_graph,
                        batch.num_graphs,
                    ),
                    _segmented_relative_l2(
                        current_next, true_next, batch.node_graph, batch.num_graphs
                    ),
                    _segment_mean(log_square, batch.edge_graph, batch.num_graphs).sqrt(),
                    conductance_pearson,
                    rank_pearson,
                    _segment_mean(
                        (batch.true_gradient.abs().amax(dim=1) > 1.0e-6).float(),
                        batch.edge_graph,
                        batch.num_graphs,
                    ),
                ),
                dim=1,
            ).cpu()
        )
        cap_batches.append(diagnostics["cap_active"].float().cpu())

        global_graphs = torch.tensor(
            [graph_id_to_index[graph_id] for graph_id in batch.graph_ids],
            dtype=torch.long,
            device=batch.edge_graph.device,
        )
        global_edge_graphs = global_graphs.index_select(0, batch.edge_graph)
        edge_counts = torch.bincount(batch.edge_graph, minlength=batch.num_graphs)
        edge_starts = edge_counts.cumsum(0) - edge_counts
        local_edges = torch.arange(batch.num_edges, device=batch.edge_graph.device) - (
            edge_starts.index_select(0, batch.edge_graph)
        )
        state_keys.append((global_edge_graphs * maximum_edges + local_edges).cpu())
        state_values.append(predicted_c.cpu())

    metrics = torch.cat(metric_batches, dim=0)
    if metrics.shape[0] != len(examples):
        raise RuntimeError(
            f"evaluation produced {metrics.shape[0]} graph metrics for {len(examples)} examples"
        )
    keys = torch.cat(state_keys)
    values = torch.cat(state_values)
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    observations = torch.bincount(inverse, minlength=unique_keys.numel()).float()
    sums = _segment_sum(values, inverse, unique_keys.numel())
    square_sums = _segment_sum(values.square(), inverse, unique_keys.numel())
    variances = (square_sums / observations - (sums / observations).square()).clamp_min(0)
    repeated = observations > 1
    if bool(repeated.any()):
        repeated_graphs = torch.div(
            unique_keys[repeated], maximum_edges, rounding_mode="floor"
        )
        graph_count = len(graph_id_to_index)
        variation_sums = _segment_sum(variances[repeated].sqrt(), repeated_graphs, graph_count)
        variation_counts = torch.bincount(repeated_graphs, minlength=graph_count).float()
        graph_has_repeated = variation_counts > 0
        state_variation = float(
            (variation_sums[graph_has_repeated] / variation_counts[graph_has_repeated]).mean()
        )
    else:
        state_variation = None
    cap_values = torch.cat(cap_batches)
    return {
        "graph_macro_flux_relative_l2": float(metrics[:, 0].mean()),
        "graph_macro_node_message_relative_l2": float(metrics[:, 1].mean()),
        "graph_macro_next_state_relative_l2": float(metrics[:, 2].mean()),
        "graph_macro_log_conductance_rmse": float(metrics[:, 3].mean()),
        "graph_macro_conductance_pearson": _finite_mean_or_none(metrics[:, 4]),
        "conductance_pearson_defined_fraction": float(torch.isfinite(metrics[:, 4]).float().mean()),
        "graph_macro_conductance_spearman": _finite_mean_or_none(metrics[:, 5]),
        "excited_edge_fraction": float(metrics[:, 6].mean()),
        "mean_conductance_state_variation": state_variation,
        "stability_cap_activation_fraction": float(cap_values.mean()),
        "num_examples": len(examples),
        "num_graph_ids": len(graph_id_to_index),
        "evaluation_batching": "vectorized sparse disjoint-union; no per-graph model forward",
    }


def least_squares_metrics(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-graph flux LS using the evaluated excitations (diagnostic ceiling)."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example["graph_id"]), []).append(example)
    flux_errors: list[float] = []
    log_errors: list[float] = []
    correlations: list[float | None] = []
    identifiable: list[float] = []
    for group in groups.values():
        numerator = None
        denominator = None
        for example in group:
            gradient = example["true_gradient"].float()
            observed = example.get("observed_flux", example["true_flux"]).float()
            current_numerator = (gradient * observed).sum(dim=1)
            current_denominator = gradient.square().sum(dim=1)
            numerator = current_numerator if numerator is None else numerator + current_numerator
            denominator = (
                current_denominator if denominator is None else denominator + current_denominator
            )
        assert numerator is not None and denominator is not None
        estimated = (numerator / denominator.clamp_min(1.0e-12)).clamp_min(1.0e-6)
        truth = group[0]["true_conductance"].float()
        excited = denominator > 1.0e-10
        identifiable.append(float(excited.float().mean()))
        if excited.any():
            log_errors.append(
                float(((estimated[excited].log() - truth[excited].log()).square().mean()).sqrt())
            )
            correlations.append(_pearson(estimated[excited], truth[excited]))
        for example in group:
            gradient = example["true_gradient"].float()
            truth_flux = example["true_flux"].float()
            predicted_flux = estimated[:, None] * gradient
            flux_errors.append(
                float((predicted_flux - truth_flux).norm() / truth_flux.norm().clamp_min(1.0e-12))
            )
    return {
        "protocol": "transductive_same-evaluation-excitations_identification_ceiling",
        "graph_macro_flux_relative_l2": _mean(flux_errors),
        "graph_macro_log_conductance_rmse": _mean(log_errors),
        "graph_macro_conductance_pearson": _mean(correlations),
        "identifiable_edge_fraction": _mean(identifiable),
        "num_graph_ids": len(groups),
    }


def _node_message_design(example: Mapping[str, Any]) -> Tensor:
    """Dense diagnostic design only; the learned layer remains gather/scatter sparse."""

    edge_index = example["edge_index"].long().cpu()
    gradient = example["true_gradient"].double().cpu()
    num_nodes = int(example["node_state"].shape[0])
    channels = int(gradient.shape[1])
    num_edges = int(edge_index.shape[1])
    design = gradient.new_zeros((num_nodes * channels, num_edges))
    edge_ids = torch.arange(num_edges).view(-1, 1).expand(-1, channels)
    channel_ids = torch.arange(channels).view(1, -1)
    tail_rows = edge_index[0].view(-1, 1) * channels + channel_ids
    head_rows = edge_index[1].view(-1, 1) * channels + channel_ids
    design[tail_rows.reshape(-1), edge_ids.reshape(-1)] = -gradient.reshape(-1)
    design[head_rows.reshape(-1), edge_ids.reshape(-1)] = gradient.reshape(-1)
    return design


def _projected_nnls(
    design: Tensor,
    target: Tensor,
    *,
    max_iterations: int = 1_000,
    tolerance: float = 1.0e-10,
) -> tuple[Tensor, int]:
    """Solve nonnegative least squares with deterministic projected FISTA."""

    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise ValueError("NNLS design and target shapes are inconsistent")
    if design.shape[1] == 0:
        return design.new_empty(0), 0
    spectral = torch.linalg.svdvals(design)
    lipschitz = spectral[0].square() if spectral.numel() else design.new_tensor(0.0)
    if float(lipschitz) <= torch.finfo(design.dtype).eps:
        return design.new_zeros(design.shape[1]), 0
    # The unconstrained solution is already the exact NNLS solution when it is
    # nonnegative.  This makes the noiseless, full-rank ceiling numerically sharp.
    unconstrained = torch.linalg.lstsq(design, target).solution
    if bool(torch.all(unconstrained >= 0)):
        return unconstrained, 0
    estimate = unconstrained.clamp_min(0)
    accelerated = estimate.clone()
    momentum = 1.0
    scale = max(float(estimate.norm()), 1.0)
    for iteration in range(1, max_iterations + 1):
        gradient = design.mT @ (design @ accelerated - target)
        updated = (accelerated - gradient / lipschitz).clamp_min(0)
        if float((updated - estimate).norm()) <= tolerance * scale:
            return updated, iteration
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - estimate)
        estimate = updated
        momentum = next_momentum
        scale = max(float(estimate.norm()), 1.0)
    raise RuntimeError(
        "projected NNLS did not converge within the explicitly declared "
        f"{max_iterations} iterations; refusing to report an unconverged ceiling"
    )


def node_message_nnls_metrics(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Same-evaluation node-output NNLS ceiling for one conductance per edge.

    Unlike :func:`least_squares_metrics`, this diagnostic never reads observed
    per-edge flux.  It estimates nonnegative edge conductances only from the
    observed node messages and the known excitation gradients.  It is still a
    transductive ceiling, not a held-graph predictive baseline.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example["graph_id"]), []).append(example)
    clean_errors: list[float] = []
    observed_fit_errors: list[float] = []
    log_errors: list[float] = []
    correlations: list[float | None] = []
    excited_fractions: list[float] = []
    rank_fractions: list[float] = []
    iterations: list[float] = []
    for group in groups.values():
        reference_edges = group[0]["edge_index"]
        reference_truth = group[0]["true_conductance"].double().cpu()
        designs: list[Tensor] = []
        observed_targets: list[Tensor] = []
        for example in group:
            if not torch.equal(example["edge_index"], reference_edges):
                raise ValueError("examples sharing graph_id must share edge_index")
            truth = example["true_conductance"].double().cpu()
            if not torch.allclose(truth, reference_truth):
                raise ValueError("node-message NNLS requires static conductance per graph_id")
            design = _node_message_design(example)
            observed = example.get("observed_node_message")
            if observed is None:
                observed = example["true_node_message"]
            designs.append(design)
            observed_targets.append(observed.double().cpu().reshape(-1))
        stacked_design = torch.cat(designs, dim=0)
        stacked_target = torch.cat(observed_targets, dim=0)
        estimated, used_iterations = _projected_nnls(stacked_design, stacked_target)
        iterations.append(float(used_iterations))
        column_energy = stacked_design.square().sum(dim=0)
        excited = column_energy > 1.0e-12
        excited_fractions.append(float(excited.float().mean()))
        rank = int(torch.linalg.matrix_rank(stacked_design))
        rank_fractions.append(rank / max(stacked_design.shape[1], 1))
        if excited.any():
            log_errors.append(
                float(
                    (
                        estimated[excited].clamp_min(1.0e-12).log()
                        - reference_truth[excited].clamp_min(1.0e-12).log()
                    )
                    .square()
                    .mean()
                    .sqrt()
                )
            )
            correlations.append(_pearson(estimated[excited], reference_truth[excited]))
        for example, design, observed_target in zip(group, designs, observed_targets, strict=True):
            predicted = design @ estimated
            clean_target = example["true_node_message"].double().cpu().reshape(-1)
            clean_errors.append(
                float((predicted - clean_target).norm() / clean_target.norm().clamp_min(1.0e-12))
            )
            observed_fit_errors.append(
                float(
                    (predicted - observed_target).norm() / observed_target.norm().clamp_min(1.0e-12)
                )
            )
    return {
        "protocol": "transductive_same-evaluation-node-messages_nnls_ceiling",
        "graph_macro_node_message_relative_l2": _mean(clean_errors),
        "graph_macro_observed_fit_relative_l2": _mean(observed_fit_errors),
        "graph_macro_log_conductance_rmse": _mean(log_errors),
        "graph_macro_conductance_pearson": _mean(correlations),
        "excited_edge_fraction": _mean(excited_fractions),
        "design_rank_fraction": _mean(rank_fractions),
        "mean_solver_iterations": _mean(iterations),
        "num_graph_ids": len(groups),
    }


@torch.no_grad()
def evaluate_rollout(
    model: SparseIncidenceConductanceLayer,
    trajectories: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    device: torch.device,
    amp: bool,
    oracle: bool,
) -> dict[str, Any]:
    if not trajectories or not horizons:
        raise ValueError("rollout evaluation requires trajectories and horizons")
    normalized_horizons = sorted({int(horizon) for horizon in horizons})
    if normalized_horizons[0] < 1:
        raise ValueError("rollout horizons must be positive")
    maximum_horizon = normalized_horizons[-1]
    if any(int(trajectory["states"].shape[0]) <= maximum_horizon for trajectory in trajectories):
        raise ValueError("a rollout trajectory is shorter than the requested horizon")
    if any(len(trajectory["steps"]) < maximum_horizon for trajectory in trajectories):
        raise ValueError("a rollout trajectory has too few integration steps")
    initial_records = [
        {
            "graph_id": trajectory["graph_id"],
            "node_state": trajectory["states"][0],
            "edge_index": trajectory["edge_index"],
            "edge_features": trajectory["edge_features"],
            "step_size": float(trajectory["steps"][0]),
        }
        for trajectory in trajectories
    ]
    batch = pack_graph_examples(initial_records).to(device)
    state = batch.node_state
    initial_norm = _segment_sum(
        state.float().square(), batch.node_graph, batch.num_graphs
    ).sum(dim=1).sqrt()
    if bool(torch.any(initial_norm <= 0)):
        raise ValueError("rollout initial states must have nonzero norm")
    previous_norm = initial_norm
    errors: dict[int, Tensor] = {}
    dissipation_violations = state.new_zeros(batch.num_graphs, dtype=torch.float32)
    cap_active = state.new_zeros(batch.num_graphs, dtype=torch.float32)
    step_matrix = torch.tensor(
        [
            [float(trajectory["steps"][index]) for index in range(maximum_horizon)]
            for trajectory in trajectories
        ],
        device=device,
        dtype=state.dtype,
    )
    for time_index in range(maximum_horizon):
        batch.requested_step = step_matrix[:, time_index]
        override = None
        if oracle:
            override = nonlinear_conductance(
                batch.edge_features, edge_gradient(batch.edge_index, state)
            )
        with _autocast(device, amp):
            state, diagnostics = model(
                batch,
                node_state=state,
                conductance_override=override,
                return_diagnostics=True,
            )
        current_norm = _segment_sum(
            state.float().square(), batch.node_graph, batch.num_graphs
        ).sum(dim=1).sqrt()
        dissipation_violations += (current_norm > previous_norm + 1.0e-6).float()
        previous_norm = current_norm
        cap_active += diagnostics["cap_active"].float()
        horizon = time_index + 1
        if horizon in normalized_horizons:
            truth = torch.cat(
                [trajectory["states"][horizon] for trajectory in trajectories], dim=0
            ).to(device)
            errors[horizon] = _segmented_relative_l2(
                state.float(), truth.float(), batch.node_graph, batch.num_graphs
            )
    result = {
        f"horizon_{horizon}_relative_l2": float(errors[horizon].mean())
        for horizon in normalized_horizons
    }
    result.update(
        {
            "final_norm_over_initial": float((previous_norm / initial_norm).mean()),
            "dissipation_violation_fraction": float(
                (dissipation_violations / maximum_horizon).mean()
            ),
            "stability_cap_activation_fraction": float((cap_active / maximum_horizon).mean()),
            "evaluation_batching": (
                "all independent trajectories packed into one sparse disjoint-union forward "
                "per time step"
            ),
        }
    )
    return result


def _model_for_examples(
    examples: Sequence[Mapping[str, Any]], mode: str, *, hidden_channels: int
) -> SparseIncidenceConductanceLayer:
    first = examples[0]
    return SparseIncidenceConductanceLayer(
        channels=int(first["node_state"].shape[1]),
        edge_feature_channels=int(first["edge_features"].shape[1]),
        hidden_channels=hidden_channels,
        requested_step=0.025,
        stability_margin=0.95,
        adaptive_stability=True,
        mode=mode,
    )


def _factorial_key(example: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    metadata = example["metadata"]
    return metadata["contrast"], metadata["active_node_fraction"], metadata["snr_db"]


def run_core(
    core: dict[str, Any],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Tensor]]]:
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    states: dict[str, dict[str, Tensor]] = {}
    baseline_specs = (
        ("isotropic", "isotropic", "node_only", "constant-conductance ablation"),
        ("edge_only", "edge_only", "node_only", "static edge-feature ablation"),
        (
            "gradient_only",
            "gradient_only",
            "node_only",
            "state-gradient-only ablation C=f(abs(BH))",
        ),
        (
            "full",
            "full",
            "node_only",
            "headline node-output-only predictive model",
        ),
        (
            "full_flux_supervised",
            "full",
            "flux_only",
            "per-edge-flux-supervised neural ceiling",
        ),
        ("full_joint", "full", "joint", "joint-supervision objective ablation"),
    )
    mode_seed_offset = {"isotropic": 0, "edge_only": 1, "gradient_only": 2, "full": 3}
    for suite_number, suite_name in enumerate(("s1", "s2", "s3", "s4")):
        suite = core[suite_name]
        train_examples = suite["train"]
        validation_examples = suite["validation"]
        test_examples = suite["test"]
        hidden_channels = 64
        suite_result: dict[str, Any] = {
            "claim": CORE_CLAIMS[suite_name],
            "description": suite["description"],
            "split_graph_counts": {
                split: len({item["graph_id"] for item in suite.get(split, [])})
                for split in ("train", "validation", "test", "seen_test")
                if split in suite
            },
            "headline_baseline": "full",
            "objective_protocol": {
                "headline": "node_only",
                "flux_supervised_ceiling": "full_flux_supervised",
                "joint_objective_ablation": "full_joint",
            },
            "baselines": {},
        }
        trained: dict[str, tuple[SparseIncidenceConductanceLayer, str]] = {}
        for baseline_name, mode, objective, role in baseline_specs:
            initialization_offset = mode_seed_offset[mode]
            seed_everything(seed + suite_number * 100 + initialization_offset)
            model = _model_for_examples(train_examples, mode, hidden_channels=hidden_channels).to(
                device
            )
            execution: dict[str, Any] = {}
            history = train_sparse_model(
                model,
                train_examples,
                validation_examples,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=num_workers,
                seed=seed + suite_number * 1000 + initialization_offset * 100,
                objective=objective,
                execution_report=execution,
            )
            for row in history:
                histories.append({"suite": suite_name, "baseline": baseline_name, **row})
            trained[baseline_name] = (model, objective)
            states[f"{suite_name}_{baseline_name}"] = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            metric = evaluate_sparse_model(
                model,
                test_examples,
                device=device,
                amp=amp,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            execution["data"]["test"] = _example_statistics(test_examples, public=False)
            execution["path_integrity"]["test_metric_evaluated_after_checkpoint_restore"] = True
            suite_result["baselines"][baseline_name] = {
                "training_objective": objective,
                "role": role,
                "unseen_graph_test": metric,
                "execution": execution,
            }
            if suite_name == "s1":
                suite_result["baselines"][baseline_name]["seen_graph_new_excitation_test"] = (
                    evaluate_sparse_model(
                        model,
                        suite["seen_test"],
                        device=device,
                        amp=amp,
                        batch_size=batch_size,
                        pin_memory=pin_memory,
                        num_workers=num_workers,
                    )
                )
            if suite_name == "s3":
                suite_result["baselines"][baseline_name]["rollout"] = evaluate_rollout(
                    model,
                    suite["rollout_test"],
                    suite["horizons"],
                    device=device,
                    amp=amp,
                    oracle=False,
                )
        oracle_model = trained["full"][0]
        suite_result["baselines"]["oracle"] = {
            "training_objective": "analytic_oracle",
            "role": "ground-truth conductance oracle",
            "model": (
                "reuses the already-trained full model's sparse propagation scaffold; "
                "ground-truth conductance overrides the estimator during oracle evaluation, "
                "so no separate unused oracle parameters are instantiated"
            ),
            "optimizer": "not_applicable: evaluation-only analytic conductance override",
            "unseen_graph_test": evaluate_sparse_model(
                oracle_model,
                test_examples,
                device=device,
                amp=amp,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                oracle=True,
            ),
        }
        if suite_name == "s1":
            suite_result["baselines"]["oracle"]["seen_graph_new_excitation_test"] = (
                evaluate_sparse_model(
                    oracle_model,
                    suite["seen_test"],
                    device=device,
                    amp=amp,
                    batch_size=batch_size,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    oracle=True,
                )
            )
        if suite_name == "s3":
            suite_result["baselines"]["oracle"]["rollout"] = evaluate_rollout(
                oracle_model,
                suite["rollout_test"],
                suite["horizons"],
                device=device,
                amp=amp,
                oracle=True,
            )
        if suite_name in {"s1", "s4"}:
            suite_result["baselines"]["flux_ls"] = {
                "training_objective": "same-evaluation observed edge flux",
                "role": "transductive per-edge-flux least-squares ceiling",
                "unseen_graph_test": least_squares_metrics(test_examples),
            }
            suite_result["baselines"]["node_message_nnls"] = {
                "training_objective": "same-evaluation observed node message",
                "role": "transductive node-output nonnegative least-squares ceiling",
                "unseen_graph_test": node_message_nnls_metrics(test_examples),
            }
            if suite_name == "s1":
                suite_result["baselines"]["flux_ls"]["seen_graph_new_excitation_test"] = (
                    least_squares_metrics(suite["seen_test"])
                )
                suite_result["baselines"]["node_message_nnls"]["seen_graph_new_excitation_test"] = (
                    node_message_nnls_metrics(suite["seen_test"])
                )
        if suite_name == "s4":
            factorial: list[dict[str, Any]] = []
            keys = sorted({_factorial_key(example) for example in test_examples}, key=str)
            for key in keys:
                subset = [example for example in test_examples if _factorial_key(example) == key]
                for baseline_name, (model, objective) in trained.items():
                    factorial.append(
                        {
                            "contrast": key[0],
                            "active_node_fraction": key[1],
                            "snr_db": key[2],
                            "baseline": baseline_name,
                            "training_objective": objective,
                            **evaluate_sparse_model(
                                model,
                                subset,
                                device=device,
                                amp=amp,
                                batch_size=batch_size,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                            ),
                        }
                    )
                factorial.append(
                    {
                        "contrast": key[0],
                        "active_node_fraction": key[1],
                        "snr_db": key[2],
                        "baseline": "flux_ls",
                        "training_objective": "same-evaluation observed edge flux",
                        **least_squares_metrics(subset),
                    }
                )
                factorial.append(
                    {
                        "contrast": key[0],
                        "active_node_fraction": key[1],
                        "snr_db": key[2],
                        "baseline": "node_message_nnls",
                        "training_objective": "same-evaluation observed node message",
                        **node_message_nnls_metrics(subset),
                    }
                )
            suite_result["factorial"] = factorial
        results[suite_name] = suite_result
    return results, histories, states


@dataclass
class PublicPacked:
    x: Tensor
    edge_index: Tensor
    edge_features: Tensor
    node_graph: Tensor
    y: Tensor
    graph_ids: list[str]
    task: str
    categorical: bool

    @property
    def num_graphs(self) -> int:
        return len(self.graph_ids)

    def to(self, device: torch.device, *, non_blocking: bool) -> PublicPacked:
        return PublicPacked(
            x=self.x.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            edge_features=self.edge_features.to(device, non_blocking=non_blocking),
            node_graph=self.node_graph.to(device, non_blocking=non_blocking),
            y=self.y.to(device, non_blocking=non_blocking),
            graph_ids=self.graph_ids,
            task=self.task,
            categorical=self.categorical,
        )

    def pin_memory(self) -> PublicPacked:
        return PublicPacked(
            x=self.x.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            edge_features=self.edge_features.pin_memory(),
            node_graph=self.node_graph.pin_memory(),
            y=self.y.pin_memory(),
            graph_ids=self.graph_ids,
            task=self.task,
            categorical=self.categorical,
        )


def pack_public(records: Sequence[Mapping[str, Any]]) -> PublicPacked:
    if not records:
        raise ValueError("empty public batch")
    task = str(records[0]["task"])
    categorical = bool(records[0]["categorical"])
    nodes: list[Tensor] = []
    edges: list[Tensor] = []
    edge_features: list[Tensor] = []
    node_graph: list[Tensor] = []
    labels: list[Tensor] = []
    graph_ids: list[str] = []
    offset = 0
    for graph_number, record in enumerate(records):
        if record["task"] != task or bool(record["categorical"]) != categorical:
            raise ValueError("public batch mixes tasks or feature types")
        x = record["x"]
        nodes.append(x)
        edges.append(record["edge_index"] + offset)
        edge_features.append(record["edge_features"])
        node_graph.append(torch.full((x.shape[0],), graph_number, dtype=torch.long))
        labels.append(record["y"])
        graph_ids.append(str(record["graph_id"]))
        offset += int(x.shape[0])
    y = (
        torch.cat(labels)
        if task == "node"
        else torch.stack([label.reshape(-1) for label in labels])
    )
    return PublicPacked(
        x=torch.cat(nodes),
        edge_index=torch.cat(edges, dim=1),
        edge_features=torch.cat(edge_features),
        node_graph=torch.cat(node_graph),
        y=y,
        graph_ids=graph_ids,
        task=task,
        categorical=categorical,
    )


class SumCategoricalEncoder(nn.Module):
    def __init__(self, columns: int, hidden: int, categories: int = 256) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, hidden) for _ in range(columns)])

    def forward(self, values: Tensor) -> Tensor:
        result = self.embeddings[0](values[:, 0].long())
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            result = result + embedding(values[:, column].long())
        return result


class PublicConductanceModel(nn.Module):
    def __init__(
        self,
        sample: Mapping[str, Any],
        *,
        hidden: int,
        num_classes: int,
        official_molecule: bool,
    ) -> None:
        super().__init__()
        node_width = int(sample["x"].shape[1])
        edge_width = int(sample["edge_features"].shape[1])
        self.task = str(sample["task"])
        if bool(sample["categorical"]) and official_molecule:
            try:
                from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
            except (ImportError, OSError) as error:  # pragma: no cover - optional path
                raise RuntimeError(
                    "official MolHIV requires OGB AtomEncoder/BondEncoder"
                ) from error
            self.node_encoder = AtomEncoder(hidden)
            self.edge_encoder = BondEncoder(hidden)
        elif bool(sample["categorical"]):
            self.node_encoder = SumCategoricalEncoder(node_width, hidden)
            self.edge_encoder = SumCategoricalEncoder(edge_width, hidden)
        else:
            self.node_encoder = nn.Linear(node_width, hidden)
            self.edge_encoder = nn.Linear(edge_width, hidden)
        self.uses_edge_features = True
        self.normalization = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes if self.task == "node" else 1)
        self.layer = SparseIncidenceConductanceLayer(
            channels=hidden,
            edge_feature_channels=hidden,
            hidden_channels=hidden,
            requested_step=0.02,
            mode="full",
        )

    def forward(self, batch: PublicPacked) -> Tensor:
        node_state = self.node_encoder(batch.x)
        edge_features = self.edge_encoder(batch.edge_features)
        edge_graph = batch.node_graph.index_select(0, batch.edge_index[0])
        sparse_batch = PackedGraphBatch(
            node_state=node_state,
            edge_index=batch.edge_index,
            edge_features=edge_features,
            node_graph=batch.node_graph,
            edge_graph=edge_graph,
            graph_ids=batch.graph_ids,
            requested_step=node_state.new_full((batch.num_graphs,), 0.02),
        )
        node_state = self.layer(sparse_batch)
        node_state = nnf.silu(self.normalization(node_state))
        if self.task == "node":
            return self.head(node_state)
        pooled = node_state.new_zeros((batch.num_graphs, node_state.shape[1]))
        pooled.index_add_(0, batch.node_graph, node_state)
        counts = torch.bincount(batch.node_graph, minlength=batch.num_graphs).to(node_state)
        return self.head(pooled / counts[:, None].clamp_min(1)).squeeze(-1)


def _public_loader(
    dataset: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=pack_public,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _public_loss(logits: Tensor, labels: Tensor, task: str) -> Tensor:
    if task == "node":
        return nnf.cross_entropy(logits, labels.long())
    valid = torch.isfinite(labels.reshape(-1))
    _require_true_async(valid.any(), "graph-property batch contains no finite labels")
    return nnf.binary_cross_entropy_with_logits(logits[valid], labels.reshape(-1)[valid].float())


def _public_loss_weight(labels: Tensor, task: str) -> int:
    """Return the number of labels represented by a mean-reduced task loss."""

    if task == "node":
        return int(labels.numel())
    return int(torch.isfinite(labels.reshape(-1)).sum())


def _macro_f1(predictions: Tensor, labels: Tensor) -> float:
    scores = []
    for label in torch.unique(labels):
        true_positive = ((predictions == label) & (labels == label)).sum().float()
        false_positive = ((predictions == label) & (labels != label)).sum().float()
        false_negative = ((predictions != label) & (labels == label)).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator > 0:
            scores.append(float(2 * true_positive / denominator))
    if not scores:
        raise ValueError("macro-F1 requires at least one represented class")
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_public(
    model: PublicConductanceModel,
    dataset: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
) -> dict[str, Any]:
    if not dataset:
        raise ValueError("public evaluation split is empty")
    model.eval()
    outputs: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in _public_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    ):
        batch = batch.to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            outputs.append(model(batch).float().cpu())
        labels.append(batch.y.float().cpu())
    output = torch.cat(outputs)
    label = torch.cat(labels)
    if model.task == "node":
        return {
            "macro_f1": _macro_f1(output.argmax(dim=1), label.long()),
            "num_labels": label.numel(),
        }
    try:
        from ogb.graphproppred import Evaluator
    except (ImportError, OSError) as error:  # pragma: no cover - optional path
        raise RuntimeError("official MolHIV evaluation requires the OGB evaluator") from error
    evaluator = Evaluator(name="ogbg-molhiv")
    score = evaluator.eval({"y_true": label.reshape(-1, 1), "y_pred": output.reshape(-1, 1)})[
        "rocauc"
    ]
    return {
        "roc_auc": float(score),
        "num_graphs": label.numel(),
        "evaluator": "ogb.graphproppred.Evaluator",
    }


def _train_public_model(
    model: PublicConductanceModel,
    train_dataset: Sequence[Mapping[str, Any]],
    validation_dataset: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Tensor], float]:
    if not train_dataset or not validation_dataset:
        raise ValueError("public training and validation splits must both be nonempty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    parameter_inventory = _parameter_inventory(model)
    optimizer_integrity = _optimizer_integrity(model, optimizer)
    scaler = _grad_scaler(amp)
    train_loader = _public_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    validation_loader = _public_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    best_validation = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    graphs_processed = 0
    labels_processed = 0
    data_wait_seconds = 0.0
    validation_seconds = 0.0
    first_step: dict[str, Any] | None = None
    _synchronize(device)
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total: Tensor | None = None
        count = 0
        epoch_graphs = 0
        if train_loader.generator is None:
            raise RuntimeError("public training DataLoader has no deterministic generator")
        train_loader.generator.manual_seed(seed + epoch)
        wait_started = time.perf_counter()
        for batch in train_loader:
            data_wait_seconds += time.perf_counter() - wait_started
            loss_weight = _public_loss_weight(batch.y, model.task)
            if loss_weight < 1:
                raise ValueError("public training batch contains no valid target labels")
            profile_this_step = first_step is None
            if profile_this_step:
                _synchronize(device)
                transfer_started = time.perf_counter()
            batch = batch.to(device, non_blocking=pin_memory)
            if profile_this_step:
                _synchronize(device)
                transfer_seconds = time.perf_counter() - transfer_started
                forward_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss = _public_loss(model(batch), batch.y, model.task)
            _require_finite_async(
                loss.detach(),
                f"public training loss at epoch={epoch}, optimizer_step={optimizer_steps + 1}",
            )
            if profile_this_step:
                _synchronize(device)
                forward_seconds = time.perf_counter() - forward_started
                backward_started = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_integrity = _gradient_integrity(model) if profile_this_step else None
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            _require_finite_async(
                gradient_norm,
                f"public clipped gradient norm at epoch={epoch}, "
                f"optimizer_step={optimizer_steps + 1}",
            )
            if profile_this_step:
                _synchronize(device)
                backward_seconds = time.perf_counter() - backward_started
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                optimizer_started = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            if profile_this_step:
                _synchronize(device)
                optimizer_seconds = time.perf_counter() - optimizer_started
                first_step = _first_step_profile(
                    batch_description={
                        "graphs": batch.num_graphs,
                        "nodes": int(batch.x.shape[0]),
                        "edges": int(batch.edge_index.shape[1]),
                        "x_shape": list(batch.x.shape),
                        "edge_index_shape": list(batch.edge_index.shape),
                        "edge_features_shape": list(batch.edge_features.shape),
                        "target_shape": list(batch.y.shape),
                        "valid_target_labels": loss_weight,
                        "loss": float(loss.detach().float().cpu()),
                    },
                    gradient=gradient_integrity or {},
                    before=before,
                    model=model,
                    timing_seconds={
                        "host_to_device": transfer_seconds,
                        "forward_and_loss": forward_seconds,
                        "backward_and_gradient_validation": backward_seconds,
                        "optimizer": optimizer_seconds,
                    },
                )
            weighted = loss.detach().float() * loss_weight
            total = weighted if total is None else total + weighted
            count += loss_weight
            labels_processed += loss_weight
            graphs_processed += batch.num_graphs
            epoch_graphs += batch.num_graphs
            wait_started = time.perf_counter()
        if epoch_graphs != len(train_dataset):
            raise RuntimeError(
                f"public training loader consumed {epoch_graphs} graphs but "
                f"{len(train_dataset)} were required"
            )
        if total is None or count < 1:
            raise RuntimeError(f"public training epoch {epoch} produced no valid loss")

        model.eval()
        validation_total: Tensor | None = None
        validation_count = 0
        validation_graphs = 0
        _synchronize(device)
        validation_started = time.perf_counter()
        with torch.no_grad():
            for batch in validation_loader:
                loss_weight = _public_loss_weight(batch.y, model.task)
                if loss_weight < 1:
                    raise ValueError("public validation batch contains no valid target labels")
                batch = batch.to(device, non_blocking=pin_memory)
                with _autocast(device, amp):
                    loss = _public_loss(model(batch), batch.y, model.task)
                _require_finite_async(loss, f"public validation loss at epoch={epoch}")
                weighted = loss.detach().float() * loss_weight
                validation_total = (
                    weighted if validation_total is None else validation_total + weighted
                )
                validation_count += loss_weight
                validation_graphs += batch.num_graphs
        _synchronize(device)
        validation_seconds += time.perf_counter() - validation_started
        if validation_graphs != len(validation_dataset):
            raise RuntimeError(
                f"public validation loader consumed {validation_graphs} graphs but "
                f"{len(validation_dataset)} were required"
            )
        if validation_total is None or validation_count < 1:
            raise RuntimeError(f"public validation epoch {epoch} produced no valid loss")
        train_loss = float(total.cpu()) / count
        validation_loss = float(validation_total.cpu()) / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    _synchronize(device)
    training_seconds = time.perf_counter() - training_started
    expected_steps = epochs * math.ceil(len(train_dataset) / batch_size)
    if optimizer_steps != expected_steps:
        raise RuntimeError(
            f"public optimizer executed {optimizer_steps} steps but {expected_steps} were required"
        )
    if first_step is None or best_state is None or not math.isfinite(best_validation):
        raise RuntimeError("public training completed without a valid update and checkpoint")
    model.load_state_dict(best_state)
    report = {
        "model": {
            **parameter_inventory,
            "model_class": type(model).__name__,
            "node_encoder": type(model.node_encoder).__name__,
            "edge_encoder": type(model.edge_encoder).__name__,
            "hidden_channels": model.head.in_features,
            "conductance_layers": 1,
            "attention_heads": None,
            "attention_heads_reason": "sparse conductance operator has no attention heads",
            "task": model.task,
        },
        "optimizer_integrity": optimizer_integrity,
        "optimization": {
            "epochs": epochs,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
            "learning_rate": learning_rate,
            "weight_decay": 1.0e-5,
            "gradient_clip_norm": 5.0,
            "precision": "float16_autocast" if amp else "float32",
            "best_validation_loss": best_validation,
        },
        "loader": {
            "physical_batch_size_graphs": batch_size,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size_graphs": batch_size,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "pin_memory": pin_memory,
            "non_blocking_transfer": pin_memory,
            "collate": "pack_public_disjoint_union",
            "sampling_ratio": 1.0,
            "drop_last": False,
        },
        "data": {
            "train": _example_statistics(train_dataset, public=True),
            "validation": _example_statistics(validation_dataset, public=True),
        },
        "first_optimizer_step": first_step,
        "path_integrity": {
            "input_forward_loss_backward_optimizer": True,
            "validation_metric_evaluated": True,
            "checkpoint_selected_by_validation_only": True,
        },
        "timing_seconds": {
            "training_including_validation": training_seconds,
            "data_wait_during_training": data_wait_seconds,
            "validation": validation_seconds,
        },
        "throughput": {
            "training_graphs_processed": graphs_processed,
            "training_labels_processed": labels_processed,
            "graphs_per_second_including_validation": (
                graphs_processed / training_seconds if training_seconds > 0 else None
            ),
            "labels_per_second_including_validation": (
                labels_processed / training_seconds if training_seconds > 0 else None
            ),
            "optimizer_steps_per_second_including_validation": (
                optimizer_steps / training_seconds if training_seconds > 0 else None
            ),
        },
        "debug_subset_fast_mode": False,
    }
    return history, report, best_state, best_validation


def run_public(
    datasets: dict[str, Any],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Tensor]]]:
    if datasets.get("fixture") is not False:
        raise ValueError(
            "Public experiments require official data; generated substitutes are unsupported"
        )
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    states: dict[str, dict[str, Tensor]] = {}
    for dataset_number, dataset_name in enumerate(("pascalvoc_sp", "ogbg_molhiv")):
        splits = datasets[dataset_name]
        sample = splits["train"][0]
        num_classes = 21 if dataset_name == "pascalvoc_sp" else 3
        hidden = 96
        results[dataset_name] = {
            "fixture": False,
            "official_result": True,
            "model_protocol": {
                "hidden_channels": hidden,
                "backbone_depth": 1,
                "model": "conductance_model",
                "split": "official",
                "competitor_execution": "not implemented; published results compared externally",
            },
            "baselines": {},
        }
        model_seed = seed + dataset_number * 101
        for model_name in ("conductance_model",):
            seed_everything(model_seed)
            model = PublicConductanceModel(
                sample,
                hidden=hidden,
                num_classes=num_classes,
                official_molecule=(dataset_name == "ogbg_molhiv"),
            ).to(device)
            model_history, execution, best_state, best_validation = _train_public_model(
                model,
                splits["train"],
                splits["validation"],
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=num_workers,
                seed=seed,
            )
            histories.extend(
                {"suite": dataset_name, "baseline": model_name, **row}
                for row in model_history
            )
            state_key = f"{dataset_name}_{model_name}"
            states[state_key] = best_state
            test_metric = evaluate_public(
                model,
                splits["test"],
                device=device,
                batch_size=batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            execution["data"]["test"] = _example_statistics(splits["test"], public=True)
            execution["path_integrity"]["test_metric_evaluated_after_checkpoint_restore"] = True
            results[dataset_name]["baselines"][model_name] = {
                "parameter_count": execution["model"]["trainable_parameters"],
                "parameter_count_policy": "trainable_active_parameters_only",
                "uses_edge_features": model.uses_edge_features,
                "best_validation_loss": best_validation,
                "test": test_metric,
                "execution": execution,
            }
    return results, histories, states


def _metric_rows(value: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_metric_rows(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_metric_rows(child, (*path, str(index))))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append({"path": "/".join(path), "value": value})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cuda_allocator_peaks(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _finish_monitor_after_failure(
    monitor: RuntimeResourceMonitor,
    device: torch.device,
    primary_error: BaseException,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Finish resource sampling once without replacing the scientific failure."""

    collection_notes: list[str] = []
    try:
        peak_allocated, peak_reserved = _cuda_allocator_peaks(device)
    except BaseException as cleanup_error:
        peak_allocated, peak_reserved = None, None
        note = (
            "CUDA allocator peak collection failed during error cleanup: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
        collection_notes.append(note)
        primary_error.add_note(note)
    try:
        resources = monitor.finish(
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except BaseException as cleanup_error:
        reason = (
            "resource monitor cleanup failed without replacing the scientific error: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
        primary_error.add_note(reason)
        return None, reason, collection_notes
    if collection_notes:
        resources["collection_notes"] = list(collection_notes)
    return resources, None, collection_notes


def _execution_plan(
    *,
    suite: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    num_workers: int,
    amp: bool,
    pin_memory: bool,
    device: torch.device,
    data: Mapping[str, Any],
    resource_start: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    selected_models: dict[str, Any] = {}
    if suite in {"core", "all"}:
        selected_models["core"] = {
            "hidden_channels": 64,
            "conductance_layers": 1,
            "conductance_estimator_linear_layers": 3,
            "attention_heads": None,
            "conditions": [
                "isotropic",
                "edge_only",
                "gradient_only",
                "full",
                "full_flux_supervised",
                "full_joint",
            ],
            "protocol_status": "supplementary mechanistic S1-S4 suite, not headline benchmark",
        }
    if suite in {"public", "all"}:
        selected_models["public"] = {
            "hidden_channels": 96,
            "conductance_layers": 1,
            "attention_heads": None,
            "datasets": ["pascalvoc_sp", "ogbg_molhiv"],
            "protocol_status": "legacy supplementary public suite, not headline benchmark",
        }
    return {
        "event": "pre_training_execution_plan",
        "suite": suite,
        "models": selected_models,
        "data": dict(data),
        "optimization": {
            "epochs_applied_without_suite_specific_cap": epochs,
            "learning_rate": learning_rate,
            "weight_decay": 1.0e-5,
            "gradient_clip_norm": 5.0,
        },
        "batching": {
            "physical_batch_size_graphs": batch_size,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size_graphs": batch_size,
            "collation": "sparse disjoint-union graph batching",
            "sampling_ratio": 1.0,
        },
        "data_loader": {
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "pin_memory": pin_memory,
            "non_blocking_transfer": pin_memory,
            "worker_policy": (
                "four-worker repository paper-runner reference default"
                if num_workers == 4
                else "explicit CLI override; interpret with measured data-wait and throughput"
            ),
        },
        "hardware": {
            "target_device": str(device),
            "precision": "float16_autocast" if amp else "float32",
            "pre_training_resource_snapshot": dict(resource_start),
            "parallelism": (
                "one selected CUDA device per process; independent top-level tracks/runs may be "
                "distributed by the parent runner"
            ),
        },
        "implementation_sha256": dict(implementation_sha256),
        "debug_subset_fast_mode": False,
        "configuration_changes_from_declared_protocol": [],
    }


def _prepare_output_dir(path: Path) -> Path:
    """Claim an empty run directory before data preparation or artifact writes."""

    resolved = path.expanduser().resolve()
    if resolved.parent == resolved:
        raise ValueError("--output-dir cannot be a filesystem root")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"--output-dir is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(
                f"--output-dir already contains artifacts; choose a new empty path: {resolved}"
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def _seed_axis_applicability(
    suite: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Describe which resolved seed axes actually affect each requested protocol."""

    applicability: dict[str, dict[str, dict[str, Any]]] = {}
    if suite in {"core", "all"}:
        applicability["core"] = {
            "data": {
                "applicable": True,
                "use": "generated graphs, excitations, trajectories, labels, and cache key",
            },
            "split": {
                "applicable": False,
                "use": "not_applicable: generated split assignment is part of data_seed",
            },
            "chart": {
                "applicable": False,
                "use": "not_applicable: conductance track has no spanning-tree chart sampling",
            },
            "model": {
                "applicable": True,
                "use": "model initialization and training DataLoader shuffle",
            },
        }
    if suite in {"public", "all"}:
        applicability["public"] = {
            "data": {
                "applicable": False,
                "use": "not_applicable: official dataset content is fixed by its source",
            },
            "split": {
                "applicable": False,
                "use": "not_applicable: official PascalVOC-SP/MolHIV splits are fixed",
            },
            "chart": {
                "applicable": False,
                "use": "not_applicable: public conductance baselines do not sample tree charts",
            },
            "model": {
                "applicable": True,
                "use": "model initialization and training DataLoader shuffle",
            },
        }
    return applicability


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "public", "all"), default="core")
    parser.add_argument("--data-root", type=Path, default=default_root / "data")
    parser.add_argument("--output-dir", type=Path, default=default_root / "results" / "paper")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="legacy fallback for any seed axis not supplied explicitly",
    )
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--chart-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-download", action="store_true", help="allow official PyG/OGB downloads"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument(
        "--num-workers",
        "--workers",
        dest="num_workers",
        type=int,
        default=4,
        help=(
            "DataLoader processes; default 4 matches the repository paper runner and is "
            "reported with measured data-wait/throughput"
        ),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    arguments = build_parser().parse_args(argv)
    seed_axes: SeedAxes = resolve_seed_axes(
        arguments.seed,
        data_seed=arguments.data_seed,
        split_seed=arguments.split_seed,
        chart_seed=arguments.chart_seed,
        model_seed=arguments.model_seed,
    )
    if arguments.batch_size < 1 or arguments.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative")
    device = resolve_device(arguments.device)
    amp = device.type == "cuda" if arguments.amp is None else bool(arguments.amp)
    if device.type != "cuda" and amp:
        raise ValueError("--amp is a CUDA float16 path; use --no-amp on CPU")
    pin_memory = (
        device.type == "cuda" if arguments.pin_memory is None else bool(arguments.pin_memory)
    )
    if device.type != "cuda":
        pin_memory = False
    epochs = arguments.epochs if arguments.epochs is not None else 100
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    # Dataset preparation receives only the data axis.  Reset the global RNG to
    # the model axis immediately before optimization below.
    seed_everything(seed_axes.data)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = _prepare_output_dir(arguments.output_dir)
    implementation_before = _implementation_hashes()
    started = time.perf_counter()
    prepared: dict[str, Any] = {}
    core = None
    public = None
    if arguments.suite in {"core", "all"}:
        core, manifest_path, manifest = prepare_core_cache(arguments.data_root, seed=seed_axes.data)
        prepared["core"] = {
            "manifest": str(manifest_path),
            "cache_key": manifest["cache_key"],
            "artifact_sha256": manifest["artifact_sha256"],
            "content_sha256": manifest["content_sha256"],
            "data_seed": seed_axes.data,
        }
    if arguments.suite in {"public", "all"}:
        public, marker_path, manifest = prepare_public_data(
            arguments.data_root,
            allow_download=arguments.allow_download,
        )
        prepared["public"] = {
            "manifest": str(marker_path),
            "fixture": manifest["fixture"],
            "processed_sha256": manifest.get("processed_sha256"),
            "data_seed": "not_applicable",
            "split_seed": "not_applicable",
            "chart_seed": "not_applicable",
        }
    seed_applicability = _seed_axis_applicability(arguments.suite)
    data_observability = _dataset_observability(core, public)
    if arguments.prepare_only:
        summary = {
            "status": "prepared",
            "suite": arguments.suite,
            "seed_axes": seed_axes.to_manifest(),
            "seed_axis_applicability": seed_applicability,
            "prepared": prepared,
            "data_observability": data_observability,
            "implementation_integrity": _verify_implementation_unchanged(
                implementation_before
            ),
            "resource_snapshot": runtime_resource_snapshot(device),
            "execution_classification": {
                "implementation": "not_executed: prepare-only",
                "static_checks": "not_run_by_this_command",
                "unit_tests": "not_run_by_this_command",
                "smoke_test": False,
                "full_training": False,
                "full_evaluation": False,
                "actual_data_prepared": True,
            },
        }
        _write_json(output_dir / "prepare_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    seed_everything(seed_axes.model)
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    model_states: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resource_monitor = RuntimeResourceMonitor(device)
    resource_start = resource_monitor.start()
    execution_plan = _execution_plan(
        suite=arguments.suite,
        epochs=epochs,
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        amp=amp,
        pin_memory=pin_memory,
        device=device,
        data=data_observability,
        resource_start=resource_start,
        implementation_sha256=implementation_before,
    )
    print(json.dumps(execution_plan, indent=2, ensure_ascii=False, allow_nan=False))
    try:
        if core is not None:
            core_results, core_history, core_states = run_core(
                core,
                device=device,
                epochs=epochs,
                learning_rate=arguments.learning_rate,
                batch_size=arguments.batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=arguments.num_workers,
                seed=seed_axes.model,
            )
            results["core"] = core_results
            histories.extend(core_history)
            model_states["core"] = core_states
        if public is not None:
            public_results, public_history, public_states = run_public(
                public,
                device=device,
                epochs=epochs,
                learning_rate=arguments.learning_rate,
                batch_size=arguments.batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=arguments.num_workers,
                seed=seed_axes.model,
            )
            results["public"] = public_results
            histories.extend(public_history)
            model_states["public"] = public_states
        implementation_integrity = _verify_implementation_unchanged(implementation_before)
    except (Exception, KeyboardInterrupt) as error:
        resources, unavailable_reason, collection_notes = _finish_monitor_after_failure(
            resource_monitor,
            device,
            error,
        )
        failure = {
            "status": "failed",
            "suite": arguments.suite,
            "error_type": type(error).__name__,
            "error": str(error),
            "execution_plan": execution_plan,
            "completed_result_scopes": sorted(results),
            "completed_history_rows": len(histories),
            "resources_until_failure": resources,
            "resource_observability_unavailable_reason": unavailable_reason,
            "resource_observability_collection_notes": collection_notes,
            "source_sha256_before_training": implementation_before,
            "recovery_policy": (
                "Preserve model, graph, dataset, epochs, and sampling. Inspect measured allocator "
                "peaks/utilization, tensor lifetimes, synchronization, loader/graph construction, "
                "mixed precision, sparse kernels, activation checkpointing, caching, chunking, and "
                "multi-GPU/process distribution first. Only after a physical-batch candidate "
                "profile demonstrates insufficient VRAM should a new explicit batch size and new "
                "output directory be selected; this runner performs no automatic reduction."
            ),
        }
        try:
            _write_json(output_dir / "failure.json", failure)
        except BaseException as reporting_error:
            error.add_note(
                "failure.json could not be written without replacing this error: "
                f"{type(reporting_error).__name__}: {reporting_error}"
            )
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise RuntimeError(
                "CUDA out of memory in the conductance paper workload. No model, dataset, epoch, "
                "sampling, or batch setting was changed automatically. Inspect failure.json for "
                "the exact configuration and measured resource state, then profile memory and "
                "pipeline causes before choosing any explicit new physical batch candidate."
            ) from error
        raise
    peak_allocated, peak_reserved = _cuda_allocator_peaks(device)
    resources = resource_monitor.finish(
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "status": "passed",
        "scope": "independent_sparse_incidence_conductance_attention",
        "suite": arguments.suite,
        "seed_axes": seed_axes.to_manifest(),
        "seed_axis_applicability": seed_applicability,
        "prepared": prepared,
        "implementation_integrity": implementation_integrity,
        "execution_plan": execution_plan,
        "data_observability": data_observability,
        "configuration": {
            "epochs": epochs,
            "epochs_applied_to_core": epochs if core is not None else None,
            "epochs_applied_to_public": epochs if public is not None else None,
            "suite_specific_epoch_cap": None,
            "learning_rate": arguments.learning_rate,
            "physical_batch_size_graphs": arguments.batch_size,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size_graphs": arguments.batch_size,
            "num_workers": arguments.num_workers,
            "persistent_workers": arguments.num_workers > 0,
            "prefetch_factor": 2 if arguments.num_workers > 0 else None,
            "pin_memory": pin_memory,
            "non_blocking_transfer": pin_memory,
            "sampling_ratio": 1.0,
            "debug_subset_fast_mode": False,
        },
        "runtime": {
            **runtime_metadata(
                device, amp=amp, pin_memory=pin_memory, batch_size=arguments.batch_size
            ),
            "elapsed_seconds": elapsed,
            "resource_observability": resources,
        },
        "execution_classification": {
            "implementation": "completed",
            "static_checks": "not_run_by_this_command",
            "unit_tests": "not_run_by_this_command",
            "smoke_test": False,
            "full_training": True,
            "full_evaluation": True,
            "actual_data_used": True,
        },
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    metric_rows = _metric_rows(results)
    _write_csv(output_dir / "metrics.csv", metric_rows, ["path", "value"])
    _write_csv(
        output_dir / "history.csv",
        histories,
        [
            "suite",
            "baseline",
            "training_objective",
            "epoch",
            "train_loss",
            "validation_loss",
        ],
    )
    torch.save(model_states, output_dir / "models.pt")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
