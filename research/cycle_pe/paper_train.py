"""Training, evaluation, and runtime accounting for the paper CLI."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from chartgat.observability import observed
from research.cycle_pe.paper_model import (
    BatchOutput,
    PaperCycleModel,
    PreparedBatch,
    PreparedGraph,
    pack_prepared_graphs,
)
from research.cycle_pe.resource_monitor import (
    FailureSafeResourceMonitor,
    resource_failure_boundary,
)


@dataclass(frozen=True)
class TrainSettings:
    device: torch.device
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    workers: int
    amp_requested: bool
    pin_memory_requested: bool
    non_blocking_requested: bool
    prefetch_factor: int = 2

    @property
    def amp(self) -> bool:
        return self.amp_requested and self.device.type == "cuda"

    @property
    def pin_memory(self) -> bool:
        return self.pin_memory_requested and self.device.type == "cuda"

    @property
    def non_blocking(self) -> bool:
        return self.non_blocking_requested and self.pin_memory


@dataclass(frozen=True)
class TargetStats:
    mean: Tensor
    std: Tensor


class PhaseTimings:
    """Low-synchronization wall/CUDA phase accounting for one training run."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds: dict[str, float] = {}
        self._cuda_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def measure(self, name: str, *, cuda: bool) -> Iterator[None]:
        if cuda and self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_events.append((name, start, end))
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (
                time.perf_counter() - started
            )

    def resolve_cuda(self) -> None:
        if not self._cuda_events:
            return
        torch.cuda.synchronize(self.device)
        for name, start, end in self._cuda_events:
            self.seconds[name] = self.seconds.get(name, 0.0) + start.elapsed_time(end) / 1000.0
        self._cuda_events.clear()


def cuda_autocast(enabled: bool):
    """Use the public autocast API available across supported PyTorch releases."""

    return torch.autocast(device_type="cuda", enabled=enabled)


def make_grad_scaler(enabled: bool):
    """Construct a CUDA GradScaler on both PyTorch 2.2 and newer releases."""

    unified_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if unified_scaler is not None:
        try:
            return unified_scaler("cuda", enabled=enabled)
        except TypeError:
            return unified_scaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device specification: {requested!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this PyTorch build cannot access CUDA. On the "
                "Linux GPU workstation or server, verify `nvidia-smi`, then install the matching "
                "CUDA-enabled PyTorch wheel using `bash scripts/setup_gpu.sh`."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; detected "
                f"{torch.cuda.device_count()} device(s)"
            )
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise ValueError("paper CLI supports only cpu, cuda, cuda:N, or auto")
    return device


def runtime_environment(settings: TrainSettings) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(settings.device),
        "amp_requested": settings.amp_requested,
        "amp_effective": settings.amp,
        "pin_memory_requested": settings.pin_memory_requested,
        "pin_memory_effective": settings.pin_memory,
        "non_blocking_requested": settings.non_blocking_requested,
        "non_blocking_effective": settings.non_blocking,
        "batch_size": settings.batch_size,
        "workers": settings.workers,
        "logical_cpu_count": os.cpu_count(),
        "prefetch_factor": (
            settings.prefetch_factor
            if settings.workers > 0
            else observed(
                None,
                reason="prefetch_factor is inactive because DataLoader workers is zero",
            )
        ),
        "persistent_workers": settings.workers > 0,
        "worker_policy": (
            "explicit user-configured multiprocessing for CPU ragged packing"
            if settings.workers > 0
            else (
                "prepared graph tensors are already resident in RAM; workers=0 avoids "
                "multiprocess serialization and duplicated dataset storage. The setting "
                "remains configurable and must be compared on the target server when "
                "loader wait dominates."
            )
        ),
    }
    if settings.device.type == "cuda":
        index = settings.device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "mig_detected_from_device_name": "mig" in properties.name.lower(),
                "visible_cuda_device_count": torch.cuda.device_count(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_capability": [properties.major, properties.minor],
                "cuda_total_memory_bytes": properties.total_memory,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return result


def _target(graph: PreparedGraph, level: str) -> Tensor | None:
    return getattr(graph, f"{level}_targets")


def fit_target_stats(
    graphs: list[PreparedGraph],
    *,
    levels: tuple[str, ...] = ("edge", "node", "graph"),
) -> dict[str, TargetStats]:
    if not graphs:
        raise ValueError("training split cannot be empty")
    result: dict[str, TargetStats] = {}
    unknown = set(levels) - {"edge", "node", "graph"}
    if unknown:
        raise ValueError(f"unknown target levels: {sorted(unknown)}")
    for level in levels:
        values = [_target(graph, level) for graph in graphs]
        present = [value for value in values if value is not None]
        if not present:
            continue
        if len(present) != len(graphs):
            raise ValueError(f"{level} targets are missing on part of the training split")
        matrix = torch.cat([value.reshape(-1, value.shape[-1]) for value in present], dim=0)
        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0, unbiased=False).clamp_min(1e-6)
        result[level] = TargetStats(mean=mean, std=std)
    if not result:
        raise ValueError("at least one supervised target level is required")
    return result


def _target_stats_on_device(
    stats: dict[str, TargetStats], device: torch.device
) -> dict[str, TargetStats]:
    """Move the small immutable normalizers once, outside minibatch loops."""

    return {
        level: TargetStats(
            mean=level_stats.mean.to(device),
            std=level_stats.std.to(device),
        )
        for level, level_stats in stats.items()
    }


def _collate(
    graphs: list[PreparedGraph],
    *,
    variant: str,
    target_levels: tuple[str, ...],
) -> PreparedBatch:
    return pack_prepared_graphs(
        graphs, variant=variant, target_levels=target_levels
    )


def _loader(
    graphs: list[PreparedGraph],
    settings: TrainSettings,
    *,
    shuffle: bool,
    variant: str,
    target_levels: tuple[str, ...],
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(settings.seed)
    options: dict[str, Any] = {}
    if settings.workers > 0:
        options["prefetch_factor"] = settings.prefetch_factor
    return DataLoader(
        graphs,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.workers,
        pin_memory=settings.pin_memory,
        collate_fn=partial(
            _collate, variant=variant, target_levels=target_levels
        ),
        generator=generator,
        drop_last=False,
        persistent_workers=settings.workers > 0,
        **options,
    )


def _move_batch(batch: PreparedBatch, settings: TrainSettings) -> PreparedBatch:
    return batch.to(settings.device, non_blocking=settings.non_blocking)


def _output(outputs: BatchOutput, level: str) -> Tensor:
    value = getattr(outputs, level)
    if value is None:
        raise RuntimeError(f"model did not produce the configured {level} head")
    return value


def _batch_target(batch: PreparedBatch, level: str) -> Tensor:
    value = getattr(batch, f"{level}_targets")
    if value is None:
        raise RuntimeError(f"batch has missing {level} targets")
    return value


def normalized_multitask_loss(
    outputs: BatchOutput,
    graphs: PreparedBatch,
    stats: dict[str, TargetStats],
) -> Tensor:
    losses: list[Tensor] = []
    for level, level_stats in stats.items():
        predictions = _output(outputs, level)
        targets = _batch_target(graphs, level)
        prediction = predictions.reshape(-1, predictions.shape[-1])
        target = targets.reshape(-1, targets.shape[-1])
        mean = level_stats.mean
        std = level_stats.std
        if mean.device != target.device or std.device != target.device:
            raise RuntimeError(
                f"{level} target normalization statistics are not on the batch device"
            )
        losses.append(torch.mean((prediction - (target - mean) / std) ** 2))
    return torch.stack(losses).mean()


@torch.no_grad()
def validation_loss(
    model: PaperCycleModel,
    graphs: list[PreparedGraph],
    stats: dict[str, TargetStats],
    settings: TrainSettings,
    *,
    loader: DataLoader | None = None,
) -> float:
    model.eval()
    weighted = torch.zeros((), device=settings.device, dtype=torch.float64)
    count = 0
    selected_loader = loader or _loader(
        graphs,
        settings,
        shuffle=False,
        variant=model.pe_encoder.variant,
        target_levels=tuple(stats),
    )
    device_stats = _target_stats_on_device(stats, settings.device)
    for cpu_batch in selected_loader:
        batch = _move_batch(cpu_batch, settings)
        with cuda_autocast(settings.amp):
            outputs = model(batch)
            if not isinstance(outputs, BatchOutput):
                raise RuntimeError("packed training input did not produce a packed output")
            loss = normalized_multitask_loss(outputs, batch, device_stats)
        weighted += loss.detach().double() * batch.batch_size
        count += batch.batch_size
    return float(weighted.cpu()) / max(1, count)


def _peak_memory(settings: TrainSettings) -> int | None:
    if settings.device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(settings.device))


def _peak_reserved_memory(settings: TrainSettings) -> int | None:
    if settings.device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_reserved(settings.device))


def validate_optimizer_ownership(model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    owned = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    if set(trainable) != owned:
        missing = sorted(name for identifier, name in trainable.items() if identifier not in owned)
        extra = len(owned - set(trainable))
        raise RuntimeError(
            "optimizer parameter ownership does not match the trainable model; "
            f"missing={missing}, extra_count={extra}"
        )
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_first_step_gradients(model: nn.Module) -> dict[str, Any]:
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(
            "first actual backward pass did not connect every trainable parameter to a "
            f"finite gradient; missing={missing}, nonfinite={nonfinite}"
        )
    return {
        "validated_on_first_actual_backward": True,
        "missing_trainable_parameters": [],
        "nonfinite_trainable_parameters": [],
        "validated_trainable_parameter_tensors": sum(
            parameter.requires_grad for parameter in model.parameters()
        ),
    }


def require_finite_loss(loss: Tensor, label: str) -> None:
    predicate = torch.isfinite(loss)
    assertion = getattr(torch, "_assert_async", None)
    if loss.device.type == "cuda" and assertion is not None:
        assertion(predicate, label)
    elif not bool(predicate):
        raise FloatingPointError(label)


def _integer_distribution(values: list[int], *, unit: str) -> dict[str, Any]:
    if not values:
        reason = "no prepared graphs were available"
        return {
            key: observed(None, reason=reason, unit=unit)
            for key in ("minimum", "mean", "maximum", "total")
        }
    return {
        "minimum": observed(min(values), unit=unit),
        "mean": observed(sum(values) / len(values), unit=unit),
        "maximum": observed(max(values), unit=unit),
        "total": observed(sum(values), unit=unit),
    }


def _training_data_observability(
    train_graphs: list[PreparedGraph], validation_graphs: list[PreparedGraph]
) -> dict[str, Any]:
    all_graphs = [*train_graphs, *validation_graphs]
    first = all_graphs[0] if all_graphs else None
    reason = None if first is not None else "no prepared graph was available"
    return {
        "loaded_split_graph_counts": {
            "train": len(train_graphs),
            "validation": len(validation_graphs),
        },
        "actual_used_graph_count": len(all_graphs),
        "actual_used_fraction_of_loaded_graphs": observed(1.0, unit="fraction"),
        "sampling_ratio": observed(1.0, unit="fraction"),
        "nodes_per_graph": _integer_distribution(
            [graph.num_nodes for graph in all_graphs], unit="nodes"
        ),
        "canonical_undirected_edges_per_graph": _integer_distribution(
            [int(graph.edges.shape[0]) for graph in all_graphs], unit="edges"
        ),
        "cycle_rank_per_graph": _integer_distribution(
            [graph.cycle_rank for graph in all_graphs], unit="cycles"
        ),
        "input_tensor_shapes": {
            "node_features": observed(
                None if first is None else [None, int(first.node_features.shape[1])],
                reason=reason,
                unit="shape",
            ),
            "edge_features": observed(
                None if first is None else [None, int(first.edge_features.shape[1])],
                reason=reason,
                unit="shape",
            ),
            "ragged_axes": "nodes, edges, and cycle rank are packed without truncation",
        },
        "time_window": observed(
            None, reason="not applicable to static graphs", unit="not_applicable"
        ),
        "input_resolution": observed(
            None, reason="not applicable to graph-structured inputs", unit="not_applicable"
        ),
        "cache": (
            "all prepared static tensors are retained in RAM and reused across epochs; "
            "graph construction and PE extraction are outside the minibatch loop"
        ),
        "debug_subset_fast_mode": False,
    }


@resource_failure_boundary
def train_supervised(
    model: PaperCycleModel,
    train_graphs: list[PreparedGraph],
    validation_graphs: list[PreparedGraph],
    settings: TrainSettings,
    *,
    target_levels: tuple[str, ...] = ("edge", "node", "graph"),
) -> tuple[PaperCycleModel, dict[str, TargetStats], list[dict[str, float]], dict[str, Any]]:
    """Train normalized edge/node/graph heads and restore best validation state."""

    if (
        settings.epochs < 1
        or settings.batch_size < 1
        or settings.workers < 0
        or settings.prefetch_factor < 1
    ):
        raise ValueError(
            "epochs/batch_size/prefetch_factor must be positive and workers non-negative"
        )
    if not train_graphs or not validation_graphs:
        raise ValueError("training and validation splits must both be nonempty")
    seed_everything(settings.seed)
    model = model.to(settings.device)
    stats = fit_target_stats(train_graphs, levels=target_levels)
    device_stats = _target_stats_on_device(stats, settings.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    trainable_parameters = validate_optimizer_ownership(model, optimizer)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    scaler = make_grad_scaler(settings.amp)
    train_loader = _loader(
        train_graphs,
        settings,
        shuffle=True,
        variant=model.pe_encoder.variant,
        target_levels=target_levels,
    )
    validation_loader = _loader(
        validation_graphs,
        settings,
        shuffle=False,
        variant=model.pe_encoder.variant,
        target_levels=target_levels,
    )
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    optimizer_steps = 0
    planned_steps = settings.epochs * len(train_loader)
    processed_training_graphs = 0
    observed_batch_sizes: list[int] = []
    gradient_connectivity: dict[str, Any] | None = None
    training_cuda_synchronized_seconds = 0.0
    phase_timings = PhaseTimings(settings.device)
    data_observability = _training_data_observability(train_graphs, validation_graphs)
    batch_observability = {
        "batch_unit": "graphs",
        "requested_physical_batch_size": settings.batch_size,
        "maximum_physical_batch_size_for_this_split": min(
            settings.batch_size, len(train_graphs)
        ),
        "training_batches_per_epoch": len(train_loader),
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "effective_batch_size_formula": (
            f"{settings.batch_size} physical x 1 accumulation x 1 data-parallel worker"
        ),
        "dataloader_workers": settings.workers,
        "persistent_workers": settings.workers > 0,
        "prefetch_factor": (
            settings.prefetch_factor
            if settings.workers > 0
            else observed(
                None,
                reason="prefetch_factor is inactive because DataLoader workers is zero",
            )
        ),
        "pin_memory": settings.pin_memory,
        "non_blocking_transfer": settings.non_blocking,
        "collate": (
            "CPU ragged pack into one disjoint-union graph, one padded raw-basis tensor, "
            "and sum(E_g^2) packed projector values"
        ),
        "per_graph_gpu_forward_loop": False,
        "worker_policy": runtime_environment(settings)["worker_policy"],
        "batch_candidate_measurement": observed(
            None,
            reason=(
                "this scientific run uses the explicitly requested physical batch; compare "
                "candidate batches in a separate target-GPU profiling run to avoid changing "
                "the optimization recipe"
            ),
            unit="graphs_per_second",
        ),
    }
    if settings.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(settings.device)
        torch.cuda.synchronize(settings.device)
    resource_monitor = FailureSafeResourceMonitor(
        settings.device, workload="cycle_paper_supervised_training"
    )
    resources_at_start = resource_monitor.start()
    print(
        json.dumps({
            "kind": "cycle_paper_pre_training_observability",
            "model": {
                "name": type(model).__name__,
                "pe_variant": model.pe_encoder.variant,
                "layers": len(model.layers),
                "hidden_dimension": model.node_encoder[0].out_features,
                "pe_dimension": model.pe_encoder.pe_dim,
                "channels": model.node_encoder[0].out_features,
                "attention_heads": observed(
                    None, reason="PaperCycleModel has no attention mechanism"
                ),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "optimizer_owned_trainable_parameters": trainable_parameters,
            },
            "data": data_observability,
            "batch": batch_observability,
            "optimization": {
                "epochs": settings.epochs,
                "planned_maximum_optimizer_steps": planned_steps,
                "actual_optimizer_steps": observed(
                    None, reason="training has not started", unit="steps"
                ),
                "optimizer": "AdamW",
                "learning_rate": settings.learning_rate,
                "weight_decay": settings.weight_decay,
            },
            "precision": {
                "amp_requested": settings.amp_requested,
                "amp_effective": settings.amp,
                "autocast_dtype": "float16" if settings.amp else "float32",
            },
            "resources_at_start": resources_at_start,
        }, sort_keys=True),
        flush=True,
    )
    started = time.perf_counter()
    for epoch in range(settings.epochs):
        model.train()
        loss_sum = torch.zeros((), device=settings.device, dtype=torch.float64)
        seen = 0
        if settings.device.type == "cuda":
            torch.cuda.synchronize(settings.device)
        training_started = time.perf_counter()
        iterator = iter(train_loader)
        while True:
            with phase_timings.measure("dataloader_wait_wall_seconds", cuda=False):
                try:
                    cpu_batch = next(iterator)
                except StopIteration:
                    cpu_batch = None
            if cpu_batch is None:
                break
            with phase_timings.measure("packed_h2d_seconds", cuda=True):
                graphs = _move_batch(cpu_batch, settings)
            optimizer.zero_grad(set_to_none=True)
            with phase_timings.measure("forward_and_loss_seconds", cuda=True):
                with cuda_autocast(settings.amp):
                    outputs = model(graphs)
                    if not isinstance(outputs, BatchOutput):
                        raise RuntimeError(
                            "packed training input did not produce a packed output"
                        )
                    loss = normalized_multitask_loss(outputs, graphs, device_stats)
                require_finite_loss(loss, "nonfinite supervised Cycle paper loss")
            with phase_timings.measure("backward_seconds", cuda=True):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if optimizer_steps == 0:
                    gradient_connectivity = validate_first_step_gradients(model)
            with phase_timings.measure("optimizer_seconds", cuda=True):
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0, error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
            optimizer_steps += 1
            batch_size = graphs.batch_size
            observed_batch_sizes.append(batch_size)
            processed_training_graphs += batch_size
            loss_sum += loss.detach().double() * batch_size
            seen += batch_size
        phase_timings.resolve_cuda()
        training_cuda_synchronized_seconds += time.perf_counter() - training_started
        current_validation = validation_loss(
            model,
            validation_graphs,
            device_stats,
            settings,
            loader=validation_loader,
        )
        if not math.isfinite(current_validation):
            raise FloatingPointError("nonfinite supervised Cycle validation loss")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(loss_sum.cpu()) / max(1, seen),
                "validation_loss": current_validation,
            }
        )
        if current_validation < best_loss:
            best_loss = current_validation
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    if gradient_connectivity is None:
        raise RuntimeError("training performed no actual backward pass")
    model.load_state_dict(best_state)
    if settings.device.type == "cuda":
        torch.cuda.synchronize(settings.device)
    wall_seconds = time.perf_counter() - started
    peak_allocated = _peak_memory(settings)
    peak_reserved = _peak_reserved_memory(settings)
    resources = resource_monitor.finish(
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    throughput = {
        "scope": (
            "CUDA-synchronized training interval including DataLoader wait, CPU ragged "
            "packing, H2D, forward, backward, and optimizer; validation, checkpoint "
            "serialization, and offline graph/PE preparation excluded"
        ),
        "training_graphs": processed_training_graphs,
        "training_cuda_synchronized_seconds": training_cuda_synchronized_seconds,
        "training_graphs_per_second": (
            observed(
                processed_training_graphs / training_cuda_synchronized_seconds,
                unit="graphs_per_second",
            )
            if training_cuda_synchronized_seconds > 0
            else observed(
                None,
                reason="observed training duration was zero",
                unit="graphs_per_second",
            )
        ),
    }
    batch_observability.update(
        {
            "observed_smallest_physical_batch_size": min(observed_batch_sizes),
            "observed_largest_physical_batch_size": max(observed_batch_sizes),
            "effective_batch_size": max(observed_batch_sizes),
        }
    )
    runtime = runtime_environment(settings)
    runtime.update(
        {
            "wall_seconds": wall_seconds,
            "peak_gpu_memory_bytes": peak_allocated,
            "peak_gpu_reserved_memory_bytes": peak_reserved,
            "peak_gpu_memory_unavailable_reason": (
                None
                if settings.device.type == "cuda"
                else "training device is CPU, so CUDA allocator peaks are unavailable"
            ),
            "best_validation_loss": best_loss,
            "epochs_completed": settings.epochs,
            "optimizer_steps_completed": optimizer_steps,
            "optimizer_steps_planned": planned_steps,
            "optimizer_step_shortfall_reason": None,
            "model_observability": {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "optimizer_owned_trainable_parameters": trainable_parameters,
            },
            "gradient_connectivity": gradient_connectivity,
            "data_observability": data_observability,
            "batch_observability": batch_observability,
            "throughput": throughput,
            "phase_timing": {
                "scope": (
                    "CUDA events for packed H2D/forward/backward/optimizer and host wall "
                    "time for DataLoader iteration; CUDA stages are resolved once per epoch"
                ),
                **phase_timings.seconds,
            },
            "resources_at_start": resources_at_start,
            "resource_observability": resources,
        }
    )
    print(
        json.dumps({
            "kind": "cycle_paper_post_training_observability",
            "optimizer_steps_completed": optimizer_steps,
            "throughput": throughput,
            "resource_summary": resources["summary"],
        }, sort_keys=True),
        flush=True,
    )
    return model, stats, history, runtime


@torch.no_grad()
def evaluate_supervised(
    model: PaperCycleModel,
    graphs: list[PreparedGraph],
    stats: dict[str, TargetStats],
    settings: TrainSettings,
    target_names: dict[str, tuple[str, ...]],
    *,
    integer_targets: bool,
) -> dict[str, Any]:
    """Return per-target MAE/RMSE, normalized MAE, and graph-macro MAE."""

    model.eval()
    predictions: dict[str, list[np.ndarray]] = {level: [] for level in stats}
    targets: dict[str, list[np.ndarray]] = {level: [] for level in stats}
    device_stats = _target_stats_on_device(stats, settings.device)
    for cpu_batch in _loader(
        graphs,
        settings,
        shuffle=False,
        variant=model.pe_encoder.variant,
        target_levels=tuple(stats),
    ):
        batch = _move_batch(cpu_batch, settings)
        with cuda_autocast(settings.amp):
            outputs = model(batch)
        if not isinstance(outputs, BatchOutput):
            raise RuntimeError("packed evaluation input did not produce a packed output")
        for level, level_stats in device_stats.items():
            raw_prediction = getattr(outputs, level)
            raw_target = getattr(batch, f"{level}_targets")
            if raw_prediction is None or raw_target is None:
                raise RuntimeError(f"missing {level} output during evaluation")
            mean = level_stats.mean
            std = level_stats.std
            prediction = raw_prediction * std + mean
            counts = (
                batch.edge_counts
                if level == "edge"
                else batch.node_counts
                if level == "node"
                else (1,) * batch.batch_size
            )
            prediction_parts = torch.split(
                prediction.detach().float().cpu(), counts, dim=0
            )
            target_parts = torch.split(
                raw_target.detach().float().cpu(), counts, dim=0
            )
            predictions[level].extend(
                value.numpy() for value in prediction_parts
            )
            targets[level].extend(
                value.numpy() for value in target_parts
            )

    result: dict[str, Any] = {
        "graphs": len(graphs),
        "nodes": sum(graph.num_nodes for graph in graphs),
        "edges": sum(graph.edges.shape[0] for graph in graphs),
        "levels": {},
    }
    macro_normalized: list[float] = []
    for level, level_stats in stats.items():
        names = target_names[level]
        if len(names) != level_stats.mean.numel():
            raise ValueError(f"target-name count mismatch for {level}")
        flat_prediction = np.concatenate(
            [value.reshape(-1, value.shape[-1]) for value in predictions[level]], axis=0
        )
        flat_target = np.concatenate(
            [value.reshape(-1, value.shape[-1]) for value in targets[level]], axis=0
        )
        per_target: dict[str, Any] = {}
        std = level_stats.std.numpy()
        for index, name in enumerate(names):
            error = flat_prediction[:, index] - flat_target[:, index]
            graph_mae = float(
                np.mean(
                    [
                        np.mean(np.abs(pred[..., index] - target[..., index]))
                        for pred, target in zip(predictions[level], targets[level], strict=True)
                    ]
                )
            )
            metrics: dict[str, Any] = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "normalized_mae": float(np.mean(np.abs(error)) / std[index]),
                "graph_macro_mae": graph_mae,
                "values": int(error.size),
            }
            if integer_targets:
                metrics["rounded_exact_accuracy"] = float(
                    np.mean(np.rint(flat_prediction[:, index]) == flat_target[:, index])
                )
            per_target[name] = metrics
            macro_normalized.append(metrics["normalized_mae"])
        result["levels"][level] = {
            "targets": per_target,
            "macro_mae": float(np.mean([value["mae"] for value in per_target.values()])),
            "macro_normalized_mae": float(
                np.mean([value["normalized_mae"] for value in per_target.values()])
            ),
        }
    result["macro_normalized_mae"] = float(np.mean(macro_normalized))
    return result


def clone_cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})


__all__ = [
    "TargetStats",
    "TrainSettings",
    "clone_cpu_state",
    "cuda_autocast",
    "evaluate_supervised",
    "fit_target_stats",
    "make_grad_scaler",
    "normalized_multitask_loss",
    "require_finite_loss",
    "resolve_device",
    "runtime_environment",
    "seed_everything",
    "train_supervised",
    "validate_first_step_gradients",
    "validate_optimizer_ownership",
    "validation_loss",
]
