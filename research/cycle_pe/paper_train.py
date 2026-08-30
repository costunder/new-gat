"""Training, evaluation, and runtime accounting for the paper CLI."""

from __future__ import annotations

import copy
import math
import platform
import random
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from research.cycle_pe.paper_model import GraphOutput, PaperCycleModel, PreparedGraph


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


def _collate(graphs: list[PreparedGraph]) -> list[PreparedGraph]:
    return graphs


def _loader(
    graphs: list[PreparedGraph],
    settings: TrainSettings,
    *,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(settings.seed + seed_offset)
    return DataLoader(
        graphs,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.workers,
        pin_memory=settings.pin_memory,
        collate_fn=_collate,
        generator=generator,
        drop_last=False,
        persistent_workers=settings.workers > 0,
    )


def _move_batch(graphs: list[PreparedGraph], settings: TrainSettings) -> list[PreparedGraph]:
    return [graph.to(settings.device, non_blocking=settings.non_blocking) for graph in graphs]


def _output(outputs: list[GraphOutput], level: str) -> list[Tensor]:
    values = [getattr(output, level) for output in outputs]
    if any(value is None for value in values):
        raise RuntimeError(f"model did not produce the configured {level} head")
    return [value for value in values if value is not None]


def normalized_multitask_loss(
    outputs: list[GraphOutput],
    graphs: list[PreparedGraph],
    stats: dict[str, TargetStats],
) -> Tensor:
    losses: list[Tensor] = []
    for level, level_stats in stats.items():
        predictions = _output(outputs, level)
        targets = [_target(graph, level) for graph in graphs]
        if any(target is None for target in targets):
            raise RuntimeError(f"batch has missing {level} targets")
        prediction = torch.cat([value.reshape(-1, value.shape[-1]) for value in predictions], dim=0)
        target = torch.cat(
            [value.reshape(-1, value.shape[-1]) for value in targets if value is not None],
            dim=0,
        )
        mean = level_stats.mean.to(target.device)
        std = level_stats.std.to(target.device)
        losses.append(torch.mean((prediction - (target - mean) / std) ** 2))
    return torch.stack(losses).mean()


@torch.no_grad()
def validation_loss(
    model: PaperCycleModel,
    graphs: list[PreparedGraph],
    stats: dict[str, TargetStats],
    settings: TrainSettings,
) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    for cpu_graphs in _loader(graphs, settings, shuffle=False):
        batch = _move_batch(cpu_graphs, settings)
        with cuda_autocast(settings.amp):
            loss = normalized_multitask_loss(model(batch), batch, stats)
        weighted += float(loss.detach().cpu()) * len(batch)
        count += len(batch)
    return weighted / max(1, count)


def _peak_memory(settings: TrainSettings) -> int:
    if settings.device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(settings.device))


def train_supervised(
    model: PaperCycleModel,
    train_graphs: list[PreparedGraph],
    validation_graphs: list[PreparedGraph],
    settings: TrainSettings,
    *,
    target_levels: tuple[str, ...] = ("edge", "node", "graph"),
) -> tuple[PaperCycleModel, dict[str, TargetStats], list[dict[str, float]], dict[str, Any]]:
    """Train normalized edge/node/graph heads and restore best validation state."""

    if settings.epochs < 1 or settings.batch_size < 1 or settings.workers < 0:
        raise ValueError("epochs/batch_size must be positive and workers non-negative")
    seed_everything(settings.seed)
    model = model.to(settings.device)
    stats = fit_target_stats(train_graphs, levels=target_levels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scaler = make_grad_scaler(settings.amp)
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    if settings.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(settings.device)
        torch.cuda.synchronize(settings.device)
    started = time.perf_counter()
    for epoch in range(settings.epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for cpu_graphs in _loader(train_graphs, settings, shuffle=True, seed_offset=epoch):
            graphs = _move_batch(cpu_graphs, settings)
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(settings.amp):
                loss = normalized_multitask_loss(model(graphs), graphs, stats)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(graphs)
            seen += len(graphs)
        current_validation = validation_loss(model, validation_graphs, stats, settings)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": loss_sum / max(1, seen),
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
    model.load_state_dict(best_state)
    if settings.device.type == "cuda":
        torch.cuda.synchronize(settings.device)
    wall_seconds = time.perf_counter() - started
    runtime = runtime_environment(settings)
    runtime.update(
        {
            "wall_seconds": wall_seconds,
            "peak_gpu_memory_bytes": _peak_memory(settings),
            "best_validation_loss": best_loss,
            "epochs_completed": settings.epochs,
        }
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
    for cpu_graphs in _loader(graphs, settings, shuffle=False):
        batch = _move_batch(cpu_graphs, settings)
        with cuda_autocast(settings.amp):
            outputs = model(batch)
        for graph, output in zip(batch, outputs, strict=True):
            for level, level_stats in stats.items():
                raw_prediction = getattr(output, level)
                raw_target = _target(graph, level)
                if raw_prediction is None or raw_target is None:
                    raise RuntimeError(f"missing {level} output during evaluation")
                mean = level_stats.mean.to(raw_prediction.device)
                std = level_stats.std.to(raw_prediction.device)
                prediction = raw_prediction * std + mean
                predictions[level].append(prediction.detach().float().cpu().numpy())
                targets[level].append(raw_target.detach().float().cpu().numpy())

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
    "resolve_device",
    "runtime_environment",
    "seed_everything",
    "train_supervised",
    "validation_loss",
]
