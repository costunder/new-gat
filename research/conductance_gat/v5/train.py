"""Train one resumable graph-conditioned conductance V5 arm on official V1 caches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from chartgat.cache import atomic_publish, atomic_write_json
from chartgat.observability import RuntimeResourceMonitor, observed

from ..ablation.model import state_sha256
from ..ablation.train import _configure_fp32, _make_data, _require_cuda, training_loss
from ..benchmark import _seed, _versions
from ..benchmark_data import load_dataset, sha256_file, tensor_hash
from .diagnostics import (
    evaluate,
    layer_diagnostics,
    require_finite_tensor,
    require_first_step_conductance_gradient,
    selected_checkpoint_interventions,
)
from .model import GraphConditionedConductanceNodeClassifier
from .protocol import (
    BATCH_SIZE_BY_DATASET,
    BETA_PARAMETERIZATIONS,
    COMMON,
    COMPARISON_DESIGN,
    CONDITIONS,
    DATASETS,
    HARDWARE_PROFILES,
    METRIC_BY_DATASET,
    SAMPLING_MODES,
    SUITE,
    TRAINING_PHASES,
    beta_configuration,
)
from .sampling import TransductiveGraphSampler

FAILURE_RESOURCE_FILENAME = "failure-resource-observability.json"

ROOT = Path(__file__).resolve().parents[3]
RESUME_SEMANTICS = (
    "epoch-boundary deterministic resume from stored model/optimizer/RNG state; "
    "CUDA kernels are not claimed bitwise deterministic"
)
_PARAMETER_GROUPS = ("backbone", "spatial_w", "beta", "conductance")
_SHARED_IMPLEMENTATION_SOURCES = (
    "src/chartgat/observability.py",
    "research/conductance_gat/ablation/train.py",
    "research/conductance_gat/benchmark.py",
    "research/conductance_gat/benchmark_data.py",
    "src/chartgat/cache.py",
)


def architecture_configuration(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        "hidden_channels": args.hidden_channels,
        "layers": args.layers,
        "heads": args.heads,
        "ffn_multiplier": args.ffn_multiplier,
        "dropout": args.dropout,
        "activation_checkpoint": args.activation_checkpoint,
    }
    result.update(
        beta_configuration(
            args.beta_parameterization,
            args.beta_initial,
            args.beta_min,
            args.beta_max,
        )
    )
    return result


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    loader_workers = args.workers if args.dataset == "ppi" else 0
    return {
        **COMMON,
        **architecture_configuration(args),
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "workers": loader_workers,
        "device": args.device,
        "edge_chunk_size": args.edge_chunk_size,
        "sampling": args.sampling,
        "num_neighbors": list(args.num_neighbors),
        "sample_seed_batch_size": args.sample_seed_batch_size,
        "phase_fractions": list(args.phase_fractions),
        "hardware_profile": args.hardware_profile,
        "precision": args.precision,
        "amp": args.precision == "bf16",
        "tf32": args.tf32,
        "pin_memory": args.pin_memory,
        "loader_workers": loader_workers,
        "persistent_workers": loader_workers > 0,
        "prefetch_factor": 2 if loader_workers > 0 else None,
        "worker_configuration_source": getattr(
            args, "worker_configuration_source", "explicit_cli"
        ),
        "loader_worker_policy": (
            "PPI uses the configured worker pool with deterministic epoch-seeded shuffling; "
            "transductive full/sampled graphs use no DataLoader workers"
        ),
        "sample_prefetch": args.sample_prefetch,
    }


def resolve_hardware_arguments(args: argparse.Namespace) -> None:
    """Resolve explicit execution defaults before identity/config creation."""

    profile = HARDWARE_PROFILES[args.hardware_profile]
    if args.workers is None:
        args.workers = 4 if args.dataset == "ppi" else 0
        args.worker_configuration_source = "dataset_default"
    elif not hasattr(args, "worker_configuration_source"):
        args.worker_configuration_source = "explicit_cli"
    if args.batch_size is None:
        args.batch_size = (
            profile["ppi_batch_size"]
            if args.dataset == "ppi"
            else BATCH_SIZE_BY_DATASET[args.dataset]
        )
    if args.edge_chunk_size is None:
        args.edge_chunk_size = profile["edge_chunk_size"]
    if args.sample_seed_batch_size is None:
        args.sample_seed_batch_size = profile["sample_seed_batch_size"]
    if args.activation_checkpoint is None:
        args.activation_checkpoint = profile["activation_checkpoint"]
    args.precision = profile["precision"]
    args.tf32 = profile["tf32"]
    args.pin_memory = profile["pin_memory"]
    args.sample_prefetch = profile["sample_prefetch"]


def validate_hardware_runtime(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Fail closed when an opt-in high-memory profile is not actually available."""

    profile = HARDWARE_PROFILES[args.hardware_profile]
    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(properties.total_memory)
    total_gib = total_bytes / 2**30
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_bytes = int(free_bytes)
    free_gib = free_bytes / 2**30
    capability = (int(properties.major), int(properties.minor))
    if total_gib < profile["minimum_total_memory_gib"]:
        raise RuntimeError(
            f"hardware profile {args.hardware_profile} requires at least "
            f"{profile['minimum_total_memory_gib']:.0f} GiB on the visible GPU; "
            f"found {total_gib:.2f} GiB"
        )
    if capability[0] < profile["minimum_compute_capability_major"]:
        raise RuntimeError(
            f"hardware profile {args.hardware_profile} requires compute capability "
            f">={profile['minimum_compute_capability_major']}.0; found "
            f"{capability[0]}.{capability[1]}"
        )
    if free_gib < profile["minimum_free_memory_gib"]:
        raise RuntimeError(
            f"hardware profile {args.hardware_profile} requires at least "
            f"{profile['minimum_free_memory_gib']:.0f} GiB free at child start; "
            f"found {free_gib:.2f} GiB"
        )
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required by this hardware profile; FP32 fallback is forbidden")
    return {
        "profile": args.hardware_profile,
        "device_name": properties.name,
        "total_memory_bytes": total_bytes,
        "total_memory_gib": total_gib,
        "free_memory_bytes_at_start": free_bytes,
        "free_memory_gib_at_start": free_gib,
        "compute_capability": list(capability),
        "precision": args.precision,
        "tf32": args.tf32,
        "dense_autocast": args.precision == "bf16",
        "conductance_geometry_dtype": "float32",
        "activation_checkpoint": args.activation_checkpoint,
        "edge_chunk_size": args.edge_chunk_size,
        "sample_seed_batch_size": args.sample_seed_batch_size,
        "graph_batch_size": args.batch_size,
        "sample_prefetch": args.sample_prefetch,
        "pin_memory": args.pin_memory,
        "loader_workers": args.workers if args.dataset == "ppi" else 0,
        "persistent_workers": args.dataset == "ppi" and args.workers > 0,
        "prefetch_factor": 2 if args.dataset == "ppi" and args.workers > 0 else None,
        "loader_worker_policy": (
            "PPI graph batches use deterministic epoch-seeded DataLoader shuffling; "
            "transductive full/sampled graphs do not construct a DataLoader"
        ),
    }


def configure_compute(args: argparse.Namespace) -> None:
    _configure_fp32()
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def autocast_context(args: argparse.Namespace):
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"
    )


def require_finite_gradient_norm_async(value: torch.Tensor) -> None:
    """Enqueue a stream-ordered finite assertion without a CUDA-to-host scalar read."""

    predicate = torch.isfinite(value)
    assertion = getattr(torch, "_assert_async", None)
    if assertion is None:
        require_finite_tensor(value, "gradient norm")
        return
    assertion(predicate, "nonfinite gradient norm")


def phase_schedule(epochs: int, fractions: list[float]) -> list[dict[str, Any]]:
    if epochs < 4 or len(fractions) != 4 or any(value <= 0 for value in fractions):
        raise ValueError("epochs must be >=4 and all four phase fractions must be positive")
    total = sum(fractions)
    if not math.isfinite(total):
        raise ValueError("phase fractions must be finite")
    raw = [epochs * value / total for value in fractions]
    lengths = [max(1, int(math.floor(value))) for value in raw]
    while sum(lengths) < epochs:
        index = max(range(4), key=lambda i: raw[i] - lengths[i])
        lengths[index] += 1
    while sum(lengths) > epochs:
        candidates = [i for i, value in enumerate(lengths) if value > 1]
        lengths[max(candidates, key=lambda i: lengths[i] - raw[i])] -= 1
    result, start = [], 1
    for name, length in zip(TRAINING_PHASES, lengths, strict=True):
        result.append(
            {"name": name, "start_epoch": start, "end_epoch": start + length - 1, "length": length}
        )
        start += length
    return result


def phase_at(schedule: list[dict[str, Any]], epoch: int) -> tuple[str, int]:
    for item in schedule:
        if item["start_epoch"] <= epoch <= item["end_epoch"]:
            return item["name"], epoch - item["start_epoch"]
    raise ValueError("epoch outside phase schedule")


def selection_eligibility(condition: str, phase: str) -> dict[str, bool]:
    """Return the condition-aware validation-selection roles for one epoch.

    The fixed arm has no latent mechanism to wait for, so every epoch is a
    scientifically valid primary checkpoint candidate.  The dynamic arm's
    warmup explicitly overrides C with ones; it may be the best pure predictor,
    but it is not evidence for learned C and therefore is auxiliary only.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"unknown V5 condition: {condition}")
    if phase not in TRAINING_PHASES:
        raise ValueError(f"unknown V5 phase: {phase}")
    dynamic = condition == "shared_dynamic_c"
    return {
        "global_prediction": True,
        "primary": not dynamic or phase != "spatial_warmup",
        "joint_early_stopping": dynamic and phase == "joint",
    }


def should_stop_early(
    condition: str,
    phase: str,
    epoch: int,
    *,
    primary_best_epoch: int,
    joint_best_epoch: int,
    patience: int,
) -> bool:
    """Apply patience to the checkpoint role valid for each condition."""

    eligibility = selection_eligibility(condition, phase)
    if condition == "fixed_c":
        return primary_best_epoch > 0 and epoch - primary_best_epoch >= patience
    return (
        eligibility["joint_early_stopping"]
        and joint_best_epoch > 0
        and epoch - joint_best_epoch >= patience
    )


def parameter_group(name: str) -> str:
    if ".operator.estimator." in name:
        return "conductance"
    if ".operator.beta_estimator." in name:
        return "beta"
    if ".operator.value_weight" in name or ".operator.output_projection." in name:
        return "spatial_w"
    return "backbone"


def make_optimizer(model) -> torch.optim.AdamW:
    grouped = {key: [] for key in ("backbone", "spatial_w", "beta", "conductance")}
    names = {key: [] for key in grouped}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = parameter_group(name)
            grouped[group].append(parameter)
            names[group].append(name)
    options = {
        "backbone": (COMMON["lr"], COMMON["weight_decay"]),
        "spatial_w": (COMMON["lr"], COMMON["weight_decay"]),
        "beta": (COMMON["lr"] * COMMON["beta_lr_multiplier"], COMMON["scalar_weight_decay"]),
        "conductance": (
            COMMON["lr"] * COMMON["conductance_lr_multiplier"],
            COMMON["conductance_weight_decay"],
        ),
    }
    groups = [
        {
            "name": name,
            "params": values,
            "parameter_names": names[name],
            "lr": options[name][0],
            "weight_decay": options[name][1],
        }
        for name, values in grouped.items()
        if values
    ]
    return torch.optim.AdamW(groups, lr=COMMON["lr"])


def validate_optimizer_parameter_ownership(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    owned: dict[int, str] = {}
    duplicates: list[str] = []
    for group in optimizer.param_groups:
        group_name = str(group["name"])
        for parameter in group["params"]:
            identifier = id(parameter)
            if identifier in owned:
                duplicates.append(expected.get(identifier, f"unknown:{identifier}"))
            owned[identifier] = group_name
    missing = [name for identifier, name in expected.items() if identifier not in owned]
    unexpected = [identifier for identifier in owned if identifier not in expected]
    wrong_group = [
        name
        for identifier, name in expected.items()
        if identifier in owned and owned[identifier] != parameter_group(name)
    ]
    if missing or unexpected or duplicates or wrong_group:
        raise RuntimeError(
            "V5 optimizer ownership mismatch; "
            f"missing={missing}, unexpected_count={len(unexpected)}, "
            f"duplicates={duplicates}, wrong_group={wrong_group}"
        )


def validate_active_gradient_connectivity(
    model: torch.nn.Module, active_groups: list[str]
) -> None:
    active = set(active_groups)
    expected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter_group(name) in active
    ]
    requires_mismatch = [
        name for name, parameter in expected if not parameter.requires_grad
    ]
    missing = [
        name
        for name, parameter in expected
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in expected
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if requires_mismatch or missing or nonfinite:
        raise RuntimeError(
            "V5 active parameter groups are not connected to finite task gradients; "
            f"requires_grad_mismatch={requires_mismatch}, missing={missing}, "
            f"nonfinite={nonfinite}"
        )


def optimizer_metadata(optimizer) -> list[dict[str, Any]]:
    return [
        {
            "name": group["name"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameter_names": list(group["parameter_names"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]


def merge_efficiency(
    previous_elapsed_seconds: float,
    previous_peak_allocated_bytes: int,
    previous_peak_reserved_bytes: int,
    current_elapsed_seconds: float,
    current_peak_allocated_bytes: int,
    current_peak_reserved_bytes: int,
) -> dict[str, float | int]:
    """Accumulate wall time and retain the maximum GPU peaks across resumes."""

    values = (
        previous_elapsed_seconds,
        previous_peak_allocated_bytes,
        previous_peak_reserved_bytes,
        current_elapsed_seconds,
        current_peak_allocated_bytes,
        current_peak_reserved_bytes,
    )
    if any(value < 0 for value in values):
        raise ValueError("efficiency counters must be nonnegative")
    return {
        "elapsed_seconds": float(previous_elapsed_seconds + current_elapsed_seconds),
        "peak_cuda_allocated_bytes": max(
            int(previous_peak_allocated_bytes), int(current_peak_allocated_bytes)
        ),
        "peak_cuda_reserved_bytes": max(
            int(previous_peak_reserved_bytes), int(current_peak_reserved_bytes)
        ),
    }


def training_throughput(
    history: list[dict[str, Any]], elapsed_seconds: float
) -> dict[str, Any]:
    """Report retained training work over the existing cumulative wall-time interval.

    History and elapsed time both include restored checkpoint state. This is not
    a training-kernel benchmark: validation/checkpoint IO and final interventions
    within the timed interval remain in the denominator.
    """

    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("throughput elapsed_seconds must be finite and nonnegative")
    if not isinstance(history, list) or not history:
        raise ValueError("throughput requires nonempty completed epoch history")
    counts = {"train_label_count": 0, "train_batches": 0}
    for epoch, row in enumerate(history, start=1):
        for field in counts:
            value = row.get(field) if isinstance(row, dict) else None
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"throughput history epoch {epoch} {field} must be a positive integer"
                )
            counts[field] += value
    elapsed_seconds = float(elapsed_seconds)

    def rate(count: int, unit: str) -> dict[str, Any]:
        return (
            observed(count / elapsed_seconds, unit=unit)
            if elapsed_seconds > 0
            else observed(
                None, reason="observed cumulative wall duration was zero", unit=unit
            )
        )

    return {
        "scope": (
            "Cumulative retained epoch-loop wall time plus selected-checkpoint intervention "
            "evaluation; includes training, validation and checkpoint IO within timed intervals"
        ),
        "timer_boundary": (
            "CUDA-synchronized immediately before the epoch loop and after selected-checkpoint "
            "interventions; model/data setup and final metrics serialization are excluded"
        ),
        "resume_accounting": (
            "Complete restored history counts divided by restored elapsed time plus this "
            "invocation's measured duration; interrupted work after the last checkpoint and "
            "checkpoint serialization after its saved timer snapshot are not reconstructible"
        ),
        "completed_epochs": len(history),
        "supervised_training_labels": counts["train_label_count"],
        "training_batches": counts["train_batches"],
        "elapsed_seconds": elapsed_seconds,
        "supervised_labels_per_second": rate(counts["train_label_count"], "labels_per_second"),
        "training_batches_per_second": rate(counts["train_batches"], "batches_per_second"),
    }


def _integer_distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty observation")
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "total": sum(values),
    }


def _payload_graph_observability(payload: dict[str, Any]) -> dict[str, Any]:
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("verified payload must contain at least one graph")
    if any(not isinstance(graph, dict) for graph in graphs):
        raise ValueError("verified payload graph rows must be mappings")
    features = [graph.get("x") for graph in graphs]
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in features):
        raise ValueError("verified payload graph features must be rank-two tensors")
    node_counts = [int(value.shape[0]) for value in features]
    feature_widths = sorted({int(value.shape[1]) for value in features})
    edge_counts: list[int] = []
    edge_fields: list[str] = []
    for graph in graphs:
        located = None
        for name in ("incidence_edge_index", "incidence", "edge_index"):
            candidate = graph.get(name)
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                located = candidate
                edge_fields.append(name)
                break
        if located is not None:
            edge_counts.append(int(located.shape[1]))
    if len(edge_counts) == len(graphs):
        edge_observation: Any = _integer_distribution(edge_counts)
        edge_reason = None
    else:
        edge_observation = observed(
            None,
            reason=(
                f"recognized an incidence/edge tensor for {len(edge_counts)} of "
                f"{len(graphs)} verified graphs"
            ),
        )
        edge_reason = edge_observation["reason"]
    target_shapes = sorted(
        {
            tuple(int(dimension) for dimension in graph["y"].shape)
            for graph in graphs
            if isinstance(graph.get("y"), torch.Tensor)
        }
    )
    return {
        "official_graph_count": len(graphs),
        "nodes_per_graph": _integer_distribution(node_counts),
        "stored_edge_columns_per_graph": edge_observation,
        "stored_edge_tensor_fields": sorted(set(edge_fields)),
        "stored_edge_count_limitation": edge_reason,
        "input_tensor_shapes": {
            "node_features": ["variable_batch_nodes", *feature_widths],
            "feature_widths": feature_widths,
            "target_shapes_observed": [list(shape) for shape in target_shapes],
        },
    }


def _v5_data_observability(
    payload: dict[str, Any],
    data: Any,
    indices: dict[str, torch.Tensor] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    graph_observation = _payload_graph_observability(payload)
    if indices is not None:
        split_counts = {name: int(value.numel()) for name, value in indices.items()}
        total_units = int(data.x.shape[0])
        unit = "nodes"
    elif isinstance(data, dict):
        split_counts = {
            name: len(loader.dataset)
            for name, loader in data.items()
            if hasattr(loader, "dataset")
        }
        total_units = graph_observation["official_graph_count"]
        unit = "graphs"
    else:
        split_counts = {}
        total_units = graph_observation["official_graph_count"]
        unit = "graphs"
    training_units = split_counts.get("train", 0)
    validation_units = sum(
        split_counts.get(name, 0) for name in ("validation", "valid", "val")
    )
    actually_used = training_units + validation_units
    if total_units > 0:
        used_fraction = observed(actually_used / total_units, unit="fraction")
    else:
        used_fraction = observed(
            None, reason="verified full dataset count is zero", unit="fraction"
        )
    sampling_ratio = (
        observed(1.0, unit="fraction")
        if args.sampling == "full"
        else observed(
            None,
            reason=(
                "neighbor/cluster sampling has no single static edge ratio; sampler metadata "
                "and observed batch counts define the exact execution"
            ),
            unit="fraction",
        )
    )
    return {
        **graph_observation,
        "full_dataset_unit": unit,
        "full_dataset_count": total_units,
        "actual_split_counts": split_counts,
        "optimization_count": training_units,
        "validation_selection_count": validation_units,
        "test_evaluation_count": 0,
        "actual_used_count": actually_used,
        "actual_used_fraction_of_full_dataset": used_fraction,
        "sampling_ratio": sampling_ratio,
        "sampling_mode": args.sampling,
        "subset_or_fast_mode": False,
        "time_window": observed(
            None, reason="not applicable to static graph benchmarks", unit="steps"
        ),
        "input_resolution": observed(
            None, reason="not applicable to graph feature tensors"
        ),
    }


def _v5_batch_observability(
    data: Any,
    indices: dict[str, torch.Tensor] | None,
    sampler: TransductiveGraphSampler | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if indices is not None and sampler is None:
        physical_batch_size, batch_unit, batches_per_epoch = 1, "full_graph", 1
    elif indices is not None:
        physical_batch_size, batch_unit, batches_per_epoch = (
            int(args.batch_size),
            "sampler_seed_nodes_or_partitions",
            None,
        )
    else:
        physical_batch_size, batch_unit = int(args.batch_size), "graphs"
        batches_per_epoch = len(data["train"])
    batches_observation = (
        observed(batches_per_epoch, unit="batches")
        if batches_per_epoch is not None
        else observed(
            None,
            reason=(
                "the sampler can produce topology-dependent batches; actual per-epoch "
                "counts are recorded in history"
            ),
            unit="batches",
        )
    )
    planned_batches = (
        observed(args.epochs * batches_per_epoch, unit="batches")
        if batches_per_epoch is not None
        else observed(
            None,
            reason="training batches per epoch are topology-dependent",
            unit="batches",
        )
    )
    return {
        "batch_unit": batch_unit,
        "configured_physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "effective_batch_size": physical_batch_size,
        "effective_batch_size_formula": (
            f"{physical_batch_size} physical x 1 accumulation x 1 data-parallel worker"
        ),
        "training_batches_per_epoch": batches_observation,
        "planned_maximum_training_batches": planned_batches,
        "dataloader_workers": args.workers if indices is None else 0,
        "pin_memory": args.pin_memory if indices is None else False,
        "persistent_workers": indices is None and args.workers > 0,
        "prefetch_factor": 2 if indices is None and args.workers > 0 else None,
        "sample_prefetch": args.sample_prefetch,
        "cache": "verified immutable official graph cache; static topology reused",
        "sampler": sampler.metadata() if sampler is not None else {"mode": "full"},
    }


def configure_phase(model, phase: str, phase_epoch: int) -> dict[str, Any]:
    dynamic = model.conductance_mode == "dynamic"
    if phase == "spatial_warmup":
        active, override, training_mode = {"backbone", "spatial_w", "beta"}, "ones", True
        coordinate = "spatial"
    elif phase == "conductance_calibration":
        # The fixed-C arm remains a fully optimized, strong baseline rather than
        # wasting dynamic-C's calibration allocation as no-op epochs.
        active, override, training_mode = (
            ({"conductance"}, None, False)
            if dynamic
            else ({"backbone", "spatial_w", "beta"}, "ones", True)
        )
        coordinate = "conductance" if dynamic else "fixed_spatial_control"
    elif phase == "alternating":
        conductance_turn = dynamic and phase_epoch % 2 == 0
        active = {"conductance"} if conductance_turn else {"backbone", "spatial_w", "beta"}
        override, training_mode = None, not conductance_turn
        coordinate = "conductance" if conductance_turn else "spatial"
    elif phase == "joint":
        active = {"backbone", "spatial_w", "beta"} | ({"conductance"} if dynamic else set())
        override, training_mode, coordinate = None, True, "joint"
    else:
        raise ValueError(f"unknown phase: {phase}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(parameter_group(name) in active)
    for operator in model.operators:
        operator.estimator.override = override
    model.train(training_mode)
    return {
        "phase": phase,
        "phase_epoch": phase_epoch,
        "coordinate": coordinate,
        "active_parameter_groups": sorted(active),
        "conductance_override": override,
        "dropout_on": training_mode,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=COMMON["hidden_channels"])
    parser.add_argument("--layers", type=int, default=COMMON["layers"])
    parser.add_argument("--heads", type=int, default=COMMON["heads"])
    parser.add_argument("--ffn-multiplier", type=int, default=COMMON["ffn_multiplier"])
    parser.add_argument("--dropout", type=float, default=COMMON["dropout"])
    parser.add_argument(
        "--beta-parameterization",
        choices=BETA_PARAMETERIZATIONS,
        default=COMMON["beta_parameterization"],
    )
    parser.add_argument("--beta-initial", type=float, default=COMMON["beta_initial"])
    parser.add_argument("--beta-min", type=float)
    parser.add_argument("--beta-max", type=float)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Default: 4 for PPI graph minibatches, 0 for transductive full graphs",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--edge-chunk-size", type=int)
    parser.add_argument("--sampling", choices=SAMPLING_MODES, default="full")
    parser.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10])
    parser.add_argument("--sample-seed-batch-size", type=int)
    parser.add_argument("--hardware-profile", choices=tuple(HARDWARE_PROFILES), default="portable")
    parser.add_argument(
        "--phase-fractions",
        type=float,
        nargs=4,
        default=[0.1, 0.1, 0.4, 0.4],
        metavar=("WARMUP", "C_CAL", "ALTERNATE", "JOINT"),
    )
    parser.add_argument(
        "--activation-checkpoint", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    resolve_hardware_arguments(args)
    integers = (
        args.epochs,
        args.patience,
        args.hidden_channels,
        args.layers,
        args.heads,
        args.ffn_multiplier,
        args.edge_chunk_size,
        args.sample_seed_batch_size,
    )
    if min(integers) < 1 or args.model_seed < 0 or any(value < 1 for value in args.num_neighbors):
        raise ValueError("integer architecture/training/sampling values must be positive")
    if args.epochs < 4 or args.hidden_channels % args.heads:
        raise ValueError("V5 requires >=4 epochs and hidden_channels divisible by heads")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    beta_configuration(
        args.beta_parameterization,
        args.beta_initial,
        args.beta_min,
        args.beta_max,
    )
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.dataset != "ppi" and args.workers != 0:
        raise ValueError("transductive V5 datasets use no DataLoader and require workers=0")
    if args.dataset != "ppi" and args.batch_size != BATCH_SIZE_BY_DATASET[args.dataset]:
        raise ValueError("transductive full/sampled graph batch-size must be 1")
    if args.dataset == "ppi" and args.hardware_profile == "portable" and args.batch_size != 2:
        raise ValueError("portable PPI retains the V1 graph batch-size of 2")
    if args.dataset == "ppi" and args.sampling != "full":
        raise ValueError(
            "PPI already supplies inductive graph minibatches; sampling is transductive-only"
        )
    if args.sampling != "full" and args.sample_seed_batch_size < 32:
        raise ValueError("sample-seed-batch-size below 32 is forbidden as accidentally tiny")
    phase_schedule(args.epochs, list(args.phase_fractions))


def validate_cached_graphs_once(payload: dict[str, Any]) -> None:
    """Validate immutable graph structure on CPU, outside the CUDA hot path."""

    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("verified payload must contain nonempty graphs")
    for graph in graphs:
        x, incidence = graph.get("x"), graph.get("incidence_edge_index")
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.ndim != 2:
            raise ValueError("cached graph.x must be a two-dimensional CPU tensor")
        if (
            not isinstance(incidence, torch.Tensor)
            or incidence.device.type != "cpu"
            or incidence.dtype != torch.long
            or incidence.ndim != 2
            or incidence.shape[0] != 2
        ):
            raise ValueError("cached incidence must be a 2 x E CPU int64 tensor")
        if incidence.numel() and (int(incidence.min()) < 0 or int(incidence.max()) >= x.shape[0]):
            raise ValueError("cached incidence endpoint lies outside graph.x")


def _prepare_data(payload, args, device):
    validate_cached_graphs_once(payload)
    if args.sampling == "full" or args.dataset == "ppi":
        data, indices = _make_data(payload, args, device)
        return data, indices, None
    from torch_geometric.data import Data

    graph = Data(**payload["graphs"][0])
    indices = {
        key: payload["splits"][key].nonzero(as_tuple=False).flatten().long()
        for key in ("train", "validation")
    }
    sampler = TransductiveGraphSampler(
        graph,
        indices["train"],
        mode=args.sampling,
        seed_batch_size=args.sample_seed_batch_size,
        fanouts=args.num_neighbors,
        model_seed=args.model_seed,
    )
    return graph, indices, sampler


def _prefetched_samples(iterator, *, pin_memory: bool):
    """Prepare one deterministic CPU sample ahead while CUDA trains the current one."""

    source = iter(iterator)

    def take_one():
        try:
            graph = next(source)
        except StopIteration:
            return None
        return graph.pin_memory() if pin_memory else graph

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(take_one)
        while True:
            graph = pending.result()
            if graph is None:
                return
            pending = executor.submit(take_one)
            yield graph


def _training_batches(data, indices, sampler, epoch, device, model_seed, args):
    if sampler is not None:
        samples = sampler.iter_epoch(epoch)
        if args.sample_prefetch:
            samples = _prefetched_samples(samples, pin_memory=args.pin_memory)
        for graph in samples:
            graph = graph.to(device, non_blocking=args.pin_memory)
            yield graph, graph.train_mask.nonzero(as_tuple=False).flatten()
    elif indices is not None:
        yield data, indices["train"]
    else:
        # Re-seeding by epoch makes PPI minibatch order identical after resume.
        if getattr(data["train"], "generator", None) is not None:
            data["train"].generator.manual_seed(model_seed + 1_000_003 * epoch)
        for graph in data["train"]:
            graph._v5_num_graphs = int(graph.num_graphs)
            yield graph.to(device, non_blocking=True), None


def _validation_source(data, sampler):
    return data if sampler is None else sampler.graph


def _save(path: Path, payload: dict[str, Any]) -> None:
    atomic_publish(path, lambda target: torch.save(payload, target))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def shared_initial_state_sha256(model: torch.nn.Module) -> str:
    """Hash only parameters and buffers that exist in both V5 comparison arms."""

    digest = hashlib.sha256()
    included = 0
    for name, tensor in model.state_dict().items():
        if ".operator.estimator." in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_hash(tensor).encode("ascii"))
        included += 1
    if included == 0:
        raise RuntimeError("V5 model has no shared state to fingerprint")
    return digest.hexdigest()


def implementation_source_hashes() -> dict[str, str]:
    """Fingerprint every source that implements the child training path."""

    paths = [ROOT / value for value in _SHARED_IMPLEMENTATION_SOURCES]
    paths.extend((ROOT / "research/conductance_gat/v5").glob("*.py"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V5 implementation source is missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(set(paths))}


def build_resume_identity(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    schedule: list[dict[str, Any]],
    *,
    initial_state_sha256: str,
    source_sha256: dict[str, str] | None = None,
    runtime_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind a last checkpoint to data, recipe, implementation, and runtime."""

    cache_sha256 = protocol.get("data_sha256")
    if not isinstance(cache_sha256, str) or len(cache_sha256) != 64:
        raise ValueError("official dataset protocol has no valid data_sha256")
    if len(initial_state_sha256) != 64:
        raise ValueError("initial model state fingerprint is invalid")
    return {
        "schema_version": 1,
        "research_suite": SUITE,
        "dataset": args.dataset,
        "condition": args.condition,
        "configuration": configuration(args),
        "schedule": schedule,
        "dataset_protocol": protocol,
        "dataset_protocol_sha256": _canonical_sha256(protocol),
        "cache_sha256": cache_sha256,
        "initial_state_sha256": initial_state_sha256,
        "source_sha256": source_sha256 or implementation_source_hashes(),
        "runtime_versions": runtime_versions or _versions(),
        "resume_semantics": RESUME_SEMANTICS,
    }


def validate_resume_identity(actual: Any, expected: dict[str, Any], stored_sha256: Any) -> None:
    """Reject stale/tampered last checkpoints before loading mutable state."""

    if not isinstance(actual, dict):
        raise ValueError("last.pt has no valid V5 resume identity")
    actual_sha256 = _canonical_sha256(actual)
    if stored_sha256 != actual_sha256:
        raise ValueError("last.pt resume identity hash mismatch")
    if actual != expected:
        keys = sorted(
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        )
        detail = ", ".join(keys) if keys else "unknown"
        raise ValueError(f"last.pt resume identity mismatch: {detail}")


def recover_best_checkpoint(checkpoint: Path, previous: Path, expected_hash: Any) -> str:
    """Recover the best->last publication crash window from a two-slot journal."""

    if expected_hash is None:
        # A crash after publishing the first joint best but before last.pt may
        # leave an unbound primary. The same epoch is replayed and republishes it.
        return "not_yet_bound"
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(value not in "0123456789abcdef" for value in expected_hash)
    ):
        raise ValueError("last.pt best-checkpoint hash is invalid")
    if checkpoint.is_file() and sha256_file(checkpoint) == expected_hash:
        return "primary"
    if not previous.is_file() or sha256_file(previous) != expected_hash:
        raise ValueError("last.pt best checkpoint has no valid recovery slot")
    previous_bytes = previous.read_bytes()
    atomic_publish(checkpoint, lambda target, data=previous_bytes: target.write_bytes(data))
    if sha256_file(checkpoint) != expected_hash:
        raise ValueError("best-checkpoint recovery did not preserve its exact hash")
    return "previous"


def publish_best_checkpoint(checkpoint: Path, previous: Path, payload: dict[str, Any]) -> str:
    """Rotate the primary into the recovery slot, then atomically publish best."""

    if checkpoint.is_file():
        previous_bytes = checkpoint.read_bytes()
        atomic_publish(previous, lambda target, data=previous_bytes: target.write_bytes(data))
    _save(checkpoint, payload)
    return sha256_file(checkpoint)


def count_effective_group_step(
    counts: dict[str, int], optimizer: torch.optim.Optimizer, active_groups: list[str]
) -> dict[str, int]:
    updated = {name: int(counts.get(name, 0)) for name in _PARAMETER_GROUPS}
    present = {str(group["name"]) for group in optimizer.param_groups}
    for name in set(active_groups) & present:
        updated[name] += 1
    return updated


def validate_selected_checkpoint(
    selected: Any,
    *,
    expected_identity: dict[str, Any],
    expected_identity_sha256: str,
    expected_epoch: int,
    expected_metric: float,
    expected_selection_role: str = "primary",
) -> None:
    if not isinstance(selected, dict):
        raise ValueError("best.pt payload is invalid")
    validate_resume_identity(
        selected.get("resume_identity"),
        expected_identity,
        selected.get("resume_identity_sha256"),
    )
    if selected.get("resume_identity_sha256") != expected_identity_sha256:
        raise ValueError("best.pt identity is not the selected last.pt identity")
    if selected.get("epoch") != expected_epoch:
        raise ValueError("best.pt epoch does not match last.pt best_epoch")
    metric = selected.get("validation")
    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(metric)
    ):
        raise ValueError("best.pt validation metadata is invalid")
    if float(metric) != expected_metric:
        raise ValueError("best.pt validation does not match last.pt best_metric")
    if selected.get("selection_role") != expected_selection_role:
        raise ValueError("best.pt selection role is invalid")


def _finish_resource_monitor_once(
    state: dict[str, Any],
    device: torch.device,
    *,
    peak_allocated_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
    preserve_primary_error: bool = False,
) -> dict[str, Any]:
    """Stop the sampler once, including when CUDA work raises."""

    if state.get("finished"):
        resources = state.get("resources")
        if not isinstance(resources, dict):
            raise RuntimeError("resource monitor finish previously failed")
        return resources
    monitor = state.get("monitor")
    if monitor is None or not callable(getattr(monitor, "finish", None)):
        raise RuntimeError("resource monitor was not started")
    if peak_allocated_bytes is None and device.type == "cuda":
        try:
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        except BaseException as exc:
            if not preserve_primary_error and not isinstance(exc, (RuntimeError, ValueError)):
                raise
            state.setdefault("peak_query_errors", []).append(
                f"max_memory_allocated failed with {type(exc).__name__}: {exc}"
            )
    if peak_reserved_bytes is None and device.type == "cuda":
        try:
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        except BaseException as exc:
            if not preserve_primary_error and not isinstance(exc, (RuntimeError, ValueError)):
                raise
            state.setdefault("peak_query_errors", []).append(
                f"max_memory_reserved failed with {type(exc).__name__}: {exc}"
            )
    state["finished"] = True
    resources = monitor.finish(
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
    )
    if state.get("peak_query_errors"):
        resources["failure_peak_query_errors"] = list(state["peak_query_errors"])
    state["resources"] = resources
    return resources


def _record_failure_resources(
    output: Path,
    args: argparse.Namespace,
    error: BaseException,
    state: dict[str, Any],
    device: torch.device,
) -> None:
    """Best-effort failure telemetry that never replaces the training exception."""

    resources = state.get("resources")
    cleanup_error = None
    if state.get("monitor") is not None and not state.get("finished"):
        try:
            resources = _finish_resource_monitor_once(
                state,
                device,
                preserve_primary_error=True,
            )
        except BaseException as exc:  # preserve and annotate the original training failure
            cleanup_error = f"{type(exc).__name__}: {exc}"
    elif state.get("finished") and not isinstance(resources, dict):
        cleanup_error = "resource monitor finish failed before telemetry was returned"
    payload = {
        "schema_version": 1,
        "status": "failed",
        "research_suite": SUITE,
        "dataset": args.dataset,
        "condition": args.condition,
        "error": f"{type(error).__name__}: {error}",
        "resource_observability": resources,
        "resource_observability_unavailable_reason": cleanup_error,
    }
    try:
        atomic_write_json(output / FAILURE_RESOURCE_FILENAME, payload)
    except BaseException as exc:  # failure reporting must not mask the scientific failure
        error.add_note(
            "failure resource telemetry could not be written: "
            f"{type(exc).__name__}: {exc}"
        )
    if cleanup_error is not None:
        error.add_note(f"resource monitor cleanup failed: {cleanup_error}")


def train_model(payload, protocol, args, device: torch.device, output: Path) -> dict[str, Any]:
    resource_state: dict[str, Any] = {}
    try:
        return _train_model_impl(
            payload,
            protocol,
            args,
            device,
            output,
            resource_state=resource_state,
        )
    except BaseException as exc:
        if resource_state.get("monitor") is not None:
            _record_failure_resources(output, args, exc, resource_state, device)
        raise


def _train_model_impl(
    payload,
    protocol,
    args,
    device: torch.device,
    output: Path,
    *,
    resource_state: dict[str, Any],
) -> dict[str, Any]:
    _require_cuda(device)
    validate_args(args)
    if payload.get("dataset") != args.dataset:
        raise ValueError("requested dataset does not match the verified payload")
    hardware_runtime = validate_hardware_runtime(args, device)
    hardware_runtime.update(
        selected_device=str(device),
        visible_cuda_device_count=torch.cuda.device_count(),
    )
    resource_monitor = RuntimeResourceMonitor(device)
    resource_start = resource_monitor.start()
    resource_state.update(monitor=resource_monitor, finished=False)
    configure_compute(args)
    _seed(args.model_seed)
    data, indices, sampler = _prepare_data(payload, args, device)
    architecture = architecture_configuration(args)
    model = GraphConditionedConductanceNodeClassifier(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        **architecture,
        conductance_mode=CONDITIONS[args.condition]["conductance_mode"],
        max_log_conductance=COMMON["max_log_conductance"],
        edge_chunk_size=args.edge_chunk_size,
    ).to(device)
    initial_state_sha256 = state_sha256(model)
    shared_state_sha256 = shared_initial_state_sha256(model)
    optimizer = make_optimizer(model)
    validate_optimizer_parameter_ownership(model, optimizer)
    schedule = phase_schedule(args.epochs, list(args.phase_fractions))
    total_parameters_at_construction = sum(value.numel() for value in model.parameters())
    optimizer_owned_parameters = sum(
        value.numel() for group in optimizer.param_groups for value in group["params"]
    )
    parameter_observability = {
        "total_parameters": total_parameters_at_construction,
        "trainable_parameters_at_construction": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "optimizer_owned_parameters": optimizer_owned_parameters,
        "frozen_parameters_at_construction": (
            total_parameters_at_construction - optimizer_owned_parameters
        ),
        "optimizer_groups": optimizer_metadata(optimizer),
    }
    data_observability = _v5_data_observability(payload, data, indices, args)
    batch_observability = _v5_batch_observability(data, indices, sampler, args)
    pre_run_observability = {
        "status": "pre_run_configuration",
        "model": {
            "name": "graph_conditioned_conductance_v5",
            "condition": args.condition,
            "layers": args.layers,
            "hidden_dimension": args.hidden_channels,
            "channels": args.hidden_channels,
            "attention_heads": args.heads,
            "ffn_multiplier": args.ffn_multiplier,
            **parameter_observability,
        },
        "data": data_observability,
        "batching": batch_observability,
        "optimization": {
            "epochs_requested": args.epochs,
            "early_stopping_patience": args.patience,
            "planned_maximum_optimizer_steps": batch_observability[
                "planned_maximum_training_batches"
            ],
            "actual_optimizer_steps": observed(
                None, reason="training has not started", unit="steps"
            ),
        },
        "precision": {
            "precision": args.precision,
            "amp": args.precision == "bf16",
            "autocast_dtype": (
                "bfloat16"
                if args.precision == "bf16"
                else observed(
                    None,
                    reason="autocast is disabled by the fp32 execution profile",
                )
            ),
            "tf32": args.tf32,
        },
        "hardware": hardware_runtime,
        "resources": resource_start,
        "modes": {
            "debug": False,
            "subset": False,
            "fast_mode": False,
            "test_evaluated": False,
        },
    }
    print(json.dumps(pre_run_observability, sort_keys=True), flush=True)
    resume_identity = build_resume_identity(
        args, protocol, schedule, initial_state_sha256=initial_state_sha256
    )
    resume_identity_sha256 = _canonical_sha256(resume_identity)
    checkpoint, previous_checkpoint, last_path, history_path = (
        output / "best.pt",
        output / "best.previous.pt",
        output / "last.pt",
        output / "history.json",
    )
    history: list[dict[str, Any]] = []
    start_epoch, best_metric, best_epoch, optimizer_steps = 1, -math.inf, 0, 0
    global_best_metric, global_best_epoch = -math.inf, 0
    joint_best_metric, joint_best_epoch = -math.inf, 0
    elapsed_before = 0.0
    peak_allocated_before = peak_reserved_before = 0
    first_c_gradient = None
    best_checkpoint_sha256 = None
    best_recovery_slot = "fresh"
    effective_group_steps = {name: 0 for name in _PARAMETER_GROUPS}
    gradient_groups_validated_this_invocation: set[str] = set()
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get("schema_version") != 3:
            raise ValueError("last.pt uses an incompatible V5 selection-state schema")
        validate_resume_identity(
            saved.get("resume_identity"),
            resume_identity,
            saved.get("resume_identity_sha256"),
        )
        saved_history, saved_epoch = saved.get("history"), saved.get("epoch")
        if (
            not isinstance(saved_history, list)
            or isinstance(saved_epoch, bool)
            or not isinstance(saved_epoch, int)
            or saved_epoch != len(saved_history)
            or saved_epoch < 1
        ):
            raise ValueError("last.pt epoch/history state is invalid")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved_history
        start_epoch = args.epochs + 1 if saved.get("complete") is True else saved_epoch + 1
        best_metric, best_epoch = float(saved["best_metric"]), int(saved["best_epoch"])
        global_best_metric = float(saved.get("global_best_metric", best_metric))
        global_best_epoch = int(saved.get("global_best_epoch", best_epoch))
        joint_best_metric = float(saved["joint_best_metric"])
        joint_best_epoch = int(saved["joint_best_epoch"])
        elapsed_before = float(saved.get("elapsed_seconds", 0.0))
        peak_allocated_before = int(saved.get("peak_cuda_allocated_bytes", 0))
        peak_reserved_before = int(saved.get("peak_cuda_reserved_bytes", 0))
        optimizer_steps, first_c_gradient = int(saved["optimizer_steps"]), saved["first_c_gradient"]
        raw_group_steps = saved.get("effective_optimizer_steps_by_group")
        if not isinstance(raw_group_steps, dict) or any(
            isinstance(raw_group_steps.get(name), bool)
            or not isinstance(raw_group_steps.get(name), int)
            or raw_group_steps[name] < 0
            for name in _PARAMETER_GROUPS
        ):
            raise ValueError("last.pt effective parameter-group step counts are invalid")
        effective_group_steps = {name: raw_group_steps[name] for name in _PARAMETER_GROUPS}
        best_checkpoint_sha256 = saved.get("best_checkpoint_sha256")
        best_recovery_slot = recover_best_checkpoint(
            checkpoint, previous_checkpoint, best_checkpoint_sha256
        )
        torch.set_rng_state(saved["cpu_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
    validation_indices = indices["validation"] if indices is not None else None
    validation_data = _validation_source(data, sampler)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        phase, local_epoch = phase_at(schedule, epoch)
        phase_state = configure_phase(model, phase, local_epoch)
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        label_count, batch_count = 0, 0
        maximum_preclip_gradient_norm = torch.zeros((), dtype=torch.float32, device=device)
        for graph, train_indices in _training_batches(
            data, indices, sampler, epoch, device, args.model_seed, args
        ):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args):
                logits = model(graph)
                loss, count = training_loss(logits, graph, train_indices)
            if phase_state["active_parameter_groups"]:
                loss.backward()
                groups_requiring_validation = set(
                    phase_state["active_parameter_groups"]
                ) - gradient_groups_validated_this_invocation
                if groups_requiring_validation:
                    validate_active_gradient_connectivity(
                        model, sorted(groups_requiring_validation)
                    )
                    gradient_groups_validated_this_invocation.update(
                        groups_requiring_validation
                    )
                if (
                    args.condition == "shared_dynamic_c"
                    and phase_state["coordinate"] in {"conductance", "joint"}
                    and first_c_gradient is None
                ):
                    first_c_gradient = require_first_step_conductance_gradient(model)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    (value for value in model.parameters() if value.requires_grad),
                    COMMON["gradient_clip_norm"],
                    error_if_nonfinite=False,
                    foreach=True,
                )
                require_finite_gradient_norm_async(gradient_norm)
                maximum_preclip_gradient_norm = torch.maximum(
                    maximum_preclip_gradient_norm, gradient_norm.float()
                )
                optimizer.step()
                optimizer_steps += 1
                effective_group_steps = count_effective_group_step(
                    effective_group_steps, optimizer, phase_state["active_parameter_groups"]
                )
            loss_sum += loss.detach().float() * count
            label_count += count
            batch_count += 1
        if not label_count:
            raise RuntimeError("training phase produced no supervised labels")
        observation = evaluate(
            model,
            validation_data if indices is not None else data["validation"],
            validation_indices,
            device=device,
            precision=args.precision,
        )
        metric = float(observation["metric"])
        train_loss_tensor = loss_sum / label_count
        require_finite_tensor(train_loss_tensor, "epoch training loss")
        train_loss = float(train_loss_tensor)
        maximum_preclip_gradient_norm_value = float(maximum_preclip_gradient_norm)
        if not math.isfinite(train_loss) or not math.isfinite(metric):
            raise FloatingPointError("nonfinite epoch loss or validation metric")
        row = {
            "epoch": epoch,
            "phase": phase_state,
            "optimizer_steps": optimizer_steps,
            "effective_optimizer_steps_by_group": dict(effective_group_steps),
            "train_loss": train_loss,
            "train_label_count": label_count,
            "train_batches": batch_count,
            "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm_value,
            "elapsed_wall_seconds": time.perf_counter() - epoch_started,
            "validation": metric,
            "layers": layer_diagnostics(model),
        }
        history.append(row)
        eligibility = selection_eligibility(args.condition, phase)
        if metric > global_best_metric:
            global_best_metric, global_best_epoch = metric, epoch
        # fixed_c selects over its complete training trajectory.  For the
        # dynamic arm, warmup C=1 is retained as an auxiliary prediction score
        # but cannot become the mechanism-bearing primary checkpoint.
        if eligibility["primary"] and metric > best_metric:
            best_metric, best_epoch = metric, epoch
            best_checkpoint_sha256 = publish_best_checkpoint(
                checkpoint,
                previous_checkpoint,
                {
                    "model_state": model.state_dict(),
                    "architecture": architecture,
                    "condition": args.condition,
                    "configuration": configuration(args),
                    "schedule": schedule,
                    "resume_identity": resume_identity,
                    "resume_identity_sha256": resume_identity_sha256,
                    "epoch": epoch,
                    "validation": metric,
                    "selection_role": "primary",
                    "selection_scope": (
                        "all_epochs" if args.condition == "fixed_c" else "c_active_epochs"
                    ),
                    "phase": phase,
                },
            )
        if eligibility["joint_early_stopping"] and metric > joint_best_metric:
            joint_best_metric, joint_best_epoch = metric, epoch
        row["selection"] = {
            "primary_eligible": eligibility["primary"],
            "primary_best_validation": best_metric if math.isfinite(best_metric) else None,
            "primary_best_epoch": best_epoch or None,
            "global_prediction_best_validation": global_best_metric,
            "global_prediction_best_epoch": global_best_epoch,
            "joint_early_stopping_eligible": eligibility["joint_early_stopping"],
            "joint_best_validation": (
                joint_best_metric if math.isfinite(joint_best_metric) else None
            ),
            "joint_best_epoch": joint_best_epoch or None,
        }
        atomic_write_json(history_path, history)
        efficiency = merge_efficiency(
            elapsed_before,
            peak_allocated_before,
            peak_reserved_before,
            time.perf_counter() - started,
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        )
        stop_after_epoch = epoch == args.epochs or should_stop_early(
            args.condition,
            phase,
            epoch,
            primary_best_epoch=best_epoch,
            joint_best_epoch=joint_best_epoch,
            patience=args.patience,
        )
        _save(
            last_path,
            {
                "schema_version": 3,
                "complete": stop_after_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "resume_identity": resume_identity,
                "resume_identity_sha256": resume_identity_sha256,
                "epoch": epoch,
                "phase": phase_state,
                "history": history,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_checkpoint_sha256": best_checkpoint_sha256,
                "optimizer_steps": optimizer_steps,
                "effective_optimizer_steps_by_group": effective_group_steps,
                "global_best_metric": global_best_metric,
                "global_best_epoch": global_best_epoch,
                "joint_best_metric": joint_best_metric,
                "joint_best_epoch": joint_best_epoch,
                "first_c_gradient": first_c_gradient,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                **efficiency,
            },
        )
        if epoch == 1 or epoch % 10 == 0:
            primary_best_text = f"{best_metric:.6f}" if math.isfinite(best_metric) else "pending"
            joint_best_text = (
                "n/a"
                if args.condition == "fixed_c"
                else f"{joint_best_metric:.6f}"
                if math.isfinite(joint_best_metric)
                else "pending"
            )
            print(
                f"{args.dataset}/{args.condition} epoch={epoch} phase={phase} "
                f"loss={row['train_loss']:.6f} val={metric:.6f} "
                f"primary_best={primary_best_text} "
                f"global_best={global_best_metric:.6f} "
                f"joint_best={joint_best_text}",
                flush=True,
            )
        if stop_after_epoch:
            break
    if best_epoch < 1 or not math.isfinite(best_metric) or best_checkpoint_sha256 is None:
        raise RuntimeError("V5 completed without a finite primary validation checkpoint")
    if args.condition == "shared_dynamic_c" and (
        joint_best_epoch < 1 or not math.isfinite(joint_best_metric)
    ):
        raise RuntimeError("dynamic V5 completed without a finite joint early-stopping metric")
    best_recovery_slot = recover_best_checkpoint(
        checkpoint, previous_checkpoint, best_checkpoint_sha256
    )
    selected = torch.load(checkpoint, map_location=device, weights_only=False)
    validate_selected_checkpoint(
        selected,
        expected_identity=resume_identity,
        expected_identity_sha256=resume_identity_sha256,
        expected_epoch=best_epoch,
        expected_metric=best_metric,
        expected_selection_role="primary",
    )
    model.load_state_dict(selected["model_state"])
    for operator in model.operators:
        operator.estimator.override = None
    interventions = selected_checkpoint_interventions(
        model,
        validation_data if indices is not None else data["validation"],
        validation_indices,
        device=device,
        precision=args.precision,
    )
    learned_metric = float(interventions["learned"]["metric"])
    if not math.isfinite(learned_metric):
        raise FloatingPointError("selected best checkpoint produced a nonfinite metric")
    selected_checkpoint_recheck = {
        "recorded": best_metric,
        "recomputed": learned_metric,
        "delta": learned_metric - best_metric,
        "declared_tolerance": 1e-6,
        "within_declared_tolerance": math.isclose(
            learned_metric, best_metric, rel_tol=0.0, abs_tol=1e-6
        ),
        "non_gating": True,
        "reason": "CUDA scatter/threshold replay is not claimed bitwise deterministic",
    }
    torch.cuda.synchronize(device)
    efficiency = merge_efficiency(
        elapsed_before,
        peak_allocated_before,
        peak_reserved_before,
        time.perf_counter() - started,
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    )
    resource_observability = _finish_resource_monitor_once(
        resource_state,
        device,
        peak_allocated_bytes=int(efficiency["peak_cuda_allocated_bytes"]),
        peak_reserved_bytes=int(efficiency["peak_cuda_reserved_bytes"]),
    )
    resource_observability["cuda_allocator_peak_boundary"] = (
        "CUDA statistics reset immediately before this invocation's epoch loop; "
        "cumulative maxima are restored from last.pt across resumed invocations"
    )
    observed_training_batches = sum(int(row["train_batches"]) for row in history)
    optimization_observability = {
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": args.patience,
        "planned_maximum_optimizer_steps": batch_observability[
            "planned_maximum_training_batches"
        ],
        "actual_training_batches": observed_training_batches,
        "actual_optimizer_steps": optimizer_steps,
        "effective_optimizer_steps_by_group": effective_group_steps,
        "gradient_accumulation_steps": 1,
        "step_definition": (
            "one optimizer update per physical batch when the scheduled phase has at "
            "least one active parameter group"
        ),
        "optimizer_step_difference_from_training_batches": (
            observed_training_batches - optimizer_steps
        ),
    }
    result = {
        "schema_version": 1,
        "status": "passed",
        "research_suite": SUITE,
        "dataset": args.dataset,
        "condition": args.condition,
        "model_seed": args.model_seed,
        **CONDITIONS[args.condition],
        "configuration": configuration(args),
        "schedule": schedule,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "optimizer_steps": optimizer_steps,
        "effective_optimizer_steps_by_group": effective_group_steps,
        "optimization_observability": optimization_observability,
        "comparison_design": COMPARISON_DESIGN,
        "hardware_execution": {
            **hardware_runtime,
            "timing_boundary": "CUDA synchronized before measured run and final accounting",
            "gpu_sm_utilization": resource_observability["interval_series"][
                "gpu_sm_utilization_percent"
            ],
            "small_full_graph_limit": (
                "Cora/Citeseer/PubMed are single small full graphs and cannot fill 48 GiB "
                "without scientifically invalid duplicate work"
            ),
        },
        "pre_run_observability": pre_run_observability,
        "resource_observability": resource_observability,
        "data_observability": data_observability,
        "batch_observability": batch_observability,
        "parameter_observability": parameter_observability,
        "gradient_clipping": {
            "max_norm": COMMON["gradient_clip_norm"],
            "error_if_nonfinite": "stream_ordered_async_assert_before_optimizer_step",
            "maximum_observed_preclip_norm": max(
                row["maximum_preclip_gradient_norm"] for row in history
            ),
        },
        "gradient_connectivity": {
            "validated_parameter_groups_this_invocation": sorted(
                gradient_groups_validated_this_invocation
            ),
            "validation_boundary": (
                "first actual task-loss backward that activates each parameter group; "
                "every tensor in that active group must have a finite gradient"
            ),
            "optimizer_parameter_ownership": "exact identity and group match validated",
        },
        "global_best_validation": global_best_metric,
        "global_best_epoch": global_best_epoch,
        "joint_best_validation": (
            joint_best_metric if args.condition == "shared_dynamic_c" else None
        ),
        "joint_best_epoch": joint_best_epoch if args.condition == "shared_dynamic_c" else None,
        "checkpoint_selection": {
            "primary_role": (
                "all_epoch_prediction_best"
                if args.condition == "fixed_c"
                else "c_active_mechanism_best"
            ),
            "primary_validation": best_metric,
            "primary_epoch": best_epoch,
            "global_prediction_validation": global_best_metric,
            "global_prediction_epoch": global_best_epoch,
            "global_prediction_checkpoint_preserved": (
                args.condition == "fixed_c" or global_best_epoch == best_epoch
            ),
            "early_stopping_monitor": (
                "primary_all_epoch_best" if args.condition == "fixed_c" else "joint_best"
            ),
            "joint_monitor_applicable": args.condition == "shared_dynamic_c",
            "joint_validation": (
                joint_best_metric if args.condition == "shared_dynamic_c" else None
            ),
            "joint_epoch": joint_best_epoch if args.condition == "shared_dynamic_c" else None,
            "test_used": False,
        },
        "validation": best_metric,
        "metric_name": METRIC_BY_DATASET[args.dataset],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_previous_checkpoint": str(previous_checkpoint.resolve()),
        "best_checkpoint_recovery_slot": best_recovery_slot,
        "resume_identity": resume_identity,
        "resume_identity_sha256": resume_identity_sha256,
        "resume_semantics": RESUME_SEMANTICS,
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history": str(history_path.resolve()),
        "optimizer_groups": optimizer_metadata(optimizer),
        "total_parameters": total_parameters_at_construction,
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "allocated_parameter_capacity": sum(value.numel() for value in model.parameters()),
        "first_active_conductance_gradient": first_c_gradient,
        "selected_checkpoint_interventions": interventions,
        "selected_checkpoint_recheck": selected_checkpoint_recheck,
        "sampling": sampler.metadata()
        if sampler is not None
        else {"mode": "full", "validation_graph": "complete_official_graph"},
        "protocol": protocol,
        "cache_sha256": protocol["data_sha256"],
        "source_sha256": resume_identity["source_sha256"],
        "initial_state_sha256": initial_state_sha256,
        "shared_initial_state_sha256": shared_state_sha256,
        "history_sha256": sha256_file(history_path),
        "evaluation_split": "validation",
        "test_evaluated": False,
        "versions": _versions(),
        "gpu": torch.cuda.get_device_name(device),
        "throughput": training_throughput(history, float(efficiency["elapsed_seconds"])),
        "peak_cuda_allocated_fraction_of_visible_capacity": (
            efficiency["peak_cuda_allocated_bytes"] / hardware_runtime["total_memory_bytes"]
        ),
        "peak_cuda_reserved_fraction_of_visible_capacity": (
            efficiency["peak_cuda_reserved_bytes"] / hardware_runtime["total_memory_bytes"]
        ),
        **efficiency,
    }
    print(
        json.dumps(
            {
                "status": "post_run_observability",
                "dataset": args.dataset,
                "condition": args.condition,
                "epochs_completed": len(history),
                "optimizer_steps": optimizer_steps,
                "resource_summary": resource_observability["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    device = torch.device(args.device)
    _require_cuda(device)
    output, data_root = (
        args.output_dir.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
    )
    if output == data_root or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("V5 output and V1 dataset cache must not overlap")
    if (
        output.exists()
        and any(output.iterdir())
        and not (args.resume and (output / "last.pt").exists())
    ):
        raise FileExistsError("nonempty output has no resumable V5 last.pt")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "running",
        "research_suite": SUITE,
        "dataset": args.dataset,
        "condition": args.condition,
        "configuration": configuration(args),
        "test_evaluated": False,
    }
    atomic_write_json(output / "metrics.json", record)
    try:
        payload, protocol = load_dataset(args.dataset, data_root, allow_download=False)
        record.update(train_model(payload, protocol, args, device, output))
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        failure_resources = output / FAILURE_RESOURCE_FILENAME
        if failure_resources.is_file():
            record.update(
                failure_resource_observability=str(failure_resources.resolve()),
                failure_resource_observability_sha256=sha256_file(failure_resources),
            )
        try:
            atomic_write_json(output / "metrics.json", record)
        except BaseException as reporting_error:
            exc.add_note(
                "failed metrics could not be written without replacing the scientific error: "
                f"{type(reporting_error).__name__}: {reporting_error}"
            )
        raise
    atomic_write_json(output / "metrics.json", record)
    print(f"passed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
