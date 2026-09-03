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

from ..ablation.model import state_sha256
from ..ablation.train import _configure_fp32, _make_data, _require_cuda, training_loss
from ..benchmark import _seed, _versions
from ..benchmark_data import load_dataset, sha256_file
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

ROOT = Path(__file__).resolve().parents[3]
RESUME_SEMANTICS = (
    "epoch-boundary deterministic resume from stored model/optimizer/RNG state; "
    "CUDA kernels are not claimed bitwise deterministic"
)
_PARAMETER_GROUPS = ("backbone", "spatial_w", "beta", "conductance")
_SHARED_IMPLEMENTATION_SOURCES = (
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
    return {
        **COMMON,
        **architecture_configuration(args),
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "workers": args.workers,
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
        "loader_workers": args.workers,
        "loader_worker_policy": (
            "zero: PPI has only 20 in-memory training graphs; subprocess IPC/startup can "
            "cost more than collating three batch-8 steps and complicates exact resume audits"
        ),
        "sample_prefetch": args.sample_prefetch,
    }


def resolve_hardware_arguments(args: argparse.Namespace) -> None:
    """Resolve explicit execution defaults before identity/config creation."""

    profile = HARDWARE_PROFILES[args.hardware_profile]
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
        "loader_workers": args.workers,
        "loader_worker_policy": (
            "zero: PPI is 20 static in-memory training graphs; multi-process IPC is not "
            "assumed faster and would widen the exact-resume audit surface"
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
    parser.add_argument("--workers", type=int, default=0)
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
    if args.workers != 0:
        raise ValueError("V5 currently requires workers=0 for exact resumability")
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


def train_model(payload, protocol, args, device: torch.device, output: Path) -> dict[str, Any]:
    _require_cuda(device)
    validate_args(args)
    if payload.get("dataset") != args.dataset:
        raise ValueError("requested dataset does not match the verified payload")
    hardware_runtime = validate_hardware_runtime(args, device)
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
    optimizer = make_optimizer(model)
    schedule = phase_schedule(args.epochs, list(args.phase_fractions))
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
    elapsed_before = 0.0
    peak_allocated_before = peak_reserved_before = 0
    first_c_gradient = None
    best_checkpoint_sha256 = None
    best_recovery_slot = "fresh"
    effective_group_steps = {name: 0 for name in _PARAMETER_GROUPS}
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
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
        atomic_write_json(history_path, history)
        if metric > global_best_metric:
            global_best_metric, global_best_epoch = metric, epoch
        # Model selection is joint-phase only. Otherwise the dynamic arm could
        # accidentally publish its C=1 warmup checkpoint as the V5 result.
        if phase == "joint" and metric > best_metric:
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
                },
            )
        efficiency = merge_efficiency(
            elapsed_before,
            peak_allocated_before,
            peak_reserved_before,
            time.perf_counter() - started,
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        )
        stop_after_epoch = epoch == args.epochs or (
            phase == "joint" and best_epoch > 0 and epoch - best_epoch >= args.patience
        )
        _save(
            last_path,
            {
                "schema_version": 2,
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
                "first_c_gradient": first_c_gradient,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                **efficiency,
            },
        )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{args.dataset}/{args.condition} epoch={epoch} phase={phase} "
                f"loss={row['train_loss']:.6f} val={metric:.6f} "
                f"joint_best={best_metric:.6f}",
                flush=True,
            )
        if stop_after_epoch:
            break
    if best_epoch < 1 or not math.isfinite(best_metric) or best_checkpoint_sha256 is None:
        raise RuntimeError("V5 completed without a finite joint-phase best checkpoint")
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
    return {
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
        "comparison_design": COMPARISON_DESIGN,
        "hardware_execution": {
            **hardware_runtime,
            "timing_boundary": "CUDA synchronized before measured run and final accounting",
            "gpu_sm_utilization": "not measured; use a time-series device monitor",
            "small_full_graph_limit": (
                "Cora/Citeseer/PubMed are single small full graphs and cannot fill 48 GiB "
                "without scientifically invalid duplicate work"
            ),
        },
        "gradient_clipping": {
            "max_norm": COMMON["gradient_clip_norm"],
            "error_if_nonfinite": "stream_ordered_async_assert_before_optimizer_step",
            "maximum_observed_preclip_norm": max(
                row["maximum_preclip_gradient_norm"] for row in history
            ),
        },
        "global_best_validation": global_best_metric,
        "global_best_epoch": global_best_epoch,
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
        "total_parameters": sum(value.numel() for value in model.parameters()),
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
        "history_sha256": sha256_file(history_path),
        "evaluation_split": "validation",
        "test_evaluated": False,
        "versions": _versions(),
        "gpu": torch.cuda.get_device_name(device),
        "throughput": {
            "supervised_labels_per_elapsed_second": (
                sum(row["train_label_count"] for row in history) / efficiency["elapsed_seconds"]
                if efficiency["elapsed_seconds"] > 0
                else None
            ),
            "training_batches_per_elapsed_second": (
                sum(row["train_batches"] for row in history) / efficiency["elapsed_seconds"]
                if efficiency["elapsed_seconds"] > 0
                else None
            ),
            "elapsed_includes_validation_checkpointing_and_interventions": True,
        },
        "peak_cuda_allocated_fraction_of_visible_capacity": (
            efficiency["peak_cuda_allocated_bytes"] / hardware_runtime["total_memory_bytes"]
        ),
        "peak_cuda_reserved_fraction_of_visible_capacity": (
            efficiency["peak_cuda_reserved_bytes"] / hardware_runtime["total_memory_bytes"]
        ),
        **efficiency,
    }


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
        atomic_write_json(output / "metrics.json", record)
        raise
    atomic_write_json(output / "metrics.json", record)
    print(f"passed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
