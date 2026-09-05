"""Train Cycle PE v2 from a sparse DFS cycle basis without matrix factorization.

This is an isolated experiment: it does not change or invoke the v1 cycle-set
model, and it never trains comparison-paper models. Actual training is CUDA-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chartgat.cache import atomic_publish, atomic_write_json
from chartgat.execution import add_execution_arguments, configure_execution
from chartgat.observability import observed
from research.cycle_pe.benchmark_data import EXPECTED_SIZES
from research.cycle_pe.resource_monitor import (
    FailureSafeResourceMonitor,
    persist_failure_artifacts,
    resource_failure_boundary,
    resource_failure_observations,
)
from research.cycle_pe.v2.data import (
    BASIS_BACKENDS,
    DATASETS,
    DEFAULT_BASIS_BACKEND,
    Graph,
    collate,
    load_benchmark,
)
from research.cycle_pe.v2.model import MODEL_NAMES, CycleBasisPEModel, architecture_protocol

TRACK_NAME = "cycle_pe"
HARDWARE_PROFILES = ("portable", "a6000-48gb")
A6000_MIN_TOTAL_BYTES = 40 * 1024**3
IMPLEMENTATION_FILES = (
    "research/cycle_pe/v2/benchmark.py",
    "research/cycle_pe/v2/calibration.py",
    "research/cycle_pe/v2/basis.py",
    "research/cycle_pe/v2/data.py",
    "research/cycle_pe/v2/model.py",
    "research/cycle_pe/benchmark_data.py",
    "research/cycle_pe/benchmark_models.py",
    "research/cycle_pe/paper_model.py",
    "research/cycle_pe/resource_monitor.py",
    "src/chartgat/algebra.py",
    "src/chartgat/cache.py",
    "src/chartgat/execution.py",
    "src/chartgat/observability.py",
)


def _amp_policy(requested: bool, device: torch.device) -> dict[str, Any]:
    """Choose a range-safe mixed-precision policy for the graph-level readout.

    This model pools both node/edge means and *sums*.  The latter make the
    backward pass particularly vulnerable to FP16's small exponent range when
    GradScaler starts at its usual 65536 scale.  A6000/Ampere has native BF16,
    which keeps FP32's exponent range and needs no loss scaling.  On CUDA
    devices without BF16 we deliberately retain FP32 rather than silently
    reintroducing the known FP16 overflow path.
    """
    if requested and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return {
            "requested": True,
            "enabled": True,
            "dtype": torch.bfloat16,
            "dtype_name": "bfloat16",
            "fallback": None,
            "gradient_scaler": False,
        }
    return {
        "requested": bool(requested),
        "enabled": False,
        "dtype": torch.float32,
        "dtype_name": "disabled",
        "fallback": "bf16_unavailable_use_fp32" if requested else None,
        "gradient_scaler": False,
    }


def _precision_identity(policy: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe arithmetic fields that bind artifacts and resume."""
    return {
        "amp_effective": policy["enabled"],
        "autocast_dtype": policy["dtype_name"],
        "fallback": policy["fallback"],
        "gradient_scaler": policy["gradient_scaler"],
    }


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in IMPLEMENTATION_FILES
    }


def _model_name(args: argparse.Namespace) -> str:
    return MODEL_NAMES[args.encoding]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", choices=("benchmark",), default="benchmark")
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    result.add_argument(
        "--encoding",
        choices=tuple(MODEL_NAMES),
        default="se",
        help="se: cycle structure; pe: the same SE plus a relative cycle-position residual",
    )
    result.add_argument("--data-root", type=Path, default=Path("data/paper"))
    result.add_argument("--output-dir", type=Path, default=Path("results/cycle_pe_v2/benchmark"))
    result.add_argument("--device", default="cuda")
    for seed in ("data", "split", "chart", "model"):
        result.add_argument(f"--{seed}-seed", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--prefetch-factor", type=int, default=2)
    result.add_argument(
        "--hardware-profile",
        choices=HARDWARE_PROFILES,
        default="portable",
        help="portable keeps legacy execution; a6000-48gb enables a guarded high-throughput run",
    )
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--epochs", type=int, default=300)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--hidden-dim", type=int, default=128)
    result.add_argument("--pe-dim", type=int, default=64)
    result.add_argument("--layers", type=int, default=10)
    result.add_argument("--ffn-multiplier", type=int, default=4)
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--layer-scale", type=float, default=0.1)
    result.add_argument("--max-parameters", type=int, default=20_000_000)
    result.add_argument(
        "--basis-backend",
        choices=BASIS_BACKENDS,
        default=DEFAULT_BASIS_BACKEND,
        help=(
            "all signed DFS fundamental cycles; sparse edge-cycle message passing, "
            "without QR, SVD, or basis truncation"
        ),
    )
    result.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume an exactly matching interrupted V2 run from its atomic epoch checkpoint",
    )
    result.add_argument("--validation-only", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--test-checkpoint", type=Path, help=argparse.SUPPRESS)
    add_execution_arguments(result)
    return result


def _validate(args: argparse.Namespace) -> None:
    if any(getattr(args, f"{seed}_seed") < 0 for seed in ("data", "split", "chart", "model")):
        raise ValueError("seeds must be nonnegative")
    for key in (
        "batch_size",
        "prefetch_factor",
        "epochs",
        "patience",
        "hidden_dim",
        "pe_dim",
        "layers",
        "ffn_multiplier",
        "max_parameters",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if (
        args.workers < 0
        or args.lr <= 0
        or args.weight_decay < 0
        or not 0 <= args.dropout < 1
        or not 0 < args.layer_scale <= 1
    ):
        raise ValueError("invalid worker count or optimizer settings")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("datasets must not contain duplicates")
    if args.validation_only and args.prepare_only:
        raise ValueError("--validation-only cannot be combined with --prepare-only")
    if args.test_checkpoint is not None and (args.validation_only or args.prepare_only):
        raise ValueError("--test-checkpoint is an isolated test-only mode")
    if args.test_checkpoint is not None and len(args.datasets) != 1:
        raise ValueError("--test-checkpoint requires exactly one dataset")
    if not args.prepare_only and (
        torch.device(args.device).type != "cuda" or not torch.cuda.is_available()
    ):
        raise RuntimeError("Cycle PE v2 benchmark training requires CUDA; no CPU fallback")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _worker_seed(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _loader(graphs: list[Graph], args: argparse.Namespace, *, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.model_seed)
    options: dict[str, Any] = {
        "dataset": graphs,
        "batch_size": getattr(args, "effective_batch_size", args.batch_size),
        "shuffle": train,
        "num_workers": args.workers,
        "pin_memory": True,
        "collate_fn": collate,
        "generator": generator,
        "worker_init_fn": _worker_seed,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(
        **options,
    )


def _resume_configuration(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "model_seed",
        "batch_size",
        "workers",
        "prefetch_factor",
        "hardware_profile",
        "epochs",
        "patience",
        "lr",
        "weight_decay",
        "hidden_dim",
        "pe_dim",
        "layers",
        "ffn_multiplier",
        "dropout",
        "layer_scale",
        "basis_backend",
        "encoding",
        "amp",
        "compile",
    )
    precision = _amp_policy(args.amp, torch.device(args.device))
    precision_identity = _precision_identity(precision)
    return {
        "schema": "cycle-dfs-se-relative-pe-v2-epoch-resume-1",
        "dataset": dataset,
        "model": _model_name(args),
        "arguments": {name: getattr(args, name) for name in names},
        # --amp is a request, not necessarily the effective arithmetic. Bind
        # resume to the resolved policy so a checkpoint cannot silently switch
        # between BF16 and the safe FP32 fallback on another CUDA device.
        "precision": precision_identity,
        "implementation_sha256": implementation_hashes(),
    }


def _capture_rng(loader: DataLoader) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "kind": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "loader": loader.generator.get_state() if loader.generator is not None else None,
    }


def _restore_rng(state: dict[str, Any], loader: DataLoader) -> None:
    if not isinstance(state, dict) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "loader",
    }:
        raise ValueError("last.pt RNG state is invalid")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["kind"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    if loader.generator is not None and state["loader"] is not None:
        loader.generator.set_state(state["loader"].cpu())


def _artifact_matches(path_value: Any, digest_value: Any) -> bool:
    if not isinstance(path_value, str) or not isinstance(digest_value, str):
        return False
    path = Path(path_value)
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest_value


def _recover_best_checkpoint(checkpoint: Path, previous: Path, expected_hash: Any) -> None:
    """Restore the best file after the best->last atomic publication crash window."""
    if not isinstance(expected_hash, str):
        raise ValueError("last.pt best-checkpoint hash is invalid")
    if (
        checkpoint.is_file()
        and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected_hash
    ):
        return
    if not previous.is_file() or hashlib.sha256(previous.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("last.pt best checkpoint has no valid recovery slot")
    previous_bytes = previous.read_bytes()
    atomic_publish(checkpoint, lambda path, payload=previous_bytes: path.write_bytes(payload))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("best-checkpoint recovery did not preserve its exact hash")


def _hardware_report(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    report = {
        "profile": args.hardware_profile,
        "selected_device": str(device),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "name": properties.name,
        "total_bytes": int(total_bytes),
        "free_bytes_at_start": int(free_bytes),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }
    if args.hardware_profile == "a6000-48gb":
        if total_bytes < A6000_MIN_TOTAL_BYTES or properties.major < 8:
            raise RuntimeError(
                "a6000-48gb requires one visible CUDA device with >=40 GiB total memory "
                "and compute capability >=8.0; refusing a MIG/small-device fallback"
            )
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    report["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    report["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    return report


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


def _require_finite_loss(loss: torch.Tensor, label: str) -> None:
    predicate = torch.isfinite(loss)
    assertion = getattr(torch, "_assert_async", None)
    if loss.device.type == "cuda" and assertion is not None:
        assertion(predicate, label)
    elif not bool(predicate):
        raise FloatingPointError(label)


def _validate_optimizer_ownership(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Require every trainable tensor to be owned exactly once by the optimizer."""
    trainable = {
        id(parameter): (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError("Cycle PE V2 model has no trainable parameters")
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    owned_ids = [id(parameter) for parameter in owned]
    duplicate_ids = sorted(
        parameter_id for parameter_id in set(owned_ids) if owned_ids.count(parameter_id) != 1
    )
    missing_ids = sorted(set(trainable) - set(owned_ids))
    unexpected_ids = sorted(set(owned_ids) - set(trainable))
    if duplicate_ids or missing_ids or unexpected_ids:
        names = {parameter_id: name for parameter_id, (name, _) in trainable.items()}
        duplicate_names = [
            names.get(parameter_id, str(parameter_id)) for parameter_id in duplicate_ids
        ]
        raise RuntimeError(
            "Cycle PE V2 optimizer ownership mismatch; "
            f"missing={[names[parameter_id] for parameter_id in missing_ids]}, "
            f"duplicate={duplicate_names}, "
            f"unexpected_parameter_ids={unexpected_ids}"
        )
    trainable_scalars = sum(parameter.numel() for _, parameter in trainable.values())
    return {
        "validated": True,
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_scalars": trainable_scalars,
        "optimizer_owned_parameter_tensors": len(owned),
        "optimizer_owned_parameter_scalars": sum(parameter.numel() for parameter in owned),
        "optimizer_parameter_groups": [
            {
                "index": index,
                "parameter_tensors": len(group["params"]),
                "parameter_scalars": sum(parameter.numel() for parameter in group["params"]),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
    }


def _validate_first_task_gradients(model: torch.nn.Module) -> dict[str, Any]:
    """Fail closed on disconnected or nonfinite trainable parameters before Adam."""
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Cycle PE V2 model has no trainable parameters")
    missing = [name for name, parameter in trainable if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    if missing:
        raise RuntimeError(
            "Cycle PE V2 has trainable parameters disconnected from the first task loss: "
            + ", ".join(missing)
        )
    if nonfinite:
        raise FloatingPointError(
            "Cycle PE V2 has nonfinite first-step task gradients: " + ", ".join(nonfinite)
        )
    return {
        "validated": True,
        "stage": "after_first_task_backward_before_gradient_clipping_and_optimizer_step",
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_scalars": sum(parameter.numel() for _, parameter in trainable),
        "missing_gradient_parameters": [],
        "nonfinite_gradient_parameters": [],
    }


def _cycle_data_observability(
    dataset: str,
    splits: dict[str, list[Graph]],
) -> dict[str, Any]:
    """Describe exactly the official graphs loaded and used by this invocation."""
    graphs = [graph for split in splits.values() for graph in split]
    if not graphs:
        raise ValueError(f"{dataset}: no official graphs were loaded")
    split_sizes = {name: len(rows) for name, rows in splits.items()}
    loaded_count = sum(split_sizes.values())
    used_count = (
        split_sizes.get("train", 0) + split_sizes.get("validation", 0) + split_sizes.get("test", 0)
    )
    official_split_counts = dict(
        zip(("train", "validation", "test"), EXPECTED_SIZES[dataset], strict=True)
    )
    official_total = sum(official_split_counts.values())
    loaded_fraction = observed(loaded_count / official_total, unit="fraction")
    node_counts = [int(graph.x.shape[0]) for graph in graphs]
    edge_counts = [int(graph.edge_index.shape[1]) for graph in graphs]
    cycle_ranks = [int(graph.cycle_basis.shape[1]) for graph in graphs]
    cycle_memberships = [graph.cycle_basis._nnz() for graph in graphs]
    return {
        "dataset": dataset,
        "source": "official benchmark adapter and immutable V2 basis cache",
        "official_split_graph_counts": official_split_counts,
        "loaded_split_graph_counts": split_sizes,
        "loaded_graph_count": loaded_count,
        "actual_used_graph_count": used_count,
        "actual_used_fraction_of_loaded_graphs": observed(
            used_count / loaded_count, unit="fraction"
        ),
        "graphs_used_by_phase": {
            "optimization": split_sizes.get("train", 0),
            "validation_selection": split_sizes.get("validation", 0),
            "final_test_evaluation": split_sizes.get("test", 0),
        },
        "official_full_graph_count": observed(official_total, unit="graphs"),
        "loaded_fraction_of_official_full_dataset": loaded_fraction,
        "actual_used_fraction_of_official_full_dataset": observed(
            used_count / official_total, unit="fraction"
        ),
        "sampling_ratio": observed(1.0, unit="fraction"),
        "subset_or_fast_mode": False,
        "nodes_per_graph": _integer_distribution(node_counts),
        "canonical_undirected_edges_per_graph": _integer_distribution(edge_counts),
        "cycle_rank_per_graph": _integer_distribution(cycle_ranks),
        "cycle_memberships_per_graph": _integer_distribution(cycle_memberships),
        "cycle_basis_storage": "signed sparse COO; no E-by-cycle-rank dense allocation",
        "input_tensor_shapes": {
            "node_feature_widths": sorted({int(graph.x.shape[1]) for graph in graphs}),
            "edge_feature_widths": sorted({int(graph.edge_attr.shape[1]) for graph in graphs}),
            "target_widths": sorted({int(graph.y.numel()) for graph in graphs}),
            "node_and_edge_axes": "ragged by graph; min/max/mean are reported above",
        },
        "time_window": observed(
            None, reason="not applicable to static molecular graphs", unit="steps"
        ),
        "input_resolution": observed(
            None, reason="not applicable to categorical molecular graph inputs"
        ),
    }


def _cycle_batch_observability(
    args: argparse.Namespace,
    *,
    effective_batch_size: int,
    training_graphs: int,
    batches_per_epoch: int,
) -> dict[str, Any]:
    last_batch = training_graphs % effective_batch_size or effective_batch_size
    return {
        "batch_unit": "molecular_graphs",
        "requested_physical_batch_size": args.batch_size,
        "configured_physical_batch_size": effective_batch_size,
        "observed_smallest_batch_size": min(effective_batch_size, last_batch),
        "observed_largest_batch_size": min(effective_batch_size, training_graphs),
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "effective_batch_size": effective_batch_size,
        "effective_batch_size_formula": (
            f"{effective_batch_size} physical x 1 accumulation x 1 data-parallel worker"
        ),
        "training_batches_per_epoch": batches_per_epoch,
        "planned_maximum_training_batches": args.epochs * batches_per_epoch,
        "dataloader_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": args.prefetch_factor
        if args.workers
        else observed(
            None,
            reason="prefetch_factor is inactive because dataloader workers is zero",
        ),
        "pin_memory": True,
        "non_blocking_transfer": True,
        "cache": "immutable prepared graph and complete cycle-basis cache",
    }


def _graph_probe_cost(graph: Graph) -> int:
    return int(
        graph.x.numel()
        + graph.edge_attr.numel()
        + graph.cycle_basis._nnz() * 3
        + graph.cycle_lengths.numel()
        + graph.edge_cycle_counts.numel()
        + graph.edge_cycle_features.numel()
        + graph.cycle_position_values.numel()
        + graph.edge_index.numel()
    )


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _calibrate_batch_size(
    model: CycleBasisPEModel,
    dataset: str,
    train_graphs: list[Graph],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    """Probe the preregistered batch before epoch 1 and fail closed on OOM.

    No optimizer step is made and torch RNG is restored around every probe.  A
    batch size is never silently resized: an OOM requires an explicit new run
    configuration/run ID, so candidate protocols remain comparable.
    """
    requested = min(args.batch_size, len(train_graphs))
    if requested < 1:
        raise ValueError(f"{dataset}: official training split is empty")
    if args.hardware_profile != "a6000-48gb":
        return requested, {
            "policy": "fixed-portable",
            "requested_batch_size": args.batch_size,
            "effective_batch_size": requested,
            "attempts": [],
        }
    largest = sorted(train_graphs, key=_graph_probe_cost, reverse=True)
    candidate = requested
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    was_training = model.training
    if any(True for _ in model.named_buffers()):
        raise RuntimeError(
            "A6000 training-mode capacity probe requires a buffer-free model; review new buffers"
        )
    try:
        probe = collate(largest[:candidate]).to(device)
        precision = _amp_policy(args.amp, device)
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[cuda_index]):
            # The current architecture is asserted buffer-free, so training mode
            # includes dropout's true activation path without mutating model state.
            model.train()
            with torch.autocast("cuda", dtype=precision["dtype"], enabled=precision["enabled"]):
                prediction = model(probe)
                loss = (prediction.float() - probe.y).abs().mean()
            _require_finite_loss(loss, f"{dataset}: nonfinite capacity-probe loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if not _is_cuda_oom(error):
            raise
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"{dataset}: preregistered A6000 batch size {candidate} failed the "
            "pre-epoch OOM probe; choose an explicit smaller --batch-size and new run ID"
        ) from error
    finally:
        model.train(was_training)
    model.zero_grad(set_to_none=True)
    del probe, prediction, loss
    torch.cuda.empty_cache()
    return candidate, {
        "policy": "preregistered-worst-case-forward-backward-probe",
        "requested_batch_size": args.batch_size,
        "effective_batch_size": candidate,
        "attempts": [{"batch_size": candidate, "status": "passed", "peak_bytes": peak}],
        "automatic_backoff": False,
        "mid_epoch_backoff": False,
        "oom_policy": "clear cache, fail, and require an explicit new run configuration",
    }


@torch.inference_mode()
def evaluate(
    model: CycleBasisPEModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = False,
) -> float:
    model.eval()
    precision = _amp_policy(amp, device)
    total = torch.zeros((), device=device, dtype=torch.float64)
    all_finite = torch.ones((), device=device, dtype=torch.bool)
    count = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=precision["dtype"], enabled=precision["enabled"]):
            predicted = model(batch).float()
        all_finite.logical_and_(torch.isfinite(predicted).all())
        total += (predicted - batch.y).abs().sum().double()
        count += batch.y.numel()
    if count == 0:
        raise ValueError("cannot evaluate an empty official split")
    if not bool(all_finite):
        raise FloatingPointError("nonfinite validation/test prediction")
    return float(total / count)


@resource_failure_boundary
def _train_model(
    dataset: str,
    splits: dict[str, list[Graph]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Cycle PE v2 benchmark training requires CUDA; no CPU fallback")
    _seed(args.model_seed)
    device = torch.device(args.device)
    hardware = _hardware_report(args, device)
    resource_monitor = FailureSafeResourceMonitor(device, workload=f"cycle_v2_{dataset}_training")
    resource_start = resource_monitor.start()
    precision = _amp_policy(args.amp, device)
    model = CycleBasisPEModel(
        dataset=dataset,
        encoding=args.encoding,
        hidden=args.hidden_dim,
        pe_dim=args.pe_dim,
        layers=args.layers,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
        layer_scale=args.layer_scale,
    ).to(device)
    execution = configure_execution(model, args, device)
    execution.update(
        encoding=args.encoding,
        basis_execution="sparse_block_diagonal",
        basis_backend=args.basis_backend,
    )
    total_parameters = sum(p.numel() for p in model.parameters())
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters > args.max_parameters:
        raise ValueError(
            f"{dataset}/{_model_name(args)}: {parameters} parameters "
            f"exceeds budget {args.max_parameters}"
        )
    validation_only = bool(getattr(args, "validation_only", False))
    expected_splits = (
        {"train", "validation"} if validation_only else {"train", "validation", "test"}
    )
    if set(splits) != expected_splits:
        raise ValueError(f"unexpected benchmark splits: {sorted(splits)}")
    run = args.output_dir / dataset / _model_name(args)
    last_checkpoint = run / "last.pt"
    checkpoint = run / "best.pt"
    previous_checkpoint = run / "best.previous.pt"
    history_path = run / "history.json"
    if run.exists() and any(run.iterdir()) and not last_checkpoint.is_file():
        known = {checkpoint, previous_checkpoint, history_path}
        if not args.resume or any(path not in known for path in run.iterdir()):
            raise FileExistsError(f"non-resumable or mismatched existing model directory: {run}")
    run.mkdir(parents=True, exist_ok=True)
    resume_configuration = _resume_configuration(dataset, args)
    if last_checkpoint.is_file():
        preview = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(preview, dict) or preview.get("resume_configuration") != (
            resume_configuration
        ):
            raise ValueError("last.pt does not match the requested V2 implementation/configuration")
        effective_batch_size = preview.get("effective_batch_size")
        batch_calibration = preview.get("batch_calibration")
        if (
            isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, int)
            or not 1 <= effective_batch_size <= args.batch_size
            or not isinstance(batch_calibration, dict)
            or batch_calibration.get("effective_batch_size") != effective_batch_size
        ):
            raise ValueError("last.pt effective batch/calibration state is invalid")
    else:
        effective_batch_size, batch_calibration = _calibrate_batch_size(
            model, dataset, splits["train"], args, device
        )
    args.effective_batch_size = effective_batch_size
    train_loader = _loader(splits["train"], args, train=True)
    validation_loader = _loader(splits["validation"], args, train=False)
    test_loader = None if validation_only else _loader(splits["test"], args, train=False)
    data_observability = _cycle_data_observability(dataset, splits)
    batch_observability = _cycle_batch_observability(
        args,
        effective_batch_size=effective_batch_size,
        training_graphs=len(splits["train"]),
        batches_per_epoch=len(train_loader),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_ownership = _validate_optimizer_ownership(model, optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=25, min_lr=1e-6
    )
    # BF16 has FP32's exponent range and does not need loss scaling. Retain a
    # disabled scaler object so epoch-resume artifacts keep one stable schema.
    scaler = torch.amp.GradScaler("cuda", enabled=precision["gradient_scaler"])
    pre_run_observability = {
        "status": "pre_run_configuration",
        "model": {
            "name": _model_name(args),
            "layers": args.layers,
            "hidden_dimension": args.hidden_dim,
            "positional_encoding_dimension": args.pe_dim,
            "ffn_multiplier": args.ffn_multiplier,
            "attention_heads": observed(
                None, reason="the Cycle PE V2 architecture has no attention heads"
            ),
            "channels": args.hidden_dim,
            "total_parameters": total_parameters,
            "trainable_parameters": parameters,
        },
        "data": data_observability,
        "batching": batch_observability,
        "optimization": {
            "epochs_requested": args.epochs,
            "early_stopping_patience": args.patience,
            "planned_maximum_optimizer_steps": args.epochs * len(train_loader),
            "actual_optimizer_steps": observed(None, reason="training has not started"),
            "optimizer": "Adam",
            "optimizer_ownership": optimizer_ownership,
        },
        "precision": _precision_identity(precision),
        "hardware": hardware,
        "resources": resource_start,
        "modes": {
            "validation_only": validation_only,
            "debug": False,
            "subset": False,
            "fast_mode": False,
        },
    }
    print(json.dumps(pre_run_observability, sort_keys=True), flush=True)
    execution.update(
        hardware=hardware,
        precision={
            "amp": bool(args.amp),
            "amp_effective": precision["enabled"],
            "autocast_dtype": precision["dtype_name"],
            "fallback": precision["fallback"],
            "gradient_scaler": precision["gradient_scaler"],
            "cycle_sparse_aggregation": "float32",
            "backbone_autocast": precision["enabled"],
        },
        data_pipeline={
            "requested_batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor if args.workers else None,
            "persistent_workers": args.workers > 0,
            "pin_memory": True,
            "cycle_membership_layout": "sparse_coo_block_diagonal",
            "cycle_factorization_in_forward": False,
            "batch_calibration": batch_calibration,
        },
    )
    history = []
    optimizer_steps = 0
    first_task_gradient_connectivity: dict[str, Any] | None = None
    best = math.inf
    best_epoch = 0
    start_epoch = 1
    elapsed_before = 0.0
    previous_peak = 0
    previous_reserved_peak = 0
    if last_checkpoint.is_file():
        state = torch.load(last_checkpoint, map_location=device, weights_only=True)
        if not isinstance(state, dict) or state.get("resume_configuration") != resume_configuration:
            raise ValueError("last.pt does not match the requested V2 implementation/configuration")
        saved_history = state.get("history")
        last_epoch = state.get("epoch")
        if (
            not isinstance(saved_history, list)
            or isinstance(last_epoch, bool)
            or not isinstance(last_epoch, int)
            or last_epoch != len(saved_history)
            or last_epoch < 1
        ):
            raise ValueError("last.pt epoch/history state is invalid")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if _validate_optimizer_ownership(model, optimizer) != optimizer_ownership:
            raise ValueError("last.pt changed Cycle PE V2 optimizer parameter ownership")
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        history = saved_history
        saved_optimizer_steps = state.get("optimizer_steps")
        expected_optimizer_steps = len(history) * len(train_loader)
        if (
            isinstance(saved_optimizer_steps, bool)
            or not isinstance(saved_optimizer_steps, int)
            or saved_optimizer_steps != expected_optimizer_steps
        ):
            raise ValueError("last.pt optimizer-step count is invalid")
        optimizer_steps = saved_optimizer_steps
        saved_gradient_connectivity = state.get("first_task_gradient_connectivity")
        if (
            not isinstance(saved_gradient_connectivity, dict)
            or saved_gradient_connectivity.get("validated") is not True
            or saved_gradient_connectivity.get("stage")
            != "after_first_task_backward_before_gradient_clipping_and_optimizer_step"
            or saved_gradient_connectivity.get("trainable_parameter_tensors")
            != optimizer_ownership["trainable_parameter_tensors"]
            or saved_gradient_connectivity.get("trainable_parameter_scalars")
            != optimizer_ownership["trainable_parameter_scalars"]
            or saved_gradient_connectivity.get("missing_gradient_parameters") != []
            or saved_gradient_connectivity.get("nonfinite_gradient_parameters") != []
        ):
            raise ValueError("last.pt first-task gradient-connectivity state is invalid")
        first_task_gradient_connectivity = saved_gradient_connectivity
        best, best_epoch = float(state["best"]), int(state["best_epoch"])
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        previous_peak = int(state.get("peak_gpu_memory_bytes", 0))
        previous_reserved_peak = int(state.get("peak_gpu_reserved_bytes", 0))
        expected_best_hash = state.get("best_checkpoint_sha256")
        _recover_best_checkpoint(checkpoint, previous_checkpoint, expected_best_hash)
        atomic_write_json(history_path, history)
        _restore_rng(state["rng_state"], train_loader)
        start_epoch = args.epochs + 1 if state.get("complete") is True else last_epoch + 1
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch - 1 - best_epoch >= args.patience:
            break
        epoch_started = time.perf_counter()
        model.train()
        train_sum = torch.zeros((), device=device, dtype=torch.float64)
        train_count = 0
        train_graph_count = 0
        epoch_optimizer_steps = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=precision["dtype"], enabled=precision["enabled"]):
                predicted = model(batch)
                loss = (predicted.float() - batch.y).abs().mean()
            _require_finite_loss(loss, f"{dataset}/{_model_name(args)}: nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if first_task_gradient_connectivity is None:
                first_task_gradient_connectivity = _validate_first_task_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            epoch_optimizer_steps += 1
            train_sum += loss.detach().double() * batch.y.numel()
            train_count += batch.y.numel()
            train_graph_count += batch.y.shape[0]
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        validation = evaluate(model, validation_loader, device, amp=args.amp)
        train_mae = float(train_sum / train_count)
        if not math.isfinite(train_mae):
            raise FloatingPointError(
                f"{dataset}/{_model_name(args)}: nonfinite epoch training loss"
            )
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        scheduler.step(validation)
        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "validation_mae": validation,
                "epoch_seconds": epoch_seconds,
                "train_cuda_synchronized_seconds": train_seconds,
                "train_graphs": train_graph_count,
                "train_graphs_per_second": train_graph_count / train_seconds,
                "optimizer_steps_this_epoch": epoch_optimizer_steps,
                "optimizer_steps": optimizer_steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        atomic_write_json(history_path, history)
        if validation < best:
            best, best_epoch = validation, epoch
            payload = {
                "state_dict": model.state_dict(),
                "implementation_sha256": resume_configuration["implementation_sha256"],
                "epoch": epoch,
                "validation_mae": validation,
                "dataset": dataset,
                "model": _model_name(args),
                "model_seed": args.model_seed,
                "effective_batch_size": effective_batch_size,
                "batch_calibration": batch_calibration,
                "hardware": hardware,
                "precision": _precision_identity(precision),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }
            if checkpoint.is_file():
                previous_bytes = checkpoint.read_bytes()
                atomic_publish(
                    previous_checkpoint,
                    lambda path, payload=previous_bytes: path.write_bytes(payload),
                )
            atomic_publish(checkpoint, lambda path, state=payload: torch.save(state, path))
        elapsed_so_far = elapsed_before + time.perf_counter() - started
        peak_so_far = max(previous_peak, torch.cuda.max_memory_allocated(device))
        reserved_peak_so_far = max(previous_reserved_peak, torch.cuda.max_memory_reserved(device))
        last_state = {
            "schema_version": 1,
            "resume_configuration": resume_configuration,
            "complete": False,
            "epoch": epoch,
            "history": history,
            "best": best,
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": _capture_rng(train_loader),
            "elapsed_seconds": elapsed_so_far,
            "peak_gpu_memory_bytes": peak_so_far,
            "peak_gpu_reserved_bytes": reserved_peak_so_far,
            "optimizer_steps": optimizer_steps,
            "first_task_gradient_connectivity": first_task_gradient_connectivity,
            "effective_batch_size": effective_batch_size,
            "batch_calibration": batch_calibration,
            "hardware": hardware,
        }
        atomic_publish(last_checkpoint, lambda path, state=last_state: torch.save(state, path))
        print(
            f"{dataset}/{_model_name(args)} epoch={epoch} train_mae={train_mae:.6f} "
            f"validation_mae={validation:.6f} best={best:.6f} seconds={epoch_seconds:.2f}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            break
    completed_state = torch.load(last_checkpoint, map_location=device, weights_only=True)
    completed_state["complete"] = True
    atomic_publish(last_checkpoint, lambda path, state=completed_state: torch.save(state, path))
    test = None
    if not validation_only:
        selected = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(selected["state_dict"], strict=True)
        # Standard runner: test is touched once after validation selects the checkpoint.
        assert test_loader is not None
        test = evaluate(model, test_loader, device, amp=args.amp)
    torch.cuda.synchronize(device)
    peak_gpu_memory_bytes = max(previous_peak, torch.cuda.max_memory_allocated(device))
    peak_gpu_reserved_bytes = max(previous_reserved_peak, torch.cuda.max_memory_reserved(device))
    if first_task_gradient_connectivity is None:
        raise RuntimeError("training completed without first-task gradient-connectivity validation")
    resource_observability = resource_monitor.finish(
        peak_allocated_bytes=peak_gpu_memory_bytes,
        peak_reserved_bytes=peak_gpu_reserved_bytes,
    )
    resource_observability["cuda_allocator_peak_boundary"] = (
        "reset after the non-optimizing batch-capacity probe; cumulative maximum is "
        "restored from last.pt across resumed invocations"
    )
    optimization_observability = {
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": args.patience,
        "training_batches_per_epoch": len(train_loader),
        "planned_maximum_optimizer_steps": args.epochs * len(train_loader),
        "actual_optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": 1,
        "optimizer": "Adam",
        "optimizer_ownership": optimizer_ownership,
        "first_task_gradient_connectivity": first_task_gradient_connectivity,
        "step_definition": "one Adam update for every complete physical minibatch",
        "shortfall_reason": (
            None
            if optimizer_steps == args.epochs * len(train_loader)
            else "validation-based early stopping ended training before the maximum epoch budget"
        ),
    }
    train_graphs = sum(int(row["train_graphs"]) for row in history)
    train_seconds = sum(float(row["train_cuda_synchronized_seconds"]) for row in history)
    throughput = {
        "scope": (
            "CUDA-synchronized complete training loop: DataLoader wait, sparse collate, "
            "H2D, forward, backward, gradient clip and Adam included; validation, "
            "initial dataset preparation and checkpoint IO excluded"
        ),
        "training_graphs": train_graphs,
        "training_cuda_synchronized_seconds": train_seconds,
        "training_graphs_per_second": observed(
            train_graphs / train_seconds,
            reason=None if train_seconds > 0 else "observed training duration was zero",
            unit="graphs_per_second",
        )
        if train_seconds > 0
        else observed(
            None,
            reason="observed training duration was zero",
            unit="graphs_per_second",
        ),
    }
    result = {
        "validation": best,
        "best_epoch": best_epoch,
        "total_parameters": total_parameters,
        "trainable_parameters": parameters,
        "frozen_parameters": total_parameters - parameters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "history": str(history_path),
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_reserved_bytes": peak_gpu_reserved_bytes,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "optimizer_ownership": optimizer_ownership,
        "first_task_gradient_connectivity": first_task_gradient_connectivity,
        "last_checkpoint": str(last_checkpoint),
        "last_checkpoint_sha256": hashlib.sha256(last_checkpoint.read_bytes()).hexdigest(),
        "execution": execution,
        "effective_batch_size": effective_batch_size,
        "batch_calibration": batch_calibration,
        "hardware": hardware,
        "pre_run_observability": pre_run_observability,
        "resource_observability": resource_observability,
        "data_observability": data_observability,
        "batch_observability": batch_observability,
        "optimization_observability": optimization_observability,
        "throughput": throughput,
        "training_graphs_per_epoch": len(splits["train"]),
        "epoch_timing": "cuda_synchronized_train_and_validation_excluding_checkpoint_io",
        "evaluation_splits": ["train", "validation"],
        "fresh_training": True,
    }
    if test is not None:
        result["test"] = test
        result["evaluation_splits"].append("test")
    print(
        json.dumps(
            {
                "status": "post_run_observability",
                "dataset": dataset,
                "model": _model_name(args),
                "optimizer_steps": optimizer_steps,
                "epochs_completed": len(history),
                "optimizer_ownership_validated": optimizer_ownership["validated"],
                "gradient_connectivity_validated": first_task_gradient_connectivity["validated"],
                "resource_summary": resource_observability["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


@resource_failure_boundary
def _evaluate_test_checkpoint(
    dataset: str,
    test_graphs: list[Graph],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate one validation-selected checkpoint without creating training state."""
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Cycle PE v2 test evaluation requires CUDA; no CPU fallback")
    checkpoint = args.test_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint}")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    device = torch.device(args.device)
    hardware = _hardware_report(args, device)
    precision = _amp_policy(args.amp, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Selected checkpoint has an invalid payload schema")
    expected_metadata = {
        "dataset": dataset,
        "model": _model_name(args),
        "model_seed": args.model_seed,
    }
    for name, expected in expected_metadata.items():
        if payload.get(name) != expected:
            raise ValueError(f"Selected checkpoint {name} mismatch")
    if payload.get("precision") != _precision_identity(precision):
        raise ValueError("Selected checkpoint precision policy mismatch")
    saved_arguments = payload.get("arguments")
    if not isinstance(saved_arguments, dict) or saved_arguments.get("validation_only") is not True:
        raise ValueError("Selected checkpoint was not produced by validation-only training")
    for name in (
        "hidden_dim",
        "pe_dim",
        "layers",
        "basis_backend",
        "encoding",
        "ffn_multiplier",
        "dropout",
        "layer_scale",
    ):
        if saved_arguments.get(name) != getattr(args, name):
            raise ValueError(f"Selected checkpoint architecture mismatch for {name}")
    if payload.get("implementation_sha256") != implementation_hashes():
        raise ValueError("Selected checkpoint implementation/source mismatch")
    validation = payload.get("validation_mae")
    epoch = payload.get("epoch")
    effective_batch_size = payload.get("effective_batch_size")
    if (
        isinstance(validation, bool)
        or not isinstance(validation, (int, float))
        or not math.isfinite(float(validation))
        or float(validation) < 0
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or isinstance(effective_batch_size, bool)
        or not isinstance(effective_batch_size, int)
        or not 1 <= effective_batch_size <= args.batch_size
    ):
        raise ValueError("Selected checkpoint validation metadata is invalid")
    args.effective_batch_size = effective_batch_size
    _seed(args.model_seed)
    model = CycleBasisPEModel(
        dataset=dataset,
        encoding=args.encoding,
        hidden=args.hidden_dim,
        pe_dim=args.pe_dim,
        layers=args.layers,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
        layer_scale=args.layer_scale,
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters > args.max_parameters:
        raise ValueError(
            f"{dataset}/{_model_name(args)}: {parameters} parameters "
            f"exceeds budget {args.max_parameters}"
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    execution = configure_execution(model, args, device)
    execution.update(
        encoding=args.encoding,
        basis_execution="sparse_block_diagonal",
        basis_backend=args.basis_backend,
        hardware=hardware,
        precision={
            "amp": bool(args.amp),
            "amp_effective": precision["enabled"],
            "autocast_dtype": precision["dtype_name"],
            "fallback": precision["fallback"],
            "gradient_scaler": precision["gradient_scaler"],
            "cycle_sparse_aggregation": "float32",
            "backbone_autocast": precision["enabled"],
        },
        data_pipeline={
            "requested_batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor if args.workers else None,
            "persistent_workers": args.workers > 0,
            "pin_memory": True,
            "cycle_membership_layout": "sparse_coo_block_diagonal",
            "cycle_factorization_in_forward": False,
        },
    )
    test_loader = _loader(test_graphs, args, train=False)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    resource_monitor = FailureSafeResourceMonitor(
        device, workload=f"cycle_v2_{dataset}_selected_test"
    )
    resources_at_start = resource_monitor.start()
    data_observability = _cycle_data_observability(dataset, {"test": test_graphs})
    batch_observability = {
        "batch_unit": "molecular_graphs",
        "requested_physical_batch_size": args.batch_size,
        "checkpoint_bound_physical_batch_size": effective_batch_size,
        "observed_largest_physical_batch_size": min(effective_batch_size, len(test_graphs)),
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "optimizer_steps": observed(
            None, reason="selected-checkpoint test evaluation is read-only", unit="steps"
        ),
        "dataloader_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": (
            args.prefetch_factor
            if args.workers > 0
            else observed(
                None,
                reason="prefetch_factor is inactive because DataLoader workers is zero",
            )
        ),
        "pin_memory": True,
        "non_blocking_transfer": True,
        "cycle_membership_layout": "sparse_coo_block_diagonal",
        "cycle_factorization_in_forward": False,
    }
    print(
        json.dumps(
            {
                "status": "pre_selected_test_observability",
                "dataset": dataset,
                "model": {
                    "name": _model_name(args),
                    "layers": args.layers,
                    "hidden_dimension": args.hidden_dim,
                    "positional_encoding_dimension": args.pe_dim,
                    "ffn_multiplier": args.ffn_multiplier,
                    "channels": args.hidden_dim,
                    "attention_heads": observed(
                        None,
                        reason="the Cycle PE V2 architecture has no attention heads",
                    ),
                    "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "trainable_parameters": parameters,
                },
                "data": data_observability,
                "batching": batch_observability,
                "precision": _precision_identity(precision),
                "hardware": hardware,
                "resources": resources_at_start,
                "optimization": {
                    "optimizer_created": False,
                    "optimizer_steps": 0,
                    "reason": "selected-checkpoint test evaluation is read-only",
                },
                "modes": {
                    "test_only": True,
                    "debug": False,
                    "subset": False,
                    "fast_mode": False,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    test = evaluate(model, test_loader, device, amp=args.amp)
    torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    resource_observability = resource_monitor.finish(
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    throughput = {
        "scope": (
            "CUDA-synchronized selected-checkpoint test evaluation; checkpoint loading "
            "and model construction excluded"
        ),
        "evaluated_graphs": len(test_graphs),
        "evaluation_seconds": evaluation_seconds,
        "evaluation_graphs_per_second": (
            observed(len(test_graphs) / evaluation_seconds, unit="graphs_per_second")
            if evaluation_seconds > 0
            else observed(
                None,
                reason="observed selected-test evaluation duration was zero",
                unit="graphs_per_second",
            )
        ),
    }
    print(
        json.dumps(
            {
                "status": "post_selected_test_observability",
                "dataset": dataset,
                "model": _model_name(args),
                "throughput": throughput,
                "resource_summary": resource_observability["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "test": test,
        "selected_validation": float(validation),
        "selected_epoch": epoch,
        "trainable_parameters": parameters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_seconds": evaluation_seconds,
        "peak_gpu_memory_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "execution": execution,
        "effective_batch_size": effective_batch_size,
        "hardware": hardware,
        "data_observability": data_observability,
        "batch_observability": batch_observability,
        "resources_at_start": resources_at_start,
        "resource_observability": resource_observability,
        "throughput": throughput,
        "optimizer_observability": {
            "optimizer_created": False,
            "optimizer_steps": 0,
            "reason": "selected-checkpoint test evaluation is read-only",
        },
        "evaluation_splits": ["test"],
        "fresh_training": False,
    }


def _completed_training_dataset(entry: Any, dataset: str, args: argparse.Namespace) -> bool:
    """Fail closed while deciding whether a completed dataset may be skipped."""
    try:
        if not isinstance(entry, dict) or entry.get("metric") != "mae":
            return False
        models = entry.get("models")
        if not isinstance(models, dict) or set(models) != {_model_name(args)}:
            return False
        result = models[_model_name(args)]
        run = (args.output_dir / dataset / _model_name(args)).resolve()
        bindings = (
            ("checkpoint", "checkpoint_sha256", run / "best.pt"),
            ("history", "history_sha256", run / "history.json"),
            ("last_checkpoint", "last_checkpoint_sha256", run / "last.pt"),
        )
        for path_key, hash_key, expected in bindings:
            path = Path(result[path_key]).resolve()
            if path != expected or not _artifact_matches(str(path), result[hash_key]):
                return False
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        return bool(
            isinstance(last, dict)
            and last.get("complete") is True
            and last.get("resume_configuration") == _resume_configuration(dataset, args)
            and result.get("fresh_training") is True
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _validate(args)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.test_checkpoint is not None:
        args.test_checkpoint = args.test_checkpoint.expanduser().resolve()
    output_was_nonempty = args.output_dir.exists() and any(args.output_dir.iterdir())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if output_was_nonempty and not manifest_path.is_file():
        raise FileExistsError(f"nonempty V2 output has no resumable manifest: {args.output_dir}")
    arguments = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for library in ("torch-geometric", "numpy", "networkx"):
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            versions[library] = "not_installed"
    run_mode = (
        "test_only"
        if args.test_checkpoint is not None
        else "validation_only"
        if args.validation_only
        else "prepare_only"
        if args.prepare_only
        else "standard"
    )
    manifest = {
        "schema_version": 2,
        "track": TRACK_NAME,
        "version": "v2",
        "suite": "benchmark",
        "status": "running",
        "protocol": "ours_only_on_official_benchmark_splits",
        "run_mode": run_mode,
        "arguments": arguments,
        "software": versions,
        "architecture": architecture_protocol(args.encoding),
        "implementation_sha256": implementation_hashes(),
        "seeds": {
            "model_seed": args.model_seed,
            "data_seed": "unused: fixed official graphs",
            "split_seed": "unused: official splits",
            "chart_seed": (
                "unused: deterministic DFS on canonical stored edge order; tree-dependent encoding"
            ),
        },
        "controls": {
            "encoding": args.encoding,
            "encoding_comparison": (
                "identical SE/backbone/parameterization; pe adds a parameter-free "
                "cycle-relative cosine-kernel residual, not a pure-PE/no-SE ablation"
            ),
            "model": _model_name(args),
            "external_models_trained": False,
            "test_checkpoint_selection": False,
            "parameter_budget": args.max_parameters,
            "target_policy": "official labels unchanged",
            "basis_input": (
                "all signed sparse DFS fundamental cycles, with no truncation; "
                "orientation-free membership drives learned edge-to-cycle-to-edge messages"
            ),
            "basis_backend": args.basis_backend,
            "basis_backend_runtime": (
                "DFS O(V+E) plus explicit output O(nnz(Z)); batched sparse aggregation "
                "O(nnz(Z)*PE_width), with no QR/SVD/Gram inverse or dense cycle matrix"
            ),
            "basis_rank_dependent_parameters": False,
            "basis_selection_dependence": "selected DFS tree; not invariant to arbitrary ZR",
            "epoch_resume": "atomic last.pt plus best.pt/best.previous.pt two-slot recovery",
            "resume_determinism": (
                "exact configuration/source/artifact binding and epoch-boundary model, "
                "optimizer, scheduler, scaler, history, RNG, and loader-state restoration; "
                "bitwise equivalence "
                "is not promised for nondeterministic CUDA kernels"
            ),
            "test_data_access": run_mode in {"standard", "test_only"},
            "fresh_training": run_mode in {"standard", "validation_only"},
            "optimizer_created": run_mode in {"standard", "validation_only"},
            "hardware_profile_optimization_identity": (
                "batch size changes optimizer-step count and trajectory; compare scores only "
                "within an identical hardware profile and batch contract"
            ),
            "automatic_oom_backoff": False,
        },
    }
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "track": TRACK_NAME,
        "version": "v2",
        "suite": "benchmark",
        "status": "running",
        "model_seed": args.model_seed,
        "run_mode": run_mode,
        "datasets": {},
    }
    resume_identity = {
        "schema": "cycle-dfs-se-relative-pe-v2-multidataset-resume-1",
        "run_mode": run_mode,
        "arguments": {key: value for key, value in arguments.items() if key != "resume"},
        "implementation_sha256": manifest["implementation_sha256"],
    }
    manifest["resume_identity"] = resume_identity
    metrics_path = args.output_dir / "metrics.json"
    if manifest_path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"V2 run already exists and --no-resume was requested: {args.output_dir}"
            )
        if not metrics_path.is_file():
            raise ValueError("resumable V2 manifest has no metrics document")
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "schema_version",
            "track",
            "version",
            "suite",
            "protocol",
            "run_mode",
            "arguments",
            "software",
            "architecture",
            "implementation_sha256",
            "seeds",
            "controls",
            "resume_identity",
        )
        if (
            not isinstance(stored_manifest, dict)
            or stored_manifest.get("status") not in {"running", "failed", "passed"}
            or any(stored_manifest.get(key) != manifest.get(key) for key in immutable_keys)
            or not isinstance(stored_metrics, dict)
            or stored_metrics.get("track") != TRACK_NAME
            or stored_metrics.get("version") != "v2"
            or stored_metrics.get("suite") != "benchmark"
            or stored_metrics.get("model_seed") != args.model_seed
            or stored_metrics.get("run_mode") != run_mode
            or not isinstance(stored_metrics.get("datasets"), dict)
            or not set(stored_metrics["datasets"]).issubset(set(args.datasets))
        ):
            raise ValueError("existing V2 output does not match this implementation/configuration")
        manifest, metrics = stored_manifest, stored_metrics
        manifest["status"] = metrics["status"] = "running"
        manifest.pop("error", None)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(metrics_path, metrics)
    try:
        for dataset in args.datasets:
            existing_dataset = metrics["datasets"].get(dataset)
            if (
                existing_dataset is not None
                and not args.prepare_only
                and args.test_checkpoint is None
            ):
                existing_models = (
                    existing_dataset.get("models") if isinstance(existing_dataset, dict) else None
                )
                if isinstance(existing_models, dict) and existing_models:
                    if _completed_training_dataset(existing_dataset, dataset, args):
                        continue
                    raise ValueError(
                        f"{dataset}: existing completed artifacts failed resume validation"
                    )
            started = time.perf_counter()
            requested_splits = (
                ("test",)
                if args.test_checkpoint is not None
                else ("train", "validation")
                if args.validation_only
                else None
            )
            if requested_splits is None:
                splits, protocol = load_benchmark(
                    args.data_root,
                    dataset,
                    allow_download=args.allow_download,
                    basis_backend=args.basis_backend,
                    workers=args.workers,
                )
            else:
                splits, protocol = load_benchmark(
                    args.data_root,
                    dataset,
                    allow_download=args.allow_download,
                    splits=requested_splits,
                    basis_backend=args.basis_backend,
                    workers=args.workers,
                )
            dataset_metrics: dict[str, Any] = {
                "metric": "mae",
                "protocol": protocol,
                "models": {},
                "data_preparation_seconds": time.perf_counter() - started,
            }
            metrics["datasets"][dataset] = dataset_metrics
            if args.test_checkpoint is not None:
                dataset_metrics["models"][_model_name(args)] = _evaluate_test_checkpoint(
                    dataset, splits["test"], args
                )
                atomic_write_json(args.output_dir / "metrics.json", metrics)
            elif not args.prepare_only:
                dataset_metrics["models"][_model_name(args)] = _train_model(dataset, splits, args)
            atomic_write_json(metrics_path, metrics)
            del splits
        metrics["status"] = manifest["status"] = "prepared" if args.prepare_only else "passed"
        atomic_write_json(metrics_path, metrics)
        manifest["dataset_protocols"] = {
            name: data["protocol"] for name, data in metrics["datasets"].items()
        }
        if not args.prepare_only:
            manifest["execution_by_dataset"] = {
                name: data["models"][_model_name(args)].get("execution")
                for name, data in metrics["datasets"].items()
            }
        atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = metrics["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["resource_failure_observations"] = resource_failure_observations(exc)
        persist_failure_artifacts(
            exc,
            (
                ("manifest.json", lambda: atomic_write_json(manifest_path, manifest)),
                ("metrics.json", lambda: atomic_write_json(metrics_path, metrics)),
            ),
        )
        raise
    print(json.dumps({"status": metrics["status"], "output_dir": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
