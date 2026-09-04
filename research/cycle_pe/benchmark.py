"""Train only our cycle-set PE on official molecular benchmark splits.

Other papers' model results belong in an external comparison table, not this run.
Actual training requires CUDA. Preparation never trains or generates substitutes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
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
from research.cycle_pe.benchmark_data import (
    DATASETS,
    EXPECTED_SIZES,
    SOURCES,
    SPLITS,
    Graph,
    collate,
    load_benchmark,
)
from research.cycle_pe.benchmark_models import MODEL_NAME, CyclePEModel, architecture_protocol
from research.cycle_pe.resource_monitor import (
    FailureSafeResourceMonitor,
    persist_failure_artifacts,
    resource_failure_boundary,
    resource_failure_observations,
)

TRACK_NAME = "cycle_pe"
IMPLEMENTATION_FILES = (
    "research/cycle_pe/benchmark.py",
    "research/cycle_pe/benchmark_data.py",
    "research/cycle_pe/benchmark_models.py",
    "research/cycle_pe/features.py",
    "research/cycle_pe/paper_model.py",
    "research/cycle_pe/resource_monitor.py",
    "src/chartgat/algebra.py",
    "src/chartgat/cache.py",
    "src/chartgat/execution.py",
    "src/chartgat/observability.py",
)


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in IMPLEMENTATION_FILES
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", choices=("benchmark",), default="benchmark")
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    result.add_argument("--data-root", type=Path, default=Path("data/paper"))
    result.add_argument("--output-dir", type=Path, default=Path("results/cycle_pe/benchmark"))
    result.add_argument("--device", default="cuda")
    for seed in ("data", "split", "chart", "model"):
        result.add_argument(f"--{seed}-seed", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--prefetch-factor", type=int, default=2)
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--epochs", type=int, default=300)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--hidden-dim", type=int, default=64)
    result.add_argument("--pe-dim", type=int, default=32)
    result.add_argument("--layers", type=int, default=3)
    result.add_argument("--max-parameters", type=int, default=500_000)
    result.add_argument("--validation-only", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--test-checkpoint", type=Path, help=argparse.SUPPRESS)
    add_execution_arguments(result)
    return result


def _validate(args: argparse.Namespace) -> None:
    if any(getattr(args, f"{seed}_seed") < 0 for seed in ("data", "split", "chart", "model")):
        raise ValueError("seeds must be nonnegative")
    for key in (
        "batch_size",
        "epochs",
        "patience",
        "hidden_dim",
        "pe_dim",
        "layers",
        "max_parameters",
        "prefetch_factor",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
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
        raise RuntimeError("Cycle PE benchmark training requires CUDA; no CPU fallback")


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
    # Keep data ordering independent of model RNG consumption.
    generator = torch.Generator().manual_seed(args.model_seed)
    options: dict[str, Any] = {
        "dataset": graphs,
        "batch_size": args.batch_size,
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
    return DataLoader(**options)


def _distribution(values: list[int], *, unit: str) -> dict[str, Any]:
    if not values:
        reason = "no loaded graphs were available for this distribution"
        return {
            name: observed(None, reason=reason, unit=unit)
            for name in ("minimum", "mean", "median", "maximum")
        }
    ordered = sorted(values)
    return {
        "minimum": observed(ordered[0], unit=unit),
        "mean": observed(sum(ordered) / len(ordered), unit=unit),
        "median": observed(float(np.median(ordered)), unit=unit),
        "maximum": observed(ordered[-1], unit=unit),
    }


def _data_observability(
    dataset: str, splits: dict[str, list[Graph]]
) -> dict[str, Any]:
    expected = dict(zip(SPLITS, EXPECTED_SIZES[dataset], strict=True))
    loaded_counts = {name: len(graphs) for name, graphs in splits.items()}
    unknown = set(loaded_counts) - set(expected)
    if unknown:
        raise ValueError(f"unexpected benchmark splits: {sorted(unknown)}")
    graphs = [graph for name in SPLITS for graph in splits.get(name, [])]
    for graph in graphs:
        if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
            raise ValueError("cycle benchmark edge_index must have shape [2, edges]")
        if graph.x.ndim != 2 or graph.edge_attr.ndim != 2 or graph.cycle_set.ndim != 2:
            raise ValueError("cycle benchmark features must be rank-two tensors")
        if graph.x.shape[0] != graph.cycle_set.shape[0]:
            raise ValueError("cycle PE rows must match the graph node count")
        if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
            raise ValueError("edge feature rows must match stored edge count")
    loaded_total = sum(loaded_counts.values())
    official_total = sum(expected.values())
    first = graphs[0] if graphs else None
    shape_reason = None if first is not None else "no graph was loaded"
    return {
        "dataset": dataset,
        "source": SOURCES[dataset],
        "official_split_counts": expected,
        "loaded_split_counts": {
            name: {
                "value": loaded_counts.get(name),
                "reason": (
                    None
                    if name in loaded_counts
                    else "split deliberately not loaded in this isolated run mode"
                ),
                "unit": "graphs",
            }
            for name in SPLITS
        },
        "loaded_graph_fraction_of_official_dataset": observed(
            loaded_total / official_total, unit="fraction"
        ),
        "actual_used_graph_count": observed(loaded_total, unit="graphs"),
        "actual_used_fraction_of_official_dataset": observed(
            loaded_total / official_total, unit="fraction"
        ),
        "official_graph_count": observed(official_total, unit="graphs"),
        "loaded_graph_count": observed(loaded_total, unit="graphs"),
        "graph_statistics": {
            "nodes_per_graph": _distribution(
                [int(graph.x.shape[0]) for graph in graphs], unit="nodes"
            ),
            "stored_edges_per_graph": _distribution(
                [int(graph.edge_index.shape[1]) for graph in graphs], unit="edges"
            ),
            "targets_per_graph": _distribution(
                [int(graph.y.numel()) for graph in graphs], unit="targets"
            ),
        },
        "input_shape_contract": {
            "node_feature_width": observed(
                None if first is None else int(first.x.shape[1]),
                reason=shape_reason,
                unit="features",
            ),
            "edge_feature_width": observed(
                None if first is None else int(first.edge_attr.shape[1]),
                reason=shape_reason,
                unit="features",
            ),
            "cycle_pe_width": observed(
                None if first is None else int(first.cycle_set.shape[1]),
                reason=shape_reason,
                unit="features",
            ),
            "target_width": observed(
                None if first is None else int(first.y.numel()),
                reason=shape_reason,
                unit="targets",
            ),
        },
        "spatial_resolution": observed(
            None,
            reason="not applicable to graph benchmark samples",
            unit="not_applicable",
        ),
        "temporal_window": observed(
            None,
            reason="not applicable to static molecular graphs",
            unit="not_applicable",
        ),
    }


def _batch_observability(
    args: argparse.Namespace, splits: dict[str, list[Graph]]
) -> dict[str, Any]:
    train_count = len(splits.get("train", []))
    steps_per_epoch = math.ceil(train_count / args.batch_size) if train_count else 0
    return {
        "requested_physical_graphs_per_batch": args.batch_size,
        "maximum_effective_graphs_per_training_batch": min(args.batch_size, train_count),
        "gradient_accumulation_steps": 1,
        "effective_batch_size": min(args.batch_size, train_count),
        "training_steps_per_epoch": steps_per_epoch,
        "workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": (
            args.prefetch_factor
            if args.workers > 0
            else observed(
                None,
                reason="DataLoader prefetch is inactive because workers is zero",
            )
        ),
        "non_blocking_transfer": True,
        "distributed_training": False,
        "cache": "immutable prepared official graph cache reused across epochs",
        "batch_candidate_throughput_sweep": observed(
            None,
            reason=(
                "not run automatically because changing the user-selected batch size would "
                "change the optimization recipe; run a separate GPU profiling run before "
                "the final experiment"
            ),
            unit="samples_per_second",
        ),
    }


def _validate_first_step_gradients(model: torch.nn.Module) -> None:
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
            "first optimizer step did not connect every trainable parameter to a finite "
            f"gradient; missing={missing}, nonfinite={nonfinite}"
        )


def _require_finite_loss(loss: torch.Tensor, label: str) -> None:
    predicate = torch.isfinite(loss)
    assertion = getattr(torch, "_assert_async", None)
    if loss.device.type == "cuda" and assertion is not None:
        assertion(predicate, label)
    elif not bool(predicate):
        raise FloatingPointError(label)


@torch.no_grad()
def evaluate(model: CyclePEModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = torch.zeros((), device=device, dtype=torch.float64)
    all_finite = torch.ones((), device=device, dtype=torch.bool)
    count = 0
    for batch in loader:
        batch = batch.to(device)
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
        raise RuntimeError("Cycle PE benchmark training requires CUDA; no CPU fallback")
    _seed(args.model_seed)
    device = torch.device(args.device)
    model = CyclePEModel(
        dataset=dataset,
        hidden=args.hidden_dim,
        pe_dim=args.pe_dim,
        layers=args.layers,
    ).to(device)
    execution = configure_execution(model, args, device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters > args.max_parameters:
        raise ValueError(
            f"{dataset}/{MODEL_NAME}: {parameters} parameters exceeds budget {args.max_parameters}"
        )
    train_loader = _loader(splits["train"], args, train=True)
    validation_loader = _loader(splits["validation"], args, train=False)
    validation_only = bool(getattr(args, "validation_only", False))
    expected_splits = (
        {"train", "validation"} if validation_only else {"train", "validation", "test"}
    )
    if set(splits) != expected_splits:
        raise ValueError(f"unexpected benchmark splits: {sorted(splits)}")
    test_loader = None if validation_only else _loader(splits["test"], args, train=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    if trainable_ids != optimizer_ids:
        raise RuntimeError("optimizer parameter ownership does not match the trainable model")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=25, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    run = args.output_dir / dataset / MODEL_NAME
    run.mkdir(parents=True, exist_ok=False)
    checkpoint = run / "best.pt"
    history_path = run / "history.json"
    history = []
    best = math.inf
    best_epoch = 0
    optimizer_steps = 0
    data_observability = _data_observability(dataset, splits)
    batch_observability = _batch_observability(args, splits)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    resource_monitor = FailureSafeResourceMonitor(
        device, workload=f"cycle_v1_{dataset}_training"
    )
    resource_start = resource_monitor.start()
    print(
        json.dumps(
            {
                "kind": "pre_run_observability",
                "dataset": dataset,
                "model": {
                    "name": MODEL_NAME,
                    "hidden_dimension": args.hidden_dim,
                    "positional_encoding_dimension": args.pe_dim,
                    "message_passing_layers": args.layers,
                    "channels": args.hidden_dim,
                    "attention_heads": observed(
                        None, reason="the Cycle PE V1 architecture has no attention heads"
                    ),
                },
                "device": {
                    "requested": args.device,
                    "resolved": str(device),
                    "name": torch.cuda.get_device_name(device),
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                "parameters": {
                    "trainable": parameters,
                    "frozen": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if not parameter.requires_grad
                    ),
                    "optimizer_owned_trainable": sum(
                        parameter.numel()
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.requires_grad
                    ),
                },
                "data": data_observability,
                "batch": batch_observability,
                "optimization": {
                    "epochs_requested": args.epochs,
                    "early_stopping_patience": args.patience,
                    "planned_maximum_optimizer_steps": args.epochs
                    * int(batch_observability["training_steps_per_epoch"]),
                    "actual_optimizer_steps": observed(
                        None, reason="training has not started", unit="steps"
                    ),
                },
                "precision": {
                    "amp_requested": args.amp,
                    "autocast_dtype": "float16" if args.amp else "float32",
                },
                "modes": {
                    "debug": False,
                    "subset": False,
                    "fast_mode": False,
                    "validation_only": validation_only,
                },
                "resources_at_start": resource_start,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_sum = torch.zeros((), device=device, dtype=torch.float64)
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                predicted = model(batch)
                loss = (predicted.float() - batch.y).abs().mean()
            _require_finite_loss(
                loss, f"{dataset}/{MODEL_NAME}: nonfinite training loss"
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if optimizer_steps == 0:
                _validate_first_step_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            train_sum += loss.detach().double() * batch.y.numel()
            train_count += batch.y.numel()
        validation = evaluate(model, validation_loader, device)
        train_mae = float(train_sum / train_count)
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        scheduler.step(validation)
        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "validation_mae": validation,
                "epoch_seconds": epoch_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        atomic_write_json(history_path, history)
        if validation < best:
            best, best_epoch = validation, epoch
            payload = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_mae": validation,
                "dataset": dataset,
                "model": MODEL_NAME,
                "model_seed": args.model_seed,
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }
            atomic_publish(checkpoint, lambda path, state=payload: torch.save(state, path))
        print(
            f"{dataset}/{MODEL_NAME} epoch={epoch} train_mae={train_mae:.6f} "
            f"validation_mae={validation:.6f} best={best:.6f} seconds={epoch_seconds:.2f}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            break
    test = None
    if not validation_only:
        selected = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(selected["state_dict"], strict=True)
        # Standard runner: test is touched once after validation selects the checkpoint.
        assert test_loader is not None
        test = evaluate(model, test_loader, device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    resource_observability = resource_monitor.finish(
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    train_epoch_seconds = sum(float(row["epoch_seconds"]) for row in history)
    train_graph_equivalents = len(splits["train"]) * len(history)
    result = {
        "validation": best,
        "best_epoch": best_epoch,
        "trainable_parameters": parameters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "history": str(history_path),
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_allocated,
        "peak_gpu_reserved_memory_bytes": peak_reserved,
        "epochs_completed": len(history),
        "optimizer_steps_completed": optimizer_steps,
        "optimizer_steps_planned_maximum": args.epochs
        * int(batch_observability["training_steps_per_epoch"]),
        "throughput": {
            "scope": (
                "CUDA-synchronized training and validation epoch timing; checkpoint IO excluded"
            ),
            "train_graph_equivalents": train_graph_equivalents,
            "cuda_synchronized_epoch_seconds_including_validation": train_epoch_seconds,
            "train_graph_equivalents_per_second": (
                observed(
                    train_graph_equivalents / train_epoch_seconds,
                    unit="graphs_per_second",
                )
                if train_epoch_seconds > 0
                else observed(
                    None, reason="observed epoch duration was zero", unit="graphs_per_second"
                )
            ),
        },
        "execution": execution,
        "data_observability": data_observability,
        "batch_observability": batch_observability,
        "resource_observability": resource_observability,
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
                "kind": "post_run_observability",
                "dataset": dataset,
                "model": MODEL_NAME,
                "elapsed_seconds": elapsed,
                "optimizer_steps_completed": optimizer_steps,
                "resources": resource_observability,
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
        raise RuntimeError("Cycle PE benchmark test evaluation requires CUDA; no CPU fallback")
    checkpoint = args.test_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint}")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    device = torch.device(args.device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Selected checkpoint has an invalid payload schema")
    expected_metadata = {
        "dataset": dataset,
        "model": MODEL_NAME,
        "model_seed": args.model_seed,
    }
    for name, expected in expected_metadata.items():
        if payload.get(name) != expected:
            raise ValueError(f"Selected checkpoint {name} mismatch")
    saved_arguments = payload.get("arguments")
    if not isinstance(saved_arguments, dict) or saved_arguments.get("validation_only") is not True:
        raise ValueError("Selected checkpoint was not produced by validation-only training")
    for name in ("hidden_dim", "pe_dim", "layers"):
        if saved_arguments.get(name) != getattr(args, name):
            raise ValueError(f"Selected checkpoint architecture mismatch for {name}")
    validation = payload.get("validation_mae")
    epoch = payload.get("epoch")
    if (
        isinstance(validation, bool)
        or not isinstance(validation, (int, float))
        or not math.isfinite(float(validation))
        or float(validation) < 0
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
    ):
        raise ValueError("Selected checkpoint validation metadata is invalid")
    _seed(args.model_seed)
    model = CyclePEModel(
        dataset=dataset,
        hidden=args.hidden_dim,
        pe_dim=args.pe_dim,
        layers=args.layers,
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters > args.max_parameters:
        raise ValueError(
            f"{dataset}/{MODEL_NAME}: {parameters} parameters exceeds budget {args.max_parameters}"
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    execution = configure_execution(model, args, device)
    test_loader = _loader(test_graphs, args, train=False)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    resource_monitor = FailureSafeResourceMonitor(
        device, workload=f"cycle_v1_{dataset}_selected_test"
    )
    resource_start = resource_monitor.start()
    test_data_observability = _data_observability(dataset, {"test": test_graphs})
    largest_test_batch = min(args.batch_size, len(test_graphs))
    last_test_batch = len(test_graphs) % args.batch_size or largest_test_batch
    test_batch_observability = {
        "batch_unit": "molecular_graphs",
        "requested_physical_graphs_per_batch": args.batch_size,
        "observed_smallest_physical_graphs_per_batch": min(
            largest_test_batch, last_test_batch
        ),
        "observed_largest_physical_graphs_per_batch": largest_test_batch,
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "training_steps_per_epoch": observed(
            None, reason="test-only evaluation has no training steps", unit="steps"
        ),
        "effective_batch_size": largest_test_batch,
        "workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": (
            args.prefetch_factor
            if args.workers > 0
            else observed(
                None,
                reason="DataLoader prefetch is inactive because workers is zero",
            )
        ),
        "non_blocking_transfer": True,
        "cache": "immutable prepared official graph cache reused during evaluation",
    }
    print(
        json.dumps(
            {
                "kind": "pre_evaluation_observability",
                "dataset": dataset,
                "model": {
                    "name": MODEL_NAME,
                    "hidden_dimension": args.hidden_dim,
                    "positional_encoding_dimension": args.pe_dim,
                    "message_passing_layers": args.layers,
                    "total_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "trainable_parameters": parameters,
                    "attention_heads": observed(
                        None, reason="the Cycle PE V1 architecture has no attention heads"
                    ),
                },
                "data": test_data_observability,
                "batch": test_batch_observability,
                "precision": {
                    "evaluation_dtype": "float32",
                    "amp_effective": False,
                    "reason": "the V1 evaluate function executes its registered FP32 path",
                },
                "device": {
                    "requested": args.device,
                    "resolved": str(device),
                    "name": torch.cuda.get_device_name(device),
                },
                "execution": execution,
                "modes": {
                    "test_only": True,
                    "debug": False,
                    "subset": False,
                    "fast_mode": False,
                },
                "resources_at_start": resource_start,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    test = evaluate(model, test_loader, device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    resources = resource_monitor.finish(
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    throughput = {
        "scope": (
            "CUDA-synchronized selected-checkpoint test evaluation; checkpoint loading "
            "and model construction excluded"
        ),
        "evaluated_graphs": len(test_graphs),
        "evaluation_seconds": elapsed,
        "evaluation_graphs_per_second": (
            observed(len(test_graphs) / elapsed, unit="graphs_per_second")
            if elapsed > 0
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
                "kind": "post_evaluation_observability",
                "dataset": dataset,
                "model": MODEL_NAME,
                "throughput": throughput,
                "resource_summary": resources["summary"],
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
        "evaluation_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_allocated,
        "peak_gpu_reserved_memory_bytes": peak_reserved,
        "execution": execution,
        "data_observability": test_data_observability,
        "batch_observability": test_batch_observability,
        "resource_observability": resources,
        "resources_at_start": resource_start,
        "throughput": throughput,
        "optimizer_observability": {
            "optimizer_created": False,
            "optimizer_steps": 0,
            "reason": "selected-checkpoint test evaluation is read-only",
        },
        "evaluation_splits": ["test"],
        "fresh_training": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _validate(args)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.test_checkpoint is not None:
        args.test_checkpoint = args.test_checkpoint.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; choose a new run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Run already exists: {args.output_dir}; choose a new output directory"
        )
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
        "suite": "benchmark",
        "status": "running",
        "protocol": "ours_only_on_official_benchmark_splits",
        "run_mode": run_mode,
        "arguments": arguments,
        "software": versions,
        "architecture": architecture_protocol(),
        "implementation_sha256": implementation_hashes(),
        "seeds": {
            "model_seed": args.model_seed,
            "data_seed": "unused: fixed official graphs",
            "split_seed": "unused: official splits",
            "chart_seed": "unused: one deterministic BFS chart, no augmentation",
        },
        "controls": {
            "model": MODEL_NAME,
            "external_models_trained": False,
            "test_checkpoint_selection": False,
            "parameter_budget": args.max_parameters,
            "target_policy": "official labels unchanged",
            "test_data_access": run_mode in {"standard", "test_only"},
            "fresh_training": run_mode in {"standard", "validation_only"},
            "optimizer_created": run_mode in {"standard", "validation_only"},
        },
    }
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "track": TRACK_NAME,
        "suite": "benchmark",
        "status": "running",
        "model_seed": args.model_seed,
        "run_mode": run_mode,
        "datasets": {},
    }
    atomic_write_json(manifest_path, manifest)
    try:
        for dataset in args.datasets:
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
                )
            else:
                splits, protocol = load_benchmark(
                    args.data_root,
                    dataset,
                    allow_download=args.allow_download,
                    splits=requested_splits,
                )
            dataset_metrics: dict[str, Any] = {
                "metric": "mae",
                "protocol": protocol,
                "models": {},
                "data_preparation_seconds": time.perf_counter() - started,
            }
            metrics["datasets"][dataset] = dataset_metrics
            if args.test_checkpoint is not None:
                dataset_metrics["models"][MODEL_NAME] = _evaluate_test_checkpoint(
                    dataset, splits["test"], args
                )
                atomic_write_json(args.output_dir / "metrics.json", metrics)
            elif not args.prepare_only:
                dataset_metrics["models"][MODEL_NAME] = _train_model(dataset, splits, args)
                atomic_write_json(args.output_dir / "metrics.json", metrics)
            del splits
        metrics["status"] = manifest["status"] = "prepared" if args.prepare_only else "passed"
        atomic_write_json(args.output_dir / "metrics.json", metrics)
        manifest["dataset_protocols"] = {
            name: data["protocol"] for name, data in metrics["datasets"].items()
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
                (
                    "metrics.json",
                    lambda: atomic_write_json(args.output_dir / "metrics.json", metrics),
                ),
            ),
        )
        raise
    print(json.dumps({"status": metrics["status"], "output_dir": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
