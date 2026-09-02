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
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chartgat.cache import atomic_publish, atomic_write_json
from chartgat.execution import add_execution_arguments, configure_execution
from research.cycle_pe.benchmark_data import DATASETS, Graph, collate, load_benchmark
from research.cycle_pe.benchmark_models import MODEL_NAME, CyclePEModel, architecture_protocol

TRACK_NAME = "cycle_pe"
IMPLEMENTATION_FILES = (
    "research/cycle_pe/benchmark.py",
    "research/cycle_pe/benchmark_data.py",
    "research/cycle_pe/benchmark_models.py",
    "research/cycle_pe/features.py",
    "research/cycle_pe/paper_model.py",
    "src/chartgat/algebra.py",
    "src/chartgat/cache.py",
    "src/chartgat/execution.py",
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
    result.add_argument("--workers", type=int, default=0)
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
    return DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
        generator=generator,
        worker_init_fn=_worker_seed,
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def evaluate(model: CyclePEModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for batch in loader:
        batch = batch.to(device)
        predicted = model(batch).float()
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("nonfinite validation/test prediction")
        total += (predicted - batch.y).abs().sum().double()
        count += batch.y.numel()
    if count == 0:
        raise ValueError("cannot evaluate an empty official split")
    return float(total / count)


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
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
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
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{dataset}/{MODEL_NAME}: nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
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
    result = {
        "validation": best,
        "best_epoch": best_epoch,
        "trainable_parameters": parameters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "history": str(history_path),
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "epochs_completed": len(history),
        "execution": execution,
        "epoch_timing": "cuda_synchronized_train_and_validation_excluding_checkpoint_io",
        "evaluation_splits": ["train", "validation"],
        "fresh_training": True,
    }
    if test is not None:
        result["test"] = test
        result["evaluation_splits"].append("test")
    return result


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
    started = time.perf_counter()
    test = evaluate(model, test_loader, device)
    torch.cuda.synchronize(device)
    return {
        "test": test,
        "selected_validation": float(validation),
        "selected_epoch": epoch,
        "trainable_parameters": parameters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "execution": execution,
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
    except Exception as exc:
        manifest["status"] = metrics["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(args.output_dir / "metrics.json", metrics)
        raise
    print(json.dumps({"status": metrics["status"], "output_dir": str(args.output_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
