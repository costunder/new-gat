#!/usr/bin/env python3
"""Run larger-model scaling experiments for both Cycle PE V1 and V2.

Every candidate uses only the official train/validation splits.  One common
profile per version and dataset is selected by mean validation MAE across all
requested model seeds; each seed's checkpoint at that profile is then evaluated
once on test without retraining.  Candidate artifacts and final test evaluations
are kept in disjoint result sections.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_json  # noqa: E402
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)

DATASETS = ("zinc12k", "peptides_struct")
VERSIONS = ("v1", "v2")
PROFILE_ORDER = ("base", "wide", "deep", "large")
PROFILES: dict[str, dict[str, int]] = {
    "base": {"hidden_dim": 64, "pe_dim": 32, "layers": 3},
    "wide": {"hidden_dim": 128, "pe_dim": 64, "layers": 3},
    "deep": {"hidden_dim": 64, "pe_dim": 32, "layers": 6},
    "large": {"hidden_dim": 128, "pe_dim": 64, "layers": 6},
}
MODEL_NAMES = {"v1": "cycle_set", "v2": "cycle_basis_v2"}
MODULES = {
    "v1": "research.cycle_pe.benchmark",
    "v2": "research.cycle_pe.v2.benchmark",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SOURCE_FILES = (
    "scripts/run_cycle_scaling.py",
    "scripts/check_dependencies.py",
    "scripts/gpu_profiles.py",
    "scripts/gpu_preflight.py",
    "scripts/verify_gpu_lock.py",
    "research/__init__.py",
    "research/cycle_pe/__init__.py",
    "research/cycle_pe/benchmark.py",
    "research/cycle_pe/benchmark_data.py",
    "research/cycle_pe/benchmark_models.py",
    "research/cycle_pe/paper_model.py",
    "research/cycle_pe/paper_data.py",
    "research/cycle_pe/features.py",
    "research/cycle_pe/v2/benchmark.py",
    "research/cycle_pe/v2/__init__.py",
    "research/cycle_pe/v2/basis.py",
    "research/cycle_pe/v2/data.py",
    "research/cycle_pe/v2/model.py",
    "src/chartgat/algebra.py",
    "src/chartgat/__init__.py",
    "src/chartgat/cache.py",
    "src/chartgat/execution.py",
    "src/chartgat/graphs.py",
)


def _run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("model seeds must be nonnegative, unique, and nonempty")
    return seeds


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--versions", nargs="+", choices=VERSIONS, default=list(VERSIONS))
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    result.add_argument("--profiles", nargs="+", choices=PROFILE_ORDER, default=list(PROFILE_ORDER))
    result.add_argument(
        "--model-seeds",
        type=_seeds,
        default=(0,),
        help="comma-separated model/minibatch seeds (default: 0)",
    )
    result.add_argument("--run-id", type=_run_id)
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--epochs", type=int, default=300)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument(
        "--max-parameters",
        type=int,
        default=5_000_000,
        help="fail-closed ceiling per model; large profile needs more than the legacy 500k cap",
    )
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--column-chunk-size", type=int, default=16)
    result.add_argument("--basis-execution", choices=("batched", "reference"), default="batched")
    result.add_argument("--basis-pair-budget", type=int, default=32768)
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument("--fail-fast", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def _validate(args: argparse.Namespace) -> None:
    for name in ("versions", "datasets", "profiles"):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values):
            raise ValueError(f"--{name} must be nonempty and contain no duplicates")
    for name in (
        "batch_size",
        "epochs",
        "patience",
        "max_parameters",
        "column_chunk_size",
        "basis_pair_budget",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0 or args.min_free_gb < 0:
        raise ValueError("invalid worker, optimizer, or GPU-memory setting")
    if not args.device.lower().startswith("cuda"):
        raise ValueError("Cycle PE scaling training requires CUDA; no CPU fallback")


def _default_run_id() -> str:
    return "cycle-scaling-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _run_dir(args: argparse.Namespace, run_id: str) -> Path:
    return (args.results_root.expanduser().resolve() / "cycle_pe/scaling" / run_id).resolve()


def make_jobs(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    data_root = args.data_root.expanduser().resolve()
    for version in args.versions:
        for profile in args.profiles:
            config = PROFILES[profile]
            for seed in args.model_seeds:
                output = run_dir / "results" / version / profile / f"model-seed-{seed}"
                job_id = f"{version}:{profile}:model-seed-{seed}"
                command = [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    MODULES[version],
                    "--suite",
                    "benchmark",
                    "--datasets",
                    *args.datasets,
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(output),
                    "--device",
                    args.device,
                    "--model-seed",
                    str(seed),
                    "--batch-size",
                    str(args.batch_size),
                    "--workers",
                    str(args.workers),
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--lr",
                    str(args.lr),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--hidden-dim",
                    str(config["hidden_dim"]),
                    "--pe-dim",
                    str(config["pe_dim"]),
                    "--layers",
                    str(config["layers"]),
                    "--max-parameters",
                    str(args.max_parameters),
                    "--validation-only",
                    "--amp" if args.amp else "--no-amp",
                    "--compile" if args.compile else "--no-compile",
                ]
                if args.allow_download:
                    command.append("--allow-download")
                if version == "v2":
                    command += [
                        "--column-chunk-size",
                        str(args.column_chunk_size),
                        "--basis-execution",
                        args.basis_execution,
                        "--basis-pair-budget",
                        str(args.basis_pair_budget),
                    ]
                jobs.append(
                    {
                        "job_id": job_id,
                        "version": version,
                        "profile": profile,
                        "model_seed": seed,
                        "datasets": list(args.datasets),
                        "config": dict(config),
                        "status": "pending",
                        "command": command,
                        "output_dir": str(output),
                        "log_path": str(
                            run_dir / "logs" / f"{version}--{profile}--seed-{seed}.log"
                        ),
                        "returncode": None,
                        "artifact_errors": [],
                    }
                )
    return jobs


def _source_snapshot() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    entries = [str(ROOT / "src"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    """Run and stream one child, terminating it if the scaling runner is interrupted."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _artifact(
    value: Any,
    digest: Any,
    *,
    expected: Path,
    label: str,
) -> tuple[str, str]:
    if not isinstance(value, str) or not isinstance(digest, str):
        raise ValueError(f"{label} path/hash metadata is missing")
    path = Path(value).expanduser().resolve()
    expected = expected.resolve()
    if path != expected or not path.is_file():
        raise ValueError(f"{label} is missing or outside its exact expected location")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != actual:
        raise ValueError(f"{label} SHA-256 mismatch")
    return str(path), actual


def read_job_rows(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Admit validation-only candidates only after fail-closed artifact checks."""
    output = Path(job["output_dir"])
    manifest = _json_object(output / "manifest.json")
    metrics = _json_object(output / "metrics.json")
    if manifest.get("status") != "passed" or metrics.get("status") != "passed":
        raise ValueError("child manifest and metrics must both have status=passed")
    if (
        manifest.get("run_mode") != "validation_only"
        or metrics.get("run_mode") != "validation_only"
    ):
        raise ValueError("candidate child must identify as validation_only")
    expected_version = job["version"]
    if expected_version == "v2" and manifest.get("version") != "v2":
        raise ValueError("V2 child did not identify itself as version v2")
    if expected_version == "v1" and manifest.get("version") not in (None, "v1"):
        raise ValueError("V1 child has an unexpected version marker")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("child manifest arguments are missing")
    expected_arguments = {
        "model_seed": job["model_seed"],
        "hidden_dim": job["config"]["hidden_dim"],
        "pe_dim": job["config"]["pe_dim"],
        "layers": job["config"]["layers"],
        "validation_only": True,
        "test_checkpoint": None,
    }
    for name, expected in expected_arguments.items():
        if arguments.get(name) != expected:
            actual = arguments.get(name)
            raise ValueError(f"child argument mismatch for {name}: {actual} != {expected}")
    if expected_version == "v2":
        expected_v2 = {
            "column_chunk_size": 16,
            "basis_execution": "batched",
            "basis_pair_budget": 32768,
        }
        command = job.get("command", [])
        for flag, name in (
            ("--column-chunk-size", "column_chunk_size"),
            ("--basis-execution", "basis_execution"),
            ("--basis-pair-budget", "basis_pair_budget"),
        ):
            if flag in command:
                value = command[command.index(flag) + 1]
                expected_v2[name] = int(value) if name != "basis_execution" else value
        for name, expected in expected_v2.items():
            if arguments.get(name) != expected:
                raise ValueError(f"child argument mismatch for {name}")
    if arguments.get("datasets") != job["datasets"]:
        raise ValueError("child dataset selection differs from the requested official datasets")
    if metrics.get("model_seed") != job["model_seed"]:
        raise ValueError("child metrics model seed mismatch")
    controls = manifest.get("controls")
    if (
        not isinstance(controls, dict)
        or controls.get("test_data_access") is not False
        or controls.get("fresh_training") is not True
        or controls.get("optimizer_created") is not True
    ):
        raise ValueError("candidate child test/training controls are invalid")
    datasets = metrics.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(job["datasets"]):
        raise ValueError("child metrics do not contain exactly the requested datasets")
    rows: list[dict[str, Any]] = []
    for dataset in job["datasets"]:
        dataset_metrics = datasets[dataset]
        if dataset_metrics.get("metric") != "mae":
            raise ValueError(f"{dataset}: expected official MAE metric")
        protocol = dataset_metrics.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("loaded_splits") != ["train", "validation"]
            or set(protocol.get("split_sizes", {})) != {"train", "validation"}
            or set(protocol.get("split_content_sha256", {})) != {"train", "validation"}
        ):
            raise ValueError(f"{dataset}: candidate protocol exposed or loaded a test split")
        models = dataset_metrics.get("models")
        model_name = MODEL_NAMES[expected_version]
        if not isinstance(models, dict) or set(models) != {model_name}:
            raise ValueError(f"{dataset}: expected only model {model_name}")
        result = models[model_name]
        if not isinstance(result, dict) or "test" in result:
            raise ValueError(f"{dataset}: candidate result leaked a test metric")
        if result.get("evaluation_splits") != ["train", "validation"]:
            raise ValueError(f"{dataset}: candidate evaluated an unexpected split")
        if result.get("fresh_training") is not True:
            raise ValueError(f"{dataset}: candidate fresh-training marker is invalid")
        validation = _finite_number(result.get("validation"), f"{dataset}/validation")
        parameters = int(
            _finite_number(result.get("trainable_parameters"), f"{dataset}/parameters", minimum=1)
        )
        elapsed = _finite_number(
            result.get("elapsed_seconds"), f"{dataset}/elapsed_seconds", minimum=0
        )
        peak_memory = int(
            _finite_number(result.get("peak_gpu_memory_bytes"), f"{dataset}/peak_gpu_memory_bytes")
        )
        best_epoch = int(
            _finite_number(result.get("best_epoch"), f"{dataset}/best_epoch", minimum=1)
        )
        epochs_completed = int(
            _finite_number(result.get("epochs_completed"), f"{dataset}/epochs_completed", minimum=1)
        )
        expected_run = output / dataset / model_name
        checkpoint, checkpoint_sha256 = _artifact(
            result.get("checkpoint"),
            result.get("checkpoint_sha256"),
            expected=expected_run / "best.pt",
            label=f"{dataset}/checkpoint",
        )
        history, history_sha256 = _artifact(
            result.get("history"),
            result.get("history_sha256"),
            expected=expected_run / "history.json",
            label=f"{dataset}/history",
        )
        history_payload = json.loads(Path(history).read_text(encoding="utf-8"))
        if (
            not isinstance(history_payload, list)
            or not history_payload
            or len(history_payload) != epochs_completed
            or not isinstance(history_payload[-1], dict)
            or history_payload[-1].get("epoch") != epochs_completed
        ):
            raise ValueError(f"{dataset}: history length/epoch metadata mismatch")
        rows.append(
            {
                "version": expected_version,
                "profile": job["profile"],
                "dataset": dataset,
                "model_seed": job["model_seed"],
                "config": dict(job["config"]),
                "validation_mae": validation,
                "trainable_parameters": parameters,
                "elapsed_seconds": elapsed,
                "peak_gpu_memory_bytes": peak_memory,
                "best_epoch": best_epoch,
                "epochs_completed": epochs_completed,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "history": history,
                "history_sha256": history_sha256,
                "output_dir": str(output),
            }
        )
    return rows


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("cannot aggregate an empty metric")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    versions: list[str],
    datasets: list[str],
    profiles: list[str],
    model_seeds: tuple[int, ...],
    complete: bool,
) -> dict[str, Any]:
    """Select one common profile per version/dataset without any test input."""
    summary: dict[str, Any] = {
        "schema_version": 2,
        "status": "pending_test_evaluation" if complete else "failed",
        "scope": "cycle_pe_v1_v2_larger_model_scaling",
        "metric": "mae_lower_is_better",
        "selection_policy": {
            "profile_selection_input": "mean validation MAE across requested model seeds",
            "selection_unit": "version x dataset",
            "test_used_for_profile_selection": False,
            "checkpoint_selection": "validation MAE only inside each candidate training",
            "final_test_report": "separate test_evaluations rows after selection",
        },
        "profiles": {name: dict(PROFILES[name]) for name in profiles},
        "requested_model_seeds": list(model_seeds),
        "runs": rows,
        "profile_aggregates": [],
        "profile_selections": [],
        "selected_checkpoints": [],
        "test_evaluations": [],
        "fresh_dataset_trainings": len(rows),
    }
    expected_keys = {
        (version, dataset, profile, seed)
        for version in versions
        for dataset in datasets
        for profile in profiles
        for seed in model_seeds
    }
    actual_keys = [
        (row["version"], row["dataset"], row["profile"], row["model_seed"]) for row in rows
    ]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        complete = False
        summary["status"] = "failed"
    aggregates: list[dict[str, Any]] = []
    for version in versions:
        for dataset in datasets:
            for profile in profiles:
                group = [
                    row
                    for row in rows
                    if row["version"] == version
                    and row["dataset"] == dataset
                    and row["profile"] == profile
                ]
                if not group:
                    continue
                seeds = [row["model_seed"] for row in group]
                if len(group) != len(model_seeds) or set(seeds) != set(model_seeds):
                    complete = False
                    summary["status"] = "failed"
                aggregates.append(
                    {
                        "version": version,
                        "dataset": dataset,
                        "profile": profile,
                        "config": dict(PROFILES[profile]),
                        "model_seeds": sorted(seeds),
                        "validation_mae": _stats([row["validation_mae"] for row in group]),
                        "trainable_parameters": sorted(
                            {row["trainable_parameters"] for row in group}
                        ),
                        "elapsed_seconds": _stats([row["elapsed_seconds"] for row in group]),
                        "peak_gpu_memory_bytes": _stats(
                            [float(row["peak_gpu_memory_bytes"]) for row in group]
                        ),
                        "epochs_completed": _stats(
                            [float(row["epochs_completed"]) for row in group]
                        ),
                    }
                )
    summary["profile_aggregates"] = aggregates
    if not complete:
        summary["selection_withheld"] = (
            "At least one requested child/result is missing or invalid; "
            "no checkpoint is selected and test remains untouched."
        )
        return summary
    for version in versions:
        for dataset in datasets:
            candidates = [
                item
                for item in aggregates
                if item["version"] == version and item["dataset"] == dataset
            ]
            if len(candidates) != len(profiles):
                summary["status"] = "failed"
                summary["selection_withheld"] = "A requested profile aggregate is missing."
                summary["profile_selections"] = []
                summary["selected_checkpoints"] = []
                return summary
            selected_profile = min(
                candidates,
                key=lambda item: (
                    item["validation_mae"]["mean"],
                    profiles.index(item["profile"]),
                ),
            )
            profile_selection_id = f"{version}:{dataset}"
            summary["profile_selections"].append(
                {
                    "profile_selection_id": profile_selection_id,
                    "version": version,
                    "dataset": dataset,
                    "selected_profile": selected_profile["profile"],
                    "config": dict(selected_profile["config"]),
                    "selection_metric": "validation_mae.mean_across_requested_model_seeds",
                    "selected_validation_mae": selected_profile["validation_mae"],
                    "model_seeds": list(model_seeds),
                    "test_used_for_selection": False,
                }
            )
            selected_rows = [
                row
                for row in rows
                if row["version"] == version
                and row["dataset"] == dataset
                and row["profile"] == selected_profile["profile"]
            ]
            selected_by_seed = {row["model_seed"]: row for row in selected_rows}
            if set(selected_by_seed) != set(model_seeds) or len(selected_rows) != len(model_seeds):
                summary["status"] = "failed"
                summary["selection_withheld"] = "Selected profile checkpoints are incomplete."
                summary["profile_selections"] = []
                summary["selected_checkpoints"] = []
                return summary
            for seed in model_seeds:
                selected = selected_by_seed[seed]
                summary["selected_checkpoints"].append(
                    {
                        "checkpoint_id": f"{version}:{dataset}:model-seed-{seed}",
                        "profile_selection_id": profile_selection_id,
                        "version": version,
                        "dataset": dataset,
                        "model_seed": seed,
                        "selected_profile": selected["profile"],
                        "config": dict(selected["config"]),
                        "selected_validation_mae": selected["validation_mae"],
                        "trainable_parameters": selected["trainable_parameters"],
                        "checkpoint": selected["checkpoint"],
                        "checkpoint_sha256": selected["checkpoint_sha256"],
                        "history": selected["history"],
                        "history_sha256": selected["history_sha256"],
                    }
                )
    return summary


def make_test_jobs(
    args: argparse.Namespace,
    run_dir: Path,
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create test-only jobs after validation has irreversibly fixed each checkpoint."""
    jobs: list[dict[str, Any]] = []
    data_root = args.data_root.expanduser().resolve()
    for selection in selections:
        version = selection["version"]
        dataset = selection["dataset"]
        seed = selection["model_seed"]
        profile = selection["selected_profile"]
        config = PROFILES[profile]
        output = run_dir / "test-evaluations" / version / dataset / f"model-seed-{seed}"
        job_id = f"test:{version}:{dataset}:model-seed-{seed}"
        command = [
            sys.executable,
            "-B",
            "-u",
            "-m",
            MODULES[version],
            "--suite",
            "benchmark",
            "--datasets",
            dataset,
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output),
            "--device",
            args.device,
            "--model-seed",
            str(seed),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--hidden-dim",
            str(config["hidden_dim"]),
            "--pe-dim",
            str(config["pe_dim"]),
            "--layers",
            str(config["layers"]),
            "--max-parameters",
            str(args.max_parameters),
            "--test-checkpoint",
            selection["checkpoint"],
            "--compile" if args.compile else "--no-compile",
        ]
        if args.allow_download:
            command.append("--allow-download")
        if version == "v2":
            command += [
                "--column-chunk-size",
                str(args.column_chunk_size),
                "--basis-execution",
                args.basis_execution,
                "--basis-pair-budget",
                str(args.basis_pair_budget),
            ]
        jobs.append(
            {
                "job_id": job_id,
                "checkpoint_id": selection["checkpoint_id"],
                "profile_selection_id": selection["profile_selection_id"],
                "version": version,
                "dataset": dataset,
                "model_seed": seed,
                "selected_profile": profile,
                "config": dict(config),
                "checkpoint": selection["checkpoint"],
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "selected_validation_mae": selection["selected_validation_mae"],
                "trainable_parameters": selection["trainable_parameters"],
                "status": "pending",
                "command": command,
                "output_dir": str(output),
                "log_path": str(
                    run_dir / "logs" / "test" / f"{version}--{dataset}--seed-{seed}.log"
                ),
                "returncode": None,
                "artifact_errors": [],
            }
        )
    return jobs


def read_test_result(job: dict[str, Any]) -> dict[str, Any]:
    """Validate one test-only evaluation and bind it to the selected checkpoint."""
    output = Path(job["output_dir"])
    manifest_path = output / "manifest.json"
    metrics_path = output / "metrics.json"
    manifest = _json_object(manifest_path)
    metrics = _json_object(metrics_path)
    if manifest.get("status") != "passed" or metrics.get("status") != "passed":
        raise ValueError("test-only manifest and metrics must both have status=passed")
    if manifest.get("run_mode") != "test_only" or metrics.get("run_mode") != "test_only":
        raise ValueError("selected checkpoint evaluation must identify as test_only")
    version = job["version"]
    if version == "v2" and manifest.get("version") != "v2":
        raise ValueError("V2 test-only child did not identify itself as version v2")
    if version == "v1" and manifest.get("version") not in (None, "v1"):
        raise ValueError("V1 test-only child has an unexpected version marker")
    arguments = manifest.get("arguments")
    expected_arguments = {
        "datasets": [job["dataset"]],
        "model_seed": job["model_seed"],
        "hidden_dim": job["config"]["hidden_dim"],
        "pe_dim": job["config"]["pe_dim"],
        "layers": job["config"]["layers"],
        "validation_only": False,
        "test_checkpoint": str(Path(job["checkpoint"]).resolve()),
    }
    if not isinstance(arguments, dict):
        raise ValueError("test-only manifest arguments are missing")
    for name, expected in expected_arguments.items():
        if arguments.get(name) != expected:
            raise ValueError(f"test-only argument mismatch for {name}")
    controls = manifest.get("controls")
    if (
        not isinstance(controls, dict)
        or controls.get("test_data_access") is not True
        or controls.get("fresh_training") is not False
        or controls.get("optimizer_created") is not False
    ):
        raise ValueError("test-only controls do not prove evaluation without retraining")
    datasets = metrics.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {job["dataset"]}:
        raise ValueError("test-only metrics must contain exactly the selected dataset")
    dataset_metrics = datasets[job["dataset"]]
    if dataset_metrics.get("metric") != "mae":
        raise ValueError("test-only result must use official MAE")
    protocol = dataset_metrics.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("loaded_splits") != ["test"]
        or set(protocol.get("split_sizes", {})) != {"test"}
        or set(protocol.get("split_content_sha256", {})) != {"test"}
    ):
        raise ValueError("test-only child must load exactly the official test split")
    model_name = MODEL_NAMES[version]
    models = dataset_metrics.get("models")
    if not isinstance(models, dict) or set(models) != {model_name}:
        raise ValueError(f"test-only child must contain only model {model_name}")
    result = models[model_name]
    if not isinstance(result, dict):
        raise ValueError("test-only model result is invalid")
    forbidden = {"validation", "history", "epochs_completed", "best_epoch"}
    if forbidden & set(result):
        raise ValueError("test-only result contains training or candidate metrics")
    if result.get("evaluation_splits") != ["test"] or result.get("fresh_training") is not False:
        raise ValueError("test-only result has invalid split/training markers")
    checkpoint = str(Path(job["checkpoint"]).resolve())
    if (
        result.get("checkpoint") != checkpoint
        or result.get("checkpoint_sha256") != job["checkpoint_sha256"]
    ):
        raise ValueError("test-only result is not bound to the selected checkpoint")
    if not Path(checkpoint).is_file():
        raise ValueError("selected checkpoint disappeared before test result admission")
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != job["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed before/during test evaluation")
    selected_validation = _finite_number(result.get("selected_validation"), "selected_validation")
    if selected_validation != job["selected_validation_mae"]:
        raise ValueError("test-only checkpoint validation metadata mismatch")
    parameters = int(
        _finite_number(result.get("trainable_parameters"), "test/trainable_parameters", minimum=1)
    )
    if parameters != job["trainable_parameters"]:
        raise ValueError("test-only model parameter count differs from selected candidate")
    return {
        "test_evaluation_id": job["job_id"],
        "checkpoint_id": job["checkpoint_id"],
        "profile_selection_id": job["profile_selection_id"],
        "version": version,
        "dataset": job["dataset"],
        "model_seed": job["model_seed"],
        "selected_profile": job["selected_profile"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": job["checkpoint_sha256"],
        "test_mae": _finite_number(result.get("test"), "test_mae"),
        "evaluation_seconds": _finite_number(
            result.get("evaluation_seconds"), "test/evaluation_seconds"
        ),
        "peak_gpu_memory_bytes": int(
            _finite_number(result.get("peak_gpu_memory_bytes"), "test/peak_gpu_memory_bytes")
        ),
        "fresh_training": False,
        "output_dir": str(output.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    }


def attach_test_results(
    summary: dict[str, Any],
    test_rows: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    expected = {item["checkpoint_id"]: item for item in summary["selected_checkpoints"]}
    actual_ids = [row["checkpoint_id"] for row in test_rows]
    if not complete or len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected):
        summary["status"] = "failed"
        summary["test_evaluations"] = []
        summary["test_results_withheld"] = (
            "Every validation-selected checkpoint must have exactly one valid test-only result."
        )
        return summary
    for row in test_rows:
        selected = expected[row["checkpoint_id"]]
        if (
            row["profile_selection_id"] != selected["profile_selection_id"]
            or row["selected_profile"] != selected["selected_profile"]
            or row["checkpoint"] != selected["checkpoint"]
            or row["checkpoint_sha256"] != selected["checkpoint_sha256"]
        ):
            summary["status"] = "failed"
            summary["test_evaluations"] = []
            summary["test_results_withheld"] = "A test result is detached from its selection."
            return summary
        selected["test_evaluation_id"] = row["test_evaluation_id"]
    summary["test_evaluations"] = test_rows
    summary["selected_test_evaluations"] = len(test_rows)
    ordered_versions = list(
        dict.fromkeys(item["version"] for item in summary["profile_selections"])
    )
    ordered_datasets = list(
        dict.fromkeys(item["dataset"] for item in summary["profile_selections"])
    )
    summary["final_test_aggregates"] = [
        {
            "version": version,
            "dataset": dataset,
            "model_seeds": list(summary["requested_model_seeds"]),
            "selected_profiles": [
                row["selected_profile"]
                for row in test_rows
                if row["version"] == version and row["dataset"] == dataset
            ],
            "test_mae": _stats(
                [
                    row["test_mae"]
                    for row in test_rows
                    if row["version"] == version and row["dataset"] == dataset
                ]
            ),
        }
        for version in ordered_versions
        for dataset in ordered_datasets
    ]
    summary["status"] = "passed"
    return summary


_JOB_STATE_FIELDS = {
    "accepted_result",
    "accepted_rows",
    "artifact_errors",
    "error",
    "finished_at_utc",
    "previous_attempts",
    "quarantined_outputs",
    "recovered_at_utc",
    "resume_artifact_errors",
    "resume_attempts",
    "returncode",
    "selection_rebinds",
    "started_at_utc",
    "status",
}

_TEST_JOB_STABLE_FIELDS = (
    "job_id",
    "checkpoint_id",
    "profile_selection_id",
    "version",
    "dataset",
    "model_seed",
    "output_dir",
    "log_path",
)


def _run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return the complete execution contract that must match on resume."""
    return {
        "versions": list(args.versions),
        "datasets": list(args.datasets),
        "profiles": list(args.profiles),
        "model_seeds": list(args.model_seeds),
        "data_root": str(args.data_root.expanduser().resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_parameters": args.max_parameters,
        "allow_download": args.allow_download,
        "amp": args.amp,
        "compile": args.compile,
        "column_chunk_size": args.column_chunk_size,
        "basis_execution": args.basis_execution,
        "basis_pair_budget": args.basis_pair_budget,
        "min_free_gb": args.min_free_gb,
    }


def _restore_job_state(
    generated: list[dict[str, Any]],
    stored: Any,
    *,
    label: str,
) -> None:
    """Bind persisted state to an identical regenerated plan, failing closed."""
    if not isinstance(stored, list):
        raise ValueError(f"stored {label} job plan is not a list")
    stored_by_id: dict[str, dict[str, Any]] = {}
    for item in stored:
        if not isinstance(item, dict) or not isinstance(item.get("job_id"), str):
            raise ValueError(f"stored {label} job is malformed")
        job_id = item["job_id"]
        if job_id in stored_by_id:
            raise ValueError(f"stored {label} job IDs are not unique")
        stored_by_id[job_id] = item
    generated_ids = {job["job_id"] for job in generated}
    if set(stored_by_id) != generated_ids:
        raise ValueError(f"stored {label} job matrix differs from this invocation")
    for job in generated:
        previous = stored_by_id[job["job_id"]]
        binding = {key: value for key, value in job.items() if key not in _JOB_STATE_FIELDS}
        for key, expected in binding.items():
            if previous.get(key) != expected:
                raise ValueError(f"stored {label} job binding differs for {job['job_id']}/{key}")
        status = previous.get("status")
        if status not in {"pending", "running", "passed", "failed"}:
            raise ValueError(f"stored {label} job has invalid status: {job['job_id']}")
        for key in _JOB_STATE_FIELDS:
            if key in previous:
                job[key] = previous[key]


def _restore_or_rebind_test_job_state(
    generated: list[dict[str, Any]],
    stored: Any,
    run_dir: Path,
) -> None:
    """Restore identical test jobs or preserve and replace a superseded selection."""
    if not isinstance(stored, list):
        raise ValueError("stored test-evaluation job plan is not a list")
    stored_by_id: dict[str, dict[str, Any]] = {}
    for item in stored:
        if not isinstance(item, dict) or not isinstance(item.get("job_id"), str):
            raise ValueError("stored test-evaluation job is malformed")
        if item["job_id"] in stored_by_id:
            raise ValueError("stored test-evaluation job IDs are not unique")
        stored_by_id[item["job_id"]] = item
    if set(stored_by_id) != {job["job_id"] for job in generated}:
        raise ValueError("stored test-evaluation job matrix differs from this invocation")
    for job in generated:
        previous = stored_by_id[job["job_id"]]
        if any(previous.get(key) != job[key] for key in _TEST_JOB_STABLE_FIELDS):
            raise ValueError(f"stored test-evaluation identity differs for {job['job_id']}")
        previous_binding = {
            key: value for key, value in previous.items() if key not in _JOB_STATE_FIELDS
        }
        generated_binding = {
            key: value for key, value in job.items() if key not in _JOB_STATE_FIELDS
        }
        if previous_binding == generated_binding:
            status = previous.get("status")
            if status not in {"pending", "running", "passed", "failed"}:
                raise ValueError(f"stored test-evaluation job has invalid status: {job['job_id']}")
            for key in _JOB_STATE_FIELDS:
                if key in previous:
                    job[key] = previous[key]
            continue

        _validate_job_paths([previous], run_dir)
        _quarantine_incomplete_output(previous, run_dir)
        history = list(previous.get("previous_attempts", []))
        history.append(
            {
                "status": previous.get("status"),
                "selected_profile": previous.get("selected_profile"),
                "checkpoint": previous.get("checkpoint"),
                "checkpoint_sha256": previous.get("checkpoint_sha256"),
                "selected_validation_mae": previous.get("selected_validation_mae"),
                "accepted_result": previous.get("accepted_result"),
                "output_dir": previous.get("output_dir"),
                "quarantined_output": (
                    previous.get("quarantined_outputs", [])[-1]
                    if previous.get("quarantined_outputs")
                    else None
                ),
            }
        )
        job["previous_attempts"] = history
        job["selection_rebinds"] = int(previous.get("selection_rebinds", 0)) + 1
        for key in ("resume_attempts", "quarantined_outputs"):
            if key in previous:
                job[key] = previous[key]


def _resume_manifest(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    jobs: list[dict[str, Any]],
    dependencies: dict[str, Any],
    sources: dict[str, str],
) -> tuple[dict[str, Any], str]:
    """Load a same-run continuation only when every immutable binding still matches."""
    manifest = _json_object(run_dir / "manifest.json")
    expected_identity = {
        "schema_version": 2,
        "scope": "cycle_pe_v1_v2_larger_model_scaling",
        "run_id": run_id,
        "output_dir": str(run_dir),
        "run_configuration": _run_configuration(args),
        "dependencies": dependencies,
        "source_sha256": sources,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"existing run cannot be resumed: {key} differs")
    previous_status = manifest.get("status")
    if previous_status not in {"running", "failed", "passed"}:
        raise ValueError("existing run cannot be resumed: status is invalid")
    if manifest.get("source_integrity_valid") is not True:
        raise ValueError("existing run cannot be resumed after a source-integrity failure")
    _restore_job_state(jobs, manifest.get("jobs"), label="candidate")
    manifest["jobs"] = jobs
    manifest["status"] = "running"
    manifest.pop("error", None)
    manifest.pop("finished_at_utc", None)
    manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    manifest["resumed_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    return manifest, previous_status


def _recover_candidate_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-admit complete candidates from disk, including a child finished before interruption."""
    rows: list[dict[str, Any]] = []
    for job in jobs:
        previous_status = job.get("status")
        if job.get("status") == "failed" and job.get("returncode") not in (None, 0):
            job["status"] = "pending"
            job["returncode"] = None
            job["artifact_errors"] = []
            continue
        output_exists = Path(job["output_dir"]).exists()
        try:
            recovered = read_job_rows(job)
            if previous_status == "passed" and job.get("accepted_rows") != recovered:
                raise ValueError("passed candidate artifacts differ from their accepted rows")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if output_exists or job.get("status") in {"running", "passed"}:
                job.setdefault("resume_artifact_errors", []).append(f"{type(exc).__name__}: {exc}")
            job["status"] = "pending"
            job["returncode"] = None
            job["artifact_errors"] = []
        else:
            rows.extend(recovered)
            job["status"] = "passed"
            job["returncode"] = 0
            job["artifact_errors"] = []
            if previous_status != "passed":
                job["accepted_rows"] = recovered
            job["recovered_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    return rows


def _recover_test_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-admit test-only artifacts while rechecking their selected-checkpoint binding."""
    rows: list[dict[str, Any]] = []
    for job in jobs:
        previous_status = job.get("status")
        if job.get("status") == "failed" and job.get("returncode") not in (None, 0):
            job["status"] = "pending"
            job["returncode"] = None
            job["artifact_errors"] = []
            continue
        output_exists = Path(job["output_dir"]).exists()
        try:
            recovered = read_test_result(job)
            if previous_status == "passed" and job.get("accepted_result") != recovered:
                raise ValueError("passed test artifact differs from its accepted result")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if output_exists or job.get("status") in {"running", "passed"}:
                job.setdefault("resume_artifact_errors", []).append(f"{type(exc).__name__}: {exc}")
            job["status"] = "pending"
            job["returncode"] = None
            job["artifact_errors"] = []
        else:
            rows.append(recovered)
            job["status"] = "passed"
            job["returncode"] = 0
            job["artifact_errors"] = []
            if previous_status != "passed":
                job["accepted_result"] = recovered
            job["recovered_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    return rows


def _quarantine_incomplete_output(job: dict[str, Any], run_dir: Path) -> None:
    """Preserve an incomplete child directory before retrying its exact output binding."""
    lexical_output = Path(os.path.abspath(Path(job["output_dir"]).expanduser()))
    output = lexical_output.resolve()
    run_dir = run_dir.resolve()
    if output != lexical_output:
        raise ValueError(f"refusing to move an indirect retry output: {lexical_output}")
    if not output.exists():
        return
    if output == run_dir or not output.is_relative_to(run_dir):
        raise ValueError(f"refusing to move unsafe retry output: {output}")
    root = (run_dir / "resume-orphans" / re.sub(r"[^A-Za-z0-9_.-]+", "--", job["job_id"])).resolve()
    if not root.is_relative_to(run_dir):
        raise ValueError("refusing to use a retry archive outside the run directory")
    root.mkdir(parents=True, exist_ok=True)
    attempt = int(job.get("resume_attempts", 0)) + 1
    target = (root / f"attempt-{attempt}").resolve()
    while target.exists():
        attempt += 1
        target = (root / f"attempt-{attempt}").resolve()
    if not target.is_relative_to(run_dir):
        raise ValueError("refusing to move retry output outside the run directory")
    lexical_output.rename(target)
    job["resume_attempts"] = attempt
    job.setdefault("quarantined_outputs", []).append(str(target))


def _validated_log_path(log_path: Path, run_dir: Path) -> Path:
    """Reject indirect or out-of-run log targets before opening them for append."""
    lexical = Path(os.path.abspath(log_path.expanduser()))
    resolved = lexical.resolve()
    log_root = (run_dir.resolve() / "logs").resolve()
    if resolved != lexical or not resolved.is_relative_to(log_root):
        raise ValueError(f"refusing to write an indirect or out-of-run log: {lexical}")
    return lexical


def _validated_write_path(path: Path, run_dir: Path, *, label: str) -> Path:
    """Validate a direct in-run file path before passing it to a child writer."""
    lexical = Path(os.path.abspath(path.expanduser()))
    resolved = lexical.resolve()
    run_dir = run_dir.resolve()
    if (
        resolved != lexical
        or resolved == run_dir
        or resolved.is_dir()
        or (lexical.exists() and not lexical.is_file())
        or not resolved.is_relative_to(run_dir)
    ):
        raise ValueError(f"refusing to write an indirect or out-of-run {label}: {lexical}")
    return lexical


def _validate_job_paths(jobs: list[dict[str, Any]], run_dir: Path) -> None:
    """Require every mutable child output and log to be a direct in-run path."""
    run_dir = run_dir.resolve()
    for job in jobs:
        lexical_output = Path(os.path.abspath(Path(job["output_dir"]).expanduser()))
        output = lexical_output.resolve()
        if output != lexical_output or output == run_dir or not output.is_relative_to(run_dir):
            raise ValueError(f"unsafe output path for {job['job_id']}: {lexical_output}")
        _validated_log_path(Path(job["log_path"]), run_dir)


def _manifest_base(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    jobs: list[dict[str, Any]],
    dependencies: dict[str, Any],
    sources: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scope": "cycle_pe_v1_v2_larger_model_scaling",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "output_dir": str(run_dir),
        "versions": list(args.versions),
        "datasets": list(args.datasets),
        "profiles": {name: dict(PROFILES[name]) for name in args.profiles},
        "model_seeds": list(args.model_seeds),
        "fresh_child_runs": len(jobs),
        "fresh_dataset_trainings": len(jobs) * len(args.datasets),
        "selected_test_evaluations_planned": len(args.versions)
        * len(args.datasets)
        * len(args.model_seeds),
        "selection_protocol": {
            "profile_selection": (
                "one common profile per version/dataset by mean validation MAE "
                "across all requested model seeds"
            ),
            "checkpoint_selection": "validation only inside each candidate child",
            "candidate_loaded_splits": ["train", "validation"],
            "test_role": "one test-only evaluation of each selected checkpoint; no retraining",
            "official_splits": True,
        },
        "resource_reporting": (
            "per-dataset trainable parameters, CUDA-synchronized training runtime, "
            "peak allocated GPU memory, epochs and best epoch"
        ),
        "dependencies": dependencies,
        "source_sha256": sources,
        "source_integrity_valid": True,
        "run_configuration": _run_configuration(args),
        "preflight": {"status": "pending"},
        "jobs": jobs,
        "test_evaluation_jobs": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_id = args.run_id or _default_run_id()
    results_root = args.results_root.expanduser().resolve()
    run_dir = _run_dir(args, run_id)
    data_root = args.data_root.expanduser().resolve()
    if not run_dir.is_relative_to(results_root):
        print("Experiment outputs must stay within the results root", file=sys.stderr)
        return 2
    paths_overlap = (
        run_dir == data_root
        or run_dir.is_relative_to(data_root)
        or data_root.is_relative_to(run_dir)
    )
    if paths_overlap:
        print("Experiment outputs and dataset directories must not overlap", file=sys.stderr)
        return 2
    jobs = make_jobs(args, run_dir)
    if args.dry_run:
        print(
            f"{len(jobs)} fresh child runs; {len(jobs) * len(args.datasets)} fresh dataset "
            "trainings (train+validation only); "
            f"{len(args.versions) * len(args.datasets) * len(args.model_seeds)} "
            "selected-checkpoint test evaluations are deferred until validation selection"
        )
        for job in jobs:
            print(f"[{job['job_id']}] {shlex.join(job['command'])}")
            print(f"  output: {job['output_dir']}")
        print(f"manifest: {run_dir / 'manifest.json'}")
        print(f"summary: {run_dir / 'summary.json'}")
        return 0
    try:
        manifest_path = _validated_write_path(
            run_dir / "manifest.json", run_dir, label="runner manifest"
        )
        summary_path = _validated_write_path(
            run_dir / "summary.json", run_dir, label="aggregate summary"
        )
    except ValueError as exc:
        print(f"Existing run cannot be resumed: {exc}", file=sys.stderr)
        return 2
    try:
        dependencies = check_dependencies()
    except DependencyCheckError as exc:
        print(error_message(exc), file=sys.stderr)
        return exc.exit_code
    sources = _source_snapshot()
    is_resume = run_dir.exists()
    resume_previous_status: str | None = None
    if is_resume:
        try:
            manifest, resume_previous_status = _resume_manifest(
                args,
                run_id,
                run_dir,
                jobs,
                dependencies,
                sources,
            )
            _validate_job_paths(jobs, run_dir)
            rows = _recover_candidate_rows(jobs)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Existing run cannot be resumed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = _manifest_base(args, run_id, run_dir, jobs, dependencies, sources)
        rows = []
    if (
        is_resume
        and resume_previous_status == "passed"
        and all(job["status"] == "passed" for job in jobs)
    ):
        try:
            verified_validation = build_summary(
                rows,
                versions=list(args.versions),
                datasets=list(args.datasets),
                profiles=list(args.profiles),
                model_seeds=args.model_seeds,
                complete=len(rows) == len(jobs) * len(args.datasets),
            )
            if verified_validation["status"] != "pending_test_evaluation":
                raise ValueError("completed candidate matrix no longer validates")
            verified_test_jobs = make_test_jobs(
                args, run_dir, verified_validation["selected_checkpoints"]
            )
            _restore_or_rebind_test_job_state(
                verified_test_jobs,
                manifest.get("test_evaluation_jobs"),
                run_dir,
            )
            _validate_job_paths(verified_test_jobs, run_dir)
            verified_test_rows = _recover_test_rows(verified_test_jobs)
            verified_complete = (
                len(verified_test_rows) == len(verified_test_jobs)
                and len(verified_test_jobs)
                == len(args.versions) * len(args.datasets) * len(args.model_seeds)
                and all(job["status"] == "passed" for job in verified_test_jobs)
            )
            verified_summary = attach_test_results(
                verified_validation, verified_test_rows, complete=verified_complete
            )
            if verified_summary["status"] == "passed":
                if _json_object(summary_path) != verified_summary:
                    raise ValueError("stored aggregate summary differs from verified artifacts")
                print(
                    f"Cycle scaling run already complete; verified {len(jobs)} candidates and "
                    f"{len(verified_test_jobs)} selected-checkpoint evaluations"
                )
                return 0
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Existing run cannot be resumed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    atomic_write_json(manifest_path, manifest, sort_keys=False)
    environment = _environment()
    test_rows: list[dict[str, Any]] = []
    validation_summary: dict[str, Any] | None = None
    failed = False
    try:
        preflight_output = _validated_write_path(
            run_dir / "gpu-preflight.json", run_dir, label="GPU preflight output"
        )
        preflight_command = [
            sys.executable,
            "-B",
            str(ROOT / "scripts/gpu_preflight.py"),
            "--device",
            args.device,
            "--require-paper-deps",
            "--min-free-gb",
            str(args.min_free_gb),
            "--json-out",
            str(preflight_output),
        ]
        preflight_log = _validated_log_path(run_dir / "logs/gpu-preflight.log", run_dir)
        preflight_code = run_logged(preflight_command, preflight_log, environment)
        preflight_errors: list[str] = []
        if preflight_code == 0:
            try:
                preflight_payload = _json_object(preflight_output)
                if preflight_payload.get("status") != "passed":
                    preflight_errors.append("GPU preflight JSON does not have status=passed")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                preflight_errors.append(f"{type(exc).__name__}: {exc}")
        manifest["preflight"] = {
            "status": "passed" if preflight_code == 0 and not preflight_errors else "failed",
            "returncode": preflight_code,
            "command": preflight_command,
            "output": str(preflight_output),
            "artifact_errors": preflight_errors,
        }
        if manifest["preflight"]["status"] != "passed":
            failed = True
        atomic_write_json(manifest_path, manifest, sort_keys=False)
        if not failed:
            for job in jobs:
                if job["status"] == "passed":
                    continue
                _quarantine_incomplete_output(job, run_dir)
                job["artifact_errors"] = []
                job["status"] = "running"
                job["started_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
                atomic_write_json(manifest_path, manifest, sort_keys=False)
                log_path = _validated_log_path(Path(job["log_path"]), run_dir)
                code = run_logged(job["command"], log_path, environment)
                job["returncode"] = code
                job["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
                if code == 0:
                    try:
                        job_rows = read_job_rows(job)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        job["artifact_errors"] = [f"{type(exc).__name__}: {exc}"]
                    else:
                        rows.extend(job_rows)
                        job["accepted_rows"] = job_rows
                job["status"] = "passed" if code == 0 and not job["artifact_errors"] else "failed"
                failed = failed or job["status"] == "failed"
                atomic_write_json(manifest_path, manifest, sort_keys=False)
                if failed and args.fail_fast:
                    break
        candidate_complete = (
            not failed
            and len(rows) == len(jobs) * len(args.datasets)
            and all(job["status"] == "passed" for job in jobs)
        )
        validation_summary = build_summary(
            rows,
            versions=list(args.versions),
            datasets=list(args.datasets),
            profiles=list(args.profiles),
            model_seeds=args.model_seeds,
            complete=candidate_complete,
        )
        if validation_summary["status"] != "pending_test_evaluation":
            failed = True
        if _source_snapshot() != sources:
            failed = True
            manifest["source_integrity_valid"] = False
            manifest["source_integrity_error"] = "experiment source changed during the run"
        stored_test_jobs = manifest.get("test_evaluation_jobs", [])
        test_jobs = (
            make_test_jobs(args, run_dir, validation_summary["selected_checkpoints"])
            if not failed
            else list(stored_test_jobs)
            if is_resume and isinstance(stored_test_jobs, list)
            else []
        )
        if not failed and is_resume and stored_test_jobs:
            _restore_or_rebind_test_job_state(test_jobs, stored_test_jobs, run_dir)
            _validate_job_paths(test_jobs, run_dir)
            test_rows = _recover_test_rows(test_jobs)
        manifest["test_evaluation_jobs"] = test_jobs
        atomic_write_json(manifest_path, manifest, sort_keys=False)
        if not failed:
            for test_job in test_jobs:
                if test_job["status"] == "passed":
                    continue
                _quarantine_incomplete_output(test_job, run_dir)
                test_job["artifact_errors"] = []
                test_job["status"] = "running"
                test_job["started_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
                atomic_write_json(manifest_path, manifest, sort_keys=False)
                log_path = _validated_log_path(Path(test_job["log_path"]), run_dir)
                code = run_logged(test_job["command"], log_path, environment)
                test_job["returncode"] = code
                test_job["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
                if code == 0:
                    try:
                        test_row = read_test_result(test_job)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        test_job["artifact_errors"] = [f"{type(exc).__name__}: {exc}"]
                    else:
                        test_rows.append(test_row)
                        test_job["accepted_result"] = test_row
                test_job["status"] = (
                    "passed" if code == 0 and not test_job["artifact_errors"] else "failed"
                )
                failed = failed or test_job["status"] == "failed"
                atomic_write_json(manifest_path, manifest, sort_keys=False)
                if failed and args.fail_fast:
                    break
        if _source_snapshot() != sources:
            failed = True
            manifest["source_integrity_valid"] = False
            manifest["source_integrity_error"] = "experiment source changed during the run"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        atomic_write_json(manifest_path, manifest, sort_keys=False)
        summary = build_summary(
            rows,
            versions=list(args.versions),
            datasets=list(args.datasets),
            profiles=list(args.profiles),
            model_seeds=args.model_seeds,
            complete=False,
        )
        summary["error"] = manifest["error"]
        atomic_write_json(summary_path, summary, sort_keys=False)
        raise
    assert validation_summary is not None
    test_jobs = manifest["test_evaluation_jobs"]
    test_complete = (
        not failed
        and len(test_rows) == len(test_jobs)
        and len(test_jobs) == len(args.versions) * len(args.datasets) * len(args.model_seeds)
        and all(job["status"] == "passed" for job in test_jobs)
    )
    summary = attach_test_results(validation_summary, test_rows, complete=test_complete)
    if summary["status"] != "passed":
        failed = True
    manifest["status"] = "failed" if failed else "passed"
    manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    manifest["summary"] = str(summary_path)
    manifest["completed_child_runs"] = sum(job["status"] == "passed" for job in jobs)
    manifest["completed_dataset_trainings"] = len(rows)
    manifest["completed_selected_test_evaluations"] = len(test_rows)
    atomic_write_json(summary_path, summary, sort_keys=False)
    atomic_write_json(manifest_path, manifest, sort_keys=False)
    if failed:
        print(f"Cycle scaling run failed; inspect {manifest_path}", file=sys.stderr)
        return 1
    print(f"Cycle scaling run passed; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
