#!/usr/bin/env python3
"""Run the fixed-C versus shared-dynamic-C Conductance V5 comparison.

The runner is deliberately restartable.  Reusing the exact run ID verifies the
stored configuration, source snapshot, and completed child artifacts, then runs
only missing or interrupted children.  It never silently falls back to CPU.
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
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_json  # noqa: E402
from research.conductance_gat.v5.protocol import (  # noqa: E402
    BETA_PARAMETERIZATIONS,
    COMPARISON_DESIGN,
    CONDITIONS,
    DATASETS,
    DEFAULT_BETA_INITIAL,
    DEFAULT_BETA_PARAMETERIZATION,
    DEFAULT_DATASETS,
    HARDWARE_PROFILES,
    SCALE_PROFILES,
    SUITE,
    beta_configuration,
)
from scripts import run_conductance_factorial as shared  # noqa: E402
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}")
SAMPLING_CHOICES = ("auto", "full", "neighbor", "cluster")
TRANSDUCTIVE_DATASETS = frozenset({"cora", "citeseer", "pubmed", "ogbn-arxiv"})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DEFAULT_DATASETS))
    result.add_argument("--profile", choices=tuple(SCALE_PROFILES), default="reference")
    result.add_argument("--hidden-channels", type=int)
    result.add_argument("--layers", type=int)
    result.add_argument("--heads", type=int)
    result.add_argument("--ffn-multiplier", type=int)
    result.add_argument("--dropout", type=float)
    result.add_argument(
        "--beta-parameterization",
        choices=BETA_PARAMETERIZATIONS,
        default=DEFAULT_BETA_PARAMETERIZATION,
    )
    result.add_argument("--beta-initial", type=float, default=DEFAULT_BETA_INITIAL)
    result.add_argument("--beta-min", type=float)
    result.add_argument("--beta-max", type=float)
    result.add_argument("--model-seed", type=int, default=0)
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--run-id")
    result.add_argument("--device", default="cuda")
    result.add_argument("--epochs", type=int, default=300)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--ppi-batch-size", type=int)
    result.add_argument("--sample-seed-batch-size", type=int)
    result.add_argument("--edge-chunk-size", type=int)
    result.add_argument("--hardware-profile", choices=tuple(HARDWARE_PROFILES), default="portable")
    result.add_argument(
        "--activation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the selected hardware profile's checkpoint policy",
    )
    result.add_argument(
        "--sampling",
        choices=SAMPLING_CHOICES,
        default="auto",
        help="auto uses cluster sampling only for ogbn-arxiv and full graphs elsewhere",
    )
    result.add_argument(
        "--num-neighbors",
        nargs="+",
        type=int,
        default=[15, 10],
        help="fanout values forwarded to neighbor sampling",
    )
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument("--dry-run", action="store_true")
    return result


def _architecture(args: argparse.Namespace) -> dict[str, Any]:
    architecture = dict(SCALE_PROFILES[args.profile])
    for name in ("hidden_channels", "layers", "heads", "ffn_multiplier", "dropout"):
        override = getattr(args, name)
        if override is not None:
            architecture[name] = override
    architecture.update(
        beta_configuration(
            args.beta_parameterization,
            args.beta_initial,
            args.beta_min,
            args.beta_max,
        )
    )
    return architecture


def _sampling(dataset: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cluster" if dataset == "ogbn-arxiv" else "full"


def _resolved_execution(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    profile = HARDWARE_PROFILES[args.hardware_profile]
    batch_size = 1
    if dataset == "ppi":
        batch_size = args.ppi_batch_size or profile["ppi_batch_size"]
    return {
        "hardware_profile": args.hardware_profile,
        "precision": profile["precision"],
        "tf32": profile["tf32"],
        "batch_size": batch_size,
        "sample_seed_batch_size": (
            args.sample_seed_batch_size or profile["sample_seed_batch_size"]
        ),
        "edge_chunk_size": args.edge_chunk_size or profile["edge_chunk_size"],
        "activation_checkpoint": (
            profile["activation_checkpoint"]
            if args.activation_checkpoint is None
            else args.activation_checkpoint
        ),
        "sample_prefetch": profile["sample_prefetch"],
        "pin_memory": profile["pin_memory"],
    }


def _effective_min_free_gb(args: argparse.Namespace) -> float:
    return max(
        float(args.min_free_gb),
        float(HARDWARE_PROFILES[args.hardware_profile]["minimum_free_memory_gib"]),
    )


def _validate(args: argparse.Namespace) -> None:
    if not args.datasets or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("datasets must be nonempty and contain no duplicates")
    if args.model_seed < 0:
        raise ValueError("model seed must be nonnegative")
    architecture = _architecture(args)
    integer_fields = ("hidden_channels", "layers", "heads", "ffn_multiplier")
    if any(
        isinstance(architecture[name], bool) or int(architecture[name]) < 1
        for name in integer_fields
    ):
        raise ValueError("width, depth, heads, and FFN multiplier must be positive")
    if architecture["hidden_channels"] % architecture["heads"]:
        raise ValueError("hidden channels must be divisible by heads")
    if not 0 <= float(architecture["dropout"]) < 1:
        raise ValueError("dropout must be in [0, 1)")
    if (
        min(
            args.epochs,
            args.patience,
            args.batch_size,
            args.ppi_batch_size or 1,
            args.sample_seed_batch_size or 1,
            args.edge_chunk_size or 1,
        )
        < 1
    ):
        raise ValueError("epochs, patience, batch sizes, and edge chunk size must be positive")
    if args.batch_size != 1:
        raise ValueError(
            "V5 runner-level batch size must be 1; PPI graph batching is profile-controlled"
        )
    if args.hardware_profile == "portable" and args.ppi_batch_size not in {None, 2}:
        raise ValueError("portable PPI retains graph batch-size 2")
    if (
        args.workers != 0
        or not args.num_neighbors
        or any(value < 1 for value in args.num_neighbors)
    ):
        raise ValueError(
            "V5 requires workers=0 for reproducible epoch resume and positive neighbor fanouts"
        )
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("CUDA is required; CPU fallback is not supported")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("minimum free GPU memory must be finite and nonnegative")
    if args.run_id is not None and RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID must be 1-120 letters, digits, underscores, or hyphens")
    if args.sampling in {"neighbor", "cluster"} and any(
        dataset not in TRANSDUCTIVE_DATASETS for dataset in args.datasets
    ):
        raise ValueError("neighbor/cluster sampling is transductive-only; PPI requires full")


def make_jobs(
    args: argparse.Namespace, run_dir: Path, architecture: dict[str, Any]
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    data_root = args.data_root.expanduser().resolve()
    for dataset in args.datasets:
        sampling = _sampling(dataset, args.sampling)
        execution = _resolved_execution(args, dataset)
        child_batch_size = execution["batch_size"]
        for condition in CONDITIONS:
            output = run_dir / dataset / condition
            command = [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "research.conductance_gat.v5.train",
                "--dataset",
                dataset,
                "--condition",
                condition,
                "--output-dir",
                str(output),
                "--data-root",
                str(data_root),
                "--device",
                args.device,
                "--model-seed",
                str(args.model_seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--workers",
                str(args.workers),
                "--batch-size",
                str(child_batch_size),
                "--hardware-profile",
                args.hardware_profile,
                "--sample-seed-batch-size",
                str(execution["sample_seed_batch_size"]),
                "--edge-chunk-size",
                str(execution["edge_chunk_size"]),
                "--sampling",
                sampling,
                (
                    "--activation-checkpoint"
                    if execution["activation_checkpoint"]
                    else "--no-activation-checkpoint"
                ),
                "--num-neighbors",
                *(str(value) for value in args.num_neighbors),
            ]
            for name, value in architecture.items():
                command.extend(("--" + name.replace("_", "-"), str(value)))
            jobs.append(
                {
                    "job_id": f"{dataset}/{condition}",
                    "dataset": dataset,
                    "condition": condition,
                    "architecture": dict(architecture),
                    "sampling": sampling,
                    "batch_size": child_batch_size,
                    "num_neighbors": list(args.num_neighbors),
                    "execution": execution,
                    "status": "pending",
                    "output_dir": str(output),
                    "metrics_path": str(output / "metrics.json"),
                    "log_path": str(run_dir / "logs" / f"{dataset}--{condition}.log"),
                    "command": command,
                }
            )
    return jobs


def _source_snapshot() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "scripts/check_dependencies.py",
        ROOT / "scripts/gpu_preflight.py",
        ROOT / "scripts/run_conductance_factorial.py",
        ROOT / "src/chartgat/cache.py",
        ROOT / "src/chartgat/execution.py",
    ]
    paths += list((ROOT / "research/conductance_gat/v5").glob("*.py"))
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def _identity(job: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "job_id",
        "dataset",
        "condition",
        "architecture",
        "sampling",
        "batch_size",
        "num_neighbors",
        "execution",
        "output_dir",
        "metrics_path",
        "log_path",
        "command",
    )
    return {key: job[key] for key in keys}


def _load_metrics(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["metrics_path"])
    if not path.is_file():
        raise RuntimeError(f"child returned without metrics: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise RuntimeError("child metrics do not certify status=passed")
    for key, expected in (
        ("dataset", job["dataset"]),
        ("condition", job["condition"]),
        ("model_seed", int(job["command"][job["command"].index("--model-seed") + 1])),
        ("evaluation_split", "validation"),
        ("test_evaluated", False),
    ):
        if payload.get(key) != expected:
            raise RuntimeError(f"child metric {key} mismatch")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or any(
        configuration.get(key) != value for key, value in job["architecture"].items()
    ):
        raise RuntimeError("child architecture does not match the requested V5 profile")
    if configuration.get("sampling") != job["sampling"]:
        raise RuntimeError("child sampling mode does not match the requested V5 job")
    if configuration.get("batch_size") != job["batch_size"]:
        raise RuntimeError("child graph batch size does not match the V5 dataset contract")
    execution = job["execution"]
    for key in ("hardware_profile", "precision", "tf32", "edge_chunk_size"):
        if configuration.get(key) != execution[key]:
            raise RuntimeError(f"child hardware execution mismatch for {key}")
    hardware = payload.get("hardware_execution")
    expected_hardware = {
        "profile": execution["hardware_profile"],
        "precision": execution["precision"],
        "tf32": execution["tf32"],
        "activation_checkpoint": execution["activation_checkpoint"],
        "edge_chunk_size": execution["edge_chunk_size"],
        "sample_seed_batch_size": execution["sample_seed_batch_size"],
        "graph_batch_size": execution["batch_size"],
        "sample_prefetch": execution["sample_prefetch"],
        "pin_memory": execution["pin_memory"],
    }
    if not isinstance(hardware, dict) or any(
        hardware.get(key) != value for key, value in expected_hardware.items()
    ):
        raise RuntimeError("child hardware execution metadata is missing or inconsistent")
    return {
        "metrics_sha256": hashlib.sha256(raw).hexdigest(),
        "validation": payload.get("validation"),
        "global_best_validation": payload.get("global_best_validation"),
        "joint_best_validation": payload.get("joint_best_validation"),
        "checkpoint_selection": payload.get("checkpoint_selection"),
        "metric_name": payload.get("metric_name"),
        "best_epoch": payload.get("best_epoch"),
        "epochs_run": payload.get("epochs_run"),
        "trainable_parameters": payload.get("trainable_parameters"),
        "peak_cuda_allocated_bytes": payload.get(
            "peak_cuda_allocated_bytes", payload.get("peak_gpu_memory_bytes")
        ),
        "peak_cuda_reserved_bytes": payload.get("peak_cuda_reserved_bytes"),
        "throughput": payload.get("throughput"),
        "hardware_execution": hardware,
    }


def _next_log(path: Path) -> Path:
    if not path.exists():
        return path
    attempt = 1
    while True:
        candidate = path.with_name(f"{path.stem}.attempt-{attempt}{path.suffix}")
        if not candidate.exists():
            return candidate
        attempt += 1


def _safe_clear(job: dict[str, Any], run_dir: Path) -> None:
    lexical = Path(os.path.abspath(Path(job["output_dir"])))
    resolved = lexical.resolve()
    if resolved != lexical or resolved == run_dir or not resolved.is_relative_to(run_dir):
        raise RuntimeError(f"refusing to clear unsafe child output: {lexical}")
    if lexical.exists():
        if not lexical.is_dir():
            raise RuntimeError(f"child output is not a directory: {lexical}")
        shutil.rmtree(lexical)


def _write_comparison(run_dir: Path, manifest: dict[str, Any]) -> None:
    from research.conductance_gat.v5.report import write_comparison

    write_comparison(run_dir, manifest)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    architecture = _architecture(args)
    run_id = args.run_id or "dynamic-c-v5-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    results_root = args.results_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    run_dir = (results_root / "conductance_gat/v5" / run_id).resolve()
    if not run_dir.is_relative_to(results_root) or (
        run_dir == data_root
        or run_dir.is_relative_to(data_root)
        or data_root.is_relative_to(run_dir)
    ):
        print(
            "experiment outputs must stay inside results and outside the data cache",
            file=sys.stderr,
        )
        return 2
    jobs = make_jobs(args, run_dir, architecture)
    config = {
        "datasets": list(args.datasets),
        "profile": args.profile,
        "architecture": architecture,
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "ppi_batch_size": args.ppi_batch_size,
        "sample_seed_batch_size": args.sample_seed_batch_size,
        "edge_chunk_size": args.edge_chunk_size,
        "activation_checkpoint": args.activation_checkpoint,
        "hardware_profile": args.hardware_profile,
        "resolved_execution_by_dataset": {
            dataset: _resolved_execution(args, dataset) for dataset in args.datasets
        },
        "sampling": args.sampling,
        "num_neighbors": list(args.num_neighbors),
        "device": args.device,
        "min_free_gb": args.min_free_gb,
        "effective_min_free_gb": _effective_min_free_gb(args),
        "data_root": str(data_root),
    }
    if args.dry_run:
        print(
            f"{len(jobs)} V5 validation-only trainings; profile={args.profile}; "
            f"architecture={architecture}"
        )
        for job in jobs:
            print(shlex.join(job["command"]))
        print(f"Manifest: {run_dir / 'manifest.json'}")
        return 0
    try:
        dependencies = check_dependencies()
    except DependencyCheckError as exc:
        print(error_message(exc), file=sys.stderr)
        return exc.exit_code
    sources = _source_snapshot()
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": 1,
                "suite": SUITE,
                "run_id": run_id,
                "config": config,
                "source_sha256": sources,
                "dependencies": dependencies,
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise RuntimeError("existing run contract differs from this invocation")
            if [_identity(job) for job in manifest.get("jobs", [])] != [
                _identity(job) for job in jobs
            ]:
                raise RuntimeError("existing V5 job matrix differs from this invocation")
            for job in manifest["jobs"]:
                if job.get("status") == "passed":
                    result = _load_metrics(job)
                    if job.get("result") != result:
                        raise RuntimeError(f"passed artifact changed: {job['job_id']}")
                else:
                    job["status"] = "pending"
                    for key in ("result", "error", "exit_code", "elapsed_seconds"):
                        job.pop(key, None)
            jobs = manifest["jobs"]
            manifest.update(status="running", resumed_at_utc=dt.datetime.now(dt.UTC).isoformat())
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest.pop("error", None)
            manifest.pop("finished_at_utc", None)
        except Exception as exc:
            print(f"Refusing to resume: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    else:
        run_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "suite": SUITE,
            "run_id": run_id,
            "status": "running",
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "config": config,
            "conditions": list(CONDITIONS),
            "jobs": jobs,
            "dependencies": dependencies,
            "source_sha256": sources,
            "protocol": {
                "comparison_design": dict(COMPARISON_DESIGN),
                "architecture": "same multi-head W/beta/FFN architecture and initialization",
                "update_allocation": (
                    "fixed-C strong spatial recipe versus dynamic-C coordinate recipe; "
                    "not a single-factor causal C contrast"
                ),
                "selection": (
                    "primary comparison uses fixed all-epoch best versus dynamic C-active "
                    "best; dynamic all-epoch prediction best is auxiliary; no test evaluation"
                ),
                "sampling": "full validation; auto uses cluster train sampling only on ogbn-arxiv",
                "hardware_execution": (
                    "portable preserves FP32/checkpointed defaults; a6000-48gb is an opt-in "
                    "BF16-dense/TF32 throughput recipe with FP32 conductance geometry, larger "
                    "real batches and no synthetic duplicate work. Profiles are distinct "
                    "optimization recipes and their metrics must not be directly compared"
                ),
                "resume": (
                    "same immutable run contract skips hash-verified passed jobs and resumes V5 "
                    "at epoch boundaries; CUDA bitwise determinism is not claimed"
                ),
            },
        }
    atomic_write_json(manifest_path, manifest)
    _write_comparison(run_dir, manifest)
    environment = shared._environment()
    environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    current: dict[str, Any] | None = None
    try:
        preflight = [
            sys.executable,
            "-B",
            str(ROOT / "scripts/gpu_preflight.py"),
            "--device",
            args.device,
            "--require-paper-deps",
            "--min-free-gb",
            str(_effective_min_free_gb(args)),
            "--json-out",
            str(run_dir / "gpu-preflight.json"),
        ]
        if shared.run_logged(preflight, _next_log(run_dir / "logs/preflight.log"), environment):
            raise RuntimeError("GPU preflight failed")
        for index, job in enumerate(jobs, start=1):
            if job["status"] == "passed":
                print(f"[{index}/{len(jobs)}] verified, skip {job['job_id']}", flush=True)
                continue
            current = job
            if _source_snapshot() != sources:
                raise RuntimeError("V5 source changed during execution")
            attempt_command = list(job["command"])
            last_checkpoint = Path(job["output_dir"]) / "last.pt"
            if last_checkpoint.is_file():
                attempt_command.append("--resume")
            else:
                _safe_clear(job, run_dir)
            job["attempt_command"] = attempt_command
            job["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            started = time.monotonic()
            status = shared.run_logged(
                attempt_command, _next_log(Path(job["log_path"])), environment
            )
            job.update(exit_code=status, elapsed_seconds=time.monotonic() - started)
            if status:
                raise RuntimeError(f"{job['job_id']} failed with exit code {status}")
            job["result"] = _load_metrics(job)
            job["status"] = "passed"
            current = None
            atomic_write_json(manifest_path, manifest)
            _write_comparison(run_dir, manifest)
        if _source_snapshot() != sources:
            raise RuntimeError("V5 source changed during execution")
        manifest.update(status="passed", finished_at_utc=dt.datetime.now(dt.UTC).isoformat())
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        )
        if current is not None:
            current.update(status="failed", error=manifest["error"])
        atomic_write_json(manifest_path, manifest)
        _write_comparison(run_dir, manifest)
        print(f"Failed: {manifest['error']}\nSaved partial results: {run_dir}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    atomic_write_json(manifest_path, manifest)
    _write_comparison(run_dir, manifest)
    print(f"V5 comparison passed: {run_dir / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
