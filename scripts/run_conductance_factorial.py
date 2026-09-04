#!/usr/bin/env python3
"""Run four isolated Conductance conditions, one seed, on existing official data."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_json  # noqa: E402
from research.conductance_gat.ablation.protocol import (  # noqa: E402
    COMMON,
    CONDITIONS,
    DATASETS,
    DEFAULT_DATASETS,
)
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)
from scripts.process_safety import (  # noqa: E402
    close_owned_child_stdout,
    run_failure_reporter,
    terminate_owned_child_after_error,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DEFAULT_DATASETS))
    result.add_argument("--model-seed", type=int, default=0, help="One seed, not a seed list")
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--run-id")
    result.add_argument("--device", default="cuda")
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument(
        "--batch-size", type=int, default=2, help="PPI only; shared across conditions"
    )
    result.add_argument(
        "--workers",
        type=int,
        default=4,
        help="PPI graph DataLoader workers; transductive children are forced to 0",
    )
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument(
        "--dry-run", action="store_true", help="Print the plan without GPU or writes"
    )
    return result


def _validate(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("CUDA is required; CPU training/fallback is not supported")
    if args.model_seed < 0 or args.workers < 0:
        raise ValueError("model seed and workers must be nonnegative")
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("duplicate datasets are not allowed")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("minimum free GPU memory must be finite and nonnegative")
    if args.run_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", args.run_id
    ):
        raise ValueError("run ID must be 1-120 letters, digits, underscores or hyphens")


def workers_for_dataset(dataset: str, requested_workers: int) -> int:
    """Resolve the real DataLoader worker count without pretending full graphs have a loader."""

    if requested_workers < 0:
        raise ValueError("workers must be nonnegative")
    return requested_workers if dataset == "ppi" else 0


def make_jobs(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for dataset in args.datasets:
        child_workers = workers_for_dataset(dataset, args.workers)
        for condition in CONDITIONS:
            output = run_dir / dataset / condition
            command = [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "research.conductance_gat.ablation.train",
                "--dataset",
                dataset,
                "--condition",
                condition,
                "--output-dir",
                str(output),
                "--data-root",
                str(args.data_root.expanduser().resolve()),
                "--device",
                args.device,
                "--model-seed",
                str(args.model_seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(child_workers),
            ]
            jobs.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "workers": child_workers,
                    "status": "pending",
                    "output_dir": str(output),
                    "metrics_path": str(output / "metrics.json"),
                    "log_path": str(run_dir / "logs" / f"{dataset}--{condition}.log"),
                    "command": command,
                }
            )
    return jobs


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


def run_logged(command: list[str], log: Path, environment: dict[str, str]) -> int:
    """Stream child output; an interrupted parent terminates its own active child."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8", newline="\n") as stream:
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
        primary_error: BaseException | None = None
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()
        except BaseException as error:
            primary_error = error
            terminate_owned_child_after_error(
                process,
                command,
                original_error=error,
                log_target=stream,
            )
            raise
        finally:
            close_owned_child_stdout(process, original_error=primary_error)


def _source_snapshot() -> dict[str, Any]:
    files = sorted((ROOT / "research/conductance_gat/ablation").glob("*.py"))
    files += [
        ROOT / "research/conductance_gat" / name
        for name in ("benchmark.py", "benchmark_data.py", "sparse.py")
    ]
    files += [
        Path(__file__),
        ROOT / "scripts/check_dependencies.py",
        ROOT / "scripts/gpu_preflight.py",
        ROOT / "scripts/process_safety.py",
        ROOT / "src/chartgat/observability.py",
    ]
    hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "git_revision": revision,
        "sha256": hashes,
        "note": "File hashes describe executed sources, including uncommitted edits.",
    }


def _comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from research.conductance_gat.ablation.report import write_comparison

    return write_comparison(run_dir, manifest)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_id = args.run_id or "factorial-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.results_root.expanduser().resolve() / "conductance_gat/ablations" / run_id
    data_root = args.data_root.expanduser().resolve()
    if run_dir == data_root or run_dir.is_relative_to(data_root):
        print("Experiment outputs must be outside the dataset directory", file=sys.stderr)
        return 2
    jobs = make_jobs(args, run_dir)
    if args.dry_run:
        print(f"One model seed: {args.model_seed}; {len(jobs)} fresh trainings; validation only")
        for job in jobs:
            print(shlex.join(job["command"]))
        print(f"Comparison: {run_dir / 'comparison.md'}")
        return 0
    if run_dir.exists():
        print(f"Run already exists; use a new run ID: {run_dir}", file=sys.stderr)
        return 2
    try:
        dependency_report = check_dependencies()
    except DependencyCheckError as exc:
        print(error_message(exc), file=sys.stderr)
        return exc.exit_code
    run_dir.mkdir(parents=True, exist_ok=False)
    common_config = {
        **COMMON,
        **{
            key: getattr(args, key)
            for key in (
                "datasets",
                "model_seed",
                "epochs",
                "patience",
                "batch_size",
                "workers",
                "device",
            )
        },
        "data_root": str(data_root),
        "workers_by_dataset": {
            dataset: workers_for_dataset(dataset, args.workers) for dataset in args.datasets
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite": "conductance_factorial",
        "run_id": run_id,
        "status": "running",
        "source_integrity_valid": True,
        "config": common_config,
        "conditions": CONDITIONS,
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "jobs": jobs,
        "dependencies": dependency_report,
        "sources": _source_snapshot(),
        "protocol": {
            "selection": "best validation per condition; identical early-stopping policy",
            "test": "not evaluated: exploratory train/validation comparison",
            "initialization": "reset same model seed before every condition; verify state hash",
            "data": "existing official caches only; same cache hash for each dataset's four arms",
            "factor_order": list(CONDITIONS),
            "normalization": (
                "row node-degree preconditioning is a distinct operator, not a speed optimization"
            ),
            "uncertainty": (
                "one seed; no seed standard deviation, CI, significance or population claim"
            ),
            "reproducibility": "same seeds, not guaranteed bitwise CUDA scatter determinism",
        },
    }
    manifest_path = run_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    environment = _environment()
    current_job = None
    try:
        _comparison(run_dir, manifest)
        preflight = [
            sys.executable,
            "-B",
            str(ROOT / "scripts/gpu_preflight.py"),
            "--device",
            args.device,
            "--require-paper-deps",
            "--min-free-gb",
            str(args.min_free_gb),
            "--json-out",
            str(run_dir / "gpu-preflight.json"),
        ]
        status = run_logged(preflight, run_dir / "logs/preflight.log", environment)
        if status != 0:
            raise RuntimeError(f"GPU preflight failed with exit code {status}")
        print(f"Run: {run_id}; seed {args.model_seed}; {len(jobs)} fresh trainings", flush=True)
        for index, job in enumerate(jobs, start=1):
            current_job = job
            if _source_snapshot()["sha256"] != manifest["sources"]["sha256"]:
                manifest["source_integrity_valid"] = False
                raise RuntimeError(
                    "Experiment source changed during the run; refusing mixed revisions"
                )
            job["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            print(f"\n[{index}/{len(jobs)}] {job['dataset']} / {job['condition']}", flush=True)
            started = time.monotonic()
            status = run_logged(job["command"], Path(job["log_path"]), environment)
            job["elapsed_seconds"] = time.monotonic() - started
            job["exit_code"] = status
            if status != 0:
                raise RuntimeError(
                    f"{job['dataset']}/{job['condition']} failed with exit code {status}"
                )
            if not Path(job["metrics_path"]).is_file():
                raise RuntimeError(f"Child returned without metrics: {job['metrics_path']}")
            job["status"] = "passed"
            atomic_write_json(manifest_path, manifest)
            _comparison(run_dir, manifest)
            current_job = None
        if _source_snapshot()["sha256"] != manifest["sources"]["sha256"]:
            manifest["source_integrity_valid"] = False
            raise RuntimeError("Experiment source changed during the run; refusing mixed revisions")
        manifest["status"] = "passed"
        manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        _comparison(run_dir, manifest)
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        if current_job is not None:
            current_job["status"] = "failed"
            current_job["error"] = manifest["error"]
        report_error = run_failure_reporter(
            lambda: _comparison(run_dir, manifest),
            original_error=exc,
            action="failed-run comparison generation",
        )
        if report_error is not None:
            manifest.setdefault("failure_persistence_errors", []).append(report_error)
        manifest_error = run_failure_reporter(
            lambda: atomic_write_json(manifest_path, manifest),
            original_error=exc,
            action="failed-run manifest persistence",
        )
        if manifest_error is not None:
            manifest.setdefault("failure_persistence_errors", []).append(manifest_error)
        print(f"Failed: {manifest['error']}\nSaved partial results: {run_dir}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    atomic_write_json(manifest_path, manifest)
    print((run_dir / "comparison.md").read_text(encoding="utf-8"), flush=True)
    print(f"Comparison: {run_dir / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
