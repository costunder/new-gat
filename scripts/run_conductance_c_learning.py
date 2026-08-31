#!/usr/bin/env python3
"""Fresh learned-C vs fixed-C=1 training under node-degree normalization, one seed."""

from __future__ import annotations

import datetime as dt
import hashlib
import shlex
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_json  # noqa: E402
from research.conductance_gat.c_learning.protocol import COMMON, CONDITIONS  # noqa: E402
from scripts import run_conductance_factorial as shared  # noqa: E402
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)

SUITE = "conductance_c_learning"
run_logged = shared.run_logged


def parser():
    result = shared.parser()
    result.description = __doc__
    return result


def make_jobs(args, run_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for dataset in args.datasets:
        for condition in CONDITIONS:
            output = run_dir / dataset / condition
            command = [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "research.conductance_gat.c_learning.train",
                "--dataset",
                dataset,
                "--condition",
                condition,
                "--output-dir",
                str(output),
                "--data-root",
                str(args.data_root.expanduser().resolve()),
            ]
            for key in ("device", "model_seed", "epochs", "patience", "batch_size", "workers"):
                command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            jobs.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "status": "pending",
                    "output_dir": str(output),
                    "metrics_path": str(output / "metrics.json"),
                    "log_path": str(run_dir / "logs" / f"{dataset}--{condition}.log"),
                    "command": command,
                }
            )
    return jobs


def _source_snapshot() -> dict[str, Any]:
    snapshot = shared._source_snapshot()
    files = list((ROOT / "research/conductance_gat/c_learning").glob("*.py"))
    files += [Path(__file__), ROOT / "src/chartgat/cache.py"]
    for path in files:
        snapshot["sha256"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return snapshot


def _comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from research.conductance_gat.c_learning.report import write_comparison

    return write_comparison(run_dir, manifest)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        shared._validate(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_id = args.run_id or "c-learning-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        args.results_root.expanduser().resolve() / "conductance_gat/c_learning" / run_id
    ).resolve()
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
        dependencies = check_dependencies()
    except DependencyCheckError as exc:
        print(error_message(exc), file=sys.stderr)
        return exc.exit_code
    run_dir.mkdir(parents=True, exist_ok=False)
    common = {
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
    }
    manifest = {
        "schema_version": 1,
        "suite": SUITE,
        "run_id": run_id,
        "status": "running",
        "source_integrity_valid": True,
        "config": common,
        "conditions": CONDITIONS,
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "jobs": jobs,
        "dependencies": dependencies,
        "sources": _source_snapshot(),
        "protocol": {
            "selection": "best validation checkpoint per arm; same early-stopping policy",
            "test": "not evaluated; exploratory validation comparison",
            "initialization": "same full state hash including unused frozen gate scaffold",
            "data": "same verified official cache/split per dataset; no downloads",
            "contrast": "fresh learned_c minus fresh fixed_c; never reuse an older score",
            "fixed_c": "exact C=1; scaffold frozen and excluded from optimizer/trainable count",
            "normalization": "node_degree in both arms; nongate Adam L2=0.0005",
            "uncertainty": "n=1; no CI, seed standard deviation or significance claim",
            "reproducibility": "same seeds; CUDA scatter need not be bitwise deterministic",
        },
    }
    manifest_path = run_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    current_job = None
    environment = shared._environment()
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
        if status:
            raise RuntimeError(f"GPU preflight failed with exit code {status}")
        print(f"Run: {run_id}; seed {args.model_seed}; {len(jobs)} fresh trainings", flush=True)
        for index, job in enumerate(jobs, start=1):
            current_job = job
            if _source_snapshot()["sha256"] != manifest["sources"]["sha256"]:
                manifest["source_integrity_valid"] = False
                raise RuntimeError(
                    "Experiment source changed during the run; refusing mixed sources"
                )
            job["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            print(f"\n[{index}/{len(jobs)}] {job['dataset']} / {job['condition']}", flush=True)
            started = time.monotonic()
            status = run_logged(job["command"], Path(job["log_path"]), environment)
            job.update(elapsed_seconds=time.monotonic() - started, exit_code=status)
            if status:
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
            raise RuntimeError("Experiment source changed during the run; refusing mixed sources")
        manifest.update(status="passed", finished_at_utc=dt.datetime.now(dt.UTC).isoformat())
        _comparison(run_dir, manifest)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        )
        if current_job is not None:
            current_job.update(status="failed", error=manifest["error"])
        atomic_write_json(manifest_path, manifest)
        try:
            _comparison(run_dir, manifest)
        except (ValueError, OSError) as report_error:
            print(f"Comparison integrity error: {report_error}", file=sys.stderr)
        print(f"Failed: {manifest['error']}\nSaved partial results: {run_dir}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    atomic_write_json(manifest_path, manifest)
    print((run_dir / "comparison.md").read_text(encoding="utf-8"), flush=True)
    print(f"Comparison: {run_dir / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
