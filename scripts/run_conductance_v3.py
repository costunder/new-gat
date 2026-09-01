#!/usr/bin/env python3
"""Train shared relative conductances vs C=1, one seed, CUDA only."""

from __future__ import annotations

import argparse
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
from research.conductance_gat.v3.protocol import (  # noqa: E402
    COMMON,
    CONDITIONS,
    DATASETS,
    DEFAULT_DATASETS,
    SUITE,
)
from scripts import run_conductance_factorial as shared  # noqa: E402
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)

run_logged = shared.run_logged


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Fixed-graph datasets: " + ", ".join(DATASETS) + "; default: ogbn-arxiv",
    )
    result.add_argument("--model-seed", type=int, default=0, help="One seed, not a seed list")
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--run-id")
    result.add_argument("--device", default="cuda")
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--batch-size", type=int, default=1, help="Must be 1: full-graph training")
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--edge-chunk-size", type=int, default=65536)
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument("--dry-run", action="store_true", help="Print plan without GPU or writes")
    return result


def _validate(args: argparse.Namespace) -> None:
    shared._validate(args)
    if "ppi" in args.datasets:
        raise ValueError(
            "PPI is outside the initial V3 benchmark protocol, which uses transductive "
            "node-classification datasets. This is a protocol limit, not a shared-gate limitation."
        )
    if not args.datasets or any(dataset not in DATASETS for dataset in args.datasets):
        raise ValueError("Unsupported fixed-graph dataset; choose: " + ", ".join(DATASETS))
    if args.edge_chunk_size < 1:
        raise ValueError("edge chunk size must be positive")
    if args.batch_size != 1 or args.workers != 0:
        raise ValueError("Full-graph V3 requires batch-size=1 and workers=0")


def make_jobs(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for dataset in args.datasets:
        for condition in CONDITIONS:
            output = run_dir / dataset / condition
            command = [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "research.conductance_gat.v3.train",
                "--dataset",
                dataset,
                "--condition",
                condition,
                "--output-dir",
                str(output),
                "--data-root",
                str(args.data_root.expanduser().resolve()),
            ]
            for key in (
                "device",
                "model_seed",
                "epochs",
                "patience",
                "batch_size",
                "workers",
                "edge_chunk_size",
            ):
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
    files = list((ROOT / "research/conductance_gat/v3").glob("*.py"))
    files += [
        Path(__file__),
        ROOT / "src/chartgat/cache.py",
        ROOT / "research/conductance_gat/v3/reproduce.sh",
        ROOT / "scripts/conda_env.sh",
    ]
    for path in files:
        snapshot["sha256"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return snapshot


def _comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from research.conductance_gat.v3.report import write_comparison

    return write_comparison(run_dir, manifest)


def _check_sources(manifest: dict[str, Any]) -> None:
    if _source_snapshot()["sha256"] != manifest["sources"]["sha256"]:
        manifest["source_integrity_valid"] = False
        raise RuntimeError("Experiment source changed during the run; refusing mixed sources")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_id = args.run_id or "relative-c-v3-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (args.results_root.expanduser().resolve() / "conductance_gat/v3" / run_id).resolve()
    data_root = args.data_root.expanduser().resolve()
    if (
        run_dir == data_root
        or run_dir.is_relative_to(data_root)
        or data_root.is_relative_to(run_dir)
    ):
        print("Experiment outputs and dataset directories must not overlap", file=sys.stderr)
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
                "edge_chunk_size",
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
            "initialization": "same full state hash including frozen shared-gate scaffold; "
            "alpha active in both arms",
            "data": "same verified official cache/split and ordered topology; no downloads",
            "contrast": "fresh relative_c minus fresh fixed_c; "
            "never reuse any V1/V2 or older score",
            "fixed_c": "exact C=1; gate scaffold frozen/excluded from optimizer, "
            "alpha remains trainable",
            "normalization": "symmetric in both arms; AdamW backbone WD=0.0005; gate/scalar WD=0",
            "transductive": "initial protocol uses official fixed-graph train/validation masks",
            "optimizer": "AdamW; gate MLP lr=2*base lr; alpha/gamma/tau lr=base lr; "
            "gate/scalar WD=0",
            "interventions": "selected checkpoint validation only: "
            "mean C, shuffled C, C=1, propagation off",
            "v2_comparison": "V2-to-V3 changes multiple factors; "
            "this is not a single-factor V2 contrast",
            "resources": "whole training-loop runtime and peak allocation include diagnostics "
            "and checkpoint IO; no isolated kernel benchmark or V1/spectral speedup claim",
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
            _check_sources(manifest)
            job["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            print(f"\n[{index}/{len(jobs)}] {job['dataset']} / {job['condition']}", flush=True)
            started = time.monotonic()
            status = run_logged(job["command"], Path(job["log_path"]), environment)
            job.update(elapsed_seconds=time.monotonic() - started, exit_code=status)
            _check_sources(manifest)
            if status:
                raise RuntimeError(
                    f"{job['dataset']}/{job['condition']} failed with exit code {status}"
                )
            metrics = Path(job["metrics_path"])
            if not metrics.is_file():
                raise RuntimeError(f"Child returned without metrics: {metrics}")
            job["metrics_sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
            job["status"] = "passed"
            atomic_write_json(manifest_path, manifest)
            _comparison(run_dir, manifest)
            current_job = None
        _check_sources(manifest)
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
