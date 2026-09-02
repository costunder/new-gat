#!/usr/bin/env python3
"""Run validation-only architecture scaling for Conductance V1, V2, V3 and V4."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shlex
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_json  # noqa: E402
from research.conductance_gat.v2.protocol import (  # noqa: E402
    CONDITIONS as V2_CONDITIONS,
)
from research.conductance_gat.v2.protocol import (  # noqa: E402
    DATASETS as V2_DATASETS,
)
from research.conductance_gat.v3.protocol import (  # noqa: E402
    CONDITIONS as V3_CONDITIONS,
)
from research.conductance_gat.v3.protocol import (  # noqa: E402
    DATASETS as V3_DATASETS,
)
from research.conductance_gat.v4.protocol import (  # noqa: E402
    CONDITIONS as V4_CONDITIONS,
)
from research.conductance_gat.v4.protocol import (  # noqa: E402
    DATASETS as V4_DATASETS,
)
from scripts import run_conductance_factorial as shared  # noqa: E402
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)

V1_DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
PROFILES: dict[str, dict[str, Any]] = {
    "base": {"hidden_channels": 64, "layers": 2, "dropout": 0.5},
    "wide": {"hidden_channels": 128, "layers": 2, "dropout": 0.5},
    "deep": {"hidden_channels": 64, "layers": 4, "dropout": 0.5},
    "large": {"hidden_channels": 128, "layers": 4, "dropout": 0.5},
}
VERSIONS: dict[str, dict[str, Any]] = {
    "v1": {
        "module": "research.conductance_gat.scaling_v1",
        "datasets": tuple(V1_DATASETS),
        "conditions": ("conductance",),
    },
    "v2": {
        "module": "research.conductance_gat.v2.train",
        "datasets": tuple(V2_DATASETS),
        "conditions": tuple(V2_CONDITIONS),
    },
    "v3": {
        "module": "research.conductance_gat.v3.train",
        "datasets": tuple(V3_DATASETS),
        "conditions": tuple(V3_CONDITIONS),
    },
    "v4": {
        "module": "research.conductance_gat.v4.train",
        "datasets": tuple(V4_DATASETS),
        "conditions": tuple(V4_CONDITIONS),
    },
}
ALL_DATASETS = tuple(
    dict.fromkeys(dataset for spec in VERSIONS.values() for dataset in spec["datasets"])
)
DEFAULT_MODEL_SEEDS = (0, 1, 2, 3, 4)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--versions", nargs="+", choices=tuple(VERSIONS), default=list(VERSIONS))
    result.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    result.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        help="Default: every dataset supported by each selected version; V2 has no PPI arm",
    )
    result.add_argument("--model-seeds", nargs="+", type=int, default=list(DEFAULT_MODEL_SEEDS))
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--run-id")
    result.add_argument("--device", default="cuda")
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--edge-chunk-size", type=int, default=65536)
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument("--dry-run", action="store_true", help="Print every child without writes")
    return result


def _validate(args: argparse.Namespace) -> None:
    for label, values in (
        ("versions", args.versions),
        ("profiles", args.profiles),
        ("model seeds", args.model_seeds),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{label} must be nonempty and contain no duplicates")
    if args.datasets is not None and len(set(args.datasets)) != len(args.datasets):
        raise ValueError("datasets must contain no duplicates")
    if any(seed < 0 for seed in args.model_seeds):
        raise ValueError("model seeds must be nonnegative")
    if min(args.epochs, args.patience, args.edge_chunk_size) < 1 or args.workers != 0:
        raise ValueError("epochs/patience/chunk size must be positive and workers must be 0")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("CUDA is required; CPU training/fallback is not supported")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("minimum free GPU memory must be finite and nonnegative")
    if args.run_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", args.run_id
    ):
        raise ValueError("run ID must be 1-120 letters, digits, underscores or hyphens")


def _selected_datasets(args: argparse.Namespace, version: str) -> list[str]:
    supported = VERSIONS[version]["datasets"]
    requested = supported if args.datasets is None else args.datasets
    return [dataset for dataset in requested if dataset in supported]


def _exclusions(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.datasets is None:
        return []
    output = []
    for version in args.versions:
        supported = VERSIONS[version]["datasets"]
        for dataset in args.datasets:
            if dataset not in supported:
                output.append(
                    {
                        "version": version,
                        "dataset": dataset,
                        "status": "not_applicable",
                        "reason": (
                            "V2 direct edge conductances are bound to one fixed topology and "
                            "cannot transfer to held-out PPI graphs"
                            if version == "v2" and dataset == "ppi"
                            else "dataset is not supported by this version"
                        ),
                    }
                )
    return output


def make_jobs(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    data_root = args.data_root.expanduser().resolve()
    for version in args.versions:
        spec = VERSIONS[version]
        for profile_name in args.profiles:
            profile = PROFILES[profile_name]
            for seed in args.model_seeds:
                for dataset in _selected_datasets(args, version):
                    for condition in spec["conditions"]:
                        output = (
                            run_dir
                            / version
                            / profile_name
                            / f"model-seed-{seed}"
                            / dataset
                            / condition
                        )
                        command = [
                            sys.executable,
                            "-B",
                            "-u",
                            "-m",
                            spec["module"],
                            "--dataset",
                            dataset,
                            "--output-dir",
                            str(output),
                            "--data-root",
                            str(data_root),
                            "--device",
                            args.device,
                            "--model-seed",
                            str(seed),
                            "--epochs",
                            str(args.epochs),
                            "--patience",
                            str(args.patience),
                            "--hidden-channels",
                            str(profile["hidden_channels"]),
                            "--layers",
                            str(profile["layers"]),
                            "--dropout",
                            str(profile["dropout"]),
                            "--workers",
                            str(args.workers),
                            "--batch-size",
                            "2" if dataset == "ppi" or version == "v1" else "1",
                        ]
                        if version != "v1":
                            command += ["--condition", condition]
                        if version in {"v2", "v3", "v4"}:
                            command += ["--edge-chunk-size", str(args.edge_chunk_size)]
                        job_id = f"{version}/{profile_name}/model-seed-{seed}/{dataset}/{condition}"
                        jobs.append(
                            {
                                "job_id": job_id,
                                "version": version,
                                "profile": profile_name,
                                "architecture": dict(profile),
                                "model_seed": seed,
                                "dataset": dataset,
                                "condition": condition,
                                "status": "pending",
                                "output_dir": str(output),
                                "metrics_path": str(output / "metrics.json"),
                                "log_path": str(
                                    run_dir
                                    / "logs"
                                    / version
                                    / profile_name
                                    / f"seed-{seed}--{dataset}--{condition}.log"
                                ),
                                "command": command,
                            }
                        )
    if not jobs:
        raise ValueError("selection produces no supported version/dataset/profile jobs")
    return jobs


def _source_snapshot() -> dict[str, str]:
    paths = [Path(__file__)]
    paths += [
        ROOT / "research/conductance_gat" / name
        for name in (
            "benchmark.py",
            "benchmark_data.py",
            "cache_validation.py",
            "public_data.py",
            "scaling_v1.py",
            "sparse.py",
        )
    ]
    paths += list((ROOT / "research/conductance_gat/ablation").glob("*.py"))
    paths += list((ROOT / "research/conductance_gat/v2").glob("*.py"))
    paths += list((ROOT / "research/conductance_gat/v3").glob("*.py"))
    paths += list((ROOT / "research/conductance_gat/v4").glob("*.py"))
    paths += [
        ROOT / "src/chartgat/cache.py",
        ROOT / "src/chartgat/execution.py",
        ROOT / "scripts/check_dependencies.py",
        ROOT / "scripts/gpu_preflight.py",
        ROOT / "scripts/run_conductance_factorial.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def _check_sources(manifest: dict[str, Any]) -> None:
    if _source_snapshot() != manifest["source_sha256"]:
        manifest["source_integrity_valid"] = False
        raise RuntimeError("Conductance scaling source changed during execution")


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output) or (minimum is not None and output < minimum):
        raise ValueError(f"{label} is outside its valid range")
    return output


def _load_child(job: dict[str, Any]) -> dict[str, Any]:
    metrics_path = Path(job["metrics_path"])
    if not metrics_path.is_file():
        raise RuntimeError(f"child returned without metrics: {metrics_path}")
    raw = metrics_path.read_bytes()
    child = json.loads(raw)
    if not isinstance(child, dict) or child.get("status") != "passed":
        raise RuntimeError("child metrics do not declare status=passed")
    for key, expected in (
        ("dataset", job["dataset"]),
        ("condition", job["condition"]),
        ("model_seed", job["model_seed"]),
        ("evaluation_split", "validation"),
        ("test_evaluated", False),
    ):
        if child.get(key) != expected:
            raise RuntimeError(f"child {key} mismatch: expected {expected!r}")
    if "test" in child:
        raise RuntimeError("scaling child must not expose a test metric")
    configuration = child.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("child configuration is missing")
    for key, expected in job["architecture"].items():
        if configuration.get(key) != expected:
            raise RuntimeError(f"child architecture mismatch for {key}")
    validation = _number(child.get("validation"), "validation", minimum=0.0)
    if validation > 1:
        raise RuntimeError("validation metric must be at most one")
    trainable = child.get("trainable_parameters")
    total = child.get("total_parameters", trainable)
    if isinstance(trainable, bool) or not isinstance(trainable, int) or trainable < 1:
        raise RuntimeError("child trainable parameter count is invalid")
    if isinstance(total, bool) or not isinstance(total, int) or total < trainable:
        raise RuntimeError("child total parameter count is invalid")
    elapsed = _number(child.get("elapsed_seconds"), "elapsed_seconds", minimum=0.0)
    peak_memory = child.get("peak_cuda_allocated_bytes", child.get("peak_gpu_memory_bytes"))
    if isinstance(peak_memory, bool) or not isinstance(peak_memory, int) or peak_memory < 0:
        raise RuntimeError("child peak CUDA allocation is invalid")
    return {
        "metrics_sha256": hashlib.sha256(raw).hexdigest(),
        "validation": validation,
        "metric_name": child.get("metric_name"),
        "best_epoch": child.get("best_epoch"),
        "epochs_run": child.get("epochs_run"),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_memory,
        "actual_configuration": {
            key: configuration[key] for key in ("hidden_channels", "layers", "dropout")
        },
        "test_evaluated": False,
    }


def _aggregate(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        if job.get("status") == "passed" and isinstance(job.get("result"), dict):
            grouped[(job["version"], job["profile"], job["dataset"], job["condition"])].append(job)
    output = []
    for key in sorted(grouped):
        members = grouped[key]
        validations = [member["result"]["validation"] for member in members]
        elapsed = [member["result"]["elapsed_seconds"] for member in members]
        memories = [member["result"]["peak_cuda_allocated_bytes"] for member in members]
        output.append(
            {
                "version": key[0],
                "profile": key[1],
                "dataset": key[2],
                "condition": key[3],
                "architecture": members[0]["architecture"],
                "passed_seeds": sorted(member["model_seed"] for member in members),
                "n": len(members),
                "validation_mean": statistics.fmean(validations),
                "validation_sample_std": (
                    statistics.stdev(validations) if len(validations) > 1 else None
                ),
                "validation_min": min(validations),
                "validation_max": max(validations),
                "trainable_parameters": sorted(
                    {member["result"]["trainable_parameters"] for member in members}
                ),
                "total_parameters": sorted(
                    {member["result"]["total_parameters"] for member in members}
                ),
                "elapsed_seconds_mean": statistics.fmean(elapsed),
                "peak_cuda_allocated_bytes_max": max(memories),
            }
        )
    return output


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = manifest["jobs"]
    counts = {
        status: sum(job["status"] == status for job in jobs)
        for status in (
            "pending",
            "running",
            "passed",
            "failed",
        )
    }
    complete = manifest.get("status") == "passed" and counts == {
        "pending": 0,
        "running": 0,
        "passed": len(jobs),
        "failed": 0,
    }
    return {
        "schema_version": 1,
        "suite": "conductance_architecture_scaling_v1_v4",
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "valid_for_validation_comparison": complete
        and manifest.get("source_integrity_valid") is True,
        "test_evaluated": False,
        "selection_scope": "validation only; no profile is selected on test",
        "job_counts": counts,
        "expected_model_seeds": manifest["config"]["model_seeds"],
        "exclusions": manifest["exclusions"],
        "rows": _aggregate(jobs),
    }


def _write_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    summary = _summary(manifest)
    atomic_write_json(run_dir / "summary.json", summary)
    lines = [
        "# Conductance V1-V4 architecture scaling",
        "",
        f"- Status: `{summary['status']}`",
        f"- Validation comparison released: `{summary['valid_for_validation_comparison']}`",
        "- Test evaluated: `false`",
        "- Profiles: base 64x2, wide 128x2, deep 64x4, large 128x4; dropout 0.5",
        "- The table is descriptive validation scaling across independent model seeds.",
        "",
        "| Version | Profile | Dataset | Condition | n | Validation mean | Sample SD | "
        "Trainable parameters |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        deviation = (
            "N/A" if row["validation_sample_std"] is None else f"{row['validation_sample_std']:.6f}"
        )
        parameters = ", ".join(str(value) for value in row["trainable_parameters"])
        lines.append(
            f"| {row['version']} | {row['profile']} | {row['dataset']} | "
            f"{row['condition']} | {row['n']} | {row['validation_mean']:.6f} | "
            f"{deviation} | {parameters} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_id = args.run_id or "scaling-v1-v4-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    results_root = args.results_root.expanduser().resolve()
    run_dir = (results_root / "conductance_gat/scaling" / run_id).resolve()
    data_root = args.data_root.expanduser().resolve()
    if (
        run_dir == data_root
        or run_dir.is_relative_to(data_root)
        or data_root.is_relative_to(run_dir)
    ):
        print("Scaling outputs and dataset cache must not overlap", file=sys.stderr)
        return 2
    try:
        jobs = make_jobs(args, run_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    exclusions = _exclusions(args)
    if args.dry_run:
        print(
            f"{len(jobs)} validation-only fresh trainings; versions={args.versions}; "
            f"profiles={args.profiles}; seeds={args.model_seeds}"
        )
        for exclusion in exclusions:
            print(f"N/A: {exclusion['version']}/{exclusion['dataset']}: {exclusion['reason']}")
        for job in jobs:
            print(shlex.join(job["command"]))
        print(f"Manifest: {run_dir / 'manifest.json'}")
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
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite": "conductance_architecture_scaling_v1_v4",
        "run_id": run_id,
        "status": "running",
        "source_integrity_valid": True,
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config": {
            "versions": args.versions,
            "profiles": {name: PROFILES[name] for name in args.profiles},
            "datasets": args.datasets,
            "datasets_by_version": {
                version: _selected_datasets(args, version) for version in args.versions
            },
            "model_seeds": args.model_seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "workers": args.workers,
            "device": args.device,
            "edge_chunk_size": args.edge_chunk_size,
            "data_root": str(data_root),
        },
        "protocol": {
            "purpose": "architecture scale response, not parameter matching",
            "profiles": "base/wide/deep/large all run for every selected version",
            "selection": "best validation checkpoint within each independent child",
            "test": "never loaded into a V1 scaling loader and never evaluated by any child",
            "aggregation": "validation mean and sample standard deviation across model seeds",
            "release": "comparison valid only after every planned child and source check passes",
        },
        "exclusions": exclusions,
        "jobs": jobs,
        "dependencies": dependencies,
        "source_sha256": _source_snapshot(),
    }
    manifest_path = run_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    _write_summary(run_dir, manifest)
    current: dict[str, Any] | None = None
    environment = shared._environment()
    environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    try:
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
        status = shared.run_logged(preflight, run_dir / "logs/preflight.log", environment)
        if status:
            raise RuntimeError(f"GPU preflight failed with exit code {status}")
        print(f"Run: {run_id}; {len(jobs)} validation-only fresh trainings", flush=True)
        for index, job in enumerate(jobs, start=1):
            current = job
            _check_sources(manifest)
            job["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            _write_summary(run_dir, manifest)
            print(f"\n[{index}/{len(jobs)}] {job['job_id']}", flush=True)
            started = time.monotonic()
            status = shared.run_logged(job["command"], Path(job["log_path"]), environment)
            job.update(exit_code=status, elapsed_wall_seconds=time.monotonic() - started)
            _check_sources(manifest)
            if status:
                raise RuntimeError(f"{job['job_id']} failed with exit code {status}")
            job["result"] = _load_child(job)
            job["metrics_sha256"] = job["result"]["metrics_sha256"]
            job["status"] = "passed"
            current = None
            atomic_write_json(manifest_path, manifest)
            _write_summary(run_dir, manifest)
        _check_sources(manifest)
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
        _write_summary(run_dir, manifest)
        print(f"Failed: {manifest['error']}\nSaved partial results: {run_dir}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    atomic_write_json(manifest_path, manifest)
    _write_summary(run_dir, manifest)
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    print(f"Summary: {run_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
