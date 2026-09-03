"""Build a compact, integrity-checked comparison of the two V5 conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from chartgat.cache import atomic_write_bytes, atomic_write_json

from .protocol import COMPARISON_DESIGN, CONDITIONS, SUITE


class ComparisonIntegrityError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact(
    child: dict[str, Any], job: dict[str, Any], path_key: str, hash_key: str, filename: str
) -> None:
    output = Path(job["output_dir"]).expanduser().resolve()
    path_value, digest = child.get(path_key), child.get(hash_key)
    if not isinstance(path_value, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ComparisonIntegrityError(f"missing {path_key}/{hash_key} integrity metadata")
    path = Path(path_value).expanduser().resolve()
    if path != output / filename or not path.is_file() or _sha256(path) != digest:
        raise ComparisonIntegrityError(f"{path_key} path/hash integrity failed")


def _job_seed(job: dict[str, Any], manifest: dict[str, Any]) -> int:
    command = job.get("command")
    if isinstance(command, list) and "--model-seed" in command:
        index = command.index("--model-seed") + 1
        if index < len(command):
            return int(command[index])
    return int(manifest.get("config", {}).get("model_seed", 0))


def _validate_child(
    child: dict[str, Any], job: dict[str, Any], manifest: dict[str, Any], path: Path
) -> dict[str, Any]:
    if child.get("research_suite") != SUITE:
        raise ComparisonIntegrityError(f"invalid V5 child metrics: {path}")
    for key, expected in (
        ("dataset", job.get("dataset")),
        ("condition", job.get("condition")),
        ("model_seed", _job_seed(job, manifest)),
        ("evaluation_split", "validation"),
        ("test_evaluated", False),
    ):
        if child.get(key) != expected:
            raise ComparisonIntegrityError(f"job/child {key} mismatch: {path}")
    configuration = child.get("configuration")
    if not isinstance(configuration, dict):
        raise ComparisonIntegrityError(f"missing child configuration: {path}")
    if any(configuration.get(key) != value for key, value in job.get("architecture", {}).items()):
        raise ComparisonIntegrityError(f"job/child architecture mismatch: {path}")
    execution = job.get("execution")
    expected_batch = (
        execution.get("batch_size")
        if isinstance(execution, dict)
        else 2
        if child["dataset"] == "ppi"
        else manifest.get("config", {}).get("batch_size", 1)
    )
    if (
        configuration.get("sampling") != job.get("sampling", configuration.get("sampling"))
        or configuration.get("batch_size") != expected_batch
        or configuration.get("model_seed") != child["model_seed"]
    ):
        raise ComparisonIntegrityError(f"job/child training configuration mismatch: {path}")
    if isinstance(execution, dict):
        hardware = child.get("hardware_execution")
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
            raise ComparisonIntegrityError(f"job/child hardware execution mismatch: {path}")
    validation = child.get("validation")
    if (
        isinstance(validation, bool)
        or not isinstance(validation, (int, float))
        or not math.isfinite(validation)
        or not 0 <= validation <= 1
    ):
        raise ComparisonIntegrityError(f"nonfinite/out-of-range validation metric: {path}")
    for key in ("cache_sha256", "initial_state_sha256"):
        if not isinstance(child.get(key), str) or len(child[key]) != 64:
            raise ComparisonIntegrityError(f"invalid {key}: {path}")
    if not isinstance(child.get("source_sha256"), dict) or not child["source_sha256"]:
        raise ComparisonIntegrityError(f"missing implementation source hashes: {path}")
    if not isinstance(child.get("versions"), dict) or not child["versions"]:
        raise ComparisonIntegrityError(f"missing runtime versions: {path}")
    if child.get("protocol", {}).get("data_sha256") != child["cache_sha256"]:
        raise ComparisonIntegrityError(f"cache/protocol hash mismatch: {path}")
    identity, identity_hash = child.get("resume_identity"), child.get("resume_identity_sha256")
    if not isinstance(identity, dict) or _canonical_sha256(identity) != identity_hash:
        raise ComparisonIntegrityError(f"resume identity hash mismatch: {path}")
    for key in ("cache_sha256", "source_sha256", "initial_state_sha256"):
        if identity.get(key) != child.get(key):
            raise ComparisonIntegrityError(f"resume identity {key} mismatch: {path}")
    if child.get("comparison_design") != COMPARISON_DESIGN:
        raise ComparisonIntegrityError(f"comparison design metadata mismatch: {path}")
    group_steps = child.get("effective_optimizer_steps_by_group")
    if not isinstance(group_steps, dict) or any(
        isinstance(group_steps.get(name), bool)
        or not isinstance(group_steps.get(name), int)
        or group_steps[name] < 0
        for name in ("backbone", "spatial_w", "beta", "conductance")
    ):
        raise ComparisonIntegrityError(f"invalid effective group step counts: {path}")
    capacity = child.get("allocated_parameter_capacity")
    trainable = child.get("trainable_parameters")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (capacity, trainable)
    ):
        raise ComparisonIntegrityError(f"invalid parameter capacity: {path}")
    _artifact(child, job, "checkpoint", "checkpoint_sha256", "best.pt")
    _artifact(child, job, "last_checkpoint", "last_checkpoint_sha256", "last.pt")
    _artifact(child, job, "history", "history_sha256", "history.json")
    recheck = child.get("selected_checkpoint_recheck")
    if (
        not isinstance(recheck, dict)
        or recheck.get("non_gating") is not True
        or not all(
            math.isfinite(float(recheck.get(key))) for key in ("recorded", "recomputed", "delta")
        )
    ):
        raise ComparisonIntegrityError(f"invalid selected-checkpoint recheck: {path}")
    return {
        "dataset": child["dataset"],
        "condition": child["condition"],
        "validation": float(validation),
        "metric": child["metric_name"],
        "parameters": int(child["total_parameters"]),
        "allocated_parameter_capacity": capacity,
        "trainable_parameters": trainable,
        "best_epoch": int(child["best_epoch"]),
        "effective_optimizer_steps_by_group": group_steps,
        "cache_sha256": child["cache_sha256"],
        "source_sha256": child["source_sha256"],
        "runtime_versions": child["versions"],
        "initial_state_sha256": child["initial_state_sha256"],
        "configuration": configuration,
        "schedule": child.get("schedule"),
        "protocol": child["protocol"],
        "metrics_path": str(path.resolve()),
    }


def _metrics_path(run_dir: Path, job: dict[str, Any]) -> Path:
    for key in ("output_dir", "directory", "path"):
        if key in job:
            path = Path(job[key])
            if not path.is_absolute():
                path = run_dir / path
            return path / "metrics.json" if path.is_dir() or not path.suffix else path
    return run_dir / job["dataset"] / job["condition"] / "metrics.json"


def build_comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = manifest.get("jobs", [])
    rows, job_status = [], []
    job_keys = [(job.get("dataset"), job.get("condition")) for job in jobs]
    if len(job_keys) != len(set(job_keys)):
        raise ComparisonIntegrityError("duplicate dataset/condition job in V5 manifest")
    for job in jobs:
        path = _metrics_path(run_dir, job)
        if not path.exists():
            job_status.append(
                {
                    "dataset": job.get("dataset"),
                    "condition": job.get("condition"),
                    "status": job.get("status", "pending"),
                    "metrics_path": str(path),
                }
            )
            continue
        child = json.loads(path.read_text(encoding="utf-8"))
        status = child.get("status", job.get("status", "pending"))
        job_status.append(
            {
                "dataset": job.get("dataset"),
                "condition": job.get("condition"),
                "status": status,
                "metrics_path": str(path.resolve()),
            }
        )
        if status != "passed":
            continue
        rows.append(_validate_child(child, job, manifest, path))
    indexed = {(row["dataset"], row["condition"]): row for row in rows}
    requested_datasets = manifest.get("config", {}).get("datasets")
    if not isinstance(requested_datasets, list):
        requested_datasets = sorted({job["dataset"] for job in jobs})
    datasets = sorted(requested_datasets)
    contrasts, complete = [], manifest.get("status") == "passed"
    expected_matrix = {(dataset, condition) for dataset in datasets for condition in CONDITIONS}
    if complete and set(job_keys) != expected_matrix:
        raise ComparisonIntegrityError(
            "passed manifest does not contain the exact dataset x 2 matrix"
        )
    for dataset in datasets:
        keys = [(dataset, condition) for condition in CONDITIONS]
        if not all(key in indexed for key in keys):
            if complete:
                raise ComparisonIntegrityError(f"incomplete fixed/dynamic pair for {dataset}")
            continue
        if not complete:
            continue
        fixed, dynamic = indexed[(dataset, "fixed_c")], indexed[(dataset, "shared_dynamic_c")]
        for field in (
            "cache_sha256",
            "source_sha256",
            "runtime_versions",
            "allocated_parameter_capacity",
            "initial_state_sha256",
            "configuration",
            "schedule",
            "protocol",
            "metric",
        ):
            if fixed[field] != dynamic[field]:
                raise ComparisonIntegrityError(f"{dataset}: fixed/dynamic {field} mismatch")
        contrasts.append(
            {
                "dataset": dataset,
                "metric": fixed["metric"],
                "fixed_c": fixed["validation"],
                "shared_dynamic_c": dynamic["validation"],
                "dynamic_minus_fixed": dynamic["validation"] - fixed["validation"],
                "comparison_design": COMPARISON_DESIGN,
                "fixed_effective_optimizer_steps_by_group": fixed[
                    "effective_optimizer_steps_by_group"
                ],
                "dynamic_effective_optimizer_steps_by_group": dynamic[
                    "effective_optimizer_steps_by_group"
                ],
            }
        )
    if complete and len(rows) != len(jobs):
        raise ComparisonIntegrityError("passed manifest does not contain all passed child metrics")
    return {
        "schema_version": 1,
        "status": "passed" if complete else "partial",
        "complete": complete,
        "research_suite": SUITE,
        "test_evaluated": False,
        "comparison_design": COMPARISON_DESIGN,
        "jobs": job_status,
        "rows": rows,
        "contrasts": contrasts,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Conductance V5 validation comparison",
        "",
        "Test labels were not evaluated.",
        "",
        "| Dataset | Metric | Fixed C | Dynamic C | Dynamic - fixed |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["contrasts"]:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['fixed_c']:.6f} | "
            f"{row['shared_dynamic_c']:.6f} | {row['dynamic_minus_fixed']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


def csv_text(report: dict[str, Any]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=list(report["contrasts"][0]) if report["contrasts"] else []
    )
    if report["contrasts"]:
        writer.writeheader()
        writer.writerows(report["contrasts"])
    return stream.getvalue()


def write_comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report = build_comparison(run_dir, manifest)
    atomic_write_json(run_dir / "comparison.json", report)
    atomic_write_bytes(run_dir / "comparison.md", markdown(report).encode("utf-8"))
    atomic_write_bytes(run_dir / "comparison.csv", csv_text(report).encode("utf-8"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = args.manifest or run_dir / "manifest.json"
    write_comparison(run_dir, json.loads(manifest_path.read_text(encoding="utf-8")))
    print(f"passed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
