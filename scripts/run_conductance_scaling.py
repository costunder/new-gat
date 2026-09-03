#!/usr/bin/env python3
"""Run validation-only reference-scale experiments for Conductance V1 through V5."""

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
from research.conductance_gat.v5.protocol import (  # noqa: E402
    BETA_PARAMETERIZATIONS,
    DEFAULT_BETA_INITIAL,
    DEFAULT_BETA_PARAMETERIZATION,
    HARDWARE_PROFILES,
    SCALE_PROFILES,
    beta_configuration,
)
from research.conductance_gat.v5.protocol import (  # noqa: E402
    CONDITIONS as V5_CONDITIONS,
)
from research.conductance_gat.v5.protocol import (  # noqa: E402
    DATASETS as V5_DATASETS,
)
from scripts import run_conductance_factorial as shared  # noqa: E402
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)

V1_DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
# V5 owns these profiles. Legacy versions receive only their supported
# width/depth/dropout subset; heads and FFN expansion remain V5-specific.
PROFILES: dict[str, dict[str, Any]] = {
    name: dict(configuration) for name, configuration in SCALE_PROFILES.items()
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
    "v5": {
        "module": "research.conductance_gat.v5.train",
        "datasets": tuple(V5_DATASETS),
        "conditions": tuple(V5_CONDITIONS),
    },
}
ALL_DATASETS = tuple(
    dict.fromkeys(dataset for spec in VERSIONS.values() for dataset in spec["datasets"])
)
DEFAULT_MODEL_SEEDS = (0,)


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
    result.add_argument("--v5-edge-chunk-size", type=int)
    result.add_argument("--v5-ppi-batch-size", type=int)
    result.add_argument(
        "--v5-beta-parameterization",
        choices=BETA_PARAMETERIZATIONS,
        default=DEFAULT_BETA_PARAMETERIZATION,
    )
    result.add_argument("--v5-beta-initial", type=float, default=DEFAULT_BETA_INITIAL)
    result.add_argument("--v5-beta-min", type=float)
    result.add_argument("--v5-beta-max", type=float)
    result.add_argument("--hardware-profile", choices=tuple(HARDWARE_PROFILES), default="portable")
    result.add_argument(
        "--v5-sampling",
        choices=("auto", "full", "neighbor", "cluster"),
        default="auto",
        help="V5 only: auto uses cluster sampling for ogbn-arxiv and full otherwise",
    )
    result.add_argument(
        "--v5-num-neighbors",
        nargs="+",
        type=int,
        default=[15, 10],
        help="V5 only: hop fanouts; first value is also the cluster budget hint",
    )
    result.add_argument("--v5-sample-seed-batch-size", type=int)
    result.add_argument(
        "--v5-activation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="V5 only: override the selected hardware profile's checkpoint policy",
    )
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
    if (
        not args.v5_num_neighbors
        or any(value < 1 for value in args.v5_num_neighbors)
        or (args.v5_sample_seed_batch_size is not None and args.v5_sample_seed_batch_size < 1)
        or (args.v5_edge_chunk_size is not None and args.v5_edge_chunk_size < 1)
        or (args.v5_ppi_batch_size is not None and args.v5_ppi_batch_size < 1)
    ):
        raise ValueError("V5 neighbor fanouts and sample seed batch size must be positive")
    if args.hardware_profile == "portable" and args.v5_ppi_batch_size not in {None, 2}:
        raise ValueError("portable V5 PPI retains graph batch-size 2")
    _v5_beta_configuration(args)
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


def _v5_execution(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    profile = HARDWARE_PROFILES[args.hardware_profile]
    batch_size = 1
    if dataset == "ppi":
        batch_size = args.v5_ppi_batch_size or profile["ppi_batch_size"]
    return {
        "hardware_profile": args.hardware_profile,
        "precision": profile["precision"],
        "tf32": profile["tf32"],
        "batch_size": batch_size,
        "sample_seed_batch_size": (
            args.v5_sample_seed_batch_size or profile["sample_seed_batch_size"]
        ),
        "edge_chunk_size": args.v5_edge_chunk_size or profile["edge_chunk_size"],
        "activation_checkpoint": (
            profile["activation_checkpoint"]
            if args.v5_activation_checkpoint is None
            else args.v5_activation_checkpoint
        ),
        "sample_prefetch": profile["sample_prefetch"],
        "pin_memory": profile["pin_memory"],
    }


def _v5_beta_configuration(args: argparse.Namespace) -> dict[str, float | str]:
    return beta_configuration(
        args.v5_beta_parameterization,
        args.v5_beta_initial,
        args.v5_beta_min,
        args.v5_beta_max,
    )


def _effective_min_free_gb(args: argparse.Namespace) -> float:
    return max(
        float(args.min_free_gb),
        float(HARDWARE_PROFILES[args.hardware_profile]["minimum_free_memory_gib"]),
    )


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
            full_profile = PROFILES[profile_name]
            profile_fields = (
                ("hidden_channels", "layers", "heads", "ffn_multiplier", "dropout")
                if version == "v5"
                else ("hidden_channels", "layers", "dropout")
            )
            profile = {key: full_profile[key] for key in profile_fields}
            if version == "v5":
                profile.update(_v5_beta_configuration(args))
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
                        execution = None
                        if version != "v1":
                            command += ["--condition", condition]
                        if version in {"v2", "v3", "v4"}:
                            command += ["--edge-chunk-size", str(args.edge_chunk_size)]
                        sampling = None
                        if version == "v5":
                            execution = _v5_execution(args, dataset)
                            batch_position = command.index("--batch-size") + 1
                            command[batch_position] = str(execution["batch_size"])
                            sampling = (
                                "cluster"
                                if args.v5_sampling == "auto" and dataset == "ogbn-arxiv"
                                else "full"
                                if args.v5_sampling == "auto"
                                else args.v5_sampling
                            )
                            if dataset == "ppi" and sampling != "full":
                                raise ValueError("V5 PPI is inductive and requires full sampling")
                            command += [
                                "--heads",
                                str(profile["heads"]),
                                "--ffn-multiplier",
                                str(profile["ffn_multiplier"]),
                                "--sampling",
                                sampling,
                                "--hardware-profile",
                                args.hardware_profile,
                                "--sample-seed-batch-size",
                                str(execution["sample_seed_batch_size"]),
                                "--edge-chunk-size",
                                str(execution["edge_chunk_size"]),
                                (
                                    "--activation-checkpoint"
                                    if execution["activation_checkpoint"]
                                    else "--no-activation-checkpoint"
                                ),
                                "--num-neighbors",
                                *(str(value) for value in args.v5_num_neighbors),
                            ]
                            for name, value in _v5_beta_configuration(args).items():
                                command += ["--" + name.replace("_", "-"), str(value)]
                        job_id = f"{version}/{profile_name}/model-seed-{seed}/{dataset}/{condition}"
                        jobs.append(
                            {
                                "job_id": job_id,
                                "version": version,
                                "profile": profile_name,
                                "architecture": dict(profile),
                                "sampling": sampling,
                                "execution": execution,
                                "occupancy_expectation": (
                                    "expected low occupancy; one small full graph/no minibatch axis"
                                    if dataset in {"cora", "citeseer", "pubmed"}
                                    else "dataset exposes a real graph/sample minibatch axis"
                                ),
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
    if args.hardware_profile == "a6000-48gb":
        # Child processes remain sequential to avoid unbounded multi-process CUDA
        # memory contention. Run the genuinely large V5 workloads first instead
        # of presenting Cora's unavoidable low utilization as the initial job.
        version_order = {"v5": 0, "v4": 1, "v3": 2, "v2": 3, "v1": 4}
        dataset_order = {"ogbn-arxiv": 0, "ppi": 1, "pubmed": 2, "citeseer": 3, "cora": 4}
        profile_order = {"reference": 0, "large": 1}
        jobs.sort(
            key=lambda job: (
                version_order[job["version"]],
                dataset_order[job["dataset"]],
                profile_order[job["profile"]],
                job["model_seed"],
                job["condition"],
            )
        )
    return jobs


def _source_snapshot() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "research/__init__.py",
        ROOT / "research/conductance_gat/__init__.py",
        ROOT / "src/chartgat/__init__.py",
    ]
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
    paths += list((ROOT / "research/conductance_gat/v5").glob("*.py"))
    paths += [
        ROOT / "src/chartgat/cache.py",
        ROOT / "src/chartgat/execution.py",
        ROOT / "scripts/check_dependencies.py",
        ROOT / "scripts/gpu_profiles.py",
        ROOT / "scripts/gpu_preflight.py",
        ROOT / "scripts/run_conductance_factorial.py",
        ROOT / "scripts/verify_gpu_lock.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def _check_sources(manifest: dict[str, Any]) -> None:
    if _source_snapshot() != manifest["source_sha256"]:
        manifest["source_integrity_valid"] = False
        raise RuntimeError("Conductance scaling source changed during execution")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hardware_requirements(hardware_profile: str) -> dict[str, int]:
    try:
        profile = HARDWARE_PROFILES[hardware_profile]
    except KeyError as exc:
        raise RuntimeError(f"unknown hardware profile: {hardware_profile}") from exc
    return {
        "minimum_total_memory_bytes": int(float(profile["minimum_total_memory_gib"]) * 1024**3),
        "minimum_free_memory_bytes": int(float(profile["minimum_free_memory_gib"]) * 1024**3),
        "minimum_compute_capability_major": int(profile["minimum_compute_capability_major"]),
    }


def _validate_hardware_preflight(payload: Any, hardware_profile: str) -> dict[str, Any]:
    """Validate and normalize the preflight certificate before any child starts."""
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise RuntimeError("GPU preflight returned without a passed certificate")
    gpu = payload.get("gpu")
    if not isinstance(gpu, dict):
        raise RuntimeError("GPU preflight is missing the visible GPU record")
    name = gpu.get("name")
    total_bytes = gpu.get("total_bytes")
    free_bytes = gpu.get("free_bytes")
    capability = gpu.get("compute_capability")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("GPU preflight device name must be a nonempty string")
    if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 1:
        raise RuntimeError("GPU preflight total_bytes must be a positive integer")
    if (
        isinstance(free_bytes, bool)
        or not isinstance(free_bytes, int)
        or free_bytes < 0
        or free_bytes > total_bytes
    ):
        raise RuntimeError("GPU preflight free_bytes must be an integer within device capacity")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or any(value < 0 for value in capability)
    ):
        raise RuntimeError("GPU preflight compute_capability must be two nonnegative integers")
    requirements = _hardware_requirements(hardware_profile)
    if total_bytes < requirements["minimum_total_memory_bytes"]:
        raise RuntimeError(
            f"{hardware_profile} requires at least "
            f"{requirements['minimum_total_memory_bytes'] / 1024**3:g} GiB of visible GPU "
            "memory; a smaller GPU or MIG slice must use the portable profile"
        )
    if free_bytes < requirements["minimum_free_memory_bytes"]:
        raise RuntimeError(
            f"{hardware_profile} requires at least "
            f"{requirements['minimum_free_memory_bytes'] / 1024**3:g} GiB free at "
            "preflight; wait for other GPU jobs or use the portable profile"
        )
    if capability[0] < requirements["minimum_compute_capability_major"]:
        raise RuntimeError(
            f"{hardware_profile} requires CUDA compute capability "
            f"{requirements['minimum_compute_capability_major']}.0 or newer"
        )
    return {
        "status": "passed",
        "hardware_profile": hardware_profile,
        "gpu": {
            "name": name,
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
            "compute_capability": capability,
        },
        "requirements": requirements,
    }


def _accepted_hardware_preflight(path: Path, hardware_profile: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"GPU preflight returned without its JSON certificate: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GPU preflight certificate is unreadable: {exc}") from exc
    accepted = _validate_hardware_preflight(payload, hardware_profile)
    return {"path": str(path), "sha256": _sha256(path), **accepted}


def _verify_preflight_evidence(
    manifest: dict[str, Any], run_dir: Path, hardware_profile: str
) -> None:
    evidence = manifest.get("gpu_preflight")
    jobs = manifest["jobs"]
    if evidence is None:
        if manifest.get("status") == "passed" or any(
            job.get("status") != "pending" for job in jobs
        ):
            raise RuntimeError("existing run has child state without accepted GPU preflight")
        return
    if not isinstance(evidence, dict):
        raise RuntimeError("existing GPU preflight evidence must be a JSON object")
    lexical = Path(os.path.abspath(Path(evidence.get("path", "")).expanduser()))
    resolved = lexical.resolve()
    expected = (run_dir / "gpu-preflight.json").resolve()
    if resolved != lexical or resolved != expected or not resolved.is_relative_to(run_dir):
        raise RuntimeError("existing GPU preflight evidence aliases or escapes the run")
    actual = _accepted_hardware_preflight(lexical, hardware_profile)
    if evidence != actual:
        raise RuntimeError("existing GPU preflight evidence or certificate hash changed")


_JOB_IDENTITY_KEYS = (
    "job_id",
    "version",
    "profile",
    "architecture",
    "execution",
    "occupancy_expectation",
    "model_seed",
    "dataset",
    "condition",
    "output_dir",
    "metrics_path",
    "log_path",
    "command",
)


def _job_identity(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job.get(key) for key in _JOB_IDENTITY_KEYS}


def _verify_passed_job(job: dict[str, Any]) -> None:
    actual = _load_child(job)
    if job.get("metrics_sha256") != actual["metrics_sha256"]:
        raise RuntimeError(f"passed job artifact hash mismatch: {job['job_id']}")
    if job.get("result") != actual:
        raise RuntimeError(f"passed job manifest result mismatch: {job['job_id']}")


def _load_resume_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    config: dict[str, Any],
    exclusions: list[dict[str, str]],
    jobs: list[dict[str, Any]],
    source_sha256: dict[str, str],
    dependencies: dict[str, Any],
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise RuntimeError("existing run directory has no manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("existing manifest must be a JSON object")
    for key, expected in (
        ("schema_version", 1),
        ("suite", "conductance_architecture_scaling_v1_v5"),
        ("run_id", run_id),
        ("config", config),
        ("exclusions", exclusions),
        ("source_sha256", source_sha256),
        ("source_integrity_valid", True),
        ("dependencies", dependencies),
    ):
        if manifest.get(key) != expected:
            raise RuntimeError(f"existing manifest {key} does not match this invocation")
    existing_jobs = manifest.get("jobs")
    if not isinstance(existing_jobs, list) or not all(
        isinstance(job, dict) for job in existing_jobs
    ):
        raise RuntimeError("existing manifest jobs are invalid")
    if [_job_identity(job) for job in existing_jobs] != [_job_identity(job) for job in jobs]:
        raise RuntimeError("existing manifest job plan does not match this invocation")
    if manifest.get("status") not in {"running", "failed", "passed"}:
        raise RuntimeError("existing manifest status is not resumable")
    run_dir = manifest_path.parent.resolve()
    for job in existing_jobs:
        if job.get("status") not in {"pending", "running", "failed", "passed"}:
            raise RuntimeError(f"existing job status is invalid: {job.get('job_id')}")
        lexical_output = Path(os.path.abspath(Path(job.get("output_dir", "")).expanduser()))
        lexical_metrics = Path(os.path.abspath(Path(job.get("metrics_path", "")).expanduser()))
        lexical_log = Path(os.path.abspath(Path(job.get("log_path", "")).expanduser()))
        output = lexical_output.resolve()
        metrics = lexical_metrics.resolve()
        log = lexical_log.resolve()
        if (
            output != lexical_output
            or metrics != lexical_metrics
            or log != lexical_log
            or not output.is_relative_to(run_dir)
            or metrics != output / "metrics.json"
            or not log.is_relative_to(run_dir / "logs")
        ):
            raise RuntimeError(
                f"existing job paths escape or indirectly alias the run: {job.get('job_id')}"
            )
        if job["status"] == "passed":
            _verify_passed_job(job)
    if manifest["status"] == "passed" and any(job["status"] != "passed" for job in existing_jobs):
        raise RuntimeError("passed manifest contains a non-passed job")
    _verify_preflight_evidence(manifest, run_dir, str(config["hardware_profile"]))
    return manifest


def _discard_incomplete_child(job: dict[str, Any], run_dir: Path) -> None:
    lexical_output = Path(os.path.abspath(Path(job["output_dir"]).expanduser()))
    output = lexical_output.resolve()
    run_dir = run_dir.resolve()
    if output != lexical_output:
        raise RuntimeError(f"refusing to clear an indirect child output path: {lexical_output}")
    if output == run_dir or not output.is_relative_to(run_dir):
        raise RuntimeError(f"refusing to clear child output outside run directory: {output}")
    if lexical_output.exists():
        if not lexical_output.is_dir():
            raise RuntimeError(f"child output path is not a directory: {lexical_output}")
        shutil.rmtree(lexical_output)
    for key in ("result", "metrics_sha256", "exit_code", "elapsed_wall_seconds", "error"):
        job.pop(key, None)


def _next_attempt_log(preferred: Path, run_dir: Path, label: str) -> Path:
    """Return a new in-run log path without overwriting an earlier attempt."""
    run_dir = run_dir.resolve()
    preferred = preferred.resolve()
    if not preferred.is_relative_to(run_dir):
        raise RuntimeError(f"log path resolves outside the run directory: {preferred}")
    if not preferred.exists():
        return preferred
    root = (run_dir / "logs/resume").resolve()
    if not root.is_relative_to(run_dir):
        raise RuntimeError("resume log directory resolves outside the run directory")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "--", label)
    attempt = 1
    while True:
        candidate = (root / f"{safe_label}.attempt-{attempt}.log").resolve()
        if not candidate.is_relative_to(run_dir):
            raise RuntimeError("resume log path resolves outside the run directory")
        if not candidate.exists():
            return candidate
        attempt += 1


def _validated_write_path(path: Path, run_dir: Path, *, label: str) -> Path:
    """Validate a direct regular-file path before any runner or child write."""
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
        raise RuntimeError(f"refusing to write an indirect or out-of-run {label}: {lexical}")
    return lexical


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
    if job["version"] == "v5":
        execution = job["execution"]
        expected_configuration = {
            "hardware_profile": execution["hardware_profile"],
            "precision": execution["precision"],
            "tf32": execution["tf32"],
            "batch_size": execution["batch_size"],
            "sample_seed_batch_size": execution["sample_seed_batch_size"],
            "edge_chunk_size": execution["edge_chunk_size"],
            "activation_checkpoint": execution["activation_checkpoint"],
            "sampling": job["sampling"],
            "sample_prefetch": execution["sample_prefetch"],
            "pin_memory": execution["pin_memory"],
        }
        if any(configuration.get(key) != value for key, value in expected_configuration.items()):
            raise RuntimeError("V5 child resolved execution configuration mismatch")
    validation = _number(child.get("validation"), "validation", minimum=0.0)
    if validation > 1:
        raise RuntimeError("validation metric must be at most one")
    trainable = child.get("trainable_parameters")
    total = child.get("total_parameters", trainable)
    if isinstance(trainable, bool) or not isinstance(trainable, int) or trainable < 1:
        raise RuntimeError("child trainable parameter count is invalid")
    if isinstance(total, bool) or not isinstance(total, int) or total < trainable:
        raise RuntimeError("child total parameter count is invalid")
    hardware_execution = child.get("hardware_execution")
    if job["version"] == "v5":
        execution = job["execution"]
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
        if not isinstance(hardware_execution, dict) or any(
            hardware_execution.get(key) != value for key, value in expected_hardware.items()
        ):
            raise RuntimeError("V5 child hardware execution metadata mismatch")
    elapsed = _number(child.get("elapsed_seconds"), "elapsed_seconds", minimum=0.0)
    peak_memory = child.get("peak_cuda_allocated_bytes", child.get("peak_gpu_memory_bytes"))
    if isinstance(peak_memory, bool) or not isinstance(peak_memory, int) or peak_memory < 0:
        raise RuntimeError("child peak CUDA allocation is invalid")
    peak_reserved = child.get("peak_cuda_reserved_bytes", peak_memory)
    if isinstance(peak_reserved, bool) or not isinstance(peak_reserved, int) or peak_reserved < 0:
        raise RuntimeError("child peak CUDA reservation is invalid")
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
        "peak_cuda_reserved_bytes": peak_reserved,
        "hardware_execution": hardware_execution,
        "throughput": child.get("throughput"),
        "actual_configuration": {key: configuration[key] for key in job["architecture"]},
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
        reservations = [member["result"]["peak_cuda_reserved_bytes"] for member in members]
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
                "peak_cuda_reserved_bytes_max": max(reservations),
                "hardware_execution": members[0]["result"].get("hardware_execution"),
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
        "suite": "conductance_architecture_scaling_v1_v5",
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
    summary_json = _validated_write_path(run_dir / "summary.json", run_dir, label="summary JSON")
    summary_markdown = _validated_write_path(
        run_dir / "summary.md", run_dir, label="summary Markdown"
    )
    atomic_write_json(summary_json, summary)
    lines = [
        "# Conductance V1-V5 reference-scale comparison",
        "",
        f"- Status: `{summary['status']}`",
        f"- Validation comparison released: `{summary['valid_for_validation_comparison']}`",
        "- Test evaluated: `false`",
        "- Profiles: reference 256x8x8-head and large 384x12x8-head; FFN expansion 4.",
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
    summary_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    run_id = args.run_id or "scaling-v1-v5-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    results_root = args.results_root.expanduser().resolve()
    run_dir = (results_root / "conductance_gat/scaling" / run_id).resolve()
    data_root = args.data_root.expanduser().resolve()
    if not run_dir.is_relative_to(results_root):
        print("Scaling outputs must stay within the results root", file=sys.stderr)
        return 2
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
    config = {
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
        "min_free_gb": args.min_free_gb,
        "edge_chunk_size": args.edge_chunk_size,
        "v5_edge_chunk_size": args.v5_edge_chunk_size,
        "v5_ppi_batch_size": args.v5_ppi_batch_size,
        "v5_beta": _v5_beta_configuration(args),
        "hardware_profile": args.hardware_profile,
        "effective_min_free_gb": _effective_min_free_gb(args),
        "v5_sampling": args.v5_sampling,
        "v5_num_neighbors": list(args.v5_num_neighbors),
        "v5_sample_seed_batch_size": args.v5_sample_seed_batch_size,
        "v5_activation_checkpoint": args.v5_activation_checkpoint,
        "v5_resolved_execution_by_dataset": {
            dataset: _v5_execution(args, dataset)
            for dataset in (_selected_datasets(args, "v5") if "v5" in args.versions else [])
        },
        "data_root": str(data_root),
    }
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
    manifest_path = run_dir / "manifest.json"
    try:
        manifest_path = _validated_write_path(manifest_path, run_dir, label="runner manifest")
        _validated_write_path(run_dir / "summary.json", run_dir, label="summary JSON")
        _validated_write_path(run_dir / "summary.md", run_dir, label="summary Markdown")
    except RuntimeError as exc:
        print(f"Refusing unsafe run state: {exc}", file=sys.stderr)
        return 1
    source_sha256 = _source_snapshot()
    try:
        dependencies = check_dependencies()
    except DependencyCheckError as exc:
        print(error_message(exc), file=sys.stderr)
        return exc.exit_code
    resuming = run_dir.exists()
    if resuming:
        try:
            manifest = _load_resume_manifest(
                manifest_path,
                run_id=run_id,
                config=config,
                exclusions=exclusions,
                jobs=jobs,
                source_sha256=source_sha256,
                dependencies=dependencies,
            )
        except Exception as exc:
            print(f"Refusing to resume: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        jobs = manifest["jobs"]
        if all(job["status"] == "passed" for job in jobs):
            manifest.update(status="passed", source_integrity_valid=True)
            manifest.pop("error", None)
            manifest.setdefault("finished_at_utc", dt.datetime.now(dt.UTC).isoformat())
            atomic_write_json(manifest_path, manifest)
            _write_summary(run_dir, manifest)
            print(f"Run already complete; verified {len(jobs)} passed child artifacts", flush=True)
            print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
            print(f"Summary: {run_dir / 'summary.md'}", flush=True)
            return 0
    if resuming:
        manifest.update(
            status="running",
            source_integrity_valid=True,
            resumed_at_utc=dt.datetime.now(dt.UTC).isoformat(),
            resume_count=int(manifest.get("resume_count", 0)) + 1,
        )
        manifest.pop("error", None)
        manifest.pop("finished_at_utc", None)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "suite": "conductance_architecture_scaling_v1_v5",
            "run_id": run_id,
            "status": "running",
            "source_integrity_valid": True,
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "config": config,
            "protocol": {
                "purpose": "architecture scale response, not parameter matching",
                "profiles": "reference/large run for every version; V5 additionally uses heads/FFN",
                "selection": "best validation checkpoint within each independent child",
                "test": "never loaded into a V1 scaling loader and never evaluated by any child",
                "aggregation": "validation mean and sample standard deviation across model seeds",
                "release": (
                    "comparison valid only after every planned child and source check passes"
                ),
                "hardware_execution": (
                    "portable preserves legacy execution; a6000-48gb changes V5's explicit "
                    "optimization/precision recipe, runs large V5 jobs first, and keeps child "
                    "processes sequential to avoid unsafe CUDA memory contention. Portable and "
                    "A6000 metrics are not directly comparable; V5 fixed/dynamic interpretation "
                    "requires the same profile. V1-V5 PPI remains descriptive because V5 uses "
                    "profile-specific real graph batches while legacy versions retain batch 2"
                ),
                "small_graph_limit": (
                    "single Cora/Citeseer/PubMed full graphs cannot fill a 48 GiB GPU; no "
                    "duplicate examples or dummy compute are introduced"
                ),
            },
            "exclusions": exclusions,
            "jobs": jobs,
            "dependencies": dependencies,
            "source_sha256": source_sha256,
        }
    atomic_write_json(manifest_path, manifest)
    _write_summary(run_dir, manifest)
    current: dict[str, Any] | None = None
    environment = shared._environment()
    environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    try:
        preflight_output = _validated_write_path(
            run_dir / "gpu-preflight.json", run_dir, label="GPU preflight output"
        )
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
            str(preflight_output),
        ]
        preflight_log = _next_attempt_log(run_dir / "logs/preflight.log", run_dir, "preflight")
        manifest["preflight_log_path"] = str(preflight_log)
        status = shared.run_logged(preflight, preflight_log, environment)
        if status:
            raise RuntimeError(f"GPU preflight failed with exit code {status}")
        manifest["gpu_preflight"] = _accepted_hardware_preflight(
            preflight_output, args.hardware_profile
        )
        atomic_write_json(manifest_path, manifest)
        remaining = sum(job["status"] != "passed" for job in jobs)
        print(
            f"Run: {run_id}; {remaining}/{len(jobs)} validation-only fresh trainings remaining",
            flush=True,
        )
        for index, job in enumerate(jobs, start=1):
            if job["status"] == "passed":
                print(f"\n[{index}/{len(jobs)}] verified, skipping {job['job_id']}", flush=True)
                continue
            current = job
            _check_sources(manifest)
            attempt_command = list(job["command"])
            last_checkpoint = Path(job["output_dir"]) / "last.pt"
            if job["version"] == "v5" and last_checkpoint.is_file():
                attempt_command.append("--resume")
            else:
                _discard_incomplete_child(job, run_dir)
            job["attempt_command"] = attempt_command
            job["status"] = "running"
            atomic_write_json(manifest_path, manifest)
            _write_summary(run_dir, manifest)
            print(f"\n[{index}/{len(jobs)}] {job['job_id']}", flush=True)
            if "expected low occupancy" in job["occupancy_expectation"]:
                print(job["occupancy_expectation"], flush=True)
            started = time.monotonic()
            attempt_log = _next_attempt_log(Path(job["log_path"]), run_dir, str(job["job_id"]))
            job["attempt_log_path"] = str(attempt_log)
            status = shared.run_logged(attempt_command, attempt_log, environment)
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
