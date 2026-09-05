#!/usr/bin/env python3
"""Run the complete Conductance, Cycle PE, and Tree larger-model scaling suites."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.resume_compat import (  # noqa: E402
    COMPATIBILITY_SOURCE_FILES,
    adopt_source_snapshot,
    snapshots_match,
)
from research.conductance_gat.v5.protocol import (  # noqa: E402
    BETA_PARAMETERIZATIONS,
    DEFAULT_BETA_INITIAL,
    DEFAULT_BETA_PARAMETERIZATION,
    beta_configuration,
)
from scripts.process_safety import (  # noqa: E402
    close_owned_child_stdout,
    run_failure_reporter,
    terminate_owned_child,
    terminate_owned_child_after_error,
)
from scripts.training_resource_plan import (  # noqa: E402
    load_resource_plan,
    resource_plan_identity,
)
from scripts.training_resource_plan import (  # noqa: E402
    source_snapshot as calibration_source_snapshot,
)

TRACKS = ("conductance", "cycle", "tree")
PROFILES = ("reference", "large")
HARDWARE_PROFILES = ("portable", "a6000-48gb")
CYCLE_V2_BASIS_BACKENDS = ("dfs_fundamental",)
DEFAULT_MODEL_SEEDS = (0,)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}")

_ACTIVE_CHILDREN_LOCK = threading.Lock()
_ACTIVE_CHILDREN: dict[subprocess.Popen[str], tuple[tuple[str, ...], Path]] = {}
_STOP_ACTIVE_CHILDREN = threading.Event()

# Keep the complete public matrices here instead of trusting child row counts.  The
# central runner passes these selections explicitly and verifies the same Cartesian
# products in every returned summary.
CONDUCTANCE_MATRIX = {
    "v1": {
        "datasets": ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"),
        "conditions": ("conductance",),
    },
    "v2": {
        "datasets": ("cora", "citeseer", "pubmed", "ogbn-arxiv"),
        "conditions": ("direct_c", "fixed_c"),
    },
    "v3": {
        "datasets": ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"),
        "conditions": ("relative_c", "fixed_c"),
    },
    "v4": {
        "datasets": ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"),
        "conditions": (
            "fixed_c_identity_w",
            "relative_c_identity_w",
            "fixed_c_spatial_w",
            "relative_c_spatial_w",
        ),
    },
    "v5": {
        "datasets": ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"),
        "conditions": ("fixed_c", "shared_dynamic_c"),
    },
}
CYCLE_VERSIONS = ("v1", "v2")
CYCLE_DATASETS = ("zinc12k", "peptides_struct")
TREE_SUITES = ("csl", "zinc")
TREE_MODELS = ("fixed_bfs", "multi_chart")
TRACK_SPECS = {
    "conductance": {
        "script": "run_conductance_scaling.py",
        "results_subdir": "conductance_gat/scaling",
    },
    "cycle": {
        "script": "run_cycle_scaling.py",
        "results_subdir": "cycle_pe/scaling",
    },
    "tree": {
        "script": "run_tree_scaling.py",
        "results_subdir": "tree_augmentation/scaling",
    },
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    result.add_argument(
        "--conductance-versions",
        nargs="+",
        choices=tuple(CONDUCTANCE_MATRIX),
        default=list(CONDUCTANCE_MATRIX),
    )
    result.add_argument(
        "--cycle-versions",
        nargs="+",
        choices=CYCLE_VERSIONS,
        default=list(CYCLE_VERSIONS),
    )
    result.add_argument(
        "--cycle-v2-encodings", nargs="+", choices=("se", "pe"), default=["se", "pe"]
    )
    result.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    result.add_argument("--model-seeds", nargs="+", type=int, default=list(DEFAULT_MODEL_SEEDS))
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--run-id")
    device = result.add_mutually_exclusive_group()
    device.add_argument(
        "--device",
        default="cuda",
        help="one CUDA device (backward-compatible single-device execution)",
    )
    device.add_argument(
        "--devices",
        nargs="+",
        help=(
            "explicit distinct CUDA devices for concurrent independent track children; "
            "tracks are assigned round-robin and each child remains single-device"
        ),
    )
    result.add_argument(
        "--hardware-profile",
        choices=HARDWARE_PROFILES,
        default="portable",
        help=(
            "portable and a6000-48gb select preregistered child resource settings; these are "
            "not measured throughput optima. The a6000-48gb profile fails closed below "
            "40 GiB visible VRAM or compute capability 8.0"
        ),
    )
    result.add_argument(
        "--conductance-legacy-ppi-batch-size",
        type=int,
        help=(
            "explicit V1/V3/V4 PPI graph-batch override forwarded to Conductance; use only "
            "after exact-version/profile measurement"
        ),
    )
    result.add_argument(
        "--conductance-v5-ppi-batch-size",
        type=int,
        help=(
            "explicit V5 PPI graph-batch override forwarded to Conductance; use only after an "
            "exact-configuration measurement because it changes the optimization recipe"
        ),
    )
    result.add_argument(
        "--conductance-v5-sample-seed-batch-size",
        type=int,
        help=(
            "explicit V5 sampled seed-node batch override forwarded to Conductance; use only "
            "after an exact-configuration measurement"
        ),
    )
    result.add_argument(
        "--cycle-batch-size",
        type=int,
        help=(
            "explicit physical graph-batch override forwarded to Cycle; one value applies to "
            "every requested Cycle version/dataset/profile and is not claimed as measured"
        ),
    )
    result.add_argument(
        "--tree-batch-size",
        type=int,
        help=(
            "explicit physical chart-view batch override forwarded to Tree; one value applies "
            "to every requested Tree suite/profile and is not claimed as measured"
        ),
    )
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument(
        "--resource-plan",
        type=Path,
        help=(
            "reuse an exact measured V5/Cycle V2 resource plan; "
            "omission calibrates before new training"
        ),
    )
    result.add_argument(
        "--v5-beta-parameterization",
        choices=BETA_PARAMETERIZATIONS,
        default=DEFAULT_BETA_PARAMETERIZATION,
    )
    result.add_argument("--v5-beta-initial", type=float, default=DEFAULT_BETA_INITIAL)
    result.add_argument("--v5-beta-min", type=float)
    result.add_argument("--v5-beta-max", type=float)
    result.add_argument(
        "--v5-activation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "V5 only: explicitly override the conductance hardware profile's block "
            "checkpoint policy; omission preserves that profile's default"
        ),
    )
    result.add_argument(
        "--cycle-v2-basis-backend",
        choices=CYCLE_V2_BASIS_BACKENDS,
        default="dfs_fundamental",
        help="Cycle V2 uses all sparse DFS cycles without QR/SVD",
    )
    result.add_argument("--allow-download", action="store_true")
    result.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop before later tracks after the first failed or unverifiable child",
    )
    result.add_argument("--dry-run", action="store_true", help="Print the plan without writes")
    return result


def _validate(args: argparse.Namespace) -> None:
    for label, values in (
        ("tracks", args.tracks),
        ("conductance versions", args.conductance_versions),
        ("cycle versions", args.cycle_versions),
        ("cycle V2 encodings", args.cycle_v2_encodings),
        ("profiles", args.profiles),
        ("model seeds", args.model_seeds),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{label} must be nonempty and contain no duplicates")
    if any(seed < 0 for seed in args.model_seeds):
        raise ValueError("model seeds must be nonnegative")
    devices = _execution_devices(args)
    if any(not re.fullmatch(r"cuda(?::[0-9]+)?", value) for value in devices):
        raise ValueError("rich scaling requires CUDA devices; CPU fallback is not supported")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct CUDA devices")
    if len(devices) > 1 and any(value == "cuda" for value in devices):
        raise ValueError("multi-device execution requires explicit indexed devices such as cuda:0")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("minimum free GPU memory must be finite and nonnegative")
    batch_overrides = {
        "--conductance-legacy-ppi-batch-size": args.conductance_legacy_ppi_batch_size,
        "--conductance-v5-ppi-batch-size": args.conductance_v5_ppi_batch_size,
        "--conductance-v5-sample-seed-batch-size": (args.conductance_v5_sample_seed_batch_size),
        "--cycle-batch-size": args.cycle_batch_size,
        "--tree-batch-size": args.tree_batch_size,
    }
    for option, value in batch_overrides.items():
        if value is not None and value < 1:
            raise ValueError(f"{option} must be positive")
    if args.conductance_legacy_ppi_batch_size is not None and (
        "conductance" not in args.tracks
        or not {"v1", "v3", "v4"}.intersection(args.conductance_versions)
    ):
        raise ValueError("legacy PPI batch override requires Conductance V1, V3, or V4")
    if (
        args.conductance_v5_ppi_batch_size is not None
        or args.conductance_v5_sample_seed_batch_size is not None
    ) and ("conductance" not in args.tracks or "v5" not in args.conductance_versions):
        raise ValueError("Conductance V5 batch overrides require the conductance/V5 track")
    if (
        args.hardware_profile == "portable"
        and args.conductance_v5_ppi_batch_size is not None
        and args.conductance_v5_ppi_batch_size < 2
    ):
        raise ValueError("portable Conductance V5 PPI cannot shrink below graph batch-size 2")
    if args.cycle_batch_size is not None and "cycle" not in args.tracks:
        raise ValueError("--cycle-batch-size requires the cycle track")
    if args.tree_batch_size is not None and "tree" not in args.tracks:
        raise ValueError("--tree-batch-size requires the tree track")
    if args.run_id is not None and RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID must be 1-120 letters, digits, underscores, or hyphens")
    if (
        "cycle" in args.tracks
        and args.cycle_v2_basis_backend != "dfs_fundamental"
        and "v2" not in args.cycle_versions
    ):
        raise ValueError("nondefault Cycle basis backend requires v2 in --cycle-versions")
    _v5_beta_configuration(args)


def _execution_devices(args: argparse.Namespace) -> list[str]:
    """Return explicit execution devices without probing or changing visibility."""

    return list(args.devices) if args.devices is not None else [args.device]


def _default_run_id() -> str:
    return "rich-scaling-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _child_run_id(run_id: str, track: str) -> str:
    candidate = f"{run_id}-{track}"
    if len(candidate) <= 120:
        return candidate
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    prefix_length = 120 - len(track) - len(digest) - 2
    return f"{run_id[:prefix_length]}-{track}-{digest}"


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _requested_matrix(
    track: str,
    profiles: list[str],
    model_seeds: list[int],
    *,
    conductance_versions: list[str],
    cycle_versions: list[str],
    cycle_v2_encodings: list[str] | tuple[str, ...] = ("se", "pe"),
) -> dict[str, Any]:
    common = {"profiles": list(profiles), "model_seeds": list(model_seeds)}
    if track == "conductance":
        selected = {version: CONDUCTANCE_MATRIX[version] for version in conductance_versions}
        requested_datasets = list(
            dict.fromkeys(dataset for spec in selected.values() for dataset in spec["datasets"])
        )
        return {
            **common,
            "versions": list(selected),
            "requested_datasets": requested_datasets,
            "datasets_by_version": {
                version: list(spec["datasets"]) for version, spec in selected.items()
            },
            "conditions_by_version": {
                version: list(spec["conditions"]) for version, spec in selected.items()
            },
        }
    if track == "cycle":
        return {
            **common,
            "versions": list(cycle_versions),
            "encodings_by_version": {
                version: list(cycle_v2_encodings) if version == "v2" else [None]
                for version in cycle_versions
            },
            "datasets": list(CYCLE_DATASETS),
        }
    return {
        **common,
        "suites": list(TREE_SUITES),
        "models": list(TREE_MODELS),
    }


def _expected_counts(track: str, matrix: dict[str, Any]) -> dict[str, int]:
    combinations = len(matrix["profiles"]) * len(matrix["model_seeds"])
    if track == "conductance":
        trainings_per_combination = sum(
            len(matrix["datasets_by_version"][version])
            * len(matrix["conditions_by_version"][version])
            for version in matrix["versions"]
        )
        return {
            "child_runs": combinations * trainings_per_combination,
            "model_trainings": combinations * trainings_per_combination,
        }
    if track == "cycle":
        condition_count = sum(
            len(matrix["encodings_by_version"][version]) for version in matrix["versions"]
        )
        child_runs = combinations * condition_count * len(matrix["datasets"])
        return {
            "child_runs": child_runs,
            "model_trainings": child_runs,
        }
    child_runs = combinations * len(matrix["suites"])
    return {
        "child_runs": child_runs,
        "model_trainings": child_runs * len(TREE_MODELS),
    }


def _v5_beta_configuration(args: argparse.Namespace) -> dict[str, float | str]:
    return beta_configuration(
        args.v5_beta_parameterization,
        args.v5_beta_initial,
        args.v5_beta_min,
        args.v5_beta_max,
    )


def make_jobs(args: argparse.Namespace, run_id: str) -> list[dict[str, Any]]:
    """Build one child job per track; execution waves bind at most one track per GPU."""
    results_root = args.results_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    jobs: list[dict[str, Any]] = []
    devices = _execution_devices(args)
    for track_index, track in enumerate(args.tracks):
        spec = TRACK_SPECS[track]
        child_run_id = _child_run_id(run_id, track)
        child_dir = (results_root / spec["results_subdir"] / child_run_id).resolve()
        profiles = list(args.profiles)
        requested_matrix = _requested_matrix(
            track,
            profiles,
            list(args.model_seeds),
            conductance_versions=list(args.conductance_versions),
            cycle_versions=list(args.cycle_versions),
            cycle_v2_encodings=list(args.cycle_v2_encodings),
        )
        command = [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / spec["script"]),
        ]
        if track == "tree":
            command += [
                "--suites",
                ",".join(requested_matrix["suites"]),
                "--profiles",
                ",".join(profiles),
                "--model-seeds",
                ",".join(str(seed) for seed in args.model_seeds),
            ]
            if args.tree_batch_size is not None:
                command += ["--batch-size", str(args.tree_batch_size)]
        elif track == "cycle":
            command += ["--versions", *requested_matrix["versions"]]
            command += ["--datasets", *requested_matrix["datasets"]]
            command += ["--profiles", *profiles]
            command += ["--model-seeds", ",".join(str(seed) for seed in args.model_seeds)]
            command += ["--basis-backend", args.cycle_v2_basis_backend]
            command += ["--encodings", *args.cycle_v2_encodings]
            if (
                args.cycle_batch_size is not None
                and getattr(args, "resolved_resource_plan", None) is None
            ):
                command += ["--batch-size", str(args.cycle_batch_size)]
            elif args.cycle_batch_size is not None and "v1" in args.cycle_versions:
                command += ["--legacy-batch-size", str(args.cycle_batch_size)]
        else:
            command += ["--versions", *requested_matrix["versions"]]
            command += ["--datasets", *requested_matrix["requested_datasets"]]
            command += ["--profiles", *profiles]
            command += ["--model-seeds", *(str(seed) for seed in args.model_seeds)]
            for name, value in _v5_beta_configuration(args).items():
                command += ["--v5-" + name.replace("_", "-"), str(value)]
            if args.conductance_legacy_ppi_batch_size is not None:
                command += [
                    "--legacy-ppi-batch-size",
                    str(args.conductance_legacy_ppi_batch_size),
                ]
            if (
                args.conductance_v5_ppi_batch_size is not None
                and getattr(args, "resolved_resource_plan", None) is None
            ):
                command += [
                    "--v5-ppi-batch-size",
                    str(args.conductance_v5_ppi_batch_size),
                ]
            if (
                args.conductance_v5_sample_seed_batch_size is not None
                and getattr(args, "resolved_resource_plan", None) is None
            ):
                command += [
                    "--v5-sample-seed-batch-size",
                    str(args.conductance_v5_sample_seed_batch_size),
                ]
            if args.v5_activation_checkpoint is not None:
                command.append(
                    "--v5-activation-checkpoint"
                    if args.v5_activation_checkpoint
                    else "--no-v5-activation-checkpoint"
                )
        assigned_device = devices[track_index % len(devices)]
        command += [
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
            "--run-id",
            child_run_id,
            "--device",
            assigned_device,
            "--hardware-profile",
            args.hardware_profile,
            "--min-free-gb",
            str(args.min_free_gb),
        ]
        # Conductance scaling intentionally consumes only pre-verified offline caches and its
        # child CLI therefore has no --allow-download option.
        download_forwarded = args.allow_download and track in {"cycle", "tree"}
        if download_forwarded:
            command.append("--allow-download")
        if getattr(args, "resolved_resource_plan", None) is not None and (
            (track == "conductance" and "v5" in args.conductance_versions)
            or (track == "cycle" and "v2" in args.cycle_versions)
        ):
            command += ["--resource-plan", str(args.resource_plan)]
        if args.dry_run:
            command.append("--dry-run")
        jobs.append(
            {
                "track": track,
                "device": assigned_device,
                "child_run_id": child_run_id,
                "status": "pending",
                "profiles": profiles,
                "shared_profiles": list(args.profiles),
                "model_seeds": list(args.model_seeds),
                "command": command,
                "output_dir": str(child_dir),
                "summary_path": str(child_dir / "summary.json"),
                "log_path": str(results_root / "rich_scaling" / run_id / "logs" / f"{track}.log"),
                "allow_download_forwarded": download_forwarded,
                "requested_matrix": requested_matrix,
                "expected_counts": _expected_counts(track, requested_matrix),
            }
        )
    return jobs


def _source_snapshot() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/training_resource_plan.py",
        ROOT / "scripts/calibrate_training_resources.py",
        ROOT / "scripts/calibration_lock.py",
        ROOT / "research/__init__.py",
        ROOT / "scripts/check_dependencies.py",
        ROOT / "scripts/gpu_profiles.py",
        ROOT / "scripts/gpu_preflight.py",
        ROOT / "scripts/process_safety.py",
        ROOT / "scripts/run_conductance_factorial.py",
        ROOT / "scripts/telemetry_validation.py",
        ROOT / "scripts/verify_gpu_lock.py",
        ROOT / "research/tree_augmentation/config.yaml",
    ]
    paths.extend(ROOT / name for name in COMPATIBILITY_SOURCE_FILES)
    paths.extend(ROOT / "scripts" / TRACK_SPECS[track]["script"] for track in TRACKS)
    for source_root in (
        ROOT / "research/conductance_gat",
        ROOT / "research/cycle_pe",
        ROOT / "research/tree_augmentation",
        ROOT / "src/chartgat",
    ):
        for pattern in ("*.py", "*.yaml", "*.yml"):
            paths.extend(
                path
                for path in source_root.rglob(pattern)
                if "tests" not in path.relative_to(source_root).parts
            )
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def _check_central_sources(manifest: dict[str, Any]) -> None:
    try:
        current = _source_snapshot()
    except Exception as error:
        manifest["source_integrity_valid"] = False
        raise RuntimeError(f"could not re-hash central scaling sources: {error}") from error
    manifest["source_sha256_final"] = current
    if current != manifest["source_sha256"]:
        manifest["source_integrity_valid"] = False
        raise RuntimeError("central scaling source changed while child experiments were running")
    manifest["source_integrity_valid"] = True


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    entries = [str(ROOT / "src"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    return environment


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as original_error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            original_error.add_note(
                "temporary manifest cleanup failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def _persist_manifest_after_error(
    manifest_path: Path,
    manifest: dict[str, Any],
    original_error: BaseException,
    *,
    action: str,
) -> None:
    note = run_failure_reporter(
        lambda: _atomic_write_json(manifest_path, manifest),
        original_error=original_error,
        action=action,
    )
    if note is not None:
        manifest.setdefault("failure_persistence_errors", []).append(note)


def _register_active_child(
    process: subprocess.Popen[str], command: list[str], log_path: Path
) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        if process in _ACTIVE_CHILDREN:
            raise RuntimeError("child process was registered more than once")
        _ACTIVE_CHILDREN[process] = (tuple(command), log_path)


def _forget_active_child(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN.pop(process, None)


def _stop_active_children(
    *, reason: str, original_error: BaseException | None = None
) -> list[dict[str, object]]:
    """Stop only exact live children created and registered by this runner."""

    with _ACTIVE_CHILDREN_LOCK:
        active = list(_ACTIVE_CHILDREN.items())
    recorded: list[dict[str, object]] = []
    for process, (command, log_path) in active:
        if original_error is None:
            events = terminate_owned_child(process, command, reason=reason, log_target=log_path)
        else:
            events = terminate_owned_child_after_error(
                process,
                command,
                original_error=original_error,
                log_target=log_path,
            )
        recorded.extend({**event, "log_path": str(log_path)} for event in events)
    return recorded


def _run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if log_path.exists() else "x"
    with log_path.open(mode, encoding="utf-8", newline="\n") as log:
        if mode == "a":
            log.write(f"\n=== resumed {dt.datetime.now(dt.UTC).isoformat()} ===\n")
            log.flush()
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
            shell=False,
        )
        _register_active_child(process, command, log_path)
        primary_error: BaseException | None = None
        try:
            if _STOP_ACTIVE_CHILDREN.is_set():
                terminate_owned_child(
                    process,
                    command,
                    reason="central rich-scaling coordinator already requested interruption",
                    log_target=log,
                )
                return process.wait()
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException as error:
            primary_error = error
            terminate_owned_child_after_error(
                process,
                command,
                original_error=error,
                log_target=log,
            )
            raise
        finally:
            _forget_active_child(process)
            close_owned_child_stdout(process, original_error=primary_error)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"child summary {label} must be a nonnegative integer")
    return value


def _exact_key_matrix(
    rows: Any,
    *,
    fields: tuple[str, ...],
    integer_fields: frozenset[str],
    expected: set[tuple[Any, ...]],
    label: str,
    nullable_fields: frozenset[str] = frozenset(),
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} must be a list")
    observed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{label}[{index}] must be an object")
        values: list[Any] = []
        for field in fields:
            value = row.get(field)
            if field in integer_fields:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RuntimeError(f"{label}[{index}].{field} must be a nonnegative integer")
            elif value is None and field in nullable_fields and field in row:
                pass
            elif not isinstance(value, str):
                raise RuntimeError(f"{label}[{index}].{field} must be a string")
            values.append(value)
        key = tuple(values)
        if key in observed:
            raise RuntimeError(f"{label} contains duplicate key {key!r}")
        observed[key] = row
    observed_keys = set(observed)
    if observed_keys != expected:
        missing = sorted(expected - observed_keys, key=repr)
        unexpected = sorted(observed_keys - expected, key=repr)
        raise RuntimeError(
            f"{label} matrix mismatch; missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )
    return observed


def _exact_string_mapping_keys(value: Any, expected: list[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimeError(f"{label} keys do not exactly match the requested profiles")
    return value


def _validate_conductance_summary(payload: dict[str, Any], job: dict[str, Any]) -> dict[str, int]:
    if payload.get("suite") != "conductance_architecture_scaling_v1_v5":
        raise RuntimeError("Conductance child summary suite mismatch")
    if payload.get("run_id") != job["child_run_id"]:
        raise RuntimeError("Conductance child summary run ID mismatch")
    if payload.get("valid_for_validation_comparison") is not True:
        raise RuntimeError("Conductance child did not release a valid validation comparison")
    if payload.get("test_evaluated") is not False:
        raise RuntimeError("Conductance child summary does not certify test_evaluated=false")

    expected = job["expected_counts"]
    counts = payload.get("job_counts")
    if not isinstance(counts, dict):
        raise RuntimeError("Conductance child summary has no job counts")
    checked = {
        status: _integer(counts.get(status), f"job_counts.{status}")
        for status in ("pending", "running", "passed", "failed")
    }
    if checked != {
        "pending": 0,
        "running": 0,
        "passed": expected["model_trainings"],
        "failed": 0,
    }:
        raise RuntimeError("Conductance child summary job counts are incomplete")

    matrix = job["requested_matrix"]
    seeds = matrix["model_seeds"]
    if payload.get("expected_model_seeds") != seeds:
        raise RuntimeError("Conductance child summary model seeds mismatch")
    expected_rows = {
        (version, profile, dataset, condition)
        for version in matrix["versions"]
        for profile in matrix["profiles"]
        for dataset in matrix["datasets_by_version"][version]
        for condition in matrix["conditions_by_version"][version]
    }
    rows = _exact_key_matrix(
        payload.get("rows"),
        fields=("version", "profile", "dataset", "condition"),
        integer_fields=frozenset(),
        expected=expected_rows,
        label="Conductance summary rows",
    )
    expected_seeds = sorted(seeds)
    for key, row in rows.items():
        passed_seeds = row.get("passed_seeds")
        if (
            not isinstance(passed_seeds, list)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in passed_seeds)
            or passed_seeds != expected_seeds
        ):
            raise RuntimeError(f"Conductance summary row {key!r} has the wrong seed set")
        if _integer(row.get("n"), f"rows[{key!r}].n") != len(expected_seeds):
            raise RuntimeError(f"Conductance summary row {key!r} has the wrong seed count")

    expected_exclusions = {
        (version, dataset)
        for version in matrix["versions"]
        for dataset in matrix["requested_datasets"]
        if dataset not in matrix["datasets_by_version"][version]
    }
    exclusions = _exact_key_matrix(
        payload.get("exclusions"),
        fields=("version", "dataset"),
        integer_fields=frozenset(),
        expected=expected_exclusions,
        label="Conductance exclusions",
    )
    if any(row.get("status") != "not_applicable" for row in exclusions.values()):
        raise RuntimeError("Conductance exclusion rows must declare not_applicable")
    return {"child_runs": checked["passed"], "model_trainings": checked["passed"]}


def _validate_cycle_summary(payload: dict[str, Any], job: dict[str, Any]) -> dict[str, int]:
    if payload.get("scope") != "cycle_pe_v1_v2_larger_model_scaling":
        raise RuntimeError("Cycle child summary scope mismatch")
    matrix = job["requested_matrix"]
    if payload.get("requested_model_seeds") != matrix["model_seeds"]:
        raise RuntimeError("Cycle child summary model seeds mismatch")
    _exact_string_mapping_keys(payload.get("profiles"), matrix["profiles"], "Cycle profiles")
    if "v2" in matrix["versions"] and (
        payload.get("requested_encodings") != matrix["encodings_by_version"]["v2"]
    ):
        raise RuntimeError("Cycle child summary encoding selection mismatch")

    expected_runs = {
        (version, encoding, profile, seed, dataset)
        for version in matrix["versions"]
        for encoding in matrix["encodings_by_version"][version]
        for profile in matrix["profiles"]
        for seed in matrix["model_seeds"]
        for dataset in matrix["datasets"]
    }
    runs = _exact_key_matrix(
        payload.get("runs"),
        fields=("version", "encoding", "profile", "model_seed", "dataset"),
        integer_fields=frozenset({"model_seed"}),
        nullable_fields=frozenset({"encoding"}),
        expected=expected_runs,
        label="Cycle training rows",
    )
    expected_children = {
        (version, encoding, profile, seed, dataset)
        for version in matrix["versions"]
        for encoding in matrix["encodings_by_version"][version]
        for profile in matrix["profiles"]
        for seed in matrix["model_seeds"]
        for dataset in matrix["datasets"]
    }
    observed_children = set(runs)
    if observed_children != expected_children:
        raise RuntimeError("Cycle child-run matrix is incomplete")
    if any("test" in key.lower() for row in runs.values() for key in row):
        raise RuntimeError("Cycle candidate training rows must not contain test metrics")

    expected_aggregates = {
        (version, encoding, dataset, profile)
        for version in matrix["versions"]
        for encoding in matrix["encodings_by_version"][version]
        for dataset in matrix["datasets"]
        for profile in matrix["profiles"]
    }
    aggregates = _exact_key_matrix(
        payload.get("profile_aggregates"),
        fields=("version", "encoding", "dataset", "profile"),
        integer_fields=frozenset(),
        nullable_fields=frozenset({"encoding"}),
        expected=expected_aggregates,
        label="Cycle profile aggregates",
    )
    expected_seeds = sorted(matrix["model_seeds"])
    for key, row in aggregates.items():
        if row.get("model_seeds") != expected_seeds:
            raise RuntimeError(f"Cycle profile aggregate {key!r} has the wrong seed set")
        if any("test" in field.lower() for field in row):
            raise RuntimeError("Cycle validation aggregates must not contain test metrics")

    expected_profile_selections = {
        (version, encoding, dataset)
        for version in matrix["versions"]
        for encoding in matrix["encodings_by_version"][version]
        for dataset in matrix["datasets"]
    }
    profile_selections = _exact_key_matrix(
        payload.get("profile_selections"),
        fields=("version", "encoding", "dataset"),
        integer_fields=frozenset(),
        nullable_fields=frozenset({"encoding"}),
        expected=expected_profile_selections,
        label="Cycle validation profile selections",
    )
    for key, row in profile_selections.items():
        version, encoding, dataset = key
        condition_id = version if encoding is None else f"{version}:{encoding}"
        selection_id = f"{condition_id}:{dataset}"
        if (
            row.get("selected_profile") not in matrix["profiles"]
            or row.get("model_seeds") != matrix["model_seeds"]
            or row.get("test_used_for_selection") is not False
            or row.get("profile_selection_id") != selection_id
        ):
            raise RuntimeError(f"Cycle validation profile selection {key!r} is invalid")

    expected_checkpoints = {
        (version, encoding, dataset, seed)
        for version in matrix["versions"]
        for encoding in matrix["encodings_by_version"][version]
        for dataset in matrix["datasets"]
        for seed in matrix["model_seeds"]
    }
    selected_checkpoints = _exact_key_matrix(
        payload.get("selected_checkpoints"),
        fields=("version", "encoding", "dataset", "model_seed"),
        integer_fields=frozenset({"model_seed"}),
        nullable_fields=frozenset({"encoding"}),
        expected=expected_checkpoints,
        label="Cycle selected validation checkpoints",
    )
    for key, row in selected_checkpoints.items():
        profile_selection = profile_selections[key[:3]]
        selection_id = profile_selection["profile_selection_id"]
        if (
            row.get("selected_profile") != profile_selection.get("selected_profile")
            or row.get("profile_selection_id") != selection_id
            or row.get("checkpoint_id") != f"{selection_id}:model-seed-{key[3]}"
        ):
            raise RuntimeError(f"Cycle selected checkpoint {key!r} uses the wrong profile")

    test_rows = _exact_key_matrix(
        payload.get("test_evaluations"),
        fields=("version", "encoding", "dataset", "model_seed"),
        integer_fields=frozenset({"model_seed"}),
        nullable_fields=frozenset({"encoding"}),
        expected=expected_checkpoints,
        label="Cycle selected-checkpoint test evaluations",
    )
    for key, row in test_rows.items():
        if (
            row.get("selected_profile") != selected_checkpoints[key].get("selected_profile")
            or row.get("checkpoint") != selected_checkpoints[key].get("checkpoint")
            or row.get("checkpoint_sha256") != selected_checkpoints[key].get("checkpoint_sha256")
            or row.get("profile_selection_id") != selected_checkpoints[key]["profile_selection_id"]
            or row.get("checkpoint_id") != selected_checkpoints[key]["checkpoint_id"]
            or row.get("test_evaluation_id") != f"test:{selected_checkpoints[key]['checkpoint_id']}"
            or row.get("fresh_training") is not False
        ):
            raise RuntimeError(f"Cycle selected-checkpoint test evaluation {key!r} is invalid")
    if _integer(payload.get("fresh_dataset_trainings"), "fresh_dataset_trainings") != len(runs):
        raise RuntimeError("Cycle fresh training count disagrees with its exact candidate matrix")
    if _integer(payload.get("selected_test_evaluations"), "selected_test_evaluations") != len(
        test_rows
    ):
        raise RuntimeError("Cycle selected-checkpoint test count is incomplete")

    expected_final_aggregates = {
        (version, encoding, dataset)
        for version in matrix["versions"]
        for encoding in matrix["encodings_by_version"][version]
        for dataset in matrix["datasets"]
    }
    final_aggregates = _exact_key_matrix(
        payload.get("final_test_aggregates"),
        fields=("version", "encoding", "dataset"),
        integer_fields=frozenset(),
        nullable_fields=frozenset({"encoding"}),
        expected=expected_final_aggregates,
        label="Cycle final test aggregates",
    )
    for key, row in final_aggregates.items():
        selected_profiles = row.get("selected_profiles")
        if (
            row.get("model_seeds") != matrix["model_seeds"]
            or not isinstance(selected_profiles, list)
            or len(selected_profiles) != len(matrix["model_seeds"])
            or any(profile not in matrix["profiles"] for profile in selected_profiles)
        ):
            raise RuntimeError(f"Cycle final test aggregate {key!r} is incomplete")
    return {"child_runs": len(expected_children), "model_trainings": len(runs)}


def _validate_tree_summary(payload: dict[str, Any], job: dict[str, Any]) -> dict[str, int]:
    if payload.get("suite") != "tree_scaling":
        raise RuntimeError("Tree child summary suite mismatch")
    if payload.get("run_id") != job["child_run_id"]:
        raise RuntimeError("Tree child summary run ID mismatch")
    matrix = job["requested_matrix"]
    if payload.get("models_per_child") != matrix["models"]:
        raise RuntimeError("Tree child summary model list mismatch")
    _exact_string_mapping_keys(
        payload.get("profile_configs"), matrix["profiles"], "Tree profile configs"
    )

    expected_rows = {
        (suite, profile, seed)
        for suite in matrix["suites"]
        for profile in matrix["profiles"]
        for seed in matrix["model_seeds"]
    }
    results = _exact_key_matrix(
        payload.get("results"),
        fields=("suite", "profile", "model_seed"),
        integer_fields=frozenset({"model_seed"}),
        expected=expected_rows,
        label="Tree candidate results",
    )
    for key, row in results.items():
        if (
            row.get("trained_models") != matrix["models"]
            or row.get("test_evaluated") is not False
            or row.get("test_used_for_selection") is not False
            or row.get("dataset_cache_integrity")
            != {
                "full_cache_loaded": True,
                "all_declared_splits_validated": True,
                "loaded_and_validated_splits": ["test", "train", "validation"],
            }
        ):
            raise RuntimeError(f"Tree candidate result {key!r} has the wrong trained models")

    expected_selection_rows = {(suite,) for suite in matrix["suites"]}
    selections = _exact_key_matrix(
        payload.get("selections"),
        fields=("suite",),
        integer_fields=frozenset(),
        expected=expected_selection_rows,
        label="Tree validation profile selections",
    )
    expected_seed_keys = {str(seed) for seed in matrix["model_seeds"]}
    for key, selection in selections.items():
        conditions = selection.get("conditions")
        if (
            selection.get("selection_split") != "validation"
            or selection.get("aggregation_axis") != "mean_across_requested_model_seeds"
            or selection.get("model_seeds") != matrix["model_seeds"]
            or selection.get("test_metrics_used_for_selection") is not False
            or not isinstance(conditions, dict)
            or set(conditions) != set(matrix["models"])
        ):
            raise RuntimeError(f"Tree validation profile selection {key!r} is invalid")
        for model, condition in conditions.items():
            checkpoints = condition.get("selected_checkpoints_by_model_seed")
            if (
                condition.get("selected_profile") not in matrix["profiles"]
                or not isinstance(checkpoints, dict)
                or set(checkpoints) != expected_seed_keys
            ):
                raise RuntimeError(
                    f"Tree validation selection {key!r}/{model} has an incomplete seed matrix"
                )

    expected_test_rows = {
        (suite, seed) for suite in matrix["suites"] for seed in matrix["model_seeds"]
    }
    test_results = _exact_key_matrix(
        payload.get("selected_test_results"),
        fields=("suite", "model_seed"),
        integer_fields=frozenset({"model_seed"}),
        expected=expected_test_rows,
        label="Tree selected-checkpoint test results",
    )
    for key, row in test_results.items():
        selected_profiles = row.get("selected_profiles")
        selected_checkpoints = row.get("selected_checkpoints")
        selection = selections[(key[0],)]
        if (
            row.get("evaluation_scope") != "selected_test"
            or row.get("training_performed") is not False
            or row.get("test_evaluated") is not True
            or row.get("test_used_for_selection") is not False
            or row.get("test_evaluations_per_selected_checkpoint") != 1
            or not isinstance(selected_profiles, dict)
            or set(selected_profiles) != set(matrix["models"])
            or not isinstance(selected_checkpoints, dict)
            or set(selected_checkpoints) != set(matrix["models"])
            or any(
                selected_profiles[model] != selection["conditions"][model]["selected_profile"]
                for model in matrix["models"]
            )
        ):
            raise RuntimeError(f"Tree selected-checkpoint test result {key!r} is invalid")

    planned_children = _integer(payload.get("planned_child_runs"), "planned_child_runs")
    planned_trainings = _integer(payload.get("planned_model_trainings"), "planned_model_trainings")
    completed_children = _integer(payload.get("completed_child_runs"), "completed_child_runs")
    completed_trainings = _integer(
        payload.get("completed_model_trainings"), "completed_model_trainings"
    )
    failed_children = _integer(payload.get("failed_child_runs"), "failed_child_runs")
    expected_profile_selections = len(matrix["suites"]) * len(matrix["models"])
    planned_profile_selections = _integer(
        payload.get("planned_profile_selections"), "planned_profile_selections"
    )
    completed_profile_selections = _integer(
        payload.get("completed_profile_selections"), "completed_profile_selections"
    )
    expected_test_runs = len(matrix["suites"]) * len(matrix["model_seeds"])
    planned_test_runs = _integer(
        payload.get("planned_selected_test_runs"), "planned_selected_test_runs"
    )
    completed_test_runs = _integer(
        payload.get("completed_selected_test_runs"), "completed_selected_test_runs"
    )
    failed_test_runs = _integer(
        payload.get("failed_selected_test_runs"), "failed_selected_test_runs"
    )
    expected_test_evaluations = expected_test_runs * len(matrix["models"])
    planned_test_evaluations = _integer(
        payload.get("planned_selected_checkpoint_test_evaluations"),
        "planned_selected_checkpoint_test_evaluations",
    )
    completed_test_evaluations = _integer(
        payload.get("completed_selected_checkpoint_test_evaluations"),
        "completed_selected_checkpoint_test_evaluations",
    )
    observed = {
        "child_runs": len(results),
        "model_trainings": sum(len(row["trained_models"]) for row in results.values()),
    }
    expected = job["expected_counts"]
    if (
        {"child_runs": planned_children, "model_trainings": planned_trainings} != expected
        or {"child_runs": completed_children, "model_trainings": completed_trainings} != observed
        or observed != expected
        or failed_children != 0
        or planned_profile_selections != expected_profile_selections
        or completed_profile_selections != expected_profile_selections
        or planned_test_runs != expected_test_runs
        or completed_test_runs != len(test_results)
        or completed_test_runs != expected_test_runs
        or failed_test_runs != 0
        or planned_test_evaluations != expected_test_evaluations
        or completed_test_evaluations != expected_test_evaluations
    ):
        raise RuntimeError("Tree child summary counts are incomplete")
    return observed


def _validate_child_summary(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["summary_path"])
    if not path.is_file():
        raise RuntimeError(f"child returned without its exact summary path: {path}")
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"child summary resolves outside its exact output path: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid child summary JSON at {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise RuntimeError("child summary does not certify status=passed")

    track = job["track"]
    validators = {
        "conductance": _validate_conductance_summary,
        "cycle": _validate_cycle_summary,
        "tree": _validate_tree_summary,
    }
    observed = validators[track](payload, job)
    if observed != job["expected_counts"]:
        raise RuntimeError(f"{track} observed matrix counts do not match the requested plan")
    return {
        "summary_path": str(path),
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "passed",
        "observed_counts": observed,
    }


_RESUME_JOB_FIELDS = (
    "track",
    "device",
    "child_run_id",
    "profiles",
    "shared_profiles",
    "model_seeds",
    "command",
    "output_dir",
    "summary_path",
    "log_path",
    "allow_download_forwarded",
    "requested_matrix",
    "expected_counts",
)


def _config_payload(
    args: argparse.Namespace, *, data_root: Path, results_root: Path
) -> dict[str, Any]:
    return {
        "tracks": list(args.tracks),
        "conductance_versions": list(args.conductance_versions),
        "cycle_versions": list(args.cycle_versions),
        "cycle_v2_encodings": list(args.cycle_v2_encodings),
        "shared_profiles": list(args.profiles),
        "tree_profiles": list(args.profiles),
        "model_seeds": list(args.model_seeds),
        "devices": _execution_devices(args),
        "device_assignment": "round_robin_by_independent_track",
        "execution_classification": "final_research_training",
        "debug_or_smoke_mode": False,
        "hardware_profile": args.hardware_profile,
        "resource_plan": resource_plan_identity(getattr(args, "resolved_resource_plan", None)),
        "explicit_batch_overrides": {
            "conductance_legacy_ppi_graphs": args.conductance_legacy_ppi_batch_size,
            "conductance_v5_ppi_graphs": args.conductance_v5_ppi_batch_size,
            "conductance_v5_sample_seed_nodes": (args.conductance_v5_sample_seed_batch_size),
            "cycle_graphs": args.cycle_batch_size,
            "tree_chart_views": args.tree_batch_size,
        },
        "min_free_gb": args.min_free_gb,
        "v5_beta": _v5_beta_configuration(args),
        "v5_activation_checkpoint": args.v5_activation_checkpoint,
        "cycle_v2_basis_backend": args.cycle_v2_basis_backend,
        "allow_download": args.allow_download,
        "data_root": str(data_root),
        "results_root": str(results_root),
        "fail_fast": args.fail_fast,
    }


def _resume_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    expected_config: dict[str, Any],
    expected_jobs: list[dict[str, Any]],
    expected_totals: dict[str, int],
    expected_sources: dict[str, str],
) -> dict[str, Any]:
    """Load an interrupted run without trusting stale status or artifacts."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing run manifest is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("existing run manifest must be a JSON object")
    if (
        payload.get("schema_version") != 1
        or payload.get("suite") != "rich_scaling"
        or payload.get("run_id") != run_id
    ):
        raise ValueError("existing run manifest identity does not match this runner")
    if payload.get("config") != expected_config:
        raise ValueError("existing run configuration differs; use its original arguments")
    if payload.get("planned_counts") != expected_totals:
        raise ValueError("existing run count contract differs from the requested plan")
    if not snapshots_match(payload.get("source_sha256"), expected_sources):
        raise ValueError("experiment source changed since this run started; use a new run ID")
    if payload.get("source_integrity_valid") is not True:
        raise ValueError("existing run failed source integrity and cannot be resumed")
    stored_jobs = payload.get("jobs")
    if not isinstance(stored_jobs, list) or len(stored_jobs) != len(expected_jobs):
        raise ValueError("existing run job matrix is incomplete")
    for stored, expected in zip(stored_jobs, expected_jobs, strict=True):
        if not isinstance(stored, dict) or any(
            stored.get(field) != expected[field] for field in _RESUME_JOB_FIELDS
        ):
            raise ValueError("existing run job contract differs from the requested plan")
        previous_status = stored.get("status")
        if previous_status == "passed":
            try:
                validated = _validate_child_summary(stored)
                if stored.get("result") != validated:
                    raise RuntimeError("passed track summary hash or certificate changed")
                stored["result"] = validated
            except Exception as error:
                stored["status"] = "pending"
                stored["resume_validation_error"] = f"{type(error).__name__}: {error}"
                stored.pop("result", None)
            else:
                stored.pop("resume_validation_error", None)
                continue
        else:
            stored["status"] = "pending"
            stored["previous_status"] = previous_status
        for field in (
            "returncode",
            "error",
            "elapsed_seconds",
            "started_at_utc",
            "finished_at_utc",
            "result",
        ):
            stored.pop(field, None)
    adopt_source_snapshot(payload, expected_sources)
    payload["status"] = "running"
    payload["source_integrity_valid"] = True
    resume_count = payload.get("resume_count", 0)
    if isinstance(resume_count, bool) or not isinstance(resume_count, int) or resume_count < 0:
        raise ValueError("existing run resume_count is invalid")
    resumed_at = payload.get("resumed_at_utc", [])
    if not isinstance(resumed_at, list) or any(not isinstance(value, str) for value in resumed_at):
        raise ValueError("existing run resumed_at_utc history is invalid")
    payload["resume_count"] = resume_count + 1
    payload["resumed_at_utc"] = [*resumed_at, dt.datetime.now(dt.UTC).isoformat()]
    for field in (
        "error",
        "finished_at_utc",
        "completed_counts",
        "source_integrity_error",
        "source_sha256_final",
    ):
        payload.pop(field, None)
    return payload


def _totals(jobs: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "track_runs": len(jobs),
        "child_runs": sum(job["expected_counts"]["child_runs"] for job in jobs),
        "model_trainings": sum(job["expected_counts"]["model_trainings"] for job in jobs),
    }


def _print_plan(args: argparse.Namespace, run_id: str, jobs: list[dict[str, Any]]) -> None:
    totals = _totals(jobs)
    print(
        f"{totals['track_runs']} track runs; {totals['child_runs']} child runs; "
        f"{totals['model_trainings']} fresh model trainings"
    )
    print(f"profiles={list(args.profiles)}; model_seeds={list(args.model_seeds)}")
    print(
        f"conductance_versions={list(args.conductance_versions)}; "
        f"cycle_versions={list(args.cycle_versions)}"
    )
    devices = _execution_devices(args)
    print("execution_classification=plan_only; training_started=false; debug_or_smoke=false")
    if _needs_resource_calibration(args):
        print(
            "V5/Cycle V2: optimizer-inclusive paired batch/worker calibration precedes training; "
            "shown batches are requested floors, not measured selections"
        )
    print(
        f"hardware_profile={args.hardware_profile}; devices={devices}; "
        f"track_concurrency={min(len(devices), len(jobs))} "
        "(one independent track child per assigned GPU)"
    )
    for job in jobs:
        expected = job["expected_counts"]
        print(
            f"[{job['track']}] device={job['device']}; {expected['child_runs']} child runs; "
            f"{expected['model_trainings']} fresh model trainings; "
            f"child profiles={job['profiles']}"
        )
        print(shlex.join(job["command"]))
        print(f"  summary: {job['summary_path']}")
    results_root = args.results_root.expanduser().resolve()
    print(f"central manifest: {results_root / 'rich_scaling' / run_id / 'manifest.json'}")


def _needs_resource_calibration(args: argparse.Namespace) -> bool:
    return ("conductance" in args.tracks and "v5" in args.conductance_versions) or (
        "cycle" in args.tracks and "v2" in args.cycle_versions
    )


def _calibration_request(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    from scripts import run_conductance_scaling, run_cycle_scaling

    baseline = copy.deepcopy(args)
    baseline.resource_plan = None
    baseline.resolved_resource_plan = None
    baseline.dry_run = False
    # Both mechanisms must fit the same selected resources, even for a one-arm final selection.
    baseline.cycle_v2_encodings = ["se", "pe"]
    paired_jobs = []
    for track_job in make_jobs(baseline, run_id):
        track = track_job["track"]
        if track == "tree":
            continue
        module = run_conductance_scaling if track == "conductance" else run_cycle_scaling
        child_args = module.parser().parse_args(track_job["command"][3:])
        module._validate(child_args)
        child_args.resolved_resource_plan = None
        for job in module.make_jobs(child_args, Path(track_job["output_dir"])):
            if job["version"] != ("v5" if track == "conductance" else "v2"):
                continue
            paired_jobs.append(
                {
                    "track": track,
                    "profile": job["profile"],
                    "dataset": job["dataset"] if track == "conductance" else job["datasets"][0],
                    "condition": job["condition"] if track == "conductance" else job["encoding"],
                    "model_seed": job["model_seed"],
                    "device": track_job["device"],
                    "command": job["command"],
                }
            )
    return {
        "schema_version": 1,
        "kind": "training_resource_calibration_request",
        "hardware_profile": args.hardware_profile,
        "profiles": list(args.profiles),
        "model_seeds": list(args.model_seeds),
        "minimum_free_gb": args.min_free_gb,
        "source_sha256": calibration_source_snapshot(),
        "jobs": paired_jobs,
    }


def _ensure_measured_plan(args: argparse.Namespace, run_id: str, run_dir: Path) -> None:
    if not _needs_resource_calibration(args):
        if args.resource_plan is not None:
            raise ValueError("resource plan requires a selected V5 or Cycle V2 track")
        return
    from scripts.training_resource_plan import validate_plan_runtime

    if args.resource_plan is not None:
        args.resource_plan = args.resource_plan.expanduser().resolve()
        args.resolved_resource_plan = load_resource_plan(
            args.resource_plan,
            hardware_profile=args.hardware_profile,
            profiles=list(args.profiles),
            model_seeds=list(args.model_seeds),
        )
        validate_plan_runtime(args.resolved_resource_plan)
        from scripts.calibrate_training_resources import verify_plan_inputs

        verify_plan_inputs(args.resolved_resource_plan, _calibration_request(args, run_id)["jobs"])
        return
    if (run_dir / "manifest.json").exists():
        existing = json.loads((run_dir / "manifest.json").read_bytes())
        if not existing.get("config", {}).get("resource_plan"):
            raise ValueError(
                "existing run has no measured resource plan; it is preserved and cannot be "
                "silently reconfigured. Use a separate new run ID"
            )
    directory = args.results_root.expanduser().resolve() / "resource_calibration" / run_id
    if directory.is_symlink() or not directory.resolve().is_relative_to(
        args.results_root.expanduser().resolve()
    ):
        raise ValueError("calibration output escapes results root")
    if _paths_overlap(directory, args.data_root.expanduser().resolve()):
        raise ValueError("calibration outputs must not overlap dataset caches")
    request = _calibration_request(args, run_id)
    request_path = directory / "request.json"
    if directory.exists():
        if not request_path.is_file() or request_path.is_symlink():
            raise ValueError("existing calibration directory is not owned by this request")
        stored_request = json.loads(request_path.read_bytes())
        # Preserve the original request and its digest in the measured certificate.
        # Only the reviewed recovery patch may differ; the recipe stays exact.
        same_recipe = isinstance(stored_request, dict) and {
            key: value for key, value in stored_request.items() if key != "source_sha256"
        } == {key: value for key, value in request.items() if key != "source_sha256"}
        if not same_recipe or not snapshots_match(
            stored_request.get("source_sha256"), request["source_sha256"]
        ):
            raise ValueError(
                "calibration source/configuration differs; previous measurements preserved; "
                "use a new run ID"
            )
    else:
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(request_path, request)
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    attempt = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    print(
        "Measuring real training batch/worker candidates before final training; "
        "completed measurements resume; model/data size unchanged",
        flush=True,
    )
    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/calibrate_training_resources.py"),
        "--request",
        str(request_path),
        "--output-dir",
        str(directory),
    ]
    code = _run_logged(command, logs / f"calibration-{attempt}.log", _environment())
    if code:
        raise RuntimeError(
            f"resource calibration failed with code {code}; no final training launched; "
            f"inspect {directory / 'progress.json'}"
        )
    args.resource_plan = directory / "resource-plan.json"
    args.resolved_resource_plan = load_resource_plan(
        args.resource_plan,
        hardware_profile=args.hardware_profile,
        profiles=list(args.profiles),
        model_seeds=list(args.model_seeds),
    )
    validate_plan_runtime(args.resolved_resource_plan)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    run_id = args.run_id or _default_run_id()
    results_root = args.results_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    run_dir = (results_root / "rich_scaling" / run_id).resolve()
    jobs = make_jobs(args, run_id)
    relevant_outputs = [run_dir, *(Path(job["output_dir"]) for job in jobs)]
    if any(not output.is_relative_to(results_root) for output in relevant_outputs):
        print("experiment outputs must resolve within the results root", file=sys.stderr)
        return 2
    if any(_paths_overlap(output, data_root) for output in relevant_outputs):
        print("experiment outputs and dataset directories must not overlap", file=sys.stderr)
        return 2
    missing = [
        ROOT / "scripts" / TRACK_SPECS[track]["script"]
        for track in args.tracks
        if not (ROOT / "scripts" / TRACK_SPECS[track]["script"]).is_file()
    ]
    if missing:
        print(f"missing child runner: {missing[0]}", file=sys.stderr)
        return 2
    if args.dry_run:
        _print_plan(args, run_id, jobs)
        print("dry run only; no files or directories were written")
        return 0
    try:
        _STOP_ACTIVE_CHILDREN.clear()
        _ensure_measured_plan(args, run_id, run_dir)
        jobs = make_jobs(args, run_id)
    except KeyboardInterrupt:
        print("resource calibration interrupted; partial measurements preserved", flush=True)
        return 130
    except (ValueError, RuntimeError, OSError, UnicodeError) as error:
        print(f"cannot start final training: {error}", file=sys.stderr, flush=True)
        return 2
    totals = _totals(jobs)
    manifest_path = run_dir / "manifest.json"
    sources = _source_snapshot()
    config = _config_payload(args, data_root=data_root, results_root=results_root)
    resumed = run_dir.exists()
    if resumed:
        try:
            manifest = _resume_manifest(
                manifest_path,
                run_id=run_id,
                expected_config=config,
                expected_jobs=jobs,
                expected_totals=totals,
                expected_sources=sources,
            )
        except (ValueError, RuntimeError) as error:
            print(f"cannot resume existing run: {error}", file=sys.stderr)
            return 2
        (run_dir / "logs").mkdir(exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "logs").mkdir(exist_ok=False)
        manifest = {
            "schema_version": 1,
            "suite": "rich_scaling",
            "run_id": run_id,
            "status": "running",
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "config": config,
            "protocol": {
                "purpose": "reference-scale training for every selected V1/V2/V3/V4/V5 track",
                "execution_classification": "final_research_training",
                "debug_or_smoke_mode": False,
                "dry_run_classification": "plan_only_without_training_or_output_writes",
                "execution": (
                    "selected independent track runners execute concurrently only when distinct "
                    "indexed CUDA devices are explicitly supplied; otherwise they stay sequential"
                ),
                "hardware_profile": {
                    "name": args.hardware_profile,
                    "portable": "conservative settings and one independent job at a time",
                    "a6000-48gb": "child runners require at least 40 GiB visible VRAM and "
                    "compute capability 8.0, then apply track-specific minibatch, AMP, worker, "
                    "and safe independent-job concurrency settings",
                    "devices": _execution_devices(args),
                    "cross_track_concurrency": min(len(_execution_devices(args)), len(jobs)),
                    "assignment": "round_robin_by_independent_track",
                    "same_device_concurrency": 1,
                    "reason": (
                        "distinct GPUs may run disjoint track children concurrently; no GPU runs "
                        "multiple track children because their peak CUDA allocations are unbounded"
                    ),
                },
                "batch_selection": {
                    "default_policy": (
                        "measured_paired_training_resources_for_v5_cycle_v2; "
                        "legacy_requested_resources_unchanged"
                    ),
                    "measured_throughput_optimum_claimed": False,
                    "automatic_downscale": False,
                    "throughput_candidate_sweep": getattr(args, "resolved_resource_plan", None)
                    is not None,
                    "measured_resource_plan": resource_plan_identity(
                        getattr(args, "resolved_resource_plan", None)
                    ),
                    "explicit_overrides": config["explicit_batch_overrides"],
                    "override_scope": (
                        "V5/Cycle V2 overrides set measured batch floors; the immutable plan "
                        "supplies final selections. Legacy and Tree overrides are unchanged"
                    ),
                    "limitation": (
                        "V5/Cycle V2 measure optimizer-inclusive paired batch/worker candidates "
                        "before final training. Selection maximizes measured worst-arm throughput "
                        "within memory headroom, never below requested batch floors. The selected "
                        "recipe is immutable on resume; finite candidate results are not a global "
                        "optimum claim. Legacy versions retain their requested configurations"
                    ),
                },
                "failure_policy": (
                    "do not start a later device wave after any failure; already-running "
                    "distinct-GPU peers finish and are recorded"
                    if args.fail_fast
                    else "continue remaining tracks"
                ),
                "tree_profiles": "reference/large are forwarded without renaming",
                "cycle_v2_basis_backend": args.cycle_v2_basis_backend,
                "cycle_v2_encodings": list(args.cycle_v2_encodings),
                "download_policy": (
                    "allow-download is forwarded to Cycle and Tree; Conductance remains "
                    "verified-cache-only because its child contract exposes no download flag"
                ),
                "resume": "the same run ID verifies immutable config/source/job contracts, "
                "skips valid passed tracks, and reruns only incomplete or invalid tracks",
                "verification": "nonzero return code, missing/wrong summary path, or an "
                "incomplete non-passed summary fails its track",
            },
            "planned_counts": totals,
            "jobs": jobs,
            "source_integrity_valid": True,
            "source_sha256": sources,
        }
    jobs = manifest["jobs"]
    _atomic_write_json(manifest_path, manifest)
    environment = _environment()
    failed = False
    interrupted = False
    primary_error: BaseException | None = None
    print(
        f"Run: {run_id}; {totals['track_runs']} tracks; "
        f"{totals['model_trainings']} planned model trainings; "
        f"resume={'yes' if resumed else 'no'}",
        flush=True,
    )
    concurrency = min(len(_execution_devices(args)), len(jobs))
    for wave_start in range(0, len(jobs), concurrency):
        if failed and args.fail_fast:
            break
        wave = jobs[wave_start : wave_start + concurrency]
        starts: dict[int, float] = {}
        for offset, job in enumerate(wave, start=wave_start + 1):
            previously_passed = job["status"] == "passed"
            job["status"] = "running"
            job["started_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
            starts[id(job)] = time.monotonic()
            action = (
                "verify completed child state" if previously_passed else "resume incomplete child"
            )
            print(
                f"\n[{offset}/{len(jobs)}] {job['track']} on {job['device']} — {action}",
                flush=True,
            )
        if primary_error is None:
            _atomic_write_json(manifest_path, manifest)
        else:
            _persist_manifest_after_error(
                manifest_path,
                manifest,
                primary_error,
                action="post-failure wave-start manifest persistence",
            )

        def run_child(job: dict[str, Any]) -> int:
            log_path = Path(job["log_path"])
            if not log_path.resolve().is_relative_to(run_dir):
                raise RuntimeError("track log path resolves outside the central run directory")
            return _run_logged(job["command"], log_path, environment)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(wave))
        futures: dict[concurrent.futures.Future[int], dict[str, Any]] = {}
        try:
            futures = {executor.submit(run_child, job): job for job in wave}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                job_error: BaseException | None = None
                try:
                    returncode = future.result()
                    job["returncode"] = returncode
                    if returncode != 0:
                        raise RuntimeError(
                            f"{job['track']} child failed with exit code {returncode}"
                        )
                    job["result"] = _validate_child_summary(job)
                    job["status"] = "passed"
                except Exception as error:
                    job_error = error
                    if primary_error is None:
                        primary_error = error
                    failed = True
                    job["status"] = "failed"
                    job["error"] = f"{type(error).__name__}: {error}"
                    print(f"Failed {job['track']}: {job['error']}", file=sys.stderr)
                finally:
                    job["elapsed_seconds"] = time.monotonic() - starts[id(job)]
                    job["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
                    persistence_error = job_error or primary_error
                    if persistence_error is None:
                        _atomic_write_json(manifest_path, manifest)
                    else:
                        _persist_manifest_after_error(
                            manifest_path,
                            manifest,
                            persistence_error,
                            action="post-child failure manifest persistence",
                        )
        except KeyboardInterrupt as error:
            primary_error = error
            interrupted = True
            failed = True
            interruption_error = f"{type(error).__name__}: {error}"
            interruption_reason = (
                "central rich-scaling coordinator received KeyboardInterrupt; "
                "stopping only its exact registered child processes before executor shutdown"
            )
            _STOP_ACTIVE_CHILDREN.set()
            for future in futures:
                future.cancel()
            signal_events = _stop_active_children(reason=interruption_reason, original_error=error)
            if signal_events:
                manifest.setdefault("child_signal_events", []).extend(signal_events)
            manifest["interruption"] = {
                "error": interruption_error,
                "reason": interruption_reason,
                "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            }
            for job in wave:
                if job["status"] == "running":
                    job["status"] = "failed"
                    job["error"] = interruption_error
                    job["elapsed_seconds"] = time.monotonic() - starts[id(job)]
                    job["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
            _persist_manifest_after_error(
                manifest_path,
                manifest,
                error,
                action="interruption manifest persistence",
            )
        finally:
            if primary_error is None:
                executor.shutdown(wait=True, cancel_futures=interrupted)
            else:
                run_failure_reporter(
                    lambda executor=executor, interrupted=interrupted: executor.shutdown(
                        wait=True, cancel_futures=interrupted
                    ),
                    original_error=primary_error,
                    action="post-failure executor shutdown",
                )
            _STOP_ACTIVE_CHILDREN.clear()
        if interrupted:
            break

    try:
        _check_central_sources(manifest)
    except Exception as error:
        if primary_error is None:
            primary_error = error
        failed = True
        manifest["source_integrity_error"] = f"{type(error).__name__}: {error}"
        print(f"Failed source integrity: {manifest['source_integrity_error']}", file=sys.stderr)

    manifest["status"] = "failed" if failed else "passed"
    manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    manifest["completed_counts"] = {
        "passed_tracks": sum(job["status"] == "passed" for job in jobs),
        "failed_tracks": sum(job["status"] == "failed" for job in jobs),
        "pending_tracks": sum(job["status"] == "pending" for job in jobs),
        "verified_child_runs": sum(
            job["expected_counts"]["child_runs"]
            for job in jobs
            if job["status"] == "passed" and manifest["source_integrity_valid"] is True
        ),
        "verified_model_trainings": sum(
            job["expected_counts"]["model_trainings"]
            for job in jobs
            if job["status"] == "passed" and manifest["source_integrity_valid"] is True
        ),
    }
    if failed:
        if primary_error is None:
            raise RuntimeError("failed rich-scaling run has no recorded primary error")
        _persist_manifest_after_error(
            manifest_path,
            manifest,
            primary_error,
            action="final failed-run manifest persistence",
        )
    else:
        _atomic_write_json(manifest_path, manifest)
    if failed:
        print(f"Rich scaling failed; inspect {manifest_path}", file=sys.stderr)
        return 130 if interrupted else 1
    print(f"Rich scaling passed; manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
