#!/usr/bin/env python3
"""Run the complete Conductance, Cycle PE, and Tree larger-model scaling suites."""

from __future__ import annotations

import argparse
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
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.conductance_gat.v5.protocol import (  # noqa: E402
    BETA_PARAMETERIZATIONS,
    DEFAULT_BETA_INITIAL,
    DEFAULT_BETA_PARAMETERIZATION,
    beta_configuration,
)

TRACKS = ("conductance", "cycle", "tree")
PROFILES = ("reference", "large")
HARDWARE_PROFILES = ("portable", "a6000-48gb")
CYCLE_V2_BASIS_BACKENDS = ("thin_q", "dfs_fundamental")
DEFAULT_MODEL_SEEDS = (0,)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}")

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
    result.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    result.add_argument("--model-seeds", nargs="+", type=int, default=list(DEFAULT_MODEL_SEEDS))
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--run-id")
    result.add_argument("--device", default="cuda")
    result.add_argument(
        "--hardware-profile",
        choices=HARDWARE_PROFILES,
        default="portable",
        help=(
            "portable keeps conservative child settings; a6000-48gb enables each track's "
            "recorded high-throughput settings and fails closed below 40 GiB visible VRAM or "
            "compute capability 8.0"
        ),
    )
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument(
        "--v5-beta-parameterization",
        choices=BETA_PARAMETERIZATIONS,
        default=DEFAULT_BETA_PARAMETERIZATION,
    )
    result.add_argument("--v5-beta-initial", type=float, default=DEFAULT_BETA_INITIAL)
    result.add_argument("--v5-beta-min", type=float)
    result.add_argument("--v5-beta-max", type=float)
    result.add_argument(
        "--cycle-v2-basis-backend",
        choices=CYCLE_V2_BASIS_BACKENDS,
        default="thin_q",
        help="Cycle V2 basis construction; thin_q is the production default",
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
        ("profiles", args.profiles),
        ("model seeds", args.model_seeds),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{label} must be nonempty and contain no duplicates")
    if any(seed < 0 for seed in args.model_seeds):
        raise ValueError("model seeds must be nonnegative")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("rich scaling requires CUDA; CPU fallback is not supported")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("minimum free GPU memory must be finite and nonnegative")
    if args.run_id is not None and RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID must be 1-120 letters, digits, underscores, or hyphens")
    _v5_beta_configuration(args)


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


def _requested_matrix(track: str, profiles: list[str], model_seeds: list[int]) -> dict[str, Any]:
    common = {"profiles": list(profiles), "model_seeds": list(model_seeds)}
    if track == "conductance":
        requested_datasets = list(
            dict.fromkeys(
                dataset for spec in CONDUCTANCE_MATRIX.values() for dataset in spec["datasets"]
            )
        )
        return {
            **common,
            "versions": list(CONDUCTANCE_MATRIX),
            "requested_datasets": requested_datasets,
            "datasets_by_version": {
                version: list(spec["datasets"]) for version, spec in CONDUCTANCE_MATRIX.items()
            },
            "conditions_by_version": {
                version: list(spec["conditions"]) for version, spec in CONDUCTANCE_MATRIX.items()
            },
        }
    if track == "cycle":
        return {
            **common,
            "versions": list(CYCLE_VERSIONS),
            "datasets": list(CYCLE_DATASETS),
        }
    return {
        **common,
        "suites": list(TREE_SUITES),
        "models": list(TREE_MODELS),
    }


def _expected_counts(track: str, profiles: list[str], model_seeds: list[int]) -> dict[str, int]:
    combinations = len(profiles) * len(model_seeds)
    if track == "conductance":
        trainings_per_combination = sum(
            len(spec["datasets"]) * len(spec["conditions"]) for spec in CONDUCTANCE_MATRIX.values()
        )
        return {
            "child_runs": combinations * trainings_per_combination,
            "model_trainings": combinations * trainings_per_combination,
        }
    if track == "cycle":
        child_runs = combinations * len(CYCLE_VERSIONS) * len(CYCLE_DATASETS)
        return {
            "child_runs": child_runs,
            "model_trainings": child_runs,
        }
    child_runs = combinations * len(TREE_SUITES)
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
    """Build one sequential child-process job per selected research track."""
    results_root = args.results_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    jobs: list[dict[str, Any]] = []
    for track in args.tracks:
        spec = TRACK_SPECS[track]
        child_run_id = _child_run_id(run_id, track)
        child_dir = (results_root / spec["results_subdir"] / child_run_id).resolve()
        profiles = list(args.profiles)
        requested_matrix = _requested_matrix(track, profiles, list(args.model_seeds))
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
        elif track == "cycle":
            command += ["--versions", *requested_matrix["versions"]]
            command += ["--datasets", *requested_matrix["datasets"]]
            command += ["--profiles", *profiles]
            command += ["--model-seeds", ",".join(str(seed) for seed in args.model_seeds)]
            command += ["--basis-backend", args.cycle_v2_basis_backend]
        else:
            command += ["--versions", *requested_matrix["versions"]]
            command += ["--datasets", *requested_matrix["requested_datasets"]]
            command += ["--profiles", *profiles]
            command += ["--model-seeds", *(str(seed) for seed in args.model_seeds)]
            for name, value in _v5_beta_configuration(args).items():
                command += ["--v5-" + name.replace("_", "-"), str(value)]
        command += [
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
            "--run-id",
            child_run_id,
            "--device",
            args.device,
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
        if args.dry_run:
            command.append("--dry-run")
        jobs.append(
            {
                "track": track,
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
                "expected_counts": _expected_counts(track, profiles, list(args.model_seeds)),
            }
        )
    return jobs


def _source_snapshot() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "research/__init__.py",
        ROOT / "scripts/check_dependencies.py",
        ROOT / "scripts/gpu_profiles.py",
        ROOT / "scripts/gpu_preflight.py",
        ROOT / "scripts/run_conductance_factorial.py",
        ROOT / "scripts/verify_gpu_lock.py",
        ROOT / "research/tree_augmentation/config.yaml",
    ]
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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
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

    expected_runs = {
        (version, profile, seed, dataset)
        for version in matrix["versions"]
        for profile in matrix["profiles"]
        for seed in matrix["model_seeds"]
        for dataset in matrix["datasets"]
    }
    runs = _exact_key_matrix(
        payload.get("runs"),
        fields=("version", "profile", "model_seed", "dataset"),
        integer_fields=frozenset({"model_seed"}),
        expected=expected_runs,
        label="Cycle training rows",
    )
    expected_children = {
        (version, profile, seed, dataset)
        for version in matrix["versions"]
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
        (version, dataset, profile)
        for version in matrix["versions"]
        for dataset in matrix["datasets"]
        for profile in matrix["profiles"]
    }
    aggregates = _exact_key_matrix(
        payload.get("profile_aggregates"),
        fields=("version", "dataset", "profile"),
        integer_fields=frozenset(),
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
        (version, dataset) for version in matrix["versions"] for dataset in matrix["datasets"]
    }
    profile_selections = _exact_key_matrix(
        payload.get("profile_selections"),
        fields=("version", "dataset"),
        integer_fields=frozenset(),
        expected=expected_profile_selections,
        label="Cycle validation profile selections",
    )
    for key, row in profile_selections.items():
        if (
            row.get("selected_profile") not in matrix["profiles"]
            or row.get("model_seeds") != matrix["model_seeds"]
            or row.get("test_used_for_selection") is not False
        ):
            raise RuntimeError(f"Cycle validation profile selection {key!r} is invalid")

    expected_checkpoints = {
        (version, dataset, seed)
        for version in matrix["versions"]
        for dataset in matrix["datasets"]
        for seed in matrix["model_seeds"]
    }
    selected_checkpoints = _exact_key_matrix(
        payload.get("selected_checkpoints"),
        fields=("version", "dataset", "model_seed"),
        integer_fields=frozenset({"model_seed"}),
        expected=expected_checkpoints,
        label="Cycle selected validation checkpoints",
    )
    for key, row in selected_checkpoints.items():
        profile_selection = profile_selections[(key[0], key[1])]
        if row.get("selected_profile") != profile_selection.get("selected_profile"):
            raise RuntimeError(f"Cycle selected checkpoint {key!r} uses the wrong profile")

    test_rows = _exact_key_matrix(
        payload.get("test_evaluations"),
        fields=("version", "dataset", "model_seed"),
        integer_fields=frozenset({"model_seed"}),
        expected=expected_checkpoints,
        label="Cycle selected-checkpoint test evaluations",
    )
    for key, row in test_rows.items():
        if (
            row.get("selected_profile") != selected_checkpoints[key].get("selected_profile")
            or row.get("checkpoint") != selected_checkpoints[key].get("checkpoint")
            or row.get("checkpoint_sha256") != selected_checkpoints[key].get("checkpoint_sha256")
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
        (version, dataset) for version in matrix["versions"] for dataset in matrix["datasets"]
    }
    final_aggregates = _exact_key_matrix(
        payload.get("final_test_aggregates"),
        fields=("version", "dataset"),
        integer_fields=frozenset(),
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
        "shared_profiles": list(args.profiles),
        "tree_profiles": list(args.profiles),
        "model_seeds": list(args.model_seeds),
        "device": args.device,
        "hardware_profile": args.hardware_profile,
        "min_free_gb": args.min_free_gb,
        "v5_beta": _v5_beta_configuration(args),
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
    if payload.get("source_sha256") != expected_sources:
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
        f"hardware_profile={args.hardware_profile}; track_concurrency=1 "
        "(independent-job concurrency is owned by each child runner)"
    )
    for job in jobs:
        expected = job["expected_counts"]
        print(
            f"[{job['track']}] {expected['child_runs']} child runs; "
            f"{expected['model_trainings']} fresh model trainings; "
            f"child profiles={job['profiles']}"
        )
        print(shlex.join(job["command"]))
        print(f"  summary: {job['summary_path']}")
    results_root = args.results_root.expanduser().resolve()
    print(f"central manifest: {results_root / 'rich_scaling' / run_id / 'manifest.json'}")


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
                "execution": "selected track runners execute sequentially without a shell",
                "hardware_profile": {
                    "name": args.hardware_profile,
                    "portable": "conservative settings and one independent job at a time",
                    "a6000-48gb": "child runners require at least 40 GiB visible VRAM and "
                    "compute capability 8.0, then apply track-specific minibatch, AMP, worker, "
                    "and safe independent-job concurrency settings",
                    "cross_track_concurrency": 1,
                    "reason": "tracks stay sequential because simultaneous peak allocations "
                    "and independent CUDA caching allocators are not bounded by this runner",
                },
                "failure_policy": (
                    "stop after first failed track"
                    if args.fail_fast
                    else "continue remaining tracks"
                ),
                "tree_profiles": "reference/large are forwarded without renaming",
                "cycle_v2_basis_backend": args.cycle_v2_basis_backend,
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
    print(
        f"Run: {run_id}; {totals['track_runs']} tracks; "
        f"{totals['model_trainings']} planned model trainings; "
        f"resume={'yes' if resumed else 'no'}",
        flush=True,
    )
    for index, job in enumerate(jobs, start=1):
        if failed and args.fail_fast:
            break
        previously_passed = job["status"] == "passed"
        job["status"] = "running"
        job["started_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        _atomic_write_json(manifest_path, manifest)
        action = "verify completed child state" if previously_passed else "resume incomplete child"
        print(f"\n[{index}/{len(jobs)}] {job['track']} — {action}", flush=True)
        started = time.monotonic()
        try:
            log_path = Path(job["log_path"])
            if not log_path.resolve().is_relative_to(run_dir):
                raise RuntimeError("track log path resolves outside the central run directory")
            returncode = _run_logged(job["command"], log_path, environment)
            job["returncode"] = returncode
            if returncode != 0:
                raise RuntimeError(f"{job['track']} child failed with exit code {returncode}")
            job["result"] = _validate_child_summary(job)
            job["status"] = "passed"
        except KeyboardInterrupt as error:
            interrupted = True
            failed = True
            job["status"] = "failed"
            job["error"] = f"{type(error).__name__}: {error}"
        except Exception as error:
            failed = True
            job["status"] = "failed"
            job["error"] = f"{type(error).__name__}: {error}"
            print(f"Failed {job['track']}: {job['error']}", file=sys.stderr)
        finally:
            job["elapsed_seconds"] = time.monotonic() - started
            job["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
            _atomic_write_json(manifest_path, manifest)
        if interrupted:
            break

    try:
        _check_central_sources(manifest)
    except Exception as error:
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
    _atomic_write_json(manifest_path, manifest)
    if failed:
        print(f"Rich scaling failed; inspect {manifest_path}", file=sys.stderr)
        return 130 if interrupted else 1
    print(f"Rich scaling passed; manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
