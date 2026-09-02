#!/usr/bin/env python3
"""Run larger Tree V1/V2 fixed-vs-multi experiments on CSL and ZINC.

Each candidate child trains ``fixed_bfs`` (V1) and ``multi_chart`` (V2) from
scratch and evaluates only the official validation split.  Profiles are selected
separately for the two conditions after aggregation across requested model seeds,
then one test-only child per seed evaluates the selected checkpoints without retraining.
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
from scripts.check_dependencies import (  # noqa: E402
    DependencyCheckError,
    check_dependencies,
    error_message,
)

SUITES = ("csl", "zinc")
MODELS = ("fixed_bfs", "multi_chart")
PROFILE_CONFIGS: dict[str, dict[str, int]] = {
    "base": {
        "hidden_dim": 64,
        "message_layers": 2,
        "optimizer_updates": 800,
        "train_charts_per_graph": 8,
        "eval_charts_per_graph": 8,
    },
    "wide": {
        "hidden_dim": 128,
        "message_layers": 2,
        "optimizer_updates": 800,
        "train_charts_per_graph": 8,
        "eval_charts_per_graph": 8,
    },
    "deep": {
        "hidden_dim": 64,
        "message_layers": 4,
        "optimizer_updates": 800,
        "train_charts_per_graph": 8,
        "eval_charts_per_graph": 8,
    },
    "large": {
        "hidden_dim": 128,
        "message_layers": 4,
        "optimizer_updates": 800,
        "train_charts_per_graph": 8,
        "eval_charts_per_graph": 8,
    },
}
PROFILES = tuple(PROFILE_CONFIGS)
DEFAULT_MODEL_SEEDS = (0, 1, 2, 3, 4)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}")


def _csv_subset(value: str, *, choices: tuple[str, ...], option: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"{option} must be non-empty and unique")
    unknown = sorted(set(selected) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"{option} contains unsupported values {unknown}; choose from {list(choices)}"
        )
    return selected


def _suites(value: str) -> tuple[str, ...]:
    return _csv_subset(value, choices=SUITES, option="--suites")


def _profiles(value: str) -> tuple[str, ...]:
    return _csv_subset(value, choices=PROFILES, option="--profiles")


def _model_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--model-seeds must contain comma-separated integers"
        ) from error
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "--model-seeds must be non-empty, unique, and non-negative"
        )
    return seeds


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suites", type=_suites, default=SUITES)
    result.add_argument("--profiles", type=_profiles, default=PROFILES)
    result.add_argument(
        "--model-seeds",
        type=_model_seeds,
        default=DEFAULT_MODEL_SEEDS,
        help="comma-separated model/minibatch seeds (default: 0,1,2,3,4)",
    )
    result.add_argument("--data-seed", type=int, default=0)
    result.add_argument("--split-seed", type=int, default=0)
    result.add_argument("--chart-seed", type=int, default=0)
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--run-id")
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--dry-run", action="store_true", help="print the full plan; no writes")
    return result


def _validate(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("Tree scaling requires an explicit CUDA device; no CPU fallback")
    if min(args.data_seed, args.split_seed, args.chart_seed, *args.model_seeds) < 0:
        raise ValueError("all seed axes must be non-negative")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers must be non-negative")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise ValueError("minimum free GPU memory must be finite and non-negative")
    if args.run_id is not None and RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID must be 1-120 letters, digits, underscores, or hyphens")


def _default_run_id() -> str:
    return "tree-scaling-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def make_jobs(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    data_root = args.data_root.expanduser().resolve()
    for suite in args.suites:
        for model_seed in args.model_seeds:
            for profile in args.profiles:
                profile_config = dict(PROFILE_CONFIGS[profile])
                output = run_dir / suite / profile / f"model-seed-{model_seed}"
                command = [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "research.tree_augmentation.paper",
                    "--suite",
                    suite,
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(output),
                    "--device",
                    args.device,
                    "--seed",
                    str(args.data_seed),
                    "--data-seed",
                    str(args.data_seed),
                    "--split-seed",
                    str(args.split_seed),
                    "--chart-seed",
                    str(args.chart_seed),
                    "--model-seed",
                    str(model_seed),
                    "--batch-size",
                    str(args.batch_size),
                    "--workers",
                    str(args.workers),
                    "--evaluation-scope",
                    "validation",
                ]
                for key, value in profile_config.items():
                    command.extend(("--" + key.replace("_", "-"), str(value)))
                if args.allow_download:
                    command.append("--allow-download")
                if args.amp is not None:
                    command.append("--amp" if args.amp else "--no-amp")
                jobs.append(
                    {
                        "suite": suite,
                        "profile": profile,
                        "profile_config": profile_config,
                        "model_seed": model_seed,
                        "trained_models": list(MODELS),
                        "status": "pending",
                        "output_dir": str(output),
                        "summary_path": str(output / "summary.json"),
                        "manifest_path": str(output / "manifest.json"),
                        "log_path": str(
                            run_dir / "logs" / f"{suite}--{profile}--seed-{model_seed}.log"
                        ),
                        "command": command,
                    }
                )
    return jobs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot() -> dict[str, Any]:
    files = sorted((ROOT / "research/tree_augmentation").glob("*.py"))
    files += sorted((ROOT / "research/tree_augmentation").glob("*.yaml"))
    files += [
        Path(__file__).resolve(),
        ROOT / "scripts/check_dependencies.py",
        ROOT / "scripts/gpu_preflight.py",
        ROOT / "src/chartgat/algebra.py",
        ROOT / "src/chartgat/cache.py",
        ROOT / "src/chartgat/graphs.py",
        ROOT / "src/chartgat/seeds.py",
    ]
    hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in sorted(set(files))}
    revision = "unknown"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"git_revision": revision, "sha256": hashes}


def _check_sources(manifest: dict[str, Any]) -> None:
    if _source_snapshot() != manifest["sources"]:
        manifest["source_integrity_valid"] = False
        raise RuntimeError("experiment source changed during the run; refusing mixed sources")


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    entries = [str(ROOT / "src"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # A stale NVML-based visibility probe can disagree with a MIG allocation.
    environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    return environment


def _run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
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


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"child returned without {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid child {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"child {label} must contain a JSON object: {path}")
    return payload


def _finite_metric_mapping(values: Any, label: str) -> dict[str, float]:
    if not isinstance(values, dict) or not values:
        raise RuntimeError(f"missing child metrics for {label}")
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"non-numeric child metric in {label}/{key}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError(f"non-finite child metric in {label}/{key}")
        result[str(key)] = numeric
    return result


def _validate_child(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output_dir"]).resolve()
    summary_path = Path(job["summary_path"])
    manifest_path = Path(job["manifest_path"])
    child_summary = _read_mapping(summary_path, "summary.json")
    child_manifest = _read_mapping(manifest_path, "manifest.json")
    if child_manifest.get("status") != "passed" or child_manifest.get("suite") != job["suite"]:
        raise RuntimeError("child manifest does not certify the requested passed suite")
    if child_summary.get("suite") != job["suite"]:
        raise RuntimeError("child summary suite does not match its job")
    expected_cache_integrity = {
        "full_cache_loaded": True,
        "all_declared_splits_validated": True,
        "loaded_and_validated_splits": ["test", "train", "validation"],
    }
    expected_model_split_usage = {
        "fit_splits": ["train"],
        "evaluation_splits": ["validation"],
        "selection_splits": ["validation"],
        "test_evaluated": False,
        "test_used_for_selection": False,
    }
    if (
        child_manifest.get("evaluation_scope") != "validation"
        or child_manifest.get("training_performed") is not True
        or child_manifest.get("dataset_cache_integrity") != expected_cache_integrity
        or child_manifest.get("model_split_usage") != expected_model_split_usage
        or child_summary.get("evaluation_scope") != "validation"
        or child_summary.get("training_performed") is not True
        or child_summary.get("test_metrics_emitted") is not False
        or child_summary.get("dataset_cache_integrity") != expected_cache_integrity
        or child_summary.get("model_split_usage") != expected_model_split_usage
    ):
        raise RuntimeError("candidate child does not certify validation-only profile selection")
    expected_axes = {
        "data": int(job["command"][job["command"].index("--data-seed") + 1]),
        "split": int(job["command"][job["command"].index("--split-seed") + 1]),
        "chart": int(job["command"][job["command"].index("--chart-seed") + 1]),
        "model": job["model_seed"],
    }
    if (
        child_summary.get("seed_axes") != expected_axes
        or child_manifest.get("seed_axes") != expected_axes
    ):
        raise RuntimeError("child seed axes do not match the requested job")
    for payload_name, payload in (("summary", child_summary), ("manifest", child_manifest)):
        settings = (
            payload.get("settings")
            if payload_name == "summary"
            else payload.get("effective_settings")
        )
        if not isinstance(settings, dict) or any(
            settings.get(key) != value for key, value in job["profile_config"].items()
        ):
            raise RuntimeError(f"child {payload_name} does not record the requested profile")
    if child_manifest.get("settings_overrides") != job["profile_config"]:
        raise RuntimeError("child manifest does not record the exact scaling overrides")
    models = child_summary.get("models")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise RuntimeError("child summary must contain exactly fixed_bfs and multi_chart")
    expected_quadrants = {
        "validation_graph_fresh_chart_seen_family",
        "validation_graph_fresh_chart_unseen_family",
    }
    checked_metrics: dict[str, Any] = {}
    checked_parameters: dict[str, Any] = {}
    selection_objectives: dict[str, Any] = {}
    for model_name in MODELS:
        model = models[model_name]
        if (
            not isinstance(model, dict)
            or model.get("optimizer_updates") != job["profile_config"]["optimizer_updates"]
        ):
            raise RuntimeError(f"child {model_name} update count does not match its profile")
        quadrants = model.get("quadrants")
        if not isinstance(quadrants, dict) or set(quadrants) != expected_quadrants:
            raise RuntimeError(f"child {model_name} evaluation quadrants are incomplete")
        checked_metrics[model_name] = {
            quadrant: _finite_metric_mapping(values, f"{model_name}/{quadrant}")
            for quadrant, values in quadrants.items()
        }
        parameters = model.get("parameters")
        if not isinstance(parameters, dict) or set(parameters) != {"total", "trainable"}:
            raise RuntimeError(f"child {model_name} parameter counts are missing")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in parameters.values()
        ):
            raise RuntimeError(f"child {model_name} parameter counts are invalid")
        if child_summary.get("parameter_counts", {}).get(model_name) != parameters:
            raise RuntimeError(f"child {model_name} parameter count records disagree")
        checked_parameters[model_name] = parameters
        metric_name = "graph_macro_accuracy" if job["suite"] == "csl" else "graph_macro_mae"
        direction = "maximize" if job["suite"] == "csl" else "minimize"
        values = [checked_metrics[model_name][quadrant][metric_name] for quadrant in quadrants]
        objective_value = sum(values) / len(values)
        selection_objectives[model_name] = {
            "metric": f"mean_validation_{metric_name}_across_seen_and_unseen_chart_families",
            "direction": direction,
            "value": objective_value,
            "components": {
                quadrant: checked_metrics[model_name][quadrant][metric_name]
                for quadrant in quadrants
            },
            "rank_score": objective_value if direction == "maximize" else -objective_value,
        }
    comparison = child_summary.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("paper_headline_eligible") is not False:
        raise RuntimeError("validation candidate must not claim test headline eligibility")
    if comparison.get("fixed_and_multi_optimizer_updates_matched") is not True:
        raise RuntimeError("child did not match fixed/multi optimizer updates")
    checkpoints = child_summary.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(MODELS):
        raise RuntimeError("child checkpoints are incomplete")
    required_artifacts = {"summary.json": summary_path}
    checked_checkpoints: dict[str, Any] = {}
    for model_name, raw_path in checkpoints.items():
        checkpoint = Path(raw_path).expanduser().resolve()
        if not checkpoint.is_relative_to(output) or not checkpoint.is_file():
            raise RuntimeError(f"child checkpoint is missing or outside its output: {checkpoint}")
        required_artifacts[checkpoint.name] = checkpoint
        checked_checkpoints[model_name] = {"path": str(checkpoint)}
    artifacts = child_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("child manifest artifact table is missing")
    artifact_hashes: dict[str, str] = {}
    for name, path in required_artifacts.items():
        record = artifacts.get(name)
        digest = _sha256(path)
        if not isinstance(record, dict) or record.get("sha256") != digest:
            raise RuntimeError(f"child artifact hash mismatch: {name}")
        artifact_hashes[name] = digest
    for model_name in MODELS:
        checked_checkpoints[model_name]["sha256"] = artifact_hashes[
            Path(checked_checkpoints[model_name]["path"]).name
        ]
    return {
        "suite": job["suite"],
        "profile": job["profile"],
        "profile_config": job["profile_config"],
        "model_seed": job["model_seed"],
        "seed_axes": expected_axes,
        "trained_models": list(MODELS),
        "parameter_counts": checked_parameters,
        "quadrant_metrics": checked_metrics,
        "selection_objectives": selection_objectives,
        "checkpoints": checked_checkpoints,
        "child_manifest_sha256": _sha256(manifest_path),
        "child_summary_sha256": _sha256(summary_path),
        "artifact_sha256": artifact_hashes,
        "child_metrics_checked": True,
        "dataset_cache_integrity": expected_cache_integrity,
        "model_split_usage": expected_model_split_usage,
        "test_evaluated": False,
        "test_used_for_selection": False,
    }


def _select_profiles(
    candidate_jobs: list[dict[str, Any]],
    *,
    suite: str,
    model_seeds: tuple[int, ...],
    profiles: tuple[str, ...],
) -> dict[str, Any]:
    candidates = {
        (job["profile"], job["model_seed"]): job for job in candidate_jobs if job["suite"] == suite
    }
    expected = {(profile, seed) for profile in profiles for seed in model_seeds}
    if set(candidates) != expected or any(
        candidates[key].get("status") != "passed" for key in expected
    ):
        raise RuntimeError(f"incomplete validation candidates for {suite}")
    conditions: dict[str, Any] = {}
    for model_name in MODELS:
        profile_aggregates: dict[str, Any] = {}
        for profile in profiles:
            components = [
                {
                    "model_seed": seed,
                    **candidates[(profile, seed)]["result"]["selection_objectives"][model_name],
                }
                for seed in model_seeds
            ]
            direction = components[0]["direction"]
            value = sum(component["value"] for component in components) / len(components)
            profile_aggregates[profile] = {
                "metric": components[0]["metric"],
                "direction": direction,
                "value": value,
                "rank_score": value if direction == "maximize" else -value,
                "seed_components": components,
            }
        selected_profile = max(
            profiles,
            key=lambda profile: profile_aggregates[profile]["rank_score"],
        )
        selected_by_seed: dict[str, Any] = {}
        for seed in model_seeds:
            result = candidates[(selected_profile, seed)]["result"]
            selected_by_seed[str(seed)] = {
                "checkpoint": result["checkpoints"][model_name],
                "parameter_counts": result["parameter_counts"][model_name],
                "validation_metrics": result["quadrant_metrics"][model_name],
                "candidate_summary_sha256": result["child_summary_sha256"],
            }
        conditions[model_name] = {
            "selected_profile": selected_profile,
            "profile_config": dict(PROFILE_CONFIGS[selected_profile]),
            "aggregate_validation_objective": profile_aggregates[selected_profile],
            "all_profile_aggregate_objectives": profile_aggregates,
            "selected_checkpoints_by_model_seed": selected_by_seed,
            "tie_break_order": list(profiles),
        }
    return {
        "suite": suite,
        "selection_split": "validation",
        "aggregation_axis": "mean_across_requested_model_seeds",
        "model_seeds": list(model_seeds),
        "conditions_selected_independently": True,
        "test_metrics_used_for_selection": False,
        "conditions": conditions,
    }


def _make_selected_test_job(
    args: argparse.Namespace,
    run_dir: Path,
    selection: dict[str, Any],
    model_seed: int,
) -> dict[str, Any]:
    suite = selection["suite"]
    selected_inputs = {
        name: {
            "selected_profile": selection["conditions"][name]["selected_profile"],
            "profile_config": selection["conditions"][name]["profile_config"],
            **selection["conditions"][name]["selected_checkpoints_by_model_seed"][str(model_seed)],
        }
        for name in MODELS
    }
    output = run_dir / "selected-test" / suite / f"model-seed-{model_seed}"
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "research.tree_augmentation.paper",
        "--suite",
        suite,
        "--data-root",
        str(args.data_root.expanduser().resolve()),
        "--output-dir",
        str(output),
        "--device",
        args.device,
        "--seed",
        str(args.data_seed),
        "--data-seed",
        str(args.data_seed),
        "--split-seed",
        str(args.split_seed),
        "--chart-seed",
        str(args.chart_seed),
        "--model-seed",
        str(model_seed),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--eval-charts-per-graph",
        str(PROFILE_CONFIGS["base"]["eval_charts_per_graph"]),
        "--evaluation-scope",
        "selected_test",
        "--fixed-checkpoint",
        selected_inputs["fixed_bfs"]["checkpoint"]["path"],
        "--multi-checkpoint",
        selected_inputs["multi_chart"]["checkpoint"]["path"],
    ]
    if args.allow_download:
        command.append("--allow-download")
    if args.amp is not None:
        command.append("--amp" if args.amp else "--no-amp")
    return {
        "suite": suite,
        "model_seed": model_seed,
        "selection": selection,
        "selected_inputs": selected_inputs,
        "evaluated_models": list(MODELS),
        "status": "pending",
        "output_dir": str(output),
        "summary_path": str(output / "summary.json"),
        "manifest_path": str(output / "manifest.json"),
        "log_path": str(run_dir / "logs" / f"{suite}--selected-test--seed-{model_seed}.log"),
        "command": command,
    }


def _validate_selected_test(job: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(job["summary_path"])
    manifest_path = Path(job["manifest_path"])
    child_summary = _read_mapping(summary_path, "selected-test summary.json")
    child_manifest = _read_mapping(manifest_path, "selected-test manifest.json")
    expected_axes = {
        "data": int(job["command"][job["command"].index("--data-seed") + 1]),
        "split": int(job["command"][job["command"].index("--split-seed") + 1]),
        "chart": int(job["command"][job["command"].index("--chart-seed") + 1]),
        "model": job["model_seed"],
    }
    expected_cache_integrity = {
        "full_cache_loaded": True,
        "all_declared_splits_validated": True,
        "loaded_and_validated_splits": ["test", "train", "validation"],
    }
    expected_model_split_usage = {
        "fit_splits": [],
        "evaluation_splits": ["test"],
        "selection_splits": [],
        "test_evaluated": True,
        "test_used_for_selection": False,
    }
    if (
        child_manifest.get("status") != "passed"
        or child_manifest.get("suite") != job["suite"]
        or child_manifest.get("seed_axes") != expected_axes
        or child_manifest.get("evaluation_scope") != "selected_test"
        or child_manifest.get("training_performed") is not False
        or child_manifest.get("dataset_cache_integrity") != expected_cache_integrity
        or child_manifest.get("model_split_usage") != expected_model_split_usage
    ):
        raise RuntimeError(
            "selected-test manifest does not certify test-only checkpoint evaluation"
        )
    if (
        child_summary.get("suite") != job["suite"]
        or child_summary.get("seed_axes") != expected_axes
        or child_summary.get("evaluation_scope") != "selected_test"
        or child_summary.get("training_performed") is not False
        or child_summary.get("test_metrics_emitted") is not True
        or child_summary.get("test_evaluations_per_selected_checkpoint") != 1
        or child_summary.get("dataset_cache_integrity") != expected_cache_integrity
        or child_summary.get("model_split_usage") != expected_model_split_usage
    ):
        raise RuntimeError("selected-test summary does not certify exactly one test phase")
    selection = job["selection"]
    selected_inputs = job["selected_inputs"]
    expected_inputs = {name: selected_inputs[name]["checkpoint"] for name in MODELS}
    if (
        child_summary.get("selected_checkpoints") != expected_inputs
        or child_manifest.get("selected_checkpoint_inputs") != expected_inputs
    ):
        raise RuntimeError("selected-test child did not use the selected checkpoint hashes")
    for record in expected_inputs.values():
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError("a selected checkpoint changed before result acceptance")
    expected_quadrants = {
        "test_graph_fresh_chart_seen_family",
        "test_graph_fresh_chart_unseen_family",
    }
    models = child_summary.get("models")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise RuntimeError("selected-test summary arms are incomplete")
    checked_metrics: dict[str, Any] = {}
    checked_parameters: dict[str, Any] = {}
    for model_name in MODELS:
        model = models[model_name]
        if (
            not isinstance(model, dict)
            or model.get("training_performed") is not False
            or model.get("history") != []
        ):
            raise RuntimeError(f"selected-test {model_name} appears to have been retrained")
        quadrants = model.get("quadrants")
        if not isinstance(quadrants, dict) or set(quadrants) != expected_quadrants:
            raise RuntimeError(f"selected-test {model_name} quadrants are incomplete")
        checked_metrics[model_name] = {
            quadrant: _finite_metric_mapping(values, f"selected-test/{model_name}/{quadrant}")
            for quadrant, values in quadrants.items()
        }
        parameters = model.get("parameters")
        expected_profile = selected_inputs[model_name]["profile_config"]
        expected_parameters = selected_inputs[model_name]["parameter_counts"]
        if (
            not isinstance(parameters, dict)
            or parameters != child_summary.get("parameter_counts", {}).get(model_name)
            or parameters != expected_parameters
        ):
            raise RuntimeError(f"selected-test {model_name} parameter counts are inconsistent")
        checkpoint_settings = child_summary.get("selected_checkpoint_settings", {}).get(model_name)
        if not isinstance(checkpoint_settings, dict) or any(
            checkpoint_settings.get(key) != value for key, value in expected_profile.items()
        ):
            raise RuntimeError(f"selected-test {model_name} architecture/profile mismatch")
        checked_parameters[model_name] = parameters
    artifacts = child_manifest.get("artifacts")
    summary_digest = _sha256(summary_path)
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != {"summary.json"}
        or not isinstance(artifacts["summary.json"], dict)
        or artifacts["summary.json"].get("sha256") != summary_digest
    ):
        raise RuntimeError("selected-test summary artifact hash is invalid")
    return {
        "suite": job["suite"],
        "model_seed": job["model_seed"],
        "evaluation_scope": "selected_test",
        "training_performed": False,
        "selected_profiles": {
            name: selection["conditions"][name]["selected_profile"] for name in MODELS
        },
        "selected_checkpoints": expected_inputs,
        "test_metrics": checked_metrics,
        "parameter_counts": checked_parameters,
        "dataset_cache_integrity": expected_cache_integrity,
        "model_split_usage": expected_model_split_usage,
        "test_evaluated": True,
        "test_used_for_selection": False,
        "test_evaluations_per_selected_checkpoint": 1,
        "child_manifest_sha256": _sha256(manifest_path),
        "child_summary_sha256": summary_digest,
    }


def _aggregate_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    completed = [job for job in manifest["jobs"] if job["status"] == "passed"]
    failed = [job for job in manifest["jobs"] if job["status"] == "failed"]
    selected_jobs = manifest.get("selected_test_jobs", [])
    completed_selected = [job for job in selected_jobs if job["status"] == "passed"]
    failed_selected = [job for job in selected_jobs if job["status"] == "failed"]
    return {
        "schema_version": 2,
        "suite": "tree_scaling",
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "planned_child_runs": len(manifest["jobs"]),
        "planned_model_trainings": len(manifest["jobs"]) * len(MODELS),
        "completed_child_runs": len(completed),
        "completed_model_trainings": len(completed) * len(MODELS),
        "failed_child_runs": len(failed),
        "planned_profile_selections": manifest["planned_profile_selections"],
        "completed_profile_selections": len(manifest.get("selections", [])) * len(MODELS),
        "planned_selected_test_runs": manifest["planned_selected_test_runs"],
        "completed_selected_test_runs": len(completed_selected),
        "failed_selected_test_runs": len(failed_selected),
        "planned_selected_checkpoint_test_evaluations": manifest["planned_selected_test_runs"]
        * len(MODELS),
        "completed_selected_checkpoint_test_evaluations": len(completed_selected) * len(MODELS),
        "models_per_child": list(MODELS),
        "profile_configs": manifest["config"]["profile_configs"],
        "chart_family_isolation": manifest["protocol"]["chart_family_isolation"],
        "results": [job["result"] for job in completed],
        "selections": manifest.get("selections", []),
        "selected_test_results": [job["result"] for job in completed_selected],
        **({"error": manifest["error"]} if "error" in manifest else {}),
    }


def _write_state(run_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "summary.json", _aggregate_summary(manifest))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    run_id = args.run_id or _default_run_id()
    run_dir = (
        args.results_root.expanduser().resolve() / "tree_augmentation/scaling" / run_id
    ).resolve()
    data_root = args.data_root.expanduser().resolve()
    if (
        run_dir == data_root
        or run_dir.is_relative_to(data_root)
        or data_root.is_relative_to(run_dir)
    ):
        print("experiment outputs and dataset directories must not overlap", file=sys.stderr)
        return 2
    jobs = make_jobs(args, run_dir)
    planned_selected_test_runs = len(args.suites) * len(args.model_seeds)
    if args.dry_run:
        print(
            f"{len(jobs)} validation-candidate child runs; "
            f"{len(jobs) * len(MODELS)} fresh model trainings; "
            f"{len(args.suites) * len(MODELS)} aggregate profile selections; "
            f"{planned_selected_test_runs * len(MODELS)} selected-checkpoint test evaluations"
        )
        for job in jobs:
            print(shlex.join(job["command"]))
        print(f"Aggregate: {run_dir / 'summary.json'}")
        return 0
    if run_dir.exists():
        print(f"run already exists; use a new run ID: {run_dir}", file=sys.stderr)
        return 2
    try:
        dependencies = check_dependencies()
    except DependencyCheckError as error:
        print(error_message(error), file=sys.stderr)
        return error.exit_code
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "suite": "tree_scaling",
        "run_id": run_id,
        "status": "running",
        "source_integrity_valid": True,
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config": {
            "suites": list(args.suites),
            "profiles": list(args.profiles),
            "profile_configs": {profile: PROFILE_CONFIGS[profile] for profile in args.profiles},
            "model_seeds": list(args.model_seeds),
            "data_seed": args.data_seed,
            "split_seed": args.split_seed,
            "chart_seed": args.chart_seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "amp_override": args.amp,
            "allow_download": args.allow_download,
            "data_root": str(data_root),
        },
        "models_per_child": list(MODELS),
        "planned_child_runs": len(jobs),
        "planned_model_trainings": len(jobs) * len(MODELS),
        "planned_selected_test_runs": planned_selected_test_runs,
        "planned_profile_selections": len(args.suites) * len(MODELS),
        "jobs": jobs,
        "selections": [],
        "selected_test_jobs": [],
        "dependencies": dependencies,
        "sources": _source_snapshot(),
        "protocol": {
            "fresh_training": "every profile/seed/suite child trains V1 fixed_bfs and V2 "
            "multi_chart independently; no checkpoint or score reuse",
            "scaling_axes": "model width and real encoder message-layer depth",
            "controlled_budget": "all profiles use 800 optimizer updates, 8 multi-chart "
            "training views per graph, and 8 evaluation charts per family",
            "paired_comparison": "within each child both arms use the same model width, update "
            "count, initialization seed, batch size, dataset split, and evaluation charts",
            "chart_family_isolation": {
                "fixed_train": "one bfs chart rooted at node 0 per graph",
                "multi_train": "8 charts per graph split across random-root bfs/dfs",
                "seen_evaluation": "fresh random-root bfs charts",
                "unseen_evaluation": "fresh Wilson uniform spanning-tree charts",
                "evaluation_charts_per_family": 8,
            },
            "selection": {
                "split": "official validation only; full caches are integrity-validated, but "
                "candidate model paths do not evaluate test data or use test metrics",
                "by": "suite x condition (fixed_bfs and multi_chart separately), using the "
                "mean preregistered scalar across all requested model seeds",
                "csl_objective": "maximize mean graph_macro_accuracy across fresh BFS and "
                "fresh Wilson validation chart families",
                "zinc_objective": "minimize mean graph_macro_mae across fresh BFS and fresh "
                "Wilson validation chart families",
                "tie_break": "first profile in the preregistered --profiles order",
                "test": "one test-only phase per selected checkpoint; no optimizer or retraining",
            },
            "failure_policy": "stop at first failed or unverifiable child; leave pending jobs "
            "unclaimed and publish failed aggregate state",
            "uncertainty": "default five model seeds; report per-seed metrics without "
            "pretending child models are independent datasets",
            "device": "CUDA required; no CPU fallback",
        },
    }
    _write_state(run_dir, manifest)
    environment = _environment()
    current_job: dict[str, Any] | None = None
    try:
        preflight_path = run_dir / "gpu-preflight.json"
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
            str(preflight_path),
        ]
        status = _run_logged(preflight, run_dir / "logs/preflight.log", environment)
        if status:
            raise RuntimeError(f"GPU preflight failed with exit code {status}")
        preflight_result = _read_mapping(preflight_path, "gpu-preflight.json")
        if preflight_result.get("status") != "passed":
            raise RuntimeError("GPU preflight returned without a passed certificate")
        manifest["gpu_preflight"] = {
            "path": str(preflight_path),
            "sha256": _sha256(preflight_path),
        }
        _write_state(run_dir, manifest)
        print(
            f"Run: {run_id}; {len(jobs)} child runs; "
            f"{len(jobs) * len(MODELS)} fresh model trainings",
            flush=True,
        )
        for index, job in enumerate(jobs, start=1):
            current_job = job
            _check_sources(manifest)
            job["status"] = "running"
            _write_state(run_dir, manifest)
            print(
                f"\n[{index}/{len(jobs)}] {job['suite']} / {job['profile']} / "
                f"model seed {job['model_seed']} (fixed_bfs + multi_chart)",
                flush=True,
            )
            started = time.monotonic()
            status = _run_logged(job["command"], Path(job["log_path"]), environment)
            job.update(exit_code=status, elapsed_seconds=time.monotonic() - started)
            _check_sources(manifest)
            if status:
                raise RuntimeError(
                    f"{job['suite']}/{job['profile']}/seed-{job['model_seed']} failed "
                    f"with exit code {status}"
                )
            job["result"] = _validate_child(job)
            job["status"] = "passed"
            _write_state(run_dir, manifest)
            current_job = None
        for suite in args.suites:
            _check_sources(manifest)
            selection = _select_profiles(
                jobs,
                suite=suite,
                model_seeds=args.model_seeds,
                profiles=args.profiles,
            )
            manifest["selections"].append(selection)
            _write_state(run_dir, manifest)
            for model_seed in args.model_seeds:
                selected_job = _make_selected_test_job(args, run_dir, selection, model_seed)
                manifest["selected_test_jobs"].append(selected_job)
                current_job = selected_job
                selected_job["status"] = "running"
                _write_state(run_dir, manifest)
                print(
                    f"\n[selected test] {suite} / model seed {model_seed}: "
                    f"fixed={selection['conditions']['fixed_bfs']['selected_profile']}, "
                    f"multi={selection['conditions']['multi_chart']['selected_profile']}",
                    flush=True,
                )
                started = time.monotonic()
                status = _run_logged(
                    selected_job["command"], Path(selected_job["log_path"]), environment
                )
                selected_job.update(
                    exit_code=status,
                    elapsed_seconds=time.monotonic() - started,
                )
                _check_sources(manifest)
                if status:
                    raise RuntimeError(
                        f"{suite}/seed-{model_seed} selected test failed with exit code {status}"
                    )
                selected_job["result"] = _validate_selected_test(selected_job)
                selected_job["status"] = "passed"
                _write_state(run_dir, manifest)
                current_job = None
        _check_sources(manifest)
        manifest.update(status="passed", finished_at_utc=dt.datetime.now(dt.UTC).isoformat())
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            finished_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        )
        if current_job is not None:
            current_job.update(status="failed", error=manifest["error"])
        _write_state(run_dir, manifest)
        print(
            f"Failed: {manifest['error']}\nSaved partial results: {run_dir}",
            file=sys.stderr,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    _write_state(run_dir, manifest)
    print(f"Passed aggregate: {run_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
