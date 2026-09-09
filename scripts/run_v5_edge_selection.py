#!/usr/bin/env python3
"""Independent full-size forest/chord and corrupted-edge selection experiments.

Structure budgets are explicit scientific interventions, never an OOM fallback.
All arms share a measured physical batch/worker plan; previous V5 runs are untouched.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import platform
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_bytes, atomic_write_json  # noqa: E402
from research.conductance_gat.edge_selection import calibration  # noqa: E402
from research.conductance_gat.edge_selection.audit_compat import (  # noqa: E402
    require_source_compatibility,
)
from research.conductance_gat.v5.protocol import (  # noqa: E402
    DATASETS,
    HARDWARE_PROFILES,
    SAMPLING_CHOICES,
    add_sampling_context_arguments,
    sampling_context_configuration,
)
from scripts import calibrate_training_resources as hardware_tools  # noqa: E402
from scripts import run_conductance_v5 as standalone  # noqa: E402
from scripts import run_v5_mechanism_experiments as common  # noqa: E402
from scripts import training_resource_plan as resources  # noqa: E402
from scripts.calibration_lock import calibration_lock  # noqa: E402

SUITE = "conductance_v5_edge_selection_experiments_v1"
TRAIN_MODULE = "research.conductance_gat.edge_selection.train"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument(
        "--suites",
        nargs="+",
        choices=("structure", "corruption"),
        default=["structure", "corruption"],
    )
    result.add_argument("--chord-fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    result.add_argument(
        "--profiles", nargs="+", choices=("reference", "large"), default=["reference", "large"]
    )
    result.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--hardware-profile", choices=tuple(HARDWARE_PROFILES), default="a6000-48gb"
    )
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--ppi-batch-size", type=int)
    result.add_argument("--sample-seed-batch-size", type=int)
    result.add_argument("--edge-chunk-size", type=int)
    result.add_argument(
        "--activation-checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    result.add_argument("--sampling", choices=SAMPLING_CHOICES, default="auto")
    result.add_argument("--num-neighbors", nargs="+", type=int, default=[15, 10])
    add_sampling_context_arguments(result)
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument("--forest-seed", type=int, default=0)
    result.add_argument("--corruption-ratio", type=float, default=0.1)
    result.add_argument("--corruption-seed", type=int, default=0)
    result.add_argument("--negative-loss-weight", type=float, default=1.0)
    result.add_argument("--l0-weight", type=float, default=1e-4)
    result.add_argument("--gate-temperature", type=float, default=2 / 3)
    result.add_argument("--repeat-evaluations", type=int, default=5)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--calibration-only", action="store_true")
    return result


def validate_args(args):
    if not standalone.RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run-id must be a safe 1-120 character experiment identifier")
    for name in ("suites", "datasets", "profiles", "model_seeds", "chord_fractions"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in args.chord_fractions):
        raise ValueError("chord fractions are explicit budgets in [0, 1]")
    if any(value < 0 for value in [*args.model_seeds, args.forest_seed, args.corruption_seed]):
        raise ValueError("all seeds must be nonnegative")
    if args.epochs < 4 or args.patience < 1 or args.workers < 0 or args.repeat_evaluations < 5:
        raise ValueError("invalid full training/worker/audit budget")
    if not str(args.device).startswith("cuda"):
        raise ValueError("edge-selection experiments require CUDA; no CPU fallback")
    if not math.isfinite(args.corruption_ratio) or not 0 < args.corruption_ratio <= 1:
        raise ValueError("corruption experiment ratio must be explicit in (0, 1]")
    for name in ("negative_loss_weight", "gate_temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.l0_weight) or args.l0_weight < 0:
        raise ValueError("l0-weight must be finite and nonnegative")
    sampling_context_configuration(args)


def variants(args):
    selected = []

    def add(identifier, suite, mode, fraction, *, corruption=0.0, negative=0.0, l0=0.0):
        selected.append(
            {
                "variant_id": identifier,
                "suite": suite,
                "declared_chord_budget": fraction,
                "configuration": {
                    "selection_mode": mode,
                    "chord_fraction": fraction
                    if mode in {"forest_random", "forest_learned", "forest_cycle"}
                    else None,
                    "forest_seed": 0 if mode == "hard_concrete" else args.forest_seed,
                    "corruption_ratio": corruption,
                    "corruption_seed": args.corruption_seed if mode == "hard_concrete" else 0,
                    "negative_loss_weight": negative,
                    "l0_weight": l0,
                    "gate_temperature": args.gate_temperature if mode == "hard_concrete" else 2 / 3,
                },
            }
        )

    for suite in args.suites:
        if suite == "structure":
            add("structure-full", suite, "full", 1.0)
            add("structure-forest-only", suite, "forest_only", 0.0)
            for fraction in args.chord_fractions:
                if fraction in {0.0, 1.0}:
                    continue  # Endpoints are the identical full/forest-only controls above.
                for mode in ("forest_random", "forest_learned", "forest_cycle"):
                    add(f"structure-{mode.replace('_', '-')}-q{fraction!s}", suite, mode, fraction)
        else:
            add(
                "corruption-task-only",
                suite,
                "hard_concrete",
                1.0,
                corruption=args.corruption_ratio,
                l0=args.l0_weight,
            )
            add(
                "corruption-negative-aux",
                suite,
                "hard_concrete",
                1.0,
                corruption=args.corruption_ratio,
                negative=args.negative_loss_weight,
                l0=args.l0_weight,
            )
    return selected


def make_jobs(args, run_dir):
    result = []
    for profile in args.profiles:
        for seed in args.model_seeds:
            for variant in variants(args):
                options = [
                    "--profile",
                    profile,
                    "--datasets",
                    *args.datasets,
                    "--model-seed",
                    str(seed),
                    "--conductance-heads",
                    "per_head",
                    "--propagation-normalization",
                    "row",
                    "--conductance-backend",
                    "optimization",
                    "--solver-cost-scaling",
                    "width_scaled",
                    "--training-schedule",
                    "joint",
                    "--beta-initial",
                    "0.5",
                    "--learning-budget-policy",
                    "reference_updates",
                ]
                for name in (
                    "data_root",
                    "results_root",
                    "device",
                    "hardware_profile",
                    "epochs",
                    "patience",
                    "workers",
                    "ppi_batch_size",
                    "sample_seed_batch_size",
                    "edge_chunk_size",
                    "sampling",
                    "sample_context_seed_batch_size",
                    "sample_context_workers",
                    "min_free_gb",
                ):
                    value = getattr(args, name)
                    if value is not None:
                        options += ["--" + name.replace("_", "-"), str(value)]
                options += ["--num-neighbors", *(str(value) for value in args.num_neighbors)]
                options.append(
                    "--activation-checkpoint"
                    if args.activation_checkpoint
                    else "--no-activation-checkpoint"
                )
                baseline = standalone.parser().parse_args(options)
                standalone._validate(baseline)
                namespace = (
                    run_dir / "variants" / variant["variant_id"] / profile / f"model-seed-{seed}"
                )
                for job in standalone.make_jobs(
                    baseline, namespace, standalone._architecture(baseline)
                ):
                    if job["condition"] != "shared_dynamic_c":
                        continue
                    command = job["command"]
                    command[command.index("-m") + 1] = TRAIN_MODULE
                    for name, value in variant["configuration"].items():
                        if value is not None:
                            command += ["--" + name.replace("_", "-"), str(value)]
                    job.update(
                        variant=copy.deepcopy(variant),
                        variant_id=variant["variant_id"],
                        track="conductance",
                        profile=profile,
                        model_seed=seed,
                        job_id=f"{profile}/{job['dataset']}/model-seed-{seed}/{variant['variant_id']}",
                    )
                    result.append(job)
    return result


def _config(args):
    return {
        name: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for name, value in vars(args).items()
        if name not in {"dry_run", "calibration_only", "run_id", "repeat_evaluations"}
    }


def _resume(path, args, planned, sources, dependencies):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "suite": SUITE,
        "run_id": args.run_id,
        "config": _config(args),
        "dependencies": dependencies,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(
            "edge-selection run identity changed; use a new run ID; no old results overwritten"
        )
    if [common._job_identity(job) for job in manifest.get("planned_jobs", [])] != [
        common._job_identity(job) for job in planned
    ]:
        raise ValueError("edge-selection arm matrix differs; no silent resume")
    transition = require_source_compatibility(
        manifest.get("source_sha256"), sources, scope="manifest"
    )
    if transition is not None:
        transitions = manifest.setdefault("source_transitions", [])
        if transition not in transitions:
            transitions.append(transition)
    return manifest


def _ensure_calibration(args, manifest, persist):
    import torch

    hardware = hardware_tools._hardware(args.device)
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if "hardware" in manifest and (
        manifest["hardware"] != hardware or manifest["runtime"] != runtime
    ):
        raise ValueError("edge-selection resource hardware/runtime changed")
    if args.hardware_profile == "a6000-48gb" and (
        hardware["total_memory_bytes"] < 40 * 1024**3 or hardware["compute_capability"][0] < 8
    ):
        raise ValueError("A6000 profile requires >=40 GiB visible VRAM and capability >=8")
    free, _ = torch.cuda.mem_get_info(torch.device(args.device))
    required = max(
        args.min_free_gb, 32.0 if args.hardware_profile == "a6000-48gb" else args.min_free_gb
    )
    if free < required * 1024**3:
        raise RuntimeError(f"calibration requires {required:g} GiB free; no processes were changed")
    manifest.update(hardware=hardware, runtime=runtime)
    entries = manifest["calibration_entries"]
    for (profile, dataset), jobs in common._grouped(manifest["planned_jobs"]).items():
        entry = next(
            (item for item in entries if (item["profile"], item["dataset"]) == (profile, dataset)),
            None,
        )
        if entry is None:
            entry = {"profile": profile, "dataset": dataset}
            entries.append(entry)
        calibration.calibrate_group(jobs, entry, persist)
    resolved = common._apply_common_resources(manifest["planned_jobs"], entries)
    if manifest.get("resources_applied"):
        if [common._job_identity(job) for job in manifest["jobs"]] != [
            common._job_identity(job) for job in resolved
        ]:
            raise ValueError("saved child resources differ from the immutable common measurement")
    else:
        manifest.update(jobs=resolved, resources_applied=True)
    manifest["calibration_status"] = "passed"
    persist()


def _read_result(job):
    from research.conductance_gat.edge_selection import train
    from research.conductance_gat.v5.train import _canonical_sha256

    path = Path(job["metrics_path"])
    if path.is_symlink():
        raise ValueError("edge-selection metrics cannot be an indirect path")
    payload = train.inspect_completed(Path(job["output_dir"]))
    child = calibration.parse_job(job)
    config = train.configuration(child)
    if payload.get("status") != "passed" or payload.get("configuration") != config:
        raise ValueError("completed edge-selection result differs from the exact measured recipe")
    identity = payload.get("resume_identity")
    if not isinstance(identity, dict) or _canonical_sha256(identity) != payload.get(
        "resume_identity_sha256"
    ):
        raise ValueError("completed edge-selection resume identity hash mismatch")
    for key in ("research_suite", "dataset", "condition", "configuration", "source_sha256"):
        if payload.get(key) != identity.get(key):
            raise ValueError(f"edge-selection result and identity disagree on {key}")
    if payload.get("research_suite") != train.SUITE or payload.get("dataset") != job["dataset"]:
        raise ValueError("foreign experiment result cannot be imported")
    require_source_compatibility(
        payload.get("source_sha256"), train.implementation_source_hashes(), scope="training"
    )
    protocol = payload.get("protocol")
    if (
        not isinstance(protocol, dict)
        or identity.get("dataset_protocol") != protocol
        or identity.get("dataset_protocol_sha256") != _canonical_sha256(protocol)
    ):
        raise ValueError("edge-selection data/split protocol identity mismatch")
    if payload.get("test_evaluated") is not False:
        raise ValueError("training comparisons must be validation-only")
    output = Path(job["output_dir"])
    hashes = {}
    for filename, key in (
        ("best.pt", "checkpoint_sha256"),
        ("last.pt", "last_checkpoint_sha256"),
        ("history.json", "history_sha256"),
    ):
        actual = common._file_sha(output / filename)
        if actual != payload.get(key):
            raise ValueError(f"edge-selection artifact changed: {filename}")
        hashes[key] = actual
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or not history
        or payload.get("epochs_run") != len(history)
        or [row.get("epoch") for row in history] != list(range(1, len(history) + 1))
    ):
        raise ValueError("edge-selection completed epoch history is incomplete")
    best, value = (
        payload.get("best_epoch"),
        payload.get("best_validation", payload.get("validation")),
    )
    if (
        type(best) is not int
        or not 1 <= best <= len(history)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("edge-selection selected validation checkpoint is invalid")
    if history[best - 1].get("validation") != value:
        raise ValueError("edge-selection selected score differs from retained epoch history")
    initial = payload.get("shared_initial_state_sha256")
    if not resources._is_sha256(initial):
        raise ValueError("edge-selection lacks shared initialization provenance")
    if payload.get("optimizer_steps") != history[-1].get("optimizer_steps"):
        raise ValueError("edge-selection final optimizer-step evidence is inconsistent")
    return {
        "validation": value,
        "best_epoch": best,
        "epochs_run": len(history),
        "shared_initial_state_sha256": initial,
        "data_sha256": protocol.get("data_sha256"),
        "split_sha256": protocol.get("split_sha256"),
        "learning_budget": payload.get("learning_budget"),
        **hashes,
    }


def _compare(jobs):
    for group in common._grouped(jobs).values():
        for seed in {job["model_seed"] for job in group}:
            completed = [
                job["result"]
                for job in group
                if job["model_seed"] == seed and job["status"] == "passed"
            ]
            for key in (
                "shared_initial_state_sha256",
                "data_sha256",
                "split_sha256",
                "learning_budget",
            ):
                values = [result.get(key) for result in completed]
                if any(value is None for value in values) or any(
                    value != values[0] for value in values
                ):
                    raise ValueError(f"edge-selection arms do not share verified {key}")


def _audit(args, job, environment, persist):
    from research.conductance_gat.edge_selection import train

    command = [
        sys.executable,
        "-B",
        "-m",
        "research.conductance_gat.edge_selection.audit",
        "--root",
        job["output_dir"],
        "--data-root",
        str(args.data_root.expanduser().resolve()),
        "--device",
        args.device,
        "--repeat-evaluations",
        str(args.repeat_evaluations),
    ]
    prior = job.get("audit", {})
    checkpoint = job["result"]["checkpoint_sha256"]
    if prior.get("status") == "passed" and prior.get("command") == command:
        if prior.get("checkpoint_sha256") != checkpoint or common._file_sha(
            Path(prior["log_path"])
        ) != prior.get("log_sha256"):
            raise ValueError("completed edge-selection audit evidence changed")
        return
    if prior:
        job.setdefault("audit_attempts", []).append(copy.deepcopy(prior))
    log = standalone._next_log(Path(job["log_path"]).with_suffix(".audit.log"))
    state = {
        "status": "running",
        "command": command,
        "log_path": str(log),
        "checkpoint_sha256": checkpoint,
        "evaluator_source_sha256": train.implementation_source_hashes(),
    }
    job["audit"] = state
    persist()
    try:
        status = standalone.shared.run_logged(command, log, environment)
        state["exit_code"] = status
        if status:
            raise RuntimeError(
                f"edge-selection audit failed ({status}); completed training retained"
            )
        if state["evaluator_source_sha256"] != train.implementation_source_hashes():
            raise ValueError("edge-selection evaluator source changed during audit")
        state.update(status="passed", log_sha256=common._file_sha(log))
        persist()
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        if log.is_file() and not log.is_symlink():
            state["log_sha256"] = common._file_sha(log)
        standalone.shared.run_failure_reporter(
            persist, original_error=error, action="independent edge audit"
        )
        raise


def _summary(run_dir, manifest):
    lines = [
        "# Edge-selection experiment progress",
        "",
        "Validation only; no SOTA or multi-seed claim.",
        "",
        "Structure chord budgets and corruption auxiliary controls are separate scientific axes.",
        "",
        "| Profile | Dataset | Seed | Arm | Training | Audit | Validation | Epoch |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for job in manifest["jobs"]:
        result, audit = job.get("result", {}), job.get("audit", {})
        label = audit.get("status", "pending")
        if audit.get("log_path"):
            label = f"[{label}](<{Path(audit['log_path']).relative_to(run_dir).as_posix()}>)"
        score = f"{result['validation']:.6f}" if "validation" in result else "pending"
        lines.append(
            f"| {job['profile']} | {job['dataset']} | {job['model_seed']} | "
            f"{job['variant_id']} | {job['status']} | {label} | {score} | "
            f"{result.get('best_epoch', '')} |"
        )
    atomic_write_bytes(run_dir / "comparison.md", ("\n".join(lines) + "\n").encode())


def _run(args, run_dir, planned, sources, dependencies):
    path = run_dir / "manifest.json"
    if path.exists():
        manifest = _resume(path, args, planned, sources, dependencies)
    else:
        if any(item.name != ".calibration.lock" for item in run_dir.iterdir()):
            raise ValueError("new edge-selection directory has untracked contents; preserved")
        manifest = {
            "schema_version": 1,
            "suite": SUITE,
            "run_id": args.run_id,
            "status": "calibrating",
            "config": _config(args),
            "source_sha256": sources,
            "dependencies": dependencies,
            "planned_jobs": planned,
            "jobs": copy.deepcopy(planned),
            "calibration_entries": [],
            "test_evaluated": False,
            "legacy_results_imported": False,
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }

    def persist():
        atomic_write_json(path, manifest)

    current = None
    try:
        _ensure_calibration(args, manifest, persist)
        if resources.source_snapshot() != sources:
            raise ValueError("edge-selection implementation changed during calibration")
        if args.calibration_only:
            manifest["status"] = "calibrated"
            persist()
            _summary(run_dir, manifest)
            return 0
        environment = standalone.shared._environment()
        environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        manifest["status"] = "running"
        for index, job in enumerate(manifest["jobs"], 1):
            if resources.source_snapshot() != sources:
                raise ValueError("edge-selection implementation changed during the run")
            if job["status"] == "passed":
                if _read_result(job) != job.get("result"):
                    raise ValueError("completed edge-selection result changed")
                print(f"[{index}/{len(planned)}] verified, skipping {job['job_id']}", flush=True)
                _audit(args, job, environment, persist)
                continue
            current = job
            command = list(job["command"])
            checkpoint = Path(job["output_dir"]) / "last.pt"
            if checkpoint.is_file() and not checkpoint.is_symlink():
                command.append("--resume")
            else:
                standalone._preserve_incomplete_child(job, run_dir)
            job.update(status="running", attempt_command=command)
            persist()
            print(f"[{index}/{len(planned)}] {job['job_id']}", flush=True)
            started = time.monotonic()
            status = standalone.shared.run_logged(
                command, standalone._next_log(Path(job["log_path"])), environment
            )
            job.update(exit_code=status, elapsed_seconds=time.monotonic() - started)
            if status:
                raise RuntimeError(f"{job['job_id']} failed with child status {status}")
            job.update(result=_read_result(job), status="passed")
            _compare(manifest["jobs"])
            current = None
            persist()
            _audit(args, job, environment, persist)
            _summary(run_dir, manifest)
        _compare(manifest["jobs"])
        if resources.source_snapshot() != sources:
            raise ValueError("edge-selection implementation changed during the final audit")
        manifest.update(status="passed", finished_at_utc=dt.datetime.now(dt.UTC).isoformat())
        manifest.pop("error", None)
        persist()
        _summary(run_dir, manifest)
        print(f"Edge-selection comparisons passed: {run_dir / 'comparison.md'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        if current is not None:
            current.update(status="failed", error=manifest["error"])
        standalone.shared.run_failure_reporter(
            persist, original_error=error, action="edge manifest"
        )
        standalone.shared.run_failure_reporter(
            lambda: _summary(run_dir, manifest), original_error=error, action="edge summary"
        )
        print(
            f"Edge-selection stopped safely: {manifest['error']}\nPreserved: {run_dir}",
            file=sys.stderr,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate_args(args)
        data = args.data_root.expanduser().resolve()
        run_dir = (
            args.results_root.expanduser().resolve()
            / "conductance_gat/edge_selection"
            / args.run_id
        )
        if (
            run_dir.resolve() != run_dir
            or run_dir.is_relative_to(data)
            or data.is_relative_to(run_dir)
        ):
            raise ValueError(
                "edge-selection outputs must be direct paths outside the dataset cache"
            )
        planned = make_jobs(args, run_dir)
        if args.dry_run:
            print(
                f"{len(variants(args))} independent arms; {len(planned)} full-size trainings; "
                f"profiles={args.profiles}; datasets={args.datasets}; seeds={args.model_seeds}"
            )
            print(
                f"Explicit chord fractions={args.chord_fractions}; "
                f"corruption ratio={args.corruption_ratio}; "
                "per-head C + row propagation in every arm."
            )
            print(
                "Common measured train+optimizer+validation+preparation calibration "
                "precedes training; old V5 runs are untouched."
            )
            for job in planned:
                print(f"{job['job_id']}: {shlex.join(job['command'])}")
            print("Dry run only: no files, GPU probes, child processes or final training created.")
            return 0
        dependencies, sources = standalone.check_dependencies(), resources.source_snapshot()
        path = run_dir / "manifest.json"
        if path.exists():
            if path.is_symlink():
                raise ValueError("edge-selection manifest must not be indirect")
            _resume(path, args, planned, sources, dependencies)
        with calibration_lock(run_dir):
            return _run(args, run_dir, planned, sources, dependencies)
    except (ValueError, RuntimeError, OSError, standalone.DependencyCheckError) as error:
        print(f"Edge-selection refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
