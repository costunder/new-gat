#!/usr/bin/env python3
"""Staged, research-scale V5 mechanism comparisons with one common measured resource plan.

Calibration probes every selected variant at the SAME physical batch/worker
candidates before any final training. Existing standalone V5 runs are never
opened, migrated, deleted or relabelled. Test data is not used for model selection.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import math
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_bytes, atomic_write_json  # noqa: E402
from research.conductance_gat.v5.protocol import (  # noqa: E402
    DATASETS,
    HARDWARE_PROFILES,
    SAMPLING_CHOICES,
    add_sampling_context_arguments,
    sampling_context_configuration,
)
from scripts import calibrate_training_resources as calibration  # noqa: E402
from scripts import run_conductance_v5 as standalone  # noqa: E402
from scripts import training_resource_plan as resources  # noqa: E402
from scripts.calibration_lock import calibration_lock  # noqa: E402

SUITE = "conductance_v5_mechanism_experiments_v1"
SUITES = ("core", "generators", "solvers", "filters")


def variants(suites: list[str]) -> list[dict[str, Any]]:
    """Union by scientific condition, so shared controls train once within the run."""
    result: dict[str, dict[str, Any]] = {}

    def add(
        identifier,
        family,
        *,
        fixed=False,
        backend="optimization",
        generator="optimized",
        heads="shared",
        normalization="symmetric",
        barrier=0.1,
        propagation_filter="linear",
    ):
        configuration = {
            "conductance_backend": backend,
            "conductance_generator": generator,
            "conductance_heads": heads,
            "propagation_normalization": normalization,
            "solver_degree_barrier": barrier,
            "propagation_filter": propagation_filter,
        }
        condition = "fixed_c" if fixed else "shared_dynamic_c"
        if identifier in result:
            if (
                result[identifier]["configuration"] != configuration
                or result[identifier]["condition"] != condition
            ):
                raise ValueError("a mechanism variant identifier has conflicting definitions")
            result[identifier]["suites"].append(family)
        else:
            result[identifier] = {
                "variant_id": identifier,
                "condition": condition,
                "configuration": configuration,
                "suites": [family],
            }

    for family in suites:
        if family == "core":
            for normalization in ("symmetric", "row"):
                add(f"fixed-{normalization}", family, fixed=True, normalization=normalization)
                add(f"optimized-shared-{normalization}", family, normalization=normalization)
                add(
                    f"optimized-per-head-{normalization}",
                    family,
                    heads="per_head",
                    normalization=normalization,
                )
        elif family == "generators":
            add("fixed-symmetric", family, fixed=True)
            add("degree-only-symmetric", family, generator="degree_only")
            add("mlp-shared-symmetric", family, backend="mlp")
            add("optimized-shared-symmetric", family)
        elif family == "solvers":
            add("optimized-shared-symmetric", family)
            add("optimized-no-barrier-symmetric", family, barrier=0.0)
            add("entropy-exact-symmetric", family, generator="entropy_exact", barrier=0.0)
        elif family == "filters":
            add("fixed-symmetric", family, fixed=True)
            add("optimized-shared-symmetric", family)
            add("fixed-polynomial3-symmetric", family, fixed=True, propagation_filter="polynomial3")
            add("optimized-polynomial3-symmetric", family, propagation_filter="polynomial3")
        else:
            raise ValueError(f"unsupported mechanism suite: {family}")
    return list(result.values())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suites", nargs="+", choices=SUITES, default=["core", "generators"])
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    result.add_argument(
        "--profiles", nargs="+", choices=("reference", "large"), default=["reference", "large"]
    )
    result.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    result.add_argument("--run-id", required=True)
    result.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--hardware-profile", choices=tuple(HARDWARE_PROFILES), default="a6000-48gb"
    )
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument(
        "--workers",
        type=int,
        default=4,
        help="PPI DataLoader initial candidate; measured before training",
    )
    result.add_argument("--ppi-batch-size", type=int)
    result.add_argument("--sample-seed-batch-size", type=int)
    result.add_argument("--edge-chunk-size", type=int)
    result.add_argument(
        "--activation-checkpoint", action=argparse.BooleanOptionalAction, default=None
    )
    result.add_argument(
        "--sampling",
        choices=SAMPLING_CHOICES,
        default="auto",
        help="common sampler for every variant; auto preserves legacy cluster sampling on arxiv",
    )
    result.add_argument("--num-neighbors", nargs="+", type=int, default=[15, 10])
    add_sampling_context_arguments(result)
    result.add_argument("--beta-initial", type=float, default=0.5)
    result.add_argument("--solver-steps", type=int, default=8)
    result.add_argument("--solver-step-size", type=float, default=0.25)
    result.add_argument("--solver-entropy", type=float, default=1.0)
    result.add_argument(
        "--learning-budget-policy",
        choices=("epochs", "reference_updates"),
        default="reference_updates",
    )
    result.add_argument("--min-free-gb", type=float, default=8.0)
    result.add_argument("--audit-reference-steps", type=int, default=64)
    result.add_argument("--audit-reference-tolerance", type=float, default=1e-4)
    result.add_argument(
        "--repeat-evaluations",
        type=int,
        default=5,
        help="read-only repetitions of one checkpoint; not additional training seeds",
    )
    result.add_argument(
        "--head-gradient-conflict",
        action="store_true",
        help="opt-in expensive per-head gradient conflict audit; no optimizer update",
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--calibration-only",
        action="store_true",
        help="measure the entire selected comparison set; do not launch final training",
    )
    return result


def validate_args(args) -> None:
    if not standalone.RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run-id must be a safe 1-120 character experiment identifier")
    for name in ("suites", "datasets", "profiles", "model_seeds"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
    if any(seed < 0 for seed in args.model_seeds):
        raise ValueError("model seeds must be nonnegative")
    if args.epochs < 4 or args.patience < 1 or args.workers < 0:
        raise ValueError("invalid full research training budget or initial worker configuration")
    if not str(args.device).startswith("cuda"):
        raise ValueError("mechanism training requires CUDA; no CPU fallback")
    if args.audit_reference_steps <= args.solver_steps or args.repeat_evaluations < 5:
        raise ValueError(
            "audit reference steps must exceed training K and repeat-evaluations must be at least 5"
        )
    if not math.isfinite(args.audit_reference_tolerance) or args.audit_reference_tolerance <= 0:
        raise ValueError("audit reference tolerance must be finite and positive")
    sampling_context_configuration(args)


def _standalone_args(args, variant, profile, seed):
    options = ["--profile", profile, "--datasets", *args.datasets, "--model-seed", str(seed)]
    names = (
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
        "beta_initial",
        "solver_steps",
        "solver_step_size",
        "solver_entropy",
        "learning_budget_policy",
        "min_free_gb",
    )
    for name in names:
        value = getattr(args, name)
        if value is not None:
            options += ["--" + name.replace("_", "-"), str(value)]
    options += ["--num-neighbors", *(str(value) for value in args.num_neighbors)]
    if args.activation_checkpoint is not None:
        options.append(
            "--activation-checkpoint"
            if args.activation_checkpoint
            else "--no-activation-checkpoint"
        )
    options += [
        "--training-schedule",
        "joint",
        "--solver-cost-scaling",
        "legacy_unit"
        if variant["configuration"]["conductance_backend"] == "mlp"
        else "width_scaled",
    ]
    for name, value in variant["configuration"].items():
        options += ["--" + name.replace("_", "-"), str(value)]
    selected = standalone.parser().parse_args(options)
    standalone._validate(selected)
    return selected


def make_jobs(args, run_dir: Path) -> list[dict[str, Any]]:
    planned = []
    selected_variants = variants(args.suites)
    for profile in args.profiles:
        for seed in args.model_seeds:
            for variant in selected_variants:
                selected = _standalone_args(args, variant, profile, seed)
                namespace = (
                    run_dir / "variants" / variant["variant_id"] / profile / f"model-seed-{seed}"
                )
                children = standalone.make_jobs(
                    selected, namespace, standalone._architecture(selected)
                )
                for child in children:
                    if child["condition"] != variant["condition"]:
                        continue
                    child.update(
                        variant=copy.deepcopy(variant),
                        variant_id=variant["variant_id"],
                        track="conductance",
                        profile=profile,
                        model_seed=seed,
                        job_id=f"{profile}/{child['dataset']}/model-seed-{seed}/{variant['variant_id']}",
                    )
                    planned.append(child)
    return planned


def _config(args) -> dict[str, Any]:
    return {
        key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key
        not in {
            "dry_run",
            "calibration_only",
            "run_id",
            "audit_reference_steps",
            "audit_reference_tolerance",
            "repeat_evaluations",
            "head_gradient_conflict",
        }
    }


def _job_identity(job):
    return {
        **standalone._identity(job),
        "variant": job["variant"],
        "profile": job["profile"],
        "model_seed": job["model_seed"],
    }


def _calibration_jobs(jobs):
    # Actual train --condition remains fixed_c/shared_dynamic_c. Only measurement
    # identity is promoted to a unique scientific variant for common selection.
    return [
        {**job, "training_condition": job["condition"], "condition": job["variant_id"]}
        for job in jobs
    ]


def _grouped(jobs):
    grouped = {}
    for job in jobs:
        grouped.setdefault((job["profile"], job["dataset"]), []).append(job)
    return grouped


def _validate_common_entry(entry, jobs, allocated_cpus):
    conditions = {job["variant_id"] for job in jobs}
    resources._validate_entry(
        entry,
        sorted({job["model_seed"] for job in jobs}),
        allocated_cpus=allocated_cpus,
        expected_conditions=conditions,
    )
    expected = [
        {
            "condition": job["variant_id"],
            "model_seed": job["model_seed"],
            "argv_sha256": resources.command_identity(job["command"]),
        }
        for job in jobs
    ]
    if entry["job_contracts"] != expected:
        raise ValueError(
            "common calibration did not measure the exact requested variants and seeds"
        )


def _apply_common_resources(jobs, entries):
    resolved = copy.deepcopy(jobs)
    lookup = {(entry["profile"], entry["dataset"]): entry for entry in entries}
    for job in resolved:
        entry = lookup[(job["profile"], job["dataset"])]
        selected = entry["selected"]
        for name, value in selected.items():
            option = "--" + name.replace("_", "-")
            if job["command"].count(option) != 1:
                raise ValueError(f"measured resource option is missing/duplicated: {option}")
            job["command"][job["command"].index(option) + 1] = str(value)
        job["batch_size"] = selected["batch_size"]
        job["workers"] = selected["workers"]
        job["execution"].update(
            batch_size=selected["batch_size"],
            dataloader_workers=selected["workers"],
            persistent_workers=selected["workers"] > 0,
            prefetch_factor=2 if selected["workers"] > 0 else None,
        )
        for name in ("sample_seed_batch_size", "sample_context_workers"):
            if name in selected:
                job["execution"][name] = selected[name]
        resources.validate_job_plan(
            {"entries": [entry]},
            track="conductance",
            profile=job["profile"],
            dataset=job["dataset"],
            condition=job["variant_id"],
            model_seed=job["model_seed"],
            command=job["command"],
        )
    return resolved


def _resume_manifest(path, *, args, planned, sources, dependencies):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "suite": SUITE,
        "run_id": args.run_id,
        "config": _config(args),
        "source_sha256": sources,
        "dependencies": dependencies,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "existing mechanism configuration/source differs; use a new run ID; "
            "old results preserved"
        )
    if [_job_identity(job) for job in manifest.get("planned_jobs", [])] != [
        _job_identity(job) for job in planned
    ]:
        raise ValueError("existing mechanism variant matrix differs; no silent reuse or overwrite")
    return manifest


def _read_result(job):
    from research.conductance_gat.v5 import train
    from scripts.audit_v5_stages import inspect_evidence

    result = standalone._load_metrics(job)
    evidence = inspect_evidence(Path(job["metrics_path"]))
    payload, history = evidence["metrics"], evidence["history"]
    child = train.build_parser().parse_args(job["command"][job["command"].index("-m") + 2 :])
    train.validate_args(child)
    _validate_result_recipe(child, payload, history)
    output = Path(job["output_dir"])
    hashes = {}
    for filename, field in (
        ("best.pt", "checkpoint_sha256"),
        ("last.pt", "last_checkpoint_sha256"),
        ("history.json", "history_sha256"),
    ):
        path = output / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"completed mechanism artifact is absent or indirect: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != payload.get(field):
            raise ValueError(f"completed mechanism artifact hash mismatch: {path}")
        hashes[field] = actual
    initial_hashes = {
        name: payload.get(name)
        for name in (
            "shared_initial_state_sha256",
            "common_backbone_initial_state_sha256",
        )
    }
    if not all(resources._is_sha256(value) for value in initial_hashes.values()):
        raise ValueError(
            "completed mechanism lacks verified shared/common backbone initialization provenance"
        )
    return {
        **result,
        **hashes,
        **initial_hashes,
        "learning_budget": payload.get("learning_budget"),
        "effective_optimizer_steps_by_group": payload.get("effective_optimizer_steps_by_group"),
        "data_sha256": payload.get("protocol", {}).get("data_sha256"),
        "split_sha256": payload.get("protocol", {}).get("split_sha256"),
    }


def _validate_result_recipe(child, payload, history):
    """Bind complete configuration, budget and stopping evidence to the actual command.

    inspect_evidence additionally verifies the immutable identity, official data
    protocol, artifact hashes and selected validation epoch. This run never imports
    legacy training, so complete epoch history must start at one without gaps.
    """
    from research.conductance_gat.v5 import train

    if payload.get("configuration") != train.configuration(child):
        raise ValueError(
            "completed mechanism configuration differs from its exact measured child command"
        )
    if payload.get("transition_provenance") is not None or payload.get("epoch_offset", 0) != 0:
        raise ValueError("mechanism comparisons do not import or relabel legacy training")
    if payload.get("source_sha256") != train.implementation_source_hashes():
        raise ValueError(
            "completed mechanism source differs from the pinned training implementation"
        )
    budget = payload.get("learning_budget")
    if not isinstance(budget, dict) or budget != train.plan_learning_budget(
        child.epochs,
        child.patience,
        budget.get("reference_batches_per_epoch"),
        budget.get("actual_batches_per_epoch"),
        policy=child.learning_budget_policy,
    ):
        raise ValueError("completed mechanism learning budget does not match the child recipe")
    epochs = len(history)
    if (
        payload.get("epochs_run") != epochs
        or epochs < 1
        or epochs > budget["planned_epochs"]
        or [row.get("epoch") for row in history] != list(range(1, epochs + 1))
    ):
        raise ValueError("completed mechanism history does not contain contiguous complete epochs")
    expected_schedule = train.phase_schedule(
        budget["planned_epochs"],
        list(child.phase_fractions),
        child.training_schedule,
    )
    if (
        payload.get("schedule") != expected_schedule
        or payload["resume_identity"].get("schedule") != expected_schedule
    ):
        raise ValueError("completed mechanism phase schedule differs from its full budget")
    actual_batches = budget["actual_batches_per_epoch"]
    for row in history:
        if (
            row.get("train_batches") != actual_batches
            or row.get("optimizer_steps") != row["epoch"] * actual_batches
            or row.get("phase", {}).get("phase")
            != train.phase_at(expected_schedule, row["epoch"])[0]
        ):
            raise ValueError("completed mechanism has incomplete epoch/update/phase evidence")
    if payload.get("optimizer_steps") != epochs * actual_batches:
        raise ValueError("completed mechanism optimizer update total is inconsistent")
    if epochs < budget["planned_epochs"] and not train.budget_should_stop(
        child,
        budget,
        history,
        primary_best_epoch=payload["best_epoch"],
        joint_best_epoch=payload.get("joint_best_epoch", 0),
    ):
        raise ValueError("completed mechanism stopped before its declared budget and patience")


def _audit_command(args, job):
    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/audit_v5_stages.py"),
        "--root",
        job["output_dir"],
        "--data-root",
        str(args.data_root.expanduser().resolve()),
        "--device",
        args.device,
        "--reference-steps",
        str(args.audit_reference_steps),
        "--reference-tolerance",
        str(args.audit_reference_tolerance),
        "--repeat-evaluations",
        str(args.repeat_evaluations),
    ]
    if args.head_gradient_conflict:
        command.append("--head-gradient-conflict")
    return command


def _file_sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular audit log is missing or indirect: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _ensure_audit(args, job, environment, persist):
    command = _audit_command(args, job)
    prior = job.get("audit", {})
    checkpoint_hash = job["result"]["checkpoint_sha256"]
    if prior.get("status") == "passed" and prior.get("command") == command:
        if prior.get("checkpoint_sha256") != checkpoint_hash or (
            _file_sha(Path(prior["log_path"])) != prior.get("log_sha256")
        ):
            raise ValueError("completed audit evidence/checkpoint changed; no silent audit reuse")
        return
    if prior:
        job.setdefault("audit_attempts", []).append(copy.deepcopy(prior))
    log = standalone._next_log(Path(job["log_path"]).with_suffix(".audit.log"))
    audit = {
        "status": "running",
        "command": command,
        "log_path": str(log),
        "checkpoint_sha256": checkpoint_hash,
        "test_evaluated": False,
        "training_seed_repetitions": False,
    }
    job["audit"] = audit
    persist()
    print(
        f"[validation audit] {job['job_id']}; repeated evaluations={args.repeat_evaluations}; "
        f"reference K={args.audit_reference_steps}",
        flush=True,
    )
    started = time.monotonic()
    try:
        status = standalone.shared.run_logged(command, log, environment)
        audit.update(exit_code=status, elapsed_seconds=time.monotonic() - started)
        if status:
            raise RuntimeError(
                f"validation audit failed with child status {status}; "
                "completed training is retained"
            )
        audit.update(status="passed", log_sha256=_file_sha(log))
        persist()
    except (Exception, KeyboardInterrupt) as error:
        audit.update(status="failed", error=f"{type(error).__name__}: {error}")
        if log.is_file() and not log.is_symlink():
            audit["log_sha256"] = _file_sha(log)
        standalone.shared.run_failure_reporter(
            persist, original_error=error, action="independent audit status"
        )
        raise


def _check_comparison_contracts(jobs):
    for grouped in _grouped(jobs).values():
        by_seed = {}
        for job in grouped:
            if job.get("status") != "passed":
                continue
            by_seed.setdefault(job["model_seed"], []).append(job)
        for selected in by_seed.values():
            for key in ("data_sha256", "split_sha256", "learning_budget"):
                values = [job["result"].get(key) for job in selected]
                if any(value is None for value in values) or any(
                    value != values[0] for value in values
                ):
                    raise ValueError(f"mechanism variants do not share verified {key}")
            by_filter = {}
            for job in selected:
                name = job["variant"]["configuration"]["propagation_filter"]
                by_filter.setdefault(name, set()).add(job["result"]["shared_initial_state_sha256"])
            if any(len(values) != 1 for values in by_filter.values()):
                raise ValueError(
                    "same-filter mechanism variants have different common initialization"
                )
            common = [job["result"].get("common_backbone_initial_state_sha256") for job in selected]
            if not all(resources._is_sha256(value) for value in common) or len(set(common)) != 1:
                raise ValueError("mechanism variants have different common backbone initialization")


def _write_summary(run_dir, manifest):
    lines = [
        "# V5 mechanism experiment progress",
        "",
        "Validation only; seed-0 screening is not a multi-seed or SOTA claim.",
        "",
        "Every selected variant in a profile/dataset uses the same measured "
        "physical batch and workers.",
        "",
        "| Profile | Dataset | Seed | Variant | Training | Audit | Best validation | Best epoch |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for job in manifest["jobs"]:
        result = job.get("result", {})
        value = result.get("validation")
        score = (
            f"{100 * value:.4f}%"
            if isinstance(value, (int, float)) and math.isfinite(value)
            else "pending"
        )
        audit = job.get("audit", {})
        audit_label = audit.get("status", "pending")
        if audit.get("log_path"):
            relative = Path(audit["log_path"]).relative_to(run_dir).as_posix()
            audit_label = f"[{audit_label}](<{relative}>)"
        lines.append(
            f"| {job['profile']} | {job['dataset']} | {job['model_seed']} | {job['variant_id']} | "
            f"{job['status']} | {audit_label} | "
            f"{score} | {result.get('best_epoch', '')} |"
        )
    atomic_write_bytes(run_dir / "comparison.md", ("\n".join(lines) + "\n").encode())


def _ensure_calibration(args, manifest, persist):
    import torch

    hardware = calibration._hardware(args.device)
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if "hardware" in manifest and (
        manifest["hardware"] != hardware or manifest["runtime"] != runtime
    ):
        raise ValueError(
            "mechanism calibration hardware/runtime changed; existing evidence preserved"
        )
    manifest.setdefault("hardware", hardware)
    manifest.setdefault("runtime", runtime)
    if args.hardware_profile == "a6000-48gb" and (
        hardware["total_memory_bytes"] < 40 * 1024**3 or hardware["compute_capability"][0] < 8
    ):
        raise RuntimeError(
            "A6000 profile requires >=40 GiB visible VRAM and CUDA capability >=8; no downscale"
        )
    minimum = max(
        args.min_free_gb, 32.0 if args.hardware_profile == "a6000-48gb" else args.min_free_gb
    )
    free, _ = torch.cuda.mem_get_info(torch.device(args.device))
    if free < minimum * 1024**3:
        raise RuntimeError(
            f"calibration requires {minimum:.2f} GiB free; no other processes were changed"
        )
    entries = manifest["calibration_entries"]
    for (profile, dataset), jobs in _grouped(manifest["planned_jobs"]).items():
        aliases = _calibration_jobs(jobs)
        entry = next(
            (
                value
                for value in entries
                if (value["profile"], value["dataset"]) == (profile, dataset)
            ),
            None,
        )
        if entry is None:
            entry = {"track": "conductance", "profile": profile, "dataset": dataset}
            entries.append(entry)
        if entry.get("status") == "passed":
            _validate_common_entry(entry, jobs, hardware["allocated_cpu_count"])
            calibration.verify_plan_inputs({"entries": [entry]}, aliases)
        else:
            calibration._calibrate_group(aliases, entry, persist)
            _validate_common_entry(entry, jobs, hardware["allocated_cpu_count"])
        gc.collect()
        torch.cuda.empty_cache()
        persist()
    manifest["calibration_status"] = "passed"
    expected_jobs = _apply_common_resources(manifest["planned_jobs"], entries)
    if manifest.get("resources_applied"):
        if [_job_identity(job) for job in manifest["jobs"]] != [
            _job_identity(job) for job in expected_jobs
        ]:
            raise ValueError(
                "stored mechanism jobs differ from immutable common measured resources"
            )
    else:
        manifest["jobs"] = expected_jobs
        manifest["resources_applied"] = True
    persist()


def _run(args, run_dir, planned, sources, dependencies):
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        if manifest_path.is_symlink():
            raise ValueError("mechanism manifest must not be a symlink")
        manifest = _resume_manifest(
            manifest_path, args=args, planned=planned, sources=sources, dependencies=dependencies
        )
    else:
        unexpected = [path for path in run_dir.iterdir() if path.name != ".calibration.lock"]
        if unexpected:
            raise ValueError("mechanism directory has untracked contents; no results overwritten")
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
            "classification": "full_research_mechanism_comparisons",
            "test_evaluated": False,
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "comparison_contract": {
                "all_variants_share_measured_resources": True,
                "maximum_supervised_update_budget_shared": True,
                "same_sampler_across_variants": True,
                "sampling_rng_seed_shared": True,
                "initialization_verified_per_filter": True,
                "old_experiments_relabelled": False,
                "test_used_for_selection": False,
            },
        }

    def persist():
        atomic_write_json(manifest_path, manifest)

    current = None
    try:
        _ensure_calibration(args, manifest, persist)
        if resources.source_snapshot() != sources:
            raise ValueError(
                "mechanism implementation changed during calibration; measured evidence retained"
            )
        if args.calibration_only:
            manifest["status"] = "calibrated"
            persist()
            _write_summary(run_dir, manifest)
            return 0
        environment = standalone.shared._environment()
        environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        manifest["status"] = "running"
        for index, job in enumerate(manifest["jobs"], start=1):
            if resources.source_snapshot() != sources:
                raise ValueError(
                    "mechanism implementation changed during the run; no stale continuation"
                )
            if job.get("status") == "passed":
                if _read_result(job) != job.get("result"):
                    raise ValueError(f"completed artifacts changed: {job['job_id']}")
                print(f"[{index}/{len(planned)}] verified, skipping {job['job_id']}", flush=True)
                _ensure_audit(args, job, environment, persist)
                continue
            current = job
            command = list(job["command"])
            checkpoint = Path(job["output_dir"]) / "last.pt"
            if checkpoint.is_file():
                command.append("--resume")
            else:
                standalone._preserve_incomplete_child(job, run_dir)
            job.update(status="running", attempt_command=command)
            job.pop("error", None)
            persist()
            print(f"[{index}/{len(planned)}] {job['job_id']}", flush=True)
            started = time.monotonic()
            status = standalone.shared.run_logged(
                command, standalone._next_log(Path(job["log_path"])), environment
            )
            job.update(exit_code=status, elapsed_seconds=time.monotonic() - started)
            if status:
                raise RuntimeError(f"{job['job_id']} failed with child status {status}")
            job["result"] = _read_result(job)
            job["status"] = "passed"
            _check_comparison_contracts(manifest["jobs"])
            current = None
            persist()
            _ensure_audit(args, job, environment, persist)
            _write_summary(run_dir, manifest)
        _check_comparison_contracts(manifest["jobs"])
        if resources.source_snapshot() != sources:
            raise ValueError(
                "mechanism implementation changed during the final audit; evidence retained"
            )
        manifest.update(status="passed", finished_at_utc=dt.datetime.now(dt.UTC).isoformat())
        manifest.pop("error", None)
        persist()
        _write_summary(run_dir, manifest)
        print(f"Mechanism comparisons passed: {run_dir / 'comparison.md'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        if current is not None:
            current.update(status="failed", error=manifest["error"])
        standalone.shared.run_failure_reporter(
            persist, original_error=error, action="mechanism manifest persistence"
        )
        standalone.shared.run_failure_reporter(
            lambda: _write_summary(run_dir, manifest),
            original_error=error,
            action="mechanism progress summary",
        )
        print(
            f"Mechanism run stopped safely: {manifest['error']}\n"
            f"Preserved partial results: {run_dir}",
            file=sys.stderr,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate_args(args)
        results_root, data_root = (
            args.results_root.expanduser().resolve(),
            args.data_root.expanduser().resolve(),
        )
        run_dir = results_root / "conductance_gat" / "mechanisms" / args.run_id
        if (
            run_dir.resolve() != run_dir
            or run_dir.is_relative_to(data_root)
            or data_root.is_relative_to(run_dir)
        ):
            raise ValueError(
                "mechanism outputs must be direct paths outside the immutable dataset cache"
            )
        planned = make_jobs(args, run_dir)
        if args.dry_run:
            print(
                f"{len(variants(args.suites))} unique variants; "
                f"{len(planned)} full-size validation-only trainings; "
                f"profiles={args.profiles}; datasets={args.datasets}; "
                f"model_seeds={args.model_seeds}"
            )
            print(
                "Common measured calibration across ALL selected variants "
                "per profile/dataset precedes training."
            )
            print(
                f"Each completed checkpoint gets a separate validation-only distribution audit: "
                f"reference K={args.audit_reference_steps}, "
                f"repeat-evaluations={args.repeat_evaluations}; "
                "failed audits never restart passed training."
            )
            for job in planned:
                print(f"{job['job_id']}: {shlex.join(job['command'])}")
            print(
                "Plan only: no files, directories, processes, GPU probes "
                "or final training were created."
            )
            return 0
        dependencies = standalone.check_dependencies()
        sources = resources.source_snapshot()
        # Refuse incompatible existing runs before even acquiring/creating their lock.
        path = run_dir / "manifest.json"
        if path.exists():
            _resume_manifest(
                path, args=args, planned=planned, sources=sources, dependencies=dependencies
            )
        with calibration_lock(run_dir):
            return _run(args, run_dir, planned, sources, dependencies)
    except (ValueError, RuntimeError, OSError, standalone.DependencyCheckError) as error:
        print(f"Mechanism run refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
