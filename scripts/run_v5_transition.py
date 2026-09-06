#!/usr/bin/env python3
"""Preserve completed V5 controls and transition only remaining V5 work.

This is an explicit architecture transition, never a relaxed normal resume.
The source tree is read-only. A distinct destination owns all new checkpoints,
resource measurements and progress. Cycle/Tree/V1--V4 are never dispatched.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_json  # noqa: E402
from research.conductance_gat.v5.protocol import (  # noqa: E402
    add_conductance_arguments,
    conductance_arguments_configuration,
    learning_budget_arguments_configuration,
)
from scripts.training_resource_plan import (  # noqa: E402
    allocated_cpu_count,
    candidate_score,
    choose_candidate,
    completed_candidate_status,
    digest,
    source_snapshot,
    worker_candidates,
)

SUITE = "v5_preserved_state_transition_v1"
TRAINING_ACTIONS = {"transition_dynamic", "resume_incomplete_fixed", "fresh_dynamic", "fresh_fixed"}
TRANSITION_MODES = {"transition_dynamic": "replace_c", "resume_incomplete_fixed": "continue_fixed"}
EXECUTION_FIELDS = ("batch_size", "workers", "sample_seed_batch_size")
TRANSITION_OPTIONS = (
    "--transition-from-checkpoint",
    "--transition-source-sha256",
    "--transition-mode",
    "--transition-extra-epochs",
    "--transition-resource-certificate",
    "--transition-resource-sha256",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-manifest", type=Path)
    source.add_argument("--probe-job", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--confirm-source-stopped", action="store_true")
    budget = result.add_mutually_exclusive_group()
    budget.add_argument("--extra-epochs", type=int, default=0)
    budget.add_argument(
        "--extra-epochs-for",
        action="append",
        default=[],
        metavar="JOB_ID=N",
        help="extend only this exact V5 job; repeatable, incompatible with global extra epochs",
    )
    add_conductance_arguments(result)
    return result


def _regular(path: Path) -> Path:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"transition artifacts must not traverse symlinks: {path}")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"required transition artifact is missing: {resolved}")
    return resolved


def _sha(path: Path) -> str:
    path = _regular(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(_regular(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _set_option(arguments: list[str], option: str, value: Any) -> list[str]:
    result = list(arguments)
    positions = [i for i, token in enumerate(result) if token == option]
    if len(positions) > 1:
        raise ValueError(f"duplicate source command option: {option}")
    if positions:
        position = positions[0]
        if position + 1 == len(result) or result[position + 1].startswith("--"):
            raise ValueError(f"source command option has no value: {option}")
        result[position + 1] = str(value)
    else:
        result += [option, str(value)]
    return result


def _without_transition(arguments: list[str]) -> list[str]:
    result, index = [], 0
    while index < len(arguments):
        value = arguments[index]
        if value in TRANSITION_OPTIONS:
            if index + 1 >= len(arguments):
                raise ValueError(f"missing value for {value}")
            index += 2
        elif value in {"--resume", "--no-resume"}:
            index += 1
        else:
            result.append(value)
            index += 1
    return result


def _training_arguments(command: list[str], *, legacy: bool = False):
    from research.conductance_gat.v5 import train

    if (
        not isinstance(command, list)
        or not all(isinstance(value, str) for value in command)
        or command.count("-m") != 1
    ):
        raise ValueError("source job must carry an exact V5 module argv")
    position = command.index("-m")
    if command[position + 1] != "research.conductance_gat.v5.train":
        raise ValueError("transition refuses any child other than Conductance V5")
    arguments = _without_transition(command[position + 2 :])
    if legacy:
        for option, value in (("--conductance-backend", "mlp"), ("--training-schedule", "staged")):
            if option not in arguments:
                arguments += [option, value]
    try:
        args = train.build_parser().parse_args(arguments)
    except SystemExit as error:
        raise ValueError("source V5 command does not parse with its preserved options") from error
    train.validate_args(args)
    return args, arguments


def _resolve_source(path: Path):
    path = _regular(path)
    initial = _read(path)
    manifests = {str(path): _sha(path)}
    if initial.get("suite") == "rich_scaling":
        jobs = [job for job in initial.get("jobs", []) if job.get("track") == "conductance"]
        if len(jobs) != 1:
            raise ValueError("rich source must contain exactly one conductance child")
        directory = Path(jobs[0]["output_dir"])
        if not directory.is_absolute():
            directory = path.parent / directory
        path = _regular(directory / "manifest.json")
        manifests[str(path)] = _sha(path)
        initial = _read(path)
    if initial.get("suite") != "conductance_architecture_scaling_v1_v5":
        raise ValueError("source must be a Conductance scaling or rich-scaling manifest")
    if initial.get("schema_version") != 1 or not isinstance(initial.get("jobs"), list):
        raise ValueError("source scaling manifest schema is invalid")
    return path, initial, manifests


def _inspect_checkpoint(path: Path):
    from research.conductance_gat.v5.transition import inspect_transition_source

    return inspect_transition_source(path, expected_sha256=_sha(path))


def _historical_reference(job, manifest, path):
    from research.conductance_gat.v5.transition_report import validate_historical_reference

    return validate_historical_reference(job, manifest, source_manifest_path=path)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Read and validate every source child before any destination write or GPU work."""
    from research.conductance_gat.v5 import train

    if args.output_dir is None or args.extra_epochs < 0:
        raise ValueError("a distinct --output-dir and nonnegative --extra-epochs are required")
    requested = conductance_arguments_configuration(args)
    if learning_budget_arguments_configuration(args):
        raise ValueError(
            "reference_updates is not a legacy transition policy; preserve the source "
            "budget or configure a separate new run instead of silently changing it"
        )
    if (
        requested["conductance_backend"] != "optimization"
        or requested["training_schedule"] != "joint"
    ):
        raise ValueError("this transition replaces legacy C with optimization and joint training")
    source_path, manifest, sources = _resolve_source(args.source_manifest)
    extra_by_job = {}
    for item in args.extra_epochs_for:
        if "=" not in item:
            raise ValueError("--extra-epochs-for must be JOB_ID=N")
        job_id, amount = item.rsplit("=", 1)
        if not re.fullmatch(r"[0-9]+", amount) or job_id in extra_by_job:
            raise ValueError("extra epoch overrides must be unique nonnegative integers")
        extra_by_job[job_id] = int(amount)
    known_jobs = {job.get("job_id") for job in manifest["jobs"] if job.get("version") == "v5"}
    if set(extra_by_job) - known_jobs:
        raise ValueError("--extra-epochs-for names an unknown V5 job")
    output = args.output_dir.expanduser().resolve()
    if output == Path(output.anchor) or _overlap(output, source_path.parent):
        raise ValueError("transition output must be distinct from and outside the source run")
    if args.output_dir.is_symlink() or any(
        parent.is_symlink() for parent in args.output_dir.parents
    ):
        raise ValueError("transition output must not traverse symlinks")
    jobs, seen = [], set()
    for source_job in manifest["jobs"]:
        if source_job.get("version") != "v5":
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(source_job.get("profile", ""))):
            raise ValueError("source V5 profile is not a safe path component")
        expected_id = (
            f"v5/{source_job['profile']}/model-seed-{source_job['model_seed']}/"
            f"{source_job['dataset']}/{source_job['condition']}"
        )
        if source_job.get("job_id") != expected_id or expected_id in seen:
            raise ValueError("source V5 job IDs are missing, duplicated or inconsistent")
        seen.add(expected_id)
        if source_job["condition"] not in {"fixed_c", "shared_dynamic_c"}:
            raise ValueError("source V5 condition is unsupported")
        old_output = Path(source_job["output_dir"])
        if not old_output.is_absolute():
            old_output = source_path.parent / old_output
        if old_output.is_symlink() or any(parent.is_symlink() for parent in old_output.parents):
            raise ValueError("source output must not traverse symlinks")
        old_output = old_output.resolve()
        if not old_output.is_relative_to(source_path.parent):
            raise ValueError("source child output escapes its conductance run")
        for filename in ("metrics.json", "last.pt", "best.pt", "best.previous.pt", "history.json"):
            artifact = old_output / filename
            sources[str(artifact)] = (
                _sha(artifact) if artifact.exists() or artifact.is_symlink() else None
            )
        last = old_output / "last.pt"
        inspected = _inspect_checkpoint(last) if last.is_file() else None
        identity = inspected["identity"] if inspected else None
        source_config = identity["configuration"] if identity else None
        old_args, arguments = _training_arguments(source_job["command"], legacy=True)
        parsed_config = train.configuration(old_args)
        if source_config is not None:
            for key, value in source_config.items():
                if key in parsed_config and value != parsed_config[key]:
                    raise ValueError(
                        f"checkpoint/source command configuration differs: {expected_id}/{key}"
                    )
            if (identity.get("dataset"), identity.get("condition")) != (
                source_job["dataset"],
                source_job["condition"],
            ):
                raise ValueError("source checkpoint dataset/condition differs from its job")
        else:
            source_config = parsed_config
        if (old_args.dataset, old_args.condition, old_args.model_seed) != (
            source_job["dataset"],
            source_job["condition"],
            source_job["model_seed"],
        ):
            raise ValueError("source job and argv dataset/condition/seed differ")
        for name, value in source_job.get("architecture", {}).items():
            if parsed_config.get(name) != value:
                raise ValueError(f"source job architecture differs from argv: {name}")
        if Path(old_args.output_dir).resolve() != old_output:
            raise ValueError("source command output differs from its source job")
        data_root = old_args.data_root.expanduser().resolve()
        if _overlap(output, data_root):
            raise ValueError("transition outputs must not overlap the dataset cache")
        epoch = int(inspected["source_epoch"]) if inspected else 0
        extra_epochs = extra_by_job.get(expected_id, args.extra_epochs)
        total = old_args.epochs + extra_epochs
        fixed = old_args.condition == "fixed_c"
        historical = None
        if source_job.get("status") == "passed":
            historical = _historical_reference(source_job, manifest, source_path)
            if inspected is None:
                raise ValueError("completed source child has no verified last checkpoint")
        if fixed and source_job.get("status") == "passed":
            if extra_by_job.get(expected_id, 0):
                raise ValueError(
                    "completed fixed controls are preserved, not extended by a transition"
                )
            action = "reuse_completed_fixed"
            total, extra_epochs = old_args.epochs, 0
        elif inspected is not None:
            if fixed and (inspected["source_complete"] or total <= epoch):
                raise ValueError(
                    "fixed source checkpoint is finished but its parent result is not verified; "
                    "finalize/recover the original result before migrating, without retraining it"
                )
            action = "resume_incomplete_fixed" if fixed else "transition_dynamic"
            if not fixed and total <= epoch:
                action = "preserve_legacy_dynamic"
        else:
            action = "fresh_fixed" if fixed else "fresh_dynamic"
        target = output / "children" / expected_id
        command_args = _set_option(arguments, "--output-dir", target)
        command_args = _set_option(command_args, "--epochs", total)
        if not fixed or inspected is None:
            for name, value in requested.items():
                command_args = _set_option(command_args, "--" + name.replace("_", "-"), value)
        else:
            command_args = _set_option(
                command_args, "--conductance-backend", old_args.conductance_backend
            )
            command_args = _set_option(
                command_args, "--training-schedule", old_args.training_schedule
            )
        if action in TRANSITION_MODES:
            command_args += [
                "--transition-from-checkpoint",
                str(last),
                "--transition-source-sha256",
                inspected["sha256"],
                "--transition-mode",
                TRANSITION_MODES[action],
                "--transition-extra-epochs",
                str(extra_epochs),
            ]
        command = [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "research.conductance_gat.v5.train",
            *command_args,
        ]
        jobs.append(
            {
                "job_id": expected_id,
                "profile": source_job["profile"],
                "dataset": old_args.dataset,
                "condition": old_args.condition,
                "model_seed": old_args.model_seed,
                "action": action,
                "source_job": source_job,
                "source_output_dir": str(old_output),
                "source_configuration": source_config,
                "source_epoch": epoch,
                "source_complete": inspected["source_complete"] if inspected else False,
                "source_checkpoint": str(last) if inspected else None,
                "source_checkpoint_sha256": inspected["sha256"] if inspected else None,
                "source_epochs_requested": old_args.epochs,
                "target_total_epochs": total,
                "extra_epochs": extra_epochs,
                "remaining_epochs": 0
                if action == "reuse_completed_fixed"
                else max(0, total - epoch),
                "requires_extra_epoch_budget": action == "preserve_legacy_dynamic",
                "historical_reference": historical,
                "source_dataset_protocol": identity.get("dataset_protocol") if identity else None,
                "source_runtime_versions": identity.get("runtime_versions") if identity else None,
                "source_legacy_revision": inspected["legacy_revision"] if inspected else None,
                "baseline_execution": {name: getattr(old_args, name) for name in EXECUTION_FIELDS},
                "output_dir": str(target),
                "command": command,
                "resource_directory": str(output / "resource_calibration" / expected_id),
                "source_resource_plan_reference": manifest.get("config", {}).get("resource_plan"),
                "source_resource_plan_is_new_model_measurement": False,
                "minimum_free_gb": manifest.get("config", {}).get("min_free_gb", 8.0),
            }
        )
    if not jobs:
        raise ValueError("source manifest contains no Conductance V5 jobs")
    plan = {
        "schema_version": 1,
        "suite": SUITE,
        "source_manifest": str(source_path),
        "requested_source_manifest": str(args.source_manifest.expanduser().resolve()),
        "source_artifact_sha256": sources,
        "output_dir": str(output),
        "extra_epochs": args.extra_epochs,
        "extra_epochs_by_job": extra_by_job,
        "new_conductance": requested,
        "source_sha256": source_snapshot(),
        "jobs": jobs,
        "scope": "V5 only; old artifacts and all Cycle/Tree/V1-V4 are preserved",
        "classification": "explicit_architecture_transition_not_fresh_paired_comparison",
    }
    _verify_sources(plan)
    return plan


def _verify_sources(plan: dict[str, Any]) -> None:
    for name, expected in plan["source_artifact_sha256"].items():
        path = Path(name)
        actual = _sha(path) if path.exists() or path.is_symlink() else None
        if actual != expected:
            raise ValueError(
                f"source artifact changed; stop original work before transitioning: {path}"
            )
    if source_snapshot() != plan["source_sha256"]:
        raise ValueError("implementation source changed during transition; no further jobs started")


def _active_source_processes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Read-only Linux process check; never signal the original experiment."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    outputs = {job["source_output_dir"] for job in plan["jobs"]}
    run_ids = {
        _read(Path(path)).get("run_id")
        for path in plan["source_artifact_sha256"]
        if Path(path).name == "manifest.json"
    }
    found = []
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            command = [
                part.decode(errors="replace")
                for part in (directory / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        if "research.conductance_gat.v5.train" in command and "--output-dir" in command:
            index = command.index("--output-dir") + 1
            if index < len(command) and str(Path(command[index]).resolve()) in outputs:
                found.append({"pid": int(directory.name), "command": command})
        elif (
            any(
                Path(token).name
                in {"run_rich_scaling.py", "run_conductance_scaling.py", "run_conductance_v5.py"}
                for token in command
            )
            and "--run-id" in command
        ):
            index = command.index("--run-id") + 1
            if index < len(command) and command[index] in run_ids:
                found.append({"pid": int(directory.name), "command": command})
    return found


def _print_plan(plan: dict[str, Any]) -> None:
    print(plan["classification"], flush=True)
    for job in plan["jobs"]:
        print(
            f"{job['job_id']}: {job['action']}; source_epoch={job['source_epoch']}; "
            f"target_total={job['target_total_epochs']}; remaining={job['remaining_epochs']}",
            flush=True,
        )
    print(
        "Completed fixed controls remain historical references, not new paired controls.",
        flush=True,
    )
    print(
        "Budget-exhausted legacy dynamic results are retained and explicitly pending extra budget.",
        flush=True,
    )
    print(
        "Source work must be stopped explicitly; this runner never stops the source session.",
        flush=True,
    )


def _probe_runtime(job: dict[str, Any]):
    from research.conductance_gat.v5 import train
    from scripts.calibrate_training_resources import _hardware

    args, _ = _training_arguments(job["command"])
    runtime = train._versions()
    if job.get("source_runtime_versions") is not None and runtime != job["source_runtime_versions"]:
        raise ValueError("transition runtime differs from the source checkpoint")
    hardware = _hardware(args.device)
    return args, runtime, hardware


def _probe(job_path: Path) -> int:
    """Run isolated disposable measurements; never save a model or accuracy checkpoint."""
    from research.conductance_gat.v5 import batch_calibration, train
    from scripts.calibrate_training_resources import _measure

    job = _read(job_path)
    directory = Path(job["resource_directory"]).resolve()
    if job_path.resolve().parent != directory:
        raise ValueError("probe job must belong to its explicitly owned resource directory")
    args, runtime, hardware = _probe_runtime(job)
    import torch

    free, _ = torch.cuda.mem_get_info(torch.device(args.device))
    if free < float(job.get("minimum_free_gb", 8.0)) * 1024**3:
        raise ValueError(
            "current free GPU memory is below the preserved source preflight requirement"
        )
    payload, protocol = batch_calibration.load_calibration_payload(args)
    expected_protocol = job.get("source_dataset_protocol")
    if expected_protocol is not None and protocol != expected_protocol:
        raise ValueError("transition calibration dataset/protocol differs from the source")
    if args.dataset == "ppi":
        maximum, axis = len(payload["splits"]["train"]), "graphs"
    elif args.sampling != "full":
        maximum, axis = int(payload["splits"]["train"].count_nonzero()), "sampled_seed_nodes"
    else:
        maximum, axis = 1, "full_graph"
    baseline = (
        1
        if axis == "full_graph"
        else args.sample_seed_batch_size
        if axis == "sampled_seed_nodes"
        else args.batch_size
    )
    natural_maximum = max(maximum, baseline)
    preserve = job["action"] == "resume_incomplete_fixed"
    worker_options = (
        [args.workers]
        if preserve
        else [
            value
            for value in worker_candidates(
                args.workers, allocated_cpu_count(), applicable=axis == "graphs"
            )
            if args.workers <= value <= max(2, 2 * args.workers)
        ]
    )
    if not worker_options:
        raise ValueError("no worker candidate preserves the old worker allocation")
    identity = {
        "job_contract_sha256": digest(job["command"]),
        "runtime_versions": runtime,
        "hardware": hardware,
        "source_sha256": source_snapshot(),
        "dataset_protocol_sha256": train._canonical_sha256(protocol),
        "cache_sha256": protocol["data_sha256"],
        "source_checkpoint_sha256": job["source_checkpoint_sha256"],
        "transition_mode": TRANSITION_MODES.get(job["action"]),
        "source_epoch": job["source_epoch"],
        "baseline_execution": job["baseline_execution"],
        "worker_candidates": worker_options,
        "natural_training_split_size": maximum,
        "batch_axis": axis,
    }
    progress_path = directory / "resource-progress.json"
    if progress_path.exists():
        progress = _read(progress_path)
        if progress.get("identity") != identity:
            raise ValueError("partial resource measurements have a different transition identity")
    else:
        progress = {"identity": identity, "candidates": [], "status": "measuring"}

    def persist():
        atomic_write_json(progress_path, progress)

    persist()
    size, plateau, best_score = baseline, 0, None
    while True:
        size_scores = []
        for workers in worker_options:
            candidate = next(
                (
                    item
                    for item in progress["candidates"]
                    if (item["batch_size"], item["workers"]) == (size, workers)
                ),
                None,
            )
            if candidate is None:
                print(
                    f"[transition resource probe] {job['job_id']} batch={size} workers={workers}",
                    flush=True,
                )
                report = _measure(
                    {
                        "track": "conductance",
                        "condition": job["condition"],
                        "model_seed": job["model_seed"],
                    },
                    payload,
                    args,
                    batch_size=size,
                    workers=workers,
                )
                # OOM reports omit these fields upstream; record the actual attempted
                # candidate, not a fabricated success or resource measurement.
                report.update(batch_size=size, workers=workers)
                candidate = {
                    "batch_size": size,
                    "workers": workers,
                    "status": completed_candidate_status([report]),
                    "measurements": [report],
                }
                progress["candidates"].append(candidate)
                persist()
            score = candidate_score(candidate)
            if score is not None:
                size_scores.append(score)
        if not size_scores or preserve or axis == "full_graph" or size >= natural_maximum:
            break
        score = max(size_scores)
        plateau = plateau + 1 if best_score is not None and score <= best_score * 1.05 else 0
        best_score = score if best_score is None else max(best_score, score)
        if plateau >= 2:
            break
        size = min(2 * size, natural_maximum)
    selected = choose_candidate(progress["candidates"], baseline)
    execution = dict(job["baseline_execution"])
    if axis != "full_graph":
        execution["sample_seed_batch_size" if axis == "sampled_seed_nodes" else "batch_size"] = (
            selected["batch_size"]
        )
    execution["workers"] = selected["workers"]
    if preserve and execution != job["baseline_execution"]:
        raise ValueError("fixed continuation measurement must preserve its exact execution")
    if source_snapshot() != identity["source_sha256"]:
        raise ValueError("source changed during transition calibration")
    selected_command = list(job["command"])
    for field, value in execution.items():
        selected_command = _set_option(selected_command, "--" + field.replace("_", "-"), value)
    selected_args, _ = _training_arguments(selected_command)
    certificate = {
        "schema_version": 1,
        "kind": "v5_transition_resource_certificate",
        "status": "passed",
        "classification": "resource_calibration_not_final_training",
        **identity,
        "selected_execution": execution,
        "selected_configuration": train.configuration(selected_args),
        "candidates": progress["candidates"],
        "selected_candidate": {
            "batch_size": selected["batch_size"],
            "workers": selected["workers"],
        },
        "selection": {
            "policy": "preserve_fixed_execution" if preserve else "measured_transition_candidates",
            "no_downscale": True,
            "global_optimum_claimed": False,
            "paired_control_retrained": False,
            "optimization_recipe_change": execution != job["baseline_execution"],
        },
        "source_resource_plan_reference": job.get("source_resource_plan_reference"),
        "source_resource_plan_is_new_model_measurement": False,
    }
    certificate_path = directory / "resource-certificate.json"
    if certificate_path.exists() and _read(certificate_path) != certificate:
        raise ValueError("completed transition resource certificate is immutable")
    atomic_write_json(certificate_path, certificate)
    progress["status"] = "passed"
    persist()
    print(f"Measured transition resources: {execution}", flush=True)
    return 0


def _validate_certificate(job: dict[str, Any], path: Path) -> dict[str, Any]:
    from research.conductance_gat.v5 import batch_calibration, train

    certificate = _read(path)
    if any(
        certificate.get(name) != expected
        for name, expected in {
            "schema_version": 1,
            "kind": "v5_transition_resource_certificate",
            "status": "passed",
            "classification": "resource_calibration_not_final_training",
            "job_contract_sha256": digest(job["command"]),
            "source_checkpoint_sha256": job["source_checkpoint_sha256"],
            "transition_mode": TRANSITION_MODES.get(job["action"]),
            "source_epoch": job["source_epoch"],
            "baseline_execution": job["baseline_execution"],
            "source_sha256": source_snapshot(),
        }.items()
    ):
        raise ValueError("resource certificate does not match this transition")
    args, runtime, hardware = _probe_runtime(job)
    if certificate.get("runtime_versions") != runtime or certificate.get("hardware") != hardware:
        raise ValueError("resource certificate runtime or allocated GPU changed")
    axis = certificate["batch_axis"]
    # This also protects fresh pending jobs, whose trainer has no transition flags.
    payload, protocol = batch_calibration.load_calibration_payload(args)
    if args.dataset == "ppi":
        natural, expected_axis = len(payload["splits"]["train"]), "graphs"
    elif args.sampling != "full":
        natural, expected_axis = (
            int(payload["splits"]["train"].count_nonzero()),
            "sampled_seed_nodes",
        )
    else:
        natural, expected_axis = 1, "full_graph"
    if (
        axis != expected_axis
        or certificate.get("natural_training_split_size") != natural
        or certificate.get("dataset_protocol_sha256") != train._canonical_sha256(protocol)
        or certificate.get("cache_sha256") != protocol["data_sha256"]
        or (
            job.get("source_dataset_protocol") is not None
            and protocol != job["source_dataset_protocol"]
        )
    ):
        raise ValueError("resource certificate dataset or natural training split differs")
    del payload
    baseline = (
        1
        if axis == "full_graph"
        else job["baseline_execution"][
            "sample_seed_batch_size" if axis == "sampled_seed_nodes" else "batch_size"
        ]
    )
    workers = certificate.get("worker_candidates")
    preserve = job["action"] == "resume_incomplete_fixed"
    expected_workers = (
        [args.workers]
        if preserve
        else [
            value
            for value in worker_candidates(
                args.workers, allocated_cpu_count(), applicable=axis == "graphs"
            )
            if args.workers <= value <= max(2, 2 * args.workers)
        ]
    )
    if (
        workers != expected_workers
        or certificate.get("selection", {}).get("no_downscale") is not True
    ):
        raise ValueError("resource certificate worker allocation or no-downscale policy differs")
    seen = set()
    for candidate in certificate["candidates"]:
        key = candidate.get("batch_size"), candidate.get("workers")
        if key in seen or key[1] not in workers or len(candidate.get("measurements", [])) != 1:
            raise ValueError("resource certificate has duplicate or incomplete candidates")
        seen.add(key)
        for measurement in candidate.get("measurements", []):
            if (measurement.get("condition"), measurement.get("model_seed")) != (
                job["condition"],
                job["model_seed"],
            ):
                raise ValueError("resource measurement belongs to another condition or seed")
            if (measurement.get("batch_size"), measurement.get("workers")) != (
                candidate.get("batch_size"),
                candidate.get("workers"),
            ):
                raise ValueError("resource measurement execution differs from its candidate")
            if measurement.get("status") == "passed" and measurement.get(
                "total_memory_bytes"
            ) != hardware.get("total_memory_bytes"):
                raise ValueError("resource measurement capacity differs from its actual GPU")
    sizes = sorted({size for size, _ in seen})
    if (
        not sizes
        or sizes[0] != baseline
        or any((size, worker) not in seen for size in sizes for worker in workers)
    ):
        raise ValueError("resource certificate omitted baseline or worker candidates")
    if preserve and sizes != [baseline]:
        raise ValueError("fixed continuation must measure only its original execution")
    if not preserve and axis != "full_graph" and natural > baseline and len(sizes) < 2:
        raise ValueError("resource certificate requires multiple physical batch measurements")
    if any(
        right != min(2 * left, max(natural, baseline))
        for left, right in zip(sizes, sizes[1:], strict=False)
    ):
        raise ValueError("resource certificate skipped a required physical batch candidate")
    selected = choose_candidate(certificate["candidates"], baseline)
    if certificate.get("selected_candidate") != {
        "batch_size": selected["batch_size"],
        "workers": selected["workers"],
    }:
        raise ValueError("resource certificate did not select its measured safe candidate")
    execution = certificate.get("selected_execution")
    if not isinstance(execution, dict) or set(execution) != set(EXECUTION_FIELDS):
        raise ValueError("resource certificate execution fields are invalid")
    expected_execution = dict(job["baseline_execution"])
    if axis != "full_graph":
        expected_execution[
            "sample_seed_batch_size" if axis == "sampled_seed_nodes" else "batch_size"
        ] = selected["batch_size"]
    expected_execution["workers"] = selected["workers"]
    if execution != expected_execution:
        raise ValueError("resource certificate execution does not match its measured selection")
    if any(execution[name] < job["baseline_execution"][name] for name in EXECUTION_FIELDS):
        raise ValueError("transition resource certificate must not downscale execution")
    if job["action"] == "resume_incomplete_fixed" and execution != job["baseline_execution"]:
        raise ValueError("fixed continuation cannot change execution")
    selected_command = list(job["command"])
    for field, value in execution.items():
        selected_command = _set_option(selected_command, "--" + field.replace("_", "-"), value)
    selected_args, _ = _training_arguments(selected_command)
    if certificate.get("selected_configuration") != train.configuration(selected_args):
        raise ValueError("resource certificate measured a different V5 model or training recipe")
    return certificate


def _owned(path: Path, root: Path) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"transition output must not traverse symlinks: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"transition artifact escapes its destination: {path}")


def _run_logged(command: list[str], log: Path) -> int:
    from scripts.run_conductance_factorial import run_logged

    if log.is_symlink() or any(parent.is_symlink() for parent in log.parents):
        raise ValueError("transition log must not traverse symlinks")
    environment = os.environ.copy()
    environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    return run_logged(command, log, environment)


def _completed(job: dict[str, Any]) -> dict[str, Any]:
    from research.conductance_gat.v5.transition_report import validate_transition_child

    return validate_transition_child(job)


def _write_report(manifest: dict[str, Any], output: Path) -> None:
    from research.conductance_gat.v5.transition_report import build_transition_report

    _owned(output / "summary.json", output)
    atomic_write_json(output / "summary.json", build_transition_report(manifest))


def _resume_or_extend_budget(manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    """Only explicitly budget previously unstarted pending jobs; never rebase trained jobs."""
    old = manifest.get("plan")
    if (
        manifest.get("suite") != SUITE
        or not isinstance(old, dict)
        or manifest.get("plan_sha256") != digest(old)
    ):
        raise ValueError("existing transition destination has an invalid immutable plan")
    stored_jobs = manifest.get("jobs")
    if not isinstance(stored_jobs, list) or len(stored_jobs) != len(old["jobs"]):
        raise ValueError("existing transition job matrix is incomplete")
    for stored, original in zip(stored_jobs, old["jobs"], strict=True):
        if any(stored.get(key) != value for key, value in original.items()):
            raise ValueError("existing transition job contract changed")
        allowed_statuses = (
            {"historical_reference"}
            if original["action"] == "reuse_completed_fixed"
            else {"pending_extra_budget"}
            if original["action"] == "preserve_legacy_dynamic"
            else {"pending", "running", "failed", "passed"}
        )
        if stored.get("status") not in allowed_statuses:
            raise ValueError("existing transition job status contradicts its action")
    if old == plan:
        return
    ignored = {"extra_epochs_by_job", "jobs"}
    if {key: value for key, value in old.items() if key not in ignored} != {
        key: value for key, value in plan.items() if key not in ignored
    }:
        raise ValueError("existing transition destination has a different immutable plan")
    if len(plan["jobs"]) != len(old["jobs"]):
        raise ValueError("budget revision cannot add or remove jobs")
    old_budgets, new_budgets = (
        old.get("extra_epochs_by_job", {}),
        plan.get("extra_epochs_by_job", {}),
    )
    if any(new_budgets.get(key) != value for key, value in old_budgets.items()):
        raise ValueError("budget revision cannot change or remove prior explicit budgets")
    changed = []
    for stored, original, revised in zip(stored_jobs, old["jobs"], plan["jobs"], strict=True):
        if original == revised:
            continue
        job_id = original["job_id"]
        extra = new_budgets.get(job_id)
        if (
            stored["status"] != "pending_extra_budget"
            or original["action"] != "preserve_legacy_dynamic"
            or revised["action"] != "transition_dynamic"
            or isinstance(extra, bool)
            or not isinstance(extra, int)
            or extra <= 0
            or revised["source_epochs_requested"] + extra <= revised["source_epoch"]
        ):
            raise ValueError("extra budget may only activate an unstarted pending-extra-budget job")
        expected = copy.deepcopy(original)
        total = original["source_epochs_requested"] + extra
        command = _set_option(original["command"], "--epochs", total)
        command += [
            "--transition-from-checkpoint",
            original["source_checkpoint"],
            "--transition-source-sha256",
            original["source_checkpoint_sha256"],
            "--transition-mode",
            "replace_c",
            "--transition-extra-epochs",
            str(extra),
        ]
        expected.update(
            action="transition_dynamic",
            target_total_epochs=total,
            extra_epochs=extra,
            remaining_epochs=total - original["source_epoch"],
            requires_extra_epoch_budget=False,
            command=command,
        )
        if revised != expected:
            raise ValueError(
                "budget revision changed more than the pending job's explicit epoch budget"
            )
        target = Path(original["output_dir"])
        _owned(target, Path(plan["output_dir"]))
        if target.exists() and (any(target.rglob("*.pt")) or (target / "metrics.json").exists()):
            raise ValueError("budget revision target already has checkpoint or metric artifacts")
        changed.append(job_id)
    if not changed or set(new_budgets) - set(old_budgets) != set(changed):
        raise ValueError("budget revision must name exactly the newly activated pending jobs")
    manifest.setdefault("plan_revisions", []).append(
        {
            "old_plan_sha256": manifest["plan_sha256"],
            "new_plan_sha256": digest(plan),
            "changed_job_ids": changed,
            "previous_plan": old,
            "reason": "explicit per-job additional epoch budget; all trained jobs unchanged",
            "at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
    )
    manifest["plan"], manifest["plan_sha256"] = copy.deepcopy(plan), digest(plan)
    for stored, revised in zip(stored_jobs, plan["jobs"], strict=True):
        if stored["job_id"] in changed:
            stored.update(copy.deepcopy(revised), status="pending")


def execute(plan: dict[str, Any], *, confirm_source_stopped: bool) -> int:
    if not confirm_source_stopped:
        raise ValueError(
            "stop the original work first, then pass --confirm-source-stopped; "
            "no process is stopped here"
        )
    active = _active_source_processes(plan)
    if active:
        raise ValueError(f"original V5 training is still active; no signals sent: {active}")
    _verify_sources(plan)
    output = Path(plan["output_dir"])
    manifest_path = output / "manifest.json"
    contract_sha = digest(plan)
    if output.exists():
        manifest = _read(manifest_path)
        _resume_or_extend_budget(manifest, plan)
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "suite": SUITE,
            "status": "running",
            "plan": plan,
            "plan_sha256": contract_sha,
            "jobs": copy.deepcopy(plan["jobs"]),
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        for job in manifest["jobs"]:
            job["status"] = (
                "historical_reference"
                if job["action"] == "reuse_completed_fixed"
                else "pending_extra_budget"
                if job["requires_extra_epoch_budget"]
                else "pending"
            )
    _owned(output / "logs", output)
    (output / "logs").mkdir(exist_ok=True)
    try:
        # Recheck every claimed completed target before any new probe or training.
        for job in manifest["jobs"]:
            if job["status"] == "passed" and _completed(job) != job.get("result"):
                raise ValueError("completed transition child changed; refusing silent retraining")
        atomic_write_json(manifest_path, manifest)
        for index, job in enumerate(manifest["jobs"], 1):
            _verify_sources(plan)
            if job["status"] in {"passed", "historical_reference", "pending_extra_budget"}:
                print(
                    f"[{index}/{len(manifest['jobs'])}] {job['status']}: {job['job_id']}",
                    flush=True,
                )
                continue
            directory = Path(job["resource_directory"])
            _owned(directory, output)
            _owned(Path(job["output_dir"]), output)
            directory.mkdir(parents=True, exist_ok=True)
            probe_job_path = directory / "job.json"
            source_job = next(item for item in plan["jobs"] if item["job_id"] == job["job_id"])
            if probe_job_path.exists() and _read(probe_job_path) != source_job:
                raise ValueError("existing probe request differs; evidence preserved")
            if not probe_job_path.exists():
                atomic_write_json(probe_job_path, source_job)
            certificate_path = directory / "resource-certificate.json"
            if not certificate_path.exists():
                code = _run_logged(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "--probe-job",
                        str(probe_job_path),
                    ],
                    output / "logs" / f"probe-{index}.log",
                )
                if code:
                    raise RuntimeError(
                        f"transition resource probe failed with code {code}: {job['job_id']}"
                    )
            certificate = _validate_certificate(source_job, certificate_path)
            command = list(job["command"])
            for field, value in certificate["selected_execution"].items():
                command = _set_option(command, "--" + field.replace("_", "-"), value)
            if job["action"] in TRANSITION_MODES:
                command += [
                    "--transition-resource-certificate",
                    str(certificate_path),
                    "--transition-resource-sha256",
                    _sha(certificate_path),
                ]
            command += ["--resume"]
            job.update(
                resolved_command=command,
                resource_certificate=str(certificate_path),
                resource_certificate_sha256=_sha(certificate_path),
                selected_execution=certificate["selected_execution"],
                status="running",
            )
            atomic_write_json(manifest_path, manifest)
            _verify_sources(plan)
            print(f"[{index}/{len(manifest['jobs'])}] {job['action']}: {job['job_id']}", flush=True)
            print(shlex.join(command), flush=True)
            code = _run_logged(command, output / "logs" / f"train-{index}.log")
            _verify_sources(plan)
            if code:
                job.update(status="failed", returncode=code)
                raise RuntimeError(f"transition child failed with code {code}: {job['job_id']}")
            result = _completed(job)
            job.update(status="passed", returncode=0, result=result)
            atomic_write_json(manifest_path, manifest)
        waiting = any(job["status"] == "pending_extra_budget" for job in manifest["jobs"])
        manifest.update(
            status="pending_extra_budget" if waiting else "passed",
            finished_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        )
        atomic_write_json(manifest_path, manifest)
        _write_report(manifest, output)
        if waiting:
            print(
                "Remaining eligible V5 work finished; some legacy dynamic results still "
                "need explicit extra epoch budget.",
                flush=True,
            )
        return 3 if waiting else 0
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        try:
            atomic_write_json(manifest_path, manifest)
        except OSError as reporting_error:
            print(f"Could not save transition failure status: {reporting_error}", file=sys.stderr)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.probe_job is not None:
            return _probe(args.probe_job)
        plan = build_plan(args)
        if args.plan_only:
            print(
                json.dumps(
                    {
                        "suite": SUITE,
                        "read_only": True,
                        "gpu_work_launched": False,
                        "source_manifest": plan["source_manifest"],
                        "output_dir": plan["output_dir"],
                        "plan_sha256": digest(plan),
                        "classification": plan["classification"],
                        "jobs": [
                            {
                                name: job[name]
                                for name in (
                                    "job_id",
                                    "action",
                                    "source_epoch",
                                    "source_checkpoint_sha256",
                                    "target_total_epochs",
                                    "extra_epochs",
                                    "remaining_epochs",
                                    "requires_extra_epoch_budget",
                                    "baseline_execution",
                                    "command",
                                )
                            }
                            for job in plan["jobs"]
                        ],
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        _print_plan(plan)
        return execute(plan, confirm_source_stopped=args.confirm_source_stopped)
    except KeyboardInterrupt:
        print(
            "Transition interrupted; source and destination checkpoints are preserved.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"V5 transition refused/failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
