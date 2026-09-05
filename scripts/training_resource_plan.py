"""Immutable, measured resources shared by paired research conditions.

This is a calibration certificate, not evidence of final training or accuracy.
No value below the requested physical batch is selected automatically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import stat
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
IGNORED_COMMAND_OPTIONS = {
    "--output-dir",
    "--batch-size",
    "--workers",
    "--sample-seed-batch-size",
    "--resource-plan",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_snapshot() -> dict[str, str]:
    paths = [ROOT / "AGENTS.md", ROOT / "pyproject.toml"]
    for directory in ("research", "src/chartgat", "scripts"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend((ROOT / "research").rglob("*.yaml"))
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
        if path.is_file()
    }


def command_identity(command: list[str]) -> str:
    """Bind the scientific recipe and measured GPU assignment, not output/resource values."""
    if "-m" not in command:
        raise ValueError("calibration requires a module training command")
    remaining = command[command.index("-m") + 1 :]
    canonical: list[str] = []
    index = 0
    while index < len(remaining):
        token = remaining[index]
        if token in IGNORED_COMMAND_OPTIONS:
            if index + 1 >= len(remaining):
                raise ValueError(f"missing command value for {token}")
            index += 2
        elif token == "--resume":
            index += 1
        else:
            canonical.append(token)
            index += 1
    return digest(canonical)


def _positive(value: Any, label: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{label} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _number(value: Any, label: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{label} is invalid")
    return float(value)


def worker_candidates(requested: int, allocated_cpus: int, *, applicable: bool) -> list[int]:
    """Test real loader parallelism; GPU-resident samplers have no worker axis."""
    _positive(allocated_cpus, "allocated CPUs")
    _positive(requested, "requested workers", zero=True)
    if not applicable:
        return [0]
    # Include the requested policy and all distinct powers of two allowed by affinity.
    values = {requested}
    count = 2
    while count <= allocated_cpus:
        values.add(count)
        count *= 2
    if allocated_cpus < 2:
        values.add(allocated_cpus)
    return sorted(values)


def allocated_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)


def measurement_is_safe(report: dict[str, Any], *, memory_margin_fraction: float = 0.10) -> bool:
    """Use peak reserve including optimizer state, not a screenshot or model-only size."""
    if not isinstance(report, dict):
        raise ValueError("measurement must be an object")
    status = report.get("status")
    if status == "oom":
        if not isinstance(report.get("error"), str) or not report["error"].strip():
            raise ValueError("OOM measurement must preserve the actual error")
        return False
    if status != "passed":
        raise ValueError("measurement is not a completed pass or explicit OOM; errors are not OOM")
    throughput = _number(report.get("samples_per_second"), "measured throughput")
    elapsed = _number(report.get("elapsed_seconds"), "measured elapsed seconds")
    units = _positive(report.get("processed_units"), "real processed units")
    if not math.isclose(throughput, units / elapsed, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("measured throughput does not match real units and elapsed time")
    steps = _positive(report.get("optimizer_steps"), "measured optimizer steps")
    requested_steps = _positive(
        report.get("measurement_steps_requested"), "requested measurement steps"
    )
    _positive(report.get("warmup_steps_requested"), "requested warmup steps")
    requested_seconds = _number(
        report.get("minimum_measure_seconds_requested"), "requested measurement duration"
    )
    measured_steps = report.get("measurement_steps", steps)
    if _positive(measured_steps, "measured steps") < requested_steps or elapsed < requested_seconds:
        raise ValueError("candidate did not complete its requested steady-state measurement window")
    if "measurement_steps" in report and steps < _positive(
        report["measurement_steps"], "measurement steps"
    ):
        raise ValueError("optimizer steps do not cover the measured training steps")
    state = _positive(report.get("optimizer_state_bytes"), "optimizer state bytes")
    peak = _positive(report.get("peak_reserved_bytes"), "peak CUDA reserve")
    total = _positive(report.get("total_memory_bytes"), "visible memory")
    free = _positive(report.get("free_bytes_before"), "free memory before candidate")
    allocated = _positive(report.get("peak_allocated_bytes"), "peak CUDA allocation")
    if (
        free > total
        or allocated > peak
        or peak > total
        or state > allocated
        or isinstance(memory_margin_fraction, bool)
        or not isinstance(memory_margin_fraction, (float, int))
        or not math.isfinite(memory_margin_fraction)
        or not 0 < memory_margin_fraction < 1
    ):
        raise ValueError("invalid memory capacity or safety margin")
    if not isinstance(report.get("unit"), str) or not report["unit"].strip():
        raise ValueError("measured throughput unit is missing")
    margin = max(2 * 1024**3, math.ceil(total * memory_margin_fraction))
    return peak + margin <= free


def completed_candidate_status(reports: list[dict[str, Any]]) -> str:
    """Never turn a generic error, partial result or empty probe into an OOM."""
    if not isinstance(reports, list) or not reports:
        raise ValueError("completed candidate has no measurements")
    statuses = []
    for report in reports:
        measurement_is_safe(report)
        statuses.append(report["status"])
    return "oom" if "oom" in statuses else "passed"


def candidate_score(candidate: dict[str, Any]) -> float | None:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    _positive(candidate.get("batch_size"), "candidate batch")
    _positive(candidate.get("workers"), "candidate workers", zero=True)
    reports = candidate.get("measurements", [])
    status = completed_candidate_status(reports)
    if candidate.get("status") != status:
        raise ValueError("candidate status contradicts its completed measurements")
    if status == "oom":
        return None
    if not all(measurement_is_safe(item) for item in reports):
        return None
    # Same dataset and supervised-unit definition within a group; optimize its slowest arm.
    units = {item["unit"] for item in reports}
    if len(units) != 1:
        raise ValueError("paired calibration reports use different throughput units")
    return min(float(item["samples_per_second"]) for item in reports)


def choose_candidate(candidates: list[dict[str, Any]], baseline: int) -> dict[str, Any]:
    _positive(baseline, "baseline batch")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate selection requires actual measurements")
    eligible: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        if _positive(candidate.get("batch_size"), "candidate batch") < baseline:
            raise ValueError("calibration cannot shrink the requested batch")
        score = candidate_score(candidate)
        if score is not None:
            eligible.append((score, candidate))
    if not eligible:
        raise RuntimeError(
            "no measured candidate has safe optimizer-inclusive memory headroom; "
            "no downscale applied"
        )
    # Prefer less worker/memory overhead only when measured throughput is exactly tied.
    return max(eligible, key=lambda item: (item[0], -item[1]["workers"], -item[1]["batch_size"]))[1]


def _validate_entry(
    entry: dict[str, Any], seeds: list[int], *, allocated_cpus: int | None = None
) -> None:
    if not isinstance(entry, dict) or entry.get("status") != "passed":
        raise ValueError("resource plan entry is not complete")
    if entry.get("track") not in {"conductance", "cycle"}:
        raise ValueError("resource plan supports only Conductance V5 and Cycle V2")
    contracts = entry.get("job_contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("resource plan is missing exact job contracts")
    expected_conditions = (
        {"fixed_c", "shared_dynamic_c"} if entry["track"] == "conductance" else {"se", "pe"}
    )
    for item in contracts:
        if not isinstance(item, dict):
            raise ValueError("job contract must be an object")
        _positive(item.get("model_seed"), "model seed", zero=True)
        if not isinstance(item.get("condition"), str) or not _is_sha256(item.get("argv_sha256")):
            raise ValueError("job contract has no valid condition/command digest")
    identities = {(item.get("condition"), item.get("model_seed")) for item in contracts}
    if identities != {
        (condition, seed) for condition in expected_conditions for seed in seeds
    } or len(identities) != len(contracts):
        raise ValueError("every paired condition and requested seed must be measured")
    baseline = _positive(entry.get("baseline_physical_batch_size"), "baseline batch")
    axis = entry.get("batch_axis")
    if axis not in {"graphs", "sampled_seed_nodes", "full_graph"}:
        raise ValueError("resource plan has no valid physical batch axis")
    if entry["track"] == "cycle" and axis != "graphs":
        raise ValueError("Cycle V2 uses physical graph batches")
    if entry["track"] == "conductance" and (entry.get("dataset") == "ppi") != (axis == "graphs"):
        raise ValueError("only inductive PPI has a Conductance graph-batch axis")
    split_size = _positive(entry.get("natural_training_split_size"), "full training split size")
    worker_options = entry.get("worker_candidates")
    if not isinstance(worker_options, list) or not worker_options:
        raise ValueError("resource plan has no measured worker candidates")
    for count in worker_options:
        _positive(count, "worker candidate", zero=True)
    if len(set(worker_options)) != len(worker_options):
        raise ValueError("worker candidates must be unique")
    if (
        axis == "graphs"
        and allocated_cpus is not None
        and allocated_cpus > 1
        and len(worker_options) < 2
    ):
        raise ValueError("graph loading needs multiple measured worker candidates")
    if axis != "graphs" and worker_options != [0]:
        raise ValueError("GPU-resident/full graph execution has no DataLoader worker axis")
    candidates = entry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("resource plan has no measured candidates")
    seen: set[tuple[int, int]] = set()
    scores_by_batch: dict[int, list[float | None]] = {}
    for candidate in candidates:
        score = candidate_score(candidate)
        key = candidate["batch_size"], candidate["workers"]
        if key in seen or key[1] not in worker_options:
            raise ValueError("duplicate candidate or unrequested worker setting")
        seen.add(key)
        scores_by_batch.setdefault(key[0], []).append(score)
        measured = set()
        for item in candidate["measurements"]:
            seed = _positive(item.get("model_seed"), "measurement model seed", zero=True)
            condition = item.get("condition")
            if not isinstance(condition, str):
                raise ValueError("measurement condition is missing")
            measured.add((condition, seed))
            if item["status"] == "passed":
                if (
                    _positive(item.get("batch_size"), "measured batch") != key[0]
                    or _positive(item.get("workers"), "measured workers", zero=True) != key[1]
                ):
                    raise ValueError("measurement batch/worker setting differs from its candidate")
        if measured != identities or len(candidate["measurements"]) != len(identities):
            raise ValueError("candidate is missing paired measurements")
    batches = sorted(scores_by_batch)
    if batches[0] != baseline:
        raise ValueError("requested baseline batch was not measured")
    if seen != {(batch, count) for batch in batches for count in worker_options}:
        raise ValueError("a measured batch is missing a worker candidate")
    if axis != "full_graph" and baseline < split_size and len(batches) < 2:
        raise ValueError("batch selection requires at least two distinct measured physical batches")
    best = choose_candidate(candidates, baseline)
    selected = entry.get("selected", {})
    if not isinstance(selected, dict):
        raise ValueError("selected resources must be an object")
    expected_fields = {"batch_size", "workers"}
    if entry["track"] == "conductance":
        expected_fields.add("sample_seed_batch_size")
    if set(selected) != expected_fields:
        raise ValueError("selected resource fields are incomplete or unknown")
    for field, value in selected.items():
        _positive(value, f"selected {field}", zero=field == "workers")
    if axis in {"sampled_seed_nodes", "full_graph"} and selected["batch_size"] != 1:
        raise ValueError("a transductive graph cannot be duplicated to fill a graph batch")
    physical_key = (
        "sample_seed_batch_size"
        if entry.get("batch_axis") == "sampled_seed_nodes"
        else "batch_size"
    )
    if (
        selected.get(physical_key) != best["batch_size"]
        or selected.get("workers") != best["workers"]
    ):
        raise ValueError("selected resources do not match the best safe measured candidate")
    reason = entry.get("stop_reason")
    last_safe = [score for score in scores_by_batch[batches[-1]] if score is not None]
    if axis == "full_graph":
        if (
            baseline != 1
            or batches != [1]
            or split_size != 1
            or reason != "full_graph_no_batch_axis"
        ):
            raise ValueError("full graph no-batch-axis certificate is inconsistent")
    elif reason == "memory_headroom_boundary":
        if last_safe:
            raise ValueError("claimed memory boundary still has a safe measured candidate")
    elif reason == "complete_training_split_boundary":
        if batches[-1] != max(split_size, baseline) or not last_safe:
            raise ValueError("claimed natural boundary did not measure the full training split")
    elif reason == "measured_throughput_plateau":
        plateau, maximum = 0, None
        for batch in batches:
            safe = [score for score in scores_by_batch[batch] if score is not None]
            if not safe:
                raise ValueError("throughput plateau cannot hide a failed capacity boundary")
            score = max(safe)
            plateau = plateau + 1 if maximum is not None and score <= maximum * 1.05 else 0
            maximum = score if maximum is None else max(maximum, score)
        if plateau < 2:
            raise ValueError("claimed throughput plateau lacks two measured non-improvements")
    else:
        raise ValueError("batch search has no measured or natural termination boundary")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_resource_plan(
    plan: dict[str, Any],
    *,
    hardware_profile: str,
    profiles: list[str],
    model_seeds: list[int],
    check_sources: bool = True,
) -> None:
    """Validate an in-memory draft before publishing a completed certificate."""
    if (
        not isinstance(plan, dict)
        or isinstance(plan.get("schema_version"), bool)
        or not isinstance(plan.get("schema_version"), int)
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("kind") != "measured_training_resource_plan"
        or plan.get("status") != "passed"
    ):
        raise ValueError("resource plan is not a completed measured certificate")
    if (
        plan.get("classification") != "resource_calibration_not_final_training"
        or plan.get("final_training_started") is not False
    ):
        raise ValueError("calibration must be separate from final training")
    if not _is_sha256(plan.get("request_sha256")):
        raise ValueError("resource plan has no exact calibration request digest")
    seeds = plan.get("model_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("resource plan requires model seeds")
    for seed in seeds:
        _positive(seed, "model seed", zero=True)
    if len(set(seeds)) != len(seeds):
        raise ValueError("model seeds must be unique")
    if (
        plan.get("hardware_profile") != hardware_profile
        or plan.get("profiles") != list(profiles)
        or seeds != list(model_seeds)
    ):
        raise ValueError("resource plan profile/seed identity differs")
    if check_sources and plan.get("source_sha256") != source_snapshot():
        raise ValueError(
            "resource plan source identity differs; recalibration requires a separate new run"
        )
    hardware = plan.get("hardware")
    runtime = plan.get("runtime")
    if (
        not isinstance(hardware, dict)
        or not hardware
        or not isinstance(runtime, dict)
        or set(runtime) != {"python", "torch", "cuda"}
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise ValueError("resource plan requires real CUDA hardware and runtime fingerprints")
    for device, record in hardware.items():
        if (
            not isinstance(device, str)
            or not re.fullmatch(r"cuda(?::[0-9]+)?", device)
            or not isinstance(record, dict)
            or record.get("device") != device
        ):
            raise ValueError("invalid measured CUDA device identity")
        _positive(record.get("total_memory_bytes"), "measured GPU capacity")
        _positive(record.get("allocated_cpu_count"), "allocated CPU count")
        if not isinstance(record.get("name"), str) or not record["name"]:
            raise ValueError("measured GPU name is missing")
        capability = record.get("compute_capability")
        if not isinstance(capability, list) or len(capability) != 2:
            raise ValueError("measured GPU compute capability is missing")
        _positive(capability[0], "compute capability major")
        _positive(capability[1], "compute capability minor", zero=True)
        uuid = record.get("uuid")
        if uuid is None:
            if (
                not isinstance(record.get("uuid_unavailable_reason"), str)
                or not record["uuid_unavailable_reason"]
            ):
                raise ValueError("unavailable GPU UUID requires an explicit reason")
        elif (
            not isinstance(uuid, str)
            or not uuid
            or record.get("uuid_unavailable_reason") is not None
        ):
            raise ValueError("GPU UUID evidence is inconsistent")
        if record.get("cuda_visible_devices") is not None and not isinstance(
            record["cuda_visible_devices"], str
        ):
            raise ValueError("visible GPU allocation must be recorded without guessing")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("resource plan has no entries")
    keys: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("resource plan entry must be an object")
        key = (entry.get("track"), entry.get("profile"), entry.get("dataset"))
        if (
            any(not isinstance(value, str) or not value for value in key)
            or key in keys
            or key[1] not in profiles
        ):
            raise ValueError("duplicate, malformed or unrequested resource plan entry")
        keys.add(key)
        _validate_entry(
            entry,
            list(model_seeds),
            allocated_cpus=max(record["allocated_cpu_count"] for record in hardware.values()),
        )
        capacities = {record["total_memory_bytes"] for record in hardware.values()}
        for candidate in entry["candidates"]:
            for report in candidate["measurements"]:
                if report["status"] == "passed" and report["total_memory_bytes"] not in capacities:
                    raise ValueError("measured candidate memory differs from the certified GPU")


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("resource plan JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"resource plan JSON contains nonfinite constant {value}")


def load_resource_plan(
    path: Path | str, *, hardware_profile: str, profiles: list[str], model_seeds: list[int]
) -> dict[str, Any]:
    path = Path(path)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError("resource plan must be a regular file, not a symlink")
    raw = path.read_bytes()
    plan = json.loads(
        raw, object_pairs_hook=_json_object_pairs, parse_constant=_reject_json_constant
    )
    validate_resource_plan(
        plan, hardware_profile=hardware_profile, profiles=profiles, model_seeds=model_seeds
    )
    plan["_sha256"] = hashlib.sha256(raw).hexdigest()
    return plan


def validate_plan_runtime(
    plan: dict[str, Any] | None,
    device_names: list[str] | None = None,
) -> dict[str, Any] | None:
    """Verify the actual visible GPU allocation and runtime, never a CPU substitute."""
    if plan is None:
        return None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to reuse a measured resource plan; CPU fallback is forbidden"
        )
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    if runtime != plan.get("runtime"):
        raise ValueError("resource plan runtime changed; a separate recalibration is required")
    expected_hardware = plan.get("hardware")
    if not isinstance(expected_hardware, dict) or not expected_hardware:
        raise ValueError("resource plan has no measured hardware identity")
    names = list(expected_hardware) if device_names is None else list(device_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("runtime validation needs unique CUDA devices")
    hardware = {}
    for name in names:
        if name not in expected_hardware:
            raise ValueError("requested CUDA device was not measured")
        device = torch.device(name)
        if device.type != "cuda":
            raise ValueError("measured resource plan requires CUDA")
        prop = torch.cuda.get_device_properties(device)
        uuid = getattr(prop, "uuid", None)
        actual = {
            "device": str(device),
            "uuid": str(uuid) if uuid is not None else None,
            "uuid_unavailable_reason": None
            if uuid is not None
            else "Torch does not expose a GPU UUID",
            "name": prop.name,
            "total_memory_bytes": prop.total_memory,
            "compute_capability": [prop.major, prop.minor],
            "allocated_cpu_count": allocated_cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        if actual != expected_hardware[name]:
            raise ValueError(
                "resource plan GPU/CPU allocation differs; a separate recalibration is required"
            )
        hardware[name] = actual
    return {"runtime": runtime, "hardware": hardware}


def resource_plan_identity(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "sha256": plan["_sha256"],
        "selections": [
            {key: entry[key] for key in ("track", "profile", "dataset", "selected")}
            for entry in plan["entries"]
        ],
    }


def validate_plan_data(
    plan: dict[str, Any] | None, jobs: list[dict[str, Any]], *, track: str
) -> None:
    """Revalidate the actual official dataset cache before standalone child training."""
    if plan is None:
        return
    if track not in {"conductance", "cycle"}:
        raise ValueError("measured data validation supports only V5 and Cycle V2")
    from scripts.calibrate_training_resources import verify_plan_inputs

    version = "v5" if track == "conductance" else "v2"
    selected = []
    for job in jobs:
        if job.get("version") != version:
            continue
        selected.append(
            {
                "track": track,
                "profile": job["profile"],
                "dataset": job["dataset"] if track == "conductance" else job["datasets"][0],
                "condition": job["condition"] if track == "conductance" else job["encoding"],
                "model_seed": job["model_seed"],
                "command": list(job["command"]),
            }
        )
    if selected:
        verify_plan_inputs(plan, selected, allow_unrequested_entries=True)


def selected_resources(
    plan: dict[str, Any] | None, *, track: str, profile: str, dataset: str
) -> dict[str, Any] | None:
    if plan is None:
        return None
    for entry in plan["entries"]:
        if (entry["track"], entry["profile"], entry["dataset"]) == (track, profile, dataset):
            return dict(entry["selected"])
    raise ValueError(f"missing measured resource plan entry: {track}/{profile}/{dataset}")


def validate_job_plan(
    plan: dict[str, Any] | None,
    *,
    track: str,
    profile: str,
    dataset: str,
    condition: str,
    model_seed: int,
    command: list[str],
) -> None:
    if plan is None:
        return
    expected = {
        "condition": condition,
        "model_seed": model_seed,
        "argv_sha256": command_identity(command),
    }
    for entry in plan["entries"]:
        if (entry["track"], entry["profile"], entry["dataset"]) == (track, profile, dataset):
            if expected not in entry["job_contracts"]:
                raise ValueError("training command differs from the measured scientific recipe")
            for key, value in entry["selected"].items():
                option = "--" + key.replace("_", "-")
                if command.count(option) != 1:
                    raise ValueError("training command has missing or duplicate measured resources")
                index = command.index(option) + 1
                if index >= len(command) or command[index] != str(value):
                    raise ValueError(
                        "training command resources differ from the measured selection"
                    )
            return
    raise ValueError("training job has no measured resource plan")
