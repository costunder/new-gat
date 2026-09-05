#!/usr/bin/env python3
"""Measure real training resources before V5/Cycle V2 final research training.

The separate probe models execute real optimizer updates but never publish an
accuracy result or checkpoint. Existing training runs are neither opened nor changed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from chartgat.cache import atomic_write_json  # noqa: E402
from chartgat.resume_compat import require_source_compatibility  # noqa: E402
from scripts.training_resource_plan import (  # noqa: E402
    SCHEMA_VERSION,
    allocated_cpu_count,
    candidate_score,
    choose_candidate,
    command_identity,
    completed_candidate_status,
    digest,
    load_resource_plan,
    source_snapshot,
    validate_resource_plan,
    worker_candidates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _group_key(job: dict[str, Any]) -> str:
    return "/".join(job[key] for key in ("track", "profile", "dataset"))


def _training_args(job: dict[str, Any]):
    command = job["command"]
    module_index = command.index("-m") + 1
    if job["track"] == "conductance":
        if command[module_index] != "research.conductance_gat.v5.train":
            raise ValueError("resource calibration refuses non-V5 training commands")
        from research.conductance_gat.v5.train import build_parser, validate_args

        args = build_parser().parse_args(command[module_index + 1 :])
        validate_args(args)
    elif job["track"] == "cycle":
        if command[module_index] != "research.cycle_pe.v2.benchmark":
            raise ValueError("resource calibration refuses non-Cycle-V2 training commands")
        from research.cycle_pe.v2.benchmark import _validate, parser

        args = parser().parse_args(command[module_index + 1 :])
        _validate(args)
        args.dataset = job["dataset"]
    else:
        raise ValueError("only the requested V5 and Cycle V2 tracks are calibrated")
    return args


def _hardware(device_name: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for measured resource selection; CPU fallback is forbidden"
        )
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("resource calibration requires CUDA")
    prop = torch.cuda.get_device_properties(device)
    uuid = getattr(prop, "uuid", None)
    return {
        "device": str(device),
        "uuid": str(uuid) if uuid is not None else None,
        "uuid_unavailable_reason": None if uuid is not None else "Torch does not expose a GPU UUID",
        "name": prop.name,
        "total_memory_bytes": prop.total_memory,
        "compute_capability": [prop.major, prop.minor],
        "allocated_cpu_count": allocated_cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _load_group(job: dict[str, Any], args):
    if job["track"] == "conductance":
        from research.conductance_gat.v5.batch_calibration import load_calibration_payload

        payload, protocol = load_calibration_payload(args)
        if args.dataset == "ppi":
            maximum = len(payload["splits"]["train"])
            axis = "graphs"
        elif args.sampling != "full":
            maximum = int(payload["splits"]["train"].count_nonzero())
            axis = "sampled_seed_nodes"
        else:
            maximum, axis = 1, "full_graph"
        identity = {
            "dataset": args.dataset,
            "data_sha256": protocol["data_sha256"],
            "split_sha256": protocol["split_sha256"],
            "protocol": protocol,
        }
        return payload, identity, maximum, axis
    from research.cycle_pe.v2.calibration import load_calibration_graphs

    graphs, identity = load_calibration_graphs(args)
    return graphs, identity, len(graphs), "graphs"


def _measure(job, loaded, args, *, batch_size: int, workers: int) -> dict[str, Any]:
    import torch

    if job["track"] == "conductance":
        from research.conductance_gat.v5.batch_calibration import run_training_candidate
    else:
        from research.cycle_pe.v2.calibration import run_training_candidate
    torch.cuda.empty_cache()
    try:
        measured = run_training_candidate(
            loaded,
            copy.deepcopy(args),
            torch.device(args.device),
            physical_batch_size=batch_size,
            workers=workers,
            warmup_steps=2,
            measurement_steps=5,
            minimum_measure_seconds=3.0,
        )
    except torch.OutOfMemoryError as error:
        measured = {
            "status": "oom",
            "error": f"{type(error).__name__}: {error}",
            "resource_observability": getattr(error, "calibration_resource_observability", None),
        }
        # Our disposable probe's traceback must not retain CUDA tensors into the next probe.
        traceback.clear_frames(error.__traceback__)
    measured.update(condition=job["condition"], model_seed=job["model_seed"])
    gc.collect()
    torch.cuda.empty_cache()
    return measured


def _scientific_input_identity(identity: dict[str, Any], *, track: str) -> dict[str, Any]:
    """Exclude only Cycle's loader observation, never mutate persisted evidence.

    Cycle's immutable cache loader records the workers used on this particular
    read alongside its source/split/preparation identity. Calibration intentionally
    selects a potentially different worker policy for training. The original
    observation stays in the plan and protocol; it is not a dataset change.
    """
    if track == "cycle":
        return {key: value for key, value in identity.items() if key != "preparation_workers"}
    return identity


def _changed_identity_fields(previous: Any, current: Any, *, path: str) -> list[str]:
    """Report precise differing fields while leaving the full evidence untouched."""
    if isinstance(previous, dict) and isinstance(current, dict):
        changed = []
        for key in sorted(previous.keys() | current.keys()):
            field = f"{path}.{key}"
            if key not in previous or key not in current:
                changed.append(field)
            else:
                changed.extend(_changed_identity_fields(previous[key], current[key], path=field))
        return changed
    return [] if previous == current else [path]


def _verify_loaded_input(
    entry: dict[str, Any], identity: dict[str, Any], maximum: int, axis: str, *, context: str
) -> None:
    previous = entry["input_identity"]
    track = entry["track"]
    changed = _changed_identity_fields(
        _scientific_input_identity(previous, track=track),
        _scientific_input_identity(identity, track=track),
        path="input_identity",
    )
    for key, value in (("natural_training_split_size", maximum), ("batch_axis", axis)):
        if entry[key] != value:
            changed.append(key)
    if changed:
        raise ValueError(f"{context}; {_group_key(entry)}; changed fields: {', '.join(changed)}")
    if track == "cycle" and previous.get("preparation_workers") != identity.get(
        "preparation_workers"
    ):
        print(
            f"[calibration input verified] {_group_key(entry)}: unchanged official data; "
            f"preparation_workers {previous.get('preparation_workers')} -> "
            f"{identity.get('preparation_workers')} is execution metadata; "
            "original plan observation preserved",
            flush=True,
        )


def _calibrate_group(jobs: list[dict[str, Any]], entry: dict[str, Any], persist) -> None:
    primary = jobs[0]
    parsed = [_training_args(job) for job in jobs]
    loaded, identity, maximum, axis = _load_group(primary, parsed[0])
    if entry.get("input_identity") is not None:
        _verify_loaded_input(
            entry,
            identity,
            maximum,
            axis,
            context=(
                "verified dataset changed since partial calibration; previous evidence preserved"
            ),
        )
    baseline = (
        parsed[0].sample_seed_batch_size if axis == "sampled_seed_nodes" else parsed[0].batch_size
    )
    baseline = 1 if axis == "full_graph" else baseline
    if maximum < 1:
        raise ValueError("the official training split is empty")
    # A configured batch larger than the complete split is never silently reduced.
    natural_maximum = max(maximum, baseline)
    workers = worker_candidates(
        parsed[0].workers, allocated_cpu_count(), applicable=axis == "graphs"
    )
    # Explore neighbouring loader policies, rather than launching hundreds of workers at once.
    workers = [value for value in workers if value <= max(2, parsed[0].workers * 2)]
    entry.update(
        track=primary["track"],
        profile=primary["profile"],
        dataset=primary["dataset"],
        baseline_physical_batch_size=baseline,
        batch_axis=axis,
        natural_training_split_size=maximum,
        input_identity=(
            entry["input_identity"] if entry.get("input_identity") is not None else identity
        ),
        job_contracts=[
            {
                "condition": job["condition"],
                "model_seed": job["model_seed"],
                "argv_sha256": command_identity(job["command"]),
            }
            for job in jobs
        ],
        worker_candidates=workers,
        worker_search_scope=(
            "requested and neighbouring powers of two up to 2x requested, bounded by CPU affinity; "
            "finite measured search, not a global optimum"
        ),
    )
    entry.setdefault("candidates", [])
    current, plateau, best_score = baseline, 0, None
    while True:
        size_has_safe_candidate = False
        size_best = None
        for count in workers:
            candidate = next(
                (
                    item
                    for item in entry["candidates"]
                    if item["batch_size"] == current and item["workers"] == count
                ),
                None,
            )
            if candidate is None:
                candidate = {
                    "batch_size": current,
                    "workers": count,
                    "status": "running",
                    "measurements": [],
                }
                entry["candidates"].append(candidate)
            if candidate.get("status") == "running":
                for job, args in zip(jobs, parsed, strict=True):
                    existing = next(
                        (
                            item
                            for item in candidate["measurements"]
                            if (item["condition"], item["model_seed"])
                            == (job["condition"], job["model_seed"])
                        ),
                        None,
                    )
                    if existing is None:
                        print(
                            f"[calibration] {_group_key(job)}/{job['condition']} "
                            f"seed={job['model_seed']} batch={current} workers={count}",
                            flush=True,
                        )
                        report = _measure(job, loaded, args, batch_size=current, workers=count)
                        candidate["measurements"].append(report)
                        persist()
                candidate["status"] = completed_candidate_status(candidate["measurements"])
                persist()
            score = candidate_score(candidate)
            if score is not None:
                size_has_safe_candidate = True
                size_best = score if size_best is None else max(size_best, score)
        if not size_has_safe_candidate:
            entry["stop_reason"] = "memory_headroom_boundary"
            break
        if axis == "full_graph" or current >= natural_maximum:
            entry["stop_reason"] = (
                "full_graph_no_batch_axis"
                if axis == "full_graph"
                else "complete_training_split_boundary"
            )
            break
        if best_score is not None and size_best <= best_score * 1.05:
            plateau += 1
        else:
            plateau = 0
        best_score = size_best if best_score is None else max(best_score, size_best)
        if plateau >= 2:
            entry["stop_reason"] = "measured_throughput_plateau"
            break
        current = min(current * 2, natural_maximum)
    chosen = choose_candidate(entry["candidates"], baseline)
    selected = {"batch_size": parsed[0].batch_size, "workers": chosen["workers"]}
    if primary["track"] == "conductance":
        selected["sample_seed_batch_size"] = parsed[0].sample_seed_batch_size
    selected["sample_seed_batch_size" if axis == "sampled_seed_nodes" else "batch_size"] = chosen[
        "batch_size"
    ]
    entry.update(
        status="passed",
        selected=selected,
        selection={
            "algorithm": "highest minimum paired throughput among safe measured candidates",
            "memory_margin": "max(2 GiB, 10% visible capacity), including optimizer peak reserve",
            "minimum_requested_batch_preserved": True,
            "global_optimum_claimed": False,
            "paired_resources_identical": True,
            "optimization_recipe_change": (
                "larger physical batches change updates per epoch and trajectory; "
                "selected resources are immutable and shared by paired arms"
            ),
        },
    )
    persist()
    print(
        f"[calibration selected] {_group_key(primary)} {selected}; boundary={entry['stop_reason']}",
        flush=True,
    )


def verify_plan_inputs(
    plan: dict[str, Any], jobs: list[dict[str, Any]], *, allow_unrequested_entries: bool = False
) -> None:
    """Reverify requested floors/recipes and actual official cache hashes."""
    for entry in plan["entries"]:
        if entry.get("status") != "passed":
            continue
        matching = [item for item in jobs if _group_key(item) == _group_key(entry)]
        if not matching:
            if allow_unrequested_entries:
                continue
            raise ValueError("resource plan contains an unrequested dataset/profile")
        parsed = []
        for job in matching:
            contract = {
                "condition": job["condition"],
                "model_seed": job["model_seed"],
                "argv_sha256": command_identity(job["command"]),
            }
            if contract not in entry["job_contracts"]:
                raise ValueError("resource plan training recipe or measured device differs")
            args = _training_args(job)
            key = (
                "sample_seed_batch_size"
                if entry["batch_axis"] == "sampled_seed_nodes"
                else "batch_size"
            )
            requested = 1 if entry["batch_axis"] == "full_graph" else getattr(args, key)
            if entry["selected"][key] < requested:
                raise ValueError(
                    f"resource plan selected physical batch for {_group_key(job)} is below "
                    f"the current requested floor {requested}; recalibrate with a new run ID; "
                    "previous measurements preserved"
                )
            parsed.append(args)
        loaded, identity, maximum, axis = _load_group(matching[0], parsed[0])
        _verify_loaded_input(
            entry,
            identity,
            maximum,
            axis,
            context="resource plan verified dataset identity changed; no stale measurement reuse",
        )
        del loaded
        gc.collect()


def _run_locked(request_path: Path, output: Path) -> Path:
    import torch

    request = json.loads(request_path.read_bytes())
    expected_sources = source_snapshot()
    try:
        transition = require_source_compatibility(request.get("source_sha256"), expected_sources)
    except ValueError as error:
        raise ValueError("calibration request source identity differs") from error
    if transition is not None:
        print(
            f"[resume compatibility] {transition['patch_id']}; original request preserved",
            flush=True,
        )
    jobs = request.get("jobs", [])
    if not jobs:
        raise ValueError("calibration request contains no supported jobs")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    devices = {job["device"]: _hardware(job["device"]) for job in jobs}
    request_hash = digest(request)
    output.mkdir(parents=True, exist_ok=True)
    progress_path, plan_path = output / "progress.json", output / "resource-plan.json"
    if any(path.is_symlink() for path in (output, progress_path, plan_path)):
        raise ValueError("calibration artifacts must not be symlinks")
    if plan_path.exists():
        plan = load_resource_plan(
            plan_path,
            hardware_profile=request["hardware_profile"],
            profiles=request["profiles"],
            model_seeds=request["model_seeds"],
        )
        if (
            plan["request_sha256"] != request_hash
            or plan["hardware"] != devices
            or plan["runtime"] != runtime
        ):
            raise ValueError(
                "completed resource plan request/hardware/runtime differs; refusing reuse"
            )
        verify_plan_inputs(plan, jobs)
        return plan_path
    minimum_free = float(request.get("minimum_free_gb", 0.0))
    if request["hardware_profile"] == "a6000-48gb":
        minimum_free = max(minimum_free, 32.0)
    for device, hardware in devices.items():
        if request["hardware_profile"] == "a6000-48gb" and (
            hardware["total_memory_bytes"] < 40 * 1024**3 or hardware["compute_capability"][0] < 8
        ):
            raise RuntimeError(
                "A6000 profile requires >=40 GiB visible memory and capability 8.0; no downscale"
            )
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        if free < minimum_free * 1024**3:
            raise RuntimeError(
                f"{device} has {free / 1024**3:.2f} GiB free; "
                f"calibration requires {minimum_free:.2f} GiB. Existing processes are not changed"
            )
    if progress_path.exists():
        progress = json.loads(progress_path.read_bytes())
        if (
            progress.get("request_sha256") != request_hash
            or progress.get("hardware") != devices
            or progress.get("runtime") != runtime
        ):
            raise ValueError(
                "partial calibration identity differs; no previous evidence overwritten"
            )
        verify_plan_inputs(progress, jobs)
    else:
        progress = {
            "schema_version": SCHEMA_VERSION,
            "kind": "measured_training_resource_plan",
            "status": "calibrating",
            "classification": "resource_calibration_not_final_training",
            "request_sha256": request_hash,
            "source_sha256": expected_sources,
            "hardware_profile": request["hardware_profile"],
            "profiles": request["profiles"],
            "model_seeds": request["model_seeds"],
            "hardware": devices,
            "runtime": runtime,
            "entries": [],
            "final_training_started": False,
        }

    def persist():
        atomic_write_json(progress_path, progress)

    persist()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(_group_key(job), []).append(job)
    try:
        for key, paired in grouped.items():
            entry = next((item for item in progress["entries"] if _group_key(item) == key), None)
            if entry is None:
                entry = {name: paired[0][name] for name in ("track", "profile", "dataset")}
                progress["entries"].append(entry)
            if entry.get("status") != "passed":
                _calibrate_group(paired, entry, persist)
            gc.collect()
            torch.cuda.empty_cache()
        if source_snapshot() != expected_sources:
            raise ValueError("source changed during calibration")
        progress["status"] = "passed"
        progress.pop("error", None)
        validate_resource_plan(
            progress,
            hardware_profile=request["hardware_profile"],
            profiles=request["profiles"],
            model_seeds=request["model_seeds"],
        )
        persist()
        atomic_write_json(plan_path, progress)
    except BaseException as error:
        progress.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        persist()
        raise
    return plan_path


def run(request_path: Path, output: Path) -> Path:
    from scripts.calibration_lock import calibration_lock

    with calibration_lock(output) as validated_output:
        return _run_locked(request_path, validated_output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = run(args.request, args.output_dir)
    except (Exception, KeyboardInterrupt) as error:
        print(
            f"Resource calibration failed: {type(error).__name__}: {error}; "
            "no final training started and no batch downscale applied",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"Measured resource plan: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
