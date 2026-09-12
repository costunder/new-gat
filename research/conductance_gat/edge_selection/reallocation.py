"""Preserve resource selection while stress-checking an equivalent GPU allocation.

Fresh disposable probes cover the original selected candidate's measured worst
cases. They do not select a new optimum or change the final training recipe.
"""

from __future__ import annotations

import copy
import datetime as dt
import math

from research.conductance_gat.edge_selection import calibration
from scripts import training_resource_plan as resources

ALLOCATION_FIELDS = frozenset({"device", "uuid", "uuid_unavailable_reason", "cuda_visible_devices"})
SCOPE = "same-class allocation stress revalidation, not new optimum/all-arm fresh measurement"


def require_equivalent_allocation(original, runtime, actual, actual_runtime):
    """Only allocation identifiers may change; unknown hardware fields fail closed."""
    required = {"name", "total_memory_bytes", "compute_capability", "allocated_cpu_count"}
    if not isinstance(original, dict) or not required <= original.keys():
        raise ValueError("original resource hardware fingerprint is incomplete; preserved")
    if not isinstance(actual, dict) or not required <= actual.keys():
        raise ValueError("current resource hardware fingerprint is incomplete")
    if not isinstance(runtime, dict) or not {"python", "torch", "cuda"} <= runtime.keys():
        raise ValueError("original resource runtime fingerprint is incomplete; preserved")
    if not isinstance(actual_runtime, dict):
        raise ValueError("current resource runtime fingerprint is incomplete")
    differences = []
    for prefix, before, after, ignored in (
        ("hardware", original, actual, ALLOCATION_FIELDS),
        ("runtime", runtime, actual_runtime, frozenset()),
    ):
        for field in sorted((before.keys() | after.keys()) - ignored):
            if field not in before or field not in after or before[field] != after[field]:
                differences.append(
                    f"{prefix}.{field}: {before.get(field)!r} -> {after.get(field)!r}"
                )
    if differences:
        raise ValueError(
            "edge-selection allocation is not equivalent: "
            + "; ".join(differences)
            + ". Only GPU allocation identifiers may change. Restore the original GPU class, "
            "runtime and allocated CPU count, or use a separate explicitly calibrated run; "
            "the existing recipe/results remain preserved."
        )


def _original_digest(manifest):
    return resources.digest(
        {key: manifest[key] for key in ("hardware", "runtime", "calibration_entries")}
    )


def _attempt_digest(attempt):
    return resources.digest(
        {key: value for key, value in attempt.items() if key != "evidence_sha256"}
    )


def needs_revalidation(manifest, hardware, runtime, required_groups):
    """Validate the accepted evidence and decide whether pending work is covered."""
    require_equivalent_allocation(manifest["hardware"], manifest.get("runtime"), hardware, runtime)
    current = manifest.get("current_allocation")
    if current is None:
        if any(item.get("status") == "passed" for item in manifest.get("allocation_history", [])):
            raise ValueError("passed allocation history has no committed current allocation")
        changed = hardware != manifest["hardware"] or runtime != manifest["runtime"]
    else:
        history = manifest.get("allocation_history", [])
        index = current.get("history_index")
        if type(index) is not int or not 0 <= index < len(history):
            raise ValueError("current allocation has no retained revalidation evidence")
        accepted = history[index]
        passed = [i for i, item in enumerate(history) if item.get("status") == "passed"]
        if (
            not passed
            or index != passed[-1]
            or accepted.get("scope") != SCOPE
            or accepted.get("evidence_sha256") != _attempt_digest(accepted)
            or current.get("evidence_sha256") != accepted["evidence_sha256"]
            or accepted.get("original_calibration_sha256") != _original_digest(manifest)
        ):
            raise ValueError("accepted allocation evidence or original calibration changed")
        require_equivalent_allocation(
            manifest["hardware"], manifest["runtime"], accepted["hardware"], accepted["runtime"]
        )
        covered = {(group["profile"], group["dataset"]) for group in accepted["groups"]}
        changed = (
            hardware != accepted["hardware"]
            or runtime != accepted["runtime"]
            or not set(required_groups) <= covered
        )
    if changed and required_groups:
        if (
            manifest.get("calibration_status") != "passed"
            or manifest.get("resources_applied") is not True
            or not manifest.get("calibration_entries")
            or any(entry.get("status") != "passed" for entry in manifest["calibration_entries"])
        ):
            raise ValueError(
                "GPU allocation changed before the original common calibration completed; "
                "partial measurements cannot be mixed across allocations. Restore the original "
                "allocation or use a separate run; original evidence is preserved."
            )
        return True
    return False


def _representatives(entry, jobs):
    """Stable union of peak-reserve, peak-allocation and total-budget-cost maxima."""
    selected = entry["selected_candidate"]
    candidate_index = next(
        index
        for index, candidate in enumerate(entry["candidates"])
        if {key: candidate[key] for key in selected} == selected
    )
    reports = entry["candidates"][candidate_index]["measurements"]
    lookup = {(job["variant_id"], job["model_seed"]): job for job in jobs}
    ordered = sorted(
        range(len(reports)),
        key=lambda index: lookup[(reports[index]["condition"], reports[index]["model_seed"])][
            "job_id"
        ],
    )
    criteria = {}
    for name, score in (
        ("maximum_peak_reserved_bytes", lambda report: report["peak_reserved_bytes"]),
        ("maximum_peak_allocated_bytes", lambda report: report["peak_allocated_bytes"]),
        (
            "maximum_projected_full_budget_seconds",
            lambda report: calibration._full_budget_seconds(report, entry["selection_policy"]),
        ),
    ):
        index = max(ordered, key=lambda value: score(reports[value]))
        criteria.setdefault(index, []).append(name)
    return [
        {
            "job_id": lookup[(reports[index]["condition"], reports[index]["model_seed"])]["job_id"],
            "condition": reports[index]["condition"],
            "model_seed": reports[index]["model_seed"],
            "original_candidate_index": candidate_index,
            "original_measurement_index": index,
            "original_measurement_sha256": resources.digest(reports[index]),
            "selection_criteria": criteria[index],
        }
        for index in ordered
        if index in criteria
    ]


def _validate_measurement(report, job, child, entry, original, hardware):
    from research.conductance_gat.edge_selection import train

    if calibration._full_budget_seconds(report, entry["selection_policy"]) is None:
        raise RuntimeError(
            f"reallocated GPU cannot safely execute unchanged resources for {job['job_id']}; "
            "no batch, worker, model or data downscale was applied"
        )
    selected = entry["selected_candidate"]
    expected = calibration._probe_args(
        child, entry["batch_axis"], selected["batch_size"], selected["workers"]
    )
    required = {
        "condition": job["variant_id"],
        "model_seed": job["model_seed"],
        "configuration": train.configuration(expected),
        "batch_size": selected["batch_size"],
        "workers": selected["workers"],
        "total_memory_bytes": hardware["total_memory_bytes"],
        "parameter_update_verified": True,
        "calibration_not_final": True,
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "effective_batch_size": selected["batch_size"],
        "validation_completed": True,
        "required_auxiliary_path": child.negative_loss_weight > 0,
        "required_cycle_preparation": child.selection_mode == "forest_cycle",
        "auxiliary_path_measured": True,
        "cycle_preparation_measured": child.selection_mode == "forest_cycle",
        "model_parameter_count": original["model_parameter_count"],
        "initial_model_sha256": original["initial_model_sha256"],
    }
    for key, value in required.items():
        actual = report.get(key)
        if actual != value or (isinstance(value, bool) and actual is not value):
            raise ValueError(f"allocation measurement differs from unchanged recipe: {key}")
    for key, minimum in (
        ("complete_measurement_epochs", 1),
        ("complete_warmup_epochs", 1),
        ("warmup_optimizer_steps", 2),
        ("measurement_steps_requested", 5),
        ("warmup_steps_requested", 2),
        ("minimum_measure_seconds_requested", 3.0),
    ):
        value = report.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
            or (key != "minimum_measure_seconds_requested" and type(value) is not int)
        ):
            raise ValueError(f"allocation measurement lacks complete full-sized probe scope: {key}")
    measured_hardware = report.get("hardware", {})
    for report_key, actual_key in (
        ("device_name", "name"),
        ("total_memory_bytes", "total_memory_bytes"),
        ("compute_capability", "compute_capability"),
    ):
        if measured_hardware.get(report_key) != hardware[actual_key]:
            raise ValueError(f"allocation measurement hardware mismatch: {report_key}")


def revalidate_allocation(manifest, hardware, runtime, grouped_jobs, persist):
    """Append an attempt and atomically accept it only after all probes pass."""
    from research.conductance_gat.edge_selection import train

    if not grouped_jobs:
        raise ValueError(
            "allocation revalidation requires pending groups with real workload probes"
        )
    require_equivalent_allocation(manifest["hardware"], manifest["runtime"], hardware, runtime)
    entries = {
        (entry["profile"], entry["dataset"]): entry for entry in manifest["calibration_entries"]
    }
    groups = []
    for key, jobs in grouped_jobs.items():
        entry = entries[key]
        calibration.validate_entry(entry, jobs)
        groups.append(
            {
                "profile": key[0],
                "dataset": key[1],
                "selected_candidate": copy.deepcopy(entry["selected_candidate"]),
                "selected": copy.deepcopy(entry["selected"]),
                "representatives": _representatives(entry, jobs),
                "measurements": [],
            }
        )
    attempt = {
        "schema_version": 1,
        "status": "running",
        "scope": SCOPE,
        "hardware": copy.deepcopy(hardware),
        "runtime": copy.deepcopy(runtime),
        "original_calibration_sha256": _original_digest(manifest),
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "representative_tie_break": "lexicographically first job_id",
        "inherited_evidence": "original complete all-arm calibration at unchanged resources",
        "groups": groups,
    }
    history = manifest.setdefault("allocation_history", [])
    history.append(attempt)
    persist()
    try:
        for group in groups:
            key = group["profile"], group["dataset"]
            entry, jobs = entries[key], grouped_jobs[key]
            lookup = {job["job_id"]: job for job in jobs}
            loaded, identity, maximum, axis = train.load_calibration_payload(
                calibration.parse_job(jobs[0])
            )
            if (identity, maximum, axis) != (
                entry["input_identity"],
                entry["natural_training_split_size"],
                entry["batch_axis"],
            ):
                raise ValueError("allocation revalidation data/split/topology identity changed")
            for reference in group["representatives"]:
                job = lookup[reference["job_id"]]
                child = calibration.parse_job(job)
                selected = entry["selected_candidate"]
                print(
                    f"[edge allocation revalidation] {job['job_id']} "
                    f"physical={selected['batch_size']} workers={selected['workers']}; "
                    "disposable same-class stress probe, unchanged recipe",
                    flush=True,
                )
                report = calibration._measure(
                    job, loaded, child, selected["batch_size"], selected["workers"]
                )
                group["measurements"].append({"job_id": job["job_id"], "report": report})
                persist()
                original = entry["candidates"][reference["original_candidate_index"]][
                    "measurements"
                ][reference["original_measurement_index"]]
                _validate_measurement(report, job, child, entry, original, hardware)
            del loaded
        attempt.update(status="passed", finished_at_utc=dt.datetime.now(dt.UTC).isoformat())
        attempt["evidence_sha256"] = _attempt_digest(attempt)
        manifest["current_allocation"] = {
            "history_index": len(history) - 1,
            "evidence_sha256": attempt["evidence_sha256"],
        }
        persist()
    except (Exception, KeyboardInterrupt) as error:
        attempt.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
            finished_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        )
        # Publication failure must not leave an in-memory accepted failed attempt.
        if manifest.get("current_allocation", {}).get("history_index") == len(history) - 1:
            previous = [
                index for index, item in enumerate(history[:-1]) if item.get("status") == "passed"
            ]
            if previous:
                manifest["current_allocation"] = {
                    "history_index": previous[-1],
                    "evidence_sha256": history[previous[-1]]["evidence_sha256"],
                }
            else:
                manifest.pop("current_allocation", None)
        try:
            persist()
        except (Exception, KeyboardInterrupt) as reporting_error:
            error.add_note(f"allocation failure evidence could not be persisted: {reporting_error}")
        raise
