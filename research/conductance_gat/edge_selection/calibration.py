"""Measured common resources for the independent edge-selection experiment suite.

Only disposable calibration probes run here. No final-training model/data/budget
is reduced, and no older V5 calibration implementation is changed.
"""

from __future__ import annotations

import copy
import gc
import math
import traceback

from scripts import training_resource_plan as resources


def parse_job(job):
    from research.conductance_gat.edge_selection import train

    args = train.build_parser().parse_args(job["command"][job["command"].index("-m") + 2 :])
    train.validate_args(args)
    return args


def _contracts(jobs):
    return [
        {
            "condition": job["variant_id"],
            "model_seed": job["model_seed"],
            "argv_sha256": resources.command_identity(job["command"]),
        }
        for job in jobs
    ]


def _probe_args(args, axis, batch, workers):
    from research.conductance_gat.v5.batch_calibration import _candidate_args

    expected_axis = (
        "graphs"
        if args.dataset == "ppi"
        else ("full_graph" if args.sampling == "full" else "sampled_seed_nodes")
    )
    if axis != expected_axis:
        raise ValueError("calibration batch axis differs from the actual dataset/sampler")
    return _candidate_args(args, batch, workers)


def _full_budget_seconds(report, policy):
    safe = resources.measurement_is_safe(report)
    if report["status"] == "oom":
        return None
    if report.get("validation_completed") is not True:
        raise ValueError("calibration did not measure complete validation")
    for name in ("validation_seconds", "topology_preparation_seconds", "setup_seconds"):
        value = report.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"calibration lacks a valid measured {name}")
    if (
        report.get("required_auxiliary_path") is True
        and report.get("auxiliary_path_measured") is not True
    ):
        raise ValueError("negative auxiliary loss was not measured in calibration")
    if (
        report.get("required_cycle_preparation") is True
        and report.get("cycle_preparation_measured") is not True
    ):
        raise ValueError("cycle preparation was not measured in calibration")
    if not safe:
        return None
    cost = resources.projected_training_budget_cost(report, policy)
    return (
        cost["projected_training_seconds"]
        + cost["learning_budget"]["planned_epochs"] * report["validation_seconds"]
        + report["setup_seconds"]
    )


def _score(candidate, policy):
    costs = [_full_budget_seconds(report, policy) for report in candidate["measurements"]]
    if not costs or any(value is None for value in costs):
        return None
    return max(costs)


def _choose(candidates, policy):
    eligible = [(value, _score(value, policy)) for value in candidates]
    eligible = [(candidate, score) for candidate, score in eligible if score is not None]
    if not eligible:
        raise RuntimeError("no common safe physical batch fits all arms; no model/data downscale")
    return min(eligible, key=lambda item: (item[1], -item[0]["batch_size"], item[0]["workers"]))[0]


def _measure(job, loaded, args, batch, workers):
    import torch

    from research.conductance_gat.edge_selection import train

    gc.collect()
    torch.cuda.empty_cache()
    try:
        report = train.run_calibration_candidate(
            loaded,
            copy.deepcopy(args),
            torch.device(args.device),
            physical_batch_size=batch,
            workers=workers,
            warmup_steps=2,
            measurement_steps=5,
            minimum_measure_seconds=3.0,
        )
    except torch.OutOfMemoryError as error:
        report = {"status": "oom", "error": f"{type(error).__name__}: {error}"}
        traceback.clear_frames(error.__traceback__)
    report.update(
        condition=job["variant_id"],
        model_seed=job["model_seed"],
        required_auxiliary_path=args.negative_loss_weight > 0,
        required_cycle_preparation=args.selection_mode == "forest_cycle",
    )
    gc.collect()
    torch.cuda.empty_cache()
    return report


def validate_entry(entry, jobs):
    from research.conductance_gat.edge_selection import train

    if entry.get("status") != "passed" or entry.get("job_contracts") != _contracts(jobs):
        raise ValueError("edge-selection common resource entry or exact arm matrix is incomplete")
    parsed = [parse_job(job) for job in jobs]
    identities = {
        (job["variant_id"], job["model_seed"]): args for job, args in zip(jobs, parsed, strict=True)
    }
    baseline, axis = entry["baseline_physical_batch_size"], entry["batch_axis"]
    expected_floor = (
        parsed[0].sample_seed_batch_size if axis == "sampled_seed_nodes" else parsed[0].batch_size
    )
    context = parsed[0].sampling == "cluster_disjoint"
    requested_workers = parsed[0].sample_context_workers if context else parsed[0].workers
    expected_workers = resources.worker_candidates(
        requested_workers, resources.allocated_cpu_count(), applicable=axis == "graphs" or context
    )
    expected_workers = [
        value for value in expected_workers if value <= max(2, 2 * requested_workers)
    ]
    if baseline != expected_floor or entry.get("worker_candidates") != expected_workers:
        raise ValueError(
            "calibration physical floor or worker search differs from the declared recipe"
        )
    if context != (entry.get("worker_axis") == "sample_context_workers"):
        raise ValueError("calibration worker axis differs from the declared sampler")
    policy = resources.learning_budget_selection_policy(
        vars(parsed[0]),
        training_split_size=entry["natural_training_split_size"],
        batch_axis=axis,
    )
    if policy != entry.get("selection_policy"):
        raise ValueError("calibration learning-budget selection recipe changed")
    seen = set()
    for candidate in entry["candidates"]:
        pair = candidate["batch_size"], candidate["workers"]
        if pair in seen or pair[0] < baseline or pair[1] not in entry["worker_candidates"]:
            raise ValueError("duplicate or unrequested physical calibration candidate")
        seen.add(pair)
        reports = candidate["measurements"]
        if {(item["condition"], item["model_seed"]) for item in reports} != set(identities) or len(
            reports
        ) != len(identities):
            raise ValueError("common calibration candidate is missing a requested arm/seed")
        if candidate["status"] != resources.completed_candidate_status(reports):
            raise ValueError("calibration status differs from actual measured evidence")
        for report in reports:
            if report["status"] == "passed":
                expected = _probe_args(
                    identities[(report["condition"], report["model_seed"])], axis, *pair
                )
                if report.get("configuration") != train.configuration(expected):
                    raise ValueError(
                        "measurement configuration differs from the exact child candidate"
                    )
                if report.get("required_auxiliary_path") != (
                    expected.negative_loss_weight > 0
                ) or report.get("required_cycle_preparation") != (
                    expected.selection_mode == "forest_cycle"
                ):
                    raise ValueError("measurement auxiliary/cycle scope differs from its exact arm")
                if (report.get("batch_size"), report.get("workers")) != pair:
                    raise ValueError("measurement physical batch/worker differs from its candidate")
                _full_budget_seconds(report, policy)
    sizes = sorted({batch for batch, _ in seen})
    if (
        not sizes
        or sizes[0] != baseline
        or seen != {(size, count) for size in sizes for count in entry["worker_candidates"]}
    ):
        raise ValueError("common candidate grid is incomplete")
    if axis != "full_graph" and baseline < entry["natural_training_split_size"] and len(sizes) < 2:
        raise ValueError("at least two physical batch candidates must be measured")
    ceiling = max(baseline, entry["natural_training_split_size"])
    if any(
        next_size != min(size * 2, ceiling)
        for size, next_size in zip(sizes, sizes[1:], strict=False)
    ):
        raise ValueError("physical candidate grid skipped an unmeasured expansion")
    costs_by_size = {
        size: [
            score
            for candidate in entry["candidates"]
            if candidate["batch_size"] == size and (score := _score(candidate, policy)) is not None
        ]
        for size in sizes
    }
    reason = entry.get("stop_reason")
    if axis == "full_graph":
        if baseline != 1 or sizes != [1] or ceiling != 1 or reason != "full_graph_no_batch_axis":
            raise ValueError("full graph cannot claim a replicated physical-batch search")
    elif reason == "memory_headroom_boundary":
        if costs_by_size[sizes[-1]]:
            raise ValueError("reported memory boundary still has a common safe candidate")
    elif reason == "complete_training_split_boundary":
        if sizes[-1] != ceiling or not costs_by_size[sizes[-1]]:
            raise ValueError("reported full-split boundary was not measured")
    elif reason == "measured_full_budget_cost_plateau":
        previous, plateau = None, 0
        for costs in costs_by_size.values():
            if not costs:
                raise ValueError("cost plateau cannot hide an unsafe memory boundary")
            best = min(costs)
            plateau = plateau + 1 if previous is not None and best >= previous / 1.05 else 0
            previous = best if previous is None else min(previous, best)
        if plateau < 2:
            raise ValueError("cost plateau lacks two measured physical expansions")
    else:
        raise ValueError("common calibration has an unknown or unmeasured stopping boundary")
    chosen = _choose(entry["candidates"], policy)
    if entry.get("selected_candidate") != {
        "batch_size": chosen["batch_size"],
        "workers": chosen["workers"],
    }:
        raise ValueError(
            "selected resources differ from the measured worst-arm full-budget minimum"
        )
    expected_selected = _selected(parsed[0], axis, chosen)
    if entry.get("selected") != expected_selected:
        raise ValueError("stored selected resources differ from the measured configuration")


def _selected(args, axis, chosen):
    result = {
        "batch_size": args.batch_size,
        "workers": chosen["workers"],
        "sample_seed_batch_size": args.sample_seed_batch_size,
    }
    result["sample_seed_batch_size" if axis == "sampled_seed_nodes" else "batch_size"] = chosen[
        "batch_size"
    ]
    if args.sampling == "cluster_disjoint":
        result.update(workers=0, sample_context_workers=chosen["workers"])
    return result


def calibrate_group(jobs, entry, persist):
    from research.conductance_gat.edge_selection import train

    parsed = [parse_job(job) for job in jobs]
    loaded, identity, maximum, axis = train.load_calibration_payload(parsed[0])
    if entry.get("input_identity") is not None and (
        entry["input_identity"] != identity
        or entry["natural_training_split_size"] != maximum
        or entry["batch_axis"] != axis
    ):
        raise ValueError(
            "official data/topology calibration identity changed; previous evidence retained"
        )
    if entry.get("status") == "passed":
        validate_entry(entry, jobs)
        return
    policy = resources.learning_budget_selection_policy(
        vars(parsed[0]), training_split_size=maximum, batch_axis=axis
    )
    if policy is None:
        raise ValueError("edge-selection calibration requires explicit reference_updates budget")
    if entry.get("selection_policy") not in (None, policy):
        raise ValueError("partial common calibration budget changed")
    baseline = (
        parsed[0].sample_seed_batch_size if axis == "sampled_seed_nodes" else parsed[0].batch_size
    )
    context = parsed[0].sampling == "cluster_disjoint"
    requested_workers = parsed[0].sample_context_workers if context else parsed[0].workers
    workers = resources.worker_candidates(
        requested_workers, resources.allocated_cpu_count(), applicable=axis == "graphs" or context
    )
    workers = [count for count in workers if count <= max(2, 2 * requested_workers)]
    entry.update(
        track="conductance",
        profile=jobs[0]["profile"],
        dataset=jobs[0]["dataset"],
        input_identity=identity,
        natural_training_split_size=maximum,
        batch_axis=axis,
        baseline_physical_batch_size=baseline,
        worker_candidates=workers,
        selection_policy=policy,
        job_contracts=_contracts(jobs),
    )
    if context:
        entry["worker_axis"] = "sample_context_workers"
    entry.setdefault("candidates", [])
    current, plateau, previous_best = baseline, 0, None
    while True:
        costs = []
        for count in workers:
            candidate = next(
                (
                    item
                    for item in entry["candidates"]
                    if (item["batch_size"], item["workers"]) == (current, count)
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
            if candidate["status"] == "running":
                for job, args in zip(jobs, parsed, strict=True):
                    if not any(
                        (item["condition"], item["model_seed"])
                        == (job["variant_id"], job["model_seed"])
                        for item in candidate["measurements"]
                    ):
                        print(
                            f"[edge calibration] {job['job_id']} "
                            f"physical={current} workers={count}",
                            flush=True,
                        )
                        candidate["measurements"].append(
                            _measure(job, loaded, args, current, count)
                        )
                        persist()
                candidate["status"] = resources.completed_candidate_status(
                    candidate["measurements"]
                )
                persist()
            score = _score(candidate, policy)
            if score is not None:
                costs.append(score)
        if not costs:
            entry["stop_reason"] = "memory_headroom_boundary"
            break
        if axis == "full_graph" or current >= max(maximum, baseline):
            entry["stop_reason"] = (
                "full_graph_no_batch_axis"
                if axis == "full_graph"
                else "complete_training_split_boundary"
            )
            break
        best = min(costs)
        plateau = plateau + 1 if previous_best is not None and best >= previous_best / 1.05 else 0
        previous_best = best if previous_best is None else min(best, previous_best)
        if plateau >= 2:
            entry["stop_reason"] = "measured_full_budget_cost_plateau"
            break
        current = min(current * 2, max(maximum, baseline))
    chosen = _choose(entry["candidates"], policy)
    entry.update(
        status="passed",
        selected=_selected(parsed[0], axis, chosen),
        selected_candidate={"batch_size": chosen["batch_size"], "workers": chosen["workers"]},
        selection={
            "objective": "minimum worst-arm projected train + full validation + one-time setup",
            "preparation_accounting": (
                "per-epoch topology work is already in measured train/validation"
            ),
            "all_arms_share_resources": True,
            "global_optimum_claimed": False,
        },
    )
    validate_entry(entry, jobs)
    persist()
