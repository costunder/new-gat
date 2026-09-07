"""CPU-only synthetic timing fixtures; never actual GPU throughput claims."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from scripts import calibrate_training_resources as calibration
from scripts import training_resource_plan as resources


def _configuration(*, reference=8, epochs=200):
    return {
        "learning_budget_policy": "reference_updates",
        "budget_reference_batch_size": reference,
        "hardware_profile": "a6000-48gb",
        "epochs": epochs,
        "patience": 50,
    }


def _policy(*, split=64, axis="sampled_seed_nodes", reference=8, epochs=200):
    return resources.learning_budget_selection_policy(
        _configuration(reference=reference, epochs=epochs),
        training_split_size=split,
        batch_axis=axis,
    )


def _report(condition, batch, *, split=64, seconds_per_step=1.0, reference=8, epochs=200):
    complete_epochs = 5
    updates = complete_epochs * math.ceil(split / batch)
    units = complete_epochs * split
    seconds = updates * seconds_per_step
    return {
        "status": "passed",
        "condition": condition,
        "model_seed": 0,
        "batch_size": batch,
        "workers": 0,
        "unit": "supervised_seed_nodes",
        "elapsed_seconds": seconds,
        "processed_units": units,
        "samples_per_second": units / seconds,
        "optimizer_steps": updates,
        "complete_measurement_epochs": complete_epochs,
        "configuration": _configuration(reference=reference, epochs=epochs),
        "optimizer_state_bytes": 1024,
        "measurement_steps_requested": 5,
        "warmup_steps_requested": 2,
        "minimum_measure_seconds_requested": 3.0,
        "peak_allocated_bytes": 8 * 1024**3,
        "peak_reserved_bytes": 10 * 1024**3,
        "total_memory_bytes": 48 * 1024**3,
        "free_bytes_before": 46 * 1024**3,
    }


def _candidate(batch, *, seconds_per_step=1.0, **kwargs):
    return {
        "status": "passed",
        "batch_size": batch,
        "workers": 0,
        "measurements": [
            _report(condition, batch, seconds_per_step=seconds_per_step, **kwargs)
            for condition in ("fixed_c", "shared_dynamic_c")
        ],
    }


def test_samples_per_second_winner_can_lose_the_same_full_update_budget():
    small, large = _candidate(8), _candidate(16, seconds_per_step=1.5)
    assert resources.choose_candidate([small, large], 8) is large
    assert resources.choose_candidate([small, large], 8, selection_policy=_policy()) is small
    cost = resources.projected_training_budget_cost(large["measurements"][0], _policy())
    assert cost["learning_budget"]["planned_epochs"] == 400
    assert cost["learning_budget"]["planned_maximum_optimizer_steps"] == 1600
    assert cost["projected_training_seconds"] == 2400
    assert "validation/checkpoint/final evaluation excluded" in cost["cost_scope"]
    assert cost["early_stopping_assumed"] is False


def test_selection_keeps_the_paired_slow_arm_and_requested_batch_floor():
    small, large = _candidate(8), _candidate(16, seconds_per_step=1.5)
    fast = large["measurements"][0]
    fast["elapsed_seconds"] = 5.0
    fast["samples_per_second"] = fast["processed_units"] / 5.0
    assert resources.choose_candidate([small, large], 8, selection_policy=_policy()) is small
    with pytest.raises(ValueError, match="cannot shrink"):
        resources.choose_candidate([small, large], 16, selection_policy=_policy())


def test_budget_cost_uses_complete_epoch_rounding_not_just_raw_target_steps():
    policy = _policy(split=10, reference=4, epochs=3)
    report = _report("fixed_c", 6, split=10, reference=4, epochs=3)
    cost = resources.projected_training_budget_cost(report, policy)
    assert cost["learning_budget"]["target_optimizer_steps"] == 9
    assert cost["learning_budget"]["planned_maximum_optimizer_steps"] == 10
    assert cost["projected_training_seconds"] == 10


def test_arxiv_750_epoch_plan_is_costed_as_9000_actual_updates():
    policy = _policy(split=90941, reference=2048)
    report = _report("shared_dynamic_c", 8192, split=90941, reference=2048)
    cost = resources.projected_training_budget_cost(report, policy)
    assert cost["learning_budget"]["actual_batches_per_epoch"] == 12
    assert cost["learning_budget"]["planned_epochs"] == 750
    assert cost["projected_training_seconds"] == 9000


def test_full_graph_has_one_batch_and_no_epoch_extension():
    policy = _policy(split=1, axis="full_graph", reference=None)
    report = _report("fixed_c", 1, split=1, reference=None)
    report["processed_units"] = 90941 * 5
    report["samples_per_second"] = report["processed_units"] / report["elapsed_seconds"]
    cost = resources.projected_training_budget_cost(report, policy)
    assert cost["learning_budget"]["planned_epochs"] == 200
    assert cost["learning_budget"]["target_optimizer_steps"] == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete_measurement_epochs", None),
        ("complete_measurement_epochs", 4),
        ("optimizer_steps", 39),
        ("processed_units", 319),
        ("configuration", {}),
    ],
)
def test_budget_objective_rejects_incomplete_or_unbound_measurement(field, value):
    report = _report("fixed_c", 8)
    report[field] = value
    with pytest.raises(ValueError):
        resources.projected_training_budget_cost(report, _policy())


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "wrong"),
        ("reference_batches_per_epoch", 7),
        ("validation_cost_included", True),
        ("early_stopping_assumed", True),
    ],
)
def test_selection_policy_is_canonical_and_cannot_claim_unmeasured_costs(field, value):
    policy = _policy()
    policy[field] = value
    with pytest.raises(ValueError, match="canonical"):
        resources.candidate_score(_candidate(8), selection_policy=policy)


def _harness(monkeypatch, *, legacy=False):
    jobs = [
        {
            "track": "conductance",
            "profile": "reference",
            "dataset": "ogbn-arxiv",
            "condition": condition,
            "model_seed": 0,
            "command": [
                "python",
                "-m",
                "research.conductance_gat.v5.train",
                "--condition",
                condition,
                "--learning-budget-policy",
                "epochs" if legacy else "reference_updates",
            ],
        }
        for condition in ("fixed_c", "shared_dynamic_c")
    ]
    config = _configuration()
    if legacy:
        config["learning_budget_policy"] = "epochs"
    args = SimpleNamespace(**config, batch_size=1, sample_seed_batch_size=8, workers=0)
    monkeypatch.setattr(calibration, "_training_args", lambda job: copy.copy(args))
    identity = {"debug_cpu_fixture_only": True}
    monkeypatch.setattr(
        calibration, "_load_group", lambda *args: (object(), identity, 64, "sampled_seed_nodes")
    )
    monkeypatch.setattr(calibration, "allocated_cpu_count", lambda: 8)
    monkeypatch.setattr(calibration, "worker_candidates", lambda *args, **kwargs: [0])
    calls = []

    def measure(job, loaded, args, *, batch_size, workers):
        calls.append((batch_size, job["condition"]))
        seconds = {8: 1.0, 16: 1.5, 32: 2.5, 64: 4.0}[batch_size]
        result = _report(job["condition"], batch_size, seconds_per_step=seconds)
        result["configuration"] = vars(args)
        return result

    monkeypatch.setattr(calibration, "_measure", measure)
    return jobs, args, calls


def test_group_uses_cost_objective_for_selection_and_growth_plateau(monkeypatch):
    jobs, _, calls = _harness(monkeypatch)
    entry = {}
    calibration._calibrate_group(jobs, entry, lambda: None)
    assert entry["selected"]["sample_seed_batch_size"] == 8
    assert entry["stop_reason"] == "measured_budget_cost_plateau"
    assert sorted({batch for batch, _ in calls}) == [8, 16, 32]
    assert entry["selection_policy"] == _policy()
    assert len(entry["selection"]["projected_budget_costs"]) == 2
    resources._validate_entry(entry, [0], allocated_cpus=8)
    before, call_count = copy.deepcopy(entry), len(calls)
    calibration._calibrate_group(jobs, entry, lambda: pytest.fail("completed plan is immutable"))
    assert entry == before and len(calls) == call_count
    calibration.verify_plan_inputs({"entries": [entry]}, jobs)


def test_legacy_completed_group_is_not_silently_reselected(monkeypatch, capsys):
    jobs, args, calls = _harness(monkeypatch, legacy=True)
    entry = {}
    calibration._calibrate_group(jobs, entry, lambda: None)
    assert "selection_policy" not in entry
    assert entry["selected"]["sample_seed_batch_size"] == 64
    before, call_count = copy.deepcopy(entry), len(calls)
    args.learning_budget_policy = "reference_updates"
    calibration._calibrate_group(jobs, entry, lambda: pytest.fail("completed plan is immutable"))
    assert entry == before and len(calls) == call_count
    calibration.verify_plan_inputs({"entries": [entry]}, jobs)
    assert "legacy samples/second selection remains immutable" in capsys.readouterr().out


def test_partial_legacy_group_rejects_new_objective_before_any_probe(monkeypatch):
    jobs, args, calls = _harness(monkeypatch, legacy=True)
    entry = {}
    calibration._calibrate_group(jobs, entry, lambda: None)
    entry["status"] = "interrupted"
    before, call_count = copy.deepcopy(entry), len(calls)
    args.learning_budget_policy = "reference_updates"
    with pytest.raises(ValueError, match="partial calibration selection policy differs"):
        calibration._calibrate_group(jobs, entry, lambda: pytest.fail("must preserve evidence"))
    assert entry == before and len(calls) == call_count


def test_new_plan_recipe_change_is_rejected_before_reuse(monkeypatch):
    jobs, args, _ = _harness(monkeypatch)
    entry = {}
    calibration._calibrate_group(jobs, entry, lambda: None)
    args.epochs += 1
    with pytest.raises(ValueError, match="selection policy differs"):
        calibration.verify_plan_inputs({"entries": [entry]}, jobs)


def test_identity_explicitly_binds_new_objective_without_changing_legacy_shape():
    entry = {
        "track": "conductance",
        "profile": "reference",
        "dataset": "ogbn-arxiv",
        "selected": {"batch_size": 1, "sample_seed_batch_size": 8, "workers": 0},
    }
    plan = {"_sha256": "cpu_fixture", "entries": [entry]}
    assert resources.resource_plan_identity(plan)["selections"] == [entry]
    entry["selection_policy"] = _policy()
    assert resources.resource_plan_identity(plan)["selections"][0]["selection_policy"] == _policy()


def test_selected_cost_metadata_is_recomputed_and_tampering_rejected(monkeypatch):
    jobs, _, _ = _harness(monkeypatch)
    entry = {}
    calibration._calibrate_group(jobs, entry, lambda: None)
    entry["selection"]["projected_budget_costs"][0]["projected_training_seconds"] = 1.0
    with pytest.raises(ValueError, match="projected budget costs differ"):
        resources._validate_entry(entry, [0], allocated_cpus=8)


def test_partial_new_policy_reuses_completed_measurements_without_rerunning(monkeypatch):
    jobs, _, calls = _harness(monkeypatch)
    entry = {}
    calibration._calibrate_group(jobs, entry, lambda: None)
    entry["status"] = "interrupted"
    before = copy.deepcopy(entry["candidates"])
    call_count = len(calls)
    calibration._calibrate_group(jobs, entry, lambda: None)
    assert entry["status"] == "passed"
    assert entry["candidates"] == before and len(calls) == call_count


def test_inconsistent_paired_budgets_fail_before_any_measurement(monkeypatch):
    jobs, _, calls = _harness(monkeypatch)
    original = calibration._training_args

    def changed(job):
        args = original(job)
        if job["condition"] == "shared_dynamic_c":
            args.epochs += 1
        return args

    monkeypatch.setattr(calibration, "_training_args", changed)
    with pytest.raises(ValueError, match="same learning budget"):
        calibration._calibrate_group(
            jobs, {}, lambda: pytest.fail("do not publish a partial recipe")
        )
    assert not calls
