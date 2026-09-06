"""CPU-only report integrity fixtures; these are not trained model results."""

from __future__ import annotations

import copy
import json

import pytest

from research.conductance_gat.v5 import report
from research.conductance_gat.v5.learning_budget import plan_learning_budget


def _child():
    plan = plan_learning_budget(200, 50, 3, 1, "reference_updates")
    return {
        "dataset": "ppi",
        "data_observability": {"optimization_count": 20},
        "configuration": {
            "epochs": 200,
            "patience": 50,
            "training_schedule": "joint",
            "learning_budget_policy": "reference_updates",
            "hardware_profile": "a6000-48gb",
            "sampling": "full",
        },
        "learning_budget": plan,
        "batch_observability": {"training_batches_per_epoch": {"value": 1}},
        "schedule": [{"name": "joint", "start_epoch": 1, "end_epoch": 600, "length": 600}],
        "epochs_run": 400,
        "optimizer_steps": 400,
        "optimization_observability": {
            "learning_budget": copy.deepcopy(plan),
            "planned_training_epochs": 600,
            "epochs_requested": 200,
            "epochs_completed": 400,
            "actual_optimizer_steps": 400,
        },
    }


def test_valid_extended_budget_and_early_stopping_are_not_mistaken_for_200epoch_overrun():
    child = _child()
    assert report._validate_learning_budget(child) == child["learning_budget"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("learning_budget", "planned_epochs"), 1),
        (("learning_budget", "planned_maximum_optimizer_steps"), 999999),
        (("learning_budget", "patience_optimizer_steps"), 1),
        (("configuration", "epochs"), 201),
        (("configuration", "epochs"), 200.0),
        (("configuration", "patience"), 49),
        (("configuration", "training_schedule"), "staged"),
        (("batch_observability", "training_batches_per_epoch", "value"), 2),
        (("batch_observability", "training_batches_per_epoch", "value"), True),
        (("schedule",), [{"name": "joint", "start_epoch": 1, "end_epoch": 200, "length": 200}]),
        (("epochs_run",), 601),
        (("epochs_run",), True),
        (("optimizer_steps",), 601),
        (("optimizer_steps",), 399),
        (("optimization_observability", "epochs_completed"), 10),
        (("optimization_observability", "planned_training_epochs"), 200),
        (("optimization_observability", "actual_optimizer_steps"), 10),
        (("transition_provenance",), {"mode": "replace_c"}),
        (("resume_identity",), {"transition_request": {"mode": "replace_c"}}),
    ],
)
def test_arithmetic_or_execution_tampering_is_never_released_as_passed(path, value):
    child = _child()
    target = child
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(report.ComparisonIntegrityError, match="learning budget"):
        report._validate_learning_budget(child)


def test_epoch_legacy_artifacts_without_new_budget_fields_remain_accepted():
    assert report._validate_learning_budget({"configuration": {"epochs": 200}}) is None


def test_recomputed_forged_reference_count_still_fails_dataset_binding():
    child = _child()
    forged = plan_learning_budget(200, 50, 1, 1, "reference_updates")
    child.update(
        learning_budget=forged,
        epochs_run=200,
        optimizer_steps=200,
        schedule=[{"name": "joint", "start_epoch": 1, "end_epoch": 200, "length": 200}],
    )
    child["optimization_observability"].update(
        learning_budget=forged,
        epochs_completed=200,
        planned_training_epochs=200,
        actual_optimizer_steps=200,
    )
    with pytest.raises(report.ComparisonIntegrityError, match="reference updates differ"):
        report._validate_learning_budget(child)


def test_full_report_checks_budget_and_compares_actual_steps_without_causal_claims(tmp_path):
    # Reuse the existing explicitly synthetic artifact fixture, including its
    # unchanged hash checks; do not bypass the real report's artifact validator.
    from research.conductance_gat.tests.test_v5_contract import (
        test_report_is_partial_safe_then_requires_complete_pairs,
    )

    test_report_is_partial_safe_then_requires_complete_pairs(tmp_path)
    jobs = []
    for condition, completed in (("fixed_c", 180), ("shared_dynamic_c", 200)):
        output = tmp_path / condition
        path = output / "metrics.json"
        child = json.loads(path.read_text(encoding="utf-8"))
        architecture = dict(child["configuration"])
        child["configuration"].update(
            epochs=200,
            patience=50,
            training_schedule="joint",
            learning_budget_policy="reference_updates",
        )
        plan = plan_learning_budget(200, 50, 1, 1, "reference_updates")
        child.update(
            learning_budget=plan,
            epochs_run=completed,
            optimizer_steps=completed,
            batch_observability={"training_batches_per_epoch": {"value": 1}},
            schedule=[{"name": "joint", "start_epoch": 1, "end_epoch": 200, "length": 200}],
            optimization_observability={
                "learning_budget": plan,
                "planned_training_epochs": 200,
                "epochs_requested": 200,
                "epochs_completed": completed,
                "actual_optimizer_steps": completed,
            },
        )
        path.write_text(json.dumps(child), encoding="utf-8")
        jobs.append(
            {
                "dataset": "cora",
                "condition": condition,
                "status": "passed",
                "output_dir": str(output),
                "architecture": architecture,
                "sampling": "full",
            }
        )
    manifest = {"status": "passed", "config": {"datasets": ["cora"]}, "jobs": jobs}
    result = report.build_comparison(tmp_path, manifest)
    assert result["status"] == "passed"
    assert all(row["learning_budget"]["policy"] == "reference_updates" for row in result["rows"])
    contrast = result["contrasts"][0]["learning_budget_comparison"]
    assert contrast["same_planned_optimizer_steps"] is True
    assert contrast["completed_optimizer_step_difference"] == 20
    assert contrast["actual_completed_updates_compared"] is True
    assert contrast["causal_comparability_claimed"] is False
    path = tmp_path / "fixed_c" / "metrics.json"
    child = json.loads(path.read_text(encoding="utf-8"))
    child["learning_budget"]["planned_epochs"] = 201
    path.write_text(json.dumps(child), encoding="utf-8")
    with pytest.raises(report.ComparisonIntegrityError, match="learning budget"):
        report.build_comparison(tmp_path, manifest)
