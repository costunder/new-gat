"""CPU-only arithmetic/contract tests; never substitute for GPU training."""

from __future__ import annotations

import copy
import json

import pytest

from research.conductance_gat.v5.learning_budget import (
    compare_learning_budgets,
    deterministic_batches_per_epoch,
    plan_learning_budget,
    should_stop_learning_budget,
    validate_learning_budget,
)


@pytest.mark.parametrize("batch,batches", [(8, 3), (16, 2), (20, 1), (24, 1)])
def test_complete_ppi_split_is_counted_without_drop_or_batch_reduction(batch, batches):
    assert deterministic_batches_per_epoch(20, batch) == batches


@pytest.mark.parametrize(
    "actual_batch,epochs,patience,updates",
    [(8, 200, 50, 600), (16, 300, 75, 600), (20, 600, 150, 600)],
)
def test_ppi_reference_updates_preserve_baseline8_without_shrinking_physical_batch(
    actual_batch, epochs, patience, updates
):
    reference = deterministic_batches_per_epoch(20, 8)
    actual = deterministic_batches_per_epoch(20, actual_batch)
    plan = plan_learning_budget(200, 50, reference, actual, policy="reference_updates")
    assert plan["requested_epochs"] == 200 and plan["requested_patience"] == 50
    assert plan["planned_epochs"] == epochs and plan["planned_patience"] == patience
    assert plan["planned_maximum_optimizer_steps"] == updates
    assert plan["patience_optimizer_steps"] == 150
    assert plan["updates_preserved_against_reference"] is True
    assert plan["patience_updates_preserved_against_reference"] is True
    assert plan["early_stopping_can_finish_before_target"] is True
    validate_learning_budget(json.loads(json.dumps(plan)))


def test_default_epoch_policy_preserves_existing_recipe_and_reports_step_reduction():
    plan = plan_learning_budget(200, 50, 3, 1)
    assert plan["policy"] == "epochs"
    assert plan["planned_epochs"] == 200 and plan["planned_patience"] == 50
    assert plan["reference_optimizer_steps"] == 600
    assert plan["target_optimizer_steps"] == plan["planned_maximum_optimizer_steps"] == 200
    assert plan["reference_patience_optimizer_steps"] == 150
    assert plan["patience_optimizer_steps"] == 50
    assert plan["updates_preserved_against_reference"] is False
    assert plan["patience_updates_preserved_against_reference"] is False


def test_epochs_and_patience_are_never_reduced_when_actual_batch_count_increases():
    plan = plan_learning_budget(200, 50, 3, 5, policy="reference_updates")
    assert plan["planned_epochs"] == 200 and plan["planned_patience"] == 50
    assert plan["reference_optimizer_steps"] == 600
    assert plan["target_optimizer_steps"] == 1000
    assert plan["patience_optimizer_steps"] == 250


def test_epoch_boundary_rounding_is_explicit_not_a_fake_exact_update_budget():
    plan = plan_learning_budget(3, 1, 3, 2, policy="reference_updates")
    assert plan["planned_epochs"] == 5 and plan["planned_patience"] == 2
    assert plan["target_optimizer_steps"] == 9
    assert plan["planned_maximum_optimizer_steps"] == 10
    assert plan["epoch_rounding_extra_steps"] == 1
    assert plan["patience_optimizer_steps"] == 3
    assert plan["planned_patience_optimizer_steps"] == 4
    assert plan["patience_rounding_extra_steps"] == 1


def test_large_integer_budgets_do_not_lose_precision_to_float_ceil():
    epochs = 2**60 + 1
    plan = plan_learning_budget(epochs, 7, 3, 2, policy="reference_updates")
    assert plan["planned_epochs"] == (epochs * 3 + 1) // 2
    assert plan["target_optimizer_steps"] == epochs * 3
    assert plan["epoch_rounding_extra_steps"] == 1


@pytest.mark.parametrize("reference,actual", [(None, None), (3, None), (None, 1)])
def test_unknown_variable_batch_counts_stay_unknown_only_for_epoch_policy(reference, actual):
    plan = plan_learning_budget(200, 50, reference, actual)
    assert plan["planned_epochs"] == 200
    assert plan["updates_preserved_against_reference"] is None
    if actual is None:
        assert plan["target_optimizer_steps"] is None
        assert plan["planned_maximum_optimizer_steps"] is None
        assert plan["patience_optimizer_steps"] is None
    validate_learning_budget(plan)
    with pytest.raises(ValueError, match="known constant"):
        plan_learning_budget(200, 50, reference, actual, policy="reference_updates")


@pytest.mark.parametrize(
    "field",
    [
        "requested_epochs",
        "requested_patience",
        "reference_batches_per_epoch",
        "actual_batches_per_epoch",
    ],
)
@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, float("nan"), float("inf"), "3"])
def test_malformed_integer_budgets_fail_explicitly(field, value):
    arguments = dict(
        requested_epochs=200,
        requested_patience=50,
        reference_batches_per_epoch=3,
        actual_batches_per_epoch=1,
    )
    arguments[field] = value
    with pytest.raises(ValueError, match="positive integer"):
        plan_learning_budget(**arguments)


@pytest.mark.parametrize("policy", [None, "", "auto", "steps", True, 1])
def test_unknown_policies_do_not_silently_use_epoch_fallback(policy):
    with pytest.raises(ValueError, match="unsupported"):
        plan_learning_budget(200, 50, 3, 1, policy=policy)


@pytest.mark.parametrize("arguments", [(0, 8), (20, 0), (True, 8), (20, 8.0), (-1, 8)])
def test_batch_count_helper_rejects_empty_or_invalid_scope(arguments):
    with pytest.raises(ValueError, match="positive integer"):
        deterministic_batches_per_epoch(*arguments)


def test_all_reference_update_plans_meet_floor_without_excessive_integer_rounding():
    for epochs in (1, 7, 200):
        for reference in range(1, 12):
            for actual in range(1, 12):
                plan = plan_learning_budget(epochs, 5, reference, actual, "reference_updates")
                assert plan["planned_epochs"] >= epochs
                assert plan["planned_patience"] >= 5
                assert plan["planned_maximum_optimizer_steps"] >= epochs * reference
                assert plan["patience_optimizer_steps"] >= 5 * reference
                assert 0 <= plan["epoch_rounding_extra_steps"] < actual
                assert 0 <= plan["patience_rounding_extra_steps"] < actual


def test_reference_update_early_stopping_uses_real_steps_not_nominal_epochs():
    plan = plan_learning_budget(200, 50, 3, 1, "reference_updates")
    assert not should_stop_learning_budget(plan)
    assert not should_stop_learning_budget(
        plan, epochs_since_best=200, optimizer_steps_since_best=149
    )
    assert should_stop_learning_budget(plan, epochs_since_best=150, optimizer_steps_since_best=150)
    assert not should_stop_learning_budget(
        plan, epochs_since_best=200, optimizer_steps_since_best=200, eligible=False
    )
    with pytest.raises(ValueError, match="actual optimizer-step age"):
        should_stop_learning_budget(plan, epochs_since_best=200)


def test_epoch_policy_early_stopping_is_backward_compatible():
    plan = plan_learning_budget(200, 50, 3, 1)
    assert not should_stop_learning_budget(
        plan, epochs_since_best=49, optimizer_steps_since_best=5000
    )
    assert should_stop_learning_budget(plan, epochs_since_best=50, optimizer_steps_since_best=1)
    with pytest.raises(ValueError, match="epoch age"):
        should_stop_learning_budget(plan, optimizer_steps_since_best=50)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs_since_best", -1),
        ("optimizer_steps_since_best", -1),
        ("optimizer_steps_since_best", True),
        ("eligible", 1),
    ],
)
def test_early_stop_does_not_hide_invalid_or_rollback_counters(field, value):
    with pytest.raises(ValueError):
        should_stop_learning_budget(plan_learning_budget(200, 50, 3, 1), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("planned_epochs", 1),
        ("requested_epochs", 201),
        ("schema_version", True),
        ("planned_maximum_optimizer_steps", 200.0),
        ("policy", "auto"),
        ("new_field", 5),
    ],
)
def test_saved_plan_tampering_is_rejected_including_bool_int_type_aliases(field, value):
    plan = plan_learning_budget(200, 50, 3, 1)
    plan[field] = value
    with pytest.raises(ValueError):
        validate_learning_budget(plan)


def test_validation_and_comparison_do_not_mutate_inputs_or_claim_equal_completed_training():
    first = plan_learning_budget(200, 50, 3, 1, "reference_updates")
    second = plan_learning_budget(200, 50, 3, 2, "reference_updates")
    snapshots = copy.deepcopy((first, second))
    report = compare_learning_budgets(first, second)
    assert (first, second) == snapshots
    assert report["same_reference_optimizer_steps"] is True
    assert report["same_planned_optimizer_steps"] is True
    assert report["same_patience_optimizer_steps"] is True
    assert report["differences"]["planned_epochs"] == {"first": 600, "second": 300}
    assert report["differences"]["actual_batches_per_epoch"] == {"first": 1, "second": 2}
    assert report["actual_completed_updates_compared"] is False
    assert report["causal_comparability_claimed"] is False


def test_unknown_counts_are_not_reported_as_equal_update_budgets():
    plan = plan_learning_budget(200, 50, None, None)
    report = compare_learning_budgets(plan, plan)
    assert report["same_reference_optimizer_steps"] is None
    assert report["same_planned_optimizer_steps"] is None
    assert report["same_patience_optimizer_steps"] is None
