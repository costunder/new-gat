"""Explicit epoch and optimizer-update budgets; no training or resource side effects.

``epochs`` preserves the historical recipe. ``reference_updates`` is an explicit
new recipe: retain at least the requested epoch/patience horizons and, when a
larger physical batch decreases updates per epoch, extend those horizons to
cover the reference update budget. No batch, model, graph or sample is reduced.

Counts mean optimizer updates, assuming one update per physical batch and a
constant complete-epoch batch count. They are not forward calls, C solver inner
iterations, or parameter-group updates in a staged schedule. Early stopping can
finish before the planned full-run capacity; this module does not claim that
capacity was actually trained or that different batches have equal trajectories.
"""

from __future__ import annotations

from typing import Any

LEARNING_BUDGET_POLICIES = ("epochs", "reference_updates")
DEFAULT_LEARNING_BUDGET_POLICY = "epochs"


def _integer(value: Any, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        qualifier = "nonnegative" if zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _optional_batch_count(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _ceil_div(numerator: int, denominator: int) -> int:
    # Integer arithmetic preserves exact budgets even above float's 2**53 limit.
    return (numerator + denominator - 1) // denominator


def deterministic_batches_per_epoch(training_units: int, physical_batch_size: int) -> int:
    """Count every unit, including a final partial batch (drop_last=False).

    The caller supplies the verified graph/seed-node count and chosen physical
    batch. This does not infer a dataset size, replicate a full graph, or support
    variable topology-dependent batch counts by inventing an average.
    """

    return _ceil_div(
        _integer(training_units, "training_units"),
        _integer(physical_batch_size, "physical_batch_size"),
    )


def plan_learning_budget(
    requested_epochs: int,
    requested_patience: int,
    reference_batches_per_epoch: int | None,
    actual_batches_per_epoch: int | None,
    policy: str = DEFAULT_LEARNING_BUDGET_POLICY,
) -> dict[str, Any]:
    """Build a JSON-safe immutable recipe description, without changing arguments.

    For known reference R and actual A, the explicit update policy plans
    E'=max(E, ceil(E*R/A)) and P'=max(P, ceil(P*R/A)). Step-based patience is
    max(P*R, P*A), evaluated at the caller's real validation boundaries. Rounding
    to a complete epoch may exceed a target; that excess is reported explicitly.

    Unknown/variable batch counts are accepted only for the legacy epoch policy
    and stay ``None`` in derived fields. They are never silently replaced by 1.
    """

    epochs = _integer(requested_epochs, "requested_epochs")
    patience = _integer(requested_patience, "requested_patience")
    reference = _optional_batch_count(reference_batches_per_epoch, "reference_batches_per_epoch")
    actual = _optional_batch_count(actual_batches_per_epoch, "actual_batches_per_epoch")
    if policy not in LEARNING_BUDGET_POLICIES:
        raise ValueError(f"unsupported learning budget policy: {policy!r}")
    if policy == "reference_updates" and (reference is None or actual is None):
        raise ValueError(
            "reference_updates requires known constant reference and actual batches per epoch; "
            "variable samplers need an explicit measured update-budget implementation"
        )

    reference_steps = None if reference is None else epochs * reference
    requested_steps = None if actual is None else epochs * actual
    reference_patience = None if reference is None else patience * reference
    requested_patience_steps = None if actual is None else patience * actual
    planned_epochs, planned_patience = epochs, patience
    target_steps, patience_steps = requested_steps, requested_patience_steps
    if policy == "reference_updates":
        target_steps = max(reference_steps, requested_steps)
        patience_steps = max(reference_patience, requested_patience_steps)
        planned_epochs = _ceil_div(target_steps, actual)
        planned_patience = _ceil_div(patience_steps, actual)
    planned_steps = None if actual is None else planned_epochs * actual
    planned_patience_steps = None if actual is None else planned_patience * actual
    return {
        "schema_version": 1,
        "policy": policy,
        "requested_epochs": epochs,
        "requested_patience": patience,
        "reference_batches_per_epoch": reference,
        "actual_batches_per_epoch": actual,
        "planned_epochs": planned_epochs,
        "planned_patience": planned_patience,
        "reference_optimizer_steps": reference_steps,
        "requested_epoch_optimizer_steps": requested_steps,
        "target_optimizer_steps": target_steps,
        "planned_maximum_optimizer_steps": planned_steps,
        "epoch_rounding_extra_steps": None if actual is None else planned_steps - target_steps,
        "reference_patience_optimizer_steps": reference_patience,
        "patience_optimizer_steps": patience_steps,
        "planned_patience_optimizer_steps": planned_patience_steps,
        "patience_rounding_extra_steps": (
            None if actual is None else planned_patience_steps - patience_steps
        ),
        "epoch_extension": planned_epochs - epochs,
        "patience_epoch_extension": planned_patience - patience,
        "updates_preserved_against_reference": (
            None
            if reference_steps is None or planned_steps is None
            else planned_steps >= reference_steps
        ),
        "patience_updates_preserved_against_reference": (
            None
            if reference_patience is None or patience_steps is None
            else patience_steps >= reference_patience
        ),
        "early_stopping_unit": "optimizer_steps" if policy == "reference_updates" else "epochs",
        "early_stopping_can_finish_before_target": True,
        "batch_count_assumption": (
            "constant complete-epoch batch count; one optimizer update per physical batch"
            if actual is not None
            else "unknown; no optimizer-update count inferred"
        ),
        "parameter_group_update_equality_claimed": False,
        "equal_optimization_trajectory_claimed": False,
    }


def validate_learning_budget(plan: dict[str, Any]) -> None:
    """Reject changed derived fields or unknown schema instead of relaxing resume."""

    if not isinstance(plan, dict):
        raise ValueError("learning budget must be a complete plan object")
    names = (
        "requested_epochs",
        "requested_patience",
        "reference_batches_per_epoch",
        "actual_batches_per_epoch",
        "policy",
    )
    if any(name not in plan for name in names):
        raise ValueError("learning budget is missing its requested recipe")
    expected = plan_learning_budget(**{name: plan[name] for name in names})
    # Dictionary equality alone treats True==1 and 200.0==200 as equal. Require
    # canonical value types as well, so resume evidence cannot change schema.
    if set(plan) != set(expected) or any(
        type(plan[key]) is not type(value) or plan[key] != value for key, value in expected.items()
    ):
        raise ValueError("learning budget does not match its exact derived recipe")


def should_stop_learning_budget(
    plan: dict[str, Any],
    *,
    epochs_since_best: int | None = None,
    optimizer_steps_since_best: int | None = None,
    eligible: bool = True,
) -> bool:
    """Apply patience using actual age of the caller's eligible best checkpoint.

    Both ages absent means no eligible best exists yet. The caller remains
    responsible for selecting fixed/global vs dynamic/joint best, updating its
    actual step counter, and persisting that counter for deterministic resume.
    This helper never substitutes ``epoch * batches`` for actual updates.
    """

    validate_learning_budget(plan)
    if type(eligible) is not bool:
        raise ValueError("early-stopping eligibility must be boolean")
    if epochs_since_best is not None:
        _integer(epochs_since_best, "epochs_since_best", zero=True)
    if optimizer_steps_since_best is not None:
        _integer(optimizer_steps_since_best, "optimizer_steps_since_best", zero=True)
    if not eligible or (epochs_since_best is None and optimizer_steps_since_best is None):
        return False
    if plan["policy"] == "reference_updates":
        if optimizer_steps_since_best is None:
            raise ValueError("reference_updates early stopping requires actual optimizer-step age")
        return optimizer_steps_since_best >= plan["patience_optimizer_steps"]
    if epochs_since_best is None:
        raise ValueError("epochs early stopping requires epoch age")
    return epochs_since_best >= plan["requested_patience"]


def compare_learning_budgets(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Expose per-condition budget differences; never assert causal comparability."""

    validate_learning_budget(first)
    validate_learning_budget(second)
    fields = (
        "policy",
        "requested_epochs",
        "requested_patience",
        "reference_batches_per_epoch",
        "actual_batches_per_epoch",
        "planned_epochs",
        "planned_patience",
        "target_optimizer_steps",
        "planned_maximum_optimizer_steps",
        "patience_optimizer_steps",
    )
    differences = {
        key: {"first": first[key], "second": second[key]}
        for key in fields
        if first[key] != second[key]
    }

    def known_equal(key: str) -> bool | None:
        return None if first[key] is None or second[key] is None else first[key] == second[key]

    return {
        "differences": differences,
        "same_planned_optimizer_steps": known_equal("planned_maximum_optimizer_steps"),
        "same_patience_optimizer_steps": known_equal("patience_optimizer_steps"),
        "same_reference_optimizer_steps": known_equal("reference_optimizer_steps"),
        "actual_completed_updates_compared": False,
        "equal_optimization_trajectory_claimed": False,
        "causal_comparability_claimed": False,
    }
