"""Analyze saved Experiment 4.1 measurements without new model evaluations.

For each original seed and layer, M = R/kappa, where
R = S_Q A^T C A S_Q Z/3.  The actual branch ratio factors as
||beta M||/||Z|| = (beta/kappa) (||R||/||Z||).
The identity holds per seed; products of aggregate means do not preserve it.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from ..branch_strength.contract import verify_coverage
from ..branch_strength.report import LEARNED, diagnostic_estimates, paired_changes

STRENGTH_METRICS = (
    "alpha",
    "beta",
    "kappa_reference",
    "beta_over_kappa",
    "raw_branch_to_input",
    "unscaled_branch_to_input",
    "alpha_l_to_input",
    "beta_t_to_input",
    "beta_t_to_alpha_l",
    "first_second_cosine",
    "cosine_correction_input",
    "kappa_ratio_mean",
    "kappa_ratio_p50",
    "kappa_ratio_p90",
    "kappa_ratio_p99",
    "kappa_ratio_max",
    "kappa_near_max_fraction",
)
FIXED_METRICS = (
    "message_norm_ratio",
    "branch_reference_cosine",
    "message_relative_change",
    "beta_t_to_input",
)
_KAPPA_METRICS = {
    "kappa_ratio_mean",
    "kappa_ratio_p50",
    "kappa_ratio_p90",
    "kappa_ratio_p99",
    "kappa_ratio_max",
    "kappa_near_max_fraction",
}
_REL_TOL = 5e-6
_ABS_TOL = 1e-8


def _number(row: dict, name: str, *, nonnegative: bool = False) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite saved measurement")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _nullable(row: dict, name: str) -> float | None:
    flag_name = "kappa_ratio_defined" if name in _KAPPA_METRICS else name + "_defined"
    flag = row.get(flag_name)
    if not isinstance(flag, bool) or name not in row or flag is not (row[name] is not None):
        raise ValueError(f"{name} must preserve its saved undefined flag")
    if not flag:
        return None
    value = _number(row, name)
    if "cosine" in name and abs(value) > 1.00001:
        raise ValueError(f"{name} is outside numerical [-1,1]")
    if "cosine" not in name and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _equal(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
        raise ValueError(f"saved {name} disagrees with per-seed branch factorization")


def _strength_row(row: dict) -> dict:
    alpha = _number(row, "alpha", nonnegative=True)
    beta = _number(row, "beta", nonnegative=True)
    kappa = _number(row, "kappa_reference")
    if kappa <= 0:
        raise ValueError("positive kappa_reference is required")
    z_norm = _number(row, "z_norm", nonnegative=True)
    branch_norm = _number(row, "branch_norm", nonnegative=True)
    reference_norm = _number(row, "reference_branch_norm", nonnegative=True)
    beta_t_norm = _number(row, "beta_t_norm", nonnegative=True)
    alpha_l_norm = _number(row, "alpha_l_norm", nonnegative=True)
    l_norm = _number(row, "l_norm", nonnegative=True)
    _equal(branch_norm, reference_norm, "baseline/reference message norm")
    _equal(_number(row, "kappa_used"), kappa, "baseline kappa denominator")
    _equal(beta_t_norm, beta * branch_norm, "beta message norm")
    _equal(alpha_l_norm, alpha * l_norm, "alpha message norm")
    result = dict(row)
    for name in STRENGTH_METRICS:
        if name not in {
            "alpha",
            "beta",
            "kappa_reference",
            "beta_over_kappa",
            "raw_branch_to_input",
            "unscaled_branch_to_input",
        }:
            _nullable(row, name)
    ratio_defined = z_norm > 0
    unscaled = branch_norm / z_norm if ratio_defined else None
    raw = kappa * unscaled if ratio_defined else None
    result.update(
        beta_over_kappa=beta / kappa,
        beta_over_kappa_defined=True,
        unscaled_branch_to_input=unscaled,
        unscaled_branch_to_input_defined=ratio_defined,
        raw_branch_to_input=raw,
        raw_branch_to_input_defined=ratio_defined,
    )
    actual = _nullable(row, "beta_t_to_input")
    if row["beta_t_to_input_defined"] is not ratio_defined:
        raise ValueError("beta_t_to_input definition must follow positive z_norm")
    first = _nullable(row, "alpha_l_to_input")
    if row["alpha_l_to_input_defined"] is not ratio_defined:
        raise ValueError("alpha_l_to_input definition must follow positive z_norm")
    if ratio_defined:
        _equal(actual, beta * unscaled, "beta_t_to_input")
        _equal(actual, (beta / kappa) * raw, "raw branch ratio")
        _equal(first, alpha_l_norm / z_norm, "alpha_l_to_input")
    second_first = _nullable(row, "beta_t_to_alpha_l")
    if row["beta_t_to_alpha_l_defined"] is not (alpha_l_norm > 0):
        raise ValueError("beta_t_to_alpha_l definition must follow positive alpha_l_norm")
    if alpha_l_norm > 0:
        _equal(second_first, beta_t_norm / alpha_l_norm, "beta_t_to_alpha_l")
    return result


def _validate_fixed(row: dict) -> None:
    candidate = _number(row, "branch_norm", nonnegative=True)
    reference = _number(row, "reference_branch_norm", nonnegative=True)
    saved_ratio = _nullable(row, "message_norm_ratio")
    if row["message_norm_ratio_defined"] is not (reference > 0):
        raise ValueError("message_norm_ratio definition must follow positive reference norm")
    if reference > 0:
        _equal(saved_ratio, candidate / reference, "fixed-Z message norm ratio")
    for name in FIXED_METRICS[1:]:
        _nullable(row, name)


def analyze(
    config: dict,
    baseline_rows: list[dict],
    treatment_rows: list[dict],
    diagnostic_rows: list[dict],
    fixed_z_rows: list[dict],
) -> dict[str, Any]:
    """Use every saved row and retain independent seeds as the statistical unit.

    Scope completeness is verified before arithmetic. Shuffle manifests are
    repeated interventions within one seed, not extra independent replicates.
    Undefined quantities remain None and are counted by the existing estimator.
    Input rows/configuration are never mutated.
    """
    if config.get("optimization_updates") != 0:
        raise ValueError("saved diagnostic must have zero optimization updates")
    coverage = verify_coverage(
        config,
        baseline_rows,
        treatment_rows,
        diagnostic_rows,
        fixed_z_rows,
    )
    strength = [
        _strength_row(row)
        for row in diagnostic_rows
        if row["condition"] in LEARNED and row["treatment"] == "baseline"
    ]
    expected = len(config["data"]["datasets"]) * len(LEARNED) * len(config["final_seeds"]) * 2
    if len(strength) != expected:
        raise ValueError("baseline learned seed/layer coverage differs from saved scope")
    for row in baseline_rows + treatment_rows:
        _number(row, "ce", nonnegative=True)
        accuracy = _number(row, "accuracy")
        if not 0 <= accuracy <= 1:
            raise ValueError("saved accuracy must be in [0,1]")
    for row in fixed_z_rows:
        _validate_fixed(row)
    return {
        "baseline_strength": strength,
        "strength_estimates": [
            estimate
            for name in STRENGTH_METRICS
            for estimate in diagnostic_estimates(strength, name)
        ],
        "layer_changes": [
            change
            for metric in ("accuracy", "ce")
            for change in paired_changes(baseline_rows, treatment_rows, metric)
        ],
        "fixed_estimates": [
            estimate
            for name in FIXED_METRICS
            for estimate in diagnostic_estimates(fixed_z_rows, name)
        ],
        "coverage": coverage,
    }
