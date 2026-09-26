"""Fixed, fail-closed contract for checkpoint score reproduction.

Metrics are ratios of integer sufficient statistics, not averaged batch scores.
Tolerance only permits floating-point serialization arithmetic; every underlying
count must agree exactly. Changing this policy requires a new run identity.
"""

import math

POLICY = {
    "version": 1,
    "score_atol": 1e-12,
    "score_rtol": 0.0,
    "counts": "exact",
    "prediction_threshold": "logit > 0 for micro_f1",
}


def validate_evaluation(value, *, label="validation"):
    if not isinstance(value, dict):
        raise ValueError(f"{label}: missing evaluation evidence")
    counts = value.get("counts")
    kind = value.get("metric_kind")
    keys = {"correct", "total"} if kind == "accuracy" else {"tp", "fp", "fn", "total"}
    if kind not in {"accuracy", "micro_f1"} or not isinstance(counts, dict) or set(counts) != keys:
        raise ValueError(f"{label}: invalid metric/count schema")
    if (
        any(type(number) is not int or number < 0 for number in counts.values())
        or counts["total"] < 1
    ):
        raise ValueError(f"{label}: counts must be nonnegative integers with positive total")
    if kind == "accuracy":
        if counts["correct"] > counts["total"]:
            raise ValueError(f"{label}: correct count exceeds total")
        expected = counts["correct"] / counts["total"]
    else:
        if counts["tp"] + counts["fp"] + counts["fn"] > counts["total"]:
            raise ValueError(f"{label}: confusion counts exceed total label decisions")
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        expected = 2 * counts["tp"] / denominator if denominator else 0.0
    score = value.get("metric")
    if (
        isinstance(score, bool)
        or not isinstance(score, (float, int))
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or abs(score - expected) > POLICY["score_atol"]
    ):
        raise ValueError(f"{label}: score disagrees with integer counts")
    return value


def require_reproduction(selected, actual, *, label):
    validate_evaluation(selected, label="selected checkpoint")
    validate_evaluation(actual, label=label)
    if (
        selected["metric_kind"] != actual["metric_kind"]
        or selected["counts"] != actual["counts"]
        or abs(selected["metric"] - actual["metric"]) > POLICY["score_atol"]
    ):
        raise ValueError(
            f"{label}: selected-checkpoint validation reproduction failed; "
            f"selected={selected}, actual={actual}, policy={POLICY}"
        )


def require_score(selected, actual, *, label):
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (float, int))
        or not math.isfinite(actual)
        or abs(selected - actual) > POLICY["score_atol"]
    ):
        raise ValueError(
            f"{label}: selected-checkpoint validation reproduction failed ({selected} -> {actual})"
        )
