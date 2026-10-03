"""Explicit DEBUG arithmetic fixtures for saved-measurement analysis."""

import copy
import json
from pathlib import Path

import pytest

from research.wedge_propagation.branch_analysis.core import (
    FIXED_METRICS,
    STRENGTH_METRICS,
    analyze,
)
from research.wedge_propagation.branch_strength.contract import fixed_variants, variants
from research.wedge_propagation.branch_strength.report import LEARNED


def debug_measurements():
    path = (
        Path(__file__).parents[1] / "research/wedge_propagation/branch_strength/config_debug.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    baseline, changed, diagnostics, fixed = [], [], [], []

    def diagnostic(base, layer, variant, source):
        first_seed = base["seed"] == 11
        z, reference = (10.0, 2.0) if first_seed else (20.0, 10.0)
        alpha, beta, kappa = (0.5, 0.2, 2.0) if first_seed else (0.4, 0.4, 8.0)
        name, index = variant["treatment"], variant["manifest_index"]
        gain = {
            "baseline": 1.0,
            "true_c_unit_denominator": kappa,
            "identity_hold_reference": 0.3,
            "identity_unit_denominator": 0.3 * kappa,
            "identity_norm_matched": 1.0,
            "branch_off": 0.0,
            "shuffle_hold_reference": 0.5 + 0.5 * index,
            "shuffle_norm_matched": 1.0,
        }[name]
        norm = reference * gain
        row = {
            **base,
            **variant,
            "source": source,
            "layer": layer,
            "alpha": alpha,
            "beta": beta,
            "kappa_reference": kappa,
            "kappa_used": kappa,
            "z_norm": z,
            "branch_norm": norm,
            "reference_branch_norm": reference,
            "l_norm": z * 0.5,
            "alpha_l_norm": alpha * z * 0.5,
            "beta_t_norm": beta * norm,
            "alpha_l_to_input": alpha * 0.5,
            "beta_t_to_input": beta * norm / z,
            "beta_t_to_alpha_l": beta * norm / (alpha * z * 0.5),
            "first_second_cosine": 0.7 if norm > 0 else None,
            "cosine_correction_input": -0.6,
            "kappa_ratio_mean": 1.0,
            "kappa_ratio_p50": 0.9,
            "kappa_ratio_p90": 1.2,
            "kappa_ratio_p99": kappa * 0.99,
            "kappa_ratio_max": kappa,
            "kappa_near_max_fraction": 0.01,
            "kappa_ratio_defined": True,
            "message_norm_ratio": norm / reference,
            "message_relative_change": abs(norm - reference) / reference,
            "branch_reference_cosine": 1.0 if norm > 0 else None,
        }
        for metric in (*STRENGTH_METRICS, *FIXED_METRICS):
            if metric in row and not metric.startswith("kappa_"):
                row[metric + "_defined"] = row[metric] is not None
        return row

    for dataset in config["data"]["datasets"]:
        for condition in config["conditions"]:
            for seed in config["final_seeds"]:
                base = {"dataset": dataset, "condition": condition, "seed": seed}
                for split in ("train", "validation", "test"):
                    metric = {**base, "split": split, "ce": 0.9, "accuracy": 0.5}
                    baseline.append(metric)
                    if condition in LEARNED:
                        for variant in variants(config, False):
                            gain = 0.01 * (1 + max(0, variant["manifest_index"]))
                            changed.append(
                                {
                                    **metric,
                                    **variant,
                                    "ce": 0.9 - gain,
                                    "accuracy": 0.5 + gain,
                                }
                            )
                for layer in (0, 1):
                    diagnostics.append(diagnostic(base, layer, variants(config)[0], "end_to_end"))
                    if condition not in LEARNED:
                        continue
                    for variant in variants(config, False):
                        diagnostics.append(diagnostic(base, layer, variant, "end_to_end"))
                    for variant in fixed_variants(config):
                        item = {**variant, "target": f"layer_{layer}"}
                        fixed.append(diagnostic(base, layer, item, "fixed_Z"))
    return config, baseline, changed, diagnostics, fixed


def selected(rows, **identity):
    return [row for row in rows if all(row.get(key) == value for key, value in identity.items())]


def test_factorization_is_per_seed_not_product_of_means_and_preserves_inputs():
    values = debug_measurements()
    original = copy.deepcopy(values)
    result = analyze(*values)
    assert values == original
    assert len(result["baseline_strength"]) == 24
    assert len(result["strength_estimates"]) == 12 * len(STRENGTH_METRICS)
    assert len(result["layer_changes"]) == 756
    assert len(result["fixed_estimates"]) == 12 * 8 * len(FIXED_METRICS)
    assert result["coverage"]["optimizer_updates"] == 0
    assert result["coverage"]["complete"] is True
    for row in result["baseline_strength"]:
        assert row["raw_branch_to_input"] == pytest.approx(
            row["kappa_reference"] * row["unscaled_branch_to_input"]
        )
        assert row["beta_t_to_input"] == pytest.approx(
            row["beta_over_kappa"] * row["raw_branch_to_input"]
        )
    group = selected(
        result["strength_estimates"], dataset="DEBUG-Cora", condition="learned_wedge_raw", layer=0
    )
    means = {row["metric"]: row["mean"] for row in group}
    assert means["beta_t_to_input"] == pytest.approx(0.12)
    assert means["beta_over_kappa"] * means["raw_branch_to_input"] == pytest.approx(0.165)
    assert means["beta_t_to_input"] != pytest.approx(
        means["beta_over_kappa"] * means["raw_branch_to_input"]
    )


def test_shuffle_manifests_are_within_seed_and_all_splits_scopes_are_retained():
    result = analyze(*debug_measurements())
    shuffle = selected(
        result["layer_changes"], treatment="shuffle_hold_reference", metric="accuracy"
    )
    assert len(shuffle) == 3 * 2 * 3 * 3
    assert all(row["count"] == 2 for row in shuffle)
    assert all(row["mean"] == pytest.approx(0.015) for row in shuffle)
    assert {row["split"] for row in shuffle} == {"train", "validation", "test"}
    assert {row["target"] for row in shuffle} == {"layer_0", "layer_1", "both"}
    fixed = selected(
        result["fixed_estimates"], treatment="shuffle_hold_reference", metric="message_norm_ratio"
    )
    assert all(row["count"] == 2 for row in fixed)
    assert all(row["mean"] == pytest.approx(0.75) for row in fixed)


def test_unit_denominator_increases_fixed_Z_branch_norm_by_each_original_kappa():
    values = debug_measurements()
    result = analyze(*values)
    unit = selected(
        result["fixed_estimates"], treatment="true_c_unit_denominator", metric="message_norm_ratio"
    )
    assert all(row["mean"] == pytest.approx(5.0) for row in unit)
    actual = selected(
        result["fixed_estimates"], treatment="true_c_unit_denominator", metric="beta_t_to_input"
    )
    assert all(row["mean"] == pytest.approx(0.84) for row in actual)
    matched = selected(
        result["fixed_estimates"], treatment="identity_norm_matched", metric="message_norm_ratio"
    )
    assert all(row["mean"] == pytest.approx(1.0) for row in matched)


def test_zero_input_keeps_undefined_ratios_and_counts_seed_missingness():
    values = debug_measurements()
    rows = selected(
        values[3],
        dataset="DEBUG-Cora",
        condition="learned_wedge_raw",
        seed=11,
        treatment="baseline",
        layer=0,
    )
    assert len(rows) == 1
    row = rows[0]
    for name in (
        "z_norm",
        "branch_norm",
        "reference_branch_norm",
        "l_norm",
        "alpha_l_norm",
        "beta_t_norm",
    ):
        row[name] = 0.0
    for name in (
        "alpha_l_to_input",
        "beta_t_to_input",
        "beta_t_to_alpha_l",
        "first_second_cosine",
        "cosine_correction_input",
    ):
        row[name], row[name + "_defined"] = None, False
    result = analyze(*values)
    derived = selected(
        result["baseline_strength"],
        dataset="DEBUG-Cora",
        condition="learned_wedge_raw",
        seed=11,
        layer=0,
    )[0]
    assert derived["raw_branch_to_input"] is None
    assert derived["unscaled_branch_to_input"] is None
    assert derived["raw_branch_to_input_defined"] is False
    aggregate = selected(
        result["strength_estimates"],
        dataset="DEBUG-Cora",
        condition="learned_wedge_raw",
        layer=0,
        metric="raw_branch_to_input",
    )[0]
    assert aggregate["count"] == 1
    assert aggregate["undefined_seeds"] == 1
    assert aggregate["undefined_observations"] == 1
    assert aggregate["mean"] == pytest.approx(4.0)


def test_zero_reference_fixed_Z_never_becomes_zero_norm_ratio():
    values = debug_measurements()
    rows = selected(
        values[4],
        dataset="DEBUG-Cora",
        condition="learned_wedge_raw",
        seed=11,
        treatment="shuffle_hold_reference",
        layer=0,
    )
    for row in rows:
        row["branch_norm"] = row["reference_branch_norm"] = 0.0
        for name in ("message_norm_ratio", "message_relative_change", "branch_reference_cosine"):
            row[name], row[name + "_defined"] = None, False
    result = analyze(*values)
    item = selected(
        result["fixed_estimates"],
        dataset="DEBUG-Cora",
        condition="learned_wedge_raw",
        layer=0,
        treatment="shuffle_hold_reference",
        metric="message_norm_ratio",
    )[0]
    assert item["mean"] == pytest.approx(0.75)
    assert item["count"] == 1 and item["undefined_seeds"] == 1
    assert item["undefined_observations"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("alpha", -0.1),
        ("beta", -0.1),
        ("beta", True),
        ("kappa_reference", 0.0),
        ("kappa_reference", -1.0),
        ("kappa_reference", None),
        ("z_norm", -1.0),
        ("branch_norm", None),
        ("beta_t_to_input", 0.8),
        ("beta_t_to_input", None),
        ("beta_t_to_input_defined", False),
        ("beta_t_norm", 0.8),
        ("alpha_l_norm", 0.2),
        ("kappa_used", 50.0),
        ("kappa_ratio_defined", False),
        ("first_second_cosine", 1.5),
    ],
)
def test_invalid_strength_measurements_are_errors(field, value):
    values = debug_measurements()
    row = selected(values[3], condition="learned_wedge_raw", treatment="baseline")[0]
    row[field] = value
    with pytest.raises(ValueError):
        analyze(*values)


@pytest.mark.parametrize("table", [1, 2, 3, 4])
def test_missing_or_duplicate_saved_rows_are_errors(table):
    values = debug_measurements()
    values[table].pop()
    with pytest.raises(ValueError, match="incomplete"):
        analyze(*values)
    values = debug_measurements()
    values[table].append(copy.deepcopy(values[table][0]))
    with pytest.raises(ValueError, match="duplicate"):
        analyze(*values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("message_norm_ratio", 200.0),
        ("message_norm_ratio_defined", False),
        ("branch_reference_cosine_defined", False),
        ("branch_norm", -1.0),
        ("beta_t_to_input", float("nan")),
    ],
)
def test_invalid_fixed_Z_measurements_are_errors(field, value):
    values = debug_measurements()
    values[4][0][field] = value
    with pytest.raises(ValueError):
        analyze(*values)


@pytest.mark.parametrize("field,value", [("accuracy", 2.0), ("ce", -1.0)])
def test_invalid_classification_metrics_are_errors(field, value):
    values = debug_measurements()
    values[2][0][field] = value
    with pytest.raises(ValueError):
        analyze(*values)


def test_training_scope_is_rejected():
    values = debug_measurements()
    values[0]["optimization_updates"] = 1
    with pytest.raises(ValueError, match="zero optimization updates"):
        analyze(*values)
