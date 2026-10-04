"""Explicit DEBUG reporting fixtures; these are not task performance evidence.

Only reporting arithmetic, contract coverage, units, nulls and provenance are
tested here. Model/geometry tests independently verify actual S/G propagation.
"""

from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np
import pytest

from research.local_context_coupling.normalization.common import CONDITIONS, condition_metadata, parse_condition, read_config
from research.local_context_coupling.normalization.report import (
    ARTIFACTS, BRANCH_METRICS, branch_estimates, comparisons, condition_id,
    interaction_comparisons, interaction_contrasts, intervention_changes,
    intervention_variants, legacy_baseline, metric_estimates, paired_comparisons,
    validate_rows, write_report,
)
from research.wedge_propagation.classification.common import digest


def evidence():
    config = read_config(profile="debug")
    datasets, seeds = config["data"]["datasets"], config["training"]["final_seeds"]
    hashes = {"DEBUG-report-fixture.py": "b" * 64}
    completion = {
        "profile": "debug", "actual_data": False, "config_digest": digest(config),
        "source": {"sha256": hashes, "code_digest": digest(hashes)},
        "data_manifest_digest": "c" * 64, "parameter_counts": {},
        "coverage": {"tuning_runs": 240, "final_runs": 120, "total_runs": 360,
                     "contract_optimizer_updates": 1080, "all_datasets_conditions_seeds_splits": True},
    }
    metrics, probes, branches, choices, proofs = [], [], [], [], []
    for dataset in datasets:
        shape = config["data"]["expected_shapes"][dataset]
        completion["parameter_counts"][dataset] = {}
        for condition in CONDITIONS:
            weight, intra, cross, variant = parse_condition(condition)
            meta = condition_metadata(condition)
            parameters = shape["features"] * config["backbone"]["hidden_dim"] + config["backbone"]["hidden_dim"] * shape["classes"] + int(variant == "learned")
            completion["parameter_counts"][dataset][condition] = parameters
            choices.append({"dataset": dataset, "condition": condition, **meta,
                            "selected_lr": config["training"]["learning_rate_candidates"][0],
                            "mean_tuning_validation_ce": .9,
                            "selection_scope": "validation_only_independent_tuning_seeds"})
            proofs.append({"dataset": dataset, "condition": condition, **meta, "seeds": list(seeds),
                           "before_sha256": "a" * 64, "after_sha256": "a" * 64,
                           "parameters_preserved": True, "optimizer_updates": 0})
            for index, seed in enumerate(seeds):
                base = {"dataset": dataset, "condition": condition, **meta, "mode": weight,
                        "seed": seed, "model_state_sha256": "a" * 64}
                rho = 0 if variant == "off" else 1 if variant == "fixed" else .4 + .1 * index
                theta = math.log(rho / (1 - rho)) if variant == "learned" else None
                acc = .62 + .007 * index + {"off": 0, "fixed": .03 + .013 * index, "learned": .02}[variant]
                acc += .004 * (intra == "local") + .003 * (cross == "edge") - .002 * (weight == "local_degree")
                ce = 1 + .005 * index - {"off": 0, "fixed": .04, "learned": .03}[variant]
                ce -= .006 * (intra == "local") + .002 * (cross == "edge")
                for split in ("train", "validation", "test"):
                    row = {**base, "split": split, "accuracy": acc, "ce": ce,
                           "num_nodes": shape["nodes"], "num_labeled_nodes": shape[split]}
                    metrics.append(row)
                    for treatment, target in intervention_variants(condition):
                        noop = variant == "fixed" and treatment == "gain1"
                        probes.append({**row, "intervention": treatment, "target": target,
                                       "no_op_expected": noop, "accuracy": acc + (0 if noop else -.002),
                                       "ce": ce + (0 if noop else .006)})
                for layer in (0, 1):
                    for treatment, target in (("original", "none"), *intervention_variants(condition)):
                        active = treatment != "original" and (target == "both" or target == f"layer_{layer}")
                        effective = int(treatment == "gain1") if active else rho
                        branches.append({
                            **base, "layer": layer, "intervention": treatment, "target": target,
                            "no_op_expected": variant == "fixed" and treatment == "gain1",
                            "parameters_per_seed": parameters, "trainable_parameters_per_seed": parameters,
                            "projected_norm": 3., "projected_nonzero": True,
                            "output_norm": 2., "off_norm": 2., "off_nonzero": True,
                            "matched_delta_norm": .1 * effective, "matched_delta_relative": .05 * effective,
                            "predicted_delta_norm": .1 * effective, "formula_residual_norm": 3e-8,
                            "formula_relative_error": 1e-8,
                            "context_norm": .1, "raw_context_norm": 1.,
                            "cross_energy_before": .2, "cross_energy_after": .2 - .01 * effective,
                            "intra_energy_before": .3, "intra_energy_after": .25,
                            "s_degree_max": .5, "g_degree_max": .5, "eta_max": .25,
                            "rho": float(effective), "rho_original": float(rho),
                            "theta": theta, "theta_available": variant == "learned",
                        })
    evaluated = dict(metric_rows=metrics, intervention_rows=probes, branch_rows=branches, provenance=proofs)
    resources = [dict(status="measured", scope="training", seconds_per_epoch=.025,
                      packed_runs=2, peak_vram_bytes=1024)]
    return config, evaluated, choices, resources, completion


def test_twenty_unique_conditions_include_six_legacy_baselines_and_no_duplicate_off():
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 20
    assert sum(legacy_baseline(c) for c in CONDITIONS) == 6
    assert sum(parse_condition(c)[3] == "off" for c in CONDITIONS) == 4
    assert all(parse_condition(c)[2] == "none" for c in CONDITIONS if c.endswith("__off"))
    assert not intervention_variants("unit__local__none__off")
    assert len(intervention_variants("unit__local__edge__learned")) == 6


def test_all_52_direct_contrasts_and_12_interactions_preserve_isolated_factors():
    pairs = comparisons()
    assert len(pairs) == len(set(pairs)) == 52
    assert Counter(scope for _, _, scope in pairs) == {
        "cross_fixed_minus_off": 8, "cross_learned_minus_off": 8,
        "gain_learned_minus_fixed": 8, "cross_edge_minus_graph": 8,
        "intra_local_minus_graph": 10, "between_C_exploratory": 10,
    }
    for condition, reference, scope in pairs:
        a, b = parse_condition(condition), parse_condition(reference)
        if scope == "cross_edge_minus_graph":
            assert a[:2] == b[:2] and a[3] == b[3] and (a[2], b[2]) == ("edge", "graph")
        elif scope == "intra_local_minus_graph":
            assert a[0] == b[0] and a[2:] == b[2:] and (a[1], b[1]) == ("local", "graph")
        elif scope == "between_C_exploratory":
            assert a[1:] == b[1:] and (a[0], b[0]) == ("local_degree", "unit")
        else:
            assert a[:2] == b[:2]
    terms = interaction_contrasts()
    assert len(terms) == len({r["contrast_id"] for r in terms}) == 12
    assert Counter(r["comparison_scope"] for r in terms) == {
        "intra_change_in_cross_minus_off": 8, "intra_change_in_edge_minus_graph": 4,
    }
    assert all(len({c for c, _ in r["terms"]}) == 4 and sum(k for _, k in r["terms"]) == 0 for r in terms)


def test_full_debug_training_budgets_and_complete_derived_coverage():
    full = read_config(profile="full")
    assert [full["training"][k] for k in ("tuning_runs", "final_runs", "total_runs", "total_updates")] == [540, 300, 840, 420000]
    config, evaluated, choices, resources, completion = evidence()
    checked = validate_rows(config, evaluated, choices, resources, completion)
    assert checked["total_updates"] == 1080 and checked["total_runs"] == 360
    assert (checked["primary_metric_rows"], checked["intervention_metric_rows"], checked["branch_rows"]) == (360, 1728, 1392)
    assert len(metric_estimates(evaluated["metric_rows"])) == 180
    assert len(paired_comparisons(evaluated["metric_rows"])) == 936
    assert len(interaction_comparisons(evaluated["metric_rows"])) == 216
    assert len(intervention_changes(evaluated["metric_rows"], evaluated["intervention_rows"])) == 1728
    assert len(branch_estimates(evaluated["branch_rows"])) == 696


def test_paired_accuracy_units_and_covariance_are_computed_before_averaging():
    config, evaluated, *_ = evidence()
    row = next(r for r in paired_comparisons(evaluated["metric_rows"])
               if r["dataset"] == config["data"]["datasets"][0]
               and r["condition"] == "unit__graph__graph__fixed" and r["reference"] == "unit__graph__none__off"
               and r["metric"] == "accuracy" and r["split"] == "test")
    np.testing.assert_allclose(row["mean"], 3.65, atol=1e-12)
    np.testing.assert_allclose(row["std"], np.std([3, 4.3], ddof=1), atol=1e-12)
    assert row["unit"] == "percentage_points" and row["count"] == 2


@pytest.mark.parametrize("scope", ["intra_change_in_cross_minus_off", "intra_change_in_edge_minus_graph"])
def test_four_condition_interactions_keep_seed_covariance_and_signed_terms(scope):
    config, evaluated, *_ = evidence()
    contrast = next(r for r in interaction_contrasts() if r["comparison_scope"] == scope)
    dataset = config["data"]["datasets"][0]
    # Large common seed shifts cancel in the contrast. Remaining seed effects
    # deliberately differ, so subtracting marginal intervals would be wrong.
    measurements = ([10., 30.], [8., 20.], [5., 14.], [4., 10.])
    for index, (condition, _) in enumerate(contrast["terms"]):
        for row in evaluated["metric_rows"]:
            if row["condition"] == condition and row["dataset"] == dataset:
                row["ce"] = measurements[index][config["training"]["final_seeds"].index(row["seed"])]
    result = next(r for r in interaction_comparisons(evaluated["metric_rows"])
                  if r["dataset"] == dataset and r["contrast_id"] == contrast["contrast_id"]
                  and r["split"] == "test" and r["metric"] == "ce")
    assert result["mean"] == 3.5
    np.testing.assert_allclose(result["std"], np.std([1., 6.], ddof=1), atol=1e-12)
    assert result["unit"] == "loss" and result["count"] == 2
    assert result["contrast_expression"].count("+") == result["contrast_expression"].count("-") == 2


def test_same_checkpoint_frozen_differences_retain_fixed_noop_controls():
    _, evaluated, *_ = evidence()
    changed = intervention_changes(evaluated["metric_rows"], evaluated["intervention_rows"])
    controls = [r for r in changed if r["no_op_expected"]]
    assert len(controls) == 432
    assert all(r["mean"] == r["std"] == r["lower"] == r["upper"] == 0 for r in controls)
    assert all(r["condition"].endswith("__fixed") and r["intervention"] == "gain1" for r in controls)
    assert all(math.isclose(r["mean"], .006, abs_tol=1e-12) for r in changed if r["intervention"] == "gain0" and r["metric"] == "ce")


def test_applied_energy_reference_and_null_ratios_are_not_imputed():
    config, evaluated, choices, resources, completion = evidence()
    row = evaluated["branch_rows"][0]
    row.update(off_norm=0., off_nonzero=False, matched_delta_relative=None,
               projected_norm=0., projected_nonzero=False, formula_relative_error=None)
    validate_rows(config, evaluated, choices, resources, completion)
    group = next(r for r in branch_estimates(evaluated["branch_rows"])
                 if (r["dataset"], r["condition"], r["layer"], r["intervention"], r["target"])
                 == tuple(row[k] for k in ("dataset", "condition", "layer", "intervention", "target")))
    assert group["matched_delta_relative_count"] == group["formula_relative_error_count"] == 1
    assert group["off_nonzero_seeds"] == 1 and group["theta_count"] == 0 and group["theta_mean"] is None
    assert group["energy_operator"] == "applied_S_G" and group["cross_diagnostic_policy"] == "graph"


@pytest.mark.parametrize("table", ["metric_rows", "intervention_rows", "branch_rows", "provenance"])
def test_incomplete_evidence_is_rejected(table):
    config, evaluated, choices, resources, completion = evidence()
    evaluated[table].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_rows(config, evaluated, choices, resources, completion)


@pytest.mark.parametrize("case", [
    "duplicate", "metric_nan", "accuracy_percent", "split_count", "wrong_state", "negative_energy",
    "missing_theta", "gain_scope", "ratio", "no_op", "actual_data", "config_hash", "source_hash",
    "budget", "counts", "proof_hash", "selection_test", "selection_lr", "no_resource",
    "weight", "intra", "cross", "energy_operator", "off_reference", "formula_ratio", "degree_bound",
])
def test_corrupt_contract_metadata_units_and_provenance_fail(case):
    config, evaluated, choices, resources, completion = evidence()
    metric, branch, probe = evaluated["metric_rows"][0], evaluated["branch_rows"][0], evaluated["intervention_rows"][0]
    if case == "duplicate":
        evaluated["metric_rows"].append(copy.deepcopy(metric))
    elif case == "metric_nan":
        metric["ce"] = float("nan")
    elif case == "accuracy_percent":
        metric["accuracy"] = 62
    elif case == "split_count":
        metric["num_labeled_nodes"] = 1
    elif case == "wrong_state":
        probe["model_state_sha256"] = "f" * 64
    elif case == "negative_energy":
        branch["cross_energy_before"] = -.1
    elif case == "missing_theta":
        next(r for r in evaluated["branch_rows"] if r["variant"] == "learned")["theta"] = None
    elif case == "gain_scope":
        next(r for r in evaluated["branch_rows"] if r["intervention"] == "gain0")["rho"] = .5
    elif case == "ratio":
        branch["matched_delta_relative"] = .8
    elif case == "no_op":
        probe["no_op_expected"] = True
    elif case == "actual_data":
        completion["actual_data"] = True
    elif case == "config_hash":
        completion["config_digest"] = "f" * 64
    elif case == "source_hash":
        completion["source"]["sha256"] = {"broken": "short"}
        completion["source"]["code_digest"] = digest(completion["source"]["sha256"])
    elif case == "budget":
        completion["coverage"]["total_runs"] -= 1
    elif case == "counts":
        completion["parameter_counts"][config["data"]["datasets"][0]][CONDITIONS[0]] -= 1
    elif case == "proof_hash":
        evaluated["provenance"][0]["after_sha256"] = "f" * 64
    elif case == "selection_test":
        choices[0]["selection_scope"] = "test"
    elif case == "selection_lr":
        choices[0]["selected_lr"] = .5
    elif case == "no_resource":
        resources.clear()
    elif case == "weight":
        metric["weight_mode"] = "local_degree"
    elif case == "intra":
        metric["intra_policy"] = "local"
    elif case == "cross":
        metric["cross_policy"] = "edge"
    elif case == "energy_operator":
        branch["energy_operator"] = "raw_A_K"
    elif case == "off_reference":
        branch["cross_diagnostic_policy"] = "edge"
    elif case == "formula_ratio":
        branch["formula_relative_error"] = .01
    elif case == "degree_bound":
        branch["g_degree_max"] = .8
    with pytest.raises(ValueError):
        validate_rows(config, evaluated, choices, resources, completion)


def test_resource_failures_and_reuse_keep_undefined_timing_explicit():
    config, evaluated, choices, resources, completion = evidence()
    resources += [dict(status="OOM", seconds_per_epoch=None), dict(status="reused_complete", seconds_per_epoch=None)]
    validate_rows(config, evaluated, choices, resources, completion)


def test_report_writes_all_declared_debug_artifacts_and_refuses_overwrite(tmp_path):
    config, evaluated, choices, resources, completion = evidence()
    checks = write_report(tmp_path, config, evaluated, choices, resources, completion)
    assert checks["complete_all_declared_rows"] and checks["paired_comparison_rows"] == 936
    assert checks["interaction_comparison_rows"] == 216
    assert all((tmp_path / name).is_file() for name in ARTIFACTS)
    summary = (tmp_path / "LOCAL_CONTEXT_NORMALIZATION_SUMMARY.md").read_text(encoding="utf-8")
    for token in ("DEBUG", "1,080", "52개", "12개", "applied S/G", "같은 final seed", "D metric", "Star", "undefined", "다중 비교"):
        assert token in summary
    saved = (tmp_path / "metric_estimates.csv").read_bytes()
    with pytest.raises(FileExistsError):
        write_report(tmp_path, config, evaluated, choices, resources, completion)
    assert (tmp_path / "metric_estimates.csv").read_bytes() == saved
    assert set(BRANCH_METRICS) >= {"rho", "theta", "raw_context_norm", "context_norm", "formula_residual_norm"}
