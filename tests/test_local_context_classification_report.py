"""DEBUG report fixtures: complete pairing, units, nulls and provenance checks.

The explicit metric fixture below tests reporting arithmetic only. It is not
real dataset evidence, model predictions or a completed classification run.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from research.local_context_coupling.classification.common import read_config
from research.local_context_coupling.classification.model import CONDITIONS, parse_condition
from research.local_context_coupling.classification.report import (
    ARTIFACTS, BRANCH_METRICS, branch_estimates, comparisons, estimate,
    intervention_changes, intervention_variants, metric_estimates,
    paired_comparisons, validate_rows, write_report,
)
from research.wedge_propagation.classification.common import digest


def evidence():
    config = read_config(profile="debug")
    seeds, datasets = config["training"]["final_seeds"], config["data"]["datasets"]
    hashes = {"DEBUG-report-fixture.py": "b" * 64}
    completion = {
        "profile": "debug", "actual_data": False, "config_digest": digest(config),
        "source": {"sha256": hashes, "code_digest": digest(hashes)},
        "data_manifest_digest": "c" * 64, "parameter_counts": {},
        "coverage": {"tuning_runs": 72, "final_runs": 36, "total_runs": 108,
                     "contract_optimizer_updates": 324, "all_datasets_conditions_seeds_splits": True},
    }
    metrics, interventions, branches, selections, proofs = [], [], [], [], []
    for dataset in datasets:
        completion["parameter_counts"][dataset] = {}
        shape = config["data"]["expected_shapes"][dataset]
        for condition in CONDITIONS:
            mode, variant = parse_condition(condition)
            parameters = shape["features"] * config["backbone"]["hidden_dim"] + config["backbone"]["hidden_dim"] * shape["classes"] + int(variant == "learned")
            completion["parameter_counts"][dataset][condition] = parameters
            selections.append({"dataset": dataset, "condition": condition, "variant": variant,
                               "selected_lr": config["training"]["learning_rate_candidates"][0],
                               "mean_tuning_validation_ce": .9, "selection_scope": "validation_only_independent_tuning_seeds"})
            proofs.append({"dataset": dataset, "condition": condition, "seeds": list(seeds),
                           "before_sha256": "a" * 64, "after_sha256": "a" * 64,
                           "parameters_preserved": True, "optimizer_updates": 0})
            for index, seed in enumerate(seeds):
                base = {"dataset": dataset, "condition": condition, "weight_mode": mode,
                        "variant": variant, "seed": seed, "model_state_sha256": "a" * 64}
                rho = 0 if variant == "off" else 1 if variant == "fixed" else .4 + .1 * index
                theta = math.log(rho / (1 - rho)) if variant == "learned" else None
                for split in ("train", "validation", "test"):
                    score = {"off": .62 + .007 * index, "fixed": .65 + .02 * index, "learned": .64 + .011 * index}[variant]
                    row = {**base, "split": split, "accuracy": score - .005 * (mode == "local_degree"),
                           "ce": 1 - {"off": 0, "fixed": .04, "learned": .03}[variant] + .005 * index,
                           "num_nodes": shape["nodes"], "num_labeled_nodes": shape[split]}
                    metrics.append(row)
                    for treatment, target in intervention_variants(condition):
                        noop = variant == "fixed" and treatment == "gain1"
                        interventions.append({**row, "intervention": treatment, "target": target,
                                              "no_op_expected": noop,
                                              "accuracy": row["accuracy"] + (0 if noop else -.002),
                                              "ce": row["ce"] + (0 if noop else .006)})
                for layer in (0, 1):
                    for treatment, target in (("original", "none"), *intervention_variants(condition)):
                        active = treatment != "original" and (target == "both" or target == f"layer_{layer}")
                        effective = int(treatment == "gain1") if active else rho
                        branches.append({
                            **base, "layer": layer, "intervention": treatment, "target": target,
                            "no_op_expected": variant == "fixed" and treatment == "gain1",
                            "parameters_per_seed": parameters, "trainable_parameters_per_seed": parameters,
                            "projected_norm": 3., "output_norm": 2., "off_norm": 2.,
                            "matched_delta_norm": .1 * effective, "matched_delta_relative": .05 * effective,
                            "off_nonzero": True, "context_norm": 1.,
                            "cross_energy_before": 2., "cross_energy_after": 2. - .1 * effective,
                            "intra_energy_before": 3., "intra_energy_after": 2.5,
                            "rho": float(effective), "rho_original": float(rho),
                            "theta": theta, "theta_available": variant == "learned",
                        })
    evaluated = {"metric_rows": metrics, "intervention_rows": interventions,
                 "branch_rows": branches, "provenance": proofs}
    resources = [{"status": "measured", "scope": "training", "seconds_per_epoch": .025,
                  "packed_runs": 2, "peak_vram_bytes": 1024}]
    return config, evaluated, selections, resources, completion


def test_estimate_uses_sample_std_and_real_paired_seed_counts():
    result = estimate([1, 2, 3])
    assert result["mean"] == 2 and result["std"] == 1 and result["count"] == 3
    assert result["lower"] < 2 < result["upper"]
    assert estimate([])["mean"] is None
    assert estimate([3])["lower"] is None
    for value in (None, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="finite"):
            estimate([value])


def test_complete_debug_evidence_and_derived_table_sizes():
    config, evaluated, selections, resources, completion = evidence()
    checked = validate_rows(config, evaluated, selections, resources, completion)
    assert checked["total_updates"] == 324
    assert (checked["primary_metric_rows"], checked["intervention_metric_rows"], checked["branch_rows"]) == (108, 432, 360)
    assert len(metric_estimates(evaluated["metric_rows"])) == 54
    assert len(paired_comparisons(evaluated["metric_rows"])) == 162
    assert len(intervention_changes(evaluated["metric_rows"], evaluated["intervention_rows"])) == 432
    assert len(branch_estimates(evaluated["branch_rows"])) == 180
    assert len(comparisons()) == 9
    assert not intervention_variants("unit__off")
    assert len(intervention_variants("unit__fixed")) == 6


def test_paired_accuracy_conversion_and_covariance_are_correct():
    config, evaluated, *_ = evidence()
    dataset = config["data"]["datasets"][0]
    row = next(r for r in paired_comparisons(evaluated["metric_rows"])
               if r["dataset"] == dataset and r["condition"] == "unit__fixed" and r["reference"] == "unit__off"
               and r["split"] == "test" and r["metric"] == "accuracy")
    # The two same-seed differences are 3 and 4.3 percentage points.
    np.testing.assert_allclose(row["mean"], 3.65, atol=1e-12)
    np.testing.assert_allclose(row["std"], np.std([3, 4.3], ddof=1), atol=1e-12)
    assert row["unit"] == "percentage_points" and row["count"] == 2
    assert row["comparison_scope"] == "within_weight_mode"
    assert len([r for r in paired_comparisons(evaluated["metric_rows"])
                if r["comparison_scope"] == "between_weight_modes_exploratory"]) == 54


def test_frozen_differences_are_same_checkpoint_and_keep_noop_controls():
    _, evaluated, *_ = evidence()
    changed = intervention_changes(evaluated["metric_rows"], evaluated["intervention_rows"])
    controls = [row for row in changed if row["no_op_expected"]]
    assert len(controls) == 108
    assert all(row["mean"] == row["std"] == row["lower"] == row["upper"] == 0 for row in controls)
    assert all(row["condition"].endswith("__fixed") and row["intervention"] == "gain1" for row in controls)
    gain0ce = [row for row in changed if row["intervention"] == "gain0" and row["metric"] == "ce"]
    assert all(math.isclose(row["mean"], .006, abs_tol=1e-12) for row in gain0ce)
    assert all(row["comparison_scope"].startswith("frozen_selected_checkpoint") for row in changed)


def test_null_branch_ratio_and_theta_remain_undefined():
    config, evaluated, selections, resources, completion = evidence()
    row = evaluated["branch_rows"][0]
    row.update(off_norm=0., off_nonzero=False, matched_delta_relative=None)
    validate_rows(config, evaluated, selections, resources, completion)
    result = branch_estimates(evaluated["branch_rows"])
    group = next(r for r in result if r["dataset"] == row["dataset"] and r["condition"] == row["condition"]
                 and r["layer"] == row["layer"] and r["intervention"] == row["intervention"] and r["target"] == row["target"])
    assert group["matched_delta_relative_count"] == 1 and group["off_nonzero_seeds"] == 1
    assert group["theta_mean"] is None and group["theta_count"] == 0


@pytest.mark.parametrize("table", ["metric_rows", "intervention_rows", "branch_rows", "provenance"])
def test_missing_evidence_is_rejected(table):
    config, evaluated, selections, resources, completion = evidence()
    evaluated[table].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_rows(config, evaluated, selections, resources, completion)


@pytest.mark.parametrize("case", ["duplicate", "metric_nan", "accuracy_percent", "split_count", "wrong_state",
                                  "branch_negative", "missing_theta", "bad_gain_scope", "bad_ratio", "bad_noop",
                                  "actual_data", "config_hash", "source_hash", "budget", "counts", "proof_change",
                                  "selection_test", "selection_lr", "missing_resource"])
def test_corrupt_or_mislabelled_evidence_fails(case):
    config, evaluated, selections, resources, completion = evidence()
    if case == "duplicate":
        evaluated["metric_rows"].append(copy.deepcopy(evaluated["metric_rows"][0]))
    elif case == "metric_nan":
        evaluated["metric_rows"][0]["ce"] = float("nan")
    elif case == "accuracy_percent":
        evaluated["metric_rows"][0]["accuracy"] = 62
    elif case == "split_count":
        evaluated["metric_rows"][0]["num_labeled_nodes"] = 1
    elif case == "wrong_state":
        evaluated["intervention_rows"][0]["model_state_sha256"] = "f" * 64
    elif case == "branch_negative":
        evaluated["branch_rows"][0]["cross_energy_before"] = -.1
    elif case == "missing_theta":
        next(r for r in evaluated["branch_rows"] if r["variant"] == "learned")["theta"] = None
    elif case == "bad_gain_scope":
        next(r for r in evaluated["branch_rows"] if r["intervention"] == "gain0")["rho"] = .5
    elif case == "bad_ratio":
        evaluated["branch_rows"][0]["matched_delta_relative"] = .8
    elif case == "bad_noop":
        evaluated["intervention_rows"][0]["no_op_expected"] = True
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
        completion["parameter_counts"][config["data"]["datasets"][0]]["unit__off"] -= 1
    elif case == "proof_change":
        evaluated["provenance"][0]["after_sha256"] = "f" * 64
    elif case == "selection_test":
        selections[0]["selection_scope"] = "test"
    elif case == "selection_lr":
        selections[0]["selected_lr"] = .5
    elif case == "missing_resource":
        resources.clear()
    with pytest.raises(ValueError):
        validate_rows(config, evaluated, selections, resources, completion)


def test_resource_failures_are_recorded_without_inventing_timing():
    config, evaluated, selections, resources, completion = evidence()
    resources.extend([{"status": "OOM", "seconds_per_epoch": None, "peak_vram_bytes": None},
                      {"status": "memory_safety_rejected", "seconds_per_epoch": .1}])
    validate_rows(config, evaluated, selections, resources, completion)


def test_full_contract_requires_all_three_real_dataset_names():
    config, evaluated, selections, resources, completion = evidence()
    config["profile"] = "full"
    with pytest.raises(ValueError, match="FULL requires"):
        validate_rows(config, evaluated, selections, resources, completion)


def test_report_writes_complete_debug_artifacts_with_scope_and_no_overwrite(tmp_path):
    config, evaluated, selections, resources, completion = evidence()
    checks = write_report(tmp_path, config, evaluated, selections, resources, completion)
    assert checks["complete_all_declared_rows"] is True
    assert all((tmp_path / path).is_file() for path in ARTIFACTS)
    summary = (tmp_path / "LOCAL_CONTEXT_CLASSIFICATION_SUMMARY.md").read_text(encoding="utf-8")
    for token in ("DEBUG", "324", "validation만", "Scalar E/J", "같은 final seed", "고정 checkpoint", "다중 비교", "L2를 제외", "undefined"):
        assert token in summary
    protected = (tmp_path / "metric_estimates.csv").read_bytes()
    with pytest.raises(FileExistsError):
        write_report(tmp_path, config, evaluated, selections, resources, completion)
    assert (tmp_path / "metric_estimates.csv").read_bytes() == protected
    assert set(BRANCH_METRICS) >= {"rho", "theta", "matched_delta_norm", "cross_energy_before"}
