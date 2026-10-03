"""DEBUG-only report fixtures exercise coverage, pairing and branch evidence."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from research.local_energy_relations.prediction.report import (
    ARTIFACTS,
    BRANCH_METRICS,
    CONDITIONS,
    branch_estimates,
    estimate,
    interaction_estimates,
    intervention_changes,
    intervention_variants,
    paired_comparisons,
    validate_rows,
    write_report,
)


def evidence():
    datasets, seeds = ("Cora", "CiteSeer", "PubMed"), (11, 23)
    shapes = {
        name: {"nodes": 9, "features": 6, "classes": 3, "train": 3, "validation": 3, "test": 3}
        for name in datasets
    }
    tuning = len(datasets) * len(CONDITIONS) * 3 * 2
    final = len(datasets) * len(CONDITIONS) * len(seeds)
    config = {
        "profile": "debug",
        "conditions": list(CONDITIONS),
        "data": {"datasets": list(datasets), "expected_shapes": shapes},
        "backbone": {"hidden_dim": 64},
        "training": {
            "learning_rate_candidates": [0.001, 0.003, 0.01],
            "tuning_seeds": [101, 202],
            "final_seeds": list(seeds),
            "epochs_per_run": 4,
            "tuning_runs": tuning,
            "final_runs": final,
            "total_runs": tuning + final,
            "total_updates": 4 * (tuning + final),
        },
    }
    contract = {
        "actual_data": False,
        "profile": "debug",
        "coverage": {
            "tuning_runs": tuning,
            "final_runs": final,
            "total_runs": tuning + final,
            "contract_optimizer_updates": 4 * (tuning + final),
            "all_datasets_conditions_seeds_splits": True,
        },
        "parameter_counts": {
            dataset: {
                condition: 578
                + {"base": 0, "within": 1, "between": 1, "both": 2}[condition.split("__")[1]] * 67
                for condition in CONDITIONS
            }
            for dataset in datasets
        },
    }
    metrics, interventions, selections, branches = [], [], [], []
    for dataset in datasets:
        for condition in CONDITIONS:
            weight, variant = condition.split("__")
            selections.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "selected_lr": 0.003,
                    "mean_tuning_validation_ce": 0.9,
                }
            )
            for index, seed in enumerate(seeds):
                identity = {
                    "dataset": dataset,
                    "condition": condition,
                    "weight_mode": weight,
                    "variant": variant,
                    "seed": seed,
                }
                for split in ("train", "validation", "test"):
                    delta = {"base": 0, "within": 0.01, "between": 0.02, "both": 0.03}[variant]
                    metric = identity | {
                        "split": split,
                        "ce": 0.9 - delta + index * 0.004,
                        "accuracy": 0.6 + delta + index * 0.02,
                        "num_nodes": 9,
                        "num_labeled_nodes": 3,
                    }
                    metrics.append(metric)
                    for treatment, target in intervention_variants(condition):
                        interventions.append(
                            metric
                            | {
                                "intervention": treatment,
                                "target": target,
                                "ce": metric["ce"] + 0.01,
                                "accuracy": metric["accuracy"] - 0.01,
                                "model_state_sha256": "a" * 64,
                            }
                        )
                for layer in (0, 1):
                    original = identity | {
                        "layer": layer,
                        "intervention": "original",
                        "target": "none",
                        "alpha": 0.5,
                        "projected_norm": 1.2,
                        "base_norm": 1.0,
                        "energy_feature_norm": 0.02,
                        "relation_feature_norm": 0.03,
                        "energy_branch_norm": 0.04 if variant in ("within", "both") else 0.0,
                        "relation_branch_norm": 0.05 if variant in ("between", "both") else 0.0,
                        "energy_lift_norm": 0.2 if variant in ("within", "both") else 0.0,
                        "relation_lift_norm": 0.3 if variant in ("between", "both") else 0.0,
                        "energy_raw_mean": 0.03,
                        "relation_raw_mean": -0.02,
                        "relation_negative_fraction": 0.7,
                        "base_nonzero": True,
                    }
                    original["energy_to_base"] = original["energy_branch_norm"]
                    original["relation_to_base"] = original["relation_branch_norm"]
                    branches.append(original)
                    for treatment, target in intervention_variants(condition):
                        changed = original | {"intervention": treatment, "target": target}
                        if target == "both" or target == f"layer_{layer}":
                            if treatment in ("within_remove", "both_remove"):
                                changed["energy_branch_norm"] = changed["energy_to_base"] = 0.0
                            if treatment in ("between_remove", "both_remove"):
                                changed["relation_branch_norm"] = changed["relation_to_base"] = 0.0
                        branches.append(changed)
    resources = [
        {
            "status": "measured",
            "seconds_per_epoch": 0.025,
            "packed_runs": 2,
            "peak_vram_bytes": 1024,
        }
    ]
    return config, metrics, interventions, selections, resources, branches, contract


def test_estimate_uses_all_observations_and_sample_std():
    result = estimate([1.0, 2.0, 3.0])
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["std"] == 1.0
    assert result["lower"] < 2 < result["upper"]
    assert estimate([1.0])["lower"] is None
    assert estimate([])["mean"] is None
    with pytest.raises(ValueError, match="finite"):
        estimate([np.nan])


def test_complete_fixture_is_valid_and_pairs_are_actual_same_seed_differences():
    config, metrics, interventions, selections, resources, branches, contract = evidence()
    validate_rows(config, metrics, interventions, selections, resources, branches, contract)
    paired = paired_comparisons(metrics)
    assert len(paired) == 3 * 14 * 3 * 2
    row = next(
        row
        for row in paired
        if (row["dataset"], row["first"], row["second"], row["split"], row["metric"])
        == ("Cora", "unit__both", "unit__base", "test", "accuracy")
    )
    assert row["mean"] == pytest.approx(0.03)
    assert row["count"] == 2
    changes = intervention_changes(interventions, metrics)
    assert all(
        row["mean"] == pytest.approx(-0.01 if row["metric"] == "accuracy" else 0.01)
        for row in changes
    )
    assert all(row["count"] == 2 for row in changes)


@pytest.mark.parametrize("kind", ["metric", "intervention", "selection", "branch"])
def test_incomplete_and_duplicate_evidence_is_rejected(kind):
    args = list(evidence())
    index = {"metric": 1, "intervention": 2, "selection": 3, "branch": 5}[kind]
    missing = copy.deepcopy(args)
    missing[index].pop()
    with pytest.raises(ValueError, match="incomplete"):
        validate_rows(*missing)
    duplicate = copy.deepcopy(args)
    duplicate[index].append(duplicate[index][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(*duplicate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("accuracy", 1.1),
        ("accuracy", np.nan),
        ("ce", -0.1),
        ("num_nodes", 8),
        ("weight_mode", "wrong"),
    ],
)
def test_invalid_metrics_are_rejected(field, value):
    args = list(evidence())
    args[1][0][field] = value
    with pytest.raises(ValueError):
        validate_rows(*args)


def test_test_selection_or_false_full_claim_is_rejected():
    args = list(evidence())
    args[3][0]["mean_test_ce"] = 0.2
    with pytest.raises(ValueError, match="test"):
        validate_rows(*args)
    args = list(evidence())
    args[-1]["actual_data"] = True
    with pytest.raises(ValueError, match="actual_data"):
        validate_rows(*args)
    args = list(evidence())
    args[-1]["coverage"]["tuning_runs"] -= 1
    with pytest.raises(ValueError, match="training coverage"):
        validate_rows(*args)


def test_frozen_hash_and_removed_branch_evidence_are_checked():
    args = list(evidence())
    args[2][0]["model_state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="state changed"):
        validate_rows(*args)
    args = list(evidence())
    changed = next(
        row for row in args[5] if row["intervention"] == "within_remove" and row["target"] == "both"
    )
    changed["energy_branch_norm"] = changed["energy_to_base"] = 0.1
    with pytest.raises(ValueError, match="removed energy"):
        validate_rows(*args)


def test_undefined_ratios_preserve_no_observation_and_signed_relations():
    args = list(evidence())
    row = args[5][0]
    row.update(base_norm=0.0, base_nonzero=False, energy_to_base=None, relation_to_base=None)
    validate_rows(*args)
    groups = branch_estimates(args[5])
    energy = next(
        value
        for value in groups
        if (value["dataset"], value["condition"], value["layer"], value["metric"])
        == ("Cora", "unit__base", 0, "energy_to_base")
    )
    assert energy["undefined_seeds"] == 1
    assert energy["count"] == 1
    assert any(value["mean"] < 0 for value in groups if value["metric"] == "relation_raw_mean")
    row["energy_to_base"] = 0.0
    with pytest.raises(ValueError, match="base_nonzero"):
        validate_rows(*args)


def test_actual_parameter_count_must_match_active_lifts():
    args = list(evidence())
    args[-1]["parameter_counts"]["Cora"]["unit__base"] += 1
    with pytest.raises(ValueError, match="parameter count"):
        validate_rows(*args)


def test_interaction_uses_same_seed_four_model_contrast_and_all_splits():
    args = evidence()
    for row in args[1]:
        if row["variant"] == "both":
            row["accuracy"] += 0.04
            row["ce"] -= 0.04
    result = interaction_estimates(args[1])
    assert len(result) == 3 * 2 * 3 * 2
    assert all(row["count"] == 2 for row in result)
    assert all(
        row["mean"] == pytest.approx(0.04 if row["metric"] == "accuracy" else -0.04)
        for row in result
    )
    assert {row["split"] for row in result} == {"train", "validation", "test"}
    with pytest.raises(ValueError, match="incomplete"):
        interaction_estimates(args[1][1:])


def test_report_writes_complete_summary_tables_figures_and_refuses_overwrite(tmp_path):
    args = evidence()
    result = write_report(tmp_path, *args)
    assert result["figures"] == 2
    assert result["paired_rows"] == 252
    assert result["interaction_rows"] == 36
    assert result["branch_strength_rows"] == 3 * 8 * 2 * len(BRANCH_METRICS)
    for artifact in ARTIFACTS:
        path = tmp_path / artifact
        assert path.is_file() and path.stat().st_size > 0
    summary = (tmp_path / result["summary"]).read_text(encoding="utf-8")
    assert "DEBUG fixture" in summary
    assert "추가 파라미터" in summary
    assert "C를 학습하지 않았다" in summary
    assert "both−within−between+base" in summary
    assert all(dataset in summary for dataset in ("Cora", "CiteSeer", "PubMed"))
    assert all(condition in summary for condition in CONDITIONS)
    with pytest.raises(FileExistsError):
        write_report(tmp_path, *args)
