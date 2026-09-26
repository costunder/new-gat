"""CPU-only control-plane arithmetic and metadata checks; no model execution."""

import copy

import pytest

from experiments.aggregation_comparison.effects import contrast_report, markdown


def job(arm, score, seed=0):
    return {
        "profile": "reference",
        "dataset": "ppi",
        "model_seed": seed,
        "variant_id": arm,
        "status": "passed",
        "audit": {"status": "passed"},
        "result": {
            "validation": score,
            "total_parameters": 100 if arm == "incidence" else 200,
            "data_sha256": "a" * 64,
            "split_sha256": {"train": "b" * 64},
            "source_sha256": {"model.py": "c" * 64},
            "learning_budget": {"epochs": 200},
            "shared_initial_state_sha256": "d" * 64,
            "comparison_configuration": {
                "ablation_arm": arm,
                "lr": 0.0005,
                "comparison_contract": {"conductance": arm},
            },
        },
    }


def available(jobs):
    return {
        row["effect"]: row
        for group in contrast_report(jobs)["groups"]
        for row in group["effects"]
        if row["status"] == "available"
    }


def test_conditional_differences_and_interaction_signs():
    rows = [
        job("incidence", 0.80),
        job("incidence_energy", 0.82),
        job("incidence_pre_lift", 0.81),
        job("incidence_energy_pre_lift", 0.86),
    ]
    effects = available(rows)
    assert effects["all_energy_vs_none_no_lift"]["delta_pp"] == pytest.approx(2)
    assert effects["all_energy_vs_none_pre_lift"]["delta_pp"] == pytest.approx(5)
    assert effects["interaction_all_energy_pre_lift"]["delta_pp"] == pytest.approx(3)
    assert effects["all_energy_vs_none_no_lift"]["parameters"] == {
        "incidence_energy": 200,
        "incidence": 100,
    }
    assert "+3.000000" in "\n".join(markdown(contrast_report(rows)))


def test_c_regime_effect_and_energy_interaction():
    effects = available(
        [
            job("incidence_fixed", 0.7),
            job("incidence_fixed_energy", 0.72),
            job("incidence_shared", 0.75),
            job("incidence_shared_energy", 0.78),
            job("incidence", 0.8),
            job("incidence_energy", 0.84),
        ]
    )
    assert effects["energy_effect_c_shared"]["delta_pp"] == pytest.approx(3)
    assert effects["c_per_head_vs_fixed_energy"]["delta_pp"] == pytest.approx(12)
    assert effects["interaction_energy_c_per_head_vs_fixed"]["delta_pp"] == pytest.approx(2)


def test_missing_audit_and_different_seeds_never_fill_cells():
    rows = [job("incidence", 0.8), job("incidence_energy", 0.9, seed=1)]
    assert not available(rows)
    rows[1]["model_seed"] = 0
    rows[1]["audit"]["status"] = "pending"
    assert not available(rows)
    rows[1]["audit"]["status"] = "passed"
    assert available(rows)


@pytest.mark.parametrize(
    "field",
    [
        "data_sha256",
        "split_sha256",
        "learning_budget",
        "source_sha256",
        "shared_initial_state_sha256",
        "comparison_configuration",
    ],
)
def test_mismatched_recipe_or_evidence_is_rejected(field):
    rows = [job("incidence", 0.8), job("incidence_energy", 0.9)]
    if field == "comparison_configuration":
        rows[1]["result"][field]["lr"] = 0.001
    else:
        rows[1]["result"][field] = "different"
    with pytest.raises(ValueError, match="unmatched"):
        available(rows)


def test_duplicate_cell_rejected():
    row = job("incidence", 0.8)
    with pytest.raises(ValueError, match="duplicate"):
        contrast_report([row, copy.deepcopy(row)])


def test_actual_cli_recipes_form_complete_19_arm_matrix():
    from pathlib import Path

    from experiments.aggregation_comparison import engine, runner
    from experiments.aggregation_comparison.model import ARMS, conductance_contract

    args = runner.parser().parse_args(
        [
            "--run-id",
            "debug-control-plan",
            "--datasets",
            "ppi",
            "--profiles",
            "reference",
            "--hardware-profile",
            "portable",
        ]
    )
    jobs = runner.make_jobs(args, Path("results/debug-control-plan"))
    assert len(jobs) == len(ARMS) == 19
    records = []
    for planned in jobs:
        child = engine.build_parser().parse_args(
            planned["command"][planned["command"].index("-m") + 2 :]
        )
        engine.validate_args(child)
        config = engine.configuration(child)
        assert (child.layers, child.hidden_channels, child.heads, child.epochs) == (8, 256, 8, 200)
        regime = conductance_contract(child.ablation_arm)
        if regime is not None:
            assert config["conductance_mode"] == regime["conductance_mode"]
            assert config["conductance_heads"] == regime["conductance_heads"]
        record = job(child.ablation_arm, 0.8)
        record["result"]["comparison_configuration"] = config
        records.append(record)
    report = contrast_report(records)
    assert len(report["groups"]) == 1
    assert all(row["status"] == "available" for row in report["groups"][0]["effects"])


def test_calibration_requires_diagnostic_memory_probe(monkeypatch):
    from experiments.aggregation_comparison import calibration

    monkeypatch.setattr(calibration.resources, "measurement_is_safe", lambda report: True)
    monkeypatch.setattr(
        calibration.resources,
        "projected_training_budget_cost",
        lambda report, policy: {
            "projected_training_seconds": 100,
            "learning_budget": {"planned_epochs": 200},
        },
    )
    report = {
        "status": "passed",
        "validation_completed": True,
        "validation_seconds": 2,
        "topology_preparation_seconds": 1,
        "setup_seconds": 3,
    }
    with pytest.raises(ValueError, match="mechanism audit"):
        calibration._full_budget_seconds(report, {})
    report.update(mechanism_audit_completed=True, mechanism_audit_seconds=12)
    assert calibration._full_budget_seconds(report, {}) == 515
