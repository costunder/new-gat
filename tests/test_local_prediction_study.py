"""Complete training budgets and validation-only selection guards."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.local_energy_relations.prediction.common import read_config
from research.local_energy_relations.prediction.study import (
    _validate_final_training,
    _workers,
    build_jobs,
    run,
    select_learning_rates,
    verify_coverage,
    verify_frozen_coverage,
)

_FOLDER = Path(__file__).resolve().parents[1] / "research/local_energy_relations/prediction"


def _config(profile="debug"):
    return read_config(_FOLDER / f"config_{profile}.json", profile)


def _tuning_rows(config):
    training = config["training"]
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "lr": lr,
            "seed": seed,
            "phase": "tuning",
            "best_epoch": 1,
            "validation_ce": 1 + lr,
            "validation_accuracy": 0.5,
            "training_epochs": training["epochs_per_run"],
            "optimizer_updates": training["epochs_per_run"],
            "new_optimizer_updates": training["epochs_per_run"],
        }
        for dataset, condition, lr, seed in itertools.product(
            config["data"]["datasets"],
            config["conditions"],
            training["learning_rate_candidates"],
            training["tuning_seeds"],
        )
    ]


def _final_rows(config):
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "split": split,
            "ce": 0.9,
            "accuracy": 0.5,
            "num_nodes": config["data"]["expected_shapes"][dataset]["nodes"],
            "num_labeled_nodes": config["data"]["expected_shapes"][dataset][split],
        }
        for dataset, condition, seed, split in itertools.product(
            config["data"]["datasets"],
            config["conditions"],
            config["training"]["final_seeds"],
            ["train", "validation", "test"],
        )
    ]


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_complete_declared_budgets_and_all_job_conditions(profile):
    config = _config(profile)
    tuning = _tuning_rows(config)
    final = _final_rows(config)
    assert len(tuning) == config["training"]["tuning_runs"]
    assert len(final) == 3 * config["training"]["final_runs"]
    assert len(config["conditions"]) == 8
    if profile == "full":
        assert config["backbone"]["layers"] == 2 and config["backbone"]["hidden_dim"] == 64
        assert config["training"]["epochs_per_run"] == 500
        assert config["training"]["total_runs"] == 336
        assert config["training"]["total_updates"] == 168000
    tuning_jobs = build_jobs(config, "tuning")
    selected = select_learning_rates(config, tuning)
    final_jobs = build_jobs(config, "final", selected)
    assert len({job["job_key"] for job in tuning_jobs}) == len(tuning_jobs)
    assert all(job["seeds"] == config["training"]["tuning_seeds"] for job in tuning_jobs)
    assert all(job["seeds"] == config["training"]["final_seeds"] for job in final_jobs)
    assert verify_coverage(config, tuning, final)["all_datasets_conditions_seeds_splits"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("backbone", "layers", 1),
        ("backbone", "hidden_dim", 8),
        ("data", "sampling_ratio", 0.5),
        ("training", "epochs_per_run", 3),
        ("training", "learning_rate_candidates", [0.003]),
        ("training", "final_seeds", [11]),
        ("training", "test_evaluation_during_tuning", True),
    ],
)
def test_full_config_cannot_shrink_or_leak_test(tmp_path, section, key, value):
    config = json.loads((_FOLDER / "config_full.json").read_text())
    config[section][key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(path, "full")


@pytest.mark.parametrize(
    "alteration",
    [
        "missing",
        "duplicate",
        "wrong_seed",
        "nonfinite",
        "short_training",
        "short_updates",
        "test_field",
    ],
)
def test_validation_selection_rejects_corrupt_or_leaked_tuning(alteration):
    config = _config()
    rows = _tuning_rows(config)
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif alteration == "wrong_seed":
        rows[0]["seed"] = -1
    elif alteration == "nonfinite":
        rows[0]["validation_ce"] = float("nan")
    elif alteration == "short_training":
        rows[0]["training_epochs"] -= 1
    elif alteration == "short_updates":
        rows[0]["optimizer_updates"] -= 1
    else:
        rows[0]["test_ce"] = 0.3
    with pytest.raises(ValueError):
        select_learning_rates(config, rows)


def test_lr_selection_averages_all_independent_tuning_seeds_with_tie_rule():
    config = _config()
    rows = _tuning_rows(config)
    rates = config["training"]["learning_rate_candidates"]
    for row in rows:
        row["validation_ce"] = 0.5
        if row["lr"] == max(rates):
            row["validation_ce"] = (
                0.25 if row["seed"] == config["training"]["tuning_seeds"][0] else 0.75
            )
    selected = select_learning_rates(config, list(reversed(rows)))
    assert all(row["selected_lr"] == min(rates) for row in selected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_nodes", 1),
        ("num_labeled_nodes", 1),
        ("split", "subset"),
        ("seed", -1),
        ("ce", float("nan")),
    ],
)
def test_final_coverage_checks_all_nodes_and_public_splits(field, value):
    config = _config()
    rows = _final_rows(config)
    rows[0][field] = value
    with pytest.raises(ValueError):
        verify_coverage(config, _tuning_rows(config), rows)


def test_all_final_training_must_be_locked_before_test():
    config = _config()
    jobs = build_jobs(config, "final", select_learning_rates(config, _tuning_rows(config)))
    rows = [
        {
            "dataset": job["dataset"],
            "condition": job["condition"],
            "seed": seed,
            "lr": job["lr"],
            "phase": "final",
            "training_epochs": 3,
            "optimizer_updates": 3,
        }
        for job in jobs
        for seed in job["seeds"]
    ]
    _validate_final_training(config, rows, jobs)
    with pytest.raises(ValueError, match="before unlocking test"):
        _validate_final_training(config, rows[:-1], jobs)


def test_existing_output_and_full_local_training_rejected(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    common = dict(profile="debug", config=None, output_dir=str(existing), device="cpu")
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(**common))
    common.update(profile="full", output_dir=str(tmp_path / "new"))
    with pytest.raises(ValueError, match="Linux server CUDA"):
        run(SimpleNamespace(**common))


def test_frozen_coverage_rejects_missing_branch_and_intervention_rows():
    config = _config()
    with pytest.raises(ValueError, match="incomplete frozen"):
        verify_frozen_coverage(config, {"intervention_rows": [], "branch_rows": []})


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("1", 1), ("2", 2), ("8", 8)])
def test_cli_workers_reaches_data_loader_as_integer(value, expected):
    assert _workers(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid", "True"])
def test_cli_workers_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _workers(value)
