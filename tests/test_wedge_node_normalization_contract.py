"""Normalization experiment budgets and imported-source provenance guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.wedge_propagation.node_normalization.common import (
    digest,
    read_config,
    source_manifest,
)

_FOLDER = Path(__file__).resolve().parents[1] / "research/wedge_propagation/node_normalization"
_OLD = _FOLDER.parent / "classification"


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_training_contract_preserves_each_condition_data_model_and_budget(profile):
    current = read_config(_FOLDER / f"config_{profile}.json", profile)
    old = json.loads((_OLD / f"config_{profile}.json").read_text(encoding="utf-8"))
    assert current["data"] == old["data"]
    assert current["backbone"] == old["backbone"]
    assert current["gate"] == old["gate"]
    for key in (
        "epochs_per_run",
        "learning_rate_candidates",
        "tuning_seeds",
        "final_seeds",
        "optimizer",
        "optimizer_betas",
        "optimizer_epsilon",
        "weight_decay",
        "physical_graph_batch_per_run",
        "effective_graph_batch_per_run",
        "gradient_accumulation_steps",
        "data_parallel_replicas_per_run",
        "early_stopping",
        "test_evaluation_during_tuning",
    ):
        assert current["training"][key] == old["training"][key]
    assert current["training"]["total_runs"] == (210 if profile == "full" else 90)
    assert current["training"]["total_updates"] == (105000 if profile == "full" else 270)
    assert len(current["conditions"]) == 5
    assert current["evaluation"]["test_already_observed_in_previous_experiment"] is True
    assert current["evaluation"]["independent_generalization_claim"] is False


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "epochs_per_run", 20),
        ("training", "final_seeds", [11]),
        ("training", "learning_rate_candidates", [0.001]),
        ("training", "test_evaluation_during_tuning", True),
        ("data", "sampling_ratio", 0.5),
        ("backbone", "hidden_dim", 8),
        ("backbone", "layers", 1),
        ("gate", "hidden_dim", 8),
        ("operators", "node_diagonal_gradient", "detach"),
        ("evaluation", "shuffle_manifests_per_dataset", 1),
        ("resources", "parallel_final_run_candidates", [1]),
    ],
)
def test_full_contract_rejects_reduction_test_leak_and_missing_normalization_gradient(
    tmp_path, section, key, value
):
    config = read_config(_FOLDER / "config_full.json", "full")
    config[section][key] = value
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(changed, "full")


def test_debug_cannot_be_used_as_full_and_hidden_caps_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        read_config(_FOLDER / "config_debug.json", "full")
    config = read_config(_FOLDER / "config_full.json", "full")
    config["max_nodes"] = 500
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(changed, "full")


def test_manifest_covers_model_data_training_and_new_normalization_with_portable_paths():
    manifest = source_manifest()
    files = manifest["sha256"]
    prefix = "research/wedge_propagation/"
    for file in (
        "classification/model.py",
        "classification/data.py",
        "classification/config_full.json",
        "node_normalization/model.py",
        "node_normalization/training.py",
        "node_normalization/config_full.json",
        "study.py",
        "operators.py",
    ):
        assert prefix + file in files
    assert manifest["code_digest"] == digest(files)
    assert all(not Path(key).is_absolute() and ":" not in key for key in files)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "all"])
def test_invalid_cpu_thread_setting_rejected(tmp_path, threads):
    config = read_config(_FOLDER / "config_full.json", "full")
    config["runtime"]["cpu_threads"] = threads
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(changed, "full")
