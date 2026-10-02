"""Regression tests for immutable source evaluation and complete scenario contracts."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from research.wedge_propagation.generalization.study import (
    original_reproduction,
    read_config,
    read_rows,
    run,
    source_manifest,
)


def test_source_manifest_covers_mathematical_dependencies():
    hashes = source_manifest()
    assert {
        "data.py",
        "operators.py",
        "study.py",
        "learned/data.py",
        "learned/model.py",
        "learned/evaluation.py",
        "learned/train.py",
        "generalization/study.py",
        "generalization/frozen.py",
        "generalization/config_full.json",
    } <= hashes.keys()
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_all_amplitudes_and_splits_preserved(profile):
    folder = Path(__file__).resolve().parents[1] / "research/wedge_propagation/generalization"
    config = read_config(folder / f"config_{profile}.json", profile)
    assert config["amplitudes"] == [0.25, 0.5, 1, 2, 4]
    assert len(config["splits"]) == 6 and len(config["targets"]) == 3
    assert len(config["conditions"]) == 5
    assert config["new_training_epochs"] == config["optimizer_updates"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"amplitudes": [1.0]},
        {"new_training_epochs": 1},
        {"source_profile": "full"},
        {"feature_realizations": 1},
    ],
)
def test_invalid_contract_cannot_silently_reduce_scope(tmp_path, change):
    path = Path(__file__).resolve().parents[1] / (
        "research/wedge_propagation/generalization/config_debug.json"
    )
    config = json.loads(path.read_text())
    config.update(change)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(invalid, "debug")


def test_csv_missing_diagnostic_stays_undefined(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text("graph_id,seed,message_relerr,weight_corr\ncase,-1,0.12,\n", encoding="utf-8")
    assert read_rows(path) == [
        {"graph_id": "case", "seed": -1, "message_relerr": 0.12, "weight_corr": None}
    ]
    path.write_text("message_relerr\nnan\n", encoding="utf-8")
    with pytest.raises(ArithmeticError):
        read_rows(path)


def test_original_reproduction_detects_changed_predictions():
    measured = {
        "target": "L",
        "condition": "first",
        "seed": -1,
        "split": "id",
        "graph_id": "case",
        "message_relerr": 0.0,
        "message_abs_rmse": 0.0,
        "u": 1.0,
        "v": None,
        "beta": None,
    }
    assert original_reproduction([measured], [dict(measured)], 1e-5)["verified"]
    altered = dict(measured, message_relerr=0.1)
    with pytest.raises(ArithmeticError, match="reproduction failed"):
        original_reproduction([measured], [altered], 1e-5)


def test_complete_debug_evaluation_preserves_source_models(wedge_completed_debug_run, tmp_path):
    from threadpoolctl import threadpool_limits

    output = tmp_path / "feature-study"
    output.mkdir()
    args = Namespace(
        run_dir=wedge_completed_debug_run,
        profile="debug",
        config=None,
        device="cpu",
        workers="1",
        batch_size="36",
    )
    with threadpool_limits(limits=1):
        run(args, output)
    completed = json.loads((output / "completion.json").read_text())
    contract = json.loads((output / "contract.json").read_text())
    # 36 graphs × 3 targets × (3 analytic + 2 gate kinds × 2 seeds) × 6 scenarios.
    assert completed["metric_rows"] == 4536
    assert completed["scale_rows"] == 3780
    assert contract["source_graph_count"] == 36
    assert contract["input_treatments"] == 864
    assert contract["scalar_feature_channels"] == 1
    assert contract["gate_hidden_dimension"] == 16
    assert contract["gate_linear_layers"] == 2
    assert contract["source_code_unchanged"]
    assert contract["optimizer_updates"] == contract["new_training_epochs"] == 0
    assert contract["models_unchanged"] and contract["source_artifacts_unchanged"]
    assert contract["original_metric_reproduction"]["verified"]
    assert contract["source_model_hashes"] == contract["model_hashes_after"]
    assert (output / "fresh-a1/dataset.npz").is_file()
    assert (output / "SYNTHETIC_GENERALIZATION_SUMMARY.md").is_file()
