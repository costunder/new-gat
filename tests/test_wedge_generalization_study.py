"""Regression tests for immutable source evaluation and complete scenario contracts."""

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.wedge_propagation.generalization.study import (
    cache_original_batches,
    original_reproduction,
    read_config,
    read_rows,
    run,
    source_manifest,
    target_rms_scales,
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
    scales = {("L", "case"): 1.0}
    assert original_reproduction([measured], [dict(measured)], 1e-5, target_rms=scales)["verified"]
    altered = dict(measured, message_relerr=0.1)
    with pytest.raises(ArithmeticError, match="reproduction failed"):
        original_reproduction([measured], [altered], 1e-5, target_rms=scales)


def test_small_residual_replay_uses_target_scale_and_rejects_real_change():
    measured = {
        "target": "path",
        "condition": "learned",
        "seed": 11,
        "split": "family_ood",
        "graph_id": "star40",
        "message_relerr": 0.01,
        "message_abs_rmse": 5.948675171168588,
        "u": None,
        "v": None,
        "beta": 1.0,
    }
    scales = {("path", "star40"): 480.0}
    repeated = dict(measured, message_abs_rmse=5.948534762838067)
    checked = original_reproduction([measured], [repeated], 1e-5, target_rms=scales)
    assert checked["verified"] and checked["maximum_absolute_error"] > 1e-4
    assert checked["maximum_target_normalized_rmse_error"] < 1e-6
    with pytest.raises(ArithmeticError):
        original_reproduction(
            [measured], [dict(repeated, message_abs_rmse=6.9485)], 1e-5, target_rms=scales
        )
    with pytest.raises(ArithmeticError):
        original_reproduction([measured], [dict(repeated, beta=1.1)], 1e-5, target_rms=scales)
    with pytest.raises(ArithmeticError):
        original_reproduction(
            [measured], [dict(repeated, message_relerr=0.1)], 1e-5, target_rms=scales
        )


def test_zero_target_replay_does_not_hide_nonzero_predictions():
    measured = {
        "target": "L",
        "condition": "first",
        "seed": -1,
        "split": "id",
        "graph_id": "zero",
        "message_relerr": 0.0,
        "message_abs_rmse": 0.0,
        "u": 1.0,
        "v": None,
        "beta": None,
    }
    scales = {("L", "zero"): 0.0}
    with pytest.raises(ArithmeticError):
        original_reproduction(
            [measured], [dict(measured, message_abs_rmse=1e-10)], 1e-5, target_rms=scales
        )


@pytest.mark.parametrize("field", ["physical_graph_batch_actual", "input_shapes"])
def test_replay_rejects_changed_source_batch_layout(wedge_completed_debug_run, field):
    import torch

    from research.wedge_propagation.generalization.data import load_source_run

    source = load_source_run(wedge_completed_debug_run, "debug")
    contract = copy.deepcopy(source.contract)
    contract[field]["family_ood"][0] = 1 if field == "physical_graph_batch_actual" else [1, 4]
    altered = SimpleNamespace(cases=source.cases, contract=contract)
    with pytest.raises(ValueError, match="source replay batch layout mismatch"):
        cache_original_batches(altered, torch.device("cpu"), torch.float32)


def test_target_rms_is_from_stored_source_precision(wedge_completed_debug_run):
    import torch

    from research.wedge_propagation.generalization.data import load_source_run

    source = load_source_run(wedge_completed_debug_run, "debug")
    scales = target_rms_scales(source.cases, torch.float32)
    case = source.cases[0]
    expected = float(case.target_path.float().double().square().mean().sqrt())
    assert scales[("path", case.graph_id)] == expected
    assert len(scales) == 3 * len(source.cases)


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
    layout = contract["original_reproduction_layout"]
    assert layout["physical_graph_batch_selected"] == 12
    assert layout["source_batch_layout_verified"]
    assert layout["split_order"] == [
        "validation",
        "id",
        "size_ood",
        "family_ood",
        "family_size_ood",
        "train",
    ]
    replay_rows = read_rows(output / "original_reproduction_metrics.csv")
    assert len(replay_rows) == 756
    assert all(row["reproduction_target_rms"] >= 0 for row in replay_rows)
    assert contract["source_model_hashes"] == contract["model_hashes_after"]
    assert (output / "fresh-a1/dataset.npz").is_file()
    assert (output / "SYNTHETIC_GENERALIZATION_SUMMARY.md").is_file()
