"""Experiment 3.1 keeps complete artifacts, matched training and frozen controls."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from threadpoolctl import threadpool_limits

from research.wedge_propagation.scale_normalization.study import (
    output_location,
    read_config,
    read_rows,
    restore_checkpoint,
    run,
    source_manifest,
)


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_complete_contract(profile):
    folder = Path(__file__).resolve().parents[1] / "research/wedge_propagation/scale_normalization"
    config = read_config(folder / f"config_{profile}.json", profile)
    assert config["amplitudes"] == [0.25, 0.5, 1, 2, 4]
    assert config["variants"] == ["raw", "normalized"]
    assert not config["scale_augmentation"] and not config["teacher_weight_loss"]
    assert config["training_contract"] == "inherit_complete_source"
    assert config["training_batch"] == "retain_source"


@pytest.mark.parametrize(
    "change",
    [
        {"amplitudes": [1]},
        {"epochs": 1},
        {"variants": ["normalized"]},
        {"teacher_weight_loss": True},
        {"normalization": "teacher_ratio"},
    ],
)
def test_config_cannot_change_design_or_hide_scope(tmp_path, change):
    folder = Path(__file__).resolve().parents[1] / "research/wedge_propagation/scale_normalization"
    config = json.loads((folder / "config_debug.json").read_text())
    config.update(change)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(path, "debug")


def test_output_cannot_overlap_sources(tmp_path):
    source = tmp_path / "original"
    features = tmp_path / "features"
    for output in (source, source / "new", features, features / "new", tmp_path):
        with pytest.raises(ValueError):
            output_location(output, source, features)
    assert output_location(tmp_path / "new", source, features) == tmp_path / "new"


def test_code_guard_covers_reused_and_new_math():
    assert {
        "learned/model.py",
        "learned/train.py",
        "generalization/data.py",
        "generalization/frozen.py",
        "scale_normalization/model.py",
        "scale_normalization/data.py",
        "scale_normalization/study.py",
        "scale_normalization/report.py",
        "scale_normalization/config_full.json",
    } <= source_manifest().keys()


@pytest.fixture(scope="module")
def completed_normalization(
    wedge_completed_debug_run, wedge_completed_feature_debug_run, tmp_path_factory
):
    output = tmp_path_factory.mktemp("wedge-normalization-debug")
    with threadpool_limits(limits=1):
        run(
            Namespace(
                source_dir=wedge_completed_debug_run,
                feature_source_dir=wedge_completed_feature_debug_run,
                config=None,
                profile="debug",
                device="cpu",
                workers="1",
                resume_from=None,
            ),
            output,
        )
    return output


def test_complete_real_debug_pipeline(completed_normalization):
    output = completed_normalization
    contract = json.loads((output / "contract.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "complete"
    assert completion["metric_rows"] == 9072
    assert completion["scale_rows"] == 7560
    assert completion["intervention_rows"] == 4320
    assert contract["graph_count_total"] == contract["graph_count_used"] == 36
    assert contract["data_fraction"] == 1 and contract["new_training_epochs"] == 4
    assert contract["normalized_gate_jobs"] == 6
    assert (
        contract["raw_optimizer_updates"]
        == contract["test_updates"]
        == contract["scalar_refits"]
        == 0
    )
    assert contract["models_unchanged"] and contract["source_code_unchanged"]
    assert contract["source_artifacts_unchanged"] and contract["feature_source_artifacts_unchanged"]
    assert contract["training_physical_graph_batch"] == 12
    assert contract["inference_physical_graph_batch"] == 36
    assert contract["optimizer_calls_in_study"] == 24
    assert len(contract["raw_metric_reproductions"]) == 6
    assert all(row["verified"] for row in contract["raw_metric_reproductions"])
    assert len(contract["normalized_jobs"]) == 6
    assert all(len(job["best_epochs"]) == 2 for job in contract["normalized_jobs"])
    assert contract["source_config"]["hidden"] == 16
    assert not (output / "failure.json").exists()


def test_frozen_controls_and_new_weights_are_actually_evaluated(completed_normalization):
    output = completed_normalization
    rows = read_rows(output / "metrics.csv")
    key_fields = ("scenario", "amplitude", "target", "condition", "seed", "split", "graph_id")
    raw = {
        tuple(row[field] for field in key_fields): row for row in rows if row["variant"] == "raw"
    }
    normalized = {
        tuple(row[field] for field in key_fields): row
        for row in rows
        if row["variant"] == "normalized"
    }
    assert raw.keys() == normalized.keys()
    for key in raw:
        if raw[key]["condition"] in ("first", "polynomial", "fixed"):
            assert raw[key]["message_relerr"] == normalized[key]["message_relerr"]
    assert any(
        raw[key]["message_relerr"] != normalized[key]["message_relerr"]
        for key in raw
        if raw[key]["condition"] == "learned"
    )
    scales = read_rows(output / "scale_checks.csv")
    measured = [
        row["student_weight_scale_relerr"]
        for row in scales
        if row["variant"] == "normalized"
        and row["condition"] == "learned"
        and row["student_weight_scale_relerr"] is not None
    ]
    assert measured and max(measured) < 1e-5
    selected = torch.load(
        output / "checkpoints/path-learned-selected.pt", map_location="cpu", weights_only=True
    )
    from research.wedge_propagation.scale_normalization.model import make_normalized_model

    initial = make_normalized_model("learned", [11, 23], 16, 1).float()
    assert any(
        not torch.equal(initial.state_dict()[name], value)
        for name, value in selected["state_dict"].items()
    )


def test_matching_resume_checkpoint_and_incompatible_raw_rejected(
    completed_normalization, tmp_path
):
    output = completed_normalization
    contract = json.loads((output / "contract.json").read_text())
    path = output / "checkpoints/path-learned-epoch0002.pt"
    saved = restore_checkpoint(
        path, contract["training_config"], contract["source_code_sha256"], 12, "cpu"
    )
    assert saved["epoch"] == 2
    bad = copy.deepcopy(contract["training_config"])
    bad.pop("normalization")
    with pytest.raises(ValueError):
        restore_checkpoint(path, bad, contract["source_code_sha256"], 12, "cpu")
    with pytest.raises(ValueError):
        restore_checkpoint(
            path, contract["training_config"], contract["source_code_sha256"], 6, "cpu"
        )
