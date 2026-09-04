"""Report integrity fixtures only: these files contain no trained model or dataset."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from research.conductance_gat.c_learning.protocol import COMMON, CONDITIONS
from research.conductance_gat.c_learning.report import (
    ComparisonIntegrityError,
    main,
    write_comparison,
)


def _fixture(tmp_path, datasets=("ppi",)):
    root = tmp_path / "c-learning"
    root.mkdir()
    config = {
        **COMMON,
        "datasets": list(datasets),
        "model_seed": 0,
        "epochs": 100,
        "patience": 20,
        "batch_size": 2,
        "workers": 0,
        "device": "cuda",
    }
    manifest = {
        "schema_version": 1,
        "suite": "conductance_c_learning",
        "status": "passed",
        "source_integrity_valid": True,
        "config": config,
        "conditions": CONDITIONS,
        "jobs": [],
    }
    configuration = {k: v for k, v in config.items() if k != "datasets"}
    configuration.update(tf32=False, pin_memory=True)
    for dataset in datasets:
        for condition, spec in CONDITIONS.items():
            output = root / dataset / condition
            output.mkdir(parents=True)
            checkpoint, history = output / "best.pt", output / "history.json"
            checkpoint.write_bytes(b"unit-fixture-no-trained-model")
            history.write_text("[]", encoding="utf-8")
            data_hash = hashlib.sha256(dataset.encode()).hexdigest()
            estimator = 100 if condition == "learned_c" else 0
            total = 100 + estimator
            parameter_tensors = 8 if condition == "learned_c" else 4
            ownership = {
                "status": "passed",
                "trainable_parameter_tensors": parameter_tensors,
                "optimizer_owned_parameter_tensors": parameter_tensors,
                "trainable_parameter_elements": total,
            }
            metrics = {
                "schema_version": 1,
                "research_suite": "conductance_c_learning",
                "status": "passed",
                "dataset": dataset,
                "condition": condition,
                "model_seed": 0,
                **spec,
                "non_gate_weight_decay": 0.0005,
                "configuration": copy.deepcopy(configuration),
                "cache_sha256": data_hash,
                "protocol": {"data_sha256": data_hash, "official": True},
                "initial_state_sha256": (
                    "a" * 64 if condition == "learned_c" else "b" * 64
                ),
                "shared_backbone_initial_state_sha256": "c" * 64,
                "best_epoch": 10,
                "epochs_run": 30,
                "validation": 0.55 if condition == "learned_c" else 0.50,
                "metric_name": "micro_f1" if dataset == "ppi" else "accuracy",
                "train_loss": 0.7,
                "elapsed_seconds": 2.0,
                "peak_cuda_allocated_bytes": 100,
                "evaluation_split": "validation",
                "test_evaluated": False,
                "versions": {"torch": "unit-fixture"},
                "gpu": "unit-fixture",
                "total_parameters": total,
                "estimator_parameters": estimator,
                "non_estimator_parameters": 100,
                "trainable_parameters": total,
                "frozen_parameters": 0,
                "pre_run_observability": {
                    "status": "pre_run_configuration",
                    "model": {
                        "total_parameters": total,
                        "trainable_parameters": total,
                        "frozen_parameters": 0,
                        "optimizer_ownership": ownership,
                    },
                },
                "first_optimizer_step_integrity": {
                    **ownership,
                    "checked_before_optimizer_step": 1,
                    "gradient_status": (
                        "all_trainable_parameter_tensors_have_finite_gradients"
                    ),
                },
            }
            for name, path in (("checkpoint", checkpoint), ("history", history)):
                metrics[name] = str(path)
                metrics[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metrics_path = output / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            manifest["jobs"].append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "status": "passed",
                    "output_dir": str(output),
                    "metrics_path": str(metrics_path),
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def _edit(manifest, callback, index=0):
    path = Path(manifest["jobs"][index]["metrics_path"])
    metrics = json.loads(path.read_text(encoding="utf-8"))
    callback(metrics)
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_complete_report_contrast_units_metrics_and_readonly_children(tmp_path):
    root, manifest = _fixture(tmp_path, ("ppi", "ogbn-arxiv"))
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    report = write_comparison(root, manifest)
    assert report["status"] == "passed" and report["complete"]
    assert report["n_model_seeds"] == 1 and report["test_evaluated"] is False
    assert [row["metric_name"] for row in report["datasets"]] == ["micro_f1", "accuracy"]
    for dataset in report["datasets"]:
        assert dataset["learned_minus_fixed"]["score_delta"] == pytest.approx(0.05)
        assert dataset["learned_minus_fixed"]["percentage_points"] == pytest.approx(5.0)
        assert dataset["parameter_contract"]["verified"] is True
        assert dataset["parameter_contract"]["total_parameter_difference"] == 100
    assert all(path.read_bytes() == data for path, data in before.items())
    text = (root / "comparison.md").read_text(encoding="utf-8")
    assert "+5.000000 pp" in text and "n=1" in text and "parameter-free" in text
    assert "Frozen scaffold" not in text
    with (root / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and float(rows[0]["learned_minus_fixed_pp"]) == pytest.approx(5.0)
    assert main([str(root)]) == 0


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_partial_results_withhold_contrast(tmp_path, status):
    root, manifest = _fixture(tmp_path)
    manifest["status"] = "failed" if status == "failed" else "running"
    manifest["jobs"][1]["status"] = status
    report = write_comparison(root, manifest)
    assert not report["complete"]
    assert report["datasets"][0]["learned_minus_fixed"] is None
    assert report["datasets"][0]["conditions"][1]["validation"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("cache_sha256", "b" * 64),
        ("shared_backbone_initial_state_sha256", "d" * 64),
        ("test_evaluated", True),
        ("model_seed", 1),
        ("normalization", "global_max"),
        ("gate_mode", "fixed_one"),
        ("gate_weight_decay", 0.0),
        ("non_gate_weight_decay", 0.0),
        ("frozen_parameters", 100),
        ("estimator_parameters", 99),
        ("non_estimator_parameters", 101),
        ("total_parameters", 300),
        ("research_suite", "conductance_factorial"),
        ("validation", float("nan")),
        ("versions", {"torch": "mismatched"}),
        ("gpu", "different"),
    ],
)
def test_mismatch_invalidates_all_deltas(tmp_path, key, value):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda metrics: metrics.update({key: value}))
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)
    report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert all(row["learned_minus_fixed"] is None for row in report["datasets"])


@pytest.mark.parametrize("change", ["source", "missing_source", "duplicate", "missing", "spec"])
def test_invalid_manifest_withholds_deltas(tmp_path, change):
    root, manifest = _fixture(tmp_path)
    if change == "source":
        manifest["source_integrity_valid"] = False
    elif change == "missing_source":
        manifest.pop("source_integrity_valid")
    elif change == "duplicate":
        manifest["jobs"].append(manifest["jobs"][0])
    elif change == "missing":
        manifest["jobs"].pop()
    else:
        manifest["conditions"] = {}
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize("artifact", ["checkpoint", "history"])
def test_modified_artifacts_rejected(tmp_path, artifact):
    root, manifest = _fixture(tmp_path)
    child = json.loads(Path(manifest["jobs"][0]["metrics_path"]).read_text(encoding="utf-8"))
    Path(child[artifact]).write_bytes(b"modified")
    with pytest.raises(ComparisonIntegrityError, match="SHA-256 mismatch"):
        write_comparison(root, manifest)


def test_escaping_artifact_cannot_make_valid_report(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda child: child.update(checkpoint=str(tmp_path / "outside.pt")))
    with pytest.raises(ComparisonIntegrityError, match="escapes"):
        write_comparison(root, manifest)


def test_output_symlink_is_rejected_before_writes(tmp_path, monkeypatch):
    root, manifest = _fixture(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == "comparison.md" or original(p))
    with pytest.raises(ValueError, match="symlinks"):
        write_comparison(root, manifest)
    assert not (root / "comparison.json").exists()


def test_common_config_and_cross_arm_extra_config_cannot_drift(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda child: child["configuration"].update(unknown_training_flag=True))
    with pytest.raises(ComparisonIntegrityError, match="configuration"):
        write_comparison(root, manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "missing"),
        ("optimizer_owned_parameter_tensors", 7),
        ("trainable_parameter_elements", 199),
    ],
)
def test_optimizer_ownership_evidence_fails_closed(tmp_path, field, value):
    root, manifest = _fixture(tmp_path)

    def mutate(metrics):
        evidence = metrics["pre_run_observability"]["model"]["optimizer_ownership"]
        evidence[field] = value

    _edit(manifest, mutate)
    with pytest.raises(ComparisonIntegrityError, match="optimizer"):
        write_comparison(root, manifest)


def test_first_step_ownership_evidence_fails_closed(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(
        manifest,
        lambda metrics: metrics["first_optimizer_step_integrity"].update(
            checked_before_optimizer_step=0
        ),
    )
    with pytest.raises(ComparisonIntegrityError, match="first optimizer step"):
        write_comparison(root, manifest)
