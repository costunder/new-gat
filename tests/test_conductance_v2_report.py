"""V2 artifact-integrity fixtures only; no datasets or trained models."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from research.conductance_gat.v2.protocol import COMMON, CONDITIONS, PARAMETERIZATION, SUITE
from research.conductance_gat.v2.report import ComparisonIntegrityError, main, write_comparison


def _fixture(tmp_path, datasets=("ogbn-arxiv",), edges=3):
    root = tmp_path / "direct-c-v2"
    root.mkdir()
    config = {
        **COMMON,
        "datasets": list(datasets),
        "model_seed": 0,
        "epochs": 100,
        "patience": 20,
        "batch_size": 1,
        "workers": 0,
        "device": "cuda",
        "edge_chunk_size": 65536,
    }
    source = {"research/conductance_gat/v2/model.py": "1" * 64}
    manifest = {
        "schema_version": 1,
        "suite": SUITE,
        "status": "passed",
        "source_integrity_valid": True,
        "config": config,
        "conditions": CONDITIONS,
        "sources": {"sha256": source},
        "jobs": [],
    }
    configuration = {k: v for k, v in config.items() if k != "datasets"}
    configuration.update(tf32=False, pin_memory=True)
    for dataset in datasets:
        for condition, spec in CONDITIONS.items():
            output = root / dataset / condition
            output.mkdir(parents=True)
            checkpoint, history = output / "best.pt", output / "history.json"
            checkpoint.write_bytes(b"unit-fixture-not-a-trained-model")
            history.write_text("[]", encoding="utf-8")
            data_hash = hashlib.sha256(dataset.encode()).hexdigest()
            direct_parameters = COMMON["layers"] * edges if condition == "direct_c" else 0
            metrics = {
                "schema_version": 1,
                "research_suite": SUITE,
                "status": "passed",
                "dataset": dataset,
                "condition": condition,
                "model_seed": 0,
                **spec,
                "non_gate_weight_decay": 0.0005,
                "configuration": copy.deepcopy(configuration),
                "cache_sha256": data_hash,
                "protocol": {"data_sha256": data_hash, "official": True},
                "initial_state_sha256": hashlib.sha256(condition.encode()).hexdigest(),
                "shared_backbone_initial_state_sha256": "c" * 64,
                "best_epoch": 10,
                "epochs_run": 30,
                "validation": 0.55 if condition == "direct_c" else 0.50,
                "metric_name": "accuracy",
                "train_loss": 0.7,
                "elapsed_seconds": 2.0,
                "peak_cuda_allocated_bytes": 100,
                "evaluation_split": "validation",
                "test_evaluated": False,
                "versions": {"torch": "unit-fixture"},
                "gpu": "unit-fixture",
                "total_parameters": 200 + direct_parameters,
                "trainable_parameters": 200 + direct_parameters,
                "frozen_parameters": 0,
                "source_sha256": source,
                "parameterization": PARAMETERIZATION,
                "topology": {"num_nodes": 4, "num_edges": edges, "incidence_sha256": "2" * 64},
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
                    "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def _edit(manifest, callback, index=0, refresh_digest=True):
    job = manifest["jobs"][index]
    path = Path(job["metrics_path"])
    metrics = json.loads(path.read_text(encoding="utf-8"))
    callback(metrics)
    path.write_text(json.dumps(metrics), encoding="utf-8")
    if refresh_digest:
        job["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_report_has_units_resources_and_unchanged_children(tmp_path):
    root, manifest = _fixture(tmp_path, ("cora", "ogbn-arxiv"))
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = write_comparison(root, manifest)
    assert result["status"] == "passed" and result["n_model_seeds"] == 1
    assert result["test_evaluated"] is False
    for dataset in result["datasets"]:
        assert dataset["direct_minus_fixed"]["percentage_points"] == pytest.approx(5.0)
        assert dataset["held_fixed"]["topology"]["num_edges"] == 3
    assert all(path.read_bytes() == value for path, value in before.items())
    text = (root / "comparison.md").read_text(encoding="utf-8")
    assert "+5.000000 pp" in text and "Peak CUDA" in text and "Train loss" in text
    assert "not a measured scalability result" in text
    with (root / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and float(rows[0]["direct_minus_fixed_pp"]) == pytest.approx(5.0)
    assert main([str(root)]) == 0


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_partial_results_have_no_contrast(tmp_path, status):
    root, manifest = _fixture(tmp_path)
    manifest["status"] = "failed" if status == "failed" else "running"
    manifest["jobs"][1]["status"] = status
    result = write_comparison(root, manifest)
    assert not result["complete"] and result["datasets"][0]["direct_minus_fixed"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("cache_sha256", "b" * 64),
        ("shared_backbone_initial_state_sha256", "b" * 64),
        ("test_evaluated", True),
        ("model_seed", 1),
        ("normalization", "global_max"),
        ("gate_mode", "fixed_one"),
        ("gate_weight_decay", 0.0005),
        ("non_gate_weight_decay", 0.0),
        ("frozen_parameters", 6),
        ("total_parameters", 300),
        ("research_suite", "conductance_c_learning"),
        ("validation", float("nan")),
        ("versions", {"torch": "different"}),
        ("gpu", "different"),
        ("parameterization", "mlp"),
        ("source_sha256", {"model.py": "f" * 64}),
        ("topology", {"num_nodes": 4, "num_edges": 3, "incidence_sha256": "b" * 64}),
        ("topology", {"num_nodes": 4, "num_edges": -1, "incidence_sha256": "2" * 64}),
        ("topology", {"num_nodes": 4, "num_edges": 3}),
    ],
)
def test_child_mismatch_invalidates_all_deltas(tmp_path, key, value):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda metrics: metrics.update({key: value}))
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)
    result = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert result["status"] == "invalid"
    assert all(dataset["direct_minus_fixed"] is None for dataset in result["datasets"])


@pytest.mark.parametrize(
    "change", ["source", "missing_source", "duplicate", "missing", "spec", "ppi"]
)
def test_invalid_manifest_withholds_contrasts(tmp_path, change):
    root, manifest = _fixture(tmp_path)
    if change == "source":
        manifest["source_integrity_valid"] = False
    elif change == "missing_source":
        manifest.pop("sources")
    elif change == "duplicate":
        manifest["jobs"].append(manifest["jobs"][0])
    elif change == "missing":
        manifest["jobs"].pop()
    elif change == "ppi":
        manifest["config"]["datasets"] = ["ppi"]
    else:
        manifest["conditions"] = {}
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize("artifact", ["checkpoint", "history"])
def test_changed_artifacts_rejected(tmp_path, artifact):
    root, manifest = _fixture(tmp_path)
    child = json.loads(Path(manifest["jobs"][0]["metrics_path"]).read_text(encoding="utf-8"))
    Path(child[artifact]).write_bytes(b"changed")
    with pytest.raises(ComparisonIntegrityError, match="SHA-256 mismatch"):
        write_comparison(root, manifest)


def test_changed_metrics_rejected_even_with_valid_score(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda child: child.update(validation=0.9), refresh_digest=False)
    with pytest.raises(ComparisonIntegrityError, match="metrics SHA-256 mismatch"):
        write_comparison(root, manifest)


def test_output_symlink_rejected_before_writes(tmp_path, monkeypatch):
    root, manifest = _fixture(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == "comparison.md" or original(p))
    with pytest.raises(ValueError, match="symlinks"):
        write_comparison(root, manifest)
    assert not (root / "comparison.json").exists()


def test_changed_chunk_size_or_unknown_config_rejected(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda child: child["configuration"].update(edge_chunk_size=1))
    with pytest.raises(ComparisonIntegrityError, match="configuration"):
        write_comparison(root, manifest)


def test_fixed_control_must_not_claim_frozen_parameters(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(
        manifest, lambda child: child.update(frozen_parameters=5, trainable_parameters=195), index=1
    )
    with pytest.raises(ComparisonIntegrityError, match="parameter counts"):
        write_comparison(root, manifest)


def test_empty_edge_topology_preserves_parameter_free_fixed_control(tmp_path):
    root, manifest = _fixture(tmp_path, edges=0)
    assert write_comparison(root, manifest)["status"] == "passed"
