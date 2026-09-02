"""Artifact-only integrity tests for the Conductance V4 factorial report."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from research.conductance_gat.v4 import report as report_module  # noqa: E402
from research.conductance_gat.v4.model import RelativeCSpatialNodeClassifier  # noqa: E402
from research.conductance_gat.v4.protocol import (  # noqa: E402
    COMMON,
    CONDITIONS,
    PARAMETERIZATION,
    SUITE,
)
from research.conductance_gat.v4.report import (  # noqa: E402
    ComparisonIntegrityError,
    main,
    write_comparison,
)
from research.conductance_gat.v4.train import (  # noqa: E402
    _parameter_metadata,
    configuration,
    make_optimizer,
    optimizer_metadata,
)

SCORES = {
    "fixed_c_identity_w": 0.50,
    "relative_c_identity_w": 0.52,
    "fixed_c_spatial_w": 0.55,
    "relative_c_spatial_w": 0.60,
}
INTERVENTIONS = (
    "mean_c",
    "shuffled_c",
    "ones_c",
    "identity_w",
    "ones_c_identity_w",
    "propagation_off",
)


def _layer(index, specification, *, gradients=False):
    c_active = specification["gate_mode"] == "relative"
    w_active = specification["spatial_mode"] == "learned"
    return {
        "layer": index,
        "score": {"std": 0.1 if c_active else 0.0},
        "conductance": {
            "mean": 1.0,
            "std": 0.02 if c_active else 0.0,
            "cv": 0.02 if c_active else 0.0,
            "min": 0.9 if c_active else 1.0,
            "max": 1.1 if c_active else 1.0,
        },
        "log_conductance": {
            "mean": 0.0,
            "std": 0.02 if c_active else 0.0,
            "min": -0.1 if c_active else 0.0,
            "max": 0.1 if c_active else 0.0,
        },
        "alpha": 0.5,
        "gamma": 0.5,
        "tau": 1.0,
        "estimator_trainable": c_active,
        "weighted_degree": {
            "quantiles": {"p50": 2.0, "p99": 4.0},
            "max_over_median": 2.0,
        },
        "relative_message_transform_change": 0.03 if w_active else 0.0,
        "relative_conv_change": 0.2,
        "gate_parameter_norm": 1.0,
        "gate_gradient_norm": 0.1 if gradients and c_active else None,
        "spatial_gradient_norm": 0.2 if gradients and w_active else None,
        "spatial_weight": {
            "spatial_mode": specification["spatial_mode"],
            "trainable": w_active,
            "parameter_norm": 8.0,
            "identity_distance_frobenius": 0.1 if w_active else 0.0,
            "identity_relative_distance": 0.0125 if w_active else 0.0,
            "singular_values": {
                "count": 64,
                "min": 0.9 if w_active else 1.0,
                "max": 1.1 if w_active else 1.0,
                "mean": 1.0,
                "std": 0.01 if w_active else 0.0,
                "condition_number": 1.1 / 0.9 if w_active else 1.0,
            },
        },
    }


def _training_record(epoch, specification):
    return {
        "epoch": epoch,
        "batch_index": 0,
        "optimizer_steps_before_batch": epoch - 1,
        "scope": "full_graph_train_mask",
        "mode": "train_dropout_on",
        "stage": "after_task_backward_before_optimizer_step",
        "layers": [_layer(index, specification, gradients=True) for index in range(2)],
    }


def _diagnostics(score, specification):
    layers = [_layer(index, specification) for index in range(2)]
    intervention_rows = [
        {
            "intervention": name,
            "intervention_kind": "read_only_selected_checkpoint",
            "fresh_training": False,
            "validation": score,
            "percentage_points": 0.0,
            "changed_prediction_fraction": 0.0,
            "logit_mean_absolute_delta": 0.0,
        }
        for name in INTERVENTIONS
    ]
    return {
        "initial_validation": {"mode": "eval", "split": "validation", "layers": layers},
        "best_validation": {
            "mode": "eval",
            "split": "validation",
            "metric": score,
            "layers": layers,
        },
        "final_validation": {"mode": "eval", "split": "validation", "layers": layers},
        "train_trajectory": [
            _training_record(1, specification),
            _training_record(2, specification),
        ],
        "best_checkpoint_interventions": {
            "status": "passed",
            "scope": "validation_selected_best_checkpoint_only",
            "layers": "all_layers_simultaneously",
            "original": {"validation": score, "loss": 1.0},
            "rows": intervention_rows,
            "shuffle_seed": 0,
            "normalization_recomputed_for_c_interventions": True,
            "mean_c_numeric_check": {
                "comparison": "mean_c_vs_ones_c",
                "role": "informational_non_gating",
                "separate_full_graph_forwards": True,
                "allclose_rtol": 1e-5,
                "allclose_atol": 1e-6,
                "within_declared_tolerance": True,
                "logit_mean_absolute_delta": 0.0,
                "logit_max_absolute_delta": 0.0,
                "changed_prediction_fraction": 0.0,
                "replacement_contracts": {
                    "mean_c": {
                        "contract": "graph_constant_positive",
                        "satisfied": True,
                        "layers_checked": 2,
                        "edge_counts": [3, 3],
                    },
                    "ones_c": {
                        "contract": "exact_one",
                        "satisfied": True,
                        "layers_checked": 2,
                        "edge_counts": [3, 3],
                    },
                },
            },
        },
    }


def _fixture(tmp_path):
    root = tmp_path / "hybrid-c-spatial-v4"
    root.mkdir()
    args = SimpleNamespace(
        model_seed=0,
        epochs=2,
        patience=1,
        batch_size=1,
        workers=0,
        device="cuda",
        edge_chunk_size=65536,
    )
    config = {
        **COMMON,
        "datasets": ["ogbn-arxiv"],
        "model_seed": 0,
        "epochs": 2,
        "patience": 1,
        "batch_size": 1,
        "workers": 0,
        "device": "cuda",
        "edge_chunk_size": 65536,
        "data_root": str(tmp_path / "data"),
    }
    source = {"research/conductance_gat/v4/model.py": "1" * 64}
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
    data_hash = "2" * 64
    for condition, specification in CONDITIONS.items():
        torch.manual_seed(19)
        model = RelativeCSpatialNodeClassifier(
            3,
            2,
            hidden_channels=64,
            layers=2,
            dropout=0.5,
            gate_mode=specification["gate_mode"],
            spatial_mode=specification["spatial_mode"],
        )
        optimizer = make_optimizer(model, condition)
        output = root / "ogbn-arxiv" / condition
        output.mkdir(parents=True)
        checkpoint, history = output / "best.pt", output / "history.json"
        checkpoint.write_bytes(b"unit-fixture-not-a-model")
        history.write_text("[]", encoding="utf-8")
        score = SCORES[condition]
        metrics = {
            "schema_version": 1,
            "research_suite": SUITE,
            "status": "passed",
            "model": SUITE,
            "dataset": "ogbn-arxiv",
            "condition": condition,
            "model_seed": 0,
            **specification,
            "non_gate_weight_decay": COMMON["weight_decay"],
            "configuration": configuration(args),
            "cache_sha256": data_hash,
            "protocol": {"data_sha256": data_hash, "official": True},
            "initial_state_sha256": "a" * 64,
            "topology": {
                "num_nodes": 4,
                "num_edges": 3,
                "incidence_sha256": "3" * 64,
            },
            "parameterization": PARAMETERIZATION,
            "source_sha256": source,
            "optimizer": "AdamW",
            "optimizer_groups": optimizer_metadata(optimizer),
            **_parameter_metadata(model),
            "evaluation_split": "validation",
            "test_evaluated": False,
            "best_epoch": 1,
            "stop_epoch": 2,
            "stopping_reason": "patience",
            "epochs_run": 2,
            "validation": score,
            "metric_name": "accuracy",
            "train_loss": 0.7,
            "selection_loop_seconds": 0.7,
            "post_selection_diagnostics_seconds": 0.2,
            "epoch_timing": {
                "count": 2,
                "total_seconds": 0.6,
                "mean_seconds": 0.3,
                "median_seconds": 0.3,
                "p90_seconds": 0.38,
                "min_seconds": 0.2,
                "max_seconds": 0.4,
                "quantile_method": "linear_order_statistic",
                "scope": "unit fixture",
            },
            "elapsed_seconds": 1.0,
            "peak_cuda_allocated_bytes": 100,
            "peak_cuda_reserved_bytes": 200,
            "versions": {"torch": "unit-fixture"},
            "gpu": "unit-fixture",
            "diagnostics": _diagnostics(score, specification),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "history": str(history.resolve()),
            "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
        }
        metrics_path = output / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        manifest["jobs"].append(
            {
                "dataset": "ogbn-arxiv",
                "condition": condition,
                "status": "passed",
                "output_dir": str(output),
                "metrics_path": str(metrics_path),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def _edit(manifest, callback, *, condition="fixed_c_identity_w", refresh=True):
    job = next(job for job in manifest["jobs"] if job["condition"] == condition)
    path = Path(job["metrics_path"])
    metrics = json.loads(path.read_text(encoding="utf-8"))
    callback(metrics)
    path.write_text(json.dumps(metrics), encoding="utf-8")
    if refresh:
        job["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_c_numeric_check(metrics):
    return metrics["diagnostics"]["best_checkpoint_interventions"]["mean_c_numeric_check"]


def test_complete_report_has_all_conditional_contrasts_and_resources(tmp_path, monkeypatch):
    root, manifest = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = write_comparison(root, manifest)
    assert result["status"] == "passed" and result["test_evaluated"] is False
    contrasts = result["datasets"][0]["factorial_contrasts"]
    assert contrasts["c_given_w_off"]["percentage_points"] == pytest.approx(2.0)
    assert contrasts["c_given_w_on"]["percentage_points"] == pytest.approx(5.0)
    assert contrasts["w_given_c_fixed"]["percentage_points"] == pytest.approx(5.0)
    assert contrasts["w_given_c_relative"]["percentage_points"] == pytest.approx(8.0)
    assert contrasts["interaction"]["percentage_points"] == pytest.approx(3.0)
    assert all(path.read_bytes() == value for path, value in before.items())
    markdown = (root / "comparison.md").read_text(encoding="utf-8")
    assert "C | W off" in markdown and "Epoch p90" in markdown and "W-I Frobenius" in markdown
    with (root / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and float(rows[0]["interaction_pp"]) == pytest.approx(3.0)
    monkeypatch.setattr(
        report_module,
        "_current_source_hashes",
        lambda: manifest["sources"]["sha256"],
    )
    assert main([str(root)]) == 0


def test_mean_c_tolerance_miss_is_informational_and_keeps_factorial_contrasts(tmp_path):
    root, manifest = _fixture(tmp_path)

    def record_cuda_roundoff(metrics):
        _mean_c_numeric_check(metrics).update(
            within_declared_tolerance=False,
            logit_mean_absolute_delta=2.0e-6,
            logit_max_absolute_delta=3.0e-5,
            changed_prediction_fraction=1.0e-4,
        )

    _edit(manifest, record_cuda_roundoff)
    result = write_comparison(root, manifest)
    assert result["status"] == "passed"
    assert result["complete"] is True
    assert result["datasets"][0]["factorial_contrasts"] is not None


@pytest.mark.parametrize(
    "field",
    [
        "comparison",
        "role",
        "separate_full_graph_forwards",
        "within_declared_tolerance",
        "allclose_rtol",
        "allclose_atol",
        "logit_mean_absolute_delta",
        "logit_max_absolute_delta",
        "changed_prediction_fraction",
        "replacement_contracts",
    ],
)
def test_mean_c_numeric_contract_rejects_missing_fields(tmp_path, field):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda metrics: _mean_c_numeric_check(metrics).pop(field))
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comparison", "ones_c_vs_mean_c"),
        ("role", "integrity_gate"),
        ("separate_full_graph_forwards", False),
        ("separate_full_graph_forwards", 1),
        ("within_declared_tolerance", 1),
        ("within_declared_tolerance", "false"),
        ("allclose_rtol", 2.0e-5),
        ("allclose_rtol", -1.0e-5),
        ("allclose_rtol", float("nan")),
        ("allclose_atol", 2.0e-6),
        ("allclose_atol", -1.0e-6),
        ("allclose_atol", float("nan")),
        ("logit_mean_absolute_delta", -1.0e-6),
        ("logit_mean_absolute_delta", float("nan")),
        ("logit_max_absolute_delta", -1.0e-6),
        ("logit_max_absolute_delta", float("nan")),
        ("changed_prediction_fraction", -0.1),
        ("changed_prediction_fraction", 1.1),
        ("changed_prediction_fraction", float("nan")),
    ],
)
def test_mean_c_numeric_contract_rejects_invalid_values(tmp_path, field, value):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda metrics: _mean_c_numeric_check(metrics).update({field: value}))
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


def test_mean_c_numeric_contract_rejects_maximum_below_mean(tmp_path):
    root, manifest = _fixture(tmp_path)

    def invert_delta_order(metrics):
        _mean_c_numeric_check(metrics).update(
            logit_mean_absolute_delta=2.0e-5,
            logit_max_absolute_delta=1.0e-5,
        )

    _edit(manifest, invert_delta_order)
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


def test_mean_c_numeric_contract_rejects_legacy_passed_field(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda metrics: _mean_c_numeric_check(metrics).update(passed=True))
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize(
    "change",
    [
        "missing_mean_c",
        "missing_ones_c",
        "extra_contract",
        "missing_contract_field",
        "extra_contract_field",
        "wrong_mean_contract",
        "wrong_ones_contract",
        "unsatisfied",
        "non_boolean_satisfied",
        "wrong_layers_checked",
        "non_integer_layers_checked",
        "edge_counts_not_list",
        "edge_counts_wrong_length",
        "negative_edge_count",
        "non_integer_edge_count",
        "edge_counts_wrong_topology",
        "different_edge_counts",
    ],
)
def test_mean_c_replacement_contracts_fail_closed_on_missing_or_tampered_data(
    tmp_path, change
):
    root, manifest = _fixture(tmp_path)

    def tamper(metrics):
        contracts = _mean_c_numeric_check(metrics)["replacement_contracts"]
        if change == "missing_mean_c":
            contracts.pop("mean_c")
        elif change == "missing_ones_c":
            contracts.pop("ones_c")
        elif change == "extra_contract":
            contracts["shuffled_c"] = copy.deepcopy(contracts["ones_c"])
        elif change == "missing_contract_field":
            contracts["mean_c"].pop("contract")
        elif change == "extra_contract_field":
            contracts["mean_c"]["unexpected"] = True
        elif change == "wrong_mean_contract":
            contracts["mean_c"]["contract"] = "exact_one"
        elif change == "wrong_ones_contract":
            contracts["ones_c"]["contract"] = "graph_constant_positive"
        elif change == "unsatisfied":
            contracts["mean_c"]["satisfied"] = False
        elif change == "non_boolean_satisfied":
            contracts["mean_c"]["satisfied"] = 1
        elif change == "wrong_layers_checked":
            contracts["mean_c"]["layers_checked"] = 1
        elif change == "non_integer_layers_checked":
            contracts["mean_c"]["layers_checked"] = True
        elif change == "edge_counts_not_list":
            contracts["mean_c"]["edge_counts"] = "3,3"
        elif change == "edge_counts_wrong_length":
            contracts["mean_c"]["edge_counts"] = [3]
        elif change == "negative_edge_count":
            contracts["mean_c"]["edge_counts"] = [3, -1]
        elif change == "non_integer_edge_count":
            contracts["mean_c"]["edge_counts"] = [3, True]
        elif change == "edge_counts_wrong_topology":
            contracts["mean_c"]["edge_counts"] = [4, 4]
            contracts["ones_c"]["edge_counts"] = [4, 4]
        else:
            contracts["ones_c"]["edge_counts"] = [3, 4]

    _edit(manifest, tamper)
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_partial_matrix_withholds_all_contrasts(tmp_path, status):
    root, manifest = _fixture(tmp_path)
    manifest["status"] = "failed" if status == "failed" else "running"
    manifest["jobs"][-1]["status"] = status
    result = write_comparison(root, manifest)
    assert not result["complete"]
    assert result["datasets"][0]["factorial_contrasts"] is None


@pytest.mark.parametrize("status", ["running", "failed"])
def test_complete_children_withhold_contrasts_until_manifest_passes(tmp_path, status):
    root, manifest = _fixture(tmp_path)
    manifest["status"] = status
    result = write_comparison(root, manifest)
    assert not result["complete"]
    assert result["status"] == status
    assert result["datasets"][0]["factorial_contrasts"] is None


@pytest.mark.parametrize(
    "change",
    [
        "metric_digest",
        "initial_hash",
        "gate_mode",
        "spatial_mode",
        "test",
        "source",
        "numeric_check",
        "initial_test_split",
        "final_test_split",
        "intervention_layers",
        "intervention_kind",
        "normalization_recomputed",
        "fixed_c_value",
        "fixed_w_value",
    ],
)
def test_tampering_or_mismatched_factor_metadata_withholds_contrasts(tmp_path, change):
    root, manifest = _fixture(tmp_path)
    if change == "metric_digest":
        _edit(manifest, lambda metrics: metrics.update(validation=0.9), refresh=False)
    elif change == "initial_hash":
        _edit(manifest, lambda metrics: metrics.update(initial_state_sha256="b" * 64))
    elif change == "gate_mode":
        _edit(manifest, lambda metrics: metrics.update(gate_mode="relative"))
    elif change == "spatial_mode":
        _edit(manifest, lambda metrics: metrics.update(spatial_mode="learned"))
    elif change == "test":
        _edit(manifest, lambda metrics: metrics.update(test_evaluated=True))
    elif change == "source":
        _edit(manifest, lambda metrics: metrics.update(source_sha256={"x.py": "f" * 64}))
    elif change == "numeric_check":
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["best_checkpoint_interventions"][
                "mean_c_numeric_check"
            ].update(passed=False),
        )
    elif change == "initial_test_split":
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["initial_validation"].update(split="test"),
        )
    elif change == "final_test_split":
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["final_validation"].update(split="test"),
        )
    elif change == "intervention_layers":
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["best_checkpoint_interventions"].update(
                layers="one_layer_only"
            ),
        )
    elif change == "intervention_kind":
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["best_checkpoint_interventions"]["rows"][
                0
            ].update(intervention_kind="retrained"),
        )
    elif change == "normalization_recomputed":
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["best_checkpoint_interventions"].update(
                normalization_recomputed_for_c_interventions=False
            ),
        )
    elif change == "fixed_c_value":
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["best_validation"]["layers"][0][
                "conductance"
            ].update(mean=0.9),
        )
    else:
        _edit(
            manifest,
            lambda metrics: metrics["diagnostics"]["best_validation"]["layers"][0][
                "spatial_weight"
            ].update(identity_distance_frobenius=0.1),
        )
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)
    report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert report["datasets"][0]["factorial_contrasts"] is None


def test_standalone_report_rejects_changed_current_sources(tmp_path, monkeypatch):
    root, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        report_module,
        "_current_source_hashes",
        lambda: {"changed.py": "f" * 64},
    )
    assert main([str(root)]) == 1
    report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert report["datasets"][0]["factorial_contrasts"] is None


def test_manifest_requires_every_unique_factorial_cell(tmp_path):
    root, manifest = _fixture(tmp_path)
    broken = copy.deepcopy(manifest)
    broken["jobs"].pop()
    with pytest.raises(ComparisonIntegrityError, match="four-arm"):
        write_comparison(root, broken)


def test_report_destinations_reject_symlinks_before_writing(tmp_path, monkeypatch):
    root, manifest = _fixture(tmp_path)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name == "comparison.md" or original(path),
    )
    with pytest.raises(ValueError, match="symlinks"):
        write_comparison(root, manifest)
    assert not (root / "comparison.json").exists()
