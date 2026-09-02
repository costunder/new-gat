"""V3 artifact-integrity fixtures only; no datasets or trained models."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from research.conductance_gat.v3.protocol import COMMON, CONDITIONS, PARAMETERIZATION, SUITE
from research.conductance_gat.v3.report import ComparisonIntegrityError, main, write_comparison


def _fixture(tmp_path, datasets=("ogbn-arxiv",), edges=3):
    root = tmp_path / "relative-c-v3"
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
        "edge_chunk_size": 65536,
    }
    source = {"research/conductance_gat/v3/model.py": "1" * 64}
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
        is_ppi = dataset == "ppi"
        child_batch_size = 2 if is_ppi else 1
        metric_name = "micro_f1" if is_ppi else "accuracy"
        prediction_unit = "node_label_decision" if is_ppi else "node"
        validation_graph_count = 2 if is_ppi else 1
        optimizer_steps_per_epoch = 10 if is_ppi else 1
        child_configuration = copy.deepcopy(configuration)
        child_configuration["batch_size"] = child_batch_size
        protocol = {
            "data_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
            "dataset": dataset,
            "split": (
                "official_inductive_graph_split"
                if is_ppi
                else "official_time_split"
                if dataset == "ogbn-arxiv"
                else "official_public_masks"
            ),
            "task": "multi_label_node_classification" if is_ppi else "node_classification",
            "metric": metric_name,
        }
        if is_ppi:
            protocol.update(
                split_counts={"train": 20, "validation": 2, "test": 2},
            )
        topology = (
            {
                "scope": "official_train_and_validation_graphs",
                "split_graph_counts": {"train": 20, "validation": 2},
                "split_num_nodes": {"train": 40, "validation": 4},
                "split_num_edges": {"train": 30, "validation": 3},
                "split_incidence_sha256": {"train": "2" * 64, "validation": "3" * 64},
            }
            if is_ppi
            else {"num_nodes": 4, "num_edges": edges, "incidence_sha256": "2" * 64}
        )
        for condition, spec in CONDITIONS.items():
            output = root / dataset / condition
            output.mkdir(parents=True)
            checkpoint, history = output / "best.pt", output / "history.json"
            checkpoint.write_bytes(b"unit-fixture-not-a-trained-model")
            history.write_text("[]", encoding="utf-8")
            data_hash = hashlib.sha256(dataset.encode()).hexdigest()
            relative = condition == "relative_c"
            frozen = 0 if relative else 98
            alpha = [f"operators.{i}.raw_alpha" for i in range(2)]
            gamma_tau = [
                f"operators.{i}.estimator.raw_{key}" for i in range(2) for key in ("gamma", "tau")
            ]
            gate = [f"operators.{i}.estimator.network.weight" for i in range(2)]
            controls = alpha + gamma_tau if relative else alpha
            names = ["encoder.weight"] + controls + (gate if relative else [])
            groups = [
                {
                    "name": "backbone",
                    "lr": COMMON["lr"],
                    "weight_decay": COMMON["weight_decay"],
                    "parameter_names": ["encoder.weight"],
                    "parameter_count": 100,
                },
                {
                    "name": "controls",
                    "lr": COMMON["lr"],
                    "weight_decay": 0.0,
                    "parameter_names": controls,
                    "parameter_count": len(controls),
                },
            ]
            if relative:
                groups.append(
                    {
                        "name": "gate_mlp",
                        "lr": COMMON["lr"] * COMMON["gate_lr_multiplier"],
                        "weight_decay": 0.0,
                        "parameter_names": gate,
                        "parameter_count": 94,
                    }
                )
            score = 0.55 if relative else 0.50
            layer_stats = [
                {
                    "layer": i,
                    "alpha": 0.5,
                    "gamma": 0.25,
                    "tau": 1.0,
                    "score": {"mean": 0.0, "std": 0.1},
                    "conductance": {"cv": 0.05},
                    "log_conductance": {"std": 0.08},
                    "weighted_degree": {
                        "quantiles": {"p50": 2.0, "p99": 3.0},
                        "max_over_median": 1.5,
                    },
                    "relative_conv_change": 0.4,
                    "gate_parameter_norm": 1.0,
                    "gate_gradient_norm": None,
                }
                for i in range(COMMON["layers"])
            ]
            interventions = {
                "status": "passed",
                "scope": "validation_selected_best_checkpoint_only",
                "original": {"validation": score, "loss": 0.7},
                "validation_graph_count": validation_graph_count,
                "prediction_unit": prediction_unit,
                "prediction_rule": (
                    "logit_gt_zero_node_label" if is_ppi else "argmax_node_class"
                ),
                "rows": [
                    {
                        "intervention": name,
                        "validation": score - 0.01,
                        "percentage_points": -1.0,
                        "logit_mean_absolute_delta": 0.1,
                        "changed_prediction_fraction": 0.02,
                        "prediction_unit": prediction_unit,
                        "prediction_rule": (
                            "logit_gt_zero_node_label" if is_ppi else "argmax_node_class"
                        ),
                    }
                    for name in ("mean_c", "shuffled_c", "ones_c", "propagation_off")
                ],
            }
            metrics = {
                "schema_version": 1,
                "research_suite": SUITE,
                "status": "passed",
                "dataset": dataset,
                "condition": condition,
                "model_seed": 0,
                **spec,
                "non_gate_weight_decay": 0.0005,
                "configuration": child_configuration,
                "cache_sha256": data_hash,
                "protocol": protocol,
                "initial_state_sha256": "a" * 64,
                "best_epoch": 10,
                "epochs_run": 30,
                "validation": 0.55 if condition == "relative_c" else 0.50,
                "metric_name": metric_name,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "optimizer_steps": 30 * optimizer_steps_per_epoch,
                "best_checkpoint_optimizer_steps": 10 * optimizer_steps_per_epoch,
                "train_loss": 0.7,
                "elapsed_seconds": 2.0,
                "peak_cuda_allocated_bytes": 100,
                "evaluation_split": "validation",
                "test_evaluated": False,
                "versions": {"torch": "unit-fixture"},
                "gpu": "unit-fixture",
                "total_parameters": 200,
                "trainable_parameters": 200 - frozen,
                "frozen_parameters": frozen,
                "optimizer": "AdamW",
                "optimizer_groups": groups,
                "trainable_parameter_names": names,
                "frozen_parameter_names": [] if relative else gate + gamma_tau,
                "diagnostics": {
                    "best_validation": {
                        "mode": "eval",
                        "split": "validation",
                        "metric": score,
                        "metric_name": metric_name,
                        "prediction_rule": (
                            "logit_gt_zero_node_label" if is_ppi else "argmax_node_class"
                        ),
                        "loss": 0.7,
                        "validation_graph_count": validation_graph_count,
                        "prediction_unit": prediction_unit,
                        "layers": layer_stats,
                    },
                    "best_checkpoint_interventions": interventions,
                    "train_trajectory": [
                        {
                            "epoch": 10,
                            "batch_index": 0,
                            "optimizer_steps_before_batch": 9 * optimizer_steps_per_epoch,
                            "scope": (
                                "first_actual_training_minibatch_only"
                                if is_ppi
                                else "full_graph_train_mask"
                            ),
                            "mode": "train_dropout_on",
                            "stage": "after_task_backward_before_optimizer_step",
                            "layers": [
                                {"layer": i, "gate_gradient_norm": 0.125 if relative else None}
                                for i in range(COMMON["layers"])
                            ],
                        }
                    ],
                },
                "source_sha256": source,
                "parameterization": PARAMETERIZATION,
                "topology": topology,
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
                    "batch_size": child_batch_size,
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
        assert dataset["relative_minus_fixed"]["percentage_points"] == pytest.approx(5.0)
        assert dataset["held_fixed"]["topology"]["num_edges"] == 3
    assert all(path.read_bytes() == value for path, value in before.items())
    text = (root / "comparison.md").read_text(encoding="utf-8")
    assert "+5.000000 pp" in text and "Peak CUDA" in text and "Train loss" in text
    assert "not a measured" in text and "scalability result" in text
    with (root / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and float(rows[0]["relative_minus_fixed_pp"]) == pytest.approx(5.0)
    assert main([str(root)]) == 0


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_partial_results_have_no_contrast(tmp_path, status):
    root, manifest = _fixture(tmp_path)
    manifest["status"] = "failed" if status == "failed" else "running"
    manifest["jobs"][1]["status"] = status
    result = write_comparison(root, manifest)
    assert not result["complete"] and result["datasets"][0]["relative_minus_fixed"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("cache_sha256", "b" * 64),
        ("initial_state_sha256", "b" * 64),
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
    assert all(dataset["relative_minus_fixed"] is None for dataset in result["datasets"])


@pytest.mark.parametrize(
    "change", ["source", "missing_source", "duplicate", "missing", "spec", "unknown"]
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
    elif change == "unknown":
        manifest["config"]["datasets"] = ["unknown"]
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


def test_fixed_frozen_count_must_match_optimizer(tmp_path):
    root, manifest = _fixture(tmp_path)
    _edit(
        manifest, lambda child: child.update(frozen_parameters=5, trainable_parameters=195), index=1
    )
    with pytest.raises(ComparisonIntegrityError, match="parameter counts"):
        write_comparison(root, manifest)


def test_empty_edges_do_not_remove_shared_frozen_scaffold(tmp_path):
    root, manifest = _fixture(tmp_path, edges=0)
    assert write_comparison(root, manifest)["status"] == "passed"


@pytest.mark.parametrize("key", ["optimizer_groups", "trainable_parameter_names", "diagnostics"])
def test_required_v3_provenance_is_not_optional(tmp_path, key):
    root, manifest = _fixture(tmp_path)
    _edit(manifest, lambda child: child.pop(key))
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize("mutation", ["alpha", "duplicate", "group_lr", "group_decay"])
def test_optimizer_contract_rejects_wrong_learning_policy(tmp_path, mutation):
    root, manifest = _fixture(tmp_path)

    def change(child):
        groups = child["optimizer_groups"]
        if mutation == "alpha":
            groups[1]["parameter_names"] = ["operators.0.estimator.raw_gamma"]
        elif mutation == "duplicate":
            groups.append(copy.deepcopy(groups[0]))
        elif mutation == "group_lr":
            groups[-1]["lr"] = COMMON["lr"]
        else:
            groups[1]["weight_decay"] = 0.0005

    _edit(manifest, change)
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize(
    "mutation", ["missing", "wrong_delta", "wrong_original", "duplicate", "wrong_scope"]
)
def test_intervention_integrity_rejected(tmp_path, mutation):
    root, manifest = _fixture(tmp_path)

    def change(child):
        audit = child["diagnostics"]["best_checkpoint_interventions"]
        if mutation == "missing":
            audit["rows"].pop()
        elif mutation == "wrong_delta":
            audit["rows"][0]["percentage_points"] = 123
        elif mutation == "wrong_original":
            audit["original"]["validation"] = 0.4
        elif mutation == "duplicate":
            audit["rows"][0] = copy.deepcopy(audit["rows"][1])
        else:
            audit["scope"] = "test"

    _edit(manifest, change)
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


def test_report_uses_relative_scalars_and_interventions_not_legacy_rho(tmp_path):
    root, manifest = _fixture(tmp_path)
    write_comparison(root, manifest)
    text = (root / "comparison.md").read_text(encoding="utf-8")
    assert "alpha" in text and "gamma" in text and "tau" in text
    assert "shuffled_c" in text and "propagation_off" in text
    assert "| ρ mean |" not in text
    assert "single-factor comparison" in text
    assert "Validation computes no gradients" in text
    assert "Actual training gradients" in text and "0.125000" in text


def test_ppi_report_uses_micro_f1_and_official_inductive_contract(tmp_path):
    root, manifest = _fixture(tmp_path, ("ppi",))
    report = write_comparison(root, manifest)
    assert report["status"] == "passed"
    assert report["datasets"][0]["metric_name"] == "micro_f1"
    assert report["datasets"][0]["held_fixed"]["topology"]["split_graph_counts"] == {
        "train": 20,
        "validation": 2,
    }
    assert "ppi (micro_f1" in (root / "comparison.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value"),
    [("dataset", "cora"), ("task", "node_classification"), ("metric", "accuracy")],
)
def test_ppi_report_rejects_wrong_cached_task_contract(tmp_path, field, value):
    root, manifest = _fixture(tmp_path, ("ppi",))
    _edit(manifest, lambda child: child["protocol"].update({field: value}))
    with pytest.raises(ComparisonIntegrityError, match="official V1 dataset contract"):
        write_comparison(root, manifest)


def test_ppi_report_rejects_wrong_prediction_threshold_contract(tmp_path):
    root, manifest = _fixture(tmp_path, ("ppi",))
    _edit(
        manifest,
        lambda child: child["diagnostics"]["best_validation"].update(
            prediction_rule="argmax_node_class"
        ),
    )
    with pytest.raises(ComparisonIntegrityError, match="prediction scope mismatch"):
        write_comparison(root, manifest)


@pytest.mark.parametrize("value", ["not-a-number", False, -1.0])
def test_malformed_optional_diagnostic_is_invalid_not_rendered(tmp_path, value):
    root, manifest = _fixture(tmp_path)
    _edit(
        manifest,
        lambda child: child["diagnostics"]["best_validation"]["layers"][0]["conductance"].update(
            cv=value
        ),
    )
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "scope", "stage", "no_grad"])
def test_actual_training_gradient_provenance_is_required(tmp_path, mutation):
    root, manifest = _fixture(tmp_path)

    def change(child):
        records = child["diagnostics"]["train_trajectory"]
        if mutation == "missing":
            records.clear()
        elif mutation == "duplicate":
            records.append(copy.deepcopy(records[0]))
        elif mutation == "no_grad":
            records[0]["layers"][0]["gate_gradient_norm"] = None
        else:
            records[0][mutation] = "validation"

    _edit(manifest, change)
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)
