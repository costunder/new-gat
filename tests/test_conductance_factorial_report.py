from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from research.conductance_gat.ablation.protocol import CONDITIONS
from research.conductance_gat.ablation.report import (
    ComparisonIntegrityError,
    main,
    write_comparison,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, datasets: tuple[str, ...] = ("ppi",)) -> tuple[Path, dict]:
    root = tmp_path / "factorial"
    root.mkdir()
    common = {
        "model_seed": 0,
        "datasets": list(datasets),
        "epochs": 100,
        "patience": 20,
        "batch_size": 2,
        "workers": 0,
        "device": "cuda",
        "data_root": str(tmp_path / "data"),
        "hidden_channels": 64,
        "layers": 2,
        "dropout": 0.5,
        "lr": 0.005,
        "weight_decay": 0.0005,
        "amp": False,
        "compile": False,
    }
    configuration = {
        key: value for key, value in common.items() if key not in {"datasets", "data_root"}
    } | {"tf32": False, "pin_memory": True}
    manifest = {
        "schema_version": 1,
        "suite": "conductance_factorial",
        "status": "passed",
        "config": common,
        "jobs": [],
    }
    scores = (0.5, 0.6, 0.7, 0.9)
    for dataset_index, dataset in enumerate(datasets):
        for index, (condition, factors) in enumerate(CONDITIONS.items()):
            output = root / dataset / condition
            metrics_path = output / "metrics.json"
            metrics = {
                "schema_version": 1,
                "research_suite": "conductance_factorial",
                "status": "passed",
                "dataset": dataset,
                "condition": condition,
                "model_seed": 0,
                "normalization": factors["normalization"],
                "gate_weight_decay": factors["gate_weight_decay"],
                "non_gate_weight_decay": 0.0005,
                "configuration": copy.deepcopy(configuration),
                "cache_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
                "protocol": {"dataset": dataset, "split_seed": 0, "source": "official"},
                "initial_state_sha256": hashlib.sha256(f"init-{dataset}".encode()).hexdigest(),
                "best_epoch": 10 + index,
                "epochs_run": 30 + index,
                "validation": scores[index] - dataset_index * 0.1,
                "metric_name": "micro_f1" if dataset == "ppi" else "accuracy",
                "train_loss": 0.7,
                "checkpoint": str(output / "checkpoint.pt"),
                "checkpoint_sha256": hashlib.sha256(b"unit-fixture-no-model").hexdigest(),
                "history": str(output / "history.json"),
                "history_sha256": hashlib.sha256(b"[]").hexdigest(),
                "elapsed_seconds": 10.0 + index,
                "peak_cuda_allocated_bytes": 4096,
                "evaluation_split": "validation",
                "test_evaluated": False,
                "diagnostics": {"final_validation": {"mean_rho": 0.5 + 0.1 * index}},
            }
            _write_json(metrics_path, metrics)
            (output / "checkpoint.pt").write_bytes(b"unit-fixture-no-model")
            _write_json(output / "history.json", [])
            manifest["jobs"].append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "status": "passed",
                    "output_dir": str(output),
                    "metrics_path": str(metrics_path),
                }
            )
    _write_json(root / "manifest.json", manifest)
    return root, manifest


def _edit_child(manifest: dict, index: int, edit) -> None:
    path = Path(manifest["jobs"][index]["metrics_path"])
    metrics = json.loads(path.read_text(encoding="utf-8"))
    edit(metrics)
    _write_json(path, metrics)


def test_complete_comparison_has_paired_factor_effects_and_interaction(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    report = write_comparison(root, manifest)
    assert report["status"] == "passed"
    assert report["complete"] is True
    assert report["n_model_seeds"] == 1
    assert report["uncertainty_status"] == "not_estimated_single_seed"
    effects = report["datasets"][0]["effects"]
    expected = {
        "gate_effect_at_global_max": 0.1,
        "normalization_effect_with_gate_wd": 0.2,
        "gate_effect_at_node_degree": 0.2,
        "normalization_effect_without_gate_wd": 0.3,
        "interaction": 0.1,
    }
    for key, delta in expected.items():
        assert effects[key]["score_delta"] == pytest.approx(delta)
        assert effects[key]["percentage_points"] == pytest.approx(100.0 * delta)
    baseline = report["datasets"][0]["conditions"][0]
    assert baseline["delta_from_baseline"]["score_delta"] == 0.0
    assert json.loads((root / "comparison.json").read_text(encoding="utf-8")) == report
    markdown = (root / "comparison.md").read_text(encoding="utf-8")
    assert "n=1" in markdown and "test not evaluated" in markdown
    assert "uniform conductance rescaling still cancels" in markdown
    assert "50.000000" in markdown and "+10.000000" in markdown
    assert not {"sample_std", "confidence_interval", "p_value"}.intersection(effects)
    with (root / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert [row["row_type"] for row in rows].count("condition") == 4
    assert [row["row_type"] for row in rows].count("contrast") == 5


def test_datasets_keep_separate_metrics_and_no_cross_dataset_mean(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path, ("ppi", "ogbn-arxiv"))
    report = write_comparison(root, manifest)
    assert [(row["dataset"], row["metric_name"]) for row in report["datasets"]] == [
        ("ppi", "micro_f1"),
        ("ogbn-arxiv", "accuracy"),
    ]
    assert report["datasets"][0]["conditions"][0]["validation"] == 0.5
    assert report["datasets"][1]["conditions"][0]["validation"] == 0.4
    assert "mean" not in report and "pooled" not in report


@pytest.mark.parametrize("job_status", ["pending", "running", "failed"])
def test_partial_comparison_has_no_fabricated_deltas(tmp_path: Path, job_status: str) -> None:
    root, manifest = _fixture(tmp_path)
    manifest["status"] = "failed" if job_status == "failed" else "running"
    manifest["jobs"][3]["status"] = job_status
    manifest["jobs"][3]["error"] = "CUDA OOM" if job_status == "failed" else None
    report = write_comparison(root, manifest)
    assert report["status"] == ("failed" if job_status == "failed" else "running")
    assert report["complete"] is False
    assert report["datasets"][0]["effects"] is None
    assert all(row["delta_from_baseline"] is None for row in report["datasets"][0]["conditions"])
    assert report["datasets"][0]["conditions"][3]["validation"] is None
    assert "Contrasts withheld" in (root / "comparison.md").read_text(encoding="utf-8")


def test_missing_matrix_job_is_explicit_and_prevents_completion(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    manifest["status"] = "running"
    manifest["jobs"].pop()
    report = write_comparison(root, manifest)
    assert report["datasets"][0]["conditions"][-1]["status"] == "missing"
    assert report["complete"] is False and report["datasets"][0]["effects"] is None
    manifest["status"] = "passed"
    with pytest.raises(ComparisonIntegrityError, match="complete four-condition matrix"):
        write_comparison(root, manifest)


def test_source_change_invalidates_completed_contrasts(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path, ("ppi", "ogbn-arxiv"))
    write_comparison(root, manifest)
    manifest["status"] = "failed"
    manifest["source_integrity_valid"] = False
    with pytest.raises(ComparisonIntegrityError, match="Source integrity failed") as caught:
        write_comparison(root, manifest)
    assert caught.value.report["source_integrity_valid"] is False
    assert all(dataset["effects"] is None for dataset in caught.value.report["datasets"])
    assert all(
        row["delta_from_baseline"] is None
        for dataset in caught.value.report["datasets"]
        for row in dataset["conditions"]
    )


def test_ordinary_job_failure_keeps_independent_completed_dataset_contrasts(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path, ("ppi", "ogbn-arxiv"))
    manifest["status"] = "failed"
    manifest["source_integrity_valid"] = True
    manifest["jobs"][-1]["status"] = "failed"
    report = write_comparison(root, manifest)
    assert report["status"] == "failed" and report["complete"] is False
    assert report["datasets"][0]["effects"] is not None
    assert report["datasets"][1]["effects"] is None


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dataset", "cora", "dataset mismatch"),
        ("model_seed", 1, "model_seed mismatch"),
        ("normalization", "global_max", "normalization mismatch"),
        ("gate_weight_decay", 0.0005, "gate_weight_decay mismatch"),
        ("non_gate_weight_decay", 0.0, "non_gate_weight_decay mismatch"),
        ("metric_name", "accuracy", "metric_name mismatch"),
        ("test_evaluated", True, "test_evaluated mismatch"),
        ("evaluation_split", "test", "evaluation_split mismatch"),
        ("schema_version", True, "schema_version mismatch"),
        ("status", "running", "status mismatch"),
        ("cache_sha256", "f" * 64, "held-fixed cache_sha256"),
        ("initial_state_sha256", "e" * 64, "held-fixed initial_state_sha256"),
        ("protocol", {"split_seed": 1}, "held-fixed protocol"),
        ("cache_sha256", "not-a-digest", "must be a SHA-256 digest"),
        ("validation", 50.0, "score in"),
        ("validation", True, "finite number"),
        ("best_epoch", 35, "epoch budget"),
        ("best_epoch", 0, "integer >= 1"),
        ("epochs_run", 101, "epoch budget"),
        ("train_loss", -1.0, "cannot be negative"),
    ],
)
def test_mismatched_child_metadata_fails_closed(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    root, manifest = _fixture(tmp_path)
    write_comparison(root, manifest)  # Ensure invalid output replaces a prior successful report.
    _edit_child(manifest, 3, lambda child: child.update({key: value}))
    with pytest.raises(ComparisonIntegrityError, match=message) as caught:
        write_comparison(root, manifest)
    report = caught.value.report
    assert report["status"] == "invalid"
    assert report["complete"] is False
    assert report["datasets"][0]["effects"] is None
    assert all(row["delta_from_baseline"] is None for row in report["datasets"][0]["conditions"])
    saved = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert saved["status"] == "invalid"
    assert "Integrity errors" in (root / "comparison.md").read_text(encoding="utf-8")


def test_training_configuration_must_match_manifest_and_other_children(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    _edit_child(manifest, 1, lambda child: child["configuration"].update({"lr": 0.1}))
    with pytest.raises(ComparisonIntegrityError, match="configuration.lr differs from manifest"):
        write_comparison(root, manifest)
    _edit_child(manifest, 1, lambda child: child["configuration"].update({"lr": 0.005}))
    _edit_child(manifest, 1, lambda child: child["configuration"].update({"pin_memory": False}))
    with pytest.raises(ComparisonIntegrityError, match="held-fixed configuration"):
        write_comparison(root, manifest)


def test_nonfinite_score_is_rejected_and_invalid_report_is_valid_json(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    _edit_child(manifest, 2, lambda child: child.update(validation=float("nan")))
    with pytest.raises(ComparisonIntegrityError):
        write_comparison(root, manifest)
    saved = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert saved["status"] == "invalid"
    assert saved["datasets"][0]["conditions"][2]["validation"] is None


def test_nonfinite_extra_metadata_fails_closed(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    _edit_child(
        manifest,
        2,
        lambda child: child["configuration"].update({"extra_value": float("inf")}),
    )
    with pytest.raises(ComparisonIntegrityError, match="nonfinite JSON value"):
        write_comparison(root, manifest)
    saved = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert saved["status"] == "invalid" and saved["datasets"][0]["effects"] is None


def test_manifest_cannot_redefine_fixed_nonfactor_hyperparameters(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    manifest["config"]["hidden_channels"] = 128
    for index in range(4):
        _edit_child(
            manifest,
            index,
            lambda child: child["configuration"].update({"hidden_channels": 128}),
        )
    with pytest.raises(ComparisonIntegrityError, match="hidden_channels must be 64"):
        write_comparison(root, manifest)


def test_duplicate_job_and_unknown_condition_are_rejected(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    manifest["jobs"].append(copy.deepcopy(manifest["jobs"][0]))
    with pytest.raises(ComparisonIntegrityError, match="duplicate manifest job"):
        write_comparison(root, manifest)
    manifest["jobs"][-1]["condition"] = "undeclared-model"
    with pytest.raises(ComparisonIntegrityError, match="unknown condition"):
        write_comparison(root, manifest)


@pytest.mark.parametrize("field", ["output_dir", "metrics_path"])
def test_manifest_artifact_paths_cannot_escape_run_root(tmp_path: Path, field: str) -> None:
    root, manifest = _fixture(tmp_path)
    outside = tmp_path / "outside" / "metrics.json"
    _write_json(outside, {"untouched": True})
    before = outside.read_bytes()
    manifest["jobs"][1][field] = str(outside)
    with pytest.raises(ComparisonIntegrityError, match="escapes"):
        write_comparison(root, manifest)
    assert outside.read_bytes() == before


@pytest.mark.parametrize("field", ["checkpoint", "history"])
def test_child_artifact_paths_cannot_escape_own_job(tmp_path: Path, field: str) -> None:
    root, manifest = _fixture(tmp_path)
    _edit_child(manifest, 1, lambda child: child.update({field: str(root / "other.bin")}))
    with pytest.raises(ComparisonIntegrityError, match="escapes"):
        write_comparison(root, manifest)


def test_metrics_cannot_point_at_different_child_output(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    manifest["jobs"][1]["metrics_path"] = manifest["jobs"][0]["metrics_path"]
    with pytest.raises(ComparisonIntegrityError, match="output_dir"):
        write_comparison(root, manifest)


@pytest.mark.parametrize("field", ["checkpoint", "history"])
def test_child_artifact_hash_must_match(tmp_path: Path, field: str) -> None:
    root, manifest = _fixture(tmp_path)
    metric_path = Path(manifest["jobs"][0]["metrics_path"])
    artifact = Path(json.loads(metric_path.read_text(encoding="utf-8"))[field])
    artifact.write_bytes(b"changed after training")
    with pytest.raises(ComparisonIntegrityError, match=f"{field} SHA-256 mismatch"):
        write_comparison(root, manifest)


def test_report_destination_symlink_is_not_followed(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("unchanged", encoding="utf-8")
    try:
        (root / "comparison.json").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires OS support/Windows developer privilege")
    with pytest.raises(ValueError, match="escapes|symlink"):
        write_comparison(root, manifest)
    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_report_symlink_guard_runs_before_any_writes_on_all_platforms(
    tmp_path: Path, monkeypatch
) -> None:
    root, manifest = _fixture(tmp_path)
    (root / "comparison.json").write_text("original", encoding="utf-8")
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == root / "comparison.md" or original(path),
    )
    with pytest.raises(ValueError, match="report destinations must not be symlinks"):
        write_comparison(root, manifest)
    assert (root / "comparison.json").read_text(encoding="utf-8") == "original"
    assert not (root / "comparison.csv").exists()


def test_cli_regenerates_without_mutating_children_or_manifest(tmp_path: Path, capsys) -> None:
    root, manifest = _fixture(tmp_path)
    watched = [root / "manifest.json"] + [Path(job["metrics_path"]) for job in manifest["jobs"]]
    before = {path: path.read_bytes() for path in watched}
    original = copy.deepcopy(manifest)
    assert main([str(root)]) == 0
    assert "single-seed 2x2" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in watched} == before
    assert manifest == original
    assert main([str(root)]) == 0
    assert {path: path.read_bytes() for path in watched} == before


def test_cli_incomplete_run_returns_nonzero_not_success(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    manifest["status"] = "running"
    manifest["jobs"][0]["status"] = "pending"
    _write_json(root / "manifest.json", manifest)
    assert main([str(root)]) == 2


def test_different_early_stop_epochs_and_diagnostics_are_allowed(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    report = write_comparison(root, manifest)
    conditions = report["datasets"][0]["conditions"]
    assert len({condition["best_epoch"] for condition in conditions}) == 4
    assert len({condition["epochs_run"] for condition in conditions}) == 4
    assert conditions[1]["diagnostics"]["final_validation"]["mean_rho"] == 0.6


def test_unknown_configuration_fields_are_not_silently_ignored(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    _edit_child(manifest, 1, lambda child: child["configuration"].update({"mystery_lr": 1.0}))
    with pytest.raises(ComparisonIntegrityError, match="held-fixed configuration"):
        write_comparison(root, manifest)


def test_tf32_cannot_silently_be_enabled(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    for index in range(4):
        _edit_child(manifest, index, lambda child: child["configuration"].update({"tf32": True}))
    with pytest.raises(ComparisonIntegrityError, match="tf32 must explicitly be False"):
        write_comparison(root, manifest)


def test_diagnostics_average_within_graph_cv_not_pooled_or_edge_weighted(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    diagnostic = {
        "best_validation": {
            "split": "validation",
            "mode": "eval",
            "layers": [
                {
                    "layer": 0,
                    "conductance": {"cv": 42.0},
                    "rho": {"mean": 0.99},
                    "relative_conv_change": 0.999,
                    "graphs": [
                        {
                            "conductance": {"count": 1, "mean": 1.0, "cv": 0.2},
                            "rho": {"count": 5, "mean": 0.1},
                            "relative_conv_change": 0.2,
                        },
                        {
                            "conductance": {"count": 100, "mean": 100.0, "cv": 0.8},
                            "rho": {"count": 1000, "mean": 0.9},
                            "relative_conv_change": 0.6,
                        },
                    ],
                },
                {
                    "layer": 1,
                    "conductance": {"cv": 0.95},
                    "graphs": [
                        {"conductance": {"mean": 1.0, "cv": 0.0}},
                        {"conductance": {"mean": 100.0, "cv": 0.0}},
                    ],
                },
            ],
            "parameter_norms": {
                "operators.0.estimator.network.0.weight": 3.0,
                "operators.0.estimator.network.2.weight": 4.0,
                "operators.1.estimator.network.0.weight": 1e-16,
                "encoder.weight": 500.0,
            },
        }
    }
    _edit_child(manifest, 0, lambda child: child.update(diagnostics=diagnostic))
    report = write_comparison(root, manifest)
    summary = report["datasets"][0]["conditions"][0]["best_validation_diagnostics"]
    assert summary[0]["conductance_cv"] == {"mean": 0.5, "valid_graph_count": 2}
    assert summary[0]["rho_mean"]["mean"] == pytest.approx(0.5)
    assert summary[0]["relative_conv_change"]["mean"] == pytest.approx(0.4)
    assert summary[0]["gate_parameter_l2"] == 5.0
    assert summary[0]["gate_parameter_tensor_count"] == 2
    assert summary[1]["conductance_cv"]["mean"] == 0.0
    assert summary[1]["rho_mean"]["mean"] is None
    markdown = (root / "comparison.md").read_text(encoding="utf-8")
    assert "within each graph before averaging" in markdown
    assert "| baseline | 0 | 0.5 (2/2) | 0.5 (2/2) | 0.4 (2/2) | 5 |" in markdown
    assert "1e-16" in markdown  # Small gate norms must not be rounded to an apparent zero.


def test_missing_optional_diagnostics_are_unknown_not_zero_or_an_integrity_error(
    tmp_path: Path,
) -> None:
    root, manifest = _fixture(tmp_path)
    for index in range(4):
        _edit_child(manifest, index, lambda child: child.pop("diagnostics"))
    report = write_comparison(root, manifest)
    assert report["status"] == "passed" and report["datasets"][0]["effects"] is not None
    for condition in report["datasets"][0]["conditions"]:
        for layer in condition["best_validation_diagnostics"]:
            assert layer["graph_count"] == 0
            assert layer["conductance_cv"]["mean"] is None
            assert layer["rho_mean"]["mean"] is None
            assert layer["relative_conv_change"]["mean"] is None
            assert layer["gate_parameter_l2"] is None
    assert "| baseline | 0 | — | — | — | — |" in (root / "comparison.md").read_text(
        encoding="utf-8"
    )
