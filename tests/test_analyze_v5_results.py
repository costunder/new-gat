"""Synthetic JSON unit fixtures only: not real experiments or GPU measurements."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import analyze_v5_results as analyzer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def metrics(**changes):
    return {
        "schema_version": 1,
        "research_suite": analyzer.SUITE,
        "status": "passed",
        "dataset": "ppi",
        "condition": "shared_dynamic_c",
        "model_seed": 0,
        "configuration": {
            "conductance_backend": "optimization",
            "training_schedule": "joint",
            "batch_size": 8,
            "sample_seed_batch_size": 1024,
            "sampling": "full",
            "epochs": 200,
        },
        "validation": 0.8,
        "best_epoch": 41,
        "metric_name": "micro_f1",
        **changes,
    }


def history():
    return [
        {
            "epoch": epoch,
            "train_loss": loss,
            "validation": validation,
            "phase": {"phase": "joint"},
            "train_batches": 3,
            "optimizer_steps": epoch * 3,
            "effective_optimizer_steps_by_group": {"conductance": epoch * 3},
        }
        for epoch, loss, validation in [
            (41, 0.5, 0.8),
            (42, 0.4, 0.7),
            (43, 0.3, 0.6),
            (44, 0.2, 0.5),
        ]
    ]


def test_recorded_curves_keep_cumulative_epochs_updates_and_actual_batch(tmp_path):
    write_json(tmp_path / "history.json", history())
    write_json(
        tmp_path / "metrics.json",
        metrics(
            batch_observability={"configured_physical_batch_size": 8, "batch_unit": "graphs"},
            learning_budget={"policy": "reference_updates", "target_optimizer_steps": 600},
        ),
    )
    record = analyzer.analyze(tmp_path)["records"][0]
    assert record["history"]["first_epoch"] == 41
    assert record["history"]["observed_best_epoch"] == 41
    assert record["history"]["epochs_after_best"] == 3
    assert record["history"]["best_to_final_drop"] == pytest.approx(0.3)
    assert record["history"]["train_loss_delta"] == pytest.approx(-0.3)
    assert record["history"]["actual_train_batches_in_recorded_history"] == 12
    assert record["actual_optimizer_steps"] == 132
    assert record["actual_optimizer_steps_by_group"] == {"conductance": 132}
    assert record["recorded_batching"]["configured_physical_batch_size"] == 8
    assert record["learning_budget"]["target_optimizer_steps"] == 600
    assert any("first quarter" in note for note in record["notes"])
    assert any("hypothesis" in note for note in record["notes"])


def test_interventions_delta_sign_layers_and_first_gradient(tmp_path):
    layer = {
        "layer": 0,
        "conductance": {"mean": 1, "cv": 0.25},
        "score": {"std": 0.1},
        "beta": {"mean": 0.7, "min": 0.6, "max": 0.8},
        "conductance_backend": "mlp",
    }
    first_gradient = {"applicable": True, "passed": True, "layers": [{"total_gradient_norm": 0.01}]}
    write_json(
        tmp_path / "metrics.json",
        metrics(
            configuration={"conductance_backend": "mlp"},
            selected_checkpoint_interventions={
                "learned": {"metric": 0.8, "layers": [layer]},
                "c_one": {"metric": 0.7},
                "shuffled_c": {"metric": 0.6},
            },
            first_active_conductance_gradient=first_gradient,
        ),
    )
    record = analyzer.analyze(tmp_path)["records"][0]
    diagnostics = record["diagnostics"]
    assert record["conductance_backend"] == "mlp"
    assert diagnostics["interventions"]["c_one"]["delta_from_learned"] == pytest.approx(-0.1)
    assert diagnostics["interventions"]["shuffled_c"]["delta_from_learned"] == pytest.approx(-0.2)
    assert diagnostics["interventions"]["mean_c"]["metric"] == "unavailable"
    assert diagnostics["layers"][0]["c_cv"] == 0.25
    assert diagnostics["layers"][0]["score_std"] == 0.1
    assert diagnostics["layers"][0]["beta_mean"] == 0.7
    assert diagnostics["first_active_conductance_gradient"] == first_gradient
    assert "not a dataset-wide" in diagnostics["layer_scope"]


def test_missing_values_are_unavailable_and_legacy_not_current_backend(tmp_path):
    write_json(
        tmp_path / "metrics.json", metrics(configuration={}, validation=0.0, optimizer_steps=0)
    )
    record = analyzer.analyze(tmp_path)["records"][0]
    assert record["conductance_backend"] == "unspecified_legacy"
    assert record["history"]["recorded_epochs"] == "unavailable"
    assert record["actual_optimizer_steps"] == 0
    assert record["validation"] == 0.0
    assert record["diagnostics"]["first_active_conductance_gradient"] == "unavailable"
    assert any("must not be assigned" in note for note in record["notes"])


def test_history_checkpoint_selection_and_layer_fallback(tmp_path):
    rows = history()
    rows[-1]["layers"] = [{"layer": 2, "conductance": {"cv": 0}}]
    write_json(tmp_path / "history.json", rows)
    write_json(tmp_path / "metrics.json", metrics(best_epoch=43))
    record = analyzer.analyze(tmp_path)["records"][0]
    assert record["selected_epoch"] == 43
    assert record["history"]["observed_best_epoch"] == 41
    assert record["diagnostics"]["layers"][0]["c_cv"] == 0
    assert record["diagnostics"]["layer_source"] == "last recorded training history epoch"
    assert any("phase policy" in note for note in record["notes"])


def test_manifest_historical_pending_and_transition_are_separate(tmp_path):
    old = write_json(
        tmp_path / "old" / "metrics.json",
        metrics(condition="fixed_c", configuration={"conductance_backend": "mlp"}),
    )
    new = write_json(
        tmp_path / "new" / "metrics.json",
        metrics(
            transition_provenance={"mode": "replace_c", "source_epoch": 40},
            post_transition_optimizer_steps=12,
        ),
    )
    manifest = write_json(
        tmp_path / "manifest.json",
        {
            "jobs": [
                {
                    "dataset": "ppi",
                    "condition": "fixed_c",
                    "action": "reuse_completed_fixed",
                    "status": "historical_reference",
                    "historical_reference": {"metrics_path": str(old)},
                },
                {
                    "dataset": "ppi",
                    "condition": "shared_dynamic_c",
                    "action": "transition_dynamic",
                    "status": "passed",
                    "output_dir": str(new.parent),
                },
                {
                    "dataset": "cora",
                    "condition": "shared_dynamic_c",
                    "action": "fresh_dynamic",
                    "status": "pending",
                    "output_dir": str(tmp_path / "missing"),
                },
            ]
        },
    )
    report = analyzer.analyze(manifest)
    assert len(report["records"]) == 2
    roles = {record["condition"]: record["roles"] for record in report["records"]}
    assert roles["fixed_c"] == ["historical_reference"]
    assert roles["shared_dynamic_c"] == ["transitioned_training"]
    assert report["unavailable_jobs"][0]["status"] == "pending"
    assert report["unavailable_jobs"][0]["evidence"] == "unavailable"
    assert "new training completion" in report["unavailable_jobs"][0]["reason"]


def test_completed_old_dynamic_is_historical_not_new_solver_completion(tmp_path):
    old = write_json(
        tmp_path / "old" / "metrics.json", metrics(configuration={"conductance_backend": "mlp"})
    )
    write_json(
        tmp_path / "manifest.json",
        {
            "jobs": [
                {
                    "condition": "shared_dynamic_c",
                    "dataset": "ppi",
                    "action": "preserve_legacy_dynamic",
                    "status": "pending_extra_budget",
                    "historical_reference": {"metrics_path": str(old)},
                }
            ]
        },
    )
    report = analyzer.analyze(tmp_path)
    assert len(report["records"]) == 1
    assert report["records"][0]["roles"] == ["historical_reference"]
    assert report["records"][0]["conductance_backend"] == "mlp"


def test_unknown_suite_and_version_are_not_v5_results(tmp_path):
    write_json(tmp_path / "v2" / "metrics.json", metrics(research_suite="conductance_direct_v2"))
    write_json(tmp_path / "future" / "metrics.json", metrics(schema_version=999))
    actual = write_json(tmp_path / "v5" / "metrics.json", metrics())
    report = analyzer.analyze(tmp_path)
    assert [record["metrics_path"] for record in report["records"]] == [str(actual.resolve())]
    assert len(report["ignored_json"]) == 2


@pytest.mark.parametrize("raw", ["{broken", '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}'])
def test_corrupt_or_nonfinite_json_is_an_explicit_path_error(tmp_path, raw, capsys):
    path = tmp_path / "metrics.json"
    path.write_text(raw, encoding="utf-8")
    assert analyzer.main(["--root", str(tmp_path), "--json"]) == 1
    captured = capsys.readouterr()
    assert str(path) in captured.err
    assert "failed" in captured.err
    assert captured.out == ""


def test_history_sha256_is_checked_and_stale_server_path_uses_local_sibling(tmp_path):
    local = write_json(tmp_path / "history.json", history())
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    target = tmp_path / "metrics.json"
    write_json(
        target, metrics(history="/not-present-server-run/history.json", history_sha256=digest)
    )
    assert analyzer.analyze(target)["records"][0]["history_path"] == str(local.resolve())
    write_json(target, metrics(history_sha256="0" * 64))
    with pytest.raises(analyzer.AnalysisError, match="SHA256 mismatch"):
        analyzer.analyze(target)


@pytest.mark.parametrize("rows", [{"epoch": 1}, [{"epoch": 2}, {"epoch": 1}], [{"epoch": True}]])
def test_invalid_history_contract_is_not_silently_replaced(tmp_path, rows):
    write_json(tmp_path / "history.json", rows)
    write_json(tmp_path / "metrics.json", metrics())
    with pytest.raises(analyzer.AnalysisError, match="History"):
        analyzer.analyze(tmp_path)


def test_manifest_identity_and_recorded_metric_hash_must_match(tmp_path):
    target = write_json(tmp_path / "child" / "metrics.json", metrics())
    manifest = tmp_path / "manifest.json"
    job = {
        "dataset": "ppi",
        "condition": "shared_dynamic_c",
        "output_dir": str(target.parent),
        "metrics_sha256": "0" * 64,
    }
    write_json(manifest, {"jobs": [job]})
    with pytest.raises(analyzer.AnalysisError, match="SHA256 mismatch"):
        analyzer.analyze(manifest)
    job.pop("metrics_sha256")
    job["dataset"] = "cora"
    write_json(manifest, {"jobs": [job]})
    with pytest.raises(analyzer.AnalysisError, match="dataset mismatch"):
        analyzer.analyze(manifest)


def test_stdout_only_without_torch_site_packages_or_checkpoint_access(tmp_path):
    write_json(tmp_path / "metrics.json", metrics())
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"not a checkpoint; must never be opened")
    before = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    script = Path(analyzer.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, "-B", "-S", str(script), "--root", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["records"][0]["dataset"] == "ppi"
    assert completed.stderr == ""
    after = {
        path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_explicit_checkpoint_and_empty_root_are_rejected(tmp_path):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"unit-only")
    with pytest.raises(analyzer.AnalysisError, match="never checkpoints"):
        analyzer.analyze(checkpoint)
    with pytest.raises(analyzer.AnalysisError, match="No supported V5"):
        analyzer.analyze(tmp_path)


def test_manifest_cannot_make_analyzer_read_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"unit-only")
    manifest = write_json(
        tmp_path / "manifest.json",
        {
            "jobs": [
                {
                    "condition": "shared_dynamic_c",
                    "metrics_path": str(checkpoint),
                }
            ]
        },
    )
    original = Path.read_bytes

    def guarded_read(path):
        assert path != checkpoint, "analyzer must not even read checkpoint bytes"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(analyzer.AnalysisError, match="never checkpoints"):
        analyzer.analyze(manifest)


def test_default_text_names_diagnostics_and_missing_data(tmp_path, capsys):
    write_json(tmp_path / "metrics.json", metrics())
    assert analyzer.main(["--root", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "Read-only" in text and "no checkpoint/GPU" in text
    assert "first_active_conductance_gradient" in text
    assert "unavailable" in text
    assert "shuffled_c" in text and "c_one" in text
