"""Metadata and reconstruction contracts on bounded, untrained unit fixtures only."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from chartgat.cache import atomic_publish, atomic_write_json
from research.conductance_gat.ablation.model import (
    shared_backbone_state_sha256,
    state_sha256,
)
from research.conductance_gat.ablation.train import checkpoint_payload
from research.conductance_gat.benchmark_data import sha256_file
from research.conductance_gat.c_learning import intervene as audit
from research.conductance_gat.c_learning.model import CLearningNodeClassifier
from research.conductance_gat.c_learning.protocol import COMMON, CONDITIONS, SUITE
from research.conductance_gat.c_learning.train import DEFINITION
from research.conductance_gat.tests.test_c_mean_audit import graph


def c_learning_fixture(tmp_path):
    # The directory name is deliberately uninformative: dispatch MUST use metadata.
    root = tmp_path / "source"
    root.mkdir()
    cache = tmp_path / "data/conductance_gat/matched_benchmark_v1/cora"
    cache.mkdir(parents=True)
    fixture_graph = graph()
    payload = {"dataset": "cora", "classes": 2, "graphs": [vars(fixture_graph)]}
    atomic_publish(cache / "data.pt", lambda path: torch.save(payload, path))
    protocol = {"data_sha256": sha256_file(cache / "data.pt"), "unit_fixture_only": True}
    atomic_write_json(cache / "manifest.json", protocol)
    config = {
        **COMMON,
        "datasets": ["cora"],
        "model_seed": 0,
        "epochs": 2,
        "patience": 2,
        "batch_size": 2,
        "workers": 0,
        "device": "cuda",
        "data_root": str(tmp_path / "data"),
    }
    model_sources = audit.MODEL_SOURCES + (
        "research/conductance_gat/c_learning/model.py",
        "research/conductance_gat/c_learning/protocol.py",
    )
    manifest = {
        "schema_version": 1,
        "suite": SUITE,
        "status": "passed",
        "source_integrity_valid": True,
        "conditions": CONDITIONS,
        "config": config,
        "sources": {
            "git_revision": "unit-fixture-not-research-training",
            "sha256": {name: sha256_file(audit.ROOT / name) for name in model_sources},
        },
        "jobs": [],
    }
    for condition, spec in CONDITIONS.items():
        torch.manual_seed(0)
        model = CLearningNodeClassifier(2, 2, gate_mode=spec["gate_mode"])
        original, _ = audit.evaluate(
            model, [fixture_graph], torch.tensor([1, 2]), torch.device("cpu")
        )
        output = root / "cora" / condition
        output.mkdir(parents=True)
        args = SimpleNamespace(
            **{key: value for key, value in config.items() if key not in COMMON},
            dataset="cora",
            condition=condition,
        )
        saved = checkpoint_payload(
            model,
            args,
            protocol,
            state_sha256(model),
            1,
            original["validation"],
            1,
            definition=DEFINITION,
            shared_initial_hash=shared_backbone_state_sha256(model),
        )
        atomic_publish(output / "best.pt", lambda path, value=saved: torch.save(value, path))
        atomic_write_json(output / "history.json", [])
        metrics = {
            key: value
            for key, value in saved.items()
            if key not in {"state_dict", "architecture", "model", "optimizer_steps"}
        }
        metrics.update(
            schema_version=1,
            status="passed",
            **spec,
            protocol=protocol,
            epochs_run=2,
            best_checkpoint_optimizer_steps=1,
            metric_name="accuracy",
            train_loss=0.7,
            elapsed_seconds=1.0,
            peak_cuda_allocated_bytes=0,
            versions={"torch": "unit-fixture"},
            gpu="unit-fixture-no-gpu",
            checkpoint=str(output / "best.pt"),
            checkpoint_sha256=sha256_file(output / "best.pt"),
            history=str(output / "history.json"),
            history_sha256=sha256_file(output / "history.json"),
        )
        total = saved["total_parameters"]
        parameter_tensors = sum(1 for _ in model.parameters())
        ownership = {
            "status": "passed",
            "trainable_parameter_tensors": parameter_tensors,
            "optimizer_owned_parameter_tensors": parameter_tensors,
            "trainable_parameter_elements": total,
        }
        metrics["pre_run_observability"] = {
            "status": "pre_run_configuration",
            "model": {
                "total_parameters": total,
                "trainable_parameters": total,
                "frozen_parameters": 0,
                "optimizer_ownership": ownership,
            },
        }
        metrics["first_optimizer_step_integrity"] = {
            **ownership,
            "gradient_status": "all_trainable_parameter_tensors_have_finite_gradients",
            "checked_before_optimizer_step": 1,
        }
        atomic_write_json(output / "metrics.json", metrics)
        manifest["jobs"].append(
            {
                "dataset": "cora",
                "condition": condition,
                "status": "passed",
                "output_dir": str(output),
                "metrics_path": str(output / "metrics.json"),
            }
        )
    atomic_write_json(root / "manifest.json", manifest)
    return root, manifest, payload, protocol


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _edit_checkpoint(root, callback, condition="learned_c"):
    folder = root / "cora" / condition
    metrics = _read(folder / "metrics.json")
    saved = torch.load(folder / "best.pt", weights_only=True)
    callback(saved, metrics)
    atomic_publish(folder / "best.pt", lambda path: torch.save(saved, path))
    metrics["checkpoint_sha256"] = sha256_file(folder / "best.pt")
    atomic_write_json(folder / "metrics.json", metrics)
    return saved, metrics


def test_source_uses_suite_metadata_not_folder_and_does_not_write(tmp_path):
    root, _, _, _ = c_learning_fixture(tmp_path)
    before = audit._hashes(path for path in root.rglob("*") if path.is_file())
    manifest, selected, snapshot = audit.validate_source(root, ["cora"])
    assert manifest["suite"] == SUITE
    assert list(selected) == ["cora"] and selected["cora"]["condition"] == "learned_c"
    assert selected["cora"]["gate_mode"] == "learned"
    assert len(snapshot) == 7  # manifest + two sets of metrics/checkpoint/history
    audit._assert_unchanged(before, "all source files")
    assert not (root / "comparison.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_suite",
        "old_suite",
        "old_conditions",
        "fixed_missing",
        "fixed_failed",
        "mixed_metrics_suite",
        "wrong_gate_mode",
        "missing_sources",
        "modified_model_source",
        "modified_protocol_source",
        "modified_checkpoint",
        "modified_history",
        "missing_dataset",
    ],
)
def test_source_mismatch_fails_closed(tmp_path, mutation):
    root, manifest, _, _ = c_learning_fixture(tmp_path)
    requested = ["cora"]
    if mutation == "unknown_suite":
        manifest["suite"] = "unrecognized"
    elif mutation == "old_suite":
        manifest["suite"] = "conductance_factorial"
    elif mutation == "old_conditions":
        from research.conductance_gat.ablation.protocol import CONDITIONS as OLD

        manifest["conditions"] = OLD
    elif mutation == "fixed_missing":
        manifest["jobs"].pop()
    elif mutation == "fixed_failed":
        manifest["jobs"][1]["status"] = "failed"
        manifest["status"] = "failed"
    elif mutation in {"mixed_metrics_suite", "wrong_gate_mode"}:
        path = root / "cora/learned_c/metrics.json"
        metrics = _read(path)
        if mutation == "mixed_metrics_suite":
            metrics["research_suite"] = "conductance_factorial"
        else:
            metrics["gate_mode"] = "fixed_one"
        atomic_write_json(path, metrics)
    elif mutation == "missing_sources":
        manifest["sources"]["sha256"].pop("research/conductance_gat/c_learning/model.py")
    elif mutation in {"modified_model_source", "modified_protocol_source"}:
        filename = "model.py" if mutation == "modified_model_source" else "protocol.py"
        manifest["sources"]["sha256"][f"research/conductance_gat/c_learning/{filename}"] = "f" * 64
    elif mutation in {"modified_checkpoint", "modified_history"}:
        filename = "best.pt" if mutation == "modified_checkpoint" else "history.json"
        (root / "cora/learned_c" / filename).write_bytes(b"modified unit artifact")
    else:
        requested = ["ppi"]
    atomic_write_json(root / "manifest.json", manifest)
    before = audit._hashes(path for path in root.rglob("*") if path.is_file())
    with pytest.raises(ValueError):
        audit.validate_source(root, requested)
    audit._assert_unchanged(before, "source after rejected validation")


@pytest.mark.parametrize(
    "key,value",
    [
        ("research_suite", "conductance_factorial"),
        ("model", "conductance_factorial"),
        ("condition", "fixed_c"),
        ("gate_mode", "fixed_one"),
        ("frozen_parameters", 1),
        ("trainable_parameters", 1),
        ("total_parameters", 1),
        ("optimizer_steps", 7),
        ("model_seed", 5),
        ("evaluation_split", "test"),
        ("test_evaluated", True),
    ],
)
def test_checkpoint_metadata_must_match_new_suite(tmp_path, key, value):
    root, _, _, _ = c_learning_fixture(tmp_path)
    _, selected, _ = audit.validate_source(root, ["cora"])
    metrics = selected["cora"]
    saved = torch.load(metrics["checkpoint"], weights_only=True)
    audit.validate_checkpoint(saved, metrics)
    changed = copy.deepcopy(saved)
    changed[key] = value
    with pytest.raises(ValueError):
        audit.validate_checkpoint(changed, metrics)


def test_fixed_checkpoint_cannot_be_an_audit_target(tmp_path):
    root, _, _, _ = c_learning_fixture(tmp_path)
    metrics = _read(root / "cora/fixed_c/metrics.json")
    saved = torch.load(metrics["checkpoint"], weights_only=True)
    with pytest.raises(ValueError):
        audit.validate_checkpoint(saved, metrics)


def test_reconstruction_has_correct_class_actual_counts_and_exact_state(tmp_path):
    root, _, payload, _ = c_learning_fixture(tmp_path)
    _, selected, _ = audit.validate_source(root, ["cora"])
    metrics = selected["cora"]
    saved = torch.load(metrics["checkpoint"], weights_only=True)
    network = audit.reconstruct_model(saved, metrics, payload, torch.device("cpu"))
    assert isinstance(network, CLearningNodeClassifier)
    assert network.gate_mode == "learned" and network.normalization == "node_degree"
    assert sum(p.numel() for p in network.parameters()) == saved["total_parameters"]
    assert all(p.requires_grad for p in network.parameters())
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved["state_dict"][name], atol=0, rtol=0)
    saved["total_parameters"] += 1
    saved["trainable_parameters"] += 1
    metrics["total_parameters"] += 1
    metrics["trainable_parameters"] += 1
    with pytest.raises(ValueError):
        audit.reconstruct_model(saved, metrics, payload, torch.device("cpu"))


def test_old_checkpoint_metadata_cannot_masquerade_as_new_saved_state(tmp_path):
    root, _, payload, _ = c_learning_fixture(tmp_path)
    metrics = _read(root / "cora/learned_c/metrics.json")
    saved = torch.load(metrics["checkpoint"], weights_only=True)
    saved["architecture"].pop("gate_mode")
    with pytest.raises(ValueError):
        audit.reconstruct_model(saved, metrics, payload, torch.device("cpu"))


def _mock_cli(monkeypatch, payload, protocol):
    monkeypatch.setattr(audit, "_require_cuda", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "unit-fixture-no-gpu")

    def offline(name, data_root, *, allow_download):
        assert allow_download is False
        return payload, protocol

    monkeypatch.setattr(audit, "load_dataset", offline)
    monkeypatch.setattr(audit, "validation_data", lambda *args: ([graph()], torch.tensor([1, 2])))


def _arguments(tmp_path, root, output):
    return [
        "--source-run",
        str(root),
        "--datasets",
        "cora",
        "--device",
        "cpu",
        "--data-root",
        str(tmp_path / "data"),
        "--output-dir",
        str(output),
    ]


def test_new_suite_baseline_mismatch_withholds_every_intervention(monkeypatch, tmp_path):
    root, _, payload, protocol = c_learning_fixture(tmp_path)

    def edit(saved, metrics):
        saved["validation"] = metrics["validation"] = 0.123456789

    _edit_checkpoint(root, edit)
    _mock_cli(monkeypatch, payload, protocol)
    output = tmp_path / "invalid-audit"
    assert audit.main(_arguments(tmp_path, root, output)) == 1
    report = _read(output / "audit.json")
    assert report["status"] == "invalid" and report["datasets"] == []
    assert "Original checkpoint validation mismatch" in report["error"]


def test_new_suite_late_source_change_withholds_every_intervention(monkeypatch, tmp_path):
    root, _, payload, protocol = c_learning_fixture(tmp_path)
    _mock_cli(monkeypatch, payload, protocol)
    original = audit.audit_model

    def altered(*args):
        result = original(*args)
        (root / "cora/fixed_c/history.json").write_bytes(b"modified after audit")
        return result

    monkeypatch.setattr(audit, "audit_model", altered)
    output = tmp_path / "invalid-audit"
    assert audit.main(_arguments(tmp_path, root, output)) == 1
    report = _read(output / "audit.json")
    assert report["status"] == "invalid" and report["datasets"] == []
    assert "changed during the audit" in report["error"]


def test_monitor_cleanup_failure_keeps_primary_audit_error(monkeypatch, tmp_path):
    root, _, payload, protocol = c_learning_fixture(tmp_path)
    _mock_cli(monkeypatch, payload, protocol)

    class FailingMonitor:
        instances: list[FailingMonitor] = []

        def __init__(self, _device):
            self.finish_calls = 0
            self.instances.append(self)

        def start(self):
            return {"measurement_scope": "unit monitor fixture"}

        def finish(self, **_kwargs):
            self.finish_calls += 1
            raise RuntimeError("unit audit telemetry cleanup failure")

    monkeypatch.setattr(audit, "RuntimeResourceMonitor", FailingMonitor)

    def fail_audit(*_args):
        raise ValueError("primary audit failure")

    monkeypatch.setattr(audit, "audit_model", fail_audit)
    output = tmp_path / "failed-monitor-audit"
    assert audit.main(_arguments(tmp_path, root, output)) == 1
    assert FailingMonitor.instances[0].finish_calls == 1
    report = _read(output / "audit.json")
    assert report["status"] == "invalid"
    assert report["error"] == "ValueError: primary audit failure"
    assert report["resource_observability"] is None
    assert "unit audit telemetry cleanup failure" in report[
        "resource_observability_unavailable_reason"
    ]


def test_new_suite_success_report_identifies_target_and_preserves_source(monkeypatch, tmp_path):
    root, _, payload, protocol = c_learning_fixture(tmp_path)
    _mock_cli(monkeypatch, payload, protocol)
    before = audit._hashes(path for path in tmp_path.rglob("*") if path.is_file())
    output = tmp_path / "learned-audit"
    assert audit.main(_arguments(tmp_path, root, output)) == 0
    report = _read(output / "audit.json")
    assert report["source_suite"] == SUITE and report["source_condition"] == "learned_c"
    assert report["training_performed"] is report["test_evaluated"] is False
    item = report["datasets"][0]
    assert item["source_suite"] == SUITE and item["source_condition"] == "learned_c"
    assert item["baseline_absolute_error"] == 0
    assert [r["intervened_layers"] for r in item["interventions"]] == [[0, 1], [0], [1]]
    text = (output / "report.md").read_text(encoding="utf-8")
    assert SUITE in text and "learned_c" in text
    audit._assert_unchanged(before, "all original source/cache files")


def test_new_suite_refuses_existing_output(monkeypatch, tmp_path):
    root, _, payload, protocol = c_learning_fixture(tmp_path)
    _mock_cli(monkeypatch, payload, protocol)
    output = tmp_path / "existing-audit"
    output.mkdir()
    sentinel = output / "report.md"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit.main(_arguments(tmp_path, root, output))
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_execution_snapshot_includes_new_and_shared_audit_dependencies():
    for name in (
        "research/conductance_gat/c_learning/model.py",
        "research/conductance_gat/c_learning/protocol.py",
        "research/conductance_gat/c_learning/report.py",
        "research/conductance_gat/c_learning/intervene.py",
        "research/conductance_gat/ablation/report.py",
    ):
        assert name in audit.AUDIT_SOURCES
