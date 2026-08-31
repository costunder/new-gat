"""Small tensor/unit-artifact fixtures only; never CPU research or public data."""

from __future__ import annotations

import copy
import json
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from chartgat.cache import atomic_publish, atomic_write_json
from research.conductance_gat.ablation.model import FactorialNodeClassifier, state_sha256
from research.conductance_gat.ablation.protocol import COMMON, CONDITIONS
from research.conductance_gat.ablation.train import checkpoint_payload
from research.conductance_gat.benchmark_data import sha256_file
from research.conductance_gat.c_learning import intervene as audit


class Graph(SimpleNamespace):
    def to(self, device, **kwargs):
        return Graph(
            **{
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in vars(self).items()
            }
        )


class ValidationOnly(dict):
    def __getitem__(self, key):
        if key != "validation":
            raise AssertionError("Audit must not read train/test indices")
        return super().__getitem__(key)


def graph():
    return Graph(
        x=torch.tensor([[0.3, 0.2], [1.2, -0.7], [3.1, 4.0], [-1.0, 0.7]]),
        y=torch.tensor([999999, 0, 1, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
    )


def model():
    torch.manual_seed(11)
    return FactorialNodeClassifier(2, 2, normalization="node_degree", hidden_channels=8)


def test_graphwise_mean_is_not_minibatch_mean_and_handles_empty_graphs():
    values = torch.tensor([1.0, 3.0, 10.0, 40.0])
    groups = torch.tensor([0, 0, 2, 2])
    torch.testing.assert_close(
        audit.graphwise_mean(values, groups, 4), torch.tensor([2.0, 2.0, 25.0, 25.0])
    )
    assert audit.graphwise_mean(torch.empty(0), torch.empty(0, dtype=torch.long), 2).numel() == 0


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_mean_rejects_nonpositive_or_nonfinite_conductance(value):
    with pytest.raises(ValueError, match="finite and positive"):
        audit.graphwise_mean(torch.tensor([value]), torch.tensor([0]), 1)


def test_mean_operator_recomputes_degree_and_equals_uniform_row_average():
    network = model()
    operator = network.operators[0]
    state = torch.randn(7, 8)
    incidence = torch.tensor([[0, 0, 3, 3, 4], [1, 2, 4, 5, 5]])
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    tail, head = incidence
    sums, degree = torch.zeros_like(state), torch.zeros(7)
    sums.index_add_(0, tail, state[head])
    sums.index_add_(0, head, state[tail])
    degree.index_add_(0, tail, torch.ones(5))
    degree.index_add_(0, head, torch.ones(5))
    expected = state.clone()
    mask = degree > 0
    expected[mask] = 0.05 * state[mask] + 0.95 * sums[mask] / degree[mask, None]
    with audit.MeanConductance(network, (0,)):
        actual = operator(state, incidence, groups, 3)
    torch.testing.assert_close(actual, expected, atol=4e-7, rtol=1e-5)
    torch.testing.assert_close(actual[6], state[6], atol=0, rtol=0)
    assert not operator._forward_pre_hooks and not operator.estimator._forward_hooks


def test_cross_graph_edge_rejected_and_hooks_removed_after_forward_failure():
    network = model()
    with pytest.raises(ValueError, match="crosses graph"):
        with audit.MeanConductance(network, (0,)):
            network.operators[0](
                torch.randn(2, 8), torch.tensor([[0], [1]]), torch.tensor([0, 1]), 2
            )
    assert not any(op._forward_pre_hooks or op.estimator._forward_hooks for op in network.operators)


def test_eval_reads_only_validation_labels_preserves_modes_state_gradients_rng():
    network = model()
    network.train()
    network.norms[0].eval()
    for parameter in network.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = state_sha256(network)
    modes = [module.training for module in network.modules()]
    gradients = [parameter.grad.clone() for parameter in network.parameters()]
    rng = torch.get_rng_state().clone()
    result, _ = audit.evaluate(
        network, [graph()], torch.tensor([1, 2]), torch.device("cpu"), (0, 1)
    )
    assert result["metric_name"] == "accuracy" and result["prediction_count"] == 2
    assert state_sha256(network) == before
    assert [module.training for module in network.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    for saved, parameter in zip(gradients, network.parameters(), strict=True):
        torch.testing.assert_close(saved, parameter.grad, atol=0, rtol=0)


def test_runtime_restores_rng_and_flags_on_error():
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    dtype = torch.get_default_dtype()
    with pytest.raises(RuntimeError):
        with audit.preserved_runtime():
            random.random()
            np.random.rand()
            torch.rand(3)
            torch.set_default_dtype(torch.float64)
            raise RuntimeError("fixture failure")
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.get_default_dtype() == dtype


def test_bad_baseline_stops_before_any_mean_intervention(monkeypatch):
    network = model()
    calls = []
    original = audit.evaluate

    def observed(*args, **kwargs):
        calls.append(args[4:] or ())
        return original(*args, **kwargs)

    monkeypatch.setattr(audit, "evaluate", observed)
    with pytest.raises(ValueError, match="Original checkpoint validation mismatch"):
        audit.audit_model(network, [graph()], torch.tensor([1, 2]), torch.device("cpu"), 0.123456)
    assert len(calls) == 1


def test_audit_runs_all_and_individual_layers_not_training():
    network = model()
    original, _ = audit.evaluate(network, [graph()], torch.tensor([1, 2]), torch.device("cpu"))
    result = audit.audit_model(
        network, [graph()], torch.tensor([1, 2]), torch.device("cpu"), original["validation"]
    )
    assert [row["intervened_layers"] for row in result["interventions"]] == [[0, 1], [0], [1]]
    for row in result["interventions"]:
        assert row["percentage_points"] == 100 * (row["validation"] - original["validation"])
        assert 0 <= row["changed_prediction_fraction"] <= 1
        assert row["logit_max_absolute_delta"] >= row["logit_mean_absolute_delta"] >= 0
    assert result["baseline_absolute_error"] == 0


def test_ppi_micro_f1_uses_all_validation_node_label_decisions():
    network = model()
    item = graph()
    item.y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    result, outputs = audit.evaluate(network, [item], None, torch.device("cpu"))
    prediction = outputs[0] > 0
    truth = item.y > 0
    expected = float(2 * (prediction & truth).sum() / (prediction.sum() + truth.sum()))
    assert result["validation"] == pytest.approx(expected)
    assert result["prediction_count"] == 8 and result["metric_name"] == "micro_f1"


def test_validation_data_never_constructs_training_or_test_loader(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in ("torch_geometric", "torch_geometric.data", "torch_geometric.loader")
    }
    modules["torch_geometric.data"].Data = Graph
    created = []

    def loader(items, **kwargs):
        created.append((items, kwargs))
        return items

    modules["torch_geometric.loader"].DataLoader = loader
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    payload = {
        "dataset": "ppi",
        "graphs": [{"id": i} for i in range(4)],
        "splits": ValidationOnly(validation=[1, 2]),
    }
    batches, indices = audit.validation_data(
        payload, {"model_seed": 0, "configuration": {"batch_size": 2}}, torch.device("cpu")
    )
    assert indices is None and [item.id for item in batches] == [1, 2]
    assert len(created) == 1 and created[0][1]["shuffle"] is False


def source_fixture(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    cache = tmp_path / "data/conductance_gat/matched_benchmark_v1/cora"
    cache.mkdir(parents=True)
    fixture = graph()
    payload = {"dataset": "cora", "classes": 2, "graphs": [vars(fixture)]}
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
    manifest = {
        "schema_version": 1,
        "suite": "conductance_factorial",
        "status": "passed",
        "source_integrity_valid": True,
        "conditions": CONDITIONS,
        "config": config,
        "sources": {
            "git_revision": "unit-fixture-no-training",
            "sha256": {name: sha256_file(audit.ROOT / name) for name in audit.MODEL_SOURCES},
        },
        "jobs": [],
    }
    network = FactorialNodeClassifier(2, 2, normalization="node_degree")
    original, _ = audit.evaluate(network, [fixture], torch.tensor([1, 2]), torch.device("cpu"))
    for condition, factors in CONDITIONS.items():
        output = root / "cora" / condition
        output.mkdir(parents=True)
        args = SimpleNamespace(
            **{key: value for key, value in config.items() if key not in COMMON},
            dataset="cora",
            condition=condition,
        )
        saved = checkpoint_payload(network, args, protocol, "a" * 64, 1, original["validation"], 1)
        atomic_publish(output / "best.pt", lambda path, value=saved: torch.save(value, path))
        atomic_write_json(output / "history.json", [])
        metrics = {
            "schema_version": 1,
            "research_suite": "conductance_factorial",
            "status": "passed",
            "dataset": "cora",
            "condition": condition,
            "model_seed": 0,
            **factors,
            "configuration": saved["configuration"],
            "non_gate_weight_decay": 0.0005,
            "cache_sha256": protocol["data_sha256"],
            "protocol": protocol,
            "initial_state_sha256": "a" * 64,
            "best_epoch": 1,
            "epochs_run": 2,
            "best_checkpoint_optimizer_steps": 1,
            "validation": original["validation"],
            "metric_name": "accuracy",
            "train_loss": 0.7,
            "elapsed_seconds": 1.0,
            "peak_cuda_allocated_bytes": 0,
            "test_evaluated": False,
            "evaluation_split": "validation",
            "checkpoint": str(output / "best.pt"),
            "checkpoint_sha256": sha256_file(output / "best.pt"),
            "history": str(output / "history.json"),
            "history_sha256": sha256_file(output / "history.json"),
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


def test_completed_source_read_is_strict_and_does_not_regenerate_old_reports(tmp_path):
    root, _, _, _ = source_fixture(tmp_path)
    before = audit._hashes(root.rglob("*.*"))
    manifest, selected, hashes = audit.validate_source(root, ["cora"])
    assert manifest["status"] == "passed" and set(selected) == {"cora"}
    audit._assert_unchanged(hashes, "fixture")
    assert audit._hashes(root.rglob("*.*")) == before
    assert not (root / "comparison.json").exists()


@pytest.mark.parametrize(
    "mutation", ["checkpoint", "metrics", "source_code", "source_integrity", "missing_dataset"]
)
def test_invalid_source_rejected_without_writes(tmp_path, mutation):
    root, manifest, _, _ = source_fixture(tmp_path)
    requested = ["cora"]
    if mutation == "checkpoint":
        (root / "cora/node_degree/best.pt").write_bytes(b"changed fixture")
    elif mutation == "metrics":
        path = root / "cora/node_degree/metrics.json"
        metrics = json.loads(path.read_text())
        metrics["configuration"]["dropout"] = 0.8
        atomic_write_json(path, metrics)
    elif mutation == "source_code":
        manifest["sources"]["sha256"][audit.MODEL_SOURCES[0]] = "f" * 64
    elif mutation == "source_integrity":
        manifest["source_integrity_valid"] = False
    else:
        requested = ["ppi"]
    atomic_write_json(root / "manifest.json", manifest)
    before = audit._hashes(root.rglob("*.*"))
    with pytest.raises(ValueError):
        audit.validate_source(root, requested)
    assert audit._hashes(root.rglob("*.*")) == before


@pytest.mark.parametrize(
    "key", ["model", "architecture", "model_seed", "validation", "cache_sha256", "optimizer_steps"]
)
def test_checkpoint_metadata_must_match_metrics(tmp_path, key):
    root, _, _, _ = source_fixture(tmp_path)
    _, selected, _ = audit.validate_source(root, ["cora"])
    metrics = selected["cora"]
    saved = torch.load(metrics["checkpoint"], weights_only=True)
    audit.validate_checkpoint(saved, metrics)
    changed = copy.deepcopy(saved)
    changed[key] = "tampered fixture"
    with pytest.raises(ValueError, match="Checkpoint"):
        audit.validate_checkpoint(changed, metrics)


def test_production_cli_rejects_cpu_without_outputs(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        audit.main(
            [
                "--source-run",
                str(tmp_path),
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert not (tmp_path / "out").exists()


def test_complete_cli_on_bounded_fixture_preserves_all_input_bytes(monkeypatch, tmp_path):
    root, _, payload, protocol = source_fixture(tmp_path)
    monkeypatch.setattr(audit, "_require_cuda", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "unit-fixture-no-gpu")
    seen = []

    def offline(name, data_root, *, allow_download):
        seen.append(allow_download)
        return payload, protocol

    monkeypatch.setattr(audit, "load_dataset", offline)
    monkeypatch.setattr(audit, "validation_data", lambda *args: ([graph()], torch.tensor([1, 2])))
    before = audit._hashes(path for path in tmp_path.rglob("*") if path.is_file())
    output = tmp_path / "audit-output"
    result = audit.main(
        [
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
    )
    assert result == 0 and seen == [False]
    audit._assert_unchanged(before, "all source/cache fixture files")
    record = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["training_performed"] is record["test_evaluated"] is False
    assert record["datasets"][0]["baseline_absolute_error"] == 0
    assert len(record["datasets"][0]["interventions"]) == 3
    assert "not whether learning C improves" in (output / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("relative", ["source/nested", "data/nested", "."])
def test_output_cannot_overlap_source_or_cache(monkeypatch, tmp_path, relative):
    root, _, _, _ = source_fixture(tmp_path)
    monkeypatch.setattr(audit, "_require_cuda", lambda device: None)
    with pytest.raises(ValueError, match="separate"):
        audit.main(
            [
                "--source-run",
                str(root),
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / relative),
                "--device",
                "cpu",
            ]
        )


def test_late_source_change_invalidates_and_withholds_all_contrasts(monkeypatch, tmp_path):
    root, _, payload, protocol = source_fixture(tmp_path)
    monkeypatch.setattr(audit, "_require_cuda", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "unit-fixture-no-gpu")
    monkeypatch.setattr(audit, "load_dataset", lambda *args, **kwargs: (payload, protocol))
    monkeypatch.setattr(audit, "validation_data", lambda *args: ([graph()], torch.tensor([1, 2])))
    original = audit.audit_model

    def changed(*args):
        result = original(*args)
        (root / "cora/node_degree/history.json").write_bytes(b"changed fixture")
        return result

    monkeypatch.setattr(audit, "audit_model", changed)
    output = tmp_path / "invalid-output"
    result = audit.main(
        [
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
    )
    assert result == 1
    report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid" and report["datasets"] == []
    assert "changed during the audit" in report["error"]
