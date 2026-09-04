"""New-runner training artifacts through the read-only selected-checkpoint audit.

This four-node UNIT fixture explicitly substitutes CUDA hardware, dependency and
data-loading/subprocess boundaries. The actual model, optimizer, training loop,
artifact serialization, source hashes, report validators and audit run unchanged.
It is not a CPU research experiment and does not download public datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chartgat.cache import atomic_publish, atomic_write_json
from research.conductance_gat.ablation import train as shared_train
from research.conductance_gat.benchmark_data import sha256_file
from research.conductance_gat.c_learning import intervene as audit
from research.conductance_gat.c_learning import train
from research.conductance_gat.c_learning.model import (
    CLearningNodeClassifier,
    FixedOneConductance,
)
from research.conductance_gat.c_learning.protocol import CONDITIONS, SUITE
from scripts import run_conductance_c_learning as runner


class FixtureGraph(SimpleNamespace):
    def to(self, device, **kwargs):
        return FixtureGraph(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


class NoTestIndices(dict):
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("This investigation must never read test indices")
        return super().__getitem__(key)


def test_new_training_artifacts_audit_actual_learned_checkpoint_without_input_mutation(
    monkeypatch, tmp_path
):
    graph = FixtureGraph(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
    )
    indices = NoTestIndices(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    payload = {"dataset": "cora", "classes": 2, "graphs": [vars(graph)]}
    data_root = tmp_path / "data/paper"
    cache = data_root / "conductance_gat/matched_benchmark_v1/cora"
    cache.mkdir(parents=True)
    atomic_publish(cache / "data.pt", lambda path: torch.save(payload, path))
    protocol = {"data_sha256": sha256_file(cache / "data.pt"), "unit_fixture_only": True}
    atomic_write_json(cache / "manifest.json", protocol)

    monkeypatch.setattr(shared_train, "_require_cuda", lambda device: None)
    monkeypatch.setattr(shared_train, "_configure_fp32", lambda: None)
    monkeypatch.setattr(shared_train, "_make_data", lambda *args: (graph, indices))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "unit_fixture_mocked_cuda")
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"unit_fixture_only": True})
    trained, preflights = {}, []

    def dispatch_fixture(command, log, environment):
        if any(Path(argument).name == "gpu_preflight.py" for argument in command):
            preflights.append(command)
            return 0
        module_index = command.index("research.conductance_gat.c_learning.train")
        args = train.build_parser().parse_args(command[module_index + 1 :])
        assert args.dataset == "cora" and args.model_seed == 0 and args.device == "cuda"
        args.output_dir.mkdir(parents=True, exist_ok=False)
        result = train.train_model(payload, protocol, args, torch.device("cpu"), args.output_dir)
        result.update(unit_fixture_only=True, hardware_mocked=True)
        atomic_write_json(args.output_dir / "metrics.json", result)
        trained[args.condition] = result
        return 0

    monkeypatch.setattr(runner, "run_logged", dispatch_fixture)
    # No source-snapshot stub: provenance must contain the real current model
    # hashes required by audit.validate_source, not an unrelated fixture digest.
    result = runner.main(
        [
            "--datasets",
            "cora",
            "--model-seed",
            "0",
            "--device",
            "cuda",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--workers",
            "0",
            "--results-root",
            str(tmp_path / "results"),
            "--data-root",
            str(data_root),
            "--run-id",
            "c-learning-to-audit-fixture",
        ]
    )
    assert result == 0 and list(trained) == list(CONDITIONS) and len(preflights) == 1
    source = tmp_path / "results/conductance_gat/c_learning/c-learning-to-audit-fixture"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"] == SUITE and manifest["status"] == "passed"
    for name in (
        "research/conductance_gat/ablation/model.py",
        "research/conductance_gat/c_learning/model.py",
        "research/conductance_gat/c_learning/protocol.py",
    ):
        assert manifest["sources"]["sha256"][name] == sha256_file(runner.ROOT / name)
    assert (
        trained["learned_c"]["shared_backbone_initial_state_sha256"]
        == trained["fixed_c"]["shared_backbone_initial_state_sha256"]
    )
    assert (
        trained["learned_c"]["initial_state_sha256"]
        != trained["fixed_c"]["initial_state_sha256"]
    )
    assert trained["fixed_c"]["estimator_parameters"] == 0
    assert trained["fixed_c"]["frozen_parameters"] == 0
    saved = torch.load(trained["learned_c"]["checkpoint"], weights_only=True)

    monkeypatch.setattr(audit, "_require_cuda", lambda device: None)
    offline_reads, checkpoint_reads = [], []
    original_load = torch.load

    def read_cache(name, root, *, allow_download):
        assert name == "cora" and root == data_root and allow_download is False
        offline_reads.append(name)
        return original_load(cache / "data.pt", weights_only=True), protocol

    def record_checkpoint_read(path, *args, **kwargs):
        if Path(path).name == "best.pt":
            checkpoint_reads.append(Path(path))
        return original_load(path, *args, **kwargs)

    def validation_fixture(data, metrics, device, workers):
        assert metrics["research_suite"] == SUITE and metrics["condition"] == "learned_c"
        assert data["dataset"] == "cora"
        assert workers == 4
        return [graph], indices["validation"]

    monkeypatch.setattr(audit, "load_dataset", read_cache)
    monkeypatch.setattr(audit, "validation_data", validation_fixture)
    monkeypatch.setattr(torch, "load", record_checkpoint_read)
    input_bytes = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "read-only-audit"
    result = audit.main(
        [
            "--source-run",
            str(source),
            "--datasets",
            "cora",
            "--device",
            "cpu",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output),
        ]
    )
    assert result == 0 and offline_reads == ["cora"]
    assert checkpoint_reads == [Path(trained["learned_c"]["checkpoint"])]
    for path, expected_bytes in input_bytes.items():
        assert path.read_bytes() == expected_bytes, f"Read-only audit modified {path}"
    audited = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audited["status"] == "passed"
    assert audited["source_suite"] == SUITE and audited["source_condition"] == "learned_c"
    assert audited["training_performed"] is audited["test_evaluated"] is False
    assert audited["evaluation_split"] == "validation" and audited["n_model_seeds"] == 1
    assert audited["execution_plan"]["subset_or_fast_mode"] is False
    assert audited["resource_observability"]["measurement_scope"]
    assert audited["throughput"]["completed_forward_batches"] == 4
    assert audited["throughput"]["forward_batches_per_second"]["value"] > 0
    item = audited["datasets"][0]
    assert item["source_suite"] == SUITE and item["source_condition"] == "learned_c"
    assert item["checkpoint_sha256"] == trained["learned_c"]["checkpoint_sha256"]
    assert item["saved_validation"] == item["original"]["validation"]
    assert item["saved_validation"] == trained["learned_c"]["validation"]
    assert item["baseline_absolute_error"] == 0
    cases = {
        "mean_c_all_layers": (0, 1),
        "mean_c_layer_0": (0,),
        "mean_c_layer_1": (1,),
    }
    assert [row["intervention"] for row in item["interventions"]] == list(cases)
    # Independent fixed-one replacements verify each graph/layer mean cancels
    # with the recomputed degree. They retain the learned checkpoint's OTHER
    # weights, unlike the separately trained fixed_c arm above.
    original = CLearningNodeClassifier(3, 2, **saved["architecture"]).eval()
    original.load_state_dict(saved["state_dict"], strict=True)
    with torch.no_grad():
        reference = original(graph).index_select(0, indices["validation"])
        for row in item["interventions"]:
            layers = cases[row["intervention"]]
            model = CLearningNodeClassifier(3, 2, **saved["architecture"]).eval()
            model.load_state_dict(saved["state_dict"], strict=True)
            for layer in layers:
                operator = model.operators[layer]
                operator.estimator = FixedOneConductance()
            logits = model(graph).index_select(0, indices["validation"])
            expected_score = float(
                (logits.argmax(-1) == graph.y[indices["validation"]]).float().mean()
            )
            expected_logit_delta = float((logits.double() - reference.double()).abs().mean())
            assert row["intervened_layers"] == list(layers)
            assert row["validation"] == expected_score
            assert row["percentage_points"] == 100 * (
                expected_score - item["original"]["validation"]
            )
            assert row["logit_mean_absolute_delta"] == pytest.approx(
                expected_logit_delta, abs=1e-6, rel=1e-4
            )
    assert (output / "report.md").is_file()
