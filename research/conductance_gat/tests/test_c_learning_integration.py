"""Real new-runner/two-arm/artifact/report path on an explicit four-node fixture.

CUDA hardware, dependencies, subprocess dispatch and official data access alone
are mocked. This is not a public-dataset or CPU research-performance experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from chartgat.cache import atomic_publish, atomic_write_json
from research.conductance_gat.ablation import train as shared_train
from research.conductance_gat.benchmark_data import sha256_file
from research.conductance_gat.c_learning import report, train
from research.conductance_gat.c_learning.protocol import CONDITIONS, SUITE
from scripts import run_conductance_c_learning as runner


def test_two_real_arms_artifacts_pass_new_runner_and_comparison(monkeypatch, tmp_path):
    graph = SimpleNamespace(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
    )

    class NoTestIndices(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("C-learning must never read test indices")
            return super().__getitem__(key)

    indices = NoTestIndices(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    payload = {"dataset": "cora", "classes": 2, "graphs": [vars(graph)]}
    fixture_path = tmp_path / "unit-fixture-data.pt"
    atomic_publish(fixture_path, lambda path: torch.save(payload, path))
    fixture_hash = sha256_file(fixture_path)
    protocol = {"data_sha256": fixture_hash, "unit_fixture_only": True}

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
    monkeypatch.setattr(
        runner,
        "_source_snapshot",
        lambda: {"sha256": {"unit_fixture": fixture_hash}, "git_revision": None},
    )

    trained, preflights = {}, []

    def dispatch_fixture(command, log, environment):
        if any(Path(argument).name == "gpu_preflight.py" for argument in command):
            preflights.append(command)
            return 0
        module_index = command.index("research.conductance_gat.c_learning.train")
        args = train.build_parser().parse_args(command[module_index + 1 :])
        assert args.dataset == "cora" and args.model_seed == 0 and args.device == "cuda"
        assert args.epochs == 2 and args.workers == 0
        args.output_dir.mkdir(parents=True, exist_ok=False)
        metrics = train.train_model(payload, protocol, args, torch.device("cpu"), args.output_dir)
        metrics.update(unit_fixture_only=True, hardware_mocked=True)
        atomic_write_json(args.output_dir / "metrics.json", metrics)
        trained[args.condition] = metrics
        return 0

    monkeypatch.setattr(runner, "run_logged", dispatch_fixture)
    status = runner.main(
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
            str(tmp_path / "cache"),
            "--run-id",
            "c-learning-integration-fixture",
        ]
    )
    assert status == 0 and len(preflights) == 1 and list(trained) == list(CONDITIONS)
    assert len({metrics["initial_state_sha256"] for metrics in trained.values()}) == 1
    for condition, metrics in trained.items():
        assert metrics["status"] == "passed" and metrics["research_suite"] == SUITE
        assert metrics["configuration"]["model_seed"] == 0
        assert "gate_mode" not in metrics["configuration"]
        assert metrics["gate_mode"] == CONDITIONS[condition]["gate_mode"]
        assert metrics["normalization"] == "node_degree"
        assert metrics["test_evaluated"] is False and "test" not in metrics
        assert metrics["evaluation_split"] == "validation"
        assert metrics["checkpoint_sha256"] == sha256_file(Path(metrics["checkpoint"]))
        assert metrics["history_sha256"] == sha256_file(Path(metrics["history"]))
        checkpoint = torch.load(metrics["checkpoint"], weights_only=True)
        assert checkpoint["model"] == checkpoint["research_suite"] == SUITE
        assert checkpoint["gate_mode"] == metrics["gate_mode"]
        assert checkpoint["initial_state_sha256"] == metrics["initial_state_sha256"]
    root = tmp_path / "results/conductance_gat/c_learning/c-learning-integration-fixture"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed" and manifest["suite"] == SUITE
    comparison = report.write_comparison(root, manifest)
    assert comparison["status"] == "passed" and comparison["complete"] is True
    assert comparison["source_integrity_valid"] is True and comparison["errors"] == []
    assert comparison["uncertainty_status"] == "not_estimated_single_seed"
    assert comparison["n_model_seeds"] == 1 and comparison["test_evaluated"] is False
    dataset = comparison["datasets"][0]
    contrast = dataset["learned_minus_fixed"]
    expected = trained["learned_c"]["validation"] - trained["fixed_c"]["validation"]
    assert contrast == {"score_delta": expected, "percentage_points": expected * 100}
    rows = {row["condition"]: row for row in dataset["conditions"]}
    assert rows["learned_c"]["frozen_parameters"] == 0
    assert rows["fixed_c"]["frozen_parameters"] > 0
    assert rows["learned_c"]["trainable_parameters"] > rows["fixed_c"]["trainable_parameters"]
    assert all(
        layer["conductance_cv"]["mean"] == 0.0
        for layer in rows["fixed_c"]["best_validation_diagnostics"]
    )
    for filename in ("comparison.json", "comparison.md", "comparison.csv"):
        assert (root / filename).is_file()
