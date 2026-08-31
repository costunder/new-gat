"""Four-arm artifact integration on a four-node fixture with mocked GPU hardware.

This is NOT a public-dataset or CPU research experiment. Only CUDA APIs, dependency
preflight, subprocess dispatch, and official data loading are replaced; model,
optimizer, training loop, checkpoint serialization, runner configuration, and
comparison integrity checks are the real implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from chartgat.cache import atomic_publish, atomic_write_json
from research.conductance_gat.ablation import report, train
from research.conductance_gat.ablation.protocol import CONDITIONS
from research.conductance_gat.benchmark_data import sha256_file
from scripts import run_conductance_factorial as runner


def test_real_four_arm_training_artifacts_pass_real_runner_and_comparison(monkeypatch, tmp_path):
    graph = SimpleNamespace(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        # The held-out fixture label is intentionally outside the two-class range.
        # A mistaken unmasked cross-entropy call would fail immediately.
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
    )

    class NoTestIndices(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("The factorial investigation must not read test indices")
            return super().__getitem__(key)

    indices = NoTestIndices(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    payload = {"dataset": "cora", "classes": 2, "graphs": [vars(graph)]}
    fixture_path = tmp_path / "unit-fixture-data.pt"
    atomic_publish(fixture_path, lambda path: torch.save(payload, path))
    fixture_hash = sha256_file(fixture_path)
    protocol = {
        "data_sha256": fixture_hash,
        "unit_fixture_only": True,
        "description": "four-node fixture; no official dataset was loaded",
    }

    monkeypatch.setattr(train, "_require_cuda", lambda device: None)
    monkeypatch.setattr(train, "_configure_fp32", lambda: None)
    monkeypatch.setattr(train, "_make_data", lambda *args: (graph, indices))
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

    trained: dict[str, dict] = {}
    preflights = []

    def fixture_dispatch(command, log, environment):
        if any(Path(argument).name == "gpu_preflight.py" for argument in command):
            preflights.append(command)
            return 0
        module_index = command.index("research.conductance_gat.ablation.train")
        args = train.build_parser().parse_args(command[module_index + 1 :])
        assert args.dataset == "cora" and args.model_seed == 0
        assert args.epochs == 2 and args.workers == 0 and args.device == "cuda"
        args.output_dir.mkdir(parents=True, exist_ok=False)
        metrics = train.train_model(payload, protocol, args, torch.device("cpu"), args.output_dir)
        metrics.update(unit_fixture_only=True, hardware_mocked=True)
        atomic_write_json(args.output_dir / "metrics.json", metrics)
        trained[args.condition] = metrics
        return 0

    monkeypatch.setattr(runner, "run_logged", fixture_dispatch)
    result = runner.main(
        [
            "--datasets",
            "cora",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--workers",
            "0",
            "--model-seed",
            "0",
            "--device",
            "cuda",
            "--results-root",
            str(tmp_path / "results"),
            "--data-root",
            str(tmp_path / "cache"),
            "--run-id",
            "integration-unit-fixture",
        ]
    )
    assert result == 0
    assert len(preflights) == 1
    assert list(trained) == list(CONDITIONS)
    assert len({metrics["initial_state_sha256"] for metrics in trained.values()}) == 1
    assert {metrics["cache_sha256"] for metrics in trained.values()} == {fixture_hash}
    for condition, metrics in trained.items():
        assert metrics["status"] == "passed"
        assert metrics["normalization"] == CONDITIONS[condition]["normalization"]
        assert metrics["gate_weight_decay"] == CONDITIONS[condition]["gate_weight_decay"]
        assert metrics["non_gate_weight_decay"] == 0.0005
        assert metrics["test_evaluated"] is False and "test" not in metrics
        assert metrics["evaluation_split"] == "validation"
        assert metrics["epochs_run"] == metrics["optimizer_steps"] == 2
        assert metrics["checkpoint_sha256"] == sha256_file(Path(metrics["checkpoint"]))
        assert metrics["history_sha256"] == sha256_file(Path(metrics["history"]))
        saved = torch.load(metrics["checkpoint"], weights_only=True)
        assert saved["architecture"]["normalization"] == metrics["normalization"]
        assert saved["research_suite"] == "conductance_factorial"
        actual_decay = metrics["diagnostics"]["train_trajectory"][0]["parameter_groups"]
        assert actual_decay["operators.0"]["weight_decay"] == metrics["gate_weight_decay"]
        assert actual_decay["non_gate"]["weight_decay"] == 0.0005

    root = tmp_path / "results/conductance_gat/ablations/integration-unit-fixture"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    manifest["source_integrity_valid"] = True
    # Re-read all real serialized artifacts, including actual SHA-256 checks.
    comparison = report.write_comparison(root, manifest)
    assert comparison["status"] == "passed" and comparison["complete"] is True
    assert comparison["source_integrity_valid"] is True
    assert comparison["test_evaluated"] is False
    assert comparison["uncertainty_status"] == "not_estimated_single_seed"
    dataset = comparison["datasets"][0]
    assert dataset["complete"] is True and len(dataset["effects"]) == 5
    scores = {condition: metrics["validation"] for condition, metrics in trained.items()}
    assert dataset["effects"]["interaction"]["score_delta"] == (
        scores["node_degree_gate_no_wd"]
        - scores["node_degree"]
        - scores["gate_no_wd"]
        + scores["baseline"]
    )
    for name in ("comparison.json", "comparison.csv", "comparison.md"):
        assert (root / name).is_file()
