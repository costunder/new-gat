"""Actual v3 training/artifacts on tiny unit fixtures with mocked CUDA hardware."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chartgat.cache import atomic_publish
from research.conductance_gat.ablation.model import shared_backbone_state_sha256
from research.conductance_gat.benchmark_data import sha256_file
from research.conductance_gat.v3 import train
from research.conductance_gat.v3.model import RelativeCNodeClassifier
from research.conductance_gat.v3.protocol import CONDITIONS, SUITE


def arguments(tmp_path, condition="relative_c"):
    return train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            condition,
            "--output-dir",
            str(tmp_path / condition),
            "--data-root",
            str(tmp_path / "data"),
            "--epochs",
            "2",
            "--patience",
            "2",
            "--edge-chunk-size",
            "2",
        ]
    )


def fixture_data():
    graph = SimpleNamespace(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]),
    )

    class NoTest(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("No test split may be read")
            return super().__getitem__(key)

    indices = NoTest(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    return graph, indices, {"dataset": "cora", "classes": 2, "graphs": [vars(graph)]}


def mock_hardware(monkeypatch):
    monkeypatch.setattr(train, "_require_cuda", lambda device: None)
    monkeypatch.setattr(train, "_configure_fp32", lambda: None)
    for name in ("reset_peak_memory_stats", "synchronize", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "unit_fixture_mocked_cuda")
    monkeypatch.setattr(train, "_source_hashes", lambda: {"unit_fixture": "a" * 64})


@pytest.mark.parametrize("condition", CONDITIONS)
def test_optimizer_groups_and_fixed_estimator_alpha_trainability(condition):
    torch.manual_seed(3)
    model = RelativeCNodeClassifier(3, 2, gate_mode=CONDITIONS[condition]["gate_mode"])
    optimizer = train.make_optimizer(model, condition)
    assert isinstance(optimizer, torch.optim.AdamW)
    metadata = {group["name"]: group for group in train.optimizer_metadata(optimizer)}
    assert metadata["backbone"]["lr"] == 0.005
    assert metadata["backbone"]["weight_decay"] == 0.0005
    assert metadata["controls"]["weight_decay"] == 0
    assert all(op.raw_alpha.requires_grad for op in model.operators)
    if condition == "fixed_c":
        assert "gate_mlp" not in metadata
        assert all(not p.requires_grad for op in model.operators for p in op.estimator.parameters())
        assert all(name.endswith("raw_alpha") for name in metadata["controls"]["parameter_names"])
    else:
        assert metadata["gate_mlp"]["lr"] == 0.01
        assert metadata["gate_mlp"]["weight_decay"] == 0


def test_initial_hash_same_and_cpu_public_training_forbidden(tmp_path):
    models = []
    for mode in ("relative", "fixed_one"):
        torch.manual_seed(2)
        models.append(RelativeCNodeClassifier(3, 2, gate_mode=mode))
    assert shared_backbone_state_sha256(models[0]) == shared_backbone_state_sha256(models[1])
    args = arguments(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.train_model({}, {}, args, torch.device("cpu"), args.output_dir)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "relative_c",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "forbidden"),
            ]
        )
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 2),
        ("workers", 1),
        ("epochs", 0),
        ("patience", 0),
        ("model_seed", -1),
        ("edge_chunk_size", 0),
    ],
)
def test_invalid_protocol_rejected(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        train._validate_args(args)


def test_child_batch_size_defaults_resolve_by_dataset(tmp_path):
    cora = arguments(tmp_path)
    assert cora.batch_size is None
    train._validate_args(cora)
    assert cora.batch_size == 1
    ppi = train.build_parser().parse_args(
        [
            "--dataset",
            "ppi",
            "--condition",
            "relative_c",
            "--output-dir",
            str(tmp_path / "ppi"),
            "--workers",
            "4",
        ]
    )
    train._validate_args(ppi)
    assert ppi.batch_size == 2 and ppi.workers == 4


def test_ppi_uses_all_train_minibatches_and_all_validation_graphs(monkeypatch, tmp_path):
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    mock_hardware(monkeypatch)
    graphs = []
    for graph_index in range(6):
        graphs.append(
            Data(
                x=torch.tensor(
                    [[0.5 + graph_index, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0]]
                ),
                y=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                incidence_edge_index=torch.tensor([[0, 1], [1, 2]]),
            )
        )

    class NoTest(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("PPI test graphs must not be read")
            return super().__getitem__(key)

    splits = NoTest(train=[0, 1, 2, 3], validation=[4, 5])
    payload = {
        "dataset": "ppi",
        "classes": 2,
        "graphs": [
            {
                "x": graph.x,
                "y": graph.y,
                "incidence_edge_index": graph.incidence_edge_index,
            }
            for graph in graphs
        ],
        "splits": splits,
    }
    loaders = {
        "train": DataLoader(graphs[:4], batch_size=2, shuffle=False),
        "validation": DataLoader(graphs[4:], batch_size=2, shuffle=False),
    }
    monkeypatch.setattr(train, "_make_data", lambda *args: (loaders, None))
    args = train.build_parser().parse_args(
        [
            "--dataset",
            "ppi",
            "--condition",
            "relative_c",
            "--output-dir",
            str(tmp_path / "ppi-train"),
            "--epochs",
            "2",
            "--patience",
            "2",
            "--edge-chunk-size",
            "2",
        ]
    )
    args.output_dir.mkdir()
    result = train.train_model(
        payload, {"data_sha256": "b" * 64}, args, torch.device("cpu"), args.output_dir
    )
    assert result["optimizer_steps_per_epoch"] == 2
    assert result["optimizer_steps"] == 2 * result["epochs_run"]
    assert result["best_checkpoint_optimizer_steps"] == 2 * result["best_epoch"]
    assert result["metric_name"] == "micro_f1"
    best = result["diagnostics"]["best_validation"]
    assert best["validation_graph_count"] == 2
    assert best["label_decision_count"] == 12
    assert best["prediction_unit"] == "node_label_decision"
    assert all(
        row["prediction_unit"] == "node_label_decision"
        for row in result["diagnostics"]["best_checkpoint_interventions"]["rows"]
    )
    assert result["diagnostics"]["train_trajectory"][0]["scope"] == (
        "first_actual_training_minibatch_only"
    )
    assert result["topology"]["split_graph_counts"] == {"train": 4, "validation": 2}


def test_real_two_arm_runner_training_checkpoint_and_report(monkeypatch, tmp_path):
    from research.conductance_gat.v3 import report
    from scripts import run_conductance_v3 as runner

    mock_hardware(monkeypatch)
    graph, indices, payload = fixture_data()
    monkeypatch.setattr(train, "_make_data", lambda *args: (graph, indices))
    fixture_file = tmp_path / "tiny_unit_fixture.pt"
    atomic_publish(fixture_file, lambda path: torch.save(payload, path))
    protocol = {
        "data_sha256": sha256_file(fixture_file),
        "dataset": "cora",
        "split": "official_public_masks",
        "task": "node_classification",
        "metric": "accuracy",
        "unit_fixture_only": True,
    }
    monkeypatch.setattr(train, "load_dataset", lambda *args, **kwargs: (payload, protocol))
    monkeypatch.setattr(
        train, "_cache_snapshot", lambda args: {str(fixture_file): sha256_file(fixture_file)}
    )
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"unit_fixture_only": True})
    monkeypatch.setattr(
        runner,
        "_source_snapshot",
        lambda: {"sha256": {"unit_fixture": "a" * 64}, "git_revision": None},
    )
    actual_train = train.train_model
    monkeypatch.setattr(
        train,
        "train_model",
        lambda payload, protocol, args, device, output: actual_train(
            payload, protocol, args, torch.device("cpu"), output
        ),
    )
    executions = []

    def dispatch(command, log, environment):
        if any(Path(argument).name == "gpu_preflight.py" for argument in command):
            return 0
        index = command.index("research.conductance_gat.v3.train")
        args = command[index + 1 :]
        executions.append(args)
        return train.main(args)

    monkeypatch.setattr(runner, "run_logged", dispatch)
    result = runner.main(
        [
            "--datasets",
            "cora",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--edge-chunk-size",
            "2",
            "--results-root",
            str(tmp_path / "results"),
            "--data-root",
            str(tmp_path / "data"),
            "--run-id",
            "unit-fixture",
        ]
    )
    assert result == 0 and len(executions) == 2
    root = tmp_path / "results/conductance_gat/v3/unit-fixture"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    comparison = report.write_comparison(root, manifest)
    assert comparison["status"] == "passed"
    hashes = []
    for condition in CONDITIONS:
        folder = root / "cora" / condition
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        hashes.append(metrics["shared_backbone_initial_state_sha256"])
        assert metrics["research_suite"] == SUITE and metrics["test_evaluated"] is False
        assert metrics["checkpoint_sha256"] == sha256_file(folder / "best.pt")
        assert metrics["history_sha256"] == sha256_file(folder / "history.json")
        assert len(metrics["diagnostics"]["train_trajectory"]) == 2
        assert len(metrics["diagnostics"]["best_checkpoint_interventions"]["rows"]) == 4
        assert len(metrics["diagnostics"]["best_validation"]["layers"]) == 2
        assert all("rho" not in row for row in metrics["diagnostics"]["best_validation"]["layers"])
        saved = torch.load(folder / "best.pt", weights_only=True)
        assert saved["source_sha256"] == metrics["source_sha256"]
        assert saved["architecture"]["normalization"] == "symmetric"
    assert len(set(hashes)) == 1


def test_scalar_nan_gradient_blocks_optimizer_step(monkeypatch, tmp_path):
    mock_hardware(monkeypatch)
    graph, indices, payload = fixture_data()
    monkeypatch.setattr(train, "_make_data", lambda *args: (graph, indices))
    factory = train.RelativeCNodeClassifier

    def poisoned(*args, **kwargs):
        model = factory(*args, **kwargs)
        model.operators[0].raw_alpha.register_hook(lambda gradient: gradient * float("nan"))
        return model

    monkeypatch.setattr(train, "RelativeCNodeClassifier", poisoned)
    steps = []
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: steps.append(1))
    args = arguments(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FloatingPointError, match="gradient"):
        train.train_model(
            payload, {"data_sha256": "b" * 64}, args, torch.device("cpu"), args.output_dir
        )
    assert not steps


def test_missing_cache_writes_failed_metrics_without_training(monkeypatch, tmp_path):
    mock_hardware(monkeypatch)

    def missing(*args, **kwargs):
        assert kwargs["allow_download"] is False
        raise FileNotFoundError("official fixture cache absent")

    monkeypatch.setattr(train, "load_dataset", missing)
    with pytest.raises(FileNotFoundError):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "relative_c",
                "--output-dir",
                str(tmp_path / "failed"),
            ]
        )
    record = json.loads((tmp_path / "failed/metrics.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed" and "absent" in record["error"]


def test_midrun_source_change_refused(monkeypatch, tmp_path):
    mock_hardware(monkeypatch)
    graph, indices, payload = fixture_data()
    monkeypatch.setattr(train, "_make_data", lambda *args: (graph, indices))
    sources = iter([{"fixture": "a" * 64}, {"fixture": "b" * 64}])
    monkeypatch.setattr(train, "_source_hashes", lambda: next(sources))
    args = arguments(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(RuntimeError, match="sources changed"):
        train.train_model(
            payload, {"data_sha256": "b" * 64}, args, torch.device("cpu"), args.output_dir
        )
