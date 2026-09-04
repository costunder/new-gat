"""Bounded fixture tests; never download data or run a CPU research experiment."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from research.conductance_gat.ablation import train
from research.conductance_gat.ablation.model import FactorialNodeClassifier, make_optimizer
from research.conductance_gat.ablation.protocol import CONDITIONS


class Graph(SimpleNamespace):
    def to(self, device, **kwargs):
        return Graph(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


def fixture_graph():
    return Graph(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
    )


def args_for(tmp_path, condition="baseline"):
    return train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            condition,
            "--output-dir",
            str(tmp_path / "arm"),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )


def test_parser_one_seed_gpu_fixed_architecture_no_download_flags(tmp_path):
    args = args_for(tmp_path)
    config = train.configuration(args)
    assert args.model_seed == 0 and args.device == "cuda"
    assert args.epochs == 200 and args.patience == 50 and args.batch_size == 2
    assert config["hidden_channels"] == 64 and config["layers"] == 2
    assert config["dropout"] == 0.5 and config["lr"] == 0.005
    assert config["amp"] is config["compile"] is config["tf32"] is False
    assert "allow_download" not in vars(args)


def test_direct_training_rejects_cpu_before_loading_or_mutating(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.train_model({}, {}, args_for(tmp_path), torch.device("cpu"), tmp_path / "arm")
    assert not (tmp_path / "arm").exists()


def test_cli_rejects_cpu_without_outputs(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "baseline",
                "--output-dir",
                str(tmp_path / "arm"),
                "--device",
                "cpu",
            ]
        )
    assert not (tmp_path / "arm").exists()


@pytest.mark.parametrize("option", ["--epochs", "--patience", "--batch-size"])
def test_nonpositive_budgets_rejected_before_cuda(tmp_path, option):
    with pytest.raises(ValueError, match="must be positive"):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "baseline",
                "--output-dir",
                str(tmp_path / "arm"),
                option,
                "0",
            ]
        )


def test_train_loss_reads_only_mask_labels_and_backprops_only_train_logits():
    graph = fixture_graph()
    logits = torch.randn(4, 2, requires_grad=True)
    indices = torch.tensor([0, 1])
    loss, count = train.training_loss(logits, graph, indices)
    assert count == 2 and torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[2:]) == 0
    assert torch.count_nonzero(logits.grad[:2]) > 0


def test_multilabel_loss_counts_all_node_label_elements():
    graph = Graph(y=torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]))
    logits = torch.randn(3, 2, requires_grad=True)
    loss, count = train.training_loss(logits, graph, None)
    assert count == 6
    torch.testing.assert_close(
        loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, graph.y)
    )


class NoTestDict(dict):
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("Test labels/split must never be read")
        return super().__getitem__(key)


def test_validation_ignores_poison_test_targets_and_restores_modes_rng():
    torch.manual_seed(38)
    model = FactorialNodeClassifier(3, 2, hidden_channels=8)
    model.train()
    model.norms[0].eval()
    graph = fixture_graph()
    splits = NoTestDict(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    result = train.evaluate_validation(model, graph, splits, torch.device("cpu"))
    assert result["metric"] in (0.0, 1.0)
    assert result["split"] == "validation"
    assert result["layers"][0]["conductance"]["count"] == 3
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("normalization", ["global_max", "node_degree"])
def test_telemetry_preserves_training_outputs_gradients_adam_update_and_rng(normalization):
    torch.manual_seed(387)
    model = FactorialNodeClassifier(3, 2, hidden_channels=8, normalization=normalization)
    reference = copy.deepcopy(model)
    graph = fixture_graph()
    optimizer = make_optimizer(model, "baseline")
    ref_optimizer = make_optimizer(reference, "baseline")
    torch.manual_seed(200)
    expected = reference(graph)
    ref_loss, _ = train.training_loss(expected, graph, torch.tensor([0, 1]))
    ref_loss.backward()
    ref_optimizer.step()
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(200)
    with train.ForwardObservation(model) as observation:
        actual = model(graph)
    loss, _ = train.training_loss(actual, graph, torch.tensor([0, 1]))
    loss.backward()
    saved_gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    report = train.gradient_observation(model, "baseline")
    assert report["operators.0"]["weight_decay"] == 0.0005
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, saved_gradients[name], atol=0, rtol=0)
    optimizer.step()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert train.state_sha256(model) == train.state_sha256(reference)
    assert observation.summary()[0]["rho"]["count"] == 4
    assert not any(op._forward_hooks or op.estimator._forward_hooks for op in model.operators)


def test_observation_hooks_restore_after_exception():
    model = FactorialNodeClassifier(3, 2, hidden_channels=8)
    with pytest.raises(RuntimeError, match="injected"):
        with train.ForwardObservation(model):
            raise RuntimeError("injected")
    assert not any(op._forward_hooks or op.estimator._forward_hooks for op in model.operators)


def test_zero_decay_ratio_is_null_not_clamped():
    model = FactorialNodeClassifier(3, 2, hidden_channels=8)
    graph = fixture_graph()
    loss, _ = train.training_loss(model(graph), graph, torch.tensor([0, 1]))
    loss.backward()
    result = train.gradient_observation(model, "gate_no_wd")
    gate = result["operators.0"]
    assert gate["task_gradient_norm"] > 0
    assert gate["decay_term_norm"] == 0
    assert gate["task_to_decay_ratio"] is None
    assert gate["ratio_policy"] == "undefined_zero_decay_norm"
    assert result["non_gate"]["weight_decay"] == 0.0005


def test_moments_exact_values_and_quantile_sampling_policy():
    stats = train._moments(torch.tensor([1.0, 2.0, 3.0]))
    assert stats["mean"] == 2.0
    assert stats["std"] == pytest.approx((2 / 3) ** 0.5)
    assert stats["cv"] == pytest.approx((2 / 3) ** 0.5 / 2)
    assert stats["quantiles"]["p50"] == 2.0
    large = train._moments(torch.ones(9000))
    assert large["cv"] == 0 and large["count"] == 9000
    assert large["quantile_sample_count"] == 4096
    assert large["quantile_policy"] == "deterministic_evenly_spaced_sample"
    with pytest.raises(RuntimeError, match="Non-finite"):
        train._moments(torch.tensor([float("nan")]))


def test_pooled_moments_include_between_graph_variation():
    values = [torch.tensor([1.0, 1.0]), torch.tensor([3.0, 3.0, 3.0])]
    records = [train._moments(value) for value in values]
    pooled = train._pooled_moments(records)
    expected = train._moments(torch.cat(values))
    assert pooled["mean"] == expected["mean"]
    assert pooled["std"] == pytest.approx(expected["std"])
    assert pooled["cv"] > 0 and all(record["cv"] == 0 for record in records)


def _install_fake_pyg(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in ("torch_geometric", "torch_geometric.data", "torch_geometric.loader")
    }
    modules["torch_geometric.data"].Data = Graph
    created = []

    class Loader:
        def __init__(self, graphs, **kwargs):
            self.graphs = graphs
            self.kwargs = kwargs
            created.append(self)

    modules["torch_geometric.loader"].DataLoader = Loader
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return created


def test_loaders_construct_only_train_validation_never_test(monkeypatch, tmp_path):
    loaders = _install_fake_pyg(monkeypatch)
    args = args_for(tmp_path)
    args.workers = 4
    payload = {
        "dataset": "ppi",
        "graphs": [{"graph_id": i} for i in range(5)],
        "splits": NoTestDict(train=[0, 1], validation=[2, 3], test=[4]),
    }
    data, indices = train._make_data(payload, args, torch.device("cpu"))
    assert indices is None and set(data) == {"train", "validation"}
    assert len(loaders) == 2
    assert [g.graph_id for loader in loaders for g in loader.graphs] == [0, 1, 2, 3]
    assert loaders[0].kwargs["shuffle"] is True
    assert loaders[1].kwargs["shuffle"] is False
    assert loaders[0].kwargs["generator"] is not loaders[1].kwargs["generator"]
    assert all(loader.kwargs["num_workers"] == 4 for loader in loaders)
    assert all(loader.kwargs["persistent_workers"] is True for loader in loaders)
    assert all(loader.kwargs["prefetch_factor"] == 2 for loader in loaders)
    assert all(loader.kwargs["pin_memory"] is True for loader in loaders)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_checkpoint_is_tagged_and_contains_normalization_provenance(tmp_path, condition):
    args = args_for(tmp_path, condition)
    model = FactorialNodeClassifier(
        3, 2, hidden_channels=64, normalization=CONDITIONS[condition]["normalization"]
    )
    saved = train.checkpoint_payload(model, args, {"data_sha256": "a" * 64}, "b" * 64, 7, 0.4, 70)
    assert saved["model"] == saved["research_suite"] == "conductance_factorial"
    assert saved["architecture"]["normalization"] == CONDITIONS[condition]["normalization"]
    assert saved["test_evaluated"] is False
    assert saved["optimizer_steps"] == 70
    assert saved["cache_sha256"] == "a" * 64


def test_main_missing_offline_cache_records_failure_without_training(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "_require_cuda", lambda device: None)
    seen = {}

    def missing(name, root, *, allow_download):
        seen.update(name=name, root=root, allow_download=allow_download)
        raise FileNotFoundError("official cache missing")

    monkeypatch.setattr(train, "load_dataset", missing)
    monkeypatch.setattr(train, "train_model", lambda *a: pytest.fail("Training must not begin"))
    args = args_for(tmp_path)
    with pytest.raises(FileNotFoundError, match="official cache missing"):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "baseline",
                "--output-dir",
                str(args.output_dir),
                "--data-root",
                str(args.data_root),
            ]
        )
    record = json.loads((args.output_dir / "metrics.json").read_text())
    assert record["status"] == "failed" and record["test_evaluated"] is False
    assert seen["allow_download"] is False


def test_nonempty_output_rejected_without_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "_require_cuda", lambda device: None)
    directory = tmp_path / "arm"
    directory.mkdir()
    keep = directory / "existing.pt"
    keep.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        train.main(["--dataset", "cora", "--condition", "baseline", "--output-dir", str(directory)])
    assert keep.read_bytes() == b"preserve"


def test_fixture_training_loop_writes_selected_validation_only_artifacts(monkeypatch, tmp_path):
    # Explicit CUDA API stubs only for this four-node loop integration test.
    # Production train_model and CLI remain strictly GPU-only (tested above).
    monkeypatch.setattr(train, "_require_cuda", lambda device: None)
    monkeypatch.setattr(train, "_configure_fp32", lambda: None)
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "fixture_not_a_gpu")
    graph = fixture_graph()
    splits = NoTestDict(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    monkeypatch.setattr(train, "_make_data", lambda *a: (graph, splits))
    args = args_for(tmp_path)
    args.epochs = 3
    args.patience = 2
    args.output_dir.mkdir()
    payload = {"graphs": [vars(graph)], "classes": 2, "dataset": "cora"}
    result = train.train_model(
        payload, {"data_sha256": "f" * 64}, args, torch.device("cpu"), args.output_dir
    )
    assert result["status"] == "passed" and result["test_evaluated"] is False
    assert "test" not in result
    assert result["best_epoch"] >= 1
    assert result["optimizer_steps"] == result["epochs_run"]
    assert result["best_checkpoint_optimizer_steps"] == result["best_epoch"]
    assert Path(result["checkpoint"]).is_file() and Path(result["history"]).is_file()
    saved = torch.load(result["checkpoint"], weights_only=True)
    assert saved["research_suite"] == "conductance_factorial"
    trajectory = result["diagnostics"]["train_trajectory"]
    assert len(trajectory) == result["epochs_run"]
    assert [r["optimizer_steps_before_batch"] for r in trajectory] == list(range(len(trajectory)))
    assert all(r["stage"] == "after_task_backward_before_optimizer_step" for r in trajectory)
    assert len(result["initial_state_sha256"]) == 64
