"""Runner protocol checks use bounded mocks, never GPU training or downloads."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from research.cycle_pe.v2 import benchmark
from research.cycle_pe.v2.data import DATASETS
from research.cycle_pe.v2.model import MODEL_NAME


def test_v2_defaults_use_deep_projector_model_on_official_data() -> None:
    args = benchmark.parser().parse_args([])
    assert tuple(args.datasets) == DATASETS == ("zinc12k", "peptides_struct")
    assert (args.hidden_dim, args.pe_dim, args.layers) == (128, 64, 10)
    assert (args.ffn_multiplier, args.dropout, args.layer_scale) == (4, 0.1, 0.1)
    assert (args.epochs, args.patience, args.lr, args.batch_size) == (300, 50, 1e-3, 32)
    assert args.max_parameters == 20_000_000
    assert args.column_chunk_size == 16
    assert args.basis_backend == "thin_q"
    assert args.hardware_profile == "portable" and args.prefetch_factor == 2
    assert args.workers == 4
    assert args.output_dir == Path("results/cycle_pe_v2/benchmark")
    assert MODEL_NAME == "cycle_projector_pe_v2"
    assert not hasattr(args, "baselines")
    assert not hasattr(args, "tiny")
    assert not hasattr(args, "max_cycle_rank")


@pytest.mark.parametrize("flag", ["--baselines", "--tiny", "--max-cycle-rank"])
def test_no_baseline_dummy_or_cycle_truncation_options(flag):
    with pytest.raises(SystemExit):
        benchmark.parser().parse_args([flag])


def test_v2_observability_reports_real_graph_and_batch_counts():
    graph = type("ObservedGraph", (), {})()
    graph.x = torch.zeros(4, 3, dtype=torch.long)
    graph.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    graph.edge_attr = torch.zeros(3, 2, dtype=torch.long)
    graph.y = torch.zeros(1)
    graph.cycle_basis = torch.zeros(3, 1)
    splits = {"train": [graph, graph], "validation": [graph], "test": [graph]}

    data_report = benchmark._cycle_data_observability("zinc12k", splits)
    assert data_report["loaded_graph_count"] == 4
    assert data_report["actual_used_graph_count"] == 4
    assert data_report["actual_used_fraction_of_loaded_graphs"]["value"] == 1.0
    assert data_report["nodes_per_graph"]["total"] == 16
    assert data_report["canonical_undirected_edges_per_graph"]["total"] == 12
    assert data_report["official_full_graph_count"]["value"] == 12_000
    assert data_report["loaded_fraction_of_official_full_dataset"]["value"] == pytest.approx(
        4 / 12_000
    )

    args = benchmark.parser().parse_args([])
    batch_report = benchmark._cycle_batch_observability(
        args,
        effective_batch_size=2,
        training_graphs=5,
        batches_per_epoch=3,
    )
    assert batch_report["configured_physical_batch_size"] == 2
    assert batch_report["gradient_accumulation_steps"] == 1
    assert batch_report["effective_batch_size"] == 2
    assert batch_report["planned_maximum_training_batches"] == args.epochs * 3


def test_v2_optimizer_ownership_and_first_task_gradients_are_fail_closed() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters())
    ownership = benchmark._validate_optimizer_ownership(model, optimizer)
    assert ownership["validated"] is True
    assert (
        ownership["trainable_parameter_scalars"] == ownership["optimizer_owned_parameter_scalars"]
    )

    model(torch.ones(2, 3)).sum().backward()
    gradients = benchmark._validate_first_task_gradients(model)
    assert gradients["validated"] is True
    assert gradients["missing_gradient_parameters"] == []
    assert gradients["nonfinite_gradient_parameters"] == []

    missing_owner = torch.optim.Adam(model[0].parameters())
    with pytest.raises(RuntimeError, match="optimizer ownership mismatch"):
        benchmark._validate_optimizer_ownership(model, missing_owner)


def test_v2_first_task_gradient_validation_rejects_disconnected_and_nonfinite() -> None:
    class Disconnected(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Linear(2, 1)
            self.unused = torch.nn.Parameter(torch.ones(1))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.used(value)

    disconnected = Disconnected()
    disconnected(torch.ones(1, 2)).sum().backward()
    with pytest.raises(RuntimeError, match="disconnected from the first task loss"):
        benchmark._validate_first_task_gradients(disconnected)

    connected = torch.nn.Linear(2, 1)
    connected(torch.ones(1, 2)).sum().backward()
    assert connected.weight.grad is not None
    connected.weight.grad.fill_(torch.inf)
    with pytest.raises(FloatingPointError, match="nonfinite first-step task gradients"):
        benchmark._validate_first_task_gradients(connected)


def test_cpu_benchmark_training_and_invalid_chunk_size_are_rejected() -> None:
    args = benchmark.parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark._validate(args)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark._train_model("zinc12k", {}, args)
    args.prepare_only = True
    benchmark._validate(args)
    args.column_chunk_size = 0
    with pytest.raises(ValueError, match="column-chunk-size"):
        benchmark._validate(args)


def test_hashes_include_basis_data_encoder_and_reused_backbone_sources() -> None:
    hashes = benchmark.implementation_hashes()
    assert {
        "research/cycle_pe/v2/benchmark.py",
        "research/cycle_pe/v2/basis.py",
        "research/cycle_pe/v2/data.py",
        "research/cycle_pe/v2/model.py",
        "research/cycle_pe/benchmark_data.py",
        "research/cycle_pe/benchmark_models.py",
        "research/cycle_pe/paper_model.py",
        "research/cycle_pe/resource_monitor.py",
        "src/chartgat/observability.py",
    } <= set(hashes)
    root = Path(benchmark.__file__).resolve().parents[3]
    for name, value in hashes.items():
        assert value == hashlib.sha256((root / name).read_bytes()).hexdigest()


def test_prepare_only_records_separate_version_without_claiming_training(tmp_path, monkeypatch):
    loaded = []

    def fake_load(root, dataset, *, allow_download, basis_backend):
        loaded.append((dataset, allow_download, basis_backend))
        return {}, {"official_splits": True, "unit_fixture_only": True, "basis": "full_left_null"}

    monkeypatch.setattr(benchmark, "load_benchmark", fake_load)
    monkeypatch.setattr(benchmark, "_train_model", lambda *a: pytest.fail("must not train"))
    output = tmp_path / "v2"
    assert (
        benchmark.main(
            [
                "--datasets",
                "zinc12k",
                "--prepare-only",
                "--device",
                "cpu",
                "--output-dir",
                str(output),
                "--data-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert loaded == [("zinc12k", False, "thin_q")]
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for document in (metrics, manifest):
        assert document["status"] == "prepared"
        assert document["track"] == "cycle_pe"
        assert document["version"] == "v2"
    assert metrics["datasets"]["zinc12k"]["models"] == {}
    assert manifest["controls"]["model"] == "cycle_projector_pe_v2"
    assert "no truncation" in manifest["controls"]["basis_input"]
    assert manifest["controls"]["basis_rank_dependent_parameters"] is False
    assert manifest["controls"]["basis_backend"] == "thin_q"
    assert "projector is chart invariant" in manifest["seeds"]["chart_seed"]
    manifest_bytes = (output / "manifest.json").read_bytes()
    metrics_bytes = (output / "metrics.json").read_bytes()
    # A different dataset/data-root command must fail before mutating artifacts.
    with pytest.raises(ValueError, match="does not match"):
        benchmark.main(["--prepare-only", "--output-dir", str(output)])
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert (output / "metrics.json").read_bytes() == metrics_bytes


def test_dfs_backend_is_recorded_and_forwarded_to_data_preparation(tmp_path, monkeypatch):
    seen = []

    def fake_load(*args, **kwargs):
        seen.append(kwargs["basis_backend"])
        return {}, {"basis_backend": kwargs["basis_backend"], "official_splits": True}

    monkeypatch.setattr(benchmark, "load_benchmark", fake_load)
    output = tmp_path / "dfs"
    benchmark.main(
        [
            "--datasets",
            "zinc12k",
            "--prepare-only",
            "--device",
            "cpu",
            "--basis-backend",
            "dfs_fundamental",
            "--output-dir",
            str(output),
            "--data-root",
            str(tmp_path),
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert seen == ["dfs_fundamental"]
    assert manifest["arguments"]["basis_backend"] == "dfs_fundamental"
    assert manifest["controls"]["basis_backend"] == "dfs_fundamental"
    assert "not an end-to-end linear-time speedup" in manifest["controls"]["basis_backend_runtime"]


def test_only_v2_model_is_dispatched_once_per_official_dataset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "_validate", lambda args: None)
    monkeypatch.setattr(benchmark, "load_benchmark", lambda *args, **kwargs: ({}, {}))

    def fake_train(dataset, splits, args):
        calls.append(dataset)
        return {"test": 0.5, "validation": 0.4}

    monkeypatch.setattr(benchmark, "_train_model", fake_train)
    output = tmp_path / "v2_only"
    benchmark.main(["--output-dir", str(output), "--data-root", str(tmp_path)])
    assert calls == list(DATASETS)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "passed"
    assert all(set(entry["models"]) == {MODEL_NAME} for entry in metrics["datasets"].values())


def test_preparation_failure_is_persisted_not_reported_as_success(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("invalid left-nullspace basis")

    monkeypatch.setattr(benchmark, "load_benchmark", broken)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="left-nullspace"):
        benchmark.main(["--prepare-only", "--output-dir", str(output)])
    for filename in ("manifest.json", "metrics.json"):
        document = json.loads((output / filename).read_text(encoding="utf-8"))
        assert document["status"] == "failed"
        assert document["version"] == "v2"


def test_runner_selects_validation_checkpoint_before_single_test_evaluation() -> None:
    source = inspect.getsource(benchmark._train_model)
    assert source.count("evaluate(model, test_loader, device, amp=args.amp)") == 1
    assert source.index(
        'model.load_state_dict(selected["state_dict"], strict=True)'
    ) < source.index("evaluate(model, test_loader, device, amp=args.amp)")
    assert "if validation < best:" in source
    assert "weights_only=True" in source


def test_selected_test_path_reports_actual_resources_and_throughput() -> None:
    source = inspect.getsource(benchmark._evaluate_test_checkpoint)
    assert "FailureSafeResourceMonitor(" in source
    assert "@resource_failure_boundary" in source
    assert "resource_monitor.start()" in source
    assert "resource_monitor.finish(" in source
    assert '"resource_observability": resource_observability' in source
    assert '"evaluation_graphs_per_second"' in source
    assert '"optimizer_created": False' in source


def test_two_slot_best_checkpoint_recovers_exact_previous_bytes(tmp_path):
    best = tmp_path / "best.pt"
    previous = tmp_path / "best.previous.pt"
    old_bytes, interrupted_new_bytes = b"selected-best", b"new-before-last"
    previous.write_bytes(old_bytes)
    best.write_bytes(interrupted_new_bytes)
    expected = hashlib.sha256(old_bytes).hexdigest()
    benchmark._recover_best_checkpoint(best, previous, expected)
    assert best.read_bytes() == old_bytes
    with pytest.raises(ValueError, match="recovery slot"):
        benchmark._recover_best_checkpoint(best, previous, "0" * 64)


def test_exact_multidataset_resume_skips_completed_datasets(tmp_path, monkeypatch):
    trained, loaded = [], []
    monkeypatch.setattr(benchmark, "_validate", lambda args: None)

    def fake_load(root, dataset, **kwargs):
        loaded.append(dataset)
        return {}, {"official_splits": True}

    def fake_train(dataset, splits, args):
        trained.append(dataset)
        return {"validation": 0.5, "fresh_training": True}

    monkeypatch.setattr(benchmark, "load_benchmark", fake_load)
    monkeypatch.setattr(benchmark, "_train_model", fake_train)
    output = tmp_path / "resume"
    command = ["--output-dir", str(output), "--data-root", str(tmp_path)]
    assert benchmark.main(command) == 0
    assert trained == loaded == list(DATASETS)

    monkeypatch.setattr(benchmark, "_completed_training_dataset", lambda *args: True)
    assert benchmark.main(command) == 0
    assert trained == loaded == list(DATASETS)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed" and manifest["resume_count"] == 1


def test_epoch_checkpoint_contract_contains_all_resume_state():
    source = inspect.getsource(benchmark._train_model)
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
        "history",
        "best_checkpoint_sha256",
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "peak_gpu_reserved_bytes",
        "effective_batch_size",
        "batch_calibration",
        "first_task_gradient_connectivity",
    ):
        assert f'"{field}"' in source
    assert "best_checkpoint_bytes" not in source
    assert 'previous_checkpoint = run / "best.previous.pt"' in source


def test_a6000_probe_never_resizes_training_batch_midrun() -> None:
    source = inspect.getsource(benchmark._calibrate_batch_size)
    assert "candidate //" not in source
    assert 'automatic_backoff": False' in source
    assert "model.train()" in source and "torch.random.fork_rng" in source
    train_source = inspect.getsource(benchmark._train_model)
    assert "effective_batch_size = preview.get" in train_source
    assert "batch_calibration.get" in train_source


def test_completed_dataset_skip_requires_exact_artifact_hashes_and_complete_last(tmp_path):
    args = benchmark.parser().parse_args(["--datasets", "zinc12k", "--output-dir", str(tmp_path)])
    args.output_dir = tmp_path.resolve()
    run = args.output_dir / "zinc12k" / MODEL_NAME
    run.mkdir(parents=True)
    best, history, last = run / "best.pt", run / "history.json", run / "last.pt"
    best.write_bytes(b"best")
    history.write_text("[]", encoding="utf-8")
    torch.save(
        {
            "complete": True,
            "resume_configuration": benchmark._resume_configuration("zinc12k", args),
        },
        last,
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = {
        "checkpoint": str(best),
        "checkpoint_sha256": digest(best),
        "history": str(history),
        "history_sha256": digest(history),
        "last_checkpoint": str(last),
        "last_checkpoint_sha256": digest(last),
        "fresh_training": True,
    }
    entry = {"metric": "mae", "models": {MODEL_NAME: result}}
    assert benchmark._completed_training_dataset(entry, "zinc12k", args)
    history.write_text("corrupt", encoding="utf-8")
    assert not benchmark._completed_training_dataset(entry, "zinc12k", args)
