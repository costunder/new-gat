"""Read-only diagnosis contracts; CPU math fixtures, never research training."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_conductance.py"


@pytest.fixture
def diag():
    return importlib.import_module("scripts.diagnose_conductance")


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_help_works_without_site_packages_or_torch():
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--ablate-graph" in result.stdout
    assert "--device" in result.stdout
    assert "--full-audit" in result.stdout
    assert "--gradient-mode" in result.stdout
    assert "Traceback" not in result.stderr


def test_cuda_gate_rejects_cpu_and_missing_gpu(diag, torch, monkeypatch):
    with pytest.raises((RuntimeError, ValueError), match="CUDA|cuda|GPU"):
        diag.require_cuda("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((RuntimeError, ValueError), match="CUDA|cuda|GPU"):
        diag.require_cuda("cuda")


def _layer_record(diag, torch, states, edges, monkeypatch):
    from research.conductance_gat.benchmark import ConductanceConv

    module = ConductanceConv(1)
    monkeypatch.setattr(module.estimator, "forward", lambda gradient, _: gradient[:, 0].abs() + 1)
    state = torch.tensor(states, dtype=torch.float32).reshape(-1, 1)
    incidence = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T
    graph_ids = torch.zeros(len(state), dtype=torch.long)
    # Keep the public diagnostic compatible with the pre-optimization server API.
    inputs = (state, incidence, graph_ids)
    with torch.no_grad():
        output = module(*inputs)
        return diag.layer_diagnostics(module, inputs, output, edge_chunk_size=1)


def test_layer_diagnostics_c_cv_and_graph_local_rho(diag, torch, monkeypatch):
    # c=(2,4), weighted degrees=(2,6,4,0), including one isolated node.
    record = _layer_record(diag, torch, [0, 1, 4, 5], [(0, 1), (1, 2)], monkeypatch)
    torch.testing.assert_close(record["_c"], torch.tensor([2.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(record["_degree"], torch.tensor([2.0, 6.0, 4.0, 0.0]))
    torch.testing.assert_close(record["_rho"], torch.tensor([0.95 / 3, 0.95, 1.9 / 3, 0.0]))
    assert record["c_cv"] == pytest.approx(1 / 3)
    assert record["c_count"] == 2
    assert record["c_sum"] == pytest.approx(6)
    assert record["c_squared_sum"] == pytest.approx(20)
    # A separate graph has its own maximum; the first graph's degree six must
    # not suppress this graph's degree-two nodes.
    other = _layer_record(diag, torch, [0, 1], [(0, 1)], monkeypatch)
    torch.testing.assert_close(other["_rho"], torch.tensor([0.95, 0.95]))


def test_layer_diagnostics_empty_edges_are_finite_and_explicit(diag, torch, monkeypatch):
    record = _layer_record(diag, torch, [1, 2, 3], [], monkeypatch)
    assert record["_c"].numel() == 0
    assert record["c_count"] == 0
    assert record["c_cv"] is None
    assert record["global_update_ratio"] == pytest.approx(0)
    torch.testing.assert_close(record["_rho"], torch.zeros(3))
    torch.testing.assert_close(record["_degree"], torch.zeros(3))


def test_prediction_helpers_use_global_ppi_counts_not_graph_macro(diag, torch):
    first = diag.prediction_statistics(
        torch.tensor([[1.0, -1.0]]), torch.tensor([[1.0, 0.0]]), multilabel=True
    )
    second = diag.prediction_statistics(-torch.ones(3, 2), torch.ones(3, 2), multilabel=True)
    result = diag.merge_predictions([first, second], multilabel=True)
    # Global TP=1, predicted positives=1, true positives=7 -> F1=0.25.
    # Averaging the two graph F1 values would incorrectly give 0.5.
    assert result["metric"] == pytest.approx(0.25)
    assert result["metric_name"] == "micro_f1"
    assert result["predicted_positive_fraction"] == pytest.approx(1 / 8)
    assert result["true_positive_fraction"] == pytest.approx(7 / 8)


def test_layer_pooled_statistics_do_not_average_unequal_graph_sizes(diag, torch, monkeypatch):
    first = _layer_record(diag, torch, [0, 1, 4, 5], [(0, 1), (1, 2)], monkeypatch)
    second = _layer_record(diag, torch, [0, 1], [(0, 1)], monkeypatch)
    summary = diag.summarize_layers([first, second])
    assert summary["graphs"] == 2
    assert summary["graph_macro"]["rho_mean"] == pytest.approx(0.7125)
    assert summary["node_pooled"]["rho"]["mean"] == pytest.approx(3.8 / 6)
    assert summary["edge_pooled"]["conductance"]["mean"] == pytest.approx(8 / 3)
    assert summary["edge_pooled"]["c_cv"] == pytest.approx(2**0.5 / 4)
    assert summary["node_pooled"]["rho_below"]["0.01"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_layer_diagnostics_reject_invalid_chunks(diag, torch, chunk_size):
    from research.conductance_gat.benchmark import ConductanceConv

    state = torch.ones(2, 1)
    inputs = (state, torch.empty(2, 0, dtype=torch.long), torch.zeros(2, dtype=torch.long), 1)
    with pytest.raises(ValueError):
        diag.layer_diagnostics(ConductanceConv(1), inputs, state, edge_chunk_size=chunk_size)


def test_layer_diagnostics_reject_multi_graph_inputs(diag, torch):
    from research.conductance_gat.benchmark import ConductanceConv

    state = torch.ones(2, 1)
    inputs = (state, torch.empty(2, 0, dtype=torch.long), torch.tensor([0, 1]), 2)
    with pytest.raises(ValueError, match="one graph"):
        diag.layer_diagnostics(ConductanceConv(1), inputs, state)


def _run_records():
    config = {
        "datasets": ["cora"],
        "model_seed": 0,
        "hidden_channels": 4,
        "layers": 2,
        "dropout": 0.1,
    }
    history = [
        {"epoch": 1, "train_loss": 1.0, "validation": 0.5},
        {"epoch": 2, "train_loss": 0.9, "validation": 0.6},
        {"epoch": 3, "train_loss": 0.8, "validation": 0.6},
    ]
    saved = {"best_epoch": 2, "epochs_run": 3, "validation": 0.6, "test": 0.4}
    common = {
        "schema_version": 2,
        "track": "conductance_gat",
        "suite": "benchmark",
        "status": "passed",
    }
    manifest = {
        **common,
        "config": config,
        "completed": ["cora/conductance"],
        "expected": ["cora/conductance"],
    }
    metrics = {**common, "model_seed": 0, "datasets": {"cora": {"models": {"conductance": saved}}}}
    return manifest, metrics, config, saved, history


def test_run_and_history_validate_first_tied_best_checkpoint(diag):
    manifest, metrics, config, saved, history = _run_records()
    assert diag.validate_run(manifest, metrics, 0, ["cora"]) == config
    summary = diag.summarize_history(history, saved)
    assert summary["best_epoch"] == 2
    assert summary["train_loss_min"] == 0.8
    assert summary["train_loss_at_selected_epoch"] == 0.9
    with pytest.raises(ValueError):
        diag.summarize_history(history, {**saved, "best_epoch": 3})


@pytest.mark.parametrize("status", ["failed", "running", "prepared", None])
@pytest.mark.parametrize("which", ["manifest", "metrics"])
def test_incomplete_and_preparation_runs_rejected(diag, status, which):
    manifest, metrics, *_ = _run_records()
    (manifest if which == "manifest" else metrics)["status"] = status
    with pytest.raises(ValueError, match="completed|passed"):
        diag.validate_run(manifest, metrics, 0, ["cora"])


@pytest.mark.parametrize("selected", [[], ["ppi"], ["cora", "cora"]])
def test_missing_or_duplicate_dataset_selection_rejected(diag, selected):
    manifest, metrics, *_ = _run_records()
    with pytest.raises(ValueError):
        diag.validate_run(manifest, metrics, 0, selected)


def test_wrong_seed_and_completion_manifest_rejected(diag):
    manifest, metrics, *_ = _run_records()
    with pytest.raises(ValueError, match="seed"):
        diag.validate_run(manifest, metrics, 2, ["cora"])
    manifest["completed"] = []
    with pytest.raises(ValueError, match="expected|completed"):
        diag.validate_run(manifest, metrics, 0, ["cora"])


def test_relative_run_root_and_cli_defaults(diag, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = diag.build_parser().parse_args(["--run-id", "example", "--results-root", "outputs"])
    assert (
        diag.resolve_run(args)
        == tmp_path / "outputs/conductance_gat/example/model-seed-0/benchmark"
    )
    assert args.device == "cuda"
    assert args.output_dir is None
    assert args.ablate_graph is False
    assert args.model_seed == 0
    assert args.datasets == list(diag.DATASETS)
    assert args.gradient_batches == 1
    assert args.gradient_mode == "eval"
    assert "test" not in vars(args)
    args.run_id = "../elsewhere"
    with pytest.raises(ValueError):
        diag.resolve_run(args)


def test_missing_manifest_fails_without_creating_a_run(diag, torch, tmp_path, capsys):
    assert diag.main(["--run-id", "missing", "--results-root", str(tmp_path)]) == 1
    assert "FileNotFoundError" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "target",
    ["recorded_data", "recorded_data_with_override", "explicit_data", "source_seed", "other_seed"],
)
def test_output_protection_precedes_any_new_directory(diag, tmp_path, monkeypatch, target):
    results_root = tmp_path / "outputs"
    run = results_root / "conductance_gat/example/model-seed-0/benchmark"
    run.mkdir(parents=True)
    data_root = tmp_path / "external_dataset_store"
    (run / "manifest.json").write_text(
        json.dumps({"config": {"data_root": str(data_root)}}), encoding="utf-8"
    )
    args = ["--run-id", "example", "--results-root", str(results_root)]
    if target in {"recorded_data", "recorded_data_with_override"}:
        output = data_root / "diagnosis"
        if target == "recorded_data_with_override":
            args += ["--data-root", str(tmp_path / "override_dataset_store")]
    elif target == "explicit_data":
        data_root = tmp_path / "override_dataset_store"
        args += ["--data-root", str(data_root)]
        output = data_root / "diagnosis"
    elif target == "source_seed":
        output = run / "diagnosis"
    else:
        output = run.parents[1] / "model-seed-4/diagnosis"

    def forbidden(*_args, **_kwargs):
        pytest.fail("Protected output must be rejected before diagnostic work")

    monkeypatch.setattr(diag, "_diagnose", forbidden)
    before = _file_snapshot(tmp_path)
    with pytest.raises(ValueError, match="source run/data"):
        diag.main([*args, "--output-dir", str(output)])
    assert not output.exists()
    assert _file_snapshot(tmp_path) == before


def test_extended_audit_auto_output_is_new_and_separate_from_source(diag, tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "ROOT", tmp_path)
    run = tmp_path / "research/conductance_gat/results/paper/example/model-seed-0/benchmark"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    seen = []

    def fake_diagnose(args, source, report):
        seen.append((args.model_seed, args.datasets, source))
        assert args.full_audit
        report["datasets"]["cora"] = {"status": "passed", "stage": "complete"}

    monkeypatch.setattr(diag, "_diagnose", fake_diagnose)
    source_before = _file_snapshot(run)
    assert diag.main(["--run-id", "example", "--full-audit"]) == 0
    output = next((tmp_path / "runs/diagnostics").iterdir())
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["policy"]["model_seed_count"] == 1
    assert report["policy"]["optimizer_steps"] == 0
    assert report["policy"]["new_test_queries"] is False
    assert report["model_seed"] == 0
    assert report["status"] == "passed"
    assert (output / "report.md").is_file()
    assert seen == [(0, list(diag.DATASETS), run)]
    assert _file_snapshot(run) == source_before


def test_diagnostic_wrapper_records_resource_and_throughput(diag, torch, tmp_path, monkeypatch):
    args = diag.build_parser().parse_args(["--run-id", "example", "--device", "cuda"])
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(diag, "require_cuda", lambda _requested: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)

    def bounded(_args, _run, report, device):
        assert device.type == "cpu"
        report["datasets"]["cora"] = {"status": "passed"}

    monkeypatch.setattr(diag, "_diagnose_impl", bounded)
    report = {"datasets": {}}
    diag._diagnose(args, tmp_path, report)

    resources = report["resource_observability"]
    assert resources["measurement_scope"]
    assert resources["summary"]["run_average_gpu_sm_utilization_percent"]["value"] is None
    assert report["throughput"]["completed_dataset_audits"] == 1
    assert report["throughput"]["dataset_audits_per_second"]["value"] > 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--shuffle-seed", "-1"],
        ["--gradient-batches", "0"],
        ["--gradient-sample-limit", "0"],
        ["--near-zero-threshold", "nan"],
        ["--near-zero-threshold", "-1"],
    ],
)
def test_invalid_extended_arguments_fail_before_any_output(diag, tmp_path, arguments):
    with pytest.raises(ValueError, match="Invalid audit"):
        diag.main(
            [
                "--run-id",
                "example",
                "--full-audit",
                "--output-dir",
                str(tmp_path / "new"),
                *arguments,
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_extended_failure_preserves_completed_diagnostics_in_reports(diag, tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "ROOT", tmp_path)
    run = tmp_path / "research/conductance_gat/results/paper/example/model-seed-0/benchmark"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"config": {}}), encoding="utf-8")

    def fail_after_first(_args, _source, report):
        report["datasets"]["cora"] = {"status": "passed", "stage": "complete"}
        report["active_dataset"] = "ogbn-arxiv"
        report["datasets"]["ogbn-arxiv"] = {
            "status": "running",
            "stage": "train_label_gradient_audit",
        }
        raise RuntimeError("CUDA out of memory: unit-only injected failure")

    monkeypatch.setattr(diag, "_diagnose", fail_after_first)
    assert diag.main(["--run-id", "example", "--full-audit"]) == 1
    output = next((tmp_path / "runs/diagnostics").iterdir())
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["datasets"]["cora"]["status"] == "passed"
    assert report["datasets"]["ogbn-arxiv"]["status"] == "failed"
    assert "No CPU fallback" in report["recovery_note"]
    assert "out of memory" in (output / "report.md").read_text(encoding="utf-8")


def test_audit_cli_accepts_one_checkpoint_seed_not_seed_sweep(diag):
    assert (
        diag.build_parser().parse_args(["--run-id", "example", "--model-seed", "3"]).model_seed == 3
    )
    with pytest.raises(SystemExit):
        diag.build_parser().parse_args(["--run-id", "example", "--model-seed", "0", "1"])


def test_full_audit_real_helpers_integrate_without_optimizer_or_test_labels(
    diag, torch, monkeypatch
):
    payload, checkpoint, _, _, config, saved, history = _payload_and_checkpoint(torch)
    config["weight_decay"] = 0.0005
    payload["graphs"][0]["y"][payload["splits"]["test"]] = 999
    model = diag.restore_model(checkpoint, payload, config, saved, history, torch.device("cpu"))
    before = diag._state_hash(model)
    original_grads = [parameter.grad for parameter in model.parameters()]
    args = diag.build_parser().parse_args(["--run-id", "example", "--full-audit", "--ablate-graph"])
    item = {"baseline": diag.evaluate_checkpoint(model, payload, torch.device("cpu"), 2)}

    def forbidden(*_args, **_kwargs):
        pytest.fail("Diagnostic audit must not construct an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    diag.additional_audits(model, payload, torch.device("cpu"), args, config, item)
    assert len(item["interventions"]["variants"]) == 10
    assert item["gate_audit"]["label_scope"] == "train_only"
    assert item["gate_audit"]["loss"]["batches"] == 1
    assert item["gate_audit"]["loss"]["train_label_elements"] == 2
    assert item["gate_audit"]["mode"] == "eval"
    assert diag._state_hash(model) == before
    assert all(
        parameter.grad is gradient
        for parameter, gradient in zip(model.parameters(), original_grads, strict=True)
    )
    assert all(not module._forward_hooks for module in model.modules())
    report = {"status": "passed", "model_seed": 0, "datasets": {"cora": item}}
    json.dumps(report, allow_nan=False)
    readable = diag.render_report(report)
    assert "mean_C_all" in readable
    assert "Task gradient norm" in readable
    assert "operators.0.estimator.network.0.weight" in readable


def test_existing_output_directory_is_never_overwritten(diag, tmp_path, monkeypatch):
    results_root = tmp_path / "outputs"
    run = results_root / "conductance_gat/example/model-seed-0/benchmark"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    output = tmp_path / "diagnosis"
    output.mkdir()
    (output / "report.json").write_text("keep existing report", encoding="utf-8")
    before = _file_snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        diag.main(
            [
                "--run-id",
                "example",
                "--results-root",
                str(results_root),
                "--output-dir",
                str(output),
            ]
        )
    assert _file_snapshot(tmp_path) == before


def _payload_and_checkpoint(torch):
    from research.conductance_gat.benchmark import ConductanceNodeClassifier
    from research.conductance_gat.benchmark_data import canonical_edges

    manifest, metrics, config, saved, history = _run_records()
    edges, incidence = canonical_edges(torch.tensor([[0, 1, 3], [1, 2, 4]]), 6)
    graph = {
        "x": torch.arange(18, dtype=torch.float32).reshape(6, 3) / 18,
        "y": torch.tensor([0, 1, 0, 1, 0, 1]),
        "edge_index": edges,
        "incidence_edge_index": incidence,
    }
    masks = {
        name: torch.tensor([index in positions for index in range(6)])
        for name, positions in {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}.items()
    }
    payload = {"dataset": "cora", "classes": 2, "graphs": [graph], "splits": masks}
    architecture = {key: config[key] for key in ("hidden_channels", "layers", "dropout")}
    model = ConductanceNodeClassifier(3, 2, **architecture).eval()
    checkpoint = {
        "model": "conductance",
        "dataset": "cora",
        "best_epoch": 2,
        "validation": 0.6,
        "architecture": architecture,
        "state_dict": model.state_dict(),
    }
    return payload, checkpoint, manifest, metrics, config, saved, history


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [("dataset", "ppi"), ("model", "gcn"), ("best_epoch", 1), ("validation", 0.5)],
)
def test_checkpoint_metadata_must_match_saved_run(diag, torch, bad_field, bad_value):
    payload, checkpoint, _, _, config, saved, history = _payload_and_checkpoint(torch)
    checkpoint[bad_field] = bad_value
    with pytest.raises(ValueError):
        diag.restore_model(checkpoint, payload, config, saved, history, torch.device("cpu"))


@pytest.mark.parametrize("fault", ["architecture", "missing_key", "unexpected_key", "shape"])
def test_checkpoint_strict_architecture_and_parameter_load(diag, torch, fault):
    payload, checkpoint, _, _, config, saved, history = _payload_and_checkpoint(torch)
    if fault == "architecture":
        checkpoint["architecture"]["layers"] = 3
    elif fault == "missing_key":
        checkpoint["state_dict"].pop("encoder.weight")
    elif fault == "unexpected_key":
        checkpoint["state_dict"]["not_a_parameter"] = torch.zeros(1)
    else:
        checkpoint["state_dict"]["encoder.weight"] = torch.zeros(1, 1)
    with pytest.raises((ValueError, RuntimeError)):
        diag.restore_model(checkpoint, payload, config, saved, history, torch.device("cpu"))


def test_full_train_and_validation_only_eval_preserves_parameters_and_hooks(diag, torch):
    payload, checkpoint, _, _, config, saved, history = _payload_and_checkpoint(torch)
    model = diag.restore_model(checkpoint, payload, config, saved, history, torch.device("cpu"))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    # Out-of-domain test labels make an accidental test loss evaluation fail.
    payload["graphs"][0]["y"][payload["splits"]["test"]] = 999
    assert not model.training
    baseline = diag.evaluate_checkpoint(model, payload, torch.device("cpu"), 2)
    ablated = diag.evaluate_checkpoint(model, payload, torch.device("cpu"), 2, ablate=True)
    assert set(baseline) == {"train", "validation"}
    assert set(ablated) == {"validation"}
    for split in baseline:
        assert baseline[split]["prediction"]["count"] == int(payload["splits"][split].sum())
    assert "layers" not in ablated["validation"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(not operator._forward_hooks for operator in model.operators)


def test_ppi_evaluation_includes_every_train_graph_not_test_graphs(diag, torch):
    from research.conductance_gat.benchmark import ConductanceNodeClassifier

    # Unit-only graphs carry logits directly; model evaluation still exercises
    # the real per-split loop and prediction merger without a PyG dependency.
    class LogitModel(ConductanceNodeClassifier):
        def __init__(self):
            super().__init__(2, 2, hidden_channels=2, layers=1, dropout=0)
            self.operators = torch.nn.ModuleList()

        def forward(self, graph):
            return graph.x

    def graph(x, y):
        return {"x": x, "y": y, "incidence_edge_index": torch.empty(2, 0, dtype=torch.long)}

    payload = {
        "dataset": "ppi",
        "classes": 2,
        "graphs": [
            graph(torch.tensor([[1.0, -1.0]]), torch.tensor([[1.0, 0.0]])),
            graph(-torch.ones(3, 2), torch.ones(3, 2)),
            graph(torch.ones(2, 2), torch.ones(2, 2)),
            {},  # Accessing a held-out graph is forbidden.
        ],
        "splits": {"train": [0, 1], "validation": [2], "test": [3]},
    }
    model = LogitModel().eval()
    result = diag.evaluate_checkpoint(model, payload, torch.device("cpu"), 2)
    assert result["train"]["prediction"]["nodes"] == 4
    assert result["train"]["prediction"]["metric"] == pytest.approx(0.25)
    assert result["validation"]["prediction"]["metric"] == pytest.approx(1)
    assert set(diag.evaluate_checkpoint(model, payload, torch.device("cpu"), 2, ablate=True)) == {
        "validation"
    }


def test_missing_cache_never_calls_downloader_or_creates_directory(torch, tmp_path, monkeypatch):
    from research.conductance_gat import benchmark_data

    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline diagnosis must not call a dataset downloader")

    monkeypatch.setattr(benchmark_data, "_download_official", forbidden)
    missing = tmp_path / "data"
    with pytest.raises(FileNotFoundError):
        benchmark_data.load_dataset("cora", missing, allow_download=False)
    assert not missing.exists()


def _file_snapshot(directory):
    return {
        str(path.relative_to(directory)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_checkpoint_restore_eval_and_offline_cache_are_read_only(
    diag, torch, tmp_path, monkeypatch
):
    from research.conductance_gat import benchmark_data

    payload, checkpoint, manifest, metrics, config, saved, history = _payload_and_checkpoint(torch)
    monkeypatch.setitem(
        benchmark_data.EXPECTED,
        "cora",
        {"nodes": 6, "features": 3, "classes": 2, "splits": [2, 2, 2]},
    )
    cache = tmp_path / "data/conductance_gat/matched_benchmark_v1/cora"
    cache.mkdir(parents=True)
    torch.save(payload, cache / "data.pt")
    protocol = {
        "schema_version": 1,
        "dataset": "cora",
        "source_url": benchmark_data.SOURCES["cora"],
        "source_files_sha256": {"unit-only-fixture": "not-an-official-download"},
        "data_sha256": benchmark_data.sha256_file(cache / "data.pt"),
        "split_sha256": {
            key: benchmark_data.tensor_hash(value) for key, value in payload["splits"].items()
        },
        "preprocessing": {"self_loops": "unit fixture"},
    }
    (cache / "manifest.json").write_text(json.dumps(protocol), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    for name, content in {
        "manifest.json": manifest,
        "metrics.json": metrics,
        "history.json": history,
    }.items():
        (run / name).write_text(json.dumps(content), encoding="utf-8")
    torch.save(checkpoint, run / "best.pt")
    before = _file_snapshot(tmp_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("A verified existing cache must not trigger downloads")

    monkeypatch.setattr(benchmark_data, "_download_official", forbidden)
    loaded, _ = benchmark_data.load_dataset("cora", tmp_path / "data", allow_download=False)
    restored = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = diag.restore_model(restored, loaded, config, saved, history, torch.device("cpu"))
    diag.evaluate_checkpoint(model, loaded, torch.device("cpu"), 2)
    diag.evaluate_checkpoint(model, loaded, torch.device("cpu"), 2, ablate=True)
    assert _file_snapshot(tmp_path) == before
