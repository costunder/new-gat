"""Tiny unit fixtures only: public-data training still requires a CUDA GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from chartgat.cache import atomic_write_json
from research.conductance_gat.ablation import train as shared
from research.conductance_gat.ablation.model import is_gate_parameter
from research.conductance_gat.v2 import train
from research.conductance_gat.v2.model import DirectCNodeClassifier
from research.conductance_gat.v2.protocol import CONDITIONS, PARAMETERIZATION, SUITE
from scripts import run_conductance_v2 as runner


def args_for(tmp_path, condition="direct_c"):
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
            "--edge-chunk-size",
            "2",
            "--epochs",
            "2",
            "--patience",
            "2",
        ]
    )


def graph_fixture():
    graph = SimpleNamespace(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        y=torch.tensor([0, 1, 0, 999999]),  # test sentinel: invalid if read by the loss
        incidence_edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]),
    )

    class NoTest(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("Test indices are not part of this experiment")
            return super().__getitem__(key)

    indices = NoTest(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    payload = {"dataset": "cora", "graphs": [vars(graph)], "classes": 2}
    return graph, indices, payload


def mock_unit_hardware(monkeypatch):
    # Hardware is mocked only in bounded 4-node fixtures. The separate guard
    # test below checks that neither public entry point permits CPU training.
    monkeypatch.setattr(shared, "_require_cuda", lambda device: None)
    monkeypatch.setattr(shared, "_configure_fp32", lambda: None)
    for name in ("reset_peak_memory_stats", "synchronize", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "unit_fixture_mocked_cuda")


def test_only_transductive_one_seed_protocol(tmp_path):
    args = args_for(tmp_path)
    assert args.model_seed == 0 and args.batch_size == 1 and args.workers == 0
    assert args.device == "cuda"
    assert train.configuration(args)["edge_chunk_size"] == 2
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(
            ["--dataset", "ppi", "--condition", "direct_c", "--output-dir", str(tmp_path)]
        )
    with pytest.raises(ValueError, match="PPI"):
        train.topology_metadata({"dataset": "ppi", "graphs": []})


def test_public_training_and_cli_reject_cpu_before_loading_or_writing(tmp_path):
    args = args_for(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.train_model({}, {}, args, torch.device("cpu"), args.output_dir)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "direct_c",
                "--device",
                "cpu",
                "--output-dir",
                str(args.output_dir),
            ]
        )
    assert not args.output_dir.exists()


@pytest.mark.parametrize("key,value", [("edge_chunk_size", 0), ("batch_size", 2), ("workers", 1)])
def test_invalid_execution_settings_not_silently_ignored(tmp_path, key, value):
    args = args_for(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError):
        train._validate_args(args)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_optimizer_only_updates_active_parameters_with_explicit_no_c_decay(condition):
    graph, _, _ = graph_fixture()
    model = DirectCNodeClassifier(
        3,
        2,
        incidence=graph.incidence_edge_index,
        num_nodes=4,
        gate_mode=CONDITIONS[condition]["gate_mode"],
    )
    optimizer = train.make_optimizer(model, condition)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["non_gate"]["weight_decay"] == 0.0005
    assert ("gate" in groups) == (condition == "direct_c")
    if "gate" in groups:
        assert groups["gate"]["weight_decay"] == 0
    included = {id(p) for group in groups.values() for p in group["params"]}
    assert included == {id(p) for p in model.parameters() if p.requires_grad}


def test_offline_failure_record_and_existing_output_protected(monkeypatch, tmp_path):
    mock_unit_hardware(monkeypatch)
    monkeypatch.setattr(train, "_source_hashes", lambda: {"fixture": "a" * 64})
    calls = []

    def missing(dataset, data_root, *, allow_download):
        calls.append(allow_download)
        raise FileNotFoundError("Verified official cache missing")

    monkeypatch.setattr(train, "load_dataset", missing)
    options = [
        "--dataset",
        "cora",
        "--condition",
        "direct_c",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    with pytest.raises(FileNotFoundError, match="cache missing"):
        train.main(options)
    path = tmp_path / "out/metrics.json"
    contents = path.read_bytes()
    saved = json.loads(contents)
    assert calls == [False] and saved["status"] == "failed"
    assert saved["research_suite"] == SUITE and saved["test_evaluated"] is False
    with pytest.raises(FileExistsError):
        train.main(options)
    assert path.read_bytes() == contents


def test_train_loop_to_runner_report_preserves_topology_sources_and_one_seed(monkeypatch, tmp_path):
    mock_unit_hardware(monkeypatch)
    graph, indices, payload = graph_fixture()
    monkeypatch.setattr(shared, "_make_data", lambda *args: (graph, indices))
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"unit_fixture_only": True})
    protocol = {"data_sha256": "a" * 64, "unit_fixture_only": True}
    trained, preflights = {}, []

    def dispatch(command, log, environment):
        if any(str(part).endswith("gpu_preflight.py") for part in command):
            preflights.append(command)
            return 0
        index = command.index("research.conductance_gat.v2.train")
        args = train.build_parser().parse_args(command[index + 1 :])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        result = train.train_model(payload, protocol, args, torch.device("cpu"), args.output_dir)
        result.update(unit_fixture_only=True, hardware_mocked=True)
        atomic_write_json(args.output_dir / "metrics.json", result)
        trained[args.condition] = result
        return 0

    monkeypatch.setattr(runner, "run_logged", dispatch)
    status = runner.main(
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
            "direct-c-fixture",
        ]
    )
    assert status == 0 and len(preflights) == 1 and list(trained) == list(CONDITIONS)
    run = tmp_path / "results/conductance_gat/v2/direct-c-fixture"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    assert manifest["status"] == report["status"] == "passed"
    assert trained["direct_c"]["shared_backbone_initial_state_sha256"] == trained["fixed_c"][
        "shared_backbone_initial_state_sha256"
    ]
    assert trained["direct_c"]["total_parameters"] == trained["fixed_c"]["total_parameters"] + 8
    assert trained["fixed_c"]["frozen_parameters"] == 0
    assert trained["direct_c"]["frozen_parameters"] == 0
    for condition, result in trained.items():
        assert result["test_evaluated"] is False and result["metric_name"] == "accuracy"
        assert result["optimizer_steps"] == 2 and result["model_seed"] == 0
        coverage = result["diagnostics"]["edge_gradient_coverage"]
        assert [row["optimizer_step"] for row in coverage] == [1, 2]
        for row in coverage:
            assert row["scope"] == "full_graph_train_mask"
            for layer in row["layers"]:
                assert layer["graph_edges"] == 4
                assert layer["edge_parameters"] == (4 if condition == "direct_c" else 0)
                assert layer["trainable"] == (condition == "direct_c")
                if condition == "direct_c":
                    assert 0 <= layer["nonzero_fraction"] <= 1
                if condition == "fixed_c":
                    assert layer["nonzero_task_gradient_edges"] == 0
                    assert layer["gradient_present"] is False
        saved = torch.load(result["checkpoint"], weights_only=True)
        assert saved["model"] == saved["research_suite"] == SUITE
        assert saved["topology"] == result["topology"] == train.topology_metadata(payload)
        assert saved["parameterization"] == result["parameterization"] == PARAMETERIZATION
        assert saved["source_sha256"] == result["source_sha256"] == manifest["sources"]["sha256"]
        assert saved["configuration"] == result["configuration"]
        assert saved["architecture"]["edge_chunk_size"] == 2
        rebuilt = DirectCNodeClassifier(
            3, 2, incidence=graph.incidence_edge_index, num_nodes=4, **saved["architecture"]
        )
        rebuilt.load_state_dict(saved["state_dict"], strict=True)
        rebuilt.eval()
        with torch.no_grad():
            selected = rebuilt(graph).index_select(0, indices["validation"])
        actual = float((selected.argmax(-1) == graph.y[indices["validation"]]).float().mean())
        assert actual == result["validation"]
        alphas = [p for name, p in rebuilt.named_parameters() if is_gate_parameter(name)]
        if condition == "fixed_c":
            assert not alphas
        else:
            assert any(torch.count_nonzero(p) > 0 for p in alphas)
            assert any(
                record["parameter_groups"]["operators.0"]["task_gradient_norm"] > 0
                for record in result["diagnostics"]["train_trajectory"]
            )


def test_edge_coverage_does_not_change_gradients_and_rejects_nonfinite():
    graph, _, _ = graph_fixture()
    model = DirectCNodeClassifier(3, 2, incidence=graph.incidence_edge_index, num_nodes=4)
    for operator in model.operators:
        operator.estimator.log_c.grad = torch.tensor([0.0, 2.0, 0.0, -1.0])
    before = [operator.estimator.log_c.grad.clone() for operator in model.operators]
    records = train.edge_gradient_coverage(model)
    assert all(row["nonzero_task_gradient_edges"] == 2 for row in records)
    assert all(row["nonzero_fraction"] == 0.5 for row in records)
    for operator, expected in zip(model.operators, before, strict=True):
        torch.testing.assert_close(operator.estimator.log_c.grad, expected)
    model.operators[0].estimator.log_c.grad[0] = float("nan")
    with pytest.raises(FloatingPointError, match="gradient"):
        train.edge_gradient_coverage(model)


def test_source_change_rejects_new_training_result(monkeypatch, tmp_path):
    mock_unit_hardware(monkeypatch)
    _, _, payload = graph_fixture()
    calls = 0

    def sources():
        nonlocal calls
        calls += 1
        return {"fixture": ("a" if calls == 1 else "b") * 64}

    monkeypatch.setattr(train, "_source_hashes", sources)
    monkeypatch.setattr(shared, "train_model", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="source changed"):
        train.train_model(
            payload, {"data_sha256": "f" * 64}, args_for(tmp_path), torch.device("cpu"), tmp_path
        )
