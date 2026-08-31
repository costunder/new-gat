"""Bounded training-loop fixtures with mocked CUDA; no public-data CPU training."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.ablation import train as shared
from research.conductance_gat.ablation.model import is_gate_parameter, state_sha256
from research.conductance_gat.c_learning import train
from research.conductance_gat.c_learning.model import CLearningNodeClassifier
from research.conductance_gat.c_learning.protocol import CONDITIONS, SUITE


def args_for(tmp_path, condition="learned_c"):
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
        ]
    )


def test_new_protocol_preserves_one_seed_and_locked_common_configuration(tmp_path):
    args = args_for(tmp_path)
    assert args.model_seed == 0 and args.device == "cuda"
    assert args.epochs == 200 and args.patience == 50
    assert train.configuration(args) == shared.configuration(args)
    assert set(CONDITIONS) == {"learned_c", "fixed_c"}
    assert all(spec["normalization"] == "node_degree" for spec in CONDITIONS.values())
    with pytest.raises(SystemExit):
        args_for(tmp_path, "baseline")


def test_direct_and_cli_require_gpu_before_outputs(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.train_model({}, {}, args_for(tmp_path), torch.device("cpu"), tmp_path / "output")
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "fixed_c",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_checkpoint_is_separate_suite_and_discloses_gate_mode_parameter_counts(tmp_path, condition):
    args = args_for(tmp_path, condition)
    model = CLearningNodeClassifier(3, 2, gate_mode=CONDITIONS[condition]["gate_mode"])
    saved = shared.checkpoint_payload(
        model,
        args,
        {"data_sha256": "a" * 64},
        state_sha256(model),
        3,
        0.5,
        definition=train.DEFINITION,
    )
    assert saved["research_suite"] == saved["model"] == SUITE
    assert saved["gate_mode"] == saved["architecture"]["gate_mode"] == model.gate_mode
    assert saved["architecture"]["normalization"] == "node_degree"
    assert saved["gate_weight_decay"] == CONDITIONS[condition]["gate_weight_decay"]
    assert saved["total_parameters"] == saved["trainable_parameters"] + saved["frozen_parameters"]
    assert (saved["frozen_parameters"] > 0) == (condition == "fixed_c")
    assert saved["evaluation_split"] == "validation" and saved["test_evaluated"] is False


def test_cli_offline_cache_failure_saves_new_suite_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(shared, "_require_cuda", lambda device: None)
    calls = []

    def missing(dataset, data_root, *, allow_download):
        calls.append(allow_download)
        raise FileNotFoundError("No verified cache")

    monkeypatch.setattr(shared, "load_dataset", missing)
    with pytest.raises(FileNotFoundError, match="verified cache"):
        train.main(
            [
                "--dataset",
                "cora",
                "--condition",
                "fixed_c",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    saved = json.loads((tmp_path / "out/metrics.json").read_text())
    assert calls == [False]
    assert saved["status"] == "failed" and saved["research_suite"] == SUITE
    assert saved["gate_mode"] == "fixed_one" and saved["test_evaluated"] is False


def test_both_fixture_arms_same_initial_hash_fixed_scaffold_unchanged_and_no_test(
    monkeypatch, tmp_path
):
    # Only this unit fixture mocks hardware and data access. The public train
    # entrypoints still reject CPU before loading data (separately tested above).
    monkeypatch.setattr(shared, "_require_cuda", lambda device: None)
    monkeypatch.setattr(shared, "_configure_fp32", lambda: None)
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "unit_fixture_only")
    graph = SimpleNamespace(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
    )

    class NoTest(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("Do not read test indices")
            return super().__getitem__(key)

    splits = NoTest(train=torch.tensor([0, 1]), validation=torch.tensor([2]))
    monkeypatch.setattr(shared, "_make_data", lambda *args: (graph, splits))
    payload = {"dataset": "cora", "graphs": [vars(graph)], "classes": 2}
    results = []
    for condition in CONDITIONS:
        args = args_for(tmp_path, condition)
        args.epochs = 2
        args.patience = 2
        args.output_dir.mkdir()
        result = train.train_model(
            payload, {"data_sha256": "f" * 64}, args, torch.device("cpu"), args.output_dir
        )
        results.append(result)
        assert result["status"] == "passed" and result["research_suite"] == SUITE
        assert result["test_evaluated"] is False and "test" not in result
        assert result["epochs_run"] == result["optimizer_steps"] == 2
        saved = torch.load(result["checkpoint"], weights_only=True)
        assert saved["model"] == SUITE and saved["condition"] == condition
        assert saved["trainable_parameters"] == result["trainable_parameters"]
        if condition == "fixed_c":
            torch.manual_seed(args.model_seed)
            initial = CLearningNodeClassifier(3, 2, gate_mode="fixed_one")
            for name, parameter in initial.named_parameters():
                if is_gate_parameter(name):
                    torch.testing.assert_close(saved["state_dict"][name], parameter, rtol=0, atol=0)
            for record in result["diagnostics"]["train_trajectory"]:
                assert record["parameter_groups"]["operators.0"]["optimizer_included"] is False
                assert all(layer["conductance"]["cv"] == 0 for layer in record["layers"])
    assert results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"]
    assert results[0]["total_parameters"] == results[1]["total_parameters"]
    assert results[0]["trainable_parameters"] > results[1]["trainable_parameters"]
