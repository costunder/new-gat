"""Synthetic CPU resume integration, never GPU or final-training evidence.

The production model, loss, backward, AdamW, RNG, checkpoint publication and
artifact validation run here. Resource/data plumbing and selection scores use
isolated debug fixtures. Exact registry pins are covered by separate tests.
"""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.conductance_gat.edge_selection import train
from research.conductance_gat.edge_selection.topology import build_topology


class DebugCheckpointBoundary(RuntimeError):
    """Python test exception; never a process signal or remote-session action."""


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            _equal(first, second)
    else:
        assert actual == expected


@pytest.fixture
def debug_training(monkeypatch):
    generator = torch.Generator().manual_seed(19)
    edges = torch.tensor([[0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6], [1, 2, 3, 2, 3, 3, 5, 7, 6, 7, 7]])
    graph = SimpleNamespace(
        x=torch.randn(9, 5, generator=generator),
        y=torch.arange(9) % 3,
        incidence_edge_index=edges,
        batch=torch.zeros(9, dtype=torch.long),
        _v5_num_graphs=1,
        edge_selection_topology=build_topology(9, edges, forest_seed=0),
    )
    provenance = [{"explicit_synthetic_cpu_resume_fixture": True}]

    class DebugInputs:
        def __init__(self, _payload, _args):
            self.data = graph
            self.indices = {"train": torch.arange(9), "validation": torch.arange(9)}
            self.sampler = None
            self.provenance = copy.deepcopy(provenance)
            self.plan_preparation_seconds = 0.0

        def _batch(self):
            return SimpleNamespace(
                graph=graph,
                selected_indices=self.indices["train"],
                origin_targets=(torch.arange(edges.shape[1]) % 2).float(),
            )

        def training_batches(self, epoch, device):
            assert device == torch.device("cpu")
            random.random()
            np.random.random()
            yield self._batch()

        def validation_batches(self, device):
            assert device == torch.device("cpu")
            yield self._batch()

        def metadata(self):
            return {"train_count": 9, "provenance": copy.deepcopy(provenance)}

    class DebugMonitor:
        def __init__(self, device):
            assert device == torch.device("cpu")

        def start(self):
            return {"explicit_synthetic_cpu_fixture": True}

        def finish(self, **_kwargs):
            return {"explicit_synthetic_cpu_fixture": True}

    state = SimpleNamespace(epoch=0, cuda_rng=torch.tensor([7, 9], dtype=torch.uint8))
    production_epoch, production_evaluate = train.run_training_epoch, train.evaluate

    def epoch(*args, **kwargs):
        state.epoch = args[5]
        return production_epoch(*args, **kwargs)

    def evaluate(*args, **kwargs):
        measured = production_evaluate(*args, **kwargs)
        measured["metric"] = 0.8 if state.epoch == 1 else 0.7
        return measured

    monkeypatch.setattr(train, "PreparedInputs", DebugInputs)
    monkeypatch.setattr(train, "RuntimeResourceMonitor", DebugMonitor)
    monkeypatch.setattr(train, "run_training_epoch", epoch)
    monkeypatch.setattr(train, "evaluate", evaluate)
    monkeypatch.setattr(train.base, "_require_cuda", lambda _device: None)
    monkeypatch.setattr(train.base, "validate_cached_graphs_once", lambda _payload: None)
    monkeypatch.setattr(train.base, "validate_hardware_runtime", lambda *_: {"debug_cpu": True})
    monkeypatch.setattr(train.base, "_v5_data_observability", lambda *_: {"debug_cpu": True})
    monkeypatch.setattr(train.base, "_versions", lambda: {"runtime": "synthetic-cpu-test"})
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda _device: state.cuda_rng.clone())
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, _device: setattr(state, "cuda_rng", value.clone()),
    )
    state.payload = {"dataset": "cora", "graphs": [{"x": graph.x}], "classes": 3}
    state.protocol = {
        "data_sha256": "a" * 64,
        "split_sha256": {"train": "b" * 64, "validation": "c" * 64},
    }
    with train._isolated_execution_state(torch.device("cpu")):
        yield state


def _arguments(directory):
    args = train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "shared_dynamic_c",
            "--selection-mode",
            "hard_concrete",
            "--corruption-ratio",
            "0.2",
            "--l0-weight",
            "0.0001",
            "--negative-loss-weight",
            "0.1",
            "--output-dir",
            str(directory),
            "--data-root",
            str(directory.parent / "debug-cache"),
            "--device",
            "cpu",
            "--hidden-channels",
            "16",
            "--layers",
            "2",
            "--heads",
            "4",
            "--edge-chunk-size",
            "3",
            "--dropout",
            "0.2",
            "--workers",
            "0",
            "--epochs",
            "4",
            "--patience",
            "10",
            "--learning-budget-policy",
            "reference_updates",
            "--no-activation-checkpoint",
        ]
    )
    train.validate_args(args)
    return args


def _run(args, debug):
    return train.train_model(
        debug.payload, debug.protocol, args, torch.device("cpu"), args.output_dir
    )


def _boundary(args, debug, monkeypatch):
    save = train.base._save

    def save_then_interrupt(path, payload):
        save(path, payload)
        if path.name == "last.pt" and payload["epoch"] == 2:
            raise DebugCheckpointBoundary("synthetic interruption after committed epoch 2")

    with monkeypatch.context() as boundary:
        boundary.setattr(train.base, "_save", save_then_interrupt)
        with pytest.raises(DebugCheckpointBoundary, match="committed epoch 2"):
            _run(args, debug)
    saved = train.base.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    assert saved["epoch"] == saved["optimizer_steps"] == 2
    return saved


@pytest.mark.parametrize("source_repair", [False, True])
def test_partial_resume_retains_identity_best_and_rng_and_finishes(
    tmp_path, monkeypatch, debug_training, source_repair
):
    debug = debug_training
    old = {"explicit_debug_source.py": "a" * 64}
    current = {"explicit_debug_source.py": "b" * 64} if source_repair else old
    monkeypatch.setattr(train, "implementation_source_hashes", lambda: copy.deepcopy(old))
    args = _arguments(tmp_path / "debug-interrupted")
    boundary = _boundary(args, debug, monkeypatch)
    before_best = (args.output_dir / "best.pt").read_bytes()
    before_configuration = (args.output_dir / "configuration.json").read_bytes()
    boundary_hash = train.base.sha256_file(args.output_dir / "last.pt")
    before_history = copy.deepcopy(boundary["history"])
    captured, resumed_epochs, proof_calls = {}, [], []
    make_optimizer, run_epoch = train.make_optimizer, train.run_training_epoch

    def exact_debug_proof(previous, following, *, scope):
        assert previous == old and following == current and scope == "training"
        proof_calls.append(True)
        return {"patch_id": "explicit-debug-source-proof", "scope": scope}

    if source_repair:
        monkeypatch.setattr(train, "require_source_compatibility", exact_debug_proof)
    monkeypatch.setattr(train, "implementation_source_hashes", lambda: copy.deepcopy(current))

    def optimizer(model):
        result = make_optimizer(model)
        captured.update(model=model, optimizer=result)
        return result

    def inspect_resume(*values, **kwargs):
        if not resumed_epochs:
            assert values[5] == 3
            _equal(captured["model"].state_dict(), boundary["model_state"])
            _equal(captured["optimizer"].state_dict(), boundary["optimizer_state"])
            _equal(
                train._checkpoint_rng(torch.device("cpu")),
                {
                    key: boundary[key]
                    for key in (
                        "python_rng_state",
                        "numpy_rng_state",
                        "cpu_rng_state",
                        "cuda_rng_state",
                    )
                },
            )
        resumed_epochs.append(values[5])
        return run_epoch(*values, **kwargs)

    with monkeypatch.context() as resumed:
        resumed.setattr(train, "make_optimizer", optimizer)
        resumed.setattr(train, "run_training_epoch", inspect_resume)
        args.resume = True
        result = _run(args, debug)
    final = train.base.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    assert resumed_epochs == [3, 4]
    assert final["epoch"] == final["optimizer_steps"] == 4
    assert final["history"][:2] == before_history
    assert final["resume_identity"] == boundary["resume_identity"]
    assert final["resume_identity_sha256"] == boundary["resume_identity_sha256"]
    assert result["resume_identity"] == boundary["resume_identity"]
    assert (args.output_dir / "best.pt").read_bytes() == before_best
    assert (args.output_dir / "configuration.json").read_bytes() == before_configuration
    assert not (args.output_dir / "best.previous.pt").exists()
    assert train.inspect_completed(args.output_dir)["best_epoch"] == 1
    if source_repair:
        assert proof_calls
        assert final["execution_source_sha256"] == result["execution_source_sha256"] == current
        assert final["source_transitions"] == result["source_transitions"]
        assert final["source_transitions"][-1]["after_epoch"] == 2
        assert final["source_transitions"][-1]["optimizer_steps"] == 2
        assert final["source_transitions"][-1]["restored_checkpoint_sha256"] == boundary_hash
        assert final["source_transitions"][-1]["source_sha256"] == current
        assert final["source_transitions"][-1]["source_compatibility"]["patch_id"] == (
            "explicit-debug-source-proof"
        )

    control_args = _arguments(tmp_path / "debug-uninterrupted-control")
    _run(control_args, debug)
    control = train.base.load_checkpoint_on_cpu(control_args.output_dir / "last.pt")
    for key in (
        "model_state",
        "optimizer_state",
        "python_rng_state",
        "numpy_rng_state",
        "cpu_rng_state",
        "cuda_rng_state",
    ):
        _equal(final[key], control[key])


@pytest.mark.parametrize(
    "field",
    [
        "configuration",
        "training_arguments",
        "dataset_protocol",
        "learning_budget",
        "runtime_versions",
        "initial_state_sha256",
        "input_provenance",
    ],
)
def test_source_repair_never_waives_other_identity_changes(field, monkeypatch):
    original = {
        "source_sha256": {"explicit_debug_source.py": "a" * 64},
        field: {"retained": "debug-contract"},
    }
    saved = {
        "resume_identity": original,
        "resume_identity_sha256": train.base._canonical_sha256(original),
    }
    expected = copy.deepcopy(original)
    expected["source_sha256"] = {"explicit_debug_source.py": "b" * 64}
    expected[field] = {"changed": "not-authorized-by-source-repair"}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("recipe/runtime mismatch must fail before source authorization")

    monkeypatch.setattr(train, "require_source_compatibility", forbidden)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train.resolve_training_resume(saved, expected)
    assert saved["resume_identity"] == original


def test_rejected_resume_keeps_checkpoint_history_and_configuration_bytes(
    tmp_path, monkeypatch, debug_training
):
    args = _arguments(tmp_path / "debug-rejected-resume")
    _boundary(args, debug_training, monkeypatch)
    protected = {
        path.name: path.read_bytes()
        for path in args.output_dir.iterdir()
        if path.name in {"best.pt", "last.pt", "history.json", "configuration.json"}
    }
    args.resume, args.dropout = True, 0.1

    def forbidden_update(*_args, **_kwargs):
        raise AssertionError("a rejected resume must never update optimizer/model state")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(args, debug_training)
    assert {name: (args.output_dir / name).read_bytes() for name in protected} == protected
