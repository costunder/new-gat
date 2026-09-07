"""Explicit CPU/synthetic debug checks; not real-data or GPU speed certification."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5 import diagnostics, timing, train
from research.conductance_gat.v5.timing import StageTimer


def test_cpu_stage_timer_reports_elapsed_calls_and_preserves_exceptions(monkeypatch):
    ticks = iter([1.0, 1.25, 2.0, 2.5, 3.0, 3.75])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    timer = StageTimer(torch.device("cpu"))
    with timer.stage("forward"):
        pass
    with timer.stage("forward"):
        pass
    with pytest.raises(RuntimeError, match="debug failure"):
        with timer.stage("backward"):
            raise RuntimeError("debug failure")
    report = timer.report(synchronize=True)
    assert report["cpu_wall_seconds"] == {"forward": 0.75, "backward": 0.75}
    assert report["stage_calls"] == {"forward": 2, "backward": 1}
    assert report["cuda_event_seconds"] == {}


def test_cuda_events_do_not_synchronize_per_stage(monkeypatch):
    events, synchronized = [], []

    class DebugEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing
            events.append(self)

        def record(self, stream):
            assert stream == "debug-stream"

        def synchronize(self):
            synchronized.append(self)

        def elapsed_time(self, other):
            assert other in events
            return 250.0

    monkeypatch.setattr(torch.cuda, "Event", DebugEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "debug-stream")
    timer = StageTimer(torch.device("cuda:0"))
    with timer.stage("sampling_and_loader_wait"):
        pass
    with timer.stage("forward"):
        pass
    with timer.stage("backward"):
        pass
    assert len(events) == 4 and synchronized == []
    report = timer.report(synchronize=True)
    assert synchronized == [events[-1]]
    assert report["cuda_event_seconds"] == {"forward": 0.25, "backward": 0.25}


def test_sample_seed_indices_are_prepared_before_graph_transfer():
    moved = []

    class DebugMask:
        def nonzero(self, *, as_tuple):
            assert not moved and as_tuple is False
            return torch.tensor([[0], [2]])

    graph = SimpleNamespace(train_mask=DebugMask())

    def transfer(device, *, non_blocking):
        moved.append(device)
        return graph

    graph.to = transfer
    sampler = SimpleNamespace(iter_epoch=lambda epoch: iter([graph]))
    args = SimpleNamespace(sample_prefetch=False, pin_memory=False)
    rows = list(train._training_batches(None, {}, sampler, 1, torch.device("cpu"), 0, args))
    assert len(rows) == 1 and rows[0][0] is graph
    assert torch.equal(rows[0][1], torch.tensor([0, 2]))
    assert len(moved) == 1


def test_training_records_stage_times_and_does_not_duplicate_validation_diagnostics(
    tmp_path, monkeypatch
):
    from test_v5_training_repair import _repair_fixture, _run

    args, payload, protocol = _repair_fixture(tmp_path / "synthetic-debug", monkeypatch)
    training_calls, evaluation_calls = [], []
    original_training, original_evaluation = train.layer_diagnostics, diagnostics.layer_diagnostics

    def training_diagnostics(*arguments, **keywords):
        training_calls.append(1)
        return original_training(*arguments, **keywords)

    def evaluation_diagnostics(*arguments, **keywords):
        evaluation_calls.append(1)
        return original_evaluation(*arguments, **keywords)

    monkeypatch.setattr(train, "layer_diagnostics", training_diagnostics)
    monkeypatch.setattr(diagnostics, "layer_diagnostics", evaluation_diagnostics)
    result = _run(args, payload, protocol)
    assert len(training_calls) == result["epochs_run"]
    assert len(evaluation_calls) == 4  # Selected-checkpoint interventions only.
    latest = json.loads((args.output_dir / "performance.json").read_text())
    saved = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    assert latest["epoch"] == saved["epoch"] == result["epochs_run"]
    assert latest["optimizer_steps"] == result["optimizer_steps"]
    assert latest["not_a_resume_checkpoint"] is True
    assert latest["checkpoint_commit_cpu_wall_seconds"] >= 0
    assert latest["stage_seconds"]["cuda_event_seconds"] == {}
    assert latest["stage_seconds"]["stage_calls"]["forward_and_loss"] == 2
    assert latest["stage_seconds"]["stage_calls"]["backward"] == 2
    assert latest["stage_seconds"]["stage_calls"]["validation"] == 1
    assert len(latest["batch_observations"]) == 2
    for row in saved["history"]:
        assert row["stage_seconds"]["stage_calls"]["diagnostics"] == 1
        assert len(row["batch_observations"]) == row["train_batches"]
