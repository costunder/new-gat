from __future__ import annotations

import ast
import inspect
import json

import numpy as np
import pytest
import torch

from research.cycle_pe import benchmark, paper, paper_train, resource_monitor
from research.cycle_pe.paper_data import PaperGraph
from research.cycle_pe.paper_model import PaperCycleModel, prepare_graph
from research.cycle_pe.resource_monitor import (
    FailureSafeResourceMonitor,
    persist_failure_artifacts,
    resource_failure_boundary,
    resource_failure_observations,
)
from research.cycle_pe.v2 import benchmark as v2_benchmark


def _prepared_triangle(split: str):
    edges = ((0, 1), (0, 2), (1, 2))
    return prepare_graph(
        PaperGraph(
            graph_id=f"{split}:triangle",
            split=split,
            family="unit_test_fixture",
            num_nodes=3,
            edges=edges,
            edge_targets=np.zeros((len(edges), 1), dtype=np.float64),
        ),
        required_variants=("no_pe",),
    )


def test_train_supervised_closes_sampler_and_preserves_original_error(
    monkeypatch,
) -> None:
    sessions: list[FailureSafeResourceMonitor] = []
    original_factory = paper_train.FailureSafeResourceMonitor

    def tracking_factory(device, *, workload):
        session = original_factory(device, workload=workload)
        sessions.append(session)
        return session

    monkeypatch.setattr(paper_train, "FailureSafeResourceMonitor", tracking_factory)
    original = RuntimeError("original forward failure")
    model = PaperCycleModel(
        variant="no_pe",
        raw_width=1,
        node_input_dim=4,
        edge_input_dim=4,
        edge_output_dim=1,
        node_output_dim=0,
        graph_output_dim=0,
        hidden_dim=8,
        pe_dim=4,
        layers=1,
        embedding_dim=0,
    )

    def fail_forward(graphs):
        raise original

    monkeypatch.setattr(model, "forward", fail_forward)
    settings = paper_train.TrainSettings(
        device=torch.device("cpu"),
        seed=0,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        workers=0,
        amp_requested=False,
        pin_memory_requested=False,
        non_blocking_requested=False,
    )
    with pytest.raises(RuntimeError) as caught:
        paper_train.train_supervised(
            model,
            [_prepared_triangle("train")],
            [_prepared_triangle("validation")],
            settings,
            target_levels=("edge",),
        )

    assert caught.value is original
    assert len(sessions) == 1 and sessions[0].finished is True
    sampler = sessions[0]._monitor._thread
    assert sampler is not None and sampler.is_alive() is False
    failures = resource_failure_observations(original)
    assert len(failures) == 1
    assert failures[0]["workload"] == "cycle_paper_supervised_training"
    assert failures[0]["resource_observability"] is not None
    assert failures[0]["monitor_cleanup_error"] is None


def test_monitor_cleanup_failure_never_replaces_original_and_runs_once(
    monkeypatch,
) -> None:
    class BrokenMonitor:
        def __init__(self, device):
            self.finish_calls = 0

        def start(self):
            return {"fixture": True}

        def finish(self, *, peak_allocated_bytes, peak_reserved_bytes):
            self.finish_calls += 1
            raise RuntimeError("cleanup failed")

    broken = BrokenMonitor(torch.device("cpu"))
    monkeypatch.setattr(
        "research.cycle_pe.resource_monitor.RuntimeResourceMonitor",
        lambda device: broken,
    )
    original = KeyboardInterrupt("original workload interruption")

    @resource_failure_boundary
    def fail():
        monitor = FailureSafeResourceMonitor(
            torch.device("cpu"), workload="cleanup_failure_fixture"
        )
        monitor.start()
        raise original

    with pytest.raises(KeyboardInterrupt) as caught:
        fail()

    assert caught.value is original
    assert broken.finish_calls == 1
    failure = resource_failure_observations(original)[0]
    assert failure["monitor_cleanup_error"] == {
        "type": "RuntimeError",
        "message": "cleanup failed",
    }
    assert any("cleanup failed" in note for note in original.__notes__)


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(RuntimeError("fixture suite failure"), id="runtime-error"),
        pytest.param(
            torch.cuda.OutOfMemoryError("fixture cuda out of memory"),
            id="cuda-out-of-memory",
        ),
        pytest.param(KeyboardInterrupt("fixture interruption"), id="keyboard-interrupt"),
    ],
)
def test_paper_failure_manifest_retains_resource_observation(
    monkeypatch, tmp_path, original
) -> None:

    @resource_failure_boundary
    def failing_core(args, device):
        monitor = FailureSafeResourceMonitor(
            device, workload="paper_failure_manifest_fixture"
        )
        monitor.start()
        raise original

    monkeypatch.setattr(paper, "run_core", failing_core)
    output = tmp_path / "run"
    with pytest.raises(type(original)) as caught:
        paper.main(
            [
                "--suite",
                "core",
                "--device",
                "cpu",
                "--prepare-only",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output),
            ]
        )
    assert caught.value is original
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    failures = manifest["resource_failure_observations"]
    assert len(failures) == 1
    assert failures[0]["workload"] == "paper_failure_manifest_fixture"
    assert failures[0]["resource_observability"] is not None
    assert failures[0]["original_error"] == {
        "type": type(original).__name__,
        "message": str(original),
    }


@pytest.mark.parametrize(
    "function",
    [
        paper_train.train_supervised,
        paper._run_supervised_bundle,
        paper.run_brec,
        benchmark._train_model,
        benchmark._evaluate_test_checkpoint,
        v2_benchmark._train_model,
        v2_benchmark._evaluate_test_checkpoint,
    ],
)
def test_every_cycle_monitor_owner_has_a_failure_boundary(function) -> None:
    assert hasattr(function, "__wrapped__")
    source = inspect.getsource(function)
    assert "@resource_failure_boundary" in source


def test_resource_monitor_does_not_silently_swallow_exceptions() -> None:
    tree = ast.parse(inspect.getsource(resource_monitor))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    assert handlers
    assert all(handler.type is not None for handler in handlers)
    assert all(
        not (len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass))
        for handler in handlers
    )


def test_failure_artifact_write_error_is_not_allowed_to_replace_primary() -> None:
    primary = torch.cuda.OutOfMemoryError("primary OOM")
    attempted: list[str] = []

    def broken_writer() -> None:
        attempted.append("broken")
        raise OSError("artifact filesystem unavailable")

    def healthy_writer() -> None:
        attempted.append("healthy")

    persist_failure_artifacts(
        primary,
        (("broken.json", broken_writer), ("healthy.json", healthy_writer)),
    )

    assert attempted == ["broken", "healthy"]
    assert any(
        "failure artifact persistence failed (broken.json)" in note
        for note in primary.__notes__
    )
