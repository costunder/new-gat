"""Unit tests of allocation selection with explicitly simulated CUDA resources.

OOM and VRAM numbers below are injected unit-test conditions, never experiment
measurements. Successful candidate kernels run the real fixed core on a small
CPU DEBUG graph, with every requested feature column intact. No scientific
report, saved result, classifier, optimizer or final-profile audit is produced.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.local_energy_relations import study
from research.local_energy_relations.contract import read_config
from research.local_energy_relations.topology import build_topology


class DebugTopologyAllocation:
    """Route the simulated CUDA allocation to real CPU DEBUG operator indices."""

    def __init__(self, cpu_top):
        self.cpu_top = cpu_top
        self.num_pairs = cpu_top.num_pairs

    def to(self, device):
        assert torch.device(device).type in ("cpu", "cuda")
        return self.cpu_top.to("cpu")


def fixture_inputs():
    pairs = np.asarray([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)], dtype=np.int64).T
    top = DebugTopologyAllocation(build_topology(6, pairs))
    generator = torch.Generator().manual_seed(20261004)
    features = torch.randn((6, 32), generator=generator, dtype=torch.float64)
    case = SimpleNamespace(graph_id="DEBUG-resource-fixture", num_features=32, x=features)
    config = read_config(profile="debug")
    config["runtime"]["relation_batch"] = 3
    config["runtime"]["channel_chunk"] = "auto"
    return [case], [top], config


def simulate_cuda_resources(monkeypatch, *, peak_for_channels=None, estimated=100):
    """Inject only allocation/resource behavior; retain the real CPU algebra."""
    events, computed_channels = [], []
    original_to = torch.Tensor.to
    real_compute = study.compute_chunk

    def redirected_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            args = (torch.device("cpu"), *args[1:])
        return original_to(tensor, *args, **kwargs)

    def free_memory(device):
        assert device.type == "cuda"
        events.append("free_query")
        return 1000, 2000

    def memory(device):
        assert device.type == "cuda"
        channels = computed_channels[-1] if computed_channels else None
        peak = 600 if peak_for_channels is None else peak_for_channels(channels)
        return {"peak_vram_bytes": peak, "steady_vram_bytes": 0, "free_vram_bytes": 1000}

    def compute(top, x, *args, **kwargs):
        computed_channels.append(x.shape[1])
        return real_compute(top, x, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", redirected_to)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", free_memory)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset_peak")
    )
    monkeypatch.setattr(study, "synchronize", lambda device: events.append("synchronize"))
    monkeypatch.setattr(study, "_estimated_bytes", lambda top, channels, relation_batch: estimated)
    monkeypatch.setattr(study, "_memory", memory)
    monkeypatch.setattr(study, "compute_chunk", compute)
    return events, computed_channels, real_compute


def test_oom_candidate_is_recorded_and_next_exact_channel_chunk_is_selected(monkeypatch):
    cases, tops, config = fixture_inputs()
    unchanged = cases[0].x.clone()
    events, computed_channels, real_compute = simulate_cuda_resources(monkeypatch)

    def first_oom(top, x, *args, **kwargs):
        computed_channels.append(x.shape[1])
        if x.shape[1] == 16:
            raise torch.cuda.OutOfMemoryError("injected CUDA resource-path OOM")
        return real_compute(top, x, *args, **kwargs)

    monkeypatch.setattr(study, "compute_chunk", first_oom)
    rows = []
    selected = study.calibrate(cases, tops, config, torch.device("cuda"), rows, "channels")
    assert selected == (32, 3)
    assert computed_channels == [16, 32]
    assert [row["status"] for row in rows] == ["CUDA_OOM", "measured"]
    assert rows[0]["feature_cells_per_second"] is None
    assert "injected" in rows[0]["error"]
    assert rows[1]["channel_chunk"] == cases[0].num_features == 32
    assert rows[1]["physical_graph_batch"] == 1
    assert rows[1]["feature_cells_per_second"] > 0
    assert events.count("empty_cache") == 4
    torch.testing.assert_close(cases[0].x, unchanged, atol=0, rtol=0)


def test_measured_peak_above_safety_budget_cannot_win_selection(monkeypatch):
    cases, tops, config = fixture_inputs()
    _, computed_channels, _ = simulate_cuda_resources(
        monkeypatch, peak_for_channels=lambda channels: 800 if channels == 16 else 600
    )
    rows = []
    selected = study.calibrate(cases, tops, config, torch.device("cuda"), rows, "channels")
    assert selected == (32, 3)
    assert computed_channels == [16, 32]
    assert [row["status"] for row in rows] == ["rejected_measured_memory", "measured"]
    assert rows[0]["estimated_working_bytes"] < 1000 * 0.75
    assert rows[0]["peak_vram_bytes"] > 1000 * 0.75
    assert rows[1]["peak_vram_bytes"] <= 1000 * 0.75


def test_inactive_allocator_cache_is_released_before_each_free_vram_query(monkeypatch):
    cases, tops, config = fixture_inputs()
    events, _, _ = simulate_cuda_resources(monkeypatch)
    config["runtime"]["channel_chunk"] = 16
    rows = []
    study.calibrate(cases, tops, config, torch.device("cuda"), rows, "channels")
    assert events[0:2] == ["empty_cache", "free_query"]
    assert events[-1] == "empty_cache"
    assert events.count("free_query") == 1
    assert len(rows) == 1 and rows[0]["status"] == "measured"


@pytest.mark.parametrize(
    "error",
    [
        ValueError("injected invalid core contract"),
        RuntimeError("injected core numerical failure"),
        torch.cuda.OutOfMemoryError("injected CPU allocation error"),
    ],
)
def test_cpu_core_errors_propagate_without_being_marked_cuda_candidate_failure(monkeypatch, error):
    cases, tops, config = fixture_inputs()
    config["runtime"]["channel_chunk"] = 16
    monkeypatch.setattr(study, "_estimated_bytes", lambda *args: 1)
    calls = []

    def fail(*args, **kwargs):
        calls.append("compute")
        raise error

    monkeypatch.setattr(study, "compute_chunk", fail)
    rows = []
    with pytest.raises(type(error)) as caught:
        study.calibrate(cases, tops, config, torch.device("cpu"), rows, "channels")
    assert caught.value is error
    assert calls == ["compute"]
    assert rows == []


def test_cuda_nonallocation_numerical_error_is_never_swallowed(monkeypatch):
    cases, tops, config = fixture_inputs()
    simulate_cuda_resources(monkeypatch)
    error = RuntimeError("injected true-residual failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(study, "compute_chunk", fail)
    rows = []
    with pytest.raises(RuntimeError) as caught:
        study.calibrate(cases, tops, config, torch.device("cuda"), rows, "channels")
    assert caught.value is error
    assert rows == []


@pytest.mark.parametrize("reason", ["OOM", "estimate", "measured_peak"])
def test_no_measured_allocation_reports_explicit_failure_preserving_inputs(monkeypatch, reason):
    cases, tops, config = fixture_inputs()
    unchanged = cases[0].x.clone()
    _, computed_channels, _ = simulate_cuda_resources(
        monkeypatch,
        estimated=900 if reason == "estimate" else 100,
        peak_for_channels=lambda _: 800 if reason == "measured_peak" else 600,
    )
    if reason == "OOM":

        def fail(top, x, *args, **kwargs):
            computed_channels.append(x.shape[1])
            raise torch.cuda.OutOfMemoryError("injected all-candidate OOM")

        monkeypatch.setattr(study, "compute_chunk", fail)
    rows = []
    with pytest.raises(
        RuntimeError, match="no measured channels allocation fits.*inputs preserved"
    ):
        study.calibrate(cases, tops, config, torch.device("cuda"), rows, "channels")
    expected_status = {
        "OOM": "CUDA_OOM",
        "estimate": "skipped_memory_estimate",
        "measured_peak": "rejected_measured_memory",
    }[reason]
    assert len(rows) == 2 and {row["status"] for row in rows} == {expected_status}
    assert computed_channels == ([] if reason == "estimate" else [16, 32])
    torch.testing.assert_close(cases[0].x, unchanged, atol=0, rtol=0)
