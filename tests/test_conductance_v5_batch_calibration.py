"""Explicit CPU synthetic calibration tests, not GPU throughput/performance evidence."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.conductance_gat.v5 import batch_calibration as calibration
from research.conductance_gat.v5 import train


def _args(dataset="ogbn-arxiv", sampling="neighbor"):
    result = train.build_parser().parse_args([
        "--dataset", dataset, "--condition", "shared_dynamic_c", "--output-dir", "unused",
        "--sampling", sampling, "--sample-seed-batch-size", "32", "--hidden-channels", "32",
        "--layers", "2", "--heads", "4", "--ffn-multiplier", "2", "--epochs", "4",
        "--no-activation-checkpoint",
    ])
    train.validate_args(result)
    return result


def _graph():
    return SimpleNamespace(
        x=torch.randn(9, 6), y=torch.arange(9) % 3,
        incidence_edge_index=torch.tensor(
            [[0, 0, 1, 2, 2, 3, 4, 5, 6, 7], [1, 2, 2, 3, 4, 4, 5, 6, 7, 8]]
        ),
    )


def test_sampled_physical_batch_report_is_supervised_seeds_not_one_graph():
    args = _args()
    sampler_type = type("Sampler", (), {
        "__len__": lambda self: 7, "metadata": lambda self: {"mode": "neighbor"}
    })
    report = train._v5_batch_observability(
        _graph(), {"train": torch.arange(9)}, sampler_type(), args
    )
    assert report["configured_physical_batch_size"] == 32
    assert report["effective_batch_size"] == 32
    assert report["batch_unit"] == "supervised_seed_nodes"
    assert report["training_batches_per_epoch"]["value"] == 7


def test_candidate_grows_only_requested_physical_axis_and_does_not_mutate_args():
    args = _args()
    original = vars(args).copy()
    candidate = calibration._candidate_args(args, 64, 0)
    assert candidate.sample_seed_batch_size == 64
    assert candidate.batch_size == 1
    assert candidate.num_neighbors == args.num_neighbors
    assert candidate.hidden_channels == args.hidden_channels
    assert candidate.layers == args.layers
    assert vars(args) == original
    with pytest.raises(ValueError, match="cannot reduce"):
        calibration._candidate_args(args, 16, 0)
    with pytest.raises(ValueError, match="not DataLoader"):
        calibration._candidate_args(args, 64, 4)


def test_fullgraph_must_not_fake_larger_batch_through_replication():
    args = _args("cora", "full")
    assert calibration._candidate_args(args, 1, 0).batch_size == 1
    with pytest.raises(ValueError, match="cannot be replicated"):
        calibration._candidate_args(args, 2, 0)


def test_ppi_larger_measured_batch_is_allowed_but_current_floor_is_retained():
    args = _args("ppi", "full")
    args.batch_size = 8
    candidate = calibration._candidate_args(args, 16, 4)
    train.validate_args(candidate)
    assert candidate.batch_size == 16
    assert candidate.sample_seed_batch_size == args.sample_seed_batch_size
    with pytest.raises(ValueError, match="cannot reduce"):
        calibration._candidate_args(args, 4, 4)
    candidate.batch_size = 1
    with pytest.raises(ValueError, match="V1 minimum"):
        train.validate_args(candidate)


def test_isolation_restores_python_numpy_torch_rng_and_precision_after_error():
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    python_state, numpy_state, torch_state = (
        random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    )
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    with pytest.raises(RuntimeError, match="synthetic"):
        with calibration._isolated_execution_state(torch.device("cpu")):
            random.random()
            np.random.random()
            torch.rand(2)
            torch.set_float32_matmul_precision("high")
            raise RuntimeError("synthetic candidate failure")
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    assert np.array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_float32_matmul_precision() == precision
    assert torch.backends.cuda.matmul.allow_tf32 == tf32


def _cpu_fixture(monkeypatch):
    args = _args()
    graph = _graph()
    indices = {"train": torch.arange(6), "validation": torch.arange(6, 9)}
    payload = {"dataset": args.dataset, "graphs": [vars(graph)], "classes": 3}
    sampler = SimpleNamespace(metadata=lambda: {"mode": "neighbor", "seed_batch_size": 64})
    calls = []

    def batches(data, split, actual_sampler, epoch, device, model_seed, actual_args, *, timing):
        assert actual_args.sample_seed_batch_size == 64
        assert actual_sampler is sampler
        calls.append(epoch)
        for selected in (torch.arange(3), torch.arange(3, 6)):
            with timing.stage("sampling_and_loader_wait"):
                prepared = graph
            with timing.stage("host_to_device"):
                yield_graph = prepared
            yield yield_graph, selected

    class Monitor:
        def __init__(self, device):
            self.device = device

        def start(self):
            return {"debug_fixture": True}

        def finish(self, **kwargs):
            return {"debug_fixture": True, **kwargs}

    monkeypatch.setattr(train, "_require_cuda", lambda device: None)
    monkeypatch.setattr(train, "validate_hardware_runtime", lambda args, device: {"debug": True})
    monkeypatch.setattr(train, "_prepare_data", lambda *args: (graph, indices, sampler))
    monkeypatch.setattr(train, "_training_batches", batches)
    monkeypatch.setattr(calibration, "RuntimeResourceMonitor", Monitor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10000, 20000))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2000)
    return args, payload, calls


def test_debug_cpu_smoke_runs_production_joint_model_loss_backward_optimizer_and_full_epochs(
    monkeypatch,
):
    args, payload, calls = _cpu_fixture(monkeypatch)
    before = torch.get_rng_state().clone()
    report = calibration.run_training_candidate(
        payload, args, torch.device("cpu"), physical_batch_size=64, workers=0,
        warmup_steps=2, measurement_steps=3, minimum_measure_seconds=0.000001,
    )
    assert report["status"] == "passed"
    assert report["calibration_not_final"] is True
    assert report["complete_warmup_epochs"] == 1
    assert report["complete_measurement_epochs"] == 2
    assert report["optimizer_steps"] == 4
    assert report["processed_units"] == 12
    assert calls == [1, 2, 3]
    assert report["optimizer_state_bytes"] > 0
    assert report["parameter_update_verified"] is True
    assert report["model_phase"] == "joint_all_condition_parameter_groups_active"
    assert report["configuration"]["layers"] == args.layers
    assert report["configuration"]["num_neighbors"] == args.num_neighbors
    assert report["samples_per_second"] > 0
    assert set(report["stage_seconds"]["cpu_wall_seconds"]) >= {
        "sampling_and_loader_wait", "host_to_device", "forward_and_loss", "backward",
        "gradient_clipping", "optimizer",
    }
    assert report["stage_seconds"]["cuda_event_seconds"] == {}
    assert torch.equal(torch.get_rng_state(), before)


def test_candidate_failure_is_reraised_with_observation_not_small_model_fallback(monkeypatch):
    args, payload, _ = _cpu_fixture(monkeypatch)
    before = torch.get_rng_state().clone()
    original_error = torch.OutOfMemoryError("synthetic oom; not a CUDA measurement")

    def fail(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(calibration, "_run_epoch", fail)
    with pytest.raises(torch.OutOfMemoryError) as failure:
        calibration.run_training_candidate(
            payload, args, torch.device("cpu"), physical_batch_size=64, workers=0,
        )
    assert failure.value is original_error
    assert failure.value.calibration_resource_observability["debug_fixture"] is True
    assert torch.equal(torch.get_rng_state(), before)


def test_production_candidate_refuses_cpu_even_when_payload_would_be_available():
    with pytest.raises(RuntimeError, match="CUDA"):
        calibration.run_training_candidate(
            {}, _args(), torch.device("cpu"), physical_batch_size=64, workers=0,
        )


def test_stage_cuda_events_use_the_requested_device_stream(monkeypatch):
    requested = torch.device("cuda:2")
    records = []
    selected_stream = object()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True

        def record(self, stream):
            records.append(stream)

        def elapsed_time(self, ended):
            assert isinstance(ended, Event)
            return 250.0

    def current_stream(device):
        assert device == requested
        return selected_stream

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    timing = calibration._StageTimer(requested)
    with timing.stage("forward_and_loss"):
        pass
    assert records == [selected_stream, selected_stream]
    assert timing.report()["cuda_event_seconds"]["forward_and_loss"] == 0.25


def test_monitor_cleanup_error_does_not_replace_primary_candidate_failure(monkeypatch):
    args, payload, _ = _cpu_fixture(monkeypatch)
    finish_calls = []
    original_error = RuntimeError("primary model failure")

    def fail(*args, **kwargs):
        raise original_error

    def fail_finish(self, **kwargs):
        finish_calls.append(kwargs)
        raise RuntimeError("secondary monitor failure")

    monkeypatch.setattr(calibration, "_run_epoch", fail)
    monkeypatch.setattr(calibration.RuntimeResourceMonitor, "finish", fail_finish)
    with pytest.raises(RuntimeError, match="primary model failure") as failure:
        calibration.run_training_candidate(
            payload, args, torch.device("cpu"), physical_batch_size=64, workers=0,
        )
    assert failure.value is original_error
    assert len(finish_calls) == 1
    assert "secondary monitor failure" in " ".join(failure.value.__notes__)


@pytest.mark.parametrize("dataset,sampling,split,expected,axis", [
    ("ppi", "full", [0, 2, 3], 3, "graphs"),
    ("ogbn-arxiv", "neighbor", torch.tensor([True, False, True, True]),
     3, "sampled_seed_nodes"),
    ("cora", "full", torch.tensor([True, False, True]), 1, "full_graph"),
])
def test_calibration_group_uses_verified_payload_splits_not_guessed_graph_fields(
    monkeypatch, dataset, sampling, split, expected, axis,
):
    from scripts.calibrate_training_resources import _load_group

    args = _args(dataset, sampling)
    payload = {"splits": {"train": split}, "graphs": [{}]}
    protocol = {"data_sha256": "a" * 64, "split_sha256": {"train": "b" * 64}}
    monkeypatch.setattr(
        calibration, "load_calibration_payload", lambda actual: (payload, protocol)
    )
    actual_payload, identity, maximum, actual_axis = _load_group({"track": "conductance"}, args)
    assert actual_payload is payload
    assert maximum == expected
    assert actual_axis == axis
    assert identity["data_sha256"] == protocol["data_sha256"]
    assert identity["split_sha256"] == protocol["split_sha256"]
