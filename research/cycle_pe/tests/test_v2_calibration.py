"""Explicit synthetic CPU fixtures for calibration mechanics, not GPU measurements."""

from __future__ import annotations

import copy
import gc
import random
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.cycle_pe.v2 import benchmark, calibration
from research.cycle_pe.v2.data import collate, prepare_graph


@pytest.fixture(autouse=True)
def bounded_debug_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def debug_args(encoding="se"):
    args = benchmark.parser().parse_args([])
    args.dataset = "zinc12k"
    args.encoding = encoding
    # Explicit CPU unit-fixture architecture; production defaults are untouched.
    args.hidden_dim, args.pe_dim, args.layers = 16, 8, 2
    args.batch_size = 2
    return args


def debug_graph():
    return prepare_graph(
        SimpleNamespace(
            num_nodes=3,
            x=torch.tensor([[0], [1], [2]]),
            edge_index=torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]]),
            edge_attr=torch.ones(6, 1, dtype=torch.long),
            y=torch.tensor([0.7]),
        ),
        basis_backend="dfs_fundamental",
    )


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_debug_real_training_step_updates_exact_probe_model_and_adam(encoding):
    args = debug_args(encoding)
    unchanged = copy.deepcopy(vars(args))
    device = torch.device("cpu")
    with calibration._isolated_rng(args.model_seed, device):
        model = calibration._probe_model(args.dataset, args, device)
        reference = benchmark.CycleBasisPEModel(
            dataset=args.dataset, encoding=args.encoding, hidden=args.hidden_dim,
            pe_dim=args.pe_dim, layers=args.layers, ffn_multiplier=args.ffn_multiplier,
            dropout=args.dropout, layer_scale=args.layer_scale,
        )
        assert {
            key: value.shape for key, value in model.state_dict().items()
        } == {key: value.shape for key, value in reference.state_dict().items()}
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        batch = collate([debug_graph(), debug_graph()])
        precision = benchmark._amp_policy(False, device)
        assert calibration._optimizer_tensor_bytes(optimizer) == 0
        report = calibration._training_step(
            model, optimizer, batch, device, precision, check_gradients=True
        )
        assert report["validated"] is True
        assert calibration._optimizer_tensor_bytes(optimizer) > 0
        assert all(state["step"].item() == 1 for state in optimizer.state.values())
        assert any(
            not torch.equal(initial[key], value) for key, value in model.state_dict().items()
        )
    assert vars(args) == unchanged


@pytest.mark.parametrize("fail", [False, True])
def test_debug_rng_restored_even_when_probe_raises(fail):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    previous = torch.backends.cudnn.benchmark
    try:
        with calibration._isolated_rng(77, torch.device("cpu")):
            random.random()
            np.random.rand()
            torch.rand(4)
            if fail:
                raise RuntimeError("debug probe failure")
    except RuntimeError as error:
        assert fail and str(error) == "debug probe failure"
    assert random.getstate() == state[0]
    restored = np.random.get_state()
    assert restored[0] == state[1][0] and np.array_equal(restored[1], state[1][1])
    assert restored[2:] == state[1][2:]
    assert torch.equal(torch.get_rng_state(), state[2])
    assert torch.backends.cudnn.benchmark == previous


@pytest.mark.parametrize("batch,workers", [(1, 2), (11, 2), (True, 2), (2, -1), (2, True)])
def test_candidates_cannot_reduce_batch_or_use_invalid_values(batch, workers):
    with pytest.raises(ValueError):
        calibration._candidate_arguments(debug_args(), batch, workers, 10)


def test_candidate_arguments_are_independent_and_preserve_full_profile():
    args = debug_args()
    original = copy.deepcopy(vars(args))
    candidate = calibration._candidate_arguments(args, 8, 6, 10)
    assert (candidate.batch_size, candidate.effective_batch_size, candidate.workers) == (8, 8, 6)
    assert vars(args) == original
    for key, value in original.items():
        if key not in {"batch_size", "workers"}:
            assert getattr(candidate, key) == value


def test_candidate_preserves_configured_floor_larger_than_complete_training_split():
    args = debug_args()
    args.batch_size = 8
    candidate = calibration._candidate_arguments(args, 8, 4, 3)
    assert candidate.batch_size == candidate.effective_batch_size == 8
    with pytest.raises(ValueError, match="configured batch floor"):
        calibration._candidate_arguments(args, 3, 4, 3)


def test_calibration_api_has_no_cpu_fallback():
    with pytest.raises(RuntimeError, match="requires CUDA; no CPU fallback"):
        calibration.run_training_candidate(
            [], debug_args(), torch.device("cpu"), physical_batch_size=2, workers=4
        )


def test_load_calibration_graphs_requires_entire_official_train_split(monkeypatch):
    calls = []
    graphs = [object()] * 10000  # Mock identity only, never passed as real training data.
    identity = {
        "official_splits": True,
        "split_sizes": {"train": 10000},
        "split_content_sha256": {"train": "a" * 64},
    }

    def load(root, dataset, **kwargs):
        calls.append((root, dataset, kwargs))
        return {"train": graphs}, identity

    monkeypatch.setattr(calibration, "load_benchmark", load)
    args = debug_args()
    result, provenance = calibration.load_calibration_graphs(args)
    assert result is graphs and provenance is identity
    assert calls[0][2] == {
        "allow_download": False, "splits": ("train",),
        "basis_backend": "dfs_fundamental", "workers": 4,
    }
    graphs.pop()
    with pytest.raises(ValueError, match="complete verified official train split"):
        calibration.load_calibration_graphs(args)


def test_full_model_parameter_budget_still_enforced():
    args = debug_args()
    args.max_parameters = 1
    with pytest.raises(ValueError, match="above the declared budget"):
        calibration._probe_model(args.dataset, args, torch.device("cpu"))


def test_main_training_scope_no_longer_mislabels_loader_and_optimizer_as_excluded():
    import inspect

    source = inspect.getsource(benchmark._train_model)
    assert "DataLoader wait, sparse collate" in source
    assert "gradient clip and Adam included" in source
    assert "validation and IO excluded" not in source
    assert "research/cycle_pe/v2/calibration.py" in benchmark.IMPLEMENTATION_FILES


def test_debug_mock_cuda_control_flow_measures_loader_h2d_backward_and_adam(monkeypatch):
    """Mock CUDA counters only; real CPU Adam verifies all eight update calls."""
    args = debug_args()
    args.workers = 3
    graphs = [SimpleNamespace(cost=cost) for cost in (1, 4, 2, 5, 3)]
    calls, loader_refs, selected_costs = [], [], []

    class DebugBatch:
        def __init__(self, count):
            self.x = torch.arange(count, dtype=torch.float32).reshape(-1, 1)
            self.y = torch.ones(count, 1)
            self.edge_index = torch.empty(2, count, dtype=torch.long)

        def pin_memory(self):
            calls.append("pin")
            return self

        def to(self, _device):
            calls.append("h2d")
            return self

    class DebugModel(torch.nn.Linear):
        def forward(self, batch):
            return super().forward(batch.x)

    class DebugLoader:
        def __iter__(self):
            yield DebugBatch(2)
            yield DebugBatch(2)
            yield DebugBatch(1)

    def loader(all_graphs, chosen, *, train):
        assert all_graphs is graphs and train
        assert (chosen.batch_size, chosen.workers) == (2, 3)
        instance = DebugLoader()
        loader_refs.append(weakref.ref(instance))
        return instance

    def stress_collate(selected):
        selected_costs.extend(graph.cost for graph in selected)
        return DebugBatch(len(selected))

    class DebugEvent:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            calls.append("event")

        def elapsed_time(self, _next):
            return 1.0

    original_step = calibration._training_step

    def cpu_step(model, optimizer, batch, _device, precision, **kwargs):
        calls.append("adam_step")
        return original_step(model, optimizer, batch, torch.device("cpu"), precision, **kwargs)

    monkeypatch.setattr(calibration, "_probe_model", lambda *_args: DebugModel(1, 1))
    monkeypatch.setattr(calibration, "_training_step", cpu_step)
    monkeypatch.setattr(calibration, "collate", stress_collate)
    monkeypatch.setattr(benchmark, "_graph_probe_cost", lambda graph: graph.cost)
    monkeypatch.setattr(benchmark, "_loader", loader)
    monkeypatch.setattr(
        benchmark, "_amp_policy",
        lambda *_args: {
            "enabled": False, "dtype": torch.float32, "dtype_name": "disabled",
            "fallback": None, "gradient_scaler": False,
        },
    )
    monkeypatch.setattr(torch.cuda, "Event", DebugEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args: 12345)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args: (100000, 200000))
    report = calibration._run_candidate(
        graphs, args, torch.device("cuda:0"), warmup_steps=2,
        measurement_steps=5, minimum_measure_seconds=1e-9,
    )
    assert selected_costs == [5, 4]
    assert calls.count("adam_step") == report["optimizer_steps"] == 8
    assert calls.count("h2d") == 8
    assert report["processed_units"] == 8
    assert report["observed_batch_sizes"] == [1, 2, 2, 1, 2]
    assert report["dataset_graph_count"] == 5 and report["training_dataset_reduced"] is False
    assert report["optimizer_state_bytes"] > 0
    assert report["first_task_gradient_connectivity"]["validated"] is True
    assert report["stage_seconds"]["h2d"] == pytest.approx(0.005)
    assert report["stage_seconds"]["optimizer"] == pytest.approx(0.005)
    assert report["samples_per_second"] == 8 / report["elapsed_seconds"]
    gc.collect()
    assert loader_refs[0]() is None


def test_debug_cuda_oom_is_explicit_measurement_and_cleans_own_monitor(monkeypatch):
    args = debug_args()
    calls = []

    class DebugMonitor:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            calls.append("monitor_start")
            return {"debug_mock": True}

        def finish(self, **kwargs):
            calls.append(("monitor_finish", kwargs))
            return {"debug_mock": True}

    def fail(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("explicit debug OOM fixture")

    monkeypatch.setattr(calibration, "FailureSafeResourceMonitor", DebugMonitor)
    monkeypatch.setattr(calibration, "_run_candidate", fail)
    # RNG restoration has independent real-CPU tests above; this test isolates OOM cleanup.
    from contextlib import nullcontext

    monkeypatch.setattr(calibration, "_isolated_rng", lambda *_args: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda *_args: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args: 12345)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_args: 23456)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args: (100000, 200000))
    original = copy.deepcopy(vars(args))
    report = calibration.run_training_candidate(
        [object(), object()], args, torch.device("cuda:0"), physical_batch_size=2, workers=4
    )
    assert report["status"] == "oom"
    assert "explicit debug OOM fixture" in report["error"]
    assert report["calibration_only"] is True and report["final_training_performed"] is False
    assert report["automatic_downsize"] is False
    assert report["measurement_steps_requested"] == 5
    assert report["warmup_steps_requested"] == 2
    assert report["minimum_measure_seconds_requested"] == 3.0
    assert report["peak_allocated_bytes"] == 12345
    assert sum(isinstance(call, tuple) and call[0] == "monitor_finish" for call in calls) == 1
    assert calls.count("empty_cache") == 2
    assert vars(args) == original
