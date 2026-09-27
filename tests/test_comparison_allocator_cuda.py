"""Explicit CUDA allocator smoke test; no benchmark/model size reduction."""

from types import SimpleNamespace

import pytest
import torch

from experiments.aggregation_comparison import engine, memory
from tests.test_aggregation_comparison_cuda import cuda_required, reference_arguments  # noqa: F401


def test_native_budget_rejects_oversized_allocation_and_keeps_gpu_usable():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    previous = torch.cuda.get_per_process_memory_fraction(device)
    live = torch.cuda.memory_allocated(device)
    limit = live + 64 * 1024**2
    try:
        actual = memory.configure(SimpleNamespace(cuda_allocator_limit_gib=limit / 1024**3), device)
        assert actual == limit
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(limit + 64 * 1024**2, dtype=torch.uint8, device=device)
        small = torch.ones(1024, device=device)
        assert small.sum().item() == 1024
        del small
    finally:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(previous, device)


def test_hardware_preflight_error_is_not_masked_by_memory_reporting(monkeypatch):
    def fail(*args):
        raise ValueError("debug rejected hardware before memory snapshot")

    monkeypatch.setattr(engine.base, "validate_hardware_runtime", fail)
    with pytest.raises(ValueError, match="debug rejected hardware"):
        engine.run_calibration_candidate(
            {},
            reference_arguments("incidence", "fp32"),
            torch.device("cuda:0"),
            physical_batch_size=1,
            workers=0,
        )
