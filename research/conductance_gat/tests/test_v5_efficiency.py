import inspect
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5 import model as v5_model
from research.conductance_gat.v5 import operator as v5_operator
from research.conductance_gat.v5.train import (
    build_parser,
    merge_efficiency,
    require_finite_gradient_norm_async,
    validate_args,
    validate_hardware_runtime,
)


def test_resume_efficiency_adds_elapsed_and_keeps_cross_segment_peak():
    merged = merge_efficiency(120.5, 8_000, 12_000, 30.25, 9_000, 11_000)
    assert merged == {
        "elapsed_seconds": 150.75,
        "peak_cuda_allocated_bytes": 9_000,
        "peak_cuda_reserved_bytes": 12_000,
    }


def test_resume_efficiency_accepts_legacy_zero_defaults_and_rejects_negative_values():
    assert merge_efficiency(0, 0, 0, 2.5, 100, 200)["elapsed_seconds"] == 2.5
    with pytest.raises(ValueError, match="nonnegative"):
        merge_efficiency(-1, 0, 0, 1, 1, 1)


def _a6000_args():
    args = build_parser().parse_args(
        [
            "--dataset",
            "ppi",
            "--condition",
            "shared_dynamic_c",
            "--output-dir",
            "out",
            "--hardware-profile",
            "a6000-48gb",
        ]
    )
    validate_args(args)
    return args


def test_a6000_profile_resolves_real_batches_and_fails_closed_on_mig(monkeypatch):
    args = _a6000_args()
    assert (args.batch_size, args.sample_seed_batch_size, args.edge_chunk_size) == (
        8,
        2048,
        131072,
    )
    assert args.precision == "bf16" and args.tf32 is True
    assert args.activation_checkpoint is False and args.sample_prefetch is True
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="MIG 1g.10gb", total_memory=10 * 2**30, major=8, minor=0
        ),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (9 * 2**30, 10 * 2**30))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    with pytest.raises(RuntimeError, match="at least 40 GiB"):
        validate_hardware_runtime(args, torch.device("cuda:0"))


def test_a6000_runtime_records_visible_capacity_and_has_no_fp32_fallback(monkeypatch):
    args = _a6000_args()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="RTX A6000", total_memory=48 * 2**30, major=8, minor=6
        ),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (44 * 2**30, 48 * 2**30))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="FP32 fallback is forbidden"):
        validate_hardware_runtime(args, torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    actual = validate_hardware_runtime(args, torch.device("cuda:0"))
    assert actual["compute_capability"] == [8, 6]
    assert actual["total_memory_bytes"] == 48 * 2**30
    assert actual["free_memory_bytes_at_start"] == 44 * 2**30
    assert actual["graph_batch_size"] == 8


def test_gradient_finite_assert_uses_async_primitive(monkeypatch):
    observed = []
    monkeypatch.setattr(
        torch, "_assert_async", lambda predicate, message: observed.append((predicate, message))
    )
    require_finite_gradient_norm_async(torch.tensor(3.0))
    assert len(observed) == 1 and bool(observed[0][0])


def test_model_and_operator_hot_paths_have_no_cuda_scalar_reduction_reads():
    source = inspect.getsource(v5_model) + inspect.getsource(v5_operator)
    for forbidden in (".item()", ".any()", ".all()", "torch.equal(", "int(batch.max"):
        assert forbidden not in source
