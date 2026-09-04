"""Numerical precision regressions for rebuilt Cycle PE V2."""

from __future__ import annotations

import inspect

import torch

from research.cycle_pe.v2 import benchmark


def test_amp_policy_uses_bfloat16_without_loss_scaling(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    policy = benchmark._amp_policy(True, torch.device("cuda:0"))

    assert policy["dtype"] == torch.bfloat16
    assert policy["enabled"] is True
    assert policy["gradient_scaler"] is False
    assert policy["fallback"] is None


def test_amp_policy_falls_back_to_fp32_not_fp16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    policy = benchmark._amp_policy(True, torch.device("cuda:0"))

    assert policy["dtype"] == torch.float32
    assert policy["enabled"] is False
    assert policy["gradient_scaler"] is False
    assert policy["fallback"] == "bf16_unavailable_use_fp32"


def test_resume_identity_binds_effective_precision(monkeypatch) -> None:
    args = benchmark.parser().parse_args(["--amp"])
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    bf16 = benchmark._resume_configuration("zinc12k", args)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    fp32 = benchmark._resume_configuration("zinc12k", args)

    assert bf16["schema"] == fp32["schema"] == "cycle-dfs-se-relative-pe-v2-epoch-resume-1"
    assert bf16["precision"]["autocast_dtype"] == "bfloat16"
    assert fp32["precision"]["autocast_dtype"] == "disabled"
    assert bf16 != fp32


def test_training_retains_strict_nonfinite_detection() -> None:
    source = inspect.getsource(benchmark._train_model)
    finite_guard_source = inspect.getsource(benchmark._require_finite_loss)
    test_source = inspect.getsource(benchmark._evaluate_test_checkpoint)

    assert "dtype=torch.float16" not in source
    assert "error_if_nonfinite=True" in source
    assert "_require_finite_loss(loss" in source
    assert "predicate = torch.isfinite(loss)" in finite_guard_source
    assert "_assert_async" in finite_guard_source
    assert "raise FloatingPointError(label)" in finite_guard_source
    assert '"precision": _precision_identity(precision)' in source
    assert "Selected checkpoint precision policy mismatch" in test_source
