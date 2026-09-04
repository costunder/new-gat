from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from research.combined_later import run_certify, run_fixed_c, run_identifiability


def test_fixed_c_never_silently_falls_back_from_requested_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        run_fixed_c,
        "_run_impl",
        lambda *_args: pytest.fail("training must not start without requested CUDA"),
    )

    with pytest.raises(RuntimeError, match="no CPU fallback"):
        run_fixed_c.run(SimpleNamespace(device="cuda"))


def test_fixed_c_cpu_request_records_honest_resource_and_throughput(monkeypatch) -> None:
    summary = {"optimizer_steps": 8, "physical_batch_size_samples": 12}
    monkeypatch.setattr(
        run_fixed_c,
        "_run_impl",
        lambda _args, device: (pd.DataFrame(), summary | {"device": str(device)}),
    )

    _history, result = run_fixed_c.run(SimpleNamespace(device="cpu"))

    assert result["device"] == "cpu"
    assert result["resource_observability"]["measurement_scope"]
    gpu = result["resource_observability"]["summary"][
        "run_average_gpu_sm_utilization_percent"
    ]
    assert gpu["value"] is None and "not CUDA" in gpu["reason"]
    assert result["throughput"]["optimizer_steps_per_second"] > 0


@pytest.mark.parametrize(
    ("module", "option", "suffix"),
    [
        (run_fixed_c, "--output-dir", "fixed"),
        (run_identifiability, "--output-dir", "identifiability"),
        (run_certify, "--output", "certification.json"),
    ],
)
def test_combined_later_entrypoints_refuse_existing_outputs(
    module, option: str, suffix: str, tmp_path, monkeypatch
) -> None:
    output = tmp_path / suffix
    if output.suffix:
        output.write_text("preserve", encoding="utf-8")
    else:
        output.mkdir()
    monkeypatch.setattr(sys, "argv", [str(module.__file__), option, str(output)])

    with pytest.raises(FileExistsError):
        module.main()

    if output.is_file():
        assert output.read_text(encoding="utf-8") == "preserve"


def test_certification_failure_uses_explicit_error_not_system_exit() -> None:
    source = run_certify.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "raise SystemExit(1)" not in text
    assert "algebraic symmetry certification exceeded" in text
