from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from chartgat import observability
from chartgat.observability import RuntimeResourceMonitor, observed


def test_missing_observation_requires_an_explicit_reason() -> None:
    with pytest.raises(ValueError, match="requires a reason"):
        observed(None)
    assert observed(None, reason="counter unavailable", unit="bytes") == {
        "value": None,
        "reason": "counter unavailable",
        "unit": "bytes",
    }


def test_cpu_monitor_reports_interval_series_and_honest_cuda_absence() -> None:
    monitor = RuntimeResourceMonitor(
        torch.device("cpu"),
        sample_interval_seconds=0.005,
    )
    start = monitor.start()
    time.sleep(0.025)
    report = monitor.finish(peak_allocated_bytes=0, peak_reserved_bytes=0)

    assert report["start"] == start
    assert report["sample_interval_seconds"] == 0.005
    assert report["interval_series"]["process_resident_bytes"]["sample_count"] >= 0
    gpu = report["interval_series"]["gpu_sm_utilization_percent"]
    assert gpu["sample_count"] == 0
    assert gpu["mean"]["value"] is None
    assert "not CUDA" in gpu["mean"]["reason"]
    assert report["summary"]["run_average_gpu_sm_utilization_percent"]["value"] is None
    allocator_peak = report["summary"]["cuda_allocator_peak_allocated_bytes"]
    assert allocator_peak["value"] is None
    assert "not CUDA" in allocator_peak["reason"]


def test_monitor_rejects_invalid_lifecycle_and_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        RuntimeResourceMonitor(torch.device("cpu"), sample_interval_seconds=0)
    monitor = RuntimeResourceMonitor(torch.device("cpu"), sample_interval_seconds=0.01)
    with pytest.raises(RuntimeError, match="not started"):
        monitor.finish(peak_allocated_bytes=0, peak_reserved_bytes=0)
    monitor.start()
    with pytest.raises(RuntimeError, match="already started"):
        monitor.start()
    monitor.finish(peak_allocated_bytes=0, peak_reserved_bytes=0)


def test_cuda_utilization_uses_safe_nvidia_smi_fallback_without_pynvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_pynvml(_device: torch.device) -> int:
        raise ModuleNotFoundError("No module named 'pynvml'")

    seen: dict[str, object] = {}

    def completed(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen.update(command=command, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout="73, 41\n", stderr="")

    monkeypatch.setattr(torch.cuda, "utilization", missing_pynvml)
    monkeypatch.setattr(torch.cuda, "memory_usage", missing_pynvml)
    monkeypatch.setattr(observability.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(observability.subprocess, "run", completed)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(uuid="GPU-12345678-1234-1234-1234-123456789abc"),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    sm, memory, sources = observability._cuda_utilization_observations(
        torch.device("cuda:0")
    )

    assert sm == {"value": 73.0, "reason": None, "unit": "percent"}
    assert memory == {"value": 41.0, "reason": None, "unit": "percent"}
    assert sources == {
        "sm_utilization_percent": "nvidia-smi",
        "memory_controller_utilization_percent": "nvidia-smi",
    }
    assert seen["command"] == [
        "/usr/bin/nvidia-smi",
        "--id=GPU-12345678-1234-1234-1234-123456789abc",
        "--query-gpu=utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    assert seen["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 3.0,
    }


def test_cuda_utilization_explains_when_both_nvml_paths_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_pynvml(_device: torch.device) -> int:
        raise ModuleNotFoundError("No module named 'pynvml'")

    monkeypatch.setattr(torch.cuda, "utilization", missing_pynvml)
    monkeypatch.setattr(torch.cuda, "memory_usage", missing_pynvml)
    monkeypatch.setattr(observability.shutil, "which", lambda _: None)

    sm, memory, sources = observability._cuda_utilization_observations(
        torch.device("cuda:0")
    )

    assert sm["value"] is memory["value"] is None
    assert "pynvml" in sm["reason"] and "nvidia-smi is not available" in sm["reason"]
    assert "pynvml" in memory["reason"] and "nvidia-smi is not available" in memory["reason"]
    assert set(sources.values()) == {"unavailable"}


def test_unmapped_cuda_device_reports_uuid_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def fail_properties(_device: torch.device) -> object:
        raise RuntimeError("driver mapping unavailable")

    monkeypatch.setattr(torch.cuda, "get_device_properties", fail_properties)

    identifier, reason = observability._nvidia_smi_device_identifier(
        torch.device("cuda:0")
    )

    assert identifier is None
    assert reason is not None
    assert "could not be mapped safely" in reason
    assert "UUID lookup failed" in reason
    assert "driver mapping unavailable" in reason


def test_numeric_visible_device_is_not_guessed_when_uuid_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(uuid=None),
    )

    identifier, reason = observability._nvidia_smi_device_identifier(
        torch.device("cuda:0")
    )

    assert identifier is None
    assert reason is not None
    assert "numeric ordinals" in reason
    assert "cannot be mapped safely" in reason


def test_legacy_mig_visible_identifier_is_accepted_as_one_safe_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_mig = "MIG-GPU-12345678-1234-1234-1234-123456789abc/7/0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", legacy_mig)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(uuid=None),
    )

    identifier, reason = observability._nvidia_smi_device_identifier(
        torch.device("cuda:0")
    )

    assert identifier == legacy_mig
    assert reason is None


def test_monitor_fails_closed_when_sampler_thread_does_not_stop() -> None:
    class StuckThread:
        def join(self, *, timeout: float) -> None:
            assert timeout >= 5.0

        def is_alive(self) -> bool:
            return True

    monitor = RuntimeResourceMonitor(torch.device("cpu"))
    monitor.start_snapshot = observability.runtime_resource_snapshot(torch.device("cpu"))
    monitor._thread = StuckThread()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="did not stop"):
        monitor.finish(peak_allocated_bytes=None, peak_reserved_bytes=None)
