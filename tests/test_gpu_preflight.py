from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import gpu_preflight as preflight


@pytest.fixture
def cuda_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(preflight.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        preflight.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(name="unit metadata", major=8, minor=0),
    )
    monkeypatch.setattr(
        preflight.torch.cuda,
        "mem_get_info",
        lambda _device: (4 * 1024**3, 8 * 1024**3),
    )


def test_hardware_report_never_creates_data_or_executes_models(
    cuda_metadata: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("hardware validation must not allocate sample tensors")

    for name in ("tensor", "randn", "rand", "zeros", "ones", "empty"):
        monkeypatch.setattr(preflight.torch, name, forbid)
    report = preflight.build_report("cuda")
    assert report["status"] == "passed"
    assert report["kind"] == "hardware_and_dependency_check"
    assert report["resolved_device"] == "cuda:0"
    assert report["dataset_loaded"] is False
    assert report["model_executed"] is False
    assert report["gpu"]["free_bytes"] == 4 * 1024**3
    assert report["visible_gpu_count"] == 1
    assert report["visible_gpus"][0]["logical_index"] == 0
    assert report["visible_gpus"][0]["mig_detected_from_name"] is False
    assert (
        report["environment_safety"][
            "pytorch_nvml_based_cuda_check_removed_before_torch_import"
        ]
        is (preflight._INHERITED_PYTORCH_NVML_BASED_CUDA_CHECK is not None)
    )
    assert "MIG" in report["environment_safety"]["reason"]
    assert {
        "logical_cpu_count",
        "logical_cpu_count_unavailable_reason",
        "cpu_affinity_count",
        "cpu_affinity_count_unavailable_reason",
        "available_ram_bytes",
        "available_ram_measurement",
        "available_ram_measurement_errors",
        "resource_environment",
    } <= report["host_resources"].keys()


def test_report_inventories_every_visible_gpu_and_marks_mig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(preflight.torch.cuda, "device_count", lambda: 2)

    def properties(device):
        index = device.index
        return SimpleNamespace(
            name="NVIDIA A100 MIG 1g.10gb" if index == 0 else "NVIDIA RTX A6000",
            major=8,
            minor=0 if index == 0 else 6,
        )

    monkeypatch.setattr(preflight.torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(
        preflight.torch.cuda,
        "mem_get_info",
        lambda device: ((10 if device.index == 0 else 40) * 1024**3, 48 * 1024**3),
    )
    monkeypatch.setattr(
        preflight,
        "_host_resources",
        lambda: {
            "logical_cpu_count": 16,
            "cpu_affinity_count": 8,
            "available_ram_bytes": 32 * 1024**3,
            "available_ram_measurement": "test",
            "resource_environment": {},
        },
    )

    report = preflight.build_report("cuda:1")
    assert report["visible_gpu_count"] == 2
    assert [row["logical_index"] for row in report["visible_gpus"]] == [0, 1]
    assert [row["mig_detected_from_name"] for row in report["visible_gpus"]] == [True, False]
    assert report["gpu"]["logical_index"] == 1
    assert report["host_resources"]["cpu_affinity_count"] == 8


@pytest.mark.parametrize("device", ["cpu", "mps", "auto", "not-a-device"])
def test_non_cuda_devices_are_rejected(device: str) -> None:
    with pytest.raises(preflight.PreflightError):
        preflight.build_report(device)


def test_missing_cuda_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: False)
    with pytest.raises(preflight.PreflightError, match="CUDA is unavailable"):
        preflight.build_report("cuda")


def test_out_of_range_gpu_is_rejected(cuda_metadata: None) -> None:
    with pytest.raises(preflight.PreflightError, match="index 2"):
        preflight.build_report("cuda:2")


def test_cuda_initialization_failure_is_normalized(
    cuda_metadata: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> int:
        raise RuntimeError("driver initialization error")

    monkeypatch.setattr(preflight.torch.cuda, "current_device", unavailable)
    with pytest.raises(preflight.PreflightError, match="CUDA initialization failed"):
        preflight.build_report("cuda")


def test_report_write_error_does_not_hide_original_gpu_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unwritable(*_args: object) -> None:
        raise PermissionError("read-only output")

    monkeypatch.setattr(preflight, "atomic_write_json", unwritable)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu_preflight.py", "--device", "cpu", "--json-out", str(tmp_path / "gpu.json")],
    )
    assert preflight.main() == 2
    stderr = capsys.readouterr().err
    assert stderr.index("requires CUDA") < stderr.index("cannot save GPU report")


@pytest.mark.parametrize("minimum", [-1, float("nan"), float("inf")])
def test_invalid_memory_requirement_is_rejected(minimum: float) -> None:
    with pytest.raises(preflight.PreflightError, match="finite and non-negative"):
        preflight.build_report("cuda", min_free_gb=minimum)


def test_insufficient_free_memory_is_rejected(cuda_metadata: None) -> None:
    with pytest.raises(preflight.PreflightError, match="4.00 GiB free"):
        preflight.build_report("cuda", min_free_gb=5)


def test_import_time_abi_failure_is_reported(
    cuda_metadata: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "PAPER_IMPORTS", {"scipy": "scipy"})

    def broken_import(_name: str) -> None:
        raise OSError("undefined symbol")

    monkeypatch.setattr(preflight.importlib, "import_module", broken_import)
    errors = preflight._paper_dependency_import_errors()
    assert errors == {"scipy": "OSError: undefined symbol"}
    with pytest.raises(preflight.PreflightError, match="dependency imports failed"):
        preflight.build_report("cuda", require_paper_dependencies=True)


def test_failed_cli_preserves_failure_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "gpu.json"
    monkeypatch.setattr(
        sys, "argv", ["gpu_preflight.py", "--device", "cpu", "--json-out", str(path)]
    )
    assert preflight.main() == 2
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "MIG" in report["environment_safety"]["reason"]


@pytest.mark.parametrize("option", ["--allow-cpu", "--profile", "--nodes-per-graph"])
def test_removed_synthetic_profile_options_are_rejected(
    option: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["gpu_preflight.py", option])
    with pytest.raises(SystemExit) as caught:
        preflight.main()
    assert caught.value.code == 2
