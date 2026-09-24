"""CPU fixtures only: no real NVML/CUDA allocation or server claims."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import launch_incidence_mig as launcher


def records():
    return [
        {
            "physical_gpu": gpu,
            "mig_index": index,
            "gpu_instance": gi,
            "compute_instance": 0,
            "uuid": f"MIG-{gpu}-{gi}",
            "total_bytes": 10 * 2**30,
            "free_bytes": free * 2**30,
        }
        for gpu, index, gi, free in [(3, 0, 7, 10), (4, 0, 7, 9), (4, 1, 8, 10)]
    ]


def test_physical_gpu_automatic_selection_stays_within_requested_parent():
    selected = launcher.select_device(
        records(), physical_gpu=4, gpu_instance=None, compute_instance=None, min_free_gb=8
    )
    assert selected["uuid"] == "MIG-4-8"


def test_explicit_gi_ci_selects_exact_instance_even_when_another_has_more_memory():
    selected = launcher.select_device(
        records(), physical_gpu=4, gpu_instance=7, compute_instance=0, min_free_gb=8
    )
    assert selected["uuid"] == "MIG-4-7"
    with pytest.raises(RuntimeError, match="No requested"):
        launcher.select_device(
            records(), physical_gpu=4, gpu_instance=7, compute_instance=0, min_free_gb=9.5
        )


def test_launch_scopes_environment_before_fresh_torch_process(monkeypatch):
    monkeypatch.setattr(launcher, "nvml_devices", lambda _: records())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5")
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    calls = []
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *a, **k: calls.append((a, k)) or SimpleNamespace(returncode=0),
    )
    assert (
        launcher.main(
            [
                "--physical-gpu",
                "4",
                "--gpu-instance",
                "7",
                "--compute-instance",
                "0",
                "--",
                "--run-id",
                "kept",
            ]
        )
        == 0
    )
    argv, options = calls[0]
    assert argv[0][-2:] == ["--run-id", "kept"]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "MIG-4-7"
    assert "PYTORCH_NVML_BASED_CUDA_CHECK" not in options["env"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5"


def test_launcher_import_does_not_import_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import experiments.launch_incidence_mig; import sys; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_nvml_mapping_reads_actual_gi_ci_and_uuid_not_mig_index(monkeypatch):
    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append(self.name)
            if self.name == "nvmlDeviceGetHandleByIndex_v2":
                assert args[0] == 4
                args[1]._obj.value = 100
            elif self.name == "nvmlDeviceGetMaxMigDeviceCount":
                args[1]._obj.value = 2
            elif self.name == "nvmlDeviceGetMigDeviceHandleByIndex":
                if args[1] == 1:
                    return 6
                args[2]._obj.value = 200
            elif self.name == "nvmlDeviceGetGpuInstanceId":
                args[1]._obj.value = 7
            elif self.name == "nvmlDeviceGetComputeInstanceId":
                args[1]._obj.value = 0
            elif self.name == "nvmlDeviceGetUUID":
                args[1].value = b"MIG-actual-uuid"
            elif self.name == "nvmlDeviceGetMemoryInfo":
                args[1]._obj.total = 10 * 2**30
                args[1]._obj.free = 9 * 2**30
                args[1]._obj.used = 2**30
            return 0

    class Library:
        def __init__(self):
            self.functions = {}

        def __getattr__(self, name):
            return self.functions.setdefault(name, Function(name))

    monkeypatch.setattr(launcher.ct, "CDLL", lambda _: Library())
    actual = launcher.nvml_devices(4)
    assert len(actual) == 1
    assert actual[0]["gpu_instance"] == 7 and actual[0]["mig_index"] == 0
    assert actual[0]["uuid"] == "MIG-actual-uuid"
    assert calls[-1] == "nvmlShutdown"
