from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.verify_gpu_lock import (
    REQUIRED_RESEARCH_PACKAGES,
    LockVerificationError,
    assert_same_pins,
    read_exact_pins,
    verify_environment,
    version_matches,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock.txt"
CUDA_TAGS = ("cu126", "cu130", "cu132")


def test_all_cuda_constraints_are_exact_and_match_portable_lock() -> None:
    expected = read_exact_pins(LOCK)
    assert REQUIRED_RESEARCH_PACKAGES <= expected.keys()
    for tag in CUDA_TAGS:
        path = ROOT / f"constraints-{tag}.txt"
        assert path.read_text(encoding="utf-8").splitlines()[0] == f"# CUDA_WHEEL_TAG={tag}"
        assert assert_same_pins(LOCK, path) == expected


def test_lock_contains_python_311_compatible_numeric_stack() -> None:
    pins = read_exact_pins(LOCK)
    assert pins["numpy"] == "2.4.6"
    assert pins["scipy"] == "1.17.1"
    assert pins["torch"] == "2.13.0"
    assert pins["torch-geometric"] == "2.8.0.post1"
    assert pins["ogb"] == "1.3.6"
    assert pins["scikit-learn"] == "1.9.0"


def test_exact_pin_parser_rejects_ranges(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("torch>=2.2\n", encoding="utf-8")
    with pytest.raises(LockVerificationError, match="exact name==version"):
        read_exact_pins(invalid)


def test_only_torch_may_have_a_cuda_local_version_suffix() -> None:
    assert version_matches("torch", "2.13.0", "2.13.0+cu126")
    assert not version_matches("torch", "2.13.0", "2.12.1+cu126")
    assert not version_matches("numpy", "2.4.6", "2.4.6+local")


def test_environment_verifier_checks_exact_versions_and_cuda_runtime() -> None:
    pins = read_exact_pins(LOCK)
    installed = {**pins, "torch": f"{pins['torch']}+cu126"}
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    def import_module(name: str) -> object:
        return fake_torch if name == "torch" else object()

    with (
        patch(
            "scripts.verify_gpu_lock.importlib.metadata.version",
            side_effect=installed.__getitem__,
        ),
        patch("scripts.verify_gpu_lock.importlib.import_module", side_effect=import_module),
    ):
        report = verify_environment(
            lock_path=LOCK,
            constraints_path=ROOT / "constraints-cu126.txt",
            cuda_tag="cu126",
        )

    assert report["status"] == "passed"
    assert report["torch_cuda_runtime"] == "12.6"
    assert report["installed_top_level_versions"]["torch"].endswith("+cu126")


def test_gpu_setup_uses_lock_and_has_no_cu118_install_branch() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    assert 'constraints_file="${project_root}/constraints-${wheel_tag}.txt"' in source
    assert '--requirement "${lock_file}"' in source
    assert "scripts/verify_gpu_lock.py" in source
    assert 'wheel_tag="cu118"' not in source
    assert "TORCH_SPEC" not in source
    assert "TORCH_INDEX_URL" not in source
    assert "requires a driver supporting CUDA 12.6+" in source


def test_gpu_setup_uses_fixed_reference_runtime_and_opt_in_unit_tests() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    assert 'wheel_tag="${CUDA_WHEEL_TAG:-cu126}"' in source
    assert 'if [[ "${RUN_TESTS:-0}" == "1" ]]' in source
    assert "SKIP_TESTS" not in source
    assert 'wheel_tag="cu132"' not in source


def test_conda_bootstrap_uses_named_environment_and_python_311() -> None:
    import yaml

    environment = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    assert environment["name"] == "new-gat"
    assert environment["channels"] == ["conda-forge", "nodefaults"]
    assert "python=3.11" in environment["dependencies"]
    assert "pip" in environment["dependencies"]
