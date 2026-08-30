from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.gpu_profiles import CUDA_RUNTIMES, lock_for_tag
from scripts.verify_gpu_lock import (
    REQUIRED_RESEARCH_PACKAGES,
    LockVerificationError,
    assert_same_pins,
    read_exact_pins,
    verify_environment,
    verify_numpy_bridge,
    version_matches,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock.txt"
CUDA_TAGS = ("cu118", "cu126", "cu130", "cu132")
PROFILE_IDS = (*CUDA_TAGS, "legacy-cu118")


@pytest.mark.parametrize("tag", CUDA_TAGS)
def test_all_cuda_constraints_are_exact_and_match_profile_lock(tag: str) -> None:
    lock = lock_for_tag(tag, root=ROOT)
    expected = read_exact_pins(lock)
    assert REQUIRED_RESEARCH_PACKAGES <= expected.keys()
    path = ROOT / f"constraints-{tag}.txt"
    assert path.read_text(encoding="utf-8").splitlines()[0] == f"# CUDA_WHEEL_TAG={tag}"
    assert assert_same_pins(lock, path) == expected


def test_lock_contains_python_311_compatible_numeric_stack() -> None:
    pins = read_exact_pins(LOCK)
    assert pins["numpy"] == "2.4.6"
    assert pins["scipy"] == "1.17.1"
    assert pins["torch"] == "2.13.0"
    assert pins["torch-geometric"] == "2.8.0.post1"
    assert pins["ogb"] == "1.3.6"
    assert pins["scikit-learn"] == "1.9.0"


def test_cu118_profile_only_changes_torch_and_matching_pyg() -> None:
    reference = read_exact_pins(LOCK)
    compatibility = read_exact_pins(lock_for_tag("cu118", root=ROOT))
    assert compatibility == {**reference, "torch": "2.7.1", "torch-geometric": "2.7.0"}


def test_legacy_cu118_lock_is_separate_and_preserves_other_research_pins() -> None:
    reference = read_exact_pins(LOCK)
    legacy_lock = ROOT / "requirements-legacy-cu118-lock.txt"
    legacy_constraints = ROOT / "constraints-legacy-cu118.txt"
    expected = {**reference, "torch": "2.6.0", "torch-geometric": "2.7.0"}
    assert assert_same_pins(legacy_lock, legacy_constraints) == expected
    assert legacy_lock != lock_for_tag("cu118", root=ROOT)


def test_exact_pin_parser_rejects_ranges(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("torch>=2.2\n", encoding="utf-8")
    with pytest.raises(LockVerificationError, match="exact name==version"):
        read_exact_pins(invalid)


def test_only_torch_may_have_a_cuda_local_version_suffix() -> None:
    assert version_matches("torch", "2.13.0", "2.13.0+cu126")
    assert not version_matches("torch", "2.13.0", "2.12.1+cu126")
    assert not version_matches("numpy", "2.4.6", "2.4.6+local")
    assert not version_matches("torch", "2.7.1", "2.7.1+custom")
    assert version_matches("torch", "2.7.1", "2.7.1+cu118", cuda_tag="cu118")
    assert not version_matches("torch", "2.7.1", "2.7.1+cu126", cuda_tag="cu118")
    assert not version_matches("torch", "2.7.1", "2.7.1", cuda_tag="cu118")


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_environment_verifier_checks_exact_versions_and_cuda_runtime(profile_id: str) -> None:
    tag = "cu118" if profile_id == "legacy-cu118" else profile_id
    lock = (
        ROOT / "requirements-legacy-cu118-lock.txt"
        if profile_id == "legacy-cu118"
        else lock_for_tag(tag, root=ROOT)
    )
    pins = read_exact_pins(lock)
    installed = {**pins, "torch": f"{pins['torch']}+{tag}"}
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=CUDA_RUNTIMES[tag]),
        cuda=SimpleNamespace(is_available=lambda: True),
        from_numpy=lambda array: SimpleNamespace(numpy=lambda: array),
    )
    fake_numpy = SimpleNamespace(empty=lambda *_args, **_kwargs: [], float32="float32")

    def import_module(name: str) -> object:
        if name == "numpy":
            return fake_numpy
        return fake_torch if name == "torch" else object()

    with (
        patch(
            "scripts.verify_gpu_lock.importlib.metadata.version",
            side_effect=installed.__getitem__,
        ),
        patch("scripts.verify_gpu_lock.importlib.import_module", side_effect=import_module),
    ):
        report = verify_environment(
            lock_path=lock,
            constraints_path=ROOT / f"constraints-{profile_id}.txt",
            cuda_tag=tag,
            profile_id=profile_id,
        )

    assert report["status"] == "passed"
    assert report["torch_cuda_runtime"] == CUDA_RUNTIMES[tag]
    assert report["numpy_torch_interop"] == "passed"
    assert report["installed_top_level_versions"]["torch"].endswith(f"+{tag}")


def test_numpy_bridge_failure_is_an_actionable_setup_failure() -> None:
    def broken_bridge(_array: object) -> None:
        raise RuntimeError("NumPy is not available")

    torch = SimpleNamespace(from_numpy=broken_bridge)
    numpy = SimpleNamespace(empty=lambda *_args, **_kwargs: [], float32="float32")
    with pytest.raises(LockVerificationError, match="NumPy/Torch interoperability failed"):
        verify_numpy_bridge(torch, numpy)


def test_gpu_setup_uses_selected_profile_lock() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    assert 'constraints_file="${project_root}/${constraints_name}"' in source
    assert 'lock_file="${project_root}/${lock_name}"' in source
    assert (
        'read -r profile_id wheel_tag lock_name constraints_name <<< "${profile_selection}"'
        in source
    )
    assert '--requirement "${lock_file}"' in source
    assert "scripts/verify_gpu_lock.py" in source
    assert '--profile "${profile_id}"' in source
    assert "TORCH_SPEC" not in source
    assert "TORCH_INDEX_URL" not in source
    assert "requires a driver supporting CUDA 12.6+" not in source
    assert "driver_cuda_code" not in source
    assert '"torch==${torch_version}+${wheel_tag}"' in source


def test_gpu_setup_delegates_auto_profile_selection_and_keeps_unit_tests_opt_in() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    assert 'requested_tag="${CUDA_WHEEL_TAG:-auto}"' in source
    assert '"${environment_python}" "${project_root}/scripts/gpu_profiles.py"' in source
    assert 'requested_profile="auto"' in source
    assert '--driver-cuda "${cuda_version}" --cuda-tag "${requested_tag}"' in source
    assert '--profile "${requested_profile}" --check-host' in source
    assert 'if [[ "${RUN_TESTS:-0}" == "1" ]]' in source
    assert "SKIP_TESTS" not in source
    assert 'wheel_tag="cu132"' not in source


def test_setup_parses_arguments_before_environment_checks() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    conda_guard = source.index('source "${project_root}/scripts/conda_env.sh"')
    assert source.index("--help|-h)") < conda_guard
    assert source.index("--profile requires a profile name") < conda_guard
    assert source.index("Unknown setup argument") < conda_guard
    assert source.index("Unsupported profile") < conda_guard
    assert "eval " not in source


def test_legacy_snapshot_default_does_not_overwrite_project_snapshot() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    assert 'if [[ "${profile_id}" == "legacy-cu118" ]]' in source
    assert 'default_snapshot_dir="${CONDA_PREFIX%/}/.new-gat-environment"' in source
    assert 'snapshot_dir="${ENVIRONMENT_SNAPSHOT_DIR:-${default_snapshot_dir}}"' in source


def test_gpu_setup_always_installs_full_locked_dependencies() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    assert "SKIP_DEPS" not in source
    assert source.count('--requirement "${lock_file}"') == 1
    assert source.count('--no-deps --no-build-isolation -e "${project_root}"') == 1
    assert source.index("command -v nvidia-smi") < source.index("-m pip install")
    assert source.index("scripts/gpu_profiles.py") < source.index("-m pip install")
    assert source.index("GPU lock files are missing") < source.index("-m pip install")
    assert source.index("must contain exactly one torch==version pin") < source.index(
        "-m pip install"
    )
    assert source.index('--requirement "${lock_file}"') < source.index(
        '--no-deps --no-build-isolation -e "${project_root}"'
    )
    assert source.index('--requirement "${lock_file}"') < source.index("scripts/verify_gpu_lock.py")


def test_conda_bootstrap_uses_named_environment_and_python_311() -> None:
    import yaml

    environment = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    assert environment["name"] == "new-gat"
    assert environment["channels"] == ["conda-forge", "nodefaults"]
    assert "python=3.11" in environment["dependencies"]
    assert "pip" in environment["dependencies"]
