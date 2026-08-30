from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_dependencies as checker
from scripts import run_paper
from scripts.gpu_profiles import CUDA_RUNTIMES, lock_for_tag

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_external_profile_selection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CUDA_WHEEL_TAG", raising=False)


def _installed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime: str | None = "12.6",
    tag: str = "cu126",
) -> dict[str, str]:
    pins = checker.read_exact_pins(lock_for_tag(tag))
    installed = {**pins, "torch": f"{pins['torch']}+{tag}"}
    monkeypatch.setattr(checker.importlib.metadata, "version", installed.__getitem__)
    # The dependency checker must not require a GPU allocation or run kernels.
    torch = SimpleNamespace(version=SimpleNamespace(cuda=runtime))
    monkeypatch.setattr(
        checker.importlib, "import_module", lambda name: torch if name == "torch" else object()
    )
    return installed


def test_missing_stack_is_reported_before_any_runtime_import(monkeypatch: pytest.MonkeyPatch):
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    def forbidden_import(name: str) -> None:
        raise AssertionError(f"Premature dependency import: {name}")

    monkeypatch.setattr(checker.importlib.metadata, "version", missing)
    monkeypatch.setattr(checker.importlib, "import_module", forbidden_import)
    with pytest.raises(checker.DependencyCheckError) as caught:
        checker.check_dependencies()
    assert "numpy: missing" in str(caught.value)
    assert "torch: missing" in str(caught.value)
    assert "torch-geometric: missing" in str(caught.value)


def test_complete_stack_is_valid_without_gpu_allocation(monkeypatch: pytest.MonkeyPatch):
    installed = _installed(monkeypatch)
    report = checker.check_dependencies()
    assert report["installed"] == installed
    assert report["cuda_runtime"] == "12.6"
    assert report["python"] == sys.executable


@pytest.mark.parametrize("tag", CUDA_RUNTIMES)
def test_installed_profile_is_reused_without_driver_query(
    monkeypatch: pytest.MonkeyPatch, tag: str
):
    installed = _installed(monkeypatch, runtime=CUDA_RUNTIMES[tag], tag=tag)
    report = checker.check_dependencies()
    assert report["installed"] == installed
    assert report["cuda_wheel_tag"] == tag
    assert Path(report["lock_path"]) == lock_for_tag(tag).resolve()
    assert len(report["lock_sha256"]) == 64


def test_cu118_does_not_accept_newer_unsupported_pyg(monkeypatch: pytest.MonkeyPatch):
    installed = _installed(monkeypatch, runtime="11.8", tag="cu118")
    installed["torch-geometric"] = "2.8.0.post1"
    with pytest.raises(checker.DependencyCheckError, match="required 2.7.0"):
        checker.check_dependencies()


def test_profile_runtime_mismatch_is_not_accepted(monkeypatch: pytest.MonkeyPatch):
    _installed(monkeypatch, runtime="12.6", tag="cu118")
    with pytest.raises(checker.DependencyCheckError, match="expected 11.8 for cu118"):
        checker.check_dependencies()


def test_explicit_profile_is_enforced_by_dependency_checker(monkeypatch: pytest.MonkeyPatch):
    _installed(monkeypatch, runtime="11.8", tag="cu118")
    monkeypatch.setenv("CUDA_WHEEL_TAG", "cu126")
    with pytest.raises(checker.DependencyCheckError, match="required 2.13.0"):
        checker.check_dependencies()


def test_bad_profile_has_an_actionable_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_WHEEL_TAG", "typo")
    with pytest.raises(checker.DependencyCheckError, match="Unsupported CUDA_WHEEL_TAG"):
        checker.check_dependencies()


@pytest.mark.parametrize("torch_version", ["2.7.1+custom", "2.7.1+cpu", "2.7.1"])
def test_explicit_profile_does_not_accept_a_custom_or_cpu_build(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
):
    installed = _installed(monkeypatch, runtime="11.8", tag="cu118")
    installed["torch"] = torch_version
    monkeypatch.setenv("CUDA_WHEEL_TAG", "cu118")
    with pytest.raises(checker.DependencyCheckError, match=r"required 2.7.1\+cu118"):
        checker.check_dependencies()


def test_wrong_package_version_is_reported(monkeypatch: pytest.MonkeyPatch):
    installed = _installed(monkeypatch)
    installed["numpy"] = "1.0.0"
    with pytest.raises(checker.DependencyCheckError, match="numpy: installed 1.0.0"):
        checker.check_dependencies()


def test_matching_versions_with_import_failure_are_not_ready(monkeypatch: pytest.MonkeyPatch):
    _installed(monkeypatch)

    def broken_import(name: str) -> object:
        if name == "numpy":
            raise OSError("binary dependency unavailable")
        return SimpleNamespace(version=SimpleNamespace(cuda="12.6"))

    monkeypatch.setattr(checker.importlib, "import_module", broken_import)
    with pytest.raises(checker.DependencyCheckError, match="numpy: import failed"):
        checker.check_dependencies()


def test_cpu_only_torch_is_not_the_research_stack(monkeypatch: pytest.MonkeyPatch):
    _installed(monkeypatch, runtime=None)
    with pytest.raises(checker.DependencyCheckError, match="torch CUDA runtime is None"):
        checker.check_dependencies()


def test_missing_lock_is_an_actionable_error(tmp_path: Path):
    with pytest.raises(checker.DependencyCheckError, match="Cannot read"):
        checker.check_dependencies(tmp_path / "missing.txt")


def test_direct_runner_stops_before_creating_output_when_dependencies_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def missing() -> None:
        raise checker.DependencyCheckError("numpy: missing")

    monkeypatch.setattr(run_paper, "check_dependencies", missing)
    monkeypatch.setattr(run_paper, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_paper.py", "--prepare-only"])
    assert run_paper.main() == 2
    error = capsys.readouterr().err
    assert "numpy: missing" in error
    assert "bash scripts/setup_gpu.sh" in error
    assert "Traceback" not in error
    assert not list(tmp_path.iterdir())


def test_runner_records_the_checked_profile_in_its_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _installed(monkeypatch, runtime="11.8", tag="cu118")
    report = checker.check_dependencies()
    monkeypatch.setattr(run_paper, "check_dependencies", lambda: report)
    monkeypatch.setattr(run_paper, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_paper, "_commands", lambda *_args: [])
    monkeypatch.setattr(run_paper, "_source_revision", lambda: {"revision": "unit-test"})
    monkeypatch.setattr(run_paper, "_environment_snapshot", lambda *_args: {})
    monkeypatch.setattr(run_paper, "_snapshot_registries", lambda *_args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_paper.py", "--prepare-only", "--run-id", "profile-record-test"],
    )
    assert run_paper.main() == 0
    manifest = json.loads(
        (tmp_path / "runs/paper/profile-record-test/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["research_environment"] == report
    assert manifest["research_environment"]["installed"]["torch"] == "2.7.1+cu118"


def _bare_python(tmp_path: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-S", *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_cache_import_does_not_require_numpy_or_torch(tmp_path: Path):
    completed = _bare_python(
        tmp_path,
        [
            "-c",
            "import sys; import chartgat.cache; "
            "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules",
        ],
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("option", ["--help", "--dry-run"])
def test_runner_read_only_commands_work_without_site_packages(tmp_path: Path, option: str):
    completed = _bare_python(tmp_path, [str(ROOT / "scripts" / "run_paper.py"), option])
    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr
    assert not list(tmp_path.iterdir())


def test_checker_cli_works_without_site_packages(tmp_path: Path):
    completed = _bare_python(tmp_path, [str(ROOT / "scripts" / "check_dependencies.py")])
    assert completed.returncode == 2
    assert "numpy: missing" in completed.stderr
    assert "bash scripts/setup_gpu.sh" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_lazy_algebra_exports_preserve_public_api():
    import chartgat
    from chartgat import algebra

    for name in chartgat.__all__:
        assert getattr(chartgat, name) is getattr(algebra, name)
    with pytest.raises(AttributeError):
        _ = chartgat.not_a_public_primitive
