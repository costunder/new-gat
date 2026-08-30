from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import gpu_profiles as profiles

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("capability", "tag"),
    [
        ("11.8", "cu118"),
        ("12.0", "cu118"),
        ("12.2", "cu118"),
        ("12.5", "cu118"),
        ("12.6", "cu126"),
        ("12.9", "cu126"),
        ("13.0", "cu126"),
        ("13.2", "cu126"),
    ],
)
def test_auto_selection_uses_two_fixed_profiles(capability: str, tag: str):
    assert profiles.select_install_tag(capability) == tag


@pytest.mark.parametrize("capability", ["10.2", "11.0", "11.7"])
def test_unsupported_old_driver_does_not_fall_back_to_cpu(capability: str):
    with pytest.raises(profiles.GPUProfileError, match="no locked profile"):
        profiles.select_install_tag(capability)


@pytest.mark.parametrize("capability", ["", "N/A", "12", "12.2oops", "12.2.0", "-12.2"])
def test_invalid_driver_output_is_rejected(capability: str):
    with pytest.raises(profiles.GPUProfileError, match="Cannot parse"):
        profiles.select_install_tag(capability)


@pytest.mark.parametrize("tag", profiles.CUDA_RUNTIMES)
def test_explicit_profile_is_preserved_on_compatible_driver(tag: str):
    assert profiles.select_install_tag("13.2", tag) == tag


@pytest.mark.parametrize("tag", ["cu126", "cu130", "cu132"])
def test_explicit_newer_runtime_is_not_silently_downgraded(tag: str):
    with pytest.raises(profiles.GPUProfileError, match="does not rely on CUDA minor-version"):
        profiles.select_install_tag("12.2", tag)


@pytest.mark.parametrize("tag", ["cu121", "cpu", "../../other", ""])
def test_only_registered_profiles_can_select_a_lock(tag: str):
    with pytest.raises(profiles.GPUProfileError, match="Unsupported CUDA_WHEEL_TAG"):
        profiles.select_install_tag("13.2", tag)
    with pytest.raises(profiles.GPUProfileError):
        profiles.lock_for_tag(tag)


def test_compatibility_profile_has_its_own_lock():
    assert profiles.lock_for_tag("cu118") == ROOT / "requirements-cu118-lock.txt"
    assert profiles.lock_for_tag("cu126") == ROOT / "requirements-lock.txt"


@pytest.mark.parametrize("version", ["2.28", "2.31", "2.35"])
def test_compatible_glibc_host_is_accepted(monkeypatch: pytest.MonkeyPatch, version: str):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(profiles.platform, "libc_ver", lambda: ("glibc", version))
    profiles.check_wheel_host("cu118")


@pytest.mark.parametrize(("libc", "version"), [("glibc", "2.17"), ("musl", "1.2"), ("", "")])
def test_incompatible_glibc_host_stops_before_install(
    monkeypatch: pytest.MonkeyPatch,
    libc: str,
    version: str,
):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(profiles.platform, "libc_ver", lambda: (libc, version))
    with pytest.raises(profiles.GPUProfileError, match="glibc"):
        profiles.check_wheel_host("cu118")


def test_cu118_arm_host_is_not_given_an_x86_wheel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "aarch64")
    with pytest.raises(profiles.GPUProfileError, match="x86_64"):
        profiles.check_wheel_host("cu118")


def test_python_without_a_cu118_wheel_stops_before_install(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(profiles.sys, "version_info", (3, 14, 0))
    with pytest.raises(profiles.GPUProfileError, match="Python 3.11-3.13"):
        profiles.check_wheel_host("cu118")


def test_profile_cli_needs_no_installed_research_packages(tmp_path: Path):
    completed = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts" / "gpu_profiles.py"), "--driver-cuda", "12.2"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "cu118 requirements-cu118-lock.txt"
    assert not list(tmp_path.iterdir())
