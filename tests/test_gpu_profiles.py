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
    assert profiles.select_install_profile(capability) == tag


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


@pytest.mark.parametrize("tag", ["cu121", "cpu", "legacy-cu118", "../../other", ""])
def test_only_registered_profiles_can_select_a_lock(tag: str):
    with pytest.raises(profiles.GPUProfileError, match="Unsupported CUDA_WHEEL_TAG"):
        profiles.select_install_tag("13.2", tag)
    with pytest.raises(profiles.GPUProfileError):
        profiles.lock_for_tag(tag)


def test_compatibility_profile_has_its_own_lock():
    assert profiles.lock_for_tag("cu118") == ROOT / "requirements-cu118-lock.txt"
    assert profiles.lock_for_tag("cu126") == ROOT / "requirements-lock.txt"


def test_legacy_profile_is_separate_from_the_cuda_tag(tmp_path: Path):
    assert profiles.lock_for_profile("legacy-cu118", tmp_path) == (
        tmp_path / "requirements-legacy-cu118-lock.txt"
    )
    assert profiles.cuda_tag_for_profile("legacy-cu118") == "cu118"
    assert profiles.PROFILE_CONSTRAINT_FILES["legacy-cu118"] == "constraints-legacy-cu118.txt"
    assert "legacy-cu118" not in profiles.CUDA_RUNTIMES
    assert "legacy-cu118" not in profiles.LOCK_FILES


@pytest.mark.parametrize("profile_id", ["auto", "cpu", "../requirements.txt", "", "cu121"])
def test_unregistered_profile_never_becomes_a_path(profile_id: str):
    for function in (
        profiles.lock_for_profile,
        profiles.cuda_tag_for_profile,
        profiles.check_wheel_host,
        profiles.check_install_target,
    ):
        with pytest.raises(profiles.GPUProfileError, match="Unsupported GPU profile"):
            function(profile_id)


@pytest.mark.parametrize("profile_id", profiles.PROFILE_LOCK_FILES)
def test_explicit_named_profile_is_preserved(profile_id: str):
    assert profiles.select_install_profile("13.2", profile_id) == profile_id


@pytest.mark.parametrize(
    ("profile_id", "tag"),
    [("legacy-cu118", "cu126"), ("cu118", "cu130"), ("cu126", "cu118")],
)
def test_conflicting_explicit_profile_and_tag_are_rejected(profile_id: str, tag: str):
    with pytest.raises(profiles.GPUProfileError, match="explicit selections must agree"):
        profiles.select_install_profile("13.2", profile_id, tag)


def test_profile_cannot_bypass_driver_or_raw_tag_validation():
    with pytest.raises(profiles.GPUProfileError, match="does not rely on CUDA minor-version"):
        profiles.select_install_profile("11.7", "legacy-cu118")
    with pytest.raises(profiles.GPUProfileError, match="Unsupported CUDA_WHEEL_TAG"):
        profiles.select_install_profile("12.2", "legacy-cu118", "legacy-cu118")
    with pytest.raises(profiles.GPUProfileError, match="Unsupported GPU profile"):
        profiles.select_install_profile("12.2", "../../other")


@pytest.mark.parametrize(
    ("version", "profile_id"),
    [
        ("2.6.0+cu118", "legacy-cu118"),
        ("2.7.1+cu118", "cu118"),
        ("2.13.0+cu126", "cu126"),
        ("2.13.0+cu130", "cu130"),
        ("2.13.0+cu132", "cu132"),
    ],
)
def test_installed_profile_identity_is_exact(version: str, profile_id: str):
    assert profiles.profile_for_torch_version(version) == profile_id


@pytest.mark.parametrize(
    "version",
    ["2.6.0", "2.6.0+cpu", "2.6.0+cu126", "2.6.0+cu118.custom", "2.7.0+cu118", ""],
)
def test_unknown_torch_identity_has_no_profile(version: str):
    assert profiles.profile_for_torch_version(version) is None


@pytest.mark.parametrize("version", ["2.28", "2.31", "2.35"])
def test_compatible_glibc_host_is_accepted(monkeypatch: pytest.MonkeyPatch, version: str):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(profiles.platform, "libc_ver", lambda: ("glibc", version))
    profiles.check_wheel_host("cu118")


def test_ubuntu_1804_accepts_only_opt_in_legacy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(profiles.platform, "libc_ver", lambda: ("glibc", "2.27"))
    monkeypatch.setattr(profiles.sys, "version_info", (3, 11, 0))
    assert profiles.select_install_profile("12.2") == "cu118"
    with pytest.raises(profiles.GPUProfileError, match="minimum 2.28"):
        profiles.check_wheel_host(profiles.select_install_profile("12.2"))
    selected = profiles.select_install_profile("12.2", "legacy-cu118")
    assert selected == "legacy-cu118"
    profiles.check_wheel_host(selected)


@pytest.mark.parametrize("version", ["2.17", "2.26"])
def test_legacy_whole_stack_still_needs_glibc_227(monkeypatch: pytest.MonkeyPatch, version: str):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(profiles.platform, "libc_ver", lambda: ("glibc", version))
    with pytest.raises(profiles.GPUProfileError, match="minimum 2.27"):
        profiles.check_wheel_host("legacy-cu118")


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


@pytest.mark.parametrize("profile_id", ["cu118", "legacy-cu118"])
def test_cu118_arm_host_is_not_given_an_x86_wheel(monkeypatch: pytest.MonkeyPatch, profile_id: str):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "aarch64")
    with pytest.raises(profiles.GPUProfileError, match="x86_64"):
        profiles.check_wheel_host(profile_id)


@pytest.mark.parametrize("profile_id", ["cu118", "legacy-cu118"])
@pytest.mark.parametrize("version", [(3, 10, 0), (3, 14, 0)])
def test_python_without_a_cu118_wheel_stops_before_install(
    monkeypatch: pytest.MonkeyPatch, profile_id: str, version: tuple[int, int, int]
):
    monkeypatch.setattr(profiles.platform, "system", lambda: "Linux")
    monkeypatch.setattr(profiles.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(profiles.sys, "version_info", version)
    with pytest.raises(profiles.GPUProfileError, match="Python 3.11-3.13"):
        profiles.check_wheel_host(profile_id)


def _missing_torch(name: str) -> str:
    assert name == "torch"
    raise profiles.importlib.metadata.PackageNotFoundError(name)


@pytest.mark.parametrize(("version", "profile_id"), profiles.TORCH_PROFILES.items())
def test_bootstrap_query_preserves_every_exact_installed_profile_without_imports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: str,
    profile_id: str,
):
    monkeypatch.setattr(profiles.importlib.metadata, "version", lambda name: version)

    def forbidden(*_args, **_kwargs):
        pytest.fail("The installed-profile query must not inspect a GPU, host ABI, or import Torch")

    monkeypatch.setattr(profiles.importlib, "import_module", forbidden)
    monkeypatch.setattr(profiles, "check_wheel_host", forbidden)
    monkeypatch.setattr(profiles, "check_install_target", forbidden)
    assert profiles.main(["--installed-profile"]) == 0
    assert capsys.readouterr().out.strip() == profile_id


def test_only_missing_torch_allows_default_bootstrap_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(profiles.importlib.metadata, "version", _missing_torch)
    assert profiles.main(["--installed-profile"]) == 0
    assert capsys.readouterr().out.strip() == "auto"


@pytest.mark.parametrize("version", ["2.6.0", "2.6.0+cpu", "2.6.0+cu118.custom", "2.14.0+cu126"])
def test_unknown_installed_torch_cannot_trigger_an_automatic_profile_change(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], version: str
):
    monkeypatch.setattr(profiles.importlib.metadata, "version", lambda name: version)
    assert profiles.main(["--installed-profile"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert f"Cannot automatically repair unregistered torch=={version}" in output.err


def test_broken_torch_metadata_cannot_be_treated_as_an_empty_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def corrupt_metadata(name: str) -> str:
        raise OSError("unreadable distribution metadata")

    monkeypatch.setattr(profiles.importlib.metadata, "version", corrupt_metadata)
    assert profiles.main(["--installed-profile"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "Cannot inspect" in output.err


@pytest.mark.parametrize(
    "extra_args",
    [["--driver-cuda", "12.2"], ["--profile", "legacy-cu118"], ["--check-host"]],
)
def test_readonly_profile_query_cannot_mix_installation_arguments(extra_args: list[str]):
    with pytest.raises(SystemExit) as error:
        profiles.main(["--installed-profile", *extra_args])
    assert error.value.code == 2


def test_legacy_install_requires_a_separate_named_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(profiles.sys, "prefix", str(tmp_path / "new-gat"))
    monkeypatch.setattr(profiles.importlib.metadata, "version", _missing_torch)
    with pytest.raises(profiles.GPUProfileError, match="separate Conda environment"):
        profiles.check_install_target("legacy-cu118")


@pytest.mark.parametrize("version", ["2.7.1+cu118", "2.13.0+cu126", "2.6.0", "2.6.0+cpu"])
def test_legacy_install_never_replaces_a_different_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str
):
    monkeypatch.setattr(profiles.sys, "prefix", str(tmp_path / "new-gat-legacy"))
    monkeypatch.setattr(profiles.importlib.metadata, "version", lambda name: version)
    with pytest.raises(profiles.GPUProfileError, match="cannot replace existing torch"):
        profiles.check_install_target("legacy-cu118")


def test_legacy_install_allows_fresh_dedicated_or_exact_existing_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(profiles.sys, "prefix", str(tmp_path / "new-gat-legacy"))
    monkeypatch.setattr(profiles.importlib.metadata, "version", _missing_torch)
    profiles.check_install_target("legacy-cu118")
    monkeypatch.setattr(profiles.importlib.metadata, "version", lambda name: "2.6.0+cu118")
    profiles.check_install_target("legacy-cu118")


def test_modern_install_does_not_apply_the_legacy_target_guard(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profiles.sys, "prefix", "/envs/new-gat")

    def unexpected_metadata_query(name: str):
        pytest.fail("modern installation must not inspect existing Torch through this guard")

    monkeypatch.setattr(profiles.importlib.metadata, "version", unexpected_metadata_query)
    for profile_id in profiles.LOCK_FILES:
        profiles.check_install_target(profile_id)


def test_profile_cli_checks_host_then_install_target(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    checked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        profiles, "check_wheel_host", lambda profile_id: checked.append(("host", profile_id))
    )
    monkeypatch.setattr(
        profiles, "check_install_target", lambda profile_id: checked.append(("target", profile_id))
    )
    assert (
        profiles.main(["--driver-cuda", "12.2", "--profile", "legacy-cu118", "--check-host"]) == 0
    )
    assert checked == [("host", "legacy-cu118"), ("target", "legacy-cu118")]
    assert capsys.readouterr().out.strip() == (
        "legacy-cu118 cu118 requirements-legacy-cu118-lock.txt constraints-legacy-cu118.txt"
    )


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
    assert completed.stdout.strip() == (
        "cu118 cu118 requirements-cu118-lock.txt constraints-cu118.txt"
    )
    assert not list(tmp_path.iterdir())


def test_installed_profile_query_works_without_site_packages(tmp_path: Path):
    completed = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts" / "gpu_profiles.py"), "--installed-profile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "auto"
    assert not list(tmp_path.iterdir())
