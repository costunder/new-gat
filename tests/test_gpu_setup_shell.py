"""Exercise the Linux setup shell with simulated drivers and no package installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None,
    reason="GPU setup is a Linux Bash entrypoint; all GPU/install commands are stubbed",
)


@pytest.fixture
def setup_sandbox(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in ("setup_gpu.sh", "conda_env.sh", "gpu_profiles.py"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    for name in (
        "requirements-lock.txt",
        "requirements-cu118-lock.txt",
        "requirements-legacy-cu118-lock.txt",
        "constraints-cu118.txt",
        "constraints-legacy-cu118.txt",
        "constraints-cu126.txt",
        "constraints-cu130.txt",
        "constraints-cu132.txt",
    ):
        shutil.copyfile(ROOT / name, project / name)

    environment = tmp_path / "conda environment with spaces"
    environment_bin = environment / "bin"
    environment_bin.mkdir(parents=True)
    commands = tmp_path / "fake commands"
    commands.mkdir()
    log = tmp_path / "python calls.jsonl"
    handler = tmp_path / "record python invocation.py"
    handler.write_text(
        textwrap.dedent(
            """\
            import json
            import importlib.metadata
            import os
            from pathlib import Path
            import platform
            import runpy
            import sys

            arguments = sys.argv[1:]
            with open(os.environ["TEST_CALL_LOG"], "a", encoding="utf-8") as output:
                output.write(json.dumps(arguments) + "\\n")
            if arguments and Path(arguments[0]).name == "verify_conda_env.py":
                raise SystemExit(0)
            if arguments and Path(arguments[0]).name == "gpu_profiles.py":
                # Simulate an explicitly supported host independently of the test host.
                platform.system = lambda: "Linux"
                platform.machine = lambda: "x86_64"
                platform.libc_ver = lambda: ("glibc", os.environ.get("TEST_GLIBC", "2.35"))
                sys.prefix = os.environ["CONDA_PREFIX"]
                def installed_version(name):
                    installed = os.environ.get("TEST_INSTALLED_TORCH")
                    if name == "torch" and installed:
                        return installed
                    raise importlib.metadata.PackageNotFoundError(name)
                importlib.metadata.version = installed_version
                sys.argv = arguments
                runpy.run_path(arguments[0], run_name="__main__")
                raise SystemExit(0)
            if arguments[:2] == ["-m", "pip"]:
                if arguments[2:3] == ["freeze"]:
                    print("simulated-package==1.0")
                raise SystemExit(0)
            if arguments and Path(arguments[0]).name in {
                "verify_gpu_lock.py", "gpu_preflight.py"
            }:
                raise SystemExit(0)
            raise SystemExit("Unexpected command in isolated setup test: " + repr(arguments))
            """
        ),
        encoding="utf-8",
    )
    python = environment_bin / "python"
    python.write_text(
        '#!/usr/bin/env bash\nexec "${TEST_REAL_PYTHON}" "${TEST_PYTHON_HANDLER}" "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    nvidia_smi = commands / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-L" ]]; then\n'
        '    printf "GPU 0: simulated test GPU\\n"\n'
        "else\n"
        '    printf "NVIDIA-SMI simulated   CUDA Version: %s\\n" "${TEST_DRIVER_CUDA}"\n'
        "fi\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    env = os.environ.copy()
    for key in ("CUDA_WHEEL_TAG", "RUN_TESTS", "ENVIRONMENT_SNAPSHOT_DIR", "DEVICE"):
        env.pop(key, None)
    env.update(
        {
            "CONDA_PREFIX": str(environment),
            "PATH": str(commands) + os.pathsep + env.get("PATH", ""),
            "TEST_REAL_PYTHON": sys.executable,
            "TEST_PYTHON_HANDLER": str(handler),
            "TEST_CALL_LOG": str(log),
            "TEST_DRIVER_CUDA": "12.2",
        }
    )
    return project, env, log


def _run_setup(
    setup_sandbox: tuple[Path, dict[str, str], Path],
    *arguments: str,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    project, env, log = setup_sandbox
    result = subprocess.run(
        ["bash", str(project / "scripts" / "setup_gpu.sh"), *arguments],
        cwd=project.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    return result, calls


@pytest.mark.parametrize(
    ("driver", "tag", "lock_name", "torch_version"),
    [
        ("12.2", "cu118", "requirements-cu118-lock.txt", "2.7.1"),
        ("12.6", "cu126", "requirements-lock.txt", "2.13.0"),
    ],
)
def test_setup_auto_selects_complete_profile_with_spaced_paths(
    setup_sandbox: tuple[Path, dict[str, str], Path],
    driver: str,
    tag: str,
    lock_name: str,
    torch_version: str,
) -> None:
    project, env, _ = setup_sandbox
    env["TEST_DRIVER_CUDA"] = driver
    result, calls = _run_setup(setup_sandbox)
    assert result.returncode == 0, result.stderr
    assert calls[0] == [str(project / "scripts" / "verify_conda_env.py")]
    assert calls[1] == [
        str(project / "scripts" / "gpu_profiles.py"),
        "--driver-cuda",
        driver,
        "--cuda-tag",
        "auto",
        "--profile",
        "auto",
        "--check-host",
    ]
    constraints = str(project / f"constraints-{tag}.txt")
    assert [
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--constraint",
        constraints,
        f"torch=={torch_version}+{tag}",
        "--index-url",
        f"https://download.pytorch.org/whl/{tag}",
    ] in calls
    assert [
        "-m",
        "pip",
        "install",
        "--constraint",
        constraints,
        "--requirement",
        str(project / lock_name),
    ] in calls
    verification = next(call for call in calls if call[0] == "scripts/verify_gpu_lock.py")
    assert verification[verification.index("--lock") + 1] == str(project / lock_name)
    assert verification[verification.index("--cuda-tag") + 1] == tag
    assert verification[verification.index("--profile") + 1] == tag
    preflight = next(call for call in calls if call[0] == "scripts/gpu_preflight.py")
    assert preflight[preflight.index("--device") + 1] == "cuda"
    assert "--require-paper-deps" in preflight
    assert f"GPU profile: {tag}" in result.stdout
    assert f"nvidia-smi CUDA compatibility: {driver}" in result.stdout


@pytest.mark.parametrize("syntax", [("--profile", "legacy-cu118"), ("--profile=legacy-cu118",)])
def test_legacy_profile_uses_real_cu118_wheel_and_separate_snapshot(
    setup_sandbox: tuple[Path, dict[str, str], Path],
    syntax: tuple[str, ...],
) -> None:
    project, env, _ = setup_sandbox
    env["TEST_GLIBC"] = "2.27"
    result, calls = _run_setup(setup_sandbox, *syntax)
    assert result.returncode == 0, result.stderr
    selection = calls[1]
    assert selection[selection.index("--profile") + 1] == "legacy-cu118"
    constraints = str(project / "constraints-legacy-cu118.txt")
    assert [
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--constraint",
        constraints,
        "torch==2.6.0+cu118",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    ] in calls
    assert [
        "-m",
        "pip",
        "install",
        "--constraint",
        constraints,
        "--requirement",
        str(project / "requirements-legacy-cu118-lock.txt"),
    ] in calls
    verification = next(call for call in calls if call[0] == "scripts/verify_gpu_lock.py")
    assert verification[verification.index("--profile") + 1] == "legacy-cu118"
    assert verification[verification.index("--cuda-tag") + 1] == "cu118"
    snapshot = Path(env["CONDA_PREFIX"]) / ".new-gat-environment" / ".gpu-environment.json"
    assert verification[verification.index("--json-out") + 1] == str(snapshot)
    assert not (project / ".gpu-environment.freeze.txt").exists()
    assert snapshot.with_name(".gpu-environment.freeze.txt").is_file()
    assert "GPU profile: legacy-cu118; CUDA wheel: cu118" in result.stdout


def test_legacy_snapshot_override_is_respected(
    setup_sandbox: tuple[Path, dict[str, str], Path],
) -> None:
    project, env, _ = setup_sandbox
    target = project.parent / "custom snapshot directory"
    env["ENVIRONMENT_SNAPSHOT_DIR"] = str(target)
    result, calls = _run_setup(setup_sandbox, "--profile", "legacy-cu118")
    assert result.returncode == 0, result.stderr
    verification = next(call for call in calls if call[0] == "scripts/verify_gpu_lock.py")
    assert verification[verification.index("--json-out") + 1] == str(
        target / ".gpu-environment.json"
    )


@pytest.mark.parametrize(
    ("arguments", "returncode", "message"),
    [
        (("--help",), 0, "Usage:"),
        (("-h",), 0, "Usage:"),
        (("--unknown",), 2, "Unknown setup argument"),
        (("legacy-cu118",), 2, "Unknown setup argument"),
        (("--profile",), 2, "requires a profile name"),
        (("--profile", "--help"), 2, "requires a profile name"),
        (("--profile=",), 2, "Unsupported profile"),
        (("--profile", "bad"), 2, "Unsupported profile"),
        (("--profile", "auto", "--profile", "cu118"), 2, "only be specified once"),
    ],
)
def test_setup_argument_handling_precedes_conda_and_installation(
    setup_sandbox: tuple[Path, dict[str, str], Path],
    arguments: tuple[str, ...],
    returncode: int,
    message: str,
) -> None:
    _, env, _ = setup_sandbox
    env.pop("CONDA_PREFIX")
    result, calls = _run_setup(setup_sandbox, *arguments)
    assert result.returncode == returncode
    assert message in result.stdout + result.stderr
    assert calls == []


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("existing_new_gat", "new-gat"),
        ("other_torch", "2.7.1"),
        ("wrong_wheel", "2.6.0+cu126"),
        ("old_glibc", "2.27"),
        ("conflicting_tag", "cu126"),
    ],
)
def test_legacy_preflight_failures_cannot_modify_the_environment(
    setup_sandbox: tuple[Path, dict[str, str], Path],
    failure: str,
    message: str,
) -> None:
    _, env, _ = setup_sandbox
    if failure == "existing_new_gat":
        prefix = Path(env["CONDA_PREFIX"])
        protected_prefix = prefix.with_name("new-gat")
        prefix.rename(protected_prefix)
        env["CONDA_PREFIX"] = str(protected_prefix)
    elif failure == "other_torch":
        env["TEST_INSTALLED_TORCH"] = "2.7.1+cu118"
    elif failure == "wrong_wheel":
        env["TEST_INSTALLED_TORCH"] = "2.6.0+cu126"
    elif failure == "old_glibc":
        env["TEST_GLIBC"] = "2.26"
    else:
        env["CUDA_WHEEL_TAG"] = "cu126"
    result, calls = _run_setup(setup_sandbox, "--profile", "legacy-cu118")
    assert result.returncode == 2
    assert message in result.stderr
    assert not any(call[:2] == ["-m", "pip"] for call in calls)


def test_auto_profile_does_not_silently_choose_legacy_on_old_glibc(
    setup_sandbox: tuple[Path, dict[str, str], Path],
) -> None:
    _, env, _ = setup_sandbox
    env["TEST_GLIBC"] = "2.27"
    result, calls = _run_setup(setup_sandbox)
    assert result.returncode == 2
    assert not any(call[:2] == ["-m", "pip"] for call in calls)


@pytest.mark.parametrize(
    ("driver", "requested_tag"),
    [("12.2", "cu126"), ("11.7", "auto"), ("12.2", "invalid")],
)
def test_setup_rejects_unsupported_profile_before_any_pip(
    setup_sandbox: tuple[Path, dict[str, str], Path],
    driver: str,
    requested_tag: str,
) -> None:
    _, env, _ = setup_sandbox
    env["TEST_DRIVER_CUDA"] = driver
    env["CUDA_WHEEL_TAG"] = requested_tag
    result, calls = _run_setup(setup_sandbox)
    assert result.returncode == 2
    assert not any(call[:2] == ["-m", "pip"] for call in calls)
    assert not any(call[0] == "scripts/gpu_preflight.py" for call in calls)


@pytest.mark.parametrize("failure", ["missing_lock", "missing_pin", "duplicate_pin"])
def test_setup_validates_profile_files_before_any_pip(
    setup_sandbox: tuple[Path, dict[str, str], Path],
    failure: str,
) -> None:
    project, _, _ = setup_sandbox
    if failure == "missing_lock":
        (project / "requirements-cu118-lock.txt").unlink()
    else:
        pins = "numpy==2.4.6\n" if failure == "missing_pin" else "torch==2.7.1\ntorch==2.7.1\n"
        (project / "constraints-cu118.txt").write_text(pins, encoding="utf-8")
    result, calls = _run_setup(setup_sandbox)
    assert result.returncode == 2
    assert not any(call[:2] == ["-m", "pip"] for call in calls)
    assert "GPU environment ready" not in result.stdout
