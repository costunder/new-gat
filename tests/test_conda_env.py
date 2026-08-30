from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_conda_env

ROOT = Path(__file__).resolve().parents[1]
BASH_ENTRYPOINTS = ("setup.sh", "setup_gpu.sh", "paper.sh", "smoke.sh")
BASH = shutil.which("bash")
LINUX_BASH_ONLY = pytest.mark.skipif(
    sys.platform != "linux" or BASH is None,
    reason="Dynamic shell contracts require Linux and Bash; unavailable on this local host.",
)


@pytest.fixture
def active_conda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    prefix = tmp_path / "Conda environments" / "new gat"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    python = prefix / "bin" / "python"
    python.touch()
    monkeypatch.setattr(
        verify_conda_env,
        "sys",
        SimpleNamespace(
            version_info=(3, 11, 9),
            prefix=str(prefix),
            base_prefix=str(prefix),
            executable=str(python),
        ),
    )
    monkeypatch.setattr(verify_conda_env, "_conda_base", lambda _environ: tmp_path / "base")
    return prefix, {"CONDA_PREFIX": str(prefix), "CONDA_DEFAULT_ENV": "new-gat"}


@pytest.mark.parametrize("environ", [{}, {"CONDA_PREFIX": ""}])
def test_missing_conda_prefix_is_rejected(environ: dict[str, str]) -> None:
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="No active Conda"):
        verify_conda_env.verify_conda_environment(environ)


def test_named_base_environment_is_rejected(active_conda: tuple[Path, dict[str, str]]) -> None:
    _, environ = active_conda
    environ["CONDA_DEFAULT_ENV"] = "base"
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Do not use Conda base"):
        verify_conda_env.verify_conda_environment(environ)


@pytest.mark.parametrize("default_env", [None, "not-named-base"])
def test_base_prefix_is_rejected_even_without_base_name(
    active_conda: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    default_env: str | None,
) -> None:
    prefix, environ = active_conda
    if default_env is None:
        environ.pop("CONDA_DEFAULT_ENV")
    else:
        environ["CONDA_DEFAULT_ENV"] = default_env
    monkeypatch.setattr(verify_conda_env, "_conda_base", lambda _environ: prefix.resolve())
    with pytest.raises(
        verify_conda_env.CondaEnvironmentError, match="Do not install into Conda base"
    ):
        verify_conda_env.verify_conda_environment(environ)


def test_virtual_env_is_rejected(active_conda: tuple[Path, dict[str, str]]) -> None:
    prefix, environ = active_conda
    environ["VIRTUAL_ENV"] = str(prefix / "nested")
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="venv is still active"):
        verify_conda_env.verify_conda_environment(environ)


def test_missing_conda_metadata_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="conda-meta is missing"):
        verify_conda_env.verify_conda_environment({"CONDA_PREFIX": str(tmp_path)})


@pytest.mark.parametrize("version", [(3, 9, 20), (3, 10, 15)])
def test_old_python_is_rejected(
    active_conda: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int, int],
) -> None:
    _, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "version_info", version)
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Python 3.11 or newer"):
        verify_conda_env.verify_conda_environment(environ)


def test_other_interpreter_prefix_is_rejected(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "prefix", str(prefix / "other"))
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="sys.prefix does not match"):
        verify_conda_env.verify_conda_environment(environ)


def test_nested_venv_is_rejected_without_virtual_env_variable(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "base_prefix", str(prefix / "other-base"))
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Nested venv Python"):
        verify_conda_env.verify_conda_environment(environ)


def test_other_executable_is_rejected(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "executable", str(prefix / "other" / "python"))
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="not CONDA_PREFIX/bin/python"):
        verify_conda_env.verify_conda_environment(environ)


def test_valid_conda_prefix_with_spaces_and_normalized_suffix(
    active_conda: tuple[Path, dict[str, str]],
) -> None:
    prefix, environ = active_conda
    environ["CONDA_PREFIX"] = f"{prefix}/."
    assert verify_conda_env.verify_conda_environment(environ) == prefix.resolve()


def test_verifier_defaults_to_process_environment(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.os, "environ", environ)
    assert verify_conda_env.verify_conda_environment() == prefix.resolve()


@pytest.mark.parametrize("conda_exe", [None, "/Conda installation/bin/conda"])
def test_conda_base_queries_configured_executable_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conda_exe: str | None
) -> None:
    environ = {} if conda_exe is None else {"CONDA_EXE": conda_exe}
    base = tmp_path / "Conda installation"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"  {base}\n")

    monkeypatch.setattr(verify_conda_env.subprocess, "run", fake_run)
    assert verify_conda_env._conda_base(environ) == base.resolve()
    assert calls == [
        (
            [conda_exe or "conda", "info", "--base"],
            {"check": True, "capture_output": True, "text": True, "timeout": 30},
        )
    ]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("conda missing"),
        PermissionError("conda not executable"),
        subprocess.CalledProcessError(1, ["conda", "info", "--base"]),
        subprocess.TimeoutExpired(["conda", "info", "--base"], 30),
    ],
)
def test_conda_base_query_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(verify_conda_env.subprocess, "run", fail_run)
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Cannot query") as caught:
        verify_conda_env._conda_base({})
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("stdout", ["", " \n\t"])
def test_conda_base_rejects_empty_result(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    monkeypatch.setattr(
        verify_conda_env.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout)
    )
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="empty path"):
        verify_conda_env._conda_base({})


@pytest.mark.parametrize("script_name", BASH_ENTRYPOINTS)
def test_bash_entrypoints_validate_conda_before_installation_or_dispatch(script_name: str) -> None:
    source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
    guard = 'source "${project_root}/scripts/conda_env.sh"'
    assert source.count(guard) == 1
    assert source.index("project_root=") < source.index(guard)
    assert source.index(guard) < source.index('"${environment_python}"')
    for forbidden in (".venv", "VENV_DIR", "USE_ACTIVE_ENV", "-m venv", "environment_python="):
        assert forbidden not in source
    assert not re.search(r"\$\{?PYTHON(?=[:}\s\"/]|$)", source)
    assert not re.search(r"^\s*conda\s+(?:create|install)\b", source, flags=re.MULTILINE)
    if script_name == "setup_gpu.sh":
        assert source.index(guard) < source.index("command -v nvidia-smi")
        assert source.index(guard) < source.index('mkdir -p "${snapshot_dir}"')


def test_shared_bash_guard_uses_only_conda_python_and_runs_verification() -> None:
    source = (ROOT / "scripts" / "conda_env.sh").read_text(encoding="utf-8")
    assert 'environment_python="${CONDA_PREFIX%/}/bin/python"' in source
    assert '[[ ! -x "${environment_python}" ]]' in source
    assert '"${environment_python}" "${project_root}/scripts/verify_conda_env.py"' in source
    for forbidden in (".venv", "VENV_DIR", "USE_ACTIVE_ENV", "-m pip", "-m venv"):
        assert forbidden not in source
    assert not re.search(r"^\s*conda\s+(?:create|install)\b", source, flags=re.MULTILINE)


def _shell_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Stub only dispatch; Python unit tests above validate real guard decisions."""

    prefix = tmp_path / "Conda environments" / "new gat"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  */verify_conda_env.py)\n"
        '    printf "verify\\n" >> "$TEST_CALL_LOG"\n'
        '    exit "${TEST_VERIFY_EXIT:-0}" ;;\n'
        "  scripts/run_paper.py)\n"
        '    printf "paper\\n" >> "$TEST_CALL_LOG"\n'
        '    printf "%s\\0" "$@" > "$TEST_DISPATCH_ARGS"\n'
        '    exit "${TEST_RUN_EXIT:-0}" ;;\n'
        '  *) printf "unexpected\\n" >> "$TEST_CALL_LOG"; exit 97 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    call_log = tmp_path / "calls.log"
    dispatch_args = tmp_path / "dispatch.args"
    environ = os.environ.copy()
    environ.pop("VIRTUAL_ENV", None)
    environ.update(
        {
            "CONDA_PREFIX": f"{prefix}/",
            "CONDA_DEFAULT_ENV": "new-gat",
            "TEST_CALL_LOG": str(call_log),
            "TEST_DISPATCH_ARGS": str(dispatch_args),
            "TEST_VERIFY_EXIT": "0",
            "TEST_RUN_EXIT": "0",
            "PYTHON": str(tmp_path / "wrong-python"),
            "VENV_DIR": str(tmp_path / "must-not-be-created"),
            "USE_ACTIVE_ENV": "0",
            "ENVIRONMENT_SNAPSHOT_DIR": str(tmp_path / "must-not-have-snapshots"),
        }
    )
    return environ, call_log, dispatch_args


@LINUX_BASH_ONLY
@pytest.mark.parametrize("script_name", BASH_ENTRYPOINTS)
@pytest.mark.parametrize("skip_deps", ["0", "1"])
def test_bash_guard_failure_stops_before_pip_or_dispatch(
    tmp_path: Path, script_name: str, skip_deps: str
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    environ.update({"TEST_VERIFY_EXIT": "23", "SKIP_DEPS": skip_deps})
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["verify"]
    assert not dispatch_args.exists()
    assert not Path(environ["VENV_DIR"]).exists()
    assert not Path(environ["ENVIRONMENT_SNAPSHOT_DIR"]).exists()


@LINUX_BASH_ONLY
@pytest.mark.parametrize("script_name", BASH_ENTRYPOINTS)
def test_bash_without_active_conda_stops_before_invoking_python(
    tmp_path: Path, script_name: str
) -> None:
    environ, call_log, _ = _shell_environment(tmp_path)
    environ.pop("CONDA_PREFIX")
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2
    assert "No active Conda environment" in result.stderr
    assert not call_log.exists()


@LINUX_BASH_ONLY
def test_paper_bash_preserves_arguments_exit_code_and_conda_selection(tmp_path: Path) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    environ["TEST_RUN_EXIT"] = "37"
    arguments = ["--run-id", "space value", "", "literal;$HOME", "--seeds", "1,2"]
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / "paper.sh"), *arguments],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 37, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["verify", "paper"]
    forwarded = dispatch_args.read_bytes().split(b"\0")[:-1]
    assert [value.decode("utf-8") for value in forwarded] == ["scripts/run_paper.py", *arguments]
    assert not Path(environ["VENV_DIR"]).exists()
