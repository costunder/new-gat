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
BASH_ENTRYPOINTS = ("setup_gpu.sh", "paper.sh")
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


def test_paper_checks_preparation_dependencies_only_after_conda_validation() -> None:
    source = (ROOT / "scripts" / "paper.sh").read_text(encoding="utf-8")
    guard = 'source "${project_root}/scripts/conda_env.sh"'
    dependency_check = (
        '"${environment_python}" "${project_root}/scripts/check_dependencies.py" --quiet'
    )
    installer = 'bash "${project_root}/scripts/setup_gpu.sh"'
    profile_query = (
        '"${environment_python}" "${project_root}/scripts/gpu_profiles.py" --installed-profile'
    )
    assert source.index(guard) < source.index(dependency_check)
    assert source.count(dependency_check) == 2
    assert source.count(installer) == 1
    assert source.index(dependency_check) < source.index(profile_query) < source.index(installer)
    assert f'{installer} --profile "${{bootstrap_profile}}"' in source
    assert (
        source.index(dependency_check) < source.index(installer) < source.rindex(dependency_check)
    )
    assert source.rindex(dependency_check) < source.index(
        '"${environment_python}" scripts/run_paper.py'
    )
    assert "--prepare-only) prepare_only=1" in source
    assert "--help|-h|--dry-run) inspection_only=1" in source
    assert '"${prepare_only}" == "1" && "${inspection_only}" == "0"' in source
    assert f"{dependency_check} || dependency_status=$?" in source
    assert f"if ! {dependency_check}" not in source
    assert 'case "${dependency_status}" in' in source
    assert 'exit "${dependency_status}"' in source
    assert source.index("        2)") < source.index(installer) < source.index("        *)")
    assert "-m pip" not in source


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
        "  */check_dependencies.py)\n"
        '    printf "dependencies\\n" >> "$TEST_CALL_LOG"\n'
        '    printf "%s\\0" "$0" "$@" >> "$TEST_DEPENDENCY_ARGS"\n'
        '    if [ -f "$TEST_DEPENDENCY_READY" ]; then\n'
        '      exit "${TEST_DEPENDENCY_AFTER_SETUP_EXIT:-0}"\n'
        "    fi\n"
        '    exit "${TEST_DEPENDENCY_EXIT:-0}" ;;\n'
        "  */gpu_profiles.py)\n"
        '    printf "profile\\n" >> "$TEST_CALL_LOG"\n'
        '    if [ "$2" != "--installed-profile" ]; then exit 98; fi\n'
        '    if [ "${TEST_PROFILE_QUERY_EXIT:-0}" != "0" ]; then\n'
        '      exit "$TEST_PROFILE_QUERY_EXIT"\n'
        "    fi\n"
        '    printf "%s\\n" "${TEST_INSTALLED_PROFILE:-auto}" ;;\n'
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
            "TEST_DEPENDENCY_EXIT": "0",
            "TEST_DEPENDENCY_AFTER_SETUP_EXIT": "0",
            "TEST_DEPENDENCY_ARGS": str(tmp_path / "dependency.args"),
            "TEST_DEPENDENCY_READY": str(tmp_path / "dependencies.ready"),
            "TEST_SETUP_ARGS": str(tmp_path / "setup.args"),
            "TEST_INSTALLED_PROFILE": "auto",
            "TEST_PROFILE_QUERY_EXIT": "0",
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
    # Setup help deliberately exits before requiring an active environment.
    arguments = [] if script_name == "setup_gpu.sh" else ["--help"]
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / script_name), *arguments],
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
    arguments = [] if script_name == "setup_gpu.sh" else ["--help"]
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / script_name), *arguments],
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


@LINUX_BASH_ONLY
@pytest.mark.parametrize(
    ("script", "defaults"),
    [
        ("scripts/prepare_data.sh", ["--suite", "benchmark", "--prepare-only", "--allow-download"]),
        ("scripts/reproduce.sh", ["--suite", "benchmark"]),
        (
            "research/conductance_gat/reproduce.sh",
            ["--suite", "benchmark", "--tracks", "conductance_gat"],
        ),
        ("research/cycle_pe/reproduce.sh", ["--suite", "benchmark", "--tracks", "cycle_pe"]),
        (
            "research/tree_augmentation/reproduce.sh",
            ["--suite", "benchmark", "--tracks", "tree_augmentation"],
        ),
    ],
)
def test_reproduction_scripts_forward_defaults_arguments_and_exit_status(
    tmp_path: Path, script: str, defaults: list[str]
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    environ["TEST_RUN_EXIT"] = "37"
    arguments = ["--run-id", "space value", "--model-seeds", "1,2"]
    result = subprocess.run(
        [BASH, str(ROOT / script), *arguments],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 37, result.stderr
    expected_calls = ["verify"]
    if "--prepare-only" in defaults:
        expected_calls.append("dependencies")
    expected_calls.append("paper")
    assert call_log.read_text(encoding="utf-8").splitlines() == expected_calls
    forwarded = dispatch_args.read_bytes().split(b"\0")[:-1]
    assert [value.decode("utf-8") for value in forwarded] == [
        "scripts/run_paper.py",
        *defaults,
        *arguments,
    ]


def _bootstrap_project(tmp_path: Path) -> Path:
    """Copy real dispatch/guard scripts, replacing only installation with a stub."""

    project = tmp_path / "Research project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for filename in ("paper.sh", "conda_env.sh", "prepare_data.sh", "gpu_profiles.py"):
        shutil.copy2(ROOT / "scripts" / filename, scripts / filename)
    (scripts / "setup_gpu.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "setup\\n" >> "$TEST_CALL_LOG"\n'
        'printf "%s" "$CONDA_PREFIX" > "$TEST_SETUP_PREFIX"\n'
        'printf "%s\\0" "$@" > "$TEST_SETUP_ARGS"\n'
        'if [[ "${TEST_SETUP_EXIT:-0}" != "0" ]]; then exit "$TEST_SETUP_EXIT"; fi\n'
        'if [[ "${TEST_SETUP_MARK_READY:-1}" == "1" ]]; then\n'
        '    touch "$TEST_DEPENDENCY_READY"\n'
        "fi\n",
        encoding="utf-8",
    )
    return project


@LINUX_BASH_ONLY
@pytest.mark.parametrize("initial_exit", [0, 2])
def test_prepare_bootstraps_only_missing_dependencies_and_rechecks_same_python(
    tmp_path: Path, initial_exit: int
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    project = _bootstrap_project(tmp_path)
    setup_prefix = tmp_path / "setup.prefix"
    environ.update(
        {
            "TEST_DEPENDENCY_EXIT": str(initial_exit),
            "TEST_SETUP_PREFIX": str(setup_prefix),
            "TEST_RUN_EXIT": "37",
        }
    )
    arguments = ["--run-id", "space value", "", "literal;$HOME"]
    result = subprocess.run(
        [BASH, str(project / "scripts" / "prepare_data.sh"), *arguments],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 37, result.stderr
    expected = ["verify", "dependencies"]
    if initial_exit:
        expected.extend(["profile", "setup", "dependencies"])
        assert setup_prefix.read_text(encoding="utf-8") == environ["CONDA_PREFIX"]
        assert Path(environ["TEST_SETUP_ARGS"]).read_bytes().split(b"\0")[:-1] == [
            b"--profile",
            b"auto",
        ]
        assert "Installing the complete locked GPU environment" in result.stdout
    else:
        assert not setup_prefix.exists()
    assert call_log.read_text(encoding="utf-8").splitlines() == [*expected, "paper"]
    dependency_arguments = Path(environ["TEST_DEPENDENCY_ARGS"]).read_bytes().split(b"\0")[:-1]
    expected_dependency_call = [
        str(Path(environ["CONDA_PREFIX"]) / "bin" / "python"),
        str(project / "scripts" / "check_dependencies.py"),
        "--quiet",
    ]
    assert [value.decode("utf-8") for value in dependency_arguments] == (
        expected_dependency_call * (2 if initial_exit else 1)
    )
    forwarded = dispatch_args.read_bytes().split(b"\0")[:-1]
    assert [value.decode("utf-8") for value in forwarded] == [
        "scripts/run_paper.py",
        "--suite",
        "benchmark",
        "--prepare-only",
        "--allow-download",
        *arguments,
    ]


@LINUX_BASH_ONLY
@pytest.mark.parametrize("initial_exit", [3, 41, 126, 127])
def test_prepare_does_not_install_or_dispatch_after_host_or_unexpected_dependency_error(
    tmp_path: Path, initial_exit: int
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    project = _bootstrap_project(tmp_path)
    setup_prefix = tmp_path / "setup.prefix"
    environ.update(
        {"TEST_DEPENDENCY_EXIT": str(initial_exit), "TEST_SETUP_PREFIX": str(setup_prefix)}
    )
    result = subprocess.run(
        [BASH, str(project / "scripts" / "prepare_data.sh")],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == initial_exit, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["verify", "dependencies"]
    assert not setup_prefix.exists()
    assert not Path(environ["TEST_DEPENDENCY_READY"]).exists()
    assert not dispatch_args.exists()
    assert "Installing the complete locked GPU environment" not in result.stdout


@LINUX_BASH_ONLY
@pytest.mark.parametrize(
    "setup_exit,mark_ready,recheck_exit,expected_exit",
    [(19, "1", 0, 19), (0, "0", 0, 2), (0, "1", 3, 3), (0, "1", 41, 41)],
)
def test_prepare_never_dispatches_after_failed_install_or_dependency_recheck(
    tmp_path: Path, setup_exit: int, mark_ready: str, recheck_exit: int, expected_exit: int
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    project = _bootstrap_project(tmp_path)
    environ.update(
        {
            "TEST_DEPENDENCY_EXIT": "2",
            "TEST_DEPENDENCY_AFTER_SETUP_EXIT": str(recheck_exit),
            "TEST_SETUP_EXIT": str(setup_exit),
            "TEST_SETUP_MARK_READY": mark_ready,
            "TEST_SETUP_PREFIX": str(tmp_path / "setup.prefix"),
        }
    )
    result = subprocess.run(
        [BASH, str(project / "scripts" / "prepare_data.sh")],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == expected_exit, result.stderr
    expected = ["verify", "dependencies", "profile", "setup"]
    if setup_exit == 0:
        expected.append("dependencies")
    assert call_log.read_text(encoding="utf-8").splitlines() == expected
    assert not dispatch_args.exists()


@LINUX_BASH_ONLY
@pytest.mark.parametrize("profile_id", ["legacy-cu118", "cu118", "cu126", "cu130", "cu132"])
def test_dependency_bootstrap_forwards_the_exact_installed_profile(
    tmp_path: Path, profile_id: str
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    project = _bootstrap_project(tmp_path)
    environ.update(
        {
            "TEST_DEPENDENCY_EXIT": "2",
            "TEST_INSTALLED_PROFILE": profile_id,
            "TEST_SETUP_PREFIX": str(tmp_path / "setup.prefix"),
        }
    )
    result = subprocess.run(
        [BASH, str(project / "scripts" / "prepare_data.sh")],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    forwarded = Path(environ["TEST_SETUP_ARGS"]).read_bytes().split(b"\0")[:-1]
    assert [value.decode("utf-8") for value in forwarded] == ["--profile", profile_id]
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "verify",
        "dependencies",
        "profile",
        "setup",
        "dependencies",
        "paper",
    ]
    assert dispatch_args.exists()


@LINUX_BASH_ONLY
@pytest.mark.parametrize("query_exit", [2, 41])
def test_dependency_bootstrap_never_installs_when_profile_identity_is_unknown(
    tmp_path: Path, query_exit: int
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    project = _bootstrap_project(tmp_path)
    environ.update({"TEST_DEPENDENCY_EXIT": "2", "TEST_PROFILE_QUERY_EXIT": str(query_exit)})
    result = subprocess.run(
        [BASH, str(project / "scripts" / "prepare_data.sh")],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == query_exit, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "verify",
        "dependencies",
        "profile",
    ]
    assert not Path(environ["TEST_SETUP_ARGS"]).exists()
    assert not dispatch_args.exists()


@LINUX_BASH_ONLY
@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--help"],
        ["--dry-run"],
        ["--prepare-only", "--help"],
        ["--prepare-only", "-h"],
        ["--prepare-only", "--dry-run"],
        ["--help", "--prepare-only"],
    ],
)
def test_training_help_and_dry_run_never_bootstrap_dependencies(
    tmp_path: Path, arguments: list[str]
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    project = _bootstrap_project(tmp_path)
    environ.update(
        {
            "TEST_DEPENDENCY_EXIT": "2",
            "TEST_SETUP_PREFIX": str(tmp_path / "setup.prefix"),
        }
    )
    result = subprocess.run(
        [BASH, str(project / "scripts" / "paper.sh"), *arguments],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["verify", "paper"]
    assert not Path(environ["TEST_DEPENDENCY_ARGS"]).exists()
    assert not Path(environ["TEST_SETUP_PREFIX"]).exists()
    forwarded = dispatch_args.read_bytes().split(b"\0")[:-1]
    assert [value.decode("utf-8") for value in forwarded] == [
        "scripts/run_paper.py",
        *arguments,
    ]
