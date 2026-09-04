from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.process_safety import (
    close_owned_child_stdout,
    run_failure_reporter,
    terminate_owned_child,
    terminate_owned_child_after_error,
)


class _OwnedProcess:
    def __init__(self, command: list[str], log: io.StringIO) -> None:
        self.args = command
        self.pid = 43127
        self._alive = True
        self._waits = 0
        self.actions: list[str] = []
        self.log = log

    def poll(self) -> int | None:
        return None if self._alive else -9

    def terminate(self) -> None:
        assert '"signal": "terminate"' in self.log.getvalue()
        self.actions.append("terminate")

    def wait(self, timeout: float | None = None) -> int:
        self._waits += 1
        if self._waits == 1:
            raise subprocess.TimeoutExpired(self.args, timeout)
        self._alive = False
        return -9

    def kill(self) -> None:
        assert '"signal": "kill"' in self.log.getvalue()
        self.actions.append("kill")


def test_owned_child_is_identified_and_reported_before_each_signal(capsys) -> None:
    command = ["python", "worker.py", "--run-id", "exact value"]
    log = io.StringIO()
    process = _OwnedProcess(command, log)

    events = terminate_owned_child(
        process,  # type: ignore[arg-type]
        command,
        reason="KeyboardInterrupt: requested by operator",
        log_target=log,
    )

    assert process.actions == ["terminate", "kill"]
    assert [event["signal"] for event in events] == ["terminate", "kill"]
    for event in events:
        assert event["pid"] == 43127
        assert event["command"] == command
        assert "KeyboardInterrupt" in str(event["reason"])
    log_events = [json.loads(line) for line in log.getvalue().splitlines()]
    stderr_events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert log_events == events == stderr_events


def test_owned_child_helper_refuses_mismatched_command_without_signal() -> None:
    log = io.StringIO()
    process = _OwnedProcess(["python", "actual.py"], log)
    with pytest.raises(RuntimeError, match="Popen args do not match"):
        terminate_owned_child(
            process,  # type: ignore[arg-type]
            ["python", "different.py"],
            reason="unit test",
            log_target=log,
        )
    assert process.actions == [] and log.getvalue() == ""


def test_owned_child_helper_refuses_shell_string_args_without_signal() -> None:
    log = io.StringIO()
    process = _OwnedProcess(["python", "actual.py"], log)
    process.args = "python actual.py"
    with pytest.raises(RuntimeError, match="shell command string"):
        terminate_owned_child(
            process,  # type: ignore[arg-type]
            ["python", "actual.py"],
            reason="unit test",
            log_target=log,
        )
    assert process.actions == [] and log.getvalue() == ""


@pytest.mark.parametrize("pid", [None, 0, -1, True, "43127"])
def test_owned_child_helper_refuses_unverified_pid_without_signal(pid: object) -> None:
    log = io.StringIO()
    process = _OwnedProcess(["python", "actual.py"], log)
    process.pid = pid
    with pytest.raises(RuntimeError, match="positive integer PID"):
        terminate_owned_child(
            process,  # type: ignore[arg-type]
            ["python", "actual.py"],
            reason="unit test",
            log_target=log,
        )
    assert process.actions == [] and log.getvalue() == ""


def test_exited_owned_child_is_not_signalled_or_logged(tmp_path: Path) -> None:
    command = ["python", "done.py"]
    log = io.StringIO()
    process = _OwnedProcess(command, log)
    process._alive = False
    assert (
        terminate_owned_child(
            process,  # type: ignore[arg-type]
            command,
            reason="late interruption",
            log_target=tmp_path / "unused.log",
        )
        == []
    )
    assert process.actions == [] and not (tmp_path / "unused.log").exists()


def test_cleanup_verification_failure_never_replaces_original_error() -> None:
    log = io.StringIO()
    process = _OwnedProcess(["python", "actual.py"], log)
    original = KeyboardInterrupt("operator interruption")

    events = terminate_owned_child_after_error(
        process,  # type: ignore[arg-type]
        ["python", "different.py"],
        original_error=original,
        log_target=log,
    )

    assert process.actions == []
    assert events[0]["event"] == "owned_child_termination_failure"
    assert "Popen args do not match" in str(events[0]["cleanup_error"])
    assert any(
        "termination failed without replacing" in note
        for note in original.__notes__
    )


def test_stdout_close_failure_preserves_error_already_in_flight() -> None:
    class BrokenStream:
        def close(self) -> None:
            raise OSError("pipe close failed")

    process = SimpleNamespace(stdout=BrokenStream())
    original = RuntimeError("primary runner failure")

    close_owned_child_stdout(process, original_error=original)

    assert any("pipe close failed" in note for note in original.__notes__)
    with pytest.raises(OSError, match="pipe close failed"):
        close_owned_child_stdout(process, original_error=None)


def test_failure_reporter_preserves_primary_error_and_reports_secondary(capsys) -> None:
    original = RuntimeError("scientific failure")

    def broken_reporter() -> None:
        raise OSError("manifest storage unavailable")

    note = run_failure_reporter(
        broken_reporter,
        original_error=original,
        action="failure manifest persistence",
    )

    assert note is not None
    assert "manifest storage unavailable" in note
    assert note in original.__notes__
    assert note in capsys.readouterr().err


def test_failure_reporter_returns_none_after_success() -> None:
    calls: list[str] = []
    original = RuntimeError("scientific failure")

    assert (
        run_failure_reporter(
            lambda: calls.append("reported"),
            original_error=original,
            action="failure manifest persistence",
        )
        is None
    )
    assert calls == ["reported"]
    assert not hasattr(original, "__notes__")


def test_failure_reporter_does_not_replace_keyboard_interrupt() -> None:
    original = KeyboardInterrupt("operator interrupt")

    note = run_failure_reporter(
        lambda: (_ for _ in ()).throw(OSError("report disk error")),
        original_error=original,
        action="interruption manifest persistence",
    )

    assert note is not None
    assert "report disk error" in note
    assert note in original.__notes__


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/run_conductance_factorial.py",
        "scripts/run_conductance_c_learning.py",
        "scripts/run_conductance_v2.py",
        "scripts/run_conductance_v3.py",
        "scripts/run_conductance_v4.py",
        "scripts/run_conductance_v5.py",
        "scripts/run_conductance_scaling.py",
        "scripts/run_tree_scaling.py",
        "scripts/run_rich_scaling.py",
    ],
)
def test_failure_path_callers_use_non_masking_reporter(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / relative_path).read_text(encoding="utf-8")

    assert "run_failure_reporter(" in source
