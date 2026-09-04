"""Audited termination of an exact subprocess created by the current runner."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

LogTarget = TextIO | Path


def run_failure_reporter(
    reporter: Callable[[], object],
    *,
    original_error: BaseException,
    action: str,
) -> str | None:
    """Run failure reporting without replacing an error already in flight.

    Failure-state persistence and summary generation are secondary operations.  A
    disk, serializer, or report error must be visible, but it must not turn the
    original scientific/child-process failure (or its intended return code) into
    an unrelated exception.
    """

    try:
        reporter()
    except BaseException as reporting_error:
        note = (
            f"{action} failed without replacing the original error: "
            f"{type(reporting_error).__name__}: {reporting_error}"
        )
        original_error.add_note(note)
        print(note, file=sys.stderr, flush=True)
        return note
    return None


def _write_event(event: dict[str, object], log_target: LogTarget) -> None:
    rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
    print(rendered, file=sys.stderr, flush=True)
    try:
        if isinstance(log_target, Path):
            with log_target.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered + "\n")
                stream.flush()
        else:
            log_target.write(rendered + "\n")
            log_target.flush()
    except (OSError, ValueError) as error:
        print(
            f"child termination audit could not be appended to its log: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )


def terminate_owned_child(
    process: subprocess.Popen[str],
    command: Sequence[str],
    *,
    reason: str,
    log_target: LogTarget,
    timeout_seconds: float = 10.0,
) -> list[dict[str, object]]:
    """Stop one exact live child after recording PID, argv, action, and reason."""

    if isinstance(command, (str, bytes)):
        raise TypeError("owned child command must be an argv sequence, not a shell string")
    exact_command = [str(argument) for argument in command]
    if not exact_command:
        raise ValueError("owned child command cannot be empty")
    actual = process.args
    if isinstance(actual, (str, bytes)):
        raise RuntimeError("refusing to signal a child created from a shell command string")
    if [str(argument) for argument in actual] != exact_command:
        raise RuntimeError("refusing to signal a process whose Popen args do not match the command")
    pid = process.pid
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("refusing to signal a child without a verified positive integer PID")
    if process.poll() is not None:
        return []

    events: list[dict[str, object]] = []
    terminate_event: dict[str, object] = {
        "event": "owned_child_signal",
        "pid": pid,
        "command": exact_command,
        "signal": "terminate",
        "reason": reason,
    }
    _write_event(terminate_event, log_target)
    events.append(terminate_event)
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            kill_event: dict[str, object] = {
                "event": "owned_child_signal",
                "pid": pid,
                "command": exact_command,
                "signal": "kill",
                "reason": (
                    f"{reason}; child remained alive for {timeout_seconds:g} seconds "
                    "after terminate"
                ),
            }
            _write_event(kill_event, log_target)
            events.append(kill_event)
            process.kill()
            process.wait(timeout=timeout_seconds)
    return events


def terminate_owned_child_after_error(
    process: subprocess.Popen[str],
    command: Sequence[str],
    *,
    original_error: BaseException,
    log_target: LogTarget,
    timeout_seconds: float = 10.0,
) -> list[dict[str, object]]:
    """Terminate one owned child without allowing cleanup to mask ``original_error``."""

    reason = f"{type(original_error).__name__}: {original_error}"
    try:
        return terminate_owned_child(
            process,
            command,
            reason=reason,
            log_target=log_target,
            timeout_seconds=timeout_seconds,
        )
    except BaseException as cleanup_error:
        original_error.add_note(
            "owned-child termination failed without replacing the original error: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
        event: dict[str, object] = {
            "event": "owned_child_termination_failure",
            "pid": getattr(process, "pid", None),
            "command": [str(argument) for argument in command],
            "reason": reason,
            "cleanup_error": f"{type(cleanup_error).__name__}: {cleanup_error}",
        }
        try:
            _write_event(event, log_target)
        except BaseException as reporting_error:
            original_error.add_note(
                "owned-child termination failure could not be logged: "
                f"{type(reporting_error).__name__}: {reporting_error}"
            )
        return [event]


def close_owned_child_stdout(
    process: subprocess.Popen[str], *, original_error: BaseException | None
) -> None:
    """Close one child pipe without replacing an error already in flight."""

    stream = process.stdout
    if stream is None:
        return
    try:
        stream.close()
    except BaseException as cleanup_error:
        if original_error is None:
            raise
        original_error.add_note(
            "child stdout cleanup failed without replacing the original error: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


__all__ = [
    "close_owned_child_stdout",
    "run_failure_reporter",
    "terminate_owned_child",
    "terminate_owned_child_after_error",
]
