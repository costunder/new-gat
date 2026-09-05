"""Crash-released, process-exclusive ownership of one calibration output directory.

The persistent one-byte lock file is not a stale-PID marker. Only the operating
system's current file lock decides ownership, so process completion/crash releases
it without deleting anything or signalling any process.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILENAME = ".calibration.lock"
ROOT = Path(__file__).resolve().parents[1]


def _reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _direct_output(output: Path) -> Path:
    path = Path(output).expanduser()
    if ".." in path.parts:
        raise ValueError("calibration output must not contain parent traversal")
    path = Path(os.path.abspath(path))
    if path in {Path(path.anchor), Path.home().resolve(), ROOT}:
        raise ValueError("calibration output must be a dedicated directory, not a broad root")
    for component in (*reversed(path.parents), path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if _reparse(metadata):
            raise ValueError(f"calibration output symlink/reparse path is forbidden: {component}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"calibration output component is not a directory: {component}")
    if path.resolve() != path:
        raise ValueError("calibration output must resolve directly without indirection")
    return path


def _regular_lock(metadata: os.stat_result) -> None:
    if _reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("calibration lock must be a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise ValueError("calibration lock must not be hard-linked to another file")
    if metadata.st_size not in {0, 1}:
        raise ValueError("calibration lock has unexpected existing contents; file preserved")


def _acquire(descriptor: int, output: Path) -> None:
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            raise RuntimeError("calibration locking is supported only on POSIX and Windows")
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise RuntimeError(
                f"another calibration is already active for {output}; "
                "wait for that run to finish, then retry the same command. "
                "Do not delete the lock file or interrupt unrelated processes"
            ) from error
        raise


@contextmanager
def calibration_lock(output: Path) -> Iterator[Path]:
    """Own exactly one calibration directory until context/process completion.

    Call this before mutating progress/request artifacts. The yielded absolute
    path is the validated output directory. This context does not touch manifests,
    probes, checkpoints or other experiment directories.
    """
    directory = _direct_output(output)
    directory.mkdir(parents=True, exist_ok=True)
    directory = _direct_output(directory)
    lock_path = directory / LOCK_FILENAME
    try:
        existing = lock_path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        _regular_lock(existing)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    primary_error = None
    try:
        os.set_inheritable(descriptor, False)
        metadata = os.fstat(descriptor)
        _regular_lock(metadata)
        _direct_output(directory)
        current = lock_path.lstat()
        _regular_lock(current)
        if not os.path.samestat(metadata, current):
            raise ValueError("calibration lock path changed while opening it")
        _acquire(descriptor, directory)
        # Locking a byte past EOF is supported by both OS lock implementations.
        # Only the owner initializes the sentinel; a competing process writes nothing.
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
        elif os.read(descriptor, 1) != b"0":
            raise ValueError("calibration lock has unexpected existing contents; file preserved")
        yield directory
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            # Closing the exact owned descriptor releases either OS lock. We never
            # unlink it: unlinking a held lock could allow a second independent owner.
            os.close(descriptor)
        except OSError as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"closing calibration lock also failed: {close_error}")
