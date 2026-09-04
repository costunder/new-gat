"""Crash-safe helpers for publishing immutable dataset-cache files."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


class CacheValidationError(RuntimeError):
    """Base class for a cache that exists but is not usable."""


class CacheIncompleteError(CacheValidationError):
    """A multi-file cache is only partially present."""


class CacheWrongRequestError(CacheValidationError):
    """A cache belongs to a different seed, profile, schema, or source."""


class CacheCorruptError(CacheValidationError):
    """A cache cannot be parsed or fails its integrity contract."""


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after a rename.

    Windows does not expose a portable directory ``fsync``.  The file itself is
    always synced; on platforms supporting directory descriptors, the rename is
    synced as well.
    """

    if os.name == "nt":
        # Windows has no portable directory descriptor/fsync operation. This is
        # a known platform capability, not a swallowed operation failure.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    except BaseException as original_error:
        try:
            os.close(descriptor)
        except OSError as cleanup_error:
            original_error.add_note(
                "directory descriptor cleanup failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    else:
        os.close(descriptor)


def atomic_publish(
    path: Path,
    writer: Callable[[Path], None],
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Write, sync, optionally validate, and atomically publish one file.

    The temporary file is unique and located beside the destination, ensuring
    that ``os.replace`` cannot cross filesystems.  A failed writer or validator
    leaves the previous destination untouched.
    """

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        # Windows requires a writable descriptor for ``FlushFileBuffers``
        # (which backs ``os.fsync``); the writer has already finished here.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if validator is not None:
            validator(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException as original_error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            original_error.add_note(
                "temporary cache cleanup failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    else:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Atomically publish bytes after an optional read-only validation."""

    def write(temporary: Path) -> None:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()

    atomic_publish(path, write, validator=validator)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = True,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Serialize JSON deterministically and atomically publish it."""

    content = (
        json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, content, validator=validator)


__all__ = [
    "CacheCorruptError",
    "CacheIncompleteError",
    "CacheValidationError",
    "CacheWrongRequestError",
    "atomic_publish",
    "atomic_write_bytes",
    "atomic_write_json",
]
