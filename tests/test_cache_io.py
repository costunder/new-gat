from __future__ import annotations

from pathlib import Path

import pytest

from chartgat import cache
from chartgat.cache import atomic_publish, atomic_write_bytes


def test_atomic_write_preserves_previous_file_when_validation_fails(tmp_path: Path) -> None:
    destination = tmp_path / "cache.bin"
    destination.write_bytes(b"previous-valid-cache")

    def reject(temporary: Path) -> None:
        assert temporary.read_bytes() == b"new-invalid-cache"
        raise ValueError("invalid temporary cache")

    with pytest.raises(ValueError, match="invalid temporary cache"):
        atomic_write_bytes(destination, b"new-invalid-cache", validator=reject)

    assert destination.read_bytes() == b"previous-valid-cache"
    assert list(tmp_path.glob(".cache.bin.*.tmp")) == []


def test_atomic_write_validates_then_replaces(tmp_path: Path) -> None:
    destination = tmp_path / "cache.bin"
    destination.write_bytes(b"old")
    atomic_write_bytes(
        destination,
        b"new",
        validator=lambda temporary: temporary.read_bytes() == b"new" or None,
    )
    assert destination.read_bytes() == b"new"


def test_directory_fsync_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache.os, "name", "posix")
    monkeypatch.setattr(cache.os, "open", lambda _path, _flags: 123)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("directory sync unavailable")

    monkeypatch.setattr(cache.os, "fsync", fail_fsync)
    monkeypatch.setattr(cache.os, "close", lambda _descriptor: None)

    with pytest.raises(OSError, match="directory sync unavailable"):
        cache._fsync_directory(tmp_path)


def test_temporary_cleanup_error_does_not_replace_writer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cache.bin"
    original_error = ValueError("writer failed")

    def fail_writer(_temporary: Path) -> None:
        raise original_error

    def fail_unlink(_path: Path, *, missing_ok: bool = False) -> None:
        assert missing_ok is True
        raise OSError("cleanup failed")

    monkeypatch.setattr(cache.Path, "unlink", fail_unlink)

    with pytest.raises(ValueError) as caught:
        atomic_publish(destination, fail_writer)

    assert caught.value is original_error
    assert any("temporary cache cleanup failed" in note for note in original_error.__notes__)
