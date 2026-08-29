from __future__ import annotations

from pathlib import Path

import pytest

from chartgat.cache import atomic_write_bytes


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
