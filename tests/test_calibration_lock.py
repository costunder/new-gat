"""Local filesystem/process lock tests; never launch training or signal a process."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.calibration_lock import LOCK_FILENAME, ROOT, calibration_lock


def _child(source: str, output: Path):
    return subprocess.run(
        [sys.executable, "-B", "-c", source, str(output)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


TRY_LOCK = """
import sys
from pathlib import Path
from scripts.calibration_lock import calibration_lock
try:
    with calibration_lock(Path(sys.argv[1])):
        print('acquired', flush=True)
except RuntimeError as error:
    print('blocked: ' + str(error), flush=True)
"""


def test_lock_is_exclusive_across_independent_processes_and_released_without_deletion(tmp_path):
    output = tmp_path / "calibration"
    with calibration_lock(output) as actual:
        assert actual == output.resolve()
        contested = _child(TRY_LOCK, output)
        assert contested.returncode == 0, contested.stderr
        assert contested.stdout.startswith("blocked: another calibration is already active")
        if os.name != "nt":
            assert (output / LOCK_FILENAME).read_bytes() == b"0"
    sentinel = (output / LOCK_FILENAME).read_bytes()
    assert sentinel == b"0"
    accepted = _child(TRY_LOCK, output)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == "acquired"
    assert (output / LOCK_FILENAME).read_bytes() == sentinel


def test_independent_holder_releases_on_normal_process_completion(tmp_path):
    output = tmp_path / "calibration"
    source = """
import sys
from pathlib import Path
from scripts.calibration_lock import calibration_lock
with calibration_lock(Path(sys.argv[1])):
    print('holding', flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", source, str(output)], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "holding"
        attempt = _child(TRY_LOCK, output)
        assert attempt.returncode == 0, attempt.stderr
        assert attempt.stdout.startswith("blocked:")
    finally:
        # The child completes its own read and context; no PID management or signals.
        _, errors = process.communicate("release\n")
    assert process.returncode == 0, errors
    with calibration_lock(output):
        pass


def test_failed_child_process_does_not_leave_stale_lock_ownership(tmp_path):
    output = tmp_path / "calibration"
    source = """
import sys
from pathlib import Path
from scripts.calibration_lock import calibration_lock
guard = calibration_lock(Path(sys.argv[1]))
guard.__enter__()
raise RuntimeError('intentional local process test failure')
"""
    completed = _child(source, output)
    assert completed.returncode != 0
    assert "intentional local process test failure" in completed.stderr
    assert (output / LOCK_FILENAME).is_file()
    with calibration_lock(output):
        pass


def test_lock_body_failure_releases_ownership_and_preserves_progress_bytes(tmp_path):
    output = tmp_path / "calibration"
    output.mkdir()
    progress = output / "progress.json"
    progress.write_bytes(b'{"existing": "untouched"}\n')
    with pytest.raises(RuntimeError, match="body failure"):
        with calibration_lock(output):
            raise RuntimeError("body failure")
    with calibration_lock(output):
        pass
    assert progress.read_bytes() == b'{"existing": "untouched"}\n'


@pytest.mark.parametrize("target", [ROOT, Path.home(), Path(Path.cwd().anchor)])
def test_broad_roots_are_refused(target):
    with pytest.raises(ValueError, match="broad root"):
        with calibration_lock(target):
            raise AssertionError("unsafe context must never start")


def test_parent_traversal_is_refused(tmp_path):
    with pytest.raises(ValueError, match="parent traversal"):
        with calibration_lock(tmp_path / "child" / ".." / "other"):
            raise AssertionError("unsafe context must never start")


@pytest.mark.parametrize("content", [b"not a lock file", b"x"])
def test_existing_unexpected_lock_file_is_preserved(tmp_path, content):
    output = tmp_path / "calibration"
    output.mkdir()
    lock = output / LOCK_FILENAME
    lock.write_bytes(content)
    with pytest.raises(ValueError, match="unexpected existing contents"):
        with calibration_lock(output):
            raise AssertionError("unexpected file must not be overwritten")
    assert lock.read_bytes() == content


def test_lock_directory_is_not_opened_as_a_regular_file(tmp_path):
    output = tmp_path / "calibration"
    (output / LOCK_FILENAME).mkdir(parents=True)
    with pytest.raises(ValueError, match="regular"):
        with calibration_lock(output):
            raise AssertionError("special target must not be opened")


def test_hardlinked_lock_is_refused_without_modifying_original(tmp_path):
    output = tmp_path / "calibration"
    output.mkdir()
    original = tmp_path / "original"
    original.write_bytes(b"0")
    os.link(original, output / LOCK_FILENAME)
    with pytest.raises(ValueError, match="hard-linked"):
        with calibration_lock(output):
            raise AssertionError("linked user data must remain untouched")
    assert original.read_bytes() == b"0"


def test_symlink_output_is_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"creating a symlink is not available: {error}")
    with pytest.raises(ValueError, match="symlink/reparse"):
        with calibration_lock(link / "output"):
            raise AssertionError("indirect output must never be created")
    assert not (target / "output").exists()


def test_reparse_directory_metadata_is_refused(tmp_path, monkeypatch):
    output = tmp_path / "calibration"
    original = Path.lstat

    def metadata(path, *args, **kwargs):
        if path == output:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", metadata)
    with pytest.raises(ValueError, match="symlink/reparse"):
        with calibration_lock(output):
            raise AssertionError("reparse output must not be followed")
