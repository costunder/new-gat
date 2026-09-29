"""Small CPU-only tests for log streaming, failure visibility and read-only monitoring."""

import json
import subprocess
import sys
from pathlib import Path

from . import progress


def epoch_record():
    return {
        "condition": "learned",
        "epoch": 7,
        "loss": 0.8,
        "validation": {"accuracy": 0.75},
        "seconds_including_inspection_validation": 120.0,
        "large_diagnostics": [123] * 50,
    }


def test_partial_log_preserves_complete_records_and_failures(tmp_path):
    path = tmp_path / "child.log"
    encoded = json.dumps(epoch_record())
    path.write_text(encoded[:30])
    tail = progress.LogTail(path)
    assert tail.read() == []
    with path.open("a") as handle:
        handle.write(encoded[30:] + "\nRuntimeError: CUDA unavailable")
    lines = tail.read()
    assert len(lines) == 1 and "epoch 7 complete" in lines[0]
    assert "75.0000%" in lines[0] and "large_diagnostics" not in lines[0]
    assert tail.read(final=True) == ["RuntimeError: CUDA unavailable"]
    assert tail.read(final=True) == []


def test_child_reports_heartbeat_and_nonzero_error(tmp_path, capsys):
    path = tmp_path / "child.log"
    code = (
        "import time; print('[progress] preparing', flush=True); "
        "time.sleep(0.3); raise RuntimeError('debug child failure')"
    )
    with path.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", code], stdout=handle, stderr=subprocess.STDOUT
        )
        result = progress.wait_with_progress(
            child, path, "debug", heartbeat_seconds=0.05, poll_seconds=0.02
        )
    output = capsys.readouterr().out
    assert result != 0
    assert "preparing" in output
    assert "process alive" in output
    assert "RuntimeError: debug child failure" in output
    assert "Traceback" in path.read_text()


def test_watch_existing_run_follows_phase_change_without_writes(tmp_path, monkeypatch, capsys):
    status = tmp_path / "status.json"
    log = tmp_path / "train-baseline.log"
    log.write_text(json.dumps(epoch_record()) + "\n")
    status.write_text(json.dumps({"state": "running", "phase": "train-baseline"}))
    seen = []

    def advance(_):
        seen.append(1)
        assert len(seen) == 1
        status.write_text(json.dumps({"state": "complete", "completed": ["train-baseline"]}))

    monkeypatch.setattr(progress.time, "sleep", advance)
    assert progress.watch_run(tmp_path) == 0
    output = capsys.readouterr().out
    assert "epoch 7 complete" in output and "state=complete" in output
    assert log.read_text() == json.dumps(epoch_record()) + "\n"
    assert set(tmp_path.iterdir()) == {status, log}


def test_monitor_runs_as_standalone_without_training_imports(tmp_path):
    standalone = tmp_path / "monitor.py"
    standalone.write_bytes(Path(progress.__file__).read_bytes())
    (tmp_path / "status.json").write_text(
        json.dumps({"state": "failed", "error": "debug recorded failure"})
    )
    result = subprocess.run(
        [sys.executable, "-I", str(standalone), "--run-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and "debug recorded failure" in result.stdout


def test_progress_metrics_do_not_dump_calibration_payload():
    row = {
        "condition": "fixed",
        "physical_seed_batch": 4096,
        "context_workers": 4,
        "supervised_nodes_per_second": 500.0,
        "paired_timings": [{"large": "payload"}],
    }
    message = progress.summarize(json.dumps(row))
    assert "batch 4096 workers 4" in message and "payload" not in message
