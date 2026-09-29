"""Console progress and a standalone, read-only monitor for existing studies."""

import argparse
import json
import subprocess
import time
from pathlib import Path

PREFIX = "[progress]"


def announce(message):
    print(f"{PREFIX} {message}", flush=True)


def summarize(line):
    """Keep complete JSON in the log; show only research progress on screen."""
    if not line.strip():
        return None
    if line.startswith(PREFIX):
        return line.rstrip()
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return line.rstrip()  # Warnings and tracebacks remain visible.
    if not isinstance(row, dict):
        return None
    if "epoch" in row and "validation" in row and "loss" in row:
        return (
            f"{row['condition']} epoch {row['epoch']} complete | "
            f"train CE {row['loss']:.5f} | "
            f"val accuracy {row['validation']['accuracy']:.4%} | "
            f"epoch {row['seconds_including_inspection_validation']:.1f}s"
        )
    if "physical_seed_batch" in row and "paired_timings" in row:
        return (
            f"calibration complete: {row['condition']} | "
            f"batch {row['physical_seed_batch']} workers {row['context_workers']} | "
            f"{row['supervised_nodes_per_second']:.1f} supervised nodes/s"
        )
    return None


class LogTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = ""

    def read(self, final=False):
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        parts = (self.pending + chunk).split("\n")
        self.pending = parts.pop()
        if final and self.pending:
            parts.append(self.pending)
            self.pending = ""
        return [message for line in parts if (message := summarize(line)) is not None]


def wait_with_progress(child, log_path, label, *, heartbeat_seconds=30, poll_seconds=1):
    """Child writes its full log directly; reading it never blocks its stdout."""
    tail = LogTail(log_path)
    started = last_heartbeat = time.monotonic()
    while True:
        for message in tail.read():
            announce(f"{label} | {message}")
        code = child.poll()
        if code is not None:
            for message in tail.read(final=True):
                announce(f"{label} | {message}")
            return code
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_seconds:
            announce(
                f"{label} | process alive, elapsed {now - started:.0f}s | "
                f"PID {child.pid}; waiting for next progress record"
            )
            last_heartbeat = now
        try:
            child.wait(timeout=poll_seconds)
        except subprocess.TimeoutExpired:
            continue


def watch_run(root, poll_seconds=2):
    """Observe saved files only. Never import training code or signal its processes."""
    status_path = root / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"study status not found: {status_path}")
    tails = {}
    previous = None
    last_heartbeat = 0.0
    while True:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # The runner rewrites status in place. Retry only this partial-write race.
            time.sleep(poll_seconds)
            continue
        phase = status.get("phase")
        state = status["state"]
        identity = (state, phase)
        if identity != previous:
            announce(
                f"state={state} phase={phase or '-'} | completed={status.get('completed', [])}"
            )
            previous = identity
        for path in sorted(root.glob("*.log")):
            new = path not in tails
            tail = tails.setdefault(path, LogTail(path))
            messages = tail.read(final=state in {"failed", "complete"})
            # Joining an existing run prints the latest record from each earlier log.
            for message in messages[-1:] if new else messages:
                announce(f"{path.stem} | {message}")
        if state in {"failed", "complete", "stopped"}:
            if status.get("error"):
                announce(status["error"])
            return 1 if state == "failed" else 0
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            folder = root / phase if phase else root
            records = list(folder.glob("*.json"))
            if records:
                latest = max(records, key=lambda path: path.stat().st_mtime)
                age = max(0, time.time() - latest.stat().st_mtime)
                announce(f"{phase} | latest saved record: {latest.name} ({age:.0f}s ago)")
            else:
                announce(f"{phase} | no saved result yet; waiting for next record")
            last_heartbeat = now
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return watch_run(args.run_dir.resolve())
    except KeyboardInterrupt:
        print("Monitor stopped; the training process was not changed.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
