from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_all.py"


def test_master_runner_dry_run_lists_every_independent_track(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--run-id",
            "portable-dry-run",
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "research.conductance_gat.run" in result.stdout
    assert "research.cycle_pe.run" in result.stdout
    assert "research.tree_augmentation.run" in result.stdout
    assert "combined_later" not in result.stdout
    assert "paper_benchmark" not in result.stdout


def test_master_runner_rejects_unsafe_run_id(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", "--run-id", "../escape"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "run id" in result.stderr.lower()
