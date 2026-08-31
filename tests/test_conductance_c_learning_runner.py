"""One-seed C-learning orchestration, with no GPU training in these contract tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_conductance_c_learning as runner


def test_exact_default_plan():
    args = runner.parser().parse_args([])
    jobs = runner.make_jobs(args, Path("fixture"))
    assert args.model_seed == 0 and args.datasets == ["ppi", "ogbn-arxiv"]
    assert len(jobs) == 4
    assert [job["condition"] for job in jobs] == ["learned_c", "fixed_c"] * 2
    for job in jobs:
        command = job["command"]
        assert command[command.index("-m") + 1] == "research.conductance_gat.c_learning.train"
        assert command[command.index("--model-seed") + 1] == "0"
        assert "--amp" not in command and "--allow-download" not in command


@pytest.mark.parametrize("option", ["--help", "--dry-run"])
def test_inspection_stdlib_only_no_writes(tmp_path, option):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(runner.ROOT / "scripts/run_conductance_c_learning.py"),
            option,
            "--results-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "options",
    [
        ["--device", "cpu"],
        ["--model-seed", "-1"],
        ["--epochs", "0"],
        ["--datasets", "ppi", "ppi"],
        ["--run-id", "../old"],
        ["--min-free-gb", "nan"],
    ],
)
def test_bad_input_no_install_or_training(monkeypatch, options):
    monkeypatch.setattr(runner, "check_dependencies", lambda: pytest.fail("must not check"))
    assert runner.main(options) == 2


def _stub(tmp_path, monkeypatch, failure=None, change_after=None):
    calls, reports = [], []
    snapshots = 0
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"unit_fixture_only": True})

    def snapshot():
        nonlocal snapshots
        snapshots += 1
        changed = change_after is not None and snapshots >= change_after
        return {"sha256": {"unit-source": "changed" if changed else "original"}}

    def dispatch(command, log, environment):
        calls.append(command)
        if any(Path(part).name == "gpu_preflight.py" for part in command):
            return 2 if failure == "preflight" else 0
        condition = command[command.index("--condition") + 1]
        if failure == condition:
            return 9
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "metrics.json").write_text("{}", encoding="utf-8")
        return 0

    def report(root, manifest):
        reports.append(json.loads(json.dumps(manifest)))
        (root / "comparison.md").write_text("unit-fixture report", encoding="utf-8")
        return {"status": manifest["status"]}

    monkeypatch.setattr(runner, "_source_snapshot", snapshot)
    monkeypatch.setattr(runner, "run_logged", dispatch)
    monkeypatch.setattr(runner, "_comparison", report)
    options = ["--datasets", "ppi", "--results-root", str(tmp_path), "--run-id", "unit-fixture"]
    return options, calls, reports


def test_two_fresh_processes_success_and_one_seed(tmp_path, monkeypatch):
    options, calls, reports = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0 and len(calls) == 3
    assert reports[-1]["status"] == "passed"
    assert reports[-1]["config"]["model_seed"] == 0
    assert [job["status"] for job in reports[-1]["jobs"]] == ["passed", "passed"]
    assert "never reuse" in reports[-1]["protocol"]["contrast"]


@pytest.mark.parametrize("failure,expected_calls", [("fixed_c", 3), ("preflight", 1)])
def test_failure_no_following_training(tmp_path, monkeypatch, failure, expected_calls):
    options, calls, reports = _stub(tmp_path, monkeypatch, failure=failure)
    assert runner.main(options) == 1 and len(calls) == expected_calls
    assert reports[-1]["status"] == "failed"
    if failure == "fixed_c":
        assert [job["status"] for job in reports[-1]["jobs"]] == ["passed", "failed"]


@pytest.mark.parametrize("change_after,calls_expected", [(2, 1), (4, 3)])
def test_source_changes_before_or_after_training_invalid(
    tmp_path, monkeypatch, change_after, calls_expected
):
    options, calls, reports = _stub(tmp_path, monkeypatch, change_after=change_after)
    assert runner.main(options) == 1
    assert len(calls) == calls_expected
    assert reports[-1]["source_integrity_valid"] is False


def test_existing_run_untouched(tmp_path, monkeypatch):
    root = tmp_path / "conductance_gat/c_learning/existing"
    root.mkdir(parents=True)
    sentinel = root / "best.pt"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(runner, "check_dependencies", lambda: pytest.fail("no install"))
    assert runner.main(["--run-id", "existing", "--results-root", str(tmp_path)]) == 2
    assert sentinel.read_bytes() == b"preserve"


def test_outputs_outside_data(tmp_path):
    assert (
        runner.main(["--results-root", str(tmp_path), "--data-root", str(tmp_path), "--dry-run"])
        == 2
    )


def test_source_snapshot_covers_shared_and_new_execution_code():
    snapshot = runner._source_snapshot()["sha256"]
    for name in (
        "research/conductance_gat/c_learning/model.py",
        "research/conductance_gat/c_learning/train.py",
        "research/conductance_gat/ablation/train.py",
        "research/conductance_gat/ablation/model.py",
        "scripts/run_conductance_c_learning.py",
        "scripts/run_conductance_factorial.py",
        "src/chartgat/cache.py",
    ):
        assert name in snapshot and len(snapshot[name]) == 64
