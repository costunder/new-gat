"""V2 orchestration contracts; mocked subprocesses, never GPU or research training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_conductance_v2 as runner


def test_default_two_fresh_trainings():
    args = runner.parser().parse_args([])
    jobs = runner.make_jobs(args, Path("fixture"))
    assert args.datasets == ["ogbn-arxiv"] and args.model_seed == 0
    assert args.edge_chunk_size == 65536 and args.batch_size == 1 and args.workers == 0
    assert [job["condition"] for job in jobs] == ["direct_c", "fixed_c"]
    for job in jobs:
        command = job["command"]
        assert command[command.index("-m") + 1] == "research.conductance_gat.v2.train"
        assert command[command.index("--model-seed") + 1] == "0"
        assert command[command.index("--edge-chunk-size") + 1] == "65536"
        assert "--amp" not in command and "--allow-download" not in command


@pytest.mark.parametrize("option", ["--help", "--dry-run"])
def test_stdlib_inspection_has_no_writes(tmp_path, option):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            str(runner.ROOT / "scripts/run_conductance_v2.py"),
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
        ["--datasets", "ppi"],
        ["--datasets", "unknown"],
        ["--datasets", "cora", "cora"],
        ["--run-id", "../old"],
        ["--min-free-gb", "nan"],
        ["--edge-chunk-size", "0"],
        ["--batch-size", "2"],
        ["--workers", "1"],
    ],
)
def test_invalid_inputs_do_not_check_dependencies_or_train(monkeypatch, options):
    monkeypatch.setattr(runner, "check_dependencies", lambda: pytest.fail("no install/check"))
    assert runner.main(options) == 2


def test_ppi_rejection_explains_fixed_graph_limit(capsys):
    assert runner.main(["--datasets", "ppi"]) == 2
    assert "held-out PPI graphs" in capsys.readouterr().err


def _stub(tmp_path, monkeypatch, failure=None, change_after=None):
    calls, reports, snapshots = [], [], 0
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
        if failure != "missing_metrics":
            (output / "metrics.json").write_text("{}", encoding="utf-8")
        return 0

    def report(root, manifest):
        reports.append(json.loads(json.dumps(manifest)))
        (root / "comparison.md").write_text("unit-fixture report", encoding="utf-8")
        return {"status": manifest["status"]}

    monkeypatch.setattr(runner, "_source_snapshot", snapshot)
    monkeypatch.setattr(runner, "run_logged", dispatch)
    monkeypatch.setattr(runner, "_comparison", report)
    return ["--results-root", str(tmp_path), "--run-id", "unit-fixture"], calls, reports


def test_success_records_metrics_digest_and_one_seed(tmp_path, monkeypatch):
    options, calls, reports = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0 and len(calls) == 3
    assert reports[-1]["status"] == "passed"
    assert reports[-1]["config"]["model_seed"] == 0
    assert [job["status"] for job in reports[-1]["jobs"]] == ["passed", "passed"]
    assert all(len(job["metrics_sha256"]) == 64 for job in reports[-1]["jobs"])
    assert "never reuse" in reports[-1]["protocol"]["contrast"]


@pytest.mark.parametrize(
    "failure,calls_expected",
    [
        ("preflight", 1),
        ("direct_c", 2),
        ("fixed_c", 3),
        ("missing_metrics", 2),
    ],
)
def test_failures_stop_following_trainings(tmp_path, monkeypatch, failure, calls_expected):
    options, calls, reports = _stub(tmp_path, monkeypatch, failure=failure)
    assert runner.main(options) == 1 and len(calls) == calls_expected
    assert reports[-1]["status"] == "failed"


@pytest.mark.parametrize("change_after,calls_expected", [(2, 1), (3, 2), (4, 2), (5, 3), (6, 3)])
def test_source_checks_before_after_each_child_and_final(
    tmp_path, monkeypatch, change_after, calls_expected
):
    options, calls, reports = _stub(tmp_path, monkeypatch, change_after=change_after)
    assert runner.main(options) == 1 and len(calls) == calls_expected
    assert reports[-1]["source_integrity_valid"] is False


def test_existing_run_untouched(tmp_path, monkeypatch):
    root = tmp_path / "conductance_gat/v2/existing"
    root.mkdir(parents=True)
    sentinel = root / "best.pt"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(runner, "check_dependencies", lambda: pytest.fail("no install"))
    assert runner.main(["--run-id", "existing", "--results-root", str(tmp_path)]) == 2
    assert sentinel.read_bytes() == b"preserve"


def test_outputs_do_not_overlap_data(tmp_path):
    assert (
        runner.main(["--results-root", str(tmp_path), "--data-root", str(tmp_path), "--dry-run"])
        == 2
    )


def test_source_snapshot_covers_shared_and_v2_code():
    snapshot = runner._source_snapshot()["sha256"]
    for name in (
        "research/conductance_gat/v2/model.py",
        "research/conductance_gat/v2/train.py",
        "research/conductance_gat/v2/report.py",
        "research/conductance_gat/ablation/train.py",
        "scripts/run_conductance_v2.py",
        "scripts/run_conductance_factorial.py",
        "src/chartgat/cache.py",
        "research/conductance_gat/v2/reproduce.sh",
    ):
        assert name in snapshot and len(snapshot[name]) == 64
