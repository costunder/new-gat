"""Orchestration contracts with stubbed subprocesses; never research training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_conductance_factorial as runner


def test_default_matrix_is_two_datasets_four_conditions_one_seed(tmp_path):
    args = runner.parser().parse_args([])
    jobs = runner.make_jobs(args, tmp_path)
    assert args.datasets == ["ppi", "ogbn-arxiv"] and args.model_seed == 0
    assert args.workers == 4
    assert len(jobs) == 8
    assert [job["condition"] for job in jobs[:4]] == list(runner.CONDITIONS)
    for job in jobs:
        command = job["command"]
        assert command[command.index("--model-seed") + 1] == "0"
        assert command[command.index("-m") + 1] == "research.conductance_gat.ablation.train"
        assert "--allow-download" not in command and "--amp" not in command
        expected_workers = 4 if job["dataset"] == "ppi" else 0
        assert job["workers"] == expected_workers
        assert command[command.index("--workers") + 1] == str(expected_workers)
        assert Path(job["metrics_path"]).parent == Path(job["output_dir"])
    assert len({job["output_dir"] for job in jobs}) == 8


def test_single_seed_override_and_common_budget_apply_to_every_arm(tmp_path):
    args = runner.parser().parse_args(
        [
            "--model-seed",
            "7",
            "--datasets",
            "ppi",
            "--epochs",
            "120",
            "--patience",
            "30",
            "--batch-size",
            "3",
            "--workers",
            "2",
        ]
    )
    jobs = runner.make_jobs(args, tmp_path)
    assert len(jobs) == 4
    for job in jobs:
        command = job["command"]
        for option, value in (
            ("--model-seed", "7"),
            ("--epochs", "120"),
            ("--patience", "30"),
            ("--batch-size", "3"),
            ("--workers", "2"),
        ):
            assert command[command.index(option) + 1] == value


def test_transductive_children_ignore_requested_ppi_worker_pool(tmp_path):
    args = runner.parser().parse_args(["--datasets", "ogbn-arxiv", "--workers", "7"])
    jobs = runner.make_jobs(args, tmp_path)
    assert {job["workers"] for job in jobs} == {0}
    assert {job["command"][job["command"].index("--workers") + 1] for job in jobs} == {"0"}


def test_child_environment_unsets_nvml_based_cuda_check(monkeypatch):
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    environment = runner._environment()
    assert "PYTORCH_NVML_BASED_CUDA_CHECK" not in environment


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_source_snapshot_decodes_non_ascii_git_stderr_without_reader_thread_error(monkeypatch):
    # Git on Windows can mix UTF-8 repository paths and a localized system
    # diagnostic in stderr. This synthetic failed child does not run research.
    diagnostic = "fatal: repository 프로젝트\n".encode() + "권한 진단\n".encode("cp949")
    command_code = (
        f"import sys; sys.stderr.buffer.write(bytes.fromhex('{diagnostic.hex()}')); "
        "raise RuntimeError('synthetic git failure')"
    )
    real_run = subprocess.run
    captured = []

    def failed_git(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        captured.append(kwargs)
        return real_run([sys.executable, "-c", command_code], **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", failed_git)
    snapshot = runner._source_snapshot()
    assert snapshot["git_revision"] is None
    assert len(captured) == 1
    assert captured[0]["encoding"] == "utf-8"
    assert captured[0]["errors"] == "replace"
    assert captured[0]["check"] is True
    source_path = Path(runner.__file__)
    relative_path = source_path.relative_to(runner.ROOT).as_posix()
    assert snapshot["sha256"][relative_path] == hashlib.sha256(source_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("arguments", [["--help"], ["--dry-run"]])
def test_inspection_requires_only_stdlib_and_writes_nothing(tmp_path, arguments):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(runner.ROOT / "scripts/run_conductance_factorial.py"),
            "--results-root",
            str(tmp_path),
            *arguments,
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
        ["--patience", "0"],
        ["--workers", "-1"],
        ["--datasets", "ppi", "ppi"],
        ["--run-id", "../old"],
        ["--run-id", "/tmp/old"],
        ["--min-free-gb", "nan"],
    ],
)
def test_invalid_inputs_stop_before_dependency_check(monkeypatch, options):
    def forbidden():
        pytest.fail("dependencies should not be checked")

    monkeypatch.setattr(runner, "check_dependencies", forbidden)
    assert runner.main(options) == 2


def test_existing_run_is_untouched(tmp_path, monkeypatch):
    output = tmp_path / "conductance_gat/ablations/existing"
    output.mkdir(parents=True)
    sentinel = output / "existing.pt"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(runner, "check_dependencies", lambda: pytest.fail("no dependency check"))
    assert runner.main(["--results-root", str(tmp_path), "--run-id", "existing"]) == 2
    assert sentinel.read_bytes() == b"preserve"


def test_dataset_directory_cannot_contain_outputs(tmp_path):
    assert (
        runner.main(
            [
                "--results-root",
                str(tmp_path),
                "--data-root",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 2
    )


def _stub_runtime(monkeypatch, tmp_path, *, failed_child=None, source_changes=False):
    calls = []
    comparisons = []
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"profile": "unit-stub"})
    snapshots = 0

    def snapshot():
        nonlocal snapshots
        snapshots += 1
        return {"sha256": {"source.py": "changed" if source_changes and snapshots > 1 else "same"}}

    def logged(command, log, environment):
        calls.append(command)
        assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
        if "gpu_preflight.py" in " ".join(command):
            assert "--json-out" in command
            return 0
        condition = command[command.index("--condition") + 1]
        if condition == failed_child:
            return 9
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "metrics.json").write_text("{}", encoding="utf-8")
        return 0

    def comparison(run_dir, manifest):
        comparisons.append(json.loads(json.dumps(manifest)))
        (run_dir / "comparison.md").write_text("unit-stub report\n", encoding="utf-8")
        return {"status": manifest["status"]}

    monkeypatch.setattr(runner, "_source_snapshot", snapshot)
    monkeypatch.setattr(runner, "run_logged", logged)
    monkeypatch.setattr(runner, "_comparison", comparison)
    options = ["--results-root", str(tmp_path), "--run-id", "unit-contract", "--datasets", "ppi"]
    return options, calls, comparisons


def test_success_records_all_four_children_and_final_report(tmp_path, monkeypatch):
    options, calls, comparisons = _stub_runtime(monkeypatch, tmp_path)
    assert runner.main(options) == 0
    assert len(calls) == 5  # One preflight, four independent processes.
    final = comparisons[-1]
    assert final["status"] == "passed"
    assert all(job["status"] == "passed" for job in final["jobs"])
    assert final["config"]["model_seed"] == 0
    assert final["config"]["hidden_channels"] == 64
    assert final["protocol"]["test"].startswith("not evaluated")


def test_failed_child_stops_and_preserves_completed_arm(tmp_path, monkeypatch):
    options, calls, comparisons = _stub_runtime(monkeypatch, tmp_path, failed_child="gate_no_wd")
    assert runner.main(options) == 1
    assert len(calls) == 3
    final = comparisons[-1]
    assert final["status"] == "failed"
    assert [job["status"] for job in final["jobs"]] == ["passed", "failed", "pending", "pending"]
    assert Path(final["jobs"][0]["metrics_path"]).is_file()
    assert "exit code 9" in final["error"]


def test_changing_sources_prevents_mixed_revision_training(tmp_path, monkeypatch):
    options, calls, comparisons = _stub_runtime(monkeypatch, tmp_path, source_changes=True)
    assert runner.main(options) == 1
    assert len(calls) == 1  # Only preflight; no condition may run.
    assert "source changed" in comparisons[-1]["error"]
    assert comparisons[-1]["source_integrity_valid"] is False


def test_source_change_after_last_child_marks_comparison_invalid(tmp_path, monkeypatch):
    options, calls, comparisons = _stub_runtime(monkeypatch, tmp_path)
    snapshots = 0

    def snapshot():
        nonlocal snapshots
        snapshots += 1
        return {"sha256": {"source.py": "changed" if snapshots == 6 else "same"}}

    monkeypatch.setattr(runner, "_source_snapshot", snapshot)
    assert runner.main(options) == 1
    assert len(calls) == 5
    assert comparisons[-1]["source_integrity_valid"] is False
    assert comparisons[-1]["status"] == "failed"


def test_preflight_failure_stops_all_training(tmp_path, monkeypatch):
    options, _, comparisons = _stub_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "run_logged", lambda *_: 3)
    assert runner.main(options) == 1
    assert all(job["status"] == "pending" for job in comparisons[-1]["jobs"])


def test_integrity_error_cannot_be_marked_success(tmp_path, monkeypatch):
    options, _, _ = _stub_runtime(monkeypatch, tmp_path)

    def mismatch(run_dir, manifest):
        if any(job["status"] == "passed" for job in manifest["jobs"]):
            raise ValueError("initial state mismatch")
        return {}

    monkeypatch.setattr(runner, "_comparison", mismatch)
    assert runner.main(options) == 1
    manifest = json.loads(
        (tmp_path / "conductance_gat/ablations/unit-contract/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert "initial state mismatch" in manifest["error"]


def test_wrapper_uses_active_conda_and_independent_runner():
    wrapper = (runner.ROOT / "research/conductance_gat/ablation/reproduce.sh").read_text()
    assert 'source "${project_root}/scripts/conda_env.sh"' in wrapper
    assert '"${environment_python}" -B scripts/run_conductance_factorial.py "$@"' in wrapper
    assert "set -" not in wrapper and "exec " not in wrapper and "exit " not in wrapper
    assert "main()" in wrapper and "must be executed, not sourced" in wrapper
    assert "setup_gpu.sh" not in wrapper and "run_paper.py" not in wrapper
