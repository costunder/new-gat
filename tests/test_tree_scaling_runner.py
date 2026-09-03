"""Tree scaling orchestration contracts; no GPU or research training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from research.tree_augmentation import paper as tree_paper
from scripts import run_tree_scaling as runner


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_matrix_trains_both_versions_across_larger_profiles() -> None:
    args = runner.parser().parse_args([])
    jobs = runner.make_jobs(args, Path("fixture"))
    assert [(job["suite"], job["profile"]) for job in jobs] == [
        ("csl", "reference"),
        ("csl", "large"),
        ("zinc", "reference"),
        ("zinc", "large"),
    ]
    assert args.suites == ("csl", "zinc")
    assert args.profiles == ("reference", "large")
    assert args.model_seeds == (0,)
    assert len(jobs) == 4
    assert sum(len(job["trained_models"]) for job in jobs) == 8
    assert len(args.suites) * len(runner.MODELS) == 4
    assert len(args.suites) * len(args.model_seeds) * len(runner.MODELS) == 4
    assert len({job["output_dir"] for job in jobs}) == len(jobs)
    assert runner.PROFILE_CONFIGS == {
        "reference": {
            "hidden_dim": 128,
            "message_layers": 8,
            "optimizer_updates": 800,
            "train_charts_per_graph": 8,
            "eval_charts_per_graph": 8,
        },
        "large": {
            "hidden_dim": 256,
            "message_layers": 12,
            "optimizer_updates": 800,
            "train_charts_per_graph": 8,
            "eval_charts_per_graph": 8,
        },
    }
    for job in jobs:
        assert job["trained_models"] == ["fixed_bfs", "multi_chart"]
        command = job["command"]
        assert command[command.index("-m") + 1] == "research.tree_augmentation.paper"
        for key, value in job["profile_config"].items():
            assert command[command.index("--" + key.replace("_", "-")) + 1] == str(value)


def test_a6000_profile_resolves_recorded_high_throughput_settings() -> None:
    args = runner.parser().parse_args(["--hardware-profile", "a6000-48gb"])
    runner._validate(args)
    assert (args.batch_size, args.workers, args.amp, args.job_concurrency) == (64, 4, True, 2)
    jobs = runner.make_jobs(args, Path("fixture"))
    assert [(job["suite"], job["profile"]) for job in jobs] == [
        ("zinc", "large"),
        ("zinc", "reference"),
        ("csl", "large"),
        ("csl", "reference"),
    ]
    assert all(job["command"][job["command"].index("--batch-size") + 1] == "64" for job in jobs)
    assert all(job["command"][job["command"].index("--workers") + 1] == "4" for job in jobs)
    assert all("--amp" in job["command"] for job in jobs)
    config = runner._run_config(args, Path("data").resolve())
    assert config["hardware_profile"] == "a6000-48gb"
    assert config["job_concurrency"] == 2


def test_profile_selection_is_independent_of_a6000_heavy_first_job_order() -> None:
    args = runner.parser().parse_args(["--hardware-profile", "a6000-48gb"])
    runner._validate(args)
    jobs = [job for job in runner.make_jobs(args, Path("fixture")) if job["suite"] == "csl"]
    for job in jobs:
        score = 0.8 if job["profile"] == "large" else 0.7
        job["status"] = "passed"
        job["result"] = {
            "selection_objectives": {
                model: {
                    "metric": "unit",
                    "direction": "maximize",
                    "value": score,
                }
                for model in runner.MODELS
            },
            "checkpoints": {
                model: {"path": f"{job['profile']}-{model}.pt", "sha256": "unit"}
                for model in runner.MODELS
            },
            "parameter_counts": {model: {"total": 1, "trainable": 1} for model in runner.MODELS},
            "quadrant_metrics": {model: {"unit": {"accuracy": score}} for model in runner.MODELS},
            "child_summary_sha256": "unit",
        }
    forward = runner._select_profiles(
        jobs, suite="csl", model_seeds=(0,), profiles=("reference", "large")
    )
    reverse = runner._select_profiles(
        list(reversed(jobs)),
        suite="csl",
        model_seeds=(0,),
        profiles=("reference", "large"),
    )
    assert forward == reverse
    assert all(
        condition["selected_profile"] == "large" for condition in forward["conditions"].values()
    )


def test_a6000_preflight_rejects_mig_and_old_compute_capability() -> None:
    base = {
        "status": "passed",
        "gpu": {
            "free_bytes": 47 * 1024**3,
            "total_bytes": 48 * 1024**3,
            "compute_capability": [8, 6],
        },
    }
    runner._validate_hardware_preflight(base, "a6000-48gb")
    mig = json.loads(json.dumps(base))
    mig["gpu"]["total_bytes"] = 10 * 1024**3
    with pytest.raises(RuntimeError, match="40 GiB"):
        runner._validate_hardware_preflight(mig, "a6000-48gb")
    busy = json.loads(json.dumps(base))
    busy["gpu"]["free_bytes"] = 31 * 1024**3
    with pytest.raises(RuntimeError, match="32 GiB free"):
        runner._validate_hardware_preflight(busy, "a6000-48gb")
    old = json.loads(json.dumps(base))
    old["gpu"]["compute_capability"] = [7, 5]
    with pytest.raises(RuntimeError, match="capability 8.0"):
        runner._validate_hardware_preflight(old, "a6000-48gb")


def test_bounded_wave_runs_independent_jobs_concurrently_with_single_manifest_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(2)
    worker_threads: list[int] = []
    writer_threads: list[int] = []
    main_thread = threading.get_ident()
    jobs = [
        {
            "status": "pending",
            "command": [f"job-{index}"],
            "log_path": str(tmp_path / f"job-{index}.log"),
        }
        for index in range(2)
    ]
    manifest = {"jobs": jobs, "sources": {}}

    def dispatch(_command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        worker_threads.append(threading.get_ident())
        barrier.wait(timeout=5)
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    monkeypatch.setattr(runner, "_check_sources", lambda _manifest: None)
    monkeypatch.setattr(
        runner,
        "_write_state",
        lambda _run_dir, _manifest: writer_threads.append(threading.get_ident()),
    )
    runner._execute_job_matrix(
        jobs,
        concurrency=2,
        environment={},
        manifest=manifest,
        run_dir=tmp_path,
        validator=lambda job: {"accepted": job["command"][0]},
        describe=lambda job: job["command"][0],
    )
    assert len(set(worker_threads)) == 2
    assert set(worker_threads) == set(worker_threads) - {main_thread}
    assert writer_threads and set(writer_threads) == {main_thread}
    assert [job["status"] for job in jobs] == ["passed", "passed"]


@pytest.mark.parametrize("failure_mode", ["nonzero", "exception"])
def test_concurrent_wave_preserves_successful_peer_and_retries_only_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:
    barrier = threading.Barrier(2)
    jobs = [
        {
            "status": "pending",
            "command": [f"job-{index}"],
            "log_path": str(tmp_path / f"job-{index}.log"),
        }
        for index in range(2)
    ]
    manifest = {"jobs": jobs, "sources": {}}

    def first_dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        barrier.wait(timeout=5)
        if command[0] == "job-1":
            if failure_mode == "exception":
                raise RuntimeError("worker failure")
            return 7
        return 0

    monkeypatch.setattr(runner, "_run_logged", first_dispatch)
    monkeypatch.setattr(runner, "_check_sources", lambda _manifest: None)
    monkeypatch.setattr(runner, "_write_state", lambda *_args: None)
    with pytest.raises(RuntimeError):
        runner._execute_job_matrix(
            jobs,
            concurrency=2,
            environment={},
            manifest=manifest,
            run_dir=tmp_path,
            validator=lambda job: {"accepted": job["command"][0]},
            describe=lambda job: job["command"][0],
        )
    assert jobs[0]["status"] == "passed"
    assert jobs[0]["result"] == {"accepted": "job-0"}
    assert jobs[1]["status"] == "failed"

    jobs[1]["status"] = "pending"
    resumed_calls: list[str] = []

    def resumed_dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        resumed_calls.append(command[0])
        return 0

    monkeypatch.setattr(runner, "_run_logged", resumed_dispatch)
    runner._execute_job_matrix(
        jobs,
        concurrency=2,
        environment={},
        manifest=manifest,
        run_dir=tmp_path,
        validator=lambda job: {"accepted": job["command"][0]},
        describe=lambda job: job["command"][0],
    )
    assert resumed_calls == ["job-1"]
    assert [job["status"] for job in jobs] == ["passed", "passed"]


def test_default_dry_run_reports_seed_zero_plan_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert runner.main(["--results-root", str(tmp_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "4 validation-candidate child runs" in output
    assert "8 fresh model trainings" in output
    assert "4 aggregate profile selections" in output
    assert "4 selected-checkpoint test evaluations" in output
    assert list(tmp_path.iterdir()) == []


def test_paper_scaling_overrides_are_opt_in_and_validated() -> None:
    args = tree_paper._parser().parse_args([])
    assert args.hidden_dim is None
    assert args.message_layers is None
    assert args.optimizer_updates is None
    assert args.train_charts_per_graph is None
    assert args.eval_charts_per_graph is None
    defaults = {
        "hidden_dim": 64,
        "message_layers": 2,
        "optimizer_updates": 800,
        "train_charts_per_graph": 8,
        "eval_charts_per_graph": 8,
    }
    effective, overrides = tree_paper._apply_setting_overrides(
        defaults,
        hidden_dim=256,
        message_layers=12,
        optimizer_updates=800,
        train_charts_per_graph=8,
        eval_charts_per_graph=8,
    )
    assert effective == runner.PROFILE_CONFIGS["large"]
    assert overrides == runner.PROFILE_CONFIGS["large"]
    unchanged, no_overrides = tree_paper._apply_setting_overrides(
        defaults,
        hidden_dim=None,
        message_layers=None,
        optimizer_updates=None,
        train_charts_per_graph=None,
        eval_charts_per_graph=None,
    )
    assert unchanged == defaults and no_overrides == {}
    with pytest.raises(ValueError, match="hidden dim must be positive"):
        tree_paper._apply_setting_overrides(
            defaults,
            hidden_dim=0,
            message_layers=None,
            optimizer_updates=None,
            train_charts_per_graph=None,
            eval_charts_per_graph=None,
        )


@pytest.mark.parametrize("option", ["--help", "--dry-run"])
def test_stdlib_inspection_has_no_writes(tmp_path: Path, option: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            str(runner.ROOT / "scripts/run_tree_scaling.py"),
            option,
            "--results-root",
            str(tmp_path),
            "--suites",
            "csl",
            "--profiles",
            "large",
            "--model-seeds",
            "0,1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
    if option == "--dry-run":
        assert "2 validation-candidate child runs; 4 fresh model trainings" in result.stdout


@pytest.mark.parametrize(
    "options",
    [
        ["--device", "cpu"],
        ["--batch-size", "0"],
        ["--workers", "-1"],
        ["--data-seed", "-1"],
        ["--min-free-gb", "nan"],
        ["--run-id", "../old"],
    ],
)
def test_invalid_inputs_do_not_check_dependencies(monkeypatch, options: list[str]) -> None:
    monkeypatch.setattr(runner, "check_dependencies", lambda: pytest.fail("dependency check"))
    assert runner.main(options) == 2


def _write_child(job: dict[str, object], *, malformed: str | None = None) -> None:
    command = job["command"]
    assert isinstance(command, list)
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    axes = {
        "data": int(command[command.index("--data-seed") + 1]),
        "split": int(command[command.index("--split-seed") + 1]),
        "chart": int(command[command.index("--chart-seed") + 1]),
        "model": int(job["model_seed"]),
    }
    profile = dict(job["profile_config"])
    parameter_count = 1000 + profile["hidden_dim"]
    parameters = {"total": parameter_count, "trainable": parameter_count}
    quadrants = {
        "validation_graph_fresh_chart_seen_family": {
            "accuracy": 0.7,
            "graph_macro_accuracy": 0.7,
            "mae": 0.2,
            "graph_macro_mae": 0.2,
        },
        "validation_graph_fresh_chart_unseen_family": {
            "accuracy": 0.6,
            "graph_macro_accuracy": 0.6,
            "mae": 0.3,
            "graph_macro_mae": 0.3,
        },
    }
    models = {
        name: {
            "optimizer_updates": profile["optimizer_updates"],
            "parameters": dict(parameters),
            "quadrants": json.loads(json.dumps(quadrants)),
        }
        for name in runner.MODELS
    }
    if malformed == "metric":
        models["multi_chart"]["quadrants"]["validation_graph_fresh_chart_seen_family"][
            "graph_macro_accuracy"
        ] = float("nan")
    checkpoints = {}
    for name in runner.MODELS:
        path = output / f"{name}_model.pt"
        path.write_bytes(f"unit-{name}".encode())
        checkpoints[name] = str(path.resolve())
    amp = "--no-amp" not in command
    runtime = {
        "batch_size": int(command[command.index("--batch-size") + 1]),
        "workers": int(command[command.index("--workers") + 1]),
        "amp_requested": amp,
        "amp_effective": amp,
        "elapsed_seconds": 1.25,
        "peak_gpu_allocated_bytes": 1_000_000,
        "peak_gpu_reserved_bytes": 2_000_000,
    }
    summary = {
        "suite": job["suite"],
        "seed_axes": axes,
        "evaluation_scope": "validation",
        "training_performed": True,
        "test_metrics_emitted": False,
        "dataset_cache_integrity": {
            "full_cache_loaded": True,
            "all_declared_splits_validated": True,
            "loaded_and_validated_splits": ["test", "train", "validation"],
        },
        "model_split_usage": {
            "fit_splits": ["train"],
            "evaluation_splits": ["validation"],
            "selection_splits": ["validation"],
            "test_evaluated": False,
            "test_used_for_selection": False,
        },
        "settings": profile,
        "parameter_counts": {name: dict(parameters) for name in runner.MODELS},
        "models": models,
        "comparison": {
            "paper_headline_eligible": False,
            "fixed_and_multi_optimizer_updates_matched": True,
        },
        "checkpoints": checkpoints,
        "runtime": runtime,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, allow_nan=True), encoding="utf-8")
    artifacts = {"summary.json": {"path": str(summary_path), "sha256": _digest(summary_path)}}
    for path in map(Path, checkpoints.values()):
        artifacts[path.name] = {"path": str(path), "sha256": _digest(path)}
    child_manifest = {
        "status": "passed",
        "suite": job["suite"],
        "seed_axes": axes,
        "evaluation_scope": "validation",
        "training_performed": True,
        "dataset_cache_integrity": summary["dataset_cache_integrity"],
        "model_split_usage": summary["model_split_usage"],
        "effective_settings": profile,
        "settings_overrides": profile,
        "runtime": runtime,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(child_manifest), encoding="utf-8")


def _write_selected_child(job: dict[str, object]) -> None:
    command = job["command"]
    assert isinstance(command, list)
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    axes = {
        "data": int(command[command.index("--data-seed") + 1]),
        "split": int(command[command.index("--split-seed") + 1]),
        "chart": int(command[command.index("--chart-seed") + 1]),
        "model": int(job["model_seed"]),
    }
    selected_inputs = job["selected_inputs"]
    expected_checkpoints = {name: selected_inputs[name]["checkpoint"] for name in runner.MODELS}
    quadrants = {
        "test_graph_fresh_chart_seen_family": {"accuracy": 0.7, "mae": 0.2},
        "test_graph_fresh_chart_unseen_family": {"accuracy": 0.6, "mae": 0.3},
    }
    models = {
        name: {
            "optimizer_updates": 800,
            "training_performed": False,
            "history": [],
            "parameters": selected_inputs[name]["parameter_counts"],
            "quadrants": quadrants,
        }
        for name in runner.MODELS
    }
    amp = "--no-amp" not in command
    runtime = {
        "batch_size": int(command[command.index("--batch-size") + 1]),
        "workers": int(command[command.index("--workers") + 1]),
        "amp_requested": amp,
        "amp_effective": amp,
        "elapsed_seconds": 0.5,
        "peak_gpu_allocated_bytes": 750_000,
        "peak_gpu_reserved_bytes": 1_500_000,
    }
    summary = {
        "suite": job["suite"],
        "seed_axes": axes,
        "evaluation_scope": "selected_test",
        "training_performed": False,
        "test_metrics_emitted": True,
        "dataset_cache_integrity": {
            "full_cache_loaded": True,
            "all_declared_splits_validated": True,
            "loaded_and_validated_splits": ["test", "train", "validation"],
        },
        "model_split_usage": {
            "fit_splits": [],
            "evaluation_splits": ["test"],
            "selection_splits": [],
            "test_evaluated": True,
            "test_used_for_selection": False,
        },
        "test_evaluations_per_selected_checkpoint": 1,
        "selected_checkpoints": expected_checkpoints,
        "selected_checkpoint_settings": {
            name: selected_inputs[name]["profile_config"] for name in runner.MODELS
        },
        "parameter_counts": {
            name: selected_inputs[name]["parameter_counts"] for name in runner.MODELS
        },
        "models": models,
        "runtime": runtime,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    child_manifest = {
        "status": "passed",
        "suite": job["suite"],
        "seed_axes": axes,
        "evaluation_scope": "selected_test",
        "training_performed": False,
        "dataset_cache_integrity": summary["dataset_cache_integrity"],
        "model_split_usage": summary["model_split_usage"],
        "selected_checkpoint_inputs": expected_checkpoints,
        "runtime": runtime,
        "artifacts": {"summary.json": {"path": str(summary_path), "sha256": _digest(summary_path)}},
    }
    (output / "manifest.json").write_text(json.dumps(child_manifest), encoding="utf-8")


def _stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str | None = None,
) -> tuple[list[str], list[list[str]]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"unit_fixture_only": True})
    monkeypatch.setattr(
        runner,
        "_source_snapshot",
        lambda: {"git_revision": "unit", "sha256": {"unit-source": "stable"}},
    )

    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        calls.append(command)
        if any(Path(part).name == "gpu_preflight.py" for part in command):
            if failure == "preflight_exit":
                return 7
            preflight = Path(command[command.index("--json-out") + 1])
            if failure != "preflight_missing":
                preflight.write_text(
                    json.dumps(
                        {
                            "status": "failed" if failure == "preflight_status" else "passed",
                            "gpu": {
                                "name": "unit A6000",
                                "free_bytes": 47 * 1024**3,
                                "total_bytes": 48 * 1024**3,
                                "compute_capability": [8, 6],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            return 0
        if failure == "child_exit":
            return 9
        output = Path(command[command.index("--output-dir") + 1])
        manifest = json.loads(
            (tmp_path / "tree_augmentation/scaling/unit/manifest.json").read_text("utf-8")
        )
        candidate = next(
            (job for job in manifest["jobs"] if Path(job["output_dir"]) == output), None
        )
        if candidate is not None:
            candidate_call_count = sum(
                "--evaluation-scope" in call
                and call[call.index("--evaluation-scope") + 1] == "validation"
                for call in calls
            )
            if failure == "second_child_exit" and candidate_call_count == 2:
                return 9
            if failure == "seed4_exit" and candidate["model_seed"] == 4:
                return 9
            if failure == "second_child_interrupt" and candidate_call_count == 2:
                raise KeyboardInterrupt
            if failure == "child_exit_with_artifact":
                _write_child(candidate)
                return 9
            if failure != "child_missing":
                _write_child(candidate, malformed="metric" if failure == "child_metric" else None)
        else:
            selected = next(
                job for job in manifest["selected_test_jobs"] if Path(job["output_dir"]) == output
            )
            selected_call_count = sum(
                "--evaluation-scope" in call
                and call[call.index("--evaluation-scope") + 1] == "selected_test"
                for call in calls
            )
            if failure == "second_selected_exit" and selected_call_count == 2:
                return 9
            _write_selected_child(selected)
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    options = [
        "--results-root",
        str(tmp_path),
        "--run-id",
        "unit",
        "--suites",
        "csl",
        "--profiles",
        "large",
        "--model-seeds",
        "3,4",
    ]
    return options, calls


def test_success_checks_metrics_parameters_artifacts_and_records_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    assert len(calls) == 5
    root = tmp_path / "tree_augmentation/scaling/unit"
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    assert manifest["status"] == summary["status"] == "passed"
    assert manifest["planned_child_runs"] == 2
    assert manifest["planned_model_trainings"] == 4
    assert summary["completed_child_runs"] == 2
    assert summary["completed_model_trainings"] == 4
    assert summary["completed_profile_selections"] == 2
    assert summary["completed_selected_test_runs"] == 2
    assert summary["completed_selected_checkpoint_test_evaluations"] == 4
    assert len(summary["selections"]) == 1
    assert len(summary["selected_test_results"]) == 2
    assert all(job["status"] == "passed" for job in manifest["jobs"])
    assert all(job["result"]["child_metrics_checked"] for job in manifest["jobs"])
    assert all(job["result"]["test_evaluated"] is False for job in manifest["jobs"])
    assert all(
        job["result"]["dataset_cache_integrity"]["full_cache_loaded"] is True
        for job in manifest["jobs"]
    )
    assert all(row["test_evaluated"] is True for row in summary["selected_test_results"])
    assert all(row["test_used_for_selection"] is False for row in summary["selected_test_results"])
    assert all(
        job["result"]["profile_config"] == runner.PROFILE_CONFIGS["large"]
        for job in manifest["jobs"]
    )
    assert all(
        set(job["result"]["parameter_counts"]) == set(runner.MODELS) for job in manifest["jobs"]
    )
    assert "chart_family_isolation" in summary


def test_completed_run_returns_without_relaunching_preflight_or_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    assert len(calls) == 5
    root = tmp_path / "tree_augmentation/scaling/unit"
    before = json.loads((root / "manifest.json").read_text("utf-8"))
    candidate_results = [job["result"] for job in before["jobs"]]
    selected_results = [job["result"] for job in before["selected_test_jobs"]]
    unexpected_calls: list[list[str]] = []

    def failing_if_relaunched(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        unexpected_calls.append(command)
        return 97

    monkeypatch.setattr(runner, "_run_logged", failing_if_relaunched)
    assert runner.main(options) == 0
    assert unexpected_calls == []

    after = json.loads((root / "manifest.json").read_text("utf-8"))
    assert after["status"] == "passed"
    assert all(job["status"] == "passed" for job in after["jobs"])
    assert all(job["status"] == "passed" for job in after["selected_test_jobs"])
    assert [job["result"] for job in after["jobs"]] == candidate_results
    assert [job["result"] for job in after["selected_test_jobs"]] == selected_results


def test_resume_skips_verified_candidate_and_retries_incomplete_child_on_new_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, first_calls = _stub(tmp_path, monkeypatch, failure="second_child_interrupt")
    assert runner.main(options) == 130
    assert len(first_calls) == 3
    root = tmp_path / "tree_augmentation/scaling/unit"
    failed_manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    assert [job["status"] for job in failed_manifest["jobs"]] == ["passed", "failed"]
    first_output = Path(failed_manifest["jobs"][0]["output_dir"])
    first_summary_digest = _digest(first_output / "summary.json")

    resume_options, resume_calls = _stub(tmp_path, monkeypatch)
    assert resume_options == options
    assert runner.main(resume_options) == 0
    assert len(resume_calls) == 4  # preflight + one candidate retry + two selected tests
    candidate_calls = [
        call
        for call in resume_calls
        if "--evaluation-scope" in call
        and call[call.index("--evaluation-scope") + 1] == "validation"
    ]
    assert len(candidate_calls) == 1
    assert Path(candidate_calls[0][candidate_calls[0].index("--output-dir") + 1]) != first_output
    assert _digest(first_output / "summary.json") == first_summary_digest

    resumed = json.loads((root / "manifest.json").read_text("utf-8"))
    assert resumed["status"] == "passed"
    assert resumed["invocation_count"] == 2
    assert resumed["jobs"][0]["attempt"] == 1
    assert resumed["jobs"][1]["attempt"] == 2
    assert len(resumed["jobs"][1]["attempt_history"]) == 1
    assert all(job["status"] == "passed" for job in resumed["jobs"])


def test_a6000_failed_concurrent_peer_preserves_passed_artifact_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, first_calls = _stub(tmp_path, monkeypatch, failure="seed4_exit")
    options += ["--hardware-profile", "a6000-48gb"]
    assert runner.main(options) == 1
    root = tmp_path / "tree_augmentation/scaling/unit"
    failed = json.loads((root / "manifest.json").read_text("utf-8"))
    assert failed["config"]["job_concurrency"] == 2
    assert failed["config"]["batch_size"] == 64
    assert [job["status"] for job in failed["jobs"]] == ["passed", "failed"]
    passed_output = Path(failed["jobs"][0]["output_dir"])
    passed_digest = _digest(passed_output / "summary.json")

    resume_options, resume_calls = _stub(tmp_path, monkeypatch)
    resume_options += ["--hardware-profile", "a6000-48gb"]
    assert runner.main(resume_options) == 0
    candidate_calls = [
        call
        for call in resume_calls
        if "--evaluation-scope" in call
        and call[call.index("--evaluation-scope") + 1] == "validation"
    ]
    assert len(candidate_calls) == 1
    assert "--amp" in candidate_calls[0]
    assert candidate_calls[0][candidate_calls[0].index("--batch-size") + 1] == "64"
    assert _digest(passed_output / "summary.json") == passed_digest
    resumed = json.loads((root / "manifest.json").read_text("utf-8"))
    assert [job["status"] for job in resumed["jobs"]] == ["passed", "passed"]
    assert all("runtime" in job["result"] for job in resumed["jobs"])
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    assert all(result["runtime"]["peak_gpu_allocated_bytes"] > 0 for result in summary["results"])


def test_a6000_concurrent_wave_fails_closed_if_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, calls = _stub(tmp_path, monkeypatch)
    options += ["--hardware-profile", "a6000-48gb"]
    stable = {"git_revision": "unit", "sha256": {"unit-source": "stable"}}
    changed = {"git_revision": "unit", "sha256": {"unit-source": "changed"}}
    snapshots = iter((stable, stable, changed))
    monkeypatch.setattr(runner, "_source_snapshot", lambda: next(snapshots, changed))
    assert runner.main(options) == 1
    manifest = json.loads(
        (tmp_path / "tree_augmentation/scaling/unit/manifest.json").read_text("utf-8")
    )
    assert manifest["source_integrity_valid"] is False
    assert manifest["status"] == "failed"
    candidate_calls = [call for call in calls if "--evaluation-scope" in call]
    assert len(candidate_calls) == 2
    assert manifest["selected_test_jobs"] == []


def test_resume_skips_candidates_and_verified_selected_checkpoint_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, first_calls = _stub(tmp_path, monkeypatch, failure="second_selected_exit")
    assert runner.main(options) == 1
    assert len(first_calls) == 5
    root = tmp_path / "tree_augmentation/scaling/unit"
    failed_manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    assert [job["status"] for job in failed_manifest["jobs"]] == ["passed", "passed"]
    assert [job["status"] for job in failed_manifest["selected_test_jobs"]] == [
        "passed",
        "failed",
    ]
    completed_test_output = Path(failed_manifest["selected_test_jobs"][0]["output_dir"])
    completed_test_digest = _digest(completed_test_output / "summary.json")

    resume_options, resume_calls = _stub(tmp_path, monkeypatch)
    assert runner.main(resume_options) == 0
    assert len(resume_calls) == 2  # preflight + only the incomplete selected test
    scoped = [call for call in resume_calls if "--evaluation-scope" in call]
    assert len(scoped) == 1
    assert scoped[0][scoped[0].index("--evaluation-scope") + 1] == "selected_test"
    assert _digest(completed_test_output / "summary.json") == completed_test_digest

    resumed = json.loads((root / "manifest.json").read_text("utf-8"))
    assert resumed["status"] == "passed"
    assert [job["attempt"] for job in resumed["selected_test_jobs"]] == [1, 2]
    assert all(job["status"] == "passed" for job in resumed["selected_test_jobs"])
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    assert summary["completed_selected_checkpoint_test_evaluations"] == 4


def test_nonzero_candidate_with_complete_artifacts_is_retried_on_a_new_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, _calls = _stub(tmp_path, monkeypatch, failure="child_exit_with_artifact")
    assert runner.main(options) == 1
    root = tmp_path / "tree_augmentation/scaling/unit"
    failed = json.loads((root / "manifest.json").read_text("utf-8"))
    failed_output = Path(failed["jobs"][0]["output_dir"])
    assert (failed_output / "summary.json").is_file()

    resume_options, resume_calls = _stub(tmp_path, monkeypatch)
    assert runner.main(resume_options) == 0
    candidate_calls = [
        call
        for call in resume_calls
        if "--evaluation-scope" in call
        and call[call.index("--evaluation-scope") + 1] == "validation"
    ]
    assert candidate_calls
    assert Path(candidate_calls[0][candidate_calls[0].index("--output-dir") + 1]) != failed_output


def test_passed_candidate_result_mismatch_schedules_a_new_attempt(tmp_path: Path) -> None:
    args = runner.parser().parse_args(
        [
            "--results-root",
            str(tmp_path),
            "--run-id",
            "result-mismatch",
            "--suites",
            "csl",
            "--profiles",
            "large",
            "--model-seeds",
            "3",
        ]
    )
    run_dir = (tmp_path / "tree_augmentation/scaling/result-mismatch").resolve()
    expected = runner.make_jobs(args, run_dir)[0]
    job = json.loads(json.dumps(expected))
    _write_child(job)
    accepted = runner._validate_child(job)
    job["status"] = "passed"
    job["result"] = json.loads(json.dumps(accepted))
    job["result"]["child_metrics_checked"] = False
    original_output = Path(job["output_dir"])

    runner._reconcile_candidate(job, expected, run_dir)

    assert job["status"] == "pending"
    assert job["attempt"] == 2
    assert Path(job["output_dir"]) != original_output
    assert original_output.is_dir()
    assert job["attempt_history"][-1]["validation_error"] == (
        "passed candidate differs from its accepted result"
    )


def test_interrupted_resume_rejects_preflight_symlink_before_child_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, first_calls = _stub(tmp_path, monkeypatch, failure="second_child_interrupt")
    assert runner.main(options) == 130
    assert len(first_calls) == 3
    run = tmp_path / "tree_augmentation/scaling/unit"
    preflight = run / "gpu-preflight.attempt-2.json"
    sentinel = tmp_path / "outside-preflight.json"
    sentinel.write_text("preserve-preflight", encoding="utf-8")
    try:
        preflight.symlink_to(sentinel)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    resume_calls: list[list[str]] = []

    def unexpected_launch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        resume_calls.append(command)
        pytest.fail("resume launched a subprocess before rejecting the preflight symlink")

    monkeypatch.setattr(runner, "_run_logged", unexpected_launch)
    assert runner.main(options) == 1
    assert resume_calls == []
    assert sentinel.read_text(encoding="utf-8") == "preserve-preflight"


def test_interrupted_resume_rejects_summary_symlink_before_writes_or_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, first_calls = _stub(tmp_path, monkeypatch, failure="second_child_interrupt")
    assert runner.main(options) == 130
    assert len(first_calls) == 3
    run = tmp_path / "tree_augmentation/scaling/unit"
    summary = run / "summary.json"
    summary.unlink()
    sentinel = tmp_path / "outside-summary.json"
    sentinel.write_text("preserve-summary", encoding="utf-8")
    try:
        summary.symlink_to(sentinel)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    manifest = run / "manifest.json"
    manifest_before = manifest.read_bytes()
    launches: list[list[str]] = []
    writes: list[Path] = []

    def unexpected_launch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        launches.append(command)
        pytest.fail("resume launched a subprocess before rejecting the summary symlink")

    def unexpected_write(path: Path, *_args, **_kwargs) -> None:
        writes.append(Path(path))
        pytest.fail("resume wrote state before rejecting the summary symlink")

    monkeypatch.setattr(runner, "_run_logged", unexpected_launch)
    monkeypatch.setattr(runner, "atomic_write_json", unexpected_write)
    assert runner.main(options) == 2
    assert launches == [] and writes == []
    assert manifest.read_bytes() == manifest_before
    assert sentinel.read_text(encoding="utf-8") == "preserve-summary"


def test_retry_path_rejects_resume_attempts_symlink_outside_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (run_dir / "resume-attempts").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    original_output = run_dir / "candidate"
    command = ["python", "child.py", "--output-dir", str(original_output)]
    expected = {
        "suite": "csl",
        "profile": "reference",
        "model_seed": 0,
        "command": command,
        "output_dir": str(original_output),
        "log_path": str(run_dir / "logs/candidate.log"),
    }
    job = {**expected, "attempt": 1, "status": "failed", "exit_code": 9}
    with pytest.raises(ValueError, match="outside the run directory"):
        runner._retry_record(
            job,
            expected,
            run_dir,
            kind="candidate",
            validation_error="interrupted",
        )


@pytest.mark.parametrize(
    "failure,calls_expected",
    [
        ("preflight_exit", 1),
        ("preflight_missing", 1),
        ("preflight_status", 1),
        ("child_exit", 2),
        ("child_missing", 2),
        ("child_metric", 2),
    ],
)
def test_fail_closed_stops_after_first_unverified_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    calls_expected: int,
) -> None:
    options, calls = _stub(tmp_path, monkeypatch, failure=failure)
    assert runner.main(options) == 1
    assert len(calls) == calls_expected
    root = tmp_path / "tree_augmentation/scaling/unit"
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    assert manifest["status"] == summary["status"] == "failed"
    assert summary["completed_child_runs"] == 0
    assert manifest["jobs"][1]["status"] == "pending"


def test_existing_run_is_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "tree_augmentation/scaling/existing"
    root.mkdir(parents=True)
    sentinel = root / "model.pt"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(runner, "check_dependencies", lambda: pytest.fail("dependency check"))
    assert runner.main(["--results-root", str(tmp_path), "--run-id", "existing"]) == 2
    assert sentinel.read_bytes() == b"preserve"


def test_source_snapshot_covers_tree_model_runner_and_shared_math() -> None:
    snapshot = runner._source_snapshot()["sha256"]
    for name in (
        "scripts/run_tree_scaling.py",
        "research/tree_augmentation/paper.py",
        "research/tree_augmentation/paper_model.py",
        "research/tree_augmentation/paper_data.py",
        "research/tree_augmentation/config.yaml",
        "src/chartgat/algebra.py",
        "src/chartgat/graphs.py",
        "src/chartgat/seeds.py",
    ):
        assert name in snapshot and len(snapshot[name]) == 64
