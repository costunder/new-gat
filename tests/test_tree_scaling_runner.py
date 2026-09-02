"""Tree scaling orchestration contracts; no GPU or research training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.tree_augmentation import paper as tree_paper
from scripts import run_tree_scaling as runner


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_matrix_trains_both_versions_across_larger_profiles() -> None:
    args = runner.parser().parse_args([])
    jobs = runner.make_jobs(args, Path("fixture"))
    assert args.suites == ("csl", "zinc")
    assert args.profiles == ("base", "wide", "deep", "large")
    assert args.model_seeds == (0, 1, 2, 3, 4)
    assert len(jobs) == 40
    assert sum(len(job["trained_models"]) for job in jobs) == 80
    assert len({job["output_dir"] for job in jobs}) == len(jobs)
    assert runner.PROFILE_CONFIGS == {
        "base": {
            "hidden_dim": 64,
            "message_layers": 2,
            "optimizer_updates": 800,
            "train_charts_per_graph": 8,
            "eval_charts_per_graph": 8,
        },
        "wide": {
            "hidden_dim": 128,
            "message_layers": 2,
            "optimizer_updates": 800,
            "train_charts_per_graph": 8,
            "eval_charts_per_graph": 8,
        },
        "deep": {
            "hidden_dim": 64,
            "message_layers": 4,
            "optimizer_updates": 800,
            "train_charts_per_graph": 8,
            "eval_charts_per_graph": 8,
        },
        "large": {
            "hidden_dim": 128,
            "message_layers": 4,
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
        hidden_dim=128,
        message_layers=4,
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
                    json.dumps({"status": "failed" if failure == "preflight_status" else "passed"}),
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
            if failure != "child_missing":
                _write_child(candidate, malformed="metric" if failure == "child_metric" else None)
        else:
            selected = next(
                job for job in manifest["selected_test_jobs"] if Path(job["output_dir"]) == output
            )
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
        "wide",
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
        job["result"]["profile_config"] == runner.PROFILE_CONFIGS["wide"]
        for job in manifest["jobs"]
    )
    assert all(
        set(job["result"]["parameter_counts"]) == set(runner.MODELS) for job in manifest["jobs"]
    )
    assert "chart_family_isolation" in summary


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
