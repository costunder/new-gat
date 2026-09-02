"""Contracts for the unified larger-model scaling runner; no research training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import run_rich_scaling as runner


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _options(command: list[str], name: str) -> list[str]:
    start = command.index(name) + 1
    values: list[str] = []
    for value in command[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def test_default_plan_includes_every_track_profile_seed_and_true_tree_deep(tmp_path: Path):
    args = runner.parser().parse_args(["--results-root", str(tmp_path)])
    runner._validate(args)
    jobs = runner.make_jobs(args, "matrix")
    assert args.tracks == list(runner.TRACKS)
    assert args.profiles == list(runner.PROFILES)
    assert args.model_seeds == list(runner.DEFAULT_MODEL_SEEDS)
    assert [job["track"] for job in jobs] == ["conductance", "cycle", "tree"]
    assert runner._totals(jobs) == {
        "track_runs": 3,
        "child_runs": 188,
        "model_trainings": 204,
    }

    conductance, cycle, tree = jobs
    assert conductance["profiles"] == ["base", "wide", "deep", "large"]
    assert conductance["command"][conductance["command"].index("--profiles") + 1 :][:4] == [
        "base",
        "wide",
        "deep",
        "large",
    ]
    assert _option(cycle["command"], "--model-seeds") == "0"
    assert _option(tree["command"], "--profiles") == "base,wide,deep,large"
    assert _option(tree["command"], "--model-seeds") == "0"
    assert _option(tree["command"], "--suites") == "csl,zinc"
    assert conductance["requested_matrix"]["versions"] == ["v1", "v2", "v3", "v4"]
    assert cycle["requested_matrix"]["datasets"] == ["zinc12k", "peptides_struct"]
    assert Path(conductance["summary_path"]).relative_to(tmp_path) == Path(
        "conductance_gat/scaling/matrix-conductance/summary.json"
    )
    assert Path(cycle["summary_path"]).relative_to(tmp_path) == Path(
        "cycle_pe/scaling/matrix-cycle/summary.json"
    )
    assert Path(tree["summary_path"]).relative_to(tmp_path) == Path(
        "tree_augmentation/scaling/matrix-tree/summary.json"
    )


def test_selection_and_download_flag_are_mapped_only_to_supported_child_clis(tmp_path: Path):
    args = runner.parser().parse_args(
        [
            "--tracks",
            "conductance",
            "cycle",
            "tree",
            "--profiles",
            "deep",
            "large",
            "--model-seeds",
            "2",
            "7",
            "--allow-download",
            "--results-root",
            str(tmp_path),
        ]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, "selected")
    assert runner._totals(jobs)["model_trainings"] == (43 + 4 + 4) * 2 * 2
    assert "--allow-download" not in jobs[0]["command"]
    assert "--allow-download" in jobs[1]["command"]
    assert "--allow-download" in jobs[2]["command"]
    assert _option(jobs[2]["command"], "--profiles") == "deep,large"


def test_central_source_inventory_covers_imported_child_code_and_tree_config():
    snapshot = runner._source_snapshot()
    assert "research/__init__.py" in snapshot
    assert "src/chartgat/graphs.py" in snapshot
    assert "research/cycle_pe/benchmark_data.py" in snapshot
    assert "research/conductance_gat/v4/train.py" in snapshot
    assert "research/tree_augmentation/paper.py" in snapshot
    assert "research/tree_augmentation/config.yaml" in snapshot
    assert "research/tree_augmentation/datasets.yaml" in snapshot
    assert "scripts/gpu_profiles.py" in snapshot
    assert "scripts/verify_gpu_lock.py" in snapshot


def test_run_logged_appends_a_resume_attempt_to_an_existing_log(tmp_path: Path):
    log = tmp_path / "track.log"
    log.write_text("first attempt\n", encoding="utf-8")
    command = [sys.executable, "-c", "print('second attempt')"]
    assert runner._run_logged(command, log, runner._environment()) == 0
    content = log.read_text(encoding="utf-8")
    assert "first attempt" in content
    assert "=== resumed " in content
    assert "second attempt" in content


def test_dry_run_prints_complete_plan_without_writes_or_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(
        runner,
        "_run_logged",
        lambda *_args: pytest.fail("dry run must not launch child processes"),
    )
    code = runner.main(
        [
            "--tracks",
            "cycle",
            "tree",
            "--profiles",
            "base",
            "deep",
            "--model-seeds",
            "0",
            "--results-root",
            str(tmp_path),
            "--run-id",
            "dry",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "2 track runs; 8 child runs; 16 fresh model trainings" in output
    assert "child profiles=['base', 'deep']" in output
    assert "--profiles base,deep" in output
    assert "no files or directories were written" in output
    assert list(tmp_path.iterdir()) == []


def _write_summary(command: list[str], *, malformed: str | None = None) -> None:
    script = Path(command[2]).name
    results_root = Path(_option(command, "--results-root"))
    run_id = _option(command, "--run-id")
    if script == "run_conductance_scaling.py":
        versions = _options(command, "--versions")
        requested_datasets = _options(command, "--datasets")
        profiles = _options(command, "--profiles")
        seeds = [int(value) for value in _options(command, "--model-seeds")]
        rows = []
        exclusions = []
        for version in versions:
            spec = runner.CONDUCTANCE_MATRIX[version]
            for dataset in requested_datasets:
                if dataset not in spec["datasets"]:
                    exclusions.append(
                        {
                            "version": version,
                            "dataset": dataset,
                            "status": "not_applicable",
                        }
                    )
            for profile in profiles:
                for dataset in spec["datasets"]:
                    for condition in spec["conditions"]:
                        rows.append(
                            {
                                "version": version,
                                "profile": profile,
                                "dataset": dataset,
                                "condition": condition,
                                "passed_seeds": sorted(seeds),
                                "n": len(seeds),
                            }
                        )
        count = len(rows) * len(seeds)
        output = results_root / "conductance_gat/scaling" / run_id
        payload = {
            "status": "failed" if malformed == "status" else "passed",
            "suite": "conductance_architecture_scaling_v1_v4",
            "run_id": run_id,
            "valid_for_validation_comparison": malformed != "status",
            "test_evaluated": False,
            "job_counts": {"pending": 0, "running": 0, "passed": count, "failed": 0},
            "expected_model_seeds": seeds,
            "exclusions": exclusions,
            "rows": rows,
        }
    elif script == "run_cycle_scaling.py":
        versions = _options(command, "--versions")
        datasets = _options(command, "--datasets")
        profiles = _options(command, "--profiles")
        seeds = [int(value) for value in _option(command, "--model-seeds").split(",")]
        rows = [
            {
                "version": version,
                "profile": profile,
                "model_seed": seed,
                "dataset": dataset,
            }
            for version in versions
            for profile in profiles
            for seed in seeds
            for dataset in datasets
        ]
        aggregates = [
            {
                "version": version,
                "dataset": dataset,
                "profile": profile,
                "model_seeds": sorted(seeds),
            }
            for version in versions
            for dataset in datasets
            for profile in profiles
        ]
        selections = [
            {
                "version": version,
                "dataset": dataset,
                "selected_profile": profiles[0],
                "model_seeds": seeds,
                "test_used_for_selection": False,
            }
            for version in versions
            for dataset in datasets
        ]
        selected_checkpoints = [
            {
                "version": selection["version"],
                "dataset": selection["dataset"],
                "model_seed": seed,
                "selected_profile": selection["selected_profile"],
                "checkpoint": f"{selection['version']}-{selection['dataset']}-{seed}.pt",
                "checkpoint_sha256": f"hash-{seed}",
            }
            for selection in selections
            for seed in seeds
        ]
        test_evaluations = [
            {**checkpoint, "fresh_training": False} for checkpoint in selected_checkpoints
        ]
        output = results_root / "cycle_pe/scaling" / run_id
        payload = {
            "status": "failed" if malformed == "status" else "passed",
            "scope": "cycle_pe_v1_v2_larger_model_scaling",
            "requested_model_seeds": seeds,
            "profiles": {profile: {} for profile in profiles},
            "runs": rows,
            "profile_aggregates": aggregates,
            "profile_selections": selections,
            "selected_checkpoints": selected_checkpoints,
            "test_evaluations": test_evaluations,
            "fresh_dataset_trainings": len(rows),
            "selected_test_evaluations": len(test_evaluations),
            "final_test_aggregates": [
                {
                    "version": version,
                    "dataset": dataset,
                    "model_seeds": seeds,
                    "selected_profiles": [profiles[0] for _seed in seeds],
                }
                for version in versions
                for dataset in datasets
            ],
        }
    else:
        suites = _option(command, "--suites").split(",")
        profiles = _option(command, "--profiles").split(",")
        seeds = [int(value) for value in _option(command, "--model-seeds").split(",")]
        children = 2 * len(profiles) * len(seeds)
        trainings = children * 2
        results = [
            {
                "suite": suite,
                "profile": profile,
                "model_seed": seed,
                "trained_models": list(runner.TREE_MODELS),
                "dataset_cache_integrity": {
                    "full_cache_loaded": True,
                    "all_declared_splits_validated": True,
                    "loaded_and_validated_splits": ["test", "train", "validation"],
                },
                "test_evaluated": False,
                "test_used_for_selection": False,
            }
            for suite in suites
            for profile in profiles
            for seed in seeds
        ]
        output = results_root / "tree_augmentation/scaling" / run_id
        payload = {
            "status": "failed" if malformed == "status" else "passed",
            "suite": "tree_scaling",
            "run_id": run_id,
            "planned_child_runs": children,
            "planned_model_trainings": trainings,
            "completed_child_runs": children,
            "completed_model_trainings": trainings,
            "failed_child_runs": 0,
            "models_per_child": list(runner.TREE_MODELS),
            "profile_configs": {profile: {} for profile in profiles},
            "results": results,
            "planned_profile_selections": len(suites) * len(runner.TREE_MODELS),
            "completed_profile_selections": len(suites) * len(runner.TREE_MODELS),
            "planned_selected_test_runs": len(suites) * len(seeds),
            "completed_selected_test_runs": len(suites) * len(seeds),
            "failed_selected_test_runs": 0,
            "planned_selected_checkpoint_test_evaluations": (
                len(suites) * len(seeds) * len(runner.TREE_MODELS)
            ),
            "completed_selected_checkpoint_test_evaluations": (
                len(suites) * len(seeds) * len(runner.TREE_MODELS)
            ),
            "selections": [
                {
                    "suite": suite,
                    "selection_split": "validation",
                    "aggregation_axis": "mean_across_requested_model_seeds",
                    "model_seeds": seeds,
                    "test_metrics_used_for_selection": False,
                    "conditions": {
                        model: {
                            "selected_profile": profiles[0],
                            "selected_checkpoints_by_model_seed": {
                                str(seed): {"checkpoint": f"{suite}-{model}-{seed}.pt"}
                                for seed in seeds
                            },
                        }
                        for model in runner.TREE_MODELS
                    },
                }
                for suite in suites
            ],
            "selected_test_results": [
                {
                    "suite": suite,
                    "model_seed": seed,
                    "evaluation_scope": "selected_test",
                    "training_performed": False,
                    "test_evaluated": True,
                    "test_used_for_selection": False,
                    "selected_profiles": {model: profiles[0] for model in runner.TREE_MODELS},
                    "selected_checkpoints": {
                        model: {"path": f"{suite}-{model}-{seed}.pt"}
                        for model in runner.TREE_MODELS
                    },
                    "test_evaluations_per_selected_checkpoint": 1,
                }
                for suite in suites
                for seed in seeds
            ],
        }

    if malformed == "duplicate":
        key = "runs" if script == "run_cycle_scaling.py" else "results"
        if script == "run_conductance_scaling.py":
            key = "rows"
        payload[key][-1] = dict(payload[key][0])
    elif malformed is not None and malformed.startswith("matrix:"):
        field = malformed.split(":", 1)[1]
        key = "runs" if script == "run_cycle_scaling.py" else "results"
        if script == "run_conductance_scaling.py":
            key = "rows"
        row = payload[key][0]
        if field == "seed_set":
            row["passed_seeds"] = [999]
        elif field == "trained_models":
            row["trained_models"] = ["fixed_bfs"]
        else:
            row[field] = 999 if field == "model_seed" else "unexpected"
    output.mkdir(parents=True)
    if malformed != "missing":
        (output / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _base_options(tmp_path: Path) -> list[str]:
    return [
        "--profiles",
        "base",
        "--model-seeds",
        "0",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path),
        "--run-id",
        "unit",
    ]


def test_success_runs_tracks_sequentially_and_certifies_all_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        calls.append(Path(command[2]).name)
        _write_summary(command)
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    assert runner.main(_base_options(tmp_path)) == 0
    assert calls == [
        "run_conductance_scaling.py",
        "run_cycle_scaling.py",
        "run_tree_scaling.py",
    ]
    manifest_path = tmp_path / "rich_scaling/unit/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["planned_counts"] == {
        "track_runs": 3,
        "child_runs": 47,
        "model_trainings": 51,
    }
    assert manifest["completed_counts"]["verified_model_trainings"] == 51
    assert all(job["status"] == "passed" for job in manifest["jobs"])
    assert all(len(job["result"]["summary_sha256"]) == 64 for job in manifest["jobs"])


@pytest.mark.parametrize("track", ["conductance", "cycle", "tree"])
def test_equal_count_duplicate_child_matrix_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, track: str
):
    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        _write_summary(command, malformed="duplicate")
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    assert runner.main(["--tracks", track, *_base_options(tmp_path)]) == 1
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["jobs"][0]["status"] == "failed"
    assert "duplicate key" in manifest["jobs"][0]["error"]


@pytest.mark.parametrize(
    ("track", "field"),
    [
        ("conductance", "version"),
        ("conductance", "dataset"),
        ("conductance", "profile"),
        ("conductance", "condition"),
        ("conductance", "seed_set"),
        ("cycle", "version"),
        ("cycle", "dataset"),
        ("cycle", "profile"),
        ("cycle", "model_seed"),
        ("tree", "suite"),
        ("tree", "profile"),
        ("tree", "model_seed"),
        ("tree", "trained_models"),
    ],
)
def test_child_summary_must_match_every_requested_matrix_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, track: str, field: str
):
    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        _write_summary(command, malformed=f"matrix:{field}")
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    assert runner.main(["--tracks", track, *_base_options(tmp_path)]) == 1
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["jobs"][0]["status"] == "failed"
    assert any(
        phrase in manifest["jobs"][0]["error"]
        for phrase in ("matrix mismatch", "wrong seed set", "wrong trained models")
    )


def test_central_sources_are_rehashed_after_children_and_change_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        _write_summary(command)
        return 0

    snapshots = iter([{"scripts/run_rich_scaling.py": "before"}, {"changed.py": "after"}])
    monkeypatch.setattr(runner, "_run_logged", dispatch)
    monkeypatch.setattr(runner, "_source_snapshot", lambda: next(snapshots))
    assert runner.main(["--tracks", "cycle", *_base_options(tmp_path)]) == 1
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["jobs"][0]["status"] == "passed"
    assert manifest["status"] == "failed"
    assert manifest["source_integrity_valid"] is False
    assert "source changed" in manifest["source_integrity_error"]
    assert manifest["completed_counts"]["verified_model_trainings"] == 0


def test_default_failure_policy_continues_later_tracks_and_marks_overall_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        script = Path(command[2]).name
        calls.append(script)
        if script == "run_conductance_scaling.py":
            return 9
        _write_summary(command)
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    assert runner.main(_base_options(tmp_path)) == 1
    assert calls == [
        "run_conductance_scaling.py",
        "run_cycle_scaling.py",
        "run_tree_scaling.py",
    ]
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert [job["status"] for job in manifest["jobs"]] == ["failed", "passed", "passed"]
    assert manifest["jobs"][0]["returncode"] == 9


def test_fail_fast_leaves_later_tracks_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        calls.append(Path(command[2]).name)
        return 7

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    assert runner.main([*_base_options(tmp_path), "--fail-fast"]) == 1
    assert calls == ["run_conductance_scaling.py"]
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert [job["status"] for job in manifest["jobs"]] == ["failed", "pending", "pending"]


@pytest.mark.parametrize("malformed", ["missing", "status"])
def test_zero_return_code_without_exact_passed_summary_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: str
):
    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        _write_summary(command, malformed=malformed)
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    options = [
        "--tracks",
        "conductance",
        *_base_options(tmp_path),
    ]
    assert runner.main(options) == 1
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["jobs"][0]["status"] == "failed"
    assert "summary" in manifest["jobs"][0]["error"]


def test_invalid_existing_central_run_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = tmp_path / "rich_scaling/existing"
    run.mkdir(parents=True)
    sentinel = run / "manifest.json"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(runner, "_run_logged", lambda *_args: pytest.fail("must not launch"))
    assert (
        runner.main(
            [
                "--results-root",
                str(tmp_path),
                "--data-root",
                str(tmp_path / "data"),
                "--run-id",
                "existing",
            ]
        )
        == 2
    )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_same_run_id_revalidates_passed_track_and_resumes_failed_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first_calls: list[str] = []

    def first_dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        script = Path(command[2]).name
        first_calls.append(script)
        if script == "run_cycle_scaling.py":
            return 9
        _write_summary(command)
        return 0

    options = ["--tracks", "conductance", "cycle", *_base_options(tmp_path)]
    monkeypatch.setattr(runner, "_run_logged", first_dispatch)
    assert runner.main(options) == 1
    assert first_calls == ["run_conductance_scaling.py", "run_cycle_scaling.py"]

    resumed_calls: list[str] = []

    def resumed_dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        script = Path(command[2]).name
        resumed_calls.append(script)
        if script == "run_cycle_scaling.py":
            _write_summary(command)
        return 0

    monkeypatch.setattr(runner, "_run_logged", resumed_dispatch)
    assert runner.main(options) == 0
    assert resumed_calls == ["run_conductance_scaling.py", "run_cycle_scaling.py"]
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "passed"
    assert manifest["resume_count"] == 1
    assert [job["status"] for job in manifest["jobs"]] == ["passed", "passed"]


def test_resume_rejects_changed_config_and_source_without_launching_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        _write_summary(command)
        return 0

    options = ["--tracks", "conductance", *_base_options(tmp_path)]
    monkeypatch.setattr(runner, "_run_logged", dispatch)
    assert runner.main(options) == 0
    manifest_path = tmp_path / "rich_scaling/unit/manifest.json"
    original = manifest_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_run_logged",
        lambda *_args: pytest.fail("rejected resume must not launch a child"),
    )
    changed = list(options)
    changed[changed.index("--profiles") + 1] = "large"
    assert runner.main(changed) == 2
    assert manifest_path.read_text(encoding="utf-8") == original

    payload = json.loads(original)
    payload["source_sha256"]["scripts/run_rich_scaling.py"] = "tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.main(options) == 2

    payload = json.loads(original)
    payload["source_integrity_valid"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.main(options) == 2
