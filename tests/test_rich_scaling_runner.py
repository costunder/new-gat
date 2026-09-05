"""Contracts for the unified larger-model scaling runner; no research training."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts import run_rich_scaling as runner


@pytest.fixture(autouse=True)
def _isolate_post_calibration_runner_contracts(monkeypatch):
    """These pre-existing tests mock training children, not GPU resource probes.

    The actual calibration gate is exercised separately in
    test_rich_resource_calibration.py; it must never start GPU work in this suite.
    """
    monkeypatch.setattr(runner, "_ensure_measured_plan", lambda *args: None)


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
    assert args.conductance_versions == list(runner.CONDUCTANCE_MATRIX)
    assert args.cycle_versions == list(runner.CYCLE_VERSIONS)
    assert args.cycle_v2_encodings == ["se", "pe"]
    assert args.profiles == list(runner.PROFILES)
    assert args.model_seeds == list(runner.DEFAULT_MODEL_SEEDS)
    assert [job["track"] for job in jobs] == ["conductance", "cycle", "tree"]
    assert runner._totals(jobs) == {
        "track_runs": 3,
        "child_runs": 122,
        "model_trainings": 126,
    }

    conductance, cycle, tree = jobs
    assert conductance["profiles"] == ["reference", "large"]
    assert conductance["command"][conductance["command"].index("--profiles") + 1 :][:2] == [
        "reference",
        "large",
    ]
    assert _option(cycle["command"], "--model-seeds") == "0"
    assert _options(cycle["command"], "--encodings") == ["se", "pe"]
    assert cycle["requested_matrix"]["encodings_by_version"] == {"v1": [None], "v2": ["se", "pe"]}
    assert _options(cycle["command"], "--versions") == ["v1", "v2"]
    assert _option(cycle["command"], "--basis-backend") == "dfs_fundamental"
    assert _option(tree["command"], "--profiles") == "reference,large"
    assert _option(tree["command"], "--model-seeds") == "0"
    assert _option(tree["command"], "--suites") == "csl,zinc"
    assert all(_option(job["command"], "--hardware-profile") == "portable" for job in jobs)
    assert _option(conductance["command"], "--v5-beta-parameterization") == "sigmoid"
    assert _option(conductance["command"], "--v5-beta-initial") == "0.1"
    assert "--v5-beta-min" not in conductance["command"]
    assert "--v5-activation-checkpoint" not in conductance["command"]
    assert "--no-v5-activation-checkpoint" not in conductance["command"]
    assert args.v5_activation_checkpoint is None
    assert all("--v5-beta-parameterization" not in job["command"] for job in (cycle, tree))
    assert conductance["requested_matrix"]["versions"] == ["v1", "v2", "v3", "v4", "v5"]
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
            "reference",
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
    assert runner._totals(jobs)["model_trainings"] == (53 + 6 + 4) * 2 * 2
    assert "--allow-download" not in jobs[0]["command"]
    assert "--allow-download" in jobs[1]["command"]
    assert "--allow-download" in jobs[2]["command"]
    assert _option(jobs[2]["command"], "--profiles") == "reference,large"


def test_completed_conductance_v1_v4_can_be_excluded_from_integrated_plan(tmp_path: Path):
    args = runner.parser().parse_args(
        [
            "--conductance-versions",
            "v5",
            "--profiles",
            "reference",
            "large",
            "--model-seeds",
            "0",
            "--results-root",
            str(tmp_path),
        ]
    )
    runner._validate(args)
    conductance, cycle, tree = runner.make_jobs(args, "remaining")
    assert _options(conductance["command"], "--versions") == ["v5"]
    assert _options(cycle["command"], "--versions") == ["v1", "v2"]
    assert cycle["requested_matrix"]["versions"] == ["v1", "v2"]
    assert runner._totals([conductance, cycle, tree]) == {
        "track_runs": 3,
        "child_runs": 36,
        "model_trainings": 40,
    }
    config = runner._config_payload(args, data_root=tmp_path / "data", results_root=tmp_path)
    assert config["conductance_versions"] == ["v5"]
    assert config["cycle_versions"] == ["v1", "v2"]


def test_only_new_v5_and_cycle_v2_plan_has_no_legacy_trainings(tmp_path: Path):
    args = runner.parser().parse_args(
        [
            "--tracks",
            "conductance",
            "cycle",
            "--conductance-versions",
            "v5",
            "--cycle-versions",
            "v2",
            "--results-root",
            str(tmp_path),
        ]
    )
    runner._validate(args)
    conductance, cycle = runner.make_jobs(args, "new-only")
    assert runner._totals([conductance, cycle]) == {
        "track_runs": 2,
        "child_runs": 28,
        "model_trainings": 28,
    }


@pytest.mark.parametrize(
    ("option", "values", "message"),
    [
        ("--conductance-versions", ["v5", "v5"], "conductance versions"),
        ("--cycle-versions", ["v2", "v2"], "cycle versions"),
        ("--cycle-v2-encodings", ["se", "se"], "cycle V2 encodings"),
    ],
)
def test_duplicate_version_selection_is_rejected(option, values, message):
    args = runner.parser().parse_args([option, *values])
    with pytest.raises(ValueError, match=message):
        runner._validate(args)


def test_removed_cycle_qr_backend_is_rejected_before_planning():
    with pytest.raises(SystemExit):
        runner.parser().parse_args(
            [
                "--tracks",
                "cycle",
                "--cycle-versions",
                "v1",
                "--cycle-v2-basis-backend",
                "thin_q",
            ]
        )


def test_a6000_hardware_profile_is_forwarded_to_every_track_and_bound_to_config(
    tmp_path: Path,
) -> None:
    args = runner.parser().parse_args(
        ["--hardware-profile", "a6000-48gb", "--results-root", str(tmp_path)]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, "a6000")
    assert all(_option(job["command"], "--hardware-profile") == "a6000-48gb" for job in jobs)
    config = runner._config_payload(args, data_root=tmp_path / "data", results_root=tmp_path)
    assert config["hardware_profile"] == "a6000-48gb"


def test_explicit_batch_overrides_are_forwarded_without_claiming_measurement(
    tmp_path: Path,
) -> None:
    args = runner.parser().parse_args(
        [
            "--conductance-versions",
            "v1",
            "v5",
            "--hardware-profile",
            "a6000-48gb",
            "--conductance-legacy-ppi-batch-size",
            "4",
            "--conductance-v5-ppi-batch-size",
            "8",
            "--conductance-v5-sample-seed-batch-size",
            "2048",
            "--cycle-batch-size",
            "64",
            "--tree-batch-size",
            "32",
            "--results-root",
            str(tmp_path),
        ]
    )
    runner._validate(args)
    conductance, cycle, tree = runner.make_jobs(args, "batch-overrides")
    assert _option(conductance["command"], "--legacy-ppi-batch-size") == "4"
    assert _option(conductance["command"], "--v5-ppi-batch-size") == "8"
    assert _option(conductance["command"], "--v5-sample-seed-batch-size") == "2048"
    assert _option(cycle["command"], "--batch-size") == "64"
    assert _option(tree["command"], "--batch-size") == "32"
    config = runner._config_payload(args, data_root=tmp_path / "data", results_root=tmp_path)
    assert config["explicit_batch_overrides"] == {
        "conductance_legacy_ppi_graphs": 4,
        "conductance_v5_ppi_graphs": 8,
        "conductance_v5_sample_seed_nodes": 2048,
        "cycle_graphs": 64,
        "tree_chart_views": 32,
    }


@pytest.mark.parametrize(
    "options",
    [
        ["--conductance-v5-ppi-batch-size", "0"],
        ["--conductance-legacy-ppi-batch-size", "0"],
        ["--conductance-v5-sample-seed-batch-size", "0"],
        ["--cycle-batch-size", "0"],
        ["--tree-batch-size", "0"],
    ],
)
def test_explicit_batch_overrides_must_be_positive(options: list[str]) -> None:
    args = runner.parser().parse_args(options)
    with pytest.raises(ValueError, match="must be positive"):
        runner._validate(args)


@pytest.mark.parametrize(
    "options",
    [
        ["--tracks", "cycle", "--conductance-v5-ppi-batch-size", "8"],
        ["--tracks", "cycle", "--conductance-legacy-ppi-batch-size", "4"],
        [
            "--tracks",
            "conductance",
            "--conductance-versions",
            "v1",
            "--conductance-v5-sample-seed-batch-size",
            "2048",
        ],
        ["--tracks", "tree", "--cycle-batch-size", "32"],
        ["--tracks", "cycle", "--tree-batch-size", "32"],
    ],
)
def test_explicit_batch_overrides_require_their_target_track(options: list[str]) -> None:
    args = runner.parser().parse_args(options)
    with pytest.raises(ValueError, match="require"):
        runner._validate(args)


def test_explicit_devices_assign_independent_tracks_round_robin(tmp_path: Path) -> None:
    args = runner.parser().parse_args(
        [
            "--devices",
            "cuda:0",
            "cuda:1",
            "--results-root",
            str(tmp_path),
        ]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, "multi")
    assert [job["device"] for job in jobs] == ["cuda:0", "cuda:1", "cuda:0"]
    assert [_option(job["command"], "--device") for job in jobs] == [
        "cuda:0",
        "cuda:1",
        "cuda:0",
    ]
    config = runner._config_payload(args, data_root=tmp_path / "data", results_root=tmp_path)
    assert config["devices"] == ["cuda:0", "cuda:1"]
    assert config["device_assignment"] == "round_robin_by_independent_track"
    assert config["execution_classification"] == "final_research_training"
    assert config["debug_or_smoke_mode"] is False


@pytest.mark.parametrize(
    "devices",
    [
        ["cuda:0", "cuda:0"],
        ["cuda", "cuda:1"],
        ["cpu", "cuda:1"],
    ],
)
def test_invalid_multi_device_requests_fail_closed(devices: list[str]) -> None:
    args = runner.parser().parse_args(["--devices", *devices])
    with pytest.raises(ValueError):
        runner._validate(args)


def test_cycle_v2_basis_backend_is_forwarded_only_to_cycle_and_bound_to_config(
    tmp_path: Path,
) -> None:
    args = runner.parser().parse_args(
        [
            "--cycle-v2-basis-backend",
            "dfs_fundamental",
            "--results-root",
            str(tmp_path),
        ]
    )
    runner._validate(args)
    conductance, cycle, tree = runner.make_jobs(args, "dfs")
    assert "--basis-backend" not in conductance["command"]
    assert _option(cycle["command"], "--basis-backend") == "dfs_fundamental"
    assert "--basis-backend" not in tree["command"]
    config = runner._config_payload(args, data_root=tmp_path / "data", results_root=tmp_path)
    assert config["cycle_v2_basis_backend"] == "dfs_fundamental"


def test_v5_margin_beta_ablation_is_forwarded_only_to_conductance_and_bound_to_config(
    tmp_path: Path,
) -> None:
    args = runner.parser().parse_args(
        [
            "--v5-beta-parameterization",
            "margin_sigmoid",
            "--v5-beta-initial",
            "0.5",
            "--v5-beta-min",
            "0.05",
            "--v5-beta-max",
            "0.95",
            "--results-root",
            str(tmp_path),
        ]
    )
    runner._validate(args)
    conductance, cycle, tree = runner.make_jobs(args, "margin")
    assert _option(conductance["command"], "--v5-beta-parameterization") == "margin_sigmoid"
    assert _option(conductance["command"], "--v5-beta-initial") == "0.5"
    assert _option(conductance["command"], "--v5-beta-min") == "0.05"
    assert _option(conductance["command"], "--v5-beta-max") == "0.95"
    assert all("--v5-beta-parameterization" not in job["command"] for job in (cycle, tree))
    config = runner._config_payload(args, data_root=tmp_path / "data", results_root=tmp_path)
    assert config["v5_beta"] == {
        "beta_parameterization": "margin_sigmoid",
        "beta_initial": 0.5,
        "beta_min": 0.05,
        "beta_max": 0.95,
    }


@pytest.mark.parametrize(
    ("option", "expected_value"),
    [
        ("--v5-activation-checkpoint", True),
        ("--no-v5-activation-checkpoint", False),
    ],
)
def test_v5_activation_checkpoint_override_is_conductance_only_and_bound_to_config(
    tmp_path: Path, option: str, expected_value: bool
) -> None:
    args = runner.parser().parse_args([option, "--results-root", str(tmp_path)])
    runner._validate(args)
    conductance, cycle, tree = runner.make_jobs(args, "checkpoint")

    assert option in conductance["command"]
    opposite = "--no-v5-activation-checkpoint" if expected_value else "--v5-activation-checkpoint"
    assert opposite not in conductance["command"]
    assert all(option not in job["command"] for job in (cycle, tree))
    config = runner._config_payload(args, data_root=tmp_path / "data", results_root=tmp_path)
    assert config["v5_activation_checkpoint"] is expected_value


def test_rich_runner_rejects_margin_values_for_default_no_margin_beta():
    args = runner.parser().parse_args(["--v5-beta-max", "0.95"])
    with pytest.raises(ValueError, match="only valid for margin_sigmoid"):
        runner._validate(args)


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
    assert "scripts/telemetry_validation.py" in snapshot
    assert "scripts/verify_gpu_lock.py" in snapshot
    assert "src/chartgat/observability.py" in snapshot


def test_run_logged_appends_a_resume_attempt_to_an_existing_log(tmp_path: Path):
    log = tmp_path / "track.log"
    log.write_text("first attempt\n", encoding="utf-8")
    command = [sys.executable, "-c", "print('second attempt')"]
    assert runner._run_logged(command, log, runner._environment()) == 0
    content = log.read_text(encoding="utf-8")
    assert "first attempt" in content
    assert "=== resumed " in content
    assert "second attempt" in content


def test_run_logged_honors_preexisting_stop_request_for_its_exact_child(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    log = tmp_path / "interrupted-child.log"
    runner._STOP_ACTIVE_CHILDREN.set()
    started = time.monotonic()
    try:
        returncode = runner._run_logged(command, log, runner._environment())
    finally:
        runner._STOP_ACTIVE_CHILDREN.clear()

    assert returncode != 0
    assert time.monotonic() - started < 10
    event = json.loads(log.read_text(encoding="utf-8"))
    assert event["event"] == "owned_child_signal"
    assert event["command"] == command
    assert event["signal"] == "terminate"
    assert isinstance(event["pid"], int) and event["pid"] > 0
    assert "coordinator already requested interruption" in event["reason"]
    assert json.loads(capsys.readouterr().err) == event
    with runner._ACTIVE_CHILDREN_LOCK:
        assert not runner._ACTIVE_CHILDREN


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
            "reference",
            "large",
            "--model-seeds",
            "0",
            "--results-root",
            str(tmp_path),
            "--run-id",
            "dry",
            "--cycle-v2-basis-backend",
            "dfs_fundamental",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "2 track runs; 16 child runs; 20 fresh model trainings" in output
    assert "conductance_versions=['v1', 'v2', 'v3', 'v4', 'v5']" in output
    assert "cycle_versions=['v1', 'v2']" in output
    assert "child profiles=['reference', 'large']" in output
    assert "--profiles reference,large" in output
    assert "--basis-backend dfs_fundamental" in output
    assert "execution_classification=plan_only" in output
    assert "debug_or_smoke=false" in output
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
            "suite": "conductance_architecture_scaling_v1_v5",
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
        encodings = _options(command, "--encodings")
        conditions = [
            (version, encoding)
            for version in versions
            for encoding in (encodings if version == "v2" else [None])
        ]
        datasets = _options(command, "--datasets")
        profiles = _options(command, "--profiles")
        seeds = [int(value) for value in _option(command, "--model-seeds").split(",")]
        rows = [
            {
                "version": version,
                "encoding": encoding,
                "profile": profile,
                "model_seed": seed,
                "dataset": dataset,
            }
            for version, encoding in conditions
            for profile in profiles
            for seed in seeds
            for dataset in datasets
        ]
        aggregates = [
            {
                "version": version,
                "encoding": encoding,
                "dataset": dataset,
                "profile": profile,
                "model_seeds": sorted(seeds),
            }
            for version, encoding in conditions
            for dataset in datasets
            for profile in profiles
        ]
        selections = [
            {
                "version": version,
                "encoding": encoding,
                "dataset": dataset,
                "selected_profile": profiles[0],
                "profile_selection_id": (
                    f"{version}:{encoding}:{dataset}"
                    if encoding is not None
                    else f"{version}:{dataset}"
                ),
                "model_seeds": seeds,
                "test_used_for_selection": False,
            }
            for version, encoding in conditions
            for dataset in datasets
        ]
        selected_checkpoints = [
            {
                "version": selection["version"],
                "encoding": selection["encoding"],
                "profile_selection_id": selection["profile_selection_id"],
                "checkpoint_id": f"{selection['profile_selection_id']}:model-seed-{seed}",
                "dataset": selection["dataset"],
                "model_seed": seed,
                "selected_profile": selection["selected_profile"],
                "checkpoint": f"{selection['profile_selection_id'].replace(':', '-')}-{seed}.pt",
                "checkpoint_sha256": f"hash-{seed}",
            }
            for selection in selections
            for seed in seeds
        ]
        test_evaluations = [
            {
                **checkpoint,
                "fresh_training": False,
                "test_evaluation_id": f"test:{checkpoint['checkpoint_id']}",
            }
            for checkpoint in selected_checkpoints
        ]
        output = results_root / "cycle_pe/scaling" / run_id
        payload = {
            "status": "failed" if malformed == "status" else "passed",
            "scope": "cycle_pe_v1_v2_larger_model_scaling",
            "requested_model_seeds": seeds,
            "requested_encodings": encodings,
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
                    "encoding": encoding,
                    "dataset": dataset,
                    "model_seeds": seeds,
                    "selected_profiles": [profiles[0] for _seed in seeds],
                }
                for version, encoding in conditions
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
        "reference",
        "--model-seeds",
        "0",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path),
        "--run-id",
        "unit",
    ]


def _cycle_encoding_summary_fixture(tmp_path: Path):
    args = runner.parser().parse_args(
        [
            "--tracks",
            "cycle",
            "--cycle-versions",
            "v2",
            *_base_options(tmp_path),
        ]
    )
    runner._validate(args)
    job = runner.make_jobs(args, "encoded")[0]
    _write_summary(job["command"])
    payload = json.loads(Path(job["summary_path"]).read_text(encoding="utf-8"))
    assert runner._validate_cycle_summary(payload, job) == {"child_runs": 4, "model_trainings": 4}
    return job, payload


@pytest.mark.parametrize(
    "section",
    (
        "runs",
        "profile_aggregates",
        "profile_selections",
        "selected_checkpoints",
        "test_evaluations",
        "final_test_aggregates",
    ),
)
def test_cycle_exact_matrix_rejects_equal_count_encoding_duplicates_in_every_section(
    tmp_path, section
):
    job, payload = _cycle_encoding_summary_fixture(tmp_path)
    rows = payload[section]
    count = len(rows)
    next(row for row in rows if row["encoding"] == "pe")["encoding"] = "se"
    assert len(rows) == count
    with pytest.raises(RuntimeError, match="duplicate|matrix mismatch"):
        runner._validate_cycle_summary(payload, job)


@pytest.mark.parametrize(
    "section,field",
    (
        ("profile_selections", "profile_selection_id"),
        ("selected_checkpoints", "profile_selection_id"),
        ("selected_checkpoints", "checkpoint_id"),
        ("test_evaluations", "profile_selection_id"),
        ("test_evaluations", "checkpoint_id"),
        ("test_evaluations", "test_evaluation_id"),
    ),
)
def test_cycle_rejects_cross_encoding_selection_ids_even_with_an_exact_matrix(
    tmp_path, section, field
):
    job, payload = _cycle_encoding_summary_fixture(tmp_path)
    rows = payload[section]
    se = next(row for row in rows if row["encoding"] == "se")
    pe = next(row for row in rows if row["encoding"] == "pe" and row["dataset"] == se["dataset"])
    pe[field] = se[field]
    with pytest.raises(RuntimeError, match="invalid|wrong profile"):
        runner._validate_cycle_summary(payload, job)


def test_selected_pe_only_plan_does_not_schedule_or_certify_se(tmp_path):
    args = runner.parser().parse_args(
        [
            "--tracks",
            "cycle",
            "--cycle-versions",
            "v2",
            "--cycle-v2-encodings",
            "pe",
            *_base_options(tmp_path),
        ]
    )
    runner._validate(args)
    job = runner.make_jobs(args, "pe-only")[0]
    assert job["requested_matrix"]["encodings_by_version"] == {"v2": ["pe"]}
    assert _options(job["command"], "--encodings") == ["pe"]
    assert runner._totals([job]) == {"track_runs": 1, "child_runs": 2, "model_trainings": 2}
    _write_summary(job["command"])
    payload = json.loads(Path(job["summary_path"]).read_text(encoding="utf-8"))
    assert runner._validate_cycle_summary(payload, job)["model_trainings"] == 2
    payload["requested_encodings"] = ["se"]
    with pytest.raises(RuntimeError, match="encoding selection"):
        runner._validate_cycle_summary(payload, job)


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
        "child_runs": 61,
        "model_trainings": 63,
    }
    assert manifest["completed_counts"]["verified_model_trainings"] == 63
    assert all(job["status"] == "passed" for job in manifest["jobs"])
    assert all(len(job["result"]["summary_sha256"]) == 64 for job in manifest["jobs"])
    assert manifest["protocol"]["execution_classification"] == "final_research_training"
    assert manifest["protocol"]["debug_or_smoke_mode"] is False
    assert manifest["protocol"]["batch_selection"]["automatic_downscale"] is False
    assert manifest["protocol"]["batch_selection"]["throughput_candidate_sweep"] is False


def test_distinct_devices_run_independent_tracks_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(2, timeout=5)
    calls: list[str] = []

    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        calls.append(_option(command, "--device"))
        barrier.wait()
        _write_summary(command)
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    options = [
        "--tracks",
        "conductance",
        "cycle",
        "--devices",
        "cuda:0",
        "cuda:1",
        *_base_options(tmp_path),
    ]
    assert runner.main(options) == 0
    assert sorted(calls) == ["cuda:0", "cuda:1"]
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol"]["hardware_profile"]["cross_track_concurrency"] == 2
    assert manifest["protocol"]["hardware_profile"]["same_device_concurrency"] == 1
    assert [job["status"] for job in manifest["jobs"]] == ["passed", "passed"]


def test_keyboard_interrupt_stops_children_before_executor_wait_and_records_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_started = threading.Event()
    child_released = threading.Event()
    order: list[str] = []

    def dispatch(_command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        child_started.set()
        assert child_released.wait(timeout=5)
        order.append("child-finished")
        return 0

    def interrupt_completion(_futures: object) -> object:
        assert child_started.wait(timeout=5)
        raise KeyboardInterrupt("unit interruption")

    signal_event = {
        "event": "owned_child_signal",
        "pid": 123,
        "command": ["python", "child.py"],
        "signal": "terminate",
        "reason": "unit interruption",
        "log_path": str(tmp_path / "child.log"),
    }

    def stop_children(
        *, reason: str, original_error: BaseException | None = None
    ) -> list[dict[str, object]]:
        assert "KeyboardInterrupt" in reason
        assert isinstance(original_error, KeyboardInterrupt)
        assert str(original_error) == "unit interruption"
        order.append("stop-children")
        child_released.set()
        return [signal_event]

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    monkeypatch.setattr(runner.concurrent.futures, "as_completed", interrupt_completion)
    monkeypatch.setattr(runner, "_stop_active_children", stop_children)

    assert runner.main(["--tracks", "conductance", *_base_options(tmp_path)]) == 130
    assert order == ["stop-children", "child-finished"]
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["jobs"][0]["status"] == "failed"
    assert manifest["interruption"]["error"] == "KeyboardInterrupt: unit interruption"
    assert manifest["child_signal_events"] == [signal_event]


def test_partial_version_matrix_runs_and_validates_only_selected_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[list[str]] = []

    def dispatch(command: list[str], _log: Path, _environment: dict[str, str]) -> int:
        calls.append(command)
        _write_summary(command)
        return 0

    options = [
        "--tracks",
        "conductance",
        "cycle",
        "--conductance-versions",
        "v5",
        "--cycle-versions",
        "v2",
        *_base_options(tmp_path),
    ]
    monkeypatch.setattr(runner, "_run_logged", dispatch)
    assert runner.main(options) == 0
    assert [_options(command, "--versions") for command in calls] == [["v5"], ["v2"]]
    manifest = json.loads(
        (tmp_path / "rich_scaling/unit/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["planned_counts"] == {
        "track_runs": 2,
        "child_runs": 14,
        "model_trainings": 14,
    }
    assert manifest["completed_counts"]["verified_model_trainings"] == 14
    assert [job["requested_matrix"]["versions"] for job in manifest["jobs"]] == [
        ["v5"],
        ["v2"],
    ]


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
        ("cycle", "encoding"),
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


def test_failed_manifest_write_does_not_replace_child_failure_return_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, "_run_logged", lambda *_args: 9)
    real_atomic_write = runner._atomic_write_json

    def atomic_write(path: Path, payload: dict) -> None:
        if payload.get("status") == "failed" or any(
            job.get("status") == "failed" for job in payload.get("jobs", [])
        ):
            raise OSError("central manifest disk failure")
        real_atomic_write(path, payload)

    monkeypatch.setattr(runner, "_atomic_write_json", atomic_write)

    assert runner.main(["--tracks", "conductance", *_base_options(tmp_path)]) == 1
    assert "central manifest disk failure" in capsys.readouterr().err


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

    for changed_versions in (
        [*options, "--conductance-versions", "v5"],
        [*options, "--cycle-versions", "v2"],
        [*options, "--cycle-v2-encodings", "pe"],
    ):
        assert runner.main(changed_versions) == 2
        assert manifest_path.read_text(encoding="utf-8") == original

    payload = json.loads(original)
    payload["source_sha256"]["scripts/run_rich_scaling.py"] = "tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.main(options) == 2

    payload = json.loads(original)
    payload["source_integrity_valid"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.main(options) == 2
