"""CPU-only orchestration tests; mocks are not real GPU calibration or training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.conductance_gat import edge_selection
from research.conductance_gat.edge_selection import calibration, protocol
from research.conductance_gat.edge_selection import train as new_train
from scripts import run_v5_edge_selection as driver


def args(*extra):
    return driver.parser().parse_args(["--run-id", "debug-edge", *extra])


def parse_child(job):
    parser = new_train.build_parser()
    child = parser.parse_args(job["command"][5:])
    new_train.validate_args(child)
    return child


def test_default_matrix_retains_full_scale(tmp_path):
    selected = args()
    driver.validate_args(selected)
    planned = driver.make_jobs(selected, tmp_path / "not-created")
    assert len(driver.variants(selected)) == 13 and len(planned) == 130
    assert {job["dataset"] for job in planned} == {
        "cora",
        "citeseer",
        "pubmed",
        "ppi",
        "ogbn-arxiv",
    }
    assert {job["model_seed"] for job in planned} == {0}
    assert len({job["output_dir"] for job in planned}) == 130
    for job in planned:
        child = parse_child(job)
        assert job["command"][4] == driver.TRAIN_MODULE
        assert (child.hidden_channels, child.layers, child.heads) == (
            (256, 8, 8) if job["profile"] == "reference" else (384, 12, 8)
        )
        assert (child.epochs, child.patience) == (200, 50)
        assert child.learning_budget_policy == "reference_updates"
        assert child.conductance_heads == "per_head" and child.propagation_normalization == "row"
        assert child.condition == "shared_dynamic_c" and child.training_schedule == "joint"
        assert child.activation_checkpoint is True
        if child.selection_mode in protocol.BUDGET_MODES:
            assert child.chord_fraction in {0.25, 0.5, 0.75}
        else:
            assert "--chord-fraction" not in job["command"]
    assert list(tmp_path.iterdir()) == []


def test_structure_and_corruption_axes_are_distinct():
    variants = driver.variants(args())
    structures = [item for item in variants if item["suite"] == "structure"]
    corrupted = [item for item in variants if item["suite"] == "corruption"]
    assert len(structures) == 11 and len(corrupted) == 2
    for fraction in (0.25, 0.5, 0.75):
        group = [item for item in structures if item["configuration"]["chord_fraction"] == fraction]
        assert {item["configuration"]["selection_mode"] for item in group} == protocol.BUDGET_MODES
    assert all(item["configuration"]["corruption_ratio"] == 0 for item in structures)
    first, second = [copy.deepcopy(item["configuration"]) for item in corrupted]
    assert first["selection_mode"] == second["selection_mode"] == "hard_concrete"
    assert first.pop("negative_loss_weight") == 0 and second.pop("negative_loss_weight") == 1
    assert first == second and first["l0_weight"] == 1e-4


def test_nondefault_parameters_are_scoped_to_their_experiment_axis(tmp_path):
    selected = args("--forest-seed", "7", "--corruption-seed", "11", "--gate-temperature", "0.8")
    jobs = driver.make_jobs(selected, tmp_path)
    assert len(jobs) == 130
    for job in jobs:
        child = parse_child(job)
        if child.selection_mode == "hard_concrete":
            assert (child.forest_seed, child.corruption_seed, child.gate_temperature) == (
                0,
                11,
                0.8,
            )
        else:
            assert (child.forest_seed, child.corruption_seed, child.gate_temperature) == (
                7,
                0,
                2 / 3,
            )
    assert list(tmp_path.iterdir()) == []


def test_endpoint_controls_are_deduplicated():
    selected = args("--suites", "structure", "--chord-fractions", "0", "0.5", "1")
    driver.validate_args(selected)
    assert len(driver.variants(selected)) == 5


def test_fraction_ids_preserve_distinct_float_budgets():
    selected = args("--suites", "structure", "--chord-fractions", "0.2500001", "0.2500002")
    variants = driver.variants(selected)
    assert len(variants) == len({item["variant_id"] for item in variants}) == 8


@pytest.mark.parametrize(
    "options",
    [
        ["--chord-fractions", "-0.1"],
        ["--chord-fractions", "nan"],
        ["--negative-loss-weight", "0"],
        ["--corruption-ratio", "0"],
        ["--model-seeds", "0", "0"],
        ["--device", "cpu"],
    ],
)
def test_invalid_recipe_refused(options):
    with pytest.raises(ValueError):
        driver.validate_args(args(*options))


def test_dry_plan_does_not_write_or_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        driver.standalone,
        "check_dependencies",
        lambda: pytest.fail("dry run queried training environment"),
    )
    monkeypatch.setattr(
        driver, "_ensure_calibration", lambda *_: pytest.fail("dry run measured GPU")
    )
    assert (
        driver.main(["--run-id", "debug-plan", "--results-root", str(tmp_path), "--dry-run"]) == 0
    )
    text = capsys.readouterr().out
    assert "13 independent arms; 130 full-size trainings" in text
    assert "old V5 runs are untouched" in text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "override",
    [
        ["--chord-fractions", "0.5"],
        ["--epochs", "201"],
        ["--negative-loss-weight", "2"],
        ["--corruption-seed", "1"],
        ["--suites", "structure"],
    ],
)
def test_recipe_change_refuses_resume(tmp_path, override):
    old = args()
    jobs = driver.make_jobs(old, tmp_path)
    manifest = {
        "schema_version": 1,
        "suite": driver.SUITE,
        "run_id": old.run_id,
        "config": driver._config(old),
        "source_sha256": {},
        "dependencies": {},
        "planned_jobs": jobs,
    }
    text = json.dumps(manifest)
    path = SimpleNamespace(read_text=lambda **_: text)
    with pytest.raises(ValueError, match="identity changed|matrix"):
        driver._resume(path, args(*override), driver.make_jobs(args(*override), tmp_path), {}, {})
    assert list(tmp_path.iterdir()) == []


def test_source_map_includes_new_and_old_dependencies():
    snapshot = driver.resources.source_snapshot()
    for relative in (
        "scripts/run_v5_edge_selection.py",
        "research/conductance_gat/edge_selection/calibration.py",
        "scripts/run_conductance_v5.py",
        "scripts/run_v5_mechanism_experiments.py",
    ):
        assert snapshot[relative] == driver.common._file_sha(driver.ROOT / relative)


def test_actual_audit_cli_contract(tmp_path, monkeypatch, capsys):
    from research.conductance_gat.edge_selection import audit

    received = []

    def inspect(*values):
        received.append(values)
        return {
            "validation": {"metric": 0.5},
            "clean_validation": {"metric": 0.5},
            "diagnostics": {"interventions": {}},
            "synthetic_fixture": True,
        }

    monkeypatch.setattr(audit, "audit", inspect)
    options = [
        "--root",
        str(tmp_path),
        "--data-root",
        str(tmp_path / "data"),
        "--device",
        "cuda:0",
        "--repeat-evaluations",
        "5",
    ]
    assert audit.main(options) == 0
    assert received[0][0] == tmp_path and received[0][3:] == (5, 32)
    assert "no test evaluated" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def fake_result():
    return {
        "validation": 0.5,
        "best_epoch": 1,
        "shared_initial_state_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "learning_budget": {"explicit_synthetic_fixture": True},
        "checkpoint_sha256": "d" * 64,
    }


def test_audit_retry_preserves_training(tmp_path, monkeypatch):
    options = [
        "--run-id",
        "debug-a",
        "--results-root",
        str(tmp_path),
        "--datasets",
        "cora",
        "--profiles",
        "reference",
        "--suites",
        "corruption",
    ]
    monkeypatch.setattr(driver.standalone, "check_dependencies", lambda: {"synthetic": True})
    monkeypatch.setattr(driver.resources, "source_snapshot", lambda: {"synthetic": "stable"})
    monkeypatch.setattr(driver, "_ensure_calibration", lambda *_: None)
    monkeypatch.setattr(driver, "_read_result", lambda _: fake_result())
    training, audits = [], []

    def dispatch(command, log, *_):
        if "--root" in command:
            audits.append(command.copy())
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("explicit synthetic audit fixture", encoding="utf-8")
            return 8 if len(audits) == 1 else 0
        training.append(command.copy())
        return 0

    monkeypatch.setattr(driver.standalone.shared, "run_logged", dispatch)
    assert driver.main(options) == 1
    path = tmp_path / "conductance_gat/edge_selection/debug-a/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    first = manifest["jobs"][0]
    assert first["status"] == "passed" and first["audit"]["status"] == "failed"
    failed_log = Path(first["audit"]["log_path"])
    failed_bytes = failed_log.read_bytes()
    assert driver.main(options) == 0
    assert len(training) == 2 and len(audits) == 3
    assert failed_log.read_bytes() == failed_bytes
    assert driver.main(options) == 0 and len(training) == 2 and len(audits) == 3


def calibration_fixture(monkeypatch, tmp_path):
    selected = args("--datasets", "ogbn-arxiv", "--profiles", "reference", "--suites", "corruption")
    jobs = driver.make_jobs(selected, tmp_path)
    parsed = {job["variant_id"]: parse_child(job) for job in jobs}
    fake_train = SimpleNamespace(
        configuration=lambda value: vars(value).copy(),
        load_calibration_payload=lambda _: (
            {"synthetic": True},
            {"data_sha256": "a" * 64},
            4096,
            "sampled_seed_nodes",
        ),
    )
    monkeypatch.setattr(edge_selection, "train", fake_train, raising=False)
    monkeypatch.setattr(
        calibration, "parse_job", lambda job: copy.deepcopy(parsed[job["variant_id"]])
    )
    monkeypatch.setattr(calibration.resources, "allocated_cpu_count", lambda: 8)
    calls = []

    def measure(job, loaded, child, batch, workers):
        calls.append((job["variant_id"], batch, workers))
        epochs = 5
        physical = calibration._probe_args(child, "sampled_seed_nodes", batch, workers)
        count = (4096 + batch - 1) // batch
        report = {
            "status": "passed",
            "condition": job["variant_id"],
            "model_seed": job["model_seed"],
            "samples_per_second": 4096 * epochs / 5,
            "processed_units": 4096 * epochs,
            "elapsed_seconds": 5.0,
            "optimizer_steps": epochs * count,
            "measurement_steps_requested": 5,
            "warmup_steps_requested": 2,
            "minimum_measure_seconds_requested": 3.0,
            "complete_measurement_epochs": epochs,
            "optimizer_state_bytes": 1024,
            "peak_allocated_bytes": 8 * 1024**3,
            "peak_reserved_bytes": 10 * 1024**3,
            "total_memory_bytes": 48 * 1024**3,
            "free_bytes_before": 46 * 1024**3,
            "unit": "synthetic_supervised_seed_nodes",
            "batch_size": batch,
            "workers": workers,
            "configuration": vars(physical).copy(),
            "validation_completed": True,
            "validation_seconds": 1.0,
            "topology_preparation_seconds": 1.0,
            "setup_seconds": 1.0,
            "auxiliary_path_measured": True,
            "required_auxiliary_path": child.negative_loss_weight > 0,
            "required_cycle_preparation": child.selection_mode == "forest_cycle",
        }
        if batch == 4096 and child.negative_loss_weight > 0:
            report["validation_seconds"] = 20.0
        return report

    monkeypatch.setattr(calibration, "_measure", measure)
    return jobs, calls


def test_ppi_candidate_preserves_measured_worker_provenance(tmp_path):
    selected = args("--datasets", "ppi", "--profiles", "reference", "--suites", "corruption")
    child = parse_child(driver.make_jobs(selected, tmp_path)[0])
    candidate = calibration._probe_args(child, "graphs", child.batch_size, 2)
    assert candidate.workers == 2
    assert candidate.worker_configuration_source == "measured_batch_calibration_candidate"
    assert new_train.configuration(candidate)["worker_configuration_source"] == (
        "measured_batch_calibration_candidate"
    )


def test_interrupted_training_resumes_only_incomplete_arm(tmp_path, monkeypatch):
    options = [
        "--run-id",
        "debug-r",
        "--results-root",
        str(tmp_path),
        "--datasets",
        "cora",
        "--profiles",
        "reference",
        "--suites",
        "corruption",
    ]
    monkeypatch.setattr(driver.standalone, "check_dependencies", lambda: {"synthetic": True})
    monkeypatch.setattr(driver.resources, "source_snapshot", lambda: {"synthetic": "stable"})
    monkeypatch.setattr(driver, "_ensure_calibration", lambda *_: None)
    monkeypatch.setattr(driver, "_read_result", lambda _: fake_result())
    calls = []

    def dispatch(command, log, *_):
        if "--root" in command:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("explicit synthetic audit fixture", encoding="utf-8")
            return 0
        calls.append(command.copy())
        if len(calls) == 2:
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True)
            (output / "last.pt").write_bytes(b"explicit synthetic resume-control fixture")
            return 9
        return 0

    monkeypatch.setattr(driver.standalone.shared, "run_logged", dispatch)
    assert driver.main(options) == 1
    assert driver.main(options) == 0 and len(calls) == 3
    assert "--resume" in calls[-1]
    assert driver.main(options) == 0 and len(calls) == 3


def test_common_calibration_includes_validation_cost(tmp_path, monkeypatch):
    jobs, calls = calibration_fixture(monkeypatch, tmp_path)
    entry = {}
    calibration.calibrate_group(jobs, entry, lambda: None)
    assert len(calls) == 4
    assert {batch for _, batch, _ in calls} == {2048, 4096}
    # Larger batch is faster per optimizer update, but full validation cost makes it worse.
    assert entry["selected"]["sample_seed_batch_size"] == 2048
    resolved = driver.common._apply_common_resources(jobs, [entry])
    assert {job["execution"]["sample_seed_batch_size"] for job in resolved} == {2048}
    assert list(tmp_path.iterdir()) == []


def test_completed_calibration_is_verified_without_new_measurement(tmp_path, monkeypatch):
    jobs, calls = calibration_fixture(monkeypatch, tmp_path)
    entry = {}
    calibration.calibrate_group(jobs, entry, lambda: None)
    prior, measured = copy.deepcopy(entry), list(calls)
    calibration.calibrate_group(
        jobs, entry, lambda: pytest.fail("completed measurement unexpectedly changed")
    )
    assert entry == prior and calls == measured


@pytest.mark.parametrize(
    "damage",
    ["missing_arm", "validation", "auxiliary", "config", "selected", "boundary", "workers"],
)
def test_calibration_rejects_partial_evidence(tmp_path, monkeypatch, damage):
    jobs, _ = calibration_fixture(monkeypatch, tmp_path)
    entry = {}
    calibration.calibrate_group(jobs, entry, lambda: None)
    if damage == "missing_arm":
        entry["candidates"][0]["measurements"].pop()
    elif damage == "validation":
        entry["candidates"][0]["measurements"][0]["validation_completed"] = False
    elif damage == "auxiliary":
        entry["candidates"][0]["measurements"][1]["auxiliary_path_measured"] = False
    elif damage == "config":
        entry["candidates"][0]["measurements"][0]["configuration"]["epochs"] = 10
    elif damage == "selected":
        entry["selected"]["sample_seed_batch_size"] = 4096
    elif damage == "boundary":
        entry["stop_reason"] = "memory_headroom_boundary"
    else:
        entry["worker_candidates"] = [0, 1]
    with pytest.raises(ValueError):
        calibration.validate_entry(entry, jobs)
