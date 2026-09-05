"""CPU/debug orchestration contracts: no real GPU or research training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_rich_scaling as runner
from scripts import training_resource_plan as resources


def _args(tmp_path, *extra):
    args = runner.parser().parse_args(
        [
            "--tracks",
            "conductance",
            "cycle",
            "--conductance-versions",
            "v5",
            "--cycle-versions",
            "v2",
            "--profiles",
            "reference",
            "large",
            "--model-seeds",
            "0",
            "--hardware-profile",
            "a6000-48gb",
            "--data-root",
            str(tmp_path / "data"),
            "--results-root",
            str(tmp_path / "results"),
            "--run-id",
            "debug-contract",
            *extra,
        ]
    )
    runner._validate(args)
    return args


def test_request_uses_actual_28_job_commands_and_paired_resources(tmp_path):
    args = _args(tmp_path)
    request = runner._calibration_request(args, args.run_id)
    assert len(request["jobs"]) == 28
    assert request["source_sha256"] == resources.source_snapshot()
    groups = {}
    for job in request["jobs"]:
        key = job["track"], job["profile"], job["dataset"]
        groups.setdefault(key, set()).add(job["condition"])
        assert job["model_seed"] == 0
        assert "--epochs" in job["command"]
        assert "--resource-plan" not in job["command"]
    assert len(groups) == 14
    for key, conditions in groups.items():
        assert conditions == (
            {"fixed_c", "shared_dynamic_c"} if key[0] == "conductance" else {"se", "pe"}
        )


def test_single_cycle_arm_still_measures_paired_cost_without_extra_final_jobs(tmp_path):
    args = _args(tmp_path, "--cycle-v2-encodings", "se")
    request = runner._calibration_request(args, args.run_id)
    assert len(request["jobs"]) == 28
    assert runner._totals(runner.make_jobs(args, args.run_id))["model_trainings"] == 24


def test_failed_calibration_prevents_all_final_training(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = []

    def failed_probe(command, log, env):
        calls.append(command)
        return 1

    monkeypatch.setattr(runner, "_run_logged", failed_probe)
    run_dir = args.results_root / "rich_scaling" / args.run_id
    with pytest.raises(RuntimeError, match="no final training launched"):
        runner._ensure_measured_plan(args, args.run_id, run_dir)
    assert len(calls) == 1
    assert Path(calls[0][2]).name == "calibrate_training_resources.py"
    assert not run_dir.exists()
    assert (args.results_root / "resource_calibration" / args.run_id / "request.json").is_file()


def test_request_source_change_refuses_overwrite_before_any_probe(tmp_path, monkeypatch):
    args = _args(tmp_path)
    directory = args.results_root / "resource_calibration" / args.run_id
    directory.mkdir(parents=True)
    request = directory / "request.json"
    request.write_text('{"source_sha256": {"old": "source"}}')
    before = request.read_bytes()
    monkeypatch.setattr(runner, "_run_logged", lambda *args: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="previous measurements preserved"):
        runner._ensure_measured_plan(
            args, args.run_id, args.results_root / "rich_scaling" / args.run_id
        )
    assert request.read_bytes() == before


def test_existing_uncalibrated_training_run_is_not_reconfigured(tmp_path, monkeypatch):
    args = _args(tmp_path)
    directory = args.results_root / "rich_scaling" / args.run_id
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text('{"config": {}}')
    monkeypatch.setattr(runner, "_run_logged", lambda *args: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="existing run has no measured resource plan"):
        runner._ensure_measured_plan(args, args.run_id, directory)
    assert json.loads(manifest.read_text()) == {"config": {}}


def test_explicit_plan_checks_runtime_and_actual_dataset(tmp_path, monkeypatch):
    from scripts import calibrate_training_resources as calibration

    args = _args(tmp_path, "--resource-plan", str(tmp_path / "plan.json"))
    plan = {"_sha256": "debug", "entries": []}
    observed = []
    monkeypatch.setattr(runner, "load_resource_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        resources, "validate_plan_runtime", lambda value: observed.append("runtime")
    )

    def check_data(value, jobs):
        assert value is plan and len(jobs) == 28
        observed.append("verified_data")

    monkeypatch.setattr(calibration, "verify_plan_inputs", check_data)
    runner._ensure_measured_plan(args, args.run_id, tmp_path / "run")
    assert observed == ["runtime", "verified_data"]


def test_resolved_plan_is_forwarded_and_baseline_overrides_are_not_reapplied(tmp_path):
    args = _args(tmp_path, "--conductance-v5-ppi-batch-size", "8", "--cycle-batch-size", "64")
    args.resource_plan = tmp_path / "plan.json"
    args.resolved_resource_plan = {"_sha256": "debug", "entries": []}
    jobs = runner.make_jobs(args, args.run_id)
    assert all("--resource-plan" in job["command"] for job in jobs)
    assert "--v5-ppi-batch-size" not in jobs[0]["command"]
    assert "--batch-size" not in jobs[1]["command"]
    assert (
        runner._config_payload(args, data_root=args.data_root, results_root=args.results_root)[
            "resource_plan"
        ]["sha256"]
        == "debug"
    )


def test_dry_run_does_not_write_or_measure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "_ensure_measured_plan", lambda *args: pytest.fail("dry run must not probe")
    )
    assert runner.main(["--dry-run", "--results-root", str(tmp_path / "results")]) == 0
    assert not (tmp_path / "results").exists()
