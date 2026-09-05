"""CPU/file-fixture recovery integration only; never real GPU training or metrics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import pytest
import torch

from chartgat import resume_compat
from scripts import calibrate_training_resources as calibration
from scripts import run_conductance_scaling as conductance
from scripts import run_cycle_scaling as cycle
from scripts import run_rich_scaling as rich
from scripts import training_resource_plan as resources


def _legacy_snapshot(current):
    """Reconstruct the reviewed predecessor from the actual checked-in registry."""
    registry = json.loads(resume_compat.REGISTRY_PATH.read_bytes())
    previous = dict(current)
    for name, change in registry["changes"].items():
        if name not in previous:
            continue
        assert previous[name] == change["after"]
        if change["before"] is None:
            previous.pop(name)
        else:
            previous[name] = change["before"]
    previous.pop(resume_compat.REGISTRY_SOURCE)
    assert previous != current
    assert resume_compat.require_source_compatibility(previous, current) is not None
    return previous


def _fixture_module(name):
    """Reuse existing explicit mock-child fixtures without importing their tests."""
    spec = importlib.util.spec_from_file_location(
        f"recovery_fixture_{name}", Path(__file__).with_name(f"{name}.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rich_args(tmp_path):
    args = rich.parser().parse_args([
        "--tracks", "conductance", "cycle", "--conductance-versions", "v5",
        "--cycle-versions", "v2", "--profiles", "reference", "large",
        "--model-seeds", "0", "--hardware-profile", "a6000-48gb",
        "--data-root", str(tmp_path / "data"), "--results-root", str(tmp_path / "results"),
        "--run-id", "debug-reviewed-recovery",
    ])
    rich._validate(args)
    return args


@pytest.mark.parametrize("drift", [None, "recipe", "source"])
def test_existing_measured_plan_keeps_original_request_and_certificate_bytes(
    tmp_path, monkeypatch, drift
):
    args = _rich_args(tmp_path)
    request = rich._calibration_request(args, args.run_id)
    request["source_sha256"] = _legacy_snapshot(request["source_sha256"])
    if drift == "source":
        request["source_sha256"]["research/cycle_pe/v2/model.py"] = "0" * 64
    directory = args.results_root / "resource_calibration" / args.run_id
    directory.mkdir(parents=True)
    request_path, plan_path = directory / "request.json", directory / "resource-plan.json"
    request_path.write_text(json.dumps(request, indent=3), encoding="utf-8")
    plan_path.write_text('{"explicit_cpu_fixture_not_a_real_measurement": true}', encoding="utf-8")
    request_before, plan_before = request_path.read_bytes(), plan_path.read_bytes()
    hardware = {"debug_only": True}
    plan = {
        "_sha256": hashlib.sha256(plan_before).hexdigest(), "entries": [],
        "request_sha256": resources.digest(request), "source_sha256": request["source_sha256"],
        "hardware": {job["device"]: hardware for job in request["jobs"]},
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    events = []
    monkeypatch.setattr(calibration, "_hardware", lambda _name: hardware)
    monkeypatch.setattr(calibration, "load_resource_plan", lambda *_a, **_kw: plan)
    monkeypatch.setattr(rich, "load_resource_plan", lambda *_a, **_kw: plan)
    monkeypatch.setattr(resources, "validate_plan_runtime", lambda _plan: events.append("runtime"))

    def verify_inputs(selected_plan, jobs):
        assert selected_plan is plan and jobs == request["jobs"]
        events.append("official_input_validation")

    def verify_existing_plan(command, _log, _environment):
        assert Path(command[command.index("--request") + 1]) == request_path
        assert calibration._run_locked(request_path, directory) == plan_path
        events.append("existing_plan_returned_without_measurement")
        return 0

    monkeypatch.setattr(calibration, "verify_plan_inputs", verify_inputs)
    monkeypatch.setattr(rich, "_run_logged", verify_existing_plan)
    if drift == "recipe":
        args.model_seeds = [0, 1]
    run_dir = args.results_root / "rich_scaling" / args.run_id
    if drift is None:
        rich._ensure_measured_plan(args, args.run_id, run_dir)
        assert events == [
            "official_input_validation", "existing_plan_returned_without_measurement", "runtime"
        ]
        assert args.resolved_resource_plan["_sha256"] == hashlib.sha256(plan_before).hexdigest()
    else:
        with pytest.raises(ValueError, match="source/configuration differs"):
            rich._ensure_measured_plan(args, args.run_id, run_dir)
        assert events == []
    assert request_path.read_bytes() == request_before
    assert plan_path.read_bytes() == plan_before


@pytest.mark.parametrize("case", ["resume", "completed", "recipe", "artifact", "source"])
def test_conductance_legacy_manifest_recovery_revalidates_and_skips_completed_children(
    tmp_path, monkeypatch, case
):
    current = conductance._source_snapshot()
    previous = _legacy_snapshot(current)
    fixture = _fixture_module("test_conductance_scaling_runner")
    options, calls = fixture._stub(tmp_path, monkeypatch)
    options += ["--versions", "v5", "--model-seeds", "0"]
    monkeypatch.setattr(conductance, "_source_snapshot", lambda: previous)
    assert conductance.main(options) == 0  # Existing synthetic subprocess fixture only.
    path = tmp_path / "conductance_gat/scaling/unit-fixture/manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["status"] = "failed"
    if case == "resume":
        manifest["jobs"][0]["status"] = "failed"
    elif case == "recipe":
        manifest["config"]["epochs"] += 1
    elif case == "source":
        manifest["source_sha256"]["research/conductance_gat/v5/model.py"] = "0" * 64
    elif case == "artifact":
        metrics_path = Path(manifest["jobs"][0]["metrics_path"])
        metric = json.loads(metrics_path.read_bytes())
        metric["validation"] = 0.5
        metrics_path.write_text(json.dumps(metric), encoding="utf-8")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    unchanged_completed = Path(manifest["jobs"][-1]["metrics_path"])
    completed_before = unchanged_completed.read_bytes()
    calls.clear()
    monkeypatch.setattr(conductance, "_source_snapshot", lambda: current)
    result = conductance.main(options)
    if case in {"recipe", "artifact", "source"}:
        assert result == 1 and calls == [] and path.read_bytes() == before
    else:
        assert result == 0
        assert len(calls) == (2 if case == "resume" else 0)
        recovered = json.loads(path.read_bytes())
        assert recovered["source_sha256"] == current
        assert recovered["source_compatibility"][0]["previous_source_sha256"] == previous
        assert recovered["status"] == "passed"
    assert unchanged_completed.read_bytes() == completed_before


@pytest.mark.parametrize("artifact_changed", [False, True])
def test_cycle_legacy_failed_manifest_adoption_does_not_trust_passed_candidate_status(
    tmp_path, monkeypatch, artifact_changed
):
    args = cycle.parser().parse_args([
        "--versions", "v2", "--encodings", "se", "pe", "--profiles", "reference",
        "--datasets", "zinc12k", "--model-seeds", "0", "--data-root", str(tmp_path / "data"),
        "--results-root", str(tmp_path), "--run-id", "debug-cycle-recovery",
    ])
    cycle._validate(args)
    run_dir = tmp_path / "cycle_pe/scaling" / args.run_id
    run_dir.mkdir(parents=True)
    current = cycle._source_snapshot()
    previous = _legacy_snapshot(current)
    jobs = cycle.make_jobs(args, run_dir)
    manifest = cycle._manifest_base(args, args.run_id, run_dir, jobs, {"debug": True}, previous)
    accepted = [{"debug_certificate": "unchanged"}]
    manifest["status"] = "failed"
    jobs[0].update(status="passed", returncode=0, accepted_rows=accepted)
    jobs[1].update(status="failed", returncode=1)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    recovered, status = cycle._resume_manifest(
        args, args.run_id, run_dir, cycle.make_jobs(args, run_dir), {"debug": True}, current
    )
    assert status == "failed" and recovered["source_sha256"] == current
    assert recovered["source_compatibility"][0]["previous_source_sha256"] == previous
    reads = []

    def read_rows(job):
        reads.append(job["job_id"])
        return [{"debug_certificate": "changed"}] if artifact_changed else accepted

    monkeypatch.setattr(cycle, "read_job_rows", read_rows)
    rows = cycle._recover_candidate_rows(recovered["jobs"])
    assert reads == [jobs[0]["job_id"]]
    assert rows == ([] if artifact_changed else accepted)
    assert recovered["jobs"][0]["status"] == ("pending" if artifact_changed else "passed")
    assert recovered["jobs"][1]["status"] == "pending"
    assert path.read_bytes() == before  # Adoption changes only the returned in-memory manifest.


def test_rich_legacy_failed_manifest_revalidates_completed_track_and_continues_only_failed_work(
    tmp_path, monkeypatch
):
    fixture = _fixture_module("test_rich_scaling_runner")
    current, calls = rich._source_snapshot(), []
    previous = _legacy_snapshot(current)
    monkeypatch.setattr(rich, "_ensure_measured_plan", lambda *_args: None)
    monkeypatch.setattr(rich, "_source_snapshot", lambda: previous)
    options = ["--tracks", "conductance", "cycle", *fixture._base_options(tmp_path)]

    def first_run(command, _log, _environment):
        if Path(command[2]).name == "run_cycle_scaling.py":
            return 9
        fixture._write_summary(command)
        return 0

    monkeypatch.setattr(rich, "_run_logged", first_run)
    assert rich.main(options) == 1
    path = tmp_path / "rich_scaling/unit/manifest.json"
    old = json.loads(path.read_bytes())
    completed_path = Path(old["jobs"][0]["summary_path"])
    completed_before = completed_path.read_bytes()

    def continue_run(command, _log, _environment):
        script = Path(command[2]).name
        calls.append(script)
        if script == "run_cycle_scaling.py":
            fixture._write_summary(command)
        return 0  # Completed conductance track only re-verifies its unchanged artifacts.

    monkeypatch.setattr(rich, "_run_logged", continue_run)
    monkeypatch.setattr(rich, "_source_snapshot", lambda: current)
    assert rich.main(options) == 0
    recovered = json.loads(path.read_bytes())
    assert recovered["status"] == "passed" and recovered["resume_count"] == 1
    assert recovered["source_sha256"] == current
    assert recovered["source_compatibility"][0]["previous_source_sha256"] == previous
    assert calls == ["run_conductance_scaling.py", "run_cycle_scaling.py"]
    assert completed_path.read_bytes() == completed_before


@pytest.mark.parametrize("runner,check", [
    (conductance, "_check_sources"), (rich, "_check_central_sources")
])
def test_reviewed_resume_does_not_allow_any_source_change_mid_run(monkeypatch, runner, check):
    current = runner._source_snapshot()
    previous = _legacy_snapshot(current)
    manifest = {"source_sha256": previous, "source_integrity_valid": True}
    resume_compat.adopt_source_snapshot(manifest, current)
    monkeypatch.setattr(runner, "_source_snapshot", lambda: current)
    getattr(runner, check)(manifest)
    # Even the reviewed transition cannot be applied during an active run.
    monkeypatch.setattr(runner, "_source_snapshot", lambda: previous)
    with pytest.raises(RuntimeError, match="source changed"):
        getattr(runner, check)(manifest)
    assert manifest["source_integrity_valid"] is False
