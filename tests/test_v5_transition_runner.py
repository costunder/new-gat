"""CPU-only migration orchestration fixtures; no GPU or research training."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.conductance_gat.v5 import train
from research.conductance_gat.v5.protocol import conductance_configuration
from scripts import run_conductance_scaling as scaling
from scripts import run_v5_transition as runner


def _value(command, name):
    return command[command.index(name) + 1]


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    args = scaling.parser().parse_args(
        [
            "--versions",
            "v5",
            "--profiles",
            "reference",
            "large",
            "--datasets",
            "ogbn-arxiv",
            "cora",
            "--hardware-profile",
            "a6000-48gb",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    jobs = scaling.make_jobs(args, source)
    infos = {}
    for job in jobs:
        legacy_command = list(job["command"])
        for name in (*conductance_configuration(), "training_schedule"):
            option = "--" + name.replace("_", "-")
            index = legacy_command.index(option)
            del legacy_command[index : index + 2]
            job["architecture"].pop(name)
        job["command"] = legacy_command
        output = Path(job["output_dir"])
        output.mkdir(parents=True)
        arxiv = job["dataset"] == "ogbn-arxiv"
        job["status"] = "passed" if arxiv else "pending"
        if arxiv:
            epoch = (
                160 if (job["profile"], job["condition"]) == ("large", "shared_dynamic_c") else 200
            )
            if epoch == 160:
                job["status"] = "failed"
            checkpoint = output / "last.pt"
            checkpoint.write_bytes(("CPU-debug-" + job["job_id"]).encode())
            source_args, _ = runner._training_arguments(legacy_command, legacy=True)
            config = train.configuration(source_args)
            for name in (*conductance_configuration(), "training_schedule"):
                config.pop(name)
            infos[str(checkpoint)] = {
                "saved": {},
                "sha256": runner._sha(checkpoint),
                "source_epoch": epoch,
                "source_complete": epoch == 200,
                "legacy_revision": "8963821",
                "identity": {
                    "dataset": job["dataset"],
                    "condition": job["condition"],
                    "configuration": config,
                    "dataset_protocol": {"data_sha256": "a" * 64},
                    "runtime_versions": {"test": "CPU debug fixture"},
                },
            }
            (output / "metrics.json").write_text(json.dumps({"status": job["status"]}))
    manifest = {
        "schema_version": 1,
        "suite": "conductance_architecture_scaling_v1_v5",
        "run_id": "source",
        "status": "failed",
        "config": {"min_free_gb": 40.0},
        "jobs": jobs,
    }
    path = source / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"debug_fixture.py": "b" * 64})
    monkeypatch.setattr(runner, "_inspect_checkpoint", lambda path: copy.deepcopy(infos[str(path)]))

    def verify(job, manifest, path):
        assert job["status"] == "passed"
        return {"role": "historical_reference", "status": "verified", "job_id": job["job_id"]}

    monkeypatch.setattr(runner, "_historical_reference", verify)
    return path, manifest, infos


def _args(source_run, tmp_path, *extra):
    return runner.parser().parse_args(
        [
            "--source-manifest",
            str(source_run[0]),
            "--output-dir",
            str(tmp_path / "transition"),
            *extra,
        ]
    )


def test_complete_controls_preserved_large160_transitions_only_remaining40(source_run, tmp_path):
    plan = runner.build_plan(_args(source_run, tmp_path))
    assert len(plan["jobs"]) == 8
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    dynamic = jobs["v5/large/model-seed-0/ogbn-arxiv/shared_dynamic_c"]
    assert dynamic["action"] == "transition_dynamic"
    assert dynamic["source_epoch"] == 160 and dynamic["remaining_epochs"] == 40
    assert _value(dynamic["command"], "--epochs") == "200"
    assert _value(dynamic["command"], "--hidden-channels") == "384"
    assert _value(dynamic["command"], "--layers") == "12"
    assert _value(dynamic["command"], "--sample-seed-batch-size") == "2048"
    assert _value(dynamic["command"], "--transition-mode") == "replace_c"
    assert _value(dynamic["command"], "--conductance-backend") == "optimization"
    assert _value(dynamic["command"], "--training-schedule") == "joint"
    control = jobs["v5/large/model-seed-0/ogbn-arxiv/fixed_c"]
    assert control["action"] == "reuse_completed_fixed" and control["remaining_epochs"] == 0
    assert control["historical_reference"]["status"] == "verified"
    old_dynamic = jobs["v5/reference/model-seed-0/ogbn-arxiv/shared_dynamic_c"]
    assert old_dynamic["action"] == "preserve_legacy_dynamic"
    assert old_dynamic["requires_extra_epoch_budget"] is True
    assert sum(job["action"].startswith("fresh_") for job in plan["jobs"]) == 4
    assert not (tmp_path / "transition").exists()


def test_per_job_extra_budget_does_not_extend_other_large_job(source_run, tmp_path):
    reference = "v5/reference/model-seed-0/ogbn-arxiv/shared_dynamic_c"
    plan = runner.build_plan(_args(source_run, tmp_path, "--extra-epochs-for", reference + "=30"))
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    assert jobs[reference]["action"] == "transition_dynamic"
    assert jobs[reference]["target_total_epochs"] == 230
    assert _value(jobs[reference]["command"], "--transition-extra-epochs") == "30"
    assert jobs["v5/large/model-seed-0/ogbn-arxiv/shared_dynamic_c"]["target_total_epochs"] == 200


@pytest.mark.parametrize(
    "overrides",
    [
        ["unknown=10"],
        ["v5/reference/model-seed-0/ogbn-arxiv/shared_dynamic_c=-2"],
        ["v5/reference/model-seed-0/ogbn-arxiv/shared_dynamic_c=1.5"],
        ["malformed"],
        ["v5/reference/model-seed-0/ogbn-arxiv/shared_dynamic_c=10"] * 2,
    ],
)
def test_bad_extra_budget_overrides_fail_before_writes(source_run, tmp_path, overrides):
    options = [value for item in overrides for value in ("--extra-epochs-for", item)]
    with pytest.raises(ValueError):
        runner.build_plan(_args(source_run, tmp_path, *options))
    assert not (tmp_path / "transition").exists()


def test_plan_only_is_json_and_does_not_create_files_or_probe(
    source_run, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        runner, "execute", lambda *_args, **_kwargs: pytest.fail("plan launched execution")
    )
    assert (
        runner.main(
            [
                "--source-manifest",
                str(source_run[0]),
                "--output-dir",
                str(tmp_path / "transition"),
                "--plan-only",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["read_only"] is True and result["gpu_work_launched"] is False
    assert len(result["jobs"]) == 8
    assert not (tmp_path / "transition").exists()


def test_rich_source_resolves_only_conductance_child(source_run, tmp_path):
    rich = tmp_path / "rich.json"
    rich.write_text(
        json.dumps(
            {
                "suite": "rich_scaling",
                "run_id": "rich",
                "jobs": [
                    {"track": "cycle", "output_dir": "do-not-open"},
                    {"track": "conductance", "output_dir": str(source_run[0].parent)},
                    {"track": "tree", "output_dir": "do-not-open-either"},
                ],
            }
        )
    )
    args = _args(source_run, tmp_path)
    args.source_manifest = rich
    plan = runner.build_plan(args)
    assert plan["source_manifest"] == str(source_run[0])
    assert all(job["job_id"].startswith("v5/") for job in plan["jobs"])


def test_corrupt_source_checkpoint_aborts_whole_plan(source_run, tmp_path, monkeypatch):
    def broken(_path):
        raise ValueError("corrupt source checkpoint")

    monkeypatch.setattr(runner, "_inspect_checkpoint", broken)
    with pytest.raises(ValueError, match="corrupt source"):
        runner.build_plan(_args(source_run, tmp_path))
    assert not (tmp_path / "transition").exists()


def test_source_mutation_and_new_pending_checkpoint_are_detected(source_run, tmp_path):
    plan = runner.build_plan(_args(source_run, tmp_path))
    pending = next(job for job in plan["jobs"] if job["action"] == "fresh_dynamic")
    new_checkpoint = Path(pending["source_output_dir"]) / "last.pt"
    new_checkpoint.write_bytes(b"source-still-running-debug")
    with pytest.raises(ValueError, match="source artifact changed"):
        runner._verify_sources(plan)


@pytest.mark.parametrize("where", ["source", "inside", "ancestor", "data"])
def test_source_and_data_overlap_are_rejected(source_run, tmp_path, where):
    args = _args(source_run, tmp_path)
    args.output_dir = {
        "source": source_run[0].parent,
        "inside": source_run[0].parent / "new",
        "ancestor": tmp_path,
        "data": tmp_path / "data" / "new",
    }[where]
    with pytest.raises(ValueError, match="distinct|overlap"):
        runner.build_plan(args)


def test_execution_requires_confirmation_and_refuses_active_original(
    source_run, tmp_path, monkeypatch
):
    plan = runner.build_plan(_args(source_run, tmp_path))
    with pytest.raises(ValueError, match="confirm-source-stopped"):
        runner.execute(plan, confirm_source_stopped=False)
    monkeypatch.setattr(
        runner, "_active_source_processes", lambda _plan: [{"pid": 123, "command": ["debug"]}]
    )
    with pytest.raises(ValueError, match="still active"):
        runner.execute(plan, confirm_source_stopped=True)
    assert not (tmp_path / "transition").exists()


def _fake_execution(monkeypatch, *, fail_once=False):
    calls = []
    failed = False
    monkeypatch.setattr(runner, "_active_source_processes", lambda _plan: [])
    monkeypatch.setattr(
        runner,
        "_validate_certificate",
        lambda job, _path: {"selected_execution": dict(job["baseline_execution"])},
    )
    monkeypatch.setattr(runner, "_write_report", lambda _manifest, _output: None)
    monkeypatch.setattr(
        runner,
        "_completed",
        lambda job: {"debug_checkpoint_sha256": runner._sha(Path(job["output_dir"]) / "last.pt")},
    )

    def dispatch(command, _log):
        nonlocal failed
        calls.append(command)
        if "--probe-job" in command:
            path = Path(_value(command, "--probe-job"))
            (path.parent / "resource-certificate.json").write_text('{"debug_only":true}')
            return 0
        output = Path(_value(command, "--output-dir"))
        output.mkdir(parents=True, exist_ok=True)
        if fail_once and not failed:
            failed = True
            (output / "last.pt").write_bytes(b"debug-interrupted-new-checkpoint")
            return 7
        (output / "last.pt").write_bytes(b"debug-completed-new-checkpoint")
        return 0

    monkeypatch.setattr(runner, "_run_logged", dispatch)
    return calls


def test_incomplete_transition_resumes_and_keeps_finished_controls_and_budget_pending(
    source_run, tmp_path, monkeypatch
):
    plan = runner.build_plan(_args(source_run, tmp_path))
    originals = {
        path: Path(path).read_bytes()
        for path, value in plan["source_artifact_sha256"].items()
        if value
    }
    calls = _fake_execution(monkeypatch, fail_once=True)
    with pytest.raises(RuntimeError, match="code 7"):
        runner.execute(plan, confirm_source_stopped=True)
    assert runner.execute(plan, confirm_source_stopped=True) == 3
    training = [command for command in calls if "-m" in command]
    assert len(training) == 6  # One interrupted attempt, five eligible final children.
    assert all("--resume" in command for command in training)
    assert all(
        _value(command, "--output-dir").startswith(str(tmp_path / "transition"))
        for command in training
    )
    assert not any(
        "reference/model-seed-0/ogbn-arxiv" in _value(command, "--output-dir").replace("\\", "/")
        for command in training
    )
    before = len(calls)
    assert runner.execute(plan, confirm_source_stopped=True) == 3
    assert len(calls) == before
    for path, contents in originals.items():
        assert Path(path).read_bytes() == contents
    manifest = runner._read(Path(plan["output_dir"]) / "manifest.json")
    assert manifest["status"] == "pending_extra_budget"
    assert sum(job["status"] == "historical_reference" for job in manifest["jobs"]) == 2
    assert sum(job["status"] == "passed" for job in manifest["jobs"]) == 5


def test_changed_completed_target_is_rejected_without_retraining(source_run, tmp_path, monkeypatch):
    plan = runner.build_plan(_args(source_run, tmp_path))
    calls = _fake_execution(monkeypatch)
    assert runner.execute(plan, confirm_source_stopped=True) == 3
    completed = next(job for job in plan["jobs"] if job["action"] == "transition_dynamic")
    (Path(completed["output_dir"]) / "last.pt").write_bytes(b"tampered-debug-target")
    before = len(calls)
    with pytest.raises(ValueError, match="completed transition child changed"):
        runner.execute(plan, confirm_source_stopped=True)
    assert len(calls) == before


def test_incomplete_fixed_keeps_legacy_schedule_and_exact_execution(source_run, tmp_path):
    path, source, infos = source_run
    job = next(
        job
        for job in source["jobs"]
        if job["dataset"] == "ogbn-arxiv" and job["condition"] == "fixed_c"
    )
    job["status"] = "failed"
    info = infos[str(Path(job["output_dir"]) / "last.pt")]
    info["source_epoch"], info["source_complete"] = 80, False
    path.write_text(json.dumps(source), encoding="utf-8")
    plan = runner.build_plan(_args(source_run, tmp_path))
    continuation = next(item for item in plan["jobs"] if item["job_id"] == job["job_id"])
    assert continuation["action"] == "resume_incomplete_fixed"
    assert continuation["source_epoch"] == 80 and continuation["remaining_epochs"] == 120
    assert _value(continuation["command"], "--training-schedule") == "staged"
    assert _value(continuation["command"], "--conductance-backend") == "mlp"
    assert _value(continuation["command"], "--transition-mode") == "continue_fixed"


def test_explicit_pending_budget_revision_keeps_all_completed_children_untouched(
    source_run, tmp_path, monkeypatch
):
    original_plan = runner.build_plan(_args(source_run, tmp_path))
    calls = _fake_execution(monkeypatch)
    assert runner.execute(original_plan, confirm_source_stopped=True) == 3
    manifest_path = Path(original_plan["output_dir"]) / "manifest.json"
    old_manifest = runner._read(manifest_path)
    completed_bytes = {
        job["output_dir"]: (Path(job["output_dir"]) / "last.pt").read_bytes()
        for job in old_manifest["jobs"]
        if job["status"] == "passed"
    }
    job_id = next(
        job["job_id"] for job in original_plan["jobs"] if job["requires_extra_epoch_budget"]
    )
    revised_plan = runner.build_plan(
        _args(source_run, tmp_path, "--extra-epochs-for", f"{job_id}=30")
    )
    before = len(calls)
    assert runner.execute(revised_plan, confirm_source_stopped=True) == 0
    additional_training = [command for command in calls[before:] if "-m" in command]
    assert len(additional_training) == 1
    assert _value(additional_training[0], "--epochs") == "230"
    assert _value(additional_training[0], "--transition-extra-epochs") == "30"
    for output, expected in completed_bytes.items():
        assert (Path(output) / "last.pt").read_bytes() == expected
    revised_manifest = runner._read(manifest_path)
    revision = revised_manifest["plan_revisions"][0]
    assert revision["previous_plan"] == original_plan
    assert revision["old_plan_sha256"] == runner.digest(original_plan)
    assert revision["new_plan_sha256"] == runner.digest(revised_plan)
    assert revision["changed_job_ids"] == [job_id]
    assert revised_manifest["status"] == "passed"
    before = len(calls)
    assert runner.execute(revised_plan, confirm_source_stopped=True) == 0
    assert len(calls) == before
    # Removing an activated budget is not a resume; refuse without writes/dispatch.
    manifest_bytes = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="cannot change or remove prior explicit budgets"):
        runner.execute(original_plan, confirm_source_stopped=True)
    assert len(calls) == before and manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize("drift", ["checkpoint", "code", "trained_budget"])
def test_pending_budget_revision_refuses_existing_target_or_unrelated_drift(
    source_run, tmp_path, monkeypatch, drift
):
    original = runner.build_plan(_args(source_run, tmp_path))
    calls = _fake_execution(monkeypatch)
    assert runner.execute(original, confirm_source_stopped=True) == 3
    pending = next(job for job in original["jobs"] if job["requires_extra_epoch_budget"])
    options = ["--extra-epochs-for", f"{pending['job_id']}=30"]
    if drift == "trained_budget":
        trained = next(job for job in original["jobs"] if job["action"] == "transition_dynamic")
        options += ["--extra-epochs-for", f"{trained['job_id']}=10"]
    revised = runner.build_plan(_args(source_run, tmp_path, *options))
    if drift == "checkpoint":
        target = Path(pending["output_dir"])
        target.mkdir(parents=True)
        (target / "last.pt").write_bytes(b"debug-unexpected-state")
    elif drift == "code":
        revised["source_sha256"] = {"debug": "different-implementation"}
    manifest_path = Path(original["output_dir"]) / "manifest.json"
    before, contents = len(calls), manifest_path.read_bytes()
    with pytest.raises(ValueError):
        runner.execute(revised, confirm_source_stopped=True)
    assert len(calls) == before and manifest_path.read_bytes() == contents


def _measurement(job, size, workers):
    gib = 1024**3
    return {
        "status": "passed",
        "condition": job["condition"],
        "model_seed": job["model_seed"],
        "batch_size": size,
        "workers": workers,
        "samples_per_second": float(size),
        "elapsed_seconds": 5.0,
        "processed_units": size * 5,
        "optimizer_steps": 5,
        "measurement_steps_requested": 5,
        "warmup_steps_requested": 2,
        "minimum_measure_seconds_requested": 3.0,
        "optimizer_state_bytes": gib,
        "peak_reserved_bytes": 8 * gib,
        "total_memory_bytes": 48 * gib,
        "free_bytes_before": 45 * gib,
        "peak_allocated_bytes": 7 * gib,
        "unit": "seed_nodes",
    }


@pytest.mark.parametrize("scenario", ["sampled", "sampled_oom", "full_graph", "fixed_full_graph"])
def test_real_probe_path_sweeps_new_solver_batches_and_certificate_binds_recipe(
    source_run, tmp_path, monkeypatch, scenario
):
    import torch

    from research.conductance_gat.v5 import batch_calibration, train, transition_training
    from scripts import calibrate_training_resources as calibration
    from scripts import training_resource_plan

    plan = runner.build_plan(_args(source_run, tmp_path))
    job = next(job for job in plan["jobs"] if job["action"] == "transition_dynamic")
    full_graph = "full_graph" in scenario
    fixed = scenario == "fixed_full_graph"
    if full_graph:
        job["command"] = runner._set_option(job["command"], "--sampling", "full")
    if fixed:
        job["action"], job["condition"] = "resume_incomplete_fixed", "fixed_c"
        for option, value in (
            ("--condition", "fixed_c"),
            ("--transition-mode", "continue_fixed"),
            ("--conductance-backend", "mlp"),
            ("--training-schedule", "staged"),
        ):
            job["command"] = runner._set_option(job["command"], option, value)
    job["source_dataset_protocol"]["split_counts"] = {"train": 8192}
    directory = Path(job["resource_directory"])
    directory.mkdir(parents=True)
    path = directory / "job.json"
    path.write_text(json.dumps(job))
    args, _ = runner._training_arguments(job["command"])
    runtime, hardware = {"unit": "runtime"}, {"total_memory_bytes": 48 * 1024**3}
    monkeypatch.setattr(
        runner, "_probe_runtime", lambda _job: (copy.deepcopy(args), runtime, hardware)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args: (45 * 1024**3, 48 * 1024**3))
    monkeypatch.setattr(
        batch_calibration,
        "load_calibration_payload",
        lambda _args: (
            {"splits": {"train": torch.ones(8192, dtype=torch.bool)}},
            job["source_dataset_protocol"],
        ),
    )
    observed = []

    def measure(probe_job, _payload, candidate_args, *, batch_size, workers):
        assert candidate_args.conductance_backend == ("mlp" if fixed else "optimization")
        # Existing production candidate builder must retain the unused PPI setting.
        checked = batch_calibration._candidate_args(candidate_args, batch_size, workers)
        if full_graph:
            assert batch_size == 1 and checked.batch_size == job["baseline_execution"]["batch_size"]
        observed.append((batch_size, workers))
        if scenario == "sampled_oom" and batch_size == 8192:
            return {
                "status": "oom",
                "error": "debug-only simulated candidate OOM",
                "condition": job["condition"],
                "model_seed": job["model_seed"],
            }
        return _measurement(probe_job, batch_size, workers)

    monkeypatch.setattr(calibration, "_measure", measure)
    assert runner._probe(path) == 0
    assert observed == ([(1, 0)] if full_graph else [(2048, 0), (4096, 0), (8192, 0)])
    certificate_path = directory / "resource-certificate.json"
    certificate = runner._validate_certificate(job, certificate_path)
    assert certificate["selected_execution"]["sample_seed_batch_size"] == (
        2048 if full_graph else 4096 if scenario == "sampled_oom" else 8192
    )
    assert certificate["natural_training_split_size"] == (1 if full_graph else 8192)
    if full_graph:
        assert certificate["selected_execution"] == job["baseline_execution"]
        assert certificate["selected_candidate"] == {"batch_size": 1, "workers": 0}
    assert certificate["selected_configuration"]["solver_steps"] == 8
    assert certificate["source_resource_plan_is_new_model_measurement"] is False
    # Cross-module contract test: consume this producer's evidence in the REAL trainer validator.
    monkeypatch.setattr(train, "_versions", lambda: runtime)
    monkeypatch.setattr(calibration, "_hardware", lambda _device: hardware)
    monkeypatch.setattr(training_resource_plan, "source_snapshot", runner.source_snapshot)
    resolved = list(job["command"])
    for field, value in certificate["selected_execution"].items():
        resolved = runner._set_option(resolved, "--" + field.replace("_", "-"), value)
    resolved += [
        "--transition-resource-certificate",
        str(certificate_path),
        "--transition-resource-sha256",
        runner._sha(certificate_path),
    ]
    selected_args = train.build_parser().parse_args(resolved[resolved.index("-m") + 2 :])
    train.validate_args(selected_args)
    inspected = {
        "sha256": job["source_checkpoint_sha256"],
        "source_epoch": job["source_epoch"],
        "identity": {"configuration": job["source_configuration"]},
    }
    verified, changes = transition_training._read_certificate(
        selected_args, inspected, job["source_dataset_protocol"], train.configuration(selected_args)
    )
    assert verified == certificate
    assert ("sample_seed_batch_size" in changes) == (not full_graph)
    # Completed disposable measurements are reused, never repeated on probe resume.
    observed.clear()
    assert runner._probe(path) == 0
    assert observed == []
    certificate["selected_configuration"]["solver_steps"] = 9
    certificate_path.write_text(json.dumps(certificate))
    with pytest.raises(ValueError, match="different V5 model"):
        runner._validate_certificate(job, certificate_path)
