"""Explicit CPU/mock driver regressions; no real training or GPU evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5 import train
from scripts import run_v5_mechanism_experiments as driver


def args(*extra):
    return driver.parser().parse_args(["--run-id", "debug-mechanism-contract", *extra])


@pytest.mark.parametrize(
    "suites,count",
    [
        (["core"], 6),
        (["generators"], 4),
        (["solvers"], 3),
        (["filters"], 4),
        (["core", "generators"], 8),
        (list(driver.SUITES), 12),
    ],
)
def test_staged_suites_deduplicate_the_same_scientific_control(suites, count):
    variants = driver.variants(suites)
    assert len(variants) == count
    assert len({variant["variant_id"] for variant in variants}) == count
    signatures = [
        json.dumps([variant["condition"], variant["configuration"]], sort_keys=True)
        for variant in variants
    ]
    assert len(set(signatures)) == count


def test_default_full_dataset_scale_seed_budget_and_explicit_generator_semantics(tmp_path):
    selected = args()
    driver.validate_args(selected)
    jobs = driver.make_jobs(selected, tmp_path / "not-created")
    assert len(jobs) == 80
    assert selected.profiles == ["reference", "large"] and selected.model_seeds == [0]
    assert set(selected.datasets) == {"cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"}
    assert selected.epochs == 200 and selected.patience == 50
    assert selected.learning_budget_policy == "reference_updates"
    assert selected.sampling == "auto"
    for job in jobs:
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        assert (child.hidden_channels, child.layers, child.heads) == (
            (256, 8, 8) if job["profile"] == "reference" else (384, 12, 8)
        )
        assert child.beta_initial == 0.5 and child.training_schedule == "joint"
        assert child.epochs == 200 and child.patience == 50 and child.model_seed == 0
        assert child.learning_budget_policy == "reference_updates"
        assert child.conductance_heads == job["variant"]["configuration"]["conductance_heads"]
        assert (
            child.propagation_normalization
            == job["variant"]["configuration"]["propagation_normalization"]
        )
        assert (
            child.conductance_generator == job["variant"]["configuration"]["conductance_generator"]
        )
        if job["variant_id"] == "mlp-shared-symmetric":
            assert child.conductance_backend == "mlp" and child.solver_cost_scaling == "legacy_unit"
        if job["variant_id"] == "degree-only-symmetric":
            assert child.conductance_generator == "degree_only"
        assert "mechanism" not in str(tmp_path) or Path(job["output_dir"]).is_relative_to(tmp_path)
    assert len({job["output_dir"] for job in jobs}) == 80
    assert len({job["job_id"] for job in jobs}) == 80
    assert list(tmp_path.iterdir()) == []


def test_all_feedback_suites_include_train_time_exact_solver_and_polynomial_filter(tmp_path):
    selected = args("--suites", *driver.SUITES)
    jobs = driver.make_jobs(selected, tmp_path)
    assert len(jobs) == 120
    exact = [job for job in jobs if job["variant_id"] == "entropy-exact-symmetric"]
    assert len(exact) == 10
    for job in exact:
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        assert child.conductance_generator == "entropy_exact" and child.solver_degree_barrier == 0
    polynomial = [job for job in jobs if "polynomial3" in job["variant_id"]]
    assert len(polynomial) == 20
    assert {job["condition"] for job in polynomial} == {"fixed_c", "shared_dynamic_c"}
    for job in polynomial:
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        assert child.propagation_filter == "polynomial3"


def test_dry_run_does_not_measure_launch_or_create_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        driver, "_ensure_calibration", lambda *_: pytest.fail("dry run measured GPU")
    )
    monkeypatch.setattr(
        driver.standalone, "check_dependencies", lambda: pytest.fail("dry run queried environment")
    )
    code = driver.main(["--run-id", "debug-dry", "--results-root", str(tmp_path), "--dry-run"])
    assert code == 0
    output = capsys.readouterr().out
    assert "8 unique variants; 80 full-size validation-only trainings" in output
    assert "common" in output.lower() and "no files" in output
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "override",
    [
        ["--epochs", "201"],
        ["--suites", "core", "generators", "solvers"],
        ["--sampling", "auto_disjoint", "--sample-context-seed-batch-size", "2048"],
    ],
)
def test_changed_recipe_or_matrix_refuses_silent_resume_without_writes(tmp_path, override):
    old_args = args()
    old_jobs = driver.make_jobs(old_args, tmp_path)
    manifest = {
        "schema_version": 1,
        "suite": driver.SUITE,
        "run_id": old_args.run_id,
        "config": driver._config(old_args),
        "source_sha256": {},
        "dependencies": {},
        "planned_jobs": old_jobs,
    }
    text = json.dumps(manifest)
    path = SimpleNamespace(read_text=lambda **_: text)
    new_args = args(*override)
    with pytest.raises(ValueError, match="new run ID|matrix"):
        driver._resume_manifest(
            path,
            args=new_args,
            planned=driver.make_jobs(new_args, tmp_path),
            sources={},
            dependencies={},
        )
    assert path.read_text() == text
    assert list(tmp_path.iterdir()) == []


def debug_probe_report(job, configuration, batch, workers, *, oom):
    if oom:
        return {
            "status": "oom",
            "condition": job["condition"],
            "model_seed": job["model_seed"],
            "error": "synthetic per-head memory boundary, not a real GPU measurement",
        }
    rate = batch * 10
    return {
        "status": "passed",
        "condition": job["condition"],
        "model_seed": job["model_seed"],
        "batch_size": batch,
        "workers": workers,
        "unit": "synthetic_supervised_seed_nodes",
        "processed_units": rate * 3,
        "elapsed_seconds": 3.0,
        "samples_per_second": rate,
        "optimizer_steps": 5,
        "optimizer_state_bytes": 1024,
        "measurement_steps_requested": 5,
        "warmup_steps_requested": 2,
        "minimum_measure_seconds_requested": 3.0,
        "peak_allocated_bytes": 8 * 1024**3,
        "peak_reserved_bytes": 10 * 1024**3,
        "total_memory_bytes": 48 * 1024**3,
        "free_bytes_before": 46 * 1024**3,
        "configuration": configuration,
    }


def test_all_variants_are_probed_together_and_one_oom_limits_every_arm_equally(
    tmp_path, monkeypatch
):
    selected = args(
        "--datasets", "ogbn-arxiv", "--profiles", "reference", "--learning-budget-policy", "epochs"
    )
    planned = driver.make_jobs(selected, tmp_path)
    manifest = {"planned_jobs": planned, "jobs": copy.deepcopy(planned), "calibration_entries": []}
    monkeypatch.setattr(
        driver.calibration,
        "_hardware",
        lambda _: {
            "total_memory_bytes": 48 * 1024**3,
            "compute_capability": [8, 6],
            "allocated_cpu_count": 8,
            "name": "synthetic CPU-test fixture, not actual hardware",
        },
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (46 * 1024**3, 48 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(driver.calibration, "allocated_cpu_count", lambda: 8)
    monkeypatch.setattr(
        driver.calibration,
        "_load_group",
        lambda *_: (
            {"synthetic_payload_only": True},
            {"data_sha256": "a" * 64, "split_sha256": "b" * 64},
            4096,
            "sampled_seed_nodes",
        ),
    )
    calls = []

    def measure(job, _loaded, parsed, *, batch_size, workers):
        calls.append((job["condition"], batch_size, workers))
        return debug_probe_report(
            job,
            train.configuration(parsed),
            batch_size,
            workers,
            oom=batch_size == 4096 and "per-head" in job["condition"],
        )

    monkeypatch.setattr(driver.calibration, "_measure", measure)
    driver._ensure_calibration(selected, manifest, lambda: None)
    assert len(calls) == 8 * 2
    assert {condition for condition, _, _ in calls} == {job["variant_id"] for job in planned}
    assert {(size, count) for _, size, count in calls} == {(2048, 0), (4096, 0)}
    assert {job["execution"]["sample_seed_batch_size"] for job in manifest["jobs"]} == {2048}
    assert manifest["calibration_status"] == "passed" and manifest["resources_applied"]
    assert manifest["calibration_entries"][0]["stop_reason"] == "memory_headroom_boundary"
    # Removing any generator from one candidate invalidates the common certificate.
    damaged = copy.deepcopy(manifest["calibration_entries"][0])
    damaged["candidates"][0]["measurements"].pop()
    with pytest.raises(ValueError, match="missing paired"):
        driver._validate_common_entry(damaged, planned, 8)
    assert list(tmp_path.iterdir()) == []


def fake_result():
    return {
        "validation": 0.5,
        "best_epoch": 1,
        "shared_initial_state_sha256": "a" * 64,
        "common_backbone_initial_state_sha256": "b" * 64,
        "learning_budget": {"synthetic": True},
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "checkpoint_sha256": "f" * 64,
    }


def test_same_namespace_skips_completed_and_resumes_only_interrupted_checkpoint(
    tmp_path, monkeypatch
):
    options = [
        "--run-id",
        "debug-resume",
        "--results-root",
        str(tmp_path),
        "--datasets",
        "cora",
        "--profiles",
        "reference",
        "--suites",
        "core",
    ]
    monkeypatch.setattr(
        driver.standalone, "check_dependencies", lambda: {"synthetic_fixture": True}
    )
    monkeypatch.setattr(
        driver.resources, "source_snapshot", lambda: {"synthetic_fixture": "stable"}
    )
    monkeypatch.setattr(driver, "_ensure_calibration", lambda *_: None)
    monkeypatch.setattr(driver, "_read_result", lambda _: fake_result())
    calls, audits = [], []

    def dispatch(command, log, *_):
        if "--root" in command:
            audits.append(command.copy())
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("Synthetic successful audit control-flow fixture\n", encoding="utf-8")
            return 0
        calls.append(command.copy())
        if len(calls) == 2:
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True)
            (output / "last.pt").write_bytes(
                b"explicit fake checkpoint for resume-control-flow test"
            )
            return 9
        return 0

    monkeypatch.setattr(driver.standalone.shared, "run_logged", dispatch)
    assert driver.main(options) == 1
    assert driver.main(options) == 0
    assert len(calls) == 7
    assert "--resume" in calls[2]
    manifest_path = tmp_path / "conductance_gat/mechanisms/debug-resume/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed" and len(manifest["jobs"]) == 6
    assert all(job["status"] == "passed" for job in manifest["jobs"])
    assert len(audits) == 6
    assert all(job["audit"]["status"] == "passed" for job in manifest["jobs"])
    assert driver.main(options) == 0 and len(calls) == 7
    assert len(audits) == 6


def test_common_initialization_and_budget_mismatch_are_not_reported_as_paired(tmp_path):
    selected = args("--datasets", "cora", "--profiles", "reference", "--suites", "generators")
    jobs = driver.make_jobs(selected, tmp_path)
    for job in jobs:
        job.update(status="passed", result=fake_result())
    driver._check_comparison_contracts(jobs)
    jobs[-1]["result"]["shared_initial_state_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="initialization"):
        driver._check_comparison_contracts(jobs)
    jobs[-1]["result"] = fake_result()
    jobs[-1]["result"]["learning_budget"] = {"synthetic": "different"}
    with pytest.raises(ValueError, match="learning_budget"):
        driver._check_comparison_contracts(jobs)


def test_source_snapshot_pins_new_driver_audit_and_core_implementation():
    snapshot = driver.resources.source_snapshot()
    for relative in (
        "scripts/run_v5_mechanism_experiments.py",
        "scripts/audit_v5_stages.py",
        "research/conductance_gat/v5/train.py",
        "research/conductance_gat/v5/model.py",
    ):
        assert snapshot[relative] == driver._file_sha(driver.ROOT / relative)


def test_missing_common_backbone_provenance_cannot_certify_paired_comparisons(tmp_path):
    jobs = driver.make_jobs(args("--datasets", "cora", "--profiles", "reference"), tmp_path)
    for job in jobs:
        job.update(status="passed", result=fake_result())
    jobs[0]["result"]["common_backbone_initial_state_sha256"] = None
    with pytest.raises(ValueError, match="backbone initialization"):
        driver._check_comparison_contracts(jobs)


def test_audit_retry_retains_training(tmp_path, monkeypatch):
    options = [
        "--run-id",
        "debug-audit",
        "--results-root",
        str(tmp_path),
        "--datasets",
        "cora",
        "--profiles",
        "reference",
        "--suites",
        "solvers",
    ]
    monkeypatch.setattr(
        driver.standalone, "check_dependencies", lambda: {"synthetic_fixture": True}
    )
    monkeypatch.setattr(
        driver.resources, "source_snapshot", lambda: {"synthetic_fixture": "stable"}
    )
    monkeypatch.setattr(driver, "_ensure_calibration", lambda *_: None)
    monkeypatch.setattr(driver, "_read_result", lambda _: fake_result())
    training, audits = [], []

    def dispatch(command, log, *_):
        if "--root" in command:
            audits.append(command.copy())
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("Explicit synthetic audit control-flow fixture\n", encoding="utf-8")
            return 8 if len(audits) == 1 else 0
        training.append(command.copy())
        return 0

    monkeypatch.setattr(driver.standalone.shared, "run_logged", dispatch)
    assert driver.main(options) == 1
    run_dir = tmp_path / "conductance_gat/mechanisms/debug-audit"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    first = manifest["jobs"][0]
    assert first["status"] == "passed" and first["audit"]["status"] == "failed"
    failed_log = Path(first["audit"]["log_path"])
    failed_bytes = failed_log.read_bytes()
    assert len(training) == 1
    assert driver.main(options) == 0
    assert len(training) == 3 and len(audits) == 4
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    first = manifest["jobs"][0]
    assert first["audit"]["status"] == "passed" and len(first["audit_attempts"]) == 1
    assert Path(first["audit"]["log_path"]) != failed_log
    assert failed_log.read_bytes() == failed_bytes
    assert ".audit" in (run_dir / "comparison.md").read_text(encoding="utf-8")
    assert driver.main(options) == 0 and len(training) == 3 and len(audits) == 4


def completed_recipe_fixture(tmp_path):
    """Explicit four-epoch CPU metadata fixture, never a training result."""
    selected = args(
        "--datasets",
        "cora",
        "--profiles",
        "reference",
        "--suites",
        "solvers",
        "--epochs",
        "4",
        "--patience",
        "1",
        "--learning-budget-policy",
        "epochs",
    )
    job = driver.make_jobs(selected, tmp_path)[0]
    child = train.build_parser().parse_args(job["command"][5:])
    train.validate_args(child)
    budget = train.plan_learning_budget(4, 1, 1, 1, policy="epochs")
    schedule = train.phase_schedule(4, list(child.phase_fractions), "joint")
    history = [
        {
            "epoch": epoch,
            "train_batches": 1,
            "optimizer_steps": epoch,
            "phase": {"phase": "joint"},
            "validation": 0.5,
        }
        for epoch in range(1, 5)
    ]
    payload = {
        "configuration": train.configuration(child),
        "source_sha256": train.implementation_source_hashes(),
        "learning_budget": budget,
        "epochs_run": 4,
        "schedule": schedule,
        "resume_identity": {"schedule": schedule},
        "optimizer_steps": 4,
        "best_epoch": 3,
        "joint_best_epoch": 3,
    }
    return job, child, payload, history


@pytest.mark.parametrize(
    "damage,match",
    [
        ("epochs", "configuration"),
        ("budget", "learning budget"),
        ("gap", "contiguous"),
        ("partial", "incomplete"),
        ("early", "patience"),
        ("source", "source"),
        ("schedule", "schedule"),
    ],
)
def test_exact_result_recipe_refuses_changed_budget_incomplete_history_or_sources(
    tmp_path, damage, match
):
    _, child, payload, history = completed_recipe_fixture(tmp_path)
    driver._validate_result_recipe(child, payload, history)
    if damage == "epochs":
        payload["configuration"]["epochs"] = 5
    elif damage == "budget":
        payload["learning_budget"]["requested_patience"] = 7
    elif damage == "gap":
        history[2]["epoch"] = 4
    elif damage == "partial":
        history[1]["train_batches"] = 0
    elif damage == "early":
        history.pop()
        payload.update(epochs_run=3, optimizer_steps=3)
    elif damage == "source":
        payload["source_sha256"] = {}
    elif damage == "schedule":
        payload["resume_identity"]["schedule"] = []
    with pytest.raises(ValueError, match=match):
        driver._validate_result_recipe(child, payload, history)


def test_result_read_checks_real_identity_and_common_backbone_hash(tmp_path, monkeypatch):
    from scripts import audit_v5_stages as audit

    job, child, payload, history = completed_recipe_fixture(tmp_path)
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    (output / "best.pt").write_bytes(b"explicit CPU provenance fixture, not trained weights")
    (output / "last.pt").write_bytes(b"explicit CPU last checkpoint fixture")
    (output / "history.json").write_text(json.dumps(history), encoding="utf-8")
    protocol = {"data_sha256": "a" * 64, "split_sha256": "b" * 64}
    identity = train.build_resume_identity(
        child, protocol, payload["schedule"], initial_state_sha256="c" * 64
    )
    payload.update(
        research_suite=audit.SUITE,
        status="passed",
        dataset=child.dataset,
        condition=child.condition,
        protocol=protocol,
        cache_sha256=protocol["data_sha256"],
        resume_identity=identity,
        resume_identity_sha256=audit._canonical(identity),
        validation=0.5,
        checkpoint_selection={"primary_epoch": 3, "primary_validation": 0.5, "test_used": False},
        checkpoint_sha256=audit._sha(output / "best.pt"),
        last_checkpoint_sha256=audit._sha(output / "last.pt"),
        history_sha256=audit._sha(output / "history.json"),
        shared_initial_state_sha256="c" * 64,
        common_backbone_initial_state_sha256="d" * 64,
        evaluation_split="validation",
        test_evaluated=False,
    )
    path = Path(job["metrics_path"])
    monkeypatch.setattr(
        driver.standalone, "_load_metrics", lambda _: {"validation": 0.5, "best_epoch": 3}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert driver._read_result(job)["common_backbone_initial_state_sha256"] == "d" * 64
    payload["common_backbone_initial_state_sha256"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization provenance"):
        driver._read_result(job)
    payload["common_backbone_initial_state_sha256"] = "d" * 64
    payload["resume_identity_sha256"] = "e" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="identity SHA256"):
        driver._read_result(job)
