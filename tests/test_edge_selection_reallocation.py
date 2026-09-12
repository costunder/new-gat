"""CPU-only allocation regression fixtures; no GPU or final-training claim."""

from __future__ import annotations

import copy
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat import edge_selection
from research.conductance_gat.edge_selection import calibration, reallocation
from scripts import run_v5_edge_selection as driver


@pytest.fixture
def allocation_fixture(tmp_path, monkeypatch):
    args = driver.parser().parse_args(
        [
            "--run-id",
            "debug-reallocation",
            "--datasets",
            "ogbn-arxiv",
            "--profiles",
            "reference",
            "--suites",
            "corruption",
        ]
    )
    jobs = driver.make_jobs(args, tmp_path)
    parsed = {job["job_id"]: calibration.parse_job(job) for job in jobs}
    identity = {"synthetic_fixture": True, "data_sha256": "a" * 64}
    hardware = {
        "device": "cuda:0",
        "uuid": "GPU-original",
        "uuid_unavailable_reason": None,
        "name": "NVIDIA RTX A6000",
        "total_memory_bytes": 48 * 1024**3,
        "compute_capability": [8, 6],
        "allocated_cpu_count": 8,
        "cuda_visible_devices": "1",
    }
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    current = dict(hardware, uuid="GPU-reallocated", cuda_visible_devices="4")

    def configuration(value):
        return {
            key: str(item) if isinstance(item, Path) else item for key, item in vars(value).items()
        }

    fake_train = SimpleNamespace(
        configuration=configuration,
        load_calibration_payload=lambda _: (
            {"synthetic_fixture": True},
            copy.deepcopy(identity),
            4096,
            "sampled_seed_nodes",
        ),
    )
    monkeypatch.setattr(edge_selection, "train", fake_train)
    monkeypatch.setattr(calibration, "parse_job", lambda job: copy.deepcopy(parsed[job["job_id"]]))
    monkeypatch.setattr(calibration.resources, "allocated_cpu_count", lambda: 8)
    calls = []

    def measure(job, loaded, child, batch, workers):
        assert loaded == {"synthetic_fixture": True}
        calls.append((job["job_id"], batch, workers))
        physical = calibration._probe_args(child, "sampled_seed_nodes", batch, workers)
        auxiliary = child.negative_loss_weight > 0
        return {
            "synthetic_fixture": True,
            "status": "passed",
            "condition": job["variant_id"],
            "model_seed": job["model_seed"],
            "samples_per_second": 4096,
            "processed_units": 4096 * 5,
            "elapsed_seconds": 5.0,
            "optimizer_steps": 5 * ((4096 + batch - 1) // batch),
            "measurement_steps_requested": 5,
            "warmup_steps_requested": 2,
            "minimum_measure_seconds_requested": 3.0,
            "complete_measurement_epochs": 5,
            "complete_warmup_epochs": 2,
            "warmup_optimizer_steps": 4,
            "optimizer_state_bytes": 1024,
            "peak_allocated_bytes": (8 if auxiliary else 12) * 1024**3,
            "peak_reserved_bytes": (10 if auxiliary else 14) * 1024**3,
            "total_memory_bytes": 48 * 1024**3,
            "free_bytes_before": 46 * 1024**3,
            "unit": "synthetic_supervised_seed_nodes",
            "batch_size": batch,
            "workers": workers,
            "configuration": configuration(physical),
            "validation_completed": True,
            "validation_seconds": (20.0 if batch == 4096 else 2.0) if auxiliary else 1.0,
            "topology_preparation_seconds": 1.0,
            "setup_seconds": 1.0,
            "auxiliary_path_measured": True,
            "required_auxiliary_path": auxiliary,
            "required_cycle_preparation": False,
            "cycle_preparation_measured": False,
            "calibration_not_final": True,
            "parameter_update_verified": True,
            "model_parameter_count": 100_000,
            "initial_model_sha256": "b" * 64,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size": batch,
            "hardware": {
                "device_name": hardware["name"],
                "total_memory_bytes": hardware["total_memory_bytes"],
                "compute_capability": hardware["compute_capability"],
            },
        }

    monkeypatch.setattr(calibration, "_measure", measure)
    entry = {}
    calibration.calibrate_group(jobs, entry, lambda: None)
    manifest = {
        "hardware": copy.deepcopy(hardware),
        "runtime": runtime,
        "planned_jobs": jobs,
        "jobs": driver.common._apply_common_resources(jobs, [entry]),
        "calibration_entries": [entry],
        "resources_applied": True,
        "calibration_status": "passed",
    }
    calls.clear()
    monkeypatch.setattr(driver.hardware_tools, "_hardware", lambda _: current)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (46 * 1024**3, 48 * 1024**3))
    return SimpleNamespace(
        args=args,
        jobs=jobs,
        manifest=manifest,
        hardware=hardware,
        current=current,
        runtime=runtime,
        calls=calls,
        measure=measure,
        identity=identity,
        train=fake_train,
        grouped=driver.common._grouped(jobs),
    )


def test_reallocation_preserves_original_recipe_and_reuses_passed_evidence(allocation_fixture):
    value = allocation_fixture
    original = copy.deepcopy(value.manifest)
    snapshots = []
    driver._ensure_calibration(
        value.args, value.manifest, lambda: snapshots.append(copy.deepcopy(value.manifest))
    )
    for key, content in original.items():
        assert value.manifest[key] == content
    assert len(value.calls) == 2
    assert {(batch, workers) for _, batch, workers in value.calls} == {(2048, 0)}
    attempt = value.manifest["allocation_history"][0]
    assert attempt["status"] == "passed" and attempt["scope"] == reallocation.SCOPE
    assert attempt["hardware"]["cuda_visible_devices"] == "4"
    assert attempt["original_calibration_sha256"] == reallocation._original_digest(original)
    assert all("current_allocation" not in item for item in snapshots[:-2])
    references = attempt["groups"][0]["representatives"]
    assert {criterion for item in references for criterion in item["selection_criteria"]} == {
        "maximum_peak_reserved_bytes",
        "maximum_peak_allocated_bytes",
        "maximum_projected_full_budget_seconds",
    }
    driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert len(value.calls) == 2 and len(value.manifest["allocation_history"]) == 1


@pytest.mark.parametrize(
    "field,new",
    [
        ("name", "different GPU"),
        ("total_memory_bytes", 24 * 1024**3),
        ("compute_capability", [9, 0]),
        ("allocated_cpu_count", 4),
    ],
)
def test_hardware_class_and_cpu_changes_fail_before_probe(allocation_fixture, field, new):
    value = allocation_fixture
    value.current[field] = new
    original = copy.deepcopy(value.manifest)
    with pytest.raises(ValueError, match=rf"hardware\.{field}"):
        driver._ensure_calibration(
            value.args, value.manifest, lambda: pytest.fail("persisted mismatch")
        )
    assert value.manifest == original and not value.calls


def test_runtime_change_reports_exact_field(allocation_fixture):
    value = allocation_fixture
    value.manifest["runtime"]["torch"] = "different-version"
    with pytest.raises(ValueError, match=r"runtime\.torch"):
        driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert not value.calls


@pytest.mark.parametrize("outcome", ["oom", "insufficient_headroom", "interrupted"])
def test_failed_probe_is_retained_and_retry_remeasures(allocation_fixture, monkeypatch, outcome):
    value = allocation_fixture
    original = copy.deepcopy(value.manifest)

    def failure(*args):
        if outcome == "interrupted":
            raise KeyboardInterrupt("synthetic fixture interruption")
        if outcome == "oom":
            return {"status": "oom", "error": "synthetic fixture OOM"}
        report = value.measure(*args)
        report["free_bytes_before"] = report["peak_reserved_bytes"]
        return report

    monkeypatch.setattr(calibration, "_measure", failure)
    expected = KeyboardInterrupt if outcome == "interrupted" else RuntimeError
    with pytest.raises(expected):
        driver._ensure_calibration(value.args, value.manifest, lambda: None)
    for key, content in original.items():
        assert value.manifest[key] == content
    assert "current_allocation" not in value.manifest
    failed = copy.deepcopy(value.manifest["allocation_history"][0])
    assert failed["status"] == ("interrupted" if outcome == "interrupted" else "failed")
    monkeypatch.setattr(calibration, "_measure", value.measure)
    driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert value.manifest["allocation_history"][0] == failed
    assert value.manifest["current_allocation"]["history_index"] == 1


@pytest.mark.parametrize(
    "field,new",
    [
        ("parameter_update_verified", False),
        ("effective_batch_size", 1),
        ("gradient_accumulation_steps", 2),
        ("data_parallel_workers", 2),
        ("model_parameter_count", 10),
        ("initial_model_sha256", "c" * 64),
        ("complete_warmup_epochs", 0),
        ("validation_completed", False),
        ("complete_warmup_epochs", float("nan")),
        ("warmup_optimizer_steps", float("inf")),
        ("parameter_update_verified", 1),
        ("complete_measurement_epochs", 1.5),
        ("auxiliary_path_measured", False),
        ("configuration", {}),
    ],
)
def test_fresh_probe_must_match_full_unchanged_recipe(allocation_fixture, monkeypatch, field, new):
    value = allocation_fixture

    def damaged(*args):
        report = value.measure(*args)
        report[field] = new
        return report

    monkeypatch.setattr(calibration, "_measure", damaged)
    with pytest.raises(ValueError):
        driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert "current_allocation" not in value.manifest


def test_data_change_rejected_before_probe(allocation_fixture):
    value = allocation_fixture
    value.identity["data_sha256"] = "changed"
    with pytest.raises(ValueError, match="data/split/topology"):
        driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert not value.calls and "current_allocation" not in value.manifest


def test_passed_evidence_tampering_rejected(allocation_fixture):
    value = allocation_fixture
    driver._ensure_calibration(value.args, value.manifest, lambda: None)
    value.manifest["allocation_history"][0]["groups"][0]["measurements"][0]["report"][
        "batch_size"
    ] = 1
    with pytest.raises(ValueError, match="evidence"):
        driver._ensure_calibration(value.args, value.manifest, lambda: None)


def test_partial_original_calibration_cannot_mix_allocations(allocation_fixture):
    value = allocation_fixture
    value.manifest["calibration_status"] = "running"
    with pytest.raises(ValueError, match="partial measurements"):
        driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert not value.calls and "allocation_history" not in value.manifest


def test_finished_groups_skip_new_probes_until_audit_command_changes(allocation_fixture):
    value = allocation_fixture
    for job in value.manifest["jobs"]:
        job.update(
            status="passed",
            audit={"status": "passed", "command": driver._audit_command(value.args, job)},
        )
    driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert not value.calls and "current_allocation" not in value.manifest
    value.args.repeat_evaluations += 1
    driver._ensure_calibration(value.args, value.manifest, lambda: None)
    assert value.calls and value.manifest["allocation_history"][0]["status"] == "passed"


def test_failed_final_publish_never_commits_allocation(allocation_fixture):
    value = allocation_fixture

    def persist():
        if "current_allocation" in value.manifest:
            raise OSError("synthetic publication failure")

    with pytest.raises(OSError, match="publication"):
        driver._ensure_calibration(value.args, value.manifest, persist)
    assert "current_allocation" not in value.manifest
    assert value.manifest["allocation_history"][0]["status"] == "failed"
