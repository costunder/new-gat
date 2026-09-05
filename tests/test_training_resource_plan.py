"""Explicit CPU fixtures for certificate validation; not measured GPU results."""

from __future__ import annotations

import copy
import json

import pytest

from scripts import training_resource_plan as plans


def _measurement(condition, batch, *, rate=None):
    rate = batch if rate is None else rate
    return {
        "status": "passed",
        "condition": condition,
        "model_seed": 0,
        "batch_size": batch,
        "workers": 0,
        "unit": "supervised_seed_nodes",
        "elapsed_seconds": 3.0,
        "processed_units": rate * 3,
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
    }


def _candidate(batch):
    return {
        "status": "passed",
        "batch_size": batch,
        "workers": 0,
        "measurements": [
            _measurement(condition, batch) for condition in ("fixed_c", "shared_dynamic_c")
        ],
    }


def _plan():
    return {
        "schema_version": 1,
        "kind": "measured_training_resource_plan",
        "status": "passed",
        "classification": "resource_calibration_not_final_training",
        "final_training_started": False,
        "request_sha256": "a" * 64,
        "source_sha256": {"debug": "fixture"},
        "hardware_profile": "a6000-48gb",
        "profiles": ["reference"],
        "model_seeds": [0],
        "runtime": {"python": "3.11", "torch": "debug", "cuda": "11.8"},
        "hardware": {
            "cuda:0": {
                "device": "cuda:0",
                "name": "explicit debug fixture",
                "uuid": "debug-fixture",
                "uuid_unavailable_reason": None,
                "total_memory_bytes": 48 * 1024**3,
                "compute_capability": [8, 6],
                "allocated_cpu_count": 8,
                "cuda_visible_devices": "3",
            }
        },
        "entries": [
            {
                "status": "passed",
                "track": "conductance",
                "profile": "reference",
                "dataset": "ogbn-arxiv",
                "baseline_physical_batch_size": 2,
                "batch_axis": "sampled_seed_nodes",
                "natural_training_split_size": 4,
                "worker_candidates": [0],
                "job_contracts": [
                    {"condition": condition, "model_seed": 0, "argv_sha256": "b" * 64}
                    for condition in ("fixed_c", "shared_dynamic_c")
                ],
                "candidates": [_candidate(2), _candidate(4)],
                "selected": {"batch_size": 1, "sample_seed_batch_size": 4, "workers": 0},
                "stop_reason": "complete_training_split_boundary",
            }
        ],
    }


def _validate(plan):
    plans.validate_resource_plan(
        plan,
        hardware_profile="a6000-48gb",
        profiles=["reference"],
        model_seeds=[0],
        check_sources=False,
    )


def test_complete_real_path_certificate_schema():
    _validate(_plan())


@pytest.mark.parametrize(
    "module",
    ["research.conductance_gat.v5.train", "research.cycle_pe.v2.benchmark"],
)
def test_command_identity_binds_measured_gpu_assignment(module):
    measured = ["python", "-B", "-m", module, "--device", "cuda:0"]
    reassigned = [*measured[:-1], "cuda:1"]
    assert plans.command_identity(measured) != plans.command_identity(reassigned)
    assert plans.command_identity(measured) == plans.command_identity(
        [*measured, "--output-dir", "new-run", "--batch-size", "8", "--workers", "2", "--resume"]
    )


def test_job_plan_rejects_track_reassigned_to_an_unmeasured_gpu():
    plan = _plan()
    measured = [
        "python",
        "-B",
        "-m",
        "research.conductance_gat.v5.train",
        "--device",
        "cuda:0",
        "--batch-size",
        "1",
        "--sample-seed-batch-size",
        "4",
        "--workers",
        "0",
    ]
    plan["entries"][0]["job_contracts"][0]["argv_sha256"] = plans.command_identity(measured)
    arguments = {
        "track": "conductance",
        "profile": "reference",
        "dataset": "ogbn-arxiv",
        "condition": "fixed_c",
        "model_seed": 0,
    }
    plans.validate_job_plan(plan, command=measured, **arguments)
    reassigned = list(measured)
    reassigned[reassigned.index("--device") + 1] = "cuda:1"
    with pytest.raises(ValueError, match="measured scientific recipe"):
        plans.validate_job_plan(plan, command=reassigned, **arguments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples_per_second", float("nan")),
        ("elapsed_seconds", 0),
        ("processed_units", True),
        ("optimizer_steps", 0),
        ("optimizer_state_bytes", 0),
        ("measurement_steps_requested", 6),
        ("minimum_measure_seconds_requested", 4.0),
        ("peak_allocated_bytes", 49 * 1024**3),
        ("free_bytes_before", 49 * 1024**3),
        ("batch_size", 8),
        ("workers", 2),
    ],
)
def test_unmeasured_or_inconsistent_gpu_evidence_is_rejected(field, value):
    plan = _plan()
    plan["entries"][0]["candidates"][0]["measurements"][0][field] = value
    with pytest.raises(ValueError):
        _validate(plan)


def test_missing_pair_is_not_an_acceptable_resource_plan():
    plan = _plan()
    plan["entries"][0]["candidates"][1]["measurements"].pop()
    with pytest.raises(ValueError, match="paired measurements"):
        _validate(plan)


def test_unmeasured_baseline_and_unselected_best_rejected():
    plan = _plan()
    plan["entries"][0]["candidates"].pop(0)
    with pytest.raises(ValueError, match="baseline"):
        _validate(plan)
    plan = _plan()
    plan["entries"][0]["selected"]["sample_seed_batch_size"] = 2
    with pytest.raises(ValueError, match="best safe"):
        _validate(plan)


def test_unsafe_larger_batch_is_not_selected_and_is_a_measured_boundary():
    plan = _plan()
    entry = plan["entries"][0]
    for report in entry["candidates"][1]["measurements"]:
        report["peak_reserved_bytes"] = 44 * 1024**3
    entry["selected"]["sample_seed_batch_size"] = 2
    entry["stop_reason"] = "memory_headroom_boundary"
    _validate(plan)


def test_oom_error_is_kept_but_generic_failure_is_not_relabelled():
    plan = _plan()
    entry = plan["entries"][0]
    entry["candidates"][1] = {
        "status": "oom",
        "batch_size": 4,
        "workers": 0,
        "measurements": [
            {"status": "oom", "error": "debug CUDA OOM fixture", "condition": c, "model_seed": 0}
            for c in ("fixed_c", "shared_dynamic_c")
        ],
    }
    entry["selected"]["sample_seed_batch_size"] = 2
    entry["stop_reason"] = "memory_headroom_boundary"
    _validate(plan)
    entry["candidates"][1]["measurements"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="errors are not OOM"):
        _validate(plan)


def test_transductive_graph_cannot_be_replicated_as_physical_batch():
    plan = _plan()
    plan["entries"][0]["selected"]["batch_size"] = 4
    with pytest.raises(ValueError, match="duplicated"):
        _validate(plan)


def test_false_plateau_and_false_capacity_boundaries_rejected():
    plan = _plan()
    for reason in ("measured_throughput_plateau", "memory_headroom_boundary"):
        plan["entries"][0]["stop_reason"] = reason
        with pytest.raises(ValueError):
            _validate(plan)


def test_file_load_hashes_exact_bytes_and_rejects_changed_source(tmp_path, monkeypatch):
    path = tmp_path / "debug-plan.json"
    path.write_text(json.dumps(_plan()), encoding="utf-8")
    monkeypatch.setattr(plans, "source_snapshot", lambda: {"debug": "fixture"})
    loaded = plans.load_resource_plan(
        path, hardware_profile="a6000-48gb", profiles=["reference"], model_seeds=[0]
    )
    assert len(loaded["_sha256"]) == 64
    changed = copy.deepcopy(loaded)
    changed["entries"][0]["selected"]["sample_seed_batch_size"] = 2
    assert plans.resource_plan_identity(changed) != plans.resource_plan_identity(loaded)
    monkeypatch.setattr(plans, "source_snapshot", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="source identity differs"):
        plans.load_resource_plan(
            path, hardware_profile="a6000-48gb", profiles=["reference"], model_seeds=[0]
        )


def test_duplicate_json_keys_fail_before_any_training(tmp_path):
    path = tmp_path / "debug-duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate keys"):
        plans.load_resource_plan(
            path, hardware_profile="a6000-48gb", profiles=["reference"], model_seeds=[0]
        )


def test_runtime_gate_never_substitutes_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        plans.validate_plan_runtime(_plan())
