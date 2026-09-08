"""Synthetic CPU calibration/certificate tests; no actual GPU measurement."""

from __future__ import annotations

import copy

import pytest

from research.conductance_gat.v5 import batch_calibration as probe
from scripts import calibrate_training_resources as calibration
from scripts import training_resource_plan as plans


def command(condition="fixed_c", mode="cluster_disjoint", *, physical=64, context=32):
    result = [
        "python",
        "-B",
        "-m",
        "research.conductance_gat.v5.train",
        "--dataset",
        "ogbn-arxiv",
        "--condition",
        condition,
        "--output-dir",
        "debug-contract-not-created",
        "--device",
        "cuda:0",
        "--hardware-profile",
        "a6000-48gb",
        "--sampling",
        mode,
        "--batch-size",
        "1",
        "--workers",
        "0",
        "--sample-seed-batch-size",
        str(physical),
        "--model-seed",
        "0",
    ]
    if mode == "cluster_disjoint":
        result += [
            "--sample-context-seed-batch-size",
            str(context),
            "--sample-context-workers",
            "2",
        ]
    return result


def jobs(mode="cluster_disjoint", **kwargs):
    return [
        {
            "track": "conductance",
            "profile": "reference",
            "dataset": "ogbn-arxiv",
            "condition": condition,
            "model_seed": 0,
            "device": "cuda:0",
            "command": command(condition, mode, **kwargs),
        }
        for condition in ("fixed_c", "shared_dynamic_c")
    ]


def debug_report(job, size, workers, mode):
    rate = size * max(workers, 1) * 10
    report = {
        "status": "passed",
        "condition": job["condition"],
        "model_seed": 0,
        "batch_size": size,
        "workers": workers,
        "unit": "explicit_synthetic_seed_nodes",
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
        "calibration_only": True,
        "final_training_performed": False,
        "resource_observability": {"synthetic_fixture_not_gpu_evidence": True},
    }
    if mode == "cluster_disjoint":
        report.update(
            worker_axis="sample_context_workers", loader_workers=0, sample_context_workers=workers
        )
    return report


def debug_calibration(
    monkeypatch, mode="cluster_disjoint", *, physical=64, context=32, maximum=256
):
    group = jobs(mode, physical=physical, context=context)
    payload = {"synthetic_fixture_only": True}
    calls, snapshots, entry = [], [], {}
    monkeypatch.setattr(calibration, "allocated_cpu_count", lambda: 8)
    monkeypatch.setattr(
        calibration,
        "_load_group",
        lambda *_: (
            payload,
            {"data_sha256": "a" * 64, "split_sha256": "b" * 64},
            maximum,
            "sampled_seed_nodes",
        ),
    )

    def measure(job, loaded, args, *, batch_size, workers):
        assert loaded is payload
        # This remains the DataLoader axis, even while the context pool is searched.
        assert args.workers == 0
        calls.append((batch_size, workers, job["condition"]))
        return debug_report(job, batch_size, workers, mode)

    monkeypatch.setattr(calibration, "_measure", measure)
    calibration._calibrate_group(group, entry, lambda: snapshots.append(copy.deepcopy(entry)))
    return group, entry, calls, snapshots


def test_disjoint_calibration_searches_context_threads_and_multiple_physical_batches(monkeypatch):
    _, entry, calls, snapshots = debug_calibration(monkeypatch)
    assert entry["worker_axis"] == "sample_context_workers"
    assert entry["worker_candidates"] == [2, 4]
    assert {size for size, _, _ in calls} == {64, 128, 256}
    assert len(calls) == 3 * 2 * 2
    assert entry["selected"] == {
        "batch_size": 1,
        "sample_seed_batch_size": 256,
        "workers": 0,
        "sample_context_workers": 4,
    }
    assert entry["stop_reason"] == "complete_training_split_boundary"
    assert entry["selection"]["paired_resources_identical"]
    assert snapshots and snapshots[-1] == entry
    plans._validate_entry(entry, [0], allocated_cpus=8)
    selected = plans.selected_resources(
        {"entries": [entry]}, track="conductance", profile="reference", dataset="ogbn-arxiv"
    )
    assert selected == entry["selected"] and selected is not entry["selected"]


def test_legacy_sampler_still_has_only_worker_zero_and_no_context_selection(monkeypatch):
    _, entry, calls, _ = debug_calibration(monkeypatch, mode="cluster")
    assert entry["worker_candidates"] == [0]
    assert all(workers == 0 for _, workers, _ in calls)
    assert "worker_axis" not in entry
    assert "sample_context_workers" not in entry["selected"]
    assert entry["selected"]["workers"] == 0
    plans._validate_entry(entry, [0], allocated_cpus=8)


def test_actual_probe_candidate_changes_context_pool_not_loader_or_model_dimensions():
    original = calibration._training_args(jobs()[0])
    candidate = probe._candidate_args(original, 128, 4)
    assert original.sample_seed_batch_size == 64 and original.sample_context_workers == 2
    assert candidate.sample_seed_batch_size == 128 and candidate.sample_context_workers == 4
    assert candidate.workers == original.workers == 0
    for key in (
        "hidden_channels",
        "layers",
        "heads",
        "ffn_multiplier",
        "num_neighbors",
        "sample_context_seed_batch_size",
        "solver_steps",
        "epochs",
        "patience",
    ):
        assert getattr(candidate, key) == getattr(original, key)
    with pytest.raises(ValueError, match="positive worker"):
        probe._candidate_args(original, 128, 0)
    with pytest.raises(ValueError, match="cannot reduce"):
        probe._candidate_args(original, 32, 4)
    legacy = calibration._training_args(jobs("cluster")[0])
    assert probe._candidate_args(legacy, 128, 0).workers == 0
    with pytest.raises(ValueError, match="not DataLoader workers"):
        probe._candidate_args(legacy, 128, 4)


def test_invalid_physical_floor_rejected_before_any_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(calibration, "_load_group", lambda *_: ({}, {}, 256, "sampled_seed_nodes"))
    monkeypatch.setattr(calibration, "_measure", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(ValueError, match="whole sampling contexts"):
        calibration._calibrate_group(jobs(physical=96, context=64), {}, lambda: None)
    assert not calls


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_pair",
        "missing_worker_probe",
        "one_physical_batch",
        "partial_measurement_window",
    ],
)
def test_incomplete_disjoint_measurements_are_not_certified(monkeypatch, mutation):
    _, entry, _, _ = debug_calibration(monkeypatch)
    if mutation == "missing_pair":
        entry["candidates"][0]["measurements"].pop()
    elif mutation == "missing_worker_probe":
        entry["candidates"].pop(0)
    elif mutation == "one_physical_batch":
        entry["candidates"] = [item for item in entry["candidates"] if item["batch_size"] == 64]
    else:
        entry["candidates"][0]["measurements"][0]["measurement_steps_requested"] = 6
    with pytest.raises(ValueError):
        plans._validate_entry(entry, [0], allocated_cpus=8)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_axis",
        "unknown_axis",
        "loader_selected",
        "missing_selected_context_worker",
        "selected_unmeasured_worker",
        "report_loader_axis",
        "report_loader_workers",
        "report_wrong_context_workers",
        "report_missing_context_workers",
    ],
)
def test_disjoint_certificate_enforces_separate_measured_worker_axis(monkeypatch, mutation):
    _, entry, _, _ = debug_calibration(monkeypatch)
    report = entry["candidates"][0]["measurements"][0]
    if mutation == "missing_axis":
        entry.pop("worker_axis")
    elif mutation == "unknown_axis":
        entry["worker_axis"] = "invented_workers"
    elif mutation == "loader_selected":
        entry["selected"]["workers"] = 4
    elif mutation == "missing_selected_context_worker":
        entry["selected"].pop("sample_context_workers")
    elif mutation == "selected_unmeasured_worker":
        entry["selected"]["sample_context_workers"] = 8
    elif mutation == "report_loader_axis":
        report["worker_axis"] = "loader_workers"
    elif mutation == "report_loader_workers":
        report["loader_workers"] = 2
    elif mutation == "report_wrong_context_workers":
        report["sample_context_workers"] = 4
    else:
        report.pop("sample_context_workers")
    with pytest.raises(ValueError):
        plans._validate_entry(entry, [0], allocated_cpus=8)


def selected_command(entry, condition="fixed_c"):
    result = command(condition)
    for field, value in entry["selected"].items():
        option = "--" + field.replace("_", "-")
        result[result.index(option) + 1] = str(value)
    return result


def validate_job(entry, argv):
    plans.validate_job_plan(
        {"entries": [entry]},
        track="conductance",
        profile="reference",
        dataset="ogbn-arxiv",
        condition="fixed_c",
        model_seed=0,
        command=argv,
    )


def test_command_binds_context_recipe_but_selected_workers_are_verified_separately(monkeypatch):
    _, entry, _, _ = debug_calibration(monkeypatch)
    actual = selected_command(entry)
    validate_job(entry, actual)
    wrong_workers = actual.copy()
    wrong_workers[wrong_workers.index("--sample-context-workers") + 1] = "2"
    assert plans.command_identity(wrong_workers) == plans.command_identity(actual)
    with pytest.raises(ValueError, match="measured selection"):
        validate_job(entry, wrong_workers)
    wrong_context = actual.copy()
    wrong_context[wrong_context.index("--sample-context-seed-batch-size") + 1] = "3"
    assert plans.command_identity(wrong_context) != plans.command_identity(actual)
    with pytest.raises(ValueError, match="scientific recipe"):
        validate_job(entry, wrong_context)


def test_context_command_cannot_use_legacy_worker_axis_even_with_matching_digest(monkeypatch):
    _, entry, _, _ = debug_calibration(monkeypatch, mode="cluster")
    argv = command()
    argv[argv.index("--sample-seed-batch-size") + 1] = str(
        entry["selected"]["sample_seed_batch_size"]
    )
    entry["job_contracts"][0]["argv_sha256"] = plans.command_identity(argv)
    with pytest.raises(ValueError, match="context|axis|disjoint"):
        validate_job(entry, argv)


def test_legacy_command_cannot_claim_context_worker_measurement(monkeypatch):
    _, entry, _, _ = debug_calibration(monkeypatch)
    argv = selected_command(entry)
    argv[argv.index("--sampling") + 1] = "cluster"
    entry["job_contracts"][0]["argv_sha256"] = plans.command_identity(argv)
    with pytest.raises(ValueError, match="context|axis|disjoint"):
        validate_job(entry, argv)
