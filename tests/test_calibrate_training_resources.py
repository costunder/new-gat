"""Explicit mocked calibration control-flow tests; no real GPU throughput claims."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import calibrate_training_resources as calibration
from scripts.training_resource_plan import command_identity

GIB = 1024**3


def debug_report(job, batch_size, workers, *, rate=100.0, unsafe=False, oom=False):
    report = {
        "condition": job["condition"],
        "model_seed": job["model_seed"],
        "batch_size": batch_size,
        "workers": workers,
    }
    if oom:
        return {**report, "status": "oom", "error": "explicit synthetic CUDA OOM test fixture"}
    # Synthetic timing evidence only; long enough for the requested window.
    processed = batch_size * 1000
    return {
        **report,
        "status": "passed",
        "unit": "debug_fixture_graphs",
        "processed_units": processed,
        "elapsed_seconds": processed / rate,
        "samples_per_second": rate,
        "optimizer_steps": 1003,
        "optimizer_state_bytes": 1024,
        "measurement_steps": 1000,
        "warmup_steps": 2,
        "measurement_steps_requested": 5,
        "warmup_steps_requested": 2,
        "minimum_measure_seconds_requested": 3.0,
        "peak_allocated_bytes": (44 if unsafe else 3) * GIB,
        "peak_reserved_bytes": (45 if unsafe else 4) * GIB,
        "total_memory_bytes": 48 * GIB,
        "free_bytes_before": 47 * GIB,
        "free_bytes_after": (2 if unsafe else 43) * GIB,
        "resource_observability": {"debug_mock_only": True},
        "calibration_only": True,
        "final_training_performed": False,
    }


def debug_harness(
    monkeypatch,
    *,
    axis="graphs",
    baseline=4,
    maximum=64,
    workers=(2, 4),
    seeds=(0,),
    report_factory=None,
):
    track = "conductance" if axis != "graphs" else "cycle"
    conditions = ("fixed_c", "shared_dynamic_c") if track == "conductance" else ("se", "pe")
    module = (
        "research.conductance_gat.v5.train"
        if track == "conductance"
        else "research.cycle_pe.v2.benchmark"
    )
    jobs = [
        {
            "track": track,
            "profile": "large",
            "dataset": "debug_official_fixture",
            "condition": condition,
            "model_seed": seed,
            "device": "cuda:0",
            "command": [
                "python",
                "-B",
                "-m",
                module,
                "--debug-fixture-identity",
                condition,
                "--model-seed",
                str(seed),
            ],
        }
        for condition in conditions
        for seed in seeds
    ]
    parsed = SimpleNamespace(
        batch_size=baseline,
        sample_seed_batch_size=baseline,
        workers=4,
        device="cuda:0",
        dataset=jobs[0]["dataset"],
    )
    loaded = object()
    identity = {"debug_mock_only": True, "full_split_sha256": "a" * 64}
    calls, snapshots = [], []
    monkeypatch.setattr(calibration, "_training_args", lambda _job: copy.copy(parsed))
    monkeypatch.setattr(
        calibration, "_load_group", lambda _job, _args: (loaded, identity, maximum, axis)
    )
    monkeypatch.setattr(calibration, "allocated_cpu_count", lambda: 8)
    monkeypatch.setattr(calibration, "worker_candidates", lambda *_args, **_kwargs: list(workers))

    def measure(job, payload, _args, *, batch_size, workers):
        assert payload is loaded
        calls.append((batch_size, workers, job["condition"], job["model_seed"]))
        factory = report_factory or (
            lambda selected, size, count: debug_report(
                selected, size, count, rate=float(size * max(count, 1))
            )
        )
        return factory(job, batch_size, workers)

    monkeypatch.setattr(calibration, "_measure", measure)
    entry = {}

    def persist():
        snapshots.append(copy.deepcopy(entry))

    return jobs, entry, persist, calls, snapshots, identity


@pytest.mark.parametrize(
    "track,axis",
    [("cycle", "graphs"), ("conductance", "graphs"), ("conductance", "sampled_seed_nodes")],
)
def test_explicit_reuse_rejects_a_new_floor_above_the_measured_selection(monkeypatch, track, axis):
    jobs, entry, persist, _, _, _ = debug_harness(monkeypatch, axis=axis, maximum=8)
    for job in jobs:
        job["track"] = track
    calibration._calibrate_group(jobs, entry, persist)
    key = "sample_seed_batch_size" if axis == "sampled_seed_nodes" else "batch_size"
    assert entry["selected"][key] == 8
    original_parse = calibration._training_args

    def increased_floor(job):
        args = original_parse(job)
        setattr(args, key, 16)
        return args

    monkeypatch.setattr(calibration, "_training_args", increased_floor)
    monkeypatch.setattr(
        calibration, "_load_group", lambda *_args: pytest.fail("reject before loading data")
    )
    with pytest.raises(ValueError, match="below the current requested floor 16"):
        calibration.verify_plan_inputs({"entries": [entry]}, jobs)


def test_explicit_reuse_accepts_a_selection_above_the_requested_floor(monkeypatch):
    jobs, entry, persist, _, _, _ = debug_harness(monkeypatch, maximum=8)
    calibration._calibrate_group(jobs, entry, persist)
    calibration.verify_plan_inputs({"entries": [entry]}, jobs)


def test_explicit_reuse_checks_every_paired_command_before_data_loading(monkeypatch):
    jobs, entry, persist, _, _, _ = debug_harness(monkeypatch, maximum=8)
    calibration._calibrate_group(jobs, entry, persist)
    jobs[-1]["command"].extend(["--device", "cuda:1"])
    monkeypatch.setattr(
        calibration, "_load_group", lambda *_args: pytest.fail("reject before loading data")
    )
    with pytest.raises(ValueError, match="recipe or measured device differs"):
        calibration.verify_plan_inputs({"entries": [entry]}, jobs)


def test_real_group_algorithm_selects_one_measured_policy_for_all_paired_arms(monkeypatch):
    rates = {4: 40.0, 8: 80.0, 16: 160.0, 32: 161.0, 64: 159.0}

    def report(job, batch, workers):
        # The faster SE arm cannot conceal the slower PE arm's actual throughput.
        rate = rates[batch] * (2 if job["condition"] == "se" else 1)
        rate *= 1.0 if workers == 4 else 0.8
        return debug_report(job, batch, workers, rate=rate)

    jobs, entry, persist, calls, snapshots, _ = debug_harness(
        monkeypatch, maximum=128, seeds=(0, 7), report_factory=report
    )
    calibration._calibrate_group(jobs, entry, persist)
    assert entry["status"] == "passed"
    assert entry["selected"] == {"batch_size": 32, "workers": 4}
    assert entry["selection"]["paired_resources_identical"] is True
    assert entry["selection"]["minimum_requested_batch_preserved"] is True
    assert entry["stop_reason"] == "measured_throughput_plateau"
    assert sorted({call[0] for call in calls}) == [4, 8, 16, 32, 64]
    assert len(calls) == 5 * 2 * 4
    for candidate in entry["candidates"]:
        assert {(row["condition"], row["model_seed"]) for row in candidate["measurements"]} == {
            (condition, seed) for condition in ("se", "pe") for seed in (0, 7)
        }
    assert all(
        contract["argv_sha256"] == command_identity(job["command"])
        for contract, job in zip(entry["job_contracts"], jobs, strict=True)
    )
    assert snapshots[-1]["status"] == "passed"


def test_growth_reaches_natural_full_split_without_arbitrary_cap(monkeypatch):
    jobs, entry, persist, calls, _, _ = debug_harness(monkeypatch, maximum=20)
    calibration._calibrate_group(jobs, entry, persist)
    assert sorted({call[0] for call in calls}) == [4, 8, 16, 20]
    assert entry["selected"]["batch_size"] == 20
    assert entry["stop_reason"] == "complete_training_split_boundary"


def test_unsafe_reserved_memory_boundary_never_accepted_as_candidate(monkeypatch):
    jobs, entry, persist, calls, _, _ = debug_harness(
        monkeypatch,
        maximum=64,
        report_factory=lambda job, batch, workers: debug_report(
            job, batch, workers, rate=float(batch * workers), unsafe=batch >= 16
        ),
    )
    calibration._calibrate_group(jobs, entry, persist)
    assert entry["selected"] == {"batch_size": 8, "workers": 4}
    assert entry["stop_reason"] == "memory_headroom_boundary"
    assert sorted({call[0] for call in calls}) == [4, 8, 16]
    assert entry["baseline_physical_batch_size"] == 4


def test_oom_in_either_paired_arm_rejects_whole_resource_candidate(monkeypatch):
    def report(job, batch, workers):
        return debug_report(
            job,
            batch,
            workers,
            rate=float(batch * workers),
            oom=batch >= 8 and job["condition"] == "pe",
        )

    jobs, entry, persist, calls, _, _ = debug_harness(monkeypatch, report_factory=report)
    calibration._calibrate_group(jobs, entry, persist)
    assert entry["selected"] == {"batch_size": 4, "workers": 4}
    failed = [candidate for candidate in entry["candidates"] if candidate["batch_size"] == 8]
    assert all(candidate["status"] == "oom" for candidate in failed)
    assert all(len(candidate["measurements"]) == 2 for candidate in failed)
    assert sorted({call[0] for call in calls}) == [4, 8]


@pytest.mark.parametrize("failure", ["oom", "unsafe"])
def test_no_safe_baseline_fails_without_trying_a_smaller_batch(monkeypatch, failure):
    jobs, entry, persist, calls, _, _ = debug_harness(
        monkeypatch,
        report_factory=lambda job, batch, workers: debug_report(
            job, batch, workers, oom=failure == "oom", unsafe=failure == "unsafe"
        ),
    )
    with pytest.raises(RuntimeError, match="no.*candidate|no downscale"):
        calibration._calibrate_group(jobs, entry, persist)
    assert {call[0] for call in calls} == {4}
    assert "selected" not in entry
    assert entry.get("status") != "passed"


def test_resume_skips_completed_candidates_and_completed_partial_members(monkeypatch):
    interrupted = False

    def report(job, batch, workers):
        nonlocal interrupted
        if not interrupted and (batch, workers, job["condition"]) == (8, 2, "pe"):
            interrupted = True
            raise KeyboardInterrupt("explicit debug interrupted calibration")
        return debug_report(job, batch, workers, rate=float(batch * workers), unsafe=batch >= 16)

    jobs, entry, persist, calls, snapshots, _ = debug_harness(monkeypatch, report_factory=report)
    with pytest.raises(KeyboardInterrupt, match="explicit debug"):
        calibration._calibrate_group(jobs, entry, persist)
    assert snapshots[-1]["candidates"][-1]["status"] == "running"
    assert len(snapshots[-1]["candidates"][-1]["measurements"]) == 1
    completed = set(calls[:-1])
    before = len(calls)
    calibration._calibrate_group(jobs, entry, persist)
    resumed = set(calls[before:])
    assert completed.isdisjoint(resumed)
    assert (8, 2, "pe", 0) in resumed
    assert entry["selected"] == {"batch_size": 8, "workers": 4}
    before = len(calls)
    calibration._calibrate_group(jobs, entry, persist)
    assert len(calls) == before


def test_floor_larger_than_whole_split_is_kept_as_configured(monkeypatch):
    jobs, entry, persist, calls, _, _ = debug_harness(monkeypatch, baseline=4, maximum=2)
    calibration._calibrate_group(jobs, entry, persist)
    assert {call[0] for call in calls} == {4}
    assert entry["selected"]["batch_size"] == 4
    assert entry["natural_training_split_size"] == 2
    assert entry["stop_reason"] == "complete_training_split_boundary"


def test_full_graph_has_no_artificial_minibatch_axis(monkeypatch):
    jobs, entry, persist, calls, _, _ = debug_harness(
        monkeypatch, axis="full_graph", baseline=1, maximum=1, workers=(0,)
    )
    calibration._calibrate_group(jobs, entry, persist)
    assert {call[:2] for call in calls} == {(1, 0)}
    assert len(calls) == 2
    assert entry["batch_axis"] == "full_graph"
    assert entry["stop_reason"] == "full_graph_no_batch_axis"
    assert entry["selected"]["batch_size"] == 1


def test_sampled_v5_selection_updates_seed_batch_not_graph_batch(monkeypatch):
    jobs, entry, persist, calls, _, _ = debug_harness(
        monkeypatch, axis="sampled_seed_nodes", baseline=4, maximum=10, workers=(0,)
    )
    calibration._calibrate_group(jobs, entry, persist)
    assert sorted({call[0] for call in calls}) == [4, 8, 10]
    assert entry["selected"] == {"batch_size": 4, "workers": 0, "sample_seed_batch_size": 10}


def test_training_args_accept_actual_generated_cycle_se_pe_commands(monkeypatch, tmp_path):
    from scripts import run_cycle_scaling

    # Parsing only: mocks CUDA availability for existing parser validation,
    # never allocates a CUDA tensor or invokes a model.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = run_cycle_scaling.parser().parse_args(
        [
            "--versions",
            "v2",
            "--encodings",
            "se",
            "pe",
            "--profiles",
            "reference",
            "large",
            "--model-seeds",
            "0",
            "--hardware-profile",
            "a6000-48gb",
            "--min-free-gb",
            "40",
            "--device",
            "cuda:0",
        ]
    )
    jobs = run_cycle_scaling.make_jobs(args, tmp_path / "debug-command-generation-only")
    assert len(jobs) == 8
    for job in jobs:
        normalized = {
            **job,
            "track": "cycle",
            "dataset": job["datasets"][0],
            "condition": job["encoding"],
        }
        parsed = calibration._training_args(normalized)
        assert parsed.dataset == job["datasets"][0]
        assert parsed.datasets == job["datasets"]
        assert parsed.encoding == job["encoding"]
        assert parsed.hidden_dim == job["config"]["hidden_dim"]
        assert parsed.layers == job["config"]["layers"]
        assert parsed.pe_dim == job["config"]["pe_dim"]
        assert parsed.batch_size == job["resources"]["batch_size"]
        assert parsed.workers == job["resources"]["workers"]


def _debug_real_cycle_loader_plan(monkeypatch, tmp_path, selected_workers):
    """Real parser, DFS preparation/cache validation and identity; fake timing only."""
    from research.cycle_pe.v2 import calibration as cycle_calibration
    from research.cycle_pe.v2 import data

    official = SimpleNamespace(
        num_nodes=3,
        x=torch.tensor([[0], [1], [2]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]]),
        edge_attr=torch.ones(6, 1, dtype=torch.long),
        y=torch.tensor([0.7]),
    )

    def debug_official_split(_root, dataset, *, allow_download, splits):
        assert dataset == "zinc12k" and not allow_download and splits == ("train",)
        return {"train": [official]}

    # Explicit one-graph CPU fixture. Production sizes and defaults are untouched.
    monkeypatch.setattr(data, "load_official_splits", debug_official_split)
    monkeypatch.setitem(cycle_calibration.EXPECTED_SIZES, "zinc12k", (1, 1, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # Parser validation only.
    monkeypatch.setattr(calibration, "allocated_cpu_count", lambda: 16)
    jobs = [
        {
            "track": "cycle",
            "profile": "reference",
            "dataset": "zinc12k",
            "condition": encoding,
            "model_seed": 0,
            "device": "cuda:0",
            "command": [
                "python",
                "-B",
                "-m",
                "research.cycle_pe.v2.benchmark",
                "--datasets",
                "zinc12k",
                "--encoding",
                encoding,
                "--data-root",
                str(tmp_path / "debug-data"),
                "--output-dir",
                str(tmp_path / "debug-not-training"),
                "--workers",
                "8",
                "--batch-size",
                "2",
                "--device",
                "cuda:0",
            ],
        }
        for encoding in ("se", "pe")
    ]

    def measure(job, loaded, _args, *, batch_size, workers):
        assert len(loaded) == 1 and loaded[0].cycle_basis.shape == (3, 1)
        return debug_report(
            job, batch_size, workers, rate=200.0 if workers == selected_workers else 100.0
        )

    monkeypatch.setattr(calibration, "_measure", measure)
    entry = {}
    calibration._calibrate_group(jobs, entry, lambda: None)
    assert entry["input_identity"]["preparation_workers"] == 8
    assert entry["selected"]["workers"] == selected_workers
    for job in jobs:
        command = job["command"]
        command[command.index("--workers") + 1] = str(selected_workers)
    return jobs, entry, official


@pytest.mark.parametrize("selected_workers", [2, 4, 16])
def test_cycle_real_loader_accepts_measured_workers_without_mutating_legacy_plan(
    monkeypatch, tmp_path, capsys, selected_workers
):
    jobs, entry, _ = _debug_real_cycle_loader_plan(monkeypatch, tmp_path, selected_workers)
    before = copy.deepcopy(entry)
    calibration.verify_plan_inputs({"entries": [entry]}, jobs)
    assert entry == before
    assert entry["input_identity"]["preparation_workers"] == 8
    assert f"preparation_workers 8 -> {selected_workers}" in capsys.readouterr().out


def test_cycle_real_loader_still_rejects_changed_official_graph_before_plan_reuse(
    monkeypatch, tmp_path
):
    from chartgat.cache import CacheWrongRequestError

    jobs, entry, official = _debug_real_cycle_loader_plan(monkeypatch, tmp_path, 2)
    before = copy.deepcopy(entry)
    official.y.add_(1)
    with pytest.raises(CacheWrongRequestError, match="Mismatched cycle PE v2 cache"):
        calibration.verify_plan_inputs({"entries": [entry]}, jobs)
    assert entry == before


@pytest.mark.parametrize(
    "field",
    [
        "input_identity.split_content_sha256.train",
        "input_identity.split_sizes.train",
        "input_identity.official_splits",
        "input_identity.preparation.implementation_sha256.v2/basis.py",
        "input_identity.basis_backend",
        "input_identity.cache_directory",
        "natural_training_split_size",
        "batch_axis",
    ],
)
def test_cycle_worker_exception_keeps_all_scientific_identity_fields_strict(
    monkeypatch, tmp_path, field
):
    jobs, entry, _ = _debug_real_cycle_loader_plan(monkeypatch, tmp_path, 2)
    if field == "input_identity.preparation.implementation_sha256.v2/basis.py":
        entry["input_identity"]["preparation"]["implementation_sha256"]["v2/basis.py"] = "changed"
    else:
        parts = field.split(".")
        parent = entry
        for part in parts[:-1]:
            parent = parent[part]
        parent[parts[-1]] = "changed"
    # batch_axis affects the requested batch lookup but remains a required exact identity.
    before = copy.deepcopy(entry)
    with pytest.raises(ValueError, match="changed fields:") as error:
        calibration.verify_plan_inputs({"entries": [entry]}, jobs)
    assert field in str(error.value)
    assert entry == before


def test_conductance_does_not_ignore_a_preparation_workers_field(monkeypatch):
    jobs, entry, persist, _, _, identity = debug_harness(
        monkeypatch, axis="sampled_seed_nodes", workers=(0,)
    )
    identity["preparation_workers"] = 8
    calibration._calibrate_group(jobs, entry, persist)
    changed = {**identity, "preparation_workers": 2}
    monkeypatch.setattr(
        calibration, "_load_group", lambda *_args: (object(), changed, 64, "sampled_seed_nodes")
    )
    before = copy.deepcopy(entry)
    with pytest.raises(ValueError, match="input_identity.preparation_workers"):
        calibration.verify_plan_inputs({"entries": [entry]}, jobs)
    assert entry == before


def test_partial_cycle_resume_preserves_original_worker_observation(monkeypatch):
    jobs, entry, persist, calls, _, identity = debug_harness(monkeypatch, maximum=8)
    identity["preparation_workers"] = 8
    calibration._calibrate_group(jobs, entry, persist)
    before, previous_calls = copy.deepcopy(entry["input_identity"]), len(calls)
    reread_identity = {**identity, "preparation_workers": 2}
    monkeypatch.setattr(
        calibration, "_load_group", lambda *_args: (object(), reread_identity, 8, "graphs")
    )
    calibration._calibrate_group(jobs, entry, persist)
    assert entry["input_identity"] == before
    assert len(calls) == previous_calls


@pytest.mark.parametrize("change", ["data", "count", "axis"])
def test_partial_cycle_resume_rejects_true_input_changes_before_any_measurement(
    monkeypatch, change
):
    jobs, entry, persist, calls, _, identity = debug_harness(monkeypatch, maximum=8)
    calibration._calibrate_group(jobs, entry, persist)
    changed_identity = dict(identity)
    if change == "data":
        changed_identity["full_split_sha256"] = "b" * 64
    maximum = 9 if change == "count" else 8
    axis = "full_graph" if change == "axis" else "graphs"
    monkeypatch.setattr(
        calibration, "_load_group", lambda *_args: (object(), changed_identity, maximum, axis)
    )
    before, previous_calls = copy.deepcopy(entry), len(calls)
    with pytest.raises(ValueError, match="since partial calibration.*changed fields:"):
        calibration._calibrate_group(jobs, entry, persist)
    assert entry == before and len(calls) == previous_calls
