"""CPU-only unit contracts for immutable, paired measured resource integration."""

from __future__ import annotations

import copy

import pytest

from scripts import run_conductance_scaling as conductance
from scripts import run_cycle_scaling as cycle
from scripts.training_resource_plan import command_identity


def _value(command, option):
    return command[command.index(option) + 1]


@pytest.fixture
def resource_fixture(monkeypatch):
    """Test only runner plumbing; shared plan validation has its own tests."""
    plan = {"fixture_identity": "measured-v1"}
    rows = {}
    bindings = []

    def selected(actual, *, track, profile, dataset):
        assert actual is plan
        return copy.deepcopy(rows.get((track, profile, dataset)))

    def validate(actual, **contract):
        if actual is not None:
            assert actual is plan
            bindings.append(contract)

    for runner in (conductance, cycle):
        monkeypatch.setattr(runner, "selected_resources", selected)
        monkeypatch.setattr(runner, "validate_job_plan", validate)
        monkeypatch.setattr(runner, "resource_plan_identity", lambda value: dict(value))
    return plan, rows, bindings


def test_v5_pairs_share_measured_resources_per_profile_and_dataset(tmp_path, resource_fixture):
    plan, rows, bindings = resource_fixture
    args = conductance.parser().parse_args(
        [
            "--versions",
            "v4",
            "v5",
            "--datasets",
            "ppi",
            "ogbn-arxiv",
            "--hardware-profile",
            "a6000-48gb",
        ]
    )
    args.resolved_resource_plan = plan
    for profile, ppi_batch, seed_batch in (("reference", 12, 8192), ("large", 8, 4096)):
        for dataset in args.datasets:
            rows["conductance", profile, dataset] = {
                "batch_size": ppi_batch if dataset == "ppi" else 1,
                "sample_seed_batch_size": seed_batch,
                "workers": 8 if dataset == "ppi" else 0,
            }
    jobs = conductance.make_jobs(args, tmp_path)
    v5_jobs = [job for job in jobs if job["version"] == "v5"]
    assert len(v5_jobs) == len(bindings) == 8
    assert {binding["condition"] for binding in bindings} == {"fixed_c", "shared_dynamic_c"}
    for job in v5_jobs:
        measured = rows["conductance", job["profile"], job["dataset"]]
        assert job["workers"] == job["execution"]["dataloader_workers"] == measured["workers"]
        assert job["batch_size"] == job["execution"]["batch_size"] == measured["batch_size"]
        for option, key in (
            ("--workers", "workers"),
            ("--batch-size", "batch_size"),
            ("--sample-seed-batch-size", "sample_seed_batch_size"),
        ):
            assert int(_value(job["command"], option)) == measured[key]
    for job in jobs:
        if job["version"] == "v4" and job["dataset"] == "ppi":
            assert job["workers"] == 4
            assert job["batch_size"] == 2


def test_legacy_cycle_explicit_batch_remains_independent_of_measured_v2(resource_fixture):
    plan, rows, _ = resource_fixture
    args = cycle.parser().parse_args(
        [
            "--versions",
            "v1",
            "v2",
            "--profiles",
            "reference",
            "--datasets",
            "zinc12k",
            "--legacy-batch-size",
            "64",
        ]
    )
    args.resolved_resource_plan = plan
    rows["cycle", "reference", "zinc12k"] = {"batch_size": 128, "workers": 2}
    cycle._validate(args)
    assert cycle._job_resources(args, "v1", "reference", "zinc12k")["batch_size"] == 64
    assert cycle._job_resources(args, "v2", "reference", "zinc12k", "se")["batch_size"] == 128
    assert cycle._run_configuration(args)["legacy_batch_size"] == 64


@pytest.mark.parametrize(
    "option,value",
    [
        ("--v5-ppi-batch-size", "3"),
        ("--v5-sample-seed-batch-size", "64"),
        ("--workers", "4"),
    ],
)
def test_v5_conflicting_explicit_resource_override_rejected(
    tmp_path, resource_fixture, option, value
):
    plan, rows, _ = resource_fixture
    args = conductance.parser().parse_args(
        ["--versions", "v5", "--profiles", "reference", "--datasets", "ppi", option, value]
    )
    args.resolved_resource_plan = plan
    args.workers_explicit = option == "--workers"
    rows["conductance", "reference", "ppi"] = {
        "batch_size": 8,
        "sample_seed_batch_size": 4096,
        "workers": 8,
    }
    with pytest.raises(ValueError, match="conflicts"):
        conductance.make_jobs(args, tmp_path)


def test_v5_matching_explicit_resources_are_accepted(tmp_path, resource_fixture):
    plan, rows, _ = resource_fixture
    args = conductance.parser().parse_args(
        [
            "--versions",
            "v5",
            "--profiles",
            "reference",
            "--datasets",
            "ppi",
            "--v5-ppi-batch-size",
            "8",
            "--v5-sample-seed-batch-size",
            "4096",
            "--workers",
            "8",
        ]
    )
    args.resolved_resource_plan = plan
    args.workers_explicit = True
    rows["conductance", "reference", "ppi"] = {
        "batch_size": 8,
        "sample_seed_batch_size": 4096,
        "workers": 8,
    }
    conductance._validate(args)
    assert len(conductance.make_jobs(args, tmp_path)) == 2


def test_cycle_se_pe_and_selected_tests_share_measured_profile_resources(
    tmp_path, resource_fixture
):
    plan, rows, bindings = resource_fixture
    args = cycle.parser().parse_args(["--versions", "v1", "v2", "--datasets", "zinc12k"])
    args.resolved_resource_plan = plan
    rows["cycle", "reference", "zinc12k"] = {"batch_size": 1024, "workers": 8}
    rows["cycle", "large", "zinc12k"] = {"batch_size": 512, "workers": 4}
    jobs = cycle.make_jobs(args, tmp_path)
    assert len(bindings) == 4
    assert {binding["condition"] for binding in bindings} == {"se", "pe"}
    selections = []
    for job in jobs:
        if job["version"] == "v1":
            assert job["resources"]["batch_size"] == 32
            continue
        measured = rows["cycle", job["profile"], "zinc12k"]
        for key in ("batch_size", "workers"):
            assert job["resources"][key] == measured[key]
            assert int(_value(job["command"], "--" + key.replace("_", "-"))) == measured[key]
        selections.append(
            {
                "version": "v2",
                "encoding": job["encoding"],
                "dataset": "zinc12k",
                "model_seed": 0,
                "selected_profile": job["profile"],
                "checkpoint": str(tmp_path / "fixture.pt"),
                "checkpoint_id": job["job_id"],
                "profile_selection_id": job["job_id"],
                "checkpoint_sha256": "0" * 64,
                "selected_validation_mae": 0.1,
                "trainable_parameters": 100,
            }
        )
    for job in cycle.make_test_jobs(args, tmp_path, selections):
        measured = rows["cycle", job["selected_profile"], "zinc12k"]
        assert job["resources"]["batch_size"] == measured["batch_size"]
        assert job["resources"]["workers"] == measured["workers"]


@pytest.mark.parametrize("option,value", [("--batch-size", "32"), ("--workers", "4")])
def test_cycle_conflicting_resources_rejected(tmp_path, resource_fixture, option, value):
    plan, rows, _ = resource_fixture
    args = cycle.parser().parse_args(
        ["--versions", "v2", "--profiles", "reference", "--datasets", "zinc12k", option, value]
    )
    args.resolved_resource_plan = plan
    rows["cycle", "reference", "zinc12k"] = {"batch_size": 64, "workers": 8}
    with pytest.raises(ValueError, match="conflicts"):
        cycle.make_jobs(args, tmp_path)


@pytest.mark.parametrize(
    "runner,version,dataset",
    [
        (conductance, "v5", "ppi"),
        (cycle, "v2", "zinc12k"),
    ],
)
def test_missing_required_plan_row_fails_before_training(
    tmp_path, resource_fixture, runner, version, dataset
):
    plan, _, _ = resource_fixture
    args = runner.parser().parse_args(
        ["--versions", version, "--profiles", "reference", "--datasets", dataset]
    )
    args.resolved_resource_plan = plan
    with pytest.raises(ValueError, match="missing"):
        runner.make_jobs(args, tmp_path)


def test_cycle_resume_identity_contains_measured_plan_digest(resource_fixture):
    plan, _, _ = resource_fixture
    args = cycle.parser().parse_args([])
    baseline = cycle._run_configuration(args)
    args.resolved_resource_plan = plan
    measured = cycle._run_configuration(args)
    assert baseline["resource_plan"] is None
    assert measured["resource_plan"] == plan
    assert measured != baseline


@pytest.mark.parametrize("batch,allowed", [(1, False), (2, True), (8, True)])
def test_portable_v5_ppi_batch_has_floor_not_measurement_blocking_ceiling(batch, allowed):
    args = conductance.parser().parse_args(["--v5-ppi-batch-size", str(batch)])
    if allowed:
        conductance._validate(args)
    else:
        with pytest.raises(ValueError, match="at least 2"):
            conductance._validate(args)


@pytest.mark.parametrize(
    "runner,version,dataset,mutation",
    [
        (conductance, "v5", "ppi", "epochs"),
        (conductance, "v5", "ppi", "v5_beta_initial"),
        (cycle, "v2", "zinc12k", "epochs"),
        (cycle, "v2", "zinc12k", "lr"),
    ],
)
def test_real_command_binding_rejects_changed_recipe(tmp_path, runner, version, dataset, mutation):
    args = runner.parser().parse_args(
        ["--versions", version, "--profiles", "reference", "--datasets", dataset]
    )
    baseline = runner.make_jobs(args, tmp_path / "original")
    track = "conductance" if runner is conductance else "cycle"
    selected = (
        {"batch_size": 4, "sample_seed_batch_size": 4096, "workers": 8}
        if runner is conductance
        else {"batch_size": 64, "workers": 8}
    )
    args.resolved_resource_plan = {
        "entries": [
            {
                "track": track,
                "profile": "reference",
                "dataset": dataset,
                "selected": selected,
                "job_contracts": [
                    {
                        "condition": job["condition"] if runner is conductance else job["encoding"],
                        "model_seed": job["model_seed"],
                        "argv_sha256": command_identity(job["command"]),
                    }
                    for job in baseline
                ],
            }
        ]
    }
    measured_jobs = runner.make_jobs(args, tmp_path / "measured")
    assert len(measured_jobs) == len(baseline)
    assert all(
        int(_value(job["command"], "--batch-size")) == selected["batch_size"]
        for job in measured_jobs
    )
    setattr(args, mutation, getattr(args, mutation) * 2)
    with pytest.raises(ValueError, match="scientific recipe"):
        runner.make_jobs(args, tmp_path / "changed")


@pytest.mark.parametrize("runner", [conductance, cycle])
def test_unreadable_plan_returns_safe_error_without_creating_results(tmp_path, runner, capsys):
    results = tmp_path / "outputs"
    result = runner.main(
        [
            "--resource-plan",
            str(tmp_path / "missing-plan.json"),
            "--results-root",
            str(results),
            "--dry-run",
        ]
    )
    assert result == 2
    assert not results.exists()
    assert "missing-plan.json" in capsys.readouterr().err


@pytest.mark.parametrize("worker_argument", [["--workers", "4"], ["--workers=4"]])
def test_conductance_main_tracks_explicit_worker_conflict(
    tmp_path, monkeypatch, resource_fixture, worker_argument
):
    plan, rows, _ = resource_fixture
    rows["conductance", "reference", "ppi"] = {
        "batch_size": 8,
        "sample_seed_batch_size": 4096,
        "workers": 8,
    }
    monkeypatch.setattr(conductance, "load_resource_plan", lambda *a, **kw: plan)
    assert (
        conductance.main(
            [
                "--versions",
                "v5",
                "--profiles",
                "reference",
                "--datasets",
                "ppi",
                "--resource-plan",
                str(tmp_path / "fixture.json"),
                "--dry-run",
                *worker_argument,
            ]
        )
        == 2
    )
