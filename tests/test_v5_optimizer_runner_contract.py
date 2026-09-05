"""CPU-only configuration/identity tests; no model training or GPU measurement."""

from __future__ import annotations

import json

import pytest

from research.conductance_gat.v5.protocol import conductance_configuration
from scripts import run_conductance_scaling as scaling
from scripts import run_conductance_v5 as standalone
from scripts import run_rich_scaling as rich


def _option(command, option):
    return command[command.index(option) + 1]


@pytest.mark.parametrize(
    "override",
    [
        {"conductance_backend": "implicit_fallback"},
        {"solver_steps": 0},
        {"solver_steps": True},
        {"solver_steps": 2.5},
        {"solver_step_size": 0},
        {"solver_step_size": float("nan")},
        {"solver_entropy": 0},
        {"solver_entropy": float("inf")},
        {"solver_degree_barrier": -0.1},
    ],
)
def test_invalid_solver_contract_is_rejected(override):
    with pytest.raises(ValueError):
        conductance_configuration(**override)


def test_default_standalone_keeps_large_backbone_and_records_solver(tmp_path):
    args = standalone.parser().parse_args(["--profile", "large", "--datasets", "cora"])
    standalone._validate(args)
    architecture = standalone._architecture(args)
    assert (architecture["hidden_channels"], architecture["layers"], architecture["heads"]) == (
        384,
        12,
        8,
    )
    assert architecture["conductance_backend"] == "optimization"
    assert architecture["training_schedule"] == "joint"
    assert args.epochs == 300
    jobs = standalone.make_jobs(args, tmp_path, architecture)
    assert len(jobs) == 2
    for job in jobs:
        for name, value in conductance_configuration().items():
            assert job["architecture"][name] == value
            assert _option(job["command"], "--" + name.replace("_", "-")) == str(value)
        assert _option(job["command"], "--training-schedule") == "joint"


def test_explicit_mlp_staged_ablation_is_distinct_and_forwarded(tmp_path):
    base_options = ["--versions", "v5", "--datasets", "cora", "--profiles", "reference"]
    default = scaling.parser().parse_args(base_options)
    legacy = scaling.parser().parse_args(
        [*base_options, "--v5-conductance-backend", "mlp", "--v5-training-schedule", "staged"]
    )
    new_jobs = scaling.make_jobs(default, tmp_path)
    old_jobs = scaling.make_jobs(legacy, tmp_path)
    assert len(new_jobs) == len(old_jobs) == 2
    assert default.epochs == legacy.epochs == 200
    assert scaling._job_identity(new_jobs[0]) != scaling._job_identity(old_jobs[0])
    for job in old_jobs:
        assert job["architecture"]["conductance_backend"] == "mlp"
        assert job["architecture"]["training_schedule"] == "staged"
        assert _option(job["command"], "--conductance-backend") == "mlp"
        assert _option(job["command"], "--training-schedule") == "staged"


def test_custom_solver_configuration_reaches_nested_training_child(tmp_path):
    options = [
        "--tracks",
        "conductance",
        "--conductance-versions",
        "v5",
        "--v5-solver-steps",
        "12",
        "--v5-solver-step-size",
        "0.125",
        "--v5-solver-entropy",
        "0.75",
        "--v5-solver-degree-barrier",
        "0.2",
    ]
    args = rich.parser().parse_args(options)
    rich._validate(args)
    jobs = rich.make_jobs(args, "optimizer-check")
    assert len(jobs) == 1
    child_args = scaling.parser().parse_args(jobs[0]["command"][3:])
    child_jobs = scaling.make_jobs(child_args, tmp_path)
    assert len(child_jobs) == 20
    for job in child_jobs:
        assert job["architecture"]["solver_steps"] == 12
        assert job["architecture"]["solver_step_size"] == 0.125
        assert job["architecture"]["solver_entropy"] == 0.75
        assert job["architecture"]["solver_degree_barrier"] == 0.2
        assert _option(job["command"], "--solver-steps") == "12"


def test_c_options_do_not_change_cycle_or_tree_child_identity(tmp_path):
    first = rich.parser().parse_args(["--tracks", "cycle", "tree"])
    second = rich.parser().parse_args(
        [
            "--tracks",
            "cycle",
            "tree",
            "--v5-conductance-backend",
            "mlp",
            "--v5-training-schedule",
            "staged",
            "--v5-solver-steps",
            "17",
        ]
    )
    assert rich.make_jobs(first, "unchanged") == rich.make_jobs(second, "unchanged")
    first_config = rich._config_payload(first, data_root=tmp_path, results_root=tmp_path)
    second_config = rich._config_payload(second, data_root=tmp_path, results_root=tmp_path)
    assert first_config == second_config
    assert first_config["v5_conductance"] is None


def test_old_rich_config_cannot_resume_as_new_optimizer_and_file_is_preserved(tmp_path):
    args = rich.parser().parse_args(["--tracks", "conductance", "--conductance-versions", "v5"])
    new_config = rich._config_payload(args, data_root=tmp_path, results_root=tmp_path)
    old_config = {key: value for key, value in new_config.items() if key != "v5_conductance"}
    payload = {
        "schema_version": 1,
        "suite": "rich_scaling",
        "run_id": "existing",
        "config": old_config,
    }
    manifest_path = tmp_path / "manifest.json"
    original = json.dumps(payload).encode()
    manifest_path.write_bytes(original)
    with pytest.raises(ValueError, match="new run ID.*tracks conductance"):
        rich._resume_manifest(
            manifest_path,
            run_id="existing",
            expected_config=new_config,
            expected_jobs=[],
            expected_totals={},
            expected_sources={},
        )
    assert manifest_path.read_bytes() == original
