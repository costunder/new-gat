"""CPU-only V5 learning-budget forwarding; no hardware probes or training."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.conductance_gat.v5 import train
from scripts import run_conductance_scaling as scaling
from scripts import run_conductance_v5 as standalone
from scripts import run_rich_scaling as rich


def _value(command, flag):
    return command[command.index(flag) + 1]


def _jobs(module, args, output):
    module._validate(args)
    return (
        standalone.make_jobs(args, output, standalone._architecture(args))
        if module is standalone
        else scaling.make_jobs(args, output)
    )


@pytest.mark.parametrize(
    "module,prefix,scope",
    [
        (standalone, "", ["--datasets", "ppi"]),
        (scaling, "v5-", ["--versions", "v5", "--datasets", "ppi", "--profiles", "reference"]),
    ],
)
@pytest.mark.parametrize("reference", [None, 8])
def test_budget_reaches_every_real_child_without_entering_model_architecture(
    tmp_path, module, prefix, scope, reference
):
    flags = [*scope, f"--{prefix}learning-budget-policy", "reference_updates"]
    if reference is not None:
        flags += [f"--{prefix}budget-reference-batch-size", str(reference)]
    args = module.parser().parse_args(flags)
    base = module.parser().parse_args(scope)
    jobs, original = _jobs(module, args, tmp_path), _jobs(module, base, tmp_path)
    assert len(jobs) == len(original) == 2
    for job, previous in zip(jobs, original, strict=True):
        assert job["architecture"] == previous["architecture"]
        assert job["learning_budget"] == {
            "learning_budget_policy": "reference_updates",
            "budget_reference_batch_size": reference,
        }
        assert "learning_budget_policy" not in job["architecture"]
        assert "budget_reference_batch_size" not in job["architecture"]
        assert _value(job["command"], "--learning-budget-policy") == "reference_updates"
        assert ("--budget-reference-batch-size" in job["command"]) is (reference is not None)
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        assert child.learning_budget_policy == "reference_updates"
        assert child.budget_reference_batch_size == reference
        assert child.epochs == args.epochs and child.patience == args.patience
        assert train.configuration(child)["learning_budget_policy"] == "reference_updates"


@pytest.mark.parametrize(
    "module,prefix,scope",
    [
        (standalone, "", ["--datasets", "ppi"]),
        (scaling, "v5-", ["--versions", "v5", "--datasets", "ppi"]),
    ],
)
def test_default_epoch_budget_does_not_change_legacy_job_identity(tmp_path, module, prefix, scope):
    implicit = module.parser().parse_args(scope)
    explicit = module.parser().parse_args([*scope, f"--{prefix}learning-budget-policy", "epochs"])
    assert _jobs(module, implicit, tmp_path) == _jobs(module, explicit, tmp_path)
    for job in _jobs(module, implicit, tmp_path):
        assert "learning_budget" not in job
        assert "--learning-budget-policy" not in job["command"]
        assert "--budget-reference-batch-size" not in job["command"]
        assert "--solver-cost-scaling" not in job["command"]


@pytest.mark.parametrize(
    "options",
    [
        ["--learning-budget-policy", "reference_updates", "--training-schedule", "staged"],
        ["--budget-reference-batch-size", "8"],
        ["--learning-budget-policy", "reference_updates", "--budget-reference-batch-size", "0"],
    ],
)
def test_standalone_budget_errors_are_rejected_before_child_launch(options):
    args = standalone.parser().parse_args(["--datasets", "ppi", *options])
    with pytest.raises(ValueError):
        standalone._validate(args)


@pytest.mark.parametrize("module", [standalone, scaling])
def test_child_cannot_report_epoch_recipe_for_a_requested_update_budget(tmp_path, module):
    prefix = "" if module is standalone else "v5-"
    scope = ["--datasets", "ppi"]
    if module is scaling:
        scope += ["--versions", "v5", "--profiles", "reference"]
    args = module.parser().parse_args(
        [*scope, f"--{prefix}learning-budget-policy", "reference_updates"]
    )
    job = _jobs(module, args, tmp_path)[0]
    payload = {
        "status": "passed",
        "dataset": job["dataset"],
        "condition": job["condition"],
        "model_seed": 0,
        "evaluation_split": "validation",
        "test_evaluated": False,
        "configuration": {
            **job["architecture"],
            "sampling": job["sampling"],
            "workers": job["workers"],
            "batch_size": job["batch_size"],
        },
    }
    path = Path(job["metrics_path"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="learning budget"):
        (standalone._load_metrics if module is standalone else scaling._load_child)(job)


def test_rich_budget_is_immutable_config_but_does_not_affect_other_tracks(tmp_path):
    flags = ["--tracks", "conductance", "--conductance-versions", "v5"]
    original = rich.parser().parse_args(flags)
    updated = rich.parser().parse_args([*flags, "--v5-learning-budget-policy", "reference_updates"])
    before = rich._config_payload(original, data_root=tmp_path, results_root=tmp_path)
    after = rich._config_payload(updated, data_root=tmp_path, results_root=tmp_path)
    assert "v5_learning_budget" not in before
    assert after.pop("v5_learning_budget") == {
        "learning_budget_policy": "reference_updates",
        "budget_reference_batch_size": None,
    }
    assert before == after
    other = rich.parser().parse_args(["--tracks", "cycle", "tree"])
    changed = copy.deepcopy(other)
    changed.v5_learning_budget_policy = "reference_updates"
    changed.v5_budget_reference_batch_size = 8
    assert rich.make_jobs(other, "unchanged") == rich.make_jobs(changed, "unchanged")
    assert rich._config_payload(
        other, data_root=tmp_path, results_root=tmp_path
    ) == rich._config_payload(changed, data_root=tmp_path, results_root=tmp_path)


def test_explicit_corrected_recipe_preserves_full_v5_matrix_and_probe_argv_contract(tmp_path):
    flags = [
        "--tracks",
        "conductance",
        "--conductance-versions",
        "v5",
        "--profiles",
        "reference",
        "large",
        "--model-seeds",
        "0",
        "--device",
        "cuda:0",
        "--hardware-profile",
        "a6000-48gb",
        "--min-free-gb",
        "40",
        "--v5-solver-cost-scaling",
        "width_scaled",
        "--v5-beta-initial",
        "0.5",
        "--v5-learning-budget-policy",
        "reference_updates",
        "--results-root",
        str(tmp_path),
    ]
    args = rich.parser().parse_args(flags)
    rich._validate(args)
    parents = rich.make_jobs(args, "corrected-v5-contract")
    assert len(parents) == 1 and parents[0]["track"] == "conductance"
    child_args = scaling.parser().parse_args(parents[0]["command"][3:])
    jobs = _jobs(scaling, child_args, Path(parents[0]["output_dir"]))
    assert len(jobs) == 20
    assert {job["dataset"] for job in jobs} == {"cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"}
    assert {job["profile"] for job in jobs} == {"reference", "large"}
    assert {job["model_seed"] for job in jobs} == {0}
    for job in jobs:
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        assert child.epochs == 200 and child.patience == 50
        assert child.solver_cost_scaling == "width_scaled" and child.beta_initial == 0.5
        assert child.learning_budget_policy == "reference_updates"
        assert child.budget_reference_batch_size is None
        assert child.hardware_profile == "a6000-48gb" and child.device == "cuda:0"
        assert "learning_budget_policy" not in job["architecture"]
        assert job["architecture"]["solver_cost_scaling"] == "width_scaled"
    request = rich._calibration_request(args, "corrected-v5-contract")
    assert len(request["jobs"]) == 20 and rich._needs_resource_calibration(args)

    def key(job):
        return job["profile"], job["dataset"], job["condition"], job["model_seed"]

    final_commands = {key(job): job["command"] for job in jobs}
    assert all(probe["command"] == final_commands[key(probe)] for probe in request["jobs"])
    assert all(
        _value(probe["command"], "--learning-budget-policy") == "reference_updates"
        for probe in request["jobs"]
    )
