"""Complete training budgets and validation-only selection guards."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.local_energy_relations.placement.common import read_config
from research.local_energy_relations.placement.model import parse_condition
from research.local_energy_relations.placement.evaluation import intervention_scopes, intervention_variants
from research.local_energy_relations.placement.study import (
    _worker,
    _validate_final_training,
    _workers,
    build_jobs,
    run,
    select_learning_rates,
    verify_coverage,
    verify_frozen_coverage,
)

_FOLDER = Path(__file__).resolve().parents[1] / "research/local_energy_relations/placement"


def _config(profile="debug"):
    return read_config(_FOLDER / f"config_{profile}.json", profile)


def _tuning_rows(config):
    training = config["training"]
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "placement": parse_condition(condition)[2],
            "lr": lr,
            "seed": seed,
            "phase": "tuning",
            "best_epoch": 1,
            "validation_ce": 1 + lr,
            "validation_accuracy": 0.5,
            "training_epochs": training["epochs_per_run"],
            "optimizer_updates": training["epochs_per_run"],
            "new_optimizer_updates": training["epochs_per_run"],
        }
        for dataset, condition, lr, seed in itertools.product(
            config["data"]["datasets"],
            config["conditions"],
            training["learning_rate_candidates"],
            training["tuning_seeds"],
        )
    ]


def _final_rows(config):
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "placement": parse_condition(condition)[2],
            "seed": seed,
            "split": split,
            "ce": 0.9,
            "accuracy": 0.5,
            "num_nodes": config["data"]["expected_shapes"][dataset]["nodes"],
            "num_labeled_nodes": config["data"]["expected_shapes"][dataset][split],
        }
        for dataset, condition, seed, split in itertools.product(
            config["data"]["datasets"],
            config["conditions"],
            config["training"]["final_seeds"],
            ["train", "validation", "test"],
        )
    ]


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_complete_declared_budgets_and_all_job_conditions(profile):
    config = _config(profile)
    tuning = _tuning_rows(config)
    final = _final_rows(config)
    assert len(tuning) == config["training"]["tuning_runs"]
    assert len(final) == 3 * config["training"]["final_runs"]
    assert len(config["conditions"]) == 20
    if profile == "full":
        assert config["backbone"]["layers"] == 2 and config["backbone"]["hidden_dim"] == 64
        assert config["training"]["epochs_per_run"] == 500
        assert config["training"]["total_runs"] == 840
        assert config["training"]["total_updates"] == 420000
    else:
        assert config["training"]["total_runs"] == 360
        assert config["training"]["total_updates"] == 1080
    tuning_jobs = build_jobs(config, "tuning")
    selected = select_learning_rates(config, tuning)
    final_jobs = build_jobs(config, "final", selected)
    assert len({job["job_key"] for job in tuning_jobs}) == len(tuning_jobs)
    assert all(job["seeds"] == config["training"]["tuning_seeds"] for job in tuning_jobs)
    assert all(job["seeds"] == config["training"]["final_seeds"] for job in final_jobs)
    assert verify_coverage(config, tuning, final)["all_datasets_conditions_seeds_splits"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("backbone", "layers", 1),
        ("backbone", "hidden_dim", 8),
        ("data", "sampling_ratio", 0.5),
        ("training", "epochs_per_run", 3),
        ("training", "learning_rate_candidates", [0.003]),
        ("training", "final_seeds", [11]),
        ("training", "test_evaluation_during_tuning", True),
    ],
)
def test_full_config_cannot_shrink_or_leak_test(tmp_path, section, key, value):
    config = json.loads((_FOLDER / "config_full.json").read_text())
    config[section][key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(path, "full")


@pytest.mark.parametrize(
    "alteration",
    [
        "missing",
        "duplicate",
        "wrong_seed",
        "nonfinite",
        "short_training",
        "short_updates",
        "test_field",
        "wrong_placement",
    ],
)
def test_validation_selection_rejects_corrupt_or_leaked_tuning(alteration):
    config = _config()
    rows = _tuning_rows(config)
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif alteration == "wrong_seed":
        rows[0]["seed"] = -1
    elif alteration == "nonfinite":
        rows[0]["validation_ce"] = float("nan")
    elif alteration == "short_training":
        rows[0]["training_epochs"] -= 1
    elif alteration == "short_updates":
        rows[0]["optimizer_updates"] -= 1
    elif alteration == "test_field":
        rows[0]["test_ce"] = 0.3
    else:
        rows[0]["placement"] = "output"
    with pytest.raises(ValueError):
        select_learning_rates(config, rows)


def test_lr_selection_averages_all_independent_tuning_seeds_with_tie_rule():
    config = _config()
    rows = _tuning_rows(config)
    rates = config["training"]["learning_rate_candidates"]
    for row in rows:
        row["validation_ce"] = 0.5
        if row["lr"] == max(rates):
            row["validation_ce"] = (
                0.25 if row["seed"] == config["training"]["tuning_seeds"][0] else 0.75
            )
    selected = select_learning_rates(config, list(reversed(rows)))
    assert all(row["selected_lr"] == min(rates) for row in selected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_nodes", 1),
        ("num_labeled_nodes", 1),
        ("split", "subset"),
        ("seed", -1),
        ("ce", float("nan")),
        ("placement", "invalid"),
    ],
)
def test_final_coverage_checks_all_nodes_and_public_splits(field, value):
    config = _config()
    rows = _final_rows(config)
    rows[0][field] = value
    with pytest.raises(ValueError):
        verify_coverage(config, _tuning_rows(config), rows)


def test_all_final_training_must_be_locked_before_test():
    config = _config()
    jobs = build_jobs(config, "final", select_learning_rates(config, _tuning_rows(config)))
    rows = [
        {
            "dataset": job["dataset"],
            "condition": job["condition"],
            "placement": job["placement"],
            "seed": seed,
            "lr": job["lr"],
            "phase": "final",
            "training_epochs": 3,
            "optimizer_updates": 3,
        }
        for job in jobs
        for seed in job["seeds"]
    ]
    _validate_final_training(config, rows, jobs)
    with pytest.raises(ValueError, match="before unlocking test"):
        _validate_final_training(config, rows[:-1], jobs)


def test_existing_output_and_full_local_training_rejected(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    common = dict(profile="debug", config=None, output_dir=str(existing), device="cpu")
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(**common))
    common.update(profile="full", output_dir=str(tmp_path / "new"))
    with pytest.raises(ValueError, match="Linux server CUDA"):
        run(SimpleNamespace(**common))


def test_frozen_coverage_rejects_missing_branch_and_intervention_rows():
    config = _config()
    with pytest.raises(ValueError, match="incomplete frozen"):
        verify_frozen_coverage(config, {"intervention_rows": [], "branch_rows": []})


def _complete_frozen_rows(config):
    interventions, branches = [], []
    for dataset, condition, seed in itertools.product(
        config["data"]["datasets"], config["conditions"], config["training"]["final_seeds"]
    ):
        base = {
            "dataset": dataset, "condition": condition,
            "placement": parse_condition(condition)[2], "seed": seed,
        }
        branches.extend({**base, "layer": layer, "intervention": "original", "target": "none"} for layer in range(2))
        for intervention, target in itertools.product(
            intervention_variants(parse_condition(condition)[1]), intervention_scopes(condition)
        ):
            interventions.extend({**base, "split": split, "intervention": intervention, "target": target} for split in ("train", "validation", "test"))
            branches.extend({**base, "layer": layer, "intervention": intervention, "target": target} for layer in range(2))
    return {"intervention_rows": interventions, "branch_rows": branches}


def test_frozen_coverage_requires_active_placements_without_no_op_scope():
    config = _config()
    evaluated = _complete_frozen_rows(config)
    coverage = verify_frozen_coverage(config, evaluated)
    assert coverage == {"intervention_rows": 900, "branch_rows": 840}
    wrong = copy.deepcopy(evaluated)
    candidate = next(row for row in wrong["intervention_rows"] if row["placement"] == "hidden")
    candidate["target"] = "layer_1"
    with pytest.raises(ValueError, match="out-of-contract"):
        verify_frozen_coverage(config, wrong)
    wrong = copy.deepcopy(evaluated)
    wrong["branch_rows"][0]["placement"] = "hidden"
    with pytest.raises(ValueError, match="placement"):
        verify_frozen_coverage(config, wrong)


def _worker_fixture(tmp_path, profile="debug", device="cpu"):
    output = tmp_path / "run"
    output.mkdir()
    (output / "workers").mkdir()
    (output / "config.json").write_text(json.dumps(_config(profile)))
    plan = {
        "output_dir": str(output), "profile": profile, "device": device,
        "stage": "evaluation", "worker_index": 0, "num_workers": 1,
        "source": {"code_digest": "DEBUG-source"},
        "test_unlock_digest": "incorrect-lock", "jobs": [], "graphs": {},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return output, path


def test_worker_rejects_source_changed_after_dispatch_before_loading_graph(tmp_path, monkeypatch):
    import research.local_energy_relations.placement.study as study
    output, path = _worker_fixture(tmp_path)
    monkeypatch.setattr(study, "source_manifest", lambda: {"code_digest": "changed"})
    with pytest.raises(ValueError, match="source changed after dispatch"):
        _worker(path)
    assert not list((output / "workers").iterdir())


def test_direct_worker_cannot_bypass_linux_server_guard(tmp_path, monkeypatch):
    import research.local_energy_relations.placement.study as study
    output, path = _worker_fixture(tmp_path, "full", "cuda:0")
    monkeypatch.setattr(study, "source_manifest", lambda: {"code_digest": "DEBUG-source"})
    monkeypatch.setattr(study.sys, "platform", "win32")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="Linux allocated CUDA"):
        _worker(path)
    assert not list((output / "workers").iterdir())


def test_worker_evaluation_requires_unchanged_final_checkpoint_lock(tmp_path, monkeypatch):
    import research.local_energy_relations.placement.study as study
    output, path = _worker_fixture(tmp_path)
    (output / "test_evaluation_unlocked.json").write_text(json.dumps({"all_final_checkpoints_locked": True}))
    monkeypatch.setattr(study, "source_manifest", lambda: {"code_digest": "DEBUG-source"})
    with pytest.raises(ValueError, match="test checkpoint lock changed"):
        _worker(path)
    assert not (output / "workers/evaluation-0/results.json").exists()


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("1", 1), ("2", 2), ("8", 8)])
def test_cli_workers_reaches_data_loader_as_integer(value, expected):
    assert _workers(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid", "True"])
def test_cli_workers_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _workers(value)
