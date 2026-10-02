"""Debug integration checks; never run the full training profile."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.wedge_propagation.learned.data import prepare_cases
from research.wedge_propagation.learned.evaluation import (
    evaluate,
    intervention_manifest,
    interventions,
    metrics_for_batch,
)
from research.wedge_propagation.learned.model import make_model
from research.wedge_propagation.learned.train import (
    cache_batches,
    read_config,
    restore_checkpoint,
    source_manifest,
    train_gate,
)


@pytest.fixture(scope="module")
def debug_data():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    path = (
        Path(__file__).resolve().parents[1] / "research/wedge_propagation/learned/config_debug.json"
    )
    config = read_config(path, "debug")
    cases = prepare_cases(config, 1)
    yield config, cases
    torch.set_num_threads(previous)


def _cache(cases, split):
    return cache_batches(
        [case for case in cases if case.split == split], 12, torch.device("cpu"), torch.float64
    )


def test_frozen_interventions_and_macro_metrics(debug_data):
    config, cases = debug_data
    cached = _cache(cases, "validation")
    model = make_model("learned", config["model_seeds"], config["hidden"], 1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rows, patterns = evaluate(model, cached, "path", "learned", config["model_seeds"], 1e-8)
    assert len(rows) == 12
    plan = intervention_manifest(
        [case for case in cases if case.split != "train"], config["master_seed"]
    )
    changed = interventions(model, cached, patterns, plan, "path", config["model_seeds"], 1e-8)
    assert len(changed) == 60

    def keys(row):
        return row["seed"], row["graph_id"]

    identity = {keys(row): row for row in changed if row["intervention"] == "identity"}
    mean = {keys(row): row for row in changed if row["intervention"] == "mean"}
    for key in identity:
        assert identity[key]["message_relerr"] == pytest.approx(
            mean[key]["message_relerr"], abs=1e-10
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(
        row["weight_corr"] is None
        for row in changed
        if row["intervention"] == "correspondence_randomization"
    )


def test_checkpoint_resumes_same_seed_updates(debug_data, tmp_path):
    config, cases = debug_data
    train, validation = _cache(cases, "train"), _cache(cases, "validation")
    source = source_manifest()
    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    direct.mkdir()
    resumed.mkdir()
    first = make_model("learned", config["model_seeds"], config["hidden"], 1)
    history, selected = train_gate(
        first, train, validation, config, "path", "learned", direct, 12, source
    )
    saved = restore_checkpoint(
        direct / "checkpoints/path-learned-epoch0002.pt", config, torch.device("cpu")
    )
    second = make_model("learned", config["model_seeds"], config["hidden"], 1)
    continued, selection = train_gate(
        second, train, validation, config, "path", "learned", resumed, 12, source, saved
    )
    for name in selected["state_dict"]:
        torch.testing.assert_close(selection["state_dict"][name], selected["state_dict"][name])
    assert selected["best_epochs"] == selection["best_epochs"]
    for left, right in zip(history, continued, strict=True):
        assert left["epoch"] == right["epoch"] and left["seed"] == right["seed"]
        assert left["train_loss"] == pytest.approx(right["train_loss"], abs=1e-12)
    last = torch.load(direct / "checkpoints/path-learned-epoch0004.pt", weights_only=True)
    final = torch.load(resumed / "checkpoints/path-learned-epoch0004.pt", weights_only=True)
    for name in last["model"]:
        torch.testing.assert_close(last["model"][name], final["model"][name])


def test_config_profiles_cannot_silently_mix(debug_data):
    config, _ = debug_data
    path = (
        Path(__file__).resolve().parents[1] / "research/wedge_propagation/learned/config_debug.json"
    )
    with pytest.raises(ValueError, match="disagree"):
        read_config(path, "full")
    assert config["epochs"] == 4
    full = read_config(path.with_name("config_full.json"), "full")
    assert full["epochs"] == 500 and full["hidden"] == 64 and len(full["model_seeds"]) == 5


def test_negative_target_has_no_weight_ground_truth(debug_data):
    config, cases = debug_data
    model = make_model("learned", config["model_seeds"], config["hidden"], 1)
    rows, _ = evaluate(model, _cache(cases, "id"), "L", "learned", config["model_seeds"], 1e-8)
    assert all(row["weight_corr"] is None and row["weight_relerr"] is None for row in rows)
    assert all(np.isfinite(row["message_relerr"]) for row in rows)


@pytest.mark.parametrize("invalid", [None, ["identity"], ["identity", "mean", "typo"]])
def test_invalid_intervention_contract_fails_before_training(debug_data, tmp_path, invalid):
    config, _ = debug_data
    changed = dict(config)
    if invalid is None:
        changed.pop("interventions")
    else:
        changed["interventions"] = invalid
    path = tmp_path / "invalid_config.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="all five exact interventions"):
        read_config(path, "debug")


def test_metrics_refuse_wrong_graph_correspondence(debug_data):
    config, cases = debug_data
    cached = _cache(cases, "validation")
    group, batch = cached[0]
    model = make_model("learned", config["model_seeds"], config["hidden"], 1)
    pred, weights = model(batch)
    with pytest.raises(ValueError, match="IDs/order"):
        metrics_for_batch(
            list(reversed(group)),
            batch,
            pred,
            weights,
            "path",
            "learned",
            config["model_seeds"],
            model,
            1e-8,
        )
