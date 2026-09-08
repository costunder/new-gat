"""Synthetic CPU checkpoint metadata tests, never evidence of real model training."""

from __future__ import annotations

import copy
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.edge_selection import integrity, train
from research.conductance_gat.v5.learning_budget import plan_learning_budget


def publish(case):
    identity = case.metrics["resume_identity"]
    identity_hash = train.base._canonical_sha256(identity)
    case.metrics["resume_identity_sha256"] = identity_hash
    case.best.update(resume_identity=identity, resume_identity_sha256=identity_hash)
    case.last.update(resume_identity=identity, resume_identity_sha256=identity_hash)
    torch.save(case.best, case.folder / "best.pt")
    best_hash = train.base.sha256_file(case.folder / "best.pt")
    case.metrics["checkpoint_sha256"] = best_hash
    case.last["best_checkpoint_sha256"] = best_hash
    torch.save(case.last, case.folder / "last.pt")
    (case.folder / "history.json").write_text(json.dumps(case.rows), encoding="utf-8")
    case.metrics.update(
        last_checkpoint_sha256=train.base.sha256_file(case.folder / "last.pt"),
        history_sha256=train.base.sha256_file(case.folder / "history.json"),
    )
    (case.folder / "metrics.json").write_text(json.dumps(case.metrics), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    arguments = train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "shared_dynamic_c",
            "--selection-mode",
            "full",
            "--output-dir",
            str(tmp_path),
            "--data-root",
            str(tmp_path / "debug-data"),
            "--epochs",
            "4",
            "--patience",
            "1",
            "--learning-budget-policy",
            "reference_updates",
        ]
    )
    train.validate_args(arguments)
    budget = plan_learning_budget(4, 1, 1, 1, "reference_updates")
    provenance = [{"explicit_synthetic_checkpoint_fixture": True}]
    protocol = {
        "data_sha256": "a" * 64,
        "split_sha256": {"train": "b" * 64, "validation": "c" * 64},
    }
    identity = train.build_identity(
        arguments, protocol, budget, "d" * 64, SimpleNamespace(provenance=provenance)
    )
    rows = [
        {
            "epoch": epoch,
            "train_batches": 1,
            "optimizer_steps": epoch,
            "processed_units": 3,
            "phase": {"phase": "joint"},
            "validation": score,
        }
        for epoch, score in enumerate((0.4, 0.5, 0.6, 0.7), 1)
    ]
    metrics = {
        "status": "passed",
        "research_suite": train.SUITE,
        "dataset": "cora",
        "condition": "full",
        "configuration": train.configuration(arguments),
        "resume_identity": identity,
        "source_sha256": identity["source_sha256"],
        "protocol": protocol,
        "learning_budget": budget,
        "initial_state_sha256": "d" * 64,
        "shared_initial_state_sha256": "e" * 64,
        "common_backbone_initial_state_sha256": "e" * 64,
        "topology": {"train_count": 3, "provenance": provenance},
        "epochs_run": 4,
        "optimizer_steps": 4,
        "best_epoch": 4,
        "best_validation": 0.7,
        "validation": 0.7,
        "test_evaluated": False,
        "debug": False,
        "subset": False,
    }
    last = {
        "epoch": 4,
        "optimizer_steps": 4,
        "history": rows,
        "best_epoch": 4,
        "best_validation": 0.7,
        "shared_initial_state_sha256": "e" * 64,
        "model_state": {"debug_weight": torch.tensor([1.0])},
        "optimizer_state": {
            "state": {0: {"step": torch.tensor(4.0)}},
            "param_groups": [{"params": [0]}],
        },
    }
    best = {
        "epoch": 4,
        "validation": 0.7,
        "selection_role": "primary",
        "model_state": {"debug_weight": torch.tensor([0.5])},
    }
    case = SimpleNamespace(
        folder=tmp_path, args=arguments, metrics=metrics, rows=rows, best=best, last=last
    )
    publish(case)
    return case


def test_completed_evidence_is_read_only_and_cpu_only(evidence, monkeypatch):
    before = {path.name: train.base.sha256_file(path) for path in evidence.folder.iterdir()}
    load = train.base.load_checkpoint_on_cpu
    last_tensor = []
    calls = []

    def cpu_load(path):
        if path.name == "best.pt":
            assert (
                last_tensor[0]() is None
            )  # No simultaneous last+best checkpoint tensor retention.
        saved = load(path)
        assert all(value.device.type == "cpu" for value in saved["model_state"].values())
        calls.append(path.name)
        if path.name == "last.pt":
            last_tensor.append(weakref.ref(saved["model_state"]["debug_weight"]))
        return saved

    monkeypatch.setattr(train.base, "load_checkpoint_on_cpu", cpu_load)
    assert integrity.inspect_completed(evidence.folder)["best_epoch"] == 4
    assert calls == ["last.pt", "best.pt"]
    assert before == {path.name: train.base.sha256_file(path) for path in evidence.folder.iterdir()}


@pytest.mark.parametrize(
    "damage,match",
    [
        ("seed", "configuration"),
        ("budget", "learning budget"),
        ("gap", "contiguous"),
        ("partial", "coverage"),
        ("updates", "coverage"),
        ("best", "strict maximum"),
        ("last_history", "last checkpoint"),
        ("last_steps", "last checkpoint"),
        ("last_best", "last checkpoint"),
        ("best_interior", "best checkpoint"),
        ("optimizer", "optimizer state"),
        ("protocol", "protocol"),
        ("source", "source_sha256"),
        ("shared", "last checkpoint"),
        ("not_finite", "finite"),
        ("empty", "positive integer"),
    ],
)
def test_internally_rehashed_but_inconsistent_evidence_is_rejected(evidence, damage, match):
    case = evidence
    if damage == "seed":
        case.metrics["resume_identity"]["training_arguments"]["model_seed"] = 7
    elif damage == "budget":
        case.metrics["learning_budget"]["requested_epochs"] = 5
    elif damage == "gap":
        case.rows[1]["epoch"] = 3
    elif damage == "partial":
        case.rows[1]["processed_units"] = 2
    elif damage == "updates":
        case.rows[1]["optimizer_steps"] = 1
    elif damage == "best":
        case.metrics.update(best_epoch=3, best_validation=0.6)
    elif damage == "last_history":
        case.last["history"] = copy.deepcopy(case.rows[:-1])
    elif damage == "last_steps":
        case.last["optimizer_steps"] = 9
    elif damage == "last_best":
        case.last["best_epoch"] = 1
    elif damage == "best_interior":
        case.best["epoch"] = 3
    elif damage == "optimizer":
        case.last["optimizer_state"]["state"] = {}
    elif damage == "protocol":
        case.metrics["protocol"] = {"data_sha256": "f" * 64}
    elif damage == "source":
        case.metrics["source_sha256"] = {"debug": "f" * 64}
    elif damage == "shared":
        case.last["shared_initial_state_sha256"] = "f" * 64
    elif damage == "not_finite":
        case.rows[2]["validation"] = float("nan")
    else:
        case.rows.clear()
        case.metrics["epochs_run"] = 0
    publish(case)
    with pytest.raises(ValueError, match=match):
        integrity.inspect_completed(case.folder)


def shorten(case, scores, best_epoch):
    case.rows[:] = case.rows[: len(scores)]
    for row, value in zip(case.rows, scores, strict=True):
        row["validation"] = value
    count, best_value = len(scores), scores[best_epoch - 1]
    case.metrics.update(
        epochs_run=count,
        optimizer_steps=count,
        best_epoch=best_epoch,
        best_validation=best_value,
        validation=best_value,
    )
    case.last.update(
        epoch=count, optimizer_steps=count, best_epoch=best_epoch, best_validation=best_value
    )
    case.best.update(epoch=best_epoch, validation=best_value)
    publish(case)


def test_legitimate_patience_stop_is_accepted(evidence):
    shorten(evidence, (0.7, 0.6), 1)
    assert integrity.inspect_completed(evidence.folder)["epochs_run"] == 2


def test_unjustified_early_completion_is_rejected(evidence):
    shorten(evidence, (0.4, 0.5, 0.6), 3)
    with pytest.raises(ValueError, match="budget and patience"):
        integrity.inspect_completed(evidence.folder)


def test_equal_best_scores_select_first_strict_maximum(evidence):
    shorten(evidence, (0.7, 0.7), 2)
    with pytest.raises(ValueError, match="first strict maximum"):
        integrity.inspect_completed(evidence.folder)


def test_identity_hash_tampering_is_rejected(evidence):
    evidence.metrics["resume_identity_sha256"] = "f" * 64
    (evidence.folder / "metrics.json").write_text(json.dumps(evidence.metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="identity is corrupt"):
        integrity.inspect_completed(evidence.folder)


def test_resume_boolean_is_not_a_new_scientific_identity(evidence):
    before = train.serializable_arguments(evidence.args)
    evidence.args.resume = True
    assert train.serializable_arguments(evidence.args) == before
