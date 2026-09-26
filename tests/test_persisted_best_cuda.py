"""Actual CUDA training with deliberate saved-weight corruption; synthetic debug only."""

import copy
import json
import weakref

import pytest
import torch

from experiments.sampled_inductive import runner, train
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401
from tests.test_sampled_inductive import synthetic_payload


def debug_case():
    payload = synthetic_payload()
    for row in payload["graphs"]:
        row["y"].fill_(1)
    options = runner.parser().parse_args(
        [
            "--run-id",
            "debug-persisted-best",
            "--context-seeds",
            "16",
            "--context-batches",
            "4",
            "8",
            "--epochs",
            "4",
            "--num-neighbors",
            "2",
            "--edge-chunk-size",
            "128",
        ]
    )
    args = runner.child_arguments(options, "incidence_fixed", 0, batch=4, workers=0)
    return payload, args


def test_corrupt_serialized_best_weights_cannot_pass(monkeypatch, tmp_path):
    payload, args = debug_case()
    original = train.engine.base._save
    selected = []

    def faulty_save(path, values):
        changed = copy.copy(values)
        changed["best_state"] = {k: v.detach().clone() for k, v in values["best_state"].items()}
        changed["best_state"]["decoder.weight"].zero_()
        changed["best_state"]["decoder.bias"].fill_(-50)
        selected.append(values["best_validation"]["metric"])
        original(path, changed)

    monkeypatch.setattr(train.engine.base, "_save", faulty_save)
    with pytest.raises(ValueError, match="persisted|reproduc|counts|selected"):
        train.train_cell(
            payload,
            {"explicit_synthetic_debug": True},
            args,
            "full",
            4,
            0,
            torch.device("cuda:0"),
            tmp_path / "corrupt",
        )
    assert max(selected) > 0
    assert not (tmp_path / "corrupt" / "metrics.json").exists()


def test_normal_disk_audit_uses_fresh_model_and_rejects_changed_evidence(monkeypatch, tmp_path):
    payload, args = debug_case()
    factory = train.engine.make_model
    references = []

    def tracked_factory(*values):
        if references:
            assert references[0]() is None, "training model must be released before reload"
        model = factory(*values)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(train.engine, "make_model", tracked_factory)
    folder = tmp_path / "normal"
    result = train.train_cell(
        payload,
        {"explicit_synthetic_debug": True},
        args,
        "full",
        4,
        0,
        torch.device("cuda:0"),
        folder,
    )
    assert len(references) == 2
    assert train.completed(folder) == result
    assert result["persisted_best_audit"]["checkpoint_sha256"] == result["last_sha256"]
    original = (folder / "metrics.json").read_text(encoding="utf-8")
    for change, message in (
        ("missing", "persisted-best"),
        ("hash", "persisted-best"),
        ("graph_ids", "different held-out graphs"),
    ):
        altered = json.loads(original)
        if change == "missing":
            altered.pop("persisted_best_audit")
        elif change == "hash":
            altered["persisted_best_audit"]["best_state_sha256"] = "changed"
        else:
            altered["audit"][0]["graph_ids"] = [5]
        (folder / "metrics.json").write_text(json.dumps(altered), encoding="utf-8")
        try:
            with pytest.raises(ValueError, match=message):
                train.completed(folder)
        finally:
            (folder / "metrics.json").write_text(original, encoding="utf-8")


def test_recorded_optimizer_and_audit_policy_match_actual_runner():
    payload, args = debug_case()
    model = train.engine.make_model(payload, args, torch.device("cuda:0"))
    optimizer = train.engine.make_optimizer(model, args.learning_rate)
    config = train.configuration(args, "sampled")
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert config["lr"] == group["lr"]
    assert config["weight_decay"] == group["weight_decay"]
    assert config["conductance_weight_decay"] == group["weight_decay"]
    assert config["scalar_weight_decay"] == group["weight_decay"]
    assert config["comparison_contract"]["mechanism_audit"] is None
    assert "last.pt/best_state" in config["comparison_contract"]["validation_audit"]
    assert config["sampling"] == "inductive_cluster_contexts"
