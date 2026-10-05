"""Complete frozen diagnostic on already-completed DEBUG artifacts, no training."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.frozen_energy_trace.study import execute

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/wedge-classification-debug-20261004-01"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires actual CUDA; no CPU substitute"
)


def test_nine_existing_models_one_seed_complete_without_optimizer(tmp_path, monkeypatch):
    if not SOURCE.is_dir():
        pytest.skip("completed DEBUG artifacts required; never train a replacement")

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("frozen diagnostic must not create an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden_optimizer)
    monkeypatch.setattr(torch.optim, "SGD", forbidden_optimizer)
    output = execute(
        SimpleNamespace(
            classification_run=SOURCE,
            results_root=ROOT / "results",
            output_dir=tmp_path / "diagnostic",
            seed=11,
            profile="debug",
            device="cuda:0",
            workers="auto",
            feature_chunks=[8, 32],
            calibration_repeats=2,
        )
    )
    done = json.loads((output / "completion.json").read_text())
    assert done["completed"] and done["profile"] == "debug"
    assert done["checkpoint_analyses"] == 9
    assert done["stage_rows"] == done["transition_rows"] == 99
    assert done["seed_count"] == 1 and done["seed"] == 11
    assert done["new_training_runs"] == done["optimizer_updates"] == 0
    rows = json.loads((output / "transitions.json").read_text())["rows"]
    for row in rows:
        if row["model"] == "mlp" and row["transition"] == "aggregation":
            assert row["E_operation_ratio"]["median"] == 1
            assert row["J_distinct_delta"]["l2"] < 1e-12
    stages = json.loads((output / "stages.json").read_text())["rows"]
    assert not any(row["stage"] == "layer_1_activated" for row in stages)
    for dataset in ("DEBUG-Cora", "DEBUG-CiteSeer", "DEBUG-PubMed"):
        for layer in (0, 1):
            a = np.load(output / f"{dataset}-standard_gcn-layer_{layer}_aggregated.npz")
            p = np.load(output / f"{dataset}-standard_gcn-layer_{layer}_replay_P.npz")
            np.testing.assert_allclose(a["E"], p["E"], rtol=1e-5, atol=1e-10)
        provenance = json.loads((output / f"{dataset}-standard_gcn-provenance.json").read_text())
        assert provenance["stage_count"] == 7 and provenance["replay_stage_count"] == 4
        assert provenance["model_updates"] == provenance["trainable_parameter_count"] == 0
        assert provenance["logits_replay"]["within_tolerance"]


def test_debug_checkpoint_rejected_as_full(tmp_path):
    if not SOURCE.is_dir():
        pytest.skip("completed DEBUG artifacts required")
    with pytest.raises(ValueError, match="configuration"):
        execute(
            SimpleNamespace(
                classification_run=SOURCE,
                results_root=ROOT / "results",
                output_dir=tmp_path / "full",
                seed=11,
                profile="full",
                device="cuda:0",
                workers="auto",
                feature_chunks=[8],
                calibration_repeats=2,
            )
        )
    assert not (tmp_path / "full").exists()
