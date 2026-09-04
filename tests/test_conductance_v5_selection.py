"""CPU-only contracts for V5 checkpoint roles and early stopping."""

from __future__ import annotations

import pytest

from research.conductance_gat.v5 import train
from research.conductance_gat.v5.protocol import TRAINING_PHASES
from research.conductance_gat.v5.report import markdown


def test_fixed_c_primary_selection_is_valid_in_every_phase():
    for phase in TRAINING_PHASES:
        roles = train.selection_eligibility("fixed_c", phase)
        assert roles == {
            "global_prediction": True,
            "primary": True,
            "joint_early_stopping": False,
        }


def test_dynamic_primary_excludes_c_one_warmup_but_keeps_every_c_active_phase():
    assert train.selection_eligibility("shared_dynamic_c", "spatial_warmup") == {
        "global_prediction": True,
        "primary": False,
        "joint_early_stopping": False,
    }
    for phase in ("conductance_calibration", "alternating", "joint"):
        roles = train.selection_eligibility("shared_dynamic_c", phase)
        assert roles["global_prediction"] is True
        assert roles["primary"] is True
        assert roles["joint_early_stopping"] is (phase == "joint")


def test_fixed_early_stopping_monitors_primary_global_best():
    assert not train.should_stop_early(
        "fixed_c",
        "conductance_calibration",
        50,
        primary_best_epoch=1,
        joint_best_epoch=0,
        patience=50,
    )
    assert train.should_stop_early(
        "fixed_c",
        "conductance_calibration",
        51,
        primary_best_epoch=1,
        joint_best_epoch=0,
        patience=50,
    )


def test_dynamic_early_stopping_uses_separate_joint_tracker():
    assert not train.should_stop_early(
        "shared_dynamic_c",
        "alternating",
        200,
        primary_best_epoch=21,
        joint_best_epoch=0,
        patience=10,
    )
    assert not train.should_stop_early(
        "shared_dynamic_c",
        "joint",
        130,
        primary_best_epoch=21,
        joint_best_epoch=121,
        patience=10,
    )
    assert train.should_stop_early(
        "shared_dynamic_c",
        "joint",
        131,
        primary_best_epoch=21,
        joint_best_epoch=121,
        patience=10,
    )


def test_selection_helpers_reject_unknown_contract_values():
    with pytest.raises(ValueError, match="condition"):
        train.selection_eligibility("unknown", "joint")
    with pytest.raises(ValueError, match="phase"):
        train.selection_eligibility("fixed_c", "unknown")


def test_report_labels_primary_and_auxiliary_dynamic_metrics():
    report = {
        "contrasts": [
            {
                "dataset": "ogbn-arxiv",
                "metric": "accuracy",
                "fixed_c": 0.70,
                "shared_dynamic_c": 0.71,
                "dynamic_minus_fixed": 0.01,
                "shared_dynamic_c_global_prediction": 0.72,
                "dynamic_global_prediction_minus_fixed": 0.02,
                "dynamic_joint_best": 0.705,
            }
        ]
    }
    rendered = markdown(report)
    assert "Primary comparison" in rendered
    assert "Dynamic C-active" in rendered
    assert "Dynamic global (aux)" in rendered
    assert "0.710000" in rendered and "0.720000" in rendered
