"""CPU-only V5 telemetry contract regressions; no training/performance claims."""

from __future__ import annotations

import json
import math

import pytest

from research.conductance_gat.v5.train import merge_efficiency, training_throughput
from scripts.telemetry_validation import validate_throughput_observability


def test_cumulative_history_and_elapsed_use_the_same_resume_boundary():
    # Synthetic unit counters deliberately have different per-segment rates.
    restored_history = [{"train_label_count": 100, "train_batches": 2}]
    current_history = [{"train_label_count": 300, "train_batches": 6}]
    efficiency = merge_efficiency(12.5, 1024, 2048, 7.5, 2048, 4096)
    report = training_throughput(
        restored_history + current_history, efficiency["elapsed_seconds"]
    )
    persisted = json.loads(json.dumps(report, allow_nan=False))
    assert validate_throughput_observability(persisted, "v5.throughput") == report
    assert report["completed_epochs"] == 2
    assert report["supervised_training_labels"] == 400
    assert report["training_batches"] == 8
    assert report["elapsed_seconds"] == 20.0
    assert report["supervised_labels_per_second"] == {
        "value": 20.0, "reason": None, "unit": "labels_per_second"
    }
    assert report["training_batches_per_second"] == {
        "value": 0.4, "reason": None, "unit": "batches_per_second"
    }
    assert "validation" in report["scope"]
    assert "checkpoint IO" in report["scope"]
    assert "intervention" in report["scope"]
    assert "setup" in report["timer_boundary"]
    assert "interrupted work" in report["resume_accounting"]


def test_zero_timer_is_explicitly_unavailable_not_zero_rate_or_bare_null():
    report = training_throughput([{"train_label_count": 100, "train_batches": 2}], 0.0)
    assert validate_throughput_observability(report, "v5.throughput") == report
    for key in ("supervised_labels_per_second", "training_batches_per_second"):
        assert report[key]["value"] is None
        assert "duration was zero" in report[key]["reason"]
        assert report[key]["unit"].endswith("_per_second")


@pytest.mark.parametrize("elapsed", [True, False, None, "1", -1, math.nan, math.inf])
def test_invalid_elapsed_is_not_published(elapsed):
    with pytest.raises(ValueError, match="elapsed_seconds"):
        training_throughput([{"train_label_count": 100, "train_batches": 2}], elapsed)


@pytest.mark.parametrize(
    "history",
    [
        None,
        [],
        [{}],
        [None],
        [{"train_label_count": True, "train_batches": 2}],
        [{"train_label_count": 100, "train_batches": False}],
        [{"train_label_count": 100.0, "train_batches": 2}],
        [{"train_label_count": -1, "train_batches": 2}],
        [{"train_label_count": 100, "train_batches": 0}],
    ],
)
def test_missing_or_invalid_history_cannot_fabricate_processed_work(history):
    with pytest.raises(ValueError, match="history"):
        training_throughput(history, 20.0)
