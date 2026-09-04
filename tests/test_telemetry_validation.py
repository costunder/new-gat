from __future__ import annotations

import copy

import pytest
import torch

from chartgat.observability import finalize_resource_observability, runtime_resource_snapshot
from scripts.telemetry_validation import (
    validate_resource_observability,
    validate_throughput_observability,
)


def _resource_report() -> dict:
    device = torch.device("cpu")
    return finalize_resource_observability(
        runtime_resource_snapshot(device),
        device,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        sample_interval_seconds=1.0,
    )


def test_resource_schema_accepts_explicit_missing_counters_and_rejects_missing_fields() -> None:
    report = _resource_report()
    assert validate_resource_observability(report, "unit") == report

    incomplete = copy.deepcopy(report)
    del incomplete["summary"]["run_average_gpu_sm_utilization_percent"]
    with pytest.raises(ValueError, match="run_average_gpu_sm_utilization_percent"):
        validate_resource_observability(incomplete, "unit")

    dishonest = copy.deepcopy(report)
    dishonest["summary"]["run_average_gpu_sm_utilization_percent"] = {
        "value": None,
        "reason": None,
        "unit": "percent",
    }
    with pytest.raises(ValueError, match="requires a nonempty reason"):
        validate_resource_observability(dishonest, "unit")


def test_executed_throughput_requires_scope_and_an_explicit_rate() -> None:
    throughput = {
        "scope": "unit measured interval",
        "processed_items": 12,
        "processed_items_per_second": 6.0,
    }
    assert validate_throughput_observability(throughput, "unit") == throughput
    with pytest.raises(ValueError, match="scope"):
        validate_throughput_observability(
            {"processed_items_per_second": 6.0}, "unit"
        )
    with pytest.raises(ValueError, match=r"\*_per_second"):
        validate_throughput_observability(
            {"scope": "no measured rate", "processed_items": 12}, "unit"
        )


def test_whole_payload_unavailability_is_allowed_only_when_explicitly_authorized() -> None:
    unavailable = {"value": None, "reason": "prepare-only performs no training"}
    with pytest.raises(ValueError, match="executed run"):
        validate_throughput_observability(unavailable, "unit")
    assert (
        validate_throughput_observability(unavailable, "unit", allow_unavailable=True)
        == unavailable
    )
