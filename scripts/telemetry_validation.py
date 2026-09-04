"""Strict, stdlib-only validation for persisted runtime telemetry.

An absent measurement is valid only when the producer emits an explicit
``{"value": null, "reason": ...}`` envelope. Bare ``None`` and missing keys
fail closed.
"""

from __future__ import annotations

import math
from typing import Any

_SNAPSHOT_FIELDS = {
    "cpu": (
        "logical_count",
        "allocated_logical_count",
        "process_cpu_seconds",
        "allocated_cpu_busy_seconds",
        "allocated_cpu_total_seconds",
    ),
    "ram": (
        "process_resident_bytes",
        "process_peak_resident_bytes",
        "system_total_bytes",
        "system_available_bytes",
    ),
    "gpu": (
        "sm_utilization_percent",
        "memory_controller_utilization_percent",
        "allocator_allocated_bytes",
        "allocator_reserved_bytes",
        "device_free_bytes",
        "device_used_bytes",
        "device_total_bytes",
    ),
}
_SERIES_FIELDS = (
    "gpu_sm_utilization_percent",
    "gpu_memory_controller_utilization_percent",
    "gpu_allocator_allocated_bytes",
    "gpu_allocator_reserved_bytes",
    "gpu_device_free_bytes",
    "gpu_device_used_bytes",
    "process_cpu_seconds",
    "allocated_cpu_busy_seconds",
    "allocated_cpu_total_seconds",
    "process_resident_bytes",
    "process_peak_resident_bytes",
    "system_available_bytes",
)
_SUMMARY_FIELDS = (
    "observed_wall_seconds",
    "process_cpu_seconds",
    "average_cpu_percent_of_one_core",
    "average_cpu_percent_of_allocated_capacity",
    "average_allocated_cpu_utilization_percent",
    "cuda_allocator_peak_allocated_bytes",
    "cuda_allocator_peak_reserved_bytes",
    "run_average_gpu_sm_utilization_percent",
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _observation(value: Any, label: str) -> dict[str, Any]:
    observation = _mapping(value, label)
    if "value" not in observation or "reason" not in observation:
        raise ValueError(f"{label} must contain value and reason")
    if "unit" in observation and (
        not isinstance(observation["unit"], str) or not observation["unit"].strip()
    ):
        raise ValueError(f"{label}.unit must be a nonempty string when supplied")
    measured = observation["value"]
    reason = observation["reason"]
    if measured is None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{label} missing value requires a nonempty reason")
    else:
        _number(measured, f"{label}.value")
        if reason is not None:
            raise ValueError(f"{label} measured value must have reason=null")
    return dict(observation)


def _explicitly_unavailable(value: Any, label: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) - {"value", "reason", "unit"}:
        return None
    if value.get("value", object()) is not None:
        return None
    return _observation(value, label)


def _snapshot(value: Any, label: str) -> None:
    snapshot = _mapping(value, label)
    for name in ("wall_time_unix_seconds", "monotonic_seconds"):
        _observation(snapshot.get(name), f"{label}.{name}")
    for group_name, fields in _SNAPSHOT_FIELDS.items():
        group = _mapping(snapshot.get(group_name), f"{label}.{group_name}")
        for field in fields:
            _observation(group.get(field), f"{label}.{group_name}.{field}")


def _series(value: Any, label: str, total_samples: int) -> None:
    series = _mapping(value, label)
    measured_count = _integer(series.get("sample_count"), f"{label}.sample_count")
    unavailable_count = _integer(
        series.get("unavailable_sample_count"), f"{label}.unavailable_sample_count"
    )
    if measured_count + unavailable_count != total_samples:
        raise ValueError(f"{label} sample counts do not match total_point_sample_count")
    reasons = series.get("unavailable_reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in reasons
    ):
        raise ValueError(f"{label}.unavailable_reasons must be nonempty strings")
    bounds = {
        name: _observation(series.get(name), f"{label}.{name}")
        for name in ("minimum", "mean", "maximum")
    }
    measured = [bounds[name]["value"] for name in ("minimum", "mean", "maximum")]
    if measured_count == 0 and any(item is not None for item in measured):
        raise ValueError(f"{label} has values despite sample_count=0")
    if measured_count > 0 and any(item is None for item in measured):
        raise ValueError(f"{label} has missing bounds despite measured samples")
    if measured_count > 0 and not (
        float(measured[0]) <= float(measured[1]) <= float(measured[2])
    ):
        raise ValueError(f"{label} minimum/mean/maximum are inconsistent")


def validate_resource_observability(
    value: Any,
    label: str,
    *,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    """Validate and return one RuntimeResourceMonitor JSON payload."""

    unavailable = _explicitly_unavailable(value, label)
    if unavailable is not None:
        if not allow_unavailable:
            raise ValueError(f"{label} cannot be unavailable for an executed run")
        return unavailable
    report = _mapping(value, label)
    scope = report.get("measurement_scope")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(f"{label}.measurement_scope must be a nonempty string")
    _number(
        report.get("sample_interval_seconds"),
        f"{label}.sample_interval_seconds",
        minimum=1.0e-12,
    )
    background = _integer(
        report.get("background_sample_count"), f"{label}.background_sample_count"
    )
    total = _integer(
        report.get("total_point_sample_count"),
        f"{label}.total_point_sample_count",
        minimum=2,
    )
    if total != background + 2:
        raise ValueError(f"{label} total samples must equal start + background + end")
    errors = report.get("sampler_errors")
    if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
        raise ValueError(f"{label}.sampler_errors must be a string list")
    _snapshot(report.get("start"), f"{label}.start")
    _snapshot(report.get("end"), f"{label}.end")
    interval_series = _mapping(report.get("interval_series"), f"{label}.interval_series")
    for field in _SERIES_FIELDS:
        _series(interval_series.get(field), f"{label}.interval_series.{field}", total)
    summary = _mapping(report.get("summary"), f"{label}.summary")
    for field in _SUMMARY_FIELDS:
        _observation(summary.get(field), f"{label}.summary.{field}")
    return dict(report)


def _walk_throughput(value: Any, label: str, rate_values: list[float | None]) -> None:
    if isinstance(value, dict):
        if "value" in value or "reason" in value:
            observation = _observation(value, label)
            if label.endswith("_per_second"):
                measured = observation["value"]
                rate_values.append(None if measured is None else float(measured))
            return
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} contains a non-string/empty key")
            _walk_throughput(nested, f"{label}.{key}", rate_values)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        measured = _number(value, label)
        if label.endswith("_per_second"):
            rate_values.append(measured)
        return
    if isinstance(value, str) and value.strip():
        return
    raise ValueError(f"{label} contains an unsupported or missing value")


def validate_throughput_observability(
    value: Any,
    label: str,
    *,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    """Validate and return measured throughput with at least one explicit rate."""

    unavailable = _explicitly_unavailable(value, label)
    if unavailable is not None:
        if not allow_unavailable:
            raise ValueError(f"{label} cannot be unavailable for an executed run")
        return unavailable
    report = _mapping(value, label)
    scope = report.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(f"{label}.scope must be a nonempty string")
    rates: list[float | None] = []
    for key, nested in report.items():
        if key != "scope":
            _walk_throughput(nested, f"{label}.{key}", rates)
    if not rates:
        raise ValueError(f"{label} must contain at least one *_per_second rate")
    return dict(report)


__all__ = ["validate_resource_observability", "validate_throughput_observability"]
