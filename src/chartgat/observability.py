"""Fail-visible runtime resource observations for research training runs.

Optional counters use value=None plus the exact reason; missing measurements
are never replaced by zero or an invented estimate.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def observed(
    value: Any, *, reason: str | None = None, unit: str | None = None
) -> dict[str, Any]:
    """Return one JSON-safe observation with an explicit missing-value reason."""
    if value is None and not reason:
        raise ValueError("a missing observation requires a reason")
    result = {"value": value, "reason": reason}
    if unit is not None:
        result["unit"] = unit
    return result


def _proc_kib(path: Path, field: str) -> dict[str, Any]:
    if not path.is_file():
        return observed(None, reason=f"{path} is unavailable", unit="bytes")
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name == field:
                pieces = remainder.split()
                if len(pieces) != 2 or pieces[1].lower() != "kb":
                    return observed(
                        None,
                        reason=f"{path}:{field} did not contain a kB value",
                        unit="bytes",
                    )
                return observed(int(pieces[0]) * 1024, unit="bytes")
    except (OSError, UnicodeError, ValueError) as exc:
        return observed(
            None,
            reason=f"{type(exc).__name__} while reading {path}:{field}: {exc}",
            unit="bytes",
        )
    return observed(None, reason=f"{path}:{field} is unavailable", unit="bytes")


def _allocated_cpu_time_counters(
    cpu_ids: set[int] | None,
    *,
    path: Path = Path("/proc/stat"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read cumulative busy/total time for the CPUs in this process' affinity mask.

    Unlike ``time.process_time()``, these counters include DataLoader workers and
    every other task scheduled on the same CPUs.  They are therefore deliberately
    labelled as allocated-CPU, rather than process-attributed, observations.
    """

    unit = "seconds"
    if cpu_ids is None:
        reason = "CPU affinity is unavailable; allocated-CPU counters cannot be selected"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        )
    if not path.is_file():
        reason = f"{path} is unavailable"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        )
    try:
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        if ticks_per_second <= 0:
            raise ValueError("SC_CLK_TCK was not positive")
        counters: dict[int, tuple[int, int]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            pieces = line.split()
            if not pieces or re.fullmatch(r"cpu\d+", pieces[0]) is None:
                continue
            cpu_index = int(pieces[0][3:])
            values = [int(value) for value in pieces[1:9]]
            if len(values) < 4:
                raise ValueError(f"{path}:{pieces[0]} has fewer than four CPU counters")
            total_ticks = sum(values)
            idle_ticks = values[3] + (values[4] if len(values) > 4 else 0)
            counters[cpu_index] = (total_ticks - idle_ticks, total_ticks)
        missing = sorted(cpu_ids - counters.keys())
        if missing:
            raise ValueError(f"{path} omitted allocated CPU ids {missing}")
        busy_ticks = sum(counters[index][0] for index in cpu_ids)
        total_ticks = sum(counters[index][1] for index in cpu_ids)
        return observed(busy_ticks / ticks_per_second, unit=unit), observed(
            total_ticks / ticks_per_second, unit=unit
        )
    except (OSError, UnicodeError, ValueError) as exc:
        reason = f"{type(exc).__name__} while reading allocated CPU counters: {exc}"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        )


def _optional_cuda_counter(
    name: str,
    device: torch.device,
    *,
    unit: str,
    transform: Callable[[Any], Any] = int,
) -> dict[str, Any]:
    function = getattr(torch.cuda, name, None)
    if function is None:
        return observed(None, reason=f"torch.cuda.{name} is unavailable", unit=unit)
    try:
        return observed(transform(function(device)), unit=unit)
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        return observed(
            None,
            reason=f"torch.cuda.{name} failed with {type(exc).__name__}: {exc}",
            unit=unit,
        )


def _safe_nvidia_identifier(value: Any) -> str | None:
    """Normalize a CUDA/NVML identifier without treating it as shell input."""

    if isinstance(value, bytes):
        try:
            value = value.decode("ascii", errors="strict")
        except UnicodeError:
            if len(value) == 16:
                hexadecimal = value.hex()
                value = "GPU-" + "-".join(
                    (
                        hexadecimal[:8],
                        hexadecimal[8:12],
                        hexadecimal[12:16],
                        hexadecimal[16:20],
                        hexadecimal[20:],
                    )
                )
            else:
                return None
    if not isinstance(value, str):
        return None
    identifier = value.strip()
    # Legacy MIG identifiers use slash-separated GPU-instance/compute-instance
    # suffixes (MIG-GPU-.../GI/CI).  subprocess receives one argv element, so
    # accepting '/' here does not introduce shell interpretation.
    if not identifier or re.fullmatch(r"[A-Za-z0-9_.:/-]+", identifier) is None:
        return None
    return identifier


def _nvidia_smi_device_identifier(device: torch.device) -> tuple[str | None, str | None]:
    """Map one logical CUDA device to an identifier accepted by ``nvidia-smi``."""

    try:
        logical_index = (
            int(device.index) if device.index is not None else int(torch.cuda.current_device())
        )
    except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"could not resolve the logical CUDA index: {type(exc).__name__}: {exc}"

    # A runtime UUID is the only identity that remains unambiguous when CUDA
    # logical devices have been reordered by CUDA_VISIBLE_DEVICES or
    # CUDA_DEVICE_ORDER.  Prefer it over parsing the environment.
    property_error: str | None = None
    try:
        properties = torch.cuda.get_device_properties(device)
        identifier = _safe_nvidia_identifier(getattr(properties, "uuid", None))
        if identifier is not None:
            return identifier, None
        property_error = "torch CUDA device properties did not expose a usable UUID"
    except (AssertionError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
        property_error = (
            "CUDA device UUID lookup failed with "
            f"{type(exc).__name__}: {exc}"
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",")]
        if logical_index >= len(tokens):
            return None, (
                f"logical CUDA index {logical_index} is outside CUDA_VISIBLE_DEVICES={visible!r}"
            )
        identifier = _safe_nvidia_identifier(tokens[logical_index])
        if identifier == "-1" or not identifier:
            return None, f"CUDA_VISIBLE_DEVICES={visible!r} exposes no usable identifier"
        if identifier is None:
            return None, "CUDA_VISIBLE_DEVICES contains an unsafe nvidia-smi identifier"
        if identifier.isdecimal():
            return None, (
                f"{property_error}; CUDA_VISIBLE_DEVICES={visible!r} uses numeric ordinals, "
                "which cannot be mapped safely to nvidia-smi indices when CUDA device "
                "ordering may differ"
            )
        return identifier, None
    return None, (
        "CUDA_VISIBLE_DEVICES is unset and the logical CUDA device could not be "
        f"mapped safely to nvidia-smi: {property_error}"
    )


def _nvidia_smi_utilization(
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Query device-wide utilization without requiring the optional pynvml package."""

    unit = "percent"
    executable = shutil.which("nvidia-smi")
    if executable is None:
        reason = "nvidia-smi is not available on PATH"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    identifier, identifier_error = _nvidia_smi_device_identifier(device)
    if identifier is None:
        reason = f"nvidia-smi device mapping failed: {identifier_error}"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    command = [
        executable,
        f"--id={identifier}",
        "--query-gpu=utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        reason = f"nvidia-smi query failed with {type(exc).__name__}: {exc}"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or "no diagnostic output"
        reason = f"nvidia-smi returned {completed.returncode}: {detail}"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        reason = f"nvidia-smi returned {len(rows)} utilization rows; exactly one was required"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    pieces = [piece.strip() for piece in rows[0].split(",")]
    if len(pieces) != 2:
        reason = "nvidia-smi utilization row did not contain exactly two columns"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    try:
        values = [float(piece) for piece in pieces]
    except ValueError:
        reason = f"nvidia-smi returned nonnumeric utilization values: {pieces!r}"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    if any(not 0.0 <= value <= 100.0 for value in values):
        reason = f"nvidia-smi returned utilization outside [0, 100]: {values!r}"
        return observed(None, reason=reason, unit=unit), observed(
            None, reason=reason, unit=unit
        ), "unavailable"
    return (
        observed(values[0], unit=unit),
        observed(values[1], unit=unit),
        "nvidia-smi",
    )


def _cuda_utilization_observations(
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Prefer PyTorch's NVML API and use nvidia-smi when pynvml is unavailable."""

    sm = _optional_cuda_counter("utilization", device, unit="percent")
    memory = _optional_cuda_counter("memory_usage", device, unit="percent")
    sources = {
        "sm_utilization_percent": "torch.cuda.utilization (NVML)",
        "memory_controller_utilization_percent": "torch.cuda.memory_usage (NVML)",
    }
    if sm["value"] is not None and memory["value"] is not None:
        return sm, memory, sources

    fallback_sm, fallback_memory, fallback_source = _nvidia_smi_utilization(device)
    if sm["value"] is None:
        if fallback_sm["value"] is not None:
            sm = fallback_sm
            sources["sm_utilization_percent"] = fallback_source
        else:
            sm = observed(
                None,
                reason=f"{sm['reason']}; fallback: {fallback_sm['reason']}",
                unit="percent",
            )
            sources["sm_utilization_percent"] = "unavailable"
    if memory["value"] is None:
        if fallback_memory["value"] is not None:
            memory = fallback_memory
            sources["memory_controller_utilization_percent"] = fallback_source
        else:
            memory = observed(
                None,
                reason=f"{memory['reason']}; fallback: {fallback_memory['reason']}",
                unit="percent",
            )
            sources["memory_controller_utilization_percent"] = "unavailable"
    return sm, memory, sources


def runtime_resource_snapshot(device: torch.device) -> dict[str, Any]:
    """Capture one point-in-time host and CUDA resource observation."""
    logical_count = os.cpu_count()
    affinity = getattr(os, "sched_getaffinity", None)
    allocated_cpu_ids: set[int] | None = None
    if affinity is None:
        allocated_count = observed(
            None,
            reason="os.sched_getaffinity is unavailable on this platform",
            unit="logical_cpus",
        )
    else:
        try:
            allocated_cpu_ids = {int(index) for index in affinity(0)}
            allocated_count = observed(
                len(allocated_cpu_ids), unit="logical_cpus"
            )
        except (OSError, TypeError, ValueError) as exc:
            allocated_count = observed(
                None,
                reason=(
                    "os.sched_getaffinity failed with "
                    f"{type(exc).__name__}: {exc}"
                ),
                unit="logical_cpus",
            )
    allocated_busy, allocated_total = _allocated_cpu_time_counters(
        allocated_cpu_ids
    )

    snapshot: dict[str, Any] = {
        "wall_time_unix_seconds": observed(time.time(), unit="seconds"),
        "monotonic_seconds": observed(time.perf_counter(), unit="seconds"),
        "cpu": {
            "logical_count": observed(
                logical_count,
                reason=None if logical_count is not None else "os.cpu_count returned None",
                unit="logical_cpus",
            ),
            "allocated_logical_count": allocated_count,
            "process_cpu_seconds": observed(time.process_time(), unit="seconds"),
            "allocated_cpu_busy_seconds": allocated_busy,
            "allocated_cpu_total_seconds": allocated_total,
        },
        "ram": {
            "process_resident_bytes": _proc_kib(Path("/proc/self/status"), "VmRSS"),
            "process_peak_resident_bytes": _proc_kib(Path("/proc/self/status"), "VmHWM"),
            "system_total_bytes": _proc_kib(Path("/proc/meminfo"), "MemTotal"),
            "system_available_bytes": _proc_kib(Path("/proc/meminfo"), "MemAvailable"),
        },
    }
    if device.type != "cuda":
        reason = f"requested device is {device.type}, not CUDA"
        snapshot["gpu"] = {
            "sm_utilization_percent": observed(None, reason=reason, unit="percent"),
            "memory_controller_utilization_percent": observed(
                None, reason=reason, unit="percent"
            ),
            "allocator_allocated_bytes": observed(None, reason=reason, unit="bytes"),
            "allocator_reserved_bytes": observed(None, reason=reason, unit="bytes"),
            "device_free_bytes": observed(None, reason=reason, unit="bytes"),
            "device_used_bytes": observed(None, reason=reason, unit="bytes"),
            "device_total_bytes": observed(None, reason=reason, unit="bytes"),
        }
        return snapshot

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_bytes, total_bytes = int(free_bytes), int(total_bytes)
        if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
            raise ValueError(
                f"invalid CUDA memory counters free={free_bytes}, total={total_bytes}"
            )
        memory_info = {
            "device_free_bytes": observed(free_bytes, unit="bytes"),
            "device_used_bytes": observed(total_bytes - free_bytes, unit="bytes"),
            "device_total_bytes": observed(total_bytes, unit="bytes"),
        }
    except (OSError, RuntimeError, ValueError) as exc:
        reason = f"torch.cuda.mem_get_info failed with {type(exc).__name__}: {exc}"
        memory_info = {
            "device_free_bytes": observed(None, reason=reason, unit="bytes"),
            "device_used_bytes": observed(None, reason=reason, unit="bytes"),
            "device_total_bytes": observed(None, reason=reason, unit="bytes"),
        }
    sm_utilization, memory_utilization, utilization_sources = (
        _cuda_utilization_observations(device)
    )
    snapshot["gpu"] = {
        "sm_utilization_percent": sm_utilization,
        "memory_controller_utilization_percent": memory_utilization,
        "utilization_sources": utilization_sources,
        "allocator_allocated_bytes": _optional_cuda_counter(
            "memory_allocated", device, unit="bytes"
        ),
        "allocator_reserved_bytes": _optional_cuda_counter(
            "memory_reserved", device, unit="bytes"
        ),
        **memory_info,
    }
    return snapshot


def _sample_values(
    samples: list[dict[str, Any]], path: tuple[str, ...]
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    reasons: list[str] = []
    for sample in samples:
        current: Any = sample
        for key in path:
            current = current[key]
        value, reason = current["value"], current["reason"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        elif isinstance(reason, str) and reason not in reasons:
            reasons.append(reason)
    return values, reasons


def _series_observability(
    samples: list[dict[str, Any]],
    path: tuple[str, ...],
    *,
    unit: str,
) -> dict[str, Any]:
    values, reasons = _sample_values(samples, path)
    if values:
        return {
            "sample_count": len(values),
            "minimum": observed(min(values), unit=unit),
            "mean": observed(sum(values) / len(values), unit=unit),
            "maximum": observed(max(values), unit=unit),
            "unavailable_sample_count": len(samples) - len(values),
            "unavailable_reasons": reasons,
        }
    reason = (
        "; ".join(reasons)
        if reasons
        else "the runtime sampler collected no usable observations"
    )
    return {
        "sample_count": 0,
        "minimum": observed(None, reason=reason, unit=unit),
        "mean": observed(None, reason=reason, unit=unit),
        "maximum": observed(None, reason=reason, unit=unit),
        "unavailable_sample_count": len(samples),
        "unavailable_reasons": reasons,
    }


def _counter_utilization_observation(
    start: dict[str, Any],
    end: dict[str, Any],
    *,
    busy_field: str,
    total_field: str,
) -> dict[str, Any]:
    """Convert two cumulative CPU counter pairs into an interval percentage."""

    start_busy = start["cpu"][busy_field]
    end_busy = end["cpu"][busy_field]
    start_total = start["cpu"][total_field]
    end_total = end["cpu"][total_field]
    observations = (start_busy, end_busy, start_total, end_total)
    missing_reasons = [
        str(item["reason"])
        for item in observations
        if item["value"] is None and item["reason"] is not None
    ]
    if missing_reasons:
        return observed(
            None,
            reason="; ".join(dict.fromkeys(missing_reasons)),
            unit="percent",
        )
    busy_delta = float(end_busy["value"]) - float(start_busy["value"])
    total_delta = float(end_total["value"]) - float(start_total["value"])
    if total_delta <= 0:
        return observed(
            None,
            reason="allocated CPU total-time counter did not increase",
            unit="percent",
        )
    if busy_delta < 0 or busy_delta > total_delta:
        return observed(
            None,
            reason=(
                "allocated CPU busy-time delta was outside the observed total-time delta"
            ),
            unit="percent",
        )
    return observed(100.0 * busy_delta / total_delta, unit="percent")


def finalize_resource_observability(
    start: dict[str, Any],
    device: torch.device,
    *,
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
    interval_samples: list[dict[str, Any]] | None = None,
    sampler_errors: list[str] | None = None,
    sample_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Combine start/end observations and measured run-boundary summaries."""
    end = runtime_resource_snapshot(device)
    samples = [start, *(interval_samples or []), end]
    start_wall = start["monotonic_seconds"]["value"]
    end_wall = end["monotonic_seconds"]["value"]
    start_cpu = start["cpu"]["process_cpu_seconds"]["value"]
    end_cpu = end["cpu"]["process_cpu_seconds"]["value"]
    elapsed = max(float(end_wall) - float(start_wall), 0.0)
    cpu_seconds = max(float(end_cpu) - float(start_cpu), 0.0)
    allocated_cpus = end["cpu"]["allocated_logical_count"]["value"]
    if elapsed > 0:
        one_core_percent = 100.0 * cpu_seconds / elapsed
        if isinstance(allocated_cpus, int) and allocated_cpus > 0:
            capacity_percent = one_core_percent / allocated_cpus
            capacity_reason = None
        else:
            capacity_percent = None
            capacity_reason = "allocated logical CPU count was unavailable"
        one_core_reason = None
    else:
        one_core_percent = None
        capacity_percent = None
        one_core_reason = capacity_reason = "observed wall duration was zero"
    gpu_sm = _series_observability(
        samples, ("gpu", "sm_utilization_percent"), unit="percent"
    )
    allocated_cpu_utilization = _counter_utilization_observation(
        start,
        end,
        busy_field="allocated_cpu_busy_seconds",
        total_field="allocated_cpu_total_seconds",
    )
    if device.type != "cuda":
        peak_reason = f"requested device is {device.type}, not CUDA"
        peak_allocated = observed(None, reason=peak_reason, unit="bytes")
        peak_reserved = observed(None, reason=peak_reason, unit="bytes")
    else:
        peak_allocated = (
            observed(int(peak_allocated_bytes), unit="bytes")
            if peak_allocated_bytes is not None
            else observed(
                None,
                reason="caller did not supply a CUDA allocator allocated-memory peak",
                unit="bytes",
            )
        )
        peak_reserved = (
            observed(int(peak_reserved_bytes), unit="bytes")
            if peak_reserved_bytes is not None
            else observed(
                None,
                reason="caller did not supply a CUDA allocator reserved-memory peak",
                unit="bytes",
            )
        )
    result = {
        "measurement_scope": (
            "periodic point samples for coordinator-process CPU time/RSS, host RAM, "
            "allocated-CPU busy/total counters, device-wide VRAM and device-wide, not "
            "process-attributed, NVML-backed GPU utilization; coordinator process CPU "
            "excludes DataLoader workers, while allocated-CPU utilization includes workers "
            "and any unrelated tasks scheduled on the same affinity CPUs; CUDA allocator "
            "peaks cover the caller-defined monitored workload boundary"
        ),
        "sample_interval_seconds": sample_interval_seconds,
        "background_sample_count": len(interval_samples or []),
        "total_point_sample_count": len(samples),
        "sampler_errors": list(sampler_errors or []),
        "start": start,
        "end": end,
        "interval_series": {
            "gpu_sm_utilization_percent": gpu_sm,
            "gpu_memory_controller_utilization_percent": _series_observability(
                samples,
                ("gpu", "memory_controller_utilization_percent"),
                unit="percent",
            ),
            "gpu_allocator_allocated_bytes": _series_observability(
                samples, ("gpu", "allocator_allocated_bytes"), unit="bytes"
            ),
            "gpu_allocator_reserved_bytes": _series_observability(
                samples, ("gpu", "allocator_reserved_bytes"), unit="bytes"
            ),
            "gpu_device_free_bytes": _series_observability(
                samples, ("gpu", "device_free_bytes"), unit="bytes"
            ),
            "gpu_device_used_bytes": _series_observability(
                samples, ("gpu", "device_used_bytes"), unit="bytes"
            ),
            "process_cpu_seconds": _series_observability(
                samples, ("cpu", "process_cpu_seconds"), unit="seconds"
            ),
            "allocated_cpu_busy_seconds": _series_observability(
                samples, ("cpu", "allocated_cpu_busy_seconds"), unit="seconds"
            ),
            "allocated_cpu_total_seconds": _series_observability(
                samples, ("cpu", "allocated_cpu_total_seconds"), unit="seconds"
            ),
            "process_resident_bytes": _series_observability(
                samples, ("ram", "process_resident_bytes"), unit="bytes"
            ),
            "process_peak_resident_bytes": _series_observability(
                samples, ("ram", "process_peak_resident_bytes"), unit="bytes"
            ),
            "system_available_bytes": _series_observability(
                samples, ("ram", "system_available_bytes"), unit="bytes"
            ),
        },
        "summary": {
            "observed_wall_seconds": observed(elapsed, unit="seconds"),
            "process_cpu_seconds": observed(cpu_seconds, unit="seconds"),
            "average_cpu_percent_of_one_core": observed(
                one_core_percent, reason=one_core_reason, unit="percent"
            ),
            "average_cpu_percent_of_allocated_capacity": observed(
                capacity_percent, reason=capacity_reason, unit="percent"
            ),
            "average_allocated_cpu_utilization_percent": allocated_cpu_utilization,
            "cuda_allocator_peak_allocated_bytes": peak_allocated,
            "cuda_allocator_peak_reserved_bytes": peak_reserved,
            "run_average_gpu_sm_utilization_percent": observed(
                gpu_sm["mean"]["value"],
                reason=gpu_sm["mean"]["reason"],
                unit="percent",
            ),
        },
    }
    return result


class RuntimeResourceMonitor:
    """Periodically sample process, RAM, CUDA allocator and NVML utilization counters."""

    def __init__(self, device: torch.device, *, sample_interval_seconds: float = 1.0) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.device = device
        self.sample_interval_seconds = float(sample_interval_seconds)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._thread: threading.Thread | None = None
        self.start_snapshot: dict[str, Any] | None = None

    def _record(self) -> None:
        try:
            sample = runtime_resource_snapshot(self.device)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                self._errors.append(
                    f"runtime_resource_snapshot failed with {type(exc).__name__}: {exc}"
                )
            return
        with self._lock:
            self._samples.append(sample)

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._record()

    def start(self) -> dict[str, Any]:
        if self._thread is not None:
            raise RuntimeError("resource monitor was already started")
        self.start_snapshot = runtime_resource_snapshot(self.device)
        self._thread = threading.Thread(
            target=self._run,
            name="chartgat-resource-observer",
            daemon=True,
        )
        try:
            self._thread.start()
        except BaseException as primary_error:
            # A Ctrl-C can arrive immediately after Thread.start has created the
            # sampler.  Signal and join it before preserving the interruption.
            self._stop.set()
            if self._thread.is_alive():
                try:
                    self._thread.join(
                        timeout=max(5.0, 2.0 * self.sample_interval_seconds)
                    )
                except BaseException as cleanup_error:
                    primary_error.add_note(
                        "resource sampler cleanup after start failure also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
        return self.start_snapshot

    def finish(
        self,
        *,
        peak_allocated_bytes: int | None,
        peak_reserved_bytes: int | None,
    ) -> dict[str, Any]:
        if self._thread is None or self.start_snapshot is None:
            raise RuntimeError("resource monitor was not started")
        self._stop.set()
        join_timeout = max(5.0, 2.0 * self.sample_interval_seconds)
        try:
            self._thread.join(timeout=join_timeout)
        except BaseException as primary_error:
            # Preserve Ctrl-C/other primary failures, but make one final bounded
            # cleanup attempt after the stop event is already visible.
            try:
                self._thread.join(timeout=join_timeout)
            except BaseException as cleanup_error:
                primary_error.add_note(
                    "resource sampler join retry also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            if self._thread.is_alive():
                primary_error.add_note(
                    "resource sampler thread remained alive after its join timeout"
                )
            raise
        if self._thread.is_alive():
            raise RuntimeError(
                "resource sampler thread did not stop before its join timeout"
            )
        with self._lock:
            samples = list(self._samples)
            errors = list(self._errors)
        return finalize_resource_observability(
            self.start_snapshot,
            self.device,
            peak_allocated_bytes=peak_allocated_bytes,
            peak_reserved_bytes=peak_reserved_bytes,
            interval_samples=samples,
            sampler_errors=errors,
            sample_interval_seconds=self.sample_interval_seconds,
        )


__all__ = [
    "RuntimeResourceMonitor",
    "finalize_resource_observability",
    "observed",
    "runtime_resource_snapshot",
]
