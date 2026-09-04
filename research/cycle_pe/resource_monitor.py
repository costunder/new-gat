"""Failure-safe lifetime management for Cycle resource monitors.

The underlying sampler is deliberately shared with the other research tracks.
This module only guarantees that a Cycle workload which starts a sampler also
finishes it exactly once when any ``BaseException`` unwinds the workload.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import torch

from chartgat.observability import RuntimeResourceMonitor

P = ParamSpec("P")
R = TypeVar("R")

_ACTIVE_MONITORS: ContextVar[tuple[FailureSafeResourceMonitor, ...]] = ContextVar(
    "cycle_active_resource_monitors", default=()
)
_FAILURE_ATTRIBUTE = "cycle_resource_failure_observations"


def resource_failure_observations(error: BaseException) -> list[dict[str, Any]]:
    """Return JSON-safe monitor failures attached while preserving ``error``."""

    value = getattr(error, _FAILURE_ATTRIBUTE, ())
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def persist_failure_artifacts(
    original_error: BaseException,
    artifacts: tuple[tuple[str, Callable[[], None]], ...],
) -> None:
    """Attempt every failure write without replacing the workload exception."""

    for label, writer in artifacts:
        try:
            writer()
        except BaseException as reporting_error:
            original_error.add_note(
                f"failure artifact persistence failed ({label}) without replacing the "
                f"workload error: {type(reporting_error).__name__}: {reporting_error}"
            )


def _register(monitor: FailureSafeResourceMonitor) -> None:
    _ACTIVE_MONITORS.set((*_ACTIVE_MONITORS.get(), monitor))


def _unregister(monitor: FailureSafeResourceMonitor) -> None:
    _ACTIVE_MONITORS.set(
        tuple(item for item in _ACTIVE_MONITORS.get() if item is not monitor)
    )


def _failure_peaks(device: torch.device) -> tuple[int | None, int | None, list[str]]:
    if device.type != "cuda":
        return None, None, []
    errors: list[str] = []
    allocated: int | None = None
    reserved: int | None = None
    try:
        allocated = int(torch.cuda.max_memory_allocated(device))
    except BaseException as error:
        errors.append(
            "max_memory_allocated failed with "
            f"{type(error).__name__}: {error}"
        )
    try:
        reserved = int(torch.cuda.max_memory_reserved(device))
    except BaseException as error:
        errors.append(
            "max_memory_reserved failed with "
            f"{type(error).__name__}: {error}"
        )
    return allocated, reserved, errors


class FailureSafeResourceMonitor:
    """One ``RuntimeResourceMonitor`` whose failure cleanup cannot mask work errors."""

    def __init__(self, device: torch.device, *, workload: str) -> None:
        self.device = device
        self.workload = workload
        self._monitor = RuntimeResourceMonitor(device)
        self._started = False
        self._finish_started = False
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def start(self) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("resource monitor session was already started")
        self._started = True
        _register(self)
        return self._monitor.start()

    def finish(
        self,
        *,
        peak_allocated_bytes: int | None,
        peak_reserved_bytes: int | None,
    ) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("resource monitor session was not started")
        if self._finish_started:
            raise RuntimeError("resource monitor session finish was already attempted")
        self._finish_started = True
        try:
            return self._monitor.finish(
                peak_allocated_bytes=peak_allocated_bytes,
                peak_reserved_bytes=peak_reserved_bytes,
            )
        finally:
            self._finished = True
            _unregister(self)

    def finish_after_failure(self, original_error: BaseException) -> dict[str, Any]:
        """Finish once, attach observations, and never raise over ``original_error``."""

        if not self._started or self._finished:
            return {}
        allocated, reserved, peak_errors = _failure_peaks(self.device)
        resources: dict[str, Any] | None = None
        cleanup_error: dict[str, str] | None = None
        if self._finish_started:
            cleanup_error = {
                "type": "RuntimeError",
                "message": "resource monitor finish had already started before failure cleanup",
            }
            self._finished = True
            _unregister(self)
        else:
            self._finish_started = True
            try:
                resources = self._monitor.finish(
                    peak_allocated_bytes=allocated,
                    peak_reserved_bytes=reserved,
                )
            except BaseException as error:
                cleanup_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            finally:
                self._finished = True
                _unregister(self)
        payload: dict[str, Any] = {
            "status": "failed",
            "workload": self.workload,
            "original_error": {
                "type": type(original_error).__name__,
                "message": str(original_error),
            },
            "resource_observability": resources,
            "peak_collection_errors": peak_errors,
            "monitor_cleanup_error": cleanup_error,
            "failure_payload_delivery_errors": [],
        }
        delivery_errors: list[dict[str, str]] = payload[
            "failure_payload_delivery_errors"
        ]
        existing = resource_failure_observations(original_error)
        try:
            setattr(original_error, _FAILURE_ATTRIBUTE, [*existing, payload])
        except (AttributeError, TypeError) as error:
            delivery_errors.append(
                {
                    "stage": "exception_attribute",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        try:
            serialized = json.dumps(payload, sort_keys=True)
            original_error.add_note(f"Cycle resource failure observation: {serialized}")
        except (AttributeError, TypeError, ValueError) as error:
            delivery_errors.append(
                {"stage": "exception_note", "type": type(error).__name__, "message": str(error)}
            )
        try:
            print(
                json.dumps(
                    {"kind": "cycle_resource_failure_observability", **payload},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        except (OSError, TypeError, UnicodeError, ValueError) as error:
            delivery_errors.append(
                {"stage": "stderr_json", "type": type(error).__name__, "message": str(error)}
            )
            try:
                original_error.add_note(
                    "Cycle resource failure stderr delivery failed with "
                    f"{type(error).__name__}: {error}"
                )
            except (AttributeError, TypeError):
                # The already-attached payload remains the authoritative record.
                delivery_errors.append(
                    {
                        "stage": "stderr_failure_note",
                        "type": "unavailable",
                        "message": "the exception did not accept an additional note",
                    }
                )
        return payload


def _finish_active_monitors(original_error: BaseException) -> None:
    for monitor in reversed(_ACTIVE_MONITORS.get()):
        try:
            monitor.finish_after_failure(original_error)
        except BaseException as cleanup_error:
            message = (
                "Cycle monitor failure cleanup raised unexpectedly but did not replace "
                f"the workload error: {type(cleanup_error).__name__}: {cleanup_error}"
            )
            try:
                original_error.add_note(message)
            except (AttributeError, TypeError):
                try:
                    print(message, file=sys.stderr, flush=True)
                except (OSError, UnicodeError):
                    # There is no remaining safe reporting channel; the original
                    # workload exception is still deliberately re-raised below.
                    continue


def resource_failure_boundary(function: Callable[P, R]) -> Callable[P, R]:
    """Close monitors opened by one call while preserving its original exception."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _ACTIVE_MONITORS.set(())
        try:
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                _finish_active_monitors(error)
                raise
            dangling = _ACTIVE_MONITORS.get()
            if dangling:
                error = RuntimeError(
                    f"{function.__qualname__} returned with an unfinished resource monitor"
                )
                _finish_active_monitors(error)
                raise error
            return result
        finally:
            _ACTIVE_MONITORS.reset(token)

    return wrapped


__all__ = [
    "FailureSafeResourceMonitor",
    "persist_failure_artifacts",
    "resource_failure_boundary",
    "resource_failure_observations",
]
