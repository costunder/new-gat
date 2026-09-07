"""Asynchronous V5 stage measurements, not a change to the learning recipe."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch


class StageTimer:
    """CPU wall time and CUDA events without per-update synchronization."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cpu_seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.events: list[tuple[str, Any, Any]] = []

    @contextmanager
    def stage(self, name: str):
        cuda = self.device.type == "cuda" and name != "sampling_and_loader_wait"
        start_event = end_event = None
        if cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        try:
            yield
        finally:
            self.cpu_seconds[name] = self.cpu_seconds.get(name, 0.0) + (
                time.perf_counter() - started
            )
            self.calls[name] = self.calls.get(name, 0) + 1
            if cuda:
                end_event.record(torch.cuda.current_stream(self.device))
                self.events.append((name, start_event, end_event))

    def report(self, *, synchronize: bool = False) -> dict[str, Any]:
        if synchronize and self.events:
            # Same-stream events: wait once at the reporting boundary, never
            # after each individual forward/backward/optimizer operation.
            self.events[-1][2].synchronize()
        gpu_seconds: dict[str, float] = {}
        for name, started, ended in self.events:
            gpu_seconds[name] = gpu_seconds.get(name, 0.0) + started.elapsed_time(ended) / 1000
        return {
            "cpu_wall_seconds": dict(self.cpu_seconds),
            "cuda_event_seconds": gpu_seconds,
            "stage_calls": dict(self.calls),
            "interpretation": (
                "CUDA event durations and CPU submission/wait durations overlap; do not add "
                "them as an end-to-end step time. Sampling includes exposed prefetch wait."
            ),
        }
