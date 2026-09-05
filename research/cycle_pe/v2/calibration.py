"""Disposable Cycle V2 real-training batch/worker probes, not final training.

The complete official training split and exact model profile are retained. A
candidate measures production loading, sparse collation, transfer, forward,
backward, clipping and an actual Adam update. No candidate writes a checkpoint
or changes the caller's arguments, model or RNG. Selection is a caller policy.
"""

from __future__ import annotations

import argparse
import copy
import gc
import heapq
import math
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from chartgat.execution import configure_execution
from research.cycle_pe.benchmark_data import EXPECTED_SIZES
from research.cycle_pe.resource_monitor import (
    FailureSafeResourceMonitor,
    resource_failure_boundary,
)
from research.cycle_pe.v2.data import Graph, collate, load_benchmark
from research.cycle_pe.v2.model import CycleBasisPEModel


def load_calibration_graphs(args: argparse.Namespace) -> tuple[list[Graph], dict[str, Any]]:
    """Prepare/cache/validate the complete official train split, without test access."""
    splits, identity = load_benchmark(
        args.data_root,
        args.dataset,
        allow_download=args.allow_download,
        splits=("train",),
        basis_backend=args.basis_backend,
        workers=args.workers,
    )
    expected = EXPECTED_SIZES[args.dataset][0]
    if (
        set(splits) != {"train"}
        or len(splits["train"]) != expected
        or identity.get("official_splits") is not True
        or identity.get("split_sizes") != {"train": expected}
        or not identity.get("split_content_sha256", {}).get("train")
    ):
        raise ValueError("Cycle calibration requires the complete verified official train split")
    return splits["train"], identity


@contextmanager
def _isolated_rng(seed: int, device: torch.device):
    """Restore RNG streams and the backend flag touched by benchmark._seed."""
    from research.cycle_pe.v2.benchmark import _seed

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    devices = [] if device.type != "cuda" else list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            _seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.backends.cudnn.benchmark = cudnn_benchmark


def _probe_model(dataset: str, args: argparse.Namespace, device: torch.device):
    """Construct exactly the architecture used by benchmark._train_model."""
    model = CycleBasisPEModel(
        dataset=dataset,
        encoding=args.encoding,
        hidden=args.hidden_dim,
        pe_dim=args.pe_dim,
        layers=args.layers,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
        layer_scale=args.layer_scale,
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters > args.max_parameters:
        raise ValueError(
            f"calibration model has {parameters} parameters, above the declared budget"
        )
    configure_execution(model, args, device)
    model.train()
    return model


def _optimizer_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        item.numel() * item.element_size()
        for state in optimizer.state.values()
        for item in state.values()
        if isinstance(item, torch.Tensor)
    )


def _training_step(
    model, optimizer, batch, device, precision, *, check_gradients=False, events=None
):
    """Actual MAE/backward/clip/Adam path; explicit CPU fixtures test this path."""
    from research.cycle_pe.v2.benchmark import (
        _require_finite_loss,
        _validate_first_task_gradients,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=precision["dtype"], enabled=precision["enabled"]):
        predicted = model(batch)
        loss = (predicted.float() - batch.y).abs().mean()
    _require_finite_loss(loss, "Cycle calibration: nonfinite task loss")
    if events is not None:
        events[0].record()
    loss.backward()
    gradients = _validate_first_task_gradients(model) if check_gradients else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    if events is not None:
        events[1].record()
    # Production Cycle precision is BF16 or FP32 with disabled GradScaler.
    # Actual Adam updates deliberately allocate both moments and step state.
    optimizer.step()
    if events is not None:
        events[2].record()
    return gradients


def _candidate_arguments(args, batch_size, workers, graph_count):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("physical batch candidate must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("worker candidate must be a nonnegative integer")
    if graph_count < 1:
        raise ValueError("official training split is empty")
    floor = args.batch_size
    natural_maximum = max(floor, graph_count)
    if not floor <= batch_size <= natural_maximum:
        raise ValueError("candidate must preserve the configured batch floor and natural boundary")
    candidate = copy.copy(args)
    candidate.batch_size = batch_size
    candidate.effective_batch_size = batch_size
    candidate.workers = workers
    return candidate


def _run_candidate(
    graphs, args, device, *, warmup_steps, measurement_steps, minimum_measure_seconds
):
    from research.cycle_pe.v2.benchmark import (
        _amp_policy,
        _graph_probe_cost,
        _loader,
        _precision_identity,
        _validate_optimizer_ownership,
    )

    model = _probe_model(args.dataset, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ownership = _validate_optimizer_ownership(model, optimizer)
    precision = _amp_policy(args.amp, device)
    stress_cpu = collate(heapq.nlargest(args.batch_size, graphs, key=_graph_probe_cost))
    stress = stress_cpu.pin_memory().to(device)
    gradients = _training_step(model, optimizer, stress, device, precision, check_gradients=True)
    torch.cuda.synchronize(device)
    stress_peak = int(torch.cuda.max_memory_allocated(device))
    del stress_cpu, stress
    loader = _loader(graphs, args, train=True)
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    try:
        for _ in range(warmup_steps):
            batch = next_batch().to(device)
            _training_step(model, optimizer, batch, device, precision)
            del batch
        torch.cuda.synchronize(device)
        batch_sizes, nodes, edges, step_events = [], [], [], []
        loader_seconds = 0.0
        elapsed = 0.0
        started = time.perf_counter()
        # Explicit calibration measurement window, never a final-training cap.
        while len(batch_sizes) < measurement_steps or elapsed < minimum_measure_seconds:
            for _ in range(measurement_steps):
                before_load = time.perf_counter()
                batch_cpu = next_batch()
                loader_seconds += time.perf_counter() - before_load
                batch_sizes.append(int(batch_cpu.y.shape[0]))
                nodes.append(int(batch_cpu.x.shape[0]))
                edges.append(int(batch_cpu.edge_index.shape[1]))
                events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(5))
                events[0].record()
                batch = batch_cpu.to(device)
                events[1].record()
                _training_step(model, optimizer, batch, device, precision, events=events[2:])
                step_events.append(events)
                del batch_cpu, batch
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        measured_count = sum(batch_sizes)
        if elapsed <= 0 or measured_count <= 0:
            raise RuntimeError("calibration observed no positive duration or real graphs")
        state_bytes = _optimizer_tensor_bytes(optimizer)
        if state_bytes <= 0 or not optimizer.state:
            raise RuntimeError("calibration never allocated Adam state")
        free_after, _ = torch.cuda.mem_get_info(device)
        stage_seconds = {
            "loader_wait": loader_seconds,
            "h2d": sum(e[0].elapsed_time(e[1]) for e in step_events) / 1000,
            "forward_loss": sum(e[1].elapsed_time(e[2]) for e in step_events) / 1000,
            "backward_clip": sum(e[2].elapsed_time(e[3]) for e in step_events) / 1000,
            "optimizer": sum(e[3].elapsed_time(e[4]) for e in step_events) / 1000,
        }
        return {
            "status": "passed",
            "samples_per_second": measured_count / elapsed,
            "elapsed_seconds": elapsed,
            "processed_units": measured_count,
            "unit": "molecular_graphs",
            "measurement_steps": len(batch_sizes),
            "warmup_steps": warmup_steps,
            "optimizer_steps": 1 + warmup_steps + len(batch_sizes),
            "optimizer_state_bytes": state_bytes,
            "optimizer_ownership": ownership,
            "first_task_gradient_connectivity": gradients,
            "scope": (
                "calibration-only real loader wait, sparse collate, H2D, forward, MAE, "
                "backward, gradient clip and Adam; not final predictive performance"
            ),
            "timer_boundary": (
                "CUDA synchronized after warmup and each measured chunk; initial dataset "
                "preparation, loader startup, capacity stress and warmup excluded"
            ),
            "dataset_graph_count": len(graphs),
            "training_dataset_reduced": False,
            "observed_batch_sizes": batch_sizes,
            "observed_nodes_per_batch": nodes,
            "observed_edges_per_batch": edges,
            "largest_graph_capacity_batch_size": args.batch_size,
            "capacity_batch_selection": (
                "highest _graph_probe_cost over the complete train split; a static tensor-size "
                "proxy, not a guarantee against every future minibatch OOM"
            ),
            "capacity_probe_peak_allocated_bytes": stress_peak,
            "stage_seconds": stage_seconds,
            "stage_timing": "CUDA events except CPU loader wait; stages may overlap",
            "free_bytes_after": int(free_after),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "architecture": {
                key: getattr(args, key)
                for key in (
                    "encoding", "hidden_dim", "pe_dim", "layers", "ffn_multiplier",
                    "dropout", "layer_scale",
                )
            },
            "precision": _precision_identity(precision),
            "loader": {
                "workers": args.workers,
                "prefetch_factor": args.prefetch_factor if args.workers else None,
                "persistent_workers": args.workers > 0,
                "pin_memory": True,
                "non_blocking_transfer": True,
                "sampler": "production shuffled full training split; no drop_last",
            },
        }
    finally:
        # Only owned loader workers are reclaimed by PyTorch iterator teardown.
        # No process signal or session/server management command is used.
        iterator = None
        loader = None


@resource_failure_boundary
def run_training_candidate(
    train_graphs: list[Graph],
    args: argparse.Namespace,
    device: torch.device,
    *,
    physical_batch_size: int,
    workers: int,
    warmup_steps: int = 2,
    measurement_steps: int = 5,
    minimum_measure_seconds: float = 3.0,
) -> dict[str, Any]:
    """Measure one exact batch/worker candidate on CUDA, without training artifacts.

    OOM is an explicit unsuccessful measurement, not a CPU or smaller-model
    fallback. Other failures propagate with resource-monitor failure evidence.
    args.dataset is the official dataset name; use load_calibration_graphs once
    per dataset to bind full-data provenance before testing multiple candidates.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Cycle training calibration requires CUDA; no CPU fallback")
    for name, value in (("warmup_steps", warmup_steps), ("measurement_steps", measurement_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive calibration-only step count")
    if (
        isinstance(minimum_measure_seconds, bool)
        or not math.isfinite(minimum_measure_seconds)
        or minimum_measure_seconds <= 0
    ):
        raise ValueError("minimum_measure_seconds must be finite and positive")
    candidate = _candidate_arguments(args, physical_batch_size, workers, len(train_graphs))
    monitor = FailureSafeResourceMonitor(
        device, workload=f"cycle_v2_{args.dataset}_batch_calibration"
    )
    resources_before = monitor.start()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    # CUDA Event.record uses the current device, not the model tensor's device.
    # Bind it explicitly so cuda:1+ probes cannot time a different GPU stream.
    with torch.cuda.device(device), _isolated_rng(args.model_seed, device):
        try:
            report = _run_candidate(
                train_graphs, candidate, device, warmup_steps=warmup_steps,
                measurement_steps=measurement_steps,
                minimum_measure_seconds=minimum_measure_seconds,
            )
        except torch.cuda.OutOfMemoryError as error:
            report = {"status": "oom", "error": f"{type(error).__name__}: {error}"}
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    resources = monitor.finish(
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved
    )
    report.update(
        dataset=args.dataset,
        batch_size=physical_batch_size,
        workers=workers,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        free_bytes_before=int(free_before),
        total_memory_bytes=int(total_bytes),
        resource_start=resources_before,
        resource_observability=resources,
        measurement_steps_requested=measurement_steps,
        warmup_steps_requested=warmup_steps,
        minimum_measure_seconds_requested=minimum_measure_seconds,
        calibration_only=True,
        final_training_performed=False,
        automatic_downsize=False,
        rng_restored=True,
    )
    if report["status"] == "oom":
        report["free_bytes_after"] = int(torch.cuda.mem_get_info(device)[0])
    gc.collect()
    torch.cuda.empty_cache()
    return report
