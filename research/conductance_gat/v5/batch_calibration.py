"""Disposable real-training V5 measurements; never a final training run.

The caller compares multiple candidates before publishing an immutable batch
plan. This module does not silently alter a training recipe or write checkpoints.
Every candidate retains the exact architecture, precision, fanouts and data.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from chartgat.observability import RuntimeResourceMonitor

from . import train


class _StageTimer:
    """CPU wall time plus asynchronous CUDA events; no per-stage synchronization."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cpu_seconds: dict[str, float] = {}
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
            if cuda:
                end_event.record(torch.cuda.current_stream(self.device))
                self.events.append((name, start_event, end_event))

    def report(self) -> dict[str, Any]:
        # Caller synchronizes once at the complete-epoch measurement boundary.
        gpu_seconds: dict[str, float] = {}
        for name, started, ended in self.events:
            gpu_seconds[name] = gpu_seconds.get(name, 0.0) + started.elapsed_time(ended) / 1000
        return {
            "cpu_wall_seconds": dict(self.cpu_seconds),
            "cuda_event_seconds": gpu_seconds,
            "interpretation": (
                "CUDA event durations and CPU submission/wait durations overlap; do not add "
                "them as an end-to-end step time. Sampling includes exposed prefetch wait."
            ),
        }


@contextmanager
def _isolated_execution_state(device: torch.device):
    """Preserve caller RNG and global precision flags even after candidate OOM."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    float32_precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_benchmark = torch.backends.cudnn.benchmark
    devices = []
    if device.type == "cuda":
        # The production _seed uses manual_seed_all; preserve every visible generator.
        devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_float32_matmul_precision(float32_precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark


def load_calibration_payload(args: argparse.Namespace):
    """Reuse the production verified V1 cache; downloading and test evaluation disabled."""
    return train.load_dataset(args.dataset, args.data_root, allow_download=False)


def _candidate_args(args, physical_batch_size: int, workers: int):
    candidate = copy.deepcopy(args)
    train.validate_args(candidate)
    if isinstance(physical_batch_size, bool) or not isinstance(physical_batch_size, int):
        raise ValueError("physical_batch_size must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    if candidate.dataset == "ppi":
        minimum = candidate.batch_size
        candidate.batch_size = physical_batch_size
        candidate.workers = workers
        candidate.worker_configuration_source = "measured_batch_calibration_candidate"
    elif candidate.sampling != "full":
        minimum = candidate.sample_seed_batch_size
        candidate.sample_seed_batch_size = physical_batch_size
        if workers != 0:
            raise ValueError("transductive sampling uses CPU CSR/prefetch, not DataLoader workers")
    else:
        minimum = 1
        if physical_batch_size != 1 or workers != 0:
            raise ValueError("a full transductive graph cannot be replicated to inflate batch size")
    if physical_batch_size < minimum:
        raise ValueError("calibration cannot reduce the currently requested physical batch size")
    return candidate


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _update(model, optimizer, graph, indices, args, phase, timing, *, validate: bool):
    with timing.stage("zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with timing.stage("forward_and_loss"):
        with train.autocast_context(args):
            logits = model(graph)
            loss, label_count = train.training_loss(logits, graph, indices)
    with timing.stage("backward"):
        loss.backward()
    if validate:
        train.validate_active_gradient_connectivity(model, phase["active_parameter_groups"])
        if args.condition == "shared_dynamic_c":
            train.require_first_step_conductance_gradient(model)
    with timing.stage("gradient_clipping"):
        norm = torch.nn.utils.clip_grad_norm_(
            (value for value in model.parameters() if value.requires_grad),
            train.COMMON["gradient_clip_norm"],
            error_if_nonfinite=False,
            foreach=True,
        )
        train.require_finite_gradient_norm_async(norm)
    with timing.stage("optimizer"):
        optimizer.step()
    return label_count


def _run_epoch(model, optimizer, data, indices, sampler, args, device, epoch, timing):
    phase = train.configure_phase(model, "joint", 0)
    processed_units = optimizer_steps = supervised_labels = 0
    largest_nodes = largest_edges = largest_graph_batch = 0
    for graph, selected in train._training_batches(
        data, indices, sampler, epoch, device, args.model_seed, args, timing=timing
    ):
        label_count = _update(
            model, optimizer, graph, selected, args, phase, timing,
            validate=not optimizer.state,
        )
        supervised_labels += label_count
        processed_units += int(graph.num_graphs) if indices is None else label_count
        optimizer_steps += 1
        largest_nodes = max(largest_nodes, int(graph.x.shape[0]))
        largest_edges = max(largest_edges, int(graph.incidence_edge_index.shape[1]))
        largest_graph_batch = max(
            largest_graph_batch, int(graph.num_graphs) if indices is None else 1
        )
    if not optimizer_steps or not processed_units:
        raise RuntimeError("calibration full training epoch produced no supervised updates")
    return {
        "processed_units": processed_units,
        "optimizer_steps": optimizer_steps,
        "supervised_labels": supervised_labels,
        "largest_measured_nodes": largest_nodes,
        "largest_measured_physical_edges": largest_edges,
        "largest_measured_graph_batch": largest_graph_batch,
    }


def _ppi_stress_update(payload, args, device, model, optimizer):
    """Fit-check the largest real training graphs together without changing final ordering."""
    from torch_geometric.data import Batch, Data

    order = sorted(
        payload["splits"]["train"],
        key=lambda index: (
            payload["graphs"][index]["incidence_edge_index"].shape[1]
            + payload["graphs"][index]["x"].shape[0]
        ),
        reverse=True,
    )
    selected = order[:args.batch_size]
    graph = Batch.from_data_list([Data(**payload["graphs"][index]) for index in selected])
    graph._v5_num_graphs = int(graph.num_graphs)
    if args.pin_memory:
        graph = graph.pin_memory()
    graph = graph.to(device, non_blocking=args.pin_memory)
    phase = train.configure_phase(model, "joint", 0)
    _update(model, optimizer, graph, None, args, phase, _StageTimer(device), validate=True)
    return {
        "scope": "largest real training graphs ranked by physical edges plus nodes",
        "graphs": len(selected),
        "nodes": int(graph.x.shape[0]),
        "physical_edges": int(graph.incidence_edge_index.shape[1]),
        "optimizer_updated": True,
        "not_final_training": True,
    }


def run_training_candidate(
    payload, args, device: torch.device, *, physical_batch_size: int, workers: int,
    warmup_steps: int = 2, measurement_steps: int = 5,
    minimum_measure_seconds: float = 3.0,
) -> dict[str, Any]:
    """Measure a disposable joint-phase model, including real AdamW state and IO.

    Warmup/measurement budgets are minima, not dataset caps: each round finishes
    an entire official training epoch. No final epoch budget/checkpoint is touched.
    The full-epoch sweep exposes every seed/graph, not just a small easy batch.
    """
    train._require_cuda(device)
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (warmup_steps, measurement_steps))
        or isinstance(minimum_measure_seconds, bool)
        or not isinstance(minimum_measure_seconds, (int, float))
        or not math.isfinite(minimum_measure_seconds)
        or minimum_measure_seconds <= 0
    ):
        raise ValueError("calibration step and duration minima must be positive")
    candidate = _candidate_args(args, physical_batch_size, workers)
    if payload.get("dataset") != candidate.dataset:
        raise ValueError("calibration dataset does not match the verified cache payload")
    model = optimizer = data = indices = sampler = None
    monitor = None
    report = None
    with _isolated_execution_state(device):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            hardware = train.validate_hardware_runtime(candidate, device)
            free_before, total = torch.cuda.mem_get_info(device)
            torch.cuda.reset_peak_memory_stats(device)
            monitor = RuntimeResourceMonitor(device)
            monitor.start()
            train.configure_compute(candidate)
            train._seed(candidate.model_seed)
            setup_started = time.perf_counter()
            data, indices, sampler = train._prepare_data(payload, candidate, device)
            model = train.GraphConditionedConductanceNodeClassifier(
                payload["graphs"][0]["x"].shape[1], payload["classes"],
                **train.architecture_configuration(candidate),
                conductance_mode=train.CONDITIONS[candidate.condition]["conductance_mode"],
                max_log_conductance=train.COMMON["max_log_conductance"],
                edge_chunk_size=candidate.edge_chunk_size,
            ).to(device)
            optimizer = train.make_optimizer(model)
            train.validate_optimizer_parameter_ownership(model, optimizer)
            initial_hash = train.state_sha256(model)
            torch.cuda.synchronize(device)
            setup_seconds = time.perf_counter() - setup_started
            stress = (
                _ppi_stress_update(payload, candidate, device, model, optimizer)
                if candidate.dataset == "ppi" else {
                    "scope": "every training seed covered by full warmup and measurement epochs",
                    "sampling": candidate.sampling,
                    "fanouts": list(candidate.num_neighbors),
                    "limitation": (
                        "sampled topology can vary in later epochs; no universal OOM guarantee"
                    ),
                }
            )
            warmup_updates = 0
            warmup_epoch = 0
            while warmup_updates < warmup_steps:
                warmup_epoch += 1
                warmup = _run_epoch(
                    model, optimizer, data, indices, sampler, candidate, device,
                    warmup_epoch, _StageTimer(device),
                )
                warmup_updates += warmup["optimizer_steps"]
            if not optimizer.state or _optimizer_state_bytes(optimizer) <= 0:
                raise RuntimeError("calibration did not materialize real AdamW optimizer state")
            torch.cuda.synchronize(device)
            timing = _StageTimer(device)
            started = time.perf_counter()
            measured_epochs = measured_steps = units = labels = 0
            largest_nodes = largest_edges = largest_graph_batch = 0
            elapsed = 0.0
            while measured_steps < measurement_steps or elapsed < minimum_measure_seconds:
                measured_epochs += 1
                values = _run_epoch(
                    model, optimizer, data, indices, sampler, candidate, device,
                    warmup_epoch + measured_epochs, timing,
                )
                measured_steps += values["optimizer_steps"]
                units += values["processed_units"]
                labels += values["supervised_labels"]
                largest_nodes = max(largest_nodes, values["largest_measured_nodes"])
                largest_edges = max(largest_edges, values["largest_measured_physical_edges"])
                largest_graph_batch = max(
                    largest_graph_batch, values["largest_measured_graph_batch"]
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            free_after, _ = torch.cuda.mem_get_info(device)
            report = {
                "status": "passed", "calibration_not_final": True,
                "elapsed_seconds": elapsed, "processed_units": units,
                "unit": "graphs" if indices is None else "supervised_seed_nodes",
                "optimizer_steps": measured_steps, "samples_per_second": units / elapsed,
                "measurement_steps_requested": measurement_steps,
                "minimum_measure_seconds_requested": minimum_measure_seconds,
                "warmup_steps_requested": warmup_steps,
                "stage_seconds": timing.report(), "setup_seconds": setup_seconds,
                "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                "free_bytes_before": int(free_before), "free_bytes_after": int(free_after),
                "total_memory_bytes": int(total), "batch_size": physical_batch_size,
                "workers": workers, "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
                "model_parameter_count": sum(value.numel() for value in model.parameters()),
                "initial_model_sha256": initial_hash,
                "parameter_update_verified": initial_hash != train.state_sha256(model),
                "warmup_optimizer_steps": warmup_updates,
                "complete_measurement_epochs": measured_epochs,
                "complete_warmup_epochs": warmup_epoch,
                "supervised_labels": labels,
                "largest_measured_nodes": largest_nodes,
                "largest_measured_physical_edges": largest_edges,
                "largest_measured_graph_batch": largest_graph_batch,
                "stress_observation": stress,
                "model_phase": "joint_all_condition_parameter_groups_active",
                "sampling": sampler.metadata() if sampler is not None else {"mode": "full"},
                "configuration": train.configuration(candidate), "hardware": hardware,
                "scope": (
                    "fresh real training batches, full training epochs; "
                    "no validation/test/checkpoints"
                ),
                "gradient_accumulation_steps": 1, "data_parallel_workers": 1,
                "effective_batch_size": physical_batch_size,
            }
            if not report["parameter_update_verified"]:
                raise RuntimeError("disposable calibration model parameters did not update")
        except BaseException as error:
            if monitor is not None:
                failed_monitor, monitor = monitor, None
                try:
                    error.calibration_resource_observability = failed_monitor.finish(
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    )
                except BaseException as cleanup_error:
                    error.add_note(
                        f"calibration resource finalization also failed: {cleanup_error}"
                    )
            raise
        finally:
            try:
                if monitor is not None:
                    resources = monitor.finish(
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    )
                    if report is not None:
                        report["resource_observability"] = resources
            finally:
                model = optimizer = data = indices = sampler = None
                gc.collect()
                torch.cuda.empty_cache()
    return report
