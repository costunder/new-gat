"""All-source receiver aggregation and conditional E/J reconstruction audit."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

from ...wedge_propagation.study import Tee, available_cpus, runtime_resources, synchronize
from ..operators import aggregate_squares, incidence, relation_terms, solve_local_laplacian
from ..study import Table, _estimated_bytes, _memory, reference_states
from .contract import assert_source_unchanged, read_config, source_manifest, write_json
from .data import assert_inputs_unchanged, prepare_cases
from .operators import (
    ambient_receipt_projection,
    batch_receiver_operators,
    fuse_receipts,
    fused_apply,
    fused_transpose,
    graph_norm_sq,
    prepare_receiver_operator,
    project_components,
    receipt_reconstruct,
    recover_fused,
    shared_flow,
    tagged_receipts,
)
from .report import write_report


def _prepare(argument):
    top, mode = argument
    torch.set_num_threads(1)  # Each sparse preparation process owns one CPU thread.
    return prepare_receiver_operator(top, mode)


def _parallel(arguments, workers):
    if workers == 1:
        previous = torch.get_num_threads()
        try:
            return [_prepare(item) for item in arguments]
        finally:
            torch.set_num_threads(previous)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_prepare, arguments))


def prepare_operators(cases, topologies, config, hardware, resources):
    arguments = [(top, mode) for top in topologies for mode in config["weights"]]
    nsynthetic = config["source"]["synthetic_graphs"]
    largest = max(case.num_nodes for case in cases[:nsynthetic])
    indices = [
        index * 2 + mode
        for index, case in enumerate(cases)
        if index >= nsynthetic or case.num_nodes == largest
        for mode in range(2)
    ]
    requested = config["runtime"]["cpu_workers"]
    candidates = (
        [requested]
        if requested != "auto"
        else sorted(
            {
                min(k, available_cpus(hardware))
                for k in config["runtime"]["preprocessing_worker_candidates"]
            }
        )
    )
    if any(k > available_cpus(hardware) for k in candidates):
        raise ValueError("CPU processes exceed actual affinity/quota")
    best, cache, selected = float("inf"), None, None
    for workers in candidates:
        start = time.perf_counter()
        prepared = _parallel([arguments[index] for index in indices], workers)
        seconds = time.perf_counter() - start
        resources.append(
            {
                "type": "sparse_operator_cpu_calibration",
                "workers": workers,
                "operators": len(indices),
                "seconds": seconds,
                "scope": "largest_whole_synthetic_graphs_and_all_whole_citation_graphs",
            }
        )
        print(
            f"[CPU operator calibration] workers={workers} operators={len(indices)} "
            f"seconds={seconds:.3f}",
            flush=True,
        )
        if seconds < best:
            best, cache, selected = seconds, prepared, workers
    prepared = dict(zip(indices, cache, strict=True))
    rest = [index for index in range(len(arguments)) if index not in prepared]
    prepared.update(
        zip(rest, _parallel([arguments[index] for index in rest], selected), strict=True)
    )
    result = [
        {mode: prepared[index * 2 + j] for j, mode in enumerate(config["weights"])}
        for index in range(len(cases))
    ]
    for case, operators in zip(cases, result, strict=True):
        for mode, op in operators.items():
            resources.append(
                {
                    "type": "prepared_operator",
                    "graph_id": case.graph_id,
                    "weight": mode,
                    **op.metadata,
                }
            )
    return result


def batch_operators(items, weights):
    return {mode: batch_receiver_operators([item[mode] for item in items]) for mode in weights}


def _numpy(value):
    return value.detach().cpu().numpy()


def make_observations(receipts, receiver_sum, energy, relations):
    """The five exact forward observations; no learned or arbitrary compression."""
    return {
        "tagged": {"receipts": receipts},
        "sum": {"sum": receiver_sum},
        "sum_within": {"sum": receiver_sum, "energy": energy},
        "sum_between": {"sum": receiver_sum, "relations": relations},
        "sum_both": {"sum": receiver_sum, "energy": energy, "relations": relations},
    }


@torch.no_grad()
def compute(operators, x, config, relation_batch, progress_label=None):
    top = next(iter(operators.values())).topology
    states = reference_states(top, x)
    result = {}
    for mode, op in operators.items():
        originals, reconstructed, metrics, energies, energy_predictions = [], [], [], [], []
        observed_relations = {}
        for stage, h in enumerate(states):
            started = time.perf_counter()
            if progress_label:
                print(
                    f"[compute] {progress_label} C={mode} H{stage} "
                    "tagged and receiver-sum recovery",
                    flush=True,
                )
            q, d = shared_flow(op, h)
            r = tagged_receipts(op, d)
            projection = ambient_receipt_projection(op, r)
            observations = make_observations(
                r,
                projection["fused"],
                aggregate_squares(
                    top, incidence(top, h[top.local_node_global]), weights=op.weights
                ),
                relation_terms(top, q, relation_batch=relation_batch),
            )
            y = observations["sum"]["sum"]
            contrast = projection["contrast"]
            contrast_sum = fuse_receipts(op, contrast)
            d_tagged = receipt_reconstruct(op, observations["tagged"]["receipts"])
            observed_relations[stage, stage] = {
                key: _numpy(observations["sum_between"]["relations"][key])
                for key in config["relations"]
            }

            def tag_progress(iteration, maximum, seconds, mode=mode, stage=stage):
                print(
                    f"[CG tagged] {progress_label} C={mode} H{stage} "
                    f"iteration={iteration}/{maximum} seconds={seconds:.1f}",
                    flush=True,
                )

            potential, tag_info = solve_local_laplacian(
                top,
                d_tagged,
                op.weights,
                tol=config["solver"]["tolerance"],
                iteration_multiplier=config["solver"]["maximum_iterations_factor"],
                progress=tag_progress if progress_label else None,
            )
            tagged_q = op.weights[:, None] * incidence(top, potential)

            def sum_progress(iteration, maximum, seconds, mode=mode, stage=stage):
                print(
                    f"[CG receiver sum] {progress_label} C={mode} H{stage} "
                    f"iteration={iteration}/{maximum} seconds={seconds:.1f}",
                    flush=True,
                )

            h_hat, info = recover_fused(
                op,
                y,
                tolerance=config["solver"]["tolerance"],
                max_iterations_factor=config["solver"]["maximum_iterations_factor"],
                progress=sum_progress if progress_label else None,
            )
            q_hat, d_hat = shared_flow(op, h_hat)
            r_hat = tagged_receipts(op, d_hat)
            y_hat = fused_apply(op, h_hat)
            values = {
                "tagged_norm_sq": graph_norm_sq(op, r, space="tagged"),
                "tag_contrast_norm_sq": graph_norm_sq(op, contrast, space="tagged"),
                "null_projection_residual_sq": graph_norm_sq(op, contrast_sum, space="local_node"),
                "q_norm_sq": graph_norm_sq(op, q, space="local_edge"),
                "tagged_q_error_sq": graph_norm_sq(op, tagged_q - q, space="local_edge"),
                "fused_q_error_sq": graph_norm_sq(op, q_hat - q, space="local_edge"),
                "centered_h_norm_sq": graph_norm_sq(
                    op, project_components(op, h), space="physical"
                ),
                "centered_h_error_sq": graph_norm_sq(
                    op, h_hat - project_components(op, h), space="physical"
                ),
                "tagged_reconstruction_error_sq": graph_norm_sq(op, r_hat - r, space="tagged"),
                "fused_norm_sq": graph_norm_sq(op, y, space="local_node"),
                "fused_residual_sq": graph_norm_sq(op, y_hat - y, space="local_node"),
                "solver_iterations_max": info["iterations"],
                "solver_residual_max": info["relative_residual"],
                "solver_normal_residual_max": info["normal_relative_residual"],
            }
            metrics.append({name: _numpy(value) for name, value in values.items()})
            energies.append(_numpy(observations["sum_within"]["energy"]))
            energy_predictions.append(
                _numpy(
                    aggregate_squares(
                        top, incidence(top, h_hat[top.local_node_global]), weights=op.weights
                    )
                )
            )
            originals.append(q)
            reconstructed.append(q_hat)
            if progress_label:
                print(
                    f"[compute done] {progress_label} C={mode} H{stage} "
                    f"iterations={info['global_iterations']} "
                    f"seconds={time.perf_counter() - started:.3f}",
                    flush=True,
                )
            del (
                values,
                observations,
                r,
                projection,
                contrast,
                contrast_sum,
                d_tagged,
                potential,
                tag_info,
                tagged_q,
                d_hat,
                r_hat,
                y_hat,
                h_hat,
                d,
                y,
            )
        relations = {}
        for left, right in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2)):
            original = (
                observed_relations[left, right]
                if left == right
                else {
                    key: _numpy(value)
                    for key, value in relation_terms(
                        top, originals[left], originals[right], relation_batch=relation_batch
                    ).items()
                }
            )
            predicted = relation_terms(
                top, reconstructed[left], reconstructed[right], relation_batch=relation_batch
            )
            relations[left, right] = {
                key: {"original": original[key], "predicted": _numpy(predicted[key])}
                for key in config["relations"]
            }
        result[mode] = {
            "metrics": metrics,
            "energy": energies,
            "energy_prediction": energy_predictions,
            "relations": relations,
        }
    return result


def merge_trace(accumulator, computed):
    """Sum original channels before squaring scalar E/J observation errors."""
    for mode, value in computed.items():
        target = accumulator.setdefault(
            mode,
            {
                "metrics": [{}, {}, {}],
                "energy": [None] * 3,
                "energy_prediction": [None] * 3,
                "relations": {},
            },
        )
        for stage, metrics in enumerate(value["metrics"]):
            for key, array in metrics.items():
                maximum = key.startswith("solver_")
                reduced = array.max(1, keepdims=True) if maximum else array.sum(1, keepdims=True)
                previous = target["metrics"][stage].get(key)
                target["metrics"][stage][key] = (
                    reduced
                    if previous is None
                    else (np.maximum(previous, reduced) if maximum else previous + reduced)
                )
            for key in ("energy", "energy_prediction"):
                reduced = value[key][stage].sum(1, keepdims=True)
                target[key][stage] = (
                    reduced if target[key][stage] is None else target[key][stage] + reduced
                )
        for stages, relations in value["relations"].items():
            destination = target["relations"].setdefault(stages, {})
            for relation, arrays in relations.items():
                field = destination.setdefault(relation, {})
                for key, array in arrays.items():
                    field[key] = field.get(key, 0) + array.sum(1, keepdims=True)
    return accumulator


def _relative(error, norm):
    return float(np.sqrt(error / norm)) if norm > 0 else None


def _meta(case):
    return {
        "graph_id": case.graph_id,
        "family": case.family,
        "dataset": case.graph_id if case.family == "citation" else "synthetic",
        "actual_data": case.actual_data,
        "feature_mode": case.feature_mode,
    }


def graph_rows(cases, operators):
    result = []
    for case, modes in zip(cases, operators, strict=True):
        op = modes["unit"]
        top = op.topology
        result.append(
            {
                **_meta(case),
                "num_nodes": case.num_nodes,
                "num_edges": top.num_edges,
                "num_features": case.num_features,
                "num_local_nodes": top.num_local_nodes,
                "components": op.num_components,
                "tagged_coordinates": op.metadata["tagged_coordinates"],
                "fused_active_coordinates": op.metadata["fused_active_coordinates"],
                "ambient_tag_kernel_per_channel": op.metadata["ambient_receipt_kernel_dimension"],
                "restricted_rank_per_channel": op.metadata["restricted_field_rank"],
                **{f"sparse_nnz_{mode}": modes[mode].metadata["sparse_nnz"] for mode in modes},
            }
        )
    return result


def collect(cases, cpu_operators, computed, graph_info, rows, table):
    top = next(iter(cpu_operators.values())).topology
    for index, case in enumerate(cases):
        meta = _meta(case)
        node_start, node_end = map(int, top.graph_node_offsets[index : index + 2])
        pair_start, pair_end = map(int, 2 * top.graph_edge_offsets[index : index + 2])
        dimensions = graph_info[case.graph_id]
        channels = 1 if case.feature_mode == "independent_scalar_columns" else case.num_features
        for mode, data in computed.items():
            for stage, metrics in enumerate(data["metrics"]):
                values = {key: array[index] for key, array in metrics.items()}
                macro = {
                    **meta,
                    "weight": mode,
                    "state": f"H{stage}",
                    "count": len(values["q_norm_sq"]),
                    **{
                        key: float(array.max()) if key.startswith("solver_") else float(array.sum())
                        for key, array in values.items()
                    },
                }
                macro["tag_contrast_fraction_sq"] = (
                    float(macro["tag_contrast_norm_sq"] / macro["tagged_norm_sq"])
                    if macro["tagged_norm_sq"] > 0
                    else None
                )
                for key, error, norm in (
                    ("tagged_q_relative_error", "tagged_q_error_sq", "q_norm_sq"),
                    ("fused_q_relative_error", "fused_q_error_sq", "q_norm_sq"),
                    ("centered_h_relative_error", "centered_h_error_sq", "centered_h_norm_sq"),
                    (
                        "tagged_reconstruction_relative_error",
                        "tagged_reconstruction_error_sq",
                        "tagged_norm_sq",
                    ),
                    ("fused_relative_residual", "fused_residual_sq", "fused_norm_sq"),
                ):
                    macro[key] = _relative(macro[error], macro[norm])
                rows["reconstruction"].append(macro)
                table.write(
                    {
                        **meta,
                        "weight": mode,
                        "state": f"H{stage}",
                        "realization": draw
                        if case.feature_mode == "independent_scalar_columns"
                        else "vector_trace",
                        **{key: float(array[draw]) for key, array in values.items()},
                    }
                    for draw in range(macro["count"])
                )
                energy = data["energy"][stage][node_start:node_end]
                energy_hat = data["energy_prediction"][stage][node_start:node_end]
                energy_norm = float(np.square(energy).sum())
                energy_error = float(np.square(energy_hat - energy).sum())
                relation_norm = relation_error = 0.0
                for arrays in data["relations"][stage, stage].values():
                    original = arrays["original"][pair_start:pair_end]
                    predicted = arrays["predicted"][pair_start:pair_end]
                    relation_norm += float(np.square(original).sum())
                    relation_error += float(np.square(predicted - original).sum())
                for condition in config_conditions():
                    within = condition in ("sum_within", "sum_both")
                    between = condition in ("sum_between", "sum_both")
                    coordinates = (
                        dimensions["tagged_coordinates"]
                        if condition == "tagged"
                        else dimensions["num_local_nodes"]
                    ) * channels
                    coordinates += case.num_nodes * int(within) + 6 * case.edges.shape[1] * int(
                        between
                    )
                    rows["condition"].append(
                        {
                            **meta,
                            "weight": mode,
                            "state": f"H{stage}",
                            "count": macro["count"],
                            "condition": condition,
                            "observed_coordinates": coordinates,
                            "restricted_rank": dimensions["restricted_rank_per_channel"] * channels,
                            "additional_rank": 0,
                            "q_relative_error": macro["tagged_q_relative_error"]
                            if condition == "tagged"
                            else macro["fused_q_relative_error"],
                            "energy_observed": within,
                            "relation_observed": between,
                            "energy_norm_sq": energy_norm if within else None,
                            "energy_error_sq": energy_error if within else None,
                            "energy_relative_error": _relative(energy_error, energy_norm)
                            if within
                            else None,
                            "relation_norm_sq": relation_norm if between else None,
                            "relation_error_sq": relation_error if between else None,
                            "relation_relative_error": _relative(relation_error, relation_norm)
                            if between
                            else None,
                        }
                    )
            for (left, right), relations in data["relations"].items():
                for relation, arrays in relations.items():
                    original = arrays["original"][pair_start:pair_end]
                    predicted = arrays["predicted"][pair_start:pair_end]
                    error = float(np.square(predicted - original).sum())
                    norm = float(np.square(original).sum())
                    rows["relation"].append(
                        {
                            **meta,
                            "weight": mode,
                            "stage_from": f"H{left}",
                            "stage_to": f"H{right}",
                            "relation": relation,
                            "count": int(original.size),
                            "original_sum": float(original.sum()),
                            "reconstructed_sum": float(predicted.sum()),
                            "error_sq": error,
                            "norm_sq": norm,
                            "relative_error": _relative(error, norm),
                            "negative_count": int((original < 0).sum()),
                        }
                    )


def config_conditions():
    return ("tagged", "sum", "sum_within", "sum_between", "sum_both")


@torch.no_grad()
def calibration_probe(operators, x, relation_batch):
    """Exact forward/adjoint kernels and conservative solver-buffer memory probe."""
    # This allocation probe is not a shortened reconstruction result.
    for op in operators.values():
        q, d = shared_flow(op, x)
        tagged = tagged_receipts(op, d)
        projection = ambient_receipt_projection(op, tagged)
        energy = aggregate_squares(
            op.topology,
            incidence(op.topology, x[op.topology.local_node_global]),
            weights=op.weights,
        )
        relations = relation_terms(op.topology, q, relation_batch=relation_batch)
        buffers = [torch.empty_like(x) for _ in range(12)]
        working = x
        for _ in range(8):
            working = project_components(op, fused_transpose(op, fused_apply(op, working)))
            working = working / op.normal_diagonal.clamp_min(1)[:, None]
        del buffers, working, q, d, tagged, projection, energy, relations


def _allocation_bytes(cpu_ops, channels, pairs):
    op = next(iter(cpu_ops.values()))
    static = sum(
        operator.matrix.values().numel() * 16
        + operator.matrix.crow_indices().numel() * 8
        + operator.matrix_transpose.values().numel() * 16
        + operator.matrix_transpose.crow_indices().numel() * 8
        for operator in cpu_ops.values()
    )
    return int(
        static
        + 2 * _estimated_bytes(op.topology, channels, pairs)
        + channels * 8 * (4 * op.metadata["tagged_coordinates"] + 16 * op.topology.n)
    )


def calibrate(cases, cpu_ops, config, device, resources, mode):
    graph_mode = mode == "graphs"
    upper = len(cases) if graph_mode else cases[0].num_features
    requested = config["runtime"]["physical_graph_batch" if graph_mode else "channel_chunk"]
    candidates = (
        sorted(
            {
                min(upper, k)
                for k in (
                    [4, 8, 16, 32, 64, 128, upper]
                    if graph_mode
                    else [16, 64, 128, 256, 512, 1024, upper]
                )
            }
        )
        if requested == "auto"
        else [requested]
    )
    if any(candidate > upper for candidate in candidates):
        raise ValueError("runtime batch exceeds complete input dimension")
    records = []
    cache = {}
    for candidate in candidates:
        prepared = (
            batch_operators(cpu_ops[:candidate], config["weights"]) if graph_mode else cpu_ops[0]
        )
        channels = cases[0].num_features if graph_mode else candidate
        count = next(iter(prepared.values())).topology.num_pairs
        requested_pairs = config["runtime"]["relation_batch"]
        pairs = (
            sorted({min(max(1, count), k) for k in (256, 4096, max(1, count))})
            if requested_pairs == "auto"
            else [requested_pairs]
        )
        for pair_batch in pairs:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            free = (
                torch.cuda.mem_get_info(device)[0]
                if device.type == "cuda"
                else psutil.virtual_memory().available
            )
            estimated = _allocation_bytes(prepared, channels, pair_batch)
            row = {
                "type": "GPU_allocation_calibration",
                "mode": mode,
                "graph_id": "synthetic" if graph_mode else cases[0].graph_id,
                "candidate": candidate,
                "relation_batch": pair_batch,
                "estimated_bytes": estimated,
                "available_bytes": free,
                "scope": "exact_forward_adjoint_and_peak_buffers_not_reconstruction",
            }
            if estimated > free * config["runtime"]["gpu_memory_safety_fraction"]:
                row.update(status="skipped_memory_estimate", cells_per_second=None)
                records.append(row)
                print(
                    f"[calibration] {mode}={candidate} pair_chunk={pair_batch} "
                    "skipped memory estimate",
                    flush=True,
                )
                continue
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            started = time.perf_counter()
            uploaded = x = None
            try:
                uploaded = {key: value.to(device) for key, value in prepared.items()}
                xcpu = (
                    torch.cat([case.x for case in cases[:candidate]], 0)
                    if graph_mode
                    else cases[0].x[:, :candidate].contiguous()
                )
                x = xcpu.to(device)
                calibration_probe(uploaded, x, pair_batch)
                synchronize(device)
                seconds = time.perf_counter() - started
                memory = _memory(device)
                safe = (
                    device.type != "cuda"
                    or memory["peak_vram_bytes"]
                    <= free * config["runtime"]["gpu_memory_safety_fraction"]
                )
                row.update(
                    status="measured" if safe else "rejected_measured_memory",
                    seconds=seconds,
                    cells_per_second=x.numel() / seconds,
                    **memory,
                )
            except torch.cuda.OutOfMemoryError as error:
                if device.type != "cuda":
                    raise
                row.update(
                    status="CUDA_OOM", error=str(error), cells_per_second=None, **_memory(device)
                )
            finally:
                del uploaded, x
            records.append(row)
            cache[candidate] = prepared
            print(
                f"[calibration] {mode}={candidate} pair_chunk={pair_batch} "
                f"status={row['status']} cells/s={row['cells_per_second']}",
                flush=True,
            )
    resources.extend(records)
    measured = [row for row in records if row["status"] == "measured"]
    if not measured:
        raise RuntimeError("no safe measured exact allocation; inputs preserved")
    best = max(measured, key=lambda row: row["cells_per_second"])
    print(
        f"[selected] {mode}={best['candidate']} pair_chunk={best['relation_batch']}; "
        "all inputs retained",
        flush=True,
    )
    return best["candidate"], best["relation_batch"], cache[best["candidate"]]


def validate_device(config, device):
    if config["profile"] == "full" and (os.name != "posix" or device.type != "cuda"):
        raise ValueError("FULL belongs on server CUDA; local checks use separate DEBUG")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("select one actually allocated GPU with CUDA_VISIBLE_DEVICES")


def run(args, output):
    config = read_config(args.config, args.profile)
    device = torch.device(args.device)
    validate_device(config, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hardware = runtime_resources(device)
    hardware["output_storage"] = shutil.disk_usage(output)._asdict()
    threads = config["runtime"]["cpu_threads"]
    threads = available_cpus(hardware) if threads == "auto" else threads
    if threads > available_cpus(hardware):
        raise ValueError("CPU threads exceed affinity/quota")
    torch.set_num_threads(threads)
    source = source_manifest()
    write_json(output / "source_manifest.json", source)
    write_json(output / "config.json", config)
    write_json(
        output / "hardware.json",
        {
            **hardware,
            "torch_cpu_threads": threads,
            "precision": "float64",
            "TF32": False,
            "physical_batch": "measured_complete_graphs",
            "effective_batch_equals_physical": True,
            "gradient_accumulation_steps": 1,
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "DataLoader": "complete_source_cache_no_resampling",
        },
    )
    print(
        f"[start] profile={config['profile']} device={device} precision=float64 "
        "conditions=5 trainable_parameters=0",
        flush=True,
    )
    print("[hardware] " + json.dumps(hardware), flush=True)
    print(
        "[scope] source-tag contrast vs restricted-flow reconstruction; "
        "E/J conditional observations; no classifier training",
        flush=True,
    )
    started = time.perf_counter()
    cases, topologies, manifest = prepare_cases(config, args.source_dir, output)
    resources = list(manifest["cpu_input_calibration"])
    cpu_ops = prepare_operators(cases, topologies, config, hardware, resources)
    graphs = graph_rows(cases, cpu_ops)
    info = {row["graph_id"]: row for row in graphs}
    nsynthetic = config["source"]["synthetic_graphs"]
    order = sorted(
        range(nsynthetic),
        key=lambda i: cpu_ops[i]["unit"].metadata["tagged_coordinates"],
        reverse=True,
    )
    synthetic_cases = [cases[i] for i in order]
    synthetic_ops = [cpu_ops[i] for i in order]
    batch_size, pairs, prepared = calibrate(
        synthetic_cases, synthetic_ops, config, device, resources, "graphs"
    )
    rows = {key: [] for key in ("reconstruction", "condition", "relation")}
    table = Table(output / "input_metrics.csv")
    completed = []
    try:
        for first in range(0, nsynthetic, batch_size):
            batch_cases = synthetic_cases[first : first + batch_size]
            prepared = (
                prepared
                if first == 0
                else batch_operators(synthetic_ops[first : first + batch_size], config["weights"])
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            uploaded = {key: op.to(device) for key, op in prepared.items()}
            x = torch.cat([case.x for case in batch_cases], 0).to(device)
            result = compute(
                uploaded,
                x,
                config,
                pairs,
                progress_label=f"synthetic-{first + 1}-{first + len(batch_cases)}",
            )
            synchronize(device)
            compute_seconds = time.perf_counter() - began
            collect_start = time.perf_counter()
            collect(batch_cases, prepared, result, info, rows, table)
            resources.append(
                {
                    "type": "execution",
                    "graph_ids": [case.graph_id for case in batch_cases],
                    "physical_graph_batch": len(batch_cases),
                    "channel_chunk": x.shape[1],
                    "all_channels": x.shape[1],
                    "relation_batch": pairs,
                    "compute_seconds": compute_seconds,
                    "export_seconds": time.perf_counter() - collect_start,
                    "cells_per_second": x.numel() / compute_seconds,
                    "rss_bytes": psutil.Process().memory_info().rss,
                    **_memory(device),
                }
            )
            completed.extend(case.graph_id for case in batch_cases)
            print(
                f"[progress] synthetic={len(completed)}/{nsynthetic} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            del uploaded, x, result
        for case, prepared in zip(cases[nsynthetic:], cpu_ops[nsynthetic:], strict=True):
            chunk, pair_chunk, _ = calibrate(
                [case], [prepared], config, device, resources, "channels"
            )
            uploaded = {key: op.to(device) for key, op in prepared.items()}
            accumulator = {}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            for first in range(0, case.num_features, chunk):
                x = case.x[:, first : first + chunk].contiguous().to(device)
                merge_trace(
                    accumulator,
                    compute(
                        uploaded,
                        x,
                        config,
                        pair_chunk,
                        progress_label=f"{case.graph_id}-channels-{first + 1}-{first + x.shape[1]}",
                    ),
                )
                print(
                    f"[progress] {case.graph_id} "
                    f"channels={min(first + chunk, case.num_features)}/{case.num_features} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                del x
            synchronize(device)
            compute_seconds = time.perf_counter() - began
            collect_start = time.perf_counter()
            collect([case], prepared, accumulator, info, rows, table)
            resources.append(
                {
                    "type": "execution",
                    "graph_ids": [case.graph_id],
                    "physical_graph_batch": 1,
                    "physical_graph_batch_reason": "entire_dataset_graph_all_local_rows",
                    "channel_chunk": chunk,
                    "all_channels": case.num_features,
                    "relation_batch": pair_chunk,
                    "compute_seconds": compute_seconds,
                    "export_seconds": time.perf_counter() - collect_start,
                    "cells_per_second": case.x.numel() / compute_seconds,
                    "rss_bytes": psutil.Process().memory_info().rss,
                    **_memory(device),
                }
            )
            completed.append(case.graph_id)
            del uploaded, accumulator
    finally:
        table.close()
    expected_units = sum(
        case.num_features if case.feature_mode == "independent_scalar_columns" else 1
        for case in cases
    )
    if (
        len(completed) != config["source"]["graphs"]
        or set(completed) != {case.graph_id for case in cases}
        or table.count != 6 * expected_units
    ):
        raise ValueError("incomplete graph/input/condition execution coverage")
    assert_source_unchanged(source)
    assert_inputs_unchanged(args.source_dir, manifest)
    completion = {
        "status": "complete",
        "profile": config["profile"],
        "debug": config["profile"] == "debug",
        "graphs": len(completed),
        "synthetic_graphs": nsynthetic,
        "citation_graphs": 3,
        "actual_citation_graphs": sum(case.actual_data for case in cases),
        "synthetic_scalar_inputs": config["source"]["synthetic_scalar_inputs"],
        "sampling_ratio": 1.0,
        "all_nodes_edges_features": True,
        "weights": config["weights"],
        "states": config["states"],
        "conditions": config["conditions"],
        "trainable_parameters": 0,
        "optimizer_updates": 0,
        "classifier_training_run": False,
        "decoder_uses_energy_or_relations": False,
        "restricted_q_rank_loss": 0,
        "source_and_inputs_preserved": True,
        "source_dir": str(args.source_dir.resolve()),
        "source_digest": source["code_digest"],
        "raw_input_metric_rows": table.count,
        "physical_graph_batch_selected": batch_size,
        "macro_rows": {key: len(value) for key, value in rows.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        "[report] receiver contrasts, reconstruction, E/J checks and scientific figures", flush=True
    )
    write_report(
        output,
        config,
        graphs,
        rows["reconstruction"],
        rows["condition"],
        rows["relation"],
        resources,
        completion,
    )
    assert_source_unchanged(source)
    assert_inputs_unchanged(args.source_dir, manifest)
    write_json(output / "completion.json", completion)
    print(f"[complete] graphs={len(completed)} results={output.resolve()}", flush=True)
    print("[scope] exact receiver audit; no learned E/J decoder or classifier training", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_device(read_config(args.config, args.profile), torch.device(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "terminal.log").open("x", encoding="utf-8") as logfile:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, logfile)),
            contextlib.redirect_stderr(Tee(sys.stderr, logfile)),
        ):
            try:
                run(args, args.output_dir)
            except Exception as error:
                traceback.print_exc()
                write_json(
                    args.output_dir / "failure.json",
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "source_inputs_preserved": True,
                    },
                )
                raise


if __name__ == "__main__":
    main()
