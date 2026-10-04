"""Server fixed audit of context formation, cancellation and delayed coupling."""

from __future__ import annotations

import argparse
import contextlib
import gc
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

from ..wedge_propagation.study import Tee, available_cpus, runtime_resources, synchronize
from ..local_energy_relations.study import Table, _memory, reference_states
from ..local_energy_relations.topology import batch_topologies
from ..local_energy_relations.receiver_aggregation.data import assert_inputs_unchanged, prepare_cases
from .contract import assert_source_unchanged, digest, read_config, source_manifest, source_reader_config, write_json
from .operators import apply_cross, apply_intra, cross_energy, immediate, intra_energy, merge, mixed_action, prepare_geometry, sandwich
from .report import write_report


def validate_device(config, device):
    if config["profile"] == "full" and (sys.platform != "linux" or device.type != "cuda" or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("FULL requires Linux server CUDA and explicit allocated CUDA_VISIBLE_DEVICES")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no CPU fallback")


def _norm_sq(values, index, graphs):
    return values.new_zeros((graphs, values.shape[-1])).index_add(0, index, values.square())


def _step(geometry):
    maximum = geometry.intra_degree.new_zeros(geometry.num_graphs).scatter_reduce_(
        0, geometry.copy_graph, geometry.intra_degree + geometry.cross_degree, reduce="amax"
    )
    denominator = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    return torch.where(maximum > 0, 0.5 / denominator, 0)


def _guards(metrics, config):
    tolerance = config["measurement"]["relative_identity_tolerance"]
    scale = np.maximum(metrics["replicated_norm_sq"], np.finfo(np.float64).tiny)
    for name in ("replicate_merge_residual_norm_sq", "initial_cross_action_norm_sq", "merged_context_cross_action_norm_sq", "immediate_increment_norm_sq", "sandwich_formula_residual_norm_sq", "persistent_1_increment_norm_sq", "persistent_2_increment_norm_sq", "persistent_3_formula_residual_norm_sq"):
        if np.any(metrics[name] > tolerance**2 * scale):
            raise ValueError(f"matrix-free identity residual exceeded relative tolerance: {name}")
    for before, after in (("replicated_norm_sq", "after_intra_1_norm_sq"), ("after_intra_1_norm_sq", "after_cross_norm_sq"), ("after_cross_norm_sq", "after_intra_2_norm_sq")):
        if np.any(metrics[after] - metrics[before] > tolerance * scale):
            raise ValueError(f"safe step contraction violated: {before} -> {after}")
    if any(not np.isfinite(value).all() for value in metrics.values()):
        raise FloatingPointError("nonfinite fixed coupling diagnostic; results preserved")


@torch.no_grad()
def compute(geometries, x, config, progress_label=None):
    """Every existing channel is an RHS; this is not a full input Jacobian."""
    states = reference_states(next(iter(geometries.values())).topology, x)
    result = {}
    for mode, geometry in geometries.items():
        for state, h in zip(config["states"], states, strict=True):
            if progress_label:
                print(f"[compute] {progress_label} C={mode} {state} all locals and channels", flush=True)
            on, details = sandwich(geometry, h, diagnostics=True)
            off = sandwich(geometry, h, cross=False)
            initial = details["replicated"]
            first, middle, last = (details[key] for key in ("after_intra_1", "after_cross", "after_intra_2"))
            context = apply_cross(geometry, first)
            mixed = mixed_action(geometry, h)
            coefficient = (geometry.gamma * geometry.eta.square())[geometry.physical_graph, None]
            expected = -coefficient * mixed
            immediate_delta = immediate(geometry, h) - immediate(geometry, h, cross=False)
            physical = lambda value: _norm_sq(value, geometry.physical_graph, geometry.num_graphs)
            copies = lambda value: _norm_sq(value, geometry.copy_graph, geometry.num_graphs)
            fields = {
                "input_norm_sq": physical(h),
                "replicated_norm_sq": copies(initial),
                "after_intra_1_norm_sq": copies(first),
                "after_cross_norm_sq": copies(middle),
                "after_intra_2_norm_sq": copies(last),
                "replicate_merge_residual_norm_sq": physical(merge(geometry, initial) - h),
                "initial_cross_action_norm_sq": copies(apply_cross(geometry, initial)),
                "context_cross_action_norm_sq": copies(context),
                "merged_context_cross_action_norm_sq": physical(merge(geometry, context)),
                "sandwich_output_norm_sq": physical(on),
                "sandwich_off_output_norm_sq": physical(off),
                "sandwich_increment_norm_sq": physical(on - off),
                "mixed_action_norm_sq": physical(mixed),
                "predicted_increment_norm_sq": physical(expected),
                "sandwich_formula_residual_norm_sq": physical(on - off - expected),
                "immediate_increment_norm_sq": physical(immediate_delta),
                "intra_energy_initial": intra_energy(geometry, initial),
                "intra_energy_after_intra_1": intra_energy(geometry, first),
                "intra_energy_after_cross": intra_energy(geometry, middle),
                "intra_energy_after_intra_2": intra_energy(geometry, last),
                "cross_energy_initial": cross_energy(geometry, initial),
                "cross_energy_after_intra_1": cross_energy(geometry, first),
                "cross_energy_after_cross": cross_energy(geometry, middle),
                "cross_energy_after_intra_2": cross_energy(geometry, last),
            }
            theta = _step(geometry)
            copy_step = theta[geometry.copy_graph, None]
            u_on, u_off = initial, initial
            for step in config["operators"]["persistent_steps"]:
                # Independent copy state persists across the three updates.
                # On/off use the identical bound including K in both cases.
                u_on = u_on - copy_step * (apply_intra(geometry, u_on) + apply_cross(geometry, u_on))
                u_off = u_off - copy_step * apply_intra(geometry, u_off)
                delta = merge(geometry, u_on) - merge(geometry, u_off)
                fields[f"persistent_{step}_increment_norm_sq"] = physical(delta)
                if step == 3:
                    predicted = -theta[geometry.physical_graph, None].pow(3) * mixed
                    fields["persistent_3_predicted_increment_norm_sq"] = physical(predicted)
                    fields["persistent_3_formula_residual_norm_sq"] = physical(delta - predicted)
            # One small packed diagnostic transfer after the whole matrix-free stage.
            keys = list(fields)
            host = torch.stack([fields[key] for key in keys]).cpu().numpy()
            metrics = dict(zip(keys, host, strict=True))
            _guards(metrics, config)
            result[mode, state] = metrics
            del on, off, initial, first, middle, last, context, mixed, expected, fields, u_on, u_off
            del details, copy_step, coefficient, theta, predicted
    return result


def _prepare(argument):
    top, mode = argument
    torch.set_num_threads(1)
    return prepare_geometry(top, mode)


def _parallel(arguments, workers):
    previous = torch.get_num_threads()
    try:
        if workers == 1:
            return [_prepare(argument) for argument in arguments]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            return list(pool.map(_prepare, arguments))
    finally:
        torch.set_num_threads(previous)


def prepare_geometries(cases, topologies, config, hardware, resources):
    available = available_cpus(hardware)
    arguments = [(top, mode) for top in topologies for mode in config["weights"]]
    synthetic = config["source"]["synthetic_graphs"]
    largest = max(case.num_nodes for case in cases[:synthetic])
    indices = [2 * index + mode for index, case in enumerate(cases) if index >= synthetic or case.num_nodes == largest for mode in range(2)]
    requested = config["runtime"]["cpu_workers"]
    candidates = [requested] if requested != "auto" else sorted({min(value, available) for value in config["runtime"]["preprocessing_worker_candidates"]})
    if any(value > available for value in candidates):
        raise ValueError("geometry CPU workers exceed actual affinity/quota")
    best, cached, selected = float("inf"), None, None
    for workers in candidates:
        start = time.perf_counter()
        values = _parallel([arguments[index] for index in indices], workers)
        seconds = time.perf_counter() - start
        resources.append({"type": "CPU_geometry_calibration", "workers": workers, "operators": len(indices), "seconds": seconds, "worker_start_method": "spawn" if workers > 1 else "in_process", "scope": "largest_synthetic_bucket_and_every_whole_citation"})
        print(f"[CPU geometry calibration] workers={workers} operators={len(indices)} seconds={seconds:.4f}", flush=True)
        if seconds < best:
            best, cached, selected = seconds, values, workers
    prepared = dict(zip(indices, cached, strict=True))
    remaining = [index for index in range(len(arguments)) if index not in prepared]
    prepared.update(zip(remaining, _parallel([arguments[index] for index in remaining], selected), strict=True))
    resources.append({"type": "CPU_geometry_selection", "workers": selected, "operators": len(arguments), "static_cache": "every_complete_graph_weight_geometry_prepared_once_and_retained_on_CPU", "calibrated_cache_reused": True})
    return [{mode: prepared[2 * index + j] for j, mode in enumerate(config["weights"])} for index in range(len(cases))]


def choose_cpu_threads(geometry, config, hardware, resources):
    available = available_cpus(hardware)
    requested = config["runtime"]["cpu_threads"]
    candidates = [requested] if requested != "auto" else sorted({min(value, available) for value in config["runtime"]["cpu_thread_candidates"]})
    if any(value > available for value in candidates):
        raise ValueError("CPU thread count exceeds actual affinity/quota")
    rows = []
    value = torch.ones(geometry.num_copies, 4, dtype=torch.float64)
    for threads in candidates:
        torch.set_num_threads(threads)
        apply_intra(geometry, value); apply_cross(geometry, value)
        start = time.perf_counter()
        for _ in range(3):
            apply_intra(geometry, value); apply_cross(geometry, value)
        seconds = (time.perf_counter() - start) / 3
        rows.append({"type": "CPU_threads_calibration", "threads": threads, "seconds": seconds, "scope": "all_copy_rows_of_largest_graph_operator_4_calibration_columns"})
        print(f"[CPU thread calibration] threads={threads} seconds={seconds:.5f}", flush=True)
    resources.extend(rows)
    selected = min(rows, key=lambda row: row["seconds"])["threads"]
    torch.set_num_threads(selected)
    return selected


def _batch_geometry(topologies, modes):
    top = batch_topologies(topologies)
    return {mode: prepare_geometry(top, mode) for mode in modes}


def _footprint(case, geometry):
    """Order calibration prefixes by all dominant copy/edge allocations."""
    return (geometry.num_copies + geometry.topology.local_edge_nodes.shape[1] + geometry.num_cross_edges) * case.num_features


def calibrate(cases, topologies, geometries, config, device, resources, graph_mode):
    upper = len(cases) if graph_mode else cases[0].num_features
    setting = "physical_graph_batch" if graph_mode else "channel_chunk"
    requested = config["runtime"][setting]
    options = config["runtime"]["graph_batch_candidates" if graph_mode else "channel_chunk_candidates"]
    candidates = sorted({min(upper, value) for value in [*options, upper]}) if requested == "auto" else [requested]
    if any(value > upper for value in candidates):
        raise ValueError(f"runtime {setting} exceeds assigned complete input dimension")
    rows, caches = [], {}
    for size in candidates:
        cpu = _batch_geometry(topologies[:size], config["weights"]) if graph_mode else geometries[0]
        features = torch.cat([case.x for case in cases[:size]]) if graph_mode else cases[0].x[:, :size].contiguous()
        baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else psutil.virtual_memory().available
        row = {"type": "GPU_batch_calibration" if device.type == "cuda" else "DEBUG_CPU_batch_calibration", "device": str(device), "allocation_dimension": setting, "size": size, "graphs": size if graph_mode else 1, "channels": features.shape[1], "input_tensor_shape": list(features.shape), "scope": "calibration_only_all_nodes_of_selected_whole_graphs"}
        uploaded = x = measured = None
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            uploaded = {mode: geometry.to(device) for mode, geometry in cpu.items()}
            x = features.to(device)
            for _ in range(config["runtime"]["calibration_warmups"]):
                compute(uploaded, x, config)
            synchronize(device)
            start = time.perf_counter()
            for _ in range(config["runtime"]["calibration_repeats"]):
                measured = compute(uploaded, x, config)
            synchronize(device)
            seconds = (time.perf_counter() - start) / config["runtime"]["calibration_repeats"]
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            safe = peak is None or peak - baseline < free * config["runtime"]["gpu_memory_safety_fraction"]
            row.update(seconds=seconds, peak_vram_bytes=peak, input_cells_per_second=features.numel()/seconds, status="measured" if safe else "memory_safety_rejected")
            caches[size] = cpu
        except torch.cuda.OutOfMemoryError as error:
            row.update(status="OOM", error=str(error))
        finally:
            del uploaded, x, measured
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[GPU calibration] device={device} {setting}={size} status={row['status']} seconds={row.get('seconds')}", flush=True)
        rows.append(row)
    resources.extend(rows)
    feasible = [row for row in rows if row["status"] == "measured"]
    if not feasible:
        raise RuntimeError("no measured safe batch/chunk; original graph/channel scope preserved")
    chosen = max(feasible, key=lambda row: (row["input_cells_per_second"], row["size"]))
    return chosen["size"], caches[chosen["size"]]


def _metadata(case, geometry, mode, state):
    return {"graph_id": case.graph_id, "family": case.family, "weight": mode, "state": state, "feature_mode": case.feature_mode, "actual_data": case.actual_data, "nodes": case.num_nodes, "physical_edges": case.edges.shape[1], "channels": case.num_features, "local_copies": geometry.num_copies, "cross_edges": geometry.num_cross_edges, "eta": float(geometry.eta[0]), "gamma": float(geometry.gamma[0])}


def _ratio(numerator, denominator):
    return float(np.sqrt(numerator/denominator)) if denominator > 0 else None


def summarize(base, sums, config):
    row = {**base, **{key: float(value) for key, value in sums.items()}}
    row["sandwich_relative_increment"] = _ratio(row["sandwich_increment_norm_sq"], row["sandwich_off_output_norm_sq"])
    row["sandwich_relative_formula_error"] = _ratio(row["sandwich_formula_residual_norm_sq"], row["replicated_norm_sq"])
    threshold_sq = config["measurement"]["numerically_resolved_relative_threshold"]**2 * row["replicated_norm_sq"]
    row["context_numerically_resolved"] = row["context_cross_action_norm_sq"] > threshold_sq
    row["predicted_increment_numerically_resolved"] = row["predicted_increment_norm_sq"] > threshold_sq
    row["observed_increment_numerically_resolved"] = row["sandwich_increment_norm_sq"] > threshold_sq
    row["dependency_measurement"] = "all_existing_feature_column_operator_actions_not_full_Jacobian"
    return row


def _execute(indices, cases, topologies, geometries, config, device, resources, table, lock, seen):
    if device.type == "cuda":
        torch.cuda.set_device(device)
    synthetic = sorted(
        (index for index in indices if index < config["source"]["synthetic_graphs"]),
        key=lambda index: (_footprint(cases[index], geometries[index]["unit"]), cases[index].graph_id),
        reverse=True,
    )
    citation = [index for index in indices if index >= config["source"]["synthetic_graphs"]]
    graph_rows = []

    def collect(batch, result, first):
        with lock:
            for (mode, state), fields in result.items():
                for position, index in enumerate(batch):
                    base = _metadata(cases[index], geometries[index][mode], mode, state)
                    matrix = {key: values[position] for key, values in fields.items()}
                    raw = []
                    for channel in range(next(iter(matrix.values())).shape[0]):
                        key = (cases[index].graph_id, mode, state, first + channel)
                        if key in seen:
                            raise ValueError("duplicate physical graph/channel/C/state diagnostic")
                        seen.add(key)
                        raw.append({**base, "channel": first + channel, **{name: float(values[channel]) for name, values in matrix.items()}})
                    table.write(raw)
                    identity = (index, mode, state)
                    totals = accumulators.setdefault(identity, {})
                    for name, values in matrix.items():
                        totals[name] = totals.get(name, 0.) + float(values.sum(dtype=np.float64))

    accumulators = {}
    if synthetic:
        assigned_cases = [cases[index] for index in synthetic]
        assigned_tops = [topologies[index] for index in synthetic]
        assigned_geo = [geometries[index] for index in synthetic]
        size, cached = calibrate(assigned_cases, assigned_tops, assigned_geo, config, device, resources, True)
        for first in range(0, len(synthetic), size):
            batch = synthetic[first:first+size]
            host = cached if first == 0 else _batch_geometry([topologies[index] for index in batch], config["weights"])
            uploaded = {mode: geometry.to(device) for mode, geometry in host.items()}
            x = torch.cat([cases[index].x for index in batch]).to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            result = compute(uploaded, x, config, f"{device}-synthetic-{first+1}-{first+len(batch)}")
            synchronize(device)
            seconds = time.perf_counter()-start
            collect(batch, result, 0)
            resources.append({"type":"execution", "device":str(device), "graph_ids":[cases[index].graph_id for index in batch], "physical_graph_batch":len(batch), "effective_batch_size":len(batch), "gradient_accumulation_steps":"N/A_fixed_audit", "channels":x.shape[1], "input_tensor_shape":list(x.shape), "seconds":seconds, "input_cells_per_second":x.numel()/seconds, "rss_bytes":psutil.Process().memory_info().rss, **_memory(device)})
            print(f"[progress] {device} synthetic={first+len(batch)}/{len(synthetic)} seconds={seconds:.3f}", flush=True)
            del uploaded, x, result
    for index in citation:
        case = cases[index]
        chunk, _ = calibrate([case], [topologies[index]], [geometries[index]], config, device, resources, False)
        uploaded = {mode: geometry.to(device) for mode, geometry in geometries[index].items()}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for first in range(0, case.num_features, chunk):
            x = case.x[:,first:first+chunk].contiguous().to(device)
            result = compute(uploaded, x, config, f"{device}-{case.graph_id}-channels-{first+1}-{first+x.shape[1]}")
            collect([index], result, first)
            print(f"[progress] {device} {case.graph_id} channels={first+x.shape[1]}/{case.num_features}", flush=True)
            del x, result
        synchronize(device)
        seconds=time.perf_counter()-start
        resources.append({"type":"execution", "device":str(device), "graph_ids":[case.graph_id], "physical_graph_batch":1, "effective_batch_size":1, "gradient_accumulation_steps":"N/A_fixed_audit", "physical_graph_batch_reason":"whole_variable_size_citation_graph_all_local_copies", "channel_chunk":chunk, "all_channels":case.num_features, "input_tensor_shape":[case.num_nodes,case.num_features], "input_chunk_shape_max":[case.num_nodes,chunk], "seconds":seconds, "input_cells_per_second":case.x.numel()/seconds, "rss_bytes":psutil.Process().memory_info().rss, **_memory(device)})
        del uploaded
    for (index, mode, state), sums in accumulators.items():
        graph_rows.append(summarize(_metadata(cases[index], geometries[index][mode], mode, state), sums, config))
    return graph_rows


def run(args, output):
    start = time.perf_counter()
    config = read_config(args.config, args.profile)
    if args.workers != "auto":
        config["runtime"]["cpu_workers"] = args.workers
    device = torch.device(args.device)
    validate_device(config, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = source_manifest()
    write_json(output/"config.json", config)
    write_json(output/"source_manifest.json", source)
    hardware = runtime_resources(device)
    hardware["output_storage"] = shutil.disk_usage(output)._asdict()
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())] if args.device == "cuda" else [device]
    hardware["worker_devices"] = [str(value) for value in devices]
    hardware["gpu_count_used"] = len(devices) if device.type == "cuda" else 0
    hardware["worker_gpus"] = []
    for target in devices:
        if target.type == "cuda":
            properties = torch.cuda.get_device_properties(target)
            free, total = torch.cuda.mem_get_info(target)
            hardware["worker_gpus"].append({
                "device": str(target), "name": properties.name,
                "uuid": str(getattr(properties, "uuid", "unavailable")),
                "total_memory_bytes": properties.total_memory,
                "runtime_total_memory_bytes": total, "free_memory_bytes": free,
                "capability": [properties.major, properties.minor],
                "multiprocessors": properties.multi_processor_count,
            })
    print(f"[start] context coupling fixed audit profile={args.profile} devices={devices} precision=float64 graphs={config['source']['graphs']} sampling=1.0 trainable_parameters=0 optimizer_updates=0 epochs=N/A", flush=True)
    print(f"[hardware] {hardware}", flush=True)
    from .verify import run_checks
    preflight = run_checks(device)
    write_json(output/"preflight_debug_checks.json", preflight)
    print("[DEBUG preflight] dense fixture algebra and CE/gradient linkage checked; outside fixed audit budget", flush=True)
    reader = source_reader_config(config)
    cases, topologies, manifest = prepare_cases(reader, args.source_dir, output, workers=args.workers)
    if len(cases) != config["source"]["graphs"] or sum(case.num_features for case in cases) != config["data"]["physical_feature_columns"]:
        raise ValueError("complete graph/channel scope differs from fixed contract")
    write_json(output/"source_adapter.json", {"config_digest":digest(config), "unchanged_legacy_reader_config_digest":digest(reader), "source_reference_digest":manifest["source_content_digest"], "scope_equal":True, "reader_only_no_receiver_inverse_run":True})
    resources = list(manifest["cpu_input_calibration"])
    hardware["source_input_workers"] = manifest.get("selected_workers")
    hardware["input_cache"] = "all_original_CPU_features_and_complete_topologies_retained_after_immutable_snapshot_read"
    geometries = prepare_geometries(cases, topologies, config, hardware, resources)
    largest = max(range(len(cases)), key=lambda index: geometries[index]["unit"].num_copies)
    hardware["torch_cpu_threads"] = choose_cpu_threads(geometries[largest]["unit"], config, hardware, resources)
    write_json(output/"hardware.json", hardware)
    assignments=[[] for _ in devices]
    costs=[0 for _ in devices]
    for index in sorted(range(len(cases)), key=lambda value: _footprint(cases[value], geometries[value]["unit"]), reverse=True):
        owner=min(range(len(devices)), key=lambda value:costs[value])
        assignments[owner].append(index)
        costs[owner]+=_footprint(cases[index], geometries[index]["unit"])
    seen, lock, graph_rows = set(), threading.RLock(), []
    table=Table(output/"channel_metrics.csv")
    try:
        arguments=[(indices,cases,topologies,geometries,config,target,resources,table,lock,seen) for target,indices in zip(devices,assignments,strict=True) if indices]
        if len(arguments)==1:
            graph_rows.extend(_execute(*arguments[0]))
        else:
            with ThreadPoolExecutor(max_workers=len(arguments)) as pool:
                futures=[pool.submit(_execute,*argument) for argument in arguments]
                for future in futures:
                    graph_rows.extend(future.result())
    finally:
        table.close()
    expected_rows = len(config["weights"])*len(config["states"])*config["data"]["physical_feature_columns"]
    expected_graph_rows = len(cases)*len(config["weights"])*len(config["states"])
    expected_keys = {(case.graph_id, mode, state) for case in cases for mode in config["weights"] for state in config["states"]}
    if len(seen)!=expected_rows or table.count!=expected_rows or len(graph_rows)!=expected_graph_rows or {(row["graph_id"],row["weight"],row["state"]) for row in graph_rows} != expected_keys:
        raise ValueError("incomplete original graph/channel/weight/reference coverage")
    assert_source_unchanged(source)
    assert_inputs_unchanged(args.source_dir, manifest)
    coverage={"graphs":len(cases),"physical_feature_columns":config["data"]["physical_feature_columns"],"channel_metric_rows":table.count,"graph_summary_rows":len(graph_rows),"all_nodes_edges_channels_locals_shared_copy_links":True}
    write_json(output/"coverage.json",coverage)
    completion={"status":"complete","profile":args.profile,"debug":args.profile=="debug","scope":"fixed_context_coupling_mechanism_audit","graphs":len(cases),"synthetic_graphs":config["source"]["synthetic_graphs"],"actual_citation_graphs":sum(case.actual_data for case in cases),"sampling_ratio":1.0,"all_nodes_edges_features":True,"trainable_parameters":0,"optimizer_updates":0,"classifier_training_run":False,"preflight_is_separate_DEBUG_fixture_check":True,"dependency_audit":"all_existing_feature_column_operator_actions_not_full_Jacobian","source_dir":str(args.source_dir.resolve()),"source_digest":source["code_digest"],"source_reference_digest":manifest["source_content_digest"],"source_and_inputs_preserved":True,"coverage":coverage,"elapsed_seconds":time.perf_counter()-start}
    print("[report] context signal, immediate cancellation and delayed output coupling",flush=True)
    write_report(output,config,graph_rows,resources,completion)
    assert_source_unchanged(source)
    assert_inputs_unchanged(args.source_dir,manifest)
    completion["elapsed_seconds"] = time.perf_counter()-start
    write_json(output/"completion.json",completion)
    print(f"[complete] graphs={len(cases)} results={output.resolve()} channel_rows={table.count}",flush=True)
    print("[scope] fixed coupling audit; no learned classifier training or prediction performance measured",flush=True)


def _workers(value):
    if value=="auto":
        return value
    try:
        result=int(value)
    except (TypeError,ValueError) as error:
        raise argparse.ArgumentTypeError("workers must be auto or a positive integer") from error
    if result<1 or str(result)!=value:
        raise argparse.ArgumentTypeError("workers must be auto or a positive integer")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=("full","debug"),default="full")
    parser.add_argument("--device",default="cuda",help="cuda uses all explicitly allocated visible GPUs")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--workers",type=_workers,default="auto")
    args=parser.parse_args()
    validate_device(read_config(args.config,args.profile),torch.device(args.device))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/"terminal.log").open("x",encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            try:
                run(args,args.output_dir)
            except Exception as error:
                traceback.print_exc()
                write_json(args.output_dir/"failure.json",{"status":"failed","type":type(error).__name__,"message":str(error),"source_inputs_preserved":True})
                raise


if __name__=="__main__":
    main()
