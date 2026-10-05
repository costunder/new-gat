"""Stage A: complete fixed-coefficient information audit. FULL belongs on server."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from ..common import file_sha256, write_json, source_manifest, assert_source_unchanged
from ..verify import run_checks
from ...local_energy_relations.receiver_aggregation.contract import read_config as input_contract
from ...local_energy_relations.receiver_aggregation.data import prepare_cases, assert_inputs_unchanged
from ...local_energy_relations.topology import batch_topologies
from ...wedge_propagation.study import Tee, available_cpus, runtime_resources, synchronize
from .actions import build_actions, feature_fields
from .contract import OPERATOR_PLAN, read_config, expected_target_rows, validate_completion
from .dense import observability_rows
from .numerics import relative
from .recovery import TargetAccumulator, dense_reconstruct, sparse_reconstruct, operator_scale
from .targets import observation_masks


class Results:
    """Lossless streamed raw CSV shards and online descriptive summaries."""
    def __init__(self, root, worker):
        self.root, self.worker = root, worker
        self.tables, self.counts, self.groups = {}, {}, {}

    def write(self, name, rows, metadata):
        for item in rows:
            row = {**metadata, **item}
            if name not in self.tables:
                path = self.root/f"{name}-worker-{self.worker}.csv.gz"
                raw = path.open("xb")
                compressed = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
                import io
                stream = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                self.tables[name] = (stream, raw, writer, path)
                self.counts[name] = 0
            self.tables[name][2].writerow(row)
            self.counts[name] += 1
            if name == "reconstruction":
                keys = (row["family"], row["recipe"], row["operator"], row["repetitions"],
                        row["observation"], row["noise_level"], row["target_kind"])
                record = self.groups.setdefault(keys, {"rows": 0, "successful_rows": 0, "unresolved_rows": 0,
                                                       "defined": 0, "absolute_sum": 0.,
                                                       "relative_sum": 0., "relative_max": None})
                record["rows"] += 1
                if row["solver_converged"] is not True:
                    record["unresolved_rows"] += 1
                    continue
                record["successful_rows"] += 1
                record["absolute_sum"] += row["absolute_error"]
                if row["relative_error"] is not None:
                    record["defined"] += 1
                    record["relative_sum"] += row["relative_error"]
                    record["relative_max"] = max(record["relative_max"] or 0., row["relative_error"])
        for stream, _, _, _ in self.tables.values():
            stream.flush()

    def close(self):
        files = []
        for name, (stream, raw, _, path) in self.tables.items():
            stream.close()
            raw.close()
            files.append({"file": path.name, "rows": self.counts[name], "sha256": file_sha256(path),
                          "bytes": path.stat().st_size, "table": name})
        summaries = []
        names = ("family", "recipe", "operator", "repetitions", "observation", "noise_level", "target_kind")
        for key, value in self.groups.items():
            summaries.append({**dict(zip(names, key, strict=True)), **value,
                              "absolute_mean": value["absolute_sum"]/value["successful_rows"] if value["successful_rows"] else None,
                              "relative_mean": value["relative_sum"]/value["defined"] if value["defined"] else None,
                              "interpretation": "stationarity_converged_complete_vectors_only_descriptive_not_seed_CI"})
        return files, summaries


def _memory_budget(device):
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(.75*free)
    import psutil
    return int(.75*psutil.virtual_memory().available)


def _measured(device, function):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return result, elapsed, peak


def _initial_pair_chunk(topology, reference, budget):
    # Measured resources determine the safe first calibration allocation. This
    # is exact work chunking; eligible pairs and reference features are retained.
    bytes_per_pair = 8*reference.shape[0]*max(1, reference.shape[-1])*12
    return max(1, min(max(1, topology.num_local_edges*topology.num_local_edges), budget//max(1, bytes_per_pair)))


def _dense_group(items, recipe, device, pair_chunk, channel_chunk):
    top = batch_topologies([item[1] for item in items])
    source = torch.cat([feature_fields(item[0], device) for item in items], -2)
    actions = build_actions(top, recipe, source, pair_chunk=pair_chunk, channel_chunk=channel_chunk)
    n, count = items[0][0].num_nodes, len(items)
    identity = torch.eye(n, device=device, dtype=torch.float64).repeat(count, 1)
    matrices = {}
    for name, repeats in OPERATOR_PLAN.items():
        for repeat in repeats:
            batches = actions[name].coefficient_batches
            value = actions[name](identity.expand(batches, -1, -1), repeat)
            matrices[name, repeat] = value.reshape(batches, count, n, n).permute(1, 0, 2, 3).contiguous()
    return matrices


def _dense_calibration(items, recipe, device):
    budget = _memory_budget(device)
    rows, accepted = [], []
    top = items[0][1].to(device)
    reference = feature_fields(items[0][0], device)
    pair_chunk = _initial_pair_chunk(items[0][1], reference, budget)
    matrix = _dense_group(items[:1], recipe, device, pair_chunk, reference.shape[-1])["Q_reference", 1][0]
    for size in sorted(set([1, 2, 4, 8, 16, top.n])):
        if size > top.n:
            continue
        estimated = 8*size*reference.shape[0]*top.n*top.n*32
        if estimated > budget:
            rows.append({"type": "target_chunk", "target_chunk": size, "status": "estimated_memory_exceeds_budget",
                         "estimated_bytes": estimated, "memory_budget_bytes": budget})
            continue
        try:
            print(f"[calibration] {device} dense target_chunk={size} nodes={top.n} scalar_draws={reference.shape[0]}", flush=True)
            def probe():
                # Include direct SVD and actual collision targets in measurement.
                return list(observability_rows(top, matrix, matrix, recipe, kind="one_hop", target_chunk=size))
            result, seconds, peak = _measured(device, probe)
            del result
            row = {"type": "target_chunk", "target_chunk": size, "status": "measured",
                   "seconds": seconds, "target_graphs_per_second": top.n/seconds, "peak_vram_bytes": peak}
            rows.append(row)
            accepted.append(row)
        except torch.cuda.OutOfMemoryError:
            rows.append({"type": "target_chunk", "target_chunk": size, "status": "calibration_OOM_only"})
            gc.collect()
            torch.cuda.empty_cache()
    if not accepted:
        raise RuntimeError("no safe target work chunk; graph/model/features preserved")
    target = max(accepted, key=lambda row: row["target_graphs_per_second"])["target_chunk"]
    accepted = []
    for size in sorted(set([1, 2, 4, 8, 16, len(items)])):
        if size > len(items):
            continue
        try:
            print(f"[calibration] {device} physical_graph_batch={size}", flush=True)
            result, seconds, peak = _measured(device, lambda: _dense_group(items[:size], recipe, device, pair_chunk, 1))
            del result
            row = {"type": "physical_graph_batch", "physical_graph_batch": size, "status": "measured",
                   "seconds": seconds, "graphs_per_second": size/seconds, "peak_vram_bytes": peak}
            rows.append(row)
            accepted.append(row)
        except torch.cuda.OutOfMemoryError:
            rows.append({"type": "physical_graph_batch", "physical_graph_batch": size, "status": "calibration_OOM_only"})
            gc.collect()
            torch.cuda.empty_cache()
    if not accepted:
        raise RuntimeError("complete physical graph does not fit; no data scale fallback")
    count = max(accepted, key=lambda row: row["graphs_per_second"])["physical_graph_batch"]
    return {"physical_graph_batch": count, "target_chunk": target, "channel_chunk": 1,
            "pair_chunk": pair_chunk, "memory_budget_bytes": budget, "measurements": rows}


def _sparse_calibration(top, source, actions, recipe, device):
    budget, rows, accepted = _memory_budget(device), [], []
    width = source.shape[-1]
    candidates = sorted(set([8, 32, 128, 512, width]))
    for size in (1, 2, 4, 8, 16, 32, 64):
        if size > top.n:
            continue
        for channels in candidates:
            if channels > width:
                continue
            # Include q, local divergence, all shared correspondences and CG buffers.
            footprint = 8*size*source.shape[0]*channels*(10*top.n+8*top.num_local_edges+6*top.num_local_nodes
                                                       +8*top.shared_node_pair.numel()+8*top.shared_edge_pair.numel())
            if footprint > budget:
                rows.append({"type": "target_channel_chunk", "target_chunk": size, "channel_chunk": channels,
                             "status": "estimated_memory_exceeds_budget", "estimated_bytes": footprint})
                continue
            try:
                print(f"[calibration] {device} observer_batch={size} channel_chunk={channels}", flush=True)
                def probe():
                    mask = observation_masks(top, "one_hop", 0, size)
                    block = source[..., :channels]
                    recovered, _ = sparse_reconstruct(actions["Q_reference"], 1, block, mask,
                                                      noise_level=0., maximum_iterations=3)
                    accumulator = TargetAccumulator(top, recipe, "one_hop", 0, size, source.shape[0], source)
                    accumulator.add(block, recovered)
                    return recovered
                result, seconds, peak = _measured(device, probe)
                del result
                row = {"type": "target_channel_chunk", "target_chunk": size, "channel_chunk": channels,
                       "status": "measured", "seconds": seconds, "target_channels_per_second": size*channels/seconds,
                       "peak_vram_bytes": peak}
                rows.append(row)
                accepted.append(row)
            except torch.cuda.OutOfMemoryError:
                rows.append({"type": "target_channel_chunk", "target_chunk": size, "channel_chunk": channels,
                             "status": "calibration_OOM_only"})
                gc.collect()
                torch.cuda.empty_cache()
    if not accepted:
        raise RuntimeError("no safe exact target/channel work chunk; all data retained")
    best = max(accepted, key=lambda row: row["target_channels_per_second"])
    # Pair chunks are also measured rather than fixed to a low arbitrary value.
    pairs = actions["Q_reference"].geometry.num_pairs
    safe_pair = actions["Q_reference"].pair_chunk
    pair_accepted = []
    for size in sorted(set([min(max(1, pairs), value) for value in (1024, 16384, 65536, safe_pair, max(1, pairs))])):
        if 8*best["target_chunk"]*best["channel_chunk"]*size*8 > budget:
            rows.append({"type": "pair_chunk", "pair_chunk": size, "status": "estimated_memory_exceeds_budget"})
            continue
        previous = actions["Q_reference"].pair_chunk
        actions["Q_reference"].pair_chunk = size
        try:
            result, seconds, peak = _measured(device, lambda: actions["Q_reference"](
                source[..., :best["channel_chunk"]].expand(best["target_chunk"], *source.shape[1:-1], best["channel_chunk"])))
            del result
            row = {"type": "pair_chunk", "pair_chunk": size, "status": "measured", "seconds": seconds,
                   "pairs_per_second": pairs/seconds, "peak_vram_bytes": peak}
            rows.append(row)
            pair_accepted.append(row)
        except torch.cuda.OutOfMemoryError:
            rows.append({"type": "pair_chunk", "pair_chunk": size, "status": "calibration_OOM_only"})
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            actions["Q_reference"].pair_chunk = previous
    if not pair_accepted:
        raise RuntimeError("no safe pair chunk measured; no topology cap")
    pair = max(pair_accepted, key=lambda row: row["pairs_per_second"])["pair_chunk"]
    for action in actions.values():
        action.pair_chunk = pair
    return {"physical_graph_batch": 1, "physical_graph_batch_reason": "one_complete_citation_graph_per_independent_GPU_job",
            "target_chunk": best["target_chunk"], "channel_chunk": best["channel_chunk"], "pair_chunk": pair,
            "memory_budget_bytes": budget, "measurements": rows}


def _ranges(n, kind, size):
    return [(0, n)] if kind == "full" else [(first, min(n, first+size)) for first in range(0, n, size)]


def _base_repeat(name, repetition):
    return repetition if name in ("L0", "Ld", "smooth_L0", "smooth_Ld") else 1


def _baseline_name(name):
    if name.startswith("Q_"):
        return "Q_diag"
    if name.startswith("P_"):
        return "P_diag"
    if name.startswith("copy_"):
        return "copy_off"
    return "Ld" if name in ("L0", "Ld") else "smooth_Ld"


def _solve_aggregate(shape, source):
    return {"channels": 0, "converged": torch.zeros(shape, device=source.device, dtype=torch.long),
            "stationarity_converged": torch.zeros(shape, device=source.device, dtype=torch.long),
            "observation_within_tolerance": torch.zeros(shape, device=source.device, dtype=torch.long),
            "breakdown": torch.zeros(shape, device=source.device, dtype=torch.long),
            "iterations": torch.zeros(shape, device=source.device, dtype=torch.long),
            "maximum_relative_residual": source.new_zeros(shape), "rhs_square": source.new_zeros(shape),
            "maximum_observation_relative_residual": source.new_zeros(shape),
            "normal_residual_square": source.new_zeros(shape), "normal_rhs_square": source.new_zeros(shape),
            "observation_residual_square": source.new_zeros(shape), "observation_rhs_square": source.new_zeros(shape),
            "noise_square": source.new_zeros(shape), "undefined_rhs_channels": torch.zeros(shape, device=source.device, dtype=torch.long),
            "undefined_observation_rhs_channels": torch.zeros(shape, device=source.device, dtype=torch.long)}


def _update_solver(record, status, batch, realizations, channels):
    record["channels"] += channels
    for name in ("converged", "stationarity_converged", "observation_within_tolerance", "breakdown"):
        record[name] += status[name].reshape(batch, realizations, channels).sum(-1)
    record["undefined_rhs_channels"] += (~status["rhs_nonzero"]).reshape(batch, realizations, channels).sum(-1)
    record["undefined_observation_rhs_channels"] += (~status["observation_rhs_nonzero"]).reshape(batch, realizations, channels).sum(-1)
    record["iterations"] = torch.maximum(record["iterations"], status["iterations"].reshape(batch, realizations, channels).amax(-1))
    record["maximum_relative_residual"] = torch.maximum(record["maximum_relative_residual"],
                                                        status["relative_residual"].reshape(batch, realizations, channels).amax(-1))
    record["maximum_observation_relative_residual"] = torch.maximum(record["maximum_observation_relative_residual"],
        status["observation_relative_residual"].reshape(batch, realizations, channels).amax(-1))
    for name in ("normal_residual", "normal_rhs", "observation_residual", "observation_rhs"):
        record[name+"_square"] += status[name+"_norm"].reshape(batch, realizations, channels).square().sum(-1)
    record["rhs_square"] += status["clean_rhs_norm"].reshape(batch, realizations, channels).square().sum(-1)
    record["noise_square"] += status["noise_norm"].reshape(batch, realizations, channels).square().sum(-1)


def _solver_rows(record, kind, first, observed_counts, scale, method):
    host = {name: value.cpu().numpy() for name, value in record.items() if isinstance(value, torch.Tensor)}
    for observer in range(record["converged"].shape[0]):
        for realization in range(record["converged"].shape[1]):
            yield {"observer": None if kind == "full" else first+observer, "reference_realization": realization,
                   "observed_nodes": observed_counts[observer], "channels_processed": record["channels"],
                   "solver_converged_channels": int(host["converged"][observer, realization]),
                   "stationarity_converged_channels": int(host["stationarity_converged"][observer, realization]),
                   "observation_residual_within_tolerance_channels": int(host["observation_within_tolerance"][observer, realization]),
                   "breakdown_channels": int(host["breakdown"][observer, realization]),
                   "normal_rhs_zero_channels": int(host["undefined_rhs_channels"][observer, realization]),
                   "observation_rhs_zero_channels": int(host["undefined_observation_rhs_channels"][observer, realization]),
                   "maximum_iterations_used": int(host["iterations"][observer, realization]),
                   "maximum_true_normal_relative_residual": float(host["maximum_relative_residual"][observer, realization]),
                   "maximum_true_observation_relative_residual": float(host["maximum_observation_relative_residual"][observer, realization]),
                   "normal_residual_norm": math.sqrt(float(host["normal_residual_square"][observer, realization])),
                   "normal_rhs_norm": math.sqrt(float(host["normal_rhs_square"][observer, realization])),
                   "observation_residual_norm": math.sqrt(float(host["observation_residual_square"][observer, realization])),
                   "observation_rhs_norm": math.sqrt(float(host["observation_rhs_square"][observer, realization])),
                   "clean_rhs_norm": math.sqrt(float(host["rhs_square"][observer, realization])),
                   "noise_norm": math.sqrt(float(host["noise_square"][observer, realization])),
                   "numerical_operator_scale": scale, "method": method, "exact_rank_claim": False}


def _case_views(case, top, recipe, device, config, calibration, output, matrices=None, actions=None):
    source = feature_fields(case, device)
    realizations = source.shape[0]
    covered, failures, observation_residuals, unresolved_targets, solver_rows = 0, 0, 0, 0, 0
    for name, repeats in OPERATOR_PLAN.items():
        for repeat in repeats:
            baseline = _baseline_name(name)
            base_repeat = _base_repeat(name, repeat)
            # Synthetic dense matrices are physically batched during construction.
            matrix = matrices[name, repeat] if matrices is not None else None
            old = matrices[baseline, base_repeat] if matrices is not None else None
            for kind in config["observations"]:
                metadata = {"graph_id": case.graph_id, "family": case.family, "actual_data": case.actual_data,
                            "recipe": recipe, "operator": name, "repetitions": repeat, "observation": kind}
                if matrix is not None:
                    output.write("certificates", observability_rows(top, matrix, old, recipe, kind=kind,
                                 target_chunk=calibration["target_chunk"], rank_tolerance=config["rank_relative_tolerance"]), metadata)
                for first, last in _ranges(top.n, kind, calibration["target_chunk"]):
                    mask = observation_masks(top, kind, first, last)
                    batch = mask.shape[0]
                    observed_counts = mask.sum(-1).to(torch.long).cpu().tolist()
                    maximum_iterations = config["solver"]["maximum_iterations_factor"]*max(observed_counts)
                    for noise in config["noise_levels"]:
                        accumulator = TargetAccumulator(top, recipe, kind, first, last, realizations, source)
                        status_sum = _solve_aggregate((batch, realizations), source)
                        reference_status_sum = _solve_aggregate((batch, realizations), source) if matrix is None and noise == 0 else None
                        near = source.new_zeros((batch, realizations, 6)) if matrix is None and noise == 0 else None
                        for channel in range(0, source.shape[-1], calibration["channel_chunk"]):
                            block = source[..., channel:channel+calibration["channel_chunk"]]
                            if matrix is not None:
                                recovered, status = dense_reconstruct(matrix, block, mask, noise,
                                    tolerance=config["rank_relative_tolerance"], solver_tolerance=config["solver"]["tolerance"],
                                    channel_offset=channel, target_offset=first)
                                scale = 1.
                            else:
                                action = actions[name]
                                recovered, status = sparse_reconstruct(action, repeat, block, mask, noise_level=noise,
                                    tolerance=config["solver"]["tolerance"], maximum_iterations=maximum_iterations,
                                    channel_offset=channel, target_offset=first,
                                    progress=lambda used, maximum: print(
                                        f"[solver progress] {case.graph_id} C={recipe} {name}^{repeat} {kind} targets={first}:{last} channels={channel}:{channel+block.shape[-1]} noise={noise:g} CG={used}/{maximum}", flush=True))
                                scale = float(status["numerical_operator_scale"].cpu())
                            accumulator.add(block, recovered)
                            _update_solver(status_sum, status, batch, realizations, block.shape[-1])
                            if near is not None:
                                reference_action = actions[baseline]
                                delta = block[None]-recovered
                                observed_delta = mask[:, None, :, None]*actions[name](delta, repeat)/operator_scale(actions[name], repeat)
                                baseline_delta = mask[:, None, :, None]*reference_action(delta, base_repeat)/operator_scale(reference_action, base_repeat)
                                if name == baseline and repeat == base_repeat:
                                    reverse_delta = delta
                                    base_status = status
                                else:
                                    restored_base, base_status = sparse_reconstruct(reference_action, base_repeat, block, mask,
                                        noise_level=0., tolerance=config["solver"]["tolerance"], maximum_iterations=maximum_iterations,
                                        channel_offset=channel, target_offset=first,
                                        progress=lambda used, maximum: print(
                                            f"[solver progress] paired_reference {case.graph_id} {baseline}^{base_repeat} {kind} targets={first}:{last} CG={used}/{maximum}", flush=True))
                                    reverse_delta = block[None]-restored_base
                                _update_solver(reference_status_sum, base_status, batch, realizations, block.shape[-1])
                                reverse_visible = mask[:, None, :, None]*actions[name](reverse_delta, repeat)/operator_scale(actions[name], repeat)
                                reverse_residual = mask[:, None, :, None]*reference_action(reverse_delta, base_repeat)/operator_scale(reference_action, base_repeat)
                                near[..., 0] += delta.square().sum((-2, -1))
                                near[..., 1] += observed_delta.square().sum((-2, -1))
                                near[..., 2] += baseline_delta.square().sum((-2, -1))
                                near[..., 3] += reverse_visible.square().sum((-2, -1))
                                near[..., 4] += block.square().sum((-2, -1))[None]
                                near[..., 5] += reverse_residual.square().sum((-2, -1))
                        details = {**metadata, "noise_level": noise, "observer_start": first,
                                   "observer_stop": last, "channels_processed": source.shape[-1],
                                   "method": "direct_SVD_least_squares_at_declared_rank" if matrix is not None else "primal_CG_minimum_norm_least_squares_if_converged"}
                        output.write("reconstruction", accumulator.rows(status_sum), details)
                        method = details["method"]
                        raw_solver = list(_solver_rows(status_sum, kind, first, observed_counts, scale, method))
                        output.write("solver", raw_solver, {**metadata, "noise_level": noise})
                        failures += sum(row["channels_processed"]-row["solver_converged_channels"] for row in raw_solver)
                        observation_residuals += sum(row["channels_processed"]-row["observation_residual_within_tolerance_channels"] for row in raw_solver)
                        status_host = status_sum["converged"].cpu().numpy()
                        for target_kind, owners in accumulator.observers.items():
                            unresolved_targets += int((status_host[owners.cpu().numpy()] != status_sum["channels"]).sum())
                        solver_rows += len(raw_solver)
                        if near is not None:
                            data = near.sqrt().cpu().numpy()
                            forward_converged = status_host == status_sum["channels"]
                            backward_converged = reference_status_sum["converged"].cpu().numpy() == reference_status_sum["channels"]
                            rows = [{"observer": None if kind == "full" else first+i, "reference_realization": r,
                                     "input_difference_norm": float(data[i, r, 0]), "current_observation_difference_norm": float(data[i, r, 1]),
                                     "baseline_visible_current_near_hidden": float(data[i, r, 2]),
                                     "current_visible_baseline_near_hidden": float(data[i, r, 3]), "input_norm": float(data[i, r, 4]),
                                     "baseline_observation_difference_norm": float(data[i, r, 5]),
                                     "current_solver_converged": bool(forward_converged[i, r]),
                                     "baseline_solver_converged": bool(backward_converged[i, r]),
                                     "both_solvers_converged": bool(forward_converged[i, r] and backward_converged[i, r]),
                                     "method": "actual_input_numeric_near_collision_residuals_not_exact_nullspace_proof",
                                     "exact_rank_claim": False}
                                    for i in range(batch) for r in range(realizations)]
                            output.write("near_collision", rows, metadata)
                        covered += (2*(last-first)+3*int(((top.pair_centers[0] >= first)&(top.pair_centers[0] < last)).sum().cpu()))*realizations
                    if time.perf_counter()-output.last_progress >= 20:
                        print(f"[audit progress] {case.graph_id} C={recipe} {name}^{repeat} {kind} targets={last}/{top.n} raw={output.counts}", flush=True)
                        output.last_progress = time.perf_counter()
            print(f"[view complete] {case.graph_id} C={recipe} {name}^{repeat} all nodes/channels", flush=True)
    expected = expected_target_rows(top, realizations)*sum(map(len, OPERATOR_PLAN.values()))*3*len(config["noise_levels"])
    if covered != expected:
        raise RuntimeError(f"target coverage differs for {case.graph_id}: {covered} != {expected}")
    return {"graph_id": case.graph_id, "recipe": recipe, "target_rows": covered,
            "all_nodes_edges_channels": True, "solver_rows": solver_rows,
            "stationarity_unresolved_channel_solves": failures,
            "observation_residual_outside_tolerance_channel_solves": observation_residuals,
            "unresolved_target_rows": unresolved_targets,
            "scalar_realizations": realizations, "channels": source.shape[-1]}


def _worker(worker, device_name, jobs, root, config):
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = Results(root, worker)
    output.last_progress = time.perf_counter()
    coverage, calibration_records = [], []
    try:
        for kind, recipe, items in jobs:
            print(f"[job start] {device} {kind} C={recipe} graphs={[case.graph_id for case, _ in items]}", flush=True)
            if kind == "dense":
                calibration = _dense_calibration(items, recipe, device)
                calibration_records.append({"worker": worker, "device": str(device), "recipe": recipe,
                                            "graphs": [item[0].graph_id for item in items], **calibration})
                size = calibration["physical_graph_batch"]
                for start in range(0, len(items), size):
                    packed = items[start:start+size]
                    matrices = _dense_group(packed, recipe, device, calibration["pair_chunk"], 1)
                    for graph, (case, top) in enumerate(packed):
                        isolated = {key: value[graph] for key, value in matrices.items()}
                        coverage.append(_case_views(case, top.to(device), recipe, device, config, calibration, output, isolated))
            else:
                case, topology = items[0]
                source = feature_fields(case, device)
                chunk = _initial_pair_chunk(topology, source, _memory_budget(device))
                actions = build_actions(topology, recipe, source, pair_chunk=chunk,
                                        channel_chunk=min(source.shape[-1], 128))
                top = topology.to(device)
                calibration = _sparse_calibration(top, source, actions, recipe, device)
                calibration_records.append({"worker": worker, "device": str(device), "recipe": recipe,
                                            "graphs": [case.graph_id], **calibration})
                coverage.append(_case_views(case, top, recipe, device, config, calibration, output, actions=actions))
        files, summaries = output.close()
        return {"worker": worker, "device": str(device), "coverage": coverage, "calibration": calibration_records,
                "files": files, "summaries": summaries}
    except Exception:
        # Preserve completed raw records on a failure; never terminate a session.
        output.close()
        raise


def _jobs(cases, topologies, recipes, devices):
    from collections import defaultdict
    buckets, jobs = defaultdict(list), []
    for case, top in zip(cases, topologies, strict=True):
        if case.feature_mode == "independent_scalar_columns":
            buckets[case.num_nodes].append((case, top))
        else:
            for recipe in recipes:
                jobs.append(("sparse", recipe, [(case, top)]))
    for _, items in sorted(buckets.items()):
        for recipe in recipes:
            jobs.append(("dense", recipe, items))
    # If the allocation exceeds size buckets, split independent graph groups;
    # never claim use of a worker that received no actual science job.
    while len(jobs) < len(devices):
        candidates = [index for index, job in enumerate(jobs) if job[0] == "dense" and len(job[2]) > 1]
        if not candidates:
            raise RuntimeError("allocated GPU count exceeds independent complete graph/recipe jobs; explicit target-work distribution is required")
        chosen = max(candidates, key=lambda index: len(jobs[index][2]))
        kind, recipe, items = jobs.pop(chosen)
        middle = len(items)//2
        jobs.extend([(kind, recipe, items[:middle]), (kind, recipe, items[middle:])])
    def footprint(job):
        return sum((top.num_local_edges+top.num_local_nodes+
                    (top.shared_node_pair.numel() if isinstance(top.shared_node_pair, torch.Tensor) else top.shared_node_pair.size))
                   *case.num_features for case, top in job[2])
    queues, sizes = [[] for _ in devices], [0 for _ in devices]
    for job in sorted(jobs, key=footprint, reverse=True):
        target = min(range(len(devices)), key=lambda i: sizes[i])
        queues[target].append(job)
        sizes[target] += footprint(job)
    return queues


def run(args):
    args._phase = "configuration"
    config = read_config(args.profile, args.config)
    if args.trained_source_dir is not None:
        raise ValueError("trained-model audit requires selected checkpoints and exact frozen-state manifests; these are unavailable. Fixed audit never substitutes fake trained states")
    if args.profile == "full" and (sys.platform != "linux" or args.device != "cuda" or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()):
        raise ValueError("FULL actual-data audit must run on the Linux CUDA server; use explicit DEBUG locally")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    elif args.device == "cpu":
        devices = ["cpu"]
    else:
        raise ValueError("device must be cpu or cuda; cuda schedules all visible allocated GPUs")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    args._created_output = root
    args._phase = "hardware_preflight"
    with (root/"terminal.log").open("x", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        started = time.perf_counter()
        hardware = [runtime_resources(torch.device(device)) for device in devices]
        workers = available_cpus(hardware[0])
        torch.set_num_threads(max(1, workers//len(devices)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        manifest = source_manifest()
        write_json(root/"config.json", config)
        write_json(root/"source_manifest.json", manifest)
        write_json(root/"hardware.json", {"devices": hardware, "gpu_count_used": len(devices) if args.device == "cuda" else 0,
                                        "cpu_threads": torch.get_num_threads(), "data_root": args.data_root,
                                        "input_policy": "entire_hash_verified_original_snapshots_no_redownload"})
        print(f"[start] Stage A profile={args.profile} float64 allocated_devices={devices} operators=21 C=2 sampling=1.0 trainable=0 epochs=N/A", flush=True)
        args._phase = "independent_DEBUG_algebra_gate"
        checks = run_checks(devices[0])
        write_json(root/"math_checks.json", checks)
        args._phase = "validate_and_cache_entire_source"
        cases, tops, inputs = prepare_cases(input_contract(profile=args.profile), args.source_dir, root)
        if len(cases) != config["source_graphs"]:
            raise RuntimeError("source graph coverage differs")
        queues = _jobs(cases, tops, config["recipes"], devices)
        scheduling = [{"worker": index, "device": device, "jobs": [{"kind": kind, "recipe": recipe,
                       "graphs": [case.graph_id for case, _ in items]} for kind, recipe, items in queue]}
                      for index, (device, queue) in enumerate(zip(devices, queues, strict=True))]
        write_json(root/"scheduling.json", scheduling)
        print(f"[data] graphs={len(cases)} all original nodes/edges/channels; GPU workers={len(devices)}", flush=True)
        args._phase = "all_allocated_GPU_operator_audit"
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(_worker, index, device, queue, root, config)
                       for index, (device, queue) in enumerate(zip(devices, queues, strict=True))]
            results = [future.result() for future in futures]
        args._phase = "coverage_and_provenance_validation"
        coverage = [row for result in results for row in result["coverage"]]
        if {(row["graph_id"], row["recipe"]) for row in coverage} != {(case.graph_id, recipe) for case in cases for recipe in config["recipes"]} or len(coverage) != 2*len(cases):
            raise RuntimeError("missing/duplicated graph-recipe coverage")
        assert_inputs_unchanged(args.source_dir, inputs)
        assert_source_unchanged(manifest)
        write_json(root/"coverage.json", coverage)
        write_json(root/"calibration.json", [row for result in results for row in result["calibration"]])
        write_json(root/"raw_manifest.json", [row for result in results for row in result["files"]])
        write_json(root/"summary.json", [row for result in results for row in result["summaries"]])
        from .report import write_report
        args._phase = "report_and_completion"
        write_report(root, config, coverage, results, checks)
        complete = {"completed": True, "status": "complete", "profile": args.profile, "experiment": config["experiment"],
                    "graphs": len(cases), "math_checks_passed": checks["passed"], "source_digest": manifest["code_digest"],
                    "input_digest": inputs["source_content_digest"], "all_original_nodes_edges_channels": True,
                    "sampling_ratio": 1., "operator_views_per_recipe": 21, "recipes": config["recipes"],
                    "optimizer_updates": 0, "classifier_training_run": False, "trained_model_audit_run": False,
                    "trained_model_audit_scope": "selected_checkpoint_states_not_supplied",
                    "least_squares_solver": "primal_CG_zero_start_or_direct_SVD",
                    "convergence_criterion": "true_relative_normal_stationarity_residual_and_no_breakdown",
                    "successful_summary_scope": "all_constituent_channels_solver_converged_only",
                    "stationarity_unresolved_channel_solves": sum(row["stationarity_unresolved_channel_solves"] for row in coverage),
                    "observation_residual_outside_tolerance_channel_solves": sum(row["observation_residual_outside_tolerance_channel_solves"] for row in coverage),
                    "unresolved_target_rows": sum(row["unresolved_target_rows"] for row in coverage),
                    "citation_exact_rank_claim": False, "gpu_count_used": len(devices) if args.device == "cuda" else 0,
                    "wall_seconds": time.perf_counter()-started,
                    "debug": args.profile == "debug", "actual_citation_data": args.profile == "full"}
        validate_completion(complete, config)
        write_json(root/"completion.json", complete)
        print(f"[complete] Stage A graphs={len(cases)} results={root} seconds={complete['wall_seconds']:.1f} stationarity_unresolved={complete['stationarity_unresolved_channel_solves']} observation_residuals={complete['observation_residual_outside_tolerance_channel_solves']}", flush=True)
        print("[scope] fixed coefficients; no learned model/classifier training or trained checkpoint-state audit", flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--data-root", default=None, help="compatibility metadata; data read only from hash-verified source snapshots")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trained-source-dir", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        import traceback
        created = getattr(args, "_created_output", None)
        if created is not None and not (created/"failure.json").exists():
            write_json(created/"failure.json", {"completed": False, "type": type(error).__name__, "message": str(error),
                       "failed_phase": getattr(args, "_phase", "unknown"), "traceback": traceback.format_exc(),
                       "prior_inputs_results_preserved": True, "session_terminated": False})
        print(f"[failed] {type(error).__name__}: {error}; prior inputs/results preserved; inspect new terminal.log/failure.json", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
