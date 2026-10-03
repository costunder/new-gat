"""Complete fixed local quadratic/bilinear audit; full runs belong on the server."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np
import psutil
import torch

from ..wedge_propagation.operators import laplacian_apply
from ..wedge_propagation.study import Tee, available_cpus, runtime_resources, synchronize
from .contract import (
    assert_source_unchanged,
    file_sha256,
    read_config,
    source_manifest,
    write_json,
)
from .data import assert_inputs_unchanged, prepare_cases
from .operators import (
    aggregate_squares,
    incidence,
    incidence_transpose,
    local_weight,
    recover_flows,
    relation_terms,
    transfer_bookkeeping,
)
from .report import write_report
from .topology import CORRESPONDENCE_KINDS, batch_topologies, build_topology


class Table:
    """Stream complete raw records without retaining millions of dictionaries."""

    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8", newline="")
        self.writer = None
        self.count = 0
        self.label = Path(path).stem
        self.last_progress = time.perf_counter()

    def write(self, rows):
        for row in rows:
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.count += 1
            if self.count % 10000 == 0 and time.perf_counter() - self.last_progress >= 20:
                print(f"[export] {self.label} rows={self.count}", flush=True)
                self.last_progress = time.perf_counter()
        self.stream.flush()

    def close(self):
        self.stream.close()


def _prepare_topology(arguments):
    n, edges = arguments
    return build_topology(n, edges)


def _parallel_topologies(arguments, workers):
    if workers == 1:
        return [_prepare_topology(item) for item in arguments]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_prepare_topology, arguments))


def prepare_topologies(cases, config, hardware, output, resource_rows):
    arguments = [(case.num_nodes, case.edges.numpy()) for case in cases]
    synthetic_count = config["synthetic"]["graphs"]
    largest = max(case.num_nodes for case in cases[:synthetic_count])
    probe_indices = [
        index
        for index, case in enumerate(cases)
        if index >= synthetic_count or case.num_nodes == largest
    ]
    probe = [arguments[index] for index in probe_indices]
    requested = config["runtime"]["cpu_workers"]
    candidates = (
        [requested]
        if requested != "auto"
        else sorted(
            {
                min(available_cpus(hardware), k)
                for k in config["runtime"]["preprocessing_worker_candidates"]
            }
        )
    )
    measured, cached, best_seconds = [], None, float("inf")
    for workers in candidates:
        start = time.perf_counter()
        prepared_probe = _parallel_topologies(probe, workers)
        seconds = time.perf_counter() - start
        row = {
            "type": "topology_worker_calibration",
            "workers": workers,
            "graphs": len(probe),
            "seconds": seconds,
            "graphs_per_second": len(probe) / seconds,
            "graph_ids": [cases[index].graph_id for index in probe_indices],
            "scope": "largest_complete_synthetic_graphs_and_all_complete_citation_graphs",
        }
        measured.append(row)
        if seconds < best_seconds:
            cached, best_seconds = prepared_probe, seconds
        print(
            f"[CPU topology calibration] workers={workers} graphs={len(probe)} "
            f"seconds={seconds:.3f}",
            flush=True,
        )
    resource_rows.extend(measured)
    selected = max(measured, key=lambda row: row["graphs_per_second"])["workers"]
    print(
        f"[topology] all {len(cases)} graphs; processes={selected}; "
        "every induced ego and adjacent pair",
        flush=True,
    )
    start = time.perf_counter()
    prepared = dict(zip(probe_indices, cached, strict=True))
    remainder = [index for index in range(len(cases)) if index not in prepared]
    prepared.update(
        zip(
            remainder,
            _parallel_topologies([arguments[index] for index in remainder], selected),
            strict=True,
        )
    )
    topologies = [prepared[index] for index in range(len(cases))]
    folder = output / "topology"
    folder.mkdir()
    records = []
    for case, top in zip(cases, topologies, strict=True):
        path = folder / f"{case.graph_id}.npz"
        with path.open("xb") as stream:
            payload = {
                field.name: getattr(top, field.name)
                for field in fields(top)
                if field.name != "correspondence_offsets"
            }
            payload.update(
                {
                    f"{kind}_offsets": np.asarray(offsets, dtype=np.int64)
                    for kind, offsets in zip(
                        CORRESPONDENCE_KINDS, top.correspondence_offsets, strict=True
                    )
                }
            )
            np.savez_compressed(stream, **payload)
        records.append(
            {
                "graph_id": case.graph_id,
                "file": f"topology/{path.name}",
                "sha256": file_sha256(path),
                "nodes": top.n,
                "physical_edges": top.num_edges,
                "local_node_occurrences": top.num_local_nodes,
                "local_edge_occurrences": top.num_local_edges,
                "directed_center_pairs": top.num_pairs,
            }
        )
        print(
            f"[topology ready] {case.graph_id} nodes={top.n} edges={top.num_edges} "
            f"local_edges={top.num_local_edges} pairs={top.num_pairs}",
            flush=True,
        )
    write_json(
        output / "topology_manifest.json",
        {"schema": "induced-one-hop-local-correspondence-v1", "graphs": records},
    )
    resource_rows.append(
        {
            "type": "topology_preparation",
            "workers": selected,
            "graphs": len(cases),
            "seconds": time.perf_counter() - start,
            "rss_bytes": psutil.Process().memory_info().rss,
        }
    )
    return topologies, records


def reference_states(top, x):
    """Two fixed physical-Laplacian steps in unchanged original feature coordinates."""
    degree = torch.bincount(top.edges.flatten(), minlength=top.n).to(x.dtype)
    graph_index = torch.bucketize(
        torch.arange(top.n, device=x.device), top.graph_node_offsets[1:], right=True
    )
    maximum = degree.new_zeros(top.num_graphs).scatter_reduce_(
        0, graph_index, degree, reduce="amax"
    )
    graph_tau = torch.where(maximum > 0, 0.5 / maximum.clamp_min(1), 0)
    tau = graph_tau[graph_index]
    result = [x]
    for _ in range(2):
        result.append(result[-1] - tau[:, None] * laplacian_apply(top.edges, result[-1]))
    return result


def _scatter(values, index, n):
    result = values.new_zeros((n, *values.shape[1:]))
    result.index_add_(0, index, values)
    return result


def _array(value):
    return value.detach().cpu().numpy()


def _assembly(top, h, c, g):
    multiplicity = torch.bincount(top.local_edge_global, minlength=top.num_edges).to(h.dtype)
    weights = c / multiplicity[top.local_edge_global]
    average_c = _scatter(weights, top.local_edge_global, top.num_edges)
    physical_g = h[top.edges[1]] - h[top.edges[0]]
    physical_energy = physical_g.square()
    local = c[:, None] * g.square()
    edge_graph = torch.bucketize(
        torch.arange(top.num_edges, device=h.device), top.graph_edge_offsets[1:], right=True
    )
    return {
        "energy_raw_local": _scatter(local, edge_graph[top.local_edge_global], top.num_graphs),
        "energy_overlap_corrected": _scatter(
            weights[:, None] * g.square(), edge_graph[top.local_edge_global], top.num_graphs
        ),
        "energy_physical_average": _scatter(
            average_c[:, None] * physical_energy, edge_graph, top.num_graphs
        ),
        "energy_physical_unit": _scatter(physical_energy, edge_graph, top.num_graphs),
    }


@torch.no_grad()
def compute_chunk(top, x, config, weight_names=None, relation_batch=None, progress_label=None):
    """GPU arithmetic across disjoint locals and all supplied channels; no per-case kernels."""
    states = reference_states(top, x)
    result = {}
    for name in weight_names or config["weights"]:
        c = local_weight(top, name, dtype=x.dtype)
        flows, local, transfer, assembly = [], [], [], []
        for stage, h in enumerate(states):
            began = time.perf_counter()
            if progress_label:
                print(f"[compute] {progress_label} C={name} H{stage} recovery/transfer", flush=True)

            def solver_progress(kind, iteration, maximum, elapsed, name=name, stage=stage):
                print(
                    f"[CG] {progress_label} C={name} H{stage} {kind} "
                    f"iteration={iteration}/{maximum} seconds={elapsed:.1f}",
                    flush=True,
                )

            g = incidence(top, h[top.local_node_global])
            q = c[:, None] * g
            d = incidence_transpose(top, q)
            recovered = recover_flows(
                top,
                q,
                c,
                tol=config["solver"]["tolerance"],
                iteration_multiplier=config["solver"]["maximum_iterations_factor"],
                progress=solver_progress if progress_label else None,
            )
            cut_info, recon_info = recovered["cut_solver"], recovered["reconstruction_solver"]
            metrics = {
                "energy": aggregate_squares(top, g, space="edge", weights=c),
                "flow_norm_sq": aggregate_squares(top, q, space="edge"),
                "divergence_norm_sq": aggregate_squares(top, d, space="node"),
                "cycle_norm_sq": aggregate_squares(top, recovered["q_cycle"], space="edge"),
                "reconstruction_error_sq": aggregate_squares(
                    top, recovered["q_recon"] - q, space="edge"
                ),
                "cut_solver_relative_residual": cut_info["relative_residual"],
                "reconstruction_solver_relative_residual": recon_info["relative_residual"],
                "cut_solver_iterations": cut_info["iterations"],
                "reconstruction_solver_iterations": recon_info["iterations"],
            }
            local.append({key: _array(value) for key, value in metrics.items()})
            transfer.append(
                {
                    key: _array(value)
                    for key, value in transfer_bookkeeping(
                        top, q, relation_batch=relation_batch
                    ).items()
                }
            )
            assembly.append({key: _array(value) for key, value in _assembly(top, h, c, g).items()})
            flows.append(q)
            del recovered, d, g, metrics
            if progress_label:
                print(
                    f"[compute done] {progress_label} C={name} H{stage} "
                    f"seconds={time.perf_counter() - began:.3f}",
                    flush=True,
                )
        relations = {}
        for left, right in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2)):
            relations[left, right] = {
                key: _array(value)
                for key, value in relation_terms(
                    top, flows[left], flows[right], relation_batch=relation_batch
                ).items()
            }
        temporal = {
            (left, left + 1): _array(
                _scatter(flows[left] * flows[left + 1], top.local_edge_center, top.n)
            )
            for left in (0, 1)
        }
        result[name] = {
            "local": local,
            "transfer": transfer,
            "assembly": assembly,
            "relations": relations,
            "temporal": temporal,
        }
    return result


def _mapping_size(top, relation_batch):
    largest = 0
    for offsets in top.correspondence_offsets:
        for first in range(0, top.num_pairs, relation_batch):
            largest = max(
                largest, offsets[min(first + relation_batch, top.num_pairs)] - offsets[first]
            )
    return largest


def _estimated_bytes(top, channels, relation_batch=None):
    relation_batch = max(1, top.num_pairs) if relation_batch is None else relation_batch
    mapping = _mapping_size(top, relation_batch)
    index_bytes = sum(
        np.asarray(getattr(top, field.name)).nbytes
        for field in fields(top)
        if field.name not in ("n", "correspondence_offsets")
    )
    # Conservative simultaneous solver, q state, correspondence gathers and reduction buffers.
    return int(
        index_bytes
        + 8
        * channels
        * (
            18 * top.num_local_nodes
            + 14 * top.num_local_edges
            + 10 * top.num_pairs
            + 8 * mapping
            + 10 * top.n
        )
    )


def _memory(device):
    if device.type != "cuda":
        return {"peak_vram_bytes": None, "steady_vram_bytes": None, "free_vram_bytes": None}
    return {
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "steady_vram_bytes": torch.cuda.memory_allocated(device),
        "free_vram_bytes": torch.cuda.mem_get_info(device)[0],
    }


def calibrate(cases, topologies, config, device, resource_rows, mode):
    """Measure exact graph batches/channel chunks; final coverage is never changed."""
    scalar = mode == "graphs"
    requested = config["runtime"]["channel_chunk" if not scalar else "physical_graph_batch"]
    upper = len(cases) if scalar else cases[0].num_features
    candidates = sorted(
        {
            min(upper, k)
            for k in (
                [4, 8, 16, 32, 64, 128, upper] if scalar else [16, 64, 128, 256, 512, 1024, upper]
            )
        }
    )
    if requested != "auto":
        if requested > upper:
            raise ValueError(
                f"requested {mode} size {requested} exceeds complete input dimension {upper}"
            )
        candidates = [requested]
    rows = []
    for candidate in candidates:
        cpu_top = batch_topologies(topologies[:candidate]) if scalar else topologies[0]
        channels = cases[0].num_features if scalar else candidate
        requested_pairs = config["runtime"]["relation_batch"]
        # Measure multiple exact pair chunks to resolve sparse-overlap temporary cost.
        pair_candidates = (
            sorted(
                {min(max(1, cpu_top.num_pairs), k) for k in (256, 4096, max(1, cpu_top.num_pairs))}
            )
            if requested_pairs == "auto"
            else [requested_pairs]
        )
        for relation_batch in pair_candidates:
            _calibrate_candidate(
                cases,
                config,
                device,
                scalar,
                candidate,
                channels,
                cpu_top,
                relation_batch,
                rows,
                mode,
            )
    resource_rows.extend(rows)
    measured = [row for row in rows if row["status"] == "measured"]
    if not measured:
        raise RuntimeError(
            f"no measured {mode} allocation fits resource contract; inputs preserved"
        )
    best = max(measured, key=lambda row: row["feature_cells_per_second"])
    selected = best["candidate"]
    print(
        f"[selected] {mode}={selected} directed_pair_chunk={best['relation_batch']}; "
        "all original graphs/nodes/edges/channels retained",
        flush=True,
    )
    return selected, best["relation_batch"]


def _calibrate_candidate(
    cases, config, device, scalar, candidate, channels, cpu_top, relation_batch, rows, mode
):
    if device.type == "cuda":
        # Release this process's inactive probe cache before querying real free VRAM.
        torch.cuda.empty_cache()
    estimated = _estimated_bytes(cpu_top, channels, relation_batch)
    free = (
        torch.cuda.mem_get_info(device)[0]
        if device.type == "cuda"
        else psutil.virtual_memory().available
    )
    row = {
        "type": "graph_batch_calibration" if scalar else "channel_chunk_calibration",
        "graph_id": "synthetic" if scalar else cases[0].graph_id,
        "candidate": candidate,
        "relation_batch": relation_batch,
        "estimated_working_bytes": estimated,
        "available_bytes": free,
    }
    if estimated > free * config["runtime"]["gpu_memory_safety_fraction"]:
        row.update(status="skipped_memory_estimate", seconds=None, feature_cells_per_second=None)
        rows.append(row)
        print(
            f"[calibration] {mode}={candidate} skipped estimated_memory={estimated} free={free}",
            flush=True,
        )
        return
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    top = x_cpu = x = computed = None
    try:
        top = cpu_top.to(device)
        x_cpu = (
            torch.cat([case.x for case in cases[:candidate]], dim=0)
            if scalar
            else cases[0].x[:, :candidate].contiguous()
        )
        x = x_cpu.to(device)
        # Probe includes all states, both reconstructions and cross-stage kernels.
        print(f"[calibration start] {mode}={candidate} pair_chunk={relation_batch}", flush=True)
        computed = compute_chunk(
            top,
            x,
            config,
            ["local_degree"],
            relation_batch,
            progress_label=f"calibration-{mode}-{candidate}",
        )
        synchronize(device)
        seconds = time.perf_counter() - start
        memory = _memory(device)
        safe = device.type != "cuda" or memory["peak_vram_bytes"] <= (
            free * config["runtime"]["gpu_memory_safety_fraction"]
        )
        row.update(
            status="measured" if safe else "rejected_measured_memory",
            seconds=seconds,
            feature_cells_per_second=x.numel() / seconds,
            physical_graph_batch=candidate if scalar else 1,
            channel_chunk=channels,
            **memory,
        )
    except torch.cuda.OutOfMemoryError as error:
        if device.type != "cuda":
            raise
        row.update(
            status="CUDA_OOM",
            error=str(error),
            seconds=time.perf_counter() - start,
            feature_cells_per_second=None,
            **_memory(device),
        )
        print(
            f"[calibration OOM] {mode}={candidate} pair_chunk={relation_batch}; "
            "recorded resource failure; continuing exact chunk candidates",
            flush=True,
        )
    finally:
        del computed, x, x_cpu, top
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rows.append(row)
    print(
        f"[calibration] {mode}={candidate} pair_chunk={relation_batch} status={row['status']} "
        f"seconds={row['seconds']:.3f} cells/s={row['feature_cells_per_second']} "
        f"peak={row['peak_vram_bytes']}",
        flush=True,
    )


def _merge_trace(accumulator, chunk):
    """Channels form a single vector trace; iteration counts/residuals take maxima."""
    for weight, group in chunk.items():
        target = accumulator.setdefault(weight, {})
        for kind in ("local", "transfer", "assembly"):
            target.setdefault(kind, [{} for _ in range(3)])
            for stage, values in enumerate(group[kind]):
                for key, array in values.items():
                    if array.ndim == 1:
                        reduced = array.copy()
                        if key not in target[kind][stage]:
                            target[kind][stage][key] = reduced
                        elif not np.array_equal(target[kind][stage][key], reduced):
                            raise ValueError("topology counts changed between feature chunks")
                        continue
                    maximum = "solver_" in key
                    reduced = (
                        array.max(axis=1, keepdims=True)
                        if maximum
                        else array.sum(axis=1, keepdims=True)
                    )
                    previous = target[kind][stage].get(key)
                    target[kind][stage][key] = (
                        reduced
                        if previous is None
                        else (np.maximum(previous, reduced) if maximum else previous + reduced)
                    )
        for kind in ("relations", "temporal"):
            target.setdefault(kind, {})
            for stages, values in group[kind].items():
                if kind == "temporal":
                    reduced = values.sum(axis=1, keepdims=True)
                    target[kind][stages] = target[kind].get(stages, 0) + reduced
                else:
                    destination = target[kind].setdefault(stages, {})
                    for name, array in values.items():
                        destination[name] = destination.get(name, 0) + array.sum(
                            axis=1, keepdims=True
                        )
    return accumulator


def _metadata(case):
    return {
        "graph_id": case.graph_id,
        "family": case.family,
        "dataset": case.graph_id if case.family == "citation" else "synthetic",
        "actual_data": case.actual_data,
        "feature_mode": case.feature_mode,
    }


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _relative_max(error, norm):
    defined = norm > 0
    return float(np.sqrt(error[defined] / norm[defined]).max()) if np.any(defined) else None


def _relation_summary(meta, weight, left, right, name, values):
    flat = values.reshape(-1)
    return {
        **meta,
        "weight": weight,
        "stage_from": f"H{left}",
        "stage_to": f"H{right}",
        "relation": name,
        "count": int(flat.size),
        "sum": float(flat.sum()),
        "mean": float(flat.mean()) if flat.size else None,
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "negative_count": int((flat < 0).sum()),
    }


def collect(cases, top, computed, tables, summaries):
    """CPU export of complete scalar realizations or a complete citation vector trace."""
    for graph_index, case in enumerate(cases):
        meta = _metadata(case)
        lo, hi = top.graph_node_offsets[graph_index : graph_index + 2]
        # Pair ordering is per graph: forward E then reverse E, offset by prior 2E.
        plo, phi = 2 * top.graph_edge_offsets[graph_index : graph_index + 2]
        pairs = top.pair_centers[:, plo:phi] - lo
        for weight, group in computed.items():
            for state, values in enumerate(group["local"]):
                data = {key: array[lo:hi] for key, array in values.items()}
                norm, error = data["flow_norm_sq"], data["reconstruction_error_sq"]
                summaries["local"].append(
                    {
                        **meta,
                        "weight": weight,
                        "state": f"H{state}",
                        "count": int(norm.size),
                        **{
                            f"{key}_sum": float(data[key].sum())
                            for key in (
                                "energy",
                                "flow_norm_sq",
                                "divergence_norm_sq",
                                "cycle_norm_sq",
                                "reconstruction_error_sq",
                            )
                        },
                        "cycle_fraction_sq": _ratio(data["cycle_norm_sq"].sum(), norm.sum()),
                        "reconstruction_relative_error_max": _relative_max(error, norm),
                    }
                )

                def local_rows(
                    case=case,
                    norm=norm,
                    meta=meta,
                    weight=weight,
                    state=state,
                    data=data,
                    error=error,
                ):
                    for center in range(case.num_nodes):
                        for realization in range(norm.shape[1]):
                            yield {
                                **meta,
                                "weight": weight,
                                "state": f"H{state}",
                                "center": center,
                                "realization": realization
                                if case.feature_mode == "independent_scalar_columns"
                                else "vector_trace",
                                **{
                                    key: float(array[center, realization])
                                    for key, array in data.items()
                                },
                                "cycle_fraction_sq": _ratio(
                                    data["cycle_norm_sq"][center, realization],
                                    norm[center, realization],
                                ),
                                "reconstruction_relative_error": None
                                if norm[center, realization] == 0
                                else float(
                                    np.sqrt(error[center, realization] / norm[center, realization])
                                ),
                            }

                tables["local"].write(local_rows())
                transfer = {key: array[plo:phi] for key, array in group["transfer"][state].items()}
                _validate_transfer(transfer)
                summaries["transfer"].append(
                    {
                        **meta,
                        "weight": weight,
                        "state": f"H{state}",
                        "count": int((phi - plo) * norm.shape[1]),
                        **{
                            f"{key}_sum": float(array.sum())
                            for key, array in transfer.items()
                            if key.endswith("norm_sq")
                        },
                    }
                )

                def transfer_rows(
                    phi=phi,
                    plo=plo,
                    norm=norm,
                    meta=meta,
                    weight=weight,
                    state=state,
                    pairs=pairs,
                    case=case,
                    transfer=transfer,
                ):
                    for pair in range(phi - plo):
                        for realization in range(norm.shape[1]):
                            yield {
                                **meta,
                                "weight": weight,
                                "state": f"H{state}",
                                "sender": int(pairs[0, pair]),
                                "receiver": int(pairs[1, pair]),
                                "realization": realization
                                if case.feature_mode == "independent_scalar_columns"
                                else "vector_trace",
                                **{
                                    key: int(array[pair])
                                    if array.ndim == 1
                                    else float(array[pair, realization])
                                    for key, array in transfer.items()
                                },
                            }

                tables["transfer"].write(transfer_rows())
                assembly = {
                    key: array[graph_index] for key, array in group["assembly"][state].items()
                }
                absolute = np.abs(
                    assembly["energy_overlap_corrected"] - assembly["energy_physical_average"]
                )
                if not np.allclose(
                    assembly["energy_overlap_corrected"],
                    assembly["energy_physical_average"],
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "overlap-corrected energy differs from physical averaged-C identity"
                    )
                summaries["assembly"].append(
                    {
                        **meta,
                        "weight": weight,
                        "state": f"H{state}",
                        "count": int(norm.shape[1]),
                        **{f"{key}_sum": float(array.sum()) for key, array in assembly.items()},
                        "identity_abs_error_max": float(absolute.max()),
                    }
                )
            for (left, right), relations in group["relations"].items():
                for name, array in relations.items():
                    values = array[plo:phi]
                    summaries["relation"].append(
                        _relation_summary(meta, weight, left, right, name, values)
                    )

                    def relation_rows(
                        phi=phi,
                        plo=plo,
                        values=values,
                        meta=meta,
                        weight=weight,
                        left=left,
                        right=right,
                        name=name,
                        pairs=pairs,
                        case=case,
                    ):
                        for pair in range(phi - plo):
                            for realization in range(values.shape[1]):
                                yield {
                                    **meta,
                                    "weight": weight,
                                    "stage_from": f"H{left}",
                                    "stage_to": f"H{right}",
                                    "relation": name,
                                    "sender": int(pairs[0, pair]),
                                    "receiver": int(pairs[1, pair]),
                                    "realization": realization
                                    if case.feature_mode == "independent_scalar_columns"
                                    else "vector_trace",
                                    "value": float(values[pair, realization]),
                                }

                    tables["relation"].write(relation_rows())
            for (left, right), array in group["temporal"].items():
                values = array[lo:hi]
                summaries["relation"].append(
                    _relation_summary(meta, weight, left, right, "same_local", values)
                )
                tables["temporal"].write(
                    {
                        **meta,
                        "weight": weight,
                        "stage_from": f"H{left}",
                        "stage_to": f"H{right}",
                        "center": center,
                        "realization": realization
                        if case.feature_mode == "independent_scalar_columns"
                        else "vector_trace",
                        "value": float(values[center, realization]),
                    }
                    for center in range(case.num_nodes)
                    for realization in range(values.shape[1])
                )


def _validate_transfer(values):
    for space, parts in (
        ("divergence", ("retained", "omitted")),
        ("flow", ("retained", "boundary", "omitted")),
    ):
        sender = values[f"sender_{space}_norm_sq"]
        partition = sum(values[f"{part}_{space}_norm_sq"] for part in parts)
        if not np.allclose(sender, partition, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{space} transfer partition bookkeeping failed")


def _graph_row(case, top):
    degrees = np.bincount(top.edges.flatten(), minlength=top.n)
    return {
        **_metadata(case),
        "num_nodes": top.n,
        "num_edges": top.num_edges,
        "num_features": case.num_features,
        "local_node_occurrences": top.num_local_nodes,
        "local_edge_occurrences": top.num_local_edges,
        "directed_pairs": top.num_pairs,
        "max_local_nodes": top.max_local_nodes,
        "reference_tau": float(0.5 / degrees.max()) if degrees.max() else 0.0,
        "all_nodes_edges_features": True,
        "sampling_ratio": 1.0,
    }


def validate_device(config, device):
    if config["profile"] == "full" and (device.type != "cuda" or os.name != "posix"):
        raise ValueError(
            "FULL is a server CUDA run; use the separate DEBUG profile for local validation"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        if torch.cuda.device_count() != 1:
            raise ValueError(
                "select the single allocated GPU with CUDA_VISIBLE_DEVICES; "
                "this run uses one assigned GPU"
            )


def run(args, output):
    config = read_config(args.config, args.profile)
    device = torch.device(args.device)
    validate_device(config, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hardware = runtime_resources(device)
    hardware["output_storage"] = shutil.disk_usage(output)._asdict()
    requested_threads = config["runtime"]["cpu_threads"]
    threads = available_cpus(hardware) if requested_threads == "auto" else requested_threads
    if threads > available_cpus(hardware):
        raise ValueError("CPU threads exceed measured affinity/quota")
    torch.set_num_threads(threads)
    print(
        f"[start] profile={config['profile']} device={device} float64 "
        "trainable_parameters=0 epochs=N/A reference_steps=2",
        flush=True,
    )
    print("[hardware] " + json.dumps(hardware, ensure_ascii=False), flush=True)
    print(
        "[scope] fixed local energy/relation/recovery/transfer audit; no classifier or learned C/W",
        flush=True,
    )
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
            "physical_graph_batch": "measured",
            "all_original_feature_channels": True,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": "same_as_physical_graph_batch",
            "optimizer_steps": 0,
            "DataLoader": "complete_CPU_cache_no_iteration_loader",
        },
    )
    resource_rows = []
    start = time.perf_counter()
    cases, manifest = prepare_cases(config, args.data_root, output, download=not args.no_download)
    resource_rows.extend(manifest["cpu_input_calibration"])
    topologies, topology_records = prepare_topologies(
        cases, config, hardware, output, resource_rows
    )
    graph_rows = [_graph_row(case, top) for case, top in zip(cases, topologies, strict=True)]
    synthetic_count = config["synthetic"]["graphs"]
    # Representative large topologies make graph batch resource calibration conservative.
    order = sorted(
        range(synthetic_count), key=lambda index: topologies[index].num_local_edges, reverse=True
    )
    synthetic_cases = [cases[index] for index in order]
    synthetic_tops = [topologies[index] for index in order]
    graph_batch, synthetic_pair_chunk = calibrate(
        synthetic_cases, synthetic_tops, config, device, resource_rows, "graphs"
    )
    tables = {
        name: Table(output / filename)
        for name, filename in {
            "local": "local_states.csv",
            "relation": "relations.csv",
            "transfer": "transfers.csv",
            "temporal": "local_temporal.csv",
        }.items()
    }
    summaries = {key: [] for key in ("local", "relation", "transfer", "assembly")}
    completed = []
    try:
        for offset in range(0, synthetic_count, graph_batch):
            batch_cases = synthetic_cases[offset : offset + graph_batch]
            cpu_top = batch_topologies(synthetic_tops[offset : offset + graph_batch])
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            began = time.perf_counter()
            top = cpu_top.to(device)
            x = torch.cat([case.x for case in batch_cases], dim=0).to(device)
            computed = compute_chunk(
                top,
                x,
                config,
                relation_batch=synthetic_pair_chunk,
                progress_label=f"synthetic-{offset + 1}-{offset + len(batch_cases)}",
            )
            synchronize(device)
            compute_seconds = time.perf_counter() - began
            collect_start = time.perf_counter()
            collect(batch_cases, cpu_top, computed, tables, summaries)
            resource_rows.append(
                {
                    "type": "execution",
                    "graph_ids": [case.graph_id for case in batch_cases],
                    "physical_graph_batch": len(batch_cases),
                    "channel_chunk": x.shape[1],
                    "channels_used": x.shape[1],
                    "relation_batch": synthetic_pair_chunk,
                    "compute_seconds": compute_seconds,
                    "export_seconds": time.perf_counter() - collect_start,
                    "graphs_per_second": len(batch_cases) / compute_seconds,
                    "feature_cells_per_second": x.numel() / compute_seconds,
                    "rss_bytes": psutil.Process().memory_info().rss,
                    "cpu_percent_snapshot": psutil.cpu_percent(),
                    **_memory(device),
                }
            )
            completed.extend(case.graph_id for case in batch_cases)
            print(
                f"[progress] synthetic={len(completed)}/{synthetic_count} "
                f"batch={len(batch_cases)} compute={compute_seconds:.3f}s "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
            del computed, x, top, cpu_top
        for case, cpu_top in zip(
            cases[synthetic_count:], topologies[synthetic_count:], strict=True
        ):
            chunk_size, pair_chunk = calibrate(
                [case], [cpu_top], config, device, resource_rows, "channels"
            )
            top = cpu_top.to(device)
            accumulator = {}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            for first in range(0, case.num_features, chunk_size):
                x = case.x[:, first : first + chunk_size].contiguous().to(device)
                _merge_trace(
                    accumulator,
                    compute_chunk(
                        top,
                        x,
                        config,
                        relation_batch=pair_chunk,
                        progress_label=f"{case.graph_id}-channels-{first + 1}-{first + x.shape[1]}",
                    ),
                )
                print(
                    f"[progress] {case.graph_id} "
                    f"channels={min(first + chunk_size, case.num_features)}/{case.num_features} "
                    f"all nodes={case.num_nodes} elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
                del x
            synchronize(device)
            compute_seconds = time.perf_counter() - began
            collect_start = time.perf_counter()
            collect([case], cpu_top, accumulator, tables, summaries)
            resource_rows.append(
                {
                    "type": "execution",
                    "graph_ids": [case.graph_id],
                    "physical_graph_batch": 1,
                    "physical_graph_batch_reason": "complete_dataset_graph_with_batched_locals",
                    "channel_chunk": chunk_size,
                    "channels_used": case.num_features,
                    "relation_batch": pair_chunk,
                    "compute_seconds": compute_seconds,
                    "export_seconds": time.perf_counter() - collect_start,
                    "feature_cells_per_second": case.x.numel() / compute_seconds,
                    "rss_bytes": psutil.Process().memory_info().rss,
                    "cpu_percent_snapshot": psutil.cpu_percent(),
                    **_memory(device),
                }
            )
            completed.append(case.graph_id)
            del top, accumulator
    finally:
        for table in tables.values():
            table.close()
    if len(completed) != config["data"]["total_graphs"] or set(completed) != {
        case.graph_id for case in cases
    }:
        raise ValueError("incomplete or duplicate graph execution coverage")
    center_units = sum(
        case.num_nodes
        * (case.num_features if case.feature_mode == "independent_scalar_columns" else 1)
        for case in cases
    )
    pair_units = sum(
        2
        * case.edges.shape[1]
        * (case.num_features if case.feature_mode == "independent_scalar_columns" else 1)
        for case in cases
    )
    expected_rows = {
        "local": 6 * center_units,
        "transfer": 6 * pair_units,
        "relation": 30 * pair_units,
        "temporal": 4 * center_units,
    }
    if {key: table.count for key, table in tables.items()} != expected_rows:
        raise ValueError("raw record coverage differs from complete center/pair/feature contract")
    assert_source_unchanged(source)
    assert_inputs_unchanged(output, manifest)
    for record in topology_records:
        if file_sha256(output / record["file"]) != record["sha256"]:
            raise ValueError("saved topology correspondence changed during execution")
    completion = {
        "status": "complete",
        "profile": config["profile"],
        "debug": config["profile"] == "debug",
        "graphs": len(completed),
        "synthetic_graphs": synthetic_count,
        "citation_graphs": 3,
        "actual_citation_graphs": sum(case.actual_data for case in cases),
        "synthetic_scalar_inputs": config["synthetic"]["scalar_inputs"],
        "sampling_ratio": 1.0,
        "all_nodes_edges_features": True,
        "states": config["states"],
        "weights": config["weights"],
        "relations": config["relations"],
        "trainable_parameters": 0,
        "optimizer_updates": 0,
        "classifier_training_run": False,
        "known_C_all_nodes_recovery_only": True,
        "raw_rows": {key: table.count for key, table in tables.items()},
        "source_digest": source["code_digest"],
        "source_and_inputs_preserved": True,
        "physical_graph_batch_selected": graph_batch,
        "elapsed_seconds": time.perf_counter() - start,
    }
    print("[report] writing complete summaries and scientific figures", flush=True)
    write_report(
        output,
        config,
        graph_rows,
        summaries["local"],
        summaries["relation"],
        summaries["transfer"],
        summaries["assembly"],
        resource_rows,
        completion,
    )
    assert_source_unchanged(source)
    write_json(output / "completion.json", completion)
    print(f"[complete] graphs={len(completed)} results={output.resolve()}", flush=True)
    print("[scope] fixed audit complete; no learned model/classifier training was run", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    # Refuse invalid device/config before creating artifacts or downloading data.
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
                        "existing_files_preserved": True,
                        "results": str(args.output_dir.resolve()),
                    },
                )
                raise


if __name__ == "__main__":
    main()
