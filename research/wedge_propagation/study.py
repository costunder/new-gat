"""Experiments 0/1: full fixed-operator measurements, never model training.

Run with ``python -m research.wedge_propagation.study --help``. The default
profile contains all 198 graphs and 3,168 feature realizations. CUDA operations
are batched across graphs of the same size and all realizations simultaneously.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from .algebra import run_algebra
from .data import (
    GraphCase,
    GraphSpec,
    feature_content_hash,
    graph_content_hash,
    make_case,
    make_specs,
)
from .operators import build_wedges, fixed_wedge_apply, fixed_wedge_fast_apply, laplacian_apply
from .report import write_report


class Tee:
    def __init__(self, stream: Any, logfile: Any) -> None:
        self.stream, self.logfile = stream, logfile

    def write(self, message: str) -> int:
        self.stream.write(message)
        self.logfile.write(message)
        return len(message)

    def flush(self) -> None:
        self.stream.flush()
        self.logfile.flush()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty result table: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def runtime_resources(device: torch.device) -> dict:
    process = psutil.Process()
    cpu_quota = None
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    if quota_path.is_file():
        quota, period = quota_path.read_text().split()
        if quota != "max":
            cpu_quota = float(quota) / float(period)
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    memory = psutil.virtual_memory()
    memory_limit = None
    memory_limit_path = Path("/sys/fs/cgroup/memory.max")
    if memory_limit_path.is_file():
        value = memory_limit_path.read_text().strip()
        if value != "max":
            memory_limit = int(value)
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "logical_cpu": os.cpu_count(),
        "cpu_affinity_count": affinity,
        "cpu_quota": cpu_quota,
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "container_memory_limit_bytes": memory_limit,
        "process_rss_bytes": process.memory_info().rss,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_device": str(device),
        "gpu_count_visible": 0,
        "gpu_count_used": 0,
        "gpu": None,
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; no CPU fallback. Check the GPU env."
            )
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        prop = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        info.update(gpu_count_visible=torch.cuda.device_count(), gpu_count_used=1)
        info["gpu"] = {
            "name": prop.name,
            "total_memory_bytes": prop.total_memory,
            "free_memory_bytes": free,
            "runtime_total_memory_bytes": total,
            "multiprocessors": prop.multi_processor_count,
            "uuid": str(getattr(prop, "uuid", "unavailable")),
            "capability": [prop.major, prop.minor],
            "cuda_runtime": torch.version.cuda,
        }
    return info


def available_cpus(resources: dict) -> int:
    count = int(resources["cpu_affinity_count"] or 1)
    if resources["cpu_quota"] is not None:
        count = min(count, max(1, int(resources["cpu_quota"])))
    return count


@dataclass
class Prepared:
    case: GraphCase
    wedges: torch.Tensor
    lap: torch.Tensor
    q: torch.Tensor
    correction: torch.Tensor
    degree_sum_std: float | None


def prepare_case(spec: GraphSpec) -> Prepared:
    case = make_case(spec)
    n = case.num_nodes
    edge = case.edges.numpy()
    wedges = build_wedges(case.edges, n)
    degree = np.bincount(edge.reshape(-1), minlength=n)
    lap = np.zeros((n, n), dtype=np.float64)
    correction = np.zeros_like(lap)
    if edge.shape[1]:
        u, v = edge
        weight = degree[u] + degree[v] - 4
        for matrix, values in ((lap, np.ones(len(u))), (correction, weight)):
            np.add.at(matrix, (u, u), values)
            np.add.at(matrix, (v, v), values)
            np.add.at(matrix, (u, v), -values)
            np.add.at(matrix, (v, u), -values)
        degree_sum_std = float(np.std(degree[u] + degree[v]))
    else:
        degree_sum_std = None
    # Nine COO entries per wedge; no per-path Python loop or dense P x N matrix.
    q = np.zeros_like(lap)
    rows = wedges.numpy()
    coefficient = (1.0, -2.0, 1.0)
    for first in range(3):
        for second in range(3):
            np.add.at(q, (rows[first], rows[second]), coefficient[first] * coefficient[second])
    return Prepared(
        case,
        wedges,
        torch.from_numpy(lap),
        torch.from_numpy(q),
        torch.from_numpy(correction),
        degree_sum_std,
    )


def prepare_parallel(specs: list[GraphSpec], workers: int) -> list[Prepared]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [prepare_case(spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(prepare_case, specs))


def choose_workers(
    specs: list[GraphSpec], resources: dict, requested: str
) -> tuple[int, list[dict]]:
    if requested != "auto":
        count = int(requested)
        if count < 1:
            raise ValueError("workers must be positive")
        return count, [{"workers": count, "selection": "explicit user override"}]
    available = available_cpus(resources)
    candidates = sorted({1, *(v for v in (2, 4, 8) if v <= available)})
    largest = max(spec.num_nodes for spec in specs)
    # Calibration is separate; every final spec is generated after selecting workers.
    probes = [spec for spec in specs if spec.num_nodes == largest]
    # Warm imports/allocator once so the first worker candidate is not penalized.
    prepare_case(probes[0])
    trials = []
    for workers in candidates:
        start = time.perf_counter()
        prepared = prepare_parallel(probes, workers)
        elapsed = time.perf_counter() - start
        del prepared
        row = {
            "workers": workers,
            "calibration_graphs": len(probes),
            "seconds": elapsed,
            "graphs_per_second": len(probes) / elapsed,
        }
        trials.append(row)
        print(
            f"[cpu calibration] workers={workers} graphs={len(probes)} "
            f"seconds={elapsed:.3f} graphs/s={row['graphs_per_second']:.1f}",
            flush=True,
        )
    return max(trials, key=lambda trial: trial["graphs_per_second"])["workers"], trials


@dataclass
class DeviceBatch:
    prepared: list[Prepared]
    lap: torch.Tensor
    q: torch.Tensor
    correction: torch.Tensor
    x: torch.Tensor
    edges: torch.Tensor
    wedges: torch.Tensor


def upload_batch(prepared: list[Prepared], device: torch.device) -> DeviceBatch:
    if not prepared:
        raise ValueError("cannot upload an empty graph batch")
    n = prepared[0].case.num_nodes
    if any(item.case.num_nodes != n for item in prepared):
        raise ValueError("a dense size bucket must have a common node count")
    edges = torch.cat([item.case.edges + index * n for index, item in enumerate(prepared)], 1)
    wedges = torch.cat([item.wedges + index * n for index, item in enumerate(prepared)], 1)
    host = [
        torch.stack([getattr(item, name) for item in prepared])
        for name in ("lap", "q", "correction")
    ]
    host += [torch.stack([item.case.features for item in prepared]), edges, wedges]
    if device.type == "cuda":
        host = [value.pin_memory() for value in host]
    data = [value.to(device, non_blocking=device.type == "cuda") for value in host]
    return DeviceBatch(prepared, *data)


def compute_batch(batch: DeviceBatch) -> dict[str, torch.Tensor]:
    """One batched tensor path; no per-graph GPU forward or scalar transfer."""
    lap, q, x = batch.lap, batch.q, batch.x
    lap2 = lap @ lap
    lx, l2x, qx = lap @ x, lap2 @ x, q @ x
    flat = x.flatten(0, 1)
    explicit = fixed_wedge_apply(batch.wedges, flat).reshape_as(x)
    fast = fixed_wedge_fast_apply(batch.edges, flat).reshape_as(x)
    sparse_lx = laplacian_apply(batch.edges, flat).reshape_as(x)
    eig_l, eig_l2, eig_q = (torch.linalg.eigvalsh(matrix) for matrix in (lap, lap2, q))
    rho_l, rho_l2, rho_q = (eigen[:, -1] for eigen in (eig_l, eig_l2, eig_q))
    q_norm, l_norm, l2_norm = (
        torch.linalg.vector_norm(matrix, dim=(-2, -1)) for matrix in (q, lap, lap2)
    )
    basis = torch.stack(
        (
            lap / l_norm.clamp_min(1e-30)[:, None, None],
            lap2 / l2_norm.clamp_min(1e-30)[:, None, None],
        ),
        -1,
    ).flatten(1, 2)
    target = q.flatten(1)
    gram = basis.transpose(1, 2) @ basis
    rhs = basis.transpose(1, 2) @ target[..., None]
    scaled_coeff = torch.linalg.pinv(gram, hermitian=True, rtol=1e-12) @ rhs
    fitted = (basis @ scaled_coeff).squeeze(-1)
    residual_abs = torch.linalg.vector_norm(target - fitted, dim=-1)
    poly_residual = residual_abs / q_norm.clamp_min(1e-30)
    raw_norm = torch.stack(
        [torch.linalg.vector_norm(action, dim=1) for action in (lx, l2x, qx)], -1
    )
    energy = torch.stack([(x * action).sum(1) for action in (lx, l2x, qx)], -1)
    normalized_q = qx / rho_q.clamp_min(1e-30)[:, None, None]
    normalized_l2 = l2x / rho_l2.clamp_min(1e-30)[:, None, None]
    denom = torch.linalg.vector_norm(normalized_q, dim=1)
    normalized_difference = torch.linalg.vector_norm(normalized_q - normalized_l2, dim=1)
    normalized_difference = torch.where(
        (denom > 1e-12) & (rho_q[:, None] > 0) & (rho_l2[:, None] > 0),
        normalized_difference / denom.clamp_min(1e-30),
        torch.nan,
    )
    operator_norm = torch.stack((rho_l, rho_l2, rho_q), -1)
    normalized_norm = torch.where(
        operator_norm[:, None, :] > 0,
        raw_norm / operator_norm[:, None, :].clamp_min(1e-30),
        torch.nan,
    )
    normalized_energy = torch.where(
        operator_norm[:, None, :] > 0,
        energy / operator_norm[:, None, :].clamp_min(1e-30),
        torch.nan,
    )
    absolute_errors = torch.stack(
        [
            torch.linalg.vector_norm(q - (lap2 + batch.correction), dim=(-2, -1)),
            torch.linalg.vector_norm(explicit - fast, dim=(-2, -1)),
            torch.linalg.vector_norm(qx - explicit, dim=(-2, -1)),
            torch.linalg.vector_norm(lx - sparse_lx, dim=(-2, -1)),
        ],
        -1,
    )
    reference_norms = torch.stack(
        [
            q_norm,
            torch.linalg.vector_norm(explicit, dim=(-2, -1)),
            torch.linalg.vector_norm(qx, dim=(-2, -1)),
            torch.linalg.vector_norm(lx, dim=(-2, -1)),
        ],
        -1,
    )
    relative_errors = torch.where(
        reference_norms > 1e-12, absolute_errors / reference_norms.clamp_min(1e-30), torch.nan
    )
    coeff = scaled_coeff.squeeze(-1) / torch.stack((l_norm, l2_norm), -1).clamp_min(1e-30)
    return {
        "eig_L": eig_l,
        "eig_L2": eig_l2,
        "eig_Q": eig_q,
        "errors": relative_errors,
        "absolute_errors": absolute_errors,
        "reference_norms": reference_norms,
        "raw_norm": raw_norm,
        "energy": energy,
        "poly": poly_residual,
        "poly_abs": residual_abs,
        "coeff": coeff,
        "q_norm": q_norm,
        "normalized_difference": normalized_difference,
        "normalized_norm": normalized_norm,
        "normalized_energy": normalized_energy,
        "raw_actions": torch.stack((lx, l2x, qx), -1),
        "lambda": operator_norm,
    }


def choose_batch_size(
    prepared: list[Prepared], device: torch.device, requested: str, profile: str
) -> tuple[int, list[dict]]:
    if requested != "auto":
        batch_size = int(requested)
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        return batch_size, [{"batch_size": batch_size, "selection": "explicit user override"}]
    n = max(item.case.num_nodes for item in prepared)
    probes = [item for item in prepared if item.case.num_nodes == n]
    candidates = sorted({min(value, len(probes)) for value in (4, 8, 16, 32, len(probes))})
    warmup, repeats = (1, 2) if profile == "debug" else (3, 10)
    trials = []
    for size in candidates:
        for _ in range(warmup):
            for offset in range(0, len(probes), size):
                batch = upload_batch(probes[offset : offset + size], device)
                compute_batch(batch)
                synchronize(device)
                del batch
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                for offset in range(0, len(probes), size):
                    batch = upload_batch(probes[offset : offset + size], device)
                    compute_batch(batch)
                    synchronize(device)
                    del batch
        synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        row = {
            "batch_size": size,
            "calibration_graphs_per_pass": len(probes),
            "warmup_passes": warmup,
            "measured_passes": repeats,
            "seconds": elapsed,
            "graphs_per_second": len(probes) * repeats / elapsed,
            "peak_vram_bytes": peak,
            "timing_scope": "CPU batch packing, H2D, validation, operator calculations",
            "simultaneously_resident_graph_batches": 1,
        }
        trials.append(row)
        print(
            f"[batch calibration] device={device} batch={size} graphs/pass={len(probes)} "
            f"graphs/s={row['graphs_per_second']:.1f} peak_bytes={peak}",
            flush=True,
        )
    chosen = max(trials, key=lambda trial: trial["graphs_per_second"])["batch_size"]
    return chosen, trials


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def collect_batch(
    batch: DeviceBatch, result: dict[str, torch.Tensor]
) -> tuple[list, list, dict, dict]:
    data = {name: tensor.detach().cpu().numpy() for name, tensor in result.items()}
    absolute, references = data["absolute_errors"], data["reference_norms"]
    if (
        not np.isfinite(absolute).all()
        or not np.isfinite(references).all()
        or np.any(absolute > 1e-10 + 1e-10 * references)
    ):
        raise ArithmeticError(
            f"operator implementations disagree: absolute={absolute.tolist()}, "
            f"reference_norms={references.tolist()}"
        )
    for name in ("raw_norm", "energy", "lambda", "poly", "poly_abs", "coeff"):
        if not np.isfinite(data[name]).all():
            raise ArithmeticError(f"nonfinite fixed-operator result: {name}")
    operators, actions, spectra, action_values = [], [], {}, {}
    for index, item in enumerate(batch.prepared):
        case = item.case
        lam = data["lambda"][index]
        for operator in ("L", "L2", "Q"):
            eig = data[f"eig_{operator}"][index]
            if eig.min() < -1e-10 * max(1, float(eig.max())):
                raise ArithmeticError(f"{case.graph_id} {operator}: PSD check failed")
            spectra[f"{case.graph_id}__{operator}"] = eig
        components = int((data["eig_L"][index] <= 1e-10 * max(1, lam[0])).sum())
        nullities = {
            f"nullity_{name}": int(
                (np.abs(data[f"eig_{name}"][index]) <= 1e-10 * max(1, lam[position])).sum()
            )
            for position, name in enumerate(("L", "L2", "Q"))
        }
        q_is_zero = data["q_norm"][index] == 0
        operators.append(
            {
                "graph_id": case.graph_id,
                "family": case.family,
                "num_nodes": case.num_nodes,
                "num_edges": case.edges.shape[1],
                "num_wedges": item.wedges.shape[1],
                "polynomial_residual": None if q_is_zero else float(data["poly"][index]),
                # This relative projection residual is invariant under global Q rescaling.
                "normalized_polynomial_residual": None if q_is_zero else float(data["poly"][index]),
                "polynomial_absolute_residual": float(data["poly_abs"][index]),
                "polynomial_coefficient_L": float(data["coeff"][index, 0]),
                "polynomial_coefficient_L2": float(data["coeff"][index, 1]),
                "degree_sum_std": item.degree_sum_std,
                "num_components": components,
                **nullities,
                "lambda_L": float(lam[0]),
                "lambda_L2": float(lam[1]),
                "lambda_Q": float(lam[2]),
                **{
                    f"{name}_{kind}_error": finite_or_none(data[key][index, position])
                    for position, name in enumerate(
                        ("identity", "explicit_fast", "dense_explicit", "dense_sparse_L")
                    )
                    for kind, key in (("relative", "errors"), ("absolute", "absolute_errors"))
                },
            }
        )
        for position, name in enumerate(("L", "L2", "Q")):
            raw = data["raw_actions"][index, :, :, position]
            action_values[f"{case.graph_id}__{name}X"] = raw
            if lam[position] > 0:
                action_values[f"{case.graph_id}__normalized_{name}X"] = raw / lam[position]
        for feature in range(case.features.shape[1]):
            norm, energy = data["raw_norm"][index, feature], data["energy"][index, feature]
            actions.append(
                {
                    "graph_id": case.graph_id,
                    "family": case.family,
                    "num_nodes": case.num_nodes,
                    "feature_index": feature,
                    "raw_L_norm": float(norm[0]),
                    "raw_L2_norm": float(norm[1]),
                    "raw_Q_norm": float(norm[2]),
                    "energy_L": float(energy[0]),
                    "energy_L2": float(energy[1]),
                    "energy_Q": float(energy[2]),
                    "normalized_action_Q_L2_relative_difference": finite_or_none(
                        data["normalized_difference"][index, feature]
                    ),
                    **{
                        f"normalized_{name}_{metric}": finite_or_none(
                            data[key][index, feature, pos]
                        )
                        for pos, name in enumerate(("L", "L2", "Q"))
                        for metric, key in (
                            ("norm", "normalized_norm"),
                            ("energy", "normalized_energy"),
                        )
                    },
                }
            )
    return operators, actions, spectra, action_values


def save_dataset(path: Path, prepared: list[Prepared]) -> list[dict]:
    arrays, manifest = {}, []
    for item in prepared:
        case = item.case
        arrays[f"{case.graph_id}__edges"] = case.edges.numpy()
        arrays[f"{case.graph_id}__features"] = case.features.numpy()
        arrays[f"{case.graph_id}__wedges"] = item.wedges.numpy()
        degree = np.bincount(case.edges.numpy().reshape(-1), minlength=case.num_nodes)
        arrays[f"{case.graph_id}__degree"] = degree
        arrays[f"{case.graph_id}__edge_degree_sum"] = degree[case.edges.numpy()].sum(0)
        manifest.append(
            {
                "graph_id": case.graph_id,
                "family": case.family,
                "num_nodes": case.num_nodes,
                "num_edges": case.edges.shape[1],
                "num_wedges": item.wedges.shape[1],
                "num_features": case.features.shape[1],
                "graph_seed": case.graph_seed,
                "feature_seed": case.feature_seed,
                "base_tree_seed": case.base_tree_seed,
                "graph_content_hash": graph_content_hash(case),
                "feature_content_hash": feature_content_hash(case),
            }
        )
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return manifest


def source_manifest() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).parent.glob("*.py"))
    digest = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "source_sha256": digest,
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
        "git_error": result.stderr.strip() if result.returncode != 0 else None,
    }


def run(args: argparse.Namespace, output: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is required; install this track's requirements.txt")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resources = runtime_resources(device)
    source_before = source_manifest()
    storage = psutil.disk_usage(str(output))
    resources["output_storage_total_bytes"] = storage.total
    resources["output_storage_free_bytes"] = storage.free
    write_json(output / "resources_before.json", resources)
    print(
        f"[start] profile={args.profile} device={device} dtype=float64 "
        "trainable_parameters=0 epochs=N/A C2=I",
        flush=True,
    )
    print("[hardware] " + json.dumps(resources, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    process = psutil.Process()
    cpu_before = process.cpu_times()
    psutil.cpu_percent(interval=None)
    # Small graph preprocessing is parallelized across graphs; avoid each
    # worker spawning another full intra-op pool for <=100-node index kernels.
    torch.set_num_threads(1)
    audit = run_algebra()
    write_json(output / "algebra.json", {"kind": "debug algebra audit", "cases": audit})
    specs = make_specs(args.seed, args.profile)
    workers, cpu_trials = choose_workers(specs, resources, args.workers)
    print(f"[prepare] all {len(specs)} graphs; workers={workers}", flush=True)
    prepare_start = time.perf_counter()
    prepared = prepare_parallel(specs, workers)
    prep_seconds = time.perf_counter() - prepare_start
    manifest = save_dataset(output / "dataset.npz", prepared)
    write_json(output / "data_manifest.json", {"master_seed": args.seed, "cases": manifest})
    feature_count = sum(item.case.features.shape[1] for item in prepared)
    graph_statistics = {
        name: {
            "min": min(values),
            "mean": float(np.mean(values)),
            "max": max(values),
            "sum": sum(values),
        }
        for name, values in {
            "nodes": [item.case.num_nodes for item in prepared],
            "edges": [item.case.edges.shape[1] for item in prepared],
            "wedges": [item.wedges.shape[1] for item in prepared],
        }.items()
    }
    if args.profile == "full" and (len(prepared), feature_count) != (198, 3168):
        raise AssertionError("full data contract changed")
    print(
        f"[data] graphs={len(prepared)} inputs={feature_count} fraction=1.0 "
        f"prepare_seconds={prep_seconds:.3f} no_sampling",
        flush=True,
    )
    print("[graph statistics] " + json.dumps(graph_statistics), flush=True)
    batch_size, gpu_trials = choose_batch_size(prepared, device, args.batch_size, args.profile)
    print(f"[selected] graph_batch={batch_size}; all feature draws run together", flush=True)
    buckets = defaultdict(list)
    for item in prepared:
        buckets[item.case.num_nodes].append(item)
    operators, actions, spectra, action_values, timings = [], [], {}, {}, []
    processed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for n, items in sorted(buckets.items()):
            for start in range(0, len(items), batch_size):
                group = items[start : start + batch_size]
                transfer_start = time.perf_counter()
                batch = upload_batch(group, device)
                synchronize(device)
                transfer_seconds = time.perf_counter() - transfer_start
                compute_start = time.perf_counter()
                computed = compute_batch(batch)
                synchronize(device)
                compute_seconds = time.perf_counter() - compute_start
                collect_start = time.perf_counter()
                op_rows, action_rows, eigenvalues, values = collect_batch(batch, computed)
                collect_seconds = time.perf_counter() - collect_start
                operators.extend(op_rows)
                actions.extend(action_rows)
                spectra.update(eigenvalues)
                action_values.update(values)
                processed += len(group)
                timings.append(
                    {
                        "nodes_per_graph": n,
                        "graph_batch": len(group),
                        "feature_batch_per_graph": group[0].case.features.shape[1],
                        "transfer_seconds": transfer_seconds,
                        "compute_seconds": compute_seconds,
                        "collect_seconds": collect_seconds,
                    }
                )
                elapsed = time.perf_counter() - started
                max_identity = max(row["identity_absolute_error"] for row in op_rows)
                print(
                    f"[fixed] graphs={processed}/{len(prepared)} n={n} batch={len(group)} "
                    f"compute={compute_seconds:.3f}s elapsed={elapsed:.1f}s "
                    f"max_identity_abs_error={max_identity:.2e}",
                    flush=True,
                )
                del computed, batch
    if len(operators) != len(prepared) or len(actions) != feature_count:
        raise AssertionError("not all graphs or feature realizations were evaluated")
    measurement_seconds = sum(
        row["transfer_seconds"] + row["compute_seconds"] + row["collect_seconds"] for row in timings
    )
    cpu_after = process.cpu_times()
    cpu_seconds = cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
    source_after = source_manifest()
    if source_before != source_after:
        raise RuntimeError("source files or Git revision changed during execution; use a fresh run")
    contract = {
        "track": "wedge_propagation",
        "experiments": [0, 1],
        "profile": args.profile,
        "debug": args.profile == "debug",
        "master_seed": args.seed,
        "model": None,
        "trainable_parameters": 0,
        "epochs": None,
        "optimizer_steps": 0,
        "C2": "identity",
        "operators": ["L=B.T B", "L2=L@L", "Q=A.T A"],
        "layers": None,
        "hidden": None,
        "heads": None,
        "channels": 1,
        "features_are_independent_scalar_draws": True,
        "graph_statistics": graph_statistics,
        "input_shapes": [
            [row["graph_batch"], row["nodes_per_graph"], row["feature_batch_per_graph"]]
            for row in timings
        ],
        "full_size_bucket_shapes": [
            [len(items), n, items[0].case.features.shape[1]] for n, items in sorted(buckets.items())
        ],
        "graph_count_total": len(prepared),
        "graph_count_evaluated": len(operators),
        "input_count_total": feature_count,
        "input_count_evaluated": len(actions),
        "data_fraction": 1.0,
        "sampling_ratio": 1.0,
        "path_sampling": False,
        "physical_graph_batch_selected": batch_size,
        "gradient_accumulation": None,
        "effective_training_batch": None,
        "dtype": "float64",
        "tf32": False,
        "cpu_preprocess_workers": workers,
        "data_loader_workers": None,
        "torch_cpu_intraop_threads": torch.get_num_threads(),
        "cpu_parallelism": "measured graph worker pool; one intra-op thread per worker",
        "available_cpu_count": available_cpus(resources),
        "data_loader_reason": "generated once, cached, bucketed tensors; no repeated file loader",
        "pinned_transfer": device.type == "cuda",
        "static_cache": True,
        "runtime": resources,
        "source": source_before,
        "cpu_calibration": cpu_trials,
        "source_unchanged_during_measurement": True,
        "validation_atol": 1e-10,
        "validation_rtol": 1e-10,
        "gpu_batch_calibration": gpu_trials,
        "preparation_seconds": prep_seconds,
        "batch_timings": timings,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "gpu_reserved_bytes_after_measurement": (
            torch.cuda.memory_reserved(device) if device.type == "cuda" else None
        ),
        "gpu_allocated_bytes_after_measurement": (
            torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        ),
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "system_cpu_percent_during_run": psutil.cpu_percent(interval=None),
        "process_cpu_seconds": cpu_seconds,
        "measurement_graphs_per_second": len(operators) / measurement_seconds,
        "measurement_inputs_per_second": len(actions) / measurement_seconds,
        "measurement_seconds_before_report": time.perf_counter() - started,
        "actual_batch_sizes": [row["graph_batch"] for row in timings],
        "normalized_polynomial_residual_note": "relative residual invariant to global Q rescaling",
        "learned_C2_training": False,
        "public_dataset_evaluation": False,
    }
    write_csv(output / "operators.csv", operators)
    write_csv(output / "actions.csv", actions)
    with (output / "spectra.npz").open("xb") as stream:
        np.savez_compressed(stream, **spectra)
    with (output / "action_values.npz").open("xb") as stream:
        np.savez_compressed(stream, **action_values)
    write_json(output / "contract.json", contract)
    print("[report] writing summary and scientific figures", flush=True)
    write_report(output, operators, actions, spectra, contract)
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "experiments": [0, 1],
            "seconds": time.perf_counter() - started,
            "graphs": len(operators),
            "inputs": len(actions),
        },
    )
    print(
        f"[complete] fixed operator study; graphs={len(operators)} inputs={len(actions)} "
        f"results={output}\n[scope] no learned C2 or classifier training was run",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new directory; never overwritten"
    )
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument(
        "--device", default="cuda", help="allocated GPU, e.g. cuda:0; explicit cpu allowed"
    )
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--batch-size", default="auto", help="measure full size bucket candidates or explicit size"
    )
    parser.add_argument(
        "--workers", default="auto", help="measure CPU preprocessing candidates or explicit count"
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "terminal.log").open("x", encoding="utf-8", buffering=1) as logfile:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, logfile)),
            contextlib.redirect_stderr(Tee(sys.stderr, logfile)),
        ):
            try:
                run(args, output)
            except Exception as error:
                traceback.print_exc()
                write_json(
                    output / "failure.json",
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "recovery": "inspect terminal.log; fix cause; use fresh output directory",
                    },
                )
                raise


if __name__ == "__main__":
    main()
