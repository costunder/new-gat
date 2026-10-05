"""Experiment B: all declared scalar-message targets, graphs and gate seeds.

FULL performs 240 learned runs x 500 optimizer updates on 531 graphs. Every
epoch uses all 240 train graphs and all 16 independent scalar realizations.
Physical graph chunks accumulate the exact full-corpus gradient; they never
change the number of optimizer updates. DEBUG uses its explicit separate
profile. All outputs use a new directory and existing evidence is preserved.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from ...wedge_propagation.study import Tee, available_cpus, runtime_resources, synchronize
from ..common import assert_source_unchanged, cpu_state, digest, save_checkpoint, source_manifest, write_csv, write_json
from ..geometry import prepare_geometry, prepare_geometries
from .data import SPLITS, SyntheticBatch, data_manifest, pack_cases, prepare_cases, save_dataset
from .model import (BASELINES, CONDITIONS, TARGETS, TRAINABLE, RawStudent, baseline_predict,
                    fit_baseline, graph_draw_errors, normalized_mse, parameter_vector_stats, target_message)
from .verify import run_math_checks

RECIPES = ("unit", "local_degree")
CONTRASTS = (("F2", "D1"), ("F2", "F1"), ("F2", "DA"),
             ("F2", "F0"), ("F1", "D1"), ("D1", "D0"))
FOLDER = Path(__file__).resolve().parent


def require_full_server(profile, device):
    if profile == "full":
        if device.type != "cuda" or sys.platform != "linux":
            raise ValueError("FULL requires an explicitly allocated Linux server CUDA environment")
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise ValueError("FULL requires explicit allocated CUDA_VISIBLE_DEVICES; machine-wide discovery is not an allocation")


def read_config(path, profile):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    default = json.loads((FOLDER / f"config_{profile}.json").read_text(encoding="utf-8"))
    # Explicit profiles are scientific contracts, not convenience defaults.
    if config != default:
        raise ValueError("config differs from the declared profile; no silent scientific reduction is permitted")
    if tuple(config["recipes"]) != RECIPES or tuple(config["teacher_recipes"]) != RECIPES:
        raise ValueError("both student and teacher recipes are required")
    if tuple(config["targets"]) != TARGETS or tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("all three targets and six gate conditions are required")
    if tuple(config["baselines"]) != BASELINES:
        raise ValueError("all four raw-message analytic baselines are required")
    return config


def budget(config, graphs):
    learned = len(RECIPES) ** 2 * len(TARGETS) * len(TRAINABLE) * len(config["model_seeds"])
    fixed = len(RECIPES) ** 2 * len(TARGETS) * (len(CONDITIONS) - len(TRAINABLE))
    fits = len(RECIPES) * len(TARGETS) * len(BASELINES)
    total = learned + fixed + fits
    return {"learned_runs": learned, "fixed_runs": fixed, "train_only_analytic_fits": fits,
            "epochs_per_learned_run": config["epochs"], "seed_optimizer_updates": learned * config["epochs"],
            "packed_optimizer_calls": learned // len(config["model_seeds"]) * config["epochs"],
            "logical_evaluation_jobs": total, "per_graph_rows": total * graphs,
            "per_realization_rows": total * graphs * config["features"],
            "fixed_seed_marker": -1, "fixed_seed_marker_is_not_an_independent_training_replica": True,
            "paired_contrasts": len(CONTRASTS) * len(RECIPES) ** 2 * len(TARGETS) * (len(SPLITS) - 1)}


class ResultTable:
    def __init__(self, path, columns):
        self.stream = path.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=columns)
        self.writer.writeheader()
        self.count = 0

    def rows(self, rows):
        self.writer.writerows(rows)
        self.count += len(rows)
        self.stream.flush()

    def close(self):
        self.stream.close()


def cache_batches(cases, recipe, size, device, dtype):
    return [pack_cases(cases[start:start + size], recipe, device, dtype)
            for start in range(0, len(cases), size)]


def choose_workers(config, hardware, requested):
    if requested != "auto":
        workers = int(requested)
        if workers < 1 or workers > available_cpus(hardware):
            raise ValueError("workers must fit the allocated CPU count")
        start = time.perf_counter()
        cases = prepare_cases(config, workers)
        return workers, cases, [{"workers": workers, "seconds": time.perf_counter() - start,
                                "graphs": len(cases), "selection": "explicit allocation"}]
    candidates = sorted({1, *(n for n in (2, 4, 8, 16) if n <= available_cpus(hardware))})
    trials, best_cases, best_rate, selected = [], None, -math.inf, None
    for workers in candidates:
        start = time.perf_counter()
        cases = prepare_cases(config, workers)
        seconds = time.perf_counter() - start
        rate = len(cases) / seconds
        trials.append({"workers": workers, "graphs": len(cases), "seconds": seconds, "graphs_per_second": rate})
        print(f"[CPU calibration] workers={workers} graphs={len(cases)} seconds={seconds:.3f} graphs/s={rate:.1f}", flush=True)
        if rate > best_rate:
            selected, best_cases, best_rate = workers, cases, rate
    return selected, best_cases, trials


def prepare_references(cases, config, workers, output):
    """Fixed CPU float64 teachers are produced before student optimization."""
    from dataclasses import fields
    folder = output / "geometry"
    folder.mkdir(exist_ok=False)
    chunks = 4096

    def prepare(case):
        references, rows = {}, []
        x = case.graph.features.T[None, ..., None]
        geometries = prepare_geometries(case.topology)
        for recipe in RECIPES:
            g = geometries[recipe]
            arrays = {field.name: getattr(g, field.name).numpy()
                      for field in fields(g) if isinstance(getattr(g, field.name), torch.Tensor)}
            np.savez_compressed(folder / f"{case.graph_id}--{recipe}.npz", **arrays)
            ghash = hashlib.sha256()
            for name, array in sorted(arrays.items()):
                ghash.update(name.encode()); ghash.update(str(array.dtype).encode())
                ghash.update(np.asarray(array.shape, dtype="<i8").tobytes()); ghash.update(array.tobytes())
            rows.append({"graph_id": case.graph_id, "recipe": recipe,
                         "physical_nodes": g.n, "physical_edges": g.num_edges,
                         "occurrences": g.num_occurrences, "eligible_pairs": g.num_pairs,
                         "geometry_sha256": ghash.hexdigest(), "metadata": g.metadata})
            for target in TARGETS:
                value = target_message(g, x, target, pair_chunk=chunks).detach()
                if not bool(torch.isfinite(value).all()):
                    raise ArithmeticError(f"nonfinite fixed teacher: {case.graph_id}/{recipe}/{target}")
                references[recipe, target] = value
        return case.graph_id, references, rows

    refs, rows = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (graph_id, reference, geometry_rows) in enumerate(executor.map(prepare, cases), 1):
            refs[graph_id] = reference
            rows.extend(geometry_rows)
            if index % 50 == 0 or index == len(cases):
                print(f"[teacher preparation] graphs={index}/{len(cases)} recipes=2 targets=3 float64 all scalar draws", flush=True)
    write_json(output / "geometry_manifest.json", {"all_eligible_pairs_used": True, "rows": rows})
    # Stored targets are evidence, never exposed as gate inputs or coefficient supervision.
    save_checkpoint(output / "teacher_messages.pt", {key: {f"{r}__{t}": value for (r, t), value in val.items()}
                                                     for key, val in refs.items()})
    return refs, rows


def batch_targets(batches, refs, recipe, target):
    return [torch.cat([refs[case.graph_id][recipe, target] for case in batch.cases], dim=-2).to(batch.x)
            for batch in batches]


def train_epoch(model, batches, targets, optimizer, total_graphs, epsilon, pair_chunk):
    optimizer.zero_grad(set_to_none=True)
    loss_total = batches[0].x.new_zeros(len(model.seeds))
    for batch, target in zip(batches, targets, strict=True):
        prediction, _ = model(batch, pair_chunk=pair_chunk)
        loss = normalized_mse(prediction, target, batch.geometry, epsilon)
        contribution = loss * (batch.num_graphs / total_graphs)
        # Sum across independent seed models; do not divide their gradients by S.
        contribution.sum().backward()
        loss_total += contribution.detach()
    gradients, parameters = parameter_vector_stats(model)
    if not bool(torch.isfinite(loss_total).all() & torch.isfinite(gradients).all()):
        raise ArithmeticError("nonfinite raw-message loss or gradient")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer.step()
    updates = torch.cat([(value.detach() - before[name]).reshape(len(model.seeds), -1)
                         for name, value in model.named_parameters()], dim=1).norm(dim=1)
    return torch.stack((loss_total, gradients, parameters, updates), -1).cpu().tolist()


def _measure_packing(cases, refs, config, recipe, size, pair_chunk, device, dtype, resident_cases=None):
    # Keep a trial's CUDA references in its own frame: a genuine OOM unwinds
    # them before the caller releases allocator cache and tries exact chunks.
    if resident_cases is None:
        resident = {"train": cache_batches(cases, recipe, size, device, dtype)}
    else:
        # Match the static all-split geometry cache of the real job, not just
        # an artificially cheap train-only allocation.
        resident = {split: cache_batches([case for case in resident_cases if case.split == split],
                                        recipe, size, device, dtype) for split in SPLITS}
    batches = resident["train"]
    targets = batch_targets(batches, refs, "local_degree", "analytic_pair")
    model = RawStudent("F2", config["model_seeds"], config["hidden"], config["epsilon"]).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    train_epoch(model, batches, targets, optimizer, len(cases), config["loss_epsilon"], pair_chunk)
    synchronize(device)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {"physical_graph_batch": size, "pair_chunk": pair_chunk, "recipe": recipe,
            "graphs": len(cases), "realizations": config["features"],
            "static_graphs_resident": sum(batch.num_graphs for group in resident.values() for batch in group),
            "parallel_seed_models": len(config["model_seeds"]),
            "seconds_per_epoch": seconds, "graphs_per_second": len(cases) / seconds,
            "peak_vram_bytes": peak, "status": "measured",
            "scope": "temporary calibration only; every train graph; no scientific checkpoint"}


def choose_packing(cases, refs, config, device, dtype, requested, requested_chunk, resident_cases=None):
    candidates = sorted({min(n, len(cases)) for n in (8, 32, 64, 128, len(cases))})
    if requested != "auto":
        candidates = [int(requested)]
        if candidates[0] < 1:
            raise ValueError("physical batch must be positive")
    chunks = (1024, 4096, 16384, 65536, 262144, 1 << 30) if requested_chunk == "auto" else (int(requested_chunk),)
    if min(chunks) < 1:
        raise ValueError("pair chunk must be positive")
    trials = []
    for size in candidates:
        for pair_chunk in chunks:
            for recipe in RECIPES:
                try:
                    row = _measure_packing(cases, refs, config, recipe, size, pair_chunk, device, dtype, resident_cases)
                    trials.append(row)
                    print(f"[GPU calibration] recipe={recipe} batch={size} pair_chunk={pair_chunk} epoch={row['seconds_per_epoch']:.3f}s peak={row['peak_vram_bytes']}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    if device.type != "cuda":
                        raise
                    trials.append({"physical_graph_batch": size, "pair_chunk": pair_chunk, "recipe": recipe,
                                   "status": "OOM", "scope": "temporary calibration; graph/model scale preserved"})
                    print(f"[GPU calibration OOM] recipe={recipe} batch={size} pair_chunk={pair_chunk}; every graph retained", flush=True)
                    # Only allocator cache belonging to this process is released.
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
    scores = []
    for size in candidates:
        for chunk in chunks:
            matching = [row for row in trials if row["physical_graph_batch"] == size and row["pair_chunk"] == chunk]
            if len(matching) == len(RECIPES) and all(row["status"] == "measured" for row in matching):
                scores.append((max(row["seconds_per_epoch"] for row in matching), size, chunk))
    if not scores:
        raise RuntimeError("no measured packing fits; model/data were preserved; inspect calibration trials")
    _, selected_size, selected_chunk = min(scores)
    return selected_size, selected_chunk, trials


@torch.no_grad()
def validation_scores(model, batches, targets, epsilon, pair_chunk):
    values = batches[0].x.new_zeros(len(model.seeds))
    total = sum(batch.num_graphs for batch in batches)
    for batch, target in zip(batches, targets, strict=True):
        prediction, _ = model(batch, pair_chunk=pair_chunk)
        values += normalized_mse(prediction, target, batch.geometry, epsilon) * (batch.num_graphs / total)
    return values


def checkpoint_record(model, optimizer, config, source, manifest_digest, job, epoch,
                      best_state, scores, best_epochs, history, batch_size, pair_chunk):
    return {"format_version": 1, "profile": config["profile"], "config_digest": digest(config),
            "source_digest": source["code_digest"], "data_manifest_digest": manifest_digest,
            "job": job, "epoch": epoch, "model": cpu_state(model.state_dict()),
            "optimizer": cpu_state(optimizer.state_dict()), "best_model": cpu_state(best_state),
            "best_scores": scores.cpu(), "best_epochs": best_epochs.cpu(), "history": history,
            "physical_graph_batch": batch_size, "pair_chunk": pair_chunk,
            "seed_optimizer_updates": epoch * len(model.seeds)}


def load_resume(folder, job, config, source, manifest_digest, device):
    if folder is None:
        return None
    options = [*(folder / "checkpoints").glob(f"{job}--epoch-*.pt")]
    selected = folder / "checkpoints" / f"{job}--selected.pt"
    if selected.is_file():
        options.append(selected)
    if not options:
        return None
    path = selected if selected.is_file() else sorted(options)[-1]
    saved = torch.load(path, map_location=device, weights_only=True)
    for name, expected in (("config_digest", digest(config)), ("source_digest", source["code_digest"]),
                           ("data_manifest_digest", manifest_digest), ("job", job)):
        if saved[name] != expected:
            raise ValueError(f"resume {name} differs; preserve the previous output and use matching source/data")
    saved["resume_file"] = str(path)
    return saved


def train_job(model, train_batches, train_targets, val_batches, val_targets, config, source,
              manifest_digest, job, output, batch_size, pair_chunk, resume):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_scores = validation_scores(model, val_batches, val_targets, config["loss_epsilon"], pair_chunk)
    best_epochs = torch.zeros(len(model.seeds), device=best_scores.device, dtype=torch.long)
    best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    history, first_epoch = [], 0
    if resume is not None:
        if resume["physical_graph_batch"] != batch_size or resume["pair_chunk"] != pair_chunk:
            # Exact effective gradients remain valid, but repeatability requires the stored numerical order.
            raise ValueError("resume packing differs; provide recorded --batch-size and --pair-chunk")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        best_scores = resume["best_scores"].to(best_scores)
        best_epochs = resume["best_epochs"].to(best_epochs)
        best_state = {name: value.to(best_state[name]) for name, value in resume["best_model"].items()}
        history, first_epoch = list(resume["history"]), int(resume["epoch"])
        print(f"[resume] {job} epoch={first_epoch}/{config['epochs']}", flush=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    total_graphs = sum(batch.num_graphs for batch in train_batches)
    started = time.perf_counter()
    elapsed = 0.
    for epoch in range(first_epoch + 1, config["epochs"] + 1):
        model.train()
        values = train_epoch(model, train_batches, train_targets, optimizer, total_graphs,
                             config["loss_epsilon"], pair_chunk)
        evaluated = epoch % config["validation_every"] == 0 or epoch == config["epochs"]
        if evaluated:
            scores = validation_scores(model, val_batches, val_targets, config["loss_epsilon"], pair_chunk)
            better = scores < best_scores
            best_scores = torch.where(better, scores, best_scores)
            best_epochs = torch.where(better, best_epochs.new_full(best_epochs.shape, epoch), best_epochs)
            for name, parameter in model.state_dict().items():
                mask = better.reshape((-1,) + (1,) * (parameter.ndim - 1))
                best_state[name] = torch.where(mask, parameter.detach(), best_state[name])
            validation = scores.cpu().tolist()
        else:
            validation = [None] * len(model.seeds)
        elapsed = time.perf_counter() - started
        for index, seed in enumerate(model.seeds):
            history.append({"job": job, "seed": seed, "epoch": epoch,
                            "train_nmse": values[index][0], "gradient_l2": values[index][1],
                            "parameter_l2": values[index][2], "optimizer_delta_l2": values[index][3],
                            "validation_nmse": validation[index], "seconds_this_execution": elapsed,
                            "scope": "raw_message_training", "all_train_graphs_used": total_graphs})
        if evaluated or epoch == first_epoch + 1:
            print(f"[progress] {job} epoch={epoch}/{config['epochs']} loss={[round(row[0], 6) for row in values]} "
                  f"val={validation} grad={[round(row[1], 6) for row in values]} "
                  f"update={[round(row[3], 6) for row in values]} elapsed={elapsed:.1f}s", flush=True)
        if epoch % 50 == 0 or epoch == config["epochs"]:
            save_checkpoint(checkpoints / f"{job}--epoch-{epoch:04d}.pt", checkpoint_record(
                model, optimizer, config, source, manifest_digest, job, epoch,
                best_state, best_scores, best_epochs, history, batch_size, pair_chunk))
    selected = checkpoint_record(model, optimizer, config, source, manifest_digest, job, config["epochs"],
                                 best_state, best_scores, best_epochs, history, batch_size, pair_chunk)
    selected["selection"] = "lowest validation equal-graph independent-scalar NMSE; earliest tie; initial epoch0 eligible"
    save_checkpoint(checkpoints / f"{job}--selected.pt", selected)
    model.load_state_dict(best_state)
    return history, best_epochs.cpu().tolist(), (config["epochs"] - first_epoch) * len(model.seeds), elapsed


def all_student_jobs():
    return [(recipe, teacher, target, condition) for recipe in RECIPES
            for teacher in RECIPES for target in TARGETS for condition in CONDITIONS]


def allocate_student_jobs(gpu_count):
    if type(gpu_count) is not int or not 1 <= gpu_count <= len(all_student_jobs()):
        raise ValueError("GPU allocation must fit the number of independent scientific jobs")
    # Trainable groups come first so D0/F0 do not cause a systematic workload
    # imbalance for two GPUs. This does not change any seed or data stream.
    ordered = sorted(all_student_jobs(), key=lambda job: job[3] not in TRAINABLE)
    shards = [[] for _ in range(gpu_count)]
    for index, job in enumerate(ordered):
        shards[index % gpu_count].append(job)
    if {job for shard in shards for job in shard} != set(all_student_jobs()):
        raise AssertionError("multi-GPU job allocation omitted a condition")
    return shards


def _shared_payload(cases, refs):
    from dataclasses import fields
    payload = []
    for case in cases:
        graph = case.graph
        top = {field.name: getattr(case.topology, field.name) for field in fields(case.topology)}
        top = {name: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value for name, value in top.items()}
        payload.append({"graph_id": graph.graph_id, "family": graph.family, "num_nodes": graph.num_nodes,
                        "edges": graph.edges, "features": graph.features, "graph_seed": graph.graph_seed,
                        "feature_seed": graph.feature_seed, "base_tree_seed": graph.base_tree_seed,
                        "split": case.split, "topology": top, "metadata": case.metadata})
    def primitives(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {name: primitives(item) for name, item in value.items()}
        if isinstance(value, tuple):
            return tuple(primitives(item) for item in value)
        if isinstance(value, list):
            return [primitives(item) for item in value]
        return value
    return primitives({"cases": payload, "refs": {key: {f"{r}__{t}": value for (r, t), value in val.items()}
                                                       for key, val in refs.items()}})


def _load_shared(folder, config, source):
    from ...local_energy_relations.topology import LocalTopology
    from ...wedge_propagation.data import GraphCase
    from .data import SyntheticCase
    manifest = json.loads((folder / "worker_input_manifest.json").read_text(encoding="utf-8"))
    cache = folder / "worker_inputs.pt"
    if manifest["source_digest"] != source["code_digest"] or manifest["config_digest"] != digest(config):
        raise ValueError("worker source/config differs from immutable parent inputs")
    if hashlib.sha256(cache.read_bytes()).hexdigest() != manifest["cache_sha256"]:
        raise ValueError("worker input cache content changed")
    saved = torch.load(cache, map_location="cpu", weights_only=True)
    cases = []
    for value in saved["cases"]:
        graph = GraphCase(**{key: value[key] for key in ("graph_id", "family", "num_nodes", "edges", "features",
                                                      "graph_seed", "feature_seed", "base_tree_seed")})
        cases.append(SyntheticCase(graph, value["split"], LocalTopology(**value["topology"]), value["metadata"]))
    refs = {key: {tuple(label.split("__")): val for label, val in values.items()} for key, values in saved["refs"].items()}
    if digest(data_manifest(cases, config)) != manifest["data_manifest_digest"]:
        raise ValueError("worker data membership or input content differs")
    return cases, refs, manifest["data_manifest_digest"]


def run_student_jobs(assignments, cases, refs, config, source, manifest_digest, output,
                     device, dtype, batch_size, pair_chunk, resume_from, graph_table, draw_table):
    splits = {split: [case for case in cases if case.split == split] for split in SPLITS}
    rows, history, jobs, new_updates, resources = [], [], [], 0, []
    for recipe in RECIPES:
        current = [job for job in assignments if job[0] == recipe]
        if not current:
            continue
        batches = {split: cache_batches(splits[split], recipe, batch_size, device, dtype) for split in SPLITS}
        cached_targets = {}
        for _, teacher, target, condition in current:
            key = (teacher, target)
            if key not in cached_targets:
                cached_targets[key] = {split: batch_targets(batches[split], refs, teacher, target) for split in SPLITS}
            targets = cached_targets[key]
            seeds = config["model_seeds"] if condition in TRAINABLE else [-1]
            model = RawStudent(condition, seeds, config["hidden"], config["epsilon"]).to(device=device, dtype=dtype)
            job = f"{teacher}--{target}--{recipe}--{condition}"
            parameters = sum(value.numel() for value in model.parameters()) // len(seeds)
            print(f"[job] {job} parameters_per_seed={parameters} seed_models={len(seeds)} fixed={condition not in TRAINABLE}", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if condition in TRAINABLE:
                resume = load_resume(resume_from, job, config, source, manifest_digest, device)
                measured, selected, updates, seconds = train_job(
                    model, batches["train"], targets["train"], batches["validation"], targets["validation"],
                    config, source, manifest_digest, job, output, batch_size, pair_chunk, resume)
                history.extend(measured); new_updates += updates
            else:
                selected, updates, seconds = None, 0, 0.
            meta = {"teacher_recipe": teacher, "target": target, "student_recipe": recipe, "condition": condition}
            for split in SPLITS:
                measured_rows = evaluate_job(model, batches[split], targets[split], meta, config["loss_epsilon"],
                                             pair_chunk, graph_table, draw_table)
                rows.extend(measured_rows)
                print(f"[evaluate] {job} split={split} graphs={len(splits[split])} mean_graph_nmse={np.mean([row['nmse'] for row in measured_rows]):.6g}", flush=True)
            jobs.append(dict(meta, seeds=seeds, parameters_per_seed=parameters, selected_epochs=selected,
                             optimizer_updates_per_seed=config["epochs"] if condition in TRAINABLE else 0,
                             optimizer="Adam" if condition in TRAINABLE else None))
            resources.append({"job": job, "parameters_per_seed": parameters,
                              "seconds_this_execution": seconds, "new_seed_optimizer_updates": updates,
                              "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
            del model
        del cached_targets, batches
    return rows, history, jobs, new_updates, resources


def _stream_worker(command, environment, ordinal):
    # Only this task's Python worker is started. A failed worker is allowed to
    # finish normally; no session or sibling process receives a kill signal.
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="strict", bufsize=1)
    for line in process.stdout:
        print(f"[GPU {ordinal}] {line.rstrip()}", flush=True)
    code = process.wait()
    return ordinal, code


def _append_csv_table(path, table, columns, numeric=()):
    collected, chunk = [], []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError("worker table column contract differs")
        for row in reader:
            if numeric:
                for name in numeric:
                    row[name] = float(row[name]) if name in ("nmse", "relative_l2_epsilon_mean", "absolute_l2_mean", "target_l2_mean") else int(row[name])
                collected.append(row)
            chunk.append(row)
            if len(chunk) == 4096:
                table.rows(chunk); chunk = []
        if chunk:
            table.rows(chunk)
    return collected


def dispatch_gpu_workers(args, output, cases, refs, config, source, manifest_digest,
                         gpu_count, workers, graph_table, draw_table):
    assignments = allocate_student_jobs(gpu_count)
    save_checkpoint(output / "worker_inputs.pt", _shared_payload(cases, refs))
    write_json(output / "worker_input_manifest.json", {"source_digest": source["code_digest"],
        "config_digest": digest(config), "data_manifest_digest": manifest_digest,
        "cache_sha256": hashlib.sha256((output / "worker_inputs.pt").read_bytes()).hexdigest(),
        "format": "torch weights_only tensor and primitive immutable cache", "all_graphs": len(cases)})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = visible.split(",") if visible else [str(index) for index in range(gpu_count)]
    if len(tokens) != gpu_count:
        raise ValueError("visible GPU tokens and actual CUDA allocation disagree")
    if args.resume_from is not None:
        prior = json.loads((args.resume_from / "multi_gpu_plan.json").read_text(encoding="utf-8"))
        if prior.get("gpu_count_used") != gpu_count:
            raise ValueError("multi-GPU resume requires the same worker allocation; previous results are preserved")
    write_json(output / "multi_gpu_plan.json", {"gpu_count_used": gpu_count, "assignments": assignments,
        "source_digest": source["code_digest"], "config_digest": digest(config),
        "data_manifest_digest": manifest_digest, "immutable_shared_inputs": True})
    workers_folder = output / "workers"; workers_folder.mkdir(exist_ok=False)
    calls = []
    for ordinal, assigned in enumerate(assignments):
        plan = output / f"worker_plan_{ordinal:02d}.json"
        write_json(plan, {"ordinal": ordinal, "jobs": assigned,
                          "source_digest": source["code_digest"], "all_data_shared": True})
        command = [sys.executable, "-B", "-u", "-m", "research.edge_metric_relations.synthetic.study",
                   "--profile", args.profile, "--device", "cuda", "--output-dir", str(workers_folder / f"gpu-{ordinal:02d}"),
                   "--shared-prepared-dir", str(output), "--worker-plan", str(plan),
                   "--workers", str(max(1, workers // gpu_count)), "--batch-size", args.batch_size,
                   "--pair-chunk", args.pair_chunk]
        prior_worker = args.resume_from / "workers" / f"gpu-{ordinal:02d}" if args.resume_from else None
        if prior_worker is not None and prior_worker.is_dir():
            command += ["--resume-from", str(prior_worker)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = tokens[ordinal]
        environment["PYTHONIOENCODING"] = "utf-8"
        calls.append((command, environment, ordinal))
    print(f"[multi GPU] assigned_gpus={gpu_count} independent_packed_jobs={sum(map(len, assignments))} shared_graphs={len(cases)} all_scalar_draws={config['features']}", flush=True)
    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        results = list(executor.map(lambda call: _stream_worker(*call), calls))
    failed = [(ordinal, code) for ordinal, code in results if code]
    if failed:
        raise RuntimeError(f"synthetic GPU workers failed: {failed}; inspect their terminal logs; sibling sessions were preserved")
    rows, history, jobs, updates, resource_rows, worker_resources = [], [], [], 0, [], []
    for ordinal in range(gpu_count):
        folder = workers_folder / f"gpu-{ordinal:02d}"
        record = json.loads((folder / "worker_completion.json").read_text(encoding="utf-8"))
        if not record["completed"] or record["source_digest"] != source["code_digest"] or record["jobs"] != [list(job) for job in assignments[ordinal]]:
            raise ValueError("worker scientific job/source coverage differs")
        rows.extend(_append_csv_table(folder / "per_graph.csv", graph_table, GRAPH_COLUMNS,
            ("seed", "nodes", "nmse", "relative_l2_epsilon_mean", "absolute_l2_mean", "target_l2_mean", "zero_target_draws", "draws")))
        _append_csv_table(folder / "per_realization.csv", draw_table, DRAW_COLUMNS)
        result = json.loads((folder / "worker_results.json").read_text(encoding="utf-8"))
        history.extend(result["history"]); jobs.extend(result["job_records"]); updates += result["new_updates"]
        resource_rows.extend([dict(row, gpu_ordinal=ordinal) for row in result["resources"]])
        worker_resources.append(json.loads((folder / "runtime_packing.json").read_text(encoding="utf-8")))
    return rows, history, jobs, updates, resource_rows, worker_resources


def worker_run(args, output):
    config = read_config(args.config or FOLDER / f"config_{args.profile}.json", args.profile)
    require_full_server(args.profile, torch.device(args.device))
    source = source_manifest()
    plan = json.loads(args.worker_plan.read_text(encoding="utf-8"))
    assignments = [tuple(job) for job in plan["jobs"]]
    if plan["source_digest"] != source["code_digest"] or any(job not in all_student_jobs() for job in assignments):
        raise ValueError("worker job assignment/source differs")
    if len(assignments) != len(set(assignments)):
        raise ValueError("duplicate worker scientific jobs")
    device = torch.device(args.device); hardware = runtime_resources(device)
    if device.type != "cuda" or hardware["gpu_count_visible"] != 1:
        raise ValueError("worker must receive exactly one assigned GPU")
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, config["dtype"])
    cases, refs, manifest_digest = _load_shared(args.shared_prepared_dir, config, source)
    checks = run_math_checks(device)
    write_json(output / "source_manifest.json", source); write_json(output / "math_checks.json", checks)
    write_json(output / "hardware.json", hardware)
    train_cases = [case for case in cases if case.split == "train"]
    requested_batch, requested_chunk = args.batch_size, args.pair_chunk
    if args.resume_from is not None and (args.resume_from / "runtime_packing.json").is_file():
        prior = json.loads((args.resume_from / "runtime_packing.json").read_text(encoding="utf-8"))
        requested_batch = str(prior["physical_graph_batch"]) if requested_batch == "auto" else requested_batch
        requested_chunk = str(prior["pair_chunk"]) if requested_chunk == "auto" else requested_chunk
    batch_size, pair_chunk, trials = choose_packing(train_cases, refs, config, device, dtype, requested_batch, requested_chunk, cases)
    write_json(output / "runtime_packing.json", {"physical_graph_batch": batch_size, "pair_chunk": pair_chunk,
        "packing_trials": trials, "worker_ordinal": plan["ordinal"], "gpu_count_used": 1,
        "source_digest": source["code_digest"], "config_digest": digest(config), "data_manifest_digest": manifest_digest})
    graph_table, draw_table = ResultTable(output / "per_graph.csv", GRAPH_COLUMNS), ResultTable(output / "per_realization.csv", DRAW_COLUMNS)
    try:
        rows, history, records, updates, resources = run_student_jobs(
            assignments, cases, refs, config, source, manifest_digest, output, device, dtype,
            batch_size, pair_chunk, args.resume_from, graph_table, draw_table)
    finally:
        graph_table.close(); draw_table.close()
    expected_runs = sum(len(config["model_seeds"]) if job[3] in TRAINABLE else 1 for job in assignments)
    if graph_table.count != expected_runs * len(cases) or draw_table.count != expected_runs * len(cases) * config["features"]:
        raise AssertionError("worker graph/scalar coverage differs")
    assert_source_unchanged(source)
    write_json(output / "worker_results.json", {"history": history, "job_records": records,
        "new_updates": updates, "resources": resources})
    write_json(output / "worker_completion.json", {"completed": True, "source_digest": source["code_digest"],
        "data_manifest_digest": manifest_digest, "math_checks_passed": checks["passed"],
        "jobs": assignments, "graph_rows": graph_table.count, "draw_rows": draw_table.count})
    print(f"[worker complete] ordinal={plan['ordinal']} jobs={len(assignments)} new_updates={updates}", flush=True)


BASE_COLUMNS = ["teacher_recipe", "target", "student_recipe", "condition", "seed", "split", "graph_id", "family", "nodes"]
GRAPH_COLUMNS = BASE_COLUMNS + ["nmse", "relative_l2_epsilon_mean", "absolute_l2_mean", "target_l2_mean", "zero_target_draws", "draws"]
DRAW_COLUMNS = BASE_COLUMNS + ["draw", "nmse", "relative_l2_epsilon", "absolute_l2", "target_l2", "zero_target"]


@torch.no_grad()
def evaluate_job(model, batches, targets, meta, epsilon, pair_chunk, graph_table, draw_table,
                 baseline_coefficients=None):
    rows = []
    for batch, target in zip(batches, targets, strict=True):
        if baseline_coefficients is None:
            prediction, _ = model(batch, pair_chunk=pair_chunk)
            seeds = model.seeds
        else:
            prediction = baseline_predict(batch, meta["condition"], baseline_coefficients)
            seeds = (-1,)
        errors = {name: value.cpu() for name, value in graph_draw_errors(prediction, target, batch.geometry, epsilon).items()}
        graphs, draws = [], []
        for index, case in enumerate(batch.cases):
            for si, seed in enumerate(seeds):
                base = dict(meta, seed=seed, split=case.split, graph_id=case.graph_id,
                            family=case.graph.family, nodes=case.graph.num_nodes)
                graph = dict(base, nmse=float(errors["nmse"][si, :, index].mean()),
                             relative_l2_epsilon_mean=float(errors["relative_l2_epsilon"][si, :, index].mean()),
                             absolute_l2_mean=float(errors["absolute_l2"][si, :, index].mean()),
                             target_l2_mean=float(errors["target_l2"][0, :, index].mean()),
                             zero_target_draws=int(errors["zero_target"][0, :, index].sum()),
                             draws=prediction.shape[1])
                graphs.append(graph)
                for draw in range(prediction.shape[1]):
                    draws.append(dict(base, draw=draw, nmse=float(errors["nmse"][si, draw, index]),
                                      relative_l2_epsilon=float(errors["relative_l2_epsilon"][si, draw, index]),
                                      absolute_l2=float(errors["absolute_l2"][si, draw, index]),
                                      target_l2=float(errors["target_l2"][0, draw, index]),
                                      zero_target=bool(errors["zero_target"][0, draw, index])))
        graph_table.rows(graphs); draw_table.rows(draws)
        rows.extend(graphs)
    return rows


def paired_comparisons(rows, config):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != "train" and row["condition"] in CONDITIONS:
            key = tuple(row[name] for name in ("teacher_recipe", "target", "student_recipe", "condition", "seed", "split"))
            grouped[key].append(row["nmse"])
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    result = []
    seeds = config["model_seeds"]
    for teacher in RECIPES:
        for target in TARGETS:
            for recipe in RECIPES:
                for split in SPLITS[1:]:
                    for candidate, reference in CONTRASTS:
                        differences = np.asarray([
                            means[teacher, target, recipe, candidate, seed, split]
                            - means[teacher, target, recipe, reference, -1 if reference not in TRAINABLE else seed, split]
                            for seed in seeds])
                        mean, sd, count = float(differences.mean()), float(differences.std(ddof=1)), len(seeds)
                        se = sd / math.sqrt(count)
                        half = float(student_t.ppf(.975, count - 1)) * se
                        p = float(2 * student_t.sf(abs(mean / se), count - 1)) if se else (0. if mean else 1.)
                        result.append({"teacher_recipe": teacher, "target": target, "student_recipe": recipe,
                                       "split": split, "candidate": candidate, "reference": reference,
                                       "metric": "equal-graph NMSE", "delta": mean, "sample_std": sd,
                                       "ci95_low": mean - half, "ci95_high": mean + half,
                                       "paired_seeds": count, "p_two_sided": p,
                                       "family": "all_360_B_gate_message_contrasts", "alpha": .05,
                                       "scope": "synthetic message recovery; no classification/weight-identifiability claim"})
    order = sorted(range(len(result)), key=lambda index: result[index]["p_two_sided"])
    previous = 0.
    for rank, index in enumerate(order):
        adjusted = min(1., max(previous, (len(result) - rank) * result[index]["p_two_sided"]))
        result[index]["p_holm"] = adjusted
        result[index]["holm_reject"] = adjusted <= .05
        previous = adjusted
    return result


def write_report(output, rows, comparisons, contract):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in ("teacher_recipe", "target", "student_recipe", "condition", "seed", "split"))].append(row["nmse"])
    summary = [dict(zip(("teacher_recipe", "target", "student_recipe", "condition", "seed", "split"), key),
                    mean_graph_nmse=float(np.mean(values)), graphs=len(values))
               for key, values in grouped.items()]
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", comparisons)
    lines = ["# 합성 실험 B: raw 메시지 규칙 학습", "",
             f"profile: **{contract['profile']}**. 전체 선언 그래프 {contract['graphs']}개와 scalar 입력 {contract['scalar_inputs']}개를 사용했다.", "",
             "학생은 Qθ(X)X를 직접 예측한다. 분류용 정규화·node projection·smoothing은 사용하지 않는다.",
             "교사 recipe와 학생 recipe는 각각 unit/local_degree로 독립해서 모두 비교한다.", "",
             f"학습 run {contract['budget']['learned_runs']}회, 예정 seed optimizer 갱신 {contract['budget']['seed_optimizer_updates']:,}회, 이번 실행의 새 갱신 {contract['new_seed_optimizer_updates']:,}회.",
             "D0/F0는 고정 모델이다. 네 수식 기준선은 train 메시지에만 least squares로 적합하며 optimizer를 사용하지 않는다.",
             "각 epoch는 전체 train 그래프와 모든 scalar 입력의 평균을 사용한다. 물리 batch는 gradient accumulation 후 한 번 갱신한다.", "",
             "## 해석 범위", "",
             "- diagonal 및 diagonal_squared의 SAME-teacher-Ld 기준선은 구현과 target 정의의 양성 대조다.",
             "- analytic_pair는 고정한 비선형 교사에 대한 메시지 회복 문제다. 학습된 행렬의 유일한 식별이나 분류 성능을 입증하지 않는다.",
             "- bounded 학생은 Qθ≤3Q0이다. raw Ld² 교사는 이 크기 범위를 넘을 수 있으므로 diagonal_squared의 실패만으로 pair 관계의 기여를 부정할 수 없다.",
             "- 같은 labeled topology의 중복은 검사한다. 그래프 동형류의 독립성은 주장하지 않는다.",
             "- 360개 paired 비교의 two-sided p-value에 하나의 Holm 보정을 적용한다. CI는 개별 paired t 구간이다.",
             "- zero target은 epsilon으로 명시적으로 처리하고 per-draw 표에 표시한다.",
             "- DEBUG는 별도의 전체 파이프라인 검사이며 FULL 성능 결과로 쓰지 않는다.", "",
             "## 파일", "",
             "per_graph.csv / per_realization.csv: 모든 graph·draw·seed의 실제 오차.",
             "training.csv / checkpoints/: loss·gradient·optimizer 변화 및 validation 선택 상태.",
             "teacher_messages.pt / graphs/ / geometry/: 사용한 입력·target·모든 occurrence와 eligible pair.",
             "data_manifest.json / source_manifest.json / contract.json / coverage.json / resources.json: 재현 조건과 예산.", ""]
    (output / "SYNTHETIC_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ri, teacher in enumerate(RECIPES):
        for ti, target in enumerate(TARGETS):
            ax = axes[ri, ti]
            for recipe in RECIPES:
                values = [np.mean([row["mean_graph_nmse"] for row in summary
                                   if row["teacher_recipe"] == teacher and row["target"] == target
                                   and row["student_recipe"] == recipe and row["condition"] == condition
                                   and row["split"] == "id"]) for condition in CONDITIONS]
                ax.plot(CONDITIONS, np.maximum(values, 1e-16), marker="o", label=recipe)
            ax.set_yscale("log"); ax.set_title(f"teacher={teacher}: {target}")
            ax.set_ylabel("ID mean graph NMSE"); ax.grid(alpha=.2)
    axes[0, 0].legend()
    folder = output / "figures"; folder.mkdir(exist_ok=False)
    fig.savefig(folder / "raw_message_recovery.png", dpi=180)
    fig.savefig(folder / "raw_message_recovery.pdf")
    plt.close(fig)


def run(args, output):
    started = time.perf_counter()
    config = read_config(args.config or FOLDER / f"config_{args.profile}.json", args.profile)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("only an explicit CPU or CUDA device is supported")
    require_full_server(args.profile, device)
    hardware = runtime_resources(device)
    gpu_count = hardware["gpu_count_visible"] if device.type == "cuda" else 0
    hardware["gpu_count_used"] = gpu_count
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, config["dtype"])
    source = source_manifest()
    write_json(output / "source_manifest.json", source)
    write_json(output / "config.json", config)
    write_json(output / "hardware.json", hardware)
    print(f"[start] experiment=B profile={args.profile} device={device} dtype={dtype} raw Q(X)X hidden={config['hidden']} epochs={config['epochs']}", flush=True)
    print(f"[hardware] {json.dumps(hardware, ensure_ascii=False)}", flush=True)
    math_checks = run_math_checks(device)
    write_json(output / "math_checks.json", math_checks)
    print(f"[math checks] dense_teacher_checks={math_checks['dense_teacher_checks']} error={math_checks['max_dense_absolute_error']:.3g} "
          f"DEBUG fixture gradient={math_checks['gradient_abs_max']:.3g} update={math_checks['optimizer_change_abs_max']:.3g}; scientific updates=0", flush=True)
    workers, cases, cpu_trials = choose_workers(config, hardware, args.workers)
    manifest = data_manifest(cases, config)
    expected_splits = {"train": 240, "validation": 60, "id": 120, "size_ood": 90, "family_ood": 12, "family_size_ood": 9}
    if args.profile == "full" and manifest["split_graph_counts"] != expected_splits:
        raise AssertionError("FULL graph membership differs from the 531-graph contract")
    manifest_digest = digest(manifest)
    write_json(output / "data_manifest.json", manifest)
    save_dataset(output / "graphs", cases)
    planned = budget(config, len(cases))
    print(f"[data] graphs={len(cases)} inputs={manifest['scalar_inputs']} split_counts={manifest['split_graph_counts']} no_sampling fraction=1", flush=True)
    print(f"[budget] {json.dumps(planned)}", flush=True)
    refs, geometry_rows = prepare_references(cases, config, workers, output)
    splits = {split: [case for case in cases if case.split == split] for split in SPLITS}
    if gpu_count <= 1:
        requested_batch, requested_chunk = args.batch_size, args.pair_chunk
        if args.resume_from is not None and (args.resume_from / "runtime_packing.json").is_file():
            old_resources = json.loads((args.resume_from / "runtime_packing.json").read_text(encoding="utf-8"))
            requested_batch = str(old_resources["physical_graph_batch"]) if requested_batch == "auto" else requested_batch
            requested_chunk = str(old_resources["pair_chunk"]) if requested_chunk == "auto" else requested_chunk
        batch_size, pair_chunk, packing_trials = choose_packing(splits["train"], refs, config, device, dtype,
                                                              requested_batch, requested_chunk, cases)
        print(f"[selected] physical_graph_batch={batch_size} pair_chunk={pair_chunk} effective_train_graphs={len(splits['train'])} all_draws={config['features']} parallel_seeds={len(config['model_seeds'])}", flush=True)
        write_json(output / "runtime_packing.json", {"physical_graph_batch": batch_size, "pair_chunk": pair_chunk,
                                                    "workers": workers, "packing_trials": packing_trials,
                                                    "effective_train_graphs": len(splits["train"]),
                                                    "config_digest": digest(config), "source_digest": source["code_digest"],
                                                    "data_manifest_digest": manifest_digest})
    else:
        batch_size, pair_chunk, packing_trials = None, None, []
    resources = {"cpu_trials": cpu_trials, "packing_trials": packing_trials, "workers": workers,
                 "temporary_successful_calibration_seed_updates": sum(row["status"] == "measured" for row in packing_trials) * len(config["model_seeds"]),
                 "preflight_debug_fixture_seed_updates": math_checks["debug_seed_optimizer_updates"],
                 "physical_graph_batch": batch_size, "pair_chunk": pair_chunk,
                 "effective_graph_batch_per_seed": len(splits["train"]), "realizations": config["features"],
                 "gradient_accumulation_batches_per_epoch": math.ceil(len(splits["train"]) / batch_size) if batch_size else None,
                 "gpu_count_used": gpu_count,
                 "seed_axis": "independent models, not data-parallel workers", "jobs": []}
    graph_table = ResultTable(output / "per_graph.csv", GRAPH_COLUMNS)
    draw_table = ResultTable(output / "per_realization.csv", DRAW_COLUMNS)
    rows, history, jobs, new_updates = [], [], [], 0
    try:
        if gpu_count <= 1:
            rows, history, jobs, new_updates, resource_jobs = run_student_jobs(
                all_student_jobs(), cases, refs, config, source, manifest_digest, output,
                device, dtype, batch_size, pair_chunk, args.resume_from, graph_table, draw_table)
            resources["jobs"].extend(resource_jobs)
        else:
            rows, history, jobs, new_updates, resource_jobs, worker_packing = dispatch_gpu_workers(
                args, output, cases, refs, config, source, manifest_digest, gpu_count, workers, graph_table, draw_table)
            resources["jobs"].extend(resource_jobs); resources["worker_packing"] = worker_packing
            # Only deterministic analytic baselines remain on the parent GPU.
            # Worker-specific measured packing remains explicitly recorded.
            batch_size = min(record["physical_graph_batch"] for record in worker_packing)
            pair_chunk = min(record["pair_chunk"] for record in worker_packing)
            resources["oracle_graph_batch"] = batch_size
            resources["oracle_pair_chunk"] = pair_chunk
        # Oracles use the teacher's own Ld, independent of every student recipe.
        for teacher in RECIPES:
            cpu_batches = cache_batches(splits["train"], teacher, batch_size, "cpu", torch.float64)
            eval_batches = {split: cache_batches(splits[split], teacher, batch_size, device, dtype) for split in SPLITS}
            for target in TARGETS:
                fit_targets = batch_targets(cpu_batches, refs, teacher, target)
                eval_targets = {split: batch_targets(eval_batches[split], refs, teacher, target) for split in SPLITS}
                for condition in BASELINES:
                    coefficients, fit = fit_baseline(cpu_batches, fit_targets, condition, config["loss_epsilon"])
                    write_json(output / f"oracle--{teacher}--{target}--{condition}.json", fit)
                    meta = {"teacher_recipe": teacher, "target": target, "student_recipe": "same_teacher_oracle", "condition": condition}
                    for split in SPLITS:
                        rows.extend(evaluate_job(None, eval_batches[split], eval_targets[split], meta, config["loss_epsilon"],
                                                 pair_chunk, graph_table, draw_table, baseline_coefficients=coefficients))
                    jobs.append(dict(meta, seeds=[-1], parameters_per_seed=0, selected_epochs=None,
                                     optimizer_updates_per_seed=0, optimizer=None))
                    print(f"[oracle] teacher={teacher} target={target} condition={condition} coefficients={fit['coefficients']} fit=train_only", flush=True)
            del cpu_batches, eval_batches
    finally:
        graph_table.close(); draw_table.close()
    assert_source_unchanged(source)
    if graph_table.count != planned["per_graph_rows"] or draw_table.count != planned["per_realization_rows"]:
        raise AssertionError("incomplete graph/draw/condition/teacher/seed coverage")
    if len(history) != planned["seed_optimizer_updates"]:
        raise AssertionError("training history omits declared seed optimizer updates")
    expected_history = {(f"{teacher}--{target}--{recipe}--{condition}", seed, epoch)
                        for recipe, teacher, target, condition in all_student_jobs() if condition in TRAINABLE
                        for seed in config["model_seeds"] for epoch in range(1, config["epochs"] + 1)}
    actual_history = {(row["job"], row["seed"], row["epoch"]) for row in history}
    if actual_history != expected_history or len(actual_history) != len(history):
        raise AssertionError("duplicate, missing or mislabeled training epoch/seed records")
    expected_keys = {(teacher, target, recipe, condition, seed, case.split, case.graph_id)
                     for teacher in RECIPES for target in TARGETS for recipe in RECIPES for condition in CONDITIONS
                     for seed in (config["model_seeds"] if condition in TRAINABLE else (-1,)) for case in cases}
    expected_keys |= {(teacher, target, "same_teacher_oracle", condition, -1, case.split, case.graph_id)
                      for teacher in RECIPES for target in TARGETS for condition in BASELINES for case in cases}
    actual_keys = {tuple(row[name] for name in ("teacher_recipe", "target", "student_recipe", "condition", "seed", "split", "graph_id")) for row in rows}
    if actual_keys != expected_keys or len(actual_keys) != len(rows):
        raise AssertionError("duplicate, missing or mislabeled graph evaluation rows")
    comparisons = paired_comparisons(rows, config)
    if len(comparisons) != planned["paired_contrasts"]:
        raise AssertionError("paired comparison coverage differs")
    coverage = {"complete": True, "declared_budget": planned, "per_graph_rows": graph_table.count,
                "per_realization_rows": draw_table.count, "history_rows": len(history),
                "teacher_recipe_target_pairs": 6, "student_gate_recipe_conditions": 12,
                "all_graphs_used": len(cases), "all_scalar_inputs_used": manifest["scalar_inputs"],
                "geometry_manifest_rows": len(geometry_rows), "paired_comparisons": len(comparisons),
                "test_optimizer_updates": 0, "sampling_ratio": 1.0}
    write_csv(output / "training.csv", history)
    write_json(output / "coverage.json", coverage)
    resources.update(process_rss_bytes=psutil.Process().memory_info().rss,
                     system_cpu_percent=psutil.cpu_percent(interval=None), seconds=time.perf_counter() - started)
    write_json(output / "resources.json", resources)
    contract = {"experiment": "B_raw_message_recovery", "profile": args.profile, "debug": args.profile == "debug",
                "graphs": len(cases), "scalar_inputs": manifest["scalar_inputs"], "data_fraction": 1.0,
                "sampling_ratio": 1.0, "config": config, "source": source, "data_manifest_digest": manifest_digest,
                "budget": planned, "new_seed_optimizer_updates": new_updates, "jobs": jobs,
                "source_unchanged_during_run": True, "actual_synthetic_data": True,
                "full_classifier_training": False, "raw_message_operator": True,
                "fixed_teacher_coefficients_not_in_loss": True, "reference_dtype": "float64",
                "model_dtype": config["dtype"], "precision_tf32": False,
                "physical_graph_batch": batch_size, "effective_train_graph_batch": len(splits["train"]),
                "parallel_scalar_realizations": config["features"], "parallel_seed_models": len(config["model_seeds"]),
                "pair_chunk": pair_chunk, "all_eligible_pairs_used": True,
                "gpu_count_used": gpu_count, "multi_gpu_job_partition": gpu_count > 1,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "test_updates": 0, "classification_or_weight_identifiability_claim": False}
    write_json(output / "contract.json", contract)
    print("[report] writing complete raw-message summaries and scientific figures", flush=True)
    write_report(output, rows, comparisons, contract)
    write_json(output / "completion.json", {"status": "complete", "completed": True, "experiment": "B_raw_message_recovery",
                                            "profile": args.profile, "graphs": len(cases), **planned,
                                            "new_seed_optimizer_updates": new_updates,
                                            "math_checks_passed": math_checks["passed"], "source_digest": source["code_digest"],
                                            "data_manifest_digest": manifest_digest,
                                            "seconds": time.perf_counter() - started,
                                            "actual_synthetic_data": True, "actual_citation_data": False})
    print(f"[complete] B raw message study profile={args.profile} graphs={len(cases)} results={output}", flush=True)
    print("[scope] all declared synthetic message targets evaluated; classifier training was not run", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--pair-chunk", default="auto")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--shared-prepared-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
        if not args.resume_from.is_dir() or args.resume_from == output:
            raise ValueError("resume source must be an existing distinct result directory")
    with (output / "terminal.log").open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                with threadpool_limits(limits=1):
                    if (args.shared_prepared_dir is None) != (args.worker_plan is None):
                        raise ValueError("worker requires both immutable shared inputs and an explicit job plan")
                    if args.shared_prepared_dir is not None:
                        worker_run(args, output)
                    else:
                        run(args, output)
            except Exception as error:
                traceback.print_exc()
                write_json(output / "failure.json", {"status": "failed", "type": type(error).__name__,
                                                     "error": str(error), "recovery": "inspect terminal.log; use a fresh output directory and matching source"})
                raise


if __name__ == "__main__":
    main()
