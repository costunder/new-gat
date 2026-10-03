"""Server Experiment 4.1: frozen branch magnitude/direction, zero training."""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..classification.common import digest, read_json, write_csv, write_json
from ..classification.evaluation import classification_metrics, model_state_hash, split_indices
from ..classification.study import _Tee
from ..classification.training import _hardware_sample, make_model
from ..study import available_cpus, runtime_resources, synchronize
from .contract import fixed_variants, read_config, source_manifest, variants, verify_coverage
from .core import FrozenStrengthClassifier, fixed_z_statistics
from .report import write_report
from .source import assert_unchanged, load_source


def _state_for_seeds(source, dataset, condition, seeds):
    lookup = {}
    for pack in source.packs:
        if (pack.dataset, pack.condition) == (dataset, condition):
            for index, seed in enumerate(pack.seeds):
                if seed in lookup:
                    raise ValueError("duplicate source seed state")
                lookup[seed] = pack, index
    if any(seed not in lookup for seed in seeds):
        raise ValueError("requested frozen seed missing from source")
    template = lookup[seeds[0]][0].best_state
    return {
        key: torch.cat(
            [
                lookup[seed][0].best_state[key][lookup[seed][1] : lookup[seed][1] + 1]
                for seed in seeds
            ],
            dim=0,
        )
        for key in template
    }


def _make_model(source, graph, condition, seeds, chunk):
    model = make_model(graph, condition, seeds, source.config, chunk)
    model.load_state_dict(_state_for_seeds(source, graph.name, condition, seeds), strict=True)
    model.requires_grad_(False).eval()
    if condition.startswith("learned_wedge"):
        model = FrozenStrengthClassifier.from_classifier(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen diagnostic must have zero trainable parameters")
    return model


def _metrics(logits, graph, seeds, variant=None):
    indices = split_indices(graph)
    pieces = []
    for split, attribute in (
        ("train", "train_mask"),
        ("validation", "val_mask"),
        ("test", "test_mask"),
    ):
        ce, accuracy = classification_metrics(
            logits, graph.y, getattr(graph, attribute), validate=False, indices=indices[split]
        )
        pieces.extend((ce, accuracy))
    pieces.append(torch.isfinite(logits).flatten(1).all(1).to(logits.dtype))
    values = torch.stack(pieces, 1).detach().cpu().numpy()
    if not np.isfinite(values).all() or not np.all(values[:, -1] == 1):
        raise FloatingPointError("nonfinite frozen logits/metrics")
    rows = []
    for index, seed in enumerate(seeds):
        for offset, split in enumerate(("train", "validation", "test")):
            ce, accuracy = values[index, 2 * offset : 2 * offset + 2]
            if ce < 0 or not 0 <= accuracy <= 1:
                raise ValueError("invalid frozen CE/accuracy")
            rows.append(
                {
                    "dataset": graph.name,
                    "seed": seed,
                    "split": split,
                    "ce": float(ce),
                    "accuracy": float(accuracy),
                    "num_nodes": graph.num_nodes,
                    "num_labeled_nodes": indices[split].numel(),
                    **(variant or {}),
                }
            )
    return rows


def _scalar_rows(detail, seeds, identity):
    scalars = {
        key: value
        for key, value in detail.items()
        if isinstance(value, torch.Tensor) and value.shape == (len(seeds),)
    }
    values = torch.stack([value.to(torch.float64) for value in scalars.values()], 1).cpu().numpy()
    rows = []
    for index, seed in enumerate(seeds):
        row = {**identity, "seed": seed}
        for (key, tensor), value in zip(scalars.items(), values[index], strict=True):
            if math.isnan(value):
                defined = scalars.get(key + "_defined")
                if key.startswith("kappa_"):
                    defined = scalars.get("kappa_ratio_defined")
                if defined is None or bool(
                    values[
                        index,
                        list(scalars).index(
                            key + "_defined"
                            if key + "_defined" in scalars
                            else "kappa_ratio_defined"
                        ),
                    ]
                ):
                    raise FloatingPointError(f"unexplained NaN diagnostic: {key}")
                row[key] = None
            elif not math.isfinite(value):
                raise FloatingPointError(f"nonfinite diagnostic: {key}")
            elif tensor.dtype == torch.bool:
                row[key] = bool(value)
            elif not tensor.is_floating_point():
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def _check_replay(rows, original_rows, config):
    original = {(r["dataset"], r["condition"], r["seed"], r["split"]): r for r in original_rows}
    checks = []
    for row in rows:
        key = row["dataset"], row["condition"], row["seed"], row["split"]
        reference = original[key]
        for label, item in (("replayed", row), ("source", reference)):
            ce, accuracy = item["ce"], item["accuracy"]
            if (
                not math.isfinite(ce)
                or ce < 0
                or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1
            ):
                raise ValueError(f"invalid {label} CE/accuracy in source replay: {key}")
        ce_error, accuracy_error = (
            abs(row["ce"] - reference["ce"]),
            abs(
                row["accuracy"] - reference["accuracy"],
            ),
        )
        allowed = config["source_replay"]["ce_atol"] + (
            config["source_replay"]["ce_rtol"] * abs(reference["ce"])
        )
        if (
            ce_error > allowed
            or accuracy_error > config["source_replay"]["accuracy_atol"]
            or row["num_nodes"] != reference["num_nodes"]
            or row["num_labeled_nodes"] != reference["num_labeled_nodes"]
        ):
            raise ValueError(
                f"selected source checkpoint replay differs: {key}; "
                f"CE error={ce_error} accuracy error={accuracy_error}"
            )
        checks.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "seed": key[2],
                "split": key[3],
                "ce_abs_error": ce_error,
                "accuracy_abs_error": accuracy_error,
                "within_declared_tolerance": True,
            }
        )
    return checks


def _device_manifests(records, device):
    # 4.1 changes only C positions. Existing random-row tensors stay on CPU.
    return [
        {
            "index": record["index"],
            "sha256": record["sha256"],
            "permutation": record["permutation"].to(device),
        }
        for record in records
    ]


def _trial(source, graph, condition, seeds, chunk, config, manifests):
    model = _make_model(source, graph, condition, seeds, chunk)
    if condition.startswith("learned_wedge"):
        model.set_treatment("shuffle_norm_matched", manifest=manifests[0])
    runtime = config["runtime"]
    with torch.inference_mode():
        for _ in range(runtime["calibration_warmups"]):
            model(graph, diagnostics=True)
        synchronize(graph.x.device)
        if graph.x.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(graph.x.device)
        start = time.perf_counter()
        for _ in range(runtime["calibration_repeats"]):
            model(graph, diagnostics=True)
        synchronize(graph.x.device)
    seconds = (time.perf_counter() - start) / runtime["calibration_repeats"]
    peak = (
        torch.cuda.max_memory_allocated(graph.x.device) if graph.x.device.type == "cuda" else None
    )
    return seconds, peak, model.parameters_per_seed


def _choose_packing(source, graph, condition, config, manifests, output):
    seeds = config["final_seeds"]
    candidates = sorted({min(len(seeds), n) for n in config["runtime"]["packed_candidates"]})
    paths = graph.paths.shape[1]
    chunks = sorted(
        {
            max(1, min(paths, n))
            for n in [*config["runtime"]["path_chunk_candidates"], max(1, paths)]
        }
    )
    if not condition.startswith("learned_wedge"):
        chunks = [max(1, paths)]
    rows = []
    for packed in candidates:
        for chunk in chunks:
            device = graph.x.device
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(device)
                baseline = torch.cuda.memory_allocated(device)
            else:
                free = baseline = None
            row = {
                "scope": "calibration",
                "dataset": graph.name,
                "condition": condition,
                "device": str(device),
                "packed_runs": packed,
                "path_chunk": chunk,
                "all_paths": paths,
                "measured": False,
            }
            try:
                seconds, peak, parameters = _trial(
                    source,
                    graph,
                    condition,
                    seeds[:packed],
                    chunk,
                    config,
                    manifests,
                )
                safe = (
                    device.type != "cuda"
                    or peak - baseline < free * config["runtime"]["gpu_memory_safety_fraction"]
                )
                row.update(
                    seconds_per_forward=seconds,
                    peak_vram_bytes=peak,
                    model_forwards_per_second=packed / seconds,
                    parameters_per_seed=parameters,
                    measured=True,
                    status="measured" if safe else "memory_safety_rejected",
                )
            except torch.cuda.OutOfMemoryError as error:
                row.update(
                    status="OOM",
                    error=str(error),
                    seconds_per_forward=None,
                    peak_vram_bytes=None,
                    model_forwards_per_second=None,
                )
            rows.append(row)
            print(
                f"[calibration] {graph.name}/{condition} packed={packed} chunk={chunk} "
                f"seconds={row['seconds_per_forward']} peak={row['peak_vram_bytes']} "
                f"status={row['status']}",
                flush=True,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    feasible = [row for row in rows if row["status"] == "measured"]
    write_json(output, {"trials": rows})
    if not feasible:
        raise RuntimeError("no measured safe exact chunk/seed pack; source graph/model kept whole")
    chosen = max(feasible, key=lambda row: (row["model_forwards_per_second"], row["packed_runs"]))
    return chosen, rows


def _evaluate_pack(source, graph, condition, seeds, chunk, config, manifests, device_resources):
    model = _make_model(source, graph, condition, seeds, chunk)
    before = model_state_hash(model)
    result = {
        "baseline_rows": [],
        "treatment_rows": [],
        "diagnostic_rows": [],
        "fixed_z_rows": [],
        "replay_rows": [],
        "provenance": [],
    }
    learned = condition.startswith("learned_wedge")
    manifest_map = {row["index"]: row for row in manifests}
    start = time.perf_counter()
    with torch.inference_mode():
        baseline_logits, baseline_details = model(graph, diagnostics=True)
        original = _metrics(baseline_logits, graph, seeds)
        for row in original:
            row["condition"] = condition
        result["baseline_rows"] += original
        result["replay_rows"] += _check_replay(original, source.original_rows, config)
        for layer, detail in enumerate(baseline_details):
            if not learned:
                detail = fixed_z_statistics(model, graph, layer, detail["z"], "baseline")
            identity = {
                "dataset": graph.name,
                "condition": condition,
                "layer": layer,
                "source": "end_to_end",
                "treatment": "baseline",
                "target": "none",
                "manifest_index": -1,
            }
            result["diagnostic_rows"] += _scalar_rows(detail, seeds, identity)
        print(f"[baseline] {graph.name}/{condition} seeds={seeds} replay=verified", flush=True)
        if learned:
            for layer, reference in enumerate(baseline_details):
                for variant in fixed_variants(config):
                    model.set_treatment(
                        variant["treatment"], "both", manifest_map.get(variant["manifest_index"])
                    )
                    detail = fixed_z_statistics(model, graph, layer, reference["z"])
                    identity = {
                        "dataset": graph.name,
                        "condition": condition,
                        "layer": layer,
                        "source": "fixed_Z",
                        **variant,
                        "target": f"layer_{layer}",
                    }
                    result["fixed_z_rows"] += _scalar_rows(detail, seeds, identity)
                    print(
                        f"[fixed Z] {graph.name}/{condition} seeds={seeds} layer={layer} "
                        f"{variant['treatment']} manifest={variant['manifest_index']}",
                        flush=True,
                    )
            for index, variant in enumerate(variants(config, False)):
                model.set_treatment(
                    variant["treatment"],
                    variant["target"],
                    manifest_map.get(variant["manifest_index"]),
                )
                logits, details = model(graph, diagnostics=True)
                rows = _metrics(logits, graph, seeds, variant)
                for row in rows:
                    row["condition"] = condition
                result["treatment_rows"] += rows
                for layer, detail in enumerate(details):
                    result["diagnostic_rows"] += _scalar_rows(
                        detail,
                        seeds,
                        {
                            "dataset": graph.name,
                            "condition": condition,
                            "layer": layer,
                            "source": "end_to_end",
                            **variant,
                        },
                    )
                print(
                    f"[case] {graph.name}/{condition} seeds={seeds} "
                    f"{index + 1}/{len(variants(config, False))} {variant['treatment']} "
                    f"target={variant['target']} manifest={variant['manifest_index']} "
                    f"testCE={np.mean([r['ce'] for r in rows if r['split'] == 'test']):.6f}",
                    flush=True,
                )
    after = model_state_hash(model)
    if before != after:
        raise RuntimeError("frozen parameters/buffers changed")
    count = len(seeds) * (len(variants(config)) if learned else 1)
    wall = time.perf_counter() - start
    result["provenance"].append(
        {
            "dataset": graph.name,
            "condition": condition,
            "seeds": seeds,
            "model_hash_before": before,
            "model_hash_after": after,
            "optimizer_updates": 0,
            "trainable_parameters": 0,
            "completed_seed_cases": count,
            "source_checkpoint_hashes": {
                str(pack.checkpoint): source.hashes[
                    pack.checkpoint.relative_to(source.directory).as_posix()
                ]
                for pack in source.packs
                if (pack.dataset, pack.condition) == (graph.name, condition)
            },
        }
    )
    result["resource_rows"] = [
        {
            "scope": "evaluation",
            "dataset": graph.name,
            "condition": condition,
            "device": str(graph.x.device),
            "packed_runs": len(seeds),
            "path_chunk": chunk,
            "completed_seed_cases": count,
            "wall_seconds": wall,
            "seconds_per_forward": None,
            "measured": True,
            "status": "measured",
            "peak_vram_bytes": torch.cuda.max_memory_allocated(graph.x.device)
            if graph.x.device.type == "cuda"
            else None,
            **_hardware_sample(graph.x.device, device_resources),
        }
    ]
    return result


def _worker(plan_path):
    plan = read_json(plan_path)
    output = Path(plan["output_dir"])
    config = read_config(output / "config.json", plan["profile"])
    if source_manifest()["code_digest"] != plan["code_digest"]:
        raise ValueError("diagnostic source changed after dispatch")
    source = load_source(plan["source_dir"], plan["profile"])
    if digest(source.hashes) != plan["source_hash_digest"]:
        raise ValueError("source artifacts changed after dispatch")
    device = torch.device(plan["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resources = runtime_resources(device)
    threads = config["runtime"]["cpu_threads"]
    if threads == "auto":
        threads = max(1, min(8, available_cpus(resources) // plan["num_workers"]))
    torch.set_num_threads(threads)
    folder = output / "workers" / f"worker-{plan['worker_index']}"
    folder.mkdir(exist_ok=False)
    result = {
        name: []
        for name in (
            "baseline_rows",
            "treatment_rows",
            "diagnostic_rows",
            "fixed_z_rows",
            "resource_rows",
            "replay_rows",
            "provenance",
        )
    }
    graphs, manifests = {}, {}
    with (folder / "terminal.log").open("x", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(_Tee(sys.stdout, logfile)):
            for index, job in enumerate(plan["jobs"]):
                dataset, condition = job["dataset"], job["condition"]
                if dataset not in graphs:
                    graphs[dataset] = source.graphs[dataset].to(device, torch.float32)
                    manifests[dataset] = _device_manifests(source.manifests[dataset], device)
                graph = graphs[dataset]
                print(
                    f"[job] worker={plan['worker_index']} {index + 1}/{len(plan['jobs'])} "
                    f"{dataset}/{condition} N={graph.num_nodes} E={graph.edges.shape[1]} "
                    f"P={graph.paths.shape[1]} input={tuple(graph.x.shape)} "
                    f"layers=2 hidden={source.config['backbone']['hidden_dim']} "
                    f"attention_heads=0 graph_batch=1 accumulation=1 epochs=N/A updates=0",
                    flush=True,
                )
                chosen, trials = _choose_packing(
                    source,
                    graph,
                    condition,
                    config,
                    manifests[dataset],
                    output / "calibration" / f"{dataset}-{condition}.json",
                )
                result["resource_rows"] += trials
                packed, chunk = chosen["packed_runs"], chosen["path_chunk"]
                print(
                    f"[selected] {dataset}/{condition} packed_seed_models={packed} "
                    f"parameters_per_model={chosen['parameters_per_seed']} "
                    f"total_parameters={packed * chosen['parameters_per_seed']} "
                    f"trainable_parameters=0 chunk={chunk} all_paths={graph.paths.shape[1]} "
                    "dropout=off precision=float32 tf32=off cache=whole_graph "
                    "data_fraction=1.0 sampling=1.0 loader_workers=N/A",
                    flush=True,
                )
                for start in range(0, len(config["final_seeds"]), packed):
                    seeds = config["final_seeds"][start : start + packed]
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    evaluated = _evaluate_pack(
                        source,
                        graph,
                        condition,
                        seeds,
                        chunk,
                        config,
                        manifests[dataset],
                        resources,
                    )
                    for name in result:
                        result[name] += evaluated[name]
                    del evaluated
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            assert_unchanged(source)
            if source_manifest()["code_digest"] != plan["code_digest"]:
                raise ValueError("diagnostic implementation changed during worker execution")
            write_json(folder / "results.json", result)
            write_json(
                folder / "completion.json",
                {
                    "completed": True,
                    "optimizer_updates": 0,
                    "source_preserved": True,
                    "jobs": len(plan["jobs"]),
                },
            )


def _dispatch(output, config, source, devices):
    assignments, costs = [[] for _ in devices], [0] * len(devices)
    for dataset in config["data"]["datasets"]:
        for condition in config["conditions"]:
            index = min(range(len(devices)), key=lambda i: costs[i])
            assignments[index].append({"dataset": dataset, "condition": condition})
            graph = source.graphs[dataset]
            costs[index] += (
                graph.paths.shape[1] * len(variants(config))
                if condition.startswith("learned_wedge")
                else graph.num_nodes
            )
    plans = []
    code_digest = source_manifest()["code_digest"]
    for index, (device, jobs) in enumerate(zip(devices, assignments, strict=True)):
        if jobs:
            path = output / "plans" / f"worker-{index}.json"
            write_json(
                path,
                {
                    "output_dir": str(output),
                    "source_dir": str(source.directory),
                    "profile": config["profile"],
                    "source_hash_digest": digest(source.hashes),
                    "code_digest": code_digest,
                    "device": device,
                    "worker_index": index,
                    "num_workers": len(devices),
                    "jobs": jobs,
                },
            )
            plans.append(path)
    if len(plans) == 1:
        _worker(plans[0])
    else:
        processes, readers = [], []
        for path in plans:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "research.wedge_propagation.branch_strength.study",
                    "--worker-plan",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            def consume(stream):
                for line in stream:
                    print(line, end="", flush=True)
                stream.close()

            reader = threading.Thread(target=consume, args=(process.stdout,), daemon=True)
            reader.start()
            readers.append(reader)
            processes.append(process)
        codes = [process.wait() for process in processes]
        for reader in readers:
            reader.join()
        if any(code != 0 for code in codes):
            raise RuntimeError(
                f"diagnostic worker failed codes={codes}; "
                "sessions/source/results preserved; inspect terminal.log"
            )
    result = {
        name: []
        for name in (
            "baseline_rows",
            "treatment_rows",
            "diagnostic_rows",
            "fixed_z_rows",
            "resource_rows",
            "replay_rows",
            "provenance",
        )
    }
    for path in plans:
        index = read_json(path)["worker_index"]
        record = read_json(output / "workers" / f"worker-{index}" / "results.json")
        for name in result:
            result[name] += record[name]
    return result


def run(args):
    config = read_config(Path(__file__).with_name(f"config_{args.profile}.json"), args.profile)
    device = torch.device(args.device)
    if args.profile == "full" and (
        device.type != "cuda" or not os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise ValueError(
            "full server diagnostic requires explicitly allocated CUDA_VISIBLE_DEVICES"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no CPU fallback")
    output = Path(args.output_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    if output.exists() or output.is_relative_to(source_dir):
        raise ValueError("output must be a NEW directory outside the preserved source run")
    devices = (
        [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if args.device == "cuda"
        else [str(device)]
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("plans", "workers", "calibration"):
        (output / name).mkdir()
    with (output / "terminal.log").open("x", encoding="utf-8") as logfile:
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, logfile)),
            contextlib.redirect_stderr(_Tee(sys.stderr, logfile)),
        ):
            try:
                start = time.perf_counter()
                diagnostic_source = source_manifest()
                write_json(output / "config.json", config)
                write_json(output / "source.json", diagnostic_source)
                print(
                    f"[start] Experiment4.1 profile={args.profile} devices={devices} "
                    "source_checkpoint=frozen optimizer_updates=0 no_selection "
                    "sampling=1.0 norm_match=whole_transductive_nodes_channels",
                    flush=True,
                )
                print(
                    f"[source] checking completed run and every artifact: {source_dir}", flush=True
                )
                source = load_source(source_dir, args.profile)
                print(
                    f"[source] verified final_models="
                    f"{config['expected_counts']['source_final_models']} "
                    f"preserved_files={len(source.hashes)} source_epochs="
                    f"{source.config['training']['epochs_per_run']}",
                    flush=True,
                )
                if (
                    source.config["data"]["datasets"] != config["data"]["datasets"]
                    or source.config["training"]["final_seeds"] != config["final_seeds"]
                ):
                    raise ValueError("source full dataset/seed contract differs")
                hardware = runtime_resources(device)
                hardware.update(
                    worker_devices=devices,
                    gpu_count_used=len(devices) if device.type == "cuda" else 0,
                )
                write_json(output / "hardware.json", hardware)
                write_json(
                    output / "source_run.json",
                    {
                        "directory": str(source_dir),
                        "source": source.source,
                        "completion": source.completion,
                        "artifact_hashes": source.hashes,
                    },
                )
                print(f"[hardware] {hardware}", flush=True)
                result = _dispatch(output, config, source, devices)
                coverage = verify_coverage(
                    config,
                    result["baseline_rows"],
                    result["treatment_rows"],
                    result["diagnostic_rows"],
                    result["fixed_z_rows"],
                )
                assert_unchanged(source)
                if source_manifest()["code_digest"] != diagnostic_source["code_digest"]:
                    raise RuntimeError("diagnostic code changed during execution")
                files = {
                    "baseline_metrics.csv": "baseline_rows",
                    "treatments.csv": "treatment_rows",
                    "layer_diagnostics.csv": "diagnostic_rows",
                    "fixed_z.csv": "fixed_z_rows",
                    "resources.csv": "resource_rows",
                    "source_replay.csv": "replay_rows",
                }
                for filename, name in files.items():
                    write_csv(output / filename, result[name])
                write_json(output / "frozen_model_provenance.json", result["provenance"])
                contract = {
                    "source_run": str(source_dir),
                    "source_config": source.config,
                    "source_hash_digest": digest(source.hashes),
                    "source_training_completion": source.completion,
                    "diagnostic_source": diagnostic_source,
                    "hardware": hardware,
                    "coverage": coverage,
                    "optimizer_updates": 0,
                    "source_files_and_models_preserved": True,
                    "actual_data": source.completion["actual_data"],
                    "interpretation": "posthoc_fixed_checkpoint_diagnostic_no_treatment_selection",
                }
                write_json(output / "contract.json", contract)
                write_json(output / "coverage.json", coverage)
                print(
                    "[report] writing branch direction/strength tables and scientific figures",
                    flush=True,
                )
                write_report(
                    output,
                    config,
                    result["baseline_rows"],
                    result["treatment_rows"],
                    result["diagnostic_rows"],
                    result["fixed_z_rows"],
                    result["resource_rows"],
                    contract,
                )
                write_json(
                    output / "completion.json",
                    {
                        "completed": True,
                        "profile": args.profile,
                        "scope": "frozen_branch_strength_diagnostic",
                        "coverage": coverage,
                        "optimizer_updates": 0,
                        "source_files_and_models_preserved": True,
                        "actual_data": contract["actual_data"],
                        "seconds": time.perf_counter() - start,
                    },
                )
                print(
                    f"[complete] results={output} coverage={coverage} optimizer_updates=0",
                    flush=True,
                )
            except Exception as error:
                traceback.print_exc()
                write_json(
                    output / "failure.json",
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "source_and_partial_results_preserved": True,
                    },
                )
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        _worker(args.worker_plan)
    else:
        if args.source_dir is None or args.output_dir is None:
            parser.error("--source-dir and --output-dir naming a NEW directory are required")
        run(args)


if __name__ == "__main__":
    main()
