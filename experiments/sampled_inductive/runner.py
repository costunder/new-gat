"""Four-cell inductive study with measured resources and immutable graph splits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from chartgat.cache import atomic_write_json
from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.runner import make_jobs
from experiments.aggregation_comparison.runner import parser as comparison_parser
from scripts.calibration_lock import calibration_lock
from scripts.training_resource_plan import allocated_cpu_count

from . import train
from .data import validate_splits
from .report import summarize


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/paper"))
    p.add_argument("--results-root", type=Path, default=Path("results/sampled_inductive"))
    p.add_argument("--profile", choices=("reference", "large"), default="reference")
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--dynamic-c", choices=("shared", "per_head"), default="per_head")
    p.add_argument(
        "--context-seeds",
        type=int,
        required=True,
        help="explicit supervised seeds per context; sampling law, not a memory fallback",
    )
    p.add_argument(
        "--context-batches",
        type=int,
        nargs="+",
        required=True,
        help="physical context batch candidates, measured for both C regimes",
    )
    p.add_argument("--graph-batches", type=int, nargs="+", default=[2, 4, 8, 16, 20])
    p.add_argument("--worker-candidates", type=int, nargs="+", default=[2, 4])
    p.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--edge-chunk-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=0.0005)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hardware-profile", choices=("portable", "a6000-48gb"), default="portable")
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--calibration-only", action="store_true")
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate all four validation-selected checkpoints only after matrix completion",
    )
    return p


def validate(options):
    from research.conductance_gat.v5.protocol import HARDWARE_PROFILES
    from scripts.run_conductance_v5 import RUN_ID_PATTERN

    if not RUN_ID_PATTERN.fullmatch(options.run_id):
        raise ValueError("unsafe run ID")
    if not options.device.startswith("cuda") or options.epochs < 4 or options.context_seeds < 1:
        raise ValueError("CUDA, >=4 epochs and positive explicit context seeds required")
    for name in ("context_batches", "graph_batches", "worker_candidates"):
        values = getattr(options, name)
        if len(set(values)) != len(values) or len(values) < 2 or min(values) < 2:
            raise ValueError(f"{name} needs at least two distinct measured candidates >=2")
    if (
        not options.model_seeds
        or len(set(options.model_seeds)) != len(options.model_seeds)
        or min(options.model_seeds) < 0
    ):
        raise ValueError("model seeds must be unique and nonnegative")
    if not options.num_neighbors or min(options.num_neighbors) < 1 or options.edge_chunk_size < 1:
        raise ValueError("sampling expansion and chunk sizes must be positive")
    if not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
        raise ValueError("invalid learning rate")
    if not math.isfinite(options.min_free_gb) or options.min_free_gb < 0:
        raise ValueError("invalid minimum GPU memory")
    if min(options.graph_batches) < HARDWARE_PROFILES[options.hardware_profile]["ppi_batch_size"]:
        raise ValueError("graph batch search cannot fall below the original hardware profile floor")


def child_arguments(options, arm, seed, batch=2, workers=2):
    parent = comparison_parser().parse_args(
        [
            "--run-id",
            options.run_id,
            "--datasets",
            "ppi",
            "--profiles",
            options.profile,
            "--arms",
            arm,
            "--model-seeds",
            str(seed),
            "--hardware-profile",
            options.hardware_profile,
            "--data-root",
            str(options.data_root),
            "--epochs",
            str(options.epochs),
            "--patience",
            str(options.epochs),
            "--workers",
            str(workers),
            "--edge-chunk-size",
            str(options.edge_chunk_size),
            "--learning-rate",
            str(options.learning_rate),
            "--device",
            options.device,
            "--activation-checkpoint",
        ]
    )
    command = make_jobs(parent, options.results_root / options.run_id)[0]["command"]
    args = engine.build_parser().parse_args(command[command.index("-m") + 2 :])
    # Architecture comes from the historical profile; this independent study
    # supplies its own measured resource plan, without changing that profile.
    args.batch_size = batch
    args.context_seeds = options.context_seeds
    args.num_neighbors = options.num_neighbors
    args.learning_budget_policy = "epochs"
    engine.validate_args(args)
    return args


def configuration(options):
    return {
        k: str(v.resolve()) if isinstance(v, Path) else v
        for k, v in vars(options).items()
        if k not in {"dry_run", "calibration_only", "evaluate_test"}
    }


def cells(options):
    dynamic = "incidence" if options.dynamic_c == "per_head" else "incidence_shared"
    return [
        (seed, mode, arm)
        for seed in options.model_seeds
        for mode in ("full", "sampled")
        for arm in ("incidence_fixed", dynamic)
    ]


def hardware_identity(device):
    p = torch.cuda.get_device_properties(device)
    return {
        "name": p.name,
        "total_memory": p.total_memory,
        "capability": [p.major, p.minor],
        "uuid": str(getattr(p, "uuid", "unavailable")),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allocated_cpus": allocated_cpu_count(),
        "versions": engine.base._versions(),
    }


def calibrate(options, payload, manifest, save, device):
    cpu_count = allocated_cpu_count()
    workers = [n for n in options.worker_candidates if n <= cpu_count]
    if len(workers) < 2:
        raise ValueError(
            "allocation cannot measure requested worker candidates; specify two valid candidates"
        )
    for mode in ("full", "sampled"):
        count = (
            len(payload["splits"]["train"])
            if mode == "full"
            else sum(
                math.ceil(payload["graphs"][int(i)]["x"].shape[0] / options.context_seeds)
                for i in payload["splits"]["train"]
            )
        )
        batches = [
            n
            for n in (options.graph_batches if mode == "full" else options.context_batches)
            if n <= count
        ]
        if len(batches) < 2:
            raise ValueError(
                "need two physical batch candidates within the actual training population"
            )
        candidates = []
        for batch in batches:
            for worker in workers:
                reports = []
                for seed, support, arm in cells(options):
                    if support != mode:
                        continue
                    key = f"{mode}/{batch}/{worker}/{seed}/{arm}"
                    if key not in manifest["calibration"]:
                        if mode in manifest["selected_resources"]:
                            raise ValueError(
                                "selected resource plan has missing calibration evidence"
                            )
                        args = child_arguments(options, arm, seed, batch, worker)
                        print(f"[inductive calibration] {key}", flush=True)
                        try:
                            report = train.probe(payload, args, mode, batch, worker, device)
                        except torch.OutOfMemoryError as error:
                            report = {"status": "oom", "error": str(error), "safe": False}
                        report = {**report, "measurement_key": key}
                        report["measurement_sha256"] = measurement_digest(report)
                        manifest["calibration"][key] = report
                        save()
                    report = manifest["calibration"][key]
                    if report.get("measurement_key") != key or report.get(
                        "measurement_sha256"
                    ) != measurement_digest(report):
                        raise ValueError("calibration evidence identity or checksum changed")
                    reports.append(report)
                if all(r["status"] == "passed" and r["safe"] for r in reports):
                    candidates.append(
                        {
                            "batch": batch,
                            "workers": worker,
                            "projected_epoch_seconds": max(
                                r["epoch_seconds"] + r["validation_seconds"] for r in reports
                            ),
                        }
                    )
        if not candidates:
            raise RuntimeError(
                f"no measured {mode} batch fits both C regimes; no fallback/downscale"
            )
        selected = min(candidates, key=lambda r: (r["projected_epoch_seconds"], -r["batch"]))
        if (
            mode in manifest["selected_resources"]
            and manifest["selected_resources"][mode] != selected
        ):
            raise ValueError("selected resource plan does not match measured calibration evidence")
        manifest["selected_resources"][mode] = selected
        save()


def measurement_digest(report):
    value = {k: v for k, v in report.items() if k != "measurement_sha256"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def evaluate_test(options, payload, protocol, manifest, folder, device, save):
    # Freeze the entire completed matrix before touching test labels.
    expected_keys = {f"seed-{s}/{m}/{a}" for s, m, a in cells(options)}
    if set(manifest["results"]) != expected_keys:
        raise ValueError("test evaluation requires every validation-selected cell")
    validated = {}
    for key in sorted(expected_keys):
        result = train.completed(folder / key)
        if result != manifest["results"][key]:
            raise ValueError("test matrix differs from validated on-disk training evidence")
        validated[key] = result
    lock = {key: result["last_sha256"] for key, result in manifest["results"].items()}
    if manifest.get("test_checkpoint_lock") not in (None, lock):
        raise ValueError("test-selected checkpoint matrix changed")
    manifest["test_checkpoint_lock"] = lock
    save()
    for seed, mode, arm in cells(options):
        key = f"seed-{seed}/{mode}/{arm}"
        result = validated[key]
        if result["last_sha256"] != lock[key]:
            raise ValueError("test-selected checkpoint changed")
        if key in manifest.setdefault("test", {}):
            saved = manifest["test"][key]
            if saved["checkpoint_sha256"] != lock[key]:
                raise ValueError("test evaluation checkpoint identity changed")
            train.validate_evaluation(saved["evaluation"], label="saved held-out test graphs")
            if saved["evaluation"]["graph_ids"] != list(payload["splits"]["test"]):
                raise ValueError("test evaluation graph split changed")
            continue
        selected = manifest["selected_resources"][mode]
        args = child_arguments(options, arm, seed, selected["batch"], selected["workers"])
        inputs = train.make_inputs(
            payload, args, mode, selected["batch"], selected["workers"], split="test"
        )
        try:
            model = engine.make_model(payload, args, device)
            checkpoint = engine.base.load_checkpoint_on_cpu(folder / key / "last.pt")
            model.load_state_dict(checkpoint["best_state"])
            before = engine.base.state_sha256(model)
            metric = train.evaluate(model, inputs, args, device)
            if before != engine.base.state_sha256(model) or result["last_sha256"] != lock[key]:
                raise ValueError("test path modified the selected model")
            manifest["test"][key] = {
                "checkpoint_sha256": lock[key],
                "evaluation": metric,
                "selection": "validation only; theta frozen; C recomputed from x/edges",
            }
            save()
        finally:
            inputs.close()


def main(argv=None):
    invocation_started = time.perf_counter()
    invocation_unix = time.time()
    options = parser().parse_args(argv)
    validate(options)
    folder = (options.results_root / options.run_id).resolve()
    data_root = options.data_root.resolve()
    if folder.is_relative_to(data_root) or data_root.is_relative_to(folder):
        raise ValueError("results cannot overlap immutable data")
    planned = [
        {
            "seed": s,
            "support": m,
            "arm": a,
            "configuration": engine.configuration(child_arguments(options, a, s)),
        }
        for s, m, a in cells(options)
    ]
    if options.dry_run:
        print(
            json.dumps(
                {
                    "suite": train.SUITE,
                    "cells": planned,
                    "sampling": configuration(options),
                    "note": "no data loaded, no calibration or training executed",
                },
                indent=2,
            )
        )
        return 0
    device = torch.device(options.device)
    engine.base._require_cuda(device)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < options.min_free_gb * 1024**3:
        raise RuntimeError("insufficient requested free GPU memory")
    load_started = time.perf_counter()
    payload, protocol = engine.base.load_dataset("ppi", data_root, allow_download=False)
    engine.base.validate_cached_graphs_once(payload)
    validate_splits(payload)
    load_seconds = time.perf_counter() - load_started
    if folder.is_symlink() or any(p.is_symlink() for p in folder.parents):
        raise ValueError("indirect output forbidden")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    expected = {
        "suite": train.SUITE,
        "configuration": configuration(options),
        "sources": train.sources(),
        "protocol": protocol,
        "hardware": hardware_identity(device),
    }
    with calibration_lock(folder):
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["identity"] != expected:
                raise ValueError(
                    "study allocation/source/configuration changed; use a fresh run ID"
                )
        else:
            if any(p.name != ".calibration.lock" for p in folder.iterdir()):
                raise FileExistsError("nonempty study folder without manifest")
            manifest = {
                "identity": expected,
                "calibration": {},
                "selected_resources": {},
                "results": {},
            }
        invocation = {
            "started_unix": invocation_unix,
            "data_loading_seconds": load_seconds,
            "status": "running",
            "calibration_only": options.calibration_only,
        }
        manifest.setdefault("invocations", []).append(invocation)

        def save():
            if expected["sources"] != train.sources():
                raise ValueError("source changed while study was running")
            invocation["elapsed_wall_seconds"] = time.perf_counter() - invocation_started
            atomic_write_json(path, manifest)

        save()
        calibrate(options, payload, manifest, save, device)
        if options.calibration_only:
            invocation["status"] = "passed"
            save()
            return 0
        for seed, mode, arm in cells(options):
            key = f"seed-{seed}/{mode}/{arm}"
            selected = manifest["selected_resources"][mode]
            args = child_arguments(options, arm, seed, selected["batch"], selected["workers"])
            manifest["results"][key] = train.train_cell(
                payload,
                protocol,
                args,
                mode,
                selected["batch"],
                selected["workers"],
                device,
                folder / key,
            )
            save()
            atomic_write_json(folder / "comparison.json", summarize(manifest))
            gc.collect()
            torch.cuda.empty_cache()
        if options.evaluate_test:
            evaluate_test(options, payload, protocol, manifest, folder, device, save)
        invocation["status"] = "passed"
        save()
        atomic_write_json(folder / "comparison.json", summarize(manifest))
    print(f"passed: {folder}", flush=True)
    return 0
