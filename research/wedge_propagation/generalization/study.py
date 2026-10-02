"""Experiment 3: freeze completed models and evaluate genuinely unseen features."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from ..learned.data import save_dataset
from ..learned.evaluation import evaluate, seed_macro
from ..learned.train import CONDITIONS, INTERVENTIONS, TARGETS, cache_batches
from ..study import Tee, available_cpus, runtime_resources, synchronize, write_csv, write_json
from .data import feature_cases, load_source_run
from .frozen import frozen_fingerprint, load_models, scale_diagnostics
from .report import write_report

SPLITS = ("train", "validation", "id", "size_ood", "family_ood", "family_size_ood")
AMPLITUDES = (0.25, 0.5, 1.0, 2.0, 4.0)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path, profile):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["profile"] != profile or value["source_profile"] != profile:
        raise ValueError("Experiment 3 and source profiles must match")
    if tuple(value["amplitudes"]) != AMPLITUDES:
        raise ValueError("all five prespecified amplitude treatments are required")
    if tuple(value["targets"]) != TARGETS or tuple(value["conditions"]) != CONDITIONS:
        raise ValueError("all three targets and five conditions are required")
    if tuple(value["splits"]) != SPLITS:
        raise ValueError("all six original splits must be retained")
    if value["new_training_epochs"] != 0 or value["optimizer_updates"] != 0:
        raise ValueError("Experiment 3 must freeze the source models")
    if value["feature_distribution"] != "standard_normal":
        raise ValueError("the declared independent standard-normal feature stream is required")
    if value["feature_realizations"] != "inherit_source":
        raise ValueError("all source feature realizations must be retained")
    if isinstance(value["master_seed"], bool) or not isinstance(value["master_seed"], int):
        raise ValueError("master_seed must be an integer")
    if value["master_seed"] < 0:
        raise ValueError("master_seed must be nonnegative")
    for name in ("metric_epsilon", "original_reproduction_atol", "original_reproduction_rtol"):
        if not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return value


def source_manifest():
    folder = Path(__file__).parent
    root = folder.parent
    files = sorted([*folder.glob("*.py"), *folder.glob("*.json")])
    dependencies = (
        "data.py",
        "operators.py",
        "study.py",
        "learned/data.py",
        "learned/model.py",
        "learned/evaluation.py",
        "learned/train.py",
    )
    files.extend(root / name for name in dependencies)
    return {file.relative_to(root).as_posix(): file_hash(file) for file in files}


def read_rows(path):
    """Read measured CSV values; empty diagnostics stay undefined."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty source result: {path}")
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            elif re.fullmatch(r"-?\d+", value):
                row[key] = int(value)
            else:
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    raise ArithmeticError(f"nonfinite measured source value: {path}:{key}")
                row[key] = number
    return rows


def metric_key(row):
    return tuple(row[key] for key in ("target", "condition", "seed", "split", "graph_id"))


def validate_source_rows(source, rows, controls, audit):
    cases = [case for case in source.cases if case.split != "train"]
    seeds = source.config["model_seeds"]
    expected = {
        (target, condition, seed, case.split, case.graph_id)
        for target in TARGETS
        for condition in CONDITIONS
        for seed in ([-1] if condition in ("first", "polynomial", "fixed") else seeds)
        for case in cases
    }
    actual = {metric_key(row) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("source metrics have missing, duplicate or mismatched evaluation rows")
    wanted = {
        (target, seed, case.split, case.graph_id, treatment)
        for target in TARGETS
        for seed in seeds
        for case in cases
        for treatment in INTERVENTIONS
    }
    obtained = {
        tuple(row[key] for key in ("target", "seed", "split", "graph_id", "intervention"))
        for row in controls
    }
    if len(controls) != len(wanted) or obtained != wanted:
        raise ValueError("source interventions have incomplete or mismatched coverage")
    if len(audit) != len(source.cases) or {row["graph_id"] for row in audit} != {
        case.graph_id for case in source.cases
    }:
        raise ValueError("source teacher operator audit does not cover every graph")


def original_reproduction(source_rows, repeated, tolerance, relative_tolerance=1e-5):
    """Verify saved selected states reproduce the actual source messages."""
    actual = {metric_key(row): row for row in repeated if row["split"] != "train"}
    if len(actual) != len(source_rows):
        raise ValueError("original model reproduction coverage differs from source")
    maximum = 0.0
    for expected in source_rows:
        current = actual[metric_key(expected)]
        for field in ("message_relerr", "message_abs_rmse", "u", "v", "beta"):
            left, right = expected[field], current[field]
            if left is None or right is None:
                if left != right:
                    raise ValueError(f"original scalar/output availability changed: {field}")
                continue
            error = abs(left - right)
            maximum = max(maximum, error)
            if not math.isclose(left, right, abs_tol=tolerance, rel_tol=relative_tolerance):
                raise ArithmeticError(
                    f"saved-model reproduction failed for {metric_key(expected)} {field}: "
                    f"{left} vs {right}; atol={tolerance}, rtol={relative_tolerance}"
                )
    return {
        "graph_seed_rows": len(actual),
        "maximum_absolute_error": maximum,
        "absolute_tolerance": tolerance,
        "relative_tolerance": relative_tolerance,
        "verified": True,
    }


def choose_workers(source, config, resources, requested):
    if requested != "auto":
        count = int(requested)
        if count < 1:
            raise ValueError("workers must be positive")
        return count, [{"workers": count, "selection": "explicit override"}]
    available = available_cpus(resources)
    choices = sorted({1, *(n for n in (2, 4, 8, 16, 32, 64, available) if n <= available)})
    rows = []
    for count in choices:
        started = time.perf_counter()
        cases = feature_cases(
            source.cases, source.config["teacher"], config["master_seed"], 1.0, count
        )
        seconds = time.perf_counter() - started
        row = {
            "workers": count,
            "graphs": len(cases),
            "seconds": seconds,
            "graphs_per_second": len(cases) / seconds,
        }
        rows.append(row)
        print(
            f"[cpu calibration] workers={count} graphs={len(cases)} seconds={seconds:.3f}",
            flush=True,
        )
        del cases
    return max(rows, key=lambda row: row["graphs_per_second"])["workers"], rows


@torch.no_grad()
def choose_batch(source, models, device, dtype, requested):
    prior = source.contract["physical_graph_batch_selected"]
    if requested != "auto":
        size = int(requested)
        if size < 1:
            raise ValueError("physical batch must be positive")
        return size, [{"batch_size": size, "selection": "explicit override"}]
    sizes = sorted({min(n, len(source.cases)) for n in (prior, 2 * prior, len(source.cases))})
    trials = []
    repeats = 1 if source.config["profile"] == "debug" else 3
    for size in sizes:
        cached = cache_batches(source.cases, size, device, dtype)
        for condition in ("learned", "random_pair"):
            model = models[("path", condition)].model
            for _, batch in cached:
                prediction, weights = model(batch)
                del prediction, weights
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(repeats):
                for _, batch in cached:
                    prediction, weights = model(batch)
                    del prediction, weights
            synchronize(device)
            seconds = time.perf_counter() - started
            row = {
                "batch_size": size,
                "condition": condition,
                "passes": repeats,
                "graphs_per_pass": len(source.cases),
                "seconds": seconds,
                "graphs_per_second": len(source.cases) * repeats / seconds,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "scope": "frozen forward with all source graphs cached",
            }
            trials.append(row)
            print(
                f"[batch calibration] batch={size} condition={condition} "
                f"graphs/s={row['graphs_per_second']:.1f} "
                f"peak_bytes={row['peak_vram_bytes']}",
                flush=True,
            )
        del cached
    score = {
        size: min(row["graphs_per_second"] for row in trials if row["batch_size"] == size)
        for size in sizes
    }
    return max(score, key=score.get), trials


def run(args, output):
    config = read_config(
        args.config or Path(__file__).with_name(f"config_{args.profile}.json"), args.profile
    )
    torch.set_num_threads(1)
    print(
        f"[preflight] reading completed Experiment 2 source={args.run_dir} profile={args.profile}",
        flush=True,
    )
    source = load_source_run(args.run_dir, args.profile)
    if config["metric_epsilon"] != source.config["metric_epsilon"]:
        raise ValueError("evaluation epsilon must match the source metric")
    device, dtype = torch.device(args.device), getattr(torch, source.config["dtype"])
    torch.backends.cuda.matmul.allow_tf32 = False
    resources = runtime_resources(device)
    resources["cpu_math_threadpools"] = threadpool_info()
    storage = psutil.disk_usage(str(output))
    resources.update(storage_total_bytes=storage.total, storage_free_bytes=storage.free)
    code_before = source_manifest()
    models = load_models(source, device, dtype)
    before = {
        f"{target}/{condition}": frozen_fingerprint(frozen.model)
        for (target, condition), frozen in models.items()
    }
    old_metrics = read_rows(source.run_dir / "metrics.csv")
    old_controls = read_rows(source.run_dir / "interventions.csv")
    old_audit = read_rows(source.run_dir / "teacher_operator_audit.csv")
    validate_source_rows(source, old_metrics, old_controls, old_audit)
    write_json(output / "config.json", config)
    write_json(output / "resources_before.json", resources)
    write_json(
        output / "source_provenance.json",
        {
            "run_dir": str(source.run_dir),
            "input_sha256": source.hashes,
            "source_config": source.config,
            "source_completion": source.completion,
            "model_provenance": {f"{t}/{c}": f.provenance for (t, c), f in models.items()},
        },
    )
    print(
        f"[start] Experiment 3 profile={args.profile} device={device} dtype={dtype} "
        f"new_epochs=0 optimizer_updates=0 source_epochs={source.config['epochs']} "
        f"hidden={source.config['hidden']} model_seeds={source.config['model_seeds']}",
        flush=True,
    )
    print("[hardware] " + json.dumps(resources), flush=True)
    print(
        f"[source] completed graphs={len(source.cases)} "
        f"inputs={len(source.cases) * source.config['features']} "
        f"models={len(models)} all_nodes_edges_paths_retained=True",
        flush=True,
    )
    print(
        "[architecture] scalar_feature_channels=1 gate_linear_layers=2 "
        "gate_hidden_layers=1 attention_heads=N/A gnn_backbone_layers=N/A "
        "operator_applications=1 true_path_hops=2 sampling_ratio=1.0",
        flush=True,
    )
    for (target, condition), frozen in models.items():
        trainable_count = sum(p.numel() for p in frozen.model.parameters() if p.requires_grad)
        print(
            f"[model] target={target} condition={condition} "
            f"parameters={sum(p.numel() for p in frozen.model.parameters())} "
            f"parameters_per_seed="
            f"{sum(p.numel() for p in frozen.model.parameters()) // len(frozen.seeds)} "
            f"trainable_parameters={trainable_count} "
            f"seed_replicas={len(frozen.seeds)}",
            flush=True,
        )
    started = time.perf_counter()
    psutil.cpu_percent(interval=None)
    workers, cpu_trials = choose_workers(source, config, resources, args.workers)
    size, gpu_trials = choose_batch(source, models, device, dtype, args.batch_size)
    print(
        f"[selected] physical_graph_batch={size} workers={workers} "
        f"parallel_fields={source.config['features']} "
        f"parallel_model_seeds={len(source.config['model_seeds'])}",
        flush=True,
    )
    metrics, scale_rows, scenarios, reproduction = [], [], [], None
    fresh_base = feature_cases(
        source.cases, source.config["teacher"], config["master_seed"], 1.0, workers
    )
    reference_cached = cache_batches(fresh_base, size, device, dtype)
    scenario_specs = [
        ("original", 1.0),
        *(("fresh", amplitude) for amplitude in config["amplitudes"]),
    ]
    peak_by_scenario = []
    for scenario, amplitude in scenario_specs:
        phase_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if scenario == "original":
            cases = source.cases
            cached = cache_batches(cases, size, device, dtype)
        elif amplitude == 1:
            cases, cached = fresh_base, reference_cached
        else:
            cases = feature_cases(
                source.cases, source.config["teacher"], config["master_seed"], amplitude, workers
            )
            cached = cache_batches(cases, size, device, dtype)
        dataset_hash = source.manifest["dataset_sha256"]
        if scenario == "fresh":
            folder = output / f"fresh-a{amplitude:g}"
            folder.mkdir()
            feature_manifest = save_dataset(folder, cases)
            dataset_hash = feature_manifest["dataset_sha256"]
        print(
            f"[scenario] name={scenario} amplitude={amplitude:g} "
            f"graphs={len(cases)}/{len(source.cases)} "
            f"inputs={len(cases) * source.config['features']} no_sampling=True",
            flush=True,
        )
        scenario_rows = []
        for (target, condition), frozen in models.items():
            rows, patterns = evaluate(
                frozen.model, cached, target, condition, frozen.seeds, config["metric_epsilon"]
            )
            del patterns
            for row in rows:
                row.update(
                    scenario=scenario,
                    amplitude=amplitude,
                    source_split=row["split"],
                    feature_status="source_features"
                    if scenario == "original"
                    else "unseen_features",
                )
            scenario_rows.extend(rows)
            if scenario == "fresh":
                for reference, scaled in zip(reference_cached, cached, strict=True):
                    checks = scale_diagnostics(
                        frozen.model,
                        reference,
                        scaled,
                        amplitude,
                        frozen.seeds,
                        config["metric_epsilon"],
                        target=target,
                    )
                    for check in checks:
                        check.update(target=target, condition=condition, scenario="fresh")
                    scale_rows.extend(checks)
            summary = {
                split: seed_macro([r for r in rows if r["split"] == split], frozen.seeds).tolist()
                for split in SPLITS
            }
            print(
                f"[evaluate] scenario={scenario} amplitude={amplitude:g} target={target} "
                f"condition={condition} graphs={len(cases)} macro_relerr={json.dumps(summary)}",
                flush=True,
            )
        if scenario == "original":
            reproduction = original_reproduction(
                old_metrics,
                scenario_rows,
                config["original_reproduction_atol"],
                config["original_reproduction_rtol"],
            )
            print(
                f"[reproduction] original selected models "
                f"maximum_error={reproduction['maximum_absolute_error']:.3g}",
                flush=True,
            )
        metrics.extend(scenario_rows)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        elapsed = time.perf_counter() - phase_start
        scenarios.append(
            {
                "scenario": scenario,
                "amplitude": amplitude,
                "graphs": len(cases),
                "inputs": len(cases) * source.config["features"],
                "seconds": elapsed,
                "dataset_sha256": dataset_hash,
                "dataset_file": str(source.run_dir / "dataset.npz")
                if scenario == "original"
                else f"fresh-a{amplitude:g}/dataset.npz",
            }
        )
        peak_by_scenario.append(
            {"scenario": scenario, "amplitude": amplitude, "peak_vram_bytes": peak}
        )
        print(
            f"[scenario complete] name={scenario} amplitude={amplitude:g} "
            f"seconds={elapsed:.2f} peak_bytes={peak}",
            flush=True,
        )
        if cached is not reference_cached:
            del cached
    expected = {
        (scenario, amplitude, target, condition, seed, case.split, case.graph_id)
        for scenario, amplitude in scenario_specs
        for target in TARGETS
        for condition in CONDITIONS
        for seed in models[(target, condition)].seeds
        for case in source.cases
    }
    keys = {
        tuple(
            row[key]
            for key in ("scenario", "amplitude", "target", "condition", "seed", "split", "graph_id")
        )
        for row in metrics
    }
    wanted_scales = {key for key in expected if key[0] == "fresh"}
    actual_scales = {
        tuple(
            row[key]
            for key in ("scenario", "amplitude", "target", "condition", "seed", "split", "graph_id")
        )
        for row in scale_rows
    }
    if keys != expected or len(metrics) != len(expected):
        raise AssertionError("feature evaluation has missing, duplicate or mislabeled rows")
    if actual_scales != wanted_scales or len(scale_rows) != len(wanted_scales):
        raise AssertionError("scale diagnostics have incomplete treatment coverage")
    after = {
        f"{target}/{condition}": frozen_fingerprint(frozen.model)
        for (target, condition), frozen in models.items()
    }
    if before != after:
        raise AssertionError("source models changed during the frozen evaluation")
    if source.hashes != {name: file_hash(source.run_dir / name) for name in source.hashes}:
        raise RuntimeError("source artifacts changed during evaluation")
    if code_before != source_manifest():
        raise RuntimeError("Experiment 3 source code/config changed during evaluation")
    graph_stats = {
        split: {
            "graphs": sum(case.split == split for case in source.cases),
            "nodes": [case.num_nodes for case in source.cases if case.split == split],
            "edges": [case.edges.shape[1] for case in source.cases if case.split == split],
            "paths": [case.wedges.shape[1] for case in source.cases if case.split == split],
        }
        for split in SPLITS
    }
    contract = {
        "experiment": 3,
        "profile": args.profile,
        "debug": args.profile == "debug",
        "config": config,
        "source_dir": str(source.run_dir),
        "source_config": source.config,
        "source_artifact_hashes": source.hashes,
        "source_model_hashes": before,
        "source_data_hash": source.manifest["dataset_sha256"],
        "fresh_data_hashes": {
            f"a{row['amplitude']:g}": row["dataset_sha256"]
            for row in scenarios
            if row["scenario"] == "fresh"
        },
        "model_hashes_after": after,
        "models_unchanged": True,
        "source_artifacts_unchanged": True,
        "source_code_sha256": code_before,
        "source_code_unchanged": True,
        "original_metric_reproduction": reproduction,
        "source_graph_count": len(source.cases),
        "source_input_count": len(source.cases) * source.config["features"],
        "feature_realizations": source.config["features"],
        "scalar_feature_channels": 1,
        "gate_linear_layers": 2,
        "gate_hidden_layers": 1,
        "gate_hidden_dimension": source.config["hidden"],
        "gnn_backbone_layers": None,
        "attention_heads": None,
        "operator_applications": 1,
        "true_path_hops": 2,
        "time_window": None,
        "input_resolution": None,
        "sampling_ratio": 1.0,
        "model_seeds": source.config["model_seeds"],
        "new_training_epochs": 0,
        "optimizer_updates": 0,
        "checkpoint_selection_updates": 0,
        "scalar_refits": 0,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "amplitudes": config["amplitudes"],
        "graph_treatments": len(source.cases) * len(scenarios),
        "input_treatments": len(source.cases) * source.config["features"] * len(scenarios),
        "data_fraction": 1.0,
        "path_sampling": False,
        "graph_statistics": graph_stats,
        "physical_graph_batch_selected": size,
        "physical_batch_actual": [batch.num_graphs for _, batch in reference_cached],
        "input_shapes": [list(batch.x.shape) for _, batch in reference_cached],
        "gradient_accumulation": 1,
        "effective_inference_batch": size * source.config["features"],
        "data_parallel_workers": 1,
        "seed_axis_is_independent_models_not_DDP": True,
        "feature_amplitudes_are_paired_not_independent_graph_draws": True,
        "new_feature_seed": config["master_seed"],
        "feature_seed_stream": "wedge-generalization-v1",
        "cpu_workers": workers,
        "source_validation_workers": source.validation_workers,
        "torch_intraop_threads": 1,
        "static_gpu_cache": device.type == "cuda",
        "data_loader_workers": None,
        "data_loader_reason": (
            "source NPZ loaded once; static geometry and complete graph batches cached"
        ),
        "pin_memory": device.type == "cuda",
        "non_blocking_transfer": device.type == "cuda",
        "prefetch": "complete active scenario and amplitude1 reference cached",
        "dtype": source.config["dtype"],
        "teacher_reference_dtype": "float64",
        "tf32": False,
        "resources": resources,
        "cpu_calibration": cpu_trials,
        "gpu_batch_calibration": gpu_trials,
        "peak_by_scenario": peak_by_scenario,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "seconds_before_report": time.perf_counter() - started,
        "metric_rows": len(metrics),
        "scale_rows": len(scale_rows),
        "source_metric_rows": len(old_metrics),
        "source_intervention_rows": len(old_controls),
        "source_teacher_operator_audit_rows": len(old_audit),
        "real_dataset_classification": False,
    }
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "scale_checks.csv", scale_rows)
    write_csv(output / "source_metrics.csv", old_metrics)
    write_csv(output / "source_interventions.csv", old_controls)
    write_csv(output / "source_teacher_operator_audit.csv", old_audit)
    write_json(output / "contract.json", contract)
    print(
        "[report] writing actual feature/amplitude results and source intervention evidence",
        flush=True,
    )
    write_report(output, metrics, scale_rows, old_metrics, old_controls, old_audit, contract)
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "experiment": 3,
            "profile": args.profile,
            "graphs": len(source.cases),
            "metric_rows": len(metrics),
            "scale_rows": len(scale_rows),
            "models_unchanged": True,
            "source_artifacts_unchanged": True,
            "seconds": time.perf_counter() - started,
        },
    )
    print(f"[complete] Experiment 3 profile={args.profile}; results={output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="completed Experiment 2 result directory, read only",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--batch-size", default="auto")
    args = parser.parse_args()
    output, source = args.output_dir.resolve(), args.run_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError("use a separate new output directory outside the source experiment")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "terminal.log").open("x", encoding="utf-8", buffering=1) as logfile:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, logfile)),
            contextlib.redirect_stderr(Tee(sys.stderr, logfile)),
        ):
            try:
                with threadpool_limits(limits=1):
                    run(args, output)
            except Exception as error:
                traceback.print_exc()
                write_json(
                    output / "failure.json",
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "source_dir": str(source),
                        "output_dir": str(output),
                    },
                )
                print(
                    "[failed] source preserved; inspect terminal.log and failure.json", flush=True
                )
                raise


if __name__ == "__main__":
    main()
