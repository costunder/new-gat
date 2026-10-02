"""Experiment 3.1: train RMS-normalized C inputs, keep actual messages unchanged."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path

import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from ..generalization.data import file_hash, load_source_run
from ..generalization.frozen import FrozenModel, frozen_fingerprint, load_models, scale_diagnostics
from ..generalization.study import (
    AMPLITUDES,
    SPLITS,
    cache_original_batches,
    choose_batch,
    choose_workers,
    original_reproduction,
    read_rows,
    target_rms_scales,
)
from ..learned.evaluation import evaluate, intervention_manifest, interventions, seed_macro
from ..learned.model import normalized_mse
from ..learned.train import (
    CONDITIONS,
    INTERVENTIONS,
    TARGETS,
    cache_batches,
    config_digest,
    train_gate,
)
from ..study import Tee, runtime_resources, synchronize, write_csv, write_json
from ..study import source_manifest as repository_manifest
from .data import load_feature_source
from .model import make_normalized_model
from .report import write_report

VARIANTS = ("raw", "normalized")


def read_config(path, profile):
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.loads(
        Path(__file__).with_name(f"config_{profile}.json").read_text(encoding="utf-8")
    )
    if config != canonical or config["profile"] != profile:
        raise ValueError("Experiment 3.1 requires its complete canonical full/debug contract")
    return config


def source_manifest():
    root = Path(__file__).resolve().parent.parent
    names = ["data.py", "operators.py", "study.py"]
    for folder in ("learned", "generalization", "scale_normalization"):
        names.extend(path.relative_to(root).as_posix() for path in (root / folder).glob("*.py"))
        names.extend(path.relative_to(root).as_posix() for path in (root / folder).glob("*.json"))
    return {name: file_hash(root / name) for name in sorted(names)}


def output_location(output, *sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("use a new output directory separate from both completed source runs")
    return output


def calibrate_training(source, device, dtype):
    """Measure whole-corpus alternatives; retain source batch for matched updates."""
    config = source.config
    cases = [case for case in source.cases if case.split == "train"]
    prior = source.contract["physical_graph_batch_selected"]
    sizes = sorted({min(value, len(cases)) for value in (64, 128, prior)})
    trials = []
    repeats = 1 if config["profile"] == "debug" else 3
    for size in sizes:
        cached = cache_batches(cases, size, device, dtype)
        for condition in CONDITIONS[3:]:
            model = make_normalized_model(
                condition, config["model_seeds"], config["hidden"], config["teacher"]["tau"]
            ).to(device=device, dtype=dtype)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            for _, batch in cached:
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(batch)
                normalized_mse(
                    prediction,
                    batch.targets["path"],
                    batch.node_graph,
                    batch.num_graphs,
                    config["loss_epsilon"],
                ).sum().backward()
                optimizer.step()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(repeats):
                for _, batch in cached:
                    optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(batch)
                    normalized_mse(
                        prediction,
                        batch.targets["path"],
                        batch.node_graph,
                        batch.num_graphs,
                        config["loss_epsilon"],
                    ).sum().backward()
                    optimizer.step()
            synchronize(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            trials.append(
                dict(
                    batch_size=size,
                    condition=condition,
                    graphs=len(cases),
                    passes=repeats,
                    seconds=seconds,
                    graphs_per_second=len(cases) * repeats / seconds,
                    peak_vram_bytes=peak,
                )
            )
            print(
                f"[training calibration] batch={size} condition={condition} "
                f"graphs/s={trials[-1]['graphs_per_second']:.1f} peak_bytes={peak}",
                flush=True,
            )
            del optimizer, model, prediction
        del cached
    print(
        f"[training batch] source physical_graph_batch={prior} retained; "
        "same graph grouping, loss, Adam steps and seed initialization",
        flush=True,
    )
    return prior, trials


def restore_checkpoint(path, config, source_hashes, batch_size, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if (
        saved.get("format_version") != 1
        or saved.get("config") != config
        or saved.get("config_hash") != config_digest(config)
        or saved.get("source", {}).get("sha256") != source_hashes
        or saved.get("condition") not in CONDITIONS[3:]
        or saved.get("target") not in TARGETS
        or saved.get("physical_graph_batch") != batch_size
        or not 0 <= saved.get("epoch", -1) <= config["epochs"]
    ):
        raise ValueError("resume checkpoint must match this normalized job, source and batch")
    return saved


def annotate(rows, variant, scenario, amplitude):
    return [
        dict(
            row,
            variant=variant,
            scenario=scenario,
            amplitude=amplitude,
            source_split=row["split"],
            feature_status=("source_features" if scenario == "original" else "unseen_features"),
        )
        for row in rows
    ]


def validate_coverage(rows, scales, controls, source, models):
    fields = (
        "variant",
        "scenario",
        "amplitude",
        "target",
        "condition",
        "seed",
        "split",
        "graph_id",
    )
    scenarios = [("original", 1.0), *(("fresh", a) for a in AMPLITUDES)]
    expected = {
        (variant, scenario, a, target, condition, seed, case.split, case.graph_id)
        for variant in VARIANTS
        for scenario, a in scenarios
        for target in TARGETS
        for condition in CONDITIONS
        for seed in models[variant][(target, condition)].seeds
        for case in source.cases
    }
    actual = {tuple(row[field] for field in fields) for row in rows}
    wanted_scales = {key for key in expected if key[1] == "fresh"}
    actual_scales = {tuple(row[field] for field in fields) for row in scales}
    wanted_controls = {
        (*key, treatment)
        for key in expected
        if key[4] == "learned" and key[2] == 1
        for treatment in INTERVENTIONS
    }
    actual_controls = {
        tuple(row[field] for field in fields) + (row["intervention"],) for row in controls
    }
    for name, values, keys, wanted in (
        ("metrics", rows, actual, expected),
        ("scales", scales, actual_scales, wanted_scales),
        ("interventions", controls, actual_controls, wanted_controls),
    ):
        if len(values) != len(wanted) or keys != wanted:
            raise AssertionError(f"{name} has missing, duplicate or mismatched treatments")


def run(args, output):
    config = read_config(
        args.config or Path(__file__).with_name(f"config_{args.profile}.json"), args.profile
    )
    output_location(output, args.source_dir, args.feature_source_dir)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    resources = runtime_resources(device)
    if resources["gpu_count_visible"] > 1:
        raise RuntimeError(
            "select the single allocated GPU with CUDA_VISIBLE_DEVICES for this study"
        )
    resources["cpu_math_threadpools"] = threadpool_info()
    storage = psutil.disk_usage(str(output))
    resources.update(storage_free_bytes=storage.free, storage_total_bytes=storage.total)
    print("[hardware] " + json.dumps(resources), flush=True)
    print(
        f"[preflight] complete source={args.source_dir}; feature_source={args.feature_source_dir}",
        flush=True,
    )
    started = time.perf_counter()
    psutil.cpu_percent(interval=None)
    code_before = source_manifest()
    source = load_source_run(args.source_dir, args.profile)
    source_validation_workers = source.validation_workers
    dtype = getattr(torch, source.config["dtype"])
    workers, cpu_trials = choose_workers(source, config, resources, args.workers)
    source = replace(source, validation_workers=workers)
    feature_source = load_feature_source(args.feature_source_dir, source, args.profile)
    raw = load_models(source, device, dtype)
    raw_before = {f"{t}/{c}": frozen_fingerprint(model) for (t, c), model in raw.items()}
    write_json(output / "config.json", config)
    write_json(output / "resources_before.json", resources)
    write_json(
        output / "source_provenance.json",
        dict(
            source_dir=str(source.run_dir),
            source_hashes=source.hashes,
            source_config=source.config,
            feature_source_dir=str(feature_source.directory),
            feature_source_hashes=feature_source.hashes,
            raw_model_provenance={f"{t}/{c}": m.provenance for (t, c), m in raw.items()},
        ),
    )
    print(
        f"[start] Experiment 3.1 profile={args.profile} epochs={source.config['epochs']} "
        f"hidden={source.config['hidden']} parallel_seeds={source.config['model_seeds']} "
        f"graphs={len(source.cases)} features={source.config['features']} "
        "normalization=graph_edge_rms",
        flush=True,
    )
    print(
        "[architecture] gate=4-hidden-ReLU-1; scalar_channels=1; "
        "gate_linear_layers=2 gate_hidden_layers=1 operator_applications=1 true_path_hops=2 "
        "gnn_backbone=N/A heads=N/A sampling_ratio=1.0; AX unchanged; teacher_loss=False",
        flush=True,
    )
    size, train_trials = calibrate_training(source, device, dtype)
    original_cached, layout = cache_original_batches(source, device, dtype)
    cached_by_split = {
        split: [(cases, batch) for cases, batch in original_cached if cases[0].split == split]
        for split in SPLITS
    }
    training_config = dict(
        source.config,
        experiment="3.1",
        normalization=config["normalization"],
        source_data_sha256=source.manifest["dataset_sha256"],
    )
    train_source = dict(sha256=code_before, git=repository_manifest()["git_commit"])
    resumed = (
        restore_checkpoint(args.resume_from, training_config, code_before, size, device)
        if args.resume_from
        else None
    )
    normalized, history, jobs = {}, [], []
    for target in TARGETS:
        for condition in CONDITIONS:
            if condition in CONDITIONS[:3]:
                normalized[(target, condition)] = raw[(target, condition)]
                continue
            model = make_normalized_model(
                condition,
                source.config["model_seeds"],
                source.config["hidden"],
                source.config["teacher"]["tau"],
            ).to(device=device, dtype=dtype)
            parameters = sum(p.numel() for p in model.parameters())
            print(
                f"[model] target={target} condition={condition} variant=normalized "
                f"parameters={parameters} parameters_per_seed={parameters // len(model.seeds)} "
                f"trainable_parameters={parameters}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            resume = (
                resumed
                if resumed and (resumed["target"], resumed["condition"]) == (target, condition)
                else None
            )
            current_history, selected = train_gate(
                model,
                cached_by_split["train"],
                cached_by_split["validation"],
                training_config,
                target,
                condition,
                output,
                size,
                train_source,
                resume=resume,
            )
            history.extend(dict(row, variant="normalized") for row in current_history)
            model.requires_grad_(False).eval()
            provenance = {
                name: selected[name]
                for name in (
                    "target",
                    "condition",
                    "seeds",
                    "best_epochs",
                    "validation_message_relerr",
                )
            }
            normalized[(target, condition)] = FrozenModel(model, model.seeds, provenance)
            jobs.append(
                dict(
                    provenance,
                    parameters_total=parameters,
                    parameters_per_seed=parameters // len(model.seeds),
                    peak_vram_bytes=(
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    ),
                )
            )
    models = {"raw": raw, "normalized": normalized}
    frozen_before = {
        f"{v}/{t}/{c}": frozen_fingerprint(m)
        for v, group in models.items()
        for (t, c), m in group.items()
    }
    evaluation_size = feature_source.contract["physical_graph_batch_selected"]
    _, inference_trials = choose_batch(source, normalized, device, dtype, "auto")
    print(
        f"[evaluation batch] source Experiment 3 physical_graph_batch={evaluation_size} "
        "retained for paired comparisons",
        flush=True,
    )
    fresh_base = feature_source.cases_by_amplitude[1.0]
    reference_cached = cache_batches(fresh_base, evaluation_size, device, dtype)
    if [batch.num_graphs for _, batch in reference_cached] != feature_source.contract[
        "physical_batch_actual"
    ] or [list(batch.x.shape) for _, batch in reference_cached] != feature_source.contract[
        "input_shapes"
    ]:
        raise ValueError("fresh input batching differs from the recorded Experiment 3 layout")
    old_rows = read_rows(source.run_dir / "metrics.csv")
    old_feature_rows = read_rows(feature_source.directory / "metrics.csv")
    rows, scales, controls, replay, scenarios, peaks = [], [], [], [], [], []
    for scenario, amplitude in [("original", 1.0), *(("fresh", a) for a in AMPLITUDES)]:
        phase_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cases = (
            source.cases if scenario == "original" else feature_source.cases_by_amplitude[amplitude]
        )
        cached = (
            original_cached
            if scenario == "original"
            else (
                reference_cached
                if amplitude == 1
                else cache_batches(cases, evaluation_size, device, dtype)
            )
        )
        print(f"[scenario] {scenario} amplitude={amplitude:g} all_graphs={len(cases)}", flush=True)
        plan = (
            intervention_manifest(cases, source.config["master_seed"]) if amplitude == 1 else None
        )
        scenario_raw = []
        baseline_rows, baseline_scales = {}, {}
        for variant, group in models.items():
            for (target, condition), frozen in group.items():
                key = (target, condition)
                if variant == "normalized" and condition in CONDITIONS[:3]:
                    measured, patterns = baseline_rows[key], None
                else:
                    measured, patterns = evaluate(
                        frozen.model,
                        cached,
                        target,
                        condition,
                        frozen.seeds,
                        source.config["metric_epsilon"],
                    )
                    if condition in CONDITIONS[:3]:
                        baseline_rows[key] = measured
                annotated = annotate(measured, variant, scenario, amplitude)
                rows.extend(annotated)
                if variant == "raw":
                    scenario_raw.extend(annotated)
                if condition == "learned" and amplitude == 1:
                    changes = interventions(
                        frozen.model,
                        cached,
                        patterns,
                        plan,
                        target,
                        frozen.seeds,
                        source.config["metric_epsilon"],
                    )
                    controls.extend(annotate(changes, variant, scenario, amplitude))
                del patterns
                if scenario == "fresh":
                    if variant == "normalized" and condition in CONDITIONS[:3]:
                        checks = baseline_scales[key]
                    else:
                        checks = []
                        for reference, scaled in zip(reference_cached, cached, strict=True):
                            checks.extend(
                                scale_diagnostics(
                                    frozen.model,
                                    reference,
                                    scaled,
                                    amplitude,
                                    frozen.seeds,
                                    source.config["metric_epsilon"],
                                    target=target,
                                )
                            )
                        if condition in CONDITIONS[:3]:
                            baseline_scales[key] = checks
                    scales.extend(
                        annotate(
                            [dict(row, target=target, condition=condition) for row in checks],
                            variant,
                            scenario,
                            amplitude,
                        )
                    )
                summary = {
                    split: seed_macro(
                        [row for row in measured if row["split"] == split], frozen.seeds
                    ).tolist()
                    for split in SPLITS
                }
                print(
                    f"[evaluate] variant={variant} target={target} condition={condition} "
                    f"scenario={scenario} amplitude={amplitude:g} "
                    f"relative_error={json.dumps(summary)}",
                    flush=True,
                )
        truth_scale = target_rms_scales(cases, dtype)
        expected = (
            old_rows
            if scenario == "original"
            else [
                row
                for row in old_feature_rows
                if row["scenario"] == scenario and row["amplitude"] == amplitude
            ]
        )
        # Helper checks the five nontrain splits; source train replay is additional evidence.
        expected = [row for row in expected if row["split"] != "train"]
        measured_replay = [
            dict(row, reproduction_target_rms=truth_scale[(row["target"], row["graph_id"])])
            for row in scenario_raw
        ]
        write_csv(output / f"raw-replay-{scenario}-a{amplitude:g}.csv", measured_replay)
        guard = original_reproduction(
            expected,
            scenario_raw,
            1e-5,
            1e-5,
            target_rms=truth_scale,
            epsilon=source.config["metric_epsilon"],
        )
        replay.append(dict(guard, scenario=scenario, amplitude=amplitude))
        print(
            f"[raw replay] {scenario} amplitude={amplitude:g} verified=True "
            f"max_normalized_rmse={guard['maximum_target_normalized_rmse_error']:.3g}",
            flush=True,
        )
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        peaks.append(dict(scenario=scenario, amplitude=amplitude, peak_vram_bytes=peak))
        scenarios.append(
            dict(
                scenario=scenario,
                amplitude=amplitude,
                graphs=len(cases),
                seconds=time.perf_counter() - phase_start,
            )
        )
        print(
            f"[scenario complete] {scenario} amplitude={amplitude:g} "
            f"seconds={scenarios[-1]['seconds']:.2f}",
            flush=True,
        )
        if cached is not original_cached and cached is not reference_cached:
            del cached
    validate_coverage(rows, scales, controls, source, models)
    frozen_after = {
        f"{v}/{t}/{c}": frozen_fingerprint(m)
        for v, group in models.items()
        for (t, c), m in group.items()
    }
    if frozen_before != frozen_after or raw_before != {
        f"{t}/{c}": frozen_fingerprint(m) for (t, c), m in raw.items()
    }:
        raise AssertionError(
            "a frozen source or selected normalized model changed during evaluation"
        )
    for origin in (source, feature_source):
        directory = origin.run_dir if hasattr(origin, "run_dir") else origin.directory
        if origin.hashes != {name: file_hash(directory / name) for name in origin.hashes}:
            raise RuntimeError("completed source artifacts changed during Experiment 3.1")
    if source_manifest() != code_before:
        raise RuntimeError("Experiment 3.1 mathematical source changed during the run")
    contract = dict(
        experiment="3.1",
        profile=args.profile,
        debug=args.profile == "debug",
        config=config,
        source_config=source.config,
        training_config=training_config,
        source_dir=str(source.run_dir),
        feature_source_dir=str(feature_source.directory),
        source_artifact_hashes=source.hashes,
        feature_source_artifact_hashes=feature_source.hashes,
        source_code_sha256=code_before,
        source_data_hash=source.manifest["dataset_sha256"],
        source_code_unchanged=True,
        source_artifacts_unchanged=True,
        feature_source_artifacts_unchanged=True,
        models_unchanged=True,
        raw_model_hashes_before=raw_before,
        frozen_model_hashes_before=frozen_before,
        frozen_model_hashes_after=frozen_after,
        original_metric_reproduction=replay[0],
        raw_metric_reproductions=replay,
        original_batch_layout=layout,
        graph_count_total=len(source.cases),
        graph_count_used=len(source.cases),
        data_fraction=1.0,
        split_graph_counts=dict(Counter(case.split for case in source.cases)),
        feature_realizations=source.config["features"],
        input_count_total=len(source.cases) * source.config["features"],
        new_training_epochs=source.config["epochs"],
        normalized_gate_jobs=6,
        model_seeds=source.config["model_seeds"],
        raw_optimizer_updates=0,
        test_updates=0,
        scalar_refits=0,
        validation_selection_only=True,
        teacher_weight_loss=False,
        scale_augmentation=False,
        normalized_jobs=jobs,
        training_physical_graph_batch=size,
        gradient_accumulation=1,
        data_parallel_workers=1,
        effective_training_graph_batch_per_seed=size,
        effective_training_fields_per_seed=size * source.config["features"],
        optimizer_updates_per_seed_per_gate_job=len(cached_by_split["train"])
        * source.config["epochs"],
        protocol_optimizer_calls=6 * len(cached_by_split["train"]) * source.config["epochs"],
        optimizer_calls_in_study=len(cached_by_split["train"])
        * (6 * source.config["epochs"] - (resumed["epoch"] if resumed else 0)),
        inference_physical_graph_batch=evaluation_size,
        training_input_shapes=[list(b.x.shape) for _, b in cached_by_split["train"]],
        inference_input_shapes=[list(b.x.shape) for _, b in reference_cached],
        precision=source.config["dtype"],
        teacher_reference_precision="float64",
        tf32=False,
        sampling_ratio=1.0,
        path_sampling=False,
        cpu_workers=workers,
        cpu_calibration=cpu_trials,
        source_validation_workers=source_validation_workers,
        fresh_validation_workers=workers,
        training_batch_calibration=train_trials,
        inference_batch_calibration=inference_trials,
        batch_selection_reason=(
            "retain completed source training and evaluation layouts for paired comparison"
        ),
        static_gpu_cache=device.type == "cuda",
        data_loader_workers=None,
        data_loader_reason="complete source NPZ and static disjoint graph batches cached",
        pin_memory=device.type == "cuda",
        non_blocking_transfer=device.type == "cuda",
        prefetch="original batches plus active scenario and amplitude1 reference cached",
        graph_statistics={
            split: {
                "nodes": [c.num_nodes for c in source.cases if c.split == split],
                "edges": [c.edges.shape[1] for c in source.cases if c.split == split],
                "paths": [c.wedges.shape[1] for c in source.cases if c.split == split],
            }
            for split in SPLITS
        },
        resources=resources,
        peak_by_scenario=peaks,
        scenarios=scenarios,
        process_rss_bytes=psutil.Process().memory_info().rss,
        system_cpu_percent=psutil.cpu_percent(interval=None),
        seconds_before_report=time.perf_counter() - started,
        metric_rows=len(rows),
        scale_rows=len(scales),
        intervention_rows=len(controls),
        real_dataset_classification=False,
        resume_from=str(args.resume_from) if args.resume_from else None,
        resume_scope=(
            "one matching normalized gate job resumes; other jobs train in this new directory"
        ),
    )
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "scale_checks.csv", scales)
    write_csv(output / "interventions.csv", controls)
    write_csv(output / "training.csv", history)
    write_json(output / "contract.json", contract)
    print("[report] writing measured raw/normalized comparisons and scientific figures", flush=True)
    write_report(output, rows, scales, controls, contract)
    if source_manifest() != code_before:
        raise RuntimeError("Experiment 3.1 source changed while writing its report")
    for origin in (source, feature_source):
        directory = origin.run_dir if hasattr(origin, "run_dir") else origin.directory
        if origin.hashes != {name: file_hash(directory / name) for name in origin.hashes}:
            raise RuntimeError("completed source artifacts changed while writing the report")
    write_json(
        output / "completion.json",
        dict(
            status="complete",
            experiment="3.1",
            profile=args.profile,
            graphs=len(source.cases),
            metric_rows=len(rows),
            scale_rows=len(scales),
            intervention_rows=len(controls),
            seconds=time.perf_counter() - started,
        ),
    )
    print(f"[complete] Experiment 3.1 profile={args.profile}; results={output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--feature-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    output = output_location(args.output_dir, args.source_dir, args.feature_source_dir)
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
                    dict(
                        status="failed",
                        type=type(error).__name__,
                        message=str(error),
                        recovery=(
                            "inspect terminal.log; use a new output directory; "
                            "resume a matching checkpoint if needed"
                        ),
                    ),
                )
                raise


if __name__ == "__main__":
    main()
