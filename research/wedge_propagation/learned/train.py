"""Full Experiment 2; train only on synthetic message targets, evaluate frozen seeds."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from ..study import Tee, available_cpus, runtime_resources, synchronize, write_csv, write_json
from .data import pack_cases, prepare_cases, save_dataset
from .diagnostics import operator_audit
from .evaluation import evaluate, intervention_manifest, interventions, seed_macro
from .model import fit_baseline, make_model, normalized_mse
from .report import write_report

TARGETS = ("L", "L2", "path")
CONDITIONS = ("first", "polynomial", "fixed", "learned", "random_pair")
SPLITS = ("validation", "id", "size_ood", "family_ood", "family_size_ood")
INTERVENTIONS = (
    "identity",
    "mean",
    "weight_shuffle",
    "other_graph_pattern",
    "correspondence_randomization",
)


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def read_config(path, profile):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["profile"] != profile:
        raise ValueError("config profile and --profile disagree")
    if tuple(config["targets"]) != TARGETS or tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("all three targets and all five conditions are required")
    if tuple(config.get("interventions", ())) != INTERVENTIONS:
        raise ValueError("all five exact interventions are required")
    for name in (
        "hidden",
        "epochs",
        "validation_every",
        "features",
        "train_draws",
        "val_draws",
        "id_draws",
        "size_ood_draws",
    ):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    seeds = config["model_seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("model_seeds must be distinct nonnegative integers")
    if config["dtype"] not in ("float32", "float64"):
        raise ValueError("only explicit float32 or float64 is supported")
    for value in (
        config["learning_rate"],
        config["loss_epsilon"],
        config["metric_epsilon"],
        config["teacher"]["tau"],
        config["teacher"]["epsilon"],
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("learning rate, temperature and epsilons must be positive and finite")
    return config


def source_manifest():
    folder = Path(__file__).parent
    files = [
        *folder.glob("*.py"),
        *folder.glob("*.json"),
        folder.parent / "operators.py",
        folder.parent / "data.py",
        folder.parent / "study.py",
    ]
    from ..study import source_manifest as fixed_source

    return {
        "sha256": {
            str(file.relative_to(folder.parent)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(files)
        },
        "git": fixed_source()["git_commit"],
    }


def cache_batches(cases, size, device, dtype):
    if size < 1 or not cases:
        raise ValueError("positive batch size and nonempty split are required")
    return [
        (cases[start : start + size], pack_cases(cases[start : start + size], device, dtype))
        for start in range(0, len(cases), size)
    ]


def choose_workers(config, resources, requested):
    if requested != "auto":
        count = int(requested)
        if count < 1:
            raise ValueError("workers must be positive")
        return count, [{"workers": count, "selection": "explicit override"}]
    available = available_cpus(resources)
    candidates = sorted({1, *(n for n in (2, 4, 8, 16) if n <= available)})
    rows = []
    for count in candidates:
        start = time.perf_counter()
        cases = prepare_cases(config, count)
        elapsed = time.perf_counter() - start
        row = {
            "workers": count,
            "graphs": len(cases),
            "seconds": elapsed,
            "graphs_per_second": len(cases) / elapsed,
        }
        rows.append(row)
        print(
            f"[cpu calibration] workers={count} graphs={len(cases)} seconds={elapsed:.3f}",
            flush=True,
        )
        del cases
    return max(rows, key=lambda row: row["graphs_per_second"])["workers"], rows


def choose_batch_size(cases, config, device, dtype, requested):
    if requested != "auto":
        size = int(requested)
        if size < 1:
            raise ValueError("batch size must be positive")
        return size, [{"batch_size": size, "selection": "explicit override"}]
    candidates = sorted({min(n, len(cases)) for n in (4, 16, 64, 128, len(cases))})
    repeats = 1 if config["profile"] == "debug" else 3
    trials = []
    for size in candidates:
        cached = cache_batches(cases, size, device, dtype)
        synchronize(device)
        for condition in ("learned", "random_pair"):
            model = make_model(
                condition, config["model_seeds"], config["hidden"], config["teacher"]["tau"]
            ).to(device=device, dtype=dtype)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            # Whole train corpus is used in every pass; no graph/path/feature cap.
            for _, batch in cached:
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(batch)
                loss = normalized_mse(
                    prediction,
                    batch.targets["path"],
                    batch.node_graph,
                    batch.num_graphs,
                    config["loss_epsilon"],
                )
                loss.sum().backward()
                optimizer.step()
                del prediction, loss, batch
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(repeats):
                for _, batch in cached:
                    optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(batch)
                    loss = normalized_mse(
                        prediction,
                        batch.targets["path"],
                        batch.node_graph,
                        batch.num_graphs,
                        config["loss_epsilon"],
                    )
                    loss.sum().backward()
                    optimizer.step()
                    del prediction, loss, batch
            synchronize(device)
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            row = {
                "batch_size": size,
                "condition": condition,
                "graphs_per_pass": len(cases),
                "seed_replicas": len(config["model_seeds"]),
                "passes": repeats,
                "seconds": seconds,
                "graphs_per_second": len(cases) * repeats / seconds,
                "peak_vram_bytes": peak,
                "scope": "full-train cache plus forward/backward/optimizer; no epoch H2D",
            }
            trials.append(row)
            print(
                f"[batch calibration] batch={size} condition={condition} "
                f"graphs/s={row['graphs_per_second']:.1f} peak_bytes={peak}",
                flush=True,
            )
            del optimizer, model
        del cached
    scores = {
        size: min(row["graphs_per_second"] for row in trials if row["batch_size"] == size)
        for size in candidates
    }
    return max(scores, key=scores.get), trials


def _checkpoint(
    path,
    model,
    optimizer,
    epoch,
    target,
    condition,
    config,
    batch_size,
    best_state,
    best_scores,
    best_epochs,
    history,
    source,
):
    value = {
        "format_version": 1,
        "config": config,
        "config_hash": config_digest(config),
        "source": source,
        "condition": condition,
        "target": target,
        "epoch": epoch,
        "physical_graph_batch": batch_size,
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "best_model": {name: tensor.detach().cpu() for name, tensor in best_state.items()},
        "best_scores": best_scores.tolist(),
        "best_epochs": best_epochs.tolist(),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    with path.open("xb") as stream:
        torch.save(value, stream)
    print(f"[checkpoint] {path}", flush=True)


def restore_checkpoint(path, config, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["format_version"] != 1 or saved["config_hash"] != config_digest(config):
        raise ValueError("resume checkpoint config/profile does not match")
    if saved["condition"] not in ("learned", "random_pair") or saved["target"] not in TARGETS:
        raise ValueError("resume checkpoint does not describe a phase2 gate job")
    if saved["source"]["sha256"] != source_manifest()["sha256"]:
        raise ValueError("resume source code/config files changed; use the matching revision")
    return saved


def train_gate(
    model,
    cached_train,
    cached_val,
    config,
    target,
    condition,
    output,
    batch_size,
    source,
    resume=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    seeds = config["model_seeds"]
    initial_rows, _ = evaluate(
        model, cached_val, target, condition, seeds, config["metric_epsilon"]
    )
    best_scores = seed_macro(initial_rows, seeds)
    best_epochs = np.zeros(len(seeds), dtype=np.int64)
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    history, initial_epoch = [], 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        best_state = {
            name: tensor.to(device=next(model.parameters()).device)
            for name, tensor in resume["best_model"].items()
        }
        best_scores = np.asarray(resume["best_scores"], dtype=np.float64)
        best_epochs = np.asarray(resume["best_epochs"], dtype=np.int64)
        history, initial_epoch = list(resume["history"]), int(resume["epoch"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if resume["cuda_rng_state"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng_state"]])
        print(f"[resume] target={target} condition={condition} epoch={initial_epoch}", flush=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    previous_seconds = history[-1]["seconds"] if history else 0.0
    start = time.perf_counter()
    for epoch in range(initial_epoch + 1, config["epochs"] + 1):
        model.train()
        epoch_loss = next(model.parameters()).new_zeros(len(seeds))
        graph_count = 0
        order = np.random.default_rng(config["master_seed"] + epoch).permutation(len(cached_train))
        for index in order:
            _, batch = cached_train[int(index)]
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch)
            loss = normalized_mse(
                prediction,
                batch.targets[target],
                batch.node_graph,
                batch.num_graphs,
                config["loss_epsilon"],
            )
            loss.sum().backward()
            optimizer.step()
            epoch_loss += loss.detach() * batch.num_graphs
            graph_count += batch.num_graphs
            del prediction, loss, batch
        train_loss = (epoch_loss / graph_count).detach().cpu().double().numpy()
        if not np.isfinite(train_loss).all():
            raise ArithmeticError(f"nonfinite training loss at epoch {epoch}")
        check = epoch == 1 or epoch % config["validation_every"] == 0 or epoch == config["epochs"]
        if check:
            val_rows, _ = evaluate(
                model, cached_val, target, condition, seeds, config["metric_epsilon"]
            )
            current = seed_macro(val_rows, seeds)
            if not np.isfinite(current).all():
                raise ArithmeticError("nonfinite validation message error")
            improved = current < best_scores
            best_scores = np.minimum(best_scores, current)
            best_epochs[improved] = epoch
            for name, tensor in model.state_dict().items():
                mask = torch.as_tensor(improved, device=tensor.device)
                best_state[name][mask] = tensor.detach()[mask]
            seconds = previous_seconds + time.perf_counter() - start
            peak = (
                torch.cuda.max_memory_allocated(next(model.parameters()).device)
                if next(model.parameters()).is_cuda
                else None
            )
            for index, seed in enumerate(seeds):
                history.append(
                    {
                        "target": target,
                        "condition": condition,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": float(train_loss[index]),
                        "val_message_relerr": float(current[index]),
                        "best_epoch": int(best_epochs[index]),
                        "seconds": seconds,
                        "peak_vram_bytes": peak,
                    }
                )
            _checkpoint(
                checkpoints / f"{target}-{condition}-epoch{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                target,
                condition,
                config,
                batch_size,
                best_state,
                best_scores,
                best_epochs,
                history,
                source,
            )
            print(
                f"[train] target={target} condition={condition} epoch={epoch}/{config['epochs']} "
                f"loss={train_loss.tolist()} val_relerr={current.tolist()} "
                f"seconds={seconds:.1f}",
                flush=True,
            )
        else:
            print(
                f"[train] target={target} condition={condition} epoch={epoch}/{config['epochs']} "
                f"loss={train_loss.tolist()} "
                f"seconds={previous_seconds + time.perf_counter() - start:.1f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    selected = {
        "target": target,
        "condition": condition,
        "seeds": seeds,
        "best_epochs": best_epochs.tolist(),
        "validation_message_relerr": best_scores.tolist(),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in best_state.items()},
        "config": config,
        "config_hash": config_digest(config),
        "source": source,
    }
    with (checkpoints / f"{target}-{condition}-selected.pt").open("xb") as stream:
        torch.save(selected, stream)
    return history, selected


def run(args, output):
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is required; use the independent track requirements.txt")
    config_path = args.config or Path(__file__).with_name(f"config_{args.profile}.json")
    config = read_config(config_path, args.profile)
    device, dtype = torch.device(args.device), getattr(torch, config["dtype"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    resources = runtime_resources(device)
    resources["cpu_math_threadpools"] = threadpool_info()
    storage = psutil.disk_usage(str(output))
    resources.update(storage_free_bytes=storage.free, storage_total_bytes=storage.total)
    source = source_manifest()
    write_json(output / "config.json", config)
    write_json(output / "resources_before.json", resources)
    print(
        f"[start] Experiment 2 profile={args.profile} dtype={dtype} device={device} "
        f"epochs={config['epochs']} hidden={config['hidden']} "
        f"parallel_seeds={config['model_seeds']}",
        flush=True,
    )
    print("[hardware] " + json.dumps(resources), flush=True)
    started = time.perf_counter()
    psutil.cpu_percent(interval=None)
    workers, cpu_trials = choose_workers(config, resources, args.workers)
    print(f"[prepare] all split graphs; workers={workers}", flush=True)
    cases = prepare_cases(config, workers)
    manifest = save_dataset(output, cases)
    splits = {name: [case for case in cases if case.split == name] for name in ("train", *SPLITS)}
    stats = {
        name: {
            "graphs": len(group),
            "inputs": len(group) * config["features"],
            "nodes": [min(case.num_nodes for case in group), max(case.num_nodes for case in group)],
            "edges": [
                min(case.edges.shape[1] for case in group),
                max(case.edges.shape[1] for case in group),
            ],
            "paths": [
                min(case.wedges.shape[1] for case in group),
                max(case.wedges.shape[1] for case in group),
            ],
        }
        for name, group in splits.items()
    }
    print("[data] " + json.dumps(stats), flush=True)
    resumed = restore_checkpoint(args.resume_from, config, device) if args.resume_from else None
    requested_size = str(resumed["physical_graph_batch"]) if resumed else args.batch_size
    batch_size, gpu_trials = choose_batch_size(
        splits["train"], config, device, dtype, requested_size
    )
    print(
        f"[selected] physical_graph_batch={batch_size} accumulation=1 "
        f"realizations_per_graph={config['features']} seeds_parallel={len(config['model_seeds'])}",
        flush=True,
    )
    print("[teacher audit] measuring every graph/realization against span{L,L2,Q}", flush=True)
    audit_rows = operator_audit(cases, device, batch_size)
    if len(audit_rows) != len(cases):
        raise AssertionError("teacher operator audit omitted graphs")
    write_csv(output / "teacher_operator_audit.csv", audit_rows)
    cached = {
        name: cache_batches(group, batch_size, device, dtype) for name, group in splits.items()
    }
    cpu_fit = pack_cases(splits["train"], torch.device("cpu"), torch.float64)
    plan = intervention_manifest(
        [case for case in cases if case.split != "train"], config["master_seed"]
    )
    write_json(output / "intervention_manifest.json", plan)
    metrics, controls, history, jobs = [], [], [], []
    peak_by_job = []
    for target in TARGETS:
        for condition in CONDITIONS:
            seeds = [-1] if condition in ("first", "polynomial", "fixed") else config["model_seeds"]
            model = make_model(condition, seeds, config["hidden"], config["teacher"]["tau"])
            per_seed_params = sum(param.numel() for param in model.parameters()) // len(seeds)
            print(
                f"[job] target={target} condition={condition} "
                f"parameters_per_seed={per_seed_params} "
                f"parallel_models={len(seeds)}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if len(seeds) == 1 and seeds[0] == -1:
                model = model.to(dtype=torch.float64)
                fitted = fit_baseline(
                    model, cpu_fit, cpu_fit.targets[target], config["loss_epsilon"]
                )
                write_json(output / f"{target}-{condition}-fit.json", fitted)
                model = model.to(device=device, dtype=dtype)
                selection = {"fit": "train-only analytic optimum", "seeds": seeds}
            else:
                model = model.to(device=device, dtype=dtype)
                resume_job = (
                    resumed
                    if resumed and (target, condition) == (resumed["target"], resumed["condition"])
                    else None
                )
                measured_history, selected = train_gate(
                    model,
                    cached["train"],
                    cached["validation"],
                    config,
                    target,
                    condition,
                    output,
                    batch_size,
                    source,
                    resume_job,
                )
                history.extend(measured_history)
                selection = {"best_epochs": selected["best_epochs"], "seeds": seeds}
            for split in SPLITS:
                rows, patterns = evaluate(
                    model, cached[split], target, condition, seeds, config["metric_epsilon"]
                )
                metrics.extend(rows)
                print(
                    f"[evaluate] target={target} condition={condition} split={split} "
                    f"macro_relerr={seed_macro(rows, seeds).tolist()}",
                    flush=True,
                )
                if condition == "learned":
                    controls.extend(
                        interventions(
                            model,
                            cached[split],
                            patterns,
                            plan,
                            target,
                            seeds,
                            config["metric_epsilon"],
                        )
                    )
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            peak_by_job.append({"target": target, "condition": condition, "peak_vram_bytes": peak})
            jobs.append(
                {
                    "target": target,
                    "condition": condition,
                    "parameters_per_seed": per_seed_params,
                    "selection": selection,
                }
            )
            del model
    if source != source_manifest():
        raise RuntimeError("source files or Git revision changed during experiment")
    evaluation_count = sum(len(splits[name]) for name in SPLITS)
    expected_metrics = evaluation_count * len(TARGETS) * (3 + 2 * len(config["model_seeds"]))
    expected_controls = evaluation_count * len(TARGETS) * len(config["model_seeds"]) * 5
    if len(metrics) != expected_metrics or len(controls) != expected_controls:
        raise AssertionError("incomplete target/condition/seed/split/intervention coverage")
    expected_keys = {
        (target, condition, seed, case.split, case.graph_id)
        for target in TARGETS
        for condition in CONDITIONS
        for seed in (
            [-1] if condition in ("first", "polynomial", "fixed") else config["model_seeds"]
        )
        for case in cases
        if case.split != "train"
    }
    actual_keys = {
        (row["target"], row["condition"], row["seed"], row["split"], row["graph_id"])
        for row in metrics
    }
    expected_control_keys = {
        (target, seed, case.split, case.graph_id, name)
        for target in TARGETS
        for seed in config["model_seeds"]
        for case in cases
        if case.split != "train"
        for name in config["interventions"]
    }
    actual_control_keys = {
        (row["target"], row["seed"], row["split"], row["graph_id"], row["intervention"])
        for row in controls
    }
    if actual_keys != expected_keys or actual_control_keys != expected_control_keys:
        raise AssertionError("missing or incorrectly labeled evaluation keys")
    contract = {
        "experiment": 2,
        "profile": args.profile,
        "debug": args.profile == "debug",
        "config": config,
        "config_hash": config_digest(config),
        "source": source,
        "source_unchanged_during_run": True,
        "runtime": resources,
        "split_statistics": stats,
        "graph_count_total": len(cases),
        "graph_count_used": len(cases),
        "data_fraction": 1.0,
        "input_count_total": len(cases) * config["features"],
        "sampling_ratio": 1.0,
        "path_sampling": False,
        "physical_graph_batch_selected": batch_size,
        "physical_graph_batch_actual": {
            name: [batch.num_graphs for _, batch in batches] for name, batches in cached.items()
        },
        "input_shapes": {
            name: [list(batch.x.shape) for _, batch in batches] for name, batches in cached.items()
        },
        "feature_channels": 1,
        "feature_realizations_parallel": config["features"],
        "seed_replicas_parallel": len(config["model_seeds"]),
        "optimizer_updates_per_seed_per_gate_job": len(cached["train"]) * config["epochs"],
        "optimizer_calls_in_full_study": len(cached["train"]) * config["epochs"] * 6,
        "independent_seed_updates_in_full_study": (
            len(cached["train"]) * config["epochs"] * 6 * len(config["model_seeds"])
        ),
        "gradient_accumulation": 1,
        "effective_training_batch_per_seed": {
            "graph_fields": batch_size * config["features"],
            "data_parallel_workers": 1,
        },
        "seed_axis_is_independent_models_not_DDP": True,
        "cpu_workers": workers,
        "torch_intraop_threads": 1,
        "data_loader_workers": None,
        "data_loader_reason": "generated once and cached by disjoint-union graph batch",
        "static_gpu_cache": device.type == "cuda",
        "pin_memory": device.type == "cuda",
        "non_blocking_transfer": device.type == "cuda",
        "prefetch": "all static batches cached",
        "precision": config["dtype"],
        "teacher_reference_precision": "float64",
        "tf32": False,
        "gnn_backbone": None,
        "hidden_layers": 1,
        "attention_heads": None,
        "optimizer": "Adam, no weight decay",
        "scalar_baselines": "train-only optimal least squares; seed=-1 is deterministic",
        "selection": "each gate seed uses validation message macro error only",
        "test_updates": 0,
        "C2_labels_in_training_loss": False,
        "teacher_operator_audit_graphs": len(audit_rows),
        "mean_C2": 1,
        "mean_gauge_does_not_imply_unique_weight_recovery": True,
        "jobs": jobs,
        "metric_rows": len(metrics),
        "intervention_rows": len(controls),
        "cpu_calibration": cpu_trials,
        "gpu_batch_calibration": gpu_trials,
        "peak_vram_by_job": peak_by_job,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "seconds_before_report": time.perf_counter() - started,
        "data_manifest_counts": manifest["split_graph_counts"],
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "resume_scope": "one matching gate job resumes; all other jobs run in the new directory",
        "resume_source": resumed["source"] if resumed else None,
        "real_dataset_classification": False,
    }
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "interventions.csv", controls)
    write_csv(output / "training.csv", history)
    write_json(output / "contract.json", contract)
    print("[report] writing actual synthetic message results and figures", flush=True)
    write_report(output, metrics, controls, history, contract)
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "experiment": 2,
            "profile": args.profile,
            "graphs": len(cases),
            "metric_rows": len(metrics),
            "intervention_rows": len(controls),
            "seconds": time.perf_counter() - started,
        },
    )
    print(f"[complete] Experiment 2 profile={args.profile}; results={output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
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
                        "status": "failed",
                        "type": type(error).__name__,
                        "error": str(error),
                        "recovery": "inspect terminal.log; use a fresh output directory",
                    },
                )
                raise


if __name__ == "__main__":
    main()
