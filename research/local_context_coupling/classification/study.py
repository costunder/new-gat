"""Server entry point: complete public-split tuning, final training and frozen audits.

Full is the immutable 252-run/126000-update contract. DEBUG uses explicit fixtures and
separate budgets. All visible, explicitly allocated GPUs receive independent
job groups, while each group packs its independent initialization seeds.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
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

from ...wedge_propagation.study import available_cpus, runtime_resources
from .common import (
    digest,
    file_sha256,
    read_config,
    read_json,
    source_manifest,
    write_csv,
    write_json,
)
from .data import (
    assert_data_unchanged,
    graph_content_digest,
    load_graph,
    prepare_datasets,
    save_graph,
)
from .evaluation import frozen_evaluate, intervention_scopes, intervention_variants
from .model import parse_condition
from .training import (
    _resume_payload,
    choose_packing,
    make_model,
    preserve_resume_calibration,
    train_pack,
)


class _Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile, self.lock = stream, logfile, threading.RLock()

    def write(self, value):
        with self.lock:
            self.stream.write(value)
            self.logfile.write(value)
            self.flush()
        return len(value)

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.logfile.flush()


def select_learning_rates(config, rows):
    expected = set(
        itertools.product(
            config["data"]["datasets"],
            config["conditions"],
            config["training"]["learning_rate_candidates"],
            config["training"]["tuning_seeds"],
        )
    )
    seen, buckets = set(), {}
    for row in rows:
        key = (row["dataset"], row["condition"], row["lr"], row["seed"])
        if key in seen or key not in expected or row.get("phase") != "tuning":
            raise ValueError("duplicate/out-of-contract tuning row")
        if row.get("variant") != parse_condition(row["condition"])[1]:
            raise ValueError("tuning row variant differs from its condition")
        if any("test" in field.lower() for field in row):
            raise ValueError("test metrics must never enter validation selection")
        if (
            not math.isfinite(row["validation_ce"])
            or row["validation_ce"] < 0
            or not 0 <= row["validation_accuracy"] <= 1
            or not 1 <= row["best_epoch"] <= config["training"]["epochs_per_run"]
        ):
            raise ValueError("invalid validation-selected row")
        if (
            row.get("training_epochs") != config["training"]["epochs_per_run"]
            or row.get("optimizer_updates") != config["training"]["epochs_per_run"]
        ):
            raise ValueError("incomplete tuning training/update budget")
        seen.add(key)
        buckets.setdefault(key[:3], []).append(row["validation_ce"])
    if seen != expected:
        raise ValueError(f"incomplete tuning coverage: missing={len(expected - seen)}")
    selected = []
    for dataset, condition in itertools.product(config["data"]["datasets"], config["conditions"]):
        options = [
            (float(np.mean(buckets[(dataset, condition, lr)])), lr)
            for lr in config["training"]["learning_rate_candidates"]
        ]
        value, lr = min(options)
        selected.append(
            {
                "dataset": dataset,
                "condition": condition,
                "weight_mode": parse_condition(condition)[0],
                "variant": parse_condition(condition)[1],
                "selected_lr": lr,
                "mean_tuning_validation_ce": value,
                "selection_scope": "validation_only_independent_tuning_seeds",
            }
        )
    return selected


def build_jobs(config, phase, selections=None):
    if phase not in ("tuning", "final"):
        raise ValueError("job phase must be tuning or final")
    chosen = {}
    if phase == "final":
        if selections is None:
            raise ValueError("final jobs require locked validation selections")
        for row in selections:
            key = row["dataset"], row["condition"]
            if (
                key in chosen
                or row["selected_lr"] not in config["training"]["learning_rate_candidates"]
            ):
                raise ValueError("invalid or duplicate locked learning-rate selection")
            if row.get("variant") != parse_condition(row["condition"])[1]:
                raise ValueError("locked selection variant differs from its condition")
            chosen[key] = row["selected_lr"]
        if set(chosen) != set(itertools.product(config["data"]["datasets"], config["conditions"])):
            raise ValueError("incomplete final selection coverage")
    jobs = []
    for dataset, condition in itertools.product(config["data"]["datasets"], config["conditions"]):
        rates = (
            config["training"]["learning_rate_candidates"]
            if phase == "tuning"
            else [chosen[(dataset, condition)]]
        )
        for lr in rates:
            jobs.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "weight_mode": parse_condition(condition)[0],
                    "variant": parse_condition(condition)[1],
                    "lr": lr,
                    "phase": phase,
                    "seeds": config["training"][f"{phase}_seeds"],
                    "job_key": f"{phase}/{dataset}/{condition}/lr-{lr:g}",
                }
            )
    return jobs


def verify_coverage(config, tuning_rows, final_metric_rows):
    select_learning_rates(config, tuning_rows)
    expected = set(
        itertools.product(
            config["data"]["datasets"],
            config["conditions"],
            config["training"]["final_seeds"],
            ("train", "validation", "test"),
        )
    )
    seen = set()
    for row in final_metric_rows:
        key = row["dataset"], row["condition"], row["seed"], row["split"]
        if key in seen or key not in expected:
            raise ValueError("duplicate/out-of-contract final metric row")
        if row.get("variant") != parse_condition(row["condition"])[1]:
            raise ValueError("final metric variant differs from its condition")
        if not math.isfinite(row["ce"]) or row["ce"] < 0 or not 0 <= row["accuracy"] <= 1:
            raise ValueError("invalid final metric")
        shape = config["data"]["expected_shapes"][row["dataset"]]
        if row.get("num_nodes") != shape["nodes"]:
            raise ValueError("final full-graph node coverage changed")
        if row.get("num_labeled_nodes") != shape[row["split"]]:
            raise ValueError("final labeled-node coverage changed")
        seen.add(key)
    if seen != expected:
        raise ValueError(f"incomplete final metric coverage: missing={len(expected - seen)}")
    return {
        "tuning_runs": len(tuning_rows),
        "final_runs": len(expected) // 3,
        "primary_metric_rows": len(seen),
        "total_runs": config["training"]["total_runs"],
        "contract_optimizer_updates": config["training"]["total_updates"],
        "all_datasets_conditions_seeds_splits": True,
    }


def _graph_digest(graph):
    return graph_content_digest(graph)


def _validate_final_training(config, selected, jobs):
    expected = {
        (job["dataset"], job["condition"], seed): job["lr"] for job in jobs for seed in job["seeds"]
    }
    seen = set()
    for row in selected:
        key = row["dataset"], row["condition"], row["seed"]
        if (
            key in seen
            or key not in expected
            or row["lr"] != expected[key]
            or row["phase"] != "final"
            or row["training_epochs"] != config["training"]["epochs_per_run"]
            or row["optimizer_updates"] != config["training"]["epochs_per_run"]
            or row.get("variant") != parse_condition(row["condition"])[1]
        ):
            raise ValueError("final checkpoint/train coverage mismatch")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("all final checkpoints must be selected before unlocking test")


def verify_frozen_coverage(config, evaluated):
    expected = {"intervention_rows": set(), "branch_rows": set()}
    for dataset, condition, seed in itertools.product(
        config["data"]["datasets"],
        config["conditions"],
        config["training"]["final_seeds"],
    ):
        prefix = dataset, condition, seed
        expected["branch_rows"].update((*prefix, layer, "original", "none") for layer in range(2))
        for intervention, target in itertools.product(
            intervention_variants(parse_condition(condition)[1]),
            intervention_scopes(condition),
        ):
            expected["intervention_rows"].update(
                (*prefix, split, intervention, target) for split in ("train", "validation", "test")
            )
            expected["branch_rows"].update(
                (*prefix, layer, intervention, target) for layer in range(2)
            )
    coverage = {}
    fields = {
        "intervention_rows": ("split", "intervention", "target"),
        "branch_rows": ("layer", "intervention", "target"),
    }
    for table, suffix in fields.items():
        seen = set()
        for row in evaluated[table]:
            key = tuple(row[field] for field in ("dataset", "condition", "seed", *suffix))
            if key in seen or key not in expected[table]:
                raise ValueError(f"duplicate/out-of-contract frozen {table} row")
            if row.get("variant") != parse_condition(row["condition"])[1]:
                raise ValueError(f"frozen {table} variant differs from its condition")
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError(f"nonfinite frozen {table} metric")
            seen.add(key)
        if seen != expected[table]:
            raise ValueError(f"incomplete frozen {table}: missing={len(expected[table] - seen)}")
        coverage[table] = len(seen)
    return coverage


def _worker(plan_path):
    plan = read_json(plan_path)
    output = Path(plan["output_dir"])
    config = read_config(output / "config.json", plan["profile"])
    current_source = source_manifest()
    if current_source["code_digest"] != plan["source"]["code_digest"]:
        raise ValueError("worker source changed after dispatch")
    device = torch.device(plan["device"])
    if config["profile"] == "full" and (
        device.type != "cuda"
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
        or sys.platform != "linux"
    ):
        raise ValueError("full server worker requires Linux allocated CUDA; no CPU fallback")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("worker requested CUDA unavailable; no CPU fallback")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resources = runtime_resources(device)
    threads = config["runtime"]["cpu_threads"]
    if threads == "auto":
        threads = max(1, min(8, available_cpus(resources) // plan["num_workers"]))
    torch.set_num_threads(int(threads))
    worker_dir = output / "workers" / f"{plan['stage']}-{plan['worker_index']}"
    worker_dir.mkdir(parents=True, exist_ok=False)
    write_json(worker_dir / "resources.json", {**resources, "torch_cpu_threads": threads})
    graphs, calibrations = {}, {}
    resource_rows, selection_rows = [], []
    result = {
        "metric_rows": [],
        "intervention_rows": [],
        "branch_rows": [],
        "provenance": [],
    }
    if plan["stage"] == "evaluation":
        unlock = read_json(output / "test_evaluation_unlocked.json")
        if digest(unlock) != plan["test_unlock_digest"]:
            raise ValueError("test checkpoint lock changed")
    for job_index, job in enumerate(plan["jobs"]):
        name, condition = job["dataset"], job["condition"]
        if name not in graphs:
            record = plan["graphs"][name]
            host = load_graph(record["path"], record["sha256"])
            if _graph_digest(host) != record["content_digest"]:
                raise ValueError("worker graph tensors changed")
            graphs[name] = host.to(device, torch.float32)
        graph = graphs[name]
        calibration_path = output / "calibration" / f"{job['phase']}-{name}-{condition}.json"
        key = job["phase"], name, condition
        if plan["stage"] == "evaluation":
            calibration = read_json(calibration_path)["selection"]
        elif key in calibrations:
            calibration = calibrations[key]
        else:
            original = (
                Path(plan["resume_from"]) / "calibration" / calibration_path.name
                if plan["resume_from"]
                else None
            )
            if original is not None and original.is_file():
                calibration = preserve_resume_calibration(
                    original,
                    calibration_path,
                    config,
                    job["phase"],
                    name,
                    condition,
                )
            else:
                calibration = choose_packing(
                    graph, condition, job["seeds"], job["lr"], config, job["phase"]
                )
                write_json(
                    calibration_path,
                    {
                        "dataset": name,
                        "condition": condition,
                        "weight_mode": parse_condition(condition)[0],
                        "variant": parse_condition(condition)[1],
                        "phase": job["phase"],
                        "config_digest": digest(config),
                        "selection": calibration,
                    },
                )
            calibrations[key] = calibration
            resource_rows += calibration["trials"]
        packed = calibration["packed_runs"]
        print(
            f"[job] worker={plan['worker_index']} {plan['stage']} "
            f"{job_index + 1}/{len(plan['jobs'])} {job['job_key']} "
            f"all_seeds={job['seeds']} measured_packed={packed}",
            flush=True,
        )
        for start in range(0, len(job["seeds"]), packed):
            seeds = job["seeds"][start : start + packed]
            pack_dir = output / "jobs" / job["job_key"] / f"pack-{start:02d}"
            if plan["stage"] != "evaluation":
                resume_dir = (
                    Path(plan["resume_from"]) / "jobs" / job["job_key"] / f"pack-{start:02d}"
                    if plan["resume_from"]
                    else None
                )
                trained = train_pack(
                    graph,
                    condition,
                    seeds,
                    job["lr"],
                    config,
                    pack_dir,
                    plan["source"],
                    plan["graphs"][name]["content_digest"],
                    job["phase"],
                    resume_dir=resume_dir,
                    calibration=calibration,
                    resources=resources,
                )
                selection_rows += trained["rows"]
                resource_rows.append(trained["resource"])
                del trained
            else:
                metadata = {
                    "experiment": config["experiment"],
                    "weight_mode": parse_condition(condition)[0],
                    "dataset": name,
                    "condition": condition,
                    "variant": parse_condition(condition)[1],
                    "seeds": seeds,
                    "lr": job["lr"],
                    "phase": "final",
                    "config_digest": digest(config),
                    "code_digest": plan["source"]["code_digest"],
                    "graph_digest": plan["graphs"][name]["content_digest"],
                    "packed_runs": len(seeds),
                    "path_chunk": calibration["path_chunk"],
                }
                payload, selected_path = _resume_payload(pack_dir, metadata)
                if selected_path is None or selected_path.name != "selected.pt":
                    raise ValueError("evaluation requires a finalized selected checkpoint")
                model = make_model(graph, condition, seeds, config, calibration["path_chunk"])
                model.load_state_dict(payload["best_state"])
                evaluated = frozen_evaluate(model, graph, seeds, config)
                for field in ("metric_rows", "intervention_rows", "branch_rows"):
                    result[field] += evaluated[field]
                result["provenance"].append(evaluated["provenance"])
                print(
                    f"[evaluated] {name}/{condition} seeds={seeds} "
                    f"test_rows={sum(r['split'] == 'test' for r in evaluated['metric_rows'])} "
                    f"interventions={len(evaluated['intervention_rows'])} frozen=True",
                    flush=True,
                )
                del model, payload, evaluated
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(
        worker_dir / "results.json",
        {
            **result,
            "selection_rows": selection_rows,
            "resource_rows": resource_rows,
            "source_digest": current_source["code_digest"],
        },
    )
    write_json(
        worker_dir / "completion.json",
        {
            "completed": True,
            "jobs": len(plan["jobs"]),
            "stage": plan["stage"],
            "models_selected": len(selection_rows),
        },
    )


def _dispatch(
    output, config, source, graphs, jobs, devices, stage, resume_from=None, test_unlock_digest=None
):
    # Keep all LRs for a dataset/condition on one worker: one calibration and cache.
    groups = {}
    for job in jobs:
        groups.setdefault((job["dataset"], job["condition"]), []).append(job)
    assignments = [[] for _ in devices]
    costs = [0] * len(devices)
    for (name, _condition), group in sorted(
        groups.items(), key=lambda item: graphs[item[0][0]]["num_local_edges"], reverse=True
    ):
        index = min(range(len(devices)), key=lambda i: costs[i])
        assignments[index].extend(group)
        cost = graphs[name]["num_local_edges"] + graphs[name]["num_cross_edges"] + graphs[name]["num_local_copies"]
        costs[index] += max(1, cost) * len(group)
    plans = []
    for index, (device, assigned) in enumerate(zip(devices, assignments, strict=True)):
        if not assigned:
            continue
        path = output / "plans" / f"{stage}-{index}.json"
        write_json(
            path,
            {
                "output_dir": str(output),
                "profile": config["profile"],
                "stage": stage,
                "worker_index": index,
                "num_workers": len(devices),
                "device": device,
                "source": source,
                "graphs": graphs,
                "jobs": assigned,
                "resume_from": str(resume_from) if resume_from else None,
                "test_unlock_digest": test_unlock_digest,
            },
        )
        plans.append(path)
    if len(plans) == 1:
        _worker(plans[0])
    else:
        processes = []
        threads = []
        for path in plans:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "research.local_context_coupling.classification.study",
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

            thread = threading.Thread(target=consume, args=(process.stdout,), daemon=True)
            thread.start()
            threads.append(thread)
            processes.append(process)
        codes = [process.wait() for process in processes]
        for thread in threads:
            thread.join()
        if any(code != 0 for code in codes):
            raise RuntimeError(
                f"worker failure codes={codes}; sessions/results preserved; "
                "inspect terminal.log and resume snapshots"
            )
    records = [
        read_json(
            output / "workers" / f"{stage}-{read_json(path)['worker_index']}" / "results.json"
        )
        for path in plans
    ]
    return {
        field: [row for record in records for row in record[field]]
        for field in (
            "selection_rows",
            "resource_rows",
            "metric_rows",
            "intervention_rows",
            "branch_rows",
            "provenance",
        )
    }


def run(args):
    config = read_config(
        args.config or Path(__file__).with_name(f"config_{args.profile}.json"), args.profile
    )
    if args.workers != "auto":
        config["runtime"]["cpu_workers"] = args.workers
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("output directory exists; preserved; choose a new result directory")
    requested = torch.device(args.device)
    if args.profile == "full" and (
        requested.type != "cuda"
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
        or sys.platform != "linux"
    ):
        raise ValueError(
            "full training requires Linux server CUDA and explicit allocated CUDA_VISIBLE_DEVICES"
        )
    source = source_manifest()
    resume = Path(args.resume_from).resolve() if args.resume_from else None
    if resume is not None:
        if read_json(resume / "config.json") != config:
            raise ValueError("resume config differs from complete run contract")
        if read_json(resume / "source.json")["code_digest"] != source["code_digest"]:
            raise ValueError("resume implementation source changed")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no CPU fallback")
    devices = (
        [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if args.device == "cuda"
        else [str(requested)]
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("plans", "graphs", "calibration", "workers", "jobs"):
        (output / name).mkdir()
    with (output / "terminal.log").open("x", encoding="utf-8") as logfile:
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, logfile)),
            contextlib.redirect_stderr(_Tee(sys.stderr, logfile)),
        ):
            try:
                write_json(output / "config.json", config)
                write_json(output / "source.json", source)
                hardware = runtime_resources(requested)
                hardware["gpu_count_used"] = len(devices) if requested.type == "cuda" else 0
                hardware["worker_devices"] = devices
                hardware["worker_gpus"] = []
                for target in devices:
                    if torch.device(target).type == "cuda":
                        properties = torch.cuda.get_device_properties(target)
                        free, total = torch.cuda.mem_get_info(target)
                        hardware["worker_gpus"].append({"device": target, "name": properties.name, "uuid": str(getattr(properties,"uuid","unavailable")), "free_memory_bytes": free, "runtime_total_memory_bytes": total, "total_memory_bytes": properties.total_memory})
                write_json(output / "hardware.json", hardware)
                print(
                    f"[start] local-context classification profile={args.profile} devices={devices} "
                    f"datasets={config['data']['datasets']} conditions={len(config['conditions'])} "
                    f"layers=2 hidden={config['backbone']['hidden_dim']} "
                    f"epochs={config['training']['epochs_per_run']} "
                    f"runs={config['training']['total_runs']} "
                    f"updates={config['training']['total_updates']} "
                    f"actual_data={config['data_source'] == 'planetoid_public'} sampling=1.0",
                    flush=True,
                )
                print(f"[hardware] {hardware}", flush=True)
                start = time.perf_counter()
                data, manifest = prepare_datasets(
                    config, args.data_root, output, args.source_dir, args.workers, not args.offline
                )
                graph_records, parameter_counts = {}, {}
                for name, graph in data.items():
                    path = output / "graphs" / f"{name}.npz"
                    saved = save_graph(graph, path)
                    graph_records[name] = {
                        "path": str(path),
                        "sha256": saved["sha256"],
                        "content_digest": _graph_digest(graph),
                        "num_local_edges": graph.topology.num_local_edges,
                        "num_pairs": graph.topology.num_pairs,
                        "num_cross_edges": graph.geometry.num_cross_edges,
                        "num_local_copies": graph.geometry.num_copies,
                    }
                    parameter_counts[name] = {
                        condition: make_model(
                            graph, condition, [config["training"]["tuning_seeds"][0]], config, 1
                        ).parameters_per_seed
                        for condition in config["conditions"]
                    }
                write_json(output / "graph_cache.json", graph_records)
                del data
                tuning = _dispatch(
                    output,
                    config,
                    source,
                    graph_records,
                    build_jobs(config, "tuning"),
                    devices,
                    "tuning",
                    resume,
                )
                selections = select_learning_rates(config, tuning["selection_rows"])
                write_json(output / "learning_rate_selection.json", selections)
                write_csv(output / "tuning_validation.csv", tuning["selection_rows"])
                print(
                    "[selection] validation learning rates locked; "
                    "starting independent final seeds",
                    flush=True,
                )
                final_jobs = build_jobs(config, "final", selections)
                final = _dispatch(
                    output, config, source, graph_records, final_jobs, devices, "final", resume
                )
                _validate_final_training(config, final["selection_rows"], final_jobs)
                write_csv(output / "final_validation_selection.csv", final["selection_rows"])
                unlock = {
                    "all_final_checkpoints_locked": True,
                    "selection_digest": digest(final["selection_rows"]),
                    "learning_rate_selection_digest": digest(selections),
                    "final_runs": len(final["selection_rows"]),
                }
                write_json(output / "test_evaluation_unlocked.json", unlock)
                print(
                    "[evaluation] all final checkpoints locked; "
                    "test and frozen interventions begin",
                    flush=True,
                )
                evaluated = _dispatch(
                    output,
                    config,
                    source,
                    graph_records,
                    final_jobs,
                    devices,
                    "evaluation",
                    test_unlock_digest=digest(unlock),
                )
                coverage = verify_coverage(
                    config, tuning["selection_rows"], evaluated["metric_rows"]
                )
                coverage.update(verify_frozen_coverage(config, evaluated))
                resources = tuning["resource_rows"] + final["resource_rows"]
                for filename, rows in (
                    ("metrics.csv", evaluated["metric_rows"]),
                    ("interventions.csv", evaluated["intervention_rows"]),
                    ("branch_diagnostics.csv", evaluated["branch_rows"]),
                    ("resources.csv", resources),
                ):
                    write_csv(output / filename, rows)
                write_json(output / "frozen_evaluation_provenance.json", evaluated["provenance"])
                if source_manifest()["code_digest"] != source["code_digest"]:
                    raise RuntimeError("implementation changed during study")
                for record in graph_records.values():
                    if file_sha256(record["path"]) != record["sha256"]:
                        raise RuntimeError("input graph cache changed during study")
                assert_data_unchanged(manifest)
                write_json(output / "coverage.json", coverage)
                contract = {
                    "profile": args.profile,
                    "parameter_counts": parameter_counts,
                    "config_digest": digest(config),
                    "source": source,
                    "hardware": hardware,
                    "data_manifest_digest": digest(manifest),
                    "coverage": coverage,
                    "resume_from": str(resume) if resume else None,
                    "new_optimizer_updates": sum(
                        row["new_optimizer_updates"]
                        for row in tuning["selection_rows"] + final["selection_rows"]
                    ),
                    "actual_data": config["data_source"] == "planetoid_public",
                }
                write_json(output / "contract.json", contract)
                print(
                    "[report] writing retrained cross condition comparisons and coupling diagnostics",
                    flush=True,
                )
                from .report import write_report

                write_report(output, config, evaluated, selections, resources, contract)
                write_json(
                    output / "completion.json",
                    {
                        "completed": True,
                        "profile": args.profile,
                        "scope": "DEBUG_pipeline_only"
                        if args.profile == "debug"
                        else "full_public_split_local_context_coupling_classification",
                        "seconds": time.perf_counter() - start,
                        "coverage": coverage,
                        "new_optimizer_updates": contract["new_optimizer_updates"],
                        "actual_data": contract["actual_data"],
                        "frozen_model_updates": 0,
                        "code_and_graphs_preserved": True,
                    },
                )
                print(
                    f"[complete] profile={args.profile} results={output} "
                    f"actual_data={contract['actual_data']} "
                    f"new_updates={contract['new_optimizer_updates']} coverage={coverage}",
                    flush=True,
                )
            except Exception as error:
                traceback.print_exc()
                failure = output / "failure.json"
                if not failure.exists():
                    write_json(
                        failure,
                        {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                            "results_preserved": True,
                            "recovery": "fix reported cause and resume into a NEW directory",
                        },
                    )
                raise


def _workers(value):
    if value == "auto":
        return value
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("workers must be auto or a positive integer") from error
    if number < 1 or str(number) != str(value):
        raise argparse.ArgumentTypeError("workers must be auto or a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "debug"), default="full")
    parser.add_argument(
        "--device", default="cuda", help="cuda uses all explicitly allocated visible GPUs"
    )
    parser.add_argument("--data-root", default="data/wedge-citation")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--source-dir", help="completed original local-energy audit; full requires all 201 graphs"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=_workers, default="auto")
    parser.add_argument(
        "--offline", action="store_true", help="require existing verified official raw cache"
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="preserve original results; write resumed run to new output",
    )
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan is not None:
        _worker(args.worker_plan)
    else:
        if not args.output_dir or not args.source_dir:
            parser.error(
                "--output-dir and --source-dir are required; choose a NEW result directory"
            )
        run(args)


if __name__ == "__main__":
    main()
