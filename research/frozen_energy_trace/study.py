"""Exact post-hoc local energy diagnostics from one existing classification seed."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from research.wedge_propagation.classification.common import file_sha256, write_csv, write_json
from research.wedge_propagation.classification.data import load_graph
from research.wedge_propagation.classification.evaluation import model_state_hash
from research.wedge_propagation.study import available_cpus, runtime_resources

from .energy import build_topology, measure
from .models import assert_source_unchanged, discover, load_frozen, operator_replays, trace_stages
from .report import flat_rows, operation_fields, summarize, transition, write_report

SEED = 11
ROOT = Path(__file__).resolve().parents[2]


def sources():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / "research/wedge_propagation" / name
        for name in ("study.py", "data.py", "operators.py", "report.py", "algebra.py")
    ]
    paths += list((ROOT / "research/wedge_propagation/classification").glob("*.py"))
    paths += list((ROOT / "research/wedge_propagation/classification").glob("*.json"))
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(set(paths))}


def save_fields(path, fields):
    arrays = {key: value.detach().cpu().numpy() for key, value in fields.items()}
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return file_sha256(path)


def cpu_fields(fields):
    # One bulk transfer boundary after GPU reductions, never inside node/feature math.
    return {key: value.detach().cpu() for key, value in fields.items()}


def verify_saved_metrics(graph, stages, descriptor):
    """Audit the selected state against saved metrics; never select another state."""
    logits = next(stage.values[0] for stage in stages if stage.name == "logits")
    result = []
    for saved in descriptor.original_metrics:
        split = saved["split"]
        mask = getattr(graph, "val_mask" if split == "validation" else f"{split}_mask")
        ce = float(torch.nn.functional.cross_entropy(logits[mask], graph.y[mask]))
        accuracy = float((logits[mask].argmax(-1) == graph.y[mask]).double().mean())
        evidence = {
            "split": split,
            "saved_ce": saved["ce"],
            "replayed_ce": ce,
            "saved_accuracy": saved["accuracy"],
            "replayed_accuracy": accuracy,
            "ce_atol": 2e-6,
            "ce_rtol": 2e-6,
            "accuracy_atol": 1e-7,
            "within_tolerance": abs(ce - saved["ce"]) <= 2e-6 + 2e-6 * abs(saved["ce"])
            and abs(accuracy - saved["accuracy"]) <= 1e-7,
        }
        if not evidence["within_tolerance"]:
            raise ValueError(
                f"selected source metric replay differs: {descriptor.dataset}/"
                f"{descriptor.condition}/{split}: {evidence}"
            )
        result.append(evidence)
    return result


def calibrate(top, h, device, candidates, repeats):
    rows = []
    candidates = sorted({min(int(c), h.shape[-1]) for c in candidates})
    for chunk in candidates:
        row = {
            "feature_chunk": chunk,
            "full_features": h.shape[-1],
            "repeats": repeats,
            "all_nodes_and_edges": True,
            "packed_states": h.shape[0] if h.ndim == 3 else 1,
        }
        try:
            torch.cuda.empty_cache()
            measure(top, h, feature_chunk=chunk)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                measured = measure(top, h, feature_chunk=chunk)
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - start)
                del measured
            free, total = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.max_memory_reserved(device)
            row.update(
                seconds=samples,
                median_seconds=float(np.median(samples)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=reserved,
                free_bytes=free,
                total_bytes=total,
                safe=reserved <= 0.9 * total and free >= 0.1 * total,
            )
            batch = h.shape[0] if h.ndim == 3 else 1
            row["stage_nodes_per_second"] = batch * top.n / row["median_seconds"]
            row["stage_pairs_per_second"] = batch * top.num_pairs / row["median_seconds"]
        except torch.cuda.OutOfMemoryError as error:
            row.update(safe=False, error=str(error), failure="CUDA OOM")
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        print(
            f"[calibration] feature_chunk={chunk} safe={row['safe']} "
            f"seconds={row.get('median_seconds')}",
            flush=True,
        )
    safe = [row for row in rows if row["safe"]]
    if not safe:
        raise RuntimeError(
            "no exact feature chunk fits with 10% VRAM headroom; no graph/model reduction"
        )
    chosen = min(safe, key=lambda row: row["median_seconds"])["feature_chunk"]
    return chosen, rows


def auto_source(results):
    found = []
    for completion in sorted(results.glob("*/completion.json")):
        folder = completion.parent
        config = folder / "config.json"
        if not config.is_file():
            continue
        record = json.loads(config.read_text(encoding="utf-8"))
        done = json.loads(completion.read_text(encoding="utf-8"))
        if (
            record.get("schema") == "wedge-classification-design-v1"
            and done.get("completed") is True
            and done.get("profile") == "full"
        ):
            found.append(folder)
    if len(found) != 1:
        raise ValueError(
            f"expected one completed full classification source; found {found}. "
            "Specify --classification-run explicitly; no performance-based choice."
        )
    return found[0]


def execute(args):
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback or training")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("use one explicitly allocated visible GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.seed != SEED:
        raise ValueError("this approved single-seed experiment is fixed to seed 11")
    if args.calibration_repeats < 2 or min(args.feature_chunks) < 1:
        raise ValueError("measure at least two repeats and positive exact feature chunks")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = (
        args.classification_run.resolve()
        if args.classification_run
        else auto_source(args.results_root.resolve())
    )
    config, descriptors = discover(root, seed=SEED, allow_debug=args.profile == "debug")
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(root):
        raise FileExistsError("use a fresh output outside existing classification results")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source_hashes = sources()
    write_json(output / "source.json", source_hashes)
    with zipfile.ZipFile(output / "execution_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes:
            archive.write(ROOT / name, name)
    hardware = runtime_resources(device)
    write_json(output / "hardware.json", hardware)
    cpu_count = available_cpus(hardware)
    workers = cpu_count if args.workers == "auto" else int(args.workers)
    if not 1 <= workers <= cpu_count:
        raise ValueError("CPU workers must fit the measured allocated CPU count")
    datasets = sorted({d.dataset for d in descriptors})
    by_dataset = {name: [d for d in descriptors if d.dataset == name] for name in datasets}
    write_json(
        output / "contract.json",
        {
            "seed": SEED,
            "seed_count": 1,
            "seed_choice": "first declared existing final seed; no score selection",
            "profile": args.profile,
            "classification_source": str(root),
            "models": ["mlp", "standard_gcn", "polynomial_2"],
            "datasets": datasets,
            "sampling_ratio": 1.0,
            "local": "induced closed 1-hop neighborhood",
            "C": "identity",
            "new_training_runs": 0,
            "new_optimizer_updates": 0,
            "LR_reselection": False,
            "checkpoint_reselection": False,
            "dropout": "disabled by eval",
            "model_precision": "source FP32",
            "energy_precision": "FP64",
            "tf32": False,
            "epsilon": 1e-12,
            "unknown_label": "exclude from label separation only",
            "J_definition": "audited common edges/nodes: J_node=2*J_shared+J_distinct",
            "degree_normalization": "local Lsym and augmented Lsym; amplitude quotients separate",
            "operator_replay": "fixed projected Z; common original GCN P for I/P/P2 on all models",
            "final_layer": "logits without ReLU; matches existing model",
            "backbone": config["backbone"],
            "workers": workers,
            "original_training_epochs": config["training"]["epochs_per_run"],
            "diagnostic_training_epochs": 0,
            "physical_seed_batch": 1,
            "physical_stage_batch": "same-width stages of three models packed; sizes calibrated",
            "gradient_accumulation_steps": 0,
            "effective_training_batch": None,
            "dataloader": "resident CUDA graph; no epoch loader; CPU topology cached once",
            "graph_batching": "all nodes/local edges on GPU; same-width model/stage axis packed",
            "statistics": "one-seed descriptive case study; no std/CI/significance",
            "checkpoints": [d.record() for d in descriptors],
        },
    )
    print(
        f"[start] seed=11 only | checkpoints={len(descriptors)} | training=0 "
        f"updates=0 | CUDA={hardware['gpu']} | CPU workers={workers}",
        flush=True,
    )
    try:

        def prepare(name):
            record = by_dataset[name][0]
            graph = load_graph(record.graph_path, record.graph_sha256)
            top = build_topology(
                graph.num_nodes, graph.edges, workers=max(1, workers // len(datasets))
            )
            return graph, top

        with ThreadPoolExecutor(max_workers=min(workers, len(datasets))) as pool:
            cpu_graphs = dict(zip(datasets, pool.map(prepare, datasets), strict=True))
        stage_rows, transition_rows, calibration_rows, coverage = [], [], [], []
        for name in datasets:
            cpu_graph, cpu_top = cpu_graphs[name]
            topology_start = time.perf_counter()
            graph, top = cpu_graph.to(device), cpu_top.to(device)
            torch.cuda.synchronize(device)
            transfer_seconds = time.perf_counter() - topology_start
            pairs = cpu_top.pair_centers
            if not isinstance(pairs, torch.Tensor):
                pairs = torch.as_tensor(pairs)
            print(
                f"[dataset] {name} nodes={graph.num_nodes} physical_edges={graph.edges.shape[1]} "
                f"features={graph.num_features} local_edges={top.num_local_edges} "
                f"pairs={top.num_pairs} "
                f"used=100% sampling=1.0 debug={args.profile == 'debug'}",
                flush=True,
            )
            chunk, measured_calibration = calibrate(
                top, graph.x, device, args.feature_chunks, args.calibration_repeats
            )
            calibration_rows += [{"dataset": name, **row} for row in measured_calibration]
            write_json(
                output / f"{name}-calibration.json",
                {"chosen_chunk": chunk, "rows": measured_calibration},
            )
            geometry = {"pairs": torch.as_tensor(pairs), "physical_edges": cpu_graph.edges}
            save_fields(output / f"{name}-geometry.npz", geometry)
            reference = cpu_fields(measure(top, graph.x, feature_chunk=chunk))
            reference_hash = save_fields(output / f"{name}-input.npz", reference)
            models, traces, original_hashes, original_stage_counts, metric_replays = (
                {},
                {},
                {},
                {},
                {},
            )
            for descriptor in by_dataset[name]:
                model = load_frozen(graph, descriptor, config, seed=SEED)
                print(
                    f"[model] {name}/{descriptor.condition} layers={config['backbone']['layers']} "
                    f"hidden={config['backbone']['hidden_dim']} input={list(graph.x.shape)} "
                    f"parameters={sum(p.numel() for p in model.parameters())} trainable=0 "
                    f"checkpoint_epoch={descriptor.selected_epoch} new_epochs=0",
                    flush=True,
                )
                original_hashes[descriptor.condition] = model_state_hash(model)
                models[descriptor.condition] = model
                stages = trace_stages(model, graph)
                metric_replays[descriptor.condition] = verify_saved_metrics(
                    graph, stages, descriptor
                )
                original_stage_counts[descriptor.condition] = len(stages)
                traces[descriptor.condition] = stages + operator_replays(model, graph, stages)
            # Width groups batch different models/stages together; graph input is measured once.
            groups = defaultdict(list)
            fields = {condition: {"input": reference} for condition in traces}
            for condition, trace in traces.items():
                for stage in trace:
                    if stage.name != "input":
                        groups[stage.values.shape[-1]].append((condition, stage))
            for width, group in groups.items():
                h = torch.cat([stage.values for _, stage in group], dim=0)
                packed_chunk, packed_calibration = calibrate(
                    top, h, device, [8, *args.feature_chunks], args.calibration_repeats
                )
                calibration_rows += [
                    {"dataset": name, "group_width": width, **row} for row in packed_calibration
                ]
                write_json(
                    output / f"{name}-packed-width-{width}-calibration.json",
                    {"chosen_chunk": packed_chunk, "rows": packed_calibration},
                )
                measured = cpu_fields(measure(top, h, feature_chunk=packed_chunk))
                for index, (condition, stage) in enumerate(group):
                    fields[condition][stage.name] = {
                        key: value[index] for key, value in measured.items()
                    }
                del h, measured
            for descriptor in by_dataset[name]:
                condition = descriptor.condition
                records = fields[condition]
                files = {"input": reference_hash}
                ratio_files = {}

                def append_transition(
                    layer,
                    kind,
                    a,
                    b,
                    *,
                    records=records,
                    name=name,
                    condition=condition,
                    ratio_files=ratio_files,
                ):
                    transition_rows.append(
                        {
                            "dataset": name,
                            "model": condition,
                            "seed": SEED,
                            "layer": layer,
                            **transition(records[a], records[b], kind),
                        }
                    )
                    filename = f"{name}-{condition}-layer_{layer}-{kind}-ratios.npz"
                    ratio_files[filename] = save_fields(
                        output / filename, operation_fields(records[a], records[b])
                    )

                for stage in traces[condition]:
                    field = records[stage.name]
                    identity = {
                        "dataset": name,
                        "model": condition,
                        "seed": SEED,
                        "stage": stage.name,
                        "layer": stage.layer,
                    }
                    stage_rows.append(
                        {**identity, **summarize(field, reference, pairs, cpu_graph.y)}
                    )
                    if stage.name != "input":
                        files[stage.name] = save_fields(
                            output / f"{name}-{condition}-{stage.name}.npz", field
                        )
                for layer in (0, 1):
                    before = "input" if layer == 0 else "layer_0_activated"
                    project, aggregate = f"layer_{layer}_projected", f"layer_{layer}_aggregated"
                    for transition_name, a, b in (
                        ("projection", before, project),
                        ("aggregation", project, aggregate),
                    ):
                        append_transition(layer, transition_name, a, b)
                    if layer == 0:
                        append_transition(layer, "relu", aggregate, "layer_0_activated")
                    replay_p, replay_p2 = f"layer_{layer}_replay_P", f"layer_{layer}_replay_P2"
                    for replay_name, a, b in (
                        ("replay_I_to_P", project, replay_p),
                        ("replay_P_to_P2", replay_p, replay_p2),
                        ("replay_I_to_P2", project, replay_p2),
                    ):
                        append_transition(layer, replay_name, a, b)
                if model_state_hash(models[condition]) != original_hashes[condition]:
                    raise RuntimeError("frozen model changed during diagnostics")
                write_json(
                    output / f"{name}-{condition}-provenance.json",
                    {
                        "source": descriptor.record(),
                        "model_sha256": original_hashes[condition],
                        "field_files_sha256": files,
                        "model_updates": 0,
                        "actual_nodes": graph.num_nodes,
                        "operation_files_sha256": ratio_files,
                        "actual_physical_edges": graph.edges.shape[1],
                        "stage_count": original_stage_counts[condition],
                        "replay_stage_count": len(traces[condition])
                        - original_stage_counts[condition],
                        "input_feature_chunk": chunk,
                        "logits_replay": models[condition].trace_replay_evidence,
                        "operator_replay": models[condition].operator_replay_evidence,
                        "saved_metric_replay": metric_replays[condition],
                        "parameter_count": sum(p.numel() for p in models[condition].parameters()),
                        "trainable_parameter_count": 0,
                        "attention_heads": None,
                        "input_shape": list(graph.x.shape),
                        "host_to_device_seconds": transfer_seconds,
                    },
                )
                coverage.append((name, condition, SEED))
                print(
                    f"[complete] {name}/{condition}/seed11 "
                    f"stages={len(traces[condition])} updates=0",
                    flush=True,
                )
            del models, traces, fields, records, graph, top, reference
            del model, stage, groups, group, field
            gc.collect()
            torch.cuda.empty_cache()
        assert_source_unchanged(descriptors)
        if sources() != source_hashes:
            raise RuntimeError("diagnostic implementation changed during execution")
        expected = {
            (name, condition, SEED)
            for name in datasets
            for condition in ("mlp", "standard_gcn", "polynomial_2")
        }
        if set(coverage) != expected or len(coverage) != len(expected):
            raise RuntimeError("incomplete or duplicate diagnostic coverage")
        expected_stages = {
            (name, condition, SEED, stage)
            for name, condition, _ in expected
            for stage in (
                "input",
                "layer_0_projected",
                "layer_0_aggregated",
                "layer_0_activated",
                "layer_1_projected",
                "layer_1_aggregated",
                "logits",
                "layer_0_replay_P",
                "layer_0_replay_P2",
                "layer_1_replay_P",
                "layer_1_replay_P2",
            )
        }
        actual_stages = [
            (row["dataset"], row["model"], row["seed"], row["stage"]) for row in stage_rows
        ]
        expected_transitions = {
            (name, condition, SEED, layer, kind)
            for name, condition, _ in expected
            for layer in (0, 1)
            for kind in (
                "projection",
                "aggregation",
                "replay_I_to_P",
                "replay_P_to_P2",
                "replay_I_to_P2",
            )
        }
        expected_transitions |= {
            (name, condition, SEED, 0, "relu") for name, condition, _ in expected
        }
        actual_transitions = [
            (row["dataset"], row["model"], row["seed"], row["layer"], row["transition"])
            for row in transition_rows
        ]
        if (
            set(actual_stages) != expected_stages
            or len(actual_stages) != len(expected_stages)
            or set(actual_transitions) != expected_transitions
            or len(actual_transitions) != len(expected_transitions)
        ):
            raise RuntimeError("incomplete or duplicate stage/transition coverage")
        write_json(output / "stages.json", {"rows": stage_rows})
        write_json(output / "transitions.json", {"rows": transition_rows})
        write_csv(output / "stages.csv", flat_rows(stage_rows))
        write_csv(output / "transitions.csv", flat_rows(transition_rows))
        write_csv(output / "calibration.csv", calibration_rows)
        write_report(output / "ENERGY_TRACE.md", stage_rows, transition_rows)
        write_json(
            output / "artifacts.json",
            {path.name: file_sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
        )
        write_json(
            output / "completion.json",
            {
                "completed": True,
                "profile": args.profile,
                "seed": SEED,
                "seed_count": 1,
                "checkpoint_analyses": len(coverage),
                "stage_rows": len(stage_rows),
                "transition_rows": len(transition_rows),
                "new_training_runs": 0,
                "optimizer_updates": 0,
                "parameters_preserved": True,
                "full_graphs": True,
                "actual_citation_data": args.profile == "full",
                "scope": "full_frozen_citation_diagnostic"
                if args.profile == "full"
                else "DEBUG_pipeline_only",
                "artifact_manifest_sha256": file_sha256(output / "artifacts.json"),
                "seconds": time.perf_counter() - start,
                "final_resources": runtime_resources(device),
            },
        )
        print(
            f"[all complete] {len(coverage)} checkpoint analyses; seed11; training=0; "
            f"results={output}",
            flush=True,
        )
    except Exception as error:
        write_json(
            output / "failure.json",
            {"error": str(error), "traceback": traceback.format_exc(), "optimizer_updates": 0},
        )
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-run", type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="source result inventory, also usable from an isolated code snapshot",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[SEED], default=SEED)
    parser.add_argument("--profile", choices=["full", "debug"], default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--feature-chunks", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--calibration-repeats", type=int, default=2)
    args = parser.parse_args()
    with torch.no_grad():
        execute(args)


if __name__ == "__main__":
    main()
