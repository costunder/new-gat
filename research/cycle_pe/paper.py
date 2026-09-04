"""Linux/CUDA paper entry point for the static cycle-PE track.

Examples
--------
python -m research.cycle_pe.paper --suite core --data-root data --output-dir runs/cycle \
    --device cuda --seed 2025
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from chartgat.observability import observed
from chartgat.seeds import SeedAxes, resolve_seed_axes
from research.cycle_pe.paper_adapters import (
    BREC_CATEGORIES,
    BREC_OFFICIAL_NUM_RELABEL,
    BREC_OFFICIAL_PAIR_COUNT,
    BREC_OFFICIAL_RECORD_COUNT,
    BREC_SOURCE_URL,
    BRECAdapter,
    load_brec_v3,
    load_zinc12k,
)
from research.cycle_pe.paper_data import (
    GENERATOR_VERSION,
    DatasetBundle,
    load_or_generate_cycle_count_ood,
    sha256_file,
)
from research.cycle_pe.paper_model import (
    PE_VARIANTS,
    BatchOutput,
    PaperCycleModel,
    PreparedBatch,
    PreparedGraph,
    pack_prepared_graphs,
    prepare_splits,
)
from research.cycle_pe.paper_train import (
    TrainSettings,
    clone_cpu_state,
    cuda_autocast,
    evaluate_supervised,
    make_grad_scaler,
    require_finite_loss,
    resolve_device,
    runtime_environment,
    seed_everything,
    train_supervised,
    validate_first_step_gradients,
    validate_optimizer_ownership,
)
from research.cycle_pe.resource_monitor import (
    FailureSafeResourceMonitor,
    resource_failure_boundary,
    resource_failure_observations,
)

PAPER_SCHEMA_VERSION = 2
BREC_OFFICIAL_SEEDS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
BREC_PROTOCOLS = ("official", "custom")
BREC_OFFICIAL_BATCH_SIZE = 16
BREC_OFFICIAL_EPOCHS = 20
BREC_OFFICIAL_LEARNING_RATE = 1e-4
BREC_OFFICIAL_WEIGHT_DECAY = 1e-4
BREC_OFFICIAL_THRESHOLD = 72.34
BREC_OUTPUT_DIM = 16
COMMAND_CONTRACT = (
    "python -m research.cycle_pe.paper --suite core|brec|zinc|all "
    "--data-root PATH --output-dir PATH --device cuda --seed N "
    "[--data-seed N --split-seed N --chart-seed N --model-seed N] [--workers N] "
    "[--prefetch-factor N] "
    "[--prepare-only] [--allow-download] [--brec-protocol official|custom] "
    "[--brec-seeds 100,...,1000]"
)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _claim_empty_output(path: Path) -> None:
    """Refuse to collide with an existing run before creating any artifact."""

    if path.parent == path:
        raise ValueError("--output-dir cannot be a filesystem root")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"--output-dir is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"--output-dir already contains artifacts; choose a new empty path: {path}"
            )
    else:
        path.mkdir(parents=True)


def _preserve_failed_suite_output(
    path: Path, suite: str | None, *, reason: str
) -> str | None:
    """Validate and preserve the current run's incomplete suite for diagnosis."""

    if suite is None:
        return None
    lexical_root = Path(os.path.abspath(path.expanduser()))
    root = lexical_root.resolve()
    if root != lexical_root or root.parent == root:
        raise RuntimeError(f"refusing to inspect unsafe paper output root: {lexical_root}")
    lexical_target = lexical_root / suite
    target = lexical_target.resolve()
    if target != lexical_target or target.parent != root or target.name != suite:
        raise RuntimeError(f"refusing to preserve unsafe suite output target: {lexical_target}")
    if not lexical_target.exists():
        return None
    event = {
        "event": "preserve_failed_suite_output",
        "path": str(lexical_target),
        "reason": reason,
    }
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
    return str(lexical_target)


def _artifact_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _argument_manifest(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            result[name] = str(value)
        elif isinstance(value, tuple):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _implementation_hashes() -> dict[str, str]:
    module_root = Path(__file__).resolve().parent
    repository_root = module_root.parents[1]
    paths = [
        *(path for path in sorted(module_root.glob("paper*.py")) if path.is_file()),
        module_root / "resource_monitor.py",
        repository_root / "src" / "chartgat" / "observability.py",
    ]
    return {
        path.relative_to(repository_root).as_posix(): sha256_file(path) for path in paths
    }


def _split_statistics(bundle: DatasetBundle) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, graphs in bundle.splits.items():
        betas = [graph.beta for graph in graphs]
        result[split] = {
            "graphs": len(graphs),
            "nodes": sum(graph.num_nodes for graph in graphs),
            "edges": sum(len(graph.edges) for graph in graphs),
            "cycle_rank_min": min(betas) if betas else None,
            "cycle_rank_max": max(betas) if betas else None,
            "families": sorted({graph.family for graph in graphs}),
        }
    return result


def _prepared_data_observability(
    bundle: DatasetBundle, prepared: dict[str, list[PreparedGraph]]
) -> dict[str, Any]:
    graphs = [graph for split in prepared.values() for graph in split]
    if not graphs:
        raise ValueError("prepared Cycle paper dataset cannot be empty")

    def distribution(values: list[int], unit: str) -> dict[str, Any]:
        return {
            "minimum": observed(min(values), unit=unit),
            "mean": observed(sum(values) / len(values), unit=unit),
            "maximum": observed(max(values), unit=unit),
            "total": observed(sum(values), unit=unit),
        }

    loaded = sum(len(values) for values in bundle.splits.values())
    actual = len(graphs)
    first = graphs[0]
    return {
        "dataset": bundle.name,
        "loaded_split_graph_counts": {
            name: len(values) for name, values in bundle.splits.items()
        },
        "actual_used_split_graph_counts": {
            name: len(values) for name, values in prepared.items()
        },
        "loaded_graph_count": loaded,
        "actual_used_graph_count": actual,
        "actual_used_fraction_of_loaded_graphs": observed(actual / loaded, unit="fraction"),
        "sampling_ratio": observed(1.0, unit="fraction"),
        "nodes_per_graph": distribution([graph.num_nodes for graph in graphs], "nodes"),
        "canonical_undirected_edges_per_graph": distribution(
            [int(graph.edges.shape[0]) for graph in graphs], "edges"
        ),
        "cycle_rank_per_graph": distribution(
            [graph.cycle_rank for graph in graphs], "cycles"
        ),
        "input_tensor_shapes": {
            "node_features": [None, int(first.node_features.shape[1])],
            "edge_features": [None, int(first.edge_features.shape[1])],
            "raw_cycle_basis": [None, None],
            "ragged_axes": "graph, node, edge, and cycle-rank axes; no truncation",
        },
        "time_window": observed(
            None, reason="not applicable to static graphs", unit="not_applicable"
        ),
        "input_resolution": observed(
            None, reason="not applicable to graph-structured inputs", unit="not_applicable"
        ),
        "preparation_cache": (
            "all prepared graph/PE tensors are retained in RAM and reused for every epoch"
        ),
        "debug_subset_fast_mode": False,
    }


def _resolve_seed_axes(args: argparse.Namespace) -> SeedAxes:
    return resolve_seed_axes(
        args.seed,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        chart_seed=args.chart_seed,
        model_seed=args.model_seed,
    )


def _seed_axis_policy(
    suite: str,
    axes: SeedAxes,
    *,
    brec_protocol: str | None = None,
    brec_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    if suite == "core":
        return {
            "data": {
                "value": axes.data,
                "used": True,
                "role": "CycleCount generation and content-addressed cache identity",
            },
            "split": {
                "value": axes.split,
                "used": False,
                "status": "not_applicable",
                "reason": (
                    "split families and size regimes are generator-defined; "
                    "data_seed generates each split"
                ),
            },
            "chart": {
                "value": axes.chart,
                "used": False,
                "status": "not_applicable",
                "reason": (
                    "static PE uses a deterministic BFS fundamental basis, not sampled charts"
                ),
            },
            "model": {
                "value": axes.model,
                "used": True,
                "role": "model initialization and supervised minibatch shuffling",
            },
        }
    if suite == "zinc":
        return {
            "data": {
                "value": axes.data,
                "used": False,
                "status": "not_applicable",
                "reason": "ZINC-12K is a fixed public dataset",
            },
            "split": {
                "value": axes.split,
                "used": False,
                "status": "not_applicable",
                "reason": "PyG official train/validation/test partitions are used unchanged",
            },
            "chart": {
                "value": axes.chart,
                "used": False,
                "status": "not_applicable",
                "reason": (
                    "static PE uses a deterministic BFS fundamental basis, not sampled charts"
                ),
            },
            "model": {
                "value": axes.model,
                "used": True,
                "role": "model initialization and supervised minibatch shuffling",
            },
        }
    if suite != "brec":
        raise ValueError(f"unknown seed-axis suite: {suite}")
    return {
        "data": {
            "value": axes.data,
            "used": False,
            "status": "not_applicable",
            "reason": "BREC v3 is a fixed official artifact",
        },
        "split": {
            "value": axes.split,
            "used": False,
            "status": "not_applicable",
            "reason": "BREC uses fixed paired RPC blocks rather than a randomized split",
        },
        "chart": {
            "value": axes.chart,
            "used": False,
            "status": "not_applicable",
            "reason": "static PE uses a deterministic BFS fundamental basis, not sampled charts",
        },
        "model": {
            "value": axes.model,
            "used": False,
            "status": "not_applicable",
            "reason": "outer model_seed is not mixed into BREC protocol seeds",
        },
        "protocol": {
            "name": f"brec_{brec_protocol or 'unspecified'}_search_seed",
            "used": True,
            "values": list(brec_seeds),
            "role": "BREC model initialization/search axis internal to the RPC protocol",
        },
    }


def _settings(args: argparse.Namespace, device: torch.device, suite: str) -> TrainSettings:
    default_epochs = {"core": 60, "zinc": 100, "brec": 20}
    default_lr = {"core": 1e-3, "zinc": 1e-3, "brec": 1e-4}
    default_weight_decay = {"core": 1e-5, "zinc": 1e-5, "brec": 1e-4}
    epochs = args.epochs if args.epochs is not None else default_epochs[suite]
    return TrainSettings(
        device=device,
        seed=_resolve_seed_axes(args).model,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=(args.learning_rate if args.learning_rate is not None else default_lr[suite]),
        weight_decay=(
            args.weight_decay if args.weight_decay is not None else default_weight_decay[suite]
        ),
        workers=args.workers,
        amp_requested=args.amp,
        pin_memory_requested=args.pin_memory,
        non_blocking_requested=args.non_blocking,
        prefetch_factor=args.prefetch_factor,
    )


def _effective_brec_protocol(args: argparse.Namespace) -> str:
    requested = getattr(args, "brec_protocol", None)
    if requested is None:
        return "official"
    return str(requested)


def _brec_reference_compatibility(protocol: str) -> dict[str, Any]:
    """Describe static upstream compatibility without claiming numerical parity."""

    return {
        "static_constants_and_control_flow_compatible": protocol == "official",
        "compatibility_scope": (
            "q=32, 400 ordered pairs, ten independent search seeds, batch size 16, "
            "20 epochs, Adam lr/weight_decay 1e-4, float32/no-AMP, no pair shuffle, "
            "no gradient clipping, and the upstream T2/reliability predicates"
        ),
        "differential_parity_verified": False,
        "parity_note": (
            "No golden-output or differential run against GraphPKU/BREC has been completed. "
            "Static protocol compatibility must not be interpreted as bytewise or numerical "
            "identity with the upstream runner."
        ),
    }


def _brec_settings(args: argparse.Namespace, device: torch.device, protocol: str) -> TrainSettings:
    if protocol == "custom":
        return _settings(args, device, "brec")
    return TrainSettings(
        device=device,
        seed=0,
        epochs=BREC_OFFICIAL_EPOCHS,
        batch_size=BREC_OFFICIAL_BATCH_SIZE,
        learning_rate=BREC_OFFICIAL_LEARNING_RATE,
        weight_decay=BREC_OFFICIAL_WEIGHT_DECAY,
        workers=0,
        amp_requested=False,
        pin_memory_requested=False,
        non_blocking_requested=False,
        prefetch_factor=args.prefetch_factor,
    )


def _model_dimensions(args: argparse.Namespace) -> tuple[int, int, int]:
    hidden = args.hidden_dim if args.hidden_dim is not None else 64
    pe = args.pe_dim if args.pe_dim is not None else 32
    layers = args.layers if args.layers is not None else 3
    return hidden, pe, layers


def _normalizer_json(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        level: {"mean": value.mean.tolist(), "std": value.std.tolist()}
        for level, value in stats.items()
    }


def _save_checkpoint(
    path: Path,
    model: PaperCycleModel,
    stats: dict[str, Any],
    *,
    variant: str,
    raw_width: int,
    model_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": PAPER_SCHEMA_VERSION,
            "variant": variant,
            "raw_width": raw_width,
            "model_seed": model_seed,
            "state_dict": clone_cpu_state(model),
            "target_normalization": _normalizer_json(stats),
        },
        path,
    )


@resource_failure_boundary
def _run_supervised_bundle(
    bundle: DatasetBundle,
    *,
    suite: str,
    suite_root: Path,
    args: argparse.Namespace,
    device: torch.device,
    train_split: str,
    validation_split: str,
    integer_targets: bool,
    target_tasks: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    preparation_started = time.perf_counter()
    prepared, raw_width = prepare_splits(
        bundle.splits,
        fit_split=train_split,
        required_variants=args.variants,
    )
    preparation_seconds = time.perf_counter() - preparation_started
    train_graphs = prepared[train_split]
    first = train_graphs[0]
    hidden_dim, pe_dim, layers = _model_dimensions(args)
    settings = _settings(args, device, suite)
    data_observability = _prepared_data_observability(bundle, prepared)
    target_names = {
        "edge": bundle.edge_target_names,
        "node": bundle.node_target_names,
        "graph": bundle.graph_target_names,
    }
    manifest: dict[str, Any] = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "suite": suite,
        "dataset": bundle.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed_axes": seed_axes.to_manifest(),
        "seed_axis_policy": _seed_axis_policy(suite, seed_axes),
        "prepare_only": args.prepare_only,
        "command_contract": COMMAND_CONTRACT,
        "cli_arguments": _argument_manifest(args),
        "implementation_sha256": _implementation_hashes(),
        "dataset_metadata": bundle.metadata or {},
        "split_statistics": _split_statistics(bundle),
        "split_sizes": {name: len(graphs) for name, graphs in bundle.splits.items()},
        "total_graphs": sum(len(graphs) for graphs in bundle.splits.values()),
        "data_observability": data_observability,
        "target_names": target_names,
        "target_tasks": {name: list(levels) for name, levels in target_tasks.items()},
        "target_independence_policy": (
            "Each target level is trained in an independent model; edge/node/graph labels "
            "are never optimized jointly."
        ),
        "raw_width": raw_width,
        "raw_width_policy": (
            f"maximum cycle rank from {train_split!r} only; OOD overflow is reported "
            "and never truncated"
        ),
        "raw_overflow_by_split": {
            split: {
                "graphs": sum(graph.cycle_rank > raw_width for graph in graphs),
                "max_cycle_rank": max((graph.cycle_rank for graph in graphs), default=None),
            }
            for split, graphs in prepared.items()
        },
        "preparation_seconds": preparation_seconds,
        "runtime_environment": runtime_environment(settings),
        "model": {
            "hidden_dim": hidden_dim,
            "pe_dim": pe_dim,
            "layers": layers,
            "node_input_dim": int(first.node_features.shape[1]),
            "edge_input_dim": int(first.edge_features.shape[1]),
        },
        "training": {
            "epochs": settings.epochs,
            "batch_size": settings.batch_size,
            "workers": settings.workers,
            "worker_policy": runtime_environment(settings)["worker_policy"],
            "prefetch_factor": (
                settings.prefetch_factor
                if settings.workers > 0
                else observed(
                    None,
                    reason="prefetch_factor is inactive because DataLoader workers is zero",
                )
            ),
            "persistent_workers": settings.workers > 0,
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "amp_requested": settings.amp_requested,
            "amp_effective": settings.amp,
            "pin_memory_effective": settings.pin_memory,
            "non_blocking_effective": settings.non_blocking,
        },
    }
    if bundle.cache_path is not None:
        manifest["cache"] = {
            "path": str(bundle.cache_path),
            "sha256": bundle.cache_sha256,
        }

    if args.prepare_only:
        manifest["variants"] = list(args.variants)
        manifest["experiments"] = {}
        manifest["artifacts"] = _artifact_checksums(suite_root)
        _write_json(suite_root / "manifest.json", manifest)
        return manifest

    experiment_summaries: dict[str, Any] = {}
    peak_gpu: int | None = None
    training_wall = 0.0
    for task_name, target_levels in target_tasks.items():
        task_summary: dict[str, Any] = {}
        for variant in args.variants:
            print(f"[{suite}] task={task_name} training variant={variant}", flush=True)
            overflow = {
                split: [graph for graph in graphs if graph.cycle_rank > raw_width]
                for split, graphs in prepared.items()
            }
            incompatible_splits = {
                split: graphs
                for split, graphs in overflow.items()
                if split != train_split and graphs
            }
            if variant == "raw" and incompatible_splits:
                summary = {
                    "status": "not_applicable_train_fitted_width_overflow",
                    "reason": (
                        "at least one complete non-training split contains a graph whose "
                        "cycle rank exceeds the train-fitted raw width; the entire raw "
                        "condition is not run, rather than selecting a compatible subset, "
                        "truncating coordinates, or fitting width on validation/test data"
                    ),
                    "fitted_raw_width": raw_width,
                    "incompatible_splits": {
                        split: {
                            "loaded_graphs": len(prepared[split]),
                            "overflow_graphs": len(graphs),
                            "maximum_cycle_rank": max(graph.cycle_rank for graph in graphs),
                            "graphs_used_for_checkpoint_selection_or_metrics": 0,
                        }
                        for split, graphs in incompatible_splits.items()
                    },
                    "training_performed": False,
                    "checkpoint_selection_performed": False,
                    "metric_calculation_performed": False,
                    "compatible_subset_used": False,
                    "truncated": False,
                    "validation_or_test_fitted": False,
                }
                task_summary[variant] = summary
                _write_json(suite_root / task_name / variant / "metrics.json", summary)
                print(
                    f"[{suite}] task={task_name} variant=raw status="
                    "not_applicable_train_fitted_width_overflow",
                    flush=True,
                )
                continue
            validation_graphs = prepared[validation_split]
            seed_everything(seed_axes.model)
            enabled = set(target_levels)
            model = PaperCycleModel(
                variant=variant,
                raw_width=raw_width,
                node_input_dim=int(first.node_features.shape[1]),
                edge_input_dim=int(first.edge_features.shape[1]),
                edge_output_dim=(len(bundle.edge_target_names) if "edge" in enabled else 0),
                node_output_dim=(len(bundle.node_target_names) if "node" in enabled else 0),
                graph_output_dim=(len(bundle.graph_target_names) if "graph" in enabled else 0),
                hidden_dim=hidden_dim,
                pe_dim=pe_dim,
                layers=layers,
                embedding_dim=0,
            )
            model, stats, history, runtime = train_supervised(
                model,
                train_graphs,
                validation_graphs,
                settings,
                target_levels=target_levels,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            evaluation_monitor = FailureSafeResourceMonitor(
                device,
                workload=f"cycle_paper_{suite}_{task_name}_{variant}_evaluation",
            )
            evaluation_resources_at_start = evaluation_monitor.start()
            evaluation_started = time.perf_counter()
            metrics: dict[str, Any] = {}
            evaluated_graphs = 0
            for split, graphs in prepared.items():
                evaluated_graphs += len(graphs)
                metrics[split] = evaluate_supervised(
                    model,
                    graphs,
                    stats,
                    settings,
                    target_names,
                    integer_targets=integer_targets,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                evaluation_peak = int(torch.cuda.max_memory_allocated(device))
                evaluation_reserved_peak = int(torch.cuda.max_memory_reserved(device))
            else:
                evaluation_peak = None
                evaluation_reserved_peak = None
            evaluation_seconds = time.perf_counter() - evaluation_started
            evaluation_resources = evaluation_monitor.finish(
                peak_allocated_bytes=evaluation_peak,
                peak_reserved_bytes=evaluation_reserved_peak,
            )
            training_peak = runtime["peak_gpu_memory_bytes"]
            training_reserved_peak = runtime["peak_gpu_reserved_memory_bytes"]
            runtime["peak_gpu_memory_bytes"] = (
                max(training_peak, evaluation_peak)
                if isinstance(training_peak, int) and isinstance(evaluation_peak, int)
                else training_peak
                if isinstance(training_peak, int)
                else evaluation_peak
            )
            runtime["peak_gpu_reserved_memory_bytes"] = (
                max(training_reserved_peak, evaluation_reserved_peak)
                if isinstance(training_reserved_peak, int)
                and isinstance(evaluation_reserved_peak, int)
                else training_reserved_peak
                if isinstance(training_reserved_peak, int)
                else evaluation_reserved_peak
            )
            runtime["evaluation_wall_seconds"] = evaluation_seconds
            runtime["evaluation_resources_at_start"] = evaluation_resources_at_start
            runtime["evaluation_resource_observability"] = evaluation_resources
            runtime["evaluation_throughput"] = {
                "scope": (
                    "CUDA-synchronized complete split evaluation; metric transfer and "
                    "aggregation included"
                ),
                "evaluated_graphs": evaluated_graphs,
                "evaluation_seconds": evaluation_seconds,
                "evaluated_graphs_per_second": (
                    observed(evaluated_graphs / evaluation_seconds, unit="graphs_per_second")
                    if evaluation_seconds > 0
                    else observed(
                        None,
                        reason="observed evaluation duration was zero",
                        unit="graphs_per_second",
                    )
                ),
            }
            runtime["total_train_evaluation_wall_seconds"] = float(runtime["wall_seconds"]) + float(
                runtime["evaluation_wall_seconds"]
            )
            variant_root = suite_root / task_name / variant
            _write_json(variant_root / "metrics.json", metrics)
            _write_json(variant_root / "history.json", history)
            _write_json(variant_root / "runtime.json", runtime)
            _save_checkpoint(
                variant_root / "model.pt",
                model,
                stats,
                variant=variant,
                raw_width=raw_width,
                model_seed=seed_axes.model,
            )
            runtime_peak = runtime["peak_gpu_memory_bytes"]
            if isinstance(runtime_peak, int):
                peak_gpu = runtime_peak if peak_gpu is None else max(peak_gpu, runtime_peak)
            training_wall += float(runtime["total_train_evaluation_wall_seconds"])
            reported_split = "test" if "test" in metrics else "id_test"
            reported = metrics[reported_split]
            task_summary[variant] = {
                "status": reported.get("status", "complete"),
                "reported_split": reported_split,
                "macro_normalized_mae": reported.get("macro_normalized_mae"),
                "best_validation_loss": runtime["best_validation_loss"],
                "total_parameters": runtime["model_observability"]["total_parameters"],
                "trainable_parameters": runtime["model_observability"][
                    "trainable_parameters"
                ],
                "optimizer_steps_completed": runtime["optimizer_steps_completed"],
                "gradient_connectivity_validated": runtime["gradient_connectivity"][
                    "validated_on_first_actual_backward"
                ],
                "training_throughput": runtime["throughput"],
                "evaluation_throughput": runtime["evaluation_throughput"],
                "total_train_evaluation_wall_seconds": runtime[
                    "total_train_evaluation_wall_seconds"
                ],
                "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        experiment_summaries[task_name] = task_summary

    manifest["variants"] = list(args.variants)
    manifest["experiments"] = experiment_summaries
    manifest["runtime_summary"] = {
        "train_evaluation_wall_seconds_sum": training_wall,
        "peak_gpu_memory_bytes_max": peak_gpu,
    }
    manifest["artifacts"] = _artifact_checksums(suite_root)
    _write_json(suite_root / "manifest.json", manifest)
    return manifest


def run_core(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    bundle = load_or_generate_cycle_count_ood(args.data_root, seed=seed_axes.data)
    if bundle.metadata is None:
        bundle.metadata = {}
    bundle.metadata.update(
        {
            "source": "built-in deterministic generator",
            "generator_version": GENERATOR_VERSION,
            "cache_identity_seed_axis": "data",
            "data_seed": seed_axes.data,
            "seed_axis_policy": _seed_axis_policy("core", seed_axes),
            "split_protocol": {
                "id_test": "held-out seeds, training graph families and size range",
                "size_ood": "held-out larger node-count range",
                "family_ood": "held-out small-world and local-chord families",
            },
            "protocol_coverage": {
                "size_and_family_ood": True,
                "degree_sequence_matched_counterfactuals": False,
                "note": (
                    "The implemented generator controls size/family but does not claim "
                    "degree-sequence-matched counterfactual coverage."
                ),
            },
        }
    )
    return _run_supervised_bundle(
        bundle,
        suite="core",
        suite_root=args.output_dir / "core",
        args=args,
        device=device,
        train_split="train",
        validation_split="validation",
        integer_targets=True,
        target_tasks={level: (level,) for level in args.core_targets},
    )


def run_zinc(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    bundle = load_zinc12k(args.data_root, allow_download=args.allow_download)
    if bundle.metadata is None:
        bundle.metadata = {}
    bundle.metadata["seed_axis_policy"] = _seed_axis_policy("zinc", seed_axes)
    return _run_supervised_bundle(
        bundle,
        suite="zinc",
        suite_root=args.output_dir / "zinc",
        args=args,
        device=device,
        train_split="train",
        validation_split="validation",
        integer_targets=False,
        target_tasks={"graph": ("graph",)},
    )


def _brec_batches(
    graphs: list[PreparedGraph],
    order: np.ndarray,
    *,
    batch_size: int,
) -> list[list[PreparedGraph]]:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("BREC graph batch size must be an even integer of at least two")
    pairs_per_batch = batch_size // 2
    result: list[list[PreparedGraph]] = []
    for start in range(0, len(order), pairs_per_batch):
        batch: list[PreparedGraph] = []
        for pair_index in order[start : start + pairs_per_batch]:
            index = 2 * int(pair_index)
            batch.extend((graphs[index], graphs[index + 1]))
        result.append(batch)
    return result


def _move_brec_batch(
    graphs: list[PreparedGraph], settings: TrainSettings, *, variant: str
) -> PreparedBatch:
    return pack_prepared_graphs(graphs, variant=variant, target_levels=()).to(
        settings.device, non_blocking=settings.non_blocking
    )


@torch.no_grad()
def brec_hotelling_t2(embeddings: Tensor) -> Tensor:
    """Match GraphPKU/BREC Release/base/test_BREC.py ``T2_calculation``."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 4 or embeddings.shape[0] % 2:
        raise ValueError("BREC embeddings must contain at least two interleaved pairs")
    # The official implementation operates in float32 without an extra q
    # multiplier: D_mean.T @ pinv(cov(D)) @ D_mean.
    matrix = embeddings.float()
    left = matrix[0::2].T
    right = matrix[1::2].T
    difference = left - right
    difference_mean = torch.mean(difference, dim=1).reshape(-1, 1)
    covariance = torch.cov(difference)
    inverse = torch.linalg.pinv(covariance)
    return torch.mm(torch.mm(difference_mean.T, inverse), difference_mean).reshape(())


def brec_rpc_decision(
    train_t2: Tensor | float,
    reliability_t2: Tensor | float,
    *,
    threshold: float,
) -> dict[str, bool]:
    """Apply the official distinguishability and reliability predicates."""

    train = torch.as_tensor(train_t2, dtype=torch.float32).reshape(())
    reliability = torch.as_tensor(reliability_t2, dtype=torch.float32).reshape(())
    distinguished = bool(
        (train > threshold).item() and not torch.isclose(train, reliability, atol=1e-6).item()
    )
    reliable = bool((reliability < threshold).item())
    return {
        "distinguished": distinguished,
        "reliable": reliable,
        "successful": distinguished and reliable,
    }


@torch.no_grad()
def _brec_t2(
    model: PaperCycleModel,
    graphs: list[PreparedGraph],
    settings: TrainSettings,
) -> float:
    model.eval()
    embeddings: list[Tensor] = []
    order = np.arange(len(graphs) // 2)
    for cpu_batch in _brec_batches(graphs, order, batch_size=settings.batch_size):
        batch = _move_brec_batch(
            cpu_batch, settings, variant=model.pe_encoder.variant
        )
        with cuda_autocast(settings.amp):
            outputs = model(batch)
        if not isinstance(outputs, BatchOutput) or outputs.embedding is None:
            raise RuntimeError("BREC packed forward did not produce graph embeddings")
        embeddings.append(outputs.embedding)
    return float(brec_hotelling_t2(torch.cat(embeddings, dim=0)).cpu())


def _train_brec_pair(
    model: PaperCycleModel,
    train_test: list[PreparedGraph],
    reliability: list[PreparedGraph],
    settings: TrainSettings,
    *,
    threshold: float,
    shuffle_pairs: bool,
    gradient_clip_norm: float | None,
) -> tuple[dict[str, Any], int | None]:
    model = model.to(settings.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    trainable_parameters = validate_optimizer_ownership(model, optimizer)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = make_grad_scaler(settings.amp)
    cosine = nn.CosineEmbeddingLoss(margin=0.0)
    if settings.pin_memory:
        train_test = [graph.pin_memory() for graph in train_test]
        reliability = [graph.pin_memory() for graph in reliability]
    if settings.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(settings.device)
        torch.cuda.synchronize(settings.device)
    started = time.perf_counter()
    final_loss = math.inf
    epochs_completed = 0
    optimizer_steps = 0
    processed_pairs = 0
    observed_graph_batch_sizes: list[int] = []
    gradient_connectivity: dict[str, Any] | None = None
    pairs_per_epoch = len(train_test) // 2
    batches_per_epoch = math.ceil(pairs_per_epoch / max(1, settings.batch_size // 2))
    for epoch in range(settings.epochs):
        model.train()
        if shuffle_pairs:
            order = np.random.default_rng(settings.seed + epoch).permutation(len(train_test) // 2)
        else:
            order = np.arange(len(train_test) // 2)
        total = torch.zeros((), device=settings.device, dtype=torch.float64)
        pairs = 0
        for cpu_batch in _brec_batches(train_test, order, batch_size=settings.batch_size):
            batch = _move_brec_batch(
                cpu_batch, settings, variant=model.pe_encoder.variant
            )
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(settings.amp):
                outputs = model(batch)
                if not isinstance(outputs, BatchOutput) or outputs.embedding is None:
                    raise RuntimeError("BREC packed forward did not produce graph embeddings")
                embedding = outputs.embedding
                target = -torch.ones(
                    embedding.shape[0] // 2,
                    device=embedding.device,
                    dtype=embedding.dtype,
                )
                loss = cosine(embedding[0::2], embedding[1::2], target)
            require_finite_loss(loss, "nonfinite BREC cosine loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if optimizer_steps == 0:
                gradient_connectivity = validate_first_step_gradients(model)
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm, error_if_nonfinite=True
                )
            scaler.step(optimizer)
            scaler.update()
            pair_count = embedding.shape[0] // 2
            optimizer_steps += 1
            processed_pairs += int(pair_count)
            observed_graph_batch_sizes.append(batch.batch_size)
            total += loss.detach().double() * pair_count
            pairs += pair_count
        final_loss = float(total.cpu()) / max(1, pairs)
        epochs_completed = epoch + 1
        scheduler.step(final_loss)
        if final_loss < 0.2:
            break
    if gradient_connectivity is None:
        raise RuntimeError("BREC training performed no actual backward pass")
    train_t2 = _brec_t2(model, train_test, settings)
    reliability_t2 = _brec_t2(model, reliability, settings)
    decision = brec_rpc_decision(train_t2, reliability_t2, threshold=threshold)
    if settings.device.type == "cuda":
        torch.cuda.synchronize(settings.device)
        peak = int(torch.cuda.max_memory_allocated(settings.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(settings.device))
    else:
        peak = None
        peak_reserved = None
    wall_seconds = time.perf_counter() - started
    result = {
        **decision,
        "train_test_t2": train_t2,
        "reliability_t2": reliability_t2,
        "threshold": threshold,
        "final_cosine_loss": final_loss,
        "epochs_completed": epochs_completed,
        "pair_shuffle": shuffle_pairs,
        "gradient_clip_norm": gradient_clip_norm,
        "wall_seconds": wall_seconds,
        "peak_gpu_memory_bytes": peak,
        "peak_gpu_reserved_memory_bytes": peak_reserved,
        "peak_gpu_memory_unavailable_reason": (
            None
            if settings.device.type == "cuda"
            else "training device is CPU, so CUDA allocator peaks are unavailable"
        ),
        "model_observability": {
            "name": type(model).__name__,
            "pe_variant": model.pe_encoder.variant,
            "layers": len(model.layers),
            "hidden_dimension": model.node_encoder[0].out_features,
            "pe_dimension": model.pe_encoder.pe_dim,
            "attention_heads": observed(
                None, reason="PaperCycleModel has no attention mechanism"
            ),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "optimizer_owned_trainable_parameters": trainable_parameters,
        },
        "gradient_connectivity": gradient_connectivity,
        "optimizer_observability": {
            "name": "Adam",
            "optimizer_steps_completed": optimizer_steps,
            "planned_maximum_optimizer_steps": settings.epochs * batches_per_epoch,
            "early_stop_criterion": "official/custom shared cosine loss < 0.2",
            "early_stop_triggered": epochs_completed < settings.epochs,
            "shortfall_reason": (
                "registered cosine-loss stopping criterion was satisfied"
                if epochs_completed < settings.epochs
                else None
            ),
        },
        "data_observability": {
            "train_test_graphs": len(train_test),
            "reliability_graphs": len(reliability),
            "train_test_pairs": pairs_per_epoch,
            "input_tensor_shapes": {
                "node_features": [
                    None,
                    int(train_test[0].node_features.shape[1]),
                ],
                "edge_features": [
                    None,
                    int(train_test[0].edge_features.shape[1]),
                ],
                "edge_index": [None, 2],
            },
            "sampling_ratio": observed(1.0, unit="fraction"),
            "actual_used_fraction_of_loaded_graphs": observed(1.0, unit="fraction"),
            "nodes_per_graph": {
                "minimum": min(graph.num_nodes for graph in [*train_test, *reliability]),
                "maximum": max(graph.num_nodes for graph in [*train_test, *reliability]),
            },
            "edges_per_graph": {
                "minimum": min(
                    int(graph.edges.shape[0]) for graph in [*train_test, *reliability]
                ),
                "maximum": max(
                    int(graph.edges.shape[0]) for graph in [*train_test, *reliability]
                ),
            },
            "time_window": observed(
                None, reason="BREC inputs are static graphs without a time axis"
            ),
            "input_resolution": observed(
                None, reason="BREC graph inputs have no spatial raster resolution"
            ),
            "debug_subset_fast_mode": False,
        },
        "batch_observability": {
            "batch_unit": "graphs in complete interleaved RPC pairs",
            "configured_physical_graph_batch_size": settings.batch_size,
            "observed_smallest_graph_batch_size": min(observed_graph_batch_sizes),
            "observed_largest_graph_batch_size": max(observed_graph_batch_sizes),
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_graph_batch_size": max(observed_graph_batch_sizes),
            "packed_disjoint_union_forward": True,
            "per_graph_gpu_forward_loop": False,
            "workers": settings.workers,
            "workers_reason": (
                "official BREC traverses paired in-memory RPC blocks in fixed order; no "
                "DataLoader, disk decode, or per-graph GPU forward occurs"
            ),
        },
        "throughput": {
            "scope": (
                "CUDA-synchronized end-to-end pair training plus final train/reliability "
                "T2 evaluation; includes CPU pair ordering and packed transfer"
            ),
            "processed_training_pairs": processed_pairs,
            "elapsed_seconds": wall_seconds,
            "training_pairs_per_second": (
                observed(processed_pairs / wall_seconds, unit="pairs_per_second")
                if wall_seconds > 0
                else observed(
                    None,
                    reason="observed BREC duration was zero",
                    unit="pairs_per_second",
                )
            ),
        },
    }
    return result, peak


def _aggregate_custom_brec_results(
    results: list[dict[str, Any]],
    *,
    pair_indices: list[int],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Compute the repository's explicitly custom reliable any-seed union."""

    by_pair: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    by_seed: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_pair[int(result["pair_index"])].append(result)
        by_seed[int(result["search_seed"])].append(result)
    pair_summary: list[dict[str, Any]] = []
    for pair_index in pair_indices:
        values = by_pair[pair_index]
        completed = [value for value in values if value.get("status") == "complete"]
        reliability_failures = sum(not bool(value["reliable"]) for value in completed)
        protocol_complete = len(completed) == len(seeds)
        final_success = bool(
            protocol_complete
            and reliability_failures == 0
            and any(bool(value["successful"]) for value in completed)
        )
        pair_summary.append(
            {
                "pair_index": pair_index,
                "category": _category_for_result(values),
                "attempts": len(values),
                "completed_attempts": len(completed),
                "not_applicable_attempts": len(values) - len(completed),
                "distinguished_seeds": sum(bool(value["distinguished"]) for value in completed),
                "successful_seeds": sum(bool(value["successful"]) for value in completed),
                "reliability_failures": reliability_failures,
                "protocol_complete": protocol_complete,
                "successful_pair": final_success,
            }
        )
    category_summary: dict[str, Any] = {}
    grouped_pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in pair_summary:
        grouped_pairs[str(value["category"])].append(value)
    for category, values in grouped_pairs.items():
        category_summary[category] = {
            "pairs": len(values),
            "successful_pairs": sum(bool(value["successful_pair"]) for value in values),
            "success_rate": float(np.mean([bool(value["successful_pair"]) for value in values])),
            "reliability_failures": sum(int(value["reliability_failures"]) for value in values),
        }
    seed_summary = {
        str(seed): {
            "pairs": len(by_seed[seed]),
            "completed_attempts": sum(value.get("status") == "complete" for value in by_seed[seed]),
            "distinguished_attempts": sum(
                bool(value["distinguished"])
                for value in by_seed[seed]
                if value.get("status") == "complete"
            ),
            "successful_attempts": sum(
                bool(value["successful"])
                for value in by_seed[seed]
                if value.get("status") == "complete"
            ),
            "reliability_failures": sum(
                not bool(value["reliable"])
                for value in by_seed[seed]
                if value.get("status") == "complete"
            ),
        }
        for seed in seeds
    }
    successful_pairs = sum(bool(value["successful_pair"]) for value in pair_summary)
    return {
        "protocol": "custom",
        "metric_name": "custom_pairwise_union",
        "pairs": len(pair_indices),
        "seeds": list(seeds),
        "attempts": len(results),
        "successful_pairs": successful_pairs,
        "success_rate": successful_pairs / max(1, len(pair_indices)),
        "reliability_failures": sum(int(value["reliability_failures"]) for value in pair_summary),
        "not_applicable_attempts": sum(
            int(value["not_applicable_attempts"]) for value in pair_summary
        ),
        "final_pair_rule": (
            "at least one seed distinguishes, every configured seed is evaluable, and no "
            "seed fails reliability"
        ),
        "per_pair": pair_summary,
        "per_seed": seed_summary,
        "categories": category_summary,
    }


def _aggregate_official_brec_results(
    results: list[dict[str, Any]],
    *,
    pair_indices: list[int],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Reproduce BREC's per-seed Correct/Fail/Real_correct reporting.

    The official search script launches ten independent complete runs and does
    not define an any-seed union score.  We therefore retain the per-seed
    results and expose only the README's global reliability-validity gate.
    """

    by_seed: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_seed[int(result["search_seed"])].append(result)

    per_seed: dict[str, Any] = {}
    for seed in seeds:
        values = by_seed[seed]
        completed = [value for value in values if value.get("status") == "complete"]
        correct = sum(bool(value["distinguished"]) for value in completed)
        failures = sum(not bool(value["reliable"]) for value in completed)
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in completed:
            grouped[str(value["category"])].append(value)
        categories: dict[str, Any] = {}
        for category, category_values in grouped.items():
            category_correct = sum(bool(value["distinguished"]) for value in category_values)
            category_failures = sum(not bool(value["reliable"]) for value in category_values)
            categories[category] = {
                "pairs": len(category_values),
                "Correct": category_correct,
                "Fail": category_failures,
                "Real_correct": category_correct - category_failures,
            }
        protocol_complete = len(completed) == len(pair_indices)
        per_seed[str(seed)] = {
            "pairs_expected": len(pair_indices),
            "attempts": len(values),
            "completed_attempts": len(completed),
            "not_applicable_attempts": len(values) - len(completed),
            "protocol_complete": protocol_complete,
            "Correct": correct,
            "Fail": failures,
            "Real_correct": correct - failures,
            "categories": categories,
        }

    all_seeds_complete = all(per_seed[str(seed)]["protocol_complete"] for seed in seeds)
    global_valid = bool(
        all_seeds_complete and all(per_seed[str(seed)]["Fail"] == 0 for seed in seeds)
    )
    return {
        "protocol": "official",
        "pairs": len(pair_indices),
        "seeds": list(seeds),
        "attempts": len(results),
        "all_seeds_complete": all_seeds_complete,
        "global_valid": global_valid,
        "global_valid_definition": (
            "repository-defined conservative gate: every configured seed is complete and "
            "has Fail == 0; this is not an upstream BREC metric"
        ),
        "per_seed": per_seed,
        "merged_score": None,
        "score_note": (
            "GraphPKU/BREC test_BREC_search.py emits independent complete runs per seed; "
            "no any-seed pair union is labeled as an official score."
        ),
    }


def _aggregate_brec_results(
    results: list[dict[str, Any]],
    *,
    pair_indices: list[int],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Compatibility alias for the explicitly custom pairwise-union metric."""

    return _aggregate_custom_brec_results(results, pair_indices=pair_indices, seeds=seeds)


def _category_for_result(values: list[dict[str, Any]]) -> str:
    return str(values[0]["category"]) if values else "unknown"


def _brec_model_seed(search_seed: int, pair_index: int) -> int:
    return int((search_seed * 1_000_003 + pair_index) % (2**31 - 1))


def _validate_official_brec_arguments(args: argparse.Namespace) -> None:
    if args.brec_num_relabel != BREC_OFFICIAL_NUM_RELABEL:
        raise ValueError("official BREC mode requires --brec-num-relabel 32")
    if args.brec_threshold is not None and not math.isclose(
        float(args.brec_threshold), BREC_OFFICIAL_THRESHOLD, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("official BREC mode requires --brec-threshold 72.34")
    if tuple(args.brec_seeds) != BREC_OFFICIAL_SEEDS:
        raise ValueError("official BREC mode requires search seeds 100,200,...,1000 in that order")


def _prepare_brec_pair(
    adapter: BRECAdapter,
    pair_index: int,
    *,
    required_variants: tuple[str, ...],
) -> tuple[str, dict[str, list[PreparedGraph]], int, list[int], list[PreparedGraph]]:
    pair = adapter.load_pair(pair_index)
    prepared, raw_width = prepare_splits(
        {
            "train_test": list(pair.train_test),
            "reliability": list(pair.reliability),
        },
        fit_split="train_test",
        required_variants=required_variants,
    )
    betas = [graph.cycle_rank for graph in prepared["train_test"] + prepared["reliability"]]
    reliability_overflow = [
        graph for graph in prepared["reliability"] if graph.cycle_rank > raw_width
    ]
    return pair.category, prepared, raw_width, betas, reliability_overflow


def _brec_seeded_settings(settings: TrainSettings, seed: int) -> TrainSettings:
    return TrainSettings(
        device=settings.device,
        seed=seed,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        workers=settings.workers,
        amp_requested=settings.amp_requested,
        pin_memory_requested=settings.pin_memory_requested,
        non_blocking_requested=settings.non_blocking_requested,
        prefetch_factor=settings.prefetch_factor,
    )


def _execute_brec_attempt(
    *,
    variant: str,
    pair_index: int,
    category: str,
    prepared: dict[str, list[PreparedGraph]],
    raw_width: int,
    betas: list[int],
    reliability_overflow: list[PreparedGraph],
    search_seed: int,
    rng_seed: int,
    rng_scope: str,
    settings: TrainSettings,
    hidden_dim: int,
    pe_dim: int,
    layers: int,
    threshold: float,
    protocol: str,
) -> tuple[dict[str, Any], int | None]:
    common = {
        "pair_index": pair_index,
        "category": category,
        "search_seed": search_seed,
        "rng_seed": rng_seed,
        "rng_scope": rng_scope,
        "raw_width": raw_width,
        "cycle_rank_min": min(betas),
        "cycle_rank_max": max(betas),
    }
    if variant == "raw" and reliability_overflow:
        return (
            {
                **common,
                "status": "not_applicable_train_fitted_width_overflow",
                "overflow_graphs": len(reliability_overflow),
                "max_overflow_cycle_rank": max(graph.cycle_rank for graph in reliability_overflow),
                "truncated": False,
                "distinguished": False,
                "reliable": False,
                "successful": False,
            },
            None,
        )

    first = prepared["train_test"][0]
    model = PaperCycleModel(
        variant=variant,
        raw_width=raw_width,
        node_input_dim=int(first.node_features.shape[1]),
        edge_input_dim=int(first.edge_features.shape[1]),
        edge_output_dim=0,
        node_output_dim=0,
        graph_output_dim=0,
        hidden_dim=hidden_dim,
        pe_dim=pe_dim,
        layers=layers,
        embedding_dim=BREC_OUTPUT_DIM,
    )
    result, peak = _train_brec_pair(
        model,
        prepared["train_test"],
        prepared["reliability"],
        _brec_seeded_settings(settings, rng_seed),
        threshold=threshold,
        shuffle_pairs=protocol == "custom",
        gradient_clip_norm=5.0 if protocol == "custom" else None,
    )
    result.update(common)
    result["status"] = "complete"
    del model
    if settings.device.type == "cuda":
        torch.cuda.empty_cache()
    return result, peak


@resource_failure_boundary
def run_brec(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    protocol = _effective_brec_protocol(args)
    if protocol == "official":
        _validate_official_brec_arguments(args)
    adapter: BRECAdapter = load_brec_v3(
        args.data_root,
        num_relabel=args.brec_num_relabel,
        allow_download=args.allow_download,
        protocol=protocol,
    )
    if protocol == "official":
        threshold = BREC_OFFICIAL_THRESHOLD
    elif args.brec_threshold is None:
        if adapter.num_relabel != 32:
            raise ValueError(
                "The official 72.34 RPC threshold is calibrated for q=32. Pass "
                "--brec-threshold explicitly for a customized --brec-num-relabel."
            )
        threshold = BREC_OFFICIAL_THRESHOLD
    else:
        threshold = float(args.brec_threshold)
    suite_root = args.output_dir / "brec"
    settings = _brec_settings(args, device, protocol)
    hidden_dim, pe_dim, layers = _model_dimensions(args)
    pair_indices = list(range(adapter.pair_count))
    manifest: dict[str, Any] = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "suite": "brec",
        "dataset": "BREC v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed_axes": seed_axes.to_manifest(),
        "seed_axis_policy": _seed_axis_policy(
            "brec",
            seed_axes,
            brec_protocol=protocol,
            brec_seeds=args.brec_seeds,
        ),
        "prepare_only": args.prepare_only,
        "command_contract": COMMAND_CONTRACT,
        "cli_arguments": _argument_manifest(args),
        "implementation_sha256": _implementation_hashes(),
        "dataset_metadata": adapter.metadata,
        "brec_protocol": {
            "effective": protocol,
            "default_policy": "official unless --brec-protocol custom is explicitly requested",
            "official_reference_compatibility": _brec_reference_compatibility(protocol),
            "custom_metric": "custom_pairwise_union" if protocol == "custom" else None,
            "outer_model_seed_used": False,
            "protocol_seed_axis": list(args.brec_seeds),
        },
        "rpc_reference": {
            "num_relabel": adapter.num_relabel,
            "embedding_dim": BREC_OUTPUT_DIM,
            "threshold": threshold,
            "search_seeds": list(args.brec_seeds),
            "t2_formula": "D_mean.T @ torch.linalg.pinv(torch.cov(D)) @ D_mean",
            "q_multiplier": False,
            "distinction_rule": (
                "train_t2 > threshold and not torch.isclose(train_t2, reliability_t2, atol=1e-6)"
            ),
            "reliability_rule": "reliability_t2 < threshold",
            "categories": BREC_CATEGORIES,
            "source_url": BREC_SOURCE_URL,
            "reference_implementation": (
                "https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC.py"
            ),
            "seed_reference": (
                "https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC_search.py"
            ),
        },
        "official_artifact_contract": {
            "required_only_in_official_mode": True,
            "num_relabel": BREC_OFFICIAL_NUM_RELABEL,
            "pair_count": BREC_OFFICIAL_PAIR_COUNT,
            "record_count": BREC_OFFICIAL_RECORD_COUNT,
            "sha256_pinned_by_upstream": False,
            "sha256_policy": "record provenance hash; do not claim an upstream published pin",
        },
        "pairs_selected": pair_indices,
        "runtime_environment": runtime_environment(settings),
        "training": {
            "protocol": protocol,
            "epochs": settings.epochs,
            "batch_size_graphs": settings.batch_size,
            "physical_batch_size_graphs": settings.batch_size,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size_formula": (
                f"{settings.batch_size} physical x 1 accumulation x 1 data-parallel worker"
            ),
            "workers": settings.workers,
            "workers_note": "BREC RPC preserves explicit pairs and does not use DataLoader",
            "prefetch_factor": observed(
                None,
                reason="BREC uses ordered in-memory RPC pair packing, not a DataLoader",
            ),
            "cache": "each decoded RPC pair is prepared once and reused across its epochs",
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "amp_effective": settings.amp,
            "pin_memory_effective": settings.pin_memory,
            "non_blocking_effective": settings.non_blocking,
            "pair_shuffle": protocol == "custom",
            "gradient_clip_norm": 5.0 if protocol == "custom" else None,
            "rng_policy": (
                "seed once per variant and official search seed, then traverse all 400 pairs "
                "in order without reseeding"
                if protocol == "official"
                else "derive and reset one model seed per variant, search seed, and pair"
            ),
            "requested_global_overrides": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "amp": args.amp,
                "pin_memory": args.pin_memory,
                "non_blocking": args.non_blocking,
            },
            "official_overrides_global_training_options": protocol == "official",
        },
        "raw_width_policy": (
            "fit on train_test graphs for each RPC pair only; reliability overflow is "
            "not applicable and is never truncated or fitted"
        ),
    }
    # Parse representative complete RPC pairs even in prepare-only mode. This
    # catches malformed graph6, disconnected graphs, and PE extraction failures.
    if args.prepare_only:
        check_indices = list(dict.fromkeys((pair_indices[0], pair_indices[-1])))
        checks: list[dict[str, Any]] = []
        for pair_index in check_indices:
            category, prepared, raw_width, _, _ = _prepare_brec_pair(
                adapter,
                pair_index,
                required_variants=args.variants,
            )
            checks.append(
                {
                    "pair": pair_index,
                    "category": category,
                    "graphs": sum(len(graphs) for graphs in prepared.values()),
                    "raw_width": raw_width,
                    "reliability_raw_overflow_graphs": sum(
                        graph.cycle_rank > raw_width for graph in prepared["reliability"]
                    ),
                }
            )
        manifest["preparation_checks"] = checks
        manifest["preparation_check_policy"] = "first and last pair of the supplied artifact"
        manifest["variants"] = list(args.variants)
        manifest["artifacts"] = _artifact_checksums(suite_root)
        _write_json(suite_root / "manifest.json", manifest)
        return manifest

    pair_results: dict[str, list[dict[str, Any]]] = {variant: [] for variant in args.variants}
    peak_gpu: int | None = None
    peak_gpu_reserved: int | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    resource_monitor = FailureSafeResourceMonitor(
        device, workload="cycle_paper_brec_suite"
    )
    resources_at_start = resource_monitor.start()
    print(
        json.dumps(
            {
                "kind": "cycle_paper_brec_pre_run_observability",
                "protocol": protocol,
                "model": {
                    "name": "PaperCycleModel",
                    "variants": list(args.variants),
                    "hidden_dimension": hidden_dim,
                    "pe_dimension": pe_dim,
                    "layers": layers,
                    "embedding_dimension": BREC_OUTPUT_DIM,
                    "attention_heads": observed(
                        None, reason="PaperCycleModel has no attention mechanism"
                    ),
                },
                "data": {
                    "official_pair_count": adapter.pair_count,
                    "selected_pair_count": len(pair_indices),
                    "selected_fraction": observed(
                        len(pair_indices) / adapter.pair_count, unit="fraction"
                    ),
                    "num_relabels": adapter.num_relabel,
                    "sampling_ratio": observed(1.0, unit="fraction"),
                    "debug_subset_fast_mode": False,
                },
                "batch": manifest["training"],
                "resources_at_start": resources_at_start,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    run_started = time.perf_counter()
    if protocol == "official":
        for variant in args.variants:
            for search_seed in args.brec_seeds:
                seed_everything(search_seed)
                for position, pair_index in enumerate(pair_indices, start=1):
                    print(
                        f"[brec:official] variant={variant} seed={search_seed} "
                        f"pair={pair_index} ({position}/{len(pair_indices)})",
                        flush=True,
                    )
                    category, prepared, raw_width, betas, reliability_overflow = _prepare_brec_pair(
                        adapter,
                        pair_index,
                        required_variants=(variant,),
                    )
                    result, pair_peak = _execute_brec_attempt(
                        variant=variant,
                        pair_index=pair_index,
                        category=category,
                        prepared=prepared,
                        raw_width=raw_width,
                        betas=betas,
                        reliability_overflow=reliability_overflow,
                        search_seed=search_seed,
                        rng_seed=search_seed,
                        rng_scope="variant_search_seed_full_pair_sequence",
                        settings=settings,
                        hidden_dim=hidden_dim,
                        pe_dim=pe_dim,
                        layers=layers,
                        threshold=float(threshold),
                        protocol=protocol,
                    )
                    pair_results[variant].append(result)
                    if pair_peak is not None:
                        peak_gpu = pair_peak if peak_gpu is None else max(peak_gpu, pair_peak)
                    pair_reserved = result.get("peak_gpu_reserved_memory_bytes")
                    if isinstance(pair_reserved, int):
                        peak_gpu_reserved = (
                            pair_reserved
                            if peak_gpu_reserved is None
                            else max(peak_gpu_reserved, pair_reserved)
                        )
    else:
        for position, pair_index in enumerate(pair_indices, start=1):
            print(f"[brec:custom] pair={pair_index} ({position}/{len(pair_indices)})", flush=True)
            category, prepared, raw_width, betas, reliability_overflow = _prepare_brec_pair(
                adapter,
                pair_index,
                required_variants=args.variants,
            )
            for variant in args.variants:
                for search_seed in args.brec_seeds:
                    model_seed = _brec_model_seed(search_seed, pair_index)
                    seed_everything(model_seed)
                    result, pair_peak = _execute_brec_attempt(
                        variant=variant,
                        pair_index=pair_index,
                        category=category,
                        prepared=prepared,
                        raw_width=raw_width,
                        betas=betas,
                        reliability_overflow=reliability_overflow,
                        search_seed=search_seed,
                        rng_seed=model_seed,
                        rng_scope="derived_per_pair_variant_search_seed",
                        settings=settings,
                        hidden_dim=hidden_dim,
                        pe_dim=pe_dim,
                        layers=layers,
                        threshold=float(threshold),
                        protocol=protocol,
                    )
                    pair_results[variant].append(result)
                    if pair_peak is not None:
                        peak_gpu = pair_peak if peak_gpu is None else max(peak_gpu, pair_peak)
                    pair_reserved = result.get("peak_gpu_reserved_memory_bytes")
                    if isinstance(pair_reserved, int):
                        peak_gpu_reserved = (
                            pair_reserved
                            if peak_gpu_reserved is None
                            else max(peak_gpu_reserved, pair_reserved)
                        )

    summaries: dict[str, Any] = {}
    for variant, results in pair_results.items():
        if protocol == "official":
            summaries[variant] = _aggregate_official_brec_results(
                results, pair_indices=pair_indices, seeds=args.brec_seeds
            )
        else:
            summaries[variant] = _aggregate_custom_brec_results(
                results, pair_indices=pair_indices, seeds=args.brec_seeds
            )
        _write_json(suite_root / variant / "pairs.json", results)
        _write_json(suite_root / variant / "metrics.json", summaries[variant])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    run_seconds = time.perf_counter() - run_started
    resources = resource_monitor.finish(
        peak_allocated_bytes=peak_gpu,
        peak_reserved_bytes=peak_gpu_reserved,
    )
    recorded_attempts = sum(len(results) for results in pair_results.values())
    manifest["variants"] = summaries
    manifest["runtime_summary"] = {
        "wall_seconds": run_seconds,
        "peak_gpu_memory_bytes_max": peak_gpu,
        "peak_gpu_reserved_memory_bytes_max": peak_gpu_reserved,
        "peak_gpu_memory_unavailable_reason": (
            None
            if device.type == "cuda"
            else "suite device is CPU, so CUDA allocator peaks are unavailable"
        ),
        "resources_at_start": resources_at_start,
        "resource_observability": resources,
        "throughput": {
            "scope": (
                "CUDA-synchronized complete BREC suite including pair preparation, training, "
                "and train/reliability T2 evaluation"
            ),
            "recorded_variant_seed_pair_attempts": recorded_attempts,
            "attempts_per_second": (
                observed(recorded_attempts / run_seconds, unit="attempts_per_second")
                if run_seconds > 0
                else observed(
                    None,
                    reason="observed BREC suite duration was zero",
                    unit="attempts_per_second",
                )
            ),
        },
    }
    print(
        json.dumps(
            {
                "kind": "cycle_paper_brec_post_run_observability",
                "recorded_attempts": recorded_attempts,
                "throughput": manifest["runtime_summary"]["throughput"],
                "resource_summary": resources["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    manifest["artifacts"] = _artifact_checksums(suite_root)
    _write_json(suite_root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static cycle-PE paper runner (CycleCount-OOD, BREC v3, ZINC-12K)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--suite", choices=("core", "brec", "zinc", "all"), default="core")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_runs/cycle_pe"))
    parser.add_argument("--device", default="cuda", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="CycleCount generation/cache axis; defaults to --seed",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="split axis; recorded as not applicable for current Cycle PE suites",
    )
    parser.add_argument(
        "--chart-seed",
        type=int,
        default=None,
        help="chart axis; static Cycle PE records this as not applicable",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="supervised initialization/minibatch axis; defaults to --seed",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--variants",
        default="raw,set,projector",
        help="own PE ablations: raw,set,projector; no_pe only when explicitly requested",
    )
    parser.add_argument(
        "--core-targets",
        default="edge,node,graph",
        help="independent CycleCount target levels selected from edge,node,graph",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers for core/ZINC")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="batches prefetched per DataLoader worker when --workers is positive",
    )
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--pe-dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use CUDA autocast and GradScaler (always disabled on CPU)",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--non-blocking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--brec-protocol",
        choices=BREC_PROTOCOLS,
        default=None,
        help="official by default; custom must be requested explicitly on a supplied artifact",
    )
    parser.add_argument("--brec-num-relabel", type=int, default=32)
    parser.add_argument("--brec-threshold", type=float)
    parser.add_argument(
        "--brec-seeds",
        default=",".join(str(seed) for seed in BREC_OFFICIAL_SEEDS),
        help="comma-separated BREC model-search seeds",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow official BREC/PyG ZINC download when a local cache is absent",
    )
    return parser


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in value.split(",") if part.strip())
    if not variants:
        raise ValueError("--variants cannot be empty")
    unknown = [variant for variant in variants if variant not in PE_VARIANTS]
    if unknown:
        raise ValueError(f"unknown PE variant(s): {', '.join(unknown)}; choose from {PE_VARIANTS}")
    if len(set(variants)) != len(variants):
        raise ValueError("--variants must not contain duplicates")
    return variants


def _parse_core_targets(value: str) -> tuple[str, ...]:
    targets = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = ("edge", "node", "graph")
    if not targets or any(target not in allowed for target in targets):
        raise ValueError(f"--core-targets must be a comma-separated subset of {allowed}")
    if len(set(targets)) != len(targets):
        raise ValueError("--core-targets must not contain duplicates")
    return targets


def _parse_brec_seeds(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError("--brec-seeds cannot be empty")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--brec-seeds must contain comma-separated integers") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError("--brec-seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--brec-seeds must not contain duplicates")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_owned = False
    completed: list[str] = []
    selected: tuple[str, ...] = ()
    try:
        args.variants = _parse_variants(args.variants)
        args.core_targets = _parse_core_targets(args.core_targets)
        args.brec_seeds = _parse_brec_seeds(args.brec_seeds)
        args.brec_protocol = _effective_brec_protocol(args)
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        seed_axes = _resolve_seed_axes(args)
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.workers < 0:
            raise ValueError("--workers must be non-negative")
        if args.prefetch_factor < 1:
            raise ValueError("--prefetch-factor must be positive")
        if args.epochs is not None and args.epochs < 1:
            raise ValueError("--epochs must be positive")
        if args.hidden_dim is not None and args.hidden_dim < 4:
            raise ValueError("--hidden-dim must be at least 4")
        if args.pe_dim is not None and args.pe_dim < 1:
            raise ValueError("--pe-dim must be positive")
        if args.layers is not None and args.layers < 1:
            raise ValueError("--layers must be positive")
        if args.learning_rate is not None and args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        if args.weight_decay is not None and args.weight_decay < 0:
            raise ValueError("--weight-decay must be non-negative")
        if args.brec_num_relabel < 2:
            raise ValueError("--brec-num-relabel must be at least 2")
        if args.brec_threshold is not None and args.brec_threshold <= 0:
            raise ValueError("--brec-threshold must be positive")
        args.data_root = args.data_root.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().resolve()
        device = resolve_device(args.device)
        runners = {
            "core": run_core,
            "brec": run_brec,
            "zinc": run_zinc,
        }
        selected = tuple(runners) if args.suite == "all" else (args.suite,)
        _claim_empty_output(args.output_dir)
        output_owned = True
        started_utc = datetime.now(UTC).isoformat()
        _write_json(
            args.output_dir / "run_manifest.json",
            {
                "schema_version": PAPER_SCHEMA_VERSION,
                "status": "running",
                "started_utc": started_utc,
                "selected_suites": list(selected),
                "completed_suites": [],
                "seed_axes": seed_axes.to_manifest(),
                "cli_arguments": _argument_manifest(args),
            },
        )
        manifests: dict[str, dict[str, Any]] = {}
        for suite in selected:
            manifests[suite] = runners[suite](args, device)
            completed.append(suite)
            _write_json(
                args.output_dir / "run_manifest.json",
                {
                    "schema_version": PAPER_SCHEMA_VERSION,
                    "status": "running",
                    "started_utc": started_utc,
                    "selected_suites": list(selected),
                    "completed_suites": completed,
                    "seed_axes": seed_axes.to_manifest(),
                    "cli_arguments": _argument_manifest(args),
                },
            )
        suite_manifests = {
            suite: {
                "path": str(args.output_dir / suite / "manifest.json"),
                "sha256": sha256_file(args.output_dir / suite / "manifest.json"),
            }
            for suite in completed
            if (args.output_dir / suite / "manifest.json").is_file()
        }
        _write_json(
            args.output_dir / "run_manifest.json",
            {
                "schema_version": PAPER_SCHEMA_VERSION,
                "status": "complete",
                "started_utc": started_utc,
                "completed_utc": datetime.now(UTC).isoformat(),
                "selected_suites": list(selected),
                "completed_suites": completed,
                "suite_manifests": suite_manifests,
                "seed_axes": seed_axes.to_manifest(),
                "cli_arguments": _argument_manifest(args),
            },
        )
    except BaseException as exc:
        failure_observations = resource_failure_observations(exc)
        if output_owned:
            failed_suite = selected[len(completed)] if len(completed) < len(selected) else None
            preserved_failed_output = _preserve_failed_suite_output(
                args.output_dir,
                failed_suite,
                reason=f"{type(exc).__name__}: {exc}",
            )
            completed_manifests = {
                suite: {
                    "path": str(args.output_dir / suite / "manifest.json"),
                    "sha256": sha256_file(args.output_dir / suite / "manifest.json"),
                }
                for suite in completed
                if (args.output_dir / suite / "manifest.json").is_file()
            }
            _write_json(
                args.output_dir / "run_manifest.json",
                {
                    "schema_version": PAPER_SCHEMA_VERSION,
                    "status": "failed",
                    "failed_utc": datetime.now(UTC).isoformat(),
                    "selected_suites": list(selected),
                    "completed_suites": completed,
                    "suite_manifests": completed_manifests,
                    "failed_suite": failed_suite,
                    "preserved_failed_suite_output": preserved_failed_output,
                    "seed_axes": seed_axes.to_manifest(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "resource_failure_observations": failure_observations,
                },
            )
        if isinstance(exc, Exception) and not failure_observations:
            parser.error(str(exc))
        raise
    summary = {
        suite: {
            "manifest": str(args.output_dir / suite / "manifest.json"),
            "variants": list(manifest.get("variants", {})),
        }
        for suite, manifest in manifests.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BREC_OFFICIAL_BATCH_SIZE",
    "BREC_OFFICIAL_SEEDS",
    "BREC_PROTOCOLS",
    "COMMAND_CONTRACT",
    "brec_hotelling_t2",
    "brec_rpc_decision",
    "build_parser",
    "main",
    "run_brec",
    "run_core",
    "run_zinc",
]
