"""Standalone four-arm conductance x spatial-message V4 training.

Every arm is trained freshly on CUDA with the same official full-graph cache.
Validation alone selects a checkpoint.  Selected-checkpoint interventions are
read-only diagnostic forwards and are never counted as additional training.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch

from chartgat.cache import atomic_publish, atomic_write_json

from ..ablation.model import state_sha256
from ..ablation.train import _configure_fp32, _make_data, _require_cuda, training_loss
from ..benchmark import _seed, _versions
from ..benchmark_data import load_dataset, sha256_file, tensor_hash
from .diagnostics import (
    ForwardObservation,
    best_checkpoint_interventions,
    evaluate_validation,
    norm,
)
from .model import RelativeCSpatialNodeClassifier
from .protocol import COMMON, CONDITIONS, DATASETS, DEFAULT_EDGE_CHUNK_SIZE, PARAMETERIZATION, SUITE

OBSERVATION_POLICY = {
    "training": (
        "Every epoch's actual full-graph train forward with dropout ON; raw task gradients "
        "after backward before AdamW step. No extra training forward/backward."
    ),
    "validation": (
        "Every epoch validation selects checkpoint; initial/selected/final validation layer "
        "observations use validation labels only."
    ),
    "statistics": (
        "Exact full observed graph score/C moments and degree population quantiles; per-layer "
        "alpha and W parameter, identity-distance, gradient, and singular-value summaries."
    ),
    "factorial_training": (
        "Four independently trained arms cross fixed/relative C with fixed-identity/learned W. "
        "Inactive scaffolds are frozen and excluded from AdamW; alpha is active in every arm."
    ),
    "symmetric_normalization": (
        "neighbor_weight_row_sum is alpha times symmetric off-diagonal row weight sum, not a "
        "stochastic mixing probability and may exceed 1."
    ),
    "interventions": (
        "Only the selected best checkpoint: all-layer graph-mean C, graph-shuffled C, C=1, "
        "W=I, C=1 plus W=I, and propagation off. C replacements recompute normalization. "
        "Mean-C versus C=1 is a numerical cancellation check, not an independent effect."
    ),
}


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **COMMON,
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "tf32": False,
        "pin_memory": True,
        "edge_chunk_size": args.edge_chunk_size,
    }


def make_optimizer(model, condition: str) -> torch.optim.AdamW:
    specification = CONDITIONS[condition]
    if (
        model.gate_mode != specification["gate_mode"]
        or model.spatial_mode != specification["spatial_mode"]
        or model.normalization != "symmetric"
    ):
        raise ValueError("V4 model and factorial condition disagree")
    selected = {
        "backbone": [],
        "conductance_gate": [],
        "spatial_w": [],
        "raw_scalars": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith((".raw_alpha", ".raw_gamma", ".raw_tau")):
            group = "raw_scalars"
        elif name.startswith("operators.") and ".estimator." in name:
            group = "conductance_gate"
        elif name.startswith("operators.") and ".message_transform." in name:
            group = "spatial_w"
        else:
            group = "backbone"
        selected[group].append((name, parameter))
    if not selected["backbone"] or not selected["raw_scalars"]:
        raise ValueError("V4 requires backbone and trainable alpha/scalar groups")
    if bool(selected["conductance_gate"]) != (model.gate_mode == "relative"):
        raise ValueError("V4 fixed estimator must be entirely frozen and excluded")
    if bool(selected["spatial_w"]) != (model.spatial_mode == "learned"):
        raise ValueError("V4 fixed identity W must be entirely frozen and excluded")
    groups = []
    for name, values in selected.items():
        if not values:
            continue
        if name == "conductance_gate":
            learning_rate = COMMON["lr"] * COMMON["gate_lr_multiplier"]
            weight_decay = specification["gate_weight_decay"]
        elif name == "raw_scalars":
            learning_rate = COMMON["lr"]
            weight_decay = COMMON["scalar_weight_decay"]
        elif name in {"backbone", "spatial_w"}:
            learning_rate = COMMON["lr"]
            weight_decay = COMMON["weight_decay"]
        else:  # pragma: no cover - fail closed if the grouping table changes.
            raise AssertionError(name)
        groups.append(
            {
                "name": name,
                "params": [value for _, value in values],
                "parameter_names": [key for key, _ in values],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(groups, lr=COMMON["lr"])
    optimized = {name for group in optimizer.param_groups for name in group["parameter_names"]}
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    if optimized != trainable:
        raise RuntimeError("V4 optimizer does not contain each trainable parameter exactly once")
    return optimizer


def optimizer_metadata(optimizer):
    return [
        {
            "name": group["name"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameter_names": list(group["parameter_names"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]


def epoch_timing_summary(history):
    values = [float(row["epoch_seconds"]) for row in history]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("V4 epoch timing history must be nonempty, finite, and nonnegative")
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": sum(values) / len(values),
        "median_seconds": quantile(0.5),
        "p90_seconds": quantile(0.9),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "quantile_method": "linear_order_statistic",
        "scope": (
            "Per epoch: actual train forward/backward/step, diagnostic summaries, validation "
            "selection, and CUDA synchronization; subsequent history/checkpoint IO is "
            "excluded from epoch_seconds and included in selection_loop_seconds."
        ),
    }


def _parameter_metadata(model):
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "trainable_parameter_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "frozen_parameter_names": [
            name for name, parameter in model.named_parameters() if not parameter.requires_grad
        ],
    }


def topology_metadata(payload):
    if payload.get("dataset") not in DATASETS or len(payload.get("graphs", [])) != 1:
        raise ValueError("V4 experiment requires one official transductive graph; PPI unsupported")
    graph = payload["graphs"][0]
    return {
        "num_nodes": int(graph["x"].shape[0]),
        "num_edges": int(graph["incidence_edge_index"].shape[1]),
        "incidence_sha256": tensor_hash(graph["incidence_edge_index"]),
    }


def _source_hashes():
    from scripts.run_conductance_v4 import _source_snapshot

    return _source_snapshot()["sha256"]


def _require_sources(expected):
    if _source_hashes() != expected:
        raise RuntimeError("V4 sources changed during execution; refusing mixed sources")


def _validate_args(args):
    if args.dataset not in DATASETS or args.condition not in CONDITIONS:
        raise ValueError("Unsupported V4 dataset/condition")
    if min(args.epochs, args.patience, args.edge_chunk_size) < 1 or args.model_seed < 0:
        raise ValueError("epochs/patience/chunk size must be positive and seed nonnegative")
    if args.batch_size != 1 or args.workers != 0:
        raise ValueError("V4 full-graph training requires batch-size=1 and workers=0")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--edge-chunk-size", type=int, default=DEFAULT_EDGE_CHUNK_SIZE)
    return parser


def train_model(payload, protocol, args, device: torch.device, output: Path):
    _require_cuda(device)
    _validate_args(args)
    if payload.get("dataset") != args.dataset:
        raise ValueError("Requested dataset does not match payload")
    topology = topology_metadata(payload)
    sources = _source_hashes()
    _configure_fp32()
    _seed(args.model_seed)
    graph, indices = _make_data(payload, args, device)
    if indices is None or not indices["train"].numel():
        raise ValueError("V4 requires a nonempty transductive train mask")
    specification = CONDITIONS[args.condition]
    model = RelativeCSpatialNodeClassifier(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        hidden_channels=COMMON["hidden_channels"],
        layers=COMMON["layers"],
        dropout=COMMON["dropout"],
        normalization="symmetric",
        gate_mode=specification["gate_mode"],
        spatial_mode=specification["spatial_mode"],
        edge_chunk_size=args.edge_chunk_size,
    ).to(device)
    initial_hash = state_sha256(model)
    optimizer = make_optimizer(model, args.condition)
    common = {
        "schema_version": 1,
        "research_suite": SUITE,
        "model": SUITE,
        "dataset": args.dataset,
        "condition": args.condition,
        "model_seed": args.model_seed,
        **specification,
        "factorial_axes": {
            "conductance": specification["gate_mode"],
            "spatial_message": specification["spatial_mode"],
        },
        "non_gate_weight_decay": COMMON["weight_decay"],
        "configuration": configuration(args),
        "cache_sha256": protocol["data_sha256"],
        "protocol": protocol,
        "initial_state_sha256": initial_hash,
        "topology": topology,
        "parameterization": PARAMETERIZATION,
        "source_sha256": sources,
        "optimizer": "AdamW",
        "optimizer_groups": optimizer_metadata(optimizer),
        **_parameter_metadata(model),
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    checkpoint, history_path = output / "best.pt", output / "history.json"
    history, trajectory = [], []
    best_validation, best_epoch = -math.inf, 0
    checkpoint_hash = history_hash = None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    initial_observation, _ = evaluate_validation(model, graph, indices["validation"])
    for epoch in range(1, args.epochs + 1):
        _require_sources(sources)
        torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with ForwardObservation(model) as observation:
            logits = model(graph)
        loss, count = training_loss(logits, graph, indices["train"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite V4 training loss at epoch {epoch}")
        loss.backward()
        gradient_groups = [
            {
                **descriptor,
                "parameter_norm": norm(group["params"]),
                "task_gradient_norm": norm(group["params"], gradient=True),
            }
            for descriptor, group in zip(
                optimizer_metadata(optimizer), optimizer.param_groups, strict=True
            )
        ]
        record = {
            "epoch": epoch,
            "batch_index": 0,
            "optimizer_steps_before_batch": epoch - 1,
            "scope": "full_graph_train_mask",
            "mode": "train_dropout_on",
            "stage": "after_task_backward_before_optimizer_step",
            "label_count": count,
            "train_loss": float(loss.detach()),
            "layers": observation.summary(gradients=True),
            "parameter_groups": gradient_groups,
        }
        trajectory.append(record)
        optimizer.step()
        validation_observation, _ = evaluate_validation(
            model, graph, indices["validation"], observe=False
        )
        validation = validation_observation["metric"]
        torch.cuda.synchronize(device)
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": epoch,
                "train_loss": record["train_loss"],
                "validation": validation,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "training_first_batch": record,
            }
        )
        atomic_write_json(history_path, history)
        history_hash = sha256_file(history_path)
        if validation > best_validation:
            best_validation, best_epoch = validation, epoch
            saved = {
                **common,
                "state_dict": {
                    name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                },
                "architecture": {
                    "hidden_channels": COMMON["hidden_channels"],
                    "layers": COMMON["layers"],
                    "dropout": COMMON["dropout"],
                    "normalization": "symmetric",
                    "gate_mode": specification["gate_mode"],
                    "spatial_mode": specification["spatial_mode"],
                    "edge_chunk_size": args.edge_chunk_size,
                },
                "best_epoch": epoch,
                "optimizer_steps": epoch,
                "validation": validation,
            }
            atomic_publish(checkpoint, lambda path, state=saved: torch.save(state, path))
            checkpoint_hash = sha256_file(checkpoint)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{args.dataset}/{args.condition} epoch={epoch} "
                f"train_loss={record['train_loss']:.6f} "
                f"val={validation:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            break
    stop_epoch = len(history)
    stopping_reason = "patience" if stop_epoch - best_epoch >= args.patience else "max_epochs"
    selection_loop_seconds = time.perf_counter() - started
    post_selection_started = time.perf_counter()
    final_observation, _ = evaluate_validation(model, graph, indices["validation"])
    _require_sources(sources)
    if sha256_file(checkpoint) != checkpoint_hash or sha256_file(history_path) != history_hash:
        raise RuntimeError("Checkpoint/history changed before best-checkpoint validation")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    for key in (
        "research_suite",
        "condition",
        "dataset",
        "configuration",
        "topology",
        "source_sha256",
        "initial_state_sha256",
        "gate_mode",
        "spatial_mode",
    ):
        if saved.get(key) != common[key]:
            raise ValueError(f"Best checkpoint metadata mismatch: {key}")
    model.load_state_dict(saved["state_dict"])
    optimizer.zero_grad(set_to_none=True)
    selected_observation, reference = evaluate_validation(model, graph, indices["validation"])
    if abs(selected_observation["metric"] - best_validation) > 1e-4:
        raise RuntimeError("Best validation recheck disagrees with checkpoint selection")
    interventions = best_checkpoint_interventions(
        model,
        graph,
        indices["validation"],
        selected_observation,
        reference,
        seed=args.model_seed,
    )
    _require_sources(sources)
    if sha256_file(checkpoint) != checkpoint_hash or sha256_file(history_path) != history_hash:
        raise RuntimeError("Read-only interventions changed source checkpoint/history")
    torch.cuda.synchronize(device)
    post_selection_diagnostics_seconds = time.perf_counter() - post_selection_started
    return {
        **common,
        "status": "passed",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "epochs_run": len(history),
        "optimizer_steps": len(history),
        "best_checkpoint_optimizer_steps": best_epoch,
        "validation": selected_observation["metric"],
        "validation_at_selection": best_validation,
        "metric_name": "accuracy",
        "train_loss": history[best_epoch - 1]["train_loss"],
        "train_loss_scope": "actual full-graph train mask loss at selected checkpoint epoch",
        "final_train_loss": history[-1]["train_loss"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "history": str(history_path.resolve()),
        "history_sha256": history_hash,
        "selection_loop_seconds": selection_loop_seconds,
        "post_selection_diagnostics_seconds": post_selection_diagnostics_seconds,
        "epoch_timing": epoch_timing_summary(history),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "versions": _versions(),
        "gpu": torch.cuda.get_device_name(device),
        "diagnostics": {
            "initial_validation": initial_observation,
            "best_validation": selected_observation,
            "final_validation": final_observation,
            "train_trajectory": trajectory,
            "best_checkpoint_interventions": interventions,
            "observation_policy": OBSERVATION_POLICY,
        },
        "execution": {
            "training": "full_graph_transductive",
            "neighbor_sampling": False,
            "edge_chunk_size": args.edge_chunk_size,
            "dense_incidence": False,
            "eigendecomposition": False,
            "spatial_message_transform": specification["spatial_mode"],
        },
        "reproducibility": "Same initialization/seed; CUDA scatter may remain nondeterministic.",
        "timing_policy": (
            "selection_loop_seconds includes initial validation, training, validation selection "
            "and history/checkpoint IO. post_selection_diagnostics_seconds includes final and "
            "selected-checkpoint observations/interventions plus integrity checks and checkpoint "
            "loading. elapsed_seconds includes both."
        ),
    }


def _cache_snapshot(args):
    cache = (
        args.data_root.expanduser().resolve()
        / "conductance_gat/matched_benchmark_v1"
        / args.dataset
    )
    return {str(path): sha256_file(path) for path in (cache / "data.pt", cache / "manifest.json")}


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device = torch.device(args.device)
    _require_cuda(device)
    output, data_root = (
        args.output_dir.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
    )
    if output == data_root or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("V4 output and dataset cache must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is not a new empty arm directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "running",
        "research_suite": SUITE,
        "dataset": args.dataset,
        "condition": args.condition,
        "model_seed": args.model_seed,
        **CONDITIONS[args.condition],
        "configuration": configuration(args),
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    atomic_write_json(output / "metrics.json", record)
    try:
        sources = _source_hashes()
        record["source_sha256"] = sources
        payload, protocol = load_dataset(args.dataset, data_root, allow_download=False)
        cache_files = _cache_snapshot(args)
        _require_sources(sources)
        record.update(cache_sha256=protocol["data_sha256"], protocol=protocol)
        result = train_model(payload, protocol, args, device, output)
        if result["source_sha256"] != sources or _cache_snapshot(args) != cache_files:
            raise RuntimeError("Cache/source changed between verification and completed training")
        _require_sources(sources)
        record.update(result, cache_files_sha256=cache_files)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output / "metrics.json", record)
        raise
    atomic_write_json(output / "metrics.json", record)
    print(f"passed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
