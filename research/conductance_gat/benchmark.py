"""Train only our conductance model on official datasets used by GAT/GATv2.

Published competitor results are external references, not locally rerun models.
Dataset overlap does not imply identical architectures, tuning or table protocols.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chartgat.cache import atomic_publish, atomic_write_json
from chartgat.execution import add_execution_arguments, configure_execution
from chartgat.observability import RuntimeResourceMonitor, observed

from .benchmark_data import DATASETS, load_dataset, sha256_file
from .sparse import SparsePositiveConductance

PROTOCOL_NOTE = (
    "Only our conductance model is trained, on official datasets/splits used by prior "
    "papers. Competitor table values must be compared externally with their complete "
    "protocols, not presented as local reproductions. Our ogbn-arxiv training is "
    "full-batch, unlike GATv2's GraphSAINT setup. No Cycle PE or tree augmentation."
)


class ConductanceConv(nn.Module):
    """Positive orientation-invariant C with stable sparse H - eta B.T C B H."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.estimator = SparsePositiveConductance(channels, 0, channels, mode="full")

    def forward(
        self,
        x: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int | None = None,
    ) -> Tensor:
        # Tensor-only callers retain the old API. The classifier supplies CPU-side
        # graph-count metadata so its normal GPU path never synchronizes for max().
        if num_graphs is None:
            num_graphs = int(node_graph.max()) + 1
        # Computing the positive edge law and degree cap in fp32 avoids fp16 squares/overflow.
        with torch.autocast(device_type=x.device.type, enabled=False):
            state = x.float()
            tail, head = incidence
            gradient = state[head] - state[tail]
            c = self.estimator(gradient, state.new_empty((gradient.shape[0], 0)))
            flux = c[:, None] * gradient
            divergence = torch.zeros_like(state)
            divergence.index_add_(0, head, flux)
            divergence.index_add_(0, tail, -flux)
            degree = state.new_zeros(state.shape[0])
            degree.index_add_(0, head, c)
            degree.index_add_(0, tail, c)
            max_degree = state.new_zeros(num_graphs)
            max_degree.scatter_reduce_(0, node_graph, degree, reduce="amax", include_self=True)
            step = 0.95 / max_degree.clamp_min(1e-12)
            result = state - step[node_graph, None] * divergence
        return result.to(x.dtype)


class ConductanceNodeClassifier(nn.Module):
    """Our encoder/conductance-stack/prediction-head node classifier."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        *,
        hidden_channels: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_channels < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden width/layers must be positive and dropout in [0, 1)")
        self.dropout = dropout
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)
        self.operators = nn.ModuleList(ConductanceConv(hidden_channels) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(layers))

    def forward(self, graph: Any) -> Tensor:
        h = F.dropout(F.elu(self.encoder(graph.x)), self.dropout, self.training)
        node_graph = getattr(graph, "batch", None)
        if node_graph is None:
            node_graph = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
            num_graphs = 1
        else:
            ptr = getattr(graph, "ptr", None)
            if ptr is not None:
                num_graphs = ptr.numel() - 1
            else:
                num_graphs = getattr(graph, "num_graphs", None)
                if num_graphs is None:
                    # Compatibility for custom tensor containers without PyG metadata.
                    # Resolve once here, rather than once per conductance layer.
                    num_graphs = int(node_graph.max()) + 1
        for operator, norm in zip(self.operators, self.norms, strict=True):
            h = operator(h, graph.incidence_edge_index, node_graph, num_graphs)
            h = F.dropout(F.elu(norm(h)), self.dropout, self.training)
        return self.decoder(h)


def _binary_counts(logits: Tensor, labels: Tensor) -> Tensor:
    """Device-side counts, so graph minibatches need no scalar CPU transfers."""
    predicted, truth = logits > 0, labels > 0
    return torch.stack(((predicted & truth).sum(), predicted.sum(), truth.sum()))


def _micro_f1_from_counts(counts: Tensor) -> float:
    true_positive, predicted_count, truth_count = counts.tolist()
    denominator = predicted_count + truth_count
    return float(2 * true_positive / denominator) if denominator else 0.0


def micro_f1(logits: Tensor, labels: Tensor) -> float:
    """Global node-label micro-F1, not per-graph averaging or multiclass argmax."""
    return _micro_f1_from_counts(_binary_counts(logits, labels))


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _device(name: str, *, prepare_only: bool) -> torch.device:
    device = torch.device(name)
    if not prepare_only and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError(
            "Matched benchmark training requires a CUDA GPU; "
            "no CPU training/fallback is implemented."
        )
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.get_device_properties(device)
    return device


def _versions() -> dict[str, str]:
    output = {"torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda)}
    for package in ("torch-geometric", "ogb", "numpy"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "not_installed"
    return output


def _integer_distribution(values: list[int]) -> dict[str, int | float]:
    """Describe every observed integer without sampling or invented values."""

    if not values:
        raise ValueError("cannot summarize an empty integer observation")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        float(ordered[midpoint])
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "median": median,
        "maximum": max(values),
        "total": sum(values),
    }


def data_observability(
    payload: dict[str, Any],
    *,
    used_splits: tuple[str, ...],
    prepared_splits: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Report exact verified-payload scale while touching only requested split metadata."""

    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("verified payload must contain at least one graph")
    if any(not isinstance(graph, dict) for graph in graphs):
        raise ValueError("verified payload graph rows must be mappings")
    node_counts: list[int] = []
    edge_counts: list[int] = []
    feature_widths: set[int] = set()
    for graph in graphs:
        features = graph.get("x")
        incidence = graph.get("incidence_edge_index")
        if not isinstance(features, Tensor) or features.ndim != 2:
            raise ValueError("every verified graph requires a rank-two x tensor")
        if not isinstance(incidence, Tensor) or incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("every verified graph requires a [2, edges] incidence tensor")
        node_counts.append(int(features.shape[0]))
        edge_counts.append(int(incidence.shape[1]))
        feature_widths.add(int(features.shape[1]))
    if len(feature_widths) != 1:
        raise ValueError("verified graphs disagree on node-feature width")

    splits = payload.get("splits")
    if not isinstance(splits, dict):
        if not isinstance(prepared_splits, dict):
            raise ValueError("verified payload requires split metadata")
        splits = prepared_splits
        split_metadata_source = "prepared_verified_split_indices"
    else:
        split_metadata_source = "verified_payload"
    missing_splits = [name for name in used_splits if name not in splits]
    if missing_splits:
        raise ValueError(f"verified payload is missing requested splits: {missing_splits}")

    if payload.get("dataset") == "ppi":
        selected_graphs: set[int] = set()
        split_counts: dict[str, int] = {}
        for name in used_splits:
            values = splits[name]
            indices = [int(value) for value in values]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{name} graph split contains duplicate indices")
            if any(index < 0 or index >= len(graphs) for index in indices):
                raise ValueError(f"{name} graph split contains an out-of-range index")
            if selected_graphs.intersection(indices):
                raise ValueError("requested official graph splits overlap")
            selected_graphs.update(indices)
            split_counts[name] = len(indices)
        full_count = len(graphs)
        used_count = len(selected_graphs)
        requested_split_member_count = used_count
        dataset_unit = "graphs"
        target_shapes = sorted(
            {
                tuple(int(dimension) for dimension in graphs[index]["y"].shape)
                for index in selected_graphs
                if isinstance(graphs[index].get("y"), Tensor)
            }
        )
        input_shape_contract: Any = [
            "sum(nodes in physical graph batch)",
            next(iter(feature_widths)),
        ]
    else:
        if len(graphs) != 1:
            raise ValueError("transductive conductance data requires exactly one graph")
        full_count = node_counts[0]
        selected_nodes = torch.zeros(full_count, dtype=torch.bool)
        split_counts = {}
        for name in used_splits:
            mask = splits[name]
            if not isinstance(mask, Tensor):
                raise ValueError(f"{name} must be a tensor split mask or index vector")
            if mask.dtype == torch.bool and mask.numel() == full_count:
                mask = mask.detach().to(device="cpu").reshape(-1)
            elif mask.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                indices = mask.detach().to(device="cpu", dtype=torch.long).reshape(-1)
                if indices.numel() and (
                    bool((indices < 0).any()) or bool((indices >= full_count).any())
                ):
                    raise ValueError(f"{name} contains an out-of-range node index")
                if indices.numel() != indices.unique().numel():
                    raise ValueError(f"{name} contains duplicate node indices")
                mask = torch.zeros(full_count, dtype=torch.bool)
                mask[indices] = True
            else:
                raise ValueError(f"{name} must be a full-length bool mask or integer indices")
            if bool((selected_nodes & mask).any()):
                raise ValueError("requested official node splits overlap")
            selected_nodes |= mask
            split_counts[name] = int(mask.count_nonzero())
        requested_split_member_count = int(selected_nodes.count_nonzero())
        # Transductive message passing consumes the complete graph even though
        # loss/metric labels are restricted to the requested official splits.
        used_count = full_count
        dataset_unit = "nodes"
        target = graphs[0].get("y")
        target_shapes = (
            [tuple(int(dimension) for dimension in target.shape)]
            if isinstance(target, Tensor)
            else []
        )
        input_shape_contract = [node_counts[0], next(iter(feature_widths))]

    return {
        "official_graph_count": len(graphs),
        "nodes_per_graph": _integer_distribution(node_counts),
        "stored_edge_columns_per_graph": _integer_distribution(edge_counts),
        "input_tensor_shape_contract": {
            "node_features": input_shape_contract,
            "feature_width": next(iter(feature_widths)),
            "target_shapes_observed_for_used_splits": [list(shape) for shape in target_shapes],
            "first_actual_training_batch": observed(
                None,
                reason="captured from the first real training batch after training begins",
            ),
        },
        "full_dataset_unit": dataset_unit,
        "full_dataset_count": full_count,
        "actual_split_counts": split_counts,
        "requested_split_member_count": requested_split_member_count,
        "requested_split_member_fraction_of_full_dataset": observed(
            requested_split_member_count / full_count, unit="fraction"
        ),
        "split_metadata_source": split_metadata_source,
        "actual_used_count": used_count,
        "actual_used_fraction_of_full_dataset": observed(
            used_count / full_count, unit="fraction"
        ),
        "sampling_ratio": observed(1.0, unit="fraction"),
        "sampling_mode": "full_graph_no_neighbor_sampling",
        "subset_or_fast_mode": False,
        "time_window": observed(
            None, reason="not applicable to static graph benchmarks", unit="steps"
        ),
        "input_resolution": observed(
            None, reason="not applicable to graph feature tensors"
        ),
    }


def batch_observability(
    data: Any,
    *,
    transductive: bool,
    args: argparse.Namespace,
    batches_per_epoch: int,
) -> dict[str, Any]:
    """Describe the actual loader contract without running a probe batch."""

    if batches_per_epoch < 1:
        raise ValueError("training requires at least one batch per epoch")
    if transductive:
        physical_batch_size = 1
        batch_unit = "complete_transductive_graph"
        loader_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch = observed(
            None, reason="no DataLoader is used for full-graph transductive training"
        )
    else:
        loader = data["train"]
        physical_batch_size = int(getattr(loader, "batch_size", args.batch_size))
        batch_unit = "graphs"
        loader_workers = int(getattr(loader, "num_workers", args.workers))
        pin_memory = bool(getattr(loader, "pin_memory", True))
        persistent_workers = bool(getattr(loader, "persistent_workers", False))
        loader_prefetch = getattr(loader, "prefetch_factor", None)
        prefetch = (
            observed(int(loader_prefetch), unit="batches_per_worker")
            if loader_prefetch is not None
            else observed(
                None,
                reason="DataLoader prefetch is disabled because no worker process is used",
                unit="batches_per_worker",
            )
        )
    return {
        "batch_unit": batch_unit,
        "configured_physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "configured_effective_batch_size": physical_batch_size,
        "effective_batch_size_formula": (
            f"{physical_batch_size} physical x 1 accumulation x 1 data-parallel worker"
        ),
        "training_batches_per_epoch": batches_per_epoch,
        "planned_maximum_optimizer_steps": args.epochs * batches_per_epoch,
        "dataloader_workers": loader_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch,
        "non_blocking_cuda_transfer": not transductive and pin_memory,
        "cache": "verified official graph tensors are loaded once and reused",
        "physical_batch_candidate_measurements": observed(
            None,
            reason=(
                "this legacy comparison preserves its declared batch contract; no extra "
                "candidate-batch probe is inserted into the scientific run"
            ),
        ),
    }


def optimizer_ownership(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Fail if trainable model tensors are absent, duplicated, or replaced in the optimizer."""

    named = dict(model.named_parameters())
    model_parameter_ids = {id(parameter) for parameter in named.values()}
    trainable = {
        id(parameter): name
        for name, parameter in named.items()
        if parameter.requires_grad
    }
    owners: dict[int, list[int]] = {}
    unknown: list[str] = []
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            identifier = id(parameter)
            owners.setdefault(identifier, []).append(group_index)
            if identifier not in model_parameter_ids:
                unknown.append(f"group_{group_index}:shape={tuple(parameter.shape)}")
    missing = [name for identifier, name in trainable.items() if identifier not in owners]
    duplicated = [
        trainable.get(identifier, f"unrecognized_parameter_{identifier}")
        for identifier, groups in owners.items()
        if len(groups) != 1
    ]
    frozen_owned = [
        name
        for name, parameter in named.items()
        if not parameter.requires_grad and id(parameter) in owners
    ]
    if missing or duplicated or unknown or frozen_owned:
        raise RuntimeError(
            "optimizer ownership integrity failed: "
            f"missing={missing}, duplicated={duplicated}, unknown={unknown}, "
            f"frozen_owned={frozen_owned}"
        )
    return {
        "status": "passed",
        "trainable_parameter_tensors": len(trainable),
        "optimizer_owned_parameter_tensors": len(owners),
        "trainable_parameter_elements": sum(
            parameter.numel() for parameter in named.values() if parameter.requires_grad
        ),
    }


def validate_first_optimizer_step(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Fail closed before step one unless every trainable tensor has a finite gradient."""

    ownership = optimizer_ownership(model, optimizer)
    missing_gradients: list[str] = []
    nonfinite_gradients: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_gradients.append(name)
        elif not bool(torch.isfinite(parameter.grad.detach()).all()):
            nonfinite_gradients.append(name)
    if missing_gradients or nonfinite_gradients:
        raise FloatingPointError(
            "first optimizer step gradient integrity failed: "
            f"missing={missing_gradients}, nonfinite={nonfinite_gradients}"
        )
    return {
        **ownership,
        "gradient_status": "all_trainable_parameter_tensors_have_finite_gradients",
        "checked_before_optimizer_step": 1,
    }


def first_batch_shapes(graph: Any) -> dict[str, Any]:
    """Capture shapes from an actual training batch, never from a synthetic probe."""

    output: dict[str, Any] = {}
    for name in ("x", "y", "incidence_edge_index"):
        value = getattr(graph, name, None)
        if not isinstance(value, Tensor):
            raise ValueError(f"actual training batch is missing tensor field {name}")
        output[name] = list(value.shape)
    batch = getattr(graph, "batch", None)
    output["batch_vector"] = list(batch.shape) if isinstance(batch, Tensor) else None
    output["num_graphs"] = int(getattr(graph, "num_graphs", 1))
    return output


def pre_run_observability(
    *,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    data_observation: dict[str, Any],
    batching: dict[str, Any],
    resource_start: dict[str, Any],
    architecture: dict[str, Any],
    precision: str,
    device: torch.device,
) -> dict[str, Any]:
    """Build the common fail-visible pre-run contract for legacy conductance models."""

    parameters = optimizer_ownership(model, optimizer)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware: dict[str, Any] = {
            "selected_device": str(device),
            "device_name": properties.name,
            "visible_cuda_device_count": torch.cuda.device_count(),
            "device_total_memory_bytes": int(properties.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "mig_detected_from_device_name": "MIG" in properties.name.upper(),
            "used_cuda_devices": 1,
        }
    else:
        hardware = {
            "selected_device": str(device),
            "device_name": observed(
                None, reason="CPU is permitted only by bounded unit-test hardware mocks"
            ),
            "visible_cuda_device_count": observed(
                None, reason="CPU is permitted only by bounded unit-test hardware mocks"
            ),
            "used_cuda_devices": 0,
        }
    return {
        "status": "pre_run_configuration",
        "model": {
            "name": model_name,
            "layers": architecture["layers"],
            "hidden_dimension": architecture["hidden_channels"],
            "channels": architecture["hidden_channels"],
            "attention_heads": observed(
                None, reason="legacy conductance operators do not use attention heads"
            ),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": total_parameters - trainable_parameters,
            "optimizer_ownership": parameters,
            **{
                key: value
                for key, value in architecture.items()
                if key not in {"layers", "hidden_channels"}
            },
        },
        "data": data_observation,
        "batching": batching,
        "optimization": {
            "epochs_requested": args.epochs,
            "early_stopping_patience": args.patience,
            "planned_maximum_optimizer_steps": batching[
                "planned_maximum_optimizer_steps"
            ],
            "actual_optimizer_steps": observed(
                None, reason="training has not started", unit="steps"
            ),
        },
        "precision": precision,
        "hardware": hardware,
        "resources": resource_start,
        "modes": {
            "debug": False,
            "subset": False,
            "fast_mode": False,
        },
    }


def finish_monitor_after_failure(
    monitor: RuntimeResourceMonitor,
    *,
    output: Path,
    device: torch.device,
    primary_error: BaseException,
) -> None:
    """Stop and publish the monitor without ever replacing the training exception."""

    try:
        allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        resources = monitor.finish(
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
        )
        atomic_write_json(
            output / "resource_observability.failed.json",
            {
                "status": "failed",
                "training_error": f"{type(primary_error).__name__}: {primary_error}",
                "resource_observability": resources,
            },
        )
    except BaseException as cleanup_error:
        primary_error.add_note(
            "resource monitor cleanup also failed without replacing this error: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _selection(values: list[str], allowed: tuple[str, ...]) -> list[str]:
    selected = [
        item.strip().lower() for value in values for item in value.split(",") if item.strip()
    ]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(item not in allowed for item in selected)
    ):
        raise ValueError(f"Choose each supported value at most once from {allowed}")
    return selected


def _prepare_split_indices(splits: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    # Verified payload masks are CPU tensors. Find indices there once, rather than
    # repeating CUDA boolean indexing (and its dynamic-shape synchronization).
    return {
        name: mask.nonzero(as_tuple=False).flatten().to(device) for name, mask in splits.items()
    }


def _make_loaders(payload: dict[str, Any], args: argparse.Namespace, device: torch.device):
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    graphs = [Data(**graph) for graph in payload["graphs"]]
    validation_only = bool(getattr(args, "validation_only", False))
    selected_splits = ("train", "validation") if validation_only else tuple(payload["splits"])
    if payload["dataset"] != "ppi":
        # Full graph/features are visible transductively; ONLY training-mask labels enter loss.
        indices = _prepare_split_indices(
            {name: payload["splits"][name] for name in selected_splits}, device
        )
        return graphs[0].to(device), indices
    loaders = {}
    for split in selected_splits:
        indices = payload["splits"][split]
        generator = torch.Generator().manual_seed(args.model_seed)
        loaders[split] = DataLoader(
            [graphs[index] for index in indices],
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            generator=generator,
            pin_memory=args.pin_memory,
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )
    return loaders, None


def _train_model_impl(
    payload: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
    *,
    resource_start: dict[str, Any],
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Benchmark training requires CUDA (including direct train_model calls).")
    _seed(args.model_seed)
    data, split_indices = _make_loaders(payload, args, device)
    train_indices = None if split_indices is None else split_indices["train"]
    train_label_count = 0 if train_indices is None else train_indices.numel()
    model = ConductanceNodeClassifier(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        hidden_channels=args.hidden_channels,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    execution = configure_execution(model, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and amp_dtype == torch.float16)
    validation_only = bool(getattr(args, "validation_only", False))
    used_splits = ("train", "validation") if validation_only else (
        "train",
        "validation",
        "test",
    )
    batches_per_epoch = 1 if split_indices is not None else len(data["train"])
    data_observation = data_observability(payload, used_splits=used_splits)
    batching = batch_observability(
        data,
        transductive=split_indices is not None,
        args=args,
        batches_per_epoch=batches_per_epoch,
    )
    pre_run = pre_run_observability(
        model_name="conductance_gat_v1",
        model=model,
        optimizer=optimizer,
        args=args,
        data_observation=data_observation,
        batching=batching,
        resource_start=resource_start,
        architecture={
            "hidden_channels": args.hidden_channels,
            "layers": args.layers,
            "dropout": args.dropout,
        },
        precision=(
            f"amp_{str(amp_dtype).removeprefix('torch.')}" if args.amp else "float32"
        ),
        device=device,
    )
    print(json.dumps(pre_run, sort_keys=True), flush=True)
    checkpoint = output / "best.pt"
    history: list[dict[str, Any]] = []
    best_validation, best_epoch = -float("inf"), 0
    optimizer_steps = 0
    training_label_updates = 0
    first_training_batch: dict[str, Any] | None = None
    first_step_integrity: dict[str, Any] | None = None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    @torch.no_grad()
    def evaluate(split: str) -> float:
        model.eval()
        if split_indices is not None:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                logits = model(data)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite {split} logits: {payload['dataset']}/conductance")
            indices = split_indices[split]
            predicted = logits.index_select(0, indices).argmax(dim=-1)
            truth = data.y.index_select(0, indices)
            return float((predicted == truth).float().mean())
        counts = torch.zeros(3, dtype=torch.int64, device=device)
        for graph in data[split]:
            graph = graph.to(device, non_blocking=args.pin_memory)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                logits = model(graph)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite {split} logits: {payload['dataset']}/conductance")
            counts.add_(_binary_counts(logits, graph.y))
        return _micro_f1_from_counts(counts)

    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        label_count = 0
        batches = [data] if split_indices is not None else data["train"]
        for graph in batches:
            if split_indices is None:
                graph = graph.to(device, non_blocking=args.pin_memory)
            if first_training_batch is None:
                first_training_batch = first_batch_shapes(graph)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                logits = model(graph)
                if train_indices is not None:
                    loss = F.cross_entropy(
                        logits.index_select(0, train_indices),
                        graph.y.index_select(0, train_indices),
                    )
                    count = train_label_count
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, graph.y)
                    count = graph.y.numel()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss: {payload['dataset']}/conductance, epoch {epoch}"
                )
            scaler.scale(loss).backward()
            if optimizer_steps == 0:
                scaler.unscale_(optimizer)
                first_step_integrity = validate_first_optimizer_step(model, optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            training_label_updates += int(count)
            loss_sum.add_(loss.detach().to(torch.float64) * count)
            label_count += count
        validation = evaluate("validation")
        train_loss = float(loss_sum / label_count)
        # Synchronized train+validation wall time; checkpoint/history I/O is excluded.
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation,
                "epoch_seconds": epoch_seconds,
            }
        )
        atomic_write_json(output / "history.json", history)
        if validation > best_validation:
            best_validation, best_epoch = validation, epoch
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            checkpoint_data = {
                "state_dict": state,
                "best_epoch": epoch,
                "validation": validation,
                "dataset": payload["dataset"],
                "model": "conductance",
                "architecture": {
                    "hidden_channels": args.hidden_channels,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
            }
            atomic_publish(checkpoint, lambda path, saved=checkpoint_data: torch.save(saved, path))
        if epoch == 1 or epoch % 10 == 0:
            metric_name = "micro_f1" if payload["dataset"] == "ppi" else "accuracy"
            print(
                f"{payload['dataset']}/conductance epoch={epoch} "
                f"train_loss={train_loss:.6f} val_{metric_name}={validation:.6f} "
                f"epoch_seconds={epoch_seconds:.3f}",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    # Scaling/model-size exploration must not repeatedly expose test labels. The
    # historical benchmark path keeps its one post-selection test evaluation.
    test_metric = None if validation_only else evaluate("test")
    if first_training_batch is None or first_step_integrity is None:
        raise RuntimeError("training completed without a validated first optimizer step")
    elapsed_seconds = time.perf_counter() - start_time
    measured_epoch_seconds = sum(float(row["epoch_seconds"]) for row in history)
    result = {
        "validation": best_validation,
        "test": test_metric,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint.resolve()),
        "history": str((output / "history.json").resolve()),
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "amp_dtype": str(amp_dtype) if args.amp else "float32",
        "training": "full_batch"
        if split_indices is not None
        else "official_inductive_graph_minibatch",
        "execution": execution,
        "epoch_timing": "cuda_synchronized_train_and_validation_excluding_checkpoint_io",
        "model_seed": args.model_seed,
        "optimizer_steps": optimizer_steps,
        "optimization_observability": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "early_stopping_patience": args.patience,
            "planned_maximum_optimizer_steps": batching[
                "planned_maximum_optimizer_steps"
            ],
            "actual_optimizer_steps": optimizer_steps,
            "training_label_decisions_processed": training_label_updates,
            "gradient_accumulation_steps": 1,
        },
        "data_observability": {
            **data_observation,
            "input_tensor_shape_contract": {
                **data_observation["input_tensor_shape_contract"],
                "first_actual_training_batch": first_training_batch,
            },
        },
        "batch_observability": batching,
        "pre_run_observability": pre_run,
        "first_optimizer_step_integrity": first_step_integrity,
        "throughput": {
            "measured_epoch_seconds": measured_epoch_seconds,
            "optimizer_steps_per_second": optimizer_steps / measured_epoch_seconds,
            "training_label_decisions_per_second": (
                training_label_updates / measured_epoch_seconds
            ),
            "scope": (
                "CUDA-synchronized epoch timing includes training and validation but excludes "
                "subsequent history/checkpoint writes"
            ),
        },
        "test_selection": (
            "not_evaluated_scaling_selection"
            if validation_only
            else "best_validation_checkpoint_only"
        ),
    }
    if validation_only:
        result.pop("test")
        result.update(evaluation_split="validation", test_evaluated=False)
    atomic_write_json(output / "metrics.json", result)
    return result


def train_model(
    payload: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    """Run V1 with periodic resource sampling and exception-safe monitor cleanup."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Benchmark training requires CUDA (including direct train_model calls).")
    torch.cuda.get_device_properties(device)
    monitor = RuntimeResourceMonitor(device)
    resource_start = monitor.start()
    try:
        result = _train_model_impl(
            payload,
            args,
            device,
            output,
            resource_start=resource_start,
        )
    except BaseException as primary_error:
        finish_monitor_after_failure(
            monitor,
            output=output,
            device=device,
            primary_error=primary_error,
        )
        raise
    resources = monitor.finish(
        peak_allocated_bytes=int(result["peak_cuda_allocated_bytes"]),
        peak_reserved_bytes=int(result["peak_cuda_reserved_bytes"]),
    )
    result["resource_observability"] = resources
    result["hardware_execution"] = {
        "gpu_sm_utilization": resources["interval_series"][
            "gpu_sm_utilization_percent"
        ],
        "gpu_memory_controller_utilization": resources["interval_series"][
            "gpu_memory_controller_utilization_percent"
        ],
        "cpu_and_ram": {
            "cpu": resources["summary"],
            "ram": {
                key: value
                for key, value in resources["interval_series"].items()
                if key
                in {
                    "process_resident_bytes",
                    "process_peak_resident_bytes",
                    "system_available_bytes",
                }
            },
        },
    }
    atomic_write_json(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("benchmark",), default="benchmark")
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/conductance_gat/benchmark")
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--chart-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--workers",
        "--num-workers",
        type=int,
        default=None,
        help="Default: 4 for PPI graph minibatches, 0 when only transductive graphs run",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    add_execution_arguments(parser)
    return parser


def resolve_worker_arguments(args: argparse.Namespace) -> None:
    """Resolve the only real DataLoader worker pool from the selected datasets."""

    if args.workers is None:
        args.workers = 4 if "ppi" in args.datasets else 0
        args.worker_configuration_source = "dataset_default"
    elif not hasattr(args, "worker_configuration_source"):
        args.worker_configuration_source = "explicit_cli"
    args.workers_by_dataset = {
        dataset: args.workers if dataset == "ppi" else 0 for dataset in args.datasets
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.datasets = _selection(args.datasets, DATASETS)
    resolve_worker_arguments(args)
    if min(args.batch_size, args.epochs, args.patience, args.layers) < 1 or args.workers < 0:
        raise ValueError(
            "batch size, epochs, patience, layers must be positive; workers nonnegative"
        )
    if args.hidden_channels < 1 or not 0 <= args.dropout < 1:
        raise ValueError("invalid hidden width/dropout")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay nonnegative")
    if min(args.data_seed, args.split_seed, args.chart_seed, args.model_seed) < 0:
        raise ValueError("seed values must be nonnegative")
    device = _device(args.device, prepare_only=args.prepare_only)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "track": "conductance_gat",
        "suite": "benchmark",
        "status": "running",
        "protocol_note": PROTOCOL_NOTE,
        "config": config,
        "versions": _versions(),
        "seed_axes": {
            "model_seed": args.model_seed,
            "data_seed": "not_applicable: fixed official source data",
            "split_seed": "not_applicable: official fixed masks/splits",
            "chart_seed": "not_applicable: no chart/PE/augmentation",
        },
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
        "completed": [],
        "expected": [f"{dataset}/conductance" for dataset in args.datasets],
        "sources": ["https://arxiv.org/abs/1710.10903", "https://arxiv.org/abs/2105.14491"],
        "implementation_sha256": {
            **{
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("benchmark.py", "benchmark_data.py", "sparse.py")
            },
            "src/chartgat/observability.py": sha256_file(
                Path(__file__).resolve().parents[2] / "src/chartgat/observability.py"
            ),
        },
        "reproducibility": (
            "Seeded runs; GPU scatter kernels can remain nondeterministic. No bitwise guarantee."
        ),
    }
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "track": "conductance_gat",
        "suite": "benchmark",
        "status": "running",
        "model_seed": args.model_seed,
        "datasets": {},
    }
    atomic_write_json(output / "manifest.json", manifest)
    try:
        for dataset in args.datasets:
            print(f"Loading official matched dataset: {dataset}", flush=True)
            payload, protocol = load_dataset(
                dataset, args.data_root, allow_download=args.allow_download
            )
            record: dict[str, Any] = {
                "metric": protocol["metric"],
                "protocol": protocol,
                "models": {},
            }
            metrics["datasets"][dataset] = record
            if args.prepare_only:
                continue
            record["models"]["conductance"] = train_model(
                payload, args, device, output / dataset / "conductance"
            )
            manifest["completed"].append(f"{dataset}/conductance")
            atomic_write_json(output / "metrics.json", metrics)
            atomic_write_json(output / "manifest.json", manifest)
            torch.cuda.empty_cache()
        if not args.prepare_only and manifest["completed"] != manifest["expected"]:
            raise RuntimeError("Incomplete matched benchmark; cannot mark passed")
        manifest["status"] = metrics["status"] = "prepared" if args.prepare_only else "passed"
    except BaseException as exc:
        manifest["status"] = metrics["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        for path, payload in (
            (output / "manifest.json", manifest),
            (output / "metrics.json", metrics),
        ):
            try:
                atomic_write_json(path, payload)
            except BaseException as reporting_error:
                exc.add_note(
                    f"{path.name} failure state could not be written without replacing this "
                    f"error: {type(reporting_error).__name__}: {reporting_error}"
                )
        raise
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(output / "metrics.json", metrics)
    print(f"{manifest['status']}: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
