"""Linux/CUDA paper runner for the independent conductance-GAT track.

Examples
--------
python -m research.conductance_gat.paper --suite core --data-root ./data \
    --output-dir ./results/conductance --device cuda --seed 17
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import platform
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader

from chartgat.seeds import SeedAxes, resolve_seed_axes

from .paper_data import nonlinear_conductance, prepare_core_cache
from .public_data import prepare_public_data
from .sparse import (
    PackedGraphBatch,
    SparseIncidenceConductanceLayer,
    edge_gradient,
    pack_graph_examples,
)

CORE_CLAIMS = {
    "s1": "Static shared conductance law generalizes to held-out graph identities.",
    "s2": "The law transfers from ER/RGG n=16..32 to larger grid/barbell graphs.",
    "s3": "State-dependent positive conductance supports stable held-graph rollout.",
    "s4": "Identification limits are mapped across contrast, excitation coverage, and SNR.",
}
TRAINING_OBJECTIVES = {"node_only", "flux_only", "joint"}


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {normalized!r} was requested but this PyTorch build cannot use CUDA"
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def runtime_metadata(
    device: torch.device, *, amp: bool, pin_memory: bool, batch_size: int
) -> dict[str, Any]:
    cuda = device.type == "cuda"
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "amp": bool(amp),
        "pin_memory": bool(pin_memory),
        "batch_size": int(batch_size),
        "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
    }
    if cuda:
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_total_memory_bytes": int(properties.total_memory),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    else:
        metadata.update({"cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0})
    return metadata


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(
    examples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=pack_graph_examples,
    )


def _normalized_loss(
    model: SparseIncidenceConductanceLayer,
    batch: PackedGraphBatch,
    *,
    objective: str,
) -> tuple[Tensor, dict[str, float | None]]:
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"unknown training objective {objective!r}")
    _, diagnostics = model(batch, return_diagnostics=True)
    flux_target = None
    if objective in {"flux_only", "joint"}:
        flux_target = batch.observed_flux if batch.observed_flux is not None else batch.true_flux
    node_target = None
    if objective in {"node_only", "joint"}:
        node_target = (
            batch.observed_node_message
            if batch.observed_node_message is not None
            else batch.true_node_message
        )
    epsilon = torch.finfo(diagnostics["edge_flux"].dtype).eps
    flux_relative = None
    if flux_target is not None:
        flux_mse = (diagnostics["edge_flux"] - flux_target).square().mean()
        flux_scale = flux_target.square().mean().clamp_min(epsilon)
        flux_relative = flux_mse / flux_scale
    node_relative = None
    if node_target is not None:
        node_mse = (diagnostics["node_message"] - node_target).square().mean()
        node_scale = node_target.square().mean().clamp_min(epsilon)
        node_relative = node_mse / node_scale
    if objective == "node_only":
        if node_relative is None:
            raise ValueError("node_only training requires a node-message target")
        loss = node_relative
    elif objective == "flux_only":
        if flux_relative is None:
            raise ValueError("flux_only training requires an edge-flux target")
        loss = flux_relative
    else:
        if flux_relative is None or node_relative is None:
            raise ValueError("joint training requires edge-flux and node-message targets")
        loss = flux_relative + node_relative
    return loss, {
        "loss": float(loss.detach().float().cpu()),
        "flux_relative_mse": (
            None if flux_relative is None else float(flux_relative.detach().float().cpu())
        ),
        "node_relative_mse": (
            None if node_relative is None else float(node_relative.detach().float().cpu())
        ),
    }


@torch.no_grad()
def _validation_loss(
    model: SparseIncidenceConductanceLayer,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    objective: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    loader = _loader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            loss, _ = _normalized_loss(model, batch, objective=objective)
        total += float(loss.float().cpu()) * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


def train_sparse_model(
    model: SparseIncidenceConductanceLayer,
    train_examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
    objective: str,
) -> list[dict[str, Any]]:
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"unknown training objective {objective!r}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    scaler = _grad_scaler(amp)
    best_validation = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        loader = _loader(
            train_examples,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        for batch in loader:
            batch = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, _ = _normalized_loss(model, batch, objective=objective)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().float().cpu()) * batch.num_graphs
            count += batch.num_graphs
        validation = _validation_loss(
            model,
            validation_examples,
            device=device,
            amp=amp,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            objective=objective,
        )
        train_loss = total / max(count, 1)
        history.append(
            {
                "epoch": epoch,
                "training_objective": objective,
                "train_loss": train_loss,
                "validation_loss": validation,
            }
        )
        if validation < best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _pearson(first: Tensor, second: Tensor) -> float | None:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    if first.numel() < 2:
        return None
    first = first - first.mean()
    second = second - second.mean()
    first_tolerance = 1.0e-7 * max(float(first.abs().max()), 1.0)
    second_tolerance = 1.0e-7 * max(float(second.abs().max()), 1.0)
    if float(first.norm()) <= first_tolerance or float(second.norm()) <= second_tolerance:
        return None
    denominator = first.norm() * second.norm()
    if float(denominator) <= torch.finfo(torch.float32).eps:
        return None
    return float(torch.dot(first, second) / denominator)


def _rank(values: Tensor) -> Tensor:
    # Synthetic conductances are continuous; ties are vanishingly rare.  The
    # deterministic stable ordering is sufficient for this diagnostic.
    order = torch.argsort(values.reshape(-1), stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
    return ranks


def _mean(values: Iterable[float | None]) -> float | None:
    selected = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    return sum(selected) / len(selected) if selected else None


@torch.no_grad()
def evaluate_sparse_model(
    model: SparseIncidenceConductanceLayer,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    oracle: bool = False,
) -> dict[str, Any]:
    model.eval()
    flux_rel: list[float] = []
    node_rel: list[float] = []
    next_rel: list[float] = []
    log_c_rmse: list[float] = []
    correlations: list[float | None] = []
    rank_correlations: list[float | None] = []
    coverage: list[float] = []
    cap_active = 0
    cap_total = 0
    predictions_by_graph: dict[str, list[Tensor]] = {}
    loader = _loader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=pin_memory)
        override = batch.true_conductance if oracle else None
        with _autocast(device, amp):
            predicted_next, diagnostics = model(
                batch, conductance_override=override, return_diagnostics=True
            )
        for graph_number, graph_id in enumerate(batch.graph_ids):
            edge_mask = batch.edge_graph == graph_number
            node_mask = batch.node_graph == graph_number
            predicted_flux = diagnostics["edge_flux"][edge_mask].float()
            predicted_c = diagnostics["conductance"][edge_mask].float()
            true_flux = batch.true_flux[edge_mask].float()
            true_c = batch.true_conductance[edge_mask].float()
            true_message = batch.true_node_message[node_mask].float()
            predicted_message = diagnostics["node_message"][node_mask].float()
            true_next = batch.true_next_state[node_mask].float()
            current_next = predicted_next[node_mask].float()
            epsilon = torch.finfo(torch.float32).eps
            flux_rel.append(
                float((predicted_flux - true_flux).norm() / true_flux.norm().clamp_min(epsilon))
            )
            node_rel.append(
                float(
                    (predicted_message - true_message).norm()
                    / true_message.norm().clamp_min(epsilon)
                )
            )
            next_rel.append(
                float((current_next - true_next).norm() / true_next.norm().clamp_min(epsilon))
            )
            log_c_rmse.append(
                float(
                    torch.mean(
                        (predicted_c.clamp_min(1e-8).log() - true_c.clamp_min(1e-8).log()).square()
                    ).sqrt()
                )
            )
            correlation = _pearson(predicted_c.cpu(), true_c.cpu())
            correlations.append(correlation)
            rank_correlations.append(
                None
                if correlation is None
                else _pearson(_rank(predicted_c.cpu()), _rank(true_c.cpu()))
            )
            gradient = batch.true_gradient[edge_mask]
            coverage.append(float((gradient.abs().amax(dim=1) > 1.0e-6).float().mean()))
            predictions_by_graph.setdefault(graph_id, []).append(predicted_c.detach().cpu())
        cap_active += int(diagnostics["cap_active"].sum())
        cap_total += int(diagnostics["cap_active"].numel())
    state_variation = []
    for values in predictions_by_graph.values():
        if len(values) > 1 and all(value.shape == values[0].shape for value in values):
            state_variation.append(float(torch.stack(values).std(dim=0, unbiased=False).mean()))
    return {
        "graph_macro_flux_relative_l2": _mean(flux_rel),
        "graph_macro_node_message_relative_l2": _mean(node_rel),
        "graph_macro_next_state_relative_l2": _mean(next_rel),
        "graph_macro_log_conductance_rmse": _mean(log_c_rmse),
        "graph_macro_conductance_pearson": _mean(correlations),
        "conductance_pearson_defined_fraction": sum(value is not None for value in correlations)
        / max(len(correlations), 1),
        "graph_macro_conductance_spearman": _mean(rank_correlations),
        "excited_edge_fraction": _mean(coverage),
        "mean_conductance_state_variation": _mean(state_variation),
        "stability_cap_activation_fraction": cap_active / max(cap_total, 1),
        "num_examples": len(examples),
        "num_graph_ids": len({str(example["graph_id"]) for example in examples}),
    }


def least_squares_metrics(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-graph flux LS using the evaluated excitations (diagnostic ceiling)."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example["graph_id"]), []).append(example)
    flux_errors: list[float] = []
    log_errors: list[float] = []
    correlations: list[float | None] = []
    identifiable: list[float] = []
    for group in groups.values():
        numerator = None
        denominator = None
        for example in group:
            gradient = example["true_gradient"].float()
            observed = example.get("observed_flux", example["true_flux"]).float()
            current_numerator = (gradient * observed).sum(dim=1)
            current_denominator = gradient.square().sum(dim=1)
            numerator = current_numerator if numerator is None else numerator + current_numerator
            denominator = (
                current_denominator if denominator is None else denominator + current_denominator
            )
        assert numerator is not None and denominator is not None
        estimated = (numerator / denominator.clamp_min(1.0e-12)).clamp_min(1.0e-6)
        truth = group[0]["true_conductance"].float()
        excited = denominator > 1.0e-10
        identifiable.append(float(excited.float().mean()))
        if excited.any():
            log_errors.append(
                float(((estimated[excited].log() - truth[excited].log()).square().mean()).sqrt())
            )
            correlations.append(_pearson(estimated[excited], truth[excited]))
        for example in group:
            gradient = example["true_gradient"].float()
            truth_flux = example["true_flux"].float()
            predicted_flux = estimated[:, None] * gradient
            flux_errors.append(
                float((predicted_flux - truth_flux).norm() / truth_flux.norm().clamp_min(1.0e-12))
            )
    return {
        "protocol": "transductive_same-evaluation-excitations_identification_ceiling",
        "graph_macro_flux_relative_l2": _mean(flux_errors),
        "graph_macro_log_conductance_rmse": _mean(log_errors),
        "graph_macro_conductance_pearson": _mean(correlations),
        "identifiable_edge_fraction": _mean(identifiable),
        "num_graph_ids": len(groups),
    }


def _node_message_design(example: Mapping[str, Any]) -> Tensor:
    """Dense diagnostic design only; the learned layer remains gather/scatter sparse."""

    edge_index = example["edge_index"].long().cpu()
    gradient = example["true_gradient"].double().cpu()
    num_nodes = int(example["node_state"].shape[0])
    channels = int(gradient.shape[1])
    num_edges = int(edge_index.shape[1])
    design = gradient.new_zeros((num_nodes * channels, num_edges))
    edge_ids = torch.arange(num_edges).view(-1, 1).expand(-1, channels)
    channel_ids = torch.arange(channels).view(1, -1)
    tail_rows = edge_index[0].view(-1, 1) * channels + channel_ids
    head_rows = edge_index[1].view(-1, 1) * channels + channel_ids
    design[tail_rows.reshape(-1), edge_ids.reshape(-1)] = -gradient.reshape(-1)
    design[head_rows.reshape(-1), edge_ids.reshape(-1)] = gradient.reshape(-1)
    return design


def _projected_nnls(
    design: Tensor,
    target: Tensor,
    *,
    max_iterations: int = 1_000,
    tolerance: float = 1.0e-10,
) -> tuple[Tensor, int]:
    """Solve nonnegative least squares with deterministic projected FISTA."""

    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise ValueError("NNLS design and target shapes are inconsistent")
    if design.shape[1] == 0:
        return design.new_empty(0), 0
    spectral = torch.linalg.svdvals(design)
    lipschitz = spectral[0].square() if spectral.numel() else design.new_tensor(0.0)
    if float(lipschitz) <= torch.finfo(design.dtype).eps:
        return design.new_zeros(design.shape[1]), 0
    # The unconstrained solution is already the exact NNLS solution when it is
    # nonnegative.  This makes the noiseless, full-rank ceiling numerically sharp.
    unconstrained = torch.linalg.lstsq(design, target).solution
    if bool(torch.all(unconstrained >= 0)):
        return unconstrained, 0
    estimate = unconstrained.clamp_min(0)
    accelerated = estimate.clone()
    momentum = 1.0
    scale = max(float(estimate.norm()), 1.0)
    for iteration in range(1, max_iterations + 1):
        gradient = design.mT @ (design @ accelerated - target)
        updated = (accelerated - gradient / lipschitz).clamp_min(0)
        if float((updated - estimate).norm()) <= tolerance * scale:
            return updated, iteration
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - estimate)
        estimate = updated
        momentum = next_momentum
        scale = max(float(estimate.norm()), 1.0)
    return estimate, max_iterations


def node_message_nnls_metrics(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Same-evaluation node-output NNLS ceiling for one conductance per edge.

    Unlike :func:`least_squares_metrics`, this diagnostic never reads observed
    per-edge flux.  It estimates nonnegative edge conductances only from the
    observed node messages and the known excitation gradients.  It is still a
    transductive ceiling, not a held-graph predictive baseline.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example["graph_id"]), []).append(example)
    clean_errors: list[float] = []
    observed_fit_errors: list[float] = []
    log_errors: list[float] = []
    correlations: list[float | None] = []
    excited_fractions: list[float] = []
    rank_fractions: list[float] = []
    iterations: list[float] = []
    for group in groups.values():
        reference_edges = group[0]["edge_index"]
        reference_truth = group[0]["true_conductance"].double().cpu()
        designs: list[Tensor] = []
        observed_targets: list[Tensor] = []
        for example in group:
            if not torch.equal(example["edge_index"], reference_edges):
                raise ValueError("examples sharing graph_id must share edge_index")
            truth = example["true_conductance"].double().cpu()
            if not torch.allclose(truth, reference_truth):
                raise ValueError("node-message NNLS requires static conductance per graph_id")
            design = _node_message_design(example)
            observed = example.get("observed_node_message")
            if observed is None:
                observed = example["true_node_message"]
            designs.append(design)
            observed_targets.append(observed.double().cpu().reshape(-1))
        stacked_design = torch.cat(designs, dim=0)
        stacked_target = torch.cat(observed_targets, dim=0)
        estimated, used_iterations = _projected_nnls(stacked_design, stacked_target)
        iterations.append(float(used_iterations))
        column_energy = stacked_design.square().sum(dim=0)
        excited = column_energy > 1.0e-12
        excited_fractions.append(float(excited.float().mean()))
        rank = int(torch.linalg.matrix_rank(stacked_design))
        rank_fractions.append(rank / max(stacked_design.shape[1], 1))
        if excited.any():
            log_errors.append(
                float(
                    (
                        estimated[excited].clamp_min(1.0e-12).log()
                        - reference_truth[excited].clamp_min(1.0e-12).log()
                    )
                    .square()
                    .mean()
                    .sqrt()
                )
            )
            correlations.append(_pearson(estimated[excited], reference_truth[excited]))
        for example, design, observed_target in zip(group, designs, observed_targets, strict=True):
            predicted = design @ estimated
            clean_target = example["true_node_message"].double().cpu().reshape(-1)
            clean_errors.append(
                float((predicted - clean_target).norm() / clean_target.norm().clamp_min(1.0e-12))
            )
            observed_fit_errors.append(
                float(
                    (predicted - observed_target).norm() / observed_target.norm().clamp_min(1.0e-12)
                )
            )
    return {
        "protocol": "transductive_same-evaluation-node-messages_nnls_ceiling",
        "graph_macro_node_message_relative_l2": _mean(clean_errors),
        "graph_macro_observed_fit_relative_l2": _mean(observed_fit_errors),
        "graph_macro_log_conductance_rmse": _mean(log_errors),
        "graph_macro_conductance_pearson": _mean(correlations),
        "excited_edge_fraction": _mean(excited_fractions),
        "design_rank_fraction": _mean(rank_fractions),
        "mean_solver_iterations": _mean(iterations),
        "num_graph_ids": len(groups),
    }


@torch.no_grad()
def evaluate_rollout(
    model: SparseIncidenceConductanceLayer,
    trajectories: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    device: torch.device,
    amp: bool,
    oracle: bool,
) -> dict[str, Any]:
    errors: dict[int, list[float]] = {int(horizon): [] for horizon in horizons}
    growth: list[float] = []
    dissipation_violations = 0
    steps_total = 0
    cap_active = 0
    for trajectory in trajectories:
        state = trajectory["states"][0].to(device)
        initial_norm = float(state.norm())
        previous_norm = initial_norm
        edge_index = trajectory["edge_index"].to(device)
        edge_features = trajectory["edge_features"].to(device)
        for time_index in range(max(horizons)):
            record = {
                "graph_id": trajectory["graph_id"],
                "node_state": state,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "step_size": float(trajectory["steps"][time_index]),
            }
            batch = pack_graph_examples([record]).to(device)
            override = None
            if oracle:
                override = nonlinear_conductance(edge_features, edge_gradient(edge_index, state))
            with _autocast(device, amp):
                state, diagnostics = model(
                    batch,
                    node_state=state,
                    conductance_override=override,
                    return_diagnostics=True,
                )
            current_norm = float(state.float().norm())
            dissipation_violations += int(current_norm > previous_norm + 1.0e-6)
            previous_norm = current_norm
            steps_total += 1
            cap_active += int(diagnostics["cap_active"].sum())
            horizon = time_index + 1
            if horizon in errors:
                truth = trajectory["states"][horizon].to(device)
                errors[horizon].append(
                    float((state.float() - truth).norm() / truth.norm().clamp_min(1e-12))
                )
        growth.append(previous_norm / max(initial_norm, 1.0e-12))
    result = {f"horizon_{horizon}_relative_l2": _mean(values) for horizon, values in errors.items()}
    result.update(
        {
            "final_norm_over_initial": _mean(growth),
            "dissipation_violation_fraction": dissipation_violations / max(steps_total, 1),
            "stability_cap_activation_fraction": cap_active / max(steps_total, 1),
        }
    )
    return result


def _model_for_examples(
    examples: Sequence[Mapping[str, Any]], mode: str, *, hidden_channels: int
) -> SparseIncidenceConductanceLayer:
    first = examples[0]
    return SparseIncidenceConductanceLayer(
        channels=int(first["node_state"].shape[1]),
        edge_feature_channels=int(first["edge_features"].shape[1]),
        hidden_channels=hidden_channels,
        requested_step=0.025,
        stability_margin=0.95,
        adaptive_stability=True,
        mode=mode,
    )


def _factorial_key(example: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    metadata = example["metadata"]
    return metadata["contrast"], metadata["active_node_fraction"], metadata["snr_db"]


def run_core(
    core: dict[str, Any],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Tensor]]]:
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    states: dict[str, dict[str, Tensor]] = {}
    baseline_specs = (
        ("isotropic", "isotropic", "node_only", "constant-conductance ablation"),
        ("edge_only", "edge_only", "node_only", "static edge-feature ablation"),
        (
            "gradient_only",
            "gradient_only",
            "node_only",
            "state-gradient-only ablation C=f(abs(BH))",
        ),
        (
            "full",
            "full",
            "node_only",
            "headline node-output-only predictive model",
        ),
        (
            "full_flux_supervised",
            "full",
            "flux_only",
            "per-edge-flux-supervised neural ceiling",
        ),
        ("full_joint", "full", "joint", "joint-supervision objective ablation"),
    )
    mode_seed_offset = {"isotropic": 0, "edge_only": 1, "gradient_only": 2, "full": 3}
    for suite_number, suite_name in enumerate(("s1", "s2", "s3", "s4")):
        suite = core[suite_name]
        train_examples = suite["train"]
        validation_examples = suite["validation"]
        test_examples = suite["test"]
        hidden_channels = 64
        suite_result: dict[str, Any] = {
            "claim": CORE_CLAIMS[suite_name],
            "description": suite["description"],
            "split_graph_counts": {
                split: len({item["graph_id"] for item in suite.get(split, [])})
                for split in ("train", "validation", "test", "seen_test")
                if split in suite
            },
            "headline_baseline": "full",
            "objective_protocol": {
                "headline": "node_only",
                "flux_supervised_ceiling": "full_flux_supervised",
                "joint_objective_ablation": "full_joint",
            },
            "baselines": {},
        }
        trained: dict[str, tuple[SparseIncidenceConductanceLayer, str]] = {}
        for baseline_name, mode, objective, role in baseline_specs:
            initialization_offset = mode_seed_offset[mode]
            seed_everything(seed + suite_number * 100 + initialization_offset)
            model = _model_for_examples(train_examples, mode, hidden_channels=hidden_channels).to(
                device
            )
            history = train_sparse_model(
                model,
                train_examples,
                validation_examples,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=num_workers,
                seed=seed + suite_number * 1000 + initialization_offset * 100,
                objective=objective,
            )
            for row in history:
                histories.append({"suite": suite_name, "baseline": baseline_name, **row})
            trained[baseline_name] = (model, objective)
            states[f"{suite_name}_{baseline_name}"] = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            metric = evaluate_sparse_model(
                model,
                test_examples,
                device=device,
                amp=amp,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            suite_result["baselines"][baseline_name] = {
                "training_objective": objective,
                "role": role,
                "unseen_graph_test": metric,
            }
            if suite_name == "s1":
                suite_result["baselines"][baseline_name]["seen_graph_new_excitation_test"] = (
                    evaluate_sparse_model(
                        model,
                        suite["seen_test"],
                        device=device,
                        amp=amp,
                        batch_size=batch_size,
                        pin_memory=pin_memory,
                        num_workers=num_workers,
                    )
                )
            if suite_name == "s3":
                suite_result["baselines"][baseline_name]["rollout"] = evaluate_rollout(
                    model,
                    suite["rollout_test"],
                    suite["horizons"],
                    device=device,
                    amp=amp,
                    oracle=False,
                )
        oracle_model = _model_for_examples(
            train_examples, "full", hidden_channels=hidden_channels
        ).to(device)
        suite_result["baselines"]["oracle"] = {
            "training_objective": "analytic_oracle",
            "role": "ground-truth conductance oracle",
            "unseen_graph_test": evaluate_sparse_model(
                oracle_model,
                test_examples,
                device=device,
                amp=amp,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                oracle=True,
            ),
        }
        if suite_name == "s1":
            suite_result["baselines"]["oracle"]["seen_graph_new_excitation_test"] = (
                evaluate_sparse_model(
                    oracle_model,
                    suite["seen_test"],
                    device=device,
                    amp=amp,
                    batch_size=batch_size,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    oracle=True,
                )
            )
        if suite_name == "s3":
            suite_result["baselines"]["oracle"]["rollout"] = evaluate_rollout(
                oracle_model,
                suite["rollout_test"],
                suite["horizons"],
                device=device,
                amp=amp,
                oracle=True,
            )
        if suite_name in {"s1", "s4"}:
            suite_result["baselines"]["flux_ls"] = {
                "training_objective": "same-evaluation observed edge flux",
                "role": "transductive per-edge-flux least-squares ceiling",
                "unseen_graph_test": least_squares_metrics(test_examples),
            }
            suite_result["baselines"]["node_message_nnls"] = {
                "training_objective": "same-evaluation observed node message",
                "role": "transductive node-output nonnegative least-squares ceiling",
                "unseen_graph_test": node_message_nnls_metrics(test_examples),
            }
            if suite_name == "s1":
                suite_result["baselines"]["flux_ls"]["seen_graph_new_excitation_test"] = (
                    least_squares_metrics(suite["seen_test"])
                )
                suite_result["baselines"]["node_message_nnls"]["seen_graph_new_excitation_test"] = (
                    node_message_nnls_metrics(suite["seen_test"])
                )
        if suite_name == "s4":
            factorial: list[dict[str, Any]] = []
            keys = sorted({_factorial_key(example) for example in test_examples}, key=str)
            for key in keys:
                subset = [example for example in test_examples if _factorial_key(example) == key]
                for baseline_name, (model, objective) in trained.items():
                    factorial.append(
                        {
                            "contrast": key[0],
                            "active_node_fraction": key[1],
                            "snr_db": key[2],
                            "baseline": baseline_name,
                            "training_objective": objective,
                            **evaluate_sparse_model(
                                model,
                                subset,
                                device=device,
                                amp=amp,
                                batch_size=batch_size,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                            ),
                        }
                    )
                factorial.append(
                    {
                        "contrast": key[0],
                        "active_node_fraction": key[1],
                        "snr_db": key[2],
                        "baseline": "flux_ls",
                        "training_objective": "same-evaluation observed edge flux",
                        **least_squares_metrics(subset),
                    }
                )
                factorial.append(
                    {
                        "contrast": key[0],
                        "active_node_fraction": key[1],
                        "snr_db": key[2],
                        "baseline": "node_message_nnls",
                        "training_objective": "same-evaluation observed node message",
                        **node_message_nnls_metrics(subset),
                    }
                )
            suite_result["factorial"] = factorial
        results[suite_name] = suite_result
    return results, histories, states


@dataclass
class PublicPacked:
    x: Tensor
    edge_index: Tensor
    edge_features: Tensor
    node_graph: Tensor
    y: Tensor
    graph_ids: list[str]
    task: str
    categorical: bool

    @property
    def num_graphs(self) -> int:
        return len(self.graph_ids)

    def to(self, device: torch.device, *, non_blocking: bool) -> PublicPacked:
        return PublicPacked(
            x=self.x.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            edge_features=self.edge_features.to(device, non_blocking=non_blocking),
            node_graph=self.node_graph.to(device, non_blocking=non_blocking),
            y=self.y.to(device, non_blocking=non_blocking),
            graph_ids=self.graph_ids,
            task=self.task,
            categorical=self.categorical,
        )

    def pin_memory(self) -> PublicPacked:
        return PublicPacked(
            x=self.x.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            edge_features=self.edge_features.pin_memory(),
            node_graph=self.node_graph.pin_memory(),
            y=self.y.pin_memory(),
            graph_ids=self.graph_ids,
            task=self.task,
            categorical=self.categorical,
        )


def pack_public(records: Sequence[Mapping[str, Any]]) -> PublicPacked:
    if not records:
        raise ValueError("empty public batch")
    task = str(records[0]["task"])
    categorical = bool(records[0]["categorical"])
    nodes: list[Tensor] = []
    edges: list[Tensor] = []
    edge_features: list[Tensor] = []
    node_graph: list[Tensor] = []
    labels: list[Tensor] = []
    graph_ids: list[str] = []
    offset = 0
    for graph_number, record in enumerate(records):
        if record["task"] != task or bool(record["categorical"]) != categorical:
            raise ValueError("public batch mixes tasks or feature types")
        x = record["x"]
        nodes.append(x)
        edges.append(record["edge_index"] + offset)
        edge_features.append(record["edge_features"])
        node_graph.append(torch.full((x.shape[0],), graph_number, dtype=torch.long))
        labels.append(record["y"])
        graph_ids.append(str(record["graph_id"]))
        offset += int(x.shape[0])
    y = (
        torch.cat(labels)
        if task == "node"
        else torch.stack([label.reshape(-1) for label in labels])
    )
    return PublicPacked(
        x=torch.cat(nodes),
        edge_index=torch.cat(edges, dim=1),
        edge_features=torch.cat(edge_features),
        node_graph=torch.cat(node_graph),
        y=y,
        graph_ids=graph_ids,
        task=task,
        categorical=categorical,
    )


class SumCategoricalEncoder(nn.Module):
    def __init__(self, columns: int, hidden: int, categories: int = 256) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, hidden) for _ in range(columns)])

    def forward(self, values: Tensor) -> Tensor:
        result = self.embeddings[0](values[:, 0].long())
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            result = result + embedding(values[:, column].long())
        return result


class PublicConductanceModel(nn.Module):
    def __init__(
        self,
        sample: Mapping[str, Any],
        *,
        hidden: int,
        num_classes: int,
        official_molecule: bool,
    ) -> None:
        super().__init__()
        node_width = int(sample["x"].shape[1])
        edge_width = int(sample["edge_features"].shape[1])
        self.task = str(sample["task"])
        if bool(sample["categorical"]) and official_molecule:
            try:
                from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
            except (ImportError, OSError) as error:  # pragma: no cover - optional path
                raise RuntimeError(
                    "official MolHIV requires OGB AtomEncoder/BondEncoder"
                ) from error
            self.node_encoder = AtomEncoder(hidden)
            self.edge_encoder = BondEncoder(hidden)
        elif bool(sample["categorical"]):
            self.node_encoder = SumCategoricalEncoder(node_width, hidden)
            self.edge_encoder = SumCategoricalEncoder(edge_width, hidden)
        else:
            self.node_encoder = nn.Linear(node_width, hidden)
            self.edge_encoder = nn.Linear(edge_width, hidden)
        self.uses_edge_features = True
        self.normalization = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes if self.task == "node" else 1)
        self.layer = SparseIncidenceConductanceLayer(
            channels=hidden,
            edge_feature_channels=hidden,
            hidden_channels=hidden,
            requested_step=0.02,
            mode="full",
        )

    def forward(self, batch: PublicPacked) -> Tensor:
        node_state = self.node_encoder(batch.x)
        edge_features = self.edge_encoder(batch.edge_features)
        edge_graph = batch.node_graph.index_select(0, batch.edge_index[0])
        sparse_batch = PackedGraphBatch(
            node_state=node_state,
            edge_index=batch.edge_index,
            edge_features=edge_features,
            node_graph=batch.node_graph,
            edge_graph=edge_graph,
            graph_ids=batch.graph_ids,
            requested_step=node_state.new_full((batch.num_graphs,), 0.02),
        )
        node_state = self.layer(sparse_batch)
        node_state = nnf.silu(self.normalization(node_state))
        if self.task == "node":
            return self.head(node_state)
        pooled = node_state.new_zeros((batch.num_graphs, node_state.shape[1]))
        pooled.index_add_(0, batch.node_graph, node_state)
        counts = torch.bincount(batch.node_graph, minlength=batch.num_graphs).to(node_state)
        return self.head(pooled / counts[:, None].clamp_min(1)).squeeze(-1)


def _public_loader(
    dataset: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=pack_public,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _public_loss(logits: Tensor, labels: Tensor, task: str) -> Tensor:
    if task == "node":
        return nnf.cross_entropy(logits, labels.long())
    valid = torch.isfinite(labels.reshape(-1))
    return nnf.binary_cross_entropy_with_logits(logits[valid], labels.reshape(-1)[valid].float())


def _public_loss_weight(labels: Tensor, task: str) -> int:
    """Return the number of labels represented by a mean-reduced task loss."""

    if task == "node":
        return int(labels.numel())
    return int(torch.isfinite(labels.reshape(-1)).sum())


def _macro_f1(predictions: Tensor, labels: Tensor) -> float:
    scores = []
    for label in torch.unique(labels):
        true_positive = ((predictions == label) & (labels == label)).sum().float()
        false_positive = ((predictions == label) & (labels != label)).sum().float()
        false_negative = ((predictions != label) & (labels == label)).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator > 0:
            scores.append(float(2 * true_positive / denominator))
    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def evaluate_public(
    model: PublicConductanceModel,
    dataset: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
) -> dict[str, Any]:
    model.eval()
    outputs: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in _public_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    ):
        batch = batch.to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            outputs.append(model(batch).float().cpu())
        labels.append(batch.y.float().cpu())
    output = torch.cat(outputs)
    label = torch.cat(labels)
    if model.task == "node":
        return {
            "macro_f1": _macro_f1(output.argmax(dim=1), label.long()),
            "num_labels": label.numel(),
        }
    try:
        from ogb.graphproppred import Evaluator
    except (ImportError, OSError) as error:  # pragma: no cover - optional path
        raise RuntimeError("official MolHIV evaluation requires the OGB evaluator") from error
    evaluator = Evaluator(name="ogbg-molhiv")
    score = evaluator.eval({"y_true": label.reshape(-1, 1), "y_pred": output.reshape(-1, 1)})[
        "rocauc"
    ]
    return {
        "roc_auc": float(score),
        "num_graphs": label.numel(),
        "evaluator": "ogb.graphproppred.Evaluator",
    }


def run_public(
    datasets: dict[str, Any],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Tensor]]]:
    if datasets.get("fixture") is not False:
        raise ValueError(
            "Public experiments require official data; generated substitutes are unsupported"
        )
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    states: dict[str, dict[str, Tensor]] = {}
    for dataset_number, dataset_name in enumerate(("pascalvoc_sp", "ogbg_molhiv")):
        splits = datasets[dataset_name]
        sample = splits["train"][0]
        num_classes = 21 if dataset_name == "pascalvoc_sp" else 3
        hidden = 96
        results[dataset_name] = {
            "fixture": False,
            "official_result": True,
            "model_protocol": {
                "hidden_channels": hidden,
                "backbone_depth": 1,
                "model": "conductance_model",
                "split": "official",
                "competitor_execution": "not implemented; published results compared externally",
            },
            "baselines": {},
        }
        model_seed = seed + dataset_number * 101
        for model_name in ("conductance_model",):
            seed_everything(model_seed)
            model = PublicConductanceModel(
                sample,
                hidden=hidden,
                num_classes=num_classes,
                official_molecule=(dataset_name == "ogbg_molhiv"),
            ).to(device)
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scaler = _grad_scaler(amp)
            best_validation = math.inf
            best_state = None
            for epoch in range(1, epochs + 1):
                model.train()
                total = 0.0
                count = 0
                for batch in _public_loader(
                    splits["train"],
                    batch_size=batch_size,
                    shuffle=True,
                    seed=seed + epoch,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                ):
                    batch = batch.to(device, non_blocking=pin_memory)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device, amp):
                        loss = _public_loss(model(batch), batch.y, model.task)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    loss_weight = _public_loss_weight(batch.y, model.task)
                    total += float(loss.detach().float().cpu()) * loss_weight
                    count += loss_weight
                model.eval()
                validation_total = 0.0
                validation_count = 0
                with torch.no_grad():
                    for batch in _public_loader(
                        splits["validation"],
                        batch_size=batch_size,
                        shuffle=False,
                        seed=0,
                        pin_memory=pin_memory,
                        num_workers=num_workers,
                    ):
                        batch = batch.to(device, non_blocking=pin_memory)
                        with _autocast(device, amp):
                            loss = _public_loss(model(batch), batch.y, model.task)
                        loss_weight = _public_loss_weight(batch.y, model.task)
                        validation_total += float(loss.float().cpu()) * loss_weight
                        validation_count += loss_weight
                validation_loss = validation_total / max(validation_count, 1)
                histories.append(
                    {
                        "suite": dataset_name,
                        "baseline": model_name,
                        "epoch": epoch,
                        "train_loss": total / max(count, 1),
                        "validation_loss": validation_loss,
                    }
                )
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            if best_state is not None:
                model.load_state_dict(best_state)
            state_key = f"{dataset_name}_{model_name}"
            states[state_key] = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            results[dataset_name]["baselines"][model_name] = {
                "parameter_count": parameter_count,
                "parameter_count_policy": "trainable_active_parameters_only",
                "uses_edge_features": model.uses_edge_features,
                "best_validation_loss": best_validation,
                "test": evaluate_public(
                    model,
                    splits["test"],
                    device=device,
                    batch_size=batch_size,
                    amp=amp,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                ),
            }
    return results, histories, states


def _metric_rows(value: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_metric_rows(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_metric_rows(child, (*path, str(index))))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append({"path": "/".join(path), "value": value})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output_dir(path: Path) -> Path:
    """Claim an empty run directory before data preparation or artifact writes."""

    resolved = path.expanduser().resolve()
    if resolved.parent == resolved:
        raise ValueError("--output-dir cannot be a filesystem root")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"--output-dir is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(
                f"--output-dir already contains artifacts; choose a new empty path: {resolved}"
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def _seed_axis_applicability(
    suite: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Describe which resolved seed axes actually affect each requested protocol."""

    applicability: dict[str, dict[str, dict[str, Any]]] = {}
    if suite in {"core", "all"}:
        applicability["core"] = {
            "data": {
                "applicable": True,
                "use": "generated graphs, excitations, trajectories, labels, and cache key",
            },
            "split": {
                "applicable": False,
                "use": "not_applicable: generated split assignment is part of data_seed",
            },
            "chart": {
                "applicable": False,
                "use": "not_applicable: conductance track has no spanning-tree chart sampling",
            },
            "model": {
                "applicable": True,
                "use": "model initialization and training DataLoader shuffle",
            },
        }
    if suite in {"public", "all"}:
        applicability["public"] = {
            "data": {
                "applicable": False,
                "use": "not_applicable: official dataset content is fixed by its source",
            },
            "split": {
                "applicable": False,
                "use": "not_applicable: official PascalVOC-SP/MolHIV splits are fixed",
            },
            "chart": {
                "applicable": False,
                "use": "not_applicable: public conductance baselines do not sample tree charts",
            },
            "model": {
                "applicable": True,
                "use": "model initialization and training DataLoader shuffle",
            },
        }
    return applicability


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "public", "all"), default="core")
    parser.add_argument("--data-root", type=Path, default=default_root / "data")
    parser.add_argument("--output-dir", type=Path, default=default_root / "results" / "paper")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="legacy fallback for any seed axis not supplied explicitly",
    )
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--chart-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-download", action="store_true", help="allow official PyG/OGB downloads"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    arguments = build_parser().parse_args(argv)
    seed_axes: SeedAxes = resolve_seed_axes(
        arguments.seed,
        data_seed=arguments.data_seed,
        split_seed=arguments.split_seed,
        chart_seed=arguments.chart_seed,
        model_seed=arguments.model_seed,
    )
    if arguments.batch_size < 1 or arguments.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative")
    device = resolve_device(arguments.device)
    amp = device.type == "cuda" if arguments.amp is None else bool(arguments.amp)
    if device.type != "cuda" and amp:
        raise ValueError("--amp is a CUDA float16 path; use --no-amp on CPU")
    pin_memory = (
        device.type == "cuda" if arguments.pin_memory is None else bool(arguments.pin_memory)
    )
    if device.type != "cuda":
        pin_memory = False
    epochs = arguments.epochs if arguments.epochs is not None else 100
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    # Dataset preparation receives only the data axis.  Reset the global RNG to
    # the model axis immediately before optimization below.
    seed_everything(seed_axes.data)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = _prepare_output_dir(arguments.output_dir)
    started = time.perf_counter()
    prepared: dict[str, Any] = {}
    core = None
    public = None
    if arguments.suite in {"core", "all"}:
        core, manifest_path, manifest = prepare_core_cache(arguments.data_root, seed=seed_axes.data)
        prepared["core"] = {
            "manifest": str(manifest_path),
            "cache_key": manifest["cache_key"],
            "data_seed": seed_axes.data,
        }
    if arguments.suite in {"public", "all"}:
        public, marker_path, manifest = prepare_public_data(
            arguments.data_root,
            allow_download=arguments.allow_download,
        )
        prepared["public"] = {
            "manifest": str(marker_path),
            "fixture": manifest["fixture"],
            "data_seed": "not_applicable",
            "split_seed": "not_applicable",
            "chart_seed": "not_applicable",
        }
    seed_applicability = _seed_axis_applicability(arguments.suite)
    if arguments.prepare_only:
        summary = {
            "status": "prepared",
            "suite": arguments.suite,
            "seed_axes": seed_axes.to_manifest(),
            "seed_axis_applicability": seed_applicability,
            "prepared": prepared,
        }
        (output_dir / "prepare_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    seed_everything(seed_axes.model)
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    model_states: dict[str, Any] = {}
    try:
        if core is not None:
            core_results, core_history, core_states = run_core(
                core,
                device=device,
                epochs=epochs,
                learning_rate=arguments.learning_rate,
                batch_size=arguments.batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=arguments.num_workers,
                seed=seed_axes.model,
            )
            results["core"] = core_results
            histories.extend(core_history)
            model_states["core"] = core_states
        if public is not None:
            public_epochs = min(epochs, 50)
            public_results, public_history, public_states = run_public(
                public,
                device=device,
                epochs=public_epochs,
                learning_rate=arguments.learning_rate,
                batch_size=arguments.batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=arguments.num_workers,
                seed=seed_axes.model,
            )
            results["public"] = public_results
            histories.extend(public_history)
            model_states["public"] = public_states
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA out of memory in the paper runner. Re-run with a smaller --batch-size "
            "(and optionally --no-amp only for numerical diagnosis; AMP normally saves memory)."
        ) from error
    elapsed = time.perf_counter() - started
    summary = {
        "scope": "independent_sparse_incidence_conductance_attention",
        "suite": arguments.suite,
        "seed_axes": seed_axes.to_manifest(),
        "seed_axis_applicability": seed_applicability,
        "prepared": prepared,
        "configuration": {
            "epochs": epochs,
            "learning_rate": arguments.learning_rate,
            "batch_size": arguments.batch_size,
            "num_workers": arguments.num_workers,
        },
        "runtime": {
            **runtime_metadata(
                device, amp=amp, pin_memory=pin_memory, batch_size=arguments.batch_size
            ),
            "elapsed_seconds": elapsed,
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metric_rows = _metric_rows(results)
    _write_csv(output_dir / "metrics.csv", metric_rows, ["path", "value"])
    _write_csv(
        output_dir / "history.csv",
        histories,
        [
            "suite",
            "baseline",
            "training_objective",
            "epoch",
            "train_loss",
            "validation_loss",
        ],
    )
    torch.save(model_states, output_dir / "models.pt")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
