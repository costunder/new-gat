"""Variable-beta encoder and fair multi-chart downstream training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .paper_data import (
    ZINC_NUM_ATOM_TYPES,
    ZINC_NUM_BOND_TYPES,
    GraphRecord,
    chart_key,
    sample_paper_charts,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GraphChartView:
    """One chart view of one physical graph."""

    graph_id: str
    graph_family: str
    graph_status: str
    chart_status: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    basis: FloatArray
    target: tuple[float, ...]
    chart_name: str
    tree_key: tuple[int, ...]
    x: tuple[int, ...] | None = None
    edge_attr: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PaddedChartBatch:
    """Dense padded batch with independent masks for edges and cycle columns."""

    basis: Tensor
    edge_features: Tensor
    edge_mask: Tensor
    cycle_mask: Tensor
    edge_index: Tensor
    node_categories: Tensor
    edge_categories: Tensor
    node_mask: Tensor
    targets: Tensor
    graph_ids: tuple[str, ...]

    @property
    def x(self) -> Tensor:
        """Categorical node input, including the explicit missing-feature sentinel."""

        return self.node_categories

    @property
    def edge_attr(self) -> Tensor:
        """Undirected-edge-aligned categorical bond input."""

        return self.edge_categories

    def pin_memory(self) -> PaddedChartBatch:
        """Pin tensor fields so ``DataLoader(pin_memory=True)`` can handle the batch."""

        return PaddedChartBatch(
            basis=self.basis.pin_memory(),
            edge_features=self.edge_features.pin_memory(),
            edge_mask=self.edge_mask.pin_memory(),
            cycle_mask=self.cycle_mask.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            node_categories=self.node_categories.pin_memory(),
            edge_categories=self.edge_categories.pin_memory(),
            node_mask=self.node_mask.pin_memory(),
            targets=self.targets.pin_memory(),
            graph_ids=self.graph_ids,
        )

    def to(
        self,
        device: torch.device,
        *,
        pin_memory: bool,
        non_blocking: bool,
    ) -> PaddedChartBatch:
        def move(tensor: Tensor) -> Tensor:
            value = tensor.pin_memory() if pin_memory and not tensor.is_pinned() else tensor
            return value.to(device, non_blocking=non_blocking)

        return PaddedChartBatch(
            basis=move(self.basis),
            edge_features=move(self.edge_features),
            edge_mask=move(self.edge_mask),
            cycle_mask=move(self.cycle_mask),
            edge_index=move(self.edge_index),
            node_categories=move(self.node_categories),
            edge_categories=move(self.edge_categories),
            node_mask=move(self.node_mask),
            targets=move(self.targets),
            graph_ids=self.graph_ids,
        )


@dataclass(frozen=True)
class FitResult:
    model: nn.Module
    target_mean: FloatArray
    target_scale: FloatArray
    history: tuple[dict[str, float], ...]


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _edge_features(record: GraphChartView) -> FloatArray:
    degrees = np.zeros(record.num_nodes, dtype=np.float64)
    for u, v in record.edges:
        degrees[u] += 1.0
        degrees[v] += 1.0
    max_degree = max(1.0, float(degrees.max()))
    num_edges = max(1, len(record.edges))
    result = np.empty((len(record.edges), 4), dtype=np.float64)
    for edge_index, (u, v) in enumerate(record.edges):
        low, high = sorted((degrees[u], degrees[v]))
        result[edge_index] = (
            low / max_degree,
            high / max_degree,
            1.0 / record.num_nodes,
            1.0 / num_edges,
        )
    return result


def collate_chart_views(views: Sequence[GraphChartView]) -> PaddedChartBatch:
    """Pad variable edge/cycle dimensions without exposing padded values."""

    if not views:
        raise ValueError("views must not be empty")
    target_dim = len(views[0].target)
    if target_dim < 1 or any(len(view.target) != target_dim for view in views):
        raise ValueError("all views must have the same positive target dimension")
    max_edges = max(len(view.edges) for view in views)
    max_nodes = max(view.num_nodes for view in views)
    max_beta = max(view.basis.shape[1] for view in views)
    batch_size = len(views)
    basis = torch.zeros((batch_size, max_edges, max_beta), dtype=torch.float32)
    edge_features = torch.zeros((batch_size, max_edges, 4), dtype=torch.float32)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool)
    cycle_mask = torch.zeros((batch_size, max_beta), dtype=torch.bool)
    edge_index = torch.zeros((batch_size, max_edges, 2), dtype=torch.long)
    node_categories = torch.full((batch_size, max_nodes), ZINC_NUM_ATOM_TYPES, dtype=torch.long)
    edge_categories = torch.full((batch_size, max_edges), ZINC_NUM_BOND_TYPES, dtype=torch.long)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    targets = torch.empty((batch_size, target_dim), dtype=torch.float32)
    for batch_index, view in enumerate(views):
        num_edges, beta = view.basis.shape
        if num_edges != len(view.edges):
            raise ValueError("basis edge dimension does not match the physical graph")
        if view.x is not None:
            if len(view.x) != view.num_nodes:
                raise ValueError("categorical node x must have one value per node")
            if any(value < 0 or value >= ZINC_NUM_ATOM_TYPES for value in view.x):
                raise ValueError("categorical node x is outside the supported ZINC range")
            node_categories[batch_index, : view.num_nodes] = torch.as_tensor(
                view.x, dtype=torch.long
            )
        if view.edge_attr is not None:
            if len(view.edge_attr) != num_edges:
                raise ValueError("categorical edge_attr must align with undirected edges")
            if any(value < 0 or value >= ZINC_NUM_BOND_TYPES for value in view.edge_attr):
                raise ValueError("categorical edge_attr is outside the supported ZINC range")
            edge_categories[batch_index, :num_edges] = torch.as_tensor(
                view.edge_attr, dtype=torch.long
            )
        if beta:
            basis[batch_index, :num_edges, :beta] = torch.as_tensor(
                np.array(view.basis, copy=True), dtype=torch.float32
            )
            cycle_mask[batch_index, :beta] = True
        edge_features[batch_index, :num_edges] = torch.as_tensor(
            _edge_features(view), dtype=torch.float32
        )
        edge_index[batch_index, :num_edges] = torch.as_tensor(view.edges, dtype=torch.long)
        edge_mask[batch_index, :num_edges] = True
        node_mask[batch_index, : view.num_nodes] = True
        targets[batch_index] = torch.as_tensor(view.target, dtype=torch.float32)
    return PaddedChartBatch(
        basis=basis,
        edge_features=edge_features,
        edge_mask=edge_mask,
        cycle_mask=cycle_mask,
        edge_index=edge_index,
        node_categories=node_categories,
        edge_categories=edge_categories,
        node_mask=node_mask,
        targets=targets,
        graph_ids=tuple(view.graph_id for view in views),
    )


class VariableBetaCycleEncoder(nn.Module):
    """Orientation-gauge-safe full-beta chart encoder with masked graph readout.

    Each edge sees a set of sign-even cycle-column memberships.  A shared
    coordinate MLP is pooled over valid columns, then a second MLP is pooled
    over valid edges.  The sign-even inputs remove arbitrary edge-orientation
    and fundamental-cycle direction gauges, while the set pooling removes edge
    and cycle-column ordering.  Neither ``max_edges`` nor ``max_beta`` is a
    learned architectural constant.

    This guarantees invariance when the same physical tree is represented with
    another orientation, ordering, or node labeling.  It does not make two
    *different* spanning-tree charts identical: label-dependent BFS/DFS
    preprocessing may still select another tree, which is the chart-shift axis
    measured by this track.
    """

    def __init__(self, *, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if hidden_dim < 4 or output_dim < 1:
            raise ValueError("hidden_dim >= 4 and output_dim >= 1 are required")
        self.coordinate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        chemistry_dim = max(4, hidden_dim // 4)
        self.atom_embedding = nn.Embedding(ZINC_NUM_ATOM_TYPES + 1, chemistry_dim)
        self.bond_embedding = nn.Embedding(ZINC_NUM_BOND_TYPES + 1, chemistry_dim)
        self.edge = nn.Sequential(
            nn.Linear(3 * hidden_dim + 4 + 3 * chemistry_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _masked_max(values: Tensor, mask: Tensor, *, dimension: int) -> Tensor:
        masked = values.masked_fill(~mask, -torch.inf)
        maximum = masked.amax(dim=dimension)
        return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    def forward(self, batch: PaddedChartBatch) -> Tensor:
        basis = batch.basis
        batch_size, max_edges, max_beta = basis.shape
        hidden_dim = self.coordinate[0].out_features
        if max_beta:
            edge_counts = batch.edge_mask.sum(dim=1).clamp_min(1)[:, None]
            normalized_cycle_support = basis.abs().sum(dim=1) / edge_counts
            normalized_cycle_support = normalized_cycle_support[:, None, :].expand(
                batch_size, max_edges, max_beta
            )
            coordinate_input = torch.stack(
                (basis.abs(), basis.square(), normalized_cycle_support), dim=-1
            )
            coordinate_hidden = self.coordinate(coordinate_input)
            coordinate_mask = (batch.edge_mask[:, :, None] & batch.cycle_mask[:, None, :])[
                :, :, :, None
            ]
            coordinate_hidden = coordinate_hidden * coordinate_mask
            count = coordinate_mask.sum(dim=2).clamp_min(1)
            coordinate_sum = coordinate_hidden.sum(dim=2)
            coordinate_mean = coordinate_sum / count
            coordinate_max = self._masked_max(
                coordinate_hidden,
                coordinate_mask,
                dimension=2,
            )
        else:
            zeros = basis.new_zeros((batch_size, max_edges, hidden_dim))
            coordinate_sum = zeros
            coordinate_mean = zeros
            coordinate_max = zeros
        atom_hidden = self.atom_embedding(batch.node_categories)
        atom_hidden = atom_hidden * batch.node_mask[:, :, None]
        batch_indices = torch.arange(batch_size, device=basis.device)[:, None]
        start = batch.edge_index[:, :, 0]
        end = batch.edge_index[:, :, 1]
        start_atom = atom_hidden[batch_indices, start]
        end_atom = atom_hidden[batch_indices, end]
        bond_hidden = self.bond_embedding(batch.edge_categories)
        chemistry = torch.cat(
            (start_atom + end_atom, (start_atom - end_atom).abs(), bond_hidden), dim=-1
        )
        edge_input = torch.cat(
            (
                coordinate_sum,
                coordinate_mean,
                coordinate_max,
                batch.edge_features,
                chemistry,
            ),
            dim=-1,
        )
        edge_hidden = self.edge(edge_input)
        edge_mask = batch.edge_mask[:, :, None]
        edge_hidden = edge_hidden * edge_mask
        edge_count = edge_mask.sum(dim=1).clamp_min(1)
        edge_sum = edge_hidden.sum(dim=1)
        edge_mean = edge_sum / edge_count
        edge_max = self._masked_max(edge_hidden, edge_mask, dimension=1)
        return self.head(torch.cat((edge_sum, edge_mean, edge_max), dim=-1))


def build_chart_views(
    records: Sequence[GraphRecord],
    *,
    chart_status: str,
    count: int,
    methods: Sequence[str],
    seed: int,
    roots: Sequence[int] | None = None,
    exclude_by_graph: Mapping[str, set[tuple[int, ...]]] | None = None,
    require_distinct: bool = False,
) -> list[GraphChartView]:
    """Generate chart views only after the physical graph split is fixed."""

    views: list[GraphChartView] = []
    for record in records:
        graph_seed = _stable_seed(f"{chart_status}:{record.graph_id}", seed)
        charts = sample_paper_charts(
            record,
            count=count,
            methods=methods,
            seed=graph_seed,
            roots=roots,
            exclude=(exclude_by_graph or {}).get(record.graph_id, set()),
            require_distinct=require_distinct,
        )
        graph_status = "ood" if record.split == "ood_test" else "id"
        for chart in charts:
            views.append(
                GraphChartView(
                    graph_id=record.graph_id,
                    graph_family=record.family,
                    graph_status=graph_status,
                    chart_status=chart_status,
                    num_nodes=record.num_nodes,
                    edges=record.edges,
                    basis=chart.basis,
                    target=record.target,
                    chart_name=chart.name,
                    tree_key=chart_key(chart),
                    x=record.x,
                    edge_attr=record.edge_attr,
                )
            )
    return views


def _unique_graph_targets(views: Sequence[GraphChartView]) -> FloatArray:
    targets: dict[str, tuple[float, ...]] = {}
    for view in views:
        previous = targets.setdefault(view.graph_id, view.target)
        if previous != view.target:
            raise ValueError("one graph_id was assigned conflicting downstream targets")
    return np.asarray(list(targets.values()), dtype=np.float64)


def fit_downstream_model(
    views: Sequence[GraphChartView],
    *,
    task_type: str,
    output_dim: int,
    hidden_dim: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> FitResult:
    """Fit with a fixed number of optimizer updates for fair chart comparisons."""

    if not views:
        raise ValueError("training views must not be empty")
    if updates < 1 or batch_size < 1 or workers < 0:
        raise ValueError("updates/batch_size must be positive and workers non-negative")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")
    use_amp = bool(amp and device.type == "cuda")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = VariableBetaCycleEncoder(hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_grad_scaler = getattr(torch.amp, "GradScaler", None)
    if amp_grad_scaler is not None:
        scaler = amp_grad_scaler("cuda", enabled=use_amp)
    else:  # pragma: no cover - compatibility with the minimum supported torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    graph_targets = _unique_graph_targets(views)
    if task_type == "regression":
        target_mean = graph_targets.mean(axis=0)
        target_scale = graph_targets.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
    elif task_type == "classification":
        target_mean = np.zeros(1, dtype=np.float64)
        target_scale = np.ones(1, dtype=np.float64)
    else:
        raise ValueError("task_type must be regression or classification")
    mean_tensor = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(target_scale, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 101)
    sampled_indices = torch.randint(len(views), (updates, batch_size), generator=generator).tolist()
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 202)
    loader = DataLoader(
        list(views),
        batch_sampler=sampled_indices,
        collate_fn=collate_chart_views,
        num_workers=workers,
        pin_memory=pin_memory and device.type == "cuda",
        generator=loader_generator,
    )
    history: list[dict[str, float]] = []
    model.train()
    for update, cpu_batch in enumerate(loader, start=1):
        batch = cpu_batch.to(
            device,
            pin_memory=pin_memory and device.type == "cuda",
            non_blocking=non_blocking and device.type == "cuda",
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(batch)
            if task_type == "classification":
                loss = nn.functional.cross_entropy(prediction, batch.targets[:, 0].to(torch.long))
            else:
                normalized = (batch.targets - mean_tensor) / scale_tensor
                loss = nn.functional.mse_loss(prediction, normalized)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if update == 1 or update == updates or update % max(1, updates // 10) == 0:
            history.append({"update": float(update), "loss": float(loss.detach().cpu())})
    return FitResult(
        model=model,
        target_mean=np.asarray(target_mean, dtype=np.float64),
        target_scale=np.asarray(target_scale, dtype=np.float64),
        history=tuple(history),
    )


@torch.no_grad()
def _predict(
    fitted: FitResult,
    views: Sequence[GraphChartView],
    *,
    task_type: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> FloatArray:
    fitted.model.eval()
    use_amp = bool(amp and device.type == "cuda")
    predictions: list[FloatArray] = []
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(0)
    loader = DataLoader(
        list(views),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_chart_views,
        num_workers=workers,
        pin_memory=pin_memory and device.type == "cuda",
        generator=loader_generator,
    )
    for cpu_batch in loader:
        batch = cpu_batch.to(
            device,
            pin_memory=pin_memory and device.type == "cuda",
            non_blocking=non_blocking and device.type == "cuda",
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = fitted.model(batch)
        values = output.float().cpu().numpy().astype(np.float64, copy=False)
        if task_type == "regression":
            values = values * fitted.target_scale + fitted.target_mean
        predictions.append(values)
    return np.concatenate(predictions, axis=0)


def _group_indices(views: Sequence[GraphChartView]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, view in enumerate(views):
        groups.setdefault(view.graph_id, []).append(index)
    return groups


def _regression_metrics(
    views: Sequence[GraphChartView], predictions: FloatArray, target_scale: FloatArray
) -> dict[str, float]:
    targets = np.asarray([view.target for view in views], dtype=np.float64)
    errors = np.abs(predictions - targets)
    view_mae = errors.mean(axis=1)
    graph_macro = []
    graph_worst = []
    chart_std = []
    flip_rates = []
    for indices in _group_indices(views).values():
        selected = np.asarray(indices, dtype=np.int64)
        graph_macro.append(float(view_mae[selected].mean()))
        graph_worst.append(float(view_mae[selected].max()))
        chart_std.append(float(predictions[selected].std(axis=0).mean()))
        rounded = np.rint(predictions[selected]).astype(np.int64)
        flip_rates.append(float(np.mean(np.any(rounded != rounded[:1], axis=1))))
    safe_scale = np.where(target_scale < 1e-6, 1.0, target_scale)
    return {
        "mae": float(errors.mean()),
        "normalized_mae": float((errors / safe_scale).mean()),
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "graph_macro_mae": float(np.mean(graph_macro)),
        "worst_chart_mae": float(np.mean(graph_worst)),
        "chart_prediction_std": float(np.mean(chart_std)),
        "prediction_flip_rate": float(np.mean(flip_rates)),
        "rounded_exact_vector_accuracy": float(
            np.mean(np.all(np.rint(predictions) == targets, axis=1))
        ),
    }


def _classification_metrics(
    views: Sequence[GraphChartView], logits: FloatArray
) -> dict[str, float]:
    targets = np.asarray([int(view.target[0]) for view in views], dtype=np.int64)
    predictions = logits.argmax(axis=1)
    correct = predictions == targets
    graph_accuracy = []
    graph_worst = []
    flip_rates = []
    probability_std = []
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    for indices in _group_indices(views).values():
        selected = np.asarray(indices, dtype=np.int64)
        graph_accuracy.append(float(correct[selected].mean()))
        graph_worst.append(float(correct[selected].min()))
        flip_rates.append(float(np.mean(predictions[selected] != predictions[selected[0]])))
        probability_std.append(float(probabilities[selected].std(axis=0).mean()))
    return {
        "accuracy": float(correct.mean()),
        "graph_macro_accuracy": float(np.mean(graph_accuracy)),
        "worst_chart_accuracy": float(np.mean(graph_worst)),
        "chart_probability_std": float(np.mean(probability_std)),
        "prediction_flip_rate": float(np.mean(flip_rates)),
    }


def evaluate_downstream_model(
    fitted: FitResult,
    views: Sequence[GraphChartView],
    *,
    task_type: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> dict[str, float]:
    if not views:
        raise ValueError("evaluation views must not be empty")
    predictions = _predict(
        fitted,
        views,
        task_type=task_type,
        batch_size=batch_size,
        device=device,
        amp=amp,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        workers=workers,
    )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("model produced non-finite predictions")
    if task_type == "classification":
        return _classification_metrics(views, predictions)
    return _regression_metrics(views, predictions, fitted.target_scale)


def run_fixed_vs_multichart(
    *,
    fixed_train_views: Sequence[GraphChartView],
    multi_train_views: Sequence[GraphChartView],
    evaluation_views: Mapping[str, Sequence[GraphChartView]],
    task_type: str,
    output_dim: int,
    hidden_dim: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> tuple[dict[str, Any], dict[str, FitResult]]:
    """Train fair fixed/multi models and evaluate every requested quadrant."""

    common = {
        "task_type": task_type,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": device,
        "seed": seed,
        "amp": amp,
        "pin_memory": pin_memory,
        "non_blocking": non_blocking,
        "workers": workers,
    }
    fixed = fit_downstream_model(fixed_train_views, **common)
    multi = fit_downstream_model(multi_train_views, **common)
    models = {"fixed_bfs": fixed, "multi_chart": multi}
    metrics: dict[str, Any] = {}
    for model_name, fitted in models.items():
        metrics[model_name] = {
            "optimizer_updates": updates,
            "num_training_views": (
                len(fixed_train_views) if model_name == "fixed_bfs" else len(multi_train_views)
            ),
            "history": list(fitted.history),
            "quadrants": {},
        }
        for quadrant, views in evaluation_views.items():
            values = evaluate_downstream_model(
                fitted,
                views,
                task_type=task_type,
                batch_size=batch_size,
                device=device,
                amp=amp,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                workers=workers,
            )
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"non-finite metric in {model_name}/{quadrant}")
            metrics[model_name]["quadrants"][quadrant] = values
    return metrics, models


__all__ = [
    "FitResult",
    "GraphChartView",
    "PaddedChartBatch",
    "VariableBetaCycleEncoder",
    "build_chart_views",
    "collate_chart_views",
    "evaluate_downstream_model",
    "fit_downstream_model",
    "run_fixed_vs_multichart",
]
