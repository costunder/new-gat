"""Sparse, variable-graph incidence conductance operators.

This module deliberately never materializes an incidence matrix.  For an
oriented edge ``tail -> head`` it evaluates ``g = H[head] - H[tail]`` and
implements ``B.T q`` with two ``index_add_`` calls.  Concatenating graphs is
therefore just concatenating nodes/edges and offsetting ``edge_index``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf


def _inverse_softplus(value: float) -> float:
    x = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(x)))


def edge_gradient(edge_index: Tensor, node_state: Tensor) -> Tensor:
    """Return oriented edge differences without constructing ``B``."""

    _validate_edge_index(edge_index, node_state.shape[0])
    tail, head = edge_index
    return node_state.index_select(0, head) - node_state.index_select(0, tail)


def edge_divergence(edge_index: Tensor, edge_flux: Tensor, num_nodes: int) -> Tensor:
    """Return ``B.T @ edge_flux`` using CUDA-safe indexed accumulation."""

    _validate_edge_index(edge_index, num_nodes)
    if edge_flux.ndim != 2 or edge_flux.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_flux must have shape (num_edges, channels)")
    tail, head = edge_index
    result = edge_flux.new_zeros((num_nodes, edge_flux.shape[1]))
    result.index_add_(0, head, edge_flux)
    result.index_add_(0, tail, -edge_flux)
    return result


def weighted_degree(edge_index: Tensor, conductance: Tensor, num_nodes: int) -> Tensor:
    """Weighted undirected degree for one scalar conductance per edge."""

    _validate_edge_index(edge_index, num_nodes)
    values = conductance.reshape(-1)
    if values.shape[0] != edge_index.shape[1]:
        raise ValueError("conductance must contain one value per edge")
    result = values.new_zeros(num_nodes)
    result.index_add_(0, edge_index[0], values)
    result.index_add_(0, edge_index[1], values)
    return result


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a long tensor with shape (2, num_edges)")
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError("edge_index contains a node outside node_state")
        if torch.any(edge_index[0] == edge_index[1]):
            raise ValueError("incidence conductance edges cannot be self-loops")


def _scatter_graph_max(values: Tensor, graph_index: Tensor, num_graphs: int) -> Tensor:
    """Per-graph max with a torch-only CUDA implementation."""

    output = values.new_full((num_graphs,), -torch.inf)
    if hasattr(output, "scatter_reduce_"):
        output.scatter_reduce_(0, graph_index, values, reduce="amax", include_self=True)
    else:  # pragma: no cover - only reached on obsolete PyTorch versions
        for graph_id in range(num_graphs):
            selected = values[graph_index == graph_id]
            if selected.numel():
                output[graph_id] = selected.max()
    return output.masked_fill(torch.isinf(output), 0.0)


@dataclass
class PackedGraphBatch:
    """A dependency-free variable-graph mini-batch.

    Targets are optional so the same container can be used for inference and
    public benchmark adapters.  Every tensor is flat over all nodes or all
    edges; ``node_graph`` and ``edge_graph`` identify the owning graph.
    """

    node_state: Tensor
    edge_index: Tensor
    edge_features: Tensor
    node_graph: Tensor
    edge_graph: Tensor
    graph_ids: list[str]
    requested_step: Tensor
    true_conductance: Tensor | None = None
    true_gradient: Tensor | None = None
    true_flux: Tensor | None = None
    true_node_message: Tensor | None = None
    true_next_state: Tensor | None = None
    observed_flux: Tensor | None = None
    observed_node_message: Tensor | None = None
    metadata: list[dict[str, Any]] | None = None

    @property
    def num_graphs(self) -> int:
        return len(self.graph_ids)

    @property
    def num_nodes(self) -> int:
        return int(self.node_state.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PackedGraphBatch:
        values: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            values[name] = (
                value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
            )
        return PackedGraphBatch(**values)

    def pin_memory(self) -> PackedGraphBatch:
        values: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            values[name] = value.pin_memory() if isinstance(value, Tensor) else value
        return PackedGraphBatch(**values)


def pack_graph_examples(examples: Iterable[Mapping[str, Any]]) -> PackedGraphBatch:
    """Pack graph dictionaries while offsetting edges exactly once."""

    records = list(examples)
    if not records:
        raise ValueError("cannot pack an empty example list")
    node_states: list[Tensor] = []
    edge_indices: list[Tensor] = []
    edge_features: list[Tensor] = []
    node_graph: list[Tensor] = []
    edge_graph: list[Tensor] = []
    graph_ids: list[str] = []
    steps: list[float] = []
    metadata: list[dict[str, Any]] = []
    optional_names = (
        "true_conductance",
        "true_gradient",
        "true_flux",
        "true_node_message",
        "true_next_state",
        "observed_flux",
        "observed_node_message",
    )
    optional: dict[str, list[Tensor]] = {name: [] for name in optional_names}
    node_offset = 0
    feature_width: int | None = None
    channels: int | None = None
    for graph_number, record in enumerate(records):
        state = record["node_state"]
        edges = record["edge_index"]
        features = record["edge_features"]
        if state.ndim != 2 or features.ndim != 2:
            raise ValueError("node_state and edge_features must be matrices")
        _validate_edge_index(edges, state.shape[0])
        if edges.shape[1] != features.shape[0]:
            raise ValueError("edge_index and edge_features disagree on edge count")
        if channels is None:
            channels = int(state.shape[1])
            feature_width = int(features.shape[1])
        if state.shape[1] != channels or features.shape[1] != feature_width:
            raise ValueError("all examples in a batch need equal feature widths")
        node_states.append(state)
        edge_indices.append(edges + node_offset)
        edge_features.append(features)
        node_graph.append(torch.full((state.shape[0],), graph_number, dtype=torch.long))
        edge_graph.append(torch.full((edges.shape[1],), graph_number, dtype=torch.long))
        graph_ids.append(str(record.get("graph_id", graph_number)))
        steps.append(float(record.get("step_size", 0.02)))
        metadata.append(dict(record.get("metadata", {})))
        for name in optional_names:
            value = record.get(name)
            if value is not None:
                optional[name].append(value)
        node_offset += int(state.shape[0])
    for name, values in optional.items():
        if values and len(values) != len(records):
            raise ValueError(f"optional target {name!r} must be present for every example")
    packed_optional = {
        name: torch.cat(values, dim=0) if values else None for name, values in optional.items()
    }
    return PackedGraphBatch(
        node_state=torch.cat(node_states, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat(edge_features, dim=0),
        node_graph=torch.cat(node_graph, dim=0),
        edge_graph=torch.cat(edge_graph, dim=0),
        graph_ids=graph_ids,
        requested_step=torch.tensor(steps, dtype=node_states[0].dtype),
        metadata=metadata,
        **packed_optional,
    )


class SparsePositiveConductance(nn.Module):
    """Positive orientation-invariant full, static, or gradient-only edge law."""

    def __init__(
        self,
        channels: int,
        edge_feature_channels: int,
        hidden_channels: int = 48,
        minimum: float = 1.0e-5,
        mode: str = "full",
    ) -> None:
        super().__init__()
        if mode not in {"full", "edge_only", "gradient_only"}:
            raise ValueError("mode must be full, edge_only, or gradient_only")
        if channels < 1 or edge_feature_channels < 0 or hidden_channels < 1 or minimum <= 0:
            raise ValueError("invalid conductance dimensions or minimum")
        self.channels = int(channels)
        self.edge_feature_channels = int(edge_feature_channels)
        self.minimum = float(minimum)
        self.mode = mode
        if mode == "full":
            width = edge_feature_channels + 2 * channels
        elif mode == "gradient_only":
            width = channels
        else:
            width = edge_feature_channels
        if width == 0:
            raise ValueError("edge_only conductance requires edge features")
        self.network = nn.Sequential(
            nn.Linear(width, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, gradient: Tensor, edge_features: Tensor) -> Tensor:
        if gradient.ndim != 2 or gradient.shape[1] != self.channels:
            raise ValueError("gradient width differs from configured channels")
        if edge_features.shape != (gradient.shape[0], self.edge_feature_channels):
            raise ValueError("edge feature shape differs from configured shape")
        pieces = [edge_features]
        if self.mode == "full":
            pieces = [gradient.abs(), gradient.square(), edge_features]
        elif self.mode == "gradient_only":
            pieces = [gradient.abs()]
        raw = self.network(torch.cat(pieces, dim=-1))
        return nnf.softplus(raw).squeeze(-1) + self.minimum


class SparseIncidenceConductanceLayer(nn.Module):
    """Dense-``B``-free ``H - eta B.T C B H`` on packed variable graphs."""

    def __init__(
        self,
        channels: int,
        edge_feature_channels: int,
        hidden_channels: int = 48,
        minimum_conductance: float = 1.0e-5,
        requested_step: float = 0.02,
        stability_margin: float = 0.95,
        adaptive_stability: bool = True,
        mode: str = "full",
        initial_isotropic: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"full", "edge_only", "gradient_only", "isotropic"}:
            raise ValueError("mode must be full, edge_only, gradient_only, or isotropic")
        if requested_step <= 0 or not 0 < stability_margin < 1:
            raise ValueError("requested_step and stability_margin are invalid")
        self.channels = int(channels)
        self.edge_feature_channels = int(edge_feature_channels)
        self.requested_step = float(requested_step)
        self.stability_margin = float(stability_margin)
        self.adaptive_stability = bool(adaptive_stability)
        self.mode = mode
        self.minimum_conductance = float(minimum_conductance)
        if mode == "isotropic":
            if initial_isotropic <= minimum_conductance:
                raise ValueError("initial isotropic value must exceed the minimum")
            raw = _inverse_softplus(initial_isotropic - minimum_conductance)
            self.raw_isotropic = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
            self.estimator = None
        else:
            self.estimator = SparsePositiveConductance(
                channels,
                edge_feature_channels,
                hidden_channels,
                minimum_conductance,
                mode,
            )

    @property
    def isotropic_conductance(self) -> Tensor:
        if self.mode != "isotropic":
            raise AttributeError("only the isotropic baseline has a scalar conductance")
        return nnf.softplus(self.raw_isotropic) + self.minimum_conductance

    def forward(
        self,
        batch: PackedGraphBatch,
        *,
        node_state: Tensor | None = None,
        conductance_override: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        state = batch.node_state if node_state is None else node_state
        if state.ndim != 2 or state.shape != batch.node_state.shape:
            raise ValueError("node_state must match the packed batch shape")
        if state.shape[1] != self.channels:
            raise ValueError("node-state width differs from configured channels")
        gradient = edge_gradient(batch.edge_index, state)
        if conductance_override is not None:
            conductance = conductance_override.to(device=state.device, dtype=state.dtype).reshape(
                -1
            )
            if conductance.shape[0] != batch.num_edges:
                raise ValueError("conductance_override needs one scalar per edge")
        elif self.mode == "isotropic":
            conductance = self.isotropic_conductance.to(state).expand(batch.num_edges)
        else:
            assert self.estimator is not None
            conductance = self.estimator(gradient, batch.edge_features.to(state))
        flux = conductance[:, None] * gradient
        message = edge_divergence(batch.edge_index, flux, batch.num_nodes)
        degree = weighted_degree(batch.edge_index, conductance, batch.num_nodes)
        max_degree = _scatter_graph_max(degree, batch.node_graph, batch.num_graphs)
        requested = batch.requested_step.to(state)
        if requested.numel() != batch.num_graphs:
            requested = state.new_full((batch.num_graphs,), self.requested_step)
        if self.adaptive_stability:
            safe = self.stability_margin / max_degree.clamp_min(torch.finfo(state.dtype).eps)
            step = torch.minimum(requested, safe)
        else:
            step = requested
        next_state = state - step.index_select(0, batch.node_graph)[:, None] * message
        if not return_diagnostics:
            return next_state
        return next_state, {
            "edge_gradient": gradient,
            "conductance": conductance,
            "edge_flux": flux,
            "node_message": message,
            "effective_step": step,
            "cap_active": step < requested,
            "max_weighted_degree": max_degree,
        }


__all__ = [
    "PackedGraphBatch",
    "SparseIncidenceConductanceLayer",
    "SparsePositiveConductance",
    "edge_divergence",
    "edge_gradient",
    "pack_graph_examples",
    "weighted_degree",
]
