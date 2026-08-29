"""Batch-safe neural models for the static cycle-PE paper experiments.

The paper path deliberately keeps graph batches ragged.  Every graph may have
a different edge count and cycle rank; raw bases are padded to the maximum
rank fitted on the training split only, never to a fixed constant or a width
selected from validation/test graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chartgat.algebra import incidence_matrix
from chartgat.graphs import spanning_tree_indices
from research.cycle_pe.features import (
    SET_STAT_NAMES,
    cycle_projector,
    cycle_set_statistics,
    static_fundamental_basis,
)
from research.cycle_pe.paper_data import PaperGraph, structural_input_features

PE_VARIANTS = ("no_pe", "raw", "set", "projector")


class RawCycleRankOverflow(RuntimeError):
    """Raised rather than silently truncating an OOD raw cycle basis."""

    def __init__(self, actual_rank: int, fitted_width: int) -> None:
        self.actual_rank = int(actual_rank)
        self.fitted_width = int(fitted_width)
        super().__init__(f"cycle rank {actual_rank} exceeds train-fitted raw width {fitted_width}")


@dataclass
class PreparedGraph:
    """One tensorized graph with only the requested expensive PE representations."""

    graph_id: str
    split: str
    family: str
    num_nodes: int
    cycle_rank: int
    edges: Tensor
    node_features: Tensor
    edge_features: Tensor
    raw_basis: Tensor
    cycle_set: Tensor | None
    projector: Tensor | None
    edge_targets: Tensor | None
    node_targets: Tensor | None
    graph_targets: Tensor | None

    def pin_memory(self) -> PreparedGraph:
        """Support recursive DataLoader pinning for ragged graph objects."""

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                setattr(self, field.name, value.pin_memory())
        return self

    def to(self, device: torch.device, *, non_blocking: bool = False) -> PreparedGraph:
        values: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device=device, non_blocking=non_blocking)
                if isinstance(value, Tensor)
                else value
            )
        return PreparedGraph(**values)  # type: ignore[arg-type]


def infer_raw_width(graphs: list[PaperGraph]) -> int:
    """Infer a lossless raw-basis width from arbitrary cycle ranks."""

    if not graphs:
        raise ValueError("cannot infer a raw width from an empty graph list")
    ranks = [graph.beta for graph in graphs]
    if any(rank < 0 for rank in ranks):
        raise ValueError("all paper graphs must be connected")
    return max(ranks)


def prepare_graph(
    graph: PaperGraph,
    *,
    required_variants: tuple[str, ...] = PE_VARIANTS,
) -> PreparedGraph:
    """Extract only requested topology PEs and convert one graph to CPU tensors."""

    unknown = set(required_variants) - set(PE_VARIANTS)
    if unknown:
        raise ValueError(f"unknown PE variants: {sorted(unknown)}")

    edge_list = list(graph.edges)
    incidence = incidence_matrix(graph.num_nodes, edge_list)
    tree = spanning_tree_indices(graph.num_nodes, edge_list, mode="bfs")
    basis = static_fundamental_basis(incidence, tree)
    if basis.shape[1] != graph.beta:
        raise RuntimeError("cycle-rank mismatch while preparing graph")
    node_features, edge_features = structural_input_features(graph)

    def float_tensor(value: np.ndarray | None) -> Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(np.asarray(value), dtype=torch.float32).contiguous()

    return PreparedGraph(
        graph_id=graph.graph_id,
        split=graph.split,
        family=graph.family,
        num_nodes=graph.num_nodes,
        cycle_rank=graph.beta,
        edges=torch.as_tensor(edge_list, dtype=torch.long).reshape(-1, 2).contiguous(),
        node_features=torch.as_tensor(node_features, dtype=torch.float32).contiguous(),
        edge_features=torch.as_tensor(edge_features, dtype=torch.float32).contiguous(),
        # Keep every coordinate. The raw encoder pads only ranks that fit the
        # train-derived width and raises explicitly on OOD overflow.
        raw_basis=torch.as_tensor(basis, dtype=torch.float32).contiguous(),
        cycle_set=(
            torch.as_tensor(cycle_set_statistics(basis), dtype=torch.float32).contiguous()
            if "set" in required_variants
            else None
        ),
        projector=(
            torch.as_tensor(cycle_projector(basis), dtype=torch.float32).contiguous()
            if "projector" in required_variants
            else None
        ),
        edge_targets=float_tensor(graph.edge_targets),
        node_targets=float_tensor(graph.node_targets),
        graph_targets=float_tensor(graph.graph_targets),
    )


def prepare_splits(
    splits: dict[str, list[PaperGraph]],
    *,
    fit_split: str | None = None,
    required_variants: tuple[str, ...] = PE_VARIANTS,
) -> tuple[dict[str, list[PreparedGraph]], int]:
    """Prepare all splits with a raw width fitted on training data only."""

    if not splits:
        raise ValueError("splits cannot be empty")
    selected = fit_split or ("train" if "train" in splits else next(iter(splits)))
    if selected not in splits:
        raise ValueError(f"fit_split {selected!r} is not present")
    raw_width = infer_raw_width(splits[selected])
    prepared = {
        split: [prepare_graph(graph, required_variants=required_variants) for graph in graphs]
        for split, graphs in splits.items()
    }
    return prepared, raw_width


class StaticPEEncoder(nn.Module):
    """Map any supported static PE to a common per-edge representation.

    The projector path is a row-wise DeepSets encoder over the full intrinsic
    cycle-space projector.  It is independent of cycle rank and invariant to
    every invertible change of cycle basis.  Absolute pair entries additionally
    remove arbitrary incidence-orientation signs.
    """

    def __init__(
        self,
        variant: str,
        *,
        raw_width: int,
        pe_dim: int,
    ) -> None:
        super().__init__()
        if variant not in PE_VARIANTS:
            raise ValueError(f"variant must be one of {PE_VARIANTS}")
        if raw_width < 0 or pe_dim < 1:
            raise ValueError("raw_width must be non-negative and pe_dim positive")
        self.variant = variant
        self.raw_width = raw_width
        self.pe_dim = pe_dim
        self.raw_encoder = (
            nn.Sequential(nn.Linear(raw_width, pe_dim), nn.GELU()) if raw_width else None
        )
        self.empty_raw = nn.Parameter(torch.zeros(pe_dim))
        self.set_encoder = nn.Sequential(
            nn.Linear(len(SET_STAT_NAMES), pe_dim),
            nn.GELU(),
            nn.Linear(pe_dim, pe_dim),
        )
        projector_hidden = max(8, pe_dim)
        self.projector_pair = nn.Sequential(
            nn.Linear(3, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, projector_hidden),
            nn.GELU(),
        )
        self.projector_row = nn.Sequential(
            nn.Linear(2 * projector_hidden + 2, pe_dim),
            nn.GELU(),
            nn.Linear(pe_dim, pe_dim),
        )

    def forward(
        self,
        raw_basis: Tensor,
        cycle_set: Tensor | None,
        projector: Tensor | None,
    ) -> Tensor:
        edge_count = raw_basis.shape[0]
        if self.variant == "no_pe":
            return raw_basis.new_zeros((edge_count, self.pe_dim))
        if self.variant == "raw":
            actual_rank = raw_basis.shape[1]
            if actual_rank > self.raw_width:
                raise RawCycleRankOverflow(actual_rank, self.raw_width)
            if actual_rank < self.raw_width:
                raw_basis = F.pad(raw_basis, (0, self.raw_width - actual_rank))
            if self.raw_encoder is None:
                return self.empty_raw.unsqueeze(0).expand(edge_count, -1)
            return self.raw_encoder(raw_basis)
        if self.variant == "set":
            if cycle_set is None:
                raise ValueError("set PE was not prepared for this graph")
            return self.set_encoder(cycle_set)

        if projector is None:
            raise ValueError("projector PE was not prepared for this graph")
        if projector.shape != (edge_count, edge_count):
            raise ValueError("projector must have shape (num_edges, num_edges)")
        if edge_count == 0:
            return raw_basis.new_zeros((0, self.pe_dim))
        absolute = projector.abs()
        diagonal = projector.diagonal().abs()
        pair_features = torch.stack(
            (
                absolute,
                absolute.square(),
                diagonal.unsqueeze(0).expand(edge_count, -1),
            ),
            dim=-1,
        )
        encoded = self.projector_pair(pair_features)
        mean = encoded.mean(dim=1)
        maximum = encoded.amax(dim=1)
        row_features = torch.cat(
            (mean, maximum, diagonal[:, None], absolute.mean(dim=1, keepdim=True)),
            dim=1,
        )
        return self.projector_row(row_features)


class GraphOutput(NamedTuple):
    edge: Tensor | None
    node: Tensor | None
    graph: Tensor | None
    embedding: Tensor


class _MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        u, v = edge_index[:, 0], edge_index[:, 1]
        symmetric = torch.cat((node[u] + node[v], (node[u] - node[v]).abs(), edge), dim=1)
        updated_edge = self.edge_norm(edge + self.edge_update(symmetric))

        source = torch.cat((u, v), dim=0)
        target = torch.cat((v, u), dim=0)
        directed_edge = torch.cat((updated_edge, updated_edge), dim=0)
        messages = self.message(torch.cat((node[source], node[target], directed_edge), dim=1))
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(node.shape[0], device=node.device, dtype=node.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=node.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)[:, None]
        updated_node = self.node_norm(node + self.node_update(torch.cat((node, aggregate), dim=1)))
        return updated_node, updated_edge


class PaperCycleModel(nn.Module):
    """Small edge-aware GNN shared by CycleCount-OOD, ZINC, and BREC."""

    def __init__(
        self,
        *,
        variant: str,
        raw_width: int,
        node_input_dim: int,
        edge_input_dim: int,
        edge_output_dim: int,
        node_output_dim: int,
        graph_output_dim: int,
        hidden_dim: int = 64,
        pe_dim: int = 32,
        layers: int = 3,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim < 4 or layers < 1:
            raise ValueError("hidden_dim must be >=4 and layers positive")
        self.pe_encoder = StaticPEEncoder(variant, raw_width=raw_width, pe_dim=pe_dim)
        self.node_encoder = nn.Sequential(nn.Linear(node_input_dim, hidden_dim), nn.GELU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_input_dim + pe_dim, hidden_dim), nn.GELU())
        self.layers = nn.ModuleList(_MessageLayer(hidden_dim) for _ in range(layers))
        self.edge_head = nn.Linear(hidden_dim, edge_output_dim) if edge_output_dim else None
        self.node_head = nn.Linear(hidden_dim, node_output_dim) if node_output_dim else None
        pooled_dim = 4 * hidden_dim
        self.graph_trunk = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.graph_head = nn.Linear(hidden_dim, graph_output_dim) if graph_output_dim else None
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)

    def forward_graph(self, graph: PreparedGraph) -> GraphOutput:
        return self.forward([graph])[0]

    def forward(self, graphs: list[PreparedGraph]) -> list[GraphOutput]:
        if not graphs:
            return []
        node_counts = [graph.node_features.shape[0] for graph in graphs]
        edge_counts = [graph.edge_features.shape[0] for graph in graphs]
        positional = [
            self.pe_encoder(graph.raw_basis, graph.cycle_set, graph.projector) for graph in graphs
        ]
        node = self.node_encoder(torch.cat([graph.node_features for graph in graphs], dim=0))
        edge = self.edge_encoder(
            torch.cat(
                [
                    torch.cat((graph.edge_features, pe), dim=1)
                    for graph, pe in zip(graphs, positional, strict=True)
                ],
                dim=0,
            )
        )
        offsets: list[int] = []
        running = 0
        for count in node_counts:
            offsets.append(running)
            running += count
        edge_index = torch.cat(
            [graph.edges + offset for graph, offset in zip(graphs, offsets, strict=True)],
            dim=0,
        )
        for layer in self.layers:
            node, edge = layer(node, edge, edge_index)
        node_parts = list(torch.split(node, node_counts, dim=0))
        edge_parts = list(torch.split(edge, edge_counts, dim=0))
        pooled_rows: list[Tensor] = []
        for node_part, edge_part in zip(node_parts, edge_parts, strict=True):
            if edge_part.shape[0]:
                edge_mean = edge_part.mean(dim=0)
                edge_maximum = edge_part.amax(dim=0)
            else:
                edge_mean = node_part.new_zeros(node_part.shape[1])
                edge_maximum = node_part.new_zeros(node_part.shape[1])
            pooled_rows.append(
                torch.cat(
                    (
                        node_part.mean(dim=0),
                        node_part.amax(dim=0),
                        edge_mean,
                        edge_maximum,
                    ),
                    dim=0,
                )
            )
        graph_state = self.graph_trunk(torch.stack(pooled_rows))
        edge_prediction = None if self.edge_head is None else self.edge_head(edge)
        node_prediction = None if self.node_head is None else self.node_head(node)
        graph_prediction = None if self.graph_head is None else self.graph_head(graph_state)
        embedding = self.embedding_head(graph_state)
        edge_outputs = (
            [None] * len(graphs)
            if edge_prediction is None
            else list(torch.split(edge_prediction, edge_counts, dim=0))
        )
        node_outputs = (
            [None] * len(graphs)
            if node_prediction is None
            else list(torch.split(node_prediction, node_counts, dim=0))
        )
        return [
            GraphOutput(
                edge=edge_outputs[index],
                node=node_outputs[index],
                graph=None if graph_prediction is None else graph_prediction[index],
                embedding=embedding[index],
            )
            for index in range(len(graphs))
        ]


__all__ = [
    "GraphOutput",
    "PE_VARIANTS",
    "PaperCycleModel",
    "PreparedGraph",
    "RawCycleRankOverflow",
    "StaticPEEncoder",
    "infer_raw_width",
    "prepare_graph",
    "prepare_splits",
]
