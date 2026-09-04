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


@dataclass
class PreparedBatch:
    """One packed disjoint-union batch with a bounded number of H2D copies.

    Ragged padding happens in the DataLoader collate process on CPU.  The
    learned PE encoders and message-passing stack then see concatenated
    tensors and execute once per physical minibatch, never once per graph.
    Projectors stay packed as ``sum(E_g ** 2)`` values rather than being
    padded to ``batch_size * max(E_g) ** 2``.
    """

    graph_ids: tuple[str, ...]
    node_counts: tuple[int, ...]
    edge_counts: tuple[int, ...]
    cycle_ranks: tuple[int, ...]
    edges: Tensor
    node_features: Tensor
    edge_features: Tensor
    raw_basis: Tensor
    cycle_set: Tensor | None
    projector_values: Tensor | None
    edge_targets: Tensor | None
    node_targets: Tensor | None
    graph_targets: Tensor | None

    @property
    def batch_size(self) -> int:
        return len(self.graph_ids)

    def pin_memory(self) -> PreparedBatch:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                setattr(self, field.name, value.pin_memory())
        return self

    def to(self, device: torch.device, *, non_blocking: bool = False) -> PreparedBatch:
        values: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device=device, non_blocking=non_blocking)
                if isinstance(value, Tensor)
                else value
            )
        return PreparedBatch(**values)  # type: ignore[arg-type]


def _concatenate_optional(
    graphs: list[PreparedGraph], field_name: str, *, stack: bool = False
) -> Tensor | None:
    values = [getattr(graph, field_name) for graph in graphs]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{field_name} is missing on part of a physical minibatch")
    tensors = [value for value in values if isinstance(value, Tensor)]
    return torch.stack(tensors, dim=0) if stack else torch.cat(tensors, dim=0)


def pack_prepared_graphs(
    graphs: list[PreparedGraph],
    *,
    variant: str | None = None,
    target_levels: tuple[str, ...] | None = None,
) -> PreparedBatch:
    """Pack ragged CPU graph objects for vectorized PE and message passing."""

    if not graphs:
        raise ValueError("cannot pack an empty physical minibatch")
    if variant is not None and variant not in PE_VARIANTS:
        raise ValueError(f"unknown PE variant: {variant}")
    if target_levels is not None and not set(target_levels) <= {"edge", "node", "graph"}:
        raise ValueError("target_levels contains an unknown supervision level")
    node_counts = tuple(int(graph.node_features.shape[0]) for graph in graphs)
    edge_counts = tuple(int(graph.edge_features.shape[0]) for graph in graphs)
    cycle_ranks = tuple(int(graph.raw_basis.shape[1]) for graph in graphs)
    if any(
        graph.edges.shape != (edges, 2)
        for graph, edges in zip(graphs, edge_counts, strict=True)
    ):
        raise ValueError("prepared edge indices must have shape [num_edges, 2]")
    if any(
        graph.raw_basis.shape[0] != edges
        for graph, edges in zip(graphs, edge_counts, strict=True)
    ):
        raise ValueError("raw cycle-basis rows must match the prepared edge count")

    maximum_rank = max(cycle_ranks, default=0) if variant in (None, "raw") else 0
    padded_basis = (
        [
            F.pad(graph.raw_basis, (0, maximum_rank - rank))
            for graph, rank in zip(graphs, cycle_ranks, strict=True)
        ]
        if maximum_rank
        else [
            graph.raw_basis.new_zeros((edge_count, 0))
            for graph, edge_count in zip(graphs, edge_counts, strict=True)
        ]
    )
    offsets: list[int] = []
    running = 0
    for count in node_counts:
        offsets.append(running)
        running += count
    edges = torch.cat(
        [graph.edges + offset for graph, offset in zip(graphs, offsets, strict=True)],
        dim=0,
    )
    projectors = [
        graph.projector if variant in (None, "projector") else None for graph in graphs
    ]
    if all(projector is None for projector in projectors):
        projector_values = None
    elif any(projector is None for projector in projectors):
        raise ValueError("projector PE is missing on part of a physical minibatch")
    else:
        checked: list[Tensor] = []
        for projector, edge_count in zip(projectors, edge_counts, strict=True):
            assert projector is not None
            if projector.shape != (edge_count, edge_count):
                raise ValueError("projector must have shape (num_edges, num_edges)")
            checked.append(projector.reshape(-1))
        projector_values = torch.cat(checked, dim=0)

    return PreparedBatch(
        graph_ids=tuple(graph.graph_id for graph in graphs),
        node_counts=node_counts,
        edge_counts=edge_counts,
        cycle_ranks=cycle_ranks,
        edges=edges,
        node_features=torch.cat([graph.node_features for graph in graphs], dim=0),
        edge_features=torch.cat([graph.edge_features for graph in graphs], dim=0),
        raw_basis=torch.cat(padded_basis, dim=0),
        cycle_set=(
            _concatenate_optional(graphs, "cycle_set")
            if variant in (None, "set")
            else None
        ),
        projector_values=projector_values,
        edge_targets=(
            _concatenate_optional(graphs, "edge_targets")
            if target_levels is None or "edge" in target_levels
            else None
        ),
        node_targets=(
            _concatenate_optional(graphs, "node_targets")
            if target_levels is None or "node" in target_levels
            else None
        ),
        graph_targets=(
            _concatenate_optional(graphs, "graph_targets", stack=True)
            if target_levels is None or "graph" in target_levels
            else None
        ),
    )


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
            nn.Sequential(nn.Linear(raw_width, pe_dim), nn.GELU())
            if variant == "raw" and raw_width
            else None
        )
        if variant == "raw" and not raw_width:
            self.empty_raw = nn.Parameter(torch.zeros(pe_dim))
        else:
            self.register_parameter("empty_raw", None)
        self.set_encoder = (
            nn.Sequential(
                nn.Linear(len(SET_STAT_NAMES), pe_dim),
                nn.GELU(),
                nn.Linear(pe_dim, pe_dim),
            )
            if variant == "set"
            else None
        )
        projector_hidden = max(8, pe_dim)
        self.projector_pair = (
            nn.Sequential(
                nn.Linear(3, projector_hidden),
                nn.GELU(),
                nn.Linear(projector_hidden, projector_hidden),
                nn.GELU(),
            )
            if variant == "projector"
            else None
        )
        self.projector_row = (
            nn.Sequential(
                nn.Linear(2 * projector_hidden + 2, pe_dim),
                nn.GELU(),
                nn.Linear(pe_dim, pe_dim),
            )
            if variant == "projector"
            else None
        )

    def forward(
        self,
        raw_basis: Tensor,
        cycle_set: Tensor | None,
        projector: Tensor | None,
    ) -> Tensor:
        edge_count = raw_basis.shape[0]
        projector_values = None if projector is None else projector.reshape(-1)
        return self.forward_batch(
            raw_basis,
            cycle_set,
            projector_values,
            edge_counts=(edge_count,),
            cycle_ranks=(int(raw_basis.shape[1]),),
        )

    def forward_batch(
        self,
        raw_basis: Tensor,
        cycle_set: Tensor | None,
        projector_values: Tensor | None,
        *,
        edge_counts: tuple[int, ...],
        cycle_ranks: tuple[int, ...],
    ) -> Tensor:
        """Encode a packed ragged minibatch with one learned call per stage."""

        if len(edge_counts) != len(cycle_ranks) or not edge_counts:
            raise ValueError("edge-count and cycle-rank metadata must align")
        edge_count = sum(edge_counts)
        if raw_basis.shape[0] != edge_count:
            raise ValueError("packed raw-basis rows do not match packed edge counts")
        if self.variant == "no_pe":
            return raw_basis.new_zeros((edge_count, self.pe_dim))
        if self.variant == "raw":
            actual_rank = max(cycle_ranks)
            if actual_rank > self.raw_width:
                raise RawCycleRankOverflow(actual_rank, self.raw_width)
            if raw_basis.shape[1] < self.raw_width:
                raw_basis = F.pad(raw_basis, (0, self.raw_width - raw_basis.shape[1]))
            if self.raw_encoder is None:
                if self.empty_raw is None:
                    raise RuntimeError("empty raw-cycle parameter was not initialized")
                return self.empty_raw.unsqueeze(0).expand(edge_count, -1)
            return self.raw_encoder(raw_basis)
        if self.variant == "set":
            if cycle_set is None:
                raise ValueError("set PE was not prepared for this physical minibatch")
            if self.set_encoder is None:
                raise RuntimeError("set PE encoder was not initialized")
            return self.set_encoder(cycle_set)

        if projector_values is None:
            raise ValueError("projector PE was not prepared for this physical minibatch")
        if projector_values.numel() != sum(count * count for count in edge_counts):
            raise ValueError("packed projector values do not match per-graph edge counts")
        if edge_count == 0:
            return raw_basis.new_zeros((0, self.pe_dim))
        if self.projector_pair is None or self.projector_row is None:
            raise RuntimeError("projector PE encoders were not initialized")

        device = projector_values.device
        counts = torch.as_tensor(edge_counts, dtype=torch.long, device=device)
        square_counts = counts.square()
        square_offsets = torch.cumsum(square_counts, dim=0) - square_counts
        edge_offsets = torch.cumsum(counts, dim=0) - counts
        graph_index = torch.repeat_interleave(
            torch.arange(len(edge_counts), device=device), square_counts
        )
        within_graph = torch.arange(projector_values.numel(), device=device) - square_offsets[
            graph_index
        ]
        local_columns = torch.remainder(within_graph, counts[graph_index])
        local_rows = torch.div(within_graph, counts[graph_index], rounding_mode="floor")
        row_index = edge_offsets[graph_index] + local_rows
        diagonal_rows = torch.repeat_interleave(
            torch.arange(len(edge_counts), device=device), counts
        )
        local_diagonal = torch.arange(edge_count, device=device) - edge_offsets[diagonal_rows]
        diagonal_index = (
            square_offsets[diagonal_rows]
            + local_diagonal * (counts[diagonal_rows] + 1)
        )
        diagonal = projector_values[diagonal_index].abs()
        absolute = projector_values.abs()
        column_diagonal = diagonal[edge_offsets[graph_index] + local_columns]
        pair_features = torch.stack(
            (
                absolute,
                absolute.square(),
                column_diagonal,
            ),
            dim=-1,
        )
        encoded = self.projector_pair(pair_features)
        mean = encoded.new_zeros((edge_count, encoded.shape[1]))
        mean.index_add_(0, row_index, encoded)
        mean = mean / counts[diagonal_rows, None]
        maximum = encoded.new_full((edge_count, encoded.shape[1]), -torch.inf)
        maximum.scatter_reduce_(
            0,
            row_index[:, None].expand_as(encoded),
            encoded,
            reduce="amax",
            include_self=True,
        )
        absolute_mean = absolute.new_zeros(edge_count)
        absolute_mean.index_add_(0, row_index, absolute)
        absolute_mean = absolute_mean / counts[diagonal_rows]
        row_features = torch.cat(
            (mean, maximum, diagonal[:, None], absolute_mean[:, None]),
            dim=1,
        )
        return self.projector_row(row_features)


class GraphOutput(NamedTuple):
    edge: Tensor | None
    node: Tensor | None
    graph: Tensor | None
    embedding: Tensor | None


class BatchOutput(NamedTuple):
    edge: Tensor | None
    node: Tensor | None
    graph: Tensor | None
    embedding: Tensor | None


class _MessageTopology(NamedTuple):
    source: Tensor
    target: Tensor
    degree: Tensor


def _message_topology(node: Tensor, edge_index: Tensor) -> _MessageTopology:
    """Prepare fixed connectivity once for a stack, including isolated nodes.

    This is local to one forward pass: it contains no learned values, buffers,
    or cross-batch cache and therefore leaves checkpoint state unchanged.
    Reuse requires unchanged node dtype/device and connectivity across layers.
    """
    u, v = edge_index[:, 0], edge_index[:, 1]
    source = torch.cat((u, v), dim=0)
    target = torch.cat((v, u), dim=0)
    degree = node.new_zeros(node.shape[0])
    degree.index_add_(0, target, torch.ones_like(target, dtype=node.dtype))
    return _MessageTopology(source, target, degree.clamp_min(1.0))


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

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        edge_index: Tensor,
        *,
        topology: _MessageTopology | None = None,
    ) -> tuple[Tensor, Tensor]:
        u, v = edge_index[:, 0], edge_index[:, 1]
        symmetric = torch.cat((node[u] + node[v], (node[u] - node[v]).abs(), edge), dim=1)
        updated_edge = self.edge_norm(edge + self.edge_update(symmetric))

        source, target, degree = (
            _message_topology(node, edge_index) if topology is None else topology
        )
        directed_edge = torch.cat((updated_edge, updated_edge), dim=0)
        messages = self.message(torch.cat((node[source], node[target], directed_edge), dim=1))
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, target, messages)
        aggregate = aggregate / degree[:, None]
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
        self.edge_head = (
            nn.Linear(3 * hidden_dim, edge_output_dim) if edge_output_dim else None
        )
        self.node_head = nn.Linear(hidden_dim, node_output_dim) if node_output_dim else None
        pooled_dim = 4 * hidden_dim
        self.graph_trunk = (
            nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            if graph_output_dim or embedding_dim
            else None
        )
        self.graph_head = nn.Linear(hidden_dim, graph_output_dim) if graph_output_dim else None
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim) if embedding_dim else None

    def forward_graph(self, graph: PreparedGraph) -> GraphOutput:
        outputs = self.forward([graph])
        assert isinstance(outputs, list)
        return outputs[0]

    def forward_batch(self, batch: PreparedBatch) -> BatchOutput:
        """Execute PE, message passing, and pooling on one disjoint union."""

        positional = self.pe_encoder.forward_batch(
            batch.raw_basis,
            batch.cycle_set,
            batch.projector_values,
            edge_counts=batch.edge_counts,
            cycle_ranks=batch.cycle_ranks,
        )
        node = self.node_encoder(batch.node_features)
        edge = self.edge_encoder(torch.cat((batch.edge_features, positional), dim=1))
        for layer in self.layers:
            node, edge = layer(node, edge, batch.edges)

        graph_state: Tensor | None = None
        if self.graph_trunk is not None:
            graph_count = batch.batch_size
            node_graph = torch.repeat_interleave(
                torch.arange(graph_count, device=node.device),
                torch.as_tensor(batch.node_counts, device=node.device),
            )
            edge_graph = torch.repeat_interleave(
                torch.arange(graph_count, device=edge.device),
                torch.as_tensor(batch.edge_counts, device=edge.device),
            )
            node_sum = node.new_zeros((graph_count, node.shape[1]))
            node_sum.index_add_(0, node_graph, node)
            node_mean = node_sum / node.new_tensor(batch.node_counts)[:, None]
            node_maximum = node.new_full((graph_count, node.shape[1]), -torch.inf)
            node_maximum.scatter_reduce_(
                0,
                node_graph[:, None].expand_as(node),
                node,
                reduce="amax",
                include_self=True,
            )
            edge_mean = edge.new_zeros((graph_count, edge.shape[1]))
            edge_maximum = edge.new_zeros((graph_count, edge.shape[1]))
            if edge.shape[0]:
                edge_mean.index_add_(0, edge_graph, edge)
                edge_counts = edge.new_tensor(batch.edge_counts)
                edge_mean = edge_mean / edge_counts.clamp_min(1)[:, None]
                edge_maximum.fill_(-torch.inf)
                edge_maximum.scatter_reduce_(
                    0,
                    edge_graph[:, None].expand_as(edge),
                    edge,
                    reduce="amax",
                    include_self=True,
                )
                edge_maximum = torch.where(
                    (edge_counts > 0)[:, None], edge_maximum, torch.zeros_like(edge_maximum)
                )
            pooled = torch.cat((node_mean, node_maximum, edge_mean, edge_maximum), dim=1)
            graph_state = self.graph_trunk(pooled)
        edge_prediction = None
        if self.edge_head is not None:
            u, v = batch.edges[:, 0], batch.edges[:, 1]
            edge_readout = torch.cat(
                (edge, node[u] + node[v], (node[u] - node[v]).abs()), dim=1
            )
            edge_prediction = self.edge_head(edge_readout)
        node_prediction = None if self.node_head is None else self.node_head(node)
        graph_prediction = None
        if self.graph_head is not None:
            if graph_state is None:
                raise RuntimeError("graph head requires the graph pooling trunk")
            graph_prediction = self.graph_head(graph_state)
        embedding = None
        if self.embedding_head is not None:
            if graph_state is None:
                raise RuntimeError("embedding head requires the graph pooling trunk")
            embedding = self.embedding_head(graph_state)
        return BatchOutput(edge_prediction, node_prediction, graph_prediction, embedding)

    def forward(
        self, graphs: list[PreparedGraph] | PreparedBatch
    ) -> list[GraphOutput] | BatchOutput:
        if isinstance(graphs, PreparedBatch):
            return self.forward_batch(graphs)
        if not graphs:
            return []
        batch = pack_prepared_graphs(graphs, variant=self.pe_encoder.variant)
        packed = self.forward_batch(batch)
        edge_outputs = (
            [None] * len(graphs)
            if packed.edge is None
            else list(torch.split(packed.edge, batch.edge_counts, dim=0))
        )
        node_outputs = (
            [None] * len(graphs)
            if packed.node is None
            else list(torch.split(packed.node, batch.node_counts, dim=0))
        )
        return [
            GraphOutput(
                edge=edge_outputs[index],
                node=node_outputs[index],
                graph=None if packed.graph is None else packed.graph[index],
                embedding=None if packed.embedding is None else packed.embedding[index],
            )
            for index in range(len(graphs))
        ]


__all__ = [
    "BatchOutput",
    "GraphOutput",
    "PE_VARIANTS",
    "PaperCycleModel",
    "PreparedBatch",
    "PreparedGraph",
    "RawCycleRankOverflow",
    "StaticPEEncoder",
    "infer_raw_width",
    "pack_prepared_graphs",
    "prepare_graph",
    "prepare_splits",
]
