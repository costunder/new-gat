"""Cycle PE v2: learn from every signed vector of the left-nullspace basis.

No precomputed cycle statistics, projector, train-fitted width or truncation is
used here. The variable number of columns is handled by a learned column-set
encoder. Its fixed-width output is a learned PE, not a lossless basis codec.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor, nn

from research.cycle_pe.benchmark_models import (
    ATOM_DIMS,
    BOND_DIMS,
    CategoricalEncoder,
    _pool,
)
from research.cycle_pe.paper_model import _message_topology, _MessageLayer
from research.cycle_pe.v2.data import DATASETS, Batch

MODEL_NAME = "cycle_basis_v2"
BASIS_EXECUTIONS = ("batched", "reference")

# graph index, first column, block width, flattened start, flattened stop
PairSlice = tuple[int, int, int, int, int]


class LeftNullBasisEncoder(nn.Module):
    """Contextual encoder of all columns of U in ker(B.T).

    For each signed column u, phi([bond_e, u_e]) is learned BEFORE the edge
    aggregation. The pooled column context is then passed to each edge through
    psi([bond_e, u_e, context]). Averaging the full nonlinear encodings f(u) and
    f(-u) handles column signs; summing columns handles their ordering. Applying
    an entrywise absolute value before the learned encoder would discard relative
    sign structure and is deliberately not done.

    This is not invariant to arbitrary orthogonal rotations U -> UQ or independent
    edge orientation changes. Data preparation fixes canonical edge orientations.
    Column chunking changes temporary allocation, never the selected basis rank.
    """

    def __init__(self, bond_dim: int, pe_dim: int, *, column_chunk_size: int = 16):
        super().__init__()
        if min(bond_dim, pe_dim, column_chunk_size) < 1:
            raise ValueError("bond_dim, pe_dim and column_chunk_size must be positive")
        self.bond_dim = bond_dim
        self.pe_dim = pe_dim
        self.column_chunk_size = column_chunk_size
        self.column_phi = nn.Sequential(
            nn.Linear(bond_dim + 1, pe_dim),
            nn.GELU(),
            nn.Linear(pe_dim, pe_dim),
            nn.GELU(),
        )
        self.edge_psi = nn.Sequential(
            nn.Linear(bond_dim + 1 + pe_dim, pe_dim),
            nn.GELU(),
            nn.Linear(pe_dim, pe_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(nn.Linear(pe_dim, pe_dim), nn.GELU(), nn.Linear(pe_dim, pe_dim))

    def _signed_columns(self, bond: Tensor, columns: Tensor) -> Tensor:
        edge_count, column_count = columns.shape
        local = torch.cat(
            (
                bond[:, None, :].expand(edge_count, column_count, self.bond_dim),
                columns[:, :, None],
            ),
            dim=2,
        )
        context = self.column_phi(local).mean(dim=0, keepdim=True)
        return self.edge_psi(torch.cat((local, context.expand(edge_count, -1, -1)), dim=2))

    def forward(self, bond: Tensor, basis: Tensor) -> Tensor:
        """Original single-graph reference, retained for execution comparisons."""
        if bond.ndim != 2 or bond.shape[1] != self.bond_dim:
            raise ValueError("bond embeddings have an invalid shape")
        if basis.ndim != 2 or basis.shape[0] != bond.shape[0]:
            raise ValueError("basis must have shape (num_edges, cycle_rank)")
        if basis.device != bond.device or not basis.is_floating_point():
            raise ValueError("basis must be floating point on the bond embedding device")
        edge_count, rank = basis.shape
        if rank == 0:
            # A forest has no cycle coordinates. Do not create a learned bias PE.
            return bond.new_zeros((edge_count, self.pe_dim))
        if edge_count == 0:
            raise ValueError("an edgeless graph cannot have nonzero cycle rank")
        # Float32 also keeps small signed SVD coordinates intact under AMP.
        with torch.autocast(device_type=bond.device.type, enabled=False):
            bond, basis = bond.float(), basis.float()
            encoded = bond.new_zeros((edge_count, self.pe_dim))
            for start in range(0, rank, self.column_chunk_size):
                columns = basis[:, start : start + self.column_chunk_size]
                positive = self._signed_columns(bond, columns)
                negative = self._signed_columns(bond, -columns)
                encoded = encoded + (0.5 * (positive + negative)).sum(dim=1)
            return self.output(encoded / math.sqrt(rank))

    def _pair_groups(self, bases: Sequence[Tensor], budget: int) -> Iterator[list[PairSlice]]:
        """Pack graph-local column blocks without allocating all pair indices.

        Splitting a column across groups is allowed: its complete edge mean is
        accumulated in the first pass before any second-pass edge encoding.
        """
        group: list[PairSlice] = []
        available = budget
        for graph, basis in enumerate(bases):
            edge_count, rank = basis.shape
            for first in range(0, rank, self.column_chunk_size):
                width = min(self.column_chunk_size, rank - first)
                start, size = 0, edge_count * width
                while start < size:
                    stop = min(size, start + available)
                    group.append((graph, first, width, start, stop))
                    available -= stop - start
                    start = stop
                    if not available:
                        yield group
                        group, available = [], budget
        if group:
            yield group

    def _pack_pairs(
        self,
        bond: Tensor,
        bases: Sequence[Tensor],
        group: list[PairSlice],
        edge_offsets: list[int],
        column_offsets: list[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        edge_ids, column_ids, values = [], [], []
        for graph, first, width, start, stop in group:
            flat = torch.arange(start, stop, device=bond.device)
            rows = flat.div(width, rounding_mode="floor")
            columns = first + flat.remainder(width)
            edge_ids.append(rows + edge_offsets[graph])
            column_ids.append(columns + column_offsets[graph])
            # Index only this bounded slice: flattening a noncontiguous column
            # block first could silently allocate its entire m-by-block array.
            values.append(bases[graph][rows, columns])
        edge_index = torch.cat(edge_ids)
        column_index = torch.cat(column_ids)
        local = torch.cat((bond[edge_index], torch.cat(values)[:, None]), dim=1)
        return local, edge_index, column_index

    def forward_batch(
        self, bond: Tensor, bases: Sequence[Tensor], *, pair_budget: int = 32768
    ) -> Tensor:
        """Same signed-column encoder with shared, bounded batched MLP calls.

        A segment identifies one (graph, basis column), never merely its column
        number. Two passes retain each column's full edge-mean context even if
        one column has more edges than the budget. Every basis entry is used.

        ``pair_budget`` bounds rows per phi/psi call for each sign, not total
        autograd memory: backward still retains activations from all groups.
        No persistent buffers or trainable parameters are added.
        """
        if pair_budget < 1:
            raise ValueError("pair_budget must be positive")
        if bond.ndim != 2 or bond.shape[1] != self.bond_dim:
            raise ValueError("bond embeddings have an invalid shape")
        edge_offsets, column_offsets = [0], [0]
        counts, ranks = [], []
        for basis in bases:
            if basis.ndim != 2:
                raise ValueError("basis must have shape (num_edges, cycle_rank)")
            if basis.device != bond.device or not basis.is_floating_point():
                raise ValueError("basis must be floating point on the bond embedding device")
            edge_count, rank = basis.shape
            if not edge_count and rank:
                raise ValueError("an edgeless graph cannot have nonzero cycle rank")
            counts.append(edge_count)
            ranks.append(rank)
            edge_offsets.append(edge_offsets[-1] + edge_count)
            column_offsets.append(column_offsets[-1] + rank)
        if edge_offsets[-1] != len(bond):
            raise ValueError("basis rows must align with the concatenated bond embeddings")
        if not column_offsets[-1]:
            return bond.new_zeros((len(bond), self.pe_dim))

        with torch.autocast(device_type=bond.device.type, enabled=False):
            bond = bond.float()
            bases = tuple(basis.float() for basis in bases)
            groups = list(self._pair_groups(bases, pair_budget))
            count_tensor = torch.tensor(counts, device=bond.device, dtype=torch.long)
            rank_tensor = torch.tensor(ranks, device=bond.device, dtype=torch.long)
            column_sizes = torch.repeat_interleave(
                count_tensor, rank_tensor, output_size=column_offsets[-1]
            ).to(bond.dtype)
            positive_context = bond.new_zeros((column_offsets[-1], self.pe_dim))
            negative_context = torch.zeros_like(positive_context)
            for group in groups:
                local, _, column_index = self._pack_pairs(
                    bond, bases, group, edge_offsets, column_offsets
                )
                negative = torch.cat((local[:, :-1], -local[:, -1:]), dim=1)
                positive_context.index_add_(0, column_index, self.column_phi(local))
                negative_context.index_add_(0, column_index, self.column_phi(negative))
            positive_context = positive_context / column_sizes[:, None]
            negative_context = negative_context / column_sizes[:, None]

            encoded = bond.new_zeros((len(bond), self.pe_dim))
            for group in groups:
                local, edge_index, column_index = self._pack_pairs(
                    bond, bases, group, edge_offsets, column_offsets
                )
                negative = torch.cat((local[:, :-1], -local[:, -1:]), dim=1)
                positive = self.edge_psi(torch.cat((local, positive_context[column_index]), dim=1))
                negative = self.edge_psi(
                    torch.cat((negative, negative_context[column_index]), dim=1)
                )
                encoded.index_add_(0, edge_index, 0.5 * (positive + negative))
            edge_ranks = torch.repeat_interleave(rank_tensor, count_tensor, output_size=len(bond))
            result = self.output(encoded / edge_ranks.clamp_min(1).to(bond.dtype).sqrt()[:, None])
            # Forest edges remain exactly zero, including output-MLP biases.
            return torch.where(edge_ranks[:, None] > 0, result, torch.zeros_like(result))


class CycleBasisPEModel(nn.Module):
    """V2 basis-vector PE with the unchanged cycle-track message backbone."""

    def __init__(
        self,
        *,
        dataset: str,
        hidden: int = 64,
        pe_dim: int = 32,
        layers: int = 3,
        column_chunk_size: int = 16,
        basis_execution: str = "batched",
        basis_pair_budget: int = 32768,
    ):
        super().__init__()
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {dataset}")
        if min(hidden, pe_dim, layers) < 1:
            raise ValueError("hidden, pe_dim and layers must be positive")
        if basis_execution not in BASIS_EXECUTIONS:
            raise ValueError(f"basis_execution must be one of {BASIS_EXECUTIONS}")
        if basis_pair_budget < 1:
            raise ValueError("basis_pair_budget must be positive")
        self.basis_execution = basis_execution
        self.basis_pair_budget = basis_pair_budget
        self.node_encoder = CategoricalEncoder((28,) if dataset == "zinc12k" else ATOM_DIMS, hidden)
        self.bond_encoder = CategoricalEncoder((4,) if dataset == "zinc12k" else BOND_DIMS, hidden)
        self.pe_encoder = LeftNullBasisEncoder(hidden, pe_dim, column_chunk_size=column_chunk_size)
        self.edge_encoder = nn.Sequential(nn.Linear(hidden + pe_dim, hidden), nn.GELU())
        self.layers = nn.ModuleList(_MessageLayer(hidden) for _ in range(layers))
        self.graph_trunk = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.graph_head = nn.Linear(hidden, 1 if dataset == "zinc12k" else 11)

    def forward(self, batch: Batch) -> Tensor:
        graph_count = len(batch.ptr) - 1
        if len(batch.cycle_bases) != graph_count:
            raise ValueError("one left-nullspace basis is required per graph")
        counts = [basis.shape[0] for basis in batch.cycle_bases]
        if sum(counts) != len(batch.edge_attr):
            raise ValueError("ragged basis rows do not align with batch bonds")
        node = self.node_encoder(batch.x)
        bond = self.bond_encoder(batch.edge_attr)
        if self.basis_execution == "reference":
            positional = torch.cat(
                [
                    self.pe_encoder(part, basis)
                    for part, basis in zip(
                        torch.split(bond, counts), batch.cycle_bases, strict=True
                    )
                ]
            )
        else:
            positional = self.pe_encoder.forward_batch(
                bond, batch.cycle_bases, pair_budget=self.basis_pair_budget
            )
        edge = self.edge_encoder(torch.cat((bond, positional), dim=1))
        with torch.autocast(device_type=node.device.type, enabled=False):
            node, edge = node.float(), edge.float()
            topology = _message_topology(node, batch.edge_index.T)
            for layer in self.layers:
                node, edge = layer(node, edge, batch.edge_index.T, topology=topology)
            node_mean, node_max = _pool(node, batch.batch, graph_count)
            edge_graph = batch.batch[batch.edge_index[0]]
            edge_mean, edge_max = _pool(edge, edge_graph, graph_count)
            pooled = torch.cat((node_mean, node_max, edge_mean, edge_max), dim=1)
        return self.graph_head(self.graph_trunk(pooled))


def architecture_protocol() -> dict[str, str]:
    return {
        "model": MODEL_NAME,
        "positional_encoding": (
            "full signed left-nullspace basis U_c of incidence B; B.T @ U_c = 0; "
            "every beta=m-n+components column enters a learned contextual column encoder"
        ),
        "basis_aggregation": (
            "phi([bond_e,U_e,k]) then edge-axis mean column context; "
            "psi([bond_e,U_e,k,context]); average nonlinear f(u),f(-u); "
            "sum all columns / sqrt(beta), then learned PE MLP"
        ),
        "basis_execution": (
            "batched (default): budget-bounded edge/column pairs with distinct graph-column "
            "segments and two-pass complete column context; reference: original per-graph "
            "encoder; identical parameters and basis formula, floating-point reductions may differ"
        ),
        "basis_width": (
            "ragged graph-local full rank; no train-fitted padding, truncation, "
            "projector or handcrafted cycle statistics; forests receive zero PE"
        ),
        "backbone": (
            "unchanged cycle_pe.paper_model._MessageLayer edge-aware GNN; "
            "not a separately trained external-model baseline"
        ),
        "pe_injection": "concatenate learned full-basis PE with categorical bond embedding",
        "pooling": "node mean/max and edge mean/max, then graph MLP",
        "cycle_symmetry": (
            "cycle-column sign and permutation invariant; not arbitrary O(beta) "
            "basis-rotation invariant, not invariant to independent edge reorientation"
        ),
        "limits": (
            "canonical incidence orientation and numerical SVD basis are part of the protocol; "
            "fixed-width learned PE is not guaranteed injective or a lossless cycle codec"
        ),
        "reference_comparison": "external published tables only; trains only our cycle_basis_v2",
        "numeric_policy": "basis encoder, message layers and scatter pooling stay FP32 under AMP",
    }
