"""Coordinate-free cycle-space PE and a deep residual molecular GNN.

For any full basis ``Z`` of ``ker(B.T)``, let

``P_Z = Z (Z.T Z)^-1 Z.T`` and ``K_Z = P_Z * P_Z`` (Hadamard square).

``P_Z`` is invariant to every invertible chart replacement ``Z -> ZR``.
Squaring entrywise also removes independent incidence-orientation signs, while
``K_Z X`` remains equivariant to edge permutations.  Consequently no arbitrary
SVD/fundamental-cycle coordinate is ever presented to a learned layer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from research.cycle_pe.benchmark_models import ATOM_DIMS, BOND_DIMS, CategoricalEncoder, _pool
from research.cycle_pe.paper_model import _message_topology
from research.cycle_pe.v2.data import DATASETS, Batch

MODEL_NAME = "cycle_projector_pe_v2"
BASIS_EXECUTIONS = ("batched", "reference")


class LeftNullBasisEncoder(nn.Module):
    """Bond-conditioned PE from the intrinsic cycle-space projector kernel.

    The historical class name is retained for import compatibility. Production
    Production data stores orthonormal ``Q`` and marks it as such, so no
    factorization occurs during training.  The diagnostic DFS-fundamental data
    backend stores raw ``Z`` and therefore takes the generic graph-local QR path.
    """

    def __init__(self, bond_dim: int, pe_dim: int, *, column_chunk_size: int = 16):
        super().__init__()
        if min(bond_dim, pe_dim, column_chunk_size) < 1:
            raise ValueError("bond_dim, pe_dim and column_chunk_size must be positive")
        self.bond_dim = bond_dim
        self.pe_dim = pe_dim
        # Kept as a public compatibility knob; low-rank contraction cores, not
        # arbitrary basis columns, are now the bounded execution unit.
        self.column_chunk_size = column_chunk_size
        self.column_phi = nn.Sequential(
            nn.Linear(bond_dim, pe_dim), nn.SiLU(), nn.Linear(pe_dim, pe_dim)
        )
        self.edge_psi = nn.Sequential(
            nn.Linear(2 * pe_dim + 3, 2 * pe_dim),
            nn.SiLU(),
            nn.Linear(2 * pe_dim, pe_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(nn.Linear(pe_dim, pe_dim), nn.SiLU())

    def _checked_q(self, basis: Tensor, *, orthonormal_input: bool) -> Tensor:
        if basis.ndim != 2:
            raise ValueError("basis must have shape [num_edges, cycle_rank]")
        if not basis.is_floating_point():
            raise ValueError("basis must be a floating point tensor [num_edges, cycle_rank]")
        edge_count, rank = basis.shape
        if edge_count == 0 and rank:
            raise ValueError("an edgeless graph cannot have nonzero cycle rank")
        if rank > edge_count:
            raise ValueError("cycle rank cannot exceed the edge count")
        if rank == 0:
            return basis.float()
        if orthonormal_input:
            # Cache validation certifies Q.T@Q=I. Avoid a GPU synchronization or
            # repeated QR in the hot path.
            return basis.float()
        q, triangular = torch.linalg.qr(basis.float(), mode="reduced")
        threshold = torch.finfo(q.dtype).eps * max(edge_count, rank) * 16.0
        if bool((triangular.diagonal().abs() <= threshold).any()):
            raise ValueError("basis must have full column rank")
        return q

    def _projector_mix(
        self, q: Tensor, values: Tensor, *, pair_budget: int | None
    ) -> tuple[Tensor, Tensor]:
        # The projector contraction is the sole forced-FP32 region.  The PE
        # MLPs and graph backbone remain eligible for the caller's autocast.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, values = q.float(), values.float()
            edge_count, rank = q.shape
            leverage = q.square().sum(dim=1)
            if edge_count == 0 or rank == 0:
                return values.new_zeros(values.shape), leverage
            feature_count = values.shape[1]
            budget = pair_budget or max(1, feature_count * rank * rank)
            rank_block = min(rank, self.column_chunk_size, max(1, math.isqrt(budget)))
            feature_block = max(1, min(feature_count, budget // (rank_block * rank_block)))
            feature_parts = []
            for feature_start in range(0, feature_count, feature_block):
                selected = values[:, feature_start : feature_start + feature_block]
                mixed = selected.new_zeros(selected.shape)
                for left_start in range(0, rank, rank_block):
                    left = q[:, left_start : left_start + rank_block]
                    for right_start in range(0, rank, rank_block):
                        right = q[:, right_start : right_start + rank_block]
                        # core[d,a,b] = sum_j V[j,d] Q[j,a] Q[j,b]. This realizes
                        # ((Q Q.T) Hadamard-square (Q Q.T)) @ V without ever
                        # allocating an edge-by-edge matrix.
                        core = torch.einsum("md,ma,mb->dab", selected, left, right)
                        mixed = mixed + torch.einsum("ma,dab,mb->md", left, core, right)
                feature_parts.append(mixed)
            return torch.cat(feature_parts, dim=1), leverage

    def _projector_mix_batch(
        self,
        qs: Sequence[Tensor],
        value_parts: Sequence[Tensor],
        *,
        pair_budget: int,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Rank-grouped projector contractions without an edge-pair matrix.

        Molecular graphs with equal cycle rank are edge-padded and contracted
        together.  The temporary ``[graphs, features, r_block, r_block]`` core
        never exceeds ``pair_budget`` elements; padding never creates an
        ``m x m`` tensor.  Results are restored to their original graph order.
        """
        if len(qs) != len(value_parts):
            raise ValueError("basis/value graph counts must agree")
        mixed_parts: list[Tensor | None] = [None] * len(qs)
        leverage_parts: list[Tensor | None] = [None] * len(qs)
        by_rank: dict[int, list[int]] = {}
        for index, (q, values) in enumerate(zip(qs, value_parts, strict=True)):
            if q.shape[0] != values.shape[0]:
                raise ValueError("basis rows must align with graph bond embeddings")
            by_rank.setdefault(q.shape[1], []).append(index)

        with torch.autocast(
            device_type=value_parts[0].device.type if value_parts else "cpu", enabled=False
        ):
            for rank, indices in by_rank.items():
                if rank == 0:
                    for index in indices:
                        values = value_parts[index].float()
                        mixed_parts[index] = values.new_zeros(values.shape)
                        leverage_parts[index] = values.new_zeros(values.shape[0])
                    continue
                feature_count = value_parts[indices[0]].shape[1]
                initial_rank_block = min(
                    rank, self.column_chunk_size, max(1, math.isqrt(pair_budget))
                )
                # Reserve at least a modest feature tile when possible.  This
                # batches many small molecules instead of launching one kernel
                # per graph, while the exact core limit remains enforced below.
                graph_block = max(
                    1,
                    pair_budget
                    // (initial_rank_block * initial_rank_block * min(feature_count, 8)),
                )
                for graph_start in range(0, len(indices), graph_block):
                    selected_indices = indices[graph_start : graph_start + graph_block]
                    graph_count = len(selected_indices)
                    q_padded = pad_sequence(
                        [qs[index].float() for index in selected_indices], batch_first=True
                    )
                    values_padded = pad_sequence(
                        [value_parts[index].float() for index in selected_indices],
                        batch_first=True,
                    )
                    leverage_padded = q_padded.square().sum(dim=2)
                    mixed_padded = values_padded.new_zeros(values_padded.shape)
                    rank_block = min(
                        rank,
                        self.column_chunk_size,
                        max(1, math.isqrt(pair_budget // graph_count)),
                    )
                    feature_block = max(
                        1,
                        min(
                            feature_count,
                            pair_budget // (graph_count * rank_block * rank_block),
                        ),
                    )
                    for feature_start in range(0, feature_count, feature_block):
                        selected = values_padded[
                            :, :, feature_start : feature_start + feature_block
                        ]
                        mixed = selected.new_zeros(selected.shape)
                        for left_start in range(0, rank, rank_block):
                            left = q_padded[:, :, left_start : left_start + rank_block]
                            for right_start in range(0, rank, rank_block):
                                right = q_padded[:, :, right_start : right_start + rank_block]
                                core = torch.einsum("gmd,gma,gmb->gdab", selected, left, right)
                                if core.numel() > pair_budget:
                                    raise RuntimeError("projector core exceeded basis_pair_budget")
                                mixed.add_(torch.einsum("gma,gdab,gmb->gmd", left, core, right))
                        mixed_padded[:, :, feature_start : feature_start + selected.shape[2]] = (
                            mixed
                        )
                    for local, index in enumerate(selected_indices):
                        edge_count = qs[index].shape[0]
                        mixed_parts[index] = mixed_padded[local, :edge_count]
                        leverage_parts[index] = leverage_padded[local, :edge_count]
        if any(part is None for part in mixed_parts + leverage_parts):
            raise RuntimeError("incomplete rank-grouped projector result")
        return (
            [part for part in mixed_parts if part is not None],
            [part for part in leverage_parts if part is not None],
        )

    @staticmethod
    def _intrinsic_features(values: Tensor, mixed: Tensor, leverage: Tensor, rank: int) -> Tensor:
        safe = leverage.clamp_min(torch.finfo(leverage.dtype).eps)
        rank_fraction = leverage.new_full((len(leverage), 1), rank / max(1, len(leverage)))
        return torch.cat(
            (
                values.float() * leverage[:, None],
                mixed / safe[:, None],
                leverage[:, None],
                leverage.sqrt()[:, None],
                rank_fraction,
            ),
            dim=1,
        )

    def _encode(
        self,
        bond: Tensor,
        basis: Tensor,
        *,
        pair_budget: int | None,
        orthonormal_input: bool,
    ) -> Tensor:
        if bond.ndim != 2 or bond.shape[1] != self.bond_dim:
            raise ValueError("bond embeddings have an invalid shape")
        if basis.ndim != 2 or basis.shape[0] != bond.shape[0]:
            raise ValueError("basis must have shape (num_edges, cycle_rank)")
        if basis.device != bond.device:
            raise ValueError("basis must be on the bond embedding device")
        edge_count, rank = basis.shape
        if rank == 0:
            return bond.new_zeros((edge_count, self.pe_dim))
        q = self._checked_q(basis, orthonormal_input=orthonormal_input)
        values = self.column_phi(bond)
        mixed, leverage = self._projector_mix(q, values, pair_budget=pair_budget)
        features = self._intrinsic_features(values, mixed, leverage, rank)
        encoded = self.output(self.edge_psi(features))
        # Bridges (hence every edge in an acyclic component) have zero cycle
        # leverage and exactly zero PE despite affine biases.
        return encoded * leverage.sqrt()[:, None].to(encoded.dtype)

    def forward(self, bond: Tensor, basis: Tensor) -> Tensor:
        return self._encode(bond, basis, pair_budget=None, orthonormal_input=False)

    def forward_batch(
        self,
        bond: Tensor,
        bases: Sequence[Tensor],
        *,
        pair_budget: int = 32768,
        orthonormal_input: bool | Sequence[bool] = False,
    ) -> Tensor:
        if pair_budget < 1:
            raise ValueError("pair_budget must be positive")
        counts = [basis.shape[0] for basis in bases]
        if sum(counts) != len(bond):
            raise ValueError("basis rows must align with concatenated bond embeddings")
        if not bases:
            return bond.new_zeros((0, self.pe_dim))
        if isinstance(orthonormal_input, bool):
            orthonormal_flags = [orthonormal_input] * len(bases)
        else:
            orthonormal_flags = list(orthonormal_input)
            if len(orthonormal_flags) != len(bases) or any(
                not isinstance(flag, bool) for flag in orthonormal_flags
            ):
                raise ValueError("one Boolean orthonormal-input flag is required per basis")
        qs = [
            self._checked_q(basis, orthonormal_input=flag)
            for basis, flag in zip(bases, orthonormal_flags, strict=True)
        ]
        values = self.column_phi(bond)
        value_parts = list(torch.split(values, counts))
        mixed_parts, leverage_parts = self._projector_mix_batch(
            qs, value_parts, pair_budget=pair_budget
        )
        features = torch.cat(
            [
                self._intrinsic_features(value, mixed, leverage, q.shape[1])
                for q, value, mixed, leverage in zip(
                    qs, value_parts, mixed_parts, leverage_parts, strict=True
                )
            ],
            dim=0,
        )
        leverage = torch.cat(leverage_parts)
        encoded = self.output(self.edge_psi(features))
        return encoded * leverage.sqrt()[:, None].to(encoded.dtype)


class _ResidualEdgeGraphBlock(nn.Module):
    """Pre-norm residual edge-aware message block with separate FFNs."""

    def __init__(
        self,
        hidden: int,
        *,
        ffn_multiplier: int,
        dropout: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        expanded = hidden * ffn_multiplier
        self.edge_update_norm = nn.LayerNorm(hidden)
        self.edge_node_norm = nn.LayerNorm(hidden)
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.edge_ffn_norm = nn.LayerNorm(hidden)
        self.edge_ffn = nn.Sequential(
            nn.Linear(hidden, expanded), nn.SiLU(), nn.Linear(expanded, hidden)
        )
        self.message_node_norm = nn.LayerNorm(hidden)
        self.message_edge_norm = nn.LayerNorm(hidden)
        self.message = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid()
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.node_ffn_norm = nn.LayerNorm(hidden)
        self.node_ffn = nn.Sequential(
            nn.Linear(hidden, expanded), nn.SiLU(), nn.Linear(expanded, hidden)
        )
        self.dropout = nn.Dropout(dropout)
        self.edge_scale = nn.Parameter(torch.full((hidden,), layer_scale))
        self.edge_ffn_scale = nn.Parameter(torch.full((hidden,), layer_scale))
        self.node_scale = nn.Parameter(torch.full((hidden,), layer_scale))
        self.node_ffn_scale = nn.Parameter(torch.full((hidden,), layer_scale))

    def forward(
        self, node: Tensor, edge: Tensor, edge_index: Tensor, *, topology
    ) -> tuple[Tensor, Tensor]:
        u, v = edge_index[:, 0], edge_index[:, 1]
        normalized_node = self.edge_node_norm(node)
        normalized_edge = self.edge_update_norm(edge)
        symmetric = torch.cat(
            (
                normalized_node[u] + normalized_node[v],
                (normalized_node[u] - normalized_node[v]).abs(),
                normalized_edge,
            ),
            dim=1,
        )
        edge = edge + self.dropout(self.edge_update(symmetric)) * self.edge_scale
        edge = edge + self.dropout(self.edge_ffn(self.edge_ffn_norm(edge))) * self.edge_ffn_scale

        source, target, degree = topology
        normalized_node = self.message_node_norm(node)
        normalized_edge = self.message_edge_norm(edge)
        directed_edge = torch.cat((normalized_edge, normalized_edge), dim=0)
        message_input = torch.cat(
            (normalized_node[source], normalized_node[target], directed_edge), dim=1
        )
        messages = self.message(message_input) * self.message_gate(message_input)
        aggregate = torch.zeros_like(node)
        # Autocast may produce FP16/BF16 messages while the residual stream is
        # FP32. index_add_ requires exact dtype agreement.
        aggregate.index_add_(0, target, messages.to(aggregate.dtype))
        aggregate = aggregate / degree[:, None]
        node = (
            node
            + self.dropout(self.node_update(torch.cat((normalized_node, aggregate), dim=1)))
            * self.node_scale
        )
        node = node + self.dropout(self.node_ffn(self.node_ffn_norm(node))) * self.node_ffn_scale
        return node, edge


class CycleBasisPEModel(nn.Module):
    """Projector-kernel PE with a deep, stable molecular graph backbone."""

    def __init__(
        self,
        *,
        dataset: str,
        hidden: int = 128,
        pe_dim: int = 64,
        layers: int = 10,
        column_chunk_size: int = 16,
        basis_execution: str = "batched",
        basis_pair_budget: int = 32768,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        layer_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {dataset}")
        if min(hidden, pe_dim, layers, ffn_multiplier) < 1:
            raise ValueError("hidden, pe_dim, layers and ffn_multiplier must be positive")
        if not 0.0 <= dropout < 1.0 or not 0.0 < layer_scale <= 1.0:
            raise ValueError("dropout or layer_scale is outside its stable range")
        if basis_execution not in BASIS_EXECUTIONS:
            raise ValueError(f"basis_execution must be one of {BASIS_EXECUTIONS}")
        if basis_pair_budget < 1:
            raise ValueError("basis_pair_budget must be positive")
        self.basis_execution = basis_execution
        self.basis_pair_budget = basis_pair_budget
        self.node_encoder = CategoricalEncoder((28,) if dataset == "zinc12k" else ATOM_DIMS, hidden)
        self.bond_encoder = CategoricalEncoder((4,) if dataset == "zinc12k" else BOND_DIMS, hidden)
        self.pe_encoder = LeftNullBasisEncoder(hidden, pe_dim, column_chunk_size=column_chunk_size)
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden + pe_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.layers = nn.ModuleList(
            _ResidualEdgeGraphBlock(
                hidden,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                layer_scale=layer_scale,
            )
            for _ in range(layers)
        )
        self.final_node_norm = nn.LayerNorm(hidden)
        self.final_edge_norm = nn.LayerNorm(hidden)
        self.graph_trunk = nn.Sequential(
            nn.Linear(6 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        )
        self.graph_head = nn.Linear(hidden, 1 if dataset == "zinc12k" else 11)

    def forward(self, batch: Batch) -> Tensor:
        graph_count = len(batch.ptr) - 1
        if len(batch.cycle_bases) != graph_count:
            raise ValueError("one cycle-space basis is required per graph")
        if len(batch.cycle_basis_is_orthonormal) != graph_count:
            raise ValueError("one cycle-space representation flag is required per graph")
        counts = [basis.shape[0] for basis in batch.cycle_bases]
        if sum(counts) != len(batch.edge_attr):
            raise ValueError("ragged basis rows do not align with batch bonds")
        node = self.node_encoder(batch.x)
        bond = self.bond_encoder(batch.edge_attr)
        if self.basis_execution == "reference":
            positional = torch.cat(
                [
                    self.pe_encoder._encode(
                        part,
                        basis,
                        pair_budget=None,
                        orthonormal_input=is_orthonormal,
                    )
                    for part, basis, is_orthonormal in zip(
                        torch.split(bond, counts),
                        batch.cycle_bases,
                        batch.cycle_basis_is_orthonormal,
                        strict=True,
                    )
                ],
                dim=0,
            )
        else:
            positional = self.pe_encoder.forward_batch(
                bond,
                batch.cycle_bases,
                pair_budget=self.basis_pair_budget,
                orthonormal_input=batch.cycle_basis_is_orthonormal,
            )
        edge = self.edge_encoder(torch.cat((bond, positional), dim=1))
        # The outer training/evaluation autocast remains active for the deep
        # backbone.  Only the projector contractions above force FP32.
        edge_index = batch.edge_index.T
        topology = _message_topology(node, edge_index)
        for layer in self.layers:
            node, edge = layer(node, edge, edge_index, topology=topology)
        node = self.final_node_norm(node)
        edge = self.final_edge_norm(edge)
        node_mean, node_max = _pool(node, batch.batch, graph_count)
        node_sizes = (batch.ptr[1:] - batch.ptr[:-1]).to(node.dtype)[:, None]
        node_sum = node_mean * node_sizes
        edge_graph = batch.batch[batch.edge_index[0]]
        edge_mean, edge_max = _pool(edge, edge_graph, graph_count)
        edge_sizes = (batch.edge_ptr[1:] - batch.edge_ptr[:-1]).to(edge.dtype)[:, None]
        edge_sum = edge_mean * edge_sizes
        pooled = torch.cat((node_sum, node_mean, node_max, edge_sum, edge_mean, edge_max), dim=1)
        return self.graph_head(self.graph_trunk(pooled))


def architecture_protocol() -> dict[str, str]:
    return {
        "model": MODEL_NAME,
        "cycle_space": (
            "default thin_q caches a thin-QR orthonormalization Q of a sparse fundamental "
            "basis Z of ker(B.T); optional dfs_fundamental caches raw DFS Z and performs "
            "graph-local QR before projector use; beta=m-n+components; no truncation"
        ),
        "positional_encoding": (
            "P=Q Q.T and orientation-free K=P Hadamard-square P; learned bond values are "
            "mixed as K phi(bond), normalized by leverage diag(P), then fused with local "
            "cycle-weighted values and leverage"
        ),
        "symmetry": (
            "invariant to every invertible cycle-basis replacement Z->ZR and independent "
            "edge-orientation flips; edge-permutation equivariant"
        ),
        "execution": (
            "thin_q has no QR/SVD/pinv in training; dfs_fundamental is a diagnostic "
            "correctness backend with repeated per-forward QR, not a speedup; K@V uses "
            "pair-free rank-grouped contractions and never m-by-m storage"
        ),
        "backbone": (
            "deep pre-norm residual edge/node message blocks; gated directed messages; "
            "separate edge and node FFNs; LayerScale and dropout for stable deep training"
        ),
        "pe_injection": "concatenate projector-kernel PE with categorical bond embedding",
        "pooling": "node and edge sum/mean/max followed by a two-layer graph trunk",
        "forest_policy": (
            "bridges, including every edge of an acyclic component, have zero leverage and PE"
        ),
        "numeric_policy": (
            "projector contraction executes in FP32; PE MLPs and the deep backbone honor "
            "outer AMP, with FP32 residual accumulation where required"
        ),
        "reference_comparison": f"external published tables only; trains only {MODEL_NAME}",
    }


__all__ = [
    "BASIS_EXECUTIONS",
    "MODEL_NAME",
    "CycleBasisPEModel",
    "LeftNullBasisEncoder",
    "architecture_protocol",
]
