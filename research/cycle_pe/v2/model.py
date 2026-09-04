"""Matched SE and relative-PE models on a complete sparse DFS cycle basis.

SE transports learned bond values through unsigned cycle membership. PE adds
a parameter-free cycle-relative cosine-kernel residual to the same SE path.
The relative kernel uses cyclic distance, independent of each cycle's origin
and traversal direction. Changing the DFS tree can change either encoding.
No projector, factorization, dense cycle matrix or edge-pair matrix is formed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from research.cycle_pe.benchmark_models import ATOM_DIMS, BOND_DIMS, CategoricalEncoder, _pool
from research.cycle_pe.paper_model import _message_topology
from research.cycle_pe.v2.data import DATASETS, Batch

ENCODINGS = ("se", "pe")
MODEL_NAMES = {"se": "cycle_dfs_se_v2", "pe": "cycle_dfs_relative_pe_v2"}
MODEL_NAME = MODEL_NAMES["se"]


class LeftNullBasisEncoder(nn.Module):
    """Learned edge -> DFS cycle -> edge PE on one sparse disjoint batch.

    The historical class name remains import-compatible. The input is unsigned
    sparse membership, not basis coordinates or a cycle-space projector.
    """

    def __init__(self, bond_dim: int, pe_dim: int, *, encoding: str = "se") -> None:
        super().__init__()
        if min(bond_dim, pe_dim) < 1:
            raise ValueError("bond_dim and pe_dim must be positive")
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        self.encoding = encoding
        self.bond_dim = bond_dim
        self.pe_dim = pe_dim
        self.column_phi = nn.Sequential(
            nn.Linear(bond_dim, pe_dim), nn.SiLU(), nn.Linear(pe_dim, pe_dim)
        )
        self.cycle_mlp = nn.Sequential(
            nn.Linear(pe_dim + 1, 2 * pe_dim),
            nn.SiLU(),
            nn.Linear(2 * pe_dim, pe_dim),
            nn.SiLU(),
        )
        self.edge_psi = nn.Sequential(
            nn.Linear(2 * pe_dim + 3, 2 * pe_dim),
            nn.SiLU(),
            nn.Linear(2 * pe_dim, pe_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(nn.Linear(pe_dim, pe_dim), nn.SiLU())

    @staticmethod
    def _sparse_mix(membership: Tensor, values: Tensor) -> Tensor:
        # Torch 2.7 CUDA COO mm stays FP32 while learned MLPs may use BF16.
        with torch.autocast(device_type=values.device.type, enabled=False):
            return torch.sparse.mm(membership.float(), values.float())

    def forward(
        self,
        bond: Tensor,
        cycle_membership: Tensor,
        cycle_lengths: Tensor,
        edge_cycle_counts: Tensor,
        edge_cycle_features: Tensor,
        cycle_position_values: Tensor | None = None,
    ) -> Tensor:
        if bond.ndim != 2 or bond.shape[1] != self.bond_dim:
            raise ValueError("bond embeddings must have shape [num_edges, bond_dim]")
        if (
            cycle_membership.layout != torch.sparse_coo
            or cycle_membership.ndim != 2
            or not cycle_membership.is_coalesced()
            or not cycle_membership.is_floating_point()
        ):
            raise ValueError("cycle_membership must be a coalesced floating sparse COO matrix")
        edge_count, cycle_count = cycle_membership.shape
        if edge_count != bond.shape[0]:
            raise ValueError("cycle membership rows must align with bond embeddings")
        if cycle_lengths.shape != (cycle_count,) or edge_cycle_counts.shape != (edge_count,):
            raise ValueError("cycle lengths and edge cycle counts must match sparse membership")
        if edge_cycle_features.shape != (edge_count, 2):
            raise ValueError("edge_cycle_features must have shape [num_edges, 2]")
        if any(
            value.device != bond.device
            for value in (cycle_membership, cycle_lengths, edge_cycle_counts, edge_cycle_features)
        ):
            raise ValueError(
                "cycle membership and normalization tensors must share the bond device"
            )
        if any(
            not value.is_floating_point()
            for value in (cycle_lengths, edge_cycle_counts, edge_cycle_features)
        ):
            raise ValueError("cycle lengths, counts and edge features must be floating point")

        if self.encoding == "pe":
            if cycle_position_values is None:
                raise ValueError(
                    "PE requires cycle_position_values; missing positions cannot use SE"
                )
            if (
                cycle_position_values.shape != (2, cycle_membership._nnz())
                or not cycle_position_values.is_floating_point()
                or cycle_position_values.device != bond.device
            ):
                raise ValueError(
                    "cycle_position_values must be floating [2, nnz] on the bond device"
                )

        values = self.column_phi(bond)
        lengths = cycle_lengths.float()
        counts = edge_cycle_counts.float()
        active = (counts > 0).float()[:, None]
        safe_counts = counts.clamp_min(1.0)[:, None]
        log_lengths = lengths.log1p()[:, None]
        cycle_sum = self._sparse_mix(cycle_membership.transpose(0, 1), values)
        cycle_mean = cycle_sum / lengths.clamp_min(1.0)[:, None]
        cycle_hidden = self.cycle_mlp(torch.cat((cycle_mean, log_lengths), dim=1))

        # Sparse block-diagonal products handle every cycle of every graph in
        # the physical batch without a graphwise loop or dense E x beta tensor.
        edge_cycle_mean = self._sparse_mix(cycle_membership, cycle_hidden) / safe_counts
        if self.encoding == "pe":
            # Shared sorted COO indices preserve the selected cycle chart. No
            # feature tensor indexed by every edge-cycle incidence is formed.
            indices = cycle_membership.indices()
            cosine = torch.sparse_coo_tensor(
                indices, cycle_position_values[0].float(), cycle_membership.shape,
                is_coalesced=True, check_invariants=False,
            )
            sine = torch.sparse_coo_tensor(
                indices, cycle_position_values[1].float(), cycle_membership.shape,
                is_coalesced=True, check_invariants=False,
            )
            cosine_moment = self._sparse_mix(cosine.transpose(0, 1), values)
            sine_moment = self._sparse_mix(sine.transpose(0, 1), values)
            cosine_moment = cosine_moment / lengths.clamp_min(1.0)[:, None]
            sine_moment = sine_moment / lengths.clamp_min(1.0)[:, None]
            positional_residual = (
                self._sparse_mix(cosine, cosine_moment)
                + self._sparse_mix(sine, sine_moment)
            ) / safe_counts
            # cos(theta_e-theta_f) is reconstructed before nonlinear layers.
            # This equals (K_[1+cos] - K_mean) @ values, averaged over each
            # edge's cycles. The existing nonlinear SE path stays unchanged.
            edge_cycle_mean = edge_cycle_mean + positional_residual
        features = torch.cat(
            (
                values.float() * active,
                edge_cycle_mean,
                counts.log1p()[:, None],
                # Static mean log-length/inverse-length features come from
                # preparation/cache rather than a third sparse product here.
                edge_cycle_features.float(),
            ),
            dim=1,
        )
        encoded = self.output(self.edge_psi(features))
        # Empty-cycle sparse products retain an autograd path: forest PE is
        # exactly zero and PE parameter gradients are zero, not disconnected.
        return encoded * active.to(encoded.dtype)


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
    """Sparse DFS-cycle PE with the unchanged deep molecular graph backbone."""

    def __init__(
        self,
        *,
        dataset: str,
        encoding: str = "se",
        hidden: int = 128,
        pe_dim: int = 64,
        layers: int = 10,
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
        self.node_encoder = CategoricalEncoder((28,) if dataset == "zinc12k" else ATOM_DIMS, hidden)
        self.bond_encoder = CategoricalEncoder((4,) if dataset == "zinc12k" else BOND_DIMS, hidden)
        self.encoding = encoding
        self.pe_encoder = LeftNullBasisEncoder(hidden, pe_dim, encoding=encoding)
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
        node = self.node_encoder(batch.x)
        bond = self.bond_encoder(batch.edge_attr)
        positional = self.pe_encoder(
            bond,
            batch.cycle_membership,
            batch.cycle_lengths,
            batch.edge_cycle_counts,
            batch.edge_cycle_features,
            batch.cycle_position_values,
        )
        edge = self.edge_encoder(torch.cat((bond, positional), dim=1))
        edge_index = batch.edge_index.T
        topology = _message_topology(node, edge_index)
        for layer in self.layers:
            node, edge = layer(node, edge, edge_index, topology=topology)
        node = self.final_node_norm(node)
        edge = self.final_edge_norm(edge)
        node_mean, node_max = _pool(node, batch.batch, graph_count)
        node_sizes = (batch.ptr[1:] - batch.ptr[:-1]).to(node.dtype)[:, None]
        node_sum = node_mean * node_sizes
        edge_mean, edge_max = _pool(edge, batch.edge_graph_index, graph_count)
        edge_sizes = (batch.edge_ptr[1:] - batch.edge_ptr[:-1]).to(edge.dtype)[:, None]
        edge_sum = edge_mean * edge_sizes
        pooled = torch.cat((node_sum, node_mean, node_max, edge_sum, edge_mean, edge_max), dim=1)
        return self.graph_head(self.graph_trunk(pooled))


def architecture_protocol(encoding: str = "se") -> dict[str, str]:
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    return {
        "model": MODEL_NAMES[encoding],
        "encoding": encoding,
        "cycle_space": (
            "complete signed sparse DFS fundamental basis Z of ker(B.T); "
            "beta=m-n+components; every selected cycle retained without truncation"
        ),
        "positional_encoding": (
            "SE: unsigned sparse membership A=abs(Z); shared learned bond values "
            "aggregate edge-to-cycle by mean, combine with log cycle length through a "
            "shared cycle MLP, then aggregate cycle-to-edge with cycle-count normalization"
            if encoding == "se"
            else "PE: retain the identical SE path and add the cycle-relative residual "
            "sum_j mean_f cos(2*pi*(t_ej-t_fj)/L_j)*phi(bond_f), averaged over edge cycles; "
            "this is K_[1+cos] minus K_mean applied to learned bond values"
        ),
        "relative_position": (
            "disabled in SE; cycle structure summaries contain no within-cycle distance"
            if encoding == "se"
            else "first-harmonic cosine kernel of undirected cyclic distance; origin/reversal "
            "invariant, not general graph shortest-path distance or guaranteed unique positions; "
            "all selected cycles and their complete supports participate"
        ),
        "parameter_matching": (
            "SE and PE have identical learned modules and parameter counts; PE adds no gate "
            "or learned parameters; nonlinear layers follow invariant harmonic reconstruction"
        ),
        "symmetry": (
            "cycle-sign, column-order and cycle-origin/reversal invariant; "
            "edge/node permutation equivariant "
            "only when the selected basis is transported with the graph; a different "
            "DFS tree or arbitrary invertible Z->ZR may change this selected-cycle PE"
        ),
        "execution": (
            "one sparse block-diagonal physical batch; sparse edge-to-cycle-to-edge "
            "matrix products; no QR, SVD, eigendecomposition, Gram inverse, projector, "
            "dense edge-by-cycle matrix or edge-by-edge matrix"
        ),
        "backbone": (
            "deep pre-norm residual edge/node message blocks; gated directed messages; "
            "separate edge and node FFNs; LayerScale and dropout for stable deep training"
        ),
        "pe_injection": "concatenate the selected learned DFS-cycle encoding with bond embedding",
        "pooling": "node and edge sum/mean/max followed by a two-layer graph trunk",
        "forest_policy": (
            "edges outside all selected cycles, including bridges and forests, have "
            "exactly zero PE; empty sparse paths retain zero parameter gradients"
        ),
        "numeric_policy": (
            "sparse COO matrix products execute in FP32; learned cycle/edge MLPs and "
            "the deep backbone honor outer AMP with FP32 residual accumulation"
        ),
        "reference_comparison": (
            f"external published tables only; trains only {MODEL_NAMES[encoding]}"
        ),
    }


__all__ = [
    "ENCODINGS",
    "MODEL_NAMES",
    "MODEL_NAME",
    "CycleBasisPEModel",
    "LeftNullBasisEncoder",
    "architecture_protocol",
]
