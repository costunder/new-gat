"""Our static cycle-set PE attached to this track's existing edge-aware GNN.

The downstream message layers are reused from paper_model, not a separately run
GINE/GAT/SignNet/PEARL baseline. Official categorical atom/bond features remain
inputs; the cycle encoding is the existing fixed-BFS set representation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from research.cycle_pe.benchmark_data import DATASETS, Batch
from research.cycle_pe.features import SET_STAT_NAMES
from research.cycle_pe.paper_model import _MessageLayer

MODEL_NAME = "cycle_set"
ATOM_DIMS = (119, 4, 12, 12, 10, 6, 6, 2, 2)
BOND_DIMS = (5, 6, 2)


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities: tuple[int, ...], output: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(width, output) for width in cardinalities])

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != len(self.embeddings):
            raise ValueError("categorical input field count disagrees with official schema")
        return torch.stack([layer(x[:, i]) for i, layer in enumerate(self.embeddings)]).sum(0)


def _pool(values: Tensor, assignment: Tensor, count: int) -> tuple[Tensor, Tensor]:
    total = values.new_zeros((count, values.shape[1])).index_add(0, assignment, values)
    sizes = torch.bincount(assignment, minlength=count).clamp_min(1).unsqueeze(1)
    maximum = values.new_full((count, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(
        0, assignment[:, None].expand_as(values), values, reduce="amax", include_self=True
    )
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return total / sizes, maximum


class CyclePEModel(nn.Module):
    """Only our cycle-set model; no architecture selector or competing run."""

    def __init__(self, *, dataset: str, hidden: int = 64, pe_dim: int = 32, layers: int = 3):
        super().__init__()
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {dataset}")
        self.node_encoder = CategoricalEncoder((28,) if dataset == "zinc12k" else ATOM_DIMS, hidden)
        self.bond_encoder = CategoricalEncoder((4,) if dataset == "zinc12k" else BOND_DIMS, hidden)
        self.pe_encoder = nn.Sequential(
            nn.Linear(len(SET_STAT_NAMES), pe_dim), nn.GELU(), nn.Linear(pe_dim, pe_dim)
        )
        self.edge_encoder = nn.Sequential(nn.Linear(hidden + pe_dim, hidden), nn.GELU())
        # This is the existing track backbone, including symmetric edge updates,
        # bidirectional messages, degree-normalized aggregation and LayerNorm.
        self.layers = nn.ModuleList(_MessageLayer(hidden) for _ in range(layers))
        self.graph_trunk = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.graph_head = nn.Linear(hidden, 1 if dataset == "zinc12k" else 11)

    def forward(self, batch: Batch) -> Tensor:
        node = self.node_encoder(batch.x)
        edge = self.edge_encoder(
            torch.cat((self.bond_encoder(batch.edge_attr), self.pe_encoder(batch.cycle_set)), dim=1)
        )
        # _MessageLayer consumes exactly one representative per undirected bond.
        # Stable FP32 scatter arithmetic is retained under optional AMP; heads
        # and feature encoders may use autocast.
        with torch.autocast(device_type=node.device.type, enabled=False):
            node, edge = node.float(), edge.float()
            for layer in self.layers:
                node, edge = layer(node, edge, batch.edge_index.T)
            graph_count = len(batch.ptr) - 1
            node_mean, node_max = _pool(node, batch.batch, graph_count)
            edge_graph = batch.batch[batch.edge_index[0]]
            edge_mean, edge_max = _pool(edge, edge_graph, graph_count)
            pooled = torch.cat((node_mean, node_max, edge_mean, edge_max), dim=1)
        return self.graph_head(self.graph_trunk(pooled))


def architecture_protocol() -> dict[str, str]:
    return {
        "model": MODEL_NAME,
        "positional_encoding": (
            "existing BFS fundamental-cycle basis and cycle_set_statistics; "
            "six sign/column-order-invariant summaries, GELU MLP"
        ),
        "backbone": (
            "existing cycle_pe.paper_model._MessageLayer edge-aware GNN; "
            "not a separate external-model baseline"
        ),
        "pe_injection": "concatenate learned cycle-set PE with categorical bond embedding",
        "pooling": "node mean/max and edge mean/max, then graph MLP",
        "cycle_symmetry": (
            "invariant to cycle-column signs/order; conditional on fixed BFS chart, "
            "not arbitrary chart replacement"
        ),
        "reference_comparison": (
            "external published tables only; this executable trains only our cycle-set model"
        ),
        "numeric_policy": "message layers and scatter pooling stay FP32 under optional AMP",
    }
