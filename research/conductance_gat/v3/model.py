"""Graph-centered relative C, isotropic residual and separate mixing strength.

Degree features are log-degree sum/absolute difference: unlike concatenating
ordered endpoint degrees, these are invariant to reversing incidence orientation.
Scalar C is shared across feature channels. Checkpointed gate chunks recompute
edge features in backward; global centering/normalization remain differentiable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .operator import symmetric_propagation


def graph_mean(values: Tensor, edge_graph: Tensor, num_graphs: int) -> Tensor:
    """Full-graph scalar edge means; empty graphs have mean zero, never NaN."""
    sums = values.new_zeros(num_graphs).index_add(0, edge_graph, values)
    counts = values.new_zeros(num_graphs).index_add(0, edge_graph, torch.ones_like(values))
    return sums / counts.clamp_min(1)


class RelativeConductance(nn.Module):
    def __init__(self, channels: int, gate_mode: str = "relative", edge_chunk_size: int = 65536):
        super().__init__()
        if gate_mode not in {"relative", "fixed_one"}:
            raise ValueError(f"Unsupported relative-C gate mode: {gate_mode}")
        if (
            isinstance(edge_chunk_size, bool)
            or not isinstance(edge_chunk_size, int)
            or edge_chunk_size < 1
        ):
            raise ValueError("edge_chunk_size must be a positive integer")
        self.gate_mode = gate_mode
        self.edge_chunk_size = edge_chunk_size
        self.input_norm = nn.LayerNorm(4 * channels + 2)
        self.network = nn.Sequential(
            nn.Linear(4 * channels + 2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            # A common final bias is removed exactly by graph centering, so omit it.
            nn.Linear(channels, 1, bias=False),
        )
        nn.init.zeros_(self.network[-1].weight)
        self.raw_gamma = nn.Parameter(torch.zeros(()))
        self.raw_tau = nn.Parameter(torch.zeros(()))
        self.last_scores: Tensor | None = None
        self.last_centered_scores: Tensor | None = None
        if gate_mode == "fixed_one":
            self.requires_grad_(False)

    @property
    def gamma(self) -> Tensor:
        return self.raw_gamma.sigmoid()

    @property
    def tau(self) -> Tensor:
        return 2 * self.raw_tau.sigmoid()

    def _chunk_scores(self, state: Tensor, tail: Tensor, head: Tensor, log_degree: Tensor):
        left, right = state[tail], state[head]
        delta = right - left
        degree_left, degree_right = log_degree[tail], log_degree[head]
        features = torch.cat(
            (
                delta.abs(),
                delta.square(),
                left + right,
                left * right,
                (degree_left + degree_right)[:, None],
                (degree_left - degree_right).abs()[:, None],
            ),
            dim=1,
        )
        return self.network(self.input_norm(features)).squeeze(-1)

    def forward(
        self, state: Tensor, incidence: Tensor, node_graph: Tensor, num_graphs: int
    ) -> Tensor:
        tail, head = incidence
        edge_graph = node_graph[tail]
        if self.gate_mode == "fixed_one":
            self.last_scores = state.new_zeros(tail.numel())
            self.last_centered_scores = self.last_scores
            return state.new_ones(tail.numel())
        degree = state.new_zeros(state.shape[0])
        ones = state.new_ones(tail.numel())
        degree.index_add_(0, tail, ones)
        degree.index_add_(0, head, ones)
        log_degree = degree.log1p()
        chunks = []
        for start in range(0, tail.numel(), self.edge_chunk_size):
            stop = start + self.edge_chunk_size
            # Pass sliced indices explicitly. No late-bound start/stop closure is
            # used when checkpoint recomputes this particular chunk in backward.
            inputs = (state, tail[start:stop], head[start:stop], log_degree)
            if torch.is_grad_enabled():
                score = checkpoint(
                    self._chunk_scores, *inputs, use_reentrant=False, preserve_rng_state=False
                )
            else:
                score = self._chunk_scores(*inputs)
            chunks.append(score)
        scores = torch.cat(chunks) if chunks else state.new_empty((0,))
        if not bool(torch.isfinite(scores.detach()).all()):
            raise FloatingPointError("Nonfinite relative conductance scores")
        centered = scores - graph_mean(scores, edge_graph, num_graphs)[edge_graph]
        unnormalized = (self.tau * centered.tanh()).exp()
        relative = unnormalized / graph_mean(unnormalized, edge_graph, num_graphs)[edge_graph]
        c = (1 - self.gamma) + self.gamma * relative
        self.last_scores = scores.detach()
        self.last_centered_scores = centered.detach()
        return c


class RelativeCConv(nn.Module):
    def __init__(self, channels: int, gate_mode: str, edge_chunk_size: int = 65536):
        super().__init__()
        self.estimator = RelativeConductance(channels, gate_mode, edge_chunk_size)
        self.raw_alpha = nn.Parameter(torch.zeros(()))
        self.normalization = "symmetric"
        self.edge_chunk_size = edge_chunk_size

    @property
    def alpha(self) -> Tensor:
        return self.raw_alpha.sigmoid()

    def forward(
        self, x: Tensor, incidence: Tensor, node_graph: Tensor, num_graphs: int | None = None
    ) -> Tensor:
        if num_graphs is None:
            num_graphs = int(node_graph.max()) + 1 if node_graph.numel() else 0
        with torch.autocast(device_type=x.device.type, enabled=False):
            state = x if x.dtype == torch.float64 else x.float()
            c = self.estimator(state, incidence, node_graph, num_graphs)
            result = symmetric_propagation(
                state,
                c,
                incidence,
                self.alpha.to(dtype=state.dtype),
                edge_chunk_size=self.edge_chunk_size,
            )
        return result.to(x.dtype)


class RelativeCNodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        classes: int,
        *,
        normalization: str = "symmetric",
        hidden_channels: int = 64,
        layers: int = 2,
        dropout: float = 0.5,
        gate_mode: str = "relative",
        edge_chunk_size: int = 65536,
    ):
        super().__init__()
        if normalization != "symmetric":
            raise ValueError("Relative-C v3 keeps symmetric normalization fixed")
        if hidden_channels < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden width/layers must be positive and dropout in [0, 1)")
        self.normalization = normalization
        self.gate_mode = gate_mode
        self.dropout = dropout
        self.edge_chunk_size = edge_chunk_size
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)
        self.operators = nn.ModuleList(
            RelativeCConv(hidden_channels, gate_mode, edge_chunk_size) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(layers))

    def forward(self, graph: Any) -> Tensor:
        x, incidence = graph.x, graph.incidence_edge_index
        if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence must be a 2 x E int64 tensor")
        if incidence.device != x.device:
            raise ValueError("state and incidence must share a device")
        if incidence.numel() and (
            bool((incidence < 0).any()) or bool((incidence >= x.shape[0]).any())
        ):
            raise ValueError("incidence endpoint is outside the node matrix")
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            num_graphs = 1
        else:
            if (
                batch.dtype != torch.long
                or batch.shape != (x.shape[0],)
                or batch.device != x.device
            ):
                raise ValueError("batch must contain one same-device int64 graph index per node")
            if batch.numel() and bool((batch < 0).any()):
                raise ValueError("batch indices must be nonnegative")
            num_graphs = int(batch.max()) + 1 if batch.numel() else 0
        if incidence.numel() and not torch.equal(batch[incidence[0]], batch[incidence[1]]):
            raise ValueError("Edges must not connect different graphs in a batch")
        h = F.dropout(F.elu(self.encoder(x)), self.dropout, self.training)
        for operator, norm in zip(self.operators, self.norms, strict=True):
            h = operator(h, incidence, batch, num_graphs)
            h = F.dropout(F.elu(norm(h)), self.dropout, self.training)
        return self.decoder(h)
