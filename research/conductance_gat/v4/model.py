"""Relative graph-operator learning plus a standard spatial feature transform.

Each layer first estimates relative physical-edge conductance ``C(H)`` from the
pre-transform state. It then applies a bias-free per-layer matrix ``W`` to form
messages and evaluates symmetric conductance propagation:

    H' = (1 - alpha) H + alpha P_C(H W)

on nonisolated nodes, while isolates remain ``H``. Thus C changes the learned
graph metric/operator and W changes feature channels carried by spatial messages.
They are separately switchable in a matched 2x2 scaffold; no second edge scalar
is multiplied into C.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .operator import symmetric_spatial_propagation


def graph_mean(values: Tensor, edge_graph: Tensor, num_graphs: int) -> Tensor:
    """Return full-graph scalar edge means; empty graphs have mean zero."""

    if values.ndim != 1 or edge_graph.shape != values.shape or edge_graph.dtype != torch.long:
        raise ValueError("values and edge_graph must be aligned one-dimensional tensors")
    if edge_graph.device != values.device:
        raise ValueError("values and edge_graph must share a device")
    if isinstance(num_graphs, bool) or not isinstance(num_graphs, int) or num_graphs < 0:
        raise ValueError("num_graphs must be a nonnegative integer")
    if edge_graph.numel() and (
        bool((edge_graph < 0).any()) or bool((edge_graph >= num_graphs).any())
    ):
        raise ValueError("edge graph index is outside num_graphs")
    sums = values.new_zeros(num_graphs).index_add(0, edge_graph, values)
    counts = values.new_zeros(num_graphs).index_add(0, edge_graph, torch.ones_like(values))
    return sums / counts.clamp_min(1)


class RelativeConductance(nn.Module):
    """Shared orientation-invariant relative-C estimator copied from the v3 design."""

    def __init__(
        self,
        channels: int,
        gate_mode: str = "relative",
        edge_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if gate_mode not in {"relative", "fixed_one"}:
            raise ValueError(f"Unsupported relative-C gate mode: {gate_mode}")
        if (
            isinstance(edge_chunk_size, bool)
            or not isinstance(edge_chunk_size, int)
            or edge_chunk_size < 1
        ):
            raise ValueError("edge_chunk_size must be a positive integer")
        self.channels = channels
        self.gate_mode = gate_mode
        self.edge_chunk_size = edge_chunk_size
        self.input_norm = nn.LayerNorm(4 * channels + 2)
        self.network = nn.Sequential(
            nn.Linear(4 * channels + 2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            # A common final bias is removed exactly by graph centering.
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

    def _chunk_scores(
        self,
        state: Tensor,
        tail: Tensor,
        head: Tensor,
        log_degree: Tensor,
    ) -> Tensor:
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
        self,
        state: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int,
    ) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.channels:
            raise ValueError("state width does not match the conductance estimator")
        if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence must be a 2 x E int64 tensor")
        if node_graph.dtype != torch.long or node_graph.shape != (state.shape[0],):
            raise ValueError("node_graph must contain one int64 graph index per node")
        if any(value.device != state.device for value in (incidence, node_graph)):
            raise ValueError("state, incidence and node_graph must share a device")
        if isinstance(num_graphs, bool) or not isinstance(num_graphs, int) or num_graphs < 0:
            raise ValueError("num_graphs must be a nonnegative integer")
        if node_graph.numel() and (
            bool((node_graph < 0).any()) or bool((node_graph >= num_graphs).any())
        ):
            raise ValueError("node graph index is outside num_graphs")
        if incidence.numel() and (
            bool((incidence < 0).any()) or bool((incidence >= state.shape[0]).any())
        ):
            raise ValueError("incidence endpoint is outside the node matrix")
        if incidence.shape[1] and bool((incidence[0] == incidence[1]).any()):
            raise ValueError("physical incidence edges must not contain self loops")

        tail, head = incidence
        if tail.numel() and not torch.equal(node_graph[tail], node_graph[head]):
            raise ValueError("Edges must not connect different graphs in a batch")
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
            inputs = (state, tail[start:stop], head[start:stop], log_degree)
            if torch.is_grad_enabled():
                score = checkpoint(
                    self._chunk_scores,
                    *inputs,
                    use_reentrant=False,
                    preserve_rng_state=False,
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
        if not bool(torch.isfinite(c.detach()).all()) or not bool((c.detach() > 0).all()):
            raise FloatingPointError("Relative conductance must remain finite and positive")
        self.last_scores = scores.detach()
        self.last_centered_scores = centered.detach()
        return c


class SpatialMessageTransform(nn.Module):
    """Bias-free square W, identity initialized without consuming model RNG.

    In ``fixed_identity`` mode the allocated weight is frozen and the exact
    identity path is used. In ``learned`` mode the forward is evaluated as
    ``H + H(W-I)``. This is algebraically the ordinary ``H W`` map, gives W the
    standard linear-map gradient, and makes W=I reduce bit-for-bit to the v3
    message state rather than depending on a matrix-multiply implementation.
    """

    def __init__(self, channels: int, spatial_mode: str = "learned") -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if spatial_mode not in {"learned", "fixed_identity"}:
            raise ValueError(f"Unsupported spatial mode: {spatial_mode}")
        self.in_features = channels
        self.out_features = channels
        self.spatial_mode = spatial_mode
        identity = torch.eye(channels)
        self.weight = nn.Parameter(identity.clone(), requires_grad=spatial_mode == "learned")
        self.register_buffer("_identity", identity, persistent=False)

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.in_features:
            raise ValueError("state width does not match the spatial transform")
        if self.spatial_mode == "fixed_identity":
            return state
        # The residual form is exactly F.linear(state, weight) algebraically.
        return state + F.linear(state, self.weight - self._identity)


class RelativeCSpatialConv(nn.Module):
    """One v4 layer: estimate C from H, then propagate the spatial message H W."""

    def __init__(
        self,
        channels: int,
        gate_mode: str,
        spatial_mode: str,
        edge_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        if (
            isinstance(edge_chunk_size, bool)
            or not isinstance(edge_chunk_size, int)
            or edge_chunk_size < 1
        ):
            raise ValueError("edge_chunk_size must be a positive integer")
        # Keep the v3 allocation order. SpatialMessageTransform uses a
        # deterministic torch.eye and therefore does not advance model RNG.
        self.estimator = RelativeConductance(channels, gate_mode, edge_chunk_size)
        self.raw_alpha = nn.Parameter(torch.zeros(()))
        self.message_transform = SpatialMessageTransform(channels, spatial_mode)
        self.gate_mode = gate_mode
        self.spatial_mode = spatial_mode
        self.normalization = "symmetric"
        self.edge_chunk_size = edge_chunk_size

    @property
    def alpha(self) -> Tensor:
        return self.raw_alpha.sigmoid()

    def forward(
        self,
        x: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int | None = None,
    ) -> Tensor:
        if num_graphs is None:
            num_graphs = int(node_graph.max()) + 1 if node_graph.numel() else 0
        with torch.autocast(device_type=x.device.type, enabled=False):
            state = x if x.dtype == torch.float64 else x.float()
            # C must be a function of the pre-W state. Do not move this call
            # below message_transform: that would confound the two mechanisms.
            c = self.estimator(state, incidence, node_graph, num_graphs)
            message = self.message_transform(state)
            result = symmetric_spatial_propagation(
                state,
                message,
                c,
                incidence,
                self.alpha.to(dtype=state.dtype),
                edge_chunk_size=self.edge_chunk_size,
            )
        return result.to(x.dtype)


class RelativeCSpatialNodeClassifier(nn.Module):
    """V3-compatible node scaffold with factorial relative-C and spatial-W controls."""

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
        spatial_mode: str = "learned",
        edge_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        if normalization != "symmetric":
            raise ValueError("Relative-C spatial v4 keeps symmetric normalization fixed")
        for name, value in (
            ("in_channels", in_channels),
            ("classes", classes),
            ("hidden_channels", hidden_channels),
            ("layers", layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        if gate_mode not in {"relative", "fixed_one"}:
            raise ValueError(f"Unsupported relative-C gate mode: {gate_mode}")
        if spatial_mode not in {"learned", "fixed_identity"}:
            raise ValueError(f"Unsupported spatial mode: {spatial_mode}")
        if (
            isinstance(edge_chunk_size, bool)
            or not isinstance(edge_chunk_size, int)
            or edge_chunk_size < 1
        ):
            raise ValueError("edge_chunk_size must be a positive integer")

        self.in_channels = in_channels
        self.classes = classes
        self.normalization = normalization
        self.gate_mode = gate_mode
        self.spatial_mode = spatial_mode
        self.dropout = float(dropout)
        self.edge_chunk_size = edge_chunk_size
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)
        self.operators = nn.ModuleList(
            RelativeCSpatialConv(
                hidden_channels,
                gate_mode,
                spatial_mode,
                edge_chunk_size,
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(layers))

    def forward(self, graph: Any) -> Tensor:
        x, incidence = graph.x, graph.incidence_edge_index
        if x.ndim != 2 or x.shape[1] != self.in_channels or not x.is_floating_point():
            raise ValueError("graph.x must be a floating node matrix with the configured width")
        if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence must be a 2 x E int64 tensor")
        if incidence.device != x.device:
            raise ValueError("state and incidence must share a device")
        if incidence.numel() and (
            bool((incidence < 0).any()) or bool((incidence >= x.shape[0]).any())
        ):
            raise ValueError("incidence endpoint is outside the node matrix")
        if incidence.shape[1] and bool((incidence[0] == incidence[1]).any()):
            raise ValueError("physical incidence edges must not contain self loops")

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


# Concise aliases for callers that describe this design as the v4 hybrid model.
HybridCSpatialConv = RelativeCSpatialConv
HybridCSpatialNodeClassifier = RelativeCSpatialNodeClassifier
