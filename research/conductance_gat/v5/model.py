"""Research-scale graph-conditioned shared-conductance network (V5)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .operator import graph_weighted_mean, shared_head_diffusion
from .protocol import (
    DEFAULT_BETA_INITIAL,
    DEFAULT_BETA_PARAMETERIZATION,
    beta_configuration,
)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _graph_node_mean(values: Tensor, node_graph: Tensor, num_graphs: int) -> Tensor:
    sums = values.new_zeros((num_graphs, values.shape[1])).index_add(0, node_graph, values)
    counts = values.new_zeros(num_graphs).index_add(0, node_graph, values.new_ones(values.shape[0]))
    return sums / counts.clamp_min(1)[:, None]


def _node_degree(state: Tensor, incidence: Tensor) -> Tensor:
    tail, head = incidence
    ones = state.new_ones(tail.numel())
    return state.new_zeros(state.shape[0]).index_add(0, tail, ones).index_add(0, head, ones)


def graph_context_features(
    state: Tensor,
    incidence: Tensor,
    node_graph: Tensor,
    num_graphs: int,
    full_degree: Tensor | None = None,
    graph_structure: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool clean hidden state and local/original structural statistics."""

    sample_degree = _node_degree(state, incidence)
    if full_degree is None:
        full_degree = sample_degree
    if full_degree.shape != sample_degree.shape or full_degree.device != state.device:
        raise ValueError("full_degree must contain one same-device value per node")
    full_degree = full_degree.to(state.dtype)
    mean = _graph_node_mean(state, node_graph, num_graphs)
    second = _graph_node_mean(state.square(), node_graph, num_graphs)
    std = (second - mean.square()).clamp_min(0).sqrt()
    coverage = sample_degree / full_degree.clamp_min(1)
    coverage_mean = _graph_node_mean(coverage[:, None], node_graph, num_graphs)
    coverage_std = (
        (
            _graph_node_mean(coverage.square()[:, None], node_graph, num_graphs)
            - coverage_mean.square()
        )
        .clamp_min(0)
        .sqrt()
    )
    if graph_structure is None:
        node_count = state.new_zeros(num_graphs).index_add(
            0, node_graph, state.new_ones(state.shape[0])
        )
        edge_graph = node_graph[incidence[0]]
        edge_count = state.new_zeros(num_graphs).index_add(
            0, edge_graph, state.new_ones(edge_graph.numel())
        )
        log_degree = full_degree.log1p()[:, None]
        degree_mean = _graph_node_mean(log_degree, node_graph, num_graphs)
        degree_std = (
            (_graph_node_mean(log_degree.square(), node_graph, num_graphs) - degree_mean.square())
            .clamp_min(0)
            .sqrt()
        )
        density = 2 * edge_count / (node_count * (node_count - 1)).clamp_min(1)
        graph_structure = torch.stack(
            (
                node_count.log1p(),
                edge_count.log1p(),
                degree_mean[:, 0],
                degree_std[:, 0],
                edge_count / node_count.clamp_min(1),
                density,
            ),
            dim=1,
        )
    if graph_structure.shape != (num_graphs, 6) or graph_structure.device != state.device:
        raise ValueError("graph_structure must be a same-device num_graphs x 6 tensor")
    return (
        torch.cat((mean, std, graph_structure.to(state.dtype), coverage_mean, coverage_std), dim=1),
        sample_degree,
        full_degree,
    )


class GraphConditionedConductance(nn.Module):
    """One symmetric positive relative-C field shared by every feature head."""

    def __init__(
        self,
        channels: int,
        *,
        mode: str = "dynamic",
        score_channels: int | None = None,
        max_log_conductance: float = 2.0,
        initial_score_std: float = 1.0e-3,
        edge_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        _positive_int("channels", channels)
        if mode not in {"dynamic", "fixed_one"}:
            raise ValueError(f"unsupported conductance mode: {mode}")
        score_channels = score_channels or max(32, min(128, channels // 2))
        _positive_int("score_channels", score_channels)
        if not math.isfinite(max_log_conductance) or max_log_conductance <= 0:
            raise ValueError("max_log_conductance must be finite and positive")
        if not math.isfinite(initial_score_std) or initial_score_std <= 0:
            raise ValueError("initial_score_std must be finite and positive")
        _positive_int("edge_chunk_size", edge_chunk_size)
        self.channels = channels
        self.mode = mode
        self.max_log_conductance = float(max_log_conductance)
        self.edge_chunk_size = edge_chunk_size
        self.node_projection = nn.Linear(channels, score_channels)
        self.context_projection = nn.Linear(2 * channels + 8, score_channels)
        edge_width = 4 * score_channels + 8
        self.score_norm = nn.LayerNorm(edge_width)
        self.score_network = nn.Sequential(
            nn.Linear(edge_width, 2 * score_channels),
            nn.SiLU(),
            nn.Linear(2 * score_channels, score_channels),
            nn.SiLU(),
            nn.Linear(score_channels, 1, bias=False),
        )
        # Nonzero identity-near init fixes V4's first-step dead-gradient path.
        nn.init.normal_(self.score_network[-1].weight, 0.0, initial_score_std)
        self.last_scores: Tensor | None = None
        self.last_log_c: Tensor | None = None
        self.last_c: Tensor | None = None
        self.override: str | None = None
        if mode == "fixed_one":
            self.requires_grad_(False)

    def _score_chunk(
        self,
        projected: Tensor,
        context: Tensor,
        tail: Tensor,
        head: Tensor,
        sample_degree: Tensor,
        full_degree: Tensor,
        edge_graph: Tensor,
    ) -> Tensor:
        left, right = projected[tail], projected[head]
        sample_ratio = (
            torch.stack(
                (
                    sample_degree[tail] / full_degree[tail].clamp_min(1),
                    sample_degree[head] / full_degree[head].clamp_min(1),
                ),
                dim=1,
            )
            .sort(dim=1)
            .values
        )
        inv_full = (
            torch.stack(
                (
                    torch.where(full_degree[tail] > 0, full_degree[tail].reciprocal(), 0),
                    torch.where(full_degree[head] > 0, full_degree[head].reciprocal(), 0),
                ),
                dim=1,
            )
            .sort(dim=1)
            .values
        )
        local = torch.cat(
            (
                torch.stack(
                    (
                        (sample_degree[tail] + sample_degree[head]).log1p(),
                        (sample_degree[tail] - sample_degree[head]).abs().log1p(),
                        (full_degree[tail] + full_degree[head]).log1p(),
                        (full_degree[tail] - full_degree[head]).abs().log1p(),
                    ),
                    dim=1,
                ),
                sample_ratio,
                inv_full,
            ),
            dim=1,
        )
        features = torch.cat(
            (
                left + right,
                (left - right).abs(),
                left * right,
                context[edge_graph],
                local,
            ),
            dim=1,
        )
        return self.score_network(self.score_norm(features)).squeeze(-1)

    def forward(
        self,
        state: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int,
        *,
        graph_context: Tensor,
        sample_degree: Tensor,
        full_degree: Tensor,
        edge_normalization_weight: Tensor | None = None,
    ) -> Tensor:
        tail, head = incidence
        edge_graph = node_graph[tail]
        if self.mode == "fixed_one" or self.override == "ones":
            c = state.new_ones(tail.numel())
            self.last_scores = state.new_zeros(tail.numel())
            self.last_log_c = state.new_zeros(tail.numel())
            self.last_c = c.detach()
            return c
        projected = F.silu(self.node_projection(state))
        context = F.silu(self.context_projection(graph_context))
        scores = (
            torch.cat(
                [
                    self._score_chunk(
                        projected,
                        context,
                        tail[start : start + self.edge_chunk_size],
                        head[start : start + self.edge_chunk_size],
                        sample_degree,
                        full_degree,
                        edge_graph[start : start + self.edge_chunk_size],
                    )
                    for start in range(0, tail.numel(), self.edge_chunk_size)
                ]
            )
            if tail.numel()
            else state.new_empty(0)
        )
        if edge_normalization_weight is None:
            edge_normalization_weight = torch.ones_like(scores)
        centered = (
            scores
            - graph_weighted_mean(scores, edge_graph, num_graphs, edge_normalization_weight)[
                edge_graph
            ]
        )
        log_c = self.max_log_conductance * torch.tanh(centered / self.max_log_conductance)
        raw_c = log_c.exp()
        c = raw_c / graph_weighted_mean(raw_c, edge_graph, num_graphs, edge_normalization_weight)[
            edge_graph
        ].clamp_min(torch.finfo(state.dtype).tiny)
        if self.override == "mean":
            c = graph_weighted_mean(c, edge_graph, num_graphs, edge_normalization_weight)[
                edge_graph
            ]
        elif self.override == "shuffle" and c.numel() > 1:
            replacement = c.clone()
            for graph_id in range(num_graphs):
                selected = (edge_graph == graph_id).nonzero(as_tuple=False).flatten()
                replacement[selected] = c[selected.flip(0)]
            c = replacement
        elif self.override not in {None, "mean", "shuffle"}:
            raise ValueError(f"unsupported C intervention: {self.override}")
        self.last_scores, self.last_log_c, self.last_c = scores.detach(), log_c.detach(), c.detach()
        return c


class GraphConditionedBeta(nn.Module):
    """Predict a graph-specific diffusion magnitude for each W head."""

    def __init__(
        self,
        channels: int,
        heads: int,
        *,
        beta_parameterization: str = DEFAULT_BETA_PARAMETERIZATION,
        beta_initial: float = DEFAULT_BETA_INITIAL,
        beta_min: float | None = None,
        beta_max: float | None = None,
    ) -> None:
        super().__init__()
        configuration = beta_configuration(beta_parameterization, beta_initial, beta_min, beta_max)
        width = max(32, min(128, channels // 2))
        self.network = nn.Sequential(
            nn.LayerNorm(2 * channels + 8),
            nn.Linear(2 * channels + 8, width),
            nn.SiLU(),
            nn.Linear(width, heads),
        )
        nn.init.normal_(self.network[-1].weight, 0.0, 1.0e-3)
        initial_fraction = float(configuration["beta_initial"])
        if beta_parameterization == "margin_sigmoid":
            initial_fraction = (initial_fraction - float(configuration["beta_min"])) / (
                float(configuration["beta_max"]) - float(configuration["beta_min"])
            )
        initial_logit = math.log(initial_fraction / (1 - initial_fraction))
        nn.init.constant_(self.network[-1].bias, initial_logit)
        self.beta_parameterization = beta_parameterization
        self.beta_initial = float(configuration["beta_initial"])
        self.beta_min = float(configuration["beta_min"]) if "beta_min" in configuration else None
        self.beta_max = float(configuration["beta_max"]) if "beta_max" in configuration else None

    def forward(self, context: Tensor) -> Tensor:
        beta = self.network(context).sigmoid()
        if self.beta_parameterization == "margin_sigmoid":
            beta = self.beta_min + (self.beta_max - self.beta_min) * beta
        return beta


class SharedConductanceMultihead(nn.Module):
    """One shared C with head-specific spatial W and graph-conditioned beta."""

    def __init__(
        self,
        channels: int,
        heads: int,
        *,
        conductance_mode: str,
        max_log_conductance: float,
        beta_parameterization: str,
        beta_initial: float,
        beta_min: float | None,
        beta_max: float | None,
        edge_chunk_size: int,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("hidden channels must be divisible by heads")
        self.channels, self.heads, self.head_width = channels, heads, channels // heads
        self.conductance_mode = conductance_mode
        self.estimator = GraphConditionedConductance(
            channels,
            mode=conductance_mode,
            max_log_conductance=max_log_conductance,
            edge_chunk_size=edge_chunk_size,
        )
        self.value_weight = nn.Parameter(torch.empty(heads, channels, self.head_width))
        nn.init.xavier_uniform_(self.value_weight.reshape(channels, channels))
        self.output_projection = nn.Linear(channels, channels, bias=False)
        self.beta_estimator = GraphConditionedBeta(
            channels,
            heads,
            beta_parameterization=beta_parameterization,
            beta_initial=beta_initial,
            beta_min=beta_min,
            beta_max=beta_max,
        )
        self.edge_chunk_size = edge_chunk_size
        self.last_beta: Tensor | None = None
        self.last_sampling_correction: Tensor | None = None

    def forward(
        self,
        state: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int,
        *,
        full_degree: Tensor | None,
        graph_structure: Tensor | None,
        edge_normalization_weight: Tensor | None,
        sampling_correction: Tensor | None,
    ) -> Tensor:
        # Dynamic-C geometry stays FP32 under an outer BF16 autocast region.
        # This includes its score network, centering/exp gauge and beta sigmoid.
        with torch.autocast(device_type=state.device.type, enabled=False):
            fp32_state = state.float()
            context, sample_degree, full_degree = graph_context_features(
                fp32_state, incidence, node_graph, num_graphs, full_degree, graph_structure
            )
            c = self.estimator(
                fp32_state,
                incidence,
                node_graph,
                num_graphs,
                graph_context=context,
                sample_degree=sample_degree,
                full_degree=full_degree,
                edge_normalization_weight=edge_normalization_weight,
            )
            beta = self.beta_estimator(context)
        value = torch.einsum("nd,hdk->nhk", state, self.value_weight)
        diffused = shared_head_diffusion(
            value,
            c,
            incidence,
            node_graph,
            beta,
            sampling_correction=sampling_correction,
            edge_chunk_size=self.edge_chunk_size,
        )
        self.last_beta = beta.detach()
        self.last_sampling_correction = (
            None if sampling_correction is None else sampling_correction.detach()
        )
        return self.output_projection(diffused.reshape(state.shape[0], self.channels))


class SwiGLU(nn.Module):
    def __init__(self, channels: int, multiplier: int) -> None:
        super().__init__()
        inner = channels * multiplier
        self.input_projection = nn.Linear(channels, 2 * inner)
        self.output_projection = nn.Linear(inner, channels)

    def forward(self, state: Tensor) -> Tensor:
        gate, value = self.input_projection(state).chunk(2, dim=-1)
        return self.output_projection(F.silu(gate) * value)


class ConductanceBlock(nn.Module):
    """Pre-norm operator residual followed by a pre-norm SwiGLU residual."""

    def __init__(
        self,
        channels: int,
        heads: int,
        ffn_multiplier: int,
        dropout: float,
        **operator_kwargs: Any,
    ) -> None:
        super().__init__()
        self.operator_norm = nn.LayerNorm(channels)
        self.operator = SharedConductanceMultihead(channels, heads, **operator_kwargs)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = SwiGLU(channels, ffn_multiplier)
        self.dropout = float(dropout)

    def forward(self, state: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        clean = self.operator_norm(state)
        state = state + F.dropout(
            self.operator(clean, *args, **kwargs), self.dropout, self.training
        )
        return state + F.dropout(self.ffn(self.ffn_norm(state)), self.dropout, self.training)


class GraphConditionedConductanceNodeClassifier(nn.Module):
    """Deep V5 classifier shared by fixed-C and graph-conditioned-C arms."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        *,
        hidden_channels: int = 256,
        layers: int = 8,
        heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.2,
        conductance_mode: str = "dynamic",
        max_log_conductance: float = 2.0,
        beta_parameterization: str = DEFAULT_BETA_PARAMETERIZATION,
        beta_initial: float = DEFAULT_BETA_INITIAL,
        beta_min: float | None = None,
        beta_max: float | None = None,
        edge_chunk_size: int = 65536,
        activation_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("classes", classes),
            ("hidden_channels", hidden_channels),
            ("layers", layers),
            ("heads", heads),
            ("ffn_multiplier", ffn_multiplier),
            ("edge_chunk_size", edge_chunk_size),
        ):
            _positive_int(name, value)
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        self.in_channels, self.classes = in_channels, classes
        self.hidden_channels, self.layers, self.heads = hidden_channels, layers, heads
        self.ffn_multiplier, self.dropout = ffn_multiplier, float(dropout)
        self.conductance_mode = conductance_mode
        self.activation_checkpoint = bool(activation_checkpoint)
        self.input_norm = nn.LayerNorm(in_channels)
        self.encoder = nn.Linear(in_channels, hidden_channels)
        operator_kwargs = {
            "conductance_mode": conductance_mode,
            "max_log_conductance": max_log_conductance,
            "beta_parameterization": beta_parameterization,
            "beta_initial": beta_initial,
            "beta_min": beta_min,
            "beta_max": beta_max,
            "edge_chunk_size": edge_chunk_size,
        }
        self.blocks = nn.ModuleList(
            ConductanceBlock(
                hidden_channels, heads, ffn_multiplier, self.dropout, **operator_kwargs
            )
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)

    @property
    def operators(self) -> list[SharedConductanceMultihead]:
        return [block.operator for block in self.blocks]

    def forward(self, graph: Any) -> Tensor:
        x, incidence = graph.x, graph.incidence_edge_index
        if x.ndim != 2 or x.shape[1] != self.in_channels or not x.is_floating_point():
            raise ValueError("graph.x must be a floating node matrix with configured width")
        if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence_edge_index must be a 2 x E int64 tensor")
        if incidence.device != x.device:
            raise ValueError("graph state and incidence must share a device")
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
                raise ValueError("batch must contain one graph index per node")
            num_graphs = getattr(graph, "_v5_num_graphs", None)
            if isinstance(num_graphs, bool) or not isinstance(num_graphs, int) or num_graphs < 1:
                raise ValueError("batched graphs must carry a positive CPU-declared _v5_num_graphs")
        h = self.encoder(self.input_norm(x))
        kwargs = {
            "full_degree": getattr(graph, "full_degree", None),
            "graph_structure": getattr(graph, "graph_structure", None),
            "edge_normalization_weight": getattr(graph, "edge_normalization_weight", None),
            "sampling_correction": getattr(graph, "sampling_correction", None),
        }
        for block in self.blocks:
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                h = checkpoint(
                    lambda value, layer=block: layer(value, incidence, batch, num_graphs, **kwargs),
                    h,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                h = block(h, incidence, batch, num_graphs, **kwargs)
        return self.decoder(self.final_norm(h))


ConductanceV5 = GraphConditionedConductanceNodeClassifier
V5NodeClassifier = GraphConditionedConductanceNodeClassifier
