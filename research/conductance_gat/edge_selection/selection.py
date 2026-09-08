"""Physical-edge selection, separate from positive V5 conductance amplitudes.

Budgeted gates use an exact hard top-k forward and a biased straight-through
gradient of the entropy-constrained relaxation sum(sigmoid((s-lambda)/T))=k.
The unconstrained corruption arm instead uses Louizos et al. hard-concrete
(https://arxiv.org/html/1712.01312v2, equations 10-12); it makes no exact-k claim.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from research.conductance_gat.v5.operator import graph_broadcast, graph_sum

CONDITIONS = (
    "full",
    "forest_only",
    "forest_random",
    "forest_learned",
    "forest_cycle",
    "hard_concrete",
)
LEARNED = {"forest_learned", "forest_cycle", "hard_concrete"}


def selection_configuration(configuration: dict) -> dict:
    defaults = {
        "chord_fraction": None,
        "selection_temperature": 1.0,
        "hard_concrete_temperature": 2 / 3,
        "hard_concrete_lower": -0.1,
        "hard_concrete_upper": 1.1,
        "selection_seed": 0,
    }
    unknown = set(configuration) - {"condition", *defaults}
    if unknown:
        raise ValueError(f"unsupported selection configuration: {sorted(unknown)}")
    result = {**defaults, **configuration}
    if result.get("condition") not in CONDITIONS:
        raise ValueError("an explicit supported edge selection condition is required")
    fraction = result["chord_fraction"]
    if fraction is not None and (
        isinstance(fraction, bool)
        or not isinstance(fraction, Real)
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("chord_fraction must be finite in [0,1]")
    if (
        result["condition"] in {"forest_random", "forest_learned", "forest_cycle"}
        and fraction is None
    ):
        raise ValueError("budgeted selection requires explicit chord_fraction")
    for key in ("selection_temperature", "hard_concrete_temperature"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    lower, upper = result["hard_concrete_lower"], result["hard_concrete_upper"]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in (lower, upper)
        )
        or not lower < 0 < 1 < upper
    ):
        raise ValueError("hard-concrete stretch must satisfy lower<0<1<upper")
    if type(result["selection_seed"]) is not int or result["selection_seed"] < 0:
        raise ValueError("selection_seed must be a nonnegative integer")
    return result


def _group_extreme(values, group, graphs, reduction):
    if graphs == 1:
        value = values.amin() if reduction == "amin" else values.amax()
        return value.reshape(1)
    initial = torch.inf if reduction == "amin" else -torch.inf
    result = values.new_full((graphs,), initial).scatter_reduce(0, group, values, reduce=reduction)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


class _ConstrainedLogistic(torch.autograd.Function):
    """Implicit first derivative of the fixed-cardinality entropy relaxation.

    The scalar threshold is solved per graph, vectorized on the device. Bisection
    is a numerical scalar root solve, not a reduction of the model's solver K.
    No per-graph GPU loop or dense graph-by-edge matrix is constructed.
    """

    @staticmethod
    def forward(ctx, scores, edge_graph, counts, budget, temperature):
        graphs = counts.numel()
        if not scores.numel():
            probability = scores.clone()
        else:
            lower = _group_extreme(scores, edge_graph, graphs, "amin") - 64 * temperature
            upper = _group_extreme(scores, edge_graph, graphs, "amax") + 64 * temperature
            for _ in range(64 if scores.dtype == torch.float64 else 32):
                threshold = (lower + upper) * 0.5
                probability = (
                    (scores - graph_broadcast(threshold, edge_graph, graphs, validate_index=False))
                    / temperature
                ).sigmoid()
                mass = graph_sum(probability, edge_graph, graphs, validate_index=False)
                above = mass > budget
                lower = torch.where(above, threshold, lower)
                upper = torch.where(above, upper, threshold)
            threshold = (lower + upper) * 0.5
            probability = (
                (scores - graph_broadcast(threshold, edge_graph, graphs, validate_index=False))
                / temperature
            ).sigmoid()
            probability = torch.where(
                budget[edge_graph] == 0, torch.zeros_like(probability), probability
            )
            probability = torch.where(
                budget[edge_graph] == counts[edge_graph], torch.ones_like(probability), probability
            )
        ctx.save_for_backward(probability, edge_graph)
        ctx.graphs, ctx.temperature = graphs, temperature
        return probability

    @staticmethod
    def backward(ctx, gradient):
        probability, edge_graph = ctx.saved_tensors
        slope = probability * (1 - probability)
        mass = graph_sum(slope, edge_graph, ctx.graphs, validate_index=False)
        weighted = graph_sum(slope * gradient, edge_graph, ctx.graphs, validate_index=False)
        offset = weighted / mass.clamp_min(torch.finfo(slope.dtype).tiny)
        result = (
            slope
            * (gradient - graph_broadcast(offset, edge_graph, ctx.graphs, validate_index=False))
            / ctx.temperature
        )
        return result, None, None, None, None


def exact_budget_gate(
    scores: Tensor,
    eligible: Tensor,
    edge_graph: Tensor,
    num_graphs: int,
    fraction: float,
    *,
    temperature: float,
    straight_through: bool,
    tie_breaker: Tensor | None = None,
):
    """Exactly floor(fraction * eligible edges) per graph in every forward."""
    if scores.ndim != 1 or eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("selection scores and eligibility must be aligned E vectors")
    if edge_graph.shape != scores.shape or edge_graph.dtype != torch.long:
        raise ValueError("edge_graph must be aligned int64 graph IDs")
    if tie_breaker is not None and (
        tie_breaker.shape != scores.shape or tie_breaker.device != scores.device
    ):
        raise ValueError("tie_breaker must align with current same-device physical edges")
    selected = eligible.nonzero(as_tuple=False).flatten()
    groups, logits = edge_graph[selected], scores[selected]
    counts = torch.bincount(groups, minlength=num_graphs)
    budget = (counts.to(torch.float64) * fraction).floor().long()
    hard = torch.zeros_like(scores)
    if selected.numel():
        order = (
            torch.arange(selected.numel(), device=scores.device)
            if tie_breaker is None
            else torch.argsort(tie_breaker[selected], descending=True, stable=True)
        )
        order = order[torch.argsort(logits[order], descending=True, stable=True)]
        order = order[torch.argsort(groups[order], stable=True)]
        starts = counts.cumsum(0) - counts
        rank = torch.arange(selected.numel(), device=scores.device) - starts[groups[order]]
        hard = hard.scatter(0, selected[order], (rank < budget[groups[order]]).to(scores.dtype))
    soft = torch.zeros_like(scores)
    if straight_through:
        probability = _ConstrainedLogistic.apply(logits, groups, counts, budget, temperature)
        soft = soft.scatter(0, selected, probability)
        gate = hard + (soft - soft.detach())
    else:
        gate, soft = hard, hard
    return gate, soft, budget


def hard_concrete(logits, *, training, temperature=2 / 3, lower=-0.1, upper=1.1):
    probability = (logits - temperature * math.log(-lower / upper)).sigmoid()
    if training:
        eps = torch.finfo(logits.dtype).eps
        uniform = torch.rand_like(logits).clamp(min=eps, max=1 - eps)
        relaxed = ((uniform.log() - torch.log1p(-uniform) + logits) / temperature).sigmoid()
    else:
        # Paper deterministic evaluation rule, not an exact binary/top-k budget.
        relaxed = logits.sigmoid()
    gate = (relaxed * (upper - lower) + lower).clamp(0, 1)
    return gate, probability


class EdgeSelector(nn.Module):
    """Symmetric endpoint/degree scores, optionally contextualized by cycles."""

    def __init__(self, channels, configuration, *, edge_chunk_size=65536):
        super().__init__()
        self.configuration = selection_configuration(configuration)
        self.condition = self.configuration["condition"]
        self.edge_chunk_size = edge_chunk_size
        if self.condition in LEARNED:
            self.node_projection = nn.Linear(channels, channels, bias=False)
            self.edge_hidden = nn.Linear(2 * channels + 4, channels)
            # A graph-wide constant score is unidentifiable under an exact k.
            self.score = nn.Linear(channels, 1, bias=self.condition == "hard_concrete")
        else:
            self.node_projection = self.edge_hidden = self.score = None
        if self.condition == "forest_cycle":
            self.cycle_weight = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("cycle_weight", None)
        self.last_logits = self.last_gate = self.last_probability = None
        self.last_budget = None
        self.live_logits = self.live_probability = None
        self.gate_override = None

    def _scores_chunk(self, projected, incidence, sample_degree, full_degree):
        tail, head = incidence
        left, right = projected[tail], projected[head]
        structural = torch.stack(
            (
                (sample_degree[tail] + sample_degree[head]).log1p(),
                (sample_degree[tail] - sample_degree[head]).abs().log1p(),
                (full_degree[tail] + full_degree[head]).log1p(),
                (full_degree[tail] - full_degree[head]).abs().log1p(),
            ),
            dim=1,
        )
        features = torch.cat((left + right, (left - right).abs(), structural), dim=1)
        return self.score(F.silu(self.edge_hidden(features))).squeeze(-1)

    def forward(
        self, state, incidence, node_graph, num_graphs, *, topology, sample_degree, full_degree
    ):
        edges = incidence.shape[1]
        if (
            topology.incidence_edge_index.shape != incidence.shape
            or topology.incidence_edge_index.device != state.device
        ):
            raise ValueError("selection topology must match this same-device graph")
        if topology.forest_mask.shape != (edges,) or topology.edge_graph.shape != (edges,):
            raise ValueError("selection topology has invalid physical edge metadata")
        torch._assert_async(
            (topology.incidence_edge_index == incidence).all(),
            "selection topology belongs to different physical edges",
        )
        torch._assert_async(
            (topology.node_graph == node_graph).all(),
            "selection topology belongs to different graph batching",
        )
        self.live_logits = self.live_probability = None
        if self.condition in LEARNED:
            projected = self.node_projection(state)
            chunks = []
            for start in range(0, edges, self.edge_chunk_size):
                args = (
                    projected,
                    incidence[:, start : start + self.edge_chunk_size],
                    sample_degree,
                    full_degree,
                )
                if torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint

                    chunks.append(
                        checkpoint(
                            self._scores_chunk, *args, use_reentrant=False, preserve_rng_state=False
                        )
                    )
                else:
                    chunks.append(self._scores_chunk(*args))
            logits = torch.cat(chunks) if chunks else state.new_empty(0)
            if self.condition == "forest_cycle":
                from .topology import cycle_context

                # Scalar edge context keeps exact all-edge cycle aggregation
                # without retaining an E x hidden-channel feature cache.
                contextual = cycle_context(
                    logits[:, None], topology, signed=False, normalize=True
                ).squeeze(-1)
                logits = logits + self.cycle_weight * contextual
        else:
            logits = state.new_zeros(edges)
        torch._assert_async(torch.isfinite(logits).all(), "edge-selection logits must be finite")
        config = self.configuration
        budget = None
        if self.condition == "full":
            gate = probability = torch.ones_like(logits)
        elif self.condition == "forest_only":
            gate = probability = topology.forest_mask.to(logits.dtype)
        elif self.condition == "hard_concrete":
            gate, probability = hard_concrete(
                logits,
                training=self.training,
                temperature=config["hard_concrete_temperature"],
                lower=config["hard_concrete_lower"],
                upper=config["hard_concrete_upper"],
            )
        else:
            ranking = topology.random_priority if self.condition == "forest_random" else logits
            gate, probability, budget = exact_budget_gate(
                ranking,
                ~topology.forest_mask,
                topology.edge_graph,
                num_graphs,
                config["chord_fraction"],
                temperature=config["selection_temperature"],
                straight_through=self.training and self.condition in LEARNED,
                tie_breaker=topology.random_priority,
            )
            gate = torch.where(topology.forest_mask, torch.ones_like(gate), gate)
            probability = torch.where(
                topology.forest_mask, torch.ones_like(probability), probability
            )
            gate, probability = gate.to(logits.dtype), probability.to(logits.dtype)
        override = self.gate_override
        if override is not None:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("gate interventions are evaluation/no-grad only")
            if isinstance(override, Tensor):
                if override.shape != gate.shape or override.device != gate.device:
                    raise ValueError(
                        "frozen gate must align with current same-device physical edges"
                    )
                torch._assert_async(
                    (torch.isfinite(override) & (override >= 0) & (override <= 1)).all(),
                    "frozen gate must be finite in [0,1]",
                )
                gate = override.to(gate.dtype)
                probability = gate
            elif override == "all":
                gate = probability = torch.ones_like(gate)
            elif override == "random_budget":
                if self.condition not in {"forest_random", "forest_learned", "forest_cycle"}:
                    raise ValueError(
                        "random_budget intervention requires a protected exact-budget arm"
                    )
                gate, probability, budget = exact_budget_gate(
                    topology.random_priority,
                    ~topology.forest_mask,
                    topology.edge_graph,
                    num_graphs,
                    config["chord_fraction"],
                    temperature=config["selection_temperature"],
                    straight_through=False,
                )
                gate = torch.where(topology.forest_mask, torch.ones_like(gate), gate).to(
                    logits.dtype
                )
                probability = gate
            else:
                raise ValueError("unsupported gate intervention")
        self.last_logits, self.last_gate, self.last_probability = (
            logits.detach(),
            gate.detach(),
            probability.detach(),
        )
        self.last_budget = None if budget is None else budget.detach()
        if self.condition == "hard_concrete":
            self.live_logits, self.live_probability = logits, probability
        return gate

    def clear_auxiliary_cache(self):
        self.live_logits = self.live_probability = None
