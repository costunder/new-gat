"""Full V5 backbone with paired lift placement and cross-hop energy terms."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from research.conductance_gat.edge_selection.model import (
    EdgeSelectionClassifier,
    EdgeSelectionOperator,
)
from research.conductance_gat.v5.model import _static_graph_context, graph_context_features
from research.conductance_gat.v5.operator import shared_head_diffusion

ARMS = {
    "baseline": ("none", False),
    "linear_lift": ("linear", False),
    "pre_lift": ("pre", False),
    "post_lift": ("post", False),
    "bilinear": ("none", True),
    "bilinear_linear_lift": ("linear", True),
    "bilinear_pre_lift": ("pre", True),
    "bilinear_post_lift": ("post", True),
}


def cross_hop_score(values, incidence, weight, coefficients, edge_chunk_size):
    """All unordered hop pairs, all physical edges, all heads, no diagonal terms.

    values: K x N x H x d; weight: E x H. The returned N x H score is
    the weighted local cross-energy divided by weighted node degree.
    Chunking changes only working memory. Checkpointing avoids retaining
    E x K x H x d intermediates during training.
    """
    hops, nodes, heads, width = values.shape
    pairs = torch.triu_indices(hops, hops, offset=1, device=values.device)
    if coefficients.shape != (heads, pairs.shape[1]):
        raise ValueError("cross-hop coefficients must cover every unordered pair")
    matrix = values.new_zeros((heads, hops, hops))
    matrix[:, pairs[0], pairs[1]] = coefficients.to(values.dtype) / 2
    matrix = matrix + matrix.transpose(-1, -2)
    score, degree = weight.new_zeros((nodes, heads)), weight.new_zeros((nodes, heads))
    edges = incidence.shape[1]
    chunk = max(edges, 1) if edge_chunk_size is None else edge_chunk_size
    if chunk <= 0:
        raise ValueError("edge chunk size must be positive")
    # Keep the live edge-feature working set comparable to one-hop diffusion.
    # Every edge is still visited; this never changes the graph or receptive field.
    chunk = max(1, chunk // hops)

    def energy(hidden, metric, endpoints):
        delta = hidden[:, endpoints[1]] - hidden[:, endpoints[0]]
        return torch.einsum("kehd,hkl,lehd->eh", delta, metric, delta) / math.sqrt(width)

    for start in range(0, edges, max(chunk or 1, 1)):
        ends = incidence[:, start : start + chunk]
        w = weight[start : start + chunk]
        local = (
            checkpoint(energy, values, matrix, ends, use_reentrant=False)
            if torch.is_grad_enabled()
            else energy(values, matrix, ends)
        )
        weighted = local.float() * w
        score = score.index_add(0, ends[0], weighted).index_add(0, ends[1], weighted)
        degree.index_add_(0, ends[0], w)
        degree.index_add_(0, ends[1], w)
    return score / degree.clamp_min(torch.finfo(degree.dtype).tiny)


class IncidenceOperator(EdgeSelectionOperator):
    def __init__(self, original, selection_config, *, layer, lift, bilinear):
        super().__init__(original, selection_config, layer=layer)
        self.lift, self.hops = lift, layer + 1
        self.capture = False
        self.last_probe = None
        self.disable_bilinear = False
        self.disable_lift_channel = False
        if lift != "none":
            projection = self.value_weight.new_zeros(
                self.heads, 2 * self.head_width, self.head_width
            )
            projection[:, : self.head_width] = torch.eye(self.head_width)
            self.lift_projection = nn.Parameter(projection)
        else:
            self.register_parameter("lift_projection", None)
        if bilinear and layer:
            self.hop_coefficients = nn.Parameter(
                self.value_weight.new_zeros(self.heads, self.hops * (self.hops - 1) // 2)
            )
        else:
            self.register_parameter("hop_coefficients", None)

    def forward(
        self,
        state,
        incidence,
        node_graph,
        num_graphs,
        *,
        full_degree,
        graph_structure,
        edge_normalization_weight,
        sampling_correction,
        edge_selection_topology,
        static_context=None,
        edge_relation_id=None,
        history=None,
    ):
        with torch.autocast(device_type=state.device.type, enabled=False):
            geometry = state.float()
            context, sample_degree, full_degree = graph_context_features(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                full_degree,
                graph_structure,
                static_context=static_context,
            )
            relation = (
                {"edge_relation_id": edge_relation_id}
                if edge_relation_id is not None or self.num_relations
                else {}
            )
            r = self.estimator(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                graph_context=context,
                sample_degree=sample_degree,
                full_degree=full_degree,
                edge_normalization_weight=edge_normalization_weight,
                **relation,
            )
            gate = self.selector(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                topology=edge_selection_topology,
                sample_degree=sample_degree,
                full_degree=full_degree,
            )
            effective = r * (gate[:, None] if r.ndim == 2 else gate)
            beta = self.beta_estimator(context)
        value = torch.einsum("nd,hdk->nhk", state, self.value_weight)

        def lift(v):
            second = v if self.lift == "linear" else v.square()
            if self.disable_lift_channel:
                second = torch.zeros_like(second)
            return torch.cat((v, second), dim=-1)

        lifted = lift(value) if self.lift in {"linear", "pre"} else value
        propagated = shared_head_diffusion(
            lifted,
            effective,
            incidence,
            node_graph,
            beta,
            sampling_correction=sampling_correction,
            edge_chunk_size=self.edge_chunk_size,
            propagation_normalization=self.propagation_normalization,
            polynomial_coefficients=self.polynomial_delta,
        )
        if self.lift == "post":
            propagated = lift(propagated)
        if self.lift_projection is not None:
            propagated = torch.einsum("nhd,hdk->nhk", propagated, self.lift_projection)
        if self.hop_coefficients is not None:
            if history is None or len(history) != self.hops:
                raise ValueError("bilinear branch requires every preceding layer state")
            history_values = torch.einsum("knd,hdw->knhw", history, self.value_weight)
            with torch.autocast(device_type=state.device.type, enabled=False):
                weight = effective.float()
                if sampling_correction is not None:
                    weight = weight * sampling_correction.reshape(-1, 1)
                cross = cross_hop_score(
                    history_values.float(),
                    incidence,
                    weight,
                    self.hop_coefficients,
                    self.edge_chunk_size,
                )
            if not self.disable_bilinear:
                propagated = propagated + torch.tanh(cross).unsqueeze(-1) * value
        self.last_gate, self.last_r, self.last_effective_c = (
            gate.detach(),
            r.detach(),
            effective.detach(),
        )
        self.last_beta = beta.detach()
        self.last_sampling_correction = (
            None if sampling_correction is None else sampling_correction.detach()
        )
        self.live_edge_graph, self.live_num_graphs = edge_selection_topology.edge_graph, num_graphs
        if self.capture:
            self.last_probe = value.detach()
        return self.output_projection(propagated.reshape(state.shape[0], self.channels))


class IncidenceClassifier(EdgeSelectionClassifier):
    def __init__(self, in_channels, classes, *, arm, selection_config, **architecture):
        if arm not in ARMS:
            raise ValueError(f"unknown ablation arm: {arm}")
        super().__init__(in_channels, classes, selection_config=selection_config, **architecture)
        if self.selection_config["condition"] != "full":
            raise ValueError("incidence ablations use all original physical edges")
        self.arm = arm
        lift, bilinear = ARMS[arm]
        self.capture = False
        self.last_history = None
        for layer, block in enumerate(self.blocks):
            block.operator = IncidenceOperator(
                block.operator,
                self.selection_config,
                layer=layer,
                lift=lift,
                bilinear=bilinear,
            )

    def forward(self, graph):
        self.clear_auxiliary_cache()
        x, incidence = graph.x, graph.incidence_edge_index
        if x.ndim != 2 or x.shape[1] != self.in_channels or not x.is_floating_point():
            raise ValueError("graph.x must match the configured input width")
        if (
            incidence.dtype != torch.long
            or incidence.ndim != 2
            or incidence.shape[0] != 2
            or incidence.device != x.device
        ):
            raise ValueError("physical incidence must be same-device 2 x E int64")
        topology = getattr(graph, "edge_selection_topology", None)
        if topology is None:
            raise ValueError("prepare graph.edge_selection_topology before forward")
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch, graphs = torch.zeros(x.shape[0], dtype=torch.long, device=x.device), 1
        else:
            graphs = getattr(graph, "_v5_num_graphs", None)
            if (
                type(graphs) is not int
                or graphs < 1
                or batch.shape != (x.shape[0],)
                or batch.dtype != torch.long
            ):
                raise ValueError("batched graphs need explicit positive _v5_num_graphs")
        kwargs = {
            name: getattr(graph, name, None)
            for name in (
                "full_degree",
                "graph_structure",
                "edge_normalization_weight",
                "sampling_correction",
                "edge_relation_id",
            )
        }
        kwargs["edge_selection_topology"] = topology
        with torch.autocast(device_type=x.device.type, enabled=False):
            kwargs["static_context"] = _static_graph_context(
                x.float(),
                incidence,
                batch,
                graphs,
                kwargs["full_degree"],
                kwargs["graph_structure"],
            )
        hidden = self.encoder(self.input_norm(x))
        history = [hidden]
        for block in self.blocks:
            block.operator.capture = self.capture

            def step(*past, layer=block):
                current = past[-1]
                clean = layer.operator_norm(current)
                previous = (
                    layer.operator_norm(torch.stack(past))
                    if layer.operator.hop_coefficients is not None
                    else None
                )
                current = current + F.dropout(
                    layer.operator(
                        clean,
                        incidence,
                        batch,
                        graphs,
                        history=previous,
                        **kwargs,
                    ),
                    layer.dropout,
                    layer.training,
                )
                return current + F.dropout(
                    layer.ffn(layer.ffn_norm(current)), layer.dropout, layer.training
                )

            if self.activation_checkpoint and torch.is_grad_enabled():
                hidden = checkpoint(step, *history, use_reentrant=False, preserve_rng_state=True)
            else:
                hidden = step(*history)
            history.append(hidden)
        if self.capture:
            self.last_history = torch.stack([value.detach() for value in history])
        return self.decoder(self.final_norm(hidden))
