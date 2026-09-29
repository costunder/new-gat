"""Same optimized C and basic diffusion; no information-completion branches."""

import copy
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.aggregation_comparison import engine
from research.conductance_gat.v5.model import _static_graph_context

from . import SUITE


class OnesConductance(nn.Module):
    """Parameter-free control. No dormant cost generator or inner solver."""

    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        self.last_scores = None

    def forward(self, state, incidence, node_graph, num_graphs, **kwargs):
        result = state.new_ones((incidence.shape[1], self.heads))
        self.last_scores = torch.zeros_like(result)
        return result


class COnlyClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone.arm != "incidence" or len(backbone.energy_readouts):
            raise ValueError("v2.0 accepts only the plain learned-C backbone")
        if any(op.lift != "none" or op.hop_coefficients is not None for op in backbone.layers):
            raise ValueError("v2.0 excludes lift and cross-hop branches")
        self.encoder, self.decoder, self.layers = (
            backbone.encoder,
            backbone.decoder,
            backbone.layers,
        )
        self.width, self.heads, self.depth = backbone.width, backbone.heads, backbone.depth
        self.dropout = backbone.dropout
        self.activation_checkpoint = backbone.activation_checkpoint
        self.condition = "learned"

    def fixed_copy(self):
        result = copy.deepcopy(self)
        for layer in result.layers:
            layer.estimator = OnesConductance(self.heads)
        result.condition = "fixed"
        return result

    def contract(self):
        return {
            "suite": SUITE,
            "condition": self.condition,
            "layers": self.depth,
            "hidden_channels": self.width,
            "heads": self.heads,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "same_nodes_and_edges_at_every_layer": True,
            "generator": "existing graph-context optimized C"
            if self.condition == "learned"
            else "ones",
            "solver_steps": [getattr(op.estimator, "solver_steps", 0) for op in self.layers],
            "completion_branches": [],
            "cross_layer_flow_history": False,
            "loss": "classification cross entropy only",
            "external_residual": False,
            "initialization": "unchanged legacy",
            "dropout": self.dropout,
            "activation_checkpoint": self.activation_checkpoint,
        }

    @contextmanager
    def c_ones(self):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("C intervention requires eval/no_grad")
        estimators = [op.estimator for op in self.layers]
        try:
            for op in self.layers:
                op.estimator = OnesConductance(self.heads)
            yield
        finally:
            for op, estimator in zip(self.layers, estimators, strict=True):
                op.estimator = estimator

    def forward(self, graph, observer=None):
        edges = graph.incidence_edge_index
        groups = getattr(graph, "batch", None)
        if groups is None:
            groups = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
            count = 1
        else:
            count = graph._v5_num_graphs
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
        kwargs["edge_selection_topology"] = graph.edge_selection_topology
        with torch.autocast(device_type=graph.x.device.type, enabled=False):
            kwargs["static_context"] = _static_graph_context(
                graph.x.float(),
                edges,
                groups,
                count,
                kwargs["full_degree"],
                kwargs["graph_structure"],
            )
        state = self.encoder(graph.x)
        for index, op in enumerate(self.layers):

            def step(value, op=op):
                result = op.forward_with_state(value, edges, groups, count, **kwargs)
                return result.message, result.conductance

            message, conductance = (
                checkpoint(step, state, use_reentrant=False)
                if self.activation_checkpoint and torch.is_grad_enabled()
                else step(state)
            )
            if observer is not None:
                observer(index, op, state, conductance, message, graph, groups, count, kwargs)
            state = F.dropout(F.relu(message), self.dropout, self.training)
        return self.decoder(state)


def make_model(payload, args, device, condition="learned"):
    if condition not in {"learned", "fixed"} or args.ablation_arm != "incidence":
        raise ValueError("v2.0 has exactly learned and fixed C conditions")
    engine.base._seed(args.model_seed)
    model = COnlyClassifier(engine.make_model(payload, args, device))
    return model if condition == "learned" else model.fixed_copy()
