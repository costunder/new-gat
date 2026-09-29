"""Direct bracket construction; no optimized-C model is ever instantiated."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.aggregation_comparison import engine
from experiments.c_learning_only.model import COnlyClassifier
from research.conductance_gat.v5.model import _static_graph_context

from . import SUITE
from .operator import BracketOperator


class BracketClassifier(COnlyClassifier):
    def __init__(
        self,
        inputs,
        classes,
        *,
        channels,
        layers,
        heads,
        dropout,
        activation_checkpoint,
        edge_chunk_size,
        checkpoint_edges=True,
    ):
        nn.Module.__init__(self)
        self.encoder, self.decoder = nn.Linear(inputs, channels), nn.Linear(channels, classes)
        self.layers = nn.ModuleList(
            [
                BracketOperator(
                    channels,
                    heads,
                    edge_chunk_size=edge_chunk_size,
                    checkpoint_edges=checkpoint_edges,
                )
                for _ in range(layers)
            ]
        )
        self.width, self.heads, self.depth = channels, heads, layers
        self.dropout, self.activation_checkpoint = dropout, activation_checkpoint
        self.condition = "learned"
        self.checkpoint_edges = checkpoint_edges

    def forward(self, graph, observer=None):
        # Same fixed-support forward, with no unused topology argument.
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
                "sampling_correction",
                "edge_relation_id",
            )
        }
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
        capture_signals = callable(getattr(observer, "record_activation", None))
        for index, op in enumerate(self.layers):
            op.capture_coefficients = observer is not None
            op.capture_signals = capture_signals

            def layer_step(value, op=op):
                result = op.forward_with_state(value, edges, groups, count, **kwargs)
                return result.message, result.conductance

            message, c = (
                checkpoint(layer_step, state, use_reentrant=False)
                if self.activation_checkpoint and torch.is_grad_enabled()
                else layer_step(state)
            )
            op.capture_coefficients = False
            op.capture_signals = False
            if observer is not None:
                observer(index, op, state, c, message, graph, groups, count, kwargs)
            op.observed_coefficients = None
            op.observed_signals = None
            activated = F.relu(message)
            state = F.dropout(activated, self.dropout, self.training)
            if capture_signals:
                observer.record_activation(index, activated, state, op.heads)
        return self.decoder(state)

    def contract(self):
        result = super().contract()
        result.update(
            {
                "suite": SUITE,
                "implementation_revision": "signal_audit_2",
                "generator": "symmetric_dot_exp" if self.condition == "learned" else "ones",
                "score": "0.5 * mean(Q_u*K_v + Q_v*K_u, head_features)",
                "score_scale": "1/head_width",
                "positive_map": "exp; no clamp or centering",
                "query_key_initialization": "independent Xavier uniform, zero bias",
                "generator_uses_graph_context": False,
                "beta_uses_graph_context": True,
                "checkpoint_edges": self.checkpoint_edges,
                "common_initializers": "unchanged algorithms; new direct construction RNG sequence",
            }
        )
        return result


def make_model(payload, args, device, condition="learned"):
    if condition not in {"learned", "fixed"}:
        raise ValueError("only learned and fixed C conditions are supported")
    engine.base._seed(args.model_seed)
    model = BracketClassifier(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        channels=args.hidden_channels,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        activation_checkpoint=args.activation_checkpoint,
        edge_chunk_size=args.edge_chunk_size,
        checkpoint_edges=args.checkpoint_edges,
    ).to(device)
    return model if condition == "learned" else model.fixed_copy()
