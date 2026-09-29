"""Existing value/beta/row diffusion with a newly constructed local C module."""

from dataclasses import dataclass

import torch
from torch import nn

from research.conductance_gat.v5.model import GraphConditionedBeta, graph_context_features

from .conductance import BracketConductance
from .diffusion import row_diffusion


@dataclass(frozen=True)
class OperatorOutput:
    message: torch.Tensor
    conductance: torch.Tensor
    effective_weight: torch.Tensor
    beta: torch.Tensor


class BracketOperator(nn.Module):
    def __init__(self, channels, heads, *, edge_chunk_size, checkpoint_edges=True):
        super().__init__()
        self.channels, self.heads, self.head_width = channels, heads, channels // heads
        self.edge_chunk_size = edge_chunk_size
        self.capture_coefficients = False
        self.observed_coefficients = None
        self.capture_signals = False
        self.observed_signals = None
        # Isolate generator RNG from common value/output/beta initialization.
        with torch.random.fork_rng(devices=[]):
            self.estimator = BracketConductance(
                channels,
                heads,
                edge_chunk_size=edge_chunk_size,
                checkpoint_edges=checkpoint_edges,
            )
        self.value_weight = nn.Parameter(torch.empty(heads, channels, self.head_width))
        nn.init.xavier_uniform_(self.value_weight.reshape(channels, channels))
        self.output_projection = nn.Linear(channels, channels, bias=False)
        self.beta_estimator = GraphConditionedBeta(
            channels,
            heads,
            beta_parameterization="sigmoid",
            beta_initial=0.5,
        )

    def forward_with_state(
        self,
        state,
        incidence,
        node_graph,
        num_graphs,
        *,
        full_degree,
        graph_structure,
        sampling_correction,
        static_context=None,
        edge_relation_id=None,
        **unused,
    ):
        if edge_relation_id is not None:
            raise ValueError("this experiment does not consume relation labels")
        with torch.autocast(device_type=state.device.type, enabled=False):
            geometry = state.to(self.value_weight.dtype)
            c = self.estimator(geometry, incidence, node_graph, num_graphs)
            # Context is used only for the unchanged beta path, never for C.
            context, _, _ = graph_context_features(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                full_degree,
                graph_structure,
                static_context=static_context,
            )
            beta = self.beta_estimator(context)
        value = torch.einsum("nd,hdk->nhk", state, self.value_weight)
        effective = c if sampling_correction is None else c * sampling_correction[:, None]
        torch._assert_async(
            (torch.isfinite(effective) & (effective > 0)).all(),
            "sampling-corrected C must be finite and positive",
        )
        propagated, coefficients = row_diffusion(
            value,
            effective,
            incidence,
            node_graph,
            beta,
            edge_chunk_size=self.edge_chunk_size,
        )
        if self.capture_coefficients:
            self.observed_coefficients = tuple(value.detach() for value in coefficients)
        message = self.output_projection(propagated.reshape(state.shape[0], self.channels))
        if self.capture_signals:
            self.observed_signals = {
                "value_projection": value.detach(),
                "neighbor_mixing": propagated.detach(),
            }
        torch._assert_async(torch.isfinite(message).all(), "nonfinite bracket propagation")
        return OperatorOutput(message, c, effective, beta)
