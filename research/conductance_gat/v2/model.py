"""Direct per-edge log-conductance, bound to one ordered transductive topology.

Each layer owns alpha_e initialized to zero and uses C=diag(exp(alpha)). No MLP,
hidden-state-to-C function, dense C, or eigendecomposition is used. A common
positive C scale cancels under row-degree normalization; no centering, cap or
projection is silently applied. The runner gives log C zero weight decay.

The fixed arm is parameter-free and constructs C=1 from the actual forward
state. Only the direct arm owns graph-bound log-C parameters; those parameters
cannot transfer to an unseen graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .operator import chunked_normalized_propagation


def _canonical_topology(incidence: Tensor, num_nodes: int) -> Tensor:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("incidence must be a 2 x E int64 tensor")
    value = incidence.detach().clone().contiguous()
    if value.numel():
        if bool((value < 0).any()) or bool((value >= num_nodes).any()):
            raise ValueError("incidence endpoint is outside the bound graph")
        if not bool((value[0] < value[1]).all()):
            raise ValueError("Bound incidence must use canonical tail < head edges")
        keys = value[0] * num_nodes + value[1]
        if keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("Bound incidence must contain sorted, unique canonical edges")
    return value


class DirectEdgeConductance(nn.Module):
    """One positive scalar per fixed physical edge, shared across feature channels."""

    def __init__(self, num_edges: int, gate_mode: str = "direct") -> None:
        super().__init__()
        if gate_mode not in {"direct", "fixed_one"}:
            raise ValueError(f"Unsupported direct-C gate mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.num_edges = num_edges
        if gate_mode == "direct":
            self.log_c = nn.Parameter(torch.zeros(num_edges))

    def forward(self, reference: Tensor | None = None) -> Tensor:
        if self.gate_mode == "fixed_one":
            if reference is None:
                return torch.ones(self.num_edges)
            return reference.new_ones(self.num_edges)
        c = self.log_c.exp()
        if not bool(torch.isfinite(c.detach()).all()) or not bool((c.detach() > 0).all()):
            raise FloatingPointError("exp(log C) overflow/underflow; no clipping is used")
        return c


class DirectCConv(nn.Module):
    def __init__(
        self, num_edges: int, *, gate_mode: str = "direct", edge_chunk_size: int = 65536
    ) -> None:
        super().__init__()
        if isinstance(edge_chunk_size, bool) or not isinstance(edge_chunk_size, int):
            raise ValueError("edge_chunk_size must be a positive integer")
        if edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be a positive integer")
        self.estimator = DirectEdgeConductance(num_edges, gate_mode)
        self.normalization = "node_degree"
        self.edge_chunk_size = edge_chunk_size

    def forward(
        self,
        x: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int | None = None,
    ) -> Tensor:
        # Keep the ordinary operator hook API for read-only C/propagation reports.
        with torch.autocast(device_type=x.device.type, enabled=False):
            state = x if x.dtype == torch.float64 else x.float()
            c = self.estimator(state).to(dtype=state.dtype)
            result = chunked_normalized_propagation(
                state, c, incidence, edge_chunk_size=self.edge_chunk_size
            )
        return result.to(x.dtype)


class DirectCNodeClassifier(nn.Module):
    """The same node encoder/norm/decoder scaffold with graph-bound direct C."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        *,
        incidence: Tensor,
        num_nodes: int,
        gate_mode: str = "direct",
        normalization: str = "node_degree",
        hidden_channels: int = 64,
        layers: int = 2,
        dropout: float = 0.5,
        edge_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        if normalization != "node_degree":
            raise ValueError("Direct-C v2 keeps node_degree normalization fixed")
        if hidden_channels < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden width/layers must be positive and dropout in [0, 1)")
        if gate_mode not in {"direct", "fixed_one"}:
            raise ValueError(f"Unsupported direct-C gate mode: {gate_mode}")
        topology = _canonical_topology(incidence, num_nodes)
        self.register_buffer("bound_incidence", topology)
        self.register_buffer("bound_num_nodes", torch.tensor(num_nodes, dtype=torch.long))
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.normalization = normalization
        self.gate_mode = gate_mode
        self.edge_chunk_size = edge_chunk_size
        # Identical allocation order in both arms. No gate MLP consumes RNG here.
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)
        self.operators = nn.ModuleList(
            DirectCConv(topology.shape[1], gate_mode=gate_mode, edge_chunk_size=edge_chunk_size)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(layers))

    def validate_topology(self, graph: Any) -> None:
        if graph.x.ndim != 2 or graph.x.shape[0] != self.num_nodes:
            raise ValueError("Direct C is bound to a different node count")
        incidence = graph.incidence_edge_index
        if incidence.device != self.bound_incidence.device:
            raise ValueError("Graph and bound direct-C topology must share a device")
        if incidence.dtype != torch.long or not torch.equal(incidence, self.bound_incidence):
            raise ValueError("Direct C requires exactly the bound edge identity and order")
        batch = getattr(graph, "batch", None)
        if batch is not None and (
            batch.shape != (self.num_nodes,) or bool((batch != 0).any())
        ):
            raise ValueError("Direct C v2 accepts one bound transductive graph, not graph batches")

    def _validate_checkpoint_topology(self, state_dict: Mapping[str, Tensor], prefix: str = ""):
        for name in ("bound_incidence", "bound_num_nodes"):
            key = prefix + name
            saved = state_dict.get(key)
            current = getattr(self, name)
            if (
                not isinstance(saved, Tensor)
                or saved.dtype != current.dtype
                or saved.shape != current.shape
                or not torch.equal(saved.to(device=current.device), current)
            ):
                raise RuntimeError("Checkpoint direct-C topology identity/order does not match")

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True, assign=False):
        # Fail before *any* parameters/buffers are replaced, even with strict=False.
        self._validate_checkpoint_topology(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Also protect this module when it is nested in a parent model.
        self._validate_checkpoint_topology(state_dict, prefix)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, graph: Any) -> Tensor:
        self.validate_topology(graph)
        h = F.dropout(F.elu(self.encoder(graph.x)), self.dropout, self.training)
        node_graph = torch.zeros(self.num_nodes, dtype=torch.long, device=h.device)
        for operator, norm in zip(self.operators, self.norms, strict=True):
            h = operator(h, graph.incidence_edge_index, node_graph, 1)
            h = F.dropout(F.elu(norm(h)), self.dropout, self.training)
        return self.decoder(h)
