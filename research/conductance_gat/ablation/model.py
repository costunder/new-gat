"""One-factor operator changes without changing the benchmark architecture.

``global_max`` delegates literally to the published benchmark implementation.
``node_degree`` is H - .95 D_C^dagger B.T C B H (zero inverse on isolated nodes).
It preserves constants/orientation invariance, but is generally not symmetric in
the Euclidean inner product. Both choices cancel a common conductance scale;
neither makes the absolute scale of C identifiable. No denominator is detached.
"""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn

from ..benchmark import ConductanceConv, ConductanceNodeClassifier
from ..benchmark_data import tensor_hash
from .protocol import COMMON, CONDITIONS


class FactorialConductanceConv(ConductanceConv):
    def __init__(self, channels: int, normalization: str = "global_max") -> None:
        if normalization not in {"global_max", "node_degree"}:
            raise ValueError(f"Unsupported normalization: {normalization}")
        super().__init__(channels)
        self.normalization = normalization

    def forward(
        self,
        x: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int | None = None,
    ) -> Tensor:
        if self.normalization == "global_max":
            return super().forward(x, incidence, node_graph, num_graphs)
        with torch.autocast(device_type=x.device.type, enabled=False):
            state = x.float()
            tail, head = incidence
            gradient = state[head] - state[tail]
            c = self.estimator(gradient, state.new_empty((gradient.shape[0], 0)))
            flux = c[:, None] * gradient
            divergence = torch.zeros_like(state)
            divergence.index_add_(0, head, flux)
            divergence.index_add_(0, tail, -flux)
            degree = state.new_zeros(state.shape[0])
            degree.index_add_(0, head, c)
            degree.index_add_(0, tail, c)
            # For isolated nodes division is harmless and divergence is exactly 0.
            # For nonisolated nodes C >= 1e-5, so this branch is the exact D^dagger.
            safe_degree = torch.where(degree > 0, degree, torch.ones_like(degree))
            result = state - 0.95 * divergence / safe_degree[:, None]
        return result.to(x.dtype)


class FactorialNodeClassifier(ConductanceNodeClassifier):
    """Identical parameter names and initialization order to the baseline."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        *,
        normalization: str = "global_max",
        hidden_channels: int = 64,
        layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        # Replacing initialized parent layers would consume RNG twice. Mirror
        # precisely its four module allocations; reuse its forward unchanged.
        nn.Module.__init__(self)
        if hidden_channels < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden width/layers must be positive and dropout in [0, 1)")
        self.dropout = dropout
        self.normalization = normalization
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)
        self.operators = nn.ModuleList(
            FactorialConductanceConv(hidden_channels, normalization) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(layers))


def is_gate_parameter(name: str) -> bool:
    return name.startswith("operators.") and ".estimator." in name


def make_optimizer(model: nn.Module, condition: str) -> torch.optim.Adam:
    """Coupled Adam L2 as in baseline; only estimator parameters change decay."""
    spec = CONDITIONS[condition]
    gate, other = [], []
    for name, parameter in model.named_parameters():
        (gate if is_gate_parameter(name) else other).append(parameter)
    if not gate or not other:
        raise ValueError("Expected nonempty, disjoint gate and non-gate parameter groups")
    return torch.optim.Adam(
        [
            {"params": other, "weight_decay": COMMON["weight_decay"], "name": "non_gate"},
            {"params": gate, "weight_decay": spec["gate_weight_decay"], "name": "gate"},
        ],
        lr=COMMON["lr"],
    )


def state_sha256(model: nn.Module) -> str:
    """Order/name/dtype/shape/value-sensitive fingerprint, independent of torch.save."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_hash(tensor).encode("ascii"))
    return digest.hexdigest()
