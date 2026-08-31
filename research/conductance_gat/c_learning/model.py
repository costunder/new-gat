"""A single change: learn edge C, or use exactly one on every incidence edge.

Both arms use H - .95 D_C^dagger B.T C B H and the same classifier. The fixed
arm retains identical initialized gate tensors solely for state/RNG matching;
they are frozen, not evaluated and excluded from the optimizer. Consequently
C=1 yields .05 H_i + .95 mean_neighbors(H_j) on nonisolated nodes, and identity
on isolates. This is an internal ablation, not an external GCN/GAT baseline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..ablation.model import FactorialNodeClassifier, is_gate_parameter
from ..sparse import SparsePositiveConductance
from .protocol import COMMON, CONDITIONS


class FixedOneConductance(nn.Module):
    """Expose effective C to the usual hook without evaluating the frozen gate."""

    def __init__(self, initialized_estimator: SparsePositiveConductance) -> None:
        super().__init__()
        # Transfer the already initialized network without any RNG draws and
        # retain the exact estimator.network.* state-dictionary key structure.
        self.network = initialized_estimator.network
        self.network.requires_grad_(False)

    def forward(self, gradient: Tensor, edge_features: Tensor) -> Tensor:
        return gradient.new_ones(gradient.shape[0])


class CLearningNodeClassifier(FactorialNodeClassifier):
    def __init__(
        self,
        in_channels: int,
        classes: int,
        *,
        gate_mode: str = "learned",
        normalization: str = "node_degree",
        hidden_channels: int = 64,
        layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        if normalization != "node_degree":
            raise ValueError("C-learning keeps node_degree normalization fixed")
        if gate_mode not in {"learned", "fixed_one"}:
            raise ValueError(f"Unsupported gate mode: {gate_mode}")
        super().__init__(
            in_channels,
            classes,
            normalization=normalization,
            hidden_channels=hidden_channels,
            layers=layers,
            dropout=dropout,
        )
        self.gate_mode = gate_mode
        if gate_mode == "fixed_one":
            for operator in self.operators:
                operator.estimator = FixedOneConductance(operator.estimator)


def make_optimizer(model: CLearningNodeClassifier, condition: str) -> torch.optim.Adam:
    specification = CONDITIONS[condition]
    if model.gate_mode != specification["gate_mode"] or model.normalization != "node_degree":
        raise ValueError("Model and C-learning condition disagree")
    gate, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (gate if is_gate_parameter(name) else other).append(parameter)
    if not other or bool(gate) != (model.gate_mode == "learned"):
        raise ValueError("C-learning trainable parameter groups do not match the gate mode")
    groups = [{"params": other, "weight_decay": COMMON["weight_decay"], "name": "non_gate"}]
    if gate:
        groups.append(
            {"params": gate, "weight_decay": specification["gate_weight_decay"], "name": "gate"}
        )
    return torch.optim.Adam(groups, lr=COMMON["lr"])
