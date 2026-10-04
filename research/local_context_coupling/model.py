"""Two-layer packed classifier with an active, shared cross-coupling gain.

This module connects the candidate to CE/backward/optimizer checks. It does
not claim a completed full classification study; the first experiment is the
full fixed audit plus clearly marked DEBUG model-connection checks.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor, nn

from ..wedge_propagation.classification.model import (
    PackedClassifier as _DropoutReference, _integer, _matrix, _named_seed,
)
from .operators import CouplingGeometry, sandwich


class PackedClassifier(nn.Module):
    """Independent seed models, each with one gain shared by both GNN layers."""

    def __init__(
        self, input_dim: int, classes: int, seeds: tuple[int, ...] = (0,), *,
        condition: str = "cross_on", mode: str = "unit", hidden: int = 64,
        dropout: float = 0.5, dataset_name: str = "unspecified",
    ) -> None:
        super().__init__()
        self.input_dim = _integer(input_dim, "input_dim")
        self.classes = _integer(classes, "classes", minimum=2)
        self.hidden = _integer(hidden, "hidden")
        if not isinstance(seeds, (tuple, list)) or not seeds:
            raise ValueError("seeds must be a nonempty sequence of integers")
        if any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds):
            raise ValueError("each seed must be a nonnegative integer")
        self.seeds = tuple(int(seed) for seed in seeds)
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if condition not in ("cross_on", "cross_off"):
            raise ValueError("condition must be cross_on or cross_off")
        if mode not in ("unit", "local_degree"):
            raise ValueError("mode must be unit or local_degree")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.condition, self.mode, self.dropout = condition, mode, float(dropout)
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name must be a nonempty string")
        self.dataset_name = dataset_name
        self.num_layers = 2
        self.weights = nn.ParameterList([
            _matrix(self.seeds, self.dataset_name, "local-context.layer0", self.input_dim, self.hidden),
            _matrix(self.seeds, self.dataset_name, "local-context.layer1", self.hidden, self.classes),
        ])
        if self.condition == "cross_on":
            self.theta_cross = nn.Parameter(torch.zeros(len(self.seeds)))
        else:
            self.register_parameter("theta_cross", None)
        keys = [
            [_named_seed(self.dataset_name, seed, f"local-context.dropout.{layer}") & 0xFFFFFFFF for seed in self.seeds]
            for layer in range(self.num_layers)
        ]
        self.register_buffer("dropout_keys", torch.tensor(keys, dtype=torch.int64), persistent=False)

    _dropout = _DropoutReference._dropout

    @property
    def cross_gain(self) -> Tensor:
        if self.theta_cross is None:
            return self.weights[0].new_zeros(len(self.seeds))
        return self.theta_cross.sigmoid()

    @property
    def parameters_per_seed(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()) // len(self.seeds)

    @property
    def trainable_parameters_per_seed(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad) // len(self.seeds)

    def parameter_count_per_seed(self) -> int:
        return self.parameters_per_seed

    def weight_decay_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        if isinstance(weight_decay, bool) or not isinstance(weight_decay, Real) or not 0 <= weight_decay < float("inf"):
            raise ValueError("weight_decay must be finite and nonnegative")
        groups: list[dict[str, Any]] = [{"params": list(self.weights), "weight_decay": float(weight_decay)}]
        if self.theta_cross is not None:
            groups.append({"params": [self.theta_cross], "weight_decay": 0.0})
        return groups

    def forward(
        self, graph: Any, *, epoch: int = 0, diagnostics: bool = False,
    ) -> tuple[Tensor, list[dict[str, Any]]]:
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if not isinstance(diagnostics, bool):
            raise ValueError("diagnostics must be boolean")
        if isinstance(graph, tuple) and len(graph) == 2:
            x, geometry = graph
        else:
            x, geometry = graph.x, graph.geometry
        if not isinstance(geometry, CouplingGeometry) or geometry.mode != self.mode:
            raise ValueError("graph geometry must be CouplingGeometry with the model's weight mode")
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape != (geometry.n, self.input_dim):
            raise ValueError("x must have shape [physical_nodes,input_dim]")
        if x.device != self.weights[0].device or x.dtype != self.weights[0].dtype:
            raise ValueError("x and model parameters must have identical device and dtype")
        h = x.unsqueeze(0).expand(len(self.seeds), -1, -1)
        details: list[dict[str, Any]] = []
        gain = self.cross_gain
        for layer, weight in enumerate(self.weights):
            h = self._dropout(h, int(epoch), layer)
            z = torch.bmm(h, weight)
            result = sandwich(
                geometry, z, cross=self.condition == "cross_on", cross_gain=gain,
                diagnostics=diagnostics,
            )
            if diagnostics:
                h, stage = result
                details.append(stage)
            else:
                h = result
            if layer == 0:
                h = torch.relu(h)
        return h, details
