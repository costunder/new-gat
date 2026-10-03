"""Packed classifiers using current local quadratic and signed bilinear features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import torch
from torch import nn

from ...wedge_propagation.classification.model import (
    PackedClassifier as _DropoutReference,
)
from ...wedge_propagation.classification.model import _integer, _matrix, _named_seed
from .operators import WEIGHTS, local_features, two_hop

VARIANTS = ("base", "within", "between", "both")
CONDITIONS = tuple(f"{mode}__{variant}" for mode in WEIGHTS for variant in VARIANTS)
INTERVENTIONS = ("within_remove", "between_remove", "both_remove")


def _norm(value):
    return value.detach().square().flatten(1).sum(1).sqrt()


class PackedClassifier(nn.Module):
    """Two identical propagation layers; independent leading seed parameters.

    Extra E/J lifts start at zero, so the four variants share their exact
    initial forward. The first classification loss updates active lifts; all
    later features are recomputed with the current trainable projection.
    """

    def __init__(
        self,
        condition: str,
        input_dim: int,
        classes: int,
        seeds: Sequence[int],
        hidden=64,
        dropout=0.5,
        path_chunk=None,
        checkpoint_paths=True,
        dataset_name="unspecified",
    ):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown local prediction condition {condition!r}")
        self.condition = condition
        self.weight_mode, self.variant = condition.split("__")
        self.input_dim = _integer(input_dim, "input_dim")
        self.classes = _integer(classes, "classes", minimum=2)
        self.hidden = _integer(hidden, "hidden")
        self.seeds = tuple(seeds)
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in self.seeds)
        ):
            raise ValueError("independent seeds must be distinct nonnegative integers")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("dataset name must be nonempty")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(checkpoint_paths, bool):
            raise TypeError("checkpoint_paths must be boolean")
        self.dataset_name, self.dropout = dataset_name, float(dropout)
        self.path_chunk = None if path_chunk is None else _integer(path_chunk, "path_chunk")
        self.checkpoint_paths = checkpoint_paths
        self.has_energy = self.variant in ("within", "both")
        self.has_relation = self.variant in ("between", "both")
        self.projections = nn.ParameterList(
            (
                _matrix(self.seeds, dataset_name, "projection.0", self.input_dim, self.hidden),
                _matrix(self.seeds, dataset_name, "projection.1", self.hidden, self.classes),
            )
        )
        self.alphas = nn.ParameterList(nn.Parameter(torch.zeros(len(self.seeds))) for _ in range(2))
        self.energy_lifts = nn.ParameterList(
            nn.Parameter(torch.zeros(len(self.seeds), channels))
            for channels in ((self.hidden, self.classes) if self.has_energy else ())
        )
        self.relation_lifts = nn.ParameterList(
            nn.Parameter(torch.zeros(len(self.seeds), channels))
            for channels in ((self.hidden, self.classes) if self.has_relation else ())
        )
        keys = [
            [_named_seed(dataset_name, s, f"dropout.{layer}") & 0xFFFFFFFF for s in self.seeds]
            for layer in range(2)
        ]
        self.register_buffer(
            "dropout_keys", torch.tensor(keys, dtype=torch.int64), persistent=False
        )

    _dropout = _DropoutReference._dropout

    @property
    def parameters_per_seed(self):
        return sum(p.numel() for p in self.parameters()) // len(self.seeds)

    @property
    def trainable_parameters_per_seed(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) // len(self.seeds)

    def parameter_count_per_seed(self):
        return self.parameters_per_seed

    def weight_decay_groups(self, weight_decay):
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        other = [p for name, p in self.named_parameters() if not name.startswith("projections.")]
        return [
            {"params": list(self.projections), "weight_decay": weight_decay},
            {"params": other, "weight_decay": 0.0},
        ]

    def forward(
        self, graph, *, epoch=0, diagnostics=False, intervention=None, intervention_layers=None
    ):
        if intervention is not None and intervention not in INTERVENTIONS:
            raise ValueError("unknown local prediction intervention")
        layers = (0, 1) if intervention_layers is None else tuple(intervention_layers)
        if not layers or any(type(layer) is not int or layer not in (0, 1) for layer in layers):
            raise ValueError("intervention layers must select layer 0 and/or 1")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if graph.x.shape[1] != self.input_dim:
            raise ValueError("graph input features disagree with the classifier")
        h = graph.x[None, :, :].expand(len(self.seeds), -1, -1)
        details = []
        for layer in range(2):
            z = torch.bmm(self._dropout(h, epoch, layer), self.projections[layer])
            alpha = self.alphas[layer].sigmoid()
            base = (1 - alpha[:, None, None]) * z + alpha[:, None, None] * two_hop(
                graph.geometry, self.weight_mode, z
            )
            if self.has_energy or self.has_relation:
                e, j, raw_e, raw_j = local_features(
                    graph.geometry,
                    self.weight_mode,
                    z,
                    energy=self.has_energy,
                    relation=self.has_relation,
                    chunk=self.path_chunk,
                    checkpoint_chunks=self.checkpoint_paths,
                )
            else:
                e = j = z.new_zeros(z.shape[:2])
                raw_e, raw_j = e, z.new_zeros((len(self.seeds), graph.topology.num_pairs))
            energy_branch = (
                e[:, :, None] * self.energy_lifts[layer][:, None, :]
                if self.has_energy
                else torch.zeros_like(z)
            )
            relation_branch = (
                j[:, :, None] * self.relation_lifts[layer][:, None, :]
                if self.has_relation
                else torch.zeros_like(z)
            )
            if layer in layers:
                if intervention in ("within_remove", "both_remove"):
                    energy_branch = torch.zeros_like(energy_branch)
                if intervention in ("between_remove", "both_remove"):
                    relation_branch = torch.zeros_like(relation_branch)
            before = base + energy_branch + relation_branch
            base_norm, en, jn = _norm(base), _norm(energy_branch), _norm(relation_branch)
            nonzero = base_norm > 0
            safe_base = torch.where(nonzero, base_norm, 1)
            zero = z.new_zeros(len(self.seeds))
            info = {
                "alpha": alpha.detach(),
                "projected_norm": _norm(z),
                "base_norm": base_norm,
                "base_nonzero": nonzero,
                "energy_feature_norm": _norm(e),
                "relation_feature_norm": _norm(j),
                "energy_branch_norm": en,
                "relation_branch_norm": jn,
                "energy_to_base": en / safe_base,
                "relation_to_base": jn / safe_base,
                "energy_lift_norm": _norm(self.energy_lifts[layer]) if self.has_energy else zero,
                "relation_lift_norm": _norm(self.relation_lifts[layer])
                if self.has_relation
                else zero,
                "energy_raw_mean": raw_e.detach().mean(1),
                "relation_raw_mean": raw_j.detach().mean(1) if raw_j.shape[1] else zero,
                "relation_negative_fraction": (raw_j.detach() < 0).to(z.dtype).mean(1)
                if raw_j.shape[1]
                else zero,
            }
            if diagnostics:
                info.update(
                    energy_feature=e.detach(),
                    relation_feature=j.detach(),
                    energy_raw=raw_e.detach(),
                    relation_raw=raw_j.detach(),
                )
            details.append(info)
            h = before.relu() if layer == 0 else before
        return h, details
