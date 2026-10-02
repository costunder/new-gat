"""Normalize gate inputs while retaining the original operator message.

For graph g and independent scalar realization r, sigma is the RMS of every
physical-edge difference. Only g1/sigma and g2/sigma are given to the original
gate; the weighted transpose operator acts on the original x. No nonzero
scale has an additive epsilon. Empty-edge or exactly zero-edge-signal fields
use sigma=1, and the sqrt is evaluated on that positive replacement so zero
fields also have finite input gradients.

Static edge-to-graph indices/counts are cached by tensor identity and mutation
version. RMS is additionally cached only for x without input gradients, as in
the static synthetic training cache. Inputs requiring gradients always compute
a fresh differentiable RMS. Inference tensors have no mutation version and
are never reused. Do not mutate cached tensors through .data. First-use CPU
topology validation follows the existing operator APIs; steady forwards do not
copy tensors to CPU or synchronize the GPU.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..learned.model import (
    SeedBatchedModel,
    _index_info,
    _positive_integer,
    _real_matrix,
    apply_weighted_pair,
    apply_weighted_wedge,
    pair_gradients,
)
from ..operators import incidence_apply

__all__ = ["NormalizedSeedBatchedModel", "make_normalized_model", "physical_edge_rms"]


@dataclass
class _EdgeGroups:
    edges: weakref.ReferenceType[Tensor]
    node_graph: weakref.ReferenceType[Tensor]
    versions: tuple[int | None, int | None]
    edge_graph: Tensor
    counts: Tensor


@dataclass
class _InputScale:
    x: weakref.ReferenceType[Tensor]
    version: int
    groups: _EdgeGroups
    sigma: Tensor


_GROUP_CACHE: dict[tuple[int, int, int, int], _EdgeGroups] = {}
_SCALE_CACHE: dict[tuple[int, int, int, int], _InputScale] = {}


def _version(tensor: Tensor) -> int | None:
    return None if torch.is_inference(tensor) else tensor._version


def _edge_groups(batch: Any) -> _EdgeGroups:
    edges, node_graph = batch.edges, batch.node_graph
    n, graphs = batch.x.shape[0], _positive_integer(batch.num_graphs, "num_graphs")
    if node_graph.device != batch.x.device or edges.device != batch.x.device:
        raise ValueError("edges, node_graph and batch.x must have the same device")
    info = _index_info(node_graph, graphs, n, "node_graph", groups=True)
    if not info.all_groups_present:
        raise ValueError("node_graph must contain nodes for every graph")
    versions = (_version(edges), _version(node_graph))
    key = (id(edges), id(node_graph), n, graphs)
    cached = _GROUP_CACHE.get(key)
    if (None not in versions and cached is not None and cached.edges() is edges
            and cached.node_graph() is node_graph and cached.versions == versions):
        return cached
    # Validation happens once for static batches, preventing a scale assignment
    # to the wrong graph when a malformed edge crosses two disjoint components.
    physical, node_groups = edges.detach().cpu(), node_graph.detach().cpu()
    groups_cpu = node_groups.index_select(0, physical[0])
    if not torch.equal(groups_cpu, node_groups.index_select(0, physical[1])):
        raise ValueError("a physical edge must stay within its graph")
    counts = torch.bincount(groups_cpu, minlength=graphs).to(edges.device)
    edge_graph = groups_cpu.to(edges.device)

    def discard(reference: weakref.ReferenceType[Tensor]) -> None:
        current = _GROUP_CACHE.get(key)
        if current is not None and (current.edges is reference or current.node_graph is reference):
            del _GROUP_CACHE[key]

    result = _EdgeGroups(weakref.ref(edges, discard), weakref.ref(node_graph, discard),
                         versions, edge_graph, counts)
    if None not in versions:
        _GROUP_CACHE[key] = result
    return result


def physical_edge_rms(batch: Any) -> Tensor:
    """Return sigma[G,R] from every physical edge, independently per scalar field.

    Sigma=1 for exactly zero edge signal or no physical edges. This represents
    a mathematical zero message and introduces no nonzero-field epsilon.
    """
    _real_matrix(batch.x, "batch.x")
    x = batch.x
    version = _version(x)
    key = (id(x), id(batch.edges), id(batch.node_graph), batch.num_graphs)
    cached = _SCALE_CACHE.get(key)
    if not x.requires_grad and version is not None and cached is not None:
        groups = cached.groups
        if (cached.x() is x and cached.version == version
                and groups.edges() is batch.edges and groups.node_graph() is batch.node_graph
                and groups.versions == (_version(batch.edges), _version(batch.node_graph))
                and None not in groups.versions):
            return cached.sigma
    # Shared incidence validation precedes indexing endpoints in _edge_groups.
    differences = incidence_apply(batch.edges, x)
    groups = _edge_groups(batch)
    energy = x.new_zeros((batch.num_graphs, x.shape[1])).index_add(
        0, groups.edge_graph, differences.square()
    ) / groups.counts.clamp_min(1).to(dtype=x.dtype)[:, None]
    # Replacing zero BEFORE sqrt avoids the 0 * inf gradient from sqrt(0).
    sigma = torch.where(energy > 0, energy, torch.ones_like(energy)).sqrt()
    if not x.requires_grad and version is not None and None not in groups.versions:
        def discard(reference: weakref.ReferenceType[Tensor]) -> None:
            current = _SCALE_CACHE.get(key)
            if current is not None and current.x is reference:
                del _SCALE_CACHE[key]
        _SCALE_CACHE[key] = _InputScale(weakref.ref(x, discard), version, groups, sigma)
    return sigma


class NormalizedSeedBatchedModel(SeedBatchedModel):
    """Unchanged seed-batched gate/readout, with physical-edge RMS gate inputs.

    Parameters, Xavier initialization, state_dict names and beta are exactly
    those of the original learned/random_pair model. No teacher tensor enters
    the forward; the original-amplitude x drives the weighted message.
    """

    def __init__(self, kind: str, seeds: Sequence[int], hidden: int = 64, tau: float = 1.0):
        if kind not in ("learned", "random_pair"):
            raise ValueError("gate normalization supports only learned and random_pair")
        super().__init__(kind, seeds, hidden, tau)

    def forward(self, batch: Any) -> tuple[Tensor, Tensor]:
        _real_matrix(batch.x, "batch.x")
        parameter = next(self.parameters())
        if parameter.dtype != batch.x.dtype or parameter.device != batch.x.device:
            raise ValueError("model parameters and batch.x must have the same dtype and device")
        sigma = physical_edge_rms(batch)
        if self.kind == "learned":
            g1 = batch.x.index_select(0, batch.wedges[1]) - batch.x.index_select(0, batch.wedges[0])
            g2 = batch.x.index_select(0, batch.wedges[2]) - batch.x.index_select(0, batch.wedges[1])
        else:
            g1, g2 = pair_gradients(batch.edges, batch.pair_edges, batch.pair_coefficients, batch.x)
        if batch.path_graph.device != batch.x.device:
            raise ValueError("path_graph and batch.x must have the same device")
        _index_info(batch.path_graph, batch.num_graphs, g1.shape[0], "path_graph", groups=True)
        path_sigma = sigma.index_select(0, batch.path_graph)
        c = self.gate(g1 / path_sigma, g2 / path_sigma, batch.path_graph, batch.num_graphs)
        if self.kind == "learned":
            message = apply_weighted_wedge(batch.wedges, batch.x, c)
        else:
            message = apply_weighted_pair(
                batch.edges, batch.pair_edges, batch.pair_coefficients, batch.x, c
            )
        return self.beta[:, None, None] * message, c


def make_normalized_model(kind: str, seeds: Sequence[int], hidden: int = 64,
                          tau: float = 1.0) -> NormalizedSeedBatchedModel:
    """Construct the independent normalized variant on CPU in float64."""
    return NormalizedSeedBatchedModel(kind, seeds, hidden, tau)
