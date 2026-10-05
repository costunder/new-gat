"""Complete occurrence and adjacent-center edge-pair geometry, prepared once.

This is a physical-node operator factorization, not independently evolving
node copies. CPU enumeration retains every eligible relation without caps.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math

import numpy as np
import torch
from torch import Tensor

from ..local_energy_relations.topology import LocalTopology


@dataclass(frozen=True)
class EdgeGeometry:
    topology: LocalTopology
    recipe: str
    edges: Tensor
    occ_edge: Tensor
    occ_center: Tensor
    occ_count: Tensor
    occ_scale: Tensor
    pair_left: Tensor
    pair_right: Tensor
    pair_anchor: Tensor
    pair_other_left: Tensor
    pair_other_right: Tensor
    pair_sign: Tensor
    pair_degree: Tensor
    pair_norm: Tensor
    pair_structure: Tensor
    c0: Tensor
    cbar0: Tensor
    d0: Tensor
    n0: Tensor
    node_graph: Tensor
    edge_graph: Tensor
    occ_graph: Tensor
    pair_graph: Tensor
    graph_node_offsets: Tensor
    graph_edge_offsets: Tensor
    metadata: dict
    tau: float = 1 / 3
    rho: float = .5

    @property
    def n(self):
        return self.topology.n

    @property
    def mode(self):
        return self.recipe

    @property
    def num_edges(self):
        return self.edges.shape[1]

    @property
    def num_occurrences(self):
        return self.occ_edge.numel()

    @property
    def num_pairs(self):
        return self.pair_left.numel()

    @property
    def num_cross_edges(self):
        return self.num_pairs

    @property
    def num_copies(self):
        return self.num_occurrences

    @property
    def num_graphs(self):
        return self.graph_node_offsets.numel() - 1

    def to(self, device, dtype=torch.float64, *, shared=None):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("geometry dtype must be float32 or float64")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if shared is not None:
            if not isinstance(shared, EdgeGeometry) or (
                shared.metadata["pair_index_sha256"] != self.metadata["pair_index_sha256"]
                or (shared.n, shared.num_graphs, shared.num_edges, shared.num_occurrences, shared.num_pairs)
                != (self.n, self.num_graphs, self.num_edges, self.num_occurrences, self.num_pairs)
            ):
                raise ValueError("shared geometry must have identical physical/occurrence/pair support")
            # All recipe-independent tensors are reused, including the topology.
            # Only four recipe weights are transferred to the selected device.
            if shared.edges.device != device or shared.c0.dtype != dtype:
                raise ValueError("shared geometry must already have the target device and dtype")
            return replace(shared, recipe=self.recipe, metadata=self.metadata,
                           **{name: getattr(self, name).to(device=device, dtype=dtype)
                              for name in ("c0", "cbar0", "d0", "n0")})
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                value = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            elif field.name == "topology":
                value = value.to(device)
            values[field.name] = value
        return EdgeGeometry(**values)


def _array(value):
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise ValueError("prepare geometry on CPU before device transfer")
        value = value.detach().numpy()
    return np.asarray(value)


def with_recipe(geometry: EdgeGeometry, recipe: str) -> EdgeGeometry:
    """Change fixed weights while sharing every recipe-independent tensor.

    Prepare both recipes on CPU before transfer to keep deterministic
    preprocessing identical to the independent construction.
    """
    if not isinstance(geometry, EdgeGeometry):
        raise TypeError("geometry must be EdgeGeometry")
    if recipe not in ("unit", "local_degree"):
        raise ValueError("recipe must be unit or local_degree")
    if geometry.recipe == recipe:
        return geometry
    top = geometry.topology
    edge, occurrence = _array(geometry.edges), _array(geometry.occ_edge)
    counts = _array(geometry.occ_count)
    c0 = np.ones(geometry.num_occurrences, dtype=np.float64)
    if recipe == "local_degree":
        endpoints, degree = _array(top.local_edge_nodes), _array(top.local_degree)
        c0 = 2 / (degree[endpoints[0]] + degree[endpoints[1]])
    cbar = np.bincount(occurrence, weights=c0, minlength=geometry.num_edges) / counts
    d0 = np.bincount(edge.reshape(-1), weights=np.tile(cbar, 2), minlength=geometry.n)
    floating = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=geometry.c0.dtype)
    return replace(geometry, recipe=recipe, metadata={**geometry.metadata, "recipe": recipe},
                   c0=floating(c0), cbar0=floating(cbar), d0=floating(d0), n0=floating((1+d0)**-.5))


def prepare_geometries(topology: LocalTopology) -> dict[str, EdgeGeometry]:
    """Enumerate complete support once for both fixed-conductance recipes."""
    unit = prepare_geometry(topology, "unit")
    return {"unit": unit, "local_degree": with_recipe(unit, "local_degree")}


def prepare_geometry(topology: LocalTopology, recipe="unit") -> EdgeGeometry:
    if not isinstance(topology, LocalTopology):
        raise TypeError("topology must be LocalTopology")
    if recipe not in ("unit", "local_degree"):
        raise ValueError("recipe must be unit or local_degree")
    top = topology
    edge = _array(top.edges)
    occ_edge, occ_center = _array(top.local_edge_global), _array(top.local_edge_center)
    local_endpoints = _array(top.local_edge_nodes)
    local_global = _array(top.local_node_global)
    local_degree = _array(top.local_degree)
    sizes = np.diff(_array(top.node_offsets))
    counts = np.bincount(occ_edge, minlength=top.num_edges)
    if top.n < 1 or np.any(counts < 1):
        raise ValueError("all physical nodes and edges must belong to their complete induced locals")
    # CSR of occurrences incident to each local-node copy. The Cartesian
    # product is created only for one actual shared endpoint at a time.
    ids = np.concatenate((np.arange(len(occ_edge)), np.arange(len(occ_edge))))
    endpoints = np.concatenate((local_endpoints[0], local_endpoints[1]))
    order = np.argsort(endpoints, kind="stable")
    ids = ids[order]
    offsets = np.r_[0, np.cumsum(np.bincount(endpoints, minlength=top.num_local_nodes))]
    pair_centers = _array(top.pair_centers)
    shared_pair = _array(top.shared_node_pair)
    shared_left, shared_right = _array(top.shared_node_left), _array(top.shared_node_right)
    overlaps = np.bincount(shared_pair, minlength=top.num_pairs)
    lefts, rights, anchors, structures = [], [], [], []
    for link in np.flatnonzero(pair_centers[0, shared_pair] < pair_centers[1, shared_pair]):
        lcopy, rcopy = int(shared_left[link]), int(shared_right[link])
        l = ids[offsets[lcopy]:offsets[lcopy + 1]]
        r = ids[offsets[rcopy]:offsets[rcopy + 1]]
        if not len(l) or not len(r):
            continue
        ll, rr = np.repeat(l, len(r)), np.tile(r, len(l))
        keep = occ_edge[ll] != occ_edge[rr]
        ll, rr = ll[keep], rr[keep]
        if not len(ll):
            continue
        anchor = int(local_global[lcopy])
        if anchor != int(local_global[rcopy]):
            raise ValueError("shared copies must have the same physical node")
        v, u = pair_centers[:, shared_pair[link]]
        dl, dr = math.log1p(int(local_degree[lcopy])), math.log1p(int(local_degree[rcopy]))
        structural = [dl + dr, abs(dl - dr), math.log1p(int(sizes[v])) + math.log1p(int(sizes[u])),
                      overlaps[shared_pair[link]] / math.sqrt(float(sizes[v] * sizes[u]))]
        lefts.append(ll); rights.append(rr)
        anchors.append(np.full(len(ll), anchor, dtype=np.int64))
        structures.append(np.broadcast_to(structural, (len(ll), 4)).copy())
    left = np.concatenate(lefts) if lefts else np.empty(0, dtype=np.int64)
    right = np.concatenate(rights) if rights else np.empty(0, dtype=np.int64)
    anchor = np.concatenate(anchors) if anchors else np.empty(0, dtype=np.int64)
    structure = np.concatenate(structures) if structures else np.empty((0, 4), dtype=np.float64)
    el, er = occ_edge[left], occ_edge[right]
    other_left = np.where(edge[0, el] == anchor, edge[1, el], edge[0, el])
    other_right = np.where(edge[0, er] == anchor, edge[1, er], edge[0, er])
    sign = np.where(edge[0, el] == anchor, -1., 1.) * np.where(edge[0, er] == anchor, -1., 1.)
    degree = np.bincount(np.r_[left, right], minlength=len(occ_edge))
    norm = 1 / np.sqrt(np.maximum(1, degree[left]) * np.maximum(1, degree[right]))
    c0 = np.ones(len(occ_edge), dtype=np.float64)
    if recipe == "local_degree":
        c0 = 2 / (local_degree[local_endpoints[0]] + local_degree[local_endpoints[1]])
    cbar = np.bincount(occ_edge, weights=c0, minlength=top.num_edges) / counts
    d0 = np.bincount(edge.reshape(-1), weights=np.tile(cbar, 2), minlength=top.n)
    gn, ge = _array(top.graph_node_offsets), _array(top.graph_edge_offsets)
    node_graph = np.repeat(np.arange(top.num_graphs), np.diff(gn))
    edge_graph = np.repeat(np.arange(top.num_graphs), np.diff(ge))
    if np.any(node_graph[edge[0]] != node_graph[edge[1]]):
        raise ValueError("physical edges cannot cross graph components")
    digest = hashlib.sha256()
    for array in (edge, occ_edge, occ_center, left, right, anchor):
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
    integer = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.long)
    floating = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64)
    return EdgeGeometry(
        topology=top.to("cpu"), recipe=recipe, edges=integer(edge), occ_edge=integer(occ_edge),
        occ_center=integer(occ_center), occ_count=floating(counts), occ_scale=floating(counts[occ_edge] ** -.5),
        pair_left=integer(left), pair_right=integer(right), pair_anchor=integer(anchor),
        pair_other_left=integer(other_left), pair_other_right=integer(other_right), pair_sign=floating(sign),
        pair_degree=integer(degree), pair_norm=floating(norm), pair_structure=floating(structure),
        c0=floating(c0), cbar0=floating(cbar), d0=floating(d0), n0=floating((1 + d0) ** -.5),
        node_graph=integer(node_graph), edge_graph=integer(edge_graph), occ_graph=integer(edge_graph[occ_edge]),
        pair_graph=integer(edge_graph[el]), graph_node_offsets=integer(gn), graph_edge_offsets=integer(ge),
        metadata={"recipe": recipe, "num_nodes": top.n, "num_edges": top.num_edges,
                  "num_graphs": top.num_graphs, "num_occurrences": len(occ_edge), "num_pairs": len(left),
                  "eligible_pairs": "all_adjacent_center_distinct_physical_edges_common_endpoint",
                  "same_physical_edge_pairs_excluded": True, "all_eligible_pairs_retained": True,
                  "pair_index_sha256": digest.hexdigest(), "occurrence_correction": "1/sqrt(r_e)",
                  "tau": 1 / 3, "rho": .5, "epsilon": 1e-4, "sampling_ratio": 1.0},
    )


__all__ = ["EdgeGeometry", "prepare_geometry", "prepare_geometries", "with_recipe"]
