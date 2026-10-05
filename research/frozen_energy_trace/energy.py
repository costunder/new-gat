"""Exact frozen induced-local energy and the audited incidence J decomposition.

Every physical node/edge and both orientations of adjacent local centers are
retained. CPU prepares static topology in parallel once; CUDA evaluates all
locals, pairs and independent stages together. Feature chunks change only
temporary memory. No learned C/K, eigendecomposition or optimizer is involved.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from numbers import Integral

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class EnergyTopology:
    n: int
    edges: Tensor
    pair_centers: Tensor
    physical_degree: Tensor
    local_node_global: Tensor
    local_node_center: Tensor
    local_node_count: Tensor
    local_degree: Tensor
    local_edge_global: Tensor
    local_edge_center: Tensor
    local_edge_nodes: Tensor
    shared_node_pair: Tensor
    shared_node_left: Tensor
    shared_node_right: Tensor
    shared_edge_pair: Tensor
    shared_edge_global: Tensor
    workers: int

    @property
    def num_edges(self) -> int:
        return self.edges.shape[1]

    @property
    def num_pairs(self) -> int:
        return self.pair_centers.shape[1]

    @property
    def num_local_nodes(self) -> int:
        return self.local_node_global.numel()

    @property
    def num_local_edges(self) -> int:
        return self.local_edge_global.numel()

    def to(self, device: str | torch.device) -> EnergyTopology:
        return EnergyTopology(
            **{
                field.name: getattr(self, field.name)
                if field.name in ("n", "workers")
                else getattr(self, field.name).to(device=device)
                for field in fields(self)
            }
        )


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _concatenate(rows):
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)


def build_topology(n: int, edges, workers: int) -> EnergyTopology:
    """Every S_v={v}∪N(v), all induced edges, exact shared nodes/physical edges.

    Physical edge IDs have a common canonical orientation across local copies.
    pair_centers[:,p]=(v,u) lists both directions of every physical edge. Static
    overlap rows never construct the Cartesian product of local edge pairs.
    """
    n = _positive_integer(n, "n")
    workers = _positive_integer(workers, "workers")
    if isinstance(edges, Tensor):
        if edges.device.type != "cpu":
            raise ValueError("static topology construction requires CPU physical edges")
        edges = edges.detach().numpy()
    supplied = np.asarray(edges)
    if supplied.ndim != 2 or supplied.shape[0] != 2 or supplied.dtype.kind not in "iu":
        raise ValueError("edges must be integer [2,E]")
    edge = np.sort(supplied.astype(np.int64, copy=False), axis=0)
    if np.any(edge < 0) or np.any(edge >= n) or np.any(edge[0] == edge[1]):
        raise ValueError("physical edges require valid distinct endpoints")
    edge = np.ascontiguousarray(edge[:, np.lexsort((edge[1], edge[0]))])
    if edge.shape[1] > 1 and np.any(np.all(edge[:, 1:] == edge[:, :-1], axis=0)):
        raise ValueError("duplicate physical edges are not allowed")
    if n > math.isqrt(np.iinfo(np.int64).max):
        raise ValueError("node count exceeds exact int64 edge-key capacity")
    key = edge[0] * n + edge[1]
    neighbors = [set() for _ in range(n)]
    for a, b in edge.T:
        neighbors[a].add(int(b))
        neighbors[b].add(int(a))
    closed_sets = [row | {v} for v, row in enumerate(neighbors)]
    closed = [np.asarray(sorted(row), dtype=np.int64) for row in closed_sets]

    def induced(v):
        endpoints = [
            (a, b) for a in closed[v] for b in sorted(neighbors[a] & closed_sets[v]) if a < b
        ]
        if not endpoints:
            return np.empty(0, dtype=np.int64)
        ends = np.asarray(endpoints, dtype=np.int64).T
        return np.searchsorted(key, ends[0] * n + ends[1]).astype(np.int64)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        local_edges = list(pool.map(induced, range(n)))
    local_node_count = np.asarray([len(row) for row in closed], dtype=np.int64)
    node_offsets = np.r_[0, np.cumsum(local_node_count)]
    node_maps = [
        {node: int(node_offsets[v] + index) for index, node in enumerate(row)}
        for v, row in enumerate(closed)
    ]
    local_edge_sets = [set(row.tolist()) for row in local_edges]
    local_edge_count = np.asarray([len(row) for row in local_edges], dtype=np.int64)
    local_edge_nodes = (
        np.asarray(
            [
                (node_maps[v][edge[0, e]], node_maps[v][edge[1, e]])
                for v, row in enumerate(local_edges)
                for e in row
            ],
            dtype=np.int64,
        )
        .reshape(-1, 2)
        .T.copy()
    )
    pair_centers = np.concatenate((edge, edge[::-1]), axis=1)

    def overlap(pair):
        v, u = pair_centers[:, pair]
        shared_nodes = sorted(closed_sets[v] & closed_sets[u])
        shared_edges = np.asarray(sorted(local_edge_sets[v] & local_edge_sets[u]), dtype=np.int64)
        left = np.asarray([node_maps[v][node] for node in shared_nodes], dtype=np.int64)
        right = np.asarray([node_maps[u][node] for node in shared_nodes], dtype=np.int64)
        return left, right, shared_edges

    with ThreadPoolExecutor(max_workers=workers) as pool:
        overlaps = list(pool.map(overlap, range(pair_centers.shape[1])))
    shared_node_count = np.asarray([len(row[0]) for row in overlaps], dtype=np.int64)
    shared_edge_count = np.asarray([len(row[2]) for row in overlaps], dtype=np.int64)
    return EnergyTopology(
        n=n,
        edges=torch.from_numpy(edge),
        pair_centers=torch.from_numpy(pair_centers),
        physical_degree=torch.from_numpy(np.bincount(edge.ravel(), minlength=n)),
        local_node_global=torch.from_numpy(_concatenate(closed)),
        local_node_center=torch.from_numpy(np.repeat(np.arange(n), local_node_count)),
        local_node_count=torch.from_numpy(local_node_count),
        local_degree=torch.from_numpy(
            np.bincount(local_edge_nodes.ravel(), minlength=int(node_offsets[-1]))
        ),
        local_edge_global=torch.from_numpy(_concatenate(local_edges)),
        local_edge_center=torch.from_numpy(np.repeat(np.arange(n), local_edge_count)),
        local_edge_nodes=torch.from_numpy(local_edge_nodes),
        shared_node_pair=torch.from_numpy(
            np.repeat(np.arange(pair_centers.shape[1]), shared_node_count)
        ),
        shared_node_left=torch.from_numpy(_concatenate([row[0] for row in overlaps])),
        shared_node_right=torch.from_numpy(_concatenate([row[1] for row in overlaps])),
        shared_edge_pair=torch.from_numpy(
            np.repeat(np.arange(pair_centers.shape[1]), shared_edge_count)
        ),
        shared_edge_global=torch.from_numpy(_concatenate([row[2] for row in overlaps])),
        workers=workers,
    )


def _degree_scale(degree: Tensor, augmented: bool = False) -> Tensor:
    degree = degree.to(torch.float64) + int(augmented)
    # Degree-zero isolated locals have no edges and exactly zero energy.
    return torch.where(degree > 0, degree.clamp_min(1).rsqrt(), 0)


@torch.no_grad()
def measure(
    top: EnergyTopology, h: Tensor, feature_chunk: int, epsilon: float = 1e-12
) -> dict[str, Tensor]:
    """E/J fields for [N,F] or packed independent [S,N,F], CUDA FP64 reductions.

    E_sym uses local D^-1/2 L D^-1/2; E_sym_augmented uses local (D+I)^-1/2
    L (D+I)^-1/2, matching self-loop-augmented GCN normalization. Both energies
    retain amplitude dependence. Separate norm quotients remove scalar amplitude.
    J_node is the shared-node divergence product; J_distinct=J_node-2J_shared
    retains signed cross-edge terms and is never clipped or called a cosine.
    """
    feature_chunk = _positive_integer(feature_chunk, "feature_chunk")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if h.ndim not in (2, 3) or h.shape[-2] != top.n or h.shape[-1] < 1:
        raise ValueError("features must be [N,F] or [S,N,F] with positive channels")
    if h.ndim == 3 and h.shape[0] < 1:
        raise ValueError("independent stage count must be positive")
    if h.device.type != "cuda" or top.edges.device != h.device:
        raise ValueError("feature arithmetic requires CUDA features and matching CUDA topology")
    if h.dtype not in (torch.float32, torch.float64):
        raise ValueError("frozen energy features require float32 or float64")
    squeeze = h.ndim == 2
    values = h.unsqueeze(0) if squeeze else h
    batch, _, channels = values.shape
    node_zero = torch.zeros((batch, top.n), dtype=torch.float64, device=h.device)
    energy, symmetric, augmented, topoood, centered, norm = (node_zero.clone() for _ in range(6))
    pair_zero = node_zero.new_zeros((batch, top.num_pairs))
    shared, node_relation = (pair_zero.clone() for _ in range(2))
    global_energy, global_sym, global_aug, global_norm, global_centered = (
        node_zero.new_zeros(batch) for _ in range(5)
    )
    local_scale = _degree_scale(top.local_degree)[None, :, None]
    local_aug_scale = _degree_scale(top.local_degree, True)[None, :, None]
    physical_scale = _degree_scale(top.physical_degree)[None, :, None]
    physical_aug_scale = _degree_scale(top.physical_degree, True)[None, :, None]
    counts = top.local_node_count.to(torch.float64)[None, :, None]
    for start in range(0, channels, feature_chunk):
        x = values[..., start : start + feature_chunk].to(torch.float64)
        width = x.shape[-1]
        q = x.index_select(1, top.edges[1]) - x.index_select(1, top.edges[0])
        edge_energy = q.square().sum(-1)
        global_energy.add_(edge_energy.sum(-1))
        global_norm.add_(x.square().sum((1, 2)))
        global_centered.add_((x - x.mean(1, keepdim=True)).square().sum((1, 2)))
        for scale, target in ((physical_scale, global_sym), (physical_aug_scale, global_aug)):
            coordinates = x * scale
            delta = coordinates.index_select(1, top.edges[1]) - coordinates.index_select(
                1, top.edges[0]
            )
            normalized_edge_energy = delta.square().sum(-1)
            target.add_(normalized_edge_energy.sum(-1))
            if target is global_aug:
                # TopoOOD's k=1 ego-edge control retains GLOBAL augmented degree.
                topoood.index_add_(
                    1,
                    top.local_edge_center,
                    normalized_edge_energy.index_select(1, top.local_edge_global),
                )
        del coordinates, delta, normalized_edge_energy
        energy.index_add_(
            1, top.local_edge_center, edge_energy.index_select(1, top.local_edge_global)
        )
        shared.index_add_(
            1, top.shared_edge_pair, edge_energy.index_select(1, top.shared_edge_global)
        )
        local_nodes = x.index_select(1, top.local_node_global)
        mean = (
            x.new_zeros((batch, top.n, width)).index_add_(1, top.local_node_center, local_nodes)
            / counts
        )
        centered_values = local_nodes - mean.index_select(1, top.local_node_center)
        centered.index_add_(1, top.local_node_center, centered_values.square().sum(-1))
        norm.index_add_(1, top.local_node_center, local_nodes.square().sum(-1))
        del mean, centered_values
        for scale, target in ((local_scale, symmetric), (local_aug_scale, augmented)):
            coordinates = local_nodes * scale
            delta = coordinates.index_select(1, top.local_edge_nodes[1]) - coordinates.index_select(
                1, top.local_edge_nodes[0]
            )
            target.index_add_(1, top.local_edge_center, delta.square().sum(-1))
        del coordinates, delta, local_nodes
        local_q = q.index_select(1, top.local_edge_global)
        divergence = x.new_zeros((batch, top.num_local_nodes, width))
        divergence.index_add_(1, top.local_edge_nodes[0], -local_q)
        divergence.index_add_(1, top.local_edge_nodes[1], local_q)
        del local_q, q
        cross = (
            divergence.index_select(1, top.shared_node_left)
            * divergence.index_select(1, top.shared_node_right)
        ).sum(-1)
        node_relation.index_add_(1, top.shared_node_pair, cross)
        del divergence, cross, edge_energy
    distinct = node_relation - 2 * shared
    pair_denominator = (
        energy.index_select(1, top.pair_centers[0]) * energy.index_select(1, top.pair_centers[1])
    ).sqrt()
    result = {
        "E": energy,
        "E_sym": symmetric,
        "E_sym_augmented": augmented,
        "E_topoood_1hop": topoood,
        "E_scale_free": energy / (centered + epsilon),
        "E_sym_scale_free": symmetric / (norm + epsilon),
        "E_sym_augmented_scale_free": augmented / (norm + epsilon),
        "E_topoood_1hop_scale_free": topoood / (norm + epsilon),
        "centered_norm_sq": centered,
        "local_feature_norm_sq": norm,
        "zero_E_denominator": centered == 0,
        "zero_sym_denominator": norm == 0,
        "zero_J_denominator": pair_denominator == 0,
        "J_shared": shared,
        "J_node": node_relation,
        "J_distinct": distinct,
        "J_shared_normalized": shared / (pair_denominator + epsilon),
        "J_node_normalized": node_relation / (pair_denominator + epsilon),
        "J_distinct_normalized": distinct / (pair_denominator + epsilon),
        "global_E": global_energy,
        "global_E_per_node": global_energy / top.n,
        "global_E_sym": global_sym,
        "global_E_sym_augmented": global_aug,
        "global_feature_norm_sq": global_norm,
        "global_centered_norm_sq": global_centered,
        "global_E_scale_free": global_energy / (global_centered + epsilon),
        "global_E_rayleigh": global_energy / (global_norm + epsilon),
        "global_E_sym_scale_free": global_sym / (global_norm + epsilon),
        "global_E_sym_augmented_scale_free": global_aug / (global_norm + epsilon),
        "zero_global_norm_denominator": global_norm == 0,
    }
    return {name: tensor[0] for name, tensor in result.items()} if squeeze else result


__all__ = ["EnergyTopology", "build_topology", "measure"]
