"""Exact local incidence features and cached two-hop common propagation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..operators import local_weight
from ..receiver_aggregation.operators import _scipy_csr, prepare_receiver_operator
from ..topology import CORRESPONDENCE_KINDS, LocalTopology

WEIGHTS = ("unit", "local_degree")
_FIELDS = ("weights", "energy_normalizer", "relation_normalizer", "transition")


@dataclass(frozen=True)
class PredictionGeometry:
    topology: LocalTopology
    weights: dict[str, Tensor]
    energy_normalizer: dict[str, Tensor]
    relation_normalizer: dict[str, Tensor]
    transition: dict[str, Tensor]
    edge_global_nodes: Tensor
    physical_degree: Tensor
    metadata: dict

    def to(self, device, dtype=torch.float32):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("prediction geometry requires float32 or float64")
        return PredictionGeometry(
            self.topology.to(device),
            *(
                {k: v.to(device=device, dtype=dtype) for k, v in getattr(self, f).items()}
                for f in _FIELDS
            ),
            self.edge_global_nodes.to(device),
            self.physical_degree.to(device),
            self.metadata,
        )


def prepare_geometry(topology):
    """CPU-only exact coefficients; no feature-dependent global normalization."""
    top = topology.to("cpu")
    weights, energies, relations, transitions = {}, {}, {}, {}
    for mode in WEIGHTS:
        c = local_weight(top, mode)
        weighted_degree = c.new_zeros(top.num_local_nodes)
        weighted_degree.index_add_(0, top.local_edge_nodes[0], c)
        weighted_degree.index_add_(0, top.local_edge_nodes[1], c)
        total = c.new_zeros(top.n).index_add(0, top.local_edge_center, c)
        frobenius = c.new_zeros(top.n).index_add(
            0, top.local_node_center, weighted_degree.square()
        ) + 2 * c.new_zeros(top.n).index_add(0, top.local_edge_center, c.square())
        pair_norm = (frobenius[top.pair_centers[0]] * frobenius[top.pair_centers[1]]).sqrt()
        # Center rows are positive directed Laplacian rows. All neighbor weights
        # are exact sums of sender-local conductances, including triangle copies.
        receiver = prepare_receiver_operator(top, mode)
        center = _scipy_csr(receiver.matrix)[top.center_positions.numpy()].tocsr()
        row, column = top.pair_centers.numpy()
        coefficient = -np.asarray(center[row, column]).reshape(-1)
        degree = center.diagonal()
        if len(coefficient) and (np.any(coefficient <= 0) or np.any(degree[row] <= 0)):
            raise ValueError("receiver center Laplacian lacks positive neighbor conductances")
        transition = coefficient / degree[row] if len(coefficient) else coefficient
        row_sum = np.bincount(row, weights=transition, minlength=top.n)
        active = np.bincount(top.edges.numpy().ravel(), minlength=top.n) > 0
        if not np.allclose(row_sum[active], 1, rtol=1e-12, atol=1e-12):
            raise ValueError("complete neighbor transition rows do not sum to one")
        weights[mode], energies[mode], relations[mode], transitions[mode] = (
            c,
            2 * total,
            pair_norm,
            torch.from_numpy(transition.copy()),
        )
    return PredictionGeometry(
        top,
        weights,
        energies,
        relations,
        transitions,
        top.local_node_global[top.local_edge_nodes],
        top.node_offsets[1:] - top.node_offsets[:-1] - 1,
        {
            "nodes": top.n,
            "edges": top.num_edges,
            "local_nodes": top.num_local_nodes,
            "local_edges": top.num_local_edges,
            "directed_relations": top.num_pairs,
            "shared_node_occurrences": top.shared_node_left.numel(),
            "all_local_nodes_edges_relations": True,
            "base_hops_per_layer": 2,
            "energy_hops_per_layer": 1,
            "relation_hops_per_layer": 2,
            "normalization": "topology_only_exact_quadratic_and_bilinear",
        },
    )


def geometry_arrays(geometry):
    if geometry.topology.edges.device.type != "cpu":
        raise ValueError("save geometry from CPU cache")
    return {
        f"{field}__{mode}": getattr(geometry, field)[mode].numpy()
        for field in _FIELDS
        for mode in WEIGHTS
    } | {
        "edge_global_nodes": geometry.edge_global_nodes.numpy(),
        "physical_degree": geometry.physical_degree.numpy(),
    }


def restore_geometry(topology, arrays, metadata):
    top = topology.to("cpu")
    expected = {f"{field}__{mode}" for field in _FIELDS for mode in WEIGHTS}
    expected |= {"edge_global_nodes", "physical_degree"}
    if set(arrays) != expected:
        raise ValueError("unknown or missing complete geometry array")
    counts = (top.num_local_edges, top.n, top.num_pairs, top.num_pairs)
    groups = []
    for field, count in zip(_FIELDS, counts, strict=True):
        values = {}
        for mode in WEIGHTS:
            value = np.asarray(arrays[f"{field}__{mode}"])
            if value.shape != (count,) or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"invalid saved geometry coefficient {field}/{mode}")
            values[mode] = torch.from_numpy(value.copy()).to(torch.float64)
        groups.append(values)
    ends = np.asarray(arrays["edge_global_nodes"])
    degree = np.asarray(arrays["physical_degree"])
    if not np.array_equal(ends, top.local_node_global[top.local_edge_nodes].numpy()):
        raise ValueError("saved global edge endpoints disagree with local incidence")
    if not np.array_equal(degree, (top.node_offsets[1:] - top.node_offsets[:-1] - 1).numpy()):
        raise ValueError("saved physical degree disagrees with exact local node sets")
    return PredictionGeometry(
        top,
        *groups,
        torch.from_numpy(ends.copy()).long(),
        torch.from_numpy(degree.copy()).long(),
        metadata,
    )


def transition(geometry, mode, z):
    top = geometry.topology
    rows, columns = top.pair_centers
    values = z.index_select(1, columns) * geometry.transition[mode][None, :, None]
    result = z.new_zeros(z.shape).index_add(1, rows, values)
    degree = geometry.physical_degree
    return result + z * (degree == 0)[None, :, None]


def two_hop(geometry, mode, z):
    return transition(geometry, mode, transition(geometry, mode, z))


def _run(function, arguments, checkpoint_chunks):
    if checkpoint_chunks and torch.is_grad_enabled():
        return checkpoint(function, *arguments, use_reentrant=False, preserve_rng_state=False)
    return function(*arguments)


def local_features(
    geometry, mode, z, *, energy=True, relation=True, chunk=None, checkpoint_chunks=True
):
    """Current projected features to exact E/J, vectorizing independent seeds.

    Every local edge and adjacent local pair participates. Chunks change only
    temporary allocation. Signed J is never rectified or converted to energy.
    """
    top = geometry.topology
    if (
        z.ndim != 3
        or z.shape[1] != top.n
        or z.shape[2] < 1
        or z.dtype != geometry.weights[mode].dtype
        or z.device != top.edges.device
    ):
        raise ValueError("features must be [seeds,nodes,channels] on geometry dtype/device")
    if chunk is not None and (
        isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk < 1
    ):
        raise ValueError("chunk must be positive or None")
    seeds, _, channels = z.shape
    e = z.new_zeros((seeds, top.n))
    d = z.new_zeros((seeds, top.num_local_nodes, channels)) if relation else None
    step = max(1, top.num_local_edges) if chunk is None else int(chunk)
    global_ends = geometry.edge_global_nodes
    for start in range(0, top.num_local_edges, step):
        stop = min(start + step, top.num_local_edges)
        endpoints = global_ends[:, start:stop]
        c = geometry.weights[mode][start:stop]

        def edge_values(value, endpoints=endpoints, c=c):
            g = value.index_select(1, endpoints[1]) - value.index_select(1, endpoints[0])
            return (g.square().sum(-1) * c[None, :], g * c[None, :, None])

        squared, flow = _run(edge_values, (z,), checkpoint_chunks)
        if energy:
            e = e.index_add(1, top.local_edge_center[start:stop], squared)
        if relation:
            d = d.index_add(1, top.local_edge_nodes[0, start:stop], -flow)
            d = d.index_add(1, top.local_edge_nodes[1, start:stop], flow)
    j = z.new_zeros((seeds, top.num_pairs))
    if relation:
        offsets = top.correspondence_offsets[CORRESPONDENCE_KINDS.index("shared_node")]
        step = max(1, top.num_pairs) if chunk is None else int(chunk)
        for start in range(0, top.num_pairs, step):
            stop = min(start + step, top.num_pairs)
            begin, end = offsets[start], offsets[stop]
            left, right = top.shared_node_left[begin:end], top.shared_node_right[begin:end]

            def pair_values(divergence, left=left, right=right):
                return (divergence.index_select(1, left) * divergence.index_select(1, right)).sum(
                    -1
                )

            values = _run(pair_values, (d,), checkpoint_chunks)
            j = j.index_add(1, top.shared_node_pair[begin:end], values)
    e_denominator = channels * geometry.energy_normalizer[mode]
    j_denominator = channels * geometry.relation_normalizer[mode]
    e_feature = e / torch.where(e_denominator > 0, e_denominator, 1)[None, :]
    pair_feature = j / torch.where(j_denominator > 0, j_denominator, 1)[None, :]
    incoming = z.new_zeros((seeds, top.n)).index_add(1, top.pair_centers[1], pair_feature)
    degree = geometry.physical_degree
    j_feature = incoming / degree.clamp_min(1)[None, :]
    return e_feature, j_feature, e, j
