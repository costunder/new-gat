"""Original individual local q/E/J targets, never two scalar input lifts."""
from __future__ import annotations

import torch

from ...local_energy_relations.operators import local_weight


def evaluate_targets(topology, value, recipe):
    """Full per-local/per-adjacent-pair targets, preserving feature coordinates.

value is [...,N,F]. Energies and relations sum over F only at the output;
synthetic scalar draws occupy an independent leading axis, never F=16.
"""
    weight = local_weight(topology, recipe, dtype=value.dtype)
    ends = topology.local_node_global[topology.local_edge_nodes]
    gradient = value.index_select(-2, ends[1])-value.index_select(-2, ends[0])
    q = gradient * weight[..., None]
    d = value.new_zeros(value.shape[:-2]+(topology.num_local_nodes, value.shape[-1]))
    d = d.index_add(-2, topology.local_edge_nodes[0], -q).index_add(-2, topology.local_edge_nodes[1], q)
    energy = value.new_zeros(value.shape[:-2]+(topology.n,)).index_add(
        -1, topology.local_edge_center, (gradient.square()*weight[..., None]).sum(-1))
    shared = value.new_zeros(value.shape[:-2]+(topology.num_pairs,)).index_add(
        -1, topology.shared_edge_pair,
        (q.index_select(-2, topology.shared_edge_left)*q.index_select(-2, topology.shared_edge_right)).sum(-1))
    node = value.new_zeros(value.shape[:-2]+(topology.num_pairs,)).index_add(
        -1, topology.shared_node_pair,
        (d.index_select(-2, topology.shared_node_left)*d.index_select(-2, topology.shared_node_right)).sum(-1))
    return {"q": q, "E": energy, "J_shared": shared, "J_node": node, "J_distinct": node-2*shared}


def dense_target_matrices(topology, recipe, dtype=torch.float64):
    """Independent physical-space linear q and symmetric E/J matrices.

Only small synthetic graphs use this dense reference. Large citation graphs
always use gather/scatter evaluate_targets and matrix-free recovery.
"""
    n = topology.n
    identity = torch.eye(n, dtype=dtype, device=topology.local_node_global.device)
    weight = local_weight(topology, recipe, dtype=dtype)
    ends = topology.local_node_global[topology.local_edge_nodes]
    b = identity[ends[1]]-identity[ends[0]]
    q = weight[:, None]*b
    energy = b.new_zeros((n, n, n)).index_add(0, topology.local_edge_center,
                                            weight[:, None, None]*b[:, :, None]*b[:, None, :])
    d = b.new_zeros((topology.num_local_nodes, n)).index_add(0, topology.local_edge_nodes[0], -q)
    d = d.index_add(0, topology.local_edge_nodes[1], q)
    def relation(left, right, owner, count):
        # Stream exact correspondence chunks rather than allocating all outer products.
        output = b.new_zeros((count, n, n))
        size = max(1, 2**20//max(1, n*n))
        for begin in range(0, owner.numel(), size):
            a, c = left[begin:begin+size], right[begin:begin+size]
            block = .5*(a[:, :, None]*c[:, None, :]+c[:, :, None]*a[:, None, :])
            output.index_add_(0, owner[begin:begin+size], block)
        return output
    shared = relation(q[topology.shared_edge_left], q[topology.shared_edge_right],
                      topology.shared_edge_pair, topology.num_pairs)
    node = relation(d[topology.shared_node_left], d[topology.shared_node_right],
                    topology.shared_node_pair, topology.num_pairs)
    return {"q": q, "E": energy, "J_shared": shared, "J_node": node, "J_distinct": node-2*shared}


def observation_masks(topology, kind, start=0, stop=None, dtype=torch.float64):
    stop = topology.n if stop is None else stop
    device = topology.local_node_global.device
    if kind == "full":
        return torch.ones((1, topology.n), dtype=dtype, device=device)
    if kind not in ("target", "one_hop") or not 0 <= start < stop <= topology.n:
        raise ValueError("unknown observation or invalid complete-target chunk")
    mask = torch.zeros((stop-start, topology.n), dtype=dtype, device=device)
    if kind == "target":
        mask[torch.arange(stop-start, device=device), torch.arange(start, stop, device=device)] = 1
    else:
        selected = (topology.local_node_center >= start) & (topology.local_node_center < stop)
        mask[topology.local_node_center[selected]-start, topology.local_node_global[selected]] = 1
    return mask
