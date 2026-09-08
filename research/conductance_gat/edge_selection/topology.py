"""Label-free DFS forests and implicit fundamental-cycle operators.

The CPU DFS and adjacency construction take O(N+E) time/space. No cycle paths,
cycle incidence matrix, dense adjacency, or cycle-count cap is materialized.
Torch prefix/scatter operators cost O((N+E)*channels) and preserve autograd.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class TopologyPlan:
    incidence_edge_index: Tensor
    forest_mask: Tensor
    edge_graph: Tensor
    node_graph: Tensor
    components: Tensor
    parent: Tensor
    parent_edge: Tensor
    depth: Tensor
    preorder: Tensor
    tin: Tensor
    tout: Tensor
    tree_nodes: Tensor
    tree_child: Tensor
    tree_sign: Tensor
    chord_indices: Tensor
    chord_ancestor: Tensor
    chord_descendant: Tensor
    chord_sign: Tensor
    cycle_length: Tensor
    cycle_count: Tensor
    random_priority: Tensor
    metadata: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return self.node_graph.numel()

    @property
    def num_edges(self) -> int:
        return self.forest_mask.numel()

    @property
    def num_graphs(self) -> int:
        return self.metadata["num_graphs"]

    def to(self, device, *, non_blocking: bool = False) -> TopologyPlan:
        """Return independent tensor storage; never mutate the cached CPU plan."""
        return TopologyPlan(
            **{
                field.name: (
                    getattr(self, field.name).to(device, non_blocking=non_blocking, copy=True)
                    if isinstance(getattr(self, field.name), Tensor)
                    else dict(getattr(self, field.name))
                )
                for field in fields(self)
            }
        )

    def pin_memory(self) -> TopologyPlan:
        """Pin every index/value tensor for a real asynchronous CPU-to-CUDA transfer.

        Unsupported pinning raises the actual runtime error; no fake CUDA or
        silent unpinned fallback. Cached tensors are never replaced in-place.
        """
        if any(
            getattr(self, field.name).device.type != "cpu"
            for field in fields(self)
            if isinstance(getattr(self, field.name), Tensor)
        ):
            raise ValueError("only CPU topology plans can be pinned")
        return TopologyPlan(
            **{
                field.name: (
                    getattr(self, field.name).pin_memory()
                    if isinstance(getattr(self, field.name), Tensor)
                    else dict(getattr(self, field.name))
                )
                for field in fields(self)
            }
        )


def validate_cpu_topology(num_nodes: int, incidence: Tensor, node_graph: Tensor | None = None):
    if type(num_nodes) is not int or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    if (
        not isinstance(incidence, Tensor)
        or incidence.device.type != "cpu"
        or incidence.dtype != torch.long
        or incidence.ndim != 2
        or incidence.shape[0] != 2
    ):
        raise ValueError("incidence must be a CPU int64 tensor with shape 2 x E")
    if node_graph is None:
        node_graph = torch.zeros(num_nodes, dtype=torch.long)
    if (
        not isinstance(node_graph, Tensor)
        or node_graph.device.type != "cpu"
        or node_graph.dtype != torch.long
        or node_graph.shape != (num_nodes,)
    ):
        raise ValueError("node_graph must contain one CPU int64 graph ID per node")
    graph = node_graph.numpy()
    if graph.size and (
        graph.min() < 0
        or graph.max() >= num_nodes
        or np.count_nonzero(np.bincount(graph)) != graph.max() + 1
    ):
        raise ValueError("node_graph IDs must be contiguous and start at zero")
    edge = incidence.numpy()
    if edge.size:
        if edge.min() < 0 or edge.max() >= num_nodes:
            raise ValueError("edge endpoint outside node range")
        if np.any(edge[0] == edge[1]):
            raise ValueError("physical incidence must not contain self-loops")
        if np.any(graph[edge[0]] != graph[edge[1]]):
            raise ValueError("physical edges may not cross original graph boundaries")
        unique = set()
        for left, right in zip(edge[0], edge[1], strict=True):
            pair = (min(int(left), int(right)), max(int(left), int(right)))
            if pair in unique:
                raise ValueError("physical incidence contains duplicate undirected edges")
            unique.add(pair)
    return edge, graph


def topology_fingerprint(num_nodes: int, incidence: Tensor, node_graph: Tensor) -> str:
    """Exact aligned-input identity, not a hash of labels or features."""
    digest = hashlib.sha256(str(num_nodes).encode("ascii"))
    for tensor in (incidence, node_graph):
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.contiguous().numpy().astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _priority(edge: np.ndarray, seed: int) -> np.ndarray:
    """Endpoint/orientation/order-stable splitmix64 priorities, independent of labels."""
    low, high = (
        np.minimum(edge[0], edge[1]).astype(np.uint64),
        np.maximum(edge[0], edge[1]).astype(np.uint64),
    )
    value = low * np.uint64(0x9E3779B97F4A7C15) ^ high * np.uint64(0xD1B54A32D192ED03)
    value ^= np.uint64(seed & ((1 << 64) - 1))
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return (value >> np.uint64(11)).astype(np.float64) / (1 << 53)


def _ordered_adjacency(num_nodes: int, edge: np.ndarray, root_order: np.ndarray):
    """Two counting/CSR passes give neighbor-rank order without comparison sorting."""
    counts = np.bincount(edge.reshape(-1), minlength=num_nodes)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    cursor = offsets[:-1].copy()
    neighbors = np.empty(2 * edge.shape[1], dtype=np.int64)
    ids = np.empty_like(neighbors)
    for edge_id, (left, right) in enumerate(zip(edge[0], edge[1], strict=True)):
        position = cursor[left]
        neighbors[position], ids[position] = right, edge_id
        cursor[left] += 1
        position = cursor[right]
        neighbors[position], ids[position] = left, edge_id
        cursor[right] += 1
    ordered_neighbors, ordered_ids = np.empty_like(neighbors), np.empty_like(ids)
    cursor = offsets[:-1].copy()
    for neighbor in root_order:
        for position in range(offsets[neighbor], offsets[neighbor + 1]):
            node = neighbors[position]
            target = cursor[node]
            ordered_neighbors[target], ordered_ids[target] = neighbor, ids[position]
            cursor[node] += 1
    return offsets, ordered_neighbors, ordered_ids


def build_topology(
    num_nodes: int,
    incidence_edge_index: Tensor,
    node_graph: Tensor | None = None,
    forest_seed: int = 0,
) -> TopologyPlan:
    """Build once from candidate edges only; edge origins/labels are not accepted."""
    if type(forest_seed) is not int or forest_seed < 0:
        raise ValueError("forest_seed must be a nonnegative integer")
    edge, graph = validate_cpu_topology(num_nodes, incidence_edge_index, node_graph)
    node_graph = torch.from_numpy(graph.copy())
    edges = edge.shape[1]
    root_order = np.random.default_rng(forest_seed).permutation(num_nodes)
    offsets, neighbors, ids = _ordered_adjacency(num_nodes, edge, root_order)
    parent = np.full(num_nodes, -2, dtype=np.int64)
    parent_edge = np.full(num_nodes, -1, dtype=np.int64)
    depth = np.zeros(num_nodes, dtype=np.int64)
    components = np.empty(num_nodes, dtype=np.int64)
    tin, tout = np.empty(num_nodes, dtype=np.int64), np.empty(num_nodes, dtype=np.int64)
    preorder = np.empty(num_nodes, dtype=np.int64)
    forest = np.zeros(edges, dtype=np.bool_)
    tree_child = np.full(edges, -1, dtype=np.int64)
    tree_sign = np.zeros(num_nodes, dtype=np.int64)
    clock, component = 0, 0
    for root in root_order:
        if parent[root] != -2:
            continue
        parent[root], components[root], tin[root] = -1, component, clock
        preorder[clock] = root
        clock += 1
        stack, positions = [int(root)], [int(offsets[root])]
        while stack:
            node, position = stack[-1], positions[-1]
            if position == offsets[node + 1]:
                tout[node] = clock
                stack.pop()
                positions.pop()
                continue
            positions[-1] += 1
            neighbor, edge_id = neighbors[position], ids[position]
            if parent[neighbor] != -2:
                continue
            parent[neighbor], parent_edge[neighbor] = node, edge_id
            depth[neighbor], components[neighbor], tin[neighbor] = depth[node] + 1, component, clock
            preorder[clock] = neighbor
            clock += 1
            forest[edge_id], tree_child[edge_id] = True, neighbor
            tree_sign[neighbor] = 1 if edge[0, edge_id] == node else -1
            stack.append(int(neighbor))
            positions.append(int(offsets[neighbor]))
        component += 1
    chord = np.flatnonzero(~forest)
    left, right = edge[:, chord]
    ancestor = np.where(tin[left] < tin[right], left, right)
    descendant = np.where(tin[left] < tin[right], right, left)
    if np.any(tin[descendant] >= tout[ancestor]):
        raise RuntimeError("DFS produced a non-ancestor chord; invalid fundamental cycle plan")
    chord_sign = np.where(left == ancestor, 1, -1)
    tree_nodes = np.flatnonzero(parent >= 0)
    impulses = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(impulses, descendant, 1)
    np.add.at(impulses, ancestor, -1)
    prefix = np.concatenate(([0], np.cumsum(impulses[preorder], dtype=np.int64)))
    cycle_count = np.zeros(edges, dtype=np.int64)
    cycle_count[chord] = 1
    cycle_count[parent_edge[tree_nodes]] = prefix[tout[tree_nodes]] - prefix[tin[tree_nodes]]
    arrays = {
        "forest_mask": forest,
        "edge_graph": graph[edge[0]],
        "components": components,
        "parent": parent,
        "parent_edge": parent_edge,
        "depth": depth,
        "preorder": preorder,
        "tin": tin,
        "tout": tout,
        "tree_nodes": tree_nodes,
        "tree_child": tree_child,
        "tree_sign": tree_sign,
        "chord_indices": chord,
        "chord_ancestor": ancestor,
        "chord_descendant": descendant,
        "chord_sign": chord_sign,
        "cycle_length": depth[descendant] - depth[ancestor] + 1,
        "cycle_count": cycle_count,
        "random_priority": _priority(edge, forest_seed),
    }
    return TopologyPlan(
        incidence_edge_index=incidence_edge_index.clone(),
        node_graph=node_graph,
        **{name: torch.from_numpy(value.copy()) for name, value in arrays.items()},
        metadata={
            "forest_seed": forest_seed,
            "num_nodes": num_nodes,
            "num_edges": edges,
            "num_graphs": int(graph.max() + 1) if graph.size else 0,
            "num_components": component,
            "cycle_rank": len(chord),
            "dfs_complexity": "O(N+E) time and storage; no explicit cycle paths",
            "basis": "one DFS fundamental cycle per non-tree physical edge",
            "forest_input": "candidate topology only; label/origin blind",
            "priority": "endpoint splitmix64, 53-bit float64; seed=forest_seed",
            "topology_sha256": topology_fingerprint(num_nodes, incidence_edge_index, node_graph),
        },
    )


def _values(values: Tensor, plan: TopologyPlan, count: int):
    if values.ndim < 1 or values.shape[0] != count or not values.is_floating_point():
        raise ValueError(
            "values must be floating tensors with the exact edge/cycle leading dimension"
        )
    if values.device != plan.incidence_edge_index.device:
        raise ValueError("values and topology plan must be on the same device")


def _factor(value: Tensor, target: Tensor) -> Tensor:
    return value.to(target.dtype).reshape((-1,) + (1,) * (target.ndim - 1))


def edge_to_cycle(edge_values: Tensor, plan: TopologyPlan, *, signed: bool = True) -> Tensor:
    """Z^T x (signed) or |Z|^T x via subtree interval updates and one prefix sum."""
    _values(edge_values, plan, plan.num_edges)
    original_dtype = edge_values.dtype
    edge_values = edge_values.to(torch.float64)
    nodes = plan.tree_nodes
    tree_value = edge_values[plan.parent_edge[nodes]]
    if signed:
        tree_value = tree_value * _factor(plan.tree_sign[nodes], tree_value)
    difference = edge_values.new_zeros((plan.num_nodes + 1,) + edge_values.shape[1:])
    difference = difference.index_add(0, plan.tin[nodes], tree_value)
    difference = difference.index_add(0, plan.tout[nodes], -tree_value)
    potential = difference.cumsum(0)
    path = potential[plan.tin[plan.chord_descendant]] - potential[plan.tin[plan.chord_ancestor]]
    chord = edge_values[plan.chord_indices]
    result = chord - _factor(plan.chord_sign, path) * path if signed else chord + path
    return result.to(original_dtype)


def cycle_to_edge(cycle_values: Tensor, plan: TopologyPlan, *, signed: bool = True) -> Tensor:
    """Z y or |Z| y, exact adjoint to edge_to_cycle without cycle-path expansion."""
    _values(cycle_values, plan, plan.chord_indices.numel())
    original_dtype = cycle_values.dtype
    cycle_values = cycle_values.to(torch.float64)
    values = -_factor(plan.chord_sign, cycle_values) * cycle_values if signed else cycle_values
    impulses = cycle_values.new_zeros((plan.num_nodes,) + cycle_values.shape[1:])
    impulses = impulses.index_add(0, plan.chord_descendant, values)
    impulses = impulses.index_add(0, plan.chord_ancestor, -values)
    prefix = torch.cat(
        (impulses.new_zeros((1,) + impulses.shape[1:]), impulses[plan.preorder].cumsum(0)), dim=0
    )
    nodes = plan.tree_nodes
    tree_value = prefix[plan.tout[nodes]] - prefix[plan.tin[nodes]]
    if signed:
        tree_value = tree_value * _factor(plan.tree_sign[nodes], tree_value)
    result = cycle_values.new_zeros((plan.num_edges,) + cycle_values.shape[1:])
    result = result.index_add(0, plan.chord_indices, cycle_values)
    return result.index_add(0, plan.parent_edge[nodes], tree_value).to(original_dtype)


def cycle_context(
    edge_values: Tensor, plan: TopologyPlan, *, signed: bool = False, normalize: bool = True
) -> Tensor:
    """Cycle means then per-edge cycle mean; unsigned is orientation invariant.

    Bridges have zero context because they belong to no cycle, not because any
    cycle was truncated. normalize=False returns the unnormalized Z Z^T action.
    """
    original_dtype = edge_values.dtype
    cycles = edge_to_cycle(edge_values.to(torch.float64), plan, signed=signed)
    if normalize:
        cycles = cycles / _factor(plan.cycle_length, cycles)
    result = cycle_to_edge(cycles, plan, signed=signed)
    result = result / _factor(plan.cycle_count.clamp_min(1), result) if normalize else result
    return result.to(original_dtype)


def batch_topologies(plans: list[TopologyPlan]) -> TopologyPlan:
    """Disjoint offset concatenation of cached CPU plans; never rerun DFS."""
    if not plans:
        raise ValueError("at least one topology plan is required")
    if any(plan.incidence_edge_index.device.type != "cpu" for plan in plans):
        raise ValueError("batch cached CPU plans before moving the disjoint plan to device")
    collected = {field.name: [] for field in fields(TopologyPlan) if field.name != "metadata"}
    node_offset = edge_offset = graph_offset = component_offset = 0
    node_fields = {
        "incidence_edge_index",
        "preorder",
        "tin",
        "tout",
        "tree_nodes",
        "chord_ancestor",
        "chord_descendant",
    }
    for plan in plans:
        for name in collected:
            value = getattr(plan, name)
            if name in node_fields:
                value = value + node_offset
            elif name in {"parent", "tree_child"}:
                value = torch.where(value >= 0, value + node_offset, value)
            elif name == "parent_edge":
                value = torch.where(value >= 0, value + edge_offset, value)
            elif name == "chord_indices":
                value = value + edge_offset
            elif name in {"edge_graph", "node_graph"}:
                value = value + graph_offset
            elif name == "components":
                value = value + component_offset
            collected[name].append(value)
        node_offset += plan.num_nodes
        edge_offset += plan.num_edges
        graph_offset += plan.metadata["num_graphs"]
        component_offset += plan.metadata["num_components"]
    tensors = {
        name: torch.cat(values, dim=1 if name == "incidence_edge_index" else 0)
        for name, values in collected.items()
    }
    return TopologyPlan(
        **tensors,
        metadata={
            "forest_seed": tuple(plan.metadata["forest_seed"] for plan in plans),
            "num_nodes": node_offset,
            "num_edges": edge_offset,
            "num_graphs": graph_offset,
            "num_components": component_offset,
            "cycle_rank": tensors["chord_indices"].numel(),
            "forest_input": "cached candidate topology plans; no rebuild or edge-origin features",
            "source_topology_sha256": tuple(plan.metadata["topology_sha256"] for plan in plans),
            "topology_sha256": topology_fingerprint(
                node_offset, tensors["incidence_edge_index"], tensors["node_graph"]
            ),
            "dfs_reexecuted": False,
        },
    )
