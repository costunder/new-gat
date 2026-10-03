"""Exact induced one-hop local graphs and sparse overlap correspondences.

All original nodes and physical edges are retained. CPU preparation loops over
centers and directed center pairs; GPU kernels use flattened disjoint locals.
No Cartesian local-edge-pair tensor is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

import numpy as np
import torch

Array = np.ndarray | torch.Tensor
CORRESPONDENCE_KINDS = (
    "shared_node",
    "shared_edge",
    "omitted_node",
    "boundary_edge",
    "omitted_edge",
)


@dataclass(frozen=True)
class LocalTopology:
    n: int
    edges: Array
    local_node_global: Array
    local_node_center: Array
    local_edge_global: Array
    local_edge_center: Array
    local_edge_nodes: Array
    node_offsets: Array
    edge_offsets: Array
    local_degree: Array
    center_positions: Array
    pair_centers: Array
    shared_node_pair: Array
    shared_node_left: Array
    shared_node_right: Array
    shared_edge_pair: Array
    shared_edge_left: Array
    shared_edge_right: Array
    omitted_node_pair: Array
    omitted_node_left: Array
    boundary_edge_pair: Array
    boundary_edge_left: Array
    omitted_edge_pair: Array
    omitted_edge_left: Array
    graph_node_offsets: Array
    graph_edge_offsets: Array
    # CPU scalar metadata permits pair chunk slices without synchronizing GPU indices.
    correspondence_offsets: tuple[tuple[int, ...], ...]

    @property
    def num_nodes(self) -> int:
        return self.n

    @property
    def num_centers(self) -> int:
        return self.n

    @property
    def num_edges(self) -> int:
        return self.edges.shape[1]

    @property
    def num_local_nodes(self) -> int:
        return self.local_node_global.shape[0]

    @property
    def num_local_edges(self) -> int:
        return self.local_edge_global.shape[0]

    @property
    def num_pairs(self) -> int:
        return self.pair_centers.shape[1]

    @property
    def num_graphs(self) -> int:
        return self.graph_node_offsets.shape[0] - 1

    @property
    def max_local_nodes(self) -> int:
        value = self.node_offsets
        if isinstance(value, torch.Tensor):
            return int((value[1:] - value[:-1]).max().item())
        return int(np.diff(value).max())

    def to(self, device: str | torch.device) -> LocalTopology:
        return LocalTopology(
            **{
                field.name: getattr(self, field.name)
                if field.name in ("n", "correspondence_offsets")
                else torch.as_tensor(getattr(self, field.name), dtype=torch.long, device=device)
                for field in fields(self)
            }
        )


def _array(value: Array) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("topology preparation/batching requires CPU arrays")
        value = value.detach().numpy()
    return np.asarray(value)


def _correspondence_offsets(columns: dict, pairs: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(np.r_[0, np.cumsum(np.bincount(columns[kind + "_pair"], minlength=pairs))])
        for kind in CORRESPONDENCE_KINDS
    )


def build_topology(n: int, edges: Array) -> LocalTopology:
    """Construct every induced S_v={v} union N(v), preserving all original IDs.

    Physical IDs are canonical lexicographically sorted undirected edges. The
    first endpoint is smaller. All occurrences use that same orientation.
    pair_centers[:,p]=(sender,receiver), with both directions of each edge.
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    edge = _array(edges)
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype.kind not in "iu":
        raise ValueError("edges must be integer [2,E]")
    if np.any(edge < 0) or np.any(edge >= n) or np.any(edge[0] == edge[1]):
        raise ValueError("physical edges must be valid, distinct-endpoint node pairs")
    edge = np.sort(edge.astype(np.int64, copy=False), axis=0)
    edge = np.ascontiguousarray(edge[:, np.lexsort((edge[1], edge[0]))])
    if edge.shape[1] > 1 and np.any(np.all(edge[:, 1:] == edge[:, :-1], axis=0)):
        raise ValueError("duplicate physical edges are not allowed")
    neighbors = [set() for _ in range(n)]
    for a, b in edge.T:
        neighbors[a].add(int(b))
        neighbors[b].add(int(a))
    local_nodes = [sorted(neighbors[v] | {v}) for v in range(n)]
    node_offsets = np.r_[0, np.cumsum([len(nodes) for nodes in local_nodes])].astype(np.int64)
    node_maps = [
        {node: int(node_offsets[v] + index) for index, node in enumerate(nodes)}
        for v, nodes in enumerate(local_nodes)
    ]
    local_edges = [[] for _ in range(n)]
    for index, (a, b) in enumerate(edge.T):
        # An edge is in S_v iff v is either endpoint or their common neighbor.
        for center in {int(a), int(b)} | (neighbors[a] & neighbors[b]):
            local_edges[center].append(index)
    edge_offsets = np.r_[0, np.cumsum([len(ids) for ids in local_edges])].astype(np.int64)
    local_node_global = np.asarray(
        [node for nodes in local_nodes for node in nodes], dtype=np.int64
    )
    local_node_center = np.repeat(np.arange(n), np.diff(node_offsets)).astype(np.int64)
    local_edge_global = np.asarray([e for ids in local_edges for e in ids], dtype=np.int64)
    local_edge_center = np.repeat(np.arange(n), np.diff(edge_offsets)).astype(np.int64)
    positions = [
        (node_maps[v][int(edge[0, e])], node_maps[v][int(edge[1, e])])
        for v, ids in enumerate(local_edges)
        for e in ids
    ]
    local_edge_nodes = np.asarray(positions, dtype=np.int64).reshape(-1, 2).T.copy()
    local_degree = np.bincount(
        local_edge_nodes.reshape(-1), minlength=len(local_node_global)
    ).astype(np.int64)
    center_positions = np.asarray([node_maps[v][v] for v in range(n)], dtype=np.int64)
    pair_centers = np.concatenate((edge, edge[::-1]), axis=1)
    edge_maps = [
        {e: int(edge_offsets[v] + index) for index, e in enumerate(ids)}
        for v, ids in enumerate(local_edges)
    ]
    columns = {
        name: []
        for name in (
            "shared_node_pair",
            "shared_node_left",
            "shared_node_right",
            "shared_edge_pair",
            "shared_edge_left",
            "shared_edge_right",
            "omitted_node_pair",
            "omitted_node_left",
            "boundary_edge_pair",
            "boundary_edge_left",
            "omitted_edge_pair",
            "omitted_edge_left",
        )
    }
    for pair, (sender, receiver) in enumerate(pair_centers.T):
        source_nodes, target_nodes = node_maps[sender], node_maps[receiver]
        for node, source_position in source_nodes.items():
            if node in target_nodes:
                columns["shared_node_pair"].append(pair)
                columns["shared_node_left"].append(source_position)
                columns["shared_node_right"].append(target_nodes[node])
            else:
                columns["omitted_node_pair"].append(pair)
                columns["omitted_node_left"].append(source_position)
        for e, source_position in edge_maps[sender].items():
            a, b = edge[:, e]
            inside = int(a in target_nodes) + int(b in target_nodes)
            if inside == 2:
                columns["shared_edge_pair"].append(pair)
                columns["shared_edge_left"].append(source_position)
                columns["shared_edge_right"].append(edge_maps[receiver][e])
            else:
                kind = "boundary" if inside == 1 else "omitted"
                columns[kind + "_edge_pair"].append(pair)
                columns[kind + "_edge_left"].append(source_position)
    columns = {name: np.asarray(values, dtype=np.int64) for name, values in columns.items()}
    return LocalTopology(
        n=int(n),
        edges=edge,
        local_node_global=local_node_global,
        local_node_center=local_node_center,
        local_edge_global=local_edge_global,
        local_edge_center=local_edge_center,
        local_edge_nodes=local_edge_nodes,
        node_offsets=node_offsets,
        edge_offsets=edge_offsets,
        local_degree=local_degree,
        center_positions=center_positions,
        pair_centers=pair_centers,
        graph_node_offsets=np.asarray((0, n), dtype=np.int64),
        graph_edge_offsets=np.asarray((0, edge.shape[1]), dtype=np.int64),
        correspondence_offsets=_correspondence_offsets(columns, pair_centers.shape[1]),
        **columns,
    )


def batch_topologies(topologies: list[LocalTopology]) -> LocalTopology:
    """Disjoint-union complete physical graphs and all their local copies."""
    if not topologies:
        raise ValueError("at least one complete topology is required")
    gathered = {
        field.name: []
        for field in fields(LocalTopology)
        if field.name not in ("n", "correspondence_offsets")
    }
    n_offset = e_offset = node_offset = local_edge_offset = pair_offset = 0
    graph_nodes, graph_edges = [0], [0]
    for top in topologies:
        for field in fields(LocalTopology):
            name = field.name
            if name in ("n", "graph_node_offsets", "graph_edge_offsets", "correspondence_offsets"):
                continue
            value = _array(getattr(top, name))
            if name in (
                "edges",
                "local_node_global",
                "local_node_center",
                "local_edge_center",
                "pair_centers",
            ):
                value = value + n_offset
            elif name == "local_edge_global":
                value = value + e_offset
            elif (
                name in ("local_edge_nodes", "center_positions")
                or name.endswith("_node_left")
                or name.endswith("_node_right")
            ):
                value = value + node_offset
            elif name.endswith("_edge_left") or name.endswith("_edge_right"):
                value = value + local_edge_offset
            elif name.endswith("_pair"):
                value = value + pair_offset
            elif name == "node_offsets":
                value = value[:-1] + node_offset
            elif name == "edge_offsets":
                value = value[:-1] + local_edge_offset
            gathered[name].append(value)
        graph_nodes.extend((_array(top.graph_node_offsets)[1:] + n_offset).tolist())
        graph_edges.extend((_array(top.graph_edge_offsets)[1:] + e_offset).tolist())
        n_offset += top.n
        e_offset += top.num_edges
        node_offset += top.num_local_nodes
        local_edge_offset += top.num_local_edges
        pair_offset += top.num_pairs
    result = {
        name: np.concatenate(
            values, axis=1 if name in ("edges", "local_edge_nodes", "pair_centers") else 0
        )
        for name, values in gathered.items()
        if values
    }
    result["node_offsets"] = np.r_[result["node_offsets"], node_offset]
    result["edge_offsets"] = np.r_[result["edge_offsets"], local_edge_offset]
    result["graph_node_offsets"] = np.asarray(graph_nodes, dtype=np.int64)
    result["graph_edge_offsets"] = np.asarray(graph_edges, dtype=np.int64)
    result["correspondence_offsets"] = _correspondence_offsets(result, pair_offset)
    return LocalTopology(n=n_offset, **result)


__all__ = ["LocalTopology", "build_topology", "batch_topologies"]
