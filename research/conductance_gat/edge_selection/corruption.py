"""Exact controlled non-edge corruption with trainer-only immutable provenance.

Uniform sampling uses compressed complement ranks, not rejection loops or a
dense N x N complement. Memory is O(N+E+excluded+requested). Candidate edges are
canonically sorted so append position never reveals which edges were inserted.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .topology import TopologyPlan, build_topology, topology_fingerprint, validate_cpu_topology


def _tensor_sha(value: Tensor) -> str:
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
    digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def canonical_incidence(incidence: Tensor) -> tuple[Tensor, Tensor]:
    """Canonical undirected endpoint order and lexicographic column permutation."""
    if (
        not isinstance(incidence, Tensor)
        or incidence.ndim != 2
        or incidence.device.type != "cpu"
        or incidence.dtype != torch.long
        or incidence.shape[0] != 2
    ):
        raise ValueError("canonical incidence requires CPU int64 2 x E")
    values = incidence.numpy()
    left, right = np.minimum(values[0], values[1]), np.maximum(values[0], values[1])
    order = np.lexsort((right, left))
    return torch.from_numpy(np.stack((left[order], right[order]))), torch.from_numpy(order.copy())


def _complement_space(num_nodes, incidence, node_graph, exclude, original_plan):
    edge, graph = validate_cpu_topology(num_nodes, incidence, node_graph)
    node_graph = torch.from_numpy(graph.copy())
    if original_plan is None:
        original_plan = build_topology(num_nodes, incidence, node_graph)
    if (
        original_plan.incidence_edge_index.device.type != "cpu"
        or original_plan.metadata["topology_sha256"]
        != topology_fingerprint(num_nodes, incidence, node_graph)
        or not torch.equal(original_plan.incidence_edge_index, incidence)
        or not torch.equal(original_plan.node_graph, node_graph)
    ):
        raise ValueError("cached original topology does not match the supplied graph")
    excluded = torch.empty((2, 0), dtype=torch.long) if exclude is None else exclude
    excluded_edge, _ = validate_cpu_topology(num_nodes, excluded, node_graph)
    # Canonical component order by smallest original node, independent of DFS
    # forest seed. Never expose these original-component labels as model features.
    groups = {}
    for node, component in enumerate(original_plan.components.tolist()):
        groups.setdefault(component, []).append(node)
    sizes = np.array([len(nodes) for nodes in groups.values()], dtype=np.int64)
    component_of = np.empty(num_nodes, dtype=np.int64)
    local_position = np.empty(num_nodes, dtype=np.int64)
    members = []
    for component, nodes in enumerate(groups.values()):
        component_of[nodes] = component
        local_position[nodes] = np.arange(len(nodes))
        members.extend(nodes)
    members = np.asarray(members, dtype=np.int64)
    if excluded_edge.size and np.any(
        component_of[excluded_edge[0]] != component_of[excluded_edge[1]]
    ):
        raise ValueError("excluded negatives cross original connected components")
    possibilities = [int(size) * (int(size) - 1) // 2 for size in sizes]
    if sum(possibilities) > np.iinfo(np.int64).max:
        raise ValueError("component pair space exceeds supported exact int64 rank indexing")
    offsets = np.concatenate(([0], np.cumsum(possibilities, dtype=np.int64)))
    node_offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))

    def ranks(edges):
        component = component_of[edges[0]]
        first = np.minimum(local_position[edges[0]], local_position[edges[1]])
        second = np.maximum(local_position[edges[0]], local_position[edges[1]])
        return (
            offsets[component]
            + first * (2 * sizes[component] - first - 1) // 2
            + second
            - first
            - 1
        )

    original_ranks, excluded_ranks = ranks(edge), ranks(excluded_edge)
    if np.intersect1d(original_ranks, excluded_ranks).size:
        raise ValueError("excluded negatives contain an original edge")
    forbidden = np.sort(np.concatenate((original_ranks, excluded_ranks)))
    return sizes, offsets, node_offsets, members, forbidden


def sample_nonedges(
    num_nodes: int,
    original_incidence_edge_index: Tensor,
    *,
    count: int,
    seed: int,
    node_graph: Tensor | None = None,
    exclude: Tensor | None = None,
    original_plan: TopologyPlan | None = None,
) -> Tensor:
    """Sample exactly count distinct original-component nonedges, excluding train negatives.

    Count is global over the union of eligible within-component pairs. Sampling
    is uniform without replacement over that union; no per-component quota or
    independence claim is silently introduced. Impossible requests fail.
    """
    if type(count) is not int or count < 0 or type(seed) is not int or seed < 0:
        raise ValueError("count and seed must be nonnegative integers")
    if count == 0 and (
        exclude is None
        or (isinstance(exclude, Tensor) and exclude.ndim == 2 and exclude.shape == (2, 0))
    ):
        _, graph = validate_cpu_topology(num_nodes, original_incidence_edge_index, node_graph)
        if exclude is not None:
            validate_cpu_topology(num_nodes, exclude, node_graph)
        if original_plan is not None and (
            original_plan.metadata["topology_sha256"]
            != topology_fingerprint(
                num_nodes, original_incidence_edge_index, torch.from_numpy(graph)
            )
        ):
            raise ValueError("cached original topology does not match the supplied graph")
        # Exactly the caller's explicit zero request, not an error/OOM fallback.
        return torch.empty((2, 0), dtype=torch.long)
    sizes, offsets, node_offsets, members, forbidden = _complement_space(
        num_nodes, original_incidence_edge_index, node_graph, exclude, original_plan
    )
    available = int(offsets[-1]) - len(forbidden)
    if count > available:
        raise ValueError(f"requested {count} nonedges but only {available} eligible pairs remain")
    # random.sample(range(...)) does not enumerate a large dense pair space.
    compressed = np.asarray(random.Random(seed).sample(range(available), count), dtype=np.int64)
    adjusted_forbidden = forbidden - np.arange(len(forbidden), dtype=np.int64)
    rank = compressed + np.searchsorted(adjusted_forbidden, compressed, side="right")
    component = np.searchsorted(offsets[1:], rank, side="right")
    local_rank = rank - offsets[component]
    size = sizes[component]
    # Integer binary search avoids floating inverse-triangular rounding errors.
    lower, upper = np.zeros(count, dtype=np.int64), size - 1
    while np.any(lower + 1 < upper):
        middle = (lower + upper) // 2
        prefix = middle * (2 * size - middle - 1) // 2
        left = prefix <= local_rank
        lower = np.where(left, middle, lower)
        upper = np.where(left, upper, middle)
    first = lower
    second = first + 1 + local_rank - first * (2 * size - first - 1) // 2
    pairs = torch.from_numpy(
        np.stack(
            (members[node_offsets[component] + first], members[node_offsets[component] + second])
        )
    )
    return canonical_incidence(pairs)[0]


@dataclass(frozen=True)
class CorruptionProvenance:
    split: str
    seed: int
    requested_count: int
    original_edge_count: int
    inserted_edge_count: int
    excluded_negative_count: int
    original_topology_sha256: str
    candidate_incidence_sha256: str
    negative_incidence_sha256: str
    excluded_incidence_sha256: str
    edge_targets_sha256: str
    sampler: str = "uniform exact compressed complement ranks within original components"
    model_feature_policy: str = "origin labels, excluded edges and provenance are trainer-only"


@dataclass(frozen=True)
class CorruptionData:
    candidate_incidence: Tensor
    negative_incidence: Tensor
    edge_targets: Tensor
    provenance: CorruptionProvenance

    def model_input(self) -> dict[str, Tensor]:
        """Whitelist-only topology copy: never return labels or provenance."""
        return {"incidence_edge_index": self.candidate_incidence.clone()}

    def verify_unchanged(self) -> None:
        for tensor, expected in (
            (self.candidate_incidence, self.provenance.candidate_incidence_sha256),
            (self.negative_incidence, self.provenance.negative_incidence_sha256),
            (self.edge_targets, self.provenance.edge_targets_sha256),
        ):
            if _tensor_sha(tensor) != expected:
                raise ValueError("controlled corruption artifact changed after provenance creation")


def build_corruption(
    num_nodes: int,
    original_incidence_edge_index: Tensor,
    *,
    count: int,
    seed: int,
    split: str,
    node_graph: Tensor | None = None,
    exclude: Tensor | None = None,
    original_plan: TopologyPlan | None = None,
) -> CorruptionData:
    """Create sorted candidates plus separate trainer targets/provenance.

    Validation/test explicitly require the excluded training-negative tensor,
    including an empty 2x0 tensor when training had no inserted negatives.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if split != "train" and exclude is None:
        raise ValueError("evaluation corruption requires explicit excluded training negatives")
    negative = sample_nonedges(
        num_nodes,
        original_incidence_edge_index,
        count=count,
        seed=seed,
        node_graph=node_graph,
        exclude=exclude,
        original_plan=original_plan,
    )
    candidate, order = canonical_incidence(
        torch.cat((original_incidence_edge_index, negative), dim=1)
    )
    targets = torch.cat((torch.ones(original_incidence_edge_index.shape[1]), torch.zeros(count)))[
        order
    ]
    graph = torch.zeros(num_nodes, dtype=torch.long) if node_graph is None else node_graph
    excluded = torch.empty((2, 0), dtype=torch.long) if exclude is None else exclude
    provenance = CorruptionProvenance(
        split=split,
        seed=seed,
        requested_count=count,
        original_edge_count=original_incidence_edge_index.shape[1],
        inserted_edge_count=count,
        excluded_negative_count=excluded.shape[1],
        original_topology_sha256=topology_fingerprint(
            num_nodes, original_incidence_edge_index, graph
        ),
        candidate_incidence_sha256=_tensor_sha(candidate),
        negative_incidence_sha256=_tensor_sha(negative),
        excluded_incidence_sha256=_tensor_sha(excluded),
        edge_targets_sha256=_tensor_sha(targets),
    )
    return CorruptionData(candidate, negative, targets, provenance)
