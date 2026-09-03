"""Dependency-stable transductive samplers with explicit boundary metadata."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor


def physical_degree(
    incidence: Tensor, num_nodes: int, *, dtype: torch.dtype = torch.float32
) -> Tensor:
    tail, head = incidence
    degree = torch.zeros(num_nodes, dtype=dtype)
    ones = torch.ones(tail.numel(), dtype=dtype)
    return degree.index_add(0, tail.cpu(), ones).index_add(0, head.cpu(), ones)


def csr_values(values: Tensor, rowptr: Tensor, rows: Tensor) -> Tensor:
    """Gather arbitrary one-dimensional CSR values without a Python row loop."""

    if (
        values.dtype != torch.long
        or values.ndim != 1
        or rowptr.dtype != torch.long
        or rowptr.ndim != 1
        or rows.dtype != torch.long
        or rows.ndim != 1
    ):
        raise ValueError("values, rowptr and rows must be int64 CSR tensors")
    if any(value.device.type != "cpu" for value in (values, rowptr, rows)):
        raise ValueError("dependency-free CSR sampling expects CPU tensors")
    if rowptr.numel() < 1 or int(rowptr[-1]) != values.numel():
        raise ValueError("CSR rowptr does not span values")
    if not rows.numel():
        return torch.empty(0, dtype=torch.long)
    if int(rows.min()) < 0 or int(rows.max()) + 1 >= rowptr.numel():
        raise ValueError("CSR row lies outside rowptr")
    starts, stops = rowptr[rows], rowptr[rows + 1]
    lengths = stops - starts
    total = int(lengths.sum())
    if total == 0:
        return torch.empty(0, dtype=torch.long)
    repeated_starts = torch.repeat_interleave(starts, lengths)
    output_row_start = torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
    within_row = torch.arange(total, dtype=torch.long) - output_row_start
    return values[repeated_starts + within_row]


def csr_neighbors(arcs: Tensor, rowptr: Tensor, nodes: Tensor) -> Tensor:
    """Gather CSR neighbor rows in node order without a Python node loop."""

    if arcs.dtype != torch.long or arcs.ndim != 2 or arcs.shape[0] != 2:
        raise ValueError("arcs must be a 2 x E int64 tensor")
    return csr_values(arcs[1], rowptr, nodes)


def physical_edge_id_csr(incidence: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Build a bidirectional node->physical-edge CSR once per full graph."""

    if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("incidence must be a 2 x E int64 tensor")
    if incidence.device.type != "cpu" or num_nodes < 1:
        raise ValueError("physical-edge CSR requires a positive CPU graph")
    if incidence.numel() and (int(incidence.min()) < 0 or int(incidence.max()) >= num_nodes):
        raise ValueError("incidence endpoint lies outside the graph")
    edge_count = incidence.shape[1]
    sources = torch.cat((incidence[0], incidence[1]))
    edge_ids = torch.arange(edge_count, dtype=torch.long).repeat(2)
    order = torch.argsort(sources * num_nodes + torch.cat((incidence[1], incidence[0])))
    sources, edge_ids = sources[order], edge_ids[order]
    counts = torch.bincount(sources, minlength=num_nodes)
    rowptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return edge_ids, rowptr


def induced_physical_edge_ids(
    incidence: Tensor,
    incident_edge_ids: Tensor,
    incident_rowptr: Tensor,
    nodes: Tensor,
    num_nodes: int,
) -> tuple[Tensor, int]:
    """Find induced physical edges from sampled-node incident rows only."""

    nodes = nodes.unique(sorted=True)
    candidates = csr_values(incident_edge_ids, incident_rowptr, nodes)
    candidate_count = int(candidates.numel())
    if not candidate_count:
        return torch.empty(0, dtype=torch.long), 0
    candidates = candidates.unique(sorted=True)
    membership = torch.zeros(num_nodes, dtype=torch.bool)
    membership[nodes] = True
    endpoints = incidence[:, candidates]
    keep = membership[endpoints[0]] & membership[endpoints[1]]
    return candidates[keep], candidate_count


class TransductiveGraphSampler:
    """Yield induced train subgraphs without PyG sampling extensions.

    Neighbor sampling expands a seed batch hop by hop, limiting each hop to an
    average fanout per original seed so deep architectures cannot grow an
    exponential sample. Cluster sampling performs one randomized breadth-first
    expansion to a node budget. Both carry original-graph degree and a bounded
    degree-ratio boundary correction. The correction is metadata, never learned C.
    """

    def __init__(
        self,
        graph,
        train_indices: Tensor,
        *,
        mode: str,
        seed_batch_size: int,
        fanouts: Sequence[int],
        model_seed: int,
    ) -> None:
        if mode not in {"neighbor", "cluster"}:
            raise ValueError("sample mode must be neighbor or cluster")
        if seed_batch_size < 1 or not fanouts or any(value < 1 for value in fanouts):
            raise ValueError("seed batch size and every fanout must be positive")
        self.graph = graph.cpu()
        self.train_indices = train_indices.detach().cpu().long()
        self.mode = mode
        self.seed_batch_size = int(seed_batch_size)
        self.fanouts = tuple(int(value) for value in fanouts)
        self.model_seed = int(model_seed)
        self.num_nodes = int(graph.x.shape[0])
        self.incidence = graph.incidence_edge_index.detach().cpu().long()
        self.incident_edge_ids, self.incident_rowptr = physical_edge_id_csr(
            self.incidence, self.num_nodes
        )
        self.last_induced_candidate_arc_count = 0
        arcs = graph.edge_index.detach().cpu().long()
        order = torch.argsort(arcs[0] * self.num_nodes + arcs[1])
        self.arcs = arcs[:, order]
        counts = torch.bincount(self.arcs[0], minlength=self.num_nodes)
        self.rowptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.full_degree = physical_degree(self.incidence, self.num_nodes)
        log_degree = self.full_degree.log1p()
        edge_count = float(self.incidence.shape[1])
        self.graph_structure = torch.tensor(
            [
                [
                    math.log1p(self.num_nodes),
                    math.log1p(edge_count),
                    float(log_degree.mean()),
                    float(log_degree.std(correction=0)),
                    edge_count / self.num_nodes,
                    2 * edge_count / max(self.num_nodes * (self.num_nodes - 1), 1),
                ]
            ],
            dtype=torch.float32,
        )
        if not self.train_indices.numel():
            raise ValueError("sampler requires nonempty train indices")
        if int(self.train_indices.min()) < 0 or int(self.train_indices.max()) >= self.num_nodes:
            raise ValueError("train index lies outside graph")

    def __len__(self) -> int:
        return math.ceil(self.train_indices.numel() / self.seed_batch_size)

    def _neighbors(self, nodes: Tensor) -> Tensor:
        return csr_neighbors(self.arcs, self.rowptr, nodes)

    @staticmethod
    def _random_limit(values: Tensor, limit: int, generator: torch.Generator) -> Tensor:
        values = values.unique()
        if values.numel() <= limit:
            return values
        return values[torch.randperm(values.numel(), generator=generator)[:limit]]

    def _neighbor_nodes(self, seeds: Tensor, generator: torch.Generator) -> Tensor:
        selected, frontier = seeds.unique(), seeds.unique()
        for fanout in self.fanouts:
            candidates = self._neighbors(frontier)
            if not candidates.numel():
                break
            candidates = self._random_limit(candidates, seeds.numel() * fanout, generator)
            unseen = candidates[~torch.isin(candidates, selected)]
            if not unseen.numel():
                break
            selected = torch.cat((selected, unseen)).unique()
            frontier = unseen
        return selected

    def _cluster_nodes(self, seeds: Tensor, generator: torch.Generator) -> Tensor:
        budget = max(seeds.numel(), seeds.numel() * (1 + sum(self.fanouts)))
        selected, frontier = seeds.unique(), seeds.unique()
        while selected.numel() < budget and frontier.numel():
            candidates = self._neighbors(frontier)
            unseen = candidates[~torch.isin(candidates, selected)]
            unseen = self._random_limit(unseen, budget - selected.numel(), generator)
            if not unseen.numel():
                break
            selected = torch.cat((selected, unseen)).unique()
            frontier = unseen
        return selected

    def _induced(self, nodes: Tensor, seeds: Tensor):
        from torch_geometric.data import Data

        nodes = nodes.unique(sorted=True)
        edge_ids, self.last_induced_candidate_arc_count = induced_physical_edge_ids(
            self.incidence,
            self.incident_edge_ids,
            self.incident_rowptr,
            nodes,
            self.num_nodes,
        )
        global_incidence = self.incidence[:, edge_ids]
        local = torch.full((self.num_nodes,), -1, dtype=torch.long)
        local[nodes] = torch.arange(nodes.numel())
        incidence = local[global_incidence]
        arcs = torch.cat((incidence, incidence.flip(0)), dim=1)
        sample_degree = physical_degree(incidence, nodes.numel())
        full_degree = self.full_degree[nodes]
        if incidence.shape[1]:
            ratio = full_degree / sample_degree.clamp_min(1)
            correction = (ratio[incidence[0]] * ratio[incidence[1]]).sqrt().clamp(1.0, 64.0)
        else:
            correction = torch.empty(0, dtype=torch.float32)
        train_mask = torch.isin(nodes, seeds)
        return Data(
            x=self.graph.x[nodes],
            y=self.graph.y[nodes],
            edge_index=arcs,
            incidence_edge_index=incidence,
            full_degree=full_degree,
            graph_structure=self.graph_structure,
            sampling_correction=correction,
            edge_normalization_weight=correction,
            train_mask=train_mask,
            global_node_id=nodes,
            sample_seed_count=torch.tensor([int(seeds.numel())]),
        )

    def iter_epoch(self, epoch: int) -> Iterator:
        generator = torch.Generator().manual_seed(self.model_seed + 1_000_003 * int(epoch))
        order = self.train_indices[torch.randperm(self.train_indices.numel(), generator=generator)]
        for start in range(0, order.numel(), self.seed_batch_size):
            seeds = order[start : start + self.seed_batch_size]
            nodes = (
                self._neighbor_nodes(seeds, generator)
                if self.mode == "neighbor"
                else self._cluster_nodes(seeds, generator)
            )
            yield self._induced(nodes, seeds)

    def metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "implementation": "dependency_free_induced_subgraph",
            "induced_edge_enumeration": (
                "bidirectional physical-edge CSR; O(sample incident arcs), no per-batch full-E scan"
            ),
            "seed_batch_size": self.seed_batch_size,
            "fanouts": list(self.fanouts),
            "batches_per_epoch": len(self),
            "original_degree_carried": True,
            "boundary_correction": (
                "sqrt(full_degree/sample_degree endpoint product), clamp[1,64]; "
                "explicit approximation, not claimed unbiased"
            ),
            "normalization_importance_weight": "same boundary correction",
            "original_graph_context_carried": True,
            "validation_graph": "complete_official_graph",
        }
