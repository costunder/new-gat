"""Dependency-stable transductive samplers with explicit boundary metadata."""

from __future__ import annotations

import copy
import hashlib
import math
import time
import warnings
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

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
    ``cluster_disjoint`` applies that same cluster rule independently to explicitly
    sized seed contexts and joins their independent copies into one physical batch.
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
        context_seed_batch_size: int | None = None,
        context_workers: int = 1,
    ) -> None:
        if mode not in {"neighbor", "cluster", "cluster_disjoint"}:
            raise ValueError("sample mode must be neighbor, cluster or cluster_disjoint")
        if (
            isinstance(seed_batch_size, bool)
            or not isinstance(seed_batch_size, int)
            or seed_batch_size < 1
            or not fanouts
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in fanouts
            )
        ):
            raise ValueError("seed batch size and every fanout must be positive integers")
        if (
            isinstance(context_workers, bool)
            or not isinstance(context_workers, int)
            or context_workers < 1
        ):
            raise ValueError("context_workers must be a positive integer")
        if mode == "cluster_disjoint":
            if (
                isinstance(context_seed_batch_size, bool)
                or not isinstance(context_seed_batch_size, int)
                or context_seed_batch_size < 1
            ):
                raise ValueError("cluster_disjoint requires positive context_seed_batch_size")
        elif context_seed_batch_size is not None or context_workers != 1:
            raise ValueError("context options apply only to cluster_disjoint")
        self.graph = graph.cpu()
        self.train_indices = train_indices.detach().cpu().long()
        self.mode = mode
        self.seed_batch_size = int(seed_batch_size)
        self.fanouts = tuple(int(value) for value in fanouts)
        self.model_seed = int(model_seed)
        self.context_seed_batch_size = (
            int(context_seed_batch_size) if context_seed_batch_size is not None else None
        )
        self.context_workers = int(context_workers)
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
        self._component_labels = None
        self._component_nodes = None
        self._component_rowptr = None
        self._component_cache_status = "not_built"
        self.last_sample_observation = None
        if (
            self.mode == "cluster"
            and self.seed_batch_size * (1 + sum(self.fanouts)) >= self.num_nodes
        ):
            warnings.warn(
                "V5 cluster node budget reaches the full graph: saturated seed batches "
                "select complete reachable components, not small local subgraphs. "
                "Sampling law, graph size and physical seed batch are unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self.train_indices.numel():
            raise ValueError("sampler requires nonempty train indices")
        if int(self.train_indices.min()) < 0 or int(self.train_indices.max()) >= self.num_nodes:
            raise ValueError("train index lies outside graph")
        if self.mode == "cluster_disjoint":
            if self.train_indices.ndim != 1:
                raise ValueError("cluster_disjoint train indices must be one-dimensional")
            if self.train_indices.unique().numel() != self.train_indices.numel():
                raise ValueError("cluster_disjoint requires unique supervised train indices")
            if (
                self.seed_batch_size < self.train_indices.numel()
                and self.seed_batch_size % self.context_seed_batch_size
            ):
                raise ValueError(
                    "physical seed batch must group whole sampling contexts "
                    "(integer multiple of context_seed_batch_size), except a batch "
                    "containing the complete train split"
                )
            if self.context_seed_batch_size * (1 + sum(self.fanouts)) >= self.num_nodes:
                warnings.warn(
                    "V5 cluster_disjoint context budget reaches the full graph: "
                    "separating physical batching does not prevent saturated contexts. "
                    "The explicit context recipe is retained and saturation is recorded.",
                    RuntimeWarning,
                    stacklevel=2,
                )

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
        if budget >= self.num_nodes:
            selected = self._complete_seed_components(seeds)
            if selected is not None:
                return selected
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

    def _complete_seed_components(self, seeds: Tensor) -> Tensor | None:
        """Exactly replace saturated BFS, which makes no random draws.

        SciPy belongs to the existing paper dependencies. Base-only installs
        and directed inputs retain the original BFS with an observable reason.
        """
        if seeds.dtype != torch.long or seeds.ndim != 1 or seeds.device.type != "cpu":
            raise ValueError("component seeds must be a one-dimensional CPU int64 tensor")
        if seeds.numel() and (int(seeds.min()) < 0 or int(seeds.max()) >= self.num_nodes):
            raise ValueError("component seed lies outside the graph")
        if self._component_cache_status == "not_built":
            try:
                import numpy as np
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
            except ImportError:
                self._component_cache_status = "original_bfs_scipy_unavailable"
                return None
            adjacency = csr_matrix(
                (
                    np.ones(self.arcs.shape[1], dtype=np.int8),
                    self.arcs[1].numpy(),
                    self.rowptr.numpy(),
                ),
                shape=(self.num_nodes, self.num_nodes),
            )
            if (adjacency != adjacency.transpose()).nnz:
                self._component_cache_status = "original_bfs_asymmetric_adjacency"
                return None
            count, labels = connected_components(adjacency, directed=False)
            self._component_labels = torch.from_numpy(labels.astype(np.int64, copy=False))
            self._component_nodes = torch.argsort(self._component_labels, stable=True)
            sizes = torch.bincount(self._component_labels, minlength=count)
            self._component_rowptr = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
            self._component_cache_status = "immutable_undirected_component_csr"
        if self._component_labels is None:
            return None
        components = self._component_labels[seeds].unique(sorted=True)
        # Preserve exactly the old torch.unique BFS ordering, without RNG use.
        return csr_values(self._component_nodes, self._component_rowptr, components).sort().values

    def _induced_result(self, nodes: Tensor, seeds: Tensor):
        """Construct one context without mutating shared sampler observations."""
        from torch_geometric.data import Data

        nodes = nodes.unique(sorted=True)
        edge_ids, candidate_count = induced_physical_edge_ids(
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
        observation = {
            "supervised_seed_nodes": int(seeds.numel()),
            "sampled_nodes": int(nodes.numel()),
            "sampled_physical_edges": int(incidence.shape[1]),
            "original_nodes": self.num_nodes,
            "original_physical_edges": int(self.incidence.shape[1]),
            "configured_cluster_node_budget": (
                int(seeds.numel()) * (1 + sum(self.fanouts))
                if self.mode in {"cluster", "cluster_disjoint"}
                else None
            ),
            "cluster_budget_saturated": (
                self.mode in {"cluster", "cluster_disjoint"}
                and seeds.numel() * (1 + sum(self.fanouts)) >= self.num_nodes
            ),
            "component_cache": self._component_cache_status,
        }
        sampled = Data(
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
        relation = getattr(self.graph, "edge_relation_id", None)
        if relation is not None:
            if relation.dtype != torch.long or relation.shape != (self.incidence.shape[1],):
                raise ValueError("edge_relation_id must align with physical incidence columns")
            sampled.edge_relation_id = relation[edge_ids]
        node_type = getattr(self.graph, "node_type", None)
        if node_type is not None:
            if node_type.dtype != torch.long or node_type.shape != (self.num_nodes,):
                raise ValueError("node_type must align with original nodes")
            sampled.node_type = node_type[nodes]
        return sampled, edge_ids, observation, candidate_count

    def _induced(self, nodes: Tensor, seeds: Tensor):
        sampled, _, observation, candidate_count = self._induced_result(nodes, seeds)
        self.last_induced_candidate_arc_count = candidate_count
        self.last_sample_observation = copy.deepcopy(observation)
        sampled.sampling_observation = copy.deepcopy(observation)
        return sampled

    @staticmethod
    def _id_hash(ids: Tensor) -> str:
        # CPU sampling metadata only; IDs never enter the history JSON itself.
        return hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()

    @staticmethod
    def _overlap(left: Tensor, right: Tensor) -> dict[str, object]:
        # Every caller supplies sorted unique original IDs. Reuse this ordering
        # instead of sorting the same context repeatedly inside torch.isin for
        # every pair. Search the shorter vector; retain ALL context comparisons.
        if left.numel() > right.numel():
            left, right = right, left
        if left.numel():
            positions = torch.searchsorted(right, left)
            inside = positions < right.numel()
            intersection = int((right[positions[inside]] == left[inside]).sum())
        else:
            intersection = 0
        union = int(left.numel() + right.numel()) - intersection
        return {
            "intersection": intersection,
            "union": union,
            "jaccard": intersection / union if union else 1.0,
            "both_empty": union == 0,
        }

    def _disjoint_context(self, specification):
        seeds, rng_seed = specification
        generator = torch.Generator().manual_seed(rng_seed)
        nodes = self._cluster_nodes(seeds, generator)
        return self._induced_result(nodes, seeds)

    def _iter_disjoint(self, order: Tensor, epoch: int) -> Iterator:
        from torch_geometric.data import Batch

        context_size = self.context_seed_batch_size
        # Initialize the optional immutable component cache BEFORE worker threads.
        # This is only the existing saturation shortcut, not another sampling law.
        if min(context_size, self.seed_batch_size) * (1 + sum(self.fanouts)) >= self.num_nodes:
            self._complete_seed_components(order[:1])
        previous_nodes = previous_edges = None
        seen_seeds = torch.zeros(self.num_nodes, dtype=torch.bool)
        context_ordinal = 0
        executor = (
            ThreadPoolExecutor(max_workers=self.context_workers)
            if self.context_workers > 1
            else None
        )
        try:
            for batch_index, start in enumerate(range(0, order.numel(), self.seed_batch_size)):
                seeds = order[start : start + self.seed_batch_size]
                specifications = []
                for context_start in range(0, seeds.numel(), context_size):
                    context_seeds = seeds[context_start : context_start + context_size]
                    # A private generator per context makes worker scheduling irrelevant
                    # and never touches the process-global torch RNG.
                    rng_seed = (
                        self.model_seed + 1_000_003 * epoch + 97_409 * (context_ordinal + 1)
                    ) % (2**63 - 1)
                    specifications.append((context_seeds, rng_seed))
                    context_ordinal += 1
                results = list(
                    executor.map(self._disjoint_context, specifications)
                    if executor is not None
                    else map(self._disjoint_context, specifications)
                )
                contexts = [item[0] for item in results]
                node_sets = [graph.global_node_id for graph in contexts]
                edge_sets = [item[1] for item in results]
                batch = Batch.from_data_list(contexts)
                batch._v5_num_graphs = len(contexts)
                # No offset: these identify original physical edges, unlike local
                # incidence_edge_index, which PyG offsets into the disjoint union.
                batch.global_physical_edge_id = torch.cat(edge_sets)
                supervised = batch.global_node_id[batch.train_mask]
                if (
                    supervised.numel() != seeds.numel()
                    or supervised.unique().numel() != seeds.numel()
                    or not torch.equal(supervised.sort().values, seeds.sort().values)
                    or bool(seen_seeds[supervised].any())
                ):
                    raise RuntimeError("disjoint sampling duplicated or lost supervised seeds")
                seen_seeds[supervised] = True
                observation_started = time.perf_counter()
                unique_nodes = torch.cat(node_sets).unique(sorted=True)
                unique_edges = torch.cat(edge_sets).unique(sorted=True)
                pairwise = []
                for i in range(len(contexts)):
                    for j in range(i + 1, len(contexts)):
                        pairwise.append(
                            {
                                "contexts": [i, j],
                                "nodes": self._overlap(node_sets[i], node_sets[j]),
                                "physical_edges": self._overlap(edge_sets[i], edge_sets[j]),
                            }
                        )
                context_observations = []
                for index, (_, edges, observation, candidate_count) in enumerate(results):
                    context_observations.append(
                        {
                            **observation,
                            "context_index": index,
                            "node_ids_sha256": self._id_hash(node_sets[index]),
                            "physical_edge_ids_sha256": self._id_hash(edges),
                            "induced_candidate_arcs": candidate_count,
                            "node_coverage_fraction": node_sets[index].numel() / self.num_nodes,
                            "selects_all_original_nodes": node_sets[index].numel()
                            == self.num_nodes,
                        }
                    )
                observation = {
                    "mode": self.mode,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "physical_seed_batch_size": self.seed_batch_size,
                    "context_seed_batch_size": context_size,
                    "context_count": len(contexts),
                    "supervised_seed_nodes": int(seeds.numel()),
                    "unique_supervised_seed_nodes": int(supervised.unique().numel()),
                    "epoch_supervised_seeds_so_far": int(seen_seeds.sum()),
                    "epoch_total_supervised_seeds": int(self.train_indices.numel()),
                    "epoch_seed_coverage_fraction": float(seen_seeds.sum())
                    / self.train_indices.numel(),
                    "sampled_nodes": int(batch.x.shape[0]),
                    "sampled_physical_edges": int(batch.incidence_edge_index.shape[1]),
                    "unique_original_nodes": int(unique_nodes.numel()),
                    "unique_original_physical_edges": int(unique_edges.numel()),
                    "duplicated_context_node_copies": int(batch.x.shape[0] - unique_nodes.numel()),
                    "original_nodes": self.num_nodes,
                    "original_physical_edges": int(self.incidence.shape[1]),
                    "node_ids_sha256": self._id_hash(unique_nodes),
                    "physical_edge_ids_sha256": self._id_hash(unique_edges),
                    "cluster_budget_saturated": any(
                        item["cluster_budget_saturated"] for item in context_observations
                    ),
                    "saturated_context_count": sum(
                        item["cluster_budget_saturated"] for item in context_observations
                    ),
                    "contexts": context_observations,
                    "within_batch_context_overlap": pairwise,
                    "previous_physical_batch_overlap": None
                    if previous_nodes is None
                    else {
                        "nodes": self._overlap(previous_nodes, unique_nodes),
                        "physical_edges": self._overlap(previous_edges, unique_edges),
                    },
                }
                observation["observation_cpu_wall_seconds"] = (
                    time.perf_counter() - observation_started
                )
                observation["observation_timing_scope"] = (
                    "unique ID unions, all-pair overlap, hashes and observation assembly; "
                    "excludes context sampling, batch collation and snapshot copies"
                )
                previous_nodes, previous_edges = unique_nodes, unique_edges
                self.last_induced_candidate_arc_count = sum(item[3] for item in results)
                self.last_sample_observation = copy.deepcopy(observation)
                # Immutable-by-convention snapshot travels with the batch even if
                # prefetch advances the sampler before this batch is consumed.
                batch.sampling_observation = copy.deepcopy(observation)
                yield batch
            if not bool(seen_seeds[self.train_indices].all()):
                raise RuntimeError("disjoint sampling did not supervise every training seed")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def iter_epoch(self, epoch: int) -> Iterator:
        generator = torch.Generator().manual_seed(self.model_seed + 1_000_003 * int(epoch))
        order = self.train_indices[torch.randperm(self.train_indices.numel(), generator=generator)]
        if self.mode == "cluster_disjoint":
            yield from self._iter_disjoint(order, int(epoch))
            return
        for start in range(0, order.numel(), self.seed_batch_size):
            seeds = order[start : start + self.seed_batch_size]
            nodes = (
                self._neighbor_nodes(seeds, generator)
                if self.mode == "neighbor"
                else self._cluster_nodes(seeds, generator)
            )
            yield self._induced(nodes, seeds)

    def metadata(self) -> dict[str, object]:
        metadata = {
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
            "cluster_saturation": {
                "configured_node_budget": self.seed_batch_size * (1 + sum(self.fanouts)),
                "full_seed_batch_reaches_graph_size": (
                    self.mode == "cluster"
                    and self.seed_batch_size * (1 + sum(self.fanouts)) >= self.num_nodes
                ),
                "policy": "complete reachable seed components; original law unchanged",
                "component_cache": self._component_cache_status,
            },
        }
        if self.mode == "cluster_disjoint":
            context_budget = self.context_seed_batch_size * (1 + sum(self.fanouts))
            metadata.update(
                {
                    "implementation": "dependency_free_cluster_contexts_pyg_disjoint_union",
                    "context_seed_batch_size": self.context_seed_batch_size,
                    "context_workers": self.context_workers,
                    "contexts_per_full_physical_batch": math.ceil(
                        self.seed_batch_size / self.context_seed_batch_size
                    ),
                    "context_sampling": (
                        "independent randomized cluster BFS using unchanged fanouts per context"
                    ),
                    "context_rng": (
                        "private generator keyed by model_seed, epoch and context ordinal"
                    ),
                    "optimization": (
                        "one loss/backward/optimizer step per physical seed batch; "
                        "every training seed once per epoch"
                    ),
                    "context_node_copies": (
                        "independent disjoint copies; "
                        "no cross-context message passing or graph statistics"
                    ),
                    "cluster_saturation": {
                        "configured_node_budget": context_budget,
                        "full_seed_batch_reaches_graph_size": context_budget >= self.num_nodes,
                        "policy": (
                            "per-context reachable components; "
                            "explicit context recipe never auto-shrunk"
                        ),
                        "component_cache": self._component_cache_status,
                    },
                }
            )
        return metadata
