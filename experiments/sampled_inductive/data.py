"""Disjoint contexts with exact once-per-epoch seed coverage on training graphs."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from research.conductance_gat.edge_selection.topology import batch_topologies, build_topology
from research.conductance_gat.v5.sampling import TransductiveGraphSampler


def validate_splits(payload):
    if payload["dataset"] != "ppi":
        raise ValueError("independent-graph study requires official PPI graph splits")
    groups = {s: [int(i) for i in payload["splits"][s]] for s in ("train", "validation", "test")}
    flattened = [i for ids in groups.values() for i in ids]
    if any(not ids for ids in groups.values()) or len(set(flattened)) != len(flattened):
        raise ValueError("train/validation/test graph IDs must be nonempty and disjoint")
    if sorted(flattened) != list(range(len(payload["graphs"]))):
        raise ValueError("every official graph must belong to exactly one split")
    return groups


class GraphContexts(Dataset):
    def __init__(self, payload, ids, *, mode, context_seeds, fanouts, seed):
        self.ids, self.mode, self.context_seeds = list(ids), mode, context_seeds
        self.seed = seed
        self.fanouts = tuple(fanouts)
        self.graphs, self.samplers, self.plans = [], {}, {}
        for index in self.ids:
            row = payload["graphs"][index]
            edges = row["incidence_edge_index"]
            graph = Data(
                x=row["x"],
                y=row["y"],
                incidence_edge_index=edges,
                edge_index=torch.cat((edges, edges.flip(0)), 1),
            )
            self.graphs.append(graph)
        self.counts = [
            math.ceil(g.num_nodes / context_seeds) if mode == "sampled" else 1 for g in self.graphs
        ]
        self.node_counts = [g.num_nodes for g in self.graphs]
        self.edge_counts = [g.incidence_edge_index.shape[1] for g in self.graphs]
        self.cache_epoch, self.orders = None, {}

    def __len__(self):
        return sum(self.counts)

    def __getitem__(self, key):
        epoch, graph_index, ordinal = key
        started = time.perf_counter()
        original = self.graphs[graph_index]
        if self.mode == "full":
            if graph_index not in self.plans:
                self.plans[graph_index] = build_topology(
                    original.num_nodes, original.incidence_edge_index
                )
            graph, plan = original, self.plans[graph_index]
            nodes = seeds = torch.arange(original.num_nodes)
            edge_ids = torch.arange(original.incidence_edge_index.shape[1])
            mask = torch.ones(original.num_nodes, dtype=torch.bool)
            saturated = False
        else:
            if graph_index not in self.samplers:
                self.samplers[graph_index] = TransductiveGraphSampler(
                    original,
                    torch.arange(original.num_nodes),
                    mode="cluster",
                    seed_batch_size=self.context_seeds,
                    fanouts=self.fanouts,
                    model_seed=self.seed,
                )
            sampler = self.samplers[graph_index]
            if self.cache_epoch != epoch:
                self.cache_epoch, self.orders = epoch, {}
            if graph_index not in self.orders:
                generator = torch.Generator().manual_seed(
                    self.seed + epoch * 1_000_003 + self.ids[graph_index] * 97_409
                )
                self.orders[graph_index] = torch.randperm(original.num_nodes, generator=generator)
            order = self.orders[graph_index]
            seeds = order[ordinal * self.context_seeds : (ordinal + 1) * self.context_seeds]
            generator = torch.Generator().manual_seed(
                self.seed + epoch * 1_000_003 + self.ids[graph_index] * 97_409 + ordinal * 65_537
            )
            nodes = sampler._cluster_nodes(seeds, generator)
            graph, edge_ids, observation, _ = sampler._induced_result(nodes, seeds)
            plan = build_topology(graph.num_nodes, graph.incidence_edge_index)
            mask = graph.train_mask
            saturated = observation["cluster_budget_saturated"]
        evidence = {
            "graph_index": graph_index,
            "source_graph_id": self.ids[graph_index],
            "nodes": nodes,
            "edges": edge_ids,
            "seeds": seeds,
            "saturated": saturated,
            "construction_seconds": time.perf_counter() - started,
        }
        return graph, plan, mask, evidence

    @property
    def fanouts(self):
        return self._fanouts

    @fanouts.setter
    def fanouts(self, values):
        self._fanouts = tuple(values)


class EpochBatches:
    def __init__(self, dataset, physical_batch, seed, shuffle):
        self.dataset, self.physical_batch, self.seed, self.shuffle = (
            dataset,
            physical_batch,
            seed,
            shuffle,
        )
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.physical_batch)

    def __iter__(self):
        keys = [
            (self.epoch, g, c) for g, count in enumerate(self.dataset.counts) for c in range(count)
        ]
        if self.shuffle:
            order = torch.randperm(
                len(keys),
                generator=torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch),
            ).tolist()
            keys = [keys[i] for i in order]
        for start in range(0, len(keys), self.physical_batch):
            yield keys[start : start + self.physical_batch]


@dataclass
class ContextBatch:
    graph: object
    topology: object
    seed_mask: torch.Tensor
    observations: list

    def pin_memory(self):
        self.graph.pin_memory()
        self.topology = self.topology.pin_memory()
        self.seed_mask = self.seed_mask.pin_memory()
        return self

    def to(self, device):
        graph = self.graph.to(device, non_blocking=True)
        graph.edge_selection_topology = self.topology.to(device, non_blocking=True)
        graph._v5_num_graphs = self.topology.num_graphs
        return ContextBatch(
            graph,
            graph.edge_selection_topology,
            self.seed_mask.to(device, non_blocking=True),
            self.observations,
        )


def collate_contexts(records):
    graphs, plans, masks, observations = zip(*records, strict=True)
    graph = Batch.from_data_list(list(graphs))
    plan = batch_topologies(list(plans))
    if not torch.equal(graph.incidence_edge_index, plan.incidence_edge_index):
        raise ValueError("disjoint context edge offsets disagree")
    return ContextBatch(graph, plan, torch.cat(masks), list(observations))


class Inputs:
    def __init__(
        self,
        payload,
        mode,
        *,
        physical_batch,
        workers,
        context_seeds,
        fanouts,
        seed,
        evaluation_split="validation",
    ):
        ids = validate_splits(payload)
        if mode not in {"full", "sampled"} or evaluation_split not in {"validation", "test"}:
            raise ValueError("invalid inductive training/evaluation support")
        self.mode, self.physical_batch, self.workers = mode, physical_batch, workers
        self.datasets, self.loaders, self.batchers = {}, {}, {}
        for split in ("train", evaluation_split):
            dataset = GraphContexts(
                payload,
                ids[split],
                mode=mode if split == "train" else "full",
                context_seeds=context_seeds,
                fanouts=fanouts,
                seed=seed,
            )
            dataset.fanouts = fanouts
            batcher = EpochBatches(dataset, physical_batch, seed, split == "train")
            self.datasets[split], self.batchers[split] = dataset, batcher
            self.loaders[split] = DataLoader(
                dataset,
                batch_sampler=batcher,
                collate_fn=collate_contexts,
                num_workers=workers,
                persistent_workers=workers > 0,
                prefetch_factor=2 if workers > 0 else None,
                pin_memory=True,
            )
        self.evaluation_split = evaluation_split

    def close(self):
        # Stop only DataLoader workers created by this instance; never a user session.
        for loader in self.loaders.values():
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None

    def metadata(self):
        return {
            "training_support": self.mode,
            "physical_batch": self.physical_batch,
            "batch_unit": "independent contexts"
            if self.mode == "sampled"
            else "independent graphs",
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "workers": self.workers,
            "persistent_workers": self.workers > 0,
            "pin_memory": True,
            "prefetch_factor": 2 if self.workers > 0 else None,
            "training_nodes": sum(self.datasets["train"].node_counts),
            "training_contexts": len(self.datasets["train"]),
            "batches_per_epoch": len(self.batchers["train"]),
            "context_seeds": self.datasets["train"].context_seeds,
            "fanouts": list(self.datasets["train"].fanouts),
            "sampling_correction": "degree-ratio sqrt product clamped [1,64], not unbiased",
            "graph_ids": {key: ds.ids for key, ds in self.datasets.items()},
            "node_counts": {key: ds.node_counts for key, ds in self.datasets.items()},
            "edge_counts": {key: ds.edge_counts for key, ds in self.datasets.items()},
            "supervision": "every training node exactly once per epoch; context labels excluded",
        }


class Coverage:
    def __init__(self, dataset):
        self.dataset = dataset
        self.seeds = [torch.zeros(n, dtype=torch.int32) for n in dataset.node_counts]
        self.nodes = [torch.zeros(n, dtype=torch.bool) for n in dataset.node_counts]
        self.edges = [torch.zeros(n, dtype=torch.bool) for n in dataset.edge_counts]
        self.node_instances = self.edge_instances = self.contexts = self.saturated = 0
        self.worker_seconds = 0.0

    def add(self, observations):
        for observation in observations:
            i = observation["graph_index"]
            self.seeds[i].index_add_(
                0, observation["seeds"], torch.ones_like(observation["seeds"], dtype=torch.int32)
            )
            self.nodes[i][observation["nodes"]] = True
            self.edges[i][observation["edges"]] = True
            self.node_instances += observation["nodes"].numel()
            self.edge_instances += observation["edges"].numel()
            self.contexts += 1
            self.saturated += observation["saturated"]
            self.worker_seconds += observation["construction_seconds"]

    def finish(self):
        if not all(bool((counts == 1).all()) for counts in self.seeds):
            raise ValueError("sampling lost or repeated supervised training nodes")
        unique_nodes = sum(int(v.sum()) for v in self.nodes)
        unique_edges = sum(int(v.sum()) for v in self.edges)
        return {
            "supervised_nodes": sum(self.dataset.node_counts),
            "seed_coverage": 1.0,
            "contexts": self.contexts,
            "saturated_contexts": self.saturated,
            "node_instances": self.node_instances,
            "edge_instances": self.edge_instances,
            "unique_nodes": unique_nodes,
            "unique_edges": unique_edges,
            "edge_coverage": unique_edges / max(1, sum(self.dataset.edge_counts)),
            "node_repeat_fraction": 1 - unique_nodes / max(1, self.node_instances),
            "edge_repeat_fraction": 1 - unique_edges / max(1, self.edge_instances),
            "worker_construction_seconds_sum": self.worker_seconds,
            "time_note": "parallel worker CPU durations overlap; do not add to wall time",
        }
