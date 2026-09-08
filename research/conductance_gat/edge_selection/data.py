"""Full official data with label-free topology plans and trainer-only corruption targets."""

from __future__ import annotations

import copy
import hashlib
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import torch

from ..v5.sampling import TransductiveGraphSampler
from .corruption import CorruptionData, build_corruption, canonical_incidence, sample_nonedges
from .topology import batch_topologies, build_topology, validate_cpu_topology


def tensor_digest(value):
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def prepare_controlled_corruption(specification) -> CorruptionData:
    """Label/origin-blind geometry preparation, separately testable without PyG."""
    row, graph_id, split, ratio, corruption_seed, forest_seed = specification
    typed = [
        name
        for name in (
            "node_type",
            "node_types",
            "node_type_id",
            "edge_relation_id",
            "edge_type",
            "edge_types",
            "edge_relation",
            "relation_id",
            "relation_type",
            "relation_types",
            "edge_attr",
        )
        if row.get(name) is not None
    ]
    if typed or row.get("num_relations", 0) not in (None, 0):
        raise ValueError(f"typed/edge-feature graph rows are unsupported; cannot discard {typed}")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio < 0
    ):
        raise ValueError("corruption ratio must be finite and nonnegative")
    if type(corruption_seed) is not int or corruption_seed < 0:
        raise ValueError("corruption_seed must be a nonnegative integer")
    original = row["incidence_edge_index"]
    nodes = row["x"].shape[0]
    validate_cpu_topology(nodes, original)
    original = canonical_incidence(original)[0]
    count = math.floor(original.shape[1] * ratio)
    seed = (corruption_seed + 1_000_003 * (graph_id + 1)) % (2**63 - 1)
    # Reuse original-component preprocessing across train/evaluation negative sets.
    original_plan = build_topology(nodes, original, forest_seed=forest_seed) if count else None
    training_negative = (
        None
        if split == "train"
        else sample_nonedges(nodes, original, count=count, seed=seed, original_plan=original_plan)
    )
    return build_corruption(
        nodes,
        original,
        count=count,
        seed=seed if split == "train" else (seed + 97_409) % (2**63 - 1),
        split=split,
        exclude=training_negative,
        original_plan=original_plan,
    )


@dataclass(frozen=True)
class RecordEvidence:
    graph_id: int
    corruption: CorruptionData

    @property
    def split(self):
        return self.corruption.provenance.split

    def verify_unchanged(self, graph=None, topology=None, targets=None):
        self.corruption.verify_unchanged()
        if graph is not None and not torch.equal(
            graph.incidence_edge_index, self.corruption.candidate_incidence
        ):
            raise ValueError("model candidate topology differs from frozen corruption provenance")
        if topology is not None and not torch.equal(
            topology.incidence_edge_index, self.corruption.candidate_incidence
        ):
            raise ValueError("forest plan differs from frozen candidate topology")
        if targets is not None and not torch.equal(targets, self.corruption.edge_targets):
            raise ValueError("trainer origin targets differ from frozen corruption provenance")

    def json_copy(self):
        self.verify_unchanged()
        provenance = self.corruption.provenance
        return {
            "graph_id": self.graph_id,
            "split": provenance.split,
            "original_edges": provenance.original_edge_count,
            "candidate_edges": provenance.original_edge_count + provenance.inserted_edge_count,
            "added_edges": provenance.inserted_edge_count,
            "origin_not_a_model_input": True,
            "corruption": asdict(provenance),
        }


def prepare_record(specification):
    """CPU preprocessing once per official graph/split; no label is inspected."""
    corruption = prepare_controlled_corruption(specification)
    from torch_geometric.data import Data

    row, graph_id, _, _, _, forest_seed = specification
    candidate, target = corruption.candidate_incidence, corruption.edge_targets
    graph = Data(
        x=row["x"],
        y=row["y"],
        incidence_edge_index=candidate,
        edge_index=torch.cat((candidate, candidate.flip(0)), dim=1),
    )
    plan = build_topology(row["x"].shape[0], candidate, forest_seed=forest_seed)
    evidence = RecordEvidence(graph_id, corruption)
    evidence.verify_unchanged(graph, plan, target)
    return graph, plan, target, evidence


@dataclass
class SelectionBatch:
    graph: object
    topology: object
    origin_targets: torch.Tensor
    selected_indices: torch.Tensor | None = None

    def to(self, device, *, non_blocking=False):
        graph = self.graph.clone().to(device, non_blocking=non_blocking)
        graph.edge_selection_topology = self.topology.to(device, non_blocking=non_blocking)
        graph._v5_num_graphs = int(getattr(self.topology, "num_graphs", 1))
        selected = (
            None
            if self.selected_indices is None
            else self.selected_indices.to(device, non_blocking=non_blocking, copy=True)
        )
        return SelectionBatch(
            graph,
            graph.edge_selection_topology,
            self.origin_targets.to(device, non_blocking=non_blocking, copy=True),
            selected,
        )

    def pin_memory(self):
        self.graph.pin_memory()
        self.topology = self.topology.pin_memory()
        self.origin_targets = self.origin_targets.pin_memory()
        if self.selected_indices is not None:
            self.selected_indices = self.selected_indices.pin_memory()
        return self


def collate_records(records):
    from torch_geometric.data import Batch

    graphs, plans, targets, evidences = zip(*records, strict=True)
    for graph, plan, target, evidence in zip(graphs, plans, targets, evidences, strict=True):
        evidence.verify_unchanged(graph, plan, target)
    graph = Batch.from_data_list(list(graphs))
    plan = batch_topologies(list(plans))
    if not torch.equal(graph.incidence_edge_index, plan.incidence_edge_index):
        raise ValueError("batched physical incidence and forest plan are misaligned")
    graph._v5_num_graphs = len(graphs)
    return SelectionBatch(graph, plan, torch.cat(targets))


class PreparedInputs:
    """Static CPU plans are reused; PPI batches merge plans without recomputing DFS."""

    def __init__(self, payload, args):
        from torch.utils.data import DataLoader

        self.args = args
        self.sampler = None
        self.indices = None
        self._records = ()
        self.plan_preparation_seconds = 0.0
        self._device_validation = None
        self._device_training = None
        started = time.perf_counter()
        specifications = []
        for split in ("train", "validation"):
            ids = payload["splits"][split] if args.dataset == "ppi" else [0]
            for graph_id in ids:
                specifications.append(
                    (
                        payload["graphs"][int(graph_id)],
                        int(graph_id),
                        split,
                        args.corruption_ratio,
                        args.corruption_seed,
                        args.forest_seed,
                    )
                )
        if args.dataset == "ppi" and args.workers > 0:
            # Independent graph preparation, on the explicitly configured CPU allocation.
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                records = list(pool.map(prepare_record, specifications))
        else:
            records = [prepare_record(specification) for specification in specifications]
        self._records = tuple(records)
        split_records = {"train": [], "validation": []}
        for record in records:
            split_records[record[3].split].append(record)
        if args.dataset == "ppi":
            self.data = {}
            for split, items in split_records.items():
                self.data[split] = DataLoader(
                    items,
                    batch_size=args.batch_size,
                    shuffle=split == "train",
                    collate_fn=collate_records,
                    num_workers=args.workers,
                    generator=torch.Generator().manual_seed(args.model_seed),
                    pin_memory=args.pin_memory,
                    persistent_workers=args.workers > 0,
                    prefetch_factor=2 if args.workers > 0 else None,
                )
            self.train_count = len(split_records["train"])
            self.validation_count = len(split_records["validation"])
        else:
            self.indices = {
                split: payload["splits"][split].nonzero(as_tuple=False).flatten().long()
                for split in ("train", "validation")
            }
            graph, plan, target, _ = split_records["train"][0]
            self.data = graph
            self.training_record = SelectionBatch(graph, plan, target, self.indices["train"])
            val_graph, val_plan, val_target, _ = split_records["validation"][0]
            self.validation_record = SelectionBatch(
                val_graph, val_plan, val_target, self.indices["validation"]
            )
            if args.sampling != "full":
                self.sampler = TransductiveGraphSampler(
                    graph,
                    self.indices["train"],
                    mode=args.sampling,
                    seed_batch_size=args.sample_seed_batch_size,
                    fanouts=args.num_neighbors,
                    model_seed=args.model_seed,
                    **(
                        {
                            "context_seed_batch_size": args.sample_context_seed_batch_size,
                            "context_workers": args.sample_context_workers,
                        }
                        if args.sampling == "cluster_disjoint"
                        else {}
                    ),
                )
            self.train_count = self.indices["train"].numel()
            self.validation_count = self.indices["validation"].numel()
        self.plan_preparation_seconds = time.perf_counter() - started

    def verify_provenance(self):
        for graph, plan, target, evidence in self._records:
            evidence.verify_unchanged(graph, plan, target)

    @property
    def provenance(self):
        self.verify_provenance()
        return [record[3].json_copy() for record in self._records]

    def metadata(self):
        return {
            "provenance": self.provenance,
            "topology_preparation_seconds": self.plan_preparation_seconds,
            "preparation_workers": self.args.workers if self.args.dataset == "ppi" else 1,
            "preparation_scope": (
                "static CPU DFS per graph; merge cached plans for disjoint batches"
            ),
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "sampling": self.sampler.metadata() if self.sampler is not None else {"mode": "full"},
            "sample_forest_scope": "each realized B_s, not unsampled global connectivity",
            "no_origin_features": True,
            "provenance_policy": (
                "frozen trainer-only CorruptionData; SHA checks at epoch/evaluation "
                "and collation boundaries"
            ),
        }

    def stress_batch(self, device):
        """Extra calibration-only largest-graph physical batch; training is untouched."""
        if self.indices is not None:
            raise ValueError("largest-graph stress batch is defined only for PPI graph batching")
        self.verify_provenance()
        records = [record for record in self._records if record[3].split == "train"]
        size = self.args.batch_size
        if type(size) is not int or size < 1 or size > len(records):
            raise ValueError("stress batch_size must fit the actual full PPI training graph count")
        ranked = sorted(
            records,
            key=lambda record: (
                -(record[0].x.shape[0] + record[0].incidence_edge_index.shape[1]),
                record[3].graph_id,
            ),
        )
        batch = collate_records(ranked[:size])
        if self.args.pin_memory:
            batch.pin_memory()
        return batch.to(device, non_blocking=self.args.pin_memory)

    def _cpu_training(self, epoch):
        self.verify_provenance()
        if self.sampler is not None:
            source_edges = self.training_record.graph.incidence_edge_index
            source_nodes = self.training_record.graph.x.shape[0]
            source_keys = source_edges[0] * source_nodes + source_edges[1]
            for graph in self.sampler.iter_epoch(epoch):
                started = time.perf_counter()
                batch = getattr(graph, "batch", None)
                plan = build_topology(
                    graph.x.shape[0],
                    graph.incidence_edge_index,
                    node_graph=batch,
                    forest_seed=self.args.forest_seed,
                )
                ids = getattr(graph, "global_physical_edge_id", None)
                if ids is None:
                    global_edges = (
                        graph.global_node_id[graph.incidence_edge_index].sort(dim=0).values
                    )
                    ids = torch.searchsorted(
                        source_keys, global_edges[0] * source_nodes + global_edges[1]
                    )
                target = self.training_record.origin_targets[ids]
                selected = graph.train_mask.nonzero(as_tuple=False).flatten()
                self.plan_preparation_seconds += time.perf_counter() - started
                yield SelectionBatch(graph, plan, target, selected)
        elif self.indices is None:
            self.data["train"].generator.manual_seed(self.args.model_seed + 1_000_003 * epoch)
            yield from self.data["train"]
        else:
            yield self.training_record

    def training_batches(self, epoch, device):
        if self.indices is not None and self.sampler is None:
            self.verify_provenance()
            if self._device_training is None:
                self._device_training = self.training_record.to(device)
            yield self._device_training
            return
        iterator = self._cpu_training(epoch)
        if self.sampler is not None and self.args.sample_prefetch:
            from ..v5.train import _prefetched_samples

            # One next-batch producer overlaps CPU graph/forest construction with CUDA.
            iterator = _prefetched_samples(iterator, pin_memory=self.args.pin_memory)
        for batch in iterator:
            yield batch.to(device, non_blocking=self.args.pin_memory)

    def validation_batches(self, device):
        self.verify_provenance()
        if self.indices is not None:
            if self._device_validation is None:
                self._device_validation = self.validation_record.to(device)
            yield self._device_validation
        else:
            for batch in self.data["validation"]:
                yield batch.to(device, non_blocking=self.args.pin_memory)

    def clean_validation(self, payload):
        clean_args = copy.deepcopy(self.args)
        clean_args.corruption_ratio = 0.0
        return PreparedInputs(payload, clean_args)
