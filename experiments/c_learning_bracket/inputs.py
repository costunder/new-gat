"""Unchanged graph samples, without selection/cycle plans or origin targets."""

import copy
import hashlib
import time
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from research.conductance_gat.edge_selection.corruption import canonical_incidence
from research.conductance_gat.v5.sampling import TransductiveGraphSampler
from research.conductance_gat.v5.train import _prefetched_samples, tensor_hash


@dataclass
class BracketBatch:
    graph: object
    selected_indices: torch.Tensor

    def pin_memory(self):
        self.graph.pin_memory()
        self.selected_indices = self.selected_indices.pin_memory()
        return self

    def pinning_status(self):
        tensors = {k: v for k, v in self.graph if isinstance(v, torch.Tensor)}
        tensors["selected_indices"] = self.selected_indices
        return {k: v.is_pinned() for k, v in tensors.items() if v.numel()}

    def to(self, device, *, non_blocking=False):
        # clone() would allocate pageable CPU storage again after pin_memory().
        graph = copy.copy(self.graph).to(device, non_blocking=non_blocking)
        graph._v5_num_graphs = self.graph._v5_num_graphs
        return BracketBatch(graph, self.selected_indices.to(device, non_blocking=non_blocking))


class BracketInputs:
    def __init__(self, payload, args):
        started = time.perf_counter()
        if args.dataset != "ogbn-arxiv" or len(payload["graphs"]) != 1:
            raise ValueError("bracket inputs require the declared single transductive graph")
        if args.corruption_ratio != 0 or args.sampling != "cluster_disjoint":
            raise ValueError("bracket preserves clean cluster_disjoint sampling")
        row = payload["graphs"][0]
        if any(
            row.get(k) is not None
            for k in (
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
        ) or row.get("num_relations", 0) not in (0, None):
            raise ValueError("cannot silently discard typed node/edge inputs")
        edges, _ = canonical_incidence(row["incidence_edge_index"])
        nodes = row["x"].shape[0]
        keys = edges[0] * nodes + edges[1]
        if edges.numel() and (
            bool((edges < 0).any())
            or bool((edges >= nodes).any())
            or bool((edges[0] == edges[1]).any())
            or bool((keys[1:] == keys[:-1]).any())
        ):
            raise ValueError("expected unique in-range physical edges without self loops")
        self.args = args
        self.data = Data(
            x=row["x"],
            y=row["y"],
            incidence_edge_index=edges,
            edge_index=torch.cat((edges, edges.flip(0)), 1),
        )
        self.indices = {
            split: payload["splits"][split].nonzero(as_tuple=False).flatten().long()
            for split in ("train", "validation")
        }
        self.sampler = TransductiveGraphSampler(
            self.data,
            self.indices["train"],
            mode=args.sampling,
            seed_batch_size=args.sample_seed_batch_size,
            fanouts=args.num_neighbors,
            model_seed=args.model_seed,
            context_seed_batch_size=args.sample_context_seed_batch_size,
            context_workers=args.sample_context_workers,
        )
        self.train_count = self.indices["train"].numel()
        self.validation_count = self.indices["validation"].numel()
        self._frozen_hashes = self._hashes()
        self._device_validation = None
        self.last_transfer_evidence = None
        self.preparation_seconds = time.perf_counter() - started

    def _hashes(self):
        return {
            **{k: tensor_hash(self.data[k]) for k in ("x", "y", "incidence_edge_index")},
            **{k: tensor_hash(v) for k, v in self.indices.items()},
        }

    def verify_provenance(self):
        if self._hashes() != self._frozen_hashes:
            raise ValueError("source graph/features/labels/splits changed during execution")

    def metadata(self):
        self.verify_provenance()
        return {
            "source_sha256": self._frozen_hashes,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "sampling": self.sampler.metadata(),
            "preparation_seconds": self.preparation_seconds,
            "topology_plans_built": 0,
            "input_pipeline": "bracket graph tensors; no forest/cycle/origin target preparation",
            "pin_memory_requested": self.args.pin_memory,
            "last_transfer": self.last_transfer_evidence,
        }

    @staticmethod
    def _transport_graph(source):
        needed = (
            "x",
            "y",
            "incidence_edge_index",
            "batch",
            "full_degree",
            "graph_structure",
            "sampling_correction",
            "global_node_id",
        )
        graph = Data(
            **{k: getattr(source, k) for k in needed if getattr(source, k, None) is not None}
        )
        groups = getattr(graph, "batch", None)
        graph._v5_num_graphs = 1 if groups is None else int(groups.max()) + 1
        for key in ("sampling_observation", "study_batch_identity"):
            if getattr(source, key, None) is not None:
                setattr(graph, key, getattr(source, key))
        return graph

    def _cpu_training(self, epoch):
        self.verify_provenance()
        for source in self.sampler.iter_epoch(epoch):
            selected = source.train_mask.nonzero(as_tuple=False).flatten()
            ids = source.global_node_id[selected]
            self._seen.index_add_(0, ids, torch.ones_like(ids))
            identity = {
                "original_nodes": tensor_hash(source.global_node_id),
                "local_edges": tensor_hash(source.incidence_edge_index),
                "supervised_original_ids": tensor_hash(ids),
            }
            self._sequence.append(hashlib.sha256("|".join(identity.values()).encode()).hexdigest())
            source.study_batch_identity = identity
            yield BracketBatch(self._transport_graph(source), selected)

    def _transfer(self, batch, device):
        start = time.perf_counter()
        if self.args.pin_memory:
            batch.pin_memory()
        status = batch.pinning_status()
        if self.args.pin_memory and not all(status.values()):
            raise RuntimeError("requested pinning was not applied before device transfer")
        self.last_transfer_evidence = {
            "pinning_requested": self.args.pin_memory,
            "is_pinned_before_copy": status,
            "pinning_and_check_seconds": time.perf_counter() - start,
            "prefetch_enabled": self.args.sample_prefetch,
        }
        result = batch.to(device, non_blocking=self.args.pin_memory)
        result.transfer_evidence = self.last_transfer_evidence
        return result

    def training_batches(self, epoch, device):
        self._seen = torch.zeros(self.data.x.shape[0], dtype=torch.long)
        self._sequence = []
        iterator = self._cpu_training(epoch)
        if self.args.sample_prefetch:
            # Pin/check in the common transfer path for both branches.
            iterator = _prefetched_samples(iterator, pin_memory=False)
        for batch in iterator:
            yield self._transfer(batch, device)
        expected = torch.zeros_like(self._seen)
        expected[self.indices["train"]] = 1
        if not torch.equal(self._seen, expected):
            raise ValueError("must supervise every train ID exactly once per epoch")
        self.last_pass_evidence = {
            "supervised_nodes": self.train_count,
            "seed_counts_sha256": tensor_hash(self._seen),
            "sample_sequence_sha256": hashlib.sha256("|".join(self._sequence).encode()).hexdigest(),
            "physical_batches": len(self._sequence),
            "every_train_seed_exactly_once": True,
        }

    def validation_batches(self, device):
        self.verify_provenance()
        if self._device_validation is None:
            record = BracketBatch(self._transport_graph(self.data), self.indices["validation"])
            self._device_validation = self._transfer(record, device)
        yield self._device_validation
