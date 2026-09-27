"""CPU provenance for complete supervised passes, before asynchronous GPU transfer."""

import hashlib

import torch

from research.conductance_gat.edge_selection.data import PreparedInputs as BaseInputs
from research.conductance_gat.v5.train import tensor_hash


class StudyInputs(BaseInputs):
    def _record_study_batch(self, batch):
        graph = batch.graph
        original_ids = getattr(graph, "global_node_id", None)
        if original_ids is None:
            original_ids = torch.arange(graph.x.shape[0])
        selected = original_ids[batch.selected_indices]
        self._seen.index_add_(0, selected, torch.ones_like(selected))
        identity = {
            "original_nodes": tensor_hash(original_ids),
            "local_edges": tensor_hash(graph.incidence_edge_index),
            "supervised_original_ids": tensor_hash(selected),
        }
        digest = hashlib.sha256("|".join(identity.values()).encode()).hexdigest()
        self._sequence.append(digest)
        graph.study_batch_identity = identity
        return batch

    def _cpu_training(self, epoch):
        for batch in super()._cpu_training(epoch):
            yield self._record_study_batch(batch)

    def training_batches(self, epoch, device):
        self._seen = torch.zeros(self.data.x.shape[0], dtype=torch.long)
        self._sequence = []
        if self.sampler is None:
            self._record_study_batch(self.training_record)
        yield from super().training_batches(epoch, device)
        expected = torch.zeros_like(self._seen)
        expected[self.indices["train"]] = 1
        if not torch.equal(self._seen, expected):
            raise ValueError("core study must supervise every train ID exactly once per pass")
        self.last_pass_evidence = {
            "supervised_nodes": self.train_count,
            "seed_counts_sha256": tensor_hash(self._seen),
            "sample_sequence_sha256": hashlib.sha256("|".join(self._sequence).encode()).hexdigest(),
            "physical_batches": len(self._sequence),
            "every_train_seed_exactly_once": True,
        }
