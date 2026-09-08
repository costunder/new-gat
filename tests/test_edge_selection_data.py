"""CPU data contracts; real PyG/pinned-memory integration explicitly skips if unavailable."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.edge_selection import data
from research.conductance_gat.edge_selection.topology import build_topology


def _row(nodes=10):
    edges = torch.stack((torch.arange(nodes - 1), torch.arange(1, nodes)))
    return {
        "x": torch.arange(nodes * 3, dtype=torch.float32).reshape(nodes, 3),
        "y": torch.arange(nodes) % 2,
        "incidence_edge_index": edges,
    }


def _spec(row=None, *, graph_id=0, split="train", ratio=0.5):
    return (_row() if row is None else row, graph_id, split, ratio, 29, 17)


def _cpu_record(nodes=10, *, graph_id=0, split="train"):
    row = _row(nodes)
    corrupted = data.prepare_controlled_corruption(_spec(row, graph_id=graph_id, split=split))
    graph = SimpleNamespace(
        x=row["x"], y=row["y"], incidence_edge_index=corrupted.candidate_incidence
    )
    plan = build_topology(nodes, graph.incidence_edge_index, forest_seed=17)
    return graph, plan, corrupted.edge_targets, data.RecordEvidence(graph_id, corrupted)


def test_controlled_preparation_uses_frozen_provenance_and_distinct_eval_nonedges():
    train = data.prepare_controlled_corruption(_spec())
    validation = data.prepare_controlled_corruption(_spec(split="validation"))
    assert train.provenance.split == "train" and validation.provenance.split == "validation"
    assert not (
        set(map(tuple, train.negative_incidence.T.tolist()))
        & set(map(tuple, validation.negative_incidence.T.tolist()))
    )
    train.verify_unchanged()
    validation.verify_unchanged()
    with pytest.raises(dataclasses.FrozenInstanceError):
        train.provenance.seed = 3


def test_corruption_preparation_does_not_read_task_labels():
    row = _row()
    row["y"] = object()  # Not a tensor: geometry preparation must never inspect y.
    result = data.prepare_controlled_corruption(_spec(row))
    assert result.provenance.inserted_edge_count == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_type", torch.zeros(10, dtype=torch.long)),
        ("node_types", ["protein"]),
        ("node_type_id", torch.zeros(10, dtype=torch.long)),
        ("edge_relation_id", torch.zeros(9, dtype=torch.long)),
        ("edge_type", torch.zeros(9, dtype=torch.long)),
        ("relation_types", ["friend"]),
        ("num_relations", 1),
        ("edge_attr", torch.ones(9, 2)),
    ],
)
def test_typed_or_edge_feature_rows_are_rejected_before_pyg_import(field, value):
    row = _row()
    row[field] = value
    with pytest.raises(ValueError, match="cannot discard"):
        data.prepare_record(_spec(row))


@pytest.mark.parametrize("ratio", [-1, float("nan"), float("inf"), True])
def test_invalid_noise_budget_is_not_silently_changed(ratio):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        data.prepare_controlled_corruption(_spec(ratio=ratio))


def test_json_provenance_is_a_verified_copy_not_the_mutable_training_source():
    inputs = object.__new__(data.PreparedInputs)
    inputs._records = (_cpu_record(),)
    first = inputs.provenance
    json.dumps(first, allow_nan=False)
    first[0]["corruption"]["seed"] = 999
    assert inputs.provenance[0]["corruption"]["seed"] != 999
    inputs._records[0][2][0] = 99
    with pytest.raises(ValueError, match="changed after provenance"):
        _ = inputs.provenance


def test_record_evidence_checks_model_and_forest_alignment_not_only_saved_targets():
    graph, plan, targets, evidence = _cpu_record()
    evidence.verify_unchanged(graph, plan, targets)
    graph.incidence_edge_index = graph.incidence_edge_index.flip(1)
    with pytest.raises(ValueError, match="model candidate topology"):
        evidence.verify_unchanged(graph, plan, targets)
    with pytest.raises(ValueError, match="trainer origin targets"):
        evidence.verify_unchanged(targets=targets + 1)


def test_stress_batch_rejects_non_ppi_and_impossible_physical_batch():
    inputs = object.__new__(data.PreparedInputs)
    inputs.indices = {"train": torch.tensor([0])}
    with pytest.raises(ValueError, match="only for PPI"):
        inputs.stress_batch(torch.device("cpu"))
    inputs.indices = None
    inputs._records = (_cpu_record(),)
    inputs.args = SimpleNamespace(batch_size=2, pin_memory=False)
    with pytest.raises(ValueError, match="actual full PPI"):
        inputs.stress_batch(torch.device("cpu"))


def test_prepare_record_pyg_whitelist_has_no_origin_or_relation_features():
    pytest.importorskip("torch_geometric")
    row = _row()
    row["is_original"] = torch.ones(9)  # Deliberately untrusted extra metadata.
    graph, plan, targets, evidence = data.prepare_record(_spec(row))
    assert set(graph.keys()) == {"x", "y", "incidence_edge_index", "edge_index"}
    evidence.verify_unchanged(graph, plan, targets)
    assert graph.incidence_edge_index.shape[1] == len(targets)


def test_pyg_collation_merges_all_graphs_and_preserves_frozen_evidence():
    pytest.importorskip("torch_geometric")
    records = [data.prepare_record(_spec(_row(n), graph_id=i)) for i, n in enumerate((6, 8, 10))]
    batch = data.collate_records(records)
    assert batch.topology.num_graphs == 3
    assert batch.graph.x.shape[0] == 24
    assert torch.equal(batch.graph.incidence_edge_index, batch.topology.incidence_edge_index)
    assert batch.origin_targets.numel() == sum(record[2].numel() for record in records)
    moved = batch.to("cpu")
    assert moved.origin_targets.data_ptr() != batch.origin_targets.data_ptr()
    assert set(moved.graph.keys()).isdisjoint({"origin_targets", "provenance", "is_original"})


def test_pyg_stress_batch_uses_largest_train_graphs_without_changing_training_records():
    pytest.importorskip("torch_geometric")
    inputs = object.__new__(data.PreparedInputs)
    inputs.indices = None
    inputs.args = SimpleNamespace(batch_size=2, pin_memory=False)
    records = [data.prepare_record(_spec(_row(n), graph_id=i)) for i, n in enumerate((5, 11, 8))]
    validation = data.prepare_record(_spec(_row(15), graph_id=7, split="validation"))
    inputs._records = tuple([*records, validation])
    before = inputs.provenance
    stress = inputs.stress_batch(torch.device("cpu"))
    assert stress.graph.x.shape[0] == 19  # Train 11+8, not validation's larger 15.
    assert stress.topology.num_graphs == 2
    assert inputs.provenance == before
    assert len(inputs._records) == 4


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real pinned-memory allocator/CUDA unavailable; no fake pinning",
)
def test_real_topology_pin_memory_pins_every_tensor_without_changing_cpu_cache():
    plan = build_topology(5, _row(5)["incidence_edge_index"])
    before = {
        name: tensor.clone()
        for name, tensor in vars(plan).items()
        if isinstance(tensor, torch.Tensor)
    }
    pinned = plan.pin_memory()
    assert all(
        tensor.is_pinned() for tensor in vars(pinned).values() if isinstance(tensor, torch.Tensor)
    )
    for name, tensor in before.items():
        assert torch.equal(getattr(plan, name), tensor)
    moved = pinned.to("cuda:0", non_blocking=True)
    assert moved.incidence_edge_index.device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real pinned-memory allocator/CUDA unavailable; no fake pinning",
)
def test_real_selection_batch_pins_topology_graph_and_origin_targets():
    pytest.importorskip("torch_geometric")
    record = data.prepare_record(_spec())
    batch = data.collate_records([record])
    batch.pin_memory()
    assert batch.graph.x.is_pinned() and batch.origin_targets.is_pinned()
    assert all(
        tensor.is_pinned()
        for tensor in vars(batch.topology).values()
        if isinstance(tensor, torch.Tensor)
    )
