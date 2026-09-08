"""Synthetic CPU sampler/gradient regressions, not real-data or GPU evidence.

The explicit container fixture exercises sampler math without optional PyG.
The separate integration test only passes when actual PyG is installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.sampling import TransductiveGraphSampler, physical_degree


class DebugData(SimpleNamespace):
    def cpu(self):
        return self

    @property
    def num_nodes(self):
        return self.x.shape[0]


class DebugBatch(DebugData):
    """Explicit test double of the used PyG concatenation/offset contract."""

    @classmethod
    def from_data_list(cls, graphs):
        ptr = torch.tensor([0] + [graph.num_nodes for graph in graphs]).cumsum(0)
        fields = {}
        for key in vars(graphs[0]):
            values = [getattr(graph, key) for graph in graphs]
            if key in {"edge_index", "incidence_edge_index"}:
                fields[key] = torch.cat([value + ptr[i] for i, value in enumerate(values)], 1)
            else:
                fields[key] = torch.cat(values, 0)
        return cls(
            **fields,
            ptr=ptr,
            batch=torch.repeat_interleave(torch.arange(len(graphs)), ptr.diff()),
            num_graphs=len(graphs),
            debug_contexts=graphs,
        )


@pytest.fixture
def debug_container(monkeypatch):
    package = ModuleType("torch_geometric")
    data = ModuleType("torch_geometric.data")
    data.Data, data.Batch = DebugData, DebugBatch
    package.data = data
    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data)


def source_graph(n=71, cls=DebugData):
    # Ring plus chords: enough frontier choices to test stochastic contexts.
    nodes = torch.arange(n)
    incidence = torch.cat(
        (torch.stack((nodes, (nodes + 1) % n)), torch.stack((nodes, (nodes + 5) % n))), 1
    )
    return cls(
        x=torch.arange(n * 5, dtype=torch.float32).reshape(n, 5).sin(),
        y=nodes % 3,
        incidence_edge_index=incidence,
        edge_index=torch.cat((incidence, incidence.flip(0)), 1),
    )


def make_sampler(
    *, physical=6, context=2, workers=1, train=None, graph=None, mode="cluster_disjoint"
):
    return TransductiveGraphSampler(
        source_graph() if graph is None else graph,
        torch.arange(17) if train is None else train,
        mode=mode,
        seed_batch_size=physical,
        fanouts=(2, 1),
        model_seed=13,
        context_seed_batch_size=context,
        context_workers=workers,
    )


@pytest.mark.parametrize("context", [None, 0, -1, 2.5, 2.0, True, False])
def test_disjoint_requires_explicit_positive_integer_context(context):
    with pytest.raises(ValueError, match="positive context_seed_batch_size"):
        make_sampler(context=context)


def test_context_options_cannot_silently_modify_legacy_sampler():
    with pytest.raises(ValueError, match="only to cluster_disjoint"):
        make_sampler(mode="cluster")
    with pytest.raises(ValueError, match="positive integer"):
        make_sampler(workers=0)
    with pytest.raises(ValueError, match="unique supervised"):
        make_sampler(train=torch.tensor([0, 1, 1]))


@pytest.mark.parametrize("count", [True, False, 1.0, 2.5])
def test_counts_cannot_be_booleans_or_silently_truncated(count):
    with pytest.raises(ValueError, match="positive integer"):
        make_sampler(workers=count)
    with pytest.raises(ValueError, match="positive integers"):
        make_sampler(physical=count)
    with pytest.raises(ValueError, match="positive integers"):
        TransductiveGraphSampler(
            source_graph(),
            torch.arange(17),
            mode="cluster",
            seed_batch_size=6,
            fanouts=(count, 1),
            model_seed=13,
        )


@pytest.mark.parametrize("physical,context", [(5, 2), (2, 3), (7, 4)])
def test_sampler_itself_rejects_regrouping_that_changes_context_seed_boundaries(physical, context):
    with pytest.raises(ValueError, match="whole sampling contexts"):
        make_sampler(physical=physical, context=context)


def test_single_full_train_batch_may_have_a_partial_final_context(debug_container):
    sampler = make_sampler(physical=17, context=3)
    batches = list(sampler.iter_epoch(2))
    assert len(batches) == 1
    assert [int(graph.train_mask.sum()) for graph in batches[0].debug_contexts] == [
        3,
        3,
        3,
        3,
        3,
        2,
    ]
    regrouped = make_sampler(physical=6, context=3)
    left = [graph for batch in batches for graph in batch.debug_contexts]
    right = [graph for batch in regrouped.iter_epoch(2) for graph in batch.debug_contexts]
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert torch.equal(a.global_node_id, b.global_node_id)
        assert torch.equal(a.train_mask, b.train_mask)
        assert torch.equal(a.incidence_edge_index, b.incidence_edge_index)


def test_disjoint_supervision_and_original_ids_cover_every_seed_once(debug_container):
    sampler = make_sampler()
    batches = list(sampler.iter_epoch(3))
    assert len(batches) == len(sampler) == 3
    assert [int(batch.train_mask.sum()) for batch in batches] == [6, 6, 5]
    supervised = torch.cat([batch.global_node_id[batch.train_mask] for batch in batches])
    assert torch.equal(supervised.sort().values, sampler.train_indices.sort().values)
    for batch in batches:
        assert batch._v5_num_graphs == batch.num_graphs == 3
        assert batch.graph_structure.shape == (3, 6)
        assert torch.equal(batch.x, sampler.graph.x[batch.global_node_id])
        assert torch.equal(batch.y, sampler.graph.y[batch.global_node_id])
        tail, head = batch.incidence_edge_index
        assert torch.equal(batch.batch[tail], batch.batch[head])
        assert torch.equal(
            batch.global_node_id[batch.incidence_edge_index],
            sampler.incidence[:, batch.global_physical_edge_id],
        )
        assert torch.equal(batch.full_degree, sampler.full_degree[batch.global_node_id])
        degree = physical_degree(batch.incidence_edge_index, batch.num_nodes)
        ratio = batch.full_degree / degree.clamp_min(1)
        expected = (ratio[tail] * ratio[head]).sqrt().clamp(1, 64)
        torch.testing.assert_close(batch.edge_normalization_weight, expected)
        assert torch.equal(batch.sampling_correction, expected)
        assert int(batch.sample_seed_count.sum()) == int(batch.train_mask.sum())
    assert batches[-1].sampling_observation["epoch_seed_coverage_fraction"] == 1


def test_physical_batch_does_not_inflate_individual_context_budget(debug_container):
    small = make_sampler(physical=2)
    large = make_sampler(physical=6)
    left = [graph for batch in small.iter_epoch(5) for graph in batch.debug_contexts]
    right = [graph for batch in large.iter_epoch(5) for graph in batch.debug_contexts]
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert torch.equal(a.global_node_id, b.global_node_id)
        assert torch.equal(a.train_mask, b.train_mask)
        assert torch.equal(a.incidence_edge_index, b.incidence_edge_index)
        assert a.num_nodes <= 2 * (1 + sum(small.fanouts))
    metadata = large.metadata()
    assert metadata["seed_batch_size"] == 6
    assert metadata["context_seed_batch_size"] == 2
    assert metadata["contexts_per_full_physical_batch"] == 3
    assert metadata["cluster_saturation"]["configured_node_budget"] == 8
    assert not metadata["cluster_saturation"]["full_seed_batch_reaches_graph_size"]


def test_context_workers_reproduce_exact_samples_without_global_rng_changes(debug_container):
    one = make_sampler(workers=1)
    many = make_sampler(workers=3)
    state = torch.random.get_rng_state().clone()
    a, b = list(one.iter_epoch(4)), list(many.iter_epoch(4))
    assert torch.equal(state, torch.random.get_rng_state())
    for left, right in zip(a, b, strict=True):
        for field in (
            "global_node_id",
            "global_physical_edge_id",
            "incidence_edge_index",
            "train_mask",
            "full_degree",
            "sampling_correction",
        ):
            assert torch.equal(getattr(left, field), getattr(right, field))
        left_observation = dict(left.sampling_observation)
        right_observation = dict(right.sampling_observation)
        assert left_observation.pop("observation_cpu_wall_seconds") >= 0
        assert right_observation.pop("observation_cpu_wall_seconds") >= 0
        assert left_observation == right_observation
    assert a[0].sampling_observation != list(one.iter_epoch(5))[0].sampling_observation


def test_observation_snapshots_are_prefetch_safe_and_report_exact_overlap(debug_container):
    sampler = make_sampler(physical=10, context=2)
    stream = sampler.iter_epoch(1)
    first = next(stream)
    saved = copy.deepcopy(first.sampling_observation)
    second = next(stream)
    assert first.sampling_observation == saved
    assert first.sampling_observation is not sampler.last_sample_observation
    json.dumps(saved, allow_nan=False)
    assert saved["observation_cpu_wall_seconds"] >= 0
    assert "all-pair overlap" in saved["observation_timing_scope"]
    nodes = first.global_node_id.unique(sorted=True)
    edges = first.global_physical_edge_id.unique(sorted=True)
    assert saved["node_ids_sha256"] == hashlib.sha256(nodes.numpy().tobytes()).hexdigest()
    assert saved["physical_edge_ids_sha256"] == hashlib.sha256(edges.numpy().tobytes()).hexdigest()
    assert saved["duplicated_context_node_copies"] == first.num_nodes - nodes.numel()
    assert saved["duplicated_context_node_copies"] > 0
    expected = set(nodes.tolist()) & set(second.global_node_id.tolist())
    union = set(nodes.tolist()) | set(second.global_node_id.tolist())
    actual = second.sampling_observation["previous_physical_batch_overlap"]["nodes"]
    assert actual["intersection"] == len(expected)
    assert actual["jaccard"] == len(expected) / len(union)
    for pair in saved["within_batch_context_overlap"]:
        i, j = pair["contexts"]
        a = set(first.debug_contexts[i].global_node_id.tolist())
        b = set(first.debug_contexts[j].global_node_id.tolist())
        assert pair["nodes"]["intersection"] == len(a & b)
        assert pair["nodes"]["jaccard"] == len(a & b) / len(a | b)
        edge_graph = first.batch[first.incidence_edge_index[0]]
        a_edges = set(first.global_physical_edge_id[edge_graph == i].tolist())
        b_edges = set(first.global_physical_edge_id[edge_graph == j].tolist())
        assert pair["physical_edges"]["intersection"] == len(a_edges & b_edges)
        assert pair["physical_edges"]["union"] == len(a_edges | b_edges)
    stream.close()


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        ([], [1]),
        ([1], []),
        ([1], [2]),
        ([1, 5], [1, 3, 5, 7]),
        ([1, 3, 5, 7], [1, 5]),
        ([1, 2, 9], [1, 2, 9]),
    ],
)
def test_sorted_overlap_preserves_exact_intersections_empty_cases_and_symmetry(left, right):
    a, b = torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)
    actual = TransductiveGraphSampler._overlap(a, b)
    intersection, union = len(set(left) & set(right)), len(set(left) | set(right))
    assert actual == {
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
        "both_empty": union == 0,
    }
    assert actual == TransductiveGraphSampler._overlap(b, a)


def test_sorted_overlap_matches_isin_for_all_random_context_pairs():
    generator = torch.Generator().manual_seed(554)
    contexts = [
        torch.randint(200, (75,), generator=generator).unique(sorted=True) for _ in range(12)
    ]
    for i, left in enumerate(contexts):
        for right in contexts[i + 1 :]:
            actual = TransductiveGraphSampler._overlap(left, right)
            assert actual["intersection"] == int(torch.isin(left, right).sum())


def test_disjoint_model_forward_and_task_gradients_equal_independent_contexts(debug_container):
    torch.manual_seed(61)
    sampler = make_sampler(physical=6, train=torch.arange(6))
    batch = next(sampler.iter_epoch(1))
    network = GraphConditionedConductanceNodeClassifier(
        5,
        3,
        hidden_channels=8,
        layers=2,
        heads=2,
        ffn_multiplier=2,
        dropout=0,
        conductance_mode="dynamic",
        conductance_backend="optimization",
        solver_steps=8,
        solver_cost_scaling="width_scaled",
        activation_checkpoint=False,
    )
    separate = copy.deepcopy(network)
    actual = network(batch)
    expected = torch.cat([separate(graph) for graph in batch.debug_contexts])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    torch.nn.functional.cross_entropy(
        actual[batch.train_mask], batch.y[batch.train_mask]
    ).backward()
    torch.nn.functional.cross_entropy(
        expected[batch.train_mask], batch.y[batch.train_mask]
    ).backward()
    nonzero_c = 0
    for (name, a), (_, b) in zip(
        network.named_parameters(), separate.named_parameters(), strict=True
    ):
        assert a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-6)
        if ".estimator." in name:
            assert bool(torch.isfinite(a.grad).all())
            nonzero_c += int(bool(a.grad.abs().sum() > 0))
    assert nonzero_c > 0


def test_saturated_contexts_are_explicit_not_silently_capped(debug_container):
    graph = source_graph(7)
    with pytest.warns(RuntimeWarning, match="context budget reaches"):
        sampler = make_sampler(physical=6, context=2, graph=graph, train=torch.arange(6), workers=3)
    batch = next(sampler.iter_epoch(1))
    assert batch.num_nodes == 3 * 7
    assert batch.sampling_observation["saturated_context_count"] == 3
    assert all(
        item["selects_all_original_nodes"] for item in batch.sampling_observation["contexts"]
    )


def test_isolated_contexts_preserve_supervision_and_mark_empty_edge_overlap(debug_container):
    source = DebugData(
        x=torch.ones(20, 5),
        y=torch.arange(20) % 3,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        incidence_edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    sampler = make_sampler(graph=source, train=torch.arange(7), physical=6, workers=2)
    batches = list(sampler.iter_epoch(1))
    assert [int(batch.train_mask.sum()) for batch in batches] == [6, 1]
    assert all(batch.incidence_edge_index.shape == (2, 0) for batch in batches)
    for pair in batches[0].sampling_observation["within_batch_context_overlap"]:
        assert pair["nodes"]["jaccard"] == 0
        assert pair["physical_edges"] == {
            "intersection": 0,
            "union": 0,
            "jaccard": 1.0,
            "both_empty": True,
        }
    assert batches[-1].sampling_observation["epoch_seed_coverage_fraction"] == 1


def test_real_pyg_disjoint_container_offsets_and_cpu_model_smoke():
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch, Data

    graph = source_graph(cls=Data)
    sampler = make_sampler(graph=graph)
    batch = next(sampler.iter_epoch(1))
    assert isinstance(batch, Batch)
    assert batch._v5_num_graphs == 3
    assert torch.equal(
        batch.global_node_id[batch.incidence_edge_index],
        sampler.incidence[:, batch.global_physical_edge_id],
    )
    assert torch.equal(
        batch.batch[batch.incidence_edge_index[0]], batch.batch[batch.incidence_edge_index[1]]
    )
    network = GraphConditionedConductanceNodeClassifier(
        5,
        3,
        hidden_channels=8,
        layers=2,
        heads=2,
        ffn_multiplier=2,
        dropout=0,
        activation_checkpoint=False,
    )
    logits = network(batch)
    torch.nn.functional.cross_entropy(
        logits[batch.train_mask], batch.y[batch.train_mask]
    ).backward()
    assert all(parameter.grad is not None for parameter in network.parameters())
