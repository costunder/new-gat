"""Synthetic CPU functional regressions, not GPU/performance or training evidence."""

from __future__ import annotations

import builtins
import copy
from types import MethodType, SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5 import diagnostics
from research.conductance_gat.v5 import model as model_module
from research.conductance_gat.v5.sampling import (
    TransductiveGraphSampler,
    induced_physical_edge_ids,
)


class DebugGraph(SimpleNamespace):
    """Tensor-container test double only; does not certify real PyG transfers."""

    def cpu(self):
        return self

    def items(self):
        return vars(self).items()

    def clone(self):
        return DebugGraph(**{key: value.clone() for key, value in self.items()})

    def to(self, device):
        for key, value in list(self.items()):
            setattr(self, key, value.to(device))
        return self


def debug_graph(n=100, *, disconnected=False, directed=False):
    nodes = torch.arange(n)
    if disconnected:
        size = n // 2
        head = torch.cat(((nodes[:size] + 1) % size, size + (nodes[size:] - size + 1) % (n - size)))
    else:
        head = (nodes + 1) % n
    incidence = torch.stack((nodes, head))
    return DebugGraph(
        x=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / n,
        y=nodes % 2,
        incidence_edge_index=incidence,
        edge_index=incidence if directed else torch.cat((incidence, incidence.flip(0)), dim=1),
    )


def legacy_cluster(sampler, seeds, generator):
    """Verbatim pre-optimization sampling law, kept as an independent oracle."""
    budget = max(seeds.numel(), seeds.numel() * (1 + sum(sampler.fanouts)))
    selected, frontier = seeds.unique(), seeds.unique()
    while selected.numel() < budget and frontier.numel():
        candidates = sampler._neighbors(frontier)
        unseen = candidates[~torch.isin(candidates, selected)]
        unseen = sampler._random_limit(unseen, budget - selected.numel(), generator)
        if not unseen.numel():
            break
        selected = torch.cat((selected, unseen)).unique()
        frontier = unseen
    return selected


def sampler_for(graph, *, batch=8, fanouts=(15, 10), train=None):
    return TransductiveGraphSampler(
        graph,
        torch.arange(19) if train is None else train,
        mode="cluster",
        seed_batch_size=batch,
        fanouts=fanouts,
        model_seed=17,
    )


@pytest.mark.parametrize("disconnected", [False, True])
@pytest.mark.parametrize("batch,fanouts", [(8, (15, 10)), (4, (15, 10)), (2, (3, 2))])
def test_cluster_nodes_and_rng_match_original_law(disconnected, batch, fanouts):
    sampler = sampler_for(debug_graph(disconnected=disconnected), batch=batch, fanouts=fanouts)
    for offset in (0, 47, 83):
        seeds = (torch.arange(batch) + offset) % 100
        old_rng, new_rng = torch.Generator().manual_seed(23), torch.Generator().manual_seed(23)
        expected = legacy_cluster(sampler, seeds, old_rng)
        actual = sampler._cluster_nodes(seeds, new_rng)
        assert torch.equal(actual, expected)
        assert torch.equal(new_rng.get_state(), old_rng.get_state())
        assert torch.equal(torch.isin(actual, seeds), torch.isin(expected, seeds))


def test_saturated_component_cache_is_reused_without_frontier_enumeration(monkeypatch):
    pytest.importorskip("scipy")
    sampler = sampler_for(debug_graph())
    first = sampler._cluster_nodes(torch.arange(8), torch.Generator().manual_seed(0))
    labels = sampler._component_labels
    assert labels is not None
    monkeypatch.setattr(sampler, "_neighbors", lambda _: pytest.fail("BFS repeated"))
    second = sampler._cluster_nodes(torch.arange(50, 58), torch.Generator().manual_seed(1))
    assert sampler._component_labels is labels
    assert torch.equal(first, second) and first.numel() == 100
    assert sampler.metadata()["cluster_saturation"]["full_seed_batch_reaches_graph_size"]
    assert (
        sampler.metadata()["cluster_saturation"]["component_cache"]
        == "immutable_undirected_component_csr"
    )


def test_asymmetric_graph_retains_directed_bfs_reachability():
    graph = debug_graph(directed=True)
    # A forward chain, not a directed ring: seeds cannot reach preceding nodes.
    graph.edge_index = graph.edge_index[:, :-1]
    sampler = sampler_for(graph)
    seeds = torch.arange(50, 58)
    generator = torch.Generator().manual_seed(12)
    actual = sampler._cluster_nodes(seeds, generator)
    expected = legacy_cluster(sampler, seeds, torch.Generator().manual_seed(12))
    assert torch.equal(actual, expected)
    assert actual.numel() == 50
    assert sampler._component_cache_status == "original_bfs_asymmetric_adjacency"


def test_missing_scipy_keeps_original_law_and_explicit_reason(monkeypatch):
    real_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("explicit synthetic missing optional dependency")
        return real_import(name, *args, **kwargs)

    sampler = sampler_for(debug_graph())
    monkeypatch.setattr(builtins, "__import__", without_scipy)
    actual = sampler._cluster_nodes(torch.arange(8), torch.Generator().manual_seed(0))
    assert actual.numel() == 100
    assert sampler._component_cache_status == "original_bfs_scipy_unavailable"


def test_epoch_seed_masks_edges_and_following_rng_order_match_original(monkeypatch):
    graph = debug_graph(n=103, disconnected=True)
    actual = sampler_for(graph)
    expected = sampler_for(graph)
    expected._cluster_nodes = MethodType(legacy_cluster, expected)

    def snapshot(sampler, nodes, seeds):
        edge_ids, _ = induced_physical_edge_ids(
            sampler.incidence,
            sampler.incident_edge_ids,
            sampler.incident_rowptr,
            nodes,
            103,
        )
        return nodes, seeds, torch.isin(nodes, seeds), edge_ids

    # Only the PyG container construction is bypassed: compare exact real
    # sampler-selected nodes/seeds, induced edge IDs, and supervised masks.
    actual._induced = MethodType(snapshot, actual)
    expected._induced = MethodType(snapshot, expected)
    for epoch in (1, 2, 9):
        left, right = list(actual.iter_epoch(epoch)), list(expected.iter_epoch(epoch))
        assert len(left) == len(right) == 3  # The last batch is NOT saturated.
        for actual_fields, expected_fields in zip(left, right, strict=True):
            for a, b in zip(actual_fields, expected_fields, strict=True):
                assert torch.equal(a, b)


def test_saturation_warning_is_explicit_without_changing_recipe():
    with pytest.warns(RuntimeWarning, match="Sampling law.*unchanged"):
        sampler = sampler_for(debug_graph(), batch=8)
    assert sampler.seed_batch_size == 8
    assert sampler.fanouts == (15, 10)
    assert sampler.num_nodes == 100


@pytest.mark.parametrize("seeds", [torch.tensor([-1, 0, 1, 2]), torch.arange(4).int()])
def test_saturated_path_does_not_relax_seed_index_contract(seeds):
    sampler = sampler_for(debug_graph())
    with pytest.raises(ValueError):
        sampler._cluster_nodes(seeds, torch.Generator().manual_seed(0))


def old_pool(values, graph_ids, count):
    sums = values.new_zeros((count, values.shape[1])).index_add(0, graph_ids, values)
    counts = values.new_zeros(count).index_add(0, graph_ids, values.new_ones(values.shape[0]))
    return sums / counts.clamp_min(1)[:, None]


@pytest.mark.parametrize("nodes,graphs", [(0, 1), (13, 1), (14, 2)])
def test_graph_pool_matches_original_values_and_gradients(nodes, graphs):
    state = torch.randn(nodes, 7, dtype=torch.float64, requires_grad=True)
    ids = torch.arange(nodes) % graphs
    result = model_module._graph_node_mean(state, ids, graphs)
    expected = old_pool(state, ids, graphs)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(result.square().sum(), state, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), state)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", ["mlp", "optimization"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_static_degree_computed_once_and_model_loss_gradients_preserved(
    monkeypatch, backend, checkpoint
):
    torch.manual_seed(41)
    graph = debug_graph(n=9)
    actual = model_module.GraphConditionedConductanceNodeClassifier(
        3,
        2,
        hidden_channels=8,
        layers=3,
        heads=2,
        ffn_multiplier=2,
        dropout=0,
        conductance_mode="dynamic",
        conductance_backend=backend,
        activation_checkpoint=checkpoint,
        edge_chunk_size=8,
    )
    expected = copy.deepcopy(actual)
    degree = model_module._node_degree
    calls = []

    def counted(*args):
        calls.append(1)
        return degree(*args)

    monkeypatch.setattr(model_module, "_node_degree", counted)
    actual_logits = actual(graph)
    torch.nn.functional.cross_entropy(actual_logits, graph.y).backward()
    assert len(calls) == 1
    context = model_module.graph_context_features

    def uncached(*args, **kwargs):
        kwargs.pop("static_context", None)
        return context(*args, **kwargs)

    monkeypatch.setattr(model_module, "graph_context_features", uncached)
    expected_logits = expected(graph)
    torch.nn.functional.cross_entropy(expected_logits, graph.y).backward()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
    for (name, a), (_, b) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)


def test_static_context_rejects_another_graph_even_when_shapes_match():
    graph = debug_graph(n=7)
    ids = torch.zeros(7, dtype=torch.long)
    context = model_module._static_graph_context(graph.x, graph.incidence_edge_index, ids, 1)
    with pytest.raises(ValueError, match="same-device forward graph"):
        model_module.graph_context_features(
            graph.x,
            graph.incidence_edge_index.clone(),
            ids,
            1,
            static_context=context,
        )


def test_single_graph_path_does_not_silently_accept_out_of_range_graph_ids():
    graph = debug_graph(n=7)
    with pytest.raises(RuntimeError, match="outside"):
        model_module.graph_context_features(
            graph.x,
            graph.incidence_edge_index,
            torch.ones(7, dtype=torch.long),
            1,
        )


def test_cached_multigraph_context_preserves_current_hidden_statistics_and_gradients():
    torch.manual_seed(31)
    state = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    edges = torch.tensor([[0, 1, 2, 0, 4, 5, 6, 4], [1, 2, 3, 3, 5, 6, 7, 7]])
    degree = torch.bincount(edges.flatten(), minlength=8).double() * 2
    structure = torch.randn(2, 6, dtype=torch.float64)
    cached = model_module._static_graph_context(state, edges, ids, 2, degree, structure)
    changed_state = state * 1.5 + torch.arange(8).double()[:, None]
    actual = model_module.graph_context_features(
        changed_state,
        edges,
        ids,
        2,
        degree,
        structure,
        static_context=cached,
    )[0]
    expected = model_module.graph_context_features(changed_state, edges, ids, 2, degree, structure)[
        0
    ]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(actual.square().sum(), state, retain_graph=True)[0]
    b = torch.autograd.grad(expected.square().sum(), state)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


class DebugPredictor(torch.nn.Module):
    def forward(self, graph):
        return torch.stack((graph.x[:, 0], -graph.x[:, 0]), dim=1)


def test_validation_cache_clones_once_preserves_source_and_same_metric(monkeypatch):
    source = debug_graph(n=8)
    original_x = source.x.clone()
    indices = torch.tensor([0, 1, 4])
    calls = []
    clone = DebugGraph.clone

    def counted_clone(self):
        calls.append(1)
        return clone(self)

    monkeypatch.setattr(DebugGraph, "clone", counted_clone)
    cache = diagnostics.PreparedValidationGraph(source, torch.device("cpu"))
    model = DebugPredictor()
    first = diagnostics.evaluate(
        model, cache, indices, device=torch.device("cpu"), collect_diagnostics=False
    )
    copied_indices = cache._selected
    second = diagnostics.evaluate(
        model, cache, indices, device=torch.device("cpu"), collect_diagnostics=False
    )
    assert first == second
    assert cache._selected is copied_indices
    assert len(calls) == 1
    assert torch.equal(source.x, original_x)
    assert cache.graph.x.data_ptr() != source.x.data_ptr()
    assert cache.metadata["cached_graph_tensor_storage_bytes"] > 0
    expected = diagnostics.evaluate(
        model, source, indices, device=torch.device("cpu"), collect_diagnostics=False
    )
    assert expected == first


def test_validation_cache_rejects_mutation_and_wrong_device():
    cache = diagnostics.PreparedValidationGraph(debug_graph(n=8), torch.device("cpu"))
    with pytest.raises(ValueError, match="another device"):
        cache.prepare(torch.arange(3), torch.device("cuda:0"))
    cache.graph.x.add_(1)
    with pytest.raises(RuntimeError, match="cache was mutated"):
        cache.prepare(torch.arange(3), torch.device("cpu"))


def test_validation_cache_tracks_updated_indices_without_mutating_original():
    cache = diagnostics.PreparedValidationGraph(debug_graph(n=8), torch.device("cpu"))
    indices = torch.tensor([0, 1, 2])
    _, first = cache.prepare(indices, torch.device("cpu"))
    indices[0] = 5
    _, second = cache.prepare(indices, torch.device("cpu"))
    assert first[0] == 0 and second[0] == 5
    assert first is not second and second.data_ptr() != indices.data_ptr()


def test_validation_cache_rejects_mutated_cached_indices():
    cache = diagnostics.PreparedValidationGraph(debug_graph(n=8), torch.device("cpu"))
    indices = torch.arange(3)
    _, selected = cache.prepare(indices, torch.device("cpu"))
    selected[0] = 7
    with pytest.raises(RuntimeError, match="index cache was mutated"):
        cache.prepare(indices, torch.device("cpu"))


def test_diagnostics_collection_can_be_disabled_without_changing_metric(monkeypatch):
    calls = []

    def collect(_):
        calls.append(1)
        return [{"synthetic_unit_diagnostic": True}]

    monkeypatch.setattr(diagnostics, "layer_diagnostics", collect)
    graph, indices, model = debug_graph(n=8), torch.arange(4), DebugPredictor()
    without = diagnostics.evaluate(
        model, graph, indices, device=torch.device("cpu"), collect_diagnostics=False
    )
    assert not calls and "layers" not in without
    default = diagnostics.evaluate(model, graph, indices, device=torch.device("cpu"))
    assert len(calls) == 1 and default.pop("layers")
    assert default == without
