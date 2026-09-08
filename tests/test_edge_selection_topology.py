"""Synthetic CPU unit checks only; not a reduced real-data experiment."""

import dataclasses
import pickle

import pytest
import torch

from research.conductance_gat.edge_selection.topology import (
    batch_topologies,
    build_topology,
    cycle_context,
    cycle_to_edge,
    edge_to_cycle,
)


def _fixture():
    return 8, torch.tensor([[0, 0, 1, 1, 2, 4, 4, 5], [1, 2, 2, 3, 3, 5, 6, 6]])


def _explicit_basis(plan):
    """Tiny-test oracle only: production code never allocates this matrix."""
    z = torch.zeros(plan.num_edges, len(plan.chord_indices), dtype=torch.float64)
    for column, edge in enumerate(plan.chord_indices.tolist()):
        z[edge, column] = 1
        node, ancestor = int(plan.chord_descendant[column]), int(plan.chord_ancestor[column])
        while node != ancestor:
            z[int(plan.parent_edge[node]), column] = -plan.chord_sign[column] * plan.tree_sign[node]
            node = int(plan.parent[node])
    return z


def test_dfs_forest_components_and_exact_incidence_kernel_basis():
    nodes, edges = _fixture()
    plan = build_topology(nodes, edges, forest_seed=7)
    assert plan.metadata["num_components"] == 3
    assert int(plan.forest_mask.sum()) == nodes - 3
    assert len(plan.chord_indices) == edges.shape[1] - nodes + 3
    z = _explicit_basis(plan)
    incidence = torch.zeros(edges.shape[1], nodes, dtype=torch.float64)
    incidence[torch.arange(edges.shape[1]), edges[0]] = -1
    incidence[torch.arange(edges.shape[1]), edges[1]] = 1
    assert torch.equal(incidence.T @ z, torch.zeros(nodes, z.shape[1], dtype=torch.float64))
    assert torch.linalg.matrix_rank(z) == z.shape[1]
    assert torch.equal(z.abs().sum(0).long(), plan.cycle_length)
    assert torch.equal(z.abs().sum(1).long(), plan.cycle_count)
    assert torch.equal(plan.incidence_edge_index, edges)


@pytest.mark.parametrize("signed", [False, True])
def test_implicit_operators_match_explicit_small_oracle_and_are_adjoints(signed):
    nodes, edges = _fixture()
    plan = build_topology(nodes, edges)
    z = _explicit_basis(plan)
    if not signed:
        z = z.abs()
    values = torch.randn(edges.shape[1], 3, dtype=torch.float64, requires_grad=True)
    cycles = torch.randn(z.shape[1], 3, dtype=torch.float64, requires_grad=True)
    forward = edge_to_cycle(values, plan, signed=signed)
    backward = cycle_to_edge(cycles, plan, signed=signed)
    torch.testing.assert_close(forward, z.T @ values)
    torch.testing.assert_close(backward, z @ cycles)
    torch.testing.assert_close((forward * cycles).sum(), (values * backward).sum())
    assert torch.autograd.gradcheck(lambda x: edge_to_cycle(x, plan, signed=signed), (values,))
    assert torch.autograd.gradcheck(lambda x: cycle_to_edge(x, plan, signed=signed), (cycles,))


def test_unsigned_normalized_context_matches_dense_cycle_means_and_edge_means():
    nodes, edges = _fixture()
    plan = build_topology(nodes, edges)
    z = _explicit_basis(plan).abs()
    value = torch.randn(edges.shape[1], 2, dtype=torch.float64, requires_grad=True)
    reference = (z @ ((z.T @ value) / plan.cycle_length[:, None])) / plan.cycle_count[
        :, None
    ].clamp_min(1)
    torch.testing.assert_close(cycle_context(value, plan), reference)
    torch.testing.assert_close(cycle_context(value, plan, normalize=False), z @ z.T @ value)
    assert torch.autograd.gradcheck(lambda x: cycle_context(x, plan), (value,))


def test_orientation_and_column_permutation_do_not_change_unsigned_context_or_forest():
    nodes, edges = _fixture()
    perm = torch.tensor([7, 1, 4, 5, 0, 3, 2, 6])
    permuted = edges.flip(0)[:, perm]
    first, second = (
        build_topology(nodes, edges, forest_seed=9),
        build_topology(nodes, permuted, forest_seed=9),
    )
    assert torch.equal(second.forest_mask, first.forest_mask[perm])
    assert torch.equal(second.random_priority, first.random_priority[perm])
    values = torch.randn(edges.shape[1], dtype=torch.float64)
    torch.testing.assert_close(
        cycle_context(values[perm], second), cycle_context(values, first)[perm]
    )


def test_cached_ppi_plans_batch_without_rebuilding_dfs_or_mixing_graphs(monkeypatch):
    nodes, edges = _fixture()
    plans = [build_topology(nodes, edges, forest_seed=seed) for seed in (2, 4)]
    import research.conductance_gat.edge_selection.topology as topology

    monkeypatch.setattr(topology, "build_topology", lambda *a, **k: pytest.fail("DFS reexecuted"))
    batch = batch_topologies(plans)
    assert batch.metadata["num_graphs"] == 2 and batch.metadata["dfs_reexecuted"] is False
    assert batch.num_graphs == 2
    assert torch.equal(batch.edge_graph, torch.tensor([0] * 8 + [1] * 8))
    assert torch.equal(batch.incidence_edge_index[:, 8:], edges + nodes)
    values = [torch.randn(8, dtype=torch.float64) for _ in plans]
    expected = torch.cat(
        [cycle_context(value, plan) for value, plan in zip(values, plans, strict=True)]
    )
    torch.testing.assert_close(cycle_context(torch.cat(values), batch), expected)
    assert torch.equal(batch.cycle_count, torch.cat([plan.cycle_count for plan in plans]))


def test_bridge_and_isolate_have_no_cycle_not_a_fake_capped_cycle():
    plan = build_topology(5, torch.tensor([[0, 1], [1, 2]]))
    assert plan.metadata["cycle_rank"] == 0
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    result = cycle_context(value, plan)
    assert torch.equal(result, torch.zeros(2))
    result.sum().backward()
    assert torch.equal(value.grad, torch.zeros(2))
    empty = build_topology(0, torch.empty((2, 0), dtype=torch.long))
    assert cycle_context(torch.empty(0), empty).numel() == 0


def test_plan_device_copy_and_pickle_keep_cache_independent():
    nodes, edges = _fixture()
    plan = build_topology(nodes, edges)
    copied = plan.to("cpu")
    assert copied.incidence_edge_index.data_ptr() != plan.incidence_edge_index.data_ptr()
    copied.forest_mask.logical_not_()
    assert not torch.equal(copied.forest_mask, plan.forest_mask)
    restored = pickle.loads(pickle.dumps(plan))
    assert torch.equal(restored.parent_edge, plan.parent_edge)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.parent = torch.zeros(nodes, dtype=torch.long)


@pytest.mark.parametrize(
    "edges,graphs",
    [
        (torch.tensor([[0], [0]]), None),
        (torch.tensor([[0, 1], [1, 0]]), None),
        (torch.tensor([[0], [5]]), None),
        (torch.tensor([[0], [2]]), torch.tensor([0, 0, 1, 1, 1])),
    ],
)
def test_invalid_physical_topology_rejected(edges, graphs):
    with pytest.raises(ValueError):
        build_topology(5, edges, graphs)


def test_cycle_rank_is_not_capped_and_plan_storage_is_linear():
    nodes = 200
    edges = torch.triu_indices(nodes, nodes, 1)
    plan = build_topology(nodes, edges)
    assert len(plan.chord_indices) == edges.shape[1] - nodes + 1
    elements = sum(
        value.numel() for value in vars(plan).values() if isinstance(value, torch.Tensor)
    )
    assert elements < 20 * (nodes + edges.shape[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_inputs_use_stable_accumulation_and_keep_output_dtype(dtype):
    nodes, edges = _fixture()
    plan = build_topology(nodes, edges)
    value = torch.tensor(
        [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=dtype, requires_grad=True
    )
    result = cycle_context(value, plan)
    assert result.dtype == dtype and torch.isfinite(result).all()
    torch.testing.assert_close(result, cycle_context(value.double(), plan).to(dtype))
    result.float().square().sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
