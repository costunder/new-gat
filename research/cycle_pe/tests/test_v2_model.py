"""Bounded forward/backward unit fixtures, never benchmark training or datasets."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from research.cycle_pe.paper_model import _MessageLayer
from research.cycle_pe.v2.data import Graph, collate, prepare_graph
from research.cycle_pe.v2.model import (
    MODEL_NAME,
    CycleBasisPEModel,
    LeftNullBasisEncoder,
    architecture_protocol,
)


def _graph(n: int = 4, *, complete: bool = False, forest: bool = False) -> Graph:
    if complete:
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    elif forest:
        edges = [(i, i + 1) for i in range(n - 1)]
    else:
        edges = sorted({tuple(sorted((i, (i + 1) % n))) for i in range(n)})
    return prepare_graph(
        SimpleNamespace(
            num_nodes=n,
            x=torch.arange(n).reshape(-1, 1),
            edge_index=torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long)
            .reshape(-1, 2)
            .T.contiguous(),
            edge_attr=torch.ones((2 * len(edges), 1), dtype=torch.long),
            y=torch.tensor([0.7]),
        )
    )


def test_raw_signed_coordinates_reach_first_learned_layer_without_truncation() -> None:
    graph = _graph(7, complete=True)
    basis = graph.cycle_basis
    encoder = LeftNullBasisEncoder(5, 8, column_chunk_size=4)
    observed = []
    hook = encoder.column_phi[0].register_forward_pre_hook(
        lambda _module, args: observed.append(args[0][:, :, -1].detach().clone())
    )
    output = encoder(torch.randn(len(basis), 5), basis)
    hook.remove()
    assert basis.shape == (21, 15)
    assert output.shape == (21, 8)
    assert max(value.shape[1] for value in observed) <= 4
    torch.testing.assert_close(torch.cat(observed[::2], dim=1), basis)
    torch.testing.assert_close(torch.cat(observed[1::2], dim=1), -basis)


def test_every_basis_column_participates_in_autograd_and_parameters_are_rank_independent() -> None:
    torch.manual_seed(23)
    encoder = LeftNullBasisEncoder(7, 11, column_chunk_size=3)
    parameters_before = sum(p.numel() for p in encoder.parameters())
    for graph in (_graph(3), _graph(7, complete=True)):
        basis = graph.cycle_basis.clone().requires_grad_()
        output = encoder(torch.randn(len(basis), 7), basis)
        output.square().sum().backward()
        assert basis.grad is not None and torch.isfinite(basis.grad).all()
        assert (basis.grad.abs().sum(dim=0) > 0).all()
        assert sum(p.numel() for p in encoder.parameters()) == parameters_before


def test_chunk_size_changes_allocation_not_values_or_gradients() -> None:
    torch.manual_seed(7)
    graph = _graph(6, complete=True)
    encoder = LeftNullBasisEncoder(5, 9, column_chunk_size=1)
    bond = torch.randn(len(graph.edge_attr), 5)
    basis = graph.cycle_basis.clone().requires_grad_()
    expected = encoder(bond, basis)
    expected.sum().backward()
    gradient = basis.grad.clone()
    basis.grad = None
    encoder.column_chunk_size = 4
    actual = encoder(bond, basis)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(basis.grad, gradient, atol=2e-6, rtol=2e-6)


def test_column_sign_and_order_symmetry_after_nonlinear_column_encoding() -> None:
    torch.manual_seed(13)
    graph = _graph(6, complete=True)
    encoder = LeftNullBasisEncoder(7, 11, column_chunk_size=3)
    bond = torch.randn(len(graph.edge_attr), 7)
    rank = graph.cycle_basis.shape[1]
    permutation = torch.randperm(rank)
    signs = torch.where(torch.arange(rank) % 2 == 0, 1.0, -1.0)
    changed = graph.cycle_basis[:, permutation] * signs
    torch.testing.assert_close(encoder(bond, graph.cycle_basis), encoder(bond, changed))


def test_relative_sign_structure_is_not_replaced_by_entrywise_absolute_values() -> None:
    torch.manual_seed(17)
    encoder = LeftNullBasisEncoder(7, 8)
    bond = torch.randn(4, 7)
    basis = torch.tensor([[0.2], [-0.3], [0.4], [-0.5]])
    changed = basis.clone()
    changed[1] *= -1
    torch.testing.assert_close(basis.abs(), changed.abs())
    assert not torch.allclose(encoder(bond, basis), encoder(bond, changed), atol=1e-8, rtol=1e-7)


def test_edge_order_equivariance_with_transported_full_basis() -> None:
    torch.manual_seed(11)
    graph = _graph(5, complete=True)
    encoder = LeftNullBasisEncoder(7, 8)
    bond = torch.randn(len(graph.edge_attr), 7)
    order = torch.randperm(len(bond))
    torch.testing.assert_close(
        encoder(bond, graph.cycle_basis)[order],
        encoder(bond[order], graph.cycle_basis[order]),
    )


def test_forest_has_exact_zero_pe_even_with_nonzero_mlp_biases() -> None:
    encoder = LeftNullBasisEncoder(5, 8)
    for parameter in encoder.parameters():
        torch.nn.init.constant_(parameter, 0.3)
    for edges in (0, 4):
        value = encoder(torch.randn(edges, 5), torch.empty(edges, 0))
        assert value.shape == (edges, 8)
        assert torch.equal(value, torch.zeros_like(value))


def test_full_model_ragged_batch_matches_individual_graphs_and_backpropagates() -> None:
    torch.manual_seed(5)
    model = CycleBasisPEModel(dataset="zinc12k", hidden=12, pe_dim=6, layers=2, column_chunk_size=2)
    graphs = [_graph(4), _graph(5, complete=True), _graph(3, forest=True)]
    batch = collate(graphs)
    output = model(batch)
    assert output.shape == (3, 1)
    assert all(isinstance(layer, _MessageLayer) for layer in model.layers)
    (output - batch.y).abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        combined = model(batch)
        separate = torch.cat([model(collate([graph])) for graph in graphs])
    torch.testing.assert_close(combined, separate, atol=3e-6, rtol=3e-6)


def test_full_model_is_node_permutation_invariant_only_with_transported_chart() -> None:
    torch.manual_seed(4)
    graph = _graph(5, complete=True)
    model = CycleBasisPEModel(dataset="zinc12k", hidden=12, pe_dim=6, layers=2).eval()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(permutation)
    transported = replace(graph, x=graph.x[permutation], edge_index=inverse[graph.edge_index])
    # Keep each incidence edge's original orientation and its entire U row.
    torch.testing.assert_close(model(collate([graph])), model(collate([transported])))


def test_optional_amp_keeps_full_basis_and_scatter_arithmetic_valid() -> None:
    model = CycleBasisPEModel(dataset="zinc12k", hidden=12, pe_dim=6, layers=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(collate([_graph(4), _graph(5, complete=True)]))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dataset,width,targets", [("zinc12k", 1, 1), ("peptides_struct", 9, 11)])
def test_official_target_width_parameter_budget_and_edgeless_readout(dataset, width, targets):
    model = CycleBasisPEModel(dataset=dataset)
    graph = _graph(1, forest=True)
    graph.x = torch.zeros((1, width), dtype=torch.long)
    graph.edge_attr = torch.empty((0, 1 if dataset == "zinc12k" else 3), dtype=torch.long)
    graph.y = torch.zeros(targets)
    output = model(collate([graph]))
    assert output.shape == (1, targets)
    assert torch.isfinite(output).all()
    assert sum(p.numel() for p in model.parameters()) <= 500_000


@pytest.mark.parametrize("kwargs", [{"bond_dim": 0}, {"pe_dim": 0}, {"column_chunk_size": 0}])
def test_encoder_rejects_invalid_sizes(kwargs):
    arguments = {"bond_dim": 5, "pe_dim": 8, **kwargs}
    with pytest.raises(ValueError, match="positive"):
        LeftNullBasisEncoder(**arguments)


def test_basis_schema_errors_fail_loudly() -> None:
    encoder = LeftNullBasisEncoder(5, 8)
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.zeros(3, 5), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="floating point"):
        encoder(torch.zeros(3, 5), torch.zeros(3, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="edgeless"):
        encoder(torch.zeros(0, 5), torch.zeros(0, 1))


def test_protocol_names_actual_basis_input_and_states_gauge_and_compression_limits() -> None:
    protocol = architecture_protocol()
    assert MODEL_NAME == "cycle_basis_v2"
    assert "full signed left-nullspace basis" in protocol["positional_encoding"]
    assert "no train-fitted padding, truncation" in protocol["basis_width"]
    assert "not arbitrary O(beta)" in protocol["cycle_symmetry"]
    assert "not guaranteed injective" in protocol["limits"]


def _disconnected_graph() -> Graph:
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    return prepare_graph(
        SimpleNamespace(
            num_nodes=6,
            x=torch.arange(6).reshape(-1, 1),
            edge_index=torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).T,
            edge_attr=torch.ones((2 * len(edges), 1), dtype=torch.long),
            y=torch.tensor([0.7]),
        )
    )


def _assert_parameter_gradients_match(first: torch.nn.Module, second: torch.nn.Module) -> None:
    actual = dict(first.named_parameters())
    expected = dict(second.named_parameters())
    assert actual.keys() == expected.keys()
    for name, parameter in actual.items():
        wanted = expected[name]
        assert (parameter.grad is None) == (wanted.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, wanted.grad, atol=3e-6, rtol=3e-5, msg=name)


@pytest.mark.parametrize("budget", [1, 5, 29, 32768])
def test_batched_pair_encoder_matches_reference_outputs_and_every_gradient(budget):
    torch.manual_seed(37)
    graphs = [
        _graph(4),
        _graph(5, complete=True),
        _graph(4, forest=True),
        _graph(1, forest=True),
        _disconnected_graph(),
    ]
    counts = [len(graph.edge_attr) for graph in graphs]
    reference = LeftNullBasisEncoder(5, 9, column_chunk_size=3)
    batched = copy.deepcopy(reference)
    ref_bond = torch.randn(sum(counts), 5, requires_grad=True)
    new_bond = ref_bond.detach().clone().requires_grad_()
    ref_bases = tuple(graph.cycle_basis.clone().requires_grad_() for graph in graphs)
    new_bases = tuple(basis.detach().clone().requires_grad_() for basis in ref_bases)
    expected = torch.cat(
        [
            reference(part, basis)
            for part, basis in zip(ref_bond.split(counts), ref_bases, strict=True)
        ]
    )
    actual = batched.forward_batch(new_bond, new_bases, pair_budget=budget)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(new_bond.grad, ref_bond.grad, atol=3e-6, rtol=3e-5)
    for wanted, value in zip(ref_bases, new_bases, strict=True):
        assert (wanted.grad is None) == (value.grad is None)
        if wanted.grad is not None:
            torch.testing.assert_close(value.grad, wanted.grad, atol=3e-6, rtol=3e-5)
            assert (value.grad.abs().sum(dim=0) > 0).all()
    _assert_parameter_gradients_match(batched, reference)
    assert reference.state_dict().keys() == batched.state_dict().keys()


@pytest.mark.parametrize("budget", [1, 7, 32768])
def test_batched_encoder_keeps_graph_column_segments_signs_and_edge_order_separate(budget):
    torch.manual_seed(31)
    encoder = LeftNullBasisEncoder(5, 9, column_chunk_size=3)
    bases = tuple(graph.cycle_basis for graph in [_graph(4, complete=True), _disconnected_graph()])
    bonds = [torch.randn(len(basis), 5) for basis in bases]
    expected = encoder.forward_batch(torch.cat(bonds), bases, pair_budget=budget)
    changed, orders = [], []
    for basis in bases:
        rank = basis.shape[1]
        columns, order = torch.randperm(rank), torch.randperm(len(basis))
        signs = torch.where(torch.arange(rank) % 2 == 0, 1.0, -1.0)
        changed.append(basis[order][:, columns] * signs)
        orders.append(order)
    actual = encoder.forward_batch(
        torch.cat([bond[order] for bond, order in zip(bonds, orders, strict=True)]),
        changed,
        pair_budget=budget,
    )
    expected_parts = expected.split([len(basis) for basis in bases])
    torch.testing.assert_close(
        actual,
        torch.cat([part[order] for part, order in zip(expected_parts, orders, strict=True)]),
        atol=3e-6,
        rtol=3e-5,
    )
    # A different graph's features must never change the first graph's context.
    altered_bonds = torch.cat((bonds[0], 20.0 * bonds[1]))
    altered = encoder.forward_batch(altered_bonds, bases, pair_budget=budget)
    torch.testing.assert_close(altered[: len(bases[0])], expected[: len(bases[0])])


def test_batched_encoder_shares_mlp_calls_and_honors_pair_budget_without_truncation() -> None:
    encoder = LeftNullBasisEncoder(5, 9, column_chunk_size=2)
    bases = tuple(_graph(5, complete=True).cycle_basis for _ in range(3))
    bond = torch.randn(sum(len(basis) for basis in bases), 5)
    pair_count = sum(basis.numel() for basis in bases)
    calls: dict[str, list[torch.Tensor]] = {"phi": [], "psi": []}
    hooks = [
        module.register_forward_pre_hook(
            lambda _module, args, key=key: calls[key].append(args[0].detach().clone())
        )
        for key, module in (("phi", encoder.column_phi[0]), ("psi", encoder.edge_psi[0]))
    ]
    try:
        encoder.forward_batch(bond, bases, pair_budget=pair_count)
        assert len(calls["phi"]) == len(calls["psi"]) == 2
        # One shared positive and one shared negative call cover all entries.
        assert sum(len(value) for value in calls["phi"]) == 2 * pair_count
        assert calls["phi"][0].shape == (pair_count, 6)
        torch.testing.assert_close(calls["phi"][0][:, -1], -calls["phi"][1][:, -1])
        for values in calls.values():
            values.clear()
        encoder.forward_batch(bond, bases, pair_budget=7)
        for values in calls.values():
            assert max(len(value) for value in values) <= 7
            assert sum(len(value) for value in values) == 2 * pair_count
    finally:
        for hook in hooks:
            hook.remove()


def test_batched_forest_and_empty_inputs_never_create_bias_pe() -> None:
    encoder = LeftNullBasisEncoder(5, 8)
    for parameter in encoder.parameters():
        torch.nn.init.constant_(parameter, 0.3)
    for bases in ((), (torch.empty(0, 0),), (torch.empty(4, 0), torch.empty(0, 0))):
        edges = sum(len(basis) for basis in bases)
        value = encoder.forward_batch(torch.randn(edges, 5), bases, pair_budget=1)
        assert torch.equal(value, torch.zeros(edges, 8))


def test_full_batched_model_matches_reference_state_dict_outputs_and_gradients() -> None:
    torch.manual_seed(47)
    reference = CycleBasisPEModel(
        dataset="zinc12k",
        hidden=12,
        pe_dim=6,
        layers=2,
        column_chunk_size=2,
        basis_execution="reference",
    )
    batched = CycleBasisPEModel(
        dataset="zinc12k",
        hidden=12,
        pe_dim=6,
        layers=2,
        column_chunk_size=2,
        basis_execution="batched",
        basis_pair_budget=7,
    )
    batched.load_state_dict(reference.state_dict(), strict=True)
    graphs = [_graph(4), _graph(5, complete=True), _graph(4, forest=True), _disconnected_graph()]
    batch = collate(graphs)
    expected, actual = reference(batch), batched(batch)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.randn_like(actual)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    _assert_parameter_gradients_match(batched, reference)


@pytest.mark.parametrize("kwargs", [{"basis_execution": "unknown"}, {"basis_pair_budget": 0}])
def test_full_model_rejects_invalid_execution_settings(kwargs):
    with pytest.raises(ValueError, match="basis_"):
        CycleBasisPEModel(dataset="zinc12k", **kwargs)


def test_batched_encoder_rejects_invalid_schema_and_budget() -> None:
    encoder = LeftNullBasisEncoder(5, 8)
    with pytest.raises(ValueError, match="positive"):
        encoder.forward_batch(torch.zeros(3, 5), (torch.zeros(3, 1),), pair_budget=0)
    with pytest.raises(ValueError, match="align"):
        encoder.forward_batch(torch.zeros(3, 5), (torch.zeros(2, 1),))
    with pytest.raises(ValueError, match="shape"):
        encoder.forward_batch(torch.zeros(3, 5), (torch.zeros(3),))
    with pytest.raises(ValueError, match="floating point"):
        encoder.forward_batch(torch.zeros(3, 5), (torch.zeros(3, 1, dtype=torch.long),))
    with pytest.raises(ValueError, match="edgeless"):
        encoder.forward_batch(torch.zeros(0, 5), (torch.zeros(0, 1),))
