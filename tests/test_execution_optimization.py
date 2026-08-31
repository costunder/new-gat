"""CPU unit equivalence checks only; no datasets, optimizer, or research training."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor

from research.cycle_pe import benchmark_models, paper_model
from research.cycle_pe.benchmark_data import Batch
from research.cycle_pe.benchmark_models import (
    ATOM_DIMS,
    BOND_DIMS,
    CategoricalEncoder,
    CyclePEModel,
    _pool,
)
from research.cycle_pe.paper_model import _message_topology, _MessageLayer


def _legacy_categorical(model: CategoricalEncoder, values: Tensor) -> Tensor:
    return torch.stack([layer(values[:, i]) for i, layer in enumerate(model.embeddings)]).sum(0)


def _legacy_pool(values: Tensor, assignment: Tensor, count: int) -> tuple[Tensor, Tensor]:
    total = values.new_zeros((count, values.shape[1])).index_add(0, assignment, values)
    sizes = torch.bincount(assignment, minlength=count).clamp_min(1).unsqueeze(1)
    maximum = values.new_full((count, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(
        0, assignment[:, None].expand_as(values), values, reduce="amax", include_self=True
    )
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return total / sizes, maximum


def _legacy_message(
    model: _MessageLayer, node: Tensor, edge: Tensor, edge_index: Tensor
) -> tuple[Tensor, Tensor]:
    u, v = edge_index[:, 0], edge_index[:, 1]
    symmetric = torch.cat((node[u] + node[v], (node[u] - node[v]).abs(), edge), dim=1)
    updated_edge = model.edge_norm(edge + model.edge_update(symmetric))
    source = torch.cat((u, v), dim=0)
    target = torch.cat((v, u), dim=0)
    directed_edge = torch.cat((updated_edge, updated_edge), dim=0)
    messages = model.message(torch.cat((node[source], node[target], directed_edge), dim=1))
    aggregate = torch.zeros_like(node)
    aggregate.index_add_(0, target, messages)
    degree = torch.zeros(node.shape[0], device=node.device, dtype=node.dtype)
    degree.index_add_(0, target, torch.ones_like(target, dtype=node.dtype))
    aggregate = aggregate / degree.clamp_min(1.0)[:, None]
    updated_node = model.node_norm(node + model.node_update(torch.cat((node, aggregate), dim=1)))
    return updated_node, updated_edge


def _assert_gradients(actual: torch.nn.Module, expected: torch.nn.Module) -> None:
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for (name, parameter), (reference_name, reference) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == reference_name
        assert parameter.grad is not None, name
        assert reference.grad is not None, name
        tolerance = (
            {"rtol": 1e-12, "atol": 1e-12}
            if parameter.dtype == torch.float64
            else {"rtol": 3e-5, "atol": 3e-6}
        )
        torch.testing.assert_close(parameter.grad, reference.grad, **tolerance)


@pytest.mark.parametrize("cardinalities", [(28,), BOND_DIMS, ATOM_DIMS])
@pytest.mark.parametrize("rows", [0, 13])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_categorical_running_sum_matches_stack_values_and_parameter_gradients(
    cardinalities, rows, dtype, monkeypatch
):
    torch.manual_seed(102)
    model = CategoricalEncoder(cardinalities, 7).to(dtype=dtype)
    reference = copy.deepcopy(model)
    values = torch.stack([torch.randint(width, (rows,)) for width in cardinalities], dim=1)
    expected = _legacy_categorical(reference, values)

    def no_stack(*args, **kwargs):
        raise AssertionError("categorical encoding must not materialize a field stack")

    monkeypatch.setattr(torch, "stack", no_stack)
    actual = model(values)
    tolerance = (
        {"rtol": 1e-12, "atol": 1e-12}
        if dtype == torch.float64
        else {"rtol": 3e-6, "atol": 2e-6}
    )
    torch.testing.assert_close(actual, expected, **tolerance)
    weights = torch.randn_like(actual)
    (actual.cos() * weights).sum().backward()
    (expected.cos() * weights).sum().backward()
    _assert_gradients(model, reference)
    assert set(model.state_dict()) == {f"embeddings.{i}.weight" for i in range(len(cardinalities))}


@pytest.mark.parametrize("assignment,count", [([0, 0, 2, 2, 2], 4), ([], 3), ([], 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_shape_pool_matches_values_and_input_gradients(assignment, count, dtype, monkeypatch):
    torch.manual_seed(203)
    indices = torch.tensor(assignment, dtype=torch.long)
    values = torch.randn(len(indices), 5, dtype=dtype, requires_grad=True)
    reference = values.detach().clone().requires_grad_(True)
    expected = _legacy_pool(reference, indices, count)

    def no_bincount(*args, **kwargs):
        raise AssertionError("pool sizes must use the known graph count")

    monkeypatch.setattr(torch, "bincount", no_bincount)
    actual = _pool(values, indices, count)
    for result, original in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, original, rtol=0, atol=0)
        assert result.shape == (count, 5)
        assert torch.isfinite(result).all()
    weights = [torch.randn_like(part) for part in actual]
    sum((result * weight).sum() for result, weight in zip(actual, weights, strict=True)).backward()
    sum(
        (result * weight).sum() for result, weight in zip(expected, weights, strict=True)
    ).backward()
    torch.testing.assert_close(values.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("edges", [[], [(0, 1)], [(0, 1), (1, 2), (3, 4)]])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_message_topology_reuse_preserves_outputs_input_and_parameter_gradients(
    edges, prepared, dtype
):
    torch.manual_seed(304)
    model = _MessageLayer(5).to(dtype=dtype)
    reference = copy.deepcopy(model)
    # Node 5 remains isolated, including when every graph has no edges.
    node = torch.randn(6, 5, dtype=dtype, requires_grad=True)
    edge = torch.randn(len(edges), 5, dtype=dtype, requires_grad=True)
    reference_node = node.detach().clone().requires_grad_(True)
    reference_edge = edge.detach().clone().requires_grad_(True)
    indices = torch.tensor(edges, dtype=torch.long).reshape(-1, 2)
    topology = _message_topology(node, indices) if prepared else None
    actual = model(node, edge, indices, topology=topology)
    expected = _legacy_message(reference, reference_node, reference_edge, indices)
    for result, original in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, original, rtol=0, atol=0)
    weights = [torch.randn_like(part) for part in actual]
    sum((result * weight).sum() for result, weight in zip(actual, weights, strict=True)).backward()
    sum(
        (result * weight).sum() for result, weight in zip(expected, weights, strict=True)
    ).backward()
    torch.testing.assert_close(node.grad, reference_node.grad, rtol=0, atol=0)
    torch.testing.assert_close(edge.grad, reference_edge.grad, rtol=0, atol=0)
    _assert_gradients(model, reference)


def _batch(dataset: str, *, empty_edges: bool) -> Batch:
    atom_dims, bond_dims = ((28,), (4,)) if dataset == "zinc12k" else (ATOM_DIMS, BOND_DIMS)
    edges = [] if empty_edges else [(0, 1), (1, 2), (3, 4)]
    return Batch(
        x=torch.stack([torch.randint(width, (6,)) for width in atom_dims], dim=1),
        edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T,
        edge_attr=torch.stack([torch.randint(width, (len(edges),)) for width in bond_dims], dim=1),
        y=torch.zeros(3, 1 if dataset == "zinc12k" else 11),
        cycle_set=torch.randn(len(edges), 6, requires_grad=True),
        batch=torch.tensor([0, 0, 0, 1, 1, 2]),
        ptr=torch.tensor([0, 3, 5, 6]),
    )


def _legacy_cycle_forward(model: CyclePEModel, batch: Batch) -> Tensor:
    node = _legacy_categorical(model.node_encoder, batch.x)
    edge = model.edge_encoder(
        torch.cat(
            (
                _legacy_categorical(model.bond_encoder, batch.edge_attr),
                model.pe_encoder(batch.cycle_set),
            ),
            dim=1,
        )
    )
    for layer in model.layers:
        node, edge = _legacy_message(layer, node, edge, batch.edge_index.T)
    node_mean, node_max = _legacy_pool(node, batch.batch, len(batch.ptr) - 1)
    edge_mean, edge_max = _legacy_pool(edge, batch.batch[batch.edge_index[0]], len(batch.ptr) - 1)
    pooled = torch.cat((node_mean, node_max, edge_mean, edge_max), dim=1)
    return model.graph_head(model.graph_trunk(pooled))


@pytest.mark.parametrize("dataset", ["zinc12k", "peptides_struct"])
@pytest.mark.parametrize("empty_edges", [False, True])
def test_shared_model_optimization_preserves_complete_forward_and_backward(
    dataset, empty_edges, monkeypatch
):
    torch.manual_seed(405)
    model = CyclePEModel(dataset=dataset, hidden=8, pe_dim=4, layers=3)
    reference = copy.deepcopy(model)
    reference.load_state_dict(model.state_dict(), strict=True)
    batch = _batch(dataset, empty_edges=empty_edges)
    reference_batch = copy.deepcopy(batch)
    topology_calls = []

    def prepare_once(node, indices):
        topology_calls.append(indices.shape[0])
        return _message_topology(node, indices)

    def no_layer_recompute(*args, **kwargs):
        raise AssertionError("layer stack must reuse the forward's prepared connectivity")

    monkeypatch.setattr(benchmark_models, "_message_topology", prepare_once)
    monkeypatch.setattr(paper_model, "_message_topology", no_layer_recompute)
    actual = model(batch)
    expected = _legacy_cycle_forward(reference, reference_batch)
    assert topology_calls == [batch.edge_index.shape[1]]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    weights = torch.randn_like(actual)
    (actual.square() * weights).sum().backward()
    (expected.square() * weights).sum().backward()
    torch.testing.assert_close(
        batch.cycle_set.grad, reference_batch.cycle_set.grad, rtol=3e-5, atol=3e-6
    )
    _assert_gradients(model, reference)
