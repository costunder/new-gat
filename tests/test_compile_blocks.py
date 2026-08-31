"""Dynamo graph-unit tests using CPU tensors and a counted eager graph backend.

These check compilation boundaries and autograd equivalence, not CUDA/Inductor
performance or research training. No optimizer or public dataset is used.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter

import pytest
import torch
import torch._dynamo

from chartgat.execution import configure_execution
from research.cycle_pe.v2.data import Graph, collate
from research.cycle_pe.v2.model import CycleBasisPEModel, LeftNullBasisEncoder


@pytest.fixture
def fresh_dynamo_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _configure_counted_cpu_blocks(model, monkeypatch):
    """Exercise production target selection, replacing only its hardware/backend."""
    counts = Counter()
    executions = Counter()
    active_module = [None]
    module_names = {id(module): name for name, module in model.named_modules()}
    original_compile = torch.compile
    state_keys = tuple(model.state_dict())

    def backend(graph_module, example_inputs):
        counts[active_module[0]] += 1

        def execute_graph(*args):
            executions[active_module[0]] += 1
            return graph_module.forward(*args)

        return execute_graph

    def record_module(module, args):
        active_module[0] = module_names[id(module)]

    def counted_compile(function, **kwargs):
        assert kwargs == {"backend": "inductor", "dynamic": True}
        assert function.__name__ == "forward"
        module = function.__self__
        # Reuse one backend like production's shared Inductor backend. Creating
        # a separate callable per block would itself cause BACKEND_MATCH guards.
        module.register_forward_pre_hook(record_module)
        return original_compile(function, backend=backend, dynamic=True)

    # Restore the real hardware predicate before any model execution. Only the
    # configuration guard is stubbed; all inputs/modules remain on the CPU.
    with monkeypatch.context() as context:
        context.setattr(torch.cuda, "is_available", lambda: True)
        context.setattr(torch, "compile", counted_compile)
        report = configure_execution(model, argparse.Namespace(compile=True), "cuda")
    assert report["scope"] == "tensor_mlp_blocks"
    assert tuple(model.state_dict()) == state_keys
    assert model._compiled_call_impl is None
    assert all(
        isinstance(model.get_submodule(name), torch.nn.Sequential)
        for name in report["compiled_modules"]
    )
    return counts, executions, report


def _compare_parameter_gradients(actual, reference):
    parameters = dict(actual.named_parameters())
    expected = dict(reference.named_parameters())
    assert parameters.keys() == expected.keys()
    for name, parameter in parameters.items():
        expected_gradient = expected[name].grad
        assert (parameter.grad is None) == (expected_gradient is None), name
        if expected_gradient is not None:
            torch.testing.assert_close(parameter.grad, expected_gradient, atol=3e-6, rtol=3e-5)


@pytest.mark.usefixtures("fresh_dynamo_cache")
def test_compiled_basis_mlp_blocks_keep_ten_ragged_shapes_outside_dynamo(monkeypatch, caplog):
    torch.manual_seed(601)
    reference = LeftNullBasisEncoder(3, 4, column_chunk_size=2)
    model = copy.deepcopy(reference)
    counts, executions, report = _configure_counted_cpu_blocks(model, monkeypatch)
    assert set(report["compiled_modules"]) == {"column_phi", "edge_psi", "output"}
    assert not counts  # Module.compile is lazy.
    middle_counts = None
    for edge_count in range(4, 14):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        bond = torch.randn(edge_count + 3, 3, requires_grad=True)
        reference_bond = bond.detach().clone().requires_grad_(True)
        bases = (
            torch.randn(edge_count, 2, requires_grad=True),
            torch.randn(3, 1, requires_grad=True),
        )
        reference_bases = tuple(basis.detach().clone().requires_grad_(True) for basis in bases)
        actual = model.forward_batch(bond, bases, pair_budget=5)
        expected = reference.forward_batch(reference_bond, reference_bases, pair_budget=5)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        weights = torch.randn_like(actual)
        (actual.square() * weights).sum().backward()
        (expected.square() * weights).sum().backward()
        torch.testing.assert_close(bond.grad, reference_bond.grad, atol=3e-6, rtol=3e-5)
        for basis, reference_basis in zip(bases, reference_bases, strict=True):
            torch.testing.assert_close(basis.grad, reference_basis.grad, atol=3e-6, rtol=3e-5)
        _compare_parameter_gradients(model, reference)
        if edge_count == 8:
            middle_counts = counts.copy()
    # Varying the ragged Python pair scheduler no longer compiles the complete
    # encoder for each shape. Size-one dimensions may have a separate graph.
    assert counts == middle_counts
    assert counts
    assert set(executions) == set(report["compiled_modules"])
    assert all(count <= 3 for count in counts.values()), counts
    assert "hit config.recompile_limit" not in caplog.text
    reference.load_state_dict(model.state_dict(), strict=True)


def _cycle_graph(nodes: int) -> Graph:
    edges = [(node, node + 1) for node in range(nodes - 1)] + [(0, nodes - 1)]
    basis = torch.ones(nodes, 1) / nodes**0.5
    basis[-1] = -basis[-1]
    return Graph(
        x=torch.randint(28, (nodes, 1)),
        edge_index=torch.tensor(edges, dtype=torch.long).T,
        edge_attr=torch.randint(4, (nodes, 1)),
        y=torch.zeros(1),
        cycle_basis=basis,
    )


@pytest.mark.usefixtures("fresh_dynamo_cache")
def test_compiled_full_model_blocks_preserve_forward_backward_and_empty_graphs(monkeypatch):
    torch.manual_seed(702)
    reference = CycleBasisPEModel(
        dataset="zinc12k", hidden=6, pe_dim=4, layers=2, basis_pair_budget=5
    )
    model = copy.deepcopy(reference)
    counts, executions, report = _configure_counted_cpu_blocks(model, monkeypatch)
    assert "pe_encoder.column_phi" in report["compiled_modules"]
    assert "layers.0.message" in report["compiled_modules"]
    assert "graph_trunk" in report["compiled_modules"]
    assert "pe_encoder" not in report["compiled_modules"]
    assert "layers.0" not in report["compiled_modules"]
    forest = Graph(
        x=torch.tensor([[1], [2]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.tensor([[0]]),
        y=torch.zeros(1),
        cycle_basis=torch.empty(1, 0),
    )
    edgeless = Graph(
        x=torch.tensor([[3]]),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, 1, dtype=torch.long),
        y=torch.zeros(1),
        cycle_basis=torch.empty(0, 0),
    )
    for nodes in (4, 7):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        batch = collate([_cycle_graph(nodes), forest, edgeless])
        actual = model(batch)
        expected = reference(batch)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        weights = torch.randn_like(actual)
        (actual.square() * weights).sum().backward()
        (expected.square() * weights).sum().backward()
        _compare_parameter_gradients(model, reference)
    assert counts
    assert set(executions) == set(report["compiled_modules"])
    assert all(count <= 3 for count in counts.values()), counts
    reference.load_state_dict(model.state_dict(), strict=True)
