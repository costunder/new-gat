"""Pure tensor equivalence checks, not CPU research training."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.ablation.model import (
    FactorialConductanceConv,
    FactorialNodeClassifier,
    is_gate_parameter,
    make_optimizer,
    state_sha256,
)
from research.conductance_gat.ablation.protocol import COMMON, CONDITIONS
from research.conductance_gat.benchmark import ConductanceNodeClassifier


def _graph():
    return SimpleNamespace(
        x=torch.randn(7, 3),
        incidence_edge_index=torch.tensor([[0, 0, 1, 3, 3], [1, 2, 2, 4, 5]]),
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        ptr=torch.tensor([0, 3, 7]),
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_all_conditions_have_exact_baseline_initial_state_and_rng(condition):
    torch.manual_seed(41)
    baseline = ConductanceNodeClassifier(3, 2, hidden_channels=64, layers=2, dropout=0.5)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    model = FactorialNodeClassifier(3, 2, normalization=CONDITIONS[condition]["normalization"])
    assert state_sha256(model) == state_sha256(baseline)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert list(model.state_dict()) == list(baseline.state_dict())
    for name, expected in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_global_max_matches_baseline_forward_and_every_gradient(training):
    torch.manual_seed(133)
    graph = _graph()
    baseline = ConductanceNodeClassifier(3, 2, hidden_channels=8, layers=2, dropout=0.5)
    model = FactorialNodeClassifier(3, 2, hidden_channels=8, layers=2, dropout=0.5)
    model.load_state_dict(baseline.state_dict())
    baseline.train(training)
    model.train(training)
    torch.manual_seed(876)
    expected = baseline(graph)
    expected.square().sum().backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(876)
    actual = model(graph)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(baseline.named_parameters())[name].grad, rtol=0, atol=0
        )


def test_baseline_parameter_groups_preserve_actual_adam_step_exactly():
    torch.manual_seed(190)
    graph = _graph()
    baseline = ConductanceNodeClassifier(3, 2, hidden_channels=8, layers=2, dropout=0)
    model = FactorialNodeClassifier(3, 2, hidden_channels=8, layers=2, dropout=0)
    model.load_state_dict(baseline.state_dict())
    old_optimizer = torch.optim.Adam(
        baseline.parameters(), lr=COMMON["lr"], weight_decay=COMMON["weight_decay"]
    )
    new_optimizer = make_optimizer(model, "baseline")
    baseline(graph).square().mean().backward()
    model(graph).square().mean().backward()
    old_optimizer.step()
    new_optimizer.step()
    assert state_sha256(model) == state_sha256(baseline)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_optimizer_changes_only_gate_decay_and_partitions_every_parameter(condition):
    model = FactorialNodeClassifier(3, 2, hidden_channels=8)
    optimizer = make_optimizer(model, condition)
    lookup = {id(p): name for name, p in model.named_parameters()}
    seen = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            seen.append(id(parameter))
            gate = is_gate_parameter(lookup[id(parameter)])
            expected = (
                CONDITIONS[condition]["gate_weight_decay"] if gate else COMMON["weight_decay"]
            )
            assert group["weight_decay"] == expected
            assert group["lr"] == COMMON["lr"]
    assert len(seen) == len(set(seen)) == len(lookup)
    assert set(seen) == set(lookup)


def test_node_degree_matches_dense_formula_including_gate_denominator_gradients():
    torch.manual_seed(17)
    operator = FactorialConductanceConv(3, "node_degree")
    reference = copy.deepcopy(operator)
    state = torch.randn(5, 3, requires_grad=True)
    ref_state = state.detach().clone().requires_grad_()
    edges = torch.tensor([[0, 0, 1, 1], [1, 2, 2, 3]])
    batch = torch.zeros(5, dtype=torch.long)
    incidence = torch.zeros(4, 5)
    incidence[torch.arange(4), edges[0]] = -1
    incidence[torch.arange(4), edges[1]] = 1
    difference = incidence @ ref_state
    c = reference.estimator(difference, state.new_empty((4, 0)))
    laplacian = incidence.T @ torch.diag(c) @ incidence
    degree = laplacian.diag()
    safe_degree = torch.where(degree > 0, degree, torch.ones_like(degree))
    expected = ref_state - 0.95 * (laplacian @ ref_state) / safe_degree[:, None]
    actual = operator(state, edges, batch, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(state.grad, ref_state.grad, atol=2e-6, rtol=2e-5)
    for parameter, ref_parameter in zip(operator.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("normalization", ["global_max", "node_degree"])
def test_orientation_constants_and_isolated_nodes(normalization):
    torch.manual_seed(13)
    operator = FactorialConductanceConv(3, normalization)
    edges = torch.tensor([[0, 0, 1], [1, 2, 3]])
    groups = torch.zeros(5, dtype=torch.long)
    state = torch.randn(5, 3)
    actual = operator(state, edges, groups, 1)
    reversed_output = operator(state, edges.flip(0), groups, 1)
    torch.testing.assert_close(actual, reversed_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[4], state[4], rtol=0, atol=0)
    constant = torch.ones_like(state) * 7
    torch.testing.assert_close(operator(constant, edges, groups, 1), constant, rtol=0, atol=0)


class _FixedConductance(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, difference, features):
        return difference.new_full((difference.shape[0],), self.value)


@pytest.mark.parametrize("normalization", ["global_max", "node_degree"])
def test_common_conductance_scale_still_cancels(normalization):
    operator = FactorialConductanceConv(1, normalization)
    state = torch.tensor([[1.0], [3.0], [7.0], [9.0]])
    edges = torch.tensor([[0, 0], [1, 2]])
    groups = torch.zeros(4, dtype=torch.long)
    operator.estimator = _FixedConductance(0.5)
    expected = operator(state, edges, groups, 1)
    operator.estimator = _FixedConductance(2.0)
    torch.testing.assert_close(operator(state, edges, groups, 1), expected, rtol=0, atol=0)


def test_node_degree_is_row_normalized_not_symmetric_and_keeps_isolate():
    operator = FactorialConductanceConv(1, "node_degree")
    operator.estimator = _FixedConductance(1.0)
    state = torch.tensor([[1.0], [3.0], [7.0], [9.0]])
    edges = torch.tensor([[0, 0], [1, 2]])
    actual = operator(state, edges, torch.zeros(4, dtype=torch.long), 1)
    expected = torch.tensor([[0.05 + 0.95 * 5], [0.05 * 3 + 0.95], [0.05 * 7 + 0.95], [9]])
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual.sum(), state.sum())


@pytest.mark.parametrize("normalization", ["global_max", "node_degree"])
def test_edgeless_graph_identity(normalization):
    operator = FactorialConductanceConv(2, normalization)
    state = torch.randn(4, 2)
    actual = operator(state, torch.empty((2, 0), dtype=torch.long), torch.zeros(4).long(), 1)
    torch.testing.assert_close(actual, state, atol=0, rtol=0)


def test_invalid_normalization_rejected():
    with pytest.raises(ValueError, match="Unsupported normalization"):
        FactorialConductanceConv(2, "detached_max")
