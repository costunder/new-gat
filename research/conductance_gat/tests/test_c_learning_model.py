"""Pure tensor tests of the isolated learned-C contribution, not GPU experiments."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.ablation import train as shared
from research.conductance_gat.ablation.model import (
    FactorialNodeClassifier,
    is_gate_parameter,
    state_sha256,
)
from research.conductance_gat.ablation.model import (
    make_optimizer as factorial_optimizer,
)
from research.conductance_gat.c_learning.model import CLearningNodeClassifier, make_optimizer
from research.conductance_gat.c_learning.protocol import CONDITIONS
from research.conductance_gat.c_learning.train import DEFINITION


def graph_fixture():
    return SimpleNamespace(
        x=torch.tensor([[0.2, 1.0, 2.0], [1.0, 2.0, 0.3], [3.0, 0.4, 1.0], [2.0, 3.0, 4.0]]),
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0], [1, 2]]),
    )


@pytest.mark.parametrize("gate_mode", ["learned", "fixed_one"])
def test_same_initial_full_state_rng_and_non_gate_parameters_as_node_degree(gate_mode):
    torch.manual_seed(903)
    original = FactorialNodeClassifier(3, 2, hidden_channels=8, normalization="node_degree")
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(903)
    model = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode=gate_mode)
    assert state_sha256(model) == state_sha256(original)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert list(model.state_dict()) == list(original.state_dict())
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (gate_mode == "learned" or not is_gate_parameter(name))


@pytest.mark.parametrize("training", [True, False])
def test_learned_arm_exactly_preserves_existing_node_degree_forward_gradients_and_adam(training):
    torch.manual_seed(62)
    original = FactorialNodeClassifier(3, 2, hidden_channels=8, normalization="node_degree")
    model = CLearningNodeClassifier(3, 2, hidden_channels=8)
    model.load_state_dict(original.state_dict())
    original.train(training)
    model.train(training)
    before_optimizer = factorial_optimizer(original, "node_degree")
    after_optimizer = make_optimizer(model, "learned_c")
    graph = graph_fixture()
    torch.manual_seed(31)
    expected = original(graph)
    expected.square().mean().backward()
    before_optimizer.step()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(31)
    actual = model(graph)
    actual.square().mean().backward()
    after_optimizer.step()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(original.named_parameters())[name].grad, atol=0, rtol=0
        )
    assert state_sha256(model) == state_sha256(original)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_fixed_operator_is_exact_unweighted_neighbor_mean_with_isolate_and_orientation():
    model = CLearningNodeClassifier(3, 2, hidden_channels=3, gate_mode="fixed_one")
    operator = model.operators[0]
    graph = graph_fixture()
    state = graph.x.clone().requires_grad_()
    actual = operator(state, graph.incidence_edge_index, torch.zeros(4).long(), 1)
    expected = torch.stack(
        [
            0.05 * state[0] + 0.95 * (state[1] + state[2]) / 2,
            0.05 * state[1] + 0.95 * state[0],
            0.05 * state[2] + 0.95 * state[0],
            state[3],
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[3], state[3], atol=0, rtol=0)
    reverse = operator(state, graph.incidence_edge_index.flip(0), torch.zeros(4).long(), 1)
    torch.testing.assert_close(reverse, actual, atol=0, rtol=0)
    weights = torch.arange(1, actual.numel() + 1).reshape_as(actual)
    grad = torch.autograd.grad((actual * weights).sum(), state, retain_graph=True)[0]
    ref_grad = torch.autograd.grad((expected * weights).sum(), state)[0]
    torch.testing.assert_close(grad, ref_grad, atol=1e-6, rtol=1e-6)


def test_fixed_gate_is_not_evaluated_and_effective_c_is_exact_one(monkeypatch):
    model = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode="fixed_one").eval()
    for operator in model.operators:
        monkeypatch.setattr(
            operator.estimator.network,
            "forward",
            lambda *args: pytest.fail("Frozen gate scaffold must never be evaluated"),
        )
    with shared.ForwardObservation(model) as observation:
        model(graph_fixture())
    for layer in observation.summary():
        assert layer["conductance"]["mean"] == 1.0
        assert layer["conductance"]["std"] == layer["conductance"]["cv"] == 0.0
        assert layer["conductance"]["count"] == 2
        assert layer["rho"]["isolated_node_count"] == 1
    gate = model.operators[0].estimator
    c = gate(torch.full((5, 8), float("nan"), requires_grad=True), torch.empty(5, 0))
    assert torch.equal(c, torch.ones(5)) and not c.requires_grad


def test_fixed_output_cannot_depend_on_frozen_scaffold_values():
    model = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode="fixed_one").eval()
    graph = graph_fixture()
    original = model(graph)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if is_gate_parameter(name):
                parameter.fill_(float("nan"))
    torch.testing.assert_close(model(graph), original, atol=0, rtol=0)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_optimizer_exactly_partitions_trainable_parameters_and_excludes_frozen(condition):
    model = CLearningNodeClassifier(
        3, 2, hidden_channels=8, **{"gate_mode": CONDITIONS[condition]["gate_mode"]}
    )
    optimizer = make_optimizer(model, condition)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    included = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(included) == len(set(included)) == len(expected)
    assert set(included) == expected
    assert all(group["weight_decay"] == 0.0005 for group in optimizer.param_groups)
    assert len(optimizer.param_groups) == (2 if condition == "learned_c" else 1)


def test_fixed_gate_no_grad_update_or_adam_state_and_telemetry_is_explicit():
    model = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode="fixed_one")
    before = copy.deepcopy(model.state_dict())
    optimizer = make_optimizer(model, "fixed_c")
    graph = graph_fixture()
    loss, _ = shared.training_loss(model(graph), graph, torch.tensor([0, 1]))
    loss.backward()
    telemetry = shared.gradient_observation(model, "fixed_c", definition=DEFINITION)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if is_gate_parameter(name):
            assert parameter.grad is None and parameter not in optimizer.state
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    assert not torch.equal(model.encoder.weight, before["encoder.weight"])
    gate = telemetry["operators.0"]
    assert gate["optimizer_included"] is False and gate["trainable_parameter_count"] == 0
    assert gate["weight_decay"] == gate["decay_term_norm"] == gate["task_gradient_norm"] == 0
    assert gate["task_to_decay_ratio"] is None
    assert telemetry["non_gate"]["weight_decay"] == 0.0005


@pytest.mark.parametrize("kwargs", [{"normalization": "global_max"}, {"gate_mode": "other"}])
def test_invalid_architecture_factor_rejected(kwargs):
    with pytest.raises(ValueError):
        CLearningNodeClassifier(3, 2, **kwargs)


def test_mismatched_optimizer_condition_rejected():
    model = CLearningNodeClassifier(3, 2, gate_mode="fixed_one")
    with pytest.raises(ValueError, match="disagree"):
        make_optimizer(model, "learned_c")
