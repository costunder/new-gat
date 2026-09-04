"""Pure tensor tests of the isolated learned-C contribution, not GPU experiments."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.ablation import train as shared
from research.conductance_gat.ablation.model import (
    FactorialNodeClassifier,
    is_gate_parameter,
    shared_backbone_state_sha256,
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
def test_same_constructor_rng_and_non_gate_initialization_as_node_degree(gate_mode):
    torch.manual_seed(903)
    original = FactorialNodeClassifier(3, 2, hidden_channels=8, normalization="node_degree")
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(903)
    model = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode=gate_mode)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert shared_backbone_state_sha256(model) == shared_backbone_state_sha256(original)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0)
    if gate_mode == "learned":
        assert state_sha256(model) == state_sha256(original)
        assert list(model.state_dict()) == list(original.state_dict())
    else:
        assert state_sha256(model) != state_sha256(original)
        assert not any(".estimator." in name for name in model.state_dict())
    for _name, parameter in model.named_parameters():
        assert parameter.requires_grad


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


def test_fixed_gate_is_parameter_free_and_effective_c_is_exact_one():
    model = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode="fixed_one").eval()
    for operator in model.operators:
        assert list(operator.estimator.parameters()) == []
        assert list(operator.estimator.buffers()) == []
        assert operator.estimator.state_dict() == {}
        assert not hasattr(operator.estimator, "network")
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


def test_fixed_state_has_no_estimator_scaffold_and_exact_parameter_delta():
    torch.manual_seed(41)
    learned = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode="learned")
    torch.manual_seed(41)
    fixed = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode="fixed_one")
    assert not any(is_gate_parameter(name) for name, _ in fixed.named_parameters())
    assert not any(".estimator." in name for name in fixed.state_dict())
    learned_gate = sum(
        parameter.numel()
        for name, parameter in learned.named_parameters()
        if is_gate_parameter(name)
    )
    assert sum(p.numel() for p in learned.parameters()) - sum(
        p.numel() for p in fixed.parameters()
    ) == learned_gate
    assert shared_backbone_state_sha256(learned) == shared_backbone_state_sha256(fixed)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_optimizer_exactly_owns_every_and_only_trainable_parameter(condition):
    model = CLearningNodeClassifier(
        3, 2, hidden_channels=8, **{"gate_mode": CONDITIONS[condition]["gate_mode"]}
    )
    optimizer = make_optimizer(model, condition)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    included = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(included) == len(set(included)) == len(expected)
    assert set(included) == expected
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(group["weight_decay"] == 0.0005 for group in optimizer.param_groups)
    assert len(optimizer.param_groups) == (2 if condition == "learned_c" else 1)


def test_fixed_gate_no_grad_update_or_adam_state_and_telemetry_is_explicit():
    model = CLearningNodeClassifier(3, 2, hidden_channels=8, gate_mode="fixed_one")
    before_encoder = model.encoder.weight.detach().clone()
    optimizer = make_optimizer(model, "fixed_c")
    graph = graph_fixture()
    loss, _ = shared.training_loss(model(graph), graph, torch.tensor([0, 1]))
    loss.backward()
    telemetry = shared.gradient_observation(model, "fixed_c", definition=DEFINITION)
    optimizer.step()
    assert not any(is_gate_parameter(name) for name, _ in model.named_parameters())
    assert not torch.equal(model.encoder.weight, before_encoder)
    assert "operators.0" not in telemetry
    assert telemetry["non_gate"]["weight_decay"] == 0.0005
    owned = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert owned == {id(parameter) for parameter in model.parameters()}


@pytest.mark.parametrize("kwargs", [{"normalization": "global_max"}, {"gate_mode": "other"}])
def test_invalid_architecture_factor_rejected(kwargs):
    with pytest.raises(ValueError):
        CLearningNodeClassifier(3, 2, **kwargs)


def test_mismatched_optimizer_condition_rejected():
    model = CLearningNodeClassifier(3, 2, gate_mode="fixed_one")
    with pytest.raises(ValueError, match="disagree"):
        make_optimizer(model, "learned_c")
