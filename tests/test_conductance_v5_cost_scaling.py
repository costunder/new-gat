"""Synthetic CPU debug tests; not real-data training or performance evidence."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.nn import functional as F

from research.conductance_gat.v5.operator import graph_weighted_mean, shared_head_diffusion
from research.conductance_gat.v5.optimization import GraphOptimizedConductance


def _inputs(channels=5):
    generator = torch.Generator().manual_seed(731)
    x = torch.randn(9, channels, generator=generator, dtype=torch.float64)
    incidence = torch.tensor([[0, 0, 0, 1, 2, 4, 5, 5], [1, 2, 3, 2, 3, 5, 6, 7]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2])
    degree = torch.bincount(incidence.flatten(), minlength=9).double()
    full_degree = degree + torch.arange(9).double() % 3
    context = torch.randn(3, 2 * channels + 8, generator=generator, dtype=torch.float64)
    omega = torch.tensor([1.0, 2.5, 0.7, 1.3, 3.0, 1.2, 0.8, 2.1], dtype=torch.float64)
    return x, incidence, batch, degree, full_degree, context, omega


def _run(model, inputs, *, x=None, context=None):
    state, incidence, batch, degree, full_degree, z, omega = inputs
    return model(
        state if x is None else x,
        incidence,
        batch,
        z.shape[0],
        graph_context=z if context is None else context,
        sample_degree=degree,
        full_degree=full_degree,
        edge_normalization_weight=omega,
    )


def _model(channels=5, **kwargs):
    return GraphOptimizedConductance(
        channels, solver_cost_scaling="width_scaled", **kwargs
    ).double()


@pytest.mark.parametrize("invalid", [None, True, "auto", "standardized", 2, [], {}])
def test_cost_scaling_requires_an_explicit_known_recipe(invalid):
    with pytest.raises(ValueError, match="solver_cost_scaling"):
        GraphOptimizedConductance(5, solver_cost_scaling=invalid)


def test_legacy_default_is_bitwise_identical_and_adds_no_parameters():
    torch.manual_seed(19)
    inputs = _inputs()
    default = GraphOptimizedConductance(5).double()
    explicit = GraphOptimizedConductance(5, solver_cost_scaling="legacy_unit").double()
    explicit.load_state_dict(default.state_dict())
    expected = _run(default, inputs)
    actual = _run(explicit, inputs)
    assert torch.equal(expected, actual)
    assert torch.equal(default.last_scores, explicit.last_scores)
    weights = torch.arange(expected.numel(), dtype=expected.dtype)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    assert all(
        torch.equal(left.grad, right.grad)
        for left, right in zip(default.parameters(), explicit.parameters(), strict=True)
    )
    scaled = _model()
    assert scaled.state_dict().keys() == default.state_dict().keys()
    assert sum(p.numel() for p in scaled.parameters()) == sum(
        p.numel() for p in default.parameters()
    )
    assert default.quadratic_scale == 1.0
    assert scaled.quadratic_scale == math.sqrt(5)


def test_quadratic_only_is_width_scaled_then_complete_graph_centered_before_bound():
    torch.manual_seed(23)
    inputs = _inputs()
    model = _model(edge_chunk_size=2)
    state, incidence, batch, degree, full_degree, context, omega = inputs
    projected = F.normalize(model.node_projection(state), dim=1, eps=1e-6)
    metric = model.context_metric(F.layer_norm(context, (context.shape[1],))).tanh()
    edge_graph = batch[incidence[0]]
    arguments = (projected, metric, *incidence, degree, full_degree, edge_graph)
    raw_scaled = model._raw_compatibility_chunk(*arguments)
    legacy = GraphOptimizedConductance(5).double()
    legacy.load_state_dict(model.state_dict())
    raw_legacy = legacy._raw_compatibility_chunk(*arguments)
    quadratic = (
        (projected[incidence[0]] - projected[incidence[1]]).square() * metric[edge_graph]
    ).sum(1)
    torch.testing.assert_close(raw_scaled - raw_legacy, (math.sqrt(5) - 1) * quadratic)
    centered = raw_scaled - graph_weighted_mean(raw_scaled, edge_graph, 3, omega)[edge_graph]
    expected = model.cost_bound * torch.tanh(centered / model.cost_bound)
    _run(model, inputs)
    torch.testing.assert_close(model.last_scores, expected)
    assert model.last_solver_diagnostics["solver_cost_scaling"] == "width_scaled"
    assert model.last_solver_diagnostics["quadratic_scale"] == math.sqrt(5)
    assert model.last_solver_diagnostics["executed_steps"] == 8


def test_graph_constant_raw_cost_offsets_do_not_cause_saturation(monkeypatch):
    torch.manual_seed(27)
    inputs = _inputs()
    model = _model(edge_chunk_size=2)
    expected = _run(model, inputs)
    expected_scores = model.last_scores.clone()
    raw = model._raw_compatibility_chunk

    def shifted(*args):
        offsets = args[0].new_tensor([700.0, -900.0, 50.0])
        return raw(*args) + offsets[args[-1]]

    monkeypatch.setattr(model, "_raw_compatibility_chunk", shifted)
    actual = _run(model, inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(model.last_scores, expected_scores, rtol=1e-10, atol=1e-11)


def test_orientation_node_edge_permutation_and_disjoint_graph_independence():
    torch.manual_seed(41)
    inputs = _inputs()
    model = _model()
    expected = _run(model, inputs)
    reversed_inputs = list(inputs)
    reversed_inputs[1] = inputs[1].flip(0)
    torch.testing.assert_close(_run(model, reversed_inputs), expected, rtol=1e-11, atol=1e-12)
    permutation = torch.tensor([8, 4, 1, 6, 0, 3, 7, 2, 5])
    edge_permutation = torch.tensor([7, 3, 0, 5, 1, 6, 4, 2])
    changed = list(inputs)
    changed[0] = inputs[0][permutation]
    changed[1] = permutation.argsort()[inputs[1][:, edge_permutation]]
    for index in (2, 3, 4):
        changed[index] = inputs[index][permutation]
    changed[6] = inputs[6][edge_permutation]
    torch.testing.assert_close(
        _run(model, changed), expected[edge_permutation], rtol=1e-10, atol=1e-11
    )
    single = [
        inputs[0][:4],
        inputs[1][:, :5],
        torch.zeros(4, dtype=torch.long),
        inputs[3][:4],
        inputs[4][:4],
        inputs[5][:1],
        inputs[6][:5],
    ]
    torch.testing.assert_close(_run(model, single), expected[:5], rtol=1e-10, atol=1e-11)
    changed = list(inputs)
    changed[-1] = inputs[-1] * torch.tensor([1000.0] * 5 + [0.03] * 3, dtype=torch.float64)
    torch.testing.assert_close(_run(model, changed), expected, rtol=1e-10, atol=1e-11)
    mean = graph_weighted_mean(expected, inputs[2][inputs[1][0]], 3, inputs[-1])
    torch.testing.assert_close(mean, torch.tensor([1.0, 1.0, 0.0], dtype=expected.dtype))


def test_chunking_preserves_values_and_parameter_gradients():
    torch.manual_seed(17)
    inputs = _inputs()
    chunked = _model(edge_chunk_size=2)
    whole = copy.deepcopy(chunked)
    whole.edge_chunk_size = 65536
    actual, expected = _run(chunked, inputs), _run(whole, inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    weights = torch.arange(actual.numel(), dtype=actual.dtype)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for left, right in zip(chunked.parameters(), whole.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-11)


def test_task_gradient_reaches_scaled_cost_through_c_and_normalized_diffusion():
    torch.manual_seed(14)
    inputs = _inputs()
    x = inputs[0].clone().requires_grad_(True)
    context = inputs[5].clone().requires_grad_(True)
    model = _model(edge_chunk_size=2)
    c = _run(model, inputs, x=x, context=context)
    message = torch.randn(9, 2, 4, dtype=torch.float64, requires_grad=True)
    propagated = shared_head_diffusion(
        message,
        c,
        inputs[1],
        inputs[2],
        torch.full((3, 2), 0.5, dtype=torch.float64),
        sampling_correction=inputs[-1],
        edge_chunk_size=2,
    )
    (propagated - torch.randn_like(propagated)).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for value in (x, context, message):
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_double_gradcheck_through_scaled_cost_and_unrolled_solver():
    torch.manual_seed(13)
    inputs = _inputs(channels=3)
    model = _model(channels=3, solver_steps=3)
    assert torch.autograd.gradcheck(
        lambda x, context: _run(model, inputs, x=x, context=context),
        (inputs[0].requires_grad_(True), inputs[5].requires_grad_(True)),
        eps=1e-6,
        atol=1e-5,
        rtol=2e-4,
        fast_mode=True,
    )


def test_same_entropy_objective_converges_to_analytic_solution_without_barrier():
    torch.manual_seed(15)
    inputs = _inputs()
    model = _model(solver_degree_barrier=0, solver_steps=128)
    with torch.no_grad():
        actual = _run(model, inputs)
    raw = (-model.last_scores / model.solver_entropy).exp()
    edge_graph = inputs[2][inputs[1][0]]
    expected = raw / graph_weighted_mean(raw, edge_graph, 3, inputs[-1])[edge_graph]
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    diagnostics = model.last_solver_diagnostics
    assert diagnostics["objective_final"].le(diagnostics["objective_initial"] + 1e-12).all()
    assert diagnostics["projected_gradient_rms_final"].max() < 1e-9
    assert model.solver_entropy == 1.0 and model.cost_bound == 2.0


def test_constant_cost_does_not_force_nonuniform_c_and_fixed_arm_stays_parameter_free():
    nodes = torch.arange(8)
    incidence = torch.stack((nodes, (nodes + 1) % 8))
    state = torch.ones(8, 5, dtype=torch.float64)
    inputs = (
        state,
        incidence,
        torch.zeros(8, dtype=torch.long),
        torch.full((8,), 2.0).double(),
        torch.full((8,), 2.0).double(),
        torch.ones(1, 18, dtype=torch.float64),
        torch.ones(8, dtype=torch.float64),
    )
    model = _model()
    torch.testing.assert_close(_run(model, inputs), inputs[-1])
    control = _model(mode="fixed_one")
    assert list(control.parameters()) == []
    torch.testing.assert_close(_run(control, inputs), inputs[-1])


@pytest.mark.parametrize("channels", [256, 384])
def test_synthetic_initialization_cost_contrast_is_not_lost_at_real_profile_width(channels):
    # Explicit initialization-scale regression, not a trained accuracy assertion.
    torch.manual_seed(0)
    count = 256
    state = torch.randn(count, channels)
    nodes = torch.arange(count)
    incidence = torch.stack(
        (nodes.repeat(4), torch.cat([(nodes + s) % count for s in (1, 3, 9, 27)]))
    )
    context = torch.randn(1, 2 * channels + 8)
    inputs = (
        state,
        incidence,
        torch.zeros(count, dtype=torch.long),
        torch.full((count,), 8.0),
        torch.full((count,), 8.0),
        context,
        torch.ones(incidence.shape[1]),
    )
    legacy = GraphOptimizedConductance(channels)
    scaled = GraphOptimizedConductance(channels, solver_cost_scaling="width_scaled")
    scaled.load_state_dict(legacy.state_dict())
    with torch.no_grad():
        old_c, new_c = _run(legacy, inputs), _run(scaled, inputs)
    assert scaled.last_scores.std(correction=0) > 3 * legacy.last_scores.std(correction=0)
    assert new_c.std(correction=0) > 3 * old_c.std(correction=0)
    assert torch.isfinite(new_c).all() and (new_c > 0).all()
    assert scaled.solver_steps == legacy.solver_steps == 8
    assert scaled.solver_entropy == legacy.solver_entropy == 1.0
    assert scaled.cost_bound == legacy.cost_bound == 2.0
