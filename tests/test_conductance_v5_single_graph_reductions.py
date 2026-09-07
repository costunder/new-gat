"""CPU debug equivalence/call-count checks, not GPU performance measurements."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.nn import functional as F

from research.conductance_gat.v5 import optimization
from research.conductance_gat.v5.operator import graph_broadcast, graph_sum, graph_weighted_mean
from research.conductance_gat.v5.optimization import (
    GraphOptimizedConductance,
    _degree,
    _graph_max,
    _normalize_log_c,
    conductance_energy,
)


def _scatter_mean(values, index, count, weights):
    numerator = values.new_zeros(count).index_add(0, index, values * weights)
    denominator = values.new_zeros(count).index_add(0, index, weights)
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny)


def _debug_inputs(*, channels=5, dtype=torch.float64, graphs=1):
    generator = torch.Generator().manual_seed(917)
    incidence = torch.tensor([[0, 0, 0, 1, 2, 4, 5, 5], [1, 2, 3, 2, 3, 5, 6, 7]])
    node_graph = torch.zeros(9, dtype=torch.long)
    if graphs == 3:
        node_graph = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2])
    degree = torch.bincount(incidence.flatten(), minlength=9).to(dtype)
    return (
        torch.randn(9, channels, generator=generator, dtype=dtype),
        incidence,
        node_graph,
        degree,
        degree + torch.arange(9, dtype=dtype) % 3,
        torch.randn(graphs, 2 * channels + 8, generator=generator, dtype=dtype),
        torch.tensor([1.0, 2.5, 0.7, 1.3, 3.0, 1.2, 0.8, 2.1], dtype=dtype),
    )


def _run(model, inputs):
    state, incidence, node_graph, degree, full_degree, context, omega = inputs
    return model(
        state,
        incidence,
        node_graph,
        context.shape[0],
        graph_context=context,
        sample_degree=degree,
        full_degree=full_degree,
        edge_normalization_weight=omega,
    )


def _legacy_scatter_solver(model, inputs):
    """Independent old scatter/unroll equations, with no static-term reuse."""
    state, incidence, node_graph, degree, full_degree, context, omega = inputs
    count = context.shape[0]
    tail, head = incidence
    index = node_graph[tail]
    projected = F.normalize(model.node_projection(state), dim=1, eps=1e-6)
    metric = model.context_metric(F.layer_norm(context, (context.shape[1],))).tanh()
    # Compatibility math is unchanged by this optimization; test the old
    # graph normalization, gradient update, and diagnostic equation below.
    raw = model._raw_compatibility_chunk(projected, metric, tail, head, degree, full_degree, index)
    if model.solver_cost_scaling == "width_scaled":
        raw = raw - _scatter_mean(raw, index, count, omega)[index]
    delta = model.cost_bound * torch.tanh(raw / model.cost_bound)
    mass = delta.new_zeros(count).index_add(0, index, omega)
    reference = _degree(omega, incidence, state.shape[0])
    active_counts = delta.new_zeros(count).index_add(0, node_graph, (reference > 0).to(delta.dtype))
    log_c = torch.zeros_like(delta)
    for _ in range(model.solver_steps):
        weighted_degree = _degree(omega * log_c.exp(), incidence, state.shape[0])
        inverse_sum = weighted_degree[tail].reciprocal() + weighted_degree[head].reciprocal()
        barrier = (
            model.solver_degree_barrier * (mass / active_counts.clamp_min(1))[index] * inverse_sum
        )
        gradient = delta + model.solver_entropy * log_c - barrier
        centered = gradient - _scatter_mean(gradient, index, count, omega)[index]
        curvature = delta.new_zeros(count).scatter_reduce(
            0, index, barrier, "amax", include_self=True
        )
        magnitude = delta.new_zeros(count).scatter_reduce(
            0, index, centered.abs(), "amax", include_self=True
        )
        curvature_step = math.exp(-1) / curvature.clamp_min(math.exp(-1) / model.solver_step_size)
        displacement_step = 0.5 / magnitude.clamp_min(0.5 / model.solver_step_size)
        step = torch.minimum(curvature_step, displacement_step)
        proposal = log_c - step[index] * centered / (1 + step[index] * model.solver_entropy)
        maximum = proposal.new_full((count,), -torch.inf).scatter_reduce(
            0, index, proposal, "amax", include_self=True
        )
        shifted = proposal - maximum[index]
        log_c = shifted - _scatter_mean(shifted.exp(), index, count, omega)[index].log()
    return log_c.exp(), delta


@pytest.mark.parametrize(("graphs", "rows"), [(0, 0), (1, 0), (3, 0), (1, 7), (3, 7)])
def test_graph_sum_and_broadcast_match_scatter_values_and_gradients(graphs, rows):
    index = torch.arange(rows, dtype=torch.long) % max(graphs, 1)
    values = torch.randn(rows, 3, dtype=torch.float64, requires_grad=True)
    expected = values.new_zeros(graphs, 3).index_add(0, index, values)
    actual = graph_sum(values, index, graphs)
    torch.testing.assert_close(actual, expected)
    assert actual.requires_grad
    upstream = torch.randn_like(actual)
    expected_grad = torch.autograd.grad((expected * upstream).sum(), values)[0]
    actual_grad = torch.autograd.grad((actual * upstream).sum(), values)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    graph_values = torch.randn(graphs, 3, dtype=torch.float64, requires_grad=True)
    expanded = graph_broadcast(graph_values, index, graphs)
    gathered = graph_values[index]
    upstream = torch.randn_like(expanded)
    torch.testing.assert_close(expanded, gathered)
    torch.testing.assert_close(
        torch.autograd.grad((expanded * upstream).sum(), graph_values)[0],
        torch.autograd.grad((gathered * upstream).sum(), graph_values)[0],
    )


@pytest.mark.parametrize("graphs", [1, 3])
def test_weighted_mean_preserves_live_weight_gradient(graphs):
    values = torch.tensor([0.3, -0.8, 2.0, 0.1], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([1.0, 0.3, 4.0, 1.2], dtype=torch.float64, requires_grad=True)
    index = torch.zeros(4, dtype=torch.long) if graphs == 1 else torch.tensor([0, 0, 1, 1])
    actual = graph_weighted_mean(values, index, graphs, weights)
    expected = _scatter_mean(values, index, graphs, weights)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (values, weights))
    expected_grads = torch.autograd.grad(expected.square().sum(), (values, weights))
    for first, second in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(first, second)
    assert actual_grads[1].abs().sum() > 0


@pytest.mark.parametrize("data", [[], [-2.0, -1.0], [0.0, 0.0, -1.0], [2.0, 2.0, 0.0]])
@pytest.mark.parametrize("initial", [0.0, -torch.inf])
def test_max_preserves_empty_and_including_self_tie_gradients(data, initial):
    values = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    index = torch.zeros(values.numel(), dtype=torch.long)
    expected = values.new_full((1,), initial).scatter_reduce(
        0, index, values, "amax", include_self=True
    )
    actual = _graph_max(values, index, 1, initial=initial)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), values)[0],
        torch.autograd.grad(expected.sum(), values)[0],
    )


def test_empty_normalization_is_differentiable_and_returns_no_edges():
    log_c = torch.empty(0, dtype=torch.float64, requires_grad=True)
    actual = _normalize_log_c(log_c, torch.empty(0, dtype=torch.long), 1, torch.empty_like(log_c))
    assert actual.shape == (0,)
    assert torch.autograd.grad(actual.sum(), log_c)[0].shape == (0,)


@pytest.mark.parametrize("index", [torch.tensor([-1]), torch.tensor([1])])
def test_single_graph_helpers_reject_invalid_ids_instead_of_ignoring_them(index):
    with pytest.raises(RuntimeError, match="graph_index"):
        graph_weighted_mean(torch.ones(1), index, 1)
    with pytest.raises(RuntimeError, match="graph_index"):
        graph_sum(torch.ones(1, 2), index, 1)
    with pytest.raises(RuntimeError, match="graph_index"):
        graph_broadcast(torch.ones(1, 2), index, 1)


@pytest.mark.parametrize("scaling", ["legacy_unit", "width_scaled"])
@pytest.mark.parametrize("graphs", [1, 3])
@pytest.mark.parametrize("rho", [0.0, 0.1, 20.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_solver_matches_old_unroll_for_c_cost_all_parameter_and_input_gradients(
    scaling, graphs, rho, dtype
):
    torch.manual_seed(134)
    inputs = _debug_inputs(graphs=graphs, dtype=dtype)
    tolerance = (
        {"rtol": 2e-9, "atol": 2e-11} if dtype == torch.float64 else {"rtol": 2e-4, "atol": 2e-5}
    )
    actual_inputs = tuple(
        value.clone().requires_grad_(True) if value.is_floating_point() else value
        for value in inputs
    )
    expected_inputs = tuple(
        value.clone().requires_grad_(True) if value.is_floating_point() else value
        for value in inputs
    )
    actual_model = GraphOptimizedConductance(
        5, solver_cost_scaling=scaling, solver_degree_barrier=rho, edge_chunk_size=3
    ).to(dtype)
    expected_model = copy.deepcopy(actual_model)
    c = _run(actual_model, actual_inputs)
    expected, expected_cost = _legacy_scatter_solver(expected_model, expected_inputs)
    torch.testing.assert_close(c, expected, **tolerance)
    torch.testing.assert_close(actual_model.last_scores, expected_cost, **tolerance)
    target = torch.arange(c.numel(), dtype=c.dtype).sin()
    ((c - target).square().sum()).backward()
    ((expected - target).square().sum()).backward()
    for (name, first), (_, second) in zip(
        actual_model.named_parameters(), expected_model.named_parameters(), strict=True
    ):
        assert first.grad is not None and torch.isfinite(first.grad).all(), name
        torch.testing.assert_close(first.grad, second.grad, msg=name, **tolerance)
    for first, second in zip(actual_inputs, expected_inputs, strict=True):
        if first.requires_grad:
            assert first.grad is not None and torch.isfinite(first.grad).all()
            torch.testing.assert_close(first.grad, second.grad, **tolerance)
    for values in (torch.ones_like(c), c.detach()):
        independent = conductance_energy(
            values,
            actual_model.last_scores,
            inputs[1],
            inputs[2],
            graphs,
            inputs[-1],
            entropy=actual_model.solver_entropy,
            degree_barrier=rho,
        )
        key = "objective_initial" if torch.equal(values, torch.ones_like(c)) else "objective_final"
        torch.testing.assert_close(actual_model.last_solver_diagnostics[key], independent)
    assert actual_model.last_solver_diagnostics["executed_steps"] == 8
    assert actual_model.state_dict().keys() == expected_model.state_dict().keys()


@pytest.mark.parametrize("channels", [256, 384])
def test_actual_width_solver_has_no_graph_destination_scatter_in_forward_or_backward(channels):
    # Same widths and K as final profiles, with an explicitly synthetic CPU
    # graph. The assertions are call counts, never a GPU throughput claim.
    torch.manual_seed(881)
    inputs = _debug_inputs(channels=channels, dtype=torch.float32)
    state = inputs[0].clone().requires_grad_(True)
    inputs = (state, *inputs[1:])
    model = GraphOptimizedConductance(channels, solver_cost_scaling="width_scaled")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as profiler:
        c = _run(model, inputs)
        (c * torch.arange(c.numel(), dtype=c.dtype)).sum().backward()
    assert (
        state.grad is not None and torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    )
    for event in profiler.events():
        if event.name in {
            "aten::index_add",
            "aten::index_add_",
            "aten::scatter_reduce",
            "aten::scatter_reduce_",
            "aten::scatter_add",
            "aten::scatter_add_",
            "aten::_index_put_impl_",
            "aten::index_put_",
        }:
            assert event.input_shapes[0][0] != 1, (event.name, event.input_shapes)
    assert model.last_solver_diagnostics["executed_steps"] == 8


def test_solver_reuses_static_and_final_degree_without_skipping_updates_or_checks(monkeypatch):
    inputs = _debug_inputs()
    model = GraphOptimizedConductance(5, solver_cost_scaling="width_scaled").double()
    original_degree = optimization._degree
    calls = []

    def record_degree(*arguments):
        calls.append(arguments[0].shape)
        return original_degree(*arguments)

    monkeypatch.setattr(optimization, "_degree", record_degree)
    with torch.no_grad():
        _run(model, inputs)
    # Reference/first iterate once, K-1 remaining iterates, final diagnostic
    # degree shared by energy and residual: 1 + (K-1) + 1, previously K+6.
    assert len(calls) == model.solver_steps + 1 == 9
    assert model.last_solver_diagnostics["executed_steps"] == 8
    invalid = list(inputs)
    invalid[0] = torch.full_like(inputs[0], torch.nan)
    with pytest.raises(RuntimeError, match="nonfinite"):
        _run(model, invalid)
