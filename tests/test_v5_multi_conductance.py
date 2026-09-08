"""CPU synthetic/debug-only mathematical and gradient tests, not training results."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.operator import (
    conductance_propagation_coefficients,
    graph_sum,
    shared_head_diffusion,
)
from research.conductance_gat.v5.optimization import GraphOptimizedConductance, conductance_energy


def _inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(741)
    x = torch.randn(9, 8, generator=generator, dtype=dtype)
    incidence = torch.tensor([[0, 0, 0, 1, 2, 4, 5, 5], [1, 2, 3, 2, 3, 5, 6, 7]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2])
    degree = torch.bincount(incidence.flatten(), minlength=9).to(dtype)
    omega = torch.tensor([1.0, 2.5, 0.7, 1.3, 3.0, 1.2, 0.8, 2.1], dtype=dtype)
    context = torch.randn(3, 24, generator=generator, dtype=dtype)
    return x, incidence, batch, degree, omega, context


def _solve(estimator, inputs, relations=None):
    x, incidence, batch, degree, omega, context = inputs
    return estimator(
        x,
        incidence,
        batch,
        3,
        graph_context=context,
        sample_degree=degree,
        full_degree=degree + 2,
        edge_normalization_weight=omega,
        edge_relation_id=relations,
    )


def _dense(message, c, incidence, batch, beta, omega, normalization, polynomial):
    # Dense adjacency exists ONLY in this explicitly tiny mathematical test.
    heads, nodes = message.shape[1], message.shape[0]
    effective = (c[:, None] if c.ndim == 1 else c) * omega[:, None]
    adjacency = message.new_zeros((nodes, nodes, heads))
    tail, head = incidence
    adjacency = adjacency.index_put((tail, head), effective.expand(-1, heads), accumulate=True)
    adjacency = adjacency.index_put((head, tail), effective.expand(-1, heads), accumulate=True)
    degree = adjacency.sum(dim=1)
    active = degree > 0
    safe = torch.where(active, degree, torch.ones_like(degree))
    if normalization == "row":
        p = adjacency / safe[:, None, :]
    else:
        p = adjacency * safe.rsqrt()[:, None, :] * safe.rsqrt()[None, :, :]
    p = p + torch.eye(nodes, dtype=message.dtype)[:, :, None] * (~active)[:, None, :]
    p1 = torch.einsum("ijh,jhd->ihd", p, message)
    result = message + beta[batch, :, None] * (p1 - message)
    if polynomial is not None:
        p2 = torch.einsum("ijh,jhd->ihd", p, p1)
        p3 = torch.einsum("ijh,jhd->ihd", p, p2)
        result = result + polynomial[None, :, 0, None] * (p2 - message)
        result = result + polynomial[None, :, 1, None] * (p3 - message)
    return result


@pytest.mark.parametrize("normalization", ["symmetric", "row"])
@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("polynomial", [False, True])
def test_dense_forward_and_gradient_equivalence(normalization, per_head, polynomial):
    torch.manual_seed(63)
    _, incidence, batch, _, omega, _ = _inputs()
    message = torch.randn(9, 2, 3, dtype=torch.float64, requires_grad=True)
    c = torch.rand((8, 2) if per_head else (8,), dtype=torch.float64).add(0.2).requires_grad_()
    beta = torch.rand(3, 2, dtype=torch.float64, requires_grad=True)
    omega = omega.requires_grad_()
    coefficients = (
        torch.randn(2, 2, dtype=torch.float64, requires_grad=True) if polynomial else None
    )
    actual = shared_head_diffusion(
        message,
        c,
        incidence,
        batch,
        beta,
        sampling_correction=omega,
        propagation_normalization=normalization,
        polynomial_coefficients=coefficients,
        edge_chunk_size=3,
    )
    expected = _dense(message, c, incidence, batch, beta, omega, normalization, coefficients)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual[8], message[8], rtol=0, atol=0)
    probe = torch.randn_like(actual)
    parameters = [message, c, beta, omega] + ([coefficients] if polynomial else [])
    actual_grad = torch.autograd.grad((actual * probe).sum(), parameters, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * probe).sum(), parameters)
    for first, second in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("normalization", ["symmetric", "row"])
def test_per_head_custom_autograd_gradgradcheck(normalization):
    incidence = torch.tensor([[0, 0, 1], [1, 2, 2]])
    batch = torch.zeros(3, dtype=torch.long)
    message = torch.randn(3, 2, 2, dtype=torch.float64, requires_grad=True)
    c = torch.rand(3, 2, dtype=torch.float64).add(0.5).requires_grad_()
    beta = torch.rand(1, 2, dtype=torch.float64, requires_grad=True)

    def fn(m, w, b):
        return shared_head_diffusion(
            m, w, incidence, batch, b, propagation_normalization=normalization, edge_chunk_size=2
        )

    assert torch.autograd.gradcheck(fn, (message, c, beta), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (message, c, beta), fast_mode=True)


@pytest.mark.parametrize("generator", ["optimized", "degree_only", "entropy_exact"])
def test_independent_head_solver_gauge_energy_and_permutations(generator):
    torch.manual_seed(85)
    inputs = _inputs()
    estimator = GraphOptimizedConductance(
        8,
        conductance_heads=3,
        generator=generator,
        solver_degree_barrier=0.0 if generator == "entropy_exact" else 0.1,
        solver_cost_scaling="width_scaled",
        edge_chunk_size=3,
    ).double()
    c = _solve(estimator, inputs)
    assert c.shape == (8, 3)
    assert torch.isfinite(c).all() and (c > 0).all()
    edge_graph = inputs[2][inputs[1][0]]
    mass = graph_sum(inputs[4], edge_graph, 3)
    means = graph_sum(c * inputs[4][:, None], edge_graph, 3) / mass.clamp_min(1e-30)[:, None]
    torch.testing.assert_close(means[:2], torch.ones_like(means[:2]))
    if generator != "degree_only":
        assert (c[:, 0] - c[:, 1]).abs().max() > 1e-5
    else:
        assert list(estimator.parameters()) == []
        assert (c - 1).abs().max() > 1e-5
        torch.testing.assert_close(c[:, 0], c[:, 1], rtol=0, atol=0)
    diagnostics = estimator.last_solver_diagnostics
    assert diagnostics["executed_steps"] == (0 if generator == "entropy_exact" else 8)
    assert (diagnostics["objective_final"] <= diagnostics["objective_initial"] + 1e-12).all()
    energy = conductance_energy(
        c,
        estimator.last_scores,
        inputs[1],
        inputs[2],
        3,
        inputs[4],
        entropy=1.0,
        degree_barrier=estimator.solver_degree_barrier,
    )
    torch.testing.assert_close(energy, diagnostics["objective_final"])
    order = torch.tensor([7, 3, 0, 6, 2, 1, 5, 4])
    permuted = list(inputs)
    permuted[1] = inputs[1].flip(0)[:, order]
    permuted[4] = inputs[4][order]
    torch.testing.assert_close(_solve(estimator, permuted), c[order], rtol=1e-10, atol=1e-11)
    if generator == "entropy_exact":
        raw = (-estimator.last_scores).exp()
        denominator = (
            graph_sum(raw * permuted[4][:, None], edge_graph[order], 3)
            / mass.clamp_min(1e-30)[:, None]
        )
        torch.testing.assert_close(
            _solve(estimator, permuted), raw / denominator[edge_graph[order]]
        )
        assert diagnostics["projected_gradient_rms_final"].max() < 1e-12


def _model_graph():
    x, incidence, batch, degree, omega, _ = _inputs(torch.float32)
    return SimpleNamespace(
        x=x,
        incidence_edge_index=incidence,
        batch=batch,
        _v5_num_graphs=3,
        full_degree=degree + 2,
        edge_normalization_weight=omega,
        sampling_correction=omega,
    )


@pytest.mark.parametrize("generator", ["optimized", "degree_only", "entropy_exact"])
@pytest.mark.parametrize("normalization", ["symmetric", "row"])
def test_forward_loss_backward_optimizer_updates_real_components(generator, normalization):
    torch.manual_seed(76)
    model = GraphConditionedConductanceNodeClassifier(
        8,
        3,
        hidden_channels=16,
        layers=2,
        heads=2,
        dropout=0.0,
        activation_checkpoint=True,
        conductance_heads="per_head",
        conductance_generator=generator,
        propagation_normalization=normalization,
        propagation_filter="polynomial3",
        solver_degree_barrier=0.0 if generator == "entropy_exact" else 0.1,
        solver_cost_scaling="width_scaled",
        edge_chunk_size=3,
    )
    graph = _model_graph()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss = F.cross_entropy(model(graph), torch.arange(9) % 3)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    changed = {
        name for name, value in model.named_parameters() if not torch.equal(before[name], value)
    }
    assert any("value_weight" in name for name in changed)
    assert any("beta_estimator" in name for name in changed)
    assert any("polynomial_delta" in name for name in changed)
    if generator == "degree_only":
        assert not any("estimator." in name and "beta_estimator" not in name for name in before)
    else:
        assert any("context_metric" in name for name in changed)
        assert any("node_projection" in name for name in changed)
        assert any("structure_metric" in name for name in changed)


def test_raw_c_is_not_attention_and_heads_are_not_averaged():
    message = torch.arange(12, dtype=torch.float64).reshape(3, 2, 2)
    c = torch.tensor([[0.2, 1.8], [1.8, 0.2]], dtype=torch.float64)
    incidence = torch.tensor([[0, 0], [1, 2]])
    batch = torch.zeros(3, dtype=torch.long)
    beta = torch.ones(1, 2, dtype=torch.float64)
    tail, head, degree = conductance_propagation_coefficients(c, incidence, 3, normalization="row")
    rows = (
        torch.zeros(3, 2, dtype=torch.float64)
        .index_add(0, incidence[0], tail)
        .index_add(0, incidence[1], head)
    )
    torch.testing.assert_close(rows[degree > 0], torch.ones_like(rows[degree > 0]))
    assert c.max() > 1 and tail.min() >= 0 and tail.max() <= 1
    result = shared_head_diffusion(
        message, c, incidence, batch, beta, propagation_normalization="row"
    )
    averaged = shared_head_diffusion(
        message, c.mean(1), incidence, batch, beta, propagation_normalization="row"
    )
    assert not torch.allclose(result, averaged)


@pytest.mark.parametrize("heads", [1, 3])
def test_relation_metric_uses_explicit_types_and_receives_task_gradient(heads):
    torch.manual_seed(51)
    inputs = _inputs()
    estimator = GraphOptimizedConductance(
        8, conductance_heads=heads, num_relations=2, solver_cost_scaling="width_scaled"
    ).double()
    ids = torch.arange(8) % 2
    with pytest.raises(ValueError, match="explicit edge_relation_id"):
        _solve(estimator, inputs)
    c = _solve(estimator, inputs, ids)
    coefficient = torch.arange(c.numel(), dtype=c.dtype).reshape_as(c)
    (c * coefficient).sum().backward()
    assert estimator.relation_metric.grad is not None
    assert estimator.relation_metric.grad.abs().sum() > 0
    flipped = _solve(estimator, inputs, 1 - ids)
    assert not torch.allclose(c, flipped)
    permuted = list(inputs)
    permuted[1] = inputs[1].flip(0)
    torch.testing.assert_close(_solve(estimator, permuted, ids), c)
    with pytest.raises(RuntimeError, match="outside"):
        _solve(estimator, inputs, ids + 2)


@pytest.mark.parametrize("normalization", ["symmetric", "row"])
def test_all_ones_control_perhead_equals_shared(normalization):
    inputs = _inputs()
    estimator = GraphOptimizedConductance(8, mode="fixed_one", conductance_heads=2).double()
    c = _solve(estimator, inputs)
    assert list(estimator.parameters()) == []
    torch.testing.assert_close(c, torch.ones_like(c), rtol=0, atol=0)
    message = torch.randn(9, 2, 4, dtype=torch.float64)
    beta = torch.rand(3, 2, dtype=torch.float64)
    common = (inputs[1], inputs[2], beta)
    actual = shared_head_diffusion(message, c, *common, propagation_normalization=normalization)
    expected = shared_head_diffusion(
        message, c[:, 0], *common, propagation_normalization=normalization
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


def test_legacy_defaults_state_dict_configuration_and_seed_pairing_unchanged():
    args = dict(hidden_channels=16, layers=2, heads=2, dropout=0.0)
    torch.manual_seed(58)
    legacy = GraphConditionedConductanceNodeClassifier(8, 3, **args).eval()
    torch.manual_seed(58)
    explicit = GraphConditionedConductanceNodeClassifier(
        8,
        3,
        conductance_heads="shared",
        propagation_normalization="symmetric",
        conductance_generator="optimized",
        propagation_filter="linear",
        num_relations=0,
        edge_direction="undirected",
        **args,
    ).eval()
    assert legacy.conductance_configuration == explicit.conductance_configuration
    assert "conductance_heads" not in legacy.conductance_configuration
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(legacy(_model_graph()), explicit(_model_graph()), rtol=0, atol=0)
    torch.manual_seed(58)
    larger_c = GraphConditionedConductanceNodeClassifier(8, 3, conductance_heads="per_head", **args)
    for name, value in legacy.state_dict().items():
        if ".estimator." not in name:
            torch.testing.assert_close(value, larger_c.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edge_direction": "directed"},
        {"conductance_backend": "mlp", "conductance_heads": "per_head"},
        {"conductance_generator": "entropy_exact", "solver_degree_barrier": 0.1},
        {"conductance_generator": "degree_only", "num_relations": 2},
    ],
)
def test_unsupported_semantics_are_rejected_not_silently_faked(kwargs):
    with pytest.raises(ValueError):
        GraphConditionedConductanceNodeClassifier(
            8, 3, hidden_channels=16, layers=2, heads=2, **kwargs
        )


def test_head_solver_matches_independent_scalar_objectives_and_gradients():
    torch.manual_seed(92)
    inputs = _inputs()
    multi = GraphOptimizedConductance(
        8, conductance_heads=3, solver_cost_scaling="width_scaled"
    ).double()
    actual = _solve(multi, inputs)
    # Independent scalar runs are a debug oracle only, never a production head loop.
    scalar_results = []
    for head in range(3):
        scalar = GraphOptimizedConductance(8, solver_cost_scaling="width_scaled").double()
        with torch.no_grad():
            scalar.node_projection.weight.copy_(multi.node_projection.weight)
            scalar.context_metric.weight.copy_(
                multi.context_metric.weight[8 * head : 8 * (head + 1)]
            )
            scalar.context_metric.bias.copy_(multi.context_metric.bias[8 * head : 8 * (head + 1)])
            scalar.structure_metric.copy_(multi.structure_metric[head])
        scalar_results.append(_solve(scalar, inputs))
    torch.testing.assert_close(actual, torch.stack(scalar_results, dim=1), rtol=1e-11, atol=1e-12)
    x, incidence, batch, degree, omega, context = inputs
    x = x.clone().requires_grad_()
    omega = omega.clone().requires_grad_()

    def fn(state, weight):
        return _solve(multi, (state, incidence, batch, degree, weight, context))

    assert torch.autograd.gradcheck(fn, (x, omega), fast_mode=True, atol=1e-6, rtol=1e-4)


def test_per_head_disjoint_batch_independence_node_permutation_and_weight_scale():
    torch.manual_seed(34)
    inputs = _inputs()
    estimator = GraphOptimizedConductance(
        8, conductance_heads=3, solver_cost_scaling="width_scaled"
    ).double()
    expected = _solve(estimator, inputs)
    x, incidence, batch, degree, omega, context = inputs
    single = estimator(
        x[:4],
        incidence[:, :5],
        torch.zeros(4, dtype=torch.long),
        1,
        graph_context=context[:1],
        sample_degree=degree[:4],
        full_degree=degree[:4] + 2,
        edge_normalization_weight=omega[:5],
    )
    torch.testing.assert_close(single, expected[:5], rtol=1e-10, atol=1e-11)
    scale = torch.tensor([2.5, 0.3, 10.0], dtype=x.dtype)
    scaled = (x, incidence, batch, degree, omega * scale[batch[incidence[0]]], context)
    torch.testing.assert_close(_solve(estimator, scaled), expected, rtol=1e-10, atol=1e-11)
    order = torch.tensor([8, 4, 1, 6, 0, 3, 7, 2, 5])
    inverse = order.argsort()
    permuted = (x[order], inverse[incidence], batch[order], degree[order], omega, context)
    torch.testing.assert_close(_solve(estimator, permuted), expected, rtol=1e-10, atol=1e-11)
    estimator.override = "shuffle"
    shuffled = _solve(estimator, inputs)
    torch.testing.assert_close(shuffled[:5], expected[:5].flip(0))
    torch.testing.assert_close(shuffled[5:], expected[5:].flip(0))
    estimator.override = "mean"
    torch.testing.assert_close(_solve(estimator, inputs), torch.ones_like(expected))


def test_relation_metric_is_used_by_classifier_task_loss_and_optimizer():
    torch.manual_seed(89)
    graph = _model_graph()
    graph.edge_relation_id = torch.arange(8) % 2
    model = GraphConditionedConductanceNodeClassifier(
        8,
        3,
        hidden_channels=16,
        layers=2,
        heads=2,
        dropout=0.0,
        conductance_heads="per_head",
        num_relations=2,
        propagation_normalization="row",
        solver_cost_scaling="width_scaled",
    )
    parameters = [operator.estimator.relation_metric for operator in model.operators]
    before = [value.detach().clone() for value in parameters]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    F.cross_entropy(model(graph), torch.arange(9) % 3).backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in parameters)
    optimizer.step()
    assert all(
        not torch.equal(previous, value) for previous, value in zip(before, parameters, strict=True)
    )


@pytest.mark.parametrize("normalization", ["row", "symmetric"])
def test_bf16_keeps_geometry_fp32_and_has_finite_backward(normalization):
    torch.manual_seed(28)
    model = GraphConditionedConductanceNodeClassifier(
        8,
        3,
        hidden_channels=16,
        layers=2,
        heads=2,
        conductance_heads="per_head",
        propagation_normalization=normalization,
        propagation_filter="polynomial3",
        dropout=0.0,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(_model_graph())
        loss = F.cross_entropy(output.float(), torch.arange(9) % 3)
    loss.backward()
    for operator in model.operators:
        assert operator.estimator.last_c.dtype == torch.float32
        assert operator.last_beta.dtype == torch.float32
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters()
    )


def test_polynomial_zero_delta_starts_as_identical_linear_filter():
    torch.manual_seed(94)
    graph = _model_graph()
    kwargs = dict(hidden_channels=16, layers=2, heads=2, dropout=0.0, conductance_heads="per_head")
    linear = GraphConditionedConductanceNodeClassifier(8, 3, **kwargs).eval()
    torch.manual_seed(94)
    polynomial = GraphConditionedConductanceNodeClassifier(
        8, 3, propagation_filter="polynomial3", **kwargs
    ).eval()
    for name, value in linear.state_dict().items():
        torch.testing.assert_close(value, polynomial.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(linear(graph), polynomial(graph), rtol=0, atol=0)
