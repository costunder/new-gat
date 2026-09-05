"""Explicit CPU debug graphs for the finite-step C optimization contract."""

from __future__ import annotations

import copy

import pytest
import torch

from research.conductance_gat.v5.operator import graph_weighted_mean, shared_head_diffusion
from research.conductance_gat.v5.optimization import (
    GraphOptimizedConductance,
    _degree,
    conductance_energy,
)


def _debug_inputs(*, dtype=torch.float64, channels=5):
    generator = torch.Generator().manual_seed(731)
    state = torch.randn(9, channels, generator=generator, dtype=dtype)
    # Graph 0 has heterogeneous degree, graph 1 a path, graph 2 an isolate.
    incidence = torch.tensor([[0, 0, 0, 1, 2, 4, 5, 5], [1, 2, 3, 2, 3, 5, 6, 7]], dtype=torch.long)
    node_graph = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=torch.long)
    degree = torch.bincount(incidence.flatten(), minlength=9).to(dtype)
    full_degree = degree + torch.arange(9, dtype=dtype) % 3
    context = torch.randn(3, 2 * channels + 8, generator=generator, dtype=dtype)
    omega = torch.tensor([1.0, 2.5, 0.7, 1.3, 3.0, 1.2, 0.8, 2.1], dtype=dtype)
    return state, incidence, node_graph, degree, full_degree, context, omega


def _run(model, inputs, *, state=None, context=None):
    x, incidence, node_graph, degree, full_degree, z, omega = inputs
    return model(
        x if state is None else state,
        incidence,
        node_graph,
        z.shape[0],
        graph_context=z if context is None else context,
        sample_degree=degree,
        full_degree=full_degree,
        edge_normalization_weight=omega,
    )


def test_positive_weighted_gauge_and_detached_finite_step_diagnostics():
    torch.manual_seed(51)
    inputs = _debug_inputs()
    model = GraphOptimizedConductance(5).double()
    c = _run(model, inputs)
    assert torch.isfinite(c).all() and (c > 0).all()
    assert not torch.allclose(c, torch.ones_like(c))
    mean = graph_weighted_mean(c, inputs[2][inputs[1][0]], 3, inputs[-1])
    torch.testing.assert_close(mean, torch.tensor([1.0, 1.0, 0.0], dtype=c.dtype))
    diagnostics = model.last_solver_diagnostics
    assert diagnostics["executed_steps"] == 8
    assert diagnostics["finite_step_approximation"] is True
    assert diagnostics["objective_final"].le(diagnostics["objective_initial"] + 1e-12).all()
    assert diagnostics["step_size_min"].gt(0).all()
    assert diagnostics["step_size_max"].le(0.25).all()
    for value in diagnostics.values():
        if isinstance(value, torch.Tensor):
            assert value.grad_fn is None and not value.requires_grad
            assert torch.isfinite(value).all()
    assert c.grad_fn is not None
    for cached in (model.last_scores, model.last_log_c, model.last_c):
        assert cached.grad_fn is None and not cached.requires_grad


def test_orientation_and_node_edge_permutation_equivariance():
    torch.manual_seed(41)
    inputs = _debug_inputs()
    model = GraphOptimizedConductance(5).double()
    expected = _run(model, inputs)
    reversed_inputs = list(inputs)
    reversed_inputs[1] = inputs[1].flip(0)
    torch.testing.assert_close(_run(model, reversed_inputs), expected, rtol=1e-11, atol=1e-12)
    permutation = torch.tensor([8, 4, 1, 6, 0, 3, 7, 2, 5])
    inverse = permutation.argsort()
    edge_permutation = torch.tensor([7, 3, 0, 5, 1, 6, 4, 2])
    changed = list(inputs)
    changed[0] = inputs[0][permutation]
    changed[1] = inverse[inputs[1][:, edge_permutation]]
    changed[2] = inputs[2][permutation]
    changed[3] = inputs[3][permutation]
    changed[4] = inputs[4][permutation]
    changed[6] = inputs[6][edge_permutation]
    actual = _run(model, changed)
    torch.testing.assert_close(actual, expected[edge_permutation], rtol=1e-10, atol=1e-11)


def test_disjoint_batch_independence_and_importance_scale_invariance():
    torch.manual_seed(73)
    inputs = _debug_inputs()
    model = GraphOptimizedConductance(5).double()
    expected = _run(model, inputs)
    single = [
        inputs[0][:4],
        inputs[1][:, :5],
        torch.zeros(4, dtype=torch.long),
        inputs[3][:4],
        inputs[4][:4],
        inputs[5][:1],
        inputs[6][:5],
    ]
    torch.testing.assert_close(_run(model, single), expected[:5], rtol=1e-10, atol=1e-12)
    changed = list(inputs)
    changed[-1] = inputs[-1] * torch.tensor(
        [1e3, 1e3, 1e3, 1e3, 1e3, 0.03, 0.03, 0.03], dtype=inputs[0].dtype
    )
    torch.testing.assert_close(_run(model, changed), expected, rtol=1e-10, atol=1e-12)


def test_fixed_one_has_no_parameters_and_interventions_are_explicit():
    inputs = _debug_inputs()
    control = GraphOptimizedConductance(5, mode="fixed_one").double()
    assert list(control.parameters()) == []
    torch.testing.assert_close(_run(control, inputs), torch.ones_like(inputs[-1]))
    assert control.last_solver_diagnostics["executed_steps"] == 0
    dynamic = GraphOptimizedConductance(5).double()
    original = _run(dynamic, inputs)
    dynamic.override = "mean"
    torch.testing.assert_close(_run(dynamic, inputs), torch.ones_like(original))
    dynamic.override = "ones"
    torch.testing.assert_close(_run(dynamic, inputs), torch.ones_like(original))
    assert dynamic.last_solver_diagnostics["executed_steps"] == 0
    dynamic.override = "shuffle"
    shuffled = _run(dynamic, inputs)
    torch.testing.assert_close(shuffled[:5], original[:5].flip(0))
    torch.testing.assert_close(shuffled[5:], original[5:].flip(0))
    dynamic.override = "unknown"
    with pytest.raises(ValueError, match="intervention"):
        _run(dynamic, inputs)


def test_edgeless_and_single_edge_graphs_have_mathematically_defined_results():
    state = torch.randn(3, 5, dtype=torch.float64)
    model = GraphOptimizedConductance(5).double()
    empty = model(
        state,
        torch.empty(2, 0, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        1,
        graph_context=torch.randn(1, 18, dtype=torch.float64),
        sample_degree=torch.zeros(3, dtype=torch.float64),
        full_degree=torch.zeros(3, dtype=torch.float64),
    )
    assert empty.shape == (0,)
    assert model.last_solver_diagnostics["reason"] == "edgeless_graph"
    incidence = torch.tensor([[0], [1]])
    one = model(
        state,
        incidence,
        torch.zeros(3, dtype=torch.long),
        1,
        graph_context=torch.randn(1, 18, dtype=torch.float64),
        sample_degree=torch.tensor([1, 1, 0], dtype=torch.float64),
        full_degree=torch.tensor([1, 1, 0], dtype=torch.float64),
    )
    torch.testing.assert_close(one, torch.ones_like(one))
    assert model.last_solver_diagnostics["executed_steps"] == 8
    assert model.last_solver_diagnostics["active_nodes"].tolist() == [2.0]
    one.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("rho", [0.0, 0.1, 10.0])
def test_analytic_energy_gradient_matches_dense_autograd(rho):
    inputs = _debug_inputs()
    state, incidence, node_graph, _, _, _, omega = inputs
    generator = torch.Generator().manual_seed(54)
    c = (torch.rand(8, generator=generator, dtype=torch.float64) + 0.4).requires_grad_(True)
    delta = torch.randn(8, generator=generator, dtype=torch.float64)
    entropy = 0.7
    model = GraphOptimizedConductance(5, solver_entropy=entropy, solver_degree_barrier=rho).double()
    energy = conductance_energy(
        c, delta, incidence, node_graph, 3, omega, entropy=entropy, degree_barrier=rho
    )
    grad = torch.autograd.grad(energy.sum(), c)[0]
    edge_graph = node_graph[incidence[0]]
    mass = c.new_zeros(3).index_add(0, edge_graph, omega)
    reference = _degree(omega, incidence, state.shape[0])
    counts = c.new_zeros(3).index_add(0, node_graph, (reference > 0).to(c.dtype))
    analytic, _ = model._scaled_gradient(
        c.log(), delta, incidence, edge_graph, omega, mass, counts, state.shape[0]
    )
    torch.testing.assert_close(grad, omega / mass[edge_graph] * analytic, rtol=1e-11, atol=1e-12)
    # Independent dense unsigned incidence gives the same degree-barrier objective.
    unsigned = c.new_zeros((state.shape[0], c.numel()))
    edges = torch.arange(c.numel())
    unsigned[incidence[0], edges] = 1
    unsigned[incidence[1], edges] = 1
    dense_degree, dense_reference = unsigned @ (omega * c), unsigned @ omega
    active = dense_reference > 0
    dense_barrier = c.new_zeros(3).index_add(
        0, node_graph[active], (dense_degree[active] / dense_reference[active]).log()
    ) / counts.clamp_min(1)
    dense = (
        graph_weighted_mean(c * delta + entropy * (c * c.log() - c + 1), edge_graph, 3, omega)
        - rho * dense_barrier
    )
    torch.testing.assert_close(energy, dense)


def test_task_loss_reaches_every_dynamic_parameter_and_changes_them():
    torch.manual_seed(14)
    inputs = _debug_inputs()
    state = inputs[0].clone().requires_grad_(True)
    context = inputs[5].clone().requires_grad_(True)
    model = GraphOptimizedConductance(5, edge_chunk_size=2).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    c = _run(model, inputs, state=state, context=context)
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
    task_target = torch.randn_like(propagated)
    loss = (propagated - task_target).square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert message.grad is not None and torch.isfinite(message.grad).all()
    optimizer.step()
    assert all(not torch.equal(before[name], value) for name, value in model.named_parameters())


def test_no_label_inference_and_zero_barrier_are_autograd_grad_free(monkeypatch):
    inputs = _debug_inputs()
    model = GraphOptimizedConductance(5, solver_degree_barrier=0).double().eval()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("The analytic solver must not call autograd.grad")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    with torch.no_grad():
        c = _run(model, inputs)
    assert (c > 0).all() and torch.isfinite(c).all()
    c = _run(model, inputs)
    (c * torch.arange(8, dtype=c.dtype)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_float64_gradcheck_includes_the_unrolled_state_and_context_path():
    torch.manual_seed(13)
    inputs = _debug_inputs(channels=3)
    model = GraphOptimizedConductance(3, solver_steps=3, edge_chunk_size=64).double()
    assert torch.autograd.gradcheck(
        lambda state, context: _run(model, inputs, state=state, context=context),
        (inputs[0].requires_grad_(True), inputs[5].requires_grad_(True)),
        eps=1e-6,
        atol=1e-5,
        rtol=2e-4,
        fast_mode=True,
    )


@pytest.mark.parametrize("rho", [0.0, 0.1, 20.0])
def test_k_sweep_monotone_energy_and_finite_high_degree_weight_skew(rho):
    torch.manual_seed(3)
    count = 30
    # Clique connected to leaves: heterogeneous degrees and importance weights.
    clique = torch.combinations(torch.arange(12), r=2).T
    leaves = torch.stack((torch.zeros(count - 12, dtype=torch.long), torch.arange(12, count)))
    incidence = torch.cat((clique, leaves), dim=1)
    degree = torch.bincount(incidence.flatten(), minlength=count).double()
    inputs = (
        torch.randn(count, 5, dtype=torch.float64),
        incidence,
        torch.zeros(count, dtype=torch.long),
        degree,
        degree * 2,
        torch.randn(1, 18, dtype=torch.float64),
        torch.logspace(-2, 2, incidence.shape[1], dtype=torch.float64),
    )
    model = GraphOptimizedConductance(5, solver_degree_barrier=rho, edge_chunk_size=19).double()
    previous_energy = None
    initial_residual = None
    for steps in (1, 2, 4, 8, 16, 32):
        model.solver_steps = steps
        with torch.no_grad():
            c = _run(model, inputs)
        diagnostics = model.last_solver_diagnostics
        energy = diagnostics["objective_final"]
        if previous_energy is not None:
            assert energy.le(previous_energy + 1e-11).all()
        assert (c > 0).all() and torch.isfinite(c).all()
        assert diagnostics["executed_steps"] == steps
        if initial_residual is None:
            initial_residual = diagnostics["projected_gradient_rms_initial"]
        previous_energy = energy
    assert diagnostics["projected_gradient_rms_final"].lt(initial_residual).all()


def test_exact_chunking_preserves_values_and_gradients(monkeypatch):
    torch.manual_seed(17)
    inputs = _debug_inputs()
    checkpointed = GraphOptimizedConductance(5, edge_chunk_size=2).double()
    direct = copy.deepcopy(checkpointed)
    c1 = _run(checkpointed, inputs)
    (c1 * torch.arange(8, dtype=c1.dtype)).sum().backward()
    monkeypatch.setattr(
        torch.utils.checkpoint, "checkpoint", lambda function, *args, **_kwargs: function(*args)
    )
    direct.edge_chunk_size = 65536
    c2 = _run(direct, inputs)
    (c2 * torch.arange(8, dtype=c2.dtype)).sum().backward()
    torch.testing.assert_close(c1, c2, rtol=1e-10, atol=1e-12)
    for first, second in zip(checkpointed.parameters(), direct.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_importance_weights_fail_instead_of_falling_back(weight):
    inputs = list(_debug_inputs())
    inputs[-1] = inputs[-1].clone()
    inputs[-1][0] = weight
    model = GraphOptimizedConductance(5).double()
    with pytest.raises(RuntimeError, match="positive sampling weights"):
        _run(model, inputs)


def test_nonfinite_compatibility_fails_explicitly():
    inputs = list(_debug_inputs())
    inputs[0] = torch.full_like(inputs[0], float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite"):
        _run(GraphOptimizedConductance(5).double(), inputs)


@pytest.mark.parametrize(
    "name",
    [
        "solver_step_size",
        "solver_entropy",
        "solver_degree_barrier",
        "cost_bound",
    ],
)
@pytest.mark.parametrize("value", [True, "0.25", None, float("nan"), float("inf")])
def test_invalid_real_solver_configuration_fails_clearly(name, value):
    with pytest.raises(ValueError, match=name):
        GraphOptimizedConductance(5, **{name: value})


def test_step_checkpoint_reduces_saved_cpu_debug_tensor_storage(monkeypatch):
    torch.manual_seed(32)
    inputs = _debug_inputs()
    model = GraphOptimizedConductance(5, solver_steps=8).double()
    original_checkpoint = torch.utils.checkpoint.checkpoint
    calls = []

    def recording(function, *arguments, **kwargs):
        calls.append(function.__name__)
        return original_checkpoint(function, *arguments, **kwargs)

    def saved_storage_bytes():
        storages = {}

        def remember(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(remember, lambda tensor: tensor):
            output = _run(model, inputs)
        assert output.grad_fn is not None
        return sum(storages.values())

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recording)
    checkpoint_bytes = saved_storage_bytes()
    assert calls.count("_step") == 8
    monkeypatch.setattr(
        torch.utils.checkpoint,
        "checkpoint",
        lambda function, *arguments, **_kwargs: function(*arguments),
    )
    direct_bytes = saved_storage_bytes()
    assert checkpoint_bytes < direct_bytes


def test_bfloat16_autocast_keeps_solver_geometry_float32_and_finite_gradients():
    torch.manual_seed(2)
    inputs = _debug_inputs(dtype=torch.float32)
    model = GraphOptimizedConductance(5, edge_chunk_size=3)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        c = _run(model, inputs)
        loss = (c * torch.arange(8, dtype=c.dtype)).sum()
    assert c.dtype == torch.float32
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_zero_barrier_converges_toward_analytic_entropy_minimizer():
    torch.manual_seed(15)
    inputs = _debug_inputs()
    model = GraphOptimizedConductance(5, solver_degree_barrier=0, solver_steps=128).double()
    with torch.no_grad():
        actual = _run(model, inputs)
    delta = model.last_scores
    edge_graph = inputs[2][inputs[1][0]]
    raw = (-delta / model.solver_entropy).exp()
    expected = raw / graph_weighted_mean(raw, edge_graph, 3, inputs[-1])[edge_graph]
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    assert model.last_solver_diagnostics["projected_gradient_rms_final"].max() < 1e-9


def test_signed_metric_can_prefer_either_similar_or_dissimilar_endpoints():
    # The same endpoints receive opposite costs when the learned metric flips.
    # This would fail for a hard-coded positive distance/homophily energy.
    model = GraphOptimizedConductance(3).double()
    projected = torch.eye(3, dtype=torch.float64)
    tail, head = torch.tensor([0, 0]), torch.tensor([0, 1])
    degree = torch.ones(3, dtype=torch.float64)
    edge_graph = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        model.structure_metric.zero_()
        positive = model._compatibility_chunk(
            projected,
            torch.ones(1, 3, dtype=torch.float64),
            tail,
            head,
            degree,
            degree,
            edge_graph,
        )
        negative = model._compatibility_chunk(
            projected,
            -torch.ones(1, 3, dtype=torch.float64),
            tail,
            head,
            degree,
            degree,
            edge_graph,
        )
    assert positive[1] > positive[0]
    assert negative[1] < negative[0]
