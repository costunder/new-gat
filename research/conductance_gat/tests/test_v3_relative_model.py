"""Tiny mathematical fixtures; no CPU research training or substitute datasets."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.conductance_gat.v3.model import (
    RelativeCNodeClassifier,
    RelativeConductance,
    graph_mean,
)
from research.conductance_gat.v3.operator import symmetric_propagation


def fixture():
    generator = torch.Generator().manual_seed(810)
    state = torch.randn(8, 5, dtype=torch.float64, generator=generator)
    c = torch.rand(7, dtype=torch.float64, generator=generator) + 0.2
    incidence = torch.tensor([[0, 0, 1, 3, 3, 4, 5], [1, 2, 2, 4, 5, 6, 6]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    return state, c, incidence, batch


def dense_reference(state, c, incidence, alpha, *, detach_degree=False):
    b = state.new_zeros((c.numel(), state.shape[0]))
    rows = torch.arange(c.numel())
    b[rows, incidence[0]], b[rows, incidence[1]] = -1, 1
    degree = b.abs().T @ c
    active = degree > 0
    inverse = torch.where(active, degree, torch.ones_like(degree)).rsqrt() * active
    if detach_degree:
        inverse = inverse.detach()
    normalized = inverse[:, None] * state
    return state - alpha * inverse[:, None] * (b.T @ (c[:, None] * (b @ normalized)))


@pytest.mark.parametrize("chunk", [1, 2, 4, 65536])
def test_symmetric_output_and_all_gradients_match_dense(chunk):
    state, c, incidence, _ = fixture()
    state.requires_grad_()
    c.requires_grad_()
    alpha = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    actual = symmetric_propagation(state, c, incidence, alpha, edge_chunk_size=chunk)
    expected = dense_reference(state, c, incidence, alpha)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    actual_grads = torch.autograd.grad(actual.square().sum(), (state, c, alpha))
    expected_grads = torch.autograd.grad(expected.square().sum(), (state, c, alpha))
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, atol=2e-13, rtol=1e-11)


def test_symmetric_fp64_gradcheck():
    state, c, incidence, _ = fixture()
    assert torch.autograd.gradcheck(
        lambda h, weights, a: symmetric_propagation(h, weights, incidence, a, edge_chunk_size=2),
        (
            state.requires_grad_(),
            c.requires_grad_(),
            torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
        ),
    )


def test_degree_terms_are_present_and_common_scale_gradient_cancels():
    state, c, incidence, _ = fixture()
    c.requires_grad_()
    alpha = state.new_tensor(0.5)
    actual = symmetric_propagation(state, c, incidence, alpha)
    wrong = dense_reference(state, c, incidence, alpha, detach_degree=True)
    grad_c = torch.autograd.grad(actual.square().sum(), c)[0]
    wrong_grad = torch.autograd.grad(wrong.square().sum(), c)[0]
    assert not torch.allclose(grad_c, wrong_grad)
    torch.testing.assert_close((grad_c * c).sum(), c.new_zeros(()), atol=1e-12, rtol=0)


def test_isolates_empty_edges_and_alpha_zero():
    state = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    c = torch.empty(0, dtype=torch.float64, requires_grad=True)
    incidence = torch.empty((2, 0), dtype=torch.long)
    alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    result = symmetric_propagation(state, c, incidence, alpha)
    torch.testing.assert_close(result, state, rtol=0, atol=0)
    grads = torch.autograd.grad(result.sum(), (state, c, alpha))
    torch.testing.assert_close(grads[0], torch.ones_like(state), rtol=0, atol=0)
    assert grads[1].numel() == 0 and grads[2] == 0
    h, weights, edges, _ = fixture()
    torch.testing.assert_close(
        symmetric_propagation(h, weights, edges, h.new_tensor(0)), h, rtol=0, atol=0
    )


def test_symmetric_kernel_invariances_and_not_constant_preserving():
    state, c, incidence, _ = fixture()
    alpha = state.new_tensor(0.5)
    expected = symmetric_propagation(state, c, incidence, alpha)
    torch.testing.assert_close(
        symmetric_propagation(state, 11 * c, incidence.flip(0), alpha), expected
    )
    order = torch.tensor([4, 6, 1, 0, 2, 5, 3])
    torch.testing.assert_close(
        symmetric_propagation(state, c[order], incidence[:, order], alpha), expected
    )
    constant = torch.ones_like(state)
    result = symmetric_propagation(constant, c, incidence, alpha)
    assert not torch.allclose(result, constant)
    torch.testing.assert_close(result[-1], constant[-1], atol=0, rtol=0)


def estimator(chunk=2, *, zero=False, mode="relative"):
    torch.manual_seed(53)
    gate = RelativeConductance(5, mode, chunk).double()
    if not zero:
        with torch.no_grad():
            gate.network[-1].weight.copy_(torch.tensor([[0.3, -0.4, 0.5, -0.8, 0.2]]))
    return gate


def test_centered_relative_formula_uses_full_graph_means():
    state, _, incidence, batch = fixture()
    gate = estimator(chunk=2)
    result = gate(state, incidence, batch, 2)
    edge_graph = batch[incidence[0]]
    scores = gate.last_scores
    centered = scores - graph_mean(scores, edge_graph, 2)[edge_graph]
    unnormalized = (gate.tau * centered.tanh()).exp()
    expected = (
        1
        - gate.gamma
        + gate.gamma * (unnormalized / graph_mean(unnormalized, edge_graph, 2)[edge_graph])
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(graph_mean(result, edge_graph, 2), torch.ones(2).double())
    torch.testing.assert_close(
        graph_mean(gate.last_centered_scores, edge_graph, 2),
        torch.zeros(2).double(),
        atol=1e-15,
        rtol=0,
    )
    assert float(result.detach().std()) > 0


@pytest.mark.parametrize("chunk", [1, 3, 65536])
def test_gate_checkpoint_chunks_match_outputs_and_all_gradients(chunk):
    state, _, incidence, batch = fixture()
    state.requires_grad_()
    gate = estimator(chunk=chunk)
    reference = estimator(chunk=65536)
    actual = gate(state, incidence, batch, 2)
    expected = reference(state, incidence, batch, 2)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    coefficient = torch.linspace(0.2, 1.4, 7).double()
    actual_grads = torch.autograd.grad((actual * coefficient).sum(), (state, *gate.parameters()))
    expected_grads = torch.autograd.grad(
        (expected * coefficient).sum(), (state, *reference.parameters())
    )
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, atol=1e-13, rtol=1e-10)


def test_estimator_feature_orientation_permutation_and_batch_independence():
    state, _, incidence, batch = fixture()
    gate = estimator()
    expected = gate(state, incidence, batch, 2)
    flipped = incidence.clone()
    flipped[:, ::2] = flipped.flip(0)[:, ::2]
    torch.testing.assert_close(gate(state, flipped, batch, 2), expected)
    order = torch.tensor([4, 6, 1, 0, 2, 5, 3])
    torch.testing.assert_close(gate(state, incidence[:, order], batch, 2), expected[order])
    changed = state.clone()
    changed[batch == 1] *= 17
    torch.testing.assert_close(gate(changed, incidence, batch, 2)[:3], expected[:3])
    separate = gate(state[:3], incidence[:, :3], batch[:3], 1)
    torch.testing.assert_close(separate, expected[:3])


def classifier(mode="relative", **kwargs):
    torch.manual_seed(79)
    return RelativeCNodeClassifier(
        5, 3, hidden_channels=5, layers=2, dropout=0, gate_mode=mode, edge_chunk_size=2, **kwargs
    ).double()


def graph():
    state, _, incidence, batch = fixture()
    return SimpleNamespace(x=state, incidence_edge_index=incidence, batch=batch)


def test_fixed_relative_initial_state_and_outputs_match_and_alpha_is_active():
    relative, fixed = classifier(), classifier("fixed_one")
    from research.conductance_gat.ablation.model import shared_backbone_state_sha256

    assert shared_backbone_state_sha256(relative) == shared_backbone_state_sha256(fixed)
    torch.testing.assert_close(relative(graph()), fixed(graph()), atol=0, rtol=0)
    for model in (relative, fixed):
        for operator in model.operators:
            assert operator.raw_alpha.requires_grad
            assert operator.alpha == 0.5
            if model.gate_mode == "relative":
                assert operator.estimator.gamma == 0.5
                assert operator.estimator.tau == 1
            else:
                assert operator.estimator.gamma is None
                assert operator.estimator.tau is None
            batch = graph()
            c = operator.estimator(batch.x, batch.incidence_edge_index, batch.batch, 2)
            torch.testing.assert_close(c, torch.ones_like(c), atol=0, rtol=0)
    assert all(list(op.estimator.parameters()) == [] for op in fixed.operators)
    assert all(p.requires_grad for op in relative.operators for p in op.estimator.parameters())


def test_gate_final_weight_and_alpha_receive_task_gradients_at_c1_initialization():
    relative, fixed = classifier(), classifier("fixed_one")
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
    for model in (relative, fixed):
        nn.functional.cross_entropy(model(graph()), labels).backward()
        assert any(float(op.raw_alpha.grad.abs()) > 0 for op in model.operators)
    for operator in relative.operators:
        gradient = operator.estimator.network[-1].weight.grad
        assert gradient is not None and float(gradient.abs().sum()) > 0
        # Exact C=1 starts gamma/tau and earlier gate layers at zero task gradients,
        # not a frozen estimator: final weights can move on the first update.
        assert operator.estimator.raw_gamma.grad == 0
        assert operator.estimator.raw_tau.grad == 0
    for operator in fixed.operators:
        assert all(p.grad is None for p in operator.estimator.parameters())


def test_combined_c_of_h_gradient_matches_dense_reference():
    state, _, incidence, batch = fixture()
    state.requires_grad_()
    gate = estimator()
    alpha = state.new_tensor(0.4, requires_grad=True)
    c = gate(state, incidence, batch, 2)
    output = symmetric_propagation(state, c, incidence, alpha, edge_chunk_size=2)
    expected = dense_reference(state, c, incidence, alpha)
    parameters = (state, alpha, *gate.parameters())
    actual_grads = torch.autograd.grad(output.square().sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)


def test_shared_model_is_node_permutation_equivariant_and_accepts_new_graph():
    model = classifier()
    for operator in model.operators:
        with torch.no_grad():
            operator.estimator.network[-1].weight.fill_(0.2)
    batch = graph()
    original = model(batch)
    order = torch.tensor([3, 0, 4, 1, 5, 2, 6, 7])
    inverse = torch.argsort(order)
    permuted = SimpleNamespace(
        x=batch.x[order],
        incidence_edge_index=inverse[batch.incidence_edge_index],
        batch=batch.batch[order],
    )
    torch.testing.assert_close(model(permuted), original[order], atol=1e-12, rtol=1e-10)
    other = SimpleNamespace(x=batch.x[:3], incidence_edge_index=batch.incidence_edge_index[:, :3])
    assert model(other).shape == (3, 3)


def test_intervention_c_hook_recomputes_weighted_degree():
    model = classifier()
    operator = model.operators[0]
    state, c, incidence, batch = fixture()
    handle = operator.estimator.register_forward_hook(lambda module, args, output: c)
    try:
        actual = operator(state, incidence, batch, 2)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, dense_reference(state, c, incidence, operator.alpha))


def test_saved_tensors_exclude_full_edge_features_or_gate_hidden_states():
    nodes, width = 9, 5
    incidence = torch.combinations(torch.arange(nodes), 2).T.contiguous()
    edges = incidence.shape[1]
    state = torch.randn(nodes, width, dtype=torch.float64, requires_grad=True)
    gate = estimator(chunk=3)
    batch = torch.zeros(nodes, dtype=torch.long)
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        c = gate(state, incidence, batch, 1)
        result = symmetric_propagation(
            state, c, incidence, state.new_tensor(0.5), edge_chunk_size=3
        )
        result.square().sum().backward()
    assert (edges, width) not in saved
    assert (edges, 4 * width + 2) not in saved
    assert (nodes, nodes) not in saved
    assert (edges, edges) not in saved


def test_forward_diagnostics_are_detached_and_do_not_change_state_dict():
    gate = estimator()
    state, _, incidence, batch = fixture()
    before = copy.deepcopy(gate.state_dict())
    result = gate(state.requires_grad_(), incidence, batch, 2)
    assert result.requires_grad
    assert not gate.last_scores.requires_grad and not gate.last_centered_scores.requires_grad
    for name, value in gate.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_double_backward_is_unsupported():
    state, c, incidence, _ = fixture()
    c.requires_grad_()
    result = symmetric_propagation(state, c, incidence, state.new_tensor(0.5))
    grad = torch.autograd.grad(result.square().sum(), c, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad.sum(), c)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_alpha_rejected(alpha):
    state, c, incidence, _ = fixture()
    with pytest.raises(FloatingPointError, match="alpha"):
        symmetric_propagation(state, c, incidence, state.new_tensor(alpha))


@pytest.mark.parametrize("chunk", [0, -1, True, 0.5])
def test_invalid_chunk_rejected(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        RelativeConductance(5, edge_chunk_size=chunk)


def test_cross_graph_edges_rejected():
    batch = graph()
    batch.incidence_edge_index[1, 0] = 3
    with pytest.raises(ValueError, match="different graphs"):
        classifier()(batch)


def test_empty_graph_gate_and_node_model_remain_finite():
    gate = estimator()
    state = torch.randn(3, 5, dtype=torch.float64)
    incidence = torch.empty((2, 0), dtype=torch.long)
    assert gate(state, incidence, torch.zeros(3, dtype=torch.long), 1).shape == (0,)
    result = classifier()(SimpleNamespace(x=state, incidence_edge_index=incidence))
    assert bool(torch.isfinite(result).all())
