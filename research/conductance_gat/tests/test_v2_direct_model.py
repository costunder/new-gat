"""Small numerical fixtures, not CPU research runs or substitute datasets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.conductance_gat.v2.model import DirectCNodeClassifier, DirectEdgeConductance
from research.conductance_gat.v2.operator import chunked_normalized_propagation


def topology():
    # Two components plus isolated node 6; sorted, unique physical edges.
    return torch.tensor([[0, 0, 0, 1, 1, 2, 4], [1, 2, 3, 2, 3, 3, 5]])


def fixture():
    generator = torch.Generator().manual_seed(401)
    state = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    c = torch.rand(7, generator=generator, dtype=torch.float64) + 0.2
    return state, c, topology()


def dense_reference(state, c, incidence, *, detached_degree=False):
    # Dense B exists only in this tiny independent mathematical test oracle.
    b = state.new_zeros((c.numel(), state.shape[0]))
    rows = torch.arange(c.numel())
    b[rows, incidence[0]] = -1
    b[rows, incidence[1]] = 1
    degree = b.abs().T @ c
    safe_degree = torch.where(degree > 0, degree, torch.ones_like(degree))
    if detached_degree:
        safe_degree = safe_degree.detach()
    return state - 0.95 * (b.T @ (c[:, None] * (b @ state))) / safe_degree[:, None]


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 65536])
def test_chunked_output_and_both_gradients_match_dense(chunk_size):
    state, c, incidence = fixture()
    state.requires_grad_()
    c.requires_grad_()
    actual = chunked_normalized_propagation(state, c, incidence, edge_chunk_size=chunk_size)
    expected = dense_reference(state, c, incidence)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
    actual_grads = torch.autograd.grad(actual.square().sum(), (state, c))
    expected_grads = torch.autograd.grad(expected.square().sum(), (state, c))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-14, rtol=1e-12)


def test_fp64_gradcheck():
    state, c, incidence = fixture()
    assert torch.autograd.gradcheck(
        lambda h, weights: chunked_normalized_propagation(
            h, weights, incidence, edge_chunk_size=2
        ),
        (state.requires_grad_(), c.requires_grad_()),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_degree_derivative_is_not_detached():
    state, c, incidence = fixture()
    c.requires_grad_()
    actual = chunked_normalized_propagation(state, c, incidence, edge_chunk_size=2)
    detached = dense_reference(state, c, incidence, detached_degree=True)
    actual_grad = torch.autograd.grad(actual.square().sum(), c)[0]
    detached_grad = torch.autograd.grad(detached.square().sum(), c)[0]
    assert not torch.allclose(actual_grad, detached_grad)
    # A uniform log-C displacement is a scale gauge, so its directional gradient is zero.
    torch.testing.assert_close((actual_grad * c).sum(), c.new_zeros(()), atol=1e-13, rtol=0)


@pytest.mark.parametrize("num_edges", [0, 1])
def test_isolates_and_single_edge_cancellation(num_edges):
    state = torch.tensor([[1.0, -2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    incidence = torch.tensor([[0], [1]], dtype=torch.long)[:, :num_edges]
    c = torch.tensor([2.0], requires_grad=True)[:num_edges]
    result = chunked_normalized_propagation(state, c, incidence, edge_chunk_size=1)
    torch.testing.assert_close(result[2], state[2], rtol=0, atol=0)
    if num_edges == 0:
        torch.testing.assert_close(result, state, rtol=0, atol=0)
    else:
        torch.testing.assert_close(result[0], 0.05 * state[0] + 0.95 * state[1])
        torch.testing.assert_close(result[1], 0.05 * state[1] + 0.95 * state[0])
    grad_state, grad_c = torch.autograd.grad(result.sum(), (state, c))
    torch.testing.assert_close(grad_state, torch.ones_like(state))
    torch.testing.assert_close(grad_c, torch.zeros_like(c), rtol=0, atol=1e-6)


def test_constants_scale_orientation_and_edge_order_invariance_of_kernel():
    state, c, incidence = fixture()
    original = chunked_normalized_propagation(state, c, incidence, edge_chunk_size=2)
    torch.testing.assert_close(
        chunked_normalized_propagation(state, 31 * c, incidence, edge_chunk_size=2), original
    )
    flipped = incidence.clone()
    flipped[:, ::2] = flipped.flip(0)[:, ::2]
    torch.testing.assert_close(chunked_normalized_propagation(state, c, flipped), original)
    order = torch.tensor([6, 3, 2, 0, 1, 5, 4])
    torch.testing.assert_close(
        chunked_normalized_propagation(state, c[order], incidence[:, order]), original
    )
    constant = state.new_full(state.shape, 3.5)
    torch.testing.assert_close(chunked_normalized_propagation(constant, c, incidence), constant)


def test_saved_tensors_do_not_include_full_edge_feature_or_dense_matrix():
    num_nodes, width = 9, 5
    incidence = torch.combinations(torch.arange(num_nodes), 2).T.contiguous()
    num_edges = incidence.shape[1]
    state = torch.randn(num_nodes, width, requires_grad=True)
    c = torch.ones(num_edges, requires_grad=True)
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = chunked_normalized_propagation(state, c, incidence, edge_chunk_size=3)
        output.sum().backward()
    assert (num_edges, width) not in saved
    assert (num_nodes, num_nodes) not in saved
    assert (num_edges, num_nodes) not in saved
    assert (num_edges, num_edges) not in saved
    assert saved.count((num_nodes, width)) == 2  # H and weighted neighbor mean.
    assert (2, num_edges) in saved


def test_double_backward_is_explicitly_unsupported():
    state, c, incidence = fixture()
    c.requires_grad_()
    result = chunked_normalized_propagation(state, c, incidence)
    grad_c = torch.autograd.grad(result.square().sum(), c, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad_c.sum(), c)


def model(mode="direct", **kwargs):
    torch.manual_seed(42)
    return DirectCNodeClassifier(
        4,
        3,
        incidence=topology(),
        num_nodes=7,
        gate_mode=mode,
        hidden_channels=5,
        layers=2,
        dropout=0.0,
        edge_chunk_size=2,
        **kwargs,
    )


def graph():
    generator = torch.Generator().manual_seed(23)
    return SimpleNamespace(
        x=torch.randn(7, 4, generator=generator), incidence_edge_index=topology()
    )


def test_direct_and_fixed_initial_state_counts_and_outputs_match():
    direct, fixed = model(), model("fixed_one")
    assert direct.state_dict().keys() == fixed.state_dict().keys()
    for name, value in direct.state_dict().items():
        torch.testing.assert_close(value, fixed.state_dict()[name], rtol=0, atol=0)
    gates = [(name, p) for name, p in direct.named_parameters() if ".estimator." in name]
    assert [name for name, _ in gates] == [
        "operators.0.estimator.log_c",
        "operators.1.estimator.log_c",
    ]
    assert sum(p.numel() for _, p in gates) == 14
    assert sum(p.numel() for p in direct.parameters() if p.requires_grad) - sum(
        p.numel() for p in fixed.parameters() if p.requires_grad
    ) == 14
    assert all(not isinstance(op.estimator, nn.Sequential) for op in direct.operators)
    for operator in direct.operators:
        torch.testing.assert_close(operator.estimator(), torch.ones(7), rtol=0, atol=0)
    torch.testing.assert_close(direct(graph()), fixed(graph()), rtol=0, atol=0)


def test_task_loss_reaches_each_direct_log_c_but_not_fixed():
    direct, fixed = model(), model("fixed_one")
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 1])
    for classifier in (direct, fixed):
        nn.functional.cross_entropy(classifier(graph()), labels).backward()
        assert classifier.encoder.weight.grad is not None
    for operator in direct.operators:
        grad = operator.estimator.log_c.grad
        assert grad is not None and bool(torch.isfinite(grad).all())
        assert float(grad.abs().sum()) > 0
    for operator in fixed.operators:
        assert operator.estimator.log_c.grad is None
        assert not operator.estimator.log_c.requires_grad


def test_c_is_direct_and_not_a_function_of_node_inputs():
    classifier = model()
    with torch.no_grad():
        classifier.operators[0].estimator.log_c.copy_(torch.linspace(-1, 1, 7))
    before = classifier.operators[0].estimator().clone()
    batch = graph()
    classifier(batch)
    batch.x.mul_(5)
    classifier(batch)
    torch.testing.assert_close(classifier.operators[0].estimator(), before, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["order", "orientation", "node_count", "batch", "edge"])
def test_changed_graph_topology_is_rejected(change):
    classifier, batch = model(), graph()
    if change == "order":
        batch.incidence_edge_index = batch.incidence_edge_index.flip(1)
    elif change == "orientation":
        batch.incidence_edge_index = batch.incidence_edge_index.flip(0)
    elif change == "node_count":
        batch.x = batch.x[:-1]
    elif change == "batch":
        batch.batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    else:
        batch.incidence_edge_index[1, -1] = 6
    with pytest.raises(ValueError, match="bound|batches"):
        classifier(batch)


@pytest.mark.parametrize("change", ["order", "edge", "num_nodes", "missing"])
def test_bad_checkpoint_topology_rejected_before_any_weights_change(change):
    classifier = model()
    before = {name: value.clone() for name, value in classifier.state_dict().items()}
    state = {name: value.clone() for name, value in before.items()}
    state["encoder.weight"].fill_(123)
    if change == "order":
        state["bound_incidence"] = state["bound_incidence"].flip(1)
    elif change == "edge":
        state["bound_incidence"][1, -1] = 6
    elif change == "num_nodes":
        state["bound_num_nodes"] += 1
    else:
        del state["bound_incidence"]
    with pytest.raises(RuntimeError, match="topology"):
        classifier.load_state_dict(state, strict=False)
    for name, value in classifier.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_matching_checkpoint_and_nested_topology_guard():
    source, destination = model(), model()
    with torch.no_grad():
        source.operators[0].estimator.log_c.add_(0.3)
    destination.load_state_dict(source.state_dict())
    torch.testing.assert_close(source(graph()), destination(graph()))
    parent = nn.ModuleDict({"classifier": destination})
    state = {name: value.clone() for name, value in parent.state_dict().items()}
    state["classifier.bound_incidence"] = state["classifier.bound_incidence"].flip(1)
    with pytest.raises(RuntimeError, match="topology"):
        parent.load_state_dict(state)


@pytest.mark.parametrize("kind", ["reverse", "duplicate", "unsorted", "selfloop", "bounds"])
def test_constructor_requires_canonical_physical_edges(kind):
    incidence = topology()
    if kind == "reverse":
        incidence = incidence.flip(0)
    elif kind == "duplicate":
        incidence[:, 1] = incidence[:, 0]
    elif kind == "unsorted":
        incidence = incidence.flip(1)
    elif kind == "selfloop":
        incidence[1, 0] = incidence[0, 0]
    else:
        incidence[1, -1] = 7
    with pytest.raises(ValueError):
        DirectCNodeClassifier(4, 3, incidence=incidence, num_nodes=7)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1000.0, -1000.0])
def test_direct_exp_fails_loudly_without_clipping(value):
    estimator = DirectEdgeConductance(2)
    with torch.no_grad():
        estimator.log_c[0] = value
    with pytest.raises(FloatingPointError, match="overflow/underflow"):
        estimator()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_kernel_rejects_invalid_conductance(value):
    state, c, incidence = fixture()
    c[0] = value
    with pytest.raises(FloatingPointError, match="strictly positive"):
        chunked_normalized_propagation(state, c, incidence)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk_size_rejected(chunk_size):
    state, c, incidence = fixture()
    with pytest.raises(ValueError, match="positive integer"):
        chunked_normalized_propagation(state, c, incidence, edge_chunk_size=chunk_size)


def test_finite_c_degree_overflow_is_not_silently_hidden():
    state = torch.ones(3, 2)
    incidence = torch.tensor([[0, 0], [1, 2]])
    c = torch.full((2,), torch.finfo(torch.float32).max * 0.75)
    with pytest.raises(FloatingPointError, match="degree overflow"):
        chunked_normalized_propagation(state, c, incidence)
