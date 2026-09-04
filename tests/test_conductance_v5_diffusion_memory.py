"""Synthetic CPU regression tests; these are not GPU/full-training results."""

from __future__ import annotations

from itertools import product

import pytest
import torch

from research.conductance_gat.v5.operator import shared_head_diffusion


def _inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(519)
    # A triangle, a separate edge and two isolates, across two batched graphs.
    incidence = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]])
    node_graph = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    message = torch.randn(7, 2, 3, generator=generator, dtype=dtype)
    conductance = torch.tensor([0.7, 1.3, 0.9, 1.1], dtype=dtype)
    beta = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=dtype)
    correction = torch.tensor([1.4, 0.6, 1.0, 1.2], dtype=dtype)
    return (message, conductance, beta, correction), incidence, node_graph


def _dense_reference(message, conductance, beta, correction, incidence, node_graph):
    """Independent dense definition, intentionally used only on tiny test graphs."""
    compute_dtype = (
        torch.float32 if message.dtype in {torch.bfloat16, torch.float16} else message.dtype
    )
    values = message.to(compute_dtype)
    edge_weight = conductance.to(compute_dtype) * correction.to(compute_dtype)
    tail, head = incidence
    adjacency = values.new_zeros((message.shape[0], message.shape[0]))
    adjacency = adjacency.index_put((tail, head), edge_weight, accumulate=True)
    adjacency = adjacency.index_put((head, tail), edge_weight, accumulate=True)
    degree = adjacency.sum(dim=1)
    active = degree > 0
    safe_degree = torch.where(active, degree, torch.ones_like(degree))
    inverse = safe_degree.rsqrt() * active.to(compute_dtype)
    transition = inverse[:, None] * adjacency * inverse[None, :]
    propagated = (transition @ values.flatten(1)).reshape_as(values)
    output = values + beta.to(compute_dtype)[node_graph, :, None] * (
        propagated - active[:, None, None] * values
    )
    return output.to(message.dtype)


def _actual(arguments, incidence, node_graph, chunk):
    message, conductance, beta, correction = arguments
    return shared_head_diffusion(
        message,
        conductance,
        incidence,
        node_graph,
        beta,
        sampling_correction=correction,
        edge_chunk_size=chunk,
    )


@pytest.mark.parametrize("chunk", [1, 3, 100])
@pytest.mark.parametrize(
    "requires_grad",
    [
        (True, True, True, True),
        (False, True, False, True),
        (True, False, True, False),
        (False, True, False, False),
        (True, False, False, False),
    ],
)
def test_diffusion_matches_dense_forward_and_all_active_gradients(chunk, requires_grad):
    inputs, incidence, node_graph = _inputs()
    actual_inputs = tuple(
        value.clone().requires_grad_(enabled)
        for value, enabled in zip(inputs, requires_grad, strict=True)
    )
    expected_inputs = tuple(
        value.clone().requires_grad_(enabled)
        for value, enabled in zip(inputs, requires_grad, strict=True)
    )
    actual = _actual(actual_inputs, incidence, node_graph, chunk)
    expected = _dense_reference(*expected_inputs, incidence, node_graph)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual[5:], actual_inputs[0][5:], rtol=0, atol=0)
    probe = torch.linspace(-1.0, 1.0, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    actual_grads = torch.autograd.grad(
        (actual * probe).sum() + actual.square().mean(),
        [value for value in actual_inputs if value.requires_grad],
    )
    expected_grads = torch.autograd.grad(
        (expected * probe).sum() + expected.square().mean(),
        [value for value in expected_inputs if value.requires_grad],
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)
        assert torch.isfinite(actual_grad).all()


def test_diffusion_double_precision_gradcheck_and_gradgradcheck():
    inputs, incidence, node_graph = _inputs()
    inputs = tuple(value.requires_grad_(True) for value in inputs)

    def function(*arguments):
        return _actual(arguments, incidence, node_graph, 3)

    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    # Second-order derivatives are deliberately supported, not silently detached.
    assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)


@pytest.mark.parametrize("empty_edges", [False, True])
def test_zero_degree_and_edgeless_graphs_have_finite_exact_identity_gradients(empty_edges):
    inputs, incidence, node_graph = _inputs()
    message, conductance, beta, correction = inputs
    if empty_edges:
        incidence = incidence[:, :0]
        conductance, correction = conductance[:0], correction[:0]
    else:
        conductance = torch.zeros_like(conductance)
    inputs = tuple(value.requires_grad_(True) for value in (message, conductance, beta, correction))
    actual = _actual(inputs, incidence, node_graph, 3)
    torch.testing.assert_close(actual, message, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.sum(), inputs)
    torch.testing.assert_close(gradients[0], torch.ones_like(message), rtol=0, atol=0)
    for gradient in gradients[1:]:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
        assert torch.isfinite(gradient).all()


def test_bfloat16_uses_fp32_geometry_and_preserves_input_gradient_dtypes():
    inputs, incidence, node_graph = _inputs(torch.float32)
    inputs = tuple(value.to(torch.bfloat16).requires_grad_(True) for value in inputs)
    saved = []

    def pack(tensor):
        saved.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = _actual(inputs, incidence, node_graph, 3)
    # The dense reference is evaluated outside autocast to retain FP32 geometry.
    expected = _dense_reference(*inputs, incidence, node_graph)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    saved_features = [dtype for shape, dtype in saved if tuple(shape) == (7, 2, 3)]
    assert saved_features and all(dtype == torch.float32 for dtype in saved_features)
    gradients = torch.autograd.grad(actual.float().square().sum(), inputs)
    expected_gradients = torch.autograd.grad(expected.float().square().sum(), inputs)
    assert all(gradient.dtype == torch.bfloat16 for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("chunk,c_requires_grad", list(product([11, 37, 500], [False, True])))
def test_first_order_saved_feature_tensors_are_nodes_not_accumulated_edge_chunks(
    chunk, c_requires_grad
):
    generator = torch.Generator().manual_seed(321)
    nodes, edges, heads, width = 23, 173, 3, 5
    incidence = torch.randint(nodes, (2, edges), generator=generator)
    message = torch.randn(nodes, heads, width, generator=generator, requires_grad=True)
    conductance = torch.linspace(0.7, 1.3, edges, requires_grad=c_requires_grad)
    beta = torch.full((1, heads), 0.3, requires_grad=True)
    correction = torch.linspace(1.0, 1.5, edges, requires_grad=True)
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.numel()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = _actual(
            (message, conductance, beta, correction),
            incidence,
            torch.zeros(nodes, dtype=torch.long),
            chunk,
        )
    feature_tensors = [shape for shape, _ in saved if len(shape) == 3 and shape[-1] == width]
    assert feature_tensors
    assert all(shape == (nodes, heads, width) for shape in feature_tensors)
    # Storage is node-feature tensors plus scalar/index edge data, independent
    # of E * heads * width and without an accumulated per-chunk feature graph.
    assert sum(numel for _, numel in saved) < 8 * nodes * heads * width + 24 * edges
    actual.square().mean().backward()
    assert message.grad is not None and torch.isfinite(message.grad).all()
    if c_requires_grad:
        assert conductance.grad is not None and torch.isfinite(conductance.grad).all()


def test_calibration_frozen_message_still_receives_conductance_gradient():
    inputs, incidence, node_graph = _inputs()
    message, conductance, beta, correction = inputs
    conductance.requires_grad_(True)
    output = _actual((message, conductance, beta, correction), incidence, node_graph, 3)
    output.square().mean().backward()
    assert conductance.grad is not None
    assert torch.isfinite(conductance.grad).all()
    assert conductance.grad[:3].abs().sum() > 0
    assert message.grad is None
