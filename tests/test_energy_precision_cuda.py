"""CUDA-only synthetic checks of the actual energy caller's precision contract."""

import copy

import pytest
import torch

from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.gram import GramReadout
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401
from tests.test_fused_integration_cuda import model_and_graph


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("implementation", ["reference", "fused"])
@pytest.mark.parametrize(
    "arm",
    [
        "incidence_shared_energy",
        "incidence_fixed_energy",
        "incidence_diagonal",
        "incidence_energy_pre_lift",
    ],
)
def test_actual_energy_readout_stays_fp32_through_backward(
    arm, implementation, precision, checkpoint, monkeypatch, record_property
):
    net, graph, args, device = model_and_graph(arm, precision, implementation)
    assert net.contract()["energy_readout_precision_policy"] == "fp32_autocast_disabled_v1"
    assert (
        engine.configuration(args)["comparison_contract"]["energy_readout_precision_policy"]
        == "fp32_autocast_disabled_v1"
    )
    net.activation_checkpoint = checkpoint
    calls = []
    original_einsum, original_apply = torch.einsum, GramReadout.apply

    def check(operation, *operands):
        assert not torch.is_autocast_enabled("cuda"), "energy readout must disable AMP"
        assert all(value.dtype == torch.float32 for value in operands[:3])
        result = operation(*operands)
        assert result.dtype == torch.float32
        calls.append(result.dtype)
        return result

    def traced_einsum(equation, *operands, **kwargs):
        if equation == "nhp,hpd->nhd" and implementation == "reference":
            return check(lambda *values: original_einsum(equation, *values, **kwargs), *operands)
        return original_einsum(equation, *operands, **kwargs)

    monkeypatch.setattr(torch, "einsum", traced_einsum)
    monkeypatch.setattr(GramReadout, "apply", lambda *values: check(original_apply, *values))
    graph.x.requires_grad_()
    with engine.autocast(args, device):
        output = net(graph)
        loss = output.float().square().mean()
    forward_calls = len(calls)
    assert forward_calls == net.depth == 8
    loss.backward()
    assert len(calls) >= forward_calls
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.isfinite(graph.x.grad).all()
    record_property("readout_dtype", "float32")
    record_property("forward_readout_calls", forward_calls)
    record_property("checkpoint", checkpoint)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("implementation", ["reference", "fused"])
def test_checkpoint_preserves_near_zero_relu_and_gradients(
    precision, implementation, monkeypatch, record_property
):
    net, graph, args, device = model_and_graph(
        "incidence_energy_pre_lift", precision, implementation
    )
    net.activation_checkpoint = False
    recomputed = copy.deepcopy(net)
    recomputed.activation_checkpoint = True
    original_relu = torch.nn.functional.relu
    activations = []

    def trace(value, *args, **kwargs):
        if value.shape == (256, 256):
            activations.append(value.detach().clone())
        return original_relu(value, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "relu", trace)
    graph.x.requires_grad_()
    with engine.autocast(args, device):
        expected = net(graph)
        ordinary_activations = activations[:]
        activations.clear()
        actual = recomputed(graph)
    assert len(ordinary_activations) == len(activations) == 8
    for left, right in zip(ordinary_activations, activations, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    values = torch.stack(ordinary_activations).float()
    # Exercise both ReLU regions and actual states close to its kink, rather
    # than accepting a checkpoint comparison with an inactive energy/lift.
    nearby = (values.abs() < 1e-3) & (values != 0)
    assert nearby.any() and (values[nearby] > 0).any() and (values[nearby] < 0).any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (graph.x, *recomputed.parameters()), direction)
    gb = torch.autograd.grad(expected, (graph.x, *net.parameters()), direction)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    record_property("nonzero_relu_inputs_within_1e-3", nearby.sum().item())
    record_property("minimum_absolute_relu_input", values.abs().min().item())


@pytest.mark.parametrize(
    "needs",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_recomputed_readout_saves_inputs_and_supports_partial_gradients(needs):
    from experiments.aggregation_comparison.gram import LocalGram

    torch.manual_seed(91)
    values = [
        torch.randn(3, 13, 2, 4, device="cuda"),
        torch.rand(14, 1, device="cuda"),
        torch.randn(2, 6, 4, device="cuda"),
    ]
    for value, needed in zip(values, needs, strict=True):
        value.requires_grad_(needed)
    edges = torch.randint(0, 13, (2, 14), device="cuda")
    actual = GramReadout.apply(*values, edges, 3, False)
    saved = actual.grad_fn.saved_tensors
    assert len(saved) == 4
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(saved, [*values, edges], strict=True))
    expected = torch.einsum(
        "nhp,hpd->nhd", LocalGram.apply(values[0], values[1], edges, 3, False), values[2]
    )
    inputs = [v for v, needed in zip(values, needs, strict=True) if needed]
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, inputs, direction)
    gb = torch.autograd.grad(expected, inputs, direction)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
