"""Actual model callers and passive diagnostics; explicit CUDA synthetic verification."""

import copy

import pytest
import torch

from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.mechanisms import MechanismCollector
from experiments.aggregation_comparison.model import ARMS
from tests.test_aggregation_comparison_cuda import (  # noqa: F401
    cuda_required,
    reference_arguments,
    synthetic_disjoint_batch,
)


def model_and_graph(arm, precision, implementation):
    engine.base._seed(377)
    device = torch.device("cuda:0")
    _, graph, payload = synthetic_disjoint_batch(device)
    args = reference_arguments(arm, precision)
    args.edge_chunk_size = 4096
    args.gram_implementation = implementation
    net = engine.make_model(payload, args, device)
    net.dropout = 0
    with torch.no_grad():
        for readout in net.energy_readouts:
            readout.normal_(0, 0.015)
    return net, graph, args, device


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("arm", [name for name in ARMS if name.startswith("incidence")])
def test_all_actual_fused_callers_output_input_and_parameter_gradients(arm, precision):
    net, graph, args, device = model_and_graph(arm, precision, "fused")
    reference = copy.deepcopy(net)
    reference.gram_implementation = "reference"
    graph.x.requires_grad_()
    with engine.autocast(args, device):
        actual, expected = net(graph), reference(graph)
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (graph.x, *net.parameters()), direction)
    gb = torch.autograd.grad(expected, (graph.x, *reference.parameters()), direction)
    tolerance = 0.05 if precision == "bf16" else 2e-4
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for left, right in zip(ga, gb, strict=True):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("implementation", ["reference", "fused"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize(
    "arm",
    [
        "incidence_energy_pre_lift",
        "incidence_diagonal",
        "incidence_shared_energy",
        "incidence_fixed_energy",
    ],
)
def test_observer_is_exactly_passive_including_interventions(
    arm, precision, implementation, monkeypatch
):
    from experiments.aggregation_comparison.gram import GramReadout

    calls = []
    original_apply = GramReadout.apply

    def traced(*args):
        calls.append(args[2].detach().clone())
        return original_apply(*args)

    monkeypatch.setattr(GramReadout, "apply", traced)
    net, graph, args, device = model_and_graph(arm, precision, implementation)
    net.eval()
    state = engine.base.state_sha256(net)
    with torch.no_grad(), engine.autocast(args, device):
        for intervention in (None, "energy_off", "cross_off", "diagonal_off"):
            net.energy_intervention = intervention
            net.diagnostic_collector = None
            before_rng = torch.cuda.get_rng_state(device)
            calls.clear()
            expected = net(graph)
            if implementation == "fused":
                assert len(calls) == 8
                for index, used_readout in enumerate(calls):
                    pairs = (
                        torch.arange(index + 1, device=device).expand(2, -1)
                        if net.diagonal_only
                        else torch.triu_indices(index + 1, index + 1, device=device)
                    )
                    diagonal = pairs[0] == pairs[1]
                    mask = (
                        torch.ones_like(diagonal)
                        if intervention is None
                        else diagonal
                        if intervention == "cross_off"
                        else ~diagonal
                        if intervention == "diagonal_off"
                        else torch.zeros_like(diagonal)
                    )
                    torch.testing.assert_close(
                        used_readout,
                        net.energy_readouts[index] * mask[None, :, None],
                        rtol=0,
                        atol=0,
                    )
            calls.clear()
            net.diagnostic_collector = MechanismCollector()
            actual = net(graph)
            assert len(calls) == (8 if implementation == "fused" else 0)
            # Exact logits are stronger than matching counts on this sample.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(torch.cuda.get_rng_state(device), before_rng)
            assert len(net.diagnostic_collector.layers) == 8
            assert engine.base.state_sha256(net) == state
        net.energy_intervention = None
        net.diagnostic_collector = None
