"""Explicit CUDA debug regressions for the independent review's counterexamples."""

import copy

import pytest
import torch

from experiments.aggregation_comparison.model import AggregationClassifier
from tests.test_aggregation_comparison_cuda import (  # noqa: F401
    cuda_required,
    synthetic_disjoint_batch,
)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_energy_clone_local_state_gradients_and_checkpoint(checkpoint):
    torch.manual_seed(773)
    _, graph, _ = synthetic_disjoint_batch(torch.device("cuda:0"))

    def make():
        return (
            AggregationClassifier(
                50,
                7,
                arm="incidence_energy_pre_lift",
                selection_config={"condition": "full"},
                hidden_channels=256,
                layers=8,
                heads=8,
                dropout=0,
                edge_chunk_size=4096,
                activation_checkpoint=checkpoint,
            )
            .cuda()
            .eval()
        )

    original = make()
    with torch.no_grad():
        for value in original.energy_readouts:
            value.normal_(0, 0.001)
    first_clone = copy.deepcopy(original)
    with torch.no_grad():
        torch.testing.assert_close(first_clone(graph), original(graph))
    clone, fresh = copy.deepcopy(original), make()
    fresh.load_state_dict(original.state_dict())
    previous = [op.last_effective_c.clone() for op in original.layers]
    next_graph = copy.copy(graph)
    next_graph.x = graph.x + torch.randn_like(graph.x) * 0.2
    actual, expected = clone(next_graph), fresh(next_graph)
    direction = torch.randn_like(actual)
    left = torch.autograd.grad(actual, tuple(clone.parameters()), direction)
    right = torch.autograd.grad(expected, tuple(fresh.parameters()), direction)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    for before, op in zip(previous, original.layers, strict=True):
        torch.testing.assert_close(before, op.last_effective_c, rtol=0, atol=0)
        assert not op.estimator._forward_hooks
        assert not hasattr(op, "live_comparison_c")
