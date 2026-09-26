"""Synthetic CUDA dense differential and allocation probes for the exact Gram."""

import json

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from experiments.aggregation_comparison.model import local_gram
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401
from tests.test_aggregation_review_cuda import old_gram


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
def test_gram_dense_gradcheck(shared, diagonal):
    torch.manual_seed(75)
    h = torch.randn(3, 7, 2, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    c = torch.rand(9, 1 if shared else 2, device="cuda", dtype=torch.float64, requires_grad=True)
    edges = torch.randint(7, (2, 9), device="cuda")
    incidence = h.new_zeros(9, 7)
    incidence.index_put_(
        (torch.arange(9, device="cuda"), edges[0]),
        -torch.ones(9, device="cuda", dtype=h.dtype),
        accumulate=True,
    )
    incidence.index_put_(
        (torch.arange(9, device="cuda"), edges[1]),
        torch.ones(9, device="cuda", dtype=h.dtype),
        accumulate=True,
    )
    delta = torch.einsum("en,knhd->kehd", incidence, h)
    pairs = (
        torch.arange(3, device="cuda").expand(2, -1)
        if diagonal
        else torch.triu_indices(3, 3, device="cuda")
    )
    products = (delta[pairs[0]] * delta[pairs[1]]).sum(-1).permute(1, 2, 0) * c[..., None] / 2
    # Endpoint membership must count a self-loop twice (its difference is zero).
    endpoint = torch.nn.functional.one_hot(edges[0], 7) + torch.nn.functional.one_hot(edges[1], 7)
    reference = torch.einsum("en,ehp->nhp", endpoint.to(h.dtype), products)
    actual = local_gram(h, edges, c, 6, diagonal_only=diagonal)
    v = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (h, c), v, retain_graph=True)
    gb = torch.autograd.grad(reference, (h, c), v)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(
        lambda a, b: local_gram(a, edges, b, 6, diagonal_only=diagonal),
        (h, c),
        fast_mode=True,
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.parametrize("chunk", [256, 4096])
def test_backward_allocates_history_once(chunk, record_property):
    torch.manual_seed(92)
    h = torch.randn(4, 4000, 4, 8, device="cuda", requires_grad=True)
    c = torch.rand(16000, 4, device="cuda", requires_grad=True)
    edges = torch.randint(4000, (2, 16000), device="cuda")
    evidence = {}
    for name, function in (("old", old_gram), ("new", local_gram)):
        value = function(h, edges, c, chunk)
        upstream = torch.ones_like(value)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
        ) as trace:
            gradients = torch.autograd.grad(value, (h, c), upstream)
            torch.cuda.synchronize()
        full = [
            e
            for e in trace.events()
            if e.name == "aten::zeros_like" and e.input_shapes[0] == list(h.shape)
        ]
        indexing = [e for e in trace.events() if e.name == "IndexBackward0"]
        history_bytes = h.numel() * h.element_size()
        # History gather's backward uses a K x chunk x H x d cotangent,
        # unlike the pairs x chunk x H x d cotangent of pair indexing.
        # Allocator rounding can make device bytes exceed logical tensor bytes.
        full_old = [
            e
            for e in trace.events()
            if e.name == "aten::new_zeros"
            and len(e.input_shapes[0]) == 4
            and e.input_shapes[0][0] == h.shape[0]
            and e.input_shapes[0][2:] == list(h.shape[2:])
            and e.device_memory_usage >= history_bytes
        ]
        evidence[name] = {
            "full_history_zero_allocations": len(full),
            "index_backward_calls": len(indexing),
            "history_bytes": h.numel() * h.element_size(),
            "full_history_index_gradient_allocations": len(full_old),
            "full_history_index_gradient_cumulative_bytes": len(full_old) * history_bytes,
            "full_history_index_gradient_allocated_bytes": sum(
                e.device_memory_usage for e in full_old
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        if name == "old":
            old_gradients = gradients
        else:
            for actual, expected in zip(gradients, old_gradients, strict=True):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert evidence["new"]["full_history_zero_allocations"] == 1
    assert evidence["new"]["index_backward_calls"] == 0
    assert evidence["old"]["full_history_index_gradient_allocations"] == 2 * (
        (16000 + chunk // 4 - 1) // (chunk // 4)
    )
    assert evidence["new"]["full_history_index_gradient_allocations"] == 0
    assert evidence["old"]["index_backward_calls"] >= 2 * ((16000 + chunk // 4 - 1) // (chunk // 4))
    record_property("backward_allocation_probe", json.dumps(evidence))
