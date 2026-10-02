"""Phase-0 debug tests; these do not run learning or production experiments."""

from __future__ import annotations

import pytest
import torch

from research.wedge_propagation.operators import (
    build_wedges,
    dense_incidence,
    dense_R,
    dense_wedge,
    fixed_wedge_apply,
    fixed_wedge_fast_apply,
    incidence_apply,
    incidence_transpose,
    laplacian_apply,
    wedge_apply,
    wedge_transpose,
)


def edges_from_pairs(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()


GRAPH_CASES = [
    ("path", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ("cycle", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
    ("star", 6, [(0, node) for node in range(1, 6)]),
    ("clique", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
    ("irregular", 7, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (4, 5)]),
]


@pytest.mark.parametrize("name,n,pairs", GRAPH_CASES)
def test_debug_dense_incidence_and_wedge_energy(name, n, pairs):
    edges = edges_from_pairs(pairs)
    wedges = build_wedges(edges, n)
    b = dense_incidence(edges, n)
    a = dense_wedge(wedges, n)
    r = dense_R(edges, wedges, n)
    generator = torch.Generator().manual_seed(811)
    x = torch.randn(n, 16, dtype=torch.float64, generator=generator)
    flows = torch.randn(len(pairs), 16, dtype=torch.float64, generator=generator)
    q = torch.randn(wedges.shape[1], 16, dtype=torch.float64, generator=generator)
    torch.testing.assert_close(a, r @ b, rtol=0, atol=0)
    torch.testing.assert_close(incidence_apply(edges, x), b @ x, rtol=0, atol=0)
    torch.testing.assert_close(incidence_transpose(edges, flows, n), b.t() @ flows)
    torch.testing.assert_close(wedge_apply(wedges, x), a @ x)
    torch.testing.assert_close(wedge_transpose(wedges, q, n), a.t() @ q)
    torch.testing.assert_close(laplacian_apply(edges, x), b.t() @ b @ x)
    operator = a.t() @ a
    torch.testing.assert_close(operator.sum(1), torch.zeros(n, dtype=torch.float64))
    assert torch.linalg.eigvalsh(operator).min() >= -1e-10
    energy = (x * fixed_wedge_apply(wedges, x)).sum()
    torch.testing.assert_close(energy, wedge_apply(wedges, x).square().sum())
    degree = torch.bincount(edges.reshape(-1), minlength=n).to(torch.float64)
    correction = degree[edges[0]] + degree[edges[1]] - 4
    laplacian = b.t() @ b
    identity = laplacian @ laplacian + b.t() @ torch.diag(correction) @ b
    torch.testing.assert_close(operator, identity, rtol=0, atol=0)


def test_debug_triangle_has_all_three_unordered_wedges():
    edges = edges_from_pairs([(0, 1), (1, 2), (2, 0)])
    wedges = build_wedges(edges, 3)
    expected = torch.tensor([[1, 0, 0], [0, 1, 2], [2, 2, 1]], dtype=torch.long)
    assert torch.equal(wedges, expected)
    assert torch.equal(wedges[0] < wedges[2], torch.ones(3, dtype=torch.bool))
    assert wedges.shape[1] == 3


@pytest.mark.parametrize("name,n,pairs", GRAPH_CASES)
def test_debug_orientation_and_endpoint_reversal(name, n, pairs):
    edges = edges_from_pairs(pairs)
    flipped = edges.clone()
    flipped[:, ::2] = edges.flip(0)[:, ::2]
    wedges = build_wedges(edges, n)
    assert torch.equal(build_wedges(flipped, n), wedges)
    signs = torch.ones(edges.shape[1], dtype=torch.float64)
    signs[::2] = -1
    r = dense_R(edges, wedges, n)
    torch.testing.assert_close(dense_R(flipped, wedges, n), r * signs, rtol=0, atol=0)
    torch.testing.assert_close(
        dense_incidence(flipped, n), signs[:, None] * dense_incidence(edges, n), rtol=0, atol=0
    )
    reversed_wedges = wedges[[2, 1, 0]]
    torch.testing.assert_close(dense_wedge(reversed_wedges, n), dense_wedge(wedges, n))
    x = torch.randn(n, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(16))
    torch.testing.assert_close(fixed_wedge_apply(reversed_wedges, x), fixed_wedge_apply(wedges, x))
    torch.testing.assert_close(fixed_wedge_fast_apply(flipped, x), fixed_wedge_fast_apply(edges, x))


def test_debug_node_permutation_equivariance():
    n = 7
    edges = edges_from_pairs(GRAPH_CASES[-1][2])
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])  # old ID -> new ID
    wedges = build_wedges(edges, n)
    permuted_edges = permutation[edges]
    permuted_wedges = build_wedges(permuted_edges, n)
    generator = torch.Generator().manual_seed(111)
    x = torch.randn(n, 16, dtype=torch.float64, generator=generator)
    permuted_x = torch.empty_like(x)
    permuted_x[permutation] = x
    torch.testing.assert_close(
        fixed_wedge_apply(permuted_wedges, permuted_x)[permutation], fixed_wedge_apply(wedges, x)
    )
    torch.testing.assert_close(
        fixed_wedge_fast_apply(permuted_edges, permuted_x)[permutation],
        fixed_wedge_fast_apply(edges, x),
    )
    original_rows = {tuple(row) for row in permutation[wedges].t().tolist()}
    canonical_original = {(min(i, k), j, max(i, k)) for i, j, k in original_rows}
    assert canonical_original == {tuple(row) for row in permuted_wedges.t().tolist()}


@pytest.mark.parametrize("name,n,pairs", GRAPH_CASES)
def test_debug_float64_explicit_fast_output_and_input_gradient(name, n, pairs):
    edges = edges_from_pairs(pairs)
    wedges = build_wedges(edges, n)
    generator = torch.Generator().manual_seed(101)
    x = torch.randn(n, 16, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    probe = torch.randn(n, 16, 2, dtype=torch.float64, generator=generator)
    explicit = fixed_wedge_apply(wedges, x)
    fast = fixed_wedge_fast_apply(edges, x)
    dense = (dense_wedge(wedges, n).t() @ dense_wedge(wedges, n) @ x.flatten(1)).reshape_as(x)
    torch.testing.assert_close(explicit, fast, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(explicit, dense, rtol=1e-10, atol=1e-10)
    explicit_gradient = torch.autograd.grad((explicit * probe).sum(), x, retain_graph=True)[0]
    fast_gradient = torch.autograd.grad((fast * probe).sum(), x, retain_graph=True)[0]
    dense_gradient = torch.autograd.grad((dense * probe).sum(), x)[0]
    torch.testing.assert_close(explicit_gradient, fast_gradient, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(explicit_gradient, dense_gradient, rtol=1e-10, atol=1e-10)


def test_debug_disjoint_union_batches_every_graph_and_16_realizations():
    cases = GRAPH_CASES[:3]
    offset = 0
    edge_parts = []
    wedge_parts = []
    dense_parts = []
    for _, n, pairs in cases:
        edges = edges_from_pairs(pairs)
        wedges = build_wedges(edges, n)
        edge_parts.append(edges + offset)
        wedge_parts.append(wedges + offset)
        a = dense_wedge(wedges, n)
        dense_parts.append(a.t() @ a)
        offset += n
    edges = torch.cat(edge_parts, dim=1)
    wedges = torch.cat(wedge_parts, dim=1)
    x = torch.randn(offset, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(24))
    reference = torch.block_diag(*dense_parts) @ x
    torch.testing.assert_close(fixed_wedge_apply(wedges, x), reference)
    torch.testing.assert_close(fixed_wedge_fast_apply(edges, x), reference)
    assert build_wedges(edges, offset).shape[1] == wedges.shape[1]


@pytest.mark.parametrize("n,pairs", [(0, []), (4, []), (4, [(0, 1), (2, 3)])])
def test_debug_empty_operator_is_mathematical_zero_with_zero_gradient(n, pairs):
    edges = edges_from_pairs(pairs)
    wedges = build_wedges(edges, n)
    assert wedges.shape == (3, 0)
    assert dense_R(edges, wedges, n).shape == (0, len(pairs))
    assert dense_wedge(wedges, n).shape == (0, n)
    x = torch.randn(n, 16, dtype=torch.float64, requires_grad=True)
    explicit = fixed_wedge_apply(wedges, x)
    fast = fixed_wedge_fast_apply(edges, x)
    torch.testing.assert_close(explicit, torch.zeros_like(x), rtol=0, atol=0)
    torch.testing.assert_close(fast, torch.zeros_like(x), rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(explicit.sum(), x)[0], torch.zeros_like(x))
    torch.testing.assert_close(torch.autograd.grad(fast.sum(), x)[0], torch.zeros_like(x))


@pytest.mark.parametrize(
    "edges,n,message",
    [
        (torch.tensor([[0], [0]]), 2, "self loops"),
        (torch.tensor([[0, 1], [1, 0]]), 2, "duplicate"),
        (torch.tensor([[-1], [1]]), 2, "outside"),
        (torch.tensor([[0], [2]]), 2, "outside"),
        (torch.empty((3, 0), dtype=torch.long), 2, "shape"),
    ],
)
def test_debug_invalid_topology_raises_explicit_error(edges, n, message):
    with pytest.raises(ValueError, match=message):
        build_wedges(edges, n)


def test_debug_type_flow_and_mutated_index_validation():
    with pytest.raises(TypeError, match="torch.long"):
        build_wedges(torch.tensor([[0.0], [1.0]]), 2)
    with pytest.raises(ValueError, match="nonnegative"):
        dense_incidence(torch.empty((2, 0), dtype=torch.long), -1)
    edges = edges_from_pairs([(0, 1), (1, 2)])
    wedges = build_wedges(edges, 3)
    with pytest.raises(ValueError, match="shape"):
        incidence_transpose(edges, torch.zeros(3, 1), 3)
    with pytest.raises(ValueError, match="absent"):
        dense_R(edges, torch.tensor([[0], [2], [1]]), 3)
    with pytest.raises(ValueError, match="duplicate"):
        dense_wedge(torch.cat((wedges, wedges[[2, 1, 0]]), dim=1), 3)
    # The second call must invalidate the cached topology rather than reuse it.
    fixed_wedge_fast_apply(edges, torch.ones(3, 2))
    edges[1, 0] = 0
    with pytest.raises(ValueError, match="self loops"):
        fixed_wedge_fast_apply(edges, torch.ones(3, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; no CPU fallback")
def test_debug_cuda_full_tensor_output_and_gradient():
    n = 7
    edges = edges_from_pairs(GRAPH_CASES[-1][2]).to("cuda")
    wedges = build_wedges(edges, n).to("cuda")
    x = torch.randn(n, 16, dtype=torch.float64, device="cuda", requires_grad=True)
    explicit = fixed_wedge_apply(wedges, x)
    fast = fixed_wedge_fast_apply(edges, x)
    torch.testing.assert_close(explicit, fast, rtol=1e-10, atol=1e-10)
    explicit_gradient = torch.autograd.grad(explicit.square().sum(), x, retain_graph=True)[0]
    fast_gradient = torch.autograd.grad(fast.square().sum(), x)[0]
    torch.testing.assert_close(explicit_gradient, fast_gradient, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="same device"):
        fixed_wedge_apply(wedges, x.cpu())
