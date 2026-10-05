"""CUDA DEBUG oracles for the full-node frozen E/J definitions, not benchmarks."""

from __future__ import annotations

import pytest
import torch

from research.frozen_energy_trace.energy import build_topology, measure


@pytest.fixture(autouse=True)
def cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for feature-dependent frozen energy tests")


def graph():
    # Triangle, an attached path, and two degree-zero isolates.
    return 7, torch.tensor([[2, 3, 1, 2, 0], [3, 4, 0, 1, 2]], dtype=torch.long)


def dense_reference(n, physical_edges, h, epsilon=1e-12):
    """Independent tiny dense global incidence matrices on CUDA.

    Distinct J is enumerated as the off-diagonal physical-edge bilinear matrix,
    not computed by subtracting the implementation's J_node/J_shared outputs.
    """
    device = h.device
    edge = physical_edges.sort(0).values
    order = (edge[0] * n + edge[1]).argsort()
    edge = edge[:, order].to(device)
    e = edge.shape[1]
    incidence = h.new_zeros((e, n))
    index = torch.arange(e, device=device)
    incidence[index, edge[0]] = -1
    incidence[index, edge[1]] = 1
    adjacency = torch.zeros((n, n), dtype=torch.bool, device=device)
    adjacency[edge[0], edge[1]] = True
    adjacency[edge[1], edge[0]] = True
    closed = adjacency | torch.eye(n, device=device, dtype=torch.bool)
    global_degree = incidence.square().sum(0)
    topoood_h = h * (global_degree + 1).rsqrt()[:, None]
    q, divergences, local_incidence, energies, symmetric, augmented, topoood, centered, norms = (
        [] for _ in range(9)
    )
    for v in range(n):
        membership = closed[v]
        included = membership[edge[0]] & membership[edge[1]]
        b = incidence * included[:, None]
        degree = b.square().sum(0)
        local = h[membership]
        flow = b @ h
        q.append(flow)
        divergences.append(b.T @ flow)
        local_incidence.append(b)
        energies.append(torch.trace(h.T @ b.T @ b @ h))
        symmetric_h = h * torch.where(degree > 0, degree.clamp_min(1).rsqrt(), 0)[:, None]
        augmented_h = h * (degree + 1).rsqrt()[:, None]
        symmetric.append(torch.trace(symmetric_h.T @ b.T @ b @ symmetric_h))
        augmented.append(torch.trace(augmented_h.T @ b.T @ b @ augmented_h))
        topoood.append(torch.trace(topoood_h.T @ b.T @ b @ topoood_h))
        centered.append((local - local.mean(0)).square().sum())
        norms.append(local.square().sum())
    energy, sym, aug, top_energy, center, norm = (
        torch.stack(rows) for rows in (energies, symmetric, augmented, topoood, centered, norms)
    )
    pairs = torch.cat((edge, edge.flip(0)), 1)
    shared, node, distinct = [], [], []
    for v, u in pairs.T:
        common = (local_incidence[v].square().sum(1) > 0) & (local_incidence[u].square().sum(1) > 0)
        shared.append((q[v][common] * q[u][common]).sum())
        node.append(torch.trace(divergences[v].T @ divergences[u]))
        cross = local_incidence[v] @ local_incidence[u].T
        cross[index, index] = 0
        distinct.append(torch.trace(q[v].T @ cross @ q[u]))
    empty = h.new_empty(0)
    j_shared, j_node, j_distinct = (
        torch.stack(rows) if rows else empty for rows in (shared, node, distinct)
    )
    denominator = (energy[pairs[0]] * energy[pairs[1]]).sqrt()
    degree = incidence.square().sum(0)
    physical_q = incidence @ h
    global_e = physical_q.square().sum()
    global_sym_h = h * torch.where(degree > 0, degree.clamp_min(1).rsqrt(), 0)[:, None]
    global_aug_h = h * (degree + 1).rsqrt()[:, None]
    global_sym = (incidence @ global_sym_h).square().sum()
    global_aug = (incidence @ global_aug_h).square().sum()
    global_norm = h.square().sum()
    global_center = (h - h.mean(0)).square().sum()
    return {
        "E": energy,
        "E_sym": sym,
        "E_sym_augmented": aug,
        "E_topoood_1hop": top_energy,
        "E_scale_free": energy / (center + epsilon),
        "E_sym_scale_free": sym / (norm + epsilon),
        "E_sym_augmented_scale_free": aug / (norm + epsilon),
        "E_topoood_1hop_scale_free": top_energy / (norm + epsilon),
        "centered_norm_sq": center,
        "local_feature_norm_sq": norm,
        "zero_E_denominator": center == 0,
        "zero_sym_denominator": norm == 0,
        "zero_J_denominator": denominator == 0,
        "J_shared": j_shared,
        "J_node": j_node,
        "J_distinct": j_distinct,
        "J_shared_normalized": j_shared / (denominator + epsilon),
        "J_node_normalized": j_node / (denominator + epsilon),
        "J_distinct_normalized": j_distinct / (denominator + epsilon),
        "global_E": global_e,
        "global_E_per_node": global_e / n,
        "global_E_sym": global_sym,
        "global_E_sym_augmented": global_aug,
        "global_feature_norm_sq": global_norm,
        "global_centered_norm_sq": global_center,
        "global_E_scale_free": global_e / (global_center + epsilon),
        "global_E_rayleigh": global_e / (global_norm + epsilon),
        "global_E_sym_scale_free": global_sym / (global_norm + epsilon),
        "global_E_sym_augmented_scale_free": global_aug / (global_norm + epsilon),
        "zero_global_norm_denominator": global_norm == 0,
    }


def assert_fields(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-11, atol=2e-11, msg=key)


def test_induced_triangle_path_isolates_and_direct_distinct_identity():
    n, edges = graph()
    top = build_topology(n, edges, workers=2).to("cuda")
    h = torch.randn(
        (n, 11), generator=torch.Generator().manual_seed(1103), dtype=torch.float64
    ).cuda()
    actual = measure(top, h, feature_chunk=3)
    assert_fields(actual, dense_reference(n, edges, h))
    torch.testing.assert_close(actual["J_node"], 2 * actual["J_shared"] + actual["J_distinct"])
    assert top.num_pairs == 2 * edges.shape[1]
    assert top.num_local_nodes == n + 2 * edges.shape[1]
    assert (
        top.num_local_edges == 2 * edges.shape[1] + 3
    )  # One triangle adds its three opposite edges.
    assert bool((actual["E"][5:] == 0).all())
    assert bool((actual["E_sym"][5:] == 0).all())
    assert bool((actual["E_sym_augmented"][5:] == 0).all())
    assert bool(actual["zero_E_denominator"][5:].all())
    assert bool((actual["J_shared"] >= 0).all())
    assert not torch.allclose(actual["E_sym_augmented"], actual["E_topoood_1hop"])
    # The signed distinct relation is not a nonnegative energy.
    path = build_topology(3, torch.tensor([[0, 1], [1, 2]]), workers=2).to("cuda")
    path_h = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64, device="cuda")
    signed = measure(path, path_h, feature_chunk=1)
    assert_fields(signed, dense_reference(3, path.edges.cpu(), path_h))
    assert bool((signed["J_distinct"] < 0).all())


def test_packed_independent_stages_and_feature_chunks_preserve_all_values():
    n, edges = graph()
    top = build_topology(n, edges, workers=2).to("cuda")
    values = torch.randn(
        (3, n, 19), generator=torch.Generator().manual_seed(1117), dtype=torch.float64
    ).cuda()
    packed = measure(top, values, feature_chunk=7)
    for index in range(values.shape[0]):
        separate = measure(top, values[index], feature_chunk=19)
        assert_fields({key: value[index] for key, value in packed.items()}, separate)
        assert_fields(separate, dense_reference(n, edges, values[index]))
    assert_fields(packed, measure(top, values, feature_chunk=1))


def test_translation_scale_and_orientation_invariance_with_named_degree_controls():
    n, edges = graph()
    top = build_topology(n, edges, workers=2).to("cuda")
    reversed_top = build_topology(n, edges.flip(0).flip(1), workers=1).to("cuda")
    h = torch.randn(
        (n, 8), generator=torch.Generator().manual_seed(1201), dtype=torch.float64
    ).cuda()
    original = measure(top, h, feature_chunk=4, epsilon=1e-30)
    assert_fields(original, measure(reversed_top, h, feature_chunk=2, epsilon=1e-30))
    translated = measure(
        top, h + torch.arange(8, device="cuda", dtype=torch.float64), feature_chunk=4, epsilon=1e-30
    )
    for key in ("E", "E_scale_free", "centered_norm_sq", "J_shared", "J_node", "J_distinct"):
        torch.testing.assert_close(original[key], translated[key], rtol=2e-12, atol=2e-12)
    # L_sym is degree normalization; it is not amplitude or translation invariance.
    assert not torch.allclose(original["E_sym"], translated["E_sym"])
    scaled = measure(top, -3.25 * h, feature_chunk=3, epsilon=1e-30)
    for key in (
        "E",
        "E_sym",
        "E_sym_augmented",
        "E_topoood_1hop",
        "J_shared",
        "J_node",
        "J_distinct",
        "global_E",
        "global_E_per_node",
    ):
        torch.testing.assert_close(scaled[key], original[key] * 3.25**2, rtol=2e-12, atol=2e-12)
    for key in (
        "E_scale_free",
        "E_sym_scale_free",
        "E_sym_augmented_scale_free",
        "E_topoood_1hop_scale_free",
        "J_shared_normalized",
        "J_node_normalized",
        "J_distinct_normalized",
    ):
        torch.testing.assert_close(scaled[key], original[key], rtol=2e-12, atol=2e-12)


def test_zero_graph_zero_feature_flags_and_frozen_no_grad():
    n, edges = graph()
    top = build_topology(n, edges, workers=2).to("cuda")
    zero = torch.zeros((n, 5), dtype=torch.float64, device="cuda", requires_grad=True)
    actual = measure(top, zero, feature_chunk=2)
    assert_fields(actual, dense_reference(n, edges, zero.detach()))
    assert bool(actual["zero_E_denominator"].all())
    assert bool(actual["zero_sym_denominator"].all())
    assert bool(actual["zero_J_denominator"].all())
    assert all(not value.requires_grad for value in actual.values())
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    empty_top = build_topology(3, empty_edges, workers=2).to("cuda")
    h = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 7.0]], dtype=torch.float64, device="cuda")
    empty = measure(empty_top, h, feature_chunk=1)
    assert_fields(empty, dense_reference(3, empty_edges, h))
    assert empty["J_node"].numel() == 0
    assert empty["J_distinct_normalized"].numel() == 0


def test_symmetric_null_space_and_normalized_J_is_not_cosine_bounded():
    n, edges = graph()
    top = build_topology(n, edges, workers=2).to("cuda")
    h = top.physical_degree.double().sqrt()[:, None]
    result = measure(top, h, feature_chunk=1)
    torch.testing.assert_close(
        result["global_E_sym"], result["global_E_sym"].new_zeros(()), atol=1e-13, rtol=0
    )
    assert result["global_E"] > 0
    # Complete locals are identical; node J is ||Lh||²/energy, which can exceed 1.
    complete_edges = torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long)
    complete = build_topology(4, complete_edges, workers=2).to("cuda")
    field = torch.tensor([[1.0], [-1.0], [0.0], [0.0]], dtype=torch.float64, device="cuda")
    relation = measure(complete, field, feature_chunk=1)
    torch.testing.assert_close(
        relation["J_node_normalized"], field.new_full((12,), 4.0), atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(
        relation["J_distinct_normalized"], field.new_full((12,), 2.0), atol=1e-11, rtol=1e-11
    )


def test_feature_arithmetic_rejects_cpu_and_invalid_static_topology():
    n, edges = graph()
    top = build_topology(n, edges, workers=2)
    with pytest.raises(ValueError, match="CUDA"):
        measure(top, torch.ones(n, 3), feature_chunk=2)
    with pytest.raises(ValueError, match="duplicate"):
        build_topology(n, torch.cat((edges, edges[:, :1]), 1), workers=2)
    with pytest.raises(ValueError, match="workers"):
        build_topology(n, edges, workers=0)
    with pytest.raises(ValueError, match="distinct endpoints"):
        build_topology(n, torch.tensor([[0], [0]]), workers=2)
