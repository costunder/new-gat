"""DEBUG dense references for complete induced locals and sparse fixed operators.

These small diagnostic graphs verify algebra and implementation. They do not
replace the full synthetic/citation audit and contain no classifier training.
"""

from __future__ import annotations

import pickle
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from research.local_energy_relations.operators import (
    aggregate_squares,
    incidence,
    incidence_transpose,
    local_weight,
    recover_flows,
    relation_terms,
    solve_local_laplacian,
    transfer_bookkeeping,
)
from research.local_energy_relations.topology import (
    LocalTopology,
    batch_topologies,
    build_topology,
)


def debug_topology() -> LocalTopology:
    # An unequal-degree triangle, branching tail and isolated node.
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5)]
    return build_topology(7, np.asarray(pairs, dtype=np.int64).T)


def dense_local(top: LocalTopology, center: int) -> tuple[torch.Tensor, torch.Tensor]:
    ns, ne = int(top.node_offsets[center]), int(top.node_offsets[center + 1])
    es, ee = int(top.edge_offsets[center]), int(top.edge_offsets[center + 1])
    matrix = torch.zeros((ee - es, ne - ns), dtype=torch.float64)
    if ee > es:
        rows = torch.arange(ee - es)
        endpoints = torch.as_tensor(top.local_edge_nodes[:, es:ee]) - ns
        matrix[rows, endpoints[0]] = -1
        matrix[rows, endpoints[1]] = 1
    embedded = torch.zeros((ee - es, top.n), dtype=torch.float64)
    embedded[:, torch.as_tensor(top.local_node_global[ns:ne])] = matrix
    return matrix, embedded


def debug_fields(top: LocalTopology, feature_count: int = 4):
    generator = torch.Generator().manual_seed(20261003)
    left = torch.randn((top.n, feature_count), generator=generator, dtype=torch.float64)
    right = torch.randn((top.n, feature_count), generator=generator, dtype=torch.float64)
    tensor_top = top.to("cpu")
    return tensor_top, left, right


def assert_close(left, right, atol=2e-9, rtol=2e-9):
    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_topology_induced_graphs_and_overlap_correspondence():
    top = debug_topology()
    assert top.num_centers == top.num_nodes == 7
    assert top.num_pairs == 2 * top.num_edges == 12
    assert top.num_graphs == 1
    assert top.max_local_nodes == 4
    assert top.local_node_global[top.center_positions].tolist() == list(range(7))
    for center in range(top.n):
        neighbors = set(top.edges[1, top.edges[0] == center])
        neighbors.update(top.edges[0, top.edges[1] == center])
        expected_nodes = sorted(neighbors | {center})
        ns, ne = top.node_offsets[center : center + 2]
        es, ee = top.edge_offsets[center : center + 2]
        assert top.local_node_global[ns:ne].tolist() == expected_nodes
        expected_edges = [
            e
            for e, (a, b) in enumerate(top.edges.T)
            if a in neighbors | {center} and b in neighbors | {center}
        ]
        assert top.local_edge_global[es:ee].tolist() == expected_edges
        assert np.all(top.local_node_center[ns:ne] == center)
        assert np.all(top.local_edge_center[es:ee] == center)
        local_degree = np.bincount(top.local_edge_nodes[:, es:ee].ravel() - ns, minlength=ne - ns)
        np.testing.assert_array_equal(top.local_degree[ns:ne], local_degree)
    for kind in ("shared_node", "shared_edge"):
        pair = getattr(top, kind + "_pair")
        left = getattr(top, kind + "_left")
        right = getattr(top, kind + "_right")
        global_ids = top.local_node_global if kind == "shared_node" else top.local_edge_global
        centers = top.local_node_center if kind == "shared_node" else top.local_edge_center
        np.testing.assert_array_equal(global_ids[left], global_ids[right])
        np.testing.assert_array_equal(centers[left], top.pair_centers[0, pair])
        np.testing.assert_array_equal(centers[right], top.pair_centers[1, pair])
    assert np.any(top.omitted_node_pair >= 0)
    assert np.any(top.boundary_edge_pair >= 0)
    assert np.all(np.diff(top.pair_centers[0]) >= -top.n)  # Complete directed order retained.
    # Static preparation remains process-pickle compatible for CPU workers.
    restored = pickle.loads(pickle.dumps(top))
    for field in fields(top):
        if isinstance(getattr(top, field.name), np.ndarray):
            np.testing.assert_array_equal(getattr(top, field.name), getattr(restored, field.name))
        else:
            assert getattr(top, field.name) == getattr(restored, field.name)


@pytest.mark.parametrize(
    "n,edges",
    [
        (0, np.empty((2, 0), dtype=np.int64)),
        (True, np.empty((2, 0), dtype=np.int64)),
        (3, np.asarray([[0], [0]])),
        (3, np.asarray([[0, 1], [1, 0]])),
        (3, np.asarray([[0], [3]])),
        (3, np.asarray([[-1], [1]])),
        (3, np.asarray([[0.0], [1.0]])),
        (3, np.asarray([0, 1])),
    ],
)
def test_invalid_physical_graph_is_explicit(n, edges):
    with pytest.raises(ValueError):
        build_topology(n, edges)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_incidence_energy_and_rectangular_relations_match_dense(mode):
    top = debug_topology()
    tensor_top, h_left, h_right = debug_fields(top)
    c = local_weight(tensor_top, mode)
    g_left = incidence(tensor_top, h_left[tensor_top.local_node_global])
    g_right = incidence(tensor_top, h_right[tensor_top.local_node_global])
    q_left, q_right = c[:, None] * g_left, c[:, None] * g_right
    d = incidence_transpose(tensor_top, q_left)
    energy = aggregate_squares(tensor_top, g_left, weights=c)
    flow_norm = aggregate_squares(tensor_top, q_left)
    relations = relation_terms(tensor_top, q_left, q_right)
    matrices = [dense_local(top, v) for v in range(top.n)]
    for center, (b, embedded) in enumerate(matrices):
        ns, ne = top.node_offsets[center : center + 2]
        es, ee = top.edge_offsets[center : center + 2]
        assert_close(g_left[es:ee], embedded @ h_left)
        assert_close(d[ns:ne], b.T @ q_left[es:ee])
        assert_close(
            energy[center], ((embedded @ h_left) * (c[es:ee, None] * (embedded @ h_left))).sum(0)
        )
        assert_close(flow_norm[center], q_left[es:ee].square().sum(0))
        laplacian = b.T @ torch.diag(c[es:ee]) @ b
        local_h = h_left[top.local_node_global[ns:ne]]
        assert_close(energy[center], (local_h * (laplacian @ local_h)).sum(0))
    for pair, (left, right) in enumerate(top.pair_centers.T):
        ls, le = top.edge_offsets[left : left + 2]
        rs, re = top.edge_offsets[right : right + 2]
        bv, bu = matrices[left][1], matrices[right][1]
        shared = torch.as_tensor(
            top.local_edge_global[ls:le, None] == top.local_edge_global[None, rs:re],
            dtype=torch.float64,
        )
        gram = bv @ bu.T
        qv, qu = q_left[ls:le], q_right[rs:re]
        assert_close(relations["shared_edges"][pair], (qv * (shared @ qu)).sum(0))
        assert_close(relations["shared_nodes"][pair], (qv * (gram @ qu)).sum(0))
        assert_close(relations["distinct_edges"][pair], (qv * ((gram - 2 * shared) @ qu)).sum(0))
        # The explicit rectangular W includes each local conductance exactly once.
        w = torch.diag(c[ls:le]) @ (gram - 2 * shared) @ torch.diag(c[rs:re])
        assert_close(
            relations["distinct_edges"][pair], (g_left[ls:le] * (w @ g_right[rs:re])).sum(0)
        )
    if mode == "local_degree":
        assert not torch.allclose(energy, flow_norm)
    assert relations["distinct_edges"].abs().max() > 0


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_projected_cg_and_full_divergence_recovery_match_dense_pseudoinverse(mode):
    top = debug_topology()
    tensor_top, h, _ = debug_fields(top)
    c = local_weight(tensor_top, mode)
    g = incidence(tensor_top, h[tensor_top.local_node_global])
    q = c[:, None] * g
    d = incidence_transpose(tensor_top, q)
    x, info = solve_local_laplacian(tensor_top, d, c, iteration_multiplier=10)
    recovered = recover_flows(tensor_top, q, c, iteration_multiplier=10)
    for center in range(top.n):
        ns, ne = top.node_offsets[center : center + 2]
        es, ee = top.edge_offsets[center : center + 2]
        b, _ = dense_local(top, center)
        reference = torch.linalg.pinv(b.T @ torch.diag(c[es:ee]) @ b, hermitian=True) @ d[ns:ne]
        assert_close(x[ns:ne], reference)
        cut = b @ torch.linalg.pinv(b.T @ b, hermitian=True) @ d[ns:ne]
        assert_close(recovered["q_cut"][es:ee], cut)
        assert_close(recovered["q_cycle"][es:ee], q[es:ee] - cut)
        assert_close(x[ns:ne].mean(0), torch.zeros(h.shape[1], dtype=h.dtype))
    assert_close(recovered["q_recon"], q)
    assert_close(incidence_transpose(tensor_top, recovered["q_cycle"]), torch.zeros_like(d))
    assert_close(
        aggregate_squares(tensor_top, recovered["q_cycle"]),
        aggregate_squares(tensor_top, q) - aggregate_squares(tensor_top, recovered["q_cut"]),
    )
    orthogonality = torch.zeros_like(aggregate_squares(tensor_top, q)).index_add(
        0, tensor_top.local_edge_center, recovered["q_cut"] * recovered["q_cycle"]
    )
    assert_close(orthogonality, torch.zeros_like(orthogonality))
    assert info["converged"].all()
    assert info["relative_residual"].max() <= 1e-9
    assert info["max_iterations"] == 10 * top.max_local_nodes
    assert torch.all(info["iterations"][6] == 0)  # Isolate zero RHS.
    if mode == "unit":
        assert_close(recovered["q_cycle"], torch.zeros_like(q))
    else:
        # This Euclidean cycle component is not an irreversible loss with known C.
        assert recovered["q_cycle"].norm() > 1e-3


def test_arbitrary_edge_flow_cycle_is_invisible_but_not_claimed_reconstructible():
    top = build_topology(3, np.asarray([[0, 0, 1], [1, 2, 2]], dtype=np.int64)).to("cpu")
    cycle = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64).repeat(3)[:, None]
    zero = incidence_transpose(top, cycle)
    assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
    results = recover_flows(top, cycle, local_weight(top, "unit"))
    assert_close(results["q_cut"], torch.zeros_like(cycle), atol=0, rtol=0)
    assert_close(results["q_cycle"], cycle, atol=0, rtol=0)
    assert_close(results["q_recon"], torch.zeros_like(cycle), atol=0, rtol=0)
    assert results["cut_solver"]["global_iterations"] == 0


@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_sparse_pair_chunking_preserves_every_relation_and_transfer(chunk):
    top = debug_topology()
    tensor_top, h, other = debug_fields(top)
    c = local_weight(tensor_top, "local_degree")
    q = c[:, None] * incidence(tensor_top, h[tensor_top.local_node_global])
    p = c[:, None] * incidence(tensor_top, other[tensor_top.local_node_global])
    for actual, expected in (
        (relation_terms(tensor_top, q, p, relation_batch=chunk), relation_terms(tensor_top, q, p)),
        (
            transfer_bookkeeping(tensor_top, q, relation_batch=chunk),
            transfer_bookkeeping(tensor_top, q),
        ),
    ):
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_partial_receiver_observation_is_exact_partition_not_inverse():
    top = debug_topology()
    tensor_top, h, _ = debug_fields(top)
    q = local_weight(tensor_top, "local_degree")[:, None] * incidence(
        tensor_top, h[tensor_top.local_node_global]
    )
    d = incidence_transpose(tensor_top, q)
    audit = transfer_bookkeeping(tensor_top, q)
    assert_close(
        audit["sender_divergence_norm_sq"],
        audit["retained_divergence_norm_sq"] + audit["omitted_divergence_norm_sq"],
    )
    assert_close(
        audit["sender_flow_norm_sq"],
        audit["retained_flow_norm_sq"]
        + audit["boundary_flow_norm_sq"]
        + audit["omitted_flow_norm_sq"],
    )
    assert torch.equal(
        audit["sender_node_count"], audit["retained_node_count"] + audit["omitted_node_count"]
    )
    assert torch.equal(
        audit["sender_edge_count"],
        audit["retained_edge_count"] + audit["boundary_edge_count"] + audit["omitted_edge_count"],
    )
    for pair, (sender, receiver) in enumerate(top.pair_centers.T):
        ns, ne = top.node_offsets[sender : sender + 2]
        es, ee = top.edge_offsets[sender : sender + 2]
        rns, rne = top.node_offsets[receiver : receiver + 2]
        receiver_nodes = set(top.local_node_global[rns:rne])
        retained = torch.tensor([node in receiver_nodes for node in top.local_node_global[ns:ne]])
        assert_close(audit["retained_divergence_norm_sq"][pair], d[ns:ne][retained].square().sum(0))
        assert_close(audit["omitted_divergence_norm_sq"][pair], d[ns:ne][~retained].square().sum(0))
        status = torch.tensor(
            [
                sum(int(node in receiver_nodes) for node in top.edges[:, e])
                for e in top.local_edge_global[es:ee]
            ]
        )
        for name, count in (("retained", 2), ("boundary", 1), ("omitted", 0)):
            assert_close(
                audit[name + "_flow_norm_sq"][pair], q[es:ee][status == count].square().sum(0)
            )
    assert audit["omitted_divergence_norm_sq"].sum() > 0
    assert audit["boundary_flow_norm_sq"].sum() > 0


def test_batch_complete_graphs_matches_independent_solves_and_nested_batch():
    first = debug_topology()
    second = build_topology(4, np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64))
    empty = build_topology(2, np.empty((2, 0), dtype=np.int64))
    packed = batch_topologies([first, second, empty])
    nested = batch_topologies([batch_topologies([first, second]), empty])
    assert packed.num_graphs == 3
    assert packed.graph_node_offsets.tolist() == [0, 7, 11, 13]
    assert packed.graph_edge_offsets.tolist() == [0, 6, 9, 9]
    for field in fields(packed):
        left, right = getattr(packed, field.name), getattr(nested, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    tensor_top, h, _ = debug_fields(packed)
    c = local_weight(tensor_top, "local_degree")
    q = c[:, None] * incidence(tensor_top, h[tensor_top.local_node_global])
    recovered = recover_flows(tensor_top, q, c, iteration_multiplier=10)
    relations = relation_terms(tensor_top, q, relation_batch=2)
    transfer = transfer_bookkeeping(tensor_top, q, relation_batch=2)
    node_offset = edge_offset = pair_offset = local_edge_offset = 0
    for individual in (first, second, empty):
        own = individual.to("cpu")
        own_h = h[node_offset : node_offset + own.n]
        own_c = local_weight(own, "local_degree")
        own_q = own_c[:, None] * incidence(own, own_h[own.local_node_global])
        own_results = recover_flows(own, own_q, own_c, iteration_multiplier=10)
        for name in ("q_cut", "q_cycle", "q_recon"):
            assert_close(
                recovered[name][local_edge_offset : local_edge_offset + own.num_local_edges],
                own_results[name],
            )
        for actual, expected in (
            (relations, relation_terms(own, own_q)),
            (transfer, transfer_bookkeeping(own, own_q)),
        ):
            for name, value in expected.items():
                assert_close(actual[name][pair_offset : pair_offset + own.num_pairs], value)
        assert np.all(
            packed.pair_centers[:, pair_offset : pair_offset + own.num_pairs] >= node_offset
        )
        assert np.all(
            packed.pair_centers[:, pair_offset : pair_offset + own.num_pairs] < node_offset + own.n
        )
        node_offset += own.n
        edge_offset += own.num_edges
        pair_offset += own.num_pairs
        local_edge_offset += own.num_local_edges
    assert edge_offset == packed.num_edges


def test_input_edge_order_and_directions_are_canonical():
    top = debug_topology()
    order = np.asarray([3, 0, 5, 1, 4, 2])
    altered = build_topology(top.n, top.edges[::-1, order])
    for field in fields(top):
        left, right = getattr(top, field.name), getattr(altered, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_consistent_signed_edge_gauge_preserves_energy_and_relations():
    top = debug_topology()
    tensor_top, h, other = debug_fields(top)
    flip_global = np.asarray([True, False, True, False, False, True])
    flip_occurrence = flip_global[top.local_edge_global]
    endpoints = top.local_edge_nodes.copy()
    endpoints[:, flip_occurrence] = endpoints[::-1, flip_occurrence]
    physical_edges = top.edges.copy()
    physical_edges[:, flip_global] = physical_edges[::-1, flip_global]
    gauged = replace(top, local_edge_nodes=endpoints, edges=physical_edges).to("cpu")
    c = local_weight(tensor_top, "local_degree")
    original_g = incidence(tensor_top, h[tensor_top.local_node_global])
    new_g = incidence(gauged, h[gauged.local_node_global])
    signs = torch.as_tensor(np.where(flip_occurrence, -1.0, 1.0))[:, None]
    assert_close(new_g, signs * original_g, atol=0, rtol=0)
    original_q, new_q = c[:, None] * original_g, c[:, None] * new_g
    assert_close(incidence_transpose(tensor_top, original_q), incidence_transpose(gauged, new_q))
    assert_close(
        aggregate_squares(tensor_top, original_g, weights=c),
        aggregate_squares(gauged, new_g, weights=c),
    )
    original_right = c[:, None] * incidence(tensor_top, other[tensor_top.local_node_global])
    new_right = c[:, None] * incidence(gauged, other[gauged.local_node_global])
    for name, value in relation_terms(tensor_top, original_q, original_right).items():
        assert_close(value, relation_terms(gauged, new_q, new_right)[name])
    original_recovery = recover_flows(tensor_top, original_q, c)
    new_recovery = recover_flows(gauged, new_q, c)
    for name in ("q_cut", "q_cycle", "q_recon"):
        assert_close(new_recovery[name], signs * original_recovery[name])


def test_node_relabeling_preserves_center_and_pair_statistics():
    top = debug_topology()
    tensor_top, h, other = debug_fields(top)
    permutation = np.asarray([4, 0, 5, 2, 6, 1, 3])  # old node -> new node.
    inverse = np.argsort(permutation)
    renamed = build_topology(top.n, permutation[top.edges]).to("cpu")
    h_new, other_new = h[inverse], other[inverse]
    old_c, new_c = local_weight(tensor_top, "local_degree"), local_weight(renamed, "local_degree")
    old_g, new_g = (
        incidence(tensor_top, h[tensor_top.local_node_global]),
        incidence(renamed, h_new[renamed.local_node_global]),
    )
    old_q, new_q = old_c[:, None] * old_g, new_c[:, None] * new_g
    assert_close(
        aggregate_squares(tensor_top, old_g, weights=old_c),
        aggregate_squares(renamed, new_g, weights=new_c)[permutation],
    )
    old_right = old_c[:, None] * incidence(tensor_top, other[tensor_top.local_node_global])
    new_right = new_c[:, None] * incidence(renamed, other_new[renamed.local_node_global])
    pairs = {tuple(pair): i for i, pair in enumerate(renamed.pair_centers.T.tolist())}
    rows = torch.tensor([pairs[tuple(permutation[pair])] for pair in top.pair_centers.T])
    for name, value in relation_terms(tensor_top, old_q, old_right).items():
        assert_close(value, relation_terms(renamed, new_q, new_right)[name][rows])


def test_empty_edges_zero_rhs_and_constant_features_are_exact_and_finite():
    top = build_topology(5, np.empty((2, 0), dtype=np.int64)).to("cpu")
    h = torch.full((top.num_local_nodes, 3), 7.0, dtype=torch.float64)
    q = incidence(top, h)
    assert q.shape == (0, 3)
    c = local_weight(top, "local_degree")
    recovered = recover_flows(top, q, c)
    assert recovered["q_recon"].shape == (0, 3)
    for info in (recovered["cut_solver"], recovered["reconstruction_solver"]):
        assert info["global_iterations"] == 0
        assert info["converged"].all()
        assert torch.count_nonzero(info["relative_residual"]) == 0
    assert aggregate_squares(top, q).shape == (5, 3)
    for value in relation_terms(top, q).values():
        assert value.shape == (0, 3)
    for value in transfer_bookkeeping(top, q).values():
        assert value.shape[0] == 0
    nonempty = debug_topology().to("cpu")
    constant = incidence(nonempty, torch.ones((nonempty.num_local_nodes, 2), dtype=torch.float64))
    assert torch.count_nonzero(constant) == 0
    assert (
        recover_flows(nonempty, constant, local_weight(nonempty, "unit"))["cut_solver"][
            "global_iterations"
        ]
        == 0
    )


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_relation_batch_is_explicit(bad):
    top = debug_topology().to("cpu")
    q = torch.zeros((top.num_local_edges, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="relation_batch"):
        relation_terms(top, q, relation_batch=bad)
    with pytest.raises(ValueError, match="relation_batch"):
        transfer_bookkeeping(top, q, relation_batch=bad)


def test_solver_invalid_inputs_and_nonconvergence_do_not_return_dummy_flow():
    cpu_top = debug_topology()
    top, h, _ = debug_fields(cpu_top)
    q = incidence(top, h[top.local_node_global])
    d = incidence_transpose(top, q)
    c = local_weight(top, "unit")
    for bad in (-c, torch.full_like(c, float("nan"))):
        with pytest.raises(ValueError, match="positive"):
            solve_local_laplacian(top, d, bad)
    inconsistent = d.clone()
    inconsistent[top.center_positions[0]] += 1.0
    with pytest.raises(ValueError, match="zero local sum"):
        solve_local_laplacian(top, inconsistent)
    with pytest.raises(ValueError, match="finite"):
        solve_local_laplacian(top, torch.full_like(d, float("nan")))
    with pytest.raises(RuntimeError, match="not converged"):
        solve_local_laplacian(top, d, tol=1e-30, iteration_multiplier=1)
    with pytest.raises(TypeError, match="to"):
        incidence(cpu_top, h[cpu_top.local_node_global])
    with pytest.raises(ValueError, match="shape"):
        incidence(top, h)  # Physical H must first be gathered to local copies.
    with pytest.raises(ValueError, match="feature"):
        incidence(top, torch.empty((top.num_local_nodes, 0), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DEBUG CUDA parity requires a local GPU")
def test_cuda_batched_fixed_operators_match_cpu_and_true_solver_residuals():
    cpu_top = batch_topologies([debug_topology(), debug_topology()])
    cpu, h, other = debug_fields(cpu_top, feature_count=5)
    gpu = cpu_top.to("cuda")
    c = local_weight(cpu, "local_degree")
    q = c[:, None] * incidence(cpu, h[cpu.local_node_global])
    p = c[:, None] * incidence(cpu, other[cpu.local_node_global])
    gpu_q, gpu_p, gpu_c = q.cuda(), p.cuda(), c.cuda()
    for name, value in relation_terms(cpu, q, p).items():
        assert_close(value, relation_terms(gpu, gpu_q, gpu_p, relation_batch=3)[name].cpu())
    for name, value in transfer_bookkeeping(cpu, q).items():
        assert_close(value, transfer_bookkeeping(gpu, gpu_q, relation_batch=3)[name].cpu())
    cpu_recovery = recover_flows(cpu, q, c, iteration_multiplier=10)
    gpu_recovery = recover_flows(gpu, gpu_q, gpu_c, iteration_multiplier=10)
    for name in ("q_cut", "q_cycle", "q_recon"):
        assert_close(cpu_recovery[name], gpu_recovery[name].cpu())
    for name in ("cut_solver", "reconstruction_solver"):
        assert gpu_recovery[name]["relative_residual"].max() <= 1e-9
