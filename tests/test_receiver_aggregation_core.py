"""DEBUG dense references; no reduced final dataset or learned-model claim."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.local_energy_relations.operators import (
    aggregate_squares,
    incidence,
    incidence_transpose,
    relation_terms,
)
from research.local_energy_relations.receiver_aggregation.operators import (
    ambient_receipt_projection,
    batch_receiver_operators,
    fuse_divergence,
    fuse_receipts,
    fuse_transpose,
    fused_apply,
    fused_transpose,
    graph_norm_sq,
    observation_jvp,
    prepare_receiver_operator,
    project_components,
    receipt_reconstruct,
    recover_fused,
    shared_flow,
    tagged_receipts,
)
from research.local_energy_relations.topology import batch_topologies, build_topology


def top(n=7):
    return build_topology(n, np.asarray([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5)]).T)


def op(mode="local_degree", device="cpu"):
    return prepare_receiver_operator(top(), mode).to(device)


def field(operator, features=4):
    generator = torch.Generator().manual_seed(20261004)
    return torch.randn(
        (operator.topology.n, features), generator=generator, dtype=torch.float64
    ).to(operator.weights.device)


def close(left, right, atol=2e-10, rtol=2e-10):
    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_cached_csr_exact_matrixfree_forward_and_adjoint(mode):
    operator = op(mode)
    h = field(operator)
    q, d = shared_flow(operator, h)
    y = fused_apply(operator, h)
    close(y, fuse_divergence(operator, d, relation_batch=1))
    close(y, fuse_receipts(operator, tagged_receipts(operator, d)))
    probe = torch.randn_like(y)
    copied = fuse_transpose(operator, probe, relation_batch=1)
    local = incidence_transpose(
        operator.topology, operator.weights[:, None] * incidence(operator.topology, copied)
    )
    matrixfree = h.new_zeros(h.shape).index_add(0, operator.topology.local_node_global, local)
    close(fused_transpose(operator, probe), matrixfree)
    close((y * probe).sum(), (h * fused_transpose(operator, probe)).sum())
    close(operator.normal_diagonal, operator.matrix.to_dense().square().sum(0))
    assert q.shape == (operator.topology.num_local_edges, h.shape[1])


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize(
    "pairs,n",
    [
        ([], 4),
        ([(0, 1)], 4),
        ([(0, 1), (1, 2), (2, 3)], 5),
        ([(0, 1), (1, 2), (2, 3), (0, 3)], 4),
        ([(0, 1), (0, 2), (1, 2), (2, 3)], 5),
    ],
)
def test_center_entries_directed_laplacian_and_exact_restricted_rank(mode, pairs, n):
    edges = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T
    operator = prepare_receiver_operator(build_topology(n, edges), mode)
    a = operator.matrix.to_dense()
    center = a[operator.topology.center_positions]
    close(center.sum(1), torch.zeros(n, dtype=torch.float64))
    offdiag = center - torch.diag(center.diagonal())
    assert bool((offdiag <= 1e-14).all())
    for u, v in edges.T:
        assert center[u, v] < 0 and center[v, u] < 0
    expected = n - operator.num_components
    assert int(torch.linalg.matrix_rank(a)) == expected
    assert int(torch.linalg.matrix_rank(center)) == expected
    assert operator.metadata["restricted_field_rank"] == expected
    assert operator.metadata["restricted_q_kernel_dimension"] == 0


def test_tagged_union_coverage_and_ambient_kernel_projection():
    operator = op()
    q, d = shared_flow(operator, field(operator))
    receipts = tagged_receipts(operator, d)
    close(receipt_reconstruct(operator, receipts), d)
    projection = ambient_receipt_projection(operator, receipts)
    contrast, minimum = projection["contrast"], projection["minimum_norm_receipts"]
    close(fuse_receipts(operator, contrast), torch.zeros_like(projection["fused"]))
    close((minimum * contrast).sum(0), torch.zeros(receipts.shape[1], dtype=torch.float64))
    close(receipts.square().sum(0), minimum.square().sum(0) + contrast.square().sum(0))
    assert bool((contrast.square().sum(0) > 0).all())
    assert operator.metadata["ambient_receipt_kernel_dimension"] > 0
    # Ambient receipt contrast is generally outside the consistent shared-H image.
    fused_h, info = recover_fused(operator, projection["fused"], tolerance=1e-11)
    reconstructed_q, reconstructed_d = shared_flow(operator, fused_h)
    close(reconstructed_q, q, atol=2e-9, rtol=2e-9)
    close(tagged_receipts(operator, reconstructed_d), receipts, atol=2e-9, rtol=2e-9)
    assert bool(info["converged"].all())


def test_inconsistent_tagged_copies_raise():
    operator = op()
    _, d = shared_flow(operator, field(operator, 1))
    receipts = tagged_receipts(operator, d).clone()
    repeated = torch.where(operator.receipt_counts[operator.topology.shared_node_left] > 1)[0]
    assert repeated.numel()
    receipts[repeated[0], 0] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        receipt_reconstruct(operator, receipts)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_global_minimum_norm_reconstruction_matches_dense_pseudoinverse(mode):
    operator = op(mode)
    h = field(operator)
    y = fused_apply(operator, h)
    reconstructed, info = recover_fused(operator, y, tolerance=1e-11)
    dense = torch.linalg.pinv(operator.matrix.to_dense()) @ y
    close(reconstructed, dense, atol=5e-9, rtol=5e-9)
    close(reconstructed, project_components(operator, h), atol=5e-9, rtol=5e-9)
    assert bool((info["relative_residual"] <= 1e-11).all())
    assert reconstructed[-1].count_nonzero() == 0  # isolated physical node


def test_disjoint_graph_and_channel_batch_equivalence():
    first, second = top(), build_topology(5, np.asarray([(0, 1), (1, 2), (2, 3)]).T)
    combined = prepare_receiver_operator(batch_topologies([first, second]), "local_degree")
    h = field(combined, 7)
    y = fused_apply(combined, h)
    recovered, info = recover_fused(combined, y, tolerance=1e-11)
    assert info["iterations"].shape == (2, 7)
    chunked = torch.cat(
        [recover_fused(combined, y[:, s : s + 2], tolerance=1e-11)[0] for s in range(0, 7, 2)],
        dim=1,
    )
    close(recovered, chunked, atol=5e-9, rtol=5e-9)
    first_op = prepare_receiver_operator(first, "local_degree")
    second_op = prepare_receiver_operator(second, "local_degree")
    expected = torch.cat(
        [
            recover_fused(first_op, fused_apply(first_op, h[:7]), tolerance=1e-11)[0],
            recover_fused(second_op, fused_apply(second_op, h[7:]), tolerance=1e-11)[0],
        ]
    )
    close(recovered, expected, atol=5e-9, rtol=5e-9)


def test_cached_operator_batch_equals_complete_topology_rebuild_and_norms():
    topologies = [top(), build_topology(5, np.asarray([(0, 1), (1, 2), (2, 3)]).T)]
    operators = [prepare_receiver_operator(t, "local_degree") for t in topologies]
    batched = batch_receiver_operators(operators)
    rebuilt = prepare_receiver_operator(batch_topologies(topologies), "local_degree")
    close(batched.matrix.to_dense(), rebuilt.matrix.to_dense())
    close(batched.matrix_transpose.to_dense(), rebuilt.matrix_transpose.to_dense())
    close(batched.normal_diagonal, rebuilt.normal_diagonal)
    assert torch.equal(batched.component_index, rebuilt.component_index)
    assert torch.equal(batched.node_graph, rebuilt.node_graph)
    assert batched.metadata["restricted_field_rank"] == rebuilt.metadata["restricted_field_rank"]
    h = field(batched, 3)
    q, d = shared_flow(batched, h)
    receipts = tagged_receipts(batched, d)
    for space, values, sizes in (
        ("physical", h, [t.n for t in topologies]),
        ("local_node", d, [t.num_local_nodes for t in topologies]),
        ("local_edge", q, [t.num_local_edges for t in topologies]),
        ("tagged", receipts, [t.shared_node_left.size for t in topologies]),
    ):
        expected = torch.stack([part.square().sum(0) for part in values.split(sizes)])
        close(graph_norm_sq(batched, values, space=space), expected)
    recovered, info = recover_fused(batched, fused_apply(batched, h), tolerance=1e-11)
    close(recovered, project_components(batched, h), atol=5e-9, rtol=5e-9)
    assert info["relative_residual"].shape == (2, 3)
    with pytest.raises(ValueError, match="common weight"):
        batch_receiver_operators([operators[0], prepare_receiver_operator(topologies[1], "unit")])


def test_zero_observations_and_nonconvergence_are_explicit():
    operator = op()
    zeros = torch.zeros((operator.topology.num_local_nodes, 3), dtype=torch.float64)
    h, info = recover_fused(operator, zeros)
    assert h.count_nonzero() == 0 and info["global_iterations"] == 0
    inconsistent = zeros.clone()
    isolated_row = torch.where(operator.topology.local_degree == 0)[0][0]
    inconsistent[isolated_row, 0] = 1
    with pytest.raises(RuntimeError, match="breakdown|not converged"):
        recover_fused(operator, inconsistent, max_iterations_factor=1)
    with pytest.raises(ValueError, match="finite"):
        recover_fused(operator, torch.full_like(zeros, float("nan")))


@pytest.mark.parametrize("tolerance", [0, 1, float("nan")])
def test_invalid_solver_tolerance_is_explicit(tolerance):
    operator = op()
    with pytest.raises(ValueError, match="tolerance"):
        recover_fused(operator, fused_apply(operator, field(operator)), tolerance=tolerance)


def test_energy_relation_jvp_is_exact_autograd_derivative():
    operator = op()
    generator = torch.Generator().manual_seed(20261004)
    local = torch.randn(
        (operator.topology.num_local_nodes, 3),
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    direction = torch.randn(local.shape, generator=generator, dtype=torch.float64)

    def observations(h):
        g = incidence(operator.topology, h)
        q = operator.weights[:, None] * g
        energy = aggregate_squares(operator.topology, g, weights=operator.weights)
        relations = relation_terms(operator.topology, q)
        return (
            energy,
            relations["shared_edges"],
            relations["shared_nodes"],
            relations["distinct_edges"],
        )

    _, automatic = torch.autograd.functional.jvp(observations, local, direction)
    explicit = observation_jvp(operator, local, direction, relation_batch=1)
    for name, expected in zip(
        ("energy", "shared_edges", "shared_nodes", "distinct_edges"), automatic, strict=True
    ):
        close(explicit[name], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DEBUG CUDA not available")
def test_cuda_sparse_forward_adjoint_and_reconstruction():
    operator = op(device="cuda")
    h = field(operator)
    q, d = shared_flow(operator, h)
    y = fused_apply(operator, h)
    close(y, fuse_divergence(operator, d, relation_batch=1), atol=2e-9, rtol=2e-9)
    reconstructed, info = recover_fused(operator, y, tolerance=1e-10)
    recovered_q, _ = shared_flow(operator, reconstructed)
    close(recovered_q, q, atol=5e-8, rtol=5e-8)
    assert bool(info["converged"].all())
