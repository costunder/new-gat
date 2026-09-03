"""Mathematical and integration contracts for rebuilt Cycle PE v2."""

from __future__ import annotations

import inspect

import numpy as np
import torch
from scipy import sparse

from research.cycle_pe.v2.basis import (
    incidence_and_cycle_rank,
    left_nullspace_basis,
    sparse_left_nullspace_basis,
    validate_cycle_basis,
)
from research.cycle_pe.v2.data import Graph, collate
from research.cycle_pe.v2.model import CycleBasisPEModel, LeftNullBasisEncoder

EDGES = np.asarray([(0, 1), (2, 1), (2, 0), (3, 4), (5, 4), (5, 3)], dtype=np.int64).T


def _projector(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float64), mode="reduced")
    return q @ q.T


def test_sparse_basis_has_exact_cycle_dimension_and_is_in_left_nullspace():
    # Two triangles plus isolated node 6: m-n+c = 6-7+3 = 2.
    incidence, rank = incidence_and_cycle_rank(7, EDGES)
    basis = sparse_left_nullspace_basis(7, EDGES)
    assert sparse.isspmatrix_csr(basis)
    assert rank == 2 and basis.shape == (6, 2)
    np.testing.assert_allclose(incidence.T @ basis.toarray(), 0.0, atol=0.0)
    assert np.linalg.matrix_rank(basis.toarray()) == rank
    validate_cycle_basis(7, EDGES, basis)

    q = left_nullspace_basis(7, EDGES)
    np.testing.assert_allclose(q.T @ q, np.eye(rank), atol=2e-6)
    np.testing.assert_allclose(incidence.T @ q, 0.0, atol=2e-6)


def test_cycle_space_projector_transforms_correctly_under_orientation_and_permutation():
    q = left_nullspace_basis(7, EDGES)
    projector = _projector(q)

    signs = np.asarray([-1, 1, -1, 1, -1, 1], dtype=np.float64)
    flipped_edges = EDGES.copy()
    flipped = signs < 0
    flipped_edges[:, flipped] = flipped_edges[::-1, flipped]
    q_flipped = left_nullspace_basis(7, flipped_edges)
    np.testing.assert_allclose(
        _projector(q_flipped), signs[:, None] * projector * signs[None, :], atol=2e-6
    )

    permutation = np.asarray([5, 2, 0, 4, 1, 3])
    q_permuted = left_nullspace_basis(7, EDGES[:, permutation])
    np.testing.assert_allclose(
        _projector(q_permuted), projector[np.ix_(permutation, permutation)], atol=2e-6
    )


def test_projector_kernel_encoder_is_basis_and_orientation_invariant_and_permutation_equivariant():
    torch.manual_seed(9)
    q = torch.from_numpy(left_nullspace_basis(7, EDGES))
    bond = torch.randn(6, 5)
    encoder = LeftNullBasisEncoder(5, 7).eval()
    reference = encoder(bond, q)

    change = torch.tensor([[2.0, -0.5], [0.75, 1.5]])
    torch.testing.assert_close(encoder(bond, q @ change), reference, atol=3e-5, rtol=3e-5)

    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    torch.testing.assert_close(encoder(bond, signs[:, None] * q), reference, atol=3e-5, rtol=3e-5)

    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    actual = encoder(bond[permutation], q[permutation])
    torch.testing.assert_close(actual, reference[permutation], atol=3e-5, rtol=3e-5)

    chunked = encoder.forward_batch(bond, (q,), pair_budget=7, orthonormal_input=True)
    torch.testing.assert_close(chunked, reference, atol=3e-5, rtol=3e-5)


def test_pair_free_low_rank_contraction_matches_explicit_kernel_and_bounds_core(monkeypatch):
    torch.manual_seed(19)
    q, _ = torch.linalg.qr(torch.randn(23, 4), mode="reduced")
    values = torch.randn(23, 6, requires_grad=True)
    encoder = LeftNullBasisEncoder(6, 8)
    original_einsum = torch.einsum
    core_sizes = []

    def observed(equation, *operands):
        result = original_einsum(equation, *operands)
        if equation == "md,ma,mb->dab":
            core_sizes.append(result.numel())
        return result

    monkeypatch.setattr(torch, "einsum", observed)
    actual, leverage = encoder._projector_mix(q, values, pair_budget=7)
    projector = q @ q.T
    expected = projector.square() @ values
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(leverage, projector.diagonal(), atol=2e-6, rtol=2e-6)
    actual.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert core_sizes and max(core_sizes) <= 7
    source = inspect.getsource(LeftNullBasisEncoder._projector_mix)
    assert "projector_rows" not in source and "@ q.T" not in source


def _graph(num_nodes: int, edges: list[tuple[int, int]]) -> Graph:
    edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T
    q = left_nullspace_basis(num_nodes, edge_index.numpy())
    return Graph(
        x=torch.arange(num_nodes, dtype=torch.long).remainder(28)[:, None],
        edge_index=edge_index,
        edge_attr=torch.zeros(len(edges), 1, dtype=torch.long),
        y=torch.zeros(1),
        cycle_basis=torch.from_numpy(q),
        cycle_basis_is_orthonormal=torch.tensor(True),
    )


def test_deep_residual_model_runs_forward_backward_with_cycle_forest_and_isolate():
    cyclic = _graph(4, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)])
    forest = _graph(3, [(0, 1)])
    batch = collate([cyclic, forest])
    model = CycleBasisPEModel(
        dataset="zinc12k",
        hidden=16,
        pe_dim=8,
        layers=6,
        ffn_multiplier=2,
        dropout=0.0,
        basis_pair_budget=8,
    )
    prediction = model(batch)
    assert prediction.shape == (2, 1) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
