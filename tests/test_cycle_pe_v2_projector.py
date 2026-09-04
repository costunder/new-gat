"""Sparse DFS V2 integration contracts (legacy filename retained for test discovery)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse

from research.cycle_pe.tests.test_v2_model import _disconnected_graph, _graph
from research.cycle_pe.v2.basis import (
    build_cycle_basis,
    incidence_and_cycle_rank,
    validate_cycle_basis,
)
from research.cycle_pe.v2.data import collate
from research.cycle_pe.v2.model import CycleBasisPEModel

EDGES = np.asarray([(0, 1), (2, 1), (2, 0), (3, 4), (5, 4), (5, 3)], dtype=np.int64).T


def test_sparse_basis_has_exact_cycle_dimension_and_zero_incidence_residual():
    incidence, rank = incidence_and_cycle_rank(7, EDGES)
    basis = build_cycle_basis(7, EDGES)
    assert sparse.isspmatrix_csr(basis)
    assert rank == 2 and basis.shape == (6, 2)
    residual = incidence.T @ basis
    residual.eliminate_zeros()
    assert residual.nnz == 0
    assert set(np.unique(basis.data)) <= {-1.0, 1.0}
    validate_cycle_basis(7, EDGES, basis)


def test_reversing_edge_orientation_preserves_sparse_cycle_membership():
    basis = build_cycle_basis(7, EDGES)
    signs = np.asarray([-1, 1, -1, 1, -1, 1], dtype=np.float32)
    flipped_edges = EDGES.copy()
    flipped_edges[:, signs < 0] = flipped_edges[::-1, signs < 0]
    changed = build_cycle_basis(7, flipped_edges)
    # Reversing a chord may also flip the whole cycle column; both row and
    # column signs disappear in the selected unsigned membership.
    difference = abs(changed) - abs(basis)
    difference.eliminate_zeros()
    assert difference.nnz == 0
    validate_cycle_basis(7, flipped_edges, changed)


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_preparation_batch_and_forward_never_factorize_or_densify_sparse_cycles(
    monkeypatch, encoding
):
    def forbidden(*args, **kwargs):
        raise AssertionError("sparse DFS pipeline invoked dense conversion or factorization")

    for name in ("qr", "svd", "eigh", "eig", "inv", "pinv", "cholesky"):
        monkeypatch.setattr(np.linalg, name, forbidden)
        monkeypatch.setattr(torch.linalg, name, forbidden)
    for kind in (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix):
        monkeypatch.setattr(kind, "toarray", forbidden)
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    graphs = [_graph(7, complete=True), _graph(3, forest=True), _disconnected_graph()]
    batch = collate(graphs)
    model = CycleBasisPEModel(
        dataset="zinc12k", encoding=encoding, hidden=16, pe_dim=8, layers=6
    )
    prediction = model(batch)
    assert prediction.shape == (3, 1) and torch.isfinite(prediction).all()
    (prediction - batch.y).abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_disjoint_sparse_batch_retains_every_cycle_and_membership_nonzero():
    graphs = [_graph(4), _graph(7, complete=True), _graph(4, forest=True), _disconnected_graph()]
    batch = collate(graphs)
    assert batch.cycle_membership.layout == torch.sparse_coo
    assert batch.cycle_membership.shape == (
        sum(len(graph.edge_attr) for graph in graphs),
        sum(graph.cycle_basis.shape[1] for graph in graphs),
    )
    assert batch.cycle_membership._nnz() == sum(graph.cycle_basis._nnz() for graph in graphs)
    assert batch.cycle_lengths.shape == (batch.cycle_membership.shape[1],)
    edge_ids, cycle_ids = batch.cycle_membership.indices()
    assert torch.equal(batch.edge_graph_index[edge_ids], batch.cycle_graph_index[cycle_ids])
    assert torch.all(batch.cycle_membership.values() == 1)


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_deep_residual_forward_backward_handles_cycles_forests_and_isolates(encoding):
    batch = collate([_graph(5, complete=True), _graph(3, forest=True), _graph(1, forest=True)])
    model = CycleBasisPEModel(
        dataset="zinc12k", encoding=encoding, hidden=16, pe_dim=8, layers=6
    )
    assert len(model.layers) == 6
    prediction = model(batch)
    assert prediction.shape == (3, 1) and torch.isfinite(prediction).all()
    (prediction - batch.y).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
