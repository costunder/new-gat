"""Small algebra/schema fixtures only; never run training or download datasets."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import fields, is_dataclass, replace
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chartgat.cache import CacheCorruptError, CacheIncompleteError, CacheWrongRequestError
from research.cycle_pe.v2 import basis, data


def _official(num_nodes=4, edges=((0, 1), (1, 2), (2, 3), (0, 3)), *, dataset="zinc12k"):
    pairs = [*edges, *((v, u) for u, v in edges)]
    atom_width, bond_width, targets = (1, 1, 1) if dataset == "zinc12k" else (9, 3, 11)
    return SimpleNamespace(
        num_nodes=num_nodes,
        x=torch.zeros((num_nodes, atom_width), dtype=torch.long),
        edge_index=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        edge_attr=torch.zeros((len(pairs), bond_width), dtype=torch.long),
        y=torch.arange(targets, dtype=torch.float32) + 0.75,
    )


def _graph(*args, basis_backend="dfs_fundamental", **kwargs):
    return data.prepare_graph(
        _official(*args, **kwargs),
        dataset=kwargs.get("dataset", "zinc12k"),
        basis_backend=basis_backend,
    )


def _edges(pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T


@pytest.mark.parametrize(
    "nodes,edges,rank",
    [
        (1, (), 0),
        (5, (), 0),
        (4, ((0, 1), (1, 2)), 0),
        (3, ((0, 1), (0, 2), (1, 2)), 1),
        (4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), 3),
        (8, ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (6, 7)), 2),
    ],
)
def test_entire_left_nullspace_is_returned_for_connected_disconnected_and_empty(nodes, edges, rank):
    edge_index = _edges(edges)
    incidence, actual_rank = basis.incidence_and_cycle_rank(nodes, edge_index)
    values = basis.left_nullspace_basis(nodes, edge_index)
    assert actual_rank == rank
    assert values.shape == (len(edges), rank)
    assert values.dtype == np.float32
    assert values.format == incidence.format == "csr"
    np.testing.assert_allclose((incidence.T @ values).toarray(), 0, atol=2e-7)
    if rank:
        assert np.linalg.matrix_rank(values.toarray()) == rank
    basis.validate_cycle_basis(nodes, edge_index, values)


def test_uses_sparse_dfs_without_any_decomposition_or_dense_materialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("spectral/rank decomposition is forbidden")

    for name in ("svd", "eig", "eigh", "matrix_rank", "qr", "cholesky"):
        monkeypatch.setattr(basis.np.linalg, name, forbidden)
    monkeypatch.setattr(basis.sparse.csr_matrix, "toarray", forbidden)
    pairs = [(u, v) for u in range(8) for v in range(u + 1, 8)]
    sparse_values = basis.sparse_left_nullspace_basis(8, _edges(pairs))
    values = basis.left_nullspace_basis(8, _edges(pairs))
    assert sparse_values.shape == (28, 21) and sparse_values.nnz < np.prod(sparse_values.shape)
    assert values.shape == (28, 21)
    basis.validate_cycle_basis(8, _edges(pairs), values)


def test_default_backend_returns_complete_raw_signed_dfs_basis():
    pairs = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edge_index = _edges(pairs)
    raw_sparse = basis.dfs_fundamental_cycle_basis(5, edge_index)
    raw = basis.build_cycle_basis(5, edge_index, backend="dfs_fundamental")
    default = basis.build_cycle_basis(5, edge_index)
    assert raw_sparse.shape == raw.shape == default.shape == (10, 6)
    np.testing.assert_array_equal(raw.toarray(), raw_sparse.toarray())
    np.testing.assert_array_equal(raw.toarray(), default.toarray())
    assert set(raw.data) <= {-1.0, 1.0}
    assert basis.BASIS_BACKENDS == ("dfs_fundamental",)
    basis.validate_cycle_basis(5, edge_index, raw_sparse)


def test_circular_positions_follow_chord_and_actual_tree_path_not_csr_row_order():
    edges = _edges([(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)])
    values, positions = basis.build_cycle_coordinates(5, edges)
    assert values.shape == (5, 1)
    assert not np.array_equal(positions, np.arange(5))
    rows = np.repeat(np.arange(5), np.diff(values.indptr))
    ordered_edges = edges[:, rows[np.argsort(positions)]]
    for index in range(5):
        assert len(set(ordered_edges[:, index]) & set(ordered_edges[:, (index + 1) % 5])) == 1
    factors = basis.cycle_position_factors(values, positions)
    assert factors.shape == (2, values.nnz)
    np.testing.assert_allclose(np.sum(factors * factors, axis=0), 1.0, atol=2e-7)
    basis.validate_cycle_positions(5, edges, values, positions)


@pytest.mark.parametrize("damage", ["duplicate", "out_of_range", "fractional", "nonadjacent"])
def test_cycle_position_validator_rejects_fake_order_even_with_full_valid_basis(damage):
    edges = _edges([(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)])
    values, positions = basis.build_cycle_coordinates(5, edges)
    if damage == "duplicate":
        positions[1] = positions[0]
    elif damage == "out_of_range":
        positions[0] = 5
    elif damage == "fractional":
        positions = positions.astype(np.float64) + 0.25
    else:
        first = int(np.flatnonzero(positions == 0)[0])
        second = int(np.flatnonzero(positions == 1)[0])
        positions[first], positions[second] = positions[second], positions[first]
    basis.validate_cycle_basis(5, edges, values)
    with pytest.raises(ValueError, match="position"):
        basis.validate_cycle_positions(5, edges, values, positions)


@pytest.mark.parametrize("shift", [0, 1, 3])
@pytest.mark.parametrize("direction", [-1, 1])
def test_cached_positions_allow_cycle_origin_shift_reversal_and_independent_basis_sign(
    shift, direction
):
    graph = _graph(5, [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)])
    position_indices = (direction * graph.cycle_position_indices + shift) % 5
    angle = 2.0 * torch.pi * position_indices.double() / 5
    transformed = replace(
        graph,
        cycle_basis=-graph.cycle_basis,
        cycle_position_indices=position_indices,
        cycle_position_values=torch.stack((angle.cos(), angle.sin())).float(),
    )
    data.validate_graph(transformed, dataset="zinc12k")
    first = graph.cycle_position_values.T @ graph.cycle_position_values
    second = transformed.cycle_position_values.T @ transformed.cycle_position_values
    torch.testing.assert_close(first, second)


def test_cached_positions_and_lengths_follow_sparse_cycle_column_permutation():
    graph = _graph(5, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (0, 4)])
    indices = graph.cycle_basis.indices().numpy()
    original = basis.sparse.coo_matrix(
        (graph.cycle_basis.values().numpy(), (indices[0], indices[1])),
        shape=tuple(graph.cycle_basis.shape),
    ).tocsr()
    order_matrix = basis.sparse.csr_matrix(
        (graph.cycle_position_indices.numpy(), original.indices, original.indptr),
        shape=original.shape,
    )
    permutation = np.arange(original.shape[1])[::-1].copy()
    changed = original[:, permutation].tocsr()
    changed.sort_indices()
    changed_order = order_matrix[:, permutation].tocsr()
    changed_order.sort_indices()
    changed_positions = changed_order.data
    assert changed_positions.shape == (original.nnz,)
    changed_coo = changed.tocoo()
    transformed = replace(
        graph,
        cycle_basis=torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack((changed_coo.row, changed_coo.col)).astype(np.int64)),
            torch.from_numpy(changed_coo.data),
            changed_coo.shape,
            is_coalesced=True,
            check_invariants=True,
        ),
        cycle_lengths=graph.cycle_lengths[torch.from_numpy(permutation)],
        cycle_position_indices=torch.from_numpy(changed_positions),
        cycle_position_values=torch.from_numpy(
            basis.cycle_position_factors(changed, changed_positions)
        ),
    )
    data.validate_graph(transformed, dataset="zinc12k")
    for component in range(2):
        original_factor = basis.sparse.csr_matrix(
            (graph.cycle_position_values[component].numpy(), original.indices, original.indptr),
            shape=original.shape,
        )
        changed_factor = basis.sparse.csr_matrix(
            (transformed.cycle_position_values[component].numpy(), changed.indices, changed.indptr),
            shape=changed.shape,
        )
        np.testing.assert_array_equal(
            changed_factor.toarray(), original_factor[:, permutation].toarray()
        )


def test_unknown_basis_backend_fails_closed():
    with pytest.raises(ValueError, match="backend"):
        basis.build_cycle_basis(3, _edges([(0, 1)]), backend="unknown")
    with pytest.raises(ValueError, match="basis_backend"):
        data.prepare_graph(_official(), basis_backend="unknown")
    with pytest.raises(ValueError, match="retired"):
        basis.build_cycle_basis(3, _edges([(0, 1)]), backend="thin_q")


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1.0, 1e4, 1e8])
def test_fundamental_basis_column_scales_and_signs_do_not_trigger_absolute_rank_threshold(scale):
    edge_index = _edges([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    q = basis.left_nullspace_basis(4, edge_index)
    changed = q.toarray()[:, ::-1] * np.asarray([scale, -scale])
    basis.validate_cycle_basis(4, edge_index, changed)
    changed[:, 1] = changed[:, 0]
    with pytest.raises(ValueError, match="full column rank"):
        basis.validate_cycle_basis(4, edge_index, changed)


def test_generic_mixed_basis_without_structural_witness_is_explicitly_unsupported():
    edges = _edges([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    changed = basis.left_nullspace_basis(4, edges) @ np.asarray([[2.0, 0.5], [-1.0, 3.0]])
    with pytest.raises(ValueError, match="arbitrary mixed bases are unsupported"):
        basis.validate_cycle_basis(4, edges, changed)


@pytest.mark.parametrize(
    "nodes,edges,error",
    [
        (0, _edges([]), "positive integer"),
        (True, _edges([]), "positive integer"),
        (3, np.array([[0.0], [1.0]]), "integer node"),
        (3, np.array([[True], [False]]), "integer node"),
        (3, np.array([0, 1]), "shape"),
        (3, _edges([(0, 3)]), "out of range"),
        (3, _edges([(-1, 1)]), "out of range"),
        (3, _edges([(0, 0)]), "self-loop"),
        (3, _edges([(0, 1), (0, 1)]), "duplicate"),
    ],
)
def test_basis_input_contract_rejects_invalid_incidence_inputs(nodes, edges, error):
    with pytest.raises(ValueError, match=error):
        basis.left_nullspace_basis(nodes, edges)


def test_arbitrary_edge_orientation_is_accepted_and_transports_projector():
    edges = _edges([(0, 1), (0, 2), (1, 2)])
    changed = edges.copy()
    changed[:, 1] = changed[::-1, 1]
    q, changed_q = basis.left_nullspace_basis(3, edges), basis.left_nullspace_basis(3, changed)
    signs = np.asarray([1.0, -1.0, 1.0])
    np.testing.assert_allclose(
        (changed_q @ changed_q.T).toarray(),
        signs[:, None] * (q @ q.T).toarray() * signs[None, :],
        atol=2e-6,
    )


@pytest.mark.parametrize("kind", ["nonfinite", "wrong_width", "outside_nullspace", "rank", "half"])
def test_basis_validation_rejects_incomplete_or_corrupt_coordinates(kind):
    edge_index = _edges([(0, 1), (0, 2), (1, 2)])
    values = basis.left_nullspace_basis(3, edge_index).toarray()
    if kind == "nonfinite":
        values[0, 0] = np.nan
    elif kind == "wrong_width":
        values = values[:, :0]
    elif kind == "outside_nullspace":
        values[0, 0] += 0.1
    elif kind == "rank":
        values[:] = 0
    else:
        values = values.astype(np.float16)
    with pytest.raises(ValueError):
        basis.validate_cycle_basis(3, edge_index, values)


def test_edge_order_alignment_is_exact_not_assumed_lexicographic():
    pairs = _edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])
    original = basis.left_nullspace_basis(4, pairs).toarray()
    permutation = np.array([3, 0, 4, 1, 2])
    transformed = basis.left_nullspace_basis(4, pairs[:, permutation]).toarray()
    # Different spanning forests may rotate coordinates; the spaces must agree.
    transported = original[permutation]
    alignment = np.linalg.lstsq(transported, transformed, rcond=None)[0]
    np.testing.assert_allclose(transported @ alignment, transformed, atol=2e-7)


@pytest.mark.parametrize("dataset", data.DATASETS)
def test_official_chemistry_targets_preserved_and_directed_copies_sorted(dataset):
    source = _official(4, [(2, 3), (0, 3), (1, 2), (0, 1)], dataset=dataset)
    source.edge_attr[:, 0] = torch.tensor([1, 2, 1, 3, 1, 2, 1, 3])
    graph = data.prepare_graph(source, dataset=dataset)
    assert {field.name for field in fields(graph)} == {
        "x",
        "edge_index",
        "edge_attr",
        "y",
        "cycle_basis",
        "cycle_lengths",
        "edge_cycle_counts",
        "edge_cycle_features",
        "cycle_position_indices",
        "cycle_position_values",
    }
    torch.testing.assert_close(graph.x, source.x)
    torch.testing.assert_close(graph.y, source.y)
    torch.testing.assert_close(graph.edge_index, torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]]))
    torch.testing.assert_close(graph.edge_attr[:, 0], torch.tensor([3, 2, 1, 1]))
    assert graph.cycle_basis.shape == (4, 1)
    assert graph.cycle_basis.dtype == torch.float32
    assert graph.cycle_basis.layout == torch.sparse_coo and graph.cycle_basis.is_coalesced()
    assert graph.cycle_position_indices.shape == (4,)
    assert graph.cycle_position_values.shape == (2, 4)
    assert sorted(graph.cycle_position_indices.tolist()) == list(range(4))
    torch.testing.assert_close(graph.cycle_position_values.square().sum(dim=0), torch.ones(4))
    torch.testing.assert_close(graph.cycle_lengths, torch.tensor([4.0]))
    torch.testing.assert_close(graph.edge_cycle_counts, torch.ones(4))
    torch.testing.assert_close(
        graph.edge_cycle_features,
        torch.tensor([[np.log1p(4), 0.25]], dtype=torch.float32).expand(4, 2),
    )
    incidence, _ = basis.incidence_and_cycle_rank(4, graph.edge_index.numpy())
    np.testing.assert_allclose(incidence.T @ graph.cycle_basis.to_dense().numpy(), 0, atol=1e-7)


@pytest.mark.parametrize("kind", ["fractional_atom", "nonfinite_bond", "range", "copies", "loop"])
def test_official_graphs_are_not_silently_coerced_to_valid_inputs(kind):
    source = _official()
    if kind == "fractional_atom":
        source.x = source.x.float()
        source.x[0, 0] = 0.5
    elif kind == "nonfinite_bond":
        source.edge_attr = source.edge_attr.float()
        source.edge_attr[0, 0] = torch.nan
    elif kind == "range":
        source.edge_index[0, 0] = 4
    elif kind == "copies":
        source.edge_attr[0, 0] = 1
    else:
        source.edge_index[0, 0] = source.edge_index[1, 0]
    with pytest.raises(ValueError):
        data.prepare_graph(source)


@pytest.mark.parametrize("kind", ["atom_width", "atom_value", "bond_width", "bond_value", "target"])
def test_dataset_categorical_and_target_schema_is_checked(kind):
    source = _official()
    if kind == "atom_width":
        source.x = torch.zeros((4, 2), dtype=torch.long)
    elif kind == "atom_value":
        source.x[0, 0] = 28
    elif kind == "bond_width":
        source.edge_attr = torch.zeros((8, 3), dtype=torch.long)
    elif kind == "bond_value":
        source.edge_attr[:] = 4
    else:
        source.y = torch.zeros(11)
    with pytest.raises(ValueError):
        data.prepare_graph(source, dataset="zinc12k")


def test_sparse_batch_keeps_every_cycle_and_never_mixes_graphs():
    graphs = [
        _graph(2, [(0, 1)]),
        _graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]),
        _graph(1, []),
        _graph(),
    ]
    batch = data.collate(graphs)
    assert batch.cycle_membership.layout == torch.sparse_coo
    assert batch.cycle_membership.is_coalesced()
    assert batch.cycle_membership.shape == (10, 3)
    torch.testing.assert_close(batch.ptr, torch.tensor([0, 2, 6, 7, 11]))
    torch.testing.assert_close(batch.edge_ptr, torch.tensor([0, 1, 6, 6, 10]))
    assert batch.cycle_basis_shapes == ((1, 0), (5, 2), (0, 0), (4, 1))
    assert batch.cycle_membership._nnz() == sum(g.cycle_basis._nnz() for g in graphs)
    torch.testing.assert_close(
        batch.cycle_position_values, torch.cat([g.cycle_position_values for g in graphs], dim=1)
    )
    matrix = batch.cycle_membership.to_dense()
    cycle_start = 0
    for index, graph in enumerate(graphs):
        start, end = batch.edge_ptr[index : index + 2]
        cycle_end = cycle_start + graph.cycle_basis.shape[1]
        torch.testing.assert_close(
            matrix[start:end, cycle_start:cycle_end], graph.cycle_basis.to_dense().abs()
        )
        assert not matrix[start:end, :cycle_start].any()
        assert not matrix[start:end, cycle_end:].any()
        cycle_start = cycle_end
        torch.testing.assert_close(
            batch.edge_index[:, start:end] - batch.ptr[index], graph.edge_index
        )
    moved = batch.to(torch.device("cpu"))
    torch.testing.assert_close(batch.cycle_membership, moved.cycle_membership)
    row, col = batch.cycle_membership.indices()
    torch.testing.assert_close(batch.edge_graph_index[row], batch.cycle_graph_index[col])


def test_sparse_pin_memory_pins_indices_and_values_not_unsupported_sparse_storage(monkeypatch):
    batch = data.collate([_graph(), _graph(1, [])])
    called = []

    def pin(tensor):
        assert tensor.layout == torch.strided
        called.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    pinned = batch.pin_memory()
    torch.testing.assert_close(pinned.cycle_membership, batch.cycle_membership)
    assert any(
        tensor.data_ptr() == batch.cycle_membership.indices().data_ptr() for tensor in called
    )
    assert any(tensor.data_ptr() == batch.cycle_membership.values().data_ptr() for tensor in called)
    assert (
        len(called)
        == sum(isinstance(getattr(batch, field.name), torch.Tensor) for field in fields(batch)) + 1
    )


def test_batch_to_transfers_one_blockdiagonal_membership_not_once_per_graph(monkeypatch):
    batch = data.collate([_graph(), _graph(), _graph(1, [])])
    transferred = []
    original = torch.Tensor.to

    def moved(tensor, *args, **kwargs):
        transferred.append(id(tensor))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", moved)
    result = batch.to("cpu")
    assert transferred.count(id(batch.cycle_membership)) == 1
    assert transferred.count(id(batch.cycle_position_values)) == 1
    assert not hasattr(result, "cycle_position_indices")
    assert result.cycle_basis_shapes == batch.cycle_basis_shapes


def test_collate_does_not_repeat_static_nullspace_algebra_or_finite_scans(monkeypatch):
    graphs = [_graph(), _graph(2, [(0, 1)])]

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed graph algebra must not repeat per minibatch")

    monkeypatch.setattr(data, "validate_cycle_basis", forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    for _ in range(3):
        assert data.collate(graphs).cycle_membership._nnz() == 4


def test_all_forest_batch_retains_empty_cycle_dimension():
    batch = data.collate([_graph(2, [(0, 1)]), _graph(1, [])])
    assert batch.cycle_membership.shape == (1, 0)
    assert batch.cycle_membership._nnz() == 0
    assert batch.cycle_lengths.shape == (0,)
    assert batch.cycle_position_values.shape == (2, 0)
    torch.testing.assert_close(batch.edge_cycle_counts, torch.zeros(1))


def test_collation_rejects_empty_or_inconsistent_graph_schemas():
    with pytest.raises(ValueError, match="empty"):
        data.collate([])
    with pytest.raises(ValueError, match="different molecular schemas"):
        data.collate([_graph(), _graph(dataset="peptides_struct")])
    with pytest.raises(ValueError, match="cycle-basis schema"):
        data.collate([replace(_graph(), cycle_basis=torch.ones(2, 1))])


def _install_official_fixture(monkeypatch):
    official = {split: [_official(), _official(2, [])] for split in data.SPLITS}
    monkeypatch.setattr(data, "load_official_splits", lambda *args, **kwargs: official)
    original_load = data.load_benchmark

    def fixture_load(*args, **kwargs):
        kwargs.setdefault("workers", 0)  # Explicit serial synthetic/cache fixtures only.
        return original_load(*args, **kwargs)

    monkeypatch.setattr(data, "load_benchmark", fixture_load)
    return official


def _cache_paths(tmp_path, protocol):
    directory = tmp_path / data.CACHE_NAMESPACE / "zinc12k"
    assert str(next(directory.iterdir())) == protocol["cache_directory"]
    return next(directory.iterdir()) / "train.pt", next(directory.iterdir()) / "train.json"


def _rewrite_cache(cache, meta, rows):
    torch.save(rows, cache)
    metadata = json.loads(meta.read_text())
    metadata["cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    meta.write_text(json.dumps(metadata))


def test_cache_roundtrip_is_isolated_hashes_implementation_and_skips_basis_rebuild(
    tmp_path, monkeypatch
):
    _install_official_fixture(monkeypatch)
    first, protocol = data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    assert protocol["official_splits"]
    assert protocol["preparation"]["representation"] == "ordered_sparse_dfs_cycle_coordinates"
    assert data.CACHE_NAMESPACE == "cycle_pe_v2_ordered_dfs_benchmark"
    assert set(protocol["preparation"]["implementation_sha256"]) == {
        "v2/basis.py",
        "v2/data.py",
        "official_adapter",
    }
    assert not (tmp_path / "cycle_pe_benchmark").exists()

    def no_svd(*args, **kwargs):
        raise AssertionError("cached sparse DFS matrices must not be recomputed")

    monkeypatch.setattr(data, "build_cycle_coordinates", no_svd)
    second, restored = data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    assert restored == protocol
    for split in data.SPLITS:
        for original, cached in zip(first[split], second[split], strict=True):
            for field in fields(original):
                torch.testing.assert_close(
                    getattr(original, field.name), getattr(cached, field.name)
                )


def test_retired_q_backend_rejected_without_creating_or_overwriting_legacy_cache(
    tmp_path, monkeypatch
):
    _install_official_fixture(monkeypatch)
    legacy = tmp_path / "cycle_pe_v2_projector_kernel_benchmark" / "old-cache"
    legacy.mkdir(parents=True)
    marker = legacy / "keep.txt"
    marker.write_text("existing results must be preserved")
    support_legacy = tmp_path / "cycle_pe_v2_sparse_dfs_benchmark" / "support-only-cache"
    support_legacy.mkdir(parents=True)
    support_marker = support_legacy / "keep.txt"
    support_marker.write_text("existing support-only cycle SE cache must be preserved")
    with pytest.raises(ValueError, match="basis_backend"):
        data.load_benchmark(
            tmp_path,
            "zinc12k",
            allow_download=False,
            splits=("train",),
            basis_backend="thin_q",
        )
    raw, raw_protocol = data.load_benchmark(
        tmp_path,
        "zinc12k",
        allow_download=False,
        splits=("train",),
        basis_backend="dfs_fundamental",
    )
    assert raw_protocol["basis_backend"] == "dfs_fundamental"
    assert "no QR/SVD/projector" in raw_protocol["basis_runtime"]
    assert "selected DFS forest can affect" in raw_protocol["basis_coordinates"]
    assert marker.read_text() == "existing results must be preserved"
    assert support_marker.read_text() == "existing support-only cycle SE cache must be preserved"
    raw_cycle = raw["train"][0].cycle_basis
    assert raw_cycle.layout == torch.sparse_coo
    assert set(raw_cycle.values().tolist()) <= {-1.0, 1.0}


@pytest.mark.parametrize("missing", ["payload", "metadata"])
def test_incomplete_cache_fails_closed(tmp_path, monkeypatch, missing):
    _install_official_fixture(monkeypatch)
    _, protocol = data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    cache, meta = _cache_paths(tmp_path, protocol)
    (cache if missing == "payload" else meta).unlink()
    with pytest.raises(CacheIncompleteError):
        data.load_benchmark(tmp_path, "zinc12k", allow_download=False)


@pytest.mark.parametrize(
    "damage",
    [
        "metadata",
        "checksum",
        "schema",
        "basis",
        "target",
        "count",
        "lengths",
        "membership_counts",
        "dense",
        "structural",
        "position_values",
        "position_indices",
    ],
)
def test_damaged_or_numeric_invalid_cache_is_rejected_not_rebuilt(tmp_path, monkeypatch, damage):
    _install_official_fixture(monkeypatch)
    _, protocol = data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    cache, meta = _cache_paths(tmp_path, protocol)
    if damage == "metadata":
        meta.write_text("broken json")
    elif damage == "checksum":
        cache.write_bytes(b"invalid archive")
    else:
        rows = torch.load(cache, weights_only=True)
        if damage == "schema":
            rows[0]["cycle_set"] = rows[0].pop("cycle_basis")
        elif damage == "basis":
            rows[0]["cycle_basis"].zero_()
        elif damage == "target":
            rows[0]["y"] += 1
        elif damage == "lengths":
            rows[0]["cycle_lengths"] += 1
        elif damage == "membership_counts":
            rows[0]["edge_cycle_counts"] += 1
        elif damage == "dense":
            rows[0]["cycle_basis"] = rows[0]["cycle_basis"].to_dense()
        elif damage == "structural":
            rows[0]["edge_cycle_features"] += 1
        elif damage == "position_values":
            rows[0]["cycle_position_values"].zero_()
        elif damage == "position_indices":
            rows[0]["cycle_position_indices"].zero_()
        else:
            rows.pop()
        _rewrite_cache(cache, meta, rows)
    before = cache.read_bytes()
    with pytest.raises(CacheCorruptError):
        data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    assert cache.read_bytes() == before


def test_changed_official_inputs_do_not_reuse_cached_basis_graphs(tmp_path, monkeypatch):
    official = _install_official_fixture(monkeypatch)
    data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    official["train"][0].y += 1
    with pytest.raises(CacheWrongRequestError):
        data.load_benchmark(tmp_path, "zinc12k", allow_download=False)


def test_no_v1_summary_or_preparer_is_used(monkeypatch):
    from research.cycle_pe import benchmark_data

    def forbidden(*args, **kwargs):
        raise AssertionError("v2 must not compute v1 cycle-set statistics")

    monkeypatch.setattr(benchmark_data, "prepare_graph", forbidden)
    monkeypatch.setattr(benchmark_data, "cycle_statistics", forbidden)
    graph = _graph()
    assert graph.cycle_basis.shape == (4, 1)


def test_random_sparse_dfs_certifies_entire_nullspace_with_oriented_disconnected_graphs():
    rng = np.random.default_rng(37)
    for nodes in range(1, 17):
        for probability in (0.08, 0.25, 0.6):
            pairs = [
                (u, v)
                for u in range(nodes)
                for v in range(u + 1, nodes)
                if rng.random() < probability
            ]
            edges = _edges(pairs)
            flips = rng.random(len(pairs)) < 0.5
            edges[:, flips] = edges[::-1, flips]
            incidence, rank = basis.incidence_and_cycle_rank(nodes, edges)
            values = basis.build_cycle_basis(nodes, edges)
            assert incidence.nnz == 2 * len(pairs)
            assert values.shape == (len(pairs), rank)
            assert (incidence.T @ values).nnz == 0
            if rank:
                assert np.linalg.matrix_rank(values.toarray()) == rank


def test_actual_process_parallel_preparation_and_cached_validation_preserve_full_order():
    # Explicit synthetic process-pool smoke, not official training.
    sources = [
        _official(),
        _official(2, []),
        _official(3, [(0, 1), (1, 2), (0, 2)]),
        _official(1, []),
    ]
    expected = [data.prepare_graph(source, dataset="zinc12k") for source in sources]
    actual = data._prepare_split(
        sources,
        dataset="zinc12k",
        split="smoke",
        basis_backend="dfs_fundamental",
        workers=2,
    )
    rows = [{field.name: getattr(graph, field.name) for field in fields(graph)} for graph in actual]
    restored = data._validate_cached_graphs(rows, sources, "zinc12k", workers=2)
    for first, parallel, cached in zip(expected, actual, restored, strict=True):
        for field in fields(first):
            torch.testing.assert_close(getattr(first, field.name), getattr(parallel, field.name))
            torch.testing.assert_close(getattr(first, field.name), getattr(cached, field.name))
        _assert_graph_has_private_storage(parallel)
        _assert_graph_has_private_storage(cached)
    for source in sources:
        assert all(
            not getattr(source, name).is_shared() for name in ("x", "edge_index", "edge_attr", "y")
        )


def _assert_graph_has_private_storage(graph):
    for field in fields(graph):
        value = getattr(graph, field.name)
        storage_tensors = (
            (value._indices(), value._values()) if value.layout == torch.sparse_coo else (value,)
        )
        assert all(not part.is_shared() for part in storage_tensors), field.name


def _assert_tensor_free_ipc(value):
    assert not isinstance(value, torch.Tensor)
    if is_dataclass(value):
        for field in fields(value):
            _assert_tensor_free_ipc(getattr(value, field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_tensor_free_ipc(key)
            _assert_tensor_free_ipc(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_tensor_free_ipc(item)
    elif isinstance(value, np.ndarray):
        assert value.dtype == np.uint8
        assert value.flags.owndata and value.flags.c_contiguous


@pytest.mark.parametrize(
    "dtype", [torch.long, torch.float32, torch.float64, torch.bfloat16, torch.bool, torch.complex64]
)
@pytest.mark.parametrize("empty", [False, True])
def test_owned_wire_preserves_dense_dtype_shape_and_values_without_tensor_reducers(dtype, empty):
    values = torch.arange(80).reshape(8, 10).to(dtype)
    view = values[::2, :0] if empty else values[::2, 2:5]
    encoded = data._encode_graph_ipc(view)
    _assert_tensor_free_ipc(encoded)
    assert encoded.buffer.nbytes == view.numel() * view.element_size()
    actual = data._decode_graph_ipc(encoded)
    assert actual.dtype == dtype and actual.shape == view.shape
    assert not actual.is_shared()
    torch.testing.assert_close(actual, view)
    if not empty:
        values.fill_(0)
        assert actual.any()  # The payload must not alias the source's backing allocation.


@pytest.mark.parametrize("cached", [False, True])
def test_preparation_and_validation_ipc_never_serialize_torch_shared_storage(monkeypatch, cached):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor storage crossed the process queue")

    monkeypatch.setattr(torch.UntypedStorage, "_share_fd_cpu_", forbidden)
    monkeypatch.setattr(torch.UntypedStorage, "_share_filename_cpu_", forbidden)
    source = _official()
    expected = data.prepare_graph(source, dataset="zinc12k")
    if cached:
        row = {field.name: getattr(expected, field.name) for field in fields(expected)}
        function, task = data._validate_cached_graph_task, (row, source, "zinc12k")
    else:
        function, task = data._prepare_task, (source, "zinc12k", "dfs_fundamental")
    request = data._encode_graph_ipc([task])
    _assert_tensor_free_ipc(request)
    # Exercise exactly the pickler used by multiprocessing, in both directions.
    incoming = ForkingPickler.loads(ForkingPickler.dumps(request))
    response = data._apply_graph_chunk(function, incoming)
    _assert_tensor_free_ipc(response)
    outgoing = ForkingPickler.loads(ForkingPickler.dumps(response))
    actual = data._decode_graph_ipc(outgoing[0])
    for field in fields(expected):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name))
    _assert_graph_has_private_storage(actual)


def test_owned_wire_copies_only_logical_official_views_not_entire_dataset_storage():
    source = _official()
    backing = torch.arange(200_000).reshape(-1, 1)
    source.x = backing[100:104]
    encoded = data._encode_graph_ipc(source)
    _assert_tensor_free_ipc(encoded)
    assert encoded.attributes["x"].buffer.nbytes == 4 * source.x.element_size()
    restored = data._decode_graph_ipc(encoded)
    assert restored.x.untyped_storage().nbytes() == 4 * source.x.element_size()
    torch.testing.assert_close(restored.x, source.x)


@pytest.mark.parametrize("scalar", [False, True])
def test_owned_wire_handles_scalars_and_zero_stride_singleton_views(scalar):
    value = torch.tensor(3.25) if scalar else torch.tensor([3.25]).as_strided((1,), (0,))
    actual = data._decode_graph_ipc(data._encode_graph_ipc(value))
    torch.testing.assert_close(actual, value)
    assert actual.shape == value.shape and not actual.is_shared()


def test_graph_wire_rejects_tensor_keys_and_non_cpu_tensors_before_process_submission():
    with pytest.raises(TypeError, match="string field names"):
        data._encode_graph_ipc({torch.tensor(1): "invalid field"})
    with pytest.raises(ValueError, match="CPU tensors; no device fallback"):
        data._encode_graph_ipc(torch.empty(3, device="meta"))
    with pytest.raises(TypeError, match="unsupported graph preparation IPC value"):
        data._encode_graph_ipc(object())


def test_empty_wire_cannot_replace_missing_nonempty_data():
    wire = data._DenseTensorWire(np.empty(0, dtype=np.uint8), torch.float32, (2,))
    with pytest.raises(ValueError, match="cannot encode a nonempty tensor"):
        data._decode_graph_ipc(wire)


def _linux_ipc_snapshot():
    if not sys.platform.startswith("linux"):
        return {"available": False, "reason": "Linux /proc is unavailable on this test host"}
    return {
        "available": True,
        "maps": len(Path("/proc/self/maps").read_text().splitlines()),
        "open_fds": len(list(Path("/proc/self/fd").iterdir())),
        "max_map_count": int(Path("/proc/sys/vm/max_map_count").read_text()),
    }


@pytest.mark.skipif(
    "CYCLE_V2_IPC_STRESS_GRAPHS" not in os.environ,
    reason="opt-in 10k+ synthetic IPC stress, separate from default unit/smoke tests",
)
def test_debug_large_synthetic_preparation_and_cache_ipc_have_bounded_os_handles():
    """No real data/training: retain 10k+ full cycle Graphs across both IPC paths.

    Run this test with CYCLE_V2_IPC_STRESS_GRAPHS=10000. On Linux it also checks
    process mmap/FD growth while all prepared and validated graphs remain alive.
    """
    count = int(os.environ["CYCLE_V2_IPC_STRESS_GRAPHS"])
    assert count >= 10_000, "this explicit regression stress must cover at least 10k graphs"
    started = time.perf_counter()
    before = _linux_ipc_snapshot()
    sources = [_official() for _ in range(count)]
    for index, source in enumerate(sources):
        source.y.fill_(index)
    prepared = data._prepare_split(
        sources,
        dataset="zinc12k",
        split="debug_ipc_stress",
        basis_backend="dfs_fundamental",
        workers=2,
    )
    after_preparation = _linux_ipc_snapshot()
    rows = [
        {field.name: getattr(graph, field.name) for field in fields(graph)} for graph in prepared
    ]
    restored = data._validate_cached_graphs(rows, sources, "zinc12k", workers=2)
    assert len(prepared) == len(restored) == count
    for index, (first, second) in enumerate(zip(prepared, restored, strict=True)):
        assert first.y.item() == second.y.item() == index
        _assert_graph_has_private_storage(first)
        _assert_graph_has_private_storage(second)
        for field in fields(first):
            torch.testing.assert_close(getattr(first, field.name), getattr(second, field.name))
    after_validation = _linux_ipc_snapshot()
    if before["available"]:
        for snapshot in (after_preparation, after_validation):
            assert snapshot["maps"] - before["maps"] < 2048
            assert snapshot["open_fds"] - before["open_fds"] < 256
    print(
        json.dumps(
            {
                "kind": "debug_synthetic_cpu_ipc_stress",
                "graphs": count,
                "workers": 2,
                "seconds": time.perf_counter() - started,
                "before": before,
                "after_preparation": after_preparation,
                "after_validation": after_validation,
                "real_data_or_gpu_training": False,
            },
            sort_keys=True,
        )
    )


def test_parallel_submission_bounds_only_inflight_buffer_not_total_graph_count():
    class Executor:
        pending = 0
        peak_pending = 0
        submitted = 0

        def submit(self, function, *args):
            self.pending += 1
            self.peak_pending = max(self.peak_pending, self.pending)
            self.submitted += 1
            owner = self

            class Future:
                def result(self):
                    owner.pending -= 1
                    return function(*args)

            return Future()

    executor = Executor()
    result = list(
        data._ordered_parallel_graphs(
            executor,
            lambda value: value * 3,
            iter(range(101)),
            workers=3,
            chunksize=4,
        )
    )
    assert result == [value * 3 for value in range(101)]
    assert executor.peak_pending == 6
    assert executor.pending == 0
    assert executor.submitted == 26


@pytest.mark.parametrize("workers,chunksize", [(0, 1), (1, 0)])
def test_parallel_buffer_never_silently_discards_input_for_invalid_execution_settings(
    workers, chunksize
):
    with pytest.raises(ValueError, match="positive"):
        list(
            data._ordered_parallel_graphs(
                None, lambda value: value, [1], workers=workers, chunksize=chunksize
            )
        )
