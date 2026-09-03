"""Small algebra/schema fixtures only; never run training or download datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
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


def _graph(*args, basis_backend="thin_q", **kwargs):
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
    assert values.flags.c_contiguous
    np.testing.assert_allclose(incidence.T @ values, 0, atol=2e-7)
    np.testing.assert_allclose(values.T @ values, np.eye(rank), atol=3e-7)
    basis.validate_cycle_basis(nodes, edge_index, values)


def test_uses_sparse_fundamental_cycles_and_thin_qr_without_svd(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("spectral/rank decomposition is forbidden")

    for name in ("svd", "eig", "eigh", "matrix_rank"):
        monkeypatch.setattr(basis.np.linalg, name, forbidden)
    pairs = [(u, v) for u in range(8) for v in range(u + 1, 8)]
    sparse_values = basis.sparse_left_nullspace_basis(8, _edges(pairs))
    values = basis.left_nullspace_basis(8, _edges(pairs))
    assert sparse_values.shape == (28, 21) and sparse_values.nnz < np.prod(sparse_values.shape)
    assert values.shape == (28, 21)
    np.testing.assert_allclose(values.T @ values, np.eye(21), atol=2e-6)


def test_dfs_fundamental_backend_returns_raw_signed_cycles_with_same_projector():
    pairs = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edge_index = _edges(pairs)
    raw_sparse = basis.dfs_fundamental_cycle_basis(5, edge_index)
    raw = basis.build_cycle_basis(5, edge_index, backend="dfs_fundamental")
    q = basis.build_cycle_basis(5, edge_index, backend="thin_q")
    assert raw_sparse.shape == raw.shape == q.shape == (10, 6)
    np.testing.assert_array_equal(raw, raw_sparse.toarray())
    assert set(np.unique(raw)) <= {-1.0, 0.0, 1.0}
    assert not np.allclose(raw.T @ raw, np.eye(6))
    raw_q, _ = np.linalg.qr(raw.astype(np.float64), mode="reduced")
    np.testing.assert_allclose(raw_q @ raw_q.T, q @ q.T, atol=2e-6)
    basis.validate_cycle_basis(5, edge_index, raw_sparse)


def test_unknown_basis_backend_fails_closed():
    with pytest.raises(ValueError, match="backend"):
        basis.build_cycle_basis(3, _edges([(0, 1)]), backend="unknown")
    with pytest.raises(ValueError, match="basis_backend"):
        data.prepare_graph(_official(), basis_backend="unknown")


def test_generic_nonorthogonal_basis_is_valid_but_rank_deficiency_is_not():
    edge_index = _edges([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    q = basis.left_nullspace_basis(4, edge_index)
    changed = q @ np.asarray([[2.0, 0.5], [-1.0, 3.0]])
    basis.validate_cycle_basis(4, edge_index, changed)
    changed[:, 1] = changed[:, 0]
    with pytest.raises(ValueError, match="full column rank"):
        basis.validate_cycle_basis(4, edge_index, changed)


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
        changed_q @ changed_q.T,
        signs[:, None] * (q @ q.T) * signs[None, :],
        atol=2e-6,
    )


@pytest.mark.parametrize("kind", ["nonfinite", "wrong_width", "outside_nullspace", "rank", "half"])
def test_basis_validation_rejects_incomplete_or_corrupt_coordinates(kind):
    edge_index = _edges([(0, 1), (0, 2), (1, 2)])
    values = basis.left_nullspace_basis(3, edge_index)
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
    original = basis.left_nullspace_basis(4, pairs)
    permutation = np.array([3, 0, 4, 1, 2])
    transformed = basis.left_nullspace_basis(4, pairs[:, permutation])
    # Different spanning forests may rotate coordinates; the spaces must agree.
    transported = original[permutation]
    alignment = transported.T @ transformed
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
        "cycle_basis_is_orthonormal",
    }
    torch.testing.assert_close(graph.x, source.x)
    torch.testing.assert_close(graph.y, source.y)
    torch.testing.assert_close(graph.edge_index, torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]]))
    torch.testing.assert_close(graph.edge_attr[:, 0], torch.tensor([3, 2, 1, 1]))
    assert graph.cycle_basis.shape == (4, 1)
    assert graph.cycle_basis.dtype == torch.float32
    assert graph.cycle_basis_is_orthonormal.item() is True
    incidence, _ = basis.incidence_and_cycle_rank(4, graph.edge_index.numpy())
    np.testing.assert_allclose(incidence.T @ graph.cycle_basis.numpy(), 0, atol=1e-7)


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


def test_ragged_batch_keeps_whole_matrices_and_never_mixes_columns():
    graphs = [
        _graph(2, [(0, 1)]),
        _graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]),
        _graph(1, []),
        _graph(),
    ]
    batch = data.collate(graphs)
    assert isinstance(batch.cycle_bases, tuple)
    assert [matrix.shape for matrix in batch.cycle_bases] == [(1, 0), (5, 2), (0, 0), (4, 1)]
    torch.testing.assert_close(batch.ptr, torch.tensor([0, 2, 6, 7, 11]))
    torch.testing.assert_close(batch.edge_ptr, torch.tensor([0, 1, 6, 6, 10]))
    assert batch.cycle_basis_shapes == ((1, 0), (5, 2), (0, 0), (4, 1))
    assert batch.cycle_basis_is_orthonormal == (True, True, True, True)
    assert batch.packed_cycle_basis.is_contiguous()
    assert batch.packed_cycle_basis.numel() == 14
    for index, graph in enumerate(graphs):
        torch.testing.assert_close(batch.cycle_bases[index], graph.cycle_basis)
        assert batch.cycle_bases[index].untyped_storage().data_ptr() == (
            batch.packed_cycle_basis.untyped_storage().data_ptr()
        )
        start, end = batch.edge_ptr[index : index + 2]
        torch.testing.assert_close(
            batch.edge_index[:, start:end] - batch.ptr[index], graph.edge_index
        )
    moved = batch.to(torch.device("cpu"))
    assert isinstance(moved.cycle_bases, tuple)
    for before, after in zip(batch.cycle_bases, moved.cycle_bases, strict=True):
        torch.testing.assert_close(before, after)


def test_ragged_pin_memory_uses_one_packed_basis_tensor(monkeypatch):
    batch = data.collate([_graph(), _graph(1, [])])
    called = []

    def pin(tensor):
        called.append(id(tensor))
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    pinned = batch.pin_memory()
    assert isinstance(pinned.cycle_bases, tuple)
    assert id(batch.packed_cycle_basis) in called
    assert all(id(matrix) not in called for matrix in batch.cycle_bases)
    assert len(called) == sum(
        isinstance(getattr(batch, field.name), torch.Tensor) for field in fields(batch)
    )


def test_batch_to_transfers_packed_basis_once_not_once_per_graph(monkeypatch):
    batch = data.collate([_graph(), _graph(), _graph(1, [])])
    transferred = []
    original = torch.Tensor.to

    def moved(tensor, *args, **kwargs):
        transferred.append(id(tensor))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", moved)
    result = batch.to("cpu")
    assert transferred.count(id(batch.packed_cycle_basis)) == 1
    assert all(id(matrix) not in transferred for matrix in batch.cycle_bases)
    assert result.cycle_basis_shapes == batch.cycle_basis_shapes


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
    assert (
        protocol["preparation"]["representation"]
        == "coordinate_free_cycle_projector_from_cached_thin_q"
    )
    assert data.CACHE_NAMESPACE == "cycle_pe_v2_projector_kernel_benchmark"
    assert set(protocol["preparation"]["implementation_sha256"]) == {
        "v2/basis.py",
        "v2/data.py",
        "official_adapter",
    }
    assert not (tmp_path / "cycle_pe_benchmark").exists()

    def no_svd(*args, **kwargs):
        raise AssertionError("cached thin-Q matrices must not be recomputed")

    monkeypatch.setattr(data, "build_cycle_basis", no_svd)
    second, restored = data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    assert restored == protocol
    for split in data.SPLITS:
        for original, cached in zip(first[split], second[split], strict=True):
            for field in fields(original):
                torch.testing.assert_close(
                    getattr(original, field.name), getattr(cached, field.name)
                )


def test_basis_backends_use_separate_caches_and_preserve_representation_metadata(
    tmp_path, monkeypatch
):
    _install_official_fixture(monkeypatch)
    thin, thin_protocol = data.load_benchmark(
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
    assert thin_protocol["cache_directory"] != raw_protocol["cache_directory"]
    assert thin_protocol["basis_backend"] == "thin_q"
    assert raw_protocol["basis_backend"] == "dfs_fundamental"
    assert "not an end-to-end linear-time speedup" in raw_protocol["basis_runtime"]
    assert all(graph.cycle_basis_is_orthonormal.item() for graph in thin["train"])
    assert not any(graph.cycle_basis_is_orthonormal.item() for graph in raw["train"])
    raw_cycle = raw["train"][0].cycle_basis
    assert set(raw_cycle.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert not torch.allclose(
        raw_cycle.T @ raw_cycle,
        torch.eye(raw_cycle.shape[1], dtype=raw_cycle.dtype),
    )


@pytest.mark.parametrize("missing", ["payload", "metadata"])
def test_incomplete_cache_fails_closed(tmp_path, monkeypatch, missing):
    _install_official_fixture(monkeypatch)
    _, protocol = data.load_benchmark(tmp_path, "zinc12k", allow_download=False)
    cache, meta = _cache_paths(tmp_path, protocol)
    (cache if missing == "payload" else meta).unlink()
    with pytest.raises(CacheIncompleteError):
        data.load_benchmark(tmp_path, "zinc12k", allow_download=False)


@pytest.mark.parametrize("damage", ["metadata", "checksum", "schema", "basis", "target", "count"])
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
