"""Citation parser fixtures and complete topology/cache guards, without training."""

from __future__ import annotations

import copy
import json
import pickle
from dataclasses import fields

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from research.wedge_propagation.classification import data


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _graph():
    x = np.array([[2.0, 2.0], [0, 0], [1, 0], [0, 2], [1, 1], [3, 1]])
    y = np.arange(6) % 2
    masks = [np.isin(np.arange(6), indices) for indices in ([0, 1], [2, 3], [4, 5])]
    edges = np.array([[0, 1, 1, 0, 2, 2, 3, 3], [1, 0, 2, 2, 0, 3, 2, 3]])
    return data.graph_from_arrays("DEBUG-unit", x, y, edges, *masks)


def _raw_fixture(path, name="CiteSeer"):
    """Synthetic raw serialization exercises the authors' real format and gaps."""
    n = 8 if name == "CiteSeer" else 7
    expected = {"nodes": n, "features": 2, "classes": 2, "train": 1, "validation": 2, "test": 3}
    test = [7, 4, 6] if name == "CiteSeer" else [6, 4, 5]
    graph = {node: [] for node in range(n)}
    graph[0] = [1, 1, 0]
    graph[1] = [0, 2]
    graph[2] = [1, 4]
    graph[4] = [2]
    objects = {
        "x": sp.csr_matrix([[2.0, 0.0]]),
        "y": np.array([[1, 0]]),
        "allx": sp.csr_matrix([[2.0, 0], [0, 3.0], [1.0, 1.0], [0, 0]]),
        "ally": np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),
        "tx": sp.csr_matrix([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]),
        "ty": np.array([[1, 0], [0, 1], [1, 0]]),
        "graph": graph,
    }
    path.mkdir(parents=True)
    for part, value in objects.items():
        with (path / f"ind.{name.lower()}.{part}").open("wb") as stream:
            pickle.dump(value, stream, protocol=2)
    (path / f"ind.{name.lower()}.test.index").write_text("\n".join(map(str, test)) + "\n")
    return expected, test


@pytest.mark.parametrize("name", ["Cora", "CiteSeer"])
def test_real_raw_format_reordering_masks_and_citeseer_gap(tmp_path, name):
    expected, indices = _raw_fixture(tmp_path / "raw", name)
    x, y, edges, train, val, test, metadata = data.decode_planetoid(
        tmp_path / "raw", name, expected
    )
    assert x.shape == (expected["nodes"], 2)
    assert np.array_equal(x[indices], [[3, 1], [1, 3], [2, 2]])
    assert np.array_equal(y[indices], [0, 1, 0])
    assert np.flatnonzero(train).tolist() == [0]
    assert np.flatnonzero(val).tolist() == [1, 2]
    assert np.flatnonzero(test).tolist() == sorted(indices)
    graph = data.graph_from_arrays("DEBUG-raw", x, y, edges, train, val, test, metadata=metadata)
    assert graph.edges.tolist() == [[0, 1, 2], [1, 2, 4]]
    if name == "CiteSeer":
        assert np.array_equal(x[5], [0, 0]) and y[5] == -1 and not test[5]
        assert metadata["citeseer_missing_test_indices"] == [5]
        assert graph.num_nodes == 8


def test_raw_duplicate_public_test_index_fails(tmp_path):
    expected, _ = _raw_fixture(tmp_path / "raw")
    (tmp_path / "raw" / "ind.citeseer.test.index").write_text("4\n4\n6\n")
    with pytest.raises(ValueError, match="public test indices"):
        data.decode_planetoid(tmp_path / "raw", "CiteSeer", expected)


def test_canonical_physical_graph_removes_loops_and_reciprocal_duplicates():
    graph = _graph()
    assert graph.edges.tolist() == [[0, 0, 1, 2], [1, 2, 2, 3]]
    assert graph.num_nodes == 6 and graph.num_features == 2 and graph.num_classes == 2
    assert graph.degree.tolist() == [2, 2, 3, 1, 0, 0]
    assert graph.metadata["isolates"] == 2
    assert torch.equal(graph.x[1], torch.zeros(2))
    assert torch.allclose(graph.x[0], torch.tensor([0.5, 0.5]))
    assert graph.sd[4] == graph.sq[4] == 0


def test_all_wedges_include_triangles_and_match_dense_diagonal():
    graph = _graph()
    neighbors = {i: [] for i in range(graph.num_nodes)}
    for i, j in graph.edges.T.tolist():
        neighbors[i].append(j)
        neighbors[j].append(i)
    expected = [
        (a, center, b)
        for center, members in neighbors.items()
        for a in sorted(members)
        for b in sorted(members)
        if a < b
    ]
    assert graph.paths.T.tolist() == [list(row) for row in expected]
    assert [1, 0, 2] in graph.paths.T.tolist()  # Endpoint edge exists: retain triangle path.
    matrix = torch.zeros(len(expected), graph.num_nodes)
    for row, (i, j, k) in enumerate(expected):
        matrix[row, i], matrix[row, j], matrix[row, k] = 1, -2, 1
    assert torch.equal(graph.qdiag, (matrix.T @ matrix).diag())
    assert torch.equal(graph.paths, torch.from_numpy(data.build_paths(graph.edges.numpy(), 6, 4)))


def test_gcn_contains_both_directions_and_exactly_one_loop_per_node():
    graph = _graph()
    dense = torch.zeros(graph.num_nodes, graph.num_nodes)
    dense[graph.edges[0], graph.edges[1]] = 1
    dense[graph.edges[1], graph.edges[0]] = 1
    dense += torch.eye(graph.num_nodes)
    normalization = dense.sum(1).rsqrt()
    expected = normalization[:, None] * dense * normalization[None, :]
    actual = torch.zeros_like(dense)
    actual[graph.gcn_edges[0], graph.gcn_edges[1]] = graph.gcn_weight
    assert torch.allclose(actual, expected, atol=1e-7)
    assert int((graph.gcn_edges[0] == graph.gcn_edges[1]).sum()) == graph.num_nodes


def test_graph_transfer_preserves_indices_and_source_storage():
    original = _graph()
    moved = original.to("cpu", torch.float64)
    assert moved.x.dtype == moved.sq.dtype == torch.float64
    assert moved.edges.dtype == moved.y.dtype == torch.long
    assert moved.train_mask.dtype == torch.bool
    assert original.x.dtype == torch.float32


def test_npz_roundtrip_checksum_and_no_overwrite(tmp_path):
    graph = _graph()
    path = tmp_path / "graph.npz"
    saved = data.save_graph(graph, path)
    reloaded = data.load_graph(path, saved["sha256"])
    for field in fields(graph):
        value = getattr(graph, field.name)
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(reloaded, field.name))
        else:
            assert value == getattr(reloaded, field.name)
    with pytest.raises(FileExistsError):
        data.save_graph(graph, path)
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        data.load_graph(path, saved["sha256"])


@pytest.mark.parametrize("field", ["paths", "qdiag", "gcn_weight", "train_mask"])
def test_processed_forged_arrays_fail_internal_equation_guards(tmp_path, field):
    path = tmp_path / "graph.npz"
    data.save_graph(_graph(), path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if field == "train_mask":
        arrays[field][2] = True
    else:
        arrays[field].flat[0] += 1
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError):
        data.load_graph(path)


def test_raw_corruption_is_rejected_before_pickle_is_read(tmp_path):
    path = tmp_path / "Cora" / "raw"
    path.mkdir(parents=True)
    original = b"not-an-official-pickle"
    (path / "ind.cora.x").write_bytes(original)
    with pytest.raises(ValueError, match="existing raw checksum mismatch"):
        data.ensure_raw("Cora", tmp_path, download=False)
    assert (path / "ind.cora.x").read_bytes() == original


def test_offline_missing_raw_file_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match="offline mode"):
        data.ensure_raw("PubMed", tmp_path, download=False)


@pytest.mark.parametrize("redirect", ["Cora/raw", "Cora/raw/ind.cora.x"])
def test_linked_raw_destination_cannot_escape_declared_data_root(tmp_path, monkeypatch, redirect):
    root = tmp_path / "declared"
    redirected = root / redirect
    outside = tmp_path / "outside"
    resolve = data.Path.resolve

    def linked_resolve(path, *args, **kwargs):
        return outside if path == redirected else resolve(path, *args, **kwargs)

    # Simulate the resolved link to keep this security test portable on Windows.
    monkeypatch.setattr(data.Path, "resolve", linked_resolve)
    with pytest.raises(ValueError, match="escapes declared data root"):
        data.ensure_raw("Cora", root, download=False)
    assert not root.exists() and not outside.exists()


def test_linked_processed_destination_fails_before_raw_preparation(tmp_path, monkeypatch):
    config = json.loads((data.Path(data.__file__).parent / "config_full.json").read_text())
    root = tmp_path / "declared"
    redirected = root / "PubMed" / "processed" / f"{data.SCHEMA}.npz"
    outside = tmp_path / "outside"
    resolve = data.Path.resolve

    def linked_resolve(path, *args, **kwargs):
        return outside if path == redirected else resolve(path, *args, **kwargs)

    def forbidden_raw(*args, **kwargs):
        pytest.fail("processed path containment must run before raw preparation")

    monkeypatch.setattr(data.Path, "resolve", linked_resolve)
    monkeypatch.setattr(data, "ensure_raw", forbidden_raw)
    with pytest.raises(ValueError, match="escapes declared data root"):
        data.prepare_datasets(config, root, tmp_path / "output", download=False)
    assert not root.exists() and not outside.exists()


def test_debug_data_is_separate_complete_and_explicit(tmp_path):
    config = {
        "profile": "debug",
        "data_source": "debug_fixture",
        "data": {"datasets": [f"DEBUG-{name}" for name in data.EXPECTED]},
        "debug_fixture": {
            "nodes": [24, 30, 36],
            "features": 12,
            "classes": 3,
            "train_per_class": 3,
            "validation_nodes": 6,
            "master_seed": 20261003,
        },
    }
    graphs, manifest = data.prepare_datasets(
        config, tmp_path / "data", tmp_path / "output", workers="1"
    )
    assert list(graphs) == config["data"]["datasets"]
    assert [graph.num_nodes for graph in graphs.values()] == [24, 30, 36]
    assert manifest["actual_citation_data"] is False and manifest["all_nodes_edges_paths"]
    assert all(
        graph.metadata["train"] == 9
        and graph.metadata["validation"] == 6
        and graph.metadata["actual_citation_data"] is False
        for graph in graphs.values()
    )
    with pytest.raises(FileExistsError):
        data.prepare_datasets(config, tmp_path / "data", tmp_path / "output", workers="1")
    invalid = copy.deepcopy(config)
    invalid["profile"] = "full"
    with pytest.raises(ValueError, match="separate DEBUG"):
        data.prepare_datasets(invalid, tmp_path / "data", tmp_path / "other", workers="1")


def test_full_scale_contract_cannot_be_replaced_with_subset(tmp_path):
    contract = json.loads((data.Path(data.__file__).parent / "design_contract.json").read_text())
    contract.update(profile="full", data_source="planetoid_public")
    contract["data"]["expected_shapes"]["PubMed"]["nodes"] = 100
    with pytest.raises(ValueError, match="data scale contract changed"):
        data.prepare_datasets(contract, tmp_path / "data", tmp_path / "output", download=False)


def test_worker_calibration_processes_complete_graph_for_every_candidate():
    edges = _graph().edges.numpy()
    selected, trials = data._choose_workers([(edges, 6)], "auto")
    assert selected in [row["workers"] for row in trials]
    assert all(row["paths"] == 5 and row["seconds"] > 0 for row in trials)
