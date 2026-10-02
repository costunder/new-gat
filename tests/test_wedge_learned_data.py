"""CPU data/target checks; these tests never train a production model."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from research.wedge_propagation.data import feature_content_hash, graph_content_hash
from research.wedge_propagation.learned import data

_DIRECTORY = Path(__file__).resolve().parents[1] / "research/wedge_propagation/learned"


def _config(profile="debug"):
    return json.loads((_DIRECTORY / f"config_{profile}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def single_intraop_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def full_cases():
    return data.prepare_cases(_config("full"), workers=2)


@pytest.fixture(scope="module")
def debug_cases():
    return data.prepare_cases(_config(), workers=2)


def test_full_contract_has_exact_531_graphs_and_8496_realizations(full_cases):
    assert len(full_cases) == 531
    assert sum(case.features.shape[1] for case in full_cases) == 8496
    assert Counter(case.split for case in full_cases) == {
        "train": 240, "validation": 60, "id": 120, "size_ood": 90,
        "family_ood": 12, "family_size_ood": 9,
    }
    assert all(case.features.shape == (case.num_nodes, 16) for case in full_cases)
    for case in full_cases:
        expected_sizes = {60, 80, 100} if "size_ood" in case.split else {20, 30, 40, 50}
        assert case.num_nodes in expected_sizes
        expected_families = {"cycle", "star", "grid"} if case.split.startswith("family_") \
            else {"er", "tree", "tree_chord"}
        assert case.family in expected_families


def test_debug_contract_is_explicit_and_separate(debug_cases):
    assert len(debug_cases) == 36
    assert sum(case.features.shape[1] for case in debug_cases) == 144
    assert Counter(case.split for case in debug_cases) == {
        "train": 12, "validation": 6, "id": 6, "size_ood": 3,
        "family_ood": 6, "family_size_ood": 3,
    }
    assert all(case.graph_id.startswith("rule-debug-") for case in debug_cases)
    assert len(data.make_specs(_config("full"))) == 531


def test_all_graph_and_feature_contents_are_disjoint_across_splits(full_cases):
    graph_hashes = [graph_content_hash(case) for case in full_cases]
    feature_hashes = [feature_content_hash(case) for case in full_cases]
    assert len(set(graph_hashes)) == len(full_cases)
    assert len(set(feature_hashes)) == len(full_cases)
    assert len({case.graph_id for case in full_cases}) == len(full_cases)
    assert all(case.graph_seed != case.feature_seed != case.pair_seed for case in full_cases)


def test_corresponding_tree_is_preserved_even_after_collision_repair(debug_cases):
    by_id = {case.graph_id: case for case in debug_cases}
    for case in debug_cases:
        if case.family != "tree_chord":
            continue
        tree = by_id[case.metadata["base_tree_graph_id"]]
        assert tree.split == case.split
        assert case.metadata["base_tree_seed"] == tree.graph_seed
        base_edges = {tuple(edge) for edge in tree.edges.T.tolist()}
        chord_edges = {tuple(edge) for edge in case.edges.T.tolist()}
        assert base_edges < chord_edges
        assert len(chord_edges - base_edges) == case.num_nodes // 4


def test_cpu_static_labels_have_valid_shapes_and_no_autograd(full_cases):
    for case in full_cases:
        n, realizations = case.features.shape
        paths = case.wedges.shape[1]
        assert case.edges.dtype == case.wedges.dtype == case.pair_edges.dtype == torch.long
        for tensor in (case.features, case.teacher_c, case.pair_coefficients,
                       case.lx, case.l2x, case.qx, case.target_path):
            assert tensor.device.type == "cpu" and tensor.dtype == torch.float64
            assert not tensor.requires_grad and torch.isfinite(tensor).all()
        assert case.teacher_c.shape == (paths, realizations)
        assert case.pair_edges.shape == case.pair_coefficients.shape == (2, paths)
        assert all(value.shape == (n, realizations) for value in
                   (case.lx, case.l2x, case.qx, case.target_path))
        if paths:
            torch.testing.assert_close(case.teacher_c.mean(0), torch.ones(realizations,
                                       dtype=torch.float64), rtol=1e-12, atol=1e-12)


def _dense(case):
    n, edges = case.num_nodes, case.edges.shape[1]
    b = torch.zeros((edges, n), dtype=torch.float64)
    b[torch.arange(edges), case.edges[0]] = -1
    b[torch.arange(edges), case.edges[1]] = 1
    a = torch.zeros((case.wedges.shape[1], n), dtype=torch.float64)
    rows = torch.arange(case.wedges.shape[1])
    a[rows, case.wedges[0]] = 1
    a[rows, case.wedges[1]] = -2
    a[rows, case.wedges[2]] = 1
    return b, a


def test_teacher_and_all_operator_targets_match_independent_dense_reference(debug_cases):
    cfg = _config()["teacher"]
    for case in debug_cases:
        b, a = _dense(case)
        lap = b.T @ b
        g1 = case.features[case.wedges[1]] - case.features[case.wedges[0]]
        g2 = case.features[case.wedges[2]] - case.features[case.wedges[1]]
        first, second = g1.numpy(), g2.numpy()
        cosine = first * second / (np.abs(first) * np.abs(second) + cfg["epsilon"])
        difference = np.abs(second - first) / (np.abs(first) + np.abs(second) + cfg["epsilon"])
        raw = np.exp(cfg["tau"] * np.tanh(cfg["theta1"] * cosine + cfg["theta2"] * difference))
        expected_c = torch.from_numpy(raw / raw.mean(axis=0, keepdims=True))
        torch.testing.assert_close(case.teacher_c, expected_c, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(case.lx, lap @ case.features, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(case.l2x, lap @ lap @ case.features, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(case.qx, a.T @ a @ case.features, rtol=1e-11, atol=1e-11)
        expected = a.T @ (expected_c * (a @ case.features))
        torch.testing.assert_close(case.target_path, expected, rtol=1e-11, atol=1e-11)


def test_random_pairs_are_unique_distinct_and_normalized_with_orientation_gauge(debug_cases):
    for case in debug_cases:
        b, _ = _dense(case)
        e1, e2 = case.pair_edges
        assert (e1 < e2).all()
        assert (e1 >= 0).all() and (e2 < case.edges.shape[1]).all()
        encoded = e1 * case.edges.shape[1] + e2
        assert torch.unique(encoded).numel() == case.wedges.shape[1]
        s1, s2 = case.pair_coefficients
        matrix = s2[:, None] * b[e2] - s1[:, None] * b[e1]
        torch.testing.assert_close(matrix.square().sum(1), torch.full(
            (case.wedges.shape[1],), 6.0, dtype=torch.float64), rtol=1e-12, atol=1e-12)
        signs = torch.ones(case.edges.shape[1], dtype=torch.float64)
        signs[::2] = -1
        flipped = b * signs[:, None]
        adjusted = case.pair_coefficients * signs[case.pair_edges]
        flipped_matrix = adjusted[1, :, None] * flipped[e2] \
            - adjusted[0, :, None] * flipped[e1]
        torch.testing.assert_close(matrix, flipped_matrix, rtol=0, atol=0)
        assert case.metadata["pair_trace"] == pytest.approx(6 * case.wedges.shape[1])
        assert 0 <= case.metadata["pair_shared_center_fraction"] <= 1


def test_parallel_order_and_worker_count_do_not_change_content(debug_cases):
    serial = data.prepare_cases(_config(), workers=1)
    assert [case.graph_id for case in serial] == [case.graph_id for case in debug_cases]
    for first, second in zip(serial, debug_cases, strict=True):
        assert first.graph_seed == second.graph_seed
        assert first.metadata == second.metadata
        for name in ("edges", "wedges", "features", "teacher_c", "pair_edges",
                     "pair_coefficients", "target_path"):
            assert torch.equal(getattr(first, name), getattr(second, name))


def test_duplicate_repair_is_recorded_and_preserves_corresponding_base_tree(monkeypatch):
    cfg = _config()
    specs = data.make_specs(cfg)
    train_tree = next(spec for spec in specs if spec.split == "train" and spec.family == "tree")
    train_chord = next(spec for spec in specs if spec.split == "train"
                       and spec.family == "tree_chord")
    val_tree = next(spec for spec in specs if spec.split == "validation" and spec.family == "tree")
    val_chord = next(spec for spec in specs if spec.split == "validation"
                     and spec.family == "tree_chord")
    duplicate = replace(val_tree, graph_spec=replace(
        val_tree.graph_spec, graph_seed=train_tree.graph_spec.graph_seed),
        original_graph_seed=train_tree.graph_spec.graph_seed)
    changed_chord = replace(val_chord, graph_spec=replace(
        val_chord.graph_spec, base_tree_seed=train_tree.graph_spec.graph_seed))
    selected = [train_tree, train_chord, duplicate, changed_chord]
    monkeypatch.setattr(data, "make_specs", lambda config: selected)
    cases = data.prepare_cases(cfg, workers=2)
    assert cases[2].metadata["resample_attempt"] >= 1
    assert cases[2].metadata["duplicate_collisions"][0]["other_graph_id"] == cases[0].graph_id
    assert cases[3].metadata["base_tree_seed"] == cases[2].graph_seed
    assert cases[2].feature_seed == duplicate.graph_spec.feature_seed
    tree_edges = {tuple(edge) for edge in cases[2].edges.T.tolist()}
    assert tree_edges < {tuple(edge) for edge in cases[3].edges.T.tolist()}
    assert len({graph_content_hash(case) for case in cases}) == 4


def test_exhausted_duplicate_retries_fail_without_dropping_cases(monkeypatch):
    cfg = _config()
    cfg["duplicate_max_attempts"] = 2
    specs = [spec for spec in data.make_specs(cfg) if spec.family == "tree"][:2]
    specs = [replace(spec, graph_spec=replace(spec.graph_spec, graph_seed=0),
                     original_graph_seed=0) for spec in specs]
    monkeypatch.setattr(data, "make_specs", lambda config: specs)
    monkeypatch.setattr(data, "_seed", lambda *args: 0)
    with pytest.raises(RuntimeError, match="2 recorded retries; no graph was dropped"):
        data.prepare_cases(cfg, workers=2)


def test_feature_and_pair_random_streams_do_not_change_graph():
    spec = data.make_specs(_config())[0]
    reference = data.prepare_case(spec, _config()["teacher"])
    changed_feature = data.prepare_case(replace(spec, graph_spec=replace(
        spec.graph_spec, feature_seed=spec.graph_spec.feature_seed + 1)), _config()["teacher"])
    changed_pair = data.prepare_case(replace(spec, pair_seed=spec.pair_seed + 1),
                                     _config()["teacher"])
    assert torch.equal(reference.edges, changed_feature.edges)
    assert torch.equal(reference.pair_edges, changed_feature.pair_edges)
    assert not torch.equal(reference.features, changed_feature.features)
    assert torch.equal(reference.features, changed_pair.features)
    assert torch.equal(reference.edges, changed_pair.edges)
    assert not torch.equal(reference.pair_edges, changed_pair.pair_edges)


def test_disjoint_batch_reconstructs_every_case_and_target(debug_cases):
    selected = [debug_cases[0], debug_cases[5], debug_cases[-1]]
    batch = data.pack_cases(selected, "cpu", torch.float64)
    assert batch.num_graphs == 3
    assert batch.graph_ids == tuple(case.graph_id for case in selected)
    assert torch.equal(batch.num_nodes, torch.tensor([case.num_nodes for case in selected]))
    node_offset = edge_offset = path_offset = 0
    for index, case in enumerate(selected):
        n, e, p = case.num_nodes, case.edges.shape[1], case.wedges.shape[1]
        torch.testing.assert_close(batch.x[node_offset:node_offset + n], case.features)
        assert torch.equal(batch.edges[:, edge_offset:edge_offset + e] - node_offset, case.edges)
        assert torch.equal(batch.wedges[:, path_offset:path_offset + p] - node_offset, case.wedges)
        assert torch.equal(batch.pair_edges[:, path_offset:path_offset + p] - edge_offset,
                           case.pair_edges)
        assert (batch.node_graph[node_offset:node_offset + n] == index).all()
        assert (batch.path_graph[path_offset:path_offset + p] == index).all()
        torch.testing.assert_close(batch.teacher_c[path_offset:path_offset + p], case.teacher_c)
        for target, attribute in (("L", "lx"), ("L2", "l2x"), ("path", "target_path")):
            torch.testing.assert_close(batch.targets[target][node_offset:node_offset + n],
                                       getattr(case, attribute))
        node_offset += n
        edge_offset += e
        path_offset += p
    assert batch.targets["L"].data_ptr() == batch.lx.data_ptr()
    assert batch.targets["L2"].data_ptr() == batch.l2x.data_ptr()


def test_packing_casts_only_float_data_and_rejects_invalid_batch(debug_cases):
    batch = data.pack_cases(debug_cases[:2], "cpu", torch.float32)
    assert batch.x.dtype == batch.teacher_c.dtype == torch.float32
    assert batch.edges.dtype == batch.path_graph.dtype == batch.num_nodes.dtype == torch.long
    with pytest.raises(ValueError, match="empty"):
        data.pack_cases([], "cpu", torch.float32)
    with pytest.raises(ValueError, match="duplicate"):
        data.pack_cases([debug_cases[0], debug_cases[0]], "cpu", torch.float32)
    with pytest.raises(TypeError, match="floating"):
        data.pack_cases(debug_cases[:2], "cpu", torch.long)


def test_exclusive_dataset_contains_all_arrays_and_hashes(tmp_path, debug_cases):
    selected = debug_cases[:3]
    manifest = data.save_dataset(tmp_path, selected)
    assert manifest["graph_count"] == 3 and manifest["feature_realization_count"] == 12
    assert manifest["graph_content_unique"] and manifest["feature_content_unique"]
    payload = tmp_path / "dataset.npz"
    assert manifest["dataset_sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8")) == manifest
    with np.load(payload) as saved:
        for case in selected:
            assert np.array_equal(saved[f"{case.graph_id}__features"], case.features.numpy())
            assert np.array_equal(saved[f"{case.graph_id}__teacher_c"], case.teacher_c.numpy())
            assert np.array_equal(saved[f"{case.graph_id}__pair_donor_path"],
                                  np.arange(case.wedges.shape[1]))
    with pytest.raises(FileExistsError, match="overwrite"):
        data.save_dataset(tmp_path, selected)


def test_invalid_contract_is_rejected():
    cfg = _config()
    cfg["train_sizes"] = [5]
    with pytest.raises(ValueError, match="deterministic"):
        data.make_specs(cfg)
    with pytest.raises(ValueError, match="workers"):
        data.prepare_cases(_config(), workers=0)
    cfg = _config()
    del cfg["features"]
    with pytest.raises(KeyError):
        data.make_specs(cfg)


def test_no_conductance_dependencies_are_imported():
    tree = ast.parse(inspect.getsource(data))
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(module and "conductance" in module for module in modules)
