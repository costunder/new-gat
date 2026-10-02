"""Debug tests of the full fixed-operator data contract, never final metrics."""

import ast
import inspect
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from research.wedge_propagation import data


@pytest.fixture(scope="module")
def full_cases():
    return data.make_cases()


def test_full_contract_has_all_198_graphs_and_3168_scalar_inputs(full_cases):
    assert len(full_cases) == 198
    assert sum(case.features.shape[1] for case in full_cases) == 3168
    assert len({case.graph_id for case in full_cases}) == 198
    assert Counter(case.num_nodes for case in full_cases) == dict.fromkeys(
        (20, 30, 40, 60, 80, 100), 33
    )
    assert Counter(case.family for case in full_cases) == {
        "cycle": 6,
        "star": 6,
        "grid": 6,
        "er": 60,
        "tree": 60,
        "tree_chord": 60,
    }


def test_cases_are_cpu_canonical_physical_edges_and_full_float64_features(full_cases):
    for case in full_cases:
        assert case.edges.device.type == case.features.device.type == "cpu"
        assert case.edges.dtype == torch.long
        assert case.features.dtype == torch.float64
        assert case.edges.ndim == 2 and case.edges.shape[0] == 2
        assert case.features.shape == (case.num_nodes, 16)
        assert torch.isfinite(case.features).all()
        left, right = case.edges
        assert (left < right).all()
        assert (left >= 0).all() and (right < case.num_nodes).all()
        keys = left * case.num_nodes + right
        assert (keys[1:] > keys[:-1]).all()


def test_full_fixed_structures_and_tree_edge_counts(full_cases):
    grid_columns = dict(zip((20, 30, 40, 60, 80, 100), (4, 5, 5, 6, 8, 10), strict=True))
    for case in full_cases:
        n = case.num_nodes
        edge_count = case.edges.shape[1]
        if case.family == "cycle":
            assert edge_count == n
            assert torch.equal(
                torch.bincount(case.edges.flatten(), minlength=n), torch.full((n,), 2)
            )
        elif case.family in ("star", "tree"):
            assert edge_count == n - 1
        elif case.family == "tree_chord":
            assert edge_count == n - 1 + n // 4
        elif case.family == "grid":
            columns = grid_columns[n]
            rows = n // columns
            assert edge_count == rows * (columns - 1) + columns * (rows - 1)


def test_tree_chord_contains_the_corresponding_tree(full_cases):
    by_id = {case.graph_id: case for case in full_cases}
    for case in full_cases:
        if case.family != "tree_chord":
            continue
        tree = by_id[case.graph_id.replace("tree_chord", "tree")]
        assert case.base_tree_seed == tree.graph_seed
        tree_edges = {tuple(pair) for pair in tree.edges.T.tolist()}
        chord_edges = {tuple(pair) for pair in case.edges.T.tolist()}
        assert tree_edges < chord_edges
        assert len(chord_edges - tree_edges) == case.num_nodes // 4


def test_tree_generation_is_connected_with_no_isolate(full_cases):
    for case in full_cases:
        if case.family not in ("tree", "tree_chord"):
            continue
        neighbors = [[] for _ in range(case.num_nodes)]
        for left, right in case.edges.T.tolist():
            neighbors[left].append(right)
            neighbors[right].append(left)
        seen = {0}
        frontier = [0]
        while frontier:
            for neighbor in neighbors[frontier.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        assert len(seen) == case.num_nodes


def test_er_uses_every_bernoulli_pair_and_keeps_declared_nodes():
    spec = data.GraphSpec("er-independent", "er", 100, 16, 0, 1)
    case = data.make_case(spec)
    left, right = np.triu_indices(spec.num_nodes, k=1)
    selected = np.random.default_rng(spec.graph_seed).random(len(left)) < 4 / (spec.num_nodes - 1)
    expected = torch.from_numpy(np.vstack((left[selected], right[selected])).copy())
    assert torch.equal(case.edges, expected)
    assert case.features.shape[0] == case.num_nodes == 100
    degree = torch.bincount(case.edges.flatten(), minlength=case.num_nodes)
    assert (degree == 0).any(), "chosen reference draw must exercise retained isolated nodes"


def test_named_seed_streams_are_stable_and_separate():
    first = data.make_specs()
    second = data.make_specs()
    assert first == second
    assert all(spec.graph_seed != spec.feature_seed for spec in first)
    assert len({spec.graph_seed for spec in first}) == 198
    assert len({spec.feature_seed for spec in first}) == 198
    assert data.named_seed(5, "graph", "case") != data.named_seed(6, "graph", "case")


def test_feature_seed_change_does_not_change_topology():
    spec = next(spec for spec in data.make_specs() if spec.family == "tree_chord")
    reference = data.make_case(spec)
    changed = data.make_case(replace(spec, feature_seed=spec.feature_seed + 1))
    assert torch.equal(reference.edges, changed.edges)
    assert data.graph_content_hash(reference) == data.graph_content_hash(changed)
    assert not torch.equal(reference.features, changed.features)
    assert data.feature_content_hash(reference) != data.feature_content_hash(changed)


def test_execution_order_does_not_change_any_case_or_global_rng():
    specs = data.make_specs(profile="debug")
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    forward = {case.graph_id: case for case in map(data.make_case, specs)}
    reverse = {case.graph_id: case for case in map(data.make_case, reversed(specs))}
    assert torch.equal(torch_state, torch.random.get_rng_state())
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    assert np.array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]
    for graph_id, case in forward.items():
        assert torch.equal(case.edges, reverse[graph_id].edges)
        assert torch.equal(case.features, reverse[graph_id].features)


def test_debug_profile_is_explicit_and_does_not_change_full_default():
    debug = data.make_cases(profile="debug")
    assert len(debug) == 18
    assert {case.num_nodes for case in debug} == {5, 8}
    assert all(case.features.shape[1] == 4 for case in debug)
    assert all(case.graph_id.startswith("debug-") for case in debug)
    assert len(data.make_specs()) == 198
    assert all(spec.num_features == 16 for spec in data.make_specs())


def test_content_hash_includes_isolated_node_count():
    one = data.GraphCase(
        "one", "empty", 5,
        torch.empty((2, 0), dtype=torch.long),
        torch.ones((5, 1), dtype=torch.float64), 0, 1,
    )
    two = replace(one, num_nodes=6, features=torch.ones((6, 1), dtype=torch.float64))
    assert data.graph_content_hash(one) != data.graph_content_hash(two)


def test_generator_imports_only_independent_runtime_dependencies():
    tree = ast.parse(inspect.getsource(data))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert set(imports) <= {"__future__", "dataclasses", "hashlib", "heapq", "numpy", "torch"}


@pytest.mark.parametrize("profile", ["subset", "fast", ""])
def test_undeclared_profile_is_rejected(profile):
    with pytest.raises(ValueError, match="unknown profile"):
        data.make_specs(profile=profile)


def test_invalid_generation_contract_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        data.make_specs(master_seed=-1)
    with pytest.raises(TypeError, match="integer"):
        data.make_specs(master_seed=True)
    bad_grid = data.GraphSpec("bad", "grid", 5, 4, 1, 2, grid_columns=2)
    with pytest.raises(ValueError, match="divide"):
        data.make_case(bad_grid)
    bad_chord = data.GraphSpec("bad", "tree_chord", 5, 4, 1, 2)
    with pytest.raises(ValueError, match="base tree seed"):
        data.make_case(bad_chord)
    bad_family = data.GraphSpec("bad", "unknown", 5, 4, 1, 2)
    with pytest.raises(ValueError, match="unknown graph family"):
        data.make_case(bad_family)
