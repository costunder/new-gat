"""Synthetic CPU controlled-corruption contracts, not a data/accuracy experiment."""

import dataclasses

import pytest
import torch

from research.conductance_gat.edge_selection.corruption import build_corruption, sample_nonedges
from research.conductance_gat.edge_selection.topology import build_topology


def _pairs(edges):
    return {tuple(sorted(pair)) for pair in edges.T.tolist()}


def _path(nodes):
    return torch.stack((torch.arange(nodes - 1), torch.arange(1, nodes)))


def test_exact_count_original_exclusion_and_deterministic_seed():
    original = _path(20)
    result = sample_nonedges(20, original, count=50, seed=123)
    assert result.shape == (2, 50)
    assert len(_pairs(result)) == 50
    assert not (_pairs(result) & _pairs(original))
    assert torch.equal(result, sample_nonedges(20, original, count=50, seed=123))
    assert not torch.equal(result, sample_nonedges(20, original, count=50, seed=124))


def test_dense_nearly_complete_complement_is_exact_without_rejection_failure():
    complete = torch.triu_indices(10, 10, 1)
    missing = torch.tensor([2, 11, 29, 40])
    keep = torch.ones(complete.shape[1], dtype=torch.bool)
    keep[missing] = False
    result = sample_nonedges(10, complete[:, keep], count=4, seed=1)
    assert _pairs(result) == _pairs(complete[:, missing])
    with pytest.raises(ValueError, match="only 4 eligible"):
        sample_nonedges(10, complete[:, keep], count=5, seed=1)


def test_no_cross_original_components_or_ppi_graphs_even_with_extra_isolate():
    original = torch.cat((_path(4), _path(4) + 4), dim=1)
    graph = torch.tensor([0] * 4 + [1] * 5)
    result = sample_nonedges(9, original, count=6, seed=1, node_graph=graph)
    assert result.shape[1] == 6
    assert not (result == 8).any()
    assert torch.equal(graph[result[0]], graph[result[1]])
    assert all((left < 4) == (right < 4) for left, right in result.T.tolist())


def test_train_eval_corruption_distinct_sorted_and_provenance_not_a_model_feature():
    original = _path(12)
    train = build_corruption(12, original, count=15, seed=42, split="train")
    validation = build_corruption(
        12, original, count=15, seed=42, split="validation", exclude=train.negative_incidence
    )
    assert not (_pairs(train.negative_incidence) & _pairs(validation.negative_incidence))
    for data in (train, validation):
        data.verify_unchanged()
        assert set(data.model_input()) == {"incidence_edge_index"}
        columns = [tuple(column) for column in data.candidate_incidence.T.tolist()]
        assert columns == sorted(columns)
        assert sum(data.edge_targets.tolist()) == original.shape[1]
        assert all(
            bool(target) == (tuple(edge) in _pairs(original))
            for edge, target in zip(columns, data.edge_targets.tolist(), strict=True)
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        train.provenance.seed = 99
    with pytest.raises(ValueError, match="explicit excluded"):
        build_corruption(12, original, count=15, seed=43, split="validation")


def test_provenance_detects_tensor_mutation_but_sanitized_copy_is_independent():
    data = build_corruption(8, _path(8), count=3, seed=0, split="train")
    data.model_input()["incidence_edge_index"].zero_()
    data.verify_unchanged()
    data.edge_targets[0] = 99
    with pytest.raises(ValueError, match="changed after provenance"):
        data.verify_unchanged()


def test_excluded_edges_must_be_genuine_same_component_nonedges():
    original = torch.tensor([[0, 1, 3], [1, 2, 4]])
    with pytest.raises(ValueError, match="original edge"):
        sample_nonedges(5, original, count=0, seed=0, exclude=original[:, :1])
    with pytest.raises(ValueError, match="original connected components"):
        sample_nonedges(5, original, count=0, seed=0, exclude=torch.tensor([[0], [3]]))


def test_cached_components_are_seed_independent_and_wrong_original_plan_is_rejected():
    original = torch.cat((_path(4), _path(5) + 4), dim=1)
    first, second = [build_topology(9, original, forest_seed=seed) for seed in (1, 8)]
    left = sample_nonedges(9, original, count=5, seed=16, original_plan=first)
    right = sample_nonedges(9, original, count=5, seed=16, original_plan=second)
    assert torch.equal(left, right)
    with pytest.raises(ValueError, match="does not match"):
        sample_nonedges(9, original.flip(1), count=5, seed=16, original_plan=first)


def test_zero_corruption_is_explicit_not_an_implicit_fallback():
    original = torch.triu_indices(4, 4, 1)
    data = build_corruption(4, original, count=0, seed=0, split="train")
    assert data.negative_incidence.shape == (2, 0)
    assert data.provenance.inserted_edge_count == 0
    assert torch.equal(data.candidate_incidence, original)
    with pytest.raises(ValueError, match="only 0 eligible"):
        sample_nonedges(4, original, count=1, seed=0)


def test_explicit_zero_request_checks_geometry_without_unnecessary_dfs(monkeypatch):
    import research.conductance_gat.edge_selection.corruption as corruption

    monkeypatch.setattr(
        corruption, "build_topology", lambda *a, **k: pytest.fail("unnecessary DFS")
    )
    assert sample_nonedges(4, _path(4), count=0, seed=0).shape == (2, 0)
    with pytest.raises(ValueError, match="self-loops"):
        sample_nonedges(4, torch.tensor([[0], [0]]), count=0, seed=0)


@pytest.mark.parametrize("nodes", [2, 3, 4, 5, 7, 13, 31])
def test_entire_complement_matches_small_explicit_oracle(nodes):
    original = _path(nodes)
    all_pairs = _pairs(torch.triu_indices(nodes, nodes, 1))
    expected = all_pairs - _pairs(original)
    result = sample_nonedges(nodes, original, count=len(expected), seed=7)
    assert _pairs(result) == expected
