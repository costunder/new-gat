"""Tests for the standalone static Cycle-PE tree-augmentation track."""

from __future__ import annotations

import numpy as np
import pytest

from chartgat.algebra import chart_transition, incidence_matrix, validate_spanning_tree
from chartgat.graphs import make_connected_graph
from research.tree_augmentation.augmentation import (
    build_tree_chart,
    cycle_projector,
    cycle_projector_diagonal,
    ensure_full_cycle_budget,
    find_unseen_chart,
    lossless_transition_error,
    run_static_cycle_pe_probe,
    sample_tree_charts,
    transition_cocycle_error,
    transport_coordinates,
)


@pytest.fixture
def graph() -> tuple[int, list[tuple[int, int]]]:
    num_nodes = 9
    return num_nodes, make_connected_graph(num_nodes, extra_edges=5, seed=23)


def test_bfs_dfs_and_random_tree_sampling(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    B = incidence_matrix(num_nodes, edges)
    for method in ("bfs", "dfs", "random"):
        chart = build_tree_chart(num_nodes, edges, method=method, seed=31)
        validate_spanning_tree(B, chart.tree_edge_indices)
        assert chart.beta == len(edges) - num_nodes + 1
        assert np.allclose(B.T @ chart.basis, 0.0)


def test_full_beta_chart_transitions_are_lossless_and_unimodular(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    charts = sample_tree_charts(num_nodes, edges, random_count=4, random_seed_start=40)
    rng = np.random.default_rng(5)
    coordinates = rng.normal(size=(charts[0].beta, 3))
    physical = charts[0].basis @ coordinates

    for target in charts[1:]:
        target_coordinates = transport_coordinates(charts[0], target, coordinates)
        transition = chart_transition(charts[0].basis, target.basis)
        assert np.allclose(target.basis @ target_coordinates, physical, atol=1e-10)
        assert np.allclose(transition, np.rint(transition), atol=1e-10)
        assert abs(round(float(np.linalg.det(transition)))) == 1

    assert lossless_transition_error(charts, coordinates) < 1e-10


def test_chart_transition_cocycle_law(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    charts = sample_tree_charts(num_nodes, edges, random_count=3, random_seed_start=70)
    assert len(charts) >= 3
    assert transition_cocycle_error(charts) < 1e-10


def test_cycle_projector_is_chart_invariant(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    charts = sample_tree_charts(num_nodes, edges, random_count=5, random_seed_start=90)
    reference = cycle_projector(charts[0].basis)
    for chart in charts[1:]:
        assert np.allclose(cycle_projector(chart.basis), reference, atol=1e-10)
        assert np.allclose(cycle_projector_diagonal(chart.basis), np.diag(reference), atol=1e-10)


def test_lossy_cycle_budget_is_explicitly_disabled() -> None:
    assert ensure_full_cycle_budget(6) == 6
    assert ensure_full_cycle_budget(6, 6) == 6
    with pytest.raises(NotImplementedError, match="lossy extension"):
        ensure_full_cycle_budget(6, 5)


def test_static_probe_reports_fixed_multi_and_unseen_evaluation(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    training = sample_tree_charts(
        num_nodes,
        edges,
        random_count=5,
        random_seed_start=120,
    )
    unseen = find_unseen_chart(num_nodes, edges, training, seed_start=900)
    result = run_static_cycle_pe_probe(
        training,
        unseen,
        hidden_dim=16,
        epochs=12,
        learning_rate=0.01,
        seed=3,
    )

    assert result["num_training_charts"] == len(training)
    assert result["cycle_rank_beta"] == len(edges) - num_nodes + 1
    for key in (
        "fixed_train_mse",
        "multi_train_mse",
        "fixed_unseen_mse",
        "multi_unseen_mse",
        "projector_oracle_unseen_mse",
    ):
        assert np.isfinite(result[key])
        assert result[key] >= 0.0
    assert result["projector_oracle_unseen_mse"] < 1e-20
