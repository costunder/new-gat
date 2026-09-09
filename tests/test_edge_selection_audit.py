"""Synthetic CPU diagnostics only, not official dataset or GPU validation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.conductance_gat.edge_selection import diagnostics as diag
from research.conductance_gat.edge_selection.audit import Observer, PredictionSummary
from research.conductance_gat.edge_selection.model import EdgeSelectionClassifier
from research.conductance_gat.edge_selection.topology import build_topology


def test_zero_safe_distribution_including_empty_values():
    assert diag.distribution([])["mean"] is None
    report = diag.distribution([0.0, 0.0, 0.5, 1.0, 2.0])
    assert report["zero_fraction"] == pytest.approx(0.4)
    assert report["above_one_count"] == 1
    assert sum(report["unit_interval_histogram_counts"]) == 4
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="nonfinite"):
        diag.distribution([float("nan")])


def test_quantile_above_torch_limit_uses_all_values_without_mutation():
    # Full-size synthetic diagnostic regression, not a reduced training profile.
    size = (1 << 24) + 1
    values = torch.arange(size, dtype=torch.float32)
    values = values.flip(0)
    before = values.clone()
    report = diag.distribution(values)
    expected = np.asarray(report["quantile_probabilities"]) * (size - 1)
    np.testing.assert_allclose(report["quantiles"], expected, rtol=0, atol=1e-8)
    assert report["count"] == report["quantile_observation_count"] == size
    assert report["above_one_count"] == size - 2
    assert sum(report["unit_interval_histogram_counts"]) == 2
    assert report["mean"] == (size - 1) / 2
    assert report["zero_fraction"] == 1 / size
    assert torch.equal(values, before)
    json.dumps(report, allow_nan=False)


def test_histogram_counts_above_float32_exact_integer_range_are_not_rounded():
    size = (1 << 24) + 3
    report = diag.distribution(torch.zeros(size))
    assert report["unit_interval_histogram_counts"][0] == size
    assert sum(report["unit_interval_histogram_counts"]) == size
    assert report["quantiles"] == [0.0] * 9
    assert report["zero_fraction"] == 1.0


@pytest.mark.parametrize("values", [[3.0, 1.0, 2.0, 5.0, -1.0], [1.0], [0.0, 0.0, 1.0, 1.0]])
def test_quantile_matches_small_float64_linear_oracle(values):
    report = diag.distribution(values)
    oracle = torch.quantile(
        torch.tensor(values, dtype=torch.float64),
        torch.tensor(report["quantile_probabilities"], dtype=torch.float64),
    )
    np.testing.assert_allclose(report["quantiles"], oracle.numpy(), rtol=0, atol=1e-14)


def test_active_components_isolates_cycle_rank_and_path_change():
    edges = np.array([[0, 0, 1], [1, 2, 2]])
    before = diag.adjacency(4, edges)
    reference = diag.path_reference(before, sources=4)
    assert diag.topology_statistics(before)["cycle_rank"] == 1
    tree = diag.adjacency(4, edges, [1, 1, 0])
    topology = diag.topology_statistics(tree)
    assert topology["components"] == 2 and topology["isolated_nodes"] == 1
    assert topology["cycle_rank"] == 0
    change = diag.path_change(tree, reference)
    assert change["lost_reachable_ordered_pairs"] == 0
    assert change["distance_increase_on_retained_pairs"]["quantiles"][-1] == 1
    disconnected = diag.path_change(diag.adjacency(4, edges, [1, 0, 0]), reference)
    assert disconnected["lost_reachable_ordered_pairs"] == 4
    empty = diag.path_change(diag.adjacency(4, edges, [0, 0, 0]), reference)
    assert empty["stretch_on_retained_pairs"]["count"] == 0


def test_origins_are_separate_and_empty_added_not_reported_as_perfect():
    result = diag.gate_origins(torch.tensor([1.0, 0.0, 0.2]), torch.tensor([1.0, 0.0, 0.0]))
    assert result["original"]["active_rate"] == 1
    assert result["added"]["active_rate"] == 0.5
    assert diag.gate_origins(torch.ones(3), torch.ones(3))["added"]["active_rate"] is None


def test_zero_neighbor_coefficients_do_not_create_nan_or_fake_neighbor():
    report = diag.coefficient_statistics(
        torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([0, 1]), torch.tensor([1, 2]), 4
    )
    assert report["incoming_mass"]["mean"] == 0
    isolated = report["active_degree_bins"]["0:1"]
    assert isolated["nodes"] == 4 and isolated["effective_neighbors"]["mean"] == 0
    json.dumps(report, allow_nan=False)


def test_prediction_summary_preserves_integer_counts_under_bf16():
    count = 29799
    values = torch.ones(count, 2, dtype=torch.bfloat16)
    values[:, 1] = 0
    batch = SimpleNamespace(
        selected_indices=torch.arange(count),
        graph=SimpleNamespace(y=torch.zeros(count, dtype=torch.long)),
    )
    metric = PredictionSummary(False)
    metric.add(values, values, batch)
    assert metric.report()["validation"] == 1.0
    assert metric.report()["logit_relative_l2"] == 0.0


@pytest.mark.parametrize(
    "mode",
    ["full", "forest_only", "forest_random", "forest_learned", "forest_cycle", "hard_concrete"],
)
def test_observer_uses_real_core_and_preserves_parameters_rng_diagnostics(mode):
    torch.manual_seed(14)
    edges = torch.tensor([[0, 0, 0, 1, 1, 2, 4], [1, 2, 3, 2, 3, 3, 5]])
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    plan = build_topology(7, edges, groups)
    graph = SimpleNamespace(
        x=torch.randn(7, 5),
        y=torch.arange(7) % 3,
        incidence_edge_index=edges,
        batch=groups,
        _v5_num_graphs=3,
        edge_selection_topology=plan,
    )
    batch = SimpleNamespace(
        graph=graph, topology=plan, origin_targets=torch.ones(7), selected_indices=torch.arange(7)
    )
    model = EdgeSelectionClassifier(
        5,
        3,
        hidden_channels=16,
        layers=2,
        heads=4,
        dropout=0.0,
        edge_chunk_size=3,
        conductance_heads="per_head",
        propagation_normalization="row",
        selection_config={"condition": mode, "chord_fraction": 0.5},
    ).eval()
    args = SimpleNamespace(
        dataset="cora", forest_seed=0, selection_mode=mode, edge_chunk_size=3, precision="fp32"
    )
    with torch.no_grad():
        logits = model(graph)
        saved = {name: value.clone() for name, value in model.state_dict().items()}
        gates = [operator.last_gate.clone() for operator in model.operators]
        rng = torch.get_rng_state()
        observer = Observer(args, path_sources=3)
        observer(model, batch, logits, 0)
        report = observer.report(float((logits.argmax(-1) == graph.y).float().mean()))
    assert len(report["layers_and_graphs"]) == 2
    assert len(report["layers_and_graphs"][0]["graphs"]) == 3
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    for operator, gate in zip(model.operators, gates, strict=True):
        torch.testing.assert_close(operator.last_gate, gate, rtol=0, atol=0)
    json.dumps(report, allow_nan=False)


def test_multilabel_summary_matches_exact_micro_f1():
    logits = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
    batch = SimpleNamespace(
        selected_indices=None, graph=SimpleNamespace(y=torch.tensor([[1, 1], [0, 1]]))
    )
    metric = PredictionSummary(True)
    metric.add(logits, logits, batch)
    assert metric.report()["validation"] == pytest.approx(4 / 6)
