"""Exact synthetic CPU graph diagnostics; no real-data or hardware performance claim."""

import math

import pytest
import torch

from research.conductance_gat.v5.distribution_audit import audit_conductance_distribution
from research.conductance_gat.v5.stage_audit import _json


def _audit(c, *, correction=None, normalization="row", batch=None, edges=None, heads=2):
    if edges is None:
        edges = torch.tensor([[0, 0, 0], [1, 2, 3]])
    if batch is None:
        batch = torch.zeros(5, dtype=torch.long)
    graphs = int(batch.max()) + 1
    return _json(
        audit_conductance_distribution(
            c,
            edges,
            batch,
            graphs,
            heads=heads,
            beta=torch.full((graphs, heads), 0.5),
            sampling_correction=correction,
            normalization=normalization,
        )
    )


def test_raw_c_quantiles_histogram_and_fractions_are_not_cv_or_alpha():
    row = _audit(torch.tensor([0.2, 0.8, 2.0]))["graphs"][0]
    raw = row["raw_c"]
    assert raw["mean"] == pytest.approx([1, 1])
    assert raw["quantiles"]["0.5"] == pytest.approx([0.8, 0.8])
    assert raw["fractions"]["c_ge_0_7"] == pytest.approx([2 / 3, 2 / 3])
    assert raw["fractions"]["abs_c_minus_1_le_0_1"] == [0, 0]
    assert raw["fractions"]["c_gt_1"] == pytest.approx([1 / 3, 1 / 3])
    assert all(sum(counts) == 3 for counts in raw["histogram"]["counts_by_head"])
    assert row["beta_by_head"] == [0.5, 0.5]


def test_isolates_and_degree_one_are_explicit_and_concentration_has_correct_values():
    row = _audit(torch.ones(3))["graphs"][0]
    assert row["isolates"] == 1 and row["degree_one_nodes"] == 3
    scope = row["probabilities"]["weighted_c_row_alpha"]["node_scopes"]
    assert scope["degree_1"]["top1"]["mean"] == [1, 1]
    assert scope["degree_1"]["normalized_entropy"]["mean"] == [0, 0]
    assert scope["degree_1"]["effective_neighbors"]["mean"] == [1, 1]
    center = scope["degree_2_4"]
    assert center["nodes"] == 1
    assert center["top1"]["mean"] == pytest.approx([1 / 3, 1 / 3])
    assert center["normalized_entropy"]["mean"] == pytest.approx([1, 1])
    assert center["effective_neighbors"]["mean"] == pytest.approx([3, 3])
    assert scope["all_nonisolates"]["nodes"] == 4


def test_symmetric_actual_coefficients_are_not_falsely_called_row_probabilities():
    report = _audit(torch.ones(3), normalization="symmetric")
    row = report["graphs"][0]
    assert "diagnostic only" in report["actual_kernel_row_relative_definition"] or (
        report["actual_kernel_row_relative_definition"] == "P / row_sum(P); diagnostic only"
    )
    assert row["actual_one_hop_row_sum"]["quantiles"]["1.0"] == pytest.approx([math.sqrt(3)] * 2)
    assert row["actual_one_hop_coefficient"]["mean"] == pytest.approx([1 / math.sqrt(3)] * 2)


@pytest.mark.parametrize("normalization", ["row", "symmetric"])
def test_scale_only_head_differences_vanish_after_neighbor_normalization(normalization):
    c = torch.tensor([[0.2, 2.0], [0.8, 8.0], [2.0, 20.0]])
    row = _audit(c, normalization=normalization)["graphs"][0]
    assert row["raw_c"]["mean"][1] > row["raw_c"]["mean"][0]
    for family in row["probabilities"].values():
        diversity = family["head_diversity_nonisolates"]
        assert diversity["total_variation"]["mean"][0] < 1e-6
        assert abs(diversity["jensen_shannon_nats"]["mean"][0]) < 1e-6


def test_head_preference_changes_have_positive_tv_and_js():
    row = _audit(torch.tensor([[0.1, 2.0], [0.9, 0.9], [2.0, 0.1]]))["graphs"][0]
    diversity = row["probabilities"]["weighted_c_row_alpha"]["head_diversity_nonisolates"]
    assert diversity["total_variation"]["mean"][0] > 0.1
    assert diversity["jensen_shannon_nats"]["mean"][0] > 0.01


def test_c_one_reference_retains_sampling_correction_not_uniform_neighbors():
    row = _audit(torch.ones(3), correction=torch.tensor([1.0, 2.0, 3.0]))["graphs"][0]
    center = row["probabilities"]["weighted_c_row_alpha"]["node_scopes"]["degree_2_4"]
    assert center["top1"]["mean"] == pytest.approx([0.5, 0.5])
    assert center["c_one_same_correction"]["top1"]["mean"] == pytest.approx([0.5, 0.5])
    assert center["c_one_total_variation"]["mean"] == [0, 0]


def test_graph_scopes_do_not_pool_disjoint_graph_distributions():
    row = _audit(
        torch.tensor([0.2, 1.8]),
        edges=torch.tensor([[0, 2], [1, 3]]),
        batch=torch.tensor([0, 0, 1, 1, 1]),
    )["graphs"]
    assert row[0]["raw_c"]["mean"] == pytest.approx([0.2, 0.2])
    assert row[1]["raw_c"]["mean"] == pytest.approx([1.8, 1.8])
    assert row[0]["isolates"] == 0 and row[1]["isolates"] == 1


def test_edgeless_scope_is_unavailable_not_fake_concentration():
    row = _audit(torch.empty(0), edges=torch.empty((2, 0), dtype=torch.long))["graphs"][0]
    assert row["isolates"] == 5
    assert row["raw_c"]["mean"] is None
    assert (
        row["probabilities"]["weighted_c_row_alpha"]["node_scopes"]["all_nonisolates"]["top1"][
            "mean"
        ]
        is None
    )
