"""Independent small CPU references; these fixtures are not citation results."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.wedge_propagation.classification.evaluation import (
    classification_metrics,
    create_intervention_manifests,
    estimate_second_operator_norm,
    frozen_evaluate,
    model_state_hash,
    split_indices,
)
from research.wedge_propagation.classification.model import PackedClassifier


def graph_fixture(empty=False):
    edges = (
        torch.empty((2, 0), dtype=torch.long)
        if empty
        else torch.tensor([[0, 1, 1, 3], [1, 2, 3, 4]])
    )
    paths = (
        torch.empty((3, 0), dtype=torch.long)
        if empty
        else torch.tensor([[0, 0, 2, 1], [1, 1, 1, 3], [2, 3, 3, 4]])
    )
    incidence = torch.zeros(edges.shape[1], 6, dtype=torch.float64)
    for index, (u, v) in enumerate(edges.T):
        incidence[index, u], incidence[index, v] = -1, 1
    a = torch.zeros(paths.shape[1], 6, dtype=torch.float64)
    for index, (i, j, k) in enumerate(paths.T):
        a[index, i], a[index, j], a[index, k] = 1, -2, 1
    degree = incidence.square().sum(0)
    qdiag = a.square().sum(0)
    adj = torch.eye(6, dtype=torch.float64)
    for u, v in edges.T:
        adj[u, v] = adj[v, u] = 1
    norm = adj.sum(1).rsqrt()
    gcn = adj * norm[:, None] * norm[None, :]
    locations = gcn.nonzero().T
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.2, 0.3, 0.5],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.3, 0.4, 0.3],
        ],
        dtype=torch.float64,
    )
    graph = SimpleNamespace(
        name="DEBUG-reference",
        x=x,
        y=torch.tensor([0, 0, 1, 1, 0, 1]),
        train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
        edges=edges,
        paths=paths,
        degree=degree,
        qdiag=qdiag,
        sd=torch.where(degree > 0, degree.clamp_min(1).rsqrt(), 0),
        sq=torch.where(qdiag > 0, qdiag.clamp_min(1).rsqrt(), 0),
        gcn_edges=locations,
        gcn_weight=gcn[locations[0], locations[1]],
    )
    return graph, a


def eval_config(count=2):
    return {
        "evaluation": {
            "scale_amplitudes": [0.25, 1.0, 4.0],
            "shuffle_and_random_manifests_per_dataset": count,
            "identity_and_shuffle_kappa_modes": [
                "recompute",
                "hold_each_layers_preintervention_kappa",
            ],
        }
    }


def test_metrics_are_per_seed_and_mask_only():
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 4.0], [float("nan"), 0.0]], [[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]]
    )
    labels, mask = torch.tensor([0, 1, -1]), torch.tensor([True, True, False])
    with pytest.raises(ValueError, match="nonfinite"):
        classification_metrics(logits, labels, mask)
    logits[0, 2, 0] = 0
    ce, accuracy = classification_metrics(logits, labels, mask)
    np.testing.assert_allclose(
        ce.numpy(),
        [
            np.mean([np.log1p(np.exp(-2)), np.log1p(np.exp(-4))]),
            np.mean([np.log1p(np.exp(2)), np.log1p(np.exp(3))]),
        ],
        rtol=1e-6,
    )
    assert accuracy.tolist() == [1.0, 0.0]
    fast = classification_metrics(logits, labels, mask, validate=False)
    torch.testing.assert_close(fast[0], ce)


def test_empty_mask_and_bad_labels_are_errors():
    with pytest.raises(ValueError, match="at least one"):
        classification_metrics(
            torch.zeros(1, 2, 2), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="class range"):
        classification_metrics(
            torch.zeros(1, 2, 2), torch.tensor([0, 2]), torch.ones(2, dtype=torch.bool)
        )


def test_manifests_preserve_count_distinct_edges_and_combined_row_norm():
    graph, _ = graph_fixture()
    records = create_intervention_manifests(graph, 2, 91)
    repeated = create_intervention_manifests(graph, 2, 91)
    assert [r["sha256"] for r in records] == [r["sha256"] for r in repeated]
    for record in records:
        assert sorted(record["permutation"].tolist()) == list(range(4))
        rows = record["random_rows"]
        dense = torch.zeros(4, 6)
        for position in range(4):
            dense.scatter_add_(
                1, rows["indices"][position, :, None], rows["coefficients"][position, :, None]
            )
        torch.testing.assert_close(dense.square().sum(1), torch.full((4,), 6.0))
        assert torch.count_nonzero(dense.sum(1)) == 0
        assert record["statistics"]["num_rows"] == graph.paths.shape[1]
        assert record["statistics"]["trace"] == pytest.approx(24)


def test_sparse_norm_estimate_matches_dense_psd_and_zero_operator():
    graph, a = graph_fixture()
    c = torch.tensor([[0.7, 1.2, 1.5, 0.9], [1.4, 0.6, 0.8, 1.3]], dtype=torch.float64)
    diagonal = torch.einsum("pn,sp,pn->sn", a, c, a)
    kappa = (diagonal / graph.qdiag.clamp_min(1)).amax(1)
    found = estimate_second_operator_norm(graph, c, kappa, iterations=100)
    expected = []
    for weights, scale in zip(c, kappa, strict=True):
        operator = (
            graph.sq[:, None] * (a.T @ torch.diag(weights) @ a) * graph.sq[None, :] / (3 * scale)
        )
        expected.append(torch.linalg.eigvalsh(operator).amax())
    torch.testing.assert_close(found, torch.stack(expected), rtol=1e-7, atol=1e-10)
    empty, _ = graph_fixture(empty=True)
    assert estimate_second_operator_norm(
        empty, torch.empty(2, 0, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    ).tolist() == [0.0, 0.0]


def test_frozen_evaluation_coverage_scale_and_immutability():
    graph, _ = graph_fixture()
    model = PackedClassifier(
        "learned_wedge_rms",
        3,
        2,
        [11, 23],
        hidden=4,
        gate_hidden=4,
        dataset_name=graph.name,
        path_chunk=2,
    ).double()
    model.train()
    before = model_state_hash(model)
    result = frozen_evaluate(
        model, graph, [11, 23], eval_config(), create_intervention_manifests(graph, 2, 42)
    )
    assert model.training and model_state_hash(model) == before
    assert result["provenance"]["optimizer_updates"] == 0
    assert len(result["metric_rows"]) == 6
    # 2 identity + 4 shuffle + 1 remove + 2 random; manifests stay within seeds.
    assert len(result["intervention_rows"]) == 9 * 6
    assert len(result["scale_rows"]) == 3 * 6 + 2 * 3 * 2
    probes = [r for r in result["scale_rows"] if r["scope"] == "fixed_layer_Z"]
    assert max(r["c_scale_relerr"] for r in probes) < 1e-12
    assert max(r["message_scale_equivariance_relerr"] for r in probes) < 1e-12
    assert all(r["num_paths"] == 4 for r in result["gate_rows"])


def test_zero_reference_remains_undefined_and_missing_manifest_rejected():
    graph, _ = graph_fixture(empty=True)
    model = PackedClassifier("learned_wedge_raw", 3, 2, [11], hidden=4, gate_hidden=4).double()
    with pytest.raises(ValueError, match="all 2"):
        frozen_evaluate(model, graph, [11], eval_config(), [])
    assert model.training
    result = frozen_evaluate(
        model, graph, [11], eval_config(), create_intervention_manifests(graph, 2, 2)
    )
    probes = [r for r in result["scale_rows"] if r["scope"] == "fixed_layer_Z"]
    assert all(
        r["c_scale_relerr"] is None and r["message_scale_equivariance_relerr"] is None
        for r in probes
    )


def test_manifest_tampering_is_rejected_before_changing_eval_mode():
    graph, _ = graph_fixture()
    model = PackedClassifier("learned_wedge_raw", 3, 2, [11], hidden=4, gate_hidden=4).double()
    records = create_intervention_manifests(graph, 2, 3)
    records[0]["random_rows"]["coefficients"][0, 0] += 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        frozen_evaluate(model, graph, [11], eval_config(), records)
    assert model.training


def test_hash_includes_nonpersistent_buffers():
    graph, _ = graph_fixture()
    model = PackedClassifier("mlp", 3, 2, [11], hidden=4, gate_hidden=4).double()
    before = model_state_hash(model)
    model.dropout_keys.add_(1)
    assert model_state_hash(model) != before


def test_cached_split_indices_match_and_invalidate_after_mask_change():
    graph, _ = graph_fixture()
    first = split_indices(graph)
    assert split_indices(graph)["train"] is first["train"]
    logits = torch.randn(2, 6, 2, dtype=torch.float64)
    expected = classification_metrics(logits, graph.y, graph.train_mask)
    found = classification_metrics(
        logits, graph.y, graph.train_mask, indices=first["train"], validate=False
    )
    torch.testing.assert_close(expected[0], found[0])
    torch.testing.assert_close(expected[1], found[1])
    graph.train_mask[0] = False
    graph.train_mask[2] = True
    changed = split_indices(graph)
    assert changed["train"] is not first["train"]
    assert changed["train"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="do not match"):
        classification_metrics(logits, graph.y, graph.train_mask, indices=first["train"])
