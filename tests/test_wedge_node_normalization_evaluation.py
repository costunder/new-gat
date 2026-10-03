"""Explicit six-node DEBUG fixtures and dense frozen-forward references."""

from types import SimpleNamespace

import pytest
import torch

from research.wedge_propagation.node_normalization.evaluation import (
    LEARNED,
    TARGETS,
    TREATMENTS,
    _norm_match,
    create_intervention_manifests,
    frozen_evaluate,
    frozen_forward,
    permutation_sha256,
)
from research.wedge_propagation.node_normalization.model import PackedClassifier


def fixture(empty=False):
    edges = torch.tensor([[0, 1, 1, 3], [1, 2, 3, 4]])
    paths = torch.tensor([[0, 0, 2, 1], [1, 1, 1, 3], [2, 3, 3, 4]])
    if empty:
        edges, paths = torch.empty((2, 0), dtype=torch.long), torch.empty((3, 0), dtype=torch.long)
    b = torch.zeros(edges.shape[1], 6, dtype=torch.float64)
    a = torch.zeros(paths.shape[1], 6, dtype=torch.float64)
    for index, (u, v) in enumerate(edges.T):
        b[index, u], b[index, v] = -1, 1
    for index, (i, j, k) in enumerate(paths.T):
        a[index, i], a[index, j], a[index, k] = 1, -2, 1
    degree, diagonal = b.square().sum(0), a.square().sum(0)
    graph = SimpleNamespace(
        name="DEBUG-frozen-node",
        edges=edges,
        paths=paths,
        degree=degree,
        qdiag=diagonal,
        sd=torch.where(degree > 0, degree.clamp_min(1).rsqrt(), 0),
        sq=torch.where(diagonal > 0, diagonal.clamp_min(1).rsqrt(), 0),
        x=torch.tensor(
            [
                [1.0, 0.2, 0.0],
                [0.0, 0.3, 1.0],
                [0.7, 0.1, 0.2],
                [0.2, 0.5, 0.8],
                [0.4, 0.6, 0.1],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
    )
    return graph, a, 0.5 * graph.sd[:, None] * (b.T @ b) * graph.sd[None]


def model(condition, seeds=(11, 23)):
    return PackedClassifier(
        condition,
        3,
        2,
        seeds,
        hidden=4,
        gate_hidden=4,
        dataset_name="DEBUG-frozen-node",
        path_chunk=2,
    ).double()


def dense_branch(graph, a, z, c, local):
    q = torch.einsum("pi,sp,pj->sij", a, c, a)
    diagonal = torch.diagonal(q, dim1=1, dim2=2)
    if local:
        scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(), 0)
        operator = scale[:, :, None] * q * scale[:, None, :] / 3
    else:
        kappa = (diagonal / graph.qdiag.clamp_min(1)[None]).amax(1)
        if not a.shape[0]:
            kappa = z.new_ones(z.shape[0])
        operator = (
            graph.sq[None, :, None] * q * graph.sq[None, None, :] / (3 * kappa[:, None, None])
        )
    return torch.bmm(operator, z)


def dense_forward(m, graph, a, lbar, treatment, target, manifest):
    h = graph.x[None].expand(len(m.seeds), -1, -1)
    for layer in (0, 1):
        z = torch.bmm(h, m.projections[layer])
        c = m._gate(graph, layer, z, m._sigma(graph, z))
        reference = dense_branch(graph, a, z, c, "_local_" in m.experiment_condition)
        active = target == "both" or target == f"layer_{layer}"
        if active:
            if treatment.startswith("c_identity"):
                c = torch.ones_like(c)
            elif treatment.startswith("c_position_shuffle"):
                c = c[:, manifest["permutation"]]
            candidate = dense_branch(graph, a, z, c, "_local_" in m.experiment_condition)
            if treatment.endswith("norm_matched"):
                gain = torch.linalg.vector_norm(reference, dim=(1, 2)) / torch.linalg.vector_norm(
                    candidate, dim=(1, 2)
                )
                candidate = candidate * gain[:, None, None]
            if treatment == "second_branch_remove":
                candidate = torch.zeros_like(candidate)
        else:
            candidate = reference
        alpha, beta = m._coefficients(layer, z)
        h = z - alpha[:, None, None] * torch.matmul(lbar, z) - beta[:, None, None] * candidate
        if layer == 0:
            h = h.relu()
    return h


@pytest.mark.parametrize("condition", LEARNED)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("treatment", TREATMENTS)
def test_all_scopes_and_native_or_norm_matched_operators_match_dense(condition, target, treatment):
    graph, a, lbar = fixture()
    m = model(condition).eval()
    manifest = create_intervention_manifests(graph, 1, 977)[0]
    actual, details = frozen_forward(m, graph, treatment, target, manifest)
    expected = dense_forward(m, graph, a, lbar, treatment, target, manifest)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    assert not actual.requires_grad
    if treatment.endswith("norm_matched"):
        for layer, detail in enumerate(details):
            if target == "both" or target == f"layer_{layer}":
                torch.testing.assert_close(
                    torch.linalg.vector_norm(detail["t_message"], dim=(1, 2)),
                    detail["reference_branch_norm"],
                    rtol=1e-12,
                    atol=1e-12,
                )


@pytest.mark.parametrize("condition", ("fixed_wedge", *LEARNED))
def test_frozen_full_evaluation_coverage_seed_names_and_parameter_preservation(condition):
    graph, _, _ = fixture()
    m = model(condition).train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    records = create_intervention_manifests(graph, 2, 977)
    result = frozen_evaluate(
        m, graph, m.seeds, {"evaluation": {"shuffle_manifests_per_dataset": 2}}, records
    )
    assert len(result["metric_rows"]) == 6
    assert len(result["intervention_rows"]) == (126 if condition in LEARNED else 0)
    assert len(result["gate_rows"]) == (88 if condition in LEARNED else 4)
    assert result["scale_rows"] == []
    assert all(r["condition"] == condition for r in result["gate_rows"])
    assert result["provenance"]["models_unchanged"] is True
    assert result["provenance"]["optimizer_updates"] == 0
    assert m.training is True
    for key, value in m.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_fixed_manifest_reproducibility_coverage_and_hash_rejection():
    graph, _, _ = fixture()
    first, second = (
        create_intervention_manifests(graph, 2, 977),
        create_intervention_manifests(graph, 2, 977),
    )
    assert all("random_rows" not in record for record in first)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left["permutation"], right["permutation"])
        assert left["sha256"] == right["sha256"] == permutation_sha256(left["permutation"])
    first[0]["permutation"][0] = first[0]["permutation"][1]
    with pytest.raises(ValueError, match="hash"):
        frozen_evaluate(
            model(LEARNED[0]),
            graph,
            (11, 23),
            {"evaluation": {"shuffle_manifests_per_dataset": 2}},
            first,
        )


def test_norm_match_zero_reference_candidate_rules():
    zero = torch.zeros(2, 3, 4)
    one = torch.ones_like(zero)
    message, gain = _norm_match(zero, zero)
    assert not message.any() and torch.equal(gain, torch.ones(2))
    message, gain = _norm_match(one, zero)
    assert not message.any() and not gain.any()
    with pytest.raises(ValueError, match="zero candidate"):
        _norm_match(zero, one)


@pytest.mark.parametrize("condition", LEARNED)
def test_no_wedge_undefined_geometry_is_preserved(condition):
    graph, _, _ = fixture(empty=True)
    records = create_intervention_manifests(graph, 1, 977)
    result = frozen_evaluate(
        model(condition),
        graph,
        (11, 23),
        {"evaluation": {"shuffle_manifests_per_dataset": 1}},
        records,
    )
    assert all(row["c_std"] is None for row in result["gate_rows"])
    assert all(row["message_norm_ratio"] is None for row in result["gate_rows"])
    assert all(row["message_norm_ratio_defined"] is False for row in result["gate_rows"])


def test_seed_order_and_eval_mode_requirements():
    graph, _, _ = fixture()
    m = model(LEARNED[0])
    with pytest.raises(ValueError, match="seed order"):
        frozen_evaluate(m, graph, (23, 11), {"evaluation": {"shuffle_manifests_per_dataset": 1}})
    with pytest.raises(ValueError, match="model.eval"):
        frozen_forward(m, graph, "c_identity", "both")
