"""Experiment 4.1 explicit DEBUG dense mathematics and frozen-state tests."""

from __future__ import annotations

import gc
import weakref
from itertools import combinations
from types import SimpleNamespace

import pytest
import torch

from research.wedge_propagation.branch_strength.core import (
    TREATMENTS,
    FrozenStrengthClassifier,
    _active_q_indices,
    fixed_z_statistics,
)
from research.wedge_propagation.classification.evaluation import model_state_hash
from research.wedge_propagation.classification.model import PackedClassifier


def debug_graph(*, device="cpu", empty=False):
    n = 8
    pairs = [] if empty else [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)]
    neighbors = [[] for _ in range(n)]
    for u, v in pairs:
        neighbors[u].append(v)
        neighbors[v].append(u)
    triples = [(i, j, k) for j in range(n) for i, k in combinations(neighbors[j], 2)]
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    paths = torch.tensor(triples, dtype=torch.long).reshape(-1, 3).T.contiguous()
    b, a = torch.zeros(len(pairs), n).double(), torch.zeros(len(triples), n).double()
    if pairs:
        b[torch.arange(len(pairs)), edges[0]] = -1
        b[torch.arange(len(pairs)), edges[1]] = 1
    if triples:
        a[torch.arange(len(triples)), paths[0]] = 1
        a[torch.arange(len(triples)), paths[1]] = -2
        a[torch.arange(len(triples)), paths[2]] = 1
    degree, qdiag = b.square().sum(0), a.square().sum(0)
    sd = torch.where(degree > 0, degree, torch.ones_like(degree)).rsqrt() * (degree > 0)
    sq = torch.where(qdiag > 0, qdiag, torch.ones_like(qdiag)).rsqrt() * (qdiag > 0)
    x = torch.randn(n, 5, generator=torch.Generator().manual_seed(717)).double()
    graph = SimpleNamespace(x=x, edges=edges, paths=paths, degree=degree, qdiag=qdiag, sd=sd, sq=sq)
    for name, value in vars(graph).items():
        setattr(graph, name, value.to(device))
    return graph, b.to(device), a.to(device)


def source_model(condition="learned_wedge_raw", *, seeds=(11, 23), device="cpu"):
    net = (
        PackedClassifier(
            condition,
            5,
            3,
            seeds,
            hidden=6,
            gate_hidden=4,
            dropout=0.5,
            path_chunk=5,
            dataset_name="DEBUG-strength",
        )
        .double()
        .to(device)
        .eval()
    )
    with torch.no_grad():
        for gate in net.gates:
            gate.w1.mul_(1.7)
            gate.w2.mul_(3.0)
    return net


def dense_t(graph, a, z, c, denominator):
    q = torch.einsum("pn,sp,pm->snm", a, c, a)
    q = q * graph.sq[None, :, None] * graph.sq[None, None, :]
    return torch.bmm(q, z) / (3 * denominator[:, None, None])


def reference_fields(base, graph, layer, z):
    sigma = base._sigma(graph, z)
    c = base._gate(graph, layer, z, sigma)
    kappa = base._kappa(graph, c)
    return c, kappa


def dense_treatment(base, graph, a, layer, z, treatment, manifest):
    c, kappa = reference_fields(base, graph, layer, z)
    reference = dense_t(graph, a, z, c, kappa)
    gain = torch.ones_like(kappa)
    denominator = kappa
    if treatment == "baseline":
        candidate = reference
    elif treatment == "true_c_unit_denominator":
        denominator = torch.ones_like(kappa)
        candidate = dense_t(graph, a, z, c, denominator)
    elif treatment.startswith("identity_"):
        c = torch.ones_like(c)
        denominator = kappa if treatment == "identity_hold_reference" else torch.ones_like(kappa)
        candidate = dense_t(graph, a, z, c, denominator)
    elif treatment == "branch_off":
        candidate = torch.zeros_like(reference)
    else:
        c = c[:, manifest["permutation"]]
        candidate = dense_t(graph, a, z, c, denominator)
    if treatment.endswith("norm_matched"):
        gain = torch.linalg.vector_norm(reference, dim=(1, 2)) / torch.linalg.vector_norm(
            candidate, dim=(1, 2)
        )
        candidate = candidate * gain[:, None, None]
    return candidate, reference, gain, denominator


@pytest.mark.parametrize("condition", ["learned_wedge_raw", "learned_wedge_rms"])
@pytest.mark.parametrize("treatment", TREATMENTS)
def test_each_treatment_matches_independent_dense_formula(condition, treatment):
    graph, _, a = debug_graph()
    base = source_model(condition)
    wrapper = FrozenStrengthClassifier.from_classifier(base)
    manifest = {"permutation": torch.arange(graph.paths.shape[1]).flip(0)}
    z = torch.bmm(graph.x[None].expand(2, -1, -1), base.projections[0])
    with torch.inference_mode():
        expected, reference, gain, denominator = dense_treatment(
            base, graph, a, 0, z, treatment, manifest
        )
        actual, detail = wrapper.set_treatment(treatment, manifest=manifest).probe_layer(
            graph, 0, z
        )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(detail["reference_t_message"], reference, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(detail["normalization_gain"], gain, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(detail["kappa_used"], denominator, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("target", ["layer_0", "layer_1", "both"])
@pytest.mark.parametrize("treatment", TREATMENTS)
def test_forward_target_and_current_state_reference(target, treatment):
    graph, b, a = debug_graph()
    base = source_model()
    wrapper = FrozenStrengthClassifier.from_classifier(base)
    manifest = {"permutation": torch.arange(graph.paths.shape[1]).flip(0)}
    lap = 0.5 * graph.sd[:, None] * (b.T @ b) * graph.sd[None]
    with torch.inference_mode():
        output, details = wrapper.set_treatment(treatment, target, manifest)(
            graph, diagnostics=True
        )
        h = graph.x[None].expand(2, -1, -1)
        for layer in range(2):
            z = torch.bmm(h, base.projections[layer])
            active = treatment if target in ("both", f"layer_{layer}") else "baseline"
            candidate, reference, _, _ = dense_treatment(base, graph, a, layer, z, active, manifest)
            alpha, beta = base._coefficients(layer, z)
            u = z - alpha[:, None, None] * (lap @ z) - beta[:, None, None] * candidate
            torch.testing.assert_close(details[layer]["z"], z, rtol=2e-12, atol=2e-12)
            torch.testing.assert_close(
                details[layer]["reference_t_message"], reference, rtol=2e-12, atol=2e-12
            )
            h = u.relu() if layer == 0 else u
    torch.testing.assert_close(output, h, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("condition", ["learned_wedge_raw", "learned_wedge_rms"])
def test_baseline_and_prior_c_interventions_reproduce_original(condition):
    graph, _, _ = debug_graph()
    base = source_model(condition)
    wrapper = FrozenStrengthClassifier.from_classifier(base)
    manifest = {"permutation": torch.arange(graph.paths.shape[1]).flip(0)}
    comparisons = (
        ("baseline", None, "recompute"),
        ("identity_hold_reference", "c_identity", "hold"),
        ("identity_unit_denominator", "c_identity", "recompute"),
        ("branch_off", "second_branch_remove", "hold"),
        ("shuffle_hold_reference", "c_position_shuffle", "hold"),
    )
    with torch.inference_mode():
        for treatment, intervention, mode in comparisons:
            expected, _ = base(graph, intervention=intervention, kappa_mode=mode, manifest=manifest)
            actual, _ = wrapper.set_treatment(treatment, manifest=manifest)(graph)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_wrapper_parameters_and_original_state_remain_identical_and_frozen():
    graph, _, _ = debug_graph()
    base = source_model()
    before = model_state_hash(base)
    wrapper = FrozenStrengthClassifier.from_classifier(base)
    assert model_state_hash(wrapper) == before
    assert not any(parameter.requires_grad for parameter in wrapper.parameters())
    assert base.training is False
    assert all(parameter.requires_grad for parameter in base.parameters())
    with torch.inference_mode():
        wrapper.set_treatment("true_c_unit_denominator")(graph)
        wrapper.set_treatment("identity_norm_matched")(graph)
    assert model_state_hash(wrapper) == before
    assert model_state_hash(base) == before
    with pytest.raises(RuntimeError, match="training is disabled"):
        wrapper.train()


def test_strength_statistics_are_actual_coefficient_weighted_messages():
    graph, _, _ = debug_graph()
    base = source_model()
    wrapper = FrozenStrengthClassifier.from_classifier(base)
    with torch.inference_mode():
        _, detail = wrapper(graph, diagnostics=True)
    row = detail[0]

    def norm(value):
        return torch.linalg.vector_norm(value, dim=(1, 2))

    first = row["alpha"][:, None, None] * row["l_message"]
    second = row["beta"][:, None, None] * row["t_message"]
    torch.testing.assert_close(row["first_norm"], norm(first))
    torch.testing.assert_close(row["second_norm"], norm(second))
    torch.testing.assert_close(row["first_to_input"], norm(first) / norm(row["z"]))
    torch.testing.assert_close(row["second_to_first"], norm(second) / norm(first))
    torch.testing.assert_close(
        row["first_second_cosine"], (first * second).sum((1, 2)) / (norm(first) * norm(second))
    )
    torch.testing.assert_close(row["message_norm_ratio"], torch.ones(2).double())
    torch.testing.assert_close(row["message_relative_change"], torch.zeros(2).double())


def test_kappa_distribution_uses_only_nonzero_q_nodes_and_reports_original_node_id():
    graph, _, a = debug_graph()
    wrapper = FrozenStrengthClassifier.from_classifier(source_model())
    with torch.inference_mode():
        _, details = wrapper(graph, diagnostics=True)
    for detail in details:
        c = detail["c_reference"]
        diagonal = torch.einsum("pn,sp,pn->sn", a, c, a)
        active = graph.qdiag > 0
        ratio = diagonal[:, active] / graph.qdiag[active][None]
        maximum, index = ratio.max(1)
        torch.testing.assert_close(detail["kappa_ratio_max"], detail["kappa_reference"])
        torch.testing.assert_close(detail["kappa_ratio_mean"], ratio.mean(1))
        torch.testing.assert_close(detail["kappa_ratio_p90"], torch.quantile(ratio, 0.9, dim=1))
        assert torch.equal(detail["kappa_max_node"], active.nonzero().flatten()[index])
        torch.testing.assert_close(
            detail["kappa_near_max_fraction"], (ratio >= 0.99 * maximum[:, None]).double().mean(1)
        )
        assert torch.equal(detail["kappa_active_node_count"], active.sum().expand(2))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("treatment", TREATMENTS)
def test_zero_messages_preserve_finite_forward_and_undefined_diagnostics(empty, treatment):
    graph, _, _ = debug_graph(empty=empty)
    graph.x.zero_()
    wrapper = FrozenStrengthClassifier.from_classifier(source_model())
    manifest = {"permutation": torch.arange(graph.paths.shape[1]).flip(0)}
    with torch.inference_mode():
        output, details = wrapper.set_treatment(treatment, manifest=manifest)(
            graph, diagnostics=True
        )
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))
    for detail in details:
        assert torch.isnan(detail["second_to_input"]).all()
        assert not detail["second_to_input_defined"].any()
        assert torch.isnan(detail["message_reference_cosine"]).all()
        assert not detail["message_reference_cosine_defined"].any()
        assert torch.equal(detail["applied_gain"], torch.ones(2).double())
        if empty:
            assert not detail["kappa_ratio_defined"].any()
            assert torch.isnan(detail["kappa_ratio_mean"]).all()


def test_norm_matching_matches_each_seed_independently_and_preserves_direction():
    graph, _, _ = debug_graph()
    wrapper = FrozenStrengthClassifier.from_classifier(source_model())
    with torch.inference_mode():
        _, clean = wrapper(graph, diagnostics=True)
        wrapper.set_treatment("identity_norm_matched")
        z = clean[0]["z"]
        matched, stats = wrapper.probe_layer(graph, 0, z)
        raw, _ = wrapper.set_treatment("identity_unit_denominator").probe_layer(graph, 0, z)
    torch.testing.assert_close(stats["message_norm_ratio"], torch.ones(2).double())
    torch.testing.assert_close(
        matched / stats["applied_gain"][:, None, None], raw, rtol=1e-12, atol=1e-12
    )


def test_fixed_z_statistics_obeys_existing_treatment_and_only_returns_scalar_fields():
    graph, _, _ = debug_graph()
    wrapper = FrozenStrengthClassifier.from_classifier(source_model())
    with torch.inference_mode():
        _, clean = wrapper(graph, diagnostics=True)
        wrapper.set_treatment("identity_norm_matched")
        stats = fixed_z_statistics(wrapper, graph, 0, clean[0]["z"])
    assert all(value is None or value.shape == (2,) for value in stats.values())
    torch.testing.assert_close(stats["message_norm_ratio"], torch.ones(2).double())


@pytest.mark.parametrize(
    "condition", ["first_order", "polynomial_2", "fixed_wedge", "fixed_wedge_node_mlp"]
)
def test_control_fixed_z_statistics_use_original_branches(condition):
    graph, _, _ = debug_graph()
    control = source_model(condition)
    with torch.inference_mode():
        _, details = control(graph, diagnostics=True)
    stats = fixed_z_statistics(control, graph, 0, details[0]["z"])
    torch.testing.assert_close(stats["second_norm"], details[0]["beta"] * details[0]["branch_norm"])
    with pytest.raises(ValueError, match="only the baseline"):
        fixed_z_statistics(control, graph, 0, details[0]["z"], "branch_off")


def test_rejects_invalid_treatment_scope_and_shuffle_manifest():
    graph, _, _ = debug_graph()
    wrapper = FrozenStrengthClassifier.from_classifier(source_model())
    with pytest.raises(ValueError, match="unknown"):
        wrapper.set_treatment("pretend")
    with pytest.raises(ValueError, match="target"):
        wrapper.set_treatment("branch_off", "layer_9")
    with pytest.raises(ValueError, match="requires manifest"):
        wrapper.set_treatment("shuffle_norm_matched")
    wrapper.set_treatment("shuffle_hold_reference", manifest={"permutation": torch.zeros(1).long()})
    with torch.inference_mode(), pytest.raises(ValueError, match=r"Long\[P\]"):
        wrapper(graph)
    with pytest.raises(ValueError, match="learned C"):
        FrozenStrengthClassifier.from_classifier(source_model("fixed_wedge"))


def test_static_positive_node_cache_invalidates_mutation_and_releases_source():
    qdiag = torch.tensor([0.0, 1.0, 0.0, 2.0])
    initial = _active_q_indices(qdiag)
    assert initial is _active_q_indices(qdiag)
    assert torch.equal(initial, torch.tensor([1, 3]))
    qdiag[0] = 3.0
    changed = _active_q_indices(qdiag)
    assert changed is not initial
    assert torch.equal(changed, torch.tensor([0, 1, 3]))
    reference = weakref.ref(qdiag)
    del qdiag
    gc.collect()
    assert reference() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA DEBUG requires available GPU")
def test_cuda_full_scope_and_norm_matching_have_cpu_equivalent_messages():
    graph, _, _ = debug_graph()
    base = source_model()
    cpu = FrozenStrengthClassifier.from_classifier(base)
    cuda_graph, _, _ = debug_graph(device="cuda")
    cuda = FrozenStrengthClassifier.from_classifier(base.to("cuda"))
    with torch.inference_mode():
        expected, expected_details = cpu.set_treatment("identity_norm_matched", "both")(graph)
        actual, actual_details = cuda.set_treatment("identity_norm_matched", "both")(cuda_graph)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-10, atol=2e-10)
    for actual_detail, expected_detail in zip(actual_details, expected_details, strict=True):
        torch.testing.assert_close(
            actual_detail["message_norm_ratio"].cpu(),
            expected_detail["message_norm_ratio"],
            rtol=2e-10,
            atol=2e-10,
        )
