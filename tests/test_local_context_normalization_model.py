"""DEBUG algebra, learned-gain linkage and locked frozen normalization probes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from research.local_context_coupling.classification.model import PackedClassifier as LegacyClassifier
from research.local_context_coupling.normalization.common import CONDITIONS, condition_metadata, parse_condition
from research.local_context_coupling.normalization.evaluation import frozen_evaluate, intervention_scopes
from research.local_context_coupling.normalization.model import PackedClassifier
from research.local_context_coupling.normalization.operators import (
    apply_s, apply_g, energy_s, energy_g, immediate, mixed_action,
    prepare_normalized_geometry, sandwich,
)
from research.local_context_coupling.normalization.verify import dense_reference
from research.local_context_coupling.operators import prepare_geometry, replicate, merge, sandwich as legacy_sandwich
from research.local_energy_relations.topology import build_topology, batch_topologies
from research.wedge_propagation.classification.evaluation import model_state_hash


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


PAIRS = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]


def topology(n=7, pairs=PAIRS):
    return build_topology(n, np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T.copy())


def graph(dtype=torch.float64, device="cpu", zero=False):
    top = topology()
    bases = {mode: prepare_geometry(top, mode) for mode in ("unit", "local_degree")}
    geometries = {(mode, intra, cross): prepare_normalized_geometry(base, intra, cross).to(device, dtype)
                  for mode, base in bases.items() for intra in ("graph", "local") for cross in ("graph", "edge")}
    x = torch.randn(7, 6, dtype=dtype, generator=torch.Generator().manual_seed(432)).to(device)
    if zero:
        x.zero_()
    data = SimpleNamespace(
        name="DEBUG_norm_nonidentical_locals", x=x, geometries=geometries,
        y=torch.tensor([0, 1, 2, 1, 2, 0, 1], dtype=torch.long, device=device),
        train_mask=torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.bool, device=device),
        val_mask=torch.tensor([0, 0, 0, 1, 1, 0, 0], dtype=torch.bool, device=device),
        test_mask=torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.bool, device=device),
        num_nodes=7, num_features=6, num_classes=3,
    )
    data.geometry_for = lambda mode, intra, cross: geometries[mode, intra, "graph" if cross == "none" else cross]
    return data


def model(condition, data, seeds=(0, 3), **kwargs):
    return PackedClassifier(condition, 6, 3, seeds, dataset_name=data.name, **kwargs).to(device=data.x.device, dtype=data.x.dtype)


def _act(matrix, value):
    return torch.einsum("ij,...jf->...if", matrix, value)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("intra", ["graph", "local"])
@pytest.mark.parametrize("cross", ["graph", "edge"])
@pytest.mark.parametrize("packed", [False, True])
def test_sparse_operators_energy_gradients_and_bounds_against_independent_dense(mode, intra, cross, packed):
    geometry = prepare_normalized_geometry(prepare_geometry(topology(), mode), intra, cross)
    r, m, s, g = dense_reference(7, PAIRS, mode, intra, cross)
    value = torch.randn((2, geometry.num_copies, 4), dtype=torch.float64, generator=torch.Generator().manual_seed(23))
    if not packed:
        value = value[0]
    value.requires_grad_()
    torch.testing.assert_close(apply_s(geometry, value, 2), _act(s, value), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(apply_g(geometry, value, 3), _act(g, value), rtol=1e-13, atol=1e-13)
    expected_s = .5 * (value * _act(s, value)).sum(-2)
    expected_g = .5 * (value * _act(g, value)).sum(-2)
    torch.testing.assert_close(energy_s(geometry, value, 2)[..., 0, :], expected_s, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(energy_g(geometry, value, 2)[..., 0, :], expected_g, rtol=1e-12, atol=1e-12)
    grad = torch.autograd.grad((energy_s(geometry, value, 2) + .7 * energy_g(geometry, value, 2)).sum(), value)[0]
    torch.testing.assert_close(grad, _act(s + .7 * g, value), rtol=1e-12, atol=1e-12)
    for matrix in (s, g):
        torch.testing.assert_close(matrix, matrix.T, rtol=0, atol=0)
        eigenvalues = torch.linalg.eigvalsh(matrix)
        assert eigenvalues.min() >= -1e-12 and eigenvalues.max() <= 1 + 1e-12
        assert matrix.diagonal().max() <= .5 + 1e-14
    torch.testing.assert_close(g @ r, torch.zeros_like(r), rtol=0, atol=1e-15)
    torch.testing.assert_close(m @ g, torch.zeros_like(m), rtol=0, atol=1e-15)
    assert geometry.metadata["energy_operator"] == "applied_S_G"
    assert geometry.num_cross_edges == geometry.base.num_cross_edges


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("intra", ["graph", "local"])
@pytest.mark.parametrize("cross", ["graph", "edge"])
def test_exact_mixed_difference_nonzero_guarantee_and_D_contraction(mode, intra, cross):
    geometry = prepare_normalized_geometry(prepare_geometry(topology(), mode), intra, cross)
    r, m, s, g = dense_reference(7, PAIRS, mode, intra, cross)
    x = torch.randn(2, 7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(17), requires_grad=True)
    rho = torch.tensor([.25, .7], dtype=torch.float64, requires_grad=True)
    on, stages = sandwich(geometry, x, cross_gain=rho, diagnostics=True, edge_chunk=2)
    off = sandwich(geometry, x, cross=False, edge_chunk=3)
    predicted = -rho[:, None, None] * mixed_action(geometry, x, 2)
    torch.testing.assert_close(on - off, predicted, rtol=1e-10, atol=1e-14)
    assert apply_g(geometry, stages["after_intra_1"]).norm() > 0 and predicted.norm() > 0
    dense_mixed = m @ s @ g @ s @ r
    torch.testing.assert_close(mixed_action(geometry, x, 2), _act(dense_mixed, x), rtol=1e-12, atol=1e-13)
    grad_actual = torch.autograd.grad(on.square().sum(), (x, rho), retain_graph=True)
    p = torch.eye(s.shape[0], dtype=s.dtype) - s
    expected = torch.stack([_act(m @ p @ (torch.eye(s.shape[0]) - gain * g) @ p @ r, x[index]) for index, gain in enumerate(rho)])
    grad_expected = torch.autograd.grad(expected.square().sum(), (x, rho), retain_graph=True)
    for actual, reference in zip(grad_actual, grad_expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-12)
    # Physical merge is contractive in the copy-count D metric, not an assumed Euclidean metric.
    d = r.T @ r
    t = m @ p @ (torch.eye(s.shape[0]) - .7 * g) @ p @ r
    torch.testing.assert_close(d @ t, (d @ t).T, rtol=1e-12, atol=1e-12)
    coordinate = d.diagonal().sqrt()
    sym = coordinate[:, None] * t / coordinate[None, :]
    eigenvalues = torch.linalg.eigvalsh(sym)
    assert eigenvalues.min() >= -1e-12 and eigenvalues.max() <= 1 + 1e-12
    immediate_on = immediate(geometry, x, cross_gain=rho, edge_chunk=2)
    immediate_off = immediate(geometry, x, cross=False, edge_chunk=3)
    torch.testing.assert_close(immediate_on, immediate_off, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("intra", ["graph", "local"])
@pytest.mark.parametrize("cross", ["graph", "edge"])
@pytest.mark.parametrize("case", ["constant", "clique", "empty"])
def test_initial_consensus_and_constant_clique_empty_null_controls(mode, intra, cross, case):
    pairs = [] if case == "empty" else [(i, j) for i in range(4) for j in range(i + 1, 4)] if case == "clique" else [(0, 1), (1, 2), (2, 3)]
    geometry = prepare_normalized_geometry(prepare_geometry(topology(4, pairs), mode), intra, cross)
    x = torch.randn(2, 4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(54))
    if case == "constant":
        x = x[:, :1].expand(-1, 4, -1).clone()
    assert apply_g(geometry, replicate(geometry.base, x)).eq(0).all()
    on, stages = sandwich(geometry, x, diagnostics=True, edge_chunk=2)
    off = sandwich(geometry, x, cross=False, edge_chunk=3)
    torch.testing.assert_close(apply_g(geometry, stages["after_intra_1"]), torch.zeros_like(stages["after_intra_1"]), rtol=0, atol=1e-15)
    torch.testing.assert_close(on, off, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("cross", ["graph", "edge"])
@pytest.mark.parametrize("pairs,n", [([(0, 1), (1, 2), (2, 3)], 4), ([(0, 1), (0, 2), (0, 3), (0, 4)], 6), ([], 4)])
def test_triangle_free_local_S_is_identical_for_two_fixed_C_modes(cross, pairs, n):
    top = topology(n, pairs)
    a = prepare_normalized_geometry(prepare_geometry(top, "unit"), "local", cross)
    b = prepare_normalized_geometry(prepare_geometry(top, "local_degree"), "local", cross)
    # Every nonempty induced ego is a star; its local C is one common scalar.
    torch.testing.assert_close(a.s_weights, b.s_weights, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(a.g_weights, b.g_weights, rtol=0, atol=0)
    x = torch.randn(2, n, 3, dtype=torch.float64)
    torch.testing.assert_close(sandwich(a, x), sandwich(b, x), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_graph_graph_reproduces_legacy_float64_macro_values_and_derivatives(mode):
    base = prepare_geometry(topology(), mode)
    geometry = prepare_normalized_geometry(base, "graph", "graph")
    x = torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([.19, .82], dtype=torch.float64, requires_grad=True)
    actual = sandwich(geometry, x, cross_gain=rho, edge_chunk=3)
    expected = legacy_sandwich(base, x, cross_gain=rho)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    ga = torch.autograd.grad(actual.square().sum(), (x, rho), retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), (x, rho))
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)


def test_disjoint_batch_uses_each_graph_local_steps_and_canonical_links():
    left, right = topology(), topology(4, [(0, 1), (0, 2), (0, 3)])
    batch = batch_topologies([left, right])
    whole = prepare_normalized_geometry(prepare_geometry(batch, "local_degree"), "local", "edge")
    singles = [prepare_normalized_geometry(prepare_geometry(top, "local_degree"), "local", "edge") for top in (left, right)]
    x = torch.randn(2, 11, 3, dtype=torch.float64)
    expected = torch.cat([sandwich(geo, value) for geo, value in zip(singles, x.split([7, 4], 1), strict=True)], 1)
    actual = sandwich(whole, x, edge_chunk=3)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    assert whole.num_cross_edges == sum(geo.num_cross_edges for geo in singles)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_full_classifier_gradient_optimizer_and_same_named_initialization(condition):
    data = graph()
    candidate = model(condition, data, edge_chunk=2)
    mode, intra, cross, variant = parse_condition(condition)
    reference = LegacyClassifier(f"{mode}__{variant}", 6, 3, (0, 3), dataset_name=data.name).to(torch.float64)
    assert candidate.parameters_per_seed == 6 * 64 + 64 * 3 + int(variant == "learned")
    assert (candidate.theta_cross is not None) == (variant == "learned")
    for a, b in zip(candidate.weights, reference.weights, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(candidate.dropout_keys, reference.dropout_keys, rtol=0, atol=0)
    logits, _ = candidate(data, epoch=2)
    loss = F.cross_entropy(logits[:, data.train_mask].reshape(-1, 3), data.y[data.train_mask].repeat(2))
    loss.backward()
    before = {key: value.detach().clone() for key, value in candidate.named_parameters()}
    for parameter in candidate.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer = torch.optim.Adam(candidate.weight_decay_groups(.0005), lr=.01)
    optimizer.step()
    assert all(not torch.equal(before[key], value) for key, value in candidate.named_parameters())


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("variant", ["off", "fixed", "learned"])
def test_graph_graph_two_layer_classifier_matches_legacy_value_and_all_parameter_gradients(mode, variant):
    data = graph()
    cross = "none" if variant == "off" else "graph"
    candidate = model(f"{mode}__graph__{cross}__{variant}", data, edge_chunk=2)
    base = data.geometry_for(mode, "graph", "graph").base
    reference = LegacyClassifier(f"{mode}__{variant}", 6, 3, (0, 3), dataset_name=data.name, path_chunk=3).to(torch.float64)
    actual = candidate(data, epoch=4)[0]
    expected = reference((data.x, base), epoch=4)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    actual.square().sum().backward(); expected.square().sum().backward()
    for name, value in candidate.named_parameters():
        torch.testing.assert_close(value.grad, dict(reference.named_parameters())[name].grad, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_frozen_gain_probe_coverage_metadata_and_parameter_preservation(condition):
    data = graph()
    candidate = model(condition, data, edge_chunk=2)
    before = model_state_hash(candidate)
    result = frozen_evaluate(candidate, data, candidate.seeds, {"evaluation": {"intervention_scopes": ["layer_0", "layer_1", "both"]}})
    assert candidate.training and model_state_hash(candidate) == before
    enabled = candidate.variant != "off"
    assert len(result["metric_rows"]) == 6
    assert len(result["intervention_rows"]) == (36 if enabled else 0)
    assert len(result["branch_rows"]) == (28 if enabled else 4)
    assert intervention_scopes(condition) == (("layer_0", "layer_1", "both") if enabled else ())
    assert result["provenance"]["optimizer_updates"] == 0
    for kind in ("metric_rows", "intervention_rows", "branch_rows"):
        for row in result[kind]:
            for key, value in condition_metadata(condition).items():
                assert row[key] == value
    for row in result["branch_rows"]:
        assert row["formula_relative_error"] < 1e-12
        assert row["s_degree_max"] <= .5 + 1e-14 and row["g_degree_max"] <= .5 + 1e-14
        assert row["raw_context_norm"] >= 0 and row["context_norm"] >= 0


def test_gain0_retains_matched_intra_baseline_and_recomputes_second_layer():
    data = graph()
    learned = model("unit__local__edge__learned", data, edge_chunk=2)
    off = model("unit__local__none__off", data, edge_chunk=2)
    actual = learned(data, epoch=7, intervention="gain0")[0]
    expected = off(data, epoch=7)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed, details = learned(data, epoch=7, intervention="gain0", intervention_layers=(0,), diagnostics=True)
    torch.testing.assert_close(details[0]["rho"], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(details[1]["rho"], learned.cross_gain)
    assert details[0]["matched_delta_norm"].eq(0).all()
    assert details[1]["matched_delta_norm"].gt(0).all()
    geo = data.geometry_for("unit", "local", "edge")
    z0 = torch.bmm(learned._dropout(data.x.unsqueeze(0).expand(2, -1, -1), 7, 0), learned.weights[0])
    h0 = sandwich(geo, z0, cross=False, edge_chunk=2).relu()
    z1 = torch.bmm(learned._dropout(h0, 7, 1), learned.weights[1])
    manual = sandwich(geo, z1, cross_gain=learned.cross_gain, edge_chunk=2)
    torch.testing.assert_close(changed, manual, rtol=0, atol=0)


def test_zero_input_ratios_null_and_no_labels_enter_forward():
    data = graph(zero=True)
    candidate = model("unit__local__edge__learned", data)
    result = frozen_evaluate(candidate, data, candidate.seeds, {"evaluation": {"intervention_scopes": ["layer_0", "layer_1", "both"]}})
    assert all(row["matched_delta_relative"] is None and row["formula_relative_error"] is None for row in result["branch_rows"])
    unlabeled = SimpleNamespace(x=data.x, geometry_for=data.geometry_for)
    assert candidate(unlabeled)[0].shape == (2, 7, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("intra", ["graph", "local"])
@pytest.mark.parametrize("cross", ["graph", "edge"])
def test_float32_cuda_chunk_checkpoint_matches_cpu_full_forward_and_gradient(mode, intra, cross):
    cpu, gpu = graph(torch.float32), graph(torch.float32, "cuda")
    condition = f"{mode}__{intra}__{cross}__learned"
    a, b = model(condition, cpu, edge_chunk=2), model(condition, gpu, edge_chunk=3)
    expected, actual = a(cpu, epoch=9)[0], b(gpu, epoch=9)[0]
    expected.square().sum().backward(); actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    for name, parameter in b.named_parameters():
        torch.testing.assert_close(parameter.grad.cpu(), dict(a.named_parameters())[name].grad, rtol=5e-5, atol=5e-6)
