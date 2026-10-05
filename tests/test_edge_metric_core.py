"""DEBUG fixtures for the complete occurrence edge-metric candidate."""
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from research.edge_metric_relations.geometry import prepare_geometry, prepare_geometries, with_recipe
from research.edge_metric_relations.gates import (
    CONDITIONS, EPSILON, EdgeMetricGate, analytic_teacher, pair_features, physical_edge_features,
)
from research.edge_metric_relations.operators import apply_metric, diagonal_action, incidence
from research.local_energy_relations.topology import batch_topologies, build_topology


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


PAIRS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]


def top(n=6, pairs=PAIRS):
    return build_topology(n, np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T)


def geo(recipe="unit", device="cpu", dtype=torch.float64):
    return prepare_geometry(top(), recipe).to(device, dtype)


def random(shape, seed=214, device="cpu", dtype=torch.float64):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype).to(device)


def gate(condition, seeds=(11, 23), device="cpu", dtype=torch.float64, active=True, namespace="DEBUG.layer_0"):
    model = EdgeMetricGate(condition, seeds, namespace=namespace).to(device=device, dtype=dtype)
    if active:
        with torch.no_grad():
            for i, module in enumerate((model.diagonal_gate, model.pair_gate)):
                if module is not None:
                    module.w2.copy_(random(module.w2.shape, seed=731 + i, device=device, dtype=dtype) * .08)
                    module.b2.fill_(.13)
    return model


def dense_metric(g, coefficients):
    b = g.c0.new_zeros((g.num_edges, g.n))
    indices = torch.arange(g.num_edges, device=b.device)
    b[indices, g.edges[0]], b[indices, g.edges[1]] = -1., 1.
    u = b.new_zeros((g.num_occurrences, g.num_edges))
    u[torch.arange(g.num_occurrences), g.occ_edge] = g.occ_scale
    calb = u @ b
    a = coefficients["diagonal_multiplier"]
    d = g.c0 * a.index_select(-1, g.occ_edge)
    k = b.new_zeros(a.shape[:-1] + (g.num_occurrences, g.num_occurrences))
    k[..., g.pair_left, g.pair_right] = coefficients["pair_coefficient"]
    k[..., g.pair_right, g.pair_left] = coefficients["pair_coefficient"]
    identity = torch.eye(g.num_occurrences, dtype=b.dtype, device=b.device)
    c = d.sqrt().unsqueeze(-1) * (identity + .5 * k) * d.sqrt().unsqueeze(-2)
    return calb.T @ c @ calb, c, calb, k


def test_exact_eligible_support_and_no_same_physical_edge():
    g = geo()
    expected = set()
    physical = {frozenset(e.tolist()) for e in g.edges.T}
    for r in range(g.num_occurrences):
        for s in range(r + 1, g.num_occurrences):
            v, u = int(g.occ_center[r]), int(g.occ_center[s])
            e, f = int(g.occ_edge[r]), int(g.occ_edge[s])
            if v == u or frozenset((v, u)) not in physical or e == f:
                continue
            common = set(g.edges[:, e].tolist()) & set(g.edges[:, f].tolist())
            if common:
                expected.add((min(r, s), max(r, s), next(iter(common))))
    observed = {(min(int(r), int(s)), max(int(r), int(s)), int(a))
                for r, s, a in zip(g.pair_left, g.pair_right, g.pair_anchor)}
    assert observed == expected
    assert len(observed) == g.num_pairs
    assert torch.equal(g.pair_degree, torch.bincount(torch.cat((g.pair_left, g.pair_right)), minlength=g.num_occurrences))
    assert g.metadata["all_eligible_pairs_retained"]
    assert torch.equal(g.occ_count, torch.bincount(g.occ_edge, minlength=g.num_edges).double())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_shared_recipe_preparation_and_transfer_match_independent_full_geometry(device):
    topology = batch_topologies([top(), top(4, [(0, 1)]), top(3, [])])
    geometries = prepare_geometries(topology)
    unit, degree = geometries["unit"], geometries["local_degree"]
    independent = prepare_geometry(topology, "local_degree")
    assert degree.metadata == independent.metadata
    assert degree.topology is unit.topology
    for field in fields(degree):
        actual, expected = getattr(degree, field.name), getattr(independent, field.name)
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if field.name not in ("c0", "cbar0", "d0", "n0"):
                assert actual is getattr(unit, field.name)
    recovered = with_recipe(degree, "unit")
    for name in ("c0", "cbar0", "d0", "n0"):
        torch.testing.assert_close(getattr(recovered, name), getattr(unit, name), atol=0, rtol=0)
    assert recovered.metadata == unit.metadata
    transferred_unit = unit.to(device, torch.float32)
    shared_degree = degree.to(device, torch.float32, shared=transferred_unit)
    independently_transferred = independent.to(device, torch.float32)
    assert shared_degree.topology is transferred_unit.topology
    for field in fields(shared_degree):
        actual, expected = getattr(shared_degree, field.name), getattr(independently_transferred, field.name)
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if field.name not in ("c0", "cbar0", "d0", "n0"):
                assert actual is getattr(transferred_unit, field.name)
    assert shared_degree.metadata == independent.metadata
    values = random((2, shared_degree.n, 4), device=device, dtype=torch.float32)
    for condition in ("D0", "F0", "F2", "DA"):
        instance = gate(condition, device=device, dtype=torch.float32)
        observed, _ = instance.apply(shared_degree, values, values, pair_chunk=7)
        expected, _ = instance.apply(independently_transferred, values, values, pair_chunk=7)
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("recipe", ["unit", "local_degree"])
@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("realizations", [False, True])
def test_dense_sparse_psd_energy_and_effective_diagonal(recipe, condition, realizations):
    g, model = geo(recipe), gate(condition)
    shape = (2, 3, g.n, 4) if realizations else (2, g.n, 4)
    z, value = random(shape), random(shape, seed=931)
    coefficients = model.coefficients(g, z, pair_chunk=3)
    q, c, calb, k = dense_metric(g, coefficients)
    output, details = model.apply(g, z, value, pair_chunk=4, diagnostics=True)
    expected = q @ value
    torch.testing.assert_close(output, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(apply_metric(g, value, coefficients["diagonal_multiplier"], coefficients["pair_coefficient"], pair_chunk=2), output)
    energy = .5 * (value * output).sum((-2, -1))
    torch.testing.assert_close(details["energy_total"].sum(-1), energy)
    torch.testing.assert_close(details["corrected_local_energy"].sum(-1), details["energy_intra"].sum(-1))
    torch.testing.assert_close(details["message_diag"] + details["message_cross"], output)
    assert torch.linalg.eigvalsh(c).min() > 0
    assert torch.linalg.eigvalsh(q).min() > -2e-12
    assert torch.linalg.matrix_norm(k, ord=2).max() <= 1 + 1e-12
    u = calb.new_zeros((g.num_occurrences, g.num_edges))
    u[torch.arange(g.num_occurrences), g.occ_edge] = g.occ_scale
    effective = u.T @ c @ u
    torch.testing.assert_close(effective.diagonal(dim1=-2, dim2=-1), g.cbar0 * coefficients["diagonal_multiplier"])
    qdiag = dense_metric(g, {**coefficients, "pair_coefficient": torch.zeros_like(coefficients["pair_coefficient"])})[0]
    assert torch.linalg.eigvalsh(q - .5 * qdiag).min() > -2e-12
    assert torch.linalg.eigvalsh(1.5 * qdiag - q).min() > -2e-12
    normalized = g.n0[:, None] * q * g.n0[None, :]
    p = torch.eye(g.n, dtype=q.dtype) - normalized / 3
    assert torch.linalg.matrix_norm(p, ord=2).max() <= 1 + 1e-12


@pytest.mark.parametrize("condition", CONDITIONS)
def test_orientation_pair_swap_and_node_permutation(condition):
    g, model = geo(), gate(condition)
    z = random((2, g.n, 5))
    out = model.apply(g, z, z, pair_chunk=3)[0]
    signs = torch.tensor([-1., 1., -1., 1., 1., -1.], dtype=z.dtype)
    new_edges = torch.where(signs[None] < 0, g.edges.flip(0), g.edges)
    factor = signs[g.occ_edge[g.pair_left]] * signs[g.occ_edge[g.pair_right]]
    flipped = replace(g, edges=new_edges, pair_sign=g.pair_sign * factor)
    torch.testing.assert_close(model.apply(flipped, z, z, pair_chunk=4)[0], out)
    swapped = replace(g, pair_left=g.pair_right, pair_right=g.pair_left,
                      pair_other_left=g.pair_other_right, pair_other_right=g.pair_other_left)
    torch.testing.assert_close(pair_features(swapped, z), pair_features(g, z))
    torch.testing.assert_close(model.apply(swapped, z, z, pair_chunk=2)[0], out)
    perm = torch.tensor([4, 2, 5, 1, 0, 3])
    newtop = build_topology(g.n, perm[g.edges].numpy())
    pg = prepare_geometry(newtop).to("cpu", z.dtype)
    pz = torch.zeros_like(z).index_copy(-2, perm, z)
    pout = model.apply(pg, pz, pz, pair_chunk=5)[0]
    torch.testing.assert_close(pout.index_select(-2, perm), out)


def test_off_equality_initialization_parameters_and_layer_independence():
    g, z = geo(), random((2, 6, 4))
    expected = diagonal_action(g, z)
    models = {condition: gate(condition, active=False) for condition in CONDITIONS}
    counts = {"D0": 0, "D1": 385, "F0": 0, "F1": 641, "F2": 1026, "DA": 1026}
    for condition, model in models.items():
        assert model.parameters_per_seed == counts[condition]
        if condition != "F0":
            torch.testing.assert_close(model.apply(g, z, z)[0], expected)
    for name, parameter in models["F2"].named_parameters():
        torch.testing.assert_close(parameter, dict(models["DA"].named_parameters())[name])
    assert models["D0"].diagonal_gate is None and models["F0"].pair_gate is None
    other = gate("F2", active=False, namespace="DEBUG.layer_1")
    assert not torch.equal(models["F2"].pair_gate.w1, other.pair_gate.w1)
    assert not torch.equal(models["F2"].pair_gate.w1[0], models["F2"].pair_gate.w1[1])


def test_da_uses_normalized_pair_row_sum_then_all_occurrence_mean():
    g, model = geo(), gate("DA", active=False)
    z = random((2, 4, g.n, 1))
    raw = torch.full(z.shape[:-2] + (g.num_pairs,), .8, dtype=z.dtype)
    coefficients = model.coefficients(g, z, pair_chunk=2, pair_override=raw)
    row = z.new_zeros(z.shape[:-2] + (g.num_occurrences,))
    coeff = raw * g.pair_norm
    row = row.index_add(-1, g.pair_left, coeff).index_add(-1, g.pair_right, coeff)
    mean = z.new_zeros(z.shape[:-2] + (g.num_edges,)).index_add(-1, g.occ_edge, row) / g.occ_count
    expected = (math_log2() * mean.tanh()).exp()
    torch.testing.assert_close(coefficients["diagonal_multiplier"], expected)
    assert not coefficients["cross_active"]
    assert torch.count_nonzero(coefficients["pair_coefficient"]) == 0
    torch.testing.assert_close(model.apply(g, z, z, pair_override=raw, pair_chunk=3)[0], diagonal_action(g, z, expected))


def math_log2():
    return np.log(2.)


@pytest.mark.parametrize("condition", ["D1", "F1", "F2", "DA"])
def test_actual_parameter_loss_backward_optimizer_and_chunk_gradients(condition):
    g, model = geo(), gate(condition)
    z = random((2, g.n, 3)).requires_grad_()
    output = model.apply(g, z, z, pair_chunk=3)[0]
    (output - random(output.shape, seed=611)).square().sum().backward()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert all(grad.abs().sum() > 0 for grad in grads.values())
    model.zero_grad(set_to_none=True)
    z2 = z.detach().clone().requires_grad_()
    output2 = model.apply(g, z2, z2, pair_chunk=7)[0]
    (output2 - random(output2.shape, seed=611)).square().sum().backward()
    torch.testing.assert_close(output2, output.detach())
    torch.testing.assert_close(z2.grad, z.grad)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, grads[name], atol=2e-10, rtol=2e-10)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}
    optimizer.step()
    assert all(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())


@pytest.mark.parametrize("condition", ["D1", "F1", "F2", "DA"])
def test_sparse_generator_gradients_match_independent_dense_metric(condition):
    g, model = geo(), gate(condition)
    z = random((2, g.n, 3)).requires_grad_()
    probe = random(z.shape, seed=811)
    applied = z * g.n0[:, None]
    sparse = model.apply(g, z, applied, pair_chunk=3)[0]
    variables = (z, *tuple(model.parameters()))
    sg = torch.autograd.grad((sparse * probe).sum(), variables)
    snapshot = model.coefficients(g, z, pair_chunk=7)
    q = dense_metric(g, snapshot)[0]
    dense = q @ (z * g.n0[:, None])
    dg = torch.autograd.grad((dense * probe).sum(), variables)
    torch.testing.assert_close(sparse, dense, atol=2e-12, rtol=2e-12)
    for sparse_gradient, dense_gradient in zip(sg, dg):
        torch.testing.assert_close(sparse_gradient, dense_gradient, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("condition", ["F1", "F2", "DA"])
def test_feature_input_gradcheck_and_smoothed_zero(condition):
    g = prepare_geometry(top(3, [(0, 1), (1, 2)])).to("cpu", torch.float64)
    model = gate(condition, seeds=(11,))
    z = random((1, g.n, 2)).requires_grad_()
    assert torch.autograd.gradcheck(lambda t: model.apply(g, t, t, pair_chunk=2)[0], (z,), fast_mode=True, eps=1e-6, atol=2e-5, rtol=2e-4)
    for constant in (0., 1.):
        zz = torch.full((1, g.n, 2), constant, dtype=z.dtype, requires_grad=True)
        value = zz * g.n0[:, None]
        output = model.apply(g, zz, value, pair_chunk=1)[0]
        output.square().sum().backward()
        assert torch.isfinite(output).all() and torch.isfinite(zz.grad).all()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        perturb = zz.detach() + random(zz.shape) * 1e-9
        nearby = model.apply(g, perturb, perturb * g.n0[:, None], pair_chunk=3)[0]
        assert (nearby - output.detach()).abs().max() < 1e-6
    assert torch.equal(pair_features(g, torch.zeros_like(z))[..., :4], torch.zeros_like(pair_features(g, z)[..., :4]))
    assert torch.equal(physical_edge_features(g, torch.zeros_like(z)), torch.zeros_like(physical_edge_features(g, z)))


def test_realization_and_seed_packing_are_independent():
    g, z = geo(), random((2, 4, 6, 1))
    model = gate("F2")
    packed = model.apply(g, z, z, pair_chunk=3)[0]
    for index in range(4):
        torch.testing.assert_close(model.apply(g, z[:, index], z[:, index], pair_chunk=4)[0], packed[:, index])
    for index, seed in enumerate(model.seeds):
        single = gate("F2", (seed,))
        for name, parameter in single.named_parameters():
            parameter.data.copy_(dict(model.named_parameters())[name].data[index:index + 1])
        torch.testing.assert_close(single.apply(g, z[index:index + 1], z[index:index + 1])[0], packed[index:index + 1])


def test_disjoint_graph_batch_preserves_all_actions_and_graph_energies():
    ta, tb = top(), top(4, [(0, 1), (1, 2), (2, 3)])
    ga, gb = prepare_geometry(ta), prepare_geometry(tb)
    batch = prepare_geometry(batch_topologies([ta, tb]))
    model = gate("F2")
    za, zb = random((2, ta.n, 3)), random((2, tb.n, 3), seed=38)
    oa, da = model.apply(ga, za, za, diagnostics=True)
    ob, db = model.apply(gb, zb, zb, diagnostics=True)
    output, details = model.apply(batch, torch.cat((za, zb), -2), torch.cat((za, zb), -2), pair_chunk=4, diagnostics=True)
    torch.testing.assert_close(output, torch.cat((oa, ob), -2))
    torch.testing.assert_close(details["energy_total"], torch.cat((da["energy_total"], db["energy_total"]), -1))
    assert batch.num_pairs == ga.num_pairs + gb.num_pairs


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("pairs", [[], [(0, 1)], [(0, 1), (1, 2)]])
def test_isolates_no_eligible_pairs_and_constants(condition, pairs):
    g = prepare_geometry(top(4, pairs))
    model = gate(condition, seeds=(11,))
    z = torch.ones((4, 2), dtype=torch.float64)
    output, details = model.apply(g, z, z, diagnostics=True, pair_chunk=2)
    torch.testing.assert_close(output, torch.zeros_like(output))
    assert torch.isfinite(output).all() and details["eligible_pair_count"] == g.num_pairs
    if len(pairs) < 2:
        assert g.num_pairs == 0 and details["pair_mean"] is None


def test_analytic_teacher_and_frozen_interventions_do_not_mutate_parameters():
    g, z = geo(), random((2, 3, 6, 1))
    feature = pair_features(g, z)
    raw = (2 * feature[..., 2] + .5 * (2 * feature[..., 7] - 1)).tanh()
    expected = gate("F0").apply(g, z, z, pair_override=raw)[0]
    torch.testing.assert_close(analytic_teacher(g, z, pair_chunk=2)[0], expected)
    model = gate("F2")
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    c = model.coefficients(g, z)
    off = model.apply(g, z, z, intervention="offdiag_zero", diagnostics=True)[0]
    torch.testing.assert_close(off, diagonal_action(g, z, c["diagonal_multiplier"]))
    one = model.coefficients(g, z, intervention="diagonal_one")
    torch.testing.assert_close(one["diagonal_multiplier"], torch.ones_like(one["diagonal_multiplier"]))
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_value_and_backward_agree_with_float64(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU tests retain full fixtures")
    gd, gf = geo(device=device), geo(device=device, dtype=torch.float32)
    md, mf = gate("F2", device=device), gate("F2", device=device, dtype=torch.float32)
    mf.load_state_dict(md.state_dict())
    zd = random((2, gd.n, 4), device=device).requires_grad_()
    zf = zd.detach().float().requires_grad_()
    od = md.apply(gd, zd, zd, pair_chunk=3)[0]
    of = mf.apply(gf, zf, zf, pair_chunk=4)[0]
    od.square().sum().backward(); of.square().sum().backward()
    torch.testing.assert_close(of.double(), od, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(zf.grad.double(), zd.grad, atol=1e-5, rtol=1e-4)
    for pd, pf in zip(md.parameters(), mf.parameters()):
        torch.testing.assert_close(pf.grad.double(), pd.grad, atol=3e-5, rtol=3e-4)


def test_invalid_gate_or_override_contracts_fail():
    with pytest.raises(ValueError):
        EdgeMetricGate("F2", epsilon=1e-6)
    with pytest.raises(ValueError):
        EdgeMetricGate("F2", seeds=(11, 11))
    g, model, z = geo(), gate("F2"), random((2, 6, 3))
    with pytest.raises(ValueError):
        model.apply(g, z, z, pair_chunk=0)
    with pytest.raises(ValueError):
        model.apply(g, z, z, pair_override=torch.full((2, g.num_pairs), 2., dtype=z.dtype))
    assert EPSILON == 1e-4
