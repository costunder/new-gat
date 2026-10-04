"""DEBUG-only independent dense references for lifted local/cross propagation.

Tiny physical graphs establish the algebra, cancellation controls and real CE
gradient connection. They are not the server FULL audit or classifier training.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from research.local_context_coupling.model import PackedClassifier
from research.local_context_coupling.operators import (
    apply_cross,
    apply_intra,
    cross_energy,
    intra_energy,
    merge,
    prepare_geometry,
    replicate,
    sandwich,
    unified,
)
from research.local_energy_relations.topology import batch_topologies, build_topology


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _edges(pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T.copy()


def _reference(n, pairs, mode="unit"):
    """Build every matrix from physical edges, without production geometry fields."""
    pairs = sorted(tuple(sorted(p)) for p in pairs)
    adjacent = [set() for _ in range(n)]
    for a, b in pairs:
        adjacent[a].add(b)
        adjacent[b].add(a)
    local = [s | {v} for v, s in enumerate(adjacent)]
    copies = [(v, a) for v, nodes in enumerate(local) for a in sorted(nodes)]
    position = {item: i for i, item in enumerate(copies)}
    identity = torch.eye(len(copies), dtype=torch.float64)
    incidence, conductance = [], []
    for v, nodes in enumerate(local):
        induced = [(a, b) for a, b in pairs if a in nodes and b in nodes]
        degree = {a: sum(a in edge for edge in induced) for a in nodes}
        for a, b in induced:
            incidence.append(identity[position[v, b]] - identity[position[v, a]])
            conductance.append(1.0 if mode == "unit" else 2.0 / (degree[a] + degree[b]))
    bmat = torch.stack(incidence) if incidence else identity.new_zeros((0, len(copies)))
    c = torch.tensor(conductance, dtype=torch.float64)
    amat = bmat.T @ (c[:, None] * bmat)
    cross = []
    for v, u in pairs:
        for a in sorted(local[v] & local[u]):
            cross.append(identity[position[u, a]] - identity[position[v, a]])
    jmat = torch.stack(cross) if cross else identity.new_zeros((0, len(copies)))
    kmat = jmat.T @ jmat
    rmat = identity.new_zeros((len(copies), n))
    for i, (_, a) in enumerate(copies):
        rmat[i, a] = 1
    counts = rmat.sum(0)
    mmat = rmat.T / counts[:, None]
    return SimpleNamespace(A=amat, K=kmat, R=rmat, M=mmat, B=bmat, C=c, J=jmat,
                           copies=copies, counts=counts)


def _act(matrix, value):
    return torch.einsum("ij,...jf->...if", matrix, value)


def _fixture(n=7, pairs=None, mode="unit", seeds=2, channels=4):
    if pairs is None:
        pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]
    top = build_topology(n, _edges(pairs))
    geo = prepare_geometry(top, mode).to("cpu", torch.float64)
    generator = torch.Generator().manual_seed(782)
    h = torch.randn((seeds, n, channels), generator=generator, dtype=torch.float64)
    return geo, h, _reference(n, pairs, mode)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("packed", [False, True])
def test_copy_intra_cross_merge_and_joint_energy_match_independent_dense(mode, packed):
    geo, h, ref = _fixture(mode=mode)
    if not packed:
        h = h[0]
    y = replicate(geo, h)
    torch.testing.assert_close(y, _act(ref.R, h), atol=0, rtol=0)
    torch.testing.assert_close(apply_intra(geo, y), _act(ref.A, y), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(apply_cross(geo, y), _act(ref.K, y), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(merge(geo, y), h, atol=1e-15, rtol=1e-15)
    assert len(geo.cross_left) == ref.J.shape[0]
    torch.testing.assert_close(geo.copy_counts, ref.counts, atol=0, rtol=0)
    # Perturb copies independently to expose the otherwise initially-zero K term.
    y = (y + torch.arange(y.numel(), dtype=y.dtype).reshape(y.shape) / 13).requires_grad_()
    intra = intra_energy(geo, y)
    cross = cross_energy(geo, y)
    expected_i = .5 * (_act(ref.B, y).square() * ref.C[:, None]).sum(-2)
    expected_j = .5 * _act(ref.J, y).square().sum(-2)
    torch.testing.assert_close(intra[..., 0, :], expected_i, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cross[..., 0, :], expected_j, atol=1e-12, rtol=1e-12)
    gradient = torch.autograd.grad((intra + .7 * cross).sum(), y)[0]
    torch.testing.assert_close(gradient, _act(ref.A + .7 * ref.K, y), atol=1e-12, rtol=1e-12)
    assert torch.linalg.eigvalsh(ref.A + .7 * ref.K).min() >= -1e-12


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_initial_cross_energy_and_immediate_cross_parameter_gradient_are_zero(mode):
    geo, h, _ = _fixture(mode=mode)
    h.requires_grad_()
    theta = torch.tensor(.19, dtype=h.dtype, requires_grad=True)
    y = replicate(geo, h)
    action = apply_cross(geo, y)
    assert action.eq(0).all()
    assert cross_energy(geo, y).eq(0).all()
    # Cross cannot change a freshly replicated state, including its gain gradient.
    immediate = merge(geo, y - theta.sigmoid() * action)
    torch.testing.assert_close(immediate, h, atol=1e-15, rtol=1e-15)
    theta_grad = torch.autograd.grad(immediate.square().sum(), theta)[0]
    assert theta_grad == 0
    # Cross also vanishes if immediately merged after one intra computation.
    y1 = y - .1 * apply_intra(geo, y)
    torch.testing.assert_close(merge(geo, apply_cross(geo, y1)), torch.zeros_like(h),
                               atol=2e-15, rtol=0)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_sandwich_has_exact_matched_difference_and_nonzero_guarantee(mode):
    geo, h, ref = _fixture(mode=mode)
    eta, gamma = .1, .35
    on, diagnostics = sandwich(geo, h, eta=eta, gamma=gamma, diagnostics=True)
    off = sandwich(geo, h, cross=False, eta=eta, gamma=gamma)
    eye = torch.eye(ref.A.shape[0], dtype=h.dtype)
    ton = ref.M @ (eye - eta * ref.A) @ (eye - gamma * ref.K) @ (eye - eta * ref.A) @ ref.R
    toff = ref.M @ (eye - eta * ref.A) @ (eye - eta * ref.A) @ ref.R
    expected_difference = -eta**2 * gamma * _act(ref.M @ ref.A @ ref.K @ ref.A @ ref.R, h)
    torch.testing.assert_close(on, _act(ton, h), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(off, _act(toff, h), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(on - off, expected_difference, atol=2e-13, rtol=1e-11)
    y1 = diagnostics["after_intra_1"]
    assert apply_cross(geo, y1).norm() > 0
    assert (on - off).norm() > 0
    torch.testing.assert_close(diagnostics["after_cross"] - y1, -gamma * apply_cross(geo, y1),
                               atol=1e-13, rtol=1e-13)
    folded = ref.R.T @ ref.A @ ref.K @ ref.A @ ref.R
    torch.testing.assert_close(folded, folded.T, atol=1e-13, rtol=1e-13)
    assert torch.linalg.eigvalsh(folded).min() >= -1e-12
    power = (h * _act(folded, h)).sum()
    assert power > 0


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("control", ["clique", "constant", "isolates"])
def test_zero_context_disagreement_controls(mode, control):
    n = 5
    pairs = ([(a, b) for a in range(n) for b in range(a + 1, n)] if control == "clique"
             else [] if control == "isolates" else [(0, 1), (1, 2), (2, 3)])
    geo, h, _ = _fixture(n, pairs, mode)
    if control == "constant":
        h = h[:, :1].expand(-1, n, -1).clone()
    on, details = sandwich(geo, h, diagnostics=True)
    off = sandwich(geo, h, cross=False)
    assert apply_cross(geo, details["after_intra_1"]).abs().max() <= 1e-14
    torch.testing.assert_close(on, off, atol=1e-14, rtol=1e-14)


def test_two_hop_dependency_is_already_present_and_j_difference_can_be_zero():
    geo, _, _ = _fixture(3, [(0, 1), (1, 2)], channels=1, seeds=1)
    h = torch.tensor([[.3], [-.7], [.9]], dtype=torch.float64, requires_grad=True)
    on = sandwich(geo, h, eta=.1, gamma=.35)
    off = sandwich(geo, h, cross=False, eta=.1, gamma=.35)
    gon = torch.autograd.grad(on[0, 0], h, retain_graph=True)[0]
    goff = torch.autograd.grad(off[0, 0], h)[0]
    torch.testing.assert_close(gon[2, 0], torch.tensor(.005, dtype=h.dtype), atol=1e-15, rtol=0)
    torch.testing.assert_close(goff[2, 0], torch.tensor(.005, dtype=h.dtype), atol=1e-15, rtol=0)
    torch.testing.assert_close(gon[2, 0] - goff[2, 0], torch.tensor(0., dtype=h.dtype),
                               atol=1e-15, rtol=0)
    # Other entries change: the endpoint control is not a proof of globally zero J effect.
    assert (on - off).norm() > 0


def test_nonadjacent_input_derivative_has_additional_cross_effect_on_irregular_graph():
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]
    geo, h, ref = _fixture(5, pairs, seeds=1, channels=1)
    h = h[0].requires_grad_()
    on = sandwich(geo, h, eta=.1, gamma=.35)
    off = sandwich(geo, h, cross=False, eta=.1, gamma=.35)
    gradient = torch.autograd.grad((on - off)[3, 0], h)[0]
    expected = -.1**2 * .35 * (ref.M @ ref.A @ ref.K @ ref.A @ ref.R)[3, 4]
    assert (3, 4) not in pairs
    assert expected.abs() > 1e-8
    torch.testing.assert_close(gradient[4, 0], expected, atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_persistent_unified_cross_is_first_visible_at_three_steps(steps):
    geo, h, ref = _fixture()
    step, strength = .1, .7
    on = unified(geo, h, steps=steps, step=step, lambda_cross=strength)
    off = unified(geo, h, steps=steps, step=step, lambda_cross=0)
    eye = torch.eye(ref.A.shape[0], dtype=h.dtype)
    matrix = ref.M @ torch.linalg.matrix_power(eye - step * (ref.A + strength * ref.K), steps) @ ref.R
    torch.testing.assert_close(on, _act(matrix, h), atol=1e-13, rtol=1e-13)
    if steps < 3:
        torch.testing.assert_close(on, off, atol=2e-14, rtol=1e-14)
    else:
        difference = -step**3 * strength * _act(ref.M @ ref.A @ ref.K @ ref.A @ ref.R, h)
        assert difference.norm() > 0
        torch.testing.assert_close(on - off, difference, atol=1e-13, rtol=1e-11)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_disjoint_graph_batch_and_packed_seeds_keep_canonical_cross_pairs(mode):
    specs = [(3, [(0, 1), (1, 2)]), (5, [(0, 1), (1, 2), (1, 3), (2, 3)])]
    tops = [build_topology(n, _edges(p)) for n, p in specs]
    geo = prepare_geometry(batch_topologies(tops), mode).to("cpu", torch.float64)
    refs = [_reference(n, p, mode) for n, p in specs]
    assert len(geo.cross_left) == sum(r.J.shape[0] for r in refs)
    # Forward/reverse pair halves restart within each input graph. Choosing the
    # first half of all batched pairs would duplicate graph 1 and omit graph 2.
    pairs = geo.topology.pair_centers
    for graph_index in range(2):
        assert int((geo.cross_graph == graph_index).sum()) == refs[graph_index].J.shape[0]
    generator = torch.Generator().manual_seed(442)
    h = torch.randn((3, 8, 2), generator=generator, dtype=torch.float64)
    offset, pieces = 0, []
    for top, (n, _) in zip(tops, specs, strict=True):
        separate = prepare_geometry(top, mode).to("cpu", h.dtype)
        pieces.append(sandwich(separate, h[:, offset:offset + n]))
        offset += n
    batched = sandwich(geo, h)
    torch.testing.assert_close(batched, torch.cat(pieces, dim=1), atol=1e-13, rtol=1e-13)
    # Dense references use each graph's own scalar eta/gamma, not one batch max.
    R, A, K, M = (torch.block_diag(*(getattr(r, name) for r in refs)) for name in ("R", "A", "K", "M"))
    y = _act(R, h)
    copy_graph = torch.cat([torch.full((r.R.shape[0],), i) for i, r in enumerate(refs)])
    eta = geo.eta[copy_graph][None, :, None]
    gamma = geo.gamma[copy_graph][None, :, None]
    y1 = y - eta * _act(A, y)
    y2 = y1 - gamma * _act(K, y1)
    expected = _act(M, y2 - eta * _act(A, y2))
    torch.testing.assert_close(batched, expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(merge(geo, apply_cross(geo, y1)), torch.zeros_like(h),
                               atol=1e-14, rtol=0)
    assert pairs.shape[1] == sum(t.num_pairs for t in tops)


def test_positive_cross_weights_and_input_gradients_match_dense():
    geo, h, ref = _fixture()
    weights = torch.linspace(.8, 1.2, ref.J.shape[0], dtype=h.dtype, requires_grad=True)
    y = (replicate(geo, h) + torch.arange(ref.R.shape[0], dtype=h.dtype)[None, :, None] / 7).requires_grad_()
    actual = apply_cross(geo, y, weights=weights)
    kmat = ref.J.T @ (weights[:, None] * ref.J)
    expected = _act(kmat, y)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    # Production ordering is canonical center pair, then physical node ID.
    probe = torch.linspace(-.3, .8, actual.numel(), dtype=h.dtype).reshape(actual.shape)
    ga = torch.autograd.grad((actual * probe).sum(), (y, weights), retain_graph=True)
    ge = torch.autograd.grad((expected * probe).sum(), (y, weights))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_real_two_layer_classifier_ce_updates_shared_cross_parameter(mode):
    geo, fields, _ = _fixture(mode=mode, channels=7)
    x = fields[0]
    model = PackedClassifier(7, 3, seeds=(11, 23), mode=mode, condition="cross_on",
                             hidden=64, dropout=0).double()
    assert model.theta_cross.shape == (2,)
    assert model.cross_gain.gt(0).all() and model.cross_gain.lt(1).all()
    assert [name for name, _ in model.named_parameters() if "theta_cross" in name] == ["theta_cross"]
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in model.parameters()}
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    logits, details = model((x, geo), diagnostics=True)
    assert logits.shape == (2, 7, 3) and len(details) == 2
    target = torch.tensor([0, 1, 2, 0, 1, 2, 1])
    loss = sum(F.cross_entropy(v, target) for v in logits)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    optimizer.step()
    for name, p in model.named_parameters():
        assert not torch.equal(before[name], p), name
    # Off has both intra passes and no disconnected cross parameter.
    off = PackedClassifier(7, 3, seeds=(11, 23), mode=mode, condition="cross_off",
                           hidden=64, dropout=0).double()
    assert off.theta_cross is None
    assert all("theta_cross" not in name for name, _ in off.named_parameters())
    assert {k for k in off.state_dict()} == {k for k in model.state_dict() if k != "theta_cross"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_cuda_sparse_forward_and_input_gradient_match_cpu_dense(mode):
    geo, h, ref = _fixture(mode=mode)
    cuda_geo = geo.to("cuda", torch.float64)
    cpu_h, gpu_h = h.clone().requires_grad_(), h.cuda().detach().requires_grad_()
    actual = sandwich(cuda_geo, gpu_h, eta=.1, gamma=.35)
    eye = torch.eye(ref.A.shape[0], dtype=h.dtype)
    matrix = ref.M @ (eye - .1 * ref.A) @ (eye - .35 * ref.K) @ (eye - .1 * ref.A) @ ref.R
    expected = _act(matrix, cpu_h)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-11, rtol=1e-11)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(gpu_h.grad.cpu(), cpu_h.grad, atol=1e-11, rtol=1e-11)
