"""Explicit debug-only scalar batches; no production training is run here."""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import pytest
import torch

from research.wedge_propagation.learned.model import (
    apply_weighted_pair,
    apply_weighted_wedge,
    fit_baseline,
    make_model,
    normalize_path_mean,
    normalized_mse,
    path_features,
    teacher_weights,
)
from research.wedge_propagation.operators import (
    build_wedges,
    dense_incidence,
    dense_wedge,
    fixed_wedge_apply,
    laplacian_apply,
)


def debug_batch(device="cpu"):
    """Three disjoint graphs, including one graph with no valid wedge."""
    pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (4, 6), (4, 7), (4, 8), (9, 10)]
    edges = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    n, realizations = 12, 4
    wedges = build_wedges(edges, n)
    node_graph = torch.tensor([0] * 4 + [1] * 5 + [2] * 3)
    path_graph = node_graph[wedges[1]]
    pair_rows = [(0, 2), (0, 1), *combinations(range(3, 7), 2)]
    pair_edges = torch.tensor(pair_rows, dtype=torch.long).t().contiguous()
    coefficients = torch.ones((2, len(pair_rows)), dtype=torch.float64)
    coefficients[0, 1::2] = -1
    coefficients[1, ::3] = -1
    b = dense_incidence(edges, n)
    pair_a = -coefficients[0, :, None] * b[pair_edges[0]]
    pair_a += coefficients[1, :, None] * b[pair_edges[1]]
    coefficients *= (6.0**0.5 / torch.linalg.vector_norm(pair_a, dim=1))[None]
    x = torch.randn(n, realizations, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(722))
    lx = laplacian_apply(edges, x)
    l2x = laplacian_apply(edges, lx)
    qx = fixed_wedge_apply(wedges, x)
    g1 = x[wedges[1]] - x[wedges[0]]
    g2 = x[wedges[2]] - x[wedges[1]]
    teacher_c = teacher_weights(g1, g2, path_graph, 3)
    fields = dict(x=x, edges=edges, wedges=wedges, path_graph=path_graph, node_graph=node_graph,
                  pair_edges=pair_edges, pair_coefficients=coefficients, num_graphs=3,
                  lx=lx, l2x=l2x, qx=qx, teacher_c=teacher_c)
    return SimpleNamespace(**{key: value.to(device) if isinstance(value, torch.Tensor) else value
                             for key, value in fields.items()})


def debug_pair_matrix(batch):
    b = dense_incidence(batch.edges, batch.x.shape[0], device=batch.x.device)
    return (-batch.pair_coefficients[0, :, None] * b[batch.pair_edges[0]]
            + batch.pair_coefficients[1, :, None] * b[batch.pair_edges[1]])


def test_debug_exact_teacher_and_graph_realization_mean():
    batch = debug_batch()
    g1 = batch.x[batch.wedges[1]] - batch.x[batch.wedges[0]]
    g2 = batch.x[batch.wedges[2]] - batch.x[batch.wedges[1]]
    epsilon, theta1, theta2, tau = 1e-8, 0.8, -0.4, 1.2
    cosine = g1 * g2 / (g1.abs() * g2.abs() + epsilon)
    ratio = (g2 - g1).abs() / (g1.abs() + g2.abs() + epsilon)
    raw = torch.exp(tau * torch.tanh(theta1 * cosine + theta2 * ratio))
    reference = torch.empty_like(raw)
    for graph in range(2):
        mask = batch.path_graph == graph
        reference[mask] = raw[mask] / raw[mask].mean(0)
    actual = teacher_weights(g1, g2, batch.path_graph, 3, theta1, theta2, tau, epsilon)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    for graph in range(2):
        torch.testing.assert_close(
            actual[batch.path_graph == graph].mean(0), torch.ones(4).double()
        )
    torch.testing.assert_close(
        teacher_weights(-g2, -g1, batch.path_graph, 3, theta1, theta2, tau, epsilon), actual,
        rtol=0, atol=0,
    )
    assert torch.all(actual > 0)
    torch.testing.assert_close(path_features(g1, g2), path_features(-g2, -g1), rtol=0, atol=0)


@pytest.mark.parametrize("seeded", [False, True])
def test_debug_weighted_wedge_dense_output_and_gradients(seeded):
    batch = debug_batch()
    x = batch.x.clone().requires_grad_()
    p = batch.wedges.shape[1]
    shape = (2, p, 4) if seeded else (p, 4)
    c = torch.rand(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(20))
    c.requires_grad_()
    a = dense_wedge(batch.wedges, x.shape[0])
    actual = apply_weighted_wedge(batch.wedges, x, c)
    reference = (torch.einsum("pn,spr,pr->snr", a, c, a @ x) if seeded
                 else a.t() @ (c * (a @ x)))
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    actual_grad = torch.autograd.grad(actual.square().sum(), (x, c), retain_graph=True)
    reference_grad = torch.autograd.grad(reference.square().sum(), (x, c))
    for first, second in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("seeded", [False, True])
def test_debug_weighted_signed_pair_dense_output_and_gradients(seeded):
    batch = debug_batch()
    x = batch.x.clone().requires_grad_()
    p = batch.pair_edges.shape[1]
    shape = (2, p, 4) if seeded else (p, 4)
    c = torch.rand(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(21))
    c.requires_grad_()
    a = debug_pair_matrix(batch)
    torch.testing.assert_close(a.square().sum(1), torch.full((p,), 6.0).double())
    actual = apply_weighted_pair(batch.edges, batch.pair_edges, batch.pair_coefficients, x, c)
    reference = (torch.einsum("pn,spr,pr->snr", a, c, a @ x) if seeded
                 else a.t() @ (c * (a @ x)))
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    actual_grad = torch.autograd.grad(actual.square().sum(), (x, c), retain_graph=True)
    reference_grad = torch.autograd.grad(reference.square().sum(), (x, c))
    for first, second in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind", ["learned", "random_pair"])
def test_debug_model_dense_output_and_every_parameter_gradient(kind):
    batch = debug_batch()
    model = make_model(kind, [11, 23], hidden=16)
    actual, c = model(batch)
    assert c is not None and c.shape == (2, batch.wedges.shape[1], 4)
    a = (dense_wedge(batch.wedges, batch.x.shape[0]) if kind == "learned"
         else debug_pair_matrix(batch))
    message = torch.einsum("pn,spr,pr->snr", a, c, a @ batch.x)
    reference = model.beta[:, None, None] * message
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    parameters = tuple(model.parameters())
    actual_grad = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
    reference_grad = torch.autograd.grad(reference.square().mean(), parameters)
    for first, second in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)


def test_debug_graph_feature_macro_loss_equals_independent_formula():
    node_graph = torch.tensor([0, 0, 1, 1, 1])
    target = torch.tensor([[0.0, 2.0], [0.0, 4.0], [1.0, -2.0], [2.0, 3.0], [3.0, 4.0]])
    target = target.double()
    pred = torch.stack((target + 1, target - 2)).requires_grad_()
    epsilon = 0.1
    reference = []
    for seed in range(2):
        values = []
        for graph in range(2):
            mask = node_graph == graph
            for feature in range(2):
                values.append((pred[seed, mask, feature] - target[mask, feature]).square().mean()
                              / (target[mask, feature].square().mean() + epsilon))
        reference.append(torch.stack(values).mean())
    actual = normalized_mse(pred, target, node_graph, 2, epsilon)
    torch.testing.assert_close(actual, torch.stack(reference), rtol=1e-12, atol=1e-12)
    assert torch.isfinite(torch.autograd.grad(actual.sum(), pred)[0]).all()


@pytest.mark.parametrize("kind", ["learned", "random_pair"])
def test_debug_seed_axis_matches_single_models_and_reordering(kind):
    batch = debug_batch()
    seeds = [11, 23]
    batched = make_model(kind, seeds, hidden=16)
    prediction, c = batched(batch)
    target = apply_weighted_wedge(batch.wedges, batch.x, batch.teacher_c)
    losses = normalized_mse(prediction, target, batch.node_graph, batch.num_graphs)
    losses.sum().backward()
    reordered = make_model(kind, list(reversed(seeds)), hidden=16)
    reordered_prediction, reordered_c = reordered(batch)
    torch.testing.assert_close(prediction.flip(0), reordered_prediction, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(c.flip(0), reordered_c, rtol=1e-12, atol=1e-12)
    for index, seed in enumerate(seeds):
        single = make_model(kind, [seed], hidden=16)
        single_prediction, single_c = single(batch)
        single_loss = normalized_mse(single_prediction, target, batch.node_graph, batch.num_graphs)
        single_loss.sum().backward()
        torch.testing.assert_close(prediction[index], single_prediction[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(c[index], single_c[0], rtol=1e-12, atol=1e-12)
        for combined, separate in zip(batched.parameters(), single.parameters(), strict=True):
            torch.testing.assert_close(combined[index], separate[0], rtol=0, atol=0)
            torch.testing.assert_close(
                combined.grad[index], separate.grad[0], rtol=1e-10, atol=1e-10
            )


@pytest.mark.parametrize("kind", ["learned", "random_pair"])
def test_debug_reversal_and_edge_orientation_covariance(kind):
    batch = debug_batch()
    model = make_model(kind, [11, 23], hidden=16)
    expected, expected_c = model(batch)
    reverse_fields = vars(batch).copy()
    reverse_fields["wedges"] = batch.wedges[[2, 1, 0]]
    reverse_fields["pair_edges"] = batch.pair_edges.flip(0)
    reverse_fields["pair_coefficients"] = -batch.pair_coefficients.flip(0)
    actual, actual_c = model(SimpleNamespace(**reverse_fields))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual_c, expected_c, rtol=0, atol=0)
    flip = torch.ones(batch.edges.shape[1], dtype=torch.float64)
    flip[::2] = -1
    flipped = batch.edges.clone()
    flipped[:, ::2] = flipped[:, ::2].flip(0)
    changed = SimpleNamespace(**(vars(batch) | {
        "edges": flipped,
        "pair_coefficients": batch.pair_coefficients * flip[batch.pair_edges],
    }))
    actual, actual_c = model(changed)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual_c, expected_c, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["learned", "random_pair"])
def test_debug_finite_nonzero_gradients_and_actual_update(kind):
    batch = debug_batch()
    model = make_model(kind, [11, 23], hidden=16)
    target = apply_weighted_wedge(batch.wedges, batch.x, batch.teacher_c)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    prediction, _ = model(batch)
    loss = normalized_mse(prediction, target, batch.node_graph, batch.num_graphs)
    loss.sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.linalg.vector_norm(parameter.grad) > 0
    optimizer.step()
    for original, parameter in zip(before, model.parameters(), strict=True):
        assert torch.count_nonzero(parameter.detach() - original) > 0


@pytest.mark.parametrize("kind", ["first", "polynomial", "fixed"])
def test_debug_closed_form_recovers_baseline_targets_and_is_stationary(kind):
    batch = debug_batch()
    target = {"first": batch.lx, "polynomial": batch.l2x, "fixed": batch.qx}[kind]
    model = make_model(kind, [-1])
    coefficients = fit_baseline(model, batch, target)
    prediction, c = model(batch)
    assert c is None
    torch.testing.assert_close(prediction[0], target, rtol=1e-10, atol=1e-10)
    if kind == "first":
        assert coefficients["u"] == pytest.approx(1, abs=1e-10)
    elif kind == "polynomial":
        assert coefficients["u"] == pytest.approx(0, abs=1e-10)
        assert coefficients["v"] == pytest.approx(1, abs=1e-10)
    else:
        assert coefficients == {"beta": pytest.approx(1, abs=1e-10)}
    loss = normalized_mse(prediction, target, batch.node_graph, batch.num_graphs)
    loss.sum().backward()
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), atol=1e-10, rtol=0)
    assert len(tuple(model.parameters())) == (2 if kind == "polynomial" else 1)


@pytest.mark.parametrize("kind", ["first", "polynomial", "fixed"])
def test_debug_closed_form_optimizes_the_exact_macro_relative_objective(kind):
    batch = debug_batch()
    target = apply_weighted_wedge(batch.wedges, batch.x, batch.teacher_c)
    model = make_model(kind, [-1])
    fit_baseline(model, batch, target, epsilon=1e-6)
    loss = normalized_mse(model(batch)[0], target, batch.node_graph, batch.num_graphs, eps=1e-6)
    loss.sum().backward()
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), atol=1e-10, rtol=0)


@pytest.mark.parametrize("kind", ["learned", "random_pair"])
def test_debug_no_paths_is_differentiable_mathematical_zero(kind):
    n, r = 3, 2
    x = torch.randn(n, r, dtype=torch.float64)
    batch = SimpleNamespace(
        x=x, edges=torch.empty((2, 0), dtype=torch.long),
        wedges=torch.empty((3, 0), dtype=torch.long),
        pair_edges=torch.empty((2, 0), dtype=torch.long),
        pair_coefficients=torch.empty((2, 0), dtype=torch.float64),
        path_graph=torch.empty(0, dtype=torch.long), node_graph=torch.zeros(n, dtype=torch.long),
        num_graphs=1, lx=torch.zeros_like(x), l2x=torch.zeros_like(x), qx=torch.zeros_like(x),
    )
    model = make_model(kind, [11, 23], hidden=16)
    prediction, c = model(batch)
    assert c.shape == (2, 0, r)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), rtol=0, atol=0)
    loss = normalized_mse(prediction, torch.zeros_like(x), batch.node_graph, 1)
    loss.sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)
    teacher = teacher_weights(x[:0], x[:0], batch.path_graph, 1)
    assert teacher.shape == (0, r)


def test_debug_explicit_configuration_and_shape_errors():
    batch = debug_batch()
    with pytest.raises(ValueError, match="unknown model"):
        make_model("placeholder", [11])
    with pytest.raises(ValueError, match="distinct"):
        make_model("learned", [11, 11])
    with pytest.raises(ValueError, match="positive"):
        make_model("learned", [11], hidden=0)
    with pytest.raises(ValueError, match="closed-form"):
        fit_baseline(make_model("learned", [11], hidden=16), batch, batch.qx)
    with pytest.raises(ValueError, match="every graph"):
        normalized_mse(batch.qx[None], batch.qx, batch.node_graph, 4)
    with pytest.raises(ValueError, match="outside"):
        normalize_path_mean(torch.ones(1, 2), torch.tensor([2]), 1)


@pytest.mark.parametrize("kind", ["first", "polynomial", "fixed", "learned", "random_pair"])
def test_debug_teacher_diagnostics_and_target_labels_cannot_change_forward(kind):
    batch = debug_batch()
    model = make_model(kind, [11, 23], hidden=16)
    expected, expected_c = model(batch)
    changed = SimpleNamespace(**(vars(batch) | {
        "teacher_c": torch.full_like(batch.teacher_c, 123456.0),
        "target": torch.full_like(batch.x, -654321.0),
    }))
    actual, actual_c = model(changed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if expected_c is None:
        assert actual_c is None
    else:
        torch.testing.assert_close(actual_c, expected_c, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["first", "polynomial", "fixed", "learned", "random_pair"])
def test_debug_exact_prescribed_forms_and_parameter_scope(kind):
    batch = debug_batch()
    seeds, hidden = [11, 23], 16
    model = make_model(kind, seeds, hidden=hidden)
    parameter_names = set(dict(model.named_parameters()))
    if kind in ("first", "polynomial"):
        assert "u" in parameter_names
        with torch.no_grad():
            model.u.fill_(0.7)
        if kind == "polynomial":
            with torch.no_grad():
                model.v.fill_(-0.3)
    else:
        assert "u" not in parameter_names
        assert not hasattr(model, "u")
        with torch.no_grad():
            model.beta.fill_(0.7)
    prediction, c = model(batch)
    if kind == "first":
        expected = (0.7 * batch.lx).expand(len(seeds), -1, -1)
        expected_count = len(seeds)
    elif kind == "polynomial":
        expected = (0.7 * batch.lx - 0.3 * batch.l2x).expand(len(seeds), -1, -1)
        expected_count = 2 * len(seeds)
    elif kind == "fixed":
        expected = (0.7 * batch.qx).expand(len(seeds), -1, -1)
        expected_count = len(seeds)
    else:
        message = (apply_weighted_wedge(batch.wedges, batch.x, c) if kind == "learned"
                   else apply_weighted_pair(
                       batch.edges, batch.pair_edges, batch.pair_coefficients, batch.x, c
                   ))
        expected = 0.7 * message
        expected_count = len(seeds) * (6 * hidden + 2)
    torch.testing.assert_close(prediction, expected, rtol=1e-12, atol=1e-12)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_count
    if kind in ("fixed", "learned", "random_pair"):
        changed = SimpleNamespace(**(vars(batch) | {"lx": torch.full_like(batch.lx, 1e6)}))
        torch.testing.assert_close(model(changed)[0], prediction, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; no CPU fallback")
@pytest.mark.parametrize("kind", ["learned", "random_pair"])
def test_debug_cuda_seed_batched_output_gradient_and_macro_loss(kind):
    cpu_batch = debug_batch()
    gpu_batch = debug_batch("cuda")
    cpu = make_model(kind, [11, 23], hidden=16)
    gpu = make_model(kind, [11, 23], hidden=16).to("cuda")
    expected, expected_c = cpu(cpu_batch)
    actual, actual_c = gpu(gpu_batch)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual_c.cpu(), expected_c, rtol=1e-10, atol=1e-10)
    for model, batch in ((cpu, cpu_batch), (gpu, gpu_batch)):
        target = apply_weighted_wedge(batch.wedges, batch.x, batch.teacher_c)
        normalized_mse(model(batch)[0], target, batch.node_graph, batch.num_graphs).sum().backward()
    for expected_parameter, actual_parameter in zip(
        cpu.parameters(), gpu.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad.cpu(), expected_parameter.grad,
                                   rtol=1e-9, atol=1e-9)
