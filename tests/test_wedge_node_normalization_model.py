"""Explicit DEBUG checks of the new operator, gradients and packed training."""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.wedge_propagation.classification.model import (
    PackedClassifier as OriginalPackedClassifier,
)
from research.wedge_propagation.node_normalization.model import (
    CONDITIONS,
    LOCAL_CONDITIONS,
    PackedClassifier,
)


def debug_graph(*, empty=False):
    """Small irregular DEBUG graph with an isolated vertex; no final data claim."""
    n, d = 8, 5
    pairs = (
        [] if empty else [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)]
    )
    edges = (
        torch.tensor(pairs, dtype=torch.long).T.contiguous()
        if pairs
        else torch.empty(2, 0, dtype=torch.long)
    )
    neighbors = [[] for _ in range(n)]
    for u, v in pairs:
        neighbors[u].append(v)
        neighbors[v].append(u)
    triples = [(i, j, k) for j in range(n) for i, k in combinations(sorted(neighbors[j]), 2)]
    paths = (
        torch.tensor(triples, dtype=torch.long).T.contiguous()
        if triples
        else torch.empty(3, 0, dtype=torch.long)
    )
    b, a = (
        torch.zeros(len(pairs), n, dtype=torch.float64),
        torch.zeros(len(triples), n, dtype=torch.float64),
    )
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
    graph = SimpleNamespace(
        x=torch.randn(n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(124)),
        y=torch.arange(n) % 3,
        edges=edges,
        paths=paths,
        degree=degree,
        qdiag=qdiag,
        sd=sd,
        sq=sq,
        train_mask=torch.arange(n) < 5,
        val_mask=torch.arange(n) == 5,
        test_mask=torch.arange(n) > 5,
    )
    return graph, b, a


def make_model(condition="learned_wedge_local_rms", seeds=(11, 23), **kwargs):
    return PackedClassifier(
        condition,
        5,
        3,
        seeds,
        hidden=6,
        gate_hidden=4,
        dropout=kwargs.pop("dropout", 0.0),
        dataset_name="DEBUG",
        **kwargs,
    ).double()


def ce(logits, graph):
    loss = F.cross_entropy(
        logits[:, graph.train_mask].flatten(0, 1),
        graph.y[graph.train_mask].repeat(logits.shape[0]),
        reduction="none",
    )
    return loss.reshape(logits.shape[0], -1).mean(1).sum()


def dense_local(a, z, c, *, detach_diagonal=False):
    diagonal = c @ a.square()
    if detach_diagonal:
        diagonal = diagonal.detach()
    active = diagonal > 0
    scale = torch.where(active, diagonal, torch.ones_like(diagonal)).rsqrt() * active
    return scale[:, :, None] * (a.T @ (c[:, :, None] * (a @ (scale[:, :, None] * z)))) / 3


def dense_forward(net, graph, b, a):
    lap = 0.5 * graph.sd[:, None] * (b.T @ b) * graph.sd[None, :]
    h = graph.x[None].expand(len(net.seeds), -1, -1)
    for layer in range(2):
        z = torch.bmm(h, net.projections[layer])
        g1 = z[:, graph.paths[1]] - z[:, graph.paths[0]]
        g2 = z[:, graph.paths[2]] - z[:, graph.paths[1]]
        if net.condition == "learned_wedge_rms":
            energy = (b @ z).square().mean((1, 2))
            sigma = torch.where(energy > 0, energy, torch.ones_like(energy)).sqrt()
            g1, g2 = g1 / sigma[:, None, None], g2 / sigma[:, None, None]
        phi = torch.cat(
            (g1.abs() + g2.abs(), g1 * g2, (g2 - g1).abs(), (g1.abs() - g2.abs()).square()), -1
        )
        gate = net.gates[layer]
        raw = (
            torch.bmm((torch.bmm(phi, gate.w1) + gate.b1[:, None]).relu(), gate.w2)
            .squeeze(-1)
            .tanh()
            .exp()
        )
        c = raw / raw.mean(1, keepdim=True)
        t, r = net.u[layer].sigmoid(), net.v[layer].sigmoid()
        alpha, beta = t * (1 - r), t * r
        u = z - alpha[:, None, None] * (lap @ z) - beta[:, None, None] * dense_local(a, z, c)
        h = u.relu() if layer == 0 else u
    return h


@pytest.mark.parametrize("condition", sorted(LOCAL_CONDITIONS))
def test_dense_two_layer_logits_input_and_all_ce_parameter_gradients(condition):
    graph, b, a = debug_graph()
    graph.x.requires_grad_()
    net = make_model(condition, path_chunk=4, checkpoint_paths=True)
    actual, details = net(graph, diagnostics=True)
    expected = dense_forward(net, graph, b, a)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    parameters = (graph.x, *net.parameters())
    actual_grad = torch.autograd.grad(ce(actual, graph), parameters)
    expected_grad = torch.autograd.grad(ce(expected, graph), parameters)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(got).all()
        assert got.abs().sum() > 0
        torch.testing.assert_close(got, want, rtol=2e-9, atol=2e-11)
    for detail in details:
        assert torch.equal(detail["kappa"], torch.ones(2, dtype=torch.float64))
        assert torch.all(detail["kappa_global"] >= 1 - 1e-12)
        assert detail["node_inv_sqrt"][:, -1].eq(0).all()
        assert detail["t_message"][:, -1].eq(0).all()
        assert all(not value.requires_grad for value in detail.values() if torch.is_tensor(value))


def test_c_gradient_includes_both_diagonal_factors_and_scale_direction_cancels():
    graph, _, a = debug_graph()
    net = make_model(path_chunk=2)
    z = torch.randn(
        2,
        8,
        6,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(51),
        requires_grad=True,
    )
    c = (
        torch.linspace(0.2, 2.8, 2 * a.shape[0], dtype=torch.float64)
        .reshape(2, -1)
        .requires_grad_()
    )
    target = torch.randn(z.shape, dtype=z.dtype, generator=torch.Generator().manual_seed(63))
    output = net.apply_node_branch(graph, z, c)
    dense = dense_local(a, z, c)
    torch.testing.assert_close(output, dense, rtol=1e-12, atol=1e-12)
    actual = torch.autograd.grad((output * target).sum(), (c, z), retain_graph=True)
    want = torch.autograd.grad((dense * target).sum(), (c, z))
    for got, expected in zip(actual, want, strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)
    wrong = torch.autograd.grad((dense_local(a, z, c, detach_diagonal=True) * target).sum(), c)[0]
    assert not torch.allclose(actual[0], wrong, rtol=1e-3, atol=1e-5)
    # Uniform C scaling cancels from BOTH inverse diagonal factors.
    torch.testing.assert_close(
        (actual[0] * c).sum(1), torch.zeros(2, dtype=z.dtype), atol=2e-12, rtol=0
    )
    torch.testing.assert_close(
        net.apply_node_branch(graph, z, c * 17), output, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("condition", sorted(LOCAL_CONDITIONS))
def test_packed_dropout_ce_and_adam_equal_independent_models(condition):
    graph, _, _ = debug_graph()
    packed = make_model(condition, seeds=(23, 11), dropout=0.5, path_chunk=4)
    singles = [
        make_model(condition, seeds=(seed,), dropout=0.5, path_chunk=7) for seed in packed.seeds
    ]
    optimizer = torch.optim.Adam(packed.weight_decay_groups(5e-4), lr=0.003, foreach=False)
    single_optimizers = [
        torch.optim.Adam(net.weight_decay_groups(5e-4), lr=0.003, foreach=False) for net in singles
    ]
    actual, _ = packed(graph, epoch=17)
    expected = torch.cat([net(graph, epoch=17)[0] for net in singles])
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    ce(actual, graph).backward()
    for net in singles:
        ce(net(graph, epoch=17)[0], graph).backward()
    for name, parameter in packed.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        want = torch.cat([dict(net.named_parameters())[name].grad for net in singles])
        torch.testing.assert_close(parameter.grad, want, rtol=1e-9, atol=1e-11)
    before = {name: p.detach().clone() for name, p in packed.named_parameters()}
    optimizer.step()
    for opt in single_optimizers:
        opt.step()
    for name, parameter in packed.named_parameters():
        assert not torch.equal(parameter, before[name])
        wants = [dict(net.named_parameters())[name] for net in singles]
        torch.testing.assert_close(parameter, torch.cat(wants), rtol=1e-10, atol=1e-11)
        for key in ("exp_avg", "exp_avg_sq"):
            target = torch.cat(
                [opt.state[p][key] for opt, p in zip(single_optimizers, wants, strict=True)]
            )
            torch.testing.assert_close(
                optimizer.state[parameter][key], target, rtol=1e-9, atol=1e-11
            )


@pytest.mark.parametrize("condition", sorted(LOCAL_CONDITIONS))
def test_chunk_checkpoint_preserve_full_paths_logits_and_gradients(condition):
    graph, _, _ = debug_graph()
    whole = make_model(condition, checkpoint_paths=False)
    chunked = make_model(condition, path_chunk=1, checkpoint_paths=True)
    out_whole, details_whole = whole(graph, diagnostics=True)
    out_chunk, details_chunk = chunked(graph, diagnostics=True)
    torch.testing.assert_close(out_whole, out_chunk, rtol=1e-11, atol=1e-12)
    for got, want in zip(details_chunk, details_whole, strict=True):
        for key in ("c", "weighted_diagonal", "node_inv_sqrt", "kappa_global"):
            torch.testing.assert_close(got[key], want[key], rtol=1e-11, atol=1e-12)
    want = torch.autograd.grad(ce(out_whole, graph), tuple(whole.parameters()))
    actual = torch.autograd.grad(ce(out_chunk, graph), tuple(chunked.parameters()))
    for got, expected in zip(actual, want, strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("condition", CONDITIONS[:3])
def test_fixed_and_global_are_exact_original_controls(condition):
    graph, _, _ = debug_graph()
    new = make_model(condition, dropout=0.5, path_chunk=3)
    original = OriginalPackedClassifier(
        condition,
        5,
        3,
        (11, 23),
        hidden=6,
        gate_hidden=4,
        dropout=0.5,
        path_chunk=3,
        dataset_name="DEBUG",
    ).double()
    assert new.normalization == "global_kappa"
    assert new.experiment_condition == condition
    for key, value in new.state_dict().items():
        assert torch.equal(value, original.state_dict()[key])
    actual, details = new(graph, epoch=21, diagnostics=True)
    expected, old_details = original(graph, epoch=21, diagnostics=True)
    assert torch.equal(actual, expected)
    for got, want in zip(details, old_details, strict=True):
        assert got.keys() == want.keys()
        for key, value in got.items():
            assert torch.equal(value, want[key]) if torch.is_tensor(value) else value == want[key]
    grads = torch.autograd.grad(ce(actual, graph), tuple(new.parameters()))
    old_grads = torch.autograd.grad(ce(expected, graph), tuple(original.parameters()))
    assert all(torch.equal(got, want) for got, want in zip(grads, old_grads, strict=True))


def test_identity_c_equals_fixed_control_with_shared_initialization():
    graph, _, _ = debug_graph()
    fixed, local = make_model("fixed_wedge", dropout=0.5), make_model(dropout=0.5)
    with torch.no_grad():
        for gate in local.gates:
            gate.w2.zero_()
    got, _ = local(graph, epoch=9)
    want, _ = fixed(graph, epoch=9)
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-12)
    local.eval()
    fixed.eval()
    got, _ = local(graph, intervention="c_identity")
    want, _ = fixed(graph)
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-12)


def test_dense_operator_bound_and_supported_nullspace_coordinates():
    graph, b, a = debug_graph()
    net = make_model(seeds=(11,))
    c = torch.exp(torch.linspace(-4, 4, a.shape[0], dtype=torch.float64))[None]
    identity = torch.eye(8, dtype=torch.float64)[None]
    matrix = net.apply_node_branch(graph, identity, c)[0]
    torch.testing.assert_close(matrix, matrix.T, rtol=1e-12, atol=1e-12)
    spectrum = torch.linalg.eigvalsh(matrix)
    assert spectrum.min() >= -1e-12 and spectrum.max() <= 1 + 1e-12
    supported = graph.qdiag > 0
    torch.testing.assert_close(
        matrix.diag()[supported], torch.full((7,), 1 / 3, dtype=torch.float64)
    )
    assert matrix[-1].eq(0).all()
    diagonal = c @ a.square()
    # The normalized null vector is sqrt(D_C)*1, not a generic constant feature.
    torch.testing.assert_close(
        matrix @ diagonal[0].sqrt(), torch.zeros(8, dtype=torch.float64), rtol=0, atol=1e-12
    )
    lap = graph.sd[:, None] * (b.T @ b) * graph.sd[None] / 2
    alpha, beta = 0.5, 0.25
    propagation = torch.eye(8, dtype=torch.float64) - alpha * lap - beta * matrix
    spectrum = torch.linalg.eigvalsh(propagation)
    assert spectrum.min() >= 1 - alpha - beta - 1e-12
    assert spectrum.max() <= 1 + 1e-12


@pytest.mark.parametrize("condition", sorted(LOCAL_CONDITIONS))
def test_endpoint_reversal_and_complete_path_permutation_invariant(condition):
    graph, _, _ = debug_graph()
    net = make_model(condition, path_chunk=3)
    original, _ = net(graph)
    reversed_graph = SimpleNamespace(**vars(graph))
    reversed_graph.paths = graph.paths.flip(0)
    reversed_graph.edges = graph.edges.flip(0)
    reversed_output, _ = net(reversed_graph)
    torch.testing.assert_close(original, reversed_output, rtol=1e-11, atol=1e-12)
    order = torch.randperm(graph.paths.shape[1], generator=torch.Generator().manual_seed(71))
    reordered = SimpleNamespace(**vars(graph))
    reordered.paths = graph.paths[:, order]
    reordered_output, _ = net(reordered)
    torch.testing.assert_close(original, reordered_output, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("condition", sorted(LOCAL_CONDITIONS))
def test_empty_paths_zero_branch_finite_projection_gradient(condition):
    graph, _, _ = debug_graph(empty=True)
    graph.x.requires_grad_()
    net = make_model(condition, path_chunk=2)
    prediction, details = net(graph, diagnostics=True)
    assert torch.isfinite(prediction).all()
    for detail in details:
        assert detail["c"].shape == (2, 0)
        assert detail["t_message"].eq(0).all()
        assert detail["node_inv_sqrt"].eq(0).all()
        assert detail["kappa_global"].eq(1).all()
    ce(prediction, graph).backward()
    assert torch.isfinite(graph.x.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in net.projections)


@pytest.mark.parametrize(
    "d,k,base,learned",
    [(1433, 7, 92160, 110596), (3703, 6, 237376, 255556), (500, 3, 32192, 49604)],
)
def test_full_parameter_contract_has_no_extra_or_unused_normalization_parameters(
    d, k, base, learned
):
    for condition in CONDITIONS:
        net = PackedClassifier(condition, d, k, (11, 23), dataset_name="count")
        expected = base + 4 if condition == "fixed_wedge" else learned
        assert net.parameters_per_seed == expected
        assert net.trainable_parameters_per_seed == expected
        assert sum(parameter.numel() for parameter in net.parameters()) == 2 * expected


def test_invalid_condition_and_inapplicable_global_hold_are_explicit_errors():
    with pytest.raises(ValueError, match="unknown node normalization"):
        make_model("mlp")
    graph, _, _ = debug_graph()
    net = make_model()
    z = torch.bmm(graph.x[None].expand(2, -1, -1), net.projections[0])
    with pytest.raises(ValueError, match="recomputing D_C"):
        net.probe_layer(graph, 0, z, kappa_mode="hold")
    with pytest.raises(ValueError, match="unsupported node normalization"):
        net.probe_layer(graph, 0, z, intervention="random_physical_edge_pair_correspondence")
    with pytest.raises(ValueError, match="every path"):
        net.probe_layer(
            graph,
            0,
            z,
            intervention="c_position_shuffle",
            manifest={"permutation": torch.zeros(graph.paths.shape[1], dtype=torch.long)},
        )


@pytest.mark.parametrize("condition", sorted(LOCAL_CONDITIONS))
def test_actual_training_float32_gate_and_diagonal_gradients_match_dense(condition):
    graph, b, a = debug_graph()
    for name, value in vars(graph).items():
        if value.is_floating_point():
            setattr(graph, name, value.float())
    b, a = b.float(), a.float()
    graph.x.requires_grad_()
    net = make_model(condition, path_chunk=3, checkpoint_paths=True).float()
    actual, _ = net(graph)
    expected = dense_forward(net, graph, b, a)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    tensors = (graph.x, *net.parameters())
    actual_grads = torch.autograd.grad(ce(actual, graph), tensors)
    dense_grads = torch.autograd.grad(ce(expected, graph), tensors)
    for got, want in zip(actual_grads, dense_grads, strict=True):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=3e-4, atol=3e-7)


@pytest.mark.parametrize("invalid", [0.0, -0.1, float("nan"), float("inf")])
def test_public_supplied_c_rejects_nonpositive_or_nonfinite_without_masking(invalid):
    graph, _, _ = debug_graph()
    net = make_model()
    z = torch.randn(2, 8, 6, dtype=torch.float64)
    c = torch.ones(2, graph.paths.shape[1], dtype=torch.float64)
    c[0, 0] = invalid
    with pytest.raises(ValueError, match="finite positive conductances"):
        net.apply_node_branch(graph, z, c)
