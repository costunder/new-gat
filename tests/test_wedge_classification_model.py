"""Explicit DEBUG mathematics and real loss/update tests, never full training."""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.wedge_propagation.classification.model import CONDITIONS, PackedClassifier


def debug_graph(*, device="cpu", constant=False, empty=False):
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
    b = torch.zeros(len(pairs), n, dtype=torch.float64)
    if pairs:
        b[torch.arange(len(pairs)), edges[0]] = -1
        b[torch.arange(len(pairs)), edges[1]] = 1
    a = torch.zeros(len(triples), n, dtype=torch.float64)
    if triples:
        a[torch.arange(len(triples)), paths[0]] = 1
        a[torch.arange(len(triples)), paths[1]] = -2
        a[torch.arange(len(triples)), paths[2]] = 1
    degree, qdiag = b.square().sum(0), a.square().sum(0)
    sd = torch.where(degree > 0, degree, torch.ones_like(degree)).rsqrt() * (degree > 0)
    sq = torch.where(qdiag > 0, qdiag, torch.ones_like(qdiag)).rsqrt() * (qdiag > 0)
    loops = torch.arange(n).repeat(2, 1)
    gcn_edges = torch.cat((edges, edges.flip(0), loops), dim=1)
    gcn_weight = (degree[gcn_edges[0]] + 1).rsqrt() * (degree[gcn_edges[1]] + 1).rsqrt()
    x = torch.randn(n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(124))
    if constant:
        x = torch.ones_like(x)
    graph = SimpleNamespace(
        x=x,
        y=torch.arange(n) % 3,
        edges=edges,
        paths=paths,
        degree=degree,
        qdiag=qdiag,
        sd=sd,
        sq=sq,
        gcn_edges=gcn_edges,
        gcn_weight=gcn_weight,
        train_mask=torch.arange(n) < 5,
        val_mask=torch.arange(n) == 5,
        test_mask=torch.arange(n) > 5,
    )
    for name, value in vars(graph).items():
        setattr(graph, name, value.to(device))
    return graph, b.to(device), a.to(device)


def model(condition="learned_wedge_rms", seeds=(11, 23), **kwargs):
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


def dense_forward(net, graph, b, a):
    """Independent dense formulas: no production incidence/scatter helpers."""
    lap = 0.5 * graph.sd[:, None] * (b.T @ b) * graph.sd[None, :]
    q = graph.sq[:, None] * (a.T @ a) * graph.sq[None, :] / 3
    adjacency = torch.zeros_like(lap)
    adjacency[graph.gcn_edges[1], graph.gcn_edges[0]] = graph.gcn_weight
    h = graph.x[None].expand(len(net.seeds), -1, -1)
    for layer in range(2):
        z = torch.bmm(h, net.projections[layer])
        if net.condition == "mlp":
            u = z
        elif net.condition == "standard_gcn":
            u = adjacency @ z
        else:
            alpha = net.u[layer].sigmoid()
            if net.condition == "first_order":
                u = z - alpha[:, None, None] * (lap @ z)
            else:
                r = net.v[layer].sigmoid()
                beta, alpha = alpha * r, alpha * (1 - r)
                if net.condition == "polynomial_2":
                    t_message = lap @ (lap @ z)
                elif net.condition in ("fixed_wedge", "fixed_wedge_node_mlp"):
                    t_message = q @ z
                else:
                    g1 = z[:, graph.paths[1]] - z[:, graph.paths[0]]
                    g2 = z[:, graph.paths[2]] - z[:, graph.paths[1]]
                    if net.condition == "learned_wedge_rms":
                        energy = (b @ z).square().mean((1, 2))
                        sigma = torch.where(energy > 0, energy, torch.ones_like(energy)).sqrt()
                        g1, g2 = g1 / sigma[:, None, None], g2 / sigma[:, None, None]
                    phi = torch.cat(
                        (
                            g1.abs() + g2.abs(),
                            g1 * g2,
                            (g2 - g1).abs(),
                            (g1.abs() - g2.abs()).square(),
                        ),
                        -1,
                    )
                    gate = net.gates[layer]
                    raw = (
                        torch.bmm((torch.bmm(phi, gate.w1) + gate.b1[:, None]).relu(), gate.w2)
                        .squeeze(-1)
                        .tanh()
                        .exp()
                    )
                    c = raw / raw.mean(1, keepdim=True)
                    diagonal = c @ a.square()
                    ratio = (
                        diagonal
                        / torch.where(graph.qdiag > 0, graph.qdiag, torch.ones_like(graph.qdiag))[
                            None
                        ]
                    )
                    kappa = torch.amax(ratio, dim=1)
                    scaled = graph.sq[None, :, None] * z
                    t_message = (
                        (a.T @ (c[:, :, None] * (a @ scaled)))
                        * graph.sq[None, :, None]
                        / (3 * kappa[:, None, None])
                    )
                u = z - alpha[:, None, None] * (lap @ z) - beta[:, None, None] * t_message
                if net.condition == "fixed_wedge_node_mlp":
                    f = net.node_mlps[layer]
                    node = torch.bmm((torch.bmm(z, f.w1) + f.b1[:, None]).relu(), f.w2)
                    u = u + beta[:, None, None] * node
        h = u.relu() if layer == 0 else u
    return h


def train_loss(logits, graph):
    # Per-seed node-mean CE summed; do not average the seed axis.
    losses = F.cross_entropy(
        logits[:, graph.train_mask].flatten(0, 1),
        graph.y[graph.train_mask].repeat(logits.shape[0]),
        reduction="none",
    )
    return losses.reshape(logits.shape[0], -1).mean(1).sum()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_dense_outputs_and_all_parameter_gradients(condition):
    graph, b, a = debug_graph()
    net = model(condition, path_chunk=4, checkpoint_paths=True)
    actual, details = net(graph, diagnostics=True)
    expected = dense_forward(net, graph, b, a)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    parameters = tuple(net.parameters())
    actual_grads = torch.autograd.grad(train_loss(actual, graph), parameters)
    dense_grads = torch.autograd.grad(train_loss(expected, graph), parameters)
    for name, got, want in zip(
        dict(net.named_parameters()), actual_grads, dense_grads, strict=True
    ):
        assert torch.isfinite(got).all(), name
        torch.testing.assert_close(got, want, rtol=2e-9, atol=2e-11)
    assert len(details) == 2
    assert all(
        not value.requires_grad
        for row in details
        for value in row.values()
        if torch.is_tensor(value)
    )


@pytest.mark.parametrize(
    "d,k,base,learned",
    [(1433, 7, 92160, 110596), (3703, 6, 237376, 255556), (500, 3, 32192, 49604)],
)
def test_full_contract_exact_active_parameter_counts(d, k, base, learned):
    for condition in CONDITIONS:
        net = PackedClassifier(condition, d, k, (11, 23), dataset_name="count")
        expected = (
            base
            if condition in ("mlp", "standard_gcn")
            else base + 2
            if condition == "first_order"
            else learned
            if condition in ("learned_wedge_raw", "learned_wedge_rms", "fixed_wedge_node_mlp")
            else base + 4
        )
        assert net.parameters_per_seed == expected
        assert net.trainable_parameters_per_seed == expected
        assert sum(p.numel() for p in net.parameters()) == 2 * expected
        groups = net.weight_decay_groups(5e-4)
        grouped = [p for group in groups for p in group["params"]]
        assert len(grouped) == len(tuple(net.parameters()))
        assert len({id(p) for p in grouped}) == len(grouped)


@pytest.mark.parametrize(
    "condition", ("learned_wedge_raw", "learned_wedge_rms", "fixed_wedge_node_mlp")
)
def test_real_classification_loss_optimizer_updates_every_parameter(condition):
    graph, _, _ = debug_graph()
    net = model(condition, path_chunk=3)
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    optimizer = torch.optim.Adam(net.weight_decay_groups(5e-4), lr=0.003)
    prediction, _ = net(graph)
    train_loss(prediction, graph).backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in net.named_parameters():
        assert not torch.equal(parameter.detach(), before[name]), name


@pytest.mark.parametrize(
    "condition", ("learned_wedge_raw", "learned_wedge_rms", "fixed_wedge_node_mlp", "standard_gcn")
)
def test_seed_packing_equals_independent_dropout_loss_gradient_and_adam_state(condition):
    graph, _, _ = debug_graph()
    packed = model(condition, seeds=(23, 11), dropout=0.5, path_chunk=4)
    singles = [model(condition, seeds=(seed,), dropout=0.5, path_chunk=7) for seed in packed.seeds]
    optimizer = torch.optim.Adam(packed.weight_decay_groups(5e-4), lr=0.003)
    single_optimizers = [
        torch.optim.Adam(single.weight_decay_groups(5e-4), lr=0.003) for single in singles
    ]
    result, _ = packed(graph, epoch=17)
    expected = torch.cat([single(graph, epoch=17)[0] for single in singles])
    torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-12)
    train_loss(result, graph).backward()
    for single in singles:
        train_loss(single(graph, epoch=17)[0], graph).backward()
    for name, parameter in packed.named_parameters():
        want = torch.cat([dict(single.named_parameters())[name].grad for single in singles])
        torch.testing.assert_close(parameter.grad, want, rtol=1e-9, atol=1e-11)
    optimizer.step()
    for single_optimizer in single_optimizers:
        single_optimizer.step()
    for name, parameter in packed.named_parameters():
        singles_named = [dict(single.named_parameters())[name] for single in singles]
        torch.testing.assert_close(parameter, torch.cat(singles_named), rtol=1e-10, atol=1e-11)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[parameter][key],
                torch.cat(
                    [
                        opt.state[p][key]
                        for opt, p in zip(single_optimizers, singles_named, strict=True)
                    ]
                ),
                rtol=1e-9,
                atol=1e-11,
            )


def test_shared_initialization_dropout_and_epoch_streams():
    raw, rms, fixed = (
        model("learned_wedge_raw", dropout=0.5),
        model(dropout=0.5),
        model("fixed_wedge", dropout=0.5),
    )
    for layer in range(2):
        assert torch.equal(raw.projections[layer], rms.projections[layer])
        assert torch.equal(raw.projections[layer], fixed.projections[layer])
        for name, value in raw.gates[layer].named_parameters():
            assert torch.equal(value, dict(rms.gates[layer].named_parameters())[name])
    h = torch.ones(2, 8, 5, dtype=torch.float64)
    mask = raw._dropout(h, 9, 0)
    assert torch.equal(mask, rms._dropout(h, 9, 0))
    assert torch.equal(mask, fixed._dropout(h, 9, 0))
    assert not torch.equal(mask, raw._dropout(h, 10, 0))
    assert not torch.equal(mask[0], mask[1])


def test_branch_initialization_and_weight_decay_scope():
    graph, _, _ = debug_graph()
    net = model("learned_wedge_raw")
    _, details = net(graph)
    for detail in details:
        torch.testing.assert_close(detail["alpha"], torch.full((2,), 0.5, dtype=torch.float64))
        torch.testing.assert_close(detail["beta"], torch.full((2,), 0.25, dtype=torch.float64))
    decay_map = {
        id(parameter): group["weight_decay"]
        for group in net.weight_decay_groups(0.0005)
        for parameter in group["params"]
    }
    for name, parameter in net.named_parameters():
        if name.startswith("projections.") or name.endswith((".w1", ".w2")):
            assert decay_map[id(parameter)] == 0.0005
        else:
            assert decay_map[id(parameter)] == 0


def test_fixed_q_cache_works_with_actual_immutable_graphdata():
    from research.wedge_propagation.classification.data import GraphData

    graph, b, a = debug_graph()
    frozen_graph = GraphData(name="DEBUG", metadata={"classes": 3}, **vars(graph))
    net = model("fixed_wedge")
    output, _ = net(frozen_graph)
    torch.testing.assert_close(output, dense_forward(net, frozen_graph, b, a))
    # A second forward uses the same static cache without adding a field to the
    # immutable source graph; its source attributes remain exactly the same.
    repeated, _ = net(frozen_graph)
    assert torch.equal(repeated, output)
    assert not hasattr(frozen_graph, "_classification_q_correction")


@pytest.mark.parametrize("condition", ("learned_wedge_raw", "learned_wedge_rms"))
def test_chunk_checkpoint_invariance_and_input_gradients(condition):
    graph, _, _ = debug_graph()
    graph.x.requires_grad_()
    whole = model(condition, path_chunk=None, checkpoint_paths=False)
    small = model(condition, path_chunk=1, checkpoint_paths=True)
    full_logits, full_detail = whole(graph, diagnostics=True)
    chunk_logits, chunk_detail = small(graph, diagnostics=True)
    torch.testing.assert_close(full_logits, chunk_logits, rtol=1e-11, atol=1e-12)
    for got, want in zip(chunk_detail, full_detail, strict=True):
        torch.testing.assert_close(got["c"], want["c"], rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(got["kappa"], want["kappa"], rtol=1e-11, atol=1e-12)
    whole_grads = torch.autograd.grad(
        train_loss(full_logits, graph), (*whole.parameters(), graph.x)
    )
    small_grads = torch.autograd.grad(
        train_loss(chunk_logits, graph), (*small.parameters(), graph.x)
    )
    for got, want in zip(small_grads, whole_grads, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-11)


def test_identity_gate_matches_fixed_wedge_at_both_layers():
    graph, _, _ = debug_graph()
    learned, fixed = model(), model("fixed_wedge")
    identity, rows = learned(graph, intervention="c_identity", diagnostics=True)
    expected, _ = fixed(graph)
    torch.testing.assert_close(identity, expected, rtol=1e-11, atol=1e-12)
    for row in rows:
        torch.testing.assert_close(row["c"], torch.ones_like(row["c"]))
        torch.testing.assert_close(row["kappa"], torch.ones_like(row["kappa"]))


def test_rms_fixed_z_invariance_and_original_amplitude_message():
    graph, _, _ = debug_graph()
    net = model().eval()
    z = torch.randn(2, 8, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(37))
    message, reference = net.probe_layer(graph, 0, z)
    for amplitude in (0.25, 0.5, 2, 4, 1e-80):
        scaled, detail = net.probe_layer(graph, 0, z, input_scale=amplitude)
        torch.testing.assert_close(detail["c"], reference["c"], rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(detail["kappa"], reference["kappa"], rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(scaled / amplitude, message, rtol=1e-11, atol=1e-12)
    output, _ = net(graph)
    changed, _ = net(graph, input_scale=4)
    torch.testing.assert_close(changed, 4 * output, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("empty", (False, True))
def test_zero_signal_and_empty_geometry_finite_forward_backward(empty):
    graph, _, _ = debug_graph(constant=True, empty=empty)
    graph.x.requires_grad_()
    net = model()
    prediction, details = net(graph, diagnostics=True)
    assert torch.isfinite(prediction).all()
    train_loss(prediction, graph).backward()
    assert torch.isfinite(graph.x.grad).all()
    for parameter in net.parameters():
        # With no paths the gate has no mathematical message and is unused for
        # this degenerate DEBUG graph; no fake dependence is introduced.
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert torch.equal(details[0]["sigma"], torch.ones(2, dtype=torch.float64))


def test_edge_orientation_path_reversal_and_translation_gate_symmetry():
    graph, _, _ = debug_graph()
    net = model().eval()
    original, rows = net(graph, diagnostics=True)
    graph.edges = graph.edges.flip(0)
    graph.paths = graph.paths.flip(0)
    changed, changed_rows = net(graph, diagnostics=True)
    torch.testing.assert_close(changed, original, rtol=1e-11, atol=1e-12)
    for got, want in zip(changed_rows, rows, strict=True):
        torch.testing.assert_close(got["c"], want["c"], rtol=1e-11, atol=1e-12)
    z = rows[0]["z"]
    _, shifted = net.probe_layer(graph, 0, z + 19)
    torch.testing.assert_close(shifted["c"], rows[0]["c"], rtol=1e-11, atol=1e-12)


def test_shuffle_hold_kappa_and_random_correspondence_dense_operator():
    graph, _, a = debug_graph()
    net = model().eval()
    z = torch.randn(2, 8, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    _, reference = net.probe_layer(graph, 0, z)
    permutation = torch.arange(a.shape[0] - 1, -1, -1)
    shuffled, detail = net.probe_layer(
        graph,
        0,
        z,
        intervention="c_position_shuffle",
        manifest={"permutation": permutation},
        kappa_mode="hold",
    )
    torch.testing.assert_close(detail["kappa"], reference["kappa"])
    assert torch.equal(detail["c"], reference["c"][:, permutation])
    scaled = z * graph.sq[None, :, None]
    expected = (
        (a.T @ (detail["c"][:, :, None] * (a @ scaled)))
        * graph.sq[None, :, None]
        / (3 * reference["kappa"][:, None, None])
    )
    torch.testing.assert_close(shuffled, expected, rtol=1e-11, atol=1e-12)
    # A four-support diagnostic including duplicate nodes, with explicit dense
    # coalescing. This verifies the random row operator independently.
    count = a.shape[0]
    indices = torch.stack(
        (
            torch.arange(count) % 7,
            (torch.arange(count) + 1) % 7,
            (torch.arange(count) + 1) % 7,
            (torch.arange(count) + 3) % 7,
        )
    )
    coefficients = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float64)[:, None].expand(
        4, count
    )
    random_rows = {"indices": indices, "coefficients": coefficients}
    random_matrix = torch.zeros(count, 8, dtype=torch.float64).index_put(
        (torch.arange(count).repeat(4), indices.flatten()), coefficients.flatten(), accumulate=True
    )
    output, detail = net.probe_layer(
        graph,
        0,
        z,
        intervention="random_physical_edge_pair_correspondence",
        manifest={"random_rows": random_rows},
    )
    expected = (
        (random_matrix.T @ (reference["c"][:, :, None] * (random_matrix @ scaled)))
        * graph.sq[None, :, None]
        / (3 * reference["kappa"][:, None, None])
    )
    torch.testing.assert_close(output, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(detail["kappa"], reference["kappa"])
    torch.testing.assert_close(
        net.apply_branch(graph, z, detail["c"], detail["kappa"], random_rows), expected
    )


def test_manifest_validation_rejects_mutated_permutation_and_wrong_row_norm():
    graph, _, a = debug_graph()
    net = model().eval()
    z = torch.randn(2, 8, 6, dtype=torch.float64)
    permutation = torch.arange(a.shape[0])
    manifest = {"permutation": permutation}
    net.probe_layer(graph, 0, z, intervention="c_position_shuffle", manifest=manifest)
    permutation[0] = permutation[1]
    with pytest.raises(ValueError, match="exactly once"):
        net.probe_layer(graph, 0, z, intervention="c_position_shuffle", manifest=manifest)
    rows = {
        "indices": graph.paths[[0, 1, 1, 2]],
        "coefficients": torch.full((4, a.shape[0]), 2.0, dtype=torch.float64),
    }
    with pytest.raises(ValueError, match="row norm"):
        net.probe_layer(
            graph,
            0,
            z,
            intervention="random_physical_edge_pair_correspondence",
            manifest={"random_rows": rows},
        )


def test_kappa_tied_maxima_equal_node_subgradient_and_norm_bound():
    graph, _, a = debug_graph()
    net = model().eval()
    c = torch.ones(2, a.shape[0], dtype=torch.float64, requires_grad=True)
    kappa = net._kappa(graph, c)
    gradient = torch.autograd.grad(kappa.sum(), c)[0]
    positive = graph.qdiag > 0
    expected = (a[:, positive].square() / graph.qdiag[positive]).mean(1)
    torch.testing.assert_close(gradient, expected[None].expand_as(gradient))
    z = torch.randn(2, 8, 6, dtype=torch.float64)
    _, details = net.probe_layer(graph, 0, z)
    for index in range(2):
        matrix = (
            graph.sq[:, None]
            * (a.T @ (details["c"][index, :, None] * a))
            * graph.sq[None, :]
            / (3 * details["kappa"][index])
        )
        eigenvalues = torch.linalg.eigvalsh(matrix)
        assert eigenvalues.min() >= -1e-13
        assert eigenvalues.max() <= 1 + 1e-13


def test_second_branch_removal_leaves_alpha_unchanged_and_uses_current_next_z():
    graph, _, _ = debug_graph()
    net = model().eval()
    original, clean = net(graph, diagnostics=True)
    removed, changed = net(graph, intervention="second_branch_remove", diagnostics=True)
    assert not torch.equal(original, removed)
    for baseline, intervention in zip(clean, changed, strict=True):
        assert torch.equal(baseline["alpha"], intervention["alpha"])
        assert torch.equal(intervention["t_message"], torch.zeros_like(intervention["t_message"]))
    assert not torch.equal(clean[1]["z"], changed[1]["z"])
    _, direct = net.probe_layer(graph, 1, changed[1]["z"])
    torch.testing.assert_close(changed[1]["c_reference"], direct["c"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not allocated")
def test_cuda_float32_forward_backward_has_no_seed_or_channel_sync():
    graph, _, _ = debug_graph(device="cuda")
    for name, value in vars(graph).items():
        if value.is_floating_point():
            setattr(graph, name, value.float())
    net = model(dropout=0.5, path_chunk=4).float().cuda()
    # Warm up checkpoint machinery before profiling the steady forward/backward.
    train_loss(net(graph, epoch=1)[0], graph).backward()
    net.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        train_loss(net(graph, epoch=2)[0], graph).backward()
    names = {event.key for event in trace.key_averages()}
    assert "aten::_local_scalar_dense" not in names
    assert "aten::item" not in names
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not allocated")
@pytest.mark.parametrize("condition", ("learned_wedge_rms", "fixed_wedge", "fixed_wedge_node_mlp"))
def test_cuda_float32_dense_math_and_dropout_counter_reproducibility(condition):
    graph, b, a = debug_graph(device="cuda")
    for name, value in vars(graph).items():
        if value.is_floating_point():
            setattr(graph, name, value.float())
    b, a = b.float(), a.float()
    net = model(condition, path_chunk=4).float().cuda()
    actual, _ = net(graph)
    expected = dense_forward(net, graph, b, a)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-6)
    got = torch.autograd.grad(train_loss(actual, graph), tuple(net.parameters()))
    want = torch.autograd.grad(train_loss(expected, graph), tuple(net.parameters()))
    for first, second in zip(got, want, strict=True):
        torch.testing.assert_close(first, second, rtol=3e-4, atol=2e-6)
    gpu_dropout, cpu_dropout = model(dropout=0.5).cuda(), model(dropout=0.5)
    cpu_input = torch.ones(2, 8, 5, dtype=torch.float64)
    assert torch.equal(
        gpu_dropout._dropout(cpu_input.cuda(), 7, 0).cpu(), cpu_dropout._dropout(cpu_input, 7, 0)
    )
