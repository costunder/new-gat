"""Separate DEBUG dense arithmetic and actual E/J classifier gradient checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from research.local_energy_relations.prediction.model import CONDITIONS, PackedClassifier
from research.local_energy_relations.prediction.operators import (
    geometry_arrays,
    local_features,
    prepare_geometry,
    restore_geometry,
    transition,
    two_hop,
)
from research.local_energy_relations.topology import build_topology


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph(device="cpu"):
    edges = torch.tensor([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6)]).T
    top = build_topology(8, edges)
    geometry = prepare_geometry(top).to(device, torch.float64)
    generator = torch.Generator().manual_seed(947)
    return SimpleNamespace(
        name="DEBUG-local",
        x=torch.randn(8, 7, generator=generator, dtype=torch.float64).to(device),
        edges=edges.to(device),
        topology=geometry.topology,
        geometry=geometry,
        y=torch.tensor([0, 1, 2, 0, 2, 1, 2, 0], device=device),
        num_classes=3,
    )


def dense_fields(edges, z, mode):
    """Dense local matrices over original coordinates, independent of flat maps."""
    seeds, n, channels = z.shape
    edge_list = edges.numpy().T
    members, laplacians, energy_weights = [], [], []
    for center in range(n):
        neighbors = {center}
        for a, b in edge_list:
            if a == center:
                neighbors.add(int(b))
            if b == center:
                neighbors.add(int(a))
        nodes = sorted(neighbors)
        selected = [(int(a), int(b)) for a, b in edge_list if a in neighbors and b in neighbors]
        incidence = np.zeros((len(selected), n))
        for row, (a, b) in enumerate(selected):
            incidence[row, a], incidence[row, b] = -1, 1
        degree = (incidence != 0).sum(0)
        c = np.asarray([1 if mode == "unit" else 2 / (degree[a] + degree[b]) for a, b in selected])
        members.append(nodes)
        laplacians.append(incidence.T @ np.diag(c) @ incidence)
        energy_weights.append(c.sum())
    center_l = np.zeros((n, n))
    for u in range(n):
        for v in members[u]:
            if v != u:
                center_l[u] += laplacians[v][u]
    p = np.eye(n)
    for u in range(n):
        if center_l[u, u]:
            p[u] -= center_l[u] / center_l[u, u]
    divergence = [np.einsum("ij,sjf->sif", matrix, z) for matrix in laplacians]
    e = np.stack([np.einsum("snf,nm,smf->s", z, matrix, z) for matrix in laplacians], 1)
    ep = (
        e
        / np.where(np.asarray(energy_weights) > 0, 2 * channels * np.asarray(energy_weights), 1)[
            None
        ]
    )
    pairs = np.concatenate((edges.numpy(), edges.numpy()[::-1]), 1).T
    raw_j, normalized = [], []
    for v, u in pairs:
        overlap = sorted(set(members[v]) & set(members[u]))
        value = (divergence[v][:, overlap] * divergence[u][:, overlap]).sum((1, 2))
        norm = channels * np.sqrt((laplacians[v] ** 2).sum() * (laplacians[u] ** 2).sum())
        raw_j.append(value)
        normalized.append(value / norm)
    raw_j, normalized = np.stack(raw_j, 1), np.stack(normalized, 1)
    jp = np.zeros((seeds, n))
    for pair, (_, u) in enumerate(pairs):
        jp[:, u] += normalized[:, pair]
    degrees = np.asarray([len(nodes) - 1 for nodes in members])
    jp /= np.maximum(degrees, 1)[None]
    return p, ep, jp, e, raw_j


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_dense_current_local_forms_and_two_hop_agree(mode):
    g = graph()
    z = torch.stack((g.x, g.x * -0.7))
    p, ep, jp, raw_e, raw_j = dense_fields(g.edges, z.numpy(), mode)
    actual = local_features(g.geometry, mode, z, chunk=2, checkpoint_chunks=False)
    for value, expected in zip(actual, (ep, jp, raw_e, raw_j), strict=True):
        np.testing.assert_allclose(value.numpy(), expected, rtol=1e-12, atol=1e-12)
    expected_p = np.einsum("nm,smf->snf", p, z.numpy())
    np.testing.assert_allclose(transition(g.geometry, mode, z).numpy(), expected_p, atol=1e-12)
    np.testing.assert_allclose(
        two_hop(g.geometry, mode, z).numpy(), np.einsum("nm,smf->snf", p @ p, z.numpy()), atol=1e-12
    )
    torch.testing.assert_close(transition(g.geometry, mode, z)[:, 7], z[:, 7])
    # C is fixed and normalization topology-only: all raw and normalized forms
    # preserve quadratic homogeneity instead of silently becoming RMS features.
    scaled = local_features(g.geometry, mode, 3 * z, chunk=1, checkpoint_chunks=False)
    for value, original in zip(scaled, actual, strict=True):
        torch.testing.assert_close(value, 9 * original, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_exact_chunk_checkpoint_gradients_and_geometry_restore(mode):
    g = graph()
    arrays = geometry_arrays(g.geometry)
    restored = restore_geometry(g.topology, arrays, g.geometry.metadata)
    with pytest.raises(ValueError, match="unknown or missing"):
        restore_geometry(g.topology, arrays | {"unknown": np.zeros(1)}, {})
    fields = torch.stack((g.x, 0.8 * g.x)).requires_grad_()
    reference = local_features(g.geometry, mode, fields, checkpoint_chunks=False)
    reference_loss = reference[0].square().sum() + reference[1].square().sum()
    expected_gradient = torch.autograd.grad(reference_loss, fields)[0]
    for chunk in (1, 3, 100):
        actual = local_features(restored, mode, fields, chunk=chunk, checkpoint_chunks=True)
        for value, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
        gradient = torch.autograd.grad(actual[0].square().sum() + actual[1].square().sum(), fields)[
            0
        ]
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_initial_forward_common_seed_projection_dropout_and_active_parameters(mode):
    g = graph()
    initial = None
    for variant in ("base", "within", "between", "both"):
        model = (
            PackedClassifier(
                f"{mode}__{variant}", 7, 3, [11, 23], hidden=9, path_chunk=2, dataset_name=g.name
            )
            .double()
            .train()
        )
        logits, _ = model(g, epoch=17)
        if initial is None:
            initial = logits
        else:
            torch.testing.assert_close(logits, initial, rtol=0, atol=0)
        branches = int(model.has_energy) + int(model.has_relation)
        assert model.parameters_per_seed == 7 * 9 + 9 * 3 + 2 + branches * (9 + 3)
        assert model.trainable_parameters_per_seed == model.parameters_per_seed
        assert not any("dropout" in key for key in model.state_dict())
        assert all(value.shape[0] == 2 for value in model.state_dict().values())
        assert len(model.energy_lifts) == (2 if model.has_energy else 0)
        assert len(model.relation_lifts) == (2 if model.has_relation else 0)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_real_classification_loss_gradient_update_and_frozen_branch_effect(condition):
    g = graph()
    model = PackedClassifier(
        condition, 7, 3, [11, 23], hidden=9, dropout=0, path_chunk=2, dataset_name=g.name
    ).double()
    optimizer = torch.optim.Adam(model.weight_decay_groups(5e-4), lr=0.01)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    logits, _ = model(g)
    loss = sum(F.cross_entropy(item[:7], g.y[:7]) for item in logits)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name]), name
    model.eval()
    logits, details = model(g, diagnostics=True)
    removed, _ = model(g, intervention="both_remove")
    if model.variant == "base":
        torch.testing.assert_close(logits, removed, rtol=0, atol=0)
    else:
        assert (logits - removed).abs().max() > 0
        g.x = g.x.detach().requires_grad_()
        live, _ = model(g)
        disabled, _ = model(g, intervention="both_remove")
        probe = torch.arange(live.numel(), dtype=live.dtype).reshape(live.shape) / live.numel()
        gradient = torch.autograd.grad(((live - disabled) * probe).sum(), g.x)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    for detail in details:
        for value in detail.values():
            assert torch.isfinite(value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_and_gradient_match_cpu():
    cpu, gpu = graph(), graph("cuda")
    a = PackedClassifier(
        "local_degree__both",
        7,
        3,
        [11, 23],
        hidden=9,
        dropout=0,
        path_chunk=3,
        dataset_name=cpu.name,
    ).double()
    b = (
        PackedClassifier(
            "local_degree__both",
            7,
            3,
            [11, 23],
            hidden=9,
            dropout=0,
            path_chunk=3,
            dataset_name=cpu.name,
        )
        .double()
        .cuda()
    )
    la, _ = a(cpu)
    lb, _ = b(gpu)
    torch.testing.assert_close(la, lb.cpu(), rtol=1e-10, atol=1e-10)
    sum(F.cross_entropy(x[:7], cpu.y[:7]) for x in la).backward()
    sum(F.cross_entropy(x[:7], gpu.y[:7]) for x in lb).backward()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad.cpu(), rtol=1e-9, atol=1e-10)
