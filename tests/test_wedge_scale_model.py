"""Meaningful unit/DEBUG checks of Experiment 3.1, without production training."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from research.wedge_propagation.learned.data import RuleCase, pack_cases, prepare_cases
from research.wedge_propagation.learned.model import (
    apply_weighted_pair,
    apply_weighted_wedge,
    make_model,
    normalized_mse,
    teacher_weights,
)
from research.wedge_propagation.operators import dense_incidence, dense_wedge
from research.wedge_propagation.scale_normalization.model import (
    NormalizedSeedBatchedModel,
    make_normalized_model,
    physical_edge_rms,
)

_KINDS = ("learned", "random_pair")
_SEEDS = [11, 23, 37, 53, 71]
_LEARNED = Path(__file__).resolve().parents[1] / "research/wedge_propagation/learned"


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def cases():
    config = json.loads((_LEARNED / "config_debug.json").read_text(encoding="utf-8"))
    return prepare_cases(config, workers=2)


@pytest.fixture
def batch(cases):
    return pack_cases(cases, "cpu", torch.float64)


def _empty_case(base: RuleCase) -> RuleCase:
    """Explicit mathematical zero-edge DEBUG case, retaining all its nodes."""
    n, r = base.features.shape
    indices = torch.empty((2, 0), dtype=torch.long)
    zero = torch.zeros((n, r), dtype=torch.float64)
    return replace(base, graph_id="debug-empty-physical-graph", family="empty",
                   edges=indices, wedges=torch.empty((3, 0), dtype=torch.long),
                   pair_edges=indices.clone(), pair_coefficients=torch.empty((2, 0)).double(),
                   teacher_c=torch.empty((0, r)).double(), lx=zero.clone(), l2x=zero.clone(),
                   qx=zero.clone(), target_path=zero.clone())


def _dense_sigma(batch):
    """Independent graph-wise dense incidence reference; test-only loops."""
    b = dense_incidence(batch.edges, batch.x.shape[0], device=batch.x.device,
                        dtype=batch.x.dtype)
    differences = b @ batch.x
    graph = batch.node_graph[batch.edges[0]]
    values = []
    for index in range(batch.num_graphs):
        current = differences[graph == index]
        energy = current.square().mean(0) if len(current) else batch.x.new_zeros(batch.x.shape[1])
        values.append(torch.where(energy > 0, energy, torch.ones_like(energy)).sqrt())
    return torch.stack(values)


@pytest.mark.parametrize("kind", _KINDS)
def test_exact_parameters_initialization_state_keys_and_full_capacity(kind):
    old, new = make_model(kind, _SEEDS), make_normalized_model(kind, _SEEDS)
    assert isinstance(new, NormalizedSeedBatchedModel)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert sum(parameter.numel() for parameter in new.parameters()) == 386 * 5
    assert new.gate.w1.shape == (5, 4, 64)
    assert new.gate.b1.shape == new.gate.w2.shape == (5, 64)
    for name, parameter in old.state_dict().items():
        assert torch.equal(parameter, new.state_dict()[name])
    assert not list(new.buffers())
    new.load_state_dict(old.state_dict(), strict=True)


@pytest.mark.parametrize("kind", ["first", "polynomial", "fixed", "invalid"])
def test_normalized_factory_explicitly_rejects_non_gate_models(kind):
    with pytest.raises(ValueError, match="only learned and random_pair"):
        make_normalized_model(kind, _SEEDS)


def test_rms_uses_all_physical_edges_per_graph_and_scalar_realization(batch):
    actual = physical_edge_rms(batch)
    torch.testing.assert_close(actual, _dense_sigma(batch), rtol=1e-12, atol=1e-12)
    assert actual.shape == (batch.num_graphs, batch.x.shape[1])
    assert (actual > 0).all()


@pytest.mark.parametrize("kind", _KINDS)
def test_positive_amplitudes_keep_c_invariant_and_original_message_equivariant(kind, batch):
    model = make_normalized_model(kind, _SEEDS)
    reference, c = model(batch)
    assert reference.shape == (5, batch.x.shape[0], batch.x.shape[1])
    assert c.shape == (5, batch.wedges.shape[1], batch.x.shape[1])
    for amplitude in (0.25, 0.5, 1, 2, 4):
        scaled = replace(batch, x=batch.x * amplitude)
        actual, weights = model(scaled)
        torch.testing.assert_close(weights, c, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual, reference * amplitude, rtol=1e-12, atol=1e-12)
        if kind == "learned":
            direct = apply_weighted_wedge(scaled.wedges, scaled.x, weights)
        else:
            direct = apply_weighted_pair(scaled.edges, scaled.pair_edges,
                                         scaled.pair_coefficients, scaled.x, weights)
        torch.testing.assert_close(actual, model.beta[:, None, None] * direct,
                                   rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", _KINDS)
def test_independent_graph_and_feature_amplitudes_do_not_mix_groups(kind, cases):
    group = [cases[0], cases[2], cases[-1]]
    batch = pack_cases(group, "cpu", torch.float64)
    model = make_normalized_model(kind, [11, 23])
    reference, weights = model(batch)
    scales = torch.tensor([[0.25, 0.5, 1, 2], [4, 1, 0.5, 0.25], [2, 4, 0.25, 0.5]]).double()
    x = batch.x * scales[batch.node_graph]
    actual, c = model(replace(batch, x=x))
    torch.testing.assert_close(c, weights, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, reference * scales[batch.node_graph][None],
                               rtol=1e-12, atol=1e-12)
    node_start = path_start = 0
    for case in group:
        solo, solo_c = model(pack_cases([case], "cpu", torch.float64))
        n, p = case.num_nodes, case.wedges.shape[1]
        torch.testing.assert_close(solo, reference[:, node_start:node_start + n],
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(solo_c, weights[:, path_start:path_start + p],
                                   rtol=1e-12, atol=1e-12)
        node_start, path_start = node_start + n, path_start + p


@pytest.mark.parametrize("kind", _KINDS)
def test_nonzero_scales_have_no_additive_epsilon(kind, batch):
    model = make_normalized_model(kind, [11, 23])
    reference, reference_c = model(batch)
    amplitude = 1e-80  # Float64 squares remain nonzero and representable.
    tiny = replace(batch, x=batch.x * amplitude)
    sigma = physical_edge_rms(tiny)
    torch.testing.assert_close(sigma / amplitude, physical_edge_rms(batch),
                               rtol=1e-12, atol=1e-12)
    actual, c = model(tiny)
    torch.testing.assert_close(c, reference_c, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(actual / amplitude, reference, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("kind", _KINDS)
def test_translation_and_corresponding_orientation_changes_preserve_output(kind, batch):
    model = make_normalized_model(kind, [11, 23])
    reference, weights = model(batch)
    offsets = torch.arange(batch.num_graphs * batch.x.shape[1]).reshape(
        batch.num_graphs, -1).double() / 8
    moved, moved_c = model(replace(batch, x=batch.x + offsets[batch.node_graph]))
    torch.testing.assert_close(moved, reference, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(moved_c, weights, rtol=1e-11, atol=1e-11)
    signs = torch.where(torch.arange(batch.edges.shape[1]) % 3 == 0, -1.0, 1.0).double()
    edges = batch.edges.clone()
    edges[:, signs < 0] = edges.flip(0)[:, signs < 0]
    changed = replace(batch, edges=edges,
                      pair_coefficients=batch.pair_coefficients * signs[batch.pair_edges])
    oriented, oriented_c = model(changed)
    torch.testing.assert_close(oriented, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(oriented_c, weights, rtol=1e-12, atol=1e-12)
    if kind == "learned":
        reversed_batch = replace(batch, wedges=batch.wedges.flip(0))
    else:
        reversed_batch = replace(batch, pair_edges=batch.pair_edges.flip(0),
                                 pair_coefficients=-batch.pair_coefficients.flip(0))
    reversed_output, reversed_c = model(reversed_batch)
    torch.testing.assert_close(reversed_output, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(reversed_c, weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", _KINDS)
def test_empty_constant_and_zero_feature_columns_have_finite_forward_backward(kind, cases):
    batch = pack_cases([cases[0], _empty_case(cases[1]), cases[-1]], "cpu", torch.float64)
    x = batch.x.clone()
    x[:, 0] = 0
    x[:, 1] = (batch.node_graph + 3).double()
    x[batch.node_graph == 2] = 7
    x.requires_grad_()
    batch = replace(batch, x=x)
    model = make_normalized_model(kind, [11, 23])
    sigma = physical_edge_rms(batch)
    torch.testing.assert_close(sigma[:, :2], torch.ones_like(sigma[:, :2]), rtol=0, atol=0)
    torch.testing.assert_close(sigma[1:], torch.ones_like(sigma[1:]), rtol=0, atol=0)
    output, c = model(batch)
    assert torch.isfinite(output).all() and torch.isfinite(c).all()
    assert torch.count_nonzero(output[:, :, :2]) == 0
    assert torch.count_nonzero(output[:, batch.node_graph != 0]) == 0
    (output.square().sum() + c.square().sum() + sigma.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("kind", _KINDS)
def test_gate_beta_input_gradients_and_optimizer_updates_are_real(kind, batch):
    x = batch.x.clone().requires_grad_()
    current = replace(batch, x=x)
    model = make_normalized_model(kind, _SEEDS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    output, _ = model(current)
    loss = normalized_mse(output, batch.targets["path"], batch.node_graph, batch.num_graphs)
    loss.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name])


@pytest.mark.parametrize("kind", _KINDS)
def test_input_gradient_matches_independent_dense_normalized_operator(kind, cases):
    batch = pack_cases([cases[0]], "cpu", torch.float64)
    x = batch.x.clone().requires_grad_()
    batch = replace(batch, x=x)
    model = make_normalized_model(kind, [11, 23])
    actual, _ = model(batch)
    b = dense_incidence(batch.edges, len(x))
    differences = b @ x
    sigma = differences.square().mean(0).sqrt()
    if kind == "learned":
        a = dense_wedge(batch.wedges, len(x))
        i, j, k = batch.wedges
        g1, g2 = x[j] - x[i], x[k] - x[j]
    else:
        left, right = batch.pair_edges
        s1, s2 = batch.pair_coefficients[:, :, None]
        a = -s1 * b[left] + s2 * b[right]
        g1, g2 = s1 * differences[left], s2 * differences[right]
    c = model.gate(g1 / sigma, g2 / sigma,
                   batch.path_graph, batch.num_graphs)
    dense = model.beta[:, None, None] * torch.einsum("pn,spr,pr->snr", a, c, a @ x)
    torch.testing.assert_close(actual, dense, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    dense_grad = torch.autograd.grad(dense.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, dense_grad, rtol=1e-11, atol=1e-10)


def test_static_scale_cache_invalidates_on_in_place_input_and_node_group_changes(cases):
    batch = pack_cases([cases[0], cases[-1]], "cpu", torch.float64)
    first = physical_edge_rms(batch)
    assert physical_edge_rms(batch) is first
    batch.x.mul_(2)
    second = physical_edge_rms(batch)
    torch.testing.assert_close(second, first * 2, rtol=1e-12, atol=1e-12)
    batch.node_graph.copy_(1 - batch.node_graph)
    third = physical_edge_rms(batch)
    torch.testing.assert_close(third, second.flip(0), rtol=1e-12, atol=1e-12)


def test_differentiable_input_scales_are_not_reused_between_backward_graphs(batch):
    x = batch.x.clone().requires_grad_()
    current = replace(batch, x=x)
    first, second = physical_edge_rms(current), physical_edge_rms(current)
    assert first is not second and first.requires_grad and second.requires_grad
    first.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    x.grad = None
    second.square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("kind", _KINDS)
def test_normalized_model_changes_raw_gate_response_without_changing_teacher(kind, batch):
    raw, normalized = make_model(kind, [11, 23]), make_normalized_model(kind, [11, 23])
    before = batch.teacher_c.clone()
    _, raw_c = raw(batch)
    _, new_c = normalized(batch)
    assert not torch.allclose(raw_c, new_c, rtol=1e-5, atol=1e-5)
    altered_teacher = replace(batch, teacher_c=batch.teacher_c * 7,
                              targets={name: value * 5 for name, value in batch.targets.items()})
    output, c = normalized(batch)
    other_output, other_c = normalized(altered_teacher)
    assert torch.equal(output, other_output) and torch.equal(c, other_c)
    assert torch.equal(batch.teacher_c, before)


def test_malformed_cross_graph_physical_edge_is_rejected(cases):
    batch = pack_cases([cases[0], cases[1]], "cpu", torch.float64)
    edges = batch.edges.clone()
    edges[1, 0] = cases[0].num_nodes
    with pytest.raises(ValueError, match="within its graph"):
        physical_edge_rms(replace(batch, edges=edges))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA unit check requires a GPU")
def test_cuda_amplitude_equivariance_finite_backward_and_steady_no_scalar_sync(kind, cases):
    batch = pack_cases(cases, "cuda", torch.float32)
    model = make_normalized_model(kind, _SEEDS).to(device="cuda", dtype=torch.float32)
    reference, c = model(batch)
    actual, scaled_c = model(replace(batch, x=batch.x * 4))
    torch.testing.assert_close(scaled_c, c, rtol=1e-5, atol=1e-5)
    # FP32 atomic sums can change order, especially where signed terms cancel.
    # Compare relative operator-action RMS rather than dividing by one nearly
    # cancelled node entry. Float64 CPU tests establish the exact identity.
    relative = (torch.linalg.vector_norm(actual - reference * 4, dim=(1, 2))
                / torch.linalg.vector_norm(reference * 4, dim=(1, 2)))
    assert relative.max() < 2e-6
    x = batch.x.clone()
    x[:, 0] = 0
    x.requires_grad_()
    output, weights = model(replace(batch, x=x))
    (output.square().mean() + weights.square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    # Warm the shared topology/group caches before measuring a steady forward.
    model(batch)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        model(batch)
    keys = {event.key for event in profile.key_averages()}
    assert "aten::_local_scalar_dense" not in keys and "aten::item" not in keys


def test_teacher_inputs_remain_the_original_physical_differences(batch):
    i, j, k = batch.wedges
    c = teacher_weights(batch.x[j] - batch.x[i], batch.x[k] - batch.x[j],
                         batch.path_graph, batch.num_graphs)
    torch.testing.assert_close(c, batch.teacher_c, rtol=1e-12, atol=1e-12)
