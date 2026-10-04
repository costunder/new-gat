"""DEBUG arithmetic, active-layer gradients and unchanged-control checks."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.local_energy_relations.placement.model import (
    CONDITIONS, PLACEMENTS, PackedClassifier, active_layers, parse_condition,
)
from research.local_energy_relations.prediction.model import PackedClassifier as Previous
from research.local_energy_relations.prediction.operators import prepare_geometry
from research.local_energy_relations.topology import build_topology


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph(device="cpu"):
    edges = torch.tensor([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6)]).T
    geometry = prepare_geometry(build_topology(8, edges)).to(device, torch.float64)
    generator = torch.Generator().manual_seed(947)
    return SimpleNamespace(
        name="DEBUG-local-placement",
        x=torch.randn(8, 7, generator=generator, dtype=torch.float64).to(device),
        topology=geometry.topology,
        geometry=geometry,
        y=torch.tensor([0, 1, 2, 0, 2, 1, 2, 0], device=device),
    )


def make(condition, g, seeds=(11, 23), dropout=0):
    return PackedClassifier(condition, 7, 3, seeds, hidden=9, dropout=dropout,
                            path_chunk=2, dataset_name=g.name).double().to(g.x.device)


@pytest.mark.parametrize("bad", ["unit__within", "unit__base__hidden", "unit__both__none",
                                 "unit__both__first", "unknown__base", None, 1])
def test_condition_requires_explicit_placement_and_base_is_unique(bad):
    with pytest.raises(ValueError, match="condition"):
        parse_condition(bad)
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 20
    assert active_layers("unit__base") == ()


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_identical_initial_forward_dropout_and_only_active_lift_parameters(mode):
    g = graph()
    reference = make(f"{mode}__base", g, dropout=0.5).train()(g, epoch=17)[0]
    for condition in (c for c in CONDITIONS if parse_condition(c)[0] == mode):
        model = make(condition, g, dropout=0.5).train()
        value, details = model(g, epoch=17)
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        layers = active_layers(condition)
        for lifts, enabled in ((model.energy_lifts, model.has_energy),
                               (model.relation_lifts, model.has_relation)):
            assert set(lifts) == ({str(layer) for layer in layers} if enabled else set())
        extra = (int(model.has_energy) + int(model.has_relation)) * sum((9, 3)[l] for l in layers)
        assert model.parameters_per_seed == 7 * 9 + 9 * 3 + 2 + extra
        assert model.trainable_parameters_per_seed == model.parameters_per_seed
        assert all(value.shape[0] == 2 for value in model.state_dict().values())
        for layer, detail in enumerate(details):
            assert bool(detail["energy_enabled"].all()) == (model.has_energy and layer in layers)
            assert bool(detail["relation_enabled"].all()) == (model.has_relation and layer in layers)


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("variant", ["base", "within", "between", "both"])
def test_fresh_base_and_all_controls_match_previous_forward_gradients_and_adam(mode, variant):
    g = graph()
    condition = f"{mode}__{variant}" + ("__all" if variant != "base" else "")
    current = make(condition, g, dropout=0.5)
    old = Previous(f"{mode}__{variant}", 7, 3, [11, 23], hidden=9,
                   dropout=0.5, path_chunk=2, dataset_name=g.name).double()
    assert current.state_dict().keys() == old.state_dict().keys()
    optimizers = [torch.optim.Adam(m.weight_decay_groups(5e-4), lr=0.003) for m in (current, old)]
    for epoch in range(3):
        a, b = current(g, epoch=epoch)[0], old(g, epoch=epoch)[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for m, logits, optimizer in zip((current, old), (a, b), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            sum(F.cross_entropy(v[:7], g.y[:7]) for v in logits).backward()
        for (key, p), (oldkey, oldp) in zip(current.named_parameters(), old.named_parameters(), strict=True):
            assert key == oldkey
            torch.testing.assert_close(p.grad, oldp.grad, atol=0, rtol=0)
        for optimizer in optimizers:
            optimizer.step()
        for key, value in current.state_dict().items():
            torch.testing.assert_close(value, old.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_actual_ce_updates_all_registered_parameters_and_features_affect_input_gradient(condition):
    g = graph()
    model = make(condition, g)
    optimizer = torch.optim.Adam(model.weight_decay_groups(5e-4), lr=0.01)
    registered = {id(p) for p in model.parameters()}
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == registered
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    logits, _ = model(g)
    sum(F.cross_entropy(v[:7], g.y[:7]) for v in logits).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0, name
    optimizer.step()
    for name, p in model.named_parameters():
        assert not torch.equal(before[name], p), name
    model.eval()
    logits, details = model(g, diagnostics=True)
    removed, _ = model(g, intervention="both_remove")
    if model.variant == "base":
        torch.testing.assert_close(logits, removed, rtol=0, atol=0)
    else:
        assert (logits - removed).abs().max() > 0
        g.x = g.x.detach().requires_grad_()
        live, _ = model(g)
        removed, _ = model(g, intervention="both_remove")
        probe = torch.arange(live.numel(), dtype=live.dtype).reshape(live.shape) / live.numel()
        gradient = torch.autograd.grad(((live - removed) * probe).sum(), g.x)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    for layer, detail in enumerate(details):
        assert all(torch.isfinite(value).all() for value in detail.values())
        if layer not in model.injection_layers:
            for prefix in ("energy", "relation"):
                assert detail[f"{prefix}_branch_norm"].eq(0).all()
                assert detail[f"{prefix}_feature_norm"].eq(0).all()


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("placement", ["hidden", "output"])
def test_single_layer_injection_matches_previous_nonzero_model_with_other_layer_removed(mode, placement):
    g = graph()
    old = Previous(f"{mode}__both", 7, 3, [11, 23], hidden=9, dropout=0,
                   path_chunk=2, dataset_name=g.name).double().eval()
    with torch.no_grad():
        for lift in (*old.energy_lifts, *old.relation_lifts):
            lift.copy_(torch.arange(lift.numel()).reshape(lift.shape) * 0.013 - 0.2)
    model = make(f"{mode}__both__{placement}", g).eval()
    state = {key: value for key, value in old.state_dict().items() if key in model.state_dict()}
    model.load_state_dict(state, strict=True)
    inactive = (1,) if placement == "hidden" else (0,)
    expected, _ = old(g, intervention="both_remove", intervention_layers=inactive)
    actual, _ = model(g)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_inactive_layers_skip_local_features_computation(monkeypatch):
    import research.local_energy_relations.placement.model as module
    original = module.local_features
    calls = []

    def record(geometry, mode, z, **kwargs):
        calls.append(z.shape[-1])
        return original(geometry, mode, z, **kwargs)

    monkeypatch.setattr(module, "local_features", record)
    g = graph()
    for placement, widths in (("hidden", [9]), ("output", [3]), ("all", [9, 3])):
        calls.clear()
        make(f"unit__both__{placement}", g)(g)
        assert calls == widths
    calls.clear()
    make("unit__base", g)(g)
    assert not calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_placement_cuda_forward_and_gradient_matches_cpu(placement):
    cpu, gpu = graph(), graph("cuda")
    a, b = make(f"local_degree__both__{placement}", cpu), make(f"local_degree__both__{placement}", gpu)
    la, lb = a(cpu)[0], b(gpu)[0]
    torch.testing.assert_close(la, lb.cpu(), rtol=1e-10, atol=1e-10)
    sum(F.cross_entropy(x[:7], cpu.y[:7]) for x in la).backward()
    sum(F.cross_entropy(x[:7], gpu.y[:7]) for x in lb).backward()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad.cpu(), rtol=1e-9, atol=1e-10)
