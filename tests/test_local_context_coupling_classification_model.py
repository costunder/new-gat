"""DEBUG mathematical, parameter, matched-control and frozen-evaluation tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from research.local_context_coupling.classification.evaluation import frozen_evaluate
from research.local_context_coupling.classification.model import CONDITIONS, PackedClassifier, parse_condition
from research.local_context_coupling.model import PackedClassifier as ParentClassifier
from research.local_context_coupling.operators import apply_cross, apply_intra, sandwich
from research.local_context_coupling.operators import prepare_geometry
from research.local_energy_relations.topology import build_topology
from research.wedge_propagation.classification.evaluation import model_state_hash


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph(dtype=torch.float64, device="cpu", zero=False):
    # Nonidentical overlapping locals plus two isolated physical nodes.
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 3), (2, 4)]
    edges = np.array(pairs, dtype=np.int64).T.copy()
    topology = build_topology(7, edges)
    geometries = {mode: prepare_geometry(topology, mode).to(device, dtype)
                  for mode in ("unit", "local_degree")}
    generator = torch.Generator().manual_seed(422)
    x = torch.randn(7, 6, generator=generator, dtype=dtype).to(device)
    if zero:
        x.zero_()
    data = SimpleNamespace(
        name="DEBUG_nonidentical_locals", x=x, geometries=geometries,
        y=torch.tensor([0, 1, 2, 1, 2, 0, 1], dtype=torch.long, device=device),
        train_mask=torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.bool, device=device),
        val_mask=torch.tensor([0, 0, 0, 1, 1, 0, 0], dtype=torch.bool, device=device),
        test_mask=torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.bool, device=device),
        num_nodes=7, num_features=6, num_classes=3,
    )
    data.geometry_for = lambda mode: data.geometries[mode]
    return data


def model(condition, data, seeds=(0, 3), **kwargs):
    return PackedClassifier(
        condition, data.x.shape[1], 3, seeds, dataset_name=data.name, **kwargs,
    ).to(device=data.x.device, dtype=data.x.dtype)


def _parent_forward(candidate, data, gains, epoch=0):
    h = data.x.unsqueeze(0).expand(len(candidate.seeds), -1, -1)
    for layer, weight in enumerate(candidate.weights):
        z = torch.bmm(candidate._dropout(h, epoch, layer), weight)
        h = sandwich(data.geometry_for(candidate.weight_mode), z, cross_gain=gains[layer])
        if layer == 0:
            h = h.relu()
    return h


@pytest.mark.parametrize("condition", CONDITIONS)
def test_active_parameter_contract_and_initialization_matches_verified_parent(condition):
    data = graph()
    candidate = model(condition, data)
    mode, variant = parse_condition(condition)
    reference = ParentClassifier(
        6, 3, (0, 3), condition="cross_on" if variant == "learned" else "cross_off",
        mode=mode, hidden=64, dropout=.5, dataset_name=data.name,
    ).to(torch.float64)
    assert candidate.num_layers == 2 and candidate.hidden == 64
    assert candidate.parameters_per_seed == 6 * 64 + 64 * 3 + int(variant == "learned")
    assert candidate.trainable_parameters_per_seed == candidate.parameters_per_seed
    for actual, expected in zip(candidate.weights, reference.weights, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate.dropout_keys, reference.dropout_keys, rtol=0, atol=0)
    assert (candidate.theta_cross is not None) == (variant == "learned")
    expected_gain = {"off": 0., "fixed": 1., "learned": .5}[variant]
    torch.testing.assert_close(candidate.cross_gain, data.x.new_full((2,), expected_gain))
    groups = candidate.weight_decay_groups(.0005)
    active = {id(parameter) for group in groups for parameter in group["params"]}
    assert active == {id(parameter) for parameter in candidate.parameters()}
    if variant == "learned":
        assert groups[-1]["weight_decay"] == 0


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
@pytest.mark.parametrize("chunk", [1, 3, None])
def test_exact_edge_chunk_forward_and_backward_match_parent_sandwich(mode, chunk):
    data = graph()
    candidate = model(f"{mode}__learned", data, edge_chunk=chunk, checkpoint_paths=False)
    geo = data.geometry_for(mode)
    z = torch.randn(2, 7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(341), requires_grad=True)
    rho = torch.tensor([.25, .73], dtype=torch.float64, requires_grad=True)
    actual = candidate._macro(geo, z, rho)
    expected = sandwich(geo, z, cross_gain=rho)
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)
    probe = torch.randn(actual.shape, dtype=actual.dtype, generator=torch.Generator().manual_seed(159))
    actual_grad = torch.autograd.grad((actual * probe).sum(), (z, rho), create_graph=True, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * probe).sum(), (z, rho), create_graph=True, retain_graph=True)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-12, atol=3e-13)
    torch.testing.assert_close(candidate._action(geo, geo.topology.local_node_global.new_zeros((2, geo.num_copies, 4), dtype=z.dtype), "intra"),
                               z.new_zeros(2, geo.num_copies, 4))


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_symmetric_custom_backward_and_second_derivative(mode):
    data = graph()
    candidate = model(f"{mode}__learned", data, edge_chunk=2, checkpoint_paths=False)
    geo = data.geometry_for(mode)
    y = torch.randn(2, geo.num_copies, 3, dtype=data.x.dtype, requires_grad=True)
    for kind, reference in (("intra", apply_intra), ("cross", apply_cross)):
        result = candidate._action(geo, y, kind)
        expected = reference(geo, y)
        torch.testing.assert_close(result, expected, rtol=1e-13, atol=1e-13)
        grad = torch.autograd.grad(result.square().sum(), y, create_graph=True)[0]
        expected_grad = 2 * reference(geo, reference(geo, y))
        torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
        second = torch.autograd.grad(grad.sum(), y)[0]
        torch.testing.assert_close(second, torch.zeros_like(y), rtol=0, atol=1e-12)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("condition", CONDITIONS)
def test_full_two_macro_layer_loss_gradients_and_optimizer_update(condition, checkpoint):
    data = graph()
    candidate = model(condition, data, edge_chunk=2, checkpoint_paths=checkpoint)
    candidate.train()
    logits, details = candidate(data, epoch=2)
    assert logits.shape == (2, 7, 3) and details == []
    expected = _parent_forward(candidate, data, (candidate.cross_gain, candidate.cross_gain), epoch=2)
    torch.testing.assert_close(logits, expected, rtol=2e-12, atol=2e-13)
    ce = F.cross_entropy(logits[:, data.train_mask].reshape(-1, 3), data.y[data.train_mask].repeat(2))
    ce.backward()
    for parameter in candidate.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    before = {key: parameter.detach().clone() for key, parameter in candidate.named_parameters()}
    optimizer = torch.optim.Adam(candidate.weight_decay_groups(.0005), lr=.01)
    optimizer.step()
    assert all(not torch.equal(parameter, before[key]) for key, parameter in candidate.named_parameters())


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_matched_gain_interventions_initialization_dropout_and_downstream_recompute(mode):
    data = graph()
    off, fixed, learned = [model(f"{mode}__{variant}", data) for variant in ("off", "fixed", "learned")]
    for candidate in (off, fixed, learned):
        candidate.train()
    for layer in (0, 1):
        h = data.x.unsqueeze(0).expand(2, -1, -1)
        torch.testing.assert_close(off._dropout(h, 7, layer), fixed._dropout(h, 7, layer), rtol=0, atol=0)
        torch.testing.assert_close(off._dropout(h, 7, layer), learned._dropout(h, 7, layer), rtol=0, atol=0)
    original_hash = model_state_hash(learned)
    off_logits = off(data, epoch=7)[0]
    zeroed = learned(data, epoch=7, intervention="gain0", intervention_layers=(0, 1))[0]
    torch.testing.assert_close(off_logits, zeroed, rtol=0, atol=0)
    fixed_logits = fixed(data, epoch=7)[0]
    one = learned(data, epoch=7, intervention="gain1", intervention_layers=(0, 1))[0]
    torch.testing.assert_close(fixed_logits, one, rtol=0, atol=0)
    changed, details = learned(data, epoch=7, intervention="gain0", intervention_layers=(0,), diagnostics=True)
    expected = _parent_forward(learned, data, (learned.cross_gain.new_zeros(2), learned.cross_gain), epoch=7)
    torch.testing.assert_close(changed, expected, rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(details[0]["rho"], torch.zeros(2, dtype=data.x.dtype))
    torch.testing.assert_close(details[1]["rho"], learned.cross_gain)
    assert details[0]["matched_delta_norm"].eq(0).all()
    assert details[1]["matched_delta_norm"].gt(0).all()
    assert model_state_hash(learned) == original_hash


def test_seed_packing_matches_individual_seeds_in_training_forward_and_gradient():
    data = graph()
    packed = model("unit__learned", data, seeds=(0, 3), edge_chunk=2)
    full, _ = packed(data, epoch=8)
    full.square().sum().backward()
    for index, seed in enumerate((0, 3)):
        single = model("unit__learned", data, seeds=(seed,), edge_chunk=1)
        actual, _ = single(data, epoch=8)
        actual.square().sum().backward()
        torch.testing.assert_close(full[index], actual[0], rtol=1e-12, atol=1e-13)
        for name, parameter in single.named_parameters():
            torch.testing.assert_close(dict(packed.named_parameters())[name].grad[index], parameter.grad[0], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_frozen_metrics_coverage_parameter_hash_training_restore_and_nullable_theta(condition):
    data = graph()
    candidate = model(condition, data)
    candidate.train()
    before = model_state_hash(candidate)
    result = frozen_evaluate(candidate, data, candidate.seeds, {"evaluation": {"intervention_scopes": ["layer_0", "layer_1", "both"]}})
    assert candidate.training and model_state_hash(candidate) == before
    assert result["provenance"]["before_sha256"] == result["provenance"]["after_sha256"]
    assert result["provenance"]["optimizer_updates"] == 0
    assert len(result["metric_rows"]) == 2 * 3
    enabled = candidate.variant != "off"
    assert len(result["intervention_rows"]) == (2 * 6 * 3 if enabled else 0)
    assert len(result["branch_rows"]) == 2 * 2 * (7 if enabled else 1)
    for row in result["branch_rows"]:
        assert row["theta_available"] == (candidate.variant == "learned")
        assert (row["theta"] is not None) == (candidate.variant == "learned")
        assert row["cross_energy_before"] >= 0 and row["cross_energy_after"] >= 0
    if candidate.variant == "fixed":
        original = {(row["seed"], row["split"]): row for row in result["metric_rows"]}
        for row in result["intervention_rows"]:
            if row["intervention"] == "gain1":
                assert row["no_op_expected"]
                assert row["ce"] == original[row["seed"], row["split"]]["ce"]


def test_zero_output_norm_ratios_are_null_and_model_ignores_labels_masks():
    data = graph(zero=True)
    candidate = model("unit__learned", data)
    result = frozen_evaluate(candidate, data, candidate.seeds, {"evaluation": {"intervention_scopes": ["layer_0", "layer_1", "both"]}})
    assert all(row["off_norm"] == 0 and row["matched_delta_relative"] is None for row in result["branch_rows"])

    class LabelsForbidden:
        x = data.x
        geometry_for = staticmethod(data.geometry_for)
        def __getattr__(self, name):
            if name in ("y", "train_mask", "val_mask", "test_mask"):
                raise AssertionError("labels entered model forward")
            raise AttributeError(name)
    logits, _ = candidate(LabelsForbidden())
    assert logits.shape == (2, 7, 3)


@pytest.mark.parametrize("bad", ["unit", "unit__cross_on", "local_degree__base", None])
def test_bad_condition_rejected(bad):
    with pytest.raises(ValueError):
        parse_condition(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_cuda_chunk_checkpoint_full_forward_and_gradient_matches_cpu(mode):
    cpu, cuda = graph(dtype=torch.float32), graph(dtype=torch.float32, device="cuda")
    a = model(f"{mode}__learned", cpu, edge_chunk=2)
    b = model(f"{mode}__learned", cuda, edge_chunk=3)
    a.train(); b.train()
    expected, _ = a(cpu, epoch=4)
    actual, _ = b(cuda, epoch=4)
    expected.square().sum().backward(); actual.square().sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    for name, parameter in b.named_parameters():
        torch.testing.assert_close(parameter.grad.cpu(), dict(a.named_parameters())[name].grad, rtol=5e-5, atol=5e-6)
