"""Explicit synthetic unit/smoke tests; never evidence of research performance."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from experiments.aggregation_comparison import engine
from research.conductance_gat.edge_selection.data import SelectionBatch
from research.conductance_gat.edge_selection.topology import build_topology
from research.conductance_gat.v5.operator import conductance_propagation_coefficients

from .evaluate import evaluate, state_digest
from .inspect_c import UpdateInspection, generator_parameters
from .model import COnlyClassifier, OnesConductance, make_model
from .train import common_digest, evaluate_selected, parser, step, train_one, validate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def arguments():
    result = parser().parse_args(
        [
            "--output-dir",
            "results/explicit-debug-c-only-unused",
            "--action",
            "calibrate",
            "--sample-seed-batch-size",
            "4096",
            "--sample-context-seed-batch-size",
            "2048",
            "--sample-context-workers",
            "2",
            "--edge-chunk-size",
            "128",
            "--physical-seed-candidates",
            "4096",
            "8192",
            "--context-worker-candidates",
            "2",
            "4",
            "--eval-context-seeds",
            "2048",
            "4096",
        ]
    )
    validate(result)
    return result


def fixture_batch(device="cpu"):
    generator = torch.Generator().manual_seed(123)
    n = 12
    nodes = torch.arange(n)
    edges = torch.cat(
        (torch.stack((nodes, (nodes + 1) % n)), torch.stack((nodes, (nodes + 4) % n))), 1
    )
    edges = edges.sort(0).values
    edges = torch.cat((edges, edges + n), 1)
    groups = torch.arange(2).repeat_interleave(n)
    graph = Data(
        x=torch.randn(2 * n, 9, generator=generator),
        y=torch.arange(2 * n) % 3,
        incidence_edge_index=edges,
        batch=groups,
        sampling_correction=torch.linspace(0.5, 2.0, edges.shape[1]),
    )
    topology = build_topology(2 * n, edges, node_graph=groups, forest_seed=0)
    return SelectionBatch(graph, topology, torch.zeros(edges.shape[1]), torch.arange(2 * n)).to(
        device
    )


def debug_model(batch, *, checkpoint=True, reference=False):
    args = arguments()
    if not reference:
        # Only explicit unit fixtures use this small shape, never a production profile.
        args.layers, args.hidden_channels, args.heads = 2, 16, 2
    args.activation_checkpoint = checkpoint
    model = make_model({"graphs": [{"x": batch.graph.x}], "classes": 3}, args, batch.graph.x.device)
    return model


def test_exact_neighbor_weights_and_sampling_correction():
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]])
    c = torch.tensor([1.0, 2.0, 7.0])[:, None]
    tail, _, _ = conductance_propagation_coefficients(c, edges, 4, normalization="row")
    torch.testing.assert_close(tail[:, 0], torch.tensor([0.1, 0.2, 0.7]))
    tail, _, _ = conductance_propagation_coefficients(
        torch.ones_like(c), edges, 4, normalization="row"
    )
    torch.testing.assert_close(tail[:, 0], torch.full((3,), 1 / 3))
    tail, _, _ = conductance_propagation_coefficients(
        torch.ones_like(c),
        edges,
        4,
        sampling_correction=torch.tensor([1.0, 2.0, 7.0]),
        normalization="row",
    )
    torch.testing.assert_close(tail[:, 0], torch.tensor([0.1, 0.2, 0.7]))


def test_no_completion_modules_and_exact_fixed_control():
    batch = fixture_batch()
    learned = debug_model(batch)
    fixed = learned.fixed_copy()
    assert common_digest(learned) == common_digest(fixed)
    assert not generator_parameters(fixed)
    assert all(isinstance(op.estimator, OnesConductance) for op in fixed.layers)
    assert not any(
        "readout" in name or "hop_coefficients" in name or "lift_projection" in name
        for name, _ in learned.named_parameters()
    )
    learned.eval()
    fixed.eval()
    identity = state_digest(learned)
    with torch.no_grad(), learned.c_ones():
        torch.testing.assert_close(learned(batch.graph), fixed(batch.graph), rtol=0, atol=0)
    assert identity == state_digest(learned)
    report = evaluate(fixed, [batch], intervention=True)
    assert report["c_ones_intervention"]["logit_change_rms"] == 0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_observer_preserves_forward_gradients_updates_and_rng(checkpoint):
    batch = fixture_batch()
    plain = debug_model(batch, checkpoint=checkpoint)
    observed = copy.deepcopy(plain)
    first, second = engine.make_optimizer(plain), engine.make_optimizer(observed)
    torch.manual_seed(400)
    loss1, _, _ = step(plain, batch, first)
    rng1 = torch.get_rng_state()
    torch.manual_seed(400)
    loss2, _, report = step(observed, batch, second, example_nodes=3)
    assert torch.equal(rng1, torch.get_rng_state())
    torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
    for (name1, p1), (name2, p2) in zip(
        plain.named_parameters(), observed.named_parameters(), strict=True
    ):
        assert name1 == name2
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)
        torch.testing.assert_close(p1.grad, p2.grad, rtol=0, atol=0)
    assert any(row["update_norm"] > 0 for row in report["generator_updates"].values())
    assert any(max(row["alpha"]["change"]["max_abs"]) > 0 for row in report["layers"])
    assert all(row["live_c_gradient_norm_and_max_by_head"] is not None for row in report["layers"])


def test_same_input_replay_before_optimizer_is_identical():
    batch = fixture_batch()
    model = debug_model(batch)
    model.train()
    observer = UpdateInspection(model, 3)
    model(batch.graph, observer=observer)
    report = observer.finish(model, batch.graph)
    for row in report["layers"]:
        assert max(row["cost"]["change"]["max_abs"]) == 0
        assert max(row["c"]["change"]["max_abs"]) == 0
        assert max(row["alpha"]["change"]["max_abs"]) == 0


def test_actual_generator_parameter_direction_matches_finite_difference():
    batch = fixture_batch()
    model = debug_model(batch, checkpoint=False)
    model.eval()
    loss = F.cross_entropy(model(batch.graph), batch.graph.y)
    loss.backward()
    params = list(generator_parameters(model).values())
    norm = torch.stack([p.grad.square().sum() for p in params]).sum().sqrt()
    assert norm > 1e-8
    originals = [p.detach().clone() for p in params]
    directions = [p.grad.detach().clone() / norm for p in params]
    estimates = []
    try:
        for epsilon in (0.03, 0.01):
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    for p, value, direction in zip(params, originals, directions, strict=True):
                        p.copy_(value + sign * epsilon * direction)
                    values.append(F.cross_entropy(model(batch.graph), batch.graph.y))
            estimates.append((values[0] - values[1]) / (2 * epsilon))
    finally:
        with torch.no_grad():
            for p, value in zip(params, originals, strict=True):
                p.copy_(value)
    for numerical in estimates:
        torch.testing.assert_close(numerical, norm, rtol=0.08, atol=2e-6)


def test_actual_edge_c_direction_recomputes_normalization():
    batch = fixture_batch()
    model = debug_model(batch, checkpoint=False)
    model.eval()
    captured = []

    def record(module, args, output):
        output.retain_grad()
        captured.append(output)

    handle = model.layers[0].estimator.register_forward_hook(record)
    F.cross_entropy(model(batch.graph), batch.graph.y).backward()
    handle.remove()
    gradient = captured[0].grad
    position = int(gradient.abs().argmax())
    analytical = gradient.flatten()[position].detach()
    assert analytical.abs() > 1e-8
    values = []
    for offset in (0.01, -0.01):

        def perturb(module, args, output, offset=offset):
            change = torch.zeros_like(output)
            change.flatten()[position] = offset
            return output + change

        handle = model.layers[0].estimator.register_forward_hook(perturb)
        try:
            with torch.no_grad():
                values.append(F.cross_entropy(model(batch.graph), batch.graph.y))
        finally:
            handle.remove()
    torch.testing.assert_close((values[0] - values[1]) / 0.02, analytical, rtol=0.08, atol=3e-6)


def test_fixed_generator_has_no_optimizer_entries():
    batch = fixture_batch()
    model = debug_model(batch).fixed_copy()
    optimizer = engine.make_optimizer(model)
    _, _, report = step(model, batch, optimizer, example_nodes=2)
    assert report["generator_updates"] == {}
    assert all(max(row["c"]["change"]["max_abs"]) == 0 for row in report["layers"])
    assert all(row["live_c_gradient_norm_and_max_by_head"] is None for row in report["layers"])


def test_reference_shape_smoke_cpu():
    batch = fixture_batch()
    model = debug_model(batch, reference=True)
    assert (model.depth, model.width, model.heads) == (8, 256, 8)
    _, count, report = step(model, batch, engine.make_optimizer(model), example_nodes=2)
    assert count == 24 and len(report["layers"]) == 8


def test_production_rejects_scope_changes():
    args = arguments()
    args.layers = 2
    with pytest.raises(ValueError, match="recipe"):
        validate(args)
    args = arguments()
    args.epochs = 20
    with pytest.raises(ValueError, match="200"):
        validate(args)
    with pytest.raises(SystemExit):
        parser().parse_args(["--v2-arms", "both"])


def test_forward_matches_existing_base_on_fixed_support():
    batch = fixture_batch()
    args = arguments()
    args.layers, args.hidden_channels, args.heads = 2, 16, 2  # explicit unit fixture only
    engine.base._seed(args.model_seed)
    backbone = engine.make_model(
        {"graphs": [{"x": batch.graph.x}], "classes": 3}, args, torch.device("cpu")
    )
    model = COnlyClassifier(copy.deepcopy(backbone))
    backbone.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch.graph), backbone(batch.graph), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA pipeline smoke test")
def test_debug_cuda_paired_training_checkpoint_and_new_contexts(tmp_path):
    # Full model, synthetic data and two epochs only: exercises orchestration,
    # not model quality. Production validation rejects this shortened budget.
    args = arguments()
    args.epochs = 2
    args.sample_seed_batch_size = 64
    args.sample_context_seed_batch_size = 32
    args.eval_context_seeds = [16, 32]
    args.precision = "fp32"
    args.pin_memory = False
    args.sample_prefetch = False
    generator = torch.Generator().manual_seed(55)
    nodes = torch.arange(128)
    edges = (
        torch.cat(
            (torch.stack((nodes, (nodes + 1) % 128)), torch.stack((nodes, (nodes + 7) % 128))), 1
        )
        .sort(0)
        .values
    )
    masks = {"train": nodes < 96, "validation": (nodes >= 96) & (nodes < 112), "test": nodes >= 112}
    payload = {
        "explicit_synthetic_debug": True,
        "classes": 3,
        "splits": masks,
        "graphs": [
            {
                "x": torch.randn(128, 9, generator=generator),
                "y": nodes % 3,
                "incidence_edge_index": edges,
            }
        ],
    }
    device = torch.device("cuda")
    learned = train_one(payload, args, "learned", tmp_path, device)
    fixed = train_one(payload, args, "fixed", tmp_path, device, paired=learned)
    assert learned["debug"] and fixed["debug"]
    assert learned["initial_common_sha256"] == fixed["initial_common_sha256"]
    results = evaluate_selected(
        payload, args, {"learned": learned, "fixed": fixed}, tmp_path, device
    )
    for condition in ("learned", "fixed"):
        assert results[condition]["full_validation"]["total"] == 16
        assert len(results[condition]["new_sampling_contexts"]) == 2
        for epoch in (0, 1):
            assert (
                learned["history"][epoch]["sampling_evidence"]
                == fixed["history"][epoch]["sampling_evidence"]
            )
    saved = torch.load(learned["best"]["path"], weights_only=True, map_location="cpu")
    assert saved["debug"] is True
