"""Explicit synthetic unit/smoke tests; never evidence of research performance."""

import copy
import json

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from experiments.aggregation_comparison import engine
from experiments.c_learning_only.model import OnesConductance
from research.conductance_gat.edge_selection.data import SelectionBatch
from research.conductance_gat.edge_selection.topology import build_topology
from research.conductance_gat.v5.operator import (
    conductance_propagation_coefficients,
    shared_head_diffusion,
)

from .conductance import BracketConductance
from .evaluate import evaluate, state_digest
from .inspect_c import UpdateInspection, generator_parameters
from .model import make_model
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
        assert max(row["score"]["change"]["max_abs"]) == 0
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


def test_raw_formula_orientation_heads_and_chunk_gradients():
    torch.manual_seed(95)
    c = BracketConductance(8, 2, edge_chunk_size=2).double()
    h = torch.randn(7, 8, dtype=torch.double, requires_grad=True)
    edges = torch.tensor([[0, 0, 1, 2, 4], [1, 3, 2, 4, 5]])
    q, k = c.query(h).reshape(7, 2, 4), c.key(h).reshape(7, 2, 4)
    expected = ((q[edges[0]] * k[edges[1]] + q[edges[1]] * k[edges[0]]).sum(-1) / 8).exp()
    actual = c(h, edges)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, c(h, edges.flip(0)), rtol=0, atol=0)
    assert actual.shape == (5, 2) and not torch.equal(actual[:, 0], actual[:, 1])
    other = copy.deepcopy(c)
    other.edge_chunk_size = 99
    other.checkpoint_edges = False
    actual.square().sum().backward()
    h2 = h.detach().clone().requires_grad_()
    other(h2, edges).square().sum().backward()
    torch.testing.assert_close(h.grad, h2.grad, rtol=1e-12, atol=1e-12)
    for p, p2 in zip(c.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p.grad, p2.grad, rtol=1e-12, atol=1e-12)


def test_raw_c_is_local_and_independent_of_batched_graphs():
    torch.manual_seed(15)
    c = BracketConductance(8, 2, edge_chunk_size=3).double()
    h = torch.randn(8, 8, dtype=torch.double)
    edges = torch.tensor([[0, 1, 2, 5], [1, 2, 3, 6]])
    before = c(h, edges)
    extra_h = torch.randn(6, 8, dtype=torch.double) * 4
    extra_edges = torch.tensor([[8, 9, 10], [9, 10, 11]])
    together = c(torch.cat((h, extra_h)), torch.cat((edges, extra_edges), 1))
    torch.testing.assert_close(before, together[:4], rtol=1e-12, atol=1e-12)
    changed = h.clone()
    changed[7] *= 100
    torch.testing.assert_close(before, c(changed, edges), rtol=0, atol=0)
    assert c(h, torch.empty(2, 0, dtype=torch.long)).shape == (0, 2)


def test_sparse_diffusion_equals_explicit_incidence_and_isolates():
    torch.manual_seed(12)
    edges = torch.tensor([[0, 0, 1, 3], [1, 2, 2, 4]])
    n, heads, width = 6, 2, 3
    value = torch.randn(n, heads, width, dtype=torch.double)
    c = torch.rand(4, heads, dtype=torch.double) + 0.2
    a = torch.linspace(0.3, 1.8, 4, dtype=torch.double)
    beta = torch.tensor([[0.3, 0.8]], dtype=torch.double)
    b = torch.zeros(4, n, dtype=torch.double)
    b[torch.arange(4), edges[0]], b[torch.arange(4), edges[1]] = -1, 1
    weight = c * a[:, None]
    laplacian = torch.einsum("ei,eh,ej->hij", b, weight, b)
    degree = laplacian.diagonal(dim1=-2, dim2=-1).T
    dv = torch.einsum("hij,jhk->ihk", laplacian, value)
    expected = value - beta[0, :, None] * dv / torch.where(degree > 0, degree, 1)[..., None]
    actual = shared_head_diffusion(
        value,
        c,
        edges,
        torch.zeros(n, dtype=torch.long),
        beta,
        sampling_correction=a,
        edge_chunk_size=2,
        propagation_normalization="row",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual[-1], value[-1], rtol=0, atol=0)
    reversed_output = shared_head_diffusion(
        value,
        c,
        edges.flip(0),
        torch.zeros(n, dtype=torch.long),
        beta,
        sampling_correction=a,
        edge_chunk_size=2,
        propagation_normalization="row",
    )
    torch.testing.assert_close(actual, reversed_output, rtol=1e-13, atol=1e-13)


def test_ce_gradcheck_includes_q_k_and_degree_derivatives():
    torch.manual_seed(92)
    c = BracketConductance(4, 2, edge_chunk_size=2, checkpoint_edges=False).double()
    h = torch.randn(5, 4, dtype=torch.double) * 0.3
    edges = torch.tensor([[0, 0, 1, 1, 2], [1, 2, 2, 3, 4]])
    values = torch.randn(5, 2, 2, dtype=torch.double)
    labels = torch.arange(5) % 4
    weights = tuple(p.detach().clone().requires_grad_() for p in c.parameters())
    names = [name for name, _ in c.named_parameters()]

    def loss(*params):
        conductance = torch.func.functional_call(
            c, dict(zip(names, params, strict=True)), (h, edges)
        )
        message = shared_head_diffusion(
            values,
            conductance,
            edges,
            torch.zeros(5, dtype=torch.long),
            torch.tensor([[0.4, 0.6]], dtype=torch.double),
            edge_chunk_size=2,
            propagation_normalization="row",
        )
        return F.cross_entropy(message.flatten(1), labels)

    assert torch.autograd.gradcheck(loss, weights, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_no_old_estimator_is_constructed(monkeypatch):
    import research.conductance_gat.v5.model as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("old C factory must never execute")

    monkeypatch.setattr(engine, "make_model", forbidden)
    monkeypatch.setattr(legacy, "GraphOptimizedConductance", forbidden)
    model = debug_model(fixture_batch())
    assert all(isinstance(op.estimator, BracketConductance) for op in model.layers)
    assert not any("context_metric" in name for name, _ in model.named_parameters())


def test_fixed_forward_preserves_existing_backbone_equation():
    from experiments.c_learning_only.model import make_model as old_make
    from experiments.c_learning_only.test_debug import arguments as old_arguments

    batch = fixture_batch()
    args = old_arguments()
    args.layers, args.hidden_channels, args.heads = 2, 16, 2
    old = old_make(
        {"graphs": [{"x": batch.graph.x}], "classes": 3}, args, torch.device("cpu"), "fixed"
    )
    new = debug_model(batch).fixed_copy()
    new.load_state_dict(old.state_dict(), strict=True)
    new.eval()
    old.eval()
    with torch.no_grad():
        torch.testing.assert_close(new(batch.graph), old(batch.graph), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA timing instrumentation")
def test_debug_cuda_paired_timing_and_bf16_observer():
    from .train import paired_timing

    args = arguments()
    args.precision = "bf16"
    batch = fixture_batch("cuda")
    model = debug_model(batch, reference=True)
    optimizer = engine.make_optimizer(model)
    row, inspection = paired_timing(model, batch, optimizer, args, torch.device("cuda"))
    assert row["same_batch_state_optimizer_rng"]
    assert row["plain"]["timing"]["cuda_event_seconds"]["forward_and_ce"] > 0
    assert row["plain"]["timing"]["cuda_event_seconds"]["backward"] > 0
    assert row["plain"]["timing"]["cuda_event_seconds"]["optimizer"] > 0
    assert row["observed"]["timing"]["cuda_event_seconds"]["inspection_same_input_replay"] > 0
    assert all("score" in layer and "cost" not in layer for layer in inspection["layers"])


def test_nonfinite_exp_stops_without_clamping():
    c = BracketConductance(4, 2, edge_chunk_size=4)
    h = torch.ones(2, 4)
    edges = torch.tensor([[0], [1]])
    with torch.no_grad():
        c.query.weight.fill_(1)
        c.key.weight.fill_(1)
    assert c(h, edges).min() > torch.exp(torch.tensor(10.0))
    with torch.no_grad():
        c.query.weight.fill_(100)
        c.key.weight.fill_(100)
    with pytest.raises(RuntimeError, match="nonfinite/zero"):
        c(h, edges)


def test_finite_edge_weights_with_overflow_degree_stop_in_operator(monkeypatch):
    from .operator import BracketOperator

    op = BracketOperator(4, 2, edge_chunk_size=2)
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]])
    finite = torch.full((3, 2), 2e38)
    assert torch.isfinite(finite).all()
    monkeypatch.setattr(op.estimator, "forward", lambda *args: finite)
    with pytest.raises(RuntimeError, match="weighted degree"):
        op.forward_with_state(
            torch.ones(4, 4),
            edges,
            torch.zeros(4, dtype=torch.long),
            1,
            full_degree=None,
            graph_structure=None,
            sampling_correction=None,
        )


def test_checked_row_diffusion_preserves_values_and_all_derivatives():
    from .diffusion import row_diffusion

    torch.manual_seed(30)
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    value = torch.randn(5, 2, 3, dtype=torch.double, requires_grad=True)
    c = (torch.rand(4, 2, dtype=torch.double) + 0.1).requires_grad_()
    beta = torch.rand(1, 2, dtype=torch.double, requires_grad=True)
    groups = torch.zeros(5, dtype=torch.long)
    actual, _ = row_diffusion(value, c, edges, groups, beta, edge_chunk_size=2)
    reference = shared_head_diffusion(
        value, c, edges, groups, beta, edge_chunk_size=2, propagation_normalization="row"
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    a = torch.autograd.grad(actual.square().sum(), (value, c, beta))
    b = torch.autograd.grad(reference.square().sum(), (value, c, beta))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def debug_payload():
    nodes = torch.arange(128)
    edges = (
        torch.cat(
            (torch.stack((nodes, (nodes + 1) % 128)), torch.stack((nodes, (nodes + 7) % 128))), 1
        )
        .sort(0)
        .values
    )
    return {
        "explicit_synthetic_debug": True,
        "classes": 3,
        "graphs": [
            {
                "x": torch.randn(128, 9, generator=torch.Generator().manual_seed(55)),
                "y": nodes % 3,
                "incidence_edge_index": edges,
            }
        ],
        "splits": {
            "train": nodes < 96,
            "validation": (nodes >= 96) & (nodes < 112),
            "test": nodes >= 112,
        },
    }


def test_input_pipeline_same_samples_without_topology(monkeypatch):
    from experiments.aggregation_comparison.study_inputs import StudyInputs as LegacyInputs
    from research.conductance_gat.edge_selection import data as legacy_data

    from .inputs import BracketInputs

    args = arguments()
    args.sample_seed_batch_size, args.sample_context_seed_batch_size = 64, 32
    args.pin_memory, args.sample_prefetch = False, False
    payload = debug_payload()
    old = LegacyInputs(payload, args)
    old_batches = list(old.training_batches(0, torch.device("cpu")))

    def forbidden(*args, **kwargs):
        raise AssertionError("unused topology must not be constructed")

    monkeypatch.setattr(legacy_data, "build_topology", forbidden)
    new = BracketInputs(payload, args)
    new_batches = list(new.training_batches(0, torch.device("cpu")))
    assert new.last_pass_evidence == old.last_pass_evidence
    for before, after in zip(old_batches, new_batches, strict=True):
        assert not hasattr(after.graph, "edge_selection_topology")
        assert not hasattr(after, "origin_targets")
        for name in (
            "x",
            "y",
            "incidence_edge_index",
            "full_degree",
            "graph_structure",
            "sampling_correction",
            "batch",
            "global_node_id",
        ):
            assert torch.equal(getattr(before.graph, name), getattr(after.graph, name))
        assert torch.equal(before.selected_indices, after.selected_indices)
    model = debug_model(old_batches[0]).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(old_batches[0].graph), model(new_batches[0].graph), rtol=0, atol=0
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real pinned CPU allocation requires CUDA"
)
@pytest.mark.parametrize("prefetch", [False, True])
def test_actual_pinning_before_transfer_with_and_without_prefetch(prefetch):
    from .inputs import BracketInputs

    args = arguments()
    args.sample_seed_batch_size, args.sample_context_seed_batch_size = 64, 32
    args.pin_memory, args.sample_prefetch = True, prefetch
    inputs = BracketInputs(debug_payload(), args)
    for batch in inputs.training_batches(0, torch.device("cuda")):
        assert all(batch.transfer_evidence["is_pinned_before_copy"].values())
        assert batch.graph.x.is_cuda
        assert batch.selected_indices.is_cuda
    val = next(inputs.validation_batches(torch.device("cuda")))
    assert all(val.transfer_evidence["is_pinned_before_copy"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA no-update noise diagnostic")
def test_fixed_cuda_never_reports_learning_from_reduction_noise():
    batch = fixture_batch("cuda")
    model = debug_model(batch).fixed_copy()
    _, _, report = step(model, batch, engine.make_optimizer(model), example_nodes=3)
    for row, op in zip(report["layers"], model.layers, strict=True):
        assert row["coefficient_edge_chunk_size"] == op.edge_chunk_size
        assert not any(row["raw_c_changed_by_head"])
        assert not any(row["alpha_change_resolved_above_observed_replay_by_head"])
        assert max(row["canonical_cpu_float64_alpha_change"]["max_abs"]) == 0
        assert row["features_before_update"]["query"] is None


def test_features_and_adamw_decay_are_observed_separately():
    batch = fixture_batch()
    model = debug_model(batch)
    _, _, report = step(model, batch, engine.make_optimizer(model), example_nodes=3)
    for row in report["layers"]:
        stats = row["features_before_update"]
        assert set(stats) == {"input_h", "query", "key"}
        assert all(x > 0 for x in stats["input_h"]["rms_by_head"])
        assert "no_update_replay_before" in row
    for values in report["generator_updates"].values():
        assert values["ideal_adamw_decay_update_norm"] >= 0
        assert values["actual_update_minus_ideal_decay_norm"] >= 0


def test_reference_effect_below_gpu_noise_is_not_resolved():
    from .inspect_c import resolved_change

    changed = torch.tensor([True, True, False, True])
    measured = torch.tensor([2e-7, 2e-7, 2e-7, 0.0])
    noise = torch.full((4,), 6e-8)
    reference = torch.tensor([1e-8, 1e-7, 1e-7, 1e-7])
    assert resolved_change(changed, measured, noise, reference).tolist() == [
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_signal_stages_match_actual_values_and_next_layer(checkpoint):
    batch = fixture_batch()
    model = debug_model(batch, checkpoint=checkpoint)
    observer = UpdateInspection(model, 2)
    recorded = {}
    handles = []
    for index, op in enumerate(model.layers):

        def capture(module, args, output, index=index):
            recorded[index] = (args[0].detach().clone(), output.detach().clone())

        handles.append(op.output_projection.register_forward_hook(capture))
    try:
        logits = model(batch.graph, observer=observer)
    finally:
        for handle in handles:
            handle.remove()
    for index, op in enumerate(model.layers):
        row = observer.layers[index]
        stages = row["signal_stages"]
        assert list(stages) == [
            "input_h",
            "value_projection",
            "neighbor_mixing",
            "output_projection",
            "relu",
            "dropout_next_h",
        ]
        propagated, projected = recorded[index]
        expected = {
            "input_h": row["input"],
            "value_projection": torch.einsum("nd,hdk->nhk", row["input"], op.value_weight),
            "neighbor_mixing": propagated,
            "output_projection": projected,
            "relu": projected.relu(),
        }
        for label, tensor in expected.items():
            assert stages[label]["rms"] == pytest.approx(
                float(tensor.detach().double().square().mean().sqrt()), rel=1e-14
            )
        if index + 1 < model.depth:
            assert (
                stages["dropout_next_h"] == observer.layers[index + 1]["signal_stages"]["input_h"]
            )
        assert op.observed_signals is None and not op.capture_signals
    before_backward = copy.deepcopy([r["signal_stages"] for r in observer.layers])
    F.cross_entropy(logits, batch.graph.y).backward()
    assert before_backward == [r["signal_stages"] for r in observer.layers]
    assert all(op.observed_signals is None for op in model.layers)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "wrong_ids"])
def test_evaluation_rejects_invalid_global_seed_coverage(failure):
    from .inputs import BracketBatch

    batch = fixture_batch()
    model = debug_model(batch).eval()
    ids = batch.selected_indices
    expected = ids.clone()
    if failure == "duplicate":
        ids = torch.cat((ids[:-1], ids[:1]))
    elif failure == "missing":
        ids = ids[:-1]
    else:
        batch.graph.global_node_id = ids + 1000
    with pytest.raises(RuntimeError, match="exactly once"):
        evaluate(model, [BracketBatch(batch.graph, ids)], expected_seed_ids=expected)


def test_calibration_saves_all_context_metrics_hashes_and_coverage(tmp_path):
    from .inputs import BracketInputs
    from .train import calibration_evaluations, write_json

    payload, args = debug_payload(), arguments()
    args.sample_seed_batch_size, args.sample_context_seed_batch_size = 16, 8
    args.eval_context_seeds = [8, 16]
    args.pin_memory, args.sample_prefetch = False, False
    device = torch.device("cpu")
    inputs = BracketInputs(payload, args)
    model = make_model(payload, args, device)
    before = state_digest(model)
    report = calibration_evaluations(model, inputs, payload, args, device)
    path = tmp_path / "debug-calibration-evaluation.json"
    write_json(path, report)
    saved = json.loads(path.read_text())
    assert saved == report
    assert set(saved["new_sampling_contexts"]) == {"8", "16"}
    for result in [saved["full_validation"], *saved["new_sampling_contexts"].values()]:
        assert result["total"] == 16
        assert 0 <= result["correct"] <= 16 and result["cross_entropy"] > 0
        assert result["parameter_sha256"] == before
        assert result["c_ones_intervention"]["cross_entropy"] > 0
        coverage = result["seed_coverage"]
        assert coverage["every_expected_seed_exactly_once"]
        assert coverage["unique_count"] == 16
        assert coverage["expected_global_ids_sha256"] == coverage["observed_global_ids_sha256"]
    for result in saved["new_sampling_contexts"].values():
        assert result["sampling_evidence"]["supervised_nodes"] == 16
        assert result["sampling_evidence"]["every_train_seed_exactly_once"]
    assert state_digest(model) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA pipeline smoke test")
def test_debug_cuda_paired_training_checkpoint_and_new_contexts(tmp_path, monkeypatch):
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
    from . import train

    expected_ids = masks["validation"].nonzero().flatten()
    evaluation_calls = []

    def checked_evaluate(model, batches, *args, **kwargs):
        # Exercise the real evaluator and reject a missing/wrong expectation at any caller.
        assert torch.equal(kwargs["expected_seed_ids"].cpu(), expected_ids)
        result = evaluate(model, batches, *args, **kwargs)
        evaluation_calls.append(result)
        return result

    monkeypatch.setattr(train, "evaluate", checked_evaluate)
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
        initial = json.loads((tmp_path / f"{condition}-initial.json").read_text())
        trained = json.loads((tmp_path / f"{condition}-trained.json").read_text())
        persisted = [initial["validation"], trained["best"]["validation"]]
        for epoch in (1, 2):
            row = json.loads((tmp_path / f"{condition}-epoch-{epoch:04d}.json").read_text())
            persisted.append(row["validation"])
        final = json.loads((tmp_path / "evaluation.json").read_text())["results"][condition]
        persisted.extend([final["full_validation"], *final["new_sampling_contexts"].values()])
        for result in persisted:
            coverage = result["seed_coverage"]
            assert coverage["every_expected_seed_exactly_once"]
            assert coverage["expected_count"] == coverage["observed_count"] == 16
            assert coverage["unique_count"] == 16
            assert coverage["expected_global_ids_sha256"] == coverage["observed_global_ids_sha256"]
            assert result["cross_entropy"] > 0
        for result in final["new_sampling_contexts"].values():
            assert result["parameter_sha256"] == final["full_validation"]["parameter_sha256"]
    # Each condition: initial + two epochs + final full + two final contexts.
    assert len(evaluation_calls) == 12
    saved = torch.load(learned["best"]["path"], weights_only=True, map_location="cpu")
    assert saved["debug"] is True
