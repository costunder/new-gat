"""Explicit synthetic CUDA diagnostics; no benchmark result or reduced production recipe."""

import pytest
import torch

from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.mechanisms import MechanismCollector, mechanism_audit
from experiments.aggregation_comparison.model import AggregationClassifier
from tests.test_aggregation_comparison_cuda import (  # noqa: F401
    cuda_required,
    reference_arguments,
    synthetic_disjoint_batch,
)


def test_observation_weighted_moments_and_histograms():
    collector = MechanismCollector()
    c = torch.tensor([[0.5, 1.0], [2.0, 4.0], [0.25, 8.0]], device="cuda")
    gram = torch.arange(18, device="cuda", dtype=torch.float32).reshape(3, 2, 3) - 4
    diagonal = torch.tensor([True, False, True], device="cuda")
    message = torch.arange(6, device="cuda", dtype=torch.float32).reshape(3, 2)
    with torch.no_grad():
        # Unequal sizes catch an incorrect average of batch means.
        for section in (slice(0, 1), slice(1, 3)):
            collector.record(
                1, c[section], gram[section], diagonal, message[section], message[section] * 0.5
            )
        row = collector.finish()["rows"][0]
    assert row["validation_batches"] == 2
    assert row["conductance"]["observations_per_head"] == 3
    assert row["conductance"]["mean"] == pytest.approx(c.double().mean(0).tolist())
    assert row["diagonal"]["mean"] == pytest.approx(
        gram[..., diagonal].double().mean((0, 2)).tolist()
    )
    assert row["cross"]["mean_abs"] == pytest.approx(
        gram[..., ~diagonal].double().abs().mean((0, 2)).tolist()
    )
    assert row["energy_to_message_l2"] == pytest.approx(0.5)
    assert [sum(v) for v in row["c_histogram"]] == [3, 3]
    assert row["c_histogram"][0] == [0, 1, 1, 0, 0, 0, 1, 0, 0]


@pytest.mark.parametrize(
    "arm", ["incidence_fixed_energy", "incidence_shared_energy", "incidence_energy"]
)
def test_c_controls_are_active_and_interventions_restore(arm):
    engine.base._seed(84)
    inputs, graph, payload = synthetic_disjoint_batch("cuda")
    args = reference_arguments(arm, "fp32")
    model = engine.make_model(payload, args, torch.device("cuda:0"))
    optimizer = engine.make_optimizer(model, args.learning_rate)
    engine.run_training_epoch(
        model, optimizer, inputs, args, torch.device("cuda:0"), 1, validate=True
    )
    groups = engine.validate_gradients(model)
    assert "layer_7.energy_readout" in groups and "layer_7.beta" in groups
    c_parameters = [p for name, p in model.named_parameters() if ".estimator." in name]
    if "fixed" in arm:
        assert not c_parameters and "layer_0.conductance" not in groups
    else:
        assert c_parameters and all(p.grad is not None for p in c_parameters)
        assert sum(groups[f"layer_{layer}.conductance"] for layer in range(8)) > 0
    model.eval()
    before = engine.base.state_sha256(model)
    with torch.no_grad():
        expected = model(graph)
        first_c = model.layers[0].last_effective_c.clone()
        if "fixed" in arm:
            assert torch.equal(first_c, torch.ones_like(first_c))
        elif "shared" in arm:
            assert first_c.ndim == 1
        else:
            assert first_c.shape == (512, 8)
        if "fixed" not in arm:
            with model.intervention("c_shuffle"):
                model(graph)
                actual = model.layers[0].last_effective_c
                # Known disjoint support: exactly 128 edges per graph, preserve head vectors.
                expected_c = first_c.reshape(4, 128, -1).flip(1).reshape_as(first_c)
                torch.testing.assert_close(actual, expected_c, rtol=0, atol=0)
            with model.intervention("c_mean"):
                model(graph)
                by_graph = model.layers[0].last_effective_c.reshape(4, 128, -1)
                torch.testing.assert_close(by_graph, by_graph[:, :1].expand_as(by_graph))
        with model.intervention("c_ones"):
            model(graph)
            assert all(
                torch.equal(op.last_effective_c, torch.ones_like(op.last_effective_c))
                for op in model.layers
            )
        with model.intervention("energy_off"):
            suppressed = model(graph)
        assert not torch.equal(suppressed, expected)
        with pytest.raises(RuntimeError, match="deliberate"):
            with model.intervention("energy_off"), model.intervention("c_ones"):
                raise RuntimeError("deliberate")
        torch.testing.assert_close(model(graph), expected, rtol=0, atol=0)
        selected = engine.evaluate(model, inputs, args, torch.device("cuda:0"))
        report = mechanism_audit(
            model, inputs, args, torch.device("cuda:0"), selected, engine.evaluate
        )
    assert before == engine.base.state_sha256(model)
    assert model.diagnostic_collector is None and model.energy_intervention is None
    assert all(op.estimator.override is None for op in model.layers)
    rows = report["diagnostics"]["rows"]
    assert len(rows) == 8 and rows[0]["cross"]["observations_per_head"] == 0
    assert rows[7]["diagonal"]["observations_per_head"] == 256 * 8
    assert rows[7]["cross"]["observations_per_head"] == 256 * 28
    assert all(row["conductance"]["observations_per_head"] == 512 for row in rows)
    assert rows[7]["energy_output_l2"] > 0


def test_no_energy_diagnostics_do_not_change_output_and_cover_every_batch():
    inputs, graph, payload = synthetic_disjoint_batch("cuda")
    args = reference_arguments("incidence", "fp32")
    model = engine.make_model(payload, args, torch.device("cuda:0")).eval()
    batch = next(inputs.validation_batches("cuda"))
    inputs.validation_batches = lambda device: iter([batch, batch])
    with torch.no_grad():
        baseline = model(graph)
        collector = MechanismCollector()
        model.diagnostic_collector = collector
        engine.evaluate(model, inputs, args, torch.device("cuda:0"))
        model.diagnostic_collector = None
        torch.testing.assert_close(model(graph), baseline, rtol=0, atol=0)
        report = collector.finish()
    assert all(row["validation_batches"] == 2 for row in report["rows"])
    assert all(row["conductance"]["observations_per_head"] == 1024 for row in report["rows"])
    assert all(row["energy_output_l2"] == 0 for row in report["rows"])


def test_energy_mask_controls_match_retrained_architecture_at_fixed_weights():
    _, graph, _ = synthetic_disjoint_batch("cuda")

    def create(arm):
        torch.manual_seed(17)
        return (
            AggregationClassifier(
                50,
                7,
                arm=arm,
                selection_config={"condition": "full"},
                hidden_channels=256,
                layers=8,
                heads=8,
                dropout=0,
                edge_chunk_size=128,
            )
            .cuda()
            .eval()
        )

    full, diagonal, base = (
        create(arm) for arm in ("incidence_energy", "incidence_diagonal", "incidence")
    )
    with torch.no_grad():
        for depth, weight in enumerate(full.energy_readouts, 1):
            weight.normal_(std=0.01)
            pairs = torch.triu_indices(depth, depth, device="cuda")
            diagonal.energy_readouts[depth - 1].copy_(weight[:, pairs[0] == pairs[1]])
        with full.intervention("cross_off"):
            torch.testing.assert_close(full(graph), diagonal(graph), rtol=1e-5, atol=1e-6)
        with full.intervention("energy_off"):
            torch.testing.assert_close(full(graph), base(graph), rtol=0, atol=0)
        first_output = full(graph)
        with full.intervention("diagonal_off"):
            assert not torch.equal(full(graph), first_output)


def test_calibration_measures_full_mechanism_path(monkeypatch):
    inputs, graph, payload = synthetic_disjoint_batch("cuda")
    inputs.plan_preparation_seconds = 0.0
    args = reference_arguments("incidence_energy_pre_lift", "fp32")
    monkeypatch.setattr(engine, "PreparedInputs", lambda *a: inputs)
    report = engine.run_calibration_candidate(
        {"payload": payload}, args, torch.device("cuda:0"), physical_batch_size=1, workers=0
    )
    assert report["validation_completed"] and report["mechanism_audit_completed"]
    assert report["mechanism_audit_seconds"] > 0
    assert report["optimizer_state_bytes"] > 0 and report["parameter_update_verified"]
    assert report["largest_measured_nodes"] == graph.x.shape[0]
    assert report["largest_measured_graph_batch"] == 4


def test_per_head_operator_initialization_preserves_previous_backbone():
    from experiments.incidence_ablation.model import IncidenceClassifier

    architecture = dict(
        hidden_channels=256,
        layers=8,
        heads=8,
        dropout=0,
        edge_chunk_size=128,
        conductance_mode="dynamic",
        conductance_heads="per_head",
        conductance_backend="optimization",
        conductance_generator="optimized",
        propagation_normalization="row",
        propagation_filter="linear",
        solver_cost_scaling="width_scaled",
    )
    torch.manual_seed(31)
    current = AggregationClassifier(
        50, 7, arm="incidence_pre_lift", selection_config={"condition": "full"}, **architecture
    ).cuda()
    torch.manual_seed(31)
    # The comparison draws common endpoints before constructing historical operators.
    torch.nn.Linear(50, 256)
    torch.nn.Linear(256, 7)
    previous = IncidenceClassifier(
        50, 7, arm="pre_lift", selection_config={"condition": "full"}, **architecture
    ).cuda()
    for left, right in zip(current.layers, previous.operators, strict=True):
        assert set(left.state_dict()) == set(right.state_dict())
        for name, tensor in left.state_dict().items():
            torch.testing.assert_close(tensor, right.state_dict()[name], rtol=0, atol=0)
