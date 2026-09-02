"""CPU fixtures for V4 selected-checkpoint diagnostic contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from research.conductance_gat.ablation.model import state_sha256  # noqa: E402
from research.conductance_gat.v4 import diagnostics  # noqa: E402
from research.conductance_gat.v4.diagnostics import (  # noqa: E402
    Intervention,
    best_checkpoint_interventions,
    evaluate_validation,
)
from research.conductance_gat.v4.model import RelativeCSpatialNodeClassifier  # noqa: E402
from research.conductance_gat.v4.operator import symmetric_spatial_propagation  # noqa: E402


def _graph():
    return SimpleNamespace(
        x=torch.tensor(
            [[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]
        ),
        y=torch.tensor([0, 1, 0, 1]),
        incidence_edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]),
    )


def _model(*, gate_mode="relative"):
    torch.manual_seed(17)
    model = RelativeCSpatialNodeClassifier(
        3,
        2,
        hidden_channels=8,
        layers=2,
        dropout=0.5,
        gate_mode=gate_mode,
        spatial_mode="fixed_identity",
        edge_chunk_size=2,
    )
    if gate_mode == "relative":
        with torch.no_grad():
            for operator in model.operators:
                operator.estimator.network[-1].weight.normal_(std=0.2)
    return model


class _MovableGraph(SimpleNamespace):
    def to(self, device, non_blocking=False):
        del non_blocking
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


def _packed_ppi_graph():
    return _MovableGraph(
        x=torch.tensor(
            [
                [0.5, 1.0, 2.0],
                [1.0, 2.0, 0.5],
                [2.0, 0.5, 1.0],
                [3.0, 1.0, 2.0],
                [0.2, 0.7, 1.3],
                [1.1, 0.1, 0.4],
            ]
        ),
        y=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        ),
        incidence_edge_index=torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]),
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        num_graphs=2,
    )


def test_graphwise_positive_constant_c_is_algebraically_equivalent_to_ones():
    residual = torch.tensor(
        [
            [0.2, -1.1],
            [1.7, 0.4],
            [-0.6, 2.2],
            [3.0, -0.5],
            [0.8, 1.3],
            [-1.4, 0.6],
        ],
        dtype=torch.float64,
    )
    message = residual.flip(0).clone()
    incidence = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    alpha = torch.tensor(0.37, dtype=torch.float64)
    graph_constant_c = torch.tensor([2.0, 2.0, 7.0, 7.0], dtype=torch.float64)
    actual = symmetric_spatial_propagation(
        residual,
        message,
        graph_constant_c,
        incidence,
        alpha,
        edge_chunk_size=2,
    )
    expected = symmetric_spatial_propagation(
        residual,
        message,
        torch.ones_like(graph_constant_c),
        incidence,
        alpha,
        edge_chunk_size=2,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_ppi_validation_uses_both_graphs_global_micro_f1_and_labelwise_predictions():
    graph, model = _packed_ppi_graph(), _model()
    data = {"validation": [graph]}
    result, logits = evaluate_validation(model, data, None, device=torch.device("cpu"))
    predicted, truth = logits > 0, graph.y > 0
    true_positive = int((predicted & truth).sum())
    denominator = int(predicted.sum() + truth.sum())
    expected = 2 * true_positive / denominator if denominator else 0.0
    assert result["metric"] == pytest.approx(expected)
    assert result["metric_name"] == "micro_f1"
    assert result["prediction_rule"] == "logit_gt_zero_node_label"
    assert result["validation_graph_count"] == 2
    assert len(result["layers"]) == len(model.operators)

    audit = best_checkpoint_interventions(
        model,
        data,
        None,
        result,
        logits,
        seed=17,
        device=torch.device("cpu"),
    )
    assert audit["metric_name"] == "micro_f1"
    assert audit["prediction_rule"] == "logit_gt_zero_node_label"
    assert audit["validation_graph_count"] == 2
    for contract in audit["mean_c_numeric_check"]["replacement_contracts"].values():
        assert contract["edge_counts"] == [4, 4]


def test_ppi_prediction_change_is_labelwise_not_argmax():
    left = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
    right = torch.tensor([[1.0, 2.0], [-2.0, 4.0]])
    assert not torch.equal(left.argmax(-1), right.argmax(-1))
    assert torch.equal(
        diagnostics._prediction_tensor(left, "logit_gt_zero_node_label"),
        diagnostics._prediction_tensor(right, "logit_gt_zero_node_label"),
    )


def test_separate_forward_jitter_is_informational_and_preserves_model(monkeypatch):
    graph, model = _graph(), _model()
    indices = torch.tensor([0, 1, 2])
    model.train()
    model.decoder.eval()
    model.operators[0].raw_alpha.grad = torch.ones(())
    original, reference = evaluate_validation(model, graph, indices)
    before_state = state_sha256(model)
    before_modes = [module.training for module in model.modules()]
    before_rng = torch.random.get_rng_state().clone()
    real_evaluate = diagnostics.evaluate_validation
    calls = 0

    def jittered_evaluate(*args, **kwargs):
        nonlocal calls
        result, logits = real_evaluate(*args, **kwargs)
        calls += 1
        if calls == 1:  # mean-C; emulate harmless separate-forward CUDA scatter drift.
            logits = logits.clone()
            logits[0, 0] += 0.01
        return result, logits

    monkeypatch.setattr(diagnostics, "evaluate_validation", jittered_evaluate)
    audit = best_checkpoint_interventions(
        model, graph, indices, original, reference, seed=17
    )

    numeric = audit["mean_c_numeric_check"]
    assert audit["status"] == "passed"
    assert numeric["within_declared_tolerance"] is False
    assert numeric["role"] == "informational_non_gating"
    assert numeric["separate_full_graph_forwards"] is True
    assert "passed" not in numeric
    assert 0 <= numeric["logit_mean_absolute_delta"] <= numeric["logit_max_absolute_delta"]
    assert numeric["replacement_contracts"]["mean_c"]["satisfied"] is True
    assert numeric["replacement_contracts"]["ones_c"]["satisfied"] is True
    assert state_sha256(model) == before_state
    assert [module.training for module in model.modules()] == before_modes
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert model.operators[0].raw_alpha.grad.item() == 1
    assert all(
        not operator._forward_hooks
        and not operator.estimator._forward_hooks
        and not operator.message_transform._forward_hooks
        for operator in model.operators
    )


def test_nonfinite_reference_logits_are_rejected():
    graph, model = _graph(), _model()
    indices = torch.tensor([0, 1, 2])
    original, reference = evaluate_validation(model, graph, indices)
    reference[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite mean_c logits"):
        best_checkpoint_interventions(model, graph, indices, original, reference, seed=0)


@pytest.mark.parametrize("name", ["mean_c", "ones_c"])
def test_fixed_c_hooks_produce_exact_ones_and_record_contract(name):
    graph, model = _graph(), _model(gate_mode="fixed_one")
    observed = []
    handles = []
    intervention = Intervention(model, name, seed=0)
    try:
        with intervention:
            handles = [
                operator.estimator.register_forward_hook(
                    lambda module, inputs, output: observed.append(output.detach().clone())
                )
                for operator in model.operators
            ]
            with torch.no_grad():
                model.eval()(graph)
    finally:
        for handle in handles:
            handle.remove()

    assert len(observed) == len(model.operators)
    assert all(torch.equal(value, torch.ones_like(value)) for value in observed)
    summary = intervention.contract_summary(len(model.operators))
    assert summary["satisfied"] is True and summary["layers_checked"] == len(model.operators)
    assert summary["contract"] == (
        "graph_constant_positive" if name == "mean_c" else "exact_one"
    )
