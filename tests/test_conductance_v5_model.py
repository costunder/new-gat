"""Memory-safety contracts for the V5 dynamic-conductance model."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.utils.checkpoint

from research.conductance_gat.v5.model import (
    GraphConditionedConductance,
    GraphConditionedConductanceNodeClassifier,
)


def _conductance_inputs(*, channels=8):
    state = torch.randn(7, channels)
    incidence = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6, 6]], dtype=torch.long)
    node_graph = torch.zeros(state.shape[0], dtype=torch.long)
    degree = torch.bincount(incidence.flatten(), minlength=state.shape[0]).float()
    context = torch.randn(1, 2 * channels + 8)
    return state, incidence, node_graph, degree, context


def _run_estimator(model, state, incidence, node_graph, degree, context):
    conductance = model(
        state,
        incidence,
        node_graph,
        1,
        graph_context=context,
        sample_degree=degree,
        full_degree=degree,
    )
    weights = torch.linspace(0.5, 1.5, conductance.numel())
    (conductance * weights).sum().backward()
    return (
        conductance.detach(),
        state.grad.detach().clone(),
        {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
    )


def test_score_chunk_checkpoint_is_eval_safe_and_preserves_gradients(monkeypatch):
    state, incidence, node_graph, degree, context = _conductance_inputs()
    checkpointed = GraphConditionedConductance(8, score_channels=4, edge_chunk_size=2).eval()
    direct = copy.deepcopy(checkpointed)
    real_checkpoint = torch.utils.checkpoint.checkpoint
    calls = []

    def recording_checkpoint(function, *arguments, **kwargs):
        calls.append(arguments[2].numel())
        return real_checkpoint(function, *arguments, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recording_checkpoint)
    actual = _run_estimator(
        checkpointed, state.clone().requires_grad_(True), incidence, node_graph, degree, context
    )
    monkeypatch.setattr(
        torch.utils.checkpoint,
        "checkpoint",
        lambda function, *arguments, **_kwargs: function(*arguments),
    )
    expected = _run_estimator(
        direct, state.clone().requires_grad_(True), incidence, node_graph, degree, context
    )

    assert calls == [2, 2, 2, 1]
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-7)
    assert actual[2].keys() == expected[2].keys()
    for name in actual[2]:
        torch.testing.assert_close(actual[2][name], expected[2][name], rtol=1e-5, atol=1e-7)


def test_block_checkpoint_is_not_disabled_by_calibration_eval_mode(monkeypatch):
    from research.conductance_gat.v5.train import configure_phase, parameter_group

    model = GraphConditionedConductanceNodeClassifier(
        5,
        3,
        hidden_channels=16,
        layers=2,
        heads=4,
        ffn_multiplier=2,
        dropout=0.2,
        conductance_mode="dynamic",
        edge_chunk_size=3,
        activation_checkpoint=True,
    )
    phase = configure_phase(model, "conductance_calibration", 0)
    assert phase["active_parameter_groups"] == ["conductance"]
    assert not model.training
    graph = SimpleNamespace(
        # x deliberately does not require gradients.  Calibration must still
        # discover the trainable conductance parameters captured by the
        # non-reentrant checkpoint closure.
        x=torch.randn(8, 5),
        incidence_edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 7, 7]], dtype=torch.long
        ),
    )
    real_checkpoint = torch.utils.checkpoint.checkpoint
    calls = []

    def recording_checkpoint(function, *arguments, **kwargs):
        calls.append(getattr(function, "__name__", type(function).__name__))
        return real_checkpoint(function, *arguments, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recording_checkpoint)
    model(graph).square().mean().backward()

    assert calls.count("<lambda>") == model.layers
    for name, parameter in model.named_parameters():
        if parameter_group(name) == "conductance":
            assert parameter.requires_grad
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert not parameter.requires_grad
            assert parameter.grad is None
    assert all(
        operator.estimator.score_network[-1].weight.grad.abs().sum() > 0
        for operator in model.operators
    )
