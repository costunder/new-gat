"""Unit-only math/cleanup fixtures for validation C interventions; no training."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def interventions():
    return importlib.import_module("scripts.conductance_interventions")


def _model(torch, layers=2):
    from research.conductance_gat.benchmark import ConductanceConv

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.operators = torch.nn.ModuleList([ConductanceConv(2) for _ in range(layers)])
            self.seen_training = []

        def forward(self, graph):
            assert not hasattr(graph, "y"), "Inference model must not receive held-out labels"
            self.seen_training.append(self.training)
            h = graph.x
            batch = torch.zeros(len(h), dtype=torch.long, device=h.device)
            for operator in self.operators:
                # Public legacy model API has exactly three positional inputs.
                h = operator(h, graph.incidence_edge_index, batch)
            return h

    return Model()


def _payload(torch):
    return {
        "dataset": "cora",
        "classes": 2,
        "graphs": [
            {
                "x": torch.tensor([[2.0, -1.0], [-1.0, 0.2], [0.5, 3.0], [3.0, -2.0]]),
                # Invalid train/test labels will break CE if ever evaluated.
                "y": torch.tensor([-999, 0, 1, -999]),
                "incidence_edge_index": torch.tensor([[0, 1, 1], [1, 2, 3]]),
            }
        ],
        "splits": {
            "train": None,
            "validation": torch.tensor([False, True, True, False]),
            "test": None,
        },
    }


def _fixed_gate(torch, model, value=2.0):
    for operator in model.operators:
        operator.estimator.forward = lambda gradient, _edge_features: gradient.new_full(
            (len(gradient),), value
        )


def _variable_gate(model):
    for operator in model.operators:
        operator.estimator.forward = lambda gradient, _edge_features: gradient[:, 0].abs() + 0.3


def _variants(result):
    return {row["name"]: row for row in result["variants"]}


def test_validation_only_constant_c_matches_learned_and_reports_all_variants(interventions, torch):
    model, payload = _model(torch), _payload(torch)
    _fixed_gate(torch, model)
    progress = []
    result = interventions.evaluate_interventions(
        model, payload, "cpu", edge_chunk_size=1, progress=progress.append
    )
    json.dumps(result, allow_nan=False)
    rows = _variants(result)
    assert result["split"] == "validation"
    assert len(rows) == 10
    assert progress == list(rows)
    for name, row in rows.items():
        assert row["prediction"]["count"] == 2
        if name.startswith(("learned_C", "mean_C", "shuffled_C")):
            assert row["delta_vs_learned"]["metric"] == 0
            assert row["delta_vs_learned"]["loss"] == pytest.approx(0, abs=1e-6)
            assert row["delta_vs_learned"]["logits_relative_l2"] == pytest.approx(0, abs=1e-6)
            assert row["delta_vs_learned"]["prediction_flip_fraction"] == 0
    for layer in rows["graph_off_all"]["layers"]:
        assert layer["edge_pooled"]["conductance"]["mean"] == 0
        assert layer["edge_pooled"]["conductance"]["count"] == 3
        assert layer["edge_pooled"]["c_cv"] is None
        assert layer["node_pooled"]["rho"]["mean"] == 0
        assert layer["global_update_ratio"] == 0


@pytest.mark.parametrize("mode", ["mean_C", "shuffled_C", "graph_off"])
def test_substituted_forward_matches_dense_laplacian_and_recomputes_dmax(
    interventions, torch, mode
):
    model, payload = _model(torch, layers=1), _payload(torch)
    _variable_gate(model)
    raw = payload["graphs"][0]
    # Unlike a star, this path changes its maximum weighted degree when C
    # is replaced by its edge mean, exposing a stale-denominator error.
    raw["incidence_edge_index"] = torch.tensor([[0, 1, 2], [1, 2, 3]])
    x, edges = raw["x"], raw["incidence_edge_index"]
    node_graph = torch.zeros(len(x), dtype=torch.long)
    original_c = (x[edges[1], 0] - x[edges[0], 0]).abs() + 0.3
    with torch.inference_mode():
        actual, c, degree = interventions._substituted_forward(
            model.operators[0],
            x,
            edges,
            node_graph,
            mode=mode,
            chunk_size=1,
            generator=interventions._shuffle_generator(0, 0, 0),
        )
    b = torch.zeros(edges.shape[1], len(x))
    b[torch.arange(len(c)), edges[0]] = -1
    b[torch.arange(len(c)), edges[1]] = 1
    dense_degree = b.abs().T @ c
    dense_update = x - (0.95 / dense_degree.max().clamp_min(1e-12)) * b.T @ (c[:, None] * (b @ x))
    torch.testing.assert_close(actual, dense_update)
    torch.testing.assert_close(degree, dense_degree)
    if mode == "mean_C":
        torch.testing.assert_close(c, original_c.mean().expand_as(c))
        assert float(degree.max()) != pytest.approx(float((b.abs().T @ original_c).max()))
    if mode == "shuffled_C":
        torch.testing.assert_close(c.sort().values, original_c.sort().values)
    if mode == "graph_off":
        torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_learned_reference_uses_exact_original_forward_and_c_not_recomputed(interventions, torch):
    model, payload = _model(torch, layers=1), _payload(torch)
    operator = model.operators[0]
    calls = []
    original_estimator_forward = operator.estimator.forward

    def counted(gradient, features):
        calls.append(len(gradient))
        return original_estimator_forward(gradient, features)

    operator.estimator.forward = counted
    raw = payload["graphs"][0]
    with torch.inference_mode():
        expected = model(
            SimpleNamespace(x=raw["x"], incidence_edge_index=raw["incidence_edge_index"])
        )
    calls.clear()
    records = [[]]
    with (
        torch.inference_mode(),
        interventions._instrument_operators(model, records, "learned_C", [], 0, 0, 1),
    ):
        actual = model(
            SimpleNamespace(x=raw["x"], incidence_edge_index=raw["incidence_edge_index"])
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [3]
    assert len(records[0]) == 1
    # Intervened gate is computed once per edge chunk, not followed by the
    # original Conv's second estimator invocation.
    calls.clear()
    with (
        torch.inference_mode(),
        interventions._instrument_operators(model, [[]], "mean_C", [0], 0, 0, 1),
    ):
        model(SimpleNamespace(x=raw["x"], incidence_edge_index=raw["incidence_edge_index"]))
    assert calls == [1, 1, 1]


def test_shuffle_determinism_is_independent_of_global_rng(interventions, torch):
    model, payload = _model(torch), _payload(torch)
    _variable_gate(model)
    torch.manual_seed(37)
    before = torch.random.get_rng_state().clone()
    first = interventions.evaluate_interventions(model, payload, "cpu", shuffle_seed=9)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    torch.manual_seed(93)
    second = interventions.evaluate_interventions(model, payload, "cpu", shuffle_seed=9)
    assert first == second


def test_layerwise_only_selected_layer_is_substituted(interventions, torch):
    model, payload = _model(torch), _payload(torch)
    _variable_gate(model)
    rows = _variants(interventions.evaluate_interventions(model, payload, "cpu"))
    learned = rows["learned_C"]
    second_only = rows["mean_C_layer_1"]
    assert second_only["selected_layers"] == [1]
    assert second_only["layers"][0] == learned["layers"][0]
    assert second_only["layers"][1]["edge_pooled"]["c_cv"] == pytest.approx(0)
    assert learned["layers"][1]["edge_pooled"]["c_cv"] > 0
    assert rows["graph_off_layer_0"]["layers"][0]["global_update_ratio"] == 0
    assert rows["graph_off_layer_0"]["layers"][1]["global_update_ratio"] > 0


def test_ppi_all_validation_graphs_mean_shuffle_graph_isolation_and_global_metrics(
    interventions, torch
):
    model = _model(torch, layers=1)
    _variable_gate(model)
    graph_one = {
        "x": torch.tensor([[1.0, -1.0], [2.0, 0.0], [5.0, 1.0]]),
        "y": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        "incidence_edge_index": torch.tensor([[0, 1], [1, 2]]),
    }
    graph_two = {
        "x": torch.tensor([[-20.0, 1.0], [10.0, 0.0]]),
        "y": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "incidence_edge_index": torch.tensor([[0], [1]]),
    }
    payload = {
        "dataset": "ppi",
        "classes": 2,
        "graphs": [{}, graph_one, graph_two, {}],
        "splits": {"train": None, "validation": [1, 2], "test": None},
    }
    rows = _variants(interventions.evaluate_interventions(model, payload, "cpu", layerwise=False))
    assert len(rows) == 4
    for row in rows.values():
        assert row["prediction"]["count"] == 10
        assert row["prediction"]["nodes"] == 5
        assert row["prediction"]["metric_name"] == "micro_f1"
        assert row["layers"][0]["graphs"] == 2
        assert "node_any_label_flip_fraction" in row["delta_vs_learned"]
    mean_summary = rows["mean_C_all"]["layers"][0]
    # Graph-local means are 2.3 and 30.3, not a pooled mean across all edges.
    assert mean_summary["edge_pooled"]["conductance"]["mean"] == pytest.approx((2.3 * 2 + 30.3) / 3)
    assert mean_summary["edge_pooled"]["conductance"]["quantiles"]["min"] == pytest.approx(2.3)
    assert mean_summary["edge_pooled"]["conductance"]["quantiles"]["max"] == pytest.approx(30.3)
    assert mean_summary["graph_macro"]["c_cv_mean"] == pytest.approx(0)
    assert mean_summary["edge_pooled"]["c_cv"] > 0  # Between-graph scale differences.
    assert (
        rows["shuffled_C_all"]["layers"][0]["edge_pooled"]
        == rows["learned_C"]["layers"][0]["edge_pooled"]
    )
    assert mean_summary["node_pooled"]["rho"]["mean"] == pytest.approx(
        (0.475 + 0.95 + 0.475 + 0.95 + 0.95) / 5
    )


def test_parameters_gradients_modes_hooks_forward_and_rng_are_unchanged(interventions, torch):
    model, payload = _model(torch), _payload(torch)
    model.train()
    model.operators[1].eval()  # Preserve mixed child modes, not only root mode.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    grads = [parameter.grad.clone() for parameter in model.parameters()]
    modes = [module.training for module in model.modules()]
    forward_attributes = [dict(vars(operator)).get("forward") for operator in model.operators]
    existing = model.operators[0].register_forward_hook(lambda *_: None)
    hook_ids = list(model.operators[0]._forward_hooks)
    interventions.evaluate_interventions(model, payload, "cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for parameter, grad in zip(model.parameters(), grads, strict=True):
        torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
    assert [module.training for module in model.modules()] == modes
    assert [vars(operator).get("forward") for operator in model.operators] == forward_attributes
    assert list(model.operators[0]._forward_hooks) == hook_ids
    assert not model.operators[1]._forward_hooks
    assert all(not operator.estimator._forward_hooks for operator in model.operators)
    assert model.seen_training and not any(model.seen_training)
    existing.remove()


def test_exception_restores_modes_rng_forward_and_hooks(interventions, torch):
    model, payload = _model(torch), _payload(torch)
    original_forward = model.operators[1].forward

    def consume_rng_then_fail(*_args, **_kwargs):
        torch.rand(11)
        raise RuntimeError("deliberate validation failure")

    model.operators[1].forward = consume_rng_then_fail
    rng = torch.random.get_rng_state().clone()
    model.train()
    with pytest.raises(RuntimeError, match="deliberate"):
        interventions.evaluate_interventions(model, payload, "cpu")
    assert model.training
    assert model.operators[1].forward is consume_rng_then_fail
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert all(not operator._forward_hooks for operator in model.operators)
    assert all(not operator.estimator._forward_hooks for operator in model.operators)
    model.operators[1].forward = original_forward

    # Fail inside an actual replaced Conv as well, ensuring its override is removed.
    def gate_failure(*_args, **_kwargs):
        raise RuntimeError("gate failure")

    model.operators[0].estimator.forward = gate_failure
    graph = payload["graphs"][0]
    with pytest.raises(RuntimeError, match="gate failure"), torch.inference_mode():
        with interventions._instrument_operators(model, [[], []], "mean_C", [0], 0, 0, 1):
            model(SimpleNamespace(x=graph["x"], incidence_edge_index=graph["incidence_edge_index"]))
    assert "forward" not in vars(model.operators[0])
    assert all(not operator._forward_hooks for operator in model.operators)


@pytest.mark.parametrize("chunk,seed", [(0, 0), (-1, 0), (2, -1)])
def test_invalid_options_are_rejected_without_mutation(interventions, torch, chunk, seed):
    model = _model(torch)
    with pytest.raises(ValueError):
        interventions.evaluate_interventions(
            model, _payload(torch), "cpu", edge_chunk_size=chunk, shuffle_seed=seed
        )
    assert model.training
    assert all(not operator._forward_hooks for operator in model.operators)


def test_no_edges_and_nonfinite_c_are_explicit(interventions, torch):
    model, payload = _model(torch), _payload(torch)
    payload["graphs"][0]["incidence_edge_index"] = torch.empty(2, 0, dtype=torch.long)
    result = interventions.evaluate_interventions(model, payload, "cpu", layerwise=False)
    json.dumps(result, allow_nan=False)
    for row in result["variants"]:
        assert row["delta_vs_learned"]["logits_relative_l2"] == 0
        assert row["layers"][0]["edge_pooled"]["conductance"]["count"] == 0
    payload = _payload(torch)
    _fixed_gate(torch, model, float("nan"))
    with pytest.raises((FloatingPointError, ValueError), match="finite|Invalid"):
        interventions.evaluate_interventions(model, payload, "cpu")
    assert all(not operator._forward_hooks for operator in model.operators)
