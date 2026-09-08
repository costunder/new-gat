"""Synthetic CPU debug tests, not real-data accuracy or GPU performance tests."""

from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.conductance_gat.v5.diagnostics import PreparedValidationGraph
from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.stage_audit import audit_stage_roles


class DebugGraph(SimpleNamespace):
    def clone(self):
        return copy.deepcopy(self)

    def to(self, device, **kwargs):
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self

    def items(self):
        return vars(self).items()


def _fixture(mode="dynamic", batched=False):
    torch.manual_seed(421)
    model = GraphConditionedConductanceNodeClassifier(
        5,
        3,
        hidden_channels=16,
        layers=2,
        heads=4,
        ffn_multiplier=2,
        dropout=0.25,
        conductance_mode=mode,
        conductance_backend="optimization",
        solver_steps=8,
        solver_cost_scaling="width_scaled",
        beta_initial=0.5,
        activation_checkpoint=False,
    )
    edges = torch.tensor([[0, 0, 1, 1, 2, 4, 4, 5], [1, 2, 2, 3, 3, 5, 6, 6]])
    graph = DebugGraph(
        x=torch.randn(8, 5), incidence_edge_index=edges, y=torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
    )
    if batched:
        graph.batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        graph.num_graphs = graph._v5_num_graphs = 2
        graph.y = torch.tensor(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=torch.float,
        )
    else:
        graph.edge_normalization_weight = torch.tensor([1.0, 2.0, 1.0, 1.5, 1.2, 1.0, 2.0, 1.0])
        graph.sampling_correction = graph.edge_normalization_weight
    return model, graph


def _audit(model, source, indices=None, **kwargs):
    return audit_stage_roles(
        model,
        source,
        indices,
        device=torch.device("cpu"),
        reference_steps=kwargs.pop("reference_steps", 32),
        reference_tolerance=kwargs.pop("reference_tolerance", 1e-4),
        **kwargs,
    )


def _state(model):
    return {
        "weights": {key: value.clone() for key, value in model.state_dict().items()},
        "gradients": [
            None if value.grad is None else value.grad.clone() for value in model.parameters()
        ],
        "training": [module.training for module in model.modules()],
        "diagnostics": [
            (module, {key: value for key, value in vars(module).items() if key.startswith("last_")})
            for module in model.modules()
        ],
        "settings": [
            (operator.estimator.override, operator.estimator.solver_steps)
            for operator in model.operators
        ],
        "torch_rng": torch.get_rng_state().clone(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def _assert_restored(model, state):
    for key, value in model.state_dict().items():
        assert torch.equal(value, state["weights"][key])
    for parameter, grad in zip(model.parameters(), state["gradients"], strict=True):
        assert parameter.grad is None if grad is None else torch.equal(parameter.grad, grad)
    assert [module.training for module in model.modules()] == state["training"]
    assert [
        (operator.estimator.override, operator.estimator.solver_steps)
        for operator in model.operators
    ] == state["settings"]
    for module, previous in state["diagnostics"]:
        current = {key: value for key, value in vars(module).items() if key.startswith("last_")}
        assert current.keys() == previous.keys()
        assert all(current[key] is value for key, value in previous.items())
    assert torch.equal(torch.get_rng_state(), state["torch_rng"])
    assert random.getstate() == state["python_rng"]
    left, right = np.random.get_state(), state["numpy_rng"]
    assert left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]
    assert not any(module._forward_hooks for module in model.operators)


def test_validation_interventions_and_local_reference_are_read_only_and_json_serializable():
    model, graph = _fixture()
    model(graph).square().mean().backward()
    model.blocks[0].eval()  # Preserve mixed flags rather than flattening them.
    model.operators[1].estimator.override = "mean"
    state, x = _state(model), graph.x.clone()
    result = _audit(model, graph, torch.tensor([1, 3, 5, 7]))
    _assert_restored(model, state)
    assert torch.equal(graph.x, x)
    json.dumps(result, allow_nan=False)
    assert result["execution_status"] == "passed"
    assert result["contribution_status"] == "observed"
    assert result["scope"]["test_used"] is False
    assert result["scope"]["parameters_updated"] is False
    assert result["interventions"]["learned"]["label_count"] == 4
    assert result["interventions"]["learned"]["logit_max_abs"] == 0
    assert result["interventions"]["c_one"]["logit_max_abs"] > 0
    assert len(result["local_layer_comparisons"]) == 2
    for row in result["local_layer_comparisons"]:
        assert row["deployed_steps"] == 8 and row["reference_steps"] == 32
        assert row["reference_is_exact_optimum"] is False
        assert row["baseline_solver"]["executed_steps"] == 8
        assert row["reference_solver"]["executed_steps"] == 32
        assert row["c_deployed_vs_reference"]["relative_l2"] > 0
        assert len(row["reference_tolerance_reached_by_graph"]) == 1


def test_equal_k_has_zero_local_and_whole_model_error_and_never_asserts_convergence():
    model, graph = _fixture()
    result = _audit(model, graph, torch.arange(8), reference_steps=8, reference_tolerance=1e-30)
    assert result["interventions"]["higher_k_full_model"]["logit_max_abs"] == 0
    for row in result["local_layer_comparisons"]:
        assert row["c_deployed_vs_reference"]["relative_l2"] == 0
        assert row["operator_deployed_vs_reference"]["max_abs"] == 0
        assert row["reference_tolerance_reached_by_graph"] == [False]


def test_ppi_disjoint_batches_are_not_split_and_generator_is_consumed_once():
    model, graph = _fixture(batched=True)
    seen = []

    def source():
        for number in range(2):
            seen.append(number)
            random.random()
            np.random.rand()
            torch.rand(1)
            yield graph

    state = _state(model)
    result = _audit(model, source())
    _assert_restored(model, state)
    assert seen == [0, 1]
    assert result["scope"]["metric"] == "micro_f1"
    assert [row["graphs"] for row in result["inputs"]] == [2, 2]
    assert [row["nodes"] for row in result["inputs"]] == [8, 8]
    assert len(result["local_layer_comparisons"]) == 4
    assert all(value["label_count"] == 48 for value in result["interventions"].values())
    assert graph.x.device.type == "cpu"


def test_fixed_c_reports_no_inner_optimization_or_contribution_evidence():
    model, graph = _fixture(mode="fixed_one")
    result = _audit(model, graph, torch.arange(8))
    assert result["contribution_status"] == "inconclusive"
    assert all(value["logit_max_abs"] == 0 for value in result["interventions"].values())
    for row in result["local_layer_comparisons"]:
        assert row["reference_tolerance_reached_by_graph"] is None
        assert row["baseline_solver"]["enabled"] is False


def test_prepared_full_graph_is_supported_without_changing_source_tensors():
    model, graph = _fixture()
    source = PreparedValidationGraph(graph, torch.device("cpu"))
    before = source.graph.x.clone()
    result = _audit(model, source, torch.arange(8))
    assert torch.equal(source.graph.x, before)
    assert result["interventions"]["learned"]["label_count"] == 8


def test_nonfinite_failure_restores_parameters_flags_settings_diagnostics_and_rng():
    model, graph = _fixture()
    model(graph)
    state = _state(model)
    handle = model.decoder.register_forward_hook(lambda module, args, output: output * float("nan"))
    try:
        with pytest.raises(FloatingPointError, match="nonfinite"):
            _audit(model, graph, torch.arange(8))
    finally:
        handle.remove()
    _assert_restored(model, state)


def test_failure_inside_reference_hook_restores_all_temporary_state(monkeypatch):
    model, graph = _fixture()
    estimator = model.operators[0].estimator
    original = estimator.forward

    def fail_reference(*args, **kwargs):
        if estimator.solver_steps == 32:
            raise RuntimeError("synthetic reference failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator, "forward", fail_reference)
    state = _state(model)
    with pytest.raises(RuntimeError, match="synthetic reference failure"):
        _audit(model, graph, torch.arange(8))
    _assert_restored(model, state)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"reference_steps": 4}, "cannot reduce"),
        ({"reference_steps": True}, "positive integer"),
        ({"reference_tolerance": 0}, "finite and positive"),
        ({"reference_tolerance": float("nan")}, "finite and positive"),
        ({"precision": "fp16"}, "precision"),
    ],
)
def test_invalid_contracts_fail_without_changing_model(kwargs, match):
    model, graph = _fixture()
    state = _state(model)
    with pytest.raises(ValueError, match=match):
        _audit(model, graph, torch.arange(8), **kwargs)
    _assert_restored(model, state)


def test_empty_or_invalid_validation_data_is_not_silently_reported_as_success():
    model, graph = _fixture()
    with pytest.raises(ValueError, match="nonempty"):
        _audit(model, graph, torch.tensor([], dtype=torch.long))
    with pytest.raises(ValueError, match="no batches"):
        _audit(model, iter(()))


def test_bf16_keeps_geometry_fp32_and_restores_flags():
    model, graph = _fixture()
    state = _state(model)
    result = _audit(model, graph, torch.arange(8), precision="bf16")
    _assert_restored(model, state)
    assert result["resources"]["precision"] == "bf16"
    assert result["resources"]["geometry_precision"] == "fp32"
    json.dumps(result, allow_nan=False)


def test_second_layer_local_reference_matches_independent_frozen_input_calculation():
    model, graph = _fixture()
    model.eval()
    captured = []
    operator = model.operators[1]
    handle = operator.register_forward_hook(
        lambda module, args, kwargs, output: captured.append(
            (args, kwargs, output.detach().clone(), module.estimator.last_c.clone())
        ),
        with_kwargs=True,
    )
    with torch.no_grad():
        model(graph)
    handle.remove()
    args, kwargs, deployed_output, deployed_c = captured[0]
    operator.estimator.solver_steps = 32
    with torch.no_grad():
        reference_output = operator(*args, **kwargs)
        reference_c = operator.estimator.last_c.clone()
    operator.estimator.solver_steps = 8
    c_relative = float(torch.linalg.vector_norm(deployed_c - reference_c)) / float(
        torch.linalg.vector_norm(reference_c)
    )
    operator_relative = float(torch.linalg.vector_norm(deployed_output - reference_output)) / float(
        torch.linalg.vector_norm(reference_output)
    )
    result = _audit(model, graph, torch.arange(8))
    row = result["local_layer_comparisons"][1]
    assert row["c_deployed_vs_reference"]["relative_l2"] == pytest.approx(c_relative, rel=1e-6)
    assert row["operator_deployed_vs_reference"]["relative_l2"] == pytest.approx(
        operator_relative, rel=1e-6
    )


def test_zero_beta_bypass_is_inconclusive_even_when_solver_c_varies():
    model, graph = _fixture()
    with torch.no_grad():
        for operator in model.operators:
            operator.beta_estimator.network[-1].weight.zero_()
            operator.beta_estimator.network[-1].bias.fill_(-1000)
    result = _audit(model, graph, torch.arange(8))
    assert result["execution_status"] == "passed"
    assert result["contribution_status"] == "inconclusive"
    assert all(value["logit_max_abs"] == 0 for value in result["interventions"].values())
    for row in result["local_layer_comparisons"]:
        assert row["baseline_c"]["std"] > 0
        assert row["baseline_beta"]["max"] == 0
        assert row["operator_c_one_vs_deployed"]["max_abs"] == 0
