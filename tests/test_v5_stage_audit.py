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
from research.conductance_gat.v5.stage_audit import _json, audit_stage_roles


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


def _fixture(mode="dynamic", batched=False, **overrides):
    torch.manual_seed(421)
    configuration = dict(
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
    model = GraphConditionedConductanceNodeClassifier(5, 3, **{**configuration, **overrides})
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
    assert result["contribution_status"] == "observed_above_repeat_noise"
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
    assert result["contribution_status"] == "not_applicable"
    assert all(
        result["interventions"][name]["logit_max_abs"] == 0
        for name in ("learned", "c_one", "mean_c", "shuffled_c")
    )
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


def test_same_checkpoint_eval_repeats_do_not_become_training_seeds():
    model, graph = _fixture()
    result = _audit(model, graph, torch.arange(8), repeat_evaluations=6)
    noise = result["repeat_noise_control"]
    assert noise["evaluation_count"] == len(noise["repeats"]) == 6
    assert noise["different_training_seeds"] is False
    assert noise["max_logit_difference_l2"] == 0
    assert "mean_head_beta" in result["interventions"]
    assert result["scope"]["full_validation_passes"] == 11


def test_fixed_c_numerical_jitter_is_never_a_contribution_certificate():
    model, graph = _fixture(mode="fixed_one")
    call = []

    def simulated_numerical_jitter(module, args, output):
        call.append(1)
        return output + 0.0001 * len(call)

    handle = model.decoder.register_forward_hook(simulated_numerical_jitter)
    try:
        result = _audit(model, graph, torch.arange(8))
    finally:
        handle.remove()
    assert result["repeat_noise_control"]["max_logit_difference_l2"] > 0
    assert result["interventions"]["c_one"]["logit_max_abs"] > 0
    assert result["contribution_status"] == "not_applicable"


@pytest.mark.parametrize("normalization", ["row", "symmetric"])
def test_per_head_real_audit_has_scoped_distribution_and_headwise_residual(normalization):
    model, graph = _fixture(
        batched=True, conductance_heads="per_head", propagation_normalization=normalization
    )
    result = _audit(model, [graph])
    json.dumps(result, allow_nan=False)
    for local in result["local_layer_comparisons"]:
        distribution = local["distribution"]
        assert distribution["conductance_layout"] == "per_head_E_H"
        assert distribution["propagation_normalization"] == normalization
        assert len(distribution["graphs"]) == 2
        assert len(local["reference_tolerance_reached_by_graph"]) == 2
        for graph_row in distribution["graphs"]:
            assert len(graph_row["raw_c"]["quantiles"]["0.5"]) == 4
            assert graph_row["heads"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "options",
    [
        {"conductance_backend": "mlp", "solver_cost_scaling": "legacy_unit"},
        {"conductance_generator": "entropy_exact", "solver_degree_barrier": 0.0},
        {"conductance_generator": "degree_only"},
    ],
)
def test_noniterative_controls_have_no_fake_higher_k_measurement(options):
    model, graph = _fixture(**options)
    result = _audit(model, graph, torch.arange(8))
    assert result["reference_contract"]["applicable"] is False
    assert "higher_k_full_model" not in result["interventions"]
    for local in result["local_layer_comparisons"]:
        assert local["c_deployed_vs_reference"] is None
        assert local["reference_tolerance_reached_by_graph"] is None
        assert local["distribution"]["graphs"]


def test_optional_actual_task_head_c_gradients_preserve_grads_weights_rng_and_hooks():
    model, graph = _fixture(conductance_heads="per_head", propagation_normalization="row")
    model(graph).square().mean().backward()
    state = _state(model)
    result = _audit(model, graph, torch.arange(8), head_gradient_conflict=True)
    _assert_restored(model, state)
    audit = result["head_gradient_conflict"]
    assert audit["measured"] is True
    assert "not theta-space" in audit["scope"]
    assert len(audit["layers"]) == 2
    assert all(row["measured"] for row in audit["layers"])
    assert all(len(row["graphs"][0]["head_pair_dot_matrix"]) == 4 for row in audit["layers"])
    assert result["scope"]["autograd_vjp_used"] is True
    assert all(not op.estimator._forward_hooks for op in model.operators)


def test_optional_shared_c_head_expansion_keeps_forward_and_parameters_unchanged():
    model, graph = _fixture()
    state = _state(model)
    result = _audit(model, graph, torch.arange(8), head_gradient_conflict=True)
    _assert_restored(model, state)
    for row in result["head_gradient_conflict"]["layers"]:
        assert row["equal_c_forward_difference"]["relative_l2"] < 1e-5
        assert row["graphs"][0]["shared_c_sum_gradient_norm"] > 0


def test_too_few_same_checkpoint_repeats_are_rejected():
    model, graph = _fixture()
    with pytest.raises(ValueError, match="at least 5"):
        _audit(model, graph, torch.arange(8), repeat_evaluations=4)


def test_optional_vjp_failure_removes_hooks_and_restores_model(monkeypatch):
    model, graph = _fixture()
    state = _state(model)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic requested VJP failure")

    monkeypatch.setattr(torch.autograd, "grad", fail)
    with pytest.raises(RuntimeError, match="requested VJP"):
        _audit(model, graph, torch.arange(8), head_gradient_conflict=True)
    _assert_restored(model, state)
    assert all(not op.estimator._forward_hooks for op in model.operators)


def test_summary_json_packs_once_per_dtype_without_losing_nested_shapes_or_int64(monkeypatch):
    calls = []
    original = torch.Tensor.cpu

    def count_cpu(tensor, *args, **kwargs):
        calls.append((tensor.dtype, tensor.numel()))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", count_cpu)
    floating = torch.tensor([[0.5, 1.5], [2.5, 3.5]]).T
    data = {
        "nested": (floating, {"same_tensor": floating, "empty": torch.empty(2, 0)}),
        "scalar": torch.tensor(0.25),
        "integer": torch.tensor(2**62 + 1),
        "boolean": torch.tensor([True, False]),
        "none": None,
    }
    result = _json(data)
    assert len(calls) == 3
    assert sum(count for dtype, count in calls if dtype == torch.float32) == 5
    assert result["nested"] == [
        [[0.5, 2.5], [1.5, 3.5]],
        {"same_tensor": [[0.5, 2.5], [1.5, 3.5]], "empty": [[], []]},
    ]
    assert result["integer"] == 2**62 + 1 and type(result["integer"]) is int
    assert result["boolean"] == [True, False] and type(result["boolean"][0]) is bool
    assert result["scalar"] == 0.25 and result["none"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "bad", [torch.tensor(float("nan")), torch.tensor(float("inf")), float("nan")]
)
def test_packed_summary_json_rejects_nonfinite_values(bad):
    with pytest.raises(FloatingPointError, match="nonfinite stage audit statistic"):
        _json({"nested": [torch.ones(2), {"bad": bad}]})
