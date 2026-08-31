"""Exact small-graph math checks, never research training or CUDA fallbacks."""

from __future__ import annotations

import copy
import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def audit():
    return importlib.import_module("scripts.conductance_gate_audit")


def _fixture(torch, *, dropout=0.4, empty_edges=False):
    from research.conductance_gat.benchmark import ConductanceNodeClassifier

    with torch.random.fork_rng():
        torch.manual_seed(73)
        model = ConductanceNodeClassifier(3, 3, hidden_channels=4, layers=2, dropout=dropout)
        x = torch.randn(6, 3)
    edges = torch.tensor([[0, 1, 2, 0, 3, 4], [1, 2, 3, 3, 4, 5]])
    if empty_edges:
        edges = torch.empty((2, 0), dtype=torch.long)
    raw = {"x": x, "y": torch.tensor([0, 1, 2, 0, 1, 2]), "incidence_edge_index": edges}
    payload = {
        "dataset": "cora",
        "classes": 3,
        "graphs": [raw],
        "splits": {
            "train": torch.tensor([True, True, False, False, False, False]),
            "validation": torch.tensor([False, False, True, True, False, False]),
            "test": torch.tensor([False, False, False, False, True, True]),
        },
    }
    return model, payload


def _snapshot(torch, model):
    return (
        {name: value.detach().clone() for name, value in model.state_dict().items()},
        [module.training for module in model.modules()],
        [
            (parameter.grad, None if parameter.grad is None else parameter.grad.clone())
            for parameter in model.parameters()
        ],
        torch.get_rng_state().clone(),
        [
            (len(module._forward_hooks), len(module._forward_pre_hooks))
            for module in model.modules()
        ],
    )


def _assert_unchanged(torch, model, before):
    state, modes, grads, rng, hooks = before
    assert modes == [module.training for module in model.modules()]
    assert hooks == [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()
    ]
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    for parameter, (reference, value) in zip(model.parameters(), grads, strict=True):
        assert parameter.grad is reference
        if value is not None:
            assert torch.equal(parameter.grad, value)
    assert torch.equal(torch.get_rng_state(), rng)


def test_exact_train_gradient_and_raw_logit_gradient(audit, torch):
    model, payload = _fixture(torch)
    model.eval()
    raw_outputs = []

    def capture(_module, _inputs, output):
        output.retain_grad()
        raw_outputs.append(output)

    handles = [
        operator.estimator.network[4].register_forward_hook(capture) for operator in model.operators
    ]
    graph = SimpleNamespace(
        **{key: value for key, value in payload["graphs"][0].items() if key != "y"}
    )
    logits = model(graph)
    mask = payload["splits"]["train"]
    loss = torch.nn.functional.cross_entropy(logits[mask], payload["graphs"][0]["y"][mask])
    parameters = list(model.parameters())
    expected_gradients = torch.autograd.grad(loss, parameters)
    for handle in handles:
        handle.remove()
    report = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    assert report["label_scope"] == "train_only"
    assert report["loss"]["value"] == pytest.approx(float(loss.detach()))
    for (name, parameter), expected in zip(
        model.named_parameters(), expected_gradients, strict=True
    ):
        record = report["parameters"][name]
        norm = float(expected.double().norm())
        assert record["task_gradient"]["l2_norm"] == pytest.approx(norm, abs=1e-12)
        assert record["weight_decay_term_norm"] == pytest.approx(
            0.0005 * float(parameter.detach().double().norm())
        )
    for layer, raw in zip(report["layers"], raw_outputs, strict=True):
        moments = layer["tensors"]["raw_logit_gradient"]["all_element_moments"]
        assert moments["l2_norm"] == pytest.approx(float(raw.grad.double().norm()))
        assert layer["tensors"]["raw_logit_gradient"]["observed_elements"] == raw.numel()
    json.dumps(report, allow_nan=False)


def test_never_reads_held_out_node_labels(audit, torch):
    model, payload = _fixture(torch)
    first = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.01)
    poisoned = copy.deepcopy(payload)
    poisoned["graphs"][0]["y"][~poisoned["splits"]["train"]] = -999999
    second = audit.audit_gate_gradients(model, poisoned, "cpu", weight_decay=0.01)
    assert first == second


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_restores_modes_existing_grads_rng_buffers_and_hooks(audit, torch, mode):
    model, payload = _fixture(torch)
    model.train()
    model.operators[0].eval()
    model.register_buffer("audit_buffer", torch.tensor([3.0]))
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, 2.0) if index % 2 else None
    before = _snapshot(torch, model)
    result = audit.audit_gate_gradients(
        model, payload, "cpu", weight_decay=0.0005, mode=mode, rng_seed=17
    )
    _assert_unchanged(torch, model, before)
    again = audit.audit_gate_gradients(
        model, payload, "cpu", weight_decay=0.0005, mode=mode, rng_seed=17
    )
    assert result == again
    _assert_unchanged(torch, model, before)


def test_failure_removes_hooks_and_restores_state(audit, torch, monkeypatch):
    model, payload = _fixture(torch)
    model.train()
    model.operators[0].eval()
    before = _snapshot(torch, model)

    def fail(_input):
        raise RuntimeError("forced decoder failure")

    monkeypatch.setattr(model.decoder, "forward", fail)
    with pytest.raises(RuntimeError, match="forced decoder"):
        audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005, mode="train")
    _assert_unchanged(torch, model, before)


def test_partial_hook_install_failure_is_cleaned(audit, torch):
    model, payload = _fixture(torch)
    model.operators[1].estimator.mode = "gradient_only"
    before = _snapshot(torch, model)
    with pytest.raises(ValueError, match="full"):
        audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    _assert_unchanged(torch, model, before)


def test_buffer_mutation_and_failure_after_gradients_are_restored(audit, torch, monkeypatch):
    model, payload = _fixture(torch)
    model.register_buffer("counter", torch.tensor(0))
    original_forward = model.forward

    def changing_forward(graph):
        model.counter.add_(1)
        return original_forward(graph)

    monkeypatch.setattr(model, "forward", changing_forward)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = _snapshot(torch, model)
    audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005, mode="train")
    _assert_unchanged(torch, model, before)

    def fail_report(*_args, **_kwargs):
        raise RuntimeError("report failure after gradient computation")

    monkeypatch.setattr(audit, "_parameter_report", fail_report)
    with pytest.raises(RuntimeError, match="after gradient"):
        audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005, mode="train")
    _assert_unchanged(torch, model, before)


def test_nonfinite_forward_is_rejected_with_hooks_removed(audit, torch):
    model, payload = _fixture(torch)
    payload["graphs"][0]["x"][0, 0] = float("nan")
    before = _snapshot(torch, model)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    _assert_unchanged(torch, model, before)


def test_audit_enables_autograd_inside_outer_inference_context(audit, torch):
    model, payload = _fixture(torch)
    with torch.inference_mode():
        report = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    assert report["parameters"]["decoder.weight"]["task_gradient"]["l2_norm"] > 0


def test_zero_gates_and_empty_edges_stay_json_finite(audit, torch):
    model, payload = _fixture(torch, empty_edges=True)
    with torch.no_grad():
        for operator in model.operators:
            for parameter in operator.estimator.parameters():
                parameter.zero_()
    report = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0)
    json.dumps(report, allow_nan=False)
    for layer in report["layers"]:
        for statistic in layer["tensors"].values():
            assert statistic["observed_elements"] == 0
            assert statistic["all_element_moments"] is None
            assert statistic["quantile_sample"]["sample_count"] == 0
    for name, record in report["parameters"].items():
        assert record["task_decay_cosine"] is None
        if ".estimator." in name:
            assert record["parameter"]["l2_norm"] == 0
            assert record["parameter"]["zero_fraction"] == 1


def test_frozen_parameter_reports_missing_not_zero_gradient(audit, torch):
    model, payload = _fixture(torch)
    parameter = model.operators[0].estimator.network[0].weight
    parameter.requires_grad_(False)
    report = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    record = report["parameters"]["operators.0.estimator.network.0.weight"]
    assert record["requires_grad"] is False
    assert record["task_gradient"] == {"is_none": True, "l2_norm": None, "max_absolute": None}
    assert record["task_to_decay_norm_ratio"] is None


def _ppi_fixture(torch):
    model, _ = _fixture(torch)
    graphs = []
    for index, nodes in enumerate([3, 5, 2, 4, 3]):
        x = torch.arange(nodes * 3, dtype=torch.float32).reshape(nodes, 3) / 10 + index
        y = (torch.arange(nodes * 3).reshape(nodes, 3) % 2).float()
        edges = torch.stack((torch.arange(nodes - 1), torch.arange(1, nodes)))
        graphs.append({"x": x, "y": y, "incidence_edge_index": edges})
    return model, {
        "dataset": "ppi",
        "classes": 3,
        "graphs": graphs,
        "splits": {"train": [0, 1, 2], "validation": [3], "test": [4]},
    }


def test_ppi_packed_batch_and_label_weighted_multi_batch_gradient(audit, torch):
    model, payload = _ppi_fixture(torch)
    model.eval()
    graph, labels = audit._pack_ppi_graphs(payload["graphs"], [0, 1, 2], "cpu")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(graph), labels)
    expected = torch.autograd.grad(loss, tuple(model.parameters()))
    report = audit.audit_gate_gradients(
        model, payload, "cpu", weight_decay=0.0005, ppi_batches=2, ppi_batch_size=2
    )
    assert [batch["graph_indices"] for batch in report["batches"]] == [[0, 1], [2]]
    assert report["loss"]["train_label_elements"] == 30
    assert report["loss"]["value"] == pytest.approx(float(loss.detach()))
    assert [batch["objective_weight"] for batch in report["batches"]] == [0.8, 0.2]
    for (name, _), gradient in zip(model.named_parameters(), expected, strict=True):
        assert report["parameters"][name]["task_gradient"]["l2_norm"] == pytest.approx(
            float(gradient.double().norm()), rel=2e-5, abs=1e-8
        )


def test_ppi_default_uses_one_train_batch_and_never_held_out_graphs(audit, torch):
    model, payload = _ppi_fixture(torch)
    first = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    assert [batch["graph_indices"] for batch in first["batches"]] == [[0, 1]]
    payload["graphs"][2]["x"].fill_(float("nan"))
    for index in [3, 4]:
        payload["graphs"][index]["x"].fill_(float("nan"))
        payload["graphs"][index]["y"].fill_(float("nan"))
    second = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    assert first == second


def test_rejects_overlapping_train_splits(audit, torch):
    model, payload = _fixture(torch)
    payload["splits"]["test"][0] = True
    with pytest.raises(ValueError, match="overlaps"):
        audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    model, payload = _ppi_fixture(torch)
    payload["splits"]["test"].append(0)
    with pytest.raises(ValueError, match="disjoint"):
        audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)


def test_streaming_stats_are_exact_but_quantiles_explicitly_sampled(audit, torch):
    statistic = audit._StreamingDistribution(expected_count=24, sample_limit=5, near_zero=1e-8)
    first = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    second = first + 100
    statistic.append(first)
    statistic.append(second)
    expected = torch.cat((first.reshape(-1), second.reshape(-1))).double()
    report = statistic.report()
    moments = report["all_element_moments"]
    assert report["observed_elements"] == 24
    assert report["observed_calls"] == 2
    assert moments["mean"] == pytest.approx(float(expected.mean()))
    assert moments["std_population"] == pytest.approx(float(expected.std(correction=0)))
    assert moments["l2_norm"] == pytest.approx(float(expected.norm()))
    assert moments["min"] == 0
    assert moments["max"] == 122
    sample = report["quantile_sample"]
    assert sample["sample_count"] == sample["sample_limit"] == 5
    positions = torch.tensor([2, 7, 12, 16, 21])
    expected_quantiles = torch.quantile(
        expected[positions], torch.tensor([0, 0.1, 0.5, 0.9, 1], dtype=torch.float64)
    ).tolist()
    assert list(sample["quantiles"].values()) == pytest.approx(expected_quantiles)


def test_actual_hooks_capture_input_feature_halves(audit, torch):
    model, payload = _fixture(torch)
    report = audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)
    layer = report["layers"][0]["tensors"]
    assert layer["input_abs_bh"]["observed_elements"] == 24
    assert layer["input_squared_bh"]["observed_elements"] == 24
    assert layer["linear_0_input"]["observed_elements"] == 48
    assert layer["input_abs_bh"]["all_element_moments"]["min"] >= 0
    abs_norm = layer["input_abs_bh"]["all_element_moments"]["l2_norm"]
    squared_mean = layer["input_squared_bh"]["all_element_moments"]["mean"]
    assert abs_norm**2 / 24 == pytest.approx(squared_mean, rel=1e-6)


def test_audit_does_not_construct_any_optimizer(audit, torch, monkeypatch):
    model, payload = _fixture(torch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("gradient audit must not instantiate or step an optimizer")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    audit.audit_gate_gradients(model, payload, "cpu", weight_decay=0.0005)


@pytest.mark.parametrize(
    "controls",
    [
        {"mode": "bad"},
        {"ppi_batches": 0},
        {"ppi_batch_size": 0},
        {"sample_limit": 0},
        {"near_zero": -1},
        {"near_zero": float("nan")},
        {"weight_decay": -1},
    ],
)
def test_invalid_controls_fail_without_changes(audit, torch, controls):
    model, payload = _fixture(torch)
    before = _snapshot(torch, model)
    kwargs = {"weight_decay": 0.0005, **controls}
    with pytest.raises(ValueError):
        audit.audit_gate_gradients(model, payload, "cpu", **kwargs)
    _assert_unchanged(torch, model, before)
