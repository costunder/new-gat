"""CPU-only mathematical contracts for Conductance V4; no dataset or training run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from research.conductance_gat.ablation.model import state_sha256  # noqa: E402
from research.conductance_gat.v3.model import RelativeCNodeClassifier  # noqa: E402
from research.conductance_gat.v4 import train as train_module  # noqa: E402
from research.conductance_gat.v4.model import (  # noqa: E402
    RelativeCSpatialConv,
    RelativeCSpatialNodeClassifier,
)
from research.conductance_gat.v4.operator import symmetric_spatial_propagation  # noqa: E402
from research.conductance_gat.v4.protocol import CONDITIONS  # noqa: E402
from research.conductance_gat.v4.train import _validate_args, topology_metadata  # noqa: E402


def _dense_reference(residual, message, c, incidence, alpha):
    nodes = residual.shape[0]
    tail, head = incidence
    adjacency = c.new_zeros(nodes, nodes)
    adjacency = adjacency.index_put((tail, head), c, accumulate=True)
    adjacency = adjacency.index_put((head, tail), c, accumulate=True)
    degree = adjacency.sum(dim=1)
    active = degree > 0
    inverse = torch.where(active, degree, torch.ones_like(degree)).rsqrt()
    inverse = inverse * active.to(c.dtype)
    propagation = inverse[:, None] * adjacency * inverse[None, :]
    return residual - alpha * (active[:, None] * residual - propagation @ message)


def _operator_inputs():
    residual = torch.tensor(
        [[0.2, -1.1], [1.7, 0.4], [-0.6, 2.2], [3.0, -0.5]],
        dtype=torch.float64,
    )
    message = torch.tensor(
        [[1.1, 0.3], [-0.7, 2.4], [0.8, -1.6], [-2.0, 0.9]],
        dtype=torch.float64,
    )
    conductance = torch.tensor([0.4, 1.3], dtype=torch.float64)
    incidence = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    alpha = torch.tensor(0.37, dtype=torch.float64)
    return residual, message, conductance, incidence, alpha


@pytest.mark.parametrize("chunk_size", [1, 2, 17])
def test_chunked_operator_matches_dense_reference_and_preserves_isolate(chunk_size):
    residual, message, conductance, incidence, alpha = _operator_inputs()
    actual = symmetric_spatial_propagation(
        residual,
        message,
        conductance,
        incidence,
        alpha,
        edge_chunk_size=chunk_size,
    )
    expected = _dense_reference(residual, message, conductance, incidence, alpha)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual[3], residual[3], rtol=0, atol=0)


def test_custom_backward_matches_dense_autograd_for_both_states_c_and_alpha():
    values = _operator_inputs()
    actual_inputs = [value.clone().requires_grad_(True) for value in values[:3]]
    actual_alpha = values[4].clone().requires_grad_(True)
    actual = symmetric_spatial_propagation(
        *actual_inputs,
        values[3],
        actual_alpha,
        edge_chunk_size=1,
    )

    reference_inputs = [value.clone().requires_grad_(True) for value in values[:3]]
    reference_alpha = values[4].clone().requires_grad_(True)
    expected = _dense_reference(
        *reference_inputs,
        values[3],
        reference_alpha,
    )
    upstream = torch.tensor(
        [[0.3, -0.9], [1.1, 0.2], [-0.5, 0.8], [0.7, -1.4]],
        dtype=torch.float64,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (*actual_inputs, actual_alpha),
        grad_outputs=upstream,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (*reference_inputs, reference_alpha),
        grad_outputs=upstream,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2e-10,
            atol=2e-10,
        )


def test_custom_operator_passes_first_order_gradcheck():
    residual, message, conductance, incidence, alpha = _operator_inputs()
    differentiable = tuple(
        value.clone().requires_grad_(True) for value in (residual, message, conductance, alpha)
    )

    def function(residual_state, message_state, c, mixing):
        return symmetric_spatial_propagation(
            residual_state,
            message_state,
            c,
            incidence,
            mixing,
            edge_chunk_size=1,
        )

    assert torch.autograd.gradcheck(function, differentiable, eps=1e-6, atol=1e-5, rtol=1e-4)


def _model(condition):
    specification = CONDITIONS[condition]
    return RelativeCSpatialNodeClassifier(
        3,
        2,
        hidden_channels=4,
        layers=2,
        dropout=0.0,
        gate_mode=specification["gate_mode"],
        spatial_mode=specification["spatial_mode"],
        edge_chunk_size=1,
    )


def test_all_factorial_arms_have_identical_initial_state_and_expected_freezing():
    models = {}
    for condition in CONDITIONS:
        torch.manual_seed(71)
        models[condition] = _model(condition)
    assert len({state_sha256(model) for model in models.values()}) == 1
    for condition, model in models.items():
        specification = CONDITIONS[condition]
        for operator in model.operators:
            assert operator.raw_alpha.requires_grad
            estimator_active = any(
                parameter.requires_grad for parameter in operator.estimator.parameters()
            )
            assert estimator_active == (specification["gate_mode"] == "relative")
            assert operator.message_transform.weight.requires_grad == (
                specification["spatial_mode"] == "learned"
            )
            torch.testing.assert_close(
                operator.message_transform.weight,
                torch.eye(4),
                rtol=0,
                atol=0,
            )


def test_identity_w_path_is_exact_v3_forward():
    torch.manual_seed(29)
    v3 = RelativeCNodeClassifier(
        3,
        2,
        hidden_channels=4,
        layers=2,
        dropout=0.0,
        gate_mode="relative",
        edge_chunk_size=1,
    )
    torch.manual_seed(29)
    v4 = RelativeCSpatialNodeClassifier(
        3,
        2,
        hidden_channels=4,
        layers=2,
        dropout=0.0,
        gate_mode="relative",
        spatial_mode="fixed_identity",
        edge_chunk_size=1,
    )
    v4_state = v4.state_dict()
    for name, value in v3.state_dict().items():
        torch.testing.assert_close(v4_state[name], value, rtol=0, atol=0)
    graph = SimpleNamespace(
        x=torch.tensor([[0.2, 1.0, -0.4], [1.2, -0.7, 0.5], [-0.3, 0.8, 2.0], [0.9, 0.1, -1.0]]),
        incidence_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    v3.eval()
    v4.eval()
    with torch.no_grad():
        expected = v3(graph)
        actual = v4(graph)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_packed_inductive_graphs_match_separate_graph_forwards():
    torch.manual_seed(43)
    model = RelativeCSpatialNodeClassifier(
        3,
        2,
        hidden_channels=4,
        layers=2,
        dropout=0.0,
        gate_mode="relative",
        spatial_mode="learned",
        edge_chunk_size=1,
    ).eval()
    first = SimpleNamespace(
        x=torch.tensor([[0.2, 1.0, -0.4], [1.2, -0.7, 0.5], [-0.3, 0.8, 2.0]]),
        incidence_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    second = SimpleNamespace(
        x=torch.tensor([[0.9, 0.1, -1.0], [0.4, -0.2, 0.7], [1.1, 0.3, -0.5]]),
        incidence_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    packed = SimpleNamespace(
        x=torch.cat((first.x, second.x)),
        incidence_edge_index=torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
    )
    with torch.no_grad():
        expected = torch.cat((model(first), model(second)))
        actual = model(packed)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_ppi_topology_and_child_batch_defaults_bind_official_protocol():
    graphs = [
        {
            "x": torch.ones(3, 2),
            "incidence_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        }
        for _ in range(24)
    ]
    payload = {
        "dataset": "ppi",
        "graphs": graphs,
        "splits": {
            "train": list(range(20)),
            "validation": [20, 21],
            "test": [22, 23],
        },
    }
    topology = topology_metadata(payload)
    assert topology["scope"] == "official_train_and_validation_graphs"
    assert topology["split_graph_counts"] == {"train": 20, "validation": 2}
    assert topology["split_num_nodes"] == {"train": 60, "validation": 6}
    assert topology["split_num_edges"] == {"train": 40, "validation": 4}
    assert all(len(value) == 64 for value in topology["split_incidence_sha256"].values())

    args = SimpleNamespace(
        dataset="ppi",
        condition="fixed_c_identity_w",
        epochs=1,
        patience=1,
        edge_chunk_size=1,
        model_seed=0,
        batch_size=None,
        workers=0,
    )
    _validate_args(args)
    assert args.batch_size == 2


class _PackedBatch(SimpleNamespace):
    def to(self, device, non_blocking=False):
        del non_blocking
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


class _NoTestSplits(dict):
    def __getitem__(self, key):
        if key == "test":
            pytest.fail("V4 training must not read the PPI test split")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "test":
            pytest.fail("V4 training must not read the PPI test split")
        return super().get(key, default)


def test_ppi_training_uses_ten_minibatch_steps_and_never_reads_test(
    tmp_path, monkeypatch
):
    individual = {
        "x": torch.ones(3, 2),
        "y": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "incidence_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }
    payload = {
        "dataset": "ppi",
        "classes": 2,
        "graphs": [
            {name: value.clone() for name, value in individual.items()} for _ in range(24)
        ],
        "splits": _NoTestSplits(
            train=list(range(20)),
            validation=[20, 21],
        ),
    }
    packed = _PackedBatch(
        x=torch.cat((individual["x"], individual["x"])),
        y=torch.cat((individual["y"], individual["y"])),
        incidence_edge_index=torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        num_graphs=2,
    )
    loaders = {"train": [packed] * 10, "validation": [packed]}
    monkeypatch.setattr(train_module, "_require_cuda", lambda device: None)
    monkeypatch.setattr(train_module, "_make_data", lambda payload, args, device: (loaders, None))
    monkeypatch.setattr(train_module, "_source_hashes", lambda: {"unit.py": "a" * 64})
    monkeypatch.setattr(train_module, "_versions", lambda: {"torch": "unit"})
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "unit-cpu")
    steps = 0
    real_step = torch.optim.AdamW.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal steps
        steps += 1
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    output = tmp_path / "ppi-arm"
    output.mkdir()
    args = SimpleNamespace(
        dataset="ppi",
        condition="fixed_c_identity_w",
        model_seed=0,
        epochs=1,
        patience=1,
        batch_size=2,
        workers=0,
        device="cpu",
        edge_chunk_size=16,
    )
    protocol = {
        "data_sha256": "b" * 64,
        "split": "official_inductive_graph_split",
        "split_counts": {"train": 20, "validation": 2, "test": 2},
    }
    result = train_module.train_model(payload, protocol, args, torch.device("cpu"), output)
    assert steps == 10
    assert result["optimizer_steps"] == 10
    assert result["best_checkpoint_optimizer_steps"] == 10
    assert result["train_batches_per_epoch"] == 10
    assert result["validation_batches"] == 1 and result["validation_graphs"] == 2
    assert result["metric_name"] == "micro_f1"
    assert result["execution"]["training"] == "official_inductive_graph_minibatch"


def test_conductance_reads_pre_w_state_and_learned_w_receives_task_gradient():
    operator = RelativeCSpatialConv(
        2,
        gate_mode="fixed_one",
        spatial_mode="learned",
        edge_chunk_size=1,
    ).double()
    state = torch.tensor(
        [[1.0, -2.0], [0.5, 3.0], [-1.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    incidence = torch.tensor([[0], [1]], dtype=torch.long)
    node_graph = torch.zeros(3, dtype=torch.long)
    observed = []
    handle = operator.estimator.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        operator.message_transform.weight.copy_(2 * torch.eye(2, dtype=torch.float64))
    try:
        output = operator(state, incidence, node_graph, 1)
    finally:
        handle.remove()
    torch.testing.assert_close(observed[0], state.detach(), rtol=0, atol=0)
    torch.testing.assert_close(output[2], state[2], rtol=0, atol=0)
    output.square().sum().backward()
    gradient = operator.message_transform.weight.grad
    assert gradient is not None and bool((gradient.abs() > 0).any())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate_mode": "unknown", "spatial_mode": "learned"},
        {"gate_mode": "relative", "spatial_mode": "unknown"},
        {"gate_mode": "relative", "spatial_mode": "learned", "edge_chunk_size": 0},
    ],
)
def test_invalid_v4_modes_and_chunk_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RelativeCSpatialConv(2, **kwargs)
