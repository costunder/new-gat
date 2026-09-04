"""Tensor-level execution equivalence only; no benchmark training or downloads."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.conductance_gat.benchmark import (
    ConductanceConv,
    ConductanceNodeClassifier,
    _binary_counts,
    _micro_f1_from_counts,
    _prepare_split_indices,
    build_parser,
    data_observability,
    optimizer_ownership,
    validate_first_optimizer_step,
)


def _previous_conv_forward(model, state, incidence, node_graph):
    """Literal pre-optimization operator for forward AND parameter-gradient checks."""
    state = state.float()
    tail, head = incidence
    gradient = state[head] - state[tail]
    conductance = model.estimator(gradient, state.new_empty((gradient.shape[0], 0)))
    flux = conductance[:, None] * gradient
    divergence = torch.zeros_like(state)
    divergence.index_add_(0, head, flux)
    divergence.index_add_(0, tail, -flux)
    degree = state.new_zeros(state.shape[0])
    degree.index_add_(0, head, conductance)
    degree.index_add_(0, tail, conductance)
    max_degree = state.new_zeros(int(node_graph.max()) + 1)
    max_degree.scatter_reduce_(0, node_graph, degree, reduce="amax", include_self=True)
    step = 0.95 / max_degree.clamp_min(1e-12)
    return state - step[node_graph, None] * divergence


def test_transductive_observability_reports_full_graph_use_and_label_split_fraction():
    payload = {
        "dataset": "cora",
        "graphs": [
            {
                "x": torch.ones(4, 3),
                "y": torch.arange(4),
                "incidence_edge_index": torch.tensor([[0, 1], [1, 2]]),
            }
        ],
        "splits": {
            "train": torch.tensor([True, False, False, False]),
            "validation": torch.tensor([False, True, True, False]),
            "test": torch.tensor([False, False, False, True]),
        },
    }
    result = data_observability(payload, used_splits=("train", "validation"))
    assert result["full_dataset_count"] == result["actual_used_count"] == 4
    assert result["actual_used_fraction_of_full_dataset"]["value"] == 1.0
    assert result["requested_split_member_count"] == 3
    assert result["requested_split_member_fraction_of_full_dataset"]["value"] == 0.75
    assert result["subset_or_fast_mode"] is False


def test_first_optimizer_step_integrity_is_fail_closed():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(2, 3)).sum().backward()
    result = validate_first_optimizer_step(model, optimizer)
    assert result["status"] == "passed"
    assert result["trainable_parameter_tensors"] == 2

    missing_owner = torch.optim.SGD([model.weight], lr=0.1)
    with pytest.raises(RuntimeError, match="optimizer ownership integrity failed"):
        optimizer_ownership(model, missing_owner)

    model.bias.grad = None
    with pytest.raises(FloatingPointError, match="missing=.*bias"):
        validate_first_optimizer_step(model, optimizer)

    model.bias.grad = torch.full_like(model.bias, float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite=.*bias"):
        validate_first_optimizer_step(model, optimizer)


def test_metadata_graph_count_preserves_forward_and_every_gradient():
    torch.manual_seed(271)
    previous = ConductanceConv(4)
    optimized = copy.deepcopy(previous)
    state = torch.randn(8, 4)
    old_state = state.clone().requires_grad_()
    new_state = state.clone().requires_grad_()
    edges = torch.tensor([[0, 1, 3, 3, 4, 5], [1, 2, 4, 5, 6, 6]])
    node_graph = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2])
    target = torch.randn_like(state)

    old_output = _previous_conv_forward(previous, old_state, edges, node_graph)
    new_output = optimized(new_state, edges, node_graph, num_graphs=3)
    torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)
    (old_output * target).sum().backward()
    (new_output * target).sum().backward()
    torch.testing.assert_close(new_state.grad, old_state.grad, rtol=0, atol=0)
    for old_parameter, new_parameter in zip(
        previous.parameters(), optimized.parameters(), strict=True
    ):
        assert old_parameter.grad is not None and new_parameter.grad is not None
        torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("metadata", ["single", "ptr", "num_graphs"])
def test_classifier_uses_graph_metadata_without_tensor_max(monkeypatch, metadata):
    torch.manual_seed(14)
    graph = SimpleNamespace(
        x=torch.randn(6, 3),
        incidence_edge_index=torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]),
    )
    if metadata != "single":
        graph.batch = torch.tensor([0, 0, 0, 1, 1, 1])
        if metadata == "ptr":
            graph.ptr = torch.tensor([0, 3, 6])
        else:
            graph.num_graphs = 2
    model = ConductanceNodeClassifier(3, 2, hidden_channels=8, layers=2, dropout=0)

    def forbidden_max(*args, **kwargs):
        raise AssertionError("graph count must come from CPU metadata, not a GPU reduction")

    monkeypatch.setattr(torch.Tensor, "max", forbidden_max)
    output = model(graph)
    assert output.shape == (6, 2)
    assert torch.isfinite(output).all()


def test_tensor_only_operator_api_keeps_graph_count_fallback():
    torch.manual_seed(91)
    model = ConductanceConv(2)
    state = torch.randn(4, 2)
    incidence = torch.tensor([[0, 2], [1, 3]])
    node_graph = torch.tensor([0, 0, 1, 1])
    torch.testing.assert_close(
        model(state, incidence, node_graph),
        model(state, incidence, node_graph, num_graphs=2),
        rtol=0,
        atol=0,
    )


def test_precomputed_indices_preserve_masked_loss_and_gradients():
    torch.manual_seed(52)
    masks = {
        "train": torch.tensor([True, False, True, False, True, False]),
        "validation": torch.tensor([False, True, False, False, False, False]),
        "test": torch.tensor([False, False, False, True, False, True]),
    }
    original_masks = {name: mask.clone() for name, mask in masks.items()}
    indices = _prepare_split_indices(masks, torch.device("cpu"))
    assert indices["train"].tolist() == [0, 2, 4]
    assert indices["train"].numel() == 3
    for name, mask in masks.items():
        assert torch.equal(mask, original_masks[name])
        assert indices[name].dtype == torch.long

    logits = torch.randn(6, 3)
    old_logits = logits.clone().requires_grad_()
    new_logits = logits.clone().requires_grad_()
    labels = torch.tensor([0, 2, 1, 1, 2, 0])
    old_loss = F.cross_entropy(old_logits[masks["train"]], labels[masks["train"]])
    new_loss = F.cross_entropy(
        new_logits.index_select(0, indices["train"]),
        labels.index_select(0, indices["train"]),
    )
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_logits.grad, old_logits.grad, rtol=0, atol=0)
    assert torch.count_nonzero(new_logits.grad[~masks["train"]]) == 0


def test_device_count_accumulation_preserves_global_micro_f1():
    logits = [
        torch.tensor([[1.0, -1.0, 2.0]]),
        torch.tensor([[-1.0, 2.0, -1.0], [1.0, -1.0, -1.0]]),
    ]
    labels = [
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
    ]
    counts = torch.zeros(3, dtype=torch.int64)
    for batch_logits, batch_labels in zip(logits, labels, strict=True):
        counts.add_(_binary_counts(batch_logits, batch_labels))
    assert counts.tolist() == [3, 4, 5]
    assert _micro_f1_from_counts(counts) == pytest.approx(2 / 3)
    assert _micro_f1_from_counts(torch.zeros(3, dtype=torch.int64)) == 0.0


def test_device_loss_accumulation_matches_previous_label_weighted_mean():
    losses = [torch.tensor(0.12345679), torch.tensor(1.375), torch.tensor(0.5)]
    label_counts = [121, 363, 242]
    previous = sum(
        float(loss) * count for loss, count in zip(losses, label_counts, strict=True)
    ) / sum(label_counts)
    loss_sum = torch.zeros((), dtype=torch.float64)
    for loss, count in zip(losses, label_counts, strict=True):
        loss_sum.add_(loss.detach().to(torch.float64) * count)
    assert float(loss_sum / sum(label_counts)) == previous
    assert not loss_sum.requires_grad


def test_execution_optimization_is_opt_in_and_amp_default_is_unchanged():
    defaults = build_parser().parse_args([])
    assert defaults.compile is False
    assert defaults.amp is False
    assert build_parser().parse_args(["--compile"]).compile is True
