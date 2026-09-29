"""Explicit numerical/debug checks, not real-data training results."""

import copy
import math

import pytest
import torch
from torch.nn import functional as F

from experiments.c_learning_bracket import model as accepted_model
from experiments.c_learning_bracket.test_debug import cpu_threads, fixture_batch
from research.conductance_gat.v5.operator import conductance_propagation_coefficients

from .log_row import log_row_coefficients
from .model import make_model
from .test_debug import arguments
from .train import step

__all__ = ["cpu_threads"]


def test_coefficients_and_score_correction_gradients_match_raw_exp():
    edges = torch.tensor([[0, 0, 1, 3], [1, 2, 2, 4]])
    scores = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    correction = torch.linspace(0.3, 2.0, 4, dtype=torch.float64).requires_grad_()
    raw = conductance_propagation_coefficients(
        scores.exp(), edges, 6, sampling_correction=correction, normalization="row"
    )
    stable = log_row_coefficients(scores, edges, 6, correction=correction)
    for a, b in zip(raw[:2], stable[:2], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-14)
    probes = [torch.randn_like(raw[0]), torch.randn_like(raw[1])]
    grad_a = torch.autograd.grad(
        sum((a * p).sum() for a, p in zip(raw[:2], probes, strict=True)), (scores, correction)
    )
    grad_b = torch.autograd.grad(
        sum((a * p).sum() for a, p in zip(stable[:2], probes, strict=True)), (scores, correction)
    )
    for a, b in zip(grad_a, grad_b, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
    assert not stable[2][5].any()  # An actual isolate, not exp underflow.
    assert torch.autograd.gradcheck(
        lambda s, c: log_row_coefficients(s, edges, 6, correction=c)[:2],
        (scores, correction),
        atol=1e-6,
        rtol=1e-4,
    )


@pytest.mark.parametrize("offset", [-1000.0, 1000.0])
def test_extreme_finite_scores_keep_coefficients_and_backward(offset):
    edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
    base = torch.tensor([[0.0, -1.0], [1.0, 0.0], [-1.0, 1.0]])
    scores = (base + offset).requires_grad_()
    correction = torch.tensor([1.0, 2.0, 0.5])
    actual = log_row_coefficients(scores, edges, 4, correction=correction)
    expected = log_row_coefficients(base.double(), edges, 4, correction=correction.double())
    assert not (torch.isfinite(scores.exp()) & (scores.exp() > 0)).all()
    for a, b in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(a.double(), b, rtol=5e-5, atol=5e-5)
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ((actual[0] * weights).sum() + actual[1].sum()).backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().max() > 0


@pytest.mark.parametrize("initialization", ["baseline", "kaiming_relu"])
@pytest.mark.parametrize("condition", ["learned", "fixed"])
@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_model_matches_raw_with_same_rng_weights_and_gradients(
    initialization, condition, checkpoint, dtype, monkeypatch
):
    if dtype == torch.float64:
        # The production classifier deliberately builds FP32 static topology.
        # For this FP64 mathematical reference only, construct that same
        # topology in FP64 for both models; production code remains unchanged.
        original_context = accepted_model._static_graph_context
        monkeypatch.setattr(
            accepted_model,
            "_static_graph_context",
            lambda state, *args, **kwargs: original_context(state.to(dtype), *args, **kwargs),
        )
    batch = fixture_batch()
    batch.graph.x = batch.graph.x.to(dtype)
    payload = {"graphs": [{"x": batch.graph.x}], "classes": 3}
    args = arguments(initialization)
    args.activation_checkpoint = checkpoint
    raw = make_model(payload, args, "cpu", condition).to(dtype)
    rng = torch.get_rng_state()
    args.conductance_evaluation = "log_row"
    stable = make_model(payload, args, "cpu", condition).to(dtype)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in raw.state_dict().items():
        assert torch.equal(value, stable.state_dict()[name])
    torch.manual_seed(120)
    old = raw(batch.graph)
    F.cross_entropy(old, batch.graph.y).backward()
    torch.manual_seed(120)
    new = stable(batch.graph)
    F.cross_entropy(new, batch.graph.y).backward()
    # Equivalent reduction orders need not be bit-identical across eight layers.
    tolerance = 8 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        old, new, rtol=2e-5 if dtype == torch.float32 else 1e-11, atol=tolerance
    )
    for (name, a), (other, b) in zip(
        raw.named_parameters(), stable.named_parameters(), strict=True
    ):
        assert name == other and a.grad is not None and b.grad is not None
        torch.testing.assert_close(
            a.grad, b.grad, rtol=3e-4 if dtype == torch.float32 else 1e-10, atol=tolerance
        )
    stable.eval()
    with torch.no_grad():
        with stable.c_ones():
            intervention = stable(batch.graph)
        fixed = stable.fixed_copy().eval()(batch.graph)
    torch.testing.assert_close(intervention, fixed, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA debug")
        ),
    ],
)
def test_observed_optimizer_step_works_beyond_raw_exp_range(device):
    batch = fixture_batch(device)
    payload = {"graphs": [{"x": batch.graph.x}], "classes": 3}
    args = arguments("baseline")
    args.conductance_evaluation = "log_row"
    model = make_model(payload, args, device)
    with torch.no_grad():
        for op in model.layers:
            op.estimator.query.weight.zero_()
            op.estimator.key.weight.zero_()
            op.estimator.query.bias.fill_(math.sqrt(1000.0))
            op.estimator.key.bias.fill_(math.sqrt(1000.0))
    saved = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    loss, _, report = step(model, batch, optimizer, 2)
    assert torch.isfinite(loss)
    assert report["gradient_coordinate"] == "d(CE)/d(log C); not d(CE)/dC"
    assert len(report["layers"]) == 8
    for row in report["layers"]:
        assert "c" not in row and row["log_c"]["before"]["mean"][0] > 900
        assert row["live_log_c_gradient_norm_and_max_by_head"] is not None
    assert any(not torch.equal(saved[n], v) for n, v in model.state_dict().items())
    invalid = torch.tensor([[float("inf")]], device=device)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        log_row_coefficients(invalid, torch.tensor([[0], [1]], device=device), 2)
    assert torch.ones(1, device=device).sum().item() == 1  # CUDA remains usable.


def test_empty_graph_coefficients_and_gradients():
    scores = torch.empty(0, 8, dtype=torch.float64, requires_grad=True)
    result = log_row_coefficients(scores, torch.empty(2, 0, dtype=torch.long), 3)
    assert not result[2].any()
    (result[0].sum() + result[1].sum()).backward()
    assert scores.grad.shape == (0, 8)
