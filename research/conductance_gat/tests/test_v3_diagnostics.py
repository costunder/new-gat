"""Small mathematical fixtures, not public-data or CPU research experiments."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.conductance_gat.ablation.model import state_sha256
from research.conductance_gat.v3.diagnostics import (
    ForwardObservation,
    Intervention,
    best_checkpoint_interventions,
    changed_prediction_fraction,
    evaluate_validation,
    moments,
)
from research.conductance_gat.v3.model import RelativeCNodeClassifier
from research.conductance_gat.v3.train import make_optimizer


def graph_fixture():
    return SimpleNamespace(
        x=torch.tensor([[0.5, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 1.0], [3.0, 1.0, 2.0]]),
        y=torch.tensor([0, 1, 0, 999999]),
        incidence_edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]]),
    )


def model_fixture(mode="relative"):
    torch.manual_seed(0)
    return RelativeCNodeClassifier(
        3, 2, hidden_channels=8, layers=2, dropout=0.5, gate_mode=mode, edge_chunk_size=2
    )


def test_observations_preserve_forward_rng_gradients_and_adamw_step():
    graph, observed = graph_fixture(), model_fixture()
    reference = deepcopy(observed)
    opt_a, opt_b = make_optimizer(observed, "relative_c"), make_optimizer(reference, "relative_c")
    rng = torch.get_rng_state()
    with ForwardObservation(observed) as observation:
        actual = observed(graph)
    rng_after = torch.get_rng_state()
    F.cross_entropy(actual[:2], graph.y[:2]).backward()
    rows = observation.summary(gradients=True)
    torch.set_rng_state(rng)
    expected = reference(graph)
    assert torch.equal(torch.get_rng_state(), rng_after)
    F.cross_entropy(expected[:2], graph.y[:2]).backward()
    assert torch.equal(actual, expected)
    for left, right in zip(observed.parameters(), reference.parameters(), strict=True):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad)
    opt_a.step()
    opt_b.step()
    assert state_sha256(observed) == state_sha256(reference)
    assert len(rows) == 2
    for row in rows:
        assert "rho" not in row
        assert row["alpha"] == 0.5 and row["gamma"] == 0.5 and row["tau"] == 1
        assert row["conductance"]["mean"] == 1 and row["conductance"]["cv"] == 0
        assert row["weighted_degree"]["quantiles"]["p50"] == 2
        assert row["weighted_degree"]["max_over_median"] == 1.5
        assert row["gate_gradient_norm"] is not None


def test_interventions_recompute_normalization_restore_modes_gradients_and_rng():
    graph, model = graph_fixture(), model_fixture()
    with torch.no_grad():
        for operator in model.operators:
            operator.estimator.network[-1].weight.normal_(std=0.5)
    model.train()
    model.decoder.eval()
    model.operators[0].raw_alpha.grad = torch.ones(())
    before, modes, rng = (
        state_sha256(model),
        [m.training for m in model.modules()],
        torch.get_rng_state(),
    )
    original, reference = evaluate_validation(model, graph, torch.tensor([0, 1, 2]))
    result = best_checkpoint_interventions(
        model, graph, torch.tensor([0, 1, 2]), original, reference, seed=123
    )
    assert result["status"] == "passed" and len(result["rows"]) == 4
    assert state_sha256(model) == before
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.operators[0].raw_alpha.grad.item() == 1
    assert all(not op._forward_hooks and not op.estimator._forward_hooks for op in model.operators)
    rows = {row["intervention"]: row for row in result["rows"]}
    assert rows["mean_c"]["validation"] == rows["ones_c"]["validation"]
    assert rows["mean_c"]["logit_mean_absolute_delta"] == pytest.approx(
        rows["ones_c"]["logit_mean_absolute_delta"], abs=1e-6
    )
    assert rows["ones_c"]["logit_mean_absolute_delta"] > 0
    repeated = best_checkpoint_interventions(
        model, graph, torch.tensor([0, 1, 2]), original, reference, seed=123
    )
    assert repeated == result
    with Intervention(model, "propagation_off", 0):
        _, off = evaluate_validation(model, graph, torch.tensor([0, 1, 2]), observe=False)
    with torch.no_grad():
        state = F.elu(model.encoder(graph.x))
        for layer_norm in model.norms:
            state = F.elu(layer_norm(state))
        expected = model.decoder(state)[:3]
    torch.testing.assert_close(off, expected)


def test_graph_local_shuffle_and_mean_do_not_cross_graphs():
    model = model_fixture()
    c = torch.tensor([1.0, 3.0, 10.0, 20.0])
    incidence = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
    inputs = (torch.zeros(6, 8), incidence, torch.tensor([0, 0, 0, 1, 1, 1]), 2)
    averaged = Intervention(model, "mean_c", 0).replace(inputs, c, 0)
    assert torch.equal(averaged, torch.tensor([2.0, 2.0, 15.0, 15.0]))
    shuffled = Intervention(model, "shuffled_c", 8).replace(inputs, c, 0)
    assert set(shuffled[:2].tolist()) == {1.0, 3.0}
    assert set(shuffled[2:].tolist()) == {10.0, 20.0}


def test_hooks_and_modes_restore_when_forward_raises(monkeypatch):
    graph, model = graph_fixture(), model_fixture()
    modes = [m.training for m in model.modules()]

    def fail(*args):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(model.operators[1], "forward", fail)
    with pytest.raises(RuntimeError, match="fixture failure"), Intervention(model, "mean_c", 1):
        evaluate_validation(model, graph, torch.tensor([0]))
    assert modes == [m.training for m in model.modules()]
    assert all(not op._forward_hooks and not op.estimator._forward_hooks for op in model.operators)


def test_empty_and_nonfinite_statistics():
    assert moments(torch.empty(0))["mean"] is None
    with pytest.raises(FloatingPointError):
        moments(torch.tensor([float("nan")]))


def test_ppi_changed_predictions_are_labelwise_threshold_decisions():
    reference = torch.tensor([[2.0, 0.1], [1.0, -1.0]])
    logits = torch.tensor([[2.0, -0.1], [1.0, -1.0]])
    assert changed_prediction_fraction(logits, reference, "accuracy") == 0.0
    assert changed_prediction_fraction(logits, reference, "micro_f1") == 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG check needs real CUDA")
def test_gpu_audit_preserves_cuda_rng():
    graph, model = graph_fixture(), model_fixture().cuda()
    graph = SimpleNamespace(**{key: value.cuda() for key, value in vars(graph).items()})
    indices = torch.tensor([0, 1, 2], device="cuda")
    before = torch.cuda.get_rng_state()
    original, reference = evaluate_validation(model, graph, indices)
    best_checkpoint_interventions(model, graph, indices, original, reference, seed=17)
    assert torch.equal(before, torch.cuda.get_rng_state())
