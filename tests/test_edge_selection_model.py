"""Synthetic CPU/debug checks only; no final-data, GPU or SOTA claims."""

import copy
import io
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.conductance_gat.edge_selection.model import EdgeSelectionClassifier
from research.conductance_gat.edge_selection.selection import (
    CONDITIONS,
    _ConstrainedLogistic,
    exact_budget_gate,
    hard_concrete,
    selection_configuration,
)
from research.conductance_gat.edge_selection.topology import build_topology
from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier


def _graph():
    generator = torch.Generator().manual_seed(15)
    incidence = torch.tensor([[0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6], [1, 2, 3, 2, 3, 3, 5, 7, 6, 7, 7]])
    batch = torch.tensor([0] * 4 + [1] * 4 + [2])
    return SimpleNamespace(
        x=torch.randn(9, 5, generator=generator),
        y=torch.arange(9) % 3,
        incidence_edge_index=incidence,
        batch=batch,
        _v5_num_graphs=3,
        edge_selection_topology=build_topology(9, incidence, batch, forest_seed=37),
    )


def _model(condition, *, checkpoint=True, dropout=0.0, **selection):
    return EdgeSelectionClassifier(
        5,
        3,
        hidden_channels=16,
        heads=4,
        layers=2,
        dropout=dropout,
        edge_chunk_size=3,
        activation_checkpoint=checkpoint,
        selection_config={
            "condition": condition,
            "chord_fraction": 0.5,
            "selection_seed": 76,
            **selection,
        },
    )


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("training", [False, True])
def test_exact_zeros_positive_r_protected_forest_and_budget(condition, training):
    torch.manual_seed(88)
    model, graph = _model(condition), _graph()
    model.train(training)
    output = model(graph)
    assert output.shape == (9, 3) and torch.isfinite(output).all()
    plan = graph.edge_selection_topology
    for operator in model.operators:
        gate, amplitude, effective = operator.last_gate, operator.last_r, operator.last_effective_c
        assert gate.shape == (11,) and amplitude.shape == (11, 4)
        assert (amplitude > 0).all() and (effective >= 0).all()
        torch.testing.assert_close(effective, amplitude * gate[:, None], rtol=0, atol=0)
        assert (effective[gate == 0] == 0).all()
        if condition != "hard_concrete":
            assert ((gate == 0) | (gate == 1)).all()
            assert (gate[plan.forest_mask] == 1).all()
            if condition in {"forest_random", "forest_learned", "forest_cycle"}:
                counts = torch.bincount(plan.edge_graph[~plan.forest_mask], minlength=3)
                selected = torch.bincount(
                    plan.edge_graph[(gate > 0) & ~plan.forest_mask], minlength=3
                )
                torch.testing.assert_close(selected, (counts.double() * 0.5).floor().long())
            if condition == "forest_only":
                torch.testing.assert_close(gate, plan.forest_mask.float())


@pytest.mark.parametrize("condition", ["forest_learned", "forest_cycle", "hard_concrete"])
def test_task_loss_and_auxiliary_update_selector_amplitude_w_and_beta(condition):
    torch.manual_seed(93)
    model, graph = _model(condition), _graph()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss = F.cross_entropy(model(graph), graph.y)
    target = (torch.arange(11) % 3 != 0).float() if condition == "hard_concrete" else None
    auxiliary = model.auxiliary_loss(target)
    loss = loss + 0.01 * auxiliary["l0"] + auxiliary["negative"]
    loss.backward()
    for name, value in model.named_parameters():
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name
    optimizer.step()
    changed = {
        name for name, value in model.named_parameters() if not torch.equal(before[name], value)
    }
    assert any("selector.node_projection" in name for name in changed)
    assert any("selector.edge_hidden" in name for name in changed)
    assert any("selector.score" in name for name in changed)
    assert any("estimator.context_metric" in name for name in changed)
    assert any("value_weight" in name for name in changed)
    assert any("beta_estimator" in name for name in changed)
    if condition == "forest_cycle":
        assert any("cycle_weight" in name for name in changed)
    model.clear_auxiliary_cache()
    assert all(
        op.selector.live_probability is None and op.selector.live_logits is None
        for op in model.operators
    )


@pytest.mark.parametrize("condition", ["full", "forest_only", "forest_random"])
def test_controls_have_no_unused_selector_parameters(condition):
    model = _model(condition)
    assert all(list(op.selector.parameters()) == [] for op in model.operators)
    loss = F.cross_entropy(model(_graph()), _graph().y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_all_arms_preserve_common_backbone_and_amplitude_initialization():
    models = []
    for condition in CONDITIONS:
        torch.manual_seed(24)
        models.append(_model(condition))
    expected = models[0].state_dict()
    for model in models[1:]:
        for name, value in expected.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    learned = models[CONDITIONS.index("forest_learned")].state_dict()
    cycle = models[CONDITIONS.index("forest_cycle")].state_dict()
    for name in learned:
        torch.testing.assert_close(learned[name], cycle[name], rtol=0, atol=0)


def test_full_condition_matches_original_positive_perhead_row_v5_exactly():
    torch.manual_seed(24)
    selection = _model("full").eval()
    torch.manual_seed(24)
    original = GraphConditionedConductanceNodeClassifier(
        5,
        3,
        hidden_channels=16,
        heads=4,
        layers=2,
        dropout=0.0,
        edge_chunk_size=3,
        activation_checkpoint=True,
        conductance_heads="per_head",
        propagation_normalization="row",
        solver_cost_scaling="width_scaled",
    ).eval()
    assert selection.state_dict().keys() == original.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, selection.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(original(_graph()), selection(_graph()), rtol=0, atol=0)


def test_fixed_budget_relaxation_has_correct_implicit_gradient_and_group_constraint():
    logits = torch.tensor([-0.8, 0.1, 1.2, -0.2, 0.5, 2.0], dtype=torch.float64, requires_grad=True)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    counts, budget = torch.tensor([3, 3]), torch.tensor([1, 2])

    def probability(value):
        return _ConstrainedLogistic.apply(value, groups, counts, budget, 0.8)

    result = probability(logits)
    sums = torch.zeros(2, dtype=result.dtype).index_add(0, groups, result)
    torch.testing.assert_close(sums, budget.double(), rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(probability, (logits,), fast_mode=True)
    gate, _, actual_budget = exact_budget_gate(
        logits,
        torch.ones(6, dtype=torch.bool),
        groups,
        2,
        0.5,
        temperature=0.8,
        straight_through=True,
    )
    assert ((gate == 0) | (gate == 1)).all()
    assert gate.sum() == 2 and actual_budget.tolist() == [1, 1]
    (gate * torch.arange(6, dtype=gate.dtype)).sum().backward()
    assert logits.grad.abs().sum() > 0
    torch.testing.assert_close(
        torch.zeros(2, dtype=gate.dtype).index_add(0, groups, logits.grad),
        torch.zeros(2, dtype=gate.dtype),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_budget_endpoints_have_exact_counts_without_nan(fraction):
    scores = torch.randn(5, requires_grad=True)
    eligible, groups = torch.tensor([True, True, False, True, False]), torch.tensor([0, 0, 0, 1, 1])
    gate, probability, budget = exact_budget_gate(
        scores, eligible, groups, 2, fraction, temperature=1.0, straight_through=True
    )
    torch.testing.assert_close(gate, eligible.float() * fraction)
    gate.sum().backward()
    assert torch.isfinite(scores.grad).all() and not scores.grad.any()


def test_hard_concrete_exact_zero_l0_probability_and_deterministic_eval():
    logits = torch.tensor([-100.0, -2.0, 0.0, 2.0, 100.0], requires_grad=True)
    gate, p = hard_concrete(logits, training=False)
    assert gate[0] == 0 and gate[-1] == 1
    expected = (logits - (2 / 3) * torch.tensor(0.1 / 1.1).log()).sigmoid()
    torch.testing.assert_close(p, expected)
    torch.testing.assert_close(gate, hard_concrete(logits, training=False)[0], rtol=0, atol=0)
    (gate.sum() + p.sum()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_checkpoint_replay_preserves_stochastic_gates_outputs_gradients_and_aux_loss():
    torch.manual_seed(72)
    plain = _model("hard_concrete", checkpoint=False, dropout=0.2)
    recompute = copy.deepcopy(plain)
    recompute.activation_checkpoint = True
    graph, target = _graph(), (torch.arange(11) % 2).float()
    results = []
    for model in (plain, recompute):
        torch.manual_seed(94)
        output = model(graph)
        before = [op.last_gate.clone() for op in model.operators]
        aux = model.auxiliary_loss(target)
        loss = F.cross_entropy(output, graph.y) + 0.01 * aux["l0"] + aux["negative"]
        loss.backward()
        for gate, op in zip(before, model.operators, strict=True):
            torch.testing.assert_close(gate, op.last_gate, rtol=0, atol=0)
        results.append(
            (output.detach(), {name: p.grad.clone() for name, p in model.named_parameters()})
        )
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-5, atol=1e-6)


def test_auxiliary_source_targets_are_not_used_in_forward_and_cache_clears():
    model, graph = _model("hard_concrete"), _graph()
    model.eval()
    output = model(graph)
    graph.corruption_origin = torch.arange(11) % 2
    torch.testing.assert_close(output, model(graph), rtol=0, atol=0)
    first = model.auxiliary_loss(torch.zeros(11))["negative"]
    second = model.auxiliary_loss(torch.ones(11))["negative"]
    assert first != second
    torch.testing.assert_close(output, model(graph), rtol=0, atol=0)
    model.clear_auxiliary_cache()
    with pytest.raises(RuntimeError, match="current forward"):
        model.auxiliary_loss()
    with pytest.raises(ValueError, match="separate corruption"):
        _model("forest_learned").auxiliary_loss(torch.ones(11))


def test_cycle_context_changes_gate_scores_and_is_orientation_invariant():
    model, graph = _model("forest_cycle"), _graph()
    model.eval()
    model(graph)
    before = [op.last_logits.clone() for op in model.operators]
    with torch.no_grad():
        for op in model.operators:
            op.selector.cycle_weight.fill_(0.7)
    output = model(graph)
    assert any(
        not torch.allclose(old, op.last_logits)
        for old, op in zip(before, model.operators, strict=True)
    )
    changed = copy.deepcopy(graph)
    changed.incidence_edge_index = graph.incidence_edge_index.flip(0)
    changed.edge_selection_topology = build_topology(
        9, changed.incidence_edge_index, changed.batch, forest_seed=37
    )
    torch.testing.assert_close(output, model(changed), rtol=1e-5, atol=1e-6)


def test_no_topology_or_wrong_topology_is_an_explicit_error():
    graph = _graph()
    del graph.edge_selection_topology
    with pytest.raises(ValueError, match="prepare"):
        _model("full")(graph)
    graph = _graph()
    graph.incidence_edge_index = graph.incidence_edge_index.flip(1)
    with pytest.raises(RuntimeError, match="different physical"):
        _model("full")(graph)


def test_checkpoint_state_roundtrip_and_eval_gate_reproduction():
    torch.manual_seed(21)
    model = _model("forest_cycle").eval()
    graph = _graph()
    expected = model(graph)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model("forest_cycle").eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    torch.testing.assert_close(restored(graph), expected, rtol=0, atol=0)
    for first, second in zip(model.operators, restored.operators, strict=True):
        torch.testing.assert_close(first.last_gate, second.last_gate, rtol=0, atol=0)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_edgeless_graph_preserves_finite_full_model(condition):
    graph = SimpleNamespace(
        x=torch.randn(3, 5), incidence_edge_index=torch.empty(2, 0, dtype=torch.long)
    )
    graph.edge_selection_topology = build_topology(3, graph.incidence_edge_index)
    model = _model(condition)
    output = model(graph)
    assert torch.isfinite(output).all()
    if condition == "hard_concrete":
        assert model.auxiliary_loss()["l0"] == 0


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"condition": "forest_learned"},
        {"condition": "full", "chord_fraction": True},
        {"condition": "full", "selection_temperature": 0},
        {"condition": "hard_concrete", "hard_concrete_lower": 0},
        {"condition": "full", "l0_weight": 0.1},
    ],
)
def test_configuration_refuses_missing_budget_or_invalid_semantics(config):
    with pytest.raises(ValueError):
        selection_configuration(config)


@pytest.mark.parametrize("condition", ["forest_learned", "forest_cycle", "hard_concrete"])
def test_bf16_geometry_and_full_backward_remain_finite(condition):
    torch.manual_seed(65)
    model, graph = _model(condition), _graph()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(graph)
        auxiliary = model.auxiliary_loss(
            (torch.arange(11) % 2).float() if condition == "hard_concrete" else None
        )
        loss = (
            F.cross_entropy(output.float(), graph.y)
            + 0.01 * auxiliary["l0"]
            + auxiliary["negative"]
        )
    loss.backward()
    for operator in model.operators:
        assert operator.last_gate.dtype == operator.last_r.dtype == torch.float32
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_hard_concrete_negative_loss_is_on_the_actual_active_probability():
    model, graph = _model("hard_concrete"), _graph()
    model.eval()
    model(graph)
    targets = torch.ones(11)
    actual = model.auxiliary_loss(targets)["negative"]
    references = []
    groups = graph.edge_selection_topology.edge_graph
    for operator in model.operators:
        losses = -operator.selector.live_probability.log()
        sums = torch.zeros(3).index_add(0, groups, losses)
        counts = torch.bincount(groups, minlength=3)
        references.append((sums / counts.clamp_min(1)).mean())
    torch.testing.assert_close(actual, torch.stack(references).mean())


def test_nonfinite_selector_scores_fail_in_eval_instead_of_silently_ranking_nan():
    model, graph = _model("forest_learned"), _graph()
    model.eval()
    with torch.no_grad():
        model.operators[0].selector.score.weight.fill_(torch.nan)
    with pytest.raises(RuntimeError, match="logits must be finite"):
        model(graph)


def test_read_only_gate_and_frozen_amplitude_interventions_restore_state_and_rng():
    torch.manual_seed(82)
    model, graph = _model("forest_cycle").eval(), _graph()
    with torch.no_grad():
        baseline = model(graph)
    gates = [operator.last_gate.clone() for operator in model.operators]
    before = {
        (id(module), name): value
        for module in model.modules()
        for name, value in vars(module).items()
        if name.startswith(("last_", "live_"))
    }
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    with model.gate_intervention("all"):
        altered = model(graph)
        assert not altered.requires_grad
        assert all((operator.last_gate == 1).all() for operator in model.operators)
        torch.rand(3)
    assert not torch.equal(baseline, altered)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    with model.gate_intervention(gates, amplitude_ones=True):
        model(graph)
        for gate, operator in zip(gates, model.operators, strict=True):
            torch.testing.assert_close(operator.last_gate, gate, rtol=0, atol=0)
            assert (operator.last_r == 1).all()
            torch.testing.assert_close(
                operator.last_effective_c, gate[:, None].expand(-1, 4), rtol=0, atol=0
            )
    for module in model.modules():
        for name, value in vars(module).items():
            if name.startswith(("last_", "live_")):
                assert value is before[(id(module), name)]
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
        assert parameter.grad is None
    assert all(operator.amplitude_override is None for operator in model.operators)
    assert all(operator.selector.gate_override is None for operator in model.operators)


def test_random_budget_intervention_matches_control_and_rejects_unconstrained_scope():
    graph = _graph()
    model = _model("forest_learned").eval()
    random = _model("forest_random").eval()
    with torch.no_grad():
        random(graph)
    with model.gate_intervention("random_budget"):
        model(graph)
        for original, reference in zip(model.operators, random.operators, strict=True):
            torch.testing.assert_close(original.last_gate, reference.last_gate, rtol=0, atol=0)
    corruption = _model("hard_concrete").eval()
    with pytest.raises(ValueError, match="protected exact-budget"):
        with corruption.gate_intervention("random_budget"):
            corruption(graph)
    assert all(operator.selector.gate_override is None for operator in corruption.operators)


def test_gate_intervention_exception_restores_caches_rng_and_rejects_training():
    model, graph = _model("forest_learned").eval(), _graph()
    with torch.no_grad():
        model(graph)
    before = [operator.last_gate for operator in model.operators]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="same-device physical"):
        with model.gate_intervention([torch.ones(3), torch.ones(3)]):
            torch.rand(4)
            model(graph)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for operator, gate in zip(model.operators, before, strict=True):
        assert operator.last_gate is gate and operator.selector.gate_override is None
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        with model.gate_intervention("all"):
            model(graph)


@pytest.mark.parametrize("condition", ["forest_random", "forest_learned", "forest_cycle"])
def test_edge_order_permutation_keeps_physical_gates_even_when_learned_scores_tie(condition):
    torch.manual_seed(53)
    graph, model = _graph(), _model(condition).eval()
    with torch.no_grad():
        for operator in model.operators:
            if operator.selector.score is not None:
                operator.selector.score.weight.zero_()
        baseline = model(graph)
    gates = [operator.last_gate.clone() for operator in model.operators]
    order = torch.tensor([8, 2, 4, 0, 10, 7, 3, 6, 1, 9, 5])
    shuffled = copy.deepcopy(graph)
    shuffled.incidence_edge_index = graph.incidence_edge_index[:, order]
    shuffled.edge_selection_topology = build_topology(
        9, shuffled.incidence_edge_index, shuffled.batch, forest_seed=37
    )
    with torch.no_grad():
        actual = model(shuffled)
    torch.testing.assert_close(actual, baseline, rtol=1e-5, atol=1e-6)
    for gate, operator in zip(gates, model.operators, strict=True):
        torch.testing.assert_close(operator.last_gate, gate[order], rtol=0, atol=0)
