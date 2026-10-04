"""Frozen actual E/J interventions and complete seed/split/layer coverage."""

from __future__ import annotations

import pytest
import torch
from test_local_placement_training import debug_config, debug_graph

from research.local_energy_relations.placement.evaluation import (
    frozen_evaluate,
    intervention_scopes,
    intervention_variants,
)
from research.local_energy_relations.placement.training import _epoch, make_model, make_optimizer


@pytest.mark.parametrize(
    "variant,interventions", [("base", 0), ("within", 1), ("between", 1), ("both", 3)]
)
@pytest.mark.parametrize("placement,scopes", [("hidden", 1), ("output", 1), ("all", 3)])
def test_frozen_evaluation_is_immutable_and_covers_all_split_layer_treatments(
    variant, interventions, placement, scopes
):
    config, graph = debug_config(), debug_graph()
    condition = "local_degree__base" if variant == "base" else f"local_degree__{variant}__{placement}"
    model = make_model(graph, condition, [11, 23], config, 2)
    optimizer = make_optimizer(model, config, 0.01)
    _epoch(model, graph, optimizer, 0)
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = frozen_evaluate(model, graph, [11, 23], config)
    assert model.training
    assert len(result["metric_rows"]) == 6
    assert len(result["intervention_rows"]) == interventions * scopes * 3 * 2
    assert len(result["branch_rows"]) == 2 * 2 * (1 + interventions * scopes)
    assert result["provenance"]["optimizer_updates"] == 0
    assert result["provenance"]["before_sha256"] == result["provenance"]["after_sha256"]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(row["num_nodes"] == graph.num_nodes for row in result["metric_rows"])
    assert {row["placement"] for row in result["metric_rows"]} == {model.placement}
    assert {row["target"] for row in result["intervention_rows"]} == (set(intervention_scopes(condition)) if interventions else set())


def test_nonzero_lifts_make_frozen_branch_removals_affect_real_logits():
    config, graph = debug_config(), debug_graph()
    model = make_model(graph, "unit__both__output", [11, 23], config, 2)
    with torch.no_grad():
        for parameter in model.energy_lifts.values():
            parameter.fill_(0.4)
        for parameter in model.relation_lifts.values():
            parameter.fill_(0.7)
    model.eval()
    original, _ = model(graph)
    within, within_info = model(graph, intervention="within_remove", intervention_layers=(1,))
    between, between_info = model(graph, intervention="between_remove", intervention_layers=(1,))
    both, both_info = model(graph, intervention="both_remove", intervention_layers=(1,))
    assert not torch.allclose(original, within)
    assert not torch.allclose(original, between)
    torch.testing.assert_close(both, within + between - original, atol=2e-7, rtol=2e-6)
    assert torch.equal(within_info[1]["energy_branch_norm"], torch.zeros(2))
    assert torch.equal(between_info[1]["relation_branch_norm"], torch.zeros(2))
    assert torch.equal(both_info[1]["energy_branch_norm"], torch.zeros(2))


def test_frozen_eval_requires_same_packed_seed_order_and_restores_eval_mode():
    config, graph = debug_config(), debug_graph()
    model = make_model(graph, "unit__base", [11, 23], config, 2)
    with pytest.raises(ValueError, match="seed"):
        frozen_evaluate(model, graph, [23, 11], config)
    model.eval()
    frozen_evaluate(model, graph, [11, 23], config)
    assert not model.training


def test_disabled_branches_have_no_frozen_removal_treatments():
    assert intervention_variants("base") == ()
    assert intervention_variants("within") == ("within_remove",)
    assert intervention_variants("between") == ("between_remove",)


@pytest.mark.parametrize("condition,expected", [
    ("unit__base", ()),
    ("unit__both__hidden", ("layer_0",)),
    ("unit__both__output", ("layer_1",)),
    ("unit__both__all", ("layer_0", "layer_1", "both")),
])
def test_no_frozen_no_op_layer_scopes(condition, expected):
    assert intervention_scopes(condition) == expected
