"""Frozen actual E/J interventions and complete seed/split/layer coverage."""

from __future__ import annotations

import pytest
import torch
from test_local_prediction_training import debug_config, debug_graph

from research.local_energy_relations.prediction.evaluation import (
    frozen_evaluate,
    intervention_variants,
)
from research.local_energy_relations.prediction.training import _epoch, make_model, make_optimizer


@pytest.mark.parametrize(
    "variant,interventions", [("base", 0), ("within", 1), ("between", 1), ("both", 3)]
)
def test_frozen_evaluation_is_immutable_and_covers_all_split_layer_treatments(
    variant, interventions
):
    config, graph = debug_config(), debug_graph()
    model = make_model(graph, f"local_degree__{variant}", [11, 23], config, 2)
    optimizer = make_optimizer(model, config, 0.01)
    _epoch(model, graph, optimizer, 0)
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = frozen_evaluate(model, graph, [11, 23], config)
    assert model.training
    assert len(result["metric_rows"]) == 6
    assert len(result["intervention_rows"]) == interventions * 3 * 3 * 2
    assert len(result["branch_rows"]) == 2 * 2 * (1 + interventions * 3)
    assert result["provenance"]["optimizer_updates"] == 0
    assert result["provenance"]["before_sha256"] == result["provenance"]["after_sha256"]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(row["num_nodes"] == graph.num_nodes for row in result["metric_rows"])


def test_nonzero_lifts_make_frozen_branch_removals_affect_real_logits():
    config, graph = debug_config(), debug_graph()
    model = make_model(graph, "unit__both", [11, 23], config, 2)
    with torch.no_grad():
        for parameter in model.energy_lifts:
            parameter.fill_(0.4)
        for parameter in model.relation_lifts:
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
