"""Frozen gain probes with explicit normalization-policy and energy metadata."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..classification.evaluation import (
    _metrics as _reference_metrics, _branches as _reference_branches,
    intervention_variants,
)
from ...wedge_propagation.classification.evaluation import model_state_hash
from .common import condition_metadata, parse_condition
from .model import SCOPES


def intervention_scopes(condition):
    return () if parse_condition(condition)[3] == "off" else tuple(SCOPES)


def _metadata(model):
    return {**condition_metadata(model.condition), "energy_operator": "applied_S_G"}


def _metrics(logits, graph, model, extra):
    return _reference_metrics(logits, graph, model, {**_metadata(model), **extra})


def _branches(details, graph, model, extra):
    if any(detail.get("energy_operator") != "applied_S_G" for detail in details):
        raise ValueError("frozen diagnostics do not describe the actually applied S/G operators")
    rows = _reference_branches(details, graph, model, {**_metadata(model), **extra})
    for row in rows:
        row["projected_nonzero"] = bool(row["projected_nonzero"])
        if not row["projected_nonzero"]:
            row["formula_relative_error"] = None
    return rows


def frozen_evaluate(model, graph, seeds: Sequence[int], config: dict):
    if tuple(seeds) != model.seeds or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must match packed model seed order")
    if model.variant != "off" and tuple(config["evaluation"]["intervention_scopes"]) != tuple(SCOPES):
        raise ValueError("frozen gain evaluation requires all three layer scopes")
    before = model_state_hash(model)
    was_training = model.training
    metrics, interventions, branches = [], [], []
    try:
        model.eval()
        with torch.no_grad():
            logits, details = model(graph, diagnostics=True)
            metrics += _metrics(logits, graph, model, {"model_state_sha256": before})
            branches += _branches(details, graph, model, {
                "intervention": "original", "target": "none", "model_state_sha256": before,
                "no_op_expected": False,
            })
            for intervention in intervention_variants(model.variant):
                for target, layers in SCOPES.items():
                    changed, details = model(
                        graph, diagnostics=True, intervention=intervention,
                        intervention_layers=layers,
                    )
                    extra = {
                        "intervention": intervention, "target": target,
                        "model_state_sha256": before,
                        "no_op_expected": model.variant == "fixed" and intervention == "gain1",
                    }
                    interventions += _metrics(changed, graph, model, extra)
                    branches += _branches(details, graph, model, extra)
    finally:
        model.train(was_training)
    after = model_state_hash(model)
    if before != after:
        raise RuntimeError("frozen normalization evaluation changed trained parameters or buffers")
    return {
        "metric_rows": metrics, "intervention_rows": interventions, "branch_rows": branches,
        "provenance": {
            "dataset": graph.name, "condition": model.condition, **_metadata(model),
            "seeds": list(seeds), "before_sha256": before, "after_sha256": after,
            "parameters_preserved": True, "optimizer_updates": 0,
        },
    }
