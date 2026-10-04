"""Frozen final-checkpoint metrics and interventions with no optimizer step.

Only this final phase reads test labels/masks. Checkpoint selection happens
before entry. Every treatment recomputes downstream hidden states with the
selected layer gain set to zero or one. Parameter and buffer hashes must remain
unchanged. Diagnostic ratios with zero denominators are exported as null.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ...wedge_propagation.classification.evaluation import (
    classification_metrics, model_state_hash, split_indices,
)
from .model import INTERVENTIONS, SCOPES, parse_condition


def intervention_variants(variant: str) -> tuple[str, ...]:
    if variant not in ("off", "fixed", "learned"):
        raise ValueError("unknown cross variant")
    return () if variant == "off" else INTERVENTIONS


def intervention_scopes(condition: str) -> tuple[str, ...]:
    return () if parse_condition(condition)[1] == "off" else tuple(SCOPES)


def _base(graph, model, seed):
    return {
        "dataset": graph.name, "condition": model.condition,
        "weight_mode": model.weight_mode, "variant": model.variant, "seed": int(seed),
    }


def _metrics(logits, graph, model, extra):
    indices = split_indices(graph)
    columns = []
    for split, attribute in (("train", "train_mask"), ("validation", "val_mask"), ("test", "test_mask")):
        if indices[split].numel() == 0:
            raise ValueError(f"empty final classification split: {split}")
        ce, accuracy = classification_metrics(
            logits, graph.y, getattr(graph, attribute), validate=False, indices=indices[split],
        )
        columns.append((split, ce, accuracy, indices[split].numel()))
    host = torch.stack([value for _, ce, acc, _ in columns for value in (ce, acc)], 1).detach().cpu().numpy()
    rows = []
    for index, seed in enumerate(model.seeds):
        for position, (split, _, _, labeled) in enumerate(columns):
            ce, accuracy = map(float, host[index, 2 * position:2 * position + 2])
            if not math.isfinite(ce) or ce < 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise FloatingPointError("invalid frozen classification metric")
            rows.append({
                **_base(graph, model, seed), **extra, "split": split,
                "ce": ce, "accuracy": accuracy, "num_nodes": graph.x.shape[0],
                "num_labeled_nodes": labeled,
            })
    return rows


def _branches(details, graph, model, extra):
    if len(details) != 2:
        raise ValueError("frozen branch diagnostics must cover both macro layers")
    rows = []
    for layer, detail in enumerate(details):
        keys = [key for key, value in detail.items()
                if isinstance(value, torch.Tensor) and value.shape == (len(model.seeds),)]
        required = {"projected_norm", "output_norm", "off_norm", "matched_delta_norm",
                    "matched_delta_relative", "off_nonzero", "context_norm",
                    "cross_energy_before", "cross_energy_after", "rho", "rho_original"}
        if not required <= set(keys):
            raise ValueError("missing per-seed local coupling diagnostics")
        host = torch.stack([detail[key].detach() for key in keys], 1).cpu().numpy()
        for index, seed in enumerate(model.seeds):
            row = {
                **_base(graph, model, seed), **extra, "layer": layer,
                "parameters_per_seed": model.parameters_per_seed,
                "trainable_parameters_per_seed": model.trainable_parameters_per_seed,
                **{key: float(value) for key, value in zip(keys, host[index], strict=True)},
            }
            if not all(math.isfinite(row[key]) for key in keys):
                raise FloatingPointError("nonfinite frozen coupling diagnostic")
            row["off_nonzero"] = bool(row["off_nonzero"])
            if not row["off_nonzero"]:
                row["matched_delta_relative"] = None
            row["theta_available"] = bool(detail["theta_available"])
            if not row["theta_available"]:
                row["theta"] = None
            rows.append(row)
    return rows


def frozen_evaluate(model, graph, seeds: Sequence[int], config: dict):
    if tuple(seeds) != model.seeds or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must match packed model seed order")
    requested = config["evaluation"]["intervention_scopes"]
    if model.variant != "off" and tuple(requested) != tuple(SCOPES):
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
                for target in intervention_scopes(model.condition):
                    changed, details = model(
                        graph, diagnostics=True, intervention=intervention,
                        intervention_layers=SCOPES[target],
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
        raise RuntimeError("frozen evaluation changed trained model parameters or buffers")
    return {
        "metric_rows": metrics, "intervention_rows": interventions, "branch_rows": branches,
        "provenance": {
            "dataset": graph.name, "condition": model.condition,
            "weight_mode": model.weight_mode, "variant": model.variant,
            "seeds": list(seeds), "before_sha256": before, "after_sha256": after,
            "parameters_preserved": True, "optimizer_updates": 0,
        },
    }
