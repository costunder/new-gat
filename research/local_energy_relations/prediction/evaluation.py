"""Frozen full-graph classification and E/J branch removal.

All final checkpoints are selected before this module is called by the study.
The model is never fitted here. Removed branches are recomputed from the current
hidden state, so subsequent layers follow the intervened forward path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ...wedge_propagation.classification.evaluation import (
    classification_metrics,
    model_state_hash,
    split_indices,
)

SCOPES = {"layer_0": (0,), "layer_1": (1,), "both": (0, 1)}


def intervention_variants(variant: str) -> tuple[str, ...]:
    return {
        "base": (),
        "within": ("within_remove",),
        "between": ("between_remove",),
        "both": ("within_remove", "between_remove", "both_remove"),
    }[variant]


def _base(graph, model, seed):
    return {
        "dataset": graph.name,
        "condition": model.condition,
        "weight_mode": model.weight_mode,
        "variant": model.variant,
        "seed": int(seed),
    }


def _metrics(logits, graph, model, extra):
    indices = split_indices(graph)
    columns = []
    for split, attribute in (
        ("train", "train_mask"),
        ("validation", "val_mask"),
        ("test", "test_mask"),
    ):
        ce, accuracy = classification_metrics(
            logits, graph.y, getattr(graph, attribute), validate=False, indices=indices[split]
        )
        columns.append((split, ce, accuracy, indices[split].numel()))
    values = torch.stack([value for _, ce, accuracy, _ in columns for value in (ce, accuracy)], 1)
    host = values.detach().cpu().numpy()
    rows = []
    for index, seed in enumerate(model.seeds):
        for position, (split, _, _, labeled) in enumerate(columns):
            ce, accuracy = map(float, host[index, 2 * position : 2 * position + 2])
            if not math.isfinite(ce) or ce < 0 or not 0 <= accuracy <= 1:
                raise FloatingPointError("invalid frozen classification metric")
            rows.append(
                {
                    **_base(graph, model, seed),
                    **extra,
                    "split": split,
                    "ce": ce,
                    "accuracy": accuracy,
                    "num_nodes": graph.num_nodes,
                    "num_labeled_nodes": labeled,
                }
            )
    return rows


def _branches(details, graph, model, extra):
    rows = []
    for layer, detail in enumerate(details):
        keys = [
            key
            for key, value in detail.items()
            if isinstance(value, torch.Tensor) and value.shape == (len(model.seeds),)
        ]
        if not keys:
            raise ValueError("missing per-seed branch diagnostics")
        host = torch.stack([detail[key].detach() for key in keys], 1).cpu().numpy()
        for index, seed in enumerate(model.seeds):
            row = {
                **_base(graph, model, seed),
                **extra,
                "layer": layer,
                "parameters_per_seed": model.parameters_per_seed,
                "trainable_parameters_per_seed": model.trainable_parameters_per_seed,
                **{key: float(value) for key, value in zip(keys, host[index], strict=True)},
            }
            if not all(math.isfinite(float(row[key])) for key in keys):
                raise FloatingPointError("nonfinite frozen branch diagnostic")
            row["base_nonzero"] = bool(row["base_nonzero"])
            if row.get("base_nonzero") == 0:
                for key in ("energy_to_base", "relation_to_base"):
                    if key in row:
                        row[key] = None
            rows.append(row)
    if len(details) != 2:
        raise ValueError("frozen diagnostic layer coverage differs from two-layer model")
    return rows


def frozen_evaluate(model, graph, seeds: Sequence[int], config: dict):
    if tuple(seeds) != tuple(model.seeds) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must match packed model seed order")
    before = model_state_hash(model)
    was_training = model.training
    metrics, interventions, branches = [], [], []
    try:
        model.eval()
        with torch.no_grad():
            logits, details = model(graph, diagnostics=True)
            metrics += _metrics(logits, graph, model, {"model_state_sha256": before})
            branches += _branches(
                details,
                graph,
                model,
                {"intervention": "original", "target": "none", "model_state_sha256": before},
            )
            for intervention in intervention_variants(model.variant):
                for target in config["evaluation"]["intervention_scopes"]:
                    changed, details = model(
                        graph,
                        diagnostics=True,
                        intervention=intervention,
                        intervention_layers=SCOPES[target],
                    )
                    extra = {
                        "intervention": intervention,
                        "target": target,
                        "model_state_sha256": before,
                    }
                    interventions += _metrics(changed, graph, model, extra)
                    branches += _branches(details, graph, model, extra)
    finally:
        model.train(was_training)
    after = model_state_hash(model)
    if before != after:
        raise RuntimeError("frozen evaluation changed trained model parameters")
    return {
        "metric_rows": metrics,
        "intervention_rows": interventions,
        "branch_rows": branches,
        "provenance": {
            "dataset": graph.name,
            "condition": model.condition,
            "seeds": list(seeds),
            "before_sha256": before,
            "after_sha256": after,
            "parameters_preserved": True,
            "optimizer_updates": 0,
        },
    }
