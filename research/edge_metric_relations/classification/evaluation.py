"""Locked-checkpoint evaluation and condition-specific frozen interventions.

This phase reads all three label masks after validation selection is locked.
Adaptive coefficients are recomputed from the resulting hidden states. It
creates no optimizer and verifies unchanged parameter/buffer hashes.
"""
from __future__ import annotations

import math

import torch

from ...wedge_propagation.classification.evaluation import classification_metrics, model_state_hash, split_indices
from .common import condition_metadata, parse_condition
from .model import SCOPES

SPLITS = (("train", "train_mask"), ("validation", "val_mask"), ("test", "test_mask"))
BRANCH_METRICS = (
    "projected_norm", "output_norm", "off_norm", "matched_delta_norm", "matched_delta_relative",
    "energy_intra", "energy_cross", "energy_total", "message_diag_norm", "message_cross_norm",
    "diagonal_mean", "diagonal_std", "pair_mean", "pair_std",
)


def intervention_variants(variant):
    mapping = {"D0": ("no_op",), "D1": ("no_op", "diagonal_one"),
               "F0": ("no_op", "offdiag_zero"), "F1": ("no_op", "offdiag_zero"),
               "F2": ("no_op", "offdiag_zero", "diagonal_one"),
               "DA": ("no_op", "pair_zero", "diagonal_one"),
               "G1": ("no_op",), "G2": ("no_op",), "P2": ("no_op",)}
    if variant not in mapping:
        raise ValueError("unknown classification variant")
    return mapping[variant]


def intervention_scopes(condition):
    parse_condition(condition)
    return tuple(SCOPES)


def _base(graph, model, seed):
    return {"dataset": graph.name, "condition": model.condition,
            **condition_metadata(model.condition), "seed": int(seed)}


def _metrics(logits, graph, model, extra, reference=None):
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("nonfinite frozen logits")
    indices = split_indices(graph)
    tensors, specifications = [], []
    for split, attribute in SPLITS:
        selected = indices[split]
        if selected.numel() == 0:
            raise ValueError(f"empty final split: {split}")
        ce, accuracy = classification_metrics(logits, graph.y, getattr(graph, attribute), validate=False, indices=selected)
        columns = [ce, accuracy]
        if reference is not None:
            current, original = logits.index_select(1, selected), reference.index_select(1, selected)
            delta = (current - original).square().flatten(1).sum(1).sqrt()
            norm = original.square().flatten(1).sum(1).sqrt()
            columns += [delta, norm, (current.argmax(-1) != original.argmax(-1)).double().mean(1)]
        tensors.extend(columns)
        specifications.append((split, selected.numel(), len(columns)))
    host = torch.stack(tensors, 1).detach().cpu().numpy()
    rows = []
    for index, seed in enumerate(model.seeds):
        offset = 0
        for split, labeled, width in specifications:
            values = list(map(float, host[index, offset:offset + width])); offset += width
            ce, accuracy = values[:2]
            if not all(math.isfinite(v) for v in values) or ce < 0 or not 0 <= accuracy <= 1:
                raise FloatingPointError("invalid frozen metric")
            row = {**_base(graph, model, seed), **extra, "split": split, "ce": ce, "accuracy": accuracy,
                   "num_nodes": graph.x.shape[0], "num_labeled_nodes": int(labeled)}
            if reference is not None:
                delta, norm, flips = values[2:]
                row.update(logit_delta_norm=delta, logit_reference_norm=norm,
                           logit_delta_relative=delta / norm if norm > 0 else None,
                           prediction_flip_fraction=flips)
            rows.append(row)
    return rows


def _branches(details, graph, model, extra):
    if len(details) != 2:
        raise ValueError("frozen diagnostics must cover both layers")
    rows = []
    expected = condition_metadata(model.condition)
    for layer, detail in enumerate(details):
        if any(detail.get(k) != v for k, v in expected.items()):
            raise ValueError("branch condition/operator metadata mismatch")
        available = []
        for key in BRANCH_METRICS:
            value = detail.get(key)
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.shape != (len(model.seeds),):
                    raise ValueError(f"diagnostic must be a per-seed tensor: {key}")
                available.append(key)
        if not {"projected_norm", "output_norm"} <= set(available):
            raise ValueError("required original branch norms are missing")
        if model.operator_family == "edge_metric" and not {
            "off_norm", "matched_delta_norm", "matched_delta_relative", "energy_intra", "energy_cross", "energy_total",
            "message_diag_norm", "message_cross_norm",
        } <= set(available):
            raise ValueError("applied edge metric diagnostics are missing")
        host = torch.stack([detail[k].detach() for k in available], 1).cpu().numpy()
        flag = detail.get("off_nonzero")
        if flag is not None and (not isinstance(flag, torch.Tensor) or flag.shape != (len(model.seeds),)):
            raise ValueError("off_nonzero must be a per-seed tensor")
        host_flags = None if flag is None else flag.detach().cpu().tolist()
        for index, seed in enumerate(model.seeds):
            row = {**_base(graph, model, seed), **extra, "layer": layer,
                   "parameters_per_seed": model.parameters_per_seed,
                   "trainable_parameters_per_seed": model.trainable_parameters_per_seed,
                   **{key: None for key in BRANCH_METRICS}}
            row.update({key: float(value) for key, value in zip(available, host[index], strict=True)})
            if not all(math.isfinite(row[k]) for k in available):
                raise FloatingPointError("nonfinite branch diagnostic")
            row["off_nonzero"] = None if host_flags is None else bool(host_flags[index])
            if row["off_nonzero"] is False:
                row["matched_delta_relative"] = None
            rows.append(row)
    return rows


def frozen_evaluate(model, graph, seeds, config):
    if tuple(seeds) != model.seeds or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must match packed model order")
    if tuple(config["evaluation"]["intervention_scopes"]) != tuple(SCOPES):
        raise ValueError("all three intervention scopes are required")
    before, was_training = model_state_hash(model), model.training
    metrics, probes, branches, noop_residuals = [], [], [], []
    try:
        model.eval()
        with torch.no_grad():
            original, details = model(graph, diagnostics=True)
            metrics.extend(_metrics(original, graph, model, {"model_state_sha256": before}))
            branches.extend(_branches(details, graph, model, {"intervention": "original", "target": "none",
                            "model_state_sha256": before, "no_op_expected": False}))
            for treatment in intervention_variants(model.variant):
                for target, layers in SCOPES.items():
                    changed, details = model(graph, diagnostics=True, intervention=treatment, intervention_layers=layers)
                    noop = treatment == "no_op"
                    if noop:
                        scale = original.abs().amax().clamp_min(1.)
                        error = (changed - original).abs().amax()
                        tolerance = 64 * torch.finfo(original.dtype).eps * scale
                        if not bool(error <= tolerance):
                            raise RuntimeError("declared no-op changed frozen logits beyond roundoff")
                        noop_residuals.append(float(error.cpu()))
                    extra = {"intervention": treatment, "target": target, "model_state_sha256": before,
                             "no_op_expected": noop, "coefficient_policy": "recompute_from_current_hidden"}
                    probes.extend(_metrics(changed, graph, model, extra, original))
                    branches.extend(_branches(details, graph, model, extra))
    finally:
        model.train(was_training)
    after = model_state_hash(model)
    if before != after:
        raise RuntimeError("frozen evaluation modified trained parameters or buffers")
    return {"metric_rows": metrics, "intervention_rows": probes, "branch_rows": branches,
            "provenance": {"dataset": graph.name, "condition": model.condition, **condition_metadata(model.condition),
                           "seeds": list(seeds), "before_sha256": before, "after_sha256": after,
                           "parameters_preserved": True, "optimizer_updates": 0,
                           "coefficient_policy": "recompute_from_current_hidden", "no_op_checks": len(noop_residuals),
                           "no_op_absolute_error_max": max(noop_residuals, default=0.)}}
