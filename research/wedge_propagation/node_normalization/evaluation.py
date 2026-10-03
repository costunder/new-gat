"""Frozen native and equal-norm C probes after validation-locked final training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..classification.evaluation import classification_metrics, model_state_hash, split_indices
from ..classification.model import _lbar

LEARNED = (
    "learned_wedge_raw",
    "learned_wedge_rms",
    "learned_wedge_local_raw",
    "learned_wedge_local_rms",
)
TREATMENTS = (
    "c_identity",
    "c_identity_norm_matched",
    "c_position_shuffle",
    "c_position_shuffle_norm_matched",
    "second_branch_remove",
)
TARGETS = ("layer_0", "layer_1", "both")
SHUFFLE = ("c_position_shuffle", "c_position_shuffle_norm_matched")


def permutation_sha256(permutation: Tensor) -> str:
    value = permutation.detach().cpu().contiguous()
    if value.dtype != torch.long or value.ndim != 1:
        raise ValueError("permutation must be Long[P]")
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}".encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def create_intervention_manifests(graph: Any, count: int, seed: int) -> list[dict]:
    """Common fixed permutations only; every existing wedge remains present."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("manifest count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("manifest seed must be a nonnegative integer")
    if graph.paths.dtype != torch.long or graph.paths.ndim != 2 or graph.paths.shape[0] != 3:
        raise ValueError("graph paths must be Long[3,P]")
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        permutation = torch.from_numpy(rng.permutation(graph.paths.shape[1]).astype(np.int64))
        result.append(
            {
                "index": index,
                "seed": seed,
                "permutation": permutation,
                "sha256": permutation_sha256(permutation),
            }
        )
    return result


def manifest_count(config: dict) -> int:
    value = config["evaluation"].get("shuffle_manifests_per_dataset")
    if value is None:
        value = config["evaluation"].get("shuffle_and_random_manifests_per_dataset")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("positive declared shuffle manifest count required")
    return value


def intervention_variants(count: int):
    for target in TARGETS:
        for treatment in TREATMENTS:
            indices = range(count) if treatment in SHUFFLE else (-1,)
            for index in indices:
                yield treatment, target, index


def _condition(model: Any) -> str:
    return str(getattr(model, "experiment_condition", model.condition))


def _ratio(numerator: Tensor, denominator: Tensor) -> tuple[Tensor, Tensor]:
    defined = denominator > 0
    safe = torch.where(defined, denominator, torch.ones_like(denominator))
    value = numerator / safe
    return torch.where(defined, value, torch.full_like(value, float("nan"))), defined


def _norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value, dim=(1, 2))


def _cosine(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    return _ratio((left * right).sum((1, 2)), _norm(left) * _norm(right))


def _norm_match(candidate: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
    candidate_norm, reference_norm = _norm(candidate), _norm(reference)
    impossible = (candidate_norm == 0) & (reference_norm > 0)
    if bool(impossible.any()):
        raise ValueError("cannot norm-match a zero candidate to a positive reference message")
    safe = torch.where(candidate_norm > 0, candidate_norm, torch.ones_like(candidate_norm))
    gain = reference_norm / safe
    gain = torch.where((candidate_norm == 0) & (reference_norm == 0), 1, gain)
    return candidate * gain[:, None, None], gain


@torch.inference_mode()
def frozen_forward(
    model: Any,
    graph: Any,
    treatment: str,
    target: str,
    manifest: dict | None = None,
) -> tuple[Tensor, list[dict]]:
    """Two original layers; references are recalculated at each CURRENT projected Z."""
    if treatment not in TREATMENTS or target not in TARGETS:
        raise ValueError("unknown frozen treatment or target")
    if _condition(model) not in LEARNED:
        raise ValueError("C interventions require a learned C model")
    if model.training:
        raise ValueError("frozen forward requires model.eval()")
    h = graph.x[None].expand(len(model.seeds), -1, -1)
    details = []
    for layer in (0, 1):
        z = torch.bmm(model._dropout(h, 0, layer), model.projections[layer])
        first = _lbar(graph, z)
        reference, reference_detail = model.probe_layer(
            graph,
            layer,
            z,
            diagnostics=True,
            _l_message=first,
        )
        active = target == "both" or target == f"layer_{layer}"
        if active:
            native = {
                "c_identity_norm_matched": "c_identity",
                "c_position_shuffle_norm_matched": "c_position_shuffle",
            }.get(treatment, treatment)
            message, detail = model.probe_layer(
                graph,
                layer,
                z,
                intervention=native,
                manifest=manifest,
                kappa_mode="recompute",
                diagnostics=True,
                _l_message=first,
            )
            gain = z.new_ones(z.shape[0])
            if treatment.endswith("norm_matched"):
                message, gain = _norm_match(message, reference)
        else:
            message, detail = reference, reference_detail
            gain = z.new_ones(z.shape[0])
        detail = {
            **detail,
            "t_message": message.detach(),
            "reference_t_message": reference.detach(),
            "reference_branch_norm": _norm(reference).detach(),
            "normalization_gain": gain.detach(),
        }
        alpha, beta = model._coefficients(layer, z)
        h = z - alpha[:, None, None] * first - beta[:, None, None] * message
        if layer == 0:
            h = h.relu()
        details.append(detail)
    return h, details


def _base(graph: Any, model: Any, seed: int) -> dict:
    return {"dataset": str(graph.name), "condition": _condition(model), "seed": int(seed)}


def _metrics_rows(logits: Tensor, graph: Any, model: Any, seeds: tuple, extra: dict) -> list[dict]:
    indices = split_indices(graph)
    names, values = [], []
    for split, attribute in (
        ("train", "train_mask"),
        ("validation", "val_mask"),
        ("test", "test_mask"),
    ):
        ce, accuracy = classification_metrics(
            logits,
            graph.y,
            getattr(graph, attribute),
            indices=indices[split],
            validate=False,
        )
        names.append(split)
        values.extend((ce, accuracy))
    data = torch.stack(values, dim=1).detach().cpu()
    if not bool(torch.isfinite(data).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite frozen classification output")
    result = []
    for index, seed in enumerate(seeds):
        for position, split in enumerate(names):
            result.append(
                {
                    **_base(graph, model, seed),
                    **extra,
                    "split": split,
                    "ce": float(data[index, 2 * position]),
                    "accuracy": float(data[index, 2 * position + 1]),
                    "num_nodes": graph.x.shape[0],
                    "num_labeled_nodes": indices[split].numel(),
                }
            )
    return result


def _layer_rows(
    details: list[dict], graph: Any, model: Any, seeds: tuple, extra: dict
) -> list[dict]:
    result = []
    for layer, detail in enumerate(details):
        z, first, second = detail["z"], detail["l_message"], detail["t_message"]
        reference = detail.get("reference_t_message", second)
        alpha, beta = detail["alpha"], detail["beta"]
        first_scaled, second_scaled = alpha[:, None, None] * first, beta[:, None, None] * second
        scalars = {
            "alpha": alpha,
            "beta": beta,
            "kappa": detail["kappa"],
            "kappa_reference": detail["kappa_reference"],
            "kappa_global": detail.get("kappa_global", detail["kappa_reference"]),
            "z_norm": _norm(z),
            "l_norm": _norm(first),
            "branch_norm": _norm(second),
            "reference_branch_norm": _norm(reference),
            "alpha_l_norm": _norm(first_scaled),
            "beta_t_norm": _norm(second_scaled),
            "normalization_gain": detail.get("normalization_gain", z.new_ones(z.shape[0])),
        }
        diagonal = detail.get("weighted_diagonal")
        if diagonal is not None:
            scalars.update(
                weighted_diagonal_mean=diagonal.mean(1),
                weighted_diagonal_min=diagonal.amin(1),
                weighted_diagonal_max=diagonal.amax(1),
            )
        for name in (
            "c_mean",
            "c_std",
            "c_min",
            "c_max",
            "weighted_diagonal_mean",
            "weighted_diagonal_min",
            "weighted_diagonal_max",
        ):
            value = detail.get(name)
            if value is not None:
                scalars[name] = value
        for name, numerator, denominator in (
            ("alpha_l_to_input", scalars["alpha_l_norm"], scalars["z_norm"]),
            ("beta_t_to_input", scalars["beta_t_norm"], scalars["z_norm"]),
            ("beta_t_to_alpha_l", scalars["beta_t_norm"], scalars["alpha_l_norm"]),
            ("message_norm_ratio", scalars["branch_norm"], scalars["reference_branch_norm"]),
            (
                "message_relative_change",
                _norm(second - reference),
                scalars["reference_branch_norm"],
            ),
        ):
            scalars[name], scalars[name + "_defined"] = _ratio(numerator, denominator)
        for name, left, right in (
            ("first_second_cosine", first_scaled, second_scaled),
            ("message_reference_cosine", second, reference),
            ("cosine_correction_input", first_scaled + second_scaled, z),
        ):
            scalars[name], scalars[name + "_defined"] = _cosine(left, right)
        names = tuple(scalars)
        data = torch.stack([scalars[name].to(torch.float64) for name in names], dim=1).cpu()
        for index, seed in enumerate(seeds):
            row = {**_base(graph, model, seed), **extra, "layer": layer}
            for position, name in enumerate(names):
                value = float(data[index, position])
                if name.endswith("_defined"):
                    row[name] = bool(value)
                elif math.isfinite(value):
                    row[name] = value
                elif math.isnan(value) and name + "_defined" in scalars:
                    flag_position = names.index(name + "_defined")
                    if bool(data[index, flag_position]):
                        raise ValueError("defined layer diagnostic is nonfinite")
                    row[name] = None
                else:
                    raise ValueError(f"nonfinite layer diagnostic {name}")
            for name in ("c_mean", "c_std", "c_min", "c_max"):
                if name not in row:
                    row[name] = None
            result.append(row)
    return result


def frozen_evaluate(
    model: Any,
    graph: Any,
    seeds: Sequence[int],
    config: dict,
    manifests: list[dict] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Evaluate one final selected pack; no optimizer, test-based selection or fitting."""
    seeds = tuple(seeds)
    if not seeds or tuple(model.seeds) != seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seed order must match distinct packed model seeds")
    condition = _condition(model)
    if condition not in ("fixed_wedge", *LEARNED):
        raise ValueError("condition outside node normalization experiment")
    records = manifests or []
    count = manifest_count(config)
    if condition in LEARNED:
        if len(records) != count or {r["index"] for r in records} != set(range(count)):
            raise ValueError("all fixed shuffle manifests required")
        for record in records:
            permutation = record["permutation"]
            if record["sha256"] != permutation_sha256(permutation):
                raise ValueError("manifest permutation hash mismatch")
            if permutation.shape != (graph.paths.shape[1],) or not torch.equal(
                permutation.detach().cpu().sort().values,
                torch.arange(graph.paths.shape[1]),
            ):
                raise ValueError("manifest must permute every existing wedge exactly once")
    before = model_state_hash(model)
    training = model.training
    interventions, gates = [], []
    model.eval()
    try:
        with torch.inference_mode():
            logits, details = model(graph, epoch=0, diagnostics=True)
            metrics = _metrics_rows(logits, graph, model, seeds, {})
            original = {"intervention": "original", "target": "none", "manifest_index": -1}
            gates += _layer_rows(details, graph, model, seeds, original)
            if condition in LEARNED:
                with torch.inference_mode(False):
                    device_records = {
                        r["index"]: {**r, "permutation": r["permutation"].to(graph.x.device)}
                        for r in records
                    }
                for treatment, target, index in intervention_variants(count):
                    record = device_records.get(index)
                    if progress is not None:
                        progress(f"{graph.name} {condition} {treatment} {target} manifest={index}")
                    changed, detail = frozen_forward(model, graph, treatment, target, record)
                    extra = {
                        "intervention": treatment,
                        "target": target,
                        "manifest_index": index,
                        "manifest_sha256": None if record is None else record["sha256"],
                        "normalization_mode": (
                            "node_diagonal" if "_local_" in condition else "global_kappa"
                        ),
                        "norm_matched": treatment.endswith("norm_matched"),
                    }
                    interventions += _metrics_rows(changed, graph, model, seeds, extra)
                    gates += _layer_rows(detail, graph, model, seeds, extra)
    finally:
        model.train(training)
    after = model_state_hash(model)
    if before != after:
        raise RuntimeError("model parameters or buffers changed during frozen evaluation")
    return {
        "metric_rows": metrics,
        "intervention_rows": interventions,
        "scale_rows": [],
        "gate_rows": gates,
        "provenance": {
            "dataset": str(graph.name),
            "condition": condition,
            "seeds": list(seeds),
            "model_hash_before": before,
            "model_hash_after": after,
            "models_unchanged": True,
            "optimizer_updates": 0,
            "manifest_hashes": [r["sha256"] for r in records],
        },
    }
