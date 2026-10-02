"""Frozen full-graph classification, interventions and scale diagnostics.

Labels enter metrics only. All seeds in a checkpoint batch are evaluated in
one forward; repeated manifests are paired treatments within each seed.
No optimizer, fitting, feature re-normalization or dense N-by-N operator is
used here. Random pairs are sampled as integer ranks rather than constructing
the E-choose-2 table. Relative diagnostics with a zero reference are undefined.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "classification_metrics",
    "split_indices",
    "create_intervention_manifests",
    "frozen_evaluate",
    "model_state_hash",
    "estimate_second_operator_norm",
]

_LEARNED = {"learned_wedge_raw", "learned_wedge_rms"}
_SPLITS = (("train", "train_mask"), ("validation", "val_mask"), ("test", "test_mask"))
_HOLD = "hold_each_layers_preintervention_kappa"
_INDEX_CACHE: dict[tuple, dict] = {}


def split_indices(graph: Any) -> dict[str, Tensor]:
    """Cache immutable split indices once without mutating the source graph.

    Weak references release cached GPU indices when masks die. Tensor mutation
    versions invalidate changed masks; inference masks without versions are not
    cached. CPU nonzero and transfer occur at preparation, not each epoch.
    """
    masks = tuple(getattr(graph, attribute) for _, attribute in _SPLITS)
    for mask in masks:
        if mask.dtype != torch.bool or mask.shape != (graph.x.shape[0],):
            raise ValueError("split masks must be boolean [nodes]")
        if mask.device != graph.x.device:
            raise ValueError("split masks must be on the graph device")
    versions = tuple(None if torch.is_inference(mask) else mask._version for mask in masks)
    key = tuple(id(mask) for mask in masks) + (graph.x.shape[0], str(graph.x.device))
    previous = _INDEX_CACHE.get(key)
    if (
        previous is not None
        and previous["versions"] == versions
        and all(
            reference() is mask for reference, mask in zip(previous["refs"], masks, strict=True)
        )
    ):
        return previous["indices"]
    with torch.inference_mode(False):
        indices = {
            split: mask.detach().cpu().nonzero().flatten().to(graph.x.device)
            for (split, _), mask in zip(_SPLITS, masks, strict=True)
        }
    if None not in versions:

        def discard(_reference: Any) -> None:
            _INDEX_CACHE.pop(key, None)

        _INDEX_CACHE[key] = {
            "versions": versions,
            "indices": indices,
            "refs": tuple(weakref.ref(mask, discard) for mask in masks),
        }
    return indices


def classification_metrics(
    logits: Tensor,
    y: Tensor,
    mask: Tensor,
    *,
    validate: bool = True,
    indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return per-seed unregularized mean CE and accuracy in [0,1]."""
    if logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be floating [seeds,nodes,classes]")
    if y.dtype != torch.long or y.shape != (logits.shape[1],):
        raise ValueError("y must be long [nodes]")
    if mask.dtype != torch.bool or mask.shape != y.shape:
        raise ValueError("mask must be boolean [nodes]")
    if y.device != logits.device or mask.device != logits.device:
        raise ValueError("logits, y and mask must share a device")
    if validate and not bool(mask.any()):
        raise ValueError("classification mask must contain at least one labeled node")
    if validate and not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite classification logits")
    if indices is not None:
        if indices.dtype != torch.long or indices.ndim != 1 or indices.device != logits.device:
            raise ValueError("indices must be Long[count] on the logits device")
        if validate and not torch.equal(indices, mask.nonzero().flatten()):
            raise ValueError("indices do not match the classification mask")
        selected, labels = logits.index_select(1, indices), y.index_select(0, indices)
    else:
        selected, labels = logits[:, mask, :], y[mask]
    if validate and bool(((labels < 0) | (labels >= logits.shape[-1])).any()):
        raise ValueError("masked label outside class range")
    losses = F.cross_entropy(
        selected.reshape(-1, logits.shape[-1]), labels.repeat(logits.shape[0]), reduction="none"
    )
    ce = losses.reshape(logits.shape[0], -1).mean(1)
    accuracy = (selected.argmax(-1) == labels[None, :]).to(logits.dtype).mean(1)
    return ce, accuracy


def model_state_hash(model: Any) -> str:
    """Hash actual parameter/buffer bytes before and after frozen evaluation."""
    digest = hashlib.sha256()
    entries = dict(model.named_parameters()) | dict(model.named_buffers())
    for name, value in sorted(entries.items()):
        array = value.detach().cpu().contiguous()
        digest.update(f"{name}:{array.dtype}:{tuple(array.shape)}".encode())
        if array.numel():
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_hash(*values: Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        data = value.detach().cpu().contiguous()
        digest.update(f"{data.dtype}:{tuple(data.shape)}".encode())
        if data.numel():
            digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sample_pair_ranks(population: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count > population:
        raise ValueError("more random rows requested than distinct physical edge pairs")
    if count == population:
        # Every returned rank is a required output row, rather than a candidate table.
        return rng.permutation(count).astype(np.int64)
    selected = np.empty(0, dtype=np.int64)
    while selected.size < count:
        missing = count - selected.size
        candidates = rng.integers(population, size=max(1024, 2 * missing), dtype=np.int64)
        _, first = np.unique(candidates, return_index=True)
        candidates = candidates[np.sort(first)]
        if selected.size:
            candidates = candidates[~np.isin(candidates, selected, assume_unique=True)]
        selected = np.concatenate((selected, candidates[:missing]))
    return selected


def _random_rows(
    edges: Tensor, count: int, rng: np.random.Generator, adjacency: Any = None, two_hop: Any = None
) -> tuple[dict, dict]:
    physical = edges.detach().cpu()
    edge_count = physical.shape[1]
    if count == 0:
        return {
            "indices": torch.empty((4, 0), dtype=torch.long),
            "coefficients": torch.empty((4, 0), dtype=torch.float32),
        }, {
            "num_rows": 0,
            "physical_edges_used": 0,
            "edge_frequency_min": 0,
            "edge_frequency_max": 0,
            "shared_center_fraction": None,
            "support_2_fraction": None,
            "support_3_fraction": None,
            "support_4_fraction": None,
            "row_norm_min": None,
            "row_norm_max": None,
            "trace": 0.0,
        }
    if edge_count < 2:
        raise ValueError("nonempty random rows require two distinct physical edges")
    ranks = _sample_pair_ranks(edge_count * (edge_count - 1) // 2, count, rng)
    starts = np.arange(edge_count, dtype=np.int64)
    starts = starts * (2 * edge_count - starts - 1) // 2
    first = np.searchsorted(starts, ranks, side="right") - 1
    second = ranks - starts[first] + first + 1
    pair = torch.from_numpy(np.stack((first, second)))
    signs = torch.from_numpy(rng.choice(np.array([-1.0, 1.0]), size=(2, count)))
    indices = torch.stack(
        (physical[0, pair[0]], physical[1, pair[0]], physical[0, pair[1]], physical[1, pair[1]])
    )
    coefficients = torch.stack((signs[0], -signs[0], -signs[1], signs[1]))
    equal = indices[:, None, :] == indices[None, :, :]
    squared = (equal * coefficients[:, None, :] * coefficients[None, :, :]).sum((0, 1))
    if bool((squared <= 0).any()):
        raise ValueError("distinct physical pair produced a zero incidence row")
    coefficients = coefficients * (6.0 / squared).sqrt()[None, :]
    combined = (equal * coefficients[None, :, :]).sum(1)
    first_occurrence = torch.stack(
        [
            torch.ones(count, dtype=torch.bool)
            if index == 0
            else ~(indices[index] == indices[:index]).any(0)
            for index in range(4)
        ]
    )
    support = (first_occurrence & (combined != 0)).sum(0)
    norms = (equal * coefficients[:, None, :] * coefficients[None, :, :]).sum((0, 1)).sqrt()
    frequency = torch.bincount(pair.reshape(-1), minlength=edge_count)
    shared = (indices[:2, None, :] == indices[None, 2:, :]).any((0, 1))
    stats = {
        "num_rows": count,
        "physical_edges_used": int((frequency > 0).sum()),
        "edge_frequency_min": int(frequency.min()),
        "edge_frequency_max": int(frequency.max()),
        "shared_center_fraction": float(shared.double().mean()),
        "support_2_fraction": float((support == 2).double().mean()),
        "support_3_fraction": float((support == 3).double().mean()),
        "support_4_fraction": float((support == 4).double().mean()),
        "row_norm_min": float(norms.min()),
        "row_norm_max": float(norms.max()),
        "trace": float(norms.square().sum()),
    }
    if adjacency is not None:
        hop_counts = [0, 0, 0]
        outside = np.zeros(count, dtype=bool)
        for first_position in range(4):
            for second_position in range(first_position + 1, 4):
                valid = (
                    (first_occurrence[first_position] & first_occurrence[second_position])
                    & (combined[first_position] != 0)
                    & (combined[second_position] != 0)
                )
                positions = valid.nonzero().flatten().numpy()
                # Own these index buffers: scipy's paired-index broadcasting
                # cannot make a writable view from a torch-backed numpy array.
                u = indices[first_position, positions].numpy().copy()
                v = indices[second_position, positions].numpy().copy()
                hop1 = np.asarray(adjacency[u, v]).reshape(-1) > 0
                hop2 = np.asarray(two_hop[u, v]).reshape(-1) > 0
                far = ~(hop1 | hop2)
                hop_counts[0] += int(hop1.sum())
                hop_counts[1] += int((hop2 & ~hop1).sum())
                hop_counts[2] += int(far.sum())
                outside[positions[far]] = True
        stats.update(
            support_node_pairs_hop1=hop_counts[0],
            support_node_pairs_hop2=hop_counts[1],
            support_node_pairs_outside2_or_disconnected=hop_counts[2],
            rows_outside2_or_disconnected_fraction=float(outside.mean()),
        )
    stats["geometry_scope"] = "unpreconditioned_random_rows_before_true_SQ"
    return {"indices": indices, "coefficients": coefficients.float()}, stats


def create_intervention_manifests(graph: Any, count: int, seed: int) -> list[dict]:
    """Create fixed common-seed shuffle/random manifests without dropping paths."""
    if isinstance(count, bool) or count < 1 or int(count) != count:
        raise ValueError("manifest count must be a positive integer")
    if graph.edges.dtype != torch.long or graph.edges.ndim != 2 or graph.edges.shape[0] != 2:
        raise ValueError("physical edges must be Long[2,E]")
    paths = int(graph.paths.shape[1])
    physical = graph.edges.detach().cpu().numpy()
    node_count = graph.x.shape[0]
    endpoints = np.concatenate((physical[0], physical[1]))
    adjacency = sp.csr_matrix(
        (
            np.ones(endpoints.size, dtype=np.int32),
            (endpoints, np.concatenate((physical[1], physical[0]))),
        ),
        shape=(node_count, node_count),
    )
    two_hop = adjacency @ adjacency
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        permutation = torch.from_numpy(rng.permutation(paths).astype(np.int64))
        rows, statistics = _random_rows(graph.edges, paths, rng, adjacency, two_hop)
        records.append(
            {
                "index": index,
                "seed": int(seed),
                "permutation": permutation,
                "random_rows": rows,
                "statistics": statistics,
                "sha256": _tensor_hash(permutation, rows["indices"], rows["coefficients"]),
            }
        )
    return records


def _device_manifest(manifest: dict, graph: Any) -> dict:
    with torch.inference_mode(False):
        return {
            **manifest,
            "permutation": manifest["permutation"].to(graph.x.device),
            "random_rows": {
                "indices": manifest["random_rows"]["indices"].to(graph.x.device),
                "coefficients": manifest["random_rows"]["coefficients"].to(
                    device=graph.x.device, dtype=graph.x.dtype
                ),
            },
        }


def estimate_second_operator_norm(
    graph: Any,
    c: Tensor,
    kappa: Tensor,
    random_rows: dict | None = None,
    *,
    iterations: int = 32,
    seed: int = 20261004,
    apply_operator: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Batched sparse PSD power estimate, not an exact norm or certified bound."""
    if c.ndim != 2 or c.shape[1] != graph.paths.shape[1]:
        raise ValueError("c must have [seeds,all_paths] for a norm estimate")
    if kappa.shape != (c.shape[0],) or bool((kappa <= 0).any()):
        raise ValueError("kappa must be positive [seeds]")
    if iterations < 1:
        raise ValueError("power iterations must be positive")
    if random_rows is None:
        indices = graph.paths
        coefficients = c.new_tensor([1.0, -2.0, 1.0])[:, None].expand(-1, c.shape[1])
    else:
        indices, coefficients = random_rows["indices"], random_rows["coefficients"]
    sq = graph.sq.to(c)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(
        (c.shape[0], graph.x.shape[0], 1), generator=generator, dtype=torch.float64
    ).to(c)

    def apply(value: Tensor) -> Tensor:
        scaled = value * sq[None, :, None]
        q = (scaled[:, indices, :] * coefficients[None, :, :, None]).sum(1)
        flux = c[:, :, None] * q
        result = torch.zeros_like(value).index_add(
            1,
            indices.reshape(-1),
            (coefficients[None, :, :, None] * flux[:, None, :, :]).reshape(c.shape[0], -1, 1),
        )
        return result * sq[None, :, None] / (3 * kappa[:, None, None])

    actual_apply = apply if apply_operator is None else apply_operator
    for _ in range(iterations):
        action = actual_apply(vector)
        norm = action.square().sum((1, 2)).sqrt()
        vector = action / norm.clamp_min(torch.finfo(c.dtype).tiny)[:, None, None]
    action = actual_apply(vector)
    result = (vector * action).sum((1, 2)).abs()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite sparse operator norm estimate")
    return result


def _relative(first: Tensor, reference: Tensor) -> list[float | None]:
    difference = (first - reference).reshape(first.shape[0], -1).norm(dim=1)
    denominator = reference.reshape(reference.shape[0], -1).norm(dim=1)
    values = torch.stack((difference, denominator), dim=1).detach().cpu().tolist()
    if any(not math.isfinite(number) for pair in values for number in pair):
        raise ValueError("nonfinite relative diagnostic")
    return [numerator / norm if norm > 0 else None for numerator, norm in values]


def _row_base(graph: Any, model: Any, seed: int) -> dict:
    return {"dataset": str(graph.name), "condition": str(model.condition), "seed": int(seed)}


def _metrics_rows(
    logits: Tensor, graph: Any, model: Any, seeds: Sequence[int], extra: dict | None = None
) -> list[dict]:
    rows = []
    prepared = split_indices(graph)
    for split, attribute in _SPLITS:
        mask = getattr(graph, attribute)
        ce, accuracy = classification_metrics(logits, graph.y, mask, indices=prepared[split])
        observations = torch.stack((ce, accuracy), dim=1).detach().cpu().tolist()
        for index, seed in enumerate(seeds):
            rows.append(
                {
                    **_row_base(graph, model, seed),
                    "split": split,
                    "ce": observations[index][0],
                    "accuracy": observations[index][1],
                    "num_nodes": int(graph.x.shape[0]),
                    "num_labeled_nodes": prepared[split].numel(),
                    **(extra or {}),
                }
            )
    return rows


def _gate_rows(
    details: list[dict],
    graph: Any,
    model: Any,
    seeds: Sequence[int],
    extra: dict | None = None,
    manifest: dict | None = None,
) -> list[dict]:
    rows = []
    if len(details) != 2:
        raise ValueError("classification diagnostics must contain exactly two layers")
    for layer, detail in enumerate(details):
        c = detail.get("c")
        statistics: dict[str, list] = {}
        if c is not None:
            if c.shape != (len(seeds), graph.paths.shape[1]):
                raise ValueError("C diagnostics must include every seed and path")
            if not bool(torch.isfinite(c).all()) or bool((c <= 0).any()):
                raise ValueError("gate C must be finite and positive")
            if c.shape[1]:
                for name, value in (
                    ("c_mean", c.mean(1)),
                    ("c_std", c.std(1, unbiased=False)),
                    ("c_min", c.amin(1)),
                    ("c_max", c.amax(1)),
                ):
                    statistics[name] = value.detach().cpu().tolist()
            else:
                statistics.update(
                    {name: [None] * len(seeds) for name in ("c_mean", "c_std", "c_min", "c_max")}
                )
        for name in ("kappa", "kappa_reference", "sigma", "alpha", "beta", "branch_norm"):
            value = detail.get(name)
            if value is not None:
                value = value.reshape(-1)
                if value.numel() != len(seeds) or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"invalid {name} diagnostics")
                statistics[name] = value.detach().cpu().tolist()
        for name in ("l_message", "t_message"):
            value = detail.get(name)
            if value is not None:
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"nonfinite {name}")
                statistics[name + "_norm"] = (
                    value.reshape(len(seeds), -1).norm(dim=1).cpu().tolist()
                )
        if c is not None and detail.get("kappa") is not None:
            removed = extra is not None and extra.get("intervention") == "second_branch_remove"
            estimate = (
                torch.zeros(len(seeds), device=c.device, dtype=c.dtype)
                if removed
                else (
                    estimate_second_operator_norm(
                        graph,
                        c,
                        detail["kappa"],
                        None if manifest is None else manifest["random_rows"],
                        apply_operator=partial(
                            model.apply_branch,
                            graph,
                            c=c,
                            kappa=detail["kappa"],
                            random_rows=None if manifest is None else manifest["random_rows"],
                        ),
                    )
                )
            )
            statistics["operator_norm_estimate"] = estimate.cpu().tolist()
        for index, seed in enumerate(seeds):
            rows.append(
                {
                    **_row_base(graph, model, seed),
                    "layer": layer,
                    "num_paths": int(graph.paths.shape[1]),
                    **{name: value[index] for name, value in statistics.items()},
                    "operator_norm_iterations": 32
                    if "operator_norm_estimate" in statistics
                    and not (extra or {}).get("intervention") == "second_branch_remove"
                    else 0,
                    **(extra or {}),
                }
            )
    return rows


def frozen_evaluate(
    model: Any,
    graph: Any,
    seeds: Sequence[int],
    config: dict,
    manifests: list[dict] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Evaluate one selected seed batch; root owns result-file serialization."""
    seeds = tuple(int(value) for value in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be nonempty and distinct")
    if hasattr(model, "seeds") and tuple(model.seeds) != seeds:
        raise ValueError("model seed order differs from evaluation seed order")
    before = model_state_hash(model)
    was_training = model.training
    metric_rows: list[dict] = []
    intervention_rows: list[dict] = []
    scale_rows: list[dict] = []
    gate_rows: list[dict] = []
    settings = config["evaluation"]
    amplitudes = settings["scale_amplitudes"]
    if any(isinstance(a, bool) or not math.isfinite(float(a)) or a <= 0 for a in amplitudes):
        raise ValueError("scale amplitudes must be finite and positive")
    if len(set(amplitudes)) != len(amplitudes) or 1.0 not in amplitudes:
        raise ValueError("scale amplitudes must be distinct and contain 1")
    records = manifests or []
    if model.condition in _LEARNED:
        expected = settings["shuffle_and_random_manifests_per_dataset"]
        if len(records) != expected:
            raise ValueError(f"learned evaluation requires all {expected} fixed manifests")
        if {record["index"] for record in records} != set(range(expected)):
            raise ValueError("manifest indices do not cover the fixed declared set")
        for record in records:
            rows = record["random_rows"]
            if record["sha256"] != _tensor_hash(
                record["permutation"], rows["indices"], rows["coefficients"]
            ):
                raise ValueError("intervention manifest hash mismatch")
    model.eval()
    try:
        with torch.inference_mode():
            logits, details = model(graph, epoch=0, diagnostics=True)
            metric_rows = _metrics_rows(logits, graph, model, seeds)
            gate_rows += _gate_rows(
                details,
                graph,
                model,
                seeds,
                {"intervention": "original", "manifest_index": -1, "kappa_mode": "recompute"},
            )
            for amplitude in amplitudes:
                if progress is not None:
                    progress(f"{graph.name} {model.condition} end_to_end amplitude={amplitude}")
                scaled, _ = model(graph, epoch=0, input_scale=float(amplitude), diagnostics=False)
                logit_difference = _relative(scaled / amplitude, logits)
                rows = _metrics_rows(
                    scaled,
                    graph,
                    model,
                    seeds,
                    {"scope": "end_to_end", "amplitude": float(amplitude), "layer": -1},
                )
                for row in rows:
                    row["logit_scale_equivariance_relerr"] = logit_difference[
                        seeds.index(row["seed"])
                    ]
                scale_rows += rows
            if model.condition in _LEARNED:
                for layer, reference in enumerate(details):
                    for amplitude in amplitudes:
                        message, probe = model.probe_layer(
                            graph,
                            layer,
                            reference["z"],
                            input_scale=float(amplitude),
                            diagnostics=True,
                        )
                        changes = {
                            "c_scale_relerr": _relative(probe["c"], reference["c"]),
                            "kappa_scale_relerr": _relative(probe["kappa"], reference["kappa"]),
                            "message_scale_equivariance_relerr": _relative(
                                message / amplitude, reference["t_message"]
                            ),
                        }
                        for index, seed in enumerate(seeds):
                            scale_rows.append(
                                {
                                    **_row_base(graph, model, seed),
                                    "scope": "fixed_layer_Z",
                                    "split": "not_applicable",
                                    "layer": layer,
                                    "amplitude": float(amplitude),
                                    **{name: values[index] for name, values in changes.items()},
                                }
                            )
                modes = settings["identity_and_shuffle_kappa_modes"]
                if set(modes) != {"recompute", _HOLD}:
                    raise ValueError(
                        "identity/shuffle require recomputed and held kappa diagnostics"
                    )
                treatments = [("c_identity", mode, None) for mode in modes]
                treatments += [
                    ("c_position_shuffle", mode, record) for mode in modes for record in records
                ]
                treatments += [("second_branch_remove", "not_applicable", None)]
                treatments += [
                    ("random_physical_edge_pair_correspondence", _HOLD, record)
                    for record in records
                ]
                for intervention, mode, record in treatments:
                    if progress is not None:
                        progress(
                            f"{graph.name} {model.condition} {intervention} kappa={mode} "
                            f"manifest={-1 if record is None else record['index']}"
                        )
                    manifest = None if record is None else _device_manifest(record, graph)
                    extra = {
                        "intervention": intervention,
                        "kappa_mode": mode,
                        "manifest_index": -1 if record is None else int(record["index"]),
                        "manifest_sha256": None if record is None else record["sha256"],
                    }
                    changed, changed_details = model(
                        graph,
                        epoch=0,
                        intervention=intervention,
                        manifest=manifest,
                        kappa_mode="recompute" if mode == "not_applicable" else mode,
                        diagnostics=True,
                    )
                    intervention_rows += _metrics_rows(changed, graph, model, seeds, extra)
                    random_manifest = manifest if intervention.startswith("random_") else None
                    gate_rows += _gate_rows(
                        changed_details, graph, model, seeds, extra, random_manifest
                    )
    finally:
        model.train(was_training)
    after = model_state_hash(model)
    if before != after:
        raise RuntimeError("model parameters/buffers changed during frozen evaluation")
    return {
        "metric_rows": metric_rows,
        "intervention_rows": intervention_rows,
        "scale_rows": scale_rows,
        "gate_rows": gate_rows,
        "provenance": {
            "dataset": str(graph.name),
            "condition": str(model.condition),
            "seeds": list(seeds),
            "model_hash_before": before,
            "model_hash_after": after,
            "models_unchanged": True,
            "optimizer_updates": 0,
            "manifest_hashes": [record["sha256"] for record in records],
        },
    }
