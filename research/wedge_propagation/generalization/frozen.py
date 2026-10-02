"""Load Experiment 2 artifacts unchanged and audit fresh-input scale behavior.

No fit, optimizer or checkpoint selection is performed here. The selected state
and train-only scalar fits are the only source of model parameters.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import weakref
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..learned.model import make_model

_TARGETS = ("L", "L2", "path")
_CONDITIONS = ("first", "polynomial", "fixed", "learned", "random_pair")
_MATH_FILES = ("learned/model.py", "learned/data.py", "learned/evaluation.py",
               "operators.py", "data.py")


@dataclass(frozen=True)
class FrozenModel:
    model: nn.Module
    seeds: tuple[int, ...]
    provenance: dict[str, Any]


def _digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _normalized_hashes(hashes: dict[str, str]) -> dict[str, str]:
    normalized = {}
    for name, value in hashes.items():
        key = name.replace("\\", "/")
        if key.startswith("/") or ".." in key.split("/") or key in normalized:
            raise ValueError("source hashes contain ambiguous or nonrelative paths")
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"invalid source SHA256 for {key}")
        normalized[key] = value
    return normalized


def _artifact_bytes(source_run: Any, relative: str, hashes: dict[str, str]) -> tuple[bytes, str]:
    if relative not in hashes:
        raise ValueError(f"source run has no integrity hash for {relative}")
    value = (Path(source_run.run_dir) / relative).read_bytes()
    digest = hashlib.sha256(value).hexdigest()
    if digest != hashes[relative]:
        raise ValueError(f"source artifact hash mismatch: {relative}")
    return value, digest


@torch.no_grad()
def load_models(source_run: Any, device: torch.device | str,
                dtype: torch.dtype) -> dict[tuple[str, str], FrozenModel]:
    """Load all fifteen prescribed models and reject mismatched provenance."""

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("frozen evaluation requires explicit float32 or float64")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable; no CPU fallback")
    config, contract = source_run.config, source_run.contract
    config_hash = _digest(config)
    if (contract.get("experiment") != 2 or contract.get("config") != config
            or contract.get("config_hash") != config_hash
            or contract.get("source_unchanged_during_run") is not True):
        raise ValueError("source config/contract or unchanged-source declaration does not match")
    if tuple(config["targets"]) != _TARGETS or tuple(config["conditions"]) != _CONDITIONS:
        raise ValueError("source must contain all three targets and all five conditions")
    source_hashes = _normalized_hashes(contract["source"]["sha256"])
    code_root = Path(__file__).resolve().parent.parent
    for relative in _MATH_FILES:
        current = hashlib.sha256((code_root / relative).read_bytes()).hexdigest()
        if source_hashes.get(relative) != current:
            raise ValueError(f"source mathematical code is incompatible: {relative}")
    hashes = _normalized_hashes(source_run.hashes)
    jobs = {}
    for job in contract["jobs"]:
        key = (job["target"], job["condition"])
        if key in jobs:
            raise ValueError("source contract contains duplicate target/condition jobs")
        jobs[key] = job
    if set(jobs) != {(target, condition) for target in _TARGETS for condition in _CONDITIONS}:
        raise ValueError("source contract does not cover every target/condition")
    result = {}
    source_dtype = getattr(torch, config["dtype"])
    for target in _TARGETS:
        for condition in _CONDITIONS:
            job = jobs[(target, condition)]
            seeds = tuple([-1] if condition in _CONDITIONS[:3] else config["model_seeds"])
            if tuple(job["selection"]["seeds"]) != seeds:
                raise ValueError(f"source contract seed order mismatch: {target}/{condition}")
            model = make_model(condition, seeds, config["hidden"], config["teacher"]["tau"])
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if job["parameters_per_seed"] != parameter_count // len(seeds):
                raise ValueError(f"source parameter count mismatch: {target}/{condition}")
            provenance = {
                "source_run": str(Path(source_run.run_dir).resolve()),
                "source_config_hash": config_hash, "source_git": contract["source"].get("git"),
                "source_math_code_sha256": {name: source_hashes[name] for name in _MATH_FILES},
                "target": target, "condition": condition, "seeds": list(seeds),
                "parameters_per_seed": parameter_count // len(seeds),
                "parameters_total": parameter_count, "refit": False,
                "checkpoint_reselection": False, "evaluation_dtype": str(dtype),
            }
            if condition in _CONDITIONS[:3]:
                relative = f"{target}-{condition}-fit.json"
                payload, digest = _artifact_bytes(source_run, relative, hashes)
                fit = json.loads(payload.decode("utf-8-sig"))
                expected = {name for name, _ in model.named_parameters()}
                if (set(fit) != expected
                        or job["selection"].get("fit") != "train-only analytic optimum"):
                    raise ValueError(f"source scalar fit schema/selection mismatch: {relative}")
                for name, value in fit.items():
                    if (isinstance(value, bool) or not isinstance(value, Real)
                            or not math.isfinite(value)):
                        raise ValueError(f"source scalar fit must be finite: {relative}/{name}")
                    getattr(model, name).fill_(value)
                provenance["scalar_fit"] = fit
            else:
                relative = f"checkpoints/{target}-{condition}-selected.pt"
                payload, digest = _artifact_bytes(source_run, relative, hashes)
                saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                if (saved["target"] != target or saved["condition"] != condition
                        or tuple(saved["seeds"]) != seeds or saved["config"] != config
                        or saved["config_hash"] != config_hash):
                    raise ValueError(f"selected checkpoint identity/config mismatch: {relative}")
                if (_normalized_hashes(saved["source"]["sha256"]) != source_hashes
                        or saved["source"].get("git") != contract["source"].get("git")):
                    raise ValueError(f"selected checkpoint source mismatch: {relative}")
                epochs, scores = saved["best_epochs"], saved["validation_message_relerr"]
                if (epochs != job["selection"]["best_epochs"] or len(epochs) != len(seeds)
                        or any(isinstance(epoch, bool) or not isinstance(epoch, int)
                               or not 0 <= epoch <= config["epochs"] for epoch in epochs)):
                    raise ValueError(f"selected checkpoint validation epochs mismatch: {relative}")
                if (len(scores) != len(seeds)
                        or any(not isinstance(score, Real) or not math.isfinite(score)
                               or score < 0 for score in scores)):
                    raise ValueError(f"selected checkpoint validation scores invalid: {relative}")
                state, expected_state = saved["state_dict"], model.state_dict()
                if set(state) != set(expected_state):
                    raise ValueError(f"selected checkpoint state keys mismatch: {relative}")
                for name, tensor in state.items():
                    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                            or tensor.shape != expected_state[name].shape
                            or tensor.dtype != source_dtype or not torch.isfinite(tensor).all()):
                        raise ValueError(f"selected checkpoint state mismatch: {relative}/{name}")
                model.load_state_dict(state, strict=True)
                provenance.update(best_epochs=list(epochs), validation_message_relerr=list(scores))
            model.to(device=device, dtype=dtype).requires_grad_(False).eval()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError(f"parameter conversion produced nonfinite values: {relative}")
            provenance.update(artifact=relative, artifact_sha256=digest)
            result[(target, condition)] = FrozenModel(model, seeds, provenance)
    return result


def frozen_fingerprint(model: nn.Module | FrozenModel) -> str:
    """Hash state bytes, precision, semantic settings and all freeze/eval flags."""

    if isinstance(model, FrozenModel):
        model = model.model
    descriptor = {
        "kind": getattr(model, "kind", None), "seeds": list(getattr(model, "seeds", ())),
        "gate_tau": getattr(getattr(model, "gate", None), "tau", None),
        "modules": [(name, type(module).__qualname__, module.training)
                    for name, module in model.named_modules()],
        "parameters": [(name, parameter.requires_grad, str(parameter.device))
                       for name, parameter in model.named_parameters()],
    }
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
    for name, tensor in sorted(model.state_dict().items()):
        descriptor = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        digest.update(json.dumps(descriptor, sort_keys=True).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class _PairCheck:
    reference: weakref.ReferenceType
    scaled: weakref.ReferenceType
    token: tuple
    tensor_check: torch.Tensor
    path_counts: torch.Tensor


_PAIR_CHECKS: dict[tuple[int, int, float], _PairCheck] = {}


def _version(tensor: torch.Tensor) -> int | None:
    return None if torch.is_inference(tensor) else tensor._version


def _check_pair(reference, scaled, amplitude: float) -> _PairCheck:
    ref_cases, ref = reference
    scaled_cases, actual = scaled
    if (not ref_cases or len(ref_cases) != len(scaled_cases)
            or ref.num_graphs != len(ref_cases) or actual.num_graphs != len(ref_cases)
            or ref.graph_ids != actual.graph_ids
            or tuple(case.graph_id for case in ref_cases) != ref.graph_ids
            or tuple(case.graph_id for case in scaled_cases) != actual.graph_ids):
        raise ValueError("reference/scaled graph IDs, counts or case order differ")
    tensor_names = ("x", "edges", "wedges", "pair_edges", "pair_coefficients",
                    "path_graph", "node_graph", "num_nodes", "teacher_c", "lx", "l2x", "qx")
    tensors = [getattr(batch, name) for batch in (ref, actual) for name in tensor_names]
    if (ref.x.shape != actual.x.shape or ref.x.dtype != actual.x.dtype
            or ref.x.device != actual.x.device):
        raise ValueError("reference/scaled input shape, dtype or device differ")
    metadata = tuple((case.graph_id, case.family, case.split,
                      case.metadata.get("feature_status"), case.metadata.get("amplitude"),
                      case.metadata.get("fresh_feature_seed"), id(case.features),
                      _version(case.features), id(case.edges), _version(case.edges),
                      id(case.wedges), _version(case.wedges))
                     for case in (*ref_cases, *scaled_cases))
    token = (metadata, tuple((id(tensor), _version(tensor)) for tensor in tensors))
    key = (id(ref), id(actual), amplitude)
    cached = _PAIR_CHECKS.get(key)
    if (cached is not None and cached.reference() is ref and cached.scaled() is actual
            and cached.token == token and all(_version(tensor) is not None for tensor in tensors)):
        return cached
    for first, second in zip(ref_cases, scaled_cases, strict=True):
        if (first.family != second.family or first.split != second.split
                or first.num_nodes != second.num_nodes
                or first.metadata.get("feature_status") != "fresh"
                or second.metadata.get("feature_status") != "fresh"
                or first.metadata.get("amplitude") != 1.0
                or second.metadata.get("amplitude") != amplitude
                or first.metadata.get("fresh_feature_seed") is None
                or first.metadata.get("fresh_feature_seed")
                != second.metadata.get("fresh_feature_seed")):
            raise ValueError("scale diagnostics require paired fresh inputs with reference alpha 1")
        for name in ("edges", "wedges", "pair_edges", "pair_coefficients"):
            if not torch.equal(getattr(first, name), getattr(second, name)):
                raise ValueError(f"reference/scaled topology differs: {first.graph_id}/{name}")
        if not torch.equal(second.features, first.features * amplitude):
            raise ValueError("reference/scaled case features are not an amplitude pair")
    checks = []
    for name in tensor_names[1:8]:
        left, right = getattr(ref, name), getattr(actual, name)
        if left.shape != right.shape or left.device != right.device or left.dtype != right.dtype:
            raise ValueError(f"reference/scaled packed topology shape differs: {name}")
        checks.append((left == right).all())
    expected_x = amplitude * ref.x
    tolerance = 8 * torch.finfo(ref.x.dtype).eps
    bound = tolerance * torch.maximum(expected_x.abs(), actual.x.abs()).clamp_min(1)
    checks.append(((actual.x - expected_x).abs() <= bound).all())
    path_counts = torch.tensor([case.wedges.shape[1] for case in ref_cases],
                               device=ref.x.device, dtype=torch.long)
    entry = _PairCheck(weakref.ref(ref), weakref.ref(actual), token,
                       torch.stack(checks).all(), path_counts)
    _PAIR_CHECKS[key] = entry
    return entry


def _graph_norm(values: torch.Tensor, groups: torch.Tensor, graph_count: int) -> torch.Tensor:
    if values.ndim == 2:
        values = values[None]
    return values.new_zeros(values.shape[0], graph_count, values.shape[-1]).index_add(
        1, groups, values.square()
    ).sqrt()


@torch.inference_mode()
def scale_diagnostics(model: nn.Module | FrozenModel, reference_batch, scaled_batch,
                      amplitude: float, seeds, epsilon: float,
                      *, target: str = "path") -> list[dict[str, Any]]:
    """Compare paired fresh inputs across all graphs/seeds/scalar realizations.

    Inputs follow the cached ``(cases, RuleBatch)`` convention. C drift is
    measured within each operator's own rows. Teacher C is a true-path diagnostic
    only for the path target; teacher messages use the requested target labels.
    Baselines and graphs without paths have no student weight-drift metric.
    """

    if (isinstance(amplitude, bool) or not isinstance(amplitude, Real)
            or not math.isfinite(amplitude) or amplitude <= 0):
        raise ValueError("amplitude must be positive and finite")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if target not in _TARGETS:
        raise ValueError("unknown scale diagnostic target")
    if isinstance(model, FrozenModel):
        model = model.model
    if (tuple(seeds) != model.seeds
            or any(parameter.requires_grad for parameter in model.parameters())
            or any(module.training for module in model.modules())):
        raise ValueError("scale diagnostics require the exact frozen/eval seed models")
    pairing = _check_pair(reference_batch, scaled_batch, float(amplitude))
    cases, reference = reference_batch
    _, scaled = scaled_batch
    if (reference.targets[target].shape != reference.x.shape
            or scaled.targets[target].shape != reference.x.shape):
        raise ValueError("teacher target shape differs from the paired inputs")
    prediction, c = model(reference)
    scaled_prediction, scaled_c = model(scaled)
    graph_count, seed_count = reference.num_graphs, len(seeds)
    message = (_graph_norm(scaled_prediction / amplitude - prediction,
                           reference.node_graph, graph_count)
               / (_graph_norm(prediction, reference.node_graph, graph_count) + epsilon)).mean(-1)
    teacher_message = (_graph_norm(scaled.targets[target] / amplitude - reference.targets[target],
                                   reference.node_graph, graph_count)
                       / (_graph_norm(reference.targets[target], reference.node_graph, graph_count)
                          + epsilon)).mean(-1).expand(seed_count, -1)
    undefined = prediction.new_full((seed_count, graph_count), float("nan"))
    nonempty = pairing.path_counts > 0
    student_weights = undefined
    if c is not None:
        student_weights = (_graph_norm(scaled_c - c, reference.path_graph, graph_count)
                           / (_graph_norm(c, reference.path_graph, graph_count) + epsilon)).mean(-1)
        student_weights = torch.where(nonempty[None], student_weights, undefined)
    teacher_weights = undefined
    if target == "path":
        teacher_weights = (_graph_norm(scaled.teacher_c - reference.teacher_c,
                                       reference.path_graph, graph_count)
                           / (_graph_norm(reference.teacher_c, reference.path_graph, graph_count)
                              + epsilon)).mean(-1).expand(seed_count, -1)
        teacher_weights = torch.where(nonempty[None], teacher_weights, undefined)
    status = pairing.tensor_check.to(prediction.dtype).expand(seed_count, graph_count)
    # One transfer serializes all metrics and the packed-pair integrity check.
    metrics = torch.stack((student_weights, teacher_weights, message, teacher_message, status), -1)
    metrics = metrics.permute(1, 0, 2).cpu().double().tolist()
    rows = []
    for case, by_seed in zip(cases, metrics, strict=True):
        for seed, values in zip(seeds, by_seed, strict=True):
            student_c, teacher_c, message_error, teacher_error, valid = values
            if valid != 1:
                raise ValueError("packed reference/scaled topology or inputs are not paired")
            expected_values = [message_error, teacher_error]
            if c is not None and case.wedges.shape[1]:
                expected_values.append(student_c)
            if target == "path" and case.wedges.shape[1]:
                expected_values.append(teacher_c)
            if not all(math.isfinite(value) for value in expected_values):
                raise ArithmeticError("nonfinite scale diagnostic")
            rows.append({
                "seed": seed, "graph_id": case.graph_id, "split": case.split,
                "family": case.family, "amplitude": float(amplitude),
                "student_weight_scale_relerr": None if math.isnan(student_c) else student_c,
                "teacher_weight_scale_relerr": None if math.isnan(teacher_c) else teacher_c,
                "message_scale_equivariance_relerr": message_error,
                "teacher_message_scale_equivariance_relerr": teacher_error,
            })
    return rows
