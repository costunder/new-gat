"""Explicit, read-only transition from reviewed legacy V5 checkpoints.

This is deliberately separate from normal resume. Only the C-estimator
namespace may be replaced; shared tensors and AdamW state are retained by
exact parameter names. Source files are never modified and loading uses
PyTorch's weights-only unpickler. An externally pinned SHA is mandatory when
preparing a transition, not merely inspecting a source for a new plan.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

TRANSITION_SCHEMA = 1
C_NAMESPACE = re.compile(r"^blocks\.[0-9]+\.operator\.estimator\.")
# SHA-256 of canonical full implementation_source_hashes() maps, computed
# from immutable Git blobs, not from the current working tree.
_LEGACY_REVISIONS = (
    "76e514a8bf444ef82a323f18ff31982908cf2d8f",
    "89638212e2188746905abe92de9d8545dd624915",
)
LEGACY_SOURCE_SNAPSHOTS = {
    "837c29151d8bfb4165a3a01792c15ce3f45567dce931ce75b07bf044f34e4c9a": _LEGACY_REVISIONS[0],
    "8c0c678e314ef37c663db07f2f7b75a6e34d2bbac392c45979817950f90b8a2f": _LEGACY_REVISIONS[1],
}
_C_CONFIGURATION = {
    "conductance_backend",
    "solver_steps",
    "solver_step_size",
    "solver_entropy",
    "solver_degree_barrier",
    "solver_cost_scaling",
    "training_schedule",
}
_UNCHANGED_SOURCES = (
    "research/conductance_gat/v5/operator.py",
    "research/conductance_gat/v5/sampling.py",
    "research/conductance_gat/benchmark_data.py",
    "src/chartgat/cache.py",
)
_GROUPS = ("backbone", "spatial_w", "beta", "conductance")
_SELECTION = (
    "best_metric",
    "best_epoch",
    "best_checkpoint_sha256",
    "global_best_metric",
    "global_best_epoch",
    "joint_best_metric",
    "joint_best_epoch",
    "first_c_gradient",
)
_EXECUTION_FIELDS = {
    "batch_size",
    "sample_seed_batch_size",
    "workers",
    "loader_workers",
    "persistent_workers",
    "prefetch_factor",
    "worker_configuration_source",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _group(name: str) -> str:
    if C_NAMESPACE.match(name):
        return "conductance"
    if ".operator.beta_estimator." in name:
        return "beta"
    if ".operator.value_weight" in name or ".operator.output_projection." in name:
        return "spatial_w"
    return "backbone"


def _stream_sha256(stream) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"transition {name} must be a CPU tensor")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"transition {name} is nonfinite")
    if value.is_complex() or value.layout != torch.strided:
        raise ValueError(f"transition {name} has an unsupported tensor representation")
    return value


def _optimizer_names(state: Any, model_state: dict[str, torch.Tensor]) -> dict[str, int]:
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("transition optimizer_state must be a named AdamW state dict")
    if not isinstance(state["state"], dict) or not isinstance(state["param_groups"], list):
        raise ValueError("transition optimizer state/groups are malformed")
    names, identifiers, group_names = {}, set(), set()
    for group in state["param_groups"]:
        if not isinstance(group, dict) or group.get("name") not in _GROUPS:
            raise ValueError("transition optimizer group name is invalid")
        group_name = group["name"]
        if group_name in group_names:
            raise ValueError("transition duplicate optimizer group")
        group_names.add(group_name)
        parameters, parameter_names = group.get("params"), group.get("parameter_names")
        if (
            not isinstance(parameters, list)
            or not isinstance(parameter_names, list)
            or not parameters
            or len(parameters) != len(parameter_names)
        ):
            raise ValueError("transition optimizer parameter_names do not align")
        for identifier, name in zip(parameters, parameter_names, strict=True):
            if (
                not _integer(identifier)
                or identifier in identifiers
                or not isinstance(name, str)
                or name in names
                or name not in model_state
                or _group(name) != group_name
            ):
                raise ValueError("transition optimizer parameter ownership is invalid")
            identifiers.add(identifier)
            names[name] = identifier
            moments = state["state"].get(identifier)
            if moments is None:
                continue  # A legitimately never-updated parameter has no Adam state.
            expected = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad", False):
                expected.add("max_exp_avg_sq")
            if not isinstance(moments, dict) or set(moments) != expected:
                raise ValueError(f"transition AdamW moments are incomplete: {name}")
            step = _tensor(moments["step"], f"{name}.step")
            if step.numel() != 1 or not torch.isfinite(step).all():
                raise ValueError(f"transition AdamW step is invalid: {name}")
            numeric_step = float(step)
            if numeric_step < 0 or not numeric_step.is_integer():
                raise ValueError(f"transition AdamW step is invalid: {name}")
            for key in expected - {"step"}:
                moment = _tensor(moments[key], f"{name}.{key}")
                if (
                    moment.shape != model_state[name].shape
                    or moment.dtype != model_state[name].dtype
                ):
                    raise ValueError(f"transition AdamW moment shape/dtype mismatch: {name}.{key}")
                if key.endswith("avg_sq") and (moment < 0).any():
                    raise ValueError(f"transition AdamW squared moment is negative: {name}.{key}")
        for key in ("lr", "weight_decay", "eps"):
            value = group.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"transition optimizer hyperparameter is invalid: {key}")
            if value < 0 or (key == "eps" and value == 0):
                raise ValueError(f"transition optimizer hyperparameter is invalid: {key}")
        betas = group.get("betas")
        if (
            not isinstance(betas, (list, tuple))
            or len(betas) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1
                for value in betas
            )
        ):
            raise ValueError("transition optimizer betas are invalid")
    if set(state["state"]) - identifiers:
        raise ValueError("transition optimizer has orphan moment state")
    if set(names) != set(model_state):
        raise ValueError("transition optimizer does not own every model parameter")
    return names


def _validate_legacy_c(state: dict, identity: dict) -> None:
    configuration = identity["configuration"]
    hidden, layers = configuration.get("hidden_channels"), configuration.get("layers")
    if not _integer(hidden, minimum=1) or not _integer(layers, minimum=1):
        raise ValueError("transition legacy hidden/layers configuration is invalid")
    score = max(32, min(128, hidden // 2))
    edge_width = 4 * score + 8
    shapes = {
        "node_projection.weight": (score, hidden),
        "node_projection.bias": (score,),
        "context_projection.weight": (score, 2 * hidden + 8),
        "context_projection.bias": (score,),
        "score_norm.weight": (edge_width,),
        "score_norm.bias": (edge_width,),
        "score_network.0.weight": (2 * score, edge_width),
        "score_network.0.bias": (2 * score,),
        "score_network.2.weight": (score, 2 * score),
        "score_network.2.bias": (score,),
        "score_network.4.weight": (1, score),
    }
    expected = (
        {
            f"blocks.{layer}.operator.estimator.{suffix}": shape
            for layer in range(layers)
            for suffix, shape in shapes.items()
        }
        if identity["condition"] == "shared_dynamic_c"
        else {}
    )
    actual = {name: tuple(value.shape) for name, value in state.items() if C_NAMESPACE.match(name)}
    if actual != expected:
        raise ValueError(
            "transition source C estimator names/shapes are not the reviewed legacy MLP"
        )


def _execution_changes(original: dict, target: dict, declared: Any, mode: str) -> dict:
    if not isinstance(declared, dict) or set(declared) - _EXECUTION_FIELDS:
        raise ValueError("transition declared execution changes are invalid")
    if mode == "continue_fixed" and declared:
        raise ValueError("fixed continuation cannot change execution settings")
    for name, change in declared.items():
        if (
            not isinstance(change, dict)
            or set(change) != {"before", "after"}
            or change["before"] != original.get(name)
            or change["after"] != target.get(name)
            or change["before"] == change["after"]
        ):
            raise ValueError(f"transition execution change does not match identities: {name}")
        if name in {"batch_size", "sample_seed_batch_size", "workers", "loader_workers"}:
            minimum = 0 if "workers" in name else 1
            if (
                not _integer(change["before"], minimum=minimum)
                or not _integer(change["after"], minimum=minimum)
                or change["after"] < change["before"]
            ):
                raise ValueError(f"transition cannot reduce physical batches/workers: {name}")
    if declared:
        workers = target.get("loader_workers")
        if (
            not _integer(workers)
            or target.get("workers") != workers
            or target.get("persistent_workers") is not (workers > 0)
            or target.get("prefetch_factor") != (2 if workers else None)
        ):
            raise ValueError("transition changed worker metadata inconsistently")
    return copy.deepcopy(declared)


def _validate_identity(identity: Any, stored_hash: Any) -> str:
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("transition identity must contain finite plain JSON metadata") from error
    if not isinstance(identity, dict) or stored_hash != canonical_sha256(identity):
        raise ValueError("transition resume identity hash mismatch")
    if (
        identity.get("schema_version") != 1
        or identity.get("research_suite") != "conductance_graph_conditioned_v5"
    ):
        raise ValueError("transition resume identity schema/suite is invalid")
    protocol = identity.get("dataset_protocol")
    if (
        not isinstance(protocol, dict)
        or identity.get("dataset_protocol_sha256") != canonical_sha256(protocol)
        or not _digest(protocol.get("data_sha256"))
        or identity.get("cache_sha256") != protocol["data_sha256"]
        or not _digest(identity.get("initial_state_sha256"))
    ):
        raise ValueError("transition official data identity is invalid")
    source = identity.get("source_sha256")
    if not isinstance(source, dict) or any(not _digest(value) for value in source.values()):
        raise ValueError("transition implementation source map is invalid")
    revision = LEGACY_SOURCE_SNAPSHOTS.get(canonical_sha256(source))
    if revision is None:
        raise ValueError("transition source is not an exactly reviewed legacy V5 snapshot")
    configuration = identity.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("conductance_backend", "mlp") != "mlp"
        or configuration.get("training_schedule", "staged") != "staged"
        or configuration.get("optimizer") != "AdamW"
        or not _integer(configuration.get("model_seed"))
        or not _integer(configuration.get("epochs"), minimum=1)
    ):
        raise ValueError("transition source is not a legacy MLP/staged AdamW recipe")
    if identity.get("condition") not in {"fixed_c", "shared_dynamic_c"}:
        raise ValueError("transition source condition is invalid")
    if not isinstance(identity.get("runtime_versions"), dict) or not identity["runtime_versions"]:
        raise ValueError("transition runtime identity is missing")
    return revision


def inspect_transition_source(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Inspect a trusted user-selected legacy source without modifying it.

    The optional expected hash permits initial plan construction; execution
    must call prepare_transition_state with the plan-pinned SHA.
    """
    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("transition source must be a regular, non-symlink checkpoint")
    if expected_sha256 is not None and not _digest(expected_sha256):
        raise ValueError("transition expected SHA-256 is invalid")
    with source_path.open("rb") as stream:
        digest = _stream_sha256(stream)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("transition source checkpoint SHA-256 mismatch")
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        if _stream_sha256(stream) != digest:
            raise ValueError("transition source changed while it was being inspected")
    if not isinstance(saved, dict) or saved.get("schema_version") != 3:
        raise ValueError("transition requires last.pt selection-state schema 3")
    revision = _validate_identity(saved.get("resume_identity"), saved.get("resume_identity_sha256"))
    identity = saved["resume_identity"]
    epoch, history = saved.get("epoch"), saved.get("history")
    if (
        not _integer(epoch, minimum=1)
        or not isinstance(history, list)
        or len(history) != epoch
        or epoch > identity["configuration"]["epochs"]
        or any(
            not isinstance(row, dict) or row.get("epoch") != index
            for index, row in enumerate(history, 1)
        )
        or type(saved.get("complete")) is not bool
    ):
        raise ValueError("transition source epoch/history budget is invalid")
    state = saved.get("model_state")
    if not isinstance(state, dict) or not state or any(not isinstance(key, str) for key in state):
        raise ValueError("transition source model_state is malformed")
    for name, value in state.items():
        _tensor(value, name)
    _validate_legacy_c(state, identity)
    _optimizer_names(saved.get("optimizer_state"), state)
    for name in ("cpu_rng_state", "cuda_rng_state"):
        value = _tensor(saved.get(name), name)
        if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"transition {name} must be a nonempty uint8 vector")
    try:
        torch.Generator(device="cpu").set_state(saved["cpu_rng_state"])
    except RuntimeError as error:
        raise ValueError("transition CPU RNG state is invalid") from error
    counters = saved.get("effective_optimizer_steps_by_group")
    if (
        not _integer(saved.get("optimizer_steps"))
        or not isinstance(counters, dict)
        or set(counters) != set(_GROUPS)
        or any(not _integer(counters[name]) for name in _GROUPS)
    ):
        raise ValueError("transition optimizer step counters are invalid")
    for name in ("elapsed_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        value = saved.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (name != "elapsed_seconds" and not _integer(value))
        ):
            raise ValueError(f"transition cumulative resource counter is invalid: {name}")
    for moment in saved["optimizer_state"]["state"].values():
        if float(moment["step"]) > saved["optimizer_steps"]:
            raise ValueError(
                "transition per-parameter steps exceed the cumulative optimizer counter"
            )
    for group in saved["optimizer_state"]["param_groups"]:
        expected_steps = counters[group["name"]]
        if expected_steps > saved["optimizer_steps"]:
            raise ValueError("transition group steps exceed the cumulative optimizer counter")
        for identifier in group["params"]:
            moment = saved["optimizer_state"]["state"].get(identifier)
            # Connectivity is checked on the first active batch, not every
            # subsequent batch. A later sampled/edgeless branch may omit a
            # parameter gradient while the scheduled group counter advances.
            # Retain each actual Adam step; do not assume group-step equality.
            if (expected_steps == 0 and moment is not None) or (
                expected_steps > 0
                and (moment is None or not 0 < float(moment["step"]) <= expected_steps)
            ):
                raise ValueError(
                    "transition AdamW moments/steps disagree with effective group steps"
                )
    for metric_name, epoch_name in (
        ("best_metric", "best_epoch"),
        ("global_best_metric", "global_best_epoch"),
        ("joint_best_metric", "joint_best_epoch"),
    ):
        metric, selected_epoch = saved.get(metric_name), saved.get(epoch_name)
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or math.isnan(metric)
            or metric == math.inf
            or not _integer(selected_epoch)
            or selected_epoch > epoch
            or ((selected_epoch == 0) != (metric == -math.inf))
        ):
            raise ValueError(f"transition selection state is invalid: {metric_name}")
    best_hash = saved.get("best_checkpoint_sha256")
    if (saved["best_epoch"] == 0 and best_hash is not None) or (
        saved["best_epoch"] > 0 and not _digest(best_hash)
    ):
        raise ValueError("transition best-checkpoint hash is invalid")
    return {
        "saved": saved,
        "sha256": digest,
        "identity": identity,
        "source_epoch": epoch,
        "source_complete": saved["complete"],
        "legacy_revision": revision,
        "path": str(source_path.resolve()),
    }


def prepare_transition_state(
    path: str | Path,
    *,
    expected_sha256: str,
    target_model: torch.nn.Module,
    target_optimizer: torch.optim.Optimizer,
    target_identity: dict[str, Any],
    mode: str = "replace_c",
    additional_epochs: int = 0,
    declared_execution_changes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate everything before returning transplant states; mutate nothing.

    Replace-C starts post-transition selection history at source_epoch+1 and
    archives all source selections/history. Fixed continuation retains them.
    Epoch budgets are cumulative, never restarted or silently extended.
    """
    if not _digest(expected_sha256):
        raise ValueError("transition execution requires a plan-pinned source SHA-256")
    if mode not in {"replace_c", "continue_fixed"}:
        raise ValueError("transition mode must be replace_c or continue_fixed")
    if not _integer(additional_epochs):
        raise ValueError("transition additional_epochs must be an explicit nonnegative integer")
    inspected = inspect_transition_source(path, expected_sha256=expected_sha256)
    saved, source_identity = inspected["saved"], inspected["identity"]
    condition = "shared_dynamic_c" if mode == "replace_c" else "fixed_c"
    if source_identity["condition"] != condition or target_identity.get("condition") != condition:
        raise ValueError("transition mode/source/target conditions disagree")
    original = source_identity["configuration"]
    target = target_identity.get("configuration")
    if not isinstance(target, dict):
        raise ValueError("transition target configuration is missing")
    if target.get("conductance_backend") != ("optimization" if mode == "replace_c" else "mlp"):
        raise ValueError("transition target conductance backend is invalid")
    if target.get("training_schedule") != ("joint" if mode == "replace_c" else "staged"):
        raise ValueError("transition target training schedule is invalid")
    if target.get("epochs") != original["epochs"] + additional_epochs:
        raise ValueError(
            "transition target epoch budget differs from source plus explicit addition"
        )
    execution_changes = _execution_changes(original, target, declared_execution_changes or {}, mode)
    expected_request = {
        "source_checkpoint_sha256": expected_sha256,
        "mode": mode,
        "additional_epochs": additional_epochs,
        "declared_execution_changes": execution_changes,
    }
    request = copy.deepcopy(target_identity.get("transition_request", expected_request))
    request_path = request.pop("source_path", None)
    certificate_hash = request.pop("resource_certificate_sha256", None)
    certificate_path = request.pop("resource_certificate_path", None)
    if (certificate_hash is not None and not _digest(certificate_hash)) or (
        certificate_path is not None and not isinstance(certificate_path, str)
    ):
        raise ValueError("transition resource certificate request is malformed")
    if request != expected_request or (
        request_path is not None and str(Path(request_path).resolve()) != inspected["path"]
    ):
        raise ValueError("transition target request does not match the verified source/action")
    if "transition_request" in source_identity:
        raise ValueError("transition source cannot itself be a migrated experiment")
    allowed_identity = {
        "configuration",
        "source_sha256",
        "schedule",
        "initial_state_sha256",
        "transition_request",
    }
    mismatches = [
        key
        for key in set(source_identity) | set(target_identity)
        if key not in allowed_identity and source_identity.get(key) != target_identity.get(key)
    ]
    allowed_config = _C_CONFIGURATION | {"epochs"} | set(execution_changes)
    mismatches.extend(
        "configuration." + key
        for key in set(original) | set(target)
        if key not in allowed_config and original.get(key) != target.get(key)
    )
    if mismatches:
        raise ValueError("transition shared identity changed: " + ", ".join(sorted(mismatches)))
    if mode == "continue_fixed":
        expected_schedule = copy.deepcopy(source_identity["schedule"])
        if additional_epochs:
            expected_schedule[-1]["end_epoch"] += additional_epochs
            expected_schedule[-1]["length"] += additional_epochs
        if expected_schedule != target_identity.get("schedule"):
            raise ValueError(
                "fixed continuation must retain phases and extend only its final phase"
            )
    for source in _UNCHANGED_SOURCES:
        if source_identity["source_sha256"].get(source) != target_identity.get(
            "source_sha256", {}
        ).get(source):
            raise ValueError(f"transition changed a shared operator/data implementation: {source}")
    source_epoch = saved["epoch"]
    if source_epoch >= target["epochs"]:
        raise ValueError(
            "transition has no remaining epochs; explicitly authorize additional epochs"
        )
    if mode == "continue_fixed" and saved["complete"]:
        raise ValueError(
            "completed fixed C must be reused as a historical reference, not continued"
        )
    if not isinstance(target_optimizer, torch.optim.AdamW):
        raise ValueError("transition target optimizer must be AdamW")
    current = target_model.state_dict()
    source = saved["model_state"]
    shared_source = {name for name in source if not C_NAMESPACE.match(name)}
    shared_target = {name for name in current if not C_NAMESPACE.match(name)}
    if shared_source != shared_target:
        raise ValueError("transition shared model tensor names do not match exactly")
    if mode == "continue_fixed" and set(source) != set(current):
        raise ValueError("fixed continuation cannot change model tensor namespaces")
    new_state = {}
    for name, value in current.items():
        chosen = value.detach().cpu()
        if name in shared_source:
            old = source[name]
            if old.shape != value.shape or old.dtype != value.dtype:
                raise ValueError(f"transition shared tensor shape/dtype mismatch: {name}")
            chosen = old
        _tensor(chosen, f"target.{name}")
        new_state[name] = chosen.clone()
    source_names = _optimizer_names(saved["optimizer_state"], source)
    target_optim = copy.deepcopy(target_optimizer.state_dict())
    actual_parameter_names = {id(value): name for name, value in target_model.named_parameters()}
    for group in target_optimizer.param_groups:
        if any(
            actual_parameter_names.get(id(parameter)) != name
            for parameter, name in zip(group["params"], group["parameter_names"], strict=True)
        ):
            raise ValueError("transition target optimizer does not own the target model parameters")
    if target_optim["state"]:
        raise ValueError("transition target optimizer must be fresh before transplantation")
    target_names = _optimizer_names(
        target_optim,
        {name: value.detach().cpu() for name, value in target_model.named_parameters()},
    )
    if {name for name in source_names if not C_NAMESPACE.match(name)} != {
        name for name in target_names if not C_NAMESPACE.match(name)
    }:
        raise ValueError("transition shared optimizer parameter names differ")
    source_groups = {group["name"]: group for group in saved["optimizer_state"]["param_groups"]}
    for index, group in enumerate(target_optim["param_groups"]):
        if group["name"] == "conductance" and mode == "replace_c":
            continue
        old_group = source_groups.get(group["name"])
        if old_group is None or set(old_group["parameter_names"]) != set(group["parameter_names"]):
            raise ValueError("transition shared optimizer groups differ")
        replacement = copy.deepcopy(old_group)
        replacement["params"] = list(group["params"])
        replacement["parameter_names"] = list(group["parameter_names"])
        target_optim["param_groups"][index] = replacement
    retained_moments = []
    for name in shared_source:
        old_identifier, new_identifier = source_names[name], target_names[name]
        if old_identifier in saved["optimizer_state"]["state"]:
            target_optim["state"][new_identifier] = copy.deepcopy(
                saved["optimizer_state"]["state"][old_identifier]
            )
            retained_moments.append(name)
    counters = {
        "optimizer_steps": saved["optimizer_steps"],
        "effective_optimizer_steps_by_group": copy.deepcopy(
            saved["effective_optimizer_steps_by_group"]
        ),
        "elapsed_seconds": saved.get("elapsed_seconds", 0.0),
        "peak_cuda_allocated_bytes": saved.get("peak_cuda_allocated_bytes", 0),
        "peak_cuda_reserved_bytes": saved.get("peak_cuda_reserved_bytes", 0),
    }
    selection = {key: copy.deepcopy(saved[key]) for key in _SELECTION}
    if mode == "replace_c":
        counters["effective_optimizer_steps_by_group"]["conductance"] = 0
        selection = {
            "best_metric": -math.inf,
            "best_epoch": 0,
            "best_checkpoint_sha256": None,
            "global_best_metric": -math.inf,
            "global_best_epoch": 0,
            "joint_best_metric": -math.inf,
            "joint_best_epoch": 0,
            "first_c_gradient": None,
        }
    dropped = sorted(name for name in source if C_NAMESPACE.match(name))
    initialized = sorted(name for name in current if C_NAMESPACE.match(name))
    archived_selection = {key: copy.deepcopy(saved[key]) for key in _SELECTION}
    applicability = {}
    for name in ("best_metric", "global_best_metric", "joint_best_metric"):
        if not math.isfinite(archived_selection[name]):
            archived_selection[name] = None
            applicability[name] = "not_yet_eligible_or_no_checkpoint_selected"
        else:
            applicability[name] = "observed_historical_validation_metric"
    provenance = {
        "schema_version": TRANSITION_SCHEMA,
        "mode": mode,
        "source_path": inspected["path"],
        "source_sha256": inspected["sha256"],
        "source_legacy_revision": inspected["legacy_revision"],
        "source_checkpoint_sha256": inspected["sha256"],
        "source_identity": copy.deepcopy(source_identity),
        "source_identity_sha256": saved["resume_identity_sha256"],
        "source_epoch": source_epoch,
        "source_complete": saved["complete"],
        "source_total_epochs": original["epochs"],
        "target_total_epochs": target["epochs"],
        "source_epochs_requested": original["epochs"],
        "epoch_offset": source_epoch if mode == "replace_c" else 0,
        "declared_execution_changes": execution_changes,
        "additional_epochs": additional_epochs,
        "retained_source_initial_state_sha256": source_identity["initial_state_sha256"],
        "retained_shared_tensor_names": sorted(shared_source),
        "retained_shared_tensor_count": len(shared_source),
        "retained_shared_parameter_elements": sum(source[name].numel() for name in shared_source),
        "retained_optimizer_state_names": sorted(retained_moments),
        "dropped_c_tensor_names": dropped,
        "initialized_c_tensor_names": initialized,
        "source_optimizer_steps": saved["optimizer_steps"],
        "source_effective_optimizer_steps_by_group": copy.deepcopy(
            saved["effective_optimizer_steps_by_group"]
        ),
        "source_selection_state": archived_selection,
        "source_selection_applicability": applicability,
        "historical_metrics_are_new_c_metrics": False,
        "test_labels_used": False,
    }
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("transition provenance must contain finite plain JSON metadata") from error
    return {
        "model_state": new_state,
        "optimizer_state": target_optim,
        "rng_state": {name: saved[name].clone() for name in ("cpu_rng_state", "cuda_rng_state")},
        "history": [] if mode == "replace_c" else copy.deepcopy(saved["history"]),
        "source_history": copy.deepcopy(saved["history"]),
        "epoch_offset": source_epoch if mode == "replace_c" else 0,
        "start_epoch": source_epoch + 1,
        "total_epochs": target["epochs"],
        "selection_state": selection,
        "counters": counters,
        "provenance": provenance,
    }
