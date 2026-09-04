"""Fail-closed report for the V4 relative-C x spatial-W factorial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

from ..ablation.report import (
    REPORT_FILENAMES,
    _atomic_write,
    _cell,
    _contained,
    _display,
    _finite_number,
    _integer,
    _load_child,
    _pair_metadata,
    _reject_nonfinite_json,
    _same,
)
from .protocol import (
    BATCH_SIZE_BY_DATASET,
    COMMON,
    CONDITIONS,
    DATASETS,
    METRIC_BY_DATASET,
    PARAMETERIZATION,
    SUITE,
)

SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
INTERVENTIONS = {
    "mean_c",
    "shuffled_c",
    "ones_c",
    "identity_w",
    "ones_c_identity_w",
    "propagation_off",
}
FACTORIAL_ORDER = (
    "fixed_c_identity_w",
    "relative_c_identity_w",
    "fixed_c_spatial_w",
    "relative_c_spatial_w",
)
CAVEATS = [
    "n=1; exploratory validation-only factorial. Test is not evaluated; no CI, p-value, "
    "seed standard deviation, SOTA or general optimality claim.",
    "All four arms train freshly from a matched shared-backbone initial state. PPI uses the "
    "official "
    "20/2/2 inductive graph split: train 20 and validation 2 run at batch 2 with "
    "BCEWithLogits and global logit>0 node-label micro-F1; test 2 is not scored. Other "
    "datasets use their official transductive masks and accuracy. No V3 checkpoint or score "
    "is reused, and a V3-to-V4 score difference is not a one-factor causal contrast.",
    "V3/V4 do not implement a conventional eigendecomposition-based spectral GNN. Relative C "
    "adapts the weighted graph operator; W is a shared spatial message-channel transform.",
    "The five factorial contrasts are descriptive configuration differences within V4. Early "
    "stopping epochs can differ and single-seed validation does not establish population effects.",
    "C and W can compensate across layers. C spread, gamma/tau, W-I distance, singular values "
    "or gradient norms alone do not prove that either mechanism is useful.",
    "Mean-C and C=1 are algebraically redundant under symmetric weighted-degree normalization. "
    "Their separate CUDA-forward logit allclose is informational and non-gating because scatter "
    "is not bitwise deterministic; these interventions are not two effects.",
    "Checkpoint interventions use separate validation forwards without retraining. They measure "
    "selected-checkpoint reliance; the four fresh arms provide the training contrasts.",
    "Elapsed time and peak CUDA memory include diagnostics, interventions, checkpoint/history IO "
    "as defined by the trainer; they are not isolated kernel benchmarks.",
    "W-on arms have more active parameters and optimizer state than W-off arms. This factorial "
    "does not parameter-budget-match unrelated architectures.",
    "Sparse exact edge chunking processes every selected graph in full; only PPI batches whole "
    "inductive graphs. Identical seeds and initial hashes do not make CUDA scatter trajectories "
    "bitwise deterministic.",
]


class ComparisonIntegrityError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("Conductance/spatial V4 integrity failed: " + "; ".join(report["errors"]))


def _source_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("nonempty source_sha256 object is required")
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
        ):
            raise ValueError("source_sha256 requires relative paths and SHA-256 digests")
    return value


def _current_source_hashes() -> dict[str, str]:
    """Recompute the runner's exact source inventory for standalone reports."""

    from scripts.run_conductance_v4 import _source_snapshot

    return _source_hashes(_source_snapshot().get("sha256"))


def _names(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label}: expected parameter-name list")
    if len(set(value)) != len(value):
        raise ValueError(f"{label}: duplicate parameter names")
    return value


def _nested(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _nonnegative_optional(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _require_close(value: Any, expected: float, label: str, *, atol: float = 1.0e-9) -> None:
    actual = _finite_number(value, label)
    if abs(actual - expected) > atol:
        raise ValueError(f"{label} must equal {expected}")


def _validate_optimizer(child: dict[str, Any], config: dict[str, Any], condition: str) -> None:
    if child.get("optimizer") != "AdamW":
        raise ValueError("optimizer must be AdamW")
    spec = CONDITIONS[condition]
    c_active = spec["gate_mode"] == "relative"
    w_active = spec["spatial_mode"] == "learned"
    active = _names(child.get("trainable_parameter_names"), "trainable_parameter_names")
    frozen = _names(child.get("frozen_parameter_names"), "frozen_parameter_names")
    active_set, frozen_set = set(active), set(frozen)
    if not active or active_set & frozen_set:
        raise ValueError("active/frozen parameter names overlap or active list is empty")
    if frozen_set:
        raise ValueError("V4 fixed controls must not register frozen parameters")
    layers = config["layers"]
    alpha = {f"operators.{index}.raw_alpha" for index in range(layers)}
    spatial = {f"operators.{index}.message_transform.weight" for index in range(layers)}
    estimator = {name for name in active_set | frozen_set if ".estimator." in name}
    if not alpha <= active_set:
        raise ValueError("every layer requires active alpha")
    if bool(estimator) != c_active or (estimator and not estimator <= active_set):
        raise ValueError("conductance estimator presence disagrees with condition")
    present_spatial = spatial & active_set
    if bool(present_spatial) != w_active or (w_active and present_spatial != spatial):
        raise ValueError("spatial W presence disagrees with condition")

    expected_groups = {"backbone", "raw_scalars"}
    if c_active:
        expected_groups.add("conductance_gate")
    if w_active:
        expected_groups.add("spatial_w")
    groups = child.get("optimizer_groups")
    if not isinstance(groups, list):
        raise ValueError("optimizer_groups must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    all_names: list[str] = []
    parameter_count = 0
    for group in groups:
        if not isinstance(group, dict) or group.get("name") not in expected_groups:
            raise ValueError("unexpected V4 optimizer parameter group")
        name = group["name"]
        if name in indexed:
            raise ValueError("duplicate optimizer parameter group")
        indexed[name] = group
        names = _names(group.get("parameter_names"), f"{name}.parameter_names")
        if not names:
            raise ValueError("optimizer groups must not be empty")
        size = _integer(group.get("parameter_count"), f"{name}.parameter_count", minimum=1)
        all_names.extend(names)
        parameter_count += size
        lr = config["lr"] * (config["gate_lr_multiplier"] if name == "conductance_gate" else 1.0)
        wd = config["weight_decay"] if name in {"backbone", "spatial_w"} else 0.0
        if not _same(group.get("lr"), lr) or not _same(group.get("weight_decay"), wd):
            raise ValueError(f"optimizer {name} lr/weight_decay mismatch")
        if name == "raw_scalars":
            required = set(alpha)
            if c_active:
                required |= {
                    f"operators.{index}.estimator.raw_{control}"
                    for index in range(layers)
                    for control in ("gamma", "tau")
                }
            if set(names) != required or size != len(names):
                raise ValueError("raw scalar controls mismatch")
        elif name == "spatial_w":
            if set(names) != spatial:
                raise ValueError("spatial_w group does not contain exactly the layer W matrices")
        elif name == "conductance_gate":
            controls = {n for n in estimator if n.endswith((".raw_gamma", ".raw_tau"))}
            if set(names) != estimator - controls:
                raise ValueError("conductance_gate group does not cover the non-scalar estimator")
        elif name == "backbone" and any(
            name.endswith((".raw_alpha", ".raw_gamma", ".raw_tau")) or ".message_transform." in name
            for name in names
        ):
            raise ValueError("backbone group contains V4 factor parameters")
    if (
        set(indexed) != expected_groups
        or len(set(all_names)) != len(all_names)
        or set(all_names) != active_set
    ):
        raise ValueError("optimizer groups do not cover exactly the trainable parameters")
    if parameter_count != child["trainable_parameters"]:
        raise ValueError("optimizer parameter counts disagree with trainable_parameters")


def _best_training_observation(child: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    trajectory = child["diagnostics"].get("train_trajectory")
    if not isinstance(trajectory, list) or any(not isinstance(row, dict) for row in trajectory):
        raise ValueError("actual training trajectory must be recorded")
    epochs = [_integer(row.get("epoch"), "training epoch", minimum=1) for row in trajectory]
    if len(set(epochs)) != len(epochs):
        raise ValueError("duplicate actual-training epoch observations")
    selected = [row for row in trajectory if row["epoch"] == child["best_epoch"]]
    if len(selected) != 1:
        raise ValueError("selected epoch is missing from actual training observations")
    record = selected[0]
    batches_per_epoch = _integer(
        child.get("train_batches_per_epoch"), "train_batches_per_epoch", minimum=1
    )
    expected_batches = 10 if child["dataset"] == "ppi" else 1
    if batches_per_epoch != expected_batches:
        raise ValueError("train_batches_per_epoch disagrees with official data protocol")
    if (
        child.get("optimizer_steps") != child["epochs_run"] * batches_per_epoch
        or child.get("best_checkpoint_optimizer_steps")
        != child["best_epoch"] * batches_per_epoch
    ):
        raise ValueError("optimizer step counts disagree with actual minibatch count")
    expected = {
        "scope": (
            "first_actual_training_minibatch_only"
            if child["dataset"] == "ppi"
            else "full_graph_train_mask"
        ),
        "mode": "train_dropout_on",
        "stage": "after_task_backward_before_optimizer_step",
        "batch_index": 0,
        "optimizer_steps_before_batch": (child["best_epoch"] - 1) * batches_per_epoch,
    }
    for key, value in expected.items():
        if not _same(record.get(key), value):
            raise ValueError(f"selected training observation {key} mismatch")
    layers = record.get("layers")
    if not isinstance(layers, list) or len(layers) != config["layers"]:
        raise ValueError("selected training layer observations are incomplete")
    spec = CONDITIONS[child["condition"]]
    indices = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("invalid training layer observation")
        indices.append(_integer(layer.get("layer"), "training layer"))
        gate = layer.get("gate_gradient_norm")
        spatial = layer.get("spatial_gradient_norm")
        if spec["gate_mode"] == "relative":
            _nonnegative_optional(gate, "actual gate task gradient")
            if gate is None:
                raise ValueError("active conductance gate lacks a task gradient")
        elif gate is not None:
            raise ValueError("frozen conductance gate has a task gradient")
        if spec["spatial_mode"] == "learned":
            _nonnegative_optional(spatial, "actual spatial-W task gradient")
            if spatial is None:
                raise ValueError("active spatial W lacks a task gradient")
        elif spatial is not None:
            raise ValueError("frozen identity W has a task gradient")
    if sorted(indices) != list(range(config["layers"])):
        raise ValueError("selected training layer indices are missing or duplicated")
    return record


def _validate_diagnostics(child: dict[str, Any], config: dict[str, Any]) -> None:
    diagnostics = child.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("V4 diagnostics are required")
    observations = {}
    expected_metric = METRIC_BY_DATASET[child["dataset"]]
    expected_prediction_rule = (
        "logit_gt_zero_node_label" if child["dataset"] == "ppi" else "argmax_node_class"
    )
    expected_validation_graphs = 2 if child["dataset"] == "ppi" else 1
    for name in ("initial_validation", "best_validation", "final_validation"):
        observation = diagnostics.get(name)
        if (
            not isinstance(observation, dict)
            or observation.get("mode") != "eval"
            or observation.get("split") != "validation"
            or observation.get("metric_name") != expected_metric
            or observation.get("prediction_rule") != expected_prediction_rule
            or observation.get("validation_graph_count") != expected_validation_graphs
        ):
            raise ValueError(f"{name} diagnostics task/split contract mismatch")
        observations[name] = observation
    best = observations["best_validation"]
    layers = best.get("layers")
    if not isinstance(layers, list) or len(layers) != config["layers"]:
        raise ValueError("best_validation layer diagnostics are incomplete")
    spec = CONDITIONS[child["condition"]]
    indices = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("invalid selected-checkpoint layer diagnostics")
        indices.append(_integer(layer.get("layer"), "diagnostic.layer"))
        _finite_number(layer.get("alpha"), "alpha", unit_interval=True)
        c_active = spec["gate_mode"] == "relative"
        if c_active:
            _finite_number(layer.get("gamma"), "gamma", unit_interval=True)
            if _finite_number(layer.get("tau"), "tau") <= 0:
                raise ValueError("tau must be positive")
            if not isinstance(layer.get("estimator_parameter_count"), int) or layer[
                "estimator_parameter_count"
            ] <= 0:
                raise ValueError("active estimator parameter count is required")
        elif (
            layer.get("gamma") is not None
            or layer.get("tau") is not None
            or layer.get("estimator_parameter_count") != 0
            or layer.get("parameter_free_fixed_control") is not True
        ):
            raise ValueError("fixed C diagnostics must declare a parameter-free estimator")
        for path in (
            ("score", "std"),
            ("conductance", "cv"),
            ("log_conductance", "std"),
            ("weighted_degree", "quantiles", "p50"),
            ("weighted_degree", "quantiles", "p99"),
            ("weighted_degree", "max_over_median"),
            ("relative_message_transform_change",),
            ("relative_conv_change",),
            ("gate_parameter_norm",),
        ):
            _nonnegative_optional(_nested(layer, *path), ".".join(path))
        spatial = layer.get("spatial_weight")
        if not isinstance(spatial, dict):
            raise ValueError("spatial_weight diagnostics are required")
        if layer.get("estimator_trainable") is not c_active:
            raise ValueError("conductance estimator trainable metadata mismatch")
        if not c_active:
            conductance = layer.get("conductance")
            log_conductance = layer.get("log_conductance")
            if not isinstance(conductance, dict) or not isinstance(log_conductance, dict):
                raise ValueError("fixed-C value diagnostics are required")
            for key, expected in (
                ("mean", 1.0),
                ("std", 0.0),
                ("cv", 0.0),
                ("min", 1.0),
                ("max", 1.0),
            ):
                _require_close(conductance.get(key), expected, f"fixed conductance.{key}")
            for key in ("mean", "std", "min", "max"):
                _require_close(log_conductance.get(key), 0.0, f"fixed log-conductance.{key}")
        if spatial.get("spatial_mode") != spec["spatial_mode"] or spatial.get("trainable") is not (
            spec["spatial_mode"] == "learned"
        ):
            raise ValueError("spatial_weight mode/trainable metadata mismatch")
        if spatial.get("parameter_present") is not (spec["spatial_mode"] == "learned"):
            raise ValueError("spatial weight parameter presence disagrees with condition")
        for key in (
            "parameter_norm",
            "identity_distance_frobenius",
            "identity_relative_distance",
        ):
            _nonnegative_optional(spatial.get(key), f"spatial_weight.{key}")
        singular = spatial.get("singular_values")
        if not isinstance(singular, dict):
            raise ValueError("spatial singular-value diagnostics are required")
        if (
            _integer(singular.get("count"), "singular count", minimum=1)
            != config["hidden_channels"]
        ):
            raise ValueError("spatial singular-value count differs from hidden width")
        for key in ("min", "max", "mean", "std", "condition_number"):
            _nonnegative_optional(singular.get(key), f"singular_values.{key}")
        if spec["spatial_mode"] == "fixed_identity":
            _require_close(
                layer.get("relative_message_transform_change"),
                0.0,
                "fixed identity message change",
            )
            _require_close(
                spatial.get("identity_distance_frobenius"),
                0.0,
                "fixed identity W distance",
            )
            _require_close(
                spatial.get("identity_relative_distance"),
                0.0,
                "fixed relative identity W distance",
            )
            for key, expected in (
                ("min", 1.0),
                ("max", 1.0),
                ("mean", 1.0),
                ("std", 0.0),
                ("condition_number", 1.0),
            ):
                _require_close(singular.get(key), expected, f"fixed identity singular {key}")
    if sorted(indices) != list(range(config["layers"])):
        raise ValueError("diagnostic layer indices are missing or duplicated")
    _best_training_observation(child, config)

    audit = diagnostics.get("best_checkpoint_interventions")
    if (
        not isinstance(audit, dict)
        or audit.get("status") != "passed"
        or audit.get("scope") != "validation_selected_best_checkpoint_only"
    ):
        raise ValueError("selected-checkpoint validation interventions are required")
    if audit.get("layers") != "all_layers_simultaneously":
        raise ValueError("checkpoint interventions must apply to all layers simultaneously")
    if audit.get("normalization_recomputed_for_c_interventions") is not True:
        raise ValueError("C interventions must recompute symmetric normalization")
    if (
        audit.get("metric_name") != expected_metric
        or audit.get("prediction_rule") != expected_prediction_rule
        or audit.get("validation_graph_count") != expected_validation_graphs
    ):
        raise ValueError("checkpoint intervention task contract mismatch")
    if _integer(audit.get("shuffle_seed"), "intervention shuffle_seed") != child["model_seed"]:
        raise ValueError("intervention shuffle_seed must equal model_seed")
    original = audit.get("original")
    if not isinstance(original, dict):
        raise ValueError("intervention original score is missing")
    score = _finite_number(original.get("validation"), "intervention original", unit_interval=True)
    if abs(score - child["validation"]) > 1.0e-7:
        raise ValueError("intervention original differs from selected validation")
    rows = audit.get("rows")
    if not isinstance(rows, list) or len(rows) != len(INTERVENTIONS):
        raise ValueError("all six selected-checkpoint interventions are required")
    names = []
    for row in rows:
        if not isinstance(row, dict) or row.get("intervention") not in INTERVENTIONS:
            raise ValueError("unknown V4 checkpoint intervention")
        names.append(row["intervention"])
        value = _finite_number(row.get("validation"), "intervention validation", unit_interval=True)
        delta = _finite_number(row.get("percentage_points"), "intervention delta")
        if abs(delta - 100.0 * (value - score)) > 1.0e-8:
            raise ValueError("intervention percentage-points delta mismatch")
        _finite_number(
            row.get("changed_prediction_fraction"), "changed predictions", unit_interval=True
        )
        _nonnegative_optional(row.get("logit_mean_absolute_delta"), "logit delta")
        if row.get("fresh_training") is not False:
            raise ValueError("checkpoint interventions must not be labeled fresh training")
        if row.get("intervention_kind") != "read_only_selected_checkpoint":
            raise ValueError("checkpoint intervention kind mismatch")
    if set(names) != INTERVENTIONS or len(set(names)) != len(names):
        raise ValueError("missing/duplicate V4 checkpoint interventions")
    numeric = audit.get("mean_c_numeric_check")
    if not isinstance(numeric, dict) or numeric.get("comparison") != "mean_c_vs_ones_c":
        raise ValueError("mean-C/C=1 numerical check is required")
    if "passed" in numeric:
        raise ValueError("legacy mean-C numeric passed field is not allowed")
    if numeric.get("role") != "informational_non_gating":
        raise ValueError("mean-C numerical check must be informational and non-gating")
    if numeric.get("separate_full_graph_forwards") is not True:
        raise ValueError("mean-C numerical check must identify separate full-graph forwards")
    if not isinstance(numeric.get("within_declared_tolerance"), bool):
        raise ValueError("mean-C within-tolerance observation must be boolean")
    _require_close(
        numeric.get("allclose_rtol"),
        1.0e-5,
        "mean-C numerical relative tolerance",
        atol=0.0,
    )
    _require_close(
        numeric.get("allclose_atol"),
        1.0e-6,
        "mean-C numerical absolute tolerance",
        atol=0.0,
    )
    mean_delta = _finite_number(
        numeric.get("logit_mean_absolute_delta"), "mean-C mean absolute logit delta"
    )
    maximum_delta = _finite_number(
        numeric.get("logit_max_absolute_delta"), "mean-C maximum absolute logit delta"
    )
    if mean_delta < 0 or maximum_delta < 0:
        raise ValueError("mean-C numerical deltas must be nonnegative")
    if maximum_delta < mean_delta:
        raise ValueError("mean-C maximum absolute delta must not be below its mean")
    _finite_number(
        numeric.get("changed_prediction_fraction"),
        "mean-C changed predictions",
        unit_interval=True,
    )
    replacement_contracts = numeric.get("replacement_contracts")
    expected_contracts = {
        "mean_c": "graph_constant_positive",
        "ones_c": "exact_one",
    }
    if not isinstance(replacement_contracts, dict) or set(replacement_contracts) != set(
        expected_contracts
    ):
        raise ValueError("mean-C numerical check requires exactly mean_c/ones_c contracts")
    topology = child.get("topology")
    topology_edge_count = _integer(
        (
            topology.get("split_num_edges", {}).get("validation")
            if child["dataset"] == "ppi" and isinstance(topology, dict)
            else topology.get("num_edges") if isinstance(topology, dict) else None
        ),
        "topology validation edge count",
    )
    expected_edge_counts = None
    for name, expected_contract in expected_contracts.items():
        replacement = replacement_contracts[name]
        if not isinstance(replacement, dict) or set(replacement) != {
            "contract",
            "satisfied",
            "layers_checked",
            "edge_counts",
        }:
            raise ValueError(f"{name} replacement contract metadata is incomplete")
        if replacement.get("contract") != expected_contract:
            raise ValueError(f"{name} replacement contract kind is invalid")
        if replacement.get("satisfied") is not True:
            raise ValueError(f"{name} replacement contract must be satisfied")
        layers_checked = _integer(
            replacement.get("layers_checked"), f"{name} replacement layers_checked"
        )
        if layers_checked != config["layers"]:
            raise ValueError(f"{name} replacement must check every configured layer")
        edge_counts = replacement.get("edge_counts")
        if not isinstance(edge_counts, list) or len(edge_counts) != layers_checked:
            raise ValueError(f"{name} replacement requires one edge count per layer")
        checked_edge_counts = [
            _integer(value, f"{name} replacement edge count") for value in edge_counts
        ]
        if any(value != topology_edge_count for value in checked_edge_counts):
            raise ValueError(f"{name} replacement edge counts must match the bound topology")
        if expected_edge_counts is None:
            expected_edge_counts = checked_edge_counts
        elif checked_edge_counts != expected_edge_counts:
            raise ValueError("mean-C and C=1 replacements must cover identical layer edges")


def _load(
    root: Path,
    job: dict[str, Any],
    config: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    path = _contained(job.get("metrics_path"), root, "metrics")
    digest = job.get("metrics_sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError("job.metrics_sha256 is required")
    try:
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot read child metrics: {exc}") from exc
    if actual_digest != digest.lower():
        raise ValueError("metrics SHA-256 mismatch")
    dataset = job["dataset"]
    child_config = {
        key: value
        for key, value in config.items()
        if key not in {"batch_size_by_dataset", "workers_by_dataset"}
    }
    child_config["batch_size"] = BATCH_SIZE_BY_DATASET[dataset]
    child_config["workers"] = config.get("workers", 0) if dataset == "ppi" else 0
    child = _load_child(root, job, child_config, suite=SUITE, conditions=CONDITIONS)
    spec = CONDITIONS[job["condition"]]
    for key, expected in (
        ("gate_mode", spec["gate_mode"]),
        ("spatial_mode", spec["spatial_mode"]),
        ("parameterization", PARAMETERIZATION),
        ("source_sha256", source_hashes),
    ):
        if not _same(child.get(key), expected):
            raise ValueError(f"{key} mismatch")
    if "gate_mode" in child["configuration"] or "spatial_mode" in child["configuration"]:
        raise ValueError("factor modes belong in arm metadata, not held-fixed configuration")
    topology = child.get("topology")
    expected_split = (
        "official_inductive_graph_split"
        if dataset == "ppi"
        else "official_time_split"
        if dataset == "ogbn-arxiv"
        else "official_public_masks"
    )
    expected_task = (
        "multi_label_node_classification" if dataset == "ppi" else "node_classification"
    )
    expected_metric = METRIC_BY_DATASET[dataset]
    protocol = child.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("dataset") != dataset
        or protocol.get("split") != expected_split
        or protocol.get("task") != expected_task
        or protocol.get("metric") != expected_metric
    ):
        raise ValueError("cached protocol does not match the official V1 dataset contract")
    if dataset == "ppi":
        expected_keys = {
            "scope",
            "split_graph_counts",
            "split_num_nodes",
            "split_num_edges",
            "split_incidence_sha256",
        }
        if not isinstance(topology, dict) or set(topology) != expected_keys:
            raise ValueError("PPI topology must fingerprint official train/validation graphs")
        if topology["scope"] != "official_train_and_validation_graphs":
            raise ValueError("PPI topology scope mismatch")
        if topology["split_graph_counts"] != {"train": 20, "validation": 2}:
            raise ValueError("PPI topology must contain the official 20/2 graph split")
        for key in ("split_num_nodes", "split_num_edges"):
            value = topology[key]
            if not isinstance(value, dict) or set(value) != {"train", "validation"}:
                raise ValueError(f"PPI topology {key} split metadata is incomplete")
            for split, count in value.items():
                _integer(count, f"topology.{key}.{split}", minimum=1)
        digests = topology["split_incidence_sha256"]
        if (
            not isinstance(digests, dict)
            or set(digests) != {"train", "validation"}
            or any(
                not isinstance(value, str) or not SHA256.fullmatch(value)
                for value in digests.values()
            )
        ):
            raise ValueError("PPI split incidence fingerprints are invalid")
        if protocol.get("split_counts") != {"train": 20, "validation": 2, "test": 2}:
            raise ValueError("PPI cached protocol is not the official 20/2/2 graph split")
    else:
        if not isinstance(topology, dict) or set(topology) != {
            "num_nodes",
            "num_edges",
            "incidence_sha256",
        }:
            raise ValueError("topology must contain num_nodes, num_edges and incidence_sha256")
        _integer(topology["num_nodes"], "topology.num_nodes", minimum=1)
        _integer(topology["num_edges"], "topology.num_edges")
        if not isinstance(topology["incidence_sha256"], str) or not SHA256.fullmatch(
            topology["incidence_sha256"]
        ):
            raise ValueError("topology.incidence_sha256 must be a SHA-256 digest")
    total = _integer(child.get("total_parameters"), "total_parameters", minimum=1)
    trainable = _integer(child.get("trainable_parameters"), "trainable_parameters", minimum=1)
    frozen = _integer(child.get("frozen_parameters"), "frozen_parameters")
    if total != trainable + frozen:
        raise ValueError("total parameter count differs from trainable+frozen")
    if frozen != 0:
        raise ValueError("V4 fixed controls must be parameter-free")
    _validate_optimizer(child, child_config, job["condition"])
    _validate_diagnostics(child, child_config)
    best_epoch = _integer(child.get("best_epoch"), "best_epoch", minimum=1)
    stop_epoch = _integer(child.get("stop_epoch"), "stop_epoch", minimum=best_epoch)
    if stop_epoch != child.get("epochs_run"):
        raise ValueError("stop_epoch must equal epochs_run")
    if child.get("stopping_reason") not in {"patience", "max_epochs"}:
        raise ValueError("unknown stopping_reason")
    if (
        child.get("validation_batches") != 1
        or child.get("validation_graphs") != (2 if dataset == "ppi" else 1)
    ):
        raise ValueError("validation coverage disagrees with official data protocol")
    for key in (
        "selection_loop_seconds",
        "post_selection_diagnostics_seconds",
        "elapsed_seconds",
    ):
        if _finite_number(child.get(key), key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if child["elapsed_seconds"] + 1.0e-9 < (
        child["selection_loop_seconds"] + child["post_selection_diagnostics_seconds"]
    ):
        raise ValueError("elapsed_seconds is shorter than its recorded timing components")
    timing = child.get("epoch_timing")
    if not isinstance(timing, dict):
        raise ValueError("epoch_timing summary is required")
    if _integer(timing.get("count"), "epoch_timing.count", minimum=1) != child["epochs_run"]:
        raise ValueError("epoch_timing.count must equal epochs_run")
    for key in (
        "total_seconds",
        "mean_seconds",
        "median_seconds",
        "p90_seconds",
        "min_seconds",
        "max_seconds",
    ):
        if _finite_number(timing.get(key), f"epoch_timing.{key}") < 0:
            raise ValueError(f"epoch_timing.{key} must be nonnegative")
    if not (
        timing["min_seconds"]
        <= timing["median_seconds"]
        <= timing["p90_seconds"]
        <= timing["max_seconds"]
    ):
        raise ValueError("epoch_timing quantiles are not ordered")
    if timing.get("quantile_method") != "linear_order_statistic" or not isinstance(
        timing.get("scope"), str
    ):
        raise ValueError("epoch_timing policy metadata mismatch")
    for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        _integer(child.get(key), key)
    if not isinstance(child.get("versions"), dict) or not child["versions"]:
        raise ValueError("missing runtime versions")
    if not isinstance(child.get("gpu"), str) or not child["gpu"]:
        raise ValueError("missing GPU identity")
    if child["protocol"].get("data_sha256") != child["cache_sha256"]:
        raise ValueError("cache_sha256 disagrees with dataset protocol")
    shared_hash = child.get("shared_backbone_initial_state_sha256")
    if not isinstance(shared_hash, str) or not SHA256.fullmatch(shared_hash):
        raise ValueError("shared backbone initialization SHA-256 is required")
    return child


def _comparison_metadata(metrics: dict[str, Any]) -> dict[str, Any]:
    metadata = _pair_metadata(metrics)
    metadata.pop("initial_state_sha256")
    metadata["shared_backbone_initial_state_sha256"] = metrics[
        "shared_backbone_initial_state_sha256"
    ]
    return metadata


def _factorial(scores: dict[str, float]) -> dict[str, dict[str, float]]:
    y00 = scores["fixed_c_identity_w"]
    y10 = scores["relative_c_identity_w"]
    y01 = scores["fixed_c_spatial_w"]
    y11 = scores["relative_c_spatial_w"]
    values = {
        "c_given_w_off": ("C | W off", y10 - y00),
        "c_given_w_on": ("C | W on", y11 - y01),
        "w_given_c_fixed": ("W | C fixed", y01 - y00),
        "w_given_c_relative": ("W | C relative", y11 - y10),
        "interaction": ("interaction", y11 - y10 - y01 + y00),
    }
    return {
        key: {"label": label, "score_delta": delta, "percentage_points": 100.0 * delta}
        for key, (label, delta) in values.items()
    }


def _best_layers(diagnostics: Any) -> list[dict[str, Any]]:
    if not isinstance(diagnostics, dict):
        return []
    best = diagnostics.get("best_validation")
    return best.get("layers", []) if isinstance(best, dict) else []


def build_comparison(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        config = {}
        errors.append("manifest.config must be an object")
    datasets = config.get("datasets", [])
    if (
        not isinstance(datasets, list)
        or not datasets
        or any(not isinstance(dataset, str) or dataset not in DATASETS for dataset in datasets)
        or len(set(datasets)) != len(datasets)
    ):
        datasets = []
        errors.append("datasets must list unique supported V4 datasets")
    for key, expected in (
        ("schema_version", 1),
        ("suite", SUITE),
        ("conditions", CONDITIONS),
        ("source_integrity_valid", True),
    ):
        if not _same(manifest.get(key), expected):
            errors.append(f"manifest.{key} mismatch")
    if manifest.get("status") not in {"running", "failed", "passed"}:
        errors.append("invalid manifest.status")
    sources: dict[str, str] = {}
    try:
        source_metadata = manifest.get("sources")
        if not isinstance(source_metadata, dict):
            raise ValueError("manifest.sources must be an object")
        sources = _source_hashes(source_metadata.get("sha256"))
        _integer(config.get("model_seed"), "model_seed")
        for key in ("epochs", "patience", "edge_chunk_size"):
            _integer(config.get(key), key, minimum=1)
        batch_sizes = config.get("batch_size_by_dataset")
        expected_batch_sizes = {
            dataset: BATCH_SIZE_BY_DATASET[dataset] for dataset in datasets
        }
        if not _same(batch_sizes, expected_batch_sizes):
            raise ValueError("manifest batch_size_by_dataset violates the V4 protocol")
        workers = _integer(config.get("workers"), "workers")
        workers_by_dataset = config.get("workers_by_dataset")
        if workers_by_dataset is not None and not _same(
            workers_by_dataset,
            {dataset: workers if dataset == "ppi" else 0 for dataset in datasets},
        ):
            raise ValueError("workers_by_dataset is inconsistent")
        if not isinstance(config.get("device"), str) or not re.fullmatch(
            r"cuda(?::[0-9]+)?", config["device"]
        ):
            raise ValueError("CUDA device required")
        for key, expected in COMMON.items():
            if not _same(config.get(key), expected):
                raise ValueError(f"fixed configuration.{key} mismatch")
    except ValueError as exc:
        errors.append(str(exc))

    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        jobs = []
        errors.append("manifest.jobs must be a list")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            errors.append("invalid manifest job")
            continue
        dataset, condition = job.get("dataset"), job.get("condition")
        if dataset not in datasets or condition not in CONDITIONS:
            errors.append("job references an unknown dataset/condition")
            continue
        key = dataset, condition
        if key in indexed:
            errors.append(f"duplicate job: {key}")
            continue
        indexed[key] = job
        expected_batch_size = BATCH_SIZE_BY_DATASET[dataset]
        if job.get("batch_size") != expected_batch_size:
            errors.append(f"{key}: job batch_size must be {expected_batch_size}")
        expected_workers = config.get("workers", 0) if dataset == "ppi" else 0
        if job.get("workers", expected_workers) != expected_workers:
            errors.append(f"{key}: job workers must be {expected_workers}")
        if job.get("status") not in {"pending", "running", "failed", "passed"}:
            errors.append(f"{key}: invalid job status")
        try:
            output = _contained(job.get("output_dir"), root, f"{key} output")
            metrics = _contained(job.get("metrics_path"), root, f"{key} metrics")
            if (
                output != (root / dataset / condition).resolve()
                or metrics != output / "metrics.json"
            ):
                raise ValueError(f"{key}: output/metrics do not match canonical paths")
        except ValueError as exc:
            errors.append(str(exc))
    if set(indexed) != {(dataset, condition) for dataset in datasets for condition in CONDITIONS}:
        errors.append("manifest must contain the complete four-arm job matrix")

    dataset_reports = []
    for dataset in datasets:
        loaded: dict[str, dict[str, Any]] = {}
        rows = []
        for condition in FACTORIAL_ORDER:
            spec = CONDITIONS[condition]
            job = indexed.get((dataset, condition))
            row = {
                "condition": condition,
                **spec,
                "status": job.get("status") if job else "missing",
                **{
                    key: None
                    for key in (
                        "validation",
                        "validation_percent",
                        "best_epoch",
                        "stop_epoch",
                        "stopping_reason",
                        "epochs_run",
                        "train_loss",
                        "total_parameters",
                        "trainable_parameters",
                        "frozen_parameters",
                        "elapsed_seconds",
                        "selection_loop_seconds",
                        "post_selection_diagnostics_seconds",
                        "peak_cuda_allocated_bytes",
                        "peak_cuda_reserved_bytes",
                    )
                },
                "best_validation_diagnostics": [],
                "best_epoch_training_observation": None,
                "best_checkpoint_interventions": None,
                "epoch_timing": None,
            }
            if job and job.get("error"):
                row["error"] = str(job["error"])
            if job and job.get("status") == "passed":
                try:
                    child = _load(root, job, config, sources)
                    loaded[condition] = child
                    for key in (
                        "validation",
                        "best_epoch",
                        "stop_epoch",
                        "stopping_reason",
                        "epochs_run",
                        "train_loss",
                        "total_parameters",
                        "trainable_parameters",
                        "frozen_parameters",
                        "elapsed_seconds",
                        "selection_loop_seconds",
                        "post_selection_diagnostics_seconds",
                        "peak_cuda_allocated_bytes",
                        "peak_cuda_reserved_bytes",
                    ):
                        row[key] = child[key]
                    row["epoch_timing"] = child["epoch_timing"]
                    row["validation_percent"] = 100.0 * child["validation"]
                    row["best_validation_diagnostics"] = _best_layers(child["diagnostics"])
                    row["best_epoch_training_observation"] = _best_training_observation(
                        child, child["configuration"]
                    )
                    row["best_checkpoint_interventions"] = child["diagnostics"][
                        "best_checkpoint_interventions"
                    ]
                except (ValueError, KeyError, TypeError) as exc:
                    row.update(status="invalid", error=str(exc))
                    errors.append(f"{dataset}/{condition}: {exc}")
            rows.append(row)

        reference = next(iter(loaded.values()), None)
        held_fixed = _comparison_metadata(reference) if reference else None
        if reference:
            extra = (
                "versions",
                "gpu",
                "topology",
                "parameterization",
                "source_sha256",
            )
            held_fixed.update({key: reference[key] for key in extra})
            for condition, child in loaded.items():
                actual = _comparison_metadata(child) | {key: child[key] for key in extra}
                for key, expected in held_fixed.items():
                    if not _same(expected, actual[key]):
                        errors.append(f"{dataset}/{condition}: held-fixed {key} mismatch")
        dataset_reports.append(
            {
                "dataset": dataset,
                "metric_name": METRIC_BY_DATASET[dataset],
                "model_seed": config.get("model_seed"),
                "conditions": rows,
                "complete": len(loaded) == len(CONDITIONS),
                "held_fixed": held_fixed,
                "factorial_contrasts": None,
            }
        )

    manifest_passed = manifest.get("status") == "passed"
    all_complete = bool(dataset_reports) and all(item["complete"] for item in dataset_reports)
    if manifest_passed and not all_complete:
        errors.append("passed manifest lacks a complete four-arm matrix")
    for item in dataset_reports:
        if errors:
            item["complete"] = False
        elif manifest_passed and item["complete"]:
            item["factorial_contrasts"] = _factorial(
                {row["condition"]: row["validation"] for row in item["conditions"]}
            )
    failed = manifest.get("status") == "failed" or any(
        job.get("status") == "failed" for job in indexed.values()
    )
    complete = all_complete and not errors and not failed and manifest_passed
    return {
        "schema_version": 1,
        "suite": SUITE,
        "status": "invalid"
        if errors
        else "passed"
        if complete
        else "failed"
        if failed
        else "running",
        "complete": complete,
        "n_model_seeds": 1,
        "model_seed": config.get("model_seed"),
        "evaluation_split": "validation",
        "test_evaluated": False,
        "uncertainty_status": "not_estimated_single_seed",
        "source_integrity_valid": manifest.get("source_integrity_valid"),
        "datasets": dataset_reports,
        "errors": errors,
        "caveats": CAVEATS,
    }


def _diagnostic_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Selected-checkpoint C and propagation diagnostics",
        "",
        "Layer indices are zero-based. Fixed-control values are N/A because those parameters "
        "are absent.",
        "",
        "| Condition | Layer | Score std | C CV | log-C std | alpha | gamma | tau | Conv change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    paths = (
        ("score", "std"),
        ("conductance", "cv"),
        ("log_conductance", "std"),
        ("alpha",),
        ("gamma",),
        ("tau",),
        ("relative_conv_change",),
    )
    for row in rows:
        for layer in row["best_validation_diagnostics"]:
            values = [_display(_nested(layer, *path)) for path in paths]
            lines.append("| " + " | ".join([row["condition"], str(layer["layer"])] + values) + " |")
    lines += [
        "",
        "### Selected-checkpoint spatial-W diagnostics",
        "",
        "| Condition | Layer | W-I Frobenius | W-I relative | singular min | singular max "
        "| condition number | message change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    paths = (
        ("spatial_weight", "identity_distance_frobenius"),
        ("spatial_weight", "identity_relative_distance"),
        ("spatial_weight", "singular_values", "min"),
        ("spatial_weight", "singular_values", "max"),
        ("spatial_weight", "singular_values", "condition_number"),
        ("relative_message_transform_change",),
    )
    for row in rows:
        for layer in row["best_validation_diagnostics"]:
            values = [_display(_nested(layer, *path)) for path in paths]
            lines.append("| " + " | ".join([row["condition"], str(layer["layer"])] + values) + " |")
    return lines + [""]


def _gradient_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Actual training gradients at the selected epoch",
        "",
        "Actual transductive full-graph or PPI first-minibatch backward, before that batch's "
        "optimizer update; parameter-absent fixed-control entries are N/A.",
        "",
        "| Condition | Epoch | Layer | C-gate gradient L2 | Spatial-W gradient L2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        record = row["best_epoch_training_observation"]
        if not record:
            continue
        for layer in record["layers"]:
            gate = layer.get("gate_gradient_norm")
            spatial = layer.get("spatial_gradient_norm")
            lines.append(
                "| "
                + " | ".join(
                    (
                        row["condition"],
                        str(record["epoch"]),
                        str(layer["layer"]),
                        "parameter absent / N/A" if gate is None else _display(gate),
                        "parameter absent / N/A" if spatial is None else _display(spatial),
                    )
                )
                + " |"
            )
    return lines + [""]


def _intervention_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Selected-checkpoint validation interventions (no retraining)",
        "",
        "| Condition | Intervention | Validation (%) | Delta original (pp) "
        "| Changed predictions (%) | Mean absolute logit delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        audit = row["best_checkpoint_interventions"]
        if not audit:
            continue
        for item in audit["rows"]:
            values = (
                row["condition"],
                item["intervention"],
                _display(100.0 * item["validation"]),
                _display(item["percentage_points"], signed=True),
                _display(100.0 * item["changed_prediction_fraction"]),
                _display(item["logit_mean_absolute_delta"]),
            )
            lines.append("| " + " | ".join(values) + " |")
    return lines + [""]


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Conductance C x spatial W V4 factorial",
        "",
        f"Status: **{report['status']}**; model seed {report['model_seed']}; validation only.",
        "",
    ]
    if report["errors"]:
        lines += ["## Integrity errors: factorial contrasts withheld", ""]
        lines += [f"- {_cell(error)}" for error in report["errors"]] + [""]
    for dataset in report["datasets"]:
        lines += [
            f"## {dataset['dataset']} ({dataset['metric_name']}, higher is better)",
            "",
            "| Condition | C mode | W mode | Status | Validation (%) | Best epoch | Stop epoch "
            "| Stop reason | Train loss | Trainable | Frozen |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for row in dataset["conditions"]:
            values = [
                row["condition"],
                row["gate_mode"],
                row["spatial_mode"],
                row["status"],
                _display(row["validation_percent"]),
                str(row["best_epoch"]) if row["best_epoch"] is not None else "—",
                str(row["stop_epoch"]) if row["stop_epoch"] is not None else "—",
                row["stopping_reason"] if row["stopping_reason"] is not None else "—",
                _display(row["train_loss"]),
                str(row["trainable_parameters"])
                if row["trainable_parameters"] is not None
                else "—",
                str(row["frozen_parameters"]) if row["frozen_parameters"] is not None else "—",
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines += ["", "### Conditional factorial contrasts", ""]
        contrasts = dataset["factorial_contrasts"]
        if contrasts:
            lines += [
                "| Contrast | Validation delta (pp) |",
                "| --- | ---: |",
            ]
            for item in contrasts.values():
                lines.append(
                    f"| {item['label']} | {_display(item['percentage_points'], signed=True)} |"
                )
        else:
            lines.append("Withheld until all four arms pass integrity checks.")
        lines += [
            "",
            "### Whole-loop resources",
            "",
            "| Condition | Epoch median (s) | Epoch p90 (s) | Selection loop (s) "
            "| Post-selection (s) | Total elapsed (s) | Peak allocated bytes "
            "| Peak reserved bytes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in dataset["conditions"]:
            values = [
                row["condition"],
                _display(_nested(row, "epoch_timing", "median_seconds")),
                _display(_nested(row, "epoch_timing", "p90_seconds")),
                _display(row["selection_loop_seconds"]),
                _display(row["post_selection_diagnostics_seconds"]),
                _display(row["elapsed_seconds"]),
                str(row["peak_cuda_allocated_bytes"])
                if row["peak_cuda_allocated_bytes"] is not None
                else "—",
                str(row["peak_cuda_reserved_bytes"])
                if row["peak_cuda_reserved_bytes"] is not None
                else "—",
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines += [""] + _diagnostic_markdown(dataset["conditions"])
        lines += _gradient_markdown(dataset["conditions"])
        lines += _intervention_markdown(dataset["conditions"])
    lines += ["## Interpretation limits", ""] + [f"- {item}" for item in CAVEATS] + [""]
    return "\n".join(lines)


def csv_text(report: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    fields = [
        "dataset",
        "metric_name",
        "model_seed",
        "condition",
        "gate_mode",
        "spatial_mode",
        "status",
        "validation",
        "validation_percent",
        "best_epoch",
        "stop_epoch",
        "stopping_reason",
        "epochs_run",
        "train_loss",
        "total_parameters",
        "trainable_parameters",
        "frozen_parameters",
        "selection_loop_seconds",
        "post_selection_diagnostics_seconds",
        "elapsed_seconds",
        "epoch_median_seconds",
        "epoch_p90_seconds",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "c_given_w_off_pp",
        "c_given_w_on_pp",
        "w_given_c_fixed_pp",
        "w_given_c_relative_pp",
        "interaction_pp",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for dataset in report["datasets"]:
        contrasts = dataset["factorial_contrasts"] or {}
        additions = {f"{key}_pp": item["percentage_points"] for key, item in contrasts.items()}
        for row in dataset["conditions"]:
            record = {key: row[key] for key in fields if key in row}
            record["epoch_median_seconds"] = _nested(row, "epoch_timing", "median_seconds")
            record["epoch_p90_seconds"] = _nested(row, "epoch_timing", "p90_seconds")
            record.update({key: dataset[key] for key in ("dataset", "metric_name", "model_seed")})
            record.update(additions)
            writer.writerow(record)
    return buffer.getvalue()


def write_comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve(strict=True)
    if not root.is_dir() or not isinstance(manifest, dict):
        raise ValueError("expected an existing run directory and manifest object")
    destinations = [_contained(name, root, name) for name in REPORT_FILENAMES]
    if any((root / name).is_symlink() for name in REPORT_FILENAMES):
        raise ValueError("report destinations must not be symlinks")
    report = build_comparison(root, manifest)
    contents = [
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        markdown(report),
        csv_text(report),
    ]
    for destination, content in zip(destinations, contents, strict=True):
        _atomic_write(destination, content)
    if report["errors"]:
        raise ComparisonIntegrityError(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.run_dir.expanduser().resolve(strict=True)
        manifest = json.loads(
            _contained("manifest.json", root, "manifest").read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
        source_metadata = manifest.get("sources")
        if not isinstance(source_metadata, dict):
            raise ValueError("manifest.sources must be an object")
        recorded_sources = _source_hashes(source_metadata.get("sha256"))
        if _current_source_hashes() != recorded_sources:
            manifest["source_integrity_valid"] = False
        report = write_comparison(root, manifest)
    except (OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    print(markdown(report))
    print(f"Reports: {root / 'comparison.md'}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
