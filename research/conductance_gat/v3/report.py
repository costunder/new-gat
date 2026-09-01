"""Fail-closed fixed-graph relative-C comparison; stdlib only, no training or test labels."""

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
from .protocol import COMMON, CONDITIONS, DATASETS, PARAMETERIZATION, SUITE

SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
CAVEATS = [
    "n=1; exploratory validation comparison; test not evaluated. No CI, p-value or seed std.",
    "Both arms train freshly with the same initial full state, official cache, topology, "
    "AdamW policy and early-stopping policy. No V1/V2 or historical score is reused.",
    "V3 uses a shared relative-log-conductance generator, symmetric normalization and learned "
    "propagation strength. The initial benchmark protocol is transductive; PPI is not included.",
    "The contrast is relative_c minus fixed_c (percentage points), an internal V3 ablation. "
    "V2-to-V3 changes several factors and must not be called a single-factor comparison.",
    "Fixed C=1 freezes its unused gate MLP/gamma/tau scaffold; alpha remains trainable in both "
    "arms. Frozen gate parameter norms do not indicate active learning.",
    "Reported alpha/gamma/tau, C CV and log-C spread describe the selected checkpoint, not "
    "proof of useful weighting. Propagation strength alpha is learned in V3.",
    "Mean-C, shuffled-C, C=1 and propagation-off interventions are separate validation forwards "
    "at the selected checkpoint, not retraining. They measure checkpoint reliance, not whether "
    "learning C improves training; the fresh C=1 arm addresses the latter.",
    "Elapsed time and peak CUDA allocation include diagnostics, interventions and checkpoint "
    "overhead as recorded by the training timer. Epoch counts can differ; these are not "
    "isolated kernel benchmarks or measured V1/V2/spectral speedups.",
    "Sparse edge-chunked execution avoids eigendecomposition and dense incidence storage, but "
    "still requires full-graph node states and shared-gate computation. This is not a measured "
    "scalability result or a claim that every spectral architecture requires decomposition.",
    "Repeated validation-guided choices can overfit validation. Identical initial states do "
    "not guarantee bitwise-identical CUDA scatter trajectories.",
]


class ComparisonIntegrityError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(
            "Relative-C V3 comparison integrity failed: " + "; ".join(report["errors"])
        )


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
            raise ValueError("source_sha256 requires relative source paths and SHA-256 digests")
    return value


INTERVENTIONS = {"mean_c", "shuffled_c", "ones_c", "propagation_off"}


def _names(value, label):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{label}: expected parameter-name list")
    if len(set(value)) != len(value):
        raise ValueError(f"{label}: duplicate parameter names")
    return value


def _validate_optimizer(child, config, condition):
    if child.get("optimizer") != "AdamW":
        raise ValueError("optimizer must be AdamW")
    active = _names(child.get("trainable_parameter_names"), "trainable_parameter_names")
    frozen = _names(child.get("frozen_parameter_names"), "frozen_parameter_names")
    if not active or set(active) & set(frozen):
        raise ValueError("active/frozen parameter names overlap or active list is empty")
    if bool(frozen) != (condition == "fixed_c"):
        raise ValueError("frozen parameter names disagree with condition")
    if any(name.endswith(".raw_alpha") for name in frozen):
        raise ValueError("alpha must remain trainable in every layer")
    groups = child.get("optimizer_groups")
    if not isinstance(groups, list):
        raise ValueError("optimizer_groups must be a list")
    expected = {"backbone", "controls"} | ({"gate_mlp"} if condition == "relative_c" else set())
    indexed, all_names, count = {}, [], 0
    for group in groups:
        if not isinstance(group, dict) or group.get("name") not in expected:
            raise ValueError("unexpected optimizer parameter group")
        name = group["name"]
        if name in indexed:
            raise ValueError("duplicate optimizer parameter group")
        indexed[name] = group
        names = _names(group.get("parameter_names"), f"{name}.parameter_names")
        if not names:
            raise ValueError("optimizer groups must not be empty")
        size = _integer(group.get("parameter_count"), f"{name}.parameter_count", minimum=1)
        count += size
        all_names.extend(names)
        lr = config["lr"] * (config["gate_lr_multiplier"] if name == "gate_mlp" else 1)
        wd = config["weight_decay"] if name == "backbone" else 0.0
        if not _same(group.get("lr"), lr) or not _same(group.get("weight_decay"), wd):
            raise ValueError(f"optimizer {name} lr/weight_decay mismatch")
        if name == "controls":
            required = {f"operators.{i}.raw_alpha" for i in range(config["layers"])}
            if condition == "relative_c":
                required |= {
                    f"operators.{i}.estimator.raw_{key}"
                    for i in range(config["layers"])
                    for key in ("gamma", "tau")
                }
            if set(names) != required or size != len(names):
                raise ValueError("scalar optimizer controls mismatch; alpha must remain active")
        elif any(p.rsplit(".", 1)[-1] in {"raw_alpha", "raw_gamma", "raw_tau"} for p in names):
            raise ValueError("scalar controls must use their no-decay parameter group")
    if (
        set(indexed) != expected
        or len(set(all_names)) != len(all_names)
        or set(all_names) != set(active)
    ):
        raise ValueError("optimizer groups do not cover exactly the trainable parameters")
    if count != child["trainable_parameters"]:
        raise ValueError("optimizer parameter counts disagree with trainable_parameters")


def _validate_diagnostics(child, config):
    diagnostics = child.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("V3 diagnostics are required")
    best = diagnostics.get("best_validation")
    if (
        not isinstance(best, dict)
        or best.get("mode") != "eval"
        or best.get("split") != "validation"
    ):
        raise ValueError("best_validation diagnostics must be validation/eval")
    layers = best.get("layers")
    if not isinstance(layers, list) or len(layers) != config["layers"]:
        raise ValueError("best_validation layer diagnostics are incomplete")
    indices = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("invalid layer diagnostics")
        indices.append(_integer(layer.get("layer"), "diagnostic.layer"))
        _finite_number(layer.get("alpha"), "alpha", unit_interval=True)
        _finite_number(layer.get("gamma"), "gamma", unit_interval=True)
        if _finite_number(layer.get("tau"), "tau") <= 0:
            raise ValueError("tau must be positive")
        for path in (
            ("score", "std"),
            ("conductance", "cv"),
            ("log_conductance", "std"),
            ("weighted_degree", "quantiles", "p50"),
            ("weighted_degree", "quantiles", "p99"),
            ("weighted_degree", "max_over_median"),
            ("relative_conv_change",),
            ("gate_parameter_norm",),
            ("gate_gradient_norm",),
        ):
            value = _nested(layer, *path)
            if value is not None and _finite_number(value, ".".join(path)) < 0:
                raise ValueError("diagnostic magnitudes must be nonnegative")
    if sorted(indices) != list(range(config["layers"])):
        raise ValueError("diagnostic layer indices are missing or duplicated")
    _best_training_observation(child, config)
    audit = diagnostics.get("best_checkpoint_interventions")
    if (
        not isinstance(audit, dict)
        or audit.get("status") != "passed"
        or audit.get("scope") != ("validation_selected_best_checkpoint_only")
    ):
        raise ValueError("selected-checkpoint validation interventions are required")
    original = audit.get("original")
    if not isinstance(original, dict):
        raise ValueError("intervention original score is missing")
    score = _finite_number(original.get("validation"), "intervention original", unit_interval=True)
    if abs(score - child["validation"]) > 1.0e-7:
        raise ValueError("intervention original differs from selected validation")
    rows = audit.get("rows")
    if not isinstance(rows, list) or len(rows) != len(INTERVENTIONS):
        raise ValueError("all four selected-checkpoint interventions are required")
    names = []
    for row in rows:
        if not isinstance(row, dict) or row.get("intervention") not in INTERVENTIONS:
            raise ValueError("unknown checkpoint intervention")
        names.append(row["intervention"])
        value = _finite_number(row.get("validation"), "intervention validation", unit_interval=True)
        delta = _finite_number(row.get("percentage_points"), "intervention percentage_points")
        if abs(delta - 100.0 * (value - score)) > 1.0e-8:
            raise ValueError("intervention percentage-points delta mismatch")
        _finite_number(
            row.get("changed_prediction_fraction"), "changed predictions", unit_interval=True
        )
        if _finite_number(row.get("logit_mean_absolute_delta"), "logit delta") < 0:
            raise ValueError("logit delta must be nonnegative")
    if set(names) != INTERVENTIONS or len(set(names)) != len(names):
        raise ValueError("missing/duplicate checkpoint interventions")


def _best_validation_summary(diagnostics):
    if not isinstance(diagnostics, dict):
        return []
    best = diagnostics.get("best_validation", {})
    return best.get("layers", []) if isinstance(best, dict) else []


def _best_training_observation(child, config):
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
    expected = {
        "scope": "full_graph_train_mask",
        "mode": "train_dropout_on",
        "stage": "after_task_backward_before_optimizer_step",
        "batch_index": 0,
        "optimizer_steps_before_batch": child["best_epoch"] - 1,
    }
    for key, value in expected.items():
        if not _same(record.get(key), value):
            raise ValueError(f"selected training observation {key} mismatch")
    layers = record.get("layers")
    if not isinstance(layers, list) or len(layers) != config["layers"]:
        raise ValueError("selected training layer observations are incomplete")
    indices = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("invalid training layer observation")
        indices.append(_integer(layer.get("layer"), "training layer"))
        value = layer.get("gate_gradient_norm")
        if child["condition"] == "relative_c":
            if _finite_number(value, "actual training gate gradient") < 0:
                raise ValueError("actual training gate gradient must be nonnegative")
        elif value is not None:
            raise ValueError("frozen gate must not have a training gradient")
    if sorted(indices) != list(range(config["layers"])):
        raise ValueError("selected training layer indices are missing or duplicated")
    return record


def _nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _diagnostic_markdown(conditions):
    lines = [
        "### Selected-checkpoint layer diagnostics",
        "",
        "Layer indices are zero-based. Validation computes no gradients; missing values are "
        "not evidence that training skipped backpropagation. "
        "Fixed-arm gamma/tau and gate norms describe an unused frozen scaffold.",
        "",
        "| Condition | Layer | Score std | C CV | log-C std | alpha | gamma | tau |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    fields = [
        ("score", "std"),
        ("conductance", "cv"),
        ("log_conductance", "std"),
        ("alpha",),
        ("gamma",),
        ("tau",),
    ]
    for condition in conditions:
        for layer in condition["best_validation_diagnostics"]:
            values = [_display(_nested(layer, *path)) for path in fields]
            lines.append(
                "| " + " | ".join([condition["condition"], str(layer["layer"])] + values) + " |"
            )
    lines += [
        "",
        "| Condition | Layer | Degree p50 | Degree p99 | Degree max/p50 "
        "| Relative propagation change | Gate L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    fields = [
        ("weighted_degree", "quantiles", "p50"),
        ("weighted_degree", "quantiles", "p99"),
        ("weighted_degree", "max_over_median"),
        ("relative_conv_change",),
        ("gate_parameter_norm",),
    ]
    for condition in conditions:
        for layer in condition["best_validation_diagnostics"]:
            values = [_display(_nested(layer, *path)) for path in fields]
            lines.append(
                "| " + " | ".join([condition["condition"], str(layer["layer"])] + values) + " |"
            )
    return lines + [""]


def _training_gradient_markdown(conditions):
    lines = [
        "### Actual training gradients at the selected epoch",
        "",
        "Recorded on the actual training-mask loss after backward and before that epoch's "
        "optimizer update, with training dropout enabled. These are not gradients recomputed "
        "at the selected post-update checkpoint. Frozen gate entries are inapplicable, not zero.",
        "",
        "| Condition | Epoch | Layer | Gate MLP task-gradient L2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in conditions:
        record = condition["best_epoch_training_observation"]
        if not record:
            continue
        for layer in record["layers"]:
            value = layer["gate_gradient_norm"]
            label = "frozen / not trainable" if value is None else _display(value)
            lines.append(
                f"| {condition['condition']} | {record['epoch']} | {layer['layer']} | {label} |"
            )
    return lines + [""]


def _intervention_markdown(conditions):
    lines = [
        "### Selected-checkpoint validation interventions (no retraining)",
        "",
        "| Condition | Intervention | Validation (%) | Delta original (pp) "
        "| Changed predictions (%) | Mean absolute logit delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in conditions:
        audit = condition["best_checkpoint_interventions"]
        if not audit:
            continue
        for row in audit["rows"]:
            values = [
                condition["condition"],
                row["intervention"],
                _display(100.0 * row["validation"]),
                _display(row["percentage_points"], signed=True),
                _display(100.0 * row["changed_prediction_fraction"]),
                _display(row["logit_mean_absolute_delta"]),
            ]
            lines.append("| " + " | ".join(values) + " |")
    return lines + [""]


def _load(root, job, config, source_hashes):
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
    child = _load_child(root, job, config, suite=SUITE, conditions=CONDITIONS)
    for key, expected in (
        ("gate_mode", CONDITIONS[job["condition"]]["gate_mode"]),
        ("parameterization", PARAMETERIZATION),
        ("source_sha256", source_hashes),
    ):
        if not _same(child.get(key), expected):
            raise ValueError(f"{key} mismatch")
    if "gate_mode" in child["configuration"]:
        raise ValueError("gate_mode belongs in arm metadata, not held-fixed configuration")
    topology = child.get("topology")
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
    if (
        total != trainable + frozen
        or (job["condition"] == "relative_c" and frozen != 0)
        or (job["condition"] == "fixed_c" and frozen == 0)
    ):
        raise ValueError("parameter counts disagree with relative-C/frozen-scaffold condition")
    _validate_optimizer(child, config, job["condition"])
    _validate_diagnostics(child, config)
    if not isinstance(child.get("versions"), dict) or not child["versions"]:
        raise ValueError("Missing runtime versions")
    if not isinstance(child.get("gpu"), str) or not child["gpu"]:
        raise ValueError("Missing GPU identity")
    if child["protocol"].get("data_sha256") != child["cache_sha256"]:
        raise ValueError("cache_sha256 disagrees with dataset protocol")
    return child


def build_comparison(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors = []
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        config = {}
        errors.append("manifest.config must be an object")
    datasets = config.get("datasets", [])
    if (
        not isinstance(datasets, list)
        or not datasets
        or any(not isinstance(d, str) or d not in DATASETS for d in datasets)
        or len(set(datasets)) != len(datasets)
    ):
        datasets = []
        errors.append("datasets must list unique supported fixed-graph datasets (PPI unsupported)")
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
    sources = {}
    try:
        source_metadata = manifest.get("sources")
        if not isinstance(source_metadata, dict):
            raise ValueError("manifest.sources must be an object")
        sources = _source_hashes(source_metadata.get("sha256"))
        _integer(config.get("model_seed"), "model_seed")
        for key in ("epochs", "patience", "edge_chunk_size"):
            _integer(config.get(key), key, minimum=1)
        if config.get("batch_size") != 1 or type(config.get("batch_size")) is not int:
            raise ValueError("full-graph batch_size must be 1")
        if config.get("workers") != 0 or type(config.get("workers")) is not int:
            raise ValueError("full-graph workers must be 0")
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
    indexed = {}
    for job in jobs:
        if not isinstance(job, dict):
            errors.append("invalid manifest job")
            continue
        dataset, condition = job.get("dataset"), job.get("condition")
        if (
            not isinstance(dataset, str)
            or dataset not in datasets
            or not isinstance(condition, str)
            or condition not in CONDITIONS
        ):
            errors.append("job references an unknown dataset/condition")
            continue
        key = dataset, condition
        if key in indexed:
            errors.append(f"duplicate job: {key}")
            continue
        indexed[key] = job
        if job.get("status") not in {"pending", "running", "failed", "passed"}:
            errors.append(f"{key}: invalid job status")
        try:
            output = _contained(job.get("output_dir"), root, f"{key} output")
            metrics = _contained(job.get("metrics_path"), root, f"{key} metrics")
            if (
                output != (root / dataset / condition).resolve()
                or metrics != output / "metrics.json"
            ):
                raise ValueError(f"{key}: output/metrics do not match canonical job paths")
        except ValueError as exc:
            errors.append(str(exc))
    if set(indexed) != {(d, c) for d in datasets for c in CONDITIONS}:
        errors.append("manifest must contain the complete two-arm job matrix")
    reports = []
    for dataset in datasets:
        loaded, rows = {}, []
        for condition, spec in CONDITIONS.items():
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
                        "epochs_run",
                        "train_loss",
                        "total_parameters",
                        "trainable_parameters",
                        "frozen_parameters",
                        "elapsed_seconds",
                        "peak_cuda_allocated_bytes",
                    )
                },
                "best_validation_diagnostics": [],
                "best_checkpoint_interventions": None,
                "best_epoch_training_observation": None,
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
                        "epochs_run",
                        "train_loss",
                        "total_parameters",
                        "trainable_parameters",
                        "frozen_parameters",
                        "elapsed_seconds",
                        "peak_cuda_allocated_bytes",
                    ):
                        row[key] = child[key]
                    row["validation_percent"] = 100.0 * child["validation"]
                    row["best_checkpoint_interventions"] = child["diagnostics"][
                        "best_checkpoint_interventions"
                    ]
                    row["best_epoch_training_observation"] = _best_training_observation(
                        child, config
                    )
                    row["best_validation_diagnostics"] = _best_validation_summary(
                        child.get("diagnostics")
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    row.update(status="invalid", error=str(exc))
                    errors.append(f"{dataset}/{condition}: {exc}")
            rows.append(row)
        reference = next(iter(loaded.values()), None)
        metadata = _pair_metadata(reference) if reference else None
        if reference:
            extra = (
                "versions",
                "gpu",
                "total_parameters",
                "topology",
                "parameterization",
                "source_sha256",
            )
            metadata.update({key: reference[key] for key in extra})
            for condition, child in loaded.items():
                actual = _pair_metadata(child) | {key: child[key] for key in extra}
                for key in metadata:
                    if not _same(metadata[key], actual[key]):
                        errors.append(f"{dataset}/{condition}: held-fixed {key} mismatch")
        reports.append(
            {
                "dataset": dataset,
                "metric_name": "accuracy",
                "model_seed": config.get("model_seed"),
                "conditions": rows,
                "complete": len(loaded) == len(CONDITIONS),
                "held_fixed": metadata,
                "relative_minus_fixed": None,
            }
        )
    all_complete = bool(reports) and all(row["complete"] for row in reports)
    if manifest.get("status") == "passed" and not all_complete:
        errors.append("passed manifest lacks a complete two-arm matrix")
    for item in reports:
        if errors:
            item["complete"] = False
        elif item["complete"]:
            scores = {row["condition"]: row["validation"] for row in item["conditions"]}
            delta = scores["relative_c"] - scores["fixed_c"]
            item["relative_minus_fixed"] = {
                "score_delta": delta,
                "percentage_points": 100.0 * delta,
            }
    failed = manifest.get("status") == "failed" or any(
        job.get("status") == "failed" for job in indexed.values()
    )
    complete = all_complete and not errors and not failed and manifest.get("status") == "passed"
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
        "datasets": reports,
        "errors": errors,
        "caveats": CAVEATS,
    }


def markdown(report):
    lines = [
        "# Shared relative C V3 vs fixed C=1",
        "",
        f"Status: **{report['status']}**; model seed {report['model_seed']}; validation only.",
        "",
    ]
    if report["errors"]:
        lines += ["## Integrity errors: contrasts withheld", ""]
        lines += [f"- {_cell(error)}" for error in report["errors"]] + [""]
    for dataset in report["datasets"]:
        lines += [
            f"## {dataset['dataset']} (accuracy, higher is better)",
            "",
            "| Condition | Status | Validation (%) | Best epoch | Epochs run "
            "| Train loss | Trainable | Frozen |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in dataset["conditions"]:
            cells = [row["condition"], row["status"], _display(row["validation_percent"])]
            cells += [
                str(row[k]) if row[k] is not None else "—" for k in ("best_epoch", "epochs_run")
            ]
            cells += [_display(row["train_loss"])]
            cells += [
                str(row[k]) if row[k] is not None else "—"
                for k in ("trainable_parameters", "frozen_parameters")
            ]
            lines.append("| " + " | ".join(cells) + " |")
        contrast = dataset["relative_minus_fixed"]
        lines += [
            "",
            "Relative − fixed: "
            + (
                _display(contrast["percentage_points"], signed=True) + " pp."
                if contrast
                else "withheld until both arms pass integrity checks."
            ),
            "",
            "### Whole training-loop resources (including diagnostic/checkpoint overhead)",
            "",
            "| Condition | Elapsed seconds | Peak CUDA allocated bytes |",
            "| --- | ---: | ---: |",
        ]
        for row in dataset["conditions"]:
            peak = row["peak_cuda_allocated_bytes"]
            lines.append(
                f"| {row['condition']} | {_display(row['elapsed_seconds'])} "
                f"| {peak if peak is not None else '—'} |"
            )
        lines += [""] + _diagnostic_markdown(dataset["conditions"])
        lines += _training_gradient_markdown(dataset["conditions"])
        lines += _intervention_markdown(dataset["conditions"])
    lines += ["## Interpretation limits", ""] + [f"- {c}" for c in CAVEATS] + [""]
    return "\n".join(lines)


def csv_text(report):
    buffer = io.StringIO(newline="")
    fields = [
        "dataset",
        "metric_name",
        "model_seed",
        "condition",
        "status",
        "validation",
        "validation_percent",
        "best_epoch",
        "epochs_run",
        "train_loss",
        "trainable_parameters",
        "frozen_parameters",
        "elapsed_seconds",
        "peak_cuda_allocated_bytes",
        "relative_minus_fixed_pp",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for dataset in report["datasets"]:
        for row in dataset["conditions"]:
            record = {key: row[key] for key in fields if key in row}
            record.update({key: dataset[key] for key in ("dataset", "metric_name", "model_seed")})
            contrast = dataset["relative_minus_fixed"]
            record["relative_minus_fixed_pp"] = contrast["percentage_points"] if contrast else None
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
        report = write_comparison(root, manifest)
    except (OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    print(markdown(report))
    print(f"Reports: {root / 'comparison.md'}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
