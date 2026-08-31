"""Fail-closed, validation-only comparisons for the single-seed 2x2 experiment.

This module only reads child artifacts. It never trains, imports Torch, deserializes checkpoints,
evaluates test labels, or changes a child result. The three comparison files are
derived artifacts and can be regenerated after each completed/failed child.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from .protocol import COMMON, CONDITIONS, DATASETS

REPORT_FILENAMES = ("comparison.json", "comparison.md", "comparison.csv")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_JOB_STATUSES = {"pending", "running", "passed", "failed"}
_CONFIG_NOT_CHILD = {"datasets", "data_root", "results_root", "run_id", "fail_fast"}
_FACTOR_KEYS = {"condition", "normalization", "gate_weight_decay"}
_OUTPUT_KEYS = {"output_dir", "metrics_path", "checkpoint", "history", "log_path"}
_EFFECTS = (
    ("gate_effect_at_global_max", "Gate WD off at global-max normalization"),
    ("normalization_effect_with_gate_wd", "Node-degree normalization with gate WD"),
    ("gate_effect_at_node_degree", "Gate WD off at node-degree normalization"),
    ("normalization_effect_without_gate_wd", "Node-degree normalization without gate WD"),
    ("interaction", "Interaction (both - gate-only - normalization-only + baseline)"),
)
_CAVEATS = [
    "One model seed (n=1): exploratory validation comparison, not a paper-level "
    "performance claim. No seed standard deviation, confidence interval, or p-value is estimated.",
    "Scores and contrasts remain separate for every dataset; PPI uses micro-F1, "
    "other supported datasets use accuracy. Scores are fractions; pp means percentage points.",
    "No test split is evaluated. Repeated validation-driven choices can overfit validation; "
    "this report is for diagnosis, not an untouched final test estimate.",
    "The node-degree variant changes the propagation operator, not just execution speed. "
    "A uniform conductance rescaling still cancels under degree normalization.",
    "Valid contrasts require shared initialization, cached data/protocol, training configuration, "
    "and early-stopping policy. Selected epochs and actual epochs run may nevertheless differ. "
    "Identical seeds/initialization do not guarantee bitwise-identical CUDA scatter trajectories.",
    "The interaction is an algebraic contrast of these four runs, not a statistical "
    "significance test. A change in validation score alone does not prove a collapse mechanism.",
]


class ComparisonIntegrityError(ValueError):
    """A saved comparison is invalid and contains no effect estimates."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__("Factorial comparison integrity failed: " + "; ".join(report["errors"]))


def _contained(path: Any, base: Path, label: str) -> Path:
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError(f"{label}: a nonempty artifact path is required")
    candidate = Path(path).expanduser()
    resolved = (base / candidate if not candidate.is_absolute() else candidate).resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise ValueError(f"{label}: artifact path escapes its allowed directory")
    return resolved


def _finite_number(value: Any, label: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number")
    result = float(value)
    if not math.isfinite(result) or (unit_interval and not 0.0 <= result <= 1.0):
        raise ValueError(
            f"{label}: expected {'a score in [0, 1]' if unit_interval else 'finite data'}"
        )
    return result


def _same(left: Any, right: Any) -> bool:
    """Avoid Python equating False with 0 in experiment metadata."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label}: expected an integer >= {minimum}")
    return value


def _metric_name(dataset: str) -> str:
    return "micro_f1" if dataset == "ppi" else "accuracy"


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"nonfinite JSON value {value} is not valid experiment metadata")


def _load_child(
    run_dir: Path,
    job: dict[str, Any],
    common: dict[str, Any],
    *,
    suite: str = "conductance_factorial",
    conditions: dict[str, dict[str, Any]] = CONDITIONS,
) -> dict[str, Any]:
    dataset, condition = job["dataset"], job["condition"]
    label = f"{dataset}/{condition}"
    output = _contained(job.get("output_dir"), run_dir, f"{label} output_dir")
    metrics_path = _contained(job.get("metrics_path"), run_dir, f"{label} metrics_path")
    if not metrics_path.is_relative_to(output):
        raise ValueError(f"{label}: metrics_path must be inside that job's output_dir")
    try:
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label}: cannot read metrics.json ({exc})") from exc
    if not isinstance(metrics, dict):
        raise ValueError(f"{label}: metrics must be a JSON object")
    expected = {
        "schema_version": 1,
        "status": "passed",
        "research_suite": suite,
        "dataset": dataset,
        "condition": condition,
        "model_seed": common["model_seed"],
        "normalization": conditions[condition]["normalization"],
        "gate_weight_decay": conditions[condition]["gate_weight_decay"],
        "non_gate_weight_decay": common["weight_decay"],
        "metric_name": _metric_name(dataset),
        "test_evaluated": False,
        "evaluation_split": "validation",
    }
    for key, value in expected.items():
        if key not in metrics or not _same(metrics[key], value):
            raise ValueError(f"{label}: {key} mismatch (expected {value!r})")
    for key in ("cache_sha256", "initial_state_sha256"):
        if not isinstance(metrics.get(key), str) or not _SHA256.fullmatch(metrics[key]):
            raise ValueError(f"{label}: {key} must be a SHA-256 digest")
    if not isinstance(metrics.get("protocol"), dict) or not metrics["protocol"]:
        raise ValueError(f"{label}: nonempty cached-data protocol is required")
    configuration = metrics.get("configuration")
    if not isinstance(configuration, dict) or not configuration:
        raise ValueError(f"{label}: configuration is missing")
    for key, value in common.items():
        if key not in _CONFIG_NOT_CHILD | _FACTOR_KEYS | _OUTPUT_KEYS:
            if key not in configuration or not _same(configuration[key], value):
                raise ValueError(f"{label}: configuration.{key} differs from manifest")
    for key in _FACTOR_KEYS:
        if key in configuration and not _same(configuration[key], expected[key]):
            raise ValueError(f"{label}: configuration.{key} contradicts metric metadata")
    if configuration.get("tf32") is not False:
        raise ValueError(f"{label}: configuration.tf32 must explicitly be False")
    metrics["validation"] = _finite_number(
        metrics.get("validation"), f"{label} validation", unit_interval=True
    )
    for key in ("train_loss", "elapsed_seconds"):
        value = _finite_number(metrics.get(key), f"{label} {key}")
        if value < 0:
            raise ValueError(f"{label}: {key} cannot be negative")
    _integer(metrics.get("peak_cuda_allocated_bytes"), f"{label} peak_cuda_allocated_bytes")
    best_epoch = _integer(metrics.get("best_epoch"), f"{label} best_epoch", minimum=1)
    epochs_run = _integer(metrics.get("epochs_run"), f"{label} epochs_run", minimum=1)
    if not best_epoch <= epochs_run <= common["epochs"]:
        raise ValueError(f"{label}: best_epoch/epochs_run exceed the configured epoch budget")
    for key in ("checkpoint", "history"):
        artifact = _contained(metrics.get(key), output, f"{label} {key}")
        expected_digest = metrics.get(f"{key}_sha256")
        if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest):
            raise ValueError(f"{label}: {key}_sha256 must be a SHA-256 digest")
        try:
            with artifact.open("rb") as stream:
                actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            raise ValueError(f"{label}: cannot read {key} artifact ({exc})") from exc
        if actual_digest != expected_digest.lower():
            raise ValueError(f"{label}: {key} SHA-256 mismatch")
    # Diagnostics are intentionally descriptive, not part of the held-fixed configuration.
    return metrics


def _pair_metadata(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "dataset",
            "model_seed",
            "metric_name",
            "cache_sha256",
            "protocol",
            "initial_state_sha256",
            "non_gate_weight_decay",
            "evaluation_split",
            "test_evaluated",
        )
    } | {
        "configuration": {
            key: value
            for key, value in metrics["configuration"].items()
            if key not in _FACTOR_KEYS | _OUTPUT_KEYS
        }
    }


def _effects(scores: dict[str, float]) -> dict[str, dict[str, float]]:
    baseline = scores["baseline"]
    gate = scores["gate_no_wd"]
    normalization = scores["node_degree"]
    both = scores["node_degree_gate_no_wd"]
    values = (gate - baseline, normalization - baseline, both - normalization, both - gate)
    values += (both - normalization - gate + baseline,)
    return {
        key: {"score_delta": value, "percentage_points": 100.0 * value}
        for (key, _), value in zip(_EFFECTS, values, strict=True)
    }


def _optional_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _best_validation_summary(diagnostics: Any) -> list[dict[str, Any]]:
    """Summarize within-graph observations, never pooled between-graph CV.

    These optional descriptions do not enter score contrasts or integrity
    matching. Missing observations remain null, not zeros. Each mean weights
    graphs equally and carries its own number of available observations.
    """
    best = diagnostics.get("best_validation") if isinstance(diagnostics, dict) else None
    if not isinstance(best, dict) or best.get("split", "validation") != "validation":
        best = {}
    if best.get("mode", "eval") != "eval":
        best = {}
    layers = best.get("layers", [])
    if not isinstance(layers, list):
        layers = []
    norms = best.get("parameter_norms", {})
    if not isinstance(norms, dict):
        norms = {}
    summary = []
    for index in range(COMMON["layers"]):
        matches = [
            layer
            for layer in layers
            if isinstance(layer, dict)
            and type(layer.get("layer")) is int
            and layer["layer"] == index
        ]
        layer = matches[0] if len(matches) == 1 else {}
        graphs = layer.get("graphs", [])
        if not isinstance(graphs, list):
            graphs = []
        observed = {"conductance_cv": [], "rho_mean": [], "relative_conv_change": []}
        for graph in graphs:
            if not isinstance(graph, dict):
                continue
            for field, section, value_key in (
                ("conductance_cv", "conductance", "cv"),
                ("rho_mean", "rho", "mean"),
                ("relative_conv_change", None, "relative_conv_change"),
            ):
                record = graph if section is None else graph.get(section, {})
                value = (
                    _optional_nonnegative(record.get(value_key))
                    if isinstance(record, dict)
                    else None
                )
                if value is not None:
                    observed[field].append(value)
        gate_norms = [
            value
            for name, raw_value in norms.items()
            if isinstance(name, str)
            and name.startswith(f"operators.{index}.estimator.")
            and (value := _optional_nonnegative(raw_value)) is not None
        ]
        summary.append(
            {
                "layer": index,
                "graph_count": len(graphs),
                "aggregation": "unweighted graph mean of within-graph measurements",
                **{
                    field: {
                        "mean": math.fsum(values) / len(values) if values else None,
                        "valid_graph_count": len(values),
                    }
                    for field, values in observed.items()
                },
                "gate_parameter_l2": math.hypot(*gate_norms) if gate_norms else None,
                "gate_parameter_tensor_count": len(gate_norms),
            }
        )
    return summary


def _build_comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("source_integrity_valid") is False:
        errors.append("Source integrity failed: experiment code changed during this run")
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        config = {}
        errors.append("manifest.config must be an object")
    datasets = config.get("datasets", [])
    if (
        not isinstance(datasets, list)
        or not datasets
        or any(not isinstance(value, str) or value not in DATASETS for value in datasets)
        or len(set(datasets)) != len(datasets)
    ):
        errors.append("manifest.config.datasets must list unique supported datasets")
        datasets = []
    for key, value in (("schema_version", 1), ("suite", "conductance_factorial")):
        if not _same(manifest.get(key), value):
            errors.append(f"manifest.{key} mismatch (expected {value!r})")
    if manifest.get("status") not in {"running", "passed", "failed"}:
        errors.append("manifest.status must be running, passed, or failed")
    try:
        _integer(config.get("model_seed"), "manifest.config.model_seed")
        _integer(config.get("epochs"), "manifest.config.epochs", minimum=1)
        for key, value in COMMON.items():
            if not _same(config.get(key), value):
                raise ValueError(f"manifest.config.{key} must be {value!r} for this fixed 2x2")
    except ValueError as exc:
        errors.append(str(exc))
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        jobs = []
        errors.append("manifest.jobs must be a list")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            errors.append("manifest job must be an object")
            continue
        dataset, condition = job.get("dataset"), job.get("condition")
        if not isinstance(dataset, str) or dataset not in datasets:
            errors.append("manifest job references a dataset outside config.datasets")
            continue
        if not isinstance(condition, str) or condition not in CONDITIONS:
            errors.append(f"{dataset}: manifest job references an unknown condition")
            continue
        key = (dataset, condition)
        if key in indexed:
            errors.append(f"{dataset}/{condition}: duplicate manifest job")
            continue
        indexed[key] = job
        if job.get("status") not in _JOB_STATUSES:
            errors.append(f"{dataset}/{condition}: unknown job status")
        try:
            output = _contained(job.get("output_dir"), run_dir, f"{key} output_dir")
            metrics_path = _contained(job.get("metrics_path"), run_dir, f"{key} metrics_path")
            if not metrics_path.is_relative_to(output):
                raise ValueError(f"{key}: metrics_path is outside its job output_dir")
        except ValueError as exc:
            errors.append(str(exc))
    reports = []
    for dataset in datasets:
        entries: list[dict[str, Any]] = []
        loaded: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            job = indexed.get((dataset, condition))
            entry: dict[str, Any] = {
                "condition": condition,
                "status": job.get("status", "invalid") if job else "missing",
                "normalization": CONDITIONS[condition]["normalization"],
                "gate_weight_decay": CONDITIONS[condition]["gate_weight_decay"],
                "validation": None,
                "validation_percent": None,
                "delta_from_baseline": None,
                "best_epoch": None,
                "epochs_run": None,
                "best_validation_diagnostics": _best_validation_summary(None),
            }
            if job and job.get("error"):
                entry["error"] = str(job["error"])
            if job and job.get("status") == "passed":
                try:
                    child = _load_child(run_dir, job, config)
                    loaded[condition] = child
                    entry.update(
                        validation=child["validation"],
                        validation_percent=child["validation"] * 100.0,
                        best_epoch=child["best_epoch"],
                        epochs_run=child["epochs_run"],
                        train_loss=child["train_loss"],
                        elapsed_seconds=child["elapsed_seconds"],
                        peak_cuda_allocated_bytes=child["peak_cuda_allocated_bytes"],
                        diagnostics=child.get("diagnostics"),
                        best_validation_diagnostics=_best_validation_summary(
                            child.get("diagnostics")
                        ),
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    entry["status"] = "invalid"
                    entry["error"] = str(exc)
                    errors.append(str(exc))
            entries.append(entry)
        reference = next(iter(loaded.values()), None)
        metadata = _pair_metadata(reference) if reference is not None else None
        for condition, child in loaded.items():
            actual = _pair_metadata(child)
            if metadata is not None:
                for key, value in metadata.items():
                    if not _same(actual[key], value):
                        errors.append(
                            f"{dataset}/{condition}: held-fixed {key} differs across runs"
                        )
        complete = len(loaded) == len(CONDITIONS)
        reports.append(
            {
                "dataset": dataset,
                "metric_name": _metric_name(dataset),
                "model_seed": config.get("model_seed"),
                "complete": complete,
                "conditions": entries,
                "held_fixed": metadata,
                "effects": None,
            }
        )
    all_complete = bool(reports) and all(item["complete"] for item in reports)
    if manifest.get("status") == "passed" and not all_complete:
        errors.append("manifest is passed but the complete four-condition matrix is not available")
    if errors:
        for item in reports:
            item["complete"] = False
    else:
        for item in reports:
            if item["complete"]:
                scores = {row["condition"]: row["validation"] for row in item["conditions"]}
                item["effects"] = _effects(scores)
                for row in item["conditions"]:
                    delta = row["validation"] - scores["baseline"]
                    row["delta_from_baseline"] = {
                        "score_delta": delta,
                        "percentage_points": delta * 100.0,
                    }
    failed = manifest.get("status") == "failed" or any(
        job.get("status") == "failed" for job in indexed.values()
    )
    complete = all_complete and not errors and manifest.get("status") == "passed" and not failed
    return {
        "schema_version": 1,
        "suite": "conductance_factorial",
        "status": "invalid"
        if errors
        else "passed"
        if complete
        else "failed"
        if failed
        else "running",
        "complete": complete,
        "source_manifest_status": manifest.get("status"),
        "source_integrity_valid": manifest.get("source_integrity_valid", True),
        "model_seed": config.get("model_seed"),
        "n_model_seeds": 1,
        "evaluation_split": "validation",
        "test_evaluated": False,
        "uncertainty_status": "not_estimated_single_seed",
        "datasets": reports,
        "errors": errors,
        "caveats": _CAVEATS,
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _display(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _diagnostic_markdown(conditions: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Best-checkpoint validation: gate and propagation diagnostics",
        "",
        "Each mean below weights validation graphs equally. C CV is computed within each graph "
        "before averaging; pooled between-graph variation is not used. Parentheses show available "
        "graphs / observed graphs; missing values are —. Layer indices are zero-based. Gate L2 "
        "combines that layer's gate parameter tensor norms at the selected best checkpoint.",
        "",
        "| Condition | Layer | Within-graph C CV mean | ρ mean | Relative Conv change mean "
        "| Gate parameter L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in conditions:
        for layer in condition["best_validation_diagnostics"]:
            cells = []
            for key in ("conductance_cv", "rho_mean", "relative_conv_change"):
                metric = layer[key]
                value = metric["mean"]
                cells.append(
                    f"{value:.6g} ({metric['valid_graph_count']}/{layer['graph_count']})"
                    if value is not None
                    else "—"
                )
            norm = layer["gate_parameter_l2"]
            norm_text = f"{norm:.6g}" if norm is not None else "—"
            lines.append(
                f"| {condition['condition']} | {layer['layer']} | "
                + " | ".join(cells)
                + f" | {norm_text} |"
            )
    return lines + [""]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Conductance: single-seed 2x2 validation comparison",
        "",
        f"Status: **{report['status']}**. Model seed: {report['model_seed']}. "
        "Evaluation: validation only; test not evaluated.",
        "",
    ]
    if report["errors"]:
        lines += ["## Integrity errors — no contrasts are reported", ""]
        lines += [f"- {_cell(error)}" for error in report["errors"]]
        lines += [""]
    for dataset in report["datasets"]:
        lines += [
            f"## {dataset['dataset']} ({dataset['metric_name']}, higher is better)",
            "",
            "| Condition | Status | Validation | Validation (%) | Δ baseline (pp) "
            "| Best epoch | Epochs run |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in dataset["conditions"]:
            delta = row["delta_from_baseline"]
            lines.append(
                f"| {row['condition']} | {_cell(row['status'])} | "
                f"{_display(row['validation'])} | {_display(row['validation_percent'])} | "
                f"{_display(delta['percentage_points'] if delta else None, signed=True)} | "
                f"{row['best_epoch'] if row['best_epoch'] is not None else '—'} | "
                f"{row['epochs_run'] if row['epochs_run'] is not None else '—'} |"
            )
        lines += [""]
        if dataset["effects"]:
            lines += [
                "| Contrast | Score delta | Percentage points |",
                "| --- | ---: | ---: |",
            ]
            for key, description in _EFFECTS:
                effect = dataset["effects"][key]
                lines.append(
                    f"| {description} | {_display(effect['score_delta'], signed=True)} | "
                    f"{_display(effect['percentage_points'], signed=True)} |"
                )
            lines += [""]
        else:
            lines += ["Contrasts withheld until all four conditions pass integrity checks.", ""]
        lines += _diagnostic_markdown(dataset["conditions"])
        for row in dataset["conditions"]:
            if row.get("error"):
                lines += [f"- {row['condition']}: {_cell(row['error'])}"]
        lines += [""]
    lines += ["## Interpretation limits", ""]
    lines += [f"- {caveat}" for caveat in report["caveats"]]
    lines += ["", "Full metadata and diagnostic trajectories: `comparison.json`.", ""]
    return "\n".join(lines)


def _csv(report: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "dataset",
            "model_seed",
            "metric_name",
            "report_status",
            "row_type",
            "condition",
            "status",
            "normalization",
            "gate_weight_decay",
            "validation",
            "validation_percent",
            "score_delta",
            "percentage_points",
            "best_epoch",
            "epochs_run",
        ],
    )
    writer.writeheader()
    for dataset in report["datasets"]:
        shared = {key: dataset[key] for key in ("dataset", "model_seed", "metric_name")}
        shared["report_status"] = report["status"]
        for condition in dataset["conditions"]:
            row = {
                key: condition[key]
                for key in (
                    "condition",
                    "status",
                    "normalization",
                    "gate_weight_decay",
                    "validation",
                    "validation_percent",
                    "best_epoch",
                    "epochs_run",
                )
            }
            writer.writerow(
                shared | row | {"row_type": "condition"} | (condition["delta_from_baseline"] or {})
            )
        if dataset["effects"]:
            for key, effect in dataset["effects"].items():
                writer.writerow(
                    shared
                    | {"row_type": "contrast", "condition": key, "status": "complete"}
                    | effect
                )
    return buffer.getvalue()


def _atomic_write(destination: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_comparison(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Regenerate reports, then raise on integrity errors (never on pending jobs).

    Existing derived comparison files are replaced atomically, including with an
    explicit invalid report if child metadata were changed. Child artifacts and
    the caller's manifest are never modified. Unsafe output symlinks are rejected
    before any publication, so reports cannot overwrite files outside this run.
    """
    root = Path(run_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("run_dir must be an existing directory")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    destinations = [_contained(name, root, name) for name in REPORT_FILENAMES]
    for name in REPORT_FILENAMES:
        if (root / name).is_symlink():
            raise ValueError(f"{name}: report destinations must not be symlinks")
    report = _build_comparison(root, manifest)
    contents = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        _markdown(report),
        _csv(report),
    )
    for destination, content in zip(destinations, contents, strict=True):
        _atomic_write(destination, content)
    if report["errors"]:
        raise ComparisonIntegrityError(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Existing factorial run containing manifest.json"
    )
    args = parser.parse_args(argv)
    try:
        root = args.run_dir.expanduser().resolve(strict=True)
        manifest_path = _contained("manifest.json", root, "manifest.json")
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json
        )
        report = write_comparison(root, manifest)
    except (OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    print(_markdown(report))
    print(
        f"Reports: {root / 'comparison.md'}, {root / 'comparison.json'}, {root / 'comparison.csv'}"
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
