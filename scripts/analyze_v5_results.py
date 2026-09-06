"""Read-only analysis of recorded V5 JSON evidence; never load a checkpoint.

This command does not train/evaluate a model, certify a run, or infer SOTA from
stored scores. Missing diagnostics stay unavailable. Output goes to stdout only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

SUITE = "conductance_graph_conditioned_v5"
CONDITIONS = {"fixed_c", "shared_dynamic_c"}
UNAVAILABLE = "unavailable"


class AnalysisError(ValueError):
    """An input is corrupt or contradicts its recorded JSON evidence."""


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number {value}")
    return result


def _read(path: Path, expected_sha256: str | None = None) -> Any:
    if path.suffix.lower() != ".json":
        raise AnalysisError(f"Only JSON artifacts can be read, never checkpoints: {path}")
    if path.is_symlink():
        raise AnalysisError(f"Refusing a symlink input: {path}")
    try:
        raw = path.read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise AnalysisError(f"Recorded SHA256 mismatch: {path}")
        return json.loads(raw.decode("utf-8-sig"), parse_constant=_nonfinite, parse_float=_float)
    except (OSError, UnicodeError, ValueError) as exc:
        raise AnalysisError(f"Cannot analyze JSON {path}: {exc}") from exc


def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return UNAVAILABLE
        value = value[key]
    return UNAVAILABLE if value is None else value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _delta(after: Any, before: Any) -> Any:
    return after - before if _number(after) and _number(before) else UNAVAILABLE


def _is_metrics(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("research_suite") == SUITE
        and value.get("schema_version") == 1
        and value.get("condition") in CONDITIONS
    )


def _path(value: Any, parent: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _history(metrics: dict, metrics_path: Path) -> tuple[list[dict] | None, Any]:
    path = _path(metrics.get("history"), metrics_path.parent)
    if path is None or not path.is_file():
        sibling = metrics_path.parent / "history.json"
        path = sibling if sibling.is_file() else None
    if path is None:
        return None, UNAVAILABLE
    rows = _read(path, metrics.get("history_sha256"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise AnalysisError(f"History must be a JSON array of epoch objects: {path}")
    epochs = [row.get("epoch") for row in rows]
    if any(not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 for epoch in epochs):
        raise AnalysisError(f"History contains an invalid cumulative epoch: {path}")
    if any(a >= b for a, b in zip(epochs, epochs[1:], strict=False)):
        raise AnalysisError(f"History cumulative epochs are not strictly increasing: {path}")
    return rows, str(path.resolve())


def _history_summary(rows: list[dict] | None) -> dict:
    fields = (
        "recorded_epochs",
        "first_epoch",
        "last_epoch",
        "first_train_loss",
        "last_train_loss",
        "train_loss_delta",
        "first_validation",
        "last_validation",
        "observed_best_validation",
        "observed_best_epoch",
        "observed_best_phase",
        "epochs_after_best",
        "best_to_final_drop",
        "observed_peak_fraction",
        "actual_train_batches_in_recorded_history",
    )
    result = dict.fromkeys(fields, UNAVAILABLE)
    if rows is None:
        return result
    result["recorded_epochs"] = len(rows)
    if not rows:
        return result
    first, last = rows[0], rows[-1]
    result.update(
        first_epoch=first["epoch"],
        last_epoch=last["epoch"],
        first_train_loss=_get(first, "train_loss"),
        last_train_loss=_get(last, "train_loss"),
        first_validation=_get(first, "validation"),
        last_validation=_get(last, "validation"),
    )
    result["train_loss_delta"] = _delta(result["last_train_loss"], result["first_train_loss"])
    observed = [(i, row) for i, row in enumerate(rows) if _number(row.get("validation"))]
    if observed:
        index, best = max(observed, key=lambda item: item[1]["validation"])
        result.update(
            observed_best_validation=best["validation"],
            observed_best_epoch=best["epoch"],
            observed_best_phase=_get(best, "phase"),
            epochs_after_best=last["epoch"] - best["epoch"],
            best_to_final_drop=_delta(best["validation"], result["last_validation"]),
            observed_peak_fraction=(index + 1) / len(rows),
        )
    batches = [row.get("train_batches") for row in rows]
    if all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in batches
    ):
        result["actual_train_batches_in_recorded_history"] = sum(batches)
    return result


def _diagnostics(metrics: dict, rows: list[dict] | None) -> dict:
    interventions = metrics.get("selected_checkpoint_interventions", {})
    if not isinstance(interventions, dict):
        raise AnalysisError("selected_checkpoint_interventions must be a JSON object")
    learned = _get(interventions, "learned", "metric")
    comparisons = {}
    for name in ("learned", "c_one", "mean_c", "shuffled_c"):
        value = _get(interventions, name, "metric")
        comparisons[name] = {
            "metric": value,
            "delta_from_learned": _delta(value, learned),
            "recorded_delta_from_learned": _get(interventions, name, "delta_from_learned"),
        }
    layers = _get(interventions, "learned", "layers")
    layer_source = "selected checkpoint learned intervention"
    if not isinstance(layers, list):
        layers = _get(rows[-1], "layers") if rows else UNAVAILABLE
        layer_source = "last recorded training history epoch"
    summaries = UNAVAILABLE
    if isinstance(layers, list):
        summaries = [
            {
                "layer": _get(layer, "layer"),
                "conductance_backend": _get(layer, "conductance_backend"),
                "c_mean": _get(layer, "conductance", "mean"),
                "c_cv": _get(layer, "conductance", "cv"),
                "beta_mean": _get(layer, "beta", "mean"),
                "beta_min": _get(layer, "beta", "min"),
                "beta_max": _get(layer, "beta", "max"),
                "score_std": _get(layer, "score", "std"),
                "c_optimization": _get(layer, "c_optimization"),
            }
            for layer in layers
        ]
    return {
        "interventions": comparisons,
        "delta_definition": "intervention metric minus learned metric; raw metric units",
        "layers": summaries,
        "layer_source": layer_source if isinstance(layers, list) else UNAVAILABLE,
        "layer_scope": (
            "recorded last forward graph/batch only, not a dataset-wide pooled statistic"
        ),
        "first_active_conductance_gradient": _get(metrics, "first_active_conductance_gradient"),
    }


def _analyze_metrics(path: Path, metrics: dict, contexts: list[dict]) -> dict:
    configuration = metrics.get("configuration", {})
    if not isinstance(configuration, dict):
        raise AnalysisError(f"configuration must be a JSON object: {path}")
    rows, history_path = _history(metrics, path)
    history = _history_summary(rows)
    backend = configuration.get("conductance_backend") or "unspecified_legacy"
    roles = sorted({context["role"] for context in contexts})
    if not roles:
        roles = [
            "transitioned_training" if metrics.get("transition_provenance") else "fresh_training"
        ]
    notes = []
    if backend == "unspecified_legacy":
        notes.append(
            "Backend was not recorded; these scores must not be assigned "
            "to the current default backend."
        )
    if not contexts and not metrics.get("transition_provenance"):
        notes.append(
            "Fresh classification means no transition lineage was recorded; "
            "it does not verify initialization."
        )
    if rows is None:
        notes.append(
            "History unavailable: learning curves and actual epoch-level updates "
            "cannot be inspected."
        )
    if (
        _number(history["train_loss_delta"])
        and history["train_loss_delta"] < 0
        and _number(history["best_to_final_drop"])
        and history["best_to_final_drop"] > 0
    ):
        notes.append(
            "Observed: train loss decreased but final validation is below its peak. "
            "Generalization/selection is a hypothesis, not a proven cause."
        )
    if _number(history["observed_peak_fraction"]) and history["observed_peak_fraction"] <= 0.25:
        notes.append(
            "Observed validation peak occurred in the first quarter of the recorded "
            "history (not necessarily the lifetime run)."
        )
    if (
        metrics.get("best_epoch") is not None
        and _number(history["observed_best_epoch"])
        and metrics["best_epoch"] != history["observed_best_epoch"]
    ):
        notes.append(
            "Selected and observed-best epochs differ; inspect checkpoint selection "
            "and phase policy before comparing scores."
        )
    execution = metrics.get("hardware_execution", {})
    batch = _get(metrics, "batch_observability")
    if not isinstance(batch, dict):
        batch = _get(execution, "batching")
    if not isinstance(batch, dict):
        batch = _get(metrics, "pre_run_observability", "batching")
    optimizer_steps = _get(metrics, "optimizer_steps")
    groups = _get(metrics, "effective_optimizer_steps_by_group")
    if rows:
        if optimizer_steps == UNAVAILABLE:
            optimizer_steps = _get(rows[-1], "optimizer_steps")
        if groups == UNAVAILABLE:
            groups = _get(rows[-1], "effective_optimizer_steps_by_group")
    return {
        "metrics_path": str(path.resolve()),
        "roles": roles,
        "contexts": contexts,
        "status": _get(metrics, "status"),
        "dataset": _get(metrics, "dataset"),
        "condition": _get(metrics, "condition"),
        "model_seed": metrics.get("model_seed", _get(configuration, "model_seed")),
        "conductance_backend": backend,
        "configuration": configuration,
        "recorded_batching": batch,
        "configured_graph_batch_size": _get(configuration, "batch_size"),
        "configured_sample_seed_batch_size": _get(configuration, "sample_seed_batch_size"),
        "sampling": _get(configuration, "sampling"),
        "actual_optimizer_steps": optimizer_steps,
        "actual_optimizer_steps_by_group": groups,
        "post_transition_optimizer_steps": _get(metrics, "post_transition_optimizer_steps"),
        "learning_budget": _get(metrics, "learning_budget"),
        "optimization_observability": _get(metrics, "optimization_observability"),
        "metric_name": _get(metrics, "metric_name"),
        "validation": _get(metrics, "validation"),
        "selected_epoch": _get(metrics, "best_epoch"),
        "checkpoint_selection": _get(metrics, "checkpoint_selection"),
        "history_path": history_path,
        "history": history,
        "diagnostics": _diagnostics(metrics, rows),
        "transition_provenance": _get(metrics, "transition_provenance"),
        "source_sha256": _get(metrics, "source_sha256"),
        "resource_observability": _get(metrics, "resource_observability", "summary"),
        "throughput": _get(metrics, "throughput"),
        "error": _get(metrics, "error"),
        "notes": notes,
    }


def analyze(root: Path) -> dict:
    """Inspect JSON files only; do not create, modify, or load any checkpoint."""
    root = Path(root)
    if not root.exists() or root.is_symlink():
        raise AnalysisError(
            f"Input must be an existing non-symlink result directory or JSON file: {root}"
        )
    root = root.resolve()
    if root.is_file():
        if root.suffix.lower() != ".json":
            raise AnalysisError(f"Only JSON files are supported, never checkpoints: {root}")
        paths = [root]
    else:
        paths = []
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = sorted(
                name for name in subdirs if not (Path(directory) / name).is_symlink()
            )
            paths.extend(
                Path(directory) / name
                for name in sorted(files)
                if name in {"metrics.json", "manifest.json"}
                and not (Path(directory) / name).is_symlink()
            )
    metrics_by_path: dict[Path, dict] = {}
    contexts: dict[Path, list[dict]] = {}
    unavailable_jobs = []
    manifests = []
    ignored = []
    for path in sorted(paths):
        value = _read(path)
        if _is_metrics(value):
            metrics_by_path[path.resolve()] = value
        elif isinstance(value, dict) and isinstance(value.get("jobs"), list):
            manifests.append((path, value))
        else:
            ignored.append(str(path))

    def reference(value: Any, parent: Path, context: dict, expected: Any = None) -> bool:
        path = _path(value, parent)
        if path is None or not path.is_file():
            return False
        metrics = _read(path, expected)
        if not _is_metrics(metrics):
            raise AnalysisError(f"Referenced job is not a supported V5 metrics object: {path}")
        for key in ("dataset", "condition"):
            if context.get(key) not in (None, UNAVAILABLE, metrics.get(key)):
                raise AnalysisError(f"Manifest/metrics {key} mismatch: {path}")
        seed = metrics.get("model_seed", _get(metrics, "configuration", "model_seed"))
        if seed != UNAVAILABLE and context.get("model_seed") not in (None, UNAVAILABLE, seed):
            raise AnalysisError(f"Manifest/metrics model_seed mismatch: {path}")
        resolved = path.resolve()
        metrics_by_path[resolved] = metrics
        contexts.setdefault(resolved, []).append(context)
        return True

    for manifest_path, manifest in manifests:
        for job in manifest["jobs"]:
            if not isinstance(job, dict):
                raise AnalysisError(f"Manifest jobs must be objects: {manifest_path}")
            if job.get("condition") not in CONDITIONS or job.get("version", "v5") != "v5":
                continue
            action = job.get("action")
            context = {
                key: _get(job, key)
                for key in (
                    "job_id",
                    "dataset",
                    "condition",
                    "profile",
                    "model_seed",
                    "action",
                    "status",
                )
            }
            context["manifest_path"] = str(manifest_path)
            historical = job.get("historical_reference")
            historical_found = False
            if isinstance(historical, dict):
                historical_context = {**context, "role": "historical_reference"}
                historical_found = reference(
                    historical.get("metrics_path"),
                    manifest_path.parent,
                    historical_context,
                    historical.get("metrics_sha256"),
                )
            if action in {"reuse_completed_fixed", "preserve_legacy_dynamic"}:
                role = "historical_reference"
                found = historical_found
            else:
                role = (
                    "transitioned_training"
                    if action in {"transition_dynamic", "resume_incomplete_fixed"}
                    else "fresh_training"
                )
                output = _path(job.get("output_dir"), manifest_path.parent)
                metrics_path = job.get("metrics_path") or (
                    str(output / "metrics.json") if output else None
                )
                expected = job.get("metrics_sha256") or _get(job, "result", "metrics_sha256")
                found = reference(
                    metrics_path,
                    manifest_path.parent,
                    {**context, "role": role},
                    None if expected == UNAVAILABLE else expected,
                )
            if not found:
                unavailable_jobs.append(
                    {
                        **context,
                        "role": role,
                        "evidence": UNAVAILABLE,
                        "planned_configuration": job.get(
                            "configuration", job.get("architecture", UNAVAILABLE)
                        ),
                        "reason": (
                            "No readable V5 metrics at the recorded path; "
                            "status is not evidence of new training completion."
                        ),
                    }
                )
    records = [
        _analyze_metrics(path, value, contexts.get(path, []))
        for path, value in sorted(metrics_by_path.items())
    ]
    if not records and not unavailable_jobs:
        raise AnalysisError(f"No supported V5 metrics or V5 manifest jobs found under {root}")
    return {
        "schema_version": 1,
        "root": str(root),
        "scope": (
            "Read-only recorded JSON diagnostics; no checkpoint/GPU execution, "
            "causal comparison, or SOTA claim."
        ),
        "records": records,
        "unavailable_jobs": unavailable_jobs,
        "ignored_json": ignored,
    }


def render_text(report: dict) -> str:
    lines = [report["scope"]]
    for record in report["records"]:
        lines.extend(
            (
                "",
                f"{record['dataset']}/{record['condition']} seed={record['model_seed']} | "
                f"{','.join(record['roles'])} | status={record['status']} | "
                f"backend={record['conductance_backend']}",
                f"  metrics: {record['metrics_path']}",
            )
        )
        for key in (
            "configuration",
            "recorded_batching",
            "actual_optimizer_steps",
            "actual_optimizer_steps_by_group",
            "post_transition_optimizer_steps",
            "learning_budget",
            "optimization_observability",
            "validation",
            "selected_epoch",
            "checkpoint_selection",
            "history",
            "diagnostics",
            "transition_provenance",
            "resource_observability",
            "throughput",
        ):
            lines.append(f"  {key}: {json.dumps(record[key], ensure_ascii=False, allow_nan=False)}")
        lines.extend(f"  note: {note}" for note in record["notes"])
    for job in report["unavailable_jobs"]:
        lines.extend(
            (
                "",
                f"{job['dataset']}/{job['condition']} | {job['role']} | "
                f"status={job['status']} | evidence=unavailable",
                f"  {job['reason']}",
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Result directory, manifest.json, or V5 metrics.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON to stdout instead of text; never write a file",
    )
    args = parser.parse_args(argv)
    try:
        report = analyze(args.root)
    except AnalysisError as exc:
        print(f"V5 analysis failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if args.json
        else render_text(report)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
