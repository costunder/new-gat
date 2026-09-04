"""Fail-closed fresh learned-C minus fixed-C validation comparison; no Torch required."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from ..ablation.report import (
    _SHA256,
    REPORT_FILENAMES,
    _atomic_write,
    _best_validation_summary,
    _cell,
    _contained,
    _diagnostic_markdown,
    _display,
    _integer,
    _load_child,
    _metric_name,
    _pair_metadata,
    _reject_nonfinite_json,
    _same,
)
from .protocol import COMMON, CONDITIONS, DATASETS

SUITE = "conductance_c_learning"
CAVEATS = [
    "n=1; exploratory validation comparison, test not evaluated. No CI, p-value or seed std.",
    "Both arms train freshly with the same initialization, data, nongate Adam L2 and policy. "
    "Actual early-stopping epochs may differ. No previous factorial score is reused.",
    "The contrast is learned_c minus fixed_c (percentage points); positive favors learned C. "
    "This is an internal architecture ablation, not an external GCN/GAT benchmark baseline.",
    "Fixed C=1 is genuinely parameter-free: no estimator tensors remain in its state or "
    "optimizer. The learned and fixed full-state hashes and total sizes therefore differ by "
    "design; their shared non-estimator backbone initialization fingerprint must match.",
    "Both arms use node-degree normalization. Common positive C scale cancels; rho=.95 on "
    "nonisolated nodes is imposed by the operator, not evidence that learned C is useful.",
    "This retraining contrast and a checkpoint mean-C intervention answer different questions. "
    "Do not treat a small intervention delta as proof of no benefit from learning C.",
    "Repeated validation-guided decisions can overfit validation. Same seed/state does not "
    "guarantee bitwise CUDA scatter determinism. Do not average PPI F1 and arxiv accuracy.",
]


class ComparisonIntegrityError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("C-learning comparison integrity failed: " + "; ".join(report["errors"]))


def _validate_optimizer_evidence(child: dict[str, Any], *, total: int, trainable: int) -> None:
    pre_run = child.get("pre_run_observability")
    if not isinstance(pre_run, dict) or pre_run.get("status") != "pre_run_configuration":
        raise ValueError("missing passed pre-run configuration evidence")
    model = pre_run.get("model")
    if not isinstance(model, dict):
        raise ValueError("pre-run model evidence is missing")
    for key, expected in (
        ("total_parameters", total),
        ("trainable_parameters", trainable),
        ("frozen_parameters", total - trainable),
    ):
        if model.get(key) != expected:
            raise ValueError(f"pre-run model {key} mismatch")
    ownership = model.get("optimizer_ownership")
    first_step = child.get("first_optimizer_step_integrity")
    for label, evidence in (("pre-run optimizer", ownership), ("first optimizer step", first_step)):
        if not isinstance(evidence, dict) or evidence.get("status") != "passed":
            raise ValueError(f"{label} ownership evidence is missing")
        tensors = _integer(
            evidence.get("trainable_parameter_tensors"),
            f"{label} trainable_parameter_tensors",
            minimum=1,
        )
        owned = _integer(
            evidence.get("optimizer_owned_parameter_tensors"),
            f"{label} optimizer_owned_parameter_tensors",
            minimum=1,
        )
        elements = _integer(
            evidence.get("trainable_parameter_elements"),
            f"{label} trainable_parameter_elements",
            minimum=1,
        )
        if tensors != owned or elements != trainable:
            raise ValueError(f"{label} does not exactly own every trainable parameter")
    if first_step.get("checked_before_optimizer_step") != 1:
        raise ValueError("first optimizer step was not checked before optimizer.step")


def _matched_metadata(child: dict[str, Any]) -> dict[str, Any]:
    metadata = _pair_metadata(child)
    metadata.pop("initial_state_sha256")
    metadata["shared_backbone_initial_state_sha256"] = child[
        "shared_backbone_initial_state_sha256"
    ]
    metadata.update({key: child[key] for key in ("versions", "gpu")})
    return metadata


def _load(run_dir, job, common):
    child_common = dict(common)
    child_common.pop("workers_by_dataset", None)
    child_common["workers"] = common.get("workers", 0) if job["dataset"] == "ppi" else 0
    child = _load_child(run_dir, job, child_common, suite=SUITE, conditions=CONDITIONS)
    spec = CONDITIONS[job["condition"]]
    if child.get("gate_mode") != spec["gate_mode"]:
        raise ValueError(f"{job['dataset']}/{job['condition']}: gate_mode mismatch")
    if "gate_mode" in child["configuration"]:
        raise ValueError("gate_mode must be arm metadata, not the held-fixed configuration")
    total = _integer(child.get("total_parameters"), "total_parameters", minimum=1)
    trainable = _integer(child.get("trainable_parameters"), "trainable_parameters", minimum=1)
    frozen = _integer(child.get("frozen_parameters"), "frozen_parameters")
    estimator = _integer(child.get("estimator_parameters"), "estimator_parameters")
    non_estimator = _integer(
        child.get("non_estimator_parameters"), "non_estimator_parameters", minimum=1
    )
    if total != trainable + frozen:
        raise ValueError("trainable/frozen parameter counts do not sum to total")
    if total != estimator + non_estimator:
        raise ValueError("estimator/non-estimator parameter counts do not sum to total")
    if frozen != 0 or trainable != total:
        raise ValueError("C-learning arms must not retain frozen parameters")
    if (job["condition"] == "learned_c" and estimator == 0) or (
        job["condition"] == "fixed_c" and estimator != 0
    ):
        raise ValueError("estimator parameter count disagrees with C-learning condition")
    for key in ("initial_state_sha256", "shared_backbone_initial_state_sha256"):
        if not isinstance(child.get(key), str) or not _SHA256.fullmatch(child[key]):
            raise ValueError(f"{key} must be a SHA-256 digest")
    _validate_optimizer_evidence(child, total=total, trainable=trainable)
    if not child.get("versions") or not isinstance(child["versions"], dict):
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
        errors.append("datasets must list unique supported datasets")
    for key, value in (
        ("schema_version", 1),
        ("suite", SUITE),
        ("conditions", CONDITIONS),
        ("source_integrity_valid", True),
    ):
        if not _same(manifest.get(key), value):
            errors.append(f"manifest.{key} mismatch")
    if manifest.get("status") not in {"running", "failed", "passed"}:
        errors.append("invalid manifest.status")
    try:
        _integer(config.get("model_seed"), "model_seed")
        for key in ("epochs", "patience", "batch_size"):
            _integer(config.get(key), key, minimum=1)
        _integer(config.get("workers"), "workers")
        workers_by_dataset = config.get("workers_by_dataset")
        if workers_by_dataset is not None and not _same(
            workers_by_dataset,
            {
                dataset: config["workers"] if dataset == "ppi" else 0
                for dataset in datasets
            },
        ):
            raise ValueError("workers_by_dataset is inconsistent")
        for key, value in COMMON.items():
            if not _same(config.get(key), value):
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
        expected_workers = config.get("workers", 0) if dataset == "ppi" else 0
        if job.get("workers", expected_workers) != expected_workers:
            errors.append(f"{dataset}/{condition}: job workers must be {expected_workers}")
        if job.get("status") not in {"pending", "running", "failed", "passed"}:
            errors.append(f"{key}: invalid job status")
        try:
            output = _contained(job.get("output_dir"), root, f"{key} output")
            metrics = _contained(job.get("metrics_path"), root, f"{key} metrics")
            if not metrics.is_relative_to(output):
                raise ValueError(f"{key}: metrics outside job output")
        except ValueError as exc:
            errors.append(str(exc))
    reports = []
    for dataset in datasets:
        loaded, rows = {}, []
        for condition, spec in CONDITIONS.items():
            job = indexed.get((dataset, condition))
            row = {
                "condition": condition,
                **spec,
                "status": job.get("status") if job else "missing",
                "validation": None,
                "validation_percent": None,
                "best_epoch": None,
                "epochs_run": None,
                "trainable_parameters": None,
                "total_parameters": None,
                "estimator_parameters": None,
                "non_estimator_parameters": None,
                "frozen_parameters": None,
                "best_validation_diagnostics": _best_validation_summary(None),
            }
            if job and job.get("error"):
                row["error"] = str(job["error"])
            if job and job.get("status") == "passed":
                try:
                    child = _load(root, job, config)
                    loaded[condition] = child
                    for key in (
                        "validation",
                        "best_epoch",
                        "epochs_run",
                        "train_loss",
                        "trainable_parameters",
                        "estimator_parameters",
                        "non_estimator_parameters",
                        "frozen_parameters",
                        "total_parameters",
                        "elapsed_seconds",
                        "peak_cuda_allocated_bytes",
                    ):
                        row[key] = child[key]
                    row["validation_percent"] = 100.0 * child["validation"]
                    row["best_validation_diagnostics"] = _best_validation_summary(
                        child.get("diagnostics")
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    row.update(status="invalid", error=str(exc))
                    errors.append(f"{dataset}/{condition}: {exc}")
            rows.append(row)
        reference = next(iter(loaded.values()), None)
        metadata = _matched_metadata(reference) if reference else None
        if reference:
            for condition, child in loaded.items():
                actual = _matched_metadata(child)
                for key in metadata:
                    if not _same(metadata[key], actual[key]):
                        errors.append(f"{dataset}/{condition}: held-fixed {key} mismatch")
        parameter_contract = None
        if len(loaded) == len(CONDITIONS):
            learned, fixed = loaded["learned_c"], loaded["fixed_c"]
            learned_gate = learned["estimator_parameters"]
            total_delta = learned["total_parameters"] - fixed["total_parameters"]
            if learned["non_estimator_parameters"] != fixed["non_estimator_parameters"]:
                errors.append(f"{dataset}: non-estimator parameter count mismatch")
            if total_delta != learned_gate:
                errors.append(
                    f"{dataset}: total parameter difference does not equal learned estimator size"
                )
            if learned["initial_state_sha256"] == fixed["initial_state_sha256"]:
                errors.append(f"{dataset}: structurally different arms share a full-state hash")
            parameter_contract = {
                "learned_total_parameters": learned["total_parameters"],
                "fixed_total_parameters": fixed["total_parameters"],
                "learned_estimator_parameters": learned_gate,
                "fixed_estimator_parameters": fixed["estimator_parameters"],
                "total_parameter_difference": total_delta,
                "shared_non_estimator_parameters": fixed["non_estimator_parameters"],
                "verified": (
                    learned["non_estimator_parameters"] == fixed["non_estimator_parameters"]
                    and total_delta == learned_gate
                    and fixed["estimator_parameters"] == 0
                ),
            }
        reports.append(
            {
                "dataset": dataset,
                "metric_name": _metric_name(dataset),
                "model_seed": config.get("model_seed"),
                "conditions": rows,
                "complete": len(loaded) == len(CONDITIONS),
                "held_fixed": metadata,
                "parameter_contract": parameter_contract,
                "learned_minus_fixed": None,
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
            delta = scores["learned_c"] - scores["fixed_c"]
            item["learned_minus_fixed"] = {
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
        "# Learned C vs fixed C=1: node-degree normalization",
        "",
        f"Status: **{report['status']}**; model seed {report['model_seed']}; validation only.",
        "",
    ]
    if report["errors"]:
        lines += ["## Integrity errors: contrasts withheld", ""]
        lines += [f"- {_cell(error)}" for error in report["errors"]] + [""]
    for dataset in report["datasets"]:
        lines += [
            f"## {dataset['dataset']} ({dataset['metric_name']}, higher is better)",
            "",
            "| Condition | Status | Validation (%) | Best epoch | Epochs run "
            "| Total parameters | Estimator parameters | Frozen parameters |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in dataset["conditions"]:
            cells = [row["condition"], row["status"], _display(row["validation_percent"])]
            cells += [
                str(row[key]) if row[key] is not None else "—"
                for key in (
                    "best_epoch",
                    "epochs_run",
                    "total_parameters",
                    "estimator_parameters",
                    "frozen_parameters",
                )
            ]
            lines += ["| " + " | ".join(cells) + " |"]
        contrast = dataset["learned_minus_fixed"]
        lines += [
            "",
            "Learned − fixed: "
            + (
                _display(contrast["percentage_points"], signed=True) + " pp."
                if contrast
                else "withheld until both arms pass integrity checks."
            ),
            "",
        ]
        lines += _diagnostic_markdown(dataset["conditions"])
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
        "total_parameters",
        "estimator_parameters",
        "non_estimator_parameters",
        "trainable_parameters",
        "frozen_parameters",
        "learned_minus_fixed_pp",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for dataset in report["datasets"]:
        for row in dataset["conditions"]:
            record = {key: row[key] for key in fields if key in row}
            record.update({key: dataset[key] for key in ("dataset", "metric_name", "model_seed")})
            contrast = dataset["learned_minus_fixed"]
            record["learned_minus_fixed_pp"] = contrast["percentage_points"] if contrast else None
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
