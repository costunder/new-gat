"""Fail-closed fixed-graph direct-C comparison; stdlib only, no training or test labels."""

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
    _best_validation_summary,
    _cell,
    _contained,
    _diagnostic_markdown,
    _display,
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
    "Both arms train freshly with the same initial full state, ordered graph, official cache, "
    "nongate Adam L2 and early-stopping policy. No V1 or historical score is reused.",
    "Direct C has graph-specific edge parameters, not an edge-generator MLP. It is transductive "
    "and cannot transfer these parameters to held-out PPI graphs or unseen edges.",
    "The contrast is direct_c minus fixed_c (percentage points), not an external GCN/GAT baseline.",
    "Fixed C=1 keeps zero edge-log parameters frozen and outside the optimizer. Its reported "
    "edge-parameter L2 is not an active learning signal.",
    "Both arms use node-degree normalization: common positive C scale cancels and rho=.95 on "
    "nonisolated nodes is imposed, not evidence of useful learned conductances.",
    "Elapsed time and peak CUDA allocation cover the training loop including diagnostic and "
    "checkpoint overhead; selected epochs and epochs run can differ. They are not isolated "
    "kernel benchmarks or measured speedups over V1 or spectral models.",
    "Sparse diagonal-C propagation has O((n+m)*d) arithmetic; no eigendecomposition is needed. "
    "This complexity statement is theoretical, not a measured scalability result. Full-graph "
    "training still stores node states, topology, edge parameters and optimizer state.",
    "Independent edge parameters can receive zero task gradients outside training-label "
    "receptive fields. Full-graph execution does not guarantee that every C entry learns; "
    "metrics.json records exact per-step edge-gradient coverage without an epsilon threshold.",
    "Repeated validation-guided choices can overfit validation. Identical initial states do not "
    "guarantee bitwise-identical CUDA scatter trajectories.",
]


class ComparisonIntegrityError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("Direct-C V2 comparison integrity failed: " + "; ".join(report["errors"]))


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
    edges = _integer(topology["num_edges"], "topology.num_edges")
    if not isinstance(topology["incidence_sha256"], str) or not SHA256.fullmatch(
        topology["incidence_sha256"]
    ):
        raise ValueError("topology.incidence_sha256 must be a SHA-256 digest")
    total = _integer(child.get("total_parameters"), "total_parameters", minimum=1)
    trainable = _integer(child.get("trainable_parameters"), "trainable_parameters", minimum=1)
    frozen = _integer(child.get("frozen_parameters"), "frozen_parameters")
    expected_frozen = config["layers"] * edges if job["condition"] == "fixed_c" else 0
    if total != trainable + frozen or frozen != expected_frozen:
        raise ValueError("parameter counts disagree with direct-C/frozen-edge condition")
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
                "best_validation_diagnostics": _best_validation_summary(None),
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
                "direct_minus_fixed": None,
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
            delta = scores["direct_c"] - scores["fixed_c"]
            item["direct_minus_fixed"] = {"score_delta": delta, "percentage_points": 100.0 * delta}
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
        "# Direct edge C V2 vs fixed C=1",
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
        contrast = dataset["direct_minus_fixed"]
        lines += [
            "",
            "Direct − fixed: "
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
        "direct_minus_fixed_pp",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for dataset in report["datasets"]:
        for row in dataset["conditions"]:
            record = {key: row[key] for key in fields if key in row}
            record.update({key: dataset[key] for key in ("dataset", "metric_name", "model_seed")})
            contrast = dataset["direct_minus_fixed"]
            record["direct_minus_fixed_pp"] = contrast["percentage_points"] if contrast else None
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
