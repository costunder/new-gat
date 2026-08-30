#!/usr/bin/env python3
"""Aggregate seed-aligned paper artifacts without mixing experimental axes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AGGREGATE_FILENAMES = {"metrics.json", "runtime.json", "summary.json"}
CONDITIONS = {
    "no_pe",
    "raw",
    "set",
    "projector",
    "isotropic",
    "edge_only",
    "gradient_only",
    "full",
    "full_flux_supervised",
    "full_joint",
    "flux_ls",
    "node_message_nnls",
    "oracle",
    "conductance",
    "cycle_set",
    "conductance_model",
    "fixed_bfs",
    "multi_chart",
}


@dataclass(frozen=True)
class AggregateMetricRule:
    """An explicit contract for one family of aggregate result fields."""

    name: str
    track: str
    artifact_pattern: re.Pattern[str]
    metric_pattern: re.Pattern[str]
    pairable: bool


def _metric_rule(
    name: str,
    track: str,
    artifact_pattern: str,
    metric_pattern: str,
    *,
    pairable: bool,
) -> AggregateMetricRule:
    return AggregateMetricRule(
        name=name,
        track=track,
        artifact_pattern=re.compile(artifact_pattern),
        metric_pattern=re.compile(metric_pattern),
        pairable=pairable,
    )


_CONDUCTANCE_PREDICTION_METRICS = (
    r"(?:graph_macro_flux_relative_l2|graph_macro_node_message_relative_l2|"
    r"graph_macro_next_state_relative_l2|graph_macro_log_conductance_rmse|"
    r"graph_macro_conductance_pearson|graph_macro_conductance_spearman|"
    r"graph_macro_observed_fit_relative_l2)"
)
_CONDUCTANCE_BASELINES = (
    r"(?:isotropic|edge_only|gradient_only|full|full_flux_supervised|full_joint|"
    r"flux_ls|node_message_nnls|oracle)"
)
_CYCLE_TEST_SPLITS = r"(?:id_test|size_ood|family_ood|test)"
_CYCLE_SUPERVISED_METRICS = (
    rf"{_CYCLE_TEST_SPLITS}\.(?:"
    r"macro_(?:normalized_)?mae|"
    r"levels\.(?:edge|node|graph)\.macro_(?:normalized_)?mae|"
    r"levels\.(?:edge|node|graph)\.targets\.[^.]+\."
    r"(?:mae|rmse|normalized_mae|graph_macro_mae|rounded_exact_accuracy))"
)
_TREE_EVALUATION_METRICS = (
    r"(?:mae|normalized_mae|rmse|graph_macro_mae|worst_chart_mae|"
    r"chart_prediction_std|rounded_exact_vector_accuracy|accuracy|graph_macro_accuracy|"
    r"worst_chart_accuracy|chart_probability_std|prediction_flip_rate)"
)

# This registry is intentionally closed.  Adding a numeric field to a result JSON
# does not make it a paper metric until a reviewer deliberately extends this
# schema.  Runtime, memory, parameter counts, configuration, sample counts, seed
# axes, and optimization histories therefore cannot leak into hypothesis tests.
# Published competitor scores belong in the cited manuscript table, not in this
# run registry or in paired statistics with our own experiment seeds.
PAPER_METRIC_SCHEMA_VERSION = 4
PAPER_METRIC_SCHEMA: tuple[AggregateMetricRule, ...] = (
    _metric_rule(
        "conductance.our_model.test",
        "conductance_gat",
        r"metrics\.json",
        r"datasets\.[^.]+\.models\.conductance\.test",
        pairable=False,
    ),
    _metric_rule(
        "cycle.our_model.test",
        "cycle_pe",
        r"metrics\.json",
        r"datasets\.[^.]+\.models\.cycle_set\.test",
        pairable=False,
    ),
    _metric_rule(
        "conductance.core.prediction",
        "conductance_gat",
        r"summary\.json",
        rf"results\.core\.s[1-4]\.baselines\.{_CONDUCTANCE_BASELINES}\."
        rf"(?:unseen_graph_test|seen_graph_new_excitation_test)\."
        rf"{_CONDUCTANCE_PREDICTION_METRICS}",
        pairable=True,
    ),
    _metric_rule(
        "conductance.core.rollout",
        "conductance_gat",
        r"summary\.json",
        rf"results\.core\.s3\.baselines\.{_CONDUCTANCE_BASELINES}\.rollout\."
        r"(?:horizon_[1-9][0-9]*_relative_l2|final_norm_over_initial|"
        r"dissipation_violation_fraction)",
        pairable=True,
    ),
    _metric_rule(
        "conductance.core.factorial",
        "conductance_gat",
        r"summary\.json",
        rf"results\.core\.s4\.factorial\.[0-9]+\.{_CONDUCTANCE_PREDICTION_METRICS}",
        # The current factorial JSON stores the baseline as a sibling string,
        # not in the numeric path.  Keep the measurements but do not manufacture
        # a paired comparison from opaque list indices.
        pairable=False,
    ),
    _metric_rule(
        "conductance.public.test",
        "conductance_gat",
        r"summary\.json",
        r"results\.public\.[^.]+\.baselines\."
        r"conductance_model\.test\.(?:macro_f1|roc_auc)",
        pairable=False,
    ),
    _metric_rule(
        "cycle.supervised.test",
        "cycle_pe",
        r"(?:core|zinc)/[^/]+/(?:no_pe|raw|set|projector)/metrics\.json",
        _CYCLE_SUPERVISED_METRICS,
        pairable=True,
    ),
    _metric_rule(
        "cycle.brec.official",
        "cycle_pe",
        r"brec/(?:no_pe|raw|set|projector)/metrics\.json",
        r"per_seed\.[0-9]+\.(?:Correct|Fail|Real_correct)",
        # BREC owns its internal ten-seed protocol; the outer model-seed axis is
        # explicitly disabled and no cross-variant paired test is defined here.
        pairable=False,
    ),
    _metric_rule(
        "cycle.brec.custom",
        "cycle_pe",
        r"brec/(?:no_pe|raw|set|projector)/metrics\.json",
        r"(?:success_rate|categories\.[^.]+\.success_rate)",
        pairable=True,
    ),
    _metric_rule(
        "tree.downstream.test",
        "tree_augmentation",
        r"summary\.json",
        rf"models\.(?:fixed_bfs|multi_chart)\.quadrants\.[^.]+\."
        rf"{_TREE_EVALUATION_METRICS}",
        pairable=True,
    ),
    _metric_rule(
        "tree.precomputed_improvement",
        "tree_augmentation",
        r"summary\.json",
        r"comparison\.quadrant_improvements\.[^.]+\."
        r"(?:mae_improvement_fixed_minus_multi|worst_chart_mae_improvement_fixed_minus_multi|"
        r"chart_std_improvement_fixed_minus_multi)",
        pairable=False,
    ),
)

# Efficiency observations are emitted as raw, seed-addressable rows in a
# separate table.  They are never bootstrapped or paired.  The allowlist is
# deliberately limited to elapsed time, peak accelerator memory, and active
# trainable parameter counts; epochs, batch size, workers, and seeds are not
# efficiency outcomes.
EFFICIENCY_METRIC_SCHEMA_VERSION = 3
EFFICIENCY_METRIC_SCHEMA: tuple[AggregateMetricRule, ...] = (
    _metric_rule(
        "conductance.our_model.efficiency",
        "conductance_gat",
        r"metrics\.json",
        r"datasets\.[^.]+\.models\.conductance\."
        r"(?:trainable_parameters|elapsed_seconds|peak_gpu_memory_bytes)",
        pairable=False,
    ),
    _metric_rule(
        "cycle.our_model.efficiency",
        "cycle_pe",
        r"metrics\.json",
        r"datasets\.[^.]+\.models\.cycle_set\."
        r"(?:trainable_parameters|elapsed_seconds|peak_gpu_memory_bytes)",
        pairable=False,
    ),
    _metric_rule(
        "conductance.runtime",
        "conductance_gat",
        r"summary\.json",
        r"runtime\.(?:elapsed_seconds|cuda_peak_allocated_bytes|cuda_peak_reserved_bytes)",
        pairable=False,
    ),
    _metric_rule(
        "conductance.active_parameters",
        "conductance_gat",
        r"summary\.json",
        r"results\.public\.[^.]+\.baselines\."
        r"conductance_model\.parameter_count",
        pairable=False,
    ),
    _metric_rule(
        "cycle.runtime",
        "cycle_pe",
        r"(?:core|zinc)/[^/]+/(?:no_pe|raw|set|projector)/runtime\.json",
        r"(?:total_train_evaluation_wall_seconds|peak_gpu_memory_bytes)",
        pairable=False,
    ),
    _metric_rule(
        "tree.runtime",
        "tree_augmentation",
        r"summary\.json",
        r"runtime\.(?:elapsed_seconds|peak_gpu_allocated_bytes|peak_gpu_reserved_bytes)",
        pairable=False,
    ),
)


def _flag(command: list[str], name: str) -> str | None:
    value: str | None = None
    for index, token in enumerate(command[:-1]):
        if token == name:
            value = command[index + 1]
    return value


def _integer_flag(command: list[str], *names: str, default: int | None = None) -> int | None:
    for name in names:
        value = _flag(command, name)
        if value is not None:
            return int(value)
    return default


def _factors(entry: dict[str, Any]) -> dict[str, int | None]:
    command = [str(value) for value in entry.get("command", [])]
    legacy = _integer_flag(command, "--seed")
    data_seed = _integer_flag(command, "--data-seed", default=legacy)
    model_seed = _integer_flag(command, "--model-seed", default=legacy)
    # Official BREC owns a separate ten-seed search protocol.  The outer
    # runner's placeholder model seed is deliberately not a BREC sample axis.
    if _flag(command, "--suite") == "brec" and _flag(command, "--brec-protocol") == "official":
        model_seed = None
    return {
        "model_seed": model_seed,
        "data_seed": data_seed,
        "split_seed": _integer_flag(command, "--split-seed", default=data_seed),
        "chart_seed": _integer_flag(command, "--chart-seed", default=data_seed),
    }


def _flatten_numeric(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, bool) or value is None:
        return rows
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            rows.append((".".join(prefix) or "value", numeric))
        return rows
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_flatten_numeric(item, (*prefix, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_flatten_numeric(item, (*prefix, str(index))))
    return rows


def _select_metric_rule(
    schema: tuple[AggregateMetricRule, ...],
    track: str,
    artifact: str,
    metric: str,
    *,
    table: str,
) -> AggregateMetricRule | None:
    matches = [
        rule
        for rule in schema
        if rule.track == track
        and rule.artifact_pattern.fullmatch(artifact)
        and rule.metric_pattern.fullmatch(metric)
    ]
    if len(matches) > 1:
        names = ", ".join(rule.name for rule in matches)
        raise RuntimeError(
            f"ambiguous {table} metric schema for {track}:{artifact}:{metric}: {names}"
        )
    return matches[0] if matches else None


def _nested_json_value(payload: Any, dotted_path: str) -> Any:
    value = payload
    for token in dotted_path.split("."):
        if not isinstance(value, dict) or token not in value:
            return None
        value = value[token]
    return value


def _efficiency_rule_is_applicable(rule: AggregateMetricRule, payload: Any, metric: str) -> bool:
    if rule.name != "conductance.active_parameters":
        return True
    parent = metric.rsplit(".", 1)[0]
    return (
        _nested_json_value(payload, f"{parent}.parameter_count_policy")
        == "trainable_active_parameters_only"
    )


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summary(values: list[float], *, key: str, bootstrap_samples: int) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    if len(values) == 1 or bootstrap_samples == 0:
        low = high = mean
    else:
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        generator = random.Random(seed)
        means = sorted(
            statistics.fmean(generator.choice(values) for _ in values)
            for _ in range(bootstrap_samples)
        )
        low = _quantile(means, 0.025)
        high = _quantile(means, 0.975)
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def _condition_template(artifact: str, metric: str) -> tuple[str, str, str] | None:
    artifact_parts = artifact.replace("\\", "/").split("/")
    metric_parts = metric.split(".")
    occurrences: list[tuple[str, int, str]] = []
    for index, token in enumerate(artifact_parts):
        if token in CONDITIONS:
            occurrences.append(("artifact", index, token))
    for index, token in enumerate(metric_parts):
        if token in CONDITIONS:
            occurrences.append(("metric", index, token))
    if len(occurrences) != 1:
        return None
    location, index, condition = occurrences[0]
    if location == "artifact":
        artifact_parts[index] = "{condition}"
    else:
        metric_parts[index] = "{condition}"
    return "/".join(artifact_parts), ".".join(metric_parts), condition


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    fieldnames = sorted({key for row in rows for key in row})
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def aggregate_manifest(
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    """Aggregate explicitly registered paper metrics from completed artifacts.

    Data, split, and chart seeds are grouping keys.  Only model seeds are averaged,
    so changing a dataset or chart does not silently inflate a model-seed standard
    deviation.  Legacy children that expose only ``--seed`` remain readable, but
    all four axes then intentionally resolve to that same value in the audit output.

    Numeric fields that are not in :data:`PAPER_METRIC_SCHEMA` or
    :data:`EFFICIENCY_METRIC_SCHEMA` are counted for the audit trail and otherwise
    ignored.  Explicit runtime, peak-memory, and active-parameter observations go
    to a raw efficiency table; they never enter bootstrap summaries or paired tests.
    Configuration, sample-count, seed, and optimizer-history numbers enter neither
    table.
    """

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else manifest_path.parent / "aggregate"
    )
    samples: list[dict[str, Any]] = []
    efficiency_samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    numeric_fields_seen = 0
    ignored_numeric_fields = 0
    for entry in manifest.get("commands", []):
        name = str(entry.get("name", "unknown"))
        if name == "gpu_preflight":
            continue
        command = [str(value) for value in entry.get("command", [])]
        track = name.split(":", 1)[0]
        suite = _flag(command, "--suite") or "unknown"
        factors = _factors(entry)
        return_code = int(entry.get("returncode", 1))
        artifact_errors = list(entry.get("artifact_errors") or [])
        output_value = entry.get("output")
        output_path = Path(output_value).expanduser().resolve() if output_value else None
        if return_code != 0 or artifact_errors or output_path is None or not output_path.exists():
            log_text = ""
            log_value = entry.get("log")
            if log_value:
                try:
                    log_text = Path(str(log_value)).read_text(encoding="utf-8", errors="replace")[
                        -100_000:
                    ]
                except OSError:
                    pass
            error_text = " | ".join(str(error) for error in artifact_errors)
            searchable_error = f"{error_text}\n{log_text}".casefold()
            failures.append(
                {
                    "command": name,
                    "track": track,
                    "suite": suite,
                    "returncode": return_code,
                    "artifact_errors": error_text,
                    "oom": any(
                        marker in searchable_error
                        for marker in (
                            "out of memory",
                            "outofmemoryerror",
                            "cublas_status_alloc_failed",
                            "cuda error: memory allocation",
                        )
                    ),
                    **factors,
                }
            )
            continue
        candidates = (
            [output_path]
            if output_path.is_file() and output_path.name in AGGREGATE_FILENAMES
            else sorted(
                path for path in output_path.rglob("*.json") if path.name in AGGREGATE_FILENAMES
            )
        )
        for artifact_path in candidates:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact = (
                artifact_path.name
                if output_path.is_file()
                else str(artifact_path.relative_to(output_path)).replace("\\", "/")
            )
            numeric_fields = _flatten_numeric(payload)
            numeric_fields_seen += len(numeric_fields)
            for metric, value in numeric_fields:
                paper_rule = _select_metric_rule(
                    PAPER_METRIC_SCHEMA, track, artifact, metric, table="paper"
                )
                efficiency_rule = _select_metric_rule(
                    EFFICIENCY_METRIC_SCHEMA, track, artifact, metric, table="efficiency"
                )
                if efficiency_rule is not None and not _efficiency_rule_is_applicable(
                    efficiency_rule, payload, metric
                ):
                    efficiency_rule = None
                if paper_rule is not None and efficiency_rule is not None:
                    raise RuntimeError(
                        f"metric belongs to paper and efficiency schemas: "
                        f"{track}:{artifact}:{metric}"
                    )
                if paper_rule is None and efficiency_rule is None:
                    ignored_numeric_fields += 1
                    continue
                common = {
                    "command": name,
                    "track": track,
                    "suite": suite,
                    "artifact": artifact,
                    "artifact_path": str(artifact_path),
                    "metric": metric,
                    "value": value,
                    **factors,
                }
                if paper_rule is not None:
                    samples.append(
                        {
                            **common,
                            "metric_rule": paper_rule.name,
                            "pairable": paper_rule.pairable,
                        }
                    )
                else:
                    assert efficiency_rule is not None
                    efficiency_samples.append(
                        {
                            **common,
                            "metric_rule": efficiency_rule.name,
                        }
                    )

    grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        key = (
            row["track"],
            row["suite"],
            row["artifact"],
            row["metric"],
            row["metric_rule"],
            row["data_seed"],
            row["split_seed"],
            row["chart_seed"],
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        track, suite, artifact, metric, metric_rule, data_seed, split_seed, chart_seed = key
        values = [float(row["value"]) for row in rows]
        model_seeds = sorted(
            {int(row["model_seed"]) for row in rows if row["model_seed"] is not None}
        )
        identity = "|".join(str(value) for value in key)
        summaries.append(
            {
                "track": track,
                "suite": suite,
                "artifact": artifact,
                "metric": metric,
                "metric_rule": metric_rule,
                "data_seed": data_seed,
                "split_seed": split_seed,
                "chart_seed": chart_seed,
                "model_seeds": ",".join(str(seed) for seed in model_seeds),
                **_summary(values, key=identity, bootstrap_samples=bootstrap_samples),
            }
        )

    paired_samples: defaultdict[tuple[Any, ...], dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in samples:
        if not row["pairable"]:
            continue
        template = _condition_template(str(row["artifact"]), str(row["metric"]))
        if template is None or row["model_seed"] is None:
            continue
        artifact_template, metric_template, condition = template
        key = (
            row["track"],
            row["suite"],
            artifact_template,
            metric_template,
            row["metric_rule"],
            row["data_seed"],
            row["split_seed"],
            row["chart_seed"],
        )
        paired_samples[key][condition][int(row["model_seed"])] = float(row["value"])
    paired: list[dict[str, Any]] = []
    for key, by_condition in sorted(paired_samples.items(), key=lambda item: str(item[0])):
        conditions = sorted(by_condition)
        for left_index, left in enumerate(conditions):
            for right in conditions[left_index + 1 :]:
                common = sorted(set(by_condition[left]) & set(by_condition[right]))
                if not common:
                    continue
                differences = [
                    by_condition[right][seed] - by_condition[left][seed] for seed in common
                ]
                identity = "|".join(str(value) for value in (*key, left, right))
                (
                    track,
                    suite,
                    artifact_template,
                    metric_template,
                    metric_rule,
                    data_seed,
                    split_seed,
                    chart_seed,
                ) = key
                difference_summary = _summary(
                    differences,
                    key=identity,
                    bootstrap_samples=bootstrap_samples,
                )
                paired.append(
                    {
                        "track": track,
                        "suite": suite,
                        "artifact_template": artifact_template,
                        "metric_template": metric_template,
                        "metric_rule": metric_rule,
                        "condition_left": left,
                        "condition_right": right,
                        "difference_definition": "right_minus_left",
                        "data_seed": data_seed,
                        "split_seed": split_seed,
                        "chart_seed": chart_seed,
                        "model_seeds": ",".join(str(seed) for seed in common),
                        **difference_summary,
                        "effect_size_name": "paired_cohens_dz",
                        "effect_size": (
                            difference_summary["mean"] / difference_summary["sample_std"]
                            if len(differences) > 1
                            and difference_summary["sample_std"]
                            > max(1e-12, abs(difference_summary["mean"]) * 1e-12)
                            else None
                        ),
                    }
                )

    payload = {
        "schema_version": 2,
        "paper_metric_schema_version": PAPER_METRIC_SCHEMA_VERSION,
        "efficiency_metric_schema_version": EFFICIENCY_METRIC_SCHEMA_VERSION,
        "paper_metric_rules": [rule.name for rule in PAPER_METRIC_SCHEMA],
        "efficiency_metric_rules": [rule.name for rule in EFFICIENCY_METRIC_SCHEMA],
        "source_manifest": str(manifest_path),
        "source_run_id": manifest.get("run_id"),
        "source_status": manifest.get("status"),
        "seed_policy": (
            "group by data/split/chart seed; summarize and pair only aligned model seeds"
        ),
        "bootstrap_samples": bootstrap_samples,
        "numeric_fields_seen": numeric_fields_seen,
        "ignored_numeric_fields": ignored_numeric_fields,
        "sample_rows": len(samples),
        "efficiency_rows": len(efficiency_samples),
        "metric_groups": len(summaries),
        "paired_groups": len(paired),
        "failed_commands": len(failures),
        "oom_failures": sum(bool(row["oom"]) for row in failures),
        "files": {
            "samples": "samples.csv",
            "metrics": "metrics.csv",
            "paired": "paired.csv",
            "efficiency": "efficiency.csv",
            "failures": "failures.csv",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "samples.csv", samples)
    _write_csv(output_dir / "metrics.csv", summaries)
    _write_csv(output_dir / "paired.csv", paired)
    _write_csv(output_dir / "efficiency.csv", efficiency_samples)
    _write_csv(output_dir / "failures.csv", failures)
    _atomic_text(
        output_dir / "aggregate.json",
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative")
    payload = aggregate_manifest(
        args.manifest,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
