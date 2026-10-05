"""Parse received console aggregates; do not reevaluate models or seed CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"


def numbers(value, expected):
    values = [float(item) for item in re.findall(NUMBER, value)]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} numeric values in {value!r}")
    return values


def parse(source: Path):
    data = source.read_bytes()
    text = data.decode("utf-8-sig")
    completion, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    tables = {}
    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:]
        if line.startswith("| ") and not line.startswith("| ---"):
            tables.setdefault(section, []).append([item.strip() for item in line.strip().strip("|").split("|")])
    expected = {
        "분류 결과": 60, "직접 비교 52개": 312,
        "정규화 상호작용 12개": 72, "같은 checkpoint의 frozen 개입과 실제 작용": 120,
    }
    for name, count in expected.items():
        if len(tables.get(name, [])) != count + 1:
            raise ValueError(f"Missing/truncated aggregate table: {name}")
    metrics = []
    for row in tables["분류 결과"][1:]:
        if len(row) != 8:
            raise ValueError(row)
        record = dict(dataset=row[0], condition=row[1], legacy_six_condition=row[2] == "True", learning_rate=float(row[3]))
        for name, value in zip(("train_accuracy", "validation_accuracy", "test_accuracy", "test_ce"), row[4:]):
            mean, std = numbers(value, 2)
            record[name] = {"reported_mean": mean, "reported_sample_std": std}
        metrics.append(record)
    direct = []
    for row in tables["직접 비교 52개"][1:]:
        if len(row) != 7:
            raise ValueError(row)
        mean, low, high = numbers(row[5], 3)
        direct.append(dict(dataset=row[0], condition=row[1], reference=row[2], metric=row[3], unit=row[4],
                           reported_mean=mean, reported_ci95=[low, high], comparison=row[6]))
    interactions = []
    for row in tables["정규화 상호작용 12개"][1:]:
        if len(row) != 5:
            raise ValueError(row)
        mean, low, high = numbers(row[4], 3)
        interactions.append(dict(dataset=row[0], interaction=row[1], metric=row[2], unit=row[3],
                                 reported_mean=mean, reported_ci95=[low, high]))
    branches = []
    for row in tables["같은 checkpoint의 frozen 개입과 실제 작용"][1:]:
        if len(row) != 8:
            raise ValueError(row)
        gain, gain_std = numbers(row[3], 2)
        ratio, ratio_std = numbers(row[4], 2)
        branches.append(dict(dataset=row[0], condition=row[1], layer=int(row[2]),
                             gain_mean=gain, gain_sample_std=gain_std,
                             matched_delta_over_off_mean=ratio, matched_delta_over_off_sample_std=ratio_std,
                             applied_context_norm=float(row[5]), applied_cross_energy_before=float(row[6]),
                             applied_cross_energy_after=float(row[7])))
    for records, keys in ((metrics, ("dataset", "condition")),
                          (direct, ("dataset", "condition", "reference", "metric", "comparison")),
                          (interactions, ("dataset", "interaction", "metric")),
                          (branches, ("dataset", "condition", "layer"))):
        if len({tuple(record[key] for key in keys) for record in records}) != len(records):
            raise ValueError("Duplicate aggregate row key")
    if completion["coverage"]["total_runs"] != 840 or completion["new_optimizer_updates"] != 420000:
        raise ValueError("Received run count differs from this FULL report contract")
    ci_counts = Counter()
    for record in direct:
        low, high = record["reported_ci95"]
        direction = "positive" if low > 0 else "negative" if high < 0 else "includes_zero"
        ci_counts[f"{record['comparison']}:{record['metric']}:{direction}"] += 1
    return {
        "source_sha256": hashlib.sha256(data).hexdigest(), "source_bytes": len(data),
        "evidence_role": "Derived from rounded console summary; not raw per-seed CSV or independent server reevaluation",
        "intervals_recomputed": False, "multiple_comparison_adjustment_applied": False,
        "completion": completion, "metrics": metrics, "direct_comparisons": direct,
        "interactions": interactions, "branch_summaries": branches,
        "reported_interval_direction_counts": dict(sorted(ci_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = parse(args.source)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "source_sha256": value["source_sha256"],
                      "rows": {name: len(value[name]) for name in ("metrics", "direct_comparisons", "interactions", "branch_summaries")},
                      "interval_counts": value["reported_interval_direction_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
