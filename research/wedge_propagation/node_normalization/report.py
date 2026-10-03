"""Report the five fresh-training conditions and frozen normalization probes."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from ..classification.common import write_csv
from .evaluation import LEARNED, intervention_variants, manifest_count

CONDITIONS = ("fixed_wedge", *LEARNED)
COMPARISONS = (
    ("learned_wedge_local_raw", "learned_wedge_raw"),
    ("learned_wedge_local_rms", "learned_wedge_rms"),
    ("learned_wedge_local_raw", "fixed_wedge"),
    ("learned_wedge_local_rms", "fixed_wedge"),
    ("learned_wedge_raw", "fixed_wedge"),
    ("learned_wedge_rms", "fixed_wedge"),
    ("learned_wedge_rms", "learned_wedge_raw"),
    ("learned_wedge_local_rms", "learned_wedge_local_raw"),
)
LABELS = {
    "fixed_wedge": "fixed C=1",
    "learned_wedge_raw": "global/raw",
    "learned_wedge_rms": "global/RMS",
    "learned_wedge_local_raw": "node/raw",
    "learned_wedge_local_rms": "node/RMS",
}
STRENGTH_METRICS = (
    "alpha",
    "beta",
    "kappa",
    "kappa_global",
    "c_std",
    "alpha_l_to_input",
    "beta_t_to_input",
    "beta_t_to_alpha_l",
    "first_second_cosine",
)
ARTIFACTS = (
    "NODE_NORMALIZATION_SUMMARY.md",
    "paired_comparisons.csv",
    "intervention_changes.csv",
    "branch_strength_estimates.csv",
)


def _finite(value, name, *, nullable=False):
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a real measurement")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; undefined uses None")


def estimate(values):
    for value in values:
        _finite(value, "seed observation")
    if not values:
        return {"mean": None, "std": None, "lower": None, "upper": None, "count": 0}
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(values) > 1 else None
    half = (
        None
        if std is None
        else float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    )
    return {
        "mean": mean,
        "std": std,
        "count": len(values),
        "lower": None if half is None else mean - half,
        "upper": None if half is None else mean + half,
    }


def paired_comparisons(rows):
    lookup = {(r["dataset"], r["condition"], r["seed"], r["split"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate primary metric")
    datasets = sorted({r["dataset"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    result = []
    for dataset in datasets:
        for first, second in COMPARISONS:
            for split in ("train", "validation", "test"):
                for metric in ("accuracy", "ce"):
                    differences = [
                        lookup[(dataset, first, s, split)][metric]
                        - lookup[(dataset, second, s, split)][metric]
                        for s in seeds
                    ]
                    result.append(
                        {
                            "dataset": dataset,
                            "first": first,
                            "second": second,
                            "split": split,
                            "metric": metric,
                            **estimate(differences),
                        }
                    )
    return result


def intervention_changes(rows, originals, metric):
    if metric not in ("accuracy", "ce"):
        raise ValueError("paired metric must be accuracy or ce")
    lookup = {(r["dataset"], r["condition"], r["seed"], r["split"]): r for r in originals}
    if len(lookup) != len(originals):
        raise ValueError("duplicate original metric")
    per_seed = defaultdict(list)
    for row in rows:
        key = tuple(
            row[k] for k in ("dataset", "condition", "seed", "split", "intervention", "target")
        )
        per_seed[key].append(row[metric] - lookup[key[:4]][metric])
    groups = defaultdict(list)
    for key, values in per_seed.items():
        groups[(key[0], key[1], *key[3:])].append(float(np.mean(values)))
    return [
        dict(zip(("dataset", "condition", "split", "intervention", "target"), key, strict=True))
        | {"metric": metric, **estimate(values)}
        for key, values in sorted(groups.items())
    ]


def branch_estimates(rows):
    original = [r for r in rows if r["intervention"] == "original"]
    groups = defaultdict(list)
    for row in original:
        for metric in STRENGTH_METRICS:
            value = row[metric]
            _finite(value, metric, nullable=True)
            groups[(row["dataset"], row["condition"], row["layer"], metric)].append(value)
    return [
        dict(zip(("dataset", "condition", "layer", "metric"), key, strict=True))
        | {
            "undefined_seeds": sum(v is None for v in values),
            **estimate([v for v in values if v is not None]),
        }
        for key, values in sorted(groups.items())
    ]


def _validate(config, metrics, interventions, scales, selections, gates, contract):
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("all five normalization conditions are required")
    datasets, seeds = config["data"]["datasets"], config["training"]["final_seeds"]
    if not datasets or not seeds or len(set(seeds)) != len(seeds) or scales:
        raise ValueError("complete declared datasets/seeds and empty scale audit required")
    if config["profile"] not in ("full", "debug"):
        raise ValueError("explicit full/debug profile required")
    expected = {
        (d, c, s, split)
        for d in datasets
        for c in CONDITIONS
        for s in seeds
        for split in ("train", "validation", "test")
    }
    identity = ("dataset", "condition", "seed", "split")
    actual = set()
    for row in metrics:
        key = tuple(row[k] for k in identity)
        if key in actual:
            raise ValueError("duplicate primary metric")
        actual.add(key)
    if actual != expected:
        raise ValueError("incomplete primary metric coverage")
    expected_interventions = {
        (d, c, s, split, treatment, target, index)
        for d in datasets
        for c in LEARNED
        for s in seeds
        for split in ("train", "validation", "test")
        for treatment, target, index in intervention_variants(manifest_count(config))
    }
    actual = set()
    for row in interventions:
        key = tuple(row[k] for k in (*identity, "intervention", "target", "manifest_index"))
        if key in actual:
            raise ValueError("duplicate frozen intervention metric")
        actual.add(key)
    if actual != expected_interventions:
        raise ValueError("incomplete frozen intervention coverage")
    for row in metrics + interventions:
        _finite(row["ce"], "ce")
        _finite(row["accuracy"], "accuracy")
        if row["ce"] < 0 or not 0 <= row["accuracy"] <= 1:
            raise ValueError("nonnegative CE and accuracy in [0,1] required")
        shapes = config["data"].get("expected_shapes")
        if shapes is not None:
            shape = shapes[row["dataset"]]
            if row.get("num_nodes") != shape["nodes"]:
                raise ValueError("full graph node coverage changed")
            if row.get("num_labeled_nodes") != shape[row["split"]]:
                raise ValueError("labeled split coverage changed")
    expected_gates = {
        (d, c, s, layer, "original", "none", -1)
        for d in datasets
        for c in CONDITIONS
        for s in seeds
        for layer in (0, 1)
    }
    expected_gates |= {
        (d, c, s, layer, treatment, target, index)
        for d in datasets
        for c in LEARNED
        for s in seeds
        for layer in (0, 1)
        for treatment, target, index in intervention_variants(manifest_count(config))
    }
    actual = set()
    for row in gates:
        key = tuple(
            row[k]
            for k in (
                "dataset",
                "condition",
                "seed",
                "layer",
                "intervention",
                "target",
                "manifest_index",
            )
        )
        if key in actual:
            raise ValueError("duplicate layer diagnostic")
        actual.add(key)
        for metric in STRENGTH_METRICS:
            _finite(row[metric], metric, nullable=True)
            flag = metric + "_defined"
            if flag in row and row[flag] != (row[metric] is not None):
                raise ValueError("layer diagnostic defined flag disagrees with its value")
        for metric in ("alpha", "beta"):
            if row[metric] is None or not 0 <= row[metric] <= 1:
                raise ValueError("alpha and beta must be finite coefficients in [0,1]")
        for metric in ("kappa", "kappa_global"):
            if row[metric] is None or row[metric] <= 0:
                raise ValueError("normalization kappa must be positive")
        if row["c_std"] is not None and row["c_std"] < 0:
            raise ValueError("C std must be nonnegative")
    if actual != expected_gates:
        raise ValueError("incomplete layer diagnostic coverage")
    selected = {(r["dataset"], r["condition"]) for r in selections}
    if len(selected) != len(selections) or selected != {
        (d, c) for d in datasets for c in CONDITIONS
    }:
        raise ValueError("complete distinct validation-locked selections required")
    for row in selections:
        if any("test" in key.lower() for key in row):
            raise ValueError("test must not enter learning rate selection")
    if contract.get("actual_data") is not (config["profile"] == "full"):
        raise ValueError("actual-data evidence must match full/debug profile")


def _show(value, scale=1):
    return "undefined" if value is None else f"{value * scale:.4g}"


def _stat(row, scale=1):
    return f"{_show(row['mean'], scale)} ± {_show(row['std'], scale)}"


def _interval(row, scale=1):
    return f"[{_show(row['lower'], scale)}, {_show(row['upper'], scale)}]"


def _summary(config, metrics, paired, changes, branches):
    profile = config["profile"].upper()
    train = config["training"]
    lines = [
        "# Node normalization — 재학습한 다섯 조건 비교",
        "",
        f"**{profile}: {train['total_runs']} runs · "
        f"각 {train['epochs_per_run']} epoch · 최종 seed {train['final_seeds']}.**",
        "Global/raw·RMS와 node/raw·RMS를 동일한 전파 구조로 각각 새로 학습했다. "
        "Fixed C=1에서는 두 정규화가 같으므로 fixed 대조군은 하나다.",
        "`global: S_Q AᵀCA S_Q/(3κ)`, `node: S_C AᵀCA S_C/3`, "
        "`S_C=diag(AᵀCA)^(-1/2)`. C·D_C·양쪽 S_C의 미분을 유지한다.",
        "",
        "## 모든 split의 최종 checkpoint",
        "",
        "아래 값은 seed 평균 ± 표본 std다. Accuracy는 %, CE는 정규화 항을 제외한 분류 손실이다.",
        "| 데이터 | 조건 | Train acc | Train CE | Val acc | Val CE | Test acc | Test CE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in config["data"]["datasets"]:
        for condition in CONDITIONS:
            cells = []
            for split in ("train", "validation", "test"):
                rows = [
                    r
                    for r in metrics
                    if (r["dataset"], r["condition"], r["split"]) == (dataset, condition, split)
                ]
                cells.extend(
                    (
                        _stat(estimate([r["accuracy"] for r in rows]), 100),
                        _stat(estimate([r["ce"] for r in rows])),
                    )
                )
            lines.append(f"| {dataset} | {LABELS[condition]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 같은 seed의 학습 모델 차이",
        "",
        "차이는 왼쪽 모델 − 오른쪽 모델이다. Accuracy는 양수가, CE는 음수가 개선이다.",
        "| 데이터 | 비교 | Δtest acc pp ± std | 95% 구간 | Δtest CE ± std | 95% 구간 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lookup = {(r["dataset"], r["first"], r["second"], r["split"], r["metric"]): r for r in paired}
    for dataset in config["data"]["datasets"]:
        for first, second in COMPARISONS:
            key = dataset, first, second, "test"
            acc, ce = lookup[(*key, "accuracy")], lookup[(*key, "ce")]
            lines.append(
                f"| {dataset} | {LABELS[first]} − {LABELS[second]} | {_stat(acc, 100)} | "
                f"{_interval(acc, 100)} | {_stat(ce)} | {_interval(ce)} |"
            )
    lines += [
        "",
        "## 원래 forward의 실제 분기",
        "",
        "Norm은 각 층의 전체 노드·channel 값이다. node의 κ는 작동 분모 1이고, "
        "global κ 진단은 원래 max(diag(AᵀCA)/diag(Q))를 따로 기록한 값이다.",
        "| 데이터 | 조건 | 층 | C std | β | 작동 κ | global κ 진단 | "
        "αL/Z | βM/Z | βM/αL | cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lookup = {(r["dataset"], r["condition"], r["layer"], r["metric"]): r for r in branches}
    for dataset in config["data"]["datasets"]:
        for condition in LEARNED:
            for layer in (0, 1):
                cells = [
                    _stat(lookup[(dataset, condition, layer, metric)])
                    for metric in (
                        "c_std",
                        "beta",
                        "kappa",
                        "kappa_global",
                        "alpha_l_to_input",
                        "beta_t_to_input",
                        "beta_t_to_alpha_l",
                        "first_second_cosine",
                    )
                ]
                lines.append(
                    f"| {dataset} | {LABELS[condition]} | {layer} | " + " | ".join(cells) + " |"
                )
    lines += [
        "",
        "## 같은 checkpoint의 C 배치 진단 — 두 층",
        "",
        "각 칸은 Δaccuracy pp / ΔCE의 seed 평균이다. "
        "C=1·shuffle은 현재 층 원래 메시지와 norm을 맞추고, α·β는 고정했다.",
        "| 데이터 | 조건 | C=1/norm match | Shuffle/norm match | 이차 분기 제거 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lookup = {
        (r["dataset"], r["condition"], r["split"], r["intervention"], r["target"], r["metric"]): r
        for r in changes
    }
    for dataset in config["data"]["datasets"]:
        for condition in LEARNED:
            cells = []
            for treatment in (
                "c_identity_norm_matched",
                "c_position_shuffle_norm_matched",
                "second_branch_remove",
            ):
                key = dataset, condition, "test", treatment, "both"
                cells.append(
                    f"{_show(lookup[(*key, 'accuracy')]['mean'], 100)} / "
                    f"{_show(lookup[(*key, 'ce')]['mean'])}"
                )
            lines.append(f"| {dataset} | {LABELS[condition]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 판단 범위",
        "",
        "C의 분산이 있다는 것과 C 배치가 분류에 유익하다는 것은 별개다. "
        "재학습 비교와 norm을 맞춘 C 교체 진단을 함께 본다. "
        "node 정규화가 학습 실패를 해결했다고 수치 확인 없이 주장하지 않는다.",
        "Native C=1/shuffle은 해당 C로 local D_C·양쪽 S_C 또는 global κ를 다시 계산한다. "
        "Norm match는 이차 분기 크기만 맞추며 전체 update·ReLU 결과를 같게 만들지 않는다.",
        "Shuffle manifest는 seed 안에서 먼저 평균한다. 95% t 구간은 기존 public split에서 "
        "초기화 변동만 나타내며 새 split/그래프 불확실성이나 다중 비교 보정을 포함하지 않는다.",
        "기존 test를 본 뒤 정한 후속 연구다. 최고 개입을 선택하지 않으며 "
        "독립 split·그래프의 일반화 검증을 대신하지 않는다.",
        "[paired_comparisons.csv](paired_comparisons.csv), "
        "[intervention_changes.csv](intervention_changes.csv), "
        "[branch_strength_estimates.csv](branch_strength_estimates.csv)에 "
        "모든 split·층 위치·개입·정의되지 않은 관측 수를 보존했다.",
        "[metrics.csv](metrics.csv), [interventions.csv](interventions.csv), "
        "[gate_diagnostics.csv](gate_diagnostics.csv), [coverage.json](coverage.json), "
        "[contract.json](contract.json), [completion.json](completion.json).",
    ]
    if config["profile"] == "debug":
        lines += [
            "",
            "**DEBUG fixture·축소 예산의 연결 검사다. 실제 citation 성능으로 제출하지 않는다.**",
        ]
    return "\n".join(lines) + "\n"


def write_report(
    output_dir,
    config,
    metric_rows,
    intervention_rows,
    scale_rows,
    selection_rows,
    resource_rows,
    gate_rows=None,
    contract=None,
):
    """Use all selected seeds; fail before writing incomplete or nonfinite evidence."""
    output = Path(output_dir)
    if not output.is_dir():
        raise ValueError("report requires an existing new result directory")
    if any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError("normalization report refuses to overwrite existing artifacts")
    gates, evidence = gate_rows or [], contract or {}
    _validate(config, metric_rows, intervention_rows, scale_rows, selection_rows, gates, evidence)
    for row in resource_rows:
        if row.get("measured") is True and row.get("status") == "measured":
            for name in ("seconds_per_epoch", "packed_runs"):
                _finite(row.get(name), name)
                if row[name] < 0 or (name == "packed_runs" and row[name] < 1):
                    raise ValueError("invalid measured resource value")
            _finite(row.get("peak_vram_bytes"), "peak_vram_bytes", nullable=True)
    paired = paired_comparisons(metric_rows)
    changes = [
        r
        for metric in ("accuracy", "ce")
        for r in intervention_changes(intervention_rows, metric_rows, metric)
    ]
    branches = branch_estimates(gates)
    summary = _summary(config, metric_rows, paired, changes, branches)
    for filename, rows in zip(ARTIFACTS[1:], (paired, changes, branches), strict=True):
        write_csv(output / filename, rows)
    with (output / ARTIFACTS[0]).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(summary)
    return {
        "summary": ARTIFACTS[0],
        "figures": 0,
        "paired_rows": len(paired),
        "intervention_change_rows": len(changes),
    }
