"""Summarize saved Experiment 4.1 observations without a new model evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

LEARNED = ("learned_wedge_raw", "learned_wedge_rms")
TARGETS = ("layer_0", "layer_1", "both")
TREATMENTS = (
    "baseline",
    "true_c_unit_denominator",
    "identity_hold_reference",
    "identity_unit_denominator",
    "identity_norm_matched",
    "branch_off",
    "shuffle_hold_reference",
    "shuffle_norm_matched",
)
STRENGTH_METRICS = (
    "alpha",
    "beta",
    "kappa_reference",
    "beta_over_kappa",
    "raw_branch_to_input",
    "unscaled_branch_to_input",
    "alpha_l_to_input",
    "beta_t_to_input",
    "beta_t_to_alpha_l",
    "first_second_cosine",
    "cosine_correction_input",
    "kappa_ratio_mean",
    "kappa_ratio_p50",
    "kappa_ratio_p90",
    "kappa_ratio_p99",
    "kappa_ratio_max",
    "kappa_near_max_fraction",
)
FIXED_METRICS = (
    "message_norm_ratio",
    "branch_reference_cosine",
    "message_relative_change",
    "beta_t_to_input",
)
SUMMARY = "STRENGTH_ANALYSIS_SUMMARY.md"
_ESTIMATE_KEYS = ("dataset", "condition", "layer", "treatment", "target", "metric")
_CHANGE_KEYS = ("dataset", "condition", "split", "treatment", "target", "metric")


def _finite(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; undefined observations use None")


def _index(rows: list[dict], fields: tuple[str, ...], name: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate {name} group: {key}")
        result[key] = row
    return result


def _validate_estimate(row: dict, seed_count: int, *, nullable: bool) -> None:
    count = row["count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= seed_count:
        raise ValueError("estimate count must be a valid independent seed count")
    if nullable:
        undefined = row["undefined_seeds"]
        observations = row["undefined_observations"]
        if (
            isinstance(undefined, bool)
            or not isinstance(undefined, int)
            or undefined != seed_count - count
            or isinstance(observations, bool)
            or not isinstance(observations, int)
            or observations < undefined
        ):
            raise ValueError("undefined seed/observation counts do not match estimates")
    elif count != seed_count:
        raise ValueError("paired metric estimate must contain every configured seed")
    for field in ("mean", "std", "lower", "upper"):
        _finite(row[field], field, nullable=True)
    if (row["mean"] is None) != (count == 0):
        raise ValueError("undefined mean requires zero defined seeds")
    if count < 2:
        if any(row[field] is not None for field in ("std", "lower", "upper")):
            raise ValueError("sample std and interval are undefined below two seeds")
    else:
        if any(row[field] is None for field in ("std", "lower", "upper")):
            raise ValueError("sample std and interval required for two or more seeds")
        if row["std"] < 0 or not row["lower"] <= row["mean"] <= row["upper"]:
            raise ValueError("invalid sample std or interval")


def _validate(config: dict, analysis: dict, provenance: dict) -> tuple[dict, dict]:
    datasets, seeds = config["data"]["datasets"], config["final_seeds"]
    if not datasets or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("datasets and distinct configured seeds are required")
    if config["profile"] not in ("full", "debug"):
        raise ValueError("report requires explicit full or debug profile")
    if any(condition not in config["conditions"] for condition in LEARNED):
        raise ValueError("both learned C conditions are required")
    if set(config["treatments"]) != set(TREATMENTS) or set(config["targets"]) != set(TARGETS):
        raise ValueError("complete Experiment 4.1 treatments and targets are required")
    if (
        provenance.get("optimizer_updates") != 0
        or provenance.get("csv_only") is not True
        or provenance.get("source_files_preserved") is not True
    ):
        raise ValueError("CSV-only analysis and unchanged source with zero updates required")
    source_epochs = provenance["source_config"]["training"]["epochs_per_run"]
    if isinstance(source_epochs, bool) or not isinstance(source_epochs, int) or source_epochs <= 0:
        raise ValueError("original source training epoch contract required")
    baseline = _index(
        analysis["baseline_strength"],
        ("dataset", "condition", "seed", "layer"),
        "baseline",
    )
    expected_baseline = {
        (d, c, s, layer) for d in datasets for c in LEARNED for s in seeds for layer in (0, 1)
    }
    if set(baseline) != expected_baseline:
        raise ValueError("baseline factor coverage differs from configured learned seeds/layers")
    for row in baseline.values():
        for metric in STRENGTH_METRICS:
            _finite(row[metric], metric, nullable=True)
    estimates = _index(analysis["strength_estimates"], _ESTIMATE_KEYS, "strength estimate")
    expected_strength = {
        (d, c, layer, "baseline", "none", metric)
        for d in datasets
        for c in LEARNED
        for layer in (0, 1)
        for metric in STRENGTH_METRICS
    }
    if set(estimates) != expected_strength:
        raise ValueError("strength estimate coverage differs from baseline factors")
    fixed = _index(analysis["fixed_estimates"], _ESTIMATE_KEYS, "fixed-Z estimate")
    expected_fixed = {
        (d, c, layer, treatment, f"layer_{layer}", metric)
        for d in datasets
        for c in LEARNED
        for layer in (0, 1)
        for treatment in TREATMENTS
        for metric in FIXED_METRICS
    }
    if set(fixed) != expected_fixed:
        raise ValueError("fixed-Z estimate coverage differs from all configured treatments")
    for row in (*estimates.values(), *fixed.values()):
        _validate_estimate(row, len(seeds), nullable=True)
    changes = _index(analysis["layer_changes"], _CHANGE_KEYS, "paired change")
    expected_changes = {
        (d, c, split, treatment, target, metric)
        for d in datasets
        for c in LEARNED
        for split in ("train", "validation", "test")
        for treatment in TREATMENTS[1:]
        for target in TARGETS
        for metric in ("accuracy", "ce")
    }
    if set(changes) != expected_changes:
        raise ValueError("paired change coverage differs from all configured splits/targets")
    for row in changes.values():
        _validate_estimate(row, len(seeds), nullable=False)
    return estimates, changes


def _show(value: float | None, *, scale: float = 1.0) -> str:
    return "undefined" if value is None else f"{value * scale:.4g}"


def _cell(row: dict, *, scale: float = 1.0) -> str:
    value = f"{_show(row['mean'], scale=scale)} ± {_show(row['std'], scale=scale)}"
    if row.get("undefined_seeds", 0) or row.get("undefined_observations", 0):
        value += f" (n={row['count']}; u={row['undefined_observations']})"
    return value


def _condition(condition: str) -> str:
    return "raw" if condition == "learned_wedge_raw" else "RMS"


def _summary(config: dict, analysis: dict, provenance: dict, estimates: dict, changes: dict) -> str:
    datasets, seeds = config["data"]["datasets"], config["final_seeds"]
    profile = config["profile"].upper()
    source_epochs = provenance["source_config"]["training"]["epochs_per_run"]
    lines = [
        "# Experiment 4.1 후속 분석 — 이차 분기의 크기 분해",
        "",
        f"**{profile} · 기존 seed {seeds} · 원래 학습 {source_epochs} epoch/run · 새 update 0회.**",
        "완료된 4.1 CSV 전체를 읽은 분석이다. 새 forward·학습·GPU 평가·개입 선택은 없다.",
        "",
        "`R = S_Q Aᵀ C A S_Q Z / 3`, `M = R/κ`, `U = Z − αL̄Z − βM`.",
        "각 seed에서 `βM/Z = (β/κ) × (R/Z)`를 계산했다. 아래 값은 seed 평균 ± 표본 std이며, "
        "평균끼리 곱한 값은 실제 평균 βM/Z와 다를 수 있다.",
        "Norm은 현재 층 전체 노드·channel의 Frobenius norm이다. "
        "0인 기준 norm은 undefined로 남겼다(u=정의되지 않은 관측 수).",
        "",
        "## 원래 forward의 크기",
        "",
        "| 데이터 | C | 층 | α | β | κ | β/κ | R/Z | βM/Z | αL̄Z/Z | βM/αL̄Z | cos(일차, 이차) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    factor_metrics = (
        "alpha",
        "beta",
        "kappa_reference",
        "beta_over_kappa",
        "raw_branch_to_input",
        "beta_t_to_input",
        "alpha_l_to_input",
        "beta_t_to_alpha_l",
        "first_second_cosine",
    )
    for dataset in datasets:
        for condition in LEARNED:
            for layer in (0, 1):
                cells = [
                    _cell(estimates[(dataset, condition, layer, "baseline", "none", metric)])
                    for metric in factor_metrics
                ]
                lines.append(
                    f"| {dataset} | {_condition(condition)} | {layer} | " + " | ".join(cells) + " |"
                )
    lines += [
        "",
        "## κ를 결정하는 노드 비율의 분포",
        "",
        "활성 노드의 `diag(AᵀCA)/diag(AᵀA)`를 요약했다. "
        "κ는 최대값이고, 근접 비율은 최대값의 1% 이내인 활성 노드 비율이다.",
        "",
        "| 데이터 | C | 층 | 평균 | p50 | p90 | p99 | 최대(κ) | 최대 근접 비율 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    tail_metrics = (
        "kappa_ratio_mean",
        "kappa_ratio_p50",
        "kappa_ratio_p90",
        "kappa_ratio_p99",
        "kappa_ratio_max",
        "kappa_near_max_fraction",
    )
    for dataset in datasets:
        for condition in LEARNED:
            for layer in (0, 1):
                cells = [
                    _cell(estimates[(dataset, condition, layer, "baseline", "none", metric)])
                    for metric in tail_metrics
                ]
                lines.append(
                    f"| {dataset} | {_condition(condition)} | {layer} | " + " | ".join(cells) + " |"
                )
    lines += [
        "",
        "## 어느 층을 바꿨을 때 달라졌는가",
        "",
        "각 칸은 `Δaccuracy pp / ΔCE`의 seed 평균이며 **개입 − 원래**다. "
        "Accuracy는 양수가, CE는 음수가 개선이다. 두 지표를 함께 읽는다.",
    ]
    for treatment, title in (
        ("true_c_unit_denominator", "학습한 C 유지 · κ 분모를 1로 변경"),
        ("branch_off", "이차 분기 제거"),
    ):
        lines += [
            "",
            f"### {title}",
            "",
            "| 데이터 | C | layer_0 | layer_1 | both |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for dataset in datasets:
            for condition in LEARNED:
                cells = []
                for target in TARGETS:
                    base = (dataset, condition, "test", treatment, target)
                    accuracy = changes[(*base, "accuracy")]
                    ce = changes[(*base, "ce")]
                    cells.append(f"{_show(accuracy['mean'], scale=100)} / {_show(ce['mean'])}")
                lines.append(f"| {dataset} | {_condition(condition)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 해석 범위와 보존된 상세값",
        "",
        "β/κ와 R/Z는 실제 크기를 나누어 보는 항이다. "
        "이 고정 checkpoint 진단과 전역 norm만으로 "
        "학습 실패의 원인이나 개별 노드 효과를 확정할 수 없다.",
        "분모 1 개입은 해당 층 현재 Z에서 메시지를 κ배 한다. "
        "첫 층 개입은 이후 층 입력도 바꾸므로 both 효과를 두 단일 층 효과의 합으로 보지 않는다.",
        "Test 관측 뒤의 분석이다. 최고 개입을 새 모델 성능으로 선택하지 않으며 "
        "새 split·새 그래프의 일반화 검증을 대신하지 않는다.",
        "[baseline_strength.csv](baseline_strength.csv)에 seed별 κ 최대 노드 ID와 인자를 남겼다. "
        "노드 ID는 평균하지 않았다. [strength_estimates.csv](strength_estimates.csv)에 "
        "β를 곱하기 전 M/Z와 correction/input cosine도 있다.",
        "[layer_changes.csv](layer_changes.csv)는 모든 개입·세 위치·train/validation/test의 "
        "paired std·95% t 구간을 포함한다. Shuffle은 manifest를 seed 안에서 먼저 평균한다.",
        "[fixed_estimates.csv](fixed_estimates.csv)는 원래 Z에서 모든 C 배치의 "
        "크기 비율·방향·변화량을 보존한다. 구간은 고정 split의 초기화 변동만 나타낸다.",
        "[coverage.json](coverage.json), [contract.json](contract.json), "
        "[completion.json](completion.json)에 범위·source 보존·완료 기록이 있다.",
    ]
    if config["profile"] == "debug":
        lines += [
            "",
            "**DEBUG fixture 분석 검사다. 실제 citation 본학습 성능으로 제출하지 않는다.**",
        ]
    return "\n".join(lines) + "\n"


def write_report(output: Path, config: dict, analysis: dict, provenance: dict) -> dict[str, str]:
    """Validate complete saved observations and write one new standalone summary."""
    estimates, changes = _validate(config, analysis, provenance)
    summary = _summary(config, analysis, provenance, estimates, changes)
    output = Path(output)
    with (output / SUMMARY).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(summary)
    return {"summary": SUMMARY}
