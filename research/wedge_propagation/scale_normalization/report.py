"""Compare raw and edge-RMS-normalized gates on paired synthetic inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..generalization.report import (
    _CONDITIONS,
    _DETERMINISTIC,
    _INTERVENTIONS,
    _MESSAGE_METRICS,
    _SCALE_METRICS,
    _SPLITS,
    _TARGETS,
    _aggregate,
    _estimate,
    _identity,
    _save,
    _scale_pairs,
    _validate_messages,
    _validate_scales,
)

_VARIANTS = ("raw", "normalized")
_FIELDS = ("variant", "target", "condition", "split", "scenario", "amplitude")
_FIGURES = ("message_error", "amplitude_error", "scale_equivariance", "weight_and_interventions")
_SERIES = (
    ("raw", "first", "shared first"),
    ("raw", "polynomial", "shared polynomial"),
    ("raw", "fixed", "shared fixed"),
    ("raw", "learned", "raw learned"),
    ("normalized", "learned", "normalized learned"),
    ("raw", "random_pair", "raw random_pair"),
    ("normalized", "random_pair", "normalized random_pair"),
)


def _key(row: dict) -> tuple:
    return _identity(row) + (row["scenario"], row["amplitude"])


def _validate(rows: list[dict], scales: list[dict], interventions: list[dict]) -> None:
    for collection in (rows, scales, interventions):
        if any(row.get("variant") not in _VARIANTS for row in collection):
            raise ValueError("variant must be raw or normalized")
    for variant in _VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            raise ValueError(f"Missing measured {variant} variant")
        _validate_messages(selected, new=True)
        checks = [row for row in scales if row["variant"] == variant]
        _validate_scales(checks)
        _scale_pairs(selected, checks)
    variants = {
        variant: {_key(row): row for row in rows if row["variant"] == variant}
        for variant in _VARIANTS
    }
    if set(variants["raw"]) != set(variants["normalized"]):
        raise ValueError("Raw/normalized graph, seed and treatment coverage differs")
    for key, raw in variants["raw"].items():
        normalized = variants["normalized"][key]
        for name in ("family", "num_nodes", "num_edges", "num_paths", "num_realizations"):
            if raw[name] != normalized[name]:
                raise ValueError("Raw/normalized paired topology or feature-axis metadata differs")
        if raw["condition"] in _DETERMINISTIC:
            for name in (*_MESSAGE_METRICS, "u", "v", "beta"):
                if raw.get(name) != normalized.get(name):
                    raise ValueError("Shared frozen control differs across variants")
    expected = {
        (_key(row), row["variant"], name)
        for row in rows
        if row["condition"] == "learned" and row["amplitude"] == 1
        for name in _INTERVENTIONS
    }
    actual = set()
    for row in interventions:
        _validate_messages([row])
        if row["condition"] != "learned" or row.get("intervention") not in _INTERVENTIONS:
            raise ValueError("Interventions require learned models and the five fixed treatments")
        if row.get("amplitude") != 1 or row.get("scenario") not in ("original", "fresh"):
            raise ValueError("Interventions must use original or fresh amplitude 1")
        key = (_key(row), row["variant"], row["intervention"])
        if key in actual:
            raise ValueError("Duplicate intervention observation")
        actual.add(key)
    if actual != expected:
        raise ValueError("Original/fresh intervention graph and seed coverage differs")


def _pair_differences(rows: list[dict], axis: str) -> list[dict]:
    """Subtract paired measurements before graph and seed aggregation."""
    choices = _VARIANTS if axis == "variant" else ("original", "fresh")
    indexed = {choice: {} for choice in choices}
    for row in rows:
        if axis == "scenario" and row["amplitude"] != 1:
            continue
        fields = ("target", "condition", "seed", "split", "graph_id", "amplitude")
        fields += ("scenario",) if axis == "variant" else ("variant",)
        indexed[row[axis]][tuple(row[name] for name in fields)] = row
    if set(indexed[choices[0]]) != set(indexed[choices[1]]):
        raise ValueError(f"Incomplete paired {axis} comparison")
    changes = []
    for key, before in indexed[choices[0]].items():
        after = indexed[choices[1]][key]
        for name in ("family", "num_nodes", "num_edges", "num_paths", "num_realizations"):
            if before[name] != after[name]:
                raise ValueError(f"Paired {axis} metadata differs")
        change = {**before}
        for name in _MESSAGE_METRICS[:2]:
            left, right = before[name], after[name]
            change[name] = None if left is None or right is None else right - left
        changes.append(change)
    fields = ("target", "condition", "split", "amplitude")
    fields += ("scenario",) if axis == "variant" else ("variant",)
    return _aggregate(changes, fields, _MESSAGE_METRICS[:2])


def _intervention_changes(rows: list[dict], interventions: list[dict]) -> list[dict]:
    baseline = {
        (row["variant"], _key(row)): row
        for row in rows
        if row["condition"] == "learned" and row["amplitude"] == 1
    }
    changes = []
    for row in interventions:
        original = baseline[(row["variant"], _key(row))]
        change = {**row}
        for name in _MESSAGE_METRICS[:2]:
            a, b = row[name], original[name]
            change[name] = None if a is None or b is None else a - b
        changes.append(change)
    return _aggregate(changes, (*_FIELDS, "intervention"), _MESSAGE_METRICS[:2])


def _scale_maxima(scales: list[dict]) -> list[dict]:
    """Worst observed amplitude per graph/seed, rather than independent treatments."""
    grouped = defaultdict(list)
    fields = ("variant", "target", "condition", "seed", "split", "graph_id")
    for row in scales:
        grouped[tuple(row[name] for name in fields)].append(row)
    rows = []
    for key, treatments in grouped.items():
        row = dict(zip(fields, key, strict=True))
        for name in _SCALE_METRICS:
            defined = [value[name] for value in treatments if value[name] is not None]
            row[name] = max(defined) if defined else None
        rows.append(row)
    return _aggregate(rows, ("variant", "target", "condition", "split"), _SCALE_METRICS)


def _label(contract: dict) -> str:
    config = contract.get("source_config", {})
    config = config if isinstance(config, dict) else {}
    seeds = contract.get("model_seeds", config.get("model_seeds"))
    seed_count = len(seeds) if isinstance(seeds, list) else "unreported"
    profile = str(contract.get("profile", contract.get("config", {}).get("profile", "unreported")))
    label = (
        f"{profile.upper()} | normalized epochs={contract.get('new_training_epochs', 'unreported')}"
    )
    label += f" | raw source epochs={config.get('epochs', 'unreported')} | seeds={seed_count}"
    if profile == "debug" or contract.get("debug") is True:
        label += " | Debug verification; not final performance"
    return label


def _finish(axis: Any, values: list[float]) -> None:
    axis.set_ylabel("Graph macro; seed mean ± sample std")
    axis.grid(True, alpha=0.25)
    if values:
        positive = [value for value in values if value > 0]
        if positive and max(positive) / min(positive) > 1000 and min(values) >= 0:
            axis.set_yscale("symlog", linthresh=1e-6)
        else:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    else:
        axis.text(0.5, 0.5, "No defined measurements", ha="center", transform=axis.transAxes)


def _split_points(
    axis: Any,
    rows: list[dict],
    series: list[tuple[str, str, str]],
    metric: str,
    *,
    field: str = "condition",
) -> None:
    values = []
    width = 0.8 / max(len(series), 1)
    for index, (variant, name, label) in enumerate(series):
        labelled = False
        for position, split in enumerate(_SPLITS):
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row[field] == name and row["split"] == split
            ]
            if not selected or selected[0]["metrics"][metric]["mean"] is None:
                continue
            metric_row = selected[0]["metrics"][metric]
            axis.errorbar(
                position + (index - (len(series) - 1) / 2) * width,
                metric_row["mean"],
                yerr=metric_row["std"],
                fmt="o",
                markersize=3,
                capsize=2,
                color=f"C{index % 10}",
                label=label if not labelled else None,
            )
            labelled = True
            values.append(metric_row["mean"])
    axis.set_xticks(range(len(_SPLITS)), _SPLITS, rotation=35, ha="right")
    _finish(axis, values)


def _plot_errors(plt: Any, output: Path, summaries: list[dict], label: str) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(17, 11), constrained_layout=True)
    try:
        for index, target in enumerate(_TARGETS):
            for column, scenario in enumerate(("original", "fresh")):
                selected = [
                    row
                    for row in summaries
                    if row["target"] == target
                    and row["scenario"] == scenario
                    and row["amplitude"] == 1
                ]
                _split_points(axes[index, column], selected, list(_SERIES), "message_relerr")
                axes[index, column].set_title(f"Target {target}: {scenario}, amplitude=1")
        figure.suptitle(
            label + "\nShared controls shown once; raw and normalized gates separate", fontsize=10
        )
        _save(figure, output, "message_error")
    finally:
        plt.close(figure)


def _plot_curves(
    plt: Any, output: Path, summaries: list[dict], label: str, *, scales: bool = False
) -> None:
    figure, axes = plt.subplots(3, 6, figsize=(32, 11), constrained_layout=True)
    try:
        for index, target in enumerate(_TARGETS):
            for column, split in enumerate(_SPLITS):
                axis, values = axes[index, column], []
                selected = [
                    row
                    for row in summaries
                    if row["target"] == target
                    and row["split"] == split
                    and (scales or row["scenario"] == "fresh")
                ]
                series = (
                    [
                        ("raw", "learned", "message_scale_equivariance_relerr", "raw message"),
                        (
                            "normalized",
                            "learned",
                            "message_scale_equivariance_relerr",
                            "normalized message",
                        ),
                        ("raw", "learned", "student_weight_scale_relerr", "raw C"),
                        ("normalized", "learned", "student_weight_scale_relerr", "normalized C"),
                        (
                            "raw",
                            "learned",
                            "teacher_message_scale_equivariance_relerr",
                            "teacher message",
                        ),
                        ("raw", "learned", "teacher_weight_scale_relerr", "teacher C"),
                    ]
                    if scales
                    else [
                        (variant, condition, "message_relerr", name)
                        for variant, condition, name in _SERIES
                    ]
                )
                for color, (variant, condition, metric, name) in enumerate(series):
                    points = sorted(
                        (
                            row
                            for row in selected
                            if row["variant"] == variant and row["condition"] == condition
                        ),
                        key=lambda row: row["amplitude"],
                    )
                    points = [row for row in points if row["metrics"][metric]["mean"] is not None]
                    if not points:
                        continue
                    means = [row["metrics"][metric]["mean"] for row in points]
                    axis.plot(
                        [row["amplitude"] for row in points],
                        means,
                        "o-",
                        markersize=3,
                        color=f"C{color}",
                        label=name,
                    )
                    for row in points:
                        stat = row["metrics"][metric]
                        if stat["std"] is not None:
                            axis.errorbar(
                                row["amplitude"],
                                stat["mean"],
                                yerr=stat["std"],
                                color=f"C{color}",
                                capsize=2,
                            )
                    values.extend(means)
                amplitudes = sorted({row["amplitude"] for row in selected})
                axis.set_xscale("log", base=2)
                axis.set_xticks(amplitudes, [f"{value:g}" for value in amplitudes])
                axis.set_xlabel("Amplitude (paired treatments)")
                axis.set_title(f"Target {target}: {split}")
                _finish(axis, values)
        note = (
            "Scale checks of learned gates; accuracy is a separate measured outcome"
            if scales
            else (
                "Fresh inputs; all controls and gate predictions against amplitude-specific targets"
            )
        )
        figure.suptitle(label + "\n" + note, fontsize=10)
        _save(figure, output, "scale_equivariance" if scales else "amplitude_error")
    finally:
        plt.close(figure)


def _plot_diagnostics(
    plt: Any, output: Path, summaries: list[dict], controls: list[dict], label: str
) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(25, 16), constrained_layout=True)
    original = [
        {**row, "intervention": "original"}
        for row in summaries
        if row["condition"] == "learned" and row["amplitude"] == 1
    ]
    try:
        for scenario_index, scenario in enumerate(("original", "fresh")):
            for column, target in enumerate(_TARGETS):
                selected = [
                    row
                    for row in original + controls
                    if row["scenario"] == scenario and row["target"] == target
                ]
                series = [
                    (variant, intervention, f"{variant} {intervention}")
                    for variant in _VARIANTS
                    for intervention in ("original", *_INTERVENTIONS)
                ]
                _split_points(
                    axes[scenario_index, column],
                    selected,
                    series,
                    "message_relerr",
                    field="intervention",
                )
                axes[scenario_index, column].set_title(
                    f"Target {target}: {scenario} frozen interventions"
                )
            for column, metric in enumerate(_MESSAGE_METRICS[2:]):
                selected = [
                    row
                    for row in summaries
                    if row["scenario"] == scenario
                    and row["target"] == "path"
                    and row["amplitude"] == 1
                ]
                series = [(variant, "learned", variant) for variant in _VARIANTS]
                _split_points(axes[scenario_index + 2, column], selected, series, metric)
                axes[scenario_index + 2, column].set_title(f"Path: {scenario} {metric}")
                if metric == "weight_corr":
                    axes[scenario_index + 2, column].set_ylim(-1.05, 1.05)
        figure.suptitle(
            label + "\nOriginal and fresh interventions separate; C is not uniquely identified",
            fontsize=10,
        )
        _save(figure, output, "weight_and_interventions")
    finally:
        plt.close(figure)


def _lookup(rows: list[dict], **criteria: Any) -> dict | None:
    return next(
        (row for row in rows if all(row.get(key) == value for key, value in criteria.items())), None
    )


def _cell(row: dict | None, metric: str = "message_relerr") -> str:
    return "unreported" if row is None else _estimate(row["metrics"][metric])


def _summary(
    summaries: list[dict],
    variant_changes: list[dict],
    feature_changes: list[dict],
    maxima: list[dict],
    interventions: list[dict],
    contract: dict,
    counts: tuple[int, int, int],
    graph_counts: dict,
) -> str:
    profile = contract.get("profile", contract.get("config", {}).get("profile"))
    lines = [
        "# Experiment 3.1: C 입력 정규화 비교",
        "",
        f"**{_label(contract)}**",
        "",
        "**DEBUG 연결 검증이다. 최종 성능이나 가설의 성공을 판정하지 않는다.**"
        if profile == "debug"
        else "보고서는 실제 관측한 차이를 기록한다. 정규화 모델의 우위를 가정하지 않는다.",
        "",
        "raw는 원본 checkpoint를 고정해 재사용하고, "
        "normalized gate는 원본 학습 데이터에서 새로 학습한다. "
        "변경은 C 생성기에 들어가는 차분을 그래프·scalar 실현별 물리 엣지 RMS로 나누는 것이다. "
        "실제 AX와 teacher는 바꾸지 않는다. 새로운 학습 뒤의 평가·개입에서는 모델을 고정한다.",
        "",
        "이상적인 양의 배율에서는 normalized C 불변성과 메시지 비례성이 구조적으로 성립한다. "
        "목표 메시지의 정확도·새 특징 일반화·C 회수는 관측 결과로 판단한다. "
        "Teacher의 epsilon=1e-8 때문에 teacher도 매 배율에서 다시 계산한다.",
        "",
        "특징 실현 평균 → 동일 비중 그래프 평균 → seed 평균·표본 std 순서로 집계한다. "
        "같은 graph ID·seed·입력끼리 대응하며, 배율을 독립 표본으로 세지 않는다. "
        "first/polynomial/fixed는 한 번 적합한 같은 frozen control이다. "
        "seed=-1과 std=0은 독립 반복이 아니다. "
        "학습 seed 하나의 std는 undefined이며 없는 진단은 0으로 채우지 않는다.",
        "",
        "original/train은 학습에서 본 연결·특징, fresh/train은 같은 연결의 새 특징이다. "
        "ID·각 OOD split은 그대로 분리한다. L/first와 L²/polynomial은 표현 가능한 양성 대조다. "
        "합성 실험의 결과로 일반적인 spectral 함수·GNN 분류 성능·신규성을 단정하지 않는다.",
        "",
        "| Split | 고유 그래프 수 |",
        "| --- | ---: |",
        *[f"| {split} | {count} |" for split, count in graph_counts.items() if count],
        "",
        "## 배율 1: 실제 목표 메시지 상대오차",
        "",
        "각 scalar 실현의 ||예측−목표||₂/(||목표||₂+epsilon) 평균이다. "
        "예측·목표가 모두 0이면 이 규약의 값은 0이다. 절대 RMSE는 CSV에도 저장한다.",
        "",
        "| Scenario | Variant | Target | Split | " + " | ".join(_CONDITIONS) + " |",
        "| --- | --- | --- | --- | " + " | ".join(["---"] * 5) + " |",
    ]
    for scenario in ("original", "fresh"):
        for variant in _VARIANTS:
            for target in _TARGETS:
                for split in _SPLITS:
                    cells = [
                        _lookup(
                            summaries,
                            scenario=scenario,
                            variant=variant,
                            target=target,
                            split=split,
                            amplitude=1,
                            condition=condition,
                        )
                        for condition in _CONDITIONS
                    ]
                    if any(cell is not None for cell in cells):
                        lines.append(
                            f"| {scenario} | {variant} | {target} | {split} | "
                            + " | ".join(_cell(cell) for cell in cells)
                            + " |"
                        )
    lines += [
        "",
        "## 동일 입력 비교: normalized − raw",
        "",
        "그래프·seed별 오차 차이를 먼저 계산했다. 음수는 관측 오차 감소, 양수는 증가다. "
        "동률과 악화도 그대로 기록하며 유의성 검정으로 해석하지 않는다. 동일한 fixed control은 "
        "두 variant의 별도 성공으로 세지 않는다.",
        "",
        "| Scenario | Target | Split | Learned difference | Random-pair difference "
        "| Observed learned ordering |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in ("original", "fresh"):
        for target in _TARGETS:
            for split in _SPLITS:
                cells = [
                    _lookup(
                        variant_changes,
                        scenario=scenario,
                        target=target,
                        split=split,
                        amplitude=1,
                        condition=condition,
                    )
                    for condition in ("learned", "random_pair")
                ]
                if not any(cell is not None for cell in cells):
                    continue
                delta = None if cells[0] is None else cells[0]["metrics"]["message_relerr"]["mean"]
                ordering = (
                    "undefined"
                    if delta is None
                    else ("lower" if delta < 0 else "higher" if delta > 0 else "equal")
                )
                lines.append(
                    f"| {scenario} | {target} | {split} | "
                    + " | ".join(_cell(cell) for cell in cells)
                    + f" | {ordering} |"
                )
    lines += [
        "",
        "## 특징 교체 비교: fresh − original",
        "",
        "동일 topology·seed를 대응한 path target의 배율 1 차이다. "
        "다른 target의 원 관측도 CSV에 남긴다.",
        "",
        "| Variant | Split | Learned difference | Random-pair difference |",
        "| --- | --- | --- | --- |",
    ]
    for variant in _VARIANTS:
        for split in _SPLITS:
            cells = [
                _lookup(
                    feature_changes,
                    variant=variant,
                    target="path",
                    split=split,
                    condition=condition,
                )
                for condition in ("learned", "random_pair")
            ]
            if any(cell is not None for cell in cells):
                lines.append(
                    f"| {variant} | {split} | " + " | ".join(_cell(cell) for cell in cells) + " |"
                )
    lines += [
        "",
        "## 스케일 진단: 관측 배율의 최대 변화",
        "",
        "메시지 값은 ||M(aX)/a−M(X)||₂/(||M(X)||₂+epsilon), C 값은 "
        "||C(aX)−C(X)||₂/(||C(X)||₂+epsilon)이다. 배율 1의 같은 fresh X를 기준으로 한다. "
        "각 graph·seed에서 관측 배율의 최대값을 구한 뒤 graph macro와 seed 통계를 계산했다. "
        "이 표는 learned/path이고, 모든 target·조건·배율은 CSV와 그림에 있다. "
        "스케일 변화가 작다는 사실만으로 정확한 메시지를 만든다고 판단하지 않는다.",
        "",
        "| Variant | Split | Student message | Teacher message | Student C | Teacher C "
        "| Undefined graph/seed observations |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    scale_names = (_SCALE_METRICS[2], _SCALE_METRICS[3], *_SCALE_METRICS[:2])
    for variant in _VARIANTS:
        for split in _SPLITS:
            row = _lookup(maxima, variant=variant, target="path", split=split, condition="learned")
            if row is not None:
                cells = [_estimate(row["metrics"][name]) for name in scale_names]
                undefined = " / ".join(
                    str(row["metrics"][name]["undefined"]) for name in scale_names
                )
                lines.append(f"| {variant} | {split} | " + " | ".join(cells) + f" | {undefined} |")
    lines += [
        "",
        "## C 회수: 원본과 새 특징을 구분한 보조 진단",
        "",
        "C의 평균 1은 전역 scale 자유도를 제한하며 C를 유일하게 식별하는 조건이 아니다. "
        "C 오차·상관은 path target의 보조 진단이다. L/L² target에는 경로 C* 정답이 없다. "
        "상관 undefined는 학생 또는 teacher의 경로 분산이 작을 때도 생기므로 그 자체로 성공·실패를 "
        "판정하지 않는다. Mean error는 Cmean=1 제약의 편차 검사다.",
        "",
        "| Scenario | Variant | Split | Weight relative error | Weight correlation "
        "| Weight mean error | Undefined error/corr/mean |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for scenario in ("original", "fresh"):
        for variant in _VARIANTS:
            for split in _SPLITS:
                row = _lookup(
                    summaries,
                    variant=variant,
                    target="path",
                    condition="learned",
                    scenario=scenario,
                    amplitude=1,
                    split=split,
                )
                if row is not None:
                    names = _MESSAGE_METRICS[2:]
                    cells = [_estimate(row["metrics"][name]) for name in names]
                    undefined = " / ".join(str(row["metrics"][name]["undefined"]) for name in names)
                    lines.append(
                        f"| {scenario} | {variant} | {split} | "
                        + " | ".join(cells)
                        + f" | {undefined} |"
                    )
    lines += [
        "",
        "## 고정 checkpoint 개입: 개입 − 원래 출력 오차",
        "",
        "같은 입력에서 graph·seed별 개입 오차와 원래 오차의 차이를 먼저 계산한다. "
        "원본과 fresh 배율 1을 분리한다. 개입이 오차를 줄인 경우도 그대로 기록한다. "
        "identity와 mean은 평균 1 제약에서 동일하므로 두 독립 효과로 세지 않는다. "
        "대응 무작위화는 연산 support도 바꿀 수 있어 독립적인 인과 효과로 단정하지 않는다. "
        "아래는 path target이며 L/L² 개입도 CSV와 그림에 있다.",
        "",
        "| Scenario | Variant | Split | " + " | ".join(_INTERVENTIONS) + " |",
        "| --- | --- | --- | " + " | ".join(["---"] * 5) + " |",
    ]
    for scenario in ("original", "fresh"):
        for variant in _VARIANTS:
            for split in _SPLITS:
                cells = [
                    _lookup(
                        interventions,
                        variant=variant,
                        target="path",
                        scenario=scenario,
                        amplitude=1,
                        split=split,
                        intervention=name,
                    )
                    for name in _INTERVENTIONS
                ]
                if any(cell is not None for cell in cells):
                    lines.append(
                        f"| {scenario} | {variant} | {split} | "
                        + " | ".join(_cell(cell) for cell in cells)
                        + " |"
                    )
    lines += [
        "",
        "## 실행 계약과 결과 파일",
        "",
        f"원 측정 행: 메시지 {counts[0]}, 스케일 {counts[1]}, 개입 {counts[2]}. "
        "행 수는 독립 표본 수가 아니다. 모델 불변성 검사는 학습 후 고정 평가 구간에 대한 것이다.",
        "",
    ]
    for name in (
        "source_dir",
        "feature_source_dir",
        "graph_count_total",
        "graph_count_used",
        "data_fraction",
        "new_training_epochs",
        "normalized_gate_jobs",
        "raw_optimizer_updates",
        "test_updates",
        "models_unchanged",
        "source_artifacts_unchanged",
        "feature_source_artifacts_unchanged",
        "source_code_unchanged",
        "original_metric_reproduction",
        "physical_graph_batch_selected",
    ):
        if name in contract:
            lines.append(f"- {name}: {json.dumps(contract[name], ensure_ascii=False)}")
    lines += [
        "",
        "- 원 측정: [metrics.csv](metrics.csv), [scale_checks.csv](scale_checks.csv), "
        "[interventions.csv](interventions.csv), [training.csv](training.csv).",
        "- 정확한 입력 hash·모델 구성·학습/선택·자원·불변성 계약: [contract.json](contract.json).",
        *[f"- [{name}]({name}.png) ([PDF]({name}.pdf))." for name in _FIGURES],
        "",
    ]
    return "\n".join(lines)


def write_report(
    output: Path, rows: list[dict], scales: list[dict], interventions: list[dict], contract: dict
) -> None:
    """Write actual comparisons and real figures, exclusively, into a new output directory."""
    output = Path(output)
    if not output.is_dir():
        raise ValueError("Output directory must already exist")
    if not isinstance(contract, dict):
        raise ValueError("Contract must be a dictionary")
    json.dumps(contract, allow_nan=False)
    for name in (
        "models_unchanged",
        "source_artifacts_unchanged",
        "feature_source_artifacts_unchanged",
        "source_code_unchanged",
    ):
        if contract.get(name) is False:
            raise ValueError(f"Evaluation/source guard failed: {name}")
    for name in ("raw_optimizer_updates", "test_updates"):
        if contract.get(name, 0) != 0:
            raise ValueError(f"Forbidden update: {name}")
    _validate(rows, scales, interventions)
    summaries = _aggregate(rows, _FIELDS, _MESSAGE_METRICS)
    scale_summary = _aggregate(
        scales, ("variant", "target", "condition", "split", "amplitude"), _SCALE_METRICS
    )
    controls = _aggregate(interventions, (*_FIELDS, "intervention"), _MESSAGE_METRICS)
    variant_changes, feature_changes = (
        _pair_differences(rows, "variant"),
        _pair_differences(rows, "scenario"),
    )
    changes = _intervention_changes(rows, interventions)
    maxima = _scale_maxima(scales)
    destinations = [output / "SCALE_NORMALIZATION_SUMMARY.md"] + [
        output / f"{name}.{extension}" for name in _FIGURES for extension in ("png", "pdf")
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite scale-normalization artifacts")
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("Matplotlib is required: python -m pip install matplotlib") from error
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8}):
        label = _label(contract)
        _plot_errors(plt, output, summaries, label)
        _plot_curves(plt, output, summaries, label)
        _plot_curves(plt, output, scale_summary, label, scales=True)
        _plot_diagnostics(plt, output, summaries, controls, label)
    graph_counts = {
        split: len({row["graph_id"] for row in rows if row["split"] == split}) for split in _SPLITS
    }
    with (output / "SCALE_NORMALIZATION_SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(
            _summary(
                summaries,
                variant_changes,
                feature_changes,
                maxima,
                changes,
                contract,
                (len(rows), len(scales), len(interventions)),
                graph_counts,
            )
        )
