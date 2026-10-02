"""Report frozen-model synthetic generalization and existing source evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_TARGETS = ("L", "L2", "path")
_CONDITIONS = ("first", "polynomial", "fixed", "learned", "random_pair")
_DETERMINISTIC = {"first", "polynomial", "fixed"}
_SPLITS = ("train", "validation", "id", "size_ood", "family_ood", "family_size_ood")
_INTERVENTIONS = (
    "identity",
    "mean",
    "weight_shuffle",
    "other_graph_pattern",
    "correspondence_randomization",
)
_MESSAGE_METRICS = (
    "message_relerr",
    "message_abs_rmse",
    "weight_relerr",
    "weight_corr",
    "weight_mean_error",
)
_SCALE_METRICS = (
    "student_weight_scale_relerr",
    "teacher_weight_scale_relerr",
    "message_scale_equivariance_relerr",
    "teacher_message_scale_equivariance_relerr",
)
_AUDIT_METRICS = (
    "teacher_span_L_L2_Q_residual",
    "teacher_span_absolute_residual_mean",
    "teacher_operator_variation",
    "teacher_operator_reference_norm",
    "teacher_c_std",
)
_FIGURES = ("feature_generalization", "amplitude_error", "scale_equivariance", "source_evidence")


def _finite(value: Any, name: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number")
    if np.iscomplexobj(value) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")


def _integer(value: Any, name: str, minimum: int) -> None:
    _finite(value, name)
    if value < minimum or not float(value).is_integer():
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _identity(row: dict) -> tuple:
    for name, choices in (("target", _TARGETS), ("condition", _CONDITIONS), ("split", _SPLITS)):
        if row.get(name) not in choices:
            raise ValueError(f"Unrecognized {name}: {row.get(name)!r}")
    for name in ("graph_id", "family"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"{name} must be a nonempty string")
    _integer(row.get("seed"), "seed", -1)
    if row["condition"] in _DETERMINISTIC:
        if row["seed"] != -1:
            raise ValueError("Deterministic fits must have seed=-1")
    elif row["seed"] < 0:
        raise ValueError("Learned models must have nonnegative seeds")
    return tuple(row[name] for name in ("target", "condition", "seed", "split", "graph_id"))


def _validate_messages(rows: list[dict], *, new: bool = False, intervention: bool = False) -> None:
    seen = set()
    for row in rows:
        key = _identity(row)
        for name, minimum in (
            ("num_nodes", 1),
            ("num_edges", 0),
            ("num_paths", 0),
            ("num_realizations", 1),
        ):
            _integer(row.get(name), name, minimum)
        if "message_relerr" not in row:
            raise ValueError("Missing message_relerr; undefined values require None")
        _finite(row["message_relerr"], "message_relerr", nullable=True)
        _finite(row.get("message_abs_rmse"), "message_abs_rmse")
        for name in _MESSAGE_METRICS[2:]:
            _finite(row.get(name), name, nullable=True)
        if new:
            scenario = row.get("scenario")
            if scenario not in ("original", "fresh"):
                raise ValueError("scenario must be original or fresh")
            _finite(row.get("amplitude"), "amplitude")
            if row["amplitude"] <= 0 or (scenario == "original" and row["amplitude"] != 1):
                raise ValueError("Positive amplitudes required; original amplitude must be 1")
            expected = "source_features" if scenario == "original" else "unseen_features"
            if row.get("feature_status") != expected or row.get("source_split") != row["split"]:
                raise ValueError("Feature status/source split does not match its scenario")
            key += (scenario, row["amplitude"])
        if intervention:
            if row["condition"] != "learned" or row.get("intervention") not in _INTERVENTIONS:
                raise ValueError("Invalid source learned-model intervention")
            key += (row["intervention"],)
        if key in seen:
            raise ValueError(f"Duplicate measured observation: {key}")
        seen.add(key)


def _validate_scales(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        key = _identity(row)
        _finite(row.get("amplitude"), "amplitude")
        if row["amplitude"] <= 0:
            raise ValueError("Scale amplitude must be positive")
        for name in _SCALE_METRICS:
            if name not in row:
                raise ValueError(f"Missing scale measurement {name}")
            _finite(row[name], name, nullable=True)
        key += (row["amplitude"],)
        if key in seen:
            raise ValueError(f"Duplicate scale measurement: {key}")
        seen.add(key)


def _scale_pairs(messages: list[dict], scales: list[dict]) -> None:
    expected = {
        _identity(row) + (row["amplitude"],): row for row in messages if row["scenario"] == "fresh"
    }
    actual = {_identity(row) + (row["amplitude"],): row for row in scales}
    if set(actual) != set(expected):
        raise ValueError("Fresh message/scale treatment coverage differs")
    for key, row in actual.items():
        if row["family"] != expected[key]["family"] or row.get("scenario", "fresh") != "fresh":
            raise ValueError("Fresh scale treatment metadata differs")


def _validate_audits(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if row.get("split") not in _SPLITS:
            raise ValueError("Teacher audit split is unrecognized")
        for name in ("graph_id", "family"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"Teacher audit {name} must be a nonempty string")
        for name in _AUDIT_METRICS:
            if name not in row:
                raise ValueError(f"Missing teacher audit measurement {name}")
            _finite(row[name], name, nullable=True)
        for name in (
            "num_features",
            "teacher_span_defined_features",
            "teacher_span_undefined_features",
            "teacher_operator_variation_defined_pairs",
        ):
            _integer(row.get(name), name, 0)
        if (
            row["teacher_span_defined_features"] + row["teacher_span_undefined_features"]
            != row["num_features"]
        ):
            raise ValueError("Teacher audit defined/undefined feature counts disagree")
        if row["graph_id"] in seen:
            raise ValueError("Duplicate teacher audit graph")
        seen.add(row["graph_id"])


def _aggregate(rows: list[dict], fields: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    """Equal graph weighting inside each seed, then equal seed weighting."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[tuple(row[name] for name in fields)][int(row["seed"])].append(row)
    output = []
    for key, seeds in groups.items():
        graph_sets = [{row["graph_id"] for row in seed_rows} for seed_rows in seeds.values()]
        if any(ids != graph_sets[0] for ids in graph_sets[1:]):
            raise ValueError(f"Different graph coverage across seeds: {key}")
        if any(len(seed_rows) != len(graph_sets[0]) for seed_rows in seeds.values()):
            raise ValueError(f"Repeated graph in one aggregation cell: {key}")
        deterministic = dict(zip(fields, key, strict=True))["condition"] in _DETERMINISTIC
        estimates = {}
        for name in metrics:
            means, undefined = [], 0
            for seed_rows in seeds.values():
                values = [float(row[name]) for row in seed_rows if row.get(name) is not None]
                undefined += len(seed_rows) - len(values)
                if values:
                    means.append(float(np.mean(values)))
            count = len(means)
            estimates[name] = {
                "mean": float(np.mean(means)) if means else None,
                "std": float(np.std(means, ddof=1))
                if count > 1
                else (0.0 if count == 1 and deterministic else None),
                "defined_seeds": count,
                "undefined": undefined,
            }
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "metrics": estimates,
                "graphs": len(graph_sets[0]),
                "seeds": len(seeds),
                "deterministic": deterministic,
            }
        )
    return output


def _paired_features(rows: list[dict]) -> list[dict]:
    """Subtract original error from fresh error on each identical graph/seed."""
    original, fresh = {}, {}
    for row in rows:
        if row["amplitude"] == 1:
            destination = original if row["scenario"] == "original" else fresh
            destination[_identity(row)] = row
    if set(original) != set(fresh):
        raise ValueError("Original/fresh amplitude=1 graph/seed pairs are incomplete")
    output = []
    for key, before in original.items():
        after = fresh[key]
        for name in ("family", "num_nodes", "num_edges", "num_paths"):
            if before[name] != after[name]:
                raise ValueError("Paired original/fresh graph topology metadata differs")
        output.append(
            {
                **before,
                "message_relerr": None
                if before["message_relerr"] is None or after["message_relerr"] is None
                else after["message_relerr"] - before["message_relerr"],
                "message_abs_rmse": after["message_abs_rmse"] - before["message_abs_rmse"],
            }
        )
    return _aggregate(output, ("target", "condition", "split"), _MESSAGE_METRICS[:2])


def _value(value: Any) -> str:
    return "undefined" if value is None else f"{value:.6g}"


def _estimate(metric: dict) -> str:
    mean, std = metric["mean"], metric["std"]
    if mean is None:
        return "undefined"
    return f"{mean:.6g} ± {std:.3g}" if std is not None else f"{mean:.6g} (std undefined)"


def _label(contract: dict) -> str:
    source_config = contract.get("source_config", {})
    source_config = source_config if isinstance(source_config, dict) else {}
    seeds = contract.get("model_seeds", source_config.get("model_seeds"))
    count = len(seeds) if isinstance(seeds, list) else "unreported"
    profile = str(contract.get("profile", "unreported"))
    source_epochs = source_config.get("epochs", "unreported")
    value = (
        f"{profile.upper()} | new epochs=0 (frozen) | source epochs={source_epochs} | seeds={count}"
    )
    return (
        value + " | Debug verification; not final performance"
        if (profile == "debug" or contract.get("debug") is True)
        else value
    )


def _save(figure: Any, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        with (directory / f"{name}.{extension}").open("xb") as stream:
            figure.savefig(stream, format=extension, dpi=180, bbox_inches="tight")


def _finish(axis: Any, values: list[float], ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if values:
        positive = [value for value in values if value > 0]
        if positive and max(positive) / min(positive) > 1000:
            axis.set_yscale("symlog", linthresh=1e-6)
        else:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    else:
        axis.text(0.5, 0.5, "No defined measurements", ha="center", transform=axis.transAxes)


def _split_points(
    axis: Any, rows: list[dict], names: tuple[str, ...], series: str, metric: str
) -> None:
    values = []
    for index, name in enumerate(names):
        labelled = False
        for split_index, split in enumerate(_SPLITS):
            selected = [row for row in rows if row[series] == name and row["split"] == split]
            if not selected or selected[0]["metrics"][metric]["mean"] is None:
                continue
            stat = selected[0]["metrics"][metric]
            axis.errorbar(
                split_index + (index - (len(names) - 1) / 2) * 0.12,
                stat["mean"],
                yerr=stat["std"],
                fmt="o",
                markersize=3,
                capsize=2,
                color=f"C{index}",
                label=name if not labelled else None,
            )
            labelled = True
            values.append(stat["mean"])
    axis.set_xticks(range(len(_SPLITS)), _SPLITS, rotation=35, ha="right")
    _finish(axis, values, "Graph macro; seed mean ± sample std")


def _plot_features(plt: Any, directory: Path, rows: list[dict], label: str) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    try:
        for index, target in enumerate(_TARGETS):
            for column, scenario in enumerate(("original", "fresh")):
                selected = [
                    row
                    for row in rows
                    if row["target"] == target
                    and row["scenario"] == scenario
                    and row["amplitude"] == 1
                ]
                _split_points(
                    axes[index, column], selected, _CONDITIONS, "condition", "message_relerr"
                )
                axes[index, column].set_title(f"Target {target}: {scenario} features, amplitude=1")
        figure.suptitle(label, fontsize=10)
        _save(figure, directory, "feature_generalization")
    finally:
        plt.close(figure)


def _curves(axis: Any, rows: list[dict], series: list[tuple[str, str, str]]) -> None:
    values = []
    for index, (condition, metric, name) in enumerate(series):
        selected = sorted(
            (row for row in rows if row["condition"] == condition), key=lambda row: row["amplitude"]
        )
        selected = [row for row in selected if row["metrics"][metric]["mean"] is not None]
        if not selected:
            continue
        means = [row["metrics"][metric]["mean"] for row in selected]
        stds = [row["metrics"][metric]["std"] for row in selected]
        axis.plot(
            [row["amplitude"] for row in selected],
            means,
            "o-",
            markersize=3,
            color=f"C{index}",
            label=name,
        )
        for row, mean, std in zip(selected, means, stds, strict=True):
            if std is not None:
                axis.errorbar(row["amplitude"], mean, yerr=std, color=f"C{index}", capsize=2)
        values.extend(means)
    axis.set_xscale("log", base=2)
    amplitudes = sorted({row["amplitude"] for row in rows})
    axis.set_xticks(amplitudes, [f"{value:g}" for value in amplitudes])
    axis.set_xlabel("Positive amplitude (paired treatments)")
    _finish(axis, values, "Graph macro; seed mean ± sample std")


def _plot_amplitudes(
    plt: Any, directory: Path, rows: list[dict], label: str, *, scales: bool = False
) -> None:
    figure, axes = plt.subplots(3, 6, figsize=(30, 11), constrained_layout=True)
    name = "scale_equivariance" if scales else "amplitude_error"
    try:
        for index, target in enumerate(_TARGETS):
            for column, split in enumerate(_SPLITS):
                selected = [
                    row
                    for row in rows
                    if row["target"] == target
                    and row["split"] == split
                    and (scales or row["scenario"] == "fresh")
                ]
                series = (
                    [
                        ("learned", "message_scale_equivariance_relerr", "student message"),
                        ("learned", "teacher_message_scale_equivariance_relerr", "teacher message"),
                        ("learned", "student_weight_scale_relerr", "student C"),
                        ("learned", "teacher_weight_scale_relerr", "teacher C"),
                    ]
                    if scales
                    else [(condition, "message_relerr", condition) for condition in _CONDITIONS]
                )
                _curves(axes[index, column], selected, series)
                axes[index, column].set_title(f"Target {target}: {split}")
        note = (
            "Scale checks compare aX with a times the same X prediction"
            if scales
            else ("Prediction error against amplitude-specific teacher; all conditions frozen")
        )
        figure.suptitle(label + "\n" + note, fontsize=10)
        _save(figure, directory, name)
    finally:
        plt.close(figure)


def _plot_source(
    plt: Any, directory: Path, source: list[dict], interventions: list[dict], label: str
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(19, 8), constrained_layout=True)
    try:
        for index, target in enumerate(_TARGETS):
            rows = [
                {**row, "intervention": "original"}
                for row in source
                if row["target"] == target and row["condition"] == "learned"
            ]
            rows += [row for row in interventions if row["target"] == target]
            _split_points(
                axes[0, index],
                rows,
                ("original", *_INTERVENTIONS),
                "intervention",
                "message_relerr",
            )
            axes[0, index].set_title(f"SOURCE ONLY: Target {target} frozen interventions")
        for index, metric in enumerate(_MESSAGE_METRICS[2:]):
            selected = [row for row in source if row["target"] == "path"]
            _split_points(axes[1, index], selected, ("learned", "random_pair"), "condition", metric)
            axes[1, index].set_title(f"SOURCE ONLY: path {metric}")
            if metric == "weight_corr":
                axes[1, index].set_ylim(-1.05, 1.05)
        figure.suptitle(
            label + "\nExisting source CSV evidence; no rerun; C is not uniquely identified",
            fontsize=10,
        )
        _save(figure, directory, "source_evidence")
    finally:
        plt.close(figure)


def _lookup(rows: list[dict], **criteria: Any) -> dict | None:
    return next(
        (row for row in rows if all(row.get(key) == value for key, value in criteria.items())), None
    )


def _cell(row: dict | None, metric: str = "message_relerr") -> str:
    return "unreported" if row is None else _estimate(row["metrics"][metric])


def _summary(
    rows: list[dict],
    pairs: list[dict],
    scales: list[dict],
    source: list[dict],
    interventions: list[dict],
    audits: list[dict],
    contract: dict,
    counts: tuple[int, ...],
    graph_counts: dict[str, int],
) -> str:
    debug = contract.get("profile") == "debug" or contract.get("debug") is True
    lines = [
        "# 합성 일반화 평가: 고정한 경로 가중치 규칙",
        "",
        f"**{_label(contract)}**",
        "",
        "**Debug 출력·평가 연결 검증이다. 최종 성능이나 연구 가설의 성공을 판정하지 않는다.**"
        if debug
        else "학습된 checkpoint와 고정 대조군을 사용한 전체 범위 평가다.",
        "",
        "새 실행에서는 optimizer update와 추가 학습이 없다. 기존 그래프를 그대로 유지하며 "
        "독립적으로 생성한 새 특징과 양의 배율을 적용한다. 실제 데이터 분류 실험이 아니다.",
        "",
        "각 그래프의 특징 실현 평균을 같은 비중으로 모은 뒤 seed 평균과 표본 std를 구한다. "
        "결정론적 first/polynomial/fixed의 seed=-1, std=0은 독립 반복 학습을 뜻하지 않는다. "
        "정의되지 않은 진단은 제외하며 0으로 대체하지 않는다. "
        "seed가 하나면 표본 std는 undefined다.",
        "",
        "original/train은 학습에서 본 그래프·특징이다. "
        "fresh/train은 같은 학습 그래프의 새 특징이다. "
        "나머지 split은 원 source의 구분을 유지한다. "
        "이들 결과를 하나의 일반화 평균으로 합치지 않는다. "
        "같은 fresh X에 배율을 적용한 측정은 대응된 처리이며 독립 표본이 아니다.",
        "",
        "| Split | 고유 그래프 수 | Original 특징 | Fresh 특징 |",
        "| --- | ---: | --- | --- |",
        *[
            f"| {split} | {graph_counts[split]} | source_features | unseen_features |"
            for split in _SPLITS
            if any(row["split"] == split for row in rows)
        ],
        "",
        "## 특징 일반화: 배율 1의 실제 상대오차",
        "",
        "상대오차는 기존 평가의 feature별 ||예측−목표||₂/(||목표||₂+epsilon) 평균이다. "
        "예측·목표가 모두 0이면 이 규약의 값은 0이다. 절대 RMSE도 원 CSV에 기록한다.",
        "",
        "| Scenario | Target | Split | " + " | ".join(_CONDITIONS) + " |",
        "| --- | --- | --- | " + " | ".join(["---"] * 5) + " |",
    ]
    for scenario in ("original", "fresh"):
        for target in _TARGETS:
            for split in _SPLITS:
                cells = [
                    _lookup(
                        rows,
                        scenario=scenario,
                        amplitude=1,
                        target=target,
                        split=split,
                        condition=condition,
                    )
                    for condition in _CONDITIONS
                ]
                if any(cell is not None for cell in cells):
                    lines.append(
                        f"| {scenario} | {target} | {split} | "
                        + " | ".join(_cell(cell) for cell in cells)
                        + " |"
                    )
    lines += [
        "",
        "## 대응 비교: fresh − original",
        "",
        "같은 topology·seed를 대응시켜 그래프별 오차 차이를 먼저 계산했다. "
        "음수는 관측 오차 감소, 양수는 증가다. "
        "동률·개선·악화를 그대로 기록하며 유의성 검정은 아니다.",
        "",
        "| Target | Split | " + " | ".join(_CONDITIONS) + " |",
        "| --- | --- | " + " | ".join(["---"] * 5) + " |",
    ]
    for target in _TARGETS:
        for split in _SPLITS:
            cells = [
                _lookup(pairs, target=target, split=split, condition=name) for name in _CONDITIONS
            ]
            if any(cell is not None for cell in cells):
                lines.append(
                    f"| {target} | {split} | " + " | ".join(_cell(cell) for cell in cells) + " |"
                )
    lines += [
        "",
        "## 양성 대조와 실제 순위",
        "",
        "L target의 first와 L² target의 polynomial은 목표를 표현할 수 있는 대조군이다. "
        "이들의 실제 오차를 위 표에서 확인한다. 경로 모델의 보편적 우위를 뜻하지 않는다.",
        "",
        "아래는 fresh/path/배율 1에서 learned − 각 대조군의 관측 평균 상대오차다. "
        "개선이나 악화의 방향을 미리 가정하지 않는다.",
        "",
        "| Split | Baseline | Learned − baseline | Observed ordering |",
        "| --- | --- | --- | --- |",
    ]
    for split in _SPLITS:
        learned = _lookup(
            rows, scenario="fresh", amplitude=1, target="path", split=split, condition="learned"
        )
        if learned is None:
            continue
        for baseline in ("first", "polynomial", "fixed", "random_pair"):
            reference = _lookup(
                rows, scenario="fresh", amplitude=1, target="path", split=split, condition=baseline
            )
            if reference is None:
                continue
            a, b = (
                learned["metrics"]["message_relerr"]["mean"],
                reference["metrics"]["message_relerr"]["mean"],
            )
            delta = None if a is None or b is None else a - b
            ordering = (
                "undefined"
                if delta is None
                else ("lower" if delta < 0 else "higher" if delta > 0 else "equal")
            )
            lines.append(f"| {split} | {baseline} | {_value(delta)} | {ordering} |")
    lines += [
        "",
        "## 스케일 변화와 등변성",
        "",
        "예측 오차와 스케일 등변성은 다른 진단이다. 메시지 등변성은 M(aX)와 aM(X)를 "
        "비교하고, 가중치 진단은 C(aX)와 C(X)를 비교한다. teacher에도 같은 검사를 수행한다. "
        "메시지·가중치가 배율에 따라 바뀌어도 예측 오차가 반드시 커지는 것은 아니다.",
        "",
        "메시지 값은 ||M(aX)/a−M(X)||₂/(||M(X)||₂+epsilon), "
        "가중치 값은 ||C(aX)−C(X)||₂/(||C(X)||₂+epsilon)을 특징별로 계산한 평균이다. "
        "경로가 없는 그래프의 가중치 진단은 undefined다.",
        "",
        "표는 learned/path의 split별 결과다. "
        "모든 target·조건·배율의 원 측정과 곡선은 CSV·그림에 있다.",
        "",
        "| Split | Amplitude | Student message | Teacher message | Student C | Teacher C "
        "| Undefined graph/seed observations |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for row in scales:
        if row["target"] == "path" and row["condition"] == "learned":
            names = (_SCALE_METRICS[2], _SCALE_METRICS[3], *_SCALE_METRICS[:2])
            undefined = " / ".join(str(row["metrics"][name]["undefined"]) for name in names)
            lines.append(
                f"| {row['split']} | {row['amplitude']:g} | "
                + " | ".join(_estimate(row["metrics"][name]) for name in names)
                + f" | {undefined} |"
            )
    lines += [
        "",
        "## 기존 source의 C와 개입 증거",
        "",
        "이 절은 source CSV를 읽은 요약이다. 새 특징에서 개입을 다시 실행한 결과가 아니다. "
        "Cmean=1은 전체 scale 자유도를 제한할 뿐 C를 유일하게 복원했다는 보장은 없다. "
        "weight error·correlation은 보조 진단이며 L/L²에는 식별 가능한 경로 teacher가 없다. "
        "weight mean error는 평균 1 제약의 편차다.",
        "",
        "| Source path split | Weight relative error | Weight correlation | Weight mean error |",
        "| --- | --- | --- | --- |",
    ]
    for row in source:
        if row["target"] == "path" and row["condition"] == "learned":
            lines.append(
                f"| {row['split']} | "
                + " | ".join(_estimate(row["metrics"][name]) for name in _MESSAGE_METRICS[2:])
                + " |"
            )
    lines += [
        "",
        "identity와 mean은 평균 1 제약에서 동일한 개입이다. 두 독립 효과로 세지 않는다. "
        "shuffle·다른 그래프 패턴·대응 무작위화는 오차를 줄일 수도 있다. "
        "대응 무작위화는 support도 바꿀 수 있어 독립적인 인과 효과로 해석하지 않는다.",
        "",
        "| Source target | Split | " + " | ".join(_INTERVENTIONS) + " |",
        "| --- | --- | " + " | ".join(["---"] * 5) + " |",
    ]
    for target in _TARGETS:
        for split in _SPLITS:
            cells = [
                _lookup(interventions, target=target, split=split, intervention=name)
                for name in _INTERVENTIONS
            ]
            if any(cell is not None for cell in cells):
                lines.append(
                    f"| {target} | {split} | " + " | ".join(_cell(cell) for cell in cells) + " |"
                )
    lines += [
        "",
        "## 기존 teacher 연산 진단",
        "",
        "특징 실현마다 teacher 행렬 T=Aᵀdiag(c*)A를 행렬 span{L,L²,Q}에 사후 적합한다. "
        "상대 Frobenius 잔차 ||T−T_fit||F/||T||F를 0이 아닌 teacher 행렬에 대해 평균한 값과 "
        "특징에 따른 teacher 행렬의 변화를 요약한다. 목표 노드 메시지의 적합 오차가 아니다. "
        "양의 잔차는 이 세 행렬의 span 밖이라는 뜻이며 임의의 spectral 함수까지 배제하지 않는다. "
        "공유 polynomial 모델의 학습 결과나 새 특징의 일반화 증거가 아니다. "
        "아래는 그래프별 값의 min / median / max다.",
        "",
        "| Source split | Graphs | Span residual | Operator variation | C std "
        "| Undefined span / variation |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ]
    for split in _SPLITS:
        selected = [row for row in audits if row["split"] == split]
        if not selected:
            continue
        cells, missing = [], []
        for name in (_AUDIT_METRICS[0], _AUDIT_METRICS[2], _AUDIT_METRICS[4]):
            values = [row[name] for row in selected if row[name] is not None]
            cells.append(
                " / ".join(_value(value) for value in (min(values), np.median(values), max(values)))
                if values
                else "undefined"
            )
            missing.append(len(selected) - len(values))
        lines.append(
            f"| {split} | {len(selected)} | "
            + " | ".join(cells)
            + f" | {missing[0]} / {missing[1]} |"
        )
    lines += [
        "",
        "## 무학습·불변성 계약과 산출물",
        "",
        f"측정 행 수: 새 메시지 {counts[0]}, 스케일 {counts[1]}, 기존 메시지 {counts[2]}, "
        f"기존 개입 {counts[3]}, 기존 teacher 진단 {counts[4]}. 행 수는 독립 표본 수가 아니다.",
        "",
        "정확한 source 경로·데이터 hash·checkpoint 전후 검사·원래 그래프 보존 및 자원 측정은 "
        "[contract.json](contract.json)에 기록한다. 누락된 검증을 통과한 것으로 표현하지 않는다.",
        "",
    ]
    for key in (
        "source_dir",
        "source_graph_count",
        "feature_realizations",
        "source_profile",
        "graph_count_total",
        "graph_count_used",
        "data_fraction",
        "optimizer_updates",
        "new_training_epochs",
        "source_data_hash",
        "fresh_data_hash",
        "source_unchanged",
        "source_unchanged_during_run",
        "model_unchanged",
        "checkpoint_unchanged",
        "source_files_unchanged",
        "models_unchanged",
        "source_artifacts_unchanged",
        "source_code_unchanged",
        "original_metric_reproduction",
    ):
        if key in contract:
            lines.append(f"- {key}: {json.dumps(contract[key], ensure_ascii=False)}")
    fresh_hashes = contract.get("fresh_data_hashes", {})
    if isinstance(fresh_hashes, dict) and fresh_hashes:
        lines += ["", "| Fresh 배율 | Dataset SHA256 |", "| --- | --- |"]
        lines += [f"| {amplitude} | {digest} |" for amplitude, digest in fresh_hashes.items()]
    lines += [
        "",
        "- 새 측정: [metrics.csv](metrics.csv), [scale_checks.csv](scale_checks.csv).",
        "- 기존 source 재분석: [source_metrics.csv](source_metrics.csv), "
        "[source_interventions.csv](source_interventions.csv), "
        "[source_teacher_operator_audit.csv](source_teacher_operator_audit.csv).",
        *[f"- [{name}]({name}.png) ([PDF]({name}.pdf))." for name in _FIGURES],
        "",
    ]
    return "\n".join(lines)


def write_report(
    output_dir: str | Path,
    metric_rows: list[dict],
    scale_rows: list[dict],
    source_metric_rows: list[dict],
    source_intervention_rows: list[dict],
    teacher_audit_rows: list[dict],
    contract: dict,
) -> None:
    """Exclusively write real charts and summaries into an existing new output directory."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ValueError("Output directory must already exist")
    if not metric_rows:
        raise ValueError("New metric rows must contain actual measurements")
    _validate_messages(metric_rows, new=True)
    _validate_messages(source_metric_rows)
    _validate_messages(source_intervention_rows, intervention=True)
    _validate_scales(scale_rows)
    _scale_pairs(metric_rows, scale_rows)
    _validate_audits(teacher_audit_rows)
    if not isinstance(contract, dict):
        raise ValueError("Contract must be a JSON-serializable dictionary")
    json.dumps(contract, allow_nan=False)
    for key in (
        "source_unchanged",
        "source_unchanged_during_run",
        "model_unchanged",
        "checkpoint_unchanged",
        "source_files_unchanged",
        "models_unchanged",
        "source_artifacts_unchanged",
        "source_code_unchanged",
    ):
        if contract.get(key) is False:
            raise ValueError(f"Frozen/source guard failed: {key}")
    if contract.get("optimizer_updates", 0) != 0:
        raise ValueError("Frozen evaluation must not contain optimizer updates")
    if contract.get("new_training_epochs", 0) != 0:
        raise ValueError("Frozen evaluation must not contain new training epochs")
    if "source_model_hashes" in contract and "model_hashes_after" in contract:
        if contract["source_model_hashes"] != contract["model_hashes_after"]:
            raise ValueError("Model hashes changed during frozen evaluation")
    reproduction = contract.get("original_metric_reproduction", {})
    if isinstance(reproduction, dict) and reproduction.get("verified") is False:
        raise ValueError("Original metric reproduction guard failed")
    if (
        isinstance(reproduction, dict)
        and {"maximum_absolute_error", "tolerance"} <= reproduction.keys()
    ):
        for name in ("maximum_absolute_error", "tolerance"):
            _finite(reproduction[name], name)
        if reproduction["maximum_absolute_error"] > reproduction["tolerance"]:
            raise ValueError("Original metric reproduction exceeded its tolerance")
    rows = _aggregate(
        metric_rows, ("target", "condition", "split", "scenario", "amplitude"), _MESSAGE_METRICS
    )
    pairs = _paired_features(metric_rows)
    scales = _aggregate(scale_rows, ("target", "condition", "split", "amplitude"), _SCALE_METRICS)
    source = _aggregate(source_metric_rows, ("target", "condition", "split"), _MESSAGE_METRICS)
    interventions = _aggregate(
        source_intervention_rows, ("target", "condition", "split", "intervention"), _MESSAGE_METRICS
    )
    destinations = [directory / "SYNTHETIC_GENERALIZATION_SUMMARY.md"] + [
        directory / f"{name}.{extension}" for name in _FIGURES for extension in ("png", "pdf")
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite existing generalization artifacts")
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("Matplotlib is required: python -m pip install matplotlib") from error
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8}):
        label = _label(contract)
        _plot_features(plt, directory, rows, label)
        _plot_amplitudes(plt, directory, rows, label)
        _plot_amplitudes(plt, directory, scales, label, scales=True)
        _plot_source(plt, directory, source, interventions, label)
    counts = tuple(
        map(
            len,
            (
                metric_rows,
                scale_rows,
                source_metric_rows,
                source_intervention_rows,
                teacher_audit_rows,
            ),
        )
    )
    graph_counts = {
        split: len({row["graph_id"] for row in metric_rows if row["split"] == split})
        for split in _SPLITS
    }
    with (directory / "SYNTHETIC_GENERALIZATION_SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(
            _summary(
                rows,
                pairs,
                scales,
                source,
                interventions,
                teacher_audit_rows,
                contract,
                counts,
                graph_counts,
            )
        )
