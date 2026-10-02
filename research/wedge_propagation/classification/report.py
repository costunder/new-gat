"""Report actual full-graph classification results without teacher metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

_LEARNED = ("learned_wedge_raw", "learned_wedge_rms")
_HOLD = "hold_each_layers_preintervention_kappa"
_FIGURES = (
    "classification_performance",
    "frozen_interventions",
    "scale_response",
    "branches_and_resources",
)
_COMPARISONS = [
    (learned, other)
    for learned in _LEARNED
    for other in ("fixed_wedge", "polynomial_2", "fixed_wedge_node_mlp")
]
_COMPARISONS.append(("learned_wedge_rms", "learned_wedge_raw"))


def _finite(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real measurement")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def estimate(values: list[float]) -> dict:
    """Seed estimate with sample std and fixed-split paired t uncertainty."""
    if not values:
        return {"mean": None, "std": None, "lower": None, "upper": None, "count": 0}
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite seed estimate")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else None
    half = (
        float(student_t.ppf(0.975, array.size - 1) * std / math.sqrt(array.size))
        if std is not None
        else None
    )
    return {
        "mean": mean,
        "std": std,
        "lower": None if half is None else mean - half,
        "upper": None if half is None else mean + half,
        "count": int(array.size),
    }


def paired_comparisons(rows: list[dict]) -> list[dict]:
    lookup = {(r["dataset"], r["condition"], r["seed"], r["split"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    result = []
    for dataset in datasets:
        for first, second in _COMPARISONS:
            for metric in ("accuracy", "ce"):
                differences = []
                for seed in seeds:
                    left = lookup.get((dataset, first, seed, "test"))
                    right = lookup.get((dataset, second, seed, "test"))
                    if left is not None and right is not None:
                        differences.append(left[metric] - right[metric])
                if differences:
                    result.append(
                        {
                            "dataset": dataset,
                            "first": first,
                            "second": second,
                            "metric": metric,
                            **estimate(differences),
                        }
                    )
    return result


def intervention_changes(rows: list[dict], originals: list[dict], metric: str) -> list[dict]:
    """Average paired manifests inside each seed before seed uncertainty."""
    baseline = {(r["dataset"], r["condition"], r["seed"], r["split"]): r[metric] for r in originals}
    treatment: dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = tuple(
            row[n] for n in ("dataset", "condition", "seed", "split", "intervention", "kappa_mode")
        )
        original = baseline[key[:4]]
        treatment[key].append(row[metric] - original)
    groups: dict[tuple, list] = defaultdict(list)
    for key, values in treatment.items():
        groups[key[:2] + key[3:]].append(float(np.mean(values)))
    return [
        {
            "dataset": key[0],
            "condition": key[1],
            "split": key[2],
            "intervention": key[3],
            "kappa_mode": key[4],
            "metric": metric,
            **estimate(values),
        }
        for key, values in sorted(groups.items())
    ]


def _validate(
    config: dict,
    metrics: list[dict],
    interventions: list[dict],
    scales: list[dict],
    gates: list[dict],
    contract: dict,
) -> None:
    datasets = config["data"]["datasets"]
    conditions = config["conditions"]
    seeds = config["training"]["final_seeds"]
    amplitudes = config["evaluation"]["scale_amplitudes"]
    expected = {
        (d, c, s, split)
        for d in datasets
        for c in conditions
        for s in seeds
        for split in ("train", "validation", "test")
    }
    actual = set()
    for row in metrics + interventions:
        _finite(row.get("ce"), "ce")
        _finite(row.get("accuracy"), "accuracy")
        if row["ce"] < 0 or not 0 <= row["accuracy"] <= 1:
            raise ValueError("CE must be nonnegative and accuracy must be in [0,1]")
        key = tuple(row[name] for name in ("dataset", "condition", "seed", "split"))
        if key not in expected:
            raise ValueError("unexpected dataset/condition/seed/split")
        if row in metrics:
            if key in actual:
                raise ValueError("duplicate original classification metric")
            actual.add(key)
    if actual != expected:
        raise ValueError("incomplete original classification metric coverage")
    count = config["evaluation"]["shuffle_and_random_manifests_per_dataset"]
    expected_interventions = set()
    for d, c, s, split in expected:
        if c not in _LEARNED:
            continue
        for mode in ("recompute", _HOLD):
            expected_interventions.add((d, c, s, split, "c_identity", mode, -1))
            for index in range(count):
                expected_interventions.add((d, c, s, split, "c_position_shuffle", mode, index))
        expected_interventions.add((d, c, s, split, "second_branch_remove", "not_applicable", -1))
        for index in range(count):
            expected_interventions.add(
                (d, c, s, split, "random_physical_edge_pair_correspondence", _HOLD, index)
            )
    observed = [
        tuple(
            row[name]
            for name in (
                "dataset",
                "condition",
                "seed",
                "split",
                "intervention",
                "kappa_mode",
                "manifest_index",
            )
        )
        for row in interventions
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected_interventions:
        raise ValueError("incomplete or duplicate frozen intervention coverage")
    expected_scales = {
        (d, c, s, "end_to_end", split, -1, float(a))
        for d, c, s, split in expected
        for a in amplitudes
    }
    expected_scales |= {
        (d, c, s, "fixed_layer_Z", "not_applicable", layer, float(a))
        for d in datasets
        for c in conditions
        if c in _LEARNED
        for s in seeds
        for layer in (0, 1)
        for a in amplitudes
    }
    observed_scales = []
    for row in scales:
        observed_scales.append(
            tuple(
                row[name]
                for name in ("dataset", "condition", "seed", "scope", "split", "layer", "amplitude")
            )
        )
        if row["scope"] == "end_to_end":
            _finite(row.get("ce"), "scaled ce")
            _finite(row.get("accuracy"), "scaled accuracy")
            _finite(row.get("logit_scale_equivariance_relerr"), "logit relative", nullable=True)
        elif row["scope"] == "fixed_layer_Z":
            for name in (
                "c_scale_relerr",
                "kappa_scale_relerr",
                "message_scale_equivariance_relerr",
            ):
                _finite(row.get(name), name, nullable=True)
                if row.get(name) is not None and row[name] < 0:
                    raise ValueError("relative diagnostics cannot be negative")
    if len(observed_scales) != len(set(observed_scales)) or set(observed_scales) != expected_scales:
        raise ValueError("incomplete or duplicate scale coverage")
    for row in gates:
        for name in (
            "c_mean",
            "c_std",
            "c_min",
            "c_max",
            "kappa",
            "kappa_reference",
            "sigma",
            "alpha",
            "beta",
            "branch_norm",
            "l_message_norm",
            "t_message_norm",
            "operator_norm_estimate",
        ):
            _finite(row.get(name), name, nullable=True)
    for name in ("models_unchanged", "source_code_unchanged", "data_unchanged"):
        if contract.get(name) is False:
            raise ValueError(f"failed {name} guard")
    if contract.get("evaluation_optimizer_updates", 0) != 0:
        raise ValueError("frozen evaluation contains optimizer updates")


def _display(value: float | None, multiplier: float = 1.0) -> str:
    return "undefined" if value is None else f"{value * multiplier:.6g}"


def _mean_std(values: list[float], multiplier: float = 1.0) -> str:
    stats = estimate(values)
    return f"{_display(stats['mean'], multiplier)} ± {_display(stats['std'], multiplier)}"


def _label(config: dict) -> str:
    profile = config.get("profile", "unreported").upper()
    label = (
        f"{profile} | layers={config['backbone']['layers']} | "
        f"epochs={config['training']['epochs_per_run']} | "
        f"final seeds={len(config['training']['final_seeds'])}"
    )
    return label + " | DEBUG verification; not final performance" if profile == "DEBUG" else label


def _summary(
    config: dict,
    metrics: list[dict],
    interventions: list[dict],
    scales: list[dict],
    selections: list[dict],
    resources: list[dict],
    gates: list[dict],
    contract: dict,
) -> str:
    intro = (
        "별도 DEBUG 입력으로 분류·평가 연결을 검사한 결과다. 실제 citation 성능이 아니다. "
        if config.get("profile") == "debug"
        else "전체 그래프의 public fixed split에서 분류 CE만으로 학습했다. "
    )
    lines = [
        "# Experiment 4: 실제 분류와 경로 가중치",
        "",
        f"**{_label(config)}**",
        "",
        intro + "Synthetic teacher·C 정답·synthetic checkpoint를 사용하지 않는다.",
        "",
        "동일 seed의 paired 차이를 먼저 계산하며 seed를 독립 그래프나 split으로 세지 않는다. "
        "95% t 구간은 고정 split의 초기화 변동에 대한 기술 통계다. "
        "여러 비교를 보정한 확증 검정이나 새 그래프 일반화의 증거로 해석하지 않는다.",
        "",
        f"관측 행: 원래 분류 {len(metrics)}, 개입 {len(interventions)}, 배율 {len(scales)}, "
        f"층 진단 {len(gates)}.",
        "",
        "## Test 결과",
        "",
        "Accuracy는 %이며 std는 학습 seed의 표본 표준편차다. CE는 L2를 제외한 실제 평균 CE다.",
        "",
        "| Dataset | Condition | Accuracy (%) | CE | Selected lr |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    selection = {
        (r["dataset"], r["condition"]): r.get("selected_lr", r.get("lr")) for r in selections
    }
    for dataset in config["data"]["datasets"]:
        for condition in config["conditions"]:
            group = [
                r
                for r in metrics
                if r["dataset"] == dataset and r["condition"] == condition and r["split"] == "test"
            ]
            lines.append(
                f"| {dataset} | {condition} | "
                f"{_mean_std([r['accuracy'] for r in group], 100)} | "
                f"{_mean_std([r['ce'] for r in group])} | "
                f"{_display(selection.get((dataset, condition)))} |"
            )
    lines += [
        "",
        "## 동일 seed의 test 차이",
        "",
        "Accuracy 차이는 percentage point이며 양수면 첫 조건이 높다. "
        "CE 차이는 음수면 첫 조건이 낮다. 실제 악화·동률도 그대로 표시한다.",
        "",
        "| Dataset | First − second | Metric | Mean difference | 95% t interval | Seeds |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in paired_comparisons(metrics):
        factor = 100 if row["metric"] == "accuracy" else 1
        lines.append(
            f"| {row['dataset']} | {row['first']} − {row['second']} | {row['metric']} | "
            f"{_display(row['mean'], factor)} | [{_display(row['lower'], factor)}, "
            f"{_display(row['upper'], factor)}] | {row['count']} |"
        )
    lines += [
        "",
        "## 고정 checkpoint 개입",
        "",
        "개입 − 원래 test accuracy다. 음수면 개입 후 정확도가 낮다. "
        "같은 seed 안에서 manifest 평균을 낸 뒤 seed 통계를 계산한다. "
        "Manifest를 추가 학습 seed로 세지 않는다. Mean C=1은 identity와 같아 중복하지 않는다.",
        "",
        "| Dataset | Condition | Intervention | κ mode | Accuracy difference (pp) |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in intervention_changes(interventions, metrics, "accuracy"):
        if row["split"] == "test":
            lines.append(
                f"| {row['dataset']} | {row['condition']} | {row['intervention']} | "
                f"{row['kappa_mode']} | {_display(row['mean'], 100)} ± "
                f"{_display(row['std'], 100)} |"
            )
    lines += [
        "",
        "κ recompute는 개입 C에서 강도를 다시 계산한다. Hold는 각 층의 변경된 현재 Z에서 "
        "개입 직전 true C의 κ를 유지한다. 두 진단을 섞지 않는다. "
        "Random correspondence는 edge 사용 빈도·노드 support·거리·norm도 바꿀 수 있다. "
        "악화를 경로 연속성만의 인과 효과로 해석하지 않는다.",
        "",
        "## 배율과 층 진단",
        "",
        "고정한 층 Z의 C·κ·branch 변화와 aX의 end-to-end 분류 반응을 구분한다. "
        "입력은 전처리 완료 X를 배율로 키우며 row 정규화를 다시 하지 않는다. Dropout은 껐다. "
        "양의 logits 배율만으로 accuracy가 같아도 softmax 온도 때문에 CE가 달라질 수 있다.",
        "",
        "| Dataset | Condition | Layer | Mean per-seed max C change | "
        "Mean per-seed max branch equivariance error | Undefined C / branch |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in config["data"]["datasets"]:
        for condition in _LEARNED:
            if condition not in config["conditions"]:
                continue
            for layer in (0, 1):
                group = [
                    r
                    for r in scales
                    if r["dataset"] == dataset
                    and r["condition"] == condition
                    and r["scope"] == "fixed_layer_Z"
                    and r["layer"] == layer
                ]
                summaries, missing = [], []
                for name in ("c_scale_relerr", "message_scale_equivariance_relerr"):
                    seed_values: dict[int, list] = defaultdict(list)
                    for row in group:
                        seed_values[row["seed"]].append(row.get(name))
                    values = [
                        max(v for v in observations if v is not None)
                        for observations in seed_values.values()
                        if any(v is not None for v in observations)
                    ]
                    summaries.append(_display(estimate(values)["mean"]))
                    missing.append(sum(all(v is None for v in obs) for obs in seed_values.values()))
                lines.append(
                    f"| {dataset} | {condition} | {layer} | {summaries[0]} | "
                    f"{summaries[1]} | {missing[0]} / {missing[1]} |"
                )
    lines += [
        "",
        "기준 norm이 0인 상대값은 undefined로 표시한다.",
        "",
        "C 분산만으로 유익성을 판단하지 않는다. κ 고정·random 진단의 operator norm은 "
        "32 step sparse power iteration 추정이며 엄밀한 값이나 상한 증명이 아니다.",
        "",
        "### 원래 모델의 층별 가중치·강도",
        "",
        "C std는 각 층 전체 path의 population std를 계산한 뒤 학습 seed 평균을 낸 값이다. "
        "아래 κ·분기 norm도 seed 평균이며 없는 관측은 undefined다.",
        "",
        "| Dataset | Condition | Layer | C mean | C std | κ | "
        "Branch norm | Operator norm estimate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in config["data"]["datasets"]:
        for condition in _LEARNED:
            for layer in (0, 1):
                group = [
                    r
                    for r in gates
                    if r["dataset"] == dataset
                    and r["condition"] == condition
                    and r["layer"] == layer
                    and r.get("intervention", "original") == "original"
                ]
                if group:
                    values = [
                        _display(
                            estimate([r[name] for r in group if r.get(name) is not None])["mean"]
                        )
                        for name in (
                            "c_mean",
                            "c_std",
                            "kappa",
                            "branch_norm",
                            "operator_norm_estimate",
                        )
                    ]
                    lines.append(
                        f"| {dataset} | {condition} | {layer} | " + " | ".join(values) + " |"
                    )
    lines += ["", "## 계산 비용과 해석 범위", ""]
    measured = [r for r in resources if r.get("measured") is True and r.get("status") == "measured"]
    lines += [
        f"자원 calibration의 실제 측정 행 {len(measured)}개를 사용했다. "
        "OOM·memory-safety 거부 행은 처리량 0으로 바꾸거나 그림에 포함하지 않는다.",
        "",
        "Full graph 하나가 각 run의 데이터 단위다. Packed run 수와 physical graph batch를 "
        "혼동하지 않는다. Epoch 시간은 train+validation을 포함한다.",
        "",
        "Fixed·polynomial·용량 대조보다 좋은지와 C 개입 반응을 함께 해석한다. "
        "개입에 민감해도 baseline보다 나쁜 모델을 유익하다고 단정하지 않는다. "
        "RMS 배율 안정성은 분류 개선을 보장하지 않는다. "
        "직접 sparse 메시지 경로와 graph-wide RMS·κ의 전역 특징 의존을 구분한다.",
        "",
        "## 실행 증거",
        "",
        f"설계 학습 예산: tuning {config['training']['tuning_runs']} + "
        f"final {config['training']['final_runs']} = {config['training']['total_runs']} run. "
        "실제 완료 coverage는 contract/completion에 기록된 실행 증거로 판단한다.",
        "",
        "[metrics.csv](metrics.csv), [interventions.csv](interventions.csv), "
        "[scale.csv](scale.csv), [gate_diagnostics.csv](gate_diagnostics.csv), "
        "[learning_rate_selection.json](learning_rate_selection.json), "
        "[final_validation_selection.csv](final_validation_selection.csv), "
        "[resources.csv](resources.csv), [contract.json](contract.json). "
        "Epoch 기록은 jobs/<job>/pack-*/epoch_history.csv에 있다.",
        "",
    ]
    for name in _FIGURES:
        lines.append(f"[{name}]({name}.png) ([PDF]({name}.pdf)).")
    if config.get("profile") == "debug":
        lines += [
            "",
            "**DEBUG 입력·축소 epoch로 연결을 검사한 결과다. "
            "Cora/CiteSeer/PubMed 본학습 성능이나 최종 성능으로 제출하지 않는다.**",
        ]
    return "\n".join(lines) + "\n"


def _curve(
    axis: Any, x: list[float], groups: list[list[float]], label: str, *, multiplier: float = 1.0
) -> None:
    values = [estimate(group) for group in groups]
    good = [index for index, value in enumerate(values) if value["mean"] is not None]
    if good:
        (line,) = axis.plot(
            [x[i] for i in good],
            [values[i]["mean"] * multiplier for i in good],
            marker="o",
            label=label,
        )
        known = [i for i in good if values[i]["std"] is not None]
        if known:
            axis.errorbar(
                [x[i] for i in known],
                [values[i]["mean"] * multiplier for i in known],
                yerr=[values[i]["std"] * multiplier for i in known],
                capsize=2,
                color=line.get_color(),
                fmt="none",
            )


def _save(figure: Any, output: Path, name: str, label: str) -> None:
    notes = {
        "classification_performance": "Error bars: sample std across initialization seeds",
        "frozen_interventions": "Manifests averaged within seed; bars: sample std across seeds",
        "scale_response": "Fixed-layer Z and scaled X are separate probes; bars: seed sample std",
        "branches_and_resources": "Actual packed-run observations; failed cases excluded",
    }
    figure.suptitle(label + "\n" + notes[name], fontsize=10)
    for axis in figure.axes:
        axis.grid(alpha=0.2)
    handles, labels = [], []
    for axis in figure.axes:
        h, text = axis.get_legend_handles_labels()
        for handle, item in zip(h, text, strict=True):
            if item not in labels:
                handles.append(handle)
                labels.append(item)
    if handles:
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 0.98), fontsize=8)
    for suffix in ("png", "pdf"):
        with (output / f"{name}.{suffix}").open("xb") as stream:
            figure.savefig(stream, format=suffix, dpi=180, bbox_inches="tight")


def write_report(
    output_dir: str | Path,
    config: dict,
    metric_rows: list[dict],
    intervention_rows: list[dict],
    scale_rows: list[dict],
    selection_rows: list[dict],
    resource_rows: list[dict],
    gate_rows: list[dict] | None = None,
    contract: dict | None = None,
) -> None:
    """Write four scientific PNG/PDF figures and a readable Korean summary."""
    output = Path(output_dir)
    if not output.is_dir():
        raise ValueError("report output must be an existing fresh result directory")
    names = ["CLASSIFICATION_SUMMARY.md"] + [f"{n}.{s}" for n in _FIGURES for s in ("png", "pdf")]
    if any((output / name).exists() for name in names):
        raise FileExistsError("classification report refuses to overwrite existing artifacts")
    gates, evidence = gate_rows or [], contract or {}
    _validate(config, metric_rows, intervention_rows, scale_rows, gates, evidence)
    for row in resource_rows:
        if row.get("measured") is True and row.get("status") == "measured":
            for name in ("seconds_per_epoch", "peak_vram_bytes", "packed_runs"):
                _finite(row.get(name), name)
                if row[name] < 0 or (name == "packed_runs" and row[name] < 1):
                    raise ValueError("invalid measured resource value")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "classification report requires matplotlib; pip install matplotlib"
        ) from exc
    datasets, conditions = config["data"]["datasets"], config["conditions"]
    label = _label(config)
    figure, axes = plt.subplots(
        2, len(datasets), figsize=(5 * len(datasets), 7), squeeze=False, layout="constrained"
    )
    for column, dataset in enumerate(datasets):
        for row_index, metric in enumerate(("accuracy", "ce")):
            axis = axes[row_index, column]
            for index, condition in enumerate(conditions):
                values = [
                    r[metric]
                    for r in metric_rows
                    if r["dataset"] == dataset
                    and r["condition"] == condition
                    and r["split"] == "test"
                ]
                _curve(
                    axis,
                    [index],
                    [values],
                    condition,
                    multiplier=100 if metric == "accuracy" else 1,
                )
            axis.set(
                title=f"{dataset}: test {metric}",
                xticks=range(len(conditions)),
                xticklabels=[str(i + 1) for i in range(len(conditions))],
                xlabel="Condition (legend order)",
                ylabel="Accuracy (%)" if metric == "accuracy" else "CE",
            )
    _save(figure, output, _FIGURES[0], label)
    plt.close(figure)

    differences = intervention_changes(intervention_rows, metric_rows, "accuracy")
    treatment_order = [
        ("c_identity", "recompute"),
        ("c_identity", _HOLD),
        ("c_position_shuffle", "recompute"),
        ("c_position_shuffle", _HOLD),
        ("second_branch_remove", "not_applicable"),
        ("random_physical_edge_pair_correspondence", _HOLD),
    ]
    figure, axes = plt.subplots(
        2, len(datasets), figsize=(5 * len(datasets), 7), squeeze=False, layout="constrained"
    )
    for column, dataset in enumerate(datasets):
        for row_index, condition in enumerate(_LEARNED):
            axis = axes[row_index, column]
            for index, treatment in enumerate(treatment_order):
                found = [
                    r
                    for r in differences
                    if r["dataset"] == dataset
                    and r["condition"] == condition
                    and r["split"] == "test"
                    and (r["intervention"], r["kappa_mode"]) == treatment
                ]
                if found:
                    point = found[0]
                    axis.errorbar(
                        index,
                        100 * point["mean"],
                        yerr=None if point["std"] is None else 100 * point["std"],
                        marker="o",
                        capsize=3,
                        label=treatment[0]
                        + (
                            " / hold"
                            if treatment[1] == _HOLD
                            else " / recompute"
                            if treatment[1] == "recompute"
                            else ""
                        ),
                    )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set(
                title=f"{dataset}: {condition}",
                ylabel="Intervention - original accuracy (pp)",
                xticks=range(6),
                xticklabels=range(1, 7),
                xlabel="Treatment (legend order)",
            )
    _save(figure, output, _FIGURES[1], label)
    plt.close(figure)

    figure, axes = plt.subplots(
        3, len(datasets), figsize=(5 * len(datasets), 10), squeeze=False, layout="constrained"
    )
    amplitudes = config["evaluation"]["scale_amplitudes"]
    for column, dataset in enumerate(datasets):
        for condition in _LEARNED:
            for layer in (0, 1):
                for row_index, metric in enumerate(
                    ("c_scale_relerr", "message_scale_equivariance_relerr")
                ):
                    groups = [
                        [
                            r[metric]
                            for r in scale_rows
                            if r["dataset"] == dataset
                            and r["condition"] == condition
                            and r["scope"] == "fixed_layer_Z"
                            and r["layer"] == layer
                            and r["amplitude"] == amplitude
                            and r[metric] is not None
                        ]
                        for amplitude in amplitudes
                    ]
                    _curve(
                        axes[row_index, column], amplitudes, groups, f"{condition} / layer {layer}"
                    )
            groups = [
                [
                    r["accuracy"]
                    for r in scale_rows
                    if r["dataset"] == dataset
                    and r["condition"] == condition
                    and r["scope"] == "end_to_end"
                    and r["split"] == "test"
                    and r["amplitude"] == amplitude
                ]
                for amplitude in amplitudes
            ]
            _curve(axes[2, column], amplitudes, groups, condition, multiplier=100)
        for row_index, title in enumerate(
            (
                "Fixed Z: C relative change",
                "Fixed Z: branch equivariance error",
                "Scaled X: test accuracy (%)",
            )
        ):
            axes[row_index, column].set(title=f"{dataset}: {title}", xlabel="Positive amplitude")
            axes[row_index, column].set_xscale("log", base=2)
            if row_index < 2:
                axes[row_index, column].set_yscale("symlog", linthresh=1e-7)
                axes[row_index, column].set_ylim(bottom=0)
                if not axes[row_index, column].lines:
                    axes[row_index, column].text(
                        0.5,
                        0.5,
                        "Undefined: zero reference norm",
                        transform=axes[row_index, column].transAxes,
                        ha="center",
                    )
    _save(figure, output, _FIGURES[2], label)
    plt.close(figure)

    figure, axes = plt.subplots(
        3, len(datasets), figsize=(5 * len(datasets), 10), squeeze=False, layout="constrained"
    )
    measured = [
        r for r in resource_rows if r.get("measured") is True and r.get("status") == "measured"
    ]
    for column, dataset in enumerate(datasets):
        for layer in (0, 1):
            for name in ("alpha", "beta"):
                groups = [
                    [
                        r[name]
                        for r in gates
                        if r["dataset"] == dataset
                        and r["condition"] == condition
                        and r["layer"] == layer
                        and r.get("intervention", "original") == "original"
                        and r.get(name) is not None
                    ]
                    for condition in conditions
                ]
                _curve(
                    axes[0, column], list(range(len(conditions))), groups, f"{name} / layer {layer}"
                )
        for row_index, metric in ((1, "seconds_per_epoch"), (2, "peak_vram_bytes")):
            # Candidate batch sizes are measurements, not independent seeds.
            for index, condition in enumerate(conditions):
                observations = [
                    r
                    for r in measured
                    if r["dataset"] == dataset
                    and r["condition"] == condition
                    and r.get(metric) is not None
                ]
                for offset, observation in enumerate(observations):
                    value = observation[metric]
                    _finite(value, metric)
                    axes[row_index, column].scatter(
                        observation["packed_runs"],
                        value / (2**30) if row_index == 2 else value,
                        color=plt.get_cmap("tab10")(index),
                        marker="o" if observation.get("phase") == "final" else "^",
                        label=condition if offset == 0 else None,
                    )
            if not any(r["dataset"] == dataset and r.get(metric) is not None for r in measured):
                axes[row_index, column].text(
                    0.5,
                    0.5,
                    "Not measured",
                    transform=axes[row_index, column].transAxes,
                    ha="center",
                )
        axes[0, column].set(title=f"{dataset}: learned branch scalars", ylabel="Coefficient")
        axes[1, column].set(
            title=f"{dataset}: measured train+val epoch", ylabel="Seconds / packed epoch"
        )
        axes[2, column].set(title=f"{dataset}: measured peak VRAM", ylabel="GiB")
        axes[0, column].set(
            xticks=range(len(conditions)),
            xticklabels=range(1, len(conditions) + 1),
            xlabel="Condition (performance legend order)",
        )
        for axis in axes[1:, column]:
            axis.set(xlabel="Packed independent runs (triangle=tuning, circle=final)")
            counts = sorted({r["packed_runs"] for r in measured if r["dataset"] == dataset})
            if counts:
                axis.set_xticks(counts)
                axis.set_xlim(min(counts) - 0.25, max(counts) + 0.25)
    _save(figure, output, _FIGURES[3], label)
    plt.close(figure)
    with (output / "CLASSIFICATION_SUMMARY.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            _summary(
                config,
                metric_rows,
                intervention_rows,
                scale_rows,
                selection_rows,
                resource_rows,
                gates,
                evidence,
            )
        )


__all__ = ["write_report", "estimate", "paired_comparisons", "intervention_changes"]
