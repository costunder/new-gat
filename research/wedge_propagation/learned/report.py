"""Report genuine synthetic message-recovery measurements for learned paths."""

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
_SPLITS = ("validation", "id", "size_ood", "family_ood", "family_size_ood")
_INTERVENTIONS = (
    "identity",
    "mean",
    "weight_shuffle",
    "other_graph_pattern",
    "correspondence_randomization",
)
_METRICS = (
    "message_relerr",
    "message_abs_rmse",
    "weight_relerr",
    "weight_corr",
    "weight_mean_error",
    "beta",
    "u",
    "v",
)
_FIGURES = ("learning_curves", "split_message_error", "intervention_error", "weight_recovery")


def _run_details(contract: dict) -> tuple[str, str, str, bool]:
    config = contract.get("config", {})
    config = config if isinstance(config, dict) else {}
    profile = str(contract.get("profile", config.get("profile", "unreported")))
    epochs = str(config.get("epochs", "unreported"))
    seeds = config.get("model_seeds")
    seed_count = str(len(seeds)) if isinstance(seeds, list) else "unreported"
    return profile, epochs, seed_count, profile == "debug" or contract.get("debug") is True


def _run_label(contract: dict) -> str:
    profile, epochs, seeds, debug = _run_details(contract)
    label = f"{profile.upper()} | epochs={epochs} | seeds={seeds}"
    return label + " | Verification only; not final performance" if debug else label


def _decorate(figure: Any, run_label: str, note: str | None = None) -> None:
    figure.suptitle(run_label + (f"\n{note}" if note else ""), fontsize=10)


def _contract_table(contract: dict) -> list[str]:
    config = contract.get("config", {})
    config = config if isinstance(config, dict) else {}
    runtime = contract.get("runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    gpu = runtime.get("gpu", {})
    gpu = gpu if isinstance(gpu, dict) else {}

    def display(value: Any) -> str:
        if value is None:
            return "unreported"
        if isinstance(value, list):
            return " / ".join(display(item) for item in value)
        return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)

    pairs = [
        ("Profile / epochs / seeds", _run_label(contract)),
        ("Model seeds", config.get("model_seeds")),
        (
            "Graphs total / used",
            [contract.get("graph_count_total"), contract.get("graph_count_used")],
        ),
        ("Feature inputs total", contract.get("input_count_total")),
        ("Data fraction", contract.get("data_fraction")),
        ("Physical graph batch", contract.get("physical_graph_batch_selected")),
        ("Effective training batch per seed", contract.get("effective_training_batch_per_seed")),
        (
            "Feature realizations / independent seeds in parallel",
            [contract.get("feature_realizations_parallel"), contract.get("seed_replicas_parallel")],
        ),
        ("Precision", contract.get("precision")),
        ("GPU / count used", [gpu.get("name"), runtime.get("gpu_count_used")]),
        ("GPU memory bytes", gpu.get("total_memory_bytes")),
        (
            "CPU workers / affinity cores",
            [contract.get("cpu_workers"), runtime.get("cpu_affinity_count")],
        ),
        ("RAM available bytes", runtime.get("ram_available_bytes")),
        ("Peak VRAM bytes by job", contract.get("peak_vram_by_job")),
    ]
    return [
        "| 실행 설정 / 자원 | 기록된 값 |",
        "| --- | --- |",
        *[f"| {name} | {display(value)} |" for name, value in pairs],
        "",
        "정확한 설정·자원 측정·source 계약은 [contract.json](contract.json)에 기록되어 있다. "
        "누락된 값은 unreported로 표시한다.",
    ]


def _finite(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number")
    if np.iscomplexobj(value) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")


def _integer(value: Any, name: str, minimum: int) -> None:
    _finite(value, name)
    if float(value) < minimum or not float(value).is_integer():
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_rows(rows: list[dict], *, intervention: bool = False) -> None:
    seen = set()
    for row in rows:
        for name, values in (("target", _TARGETS), ("condition", _CONDITIONS), ("split", _SPLITS)):
            if row.get(name) not in values:
                raise ValueError(f"Unrecognized {name}: {row.get(name)!r}")
        for name in ("graph_id", "family"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"{name} must be a nonempty string")
        for name, minimum in (
            ("seed", -1),
            ("num_nodes", 1),
            ("num_edges", 0),
            ("num_paths", 0),
            ("num_realizations", 1),
        ):
            _integer(row.get(name), name, minimum)
        if row["condition"] in _DETERMINISTIC:
            if row["seed"] != -1:
                raise ValueError("Deterministic fitted baselines must use seed=-1")
        elif row["seed"] < 0:
            raise ValueError("Learned conditions must use a nonnegative training seed")
        if "message_relerr" not in row:
            raise ValueError("Missing message_relerr; undefined references must use None")
        _finite(row["message_relerr"], "message_relerr", nullable=True)
        _finite(row.get("message_abs_rmse"), "message_abs_rmse")
        for name in _METRICS[2:]:
            _finite(row.get(name), name, nullable=True)
        key = tuple(row[name] for name in ("target", "condition", "seed", "split", "graph_id"))
        if intervention:
            if row.get("intervention") not in _INTERVENTIONS:
                raise ValueError(f"Unrecognized intervention: {row.get('intervention')!r}")
            if row["condition"] != "learned":
                raise ValueError("Intervention rows must use the learned condition")
            key += (row["intervention"],)
        if key in seen:
            raise ValueError(f"Duplicate graph measurement: {key}")
        seen.add(key)


def _validate_training(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if row.get("target") not in _TARGETS:
            raise ValueError("Training target is not recognized")
        if row.get("condition") not in ("learned", "random_pair"):
            raise ValueError("Training rows must describe learned or random_pair conditions")
        _integer(row.get("seed"), "seed", 0)
        _integer(row.get("epoch"), "epoch", 0)
        _finite(row.get("train_loss"), "train_loss")
        if "val_message_relerr" not in row:
            raise ValueError("Missing val_message_relerr")
        _finite(row["val_message_relerr"], "val_message_relerr", nullable=True)
        if row.get("best_epoch") is not None:
            _integer(row["best_epoch"], "best_epoch", 0)
        _finite(row.get("seconds"), "seconds")
        _finite(row.get("peak_vram_bytes"), "peak_vram_bytes", nullable=True)
        key = tuple(row[name] for name in ("target", "condition", "seed", "epoch"))
        if key in seen:
            raise ValueError(f"Duplicate training epoch: {key}")
        seen.add(key)


def _summarize(rows: list[dict], *, intervention: bool = False) -> list[dict]:
    """Average graphs within each seed, then compute variation across seeds."""
    fields = ("target", "condition", "split") + (("intervention",) if intervention else ())
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[tuple(row[name] for name in fields)][int(row["seed"])].append(row)
    output = []
    for key, seeds in groups.items():
        graph_sets = [{row["graph_id"] for row in values} for values in seeds.values()]
        if any(graph_set != graph_sets[0] for graph_set in graph_sets[1:]):
            raise ValueError(f"Different graph coverage across seeds: {key}")
        deterministic = key[1] in _DETERMINISTIC
        metrics = {}
        for name in _METRICS:
            seed_means = []
            undefined = 0
            for values in seeds.values():
                defined = [float(row[name]) for row in values if row.get(name) is not None]
                undefined += len(values) - len(defined)
                if defined:
                    seed_means.append(float(np.mean(defined)))
            count = len(seed_means)
            metrics[name] = {
                "mean": float(np.mean(seed_means)) if count else None,
                "std": (
                    float(np.std(seed_means, ddof=1))
                    if count > 1
                    else 0.0
                    if count == 1 and deterministic
                    else None
                ),
                "defined_seed_count": count,
                "undefined_graph_observations": undefined,
            }
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "seed_count": len(seeds),
                "graph_count": len(graph_sets[0]),
                "deterministic": deterministic,
                "metrics": metrics,
            }
        )
    return output


def _value(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6g}"


def _estimate(metric: dict) -> str:
    if metric["mean"] is None:
        return "undefined"
    if metric["std"] is None:
        return f"{metric['mean']:.6g} (std undefined)"
    return f"{metric['mean']:.6g} ± {metric['std']:.3g}"


def _save(figure: Any, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        with (directory / f"{name}.{extension}").open("xb") as stream:
            figure.savefig(stream, format=extension, dpi=180, bbox_inches="tight")


def _error_scale(axis: Any, values: list[float], *, floor: float) -> None:
    """Keep close values readable and retain zero when errors span many orders."""
    positive = [value for value in values if value > 0]
    if positive and max(positive) / min(positive) > 1000:
        axis.set_yscale("symlog", linthresh=floor)
    else:
        axis.set_yscale("linear")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)


def _plot_learning(plt: Any, directory: Path, rows: list[dict], run_label: str) -> None:
    targets = [target for target in _TARGETS if any(row["target"] == target for row in rows)]
    targets = targets or list(_TARGETS)
    figure, axes = plt.subplots(
        len(targets),
        2,
        figsize=(11.5, 3.4 * len(targets)),
        squeeze=False,
        constrained_layout=True,
    )
    try:
        for index, target in enumerate(targets):
            selected = [row for row in rows if row["target"] == target]
            for column, metric in enumerate(("train_loss", "val_message_relerr")):
                axis = axes[index, column]
                plotted = False
                for color, condition in enumerate(("learned", "random_pair")):
                    condition_rows = [row for row in selected if row["condition"] == condition]
                    epochs = defaultdict(list)
                    seeds = defaultdict(list)
                    for row in condition_rows:
                        if row[metric] is not None:
                            epochs[row["epoch"]].append(row[metric])
                            seeds[row["seed"]].append(row)
                    for seed_rows in seeds.values():
                        seed_rows.sort(key=lambda row: row["epoch"])
                        axis.plot(
                            [row["epoch"] for row in seed_rows],
                            [row[metric] for row in seed_rows],
                            color=f"C{color}",
                            alpha=0.2,
                            linewidth=0.7,
                        )
                    if epochs:
                        epoch_ids = sorted(epochs)
                        axis.plot(
                            epoch_ids,
                            [np.mean(epochs[epoch]) for epoch in epoch_ids],
                            color=f"C{color}",
                            label=condition,
                            linewidth=1.5,
                        )
                        plotted = True
                if not plotted:
                    axis.text(
                        0.5,
                        0.5,
                        "No defined measurements supplied",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                else:
                    axis.legend(
                        fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0
                    )
                axis.set_title(f"Target {target}: {metric}")
                axis.set_xlabel("Epoch")
                axis.set_ylabel(
                    "Training loss" if column == 0 else "Validation graph-macro relative error"
                )
                _error_scale(
                    axis, [row[metric] for row in selected if row[metric] is not None], floor=1e-8
                )
                axis.grid(True, alpha=0.25)
        _decorate(figure, run_label)
        _save(figure, directory, "learning_curves")
    finally:
        plt.close(figure)


def _plot_points(
    axis: Any,
    summaries: list[dict],
    target: str,
    metric: str,
    series: list[str],
    *,
    intervention: bool = False,
) -> list[float]:
    width = 0.75 / max(1, len(series))
    plotted = False
    measured_values = []
    for series_index, name in enumerate(series):
        labelled = False
        for split_index, split in enumerate(_SPLITS):
            selected = [
                row
                for row in summaries
                if row["target"] == target
                and row["split"] == split
                and row["intervention" if intervention else "condition"] == name
            ]
            if not selected:
                continue
            stat = selected[0]["metrics"][metric]
            if stat["mean"] is None:
                continue
            offset = (series_index - (len(series) - 1) / 2) * width
            axis.errorbar(
                split_index + offset,
                stat["mean"],
                yerr=stat["std"],
                fmt="o",
                markersize=4,
                capsize=2,
                color=f"C{series_index % 10}",
                label=name if not labelled else None,
            )
            labelled = True
            plotted = True
            measured_values.append(stat["mean"])
    axis.set_xticks(range(len(_SPLITS)), _SPLITS, rotation=30, ha="right")
    axis.set_xlabel("Evaluation split")
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    else:
        axis.text(
            0.5,
            0.5,
            "All measurements undefined or absent",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    return measured_values


def _plot_splits(plt: Any, directory: Path, summaries: list[dict], run_label: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    try:
        for axis, target in zip(axes, _TARGETS, strict=True):
            values = _plot_points(axis, summaries, target, "message_relerr", list(_CONDITIONS))
            axis.set_title(f"Target {target}: message error")
            axis.set_ylabel("Graph-macro relative error; seed mean ± sample std")
            _error_scale(axis, values, floor=1e-6)
        _decorate(figure, run_label)
        _save(figure, directory, "split_message_error")
    finally:
        plt.close(figure)


def _plot_interventions(
    plt: Any,
    directory: Path,
    summaries: list[dict],
    originals: list[dict],
    run_label: str,
) -> None:
    original_rows = [
        {**row, "intervention": "original"} for row in originals if row["condition"] == "learned"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    try:
        for axis, target in zip(axes, _TARGETS, strict=True):
            values = _plot_points(
                axis,
                original_rows + summaries,
                target,
                "message_relerr",
                ["original", *_INTERVENTIONS],
                intervention=True,
            )
            axis.set_title(f"Target {target}: frozen-checkpoint interventions")
            axis.set_ylabel("Graph-macro relative error; seed mean ± sample std")
            _error_scale(axis, values, floor=1e-6)
        _decorate(figure, run_label)
        _save(figure, directory, "intervention_error")
    finally:
        plt.close(figure)


def _plot_weights(plt: Any, directory: Path, summaries: list[dict], run_label: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    try:
        for axis, metric in zip(
            axes, ("weight_relerr", "weight_corr", "weight_mean_error"), strict=True
        ):
            values = _plot_points(axis, summaries, "path", metric, ["learned", "random_pair"])
            axis.set_title(f"Path teacher: {metric}")
            axis.set_ylabel("Graph-macro diagnostic; seed mean ± sample std")
            if metric == "weight_corr":
                axis.set_ylim(-1.05, 1.05)
            else:
                _error_scale(axis, values, floor=1e-8)
        _decorate(
            figure, run_label, "Weight diagnostics do not prove unique recovery of path weights"
        )
        _save(figure, directory, "weight_recovery")
    finally:
        plt.close(figure)


def _identity_mean_check(rows: list[dict]) -> tuple[int, float | None, float | None]:
    pairs = defaultdict(dict)
    for row in rows:
        if row["intervention"] in ("identity", "mean"):
            key = tuple(row[name] for name in ("target", "seed", "split", "graph_id"))
            pairs[key][row["intervention"]] = row
    relative, absolute = [], []
    for pair in pairs.values():
        if set(pair) != {"identity", "mean"}:
            continue
        identity, mean = pair["identity"], pair["mean"]
        absolute.append(abs(identity["message_abs_rmse"] - mean["message_abs_rmse"]))
        if identity["message_relerr"] is not None and mean["message_relerr"] is not None:
            relative.append(abs(identity["message_relerr"] - mean["message_relerr"]))
    return len(absolute), max(relative) if relative else None, max(absolute) if absolute else None


def _summary(
    summaries: list[dict],
    interventions: list[dict],
    intervention_rows: list[dict],
    training_rows: list[dict],
    contract: dict,
) -> str:
    profile, epochs, seeds, debug = _run_details(contract)
    lines = [
        "# 경로 가중치 학습: 합성 메시지 회수 실험 결과",
        "",
        f"**실행 profile: {profile}; 학습 계약: {epochs} epoch, {seeds} seed.**",
        "",
        (
            "**이 실행은 debug 검증이다. 짧은 학습의 출력·미분·평가 연결을 확인한 결과이며, "
            "최종 성능이나 연구 가설의 성공·실패를 판정하는 결과가 아니다.**"
            if debug
            else "실행 범위와 학습 기간은 아래 계약을 따른다."
        ),
        "",
        "이 결과는 알려진 목표 메시지를 회수하고, 고정한 학습 규칙을 새로운 그래프에 적용한 "
        "합성 실험이다. 실제 데이터셋의 분류 성능을 측정한 결과는 아니다.",
        "",
        "각 행의 특징 실현 평균을 먼저 그래프별 측정으로 사용한다. 그래프를 같은 비중으로 "
        "평균한 뒤 seed 평균과 표본 표준편차를 계산한다. 특징 실현 수나 경로 수로 그래프의 "
        "비중을 늘리지 않는다. 표의 그래프 수는 seed당 그래프 수다.",
        "",
        "현재 message_relerr는 각 특징 실현의 ||예측−목표||₂ / (||목표||₂+epsilon)을 "
        "그래프 안에서 평균한 값이다. epsilon은 실행 계약에 기록한다. 목표와 예측이 모두 "
        "0인 경우 이 규약의 값은 0이다. 메시지 absolute RMSE는 그래프의 노드·특징 실현 전체에 "
        "대한 오차의 제곱평균 제곱근이다.",
        "",
        "`first`, `polynomial`, `fixed`는 seed=-1의 결정론적 적합 한 번이다. 이들의 std=0은 "
        "독립적인 반복 학습을 수행했다는 뜻이 아니다. 학습 조건의 seed가 하나이면 표본 std는 "
        "undefined다. None인 상대오차·가중치 진단은 제외하고, undefined 개수를 명시한다.",
        "",
        "## 실제 메시지 오차",
        "",
        "| Target | Condition | Split | Graphs/seed | Seeds | Relative error: mean ± std "
        "| Absolute RMSE: mean ± std | Undefined relative graph observations |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for row in summaries:
        metric = row["metrics"]
        lines.append(
            f"| {row['target']} | {row['condition']} | {row['split']} | {row['graph_count']} "
            f"| {row['seed_count']} | {_estimate(metric['message_relerr'])} "
            f"| {_estimate(metric['message_abs_rmse'])} "
            f"| {metric['message_relerr']['undefined_graph_observations']} |"
        )
    lines.extend(
        [
            "",
            "## 양성 대조와 경로 모델의 실제 차이",
            "",
            "Target L에는 first, Target L2에는 polynomial이 충분한 구조를 가진다. 해당 조건의 "
            "실제 오차를 위 표에서 확인한다. 이 대조가 작동했다고 경로 모델의 보편적 우위를 "
            "입증한 것은 아니다.",
            "",
            "아래 차이는 learned의 평균 상대오차에서 각 고정 대조군의 오차를 뺀 값이다. "
            "음수는 관측된 평균 오차가 더 작고, 양수는 더 크다는 뜻이다. "
            "동률과 악화도 그대로 기록하며 통계적 유의성 검정으로 해석하지 않는다.",
            "",
            "| Path split | Learned relative error | Baseline | Baseline relative error "
            "| Learned − baseline | Observed mean ordering |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    indexed = {(row["target"], row["condition"], row["split"]): row for row in summaries}
    for split in _SPLITS:
        learned = indexed.get(("path", "learned", split))
        if learned is None:
            continue
        learned_value = learned["metrics"]["message_relerr"]["mean"]
        for baseline in ("first", "polynomial", "fixed", "random_pair"):
            reference = indexed.get(("path", baseline, split))
            if reference is None:
                continue
            baseline_value = reference["metrics"]["message_relerr"]["mean"]
            delta = (
                None
                if learned_value is None or baseline_value is None
                else learned_value - baseline_value
            )
            ordering = (
                "undefined"
                if delta is None
                else "lower"
                if delta < 0
                else "higher"
                if delta > 0
                else "equal"
            )
            lines.append(
                f"| {split} | {_value(learned_value)} | {baseline} "
                f"| {_value(baseline_value)} | {_value(delta)} | {ordering} |"
            )
    lines.extend(
        [
            "",
            "## 가중치 진단",
            "",
            "C의 평균을 1로 맞추면 전역 scale 보상을 제한할 수 있지만, 메시지 M에서 C를 "
            "유일하게 식별할 수 있다는 뜻은 아니다. q=0인 경로와 경로 행렬의 종속성 때문에 "
            "다른 가중치가 같은 메시지를 만들 수 있다. 출력 오차를 주지표로 사용하고, "
            "가중치 오차·상관은 보조 진단으로 읽는다. L/L2 target에는 식별 가능한 경로 "
            "teacher가 없으므로 teacher와의 가중치 오차·상관을 정의하지 않는다. "
            "Weight mean error는 Cmean=1 제약의 편차 검사다.",
            "",
            "| Path condition | Split | Weight relative error | Weight correlation "
            "| Weight mean error | Undefined graph observations: error / corr / mean |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in summaries:
        if row["target"] != "path":
            continue
        metrics = row["metrics"]
        names = ("weight_relerr", "weight_corr", "weight_mean_error")
        estimates = " | ".join(_estimate(metrics[name]) for name in names)
        undefined = " / ".join(str(metrics[name]["undefined_graph_observations"]) for name in names)
        lines.append(f"| {row['condition']} | {row['split']} | {estimates} | {undefined} |")
    count, relative_difference, absolute_difference = _identity_mean_check(intervention_rows)
    lines.extend(
        [
            "",
            "## 고정 checkpoint 개입",
            "",
            "Cmean=1인 설계에서는 identity와 mean 개입이 같은 결과를 내야 한다. "
            "이를 서로 다른 두 효과로 세지 않는다. 실제로 대응되는 측정 행의 차이도 확인했다.",
            "",
            f"Identity/mean 대응 행: {count}; "
            f"상대오차의 최대 절대 차이: {_value(relative_difference)}; "
            f"absolute RMSE의 최대 절대 차이: {_value(absolute_difference)}.",
            "",
            "| Target | Split | Intervention | Graphs/seed | Seeds | Relative error "
            "| Absolute RMSE | Undefined relative graph observations |",
            "| --- | --- | --- | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for row in interventions:
        metrics = row["metrics"]
        lines.append(
            f"| {row['target']} | {row['split']} | {row['intervention']} | {row['graph_count']} "
            f"| {row['seed_count']} | {_estimate(metrics['message_relerr'])} "
            f"| {_estimate(metrics['message_abs_rmse'])} "
            f"| {metrics['message_relerr']['undefined_graph_observations']} |"
        )
    lines.extend(
        [
            "",
            "경로 대응을 무작위화하면 입력 관계와 연산의 support도 달라질 수 있다. "
            "개입 오차 하나만으로 연속성의 독립적인 인과 효과를 단정하지 않는다.",
            "",
            "## 실행과 결과 파일",
            "",
            f"학습 로그 행: {len(training_rows)}. "
            "로그 행을 독립적인 seed나 그래프 표본으로 세지 않는다.",
            "",
            "- 원 측정: [metrics.csv](metrics.csv), [interventions.csv](interventions.csv), "
            "[training.csv](training.csv), [contract.json](contract.json).",
            "- [Teacher 연산 진단](teacher_operator_audit.csv): 특징 실현마다 목표 메시지를 "
            "span{LX, L²X, QX}에 사후적으로 맞춘 잔차와 입력에 따른 연산 변화를 기록한다. "
            "학습 모델의 일반화 성능이나 하나의 공유 다항식 모델을 평가한 결과는 아니다.",
            "- [학습 곡선](learning_curves.png) ([PDF](learning_curves.pdf)).",
            "- [Split별 메시지 오차](split_message_error.png) ([PDF](split_message_error.pdf)).",
            "- [개입 오차](intervention_error.png) ([PDF](intervention_error.pdf)).",
            "- [가중치 진단](weight_recovery.png) ([PDF](weight_recovery.pdf)).",
            "",
            "### 실행 계약 요약",
            "",
            *_contract_table(contract),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    output_dir: str | Path,
    metric_rows: list[dict],
    intervention_rows: list[dict],
    training_rows: list[dict],
    contract: dict,
) -> None:
    """Write graph-macro/seed summaries and real plots into an existing new run directory."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ValueError("Report output directory must already exist")
    if not metric_rows:
        raise ValueError("metric_rows must contain genuine measured graph outputs")
    _validate_rows(metric_rows)
    _validate_rows(intervention_rows, intervention=True)
    _validate_training(training_rows)
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON-serializable dictionary")
    json.dumps(contract, allow_nan=False)
    summaries = _summarize(metric_rows)
    interventions = _summarize(intervention_rows, intervention=True)
    destinations = [directory / "LEARNED_SUMMARY.md"] + [
        directory / f"{name}.{extension}" for name in _FIGURES for extension in ("png", "pdf")
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite existing learned-report artifacts")
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("Matplotlib is required: python -m pip install matplotlib") from error
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8}):
        run_label = _run_label(contract)
        _plot_learning(plt, directory, training_rows, run_label)
        _plot_splits(plt, directory, summaries, run_label)
        _plot_interventions(plt, directory, interventions, summaries, run_label)
        _plot_weights(plt, directory, summaries, run_label)
    with (directory / "LEARNED_SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(_summary(summaries, interventions, intervention_rows, training_rows, contract))
