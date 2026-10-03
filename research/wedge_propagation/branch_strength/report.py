"""Report frozen, post-hoc branch probes without choosing a new classifier."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

LEARNED = ("learned_wedge_raw", "learned_wedge_rms")
SINGLE = (
    "true_c_unit_denominator",
    "identity_hold_reference",
    "identity_unit_denominator",
    "identity_norm_matched",
    "branch_off",
)
SHUFFLE = ("shuffle_hold_reference", "shuffle_norm_matched")
TREATMENTS = ("baseline", *SINGLE, *SHUFFLE)
FIGURES = (
    "original_classification",
    "weight_and_denominator",
    "norm_matched_placement",
    "branch_message_geometry",
)
_BASE_KEY = ("dataset", "condition", "seed", "split")
_DIAGNOSTIC_NUMBERS = (
    "z_norm", "l_norm", "reference_branch_norm", "branch_norm", "alpha_l_norm",
    "beta_t_norm", "reference_beta_t_norm", "branch_delta_norm", "alpha_l_to_input",
    "beta_t_to_input", "beta_t_to_alpha_l", "branch_delta_to_input", "cosine_l_input",
    "cosine_t_input", "cosine_l_t", "branch_reference_cosine", "cosine_correction_input",
    "normalization_gain", "alpha", "beta", "kappa", "kappa_reference",
)


def _finite(value: Any, name: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real measurement")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; undefined measurements must be None")


def estimate(values: list[float]) -> dict:
    """Sample std and descriptive t interval for independent initialization seeds."""
    if not values:
        return {"mean": None, "std": None, "lower": None, "upper": None, "count": 0}
    for value in values:
        _finite(value, "seed estimate")
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else None
    half = (
        float(student_t.ppf(0.975, len(array) - 1) * std / math.sqrt(len(array)))
        if std is not None else None
    )
    return {
        "mean": mean, "std": std,
        "lower": None if half is None else mean - half,
        "upper": None if half is None else mean + half,
        "count": len(array),
    }


def paired_changes(
    baseline_rows: list[dict], treatment_rows: list[dict], metric: str
) -> list[dict]:
    """Average manifest differences within each seed before seed uncertainty."""
    if metric not in ("accuracy", "ce"):
        raise ValueError("paired metric must be accuracy or ce")
    baseline = {tuple(r[k] for k in _BASE_KEY): r[metric] for r in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError("duplicate baseline metric")
    per_seed: dict[tuple, list[float]] = defaultdict(list)
    for row in treatment_rows:
        base = tuple(row[k] for k in _BASE_KEY)
        if base not in baseline:
            raise ValueError("treatment has no paired original measurement")
        _finite(row[metric], metric)
        key = (*base, row["treatment"], row["target"])
        per_seed[key].append(row[metric] - baseline[base])
    groups: dict[tuple, list[float]] = defaultdict(list)
    for key, observations in per_seed.items():
        # dataset, condition, split, treatment, target; seed is the replicate.
        groups[(key[0], key[1], *key[3:])].append(float(np.mean(observations)))
    return [
        dict(zip(("dataset", "condition", "split", "treatment", "target"), key, strict=True))
        | {"metric": metric, **estimate(values)}
        for key, values in sorted(groups.items())
    ]


def diagnostic_value(row: dict, name: str) -> float | None:
    """Derive an actual message-norm ratio; never replace zero reference by zero."""
    if name == "message_norm_ratio":
        candidate = row.get("branch_norm")
        reference = row.get("reference_branch_norm")
        if candidate is None or reference is None or reference <= 0:
            return None
        return float(candidate / reference)
    return row.get(name)


def diagnostic_estimates(rows: list[dict], name: str) -> list[dict]:
    """Each manifest set belongs to one seed; count undefined seeds separately."""
    per_seed: dict[tuple, list] = defaultdict(list)
    for row in rows:
        fields = ("dataset", "condition", "seed", "layer", "treatment", "target")
        key = tuple(row[k] for k in fields)
        value = diagnostic_value(row, name)
        _finite(value, name, nullable=True)
        per_seed[key].append(value)
    groups: dict[tuple, list] = defaultdict(list)
    undefined: dict[tuple, int] = defaultdict(int)
    missing: dict[tuple, int] = defaultdict(int)
    for key, observations in per_seed.items():
        group = (key[0], key[1], *key[3:])
        valid = [value for value in observations if value is not None]
        missing[group] += len(observations) - len(valid)
        if valid:
            groups[group].append(float(np.mean(valid)))
        else:
            undefined[group] += 1
        groups.setdefault(group, [])
    names = ("dataset", "condition", "layer", "treatment", "target")
    return [
        dict(zip(names, key, strict=True)) | {
            "metric": name, "undefined_seeds": undefined[key],
            "undefined_observations": missing[key], **estimate(values),
        }
        for key, values in sorted(groups.items())
    ]


def _manifest_indices(treatment: str, count: int) -> range | tuple:
    return range(count) if treatment in SHUFFLE else (-1,)


def _validate(config, baseline, treatments, diagnostics, fixed_z, resources, contract):
    datasets = config["data"]["datasets"]
    conditions, seeds = config["conditions"], config["final_seeds"]
    count = config["manifests_per_dataset"]
    if not datasets or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("datasets and distinct final seeds are required")
    if set(config["treatments"]) != set(TREATMENTS):
        raise ValueError("report requires the complete declared branch-strength treatments")
    if set(config["targets"]) != {"layer_0", "layer_1", "both"}:
        raise ValueError("all three intervention targets are required")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("positive manifest count required")
    if any(name not in conditions for name in LEARNED):
        raise ValueError("both learned raw/RMS conditions required")
    expected_baseline = {
        (d, c, s, split) for d in datasets for c in conditions for s in seeds
        for split in ("train", "validation", "test")
    }
    actual = set()
    for row in baseline + treatments:
        for name in ("ce", "accuracy"):
            _finite(row.get(name), name)
        if row["ce"] < 0 or not 0 <= row["accuracy"] <= 1:
            raise ValueError("nonnegative CE and accuracy in [0,1] required")
    for row in baseline:
        key = tuple(row[k] for k in _BASE_KEY)
        if key in actual:
            raise ValueError("duplicate baseline measurement")
        actual.add(key)
    if actual != expected_baseline:
        raise ValueError("baseline coverage differs from the full configured set")
    expected_treatment = {
        (d, c, s, split, treatment, target, index)
        for d in datasets for c in LEARNED for s in seeds
        for split in ("train", "validation", "test")
        for treatment in TREATMENTS[1:] for target in config["targets"]
        for index in _manifest_indices(treatment, count)
    }
    actual = set()
    for row in treatments:
        key = tuple(row[k] for k in (*_BASE_KEY, "treatment", "target", "manifest_index"))
        if key in actual:
            raise ValueError("duplicate treatment measurement")
        actual.add(key)
    if actual != expected_treatment:
        raise ValueError("treatment coverage differs from the full configured set")
    expected_diag = {
        (d, c, s, layer, "baseline", "none", -1)
        for d in datasets for c in conditions for s in seeds for layer in (0, 1)
    }
    expected_diag |= {
        (d, c, s, layer, treatment, target, index)
        for d in datasets for c in LEARNED for s in seeds for layer in (0, 1)
        for treatment in TREATMENTS[1:] for target in config["targets"]
        for index in _manifest_indices(treatment, count)
    }
    expected_fixed = {
        (d, c, s, layer, treatment, f"layer_{layer}", index)
        for d in datasets for c in LEARNED for s in seeds for layer in (0, 1)
        for treatment in TREATMENTS for index in _manifest_indices(treatment, count)
    }
    diag_keys = ("dataset", "condition", "seed", "layer", "treatment", "target", "manifest_index")
    for rows, expected, source in (
        (diagnostics, expected_diag, "end_to_end"), (fixed_z, expected_fixed, "fixed_Z")
    ):
        actual = set()
        for row in rows:
            key = tuple(row[k] for k in diag_keys)
            if key in actual:
                raise ValueError("duplicate layer diagnostic")
            actual.add(key)
            if row.get("source") != source:
                raise ValueError("layer diagnostic source differs from its table")
            for name in _DIAGNOSTIC_NUMBERS:
                if name in row:
                    _finite(row[name], name, nullable=True)
                    flag = name + "_defined"
                    if flag in row and row[flag] is not (row[name] is not None):
                        raise ValueError(f"{name} undefined flag disagrees with measurement")
                    if row[name] is not None and "cosine" in name and abs(row[name]) > 1.00001:
                        raise ValueError("cosine outside numerical [-1,1] range")
            for name in ("branch_norm", "reference_branch_norm", "beta_t_to_input",
                         "branch_reference_cosine"):
                if name not in row:
                    raise ValueError(f"missing diagnostic {name}")
            for name in ("branch_norm", "reference_branch_norm"):
                if row[name] is not None and row[name] < 0:
                    raise ValueError("message norm must be nonnegative")
        if actual != expected:
            raise ValueError(f"{source} diagnostic coverage differs from configured set")
    lengths = {
        "baseline_rows": len(baseline), "treatment_rows": len(treatments),
        "diagnostic_rows": len(diagnostics), "fixed_z_rows": len(fixed_z),
    }
    expected_lengths = {key: config.get("expected_counts", {}).get(key) for key in lengths}
    if expected_lengths != lengths:
        raise ValueError("configured expected row counts differ from actual coverage")
    for name in ("models_unchanged", "source_artifacts_unchanged", "code_and_graphs_preserved",
                 "source_files_and_models_preserved"):
        if name in contract and contract[name] is not True:
            raise ValueError(f"frozen source guard failed: {name}")
    for name in ("optimizer_updates", "new_optimizer_updates", "new_training_epochs",
                 "frozen_model_updates", "optimization_updates"):
        if name in contract and contract[name] != 0:
            raise ValueError(f"post-hoc evaluation must have zero {name}")
    epochs = contract.get("source_config", {}).get("training", {}).get("epochs_per_run")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("source_config must record actual positive source training epochs")
    for row in resources:
        if row.get("measured") is True:
            for name in ("seconds_per_forward", "peak_vram_bytes", "model_forwards_per_second",
                         "wall_seconds"):
                if name in row:
                    _finite(row[name], name, nullable=True)
    return lengths


def _show(value, scale=1.0):
    return "undefined" if value is None else f"{value * scale:.6g}"


def _stat(values, scale=1.0):
    result = estimate([v for v in values if v is not None])
    return f"{_show(result['mean'], scale)} ± {_show(result['std'], scale)}"


def _interval(row, scale=1.0):
    return f"[{_show(row['lower'], scale)}, {_show(row['upper'], scale)}]"


def _label(config, contract):
    source = contract.get("source_config", {})
    epochs = source.get("training", {}).get("epochs_per_run", "recorded in source contract")
    return (
        f"{config['profile'].upper()} | source epochs={epochs} | "
        f"final seeds={len(config['final_seeds'])}\n"
        "Frozen post-hoc evaluation | new epochs=0 | updates=0"
    )


def _summary(config, baseline, treatments, diagnostics, fixed_z, resources, contract, counts):
    label = _label(config, contract)
    lines = [
        "# Experiment 4.1 — C 위치와 이차 분기 크기 분리", "", f"**{label}**", "",
        "이미 선택된 Experiment 4 checkpoint에서 수행한 **사후 탐색 관측**이다. "
        "새 학습·학습률 선택·checkpoint 선택·최적 개입 선택을 하지 않았다. "
        "Test를 관측한 후속 진단이므로 개선된 개입값을 "
        "새 모델의 확정 성능으로 제시하지 않는다.", "",
        "| 비교 | 바꾸는 것 | 유지하는 것 |", "| --- | --- | --- |",
        "| 같은 분기 크기 | C=1 또는 C 위치 shuffle | 현재 층 원래 메시지의 Frobenius norm |",
        "| 같은 방향, 다른 크기 | true C에서 κ 분모를 1로 변경 | "
        "현재 층 이차 메시지 방향, 학습한 α·β |",
        "| 두 요인 분리 | true C/C=1 × 원래 κ/분모 1 | checkpoint·입력·물리 그래프 |", "",
        "Norm matching은 모든 노드·channel의 이차 분기 크기를 seed별로 맞춘다. "
        "전체 update의 norm, 다른 분기와의 상쇄, ReLU 이후 표현까지 같게 만드는 것은 아니다. "
        "End-to-end에서는 변경된 현재 Z에서 층마다 true C와 참조 κ를 다시 계산한다. "
        "Fixed-Z 진단은 원래 forward의 깨끗한 Z를 고정해 연산 자체를 비교한다.", "",
    ]
    if config["profile"] == "debug":
        lines += [
            "**DEBUG fixture 및 source의 축소 학습 예산을 사용한 연결 검사다. "
            "실제 citation 본학습 성능으로 제출하지 않는다.**", "",
        ]
    lines += [
        "## 원래 선택된 모델", "",
        "| Dataset | Condition | Test accuracy % ± seed std | Test CE ± seed std |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset in config["data"]["datasets"]:
        for condition in config["conditions"]:
            rows = [r for r in baseline if r["dataset"] == dataset
                    and r["condition"] == condition and r["split"] == "test"]
            lines.append(
                f"| {dataset} | {condition} | {_stat([r['accuracy'] for r in rows], 100)} | "
                f"{_stat([r['ce'] for r in rows])} |"
            )
    changes = {
        metric: paired_changes(baseline, treatments, metric) for metric in ("accuracy", "ce")
    }
    lines += [
        "", "## 두 층을 함께 바꾼 결과", "",
        "차이는 **개입 − 원래 checkpoint**다. Accuracy는 양수가 개선(pp), CE는 음수가 개선이다. "
        "Shuffle manifest를 각 seed 안에서 먼저 평균한 뒤 "
        "seed 평균·표본 std·paired 95% t 구간을 계산했다. "
        "Manifest를 독립 학습 seed로 세지 않는다. "
        "구간은 고정 public split에서 초기화 변동만 나타내며 새 split/그래프의 불확실성이나 "
        "다중 비교 보정을 포함하지 않는다.", "",
        "| Dataset | Condition | Treatment | Δaccuracy pp ± seed std | Paired 95% interval | ΔCE |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    ce_lookup = {
        (r["dataset"], r["condition"], r["split"], r["treatment"], r["target"]): r
        for r in changes["ce"]
    }
    for row in changes["accuracy"]:
        if row["split"] != "test" or row["target"] != "both":
            continue
        key = tuple(row[k] for k in ("dataset", "condition", "split", "treatment", "target"))
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['treatment']} | "
            f"{_show(row['mean'], 100)} ± {_show(row['std'], 100)} | "
            f"{_interval(row, 100)} | {_show(ce_lookup[key]['mean'])} |"
        )
    lines += [
        "", "원래 κ를 유지하는 것과 실제 메시지 norm을 맞추는 것은 다르다. "
        "`identity_norm_matched`/`shuffle_norm_matched`의 변화는 "
        "분기 크기를 통제한 가중치 배치 진단이다. "
        "`true_c_unit_denominator`는 현재 층의 원래 메시지를 κ배 하므로 "
        "방향을 유지하고 강도를 바꾼다. "
        "분모를 없앤 연산과 norm-matched 연산에는 "
        "원래의 spectral norm≤1 보장이 그대로 적용되지 않는다. "
        "관측 개선만으로 κ가 학습 실패의 원인이라고 확정하지 않는다.", "",
        "## 실제 메시지 크기·방향", "",
        "깨끗한 Z를 고정한 층 진단이다. Norm ratio는 candidate/reference이며 "
        "기준 norm=0이면 undefined다. "
        "Cosine은 원래 메시지와의 방향 관계다. βT/Z는 학습 β를 포함한 이차 분기 크기다. "
        "α·β는 재학습하지 않는다. Undefined 관측은 0으로 채우지 않는다.", "",
        "| Dataset | Condition | Layer | Treatment | Norm ratio | Reference cosine | βT/Z | "
        "Undefined ratio / cosine observations |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    statistics = {
        metric: diagnostic_estimates(fixed_z, metric)
        for metric in ("message_norm_ratio", "branch_reference_cosine", "beta_t_to_input")
    }
    lookup = {
        metric: {tuple(r[k] for k in ("dataset", "condition", "layer", "treatment", "target")): r
                 for r in records}
        for metric, records in statistics.items()
    }
    for row in statistics["message_norm_ratio"]:
        if row["treatment"] not in ("baseline", "identity_norm_matched", "shuffle_norm_matched"):
            continue
        key = tuple(row[k] for k in ("dataset", "condition", "layer", "treatment", "target"))
        cosine = lookup["branch_reference_cosine"][key]
        ratio = lookup["beta_t_to_input"][key]
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['layer']} | {row['treatment']} | "
            f"{_show(row['mean'])} | {_show(cosine['mean'])} | {_show(ratio['mean'])} | "
            f"{row['undefined_observations']} / {cosine['undefined_observations']} |"
        )
    measured = [r for r in resources if r.get("measured") is True and r.get("status") == "measured"]
    lines += [
        "", "그림 treatment index: 1=baseline, 2=true C/1, 3=C=1/참조 κ, "
        "4=C=1/1, 5=C=1/norm match, 6=shuffle/참조 κ, 7=shuffle/norm match, "
        "8=branch off.",
        "", "## 범위와 실행 증거", "",
        f"원래 분류 {counts['baseline_rows']}, 개입 {counts['treatment_rows']}, "
        f"end-to-end 층 진단 {counts['diagnostic_rows']}, fixed-Z 진단 {counts['fixed_z_rows']}행. "
        "각 층 개별 개입(layer_0/layer_1)과 두 층 개입(both)을 모두 보존했다. "
        "원래 모델 재현 수치와 source/model/code 보존 여부는 "
        "contract의 실제 검사 기록을 따른다.", "",
        "새 optimizer update와 epoch는 0이다. 원래 8조건 120 checkpoint(Full)와 learned 30개를 "
        "재사용하며 추가 다운로드나 학습을 하지 않는다. 새 독립 split/graph 일반화나 "
        "경로 연속성만의 인과 효과를 입증하지 않는다.", "",
        f"측정된 자원 행 {len(measured)}개. OOM·거부·미측정 값은 처리량 0으로 바꾸지 않는다. "
        "Packed seed 수는 독립 모델 동시 처리 수이며 graph batch를 바꾼 것이 아니다.", "",
        "| Dataset | Condition | Packed seeds | Completed seed cases | "
        "Peak VRAM GiB | Evaluation wall seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in measured:
        if row.get("scope") != "evaluation":
            continue
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['packed_runs']} | "
            f"{_show(row.get('completed_seed_cases'))} | "
            f"{_show(row.get('peak_vram_bytes'), 1 / 2**30)} | "
            f"{_show(row.get('wall_seconds'))} |"
        )
    lines += [
        "",
        "Evaluation wall 시간에는 fixed-Z, metric, 진단 수집이 포함된다. "
        "순수 forward 처리량으로 해석하지 않는다. "
        "Calibration 후보의 실제 forward 시간·처리량은 resources.csv에 따로 기록했다.", "",
        "[baseline_metrics.csv](baseline_metrics.csv), [treatments.csv](treatments.csv), "
        "[layer_diagnostics.csv](layer_diagnostics.csv), [fixed_z.csv](fixed_z.csv), "
        "[resources.csv](resources.csv), [contract.json](contract.json), "
        "[completion.json](completion.json).", "",
    ]
    for name in FIGURES:
        lines.append(f"[{name}]({name}.png) ([PDF]({name}.pdf)).")
    return "\n".join(lines) + "\n"


def _point_curve(axis, positions, statistics, label, *, scale=1.0, interval=False):
    present = [(position, row) for position, row in zip(positions, statistics, strict=True)
               if row["mean"] is not None]
    if not present:
        return
    x = [p for p, _ in present]
    y = [r["mean"] * scale for _, r in present]
    (line,) = axis.plot(x, y, marker="o", label=label)
    known = [(p, r) for p, r in present if r["std"] is not None]
    if known:
        errors = [(r["upper"] - r["mean"] if interval else r["std"]) * scale
                  for _, r in known]
        axis.errorbar([p for p, _ in known], [r["mean"] * scale for _, r in known],
                      yerr=errors, fmt="none", capsize=3, color=line.get_color())


def _save(figure, output, name, label):
    figure.suptitle(label, fontsize=11)
    handles, labels = [], []
    for axis in figure.axes:
        for handle, name_label in zip(*axis.get_legend_handles_labels(), strict=True):
            if name_label not in labels:
                handles.append(handle)
                labels.append(name_label)
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncols=min(4, len(labels)),
                      fontsize=8)
    for extension in ("png", "pdf"):
        with (output / f"{name}.{extension}").open("xb") as stream:
            figure.savefig(stream, format=extension, dpi=180, bbox_inches="tight")


def write_report(
    output_dir: Path, config: dict, baseline_rows: list[dict], treatment_rows: list[dict],
    diagnostic_rows: list[dict], fixed_z_rows: list[dict], resource_rows: list[dict],
    contract: dict,
) -> dict:
    """Write real PNG/PDF figures and a readable summary to a fresh directory."""
    output = Path(output_dir)
    if not output.is_dir():
        raise NotADirectoryError("runner must create a fresh output directory before reporting")
    artifacts = [output / "BRANCH_STRENGTH_SUMMARY.md"]
    artifacts += [output / f"{name}.{ext}" for name in FIGURES for ext in ("png", "pdf")]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("previous report artifacts exist and are preserved")
    counts = _validate(config, baseline_rows, treatment_rows, diagnostic_rows,
                       fixed_z_rows, resource_rows, contract)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("real report figures require matplotlib; install matplotlib") from error
    datasets, conditions = config["data"]["datasets"], config["conditions"]
    label = _label(config, contract)
    changes = {metric: paired_changes(baseline_rows, treatment_rows, metric)
               for metric in ("accuracy", "ce")}
    change_lookup = {
        metric: {tuple(r[k] for k in ("dataset", "condition", "target", "treatment")): r
                 for r in records if r["split"] == "test"}
        for metric, records in changes.items()
    }
    figure, axes = plt.subplots(2, len(datasets), figsize=(5.5 * len(datasets), 7),
                                squeeze=False, layout="constrained")
    for column, dataset in enumerate(datasets):
        for row_index, metric in enumerate(("accuracy", "ce")):
            stats = [estimate([r[metric] for r in baseline_rows if r["dataset"] == dataset
                               and r["condition"] == condition and r["split"] == "test"])
                     for condition in conditions]
            axis = axes[row_index, column]
            _point_curve(axis, list(range(len(conditions))), stats, "Original checkpoints",
                         scale=100 if metric == "accuracy" else 1)
            axis.set(title=f"{dataset}: original test {metric}",
                     ylabel="Accuracy % ± seed std" if metric == "accuracy" else "CE ± seed std",
                     xticks=range(len(conditions)), xticklabels=range(1, len(conditions) + 1),
                     xlabel="Condition index (see summary table)")
            axis.grid(alpha=0.2)
    _save(figure, output, FIGURES[0], label)
    plt.close(figure)

    factorial = ("true_c_unit_denominator", "identity_hold_reference", "identity_unit_denominator")
    figure, axes = plt.subplots(2, len(datasets), figsize=(5.5 * len(datasets), 7),
                                squeeze=False, layout="constrained")
    for column, dataset in enumerate(datasets):
        for row_index, metric in enumerate(("accuracy", "ce")):
            axis = axes[row_index, column]
            for condition in LEARNED:
                stats = [change_lookup[metric][(dataset, condition, "both", treatment)]
                         for treatment in factorial]
                _point_curve(axis, list(range(3)), stats, condition.removeprefix("learned_wedge_"),
                             scale=100 if metric == "accuracy" else 1, interval=True)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set(title=f"{dataset}: both layers, change from original",
                     ylabel="Accuracy change pp / paired 95% t" if metric == "accuracy"
                     else "CE change / paired 95% t", xticks=range(3),
                     xticklabels=("True C / 1", "C=1 / ref κ", "C=1 / 1"))
            axis.grid(alpha=0.2)
    _save(figure, output, FIGURES[1], label)
    plt.close(figure)

    placement = ("identity_hold_reference", "identity_norm_matched",
                 "shuffle_hold_reference", "shuffle_norm_matched")
    figure, axes = plt.subplots(3, len(datasets), figsize=(5.5 * len(datasets), 10),
                                squeeze=False, layout="constrained")
    for column, dataset in enumerate(datasets):
        for row_index, target in enumerate(config["targets"]):
            axis = axes[row_index, column]
            for condition in LEARNED:
                stats = [change_lookup["accuracy"][(dataset, condition, target, treatment)]
                         for treatment in placement]
                _point_curve(axis, list(range(4)), stats, condition.removeprefix("learned_wedge_"),
                             scale=100, interval=True)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set(title=f"{dataset}: {target}", ylabel="Accuracy change pp / paired 95% t",
                     xticks=range(4), xticklabels=("Identity / ref κ", "Identity / same norm",
                                                 "Shuffle / ref κ", "Shuffle / same norm"))
            axis.tick_params(axis="x", rotation=15)
            axis.grid(alpha=0.2)
    _save(figure, output, FIGURES[2], label)
    plt.close(figure)

    geometry = ("baseline", "true_c_unit_denominator", "identity_hold_reference",
                "identity_unit_denominator", "identity_norm_matched", "shuffle_hold_reference",
                "shuffle_norm_matched", "branch_off")
    names = ("message_norm_ratio", "beta_t_to_input", "branch_reference_cosine")
    lookup = {
        metric: {tuple(r[k] for k in ("dataset", "condition", "layer", "treatment")): r
                 for r in diagnostic_estimates(fixed_z_rows, metric)} for metric in names
    }
    figure, axes = plt.subplots(3, len(datasets), figsize=(5.5 * len(datasets), 10),
                                squeeze=False, layout="constrained")
    for column, dataset in enumerate(datasets):
        for row_index, metric in enumerate(names):
            axis = axes[row_index, column]
            for condition in LEARNED:
                for layer in (0, 1):
                    stats = [lookup[metric][(dataset, condition, layer, treatment)]
                             for treatment in geometry]
                    _point_curve(axis, list(range(len(geometry))), stats,
                                 f"{condition.removeprefix('learned_wedge_')} / layer {layer}")
            axis.set(title=f"{dataset}: clean fixed Z", ylabel=f"{metric} ± seed std",
                     xticks=range(len(geometry)), xticklabels=range(1, len(geometry) + 1),
                     xlabel="Treatment index (see summary)")
            if metric == "branch_reference_cosine":
                axis.set_ylim(-1.05, 1.05)
            if not any(line.get_xdata().size for line in axis.lines):
                axis.text(0.5, 0.5, "Undefined: zero reference norm", transform=axis.transAxes,
                          ha="center")
            axis.grid(alpha=0.2)
    _save(figure, output, FIGURES[3], label)
    plt.close(figure)
    summary_path = output / "BRANCH_STRENGTH_SUMMARY.md"
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_summary(config, baseline_rows, treatment_rows, diagnostic_rows,
                              fixed_z_rows, resource_rows, contract, counts))
    return {"report_counts": counts, "figure_count": len(FIGURES), "new_optimizer_updates": 0}


__all__ = ["write_report", "estimate", "paired_changes", "diagnostic_value", "diagnostic_estimates"]
