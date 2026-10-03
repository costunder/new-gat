"""Complete paired classification and actual E/J branch-use reports."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from ...wedge_propagation.classification.common import write_csv

WEIGHTS = ("unit", "local_degree")
VARIANTS = ("base", "within", "between", "both")
CONDITIONS = tuple(f"{weight}__{variant}" for weight in WEIGHTS for variant in VARIANTS)
SPLITS = ("train", "validation", "test")
BRANCH_METRICS = (
    "alpha",
    "projected_norm",
    "base_norm",
    "energy_feature_norm",
    "relation_feature_norm",
    "energy_branch_norm",
    "relation_branch_norm",
    "energy_lift_norm",
    "relation_lift_norm",
    "energy_to_base",
    "relation_to_base",
    "energy_raw_mean",
    "relation_raw_mean",
    "relation_negative_fraction",
)
FIGURES = ("paired_contributions", "branch_use")
ARTIFACTS = (
    "LOCAL_PREDICTION_SUMMARY.md",
    "metric_estimates.csv",
    "paired_comparisons.csv",
    "interaction_estimates.csv",
    "intervention_changes.csv",
    "branch_strength_estimates.csv",
    *(f"figures/{name}.{suffix}" for name in FIGURES for suffix in ("png", "pdf")),
)


def _finite(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a real measurement")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; undefined uses None")
    return float(value)


def estimate(values):
    """Variation across all declared initialization seeds on the fixed split."""
    for value in values:
        _finite(value, "seed observation")
    if not values:
        return {"mean": None, "std": None, "lower": None, "upper": None, "count": 0}
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if len(data) > 1 else None
    half = (
        float(student_t.ppf(0.975, len(data) - 1) * std / math.sqrt(len(data)))
        if std is not None
        else None
    )
    return {
        "mean": mean,
        "std": std,
        "lower": None if half is None else mean - half,
        "upper": None if half is None else mean + half,
        "count": len(data),
    }


def intervention_variants(condition):
    _, variant = condition.split("__", 1)
    treatments = {
        "base": (),
        "within": ("within_remove",),
        "between": ("between_remove",),
        "both": ("within_remove", "between_remove", "both_remove"),
    }[variant]
    return [
        (treatment, target) for treatment in treatments for target in ("layer_0", "layer_1", "both")
    ]


def comparisons():
    result = [
        (f"{weight}__{first}", f"{weight}__{second}")
        for weight in WEIGHTS
        for first, second in (
            ("within", "base"),
            ("between", "base"),
            ("both", "base"),
            ("both", "within"),
            ("both", "between"),
        )
    ]
    return result + [(f"local_degree__{variant}", f"unit__{variant}") for variant in VARIANTS]


def _condition(row):
    condition = row.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("unknown prediction condition")
    weight, variant = condition.split("__", 1)
    if row.get("weight_mode") != weight or row.get("variant") != variant:
        raise ValueError("weight_mode/variant must match condition")


def _coverage(actual, expected, name):
    if actual != expected:
        raise ValueError(
            f"incomplete {name} coverage: missing={len(expected - actual)}, "
            f"extra={len(actual - expected)}"
        )


def _keys(rows, fields, name):
    result = set()
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate {name}")
        result.add(key)
    return result


def validate_rows(config, metrics, interventions, selections, resources, branches, contract):
    """Reject incomplete, nonfinite or incorrectly labelled scientific evidence."""
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("all eight fixed-C prediction conditions are required")
    if config.get("profile") not in ("full", "debug"):
        raise ValueError("explicit full/debug profile required")
    datasets = config["data"]["datasets"]
    seeds = config["training"]["final_seeds"]
    if (
        not datasets
        or not seeds
        or len(set(datasets)) != len(datasets)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("distinct complete declared datasets and seeds required")
    if config["profile"] == "full" and set(datasets) != {"Cora", "CiteSeer", "PubMed"}:
        raise ValueError("full prediction requires all three actual citation datasets")
    train = config["training"]
    tuning = (
        len(datasets)
        * len(CONDITIONS)
        * len(train["learning_rate_candidates"])
        * len(train["tuning_seeds"])
    )
    final = len(datasets) * len(CONDITIONS) * len(seeds)
    expected_budget = (tuning, final, tuning + final, (tuning + final) * train["epochs_per_run"])
    actual_budget = tuple(
        train[key] for key in ("tuning_runs", "final_runs", "total_runs", "total_updates")
    )
    if actual_budget != expected_budget:
        raise ValueError("inconsistent complete training budget")
    if contract.get("actual_data") is not (config["profile"] == "full"):
        raise ValueError("actual_data must distinguish FULL citation from DEBUG fixtures")
    if contract.get("profile", config["profile"]) != config["profile"]:
        raise ValueError("contract profile mismatch")
    coverage = contract.get("coverage", {})
    required = {
        "tuning_runs": tuning,
        "final_runs": final,
        "total_runs": tuning + final,
        "contract_optimizer_updates": expected_budget[-1],
        "all_datasets_conditions_seeds_splits": True,
    }
    if any(coverage.get(key) != value for key, value in required.items()):
        raise ValueError("completed training coverage must match the complete contract")

    identity = ("dataset", "condition", "seed", "split")
    expected = {
        (d, c, s, split) for d in datasets for c in CONDITIONS for s in seeds for split in SPLITS
    }
    _coverage(_keys(metrics, identity, "primary metric"), expected, "primary metric")
    expected_interventions = {
        (*key, treatment, target)
        for key in expected
        for treatment, target in intervention_variants(key[1])
    }
    _coverage(
        _keys(interventions, (*identity, "intervention", "target"), "frozen metric"),
        expected_interventions,
        "frozen metric",
    )
    hashes = {}
    for row in metrics + interventions:
        _condition(row)
        ce, accuracy = _finite(row.get("ce"), "ce"), _finite(row.get("accuracy"), "accuracy")
        if ce < 0 or not 0 <= accuracy <= 1:
            raise ValueError("CE must be nonnegative and accuracy must be in [0,1]")
        shape = config["data"]["expected_shapes"][row["dataset"]]
        if (
            row.get("num_nodes") != shape["nodes"]
            or row.get("num_labeled_nodes") != shape[row["split"]]
        ):
            raise ValueError("full graph/split node coverage changed")
        if "intervention" in row:
            sha = row.get("model_state_sha256")
            if (
                not isinstance(sha, str)
                or len(sha) != 64
                or any(char not in "0123456789abcdef" for char in sha)
            ):
                raise ValueError("frozen metrics require model state SHA256")
            key = tuple(row[field] for field in identity[:3])
            if key in hashes and hashes[key] != sha:
                raise ValueError("frozen model state changed across interventions")
            hashes[key] = sha

    selected = _keys(selections, ("dataset", "condition"), "LR selection")
    _coverage(selected, {(d, c) for d in datasets for c in CONDITIONS}, "LR selection")
    for row in selections:
        if any("test" in field.lower() for field in row):
            raise ValueError("test must not enter validation LR selection")
        lr = row.get("selected_lr", row.get("learning_rate"))
        if lr not in train["learning_rate_candidates"]:
            raise ValueError("selection must use a declared learning rate")
        ce = row.get("mean_tuning_validation_ce", row.get("mean_validation_ce"))
        if _finite(ce, "mean selected tuning validation CE") < 0:
            raise ValueError("selected validation CE must be nonnegative")

    branch_identity = ("dataset", "condition", "seed", "layer", "intervention", "target")
    expected_branches = {
        (d, c, s, layer, "original", "none")
        for d in datasets
        for c in CONDITIONS
        for s in seeds
        for layer in (0, 1)
    } | {
        (d, c, s, layer, treatment, target)
        for d in datasets
        for c in CONDITIONS
        for s in seeds
        for layer in (0, 1)
        for treatment, target in intervention_variants(c)
    }
    _coverage(
        _keys(branches, branch_identity, "layer diagnostic"), expected_branches, "layer diagnostic"
    )
    for row in branches:
        _condition(row)
        if not isinstance(row.get("base_nonzero"), (bool, np.bool_)):
            raise ValueError("base_nonzero must explicitly mark undefined ratios")
        for metric in BRANCH_METRICS:
            nullable = metric in ("energy_to_base", "relation_to_base")
            value = _finite(row.get(metric), metric, nullable=nullable)
            if nullable and (value is not None) != bool(row["base_nonzero"]):
                raise ValueError("branch ratio disagrees with base_nonzero")
            if value is not None and metric != "relation_raw_mean" and value < 0:
                raise ValueError("norm/energy/ratio diagnostics must be nonnegative")
        if not 0 <= row["alpha"] <= 1 or not 0 <= row["relation_negative_fraction"] <= 1:
            raise ValueError("coefficient/negative fraction must be in [0,1]")
        if bool(row["base_nonzero"]) != (row["base_norm"] > 0):
            raise ValueError("base norm and nonzero flag disagree")
        for prefix in ("energy", "relation"):
            ratio = row[f"{prefix}_to_base"]
            if ratio is not None and not math.isclose(
                ratio, row[f"{prefix}_branch_norm"] / row["base_norm"], rel_tol=1e-5, abs_tol=1e-10
            ):
                raise ValueError("branch ratio is inconsistent with measured norms")
        target = row["target"]
        affected = target == "both" or target == f"layer_{row['layer']}"
        if (
            affected
            and row["intervention"] in ("within_remove", "both_remove")
            and row["energy_branch_norm"] != 0
        ):
            raise ValueError("removed energy branch must contribute exactly zero")
        if (
            affected
            and row["intervention"] in ("between_remove", "both_remove")
            and row["relation_branch_norm"] != 0
        ):
            raise ValueError("removed relation branch must contribute exactly zero")
    for row in resources:
        if row.get("status") == "measured":
            for key in ("seconds_per_epoch", "packed_runs"):
                value = _finite(row.get(key), key)
                if value < 0 or (key == "packed_runs" and value < 1):
                    raise ValueError("invalid measured resource value")
            value = _finite(row.get("peak_vram_bytes"), "peak_vram_bytes", nullable=True)
            if value is not None and value < 0:
                raise ValueError("peak VRAM must be nonnegative")
    parameter_counts = contract.get("parameter_counts")
    if parameter_counts is not None:
        if set(parameter_counts) != set(datasets):
            raise ValueError("parameter counts require every dataset")
        for dataset in datasets:
            shape = config["data"]["expected_shapes"][dataset]
            hidden = config["backbone"]["hidden_dim"]
            base = shape["features"] * hidden + hidden * shape["classes"] + 2
            if set(parameter_counts[dataset]) != set(CONDITIONS):
                raise ValueError("parameter counts require every condition")
            for condition in CONDITIONS:
                branches_count = {"base": 0, "within": 1, "between": 1, "both": 2}[
                    condition.split("__")[1]
                ]
                expected_count = base + branches_count * (hidden + shape["classes"])
                if parameter_counts[dataset][condition] != expected_count:
                    raise ValueError("actual parameter count differs from active architecture")


def metric_estimates(rows):
    groups = defaultdict(list)
    for row in rows:
        for metric in ("accuracy", "ce"):
            groups[(row["dataset"], row["condition"], row["split"], metric)].append(row[metric])
    return [
        dict(zip(("dataset", "condition", "split", "metric"), key, strict=True)) | estimate(values)
        for key, values in sorted(groups.items())
    ]


def paired_comparisons(rows):
    lookup = {(row["dataset"], row["condition"], row["seed"], row["split"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate primary metric")
    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    return [
        {
            "dataset": d,
            "first": first,
            "second": second,
            "split": split,
            "metric": metric,
            **estimate(
                [
                    lookup[(d, first, s, split)][metric] - lookup[(d, second, s, split)][metric]
                    for s in seeds
                ]
            ),
        }
        for d in datasets
        for first, second in comparisons()
        for split in SPLITS
        for metric in ("accuracy", "ce")
    ]


def intervention_changes(rows, originals):
    lookup = {
        (row["dataset"], row["condition"], row["seed"], row["split"]): row for row in originals
    }
    groups = defaultdict(list)
    for row in rows:
        identity = tuple(row[field] for field in ("dataset", "condition", "seed", "split"))
        for metric in ("accuracy", "ce"):
            key = identity[:2] + identity[3:] + (row["intervention"], row["target"], metric)
            groups[key].append(row[metric] - lookup[identity][metric])
    return [
        dict(
            zip(
                ("dataset", "condition", "split", "intervention", "target", "metric"),
                key,
                strict=True,
            )
        )
        | estimate(values)
        for key, values in sorted(groups.items())
    ]


def interaction_estimates(rows):
    """Same-seed both−within−between+base, for every C/split/metric."""
    lookup = {(row["dataset"], row["condition"], row["seed"], row["split"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate primary metric")
    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    expected = {
        (dataset, condition, seed, split)
        for dataset in datasets
        for condition in CONDITIONS
        for seed in seeds
        for split in SPLITS
    }
    _coverage(set(lookup), expected, "interaction metric")
    result = []
    for dataset in datasets:
        for weight in WEIGHTS:
            for split in SPLITS:
                for metric in ("accuracy", "ce"):
                    values = [
                        lookup[(dataset, f"{weight}__both", seed, split)][metric]
                        - lookup[(dataset, f"{weight}__within", seed, split)][metric]
                        - lookup[(dataset, f"{weight}__between", seed, split)][metric]
                        + lookup[(dataset, f"{weight}__base", seed, split)][metric]
                        for seed in seeds
                    ]
                    result.append(
                        {
                            "dataset": dataset,
                            "weight_mode": weight,
                            "split": split,
                            "metric": metric,
                            "contrast": "both-within-between+base",
                            **estimate(values),
                        }
                    )
    return result


def branch_estimates(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["intervention"] != "original":
            continue
        for metric in BRANCH_METRICS:
            groups[(row["dataset"], row["condition"], row["layer"], metric)].append(row[metric])
    return [
        dict(zip(("dataset", "condition", "layer", "metric"), key, strict=True))
        | {
            "undefined_seeds": sum(value is None for value in values),
            **estimate([value for value in values if value is not None]),
        }
        for key, values in sorted(groups.items())
    ]


def _show(value, scale=1):
    return "undefined" if value is None else f"{value * scale:.5g}"


def _stat(row, scale=1):
    return f"{_show(row['mean'], scale)} ± {_show(row['std'], scale)}"


def _interval(row, scale=1):
    return f"[{_show(row['lower'], scale)}, {_show(row['upper'], scale)}]"


def _summary(
    config, metrics, paired, interactions, changes, branches, selections, resources, contract
):
    train = config["training"]
    lookup = {
        (row["dataset"], row["condition"], row["split"], row["metric"]): row for row in metrics
    }
    selected = {(row["dataset"], row["condition"]): row for row in selections}
    lines = [
        "# 로컬 이차 에너지·쌍선형 관계 — 실제 분류 결과",
        "",
        f"**{config['profile'].upper()}: {train['total_runs']} runs · "
        f"각 {train['epochs_per_run']} epoch · 최종 seed {train['final_seeds']}.**",
        f"실제 citation 데이터: {contract['actual_data']}. "
        "전체 데이터·노드·엣지·local 대응을 유지했다.",
        "두 고정 C × base/E/J/both의 여덟 조건을 새로 학습했다. "
        f"공통 기본 전파는 `(1−α)Z+αP_C²Z`, "
        f"2층 hidden {config['backbone']['hidden_dim']}다. C를 학습하지 않았다.",
        "E는 `E/(2FΣc)`, J는 `J_node/(F sqrt(tr(L_v²)tr(L_u²)))`의 이웃 평균이다. "
        "Topology 분모를 사용하며 feature RMS는 넣지 않았다. "
        "E/J의 부호와 실제 이차·쌍선형 정의를 유지했다.",
        "",
        "## 모든 최종 checkpoint와 split",
        "",
        "평균 ± 표본 std다. Accuracy는 %, CE는 L2를 제외한 평가 분류 손실이다.",
        "| 데이터 | 조건 | LR | Train acc | Train CE | Val acc | Val CE | Test acc | Test CE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in config["data"]["datasets"]:
        for condition in CONDITIONS:
            selection = selected[(dataset, condition)]
            lr = selection.get("selected_lr", selection.get("learning_rate"))
            cells = [
                value
                for split in SPLITS
                for value in (
                    _stat(lookup[(dataset, condition, split, "accuracy")], 100),
                    _stat(lookup[(dataset, condition, split, "ce")]),
                )
            ]
            lines.append(f"| {dataset} | {condition} | {lr:g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 두 분기를 함께 쓴 성능 상호작용",
        "",
        "같은 seed의 `both−within−between+base`다. 별도로 학습한 네 모델의 "
        "성능 차이이며 raw 에너지식의 교차항을 직접 측정한 값이 아니다.",
        "| 데이터 | C | Δtest accuracy pp ± std | 95% 구간 | Δtest CE ± std | 95% 구간 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    interaction_lookup = {
        (row["dataset"], row["weight_mode"], row["split"], row["metric"]): row
        for row in interactions
    }
    for dataset in config["data"]["datasets"]:
        for weight in WEIGHTS:
            key = dataset, weight, "test"
            accuracy, ce = interaction_lookup[(*key, "accuracy")], interaction_lookup[(*key, "ce")]
            lines.append(
                f"| {dataset} | {weight} | {_stat(accuracy, 100)} | "
                f"{_interval(accuracy, 100)} | {_stat(ce)} | {_interval(ce)} |"
            )
    lines += [
        "",
        "## 같은 seed의 예측 기여",
        "",
        "왼쪽 − 오른쪽이다. Accuracy는 양수, CE는 음수가 개선이다. "
        "95% t 구간은 같은 public split의 초기화 변동이며 다중 비교 보정을 포함하지 않는다.",
        "| 데이터 | 비교 | Δtest accuracy pp ± std | 95% 구간 | Δtest CE ± std | 95% 구간 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    pairs = {
        (row["dataset"], row["first"], row["second"], row["split"], row["metric"]): row
        for row in paired
    }
    for dataset in config["data"]["datasets"]:
        for first, second in comparisons():
            key = dataset, first, second, "test"
            accuracy, ce = pairs[(*key, "accuracy")], pairs[(*key, "ce")]
            lines.append(
                f"| {dataset} | {first} − {second} | {_stat(accuracy, 100)} | "
                f"{_interval(accuracy, 100)} | {_stat(ce)} | {_interval(ce)} |"
            )
    lines += [
        "",
        "## 실제 사용한 E/J 분기",
        "",
        "Norm은 모든 노드·채널의 값이다. 비율 분모가 0이면 undefined다. "
        "초기 lift 벡터는 0이고 아래는 선택된 최종 checkpoint의 관측이다.",
        "| 데이터 | 조건 | 층 | α | E feature norm | J feature norm | "
        "E lift norm | J lift norm | E/base | J/base | J 음수 % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    strength = {
        (row["dataset"], row["condition"], row["layer"], row["metric"]): row for row in branches
    }
    for dataset in config["data"]["datasets"]:
        for condition in CONDITIONS:
            for layer in (0, 1):
                fields = (
                    "alpha",
                    "energy_feature_norm",
                    "relation_feature_norm",
                    "energy_lift_norm",
                    "relation_lift_norm",
                    "energy_to_base",
                    "relation_to_base",
                    "relation_negative_fraction",
                )
                cells = [
                    _stat(
                        strength[(dataset, condition, layer, field)],
                        100 if field == "relation_negative_fraction" else 1,
                    )
                    for field in fields
                ]
                lines.append(f"| {dataset} | {condition} | {layer} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 같은 checkpoint의 분기 제거 — 두 층",
        "",
        "개입 − 원래 모델의 평균 Δtest accuracy pp / Δtest CE다. "
        "제거 뒤 다음 층 E/J는 현재 Z에서 다시 계산한다. 파라미터를 재학습하지 않았다.",
        "| 데이터 | 조건 | E 제거 | J 제거 | E/J 제거 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    change = {
        (
            row["dataset"],
            row["condition"],
            row["split"],
            row["intervention"],
            row["target"],
            row["metric"],
        ): row
        for row in changes
    }
    for dataset in config["data"]["datasets"]:
        for condition in CONDITIONS:
            if condition.endswith("__base"):
                continue
            cells = []
            active = {treatment for treatment, _ in intervention_variants(condition)}
            for treatment in ("within_remove", "between_remove", "both_remove"):
                if treatment not in active:
                    cells.append("적용 안 함")
                else:
                    key = dataset, condition, "test", treatment, "both"
                    cells.append(
                        f"{_show(change[(*key, 'accuracy')]['mean'], 100)} / "
                        f"{_show(change[(*key, 'ce')]['mean'])}"
                    )
            lines.append(f"| {dataset} | {condition} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 완료 범위와 자원",
        "",
        f"Tuning {train['tuning_runs']} / final {train['final_runs']} runs, "
        f"계약 optimizer updates {train['total_updates']}. "
        "모든 조건·seed·split의 coverage를 검증했다.",
    ]
    parameters = contract.get("parameter_counts")
    if parameters is not None:
        lines += [
            "",
            "실제 active trainable parameter 수:",
            "",
            "| 데이터 | base | E 또는 J | 둘 다 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for dataset in config["data"]["datasets"]:
            counts = parameters[dataset]
            lines.append(
                f"| {dataset} | {counts['unit__base']} | "
                f"{counts['unit__within']} | {counts['unit__both']} |"
            )
    measured = [row for row in resources if row.get("status") == "measured"]
    if measured:
        lines += [
            "",
            "실제로 계측한 resource 후보의 범위:",
            "",
            f"후보 {len(measured)}개, seconds/epoch "
            f"{min(row['seconds_per_epoch'] for row in measured):.5g}–"
            f"{max(row['seconds_per_epoch'] for row in measured):.5g}. "
            "선택된 packed 수·chunk·VRAM과 전체 측정값은 resources.csv에 보존했다.",
        ]
    lines += [
        "",
        "## 해석 범위",
        "",
        "재학습 비교는 추가 E/J 특징과 추가 파라미터를 함께 넣은 총 기여다. "
        "별도 용량 대조를 학습하지 않아 유일한 기하 구조 효과를 분리한 결과로 주장하지 않는다.",
        "분기 제거 반응만으로 기본 모델 대비 개선을 입증하지 않는다. "
        "재학습 차이·실제 분기 norm·lift 사용·제거 개입을 함께 읽는다.",
        "E/J는 projected 특징의 제곱 크기이므로 작게 남을 수 있다. "
        "음성 결과를 이차 에너지·쌍선형 관계 원리의 일반적인 실패로 해석하지 않는다.",
        "기존 public test를 본 뒤 정한 후속 실험이다. 독립 split/graph 일반화·learned C 성공·"
        "복구 불가능한 사이클 정보의 복원을 주장하지 않는다. 전역 Y의 반복 역복원 성공은 "
        "제한된 깊이의 국소 분류 개선과 다르다.",
        "[metrics.csv](metrics.csv), [interventions.csv](interventions.csv), "
        "[branch_diagnostics.csv](branch_diagnostics.csv), [resources.csv](resources.csv), "
        "[paired_comparisons.csv](paired_comparisons.csv), "
        "[interaction_estimates.csv](interaction_estimates.csv), "
        "[intervention_changes.csv](intervention_changes.csv), "
        "[branch_strength_estimates.csv](branch_strength_estimates.csv), "
        "[coverage.json](coverage.json), "
        "[contract.json](contract.json), [completion.json](completion.json).",
    ]
    if config["profile"] == "debug":
        lines += [
            "",
            "**DEBUG fixture·축소 예산의 연결 검사다. 실제 citation 성능으로 제출하지 않는다.**",
        ]
    return "\n".join(lines) + "\n"


def _figures(output, config, paired, branches):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = config["data"]["datasets"]
    labels = [(dataset, weight) for dataset in datasets for weight in WEIGHTS]
    colors = ("#176d8b", "#e1973f", "#59784a")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    lookup = {
        (row["dataset"], row["first"], row["second"], row["split"], row["metric"]): row
        for row in paired
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(12, max(4.5, len(labels) * 0.7)), constrained_layout=True
    )
    for axis, metric in zip(axes, ("accuracy", "ce"), strict=True):
        for offset, (variant, color) in enumerate(
            zip(("within", "between", "both"), colors, strict=True)
        ):
            values = [
                lookup[(dataset, f"{weight}__{variant}", f"{weight}__base", "test", metric)]
                for dataset, weight in labels
            ]
            scale = 100 if metric == "accuracy" else 1
            means = np.asarray([row["mean"] * scale for row in values])
            y = np.arange(len(labels)) + (offset - 1) * 0.18
            half = np.asarray(
                [
                    0 if row["lower"] is None else (row["mean"] - row["lower"]) * scale
                    for row in values
                ]
            )
            axis.errorbar(means, y, xerr=half, fmt="o", capsize=3, color=color, label=variant)
        axis.axvline(0, color="0.4", linewidth=1)
        axis.set_yticks(
            np.arange(len(labels)), [f"{dataset} / {weight}" for dataset, weight in labels]
        )
        axis.invert_yaxis()
        axis.set_xlabel(
            "Test accuracy change (percentage points)" if metric == "accuracy" else "Test CE change"
        )
        axis.set_title("Same-seed difference from base\n95% t interval")
        axis.grid(axis="x", alpha=0.2)
    axes[0].legend(loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"paired_contributions.{suffix}", dpi=180)
    plt.close(fig)

    strength = {
        (row["dataset"], row["condition"], row["layer"], row["metric"]): row for row in branches
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for layer, axis in enumerate(axes):
        for offset, (name, color) in enumerate(
            zip(("energy", "relation"), colors[:2], strict=True)
        ):
            values = [
                strength[(dataset, f"{weight}__both", layer, f"{name}_to_base")]
                for dataset, weight in labels
            ]
            defined = [index for index, row in enumerate(values) if row["mean"] is not None]
            y = [values[index]["mean"] for index in defined]
            axis.scatter(np.asarray(defined) + (offset - 0.5) * 0.18, y, color=color, label=name)
            for index, row in enumerate(values):
                if row["mean"] is None:
                    axis.text(index, 0, "undefined", rotation=90, fontsize=8, va="bottom")
        axis.set_xticks(
            np.arange(len(labels)),
            [f"{dataset}\n{weight}" for dataset, weight in labels],
            rotation=30,
            ha="right",
        )
        axis.set_title(f"Selected both model, layer {layer}")
        axis.set_ylabel("Actual branch norm / common base norm (seed mean)")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"branch_use.{suffix}", dpi=180)
    plt.close(fig)


def write_report(
    output, config, metric_rows, intervention_rows, selections, resource_rows, branch_rows, contract
):
    """Write reviewable results only after checking the complete declared experiment."""
    output = Path(output)
    if not output.is_dir():
        raise ValueError("report requires an existing new result directory")
    if any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError("prediction report refuses to overwrite existing artifacts")
    validate_rows(
        config, metric_rows, intervention_rows, selections, resource_rows, branch_rows, contract
    )
    metrics = metric_estimates(metric_rows)
    paired = paired_comparisons(metric_rows)
    interactions = interaction_estimates(metric_rows)
    changes = intervention_changes(intervention_rows, metric_rows)
    branches = branch_estimates(branch_rows)
    text = _summary(
        config,
        metrics,
        paired,
        interactions,
        changes,
        branches,
        selections,
        resource_rows,
        contract,
    )
    for filename, rows in (
        ("metric_estimates.csv", metrics),
        ("paired_comparisons.csv", paired),
        ("interaction_estimates.csv", interactions),
        ("intervention_changes.csv", changes),
        ("branch_strength_estimates.csv", branches),
    ):
        write_csv(output / filename, rows)
    _figures(output, config, paired, branches)
    with (output / "LOCAL_PREDICTION_SUMMARY.md").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(text)
    return {
        "summary": "LOCAL_PREDICTION_SUMMARY.md",
        "figures": len(FIGURES),
        "metric_estimate_rows": len(metrics),
        "paired_rows": len(paired),
        "interaction_rows": len(interactions),
        "intervention_change_rows": len(changes),
        "branch_strength_rows": len(branches),
    }
