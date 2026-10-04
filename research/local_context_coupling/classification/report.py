"""Complete classification comparisons and separate frozen gain interventions."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from ...wedge_propagation.classification.common import digest, write_csv, write_json
from .model import CONDITIONS, SCOPES, VARIANTS, WEIGHTS, parse_condition

SPLITS = ("train", "validation", "test")
BRANCH_METRICS = (
    "projected_norm", "output_norm", "off_norm", "matched_delta_norm",
    "matched_delta_relative", "context_norm", "cross_energy_before",
    "cross_energy_after", "intra_energy_before", "intra_energy_after",
    "rho", "rho_original", "theta",
)
FIGURES = ("cross_effects", "frozen_effects", "branch_use")
ARTIFACTS = (
    "LOCAL_CONTEXT_CLASSIFICATION_SUMMARY.md", "metric_estimates.csv",
    "paired_comparisons.csv", "intervention_changes.csv", "branch_estimates.csv",
    "report_checks.json",
    *(f"figures/{name}.{suffix}" for name in FIGURES for suffix in ("png", "pdf")),
)


def _finite(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; undefined measurements use None")
    return float(value)


def estimate(values):
    """Mean/sample std/t interval across declared initialization seeds."""
    data = np.asarray([_finite(value, "seed observation") for value in values], dtype=float)
    if not data.size:
        return {"mean": None, "std": None, "lower": None, "upper": None, "count": 0}
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if data.size > 1 else None
    half = None if std is None else float(student_t.ppf(.975, data.size - 1) * std / math.sqrt(data.size))
    return {"mean": mean, "std": std, "lower": None if half is None else mean - half,
            "upper": None if half is None else mean + half, "count": int(data.size)}


def comparisons():
    within = [(f"{mode}__{a}", f"{mode}__{b}", "within_weight_mode")
              for mode in WEIGHTS for a, b in (("fixed", "off"), ("learned", "off"), ("learned", "fixed"))]
    return within + [(f"local_degree__{variant}", f"unit__{variant}", "between_weight_modes_exploratory")
                     for variant in VARIANTS]


def intervention_variants(condition):
    return () if parse_condition(condition)[1] == "off" else tuple(
        (treatment, target) for treatment in ("gain0", "gain1") for target in SCOPES
    )


def _identity(row):
    mode, variant = parse_condition(row["condition"])
    if row.get("weight_mode") != mode or row.get("variant") != variant:
        raise ValueError("condition must match weight_mode and variant")


def _unique(rows, fields, name):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate {name} row")
        result[key] = row
    return result


def _coverage(actual, expected, name):
    if set(actual) != set(expected):
        raise ValueError(f"incomplete {name} coverage: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}")


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be SHA256")


def validate_rows(config, evaluated, selections, resources, completion):
    """Reject incomplete evidence, invalid values, mislabeled controls and units."""
    if tuple(config.get("conditions", ())) != CONDITIONS or config.get("profile") not in ("full", "debug"):
        raise ValueError("explicit profile and all six conditions required")
    datasets, seeds = config["data"]["datasets"], config["training"]["final_seeds"]
    if not datasets or len(datasets) != len(set(datasets)) or len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("distinct datasets and at least two final seeds required")
    if config["profile"] == "full" and set(datasets) != {"Cora", "CiteSeer", "PubMed"}:
        raise ValueError("FULL requires all three citation datasets")
    if completion.get("actual_data") is not (config["profile"] == "full"):
        raise ValueError("actual_data must distinguish DEBUG fixtures from FULL")
    if completion.get("profile") != config["profile"] or completion.get("config_digest") != digest(config):
        raise ValueError("configuration provenance mismatch")
    source = completion.get("source", {})
    if not isinstance(source.get("sha256"), dict) or not source["sha256"] or source.get("code_digest") != digest(source["sha256"]):
        raise ValueError("scientific source provenance mismatch")
    for value in (*source["sha256"].values(), completion.get("data_manifest_digest")):
        _sha(value, "source/data hash")
    training = config["training"]
    tuning = len(datasets) * len(CONDITIONS) * len(training["learning_rate_candidates"]) * len(training["tuning_seeds"])
    final = len(datasets) * len(CONDITIONS) * len(seeds)
    budget = dict(tuning_runs=tuning, final_runs=final, total_runs=tuning + final,
                  total_updates=(tuning + final) * training["epochs_per_run"])
    if any(training.get(key) != value for key, value in budget.items()):
        raise ValueError("complete training budget mismatch")
    coverage = completion.get("coverage", {})
    expected_contract = {key: budget[key] for key in ("tuning_runs", "final_runs", "total_runs")}
    expected_contract["contract_optimizer_updates"] = budget["total_updates"]
    if any(coverage.get(key) != value for key, value in expected_contract.items()) or coverage.get("all_datasets_conditions_seeds_splits") is not True:
        raise ValueError("completion training coverage mismatch")
    metrics, interventions, branches = (evaluated[key] for key in ("metric_rows", "intervention_rows", "branch_rows"))
    originals = _unique(metrics, ("dataset", "condition", "seed", "split"), "primary")
    _coverage(originals, {(d, c, s, split) for d in datasets for c in CONDITIONS for s in seeds for split in SPLITS}, "primary")
    intervened = _unique(interventions, ("dataset", "condition", "seed", "split", "intervention", "target"), "intervention")
    _coverage(intervened, {(d, c, s, split, i, target) for d in datasets for c in CONDITIONS for s in seeds
                          for split in SPLITS for i, target in intervention_variants(c)}, "intervention")
    branch_index = _unique(branches, ("dataset", "condition", "seed", "layer", "intervention", "target"), "branch")
    _coverage(branch_index, {(d, c, s, layer, i, target) for d in datasets for c in CONDITIONS for s in seeds
                            for layer in (0, 1) for i, target in (("original", "none"), *intervention_variants(c))}, "branch")
    selected = _unique(selections, ("dataset", "condition"), "selection")
    _coverage(selected, {(d, c) for d in datasets for c in CONDITIONS}, "selection")
    for row in selections:
        _, variant = parse_condition(row["condition"])
        if row.get("variant") != variant or row.get("selection_scope") != "validation_only_independent_tuning_seeds":
            raise ValueError("selection must use validation only and matching variant")
        if row.get("selected_lr") not in training["learning_rate_candidates"]:
            raise ValueError("selected LR is outside declared candidates")
        if _finite(row["mean_tuning_validation_ce"], "tuning CE") < 0:
            raise ValueError("tuning CE must be nonnegative")
    for row in (*metrics, *interventions):
        _identity(row)
        if _finite(row["ce"], "CE") < 0 or not 0 <= _finite(row["accuracy"], "accuracy fraction") <= 1:
            raise ValueError("classification CE/accuracy range invalid")
        shape = config["data"]["expected_shapes"][row["dataset"]]
        if row.get("num_nodes") != shape["nodes"] or row.get("num_labeled_nodes") != shape[row["split"]]:
            raise ValueError("reported node/split count must match complete input")
        _sha(row.get("model_state_sha256"), "frozen model state")
        if "intervention" in row:
            original = originals[row["dataset"], row["condition"], row["seed"], row["split"]]
            if row["model_state_sha256"] != original["model_state_sha256"]:
                raise ValueError("frozen intervention changed model state")
            noop = row["variant"] == "fixed" and row["intervention"] == "gain1"
            if row.get("no_op_expected") is not noop:
                raise ValueError("frozen no-op control label mismatch")
    counts = completion.get("parameter_counts", {})
    for row in branches:
        _identity(row)
        if row.get("theta_available") is not (row["variant"] == "learned") or type(row.get("off_nonzero")) is not bool:
            raise ValueError("branch parameter/denominator flags must be explicit")
        for key in BRANCH_METRICS:
            _finite(row.get(key), key, nullable=key in ("theta", "matched_delta_relative"))
        if (row["theta"] is None) != (row["variant"] != "learned"):
            raise ValueError("theta is present only in learned conditions")
        if any(row[key] < 0 for key in BRANCH_METRICS if key not in ("theta", "matched_delta_relative")):
            raise ValueError("norms, energies and gains must be nonnegative")
        if not 0 <= row["rho"] <= 1 or not 0 <= row["rho_original"] <= 1:
            raise ValueError("cross gain must be in [0,1]")
        original_gain = {"off": 0, "fixed": 1}.get(row["variant"])
        if original_gain is not None and row["rho_original"] != original_gain:
            raise ValueError("fixed/off original gain mismatch")
        active = row["intervention"] != "original" and row["layer"] in SCOPES[row["target"]]
        expected_gain = int(row["intervention"] == "gain1") if active else row["rho_original"]
        if row["rho"] != expected_gain:
            raise ValueError("effective gain does not match selected intervention layers")
        if row["off_nonzero"] is not (row["off_norm"] > 0):
            raise ValueError("off_nonzero must match off norm")
        if row["off_nonzero"]:
            if row["matched_delta_relative"] is None or not math.isclose(row["matched_delta_relative"], row["matched_delta_norm"] / row["off_norm"], rel_tol=3e-6, abs_tol=1e-12):
                raise ValueError("matched ratio must use actual off norm")
        elif row["matched_delta_relative"] is not None:
            raise ValueError("zero-denominator matched ratio must be undefined")
        shape = config["data"]["expected_shapes"][row["dataset"]]
        expected_parameters = shape["features"] * config["backbone"]["hidden_dim"] + config["backbone"]["hidden_dim"] * shape["classes"] + int(row["variant"] == "learned")
        if counts.get(row["dataset"], {}).get(row["condition"]) != expected_parameters:
            raise ValueError("completion parameter count mismatch")
        if row.get("parameters_per_seed") != expected_parameters or row.get("trainable_parameters_per_seed") != expected_parameters:
            raise ValueError("active model parameter count mismatch")
        original = originals[row["dataset"], row["condition"], row["seed"], "test"]
        if row.get("model_state_sha256") != original["model_state_sha256"]:
            raise ValueError("branch diagnostic model state mismatch")
        noop = row["variant"] == "fixed" and row["intervention"] == "gain1"
        if row.get("no_op_expected") is not noop:
            raise ValueError("branch no-op control label mismatch")
    if not isinstance(resources, list) or not resources:
        raise ValueError("measured resource rows required")
    for row in resources:
        if row.get("status") not in ("measured", "reused_complete", "out_of_memory", "OOM", "memory_safety_rejected"):
            raise ValueError("resource status must preserve measurement/failure/reuse")
        if row.get("seconds_per_epoch") is not None and _finite(row["seconds_per_epoch"], "epoch seconds") < 0:
            raise ValueError("epoch time must be nonnegative")
    proofs = evaluated.get("provenance")
    if not isinstance(proofs, list) or not proofs:
        raise ValueError("all frozen model preservation records required")
    preserved = set()
    for proof in proofs:
        if proof.get("parameters_preserved") is not True or proof.get("optimizer_updates") != 0:
            raise ValueError("frozen preservation proof changed model or optimizer")
        _sha(proof.get("before_sha256"), "before frozen model")
        if proof.get("before_sha256") != proof.get("after_sha256"):
            raise ValueError("before/after frozen model hashes differ")
        for seed in proof["seeds"]:
            key = (proof["dataset"], proof["condition"], seed)
            if key in preserved:
                raise ValueError("duplicate frozen preservation coverage")
            preserved.add(key)
            original = originals[*key, "test"]
            if proof["before_sha256"] != original["model_state_sha256"]:
                raise ValueError("frozen preservation proof does not match metric state")
    _coverage(preserved, {(d, c, s) for d in datasets for c in CONDITIONS for s in seeds}, "frozen preservation")
    return {**budget, "primary_metric_rows": len(metrics), "intervention_metric_rows": len(interventions),
            "branch_rows": len(branches), "complete_all_declared_rows": True,
            "scope": "same_fixed_public_split_initialization_seed_variation" if config["profile"] == "full" else "DEBUG_fixture_pipeline_only"}


def metric_estimates(metrics):
    groups = defaultdict(list)
    for row in metrics:
        groups[row["dataset"], row["condition"], row["split"]].append(row)
    result = []
    for (dataset, condition, split), rows in sorted(groups.items()):
        acc = estimate([100 * row["accuracy"] for row in rows])
        ce = estimate([row["ce"] for row in rows])
        result.append({"dataset": dataset, "condition": condition, "split": split,
                       "weight_mode": rows[0]["weight_mode"], "variant": rows[0]["variant"],
                       "seed_count": len(rows), "accuracy_mean_percent": acc["mean"], "accuracy_std_pp": acc["std"],
                       "ce_mean": ce["mean"], "ce_std": ce["std"]})
    return result


def paired_comparisons(metrics):
    indexed = _unique(metrics, ("dataset", "condition", "seed", "split"), "paired primary")
    datasets, seeds = sorted({row["dataset"] for row in metrics}), sorted({row["seed"] for row in metrics})
    result = []
    for dataset in datasets:
        for condition, reference, scope in comparisons():
            for split in SPLITS:
                for metric, scale, unit in (("accuracy", 100, "percentage_points"), ("ce", 1, "loss")):
                    values = [scale * (indexed[dataset, condition, seed, split][metric] - indexed[dataset, reference, seed, split][metric]) for seed in seeds]
                    result.append({"dataset": dataset, "condition": condition, "reference": reference, "split": split,
                                   "comparison_scope": scope, "metric": metric, "unit": unit, **estimate(values)})
    return result


def intervention_changes(metrics, interventions):
    originals = _unique(metrics, ("dataset", "condition", "seed", "split"), "intervention reference")
    groups = defaultdict(list)
    for row in interventions:
        groups[row["dataset"], row["condition"], row["split"], row["intervention"], row["target"]].append(row)
    result = []
    for (dataset, condition, split, treatment, target), rows in sorted(groups.items()):
        for metric, scale, unit in (("accuracy", 100, "percentage_points"), ("ce", 1, "loss")):
            values = [scale * (row[metric] - originals[dataset, condition, row["seed"], split][metric]) for row in rows]
            result.append({"dataset": dataset, "condition": condition, "split": split, "intervention": treatment,
                           "target": target, "metric": metric, "unit": unit,
                           "comparison_scope": "frozen_selected_checkpoint_intervention_minus_original",
                           "no_op_expected": rows[0]["no_op_expected"], **estimate(values)})
    return result


def branch_estimates(branches):
    groups = defaultdict(list)
    for row in branches:
        groups[row["dataset"], row["condition"], row["layer"], row["intervention"], row["target"]].append(row)
    result = []
    for (dataset, condition, layer, treatment, target), rows in sorted(groups.items()):
        value = {"dataset": dataset, "condition": condition, "layer": layer, "intervention": treatment,
                 "target": target, "seed_count": len(rows), "theta_available": rows[0]["theta_available"],
                 "no_op_expected": rows[0]["no_op_expected"], "off_nonzero_seeds": sum(row["off_nonzero"] for row in rows)}
        for metric in BRANCH_METRICS:
            measured = estimate([row[metric] for row in rows if row[metric] is not None])
            value.update({f"{metric}_{key}": measured[key] for key in ("mean", "std", "count")})
        result.append(value)
    return result


def _number(value):
    return "undefined" if value is None else f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)


def _meanstd(mean, std):
    return _number(mean) + (" ± " + _number(std) if std is not None else "")


def _interval(row):
    return f"{_number(row['mean'])} [{_number(row['lower'])}, {_number(row['upper'])}]"


def _table(rows, columns):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(_number(row[key]) for key in columns) + " |" for row in rows)])


def _figures(output, config, paired, frozen, branches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    datasets = config["data"]["datasets"]
    ordered = comparisons()[:6]
    labels = [c.replace("local_degree", "degree").replace("__", " ") + " − " + r.split("__")[1] for c, r, _ in ordered]
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.6 * len(datasets), 8.3), layout="constrained", squeeze=False)
    for col, dataset in enumerate(datasets):
        for line, metric in enumerate(("accuracy", "ce")):
            axis = axes[line, col]
            rows = [next(row for row in paired if row["dataset"] == dataset and row["condition"] == c and row["reference"] == r and row["split"] == "test" and row["metric"] == metric) for c, r, _ in ordered]
            means = np.asarray([row["mean"] for row in rows])
            errors = np.asarray([[row["mean"] - row["lower"] for row in rows], [row["upper"] - row["mean"] for row in rows]])
            axis.barh(np.arange(6), means, xerr=errors, color=["#2171b5"] * 3 + ["#d95f0e"] * 3, capsize=3)
            axis.set_yticks(np.arange(6), labels, fontsize=8)
            axis.axvline(0, color="black", linewidth=.8)
            axis.invert_yaxis()
            axis.set_title(dataset)
            axis.set_xlabel("Accuracy difference (pp); higher is better" if metric == "accuracy" else "CE difference; lower is better")
            if metric == "ce":
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-3, 3))
                axis.xaxis.set_major_formatter(formatter)
            axis.grid(axis="x", alpha=.2)
    fig.suptitle(f"{config['profile'].upper()} · retrained comparisons · paired 95% t intervals")
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"cross_effects.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.9 * len(datasets), 4.8), layout="constrained", squeeze=False)
    active = [c for c in CONDITIONS if not c.endswith("__off")]
    treatments = intervention_variants(active[0])
    limit = max(abs(row["mean"]) for row in frozen if row["metric"] == "ce" and row["split"] == "test") or 1e-12
    for col, dataset in enumerate(datasets):
        axis = axes[0, col]
        values = [[next(row["mean"] for row in frozen if row["dataset"] == dataset and row["condition"] == condition and row["intervention"] == i and row["target"] == target and row["split"] == "test" and row["metric"] == "ce") for i, target in treatments] for condition in active]
        image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(6), [i + " " + target.replace("layer_", "L") for i, target in treatments], rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(4), [c.replace("local_degree", "degree").replace("__", " ") for c in active], fontsize=8)
        axis.set_title(dataset)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Frozen test ΔCE; negative is better", shrink=.8)
    fig.suptitle(f"{config['profile'].upper()} · same-checkpoint interventions · fixed gain1 is a no-op control")
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"frozen_effects.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.7 * len(datasets), 7.4), layout="constrained", squeeze=False)
    labels = [c.replace("local_degree", "degree").replace("__", " ") for c in CONDITIONS]
    for col, dataset in enumerate(datasets):
        original = [row for row in branches if row["dataset"] == dataset and row["intervention"] == "original"]
        gain = [next(row["rho_mean"] for row in original if row["condition"] == c and row["layer"] == 0) for c in CONDITIONS]
        axes[0, col].bar(np.arange(6), gain, color=["#2171b5"] * 3 + ["#d95f0e"] * 3)
        axes[0, col].set(ylim=(0, 1.05), ylabel="Shared original cross gain", title=dataset)
        for layer, color in ((0, "#225ea8"), (1, "#b35806")):
            values = [next(row["matched_delta_relative_mean"] for row in original if row["condition"] == c and row["layer"] == layer) for c in CONDITIONS]
            axes[1, col].plot(np.arange(6), values, "o-", color=color, label=f"layer {layer}")
        axes[1, col].set_ylabel("Actual matched increment / off norm")
        axes[1, col].legend(fontsize=8)
        for line in (0, 1):
            axes[line, col].set_xticks(np.arange(6), labels, rotation=40, ha="right", fontsize=8)
            axes[line, col].grid(axis="y", alpha=.2)
    fig.suptitle(f"{config['profile'].upper()} · selected checkpoints · gain and actual layer effects")
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"branch_use.{suffix}", dpi=160)
    plt.close(fig)


def write_report(output, config, evaluated, selections, resources, completion):
    output = Path(output)
    checks = validate_rows(config, evaluated, selections, resources, completion)
    metrics = metric_estimates(evaluated["metric_rows"])
    paired = paired_comparisons(evaluated["metric_rows"])
    frozen = intervention_changes(evaluated["metric_rows"], evaluated["intervention_rows"])
    branches = branch_estimates(evaluated["branch_rows"])
    for name, rows in (("metric_estimates.csv", metrics), ("paired_comparisons.csv", paired),
                       ("intervention_changes.csv", frozen), ("branch_estimates.csv", branches)):
        write_csv(output / name, rows)
    checks.update(metric_estimate_rows=len(metrics), paired_comparison_rows=len(paired),
                  intervention_estimate_rows=len(frozen), branch_estimate_rows=len(branches))
    write_json(output / "report_checks.json", checks)
    mindex = {(row["dataset"], row["condition"], row["split"]): row for row in metrics}
    selection = {(row["dataset"], row["condition"]): row for row in selections}
    main = []
    for dataset in config["data"]["datasets"]:
        for condition in CONDITIONS:
            rows = {split: mindex[dataset, condition, split] for split in SPLITS}
            main.append({"데이터": dataset, "조건": condition, "LR": selection[dataset, condition]["selected_lr"],
                         "Train acc %": _meanstd(rows["train"]["accuracy_mean_percent"], rows["train"]["accuracy_std_pp"]),
                         "Val acc %": _meanstd(rows["validation"]["accuracy_mean_percent"], rows["validation"]["accuracy_std_pp"]),
                         "Test acc %": _meanstd(rows["test"]["accuracy_mean_percent"], rows["test"]["accuracy_std_pp"]),
                         "Test CE": _meanstd(rows["test"]["ce_mean"], rows["test"]["ce_std"])})
    contrast_rows = [{"데이터": row["dataset"], "조건": row["condition"], "기준": row["reference"],
                      "지표": "accuracy pp" if row["metric"] == "accuracy" else "CE", "차이 [95% 구간]": _interval(row),
                      "범위": row["comparison_scope"]} for row in paired if row["split"] == "test"]
    intervention_rows = [{"데이터": row["dataset"], "조건": row["condition"], "개입": row["intervention"],
                         "층": row["target"], "지표": "accuracy pp" if row["metric"] == "accuracy" else "CE",
                         "차이 [95% 구간]": _interval(row), "no-op": row["no_op_expected"]}
                        for row in frozen if row["split"] == "test"]
    branch_rows = [{"데이터": row["dataset"], "조건": row["condition"], "층": row["layer"],
                   "rho": _meanstd(row["rho_mean"], row["rho_std"]), "theta": _meanstd(row["theta_mean"], row["theta_std"]),
                   "실제 Δ/off": _meanstd(row["matched_delta_relative_mean"], row["matched_delta_relative_std"]),
                   "context norm": row["context_norm_mean"], "cross E 전": row["cross_energy_before_mean"],
                   "cross E 후": row["cross_energy_after_mean"]} for row in branches if row["intervention"] == "original"]
    text = f"""# 로컬 문맥 결합: 노드 분류 결과

Profile **{config['profile'].upper()}**, 실제 citation 데이터 **{completion['actual_data']}**.
두 macro 층·hidden{config['backbone']['hidden_dim']}·각 {config['training']['epochs_per_run']}epoch,
전체 {checks['tuning_runs']}tuning+{checks['final_runs']}final={checks['total_runs']}run,
계약상 독립 optimizer update {checks['total_updates']}개다.
재개 실행의 새 update는 완료 파일의 `new_optimizer_updates`와 구분한다.
DEBUG이면 별도 fixture의 구현 검사이며 citation 본학습 성능이 아니다.

## 실제 비교

각 층은 투영 특징을 모든 로컬 copy에 복제하고 intra→cross→intra→균등 merge한다.
두 C조건은 고정 unit/local_degree이며, 학습하는 것은 learned 조건의 seed당 cross 강도 θ 하나다.
ρ=sigmoid(θ)를 두 macro 층에서 공유한다. Fixed는ρ=1, off는ρ=0이며 θ 파라미터가 없다.
모든 조건의 내부 전파 두 번·정규화·전체 graph·seed 초기 projection/dropout 규칙을 맞췄다.
Scalar E/J lift나 learned edge C를 쓰는 모델의 결과로 해석하지 않는다.

## 주 분류 결과

평균±표본 std다. Accuracy는 %, CE는 L2를 제외한 평가 분류 손실이다.
LR/checkpoint는 validation만으로 선택하고 모든 final 선택을 고정한 뒤 test를 읽었다.

{_table(main, tuple(main[0]))}

## 재학습한 조건 간 paired 차이

아래 차이는 조건−기준이다. Accuracy pp는 양수, CE는 음수가 개선이다.
같은 final seed끼리 먼저 차이를 계산한 뒤 평균·표본 std·95% t 구간을 만들었다.
동일 C의 fixed−off/learned−off/learned−fixed가 핵심 비교이고, C조건 간 비교는 탐색적이다.
Train/validation/test 전체는 `paired_comparisons.csv`에 있다.

{_table(contrast_rows, tuple(contrast_rows[0]))}

## 고정 checkpoint의 gain 개입

차이는 개입 후−동일 checkpoint의 원래 예측이다. Gain0/1을 지정 층에 적용하고 후속 특징을 다시 계산했다.
원래 학습 파라미터·buffer hash를 보존하며 optimizer update는0이다.
Fixed/gain1은 예상 no-op control이고 off에서는 별도 개입을 출력하지 않는다.
새로 off로 학습한 모델과 gain0으로 바꾼 checkpoint의 효과는 다른 질문이다.

{_table(intervention_rows, tuple(intervention_rows[0]))}

## 선택된 모델의 실제 cross 작용

Δ/off는 현재 층의 동일 투영 특징에서 `||Tρ Z−T0 Z||/||T0 Z||`다.
Θ가 있거나ρ가 양수라는 사실과 실제 context/출력 변화·분류 기여를 구분한다.
Zero denominator는 undefined로 기록하며0을 대신 넣지 않는다.
원래/개입 두 층의 전체 norm·energy·gain은 `branch_estimates.csv`와 원시 branch CSV에 있다.

{_table(branch_rows, tuple(branch_rows[0]))}

## 범위와 자원

- Primary metric {checks['primary_metric_rows']}행, frozen metric {checks['intervention_metric_rows']}행, layer branch {checks['branch_rows']}행을 전부 검증했다.
- Resource 기록 {len(resources)}행에는 CPU 준비·packed seed/edge chunk 후보·학습 측정·재개 여부를 보존한다.
  Calibration/실제 학습/재개 기록을 섞어 평균 epoch 시간이나 완료 시간을 만들지 않는다.
- Source{config['source']['graphs']} graph 중 분류에 사용하는 것은 선언된 세 citation 또는 DEBUG graph다.
  합성{config['source']['synthetic_graphs']} graph를 label 학습에 쓰지 않는다.
- Citation public split의{len(config['training']['final_seeds'])}개 초기화 seed 변동은 독립 split/새 graph 일반화가 아니다.
- 이전 test를 확인한 후의 후속 연구이고 다중 비교 보정이 없다. 0 포함 구간은 동등성 증명이 아니다.
- 통합 copy 에너지는 전파의 정의다. Classification CE로 학습하며 에너지/복원 loss를 추가하지 않았다.
- Cross-on/off 연산 차이만으로 정확도 개선·정보 복원·선행연구 대비 신규성이 입증되지는 않는다.

원시 metric·intervention·branch·frozen provenance, source/data hash, resource, completion과 이 요약을 함께 읽는다.
"""
    with (output / "LOCAL_CONTEXT_CLASSIFICATION_SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(text)
    _figures(output, config, paired, frozen, branches)
    return checks
