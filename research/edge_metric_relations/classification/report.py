"""Complete seed-paired contrasts, Holm18 and separate frozen interventions."""
from __future__ import annotations

from collections import defaultdict
import itertools
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .common import CONDITIONS, RECIPES, VARIANTS, condition_metadata, digest, parse_condition, write_csv, write_json
from .evaluation import BRANCH_METRICS, intervention_scopes, intervention_variants

SPLITS = ("train", "validation", "test")
FIGURES = ("primary_contrasts", "classification", "branch_use")
ARTIFACTS = ("EDGE_METRIC_CLASSIFICATION_SUMMARY.md", "metric_estimates.csv", "paired_comparisons.csv",
             "intervention_changes.csv", "branch_estimates.csv", "report_checks.json",
             *(f"figures/{name}.{suffix}" for name in FIGURES for suffix in ("png", "pdf")))


def _finite(value, name, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; undefined values are None")
    return float(value)


def estimate(values):
    data = np.asarray([_finite(v, "seed observation") for v in values], dtype=float)
    if not data.size:
        return {"count": 0, "mean": None, "std": None, "lower": None, "upper": None}
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if data.size > 1 else None
    half = None if std is None else float(student_t.ppf(.975, data.size - 1)) * std / math.sqrt(data.size)
    return {"count": int(data.size), "mean": mean, "std": std,
            "lower": mean - half if half is not None else None, "upper": mean + half if half is not None else None}


def paired_test(values):
    result = estimate(values)
    if result["count"] < 2:
        raise ValueError("paired inference requires at least two independent initialization seeds")
    se = result["std"] / math.sqrt(result["count"])
    if se > 0:
        statistic = result["mean"] / se
        p = float(2 * student_t.sf(abs(statistic), result["count"] - 1))
    else:
        statistic = 0. if result["mean"] == 0 else None
        p = 1. if result["mean"] == 0 else 0.
    return {**result, "standard_error": se, "t_statistic": statistic, "degrees_of_freedom": result["count"] - 1,
            "p_value": p, "zero_sample_variance": se == 0, "test": "two_sided_paired_t"}


def holm(values):
    """Adjusted probabilities in input order, valid without independence assumptions."""
    p = np.asarray([_finite(v, "p value") for v in values])
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted, previous = np.empty_like(p), 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., float(p[index] * (len(p) - rank))))
        adjusted[index] = previous
    return adjusted.tolist()


def comparisons():
    result = []
    for recipe in RECIPES:
        c = lambda v: f"{recipe}__{v}"
        result.extend(((c("F2"), c("D1"), "F2-D1", True), (c("F2"), c("DA"), "F2-DA", True),
                       (c("F2"), "P2", "F2-P2", True)))
        result.extend((c(a), c(b), f"{a}-{b}", False) for a, b in
                      (("F1", "D0"), ("F1", "F0"), ("F2", "F1"), ("D1", "D0"), ("F0", "D0"), ("DA", "D1")))
        for variant in VARIANTS:
            result.extend((c(variant), baseline, f"{variant}-{baseline}", False)
                          for baseline in ("G1", "G2", "P2") if not (variant == "F2" and baseline == "P2"))
    result.extend((f"local_degree__{v}", f"unit__{v}", "between_recipe", False) for v in VARIANTS)
    result.extend((("G2", "G1", "baseline_G2-G1", False), ("P2", "G1", "baseline_P2-G1", False)))
    if len({(c, r) for c, r, _, _ in result}) != len(result):
        raise RuntimeError("duplicate declared contrasts")
    return tuple(result)


def _identity(row):
    if any(row.get(k) != v for k, v in condition_metadata(row["condition"]).items()):
        raise ValueError("condition metadata differs from declared operator")


def _unique(rows, fields, name):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        if key in result:
            raise ValueError(f"duplicate {name} row")
        result[key] = row
    return result


def _coverage(actual, expected, name):
    if set(actual) != set(expected):
        raise ValueError(f"incomplete {name} coverage: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}")


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("SHA256 provenance is required")


def expected_parameters(config, dataset, condition):
    shape = config["data"]["expected_shapes"][dataset]
    hidden = config["backbone"]["hidden_dim"]
    variant = parse_condition(condition)[3]
    extra = 2 * (385 * int(variant in ("D1", "F2", "DA")) + 641 * int(variant in ("F1", "F2", "DA")))
    return shape["features"] * hidden + hidden * shape["classes"] + extra + 6 * int(variant == "P2")


def validate_rows(config, evaluated, selections, resources, completion):
    if tuple(config.get("conditions", ())) != CONDITIONS or config.get("profile") not in ("full", "debug"):
        raise ValueError("explicit profile and all fifteen conditions required")
    datasets, seeds, training = config["data"]["datasets"], config["training"]["final_seeds"], config["training"]
    if len(set(datasets)) != 3 or len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("three distinct datasets and distinct final seeds required")
    if tuple(config["evaluation"]["primary_contrasts"]) != ("F2-D1", "F2-DA", "F2-P2"):
        raise ValueError("the eighteen primary accuracy contrasts are locked")
    if config["evaluation"]["practical_threshold_pp"] is not None:
        raise ValueError("no practical effect threshold has been approved")
    if completion.get("actual_data") is not (config["profile"] == "full") or completion.get("profile") != config["profile"]:
        raise ValueError("DEBUG and actual citation scope must remain separate")
    if completion.get("config_digest") != digest(config):
        raise ValueError("configuration provenance differs")
    source = completion.get("source", {})
    if not source.get("sha256") or source.get("code_digest") != digest(source["sha256"]):
        raise ValueError("scientific source provenance differs")
    for value in (*source["sha256"].values(), completion.get("data_manifest_digest")):
        _sha(value)
    tuning = len(datasets) * len(CONDITIONS) * len(training["learning_rate_candidates"]) * len(training["tuning_seeds"])
    final = len(datasets) * len(CONDITIONS) * len(seeds)
    budget = {"tuning_runs": tuning, "final_runs": final, "total_runs": tuning + final,
              "total_updates": (tuning + final) * training["epochs_per_run"]}
    if any(training[k] != v for k, v in budget.items()):
        raise ValueError("complete run/update budget differs")
    coverage = completion.get("coverage", {})
    required = {**{k: budget[k] for k in ("tuning_runs", "final_runs", "total_runs")}, "contract_optimizer_updates": budget["total_updates"]}
    if any(coverage.get(k) != v for k, v in required.items()) or coverage.get("all_datasets_conditions_seeds_splits") is not True:
        raise ValueError("completion training coverage differs")
    metrics, probes, branches = (evaluated[k] for k in ("metric_rows", "intervention_rows", "branch_rows"))
    originals = _unique(metrics, ("dataset", "condition", "seed", "split"), "original")
    _coverage(originals, itertools.product(datasets, CONDITIONS, seeds, SPLITS), "original")
    treatments = lambda c: tuple(itertools.product(intervention_variants(parse_condition(c)[3]), intervention_scopes(c)))
    probe_index = _unique(probes, ("dataset", "condition", "seed", "split", "intervention", "target"), "intervention")
    _coverage(probe_index, {(d,c,s,p,i,t) for d,c,s,p in itertools.product(datasets,CONDITIONS,seeds,SPLITS)
                            for i,t in treatments(c)}, "intervention")
    branch_index = _unique(branches, ("dataset", "condition", "seed", "layer", "intervention", "target"), "branch")
    _coverage(branch_index, {(d,c,s,l,i,t) for d,c,s,l in itertools.product(datasets,CONDITIONS,seeds,(0,1))
                             for i,t in (("original","none"), *treatments(c))}, "branch")
    chosen = _unique(selections, ("dataset", "condition"), "selection")
    _coverage(chosen, itertools.product(datasets, CONDITIONS), "selection")
    for row in selections:
        _identity(row)
        if row.get("selection_scope") != "validation_only_independent_tuning_seeds" or row.get("selected_lr") not in training["learning_rate_candidates"]:
            raise ValueError("validation-only selection contract differs")
        if _finite(row["mean_tuning_validation_ce"], "tuning CE") < 0:
            raise ValueError("invalid tuning CE")
    for row in (*metrics, *probes):
        _identity(row); _sha(row.get("model_state_sha256"))
        if _finite(row["ce"], "CE") < 0 or not 0 <= _finite(row["accuracy"], "accuracy") <= 1:
            raise ValueError("invalid classification metric")
        shape = config["data"]["expected_shapes"][row["dataset"]]
        if row["num_nodes"] != shape["nodes"] or row["num_labeled_nodes"] != shape[row["split"]]:
            raise ValueError("complete node and split counts differ")
        if "intervention" in row:
            ref = originals[row["dataset"],row["condition"],row["seed"],row["split"]]
            if row["model_state_sha256"] != ref["model_state_sha256"] or row["no_op_expected"] is not (row["intervention"] == "no_op"):
                raise ValueError("frozen hash/no-op label differs")
            if row.get("coefficient_policy") != "recompute_from_current_hidden":
                raise ValueError("frozen adaptive coefficient policy differs")
            for k in ("logit_delta_norm", "logit_reference_norm", "prediction_flip_fraction"):
                if _finite(row[k], k) < 0:
                    raise ValueError("negative frozen change diagnostic")
            if row["prediction_flip_fraction"] > 1:
                raise ValueError("invalid prediction flip fraction")
            ratio = row["logit_delta_relative"]
            if row["logit_reference_norm"] > 0:
                if not math.isclose(_finite(ratio, "logit relative change"), row["logit_delta_norm"]/row["logit_reference_norm"], rel_tol=2e-6, abs_tol=1e-12):
                    raise ValueError("frozen ratio does not use original logits")
            elif ratio is not None:
                raise ValueError("zero reference norm has undefined ratio")
    counts = completion.get("parameter_counts", {})
    for row in branches:
        _identity(row)
        if row.get("no_op_expected") is not (row["intervention"] == "no_op"):
            raise ValueError("branch no-op label differs")
        for key in BRANCH_METRICS:
            _finite(row.get(key), key, nullable=key not in ("projected_norm", "output_norm"))
        if any(row[k] is not None and row[k] < 0 for k in BRANCH_METRICS if k not in ("energy_cross", "pair_mean")):
            raise ValueError("negative norm, standard deviation or positive energy")
        if row["operator_family"] == "edge_metric":
            if any(row[k] is None for k in ("off_norm","matched_delta_norm","energy_intra","energy_cross","energy_total","message_diag_norm","message_cross_norm")):
                raise ValueError("actual edge-metric diagnostics are missing")
            if type(row["off_nonzero"]) is not bool or row["off_nonzero"] is not (row["off_norm"] > 0):
                raise ValueError("branch denominator flag differs")
            if row["off_nonzero"]:
                if not math.isclose(_finite(row["matched_delta_relative"], "matched ratio"), row["matched_delta_norm"]/row["off_norm"], rel_tol=3e-6, abs_tol=1e-12):
                    raise ValueError("matched ratio does not use actual off norm")
            elif row["matched_delta_relative"] is not None:
                raise ValueError("zero off norm has undefined ratio")
            if not math.isclose(row["energy_total"], row["energy_intra"]+row["energy_cross"], rel_tol=3e-5, abs_tol=1e-9):
                raise ValueError("energy terms do not sum to the applied metric")
            if row["diagonal_mean"] is not None and not .5-1e-6 <= row["diagonal_mean"] <= 2+1e-6:
                raise ValueError("diagonal multiplier bound differs")
            if row["pair_mean"] is not None and not -1-1e-6 <= row["pair_mean"] <= 1+1e-6:
                raise ValueError("raw pair coefficient bound differs")
        else:
            if any(row[k] is not None for k in BRANCH_METRICS if k not in ("projected_norm", "output_norm")) or row["off_nonzero"] is not None:
                raise ValueError("external baselines must not invent edge-metric measurements")
        expected = expected_parameters(config, row["dataset"], row["condition"])
        if counts.get(row["dataset"],{}).get(row["condition"]) != expected or row["parameters_per_seed"] != expected or row["trainable_parameters_per_seed"] != expected:
            raise ValueError("active model parameter count differs")
        if row["model_state_sha256"] != originals[row["dataset"],row["condition"],row["seed"],"test"]["model_state_sha256"]:
            raise ValueError("branch state hash differs")
    if not resources or any(row.get("status") not in ("measured","reused_complete","out_of_memory","OOM","memory_safety_rejected") for row in resources):
        raise ValueError("measured resource coverage is required")
    preserved = set()
    for proof in evaluated.get("provenance", []):
        _identity(proof); _sha(proof.get("before_sha256"))
        if proof.get("before_sha256") != proof.get("after_sha256") or proof.get("parameters_preserved") is not True or proof.get("optimizer_updates") != 0:
            raise ValueError("frozen model preservation failed")
        if proof.get("no_op_checks") != 3 or _finite(proof.get("no_op_absolute_error_max"), "no-op error") < 0:
            raise ValueError("all no-op controls must be measured")
        for seed in proof["seeds"]:
            key = proof["dataset"],proof["condition"],seed
            if key in preserved or proof["before_sha256"] != originals[*key,"test"]["model_state_sha256"]:
                raise ValueError("frozen provenance does not match metrics")
            preserved.add(key)
    _coverage(preserved, itertools.product(datasets, CONDITIONS, seeds), "frozen provenance")
    return {**budget, "primary_metric_rows": len(metrics), "intervention_metric_rows": len(probes),
            "branch_rows": len(branches), "direct_contrasts": len(comparisons()),
            "primary_accuracy_contrasts": 18, "complete_all_declared_rows": True,
            "scope": "exploratory_fixed_public_split_initialization_seed_variation" if config["profile"] == "full" else "DEBUG_fixture_pipeline_only"}


def metric_estimates(metrics):
    groups = defaultdict(list)
    for row in metrics:
        groups[row["dataset"],row["condition"],row["split"]].append(row)
    result = []
    for (dataset,condition,split), rows in sorted(groups.items()):
        acc, ce = estimate([100*r["accuracy"] for r in rows]), estimate([r["ce"] for r in rows])
        result.append({"dataset":dataset,"condition":condition,"split":split,**condition_metadata(condition),
                       "seed_count":len(rows),"accuracy_mean_percent":acc["mean"],"accuracy_std_pp":acc["std"],
                       "ce_mean":ce["mean"],"ce_std":ce["std"]})
    return result


def paired_comparisons(metrics):
    indexed = _unique(metrics,("dataset","condition","seed","split"),"paired metrics")
    datasets, seeds = sorted({r["dataset"] for r in metrics}), sorted({r["seed"] for r in metrics})
    result = []
    for dataset in datasets:
        for condition,reference,contrast,primary in comparisons():
            for split, (metric,scale,unit) in itertools.product(SPLITS, (("accuracy",100,"percentage_points"),("ce",1,"loss"))):
                differences = [scale*(indexed[dataset,condition,s,split][metric]-indexed[dataset,reference,s,split][metric]) for s in seeds]
                selected = primary and split == "test" and metric == "accuracy"
                result.append({"dataset":dataset,"condition":condition,"reference":reference,"contrast":contrast,
                               "split":split,"metric":metric,"unit":unit,"primary_accuracy_test":selected,
                               "comparison_scope":"retrained_validation_selected_condition_minus_reference",
                               "inference_scope":"primary_Holm18" if selected else "exploratory_unadjusted",
                               "seed_ids":",".join(map(str,seeds)),**paired_test(differences),
                               "p_holm":None,"holm_reject":None,"multiplicity_family_size":18 if selected else None})
    primary_rows = [row for row in result if row["primary_accuracy_test"]]
    if len(primary_rows) != 18:
        raise ValueError("all eighteen primary accuracy contrasts are required for Holm")
    for row, adjusted in zip(primary_rows,holm([r["p_value"] for r in primary_rows]),strict=True):
        row.update(p_holm=adjusted,holm_reject=adjusted <= .05)
    return result


def intervention_changes(metrics, probes):
    indexed = _unique(metrics,("dataset","condition","seed","split"),"frozen originals")
    groups = defaultdict(list)
    for row in probes:
        groups[row["dataset"],row["condition"],row["split"],row["intervention"],row["target"]].append(row)
    result = []
    for (dataset,condition,split,treatment,target),rows in sorted(groups.items()):
        for metric,scale,unit in (("accuracy",100,"percentage_points"),("ce",1,"loss")):
            differences = [scale*(r[metric]-indexed[dataset,condition,r["seed"],split][metric]) for r in rows]
            result.append({"dataset":dataset,"condition":condition,"split":split,**condition_metadata(condition),
                           "intervention":treatment,"target":target,"metric":metric,"unit":unit,
                           "comparison_scope":"frozen_selected_checkpoint_minus_original",
                           "inference_scope":"exploratory_unadjusted", "no_op_expected":treatment=="no_op",**paired_test(differences)})
    return result


def branch_estimates(branches):
    groups = defaultdict(list)
    for row in branches:
        groups[row["dataset"],row["condition"],row["layer"],row["intervention"],row["target"]].append(row)
    result = []
    for (dataset,condition,layer,treatment,target),rows in sorted(groups.items()):
        value = {"dataset":dataset,"condition":condition,"layer":layer,**condition_metadata(condition),
                 "intervention":treatment,"target":target,"seed_count":len(rows),"no_op_expected":treatment=="no_op"}
        for key in BRANCH_METRICS:
            stats = estimate([r[key] for r in rows if r[key] is not None])
            value.update({f"{key}_{field}":stats[field] for field in ("mean","std","count")})
        result.append(value)
    return result


def _number(value):
    return "undefined" if value is None else f"{value:.6g}" if isinstance(value,float) else str(value)


def _table(rows, fields):
    return "\n".join(["| "+" | ".join(fields)+" |","| "+" | ".join("---" for _ in fields)+" |",
                      *("| "+" | ".join(_number(row[key]) for key in fields)+" |" for row in rows)])


def _figures(output, config, estimates, paired, branches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(exist_ok=True)
    datasets = config["data"]["datasets"]
    fig, axes = plt.subplots(2,3,figsize=(17,9),layout="constrained",squeeze=False)
    primary_pairs = [(c,r) for c,r,_,p in comparisons() if p]
    for col,dataset in enumerate(datasets):
        for line,metric in enumerate(("accuracy","ce")):
            rows = {(r["condition"],r["reference"]):r for r in paired if r["dataset"]==dataset and r["split"]=="test" and r["metric"]==metric}
            selected = [rows[key] for key in primary_pairs]
            means = np.asarray([r["mean"] for r in selected])
            errors = np.asarray([[r["mean"]-r["lower"] for r in selected],[r["upper"]-r["mean"] for r in selected]])
            ax = axes[line,col]
            ax.errorbar(means,np.arange(len(selected)),xerr=errors,fmt="o",capsize=3,color="#20639b")
            labels = [r["condition"].replace("local_degree","degree").replace("__"," ")+" - "+r["reference"].replace("local_degree","degree").replace("__"," ") for r in selected]
            ax.set_yticks(np.arange(len(selected)),labels,fontsize=8); ax.invert_yaxis(); ax.axvline(0,color="black",lw=.7)
            ax.grid(axis="x",alpha=.2); ax.set_title(dataset)
            ax.set_xlabel("Test accuracy difference (pp); Holm18 p-values in CSV" if metric=="accuracy" else "Test CE difference; exploratory unadjusted")
    fig.suptitle(config["profile"].upper()+" | Retrained contrasts | Individual paired 95% t intervals")
    for suffix in ("png","pdf"): fig.savefig(folder/f"primary_contrasts.{suffix}",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,9),layout="constrained",squeeze=False)
    for col,dataset in enumerate(datasets):
        lookup={r["condition"]:r for r in estimates if r["dataset"]==dataset and r["split"]=="test"}
        ax=axes[0,col]; rows=[lookup[c] for c in CONDITIONS]
        ax.errorbar([r["accuracy_mean_percent"] for r in rows],np.arange(len(rows)),xerr=[r["accuracy_std_pp"] for r in rows],fmt="o",capsize=3)
        ax.set_yticks(np.arange(len(rows)),[c.replace("local_degree","degree").replace("__"," ") for c in CONDITIONS],fontsize=8)
        ax.invert_yaxis(); ax.set_title(dataset); ax.set_xlabel("Test accuracy (%); sample std across seeds"); ax.grid(axis="x",alpha=.2)
    fig.suptitle(config["profile"].upper()+" | All declared selected models")
    for suffix in ("png","pdf"): fig.savefig(folder/f"classification.{suffix}",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout="constrained",squeeze=False)
    conditions=[c for c in CONDITIONS if "__" in c]
    for col,dataset in enumerate(datasets):
        lookup={(r["condition"],r["layer"]):r for r in branches if r["dataset"]==dataset and r["intervention"]=="original"}
        for line,key in enumerate(("matched_delta_relative_mean","energy_cross_mean")):
            ax=axes[line,col]
            for layer,color in ((0,"#20639b"),(1,"#cf6a21")):
                values=[lookup[c,layer][key] for c in conditions]
                ax.plot(values,np.arange(len(values)),"o-",label=f"layer {layer}",color=color)
            ax.set_yticks(np.arange(len(conditions)),[c.replace("local_degree","degree").replace("__"," ") for c in conditions],fontsize=7)
            ax.invert_yaxis(); ax.set_title(dataset); ax.grid(axis="x",alpha=.2); ax.legend(fontsize=8)
            ax.set_xlabel("Matched projected-state off-diagonal change / off norm" if line==0 else "Signed cross energy of applied edge metric")
    fig.suptitle(config["profile"].upper()+" | Original checkpoints | DA has no off-diagonal action")
    for suffix in ("png","pdf"): fig.savefig(folder/f"branch_use.{suffix}",dpi=170)
    plt.close(fig)


def write_report(output, config, evaluated, selections, resources, completion):
    output=Path(output)
    if any((output/name).exists() for name in ARTIFACTS):
        raise FileExistsError("report artifacts already exist; use a new result directory")
    checks=validate_rows(config,evaluated,selections,resources,completion)
    metrics=metric_estimates(evaluated["metric_rows"]); paired=paired_comparisons(evaluated["metric_rows"])
    frozen=intervention_changes(evaluated["metric_rows"],evaluated["intervention_rows"]); branches=branch_estimates(evaluated["branch_rows"])
    for name,rows in (("metric_estimates.csv",metrics),("paired_comparisons.csv",paired),("intervention_changes.csv",frozen),("branch_estimates.csv",branches)):
        write_csv(output/name,rows)
    checks.update(primary_holm_tests=sum(r["primary_accuracy_test"] for r in paired),metric_estimate_rows=len(metrics),
                  paired_comparison_rows=len(paired),intervention_estimate_rows=len(frozen),branch_estimate_rows=len(branches))
    write_json(output/"report_checks.json",checks)
    selected={(r["dataset"],r["condition"]):r for r in selections}
    main=[{"데이터":r["dataset"],"조건":r["condition"],"LR":selected[r["dataset"],r["condition"]]["selected_lr"],
           "Test accuracy %":f"{_number(r['accuracy_mean_percent'])} ± {_number(r['accuracy_std_pp'])}",
           "Test CE":f"{_number(r['ce_mean'])} ± {_number(r['ce_std'])}"} for r in metrics if r["split"]=="test"]
    primary=[{"데이터":r["dataset"],"조건":r["condition"],"기준":r["reference"],"Δaccuracy pp":r["mean"],
              "개별 95% CI":f"[{_number(r['lower'])}, {_number(r['upper'])}]","양측 p":r["p_value"],
              "Holm18 p":r["p_holm"],"Holm 기각":r["holm_reject"]} for r in paired if r["primary_accuracy_test"]]
    ce_rows=[{"데이터":r["dataset"],"조건":r["condition"],"기준":r["reference"],"ΔCE":r["mean"],
              "개별 95% CI":f"[{_number(r['lower'])}, {_number(r['upper'])}]"} for r in paired
             if r["split"]=="test" and r["metric"]=="ce" and r["contrast"] in ("F2-D1","F2-DA","F2-P2")]
    text=f"""# 엣지 metric: 분류 실험 결과

Profile **{config['profile'].upper()}**, 실제 citation 데이터 **{completion['actual_data']}**.
2층, hidden {config['backbone']['hidden_dim']}, run당 {config['training']['epochs_per_run']} epoch.
Tuning {checks['tuning_runs']}회와 final {checks['final_runs']}회, 총 {checks['total_runs']}회,
계약상 독립 모델 갱신 {checks['total_updates']:,}회다. 재개 실행의 새 갱신은 `new_optimizer_updates`로 따로 기록한다.
DEBUG는 명시된 fixture의 구현 검사이며 실제 citation 성능이 아니다.

## 실제 모델과 대조

모든 물리 노드·엣지·induced 1홉 로컬과 선언한 eligible pair를 사용한다.
Pair는 인접 중심의 서로 다른 로컬에서 서로 다른 물리 엣지가 공통 끝점을 가지는 경우다.
로컬 엣지 occurrence는 `1/sqrt(r_e)`로 보정한다.
`Ccal=sqrt(D)(I+.5K)sqrt(D)`, `Q=BcalᵀCcalBcal`이며 각 층은 `Z−(1/3)N0 Q(Z) N0Z`다.
게이트는 현재 Z를 읽고 전체 feature message를 유지한다. 학습 loss는 분류 CE다.
입력 의존 에너지의 전체 gradient를 전파한다는 주장은 하지 않는다.

D0는 고정 diagonal, D1은 learned diagonal, F0는 고정 pair,
F1은 learned pair, F2는 learned diagonal과 pair를 함께 사용한다.
DA는 같은 pair 입력·MLP를 normalized row sum과 occurrence 평균을 통해 diagonal에만 사용한다.
F2와 DA의 parameter·입력 접근은 같지만 함수 공간·작용 크기·gradient 규모가 같다는 뜻은 아니다.
두 C recipe를 모두 보고한다. G1은 표준 self-loop GCN, G2는 각 층의 GCN 제곱,
P2는 구간 sup로 정규화한 학습 가능한 2차 다항식이다. 과거 결과를 새 비교표에 섞지 않았다.

## 분류 결과

평균 ± 표본 std다. Accuracy는 %, CE는 L2를 제외한 평가 손실이다.
LR와 checkpoint를 validation으로 선택하고 모든 final 선택을 잠근 뒤 test를 평가했다.

{_table(main,tuple(main[0]))}

## 사전 지정한 accuracy 비교 18개

각 recipe·dataset에서 F2−D1, F2−DA, F2−P2다. 값은 조건−기준이며 accuracy 양수가 개선이다.
동일 final seed의 차이를 먼저 계산했다. 개별 CI는 paired 95% t 구간이고,
양측 p에 18개 전체의 Holm 보정을 적용한다. CI 자체는 동시 구간으로 보정하지 않았다.
별도 실용 차이 기준은 승인되지 않아 사용하지 않는다. CI의 0 포함은 동등성 증명이 아니다.

{_table(primary,tuple(primary[0]))}

## 같은 대비의 보조 CE

CE 음수가 개선이다. 아래 CE와 그 밖의 모든 보조 대비는 탐색적이며 다중 비교 보정을 적용하지 않았다.
F1−D0, F1−F0, F2−F1, 모든 recipe/조건의 G1·G2 대조를 포함한 전체 split 결과는 CSV에 있다.

{_table(ce_rows,tuple(ce_rows[0]))}

## Frozen 개입과 실제 작용

조건마다 적용 가능한 `offdiag_zero`, `diagonal_one`, `pair_zero`와 `no_op`을
layer_0/layer_1/both에서 평가했다. 체크포인트와 buffer hash를 보존하며 optimizer 갱신은 0회다.
개입 뒤 후속 hidden과 adaptive gate는 다시 계산한다. 원래 수치 계수를 고정한 연산자 감사와 구분한다.
모든 no-op의 출력 잔차도 기록했다. 원시 CE·accuracy·logit 변화·prediction flip은 `interventions.csv`,
같은 checkpoint 대비 차이는 `intervention_changes.csv`에 있다.

Branch 에너지는 실제 `Ccal`의 내부·교차·총 에너지이며 cross 항은 음수일 수 있다.
`matched_delta_relative`는 같은 projected Z와 실제 diagonal을 유지한 off-diagonal 제거의 출력 차이다.
DA는 이 비대각 차이가 0이어도 pair 정보를 diagonal에 사용할 수 있으므로 `pair_zero`로 따로 검사한다.
분모 0과 baseline에 해당하지 않는 에너지는 undefined다. 값을 0으로 대체하지 않았다.

## 전체 범위와 해석

- 원시 metric {checks['primary_metric_rows']:,}행, frozen metric {checks['intervention_metric_rows']:,}행,
  branch {checks['branch_rows']:,}행 및 모든 상태 hash의 범위를 확인했다.
- Sparse 연산·parameter gradient·gate update·정보 접근·CE·accuracy는 별도의 판단이다.
- 고정 계수의 PSD와 전파 norm 상계는 adaptive Jacobian 또는 전체 모델의 비팽창 보장이 아니다.
- 정확 복원이나 새로운 그래프 일반화는 이번 public-split 분류만으로 입증하지 않는다.
- 이전 test를 본 후 설계한 후속 탐색이며 seed 변동은 새 graph/split의 불확실성이 아니다.

원시 checkpoint·학습 gradient/update·source/data manifest·resource·completion과 CSV를 함께 보관한다.
"""
    with (output/"EDGE_METRIC_CLASSIFICATION_SUMMARY.md").open("x",encoding="utf-8") as stream:
        stream.write(text)
    _figures(output,config,metrics,paired,branches)
    return checks
