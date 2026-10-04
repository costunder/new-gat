"""All normalization contrasts, paired seed interactions and frozen probes."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...wedge_propagation.classification.common import digest, write_csv, write_json
from ..classification.report import estimate
from .common import CONDITIONS, CROSS_POLICIES, INTRA_POLICIES, SCOPES, WEIGHTS, condition_metadata, parse_condition

SPLITS = ("train", "validation", "test")
TARGET_LAYERS = {"layer_0": (0,), "layer_1": (1,), "both": (0, 1)}
BRANCH_METRICS = (
    "projected_norm", "output_norm", "off_norm", "matched_delta_norm",
    "matched_delta_relative", "context_norm", "cross_energy_before",
    "cross_energy_after", "intra_energy_before", "intra_energy_after",
    "rho", "rho_original", "theta", "predicted_delta_norm", "formula_residual_norm",
    "formula_relative_error", "raw_context_norm", "s_degree_max", "g_degree_max", "eta_max",
)
FIGURES = ("cross_effects", "cross_normalization", "intra_normalization", "interactions", "frozen_effects", "branch_use")
ARTIFACTS = (
    "LOCAL_CONTEXT_NORMALIZATION_SUMMARY.md", "metric_estimates.csv",
    "paired_comparisons.csv", "interaction_comparisons.csv",
    "intervention_changes.csv", "branch_estimates.csv", "report_checks.json",
    *(f"figures/{name}.{suffix}" for name in FIGURES for suffix in ("png", "pdf")),
)


def condition_id(weight, intra, cross, variant):
    return f"{weight}__{intra}__{cross}__{variant}"


def metadata(condition):
    return condition_metadata(condition)


def legacy_baseline(condition):
    _, intra, cross, variant = parse_condition(condition)
    return intra == "graph" and (cross, variant) in (("none", "off"), ("graph", "fixed"), ("graph", "learned"))


def comparisons():
    """52 predeclared direct contrasts; no test-selected comparison subset."""
    result = []
    for weight in WEIGHTS:
        for intra in INTRA_POLICIES:
            off = condition_id(weight, intra, "none", "off")
            for cross in CROSS_POLICIES:
                fixed, learned = (condition_id(weight, intra, cross, variant) for variant in ("fixed", "learned"))
                result.extend(((fixed, off, "cross_fixed_minus_off"),
                               (learned, off, "cross_learned_minus_off"),
                               (learned, fixed, "gain_learned_minus_fixed")))
            for variant in ("fixed", "learned"):
                result.append((condition_id(weight, intra, "edge", variant),
                               condition_id(weight, intra, "graph", variant), "cross_edge_minus_graph"))
        for cross, variant in (("none", "off"), *((k, v) for k in CROSS_POLICIES for v in ("fixed", "learned"))):
            result.append((condition_id(weight, "local", cross, variant),
                           condition_id(weight, "graph", cross, variant), "intra_local_minus_graph"))
    for intra in INTRA_POLICIES:
        for cross, variant in (("none", "off"), *((k, v) for k in CROSS_POLICIES for v in ("fixed", "learned"))):
            result.append((condition_id("local_degree", intra, cross, variant),
                           condition_id("unit", intra, cross, variant), "between_C_exploratory"))
    return tuple(result)


def interaction_contrasts():
    """12 four-condition contrasts, calculated within each paired seed."""
    result = []
    for weight in WEIGHTS:
        for cross in CROSS_POLICIES:
            for variant in ("fixed", "learned"):
                terms = (
                    (condition_id(weight, "local", cross, variant), 1),
                    (condition_id(weight, "local", "none", "off"), -1),
                    (condition_id(weight, "graph", cross, variant), -1),
                    (condition_id(weight, "graph", "none", "off"), 1),
                )
                result.append(dict(contrast_id=f"{weight}__{cross}__{variant}__intra_cross_contribution",
                                   comparison_scope="intra_change_in_cross_minus_off", terms=terms))
        for variant in ("fixed", "learned"):
            terms = (
                (condition_id(weight, "local", "edge", variant), 1),
                (condition_id(weight, "local", "graph", variant), -1),
                (condition_id(weight, "graph", "edge", variant), -1),
                (condition_id(weight, "graph", "graph", variant), 1),
            )
            result.append(dict(contrast_id=f"{weight}__{variant}__intra_cross_normalization_interaction",
                               comparison_scope="intra_change_in_edge_minus_graph", terms=terms))
    return tuple(result)


def intervention_variants(condition):
    return () if parse_condition(condition)[3] == "off" else tuple(
        (treatment, target) for treatment in ("gain0", "gain1") for target in SCOPES
    )


def _finite(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; undefined measurements use None")
    return float(value)


def _identity(row, *, energy=False):
    expected = metadata(row["condition"])
    fields = ("weight_mode", "intra_policy", "cross_policy", "variant")
    if any(row.get(key) != expected[key] for key in fields):
        raise ValueError("condition must match C, intra, cross and gain metadata")
    if "mode" in row and row["mode"] != expected["weight_mode"]:
        raise ValueError("legacy mode alias must match weight_mode")
    if energy and row.get("energy_operator") != "applied_S_G":
        raise ValueError("branch energies must refer to applied S/G operators")
    if energy and row.get("cross_diagnostic_policy") != expected["cross_diagnostic_policy"]:
        raise ValueError("off uses explicit graph G diagnostic reference")


def _unique(rows, fields, name):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError(f"duplicate {name} row")
        indexed[key] = row
    return indexed


def _coverage(actual, expected, name):
    if set(actual) != set(expected):
        raise ValueError(f"incomplete {name} coverage: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}")


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be SHA256")


def validate_rows(config, evaluated, selections, resources, completion):
    if tuple(config.get("conditions", ())) != CONDITIONS or config.get("profile") not in ("full", "debug"):
        raise ValueError("explicit profile and all 20 unique conditions required")
    datasets, seeds = config["data"]["datasets"], config["training"]["final_seeds"]
    if not datasets or len(datasets) != len(set(datasets)) or len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("distinct datasets and at least two distinct final seeds required")
    if config["profile"] == "full" and set(datasets) != {"Cora", "CiteSeer", "PubMed"}:
        raise ValueError("FULL requires all three complete citation datasets")
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
    required = {key: budget[key] for key in ("tuning_runs", "final_runs", "total_runs")}
    required["contract_optimizer_updates"] = budget["total_updates"]
    if any(coverage.get(key) != value for key, value in required.items()) or coverage.get("all_datasets_conditions_seeds_splits") is not True:
        raise ValueError("completion training coverage mismatch")
    metrics, probes, branches = (evaluated[key] for key in ("metric_rows", "intervention_rows", "branch_rows"))
    originals = _unique(metrics, ("dataset", "condition", "seed", "split"), "primary")
    _coverage(originals, {(d, c, s, p) for d in datasets for c in CONDITIONS for s in seeds for p in SPLITS}, "primary")
    intervened = _unique(probes, ("dataset", "condition", "seed", "split", "intervention", "target"), "intervention")
    _coverage(intervened, {(d, c, s, p, i, target) for d in datasets for c in CONDITIONS for s in seeds
                          for p in SPLITS for i, target in intervention_variants(c)}, "intervention")
    branch_index = _unique(branches, ("dataset", "condition", "seed", "layer", "intervention", "target"), "branch")
    _coverage(branch_index, {(d, c, s, layer, i, target) for d in datasets for c in CONDITIONS for s in seeds
                            for layer in (0, 1) for i, target in (("original", "none"), *intervention_variants(c))}, "branch")
    chosen = _unique(selections, ("dataset", "condition"), "selection")
    _coverage(chosen, {(d, c) for d in datasets for c in CONDITIONS}, "selection")
    for row in selections:
        _identity(row)
        if row.get("selection_scope") != "validation_only_independent_tuning_seeds":
            raise ValueError("selection must use independent tuning validation only")
        if row.get("selected_lr") not in training["learning_rate_candidates"]:
            raise ValueError("selected LR is outside declared candidates")
        if _finite(row["mean_tuning_validation_ce"], "tuning CE") < 0:
            raise ValueError("tuning CE must be nonnegative")
    for row in (*metrics, *probes):
        _identity(row)
        if _finite(row["ce"], "CE") < 0 or not 0 <= _finite(row["accuracy"], "accuracy fraction") <= 1:
            raise ValueError("classification CE/accuracy range invalid")
        shape = config["data"]["expected_shapes"][row["dataset"]]
        if row.get("num_nodes") != shape["nodes"] or row.get("num_labeled_nodes") != shape[row["split"]]:
            raise ValueError("reported counts must match complete input and split")
        _sha(row.get("model_state_sha256"), "model state")
        if "intervention" in row:
            original = originals[row["dataset"], row["condition"], row["seed"], row["split"]]
            if row["model_state_sha256"] != original["model_state_sha256"]:
                raise ValueError("frozen intervention changed model state")
            noop = row["variant"] == "fixed" and row["intervention"] == "gain1"
            if row.get("no_op_expected") is not noop:
                raise ValueError("frozen no-op control label mismatch")
    counts = completion.get("parameter_counts", {})
    for row in branches:
        _identity(row, energy=True)
        if (row.get("theta_available") is not (row["variant"] == "learned")
                or type(row.get("off_nonzero")) is not bool or type(row.get("projected_nonzero")) is not bool):
            raise ValueError("branch parameter/denominator flags must be explicit")
        for key in BRANCH_METRICS:
            _finite(row.get(key), key, nullable=key in ("theta", "matched_delta_relative", "formula_relative_error"))
        if (row["theta"] is None) != (row["variant"] != "learned"):
            raise ValueError("theta is present only in learned conditions")
        if any(row[key] < 0 for key in BRANCH_METRICS if key not in ("theta", "matched_delta_relative", "formula_relative_error")):
            raise ValueError("norms, applied energies and gains must be nonnegative")
        if not 0 <= row["rho"] <= 1 or not 0 <= row["rho_original"] <= 1:
            raise ValueError("cross gain must be in [0,1]")
        original_gain = {"off": 0, "fixed": 1}.get(row["variant"])
        if original_gain is not None and row["rho_original"] != original_gain:
            raise ValueError("fixed/off original gain mismatch")
        active = row["intervention"] != "original" and row["layer"] in TARGET_LAYERS[row["target"]]
        expected_gain = int(row["intervention"] == "gain1") if active else row["rho_original"]
        if row["rho"] != expected_gain:
            raise ValueError("effective gain does not match intervention layers")
        if row["off_nonzero"] is not (row["off_norm"] > 0):
            raise ValueError("off_nonzero must match off norm")
        if row["off_nonzero"]:
            if row["matched_delta_relative"] is None or not math.isclose(row["matched_delta_relative"], row["matched_delta_norm"] / row["off_norm"], rel_tol=3e-6, abs_tol=1e-12):
                raise ValueError("matched ratio must use actual off norm")
        elif row["matched_delta_relative"] is not None:
            raise ValueError("zero-denominator ratio must be undefined")
        if row["projected_nonzero"] is not (row["projected_norm"] > 0):
            raise ValueError("projected_nonzero must match projected norm")
        if row["projected_nonzero"]:
            if row["formula_relative_error"] is None or not math.isclose(
                    row["formula_relative_error"], row["formula_residual_norm"] / row["projected_norm"],
                    rel_tol=3e-6, abs_tol=1e-12):
                raise ValueError("formula ratio must use actual projected norm")
        elif row["formula_relative_error"] is not None:
            raise ValueError("zero projected norm makes formula ratio undefined")
        if row["s_degree_max"] > .5 + 1e-5 or row["g_degree_max"] > .5 + 1e-5:
            raise ValueError("applied S/G degree exceeds contraction bound")
        shape = config["data"]["expected_shapes"][row["dataset"]]
        expected_parameters = shape["features"] * config["backbone"]["hidden_dim"] + config["backbone"]["hidden_dim"] * shape["classes"] + int(row["variant"] == "learned")
        if counts.get(row["dataset"], {}).get(row["condition"]) != expected_parameters:
            raise ValueError("completion parameter count mismatch")
        if row.get("parameters_per_seed") != expected_parameters or row.get("trainable_parameters_per_seed") != expected_parameters:
            raise ValueError("active parameter count mismatch")
        original = originals[row["dataset"], row["condition"], row["seed"], "test"]
        if row.get("model_state_sha256") != original["model_state_sha256"]:
            raise ValueError("branch model state mismatch")
        noop = row["variant"] == "fixed" and row["intervention"] == "gain1"
        if row.get("no_op_expected") is not noop:
            raise ValueError("branch no-op label mismatch")
    if not isinstance(resources, list) or not resources:
        raise ValueError("measured resource rows required")
    for row in resources:
        if row.get("status") not in ("measured", "reused_complete", "out_of_memory", "OOM", "memory_safety_rejected"):
            raise ValueError("resource status must preserve measurement/failure/reuse")
        if row.get("seconds_per_epoch") is not None and _finite(row["seconds_per_epoch"], "epoch seconds") < 0:
            raise ValueError("epoch time must be nonnegative")
    proofs = evaluated.get("provenance")
    if not isinstance(proofs, list) or not proofs:
        raise ValueError("frozen model preservation proofs required")
    preserved = set()
    for proof in proofs:
        _identity(proof)
        if proof.get("parameters_preserved") is not True or proof.get("optimizer_updates") != 0:
            raise ValueError("frozen preservation changed model or optimizer")
        _sha(proof.get("before_sha256"), "before frozen model")
        if proof.get("before_sha256") != proof.get("after_sha256"):
            raise ValueError("before/after frozen hashes differ")
        for seed in proof["seeds"]:
            key = (proof["dataset"], proof["condition"], seed)
            if key in preserved:
                raise ValueError("duplicate frozen preservation coverage")
            preserved.add(key)
            if proof["before_sha256"] != originals[*key, "test"]["model_state_sha256"]:
                raise ValueError("frozen proof does not match metric state")
    _coverage(preserved, {(d, c, s) for d in datasets for c in CONDITIONS for s in seeds}, "frozen preservation")
    return {**budget, "primary_metric_rows": len(metrics), "intervention_metric_rows": len(probes),
            "branch_rows": len(branches), "direct_contrasts": len(comparisons()),
            "interaction_contrasts": len(interaction_contrasts()), "complete_all_declared_rows": True,
            "energy_operator": "applied_S_G",
            "scope": "same_fixed_public_split_initialization_seed_variation" if config["profile"] == "full" else "DEBUG_fixture_pipeline_only"}


def metric_estimates(metrics):
    groups = defaultdict(list)
    for row in metrics:
        groups[row["dataset"], row["condition"], row["split"]].append(row)
    result = []
    for (dataset, condition, split), rows in sorted(groups.items()):
        acc, ce = estimate([100 * r["accuracy"] for r in rows]), estimate([r["ce"] for r in rows])
        result.append(dict(dataset=dataset, condition=condition, split=split, **metadata(condition),
                           legacy_baseline=legacy_baseline(condition), seed_count=len(rows),
                           accuracy_mean_percent=acc["mean"], accuracy_std_pp=acc["std"],
                           ce_mean=ce["mean"], ce_std=ce["std"]))
    return result


def _axes(metrics):
    return sorted({r["dataset"] for r in metrics}), sorted({r["seed"] for r in metrics})


def paired_comparisons(metrics):
    indexed = _unique(metrics, ("dataset", "condition", "seed", "split"), "paired primary")
    datasets, seeds = _axes(metrics)
    result = []
    for dataset in datasets:
        for condition, reference, scope in comparisons():
            for split in SPLITS:
                for metric, scale, unit in (("accuracy", 100, "percentage_points"), ("ce", 1, "loss")):
                    differences = [scale * (indexed[dataset, condition, seed, split][metric] - indexed[dataset, reference, seed, split][metric]) for seed in seeds]
                    result.append(dict(dataset=dataset, condition=condition, reference=reference, split=split,
                                       comparison_scope=scope, metric=metric, unit=unit, **estimate(differences)))
    return result


def interaction_comparisons(metrics):
    indexed = _unique(metrics, ("dataset", "condition", "seed", "split"), "interaction primary")
    datasets, seeds = _axes(metrics)
    result = []
    for dataset in datasets:
        for contrast in interaction_contrasts():
            terms = contrast["terms"]
            expression = " ".join(("+" if coefficient > 0 else "-") + condition for condition, coefficient in terms)
            for split in SPLITS:
                for metric, scale, unit in (("accuracy", 100, "percentage_points"), ("ce", 1, "loss")):
                    differences = [scale * sum(coefficient * indexed[dataset, condition, seed, split][metric] for condition, coefficient in terms) for seed in seeds]
                    result.append(dict(dataset=dataset, contrast_id=contrast["contrast_id"], split=split,
                                       comparison_scope=contrast["comparison_scope"], contrast_expression=expression,
                                       metric=metric, unit=unit, **estimate(differences)))
    return result


def intervention_changes(metrics, probes):
    originals = _unique(metrics, ("dataset", "condition", "seed", "split"), "intervention reference")
    groups = defaultdict(list)
    for row in probes:
        groups[row["dataset"], row["condition"], row["split"], row["intervention"], row["target"]].append(row)
    result = []
    for (dataset, condition, split, treatment, target), rows in sorted(groups.items()):
        for metric, scale, unit in (("accuracy", 100, "percentage_points"), ("ce", 1, "loss")):
            differences = [scale * (r[metric] - originals[dataset, condition, r["seed"], split][metric]) for r in rows]
            result.append(dict(dataset=dataset, condition=condition, split=split, **metadata(condition),
                               intervention=treatment, target=target, metric=metric, unit=unit,
                               comparison_scope="frozen_selected_checkpoint_minus_original",
                               no_op_expected=rows[0]["no_op_expected"], **estimate(differences)))
    return result


def branch_estimates(branches):
    groups = defaultdict(list)
    for row in branches:
        groups[row["dataset"], row["condition"], row["layer"], row["intervention"], row["target"]].append(row)
    result = []
    for (dataset, condition, layer, treatment, target), rows in sorted(groups.items()):
        value = dict(dataset=dataset, condition=condition, layer=layer, **metadata(condition),
                     intervention=treatment, target=target, seed_count=len(rows),
                     theta_available=rows[0]["theta_available"], no_op_expected=rows[0]["no_op_expected"],
                     off_nonzero_seeds=sum(r["off_nonzero"] for r in rows))
        for metric in BRANCH_METRICS:
            measured = estimate([r[metric] for r in rows if r[metric] is not None])
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


def _figures(output, config, paired, interactions, frozen, branches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    datasets = config["data"]["datasets"]
    families = {
        "cross_effects": ("cross_fixed_minus_off", "cross_learned_minus_off", "gain_learned_minus_fixed"),
        "cross_normalization": ("cross_edge_minus_graph",),
        "intra_normalization": ("intra_local_minus_graph",),
        "interactions": None,
    }
    for name, scopes in families.items():
        selected = interactions if scopes is None else [r for r in paired if r["comparison_scope"] in scopes]
        first = [r for r in selected if r["dataset"] == datasets[0] and r["split"] == "test" and r["metric"] == "accuracy"]
        keys = [r["contrast_id"] if scopes is None else (r["condition"], r["reference"]) for r in first]
        labels = [r["contrast_id"] if scopes is None else r["condition"] + " − " + r["reference"] for r in first]
        labels = [label.replace("local_degree", "degree").replace("__", " ") for label in labels]
        fig, axes = plt.subplots(2, len(datasets), figsize=(6.0 * len(datasets), max(9, .48 * len(first) + 3)),
                                 layout="constrained", squeeze=False)
        for col, dataset in enumerate(datasets):
            for line, metric in enumerate(("accuracy", "ce")):
                axis = axes[line, col]
                rows = [r for r in selected if r["dataset"] == dataset and r["split"] == "test" and r["metric"] == metric]
                indexed = {r["contrast_id"] if scopes is None else (r["condition"], r["reference"]): r for r in rows}
                ordered = [indexed[key] for key in keys]
                means = np.asarray([r["mean"] for r in ordered])
                errors = np.asarray([[r["mean"] - r["lower"] for r in ordered], [r["upper"] - r["mean"] for r in ordered]])
                axis.barh(np.arange(len(keys)), means, xerr=errors, capsize=2, color="#2171b5")
                axis.set_yticks(np.arange(len(keys)), labels, fontsize=7)
                axis.invert_yaxis()
                axis.axvline(0, color="black", linewidth=.8)
                axis.grid(axis="x", alpha=.2)
                axis.set_title(dataset)
                axis.set_xlabel("Accuracy difference (pp); higher is better" if metric == "accuracy" else "CE difference; lower is better")
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
                if metric == "ce":
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-3, 3))
                    axis.xaxis.set_major_formatter(formatter)
        fig.suptitle(f"{config['profile'].upper()} · {name.replace('_', ' ')} · paired seed 95% t intervals")
        for suffix in ("png", "pdf"):
            fig.savefig(folder / f"{name}.{suffix}", dpi=160)
        plt.close(fig)
    active = [c for c in CONDITIONS if parse_condition(c)[3] != "off"]
    treatments = intervention_variants(active[0])
    limit = max(abs(r["mean"]) for r in frozen if r["split"] == "test" and r["metric"] == "ce") or 1e-12
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 9), layout="constrained", squeeze=False)
    for col, dataset in enumerate(datasets):
        axis = axes[0, col]
        lookup = {(r["condition"], r["intervention"], r["target"]): r["mean"] for r in frozen
                  if r["dataset"] == dataset and r["split"] == "test" and r["metric"] == "ce"}
        values = [[lookup[c, i, target] for i, target in treatments] for c in active]
        im = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_yticks(range(len(active)), [c.replace("local_degree", "degree").replace("__", " ") for c in active], fontsize=8)
        axis.set_xticks(range(6), [i + " " + target.replace("layer_", "L") for i, target in treatments], rotation=40, ha="right", fontsize=8)
        axis.set_title(dataset)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Frozen test ΔCE; negative is better", shrink=.8)
    fig.suptitle(f"{config['profile'].upper()} · same-checkpoint gain probes · fixed gain1 is a no-op")
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"frozen_effects.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, len(datasets), figsize=(6 * len(datasets), 17), layout="constrained", squeeze=False)
    labels = [c.replace("local_degree", "degree").replace("__", " ") for c in CONDITIONS]
    for col, dataset in enumerate(datasets):
        lookup = {(r["condition"], r["layer"]): r for r in branches if r["dataset"] == dataset and r["intervention"] == "original"}
        for line, (key, title) in enumerate((("rho_mean", "Shared original gain"),
                                            ("matched_delta_relative_mean", "Actual matched Δ / off norm"),
                                            ("context_norm_mean", "Applied context norm ||G Y1||"))):
            axis = axes[line, col]
            for layer, color in ((0, "#225ea8"), (1, "#b35806")):
                values = [lookup[c, layer][key] for c in CONDITIONS]
                axis.plot(values, np.arange(len(CONDITIONS)), "o-", color=color, label=f"layer {layer}")
            axis.set_yticks(np.arange(len(CONDITIONS)), labels, fontsize=7)
            axis.invert_yaxis()
            axis.set_xlabel(title)
            axis.set_title(dataset)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            axis.xaxis.set_major_formatter(formatter)
            axis.grid(axis="x", alpha=.2)
            axis.legend(fontsize=7)
    fig.suptitle(f"{config['profile'].upper()} · original selected checkpoints · applied S/G diagnostics")
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"branch_use.{suffix}", dpi=160)
    plt.close(fig)


def write_report(output, config, evaluated, selections, resources, completion):
    output = Path(output)
    if any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError("report output already exists; use a new result directory")
    checks = validate_rows(config, evaluated, selections, resources, completion)
    metrics = metric_estimates(evaluated["metric_rows"])
    paired = paired_comparisons(evaluated["metric_rows"])
    interactions = interaction_comparisons(evaluated["metric_rows"])
    frozen = intervention_changes(evaluated["metric_rows"], evaluated["intervention_rows"])
    branches = branch_estimates(evaluated["branch_rows"])
    for name, rows in (("metric_estimates.csv", metrics), ("paired_comparisons.csv", paired),
                       ("interaction_comparisons.csv", interactions), ("intervention_changes.csv", frozen),
                       ("branch_estimates.csv", branches)):
        write_csv(output / name, rows)
    checks.update(metric_estimate_rows=len(metrics), paired_comparison_rows=len(paired),
                  interaction_comparison_rows=len(interactions), intervention_estimate_rows=len(frozen),
                  branch_estimate_rows=len(branches))
    write_json(output / "report_checks.json", checks)
    mindex = {(r["dataset"], r["condition"], r["split"]): r for r in metrics}
    selection = {(r["dataset"], r["condition"]): r for r in selections}
    main = []
    for dataset in config["data"]["datasets"]:
        for condition in CONDITIONS:
            rows = {split: mindex[dataset, condition, split] for split in SPLITS}
            main.append({"데이터": dataset, "조건": condition, "기존 6조건": legacy_baseline(condition),
                         "LR": selection[dataset, condition]["selected_lr"],
                         "Train acc %": _meanstd(rows["train"]["accuracy_mean_percent"], rows["train"]["accuracy_std_pp"]),
                         "Val acc %": _meanstd(rows["validation"]["accuracy_mean_percent"], rows["validation"]["accuracy_std_pp"]),
                         "Test acc %": _meanstd(rows["test"]["accuracy_mean_percent"], rows["test"]["accuracy_std_pp"]),
                         "Test CE": _meanstd(rows["test"]["ce_mean"], rows["test"]["ce_std"])})
    contrasts = [{"데이터": r["dataset"], "조건": r["condition"], "기준": r["reference"],
                  "지표": r["metric"], "단위": r["unit"], "차이 [95% 구간]": _interval(r), "비교": r["comparison_scope"]}
                 for r in paired if r["split"] == "test"]
    joint = [{"데이터": r["dataset"], "상호작용": r["contrast_id"], "지표": r["metric"],
              "단위": r["unit"], "차이 [95% 구간]": _interval(r)} for r in interactions if r["split"] == "test"]
    observed = [{"데이터": r["dataset"], "조건": r["condition"], "층": r["layer"],
                 "gain": _meanstd(r["rho_mean"], r["rho_std"]),
                 "실제 Δ/off": _meanstd(r["matched_delta_relative_mean"], r["matched_delta_relative_std"]),
                 "||G Y1||": r["context_norm_mean"], "G 에너지 전": r["cross_energy_before_mean"],
                 "G 에너지 후": r["cross_energy_after_mean"]} for r in branches if r["intervention"] == "original"]
    text = f"""# 로컬 문맥 결합: 정규화 비교 결과

Profile **{config['profile'].upper()}**, 실제 citation 데이터 **{completion['actual_data']}**.
2개 macro 층, hidden dimension {config['backbone']['hidden_dim']}, 학습마다 {config['training']['epochs_per_run']} epoch.
Tuning {checks['tuning_runs']}회와 final {checks['final_runs']}회, 총 {checks['total_runs']}회 학습이며
계약상 독립 모델 갱신은 {checks['total_updates']:,}회다. 재개 실행의 새 갱신 횟수는 `new_optimizer_updates`로 따로 확인한다.
DEBUG 결과는 별도 fixture의 구현 검사이며 citation 본학습 성능이 아니다.

## 바꾼 것과 비교 기준

조건 ID는 C__intra__cross__variant다. C는 unit/local_degree,
intra는 graph/local, 활성 cross는 graph/edge다. Off에는 none__off 한 조건만 둔다.
각 층은 `M(I−S)(I−ρG)(I−S)RZ`이며 같은 S를 앞뒤에 적용한다.
Graph intra는 기존 그래프 전체 degree bound, local intra는 각 ego의 degree bound를 쓴다.
Graph G는 기존 γK, edge G는 고정 unit cross degree에 따른 대칭 엣지 가중치를 쓴다.
고정 gain은 1, learned gain은 독립 모델마다 θ 하나를 두 층에서 공유하는 sigmoid(θ)다.
기존 6조건을 이번 계약에서도 처음부터 튜닝·학습했다. 과거 수치를 새 결과에 끼워 넣지 않았다.

## 분류 결과

평균 ± 표본 std다. Accuracy는 %, CE는 L2 항을 제외한 평가 손실이다.
LR와 checkpoint는 validation으로 선택하고 모든 final 선택을 고정한 뒤 test를 평가했다.

{_table(main, tuple(main[0]))}

## 직접 비교 52개

24개 cross/gain 비교, 8개 edge−graph cross 정규화 비교, 10개 local−graph 내부 정규화 비교,
10개 C 비교를 모두 기록한다. C 비교는 탐색적이다. 아래 값은 조건−기준이다.
Accuracy의 양수와 CE의 음수가 개선이다. 같은 final seed끼리 차이를 먼저 계산한 paired 95% t 구간이다.
조건별 LR 선택을 포함한 학습 절차의 비교이며, 학습된 파라미터를 고정한 순수 연산 비교와 구분한다.

{_table(contrasts, tuple(contrasts[0]))}

## 정규화 상호작용 12개

8개는 `(local cross−local off)−(graph cross−graph off)`이며,
4개는 `(local edge−local graph)−(graph edge−graph graph)`다.
각 seed에서 네 조건의 contrast를 계산한 뒤 통계를 낸다. 이미 집계한 오차막대를 빼서 구간을 만들지 않는다.
수식의 cross 경로는 `−ρ M S G S R`지만, 분류 지표의 이 차이를 경로별 분류 기여율로 해석하지 않는다.

{_table(joint, tuple(joint[0]))}

## 같은 checkpoint의 frozen 개입과 실제 작용

활성 cross 조건의 gain0/gain1 × layer_0/layer_1/both를 모두 평가한다.
Fixed gain1은 예상 no-op 대조다. Off에는 별도 개입을 만들지 않는다.
모델·buffer hash를 보존하고 optimizer 갱신은 0회다. 첫 층 개입이면 후속 특징을 다시 계산한다.
원시 값과 같은 checkpoint 대비 차이는 `interventions.csv`와 `intervention_changes.csv`에 있다.

Branch 에너지는 **applied S/G**의 ½trace다. 원시 A/K 에너지와 같은 값으로 읽지 않는다.
Off의 G 에너지와 context norm은 graph G에 대한 참고 진단이며 gain 0으로 실제 교차 경로는 꺼져 있다.
실제 Δ/off는 현재 층의 동일 Z에 대한 `||TρZ−T0Z||/||T0Z||`다. 분모 0은 undefined다.
Gain 학습, 실제 출력 변화, 분류 개선을 각각 판단한다.

{_table(observed, tuple(observed[0]))}

## 범위와 한계

- 원시 metric {checks['primary_metric_rows']:,}행, frozen {checks['intervention_metric_rows']:,}행,
  branch {checks['branch_rows']:,}행의 전체 범위와 hash를 검증했다.
- Source 그래프 {config['source']['graphs']}개를 보존한다. 선언된 citation 또는 DEBUG 그래프만 분류에 사용하며 합성 그래프는 label 학습에 쓰지 않는다.
- 자원 기록 {len(resources)}행에서 calibration·본학습·완료된 학습의 재사용을 구분한다.
- Degree bound는 copy 공간 및 물리 D metric의 contraction을 보장한다. 일반 Euclidean node norm이나 모든 종류의 그래프 에너지 감소를 보장하는 주장이 아니다.
- Star처럼 local C 변화가 ego step과 정확히 상쇄되는 구조가 있다. C 조건이 다르다는 사실만으로 적용된 S가 다르다고 판단하지 않는다.
- Public split의 seed {len(config['training']['final_seeds'])}개에 따른 변동이며 독립 split/새 그래프 일반화가 아니다.
- 이전 test를 확인한 후 설계한 탐색 연구다. 다중 비교 보정을 적용하지 않았고 0 포함 구간은 동등성 증명이 아니다.
- 에너지는 전파를 정의한다. 학습 loss는 분류 CE이며 정보 복원이나 신규성은 이 실험만으로 입증하지 않는다.

전체 split의 요약은 CSV, 원시 증거는 metric·intervention·branch·frozen provenance·source/data hash·resource·completion에서 확인한다.
"""
    with (output / "LOCAL_CONTEXT_NORMALIZATION_SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(text)
    _figures(output, config, paired, interactions, frozen, branches)
    return checks
