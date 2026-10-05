"""Scalar summaries of exact frozen E/J fields; one seed has no seed interval."""

from __future__ import annotations

import torch

ENERGIES = (
    "E",
    "E_sym",
    "E_sym_augmented",
    "E_scale_free",
    "E_sym_scale_free",
    "E_sym_augmented_scale_free",
    "E_topoood_1hop",
    "E_topoood_1hop_scale_free",
)
RELATIONS = ("J_shared", "J_distinct", "J_node")


def retention(current, reference):
    current, reference = current.double().flatten(), reference.double().flatten()
    if current.shape != reference.shape:
        raise ValueError("energy fields must share identical node/pair ordering")
    if current.numel() == 0:
        return {"count": 0, "magnitude_ratio": None, "pattern_cosine": None}
    if not bool(torch.isfinite(current).all() & torch.isfinite(reference).all()):
        raise ValueError("nonfinite values on the declared valid support")
    a, b = current.norm(), reference.norm()
    return {
        "count": current.numel(),
        "magnitude_ratio": float(a / b) if bool(b > 0) else None,
        "pattern_cosine": float(torch.dot(current, reference) / (a * b))
        if bool((a > 0) & (b > 0))
        else None,
    }


def distribution(value):
    value = value.double().flatten()
    if not value.numel():
        return {
            "count": 0,
            "mean": None,
            "l2": None,
            "minimum": None,
            "maximum": None,
            "p10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "p90": None,
            "IQR": None,
        }
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite measured field")
    quantiles = value.quantile(value.new_tensor([0.1, 0.25, 0.5, 0.75, 0.9])).tolist()
    return {
        "count": value.numel(),
        "mean": float(value.mean()),
        "l2": float(value.norm()),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        **dict(zip(("p10", "q25", "median", "q75", "p90"), quantiles, strict=True)),
        "IQR": quantiles[3] - quantiles[1],
    }


def summarize(measured, reference, pairs, labels):
    known = (labels[pairs[0]] >= 0) & (labels[pairs[1]] >= 0)
    same = known & (labels[pairs[0]] == labels[pairs[1]])
    different = known & ~same
    same_values = measured["J_distinct_normalized"][same]
    different_values = measured["J_distinct_normalized"][different]
    row = {
        "adjacent_local_pair_count": pairs.shape[1],
        "known_label_pair_count": int(known.sum()),
        "same_label_pair_count": int(same.sum()),
        "different_label_pair_count": int(different.sum()),
        "zero_local_energy_count": int((measured["E"] == 0).sum()),
        "zero_J_energy_product_count": int(measured["zero_J_denominator"].sum()),
        "delta_J_label": float(same_values.mean() - different_values.mean())
        if same_values.numel() and different_values.numel()
        else None,
        "label_same_distinct_normalized": distribution(same_values),
        "label_different_distinct_normalized": distribution(different_values),
        "label_analysis": "secondary post-hoc only; no checkpoint selection",
        "global_energy": {
            key: float(value) for key, value in measured.items() if key.startswith("global_")
        },
    }
    for key in (*ENERGIES, *RELATIONS, *(key + "_normalized" for key in RELATIONS)):
        value = measured[key]
        baseline = reference[key]
        row[key] = distribution(value)
        row[key + "_input_retention"] = retention(value, baseline)
    return row


def transition(before, after, name):
    row = {"transition": name}
    for key in (*ENERGIES, *RELATIONS, *(key + "_normalized" for key in RELATIONS)):
        a, b = before[key], after[key]
        row[key + "_delta"] = distribution(b - a)
        row[key + "_retention"] = retention(b, a)
        if key.startswith("E"):
            valid = a > 0
            row[key + "_operation_ratio"] = distribution(b[valid] / a[valid])
            row[key + "_undefined_ratio_count"] = int((~valid).sum())
        if key.endswith("_normalized"):
            start = float(a.abs().mean()) if a.numel() else None
            stop = float(b.abs().mean()) if b.numel() else None
            row[key + "_mean_abs_ratio"] = stop / (start + 1e-12) if start is not None else None
            row[key + "_zero_reference_mean_abs"] = start == 0 if start is not None else None
    row["J_ratio_origin"] = "project extension, not a standard literature metric"
    for key in before:
        if key.startswith("global_"):
            a, b = float(before[key]), float(after[key])
            row[key + "_operation_ratio"] = b / a if a > 0 else None
    return row


def flat_rows(records):
    rows = []

    def visit(prefix, value, row):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}_{key}" if prefix else key, item, row)
        else:
            row[prefix] = value

    for record in records:
        row = {}
        visit("", record, row)
        rows.append(row)
    return rows


def operation_fields(before, after):
    """Persist every node ratio, including an explicit mask for undefined zero inputs."""
    fields = {}
    for key in ENERGIES:
        a, b = before[key], after[key]
        valid = a > 0
        ratio = torch.full_like(a, torch.nan)
        ratio[valid] = b[valid] / a[valid]
        fields[key + "_ratio"] = ratio
        fields[key + "_ratio_defined"] = valid
    for key in RELATIONS:
        fields[key + "_delta"] = after[key] - before[key]
        fields[key + "_normalized_delta"] = after[key + "_normalized"] - before[key + "_normalized"]
    return fields


def write_report(path, stages, transitions):
    def number(value):
        return "undefined" if value is None else f"{value:.6g}"

    lines = [
        "# Frozen local-energy trace — seed 11 only",
        "",
        "Existing selected MLP/GCN/Polynomial-2 checkpoints; no new training, "
        "optimizer update or LR/checkpoint selection. All graph nodes and physical edges are used.",
        "",
        "Primary comparisons use aggregation-input Z versus aggregation-output A. "
        "Energy magnitude change alone does not establish information loss. "
        "J_shared/J_node/J_distinct use the audited signed local-incidence decomposition. "
        "Degree-normalized DE still depends on feature amplitude; explicit norm quotients "
        "are retained separately. J ratios and label separation extend the literature methods.",
        "",
        "| Dataset | Model | Layer | median E ratio Z→A | IQR | "
        "normalized distinct J magnitude ratio | distinct J pattern cosine |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in transitions:
        if row["transition"] != "aggregation":
            continue
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['layer']} | "
            f"{number(row['E_operation_ratio']['median'])} | "
            f"{number(row['E_operation_ratio']['IQR'])} | "
            f"{number(row['J_distinct_normalized_mean_abs_ratio'])} | "
            f"{number(row['J_distinct_retention']['pattern_cosine'])} |"
        )
    lines += [
        "",
        "The replay uses the same Z and common GCN P, with no intervening W or activation. "
        "Polynomial-2's trained operator is separate from this common P² control.",
        "",
        "| Dataset | Z from model | Layer | Replay | median E ratio | "
        "normalized distinct J magnitude ratio | normalized distinct J cosine |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in transitions:
        if not row["transition"].startswith("replay_"):
            continue
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['layer']} | {row['transition']} | "
            f"{number(row['E_operation_ratio']['median'])} | "
            f"{number(row['J_distinct_normalized_mean_abs_ratio'])} | "
            f"{number(row['J_distinct_normalized_retention']['pattern_cosine'])} |"
        )
    lines += [
        "",
        "FT, GP and ReLU are compared on the same checkpoint; MLP's actual GP is I. "
        "A smaller E with stable normalized J supports amplitude/smoothing change, "
        "without demonstrating relation loss. Changes also present under FT or MLP "
        "weaken a propagation-specific explanation. P→P² measures the additional "
        "effect of one repeated propagation. There is no automatic pass/fail threshold.",
        "",
        "| Dataset | Model | Stage | R_E | C_E | R_J_distinct | C_J_distinct | "
        "ΔJ(label) | adjacent pairs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stages:
        e, j = row["E_input_retention"], row["J_distinct_input_retention"]
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['stage']} | "
            f"{number(e['magnitude_ratio'])} | {number(e['pattern_cosine'])} | "
            f"{number(j['magnitude_ratio'])} | {number(j['pattern_cosine'])} | "
            f"{number(row['delta_J_label'])} | {row['adjacent_local_pair_count']} |"
        )
    lines += [
        "",
        "One initialization seed is a descriptive case study. Nodes, directed pairs "
        "and feature channels are not independent model-seed replicates. "
        "No confidence interval, significance or out-of-graph generalization is asserted.",
        "",
        "Undefined zero-reference retention is recorded as null. Zero-energy "
        "denominators are flagged and excluded from node operation ratios. "
        "Unknown labels are excluded only from "
        "label separation. All-label summaries are post-hoc diagnostics and do not select models.",
    ]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
