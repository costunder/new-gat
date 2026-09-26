"""Within-dataset/seed contrasts; never substitute external baselines for ablations."""

from __future__ import annotations

import copy
import math


def _recipe(result):
    config = copy.deepcopy(result["comparison_configuration"])
    for key in ("ablation_arm", "conductance_mode", "conductance_heads"):
        config.pop(key, None)
    for key in ("conductance", "c_control_lift"):
        config["comparison_contract"].pop(key, None)
    return config


def _contrasts():
    definitions = []

    def add(name, weights, interpretation):
        definitions.append((name, weights, interpretation))

    lifts = ("", "_linear_lift", "_pre_lift", "_post_lift")
    energies = ("", "_diagonal", "_energy")
    for lift in lifts:
        base, diagonal, full = ("incidence" + energy + lift for energy in energies)
        for label, positive, negative in (
            ("diagonal_vs_none", diagonal, base),
            ("cross_added_to_diagonal", full, diagonal),
            ("all_energy_vs_none", full, base),
        ):
            add(
                label + (lift or "_no_lift"),
                {positive: 1, negative: -1},
                "per-head dynamic C; same lift",
            )
    for energy in energies:
        base, linear, pre, post = ("incidence" + energy + lift for lift in lifts)
        for label, positive, negative in (
            ("linear_vs_none", linear, base),
            ("pre_vs_linear", pre, linear),
            ("pre_vs_post", pre, post),
            ("post_vs_linear", post, linear),
        ):
            add(
                label + (energy or "_no_energy"),
                {positive: 1, negative: -1},
                "same energy configuration; parameter counts reported, not assumed equal",
            )
    for low, high, label in (
        ("", "_diagonal", "diagonal"),
        ("_diagonal", "_energy", "cross"),
        ("", "_energy", "all_energy"),
    ):
        for lift in lifts[1:]:
            add(
                "interaction_" + label + lift,
                {
                    "incidence" + high + lift: 1,
                    "incidence" + high: -1,
                    "incidence" + low + lift: -1,
                    "incidence" + low: 1,
                },
                "difference in energy effect between this lift and no lift",
            )
    regimes = {"fixed": "incidence_fixed", "shared": "incidence_shared", "per_head": "incidence"}
    for name, arm in regimes.items():
        add(
            "energy_effect_c_" + name,
            {arm + "_energy": 1, arm: -1},
            "no lift; retrained energy-on minus energy-off",
        )
    # Existing per-head energy arm is incidence_energy; the same naming rule applies.
    for low, high in (("fixed", "shared"), ("fixed", "per_head"), ("shared", "per_head")):
        a, b = regimes[low], regimes[high]
        for suffix in ("", "_energy"):
            add(
                f"c_{high}_vs_{low}" + (suffix or "_no_energy"),
                {b + suffix: 1, a + suffix: -1},
                "no lift; fixed/shared/per-head training control",
            )
        add(
            f"interaction_energy_c_{high}_vs_{low}",
            {b + "_energy": 1, b: -1, a + "_energy": -1, a: 1},
            "no lift; difference in energy effect across conductance regimes",
        )
    return definitions


def contrast_report(jobs):
    groups = {}
    for job in jobs:
        key = (job["profile"], job["dataset"], job["model_seed"])
        group = groups.setdefault(key, {})
        arm = job["variant_id"]
        if arm in group:
            raise ValueError(f"duplicate result cell: {key}/{arm}")
        group[arm] = job
    report = []
    for (profile, dataset, seed), group in sorted(groups.items()):
        rows = []
        for name, weights, interpretation in _contrasts():
            missing = [
                arm
                for arm in weights
                if arm not in group
                or group[arm]["status"] != "passed"
                or group[arm].get("audit", {}).get("status") != "passed"
            ]
            row = {"effect": name, "weights": weights, "interpretation": interpretation}
            if missing:
                rows.append({**row, "status": "incomplete", "missing_or_unaudited": missing})
                continue
            results = {arm: group[arm]["result"] for arm in weights}
            first = next(iter(results.values()))
            for result in results.values():
                for field in (
                    "data_sha256",
                    "split_sha256",
                    "learning_budget",
                    "source_sha256",
                    "shared_initial_state_sha256",
                ):
                    if not first.get(field) or result.get(field) != first[field]:
                        raise ValueError(f"contrast {name}: unmatched {field}")
                if _recipe(result) != _recipe(first):
                    raise ValueError(f"contrast {name}: unmatched training recipe")
                score = result.get("validation")
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError(f"contrast {name}: invalid score")
            rows.append(
                {
                    **row,
                    "status": "available",
                    "delta_pp": 100
                    * sum(weights[arm] * r["validation"] for arm, r in results.items()),
                    "scores": {arm: r["validation"] for arm, r in results.items()},
                    "parameters": {arm: r["total_parameters"] for arm, r in results.items()},
                }
            )
        report.append({"profile": profile, "dataset": dataset, "model_seed": seed, "effects": rows})
    return {
        "scope": "paired within dataset/profile/seed; validation percentage-point differences; "
        "no significance or multi-seed claim",
        "groups": report,
    }


def markdown(report):
    lines = [
        "",
        "## Paired ablation effects",
        "",
        report["scope"],
        "",
        "| Profile | Dataset | Seed | Effect | Delta (pp) |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for group in report["groups"]:
        available = [row for row in group["effects"] if row["status"] == "available"]
        for row in available:
            lines.append(
                f"| {group['profile']} | {group['dataset']} | {group['model_seed']} | "
                f"{row['effect']} | {row['delta_pp']:+.6f} |"
            )
        missing = len(group["effects"]) - len(available)
        lines.extend(
            [
                "",
                f"{group['profile']}/{group['dataset']}/seed-{group['model_seed']}: "
                f"{missing} contrasts incomplete; missing cells are listed in effects.json.",
                "",
            ]
        )
    return lines
