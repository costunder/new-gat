"""Explicit arithmetic DEBUG fixtures for the saved-CSV summary renderer."""

import copy

import pytest

from research.wedge_propagation.branch_analysis.report import (
    FIXED_METRICS,
    LEARNED,
    STRENGTH_METRICS,
    SUMMARY,
    TARGETS,
    TREATMENTS,
    write_report,
)


def report_fixture():
    """Complete declared groups; these measurements are test arithmetic only."""
    config = {
        "profile": "debug",
        "data": {"datasets": ["DEBUG-arithmetic"]},
        "conditions": list(LEARNED),
        "final_seeds": [11, 23],
        "treatments": list(TREATMENTS),
        "targets": list(TARGETS),
    }
    provenance = {
        "source_config": {"training": {"epochs_per_run": 3}},
        "optimizer_updates": 0,
        "csv_only": True,
        "source_files_preserved": True,
    }
    baseline, strength, changes, fixed = [], [], [], []

    def estimate(value, metric):
        return {
            "metric": metric,
            "mean": value,
            "std": 0.125,
            "lower": value - 1.0,
            "upper": value + 1.0,
            "count": 2,
            "undefined_seeds": 0,
            "undefined_observations": 0,
        }

    values = dict.fromkeys(STRENGTH_METRICS, 0.5)
    values.update(
        {
            "alpha": 0.4,
            "beta": 0.2,
            "kappa_reference": 4.0,
            "beta_over_kappa": 0.05,
            "raw_branch_to_input": 0.8,
            "unscaled_branch_to_input": 0.2,
            "beta_t_to_input": 0.04,
            "alpha_l_to_input": 0.2,
            "beta_t_to_alpha_l": 0.2,
            "first_second_cosine": -0.75,
        }
    )
    for condition in LEARNED:
        group = {"dataset": "DEBUG-arithmetic", "condition": condition}
        for layer in (0, 1):
            for seed in config["final_seeds"]:
                baseline.append({**group, "seed": seed, "layer": layer, **values})
            for metric in STRENGTH_METRICS:
                strength.append(
                    {
                        **group,
                        "layer": layer,
                        "treatment": "baseline",
                        "target": "none",
                        **estimate(values[metric], metric),
                    }
                )
            for treatment in TREATMENTS:
                for metric in FIXED_METRICS:
                    fixed.append(
                        {
                            **group,
                            "layer": layer,
                            "treatment": treatment,
                            "target": f"layer_{layer}",
                            **estimate(0.25, metric),
                        }
                    )
        for split in ("train", "validation", "test"):
            for target_index, target in enumerate(TARGETS):
                for treatment in TREATMENTS[1:]:
                    for metric in ("accuracy", "ce"):
                        # Preserve the accuracy/CE tradeoff and distinct intervention scopes.
                        value = (target_index + 1) * (0.01 if metric == "accuracy" else 0.03)
                        if treatment == "branch_off":
                            value = -value
                        changes.append(
                            {
                                **group,
                                "split": split,
                                "treatment": treatment,
                                "target": target,
                                **estimate(value, metric),
                            }
                        )
    return (
        config,
        {
            "baseline_strength": baseline,
            "strength_estimates": strength,
            "layer_changes": changes,
            "fixed_estimates": fixed,
        },
        provenance,
    )


def test_summary_preserves_factorization_real_signs_scopes_and_debug_contract(tmp_path):
    config, analysis, provenance = report_fixture()
    assert write_report(tmp_path, config, analysis, provenance) == {"summary": SUMMARY}
    summary = (tmp_path / SUMMARY).read_text(encoding="utf-8")
    assert "DEBUG · 기존 seed [11, 23] · 원래 학습 3 epoch/run · 새 update 0회" in summary
    assert "각 seed에서 `βM/Z = (β/κ) × (R/Z)`" in summary
    assert "평균끼리 곱한 값은 실제 평균 βM/Z와 다를 수 있다" in summary
    assert "0.04 ± 0.125" in summary
    assert "-0.75 ± 0.125" in summary
    assert "| 1 / 0.03 | 2 / 0.06 | 3 / 0.09 |" in summary
    assert "| -1 / -0.03 | -2 / -0.06 | -3 / -0.09 |" in summary
    assert "노드 ID는 평균하지 않았다" in summary
    assert "실제 citation 본학습 성능으로 제출하지 않는다" in summary
    assert "새 forward·학습·GPU 평가·개입 선택은 없다" in summary


def test_undefined_estimate_is_explicit_not_a_zero(tmp_path):
    config, analysis, provenance = report_fixture()
    chosen = next(
        row for row in analysis["strength_estimates"] if row["metric"] == "first_second_cosine"
    )
    chosen.update(
        {
            "mean": None,
            "std": None,
            "lower": None,
            "upper": None,
            "count": 0,
            "undefined_seeds": 2,
            "undefined_observations": 2,
        }
    )
    write_report(tmp_path, config, analysis, provenance)
    summary = (tmp_path / SUMMARY).read_text(encoding="utf-8")
    assert "undefined ± undefined (n=0; u=2)" in summary


@pytest.mark.parametrize(
    "table",
    [
        "baseline_strength",
        "strength_estimates",
        "layer_changes",
        "fixed_estimates",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_incomplete_or_duplicate_observations_fail_before_writing(tmp_path, table, mutation):
    config, analysis, provenance = report_fixture()
    if mutation == "missing":
        analysis[table].pop()
    else:
        analysis[table].append(copy.deepcopy(analysis[table][0]))
    with pytest.raises(ValueError, match="coverage|duplicate"):
        write_report(tmp_path, config, analysis, provenance)
    assert not (tmp_path / SUMMARY).exists()


@pytest.mark.parametrize("table", ["strength_estimates", "layer_changes", "fixed_estimates"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", float("nan")),
        ("std", float("inf")),
        ("std", -1),
        ("count", True),
        ("count", 3),
        ("mean", None),
        ("lower", 9),
        ("upper", -9),
    ],
)
def test_invalid_statistics_are_not_rendered(tmp_path, table, field, value):
    config, analysis, provenance = report_fixture()
    analysis[table][0][field] = value
    with pytest.raises(ValueError):
        write_report(tmp_path, config, analysis, provenance)
    assert not (tmp_path / SUMMARY).exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_updates", 1),
        ("csv_only", False),
        ("source_files_preserved", False),
    ],
)
def test_report_requires_csv_only_frozen_provenance(tmp_path, field, value):
    config, analysis, provenance = report_fixture()
    provenance[field] = value
    with pytest.raises(ValueError, match="CSV-only"):
        write_report(tmp_path, config, analysis, provenance)


def test_report_does_not_overwrite_existing_summary(tmp_path):
    config, analysis, provenance = report_fixture()
    path = tmp_path / SUMMARY
    path.write_text("existing result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_report(tmp_path, config, analysis, provenance)
    assert path.read_text(encoding="utf-8") == "existing result"


def test_full_sized_table_stays_compact_without_debug_claims(tmp_path):
    config, analysis, provenance = report_fixture()
    config["profile"] = "full"
    config["data"]["datasets"] = ["Cora", "CiteSeer", "PubMed"]
    original = copy.deepcopy(analysis)
    for table, rows in original.items():
        analysis[table] = [
            dict(row, dataset=dataset) for dataset in config["data"]["datasets"] for row in rows
        ]
    write_report(tmp_path, config, analysis, provenance)
    summary = (tmp_path / SUMMARY).read_text(encoding="utf-8")
    assert len(summary.splitlines()) < 100
    assert "**FULL" in summary
    assert "DEBUG fixture" not in summary
    assert summary.count("| Cora |") == 12


def test_undefined_counts_cannot_hide_partial_observations(tmp_path):
    config, analysis, provenance = report_fixture()
    analysis["fixed_estimates"][0]["undefined_seeds"] = 1
    with pytest.raises(ValueError, match="undefined"):
        write_report(tmp_path, config, analysis, provenance)
