"""Arithmetic DEBUG report fixtures; no training result is manufactured here."""

import copy

import pytest

from research.wedge_propagation.node_normalization.evaluation import (
    LEARNED,
    intervention_variants,
)
from research.wedge_propagation.node_normalization.report import (
    ARTIFACTS,
    COMPARISONS,
    CONDITIONS,
    STRENGTH_METRICS,
    branch_estimates,
    intervention_changes,
    paired_comparisons,
    write_report,
)


def fixture_rows():
    config = {
        "profile": "debug",
        "data": {"datasets": ["DEBUG-arithmetic"]},
        "conditions": list(CONDITIONS),
        "training": {"final_seeds": [11, 23], "total_runs": 70, "epochs_per_run": 3},
        "evaluation": {"shuffle_manifests_per_dataset": 2},
    }
    metrics, interventions, gates = [], [], []
    for index, condition in enumerate(CONDITIONS):
        for seed_index, seed in enumerate(config["training"]["final_seeds"]):
            base = {"dataset": "DEBUG-arithmetic", "condition": condition, "seed": seed}
            accuracy = 0.8 + seed_index * 0.02 - index * 0.01
            ce = 0.5 + seed_index * 0.1 + index * 0.01
            for split in ("train", "validation", "test"):
                row = {**base, "split": split, "accuracy": accuracy, "ce": ce}
                metrics.append(row)
                if condition in LEARNED:
                    for treatment, target, manifest in intervention_variants(2):
                        change = 0.01 if manifest < 0 else 0.02 if manifest == 0 else -0.04
                        interventions.append(
                            {
                                **row,
                                "intervention": treatment,
                                "target": target,
                                "manifest_index": manifest,
                                "accuracy": accuracy + change,
                                "ce": ce + change,
                            }
                        )
            variants = [("original", "none", -1)]
            if condition in LEARNED:
                variants += list(intervention_variants(2))
            for layer in (0, 1):
                for treatment, target, manifest in variants:
                    scalar = dict.fromkeys(STRENGTH_METRICS, 0.25)
                    scalar.update(
                        kappa=1.0 if "_local_" in condition else 4.0,
                        kappa_global=4.0,
                        first_second_cosine=-0.5,
                    )
                    gates.append(
                        {
                            **base,
                            "layer": layer,
                            "intervention": treatment,
                            "target": target,
                            "manifest_index": manifest,
                            **scalar,
                        }
                    )
    selections = [
        {"dataset": "DEBUG-arithmetic", "condition": condition, "selected_lr": 0.003}
        for condition in CONDITIONS
    ]
    return config, metrics, interventions, [], selections, [], gates, {"actual_data": False}


def test_all_declared_paired_comparisons_include_all_splits_and_real_signs():
    _, metrics, *_ = fixture_rows()
    rows = paired_comparisons(metrics)
    assert len(COMPARISONS) == 8
    assert len(rows) == 48
    assert all(row["count"] == 2 for row in rows)
    row = next(
        row
        for row in rows
        if row["first"] == "learned_wedge_local_raw"
        and row["second"] == "learned_wedge_raw"
        and row["metric"] == "accuracy"
    )
    assert row["mean"] == pytest.approx(-0.02)


def test_shuffle_manifest_average_occurs_within_each_seed():
    _, metrics, treatments, *_ = fixture_rows()
    rows = intervention_changes(treatments, metrics, "accuracy")
    assert len(rows) == 4 * 3 * 3 * 5
    shuffled = [r for r in rows if r["intervention"] == "c_position_shuffle_norm_matched"]
    assert all(r["count"] == 2 for r in shuffled)
    assert all(r["mean"] == pytest.approx(-0.01) for r in shuffled)


def test_report_writes_summary_and_complete_csvs_without_selecting_best_intervention(tmp_path):
    result = write_report(tmp_path, *fixture_rows())
    assert result["paired_rows"] == 48
    assert result["intervention_change_rows"] == 360
    assert result["figures"] == 0
    assert all((tmp_path / name).is_file() for name in ARTIFACTS)
    summary = (tmp_path / ARTIFACTS[0]).read_text(encoding="utf-8")
    assert "| Test acc | Test CE |" in summary
    assert "node/raw − global/raw" in summary
    assert "node/RMS − node/raw" in summary
    assert "-1 / -0.01" in summary
    assert "node의 κ는 작동 분모 1" in summary
    assert "최고 개입을 선택하지 않으며" in summary
    assert "실제 citation 성능으로 제출하지 않는다" in summary
    assert "[metrics.csv](metrics.csv)" in summary
    assert "classification.csv" not in summary


@pytest.mark.parametrize("table_index", [1, 2, 6])
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_report_refuses_missing_and_duplicate_scope_before_writing(tmp_path, table_index, mutation):
    rows = list(fixture_rows())
    if mutation == "missing":
        rows[table_index].pop()
    else:
        rows[table_index].append(copy.deepcopy(rows[table_index][0]))
    with pytest.raises(ValueError, match="duplicate|incomplete"):
        write_report(tmp_path, *rows)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "table_index,key,value",
    [
        (1, "ce", float("nan")),
        (2, "accuracy", 1.1),
        (6, "beta_t_to_input", float("inf")),
    ],
)
def test_nonfinite_or_invalid_observations_are_not_zero_filled(tmp_path, table_index, key, value):
    rows = list(fixture_rows())
    rows[table_index][0][key] = value
    with pytest.raises(ValueError):
        write_report(tmp_path, *rows)
    assert list(tmp_path.iterdir()) == []


def test_undefined_branch_geometry_stays_none_and_records_seed_count(tmp_path):
    rows = list(fixture_rows())
    for row in rows[6]:
        if row["condition"] == "learned_wedge_local_raw" and row["intervention"] == "original":
            row["first_second_cosine"] = None
            row["first_second_cosine_defined"] = False
    estimates = branch_estimates(rows[6])
    chosen = [
        r
        for r in estimates
        if r["condition"] == "learned_wedge_local_raw" and r["metric"] == "first_second_cosine"
    ]
    assert all(r["mean"] is None and r["count"] == 0 and r["undefined_seeds"] == 2 for r in chosen)
    write_report(tmp_path, *rows)
    assert "undefined ± undefined" in (tmp_path / ARTIFACTS[0]).read_text(encoding="utf-8")


def test_report_preserves_existing_artifacts(tmp_path):
    existing = tmp_path / ARTIFACTS[0]
    existing.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_report(tmp_path, *fixture_rows())
    assert existing.read_text(encoding="utf-8") == "existing"


def test_validation_selection_cannot_contain_test_metrics(tmp_path):
    rows = list(fixture_rows())
    rows[4][0]["test_accuracy"] = 0.99
    with pytest.raises(ValueError, match="test"):
        write_report(tmp_path, *rows)


@pytest.mark.parametrize("table_index", [1, 2])
@pytest.mark.parametrize("field,value", [("num_nodes", 5), ("num_labeled_nodes", 1)])
def test_full_graph_and_labeled_counts_cannot_change(tmp_path, table_index, field, value):
    rows = list(fixture_rows())
    rows[0]["data"]["expected_shapes"] = {
        "DEBUG-arithmetic": {"nodes": 6, "train": 2, "validation": 2, "test": 2},
    }
    for table in (rows[1], rows[2]):
        for row in table:
            row.update(num_nodes=6, num_labeled_nodes=2)
    rows[table_index][0][field] = value
    with pytest.raises(ValueError, match="coverage"):
        write_report(tmp_path, *rows)
