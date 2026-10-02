"""Debug arithmetic fixtures test reporting, not normalized model performance."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from PIL import Image

import research.wedge_propagation.scale_normalization.report as report


def _row(
    *,
    variant="raw",
    target="path",
    condition="learned",
    seed=0,
    split="id",
    graph_id="g0",
    scenario="original",
    amplitude=1.0,
    error=0.4,
):
    return {
        "variant": variant,
        "target": target,
        "condition": condition,
        "seed": seed,
        "split": split,
        "graph_id": graph_id,
        "family": "tree",
        "num_nodes": 5,
        "num_edges": 4,
        "num_paths": 3,
        "num_realizations": 2,
        "scenario": scenario,
        "amplitude": amplitude,
        "source_split": split,
        "feature_status": "source_features" if scenario == "original" else "unseen_features",
        "message_relerr": error,
        "message_abs_rmse": 0.0 if error is None else error / 10,
        "weight_relerr": 0.2 if target == "path" and condition == "learned" else None,
        "weight_corr": 0.6 if target == "path" and condition == "learned" else None,
        "weight_mean_error": 0.0 if condition not in report._DETERMINISTIC else None,
        "beta": None,
        "u": None,
        "v": None,
    }


def _scale(row):
    change = abs(math.log2(row["amplitude"])) * (0.1 if row["variant"] == "raw" else 1e-8)
    return {
        **{
            name: row[name]
            for name in (
                "variant",
                "target",
                "condition",
                "seed",
                "split",
                "graph_id",
                "family",
                "amplitude",
            )
        },
        "student_weight_scale_relerr": change
        if row["condition"] not in report._DETERMINISTIC
        else None,
        "teacher_weight_scale_relerr": 1e-9 if row["target"] == "path" else None,
        "message_scale_equivariance_relerr": change
        if row["condition"] not in report._DETERMINISTIC
        else 0.0,
        "teacher_message_scale_equivariance_relerr": 1e-9 if row["target"] == "path" else 0.0,
    }


def _contract():
    return {
        "profile": "debug",
        "source_config": {"epochs": 4, "model_seeds": [11, 23]},
        "new_training_epochs": 4,
        "graph_count_total": 6,
        "graph_count_used": 6,
        "raw_optimizer_updates": 0,
        "test_updates": 0,
        "models_unchanged": True,
        "source_artifacts_unchanged": True,
        "feature_source_artifacts_unchanged": True,
        "source_code_unchanged": True,
        "normalized_gate_jobs": 6,
    }


@pytest.fixture
def observations():
    rows, scales, interventions = [], [], []
    for split in report._SPLITS:
        for target in report._TARGETS:
            for condition in report._CONDITIONS:
                seeds = [-1] if condition in report._DETERMINISTIC else [11, 23]
                for seed in seeds:
                    for variant in report._VARIANTS:
                        treatments = [
                            ("original", 1.0),
                            *(("fresh", amplitude) for amplitude in (0.25, 1.0, 4.0)),
                        ]
                        for scenario, amplitude in treatments:
                            error = (
                                0.0
                                if (target, condition) in (("L", "first"), ("L2", "polynomial"))
                                else 0.4
                            )
                            if condition not in report._DETERMINISTIC:
                                error += 0.1 if scenario == "fresh" else 0.0
                                if variant == "normalized":
                                    # Lower scale drift can accompany worse accuracy.
                                    error += (
                                        0.2 if scenario == "fresh" and target == "path" else -0.1
                                    )
                            row = _row(
                                variant=variant,
                                target=target,
                                condition=condition,
                                seed=seed,
                                split=split,
                                graph_id=f"g-{split}",
                                scenario=scenario,
                                amplitude=amplitude,
                                error=error,
                            )
                            rows.append(row)
                            if scenario == "fresh":
                                scales.append(_scale(row))
                            if condition == "learned" and amplitude == 1:
                                for name in report._INTERVENTIONS:
                                    value = 0.2 if name in ("identity", "mean") else 0.6
                                    intervention = {
                                        **row,
                                        "intervention": name,
                                        "message_relerr": value,
                                        "message_abs_rmse": value / 10,
                                    }
                                        # Intervention status fields are optional.
                                    del intervention["feature_status"], intervention["source_split"]
                                    interventions.append(intervention)
    return rows, scales, interventions


def test_equal_graph_then_seed_and_shared_controls_are_not_repeated_fits():
    rows = [
        _row(seed=seed, graph_id=graph, error=value)
        for seed, graph, value in ((0, "a", 1), (0, "b", 3), (1, "a", 5), (1, "b", 7))
    ]
    rows[0]["num_realizations"] = 100
    rows[2]["num_realizations"] = 100
    result = report._aggregate(rows, report._FIELDS, report._MESSAGE_METRICS)[0]
    assert result["graphs"] == 2 and result["seeds"] == 2
    assert result["metrics"]["message_relerr"]["mean"] == pytest.approx(4)
    assert result["metrics"]["message_relerr"]["std"] == pytest.approx(math.sqrt(8))
    baseline = report._aggregate(
        [_row(condition="fixed", seed=-1)], report._FIELDS, report._MESSAGE_METRICS
    )[0]
    assert baseline["seeds"] == 1 and baseline["deterministic"]
    assert baseline["metrics"]["message_relerr"]["std"] == 0.0


def test_variant_difference_is_paired_before_std():
    rows = []
    for seed, graph, raw, normalized in (
        (0, "a", 1, 1.5),
        (0, "b", 4, 4.5),
        (1, "a", 2, 1.5),
        (1, "b", 7, 6.5),
    ):
        rows += [
            _row(seed=seed, graph_id=graph, error=raw),
            _row(seed=seed, graph_id=graph, variant="normalized", error=normalized),
        ]
    result = report._pair_differences(rows, "variant")[0]
    assert result["metrics"]["message_relerr"]["mean"] == pytest.approx(0.0)
    assert result["metrics"]["message_relerr"]["std"] == pytest.approx(math.sqrt(0.5))


def test_amplitudes_are_paired_treatments_not_extra_graph_observations():
    row = _row(scenario="fresh")
    values = [_scale({**row, "amplitude": amplitude}) for amplitude in (0.25, 1.0, 4.0)]
    result = report._scale_maxima(values)[0]
    assert result["graphs"] == 1 and result["seeds"] == 1
    assert result["metrics"]["message_scale_equivariance_relerr"]["mean"] == pytest.approx(0.2)
    assert result["metrics"]["message_scale_equivariance_relerr"]["std"] is None


def test_report_outputs_actual_worse_accuracy_despite_small_scale_drift(
    tmp_path, observations, monkeypatch
):
    titles = {}
    real_save = report._save

    def capture(figure, directory, name):
        titles[name] = figure._suptitle.get_text()
        for axis in figure.axes:
            if axis.get_legend() is not None:
                anchor = (
                    axis.get_legend().get_bbox_to_anchor().transformed(axis.transAxes.inverted())
                )
                assert anchor.x0 > 1.0
        real_save(figure, directory, name)

    monkeypatch.setattr(report, "_save", capture)
    report.write_report(tmp_path, *observations, _contract())
    assert set(titles) == set(report._FIGURES)
    for title in titles.values():
        assert "normalized epochs=4 | raw source epochs=4 | seeds=2" in title
        assert "not final performance" in title
    assert len(list(tmp_path.glob("*.png"))) == len(list(tmp_path.glob("*.pdf"))) == 4
    for path in tmp_path.glob("*.png"):
        with Image.open(path) as picture:
            pixels = np.asarray(picture.convert("RGB"))
            assert picture.width > 700 and picture.height > 300
            assert pixels.std() > 10 and np.count_nonzero(np.min(pixels, axis=-1) < 200) > 1000
    for path in tmp_path.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF-")
    text = (tmp_path / "SCALE_NORMALIZATION_SUMMARY.md").read_text(encoding="utf8")
    assert "| fresh | path | id | 0.2 ± 0 | 0.2 ± 0 | higher |" in text
    assert "| original | path | id | -0.1 ± 0 | -0.1 ± 0 | lower |" in text
    assert "| original | raw | id | -0.2 ± 0 | -0.2 ± 0 |" in text
    assert "| fresh | normalized | id | -0.5 ± 0 | -0.5 ± 0 |" in text
    assert "C를 유일하게 식별하는 조건이 아니다" in text
    assert "normalized gate는 원본 학습 데이터에서 새로 학습" in text
    assert "구조적으로 성립" in text and "최종 성능이나 가설의 성공을 판정하지 않는다" in text
    assert "first/polynomial/fixed는 한 번 적합한 같은 frozen control" in text
    assert "weight_and_interventions.pdf" in text


@pytest.mark.parametrize(
    "problem",
    [
        "nan",
        "missing_variant",
        "shared_control",
        "scale_pair",
        "feature_pair",
        "intervention_pair",
        "duplicate_intervention",
        "frozen_guard",
        "raw_update",
        "test_update",
    ],
)
def test_invalid_treatment_or_frozen_control_fails_before_writing(tmp_path, observations, problem):
    rows, scales, interventions = copy.deepcopy(observations)
    contract = _contract()
    if problem == "nan":
        rows[0]["message_abs_rmse"] = float("nan")
    elif problem == "missing_variant":
        rows[0]["variant"] = "missing"
    elif problem == "shared_control":
        rows[0]["message_relerr"] += 0.1
    elif problem == "scale_pair":
        scales.pop()
    elif problem == "feature_pair":
        next(
            row for row in rows if row["variant"] == "normalized" and row["scenario"] == "original"
        )["graph_id"] = "other"
    elif problem == "intervention_pair":
        interventions.pop()
    elif problem == "duplicate_intervention":
        interventions.append(interventions[0].copy())
    elif problem == "frozen_guard":
        contract["models_unchanged"] = False
    elif problem == "raw_update":
        contract["raw_optimizer_updates"] = 1
    elif problem == "test_update":
        contract["test_updates"] = 1
    with pytest.raises(ValueError):
        report.write_report(tmp_path, rows, scales, interventions, contract)
    assert not list(tmp_path.iterdir())


def test_undefined_weight_metrics_are_not_fabricated_zero():
    rows = [_row(error=None)]
    for name in report._MESSAGE_METRICS[2:]:
        rows[0][name] = None
    summary = report._aggregate(rows, report._FIELDS, report._MESSAGE_METRICS)[0]
    assert summary["metrics"]["message_relerr"]["mean"] is None
    assert summary["metrics"]["weight_corr"]["mean"] is None
    assert summary["metrics"]["weight_corr"]["undefined"] == 1
    assert report._estimate(summary["metrics"]["weight_corr"]) == "undefined"


def test_full_label_uses_actual_epochs_and_seed_count():
    label = report._label(
        {
            "profile": "full",
            "new_training_epochs": 500,
            "source_config": {"epochs": 500, "model_seeds": [11, 23, 37, 53, 71]},
        }
    )
    assert "FULL | normalized epochs=500 | raw source epochs=500 | seeds=5" == label


def test_existing_report_artifacts_are_preserved(tmp_path, observations):
    destination = tmp_path / "amplitude_error.pdf"
    destination.write_bytes(b"previous experiment")
    with pytest.raises(FileExistsError, match="overwrite"):
        report.write_report(tmp_path, *observations, _contract())
    assert destination.read_bytes() == b"previous experiment"
    assert list(tmp_path.iterdir()) == [destination]
