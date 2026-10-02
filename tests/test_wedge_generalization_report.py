"""Arithmetic and presentation fixtures, separate from research performance evidence."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from PIL import Image

import research.wedge_propagation.generalization.report as report


def _row(
    target="path",
    condition="learned",
    seed=0,
    graph_id="g0",
    error=0.4,
    scenario="original",
    amplitude=1.0,
    split="id",
):
    return {
        "target": target,
        "condition": condition,
        "seed": seed,
        "graph_id": graph_id,
        "family": "tree",
        "num_nodes": 5,
        "num_edges": 4,
        "num_paths": 3,
        "num_realizations": 2,
        "message_relerr": error,
        "message_abs_rmse": 0.0 if error is None else error / 10,
        "weight_relerr": 0.2 if target == "path" and condition == "learned" else None,
        "weight_corr": 0.6 if target == "path" and condition == "learned" else None,
        "weight_mean_error": 0.0 if condition == "learned" else None,
        "split": split,
        "source_split": split,
        "scenario": scenario,
        "amplitude": amplitude,
        "feature_status": "source_features" if scenario == "original" else "unseen_features",
    }


def _scale(row, amplitude=1.0):
    return {
        **{
            name: row[name]
            for name in ("target", "condition", "seed", "graph_id", "split", "family")
        },
        "amplitude": amplitude,
        "student_weight_scale_relerr": 0.0 if row["condition"] == "learned" else None,
        "teacher_weight_scale_relerr": 0.0 if row["target"] == "path" else None,
        "message_scale_equivariance_relerr": 0.0,
        "teacher_message_scale_equivariance_relerr": 0.0,
    }


def _contract():
    return {
        "profile": "debug",
        "source_config": {"epochs": 4, "model_seeds": [11, 23]},
        "model_seeds": [11, 23],
        "source_graph_count": 6,
        "feature_realizations": 2,
        "optimizer_updates": 0,
        "new_training_epochs": 0,
        "models_unchanged": True,
        "source_artifacts_unchanged": True,
        "source_model_hashes": {"path": "same"},
        "model_hashes_after": {"path": "same"},
        "source_data_hash": "sourcehash",
        "fresh_data_hash": "freshhash",
        "original_metric_reproduction": {"maximum_absolute_error": 0.0, "tolerance": 1e-6},
    }


@pytest.fixture
def observations():
    metrics, scales, source, interventions, audits = [], [], [], [], []
    for split in report._SPLITS:
        graph_id = f"g-{split}"
        audits.append(
            {
                "graph_id": graph_id,
                "family": "tree",
                "split": split,
                "num_features": 2,
                "teacher_span_L_L2_Q_residual": 0.3,
                "teacher_span_absolute_residual_mean": 1.0,
                "teacher_span_defined_features": 2,
                "teacher_span_undefined_features": 0,
                "teacher_operator_variation": 0.2,
                "teacher_operator_reference_norm": 4.0,
                "teacher_operator_variation_defined_pairs": 1,
                "teacher_c_std": 0.4,
            }
        )
        for target in report._TARGETS:
            for condition in report._CONDITIONS:
                seeds = [-1] if condition in report._DETERMINISTIC else [11, 23]
                error = (
                    0.0 if (target, condition) in (("L", "first"), ("L2", "polynomial")) else 0.4
                )
                for seed in seeds:
                    row = _row(target, condition, seed, graph_id, error, split=split)
                    metrics.append(row)
                    if split != "train":
                        source.append(row.copy())
                    for amplitude in (0.25, 1.0, 4.0):
                        fresh_error = error + 0.2 if condition == "learned" else error
                        metrics.append(
                            _row(
                                target,
                                condition,
                                seed,
                                graph_id,
                                fresh_error,
                                scenario="fresh",
                                amplitude=amplitude,
                                split=split,
                            )
                        )
                        scales.append(_scale(row, amplitude))
                    if condition == "learned" and split != "train":
                        for intervention in report._INTERVENTIONS:
                            # A frozen intervention can improve; reports must retain this.
                            interventions.append(
                                {
                                    **row,
                                    "intervention": intervention,
                                    "message_relerr": 0.1,
                                    "message_abs_rmse": 0.01,
                                }
                            )
    return metrics, scales, source, interventions, audits


def test_equal_graph_macro_then_seed_and_missing_metrics():
    rows = [
        _row(seed=seed, graph_id=graph, error=value)
        for seed, graph, value in ((0, "a", 1.0), (0, "b", 3.0), (1, "a", 5.0), (1, "b", 7.0))
    ]
    rows[0]["num_realizations"] = 100
    rows[2]["num_realizations"] = 100
    rows[0]["weight_corr"] = None
    rows[2]["weight_corr"] = None
    group = report._aggregate(rows, ("target", "condition", "split"), report._MESSAGE_METRICS)[0]
    assert group["graphs"] == 2 and group["seeds"] == 2
    assert group["metrics"]["message_relerr"]["mean"] == pytest.approx(4.0)
    assert group["metrics"]["message_relerr"]["std"] == pytest.approx(math.sqrt(8))
    assert group["metrics"]["weight_corr"]["undefined"] == 2


def test_paired_feature_difference_uses_corresponding_graphs_before_seed_std():
    rows = []
    for seed, graph, original, fresh in (
        (0, "a", 1.0, 1.5),
        (0, "b", 4.0, 4.5),
        (1, "a", 2.0, 1.5),
        (1, "b", 7.0, 6.5),
    ):
        rows += [
            _row(seed=seed, graph_id=graph, error=original),
            _row(seed=seed, graph_id=graph, error=fresh, scenario="fresh"),
        ]
    group = report._paired_features(rows)[0]
    assert group["metrics"]["message_relerr"]["mean"] == pytest.approx(0.0)
    assert group["metrics"]["message_relerr"]["std"] == pytest.approx(math.sqrt(0.5))


def test_report_creates_real_plots_and_honest_scope(tmp_path, observations, monkeypatch):
    titles = {}
    original_save = report._save

    def capture(figure, directory, name):
        titles[name] = figure._suptitle.get_text()
        if name == "source_evidence":
            # Source CSV has five evaluation splits, so the train tick is unmeasured.
            for axis in figure.axes:
                assert all(float(x) > 0.5 for line in axis.lines for x in line.get_xdata())
        for axis in figure.axes:
            legend = axis.get_legend()
            if legend is not None:
                box = legend.get_bbox_to_anchor().transformed(axis.transAxes.inverted())
                assert box.x0 > 1.0
        original_save(figure, directory, name)

    monkeypatch.setattr(report, "_save", capture)
    report.write_report(tmp_path, *observations, _contract())
    assert set(titles) == set(report._FIGURES)
    for title in titles.values():
        assert "DEBUG | new epochs=0 (frozen) | source epochs=4 | seeds=2" in title
        assert "not final performance" in title
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4
    for path in tmp_path.glob("*.png"):
        with Image.open(path) as picture:
            pixels = np.asarray(picture.convert("RGB"))
            assert picture.width > 700 and picture.height > 300
            assert pixels.std() > 10
            assert np.count_nonzero(np.min(pixels, axis=-1) < 200) > 1000
    for path in tmp_path.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF-")
    summary = (tmp_path / "SYNTHETIC_GENERALIZATION_SUMMARY.md").read_text(encoding="utf8")
    assert "최종 성능이나 연구 가설의 성공을 판정하지 않는다" in summary
    assert "original/train은 학습에서 본 그래프·특징" in summary
    assert "fresh/train은 같은 학습 그래프의 새 특징" in summary
    assert "독립 표본이 아니다" in summary
    assert "source_interventions.csv" in summary
    assert "source_teacher_operator_audit.csv" in summary
    assert "| id | fixed | 0.2 | higher |" in summary
    assert "| path | id | 0.1 ± 0 | 0.1 ± 0 |" in summary
    assert "유일하게 복원" in summary
    assert "teacher 행렬 T=Aᵀdiag(c*)A를 행렬 span{L,L²,Q}에 사후 적합" in summary
    assert "상대 Frobenius 잔차 ||T−T_fit||F/||T||F" in summary
    assert "0이 아닌 teacher 행렬에 대해 평균" in summary
    assert "목표 노드 메시지의 적합 오차가 아니다" in summary
    assert "임의의 spectral 함수까지 배제하지 않는다" in summary
    assert "span{LX,L²X,QX}" not in summary
    assert "sourcehash" in summary and "freshhash" in summary
    assert "models_unchanged: true" in summary


def test_preserves_existing_artifacts(tmp_path, observations):
    (tmp_path / "amplitude_error.pdf").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        report.write_report(tmp_path, *observations, _contract())
    assert (tmp_path / "amplitude_error.pdf").read_bytes() == b"existing"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "problem",
    [
        "nan",
        "missing_rmse",
        "duplicate",
        "missing_pair",
        "topology",
        "feature_status",
        "seed_coverage",
        "guard",
        "hashes",
        "optimizer",
        "epochs",
        "reproduction",
        "audit_count",
        "scale_coverage",
        "reproduction_guard",
    ],
)
def test_invalid_or_tampered_observations_fail_before_output(tmp_path, observations, problem):
    metrics, scales, source, interventions, audits = copy.deepcopy(observations)
    contract = _contract()
    if problem == "nan":
        metrics[0]["message_relerr"] = float("nan")
    elif problem == "missing_rmse":
        del metrics[0]["message_abs_rmse"]
    elif problem == "duplicate":
        metrics.append(metrics[0].copy())
    elif problem == "missing_pair":
        metrics.pop(0)
    elif problem == "topology":
        metrics[0]["num_edges"] += 1
    elif problem == "feature_status":
        metrics[0]["feature_status"] = "unseen_features"
    elif problem == "seed_coverage":
        next(row for row in metrics if row["condition"] == "learned" and row["seed"] == 11)[
            "graph_id"
        ] = "different"
    elif problem == "guard":
        contract["models_unchanged"] = False
    elif problem == "hashes":
        contract["model_hashes_after"]["path"] = "changed"
    elif problem == "optimizer":
        contract["optimizer_updates"] = 1
    elif problem == "epochs":
        contract["new_training_epochs"] = 1
    elif problem == "reproduction":
        contract["original_metric_reproduction"]["maximum_absolute_error"] = 0.5
    elif problem == "audit_count":
        audits[0]["teacher_span_defined_features"] = 1
    elif problem == "scale_coverage":
        scales.pop()
    elif problem == "reproduction_guard":
        contract["original_metric_reproduction"]["verified"] = False
    with pytest.raises(ValueError):
        report.write_report(tmp_path, metrics, scales, source, interventions, audits, contract)
    assert not list(tmp_path.iterdir())


def test_zero_messages_and_nullable_weight_diagnostics(tmp_path):
    before = _row(error=0.0)
    before["num_paths"] = 0
    after = {**before, "scenario": "fresh", "feature_status": "unseen_features"}
    for row in (before, after):
        for name in report._MESSAGE_METRICS[2:]:
            row[name] = None
    scale = _scale(before)
    scale["student_weight_scale_relerr"] = None
    scale["teacher_weight_scale_relerr"] = None
    report.write_report(tmp_path, [before, after], [scale], [], [], [], _contract())
    summary = (tmp_path / "SYNTHETIC_GENERALIZATION_SUMMARY.md").read_text(encoding="utf8")
    assert "0 (std undefined)" in summary
    assert "undefined | undefined | 0 / 0 / 1 / 1" in summary
    assert "unreported" in summary


def test_full_label_uses_source_configuration():
    label = report._label(
        {"profile": "full", "source_config": {"epochs": 500, "model_seeds": [11, 23, 37, 53, 71]}}
    )
    assert "FULL | new epochs=0 (frozen) | source epochs=500 | seeds=5" in label
    assert "not final performance" not in label
