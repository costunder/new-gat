"""Debug fixtures test report arithmetic; they are not learned-model results."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from PIL import Image

import research.wedge_propagation.learned.report as report_module
from research.wedge_propagation.learned.report import _run_label, _summarize, write_report


def _row(target="path", condition="learned", seed=0, graph_id="g0", error=0.5):
    return {
        "target": target,
        "condition": condition,
        "seed": seed,
        "split": "id",
        "graph_id": graph_id,
        "family": "tree",
        "num_nodes": 4,
        "num_edges": 3,
        "num_paths": 2,
        "num_realizations": 2,
        "message_relerr": error,
        "message_abs_rmse": 0.0 if error is None else error / 10,
        "weight_relerr": 0.2 if target == "path" else None,
        "weight_corr": 0.6 if target == "path" else None,
        "weight_mean_error": 0.0 if target == "path" else None,
        "beta": None,
        "u": None,
        "v": None,
    }


@pytest.fixture
def measured_rows():
    metrics = []
    interventions = []
    training = []
    for target in ("L", "L2", "path"):
        for condition in ("first", "polynomial", "fixed"):
            error = 0.0 if (target, condition) in (("L", "first"), ("L2", "polynomial")) else 0.5
            row = _row(target, condition, -1, error=error)
            for field in ("weight_relerr", "weight_corr", "weight_mean_error"):
                row[field] = None
            metrics.append(row)
        for condition in ("learned", "random_pair"):
            for seed in (0, 1):
                metrics.append(
                    _row(target, condition, seed, error=0.5 if condition == "learned" else 0.8)
                )
                for epoch in (1, 2):
                    training.append(
                        {
                            "target": target,
                            "condition": condition,
                            "seed": seed,
                            "epoch": epoch,
                            "train_loss": 0.5 / epoch,
                            "val_message_relerr": 0.2 / epoch,
                            "best_epoch": epoch,
                            "seconds": 0.1,
                            "peak_vram_bytes": None,
                        }
                    )
                if condition == "learned":
                    for intervention in (
                        "identity",
                        "mean",
                        "weight_shuffle",
                        "other_graph_pattern",
                        "correspondence_randomization",
                    ):
                        row = _row(target, condition, seed, error=0.6)
                        row["intervention"] = intervention
                        interventions.append(row)
    return metrics, interventions, training


def test_graph_macro_then_seed_statistics_do_not_weight_realization_counts():
    rows = [
        _row(seed=0, graph_id="a", error=1.0),
        _row(seed=0, graph_id="b", error=3.0),
        _row(seed=1, graph_id="a", error=5.0),
        _row(seed=1, graph_id="b", error=7.0),
    ]
    rows[0]["num_realizations"] = 100
    rows[2]["num_realizations"] = 100
    rows[0]["weight_corr"] = None
    rows[2]["weight_corr"] = None
    summary = _summarize(rows)[0]
    assert summary["graph_count"] == 2 and summary["seed_count"] == 2
    statistic = summary["metrics"]["message_relerr"]
    assert statistic["mean"] == pytest.approx(4.0)
    assert statistic["std"] == pytest.approx(math.sqrt(8.0))
    assert summary["metrics"]["weight_corr"]["undefined_graph_observations"] == 2


def test_deterministic_fit_and_single_learned_seed_have_distinct_std():
    fitted = _summarize([_row(condition="fixed", seed=-1)])[0]
    learned = _summarize([_row(seed=0)])[0]
    assert fitted["deterministic"]
    assert fitted["seed_count"] == 1
    assert fitted["metrics"]["message_relerr"]["std"] == 0.0
    assert learned["metrics"]["message_relerr"]["std"] is None


def test_report_creates_real_figures_and_reports_ties(tmp_path, measured_rows, monkeypatch):
    metrics, interventions, training = measured_rows
    titles = {}
    original_save = report_module._save

    def inspect_and_save(figure, directory, name):
        titles[name] = figure._suptitle.get_text()
        for axis in figure.axes:
            legend = axis.get_legend()
            if legend is not None:
                anchor = legend.get_bbox_to_anchor().transformed(axis.transAxes.inverted())
                assert anchor.x0 > 1.0
        original_save(figure, directory, name)

    monkeypatch.setattr(report_module, "_save", inspect_and_save)
    contract = {"profile": "debug", "config": {"epochs": 2, "model_seeds": [0, 1]}}
    write_report(tmp_path, metrics, interventions, training, contract)
    assert set(titles) == {
        "learning_curves",
        "split_message_error",
        "intervention_error",
        "weight_recovery",
    }
    for title in titles.values():
        assert "DEBUG | epochs=2 | seeds=2" in title
        assert "Verification only; not final performance" in title
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4
    for path in tmp_path.glob("*.png"):
        with Image.open(path) as picture:
            assert picture.width > 700 and picture.height > 300
            pixels = np.asarray(picture.convert("RGB"))
            assert np.std(pixels) > 10
            assert np.count_nonzero(np.min(pixels, axis=-1) < 200) > 1000
    for path in tmp_path.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF-")
    summary = (tmp_path / "LEARNED_SUMMARY.md").read_text(encoding="utf8")
    assert "실제 데이터셋의 분류 성능을 측정한 결과는 아니다" in summary
    assert "| id | 0.5 | fixed | 0.5 | 0 | equal |" in summary
    assert "Identity/mean 대응 행: 6" in summary
    assert "metrics.csv" in summary and "training.csv" in summary
    assert "유일하게 식별" in summary
    opening = summary.split("## 실제 메시지 오차")[0]
    assert "profile: debug; 학습 계약: 2 epoch, 2 seed" in opening
    assert "최종 성능이나 연구 가설의 성공·실패를 판정하는 결과가 아니다" in opening
    assert "teacher_operator_audit.csv" in summary
    assert "사후적으로 맞춘 잔차" in summary
    assert "### 실행 계약 요약" in summary
    assert "[contract.json](contract.json)" in summary
    assert "unreported" in summary
    assert "```json" not in summary
    assert len(summary.splitlines()) > 30


def test_report_does_not_overwrite_existing_results(tmp_path, measured_rows):
    metrics, interventions, training = measured_rows
    (tmp_path / "weight_recovery.pdf").write_bytes(b"previous report")
    with pytest.raises(FileExistsError, match="overwrite"):
        write_report(tmp_path, metrics, interventions, training, {})
    assert (tmp_path / "weight_recovery.pdf").read_bytes() == b"previous report"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "failure",
    [
        "nan_metric",
        "nan_weight",
        "missing_rmse",
        "duplicate_metric",
        "duplicate_epoch",
        "seed_coverage",
    ],
)
def test_invalid_measurements_fail_before_creating_artifacts(tmp_path, measured_rows, failure):
    metrics, interventions, training = copy.deepcopy(measured_rows)
    if failure == "nan_metric":
        metrics[0]["message_relerr"] = float("nan")
    elif failure == "nan_weight":
        metrics[0]["weight_corr"] = float("nan")
    elif failure == "missing_rmse":
        del metrics[0]["message_abs_rmse"]
    elif failure == "duplicate_metric":
        metrics.append(metrics[0].copy())
    elif failure == "duplicate_epoch":
        training.append(training[0].copy())
    elif failure == "seed_coverage":
        next(
            row
            for row in metrics
            if row["target"] == "L" and row["condition"] == "learned" and row["seed"] == 0
        )["graph_id"] = "different-graph"
    with pytest.raises(ValueError):
        write_report(tmp_path, metrics, interventions, training, {})
    assert not list(tmp_path.iterdir())


def test_zero_path_outputs_and_absent_weight_diagnostics_are_undefined(tmp_path):
    row = _row(error=None)
    row["num_paths"] = 0
    row["message_abs_rmse"] = 0.0
    for name in ("weight_relerr", "weight_corr", "weight_mean_error"):
        del row[name]
    training = [
        {
            "target": "path",
            "condition": "learned",
            "seed": 0,
            "epoch": 1,
            "train_loss": 0.0,
            "val_message_relerr": None,
            "best_epoch": None,
            "seconds": 0.1,
            "peak_vram_bytes": None,
        }
    ]
    write_report(
        tmp_path,
        [row],
        [],
        training,
        {
            "profile": "debug",
            "config": {"epochs": 1, "model_seeds": [0]},
            "zero_path_fixture": True,
        },
    )
    summary = (tmp_path / "LEARNED_SUMMARY.md").read_text(encoding="utf8")
    assert "| path | learned | id | 1 | 1 | undefined | 0 (std undefined) | 1 |" in summary
    assert "| learned | id | undefined | undefined | undefined | 1 / 1 / 1 |" in summary
    assert len(list(tmp_path.glob("*.png"))) == 4


@pytest.mark.parametrize(
    ("profile", "epochs", "seeds", "expected"),
    [
        ("debug", 4, [11, 23], "DEBUG | epochs=4 | seeds=2"),
        ("full", 500, [11, 23, 37, 53, 71], "FULL | epochs=500 | seeds=5"),
    ],
)
def test_run_label_uses_contract_epochs_and_seeds(profile, epochs, seeds, expected):
    label = _run_label({"profile": profile, "config": {"epochs": epochs, "model_seeds": seeds}})
    assert expected in label
    assert ("not final performance" in label) == (profile == "debug")
