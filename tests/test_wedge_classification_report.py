"""Arithmetic reporting fixtures, not real classification performance."""

import copy

import matplotlib.image as mpimg
import numpy as np
import pytest

from research.wedge_propagation.classification.report import (
    estimate,
    intervention_changes,
    paired_comparisons,
    write_report,
)

CONDITIONS = [
    "mlp",
    "first_order",
    "polynomial_2",
    "fixed_wedge",
    "learned_wedge_raw",
    "learned_wedge_rms",
    "fixed_wedge_node_mlp",
    "standard_gcn",
]
HOLD = "hold_each_layers_preintervention_kappa"


def fixture_rows():
    config = {
        "profile": "debug",
        "data": {"datasets": ["DEBUG-arithmetic"]},
        "conditions": CONDITIONS,
        "backbone": {"layers": 2},
        "training": {
            "final_seeds": [11, 23],
            "epochs_per_run": 3,
            "tuning_runs": 16,
            "final_runs": 16,
            "total_runs": 32,
        },
        "evaluation": {
            "scale_amplitudes": [0.25, 1.0, 4.0],
            "shuffle_and_random_manifests_per_dataset": 1,
        },
    }
    metrics, scales, interventions, gates = [], [], [], []
    for condition in CONDITIONS:
        for index, seed in enumerate([11, 23]):
            accuracy = (
                [0.79, 0.85][index]
                if condition == "learned_wedge_raw"
                else (
                    [0.77, 0.80][index] if condition == "learned_wedge_rms" else 0.8 + 0.02 * index
                )
            )
            base = {"dataset": "DEBUG-arithmetic", "condition": condition, "seed": seed}
            for split in ("train", "validation", "test"):
                row = {
                    **base,
                    "split": split,
                    "ce": 1 - accuracy,
                    "accuracy": accuracy,
                    "num_nodes": 6,
                    "num_labeled_nodes": 2,
                }
                metrics.append(row)
                for amplitude in config["evaluation"]["scale_amplitudes"]:
                    scales.append(
                        {
                            **row,
                            "scope": "end_to_end",
                            "layer": -1,
                            "amplitude": amplitude,
                            "logit_scale_equivariance_relerr": 0.0,
                        }
                    )
                if condition.startswith("learned_"):
                    treatments = [("c_identity", mode, -1) for mode in ("recompute", HOLD)]
                    treatments += [("c_position_shuffle", mode, 0) for mode in ("recompute", HOLD)]
                    treatments += [
                        ("second_branch_remove", "not_applicable", -1),
                        ("random_physical_edge_pair_correspondence", HOLD, 0),
                    ]
                    for intervention, mode, manifest in treatments:
                        # A real observed improvement must not be converted into degradation.
                        change = 0.01 if intervention == "c_identity" else -0.02
                        interventions.append(
                            {
                                **row,
                                "accuracy": accuracy + change,
                                "ce": 1 - accuracy - change,
                                "intervention": intervention,
                                "kappa_mode": mode,
                                "manifest_index": manifest,
                            }
                        )
            for layer in (0, 1):
                gates.append(
                    {
                        **base,
                        "layer": layer,
                        "intervention": "original",
                        "alpha": 0.5,
                        "beta": 0.25,
                        "branch_norm": 0.8,
                        "c_mean": 1 if condition.startswith("learned_") else None,
                        "c_std": 0.2 if condition.startswith("learned_") else None,
                    }
                )
                if condition.startswith("learned_"):
                    for amplitude in config["evaluation"]["scale_amplitudes"]:
                        change = 0.0 if condition.endswith("rms") or amplitude == 1 else 0.3
                        scales.append(
                            {
                                **base,
                                "scope": "fixed_layer_Z",
                                "split": "not_applicable",
                                "layer": layer,
                                "amplitude": amplitude,
                                "c_scale_relerr": change,
                                "kappa_scale_relerr": change,
                                "message_scale_equivariance_relerr": change,
                            }
                        )
    selections = [
        {"dataset": "DEBUG-arithmetic", "condition": condition, "selected_lr": 0.001}
        for condition in CONDITIONS
    ]
    resources = [
        {
            "dataset": "DEBUG-arithmetic",
            "condition": condition,
            "phase": "final",
            "measured": True,
            "status": "measured",
            "packed_runs": 2,
            "seconds_per_epoch": 1.2,
            "peak_vram_bytes": 2**30,
            "parameters_per_seed": 100,
        }
        for condition in CONDITIONS
    ]
    resources += [
        {
            "dataset": "DEBUG-arithmetic",
            "condition": "mlp",
            "measured": False,
            "status": "OOM",
            "seconds_per_epoch": None,
            "peak_vram_bytes": None,
        }
    ]
    return config, metrics, interventions, scales, selections, resources, gates


def test_paired_estimate_uses_seed_differences_and_real_uncertainty():
    config, metrics, *_ = fixture_rows()
    assert config["profile"] == "debug"
    found = [
        r
        for r in paired_comparisons(metrics)
        if r["first"] == "learned_wedge_rms"
        and r["second"] == "learned_wedge_raw"
        and r["metric"] == "accuracy"
    ][0]
    np.testing.assert_allclose(found["mean"], -0.035)
    np.testing.assert_allclose(found["std"], np.std([-0.02, -0.05], ddof=1))
    assert found["count"] == 2 and found["lower"] < found["mean"] < found["upper"]
    assert estimate([0.2])["std"] is None
    assert estimate([])["mean"] is None


def test_manifest_treatments_are_averaged_within_seed_first():
    _, metrics, interventions, *_ = fixture_rows()
    selected = [r for r in interventions if r["intervention"] == "c_position_shuffle"]
    doubled = selected + [{**r, "manifest_index": 1} for r in selected]
    estimates = intervention_changes(doubled, metrics, "accuracy")
    assert all(r["count"] == 2 and r["mean"] == pytest.approx(-0.02) for r in estimates)


def test_actual_plots_and_summary_preserve_worse_rms_and_improving_intervention(tmp_path):
    args = fixture_rows()
    write_report(tmp_path, *args)
    summary = (tmp_path / "CLASSIFICATION_SUMMARY.md").read_text(encoding="utf-8")
    assert "DEBUG" in summary and "epochs=3" in summary and "final seeds=2" in summary
    assert "learner" not in summary
    assert "learned_wedge_rms − learned_wedge_raw | accuracy | -3.5" in summary
    assert "c_identity | recompute | 1" in summary
    assert "원래 분류 48, 개입 72, 배율 168" in summary
    assert "교사" not in summary or "Synthetic teacher" in summary
    for name in (
        "classification_performance",
        "frozen_interventions",
        "scale_response",
        "branches_and_resources",
    ):
        pixels = mpimg.imread(tmp_path / f"{name}.png")
        assert pixels.ndim == 3 and pixels.shape[0] > 500 and pixels.std() > 0.01
        assert (tmp_path / f"{name}.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    "kind",
    [
        "missing_metric",
        "duplicate_metric",
        "nan",
        "missing_intervention",
        "missing_scale",
        "guard",
        "updates",
    ],
)
def test_invalid_or_incomplete_measurements_fail_before_writing(tmp_path, kind):
    args = list(copy.deepcopy(fixture_rows()))
    contract = {}
    if kind == "missing_metric":
        args[1].pop()
    elif kind == "duplicate_metric":
        args[1].append(copy.deepcopy(args[1][0]))
    elif kind == "nan":
        args[1][0]["ce"] = float("nan")
    elif kind == "missing_intervention":
        args[2].pop()
    elif kind == "missing_scale":
        args[3].pop()
    elif kind == "guard":
        contract["models_unchanged"] = False
    else:
        contract["evaluation_optimizer_updates"] = 1
    with pytest.raises(ValueError):
        write_report(tmp_path, *args, contract=contract)
    assert not list(tmp_path.iterdir())


def test_previous_artifacts_are_preserved(tmp_path):
    (tmp_path / "scale_response.pdf").write_bytes(b"previous")
    with pytest.raises(FileExistsError):
        write_report(tmp_path, *fixture_rows())
    assert (tmp_path / "scale_response.pdf").read_bytes() == b"previous"
    assert len(list(tmp_path.iterdir())) == 1


def test_zero_reference_diagnostics_are_not_fabricated_as_zero(tmp_path):
    args = list(fixture_rows())
    for row in args[3]:
        if row["scope"] == "fixed_layer_Z":
            row["c_scale_relerr"] = None
            row["message_scale_equivariance_relerr"] = None
    write_report(tmp_path, *args)
    summary = (tmp_path / "CLASSIFICATION_SUMMARY.md").read_text(encoding="utf-8")
    assert "| undefined | undefined | 2 / 2 |" in summary
