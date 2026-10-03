"""Arithmetic report fixtures are explicitly DEBUG, not citation results."""

import copy

import matplotlib.image as mpimg
import numpy as np
import pytest

from research.wedge_propagation.branch_strength.report import (
    FIGURES,
    LEARNED,
    SHUFFLE,
    SINGLE,
    TREATMENTS,
    diagnostic_estimates,
    diagnostic_value,
    estimate,
    paired_changes,
    write_report,
)


def fixture_rows():
    conditions = [
        "mlp", "first_order", "polynomial_2", "fixed_wedge", *LEARNED,
        "fixed_wedge_node_mlp", "standard_gcn",
    ]
    config = {
        "profile": "debug", "data": {"datasets": ["DEBUG-arithmetic"]},
        "conditions": conditions, "final_seeds": [11, 23], "manifests_per_dataset": 2,
        "treatments": list(TREATMENTS), "targets": ["layer_0", "layer_1", "both"],
    }
    baseline, treatments, diagnostics, fixed_z = [], [], [], []

    def diagnostic(base, treatment, target, layer, manifest, source):
        is_matched = treatment.endswith("norm_matched")
        branch = {
            "baseline": 1.0, "true_c_unit_denominator": 4.0,
            "identity_hold_reference": 0.3, "identity_unit_denominator": 1.2,
            "identity_norm_matched": 1.0, "branch_off": 0.0,
            "shuffle_hold_reference": 0.75, "shuffle_norm_matched": 1.0,
        }[treatment]
        return {
            **base, "layer": layer, "treatment": treatment, "target": target,
            "manifest_index": manifest, "source": source,
            "z_norm": 2.0, "alpha_l_norm": 0.4, "beta_t_norm": branch * 0.2,
            "alpha_l_to_input": 0.2, "beta_t_to_input": branch * 0.1,
            "beta_t_to_alpha_l": branch * 0.5,
            "branch_norm": branch, "reference_branch_norm": 1.0,
            "branch_reference_cosine": (
                None if branch == 0 else 1.0
                if treatment in ("baseline", "true_c_unit_denominator") else 0.8
            ),
            "branch_reference_cosine_defined": branch != 0,
            "branch_delta_to_input": 0.1,
            "normalization_gain": 1.3 if is_matched else 1.0,
            "alpha": 0.5, "beta": 0.2, "kappa": 1.0, "kappa_reference": 4.0,
        }

    for condition in conditions:
        for seed_index, seed in enumerate(config["final_seeds"]):
            base = {"dataset": "DEBUG-arithmetic", "condition": condition, "seed": seed}
            for split in ("train", "validation", "test"):
                metric = {
                    **base, "split": split, "ce": 0.6 + seed_index * 0.1,
                    "accuracy": 0.8 + seed_index * 0.02, "num_nodes": 6,
                    "num_labeled_nodes": 2,
                }
                baseline.append(metric)
                if condition not in LEARNED:
                    continue
                for treatment in (*SINGLE, *SHUFFLE):
                    indices = range(2) if treatment in SHUFFLE else (-1,)
                    for target in config["targets"]:
                        for manifest in indices:
                            # Matched C=1 gets worse: preserve the actual sign.
                            change = -0.02 if treatment == "identity_norm_matched" else 0.01
                            treatments.append({
                                **metric, "treatment": treatment, "target": target,
                                "manifest_index": manifest, "accuracy": metric["accuracy"] + change,
                                "ce": metric["ce"] - change,
                            })
            for layer in (0, 1):
                diagnostics.append(diagnostic(base, "baseline", "none", layer, -1, "end_to_end"))
                if condition not in LEARNED:
                    continue
                for treatment in TREATMENTS:
                    indices = range(2) if treatment in SHUFFLE else (-1,)
                    for manifest in indices:
                        fixed_z.append(diagnostic(
                            base, treatment, f"layer_{layer}", layer, manifest, "fixed_Z"
                        ))
                        if treatment == "baseline":
                            continue
                        for target in config["targets"]:
                            diagnostics.append(diagnostic(
                                base, treatment, target, layer, manifest, "end_to_end"
                            ))
    config["expected_counts"] = {
        "baseline_rows": len(baseline), "treatment_rows": len(treatments),
        "diagnostic_rows": len(diagnostics), "fixed_z_rows": len(fixed_z),
        "source_final_models": 16, "learned_final_models": 4,
    }
    resources = [{
        "scope": "evaluation", "dataset": "DEBUG-arithmetic", "condition": "learned_wedge_raw",
        "device": "cpu", "packed_runs": 2, "path_chunk": 64, "seconds_per_forward": 0.1,
        "peak_vram_bytes": None, "model_forwards_per_second": 20.0,
        "measured": True, "status": "measured", "completed_seed_cases": 2,
        "wall_seconds": 0.1,
    }, {
        "scope": "calibration", "measured": False, "status": "OOM",
        "seconds_per_forward": None, "peak_vram_bytes": None,
    }]
    contract = {
        "source_config": {"training": {"epochs_per_run": 4}},
        "models_unchanged": True, "source_artifacts_unchanged": True,
        "code_and_graphs_preserved": True, "new_optimizer_updates": 0,
        "new_training_epochs": 0, "source_files_and_models_preserved": True,
    }
    return config, baseline, treatments, diagnostics, fixed_z, resources, contract


def test_manifest_averages_do_not_increase_seed_count_or_hide_actual_sign():
    _, baseline, treatments, *_ = fixture_rows()
    chosen = [r for r in treatments if r["treatment"] == "shuffle_norm_matched"]
    for row in chosen:
        row["accuracy"] += (0.03 if row["manifest_index"] == 0 else -0.01)
    rows = paired_changes(baseline, chosen, "accuracy")
    assert all(row["count"] == 2 for row in rows)
    assert all(row["mean"] == pytest.approx(0.02) for row in rows)
    assert estimate([0.2])["std"] is None
    assert estimate([])["mean"] is None


def test_seed_interval_is_computed_from_paired_differences():
    _, baseline, treatments, *_ = fixture_rows()
    selected = [r for r in treatments if r["treatment"] == "identity_norm_matched"]
    for row in selected:
        if row["seed"] == 23:
            row["accuracy"] -= 0.04
    result = paired_changes(baseline, selected, "accuracy")[0]
    assert result["mean"] == pytest.approx(-0.04)
    assert result["std"] == pytest.approx(np.std([-0.02, -0.06], ddof=1))
    assert result["lower"] < result["mean"] < result["upper"]


def test_undefined_ratios_and_manifest_diagnostics_preserve_missingness():
    rows = [
        {"dataset": "d", "condition": "c", "seed": seed, "layer": 0,
         "treatment": "shuffle_norm_matched", "target": "layer_0",
         "manifest_index": index, "branch_norm": value, "reference_branch_norm": reference}
        for seed, index, value, reference in [(11, 0, 1, 1), (11, 1, 3, 1), (23, 0, 0, 0)]
    ]
    result = diagnostic_estimates(rows, "message_norm_ratio")[0]
    assert result["mean"] == pytest.approx(2)
    assert result["count"] == 1 and result["std"] is None
    assert result["undefined_seeds"] == 1 and result["undefined_observations"] == 1
    assert diagnostic_value(rows[-1], "message_norm_ratio") is None


def test_real_figures_and_summary_label_posthoc_debug_and_actual_worse_change(tmp_path):
    values = fixture_rows()
    metadata = write_report(tmp_path, *values)
    summary = (tmp_path / "BRANCH_STRENGTH_SUMMARY.md").read_text(encoding="utf-8")
    assert "DEBUG" in summary and "source epochs=4" in summary and "final seeds=2" in summary
    assert "new epochs=0" in summary and "사후 탐색 관측" in summary
    assert "최적 개입 선택을 하지 않았다" in summary
    assert "identity_norm_matched | -2" in summary
    assert "분기 크기를 통제한" in summary and "전체 update의 norm" in summary
    assert "paired 95% t" in summary and "다중 비교 보정" in summary
    assert "기준 norm=0이면 undefined" in summary
    assert all(values[0]["expected_counts"][key] == count
               for key, count in metadata["report_counts"].items())
    assert "Completed seed cases" in summary and "| 2 | 2 | undefined | 0.1 |" in summary
    assert "순수 forward 처리량으로 해석하지 않는다" in summary
    for name in FIGURES:
        pixels = mpimg.imread(tmp_path / f"{name}.png")
        assert pixels.ndim == 3 and pixels.shape[0] > 500 and pixels.std() > 0.01
        assert (tmp_path / f"{name}.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("kind", [
    "missing_baseline", "missing_treatment", "missing_diagnostic", "missing_fixed_z",
    "duplicate", "nonfinite", "flag", "guard", "updates", "count", "source",
    "source_preserved", "source_epochs",
])
def test_invalid_measurements_fail_before_artifact_creation(tmp_path, kind):
    values = list(copy.deepcopy(fixture_rows()))
    if kind.startswith("missing_"):
        index = {"missing_baseline": 1, "missing_treatment": 2,
                 "missing_diagnostic": 3, "missing_fixed_z": 4}[kind]
        values[index].pop()
    elif kind == "duplicate":
        values[2].append(copy.deepcopy(values[2][0]))
    elif kind == "nonfinite":
        values[4][0]["beta_t_to_input"] = float("nan")
    elif kind == "flag":
        values[4][0]["branch_reference_cosine_defined"] = False
    elif kind == "guard":
        values[6]["models_unchanged"] = False
    elif kind == "source_preserved":
        values[6]["source_files_and_models_preserved"] = False
    elif kind == "source_epochs":
        values[6]["source_config"]["training"].pop("epochs_per_run")
    elif kind == "updates":
        values[6]["new_optimizer_updates"] = 1
    elif kind == "count":
        values[0]["expected_counts"]["fixed_z_rows"] += 1
    else:
        values[4][0]["source"] = "end_to_end"
    with pytest.raises(ValueError):
        write_report(tmp_path, *values)
    assert not list(tmp_path.iterdir())


def test_previous_report_artifacts_are_preserved(tmp_path):
    previous = tmp_path / f"{FIGURES[-1]}.pdf"
    previous.write_bytes(b"original scientific result")
    with pytest.raises(FileExistsError):
        write_report(tmp_path, *fixture_rows())
    assert previous.read_bytes() == b"original scientific result"
    assert len(list(tmp_path.iterdir())) == 1


def test_all_zero_reference_geometry_remains_undefined_and_still_renders(tmp_path):
    values = list(fixture_rows())
    for row in values[3] + values[4]:
        row["reference_branch_norm"] = row["branch_norm"] = 0.0
        row["branch_reference_cosine"] = None
        row["branch_reference_cosine_defined"] = False
    write_report(tmp_path, *values)
    summary = (tmp_path / "BRANCH_STRENGTH_SUMMARY.md").read_text(encoding="utf-8")
    assert "identity_norm_matched | undefined | undefined" in summary
    assert "2 / 2 |" in summary
