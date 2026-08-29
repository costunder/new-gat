from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.gpu_preflight import (
    PROFILE_NAMES,
    PreflightError,
    ProfileConfig,
    _normalize_profiles,
    _paper_dependency_import_errors,
    build_report,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "gpu_preflight.py"


def test_cpu_preflight_exercises_incidence_path() -> None:
    report = build_report(
        "cpu",
        allow_cpu=True,
        require_paper_dependencies=False,
        min_free_gb=0.0,
    )
    assert report["status"] == "passed"
    assert report["resolved_device"] == "cpu"
    assert report["gpu"] is None
    assert report["profile_kind"] == "synthetic_shape_stress_not_dataset_e2e"
    assert report["selected_profiles"] == ["conductance"]
    assert set(report["profiles"]) == {"conductance"}
    assert report["incidence_forward_backward"]["loss"] > 0.0
    assert report["incidence_forward_backward"]["message_sum_abs"] < 1.0e-3
    assert report["profiles"]["conductance"]["peak_allocated"] == 0


def test_cpu_suite_profiles_cover_dense_forward_backward_and_brec_pinv() -> None:
    selected = ("cycle-projector", "tree-chart", "brec")
    report = build_report(
        "cpu",
        allow_cpu=True,
        require_paper_dependencies=False,
        min_free_gb=0.0,
        profiles=selected,
        profile_config=ProfileConfig(
            batch_size=3,
            brec_batch_size=3,
            nodes_per_graph=10,
            edges_per_graph=16,
            cycle_rank=5,
            cycle_variants=("no_pe", "raw", "set", "projector"),
            brec_protocol="custom",
            brec_amp=True,
        ),
    )
    assert tuple(report["profiles"]) == selected
    for profile in report["profiles"].values():
        assert profile["status"] == "passed"
        assert profile["loss"] >= 0.0
        assert profile["memory_unit"] == "bytes"
        assert profile["wall_time_unit"] == "seconds"
        assert profile["wall_time"] > 0.0
        assert {
            "allocated",
            "reserved",
            "peak_allocated",
            "peak_reserved",
        } <= set(profile)
    assert report["profiles"]["cycle-projector"]["spec"]["projector_shape_per_graph"] == [
        16,
        16,
    ]
    assert report["profiles"]["cycle-projector"]["spec"]["selected_variants"] == [
        "no_pe",
        "raw",
        "set",
        "projector",
    ]
    assert set(report["profiles"]["cycle-projector"]["variant_losses"]) == {
        "no_pe",
        "raw",
        "set",
        "projector",
    }
    assert report["profiles"]["tree-chart"]["spec"]["dense_chart_shape"] == [2, 16, 5]
    assert report["profiles"]["brec"]["spec"]["dtype"] == "float32"
    assert report["profiles"]["brec"]["spec"]["amp"] is False
    assert report["profiles"]["brec"]["spec"]["protocol"] == "custom"
    assert report["profiles"]["brec"]["spec"]["requested_batch_size"] == 3
    assert report["profiles"]["brec"]["spec"]["batch_size_graphs"] == 2
    assert set(report["profiles"]["brec"]["variant_losses"]) == {
        "no_pe",
        "raw",
        "set",
        "projector",
    }
    assert set(report["profiles"]["brec"]["variant_hotelling_t2"]) == {
        "no_pe",
        "raw",
        "set",
        "projector",
    }
    assert report["profiles"]["brec"]["spec"]["num_relabel_pairs_for_t2"] == 32


def test_profile_selection_and_shape_validation_are_fail_closed() -> None:
    assert _normalize_profiles(("all",)) == PROFILE_NAMES
    assert _normalize_profiles(("conductance", "conductance")) == ("conductance",)
    with pytest.raises(PreflightError, match="cannot be combined"):
        _normalize_profiles(("all", "conductance"))
    with pytest.raises(PreflightError, match="cycle-rank"):
        ProfileConfig(edges_per_graph=8, cycle_rank=9).validate()
    with pytest.raises(PreflightError, match="requires --brec-batch-size 16"):
        ProfileConfig(brec_batch_size=15).validate()
    with pytest.raises(PreflightError, match="requires --no-brec-amp"):
        ProfileConfig(brec_amp=True).validate()
    ProfileConfig(
        brec_batch_size=15,
        brec_protocol="custom",
        brec_amp=True,
    ).validate()


def test_gpu_preflight_refuses_cpu_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(
        sys,
        "argv",
        [str(PREFLIGHT), "--device", "cpu", "--min-free-gb", "0"],
    ):
        assert main() == 2
    assert "requires CUDA" in capsys.readouterr().out


def test_paper_dependency_check_detects_import_time_abi_failure() -> None:
    import scripts.gpu_preflight as preflight

    real_import = preflight.importlib.import_module

    def broken_scipy(name: str):
        if name == "scipy":
            raise OSError("undefined symbol from binary extension")
        return real_import(name)

    with (
        patch.dict(preflight.PAPER_IMPORTS, {"scipy": "scipy"}, clear=True),
        patch.object(preflight.importlib.util, "find_spec", return_value=object()),
        patch.object(preflight.importlib, "import_module", side_effect=broken_scipy),
    ):
        errors = _paper_dependency_import_errors()
    assert errors["scipy"].startswith("OSError: undefined symbol")
