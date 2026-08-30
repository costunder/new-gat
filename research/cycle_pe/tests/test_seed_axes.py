from __future__ import annotations

import json

import torch

from chartgat.seeds import SeedAxes
from research.cycle_pe import paper
from research.cycle_pe.paper import _seed_axis_policy, _settings, build_parser, main
from research.cycle_pe.tests.fixtures import small_cyclecount_loader


def test_cycle_settings_use_model_axis_and_record_not_applicable_axes() -> None:
    args = build_parser().parse_args(
        [
            "--seed",
            "7",
            "--data-seed",
            "11",
            "--split-seed",
            "13",
            "--chart-seed",
            "17",
            "--model-seed",
            "19",
        ]
    )
    settings = _settings(args, torch.device("cpu"), "core")
    assert settings.seed == 19

    axes = SeedAxes(data=11, split=13, chart=17, model=19)
    core = _seed_axis_policy("core", axes)
    assert core["data"]["used"] is True
    assert core["split"]["status"] == "not_applicable"
    assert core["chart"]["status"] == "not_applicable"
    assert core["model"]["used"] is True

    zinc = _seed_axis_policy("zinc", axes)
    assert zinc["split"]["status"] == "not_applicable"
    assert "official" in zinc["split"]["reason"]
    assert zinc["chart"]["status"] == "not_applicable"


def test_cyclecount_cache_identity_uses_data_seed_not_model_seed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paper, "load_or_generate_cycle_count_ood", small_cyclecount_loader)
    data_root = tmp_path / "data"
    manifests = []
    for model_seed in (31, 37):
        output_root = tmp_path / f"run-{model_seed}"
        assert (
            main(
                [
                    "--suite",
                    "core",
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(output_root),
                    "--device",
                    "cpu",
                    "--prepare-only",
                    "--seed",
                    "5",
                    "--data-seed",
                    "23",
                    "--split-seed",
                    "29",
                    "--chart-seed",
                    "30",
                    "--model-seed",
                    str(model_seed),
                ]
            )
            == 0
        )
        manifests.append(
            json.loads((output_root / "core" / "manifest.json").read_text(encoding="utf-8"))
        )

    first, second = manifests
    assert first["seed_axes"] == {"data": 23, "split": 29, "chart": 30, "model": 31}
    assert second["seed_axes"]["model"] == 37
    assert first["cache"] == second["cache"]
    assert first["dataset_metadata"]["seed"] == 23
    assert first["seed_axis_policy"]["split"]["used"] is False
    assert first["seed_axis_policy"]["chart"]["used"] is False


def test_brec_policy_uses_internal_protocol_seed_axis_only() -> None:
    axes = SeedAxes(data=1, split=2, chart=3, model=4)
    policy = _seed_axis_policy("brec", axes, brec_protocol="official", brec_seeds=(100, 200))
    assert policy["model"]["used"] is False
    assert policy["protocol"]["used"] is True
    assert policy["protocol"]["values"] == [100, 200]
