from __future__ import annotations

import json

import pytest
import torch

from research.cycle_pe import paper
from research.cycle_pe.tests.fixtures import small_cyclecount_loader, write_brec_fixture

main = paper.main


@pytest.fixture(autouse=True)
def unit_test_core_loader(monkeypatch) -> None:
    monkeypatch.setattr(paper, "load_or_generate_cycle_count_ood", small_cyclecount_loader)


def test_core_cli_trains_injected_data_and_writes_manifest(tmp_path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "runs"
    exit_code = main(
        [
            "--suite",
            "core",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_root),
            "--device",
            "cpu",
            "--seed",
            "11",
            "--data-seed",
            "19",
            "--model-seed",
            "37",
            "--epochs",
            "1",
            "--batch-size",
            "5",
            "--variants",
            "no_pe",
            "--core-targets",
            "edge",
        ]
    )
    assert exit_code == 0
    manifest_path = output_root / "core" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["raw_width"] == manifest["split_statistics"]["train"]["cycle_rank_max"]
    assert manifest["seed_axes"] == {"data": 19, "split": 19, "chart": 19, "model": 37}
    assert manifest["dataset_metadata"]["data_seed"] == 19
    assert manifest["seed_axis_policy"]["chart"]["status"] == "not_applicable"
    assert "'train' only" in manifest["raw_width_policy"]
    assert "never truncated" in manifest["raw_width_policy"]
    assert manifest["training"]["amp_effective"] is False
    assert manifest["training"]["workers"] == 0
    assert manifest["training"]["pin_memory_effective"] is False
    assert manifest["training"]["non_blocking_effective"] is False
    assert manifest["experiments"]["edge"]["no_pe"]["reported_split"] == "id_test"
    artifacts = manifest["artifacts"]
    assert "edge/no_pe/model.pt" in artifacts
    assert "edge/no_pe/metrics.json" in artifacts
    metrics = json.loads(
        (output_root / "core" / "edge" / "no_pe" / "metrics.json").read_text(encoding="utf-8")
    )
    assert set(metrics) == {
        "train",
        "validation",
        "id_test",
        "size_ood",
        "family_ood",
    }
    assert "edge_shortest_cycle" in metrics["size_ood"]["levels"]["edge"]["targets"]
    checkpoint = torch.load(
        output_root / "core" / "edge" / "no_pe" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["model_seed"] == 37


def test_core_prepare_only_stops_before_training(tmp_path) -> None:
    output_root = tmp_path / "runs"
    assert (
        main(
            [
                "--suite",
                "core",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--prepare-only",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "core" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_only"] is True
    assert manifest["variants"] == ["raw", "set", "projector"]
    assert manifest["experiments"] == {}
    assert manifest["runtime_environment"]["workers"] == 1
    assert not list((output_root / "core").glob("*/model.pt"))


def test_brec_prepare_only_uses_explicit_custom_artifact(tmp_path) -> None:
    write_brec_fixture(tmp_path / "data" / "BREC" / "Data" / "raw" / "brec_v3.npy")
    output_root = tmp_path / "runs"
    assert (
        main(
            [
                "--suite",
                "brec",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--brec-protocol",
                "custom",
                "--prepare-only",
                "--brec-num-relabel",
                "2",
                "--brec-threshold",
                "1",
                "--brec-seeds",
                "100,200",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "brec" / "manifest.json").read_text("utf-8"))
    assert manifest["dataset_metadata"]["pair_count"] == 2
    assert manifest["brec_protocol"]["effective"] == "custom"
    assert "official_training_reference_matched" not in manifest["brec_protocol"]
    compatibility = manifest["brec_protocol"]["official_reference_compatibility"]
    assert compatibility["static_constants_and_control_flow_compatible"] is False
    assert compatibility["differential_parity_verified"] is False
    assert manifest["rpc_reference"]["search_seeds"] == [100, 200]
    assert manifest["seed_axis_policy"]["model"]["used"] is False
    assert manifest["seed_axis_policy"]["protocol"]["values"] == [100, 200]
    assert len(manifest["preparation_checks"]) == 2


def test_brec_custom_training_is_labeled_separately(tmp_path) -> None:
    write_brec_fixture(tmp_path / "data" / "BREC" / "Data" / "raw" / "brec_v3.npy")
    output_root = tmp_path / "runs"
    assert (
        main(
            [
                "--suite",
                "brec",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--brec-protocol",
                "custom",
                "--brec-num-relabel",
                "2",
                "--brec-threshold",
                "1",
                "--brec-seeds",
                "100",
                "--variants",
                "no_pe",
                "--epochs",
                "1",
                "--batch-size",
                "4",
            ]
        )
        == 0
    )
    metrics = json.loads(
        (output_root / "brec" / "no_pe" / "metrics.json").read_text(encoding="utf-8")
    )
    pairs = json.loads((output_root / "brec" / "no_pe" / "pairs.json").read_text(encoding="utf-8"))
    assert metrics["protocol"] == "custom"
    assert metrics["metric_name"] == "custom_pairwise_union"
    assert pairs[0]["rng_scope"] == "derived_per_pair_variant_search_seed"
    assert pairs[0]["pair_shuffle"] is True
    assert pairs[0]["gradient_clip_norm"] == 5.0


def test_all_suite_forwards_prepare_and_download_policy(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, bool, bool]] = []

    def runner(name):
        def run(args, device):
            calls.append((name, args.prepare_only, args.allow_download))
            return {"variants": []}

        return run

    monkeypatch.setattr(paper, "run_core", runner("core"))
    monkeypatch.setattr(paper, "run_brec", runner("brec"))
    monkeypatch.setattr(paper, "run_zinc", runner("zinc"))
    assert (
        main(
            [
                "--suite",
                "all",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / "runs"),
                "--device",
                "cpu",
                "--prepare-only",
                "--allow-download",
            ]
        )
        == 0
    )
    assert calls == [
        ("core", True, True),
        ("brec", True, True),
        ("zinc", True, True),
    ]


def test_existing_output_collision_is_rejected_without_modification(tmp_path) -> None:
    output_root = tmp_path / "existing"
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("user artifact", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--suite",
                "core",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--prepare-only",
            ]
        )
    assert marker.read_text(encoding="utf-8") == "user artifact"
    assert list(output_root.iterdir()) == [marker]


def test_all_suite_failure_preserves_completed_and_removes_failed_suite_artifacts(
    monkeypatch, tmp_path
) -> None:
    def successful_core(args, device):
        partial = args.output_dir / "core" / "partial.txt"
        partial.parent.mkdir(parents=True)
        partial.write_text("partial", encoding="utf-8")
        return {"variants": []}

    def failing_brec(args, device):
        partial = args.output_dir / "brec" / "partial.txt"
        partial.parent.mkdir(parents=True)
        partial.write_text("partial", encoding="utf-8")
        raise RuntimeError("fixture BREC failure")

    def unexpected_zinc(args, device):
        raise AssertionError("ZINC must not run after BREC fails")

    monkeypatch.setattr(paper, "run_core", successful_core)
    monkeypatch.setattr(paper, "run_brec", failing_brec)
    monkeypatch.setattr(paper, "run_zinc", unexpected_zinc)
    output_root = tmp_path / "runs"
    with pytest.raises(SystemExit):
        main(
            [
                "--suite",
                "all",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--prepare-only",
            ]
        )
    assert sorted(path.name for path in output_root.iterdir()) == ["core", "run_manifest.json"]
    assert (output_root / "core" / "partial.txt").read_text(encoding="utf-8") == "partial"
    assert not (output_root / "brec").exists()
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_suite"] == "brec"
    assert manifest["completed_suites"] == ["core"]


@pytest.mark.parametrize("suite", ["core", "brec", "zinc"])
def test_production_cli_rejects_removed_tiny_option(suite) -> None:
    with pytest.raises(SystemExit):
        paper.build_parser().parse_args(["--suite", suite, "--tiny"])
