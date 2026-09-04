from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from research.cycle_pe import paper
from research.cycle_pe.paper_data import DatasetBundle, PaperGraph
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
            "--workers",
            "0",
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
    runtime = json.loads(
        (output_root / "core" / "edge" / "no_pe" / "runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime["optimizer_steps_completed"] == 2
    assert runtime["optimizer_steps_planned"] == 2
    assert runtime["gradient_connectivity"]["validated_on_first_actual_backward"] is True
    assert runtime["batch_observability"]["per_graph_gpu_forward_loop"] is False
    assert runtime["throughput"]["training_graphs_per_second"]["value"] > 0
    assert {
        "dataloader_wait_wall_seconds",
        "packed_h2d_seconds",
        "forward_and_loss_seconds",
        "backward_seconds",
        "optimizer_seconds",
    } <= set(runtime["phase_timing"])
    assert runtime["resource_observability"]["summary"][
        "run_average_gpu_sm_utilization_percent"
    ]["value"] is None
    assert runtime["resource_observability"]["summary"][
        "run_average_gpu_sm_utilization_percent"
    ]["reason"]
    assert runtime["evaluation_throughput"]["evaluated_graphs_per_second"]["value"] > 0


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


@pytest.mark.parametrize("overflow_split", ["validation", "id_test"])
def test_raw_overflow_skips_the_whole_condition_without_a_compatible_subset(
    monkeypatch, tmp_path, overflow_split
) -> None:
    triangle_edges = ((0, 1), (0, 2), (1, 2))
    rank_two_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))

    def graph(split: str, *, overflow: bool) -> PaperGraph:
        edges = rank_two_edges if overflow else triangle_edges
        return PaperGraph(
            graph_id=f"{split}:fixture",
            split=split,
            family="unit_test_fixture",
            num_nodes=4 if overflow else 3,
            edges=edges,
            edge_targets=np.zeros((len(edges), 1), dtype=np.float64),
        )

    splits = {
        split: [graph(split, overflow=split == overflow_split)]
        for split in ("train", "validation", "id_test", "size_ood", "family_ood")
    }
    bundle = DatasetBundle(
        name="CycleCount-OOD",
        splits=splits,
        edge_target_names=("edge_fixture",),
    )
    monkeypatch.setattr(
        paper,
        "load_or_generate_cycle_count_ood",
        lambda data_root, *, seed: bundle,
    )

    def unexpected_training(*args, **kwargs):
        raise AssertionError("an incompatible raw condition must not train")

    monkeypatch.setattr(paper, "train_supervised", unexpected_training)
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
                "--workers",
                "0",
                "--variants",
                "raw",
                "--core-targets",
                "edge",
                "--epochs",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "core" / "manifest.json").read_text("utf-8"))
    summary = manifest["experiments"]["edge"]["raw"]
    assert summary["status"] == "not_applicable_train_fitted_width_overflow"
    assert summary["training_performed"] is False
    assert summary["checkpoint_selection_performed"] is False
    assert summary["metric_calculation_performed"] is False
    assert summary["compatible_subset_used"] is False
    assert summary["incompatible_splits"][overflow_split]["overflow_graphs"] == 1
    assert summary["incompatible_splits"][overflow_split][
        "graphs_used_for_checkpoint_selection_or_metrics"
    ] == 0
    variant_root = output_root / "core" / "edge" / "raw"
    metrics = json.loads((variant_root / "metrics.json").read_text("utf-8"))
    assert metrics == summary
    assert not (variant_root / "history.json").exists()
    assert not (variant_root / "runtime.json").exists()
    assert not (variant_root / "model.pt").exists()


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
    assert pairs[0]["gradient_connectivity"]["validated_on_first_actual_backward"] is True
    assert pairs[0]["batch_observability"]["per_graph_gpu_forward_loop"] is False
    manifest = json.loads((output_root / "brec" / "manifest.json").read_text("utf-8"))
    assert manifest["runtime_summary"]["throughput"]["attempts_per_second"]["value"] > 0
    assert "resource_observability" in manifest["runtime_summary"]


def test_paper_source_identity_includes_resource_monitor_implementation() -> None:
    hashes = paper._implementation_hashes()
    assert "research/cycle_pe/paper.py" in hashes
    assert "research/cycle_pe/paper_train.py" in hashes
    assert "research/cycle_pe/paper_model.py" in hashes
    assert "research/cycle_pe/resource_monitor.py" in hashes
    assert "src/chartgat/observability.py" in hashes


def test_direct_paper_cli_defaults_to_parallel_data_loading() -> None:
    args = paper.build_parser().parse_args(["--suite", "core"])
    assert args.workers == 4
    assert args.prefetch_factor == 2


def test_supervised_evaluation_transfers_each_packed_level_not_each_graph() -> None:
    source = inspect.getsource(paper.evaluate_supervised)
    assert "prediction.detach().float().cpu()" in source
    assert "raw_target.detach().float().cpu()" in source
    assert ".cpu().numpy() for value in prediction_parts" not in source
    assert ".cpu().numpy() for value in target_parts" not in source


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


def test_all_suite_failure_preserves_completed_and_failed_suite_artifacts(
    monkeypatch, tmp_path, capsys
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
    assert sorted(path.name for path in output_root.iterdir()) == [
        "brec",
        "core",
        "run_manifest.json",
    ]
    assert (output_root / "core" / "partial.txt").read_text(encoding="utf-8") == "partial"
    assert (output_root / "brec" / "partial.txt").read_text(encoding="utf-8") == "partial"
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_suite"] == "brec"
    assert manifest["completed_suites"] == ["core"]
    assert manifest["preserved_failed_suite_output"] == str(output_root / "brec")
    first_error_line = capsys.readouterr().err.splitlines()[0]
    assert json.loads(first_error_line)["path"] == str(output_root / "brec")


@pytest.mark.parametrize("suite", ["core", "brec", "zinc"])
def test_production_cli_rejects_removed_tiny_option(suite) -> None:
    with pytest.raises(SystemExit):
        paper.build_parser().parse_args(["--suite", suite, "--tiny"])
