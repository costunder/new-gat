from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.aggregate_paper import _summary, aggregate_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("bootstrap_samples", [0, 100])
def test_singleton_summary_has_no_estimated_seed_uncertainty(bootstrap_samples: int) -> None:
    summary = _summary([0.625], key="single-seed", bootstrap_samples=bootstrap_samples)

    assert summary["n"] == 1
    assert summary["mean"] == summary["median"] == summary["minimum"] == summary["maximum"] == 0.625
    assert summary["uncertainty_status"] == "insufficient_samples"
    serialized = json.loads(json.dumps(summary, allow_nan=False))
    for key in ("sample_std", "bootstrap_95_low", "bootstrap_95_high"):
        assert serialized[key] is None


@pytest.mark.parametrize(
    "values,mean,std,low,high",
    [([0.0, 1.0], 0.5, 2**-0.5, 0.0, 1.0), ([0.3, 0.3], 0.3, 0.0, 0.3, 0.3)],
)
def test_explicit_multiple_seeds_keep_existing_estimated_statistics(values, mean, std, low, high):
    summary = _summary(values, key="multi-seed-regression", bootstrap_samples=100)

    assert summary["n"] == 2
    assert summary["mean"] == pytest.approx(mean)
    assert summary["sample_std"] == pytest.approx(std)
    assert summary["bootstrap_95_low"] == pytest.approx(low)
    assert summary["bootstrap_95_high"] == pytest.approx(high)
    assert summary["uncertainty_status"] == "bootstrap_estimated"


def test_disabled_bootstrap_keeps_multi_seed_statistics_but_is_explicitly_labelled() -> None:
    summary = _summary([0.0, 1.0], key="bootstrap-off", bootstrap_samples=0)

    assert summary["n"] == 2
    assert summary["sample_std"] == pytest.approx(2**-0.5)
    assert summary["bootstrap_95_low"] == summary["bootstrap_95_high"] == summary["mean"] == 0.5
    assert summary["uncertainty_status"] == "bootstrap_disabled"


@pytest.mark.parametrize(
    ("track", "dataset", "model"),
    [
        ("conductance_gat", "cora", "conductance"),
        ("cycle_pe", "zinc12k", "cycle_set"),
        ("cycle_pe", "zinc12k", "cycle_basis_v2"),
    ],
)
def test_benchmarks_aggregate_only_our_model_and_ignore_published_scores(
    tmp_path: Path,
    track: str,
    dataset: str,
    model: str,
) -> None:
    commands = []
    for seed in (0, 1):
        output = tmp_path / f"seed-{seed}"
        _write_json(
            output / "metrics.json",
            {
                "track": track,
                "suite": "benchmark",
                "datasets": {
                    dataset: {
                        "models": {
                            model: {
                                "test": 0.1 + seed * 0.01,
                                "validation": 0.05,
                                "best_epoch": 15,
                                "trainable_parameters": 1000,
                                "elapsed_seconds": 3.0,
                                "peak_gpu_memory_bytes": 2048,
                                "history": [{"test": 0.001, "validation": 0.002}],
                            },
                            "external_model": {"test": 0.5, "elapsed_seconds": 8.0},
                        },
                        "published_reference": {"test": 0.4, "std": 0.02},
                        "baselines": {"gat": {"test": 0.3}, "signnet": {"test": 0.2}},
                    },
                },
            },
        )
        commands.append(
            {
                "name": f"{track}:benchmark:model-seed-{seed}",
                "command": [
                    "python",
                    "--suite",
                    "benchmark",
                    "--model-seed",
                    str(seed),
                    "--data-seed",
                    "0",
                    "--split-seed",
                    "0",
                    "--chart-seed",
                    "0",
                ],
                "returncode": 0,
                "artifact_errors": [],
                "output": str(output),
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"run_id": "matched", "status": "passed", "commands": commands})
    result = aggregate_manifest(manifest, bootstrap_samples=0)
    assert result["metric_groups"] == 1
    assert result["sample_rows"] == 2
    assert result["efficiency_rows"] == 6
    assert result["ignored_numeric_fields"] > 0
    with (tmp_path / "aggregate" / "paired.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert pairs == []


def test_cycle_v1_and_basis_v2_keep_independent_summary_and_efficiency_rows(
    tmp_path: Path,
) -> None:
    commands = []
    for seed in (0, 1):
        output = tmp_path / f"seed-{seed}"
        models = {}
        for model, offset in (("cycle_set", 0.1), ("cycle_basis_v2", 0.7)):
            models[model] = {
                "test": offset + seed * 0.02,
                "validation": 9.0,
                "trainable_parameters": 1000,
                "elapsed_seconds": 3.0,
                "peak_gpu_memory_bytes": 2048,
                "history": [{"test": 8.0, "validation": 7.0}],
            }
        models["external_model"] = {"test": 6.0, "elapsed_seconds": 5.0}
        _write_json(
            output / "metrics.json",
            {
                "track": "cycle_pe",
                "suite": "benchmark",
                "datasets": {
                    "zinc12k": {
                        "models": models,
                        "published_reference": {"test": 4.0, "std": 0.02},
                    }
                },
            },
        )
        commands.append(
            {
                "name": f"cycle_pe:benchmark:model-seed-{seed}",
                "command": [
                    "python",
                    "--suite",
                    "benchmark",
                    "--model-seed",
                    str(seed),
                    "--data-seed",
                    "0",
                    "--split-seed",
                    "0",
                    "--chart-seed",
                    "0",
                ],
                "returncode": 0,
                "artifact_errors": [],
                "output": str(output),
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"run_id": "both-versions", "status": "passed", "commands": commands})

    result = aggregate_manifest(manifest, bootstrap_samples=0)

    assert result["sample_rows"] == 4
    assert result["metric_groups"] == 2
    assert result["efficiency_rows"] == 12
    assert result["paired_groups"] == 0
    assert result["ignored_numeric_fields"] == result["numeric_fields_seen"] - 16
    with (tmp_path / "aggregate" / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        summaries = {row["metric"]: row for row in csv.DictReader(stream)}
    assert set(summaries) == {
        "datasets.zinc12k.models.cycle_set.test",
        "datasets.zinc12k.models.cycle_basis_v2.test",
    }
    for model, expected_mean, expected_rule in (
        ("cycle_set", 0.11, "cycle.our_model.test"),
        ("cycle_basis_v2", 0.71, "cycle.basis_v2.test"),
    ):
        row = summaries[f"datasets.zinc12k.models.{model}.test"]
        assert float(row["mean"]) == pytest.approx(expected_mean)
        assert row["model_seeds"] == "0,1"
        assert row["metric_rule"] == expected_rule
    with (tmp_path / "aggregate" / "efficiency.csv").open(encoding="utf-8", newline="") as stream:
        efficiency = list(csv.DictReader(stream))
    assert {row["metric_rule"] for row in efficiency} == {
        "cycle.our_model.efficiency",
        "cycle.basis_v2.efficiency",
    }
    assert {row["metric"] for row in efficiency} == {
        f"datasets.zinc12k.models.{model}.{metric}"
        for model in ("cycle_set", "cycle_basis_v2")
        for metric in ("trainable_parameters", "elapsed_seconds", "peak_gpu_memory_bytes")
    }
    with (tmp_path / "aggregate" / "paired.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []


def test_aggregate_keeps_data_axes_fixed_and_pairs_model_seeds(tmp_path: Path) -> None:
    commands = []
    for model_seed, full, edge in ((1, 0.2, 0.5), (2, 0.4, 0.7)):
        output = tmp_path / f"seed-{model_seed}"
        _write_json(
            output / "summary.json",
            {
                "configuration": {"epochs": 100, "batch_size": 16},
                "runtime": {"elapsed_seconds": 3.0},
                "seed_axes": {"data": 11, "split": 13, "chart": 17, "model": model_seed},
                "results": {
                    "core": {
                        "s1": {
                            "baselines": {
                                "full": {
                                    "unseen_graph_test": {
                                        "graph_macro_flux_relative_l2": full,
                                        "num_examples": 20,
                                    }
                                },
                                "edge_only": {
                                    "unseen_graph_test": {
                                        "graph_macro_flux_relative_l2": edge,
                                        "num_examples": 20,
                                    }
                                },
                            }
                        }
                    },
                    "public": {
                        "pascalvoc_sp": {
                            "baselines": {
                                "conductance_model": {
                                    "parameter_count": 1_234,
                                    "parameter_count_policy": "trainable_active_parameters_only",
                                },
                                "gcn": {
                                    "parameter_count": 9_999,
                                    "parameter_count_policy": "all_constructed_parameters",
                                },
                            }
                        }
                    },
                },
            },
        )
        commands.append(
            {
                "name": f"conductance_gat:core:model-seed-{model_seed}",
                "command": [
                    "python",
                    "-m",
                    "research.conductance_gat.paper",
                    "--suite",
                    "core",
                    "--model-seed",
                    str(model_seed),
                    "--data-seed",
                    "11",
                    "--split-seed",
                    "13",
                    "--chart-seed",
                    "17",
                ],
                "returncode": 0,
                "artifact_errors": [],
                "output": str(output),
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"run_id": "fixture", "status": "passed", "commands": commands})

    payload = aggregate_manifest(manifest, bootstrap_samples=100)

    assert payload["failed_commands"] == 0
    assert payload["metric_groups"] == 2
    assert payload["sample_rows"] == 4
    assert payload["efficiency_rows"] == 4
    assert payload["ignored_numeric_fields"] > 0
    with (tmp_path / "aggregate" / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    full_row = next(
        row
        for row in rows
        if row["metric"]
        == "results.core.s1.baselines.full.unseen_graph_test.graph_macro_flux_relative_l2"
    )
    assert float(full_row["mean"]) == pytest.approx(0.3)
    assert full_row["data_seed"] == "11"
    assert full_row["model_seeds"] == "1,2"
    with (tmp_path / "aggregate" / "paired.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["condition_left"] == "edge_only"
    assert pair["condition_right"] == "full"
    assert float(pair["mean"]) == pytest.approx(-0.3)
    assert pair["difference_definition"] == "right_minus_left"
    assert pair["effect_size_name"] == "paired_cohens_dz"
    assert pair["effect_size"] == ""
    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        sample = next(csv.DictReader(stream))
    assert Path(sample["artifact_path"]).is_absolute()
    with (tmp_path / "aggregate" / "efficiency.csv").open(encoding="utf-8", newline="") as stream:
        efficiency = list(csv.DictReader(stream))
    assert {row["metric"] for row in efficiency} == {
        "runtime.elapsed_seconds",
        "results.public.pascalvoc_sp.baselines.conductance_model.parameter_count",
    }
    assert all("batch_size" not in row["metric"] for row in efficiency)


def test_aggregate_preserves_failures_and_legacy_seed_axes(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    _write_json(
        output / "core" / "edge" / "no_pe" / "metrics.json",
        {
            "id_test": {"macro_normalized_mae": 1.25, "graphs": 20},
            "train": {"macro_normalized_mae": 0.25},
        },
    )
    _write_json(
        output / "core" / "edge" / "no_pe" / "runtime.json",
        {
            "total_train_evaluation_wall_seconds": 5.0,
            "peak_gpu_memory_bytes": 2_048,
            "batch_size": 16,
            "epochs_completed": 20,
        },
    )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "run_id": "legacy",
            "status": "failed",
            "commands": [
                {
                    "name": "cycle_pe:core:seed-7",
                    "command": ["python", "--suite", "core", "--seed", "7"],
                    "returncode": 0,
                    "artifact_errors": [],
                    "output": str(output),
                },
                {
                    "name": "tree_augmentation:seed-8",
                    "command": ["python", "--suite", "core", "--seed", "8"],
                    "returncode": 1,
                    "artifact_errors": ["CUDA out of memory"],
                    "output": str(tmp_path / "missing"),
                },
            ],
        },
    )

    payload = aggregate_manifest(manifest, bootstrap_samples=0)

    assert payload["failed_commands"] == 1
    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        sample = next(csv.DictReader(stream))
    assert sample["model_seed"] == sample["data_seed"] == sample["split_seed"] == "7"
    assert sample["metric"] == "id_test.macro_normalized_mae"
    assert payload["efficiency_rows"] == 2
    with (tmp_path / "aggregate" / "failures.csv").open(encoding="utf-8", newline="") as stream:
        failure = next(csv.DictReader(stream))
    assert failure["oom"] == "True"


def test_aggregate_reads_oom_logs_and_ignores_outer_seed_for_official_brec(
    tmp_path: Path,
) -> None:
    brec_output = tmp_path / "brec"
    _write_json(
        brec_output / "brec" / "no_pe" / "metrics.json",
        {
            "protocol": "official",
            "global_valid": True,
            "per_seed": {
                "100": {
                    "pairs_expected": 400,
                    "Correct": 12,
                    "Fail": 0,
                    "Real_correct": 12,
                }
            },
        },
    )
    oom_log = tmp_path / "oom.log"
    oom_log.write_text("torch.OutOfMemoryError: CUDA out of memory", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "run_id": "brec-and-oom",
            "status": "failed",
            "commands": [
                {
                    "name": "cycle_pe:brec:official-10-seed",
                    "command": [
                        "python",
                        "--suite",
                        "brec",
                        "--brec-protocol",
                        "official",
                        "--model-seed",
                        "0",
                        "--data-seed",
                        "3",
                    ],
                    "returncode": 0,
                    "artifact_errors": [],
                    "output": str(brec_output),
                },
                {
                    "name": "conductance_gat:model-seed-4",
                    "command": ["python", "--suite", "core", "--model-seed", "4"],
                    "returncode": 1,
                    "artifact_errors": [],
                    "log": str(oom_log),
                    "output": str(tmp_path / "missing"),
                },
            ],
        },
    )

    aggregate_manifest(manifest, bootstrap_samples=0)

    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 3
    assert {row["metric"] for row in samples} == {
        "per_seed.100.Correct",
        "per_seed.100.Fail",
        "per_seed.100.Real_correct",
    }
    assert all(row["model_seed"] == "" for row in samples)
    assert all(row["pairable"] == "False" for row in samples)
    with (tmp_path / "aggregate" / "failures.csv").open(encoding="utf-8", newline="") as stream:
        failure = next(csv.DictReader(stream))
    assert failure["oom"] == "True"


@pytest.mark.parametrize("seed_count", [1, 2])
def test_aggregate_tree_schema_pairs_only_registered_downstream_metrics(
    tmp_path: Path, seed_count: int
) -> None:
    commands = []
    for model_seed, fixed, multi in ((1, 0.8, 0.5), (2, 0.6, 0.4))[:seed_count]:
        output = tmp_path / f"tree-{model_seed}"
        _write_json(
            output / "summary.json",
            {
                "settings": {
                    "optimizer_updates": 100,
                    "batch_size": 32,
                    "seed_axes": {"model": model_seed},
                },
                "runtime": {"elapsed_seconds": 4.0, "peak_gpu_allocated_bytes": 1024},
                "models": {
                    "fixed_bfs": {
                        "optimizer_updates": 100,
                        "history": [{"update": 1, "loss": 2.0}],
                        "quadrants": {"ood": {"mae": fixed, "num_examples": 10}},
                    },
                    "multi_chart": {
                        "optimizer_updates": 100,
                        "history": [{"update": 1, "loss": 1.0}],
                        "quadrants": {"ood": {"mae": multi, "num_examples": 10}},
                    },
                },
                "comparison": {
                    "quadrant_improvements": {
                        "ood": {"mae_improvement_fixed_minus_multi": fixed - multi}
                    }
                },
                "diagnostics": {"mae": 999.0},
            },
        )
        commands.append(
            {
                "name": f"tree_augmentation:core:model-seed-{model_seed}",
                "command": [
                    "python",
                    "--suite",
                    "core",
                    "--model-seed",
                    str(model_seed),
                    "--data-seed",
                    "11",
                    "--split-seed",
                    "13",
                    "--chart-seed",
                    "17",
                ],
                "returncode": 0,
                "artifact_errors": [],
                "output": str(output),
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"run_id": "tree-schema", "status": "passed", "commands": commands})

    payload = aggregate_manifest(manifest, bootstrap_samples=0)

    assert payload["schema_version"] == 3
    assert payload["sample_rows"] == 3 * seed_count
    assert payload["efficiency_rows"] == 2 * seed_count
    assert payload["metric_groups"] == 3
    assert payload["paired_groups"] == 1
    assert payload["ignored_numeric_fields"] == payload["numeric_fields_seen"] - 5 * seed_count
    assert "not confidence intervals" in payload["uncertainty_policy"]["bootstrap_disabled"]
    for filename in ("metrics.csv", "paired.csv"):
        with (tmp_path / "aggregate" / filename).open(encoding="utf-8", newline="") as stream:
            summaries = list(csv.DictReader(stream))
        for row in summaries:
            assert int(row["n"]) == seed_count
            if seed_count == 1:
                assert row["uncertainty_status"] == "insufficient_samples"
                for key in ("sample_std", "bootstrap_95_low", "bootstrap_95_high"):
                    assert row[key] == ""
                if filename == "paired.csv":
                    assert row["effect_size"] == ""
            else:
                assert row["uncertainty_status"] == "bootstrap_disabled"
                assert row["sample_std"] != ""
                assert row["bootstrap_95_low"] == row["bootstrap_95_high"] == row["mean"]
                if filename == "paired.csv":
                    assert float(row["effect_size"]) == pytest.approx(
                        float(row["mean"]) / float(row["sample_std"])
                    )
    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert all("history" not in row["metric"] for row in samples)
    assert all("runtime" not in row["metric"] for row in samples)
    assert all("settings" not in row["metric"] for row in samples)
    improvements = [row for row in samples if row["metric_rule"] == "tree.precomputed_improvement"]
    assert len(improvements) == seed_count
    assert all(row["pairable"] == "False" for row in improvements)
    with (tmp_path / "aggregate" / "efficiency.csv").open(encoding="utf-8", newline="") as stream:
        efficiency = list(csv.DictReader(stream))
    assert {row["metric"] for row in efficiency} == {
        "runtime.elapsed_seconds",
        "runtime.peak_gpu_allocated_bytes",
    }
    assert all("batch_size" not in row["metric"] for row in efficiency)
