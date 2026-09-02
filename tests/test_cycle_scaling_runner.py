"""Contracts for the V1+V2 Cycle PE larger-model scaling runner."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from scripts import run_cycle_scaling as scaling


def _args(*extra: str):
    return scaling.parser().parse_args(list(extra))


def test_default_matrix_runs_both_versions_all_profiles_and_five_seeds(tmp_path):
    args = _args("--results-root", str(tmp_path))
    scaling._validate(args)
    jobs = scaling.make_jobs(args, scaling._run_dir(args, "matrix"))
    assert args.model_seeds == (0, 1, 2, 3, 4)
    assert len(jobs) == 2 * 4 * 5
    assert len(jobs) * len(args.datasets) == 80
    assert {(job["version"], job["profile"]) for job in jobs} == {
        (version, profile) for version in scaling.VERSIONS for profile in scaling.PROFILE_ORDER
    }
    assert all(job["datasets"] == ["zinc12k", "peptides_struct"] for job in jobs)
    assert len({job["output_dir"] for job in jobs}) == len(jobs)


@pytest.mark.parametrize("version", scaling.VERSIONS)
@pytest.mark.parametrize("profile", scaling.PROFILE_ORDER)
def test_child_commands_use_real_version_cli_and_exact_profile(version, profile, tmp_path):
    args = _args(
        "--versions",
        version,
        "--profiles",
        profile,
        "--model-seeds",
        "7",
        "--device",
        "cuda:0",
        "--results-root",
        str(tmp_path),
    )
    job = scaling.make_jobs(args, scaling._run_dir(args, "commands"))[0]
    command = job["command"]
    assert command[3:5] == ["-m", scaling.MODULES[version]]
    child_parser = (
        __import__("research.cycle_pe.benchmark", fromlist=["parser"]).parser
        if version == "v1"
        else __import__("research.cycle_pe.v2.benchmark", fromlist=["parser"]).parser
    )
    child = child_parser().parse_args(command[5:])
    expected = scaling.PROFILES[profile]
    assert (child.hidden_dim, child.pe_dim, child.layers) == (
        expected["hidden_dim"],
        expected["pe_dim"],
        expected["layers"],
    )
    assert child.datasets == ["zinc12k", "peptides_struct"]
    assert child.model_seed == 7 and child.device == "cuda:0"
    assert child.max_parameters == 5_000_000
    assert child.validation_only is True
    assert child.test_checkpoint is None
    if version == "v2":
        assert child.basis_execution == "batched" and child.basis_pair_budget == 32768


def test_profiles_include_independent_width_depth_and_combined_growth():
    assert scaling.PROFILES == {
        "base": {"hidden_dim": 64, "pe_dim": 32, "layers": 3},
        "wide": {"hidden_dim": 128, "pe_dim": 64, "layers": 3},
        "deep": {"hidden_dim": 64, "pe_dim": 32, "layers": 6},
        "large": {"hidden_dim": 128, "pe_dim": 64, "layers": 6},
    }


def test_direct_runner_environment_explicitly_unsets_nvml_cuda_check(monkeypatch):
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    assert "PYTORCH_NVML_BASED_CUDA_CHECK" not in scaling._environment()
    assert "src/chartgat/algebra.py" in scaling.SOURCE_FILES


def _row(version: str, dataset: str, profile: str, seed: int, validation: float):
    prefix = f"/{version}/{dataset}/{profile}/{seed}"
    return {
        "version": version,
        "profile": profile,
        "dataset": dataset,
        "model_seed": seed,
        "config": dict(scaling.PROFILES[profile]),
        "validation_mae": validation,
        "trainable_parameters": 100 if profile == "base" else 200,
        "elapsed_seconds": 10.0,
        "peak_gpu_memory_bytes": 1024,
        "best_epoch": 3,
        "epochs_completed": 5,
        "checkpoint": f"{prefix}/best.pt",
        "checkpoint_sha256": "a" * 64,
        "history": f"{prefix}/history.json",
        "history_sha256": "b" * 64,
        "output_dir": prefix,
    }


def test_profile_selection_uses_mean_validation_and_one_common_profile():
    rows = [
        _row("v1", "zinc12k", "base", 0, 0.10),
        _row("v1", "zinc12k", "large", 0, 0.20),
        _row("v1", "zinc12k", "base", 1, 0.30),
        _row("v1", "zinc12k", "large", 1, 0.30),
    ]
    summary = scaling.build_summary(
        rows,
        versions=["v1"],
        datasets=["zinc12k"],
        profiles=["base", "large"],
        model_seeds=(0, 1),
        complete=True,
    )
    assert summary["status"] == "pending_test_evaluation"
    assert [row["selected_profile"] for row in summary["profile_selections"]] == ["base"]
    assert len(summary["selected_checkpoints"]) == 2
    assert {row["selected_profile"] for row in summary["selected_checkpoints"]} == {"base"}
    assert summary["profile_selections"][0]["test_used_for_selection"] is False
    assert summary["test_evaluations"] == []
    assert all("test" not in key for row in summary["runs"] for key in row)


def test_incomplete_seed_matrix_withholds_selection_fail_closed():
    summary = scaling.build_summary(
        [_row("v2", "peptides_struct", "base", 0, 0.2)],
        versions=["v2"],
        datasets=["peptides_struct"],
        profiles=["base"],
        model_seeds=(0, 1),
        complete=True,
    )
    assert summary["status"] == "failed"
    assert summary["profile_selections"] == []
    assert summary["selected_checkpoints"] == []
    assert "selection_withheld" in summary


def test_selected_profile_creates_one_test_only_job_per_seed_and_attaches_separately(tmp_path):
    rows = [
        _row("v1", "zinc12k", profile, seed, value)
        for profile, values in (("base", (0.1, 0.2)), ("large", (0.4, 0.3)))
        for seed, value in enumerate(values)
    ]
    summary = scaling.build_summary(
        rows,
        versions=["v1"],
        datasets=["zinc12k"],
        profiles=["base", "large"],
        model_seeds=(0, 1),
        complete=True,
    )
    args = _args(
        "--versions",
        "v1",
        "--datasets",
        "zinc12k",
        "--profiles",
        "base",
        "large",
        "--model-seeds",
        "0,1",
    )
    jobs = scaling.make_test_jobs(args, tmp_path, summary["selected_checkpoints"])
    assert len(jobs) == 2
    assert all("--test-checkpoint" in job["command"] for job in jobs)
    assert all("--validation-only" not in job["command"] for job in jobs)
    test_rows = [
        {
            "test_evaluation_id": job["job_id"],
            "checkpoint_id": job["checkpoint_id"],
            "profile_selection_id": job["profile_selection_id"],
            "version": job["version"],
            "dataset": job["dataset"],
            "model_seed": job["model_seed"],
            "selected_profile": job["selected_profile"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "test_mae": 0.5 + job["model_seed"],
            "fresh_training": False,
        }
        for job in jobs
    ]
    final = scaling.attach_test_results(summary, test_rows, complete=True)
    assert final["status"] == "passed"
    assert len(final["profile_selections"]) == 1
    assert len(final["selected_checkpoints"]) == len(final["test_evaluations"]) == 2
    assert all("test_mae" not in row for row in final["selected_checkpoints"])


@pytest.mark.parametrize(
    "module_name",
    ["research.cycle_pe.benchmark", "research.cycle_pe.v2.benchmark"],
)
def test_validation_only_child_loads_no_test_split(module_name, tmp_path, monkeypatch):
    benchmark = importlib.import_module(module_name)
    loaded = []

    def fake_load(_root, _dataset, **kwargs):
        loaded.append(kwargs["splits"])
        return {"train": [], "validation": []}, {
            "loaded_splits": ["train", "validation"],
            "split_sizes": {"train": 1, "validation": 1},
            "split_content_sha256": {"train": "a", "validation": "b"},
        }

    monkeypatch.setattr(benchmark, "_validate", lambda _args: None)
    monkeypatch.setattr(benchmark, "load_benchmark", fake_load)
    monkeypatch.setattr(benchmark, "_train_model", lambda *_args: {"validation": 0.2})
    output = tmp_path / module_name.replace(".", "-")
    assert (
        benchmark.main(
            [
                "--datasets",
                "zinc12k",
                "--validation-only",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert loaded == [("train", "validation")]
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "test" not in metrics["datasets"]["zinc12k"]["models"][benchmark.MODEL_NAME]


def test_read_job_rows_accepts_only_validation_artifacts_and_rejects_test_leakage(tmp_path):
    output = tmp_path / "child"
    output.mkdir()
    job = {
        "version": "v2",
        "profile": "wide",
        "model_seed": 3,
        "datasets": ["zinc12k"],
        "config": dict(scaling.PROFILES["wide"]),
        "output_dir": str(output),
        "command": [
            "python",
            "-m",
            "research.cycle_pe.v2.benchmark",
            "--column-chunk-size",
            "16",
            "--basis-execution",
            "batched",
            "--basis-pair-budget",
            "32768",
        ],
    }
    run = output / "zinc12k/cycle_basis_v2"
    run.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    history = run / "history.json"
    history.write_text(json.dumps([{"epoch": epoch} for epoch in range(1, 5)]), encoding="utf-8")
    manifest = {
        "status": "passed",
        "run_mode": "validation_only",
        "version": "v2",
        "arguments": {
            "model_seed": 3,
            "hidden_dim": 128,
            "pe_dim": 64,
            "layers": 3,
            "datasets": ["zinc12k"],
            "validation_only": True,
            "test_checkpoint": None,
            "column_chunk_size": 16,
            "basis_execution": "batched",
            "basis_pair_budget": 32768,
        },
        "controls": {
            "test_data_access": False,
            "fresh_training": True,
            "optimizer_created": True,
        },
    }
    metrics = {
        "status": "passed",
        "run_mode": "validation_only",
        "model_seed": 3,
        "datasets": {
            "zinc12k": {
                "metric": "mae",
                "protocol": {
                    "loaded_splits": ["train", "validation"],
                    "split_sizes": {"train": 10, "validation": 2},
                    "split_content_sha256": {"train": "a", "validation": "b"},
                },
                "models": {
                    "cycle_basis_v2": {
                        "validation": 0.2,
                        "trainable_parameters": 1234,
                        "elapsed_seconds": 5.0,
                        "peak_gpu_memory_bytes": 2048,
                        "best_epoch": 2,
                        "epochs_completed": 4,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "history": str(history.resolve()),
                        "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
                        "evaluation_splits": ["train", "validation"],
                        "fresh_training": True,
                    }
                },
            }
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    row = scaling.read_job_rows(job)[0]
    assert row["validation_mae"] == 0.2 and "test_mae" not in row
    assert row["trainable_parameters"] == 1234
    metrics["datasets"]["zinc12k"]["models"]["cycle_basis_v2"]["test"] = 0.3
    (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="leaked a test metric"):
        scaling.read_job_rows(job)


def test_dry_run_prints_full_plan_without_dependency_or_gpu_checks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        scaling,
        "check_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("dry run must not check dependencies")),
    )
    code = scaling.main(
        [
            "--versions",
            "v1",
            "v2",
            "--profiles",
            "base",
            "large",
            "--model-seeds",
            "0,2",
            "--datasets",
            "zinc12k",
            "--results-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0 and "8 fresh child runs" in output and "8 fresh dataset trainings" in output
    assert "train+validation only" in output
    assert "4 selected-checkpoint test evaluations" in output
    assert not (tmp_path / "cycle_pe/scaling").exists()


def test_failed_child_leaves_failed_manifest_and_withholds_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(scaling, "check_dependencies", lambda: {"status": "passed"})
    monkeypatch.setattr(scaling, "_source_snapshot", lambda: {"source": "stable"})

    def fake_run(command, _log_path: Path, _environment):
        if "gpu_preflight.py" in " ".join(command):
            output = Path(command[command.index("--json-out") + 1])
            output.write_text('{"status":"passed"}', encoding="utf-8")
            return 0
        return 9

    monkeypatch.setattr(scaling, "run_logged", fake_run)
    code = scaling.main(
        [
            "--versions",
            "v1",
            "--profiles",
            "base",
            "--model-seeds",
            "0",
            "--datasets",
            "zinc12k",
            "--data-root",
            str(tmp_path / "data"),
            "--results-root",
            str(tmp_path),
            "--run-id",
            "failed-child",
        ]
    )
    run = tmp_path / "cycle_pe/scaling/failed-child"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert code == 1 and manifest["status"] == "failed"
    assert manifest["jobs"][0]["returncode"] == 9
    assert summary["status"] == "failed" and "selection_withheld" in summary
