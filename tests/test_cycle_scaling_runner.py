"""Contracts for the V1+V2 Cycle PE larger-model scaling runner."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from chartgat.observability import finalize_resource_observability, runtime_resource_snapshot
from scripts import run_cycle_scaling as scaling


def _args(*extra: str):
    return scaling.parser().parse_args(list(extra))


def _resource_observability() -> dict:
    device = torch.device("cpu")
    return finalize_resource_observability(
        runtime_resource_snapshot(device),
        device,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        sample_interval_seconds=1.0,
    )


def test_default_matrix_runs_both_versions_all_profiles_and_seed_zero(tmp_path):
    args = _args("--results-root", str(tmp_path))
    scaling._validate(args)
    jobs = scaling.make_jobs(args, scaling._run_dir(args, "matrix"))
    assert args.model_seeds == (0,)
    assert len(jobs) == 8
    assert len(jobs) == 8
    manifest = scaling._manifest_base(
        args,
        "matrix",
        scaling._run_dir(args, "matrix"),
        jobs,
        {"status": "passed"},
        {"source": "stable"},
    )
    assert manifest["fresh_child_runs"] == 8
    assert manifest["fresh_dataset_trainings"] == 8
    assert manifest["selected_test_evaluations_planned"] == 4
    assert {(job["version"], job["profile"]) for job in jobs} == {
        (version, profile) for version in scaling.VERSIONS for profile in scaling.PROFILE_ORDER
    }
    assert {tuple(job["datasets"]) for job in jobs} == {
        ("zinc12k",),
        ("peptides_struct",),
    }
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
    dataset = job["datasets"][0]
    expected = scaling._profile_config(profile, dataset, version)
    assert (child.hidden_dim, child.pe_dim, child.layers) == (
        expected["hidden_dim"],
        expected["pe_dim"],
        expected["layers"],
    )
    assert child.datasets == [dataset]
    assert child.model_seed == 7 and child.device == "cuda:0"
    assert child.max_parameters == 50_000_000
    assert child.validation_only is True
    assert child.test_checkpoint is None
    if version == "v2":
        assert child.basis_backend == "thin_q"
        assert child.basis_execution == "batched" and child.basis_pair_budget == 32768


def test_dfs_basis_backend_is_forwarded_only_to_v2_and_bound_to_run_identity(tmp_path):
    args = _args(
        "--versions",
        "v1",
        "v2",
        "--profiles",
        "reference",
        "--datasets",
        "zinc12k",
        "--basis-backend",
        "dfs_fundamental",
        "--results-root",
        str(tmp_path),
    )
    scaling._validate(args)
    jobs = scaling.make_jobs(args, scaling._run_dir(args, "dfs"))
    v1, v2 = sorted(jobs, key=lambda job: job["version"])
    assert "--basis-backend" not in v1["command"]
    assert v1["resources"]["basis_backend"] is None
    assert v2["command"][v2["command"].index("--basis-backend") + 1] == "dfs_fundamental"
    assert v2["resources"]["basis_backend"] == "dfs_fundamental"
    assert scaling._run_configuration(args)["basis_backend"] == "dfs_fundamental"

    v1_only = _args("--versions", "v1", "--basis-backend", "dfs_fundamental")
    with pytest.raises(ValueError, match="requires v2"):
        scaling._validate(v1_only)


def test_profiles_are_dataset_aware_reference_scale_and_large():
    assert scaling.PROFILES == {
        "reference": {
            "zinc12k": {
                "hidden_dim": 128,
                "pe_dim": 64,
                "layers": 10,
                "ffn_multiplier": 4,
                "dropout": 0.1,
                "layer_scale": 0.1,
            },
            "peptides_struct": {
                "hidden_dim": 256,
                "pe_dim": 64,
                "layers": 6,
                "ffn_multiplier": 4,
                "dropout": 0.1,
                "layer_scale": 0.1,
            },
        },
        "large": {
            "zinc12k": {
                "hidden_dim": 192,
                "pe_dim": 96,
                "layers": 12,
                "ffn_multiplier": 4,
                "dropout": 0.1,
                "layer_scale": 0.1,
            },
            "peptides_struct": {
                "hidden_dim": 320,
                "pe_dim": 96,
                "layers": 8,
                "ffn_multiplier": 4,
                "dropout": 0.1,
                "layer_scale": 0.1,
            },
        },
    }


def test_a6000_profile_uses_static_dataset_batches_amp_and_heavy_first(tmp_path):
    args = _args(
        "--hardware-profile",
        "a6000-48gb",
        "--results-root",
        str(tmp_path),
    )
    jobs = scaling.make_jobs(args, scaling._run_dir(args, "a6000"))
    assert jobs[0]["version"] == "v2"
    assert jobs[0]["profile"] == "large"
    assert jobs[0]["datasets"] == ["peptides_struct"]
    expected_batches = {
        ("reference", "zinc12k"): 512,
        ("reference", "peptides_struct"): 128,
        ("large", "zinc12k"): 256,
        ("large", "peptides_struct"): 64,
    }
    for job in jobs:
        resources = job["resources"]
        dataset = job["datasets"][0]
        assert resources["batch_size"] == expected_batches[job["profile"], dataset]
        assert resources["workers"] == 8
        assert resources["amp"] is True
        command = job["command"]
        assert command[command.index("--batch-size") + 1] == str(resources["batch_size"])
        assert "--amp" in command
        if job["version"] == "v2":
            assert resources["prefetch_factor"] == 4
            assert resources["basis_pair_budget"] == 4_194_304
            assert command[command.index("--hardware-profile") + 1] == "a6000-48gb"
        else:
            assert resources["prefetch_factor"] == 2
            assert resources["basis_pair_budget"] is None
            assert "--prefetch-factor" not in command


def test_v2_precision_contract_accepts_only_hardware_valid_modes() -> None:
    disabled_resources = scaling._job_resources(_args(), "v2", "reference", "zinc12k")
    disabled = {
        "amp": False,
        "amp_effective": False,
        "autocast_dtype": "disabled",
        "fallback": None,
        "gradient_scaler": False,
        "projector_contraction": "float32",
        "backbone_autocast": False,
    }
    portable_resources = scaling._job_resources(_args("--amp"), "v2", "reference", "zinc12k")
    bf16 = {
        **disabled,
        "amp": True,
        "amp_effective": True,
        "autocast_dtype": "bfloat16",
        "backbone_autocast": True,
    }
    fallback = {
        **disabled,
        "amp": True,
        "fallback": "bf16_unavailable_use_fp32",
    }
    a6000_resources = scaling._job_resources(
        _args("--hardware-profile", "a6000-48gb"),
        "v2",
        "reference",
        "zinc12k",
    )

    assert scaling._v2_precision_matches(disabled_resources, disabled)
    assert scaling._v2_precision_matches(portable_resources, bf16)
    assert scaling._v2_precision_matches(portable_resources, fallback)
    assert scaling._v2_precision_matches(a6000_resources, bf16)
    assert not scaling._v2_precision_matches(a6000_resources, fallback)

    corrupt_values = {
        "amp": False,
        "amp_effective": False,
        "autocast_dtype": "float16",
        "fallback": "tampered",
        "gradient_scaler": True,
        "projector_contraction": "float16",
        "backbone_autocast": False,
    }
    for field, value in corrupt_values.items():
        corrupt = dict(bf16)
        corrupt[field] = value
        assert not scaling._v2_precision_matches(a6000_resources, corrupt), field
    for field in tuple(bf16):
        incomplete = dict(bf16)
        del incomplete[field]
        assert not scaling._v2_precision_matches(a6000_resources, incomplete), field
    extra = {**bf16, "unregistered_precision_field": True}
    assert not scaling._v2_precision_matches(a6000_resources, extra)


def test_both_v2_artifact_admission_paths_enforce_full_precision_contract() -> None:
    assert "_v2_precision_matches" in inspect.getsource(scaling.read_job_rows)
    assert "_v2_precision_matches" in inspect.getsource(scaling.read_test_result)


@pytest.mark.parametrize(
    "gpu,passes",
    [
        ({"total_bytes": 48 * 1024**3, "compute_capability": [8, 6]}, True),
        ({"total_bytes": 10 * 1024**3, "compute_capability": [8, 0]}, False),
        ({"total_bytes": 48 * 1024**3, "compute_capability": [7, 5]}, False),
    ],
)
def test_a6000_capacity_guard_is_capacity_based_not_name_based(gpu, passes):
    report = {"gpu": {"name": "arbitrary", **gpu}}
    if passes:
        scaling._validate_preflight_hardware("a6000-48gb", report)
    else:
        with pytest.raises(ValueError, match="40 GiB"):
            scaling._validate_preflight_hardware("a6000-48gb", report)
    scaling._validate_preflight_hardware("portable", report)


def test_v2_test_only_admission_binds_every_a6000_runtime_field():
    source = inspect.getsource(scaling.read_test_result) + inspect.getsource(
        scaling._v2_precision_matches
    )
    for field in (
        "ffn_multiplier",
        "column_chunk_size",
        "basis_pair_budget",
        "prefetch_factor",
        "hardware_profile",
        "effective_batch_size",
        "packed_cycle_basis_h2d_tensors_per_batch",
        "peak_gpu_reserved_bytes",
        "projector_contraction",
    ):
        assert field in source


def test_direct_runner_environment_explicitly_unsets_nvml_cuda_check(monkeypatch):
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    assert "PYTORCH_NVML_BASED_CUDA_CHECK" not in scaling._environment()
    assert "src/chartgat/algebra.py" in scaling.SOURCE_FILES
    assert "src/chartgat/graphs.py" in scaling.SOURCE_FILES
    assert "research/__init__.py" in scaling.SOURCE_FILES
    assert "research/cycle_pe/__init__.py" in scaling.SOURCE_FILES
    assert "research/cycle_pe/v2/__init__.py" in scaling.SOURCE_FILES
    assert "research/cycle_pe/paper_data.py" in scaling.SOURCE_FILES
    assert "research/cycle_pe/resource_monitor.py" in scaling.SOURCE_FILES
    assert "src/chartgat/__init__.py" in scaling.SOURCE_FILES
    assert "scripts/gpu_profiles.py" in scaling.SOURCE_FILES
    assert "scripts/telemetry_validation.py" in scaling.SOURCE_FILES
    assert "scripts/verify_gpu_lock.py" in scaling.SOURCE_FILES
    assert "src/chartgat/observability.py" in scaling.SOURCE_FILES


def _row(version: str, dataset: str, profile: str, seed: int, validation: float):
    prefix = f"/{version}/{dataset}/{profile}/{seed}"
    return {
        "version": version,
        "profile": profile,
        "dataset": dataset,
        "model_seed": seed,
        "config": scaling._profile_config(profile, dataset, version),
        "validation_mae": validation,
        "trainable_parameters": 100 if profile == "reference" else 200,
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
        _row("v1", "zinc12k", "reference", 0, 0.10),
        _row("v1", "zinc12k", "large", 0, 0.20),
        _row("v1", "zinc12k", "reference", 1, 0.30),
        _row("v1", "zinc12k", "large", 1, 0.30),
    ]
    summary = scaling.build_summary(
        rows,
        versions=["v1"],
        datasets=["zinc12k"],
        profiles=["reference", "large"],
        model_seeds=(0, 1),
        complete=True,
    )
    assert summary["status"] == "pending_test_evaluation"
    assert [row["selected_profile"] for row in summary["profile_selections"]] == ["reference"]
    assert len(summary["selected_checkpoints"]) == 2
    assert {row["selected_profile"] for row in summary["selected_checkpoints"]} == {"reference"}
    assert summary["profile_selections"][0]["test_used_for_selection"] is False
    assert summary["test_evaluations"] == []
    assert all("test" not in key for row in summary["runs"] for key in row)


def test_incomplete_seed_matrix_withholds_selection_fail_closed():
    summary = scaling.build_summary(
        [_row("v2", "peptides_struct", "reference", 0, 0.2)],
        versions=["v2"],
        datasets=["peptides_struct"],
        profiles=["reference"],
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
        for profile, values in (("reference", (0.1, 0.2)), ("large", (0.4, 0.3)))
        for seed, value in enumerate(values)
    ]
    summary = scaling.build_summary(
        rows,
        versions=["v1"],
        datasets=["zinc12k"],
        profiles=["reference", "large"],
        model_seeds=(0, 1),
        complete=True,
    )
    args = _args(
        "--versions",
        "v1",
        "--datasets",
        "zinc12k",
        "--profiles",
        "reference",
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
        "profile": "reference",
        "model_seed": 3,
        "datasets": ["zinc12k"],
        "config": scaling._profile_config("reference", "zinc12k", "v2"),
        "resources": scaling._job_resources(_args(), "v2", "reference", "zinc12k"),
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
    run = output / "zinc12k/cycle_projector_pe_v2"
    run.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    history = run / "history.json"
    history.write_text(
        json.dumps(
            [
                {
                    "epoch": epoch,
                    "train_graphs": 10,
                    "train_cuda_synchronized_seconds": 1.0,
                    "train_graphs_per_second": 10.0,
                }
                for epoch in range(1, 5)
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "passed",
        "run_mode": "validation_only",
        "version": "v2",
        "arguments": {
            "model_seed": 3,
            "hidden_dim": 128,
            "pe_dim": 64,
            "layers": 10,
            "ffn_multiplier": 4,
            "dropout": 0.1,
            "layer_scale": 0.1,
            "datasets": ["zinc12k"],
            "validation_only": True,
            "test_checkpoint": None,
            "column_chunk_size": 16,
            "basis_backend": "thin_q",
            "basis_execution": "batched",
            "basis_pair_budget": 32768,
            "batch_size": 32,
            "workers": 4,
            "amp": False,
            "prefetch_factor": 2,
            "hardware_profile": "portable",
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
                    "cycle_projector_pe_v2": {
                        "validation": 0.2,
                        "trainable_parameters": 1234,
                        "elapsed_seconds": 5.0,
                        "peak_gpu_memory_bytes": 2048,
                        "peak_gpu_reserved_bytes": 4096,
                        "effective_batch_size": 32,
                        "execution": {
                            "data_pipeline": {
                                "effective_batch_size": 32,
                                "workers": 4,
                                "prefetch_factor": 2,
                                "packed_cycle_basis_h2d_tensors_per_batch": 1,
                            },
                            "precision": {
                                "amp": False,
                                "amp_effective": False,
                                "autocast_dtype": "disabled",
                                "fallback": None,
                                "gradient_scaler": False,
                                "projector_contraction": "float32",
                                "backbone_autocast": False,
                            },
                            "hardware": {"profile": "portable"},
                        },
                        "best_epoch": 2,
                        "epochs_completed": 4,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "history": str(history.resolve()),
                        "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
                        "evaluation_splits": ["train", "validation"],
                        "fresh_training": True,
                        "resource_observability": _resource_observability(),
                        "throughput": {
                            "scope": "unit fixture measured training",
                            "training_graphs_per_second": 10.0,
                        },
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
    assert row["resource_observability"]["total_point_sample_count"] == 2
    assert row["throughput"]["training_graphs_per_second"] == 10.0
    for field in ("resource_observability", "throughput"):
        incomplete = json.loads(json.dumps(metrics))
        del incomplete["datasets"]["zinc12k"]["models"]["cycle_projector_pe_v2"][field]
        (output / "metrics.json").write_text(json.dumps(incomplete), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            scaling.read_job_rows(job)
    metrics["datasets"]["zinc12k"]["models"]["cycle_projector_pe_v2"]["test"] = 0.3
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
            "reference",
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
            "reference",
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


def _start_interrupted_cycle_run(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(scaling, "check_dependencies", lambda: {"status": "passed"})
    monkeypatch.setattr(scaling, "_source_snapshot", lambda: {"source": "stable"})
    calls: list[list[str]] = []

    def interrupt_after_preflight(command, _log_path: Path, _environment):
        calls.append(command)
        if "gpu_preflight.py" in " ".join(command):
            output = Path(command[command.index("--json-out") + 1])
            output.write_text('{"status":"passed"}', encoding="utf-8")
            return 0
        raise KeyboardInterrupt

    monkeypatch.setattr(scaling, "run_logged", interrupt_after_preflight)
    arguments = [
        "--versions",
        "v1",
        "--profiles",
        "reference",
        "--model-seeds",
        "0",
        "--datasets",
        "zinc12k",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path),
        "--run-id",
        run_id,
    ]
    with pytest.raises(KeyboardInterrupt):
        scaling.main(arguments)
    assert len(calls) == 2
    return arguments, tmp_path / "cycle_pe/scaling" / run_id


def test_interrupted_resume_rejects_preflight_symlink_before_child_launch(tmp_path, monkeypatch):
    arguments, run = _start_interrupted_cycle_run(tmp_path, monkeypatch, "preflight-symlink")
    preflight = run / "gpu-preflight.json"
    preflight.unlink()
    sentinel = tmp_path / "outside-preflight.json"
    sentinel.write_text("preserve-preflight", encoding="utf-8")
    try:
        preflight.symlink_to(sentinel)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    resume_calls: list[list[str]] = []

    def unexpected_launch(command, _log_path: Path, _environment):
        resume_calls.append(command)
        pytest.fail("resume launched a subprocess before rejecting the preflight symlink")

    monkeypatch.setattr(scaling, "run_logged", unexpected_launch)
    with pytest.raises(ValueError, match="indirect.*GPU preflight output"):
        scaling.main(arguments)
    assert resume_calls == []
    assert sentinel.read_text(encoding="utf-8") == "preserve-preflight"


def test_interrupted_resume_rejects_summary_symlink_before_writes_or_launch(tmp_path, monkeypatch):
    arguments, run = _start_interrupted_cycle_run(tmp_path, monkeypatch, "summary-symlink")
    summary = run / "summary.json"
    summary.unlink()
    sentinel = tmp_path / "outside-summary.json"
    sentinel.write_text("preserve-summary", encoding="utf-8")
    try:
        summary.symlink_to(sentinel)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    manifest = run / "manifest.json"
    manifest_before = manifest.read_bytes()
    launches: list[list[str]] = []
    writes: list[Path] = []

    def unexpected_launch(command, _log_path: Path, _environment):
        launches.append(command)
        pytest.fail("resume launched a subprocess before rejecting the summary symlink")

    def unexpected_write(path, *_args, **_kwargs):
        writes.append(Path(path))
        pytest.fail("resume wrote state before rejecting the summary symlink")

    monkeypatch.setattr(scaling, "run_logged", unexpected_launch)
    monkeypatch.setattr(scaling, "atomic_write_json", unexpected_write)
    assert scaling.main(arguments) == 2
    assert launches == [] and writes == []
    assert manifest.read_bytes() == manifest_before
    assert sentinel.read_text(encoding="utf-8") == "preserve-summary"


def test_same_run_id_resumes_valid_children_and_test_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(scaling, "check_dependencies", lambda: {"status": "passed"})
    monkeypatch.setattr(scaling, "_source_snapshot", lambda: {"source": "stable"})
    candidate_launches: list[int] = []
    test_launches: list[int] = []
    interrupt_second_test = True

    def completed_marker(job):
        return Path(job["output_dir"]) / "completed.marker"

    def fake_candidate_rows(job):
        if not completed_marker(job).is_file():
            raise OSError("candidate is incomplete")
        checkpoint = Path(job["output_dir"]) / "zinc12k/cycle_set/best.pt"
        history = checkpoint.with_name("history.json")
        return [
            {
                "version": job["version"],
                "profile": job["profile"],
                "dataset": "zinc12k",
                "model_seed": job["model_seed"],
                "config": dict(job["config"]),
                "validation_mae": 0.1 + job["model_seed"] / 100,
                "trainable_parameters": 123,
                "elapsed_seconds": 1.0,
                "peak_gpu_memory_bytes": 1024,
                "best_epoch": 2,
                "epochs_completed": 3,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": "a" * 64,
                "history": str(history.resolve()),
                "history_sha256": "b" * 64,
                "output_dir": job["output_dir"],
            }
        ]

    def fake_test_result(job):
        if not completed_marker(job).is_file():
            raise OSError("test evaluation is incomplete")
        return {
            "test_evaluation_id": job["job_id"],
            "checkpoint_id": job["checkpoint_id"],
            "profile_selection_id": job["profile_selection_id"],
            "version": job["version"],
            "dataset": job["dataset"],
            "model_seed": job["model_seed"],
            "selected_profile": job["selected_profile"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "test_mae": 0.2 + job["model_seed"] / 100,
            "fresh_training": False,
        }

    def fake_run(command, _log_path: Path, _environment):
        nonlocal interrupt_second_test
        if "gpu_preflight.py" in " ".join(command):
            output = Path(command[command.index("--json-out") + 1])
            output.write_text('{"status":"passed"}', encoding="utf-8")
            return 0
        output = Path(command[command.index("--output-dir") + 1])
        seed = int(command[command.index("--model-seed") + 1])
        if "--validation-only" in command:
            candidate_launches.append(seed)
        else:
            test_launches.append(seed)
            if interrupt_second_test and len(test_launches) == 2:
                output.mkdir(parents=True)
                (output / "partial.tmp").write_text("interrupted", encoding="utf-8")
                interrupt_second_test = False
                raise KeyboardInterrupt
        output.mkdir(parents=True, exist_ok=True)
        (output / "completed.marker").write_text("passed", encoding="utf-8")
        return 0

    monkeypatch.setattr(scaling, "run_logged", fake_run)
    monkeypatch.setattr(scaling, "read_job_rows", fake_candidate_rows)
    monkeypatch.setattr(scaling, "read_test_result", fake_test_result)
    arguments = [
        "--versions",
        "v1",
        "--profiles",
        "reference",
        "--model-seeds",
        "0,1",
        "--datasets",
        "zinc12k",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path),
        "--run-id",
        "resume-cycle",
    ]

    with pytest.raises(KeyboardInterrupt):
        scaling.main(arguments)
    mismatched = list(arguments)
    mismatched[mismatched.index("0,1")] = "0"
    assert scaling.main(mismatched) == 2
    assert candidate_launches == [0, 1]
    assert test_launches == [0, 1]
    assert scaling.main(arguments) == 0

    run = tmp_path / "cycle_pe/scaling/resume-cycle"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert candidate_launches == [0, 1]
    assert test_launches == [0, 1, 1]
    assert manifest["resume_count"] == 1
    assert manifest["completed_child_runs"] == 2
    assert manifest["completed_selected_test_evaluations"] == 2
    assert manifest["test_evaluation_jobs"][1]["quarantined_outputs"]
    assert summary["status"] == "passed"
    assert len(summary["test_evaluations"]) == 2


def test_retrained_selected_candidate_rebinds_and_reruns_test_once(tmp_path, monkeypatch):
    monkeypatch.setattr(scaling, "check_dependencies", lambda: {"status": "passed"})
    monkeypatch.setattr(scaling, "_source_snapshot", lambda: {"source": "stable"})
    candidate_launches: list[int] = []
    test_launches: list[str] = []

    def fake_candidate_rows(job):
        output = Path(job["output_dir"])
        state = json.loads((output / "candidate-state.json").read_text(encoding="utf-8"))
        generation = state["generation"]
        run = output / "zinc12k/cycle_set"
        checkpoint = run / "best.pt"
        history = run / "history.json"
        return [
            {
                "version": job["version"],
                "profile": job["profile"],
                "dataset": "zinc12k",
                "model_seed": job["model_seed"],
                "config": dict(job["config"]),
                "validation_mae": 0.3 - generation / 10,
                "trainable_parameters": 123,
                "elapsed_seconds": 1.0,
                "peak_gpu_memory_bytes": 1024,
                "best_epoch": 2,
                "epochs_completed": 3,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "history": str(history.resolve()),
                "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
                "output_dir": job["output_dir"],
            }
        ]

    def fake_test_result(job):
        marker = json.loads(
            (Path(job["output_dir"]) / "test-result.json").read_text(encoding="utf-8")
        )
        if marker["checkpoint_sha256"] != job["checkpoint_sha256"]:
            raise ValueError("test result is bound to a stale checkpoint")
        return {
            "test_evaluation_id": job["job_id"],
            "checkpoint_id": job["checkpoint_id"],
            "profile_selection_id": job["profile_selection_id"],
            "version": job["version"],
            "dataset": job["dataset"],
            "model_seed": job["model_seed"],
            "selected_profile": job["selected_profile"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "test_mae": 0.2 + marker["launch"] / 100,
            "fresh_training": False,
        }

    def fake_run(command, _log_path: Path, _environment):
        if "gpu_preflight.py" in " ".join(command):
            output = Path(command[command.index("--json-out") + 1])
            output.write_text('{"status":"passed"}', encoding="utf-8")
            return 0
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if "--validation-only" in command:
            generation = len(candidate_launches) + 1
            candidate_launches.append(generation)
            run = output / "zinc12k/cycle_set"
            run.mkdir(parents=True, exist_ok=True)
            (run / "best.pt").write_bytes(f"checkpoint-{generation}".encode())
            (run / "history.json").write_text(
                json.dumps([{"generation": generation}]), encoding="utf-8"
            )
            (output / "candidate-state.json").write_text(
                json.dumps({"generation": generation}), encoding="utf-8"
            )
            return 0
        checkpoint = Path(command[command.index("--test-checkpoint") + 1])
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        launch = len(test_launches) + 1
        test_launches.append(checkpoint_sha256)
        (output / "test-result.json").write_text(
            json.dumps({"launch": launch, "checkpoint_sha256": checkpoint_sha256}),
            encoding="utf-8",
        )
        if launch == 1:
            (output / "old-test-sentinel.txt").write_text("preserve-old-test", encoding="utf-8")
        return 0

    monkeypatch.setattr(scaling, "run_logged", fake_run)
    monkeypatch.setattr(scaling, "read_job_rows", fake_candidate_rows)
    monkeypatch.setattr(scaling, "read_test_result", fake_test_result)
    arguments = [
        "--versions",
        "v1",
        "--profiles",
        "reference",
        "--model-seeds",
        "0",
        "--datasets",
        "zinc12k",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path),
        "--run-id",
        "candidate-rebind",
    ]
    assert scaling.main(arguments) == 0
    run = tmp_path / "cycle_pe/scaling/candidate-rebind"
    initial = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    old_test = initial["test_evaluation_jobs"][0]
    old_accepted_result = old_test["accepted_result"]
    old_test_output = Path(old_test["output_dir"])
    assert (old_test_output / "old-test-sentinel.txt").is_file()

    candidate_output = Path(initial["jobs"][0]["output_dir"])
    (candidate_output / "candidate-state.json").write_text("{corrupt", encoding="utf-8")
    test_launches_before_resume = len(test_launches)
    assert scaling.main(arguments) == 0

    final = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rebound = final["test_evaluation_jobs"][0]
    assert final["status"] == summary["status"] == "passed"
    assert candidate_launches == [1, 2]
    assert len(test_launches) == test_launches_before_resume + 1
    assert test_launches[0] != test_launches[1]
    assert rebound["selection_rebinds"] == 1
    assert rebound["accepted_result"] != old_accepted_result
    assert rebound["accepted_result"]["checkpoint_sha256"] == test_launches[1]
    previous = rebound["previous_attempts"][0]
    assert previous["accepted_result"] == old_accepted_result
    quarantined = Path(previous["quarantined_output"])
    assert quarantined != Path(rebound["output_dir"])
    assert (quarantined / "old-test-sentinel.txt").read_text(encoding="utf-8") == (
        "preserve-old-test"
    )
    assert not (Path(rebound["output_dir"]) / "old-test-sentinel.txt").exists()


def test_completed_run_returns_without_relaunching_preflight_or_children(tmp_path, monkeypatch):
    monkeypatch.setattr(scaling, "check_dependencies", lambda: {"status": "passed"})
    monkeypatch.setattr(scaling, "_source_snapshot", lambda: {"source": "stable"})
    calls: list[list[str]] = []

    def completed_marker(job):
        return Path(job["output_dir"]) / "completed.marker"

    def fake_candidate_rows(job):
        if not completed_marker(job).is_file():
            raise OSError("candidate is incomplete")
        checkpoint = Path(job["output_dir"]) / "zinc12k/cycle_set/best.pt"
        history = checkpoint.with_name("history.json")
        return [
            {
                "version": job["version"],
                "profile": job["profile"],
                "dataset": "zinc12k",
                "model_seed": job["model_seed"],
                "config": dict(job["config"]),
                "validation_mae": 0.1,
                "trainable_parameters": 123,
                "elapsed_seconds": 1.0,
                "peak_gpu_memory_bytes": 1024,
                "best_epoch": 2,
                "epochs_completed": 3,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": "a" * 64,
                "history": str(history.resolve()),
                "history_sha256": "b" * 64,
                "output_dir": job["output_dir"],
            }
        ]

    def fake_test_result(job):
        if not completed_marker(job).is_file():
            raise OSError("test evaluation is incomplete")
        return {
            "test_evaluation_id": job["job_id"],
            "checkpoint_id": job["checkpoint_id"],
            "profile_selection_id": job["profile_selection_id"],
            "version": job["version"],
            "dataset": job["dataset"],
            "model_seed": job["model_seed"],
            "selected_profile": job["selected_profile"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "test_mae": 0.2,
            "fresh_training": False,
        }

    def complete_run(command, _log_path: Path, _environment):
        calls.append(command)
        if "gpu_preflight.py" in " ".join(command):
            output = Path(command[command.index("--json-out") + 1])
            output.write_text('{"status":"passed"}', encoding="utf-8")
            return 0
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "completed.marker").write_text("passed", encoding="utf-8")
        return 0

    monkeypatch.setattr(scaling, "run_logged", complete_run)
    monkeypatch.setattr(scaling, "read_job_rows", fake_candidate_rows)
    monkeypatch.setattr(scaling, "read_test_result", fake_test_result)
    arguments = [
        "--versions",
        "v1",
        "--profiles",
        "reference",
        "--model-seeds",
        "0",
        "--datasets",
        "zinc12k",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path),
        "--run-id",
        "completed-cycle",
    ]
    assert scaling.main(arguments) == 0
    assert len(calls) == 3  # preflight, one candidate, and one selected-checkpoint test

    run = tmp_path / "cycle_pe/scaling/completed-cycle"
    before = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    accepted_rows = before["jobs"][0]["accepted_rows"]
    accepted_result = before["test_evaluation_jobs"][0]["accepted_result"]
    unexpected_calls: list[list[str]] = []

    def failing_if_relaunched(command, _log_path: Path, _environment):
        unexpected_calls.append(command)
        return 97

    monkeypatch.setattr(scaling, "run_logged", failing_if_relaunched)
    assert scaling.main(arguments) == 0
    assert unexpected_calls == []

    after = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert after["status"] == "passed"
    assert after["jobs"][0]["status"] == "passed"
    assert after["test_evaluation_jobs"][0]["status"] == "passed"
    assert after["jobs"][0]["accepted_rows"] == accepted_rows
    assert after["test_evaluation_jobs"][0]["accepted_result"] == accepted_result


def test_recovered_nonpassed_cycle_jobs_store_acceptance_anchors(tmp_path, monkeypatch):
    candidate_rows = [{"candidate": "verified"}]
    candidate = {
        "status": "running",
        "returncode": None,
        "output_dir": str(tmp_path / "candidate"),
        "artifact_errors": ["old"],
    }
    Path(candidate["output_dir"]).mkdir()
    monkeypatch.setattr(scaling, "read_job_rows", lambda _job: candidate_rows)
    assert scaling._recover_candidate_rows([candidate]) == candidate_rows
    assert candidate["status"] == "passed"
    assert candidate["accepted_rows"] == candidate_rows

    test_result = {"test": "verified"}
    test_job = {
        "status": "failed",
        "returncode": 0,
        "output_dir": str(tmp_path / "selected-test"),
        "artifact_errors": ["old"],
    }
    Path(test_job["output_dir"]).mkdir()
    monkeypatch.setattr(scaling, "read_test_result", lambda _job: test_result)
    assert scaling._recover_test_rows([test_job]) == [test_result]
    assert test_job["status"] == "passed"
    assert test_job["accepted_result"] == test_result


def test_quarantine_rejects_resume_orphans_symlink_outside_run(tmp_path):
    run_dir = tmp_path / "run"
    output = run_dir / "candidate"
    output.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (run_dir / "resume-orphans").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    job = {"job_id": "v1/base/seed-0", "output_dir": str(output)}
    with pytest.raises(ValueError, match="indirect|outside the run directory"):
        scaling._quarantine_incomplete_output(job, run_dir)
    assert output.is_dir()


def test_projector_v2_attempt_preserves_last_checkpoint_and_requests_resume(tmp_path):
    run_dir = tmp_path / "run"
    args = scaling.parser().parse_args(
        [
            "--versions",
            "v2",
            "--profiles",
            "reference",
            "--datasets",
            "zinc12k",
            "--model-seeds",
            "0",
        ]
    )
    job = scaling.make_jobs(args, run_dir)[0]
    checkpoint = Path(job["output_dir"]) / "zinc12k/cycle_projector_pe_v2/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"epoch-state")

    attempt = scaling._candidate_attempt_command(job, run_dir)

    assert attempt[-1] == "--resume"
    assert checkpoint.read_bytes() == b"epoch-state"
    assert "quarantined_outputs" not in job
