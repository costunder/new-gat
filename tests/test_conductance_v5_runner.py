"""V5 orchestration contracts; no GPU training is launched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.conductance_gat.v5 import train
from scripts import run_conductance_v5 as runner


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_reference_plan_parses_with_real_child_cli_and_memory_controls(tmp_path):
    args = runner.parser().parse_args(["--datasets", "ogbn-arxiv", "--results-root", str(tmp_path)])
    runner._validate(args)
    architecture = runner._architecture(args)
    jobs = runner.make_jobs(args, tmp_path / "run", architecture)
    assert architecture == {
        "hidden_channels": 256,
        "layers": 8,
        "heads": 8,
        "ffn_multiplier": 4,
        "dropout": 0.2,
        "beta_parameterization": "sigmoid",
        "beta_initial": 0.1,
    }
    assert len(jobs) == 2
    assert {job["condition"] for job in jobs} == {"fixed_c", "shared_dynamic_c"}
    assert {job["sampling"] for job in jobs} == {"cluster"}
    for job in jobs:
        child = train.build_parser().parse_args(job["command"][5:])
        assert child.sample_seed_batch_size == 1024
        assert child.activation_checkpoint is True
        assert child.num_neighbors == [15, 10]
        assert child.hidden_channels == 256 and child.layers == 8 and child.heads == 8
        assert child.beta_parameterization == "sigmoid" and child.beta_initial == 0.1
        assert child.beta_min is None and child.beta_max is None


def test_margin_beta_ablation_is_validated_and_forwarded_to_every_child(tmp_path):
    options = [
        "--datasets",
        "cora",
        "--beta-parameterization",
        "margin_sigmoid",
        "--beta-initial",
        "0.5",
        "--beta-min",
        "0.05",
        "--beta-max",
        "0.95",
    ]
    args = runner.parser().parse_args(options)
    runner._validate(args)
    architecture = runner._architecture(args)
    assert architecture["beta_parameterization"] == "margin_sigmoid"
    assert architecture["beta_initial"] == 0.5
    assert architecture["beta_min"] == 0.05
    assert architecture["beta_max"] == 0.95
    jobs = runner.make_jobs(args, tmp_path / "run", architecture)
    for job in jobs:
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        assert child.beta_parameterization == "margin_sigmoid"
        assert (child.beta_initial, child.beta_min, child.beta_max) == (0.5, 0.05, 0.95)


def test_default_runner_contract_contains_no_irrelevant_beta_margins():
    args = runner.parser().parse_args(["--datasets", "cora"])
    architecture = runner._architecture(args)
    assert architecture["beta_parameterization"] == "sigmoid"
    assert architecture["beta_initial"] == 0.1
    assert "beta_min" not in architecture and "beta_max" not in architecture
    invalid = runner.parser().parse_args(["--datasets", "cora", "--beta-min", "0.05"])
    with pytest.raises(ValueError, match="only valid for margin_sigmoid"):
        runner._validate(invalid)


def test_ppi_auto_uses_full_graph_and_graph_batch_two(tmp_path):
    args = runner.parser().parse_args(["--datasets", "ppi"])
    jobs = runner.make_jobs(args, tmp_path / "run", runner._architecture(args))
    assert args.workers == 4
    assert {job["sampling"] for job in jobs} == {"full"}
    assert {_value(job["command"], "--batch-size") for job in jobs} == {"2"}
    assert {_value(job["command"], "--workers") for job in jobs} == {"4"}
    assert {job["batch_size"] for job in jobs} == {2}


def test_a6000_profile_forwards_real_larger_batches_and_numeric_policy(tmp_path):
    args = runner.parser().parse_args(
        ["--datasets", "ogbn-arxiv", "ppi", "--hardware-profile", "a6000-48gb"]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, tmp_path / "run", runner._architecture(args))
    arxiv = next(job for job in jobs if job["dataset"] == "ogbn-arxiv")
    ppi = next(job for job in jobs if job["dataset"] == "ppi")
    assert arxiv["execution"] == {
        "hardware_profile": "a6000-48gb",
        "precision": "bf16",
        "tf32": True,
        "batch_size": 1,
        "sample_seed_batch_size": 2048,
        "edge_chunk_size": 131072,
        "activation_checkpoint": False,
        "sample_prefetch": True,
        "pin_memory": True,
        "dataloader_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
    }
    assert ppi["batch_size"] == 8
    assert ppi["execution"]["dataloader_workers"] == 4
    assert ppi["execution"]["persistent_workers"] is True
    assert ppi["execution"]["prefetch_factor"] == 2
    assert _value(ppi["command"], "--batch-size") == "8"
    assert "--no-activation-checkpoint" in ppi["command"]


def test_runner_rejects_batch_override_that_would_change_dataset_contract():
    args = runner.parser().parse_args(["--batch-size", "2"])
    with pytest.raises(ValueError, match="runner-level batch size must be 1"):
        runner._validate(args)


def test_incomplete_output_without_checkpoint_is_preserved_before_retry(tmp_path, capsys):
    run_dir = tmp_path / "run"
    output = run_dir / "cora" / "fixed_c"
    output.mkdir(parents=True)
    (output / "partial.log").write_text("keep", encoding="utf-8")
    job = {"output_dir": str(output), "status": "failed"}

    runner._preserve_incomplete_child(job, run_dir)

    preserved = output.with_name("fixed_c.preserved-attempt-1")
    assert not output.exists()
    assert (preserved / "partial.log").read_text(encoding="utf-8") == "keep"
    assert job["preserved_incomplete_outputs"][0]["destination"] == str(preserved)
    report = json.loads(capsys.readouterr().err)
    assert report["source"] == str(output)
    assert report["destination"] == str(preserved)


def test_same_run_preserves_last_checkpoint_and_adds_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"fixture": True})
    monkeypatch.setattr(runner, "_source_snapshot", lambda: {"source": "stable"})
    monkeypatch.setattr(runner, "_write_comparison", lambda *_args: None)
    attempts: list[list[str]] = []
    first = True

    def dispatch(command, _log, _environment):
        nonlocal first
        if any(Path(item).name == "gpu_preflight.py" for item in command):
            return 0
        attempts.append(list(command))
        output = Path(_value(command, "--output-dir"))
        output.mkdir(parents=True, exist_ok=True)
        if first:
            first = False
            (output / "last.pt").write_bytes(b"resume-state")
            return 9
        configuration = {
            **runner._architecture(runner.parser().parse_args(["--datasets", "cora"])),
            "sampling": _value(command, "--sampling"),
            "batch_size": int(_value(command, "--batch-size")),
            "hardware_profile": _value(command, "--hardware-profile"),
            "precision": "fp32",
            "tf32": False,
            "edge_chunk_size": int(_value(command, "--edge-chunk-size")),
            "workers": int(_value(command, "--workers")),
        }
        hardware_execution = {
            "profile": "portable",
            "precision": "fp32",
            "tf32": False,
            "activation_checkpoint": "--activation-checkpoint" in command,
            "edge_chunk_size": int(_value(command, "--edge-chunk-size")),
            "sample_seed_batch_size": int(_value(command, "--sample-seed-batch-size")),
            "graph_batch_size": int(_value(command, "--batch-size")),
            "sample_prefetch": False,
            "pin_memory": True,
            "loader_workers": int(_value(command, "--workers")),
            "persistent_workers": False,
            "prefetch_factor": None,
        }
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "dataset": _value(command, "--dataset"),
                    "condition": _value(command, "--condition"),
                    "model_seed": int(_value(command, "--model-seed")),
                    "evaluation_split": "validation",
                    "test_evaluated": False,
                    "configuration": configuration,
                    "validation": 0.5,
                    "metric_name": "accuracy",
                    "best_epoch": 2,
                    "epochs_run": 3,
                    "trainable_parameters": 1_000_000,
                    "peak_cuda_allocated_bytes": 1024,
                    "peak_cuda_reserved_bytes": 2048,
                    "hardware_execution": hardware_execution,
                    "throughput": {"training_batches_per_elapsed_second": 1.0},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(runner.shared, "run_logged", dispatch)
    options = [
        "--datasets",
        "cora",
        "--results-root",
        str(tmp_path),
        "--run-id",
        "resume-v5",
    ]
    assert runner.main(options) == 1
    assert runner.main(options) == 0
    assert "--resume" in attempts[1]
    assert (tmp_path / "conductance_gat/v5/resume-v5/cora/fixed_c/last.pt").read_bytes() == (
        b"resume-state"
    )
