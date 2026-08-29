from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.run_paper import (
    CYCLE_BREC_OFFICIAL_SEEDS,
    _run_logged,
    _stop_after_failure,
)
from scripts.run_paper import main as run_paper_main

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_paper.py"
CYCLE_RUNNER = ROOT / "research" / "cycle_pe" / "paper.py"


def _dry_run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> SimpleNamespace:
    with patch.object(sys, "argv", [str(RUNNER), *arguments]):
        return_code = run_paper_main()
    captured = capsys.readouterr()
    return SimpleNamespace(returncode=return_code, stdout=captured.out, stderr=captured.err)


def _literal_assignment(path: Path, name: str) -> object:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"assignment {name!r} was not found in {path}")


def test_shared_preflight_failure_is_always_fatal() -> None:
    assert _stop_after_failure("gpu_preflight", fail_fast=False)
    assert not _stop_after_failure("cycle_pe:seed-0", fail_fast=False)
    assert _stop_after_failure("cycle_pe:seed-0", fail_fast=True)


def test_root_brec_protocol_matches_cycle_runner() -> None:
    assert CYCLE_BREC_OFFICIAL_SEEDS == _literal_assignment(CYCLE_RUNNER, "BREC_OFFICIAL_SEEDS")


def test_logged_child_uses_utf8_for_non_ascii_artifacts(tmp_path: Path) -> None:
    log_path = tmp_path / "child.log"
    return_code = _run_logged(
        [sys.executable, "-c", "print('프로젝트/결과/β')"],
        log_path=log_path,
    )
    assert return_code == 0
    assert log_path.read_text(encoding="utf-8").strip() == "프로젝트/결과/β"


def test_paper_runner_defaults_to_cuda_and_every_independent_track(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--suite",
            "all",
            "--run-id",
            "paper-dry-run",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gpu_preflight.py" in completed.stdout
    assert "--device cuda" in completed.stdout
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    for profile in ("conductance", "cycle-projector", "tree-chart", "brec", "public-pyg"):
        assert f"--profile {profile}" in preflight_line
    assert "--batch-size 32" in preflight_line
    assert "--brec-batch-size 16" in preflight_line
    assert "--cycle-variants no_pe,raw,set,projector" in preflight_line
    assert "--brec-protocol official" in preflight_line
    assert "--no-brec-amp" in preflight_line
    for module in (
        "research.conductance_gat.paper",
        "research.cycle_pe.paper",
        "research.tree_augmentation.paper",
    ):
        assert module in completed.stdout
    assert "combined_later" not in completed.stdout
    assert completed.stdout.count("[cycle_pe:brec:official-10-seed]") == 1
    assert completed.stdout.count("--suite brec") == 1
    brec_line = next(
        line for line in completed.stdout.splitlines() if "[cycle_pe:brec:official-10-seed]" in line
    )
    assert "--batch-size 16" in brec_line
    assert "--workers 0" in brec_line
    assert "--no-amp" in brec_line
    assert "--brec-protocol official" in brec_line
    assert "--batch-size 32" not in brec_line
    assert "--amp" not in brec_line
    assert "--brec-seeds 100,200,300,400,500,600,700,800,900,1000" in completed.stdout


def test_paper_runner_refuses_full_cpu_execution(capsys: pytest.CaptureFixture[str]) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "bad-cpu",
            "--device",
            "cpu",
        ],
        capsys,
    )
    assert completed.returncode == 2
    assert "requires CUDA" in completed.stderr


def test_paper_runner_allows_only_explicit_tiny_cpu_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result_root = tmp_path / "scratch results"
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "tiny-cpu",
            "--device",
            "cpu",
            "--tiny",
            "--allow-cpu",
            "--seeds",
            "7",
            "--results-root",
            str(result_root),
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--tiny" in completed.stdout
    assert "--no-amp" in completed.stdout
    assert "model-seed-7" in completed.stdout
    assert "--model-seed 7" in completed.stdout
    assert "--data-seed 0" in completed.stdout
    assert "--split-seed 0" in completed.stdout
    assert "--chart-seed 0" in completed.stdout
    assert str(result_root.resolve()) in completed.stdout


def test_paper_runner_allows_cpu_data_preparation_without_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "prepare-cpu",
            "--device",
            "cpu",
            "--prepare-only",
            "--suite",
            "all",
            "--allow-download",
            "--seeds",
            "11,12",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--prepare-only" in completed.stdout
    assert "--allow-download" in completed.stdout
    assert completed.stdout.count("--model-seed 11") == 5
    assert "--model-seed 12" not in completed.stdout
    assert "--seed" not in completed.stdout
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--device cpu" in preflight_line
    assert "--profile conductance" in preflight_line
    for expensive_profile in ("cycle-projector", "tree-chart", "brec", "public-pyg"):
        assert f"--profile {expensive_profile}" not in preflight_line


def test_paper_runner_routes_independent_seed_axes(capsys: pytest.CaptureFixture[str]) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "seed-axes",
            "--tracks",
            "tree_augmentation",
            "--model-seeds",
            "5,7",
            "--data-seed",
            "11",
            "--split-seed",
            "13",
            "--chart-seed",
            "17",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("--data-seed 11") == 2
    assert completed.stdout.count("--split-seed 13") == 2
    assert completed.stdout.count("--chart-seed 17") == 2
    assert completed.stdout.count("--model-seed 5") == 1
    assert completed.stdout.count("--model-seed 7") == 1


def test_paper_runner_exposes_cycle_candidate_reduction_without_overriding_official_brec(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "cycle-candidates",
            "--tracks",
            "cycle_pe",
            "--suite",
            "all",
            "--model-seeds",
            "3",
            "--cycle-variants",
            "no_pe,projector",
            "--cycle-core-targets",
            "graph",
            "--cycle-epochs",
            "7",
            "--cycle-learning-rate",
            "0.002",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    cycle_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("[cycle_pe:") and "research.cycle_pe.paper" in line
    ]
    assert len(cycle_lines) == 3
    assert all("--variants no_pe,projector" in line for line in cycle_lines)
    core_line = next(line for line in cycle_lines if "--suite core" in line)
    zinc_line = next(line for line in cycle_lines if "--suite zinc" in line)
    brec_line = next(line for line in cycle_lines if "--suite brec" in line)
    assert "--core-targets graph" in core_line
    assert "--core-targets" not in zinc_line
    assert "--core-targets" not in brec_line
    assert "--epochs 7" in core_line and "--epochs 7" in zinc_line
    assert "--learning-rate 0.002" in core_line
    assert "--learning-rate 0.002" in zinc_line
    assert "--epochs" not in brec_line
    assert "--learning-rate" not in brec_line
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--profile cycle-projector" in preflight_line
    assert "--profile brec" in preflight_line
    assert "--profile public-pyg" not in preflight_line
    assert "--profile conductance" not in preflight_line
    assert "--profile tree-chart" not in preflight_line
    assert "--cycle-variants no_pe,projector" in preflight_line
    assert "--brec-protocol official" in preflight_line
    assert "--brec-batch-size 16" in preflight_line
    assert "--no-brec-amp" in preflight_line


def test_cycle_preflight_runs_selected_non_projector_variants(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "cycle-no-projector",
            "--tracks",
            "cycle_pe",
            "--suite",
            "core",
            "--model-seeds",
            "3",
            "--cycle-variants",
            "no_pe,raw,set",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--profile cycle-projector" in preflight_line
    assert "--profile conductance" not in preflight_line
    assert "--cycle-variants no_pe,raw,set" in preflight_line


def test_tiny_custom_brec_preflight_matches_batch_amp_and_variants(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "cycle-tiny-custom",
            "--tracks",
            "cycle_pe",
            "--suite",
            "all",
            "--tiny",
            "--batch-size",
            "7",
            "--model-seeds",
            "3",
            "--cycle-variants",
            "no_pe,raw",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--profile cycle-projector" in preflight_line
    assert "--profile brec" in preflight_line
    assert "--profile public-pyg" not in preflight_line
    assert "--cycle-variants no_pe,raw" in preflight_line
    assert "--brec-protocol custom" in preflight_line
    assert "--brec-batch-size 7" in preflight_line
    assert "--brec-amp" in preflight_line
    brec_line = next(
        line for line in completed.stdout.splitlines() if "[cycle_pe:brec:custom-tiny]" in line
    )
    assert "--batch-size 7" in brec_line
    assert "--amp" in brec_line
    assert "--brec-protocol custom" in brec_line
    assert "--variants no_pe,raw" in brec_line
