"""Optimization flags must preserve track, seed, precision and dataset contracts."""

import sys

import pytest

from scripts import run_paper


@pytest.mark.parametrize(
    "track,version", [("conductance_gat", "v1"), ("cycle_pe", "v1"), ("cycle_pe", "v2")]
)
def test_compile_is_forwarded_only_when_requested(track, version):
    common = ["--tracks", track, "--cycle-pe-version", version, "--model-seeds", "0"]
    for enabled in (False, True):
        args = run_paper._parser().parse_args(common + (["--compile"] if enabled else []))
        command = run_paper._commands(args, "optimization-unit")[-1][1]
        assert ("--compile" in command) is enabled
        assert "--no-amp" in command
        assert "--amp" not in command
        module = __import__(command[2], fromlist=["parser"])
        parser = module.build_parser() if track == "conductance_gat" else module.parser()
        parsed = parser.parse_args(command[3:])
        assert parsed.compile is enabled
        assert parsed.model_seed == 0


def test_preparation_never_compiles():
    args = run_paper._parser().parse_args(
        ["--tracks", "conductance_gat", "--compile", "--prepare-only"]
    )
    for _, command, _ in run_paper._commands(args, "prepare-unit"):
        assert "--compile" not in command


def test_v2_basis_execution_options_reach_child():
    args = run_paper._parser().parse_args(
        [
            "--tracks",
            "cycle_pe",
            "--cycle-pe-version",
            "v2",
            "--basis-execution",
            "reference",
            "--basis-pair-budget",
            "1024",
            "--model-seeds",
            "0",
        ]
    )
    command = run_paper._commands(args, "basis-unit")[-1][1]
    from research.cycle_pe.v2.benchmark import parser

    child = parser().parse_args(command[3:])
    assert child.basis_execution == "reference"
    assert child.basis_pair_budget == 1024


@pytest.mark.parametrize(
    "selection",
    [["--tracks", "tree_augmentation"], ["--suite", "core", "--tracks", "conductance_gat"], []],
)
def test_unsupported_compilation_rejected_before_dependencies(selection, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_paper.py", "--compile", "--dry-run", *selection])
    monkeypatch.setattr(run_paper, "check_dependencies", lambda: pytest.fail("too late"))
    assert run_paper.main() == 2
    assert "--compile supports" in capsys.readouterr().err


def test_basis_budget_rejected_before_data(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_paper.py", "--basis-pair-budget", "0", "--dry-run"])
    assert run_paper.main() == 2
    assert "must be positive" in capsys.readouterr().err
