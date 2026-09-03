"""Dispatch/artifact contracts only; no research training or downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import run_paper

ROOT = Path(__file__).resolve().parents[1]


def _args(*extra: str):
    return run_paper._parser().parse_args(
        ["--tracks", "cycle_pe", "--cycle-pe-version", "v2", *extra]
    )


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize(
    "selection,expected_seeds", [([], (0,)), (["--model-seeds", "3,7"], (3, 7))]
)
def test_v2_dispatch_is_only_basis_model_and_keeps_requested_seeds(
    prepare: bool, selection, expected_seeds
):
    from research.cycle_pe.v2.benchmark import parser

    args = _args(*selection, *(["--prepare-only", "--allow-download"] if prepare else []))
    commands = run_paper._commands(args, "v2-unit-contract")
    children = [entry for entry in commands if entry[0] != "gpu_preflight"]
    executed_seeds = expected_seeds[:1] if prepare else expected_seeds
    assert len(children) == len(executed_seeds)
    for seed, (name, command, output) in zip(executed_seeds, children, strict=True):
        assert name == f"cycle_pe:benchmark-v2:model-seed-{seed}"
        assert command[2] == "research.cycle_pe.v2.benchmark"
        child = parser().parse_args(command[3:])
        assert child.model_seed == seed
        assert child.basis_backend == "thin_q"
        assert child.prepare_only is prepare
        assert child.allow_download is prepare
        assert child.batch_size == 32 and child.workers == 4
        assert child.device == ("cpu" if prepare else "cuda")
        assert output == (
            ROOT
            / "research/cycle_pe/v2/results/paper/v2-unit-contract"
            / f"model-seed-{seed}/benchmark"
        )
    assert sum(name == "gpu_preflight" for name, _, _ in commands) == (0 if prepare else 1)


def test_root_runner_forwards_dfs_basis_backend_only_for_v2():
    from research.cycle_pe.v2.benchmark import parser

    args = _args("--basis-backend", "dfs_fundamental")
    children = [entry for entry in run_paper._commands(args, "dfs") if entry[0] != "gpu_preflight"]
    assert len(children) == 1
    child = parser().parse_args(children[0][1][3:])
    assert child.basis_backend == "dfs_fundamental"

    v1 = run_paper._parser().parse_args(["--tracks", "cycle_pe"])
    v1_command = run_paper._commands(v1, "v1-default")[-1][1]
    assert "--basis-backend" not in v1_command


def test_v1_default_dispatch_and_paths_remain_unchanged():
    args = run_paper._parser().parse_args(["--tracks", "cycle_pe", "--model-seeds", "0"])
    assert args.cycle_pe_version == "v1"
    name, command, output = run_paper._commands(args, "same-id")[-1]
    assert name == "cycle_pe:benchmark:model-seed-0"
    assert command[2] == "research.cycle_pe.benchmark"
    assert output == ROOT / "research/cycle_pe/results/paper/same-id/model-seed-0/benchmark"


def test_v2_custom_output_and_optimizer_overrides_are_version_specific(tmp_path):
    args = _args(
        "--results-root",
        str(tmp_path),
        "--model-seeds",
        "7",
        "--cycle-epochs",
        "123",
        "--cycle-learning-rate",
        "0.002",
    )
    name, command, output = run_paper._commands(args, "same-id")[-1]
    assert name.endswith("model-seed-7")
    assert output == tmp_path / "cycle_pe_v2/same-id/model-seed-7/benchmark"
    assert command[command.index("--epochs") + 1] == "123"
    assert command[command.index("--lr") + 1] == "0.002"
    assert run_paper._track_run_root("cycle_pe", "same-id", tmp_path) != (
        run_paper._track_run_root("cycle_pe", "same-id", tmp_path, cycle_pe_version="v2")
    )


@pytest.mark.parametrize(
    "selection", [[], ["--tracks", "conductance_gat"], ["--suite", "all"], ["--suite", "core"]]
)
def test_v2_rejects_unrelated_or_supplementary_tracks_before_dependency_checks(
    selection, monkeypatch, capsys
):
    arguments = ["--cycle-pe-version", "v2", "--dry-run"]
    if selection and selection[0] == "--suite":
        arguments.extend(["--tracks", "cycle_pe"])
    monkeypatch.setattr(sys, "argv", ["run_paper.py", *arguments, *selection])
    assert run_paper.main() == 2
    assert "v2 is independent" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["prepare_data.sh", "reproduce.sh"])
def test_v2_wrappers_use_direct_python_without_shell_set_flags(script):
    source = (ROOT / "research/cycle_pe/v2" / script).read_text(encoding="utf-8")
    assert "set -" not in source and "scripts/paper.sh" not in source
    if script == "reproduce.sh":
        assert "scripts/run_cycle_scaling.py" in source
        assert "--versions v2 --profiles reference" in source
    else:
        assert "-m research.cycle_pe.v2.benchmark" in source
        assert "--prepare-only --allow-download --device cpu" in source


def test_registry_snapshot_describes_projector_v2_not_raw_basis_coordinates(tmp_path):
    snapshot = run_paper._snapshot_registries(tmp_path, ("cycle_pe",), cycle_pe_version="v2")
    payload = Path(snapshot["cycle_pe"]["path"]).read_text(encoding="utf-8")
    assert "model: cycle_projector_pe_v2" in payload and "version: v2" in payload
    assert "sparse fundamental" in payload and "truncation: none" in payload
    assert "no raw basis coordinates" in payload


def test_v2_manifest_records_version_and_will_not_reuse_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_paper, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_paper, "check_dependencies", lambda: {"profile_id": "legacy-cu118"})
    monkeypatch.setattr(run_paper, "_commands", lambda *_args: [])
    monkeypatch.setattr(run_paper, "_source_revision", lambda: {})
    monkeypatch.setattr(run_paper, "_environment_snapshot", lambda *_args: {})
    monkeypatch.setattr(run_paper, "_snapshot_registries", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_paper.py",
            "--tracks",
            "cycle_pe",
            "--cycle-pe-version",
            "v2",
            "--basis-backend",
            "dfs_fundamental",
            "--prepare-only",
            "--run-id",
            "version-record",
        ],
    )
    assert run_paper.main() == 0
    manifest = json.loads((tmp_path / "runs/paper/version-record/manifest.json").read_text())
    assert manifest["execution_protocol"]["cycle_pe_version"] == "v2"
    assert manifest["execution_protocol"]["basis_backend"] == "dfs_fundamental"
    assert manifest["research_environment"]["profile_id"] == "legacy-cu118"
    assert manifest["tracks"] == ["cycle_pe"]
    assert run_paper.main() == 2
    # Independently check the v2 track-root guard, without relying on a global run folder.
    isolated = tmp_path / "research/cycle_pe/v2/results/paper/already-exists"
    isolated.mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_paper.py",
            "--tracks",
            "cycle_pe",
            "--cycle-pe-version",
            "v2",
            "--prepare-only",
            "--run-id",
            "already-exists",
        ],
    )
    assert run_paper.main() == 2
    assert not (tmp_path / "runs/paper/already-exists").exists()
