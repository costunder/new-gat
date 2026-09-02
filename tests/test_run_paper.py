from __future__ import annotations

import ast
import re
import shlex
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
    assert "--profile" not in completed.stdout
    assert "--nodes-per-graph" not in completed.stdout
    assert "--device cuda" in completed.stdout
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


def test_paper_runner_routes_custom_output_and_seed_without_dummy_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result_root = tmp_path / "scratch results"
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "custom-output",
            "--device",
            "cuda",
            "--no-amp",
            "--seeds",
            "7",
            "--results-root",
            str(result_root),
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--tiny" not in completed.stdout
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
    assert "gpu_preflight.py" not in completed.stdout
    assert "--allow-cpu" not in completed.stdout


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
    assert completed.stdout.count("--data-seed 11") == 4
    assert completed.stdout.count("--split-seed 13") == 4
    assert completed.stdout.count("--chart-seed 17") == 4
    assert completed.stdout.count("--model-seed 5") == 2
    assert completed.stdout.count("--model-seed 7") == 2


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


def test_cycle_runner_forwards_selected_non_projector_variants(
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


def test_supplementary_default_runs_own_pe_variants_without_no_pe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        ["--dry-run", "--tracks", "cycle_pe", "--suite", "core", "--model-seeds", "0"],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--variants raw,set,projector" in completed.stdout
    assert "--variants no_pe" not in completed.stdout


def test_brec_keeps_official_protocol_when_other_batch_sizes_are_overridden(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "cycle-official",
            "--tracks",
            "cycle_pe",
            "--suite",
            "all",
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
    brec_line = next(
        line for line in completed.stdout.splitlines() if "[cycle_pe:brec:official-10-seed]" in line
    )
    assert "--batch-size 16" in brec_line
    assert "--no-amp" in brec_line
    assert "--brec-protocol official" in brec_line
    assert "--variants no_pe,raw" in brec_line


@pytest.mark.parametrize("argument", ["--tiny", "--allow-cpu"])
def test_paper_runner_rejects_removed_dummy_options(argument: str) -> None:
    from scripts.run_paper import _parser

    with pytest.raises(SystemExit) as caught:
        _parser().parse_args([argument])
    assert caught.value.code == 2


def test_paper_runner_rejects_unsafe_run_id() -> None:
    from scripts.run_paper import _parser

    with pytest.raises(SystemExit) as caught:
        _parser().parse_args(["--run-id", "../escape"])
    assert caught.value.code == 2


def test_readme_commands_use_full_independent_protocols() -> None:
    from scripts import run_conductance_v2, run_conductance_v3, run_conductance_v4
    from scripts.run_paper import _parser

    readme = (ROOT / "docs/GETTING_STARTED.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", readme, flags=re.DOTALL)
    bash_commands = [
        line for block in blocks for line in block.splitlines() if line.startswith("bash ")
    ]
    setup_commands = [
        line for line in bash_commands if shlex.split(line)[1] == "scripts/setup_gpu.sh"
    ]
    assert setup_commands == [
        "bash scripts/setup_gpu.sh",
        "bash scripts/setup_gpu.sh --profile legacy-cu118",
    ]
    commands = [line for line in bash_commands if line not in setup_commands]
    v2_commands = [
        line
        for line in commands
        if shlex.split(line)[1] == "research/conductance_gat/v2/reproduce.sh"
    ]
    assert len(v2_commands) == 1
    v2_command = shlex.split(v2_commands[0])
    v2_source = (ROOT / v2_command[1]).read_text(encoding="utf-8")
    assert "set -euo pipefail" in v2_source
    assert 'source "${project_root}/scripts/conda_env.sh"' in v2_source
    assert 'exec "${environment_python}" -B scripts/run_conductance_v2.py "$@"' in v2_source
    v2_args = run_conductance_v2.parser().parse_args(v2_command[2:])
    run_conductance_v2._validate(v2_args)
    assert v2_args.datasets == ["cora", "citeseer", "pubmed", "ogbn-arxiv"]
    assert v2_args.model_seed == 0
    assert len(run_conductance_v2.make_jobs(v2_args, ROOT / "results/unit-contract")) == 8
    commands = [line for line in commands if line not in v2_commands]
    v3_commands = [
        line
        for line in commands
        if shlex.split(line)[1] == "research/conductance_gat/v3/reproduce.sh"
    ]
    assert len(v3_commands) == 1
    v3_command = shlex.split(v3_commands[0])
    v3_source = (ROOT / v3_command[1]).read_text(encoding="utf-8")
    assert "set -euo pipefail" in v3_source
    assert 'source "${project_root}/scripts/conda_env.sh"' in v3_source
    assert 'exec "${environment_python}" -B scripts/run_conductance_v3.py "$@"' in v3_source
    v3_args = run_conductance_v3.parser().parse_args(v3_command[2:])
    run_conductance_v3._validate(v3_args)
    assert v3_args.datasets == ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
    assert v3_args.model_seed == 0
    v3_jobs = run_conductance_v3.make_jobs(v3_args, ROOT / "results/unit-contract")
    assert len(v3_jobs) == 10
    assert {job["dataset"]: job["batch_size"] for job in v3_jobs} == {
        "cora": 1,
        "citeseer": 1,
        "pubmed": 1,
        "ppi": 2,
        "ogbn-arxiv": 1,
    }
    commands = [line for line in commands if line not in v3_commands]
    v4_commands = [
        line
        for line in commands
        if shlex.split(line)[1] == "research/conductance_gat/v4/reproduce.sh"
    ]
    assert len(v4_commands) == 1
    v4_command = shlex.split(v4_commands[0])
    v4_source = (ROOT / v4_command[1]).read_text(encoding="utf-8")
    assert "set -euo pipefail" in v4_source
    assert 'source "${project_root}/scripts/conda_env.sh"' in v4_source
    assert 'exec "${environment_python}" -B scripts/run_conductance_v4.py "$@"' in v4_source
    v4_args = run_conductance_v4.parser().parse_args(v4_command[2:])
    run_conductance_v4._validate(v4_args)
    assert v4_args.datasets == ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
    assert v4_args.model_seed == 0
    v4_jobs = run_conductance_v4.make_jobs(v4_args, ROOT / "results/unit-contract")
    assert len(v4_jobs) == 20
    assert {job["dataset"]: job["batch_size"] for job in v4_jobs} == {
        "cora": 1,
        "citeseer": 1,
        "pubmed": 1,
        "ppi": 2,
        "ogbn-arxiv": 1,
    }
    commands = [line for line in commands if line not in v4_commands]
    assert len(commands) == 5  # original full protocols remain unchanged
    parsed = []
    for line in commands:
        command = shlex.split(line)
        assert len(command) == 2
        wrapper = ROOT / command[1]
        source = wrapper.read_text(encoding="utf-8")
        dispatch = next(row for row in source.splitlines() if row.startswith("exec bash "))
        words = shlex.split(dispatch)
        assert words[2] == "${project_root}/scripts/paper.sh"
        assert words[-1] == "$@"
        assert "set -euo pipefail" in source
        parsed.append(_parser().parse_args(words[3:-1]))
    assert sum(args.prepare_only for args in parsed) == 1
    assert all(args.suite == "benchmark" for args in parsed)
    assert {tuple(args.tracks) for args in parsed if not args.prepare_only} == {
        ("conductance_gat",),
        ("cycle_pe",),
        ("tree_augmentation",),
        ("all",),
    }
    assert all(args.device == "cuda" and args.model_seeds == (0,) for args in parsed)
    assert "--tiny" not in readme
    assert "python -c" not in readme
    assert "\\\n" not in readme
    assert "tmux new" not in readme
    assert 'source "$(conda info --base)' not in readme


def test_default_workspace_directories_exist_in_a_clone() -> None:
    paths = [
        "data/.gitkeep",
        "results/.gitkeep",
        "research/conductance_gat/results/.gitkeep",
        "research/cycle_pe/results/.gitkeep",
        "research/tree_augmentation/results/.gitkeep",
    ]
    assert all((ROOT / path).is_file() for path in paths)


def test_default_benchmarks_match_each_track_without_generated_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(["--dry-run"], capsys)
    assert completed.returncode == 0, completed.stderr
    assert "research.conductance_gat.benchmark" in completed.stdout
    assert "research.cycle_pe.benchmark" in completed.stdout
    assert "research.tree_augmentation.paper --suite csl" in completed.stdout
    assert "research.tree_augmentation.paper --suite zinc" in completed.stdout
    assert "--require-paper-deps" in completed.stdout
    assert "--suite core" not in completed.stdout
    assert "--suite brec" not in completed.stdout
    assert "--variants" not in completed.stdout
    assert "--baselines" not in completed.stdout
    assert "research.conductance_gat.paper" not in completed.stdout
    assert completed.stdout.count("--model-seed 0") == 4


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize(
    "selection,expected_seeds",
    [([], (0,)), (["--model-seeds", "2,5"], (2, 5)), (["--seeds", "11,12"], (11, 12))],
)
def test_benchmark_default_and_explicit_seed_sweeps(prepare_only, selection, expected_seeds):
    from scripts.run_paper import _commands, _parser

    args = _parser().parse_args(selection + (["--prepare-only"] if prepare_only else []))
    assert args.model_seeds == expected_seeds
    children = [
        command
        for name, command, _ in _commands(args, "seed-dispatch-contract")
        if name != "gpu_preflight"
    ]
    executed_seeds = expected_seeds[:1] if prepare_only else expected_seeds
    assert len(children) == 4 * len(executed_seeds)
    for seed in executed_seeds:
        seed_children = [
            command
            for command in children
            if command[command.index("--model-seed") + 1] == str(seed)
        ]
        assert [command[2] for command in seed_children] == [
            "research.conductance_gat.benchmark",
            "research.cycle_pe.benchmark",
            "research.tree_augmentation.paper",
            "research.tree_augmentation.paper",
        ]


def test_benchmark_prepares_each_public_suite_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        ["--dry-run", "--prepare-only", "--allow-download", "--model-seeds", "2,3"],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("--model-seed 2") == 4
    assert "--model-seed 3" not in completed.stdout
    assert "gpu_preflight.py" not in completed.stdout
    assert "--suite core" not in completed.stdout


@pytest.mark.parametrize("prepare_only", [False, True])
def test_own_model_child_arguments_parse_with_actual_track_clis(prepare_only: bool) -> None:
    from research.conductance_gat.benchmark import build_parser as conductance_parser
    from research.cycle_pe.benchmark import parser as cycle_parser
    from research.tree_augmentation.paper import _parser as tree_parser
    from scripts.run_paper import _commands, _parser

    parsers = {
        "research.conductance_gat.benchmark": conductance_parser(),
        "research.cycle_pe.benchmark": cycle_parser(),
        "research.tree_augmentation.paper": tree_parser(),
    }
    args = _parser().parse_args(["--prepare-only", "--allow-download"] if prepare_only else [])
    commands = _commands(args, "argument-contract")
    children = [command for name, command, _ in commands if name != "gpu_preflight"]
    assert len(children) == 4
    for command in children:
        parsed = parsers[command[2]].parse_args(command[3:])
        assert parsed.prepare_only is prepare_only
        assert parsed.model_seed == 0
        assert parsed.device == ("cpu" if prepare_only else "cuda")
        assert not hasattr(parsed, "baselines")
        assert not parsed.amp


def test_legacy_demo_entrypoints_are_removed() -> None:
    paths = [
        "scripts/run_all.py",
        "scripts/smoke.sh",
        "scripts/smoke.ps1",
        "scripts/setup.sh",
        "scripts/setup.ps1",
        "research/conductance_gat/run.py",
        "research/cycle_pe/run.py",
        "research/tree_augmentation/run.py",
    ]
    assert all(not (ROOT / path).exists() for path in paths)


def test_project_docs_and_gpt_handoff_are_separated() -> None:
    ignored = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv-gpu",
        "data",
        "results",
    }
    markdown = [
        path
        for path in ROOT.rglob("*.md")
        if not any(part in ignored or part.startswith(".venv") for part in path.parts)
    ]
    outside_document_folders = {
        path.relative_to(ROOT).as_posix()
        for path in markdown
        if path.parent not in {ROOT / "docs", ROOT / "gpt_handoff"}
    }
    assert outside_document_folders == {"README.md"}

    handoff_files = {path.name for path in (ROOT / "gpt_handoff").glob("*.md")}
    assert handoff_files == {
        "README_FIRST.md",
        "HANDOFF.md",
        "EXPERIMENT_STATUS.md",
        "CONDUCTANCE_V2.md",
        "CONDUCTANCE_V3.md",
        "CONDUCTANCE_V4.md",
        "CYCLE_PE_V2.md",
        "RICH_SCALING_EXPERIMENTS.md",
        "CODE_SUMMARY.md",
    }

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(root_readme) < 1000
    assert "docs/README.md" in root_readme
    assert "docs/GETTING_STARTED.md" in root_readme
    assert "gpt_handoff/README_FIRST.md" in root_readme

    index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    for document in (ROOT / "docs").glob("*.md"):
        if document.name != "README.md":
            assert f"({document.name})" in index, document.name

    package_readme = (ROOT / "gpt_handoff/README_FIRST.md").read_text(encoding="utf-8")
    for document in handoff_files:
        assert document in package_readme
    assert "V4만이 아니라 NEW GAT 전체 프로젝트" in package_readme

    hub = (ROOT / "gpt_handoff/CONDUCTANCE_V4.md").read_text(encoding="utf-8")
    for required in (
        "V3 자체를 spectral GNN이라고 분류하는 실험이 아니다",
        "C(H_pre-W)",
        "P_C(HW)",
        "fixed_c_identity_w",
        "relative_c_spatial_w",
        "research/conductance_gat/v4/reproduce.sh",
        "results/conductance_gat/v4/<run-id>/",
        "현재 상태",
    ):
        assert required in hub


def test_all_local_document_links_resolve() -> None:
    links = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    documents = [
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "gpt_handoff").glob("*.md"),
    ]
    for document in documents:
        if document.name == "CODE_SUMMARY.md":
            continue
        contents = document.read_text(encoding="utf-8")
        for destination in links.findall(contents):
            destination = destination.strip()
            if destination.startswith(("#", "http://", "https://", "mailto:")):
                continue
            assert not any(char.isspace() for char in destination), (
                document,
                destination,
            )
            local = destination.partition("#")[0]
            assert (document.parent / local).resolve().exists(), (document, destination)


def test_gpt_handoff_markdown_links_are_self_contained() -> None:
    links = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    package = (ROOT / "gpt_handoff").resolve()
    for document in (ROOT / "gpt_handoff").glob("*.md"):
        if document.name == "CODE_SUMMARY.md":
            continue
        contents = document.read_text(encoding="utf-8")
        for destination in links.findall(contents):
            destination = destination.strip()
            if destination.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local = destination.partition("#")[0]
            resolved = (document.parent / local).resolve()
            assert resolved.parent == package, (document, destination)
