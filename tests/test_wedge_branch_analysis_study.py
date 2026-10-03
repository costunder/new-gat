"""Saved scalar analysis must preserve the run and never invoke a model or GPU."""

from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def completed_branch_run():
    path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "wedge-branch-strength-debug-20261004-02"
    )
    if not path.is_dir():
        pytest.skip("completed Experiment 4.1 DEBUG scalar artifact is required")
    return path


def test_real_saved_debug_cli_analysis_is_complete_and_has_no_model_execution(
    completed_branch_run, tmp_path, monkeypatch, capsys
):
    import torch

    from research.wedge_propagation.branch_analysis.source import assert_unchanged, load_run
    from research.wedge_propagation.branch_analysis.study import run
    from research.wedge_propagation.classification.common import read_json
    from research.wedge_propagation.classification.model import PackedClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError("saved CSV analysis invoked a checkpoint/model/optimizer/CUDA")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(PackedClassifier, "forward", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(torch.cuda, "memory_allocated", forbidden)
    source = load_run(completed_branch_run)
    output = tmp_path / "saved-analysis-debug"
    run(Namespace(source_dir=completed_branch_run, output_dir=output))
    assert_unchanged(source)
    completion = read_json(output / "completion.json")
    assert completion["completed"] is completion["source_files_preserved"] is True
    assert completion["profile"] == "debug"
    assert completion["actual_data"] is False
    assert completion["optimizer_updates"] == completion["model_forwards"] == 0
    assert completion["source_coverage"] == source.completion["coverage"]
    assert completion["analysis_rows"] == {
        "baseline_strength": 24,
        "strength_estimates": 204,
        "layer_changes": 756,
        "fixed_estimates": 384,
    }
    for name in completion["analysis_rows"]:
        assert (output / f"{name}.csv").is_file()
    summary = (output / "STRENGTH_ANALYSIS_SUMMARY.md").read_text(encoding="utf-8")
    displayed = capsys.readouterr().out
    assert summary in displayed
    assert "[start]" in displayed and "[complete]" in displayed
    assert "model_forwards=0" in displayed
    assert "DEBUG" in summary and "원래 학습 3 epoch/run" in summary
    assert (output / "terminal.log").read_text(encoding="utf-8") == displayed[
        displayed.index("[start]") :
    ]
    contract = read_json(output / "contract.json")
    assert contract["csv_only"] is True
    assert contract["source_hashes"] == source.hashes
    assert contract["hardware"]["gpu_count_used"] == 0


def test_existing_output_is_preserved_without_opening_source(tmp_path, monkeypatch):
    from research.wedge_propagation.branch_analysis import study

    output = tmp_path / "original"
    output.mkdir()
    marker = output / "user-file.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(study, "load_run", lambda *args: pytest.fail("source was opened"))
    with pytest.raises(ValueError, match="NEW"):
        study.run(Namespace(source_dir=tmp_path / "missing", output_dir=output))
    assert marker.read_text(encoding="utf-8") == "keep"
    assert set(output.iterdir()) == {marker}


def test_output_inside_source_is_rejected(tmp_path):
    from research.wedge_propagation.branch_analysis.study import run

    with pytest.raises(ValueError, match="outside"):
        run(Namespace(source_dir=tmp_path, output_dir=tmp_path / "do-not-create"))
    assert not (tmp_path / "do-not-create").exists()


def test_source_failure_is_reported_and_preserves_partial_output(tmp_path, monkeypatch):
    from research.wedge_propagation.branch_analysis import study
    from research.wedge_propagation.classification.common import read_json

    def corrupt(*args):
        raise ValueError("corrupt completed source")

    monkeypatch.setattr(study, "load_run", corrupt)
    output = tmp_path / "failed-analysis"
    with pytest.raises(ValueError, match="corrupt"):
        study.run(Namespace(source_dir=tmp_path / "source", output_dir=output))
    failure = read_json(output / "failure.json")
    assert failure["source_and_partial_results_preserved"] is True
    assert failure["type"] == "ValueError"
    assert "corrupt" in failure["message"]
    assert not (output / "completion.json").exists()
