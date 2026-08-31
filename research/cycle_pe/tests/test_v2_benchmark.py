"""Runner protocol checks use bounded mocks, never GPU training or downloads."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from research.cycle_pe.v2 import benchmark
from research.cycle_pe.v2.data import DATASETS
from research.cycle_pe.v2.model import MODEL_NAME


def test_v2_defaults_match_official_v1_protocol_but_not_its_representation() -> None:
    args = benchmark.parser().parse_args([])
    assert tuple(args.datasets) == DATASETS == ("zinc12k", "peptides_struct")
    assert (args.hidden_dim, args.pe_dim, args.layers) == (64, 32, 3)
    assert (args.epochs, args.patience, args.lr, args.batch_size) == (300, 50, 1e-3, 32)
    assert args.max_parameters == 500_000
    assert args.column_chunk_size == 16
    assert args.output_dir == Path("results/cycle_pe_v2/benchmark")
    assert MODEL_NAME == "cycle_basis_v2"
    assert not hasattr(args, "baselines")
    assert not hasattr(args, "tiny")
    assert not hasattr(args, "max_cycle_rank")


@pytest.mark.parametrize("flag", ["--baselines", "--tiny", "--max-cycle-rank"])
def test_no_baseline_dummy_or_cycle_truncation_options(flag):
    with pytest.raises(SystemExit):
        benchmark.parser().parse_args([flag])


def test_cpu_benchmark_training_and_invalid_chunk_size_are_rejected() -> None:
    args = benchmark.parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark._validate(args)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark._train_model("zinc12k", {}, args)
    args.prepare_only = True
    benchmark._validate(args)
    args.column_chunk_size = 0
    with pytest.raises(ValueError, match="column-chunk-size"):
        benchmark._validate(args)


def test_hashes_include_basis_data_encoder_and_reused_backbone_sources() -> None:
    hashes = benchmark.implementation_hashes()
    assert {
        "research/cycle_pe/v2/benchmark.py",
        "research/cycle_pe/v2/basis.py",
        "research/cycle_pe/v2/data.py",
        "research/cycle_pe/v2/model.py",
        "research/cycle_pe/benchmark_data.py",
        "research/cycle_pe/benchmark_models.py",
        "research/cycle_pe/paper_model.py",
    } <= set(hashes)
    root = Path(benchmark.__file__).resolve().parents[3]
    for name, value in hashes.items():
        assert value == hashlib.sha256((root / name).read_bytes()).hexdigest()


def test_prepare_only_records_separate_version_without_claiming_training(tmp_path, monkeypatch):
    loaded = []

    def fake_load(root, dataset, *, allow_download):
        loaded.append((dataset, allow_download))
        return {}, {"official_splits": True, "unit_fixture_only": True, "basis": "full_left_null"}

    monkeypatch.setattr(benchmark, "load_benchmark", fake_load)
    monkeypatch.setattr(benchmark, "_train_model", lambda *a: pytest.fail("must not train"))
    output = tmp_path / "v2"
    assert (
        benchmark.main(
            [
                "--datasets",
                "zinc12k",
                "--prepare-only",
                "--device",
                "cpu",
                "--output-dir",
                str(output),
                "--data-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert loaded == [("zinc12k", False)]
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for document in (metrics, manifest):
        assert document["status"] == "prepared"
        assert document["track"] == "cycle_pe"
        assert document["version"] == "v2"
    assert metrics["datasets"]["zinc12k"]["models"] == {}
    assert manifest["controls"]["model"] == "cycle_basis_v2"
    assert "no truncation" in manifest["controls"]["basis_input"]
    assert manifest["controls"]["basis_rank_dependent_parameters"] is False
    assert "SVD basis" in manifest["seeds"]["chart_seed"]
    with pytest.raises(FileExistsError):
        benchmark.main(["--prepare-only", "--output-dir", str(output)])


def test_only_v2_model_is_dispatched_once_per_official_dataset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "_validate", lambda args: None)
    monkeypatch.setattr(benchmark, "load_benchmark", lambda *args, **kwargs: ({}, {}))

    def fake_train(dataset, splits, args):
        calls.append(dataset)
        return {"test": 0.5, "validation": 0.4}

    monkeypatch.setattr(benchmark, "_train_model", fake_train)
    output = tmp_path / "v2_only"
    benchmark.main(["--output-dir", str(output), "--data-root", str(tmp_path)])
    assert calls == list(DATASETS)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "passed"
    assert all(set(entry["models"]) == {MODEL_NAME} for entry in metrics["datasets"].values())


def test_preparation_failure_is_persisted_not_reported_as_success(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("invalid left-nullspace basis")

    monkeypatch.setattr(benchmark, "load_benchmark", broken)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="left-nullspace"):
        benchmark.main(["--prepare-only", "--output-dir", str(output)])
    for filename in ("manifest.json", "metrics.json"):
        document = json.loads((output / filename).read_text(encoding="utf-8"))
        assert document["status"] == "failed"
        assert document["version"] == "v2"


def test_runner_selects_validation_checkpoint_before_single_test_evaluation() -> None:
    source = inspect.getsource(benchmark._train_model)
    assert source.count("evaluate(model, test_loader, device)") == 1
    assert source.index('model.load_state_dict(selected["state_dict"])') < source.index(
        "evaluate(model, test_loader, device)"
    )
    assert "if validation < best:" in source
    assert "weights_only=True" in source
