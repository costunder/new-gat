"""Architecture-scaling orchestration contracts; subprocesses are always mocked."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from research.conductance_gat import benchmark
from research.conductance_gat.ablation import train as ablation_train
from research.conductance_gat.v2 import train as v2_train
from research.conductance_gat.v3 import train as v3_train
from research.conductance_gat.v4 import train as v4_train
from scripts import run_conductance_scaling as runner


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_default_plan_covers_all_versions_profiles_seed_zero_and_supported_datasets():
    args = runner.parser().parse_args([])
    jobs = runner.make_jobs(args, Path("fixture"))
    assert args.versions == ["v1", "v2", "v3", "v4"]
    assert args.profiles == ["base", "wide", "deep", "large"]
    assert args.model_seeds == [0]
    assert len(jobs) == 172
    assert {job["profile"] for job in jobs} == set(runner.PROFILES)
    assert {job["model_seed"] for job in jobs} == set(runner.DEFAULT_MODEL_SEEDS)
    assert not any(job["version"] == "v2" and job["dataset"] == "ppi" for job in jobs)
    assert any(job["version"] == "v1" and job["dataset"] == "ppi" for job in jobs)
    assert any(job["version"] == "v4" and job["dataset"] == "ppi" for job in jobs)


@pytest.mark.parametrize(
    "mutated_relative_path",
    ["scripts/run_conductance_factorial.py", "scripts/check_dependencies.py"],
)
def test_source_inventory_covers_imported_scripts_and_rejects_mutation(
    monkeypatch, mutated_relative_path
):
    snapshot = runner._source_snapshot()
    imported_scripts = {
        "scripts/run_conductance_factorial.py",
        "scripts/check_dependencies.py",
    }
    assert imported_scripts <= snapshot.keys()

    target = (runner.ROOT / mutated_relative_path).resolve()
    original_read_bytes = Path.read_bytes

    def read_bytes_with_mutation(path):
        contents = original_read_bytes(path)
        if path.resolve() == target:
            return contents + b"\n# simulated source mutation\n"
        return contents

    manifest = {"source_sha256": snapshot, "source_integrity_valid": True}
    monkeypatch.setattr(Path, "read_bytes", read_bytes_with_mutation)
    with pytest.raises(RuntimeError, match="source changed during execution"):
        runner._check_sources(manifest)
    assert manifest["source_integrity_valid"] is False


def test_large_profile_is_forwarded_to_every_child_and_outputs_are_unique():
    args = runner.parser().parse_args(
        [
            "--profiles",
            "large",
            "--datasets",
            "cora",
            "--model-seeds",
            "3",
            "4",
        ]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, Path("fixture"))
    assert len(jobs) == 18
    assert len({job["output_dir"] for job in jobs}) == len(jobs)
    for job in jobs:
        assert _argument(job["command"], "--hidden-channels") == "128"
        assert _argument(job["command"], "--layers") == "4"
        assert _argument(job["command"], "--dropout") == "0.5"
        assert job["architecture"] == runner.PROFILES["large"]
        module = _argument(job["command"], "-m")
        if job["version"] == "v1":
            assert module == "research.conductance_gat.scaling_v1"
            assert "--condition" not in job["command"]
        else:
            assert _argument(job["command"], "--condition") == job["condition"]


def test_explicit_ppi_records_v2_as_not_applicable_instead_of_inventing_transfer():
    args = runner.parser().parse_args(["--datasets", "ppi"])
    jobs = runner.make_jobs(args, Path("fixture"))
    exclusions = runner._exclusions(args)
    assert {job["version"] for job in jobs} == {"v1", "v3", "v4"}
    assert exclusions == [
        {
            "version": "v2",
            "dataset": "ppi",
            "status": "not_applicable",
            "reason": (
                "V2 direct edge conductances are bound to one fixed topology and cannot "
                "transfer to held-out PPI graphs"
            ),
        }
    ]


@pytest.mark.parametrize("option", ["--help", "--dry-run"])
def test_stdlib_inspection_has_no_writes(tmp_path, option):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            str(runner.ROOT / "scripts/run_conductance_scaling.py"),
            option,
            "--results-root",
            str(tmp_path),
            "--versions",
            "v1",
            "--profiles",
            "base",
            "--datasets",
            "cora",
            "--model-seeds",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_v2_v3_v4_children_accept_and_record_real_architecture_overrides():
    v2_args = v2_train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "direct_c",
            "--output-dir",
            "unused",
            "--hidden-channels",
            "128",
            "--layers",
            "4",
            "--dropout",
            "0.25",
        ]
    )
    v3_args = v3_train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "relative_c",
            "--output-dir",
            "unused",
            "--hidden-channels",
            "128",
            "--layers",
            "4",
            "--dropout",
            "0.25",
        ]
    )
    v4_args = v4_train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "relative_c_spatial_w",
            "--output-dir",
            "unused",
            "--hidden-channels",
            "128",
            "--layers",
            "4",
            "--dropout",
            "0.25",
        ]
    )
    for args, validate, configuration in (
        (v2_args, v2_train._validate_args, v2_train.configuration),
        (v3_args, v3_train._validate_args, v3_train.configuration),
        (v4_args, v4_train._validate_args, v4_train.configuration),
    ):
        validate(args)
        actual = configuration(args)
        assert {key: actual[key] for key in ("hidden_channels", "layers", "dropout")} == {
            "hidden_channels": 128,
            "layers": 4,
            "dropout": 0.25,
        }

    defaults = ablation_train.build_parser().parse_args(
        ["--dataset", "cora", "--condition", "baseline", "--output-dir", "unused"]
    )
    assert ablation_train.architecture_configuration(defaults) == {
        "hidden_channels": 64,
        "layers": 2,
        "dropout": 0.5,
    }


def test_v1_validation_only_path_does_not_construct_a_test_loader(monkeypatch):
    class FakeData:
        def __init__(self, **values):
            self.values = values

    class FakeLoader:
        def __init__(self, graphs, **kwargs):
            self.graphs = graphs
            self.kwargs = kwargs

    geometric = ModuleType("torch_geometric")
    data_module = ModuleType("torch_geometric.data")
    loader_module = ModuleType("torch_geometric.loader")
    data_module.Data = FakeData
    loader_module.DataLoader = FakeLoader
    monkeypatch.setitem(sys.modules, "torch_geometric", geometric)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)
    monkeypatch.setitem(sys.modules, "torch_geometric.loader", loader_module)
    payload = {
        "dataset": "ppi",
        "graphs": [{"identifier": index} for index in range(4)],
        "splits": {"train": [0, 1], "validation": [2], "test": [3]},
    }
    args = SimpleNamespace(
        validation_only=True,
        model_seed=0,
        batch_size=2,
        workers=0,
        pin_memory=True,
    )
    loaders, indices = benchmark._make_loaders(payload, args, benchmark.torch.device("cpu"))
    assert indices is None
    assert set(loaders) == {"train", "validation"}
    observed = [
        graph.values["identifier"] for loader in loaders.values() for graph in loader.graphs
    ]
    assert observed == [0, 1, 2]
    assert 3 not in observed


def _stub(tmp_path, monkeypatch, *, expose_test: bool = False):
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"unit_fixture_only": True})
    monkeypatch.setattr(runner, "_source_snapshot", lambda: {"unit-source": "stable"})

    def dispatch(command, log, environment):
        calls.append(command)
        if any(Path(part).name == "gpu_preflight.py" for part in command):
            return 0
        output = Path(_argument(command, "--output-dir"))
        output.mkdir(parents=True)
        architecture = {
            "hidden_channels": int(_argument(command, "--hidden-channels")),
            "layers": int(_argument(command, "--layers")),
            "dropout": float(_argument(command, "--dropout")),
        }
        record = {
            "status": "passed",
            "dataset": _argument(command, "--dataset"),
            "condition": (
                _argument(command, "--condition") if "--condition" in command else "conductance"
            ),
            "model_seed": int(_argument(command, "--model-seed")),
            "configuration": architecture,
            "evaluation_split": "validation",
            "test_evaluated": False,
            "validation": 0.75,
            "metric_name": "accuracy",
            "best_epoch": 3,
            "epochs_run": 5,
            "trainable_parameters": 1234,
            "total_parameters": 1300,
            "elapsed_seconds": 1.5,
            "peak_cuda_allocated_bytes": 2048,
        }
        if expose_test:
            record["test"] = 0.9
        (output / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
        return 0

    monkeypatch.setattr(runner.shared, "run_logged", dispatch)
    options = [
        "--results-root",
        str(tmp_path),
        "--run-id",
        "unit-fixture",
        "--versions",
        "v1",
        "v4",
        "--profiles",
        "base",
        "--datasets",
        "cora",
        "--model-seeds",
        "0",
        "1",
    ]
    return options, calls


def test_success_is_released_only_after_every_child_metric_is_valid(tmp_path, monkeypatch):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    assert len(calls) == 11  # preflight + (V1 1 + V4 4) x two seeds
    root = tmp_path / "conductance_gat/scaling/unit-fixture"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert all(job["status"] == "passed" for job in manifest["jobs"])
    assert summary["valid_for_validation_comparison"] is True
    assert summary["test_evaluated"] is False
    assert {row["n"] for row in summary["rows"]} == {2}
    assert all(row["validation_mean"] == 0.75 for row in summary["rows"])


def test_completed_run_is_verified_and_reused_without_any_execution(tmp_path, monkeypatch):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    calls.clear()

    assert runner.main(options) == 0
    assert calls == []


def test_retry_logs_use_new_paths_inside_the_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    preferred = run_dir / "logs/job.log"
    preferred.parent.mkdir(parents=True)
    preferred.write_text("first", encoding="utf-8")
    retry_one = runner._next_attempt_log(preferred, run_dir, "v1/cora")
    assert retry_one != preferred.resolve()
    assert retry_one.is_relative_to(run_dir.resolve())
    retry_one.parent.mkdir(parents=True)
    retry_one.write_text("second", encoding="utf-8")
    retry_two = runner._next_attempt_log(preferred, run_dir, "v1/cora")
    assert retry_two != retry_one
    assert retry_two.is_relative_to(run_dir.resolve())


def test_incomplete_child_symlink_cannot_delete_a_passed_sibling(tmp_path):
    run_dir = tmp_path / "run"
    passed = run_dir / "passed"
    passed.mkdir(parents=True)
    marker = passed / "metrics.json"
    marker.write_text("preserve", encoding="utf-8")
    indirect = run_dir / "incomplete"
    try:
        indirect.symlink_to(passed, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    with pytest.raises(RuntimeError, match="indirect child output"):
        runner._discard_incomplete_child({"output_dir": str(indirect)}, run_dir)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_resume_rejects_passed_output_symlink_alias_before_preflight(tmp_path, monkeypatch):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    run_dir = tmp_path / "conductance_gat/scaling/unit-fixture"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    aliased_output = Path(manifest["jobs"][0]["output_dir"])
    sibling_output = Path(manifest["jobs"][1]["output_dir"])
    sibling_sentinel = sibling_output / "preserve-sibling.txt"
    sibling_sentinel.write_text("preserve", encoding="utf-8")
    (aliased_output / "metrics.json").unlink()
    aliased_output.rmdir()
    try:
        aliased_output.symlink_to(sibling_output, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    calls.clear()

    assert runner.main(options) == 1
    assert calls == []
    assert sibling_sentinel.read_text(encoding="utf-8") == "preserve"


def test_interrupted_resume_rejects_external_preflight_symlink_before_execution(
    tmp_path, monkeypatch
):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    run_dir = tmp_path / "conductance_gat/scaling/unit-fixture"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["jobs"][0]["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outside_sentinel = tmp_path / "outside-preflight-sentinel.txt"
    outside_sentinel.write_text("preserve", encoding="utf-8")
    (run_dir / "gpu-preflight.json").unlink(missing_ok=True)
    try:
        (run_dir / "gpu-preflight.json").symlink_to(outside_sentinel)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    calls.clear()

    assert runner.main(options) == 1
    assert calls == []
    assert outside_sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("summary_name", ["summary.json", "summary.md"])
def test_interrupted_resume_rejects_external_summary_symlink_before_execution(
    tmp_path, monkeypatch, summary_name
):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    run_dir = tmp_path / "conductance_gat/scaling/unit-fixture"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["jobs"][0]["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outside_sentinel = tmp_path / f"outside-{summary_name}-sentinel.txt"
    outside_sentinel.write_text("preserve", encoding="utf-8")
    summary_path = run_dir / summary_name
    summary_path.unlink()
    try:
        summary_path.symlink_to(outside_sentinel)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    calls.clear()

    assert runner.main(options) == 1
    assert calls == []
    assert outside_sentinel.read_text(encoding="utf-8") == "preserve"


def test_resume_binds_minimum_free_gpu_memory(tmp_path, monkeypatch):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    calls.clear()
    assert runner.main([*options, "--min-free-gb", "9.0"]) == 1
    assert calls == []


@pytest.mark.parametrize("interrupted_status", ["pending", "running", "failed"])
def test_resume_skips_passed_jobs_and_reruns_only_nonpassed_job(
    tmp_path, monkeypatch, interrupted_status
):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    root = tmp_path / "conductance_gat/scaling/unit-fixture"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    retried = manifest["jobs"][0]
    retried["status"] = interrupted_status
    manifest["status"] = "failed" if interrupted_status == "failed" else "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls.clear()

    assert runner.main(options) == 0
    assert len(calls) == 2  # preflight plus exactly one retried child
    assert calls[1] == retried["command"]
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed["status"] == "passed"
    assert resumed["resume_count"] == 1
    assert all(job["status"] == "passed" for job in resumed["jobs"])


def test_corrupted_passed_artifact_fails_closed_before_any_execution(tmp_path, monkeypatch):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    root = tmp_path / "conductance_gat/scaling/unit-fixture"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metrics_path = Path(manifest["jobs"][0]["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["validation"] = 0.5
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    calls.clear()

    assert runner.main(options) == 1
    assert calls == []


@pytest.mark.parametrize("mismatch", ["config", "source", "job_plan"])
def test_resume_requires_exact_config_source_and_job_plan(tmp_path, monkeypatch, mismatch):
    options, calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    manifest_path = tmp_path / "conductance_gat/scaling/unit-fixture/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mismatch == "config":
        manifest["config"]["epochs"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mismatch == "source":
        monkeypatch.setattr(runner, "_source_snapshot", lambda: {"unit-source": "changed"})
    else:
        manifest["jobs"][0]["command"].append("--unexpected")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls.clear()

    assert runner.main(options) == 1
    assert calls == []


def test_any_test_metric_fails_closed_and_stops_following_children(tmp_path, monkeypatch):
    options, calls = _stub(tmp_path, monkeypatch, expose_test=True)
    assert runner.main(options) == 1
    assert len(calls) == 2
    summary = json.loads(
        (tmp_path / "conductance_gat/scaling/unit-fixture/summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["valid_for_validation_comparison"] is False
