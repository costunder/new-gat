"""Architecture-scaling orchestration contracts; subprocesses are always mocked."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from chartgat.observability import finalize_resource_observability, runtime_resource_snapshot
from research.conductance_gat import benchmark, scaling_v1
from research.conductance_gat.ablation import train as ablation_train
from research.conductance_gat.v2 import train as v2_train
from research.conductance_gat.v3 import train as v3_train
from research.conductance_gat.v4 import train as v4_train
from research.conductance_gat.v5 import train as v5_train
from scripts import run_conductance_scaling as runner


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _resource_observability() -> dict:
    device = torch.device("cpu")
    return finalize_resource_observability(
        runtime_resource_snapshot(device),
        device,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        sample_interval_seconds=1.0,
    )


def test_default_plan_covers_all_versions_profiles_seed_zero_and_supported_datasets():
    args = runner.parser().parse_args([])
    jobs = runner.make_jobs(args, Path("fixture"))
    assert args.versions == ["v1", "v2", "v3", "v4", "v5"]
    assert args.profiles == ["reference", "large"]
    assert args.model_seeds == [0]
    assert args.workers == 4
    assert len(jobs) == 106
    assert {job["profile"] for job in jobs} == set(runner.PROFILES)
    assert {job["model_seed"] for job in jobs} == set(runner.DEFAULT_MODEL_SEEDS)
    assert not any(job["version"] == "v2" and job["dataset"] == "ppi" for job in jobs)
    assert any(job["version"] == "v1" and job["dataset"] == "ppi" for job in jobs)
    assert any(job["version"] == "v4" and job["dataset"] == "ppi" for job in jobs)
    assert any(job["version"] == "v5" and job["dataset"] == "ppi" for job in jobs)
    v5_jobs = [job for job in jobs if job["version"] == "v5"]
    assert all(job["architecture"]["beta_parameterization"] == "sigmoid" for job in v5_jobs)
    assert all(job["architecture"]["beta_initial"] == 0.1 for job in v5_jobs)
    assert all("beta_min" not in job["architecture"] for job in v5_jobs)
    assert all(_argument(job["command"], "--beta-initial") == "0.1" for job in v5_jobs)
    for job in jobs:
        expected_workers = 4 if job["dataset"] == "ppi" else 0
        assert job["workers"] == expected_workers
        assert _argument(job["command"], "--workers") == str(expected_workers)


def test_legacy_ppi_batch_override_reaches_v1_v3_v4_but_not_v5() -> None:
    args = runner.parser().parse_args(
        ["--datasets", "ppi", "--legacy-ppi-batch-size", "5"]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, Path("fixture"))
    for job in jobs:
        batch_size = int(_argument(job["command"], "--batch-size"))
        assert job["batch_size"] == batch_size
        if job["version"] in {"v1", "v3", "v4"}:
            assert batch_size == 5
        elif job["version"] == "v5":
            assert batch_size == 2


@pytest.mark.parametrize(
    "mutated_relative_path",
    [
        "scripts/run_conductance_factorial.py",
        "scripts/check_dependencies.py",
        "scripts/telemetry_validation.py",
        "src/chartgat/observability.py",
    ],
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
    assert len(jobs) == 22
    assert len({job["output_dir"] for job in jobs}) == len(jobs)
    for job in jobs:
        assert _argument(job["command"], "--hidden-channels") == "384"
        assert _argument(job["command"], "--layers") == "12"
        assert _argument(job["command"], "--dropout") == "0.2"
        expected_architecture = {
            key: runner.PROFILES["large"][key] for key in ("hidden_channels", "layers", "dropout")
        }
        if job["version"] == "v5":
            expected_architecture.update(
                heads=8,
                ffn_multiplier=4,
                beta_parameterization="sigmoid",
                beta_initial=0.1,
            )
        assert job["architecture"] == expected_architecture
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
    assert {job["version"] for job in jobs} == {"v1", "v3", "v4", "v5"}
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


def test_a6000_profile_runs_large_v5_work_first_and_binds_resolved_execution():
    args = runner.parser().parse_args(
        [
            "--versions",
            "v5",
            "--profiles",
            "reference",
            "--datasets",
            "cora",
            "ppi",
            "ogbn-arxiv",
            "--hardware-profile",
            "a6000-48gb",
        ]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, Path("fixture"))
    assert jobs[0]["dataset"] == "ogbn-arxiv"
    ppi = next(job for job in jobs if job["dataset"] == "ppi")
    assert ppi["execution"]["batch_size"] == 8
    assert ppi["execution"]["precision"] == "bf16"
    assert ppi["execution"]["tf32"] is True
    assert ppi["execution"]["activation_checkpoint"] is False
    assert ppi["command"].count("--edge-chunk-size") == 1
    assert _argument(ppi["command"], "--edge-chunk-size") == "131072"
    cora = next(job for job in jobs if job["dataset"] == "cora")
    assert "expected low occupancy" in cora["occupancy_expectation"]


def test_v5_margin_beta_ablation_is_validated_and_forwarded_by_scaling_runner():
    args = runner.parser().parse_args(
        [
            "--versions",
            "v5",
            "--profiles",
            "reference",
            "--datasets",
            "cora",
            "--v5-beta-parameterization",
            "margin_sigmoid",
            "--v5-beta-initial",
            "0.5",
            "--v5-beta-min",
            "0.05",
            "--v5-beta-max",
            "0.95",
        ]
    )
    runner._validate(args)
    jobs = runner.make_jobs(args, Path("fixture"))
    for job in jobs:
        assert {
            key: job["architecture"][key]
            for key in ("beta_parameterization", "beta_initial", "beta_min", "beta_max")
        } == {
            "beta_parameterization": "margin_sigmoid",
            "beta_initial": 0.5,
            "beta_min": 0.05,
            "beta_max": 0.95,
        }
        child = v5_train.build_parser().parse_args(job["command"][5:])
        v5_train.validate_args(child)
        assert (child.beta_parameterization, child.beta_initial) == ("margin_sigmoid", 0.5)
        assert (child.beta_min, child.beta_max) == (0.05, 0.95)


def test_v5_default_beta_rejects_irrelevant_margin_flags_in_scaling_runner():
    args = runner.parser().parse_args(["--v5-beta-min", "0.05"])
    with pytest.raises(ValueError, match="only valid for margin_sigmoid"):
        runner._validate(args)


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
            "reference",
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
        workers=4,
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
    assert all(loader.kwargs["num_workers"] == 4 for loader in loaders.values())
    assert all(loader.kwargs["persistent_workers"] is True for loader in loaders.values())
    assert all(loader.kwargs["prefetch_factor"] == 2 for loader in loaders.values())


def test_v1_child_worker_contract_is_dataset_specific():
    ppi = scaling_v1.build_parser().parse_args(
        ["--dataset", "ppi", "--output-dir", "out"]
    )
    scaling_v1._validate(ppi)
    assert ppi.workers == 4
    assert ppi.batch_size == 2
    assert ppi.worker_configuration_source == "dataset_default"
    cora = scaling_v1.build_parser().parse_args(
        ["--dataset", "cora", "--output-dir", "out"]
    )
    scaling_v1._validate(cora)
    assert cora.workers == 0
    assert cora.batch_size == 1
    assert cora.worker_configuration_source == "dataset_default"
    cora.workers = 1
    with pytest.raises(ValueError, match="no DataLoader"):
        scaling_v1._validate(cora)


def test_legacy_children_accept_explicit_ppi_batches_but_keep_full_graph_at_one():
    v1_ppi = scaling_v1.build_parser().parse_args(
        ["--dataset", "ppi", "--output-dir", "out", "--batch-size", "5"]
    )
    scaling_v1._validate(v1_ppi)
    assert v1_ppi.batch_size == 5

    for module, condition in (
        (v3_train, "relative_c"),
        (v4_train, "relative_c_spatial_w"),
    ):
        ppi = module.build_parser().parse_args(
            [
                "--dataset",
                "ppi",
                "--condition",
                condition,
                "--output-dir",
                "out",
                "--batch-size",
                "5",
            ]
        )
        module._validate_args(ppi)
        assert ppi.batch_size == 5

        full_graph = module.build_parser().parse_args(
            [
                "--dataset",
                "cora",
                "--condition",
                condition,
                "--output-dir",
                "out",
                "--batch-size",
                "2",
            ]
        )
        with pytest.raises(ValueError, match="one full graph"):
            module._validate_args(full_graph)


def test_direct_v3_v4_and_ablation_default_workers_are_dataset_specific():
    v3_ppi = v3_train.build_parser().parse_args(
        ["--dataset", "ppi", "--condition", "relative_c", "--output-dir", "out"]
    )
    v3_train._validate_args(v3_ppi)
    assert v3_ppi.workers == 4
    assert v3_ppi.worker_configuration_source == "dataset_default"

    v4_ppi = v4_train.build_parser().parse_args(
        [
            "--dataset",
            "ppi",
            "--condition",
            "relative_c_spatial_w",
            "--output-dir",
            "out",
        ]
    )
    v4_train._validate_args(v4_ppi)
    assert v4_ppi.workers == 4
    assert v4_ppi.worker_configuration_source == "dataset_default"

    ablation_ppi = ablation_train.build_parser().parse_args(
        ["--dataset", "ppi", "--condition", "baseline", "--output-dir", "out"]
    )
    ablation_train.resolve_worker_arguments(ablation_ppi)
    assert ablation_ppi.workers == 4
    assert ablation_ppi.worker_configuration_source == "dataset_default"

    ablation_cora = ablation_train.build_parser().parse_args(
        ["--dataset", "cora", "--condition", "baseline", "--output-dir", "out"]
    )
    ablation_train.resolve_worker_arguments(ablation_cora)
    assert ablation_cora.workers == 0


def _stub(
    tmp_path,
    monkeypatch,
    *,
    expose_test: bool = False,
    preflight_gpu: dict | None = None,
):
    calls: list[list[str]] = []
    monkeypatch.setattr(runner, "check_dependencies", lambda: {"unit_fixture_only": True})
    monkeypatch.setattr(runner, "_source_snapshot", lambda: {"unit-source": "stable"})

    def dispatch(command, log, environment):
        calls.append(command)
        if any(Path(part).name == "gpu_preflight.py" for part in command):
            json_out = Path(_argument(command, "--json-out"))
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "gpu": preflight_gpu
                        or {
                            "name": "NVIDIA RTX A6000 fixture",
                            "total_bytes": 48 * 1024**3,
                            "free_bytes": 47 * 1024**3,
                            "compute_capability": [8, 6],
                        },
                    }
                ),
                encoding="utf-8",
            )
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
            "configuration": architecture
            | {
                "workers": int(_argument(command, "--workers")),
                "batch_size": int(_argument(command, "--batch-size")),
            },
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
            "resource_observability": _resource_observability(),
            "throughput": {
                "scope": "unit fixture measured work",
                "optimizer_steps_per_second": 2.0,
            },
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
        "reference",
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
    preflight = manifest["gpu_preflight"]
    assert preflight["status"] == "passed"
    assert preflight["hardware_profile"] == "portable"
    assert preflight["sha256"] == runner._sha256(Path(preflight["path"]))
    assert preflight["gpu"]["name"] == "NVIDIA RTX A6000 fixture"
    assert preflight["requirements"] == runner._hardware_requirements("portable")
    assert all(job["status"] == "passed" for job in manifest["jobs"])
    assert summary["valid_for_validation_comparison"] is True
    assert summary["test_evaluated"] is False
    assert {row["n"] for row in summary["rows"]} == {2}
    assert all(row["validation_mean"] == 0.75 for row in summary["rows"])
    assert len(summary["runs"]) == 10
    assert all("resource_observability" in row for row in summary["runs"])
    assert all("throughput" in row for row in summary["runs"])


@pytest.mark.parametrize("field", ["resource_observability", "throughput"])
def test_child_telemetry_is_required_fail_closed(tmp_path, monkeypatch, field):
    options, _calls = _stub(tmp_path, monkeypatch)
    assert runner.main(options) == 0
    root = tmp_path / "conductance_gat/scaling/unit-fixture"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    job = manifest["jobs"][0]
    metrics_path = Path(job["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.pop(field)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        runner._load_child(job)


def test_hardware_preflight_schema_rejects_bool_and_malformed_gpu_fields():
    valid = {
        "status": "passed",
        "gpu": {
            "name": "NVIDIA RTX A6000 fixture",
            "total_bytes": 48 * 1024**3,
            "free_bytes": 47 * 1024**3,
            "compute_capability": [8, 6],
        },
    }
    accepted = runner._validate_hardware_preflight(valid, "a6000-48gb")
    assert accepted["hardware_profile"] == "a6000-48gb"
    assert accepted["requirements"]["minimum_total_memory_bytes"] == 40 * 1024**3

    for key, value in (
        ("total_bytes", True),
        ("free_bytes", False),
        ("compute_capability", [True, 0]),
    ):
        malformed = json.loads(json.dumps(valid))
        malformed["gpu"][key] = value
        with pytest.raises(RuntimeError, match="GPU preflight"):
            runner._validate_hardware_preflight(malformed, "a6000-48gb")


@pytest.mark.parametrize(
    "gpu,error",
    [
        (
            {
                "name": "MIG 1g.10gb fixture",
                "total_bytes": 10 * 1024**3,
                "free_bytes": 9 * 1024**3,
                "compute_capability": [8, 0],
            },
            "40 GiB",
        ),
        (
            {
                "name": "NVIDIA RTX A6000 fixture",
                "total_bytes": 48 * 1024**3,
                "free_bytes": 31 * 1024**3,
                "compute_capability": [8, 6],
            },
            "32 GiB",
        ),
        (
            {
                "name": "48 GiB legacy fixture",
                "total_bytes": 48 * 1024**3,
                "free_bytes": 47 * 1024**3,
                "compute_capability": [7, 5],
            },
            "8.0",
        ),
    ],
)
def test_a6000_hardware_preflight_enforces_capacity_free_memory_and_capability(gpu, error):
    with pytest.raises(RuntimeError, match=error):
        runner._validate_hardware_preflight({"status": "passed", "gpu": gpu}, "a6000-48gb")


def test_a6000_legacy_only_plan_rejects_mig_before_any_child_launch(tmp_path, monkeypatch):
    options, calls = _stub(
        tmp_path,
        monkeypatch,
        preflight_gpu={
            "name": "MIG 1g.10gb fixture",
            "total_bytes": 10 * 1024**3,
            "free_bytes": 9 * 1024**3,
            "compute_capability": [8, 0],
        },
    )
    options.remove("v4")
    options += ["--hardware-profile", "a6000-48gb"]

    assert runner.main(options) == 1
    assert len(calls) == 1
    assert any(Path(part).name == "gpu_preflight.py" for part in calls[0])
    manifest = json.loads(
        (tmp_path / "conductance_gat/scaling/unit-fixture/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert all(job["status"] == "pending" for job in manifest["jobs"])
    assert "gpu_preflight" not in manifest


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


def test_incomplete_child_symlink_cannot_move_a_passed_sibling(tmp_path):
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
        runner._preserve_incomplete_child({"output_dir": str(indirect)}, run_dir)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_incomplete_child_is_preserved_in_unique_sibling_before_retry(tmp_path, capsys):
    run_dir = tmp_path / "run"
    output = run_dir / "children" / "candidate"
    output.mkdir(parents=True)
    marker = output / "partial.log"
    marker.write_text("diagnostic evidence", encoding="utf-8")
    job = {"output_dir": str(output), "status": "failed"}

    runner._preserve_incomplete_child(job, run_dir)

    preserved = output.with_name("candidate.preserved-attempt-1")
    assert not output.exists()
    assert (preserved / marker.name).read_text(encoding="utf-8") == "diagnostic evidence"
    assert job["preserved_incomplete_outputs"] == [
        {
            "event": "preserve_incomplete_child_output",
            "source": str(output),
            "destination": str(preserved),
            "reason": "no resumable checkpoint exists; preserve prior artifacts before retry",
        }
    ]
    report = json.loads(capsys.readouterr().err)
    assert report["source"] == str(output)
    assert report["destination"] == str(preserved)


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


def test_failure_output_write_errors_do_not_replace_child_validation_return_code(
    tmp_path, monkeypatch, capsys
):
    options, _calls = _stub(tmp_path, monkeypatch, expose_test=True)
    real_atomic_write = runner.atomic_write_json
    real_write_summary = runner._write_summary

    def atomic_write(path, payload):
        if payload.get("status") == "failed":
            raise OSError("manifest disk failure")
        return real_atomic_write(path, payload)

    def write_summary(run_dir, manifest):
        if manifest.get("status") == "failed":
            raise OSError("summary disk failure")
        return real_write_summary(run_dir, manifest)

    monkeypatch.setattr(runner, "atomic_write_json", atomic_write)
    monkeypatch.setattr(runner, "_write_summary", write_summary)

    assert runner.main(options) == 1
    stderr = capsys.readouterr().err
    assert "summary disk failure" in stderr
    assert "manifest disk failure" in stderr
