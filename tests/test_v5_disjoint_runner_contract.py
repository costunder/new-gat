"""CPU-only orchestration/configuration tests; no training or result writes."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5 import protocol, train
from scripts import run_conductance_scaling as scaling
from scripts import run_conductance_v5 as standalone
from scripts import run_rich_scaling as rich

CONTEXT_KEYS = {"sample_context_seed_batch_size", "sample_context_workers"}


def context_options(prefix=""):
    return [
        f"--{prefix}sampling",
        "auto_disjoint",
        f"--{prefix}sample-context-seed-batch-size",
        "2048",
        f"--{prefix}sample-context-workers",
        "2",
    ]


@pytest.mark.parametrize("dataset", ["ogbn-arxiv", "ppi", "cora", "citeseer", "pubmed"])
def test_legacy_auto_stable_and_new_recipe_explicit(dataset):
    assert protocol.resolve_sampling(dataset, "auto") == (
        "cluster" if dataset == "ogbn-arxiv" else "full"
    )
    assert protocol.resolve_sampling(dataset, "auto_disjoint") == (
        "cluster_disjoint" if dataset == "ogbn-arxiv" else "full"
    )
    assert protocol.resolve_sampling(dataset, "neighbor") == "neighbor"


@pytest.mark.parametrize("size", [None, 0, -1, True, 2.5])
def test_context_recipe_requires_explicit_valid_size(size):
    args = SimpleNamespace(
        sampling="auto_disjoint", sample_context_seed_batch_size=size, sample_context_workers=2
    )
    with pytest.raises(ValueError, match="positive explicit context"):
        protocol.sampling_context_configuration(args)


def test_context_options_do_not_leak_into_full_dataset_execution():
    args = SimpleNamespace(
        sampling="auto_disjoint", sample_context_seed_batch_size=2048, sample_context_workers=2
    )
    assert protocol.sampling_context_configuration(args, sampling="full") == {}
    assert protocol.sampling_context_configuration(args, sampling="cluster_disjoint") == {
        "sample_context_seed_batch_size": 2048,
        "sample_context_workers": 2,
    }
    args.sampling = "auto"
    with pytest.raises(ValueError, match="explicit cluster_disjoint/auto_disjoint"):
        protocol.sampling_context_configuration(args)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_context_workers_are_not_silently_coerced(workers):
    args = SimpleNamespace(
        sampling="cluster_disjoint",
        sample_context_seed_batch_size=2048,
        sample_context_workers=workers,
    )
    with pytest.raises(ValueError, match="positive integer"):
        protocol.sampling_context_configuration(args)


def test_standalone_execution_and_real_child_configuration_match_without_writes(tmp_path):
    args = standalone.parser().parse_args(
        [
            "--datasets",
            "ogbn-arxiv",
            "ppi",
            "cora",
            "--hardware-profile",
            "a6000-48gb",
            "--sample-seed-batch-size",
            "8192",
            *context_options(),
        ]
    )
    standalone._validate(args)
    jobs = standalone.make_jobs(args, tmp_path / "not_created", standalone._architecture(args))
    assert len(jobs) == 6
    for job in jobs:
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        configuration = train.configuration(child)
        if job["dataset"] == "ogbn-arxiv":
            assert child.sampling == "cluster_disjoint"
            assert child.sample_seed_batch_size == 8192
            assert child.sample_context_seed_batch_size == 2048
            assert child.sample_context_workers == 2
            for key in CONTEXT_KEYS:
                assert configuration[key] == job["execution"][key]
        else:
            assert child.sampling == "full"
            assert not CONTEXT_KEYS.intersection(configuration)
            assert not CONTEXT_KEYS.intersection(job["execution"])
            assert "--sample-context-seed-batch-size" not in job["command"]
            assert "--sample-context-workers" not in job["command"]
        assert child.hidden_channels == 256 and child.layers == 8 and child.heads == 8
    assert list(tmp_path.iterdir()) == []


def test_nested_rich_scaling_forwards_sampling_and_context_to_real_train_parser(tmp_path):
    options = [
        "--tracks",
        "conductance",
        "--conductance-versions",
        "v5",
        "--profiles",
        "reference",
        "--model-seeds",
        "0",
        "--hardware-profile",
        "a6000-48gb",
        "--conductance-v5-sample-seed-batch-size",
        "8192",
        "--v5-num-neighbors",
        "15",
        "10",
        *context_options("v5-"),
    ]
    args = rich.parser().parse_args(options)
    rich._validate(args)
    jobs = rich.make_jobs(args, "disjoint-plan")
    assert len(jobs) == 1
    scale_args = scaling.parser().parse_args(jobs[0]["command"][3:])
    scaling._validate(scale_args)
    assert scale_args.v5_sampling == "auto_disjoint"
    assert scale_args.v5_num_neighbors == [15, 10]
    child_jobs = scaling.make_jobs(scale_args, tmp_path / "not_created")
    assert len(child_jobs) == 10
    for job in child_jobs:
        child = train.build_parser().parse_args(job["command"][5:])
        train.validate_args(child)
        if child.dataset == "ogbn-arxiv":
            assert child.sampling == "cluster_disjoint"
            assert child.sample_context_seed_batch_size == 2048
            assert child.sample_context_workers == 2
            assert child.sample_seed_batch_size == 8192
        else:
            assert child.sampling == "full"
            assert child.sample_context_seed_batch_size is None
            assert child.sample_context_workers is None
    assert list(tmp_path.iterdir()) == []


def test_dry_run_emits_new_context_contract_without_processes_or_result_directories(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(rich, "_run_logged", lambda *_: pytest.fail("dry run launched a child"))
    code = rich.main(
        [
            "--tracks",
            "conductance",
            "--conductance-versions",
            "v5",
            "--profiles",
            "reference",
            "--model-seeds",
            "0",
            "--hardware-profile",
            "a6000-48gb",
            "--results-root",
            str(tmp_path),
            "--run-id",
            "disjoint-plan",
            "--dry-run",
            *context_options("v5-"),
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "--v5-sampling auto_disjoint" in output
    assert "--v5-sample-context-seed-batch-size 2048" in output
    assert "--v5-sample-context-workers 2" in output
    assert "execution_classification=plan_only" in output
    assert "no files or directories were written" in output
    assert list(tmp_path.iterdir()) == []


def test_legacy_configs_are_not_polluted_and_new_sampling_has_distinct_job_identity(tmp_path):
    old_args = standalone.parser().parse_args(["--datasets", "ogbn-arxiv"])
    old_job = standalone.make_jobs(old_args, tmp_path, standalone._architecture(old_args))[0]
    old_child = train.build_parser().parse_args(old_job["command"][5:])
    train.validate_args(old_child)
    assert not CONTEXT_KEYS.intersection(train.configuration(old_child))
    assert not CONTEXT_KEYS.intersection(old_job["execution"])
    old_rich = rich.parser().parse_args(["--tracks", "conductance", "--conductance-versions", "v5"])
    assert "v5_sampling_recipe" not in rich._config_payload(
        old_rich, data_root=tmp_path, results_root=tmp_path
    )
    base = ["--versions", "v5", "--datasets", "ogbn-arxiv", "--profiles", "reference"]
    old = scaling.make_jobs(scaling.parser().parse_args(base), tmp_path)
    new = scaling.make_jobs(scaling.parser().parse_args([*base, *context_options("v5-")]), tmp_path)
    assert scaling._job_identity(old[0]) != scaling._job_identity(new[0])
    assert old[0]["architecture"] == new[0]["architecture"]


def test_new_sampling_cannot_silently_resume_legacy_run(tmp_path):
    base = ["--tracks", "conductance", "--conductance-versions", "v5"]
    legacy = rich._config_payload(
        rich.parser().parse_args(base), data_root=tmp_path, results_root=tmp_path
    )
    corrected = rich._config_payload(
        rich.parser().parse_args([*base, *context_options("v5-")]),
        data_root=tmp_path,
        results_root=tmp_path,
    )
    serialized = json.dumps(
        {
            "schema_version": 1,
            "suite": "rich_scaling",
            "run_id": "existing",
            "config": legacy,
        }
    )
    # A read-only manifest test double; no historical file is created or modified.
    manifest = SimpleNamespace(read_text=lambda **_: serialized)
    with pytest.raises(ValueError, match="new run ID"):
        rich._resume_manifest(
            manifest,
            run_id="existing",
            expected_config=corrected,
            expected_jobs=[],
            expected_totals={},
            expected_sources={},
        )
    assert manifest.read_text() == serialized


def test_unused_v5_context_options_leave_cycle_tree_job_and_config_identity_unchanged(tmp_path):
    base = rich.parser().parse_args(["--tracks", "cycle", "tree"])
    new = rich.parser().parse_args(["--tracks", "cycle", "tree", *context_options("v5-")])
    assert rich.make_jobs(base, "untouched") == rich.make_jobs(new, "untouched")
    assert rich._config_payload(
        base, data_root=tmp_path, results_root=tmp_path
    ) == rich._config_payload(
        new,
        data_root=tmp_path,
        results_root=tmp_path,
    )


@pytest.fixture
def prepare_inputs(monkeypatch):
    module = ModuleType("torch_geometric.data")
    module.Data = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", module)
    n = 40
    train_mask = torch.arange(n) < 20
    payload = {
        "graphs": [
            {
                "x": torch.ones(n, 3),
                "y": torch.arange(n) % 2,
                "incidence_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            }
        ],
        "splits": {"train": train_mask, "validation": ~train_mask},
    }
    calls = []

    def captured_sampler(graph, indices, **kwargs):
        calls.append((graph, indices, kwargs))
        return "synthetic_sampler_constructor_capture"

    monkeypatch.setattr(train, "TransductiveGraphSampler", captured_sampler)
    return payload, calls


def prepare_args(physical):
    return SimpleNamespace(
        dataset="ogbn-arxiv",
        sampling="cluster_disjoint",
        model_seed=0,
        sample_seed_batch_size=physical,
        num_neighbors=[15, 10],
        sample_context_seed_batch_size=4,
        sample_context_workers=2,
    )


def test_prepare_data_connects_explicit_context_without_altering_seed_batch(prepare_inputs):
    payload, calls = prepare_inputs
    graph, indices, sampler = train._prepare_data(payload, prepare_args(8), torch.device("cpu"))
    assert sampler == "synthetic_sampler_constructor_capture"
    assert torch.equal(indices["train"], torch.arange(20))
    assert calls[0][2] == {
        "mode": "cluster_disjoint",
        "seed_batch_size": 8,
        "fanouts": [15, 10],
        "model_seed": 0,
        "context_seed_batch_size": 4,
        "context_workers": 2,
    }


@pytest.mark.parametrize("physical", [2, 6])
def test_prepare_data_rejects_physical_candidates_that_change_context_partition(
    prepare_inputs, physical
):
    payload, calls = prepare_inputs
    with pytest.raises(ValueError, match="context"):
        train._prepare_data(payload, prepare_args(physical), torch.device("cpu"))
    assert not calls


def test_single_complete_seed_epoch_allows_final_partial_context(prepare_inputs):
    payload, calls = prepare_inputs
    train._prepare_data(payload, prepare_args(21), torch.device("cpu"))
    assert calls[0][2]["seed_batch_size"] == 21


def test_measured_context_workers_do_not_silently_override_explicit_request(monkeypatch):
    args = scaling.parser().parse_args(
        [
            "--versions",
            "v5",
            "--hardware-profile",
            "a6000-48gb",
            *context_options("v5-"),
        ]
    )
    args.resolved_resource_plan = {"synthetic_selection_contract": True}
    monkeypatch.setattr(
        scaling,
        "selected_resources",
        lambda *_args, **_kwargs: {
            "batch_size": 1,
            "sample_seed_batch_size": 8192,
            "workers": 0,
            "sample_context_workers": 4,
        },
    )
    with pytest.raises(ValueError, match="context.*conflicts|context.*plan"):
        scaling._v5_execution(args, "ogbn-arxiv", "reference")


def test_measured_context_workers_replace_only_unmeasured_initial_candidate(monkeypatch):
    args = scaling.parser().parse_args(
        [
            "--versions",
            "v5",
            "--hardware-profile",
            "a6000-48gb",
            "--v5-sampling",
            "auto_disjoint",
            "--v5-sample-context-seed-batch-size",
            "2048",
        ]
    )
    args.resolved_resource_plan = {"synthetic_selection_contract": True}
    monkeypatch.setattr(
        scaling,
        "selected_resources",
        lambda *_args, **_kwargs: {
            "batch_size": 1,
            "sample_seed_batch_size": 8192,
            "workers": 0,
            "sample_context_workers": 8,
        },
    )
    actual = scaling._v5_execution(args, "ogbn-arxiv", "reference")
    assert actual["sample_context_workers"] == 8
    assert actual["sample_context_seed_batch_size"] == 2048
    assert actual["sample_seed_batch_size"] == 8192
    assert actual["dataloader_workers"] == 0


def test_old_loader_worker_measurement_is_not_claimed_as_disjoint_context_measurement(monkeypatch):
    args = scaling.parser().parse_args(["--versions", "v5", *context_options("v5-")])
    args.resolved_resource_plan = {"synthetic_legacy_selection": True}
    monkeypatch.setattr(
        scaling,
        "selected_resources",
        lambda *_args, **_kwargs: {
            "batch_size": 1,
            "sample_seed_batch_size": 8192,
            "workers": 0,
        },
    )
    with pytest.raises(ValueError, match="own measured context worker"):
        scaling._v5_execution(args, "ogbn-arxiv", "reference")
