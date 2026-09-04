"""Performance-tool contracts using unit tensors/mocks, never a speed experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import benchmark_speed as speed


def test_help_needs_no_site_packages_or_gpu():
    result = subprocess.run(
        [sys.executable, "-S", str(Path(speed.__file__)), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--include-compile" in result.stdout
    assert "--batch-sizes" in result.stdout
    assert "--track" in result.stdout
    assert "legacy V1 classifier" in result.stdout


@pytest.mark.parametrize(
    ("track", "dataset", "batch_size"),
    [
        ("conductance_gat", "cora", 2),
        ("cycle_pe_v1", "zinc12k", 32),
        ("cycle_pe_v2", "zinc12k", 32),
        ("tree_augmentation", "csl", 16),
    ],
)
def test_defaults_resolve_to_official_data_and_no_compile(track, dataset, batch_size):
    args = speed.build_parser().parse_args(["--track", track])
    speed._validate(args)
    assert args.dataset == dataset and args.batch_size == batch_size
    assert args.batch_sizes == [batch_size]
    assert args.steps == 20 and args.warmup == 5 and not args.include_compile
    assert args.minimum_measure_seconds == 2.0
    assert args.resource_sample_interval_seconds == 0.1
    assert args.data_root == speed.ROOT / "data/paper"
    assert args.tree_data_root == speed.ROOT / "research/tree_augmentation/data"


@pytest.mark.parametrize(
    "extra",
    [
        ["--device", "cpu"],
        ["--dataset", "toy"],
        ["--steps", "0"],
        ["--warmup", "0"],
        ["--batch-size", "0"],
        ["--batch-sizes", "1", "1"],
        ["--batch-sizes", "1", "2"],
        ["--resource-sample-interval-seconds", "0"],
        ["--minimum-measure-seconds", "0"],
        ["--seed", "-1"],
    ],
)
def test_invalid_request_fails_before_artifacts(tmp_path, extra):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        speed.main(["--track", "conductance_gat", "--output-dir", str(output), *extra])
    assert caught.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "track,dataset",
    [
        ("conductance_gat", "ppi"),
        ("cycle_pe_v1", "zinc12k"),
        ("cycle_pe_v2", "zinc12k"),
        ("tree_augmentation", "zinc"),
    ],
)
def test_minibatch_tracks_accept_multiple_physical_candidates(track, dataset):
    args = speed.build_parser().parse_args(
        ["--track", track, "--dataset", dataset, "--batch-sizes", "2", "8", "16"]
    )
    speed._validate(args)
    assert args.batch_size is None
    assert args.batch_sizes == [2, 8, 16]


def test_current_only_tracks_have_explicit_variant_policy_and_tree_rejects_compile():
    cycle = speed.build_parser().parse_args(["--track", "cycle_pe_v1", "--include-compile"])
    speed._validate(cycle)
    assert speed._planned_variants(cycle) == ["current", "compiled"]

    tree = speed.build_parser().parse_args(["--track", "tree_augmentation"])
    speed._validate(tree)
    assert speed._planned_variants(tree) == ["current"]
    assert tree.tree_precision == "float16_autocast"

    unsupported = speed.build_parser().parse_args(
        ["--track", "tree_augmentation", "--include-compile"]
    )
    with pytest.raises(ValueError, match="does not support torch.compile"):
        speed._validate(unsupported)


def test_v5_a6000_ogbn_defaults_to_exact_cluster_seed_batch_profile():
    args = speed.build_parser().parse_args(
        [
            "--track",
            "conductance_v5",
            "--dataset",
            "ogbn-arxiv",
            "--v5-hardware-profile",
            "a6000-48gb",
        ]
    )
    speed._validate(args)
    assert args.v5_sampling_resolved == "cluster"
    assert args.batch_size == 2048 and args.batch_sizes == [2048]
    assert args.v5_precision == "bf16" and args.v5_tf32 is True
    assert speed._physical_batch_size_applicable(args) is True


def test_v5_sample_seed_candidates_are_accepted_but_full_graph_sweep_is_rejected():
    sampled = speed.build_parser().parse_args(
        [
            "--track",
            "conductance_v5",
            "--dataset",
            "ogbn-arxiv",
            "--batch-sizes",
            "1024",
            "2048",
            "4096",
        ]
    )
    speed._validate(sampled)
    assert sampled.v5_sampling_resolved == "cluster"
    assert sampled.batch_sizes == [1024, 2048, 4096]
    full = speed.build_parser().parse_args(
        [
            "--track",
            "conductance_v5",
            "--dataset",
            "ogbn-arxiv",
            "--v5-sampling",
            "full",
            "--batch-sizes",
            "1",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="full graph"):
        speed._validate(full)


def test_v5_rejects_sampling_ppi_and_nonpositive_fanout():
    ppi = speed.build_parser().parse_args(
        [
            "--track",
            "conductance_v5",
            "--dataset",
            "ppi",
            "--v5-sampling",
            "neighbor",
        ]
    )
    with pytest.raises(ValueError, match="PPI.*inapplicable"):
        speed._validate(ppi)
    fanout = speed.build_parser().parse_args(
        [
            "--track",
            "conductance_v5",
            "--dataset",
            "ogbn-arxiv",
            "--v5-num-neighbors",
            "15",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="fanout"):
        speed._validate(fanout)
    tiny = speed.build_parser().parse_args(
        [
            "--track",
            "conductance_v5",
            "--dataset",
            "ogbn-arxiv",
            "--batch-sizes",
            "16",
            "32",
        ]
    )
    with pytest.raises(ValueError, match="below 32"):
        speed._validate(tiny)
    wrong_full_value = speed.build_parser().parse_args(
        [
            "--track",
            "conductance_v5",
            "--dataset",
            "cora",
            "--batch-size",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="must be 1"):
        speed._validate(wrong_full_value)


def test_legacy_batch_size_and_candidate_list_are_mutually_exclusive():
    parser = speed.build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(
            [
                "--track",
                "cycle_pe_v2",
                "--batch-size",
                "8",
                "--batch-sizes",
                "8",
                "16",
            ]
        )
    assert caught.value.code == 2


def test_cpu_gpu_absence_has_no_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU"):
        speed._require_cuda("cuda")


@pytest.mark.parametrize(("requested", "resolved"), [("cuda", "cuda:2"), ("cuda:1", "cuda:1")])
def test_cuda_device_resolves_index_and_sets_event_device(monkeypatch, requested, resolved):
    selected = []
    checked = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(str(device)))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: checked.append(str(device))
    )
    assert str(speed._require_cuda(requested)) == resolved
    assert selected == checked == [resolved]


@pytest.mark.parametrize("existing_report", [True, False])
def test_existing_output_even_empty_is_not_overwritten(tmp_path, existing_report):
    output = tmp_path / "owned"
    output.mkdir()
    if existing_report:
        (output / "report.json").write_text("existing report", encoding="utf-8")
    with pytest.raises(FileExistsError):
        speed.main(["--track", "conductance_gat", "--output-dir", str(output)])
    if existing_report:
        assert (output / "report.json").read_text(encoding="utf-8") == "existing report"
    else:
        assert list(output.iterdir()) == []


def test_execution_error_writes_failed_report_not_success(monkeypatch, tmp_path):
    def fail(*args):
        raise RuntimeError("unit-mocked compiler error")

    monkeypatch.setattr(speed, "_execute", fail)
    output = tmp_path / "failed"
    assert (
        speed.main(
            [
                "--track",
                "cycle_pe_v2",
                "--include-compile",
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "compiler error" in report["error"]
    assert report["variants"] == []
    assert report["controls"]["optimizer_steps"] == 0
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == []


def test_partial_results_stay_failed_on_optional_compile_error(monkeypatch, tmp_path):
    def fail_after_reference(args, report, output):
        report["variants"].append({"variant": "reference", "measured_steps": 0})
        report["active_variant"] = "compiled"
        raise RuntimeError("unit-mocked optional compile failure")

    monkeypatch.setattr(speed, "_execute", fail_after_reference)
    output = tmp_path / "partial"
    assert (
        speed.main(
            [
                "--track",
                "conductance_gat",
                "--include-compile",
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["active_variant"] == "compiled"
    assert report["variants"][0]["variant"] == "reference"


def _unit_case():
    batch = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    return speed.SpeedCase(
        batch=batch,
        make_model=lambda _: torch.nn.Linear(2, 1),
        objective=lambda prediction: prediction.square().mean(),
        protocol={},
        description={},
        comparison_scope="unit tensor only",
    )


def test_probe_checks_forward_and_every_gradient_without_updating_parameters():
    torch.manual_seed(7)
    case = _unit_case()
    model = case.make_model("reference")
    other = copy.deepcopy(model)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    reference, candidate = speed._probe(model, case), speed._probe(other, case)
    result = speed._compare_probes(reference, candidate)
    assert result["passed"] and result["parameter_gradients_compared"] == 2
    assert result["prediction_max_abs_error"] == result["gradient_max_abs_error"] == 0
    assert reference["integrity"] == {
        "status": "passed",
        "finite_prediction": True,
        "finite_loss": True,
        "all_trainable_parameters_have_gradients": True,
        "all_trainable_parameter_gradients_finite": True,
        "trainable_parameter_gradient_tensors": 2,
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }
    no_oracle = speed._equivalence_result(None, reference)
    assert no_oracle["status"] == "not_applicable"
    assert no_oracle["passed"] is None
    assert "self-comparison" in no_oracle["reason"]
    independent = speed._equivalence_result(reference, candidate)
    assert independent["status"] == "passed" and independent["passed"] is True
    for key, value in model.state_dict().items():
        assert torch.equal(initial[key], value)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_correctness_mismatch_stops_before_performance_acceptance():
    case = _unit_case()
    model = case.make_model("reference")
    reference = speed._probe(model, case)
    candidate = copy.deepcopy(reference)
    candidate["gradients"]["weight"].add_(1)
    with pytest.raises(AssertionError, match="gradient"):
        speed._compare_probes(reference, candidate)
    candidate = copy.deepcopy(reference)
    candidate["gradients"]["weight"] = None
    with pytest.raises(AssertionError, match="participation"):
        speed._compare_probes(reference, candidate)


def test_nonfinite_probe_fails_closed():
    case = _unit_case()
    model = case.make_model("reference")
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        speed._probe(model, case)


def test_probe_rejects_trainable_parameter_disconnected_from_loss():
    class Disconnected(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(2, 1)
            self.unused = torch.nn.Parameter(torch.ones(1))

        def forward(self, inputs):
            return self.used(inputs)

    case = speed.SpeedCase(
        batch=torch.ones(2, 2),
        make_model=lambda _: Disconnected(),
        objective=lambda prediction: prediction.square().mean(),
        protocol={},
        description={},
        comparison_scope="unit tensor",
    )
    with pytest.raises(AssertionError, match="disconnected.*unused"):
        speed._probe(Disconnected(), case)


def test_parameter_update_guard_detects_mutation():
    model = torch.nn.Linear(2, 1)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    speed._assert_trainable_parameters_unchanged(model, state)
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(AssertionError, match="without an optimizer"):
        speed._assert_trainable_parameters_unchanged(model, state)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (torch.cuda.OutOfMemoryError("candidate too large"), "cuda_out_of_memory"),
        (RuntimeError("CUDA out of memory while allocating"), "cuda_out_of_memory"),
        (ValueError("bad candidate"), "execution_error"),
    ],
)
def test_candidate_failure_is_explicit_and_never_reduces_batch(error, kind):
    metadata = speed._failure_metadata(error)
    assert metadata["status"] == "failed"
    assert metadata["failure_kind"] == kind
    assert metadata["fallback_or_automatic_batch_reduction_applied"] is False
    assert str(error) in metadata["error"]


def _run_variant_with_measured_failure(
    monkeypatch,
    *,
    measured_error: BaseException,
    monitor_error: BaseException | None,
):
    from chartgat import execution, observability

    class CPUOnlyLinear(torch.nn.Linear):
        def to(self, *_args, **_kwargs):
            return self

    class UnitMonitor:
        latest = None

        def __init__(self, *_args, **_kwargs):
            self.start_calls = 0
            self.finish_calls = 0
            UnitMonitor.latest = self

        def start(self):
            self.start_calls += 1

        def finish(self, **_kwargs):
            self.finish_calls += 1
            if monitor_error is not None:
                raise monitor_error
            return {"status": "unit monitor finished"}

    case = speed.SpeedCase(
        batch=torch.ones(2, 2),
        make_model=lambda _kind: CPUOnlyLinear(2, 1),
        objective=lambda prediction: prediction.square().mean(),
        protocol={"source": "unit"},
        description={
            "requested_physical_batch_size": 2,
            "actual_physical_batch_size": 2,
            "production_path_identity": {"model": "unit"},
        },
        comparison_scope="unit failure path",
    )
    initial = case.make_model("current")
    state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    probe = {"integrity": {"status": "passed"}}
    monkeypatch.setattr(execution, "configure_execution", lambda *_args: {"mode": "unit"})
    monkeypatch.setattr(observability, "RuntimeResourceMonitor", UnitMonitor)
    monkeypatch.setattr(speed, "_seed", lambda _seed: None)
    monkeypatch.setattr(speed, "_probe", lambda *_args: probe)
    monkeypatch.setattr(
        speed,
        "_measure_block",
        lambda *_args, **_kwargs: {"wall_seconds": 0.01},
    )

    def fail_measurement(*_args, **_kwargs):
        raise measured_error

    monkeypatch.setattr(speed, "_measure_for_minimum_duration", fail_measurement)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_args: 10)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_args: 20)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args: 30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_args: 40)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = argparse.Namespace(
        seed=0,
        warmup=1,
        steps=1,
        minimum_measure_seconds=0.1,
        resource_sample_interval_seconds=0.01,
    )
    return speed._run_variant(
        args,
        torch.device("cuda:0"),
        case,
        state,
        "current",
        None,
    ), UnitMonitor


def test_variant_oom_metadata_survives_monitor_cleanup_failure(monkeypatch):
    oom = torch.cuda.OutOfMemoryError("unit primary OOM")
    cleanup = RuntimeError("unit monitor cleanup")
    (row, _probe), monitor_type = _run_variant_with_measured_failure(
        monkeypatch,
        measured_error=oom,
        monitor_error=cleanup,
    )

    assert row["status"] == "failed"
    assert row["failure_kind"] == "cuda_out_of_memory"
    assert row["error"] == "OutOfMemoryError: unit primary OOM"
    assert row["fallback_or_automatic_batch_reduction_applied"] is False
    assert row["resource_monitor_finish"]["status"] == "failed"
    assert row["resource_monitor_finish"]["attempted"] is True
    assert row["resource_monitor_finish"]["errors"] == [
        {
            "stage": "runtime_resource_monitor_finish",
            "error_type": "RuntimeError",
            "error": "unit monitor cleanup",
        }
    ]
    assert "unit monitor cleanup" in row["resource_observability_error"]
    assert monitor_type.latest.start_calls == 1
    assert monitor_type.latest.finish_calls == 1
    assert any(
        "without replacing the primary error" in note for note in getattr(oom, "__notes__", [])
    )


def test_variant_oom_metadata_survives_post_failure_cleanup_error(monkeypatch):
    oom = torch.cuda.OutOfMemoryError("unit primary OOM")

    def fail_gc() -> None:
        raise RuntimeError("unit post-failure gc cleanup")

    monkeypatch.setattr(speed.gc, "collect", fail_gc)
    (row, _probe), _monitor_type = _run_variant_with_measured_failure(
        monkeypatch,
        measured_error=oom,
        monitor_error=None,
    )

    assert row["status"] == "failed"
    assert row["failure_kind"] == "cuda_out_of_memory"
    assert row["error"] == "OutOfMemoryError: unit primary OOM"
    assert row["post_variant_cleanup_errors"] == [
        {
            "stage": "python_gc",
            "error_type": "RuntimeError",
            "error": "unit post-failure gc cleanup",
        }
    ]
    assert any(
        "post-failure cleanup failed without replacing" in note
        for note in getattr(oom, "__notes__", [])
    )


def test_variant_keyboard_interrupt_finishes_monitor_once_and_reraises_same_object(
    monkeypatch,
):
    interrupt = KeyboardInterrupt("unit Ctrl-C")
    cleanup = RuntimeError("unit monitor cleanup during Ctrl-C")
    with pytest.raises(KeyboardInterrupt) as caught:
        _run_variant_with_measured_failure(
            monkeypatch,
            measured_error=interrupt,
            monitor_error=cleanup,
        )

    assert caught.value is interrupt
    from chartgat.observability import RuntimeResourceMonitor

    assert RuntimeResourceMonitor.latest.start_calls == 1
    assert RuntimeResourceMonitor.latest.finish_calls == 1
    assert any(
        "unit monitor cleanup during Ctrl-C" in note for note in getattr(interrupt, "__notes__", [])
    )


def test_measurement_extends_steps_until_gpu_sampling_duration_without_reduction(
    monkeypatch,
):
    calls = []

    def block(model, case, steps, device):
        calls.append(steps)
        return {
            "wall_seconds": steps * 0.25,
            "cuda_event_seconds": steps * 0.2,
            "seconds_per_step": 0.25,
            "steps_per_second": 4.0,
        }

    monkeypatch.setattr(speed, "_measure_block", block)
    result = speed._measure_for_minimum_duration(
        None,
        _unit_case(),
        requested_steps=2,
        minimum_seconds=1.0,
        device=torch.device("cpu"),
    )
    assert calls == [2, 2]
    assert result["requested_minimum_steps"] == 2
    assert result["measured_steps"] == 4
    assert result["wall_seconds"] == 1.0
    assert result["minimum_measure_duration_met"] is True
    assert result["measurement_blocks"] == 2


def test_measurement_never_reduces_requested_steps(monkeypatch):
    calls = []

    def block(model, case, steps, device):
        calls.append(steps)
        return {
            "wall_seconds": 3.0,
            "cuda_event_seconds": 2.5,
            "seconds_per_step": 3.0 / steps,
            "steps_per_second": steps / 3.0,
        }

    monkeypatch.setattr(speed, "_measure_block", block)
    result = speed._measure_for_minimum_duration(
        None,
        _unit_case(),
        requested_steps=20,
        minimum_seconds=2.0,
        device=torch.device("cpu"),
    )
    assert calls == [20]
    assert result["measured_steps"] == 20


def test_conductance_builder_uses_offline_loader_and_only_training_indices(monkeypatch, tmp_path):
    from research.conductance_gat import benchmark, benchmark_data

    seen = {}
    graph = SimpleNamespace(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        y=torch.tensor([0, 1, 0]),
        incidence_edge_index=torch.tensor([[0, 1], [1, 2]]),
    )

    def load(dataset, root, *, allow_download):
        seen.update(dataset=dataset, root=root, allow_download=allow_download)
        return {"classes": 2}, {"source": "unit mocked official-loader contract"}

    monkeypatch.setattr(benchmark_data, "load_dataset", load)
    monkeypatch.setattr(
        benchmark,
        "_make_loaders",
        lambda *args: (graph, {"train": torch.tensor([0]), "test": torch.tensor([1, 2])}),
    )
    args = argparse.Namespace(dataset="cora", data_root=tmp_path, seed=0, batch_size=2)
    case = speed._build_conductance_case(args, torch.device("cpu"))
    assert seen == {"dataset": "cora", "root": tmp_path, "allow_download": False}
    prediction = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    expected = torch.nn.functional.cross_entropy(prediction[:1], graph.y[:1])
    assert torch.equal(case.objective(prediction), expected)
    assert case.description["actual_physical_batch_size"] is None
    assert case.description["physical_batch_size_applicable"] is False
    assert "transductive full graph" in case.description["physical_batch_size_reason"]
    graph.y[1:] = 0
    assert torch.equal(case.objective(prediction), expected)
    reference = case.make_model("reference")
    optimized = case.make_model("optimized")
    optimized.load_state_dict(reference.state_dict())
    comparison = speed._compare_probes(speed._probe(reference, case), speed._probe(optimized, case))
    assert comparison["passed"]


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_cycle_builder_selects_train_only_and_no_download(monkeypatch, tmp_path, encoding):
    from research.cycle_pe.v2 import data

    seen = {}
    train_graphs = [object(), object(), object()]
    batch = SimpleNamespace(
        x=torch.zeros(4, 1, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        cycle_basis_shapes=((2, 1),),
        cycle_membership=torch.ones(2, 1).to_sparse().coalesce(),
        y=torch.zeros(1, 1),
    )
    batch.to = lambda _: batch

    def load(root, dataset, *, allow_download):
        seen.update(root=root, dataset=dataset, allow_download=allow_download)
        return {"train": train_graphs, "validation": object(), "test": object()}, {}

    def collate(graphs):
        seen["selected"] = graphs
        return batch

    monkeypatch.setattr(data, "load_benchmark", load)
    monkeypatch.setattr(data, "collate", collate)
    args = argparse.Namespace(
        dataset="zinc12k", data_root=tmp_path, batch_size=1, cycle_v2_encoding=encoding
    )
    case = speed._build_cycle_case(args, torch.device("cpu"))
    assert seen["allow_download"] is False and seen["selected"] == train_graphs[:1]
    assert case.description["cycle_memberships"] == 2
    assert case.description["actual_physical_batch_size"] == 1
    assert case.description["physical_batch_size_applicable"] is True
    assert not hasattr(case.make_model("current"), "basis_execution")
    expected_name = {"se": "cycle_dfs_se_v2", "pe": "cycle_dfs_relative_pe_v2"}[encoding]
    assert case.description["model_configuration"]["name"] == expected_name
    assert case.description["model_configuration"]["encoding"] == encoding
    assert case.make_model("current").encoding == encoding
    assert speed._planned_variants(
        argparse.Namespace(track="cycle_pe_v2", include_compile=True)
    ) == ["current", "compiled"]


def test_cycle_builder_rejects_candidate_larger_than_official_training_split(monkeypatch, tmp_path):
    from research.cycle_pe.v2 import data

    monkeypatch.setattr(
        data,
        "load_benchmark",
        lambda *args, **kwargs: ({"train": [object()]}, {}),
    )
    args = argparse.Namespace(dataset="zinc12k", data_root=tmp_path, batch_size=2)
    with pytest.raises(ValueError, match="contains 1 graphs"):
        speed._build_cycle_case(args, torch.device("cpu"))


def test_v5_builder_reuses_exact_sampled_training_batch_and_joint_phase(monkeypatch, tmp_path):
    from research.conductance_gat.v5 import model, train

    batch = SimpleNamespace(
        x=torch.ones(7, 3),
        y=torch.tensor([0, 1, 0, 1, 0, 1, 0]),
        incidence_edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    selected = torch.tensor([0, 2, 4, 6])
    seen = {}

    def prepare(payload, execution_args, device):
        seen["execution_args"] = execution_args
        return object(), {"train": selected}, object()

    def batches(data, indices, sampler, epoch, device, seed, execution_args):
        seen.update(epoch=epoch, seed=seed, device=device)
        yield batch, selected

    class DummyV5(torch.nn.Module):
        def __init__(self, in_channels, classes, **kwargs):
            super().__init__()
            self.projection = torch.nn.Linear(in_channels, classes)
            self.conductance_mode = kwargs["conductance_mode"]
            self.operators = []
            seen["model_kwargs"] = kwargs

        def forward(self, graph):
            return self.projection(graph.x)

    monkeypatch.setattr(train, "_prepare_data", prepare)
    monkeypatch.setattr(train, "_training_batches", batches)
    monkeypatch.setattr(
        train,
        "validate_hardware_runtime",
        lambda execution_args, device: {"status": "unit_validated"},
    )
    monkeypatch.setattr(
        train,
        "configure_phase",
        lambda instance, phase, phase_epoch: seen.update(phase=phase, phase_epoch=phase_epoch),
    )
    monkeypatch.setattr(model, "GraphConditionedConductanceNodeClassifier", DummyV5)
    args = argparse.Namespace(
        dataset="ogbn-arxiv",
        data_root=tmp_path,
        batch_size=4,
        seed=9,
        v5_hardware_profile="portable",
        v5_sampling_resolved="cluster",
        v5_num_neighbors=[15, 10],
        v5_scale_profile="reference",
        v5_condition="shared_dynamic_c",
    )
    payload = {"dataset": "ogbn-arxiv", "classes": 2, "graphs": [{"x": batch.x}]}
    case = speed._build_v5_case(
        args,
        torch.device("cpu"),
        (payload, {"source": "unit official cache"}),
    )
    assert seen["execution_args"].sample_seed_batch_size == 4
    assert seen["execution_args"].sampling == "cluster"
    assert seen["epoch"] == 1 and seen["seed"] == 9
    assert case.description["actual_physical_batch_size"] == 4
    assert case.description["physical_batch_size_unit"] == "seed_nodes"
    assert "_training_batches" in case.description["production_path_identity"]["training_batch"]
    assert case.description["production_path_identity"]["loss"].endswith(".training_loss")
    assert case.description["v5_architecture"]["hidden_channels"] == 256
    candidate = case.make_model("current")
    assert seen["phase"] == "joint" and seen["phase_epoch"] == 0
    assert candidate.conductance_mode == "dynamic"
    expected = torch.nn.functional.cross_entropy(
        candidate(batch).index_select(0, selected),
        batch.y.index_select(0, selected),
    )
    assert torch.equal(case.objective(candidate(batch)), expected)


def _resource_observation_for_test(*, free_bytes=900, total_bytes=1000):
    def value(number):
        return {"value": number, "reason": None}

    return {
        "start": {
            "gpu": {
                "device_free_bytes": value(free_bytes),
                "device_total_bytes": value(total_bytes),
            }
        },
        "interval_series": {
            "gpu_sm_utilization_percent": {"mean": value(70.0), "maximum": value(90.0)},
            "gpu_memory_controller_utilization_percent": {
                "mean": value(40.0),
                "maximum": value(55.0),
            },
            "process_resident_bytes": {"maximum": value(1234)},
            "system_available_bytes": {"minimum": value(5678)},
        },
        "summary": {
            "average_cpu_percent_of_allocated_capacity": value(12.5),
        },
    }


def test_resource_and_throughput_columns_are_reported_per_variant():
    resource = _resource_observation_for_test()
    row = {"wall_seconds": 2.0, "measured_steps": 4}
    case = speed.SpeedCase(
        batch=None,
        make_model=lambda _: None,
        objective=lambda _: None,
        protocol={},
        description={
            "graphs": 8,
            "nodes": 80,
            "physical_edges": 160,
            "labels_in_loss": 16,
            "actual_physical_batch_size": 8,
            "physical_batch_size_unit": "graphs",
        },
        comparison_scope="unit metadata",
    )
    speed._add_resource_and_throughput_columns(row, resource, case)
    assert row["graphs_per_second"] == 16
    assert row["physical_batch_items_per_second"] == 16
    assert row["physical_batch_item_unit"] == "graphs"
    assert row["nodes_per_second"] == 160
    assert row["gpu_sm_utilization_mean_percent"] == 70
    assert row["gpu_memory_controller_utilization_max_percent"] == 55
    assert row["average_cpu_percent_of_allocated_capacity"] == 12.5
    assert row["peak_process_resident_bytes"] == 1234


def test_batch_recommendation_selects_fastest_safe_candidate_and_ignores_failure():
    resource = _resource_observation_for_test()

    def candidate(size, rate, incremental=0, status="passed"):
        variants = []
        if status == "passed":
            variants.append(
                {
                    "variant": "optimized",
                    "status": "passed",
                    "graphs_per_second": rate,
                    "physical_batch_items_per_second": rate,
                    "physical_batch_item_unit": "graphs",
                    "peak_cuda_reserved_incremental_bytes": incremental,
                    "resource_observability": resource,
                }
            )
        return {
            "requested_physical_batch_size": size,
            "status": status,
            "variants": variants,
        }

    result = speed._batch_candidate_analysis(
        [candidate(8, 80), candidate(16, 120), candidate(32, 0, status="failed")],
        transductive=False,
    )
    assert result["status"] == "informational_microbenchmark_ranking"
    assert result["selected_physical_batch_size"] is None
    assert result["training_batch_selection_performed"] is False
    assert result["highest_observed_microbenchmark_physical_batch_size"] == 16
    assert result["ranking_throughput_unit"] == "graphs"
    failed = result["candidate_evaluations"][2]
    assert failed["rankable_for_microbenchmark"] is False
    assert "did not complete" in failed["reason"]
    full_graph = speed._batch_candidate_analysis([], transductive=True)
    assert full_graph["status"] == "not_applicable"
    assert full_graph["selected_physical_batch_size"] is None
    assert full_graph["training_batch_selection_performed"] is False


def test_candidate_sweep_records_oom_and_continues_without_batch_fallback(monkeypatch, tmp_path):
    import chartgat.observability as observability

    resource = _resource_observation_for_test()
    case = speed.SpeedCase(
        batch=torch.ones(2, 2),
        make_model=lambda _: torch.nn.Linear(2, 1),
        objective=lambda prediction: prediction.square().mean(),
        protocol={"source": "unit official"},
        description={
            "graphs": 8,
            "nodes": 80,
            "physical_edges": 160,
            "labels_in_loss": 16,
            "requested_physical_batch_size": 8,
            "actual_physical_batch_size": 8,
            "physical_batch_size_unit": "graphs",
            "physical_batch_size_applicable": True,
        },
        comparison_scope="unit candidate control flow",
    )
    built = []

    def build(candidate_args, device, loaded):
        built.append(candidate_args.batch_size)
        if candidate_args.batch_size == 16:
            raise torch.cuda.OutOfMemoryError("unit candidate OOM")
        result = copy.copy(case)
        result.description = dict(
            case.description,
            requested_physical_batch_size=candidate_args.batch_size,
            actual_physical_batch_size=candidate_args.batch_size,
        )
        return result

    def run_variant(candidate_args, device, current_case, state, variant, reference):
        return (
            {
                "variant": variant,
                "status": "passed",
                "requested_physical_batch_size": candidate_args.batch_size,
                "actual_physical_batch_size": candidate_args.batch_size,
                "seconds_per_step": 1.0,
                "steps_per_second": 1.0,
                "graphs_per_second": float(candidate_args.batch_size),
                "physical_batch_items_per_second": float(candidate_args.batch_size),
                "physical_batch_item_unit": "graphs",
                "gpu_sm_utilization_mean_percent": 70.0,
                "gpu_sm_utilization_max_percent": 90.0,
                "peak_cuda_allocated_bytes": 100,
                "peak_cuda_reserved_incremental_bytes": 10,
                "resource_observability": resource,
            },
            {"unit": torch.tensor(1)},
        )

    monkeypatch.setattr(speed, "_require_cuda", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(speed, "_implementation_hashes", lambda _: {"unit": "digest"})
    monkeypatch.setattr(speed, "_load_case_inputs", lambda _: object())
    monkeypatch.setattr(speed, "_build_case", build)
    monkeypatch.setattr(speed, "_run_variant", run_variant)
    monkeypatch.setattr(
        observability, "runtime_resource_snapshot", lambda _: {"status": "unit snapshot"}
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Unit GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1000, multi_processor_count=1),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = argparse.Namespace(
        track="cycle_pe_v2",
        dataset="zinc12k",
        device="cuda:0",
        seed=0,
        batch_sizes=[8, 16, 32],
        include_compile=False,
        v5_tf32=None,
    )
    report = {"variants": [], "batch_candidates": []}
    output = tmp_path / "sweep"
    output.mkdir()
    with pytest.raises(RuntimeError, match="1 of 3.*no automatic"):
        speed._execute(args, report, output)
    assert built == [8, 16, 32]
    assert report["candidate_summary"] == {"planned": 3, "passed": 2, "failed": 1}
    assert [item["status"] for item in report["batch_candidates"]] == [
        "passed",
        "failed",
        "passed",
    ]
    failed = report["batch_candidates"][1]
    assert failed["failure_kind"] == "cuda_out_of_memory"
    assert failed["fallback_or_automatic_batch_reduction_applied"] is False
    assert "unit candidate OOM" in failed["error"]


def test_execute_persists_interrupted_candidate_and_reraises_same_object(monkeypatch, tmp_path):
    import chartgat.observability as observability

    interrupt = KeyboardInterrupt("unit execute Ctrl-C")
    case = speed.SpeedCase(
        batch=torch.ones(2, 2),
        make_model=lambda _kind: torch.nn.Linear(2, 1),
        objective=lambda prediction: prediction.square().mean(),
        protocol={"source": "unit official"},
        description={
            "requested_physical_batch_size": 8,
            "actual_physical_batch_size": 8,
            "physical_batch_size_unit": "graphs",
            "physical_batch_size_applicable": True,
        },
        comparison_scope="unit execute interrupt",
    )
    monkeypatch.setattr(speed, "_require_cuda", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(speed, "_implementation_hashes", lambda _: {"unit": "digest"})
    monkeypatch.setattr(speed, "_load_case_inputs", lambda _: object())
    monkeypatch.setattr(speed, "_build_case", lambda *_args: case)

    def interrupt_variant(*_args, **_kwargs):
        raise interrupt

    monkeypatch.setattr(speed, "_run_variant", interrupt_variant)
    monkeypatch.setattr(
        observability, "runtime_resource_snapshot", lambda _: {"status": "unit snapshot"}
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Unit GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=1000, multi_processor_count=1),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    args = argparse.Namespace(
        track="cycle_pe_v1",
        dataset="zinc12k",
        device="cuda:0",
        seed=0,
        batch_sizes=[8],
        include_compile=False,
        v5_tf32=None,
    )
    report = {"status": "running", "variants": [], "batch_candidates": []}
    output = tmp_path / "interrupted"
    output.mkdir()
    with pytest.raises(KeyboardInterrupt) as caught:
        speed._execute(args, report, output)

    assert caught.value is interrupt
    assert report["status"] == "interrupted"
    candidate = report["batch_candidates"][0]
    assert candidate["status"] == "interrupted"
    assert candidate["failure_kind"] == "keyboard_interrupt"
    assert candidate["fallback_or_automatic_batch_reduction_applied"] is False
    persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "interrupted"
    assert persisted["batch_candidates"][0]["status"] == "interrupted"


def test_cycle_v1_builder_uses_exact_seeded_loader_model_and_mae(monkeypatch, tmp_path):
    from research.cycle_pe import benchmark, benchmark_models

    batch = SimpleNamespace(
        x=torch.ones(5, 1, dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        y=torch.tensor([[1.0], [3.0]]),
        ptr=torch.tensor([0, 2, 5]),
    )
    batch.to = lambda _device: batch
    seen = {}

    def loader(graphs, args, *, train):
        seen.update(
            graphs=graphs,
            model_seed=args.model_seed,
            batch_size=args.batch_size,
            workers=args.workers,
            train=train,
        )
        return [batch]

    class FakeCycleV1(torch.nn.Module):
        def __init__(self, **configuration):
            super().__init__()
            self.configuration = configuration
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, current_batch):
            return current_batch.y * self.weight

    monkeypatch.setattr(benchmark, "_loader", loader)
    monkeypatch.setattr(benchmark_models, "CyclePEModel", FakeCycleV1)
    monkeypatch.setattr(
        benchmark_models,
        "architecture_protocol",
        lambda: {"model": "exact-cycle-v1"},
    )
    train_graphs = [object(), object(), object()]
    args = argparse.Namespace(
        dataset="zinc12k",
        data_root=tmp_path,
        seed=7,
        batch_size=2,
    )
    case = speed._build_cycle_v1_case(
        args,
        torch.device("cpu"),
        ({"train": train_graphs}, {"source": "verified official"}),
    )

    assert seen == {
        "graphs": train_graphs,
        "model_seed": 7,
        "batch_size": 2,
        "workers": 0,
        "train": True,
    }
    assert case.description["actual_physical_batch_size"] == 2
    assert case.description["physical_batch_size_unit"] == "graphs"
    assert case.protocol["microbenchmark_train_split_graphs"] == 3
    assert case.description["production_path_identity"] == {
        "model": "research.cycle_pe.benchmark_models.CyclePEModel",
        "training_batch": "research.cycle_pe.benchmark._loader(train=True)",
        "loss": "research.cycle_pe.benchmark MAE training objective",
    }
    model = case.make_model("current")
    assert model.configuration == {
        "dataset": "zinc12k",
        "hidden": 64,
        "pe_dim": 32,
        "layers": 3,
    }
    predicted = torch.tensor([[2.0], [1.0]])
    assert torch.equal(case.objective(predicted), torch.tensor(1.5))
    assert "zero parameter updates" in case.comparison_scope


def _tree_view(graph_id: str, target: float):
    from research.tree_augmentation.paper_model import GraphChartView

    return GraphChartView(
        graph_id=graph_id,
        graph_family="unit",
        graph_status="id",
        chart_status="train_multi_bfs_dfs_families",
        num_nodes=3,
        edges=((0, 1), (1, 2), (0, 2)),
        basis=np.asarray([[1.0], [1.0], [-1.0]], dtype=np.float64),
        target=(target,),
        chart_name=f"chart-{graph_id}",
        tree_key=(0, 1),
    )


def _tree_loaded(tmp_path, *, task_type: str, target_names: tuple[str, ...]):
    seed_axes = SimpleNamespace(
        model=5,
        chart=5,
        to_manifest=lambda: {
            "data": 5,
            "split": 5,
            "chart": 5,
            "model": 5,
        },
    )
    views = [_tree_view("graph-a", 0.0), _tree_view("graph-b", 1.0)]
    dataset = SimpleNamespace(
        suite="csl" if task_type == "classification" else "zinc",
        task_type=task_type,
        target_names=target_names,
        data_sha256="verified-digest",
        records=(
            SimpleNamespace(split="train"),
            SimpleNamespace(split="validation"),
            SimpleNamespace(split="test"),
        ),
    )
    return {
        "dataset": dataset,
        "settings": {
            "hidden_dim": 128,
            "message_layers": 8,
            "amp": True,
            "pin_memory": True,
            "non_blocking": True,
        },
        "config_path": tmp_path / "config.yaml",
        "seed_axes": seed_axes,
        "views": {"fixed_bfs": views[:1], "multi_chart": views},
    }


def test_tree_builder_uses_exact_padded_sampler_classification_loss_and_unit(tmp_path):
    loaded = _tree_loaded(
        tmp_path,
        task_type="classification",
        target_names=tuple(f"class_{index}" for index in range(10)),
    )
    args = argparse.Namespace(
        dataset="csl",
        tree_arm="multi_chart",
        batch_size=4,
    )
    case = speed._build_tree_case(args, torch.device("cpu"), loaded)

    assert case.description["actual_physical_batch_size"] == 4
    assert case.description["physical_batch_size_unit"] == "chart_views"
    assert case.description["chart_views"] == 4
    assert case.description["unique_physical_graphs"] <= 2
    assert case.description["padded_input_shapes"]["basis"] == [4, 3, 1]
    assert case.protocol["dataset_cache_integrity"]["full_cache_loaded"] is True
    assert case.protocol["constructed_training_chart_views"] == 2
    assert case.protocol["official_training_graphs"] == 2
    assert "collate_chart_views" in case.description["production_path_identity"]["batch"]
    model = case.make_model("current")
    assert model.hidden_dim == 128 and model.message_layers == 8
    predicted = torch.zeros(4, 10)
    expected = torch.nn.functional.cross_entropy(predicted, case.batch.targets[:, 0].long())
    assert torch.equal(case.objective(predicted), expected)
    assert "chart views" in case.comparison_scope
    assert "zero parameter updates" in case.comparison_scope


def test_tree_builder_uses_all_arm_targets_for_exact_zinc_normalized_mse(tmp_path):
    loaded = _tree_loaded(
        tmp_path,
        task_type="regression",
        target_names=("constrained_logP",),
    )
    loaded["views"]["multi_chart"] = [
        _tree_view("graph-a", 1.0),
        _tree_view("graph-b", 3.0),
    ]
    args = argparse.Namespace(
        dataset="zinc",
        tree_arm="multi_chart",
        batch_size=3,
    )
    case = speed._build_tree_case(args, torch.device("cpu"), loaded)

    normalization = case.description["target_normalization"]
    assert normalization["mean"] == [2.0]
    assert normalization["scale"] == [1.0]
    predicted = torch.zeros_like(case.batch.targets)
    expected = torch.nn.functional.mse_loss(predicted, (case.batch.targets - 2.0) / 1.0)
    assert torch.equal(case.objective(predicted), expected)


def test_tree_inputs_load_full_verified_cache_once_and_construct_both_arms(monkeypatch, tmp_path):
    from chartgat import seeds
    from research.tree_augmentation import paper

    axes = SimpleNamespace(chart=13)
    dataset = object()
    seen = {}
    monkeypatch.setattr(seeds, "resolve_seed_axes", lambda seed: axes)
    monkeypatch.setattr(
        paper,
        "_load_settings",
        lambda: ({"batch_size": 16}, tmp_path / "config.yaml"),
    )

    def prepare(suite, root, *, seed_axes, allow_download):
        seen["prepare"] = (suite, root, seed_axes, allow_download)
        return dataset

    def training_views(current, *, settings, chart_seed):
        seen["training_views"] = (current, settings, chart_seed)
        return ["fixed"], ["multi"]

    monkeypatch.setattr(paper, "_prepare_dataset", prepare)
    monkeypatch.setattr(paper, "_training_views", training_views)
    args = argparse.Namespace(dataset="zinc", tree_data_root=tmp_path, seed=13)
    loaded = speed._load_tree_inputs(args)

    assert seen["prepare"] == ("zinc", tmp_path, axes, False)
    assert seen["training_views"] == (dataset, {"batch_size": 16}, 13)
    assert loaded["views"] == {"fixed_bfs": ["fixed"], "multi_chart": ["multi"]}


def test_shell_wrapper_uses_conda_and_has_no_install_or_download_step():
    content = (speed.ROOT / "scripts/benchmark_speed.sh").read_text(encoding="utf-8")
    assert 'source "${project_root}/scripts/conda_env.sh"' in content
    assert "--help|-h) inspection_only=1" in content
    assert 'if [[ "${inspection_only}" == "0" ]]' in content
    assert 'scripts/check_dependencies.py" --quiet' in content
    assert "setup_gpu.sh" not in content and "pip install" not in content
    assert '"${environment_python}" scripts/benchmark_speed.py "$@"' in content
    assert "set -" not in content and "exec " not in content and "exit " not in content
    assert "main()" in content and "must be executed, not sourced" in content


@pytest.mark.parametrize(
    "track",
    [
        "conductance_gat",
        "conductance_v5",
        "cycle_pe_v1",
        "cycle_pe_v2",
        "tree_augmentation",
    ],
)
def test_report_source_hashes_include_execution_and_active_model(track):
    hashes = speed._implementation_hashes(track)
    assert {
        "scripts/benchmark_speed.py",
        "scripts/benchmark_speed.sh",
        "src/chartgat/execution.py",
        "src/chartgat/observability.py",
    } <= hashes.keys()
    model_file = {
        "conductance_gat": "research/conductance_gat/sparse.py",
        "conductance_v5": "research/conductance_gat/v5/model.py",
        "cycle_pe_v1": "research/cycle_pe/benchmark_models.py",
        "cycle_pe_v2": "research/cycle_pe/v2/model.py",
        "tree_augmentation": "research/tree_augmentation/paper_model.py",
    }[track]
    assert model_file in hashes
    for name, digest in hashes.items():
        assert digest == hashlib.sha256((speed.ROOT / name).read_bytes()).hexdigest()
