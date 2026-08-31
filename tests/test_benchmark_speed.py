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
    assert "--track" in result.stdout


@pytest.mark.parametrize(
    ("track", "dataset", "batch_size"),
    [("conductance_gat", "cora", 2), ("cycle_pe_v2", "zinc12k", 32)],
)
def test_defaults_resolve_to_official_data_and_no_compile(track, dataset, batch_size):
    args = speed.build_parser().parse_args(["--track", track])
    speed._validate(args)
    assert args.dataset == dataset and args.batch_size == batch_size
    assert args.steps == 20 and args.warmup == 5 and not args.include_compile
    assert args.data_root == speed.ROOT / "data/paper"


@pytest.mark.parametrize(
    "extra",
    [
        ["--device", "cpu"],
        ["--dataset", "toy"],
        ["--steps", "0"],
        ["--warmup", "0"],
        ["--batch-size", "0"],
        ["--seed", "-1"],
    ],
)
def test_invalid_request_fails_before_artifacts(tmp_path, extra):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        speed.main(["--track", "conductance_gat", "--output-dir", str(output), *extra])
    assert caught.value.code == 2
    assert not output.exists()


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
    graph.y[1:] = 0
    assert torch.equal(case.objective(prediction), expected)
    reference = case.make_model("reference")
    optimized = case.make_model("optimized")
    optimized.load_state_dict(reference.state_dict())
    comparison = speed._compare_probes(speed._probe(reference, case), speed._probe(optimized, case))
    assert comparison["passed"]


def test_cycle_builder_selects_train_only_and_no_download(monkeypatch, tmp_path):
    from research.cycle_pe.v2 import data

    seen = {}
    train_graphs = [object(), object(), object()]
    batch = SimpleNamespace(
        x=torch.zeros(4, 1, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        cycle_bases=(torch.ones(2, 1),),
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
    args = argparse.Namespace(dataset="zinc12k", data_root=tmp_path, batch_size=1)
    case = speed._build_cycle_case(args, torch.device("cpu"))
    assert seen["allow_download"] is False and seen["selected"] == train_graphs[:1]
    assert case.description["basis_pairs"] == 2
    assert case.make_model("reference").basis_execution == "reference"
    assert case.make_model("optimized").basis_execution == "batched"


def test_shell_wrapper_uses_conda_and_has_no_install_or_download_step():
    content = (speed.ROOT / "scripts/benchmark_speed.sh").read_text(encoding="utf-8")
    assert 'source "${project_root}/scripts/conda_env.sh"' in content
    assert "--help|-h) inspection_only=1" in content
    assert 'if [[ "${inspection_only}" == "0" ]]' in content
    assert 'scripts/check_dependencies.py" --quiet' in content
    assert "setup_gpu.sh" not in content and "pip install" not in content
    assert 'exec "${environment_python}" scripts/benchmark_speed.py "$@"' in content


@pytest.mark.parametrize("track", ["conductance_gat", "cycle_pe_v2"])
def test_report_source_hashes_include_execution_and_active_model(track):
    hashes = speed._implementation_hashes(track)
    assert {
        "scripts/benchmark_speed.py",
        "scripts/benchmark_speed.sh",
        "src/chartgat/execution.py",
    } <= hashes.keys()
    model_file = (
        "research/conductance_gat/sparse.py"
        if track == "conductance_gat"
        else "research/cycle_pe/v2/model.py"
    )
    assert model_file in hashes
    for name, digest in hashes.items():
        assert digest == hashlib.sha256((speed.ROOT / name).read_bytes()).hexdigest()
