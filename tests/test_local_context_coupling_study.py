"""Explicit DEBUG checks of fixed coupling scope, actions and complete reporting."""

from __future__ import annotations

import copy
import csv
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.local_context_coupling import study
from research.local_context_coupling.contract import read_config, source_manifest, source_reader_config, validate_config
from research.local_context_coupling.operators import prepare_geometry
from research.local_context_coupling.report import validate_rows, write_report
from research.local_context_coupling.verify import _dense
from research.local_energy_relations.data import AuditCase
from research.local_energy_relations.receiver_aggregation.contract import validate_config as validate_reader
from research.local_energy_relations.topology import build_topology


@pytest.fixture(autouse=True)
def single_debug_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_case(name="DEBUG-arithmetic", channels=7, vector=False, zero=False):
    edges = torch.tensor([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)], dtype=torch.long).T.contiguous()
    x = torch.sin(torch.arange(6 * channels, dtype=torch.float64)).reshape(6, channels)
    x[:, 0] = 0
    if zero:
        x.zero_()
    return AuditCase(name, "citation" if vector else "DEBUG-arithmetic", edges, x, False,
                     "vector_trace" if vector else "independent_scalar_columns", {"debug": True})


def geometry(case):
    top = build_topology(case.num_nodes, case.edges)
    return {mode: prepare_geometry(top, mode) for mode in ("unit", "local_degree")}


def dense_states(case):
    b = torch.zeros((case.edges.shape[1], case.num_nodes), dtype=torch.float64)
    b[torch.arange(b.shape[0]), case.edges[0]] = -1
    b[torch.arange(b.shape[0]), case.edges[1]] = 1
    l = b.T @ b
    p = torch.eye(case.num_nodes, dtype=torch.float64) - 0.5 / l.diag().max() * l
    return [case.x, p @ case.x, p @ p @ case.x]


@pytest.mark.parametrize("profile,graphs,features,raw", [("full", 201, 8804, 52824), ("debug", 21, 108, 648)])
def test_canonical_scope_and_legacy_reader_adapter(profile, graphs, features, raw):
    config = read_config(profile=profile)
    assert config["source"]["graphs"] == graphs
    assert config["data"]["physical_feature_columns"] == features
    assert 2 * 3 * features == raw
    assert config["training"] == {"trainable_parameters": 0, "optimizer_updates": 0, "classifier_training_run": False, "epochs": "N/A"}
    reader = source_reader_config(config)
    validate_reader(reader)
    assert reader["schema"] != config["schema"]
    assert reader["source"] == config["source"]
    assert reader["weights"] == ["unit", "local_degree"] and reader["states"] == ["H0", "H1", "H2"]


@pytest.mark.parametrize("section,key,value", [
    ("source", "graphs", 200), ("data", "physical_feature_columns", 1),
    ("data", "sampling_ratio", 0.5), ("data", "all_induced_one_hop_locals", False),
    ("operators", "persistent_steps", [1]), ("operators", "fixed_coefficients", False),
    ("training", "optimizer_updates", 1), ("measurement", "requires_nonzero_on_all_inputs", True),
    ("runtime", "graph_batch_candidates", [1]), ("runtime", "calibration_repeats", 1),
])
def test_full_scientific_scope_cannot_be_silently_changed(section, key, value):
    config = read_config(profile="full")
    config[section][key] = value
    with pytest.raises(ValueError, match="changed"):
        validate_config(config)


@pytest.mark.parametrize("field,bad", [("cpu_workers", 0), ("cpu_threads", True), ("channel_chunk", 1.5), ("physical_graph_batch", "small"), ("gpu_memory_safety_fraction", float("nan")), ("gpu_memory_safety_fraction", 1)])
def test_allocation_validation(field, bad):
    config = read_config(profile="debug")
    config["runtime"][field] = bad
    with pytest.raises(ValueError):
        validate_config(config)


def test_runtime_allocations_propagate_to_unchanged_snapshot_reader():
    config = read_config(profile="debug")
    config["runtime"].update(cpu_workers=2, cpu_threads=2, channel_chunk=5, physical_graph_batch=4, gpu_memory_safety_fraction=0.6)
    validate_config(config)
    reader = source_reader_config(config)
    assert all(reader["runtime"][name] == config["runtime"][name] for name in ("cpu_workers", "cpu_threads", "channel_chunk", "physical_graph_batch", "gpu_memory_safety_fraction"))


def test_scientific_import_closure_includes_all_new_modules_and_reused_classifier():
    manifest = source_manifest()
    for name in ("study", "contract", "report", "operators", "model", "verify"):
        assert f"research/local_context_coupling/{name}.py" in manifest["sha256"]
    assert "research/wedge_propagation/classification/model.py" in manifest["sha256"]
    assert "research/local_energy_relations/receiver_aggregation/data.py" in manifest["sha256"]
    assert "research/local_energy_relations/receiver_aggregation/contract.py" in manifest["sha256"]


@pytest.mark.parametrize("mode", ["unit", "local_degree"])
def test_all_reference_states_and_every_channel_match_independent_dense_actions(mode):
    case = make_case()
    operators = geometry(case)
    result = study.compute(operators, case.x, read_config(profile="debug"))
    r, m, a, k = _dense(case.num_nodes, case.edges.T.tolist(), mode, torch.device("cpu"))
    eta, gamma = operators[mode].eta[0], operators[mode].gamma[0]
    eye = torch.eye(a.shape[0], dtype=torch.float64)
    p = eye - eta * a
    theta = 0.5 / (a.diag() + k.diag()).max()
    for name, h in zip(("H0", "H1", "H2"), dense_states(case), strict=True):
        z, first = r @ h, p @ r @ h
        middle = (eye - gamma * k) @ first
        last = p @ middle
        on, off = m @ last, m @ p @ first
        direct = -gamma * eta.square() * m @ a @ k @ a @ r @ h
        expected = {
            "input_norm_sq": h.square().sum(0), "replicated_norm_sq": z.square().sum(0),
            "after_intra_1_norm_sq": first.square().sum(0), "after_cross_norm_sq": middle.square().sum(0),
            "after_intra_2_norm_sq": last.square().sum(0),
            "context_cross_action_norm_sq": (k @ first).square().sum(0),
            "sandwich_output_norm_sq": on.square().sum(0), "sandwich_off_output_norm_sq": off.square().sum(0),
            "sandwich_increment_norm_sq": (on-off).square().sum(0),
            "predicted_increment_norm_sq": direct.square().sum(0),
            "intra_energy_initial": 0.5 * (z * (a @ z)).sum(0),
            "cross_energy_after_intra_1": 0.5 * (first * (k @ first)).sum(0),
        }
        for step in (1, 2, 3):
            pu = torch.linalg.matrix_power(eye-theta*(a+k), step)
            po = torch.linalg.matrix_power(eye-theta*a, step)
            expected[f"persistent_{step}_increment_norm_sq"] = (m @ (pu-po) @ r @ h).square().sum(0)
        for field, value in expected.items():
            np.testing.assert_allclose(result[mode, name][field][0], value.numpy(), atol=1e-12, rtol=1e-11)
        assert result[mode, name]["sandwich_increment_norm_sq"][0, 1:].max() > 0


def test_null_inputs_are_valid_and_zero_denominator_stays_undefined():
    case = make_case(zero=True)
    config = read_config(profile="debug")
    operators = geometry(case)
    result = study.compute(operators, case.x, config)
    for (mode, state), fields in result.items():
        assert all(np.array_equal(values, np.zeros_like(values)) for values in fields.values())
        row = study.summarize(study._metadata(case, operators[mode], mode, state), {k: v.sum() for k, v in fields.items()}, config)
        assert row["sandwich_relative_increment"] is None
        assert row["sandwich_relative_formula_error"] is None
        assert row["context_numerically_resolved"] is False
        assert row["observed_increment_numerically_resolved"] is False


def test_disjoint_graph_batch_has_the_same_whole_graph_actions():
    cases = [make_case("DEBUG-a"), make_case("DEBUG-b")]
    tops = [build_topology(case.num_nodes, case.edges) for case in cases]
    config = read_config(profile="debug")
    combined = study.compute(study._batch_geometry(tops, config["weights"]), torch.cat([case.x for case in cases]), config)
    for graph, case in enumerate(cases):
        individual = study.compute(geometry(case), case.x, config)
        for key, fields in individual.items():
            for field, values in fields.items():
                np.testing.assert_allclose(combined[key][field][graph], values[0], atol=1e-12, rtol=1e-11)


def execute_case(tmp_path, case, chunk):
    config = read_config(profile="debug")
    config["source"]["synthetic_graphs"] = 0  # Explicit arithmetic test, not a accepted production config.
    config["runtime"]["channel_chunk"] = chunk
    top = build_topology(case.num_nodes, case.edges)
    table = study.Table(tmp_path / f"DEBUG-channel-{chunk}.csv")
    seen = set()
    try:
        rows = study._execute([0], [case], [top], [geometry(case)], config, torch.device("cpu"), [], table, threading.RLock(), seen)
    finally:
        table.close()
    assert table.count == case.num_features * 6 == len(seen)
    return rows


def test_chunking_covers_all_channels_and_reduces_vectors_after_sum_of_squares(tmp_path):
    case = make_case(vector=True)
    full = execute_case(tmp_path, case, case.num_features)
    chunked = execute_case(tmp_path, case, 3)
    for left, right in zip(full, chunked, strict=True):
        for key, value in left.items():
            if isinstance(value, float):
                np.testing.assert_allclose(value, right[key], atol=1e-12, rtol=1e-11)
            else:
                assert value == right[key]
        assert left["sandwich_relative_increment"] == pytest.approx(np.sqrt(left["sandwich_increment_norm_sq"] / left["sandwich_off_output_norm_sq"]))


def test_report_rejects_incomplete_duplicates_and_mean_of_channel_ratios(tmp_path):
    config = read_config(profile="debug")
    config["source"]["graphs"] = 1
    rows = execute_case(tmp_path, make_case(vector=True), 3)
    validate_rows(config, rows)
    with pytest.raises(ValueError, match="all unique"):
        validate_rows(config, rows[:-1])
    with pytest.raises(ValueError, match="all unique"):
        validate_rows(config, rows + [rows[0]])
    wrong = copy.deepcopy(rows)
    wrong[0]["sandwich_relative_increment"] *= 1.2
    with pytest.raises(ValueError, match="sum-of-squares"):
        validate_rows(config, wrong)


def test_report_accepts_all_zero_input_without_fabricated_relative_error(tmp_path):
    config = read_config(profile="debug")
    config["source"]["graphs"] = 1
    case = make_case(vector=True, zero=True)
    rows = execute_case(tmp_path, case, 3)
    write_report(tmp_path, config, rows, [], {"graphs": 1, "actual_citation_graphs": 0})
    checks = json.loads((tmp_path / "mechanism_checks.json").read_text())
    assert checks["sandwich_formula_relative_error_max"] is None
    assert checks["observed_increment_resolved_graph_state_cells"] == 0
    assert (tmp_path / "LOCAL_CONTEXT_SUMMARY.md").is_file()


def test_identity_failure_and_nonfinite_values_are_explicit():
    config = read_config(profile="debug")
    case = make_case()
    fields = study.compute(geometry(case), case.x, config)["unit", "H0"]
    broken = copy.deepcopy(fields)
    broken["immediate_increment_norm_sq"][:, 1] += 1
    with pytest.raises(ValueError, match="immediate_increment"):
        study._guards(broken, config)
    broken = copy.deepcopy(fields)
    broken["cross_energy_after_intra_1"][:, 1] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        study._guards(broken, config)


def test_safe_persistent_step_does_not_clamp_small_positive_weighted_degree():
    geo = geometry(make_case())["unit"]
    altered = replace(geo, intra_degree=torch.full_like(geo.intra_degree, 0.1), cross_degree=torch.full_like(geo.cross_degree, 0.2))
    torch.testing.assert_close(study._step(altered), torch.tensor([0.5 / 0.3], dtype=torch.float64))


def test_full_requires_server_cuda_and_debug_never_silently_falls_back(monkeypatch):
    with pytest.raises(ValueError, match="Linux server CUDA"):
        study.validate_device(read_config(profile="full"), torch.device("cpu"))
    monkeypatch.setattr(study.sys, "platform", "linux")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="explicit allocated"):
        study.validate_device(read_config(profile="full"), torch.device("cuda"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        study.validate_device(read_config(profile="full"), torch.device("cuda"))


def test_dimension_override_is_explicit_and_cpu_preparation_uses_spawn(monkeypatch):
    case = make_case()
    top = build_topology(case.num_nodes, case.edges)
    config = read_config(profile="debug")
    config["runtime"]["channel_chunk"] = case.num_features + 1
    with pytest.raises(ValueError, match="exceeds"):
        study.calibrate([case], [top], [geometry(case)], config, torch.device("cpu"), [], False)
    contexts = []
    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 2
            contexts.append(mp_context.get_start_method())
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def map(self, function, arguments):
            return map(function, arguments)
    monkeypatch.setattr(study, "ProcessPoolExecutor", Pool)
    before = torch.get_num_threads()
    result = study._parallel([(top, "unit"), (top, "local_degree")], 2)
    assert contexts == ["spawn"] and torch.get_num_threads() == before
    assert all(geo.intra_weights.device.type == "cpu" for geo in result)


def test_complete_debug_runner_adapter_and_artifacts(tmp_path, monkeypatch):
    cases = [make_case(f"DEBUG-s-{i:02d}", channels=4) for i in range(18)]
    cases += [make_case(f"DEBUG-c-{i}", channels=12, vector=True) for i in range(3)]
    tops = [build_topology(case.num_nodes, case.edges) for case in cases]
    config = read_config(profile="debug")
    config["runtime"].update(cpu_workers=2, cpu_threads=1, physical_graph_batch=4, channel_chunk=5)
    config_path = tmp_path / "DEBUG_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output, source = tmp_path / "DEBUG_results", tmp_path / "immutable_DEBUG_source"
    output.mkdir(); source.mkdir()
    marker = source / "DEBUG_original.txt"
    marker.write_text("unchanged arithmetic fixture", encoding="utf-8")
    checks = []
    def reader(legacy, source_dir, output_dir, workers):
        validate_reader(legacy)
        assert legacy["runtime"]["cpu_workers"] == workers == 1
        assert legacy["source"] == config["source"]
        return cases, tops, {"source_content_digest": "DEBUG-fixture-reference", "cpu_input_calibration": []}
    def assert_preserved(source_dir, manifest):
        assert marker.read_text() == "unchanged arithmetic fixture"
        checks.append(manifest["source_content_digest"])
    from research.local_context_coupling import verify
    monkeypatch.setattr(study, "prepare_cases", reader)
    monkeypatch.setattr(study, "assert_inputs_unchanged", assert_preserved)
    monkeypatch.setattr(verify, "run_checks", lambda device: {"status": "passed", "scope": "DEBUG test fixture stub; actual CE checks covered separately"})
    args = SimpleNamespace(config=config_path, profile="debug", device="cpu", source_dir=source, output_dir=output, workers=1)
    study.run(args, output)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["debug"] is True
    assert completion["optimizer_updates"] == 0 and completion["classifier_training_run"] is False
    assert completion["coverage"]["channel_metric_rows"] == 648
    assert completion["coverage"]["graph_summary_rows"] == 126
    assert completion["source_and_inputs_preserved"] is True
    assert len(checks) == 2
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["runtime"]["cpu_workers"] == 1
    for name in ("source_manifest.json", "source_adapter.json", "coverage.json", "hardware.json", "preflight_debug_checks.json", "resources.json", "mechanism_checks.json", "LOCAL_CONTEXT_SUMMARY.md", "figures/context_coupling_checks.png", "figures/context_coupling_checks.pdf"):
        assert (output / name).is_file()
    with (output / "channel_metrics.csv").open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    assert len(raw) == 648
    assert {row["graph_id"] for row in raw} == {case.graph_id for case in cases}
    with (output / "graph_summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 126
    assert all(row["dependency_measurement"].endswith("not_full_Jacobian") for row in summaries)
