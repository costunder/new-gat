"""Complete DEBUG runner, physical batching and exact feature-trace integration."""

from __future__ import annotations

import csv
import json
from argparse import Namespace
from collections import defaultdict

import numpy as np
import pytest
import torch

from research.local_energy_relations import study
from research.local_energy_relations.contract import read_config, validate_config
from research.local_energy_relations.data import make_synthetic_case
from research.local_energy_relations.topology import (
    CORRESPONDENCE_KINDS,
    batch_topologies,
    build_topology,
)
from research.wedge_propagation.data import make_specs


class MemoryTable:
    def __init__(self):
        self.rows = []

    def write(self, rows):
        self.rows.extend(rows)


@pytest.fixture(autouse=True)
def single_thread_debug_math():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_cases():
    return [make_synthetic_case(spec) for spec in make_specs(profile="debug")]


def tables_and_summaries():
    tables = {name: MemoryTable() for name in ("local", "relation", "transfer", "temporal")}
    summaries = {name: [] for name in ("local", "relation", "transfer", "assembly")}
    return tables, summaries


def test_full_cpu_device_rejected_before_any_data_preparation(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("FULL CPU must reject before preparing inputs")

    monkeypatch.setattr(study, "prepare_cases", unexpected)
    output = tmp_path / "never-created"
    args = Namespace(
        config=None, profile="full", device="cpu", data_root=tmp_path, no_download=True
    )
    with pytest.raises(ValueError, match="server CUDA"):
        study.run(args, output)
    assert not output.exists()


def test_cuda_unavailable_is_explicit_and_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        study.validate_device(read_config(profile="debug"), torch.device("cuda"))


def test_full_profile_and_runtime_batch_guards():
    config = read_config()
    assert config["runtime"]["physical_graph_batch"] == "auto"
    config["runtime"]["physical_graph_batch"] = 198
    assert validate_config(config)["synthetic"]["graphs"] == 198
    config["runtime"]["physical_graph_batch"] = True
    with pytest.raises(ValueError, match="positive integer"):
        validate_config(config)


def test_reference_states_use_each_physical_graph_degree_and_preserve_isolates():
    cases = synthetic_cases()[:3]
    tops = [build_topology(case.num_nodes, case.edges.numpy()) for case in cases]
    batched = batch_topologies(tops).to("cpu")
    x = torch.cat([case.features for case in cases])
    actual = study.reference_states(batched, x)
    assert torch.equal(actual[0], x)
    for state in range(3):
        expected = torch.cat(
            [
                study.reference_states(top.to("cpu"), case.features)[state]
                for case, top in zip(cases, tops, strict=True)
            ]
        )
        torch.testing.assert_close(actual[state], expected, rtol=1e-12, atol=1e-12)
    empty = build_topology(3, np.empty((2, 0), dtype=np.int64)).to("cpu")
    signal = torch.tensor([[1.0, -1.0], [2.0, 0.0], [4.0, 2.0]], dtype=torch.float64)
    assert all(torch.equal(state, signal) for state in study.reference_states(empty, signal))


def test_collect_complete_batched_scalar_records_retains_original_ids_and_counts():
    cases = synthetic_cases()[:3]
    top = batch_topologies([build_topology(case.num_nodes, case.edges.numpy()) for case in cases])
    config = read_config(profile="debug")
    computed = study.compute_chunk(
        top.to("cpu"), torch.cat([case.features for case in cases]), config
    )
    tables, summaries = tables_and_summaries()
    study.collect(cases, top, computed, tables, summaries)
    assert len(tables["local"].rows) == sum(
        case.num_nodes * case.num_features * 6 for case in cases
    )
    assert len(tables["transfer"].rows) == sum(
        2 * case.edges.shape[1] * case.num_features * 6 for case in cases
    )
    assert len(tables["relation"].rows) == sum(
        2 * case.edges.shape[1] * case.num_features * 30 for case in cases
    )
    assert len(tables["temporal"].rows) == sum(
        case.num_nodes * case.num_features * 4 for case in cases
    )
    for case in cases:
        rows = [row for row in tables["local"].rows if row["graph_id"] == case.graph_id]
        assert {row["center"] for row in rows} == set(range(case.num_nodes))
        assert {row["realization"] for row in rows} == set(range(case.num_features))
        transfers = [row for row in tables["transfer"].rows if row["graph_id"] == case.graph_id]
        expected = set(map(tuple, case.edges.t().tolist()))
        expected |= {(v, u) for u, v in expected}
        assert {(row["sender"], row["receiver"]) for row in transfers} == expected
    assert len(summaries["local"]) == len(cases) * 6
    assert len(summaries["relation"]) == len(cases) * 34
    assert len(summaries["transfer"]) == len(cases) * 6
    assert len(summaries["assembly"]) == len(cases) * 6


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_feature_channel_chunks_equal_one_vector_trace(chunk_size):
    case = synthetic_cases()[-1]
    top = build_topology(case.num_nodes, case.edges.numpy()).to("cpu")
    config = read_config(profile="debug")
    full = study._merge_trace({}, study.compute_chunk(top, case.features, config))
    chunked = {}
    for first in range(0, case.num_features, chunk_size):
        study._merge_trace(
            chunked, study.compute_chunk(top, case.features[:, first : first + chunk_size], config)
        )
    assert full.keys() == chunked.keys()
    for weight in full:
        for kind in ("local", "transfer", "assembly"):
            for expected, actual in zip(full[weight][kind], chunked[weight][kind], strict=True):
                for key in expected:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=1e-8, atol=1e-12)
        for kind in ("relations", "temporal"):
            for stages, expected in full[weight][kind].items():
                actual = chunked[weight][kind][stages]
                if kind == "temporal":
                    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
                else:
                    for key in expected:
                        np.testing.assert_allclose(
                            actual[key], expected[key], rtol=1e-8, atol=1e-12
                        )


def test_topology_npz_preserves_numeric_correspondence_offsets_without_pickle(tmp_path):
    cases = synthetic_cases()[:2]
    config = read_config(profile="debug")
    config["runtime"]["cpu_workers"] = 1
    config["runtime"]["cpu_threads"] = 1
    hardware = study.runtime_resources(torch.device("cpu"))
    tops, records = study.prepare_topologies(cases, config, hardware, tmp_path, [])
    for top, record in zip(tops, records, strict=True):
        with np.load(tmp_path / record["file"], allow_pickle=False) as saved:
            for kind, offsets in zip(
                CORRESPONDENCE_KINDS, top.correspondence_offsets, strict=True
            ):
                assert saved[f"{kind}_offsets"].dtype == np.int64
                assert saved[f"{kind}_offsets"].shape == (top.num_pairs + 1,)
                np.testing.assert_array_equal(saved[f"{kind}_offsets"], offsets)
            np.testing.assert_array_equal(saved["pair_centers"], top.pair_centers)
            np.testing.assert_array_equal(saved["local_node_global"], top.local_node_global)


def test_complete_debug_run_records_all_case_coverage_and_preserves_inputs(tmp_path):
    config = read_config(profile="debug")
    config["runtime"].update(
        cpu_workers=1, cpu_threads=1, physical_graph_batch=18, channel_chunk=3, relation_batch=4
    )
    config_path = tmp_path / "debug-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "DEBUG-run"
    output.mkdir()
    args = Namespace(
        config=config_path,
        profile="debug",
        device="cpu",
        data_root=tmp_path / "fixture-cache",
        no_download=True,
    )
    study.run(args, output)
    complete = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert complete["profile"] == "debug" and complete["debug"] is True
    assert complete["graphs"] == 21 and complete["synthetic_graphs"] == 18
    assert complete["actual_citation_graphs"] == 0 and complete["citation_graphs"] == 3
    assert complete["synthetic_scalar_inputs"] == 72
    assert complete["optimizer_updates"] == 0
    assert complete["classifier_training_run"] is False
    assert complete["source_and_inputs_preserved"] is True
    with (output / "graph_summary.csv").open(encoding="utf-8", newline="") as stream:
        graphs = list(csv.DictReader(stream))
    assert len(graphs) == 21
    expected = {case.graph_id for case in synthetic_cases()} | {
        "DEBUG-Cora",
        "DEBUG-CiteSeer",
        "DEBUG-PubMed",
    }
    assert {row["graph_id"] for row in graphs} == expected
    expected_raw = defaultdict(int)
    for row in graphs:
        n, e = int(row["num_nodes"]), int(row["num_edges"])
        units = (
            int(row["num_features"]) if row["feature_mode"] == "independent_scalar_columns" else 1
        )
        expected_raw["local"] += n * units * 6
        expected_raw["transfer"] += 2 * e * units * 6
        expected_raw["relation"] += 2 * e * units * 30
        expected_raw["temporal"] += n * units * 4
    assert complete["raw_rows"] == dict(expected_raw)
    with (output / "local_states.csv").open(encoding="utf-8", newline="") as stream:
        vector_rows = [row for row in csv.DictReader(stream) if row["family"] == "citation"]
    assert len(vector_rows) == (24 + 30 + 36) * 6
    assert {row["realization"] for row in vector_rows} == {"vector_trace"}
    # Repeating a run cannot overwrite the completed contract or input snapshots.
    preserved = (output / "completion.json").read_bytes()
    with pytest.raises(FileExistsError):
        study.run(args, output)
    assert (output / "completion.json").read_bytes() == preserved
