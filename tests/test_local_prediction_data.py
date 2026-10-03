"""Complete DEBUG data closure, public provenance and numeric prediction caches."""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from research.local_energy_relations import contract as audit_contract
from research.local_energy_relations import data as audit_data
from research.local_energy_relations.prediction import common, data
from research.local_energy_relations.topology import CORRESPONDENCE_KINDS, build_topology
from research.wedge_propagation.classification import data as public_data


def _save_topology(topology, path, name):
    arrays = {
        field.name: getattr(topology, field.name)
        for field in fields(topology)
        if field.name != "correspondence_offsets"
    }
    arrays.update(
        {
            kind + "_offsets": np.asarray(offsets, dtype=np.int64)
            for kind, offsets in zip(
                CORRESPONDENCE_KINDS, topology.correspondence_offsets, strict=True
            )
        }
    )
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "graph_id": name,
        "file": "topology/" + path.name,
        "sha256": common.file_sha256(path),
        "nodes": topology.n,
        "physical_edges": topology.num_edges,
        "local_node_occurrences": topology.num_local_nodes,
        "local_edge_occurrences": topology.num_local_edges,
        "directed_center_pairs": topology.num_pairs,
    }


@pytest.fixture(scope="module")
def completed_debug_source(tmp_path_factory):
    """Explicit 21-graph DEBUG fixture; no fabricated full-training evidence."""
    folder = tmp_path_factory.mktemp("prediction-debug-source")
    source = folder / "source"
    config = audit_contract.read_config(profile="debug")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cases, _ = audit_data.prepare_cases(
            config, folder / "debug-data", source, workers=1, download=False
        )
        (source / "topology").mkdir()
        records, center_units, pair_units = [], 0, 0
        for case in cases:
            top = build_topology(case.num_nodes, case.edges)
            records.append(
                _save_topology(top, source / "topology" / (case.graph_id + ".npz"), case.graph_id)
            )
            count = case.num_features if case.feature_mode == "independent_scalar_columns" else 1
            center_units += top.n * count
            pair_units += top.num_pairs * count
        common.write_json(source / "config.json", config)
        provenance = audit_contract.source_manifest()
        common.write_json(source / "source_manifest.json", provenance)
        common.write_json(
            source / "topology_manifest.json",
            {"schema": "induced-one-hop-local-correspondence-v1", "graphs": records},
        )
        common.write_json(
            source / "completion.json",
            {
                "status": "complete",
                "profile": "debug",
                "debug": True,
                "graphs": 21,
                "synthetic_graphs": 18,
                "synthetic_scalar_inputs": 72,
                "citation_graphs": 3,
                "actual_citation_graphs": 0,
                "sampling_ratio": 1.0,
                "all_nodes_edges_features": True,
                "source_and_inputs_preserved": True,
                "weights": config["weights"],
                "states": config["states"],
                "trainable_parameters": 0,
                "optimizer_updates": 0,
                "classifier_training_run": False,
                "source_digest": provenance["code_digest"],
                "raw_rows": {
                    "local": 6 * center_units,
                    "transfer": 6 * pair_units,
                    "relation": 30 * pair_units,
                    "temporal": 4 * center_units,
                },
            },
        )
        yield source
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def prepared_debug(tmp_path_factory, completed_debug_source):
    output = tmp_path_factory.mktemp("prediction-debug-classification")
    config = common.read_config(profile="debug")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        graphs, manifest = data.prepare_datasets(
            config, output / "debug-data", output, completed_debug_source, workers=1, download=False
        )
        yield graphs, manifest, output
    finally:
        torch.set_num_threads(threads)


def test_full_contract_preserves_scale_and_336_run_budget():
    full = common.read_config(profile="full")
    assert full["data"]["expected_shapes"] == public_data.EXPECTED
    assert full["backbone"]["layers"] == 2
    assert full["backbone"]["hidden_dim"] == 64
    assert full["training"]["epochs_per_run"] == 500
    assert (
        full["training"]["tuning_runs"],
        full["training"]["final_runs"],
        full["training"]["total_runs"],
        full["training"]["total_updates"],
    ) == (216, 120, 336, 168000)
    assert len(full["conditions"]) == 8
    assert full["source"]["graphs"] == 201
    assert full["source"]["synthetic_graphs_used_for_training"] == 0
    assert not full["operators"]["learned_C"]
    assert not full["operators"]["global_inverse_in_forward"]


def test_debug_contract_is_explicit_and_separate():
    debug = common.read_config(profile="debug")
    assert debug["data_source"] == "debug_fixture"
    assert debug["source"]["graphs"] == 21
    assert debug["source"]["actual_citation_graphs"] == 0
    assert debug["backbone"]["hidden_dim"] == 8
    assert debug["training"]["epochs_per_run"] == 3
    assert (debug["training"]["total_runs"], debug["training"]["total_updates"]) == (144, 432)
    with pytest.raises(ValueError, match="configuration|changed"):
        common.validate_config(debug, "full")


@pytest.mark.parametrize(
    "sector,key,value",
    [
        ("backbone", "hidden_dim", 32),
        ("backbone", "layers", 1),
        ("data", "sampling_ratio", 0.5),
        ("training", "epochs_per_run", 2),
        ("training", "final_seeds", [11]),
        ("operators", "learned_C", True),
    ],
)
def test_rejects_science_reduction_and_scope_change(sector, key, value):
    config = copy.deepcopy(common.read_config(profile="full"))
    config[sector][key] = value
    with pytest.raises(ValueError, match="changed"):
        common.validate_config(config)


def test_runtime_exact_chunk_allocation_can_change_without_science_change():
    config = common.read_config(profile="full")
    config["runtime"].update(
        cpu_threads=4,
        cpu_workers=2,
        relation_chunk_candidates=[64, 512],
        gpu_memory_safety_fraction=0.75,
    )
    assert common.validate_config(config) is config


@pytest.mark.parametrize(
    "key,value",
    [
        ("cpu_threads", 0),
        ("cpu_workers", True),
        ("relation_chunk_candidates", [1, 1]),
        ("relation_chunk_candidates", [0]),
        ("gpu_memory_safety_fraction", 1),
        ("gpu_memory_safety_fraction", float("nan")),
    ],
)
def test_invalid_runtime_allocations_are_explicit_errors(key, value):
    config = common.read_config(profile="debug")
    config["runtime"][key] = value
    with pytest.raises(ValueError):
        common.validate_config(config)


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"profile":"full","profile":"debug"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        common.read_json(duplicate)
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"metric":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        common.read_json(nonfinite)


def test_complete_debug_classifier_uses_only_three_citation_inputs(prepared_debug):
    graphs, manifest, _ = prepared_debug
    assert list(graphs) == ["DEBUG-Cora", "DEBUG-CiteSeer", "DEBUG-PubMed"]
    assert manifest["classification_graphs"] == 3
    assert manifest["source_audit"]["graphs"] == 21
    assert manifest["prior_synthetic_graphs_preserved"] == 18
    assert manifest["synthetic_graphs_used_for_training"] == 0
    assert not manifest["actual_citation_data"]
    assert manifest["raw"] == {}
    assert all(g.metadata["debug"] for g in graphs.values())
    data.assert_data_unchanged(manifest)


def test_classification_features_edges_match_prior_audit_exactly(prepared_debug):
    graphs, manifest, _ = prepared_debug
    for name, graph in graphs.items():
        prior = audit_data.load_case(manifest["source_dir"] + "/inputs/" + name + ".npz")
        assert torch.equal(graph.x.double(), prior.features)
        assert torch.equal(graph.edges, prior.edges)
        assert graph.topology.n == graph.num_nodes
        assert graph.topology.num_pairs == 2 * graph.edges.shape[1]
        assert graph.num_features == 12
        assert not hasattr(graph, "paths")
        assert not hasattr(graph, "qdiag")


def test_numeric_snapshot_roundtrip_reuses_cached_geometry(prepared_debug, tmp_path, monkeypatch):
    graph = prepared_debug[0]["DEBUG-Cora"]
    snapshot = tmp_path / "graph.npz"
    record = data.save_graph(graph, snapshot)
    monkeypatch.setattr(
        data, "prepare_geometry", lambda *_: pytest.fail("load must not rebuild geometry")
    )
    restored = data.load_graph(snapshot, record["sha256"])
    assert data.graph_content_digest(restored) == record["content_digest"]
    assert record["cached_topology_and_geometry"]
    assert torch.equal(restored.x, graph.x)
    assert torch.equal(restored.y, graph.y)
    assert torch.equal(restored.train_mask, graph.train_mask)
    assert torch.equal(restored.topology.local_edge_nodes, graph.topology.local_edge_nodes)
    for mode in ("unit", "local_degree"):
        assert torch.equal(restored.geometry.transition[mode], graph.geometry.transition[mode])
    with np.load(snapshot, allow_pickle=False) as saved:
        assert all(saved[key].dtype != object for key in saved.files)


def test_existing_snapshot_is_preserved_and_corruption_rejected(prepared_debug, tmp_path):
    graph = prepared_debug[0]["DEBUG-CiteSeer"]
    snapshot = tmp_path / "graph.npz"
    record = data.save_graph(graph, snapshot)
    before = snapshot.read_bytes()
    with pytest.raises(FileExistsError):
        data.save_graph(graph, snapshot)
    assert snapshot.read_bytes() == before
    snapshot.write_bytes(before + b"changed-debug-copy")
    with pytest.raises(ValueError, match="checksum"):
        data.load_graph(snapshot, record["sha256"])


def test_missing_source_and_output_inside_prior_results_are_rejected(
    tmp_path, completed_debug_source
):
    config = common.read_config(profile="debug")
    with pytest.raises(ValueError, match="missing"):
        data.prepare_datasets(
            config,
            tmp_path / "data",
            tmp_path / "missing-source-output",
            tmp_path / "missing-source",
            workers=1,
            download=False,
        )
    with pytest.raises(ValueError, match="separate"):
        data.prepare_datasets(
            config,
            tmp_path / "data",
            completed_debug_source / "output",
            completed_debug_source,
            workers=1,
            download=False,
        )


def test_raw_feature_mismatch_is_rejected_before_geometry(prepared_debug, monkeypatch):
    graph = prepared_debug[0]["DEBUG-Cora"]
    loader_config = common.read_json(
        Path(public_data.__file__).with_name("config_debug.json")
    )
    arrays = public_data._debug_arrays(loader_config)[graph.name]
    changed = list(arrays)
    changed[0] = np.array(arrays[0], copy=True)
    changed[0][0, 0] += 1
    prior = _audit_case_for_test(graph)
    monkeypatch.setattr(
        data, "prepare_geometry", lambda *_: pytest.fail("bad raw inputs must fail before geometry")
    )
    with pytest.raises(ValueError, match="differ"):
        data.graph_from_arrays(graph.name, changed, graph.topology, source_case=prior)


def _audit_case_for_test(graph):
    return audit_data.AuditCase(
        graph.name, "citation", graph.edges, graph.x.double(), False, "vector_trace", {}
    )


def test_graph_to_preserves_masks_and_topology_with_precision_conversion(prepared_debug):
    graph = prepared_debug[0]["DEBUG-PubMed"].to("cpu", torch.float64)
    assert graph.x.dtype == torch.float64
    assert graph.y.dtype == torch.long
    assert graph.train_mask.dtype == torch.bool
    assert graph.topology.local_edge_nodes.dtype == torch.long
    assert graph.geometry.weights["unit"].dtype == torch.float64


def test_source_manifest_covers_prediction_and_actual_receiver_geometry_helper():
    source = common.source_manifest()
    assert source["code_digest"] == common.digest(source["sha256"])
    assert "research/local_energy_relations/prediction/data.py" in source["sha256"]
    assert "research/local_energy_relations/receiver_aggregation/operators.py" in source["sha256"]
    assert "research/wedge_propagation/classification/data.py" in source["sha256"]
    assert "research/wedge_propagation/classification/model.py" in source["sha256"]
    assert "research/local_energy_relations/prediction/config_full.json" in source["sha256"]


def test_public_raw_checksum_rejection_preserves_file(tmp_path):
    raw = tmp_path / "Cora" / "raw"
    raw.mkdir(parents=True)
    path = raw / "ind.cora.x"
    path.write_bytes(b"explicit corrupted unit test raw fixture")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        public_data.ensure_raw("Cora", tmp_path, download=False)
    assert path.read_bytes() == before
