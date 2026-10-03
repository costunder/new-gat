"""Complete DEBUG source reuse and strict immutable receiver experiment contracts."""

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pytest
import torch

from research.local_energy_relations import contract as original_contract
from research.local_energy_relations import data as original_data
from research.local_energy_relations.receiver_aggregation import contract, data
from research.local_energy_relations.topology import CORRESPONDENCE_KINDS, build_topology


def save_topology(top, path, graph_id):
    payload = {
        field.name: getattr(top, field.name)
        for field in fields(top)
        if field.name != "correspondence_offsets"
    }
    payload.update(
        {
            kind + "_offsets": np.asarray(offsets, dtype=np.int64)
            for kind, offsets in zip(CORRESPONDENCE_KINDS, top.correspondence_offsets, strict=True)
        }
    )
    np.savez_compressed(path, **payload)
    return {
        "graph_id": graph_id,
        "file": "topology/" + path.name,
        "sha256": contract.file_sha256(path),
        "nodes": top.n,
        "physical_edges": top.num_edges,
        "local_node_occurrences": top.num_local_nodes,
        "local_edge_occurrences": top.num_local_edges,
        "directed_center_pairs": top.num_pairs,
    }


@pytest.fixture
def completed_debug_source(tmp_path):
    """Create all 21 explicitly DEBUG inputs; never synthetic final performance evidence."""
    root = tmp_path / "source"
    config = original_contract.read_config(profile="debug")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cases, _ = original_data.prepare_cases(
            config, tmp_path / "debug-data", root, workers=1, download=False
        )
        (root / "topology").mkdir()
        records, center_units, pair_units = [], 0, 0
        for case in cases:
            top = build_topology(case.num_nodes, case.edges)
            records.append(
                save_topology(top, root / "topology" / (case.graph_id + ".npz"), case.graph_id)
            )
            features = case.num_features if case.feature_mode == "independent_scalar_columns" else 1
            center_units += top.n * features
            pair_units += top.num_pairs * features
        original_contract.write_json(
            root / "topology_manifest.json",
            {"schema": "induced-one-hop-local-correspondence-v1", "graphs": records},
        )
        original_contract.write_json(root / "config.json", config)
        source = original_contract.source_manifest()
        original_contract.write_json(root / "source_manifest.json", source)
        original_contract.write_json(
            root / "completion.json",
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
                "source_digest": source["code_digest"],
                "raw_rows": {
                    "local": 6 * center_units,
                    "transfer": 6 * pair_units,
                    "relation": 30 * pair_units,
                    "temporal": 4 * center_units,
                },
            },
        )
        yield root
    finally:
        torch.set_num_threads(threads)


def test_complete_source_reuse_original_channels_and_topology_no_regeneration(
    completed_debug_source, tmp_path, monkeypatch
):
    original_bytes = {
        p.relative_to(completed_debug_source): p.read_bytes()
        for p in completed_debug_source.rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("source preparation must not regenerate graphs")

    monkeypatch.setattr(original_data, "make_synthetic_case", forbidden)
    config = contract.read_config(profile="debug")
    output = tmp_path / "new"
    cases, topologies, manifest = data.prepare_cases(
        config, completed_debug_source, output, workers=2
    )
    assert len(cases) == len(topologies) == 21
    assert (
        sum(c.num_features for c in cases if c.feature_mode == "independent_scalar_columns") == 72
    )
    assert [c.num_features for c in cases[-3:]] == [12, 12, 12]
    assert manifest["graphs"] == 21 and manifest["no_regeneration_or_download"] is True
    assert set(p.name for p in output.iterdir()) == {"source_input_manifest.json"}
    for case, top in zip(cases, topologies, strict=True):
        assert np.array_equal(top.edges, case.edges.numpy())
    assert original_bytes == {
        p.relative_to(completed_debug_source): p.read_bytes()
        for p in completed_debug_source.rglob("*")
        if p.is_file()
    }
    data.assert_inputs_unchanged(completed_debug_source, manifest)
    with pytest.raises(FileExistsError, match="preserved"):
        data.prepare_cases(config, completed_debug_source, output, workers=2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("weights", ["unit"]),
        ("states", ["H0"]),
        ("conditions", ["sum"]),
        ("relations", ["shared_edges"]),
        ("precision", "float32"),
        ("profile", "debug"),
    ],
)
def test_scientific_scope_is_immutable(key, value):
    config = contract.read_config()
    config[key] = value
    with pytest.raises(ValueError, match="changed"):
        contract.validate_config(config, "full")


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("source", "graphs", 21),
        ("source", "synthetic_scalar_inputs", 16),
        ("data", "all_nodes_edges_features", 1),
        ("solver", "tolerance", 1e-3),
        ("reference", "reference_steps", 1),
    ],
)
def test_nested_data_model_and_numerical_contracts_are_immutable(section, key, value):
    config = contract.read_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="changed"):
        contract.validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("cpu_workers", False),
        ("channel_chunk", 0),
        ("relation_batch", 0.5),
        ("gpu_memory_safety_fraction", True),
        ("gpu_memory_safety_fraction", 1),
    ],
)
def test_only_valid_resource_allocations_can_change(key, value):
    config = contract.read_config()
    config["runtime"][key] = value
    with pytest.raises(ValueError):
        contract.validate_config(config)


def test_resources_do_not_change_full_data_or_condition_contract():
    config = contract.read_config()
    config["runtime"].update(
        cpu_workers=2,
        cpu_threads=4,
        channel_chunk=32,
        relation_batch=1024,
        physical_graph_batch=64,
        gpu_memory_safety_fraction=0.8,
    )
    assert contract.validate_config(config)["source"]["graphs"] == 201
    assert config["conditions"] == ["tagged", "sum", "sum_within", "sum_between", "sum_both"]


def test_duplicate_unknown_or_mixed_profile_config_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"profile":"full","profile":"debug"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        contract.read_config(path)
    config = contract.read_config()
    config["max_nodes"] = 4
    with pytest.raises(ValueError, match="unknown"):
        contract.validate_config(config)
    with pytest.raises(ValueError, match="changed"):
        contract.validate_config(contract.read_config(profile="debug"), "full")


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("completion.json", "status", "failed"),
        ("completion.json", "all_nodes_edges_features", False),
        ("completion.json", "source_digest", "0" * 64),
        ("input_manifest.json", "graphs", 1),
    ],
)
def test_incomplete_or_tampered_source_metadata_fails(
    completed_debug_source, tmp_path, name, key, value
):
    path = completed_debug_source / name
    record = json.loads(path.read_text(encoding="utf-8"))
    record[key] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        data.prepare_cases(
            contract.read_config(profile="debug"),
            completed_debug_source,
            tmp_path / "new",
            workers=1,
        )


def test_source_missing_graph_channel_snapshot_cannot_be_recreated(
    completed_debug_source, tmp_path
):
    record = contract.read_json(completed_debug_source / "input_manifest.json")["inputs"][0]
    input_path = completed_debug_source / "inputs" / record["file"]
    input_path.rename(input_path.with_suffix(".preserved"))
    with pytest.raises(ValueError, match="source file missing"):
        data.prepare_cases(
            contract.read_config(profile="debug"),
            completed_debug_source,
            tmp_path / "new",
            workers=1,
        )


def test_source_references_cannot_escape_or_write_inside_old_results(completed_debug_source):
    with pytest.raises(ValueError, match="separate"):
        data.prepare_cases(
            contract.read_config(profile="debug"),
            completed_debug_source,
            completed_debug_source / "new",
            workers=1,
        )
    with pytest.raises(ValueError, match="leaves"):
        data._safe_file(completed_debug_source, "../elsewhere.json")


def test_source_changes_detected_after_loading(completed_debug_source, tmp_path):
    _, _, manifest = data.prepare_cases(
        contract.read_config(profile="debug"), completed_debug_source, tmp_path / "new", workers=1
    )
    path = completed_debug_source / "completion.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        data.assert_inputs_unchanged(completed_debug_source, manifest)


@pytest.mark.parametrize(
    "edges,n",
    [
        ([[0, 1], [1, 2]], 4),
        ([[0, 0, 1], [1, 2, 2]], 3),
        ([[], []], 3),
    ],
)
def test_topology_roundtrip_all_correspondence_kinds_isolates_and_empty(tmp_path, edges, n):
    case = original_data.AuditCase(
        "fixture",
        "fixture",
        torch.tensor(edges, dtype=torch.long),
        torch.zeros(n, 2, dtype=torch.float64),
        False,
        "independent_scalar_columns",
        {},
    )
    top = build_topology(n, case.edges)
    path = tmp_path / "fixture.npz"
    record = save_topology(top, path, case.graph_id)
    loaded = data.load_topology(path, case, record)
    for field in fields(top):
        assert np.array_equal(getattr(loaded, field.name), getattr(top, field.name))


def test_wrong_shared_node_identity_rejected_even_if_file_hash_updated(tmp_path):
    case = original_data.AuditCase(
        "fixture",
        "fixture",
        torch.tensor([[0, 1], [1, 2]]),
        torch.ones(3, 2, dtype=torch.float64),
        False,
        "independent_scalar_columns",
        {},
    )
    top = build_topology(3, case.edges)
    top.shared_node_right[0] += 1
    path = tmp_path / "fixture.npz"
    record = save_topology(top, path, case.graph_id)
    with pytest.raises(ValueError, match="identity"):
        data.load_topology(path, case, record)


def test_implementation_manifest_covers_nested_new_package_and_parent_dependency_closure():
    manifest = contract.source_manifest()
    prefix = "research/local_energy_relations/receiver_aggregation/"
    assert prefix + "data.py" in manifest["sha256"]
    assert prefix + "config_full.json" in manifest["sha256"]
    assert "research/local_energy_relations/operators.py" in manifest["sha256"]
    assert "research/wedge_propagation/classification/data.py" in manifest["sha256"]
    assert all("experiments/" not in path for path in manifest["sha256"])
    contract.assert_source_unchanged(manifest)
