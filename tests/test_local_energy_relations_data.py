"""DEBUG fixture and independent input/provenance contracts; no server study."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from research.local_energy_relations import data
from research.local_energy_relations.contract import (
    assert_source_unchanged,
    read_config,
    source_manifest,
    validate_config,
)
from research.wedge_propagation.data import make_case as original_case
from research.wedge_propagation.data import make_specs


def fixture_case():
    return data.AuditCase(
        "fixture-isolates",
        "fixture",
        torch.tensor([[0], [1]]),
        torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=torch.float64),
        False,
        "independent_scalar_columns",
        {"actual_citation_data": False},
    )


def test_full_uses_original_198_graph_3168_scalar_contract_and_three_whole_citations():
    config = read_config()
    specs = make_specs(config["synthetic"]["master_seed"], "full")
    assert len(specs) == 198
    assert sum(spec.num_features for spec in specs) == 3168
    assert sorted({spec.num_nodes for spec in specs}) == [20, 30, 40, 60, 80, 100]
    assert config["data"]["total_graphs"] == 201
    assert config["citation"]["datasets"] == ["Cora", "CiteSeer", "PubMed"]
    assert [
        config["citation"]["expected_shapes"][name]["features"]
        for name in config["citation"]["datasets"]
    ] == [1433, 3703, 500]
    assert config["training"] == {
        "trainable_parameters": 0,
        "optimizer_updates": 0,
        "epochs": "N/A",
    }
    assert config["citation"]["encoder"] is None
    assert config["citation"]["use_labels_or_masks"] is False


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_wrapper_retains_source_topology_features_and_named_seeds(profile):
    for spec in make_specs(profile=profile)[:: max(1, len(make_specs(profile=profile)) // 6)]:
        expected = original_case(spec)
        case = data.make_synthetic_case(spec)
        assert case.graph_id == expected.graph_id
        assert torch.equal(case.edges, expected.edges)
        assert torch.equal(case.features, expected.features)
        assert case.metadata["graph_seed"] == expected.graph_seed
        assert case.metadata["feature_seed"] == expected.feature_seed
        assert not case.actual_data
        assert case.feature_mode == "independent_scalar_columns"


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights", ["unit"]),
        ("relations", ["shared_edges"]),
        ("states", ["H0"]),
        ("precision", "float32"),
        ("training", {"trainable_parameters": 1, "optimizer_updates": 0, "epochs": "N/A"}),
    ],
)
def test_scientific_scope_cannot_be_reduced(field, value):
    config = read_config()
    config[field] = value
    with pytest.raises(ValueError, match="changed"):
        validate_config(config)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("synthetic", "graphs", 18),
        ("synthetic", "scalar_realizations", 1),
        ("citation", "datasets", ["Cora"]),
        ("data", "sampling_ratio", 0.5),
        ("data", "all_nodes", 1),
        ("solver", "tolerance", 1e-3),
        ("runtime", "reference_steps", 1),
    ],
)
def test_nested_budget_contract_and_json_types_are_locked(section, key, value):
    config = read_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="changed"):
        validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("cpu_workers", False),
        ("cpu_threads", 0),
        ("physical_graph_batch", False),
        ("channel_chunk", 2.5),
        ("relation_batch", -1),
        ("gpu_memory_safety_fraction", 1),
        ("gpu_memory_safety_fraction", False),
    ],
)
def test_runtime_allocation_requires_valid_types(key, value):
    config = read_config()
    config["runtime"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_resource_chunk_changes_do_not_reduce_any_scientific_dimensions():
    config = read_config()
    config["runtime"].update(
        cpu_workers=2,
        cpu_threads=4,
        channel_chunk=32,
        relation_batch=128,
        gpu_memory_safety_fraction=0.8,
    )
    assert validate_config(config)["synthetic"]["graphs"] == 198


def test_duplicate_unknown_or_mixed_profile_configuration_fails(tmp_path):
    unknown = read_config()
    unknown["max_nodes"] = 5
    with pytest.raises(ValueError, match="unknown"):
        validate_config(unknown)
    with pytest.raises(ValueError, match="changed"):
        validate_config(read_config(profile="debug"), "full")
    filename = tmp_path / "duplicate.json"
    filename.write_text('{"profile":"full","profile":"debug"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_config(filename)


def test_exclusive_snapshot_roundtrip_contains_all_nodes_edges_features_without_labels(tmp_path):
    case = fixture_case()
    path = tmp_path / "input.npz"
    record = data.save_case(case, path)
    loaded = data.load_case(path, record["sha256"])
    assert loaded.graph_id == case.graph_id
    assert torch.equal(loaded.features, case.features)
    assert torch.equal(loaded.edges, case.edges)
    assert record["nodes"] == 3
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"edges", "features", "original_node_ids", "metadata"}
        assert np.array_equal(saved["original_node_ids"], np.arange(3))
    contents = path.read_bytes()
    with pytest.raises(FileExistsError):
        data.save_case(case, path)
    assert path.read_bytes() == contents


def test_content_hash_tracks_isolated_nodes_edges_and_every_feature():
    case = fixture_case()
    enlarged = data.AuditCase(
        case.graph_id,
        case.family,
        case.edges,
        torch.cat((case.features, torch.zeros(1, 2, dtype=torch.float64))),
        False,
        case.feature_mode,
        case.metadata,
    )
    changed = data.AuditCase(
        case.graph_id,
        case.family,
        case.edges,
        case.features.clone(),
        False,
        case.feature_mode,
        case.metadata,
    )
    changed.features[2, 1] += 1
    assert data.graph_content_hash(case) != data.graph_content_hash(enlarged)
    assert data.graph_content_hash(case) == data.graph_content_hash(changed)
    assert data.feature_content_hash(case) != data.feature_content_hash(changed)


def test_tampered_feature_or_node_mapping_is_detected_without_external_file_hash(tmp_path):
    path = tmp_path / "input.npz"
    data.save_case(fixture_case(), path)
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in original.files}
    arrays["features"][0, 0] += 1
    altered = tmp_path / "altered.npz"
    np.savez_compressed(altered, **arrays)
    with pytest.raises(ValueError, match="content checksum"):
        data.load_case(altered)
    arrays["original_node_ids"] = np.array([0, 2, 1])
    remapped = tmp_path / "remapped.npz"
    np.savez_compressed(remapped, **arrays)
    with pytest.raises(ValueError, match="correspondence"):
        data.load_case(remapped)


@pytest.mark.parametrize(
    "change", ["nan", "float32", "reverse_edge", "duplicate", "unsafe_id", "flag"]
)
def test_invalid_input_does_not_silently_canonicalize_or_replace(change):
    case = fixture_case()
    kwargs = dict(case.__dict__)
    if change == "nan":
        kwargs["features"] = case.features.clone()
        kwargs["features"][0, 0] = float("nan")
    elif change == "float32":
        kwargs["features"] = case.features.float()
    elif change == "reverse_edge":
        kwargs["edges"] = case.edges.flip(0)
    elif change == "duplicate":
        kwargs["edges"] = case.edges.repeat(1, 2)
    elif change == "unsafe_id":
        kwargs["graph_id"] = "../escape"
    elif change == "flag":
        kwargs["actual_data"] = True
    with pytest.raises(ValueError):
        data.validate_case(data.AuditCase(**kwargs))


def test_debug_prepares_all_21_graphs_and_labels_every_fixture_as_debug(tmp_path):
    config = read_config(profile="debug")
    cases, manifest = data.prepare_cases(
        config, tmp_path / "citation-cache", tmp_path / "run", workers=1, download=False
    )
    assert len(cases) == 21
    assert len(manifest["inputs"]) == 21
    assert manifest["synthetic_graphs"] == 18
    assert manifest["synthetic_scalar_inputs"] == 72
    assert manifest["actual_citation_graphs"] == 0
    citations = cases[-3:]
    assert [case.graph_id for case in citations] == config["citation"]["datasets"]
    assert [list(case.features.shape) for case in citations] == [[24, 12], [30, 12], [36, 12]]
    assert all(case.feature_mode == "vector_trace" and not case.actual_data for case in citations)
    assert all(case.metadata["labels_and_masks_used"] is False for case in citations)
    assert all(case.metadata["profile"] == "debug" for case in cases)
    data.assert_inputs_unchanged(tmp_path / "run", manifest)
    record = manifest["inputs"][-1]
    loaded = data.load_case(tmp_path / "run" / "inputs" / record["file"], record["sha256"])
    assert torch.equal(loaded.features, citations[-1].features)
    with pytest.raises(FileExistsError, match="preserved"):
        data.prepare_cases(
            config, tmp_path / "citation-cache", tmp_path / "run", workers=1, download=False
        )


def test_input_guard_detects_post_preparation_changes(tmp_path):
    _, manifest = data.prepare_cases(
        read_config(profile="debug"),
        tmp_path / "cache",
        tmp_path / "run",
        workers=1,
        download=False,
    )
    snapshot = tmp_path / "run" / "inputs" / manifest["inputs"][0]["file"]
    with snapshot.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="input snapshot changed"):
        data.assert_inputs_unchanged(tmp_path / "run", manifest)


def test_cpu_worker_calibration_preserves_input_order_and_values(monkeypatch):
    monkeypatch.setattr(data, "_available_cpus", lambda: 2)
    selected, trials = data.choose_workers(make_specs(profile="debug"))
    assert selected in (1, 2)
    assert [trial["workers"] for trial in trials] == [1, 2]
    assert all(trial["nodes_per_graph"] == 8 and trial["graphs"] == 9 for trial in trials)
    with pytest.raises(ValueError, match="affinity"):
        data.choose_workers(make_specs(profile="debug"), 3)


def test_source_manifest_has_actual_generic_closure_and_excludes_old_model_tracks():
    record = source_manifest()
    names = record["sha256"]
    assert "research/local_energy_relations/data.py" in names
    assert "research/wedge_propagation/classification/data.py" in names
    assert "research/wedge_propagation/classification/common.py" in names
    assert "research/wedge_propagation/study.py" in names
    assert "research/wedge_propagation/operators.py" in names
    assert not any(
        "/model.py" in name or "/node_normalization/" in name or "/branch_strength/" in name
        for name in names
    )
    assert_source_unchanged(record)
    altered = copy.deepcopy(record)
    altered["sha256"]["research/local_energy_relations/data.py"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        assert_source_unchanged(altered)
