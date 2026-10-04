"""The placement adapter uses byte-identical prior classification inputs."""

import copy

import pytest
import torch

from research.local_energy_relations.placement import common, data
from research.local_energy_relations.prediction import data as original_data
from test_local_prediction_data import completed_debug_source, prepared_debug


@pytest.fixture(scope="module")
def placement_prepared_debug(tmp_path_factory, completed_debug_source):
    output = tmp_path_factory.mktemp("placement-debug-classification")
    config = common.read_config(profile="debug")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        graphs, manifest = data.prepare_datasets(
            config, output / "data", output, completed_debug_source, workers=1, download=False
        )
        yield graphs, manifest, output
    finally:
        torch.set_num_threads(previous_threads)


def test_reused_graph_objects_have_exact_previous_numeric_identity(placement_prepared_debug, prepared_debug):
    graphs, manifest, _ = placement_prepared_debug
    original_graphs, previous_manifest, _ = prepared_debug
    assert set(graphs) == set(original_graphs)
    for name in graphs:
        assert data.graph_content_digest(graphs[name]) == original_data.graph_content_digest(original_graphs[name])
        assert torch.equal(graphs[name].x, original_graphs[name].x)
        assert torch.equal(graphs[name].edges, original_graphs[name].edges)
        assert torch.equal(graphs[name].train_mask, original_graphs[name].train_mask)
        assert manifest["datasets"][name] == previous_manifest["datasets"][name]
    assert manifest["classification_graphs"] == 3
    assert manifest["source_audit"]["graphs"] == 21
    assert manifest["prior_synthetic_graphs_preserved"] == 18
    assert manifest["synthetic_graphs_used_for_training"] == 0
    assert manifest["sampling_ratio"] == 1.0
    assert not manifest["actual_citation_data"]
    data.assert_data_unchanged(manifest)


def test_exclusive_sidecar_records_full_placement_contract_identity(placement_prepared_debug):
    graphs, manifest, output = placement_prepared_debug
    proof = common.read_json(output / "placement_data_contract.json")
    assert proof["placement_config_digest"] == common.digest(common.read_config(profile="debug"))
    assert proof["per_run_training_unchanged"]
    assert proof["original_prediction_code_unchanged"]
    assert proof["original_data_manifest_digest"] == common.digest(common.read_json(output / "data_manifest.json"))
    assert proof["classification_graph_content_digest"] == {name: data.graph_content_digest(graph) for name, graph in graphs.items()}
    assert common.read_json(output / "placement_data_manifest.json") == manifest
    assert common.file_sha256(output / "placement_data_contract.json") == manifest["placement_contract"]["sha256"]


def test_repeated_preparation_preserves_existing_provenance(placement_prepared_debug):
    _, manifest, output = placement_prepared_debug
    before = (output / "placement_data_contract.json").read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        data.prepare_datasets(common.read_config(profile="debug"), output / "data", output, manifest["source_dir"], workers=1, download=False)
    assert (output / "placement_data_contract.json").read_bytes() == before


def test_provenance_corruption_and_missing_adapter_record_fail(placement_prepared_debug):
    _, manifest, _ = placement_prepared_debug
    changed = copy.deepcopy(manifest)
    changed["placement_contract"]["sha256"] = "incorrect"
    with pytest.raises(ValueError, match="checksum"):
        data.assert_data_unchanged(changed)
    missing = copy.deepcopy(manifest)
    del missing["placement_contract"]
    with pytest.raises(ValueError, match="missing"):
        data.assert_data_unchanged(missing)


def test_missing_source_and_nested_output_rejected(tmp_path, completed_debug_source):
    config = common.read_config(profile="debug")
    with pytest.raises(ValueError, match="missing"):
        data.prepare_datasets(config, tmp_path / "data", tmp_path / "output", tmp_path / "missing", workers=1, download=False)
    with pytest.raises(ValueError, match="separate"):
        data.prepare_datasets(config, tmp_path / "data", completed_debug_source / "output", completed_debug_source, workers=1, download=False)


def test_adapter_proof_independently_rejects_changed_backbone(monkeypatch):
    config = common.read_config(profile="debug")
    config["backbone"]["hidden_dim"] += 1
    monkeypatch.setattr(data, "validate_config", lambda value: value)
    with pytest.raises(ValueError, match="backbone"):
        data.compatible_prediction_config(config)
