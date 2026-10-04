"""Complete contracts, pinned data and validation locks for context classification."""
from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace

import pytest
import torch

from test_local_prediction_data import completed_debug_source
from research.local_context_coupling.normalization import common, data, study
from research.local_context_coupling.normalization.common import parse_condition, condition_metadata
from research.local_energy_relations import data as audit_data
from research.local_energy_relations.prediction import common as original_common


@pytest.fixture(autouse=True)
def fixture_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tuning_rows(config):
    return [dict(dataset=d, condition=c, **condition_metadata(c), lr=lr, seed=s, phase="tuning",
                 best_epoch=1, validation_ce=1., validation_accuracy=.4,
                 training_epochs=config["training"]["epochs_per_run"],
                 optimizer_updates=config["training"]["epochs_per_run"])
            for d, c, lr, s in itertools.product(config["data"]["datasets"],
                config["conditions"], config["training"]["learning_rate_candidates"],
                config["training"]["tuning_seeds"])]


def final_rows(config):
    result = []
    for d, c, s, split in itertools.product(config["data"]["datasets"],
            config["conditions"], config["training"]["final_seeds"],
            ("train", "validation", "test")):
        shape = config["data"]["expected_shapes"][d]
        result.append(dict(dataset=d, condition=c, **condition_metadata(c), seed=s, split=split, ce=1., accuracy=.4,
            num_nodes=shape["nodes"], num_labeled_nodes=shape[split]))
    return result


def test_full_contract_keeps_all_data_and_840_complete_runs():
    config = common.read_config(profile="full")
    assert config["backbone"]["layers"] == 2
    assert config["backbone"]["hidden_dim"] == 64
    assert config["training"]["epochs_per_run"] == 500
    assert tuple(config["training"][k] for k in
        ("tuning_runs", "final_runs", "total_runs", "total_updates")) == (540, 300, 840, 420000)
    assert config["source"]["graphs"] == 201
    assert config["source"]["synthetic_graphs_used_for_training"] == 0
    assert config["data"]["sampling_ratio"] == 1
    assert config["data"]["expected_shapes"] == original_common.read_config()["data"]["expected_shapes"]
    assert config["conditions"] == list(common.CONDITIONS)
    assert not config["operators"]["learned_C"]


@pytest.mark.parametrize("section,key,value", [
    ("backbone", "layers", 1), ("backbone", "hidden_dim", 32),
    ("data", "sampling_ratio", .5), ("training", "epochs_per_run", 2),
    ("training", "final_seeds", [11]), ("operators", "learned_C", True),
    ("runtime", "activation_checkpointing", False),
])
def test_rejects_reduced_or_changed_scientific_contract(section, key, value):
    config = common.read_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="changed"):
        common.validate_config(config)


def test_exact_allocation_override_preserves_input_reader_contract():
    config = common.read_config()
    config["runtime"].update(cpu_threads=4, cpu_workers=2,
        relation_chunk_candidates=[128, 2048], gpu_memory_safety_fraction=.75)
    common.validate_config(config)
    reader = data.compatible_input_config(config)
    assert all(reader[k] == config[k] for k in ("profile", "data_source", "data", "source"))
    assert reader["experiment"] != config["experiment"]
    assert reader["runtime"]["relation_chunk_candidates"] == [128, 2048]


def test_scientific_import_closure_contains_new_model_and_reused_inputs():
    manifest = common.source_manifest()
    for name in ("common", "data", "operators", "model", "training", "evaluation", "report", "study"):
        assert f"research/local_context_coupling/normalization/{name}.py" in manifest["sha256"]
    assert "research/local_context_coupling/operators.py" in manifest["sha256"]
    assert "research/local_energy_relations/prediction/data.py" in manifest["sha256"]
    assert manifest["code_digest"] == common.digest(manifest["sha256"])


def test_debug_input_roundtrip_preserves_all_original_nodes_features_and_copies(
        tmp_path, completed_debug_source):
    config = common.read_config(profile="debug")
    graphs, manifest = data.prepare_datasets(config, tmp_path/"raw", tmp_path/"new",
        completed_debug_source, workers=1, download=False)
    assert len(graphs) == 3 and manifest["source_audit"]["graphs"] == 21
    assert manifest["prior_synthetic_graphs_preserved"] == 18
    assert manifest["actual_citation_data"] is False
    for name, graph in graphs.items():
        prior = audit_data.load_case(completed_debug_source/"inputs"/f"{name}.npz")
        assert torch.equal(graph.x.double(), prior.features)
        assert torch.equal(graph.edges, prior.edges)
        assert graph.metadata["debug"] and graph.num_features == 12
        assert graph.geometry_for("unit", "graph", "edge").num_cross_edges == graph.geometry_for("local_degree", "local", "graph").num_cross_edges
        path = tmp_path/f"{name}.npz"
        record = data.save_graph(graph, path)
        restored = data.load_graph(path, record["sha256"])
        assert data.graph_content_digest(graph) == data.graph_content_digest(restored)
        assert torch.equal(graph.y, restored.y)
        assert torch.equal(graph.test_mask, restored.test_mask)
        moved = graph.to("cpu", torch.float64)
        assert moved.x.dtype == torch.float64
        assert moved.geometry_for("unit", "graph", "edge").num_copies == graph.topology.num_local_nodes
        assert moved.geometry_for("local_degree", "local", "graph").num_copies == graph.topology.num_local_nodes
    data.assert_data_unchanged(manifest)
    assert common.read_json(tmp_path/"new/normalization_data_contract.json")["legacy_model_not_used"]


def test_lr_ties_use_smaller_rate_and_full_coverage_is_required():
    config = common.read_config(profile="debug")
    rows = tuning_rows(config)
    selections = study.select_learning_rates(config, rows)
    assert len(selections) == 60
    assert all(r["selected_lr"] == min(config["training"]["learning_rate_candidates"]) for r in selections)
    assert len(study.build_jobs(config, "tuning")) == 120
    assert len(study.build_jobs(config, "final", selections)) == 60
    with pytest.raises(ValueError, match="incomplete"):
        study.select_learning_rates(config, rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        study.select_learning_rates(config, rows + [copy.deepcopy(rows[0])])


@pytest.mark.parametrize("field,value", [
    ("test_accuracy", .9), ("training_epochs", 1), ("optimizer_updates", 2),
    ("variant", "off"), ("validation_ce", float("nan")),
])
def test_selection_rejects_test_leak_incomplete_training_or_invalid_rows(field, value):
    config = common.read_config(profile="debug")
    rows = tuning_rows(config)
    target = next(r for r in rows if r["variant"] == "learned")
    target[field] = value
    with pytest.raises(ValueError):
        study.select_learning_rates(config, rows)


def test_complete_final_training_is_required_before_test_unlock():
    config = common.read_config(profile="debug")
    jobs = study.build_jobs(config, "final", study.select_learning_rates(config, tuning_rows(config)))
    rows = [dict(dataset=j["dataset"], condition=j["condition"], **condition_metadata(j["condition"]),
        seed=s, lr=j["lr"], phase="final", training_epochs=3, optimizer_updates=3)
        for j in jobs for s in j["seeds"]]
    study._validate_final_training(config, rows, jobs)
    with pytest.raises(ValueError, match="checkpoints"):
        study._validate_final_training(config, rows[:-1], jobs)
    rows[0]["optimizer_updates"] = 2
    with pytest.raises(ValueError, match="coverage"):
        study._validate_final_training(config, rows, jobs)


@pytest.mark.parametrize("field,value", [("num_nodes", 1), ("num_labeled_nodes", 1),
    ("accuracy", 2), ("ce", float("nan"))])
def test_final_metrics_cannot_hide_incomplete_inputs(field, value):
    config = common.read_config(profile="debug")
    rows = final_rows(config)
    rows[0][field] = value
    with pytest.raises(ValueError):
        study.verify_coverage(config, tuning_rows(config), rows)


def test_frozen_coverage_includes_both_layers_and_all_six_interventions():
    config = common.read_config(profile="debug")
    evaluated = dict(intervention_rows=[], branch_rows=[])
    for d, c, s in itertools.product(config["data"]["datasets"], config["conditions"],
            config["training"]["final_seeds"]):
        base = dict(dataset=d, condition=c, seed=s, **condition_metadata(c))
        for layer in range(2):
            evaluated["branch_rows"].append(dict(base, layer=layer, intervention="original", target="none"))
        for i, target in itertools.product(study.intervention_variants(base["variant"]),
                study.intervention_scopes(c)):
            for split in ("train", "validation", "test"):
                evaluated["intervention_rows"].append(dict(base, split=split, intervention=i, target=target))
            for layer in range(2):
                evaluated["branch_rows"].append(dict(base, layer=layer, intervention=i, target=target))
    assert study.verify_frozen_coverage(config, evaluated) == dict(intervention_rows=1728, branch_rows=1392)
    evaluated["branch_rows"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        study.verify_frozen_coverage(config, evaluated)


def test_full_cannot_run_on_local_cpu_or_without_server_allocation(tmp_path):
    args = SimpleNamespace(config=None, profile="full", workers="auto",
        output_dir=str(tmp_path/"must-not-exist"), device="cpu")
    with pytest.raises(ValueError, match="Linux server CUDA"):
        study.run(args)
    assert not (tmp_path/"must-not-exist").exists()


def test_existing_six_are_included_without_duplicating_off_controls():
    from research.local_context_coupling.classification.common import CONDITIONS as old
    mapped = {f"{mode}__graph__{'none' if variant == 'off' else 'graph'}__{variant}"
              for mode, variant in (condition.split("__") for condition in old)}
    assert mapped.issubset(common.CONDITIONS)
    assert len(common.CONDITIONS) == len(set(common.CONDITIONS)) == 20
    assert sum(parse_condition(c)[3] == "off" for c in common.CONDITIONS) == 4
    with pytest.raises(ValueError, match="unknown"):
        parse_condition("unit__local__edge__off")


@pytest.mark.parametrize("field,value", [("intra_policy", "wrong"),
    ("cross_policy", "edge"), ("weight_mode", "wrong"),
    ("energy_operator", "raw_A_K"), ("cross_diagnostic_policy", "none")])
def test_selection_checks_all_normalization_metadata(field, value):
    config = common.read_config(profile="debug")
    rows = tuning_rows(config)
    rows[0][field] = value
    with pytest.raises(ValueError, match=field):
        study.select_learning_rates(config, rows)


def test_complete_normalization_data_roundtrip_detects_coefficient_mutation(tmp_path, completed_debug_source):
    config = common.read_config(profile="debug")
    graphs, manifest = data.prepare_datasets(config, tmp_path/"raw", tmp_path/"new",
        completed_debug_source, workers=1, download=False)
    graph = next(iter(graphs.values()))
    original = data.graph_content_digest(graph)
    geometry = graph.geometry_for("unit", "local", "edge")
    geometry.s_weights[0] *= .9
    assert data.graph_content_digest(graph) != original
    (tmp_path/"new/normalization_data_contract.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="normalization input contract"):
        data.assert_data_unchanged(manifest)
