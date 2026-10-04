"""Pinned full budget and original settings for the complete placement factorial."""

import copy

import pytest

from research.local_energy_relations.placement import common, data
from research.local_energy_relations.placement.model import active_layers, parse_condition
from research.local_energy_relations.prediction import common as original


@pytest.mark.parametrize("profile,budget", [("full", (540, 300, 840, 420000)), ("debug", (240, 120, 360, 1080))])
def test_complete_placement_conditions_and_budget(profile, budget):
    config = common.read_config(profile=profile)
    training = config["training"]
    assert tuple(training[k] for k in ("tuning_runs", "final_runs", "total_runs", "total_updates")) == budget
    assert len(config["conditions"]) == len(set(config["conditions"])) == 20
    assert config["conditions"] == list(common.CONDITIONS)
    for mode in ("unit", "local_degree"):
        assert sum(c == mode + "__base" for c in config["conditions"]) == 1
        for variant in common.VARIANTS[1:]:
            for placement in common.PLACEMENTS:
                name = f"{mode}__{variant}__{placement}"
                assert parse_condition(name) == (mode, variant, placement)
                assert list(active_layers(name)) == config["placement"]["injection_layers"][placement]
    assert training["epochs_per_run"] == (500 if profile == "full" else 3)


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_adapter_preserves_original_data_operators_backbone_and_per_run_training(profile):
    config = common.read_config(profile=profile)
    previous = original.read_config(profile=profile)
    compatible = data.compatible_prediction_config(config)
    assert compatible == previous
    for key in ("data", "source", "backbone", "operators", "resources"):
        assert config[key] == previous[key]
    for key, value in previous["training"].items():
        if key not in ("tuning_runs", "final_runs", "total_runs", "total_updates"):
            assert config["training"][key] == value


@pytest.mark.parametrize("section,key,value", [
    ("data", "sampling_ratio", 0.5), ("backbone", "layers", 1),
    ("backbone", "hidden_dim", 8), ("training", "epochs_per_run", 3),
    ("training", "final_seeds", [11]), ("training", "total_updates", 1000),
    ("operators", "learned_C", True), ("resources", "parallel_final_run_candidates", [1]),
    ("evaluation", "interventions_only_for_enabled_layers", False),
])
def test_full_cannot_silently_change_science_or_shrink(section, key, value):
    config = common.read_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="changed"):
        common.validate_config(config)


def test_incomplete_conditions_or_placement_contract_rejected():
    config = common.read_config(profile="debug")
    config["conditions"].pop()
    with pytest.raises(ValueError):
        common.validate_config(config)
    config = common.read_config(profile="debug")
    config["placement"]["injection_layers"]["hidden"] = [0, 1]
    with pytest.raises(ValueError):
        common.validate_config(config)


def test_runtime_allocation_changes_preserve_source_loader_contract():
    config = common.read_config()
    config["runtime"].update(cpu_threads=2, cpu_workers=4, gpu_memory_safety_fraction=0.8, relation_chunk_candidates=[128, 4096])
    common.validate_config(config)
    compatible = data.compatible_prediction_config(config)
    assert compatible["runtime"] == config["runtime"]
    assert compatible["training"]["epochs_per_run"] == 500
    assert len(compatible["conditions"]) == 8


@pytest.mark.parametrize("key,value", [
    ("cpu_threads", 0), ("cpu_workers", True), ("gpu_memory_safety_fraction", 1),
    ("gpu_memory_safety_fraction", float("nan")), ("relation_chunk_candidates", []),
    ("relation_chunk_candidates", [16, 16]), ("relation_chunk_candidates", [True]),
])
def test_invalid_allocations_fail_explicitly(key, value):
    config = common.read_config(profile="debug")
    config["runtime"][key] = value
    with pytest.raises(ValueError):
        common.validate_config(config)


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_expected_capacity_counts_only_active_layer_vectors(profile):
    config = common.read_config(profile=profile)
    hidden = config["backbone"]["hidden_dim"]
    for dataset, shape in config["data"]["expected_shapes"].items():
        classes = shape["classes"]
        entries = config["capacity"]["expected_parameters_per_seed"][dataset]
        base = shape["features"] * hidden + hidden * classes + 2
        assert entries["base"] == base
        for variant in common.VARIANTS[1:]:
            branches = 2 if variant == "both" else 1
            for placement, width in (("hidden", hidden), ("output", classes), ("all", hidden + classes)):
                assert entries[f"{variant}__{placement}"] == base + branches * width
    assert config["capacity"]["disabled_branch_parameters"] == 0


def test_source_manifest_has_actual_unchanged_operator_loader_and_new_contract_closure():
    manifest = common.source_manifest()
    assert common.digest(manifest["sha256"]) == manifest["code_digest"]
    for relative in (
        "research/local_energy_relations/placement/common.py",
        "research/local_energy_relations/placement/data.py",
        "research/local_energy_relations/placement/model.py",
        "research/local_energy_relations/placement/config_full.json",
        "research/local_energy_relations/placement/config_debug.json",
        "research/local_energy_relations/placement/design_contract.json",
        "research/local_energy_relations/prediction/data.py",
        "research/local_energy_relations/prediction/operators.py",
        "research/local_energy_relations/receiver_aggregation/operators.py",
        "research/wedge_propagation/classification/model.py",
        "research/wedge_propagation/classification/data.py",
    ):
        assert relative in manifest["sha256"]
    damaged = copy.deepcopy(manifest)
    damaged["code_digest"] = "changed"
    with pytest.raises(ValueError, match="source changed"):
        common.assert_source_unchanged(damaged)


def test_placement_strict_json_rejects_duplicate_and_nonfinite(tmp_path):
    for filename, text in (("duplicate.json", '{"profile":"full","profile":"debug"}'), ("nan.json", '{"value":NaN}')):
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            common.read_json(path)
