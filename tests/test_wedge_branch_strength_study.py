"""Whole frozen-run contracts and DEBUG integration guards for Experiment 4.1."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.wedge_propagation.branch_strength import contract

_FOLDER = Path(__file__).resolve().parents[1] / "research/wedge_propagation/branch_strength"
_SINGLES = (
    "true_c_unit_denominator",
    "identity_hold_reference",
    "identity_unit_denominator",
    "identity_norm_matched",
    "branch_off",
)
_SHUFFLES = ("shuffle_hold_reference", "shuffle_norm_matched")
_SPLITS = ("train", "validation", "test")


def _config(profile="debug"):
    return json.loads((_FOLDER / f"config_{profile}.json").read_text(encoding="utf-8"))


def _write_config(tmp_path, config):
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _changes(config):
    return [
        {"treatment": treatment, "target": target, "manifest_index": index}
        for target in ("layer_0", "layer_1", "both")
        for treatment in (*_SINGLES, *_SHUFFLES)
        for index in (range(config["manifests_per_dataset"]) if treatment in _SHUFFLES else (-1,))
    ]


def _coverage_tables(config):
    baseline, treatments, diagnostics, fixed = [], [], [], []
    for dataset, condition, seed in itertools.product(
        config["data"]["datasets"], config["conditions"], config["final_seeds"]
    ):
        model = {"dataset": dataset, "condition": condition, "seed": seed}
        baseline += [{**model, "split": split, "ce": 0.8, "accuracy": 0.5} for split in _SPLITS]
        diagnostics += [
            {
                **model,
                "treatment": "baseline",
                "target": "none",
                "manifest_index": -1,
                "layer": layer,
            }
            for layer in range(2)
        ]
        if not condition.startswith("learned_wedge"):
            continue
        for variant in _changes(config):
            treatments += [
                {**model, **variant, "split": split, "ce": 0.8, "accuracy": 0.5}
                for split in _SPLITS
            ]
            diagnostics += [{**model, **variant, "layer": layer} for layer in range(2)]
        fixed_variants = [
            {"treatment": "baseline", "manifest_index": -1},
            *[{"treatment": treatment, "manifest_index": -1} for treatment in _SINGLES],
            *[
                {"treatment": treatment, "manifest_index": index}
                for treatment in _SHUFFLES
                for index in range(config["manifests_per_dataset"])
            ],
        ]
        fixed += [
            {**model, **variant, "target": f"layer_{layer}", "layer": layer}
            for variant in fixed_variants
            for layer in range(2)
        ]
    return baseline, treatments, diagnostics, fixed


@pytest.mark.parametrize(
    "profile,counts,models,learned,cases",
    [
        ("full", (360, 6750, 4740, 1560), 120, 30, 2370),
        ("debug", (144, 972, 744, 240), 48, 12, 372),
    ],
)
def test_canonical_profiles_keep_all_models_and_complete_frozen_counts(
    profile, counts, models, learned, cases
):
    config = contract.read_config(_FOLDER / f"config_{profile}.json", profile)
    declared = contract.expected_counts(config)
    assert (
        tuple(
            declared[name]
            for name in ("baseline_rows", "treatment_rows", "diagnostic_rows", "fixed_z_rows")
        )
        == counts
    )
    assert declared["source_final_models"] == models
    assert declared["learned_final_models"] == learned
    assert declared["end_to_end_seed_cases"] == cases
    assert declared["optimizer_updates"] == 0
    assert len(config["data"]["datasets"]) == 3 and len(config["conditions"]) == 8


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_treatments_include_every_scope_and_manifest_exactly_once(profile):
    config = _config(profile)
    expected = [
        {"treatment": "baseline", "target": "none", "manifest_index": -1},
        *_changes(config),
    ]
    actual = contract.variants(config)
    assert actual == expected
    assert len({(row["treatment"], row["target"], row["manifest_index"]) for row in actual}) == len(
        actual
    )
    assert contract.variants(config, include_baseline=False) == expected[1:]
    assert contract.fixed_variants(config) == [
        row for row in expected if row["target"] in ("none", "both")
    ]


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_coverage_validates_entire_graph_model_seed_layer_scope_and_split_contract(profile):
    config = _config(profile)
    tables = _coverage_tables(config)
    verified = contract.verify_coverage(config, *tables)
    assert verified == {**config["expected_counts"], "complete": True}


@pytest.mark.parametrize("table_index", range(4))
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected_seed", "nonfinite"])
def test_coverage_rejects_partial_duplicate_foreign_and_nonfinite_rows(table_index, mutation):
    config = _config()
    tables = list(_coverage_tables(config))
    rows = tables[table_index]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "unexpected_seed":
        rows[0]["seed"] = 999
    else:
        rows[0]["message_norm_ratio"] = float("inf")
    with pytest.raises(ValueError):
        contract.verify_coverage(config, *tables)


def test_undefined_ratios_are_nullable_in_complete_serialized_tables():
    config = _config()
    tables = _coverage_tables(config)
    tables[2][0].update(second_to_input=None, second_to_input_defined=False)
    assert contract.verify_coverage(config, *tables)["complete"] is True


@pytest.mark.parametrize(
    "key,value",
    [
        ("conditions", ["learned_wedge_rms"]),
        ("treatments", ["baseline"]),
        ("targets", ["both"]),
        ("final_seeds", [11]),
        ("manifests_per_dataset", 1),
        ("optimization_updates", 1),
        ("sampling_ratio", 0.5),
        ("precision", "float16"),
        ("expected_counts", {}),
    ],
)
def test_full_config_rejects_scope_shrinkage_and_optimizer_updates(tmp_path, key, value):
    config = _config("full")
    config[key] = value
    with pytest.raises(ValueError):
        contract.read_config(_write_config(tmp_path, config), "full")


def test_debug_profile_and_graph_names_cannot_be_claimed_as_full(tmp_path):
    with pytest.raises(ValueError):
        contract.read_config(_FOLDER / "config_debug.json", "full")
    config = _config("full")
    config["data"]["datasets"][0] = "DEBUG-Cora"
    with pytest.raises(ValueError):
        contract.read_config(_write_config(tmp_path, config), "full")


@pytest.mark.parametrize(
    "key,value",
    [
        ("tf32", True),
        ("schema", "unknown"),
        ("norm_match", "test_label_mask_only"),
        ("selection", "select_treatment_by_test_accuracy"),
        ("source_replay", {"ce_atol": 1.0, "ce_rtol": 1.0, "accuracy_atol": 1.0}),
        ("runtime", {"packed_candidates": [1]}),
    ],
)
def test_canonical_guard_rejects_changed_numerics_scope_and_replay_tolerance(tmp_path, key, value):
    config = _config("full")
    config[key] = value
    with pytest.raises(ValueError):
        contract.read_config(_write_config(tmp_path, config), "full")


def test_scalar_serializer_keeps_types_and_explains_undefined_ratios():
    from research.wedge_propagation.branch_strength.study import _scalar_rows

    details = {
        "alpha": torch.tensor([0.5, 0.6]),
        "kappa_max_node": torch.tensor([4, 7]),
        "message_reference_cosine": torch.tensor([0.9, torch.nan]),
        "message_reference_cosine_defined": torch.tensor([True, False]),
        "z": torch.ones(2, 4, 3),
    }
    identity = {"dataset": "DEBUG-Cora", "condition": "learned_wedge_rms", "layer": 0}
    rows = _scalar_rows(details, (11, 23), identity)
    assert rows[0]["seed"] == 11 and rows[1]["seed"] == 23
    assert rows[0]["kappa_max_node"] == 4 and isinstance(rows[0]["kappa_max_node"], int)
    assert rows[1]["message_reference_cosine"] is None
    assert rows[1]["message_reference_cosine_defined"] is False
    assert "z" not in rows[0]
    assert json.loads(json.dumps(rows, allow_nan=False)) == rows


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_scalar_serializer_rejects_unexplained_nonfinite_metrics(value):
    from research.wedge_propagation.branch_strength.study import _scalar_rows

    with pytest.raises(FloatingPointError):
        _scalar_rows({"alpha": torch.tensor([value])}, (11,), {})


def test_scalar_serializer_requires_false_defined_flag_for_nan():
    from research.wedge_propagation.branch_strength.study import _scalar_rows

    with pytest.raises(FloatingPointError):
        _scalar_rows(
            {"cosine": torch.tensor([torch.nan]), "cosine_defined": torch.tensor([True])}, (11,), {}
        )
    rows = _scalar_rows(
        {
            "kappa_ratio_mean": torch.tensor([torch.nan]),
            "kappa_ratio_defined": torch.tensor([False]),
        },
        (11,),
        {},
    )
    assert rows == [{"seed": 11, "kappa_ratio_mean": None, "kappa_ratio_defined": False}]


def _metric_row():
    return {
        "dataset": "DEBUG-Cora",
        "condition": "learned_wedge_rms",
        "seed": 11,
        "split": "test",
        "ce": 0.8,
        "accuracy": 0.5,
        "num_nodes": 24,
        "num_labeled_nodes": 9,
    }


def test_source_replay_accepts_declared_roundoff_and_checks_graph_sizes():
    from research.wedge_propagation.branch_strength.study import _check_replay

    original, replay = _metric_row(), _metric_row()
    replay["ce"] += 1e-6
    checks = _check_replay([replay], [original], _config())
    assert checks[0]["within_declared_tolerance"] is True
    assert 0 < checks[0]["ce_abs_error"] < 1e-5
    replay["num_nodes"] -= 1
    with pytest.raises(ValueError):
        _check_replay([replay], [original], _config())


@pytest.mark.parametrize("key,value", [("ce", 0.9), ("accuracy", 0.51), ("num_labeled_nodes", 8)])
def test_source_replay_rejects_changed_selected_checkpoint_metrics(key, value):
    from research.wedge_propagation.branch_strength.study import _check_replay

    original, replay = _metric_row(), _metric_row()
    replay[key] = value
    with pytest.raises(ValueError):
        _check_replay([replay], [original], _config())


@pytest.mark.parametrize("key,value", [("ce", float("nan")), ("accuracy", float("nan"))])
@pytest.mark.parametrize("corrupted_side", ["replay", "original"])
def test_source_replay_nonfinite_value_cannot_pass_tolerance_comparison(key, value, corrupted_side):
    from research.wedge_propagation.branch_strength.study import _check_replay

    original, replay = _metric_row(), _metric_row()
    (replay if corrupted_side == "replay" else original)[key] = value
    with pytest.raises((ValueError, FloatingPointError)):
        _check_replay([replay], [original], _config())


@pytest.mark.parametrize("key,value", [("ce", -0.5), ("accuracy", 1.5), ("accuracy", -0.5)])
def test_matching_invalid_metric_values_cannot_be_claimed_as_successful_replay(key, value):
    from research.wedge_propagation.branch_strength.study import _check_replay

    original, replay = _metric_row(), _metric_row()
    original[key] = replay[key] = value
    with pytest.raises(ValueError):
        _check_replay([replay], [original], _config())


@pytest.fixture(scope="module")
def completed_source(wedge_completed_classification_debug_run):
    from research.wedge_propagation.branch_strength.source import load_source

    return load_source(wedge_completed_classification_debug_run, "debug")


@pytest.mark.parametrize("condition", ["first_order", "fixed_wedge", "learned_wedge_rms"])
def test_actual_source_seed_repacking_preserves_selected_states(completed_source, condition):
    from research.wedge_propagation.branch_strength.study import _make_model, _state_for_seeds
    from research.wedge_propagation.classification.training import make_model

    source = completed_source
    graph = source.graphs["DEBUG-Cora"]
    state = _state_for_seeds(source, graph.name, condition, (23, 11))
    model = _make_model(source, graph, condition, (23, 11), 7)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.training is False
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, state[name])
    with torch.inference_mode():
        actual, _ = model(graph)
        for index, seed in enumerate((23, 11)):
            single = make_model(graph, condition, (seed,), source.config, 5).eval()
            single.load_state_dict(_state_for_seeds(source, graph.name, condition, (seed,)))
            expected, _ = single(graph)
            torch.testing.assert_close(actual[index], expected[0], rtol=1e-5, atol=1e-6)


def test_actual_source_seed_repacking_rejects_missing_or_duplicate_states(completed_source):
    from research.wedge_propagation.branch_strength.study import _state_for_seeds

    source = completed_source
    pack = source.learned_packs[0]
    with pytest.raises(ValueError, match="missing"):
        _state_for_seeds(source, pack.dataset, pack.condition, (999,))
    duplicate = SimpleNamespace(packs=[pack, pack])
    with pytest.raises(ValueError, match="duplicate"):
        _state_for_seeds(duplicate, pack.dataset, pack.condition, pack.seeds)


def test_actual_completed_debug_checkpoint_frozen_pack_smoke(completed_source):
    from research.wedge_propagation.branch_strength.source import assert_unchanged
    from research.wedge_propagation.branch_strength.study import _device_manifests, _evaluate_pack

    source = completed_source
    graph = source.graphs["DEBUG-Cora"]
    config = _config()
    manifests = _device_manifests(source.manifests[graph.name], torch.device("cpu"))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        output = _evaluate_pack(
            source, graph, "learned_wedge_rms", (11, 23), 5, config, manifests, {}
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert len(output["baseline_rows"]) == 6
    assert len(output["treatment_rows"]) == 162
    assert len(output["diagnostic_rows"]) == 112
    assert len(output["fixed_z_rows"]) == 40
    assert len(output["replay_rows"]) == 6
    assert all(row["within_declared_tolerance"] for row in output["replay_rows"])
    provenance = output["provenance"][0]
    assert provenance["optimizer_updates"] == provenance["trainable_parameters"] == 0
    assert provenance["model_hash_before"] == provenance["model_hash_after"]
    assert provenance["completed_seed_cases"] == 56
    norm_matched = [
        row for row in output["fixed_z_rows"] if row["treatment"].endswith("norm_matched")
    ]
    assert norm_matched
    assert all(abs(row["message_norm_ratio"] - 1) < 2e-6 for row in norm_matched)
    assert_unchanged(source)


def test_actual_two_cpu_subprocess_dispatch_keeps_complete_debug_scope(
    completed_source, tmp_path, capsys
):
    """Real two-worker smoke: every DEBUG seed/state/graph and treatment stays whole."""
    from research.wedge_propagation.branch_strength.source import assert_unchanged
    from research.wedge_propagation.branch_strength.study import _dispatch
    from research.wedge_propagation.classification.common import read_json, write_json

    source = completed_source
    config = _config()
    output = tmp_path / "two-worker-debug"
    output.mkdir()
    for name in ("plans", "workers", "calibration"):
        (output / name).mkdir()
    write_json(output / "config.json", config)
    result = _dispatch(output, config, source, ["cpu", "cpu"])
    coverage = contract.verify_coverage(
        config,
        result["baseline_rows"],
        result["treatment_rows"],
        result["diagnostic_rows"],
        result["fixed_z_rows"],
    )
    assert coverage == {**config["expected_counts"], "complete": True}
    assert len(result["replay_rows"]) == 144
    assert all(row["within_declared_tolerance"] for row in result["replay_rows"])
    assert len({(row["dataset"], row["condition"]) for row in result["baseline_rows"]}) == 24
    assert {row["seed"] for row in result["baseline_rows"]} == {11, 23}
    assert sum(row["completed_seed_cases"] for row in result["provenance"]) == 372
    for row in result["provenance"]:
        assert row["optimizer_updates"] == row["trainable_parameters"] == 0
        assert row["model_hash_before"] == row["model_hash_after"]
    measured = [row for row in result["resource_rows"] if row["scope"] == "calibration"]
    assert measured and {row["packed_runs"] for row in measured} == {1, 2}
    assert all(row["device"] == "cpu" and row["measured"] is True for row in measured)
    plans = [read_json(output / "plans" / f"worker-{index}.json") for index in range(2)]
    jobs = [(job["dataset"], job["condition"]) for plan in plans for job in plan["jobs"]]
    assert len(jobs) == len(set(jobs)) == 24
    assert all(plan["device"] == "cpu" and plan["num_workers"] == 2 for plan in plans)
    assert all(plan["jobs"] for plan in plans)
    for index in range(2):
        folder = output / "workers" / f"worker-{index}"
        completed = read_json(folder / "completion.json")
        assert completed["completed"] is completed["source_preserved"] is True
        assert completed["optimizer_updates"] == 0
        assert completed["jobs"] == len(plans[index]["jobs"])
        assert (folder / "results.json").is_file()
        log = (folder / "terminal.log").read_text(encoding="utf-8")
        assert f"[job] worker={index}" in log
        assert "[baseline]" in log and "replay=verified" in log
    displayed = capsys.readouterr().out
    assert "[job] worker=0" in displayed and "[job] worker=1" in displayed
    assert "[case]" in displayed and "[fixed Z]" in displayed
    assert_unchanged(source)
