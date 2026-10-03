"""Portable, complete scalar-artifact validation with an actual completed DEBUG run."""

from __future__ import annotations

import csv
import itertools
import json
import shutil
from pathlib import Path

import pytest
import torch

from research.wedge_propagation.branch_analysis import source
from research.wedge_propagation.branch_strength.contract import expected_counts, read_config

ROOT = Path(__file__).resolve().parents[1]
CACHED = ROOT / "results/wedge-branch-strength-debug-20261004-02"


@pytest.fixture
def saved_debug(tmp_path):
    if not CACHED.is_dir():
        pytest.skip("actual completed Experiment 4.1 DEBUG scalar artifacts are unavailable")
    output = tmp_path / "portable-debug"
    output.mkdir()
    for name in source.REQUIRED_FILES:
        shutil.copyfile(CACHED / name, output / name)
    return output


def _change_json(folder, name, modify):
    path = folder / name
    value = json.loads(path.read_text(encoding="utf-8"))
    modify(value)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _change_csv(folder, name, modify):
    path = folder / name
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    modify(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_actual_complete_debug_is_portable_and_does_not_execute_models_or_cuda(
    saved_debug, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("scalar analysis must not load/execute checkpoints or initialize CUDA")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    loaded = source.load_run(saved_debug)
    assert loaded.config["profile"] == "debug"
    assert loaded.completion["actual_data"] is False
    assert {key: len(loaded.tables[key]) for key in loaded.tables if key != "resource_rows"} == {
        "baseline_rows": 144,
        "treatment_rows": 972,
        "diagnostic_rows": 744,
        "fixed_z_rows": 240,
        "replay_rows": 144,
    }
    assert set(loaded.hashes) == set(source.REQUIRED_FILES)
    source.assert_unchanged(loaded)


@pytest.mark.parametrize("name", source.REQUIRED_FILES)
def test_required_artifacts_cannot_be_omitted(saved_debug, name):
    (saved_debug / name).unlink()
    with pytest.raises(FileNotFoundError):
        source.load_run(saved_debug)


@pytest.mark.parametrize("name", source.REQUIRED_FILES)
def test_artifact_edits_after_loading_are_detected(saved_debug, name):
    loaded = source.load_run(saved_debug)
    with (saved_debug / name).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed during analysis"):
        source.assert_unchanged(loaded)


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("completion.json", "completed", False),
        ("completion.json", "profile", "full"),
        ("completion.json", "actual_data", True),
        ("completion.json", "optimizer_updates", 1),
        ("completion.json", "optimizer_updates", False),
        ("completion.json", "source_files_and_models_preserved", False),
        ("completion.json", "scope", "full_public_split_classification"),
        ("contract.json", "optimizer_updates", 1),
        ("contract.json", "source_files_and_models_preserved", False),
        ("contract.json", "interpretation", "choose_best_treatment_by_test_accuracy"),
        ("config.json", "sampling_ratio", 0.1),
        ("config.json", "manifests_per_dataset", 1),
        ("config.json", "final_seeds", [11]),
    ],
)
def test_frozen_scope_and_canonical_contract_cannot_change(saved_debug, name, key, value):
    _change_json(saved_debug, name, lambda record: record.update({key: value}))
    with pytest.raises(ValueError):
        source.load_run(saved_debug)


@pytest.mark.parametrize(
    "name",
    [
        "baseline_metrics.csv",
        "treatments.csv",
        "layer_diagnostics.csv",
        "fixed_z.csv",
        "source_replay.csv",
    ],
)
@pytest.mark.parametrize("operation", ["missing", "duplicate", "foreign_seed"])
def test_every_saved_scalar_case_is_required_once(saved_debug, name, operation):
    def mutate(rows):
        if operation == "missing":
            rows.pop()
        elif operation == "duplicate":
            rows.append(dict(rows[0]))
        else:
            rows[0]["seed"] = "999"

    _change_csv(saved_debug, name, mutate)
    with pytest.raises(ValueError):
        source.load_run(saved_debug)


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("baseline_metrics.csv", "ce", "NaN"),
        ("baseline_metrics.csv", "ce", "-1"),
        ("baseline_metrics.csv", "accuracy", "1.1"),
        ("baseline_metrics.csv", "seed", "11.0"),
        ("baseline_metrics.csv", "num_nodes", "12"),
        ("treatments.csv", "num_labeled_nodes", "1"),
        ("layer_diagnostics.csv", "beta_t_norm", "-1"),
        ("layer_diagnostics.csv", "kappa_reference", "0"),
        ("layer_diagnostics.csv", "alpha", "1.2"),
        ("layer_diagnostics.csv", "alpha_l_to_input_defined", "False"),
        ("layer_diagnostics.csv", "cosine_l_input", "2"),
        ("fixed_z.csv", "beta_t_norm", "Inf"),
        ("source_replay.csv", "within_declared_tolerance", "False"),
        ("source_replay.csv", "accuracy_abs_error", "0.0001"),
        ("source_replay.csv", "ce_abs_error", "1"),
        ("source_replay.csv", "ce_abs_error", "-0.001"),
    ],
)
def test_invalid_numbers_flags_and_replay_errors_fail(saved_debug, name, key, value):
    _change_csv(saved_debug, name, lambda rows: rows[0].update({key: value}))
    with pytest.raises(ValueError):
        source.load_run(saved_debug)


@pytest.mark.parametrize(
    "kind", ["hash", "updates", "trainable", "cases", "missing", "duplicate", "checkpoint"]
)
def test_frozen_model_provenance_requires_every_selected_model_and_case(saved_debug, kind):
    def mutate(rows):
        if kind == "hash":
            rows[0]["model_hash_after"] = "0" * 64
        elif kind == "updates":
            rows[0]["optimizer_updates"] = 1
        elif kind == "trainable":
            rows[0]["trainable_parameters"] = 1
        elif kind == "cases":
            rows[0]["completed_seed_cases"] = 1
        elif kind == "missing":
            rows.pop()
        elif kind == "duplicate":
            rows.append(dict(rows[0]))
        else:
            key = next(iter(rows[0]["source_checkpoint_hashes"]))
            rows[0]["source_checkpoint_hashes"][key] = "0" * 64

    _change_json(saved_debug, "frozen_model_provenance.json", mutate)
    with pytest.raises(ValueError):
        source.load_run(saved_debug)


def test_diagnostic_code_map_must_match_preserved_code(saved_debug):
    _change_json(
        saved_debug, "source.json", lambda record: record["sha256"].update({"core.py": "0" * 64})
    )
    with pytest.raises(ValueError, match="code/config hashes"):
        source.load_run(saved_debug)


def test_recorded_training_contract_and_artifact_hashes_remain_consistent(saved_debug):
    _change_json(
        saved_debug,
        "source_run.json",
        lambda record: record["artifact_hashes"].update({"metrics.csv": "0" * 64}),
    )
    with pytest.raises(ValueError, match="artifact digest"):
        source.load_run(saved_debug)


def test_original_training_folder_does_not_need_to_exist(saved_debug):
    def change_run(record):
        record["directory"] = "/portable/missing/training"

    _change_json(saved_debug, "source_run.json", change_run)
    _change_json(
        saved_debug,
        "contract.json",
        lambda record: record.update(source_run="/portable/missing/training"),
    )

    def change_paths(rows):
        old = json.loads((CACHED / "source_run.json").read_text(encoding="utf-8"))["directory"]
        for row in rows:
            row["source_checkpoint_hashes"] = {
                path.replace(old, "/portable/missing/training").replace("\\", "/"): sha
                for path, sha in row["source_checkpoint_hashes"].items()
            }

    _change_json(saved_debug, "frozen_model_provenance.json", change_paths)
    loaded = source.load_run(saved_debug)
    assert loaded.contract["source_run"] == "/portable/missing/training"


@pytest.mark.parametrize("header", ["seed,seed\n11,11\n", "seed,,ce\n11,,1\n", "seed,ce\n11\n"])
def test_csv_duplicate_blank_and_misaligned_columns_are_rejected(tmp_path, header):
    path = tmp_path / "invalid.csv"
    path.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError):
        source._read_csv(path)


def test_json_duplicate_keys_and_nonfinite_constants_are_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    for text in ('{"profile":"debug","profile":"full"}', '{"x":NaN}'):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            source._json(path)


def test_full_contract_and_provenance_counts_without_fabricating_performance():
    config = read_config(
        ROOT / "research/wedge_propagation/branch_strength/config_full.json", "full"
    )
    declared = expected_counts(config)
    assert declared["source_final_models"] == 120
    assert declared["end_to_end_seed_cases"] == 2370
    # Structural unit fixture only: no graph, prediction, metric or completion artifact.
    provenance = []
    checkpoint_sha = "a" * 64
    for dataset, condition in itertools.product(config["data"]["datasets"], config["conditions"]):
        count = len(source.variants(config)) if condition.startswith("learned_wedge") else 1
        provenance.append(
            {
                "dataset": dataset,
                "condition": condition,
                "seeds": config["final_seeds"],
                "model_hash_before": "b" * 64,
                "model_hash_after": "b" * 64,
                "optimizer_updates": 0,
                "trainable_parameters": 0,
                "completed_seed_cases": count * len(config["final_seeds"]),
                "source_checkpoint_hashes": {"/unit-fixture/selected.pt": checkpoint_sha},
            }
        )
    source._verify_provenance(
        config,
        provenance,
        {
            "directory": "/unit-fixture",
            "artifact_hashes": {"selected.pt": checkpoint_sha},
        },
    )
    provenance[0]["seeds"] = config["final_seeds"][:-1]
    provenance[0]["completed_seed_cases"] -= 1
    with pytest.raises(ValueError, match="coverage is incomplete"):
        source._verify_provenance(
            config,
            provenance,
            {
                "directory": "/unit-fixture",
                "artifact_hashes": {"selected.pt": checkpoint_sha},
            },
        )
