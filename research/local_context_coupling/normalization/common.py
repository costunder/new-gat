"""Immutable scientific contract; only measured resource allocations may vary."""
from __future__ import annotations

import math
from pathlib import Path

from ..classification.common import (
    cpu_state, digest, file_sha256, read_json, save_checkpoint, write_csv, write_json,
    source_manifest as classification_source_manifest,
)

_FOLDER = Path(__file__).resolve().parent
_REPOSITORY = _FOLDER.parents[2]
WEIGHTS = ("unit", "local_degree")
INTRA_POLICIES = ("graph", "local")
CROSS_POLICIES = ("graph", "edge")
VARIANTS = ("off", "fixed", "learned")
SCOPES = ("layer_0", "layer_1", "both")
CONDITIONS = tuple(
    f"{weight}__{intra}__{cross}__{variant}"
    for weight in WEIGHTS for intra in INTRA_POLICIES
    for cross, variant in (("none", "off"), ("graph", "fixed"), ("graph", "learned"),
                           ("edge", "fixed"), ("edge", "learned"))
)
_ALLOCATIONS = {"cpu_threads", "cpu_workers", "gpu_memory_safety_fraction", "relation_chunk_candidates"}


def parse_condition(condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown normalization condition; complete four-field identifier required")
    return tuple(condition.split("__"))


def condition_metadata(condition):
    weight, intra, cross, variant = parse_condition(condition)
    return {"weight_mode": weight, "intra_policy": intra, "cross_policy": cross,
            "variant": variant, "energy_operator": "applied_S_G",
            "cross_diagnostic_policy": "graph" if cross == "none" else cross}


def validate_config(config, profile=None):
    if not isinstance(config, dict):
        raise ValueError("normalization configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("explicit full or debug profile required")
    expected = read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(expected):
        raise ValueError("unknown or missing normalization contract field")
    for key, value in expected.items():
        if key != "runtime" and digest(config[key]) != digest(value):
            raise ValueError(f"changed {key}: complete normalization contract required")
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("all twenty normalization conditions required")
    if profile == "full" and digest(expected) != digest(read_json(_FOLDER / "design_contract.json")):
        raise ValueError("FULL configuration differs from design contract")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(expected["runtime"]):
        raise ValueError("unknown or missing runtime allocation")
    for key, value in expected["runtime"].items():
        if key not in _ALLOCATIONS and digest(runtime[key]) != digest(value):
            raise ValueError(f"changed runtime {key}")
    for key in ("cpu_workers", "cpu_threads"):
        value = runtime[key]
        if value != "auto" and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be auto or a positive integer")
    fraction = runtime["gpu_memory_safety_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("GPU memory safety fraction must be finite in (0,1)")
    chunks = runtime["relation_chunk_candidates"]
    if not isinstance(chunks, list) or not chunks or len(set(chunks)) != len(chunks) or any(type(x) is not int or x < 1 for x in chunks):
        raise ValueError("exact edge chunk candidates must be distinct positive integers")
    train = config["training"]
    cells = len(config["data"]["datasets"]) * len(CONDITIONS)
    tuning = cells * len(train["learning_rate_candidates"]) * len(train["tuning_seeds"])
    final = cells * len(train["final_seeds"])
    if (tuning, final, tuning + final, (tuning + final) * train["epochs_per_run"]) != (train["tuning_runs"], train["final_runs"], train["total_runs"], train["total_updates"]):
        raise ValueError("complete run/update budget differs")
    return config


def read_config(path=None, profile="full"):
    return validate_config(read_json(path or _FOLDER / f"config_{profile}.json"), profile)


def source_manifest():
    original = classification_source_manifest()
    hashes = dict(original["sha256"])
    for path in sorted([*_FOLDER.glob("*.py"), *_FOLDER.glob("*.json")]):
        hashes[path.relative_to(_REPOSITORY).as_posix()] = file_sha256(path)
    return {"sha256": hashes, "code_digest": digest(hashes), "git_commit": original["git_commit"],
            "legacy_classification_code_digest": original["code_digest"]}


def assert_source_unchanged(record):
    if source_manifest()["sha256"] != record["sha256"]:
        raise ValueError("scientific source changed during normalization; results preserved")
