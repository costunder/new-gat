"""Pinned classification contract and the full scientific import closure."""
from __future__ import annotations

import math
from pathlib import Path

from ...local_energy_relations.prediction.common import (
    cpu_state, digest, file_sha256, read_json, save_checkpoint, write_csv, write_json,
    source_manifest as data_source_manifest,
)
from ..contract import source_manifest as coupling_source_manifest

_FOLDER = Path(__file__).resolve().parent
_REPOSITORY = _FOLDER.parents[2]
VARIANTS = ("off", "fixed", "learned")
CONDITIONS = tuple(f"{mode}__{variant}" for mode in ("unit", "local_degree") for variant in VARIANTS)
_ALLOCATIONS = {"cpu_threads", "cpu_workers", "gpu_memory_safety_fraction", "relation_chunk_candidates"}


def validate_config(config, profile=None):
    if not isinstance(config, dict):
        raise ValueError("classification configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("explicit full or debug profile required")
    expected = read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(expected):
        raise ValueError("unknown or missing classification contract field")
    for key, value in expected.items():
        if key != "runtime" and digest(config[key]) != digest(value):
            raise ValueError(f"changed {key}: complete off/fixed/learned contract required")
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("all six conditions are required")
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
    coupling, data = coupling_source_manifest(), data_source_manifest()
    hashes = {**coupling["sha256"], **data["sha256"]}
    for path in sorted([*_FOLDER.glob("*.py"), *_FOLDER.glob("*.json")]):
        hashes[path.relative_to(_REPOSITORY).as_posix()] = file_sha256(path)
    return {"sha256": hashes, "code_digest": digest(hashes), "git_commit": coupling["git_commit"], "coupling_code_digest": coupling["code_digest"], "data_code_digest": data["code_digest"]}


def assert_source_unchanged(record):
    if source_manifest()["sha256"] != record["sha256"]:
        raise ValueError("scientific source changed during classification; results preserved")


__all__ = ["VARIANTS", "CONDITIONS", "cpu_state", "digest", "file_sha256", "read_json", "save_checkpoint", "write_csv", "write_json", "read_config", "validate_config", "source_manifest", "assert_source_unchanged"]
