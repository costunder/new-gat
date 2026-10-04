"""Strict placement contracts and provenance for actual reused dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..prediction.common import (
    cpu_state,
    digest,
    file_sha256,
    read_json,
    save_checkpoint,
    source_manifest as prediction_source_manifest,
    write_csv,
    write_json,
)

_FOLDER = Path(__file__).resolve().parent
_REPOSITORY = _FOLDER.parents[2]
DATASETS = ("Cora", "CiteSeer", "PubMed")
VARIANTS = ("base", "within", "between", "both")
PLACEMENTS = ("hidden", "output", "all")
CONDITIONS = tuple(
    condition
    for mode in ("unit", "local_degree")
    for condition in (
        f"{mode}__base",
        *(f"{mode}__{variant}__{placement}"
          for variant in VARIANTS[1:] for placement in PLACEMENTS),
    )
)
_RUNTIME_VARIABLE = {
    "cpu_threads",
    "cpu_workers",
    "gpu_memory_safety_fraction",
    "relation_chunk_candidates",
}


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def validate_config(config, profile=None):
    """Only allocation choices may vary; the complete 20-condition study is pinned."""
    if not isinstance(config, dict):
        raise ValueError("placement configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("explicit full or debug profile required")
    expected = read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(expected):
        raise ValueError("unknown or missing placement configuration field")
    for key, value in expected.items():
        if key != "runtime" and not _same(config[key], value):
            raise ValueError(f"changed {key}: complete placement contract required")
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("all distinct placement conditions are required")
    if profile == "full" and not _same(expected, read_json(_FOLDER / "design_contract.json")):
        raise ValueError("full executable configuration differs from placement design contract")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(expected["runtime"]):
        raise ValueError("unknown or missing runtime allocation")
    for key, value in expected["runtime"].items():
        if key not in _RUNTIME_VARIABLE and not _same(runtime[key], value):
            raise ValueError(f"changed runtime {key}")
    for key in ("cpu_threads", "cpu_workers"):
        value = runtime[key]
        if value != "auto" and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be auto or a positive integer")
    fraction = runtime["gpu_memory_safety_fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction < 1
    ):
        raise ValueError("GPU memory safety fraction must lie strictly between zero and one")
    chunks = runtime["relation_chunk_candidates"]
    if (
        not isinstance(chunks, list)
        or not chunks
        or any(type(value) is not int or value < 1 for value in chunks)
        or len(set(chunks)) != len(chunks)
    ):
        raise ValueError("exact relation chunk candidates must be distinct positive integers")
    train = config["training"]
    cells = len(config["data"]["datasets"]) * len(config["conditions"])
    tuning = cells * len(train["learning_rate_candidates"]) * len(train["tuning_seeds"])
    final = cells * len(train["final_seeds"])
    if (tuning, final, tuning + final, (tuning + final) * train["epochs_per_run"]) != (
        train["tuning_runs"], train["final_runs"], train["total_runs"], train["total_updates"]
    ):
        raise ValueError("run and optimization budget disagrees with complete placement contract")
    return config


def read_config(path=None, profile="full"):
    if profile not in ("full", "debug"):
        raise ValueError("explicit full or debug profile required")
    return validate_config(read_json(path or _FOLDER / f"config_{profile}.json"), profile)


def source_manifest():
    """Include placement code and the unchanged prediction loader/operator closure."""
    parent = prediction_source_manifest()
    hashes = dict(parent["sha256"])
    for path in sorted(set([*_FOLDER.rglob("*.py"), *_FOLDER.rglob("*.json")])):
        if not path.is_file():
            raise ValueError(f"required placement source dependency missing: {path}")
        hashes[path.relative_to(_REPOSITORY).as_posix()] = file_sha256(path)
    return {
        "sha256": hashes,
        "code_digest": digest(hashes),
        "git_commit": parent["git_commit"],
        "parent_code_digest": parent["code_digest"],
    }


def assert_source_unchanged(record):
    current = source_manifest()
    if current["sha256"] != record["sha256"] or current["code_digest"] != record["code_digest"]:
        raise ValueError("placement source changed during execution; artifacts preserved")


__all__ = [
    "DATASETS", "VARIANTS", "PLACEMENTS", "CONDITIONS", "cpu_state", "digest",
    "file_sha256", "save_checkpoint", "write_csv", "write_json", "read_json",
    "read_config", "validate_config", "source_manifest", "assert_source_unchanged",
]
