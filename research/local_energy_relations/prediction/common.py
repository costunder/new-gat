"""Strict prediction contracts, exclusive artifacts and actual source provenance."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ...wedge_propagation.classification.common import (
    cpu_state,
    digest,
    file_sha256,
    save_checkpoint,
    write_csv,
    write_json,
)
from ..contract import source_manifest as parent_source_manifest

_FOLDER = Path(__file__).resolve().parent
_REPOSITORY = _FOLDER.parents[2]
_RUNTIME_VARIABLE = {
    "cpu_threads",
    "cpu_workers",
    "gpu_memory_safety_fraction",
    "relation_chunk_candidates",
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
    )


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def validate_config(config, profile=None):
    if not isinstance(config, dict):
        raise ValueError("prediction configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("explicit full or debug profile required")
    expected = read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(expected):
        raise ValueError("unknown or missing prediction configuration field")
    for key, value in expected.items():
        if key != "runtime" and not _same(config[key], value):
            raise ValueError(f"changed {key}: complete prediction contract required")
    if profile == "full":
        design = read_json(_FOLDER / "design_contract.json")
        if not _same(expected, design):
            raise ValueError("full executable configuration differs from design contract")
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
        or len(set(chunks)) != len(chunks)
        or any(type(value) is not int or value < 1 for value in chunks)
    ):
        raise ValueError("exact relation chunk candidates must be distinct positive integers")
    train = config["training"]
    cells = len(config["data"]["datasets"]) * len(config["conditions"])
    tuning = cells * len(train["learning_rate_candidates"]) * len(train["tuning_seeds"])
    final = cells * len(train["final_seeds"])
    if (tuning, final, tuning + final, (tuning + final) * train["epochs_per_run"]) != (
        train["tuning_runs"],
        train["final_runs"],
        train["total_runs"],
        train["total_updates"],
    ):
        raise ValueError("run and optimization budget disagrees with complete contract")
    return config


def read_config(path=None, profile="full"):
    if profile not in ("full", "debug"):
        raise ValueError("explicit full or debug profile required")
    return validate_config(read_json(path or _FOLDER / f"config_{profile}.json"), profile)


def source_manifest():
    """New package plus actual generic source and prior snapshot-loader helpers."""
    parent = parent_source_manifest()
    hashes = dict(parent["sha256"])
    receiver = _FOLDER.parent / "receiver_aggregation"
    reused = [
        receiver / name
        for name in (
            "__init__.py",
            "data.py",
            "contract.py",
            "operators.py",
            "config_full.json",
            "config_debug.json",
        )
    ]
    reused.append(
        _REPOSITORY / "research" / "wedge_propagation" / "classification" / "evaluation.py"
    )
    reused.append(
        _REPOSITORY / "research" / "wedge_propagation" / "classification" / "model.py"
    )
    files = sorted(set([*_FOLDER.rglob("*.py"), *_FOLDER.rglob("*.json"), *reused]))
    for path in files:
        if not path.is_file():
            raise ValueError(f"required source dependency missing: {path}")
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
        raise ValueError("prediction source changed during execution; artifacts preserved")


__all__ = [
    "cpu_state",
    "digest",
    "file_sha256",
    "save_checkpoint",
    "write_csv",
    "write_json",
    "read_json",
    "read_config",
    "validate_config",
    "source_manifest",
    "assert_source_unchanged",
]
