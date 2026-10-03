"""Immutable science, complete source scope and portable implementation hashes."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..contract import digest, file_sha256, write_csv, write_json
from ..contract import source_manifest as parent_source_manifest

_FOLDER = Path(__file__).resolve().parent
_REPOSITORY = _FOLDER.parents[2]
_VARIABLE = {
    "cpu_workers",
    "cpu_threads",
    "physical_graph_batch",
    "channel_chunk",
    "relation_batch",
    "gpu_memory_safety_fraction",
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
    )


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def validate_config(config, profile=None):
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("profile must be full or debug")
    expected = read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(expected):
        raise ValueError("unknown or missing configuration field")
    for key, value in expected.items():
        if key != "runtime" and not _same(config[key], value):
            raise ValueError(f"changed {key}: complete receiver aggregation contract required")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(expected["runtime"]):
        raise ValueError("unknown or missing runtime allocation")
    for key, value in expected["runtime"].items():
        if key not in _VARIABLE and not _same(runtime[key], value):
            raise ValueError(f"changed runtime {key}")
    for key in _VARIABLE - {"gpu_memory_safety_fraction"}:
        value = runtime[key]
        if value != "auto" and (type(value) is not int or value < 1):
            raise ValueError(f"runtime {key} must be auto or positive integer")
    value = runtime["gpu_memory_safety_fraction"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError("GPU memory safety fraction must be finite and between zero and one")
    return config


def read_config(path=None, profile="full"):
    if profile not in ("full", "debug"):
        raise ValueError("profile must be full or debug")
    return validate_config(read_json(path or _FOLDER / f"config_{profile}.json"), profile)


def source_manifest():
    """The recursive new package plus the actual unchanged parent import closure."""
    parent = parent_source_manifest()
    hashes = dict(parent["sha256"])
    files = sorted([*_FOLDER.rglob("*.py"), *_FOLDER.rglob("*.json")])
    for path in files:
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
        raise ValueError("receiver implementation changed during execution; output preserved")


__all__ = [
    "read_config",
    "validate_config",
    "read_json",
    "source_manifest",
    "assert_source_unchanged",
    "digest",
    "file_sha256",
    "write_csv",
    "write_json",
]
