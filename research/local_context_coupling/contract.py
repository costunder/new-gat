"""Immutable complete-input contract for the first persistent-context audit."""

from __future__ import annotations

import math
from pathlib import Path

from ..local_energy_relations.contract import digest, file_sha256, write_csv, write_json
from ..local_energy_relations.receiver_aggregation.contract import read_json
from ..local_energy_relations.receiver_aggregation.contract import read_config as legacy_config
from ..local_energy_relations.contract import source_manifest as parent_manifest

_FOLDER = Path(__file__).resolve().parent
_REPOSITORY = _FOLDER.parents[1]
_ALLOCATIONS = {"cpu_workers", "cpu_threads", "physical_graph_batch", "channel_chunk", "gpu_memory_safety_fraction"}


def validate_config(config, profile=None):
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("profile must be full or debug")
    canonical = read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(canonical):
        raise ValueError("unknown or missing coupling contract field")
    for field, value in canonical.items():
        if field != "runtime" and digest(config[field]) != digest(value):
            raise ValueError(f"changed {field}: complete context coupling scope required")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(canonical["runtime"]):
        raise ValueError("unknown or missing runtime allocation")
    for field, value in canonical["runtime"].items():
        if field not in _ALLOCATIONS and digest(runtime[field]) != digest(value):
            raise ValueError(f"changed runtime {field}")
    for field in _ALLOCATIONS - {"gpu_memory_safety_fraction"}:
        value = runtime[field]
        if value != "auto" and (type(value) is not int or value < 1):
            raise ValueError(f"runtime {field} must be auto or a positive integer")
    fraction = runtime["gpu_memory_safety_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("memory safety fraction must be finite and between zero and one")
    return config


def read_config(path=None, profile="full"):
    return validate_config(read_json(path or _FOLDER / f"config_{profile}.json"), profile)


def source_reader_config(config):
    """Use the unchanged snapshot reader's canonical contract, not its algorithm."""
    validate_config(config)
    legacy = legacy_config(profile=config["profile"])
    if digest(legacy["source"]) != digest(config["source"]) or legacy["weights"] != config["weights"] or legacy["states"] != config["states"]:
        raise ValueError("initial source graph/features/C/reference contract differs")
    for field in _ALLOCATIONS:
        legacy["runtime"][field] = config["runtime"][field]
    return legacy


def source_manifest():
    parent = parent_manifest()
    hashes = dict(parent["sha256"])
    reader = _REPOSITORY / "research/local_energy_relations/receiver_aggregation"
    files = [*_FOLDER.glob("*.py"), *_FOLDER.glob("*.json")]
    files += [reader / name for name in ("__init__.py", "data.py", "contract.py", "config_full.json", "config_debug.json")]
    files.append(_REPOSITORY / "research/wedge_propagation/classification/model.py")
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(f"required scientific source is missing: {path}")
        hashes[path.relative_to(_REPOSITORY).as_posix()] = file_sha256(path)
    return {"sha256": hashes, "code_digest": digest(hashes), "git_commit": parent["git_commit"], "parent_code_digest": parent["code_digest"]}


def assert_source_unchanged(record):
    if source_manifest()["sha256"] != record["sha256"]:
        raise ValueError("scientific source changed during audit; results preserved")


__all__ = ["read_config", "validate_config", "source_reader_config", "read_json", "source_manifest", "assert_source_unchanged", "digest", "file_sha256", "write_csv", "write_json"]
