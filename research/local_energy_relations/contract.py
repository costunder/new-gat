"""Immutable scientific scope and portable provenance for the fixed local audit."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from ..wedge_propagation.classification.common import digest, file_sha256, write_csv, write_json

_FOLDER = Path(__file__).resolve().parent
_REPOSITORY = _FOLDER.parents[1]
_RUNTIME_VARIABLE = {
    "cpu_workers",
    "cpu_threads",
    "physical_graph_batch",
    "channel_chunk",
    "relation_batch",
    "gpu_memory_safety_fraction",
}


def _object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_object_pairs,
        parse_constant=_nonfinite,
    )


def _same(left, right):
    # JSON equality distinguishes True from 1 and 1 from 1.0, unlike Python ==.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def validate_config(config, profile=None):
    """Allow resource allocation changes only; never alter data, operators or scope."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("profile must be full or debug")
    canonical = _read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(canonical):
        raise ValueError("unknown or missing configuration field")
    for key, expected in canonical.items():
        if key != "runtime" and not _same(config[key], expected):
            raise ValueError(f"changed {key}: complete fixed local audit contract required")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(canonical["runtime"]):
        raise ValueError("unknown or missing runtime field")
    for key, expected in canonical["runtime"].items():
        if key not in _RUNTIME_VARIABLE and not _same(runtime[key], expected):
            raise ValueError(f"changed runtime {key}")
    for key in _RUNTIME_VARIABLE - {"gpu_memory_safety_fraction"}:
        value = runtime[key]
        if value != "auto" and (type(value) is not int or value < 1):
            raise ValueError(f"runtime {key} must be auto or a positive integer")
    fraction = runtime["gpu_memory_safety_fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 < fraction < 1
    ):
        raise ValueError("GPU memory safety fraction must be finite and between zero and one")
    return config


def read_config(path=None, profile="full"):
    """Read the separate full or DEBUG contract and reject silent reductions."""
    if profile not in ("full", "debug"):
        raise ValueError("profile must be full or debug")
    return validate_config(_read_json(path or _FOLDER / f"config_{profile}.json"), profile)


def source_manifest():
    """Hash the new package and actual generic import closure, not old model tracks."""
    wedge = _REPOSITORY / "research" / "wedge_propagation"
    reused = [
        wedge / "__init__.py",
        wedge / "data.py",
        wedge / "operators.py",
        wedge / "study.py",
        wedge / "algebra.py",
        wedge / "report.py",
        wedge / "classification" / "__init__.py",
        wedge / "classification" / "data.py",
        wedge / "classification" / "common.py",
        wedge / "classification" / "config_full.json",
        wedge / "classification" / "config_debug.json",
    ]
    files = sorted(set([*_FOLDER.glob("*.py"), *_FOLDER.glob("*.json"), *reused]))
    if any(not path.is_file() for path in files):
        raise ValueError("required source dependency missing")
    hashes = {path.relative_to(_REPOSITORY).as_posix(): file_sha256(path) for path in files}
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={_REPOSITORY.as_posix()}", "rev-parse", "HEAD"],
            cwd=_REPOSITORY,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return {"sha256": hashes, "code_digest": digest(hashes), "git_commit": commit}


def assert_source_unchanged(record):
    current = source_manifest()
    if current["sha256"] != record["sha256"] or current["code_digest"] != record["code_digest"]:
        raise ValueError("audit implementation changed during execution; output preserved")


__all__ = [
    "read_config",
    "validate_config",
    "source_manifest",
    "assert_source_unchanged",
    "digest",
    "file_sha256",
    "write_csv",
    "write_json",
]
