"""Read-only, exact-source compatibility for the reviewed RNG/worker repair.

Only recorded old and new file digests may differ. Model, data and recipe
checks remain mandatory at their callers. Published artifacts are not edited.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HELPER_SOURCE = "src/chartgat/resume_compat.py"
REGISTRY_SOURCE = "scripts/resume_compatibility_v1.json"
REGISTRY_PATH = ROOT / REGISTRY_SOURCE
COMPATIBILITY_SOURCE_FILES = (HELPER_SOURCE, REGISTRY_SOURCE)
PATCH_ID = "v5-rng-cycle-workers-v1"
BASE_COMMIT = "76e514a8bf444ef82a323f18ff31982908cf2d8f"


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and path.as_posix() == value and ".." not in path.parts


def _source_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(_source_path(key) and _digest(digest) for key, digest in value.items())
    )


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("source compatibility registry contains duplicate JSON keys")
        result[key] = value
    return result


def _registry() -> tuple[dict[str, Any], str]:
    if REGISTRY_PATH.is_symlink() or not REGISTRY_PATH.is_file():
        raise ValueError("source compatibility registry must be a regular file")
    try:
        raw = REGISTRY_PATH.read_bytes()
        registry = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("source compatibility registry is unreadable") from error
    if (
        not isinstance(registry, dict)
        or type(registry.get("schema_version")) is not int
        or registry["schema_version"] != 1
        or registry.get("patch_id") != PATCH_ID
        or registry.get("base_commit") != BASE_COMMIT
        or not isinstance(registry.get("changes"), dict)
        or not registry["changes"]
    ):
        raise ValueError("source compatibility registry identity is invalid")
    for name, change in registry["changes"].items():
        if (
            not _source_path(name)
            or name == REGISTRY_SOURCE
            or not isinstance(change, dict)
            or set(change) != {"before", "after"}
            or (change["before"] is not None and not _digest(change["before"]))
            or not _digest(change["after"])
            or change["before"] == change["after"]
        ):
            raise ValueError("source compatibility registry change is invalid")
    return registry, hashlib.sha256(raw).hexdigest()


def require_source_compatibility(previous: Any, current: Any) -> dict[str, Any] | None:
    """Accept equality or the exact reviewed one-way source transition.

    Equality retains existing callers' semantics, including their synthetic
    test snapshots. A transition requires actual SHA-256 maps and a live,
    hash-bound registry; unrelated or partially applied changes are rejected.
    """
    if previous == current:
        return None
    if not _source_map(previous) or not _source_map(current):
        raise ValueError("source snapshots differ and are not valid SHA-256 maps")
    registry, registry_sha256 = _registry()
    if any(name not in current for name in COMPATIBILITY_SOURCE_FILES):
        raise ValueError("source transition is missing its compatibility implementation")
    if current[REGISTRY_SOURCE] != registry_sha256:
        raise ValueError("source compatibility registry changed after the snapshot")
    helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current[HELPER_SOURCE] != helper_sha256:
        raise ValueError("source compatibility helper changed after the snapshot")
    helper_change = registry["changes"].get(HELPER_SOURCE)
    if helper_change != {"before": None, "after": helper_sha256}:
        raise ValueError("source compatibility helper is not pinned by the registry")
    removed = set(previous) - set(current)
    if removed:
        raise ValueError(f"source transition cannot remove files: {sorted(removed)}")
    for name, change in registry["changes"].items():
        if name in current and (
            current[name] != change["after"] or previous.get(name) != change["before"]
        ):
            raise ValueError(f"source transition is not the complete reviewed repair: {name}")
    for name in sorted(set(previous) | set(current)):
        before, after = previous.get(name), current.get(name)
        if before == after:
            continue
        if name == REGISTRY_SOURCE:
            if before is not None or after != registry_sha256:
                raise ValueError("source compatibility registry transition is not an addition")
            continue
        if registry["changes"].get(name) != {"before": before, "after": after}:
            raise ValueError(f"source change is not covered by the reviewed repair: {name}")
    return {
        "patch_id": registry["patch_id"],
        "base_commit": registry["base_commit"],
        "registry_sha256": registry_sha256,
        "previous_source_sha256": copy.deepcopy(previous),
        "current_source_sha256": copy.deepcopy(current),
    }


def adopt_source_snapshot(record: dict[str, Any], current: dict[str, str]) -> dict[str, Any] | None:
    """Update only a caller-owned in-memory record, retaining exact provenance."""
    evidence = require_source_compatibility(record.get("source_sha256"), current)
    if evidence is None:
        return None
    history = record.get("source_compatibility", [])
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("source compatibility provenance must be a list of objects")
    record["source_compatibility"] = [*history, copy.deepcopy(evidence)]
    record["source_sha256"] = copy.deepcopy(current)
    return evidence


def snapshots_match(previous: Any, current: Any) -> bool:
    """Test equality or the same narrow forward transition without mutations."""
    try:
        require_source_compatibility(previous, current)
    except ValueError:
        return False
    return True
