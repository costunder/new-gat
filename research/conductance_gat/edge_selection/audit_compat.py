"""One-way, exact-source permission for the 03ec0f6 audit-only repair.

This never rewrites a training identity or permits changed model/data/optimizer
code. The registry pins both source scopes and every reviewed file byte change.
It is deliberately not a general source-ignore or checkpoint migration API.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
HELPER_SOURCE = "research/conductance_gat/edge_selection/audit_compat.py"
REGISTRY_PATH = Path(__file__).with_name("audit_compatibility_v1.json")
PATCH_ID = "edge-selection-exact-large-audit-v1"
BASE_COMMIT = "03ec0f644da2636f92d022a8efb3533a0dcf614a"
CHANGED_SOURCES = frozenset(
    {
        "research/conductance_gat/edge_selection/diagnostics.py",
        "research/conductance_gat/edge_selection/audit.py",
        "scripts/run_v5_edge_selection.py",
        HELPER_SOURCE,
    }
)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_map(value):
    if not isinstance(value, dict) or not value:
        return False
    for name, digest in value.items():
        if not isinstance(name, str) or not name or "\\" in name or ":" in name:
            return False
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or not _digest(digest)
        ):
            return False
    return True


def source_map_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("audit repair registry contains duplicate JSON keys")
        result[key] = value
    return result


def _registry():
    if REGISTRY_PATH.is_symlink() or not REGISTRY_PATH.is_file():
        raise ValueError("audit repair registry must be a regular file")
    raw = REGISTRY_PATH.read_bytes()
    registry = json.loads(raw, object_pairs_hook=_unique_pairs)
    if (
        not isinstance(registry, dict)
        or set(registry)
        != {"schema_version", "patch_id", "base_commit", "base_source_digests", "changes"}
        or type(registry["schema_version"]) is not int
        or registry["schema_version"] != 1
        or registry["patch_id"] != PATCH_ID
        or registry["base_commit"] != BASE_COMMIT
        or not isinstance(registry["base_source_digests"], dict)
        or set(registry["base_source_digests"]) != {"manifest", "training"}
        or not all(_digest(value) for value in registry["base_source_digests"].values())
        or not isinstance(registry["changes"], dict)
        or set(registry["changes"]) != CHANGED_SOURCES
    ):
        raise ValueError("audit repair registry identity or exact change set is invalid")
    for name, change in registry["changes"].items():
        if (
            not isinstance(change, dict)
            or set(change) != {"before", "after"}
            or not _digest(change["after"])
            or (
                change["before"] is not None
                if name == HELPER_SOURCE
                else not _digest(change["before"])
            )
            or change["before"] == change["after"]
        ):
            raise ValueError(f"audit repair source pin is invalid: {name}")
    return registry, hashlib.sha256(raw).hexdigest()


def require_source_compatibility(previous, current, *, scope):
    """Allow identical sources or this one reviewed complete old-to-live pair.

    All callers still validate configuration, data, runtime, budget, artifacts,
    and states independently. An old incomplete training checkpoint is not
    authorized here: the core trainer's strict identity check stays unchanged.
    """
    if scope not in {"manifest", "training"}:
        raise ValueError("unknown audit repair source scope")
    if previous == current:
        return None
    if not _source_map(previous) or not _source_map(current):
        raise ValueError("audit repair requires complete SHA256 source maps")
    registry, registry_sha = _registry()
    if source_map_digest(previous) != registry["base_source_digests"][scope]:
        raise ValueError("audit repair source is not the pinned 03ec0f6 release")
    if set(previous) - set(current):
        raise ValueError("audit repair cannot remove source files")
    for name, change in registry["changes"].items():
        path = ROOT / name
        if (
            current.get(name) != change["after"]
            or previous.get(name) != change["before"]
            or path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != change["after"]
        ):
            raise ValueError(f"audit repair is not the complete pinned live implementation: {name}")
    for name in set(previous) | set(current):
        before, after = previous.get(name), current.get(name)
        if before != after and registry["changes"].get(name) != {"before": before, "after": after}:
            raise ValueError(f"unreviewed source change cannot reuse prior evidence: {name}")
    return {
        "patch_id": PATCH_ID,
        "base_commit": BASE_COMMIT,
        "scope": scope,
        "registry_sha256": registry_sha,
        "previous_source_map_sha256": source_map_digest(previous),
        "current_source_map_sha256": source_map_digest(current),
        "changed_sources": copy.deepcopy(registry["changes"]),
        "source_semantics": "audit_only_exact_large_distribution_repair",
        "training_artifacts_rewritten": False,
        "calibration_semantics": (
            "original completed measurement retained; no new measurement claimed"
        ),
    }
