"""Explicit synthetic source fixtures; no training or artifact migrations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from chartgat import resume_compat as compat


@pytest.fixture
def transition(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setattr(compat, "REGISTRY_PATH", registry_path)
    helper_hash = hashlib.sha256(Path(compat.__file__).read_bytes()).hexdigest()
    registry = {
        "schema_version": 1,
        "patch_id": compat.PATCH_ID,
        "base_commit": compat.BASE_COMMIT,
        "changes": {
            "research/example/train.py": {"before": "a" * 64, "after": "b" * 64},
            compat.HELPER_SOURCE: {"before": None, "after": helper_hash},
        },
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    previous = {"research/example/train.py": "a" * 64, "model.py": "c" * 64}
    current = {
        "research/example/train.py": "b" * 64,
        "model.py": "c" * 64,
        compat.HELPER_SOURCE: helper_hash,
        compat.REGISTRY_SOURCE: hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    }
    return registry_path, registry, previous, current


def test_equal_snapshots_require_no_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "REGISTRY_PATH", tmp_path / "absent.json")
    assert compat.require_source_compatibility({"fixture": "stable"}, {"fixture": "stable"}) is None


def test_exact_forward_transition_retains_independent_provenance(transition):
    _, _, previous, current = transition
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    evidence = compat.require_source_compatibility(previous, current)
    assert evidence == {
        "patch_id": compat.PATCH_ID,
        "base_commit": compat.BASE_COMMIT,
        "registry_sha256": current[compat.REGISTRY_SOURCE],
        "previous_source_sha256": old,
        "current_source_sha256": new,
    }
    assert previous == old and current == new
    previous["model.py"] = "d" * 64
    current["model.py"] = "e" * 64
    assert evidence["previous_source_sha256"] == old
    assert evidence["current_source_sha256"] == new


@pytest.mark.parametrize("side", ["previous", "current"])
def test_modified_unchanged_source_is_rejected(transition, side):
    _, _, previous, current = transition
    (previous if side == "previous" else current)["model.py"] = "d" * 64
    with pytest.raises(ValueError, match="not covered"):
        compat.require_source_compatibility(previous, current)


@pytest.mark.parametrize("side", ["previous", "current"])
def test_inexact_reviewed_digest_is_rejected(transition, side):
    _, _, previous, current = transition
    (previous if side == "previous" else current)["research/example/train.py"] = "d" * 64
    assert not compat.snapshots_match(previous, current)


def test_source_deletion_is_rejected(transition):
    _, _, previous, current = transition
    current.pop("model.py")
    with pytest.raises(ValueError, match="cannot remove"):
        compat.require_source_compatibility(previous, current)


def test_unregistered_addition_is_rejected(transition):
    _, _, previous, current = transition
    current["extra.py"] = "d" * 64
    assert not compat.snapshots_match(previous, current)


@pytest.mark.parametrize("source", compat.COMPATIBILITY_SOURCE_FILES)
def test_both_compatibility_files_are_required(transition, source):
    _, _, previous, current = transition
    current.pop(source)
    with pytest.raises(ValueError, match="missing its compatibility"):
        compat.require_source_compatibility(previous, current)


def test_reverse_transition_is_not_implicitly_approved(transition):
    _, _, previous, current = transition
    assert compat.snapshots_match(previous, current)
    assert not compat.snapshots_match(current, previous)


def test_registry_changes_are_not_cached(transition):
    registry_path, registry, previous, current = transition
    assert compat.snapshots_match(previous, current)
    registry["description"] = "changed after initial snapshot"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="registry changed"):
        compat.require_source_compatibility(previous, current)


def test_helper_digest_must_match_live_implementation(transition):
    _, _, previous, current = transition
    current[compat.HELPER_SOURCE] = "d" * 64
    with pytest.raises(ValueError, match="helper changed"):
        compat.require_source_compatibility(previous, current)


def test_helper_digest_must_be_pinned(transition):
    registry_path, registry, previous, current = transition
    registry["changes"][compat.HELPER_SOURCE]["after"] = "d" * 64
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    current[compat.REGISTRY_SOURCE] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not pinned"):
        compat.require_source_compatibility(previous, current)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("patch_id", "unreviewed"), ("base_commit", "0" * 40)],
)
def test_registry_identity_is_pinned(transition, field, value):
    registry_path, registry, previous, current = transition
    registry[field] = value
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    current[compat.REGISTRY_SOURCE] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    assert not compat.snapshots_match(previous, current)


@pytest.mark.parametrize("name", ["/absolute.py", "../escape.py", "a\\b.py", "./a.py"])
def test_noncanonical_paths_are_rejected(transition, name):
    _, _, previous, current = transition
    previous[name] = "d" * 64
    current[name] = "d" * 64
    assert not compat.snapshots_match(previous, current)


def test_duplicate_registry_keys_are_rejected(transition):
    registry_path, _, previous, current = transition
    registry_path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        compat.require_source_compatibility(previous, current)


def test_adopt_updates_memory_and_keeps_original_provenance(transition):
    _, _, previous, current = transition
    original = copy.deepcopy(previous)
    record = {"source_sha256": previous, "scientific_recipe": {"epochs": 200}}
    evidence = compat.adopt_source_snapshot(record, current)
    assert record["source_sha256"] == current
    assert record["source_compatibility"] == [evidence]
    assert record["source_compatibility"][0]["previous_source_sha256"] == original
    assert record["scientific_recipe"] == {"epochs": 200}
    assert previous == original
    assert compat.adopt_source_snapshot(record, current) is None
    assert len(record["source_compatibility"]) == 1


def test_failed_adoption_is_atomic_in_memory(transition):
    _, _, previous, current = transition
    record = {"source_sha256": previous, "source_compatibility": "invalid"}
    original = copy.deepcopy(record)
    with pytest.raises(ValueError, match="provenance"):
        compat.adopt_source_snapshot(record, current)
    assert record == original


def test_partial_snapshot_still_requires_compatibility_additions(transition):
    _, _, previous, current = transition
    previous.pop("research/example/train.py")
    current.pop("research/example/train.py")
    assert compat.snapshots_match(previous, current)


def test_existing_registry_is_not_an_approved_addition(transition):
    _, _, previous, current = transition
    previous[compat.REGISTRY_SOURCE] = "d" * 64
    assert not compat.snapshots_match(previous, current)


@pytest.mark.parametrize("side", ["old", "new"])
def test_half_applied_release_is_rejected(transition, side):
    _, _, previous, current = transition
    if side == "old":
        current["research/example/train.py"] = previous["research/example/train.py"]
    else:
        previous["research/example/train.py"] = current["research/example/train.py"]
    with pytest.raises(ValueError, match="complete reviewed"):
        compat.require_source_compatibility(previous, current)
