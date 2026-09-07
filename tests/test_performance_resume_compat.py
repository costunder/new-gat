"""Synthetic digest fixtures only; no training or actual artifact migrations."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from chartgat import resume_compat as compat


@pytest.fixture
def performance(tmp_path, monkeypatch):
    path = tmp_path / "debug-performance-registry.json"
    monkeypatch.setattr(compat, "REGISTRY_PATH", path)
    live_helper = hashlib.sha256(Path(compat.__file__).read_bytes()).hexdigest()
    old_helper = "1" * 64
    old_registry = {
        "schema_version": 1,
        "patch_id": compat.PATCH_ID,
        "base_commit": compat.BASE_COMMIT,
        "changes": {
            "research/train.py": {"before": "2" * 64, "after": "3" * 64},
            compat.HELPER_SOURCE: {"before": None, "after": old_helper},
        },
    }
    original_bytes = json.dumps(old_registry).encode()
    old_hash = hashlib.sha256(original_bytes).hexdigest()
    registry = copy.deepcopy(old_registry)
    registry["performance_repair"] = {
        "patch_id": compat.PERFORMANCE_PATCH_ID,
        "base_commit": compat.PERFORMANCE_BASE_COMMIT,
        "registry_before_sha256": old_hash,
        "changes": {
            "research/train.py": {"before": "3" * 64, "after": "4" * 64},
            "src/new_performance.py": {"before": None, "after": "5" * 64},
            compat.HELPER_SOURCE: {"before": old_helper, "after": live_helper},
        },
    }
    path.write_text(json.dumps(registry), encoding="utf-8")
    previous = {
        "research/train.py": "3" * 64,
        "unchanged_model.py": "6" * 64,
        compat.HELPER_SOURCE: old_helper,
        compat.REGISTRY_SOURCE: old_hash,
    }
    current = {
        "research/train.py": "4" * 64,
        "unchanged_model.py": "6" * 64,
        "src/new_performance.py": "5" * 64,
        compat.HELPER_SOURCE: live_helper,
        compat.REGISTRY_SOURCE: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return path, registry, previous, current


def _publish_fixture(path, registry, current):
    path.write_text(json.dumps(registry), encoding="utf-8")
    current[compat.REGISTRY_SOURCE] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_performance_repair_is_one_way_state_preserving_not_bitwise(performance):
    _, _, previous, current = performance
    old, new = copy.deepcopy(previous), copy.deepcopy(current)
    evidence = compat.require_source_compatibility(previous, current)
    assert evidence["patch_id"] == compat.PERFORMANCE_PATCH_ID
    assert evidence["base_commit"] == compat.PERFORMANCE_BASE_COMMIT
    assert evidence["registry_before_sha256"] == previous[compat.REGISTRY_SOURCE]
    assert evidence["source_semantics"] == "state_preserving_performance_repair"
    assert evidence["bitwise_numerical_identity"] is False
    assert evidence["resource_plan_semantics"] == (
        "historical measured selection retained; no new measurement claimed"
    )
    assert previous == old and current == new
    assert not compat.snapshots_match(current, previous)
    previous["unchanged_model.py"] = "7" * 64
    assert evidence["previous_source_sha256"] == old


@pytest.mark.parametrize("source", [compat.REGISTRY_SOURCE, compat.HELPER_SOURCE])
def test_old_registry_and_helper_must_match_the_exact_reviewed_release(performance, source):
    _, _, previous, current = performance
    previous[source] = "7" * 64
    assert not compat.snapshots_match(previous, current)


@pytest.mark.parametrize("source", compat.COMPATIBILITY_SOURCE_FILES)
def test_live_registry_and_helper_must_match_current_snapshots(performance, source):
    _, _, previous, current = performance
    current[source] = "7" * 64
    assert not compat.snapshots_match(previous, current)


@pytest.mark.parametrize("side", ["previous", "current"])
def test_unrelated_source_change_is_still_rejected(performance, side):
    _, _, previous, current = performance
    (previous if side == "previous" else current)["unchanged_model.py"] = "7" * 64
    with pytest.raises(ValueError, match="not covered"):
        compat.require_source_compatibility(previous, current)


@pytest.mark.parametrize("side", ["before", "after"])
def test_half_applied_performance_patch_is_rejected(performance, side):
    _, _, previous, current = performance
    if side == "before":
        previous["research/train.py"] = current["research/train.py"]
    else:
        current["research/train.py"] = previous["research/train.py"]
    with pytest.raises(ValueError, match="complete reviewed performance"):
        compat.require_source_compatibility(previous, current)


def test_deletions_and_unregistered_additions_are_rejected(performance):
    _, _, previous, current = performance
    changed = dict(current)
    changed.pop("unchanged_model.py")
    with pytest.raises(ValueError, match="cannot remove"):
        compat.require_source_compatibility(previous, changed)
    changed = {**current, "unexpected.py": "7" * 64}
    assert not compat.snapshots_match(previous, changed)


def test_known_added_file_cannot_have_preexisting_or_wrong_content(performance):
    _, _, previous, current = performance
    previous["src/new_performance.py"] = "5" * 64
    assert not compat.snapshots_match(previous, current)


@pytest.mark.parametrize(
    "field,value",
    [
        ("patch_id", "unreviewed"),
        ("base_commit", "0" * 40),
        ("registry_before_sha256", None),
        ("changes", {}),
    ],
)
def test_performance_registry_identity_is_pinned(performance, field, value):
    path, registry, previous, current = performance
    registry["performance_repair"][field] = value
    _publish_fixture(path, registry, current)
    assert not compat.snapshots_match(previous, current)


@pytest.mark.parametrize(
    "name", ["/absolute.py", "../escape.py", "a\\b.py", compat.REGISTRY_SOURCE]
)
def test_performance_registry_change_paths_cannot_escape_or_self_pin(performance, name):
    path, registry, previous, current = performance
    registry["performance_repair"]["changes"][name] = {"before": None, "after": "7" * 64}
    _publish_fixture(path, registry, current)
    assert not compat.snapshots_match(previous, current)


def test_performance_helper_must_match_preserved_original_registry_pin(performance):
    path, registry, previous, current = performance
    registry["changes"][compat.HELPER_SOURCE]["after"] = "7" * 64
    _publish_fixture(path, registry, current)
    with pytest.raises(ValueError, match="helpers are not pinned"):
        compat.require_source_compatibility(previous, current)


def test_original_76_release_is_not_implicitly_chained_through_performance_repair(performance):
    _, _, previous, current = performance
    previous.pop(compat.HELPER_SOURCE)
    previous.pop(compat.REGISTRY_SOURCE)
    previous["research/train.py"] = "2" * 64
    with pytest.raises(ValueError, match="pinned old registry"):
        compat.require_source_compatibility(previous, current)


def test_performance_evidence_is_appended_without_mutating_recipe_or_resource_selection(
    performance,
):
    _, _, previous, current = performance
    record = {
        "source_sha256": copy.deepcopy(previous),
        "source_compatibility": [{"patch_id": "earlier historical record"}],
        "config": {"epochs": 750, "resource_plan": {"sha256": "historical selection"}},
        "epoch": 66,
    }
    original_config = copy.deepcopy(record["config"])
    evidence = compat.adopt_source_snapshot(record, current)
    assert record["config"] == original_config and record["epoch"] == 66
    assert len(record["source_compatibility"]) == 2
    assert record["source_compatibility"][-1] == evidence
    assert compat.adopt_source_snapshot(record, current) is None


def test_equal_source_snapshots_still_ignore_missing_or_malformed_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "REGISTRY_PATH", tmp_path / "absent.json")
    assert compat.require_source_compatibility({"debug": "same"}, {"debug": "same"}) is None
    assert compat.COMPATIBILITY_SOURCE_FILES == (compat.HELPER_SOURCE, compat.REGISTRY_SOURCE)


def test_performance_registry_does_not_allow_unknown_policy_fields(performance):
    path, registry, previous, current = performance
    registry["performance_repair"]["allow_any_source"] = True
    _publish_fixture(path, registry, current)
    assert not compat.snapshots_match(previous, current)


def _historical_pair(registry):
    before, after = {}, {}
    for name, change in registry["changes"].items():
        after[name] = change["after"]
        if change["before"] is not None:
            before[name] = change["before"]
    after[compat.REGISTRY_SOURCE] = registry["performance_repair"]["registry_before_sha256"]
    return before, after


def test_archived_source_pair_is_attested_after_registry_extension_without_live_upgrade(
    performance,
):
    _, registry, _, live = performance
    before, after = _historical_pair(registry)
    original = copy.deepcopy((before, after))
    evidence = compat.require_source_compatibility(before, after)
    assert evidence["patch_id"] == compat.PATCH_ID
    assert evidence["registry_sha256"] == after[compat.REGISTRY_SOURCE]
    assert evidence["source_semantics"] == "historical_source_pair_attestation"
    assert evidence["runtime_upgrade_authorized"] is False
    assert (before, after) == original
    assert not compat.snapshots_match(after, before)
    assert not compat.snapshots_match(before, live)


@pytest.mark.parametrize("substitution", ["registry", "helper", "patched_model"])
def test_historical_attestation_cannot_smuggle_any_live_patch_source(performance, substitution):
    _, registry, _, live = performance
    before, after = _historical_pair(registry)
    key = {
        "registry": compat.REGISTRY_SOURCE,
        "helper": compat.HELPER_SOURCE,
        "patched_model": "research/train.py",
    }[substitution]
    after[key] = live[key]
    assert not compat.snapshots_match(before, after)


def test_historical_attestation_preserves_original_exact_before_and_unrelated_guards(performance):
    _, registry, _, _ = performance
    before, after = _historical_pair(registry)
    before["research/train.py"] = "7" * 64
    assert not compat.snapshots_match(before, after)
    before, after = _historical_pair(registry)
    before["unreviewed.py"] = "7" * 64
    after["unreviewed.py"] = "8" * 64
    assert not compat.snapshots_match(before, after)


def test_historical_attestation_rejects_wrong_old_registry_even_with_exact_old_helper(performance):
    _, registry, _, _ = performance
    before, after = _historical_pair(registry)
    after[compat.REGISTRY_SOURCE] = "7" * 64
    assert not compat.snapshots_match(before, after)


def test_historical_attestation_does_not_treat_an_existing_helper_as_an_addition(performance):
    _, registry, _, _ = performance
    before, after = _historical_pair(registry)
    before[compat.HELPER_SOURCE] = after[compat.HELPER_SOURCE]
    assert not compat.snapshots_match(before, after)


@pytest.mark.parametrize("inventory", ["rich", "conductance", "resource"])
def test_registered_performance_repair_covers_real_server_source_inventory(inventory):
    """Read actual base Git blobs; normalize only the explicit server-LF test view."""
    from scripts import run_conductance_scaling, run_rich_scaling, training_resource_plan

    git = shutil.which("git")
    if git is None:
        pytest.skip("read-only real-source regression requires Git")
    root = compat.ROOT
    prefix = [git, "-c", f"safe.directory={root.as_posix()}"]
    base = compat.PERFORMANCE_BASE_COMMIT
    listing = subprocess.run(
        [*prefix, "ls-tree", "-r", "--name-only", base],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if listing.returncode:
        pytest.skip("read-only real-source regression requires the recorded base commit")
    providers = {
        "rich": run_rich_scaling._source_snapshot,
        "conductance": run_conductance_scaling._source_snapshot,
        "resource": training_resource_plan.source_snapshot,
    }
    recorded = json.loads(compat.REGISTRY_PATH.read_bytes())
    assert recorded["performance_repair"]["base_commit"] == base
    current = providers[inventory]()
    linux_current = {
        name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in current
    }
    if any(current[name] != linux_current[name] for name in compat.COMPATIBILITY_SOURCE_FILES):
        pytest.skip("server-LF compatibility fixture requires LF helper and registry files")
    known = set(listing.stdout.decode().splitlines())
    names = sorted(name for name in current if name in known)
    blobs = subprocess.run(
        [*prefix, "cat-file", "--batch"],
        cwd=root,
        input="".join(f"{base}:{name}\n" for name in names).encode(),
        capture_output=True,
        check=True,
    ).stdout
    position, previous = 0, {}
    for name in names:
        boundary = blobs.index(b"\n", position)
        header = blobs[position:boundary].split()
        assert header[1] == b"blob"
        size = int(header[2])
        start = boundary + 1
        previous[name] = hashlib.sha256(blobs[start : start + size]).hexdigest()
        position = start + size + 1
    registered = set(recorded["performance_repair"]["changes"]) | {compat.REGISTRY_SOURCE}
    assert {
        name for name in linux_current if previous.get(name) != linux_current[name]
    } <= registered
    evidence = compat.require_source_compatibility(previous, linux_current)
    assert evidence["patch_id"] == compat.PERFORMANCE_PATCH_ID
    assert evidence["resource_plan_semantics"] == (
        "historical measured selection retained; no new measurement claimed"
    )
