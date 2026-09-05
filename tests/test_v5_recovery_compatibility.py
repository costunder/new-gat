"""CPU-only recovery boundaries plus unstubbed, checked-in registry integration."""

from __future__ import annotations

import copy
import json

import pytest

from chartgat import resume_compat
from research.conductance_gat.tests.test_v5_contract import (
    test_report_is_partial_safe_then_requires_complete_pairs as _write_valid_pair,
)
from research.conductance_gat.tests.test_v5_p0_integrity import _identity
from research.conductance_gat.v5 import report, train


@pytest.fixture
def reviewed_sources(monkeypatch):
    """Boundary-only stub; the real-registry test below does not use this fixture."""
    before = {"research/conductance_gat/v5/train.py": "a" * 64}
    after = {"research/conductance_gat/v5/train.py": "b" * 64}

    def one_reviewed_transition(previous, current):
        return previous == current or (previous == before and current == after)

    monkeypatch.setattr(train, "snapshots_match", one_reviewed_transition)
    monkeypatch.setattr(report, "snapshots_match", one_reviewed_transition)
    return before, after


def _identities(sources):
    previous = _identity()
    previous["source_sha256"] = copy.deepcopy(sources[0])
    current = copy.deepcopy(previous)
    current["source_sha256"] = copy.deepcopy(sources[1])
    return previous, current


def _best(identity):
    return {
        "resume_identity": copy.deepcopy(identity),
        "resume_identity_sha256": train._canonical_sha256(identity),
        "epoch": 4,
        "validation": 0.75,
        "selection_role": "primary",
    }


def _validate_best(selected, current, **overrides):
    options = {
        "expected_identity": current,
        "expected_identity_sha256": train._canonical_sha256(current),
        "expected_epoch": 4,
        "expected_metric": 0.75,
    }
    options.update(overrides)
    train.validate_selected_checkpoint(selected, **options)


def _write_pair(tmp_path, sources):
    # Reuse the existing artifact fixture, not a fake _validate_child result.
    _write_valid_pair(tmp_path)
    jobs = []
    for condition, source in zip(("fixed_c", "shared_dynamic_c"), sources, strict=True):
        path = tmp_path / condition / "metrics.json"
        child = json.loads(path.read_text(encoding="utf-8"))
        child["source_sha256"] = copy.deepcopy(source)
        child["resume_identity"]["source_sha256"] = copy.deepcopy(source)
        child["resume_identity_sha256"] = train._canonical_sha256(child["resume_identity"])
        path.write_text(json.dumps(child), encoding="utf-8")
        jobs.append({"dataset": "cora", "condition": condition, "output_dir": str(path.parent)})
    return {
        "status": "passed",
        "config": {"datasets": ["cora"], "model_seed": 0, "batch_size": 1},
        "jobs": jobs,
    }


def test_old_last_and_best_accept_reviewed_source_change_without_rewriting(reviewed_sources):
    previous, current = _identities(reviewed_sources)
    selected = _best(previous)
    before = copy.deepcopy((previous, current, selected))
    train.validate_resume_identity(previous, current, train._canonical_sha256(previous))
    _validate_best(selected, current)
    assert (previous, current, selected) == before
    assert selected["resume_identity_sha256"] != train._canonical_sha256(current)


@pytest.mark.parametrize(
    ("field", "subkey", "replacement"),
    [
        ("configuration", "hidden_channels", 128),
        ("configuration", "workers", 4),
        ("configuration", "batch_size", 8),
        ("configuration", "model_seed", 1),
        ("dataset_protocol", "split", "different-split"),
        ("runtime_versions", "torch", "different-runtime"),
        ("cache_sha256", None, "e" * 64),
        ("dataset_protocol_sha256", None, "e" * 64),
        ("initial_state_sha256", None, "e" * 64),
        ("condition", None, "fixed_c"),
        ("schedule", None, []),
    ],
)
def test_reviewed_sources_never_relax_recipe_data_or_runtime_contracts(
    reviewed_sources, field, subkey, replacement
):
    previous, current = _identities(reviewed_sources)
    if subkey is None:
        current[field] = replacement
    else:
        current[field][subkey] = replacement
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train.validate_resume_identity(previous, current, train._canonical_sha256(previous))
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _validate_best(_best(previous), current)


def test_last_identity_hash_and_unreviewed_sources_remain_rejected(reviewed_sources):
    previous, current = _identities(reviewed_sources)
    with pytest.raises(ValueError, match="identity hash mismatch"):
        train.validate_resume_identity(previous, current, "0" * 64)
    current["source_sha256"]["research/conductance_gat/v5/train.py"] = "c" * 64
    with pytest.raises(ValueError, match="source_sha256"):
        train.validate_resume_identity(previous, current, train._canonical_sha256(previous))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("resume_identity_sha256", "0" * 64, "identity hash mismatch"),
        ("epoch", 3, "epoch"),
        ("validation", 0.74, "best_metric"),
        ("selection_role", "global_prediction_auxiliary", "selection role"),
    ],
)
def test_old_best_still_checks_its_own_hash_epoch_metric_and_role(
    reviewed_sources, field, replacement, message
):
    previous, current = _identities(reviewed_sources)
    selected = _best(previous)
    selected[field] = replacement
    with pytest.raises(ValueError, match=message):
        _validate_best(selected, current)


def test_best_requires_current_expected_identity_hash_too(reviewed_sources):
    previous, current = _identities(reviewed_sources)
    with pytest.raises(ValueError, match="selected last.pt identity"):
        _validate_best(
            _best(previous), current, expected_identity_sha256=train._canonical_sha256(previous)
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_completed_pair_accepts_both_orderings_without_rewriting_artifacts(
    tmp_path, reviewed_sources, reverse
):
    sources = tuple(reversed(reviewed_sources)) if reverse else reviewed_sources
    manifest = _write_pair(tmp_path, sources)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    comparison = report.build_comparison(tmp_path, manifest)
    assert comparison["status"] == "passed"
    assert comparison["contrasts"][0]["dynamic_minus_fixed"] == 0
    assert {row["condition"] for row in comparison["rows"]} == {"fixed_c", "shared_dynamic_c"}
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize(
    ("field", "subkey", "replacement"),
    [
        ("configuration", "dropout", 0.6),
        ("versions", "torch", "different-runtime"),
        ("protocol", "split", "different-split"),
        ("shared_initial_state_sha256", None, "e" * 64),
        ("source_sha256", "research/conductance_gat/v5/train.py", "c" * 64),
    ],
)
def test_mixed_pair_keeps_non_source_and_unreviewed_source_mismatch_guards(
    tmp_path, reviewed_sources, field, subkey, replacement
):
    manifest = _write_pair(tmp_path, reviewed_sources)
    path = tmp_path / "shared_dynamic_c" / "metrics.json"
    child = json.loads(path.read_text(encoding="utf-8"))
    if subkey is None:
        child[field] = replacement
    else:
        child[field][subkey] = replacement
    if field == "source_sha256":
        child["resume_identity"][field] = copy.deepcopy(child[field])
        child["resume_identity_sha256"] = train._canonical_sha256(child["resume_identity"])
    path.write_text(json.dumps(child), encoding="utf-8")
    with pytest.raises(report.ComparisonIntegrityError, match="fixed/dynamic .* mismatch"):
        report.build_comparison(tmp_path, manifest)


def test_real_registry_preserves_old_checkpoint_contracts_and_completed_pair(tmp_path):
    # No snapshots_match/require_source_compatibility mocks in this test.
    registry = json.loads(resume_compat.REGISTRY_PATH.read_text(encoding="utf-8"))
    current_sources = train.implementation_source_hashes()
    previous_sources = copy.deepcopy(current_sources)
    previous_sources.pop(resume_compat.REGISTRY_SOURCE)
    for name, change in registry["changes"].items():
        if name not in previous_sources:
            continue
        if change["before"] is None:
            previous_sources.pop(name)
        else:
            previous_sources[name] = change["before"]
    evidence = resume_compat.require_source_compatibility(previous_sources, current_sources)
    assert evidence["patch_id"] == "v5-rng-cycle-workers-v1"
    assert resume_compat.snapshots_match(previous_sources, current_sources)
    assert not resume_compat.snapshots_match(current_sources, previous_sources)
    previous, current = _identities((previous_sources, current_sources))
    train.validate_resume_identity(previous, current, train._canonical_sha256(previous))
    _validate_best(_best(previous), current)
    manifest = _write_pair(tmp_path, (previous_sources, current_sources))
    assert report.build_comparison(tmp_path, manifest)["status"] == "passed"
