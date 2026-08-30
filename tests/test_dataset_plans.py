from __future__ import annotations

import copy
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chartgat.cache import CacheCorruptError, CacheIncompleteError, CacheWrongRequestError
from scripts import check_datasets

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_datasets.py"


def _check(
    profile: str,
    cwd: Path,
    *,
    as_json: bool = False,
    data_root: Path | None = None,
    require_cache: bool = False,
    seeds: tuple[int, ...] = (0,),
    split_seeds: tuple[int, ...] | None = None,
) -> SimpleNamespace:
    command = [str(CHECKER), "--profile", profile]
    if as_json:
        command.append("--json")
    if data_root is not None:
        command.extend(("--data-root", str(data_root)))
    if require_cache:
        command.append("--require-cache")
    command.extend(("--seeds", ",".join(str(seed) for seed in seeds)))
    if split_seeds is not None:
        command.extend(("--split-seeds", ",".join(str(seed) for seed in split_seeds)))
    stdout = StringIO()
    stderr = StringIO()
    with (
        patch.object(sys, "argv", command),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            return_code = check_datasets.main()
        except SystemExit as error:
            return_code = int(error.code)
    # Preserve the old subprocess-like test interface without starting a second
    # interpreter after PyTorch has initialized Windows worker threads.
    assert cwd.is_dir()
    return SimpleNamespace(
        returncode=return_code,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def test_removed_smoke_dataset_profile_is_rejected(tmp_path: Path) -> None:
    result = _check("smoke", tmp_path)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_paper_dataset_profile_matches_complete_core_code(tmp_path: Path) -> None:
    result = _check("paper", tmp_path, as_json=True)
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["ready"] is True
    assert payload["code_ready"] is True
    assert payload["paper_benchmark_suite_complete"] is True
    assert all(row["tier"] == "paper_core" for row in payload["rows"])
    assert all(row["status"] == "implemented" for row in payload["rows"])
    assert all(row["cache_status"] == "not_checked" for row in payload["rows"])


def test_complete_flag_is_derived_only_from_required_core_status() -> None:
    registry = check_datasets.load_registry("conductance_gat")
    inconsistent = copy.deepcopy(registry)
    inconsistent["paper_suite_complete"] = False
    errors = check_datasets.validate_registry("conductance_gat", inconsistent)
    assert any("paper_suite_complete must be true" in error for error in errors)

    optional_change = copy.deepcopy(registry)
    optional_entry = next(
        entry for entry in optional_change["datasets"] if entry["tier"] == "optional"
    )
    optional_entry["status"] = "blocked"
    errors = check_datasets.validate_registry("conductance_gat", optional_change)
    assert not any("paper_suite_complete" in error for error in errors)


def test_code_readiness_does_not_claim_cache_presence(tmp_path: Path) -> None:
    empty_data_root = tmp_path / "empty-data"
    empty_data_root.mkdir()
    result = _check("paper", tmp_path, as_json=True, data_root=empty_data_root)
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["code_ready"] is True
    assert payload["cache_checked"] is True
    assert payload["cached_data_ready"] is False
    assert any(row["cache_status"] == "missing" for row in payload["rows"])


def test_require_cache_controls_ready_and_exit_status(tmp_path: Path) -> None:
    without_root = _check("paper", tmp_path, as_json=True, require_cache=True)
    assert without_root.returncode == 2
    assert "--require-cache requires --data-root" in without_root.stderr

    empty_data_root = tmp_path / "empty-data"
    empty_data_root.mkdir()
    missing = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=empty_data_root,
        require_cache=True,
    )
    missing_payload = json.loads(missing.stdout)
    assert missing.returncode == 2
    assert missing_payload["code_ready"] is True
    assert missing_payload["cached_data_ready"] is False
    assert missing_payload["ready"] is False

    registries = {track: check_datasets.load_registry(track) for track in check_datasets.TRACKS}
    for registry in registries.values():
        for entry in registry["datasets"]:
            if entry["tier"] != "paper_core":
                continue
            pattern = entry.get("cache_glob")
            if pattern is None:
                continue
            fixture_path = empty_data_root / pattern.replace("*", "fixture")
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text("fixture\n", encoding="utf-8")
    present = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=empty_data_root,
        require_cache=True,
    )
    present_payload = json.loads(present.stdout)
    assert present.returncode == 2
    assert present_payload["cached_data_ready"] is False
    assert present_payload["ready"] is False
    assert any(
        row["cache_status"] in {"missing", "incomplete", "corrupt", "wrong_request"}
        for row in present_payload["rows"]
    )


def test_checker_routes_requested_axes_to_full_dataset_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_resolver = check_datasets._load_python_reference
    calls: list[tuple[str, Path, dict[str, object]]] = []

    def validator(dataset_id: str, data_root: Path, **kwargs: object) -> dict[str, object]:
        calls.append((dataset_id, data_root, kwargs))
        return {"validated": dataset_id}

    def resolve(reference: str) -> object:
        if reference.endswith(".validate_dataset_cache"):
            return validator
        return original_resolver(reference)

    monkeypatch.setattr(check_datasets, "_load_python_reference", resolve)
    result = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=tmp_path,
        require_cache=True,
        seeds=(11, 17),
        split_seeds=(13,),
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["cached_data_ready"] is True
    assert payload["requested_seed_axes"] == {"data": [11, 17], "split": [13]}
    assert "tiny" not in payload
    assert calls
    for _, root, kwargs in calls:
        assert root == tmp_path.resolve()
        assert kwargs == {"data_seeds": (11, 17), "split_seeds": (13,)}


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (FileNotFoundError("missing"), "missing"),
        (CacheIncompleteError("incomplete"), "incomplete"),
        (CacheWrongRequestError("wrong seed"), "wrong_request"),
        (CacheCorruptError("bad checksum"), "corrupt"),
        (RuntimeError("unexpected parser failure"), "corrupt"),
    ],
)
def test_read_only_validator_failures_are_not_reported_as_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, status: str
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(check_datasets, "_load_python_reference", lambda _reference: fail)
    entry = {"id": "requested", "cache_glob": "requested.json", "validator": "unit.validator"}
    result = check_datasets._validate_cache(entry, tmp_path, data_seeds=(3,))
    assert result["cache_status"] == status
    assert str(error) in result["cache_detail"]
    assert not list(tmp_path.iterdir())


def test_checker_has_no_dummy_cache_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(CHECKER), "--tiny"])
    with pytest.raises(SystemExit) as caught:
        check_datasets.main()
    assert caught.value.code == 2


def test_implemented_adapters_and_generated_sources_resolve() -> None:
    for track in check_datasets.TRACKS:
        registry = check_datasets.load_registry(track)
        errors = check_datasets.validate_registry(track, registry)
        assert not errors, "\n".join(errors)


def test_optional_is_a_tier_not_a_code_status() -> None:
    for track in check_datasets.TRACKS:
        registry = check_datasets.load_registry(track)
        for entry in registry["datasets"]:
            assert entry["tier"] in check_datasets.ALLOWED_TIERS
            assert entry["status"] in check_datasets.ALLOWED_STATUSES
            assert entry["data_policy"] in check_datasets.ALLOWED_DATA_POLICIES
            assert entry["status"] not in {"optional", "implemented_optional"}
