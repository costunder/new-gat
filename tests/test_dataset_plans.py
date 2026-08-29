from __future__ import annotations

import copy
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    tiny: bool = False,
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
    if tiny:
        command.append("--tiny")
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


def test_smoke_dataset_profile_is_implemented(tmp_path: Path) -> None:
    result = _check("smoke", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "READY" in result.stdout
    assert "code=implemented" in result.stdout


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


def _prepare_valid_tiny_paper_caches(data_root: Path, seed: int) -> None:
    import torch

    from research.conductance_gat.paper_data import prepare_core_cache
    from research.conductance_gat.public_data import prepare_public_data
    from research.cycle_pe.paper_adapters import write_tiny_brec_fixture
    from research.cycle_pe.paper_data import load_or_generate_cycle_count_ood
    from research.tree_augmentation.paper_data import (
        GraphRecord,
        _cache_records,
        prepare_cyclecount_dataset,
    )

    prepare_core_cache(data_root, seed=seed, tiny=True)
    prepare_public_data(data_root, seed=seed, tiny=True)
    load_or_generate_cycle_count_ood(data_root, seed=seed, tiny=True)
    write_tiny_brec_fixture(data_root / "cycle_pe_fixtures" / "brec_v3_q32.npy", num_relabel=32)
    processed = data_root / "ZINC12K" / "subset" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for split, count in {"train": 32, "val": 8, "test": 8}.items():
        torch.save(({}, {"x": torch.arange(count + 1)}), processed / f"{split}.pt")

    prepare_cyclecount_dataset(data_root, seed=seed, tiny=True)

    def records(counts: dict[str, int], *, zinc: bool) -> tuple[GraphRecord, ...]:
        result = []
        for split, count in counts.items():
            for index in range(count):
                result.append(
                    GraphRecord(
                        graph_id=f"fixture-{'zinc' if zinc else 'csl'}-{split}-{index}",
                        family="fixture",
                        split=split,
                        num_nodes=3,
                        edges=((0, 1), (0, 2), (1, 2)),
                        target=(float(index % 10) if not zinc else float(index) / 10.0,),
                        task_type="regression" if zinc else "classification",
                        x=(0, 1, 2) if zinc else None,
                        edge_attr=(0, 0, 0) if zinc else None,
                    )
                )
        return tuple(result)

    tree_specs = {
        "csl": (
            {"train": 30, "validation": 10, "test": 10},
            tuple(f"class_{index}" for index in range(10)),
            "classification",
            "PyG:GNNBenchmarkDataset/CSL",
        ),
        "zinc": (
            {"train": 32, "validation": 12, "test": 12},
            ("constrained_logP",),
            "regression",
            "PyG:ZINC(subset=True)",
        ),
    }
    for suite, (counts, target_names, task_type, source) in tree_specs.items():
        cache_dir = data_root / f"{suite}_pyg_v2"
        _cache_records(
            suite=suite,
            records=records(counts, zinc=suite == "zinc"),
            data_path=cache_dir / f"seed-{seed}-tiny.json",
            manifest_path=cache_dir / f"seed-{seed}-tiny.manifest.json",
            target_names=target_names,
            task_type=task_type,
            source=source,
            seed=seed,
            tiny=True,
        )


def test_require_cache_runs_content_validators_for_requested_tiny_seed(tmp_path: Path) -> None:
    data_root = tmp_path / "valid-data"
    seed = 7
    _prepare_valid_tiny_paper_caches(data_root, seed)
    result = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=data_root,
        require_cache=True,
        seeds=(seed,),
        tiny=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["cached_data_ready"] is True
    assert payload["requested_seeds"] == [seed]
    assert payload["requested_seed_axes"] == {"data": [seed], "split": [seed]}
    assert payload["tiny"] is True
    assert all(row["cache_status"] == "valid" for row in payload["rows"])


def test_require_cache_rejects_corrupt_requested_cache(tmp_path: Path) -> None:
    data_root = tmp_path / "corrupt-data"
    seed = 11
    _prepare_valid_tiny_paper_caches(data_root, seed)
    cycle_cache = next((data_root / "cycle_count_ood").glob("*.json.gz"))
    cycle_cache.write_bytes(b"truncated")
    result = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=data_root,
        require_cache=True,
        seeds=(seed,),
        tiny=True,
    )
    payload = json.loads(result.stdout)
    cycle_row = next(row for row in payload["rows"] if row["id"] == "cyclecount_ood")
    assert result.returncode == 2
    assert cycle_row["cache_status"] == "corrupt"


def test_validator_distinguishes_wrong_request_and_incomplete_cache(tmp_path: Path) -> None:
    from research.conductance_gat.paper_data import prepare_core_cache

    entry = next(
        item
        for item in check_datasets.load_registry("conductance_gat")["datasets"]
        if item["id"] == "static_multigraph_identification"
    )

    wrong_root = tmp_path / "wrong"
    _, manifest_path, manifest = prepare_core_cache(wrong_root, seed=3, tiny=True)
    manifest["request"]["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong = check_datasets._validate_cache(entry, wrong_root, seeds=(3,), tiny=True)
    assert wrong["cache_status"] == "wrong_request"

    incomplete_root = tmp_path / "incomplete"
    _, incomplete_manifest, _ = prepare_core_cache(incomplete_root, seed=3, tiny=True)
    incomplete_manifest.with_name("core.pt").unlink()
    incomplete = check_datasets._validate_cache(entry, incomplete_root, seeds=(3,), tiny=True)
    assert incomplete["cache_status"] == "incomplete"


def test_checker_routes_tree_csl_cache_to_split_seed_axis(tmp_path: Path) -> None:
    data_root = tmp_path / "independent-seed-axes"
    _prepare_valid_tiny_paper_caches(data_root, 11)
    _prepare_valid_tiny_paper_caches(data_root, 13)

    result = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=data_root,
        require_cache=True,
        seeds=(11,),
        split_seeds=(13,),
        tiny=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["requested_seed_axes"] == {"data": [11], "split": [13]}
    csl = next(row for row in payload["rows"] if row["id"] == "csl_chart_sanity")
    tree_core = next(row for row in payload["rows"] if row["id"] == "cyclecount_ood_multichart")
    assert csl["cache_detail"]["requested_axis"] == "split"
    assert csl["cache_detail"]["requested_seeds"] == [13]
    assert tree_core["cache_detail"]["requested_axis"] == "data"
    assert tree_core["cache_detail"]["requested_seeds"] == [11]


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
