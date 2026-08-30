"""Read-only cache validators used by the repository-level dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from chartgat.cache import CacheCorruptError, CacheIncompleteError

from .paper_adapters import BRECAdapter, find_brec_v3
from .paper_data import sha256_file, validate_cycle_count_ood_cache


def _load_torch_cache(path: Path) -> Any:
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            return torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise CacheCorruptError(f"failed to parse PyG processed cache: {path}") from error


def _pyg_processed_count(path: Path) -> int:
    payload = _load_torch_cache(path)
    if not isinstance(payload, tuple) or len(payload) < 2 or not isinstance(payload[1], dict):
        raise CacheCorruptError(f"unsupported PyG processed-cache layout: {path}")
    slices = payload[1]
    counts: set[int] = set()
    for value in slices.values():
        try:
            count = int(len(value)) - 1
        except TypeError as error:
            raise CacheCorruptError(f"invalid PyG slice table: {path}") from error
        if count >= 0:
            counts.add(count)
    if len(counts) != 1:
        raise CacheCorruptError(f"inconsistent PyG split cardinality: {path}")
    return counts.pop()


def _validate_zinc(data_root: Path) -> dict[str, Any]:
    processed = data_root.expanduser().resolve() / "ZINC12K" / "subset" / "processed"
    paths = {split: processed / f"{split}.pt" for split in ("train", "val", "test")}
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        raise FileNotFoundError(f"Cycle PE ZINC processed cache is missing: {processed}")
    if not all(present.values()):
        missing = [name for name, exists in present.items() if not exists]
        raise CacheIncompleteError(f"Cycle PE ZINC processed splits are missing: {missing}")
    counts = {name: _pyg_processed_count(path) for name, path in paths.items()}
    expected = {"train": 10_000, "val": 1_000, "test": 1_000}
    if counts != expected:
        raise CacheCorruptError(f"Cycle PE ZINC official split cardinalities are invalid: {counts}")
    return {
        "paths": [str(path) for path in paths.values()],
        "split_sizes": counts,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def _validate_brec(data_root: Path) -> dict[str, Any]:
    path = find_brec_v3(data_root)
    expected_pairs = 400
    try:
        adapter = BRECAdapter(
            path,
            num_relabel=32,
            protocol="official",
        )
    except RuntimeError as error:
        raise CacheCorruptError(f"invalid BREC cache: {path}") from error
    if adapter.pair_count != expected_pairs:
        raise CacheCorruptError(
            f"BREC pair cardinality must be {expected_pairs}, got {adapter.pair_count}"
        )
    for pair_index in range(adapter.pair_count):
        try:
            adapter.load_pair(pair_index)
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise CacheCorruptError(f"BREC graph6 decode failed at pair {pair_index}") from error
    return {
        "paths": [str(path)],
        "pair_count": adapter.pair_count,
        "records": int(adapter.metadata["records"]),
        "sha256": adapter.metadata["sha256"],
    }


def validate_dataset_cache(
    dataset_id: str,
    data_root: Path,
    *,
    data_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Validate every requested cycle cache without generating or downloading data."""

    del split_seeds
    if dataset_id == "cyclecount_ood":
        paths = []
        for seed in data_seeds:
            bundle = validate_cycle_count_ood_cache(data_root, seed=seed)
            if bundle.cache_path is not None:
                paths.append(str(bundle.cache_path))
        return {"paths": paths, "requested_data_seeds": list(data_seeds)}
    if dataset_id == "brec_v3":
        return _validate_brec(data_root)
    if dataset_id == "zinc12k":
        return _validate_zinc(data_root)
    raise ValueError(f"unsupported cycle cache dataset {dataset_id!r}")


__all__ = ["validate_dataset_cache"]
