"""Read-only cache validators used by the repository-level dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paper_data import validate_prepared_cache

SUITES = {
    "cyclecount_ood_multichart": "core",
    "csl_chart_sanity": "csl",
    "zinc12k_multichart": "zinc",
}


def validate_dataset_cache(
    dataset_id: str,
    data_root: Path,
    *,
    data_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Validate every requested processed tree cache without writing."""

    try:
        suite = SUITES[dataset_id]
    except KeyError as error:
        raise ValueError(f"unsupported tree cache dataset {dataset_id!r}") from error
    seeds = split_seeds if suite == "csl" else data_seeds
    paths: list[str] = []
    for seed in seeds:
        prepared = validate_prepared_cache(suite, data_root, seed=seed)
        paths.extend((str(prepared.data_path), str(prepared.manifest_path)))
    return {
        "paths": sorted(set(paths)),
        "requested_axis": "split" if suite == "csl" else "data",
        "requested_seeds": list(seeds),
    }


__all__ = ["validate_dataset_cache"]
