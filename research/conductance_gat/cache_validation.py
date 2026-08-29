"""Read-only cache validators used by the repository-level dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paper_data import validate_core_cache
from .public_data import validate_public_cache

CORE_DATASETS = {
    "static_multigraph_identification",
    "topology_size_ood",
    "nonlinear_rollout",
    "identifiability_robustness",
}
PUBLIC_DATASETS = {"pascalvoc_sp", "ogbg_molhiv"}


def validate_dataset_cache(
    dataset_id: str,
    data_root: Path,
    *,
    data_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    tiny: bool,
) -> dict[str, Any]:
    """Validate every requested cache for one conductance registry entry."""

    del split_seeds
    paths: list[str] = []
    if dataset_id in CORE_DATASETS:
        for seed in data_seeds:
            _, manifest_path, _ = validate_core_cache(data_root, seed=seed, tiny=tiny)
            paths.append(str(manifest_path))
    elif dataset_id in PUBLIC_DATASETS:
        validation_seeds = data_seeds if tiny else data_seeds[:1]
        for seed in validation_seeds:
            marker, _ = validate_public_cache(data_root, seed=seed, tiny=tiny)
            paths.append(str(marker))
    else:
        raise ValueError(f"unsupported conductance cache dataset {dataset_id!r}")
    return {"paths": sorted(set(paths)), "requested_data_seeds": list(data_seeds)}


__all__ = ["validate_dataset_cache"]
