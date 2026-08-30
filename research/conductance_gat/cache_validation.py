"""Read-only cache validators used by the repository-level dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark_data import DATASETS as BENCHMARK_DATASETS
from .benchmark_data import load_dataset
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
) -> dict[str, Any]:
    """Validate every requested cache for one conductance registry entry."""

    del split_seeds
    paths: list[str] = []
    if dataset_id in BENCHMARK_DATASETS:
        _, manifest = load_dataset(dataset_id, data_root, allow_download=False)
        paths.append(
            str(
                data_root
                / "conductance_gat"
                / "matched_benchmark_v1"
                / dataset_id
                / "manifest.json"
            )
        )
        return {
            "paths": paths,
            "data_sha256": manifest["data_sha256"],
            "split_sha256": manifest["split_sha256"],
            "seed_policy": "official fixed data/splits",
        }
    if dataset_id in CORE_DATASETS:
        for seed in data_seeds:
            _, manifest_path, _ = validate_core_cache(data_root, seed=seed)
            paths.append(str(manifest_path))
    elif dataset_id in PUBLIC_DATASETS:
        marker, _ = validate_public_cache(data_root)
        paths.append(str(marker))
    else:
        raise ValueError(f"unsupported conductance cache dataset {dataset_id!r}")
    return {"paths": sorted(set(paths)), "requested_data_seeds": list(data_seeds)}


__all__ = ["validate_dataset_cache"]
