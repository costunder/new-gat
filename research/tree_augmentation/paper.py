"""Run the independent multi-chart paper protocol.

Examples
--------
python -m research.tree_augmentation.paper --suite core --device cuda --amp
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from chartgat.seeds import SeedAxes, resolve_seed_axes

from .paper_data import (
    OptionalDatasetError,
    PreparedDataset,
    prepare_cyclecount_dataset,
    prepare_optional_pyg_dataset,
)
from .paper_model import (
    FitResult,
    VariableBetaCycleEncoder,
    build_chart_views,
    evaluate_downstream_model,
    run_fixed_vs_multichart,
)

SUITES = ("core", "csl", "zinc")


def _sampler_protocol() -> dict[str, Any]:
    """Return the declared sampler exposure for the fixed/multi comparison."""

    return {
        "train_fixed": ["bfs_root_0"],
        "train_multi": ["bfs_random_root", "dfs_random_root"],
        "fresh_chart_seen_family": ["bfs_random_root"],
        "fresh_chart_unseen_family": ["wilson_ust"],
        "unseen_family_is_disjoint_from_all_training_families": True,
        "exact_tree_overlap_between_families_allowed": True,
        "wilson_draws_conditioned_on_bfs_outputs": False,
    }


def _protocol_name(suite: str) -> str:
    return (
        "cyclecount_graph_x_fresh_chart_family_2x2_v2"
        if suite == "core"
        else "public_pyg_fresh_chart_family_benchmark_v2"
    )


def _dataset_seed_policy(suite: str) -> dict[str, Any]:
    if suite == "core":
        return {
            "cache_identity_axis": "data",
            "record_generation_axis": "data",
            "split_assignment_axis": "data",
            "split_seed_used": False,
        }
    if suite == "csl":
        return {
            "cache_identity_axis": "split",
            "record_source": "fixed_public_dataset",
            "split_assignment_axis": "split",
            "data_seed_used": False,
        }
    return {
        "cache_identity_axis": "data",
        "record_source": "fixed_public_dataset",
        "split_assignment_axis": "official",
        "data_seed_changes_records": False,
        "split_seed_changes_records": False,
    }


def resolve_device(requested: str) -> torch.device:
    """Resolve CPU/CUDA requests and fail before a long experiment starts."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {normalized!r} was requested, but this PyTorch build cannot use CUDA"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        device = torch.device("cuda", index)
    return device


def _seed_runtime(seed: int, device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"output directory is not empty; refusing to overwrite: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_settings() -> tuple[dict[str, Any], Path]:
    config_path = Path(__file__).with_name("config.yaml").resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("paper"), dict):
        raise ValueError("config.yaml must contain a paper mapping")
    paper = dict(config["paper"])
    profile = paper.get("full")
    if not isinstance(profile, dict):
        raise ValueError("paper.full must be a mapping")
    merged = {key: value for key, value in paper.items() if key != "full"}
    merged.update(profile)
    return merged, config_path


def _apply_setting_overrides(
    settings: dict[str, Any],
    *,
    hidden_dim: int | None,
    message_layers: int | None = None,
    optimizer_updates: int | None,
    train_charts_per_graph: int | None,
    eval_charts_per_graph: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply explicit scaling overrides while preserving config.yaml defaults."""

    overrides = {
        key: int(value)
        for key, value in {
            "hidden_dim": hidden_dim,
            "message_layers": message_layers,
            "optimizer_updates": optimizer_updates,
            "train_charts_per_graph": train_charts_per_graph,
            "eval_charts_per_graph": eval_charts_per_graph,
        }.items()
        if value is not None
    }
    effective = dict(settings)
    effective.update(overrides)
    for key in (
        "hidden_dim",
        "message_layers",
        "optimizer_updates",
        "train_charts_per_graph",
        "eval_charts_per_graph",
    ):
        if int(effective[key]) < 1:
            raise ValueError(f"{key.replace('_', ' ')} must be positive")
    return effective, overrides


def _prepare_dataset(
    suite: str,
    data_root: Path,
    *,
    seed_axes: SeedAxes,
    allow_download: bool,
) -> PreparedDataset:
    if suite == "core":
        return prepare_cyclecount_dataset(data_root, seed=seed_axes.data)
    cache_seed = seed_axes.split if suite == "csl" else seed_axes.data
    return prepare_optional_pyg_dataset(
        suite,
        data_root,
        seed=cache_seed,
        allow_download=allow_download,
    )


def _runtime_metadata(
    *,
    device: torch.device,
    amp_requested: bool,
    pin_memory: bool,
    non_blocking: bool,
    batch_size: int,
    workers: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    cuda = device.type == "cuda"
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "amp_requested": amp_requested,
        "amp_effective": bool(amp_requested and cuda),
        "pin_memory": bool(pin_memory and cuda),
        "non_blocking": bool(non_blocking and cuda),
        "batch_size": batch_size,
        "workers": workers,
        "elapsed_seconds": elapsed_seconds,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "peak_gpu_allocated_bytes": 0,
        "peak_gpu_reserved_bytes": 0,
    }
    if cuda:
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return metadata


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _save_models(
    output_dir: Path,
    models: dict[str, Any],
    *,
    settings: dict[str, Any],
    task_type: str,
    output_dim: int,
    suite: str,
    dataset_data_sha256: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, fitted in models.items():
        path = output_dir / f"{name}_model.pt"
        torch.save(
            {
                "checkpoint_schema_version": 1,
                "model_name": name,
                "state_dict": _cpu_state_dict(fitted.model),
                "target_mean": torch.as_tensor(fitted.target_mean),
                "target_scale": torch.as_tensor(fitted.target_scale),
                "settings": settings,
                "task_type": task_type,
                "output_dim": output_dim,
                "suite": suite,
                "dataset_data_sha256": dataset_data_sha256,
            },
            path,
        )
        paths[name] = str(path)
    return paths


def _load_selected_model(
    path: Path,
    *,
    expected_model_name: str,
    expected_task_type: str,
    expected_output_dim: int,
    expected_suite: str,
    expected_dataset_data_sha256: str,
    device: torch.device,
) -> tuple[FitResult, dict[str, Any]]:
    """Load one selected scaling checkpoint without permitting code execution."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"selected checkpoint is missing: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema_version") != 1:
        raise ValueError(f"unsupported selected checkpoint schema: {resolved}")
    if payload.get("model_name") != expected_model_name:
        raise ValueError(f"selected checkpoint arm mismatch: {resolved}")
    if payload.get("task_type") != expected_task_type:
        raise ValueError(f"selected checkpoint task mismatch: {resolved}")
    if payload.get("output_dim") != expected_output_dim:
        raise ValueError(f"selected checkpoint output dimension mismatch: {resolved}")
    if payload.get("suite") != expected_suite:
        raise ValueError(f"selected checkpoint suite mismatch: {resolved}")
    if payload.get("dataset_data_sha256") != expected_dataset_data_sha256:
        raise ValueError(f"selected checkpoint dataset mismatch: {resolved}")
    settings = payload.get("settings")
    state_dict = payload.get("state_dict")
    target_mean = payload.get("target_mean")
    target_scale = payload.get("target_scale")
    if (
        not isinstance(settings, dict)
        or not isinstance(state_dict, dict)
        or not isinstance(target_mean, torch.Tensor)
        or not isinstance(target_scale, torch.Tensor)
    ):
        raise ValueError(f"selected checkpoint payload is incomplete: {resolved}")
    hidden_dim = settings.get("hidden_dim")
    message_layers = settings.get("message_layers")
    if (
        isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim < 1
        or isinstance(message_layers, bool)
        or not isinstance(message_layers, int)
        or message_layers < 1
    ):
        raise ValueError(f"selected checkpoint architecture is invalid: {resolved}")
    model = VariableBetaCycleEncoder(
        hidden_dim=hidden_dim,
        output_dim=expected_output_dim,
        message_layers=message_layers,
    )
    model.load_state_dict(state_dict, strict=True)
    fitted = FitResult(
        model=model.to(device),
        target_mean=target_mean.detach().cpu().numpy().astype(np.float64, copy=False),
        target_scale=target_scale.detach().cpu().numpy().astype(np.float64, copy=False),
        history=(),
    )
    return fitted, settings


def _split(records: tuple[Any, ...], name: str) -> list[Any]:
    return [record for record in records if record.split == name]


def _chart_keys_by_graph(views: list[Any]) -> dict[str, set[tuple[int, ...]]]:
    grouped: dict[str, set[tuple[int, ...]]] = {}
    for view in views:
        grouped.setdefault(view.graph_id, set()).add(view.tree_key)
    return grouped


def _chart_overlap_stats(left: list[Any], right: list[Any]) -> dict[str, int]:
    left_keys = _chart_keys_by_graph(left)
    right_keys = _chart_keys_by_graph(right)
    intersections = [
        left_keys[graph_id] & right_keys.get(graph_id, set()) for graph_id in left_keys
    ]
    return {
        "graphs_with_exact_tree_overlap": sum(bool(overlap) for overlap in intersections),
        "unique_graph_tree_overlaps": sum(len(overlap) for overlap in intersections),
    }


def _fresh_axis_overlap_stats(evaluation: dict[str, list[Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    marker = "_fresh_chart_seen_family"
    for seen_name, seen_views in evaluation.items():
        if not seen_name.endswith(marker):
            continue
        unseen_name = seen_name.replace(marker, "_fresh_chart_unseen_family")
        result[seen_name.removesuffix(marker)] = _chart_overlap_stats(
            seen_views, evaluation[unseen_name]
        )
    return result


def _training_views(
    dataset: PreparedDataset,
    *,
    settings: dict[str, Any],
    chart_seed: int,
) -> tuple[list[Any], list[Any]]:
    train_records = _split(dataset.records, "train")
    if not train_records:
        raise ValueError(f"suite {dataset.suite} contains no training graphs")
    train_charts = int(settings["train_charts_per_graph"])
    fixed_train = build_chart_views(
        train_records,
        chart_status="train_fixed_bfs_family",
        count=1,
        methods=("bfs",),
        roots=(0,),
        seed=chart_seed + 1_000,
    )
    multi_train = build_chart_views(
        train_records,
        chart_status="train_multi_bfs_dfs_families",
        count=train_charts,
        methods=("bfs", "dfs"),
        seed=chart_seed + 2_000,
    )
    return fixed_train, multi_train


def _evaluation_views(
    dataset: PreparedDataset,
    *,
    settings: dict[str, Any],
    chart_seed: int,
    evaluation_scope: str,
) -> dict[str, list[Any]]:
    if evaluation_scope not in {"validation", "test"}:
        raise ValueError("evaluation scope must be validation or test")
    eval_charts = int(settings["eval_charts_per_graph"])
    if evaluation_scope == "validation":
        validation_records = _split(dataset.records, "validation")
        if not validation_records:
            raise ValueError(f"suite {dataset.suite} contains no validation graphs")
        validation_seen = build_chart_views(
            validation_records,
            chart_status="fresh_chart_seen_family",
            count=eval_charts,
            methods=("bfs",),
            seed=chart_seed + 10_000,
        )
        validation_unseen = build_chart_views(
            validation_records,
            chart_status="fresh_chart_unseen_family",
            count=eval_charts,
            methods=("wilson_ust",),
            seed=chart_seed + 20_000,
        )
        return {
            "validation_graph_fresh_chart_seen_family": validation_seen,
            "validation_graph_fresh_chart_unseen_family": validation_unseen,
        }
    if dataset.suite == "core":
        id_records = _split(dataset.records, "id_test")
        ood_records = _split(dataset.records, "ood_test")
        if not id_records or not ood_records:
            raise ValueError("core suite requires non-empty id_test and ood_test graph splits")
        id_seen = build_chart_views(
            id_records,
            chart_status="fresh_chart_seen_family",
            count=eval_charts,
            methods=("bfs",),
            seed=chart_seed + 10_000,
        )
        id_unseen = build_chart_views(
            id_records,
            chart_status="fresh_chart_unseen_family",
            count=eval_charts,
            methods=("wilson_ust",),
            seed=chart_seed + 20_000,
        )
        ood_seen = build_chart_views(
            ood_records,
            chart_status="fresh_chart_seen_family",
            count=eval_charts,
            methods=("bfs",),
            seed=chart_seed + 30_000,
        )
        ood_unseen = build_chart_views(
            ood_records,
            chart_status="fresh_chart_unseen_family",
            count=eval_charts,
            methods=("wilson_ust",),
            seed=chart_seed + 40_000,
        )
        return {
            "id_graph_fresh_chart_seen_family": id_seen,
            "id_graph_fresh_chart_unseen_family": id_unseen,
            "ood_graph_fresh_chart_seen_family": ood_seen,
            "ood_graph_fresh_chart_unseen_family": ood_unseen,
        }
    else:
        test_records = _split(dataset.records, "test")
        if not test_records:
            raise ValueError(f"suite {dataset.suite} contains no test graphs")
        test_seen = build_chart_views(
            test_records,
            chart_status="fresh_chart_seen_family",
            count=eval_charts,
            methods=("bfs",),
            seed=chart_seed + 10_000,
        )
        test_unseen = build_chart_views(
            test_records,
            chart_status="fresh_chart_unseen_family",
            count=eval_charts,
            methods=("wilson_ust",),
            seed=chart_seed + 20_000,
        )
        return {
            "test_graph_fresh_chart_seen_family": test_seen,
            "test_graph_fresh_chart_unseen_family": test_unseen,
        }


def _protocol_views(
    dataset: PreparedDataset,
    *,
    settings: dict[str, Any],
    chart_seed: int,
    evaluation_scope: str = "test",
) -> tuple[list[Any], list[Any], dict[str, list[Any]]]:
    fixed_train, multi_train = _training_views(
        dataset,
        settings=settings,
        chart_seed=chart_seed,
    )
    evaluation = _evaluation_views(
        dataset,
        settings=settings,
        chart_seed=chart_seed,
        evaluation_scope=evaluation_scope,
    )
    return fixed_train, multi_train, evaluation


def _output_dim(dataset: PreparedDataset) -> int:
    output_dim = len(dataset.target_names)
    if output_dim < 1:
        raise ValueError("dataset target_names metadata must declare at least one output")
    return output_dim


def _dataset_cache_integrity(dataset: PreparedDataset) -> dict[str, Any]:
    return {
        "full_cache_loaded": True,
        "all_declared_splits_validated": True,
        "loaded_and_validated_splits": sorted({record.split for record in dataset.records}),
    }


def _model_split_usage(
    dataset: PreparedDataset,
    *,
    evaluation_scope: str,
    prepare_only: bool,
) -> dict[str, Any]:
    if prepare_only:
        fit_splits: list[str] = []
        evaluation_splits: list[str] = []
        selection_splits: list[str] = []
    elif evaluation_scope == "validation":
        fit_splits = ["train"]
        evaluation_splits = ["validation"]
        selection_splits = ["validation"]
    else:
        fit_splits = [] if evaluation_scope == "selected_test" else ["train"]
        evaluation_splits = ["id_test", "ood_test"] if dataset.suite == "core" else ["test"]
        selection_splits = []
    return {
        "fit_splits": fit_splits,
        "evaluation_splits": evaluation_splits,
        "selection_splits": selection_splits,
        "test_evaluated": evaluation_scope in {"test", "selected_test"} and not prepare_only,
        "test_used_for_selection": False,
    }


def _evaluate_fitted_models(
    models: dict[str, FitResult],
    evaluation: dict[str, list[Any]],
    *,
    task_type: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
    checkpoint_settings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate already-selected checkpoints without an optimizer or retraining."""

    metrics: dict[str, Any] = {}
    for model_name, fitted in models.items():
        settings = checkpoint_settings[model_name]
        metrics[model_name] = {
            "optimizer_updates": int(settings["optimizer_updates"]),
            "training_performed": False,
            "history": [],
            "quadrants": {},
        }
        for quadrant, views in evaluation.items():
            values = evaluate_downstream_model(
                fitted,
                views,
                task_type=task_type,
                batch_size=batch_size,
                device=device,
                amp=amp,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                workers=workers,
            )
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"non-finite metric in {model_name}/{quadrant}")
            metrics[model_name]["quadrants"][quadrant] = values
    return metrics


def _headline_comparison(metrics: dict[str, Any], *, suite: str) -> dict[str, Any]:
    if suite == "core":
        eligibility_reason = "full independent CycleCount-style core protocol"
    elif suite == "zinc":
        eligibility_reason = "official ZINC-12K split with atom/bond chemistry and held-out charts"
    else:
        eligibility_reason = "full CSL controlled chart-robustness protocol"
    comparison: dict[str, Any] = {
        "paper_headline_eligible": True,
        "paper_headline_eligibility_reason": eligibility_reason,
        "projector_target_used": False,
        "fixed_and_multi_optimizer_updates_matched": True,
    }
    if suite == "core":
        improvements = {}
        for quadrant, fixed in metrics["fixed_bfs"]["quadrants"].items():
            multi = metrics["multi_chart"]["quadrants"][quadrant]
            improvements[quadrant] = {
                "mae_improvement_fixed_minus_multi": fixed["mae"] - multi["mae"],
                "worst_chart_mae_improvement_fixed_minus_multi": (
                    fixed["worst_chart_mae"] - multi["worst_chart_mae"]
                ),
                "chart_std_improvement_fixed_minus_multi": (
                    fixed["chart_prediction_std"] - multi["chart_prediction_std"]
                ),
            }
        comparison["quadrant_improvements"] = improvements
    return comparison


def _view_stats(views: list[Any]) -> dict[str, Any]:
    sampler_counts: dict[str, int] = {}
    chart_status_counts: dict[str, int] = {}
    for view in views:
        method = str(view.chart_name).split(":", 1)[0]
        sampler_counts[method] = sampler_counts.get(method, 0) + 1
        status = str(view.chart_status)
        chart_status_counts[status] = chart_status_counts.get(status, 0) + 1
    return {
        "views": len(views),
        "graphs": len({view.graph_id for view in views}),
        "unique_graph_tree_pairs": len({(view.graph_id, view.tree_key) for view in views}),
        "sampler_counts": sampler_counts,
        "chart_status_counts": chart_status_counts,
    }


def run_suite(
    suite: str,
    *,
    data_root: Path,
    output_dir: Path,
    requested_device: str,
    seed: int,
    data_seed: int | None = None,
    split_seed: int | None = None,
    chart_seed: int | None = None,
    model_seed: int | None = None,
    prepare_only: bool,
    amp_override: bool | None,
    batch_size_override: int | None,
    pin_memory_override: bool | None,
    non_blocking_override: bool | None,
    workers: int = 0,
    allow_download: bool = False,
    hidden_dim_override: int | None = None,
    message_layers_override: int | None = None,
    optimizer_updates_override: int | None = None,
    train_charts_per_graph_override: int | None = None,
    eval_charts_per_graph_override: int | None = None,
    evaluation_scope: str = "test",
    selected_checkpoints: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Prepare and optionally train exactly one independent suite."""

    seed_axes = resolve_seed_axes(
        seed,
        data_seed=data_seed,
        split_seed=split_seed,
        chart_seed=chart_seed,
        model_seed=model_seed,
    )
    settings, config_path = _load_settings()
    settings, settings_overrides = _apply_setting_overrides(
        settings,
        hidden_dim=hidden_dim_override,
        message_layers=message_layers_override,
        optimizer_updates=optimizer_updates_override,
        train_charts_per_graph=train_charts_per_graph_override,
        eval_charts_per_graph=eval_charts_per_graph_override,
    )
    output = _prepare_output_dir(output_dir)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "preparing",
        "suite": suite,
        "protocol": _protocol_name(suite),
        "seed_axes": seed_axes.to_manifest(),
        "dataset_seed_policy": _dataset_seed_policy(suite),
        "prepare_only": prepare_only,
        "evaluation_scope": evaluation_scope,
        "training_performed": evaluation_scope != "selected_test" and not prepare_only,
        "dataset_cache_integrity": {
            "full_cache_loaded": False,
            "all_declared_splits_validated": False,
            "loaded_and_validated_splits": [],
        },
        "allow_download": allow_download,
        "workers": workers,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "effective_settings": settings,
        "settings_overrides": settings_overrides,
        "source_files": {
            path.name: _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("paper_data.py").resolve(),
                Path(__file__).with_name("paper_model.py").resolve(),
                Path(__file__).with_name("augmentation.py").resolve(),
                Path(__file__).with_name("datasets.yaml").resolve(),
            )
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "networkx", "torch", "PyYAML")
        },
        "output_dir": str(output),
        "sampler_protocol": _sampler_protocol(),
        "orientation_gauge_policy": (
            "sign-even fundamental-cycle coordinates; exact only for the same physical tree"
        ),
    }
    _write_json(manifest_path, manifest)
    try:
        if evaluation_scope not in {"test", "validation", "selected_test"}:
            raise ValueError("evaluation_scope must be test, validation, or selected_test")
        if prepare_only and evaluation_scope != "test":
            raise ValueError("prepare-only cannot be combined with a scaling evaluation scope")
        if evaluation_scope == "selected_test":
            if (
                selected_checkpoints is None
                or set(selected_checkpoints)
                != {
                    "fixed_bfs",
                    "multi_chart",
                }
                or any(not isinstance(path, Path) for path in selected_checkpoints.values())
            ):
                raise ValueError("selected_test requires fixed_bfs and multi_chart checkpoints")
            if any(
                value is not None
                for value in (
                    hidden_dim_override,
                    message_layers_override,
                    optimizer_updates_override,
                    train_charts_per_graph_override,
                )
            ):
                raise ValueError(
                    "selected_test architecture/training settings come from checkpoints"
                )
        elif selected_checkpoints is not None:
            raise ValueError("selected checkpoints are only valid for selected_test")
        if workers < 0:
            raise ValueError("workers must be non-negative")
        dataset = _prepare_dataset(
            suite,
            data_root,
            seed_axes=seed_axes,
            allow_download=allow_download,
        )
        dataset_cache_integrity = _dataset_cache_integrity(dataset)
        model_split_usage = _model_split_usage(
            dataset,
            evaluation_scope=evaluation_scope,
            prepare_only=prepare_only,
        )
        manifest["dataset_cache_integrity"] = dataset_cache_integrity
        manifest["model_split_usage"] = model_split_usage
        manifest["dataset"] = {
            "data_path": str(dataset.data_path),
            "manifest_path": str(dataset.manifest_path),
            "manifest_sha256": _sha256(dataset.manifest_path),
            "data_sha256": dataset.data_sha256,
            "num_graphs": len(dataset.records),
            "task_type": dataset.task_type,
            "target_names": list(dataset.target_names),
        }
        if prepare_only:
            manifest["status"] = "prepared"
            manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
            _write_json(manifest_path, manifest)
            return manifest

        device = resolve_device(requested_device)
        _seed_runtime(seed_axes.model, device)
        amp = bool(settings.get("amp", True)) if amp_override is None else amp_override
        batch_size = (
            int(settings["batch_size"]) if batch_size_override is None else batch_size_override
        )
        pin_memory = (
            bool(settings.get("pin_memory", True))
            if pin_memory_override is None
            else pin_memory_override
        )
        non_blocking = (
            bool(settings.get("non_blocking", True))
            if non_blocking_override is None
            else non_blocking_override
        )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if evaluation_scope == "selected_test":
            assert selected_checkpoints is not None
            evaluation = _evaluation_views(
                dataset,
                settings=settings,
                chart_seed=seed_axes.chart,
                evaluation_scope="test",
            )
            output_dim = _output_dim(dataset)
            checkpoint_paths = {
                name: path.expanduser().resolve() for name, path in selected_checkpoints.items()
            }
            checkpoint_hashes = {name: _sha256(path) for name, path in checkpoint_paths.items()}
            selected_models: dict[str, FitResult] = {}
            selected_settings: dict[str, dict[str, Any]] = {}
            for name in ("fixed_bfs", "multi_chart"):
                fitted, checkpoint_settings = _load_selected_model(
                    checkpoint_paths[name],
                    expected_model_name=name,
                    expected_task_type=dataset.task_type,
                    expected_output_dim=output_dim,
                    expected_suite=suite,
                    expected_dataset_data_sha256=dataset.data_sha256,
                    device=device,
                )
                if checkpoint_settings.get("seed_axes") != seed_axes.to_manifest():
                    raise ValueError(f"selected {name} checkpoint seed axes mismatch")
                if checkpoint_settings.get("eval_charts_per_graph") != settings.get(
                    "eval_charts_per_graph"
                ):
                    raise ValueError(f"selected {name} checkpoint evaluation chart count mismatch")
                selected_models[name] = fitted
                selected_settings[name] = checkpoint_settings
            started = time.perf_counter()
            metrics = _evaluate_fitted_models(
                selected_models,
                evaluation,
                task_type=dataset.task_type,
                batch_size=batch_size,
                device=device,
                amp=amp,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                workers=workers,
                checkpoint_settings=selected_settings,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if {
                name: _sha256(path) for name, path in checkpoint_paths.items()
            } != checkpoint_hashes:
                raise RuntimeError("a selected checkpoint changed during test evaluation")
            parameter_counts = {
                name: {
                    "total": sum(parameter.numel() for parameter in fitted.model.parameters()),
                    "trainable": sum(
                        parameter.numel()
                        for parameter in fitted.model.parameters()
                        if parameter.requires_grad
                    ),
                }
                for name, fitted in selected_models.items()
            }
            for name, counts in parameter_counts.items():
                metrics[name]["parameters"] = counts
            runtime = _runtime_metadata(
                device=device,
                amp_requested=amp,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                batch_size=batch_size,
                workers=workers,
                elapsed_seconds=elapsed,
            )
            split_counts: dict[str, int] = {}
            for record in dataset.records:
                split_counts[record.split] = split_counts.get(record.split, 0) + 1
            summary = {
                "track": "tree_augmentation_scaling_selected_test",
                "suite": suite,
                "seed_axes": seed_axes.to_manifest(),
                "protocol": _protocol_name(suite),
                "evaluation_scope": "selected_test",
                "training_performed": False,
                "test_metrics_emitted": True,
                "dataset_cache_integrity": dataset_cache_integrity,
                "model_split_usage": model_split_usage,
                "test_evaluations_per_selected_checkpoint": 1,
                "graph_split_before_chart_sampling": True,
                "split_counts": split_counts,
                "view_counts": {
                    "evaluation": {name: _view_stats(views) for name, views in evaluation.items()}
                },
                "settings": {
                    "eval_charts_per_graph": int(settings["eval_charts_per_graph"]),
                    "batch_size": batch_size,
                    "amp": amp,
                    "pin_memory": pin_memory,
                    "non_blocking": non_blocking,
                    "workers": workers,
                    "seed_axes": seed_axes.to_manifest(),
                },
                "selected_checkpoint_settings": selected_settings,
                "selected_checkpoints": {
                    name: {"path": str(path), "sha256": checkpoint_hashes[name]}
                    for name, path in checkpoint_paths.items()
                },
                "runtime": runtime,
                "parameter_counts": parameter_counts,
                "models": metrics,
                "comparison": _headline_comparison(metrics, suite=suite),
            }
            summary_path = output / "summary.json"
            _write_json(summary_path, summary)
            manifest.update(
                {
                    "status": "passed",
                    "device": str(device),
                    "runtime": runtime,
                    "selected_checkpoint_inputs": summary["selected_checkpoints"],
                    "artifacts": {
                        summary_path.name: {
                            "path": str(summary_path),
                            "sha256": _sha256(summary_path),
                        }
                    },
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            _write_json(manifest_path, manifest)
            return summary
        fixed_train, multi_train, evaluation = _protocol_views(
            dataset,
            settings=settings,
            chart_seed=seed_axes.chart,
            evaluation_scope=evaluation_scope,
        )
        started = time.perf_counter()
        metrics, models = run_fixed_vs_multichart(
            fixed_train_views=fixed_train,
            multi_train_views=multi_train,
            evaluation_views=evaluation,
            task_type=dataset.task_type,
            output_dim=_output_dim(dataset),
            hidden_dim=int(settings["hidden_dim"]),
            message_layers=int(settings["message_layers"]),
            updates=int(settings["optimizer_updates"]),
            batch_size=batch_size,
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            device=device,
            seed=seed_axes.model,
            amp=amp,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            workers=workers,
        )
        parameter_counts = {
            name: {
                "total": sum(parameter.numel() for parameter in fitted.model.parameters()),
                "trainable": sum(
                    parameter.numel()
                    for parameter in fitted.model.parameters()
                    if parameter.requires_grad
                ),
            }
            for name, fitted in models.items()
        }
        for name, counts in parameter_counts.items():
            metrics[name]["parameters"] = counts
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        runtime = _runtime_metadata(
            device=device,
            amp_requested=amp,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            batch_size=batch_size,
            workers=workers,
            elapsed_seconds=elapsed,
        )
        model_paths = _save_models(
            output,
            models,
            settings=settings
            | {
                "batch_size": batch_size,
                "amp": amp,
                "pin_memory": pin_memory,
                "non_blocking": non_blocking,
                "workers": workers,
                "seed_axes": seed_axes.to_manifest(),
            },
            task_type=dataset.task_type,
            output_dim=_output_dim(dataset),
            suite=suite,
            dataset_data_sha256=dataset.data_sha256,
        )
        split_counts: dict[str, int] = {}
        for record in dataset.records:
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
        summary = {
            "track": "tree_augmentation_only",
            "suite": suite,
            "seed_axes": seed_axes.to_manifest(),
            "dataset_seed_policy": _dataset_seed_policy(suite),
            "protocol": _protocol_name(suite),
            "evaluation_scope": evaluation_scope,
            "training_performed": True,
            "test_metrics_emitted": evaluation_scope == "test",
            "dataset_cache_integrity": dataset_cache_integrity,
            "model_split_usage": model_split_usage,
            "downstream_target": list(dataset.target_names),
            "target_is_independent_of_chart": True,
            "samplers": {
                "uniform": "wilson_ust",
                "traversal": ["bfs_random_root", "dfs_random_root"],
                "legacy_nonuniform_baseline": "random_priority_kruskal",
            },
            "sampler_protocol": _sampler_protocol(),
            "orientation_gauge_policy": {
                "coordinate_features": ["abs_f", "square_f", "normalized_cycle_support"],
                "same_physical_tree_invariant": True,
                "different_tree_chart_invariant": False,
                "label_dependent_tree_selection_is_chart_shift": True,
            },
            "graph_split_before_chart_sampling": True,
            "fresh_axis_exact_tree_overlap": _fresh_axis_overlap_stats(evaluation),
            "split_counts": split_counts,
            "view_counts": {
                "fixed_train": _view_stats(fixed_train),
                "multi_train": _view_stats(multi_train),
                "evaluation": {name: _view_stats(views) for name, views in evaluation.items()},
            },
            "settings": settings
            | {
                "batch_size": batch_size,
                "amp": amp,
                "pin_memory": pin_memory,
                "non_blocking": non_blocking,
                "workers": workers,
                "seed_axes": seed_axes.to_manifest(),
            },
            "runtime": runtime,
            "parameter_counts": parameter_counts,
            "models": metrics,
            "comparison": _headline_comparison(metrics, suite=suite),
            "checkpoints": model_paths,
        }
        if evaluation_scope == "validation":
            summary["comparison"].update(
                {
                    "paper_headline_eligible": False,
                    "paper_headline_eligibility_reason": (
                        "validation-only scaling candidate; the full cache was integrity-"
                        "validated, but the test split was not evaluated and no test metric "
                        "entered profile selection"
                    ),
                }
            )
        summary_path = output / "summary.json"
        _write_json(summary_path, summary)
        artifacts = [summary_path, *(Path(path) for path in model_paths.values())]
        manifest.update(
            {
                "status": "passed",
                "device": str(device),
                "runtime": runtime,
                "artifacts": {
                    path.name: {"path": str(path), "sha256": _sha256(path)} for path in artifacts
                },
                "finished_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(manifest_path, manifest)
        return summary
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(manifest_path, manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="core")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).with_name("data"),
        help="deterministic processed-cache root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "paper",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="dataset-generation/cache axis; defaults to --seed",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="split-assignment axis; defaults to the resolved data seed",
    )
    parser.add_argument(
        "--chart-seed",
        type=int,
        default=None,
        help="spanning-tree chart sampling axis; defaults to the resolved data seed",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="model initialization/minibatch axis; defaults to --seed",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="override paper.full.hidden_dim; omitted uses config.yaml",
    )
    parser.add_argument(
        "--message-layers",
        type=int,
        default=None,
        help="override paper.full.message_layers; omitted uses config.yaml",
    )
    parser.add_argument(
        "--optimizer-updates",
        type=int,
        default=None,
        help="override paper.full.optimizer_updates; omitted uses config.yaml",
    )
    parser.add_argument(
        "--train-charts-per-graph",
        type=int,
        default=None,
        help="override training chart count for the multi-chart arm",
    )
    parser.add_argument(
        "--eval-charts-per-graph",
        type=int,
        default=None,
        help="override fresh-chart count per evaluation family",
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=("test", "validation", "selected_test"),
        default="test",
        help="standard test run, validation-only scaling candidate, or selected test-only run",
    )
    parser.add_argument(
        "--fixed-checkpoint",
        type=Path,
        default=None,
        help="selected fixed_bfs checkpoint; only valid with selected_test",
    )
    parser.add_argument(
        "--multi-checkpoint",
        type=Path,
        default=None,
        help="selected multi_chart checkpoint; only valid with selected_test",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker processes (0 loads in the main process)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow optional CSL/ZINC adapters to access public download endpoints",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--non-blocking", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _run_from_args(args: argparse.Namespace, suite: str, output_dir: Path) -> dict[str, Any]:
    selected_checkpoints = None
    if args.fixed_checkpoint is not None or args.multi_checkpoint is not None:
        selected_checkpoints = {
            "fixed_bfs": args.fixed_checkpoint,
            "multi_chart": args.multi_checkpoint,
        }
    return run_suite(
        suite,
        data_root=args.data_root,
        output_dir=output_dir,
        requested_device=args.device,
        seed=args.seed,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        chart_seed=args.chart_seed,
        model_seed=args.model_seed,
        prepare_only=args.prepare_only,
        amp_override=args.amp,
        batch_size_override=args.batch_size,
        pin_memory_override=args.pin_memory,
        non_blocking_override=args.non_blocking,
        workers=args.workers,
        allow_download=args.allow_download,
        hidden_dim_override=args.hidden_dim,
        message_layers_override=args.message_layers,
        optimizer_updates_override=args.optimizer_updates,
        train_charts_per_graph_override=args.train_charts_per_graph,
        eval_charts_per_graph_override=args.eval_charts_per_graph,
        evaluation_scope=args.evaluation_scope,
        selected_checkpoints=selected_checkpoints,
    )


def _run_all(args: argparse.Namespace, output_root: Path) -> int:
    """Run every independent suite and leave an aggregate manifest on partial failure."""

    seed_axes = resolve_seed_axes(
        args.seed,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        chart_seed=args.chart_seed,
        model_seed=args.model_seed,
    )
    _prepare_output_dir(output_root)
    aggregate_path = output_root / "manifest.json"
    aggregate: dict[str, Any] = {
        "status": "preparing",
        "suite": "all",
        "seed_axes": seed_axes.to_manifest(),
        "prepare_only": args.prepare_only,
        "allow_download": args.allow_download,
        "workers": args.workers,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "suites": {},
    }
    _write_json(aggregate_path, aggregate)
    results: dict[str, Any] = {}
    optional_failure = False
    protocol_failure = False
    for suite in SUITES:
        suite_output = output_root / suite
        try:
            result = _run_from_args(args, suite, suite_output)
            child_status = "prepared" if args.prepare_only else "passed"
            results[suite] = result
            aggregate["suites"][suite] = {
                "status": child_status,
                "manifest_path": str(suite_output / "manifest.json"),
                "manifest_sha256": _sha256(suite_output / "manifest.json"),
            }
        except OptionalDatasetError as error:
            optional_failure = True
            failure = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "manifest_path": str(suite_output / "manifest.json"),
            }
            results[suite] = failure
            if (suite_output / "manifest.json").is_file():
                failure["manifest_sha256"] = _sha256(suite_output / "manifest.json")
            aggregate["suites"][suite] = failure
        except Exception as error:  # keep independent suites observable on partial failure
            protocol_failure = True
            failure = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "manifest_path": str(suite_output / "manifest.json"),
            }
            results[suite] = failure
            if (suite_output / "manifest.json").is_file():
                failure["manifest_sha256"] = _sha256(suite_output / "manifest.json")
            aggregate["suites"][suite] = failure
    if optional_failure or protocol_failure:
        aggregate["status"] = "failed"
    else:
        aggregate["status"] = "prepared" if args.prepare_only else "passed"
    aggregate["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(aggregate_path, aggregate)
    print(json.dumps(results, indent=2, sort_keys=True))
    if protocol_failure:
        print(f"one or more paper suites failed; see {aggregate_path}", file=sys.stderr)
        return 1
    if optional_failure:
        print(
            f"one or more optional datasets are unavailable; see {aggregate_path}",
            file=sys.stderr,
        )
        return 2
    return 0


def main() -> int:
    args = _parser().parse_args()
    suites = SUITES if args.suite == "all" else (args.suite,)
    output_root = args.output_dir.expanduser().resolve()
    try:
        if len(suites) > 1:
            return _run_all(args, output_root)
        results = {suites[0]: _run_from_args(args, suites[0], output_root)}
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    except OptionalDatasetError as error:
        print(f"optional dataset unavailable: {error}", file=sys.stderr)
        return 2
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"paper protocol failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
