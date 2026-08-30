#!/usr/bin/env python3
"""Check CUDA hardware and package imports without creating data or training a model."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from chartgat.cache import atomic_write_json


class PreflightError(RuntimeError):
    """The requested GPU or dependency environment is unavailable."""


PAPER_IMPORTS = {
    "networkx": "networkx",
    "numpy": "numpy",
    "ogb": "ogb",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "torch-geometric": "torch_geometric",
    "PyYAML": "yaml",
}


def _paper_dependency_import_errors() -> dict[str, str]:
    errors = {}
    for distribution, module in PAPER_IMPORTS.items():
        try:
            importlib.import_module(module)
        except Exception as error:
            errors[distribution] = f"{type(error).__name__}: {error}"
    return errors


def _resolve_device(requested: str) -> torch.device:
    try:
        device = torch.device(requested.strip().lower())
    except (RuntimeError, ValueError) as error:
        raise PreflightError(f"invalid device: {requested!r}") from error
    if device.type != "cuda":
        raise PreflightError("paper execution requires CUDA; no CPU fallback is available")
    try:
        available = torch.cuda.is_available()
        index = torch.cuda.current_device() if available and device.index is None else device.index
        visible_count = torch.cuda.device_count() if available else 0
    except (RuntimeError, AssertionError) as error:
        raise PreflightError(f"CUDA initialization failed: {error}") from error
    if not available:
        raise PreflightError(
            "CUDA is unavailable; activate the CUDA environment and expose an NVIDIA GPU"
        )
    assert index is not None
    if index < 0 or index >= visible_count:
        raise PreflightError(
            f"CUDA device index {index} is invalid; visible count is {visible_count}"
        )
    return torch.device("cuda", index)


def build_report(
    requested_device: str,
    *,
    require_paper_dependencies: bool = False,
    min_free_gb: float = 2.0,
) -> dict[str, Any]:
    if not math.isfinite(min_free_gb) or min_free_gb < 0:
        raise PreflightError("--min-free-gb must be finite and non-negative")
    device = _resolve_device(requested_device)
    if require_paper_dependencies:
        errors = _paper_dependency_import_errors()
        if errors:
            raise PreflightError(f"paper dependency imports failed: {errors}")
    try:
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except RuntimeError as error:
        raise PreflightError(f"cannot query CUDA device {device}: {error}") from error
    if free_bytes < min_free_gb * (1024**3):
        raise PreflightError(
            f"{device} has {free_bytes / (1024**3):.2f} GiB free; "
            f"at least {min_free_gb:g} GiB was requested"
        )
    versions = {"torch": str(torch.__version__)}
    if require_paper_dependencies:
        for distribution in PAPER_IMPORTS:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "unknown"
    return {
        "status": "passed",
        "kind": "hardware_and_dependency_check",
        "requested_device": requested_device,
        "resolved_device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_runtime": torch.version.cuda,
        "packages": versions,
        "gpu": {
            "name": properties.name,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "compute_capability": [properties.major, properties.minor],
        },
        "min_free_gb": min_free_gb,
        "dataset_loaded": False,
        "model_executed": False,
        "scope": "availability only; does not certify dataset fit or experiment results",
    }


def _save_report(path: Path | None, report: dict[str, Any]) -> bool:
    if path is None:
        return True
    try:
        atomic_write_json(path, report)
    except OSError as error:
        print(f"cannot save GPU report to {path}: {error}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--require-paper-deps", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(
            args.device,
            require_paper_dependencies=args.require_paper_deps,
            min_free_gb=args.min_free_gb,
        )
    except PreflightError as error:
        report = {"status": "failed", "kind": "hardware_and_dependency_check", "error": str(error)}
        print(str(error), file=sys.stderr)
        _save_report(args.json_out, report)
        return 2
    if not _save_report(args.json_out, report):
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
