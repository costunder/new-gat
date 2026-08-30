#!/usr/bin/env python3
"""Check the full research stack before importing any project training code.

This checker itself uses only Python's standard library. It never installs
packages, creates run output, or requires an allocated GPU for data preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.gpu_profiles import GPUProfileError, lock_for_tag
    from scripts.verify_gpu_lock import (
        CUDA_RUNTIMES,
        IMPORT_NAMES,
        LockVerificationError,
        read_exact_pins,
        version_matches,
    )
except ModuleNotFoundError:
    from gpu_profiles import GPUProfileError, lock_for_tag
    from verify_gpu_lock import (
        CUDA_RUNTIMES,
        IMPORT_NAMES,
        LockVerificationError,
        read_exact_pins,
        version_matches,
    )

ROOT = Path(__file__).resolve().parents[1]


class DependencyCheckError(RuntimeError):
    """The active interpreter is missing the installed research environment."""


def _installed_cuda_tag() -> str:
    """Identify the installed official wheel without importing Torch or querying a GPU."""
    requested = os.environ.get("CUDA_WHEEL_TAG", "auto") or "auto"
    if requested != "auto":
        lock_for_tag(requested)
        return requested
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return "cu126"  # Missing stack: report reference pins; setup selects using the driver.
    tag = version.partition("+")[2]
    if tag in CUDA_RUNTIMES:
        return tag
    # Still report missing/wrong versions together for an absent or non-CUDA wheel.
    return "cu126"


def check_dependencies(lock_path: Path | None = None) -> dict[str, Any]:
    """Check every direct pin, runtime import, and CUDA wheel without using a GPU."""
    try:
        cuda_tag = _installed_cuda_tag()
        if lock_path is None:
            lock_path = lock_for_tag(cuda_tag)
        pins = read_exact_pins(lock_path)
    except (GPUProfileError, LockVerificationError, OSError) as error:
        raise DependencyCheckError(f"Cannot read the research dependency lock: {error}") from error

    installed: dict[str, str] = {}
    problems: list[str] = []
    for name, expected in sorted(pins.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name}: missing (required {expected})")
            continue
        installed[name] = actual
        if not version_matches(name, expected, actual, cuda_tag=cuda_tag):
            expected_label = f"{expected}+{cuda_tag}" if name == "torch" else expected
            problems.append(f"{name}: installed {actual}, required {expected_label}")
    if problems:
        # Report the entire missing stack at once, before trying NumPy or Torch.
        raise DependencyCheckError("\n  ".join(problems))

    modules: dict[str, Any] = {}
    for distribution, module_name in {**IMPORT_NAMES, "torch": "torch", "tqdm": "tqdm"}.items():
        try:
            modules[distribution] = importlib.import_module(module_name)
        except Exception as error:
            problems.append(f"{distribution}: import failed ({type(error).__name__}: {error})")
    if problems:
        raise DependencyCheckError("\n  ".join(problems))
    runtime = str(modules["torch"].version.cuda)
    expected_runtime = CUDA_RUNTIMES[cuda_tag]
    if runtime != expected_runtime:
        raise DependencyCheckError(
            f"torch CUDA runtime is {runtime}, expected {expected_runtime} for {cuda_tag}; "
            "install the project's CUDA wheel with setup_gpu.sh"
        )
    return {
        "python": sys.executable,
        "installed": installed,
        "cuda_runtime": runtime,
        "cuda_wheel_tag": cuda_tag,
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    }


def error_message(error: Exception) -> str:
    return (
        f"RESEARCH DEPENDENCIES NOT READY\nPython: {sys.executable}\n  {error}\n"
        "Conda activation alone does not install the research packages.\n"
        "Run: bash scripts/setup_gpu.sh\n"
        "Data preparation also installs missing dependencies automatically on a Linux GPU host."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, help="override the installed CUDA profile's lock")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = check_dependencies(args.lock)
    except DependencyCheckError as error:
        print(error_message(error), file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"Research dependencies ready: {report['python']} (CUDA {report['cuda_runtime']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
