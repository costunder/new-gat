#!/usr/bin/env python3
"""Check the full research stack before importing any project training code.

This checker itself uses only Python's standard library. It never installs
packages, creates run output, or requires an allocated GPU for data preparation.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.verify_gpu_lock import (
        CUDA_RUNTIMES,
        IMPORT_NAMES,
        LockVerificationError,
        read_exact_pins,
        version_matches,
    )
except ModuleNotFoundError:
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


def check_dependencies(lock_path: Path = ROOT / "requirements-lock.txt") -> dict[str, Any]:
    """Check every direct pin, runtime import, and CUDA wheel without using a GPU."""
    try:
        pins = read_exact_pins(lock_path)
    except (LockVerificationError, OSError) as error:
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
        if not version_matches(name, expected, actual):
            problems.append(f"{name}: installed {actual}, required {expected}")
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
    if runtime not in CUDA_RUNTIMES.values():
        raise DependencyCheckError(
            f"torch CUDA runtime is {runtime}; install the project's CUDA wheel with setup_gpu.sh"
        )
    return {"python": sys.executable, "installed": installed, "cuda_runtime": runtime}


def error_message(error: Exception) -> str:
    return (
        f"RESEARCH DEPENDENCIES NOT READY\nPython: {sys.executable}\n  {error}\n"
        "Conda activation alone does not install the research packages.\n"
        "Run: bash scripts/setup_gpu.sh\n"
        "Data preparation also installs missing dependencies automatically on a Linux GPU host."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "requirements-lock.txt")
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
