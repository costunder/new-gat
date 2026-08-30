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
    from scripts.gpu_profiles import (
        GPUProfileError,
        check_wheel_host,
        cuda_tag_for_profile,
        lock_for_profile,
        lock_for_tag,
        profile_for_torch_version,
    )
    from scripts.verify_gpu_lock import (
        CUDA_RUNTIMES,
        IMPORT_NAMES,
        LockVerificationError,
        read_exact_pins,
        version_matches,
    )
except ModuleNotFoundError:
    from gpu_profiles import (
        GPUProfileError,
        check_wheel_host,
        cuda_tag_for_profile,
        lock_for_profile,
        lock_for_tag,
        profile_for_torch_version,
    )
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

    exit_code = 2


class HostCompatibilityError(DependencyCheckError):
    """Package reinstallation cannot fix the current operating-system ABI."""

    exit_code = 3


def _installed_profile() -> str:
    """Identify the installed official wheel without importing Torch or querying a GPU."""
    requested = os.environ.get("CUDA_WHEEL_TAG", "auto") or "auto"
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        version = ""
    detected = profile_for_torch_version(version)
    if requested != "auto":
        lock_for_tag(requested)  # This variable is a CUDA tag, not a profile identifier.
        if detected is not None and cuda_tag_for_profile(detected) == requested:
            return detected
        return requested
    if detected is not None:
        return detected
    tag = version.partition("+")[2]
    if tag in CUDA_RUNTIMES:
        return tag
    # Still report missing/wrong versions together for an absent or non-CUDA wheel.
    return "cu126"


def check_dependencies(lock_path: Path | None = None) -> dict[str, Any]:
    """Check every direct pin, runtime import, and CUDA wheel without using a GPU."""
    try:
        profile_id = _installed_profile()
        cuda_tag = cuda_tag_for_profile(profile_id)
        if lock_path is None:
            lock_path = lock_for_profile(profile_id)
        pins = read_exact_pins(lock_path)
    except (GPUProfileError, LockVerificationError, OSError) as error:
        raise DependencyCheckError(f"Cannot read the research dependency lock: {error}") from error

    if sys.platform == "linux":
        try:
            check_wheel_host(profile_id)
        except GPUProfileError as error:
            raise HostCompatibilityError(str(error)) from error

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
        "profile_id": profile_id,
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    }


def error_message(error: Exception) -> str:
    if isinstance(error, HostCompatibilityError):
        return (
            f"RESEARCH HOST NOT COMPATIBLE\nPython: {sys.executable}\n  {error}\n"
            "Automatic package reinstallation is disabled for host ABI errors.\n"
            "For Ubuntu 18.04/glibc 2.27, see docs/ENVIRONMENT.md: "
            "use the opt-in legacy-cu118 profile in a NEW dedicated Conda environment.\n"
            "Do not replace system libc or downgrade an environment used by another run."
        )
    setup_command = "bash scripts/setup_gpu.sh"
    try:
        installed_profile = profile_for_torch_version(importlib.metadata.version("torch"))
    except Exception:
        # Error reporting must still work with absent or damaged distribution metadata.
        installed_profile = None
    if installed_profile is not None:
        setup_command += f" --profile {installed_profile}"
    return (
        f"RESEARCH DEPENDENCIES NOT READY\nPython: {sys.executable}\n  {error}\n"
        "Conda activation alone does not install the research packages.\n"
        f"Run: {setup_command}\n"
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
        return error.exit_code
    if not args.quiet:
        print(f"Research dependencies ready: {report['python']} (CUDA {report['cuda_runtime']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
