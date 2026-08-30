#!/usr/bin/env python3
"""Select a pinned CUDA environment without importing any research packages.

The automatic installer uses a conservative driver policy, not CUDA minor-version
compatibility: the wheel runtime must not exceed nvidia-smi's reported capability.
Package versions are fixed by the selected lock, never by the package resolver.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_RUNTIMES = {"cu118": "11.8", "cu126": "12.6", "cu130": "13.0", "cu132": "13.2"}
LOCK_FILES = {
    "cu118": "requirements-cu118-lock.txt",
    "cu126": "requirements-lock.txt",
    "cu130": "requirements-lock.txt",
    "cu132": "requirements-lock.txt",
}


class GPUProfileError(ValueError):
    """No supported, explicitly locked GPU profile matches the request."""


def lock_for_tag(tag: str, root: Path = ROOT) -> Path:
    if tag not in LOCK_FILES:
        raise GPUProfileError(
            f"Unsupported CUDA_WHEEL_TAG={tag}; choose auto, {', '.join(LOCK_FILES)}."
        )
    return root / LOCK_FILES[tag]


def _cuda_version(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+", value.strip()):
        raise GPUProfileError(f"Cannot parse nvidia-smi CUDA capability: {value!r}")
    major, minor = value.strip().split(".")
    return int(major), int(minor)


def select_install_tag(driver_cuda: str, requested_tag: str = "auto") -> str:
    capability = _cuda_version(driver_cuda)
    if requested_tag == "auto":
        if capability >= (12, 6):
            tag = "cu126"
        elif capability >= (11, 8):
            tag = "cu118"
        else:
            raise GPUProfileError(
                f"nvidia-smi reports CUDA {driver_cuda}; no locked profile is available "
                "under the conservative driver policy (minimum displayed capability: 11.8). "
                "No packages or drivers have been changed."
            )
    else:
        lock_for_tag(requested_tag)  # Reject unsupported tags before any installation.
        tag = requested_tag
    if capability < _cuda_version(CUDA_RUNTIMES[tag]):
        advice = (
            "Use CUDA_WHEEL_TAG=cu118 bash scripts/setup_gpu.sh for the compatibility profile."
            if capability >= (11, 8)
            else "No locked profile meets this installer's conservative driver policy."
        )
        raise GPUProfileError(
            f"CUDA_WHEEL_TAG={tag} selects runtime {CUDA_RUNTIMES[tag]}, but nvidia-smi "
            f"reports CUDA {driver_cuda}. The installer does not rely on CUDA minor-version "
            f"compatibility or silently change an explicit selection. {advice}"
        )
    return tag


def check_wheel_host(tag: str) -> None:
    """Reject known binary-platform mismatches before downloading large wheels."""
    if platform.system() != "Linux":
        raise GPUProfileError("The GPU installation profiles require Linux.")
    if tag == "cu118" and platform.machine().lower() not in {"x86_64", "amd64"}:
        raise GPUProfileError("The locked PyTorch 2.7.1 cu118 Linux wheel requires x86_64.")
    if tag == "cu118" and not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise GPUProfileError(
            "The cu118 research profile requires Python 3.11-3.13; "
            "environment.yml uses the reference Python 3.11."
        )
    libc, version = platform.libc_ver()
    if libc != "glibc" or not version:
        raise GPUProfileError("Cannot verify glibc; the locked GPU wheels require glibc >= 2.28.")
    try:
        release = tuple(int(part) for part in version.split(".")[:2])
    except ValueError as error:
        raise GPUProfileError(f"Cannot parse glibc version: {version!r}") from error
    if release < (2, 28):
        raise GPUProfileError(
            f"glibc {version} is older than the locked wheels' minimum 2.28. "
            "Use a compatible Linux host/container; this script will not modify system libraries."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-cuda", required=True)
    parser.add_argument("--cuda-tag", default="auto")
    parser.add_argument("--check-host", action="store_true")
    args = parser.parse_args(argv)
    try:
        tag = select_install_tag(args.driver_cuda, args.cuda_tag)
        if args.check_host:
            check_wheel_host(tag)
    except GPUProfileError as error:
        print(error, file=sys.stderr)
        return 2
    # This restricted output is consumed by Bash read, never eval.
    print(tag, lock_for_tag(tag).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
