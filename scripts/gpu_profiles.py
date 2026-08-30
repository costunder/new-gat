#!/usr/bin/env python3
"""Select a pinned CUDA environment without importing any research packages.

The automatic installer uses a conservative driver policy, not CUDA minor-version
compatibility: the wheel runtime must not exceed nvidia-smi's reported capability.
Package versions are fixed by the selected lock, never by the package resolver.
"""

from __future__ import annotations

import argparse
import importlib.metadata
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
# A profile identifies the entire pinned stack, not merely its CUDA runtime.
# Legacy support is deliberately opt-in; the cu118 profile remains PyTorch 2.7.1.
PROFILE_LOCK_FILES = {
    **LOCK_FILES,
    "legacy-cu118": "requirements-legacy-cu118-lock.txt",
}
PROFILE_CUDA_TAGS = {**{tag: tag for tag in CUDA_RUNTIMES}, "legacy-cu118": "cu118"}
PROFILE_CONSTRAINT_FILES = {
    profile_id: f"constraints-{profile_id}.txt" for profile_id in PROFILE_LOCK_FILES
}
TORCH_PROFILES = {
    "2.7.1+cu118": "cu118",
    "2.13.0+cu126": "cu126",
    "2.13.0+cu130": "cu130",
    "2.13.0+cu132": "cu132",
    "2.6.0+cu118": "legacy-cu118",
}


class GPUProfileError(ValueError):
    """No supported, explicitly locked GPU profile matches the request."""


def lock_for_tag(tag: str, root: Path = ROOT) -> Path:
    if tag not in LOCK_FILES:
        raise GPUProfileError(
            f"Unsupported CUDA_WHEEL_TAG={tag}; choose auto, {', '.join(LOCK_FILES)}."
        )
    return root / LOCK_FILES[tag]


def _validate_profile(profile_id: str) -> None:
    if profile_id not in PROFILE_LOCK_FILES:
        raise GPUProfileError(
            f"Unsupported GPU profile={profile_id}; choose auto, {', '.join(PROFILE_LOCK_FILES)}."
        )


def lock_for_profile(profile_id: str, root: Path = ROOT) -> Path:
    """Return a registered profile's lock, never a user-supplied relative path."""
    _validate_profile(profile_id)
    return root / PROFILE_LOCK_FILES[profile_id]


def cuda_tag_for_profile(profile_id: str) -> str:
    _validate_profile(profile_id)
    return PROFILE_CUDA_TAGS[profile_id]


def profile_for_torch_version(exactinstalled: str) -> str | None:
    """Recognize only the five exact official Torch identities in the locks."""
    return TORCH_PROFILES.get(exactinstalled)


def installed_profile_for_bootstrap() -> str:
    """Preserve a known installed stack; only a Torch-free environment selects auto.

    Distribution metadata remains readable when Torch cannot import because
    another dependency is missing. An unknown installed build is not permission
    to replace it with a different research stack.
    """
    try:
        installed = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return "auto"
    except Exception as error:
        raise GPUProfileError(f"Cannot inspect the installed Torch metadata: {error}") from error
    profile_id = profile_for_torch_version(installed)
    if profile_id is None:
        raise GPUProfileError(
            f"Cannot automatically repair unregistered torch=={installed}. "
            "Choose a documented profile explicitly in a dedicated Conda environment; "
            "no installed Torch build will be replaced automatically."
        )
    return profile_id


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


def select_install_profile(
    driver_cuda: str,
    requested_profile: str = "auto",
    requested_tag: str = "auto",
) -> str:
    """Select a complete stack without implicitly opting into legacy PyTorch."""
    if requested_profile != "auto":
        _validate_profile(requested_profile)
    if requested_tag != "auto":
        lock_for_tag(requested_tag)
    if requested_profile == "auto":
        return select_install_tag(driver_cuda, requested_tag)
    tag = cuda_tag_for_profile(requested_profile)
    if requested_tag not in {"auto", tag}:
        raise GPUProfileError(
            f"GPU profile={requested_profile} requires CUDA_WHEEL_TAG={tag}, "
            f"not {requested_tag}; explicit selections must agree."
        )
    select_install_tag(driver_cuda, tag)
    return requested_profile


def check_wheel_host(profile_id: str) -> None:
    """Reject known binary-platform mismatches before downloading large wheels."""
    tag = cuda_tag_for_profile(profile_id)
    if platform.system() != "Linux":
        raise GPUProfileError("The GPU installation profiles require Linux.")
    if tag == "cu118" and platform.machine().lower() not in {"x86_64", "amd64"}:
        raise GPUProfileError(f"The {profile_id} Linux wheel requires x86_64.")
    if tag == "cu118" and not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise GPUProfileError(
            "The cu118 research profile requires Python 3.11-3.13; "
            "environment.yml uses the reference Python 3.11."
        )
    # Torch 2.6 cu118 itself targets manylinux2014 (glibc 2.17), but the
    # complete legacy research stack is supported only from glibc 2.27.
    minimum = (2, 27) if profile_id == "legacy-cu118" else (2, 28)
    minimum_label = ".".join(str(part) for part in minimum)
    libc, version = platform.libc_ver()
    if libc != "glibc" or not version:
        raise GPUProfileError(
            f"Cannot verify glibc; profile {profile_id} requires glibc >= {minimum_label}."
        )
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)*", version):
        raise GPUProfileError(f"Cannot parse glibc version: {version!r}")
    release = tuple(int(part) for part in version.split(".")[:2])
    if release < minimum:
        raise GPUProfileError(
            f"glibc {version} is older than profile {profile_id}'s minimum {minimum_label}. "
            "Use a compatible Linux host/container; this script will not modify system libraries."
        )


def check_install_target(profile_id: str) -> None:
    """Protect existing environments from an in-place, legacy-only downgrade.

    This installation guard uses distribution metadata, not a Torch import. It
    must not be used by read-only runtime checks: a valid installed profile can
    be checked independently of the environment's name.
    """
    _validate_profile(profile_id)
    if profile_id != "legacy-cu118":
        return
    if Path(sys.prefix).name.casefold() == "new-gat":
        raise GPUProfileError(
            "legacy-cu118 must use a separate Conda environment, not new-gat. "
            "Create and activate a dedicated environment before installation; "
            "the existing environment will not be downgraded."
        )
    try:
        installed = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return
    except Exception as error:
        raise GPUProfileError(
            f"Cannot inspect the installation target's Torch metadata: {error}"
        ) from error
    if installed != "2.6.0+cu118":
        raise GPUProfileError(
            f"legacy-cu118 cannot replace existing torch=={installed} in {sys.prefix}. "
            "Use a fresh dedicated Conda environment; only an existing exact "
            "torch==2.6.0+cu118 installation may be reused."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--driver-cuda")
    mode.add_argument(
        "--installed-profile",
        action="store_true",
        help="print the exact installed profile, or auto only when Torch is absent; read-only",
    )
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--cuda-tag", default="auto")
    parser.add_argument("--check-host", action="store_true")
    args = parser.parse_args(argv)
    if args.installed_profile and (
        args.profile != "auto" or args.cuda_tag != "auto" or args.check_host
    ):
        parser.error("--installed-profile cannot be combined with installation selection options")
    try:
        if args.installed_profile:
            print(installed_profile_for_bootstrap())
            return 0
        profile_id = select_install_profile(args.driver_cuda, args.profile, args.cuda_tag)
        if args.check_host:
            check_wheel_host(profile_id)
            check_install_target(profile_id)
    except GPUProfileError as error:
        print(error, file=sys.stderr)
        return 2
    # This restricted output is consumed by Bash read, never eval.
    print(
        profile_id,
        cuda_tag_for_profile(profile_id),
        lock_for_profile(profile_id).name,
        PROFILE_CONSTRAINT_FILES[profile_id],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
