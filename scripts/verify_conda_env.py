#!/usr/bin/env python3
"""Reject base, nested venvs, and mismatched Python before any package installation."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


class CondaEnvironmentError(RuntimeError):
    """A Bash entrypoint is not using a dedicated active Conda environment."""


def _conda_base(environ: Mapping[str, str]) -> Path:
    command = environ.get("CONDA_EXE") or "conda"
    try:
        result = subprocess.run(
            [command, "info", "--base"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CondaEnvironmentError(
            "Cannot query 'conda info --base'; initialize Conda and activate new-gat again."
        ) from error
    if not result.stdout.strip():
        raise CondaEnvironmentError("'conda info --base' returned an empty path.")
    return Path(result.stdout.strip()).resolve()


def verify_conda_environment(environ: Mapping[str, str] | None = None) -> Path:
    """Validate the current interpreter without installing or creating anything."""

    environ = os.environ if environ is None else environ
    raw_prefix = environ.get("CONDA_PREFIX", "")
    if not raw_prefix:
        raise CondaEnvironmentError("No active Conda environment; run 'conda activate new-gat'.")
    if environ.get("CONDA_DEFAULT_ENV") == "base":
        raise CondaEnvironmentError("Do not use Conda base; create and activate new-gat first.")
    if environ.get("VIRTUAL_ENV"):
        raise CondaEnvironmentError(
            "A venv is still active; deactivate it before activating the Conda environment."
        )

    expected = Path(raw_prefix).resolve()
    if not (expected / "conda-meta").is_dir():
        raise CondaEnvironmentError(f"Not a Conda environment (conda-meta is missing): {expected}")
    if sys.version_info < (3, 11):  # noqa: UP036 - this runs before package installation
        raise CondaEnvironmentError(
            "Python 3.11 or newer is required; create new-gat with Python 3.11."
        )
    if Path(sys.prefix).resolve() != expected:
        raise CondaEnvironmentError("Python sys.prefix does not match the active CONDA_PREFIX.")
    if Path(sys.base_prefix).resolve() != expected:
        raise CondaEnvironmentError(
            "Nested venv Python is not supported; use Conda's Python directly."
        )
    expected_python = (expected / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise CondaEnvironmentError("The interpreter is not CONDA_PREFIX/bin/python.")
    if expected == _conda_base(environ):
        raise CondaEnvironmentError(
            "Do not install into Conda base; activate a dedicated environment."
        )
    return expected


def main() -> int:
    try:
        prefix = verify_conda_environment()
    except (CondaEnvironmentError, OSError) as error:
        print(f"CONDA ENVIRONMENT CHECK FAILED: {error}", file=sys.stderr)
        return 2
    print(f"Conda environment: {prefix}")
    print(f"Python: {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
