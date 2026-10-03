"""Immutable experiment contracts and hashes for normalization retraining."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..classification.common import (
    cpu_state,
    digest,
    file_sha256,
    read_json,
    save_checkpoint,
    write_csv,
    write_json,
)

_FOLDER = Path(__file__).resolve().parent
_PROTECTED = (
    "schema",
    "experiment",
    "profile",
    "data_source",
    "data",
    "backbone",
    "operators",
    "gate",
    "conditions",
    "capacity_control",
    "training",
    "evaluation",
    "resources",
    "provenance",
    "debug_fixture",
)


def read_config(path, profile):
    """Reject reduced budgets, mixed profiles, test selection and hidden caps."""
    if profile not in ("full", "debug"):
        raise ValueError("profile must be full or debug")
    config = read_json(path)
    canonical = read_json(_FOLDER / f"config_{profile}.json")
    if set(config) != set(canonical):
        raise ValueError("unknown or missing config field; complete contract required")
    for key in _PROTECTED:
        if config.get(key) != canonical.get(key):
            raise ValueError(f"changed {key}: approved normalization contract is immutable")
    if config["status"] != canonical["status"] or config["date"] != canonical["date"]:
        raise ValueError("config status/date differ from the declared contract")
    runtime = config["runtime"]
    if set(runtime) != set(canonical["runtime"]):
        raise ValueError("unknown or missing runtime field")
    # CPU allocation can differ across machines. All physical packing/chunk
    # candidates and checkpoint cadence otherwise remain the measured contract.
    for key, value in canonical["runtime"].items():
        if key != "cpu_threads" and runtime[key] != value:
            raise ValueError(f"changed runtime {key}")
    threads = runtime["cpu_threads"]
    if threads != "auto" and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("cpu_threads must be auto or a positive integer")
    train = config["training"]
    tuning = (
        len(config["data"]["datasets"])
        * len(config["conditions"])
        * len(train["learning_rate_candidates"])
        * len(train["tuning_seeds"])
    )
    final = len(config["data"]["datasets"]) * len(config["conditions"]) * len(train["final_seeds"])
    if (tuning, final, tuning + final, (tuning + final) * train["epochs_per_run"]) != (
        train["tuning_runs"],
        train["final_runs"],
        train["total_runs"],
        train["total_updates"],
    ):
        raise ValueError("inconsistent complete training budget")
    return config


def source_manifest():
    """Hash local code and imported experiment dependencies, with portable paths."""
    wedge = _FOLDER.parent
    repo = wedge.parents[1]
    folders = (_FOLDER, wedge / "classification", wedge / "branch_strength", wedge)
    files = sorted(
        {
            file
            for folder in folders
            for pattern in ("*.py", "*.json")
            for file in folder.glob(pattern)
        }
    )
    if not files:
        raise ValueError("implementation source is missing")
    hashes = {file.relative_to(repo).as_posix(): file_sha256(file) for file in files}
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return {
        "schema": "wedge-node-normalization-source-v1",
        "sha256": hashes,
        "code_digest": digest(hashes),
        "git_commit": commit,
        "dependency_code_included": True,
    }


__all__ = [
    "cpu_state",
    "digest",
    "file_sha256",
    "read_config",
    "read_json",
    "save_checkpoint",
    "source_manifest",
    "write_csv",
    "write_json",
]
