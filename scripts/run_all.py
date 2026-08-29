#!/usr/bin/env python3
"""Run every currently implemented research track without combining models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCE_DIRS = (
    "src",
    "tests",
    "research/conductance_gat",
    "research/cycle_pe",
    "research/tree_augmentation",
    "scripts",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("smoke-%Y%m%dT%H%M%S%fZ")


def _validate_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _commands(
    *,
    python: str,
    device: str,
    run_id: str,
    skip_tests: bool,
    skip_lint: bool,
) -> list[tuple[str, list[str]]]:
    conductance_output = PROJECT_ROOT / "research" / "conductance_gat" / "results" / run_id
    cycle_output = PROJECT_ROOT / "research" / "cycle_pe" / "results" / run_id / "summary.json"
    tree_output = (
        PROJECT_ROOT / "research" / "tree_augmentation" / "results" / run_id / "summary.json"
    )
    commands: list[tuple[str, list[str]]] = []
    commands.append(("dependencies", [python, "-m", "pip", "check"]))
    commands.append(
        (
            "dataset_plan",
            [python, str(PROJECT_ROOT / "scripts" / "check_datasets.py"), "--profile", "smoke"],
        )
    )
    if not skip_tests:
        commands.append(("tests", [python, "-m", "pytest", "-q"]))
    if not skip_lint:
        commands.append(("lint", [python, "-m", "ruff", "check", *ACTIVE_SOURCE_DIRS]))
    commands.extend(
        [
            (
                "conductance_gat",
                [
                    python,
                    "-m",
                    "research.conductance_gat.run",
                    "--config",
                    str(PROJECT_ROOT / "research" / "conductance_gat" / "config.yaml"),
                    "--device",
                    device,
                    "--output-dir",
                    str(conductance_output),
                ],
            ),
            (
                "cycle_pe",
                [
                    python,
                    "-m",
                    "research.cycle_pe.run",
                    "--config",
                    str(PROJECT_ROOT / "research" / "cycle_pe" / "config.yaml"),
                    "--output",
                    str(cycle_output),
                ],
            ),
            (
                "tree_augmentation",
                [
                    python,
                    "-m",
                    "research.tree_augmentation.run",
                    "--config",
                    str(PROJECT_ROOT / "research" / "tree_augmentation" / "config.yaml"),
                    "--output",
                    str(tree_output),
                ],
            ),
        ]
    )
    return commands


def _output_paths(run_id: str) -> dict[str, Path]:
    return {
        "conductance_gat": PROJECT_ROOT / "research" / "conductance_gat" / "results" / run_id,
        "cycle_pe": PROJECT_ROOT / "research" / "cycle_pe" / "results" / run_id / "summary.json",
        "tree_augmentation": PROJECT_ROOT
        / "research"
        / "tree_augmentation"
        / "results"
        / run_id
        / "summary.json",
    }


def _expected_artifacts(outputs: dict[str, Path]) -> tuple[Path, ...]:
    conductance = outputs["conductance_gat"]
    return (
        conductance / "summary.json",
        conductance / "learned_history.csv",
        conductance / "isotropic_history.csv",
        conductance / "learned_model.pt",
        outputs["cycle_pe"],
        outputs["tree_augmentation"],
    )


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_run(run_dir: Path) -> dict[str, Any]:
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    snapshots: dict[str, Any] = {}
    for track in ("conductance_gat", "cycle_pe", "tree_augmentation"):
        for source_name in ("config.yaml", "datasets.yaml"):
            source = PROJECT_ROOT / "research" / track / source_name
            target = config_dir / f"{track}-{source_name}"
            shutil.copy2(source, target)
            snapshots[f"{track}-{source_name}"] = {
                "path": str(target),
                "sha256": _sha256(target),
            }

    distributions = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    environment_path = run_dir / "environment.txt"
    environment_path.write_text("\n".join(distributions) + "\n", encoding="utf-8")
    snapshots["environment"] = {
        "path": str(environment_path),
        "sha256": _sha256(environment_path),
    }
    return snapshots


def _source_revision() -> dict[str, Any]:
    if not (PROJECT_ROOT / ".git").exists():
        return {"git_available": False, "revision": None, "dirty": None}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_available": revision.returncode == 0,
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--run-id", type=_validate_run_id, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or _default_run_id()
    commands = _commands(
        python=sys.executable,
        device=args.device,
        run_id=run_id,
        skip_tests=args.skip_tests,
        skip_lint=args.skip_lint,
    )
    run_dir = PROJECT_ROOT / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    outputs = _output_paths(run_id)
    payload: dict[str, Any] = {
        "scope": "all_currently_implemented_independent_smoke_tracks",
        "paper_benchmark_suite_complete": False,
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source": _source_revision(),
        "device_request": args.device,
        "commands": [],
    }

    environment = os.environ.copy()
    python_path = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["PYTHONUTF8"] = "1"

    if args.dry_run:
        for name, command in commands:
            print(f"[{name}] {_command_text(command)}")
        print(f"[manifest] {manifest_path}")
        return 0

    collisions = [
        path
        for path in (
            run_dir,
            outputs["conductance_gat"],
            outputs["cycle_pe"].parent,
            outputs["tree_augmentation"].parent,
        )
        if path.exists()
    ]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        print(f"run id {run_id!r} already exists; refusing to overwrite: {joined}", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=False)
    payload["snapshots"] = _snapshot_run(run_dir)
    payload["status"] = "running"
    _write_manifest(manifest_path, payload)

    try:
        for name, command in commands:
            print(f"\n== {name}: {_command_text(command)} ==", flush=True)
            started = datetime.now(UTC)
            log_path = run_dir / "logs" / f"{name}.log"
            returncode = _run_logged(
                command,
                cwd=PROJECT_ROOT,
                environment=environment,
                log_path=log_path,
            )
            record = {
                "name": name,
                "command": command,
                "returncode": returncode,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                "log": str(log_path),
            }
            payload["commands"].append(record)
            _write_manifest(manifest_path, payload)
            if returncode != 0:
                payload["status"] = "failed"
                payload["failed_step"] = name
                payload["finished_at_utc"] = datetime.now(UTC).isoformat()
                _write_manifest(manifest_path, payload)
                print(f"failed at {name}; manifest: {manifest_path}", file=sys.stderr)
                return returncode
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, payload)
        print(f"interrupted; manifest: {manifest_path}", file=sys.stderr)
        return 130

    missing = [str(path) for path in _expected_artifacts(outputs) if not path.is_file()]
    if missing:
        payload["status"] = "failed"
        payload["failed_step"] = "artifact_validation"
        payload["missing_artifacts"] = missing
        payload["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, payload)
        print(f"missing expected artifacts: {missing}", file=sys.stderr)
        return 1

    try:
        for summary in (
            outputs["conductance_gat"] / "summary.json",
            outputs["cycle_pe"],
            outputs["tree_augmentation"],
        ):
            content = json.loads(summary.read_text(encoding="utf-8"))
            if not _all_finite(content):
                raise ValueError(f"non-finite value in {summary}")
    except (json.JSONDecodeError, ValueError) as error:
        payload["status"] = "failed"
        payload["failed_step"] = "artifact_validation"
        payload["artifact_error"] = str(error)
        payload["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, payload)
        print(f"invalid result artifact: {error}", file=sys.stderr)
        return 1

    payload["status"] = "passed"
    payload["finished_at_utc"] = datetime.now(UTC).isoformat()
    payload["outputs"] = {name: str(path) for name, path in outputs.items()}
    _write_manifest(manifest_path, payload)
    print(f"\nall implemented tracks passed; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
