#!/usr/bin/env python3
"""Preserve one failed V5/Cycle-V2 rich run without deleting or rewriting artifacts.

The default is a read-only plan. --apply requires Linux process verification and
renames only the explicitly bound run directories. Archived manifests retain
their original paths and hashes: this is historical preservation, not resume or
checkpoint migration. Use a new run ID after changing experiment source.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime as dt
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rich_scaling import TRACK_SPECS, _child_run_id  # noqa: E402

ARCHIVE_DIRECTORY = "_archived_failed_runs"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}")
TERMINAL = ("passed", "failed", "interrupted")
ALLOWED_TRACKS = {"conductance": "v5", "cycle": "v2"}
OUTPUT_FLAGS = {"--output-dir", "--output", "--results-dir"}


class ArchiveError(RuntimeError):
    """The exact archive boundary could not be verified."""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--apply", action="store_true", help="perform verified Linux-only renames")
    return result


def _direct_path(value: Path | str, *, within: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ArchiveError(f"parent traversal is forbidden: {path}")
    path = Path(os.path.abspath(path))
    for component in (*reversed(path.parents), path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ArchiveError(f"symlink/reparse path is forbidden: {component}")
    if path.resolve() != path:
        raise ArchiveError(f"indirect path is forbidden: {path}")
    if within is not None and (path == within or not path.is_relative_to(within)):
        raise ArchiveError(f"path is outside the exact results root: {path}")
    return path


def _json_object(path: Path) -> dict[str, Any]:
    path = _direct_path(path)
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ArchiveError(f"manifest is not a regular file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArchiveError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ArchiveError(f"manifest is not an object: {path}")
    return payload


def _assert_safe_tree(path: Path) -> None:
    """Inspect directory entries only; never read model tensor/checkpoint bytes."""
    if not path.is_dir():
        raise ArchiveError(f"run directory is missing or not a directory: {path}")

    def fail_walk(error: OSError) -> None:
        raise ArchiveError(f"cannot verify artifact directory entries: {error}") from error

    for directory, subdirectories, files in os.walk(path, followlinks=False, onerror=fail_walk):
        for name in (*subdirectories, *files):
            candidate = Path(directory) / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or (
                getattr(metadata, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise ArchiveError(f"symlink/reparse artifact is forbidden: {candidate}")


def _metadata_hashes(directory: Path) -> dict[str, str]:
    result = {}
    for name in ("manifest.json", "summary.json"):
        path = _direct_path(directory / name, within=directory)
        if path.exists():
            if not path.is_file():
                raise ArchiveError(f"metadata is not a regular file: {path}")
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _version_selection(config: Any, field: str, expected: str, label: str) -> None:
    if not isinstance(config, dict) or config.get(field) != [expected]:
        raise ArchiveError(
            f"{label} must select only {expected}; legacy experiments stay untouched"
        )


def plan_archive(run_id: str, results_root: Path) -> dict[str, Any]:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ArchiveError("run ID must be 1-120 letters, digits, underscores or hyphens")
    results_root = _direct_path(results_root)
    if results_root in {Path(results_root.anchor), Path.home().resolve(), ROOT}:
        raise ArchiveError("a filesystem, home or repository root is not a results directory")
    parent = _direct_path(results_root / "rich_scaling" / run_id, within=results_root)
    root_manifest = _json_object(parent / "manifest.json")
    if (
        root_manifest.get("schema_version") != 1
        or root_manifest.get("suite") != "rich_scaling"
        or root_manifest.get("run_id") != run_id
        or root_manifest.get("status") not in ("failed", "interrupted")
    ):
        raise ArchiveError("only an explicitly failed/interrupted rich run can be archived")
    config = root_manifest.get("config")
    jobs = root_manifest.get("jobs")
    if not isinstance(config, dict) or not isinstance(jobs, list) or not jobs:
        raise ArchiveError("rich run configuration or child job list is missing")
    if (
        not isinstance(config.get("results_root"), str)
        or _direct_path(config["results_root"]) != results_root
    ):
        raise ArchiveError("rich configuration results root differs from this exact directory")
    tracks = [job.get("track") if isinstance(job, dict) else None for job in jobs]
    if (
        any(not isinstance(track, str) for track in tracks)
        or len(set(tracks)) != len(tracks)
        or any(track not in ALLOWED_TRACKS for track in tracks)
        or config.get("tracks") != tracks
    ):
        raise ArchiveError(
            "only distinct explicitly bound conductance/V5 and cycle/V2 tracks apply"
        )
    targets = [{"track": "rich", "run_id": run_id, "original": str(parent)}]
    for job in jobs:
        track = job["track"]
        expected_version = ALLOWED_TRACKS[track]
        _version_selection(config, f"{track}_versions", expected_version, "rich config")
        _version_selection(job.get("requested_matrix"), "versions", expected_version, track)
        expected_id = _child_run_id(run_id, track)
        expected_path = _direct_path(
            results_root / TRACK_SPECS[track]["results_subdir"] / expected_id,
            within=results_root,
        )
        if (
            job.get("child_run_id") != expected_id
            or job.get("status") not in TERMINAL
            or not isinstance(job.get("output_dir"), str)
            or _direct_path(job["output_dir"], within=results_root) != expected_path
            or job.get("summary_path") != str(expected_path / "summary.json")
        ):
            raise ArchiveError(f"{track} child identity/path/terminal-status binding is invalid")
        child = _json_object(expected_path / "manifest.json")
        if child.get("run_id") != expected_id or child.get("status") not in TERMINAL:
            raise ArchiveError(f"{track} child manifest is not a terminal matching run")
        if track == "conductance":
            if child.get("schema_version") != 1 or child.get("suite") not in (
                "conductance_architecture_scaling_v1_v4",
                "conductance_architecture_scaling_v1_v5",
            ):
                raise ArchiveError("unrecognized Conductance scaling manifest schema")
            _version_selection(child.get("config"), "versions", "v5", "Conductance child")
        else:
            if (
                child.get("schema_version") != 2
                or child.get("scope") != "cycle_pe_v1_v2_larger_model_scaling"
                or child.get("output_dir") != str(expected_path)
            ):
                raise ArchiveError("unrecognized Cycle scaling manifest schema/path")
            _version_selection(child, "versions", "v2", "Cycle child")
        # Pending leaf jobs can be unstarted work after a terminal parent failure.
        # A recorded running leaf, however, is ambiguous even without a live PID.
        job_sections = ("jobs", "test_evaluation_jobs") if track == "cycle" else ("jobs",)
        for section in job_sections:
            leaves = child.get(section)
            if not isinstance(leaves, list) or any(
                not isinstance(leaf, dict)
                or leaf.get("status") not in TERMINAL + ("pending",)
                or leaf.get("version") != expected_version
                for leaf in leaves
            ):
                raise ArchiveError(
                    f"{track}/{section} contains a running/ambiguous or legacy-version leaf job"
                )
        legacy_roots = (
            [expected_path / version for version in ("v1", "v2", "v3", "v4")]
            if track == "conductance"
            else [expected_path / section / "v1" for section in ("results", "test-evaluations")]
        )
        if any(path.exists() for path in legacy_roots):
            raise ArchiveError(f"{track} contains legacy-version directories; nothing will move")
        targets.append({"track": track, "run_id": expected_id, "original": str(expected_path)})
    paths = [Path(target["original"]) for target in targets]
    if any(
        first == second or first.is_relative_to(second) or second.is_relative_to(first)
        for index, first in enumerate(paths)
        for second in paths[index + 1 :]
    ):
        raise ArchiveError("archive targets overlap or duplicate one another")
    for target in targets:
        path = Path(target["original"])
        _assert_safe_tree(path)
        target["metadata_sha256"] = _metadata_hashes(path)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "results_root": str(results_root),
        "source_status": root_manifest["status"],
        "targets": targets,
        "policy": (
            "preserve artifact bytes and original metadata; never resume under changed source"
        ),
        "next_run": "use a new run ID after source changes; do not copy old checkpoints into it",
    }


def _flag_values(arguments: list[str], names: set[str]) -> list[str]:
    values = []
    for index, argument in enumerate(arguments):
        if argument in names:
            if index + 1 >= len(arguments):
                raise ArchiveError(f"live process has a flag without a value: {argument}")
            values.append(arguments[index + 1])
        else:
            key, separator, value = argument.partition("=")
            if separator and key in names:
                values.append(value)
    return values


def active_processes(
    plan: dict[str, Any],
    *,
    proc_root: Path = Path("/proc"),
    uid: int | None = None,
    own_pid: int | None = None,
) -> list[dict[str, Any]]:
    """Inspect same-UID Linux command arguments; no signals or process changes."""
    if uid is None:
        uid = os.getuid()
    if own_pid is None:
        own_pid = os.getpid()
    if not proc_root.is_dir():
        raise ArchiveError("Linux /proc is unavailable; active-process verification is required")
    run_ids = {target["run_id"] for target in plan["targets"]}
    targets = [Path(target["original"]) for target in plan["targets"]]
    matches = []
    for process in proc_root.iterdir():
        if not process.name.isdecimal() or int(process.name) == own_pid:
            continue
        try:
            status = (process / "status").read_text(encoding="utf-8")
        except FileNotFoundError as error:
            if not process.exists():
                continue
            raise ArchiveError(f"cannot verify UID of live process {process.name}") from error
        except (OSError, UnicodeError) as error:
            raise ArchiveError(
                f"cannot verify UID of live process {process.name}: {error}"
            ) from error
        match = re.search(r"(?m)^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", status)
        if match is None:
            raise ArchiveError(f"live process {process.name} has no verifiable UID")
        if uid not in {int(value) for value in match.groups()}:
            continue
        try:
            command = [
                os.fsdecode(value)
                for value in (process / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except FileNotFoundError as error:
            if not process.exists():
                continue
            raise ArchiveError(f"cannot read same-UID live process {process.name}") from error
        except OSError as error:
            raise ArchiveError(
                f"cannot read same-UID live process {process.name}: {error}"
            ) from error
        reasons = []
        if run_ids.intersection(_flag_values(command, {"--run-id"})):
            reasons.append("exact root/child run ID")
        for value in _flag_values(command, OUTPUT_FLAGS):
            candidate = Path(value)
            if not candidate.is_absolute():
                try:
                    candidate = (process / "cwd").resolve(strict=True) / candidate
                except OSError as error:
                    raise ArchiveError(
                        f"cannot resolve same-UID process {process.name} output: {error}"
                    ) from error
            candidate = candidate.resolve()
            if any(candidate == target or candidate.is_relative_to(target) for target in targets):
                reasons.append("output inside an archive target")
        if reasons:
            matches.append({"pid": int(process.name), "reasons": reasons, "command": command})
    return matches


def _require_inactive(plan: dict[str, Any]) -> None:
    matches = active_processes(plan)
    if matches:
        raise ArchiveError("active matching processes; no signals sent: " + json.dumps(matches))


def _append_event(journal, event: dict[str, Any]) -> None:
    journal.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    journal.flush()
    os.fsync(journal.fileno())


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Linux atomic same-filesystem rename which never replaces an existing path."""
    library_name = ctypes.util.find_library("c")
    if library_name is None:
        raise ArchiveError("cannot locate libc for atomic no-replace rename")
    library = ctypes.CDLL(library_name, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise ArchiveError("renameat2 is unavailable; no overwrite-capable fallback is permitted")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100, RENAME_NOREPLACE=1; EXDEV/unsupported calls fail without moving.
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(source), None, str(destination))


def apply_archive(plan: dict[str, Any]) -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise ArchiveError("--apply requires Linux /proc verification; dry-run is available here")
    # Re-read every binding immediately before mutation instead of trusting a stale plan.
    current = plan_archive(plan["run_id"], Path(plan["results_root"]))
    if current != plan:
        raise ArchiveError("run metadata changed after planning; no outputs moved")
    _require_inactive(plan)
    results_root = Path(plan["results_root"])
    archive_parent = _direct_path(results_root / ARCHIVE_DIRECTORY, within=results_root)
    archive_parent.mkdir(exist_ok=True)
    locks = _direct_path(archive_parent / "_locks", within=archive_parent)
    locks.mkdir(exist_ok=True)
    lock_path = _direct_path(locks / f"{plan['run_id']}.lock", within=locks)
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ArchiveError(
            f"archive lock already exists; inspect its owner: {lock_path}"
        ) from error
    moves: list[dict[str, Any]] = []
    original_error: BaseException | None = None
    try:
        with lock:
            json.dump({"pid": os.getpid(), "run_id": plan["run_id"]}, lock)
            lock.flush()
            suffix = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex
            archive = _direct_path(
                archive_parent / f"{plan['run_id']}-{suffix}", within=archive_parent
            )
            archive.mkdir(mode=0o700, exist_ok=False)
            journal_path = archive / "archive_manifest.jsonl"
            with journal_path.open("x", encoding="utf-8") as journal:
                _append_event(journal, {"event": "archive_planned", **plan})
                try:
                    for target in plan["targets"]:
                        _require_inactive(plan)
                        source = _direct_path(target["original"], within=results_root)
                        if _metadata_hashes(source) != target["metadata_sha256"]:
                            raise ArchiveError(f"run metadata changed before rename: {source}")
                        destination = _direct_path(archive / target["track"], within=archive)
                        if destination.exists() or source.stat().st_dev != archive.stat().st_dev:
                            raise ArchiveError(
                                "archive destination exists or is on another filesystem"
                            )
                        move = {
                            "track": target["track"],
                            "original": str(source),
                            "destination": str(destination),
                            "metadata_sha256": target["metadata_sha256"],
                        }
                        _append_event(journal, {"event": "move_started", **move})
                        _rename_no_replace(source, destination)
                        moves.append(move)
                        _append_event(journal, {"event": "move_completed", **move})
                    result = {
                        "status": "archived",
                        "archive": str(archive),
                        "journal": str(journal_path),
                        "moves": moves,
                        "next_run": plan["next_run"],
                    }
                    _append_event(journal, {"event": "archive_completed", **result})
                    return result
                except BaseException as error:
                    failure = {
                        "event": "archive_failed",
                        "error": f"{type(error).__name__}: {error}",
                        "archive": str(archive),
                        "completed_moves": moves,
                        "unmoved": [
                            target
                            for target in plan["targets"]
                            if target["track"] not in {move["track"] for move in moves}
                        ],
                        "recovery": (
                            "No automatic rollback or deletion. Preserve this journal. "
                            "Restore each "
                            "destination to its recorded original only after checking no active "
                            "process and that the original is absent; never overwrite either path."
                        ),
                    }
                    print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
                    try:
                        _append_event(journal, failure)
                    except OSError as reporting_error:
                        print(
                            f"Could not append failure journal: {reporting_error}",
                            file=sys.stderr,
                            flush=True,
                        )
                    raise
    except BaseException as error:
        original_error = error
        raise
    finally:
        # This is only the exclusive lock created by this invocation, never user artifacts.
        try:
            lock_path.unlink()
        except OSError as cleanup_error:
            if original_error is None:
                raise
            print(f"Archive lock cleanup also failed: {cleanup_error}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = plan_archive(args.run_id, args.results_root)
        result = (
            apply_archive(plan)
            if args.apply
            else {
                "status": "dry_run",
                **plan,
                "active_process_check": "required before --apply",
                "apply_supported_on_this_host": sys.platform.startswith("linux"),
            }
        )
    except (ArchiveError, OSError, ValueError) as error:
        print(f"Archive refused/failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
