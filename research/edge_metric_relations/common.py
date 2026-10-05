"""Shared provenance and exclusive artifact writers for the new research track."""
from __future__ import annotations

from pathlib import Path

from ..local_context_coupling.classification.common import (
    cpu_state, digest, file_sha256, read_json, save_checkpoint, write_csv, write_json,
    source_manifest as _input_source_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent


def source_manifest():
    inherited = _input_source_manifest()
    hashes = dict(inherited["sha256"])
    for name in ("__init__.py","data.py"):
        path=ROOT/"research"/"wedge_propagation"/"learned"/name
        hashes[path.relative_to(ROOT).as_posix()]=file_sha256(path)
    for path in sorted([*FOLDER.rglob("*.py"), *FOLDER.rglob("*.json")]):
        if "__pycache__" not in path.parts:
            hashes[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return {"sha256": hashes, "code_digest": digest(hashes),
            "git_commit": inherited["git_commit"], "data_reader_digest": inherited["code_digest"]}


def assert_source_unchanged(record):
    if source_manifest()["sha256"] != record["sha256"]:
        raise RuntimeError("Scientific source changed; existing results are preserved")
