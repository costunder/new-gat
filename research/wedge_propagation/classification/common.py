"""Strict artifact I/O and provenance for classification runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty metric table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists; preserved: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-filesystem hard-link publication is atomic and refuses replacement.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def source_manifest():
    folder = Path(__file__).parent
    files = sorted([*folder.glob("*.py"), *folder.glob("*.json")])
    hashes = {file.name: file_sha256(file) for file in files}
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={folder.parents[2].as_posix()}", "rev-parse", "HEAD"],
            cwd=folder,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return {"sha256": hashes, "code_digest": digest(hashes), "git_commit": commit}
