#!/usr/bin/env python3
"""Generate the reviewable ``# path`` + exact source ``code_summary.md`` snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "code_summary.md"

SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".sh", ".ps1"}
EXCLUDED_PARTS = {
    ".git",
    ".agents",
    ".codex",
    ".matplotlib",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "results",
    "runs",
}
LANGUAGES = {
    ".py": "python",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".ps1": "powershell",
    ".txt": "text",
}


def _excluded(path: Path, *, root: Path) -> bool:
    relative = path.relative_to(root)
    return any(
        part in EXCLUDED_PARTS or part.startswith(".venv") or part.endswith(".egg-info")
        for part in relative.parts
    )


def _is_source(path: Path, *, root: Path) -> bool:
    if not path.is_file() or _excluded(path, root=root):
        return False
    if path.name == ".gitignore":
        return True
    if path.suffix in SOURCE_SUFFIXES:
        return True
    return path.suffix == ".txt" and path.name.startswith(("requirements", "constraints"))


def discover_sources(root: Path = PROJECT_ROOT) -> list[Path]:
    """Return the deterministic set of human-authored code/configuration sources."""

    return sorted(
        (path for path in root.rglob("*") if _is_source(path, root=root)),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def render_summary(root: Path = PROJECT_ROOT) -> tuple[str, list[str]]:
    """Render sources with normalized LF separators and no content omissions."""

    sections: list[str] = []
    relative_paths: list[str] = []
    for path in discover_sources(root):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if source.endswith("\n"):
            source = source[:-1]
        language = LANGUAGES.get(path.suffix, "text")
        sections.append(f"# {relative}\n\n````{language}\n{source}\n````")
        relative_paths.append(relative)
    return "\n\n".join(sections) + "\n", relative_paths


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _report(content: str, sources: list[str], *, status: str) -> dict[str, object]:
    encoded = content.encode("utf-8")
    return {
        "status": status,
        "output": str(OUTPUT_PATH),
        "source_files": len(sources),
        "bytes": len(encoded),
        "lines": len(content.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest().upper(),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if code_summary.md does not exactly match the current selected sources",
    )
    parser.add_argument("--json", action="store_true", help="include the selected source list")
    args = parser.parse_args()

    content, sources = render_summary()
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else None
        matches = current == content
        report = _report(content, sources, status="current" if matches else "stale")
        if not args.json:
            report.pop("sources")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if matches else 1

    _atomic_write(OUTPUT_PATH, content)
    report = _report(content, sources, status="written")
    if not args.json:
        report.pop("sources")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
