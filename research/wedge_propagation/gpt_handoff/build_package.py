"""Create a new, verified GPT handoff without running scientific experiments.

Run after committing the handoff. Existing source, evidence and outputs are read-only.
The attachment directory contains user-supplied source material, not instructions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
DOC_ROOT = Path(__file__).resolve().parent
WEDGE = "research/wedge_propagation/"
HANDOFF = WEDGE + "gpt_handoff/"
SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".sh", ".ps1"}
DOCS = (
    "00_READ_FIRST.md",
    "01_RESEARCH_STATUS.md",
    "02_MODEL_AND_MATH.md",
    "03_EXPERIMENTS_AND_RESULTS.md",
    "04_GPT_REVIEW_PROMPT.md",
    "05_EVIDENCE_AND_LIMITS.md",
    "06_PRIOR_TRACK_HISTORY.md",
)
SECOND_LAYER = (
    ("CiteSeer", "raw", "accuracy", "-0.02", "-0.075528", "0.035528"),
    ("CiteSeer", "rms", "accuracy", "0.02", "-0.21884", "0.25884"),
    ("Cora", "raw", "accuracy", "0", "-0.087798", "0.087798"),
    ("Cora", "rms", "accuracy", "0.14", "0.071995", "0.20801"),
    ("PubMed", "raw", "accuracy", "0.18", "0.018106", "0.34189"),
    ("PubMed", "rms", "accuracy", "0.079999", "-0.26455", "0.42455"),
    ("CiteSeer", "raw", "ce", "-0.00011344", "-0.00041775", "0.00019087"),
    ("CiteSeer", "rms", "ce", "-0.00076339", "-0.0012577", "-0.00026909"),
    ("Cora", "raw", "ce", "-0.00015439", "-0.00041737", "0.00010859"),
    ("Cora", "rms", "ce", "-0.0005968", "-0.00085903", "-0.00033457"),
    ("PubMed", "raw", "ce", "-0.0012165", "-0.0014868", "-0.00094613"),
    ("PubMed", "rms", "ce", "-0.0013966", "-0.001688", "-0.0011052"),
)


def git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT
    )


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory only")
    parser.add_argument("--attachments-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing handoff: {output}")
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Handoff output must stay in this repository's results directory")
    commit = git("rev-parse", "HEAD").decode().strip()
    git("diff", "--quiet", commit)
    tracked = [p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p]
    entries: dict[str, bytes] = {}
    origins: dict[str, dict] = {}
    snapshots: dict[Path, str] = {}
    mapping: dict[str, str] = {}

    def add(name: str, data: bytes, category: str, origin: str) -> None:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name in entries:
            raise ValueError(f"Unsafe or duplicate archive name: {name}")
        entries[name] = data
        origins[name] = {"category": category, "origin": origin}

    def add_file(path: Path, name: str, category: str, origin: str) -> None:
        data = path.read_bytes()
        snapshots[path] = sha(data)
        add(name, data, category, origin)

    # Snapshot exact workspace bytes. Git identity is separate from file-byte SHA256.
    # Scientific source and full budgets are unchanged. Historical files have their own tree.
    selected = []
    for relative in tracked:
        path = PurePosixPath(relative)
        if relative.startswith(HANDOFF):
            continue
        if "results" in path.parts:
            if path.suffix in {".json", ".csv"}:
                selected.append((relative, "history/repository_evidence/" + relative))
            continue
        if path.suffix not in SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
            continue
        current = (
            relative.startswith(WEDGE)
            or relative.startswith("tests/test_wedge_")
            or relative
            in {
                "tests/conftest.py",
                "research/__init__.py",
                "AGENTS.md",
                "pyproject.toml",
                "docs/GETTING_STARTED.md",
            }
            or (len(path.parts) == 1 and path.suffix == ".txt")
            or relative.startswith("src/chartgat/")
        )
        destination = ("code/" if current else "history/code/") + relative
        selected.append((relative, destination))
        mapping[relative] = destination
    for relative, destination in selected:
        category = "current_source" if destination.startswith("code/") else "historical_repository"
        add_file(ROOT / relative, destination, category, f"workspace@{commit}:{relative}")

    # Initial information-flow v2 survived as an untracked local source tree.
    # Preserve that fact in the manifest rather than attributing it to HEAD.
    initial = ROOT / "experiments/information_flow_v2"
    if initial.is_dir():
        for path in sorted(initial.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            destination = "history/code/" + relative
            if destination not in entries:
                add_file(path, destination, "historical_untracked_source", relative)
                mapping[relative] = destination

    # Original delivered explanations retain the old version's wording and are clearly historical.
    for folder in ("gpt_handoff_v2_20260928", "gpt_handoff_v2_0_20260928"):
        base = ROOT / folder
        if not base.is_dir():
            raise FileNotFoundError(f"Missing historical delivery: {base}")
        for path in sorted(base.iterdir()):
            if not path.is_file() or path.suffix not in {".md", ".json", ".py", ".txt"}:
                continue
            add_file(
                path,
                f"history/original_deliveries/{folder}/{path.name}",
                "historical_delivered_document",
                path.relative_to(ROOT).as_posix(),
            )

    # Rewrite only the seven new summary documents' relative Markdown links for this ZIP layout.
    # Actual scientific source/docs above retain their original bytes and source hashes.
    link_checks = []
    for filename in DOCS:
        source_path = DOC_ROOT / filename
        original = source_path.read_bytes()
        snapshots[source_path] = sha(original)

        def link(match: re.Match[str], source_filename: str = filename) -> str:
            label, target = match.groups()
            if "://" in target or target.startswith("#"):
                return match.group(0)
            target = target.strip("<>")
            plain, sep, fragment = target.partition("#")
            resolved = (DOC_ROOT / plain).resolve()
            if resolved.parent == DOC_ROOT and resolved.name in DOCS:
                destination = resolved.name
            else:
                relative = resolved.relative_to(ROOT).as_posix()
                if relative not in mapping:
                    raise ValueError(
                        f"Summary link missing from package: {source_filename}: {target}"
                    )
                destination = mapping[relative]
            link_checks.append({"from": source_filename, "to": destination})
            return f"[{label}]({destination}{sep}{fragment})"

        content = re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", link, original.decode("utf-8-sig"))
        add(
            filename,
            content.encode("utf-8"),
            "review_summary",
            source_path.relative_to(ROOT).as_posix(),
        )
        add(
            "code/" + HANDOFF + filename,
            original,
            "review_summary_original",
            source_path.relative_to(ROOT).as_posix(),
        )
    add(
        "REVIEW_ALL.md",
        b"\n\n---\n\n".join(entries[f] for f in DOCS),
        "review_summary",
        "Concatenation of 00..06 documents; no code/evidence substitution",
    )
    add_file(
        DOC_ROOT / "build_package.py",
        "build_package.py",
        "package_tool",
        HANDOFF + "build_package.py",
    )
    add_file(
        DOC_ROOT / "evidence_catalog.json",
        "evidence_catalog.json",
        "package_tool",
        HANDOFF + "evidence_catalog.json",
    )
    add_file(
        DOC_ROOT / "fixed_completion_user_message.txt",
        "evidence/fixed_completion_user_message.txt",
        "user_message_transcription",
        "User-provided full fixed study terminal completion in conversation",
    )

    evidence = []
    catalog = json.loads((DOC_ROOT / "evidence_catalog.json").read_text(encoding="utf-8"))
    for item in catalog["attachments"]:
        source = args.attachments_root / item["id"] / "붙여넣은 텍스트.txt"
        if not source.is_file():
            if item["required"]:
                raise FileNotFoundError(f"Required original evidence unavailable: {item['id']}")
            evidence.append({**item, "included": False, "reason": "Attachment not present locally"})
            continue
        destination = f"evidence/server/{item['label']}__{item['id']}.txt"
        add_file(source, destination, item["kind"], f"user_attachment:{item['id']}")
        evidence.append(
            {**item, "included": True, "path": destination, "sha256": sha(entries[destination])}
        )

    table = io.StringIO(newline="")
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(
        [
            "dataset",
            "gate",
            "metric",
            "delta",
            "ci95_low",
            "ci95_high",
            "scope",
            "intervention",
            "unit",
            "source",
        ]
    )
    transcript = []
    for dataset, gate, metric, value, lo, hi in SECOND_LAYER:
        unit = "percentage_points" if metric == "accuracy" else "loss"
        writer.writerow(
            [
                dataset,
                gate,
                metric,
                value,
                lo,
                hi,
                "layer_1",
                "c_identity_norm_matched",
                unit,
                "user_message_transcription",
            ]
        )
        transcript.append(f"{dataset} {gate} {metric} {value} [{lo}, {hi}]")
    add(
        "evidence/second_layer_identity.csv",
        table.getvalue().encode(),
        "user_message_transcription",
        "Latest user-provided 12 summary rows; not per-seed raw CSV",
    )
    add(
        "evidence/second_layer_identity.txt",
        (
            "Source: latest user message, transcribed numerically; delta=replacement-original.\n\n"
            + "\n".join(transcript)
            + "\n"
        ).encode(),
        "user_message_transcription",
        "Latest user message",
    )
    add(
        "EVIDENCE_INDEX.json",
        json_bytes(
            {
                "attachments": evidence,
                "second_layer_summary_rows": 12,
                "fixed_completion_source": "evidence/fixed_completion_user_message.txt",
                "server_raw_checkpoints_included": False,
                "server_complete_seed_csv_included": False,
            }
        ),
        "evidence_index",
        "Explicit attachment catalog and last user message",
    )

    python_count = 0
    json_count = 0
    for name, data in entries.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
            python_count += 1
        elif name.endswith(".json"):
            json.loads(data)
            json_count += 1
    required = [
        *(
            "code/" + WEDGE + name
            for name in (
                "operators.py",
                "SERVER_FIXED_RESULTS.md",
                "SERVER_LEARNED_RESULTS.md",
                "SERVER_GENERALIZATION_RESULTS.md",
                "SERVER_SCALE_NORMALIZATION_RESULTS.md",
                "SERVER_CLASSIFICATION_RESULTS.md",
                "SERVER_BRANCH_STRENGTH_RESULTS.md",
                "SERVER_BRANCH_ANALYSIS_RESULTS.md",
                "SERVER_NODE_NORMALIZATION_RESULTS.md",
            )
        ),
        *(
            "code/" + WEDGE + package + "/" + filename
            for package, filename in (
                ("learned", "config_full.json"),
                ("generalization", "config_full.json"),
                ("scale_normalization", "config_full.json"),
                ("classification", "design_contract.json"),
                ("classification", "config_full.json"),
                ("branch_strength", "config_full.json"),
                ("branch_analysis", "core.py"),
                ("node_normalization", "config_full.json"),
            )
        ),
        *DOCS,
        "04_GPT_REVIEW_PROMPT.md",
        "evidence/second_layer_identity.csv",
    ]
    missing = set(required) - entries.keys()
    if missing:
        raise ValueError(f"Required review coverage missing: {sorted(missing)}")
    for checked in link_checks:
        if checked["to"] not in entries:
            raise ValueError(f"Broken summary link: {checked}")
    changed = [str(path) for path, digest in snapshots.items() if sha(path.read_bytes()) != digest]
    if changed:
        raise ValueError(f"Source/evidence changed during packaging: {changed}")
    checks = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "status": "passed",
        "required_coverage_count": len(set(required)),
        "python_ast_parsed": python_count,
        "json_parsed": json_count,
        "summary_links_checked": len(link_checks),
        "source_and_evidence_unchanged": True,
        "required_attachment_count": sum(i["required"] for i in evidence),
        "included_attachment_count": sum(i["included"] for i in evidence),
        "zip_crc_and_manifest_verified_by_builder_after_write": True,
        "new_scientific_training_run": False,
        "new_scientific_evaluation_run": False,
        "server_artifacts_independently_reevaluated": False,
        "prior_regression_tests_rerun_for_packaging": False,
    }
    add(
        "PACKAGE_CHECKS.json",
        json_bytes(checks),
        "package_check",
        "Builder checks; not scientific model validation",
    )
    manifest = {
        "source_commit": commit,
        "file_count_excluding_manifest": len(entries),
        "category_counts": dict(Counter(o["category"] for o in origins.values())),
        "files": {
            name: {"sha256": sha(data), "bytes": len(data), **origins[name]}
            for name, data in sorted(entries.items())
        },
    }
    add("MANIFEST.json", json_bytes(manifest), "manifest", "SHA256 of all other archive entries")

    output.mkdir(parents=True, exist_ok=False)
    zip_path = output / "GPT_ALL_EXPERIMENTS_20261003.zip"
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("ZIP CRC/duplicate entry validation failed")
        for name, item in manifest["files"].items():
            if sha(archive.read(name)) != item["sha256"]:
                raise ValueError(f"ZIP SHA256 verification failed: {name}")
    # Expose the reading files next to the ZIP without making users extract to see the prompt.
    for name in (
        *DOCS,
        "REVIEW_ALL.md",
        "MANIFEST.json",
        "PACKAGE_CHECKS.json",
        "EVIDENCE_INDEX.json",
    ):
        with (output / name).open("xb") as stream:
            stream.write(entries[name])
    (output / "ZIP_SHA256.txt").write_text(
        f"{sha(zip_path.read_bytes())}  {zip_path.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "zip": str(zip_path),
                "bytes": zip_path.stat().st_size,
                "files": len(entries),
                "source_commit": commit,
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
