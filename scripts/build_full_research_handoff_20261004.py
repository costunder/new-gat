"""Preserve research history and execution evidence in a fresh GPT review ZIP.

This script reads existing artifacts. It does not import scientific modules or
train, evaluate, select, or change any model or result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from build_all_experiments_handoff_20261004 import (
    HISTORY_SUFFIXES, LEGACY_DIRS, LOCAL_TABLES, SECOND_LAYER,
)

ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs/gpt_all_research_20261004"
DOCS = (
    "00_READ_FIRST.md", "01_THEORY_AND_MODELS.md", "02_EARLY_EXPERIMENTS.md",
    "03_WEDGE_EXPERIMENTS.md", "04_LOCAL_AND_COPY_EXPERIMENTS.md",
    "05_EVIDENCE_AND_OPEN_QUESTIONS.md", "06_GPT_REVIEW_PROMPT.md",
    "07_REFERENCES_AND_SOURCE_MAP.md",
)
LOCAL_TOP_SUFFIXES = {".md", ".json", ".txt", ".csv", ".log"}
LOCAL_JOB_FILES = {
    "completion.json", "epoch_history.csv", "selected.json",
    "selected_validation.json", "summary.json", "training_resource.json",
    "terminal.log", "hardware_samples.json", "config.json", "contract.json",
    "coverage.json", "hardware.json", "preflight_debug_checks.json",
}
EXCLUDED_RESULT_PREFIXES = ("gpt-", "pytest-", "debug-tests-")
SCIENTIFIC_BASELINE = "2439dd3"


def git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
    )


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attachments-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Use a new output directory inside this repository's results folder")
    commit = git("rev-parse", "HEAD").decode().strip()
    git("diff", "--quiet", commit)
    scientific_changes = git("diff", "--name-only", SCIENTIFIC_BASELINE, commit, "--",
                             "research", "experiments", "src", "tests", "configs").decode().splitlines()
    if scientific_changes:
        raise ValueError(f"Scientific source changed during this documentation task: {scientific_changes}")
    tracked = [p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p]
    entries, origins, snapshots, mapping = {}, {}, {}, {}

    def add(name, data, category, origin):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name in entries:
            raise ValueError(f"Unsafe or duplicate archive name: {name}")
        entries[name] = data
        origins[name] = {"category": category, "origin": origin}

    def add_file(path, name, category, origin):
        data = path.read_bytes()
        snapshots[path] = sha(data)
        add(name, data, category, origin)

    for relative in tracked:
        destination = "repository/" + relative
        add_file(ROOT / relative, destination, "tracked_repository_snapshot", f"workspace@{commit}:{relative}")
        mapping[relative] = destination

    local_history = []
    for relative_dir in LEGACY_DIRS:
        folder = ROOT / relative_dir
        record = {"folder": relative_dir, "present": folder.is_dir(), "included_files": 0}
        local_history.append(record)
        if not record["present"]:
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix not in HISTORY_SUFFIXES or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in mapping:
                continue
            destination = "local_history/" + relative
            add_file(path, destination, "untracked_historical_snapshot", relative)
            mapping[relative] = destination
            record["included_files"] += 1

    local_runs, excluded_result_dirs = [], []
    for folder in sorted((ROOT / "results").iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith(EXCLUDED_RESULT_PREFIXES):
            excluded_result_dirs.append({"folder": folder.name, "reason": "Previous packaging or temporary test output"})
            continue
        record = {
            "folder": folder.relative_to(ROOT).as_posix(),
            "role": "Existing local artifact; classify using its config/completion and research documents",
            "top_level_completion_present": (folder / "completion.json").is_file(),
            "included": [], "excluded": [],
        }
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            tail = path.relative_to(folder)
            if relative in mapping:
                record["included"].append({"source": relative, "archive": mapping[relative], "policy": "Already tracked repository artifact"})
                continue
            include = ((len(tail.parts) == 1 and path.suffix.lower() in LOCAL_TOP_SUFFIXES)
                       or path.name in LOCAL_JOB_FILES)
            if include and "__pycache__" not in path.parts:
                destination = "evidence/local_runs/" + folder.name + "/" + tail.as_posix()
                add_file(path, destination, "local_execution_evidence", relative)
                mapping[relative] = destination
                record["included"].append({"source": relative, "archive": destination})
            else:
                record["excluded"].append({
                    "source": relative, "bytes": path.stat().st_size,
                    "reason": "Untracked binary/data/figure/cache or intermediate file outside declared text-evidence policy",
                })
        local_runs.append(record)
    add("LOCAL_RUN_INDEX.json", encode({
        "scope": "Read-only preservation of local execution evidence; no new scientific execution",
        "top_level_suffixes": sorted(LOCAL_TOP_SUFFIXES), "nested_job_basenames": sorted(LOCAL_JOB_FILES),
        "runs": local_runs, "excluded_directories": excluded_result_dirs,
        "untracked_binary_checkpoints_fully_copied": False,
        "no_file_size_or_row_count_cap": True,
    }), "local_run_index", "Explicit local artifact inclusion and exclusion policy")

    catalog = json.loads((DOC_ROOT / "evidence_catalog.json").read_text(encoding="utf-8"))
    legacy = json.loads((ROOT / catalog["legacy_catalog"]).read_text(encoding="utf-8"))
    prior = json.loads((ROOT / catalog["previous_catalog"]).read_text(encoding="utf-8"))
    attachments = legacy["attachments"] + prior["extra_attachments"] + catalog["extra_attachments"]
    if len({item["id"] for item in attachments}) != len(attachments):
        raise ValueError("Duplicate attachment IDs in evidence catalog")
    evidence = []
    for item in attachments:
        source = args.attachments_root / item["id"] / "붙여넣은 텍스트.txt"
        if not source.is_file():
            if item["required"]:
                raise FileNotFoundError(f"Required original attachment unavailable: {item['id']}")
            evidence.append({**item, "included": False, "reason": "Not present on this PC"})
            continue
        destination = f"evidence/attachments/{item['label']}__{item['id']}.txt"
        add_file(source, destination, item["kind"], f"user_attachment:{item['id']}")
        evidence.append({**item, "included": True, "path": destination, "sha256": sha(entries[destination])})

    header = "Source: direct user message, transcribed; not original per-seed CSV.\n\n"
    add("evidence/conversation/wedge_second_layer_identity.txt", (header + SECOND_LAYER).encode(),
        "conversation_transcription", "User's 12 second-layer C identity comparison rows")
    add("evidence/conversation/local_correlations_and_omissions.txt", (header + LOCAL_TABLES).encode(),
        "conversation_transcription", "User's 30 local relation and 18 omitted/boundary rows")
    add_file(ROOT / "research/wedge_propagation/gpt_handoff/fixed_completion_user_message.txt",
             "evidence/conversation/wedge_fixed_completion.txt", "conversation_transcription",
             "Previously preserved direct user fixed-study completion message")

    links = []
    for filename in DOCS:
        source = DOC_ROOT / filename
        if source.relative_to(ROOT).as_posix() not in tracked:
            raise ValueError(f"Commit new review documents before packaging: {filename}")
        original = source.read_bytes()

        def rewrite(match):
            label, target = match.groups()
            if "://" in target or target.startswith("#"):
                return match.group(0)
            plain, sep, fragment = target.strip("<>").partition("#")
            resolved = (DOC_ROOT / plain).resolve()
            if not resolved.is_relative_to(ROOT):
                raise ValueError(f"Review link leaves repository: {target}")
            relative = resolved.relative_to(ROOT).as_posix()
            if resolved.parent == DOC_ROOT and resolved.name in DOCS:
                destination = resolved.name
            elif relative in mapping:
                destination = mapping[relative]
            else:
                candidates = ("repository/" + relative.rstrip("/") + "/",
                              "local_history/" + relative.rstrip("/") + "/")
                destination = next((prefix for prefix in candidates if any(name.startswith(prefix) for name in entries)), None)
                if not resolved.is_dir() or destination is None:
                    raise ValueError(f"Missing review link: {filename}: {target}")
            links.append({"from": filename, "to": destination})
            return f"[{label}]({destination}{sep}{fragment})"

        content = re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", rewrite, original.decode("utf-8-sig"))
        add(filename, content.encode("utf-8"), "review_document_with_archive_links", source.relative_to(ROOT).as_posix())
    add("REVIEW_ALL.md", b"\n\n---\n\n".join(entries[name] for name in DOCS),
        "combined_review", "Review documents in declared reading order")
    add("GPT_REVIEW_PROMPT.md", entries["06_GPT_REVIEW_PROMPT.md"], "review_prompt", "Same as 06_GPT_REVIEW_PROMPT.md")
    add("EVIDENCE_INDEX.json", encode({
        "attachments": evidence, "local_history": local_history,
        "recent_server_raw_checkpoints_included": False, "recent_server_seed_csv_included": False,
        "shared_chat_full_bodies_retrieved": False,
        "direct_message_transcriptions": [name for name in entries if name.startswith("evidence/conversation/")],
        "local_execution_index": "LOCAL_RUN_INDEX.json",
    }), "evidence_index", "Declared evidence catalog and conversation transcriptions")

    required = [*DOCS, "repository/AGENTS.md", "repository/gpt_handoff/EXPERIMENT_STATUS.md",
        "repository/research/wedge_propagation/SERVER_FIXED_RESULTS.md",
        "repository/research/wedge_propagation/SERVER_LEARNED_RESULTS.md",
        "repository/research/wedge_propagation/SERVER_GENERALIZATION_RESULTS.md",
        "repository/research/wedge_propagation/SERVER_SCALE_NORMALIZATION_RESULTS.md",
        "repository/research/wedge_propagation/SERVER_CLASSIFICATION_RESULTS.md",
        "repository/research/wedge_propagation/SERVER_BRANCH_STRENGTH_RESULTS.md",
        "repository/research/wedge_propagation/SERVER_BRANCH_ANALYSIS_RESULTS.md",
        "repository/research/wedge_propagation/SERVER_NODE_NORMALIZATION_RESULTS.md",
        "repository/research/local_energy_relations/prediction/operators.py",
        "repository/research/local_energy_relations/placement/config_full.json",
        "repository/research/local_context_coupling/classification/model.py",
        "repository/research/local_context_coupling/normalization/model.py",
        "repository/research/local_context_coupling/normalization/operators.py",
        "repository/research/local_context_coupling/normalization/config_full.json",
        "repository/research/local_context_coupling/normalization/VERIFICATION.md",
        "evidence/local_runs/local-context-normalization-DEBUG-20261004-01/completion.json",
        "evidence/local_runs/local-context-normalization-resume-DEBUG-20261004-01/completion.json",
        *("repository/docs/evidence/" + name for name in (
            "local_energy_server_full_20261004.txt", "receiver_aggregation_server_full_20261004.txt",
            "local_prediction_server_full_20261004.txt", "local_prediction_layer_removal_server_20261004.txt",
            "local_placement_server_full_20261004.txt", "local_context_coupling_server_full_20261004.txt",
            "local_context_classification_server_full_20261004.txt")),
    ]
    missing = set(required) - entries.keys()
    if missing:
        raise ValueError(f"Required research coverage missing: {sorted(missing)}")
    python_count = json_count = 0
    invalid_evidence_json = []
    for name, data in entries.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
            python_count += 1
        elif name.endswith(".json"):
            try:
                json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if origins[name]["category"] != "local_execution_evidence":
                    raise
                invalid_evidence_json.append({"path": name, "error": str(error),
                    "preserved_bytes": True, "reason": "Existing execution/failure artifact; not a new source configuration"})
            else:
                json_count += 1
    for check in links:
        target = check["to"]
        if target not in entries and not any(name.startswith(target.rstrip("/") + "/") for name in entries):
            raise ValueError(f"Broken archive review link: {check}")
    changed = [str(path) for path, digest in snapshots.items() if sha(path.read_bytes()) != digest]
    if changed:
        raise ValueError(f"Source changed during packaging: {changed}")

    checks = {
        "created_utc": datetime.now(UTC).isoformat(), "source_commit": commit,
        "archive_validation_status": "passed", "tracked_repository_files": len(tracked),
        "required_coverage_count": len(required), "python_ast_parsed": python_count,
        "valid_json_parsed": json_count, "invalid_existing_evidence_json": invalid_evidence_json,
        "summary_links_checked": len(links),
        "included_attachments": sum(item["included"] for item in evidence),
        "catalog_attachments": len(evidence), "local_result_folders_indexed": len(local_runs),
        "local_execution_files_included": sum(item["category"] == "local_execution_evidence" for item in origins.values()),
        "source_and_evidence_unchanged": True, "zip_crc_and_sha256_verified_after_write": True,
        "scientific_baseline_commit": SCIENTIFIC_BASELINE,
        "scientific_source_changed_since_baseline": False,
        "new_scientific_training_run": False, "new_scientific_evaluation_run": False,
        "server_artifacts_independently_reevaluated": False, "regression_tests_rerun_for_packaging": False,
    }
    add("PACKAGE_CHECKS.json", encode(checks), "package_check", "Package validation only")
    manifest = {
        "source_commit": commit, "file_count_excluding_manifest": len(entries),
        "category_counts": dict(Counter(item["category"] for item in origins.values())),
        "files": {name: {"sha256": sha(data), "bytes": len(data), **origins[name]} for name, data in sorted(entries.items())},
    }
    add("MANIFEST.json", encode(manifest), "manifest", "SHA256 of all other archive entries")
    output.mkdir(parents=True, exist_ok=False)
    zip_path = output / "GPT_ALL_RESEARCH_20261004.zip"
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("ZIP integrity/unique-name validation failed")
        if set(archive.namelist()) != set(entries):
            raise ValueError("ZIP file coverage differs from manifest")
        for name, item in manifest["files"].items():
            if sha(archive.read(name)) != item["sha256"]:
                raise ValueError(f"ZIP SHA256 mismatch: {name}")
    for name in (*DOCS, "REVIEW_ALL.md", "GPT_REVIEW_PROMPT.md", "MANIFEST.json",
                 "PACKAGE_CHECKS.json", "EVIDENCE_INDEX.json", "LOCAL_RUN_INDEX.json"):
        with (output / name).open("xb") as stream:
            stream.write(entries[name])
    with (output / "ZIP_SHA256.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"{sha(zip_path.read_bytes())}  {zip_path.name}\n")
    print(json.dumps({"zip": str(zip_path), "bytes": zip_path.stat().st_size,
        "files": len(entries), "source_commit": commit, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
