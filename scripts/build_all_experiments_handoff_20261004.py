"""Build a fresh, checked GPT experiment archive; never run scientific jobs."""

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

ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs/gpt_experiments_20261004"
DOCS = (
    "00_READ_FIRST.md", "01_RESEARCH_AND_MATH.md", "04_LOCAL_HISTORY.md",
    "02_WEDGE_HISTORY.md", "03_EARLIER_HISTORY.md", "06_EVIDENCE_AND_LIMITS.md",
    "05_GPT_REVIEW_PROMPT.md",
)
LEGACY_DIRS = (
    "experiments/information_flow_v2", "gpt_handoff_v2_20260928",
    "gpt_handoff_v2_0_20260928", "gpt_handoff_bracket_20260929",
    "gpt_handoff_bracket_coverage_20260929", "gpt_handoff_bracket_reviewfix_20260929",
    "gpt_handoff_bracket_signal_20260929", "gpt_handoff_output_init_20260929",
    "gpt_handoff_output_init_server_20260929",
)
HISTORY_SUFFIXES = {".py", ".md", ".json", ".txt", ".csv", ".toml", ".yml", ".yaml"}
SECOND_LAYER = """CiteSeer raw accuracy -0.02 [-0.075528, +0.035528]
CiteSeer rms accuracy +0.02 [-0.21884, +0.25884]
Cora raw accuracy +0 [-0.087798, +0.087798]
Cora rms accuracy +0.14 [+0.071995, +0.20801]
PubMed raw accuracy +0.18 [+0.018106, +0.34189]
PubMed rms accuracy +0.079999 [-0.26455, +0.42455]
CiteSeer raw ce -0.00011344 [-0.00041775, +0.00019087]
CiteSeer rms ce -0.00076339 [-0.0012577, -0.00026909]
Cora raw ce -0.00015439 [-0.00041737, +0.00010859]
Cora rms ce -0.0005968 [-0.00085903, -0.00033457]
PubMed raw ce -0.0012165 [-0.0014868, -0.00094613]
PubMed rms ce -0.0013966 [-0.001688, -0.0011052]
"""
LOCAL_TABLES = """dataset C stage | shared_mean node_mean distinct_mean distinct_negative%
CiteSeer local_degree H0 H0 | 0.0175487 0.0793635 0.044266 0.00%
CiteSeer local_degree H0 H1 | 0.0170798 0.0752123 0.0410526 0.00%
CiteSeer local_degree H1 H1 | 0.0166503 0.0717627 0.0384622 0.00%
CiteSeer local_degree H1 H2 | 0.0162488 0.0687542 0.0362567 0.00%
CiteSeer local_degree H2 H2 | 0.0158694 0.066053 0.0343143 0.00%
CiteSeer unit H0 H0 | 0.140585 1.31141 1.03024 0.00%
CiteSeer unit H0 H1 | 0.132259 1.12477 0.86025 0.00%
CiteSeer unit H1 H1 | 0.125332 0.999063 0.748399 0.00%
CiteSeer unit H1 H2 | 0.119304 0.906754 0.668145 0.00%
CiteSeer unit H2 H2 | 0.113915 0.834134 0.606303 0.00%
Cora local_degree H0 H0 | 0.0322782 0.188575 0.124019 0.00%
Cora local_degree H0 H1 | 0.0315021 0.179743 0.116739 0.01%
Cora local_degree H1 H1 | 0.0308091 0.17253 0.110912 0.02%
Cora local_degree H1 H2 | 0.0301711 0.166323 0.105981 0.02%
Cora local_degree H2 H2 | 0.0295733 0.160804 0.101657 0.04%
Cora unit H0 H0 | 0.377164 3.97496 3.22063 0.00%
Cora unit H0 H1 | 0.361763 3.40933 2.68581 0.01%
Cora unit H1 H1 | 0.348963 3.04107 2.34314 0.02%
Cora unit H1 H2 | 0.33779 2.78088 2.1053 0.04%
Cora unit H2 H2 | 0.32774 2.58378 1.9283 0.06%
PubMed local_degree H0 H0 | 0.00650064 0.0685208 0.0555196 0.03%
PubMed local_degree H0 H1 | 0.00624844 0.0633463 0.0508495 0.04%
PubMed local_degree H1 H1 | 0.00601977 0.058877 0.0468375 0.05%
PubMed local_degree H1 H2 | 0.00581026 0.0549491 0.0433286 0.07%
PubMed local_degree H2 H2 | 0.00561732 0.0514678 0.0402332 0.09%
PubMed unit H0 H0 | 0.176202 3.16065 2.80824 0.03%
PubMed unit H0 H1 | 0.163558 2.76818 2.44107 0.04%
PubMed unit H1 H1 | 0.152598 2.45437 2.14918 0.05%
PubMed unit H1 H2 | 0.142948 2.19582 1.90993 0.07%
PubMed unit H2 H2 | 0.134387 1.97921 1.71044 0.09%

dataset C state | omitted_divergence% boundary_flow% omitted_flow%
Cora unit H0 | 17.43% 53.11% 31.24%
Cora unit H1 | 27.68% 50.82% 32.79%
Cora unit H2 | 35.50% 49.85% 33.37%
Cora local_degree H0 | 46.89% 28.15% 39.38%
Cora local_degree H1 | 47.28% 28.06% 39.47%
Cora local_degree H2 | 47.41% 28.01% 39.45%
CiteSeer unit H0 | 17.45% 55.50% 20.90%
CiteSeer unit H1 | 26.50% 53.64% 21.54%
CiteSeer unit H2 | 32.29% 53.06% 21.54%
CiteSeer local_degree H0 | 33.11% 32.69% 21.03%
CiteSeer local_degree H1 | 32.79% 32.61% 20.36%
CiteSeer local_degree H2 | 32.43% 32.54% 19.77%
PubMed unit H0 | 19.88% 61.91% 27.30%
PubMed unit H1 | 23.85% 61.86% 27.13%
PubMed unit H2 | 26.87% 62.14% 26.73%
PubMed local_degree H0 | 47.51% 30.04% 42.18%
PubMed local_degree H1 | 46.90% 30.58% 40.88%
PubMed local_degree H2 | 46.31% 31.04% 39.71%
"""


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attachments-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "results"):
        raise ValueError("Use a new output directory inside this repository's results folder")
    commit = git("rev-parse", "HEAD").decode().strip()
    git("diff", "--quiet", commit)
    tracked = [p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p]
    entries, origins, snapshots, mapping = {}, {}, {}, {}

    def add(name, data, category, origin):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name in entries:
            raise ValueError(f"Unsafe or duplicate archive name: {name}")
        entries[name], origins[name] = data, {"category": category, "origin": origin}

    def add_file(path, name, category, origin):
        data = path.read_bytes()
        snapshots[path] = sha(data)
        add(name, data, category, origin)

    # Include the complete tracked repository, including curated old result artifacts.
    # Ignore generated data, environments and server result directories; they are not tracked.
    for relative in tracked:
        destination = "repository/" + relative
        add_file(ROOT / relative, destination, "tracked_repository_snapshot", f"workspace@{commit}:{relative}")
        mapping[relative] = destination
    local_history = []
    for relative_dir in LEGACY_DIRS:
        folder = ROOT / relative_dir
        present = folder.is_dir()
        local_history.append({"folder": relative_dir, "present": present})
        if not present:
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

    catalog = json.loads((DOC_ROOT / "evidence_catalog.json").read_text(encoding="utf-8"))
    legacy = json.loads((ROOT / catalog["legacy_catalog"]).read_text(encoding="utf-8"))
    attachments = legacy["attachments"] + catalog["extra_attachments"]
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
    conversation_header = "Source: direct user message, transcribed; not original per-seed CSV.\n\n"
    add("evidence/conversation/wedge_second_layer_identity.txt",
        (conversation_header + SECOND_LAYER).encode(), "conversation_transcription", "User's 12 second-layer C identity comparison rows")
    add("evidence/conversation/local_correlations_and_omissions.txt",
        (conversation_header + LOCAL_TABLES).encode(), "conversation_transcription", "User's 30 local relation and 18 omitted/boundary rows")
    add_file(ROOT / "research/wedge_propagation/gpt_handoff/fixed_completion_user_message.txt",
             "evidence/conversation/wedge_fixed_completion.txt", "conversation_transcription", "Previously preserved direct user fixed study completion message")

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
                destination = "repository/" + relative.rstrip("/") + "/"
                if not resolved.is_dir() or not any(name.startswith(destination) for name in entries):
                    raise ValueError(f"Missing review link: {filename}: {target}")
            links.append({"from": filename, "to": destination})
            return f"[{label}]({destination}{sep}{fragment})"

        content = re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", rewrite, original.decode("utf-8-sig"))
        add(filename, content.encode("utf-8"), "review_document_with_archive_links", source.relative_to(ROOT).as_posix())
    add("REVIEW_ALL.md", b"\n\n---\n\n".join(entries[name] for name in DOCS), "combined_review", "Seven documents in reading order")
    add("GPT_REVIEW_PROMPT.md", entries["05_GPT_REVIEW_PROMPT.md"], "review_prompt", "Same as 05_GPT_REVIEW_PROMPT.md")
    add("EVIDENCE_INDEX.json", encode({"attachments": evidence, "local_history": local_history,
        "recent_server_raw_checkpoints_included": False, "recent_server_seed_csv_included": False,
        "direct_message_transcriptions": [name for name in entries if name.startswith("evidence/conversation/")]}), "evidence_index", "Declared evidence catalog and conversation transcriptions")

    required = [
        *("repository/research/wedge_propagation/" + name for name in (
            "SERVER_FIXED_RESULTS.md", "SERVER_LEARNED_RESULTS.md", "SERVER_GENERALIZATION_RESULTS.md",
            "SERVER_SCALE_NORMALIZATION_RESULTS.md", "SERVER_CLASSIFICATION_RESULTS.md",
            "SERVER_BRANCH_STRENGTH_RESULTS.md", "SERVER_BRANCH_ANALYSIS_RESULTS.md", "SERVER_NODE_NORMALIZATION_RESULTS.md")),
        *("repository/docs/evidence/" + name for name in (
            "local_energy_server_full_20261004.txt", "receiver_aggregation_server_full_20261004.txt",
            "local_prediction_server_full_20261004.txt", "local_prediction_layer_removal_server_20261004.txt",
            "local_placement_server_full_20261004.txt")),
        "repository/research/local_energy_relations/placement/model.py",
        "repository/research/local_energy_relations/prediction/operators.py",
        "repository/research/local_energy_relations/placement/config_full.json",
        "repository/gpt_handoff/EXPERIMENT_STATUS.md", *DOCS,
    ]
    missing = set(required) - entries.keys()
    if missing:
        raise ValueError(f"Required experiment coverage missing: {sorted(missing)}")
    python_count = json_count = 0
    for name, data in entries.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
            python_count += 1
        elif name.endswith(".json"):
            json.loads(data)
            json_count += 1
    for check in links:
        target = check["to"]
        if target not in entries and not any(name.startswith(target.rstrip("/") + "/") for name in entries):
            raise ValueError(f"Broken archive review link: {check}")
    changed = [str(p) for p, digest in snapshots.items() if sha(p.read_bytes()) != digest]
    if changed:
        raise ValueError(f"Source changed during packaging: {changed}")
    checks = {"created_utc": datetime.now(UTC).isoformat(), "source_commit": commit,
        "status": "passed", "tracked_repository_files": len(tracked), "required_coverage_count": len(required),
        "python_ast_parsed": python_count, "json_parsed": json_count, "summary_links_checked": len(links),
        "included_attachments": sum(i["included"] for i in evidence), "catalog_attachments": len(evidence),
        "source_and_evidence_unchanged": True, "zip_crc_and_sha256_verified_after_write": True,
        "new_scientific_training_run": False, "new_scientific_evaluation_run": False,
        "server_artifacts_independently_reevaluated": False, "regression_tests_rerun_for_packaging": False}
    add("PACKAGE_CHECKS.json", encode(checks), "package_check", "Package validation only")
    manifest = {"source_commit": commit, "file_count_excluding_manifest": len(entries),
        "category_counts": dict(Counter(item["category"] for item in origins.values())),
        "files": {name: {"sha256": sha(data), "bytes": len(data), **origins[name]} for name, data in sorted(entries.items())}}
    add("MANIFEST.json", encode(manifest), "manifest", "SHA256 of all other archive entries")
    output.mkdir(parents=True, exist_ok=False)
    zip_path = output / "GPT_ALL_EXPERIMENTS_20261004.zip"
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
    for name in (*DOCS, "REVIEW_ALL.md", "GPT_REVIEW_PROMPT.md", "MANIFEST.json", "PACKAGE_CHECKS.json", "EVIDENCE_INDEX.json"):
        with (output / name).open("xb") as stream:
            stream.write(entries[name])
    with (output / "ZIP_SHA256.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"{sha(zip_path.read_bytes())}  {zip_path.name}\n")
    print(json.dumps({"zip": str(zip_path), "bytes": zip_path.stat().st_size,
                      "files": len(entries), "source_commit": commit, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
