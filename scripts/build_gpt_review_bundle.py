#!/usr/bin/env python3
"""Build a hash-verified review ZIP from committed text and recorded test evidence.

No model execution, network access, dataset, checkpoint or pretrained weight import.
Existing ZIPs are never overwritten. Commit the refreshed handoff before running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_COMMIT = "4b4df52"
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".sh", ".ps1", ".json", ".txt"}
EVIDENCE = {
    "current_regression": ("results/arxiv-baselines-debug-20260927-01.xml", 105, 0),
    "final_affected_repeat": ("results/arxiv-baselines-debug-20260927-02.xml", 13, 0),
    "historical_cuda_audit": ("results/deep-audit-verified-20260927-02.xml", None, 0),
    "historical_control_audit": ("results/deep-audit-control-20260927-03.xml", None, 0),
    "historical_deliberate_before_fix_failure": (
        "results/deep-audit-disk-fault-before-20260927.xml",
        None,
        1,
    ),
}
EXTRA_EVIDENCE = (
    "results/arxiv-baselines-dry-run-20260927.txt",
    "results/deep-audit-math-summary-20260927.json",
    "results/deep-audit-disk-fault-proof-20260927.json",
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
    parser.add_argument("--output", type=Path, required=True, help="New ZIP path; must not exist")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing review artifact: {output}")
    commit = git("rev-parse", "HEAD").decode().strip()
    git("merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, commit)
    entries: dict[str, bytes] = {}
    for raw in git("ls-tree", "-r", "--name-only", "-z", commit).split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = Path(name)
        if "results" in path.parts or "data" in path.parts:
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
            continue
        # Fail on uncommitted changes instead of silently packaging stale source/docs.
        git("diff", "--quiet", commit, "--", name)
        entries[name] = git("show", f"{commit}:{name}")

    # CODE_SUMMARY must exactly contain every source selected by the generator.
    from generate_code_summary import discover_sources, render_summary

    summary, summary_paths = render_summary(ROOT)
    if entries["gpt_handoff/CODE_SUMMARY.md"].decode("utf-8") != summary:
        raise ValueError("Regenerate and commit CODE_SUMMARY.md before packaging")
    for source in discover_sources(ROOT):
        name = source.relative_to(ROOT).as_posix()
        if name not in entries:
            raise ValueError(f"Source missing from committed bundle: {name}")
        if entries[name].decode("utf-8").replace("\r\n", "\n") != source.read_text(
            encoding="utf-8"
        ):
            raise ValueError(f"Committed/source snapshot mismatch: {name}")

    reports = {}
    for category, (name, expected_tests, expected_failures) in EVIDENCE.items():
        data = (ROOT / name).read_bytes()
        tree = ET.fromstring(data)
        counts = {
            key: len(tree.findall(f".//{key}"))
            for key in ("testcase", "failure", "error", "skipped")
        }
        if not counts["testcase"] or (
            expected_tests is not None and counts["testcase"] != expected_tests
        ):
            raise ValueError(f"Unexpected test count: {name}: {counts}")
        if counts["failure"] != expected_failures or counts["error"] or counts["skipped"]:
            raise ValueError(f"Unexpected evidence outcome: {name}: {counts}")
        entries[name] = data
        reports[category] = {"path": name, "sha256": sha(data), **counts}
    if (
        reports["historical_cuda_audit"]["testcase"]
        + reports["historical_control_audit"]["testcase"]
        != 93
    ):
        raise ValueError("Historical audit must remain a separate 93-test record")
    for name in EXTRA_EVIDENCE:
        entries[name] = (ROOT / name).read_bytes()
    entries["evidence/before_fix/sampled_inductive_train.py.txt"] = git(
        "show", "cd411cf:experiments/sampled_inductive/train.py"
    )
    entries["VERIFICATION.json"] = json_bytes(
        {
            "package_commit": commit,
            "implementation_commit": git("rev-parse", IMPLEMENTATION_COMMIT).decode().strip(),
            "historical_audit_commit": git("rev-parse", "cc173ed").decode().strip(),
            "evidence": reports,
            "source_files_in_code_summary": len(summary_paths),
            "real_dataset_training": False,
            "official_real_dataset_test_evaluation": False,
            "a100_mig_10gb_verified": False,
            "research_requirements_complete": False,
            "model_or_weight_downloaded": False,
            "notes": [
                "105 regression tests; 13 repeated after final changes. Do not add them.",
                "The prior 93-test audit was not rerun on this package commit.",
                "The deliberate pre-fix failure is synthetic corruption, not user result damage.",
                "Model forward/backward checks used CUDA and explicit synthetic debug inputs.",
                "CPU metadata/control checks do not constitute CPU model training.",
                "No arxiv result, speedup, multi-seed superiority or MIG fit is established.",
                "Arxiv's fixed/learned C x full/sampled controller remains incomplete.",
                "The independent-graph evaluation protocol remains incomplete.",
                "Historical PPI source is preserved; current production comparison rejects PPI.",
            ],
        }
    )
    entries["REVIEW_FIRST.md"] = (
        "# Current GPT review package — 2026-09-27\n\n"
        f"Package commit: `{commit}`. Implementation: `4b4df52`.\n\n"
        "Start with gpt_handoff/README_FIRST.md's 2026-09-27 section and review prompt.\n"
        "Then read HANDOFF.md, EXPERIMENT_STATUS.md, docs/ARXIV_BASELINE_COMPARISON.md,\n"
        "VERIFICATION.json, and actual source. Older sections retain historical context only.\n"
        "Current benchmark: ogbn-arxiv, 21 conditions including GCN, GraphSAGE and GATv2.\n"
        "No pretrained weights; no actual benchmark training/test or MIG fit measurements.\n"
        "The full original sampled-inductive research goal is not complete.\n\n"
        "MANIFEST.json hashes every archive member except itself. Evidence XMLs distinguish\n"
        "current checks, repeats, historical audits and deliberate pre-fix failures.\n"
        "Review source independently; documentation alone is not proof of successful execution.\n"
    ).encode()
    manifest = {
        "package_commit": commit,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "files": {name: sha(data) for name, data in sorted(entries.items())},
    }
    entries["MANIFEST.json"] = json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(entries):
            raise ValueError("Archive integrity/member verification failed")
        for name, expected in manifest["files"].items():
            if sha(archive.read(name)) != expected:
                raise ValueError(f"Archive SHA-256 mismatch: {name}")
    print(
        json.dumps(
            {
                "archive": str(output),
                "sha256": sha(output.read_bytes()),
                "bytes": output.stat().st_size,
                "entries": len(entries),
                "commit": commit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
