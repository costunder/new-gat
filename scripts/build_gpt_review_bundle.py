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
IMPLEMENTATION_COMMIT = "4ddead7"  # Previous review base; current implementation is package HEAD.
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".sh", ".ps1", ".json", ".txt"}
EVIDENCE = {
    "current_affected_regression": ("results/arxiv-memory-regression-20260927.xml", None, 0),
    "previous_4ddead7_regression": ("results/energy-precision-after-20260927.xml", 230, 0),
    "previous_3eb0eb6_regression": ("results/fused-review-after-20260927.xml", 110, 0),
    "previous_4c2d7f4_regression": ("results/revision-final-debug-20260927-01.xml", 152, 0),
    "deliberate_before_fix_reproduction": ("results/fused-review-before-20260927.xml", 4, 3),
    "before_precision_fix_cuda_bf16": ("results/energy-precision-before-actual-20260927.xml", 1, 1),
    "invalid_bf16_fixture_fp32_control": ("results/energy-precision-before-20260927.xml", 1, 0),
    "dtype_only_repair_ordering_failures": (
        "results/energy-precision-focused-20260927.xml",
        112,
        20,
    ),
}
EXTRA_EVIDENCE = (
    "results/revision-gram-profile-20260927-02.json",
    "results/review-four-documents-20260927/reproduced.json",
    "results/review-four-documents-20260927/fusion-sampling.json",
)
REAL_DATA_CALIBRATIONS = tuple(
    f"results/arxiv-mig-memory/{condition}-final-cap7.json"
    for condition in ("full-fixed", "full-learned", "sampled-fixed", "sampled-learned")
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
    for name in EXTRA_EVIDENCE:
        entries[name] = (ROOT / name).read_bytes()
    for path in sorted((ROOT / "results/arxiv-mig-memory").glob("*.json")):
        entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in sorted((ROOT / "results").glob("arxiv-*.log")):
        entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    real_data = {}
    for name in REAL_DATA_CALIBRATIONS:
        measured = json.loads(entries[name])
        if measured["status"] != "passed" or not measured["source_unchanged_during_measurement"]:
            raise ValueError(f"Incomplete or modified-source real-data calibration: {name}")
        real_data[name] = {
            "sha256": sha(entries[name]),
            "actual_gpu": measured["actual_gpu"],
            "allocator_limit_bytes": measured["allocator_limit_bytes"],
            "peak_allocated_bytes": measured["measurement"]["peak_allocated_bytes"],
            "peak_reserved_bytes": measured["measurement"]["peak_reserved_bytes"],
        }
    entries["evidence/before_fix/sampled_inductive_train.py.txt"] = git(
        "show", "cd411cf:experiments/sampled_inductive/train.py"
    )
    entries["VERIFICATION.json"] = json_bytes(
        {
            "package_commit": commit,
            "implementation_commit": commit,
            "previous_review_base": git("rev-parse", IMPLEMENTATION_COMMIT).decode().strip(),
            "historical_audit_commit": git("rev-parse", "cc173ed").decode().strip(),
            "evidence": reports,
            "source_files_in_code_summary": len(summary_paths),
            "real_dataset_training": "disposable calibration only; no final training",
            "real_dataset_calibration": real_data,
            "final_real_dataset_training": False,
            "official_real_dataset_test_evaluation": False,
            "a100_mig_10gb_verified": False,
            "research_requirements_complete": False,
            "model_or_weight_downloaded": False,
            "notes": [
                "Current regressions and previous 230/110/152 checks are separate records.",
                "Complete official arxiv calibration ran on local RTX 5070 Ti with a 7GiB cap.",
                "Allocator cap is not A100 MIG emulation; actual MIG speed/fit remains unverified.",
                "No-energy diagnostic projection streams nodes; prediction/learning is unchanged.",
                "Previous model BF16 fixture ran FP32; historical labels are not BF16 evidence.",
                "Current CUDA fixture asserts actual AMP state after hardware argument resolution.",
                "Energy readout reference/fused uses explicit FP32 with autocast disabled (v1).",
                "Fused v2 preserves reference contraction order and recomputes Gram in backward.",
                "Transient node Gram is allocated; no current speed/peak-memory benefit claimed.",
                "Precision policy/source identity changed: new run ID; old results preserved.",
                "CUDA before repair: 3 deliberate failures and 1 passing control.",
                "Diagnostics and interventions now preserve the declared numerical path.",
                "Before-fix counterexamples use synthetic evidence, not user result damage.",
                "Model forward/backward checks used CUDA and explicit synthetic debug inputs.",
                "CPU metadata/control checks do not constitute CPU model training.",
                "No final arxiv score, speedup, multi-seed superiority or MIG fit is established.",
                "Core full/sampled x fixed/dynamic C uses complete matched supervised passes.",
                "Custom arxiv node-year views are not an OGB score or independent-graph test.",
                "Optional fused Gram/readout keeps projected history; no end-to-end speed claim.",
                "Matched-update secondary study and independent-graph testing are not implemented.",
                "Historical PPI source is preserved; current production comparison rejects PPI.",
            ],
        }
    )
    entries["REVIEW_FIRST.md"] = (
        "# Current GPT review package — 2026-09-27\n\n"
        f"Package and implementation commit: `{commit}`.\n\n"
        "Start with docs/ARXIV_MEMORY_REVIEW_20260927.md and gpt_handoff/README_FIRST.md.\n"
        "Then read docs/FOUR_DOCUMENT_REVIEW_20260927.md (before-fix review),\n"
        "VERIFICATION.json, and actual source. Older sections retain historical context only.\n"
        "Current benchmark: ogbn-arxiv, 21 conditions including GCN, GraphSAGE and GATv2.\n"
        "No pretrained weights; real arxiv calibration on local RTX with a 7GiB allocator cap.\n"
        "No final benchmark training/test or actual MIG fit measurements.\n"
        "Includes core four-cell controller, temporal visibility, and optional fused Gram.\n"
        "Real-data accuracy/cost and independent-graph generalization remain unverified.\n\n"
        "MANIFEST.json hashes every archive member except itself. Evidence XMLs distinguish\n"
        "current checks, repeats, historical audits and deliberate pre-fix failures.\n"
        "Review source independently; documentation alone is not proof of successful execution.\n"
    ).encode()
    manifest = {
        "package_commit": commit,
        "implementation_commit": commit,
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
