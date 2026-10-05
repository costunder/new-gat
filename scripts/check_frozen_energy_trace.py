"""Read frozen energy results, verify completion/coverage/file hashes, print E/J."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import product
from pathlib import Path


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value):
    return "undefined" if value is None else f"{value:.6f}"


def check(run):
    print(f"\nRUN: {run.name}")
    provenance = list(run.glob("*-provenance.json"))
    print(f"checkpoint 분석 기록: {len(provenance)}/9")
    if (run / "failure.json").is_file():
        print("상태: 실패")
        print("오류:", read(run / "failure.json")["error"])
        return
    if not (run / "completion.json").is_file():
        print("상태: 완료 기록 없음 — 진행/중단 여부는 이 기록만으로 판단 불가")
        return
    done = read(run / "completion.json")
    debug = done.get("profile") == "debug"
    if done.get("profile") not in ("full", "debug"):
        raise ValueError("unknown full/debug profile")
    required = {
        "completed": True,
        "seed": 11,
        "seed_count": 1,
        "checkpoint_analyses": 9,
        "stage_rows": 99,
        "transition_rows": 99,
        "new_training_runs": 0,
        "optimizer_updates": 0,
        "parameters_preserved": True,
        "full_graphs": True,
        "actual_citation_data": not debug,
        "scope": "DEBUG_pipeline_only" if debug else "full_frozen_citation_diagnostic",
    }
    mismatch = [
        key
        for key, value in required.items()
        if type(done.get(key)) is not type(value) or done.get(key) != value
    ]
    if mismatch:
        raise ValueError(f"completion contract differs/missing: {mismatch}")
    datasets = [f"DEBUG-{name}" if debug else name for name in ("Cora", "CiteSeer", "PubMed")]
    cases = set(product(datasets, ("mlp", "standard_gcn", "polynomial_2"), (11,)))
    expected_provenance = {f"{data}-{model}-provenance.json" for data, model, _ in cases}
    if {p.name for p in provenance} != expected_provenance:
        raise ValueError("missing/duplicate/unexpected checkpoint analysis provenance")
    stages = read(run / "stages.json")["rows"]
    rows = read(run / "transitions.json")["rows"]
    stage_names = {"input", "logits", "layer_0_activated"} | {
        f"layer_{layer}_{name}"
        for layer in (0, 1)
        for name in ("projected", "aggregated", "replay_P", "replay_P2")
    }
    operations = {
        (layer, name)
        for layer in (0, 1)
        for name in (
            "projection",
            "aggregation",
            "replay_I_to_P",
            "replay_P_to_P2",
            "replay_I_to_P2",
        )
    } | {(0, "relu")}
    actual_stages = {(r["dataset"], r["model"], r["seed"], r["stage"]) for r in stages}
    expected_stages = {(*case, name) for case in cases for name in stage_names}
    actual_rows = {(r["dataset"], r["model"], r["seed"], r["layer"], r["transition"]) for r in rows}
    expected_rows = {(*case, layer, name) for case in cases for layer, name in operations}
    if len(stages) != 99 or actual_stages != expected_stages:
        raise ValueError("stage coverage differs")
    if len(rows) != 99 or actual_rows != expected_rows:
        raise ValueError("transition coverage differs")
    manifest_path = run / "artifacts.json"
    if sha256(manifest_path) != done["artifact_manifest_sha256"]:
        raise ValueError("artifact manifest SHA256 differs")
    manifest = read(manifest_path)
    if not manifest:
        raise ValueError("empty artifact manifest")
    critical_files = {
        "contract.json",
        "source.json",
        "stages.json",
        "transitions.json",
        "ENERGY_TRACE.md",
    } | expected_provenance
    if not critical_files.issubset(manifest):
        raise ValueError("artifact manifest omits required result files")
    for name, expected in manifest.items():
        path = run / name
        if Path(name).name != name or not path.resolve().is_relative_to(run.resolve()):
            raise ValueError(f"invalid artifact path: {name}")
        if sha256(path) != expected:
            raise ValueError(f"artifact SHA256 differs: {name}")
    label = "DEBUG 완료 — 최종 본실험 아님" if debug else "완료 — full 본실험"
    print(f"상태: {label} | 파일 SHA256 {len(manifest)}개 일치")
    print("seed: 11 하나 | stage: 99/99 | transition: 99/99 | 새 학습/update: 0")
    for title, selected in (
        ("실제 aggregation Z→A", [r for r in rows if r["transition"] == "aggregation"]),
        (
            "GCN 동일 Z의 P→P²",
            [
                r
                for r in rows
                if r["model"] == "standard_gcn" and r["transition"] == "replay_P_to_P2"
            ],
        ),
    ):
        print(f"\n{title}")
        print(
            f"{'Dataset':<15} {'Model':<14} L {'E median ratio':>15} "
            f"{'E ratio IQR':>12} {'|Jnorm| ratio':>15} {'Jnorm cosine':>15}"
        )
        for r in sorted(selected, key=lambda r: (r["dataset"], r["model"], r["layer"])):
            print(
                f"{r['dataset']:<15} {r['model']:<14} {r['layer']} "
                f"{number(r['E_operation_ratio']['median']):>15} "
                f"{number(r['E_operation_ratio']['IQR']):>12} "
                f"{number(r['J_distinct_normalized_mean_abs_ratio']):>15} "
                f"{number(r['J_distinct_normalized_retention']['pattern_cosine']):>15}"
            )
    print("전체 보고서:", run / "ENERGY_TRACE.md")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if (root / "contract.json").is_file():
        runs = [root]
    else:
        runs = sorted(
            set(root.glob("frozen-energy-seed11-*"))
            | set(root.glob("frozen-energy-trace-seed11-*"))
        )
    if not runs:
        print("frozen energy 분석 결과 폴더가 없습니다:", root.resolve())
    for run in runs:
        try:
            check(run)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"상태: 결과 파일 검증 실패 | {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
