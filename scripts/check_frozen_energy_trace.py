"""Read frozen energy results, verify completion/coverage/file hashes, print E/J."""

from __future__ import annotations

import argparse
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
    return "undefined" if value is None else f"{value:.7g}"


def normalized_j_ratio(row):
    if row["J_distinct_normalized_zero_reference_mean_abs"]:
        return None
    return row["J_distinct_normalized_mean_abs_ratio"]


def table(title, columns, records):
    print(f"\n{title}")
    records = [[str(value) for value in record] for record in records]
    widths = [max(len(name), *(len(row[i]) for row in records)) for i, name in enumerate(columns)]
    print("  ".join(name.ljust(width) for name, width in zip(columns, widths, strict=True)))
    for record in records:
        print("  ".join(value.ljust(width) for value, width in zip(record, widths, strict=True)))


def stage_pair(row, stage_index):
    layer = row["layer"]
    project, aggregate = f"layer_{layer}_projected", f"layer_{layer}_aggregated"
    p, p2 = f"layer_{layer}_replay_P", f"layer_{layer}_replay_P2"
    before, after = {
        "projection": ("input" if layer == 0 else "layer_0_activated", project),
        "aggregation": (project, aggregate),
        "relu": (aggregate, "layer_0_activated"),
        "replay_I_to_P": (project, p),
        "replay_P_to_P2": (p, p2),
        "replay_I_to_P2": (project, p2),
    }[row["transition"]]
    case = row["dataset"], row["model"], row["seed"]
    return stage_index[(*case, before)], stage_index[(*case, after)]


def detail_tables(stages, rows):
    stage_index = {(r["dataset"], r["model"], r["seed"], r["stage"]): r for r in stages}
    model_names = {"mlp": "MLP", "standard_gcn": "GCN", "polynomial_2": "Poly2"}
    operation_names = {
        "projection": "FT",
        "aggregation": "GP",
        "relu": "ReLU",
        "replay_I_to_P": "I->P",
        "replay_P_to_P2": "P->P2",
        "replay_I_to_P2": "I->P2",
    }
    ordering = {name: i for i, name in enumerate(operation_names)}
    print("\n상세: 저장된 99개 transition 전부, 실제 연산 45개 + 공통 P replay 54개")
    print("FT=H->HW, GP=Z->실제 aggregation, ReLU=첫 층 활성화")
    print("replay는 각 모델의 동일 Z에 공통 GCN P를 적용; Poly2의 실제 연산과 P²는 별개")
    print("E_med/p10/p90/IQR는 유효 노드별 after/before 비율의 분포")
    print("E_sf_med는 E/(로컬 평균을 뺀 특징 norm²+epsilon)의 노드별 변화 비율")
    print("G_raw/G_aug/Norm²는 전체 그래프 raw E/augmented normalized E/특징 norm² 비율")
    print("J_*_L2는 signed J 벡터의 L2 크기비; Jn_abs는 정규화 distinct J 평균 절댓값 비율")
    print("정규화 J = J/(sqrt(E_v*E_u)+1e-12). 작은 값은 과학적 표기로 표시")
    print("cos는 같은 쌍 순서의 패턴 cosine이며 정보 보존율이 아님")
    print("ΔJ=정규화 distinct J의 same-label 평균 - different-label 평균; 사후 진단")
    print("undefined는 분모 0/빈 그룹. zero_E는 raw E 비율에서 제외된 노드 수")
    for title, selected in (
        ("실제 FT / GP / ReLU", [r for r in rows if not r["transition"].startswith("replay_")]),
        (
            "공통 P의 I / P / P² replay — 세 모델의 Z 모두",
            [r for r in rows if r["transition"].startswith("replay_")],
        ),
    ):
        selected.sort(
            key=lambda r: (r["dataset"], r["model"], r["layer"], ordering[r["transition"]])
        )
        energy_records, relation_records, label_records = [], [], []
        for r in selected:
            identity = (
                r["dataset"],
                model_names[r["model"]],
                r["layer"],
                operation_names[r["transition"]],
            )
            before, after = stage_pair(r, stage_index)
            energy_records.append(
                (
                    *identity,
                    *(
                        number(r["E_operation_ratio"][key])
                        for key in ("p10", "median", "p90", "IQR")
                    ),
                    number(r["E_scale_free_operation_ratio"]["median"]),
                    number(r["E_sym_operation_ratio"]["median"]),
                    number(r["E_sym_augmented_operation_ratio"]["median"]),
                    number(r["global_E_operation_ratio"]),
                    number(r["global_E_sym_augmented_operation_ratio"]),
                    number(r["global_feature_norm_sq_operation_ratio"]),
                    r["E_undefined_ratio_count"],
                )
            )
            relation_records.append(
                (
                    *identity,
                    *(
                        number(r[key + "_retention"]["magnitude_ratio"])
                        for key in ("J_shared", "J_node", "J_distinct")
                    ),
                    number(r["J_distinct_retention"]["pattern_cosine"]),
                    number(normalized_j_ratio(r)),
                    number(r["J_distinct_normalized_retention"]["pattern_cosine"]),
                    number(before["J_distinct"]["mean"]),
                    number(after["J_distinct"]["mean"]),
                )
            )
            label_before, label_after = before["delta_J_label"], after["delta_J_label"]
            difference = (
                None if label_before is None or label_after is None else label_after - label_before
            )
            label_records.append(
                (
                    *identity,
                    number(label_before),
                    number(label_after),
                    number(difference),
                    number(before["label_same_distinct_normalized"]["mean"]),
                    number(after["label_same_distinct_normalized"]["mean"]),
                    number(before["label_different_distinct_normalized"]["mean"]),
                    number(after["label_different_distinct_normalized"]["mean"]),
                )
            )
        identity_columns = ("Dataset", "Model", "L", "Op")
        table(
            title + " — 에너지와 크기 통제",
            (
                *identity_columns,
                "E_p10",
                "E_med",
                "E_p90",
                "E_IQR",
                "E_sf_med",
                "E_sym_med",
                "E_aug_med",
                "G_raw",
                "G_aug",
                "Norm²",
                "zero_E",
            ),
            energy_records,
        )
        table(
            title + " — raw J / normalized J",
            (
                *identity_columns,
                "Jsh_L2",
                "Jnode_L2",
                "Jdist_L2",
                "Jraw_cos",
                "Jn_abs",
                "Jn_cos",
                "Jraw_mean_before",
                "Jraw_mean_after",
            ),
            relation_records,
        )
        table(
            title + " — label separation",
            (
                *identity_columns,
                "ΔJ_before",
                "ΔJ_after",
                "Δ(ΔJ)",
                "same_before",
                "same_after",
                "diff_before",
                "diff_after",
            ),
            label_records,
        )
    inputs = [r for r in stages if r["stage"] == "input"]
    table(
        "측정 범위 / label 진단의 유효 쌍 수 (양방향 포함; 독립 반복 아님)",
        ("Dataset", "Model", "pairs", "known", "same", "different"),
        [
            (
                r["dataset"],
                model_names[r["model"]],
                r["adjacent_local_pair_count"],
                r["known_label_pair_count"],
                r["same_label_pair_count"],
                r["different_label_pair_count"],
            )
            for r in sorted(inputs, key=lambda r: (r["dataset"], r["model"]))
        ],
    )
    print(
        "\nG_aug 감소는 GCN의 실제 GP와 공통 P replay에서 기대됨. "
        "local E·FT·ReLU·Poly2 GP에는 같은 보장이 없음."
    )
    print(
        "FP32 전파에는 반올림 오차가 있음. "
        "에너지 크기·cosine·label 평균 차이만으로 정보 손실/유의성을 판정하지 않음."
    )


def check(run, detail=False):
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
                f"{number(normalized_j_ratio(r)):>15} "
                f"{number(r['J_distinct_normalized_retention']['pattern_cosine']):>15}"
            )
    if detail:
        detail_tables(stages, rows)
    print("전체 보고서:", run / "ENERGY_TRACE.md")
    print("전체 원본 표:", run / "stages.csv", run / "transitions.csv")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", nargs="?", type=Path, default=Path("results"))
    parser.add_argument(
        "--detail",
        action="store_true",
        help="print all 99 operation diagnostics, raw/normalized E/J, label separation",
    )
    args = parser.parse_args()
    root = args.results_root
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
            check(run, detail=args.detail)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"상태: 결과 파일 검증 실패 | {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
