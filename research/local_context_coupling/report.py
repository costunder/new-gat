"""Complete graph/vector reductions for a fixed coupling mechanism audit."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .contract import write_csv, write_json


def _fmt(value):
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return str(value)
    return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)


def _table(rows, columns):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |", *["| " + " | ".join(_fmt(row.get(key)) for key in columns) + " |" for row in rows]])


def validate_rows(config, rows):
    keys = {(row["graph_id"], row["weight"], row["state"]) for row in rows}
    graph_ids = {row["graph_id"] for row in rows}
    expected = {(name, mode, state) for name in graph_ids for mode in config["weights"] for state in config["states"]}
    if len(graph_ids) != config["source"]["graphs"] or len(keys) != len(rows) or keys != expected:
        raise ValueError("report requires all unique original graph/C/reference states")
    for row in rows:
        if row["feature_mode"] not in ("independent_scalar_columns", "vector_trace") or type(row["actual_data"]) is not bool:
            raise ValueError("feature interpretation or actual-citation flag missing")
        for key, value in row.items():
            if isinstance(value, float) and (not math.isfinite(value) or (key.endswith("_norm_sq") and value < 0)):
                raise ValueError(f"invalid reported fixed diagnostic: {key}")
        if row["dependency_measurement"] != "all_existing_feature_column_operator_actions_not_full_Jacobian":
            raise ValueError("operator-action audit must not claim full Jacobian coverage")
        numerator, denominator = row["sandwich_increment_norm_sq"], row["sandwich_off_output_norm_sq"]
        ratio = row["sandwich_relative_increment"]
        if denominator == 0 and ratio is not None:
            raise ValueError("zero denominator relative increment must be undefined")
        if denominator > 0 and (ratio is None or not math.isclose(ratio, math.sqrt(numerator/denominator), rel_tol=1e-12, abs_tol=0)):
            raise ValueError("graph vector ratio must follow sum-of-squares reduction")
    return rows


def write_report(output, config, rows, resources, completion):
    output = Path(output)
    validate_rows(config, rows)
    rows = sorted(rows, key=lambda row: (row["graph_id"], row["weight"], row["state"]))
    write_csv(output/"graph_summary.csv", rows)
    write_json(output/"resources.json", resources)
    citations = [row for row in rows if row["feature_mode"] == "vector_trace"]
    formula_errors = [row["sandwich_relative_formula_error"] for row in rows if row["sandwich_relative_formula_error"] is not None]
    maxima = {
        "sandwich_formula_relative_error_max": max(formula_errors) if formula_errors else None,
        "immediate_increment_norm_max": max(math.sqrt(row["immediate_increment_norm_sq"]) for row in rows),
        "persistent_1_increment_norm_max": max(math.sqrt(row["persistent_1_increment_norm_sq"]) for row in rows),
        "persistent_2_increment_norm_max": max(math.sqrt(row["persistent_2_increment_norm_sq"]) for row in rows),
        "context_resolved_graph_state_cells": sum(row["context_numerically_resolved"] for row in rows),
        "predicted_increment_resolved_graph_state_cells": sum(row["predicted_increment_numerically_resolved"] for row in rows),
        "observed_increment_resolved_graph_state_cells": sum(row["observed_increment_numerically_resolved"] for row in rows),
    }
    write_json(output/"mechanism_checks.json", maxima)
    displayed=[]
    for row in citations:
        displayed.append({"graph":row["graph_id"],"C":row["weight"],"state":row["state"],"cross_energy_after_intra":row["cross_energy_after_intra_1"],"sandwich_delta_norm":math.sqrt(row["sandwich_increment_norm_sq"]),"sandwich_delta_relative":row["sandwich_relative_increment"],"formula_relative_error":row["sandwich_relative_formula_error"],"persistent_step3_delta_norm":math.sqrt(row["persistent_3_increment_norm_sq"]),"observed_resolved":row["observed_increment_numerically_resolved"]})
    text=f"""# 로컬 문맥 결합: 첫 고정 메커니즘 감사

Profile **{config['profile']}**, 전체 그래프 **{completion['graphs']}개**,
실제 citation **{completion['actual_citation_graphs']}개**.
모든 원래 노드·엣지·특징·1홉 induced local·공통 노드 copy 연결을 유지했다.
**학습 parameter 0, optimizer update 0, epoch 해당 없음**인 고정 연산 감사다.
시작 시의 독립 DEBUG fixture CE/gradient 검사는 `preflight_debug_checks.json`에 따로 기록한다.
분류기의 본학습이나 예측 성능은 이번 실행에서 측정하지 않았다.

## 실제 비교

A는 local별 weighted Laplacian, K는 서로 인접한 중심의 공통 원래 노드 copy를 잇는
W=1의 copy Laplacian이다. 동일 undirected 중심 쌍·공통 노드 링크는 한 번만 센다.
R은 특징을 모든 copy로 복사하고 M은 같은 원래 노드의 copy 평균으로 합친다.
η·γ는 각 물리 그래프에서 실제 weighted degree를 써서 안전하게 고정한다.

- Immediate: `M(I−γK)(I−ηA)RH`; cross 뒤 즉시 merge하면 K의 변화가 소거되는 대조다.
- Sandwich: `M(I−ηA)(I−γK)(I−ηA)RH`; cross 뒤 다시 local A를 적용한다.
- 같은 η의 off 대조와의 차이는 정확히 `−γη² M A K A R H`다. 전체 행렬을 dense로 만들지 않고 직접 action으로 대조했다.
- Persistent: 같은 step을 쓰는 `M(I−step(A+K))^s RH`와 K off를 s=1/2/3에서 비교했다.
  첫 두 단계 merge 차이는 0이며 세 번째 차이는 `−step³ M A K A R H`다.

H0/H1/H2는 이전 물리 그래프의 고정 확산 reference 입력이며 새 모델의 학습 층이 아니다.
각 상태에서 위 연산을 독립 적용했다. Cross 에너지는 `½ ZᵀKZ`, intra도 `½ ZᵀAZ`다.
새 에너지를 scalar lift로 더하지 않고 copy 상태의 연산에 직접 사용했다.

## 전체 입력의 메커니즘 검사

{_table([maxima],tuple(maxima))}

Norm residual과 수치상 구별되는 변화, 수학적 항등식은 다른 지표다.
모든 입력에서 cross·출력 차이가 양수라고 요구하지 않는다. 상수/영공간 입력에서는 0일 수 있다.
미세한 신호의 on−off 차이는 float64 상쇄 영향을 받을 수 있으므로 실제 값과 직접 계산한 예측값을 모두 CSV에 남겼다.
Resolved flag는 입력 copy norm에 대한 상대 threshold {config['measurement']['numerically_resolved_relative_threshold']}를 사용한다.
Residual은 상대 허용오차 {config['measurement']['relative_identity_tolerance']}로 검사하며 실패를 숨기지 않는다.

## 전체 citation 벡터의 결과

채널별 norm 제곱을 먼저 모두 합한 뒤 norm·상대 비율을 만들었다.
Citation의 {config['data']['physical_feature_columns']-config['source']['synthetic_scalar_inputs']}개 특징 채널을 독립 표본이나 학습 seed로 세지 않는다.

{_table(displayed,tuple(displayed[0]))}

이 표는 fixed 연산이 현재 입력에 실제로 작용했는지 보여 준다.
모든 기존 feature column에서 연산자의 action을 검사했으며 전체 input Jacobian을 dense하게 계산한 검사가 아니다.
일반적인 성능 개선·learned C·독립 그래프 일반화·새 GNN의 우월성을 입증하지 않는다.

## 전체 기록

- `channel_metrics.csv`: 모든 원래 graph/feature column/C/reference의 raw 진단.
- `graph_summary.csv`: 모든 graph/C/reference의 전체 채널 제곱 norm 합과 벡터 비율.
- `resources.json`: 입력 읽기·geometry CPU worker·CPU thread·GPU batch/chunk 후보와 최종 실제 실행 자원.
- `source_input_manifest.json`, `source_adapter.json`, `source_manifest.json`: 기존 source와 현재 scientific import closure의 hash.
- `coverage.json`, `completion.json`: 전체 coverage, scope, source 보존, elapsed.

GPU chunking과 graph batching은 계산만 분할했다. 노드·엣지·feature column을 제외하지 않았다.
DEBUG이면 명시된 fixture 경로이며 실제 데이터 분류 결과로 해석하지 않는다.
"""
    with (output/"LOCAL_CONTEXT_SUMMARY.md").open("x",encoding="utf-8") as stream:
        stream.write(text)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    for mode,color in zip(config["weights"],("#225ea8","#d95f0e"),strict=True):
        selected=[row for row in rows if row["weight"]==mode]
        axes[0].scatter([math.sqrt(row["context_cross_action_norm_sq"]) for row in selected],[math.sqrt(row["sandwich_increment_norm_sq"]) for row in selected],s=12,alpha=.4,label=mode,color=color)
        axes[1].scatter([math.sqrt(row["predicted_increment_norm_sq"]) for row in selected],[math.sqrt(row["sandwich_increment_norm_sq"]) for row in selected],s=12,alpha=.4,label=mode,color=color)
    for axis in axes:
        axis.set_xscale("symlog",linthresh=1e-10)
        axis.set_yscale("symlog",linthresh=1e-10)
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set(xlabel="K action after first local step (norm)",ylabel="Sandwich on-off increment (norm)",title="All graph/reference vector reductions")
    axes[1].set(xlabel="Direct -gamma eta² M A K A R H (norm)",ylabel="Measured sandwich increment (norm)",title="Matrix-free folded identity")
    folder=output/"figures"
    folder.mkdir(exist_ok=False)
    figure.savefig(folder/"context_coupling_checks.png",dpi=160)
    figure.savefig(folder/"context_coupling_checks.pdf")
    plt.close(figure)
