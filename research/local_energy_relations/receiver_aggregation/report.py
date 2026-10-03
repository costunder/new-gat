"""Complete receiver aggregation summaries, with measured scope and strict ratios."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

WEIGHTS = ("unit", "local_degree")
STATES = (0, 1, 2)
STAGE_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2))
RELATIONS = ("shared_edges", "shared_nodes", "distinct_edges")
CONDITIONS = ("tagged", "sum", "sum_within", "sum_between", "sum_both")
TABLES = ("graph_summary", "reconstruction_summary", "condition_summary", "relation_summary")
FIGURES = ("tag_contrast_and_recovery", "energy_relation_reconstruction")
ARTIFACTS = (
    "RECEIVER_AGGREGATION_SUMMARY.md",
    *(f"{name}.csv" for name in TABLES),
    "resources.json",
    *(f"figures/{name}.{suffix}" for name in FIGURES for suffix in ("png", "pdf")),
)


def _number(value, label, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _integer(value, label, *, minimum=0):
    value = _number(value, label)
    if value < minimum or not value.is_integer():
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _nonnegative(row, field, label):
    value = _number(row.get(field), f"{label}.{field}")
    if value < 0:
        raise ValueError(f"{label}.{field} must be nonnegative")
    return value


def _close(left, right, label):
    if not math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {left} versus {right}")


def _ratio(row, norm_field, error_field, ratio_field, label, *, squared=False):
    norm = _nonnegative(row, norm_field, label)
    error = _nonnegative(row, error_field, label)
    ratio = _number(row.get(ratio_field), f"{label}.{ratio_field}", nullable=True)
    if norm == 0:
        if ratio is not None:
            raise ValueError(f"{label}.{ratio_field}: zero denominator must be undefined")
    elif ratio is None or ratio < 0:
        raise ValueError(f"{label}.{ratio_field}: positive denominator needs a nonnegative ratio")
    else:
        expected = error / norm if squared else math.sqrt(error / norm)
        _close(ratio, expected, f"{label}.{ratio_field}")
    return norm, error, ratio


def _stage(value):
    if isinstance(value, str) and value in ("H0", "H1", "H2"):
        return int(value[1])
    value = _integer(value, "state")
    if value not in STATES:
        raise ValueError("state must be H0, H1 or H2")
    return value


def _meta(row, graphs, label):
    graph = row.get("graph_id")
    if graph not in graphs:
        raise ValueError(f"{label}: unknown graph")
    weight = row.get("weight")
    if weight not in WEIGHTS:
        raise ValueError(f"{label}: unknown weight")
    for field in ("family", "dataset", "actual_data", "feature_mode"):
        if field in row and row[field] != graphs[graph][field]:
            raise ValueError(f"{label}: mismatched {field}")
    return graph, weight


def _coverage(seen, expected, label):
    if seen != expected:
        raise ValueError(
            f"{label} coverage mismatch: missing={len(expected - seen)}, "
            f"extra={len(seen - expected)}"
        )


def validate_rows(config, graph_rows, reconstruction_rows, condition_rows, relation_rows):
    """Validate every graph, state, condition and stage pair before creating artifacts."""
    profile = config.get("profile")
    if profile not in ("full", "debug"):
        raise ValueError("report profile must explicitly be full or debug")
    expected_count = 201 if profile == "full" else 21
    if len(graph_rows) != expected_count:
        raise ValueError(f"{profile} requires {expected_count} complete graphs")
    graphs = {}
    for index, row in enumerate(graph_rows):
        label = f"graph[{index}]"
        graph = row.get("graph_id")
        if not isinstance(graph, str) or not graph or graph in graphs:
            raise ValueError(f"{label}: graph_id must be unique and nonempty")
        for field in ("family", "dataset"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{label}: {field} must be nonempty")
        n = _integer(row.get("num_nodes"), f"{label}.num_nodes", minimum=1)
        _integer(row.get("num_edges"), f"{label}.num_edges")
        _integer(row.get("num_features"), f"{label}.num_features", minimum=1)
        nl = _integer(row.get("num_local_nodes"), f"{label}.num_local_nodes", minimum=n)
        components = _integer(row.get("components"), f"{label}.components", minimum=1)
        if components > n:
            raise ValueError(f"{label}: components exceed nodes")
        rank = _integer(
            row.get("restricted_rank_per_channel"), f"{label}.restricted_rank_per_channel"
        )
        if rank != n - components:
            raise ValueError(f"{label}: restricted rank must equal N-components")
        tagged = _integer(row.get("tagged_coordinates"), f"{label}.tagged_coordinates")
        active = _integer(row.get("fused_active_coordinates"), f"{label}.fused_active_coordinates")
        hidden = _integer(
            row.get("ambient_tag_kernel_per_channel"), f"{label}.ambient_tag_kernel_per_channel"
        )
        if active > nl or tagged < active or hidden != tagged - active:
            raise ValueError(f"{label}: ambient receipt kernel dimension mismatch")
        for weight in WEIGHTS:
            _integer(row.get(f"sparse_nnz_{weight}"), f"{label}.sparse_nnz_{weight}")
        if not isinstance(row.get("actual_data"), bool):
            raise ValueError(f"{label}: actual_data must be explicit")
        if row.get("feature_mode") not in ("independent_scalar_columns", "vector_trace"):
            raise ValueError(f"{label}: unknown feature mode")
        if row["actual_data"] and row["feature_mode"] != "vector_trace":
            raise ValueError(f"{label}: citation must use vector trace")
        graphs[graph] = row
    actual = [row for row in graphs.values() if row["actual_data"]]
    if profile == "full":
        if len(actual) != 3 or {row["dataset"] for row in actual} != {"Cora", "CiteSeer", "PubMed"}:
            raise ValueError("full requires all three actual citation datasets")
    elif actual:
        raise ValueError("DEBUG fixtures must not claim actual citation data")

    expected_states = {(g, w, t) for g in graphs for w in WEIGHTS for t in STATES}
    reconstruction = {}
    ratio_pairs = (
        ("q_norm_sq", "tagged_q_error_sq", "tagged_q_relative_error"),
        ("q_norm_sq", "fused_q_error_sq", "fused_q_relative_error"),
        ("centered_h_norm_sq", "centered_h_error_sq", "centered_h_relative_error"),
        (
            "tagged_norm_sq",
            "tagged_reconstruction_error_sq",
            "tagged_reconstruction_relative_error",
        ),
        ("fused_norm_sq", "fused_residual_sq", "fused_relative_residual"),
    )
    for index, row in enumerate(reconstruction_rows):
        label = f"reconstruction[{index}]"
        g, w = _meta(row, graphs, label)
        t = _stage(row.get("state"))
        key = g, w, t
        if key in reconstruction:
            raise ValueError(f"duplicate reconstruction: {key}")
        units = (
            graphs[g]["num_features"]
            if graphs[g]["feature_mode"] == "independent_scalar_columns"
            else 1
        )
        if _integer(row.get("count"), f"{label}.count") != units:
            raise ValueError(f"{label}: count must cover all declared input fields")
        tagged, contrast, _ = _ratio(
            row,
            "tagged_norm_sq",
            "tag_contrast_norm_sq",
            "tag_contrast_fraction_sq",
            label,
            squared=True,
        )
        if contrast > tagged * (1 + 1e-8) + 1e-12:
            raise ValueError(f"{label}: contrast projection norm exceeds original norm")
        _nonnegative(row, "null_projection_residual_sq", label)
        for norm_field, error_field, ratio_field in ratio_pairs:
            _ratio(row, norm_field, error_field, ratio_field, label)
        _integer(row.get("solver_iterations_max"), f"{label}.solver_iterations_max")
        for field in ("solver_residual_max", "solver_normal_residual_max"):
            _nonnegative(row, field, label)
        reconstruction[key] = row
    _coverage(set(reconstruction), expected_states, "reconstruction")

    conditions = {}
    for index, row in enumerate(condition_rows):
        label = f"condition[{index}]"
        g, w = _meta(row, graphs, label)
        t = _stage(row.get("state"))
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ValueError(f"{label}: unknown condition")
        key = g, w, t, condition
        if key in conditions:
            raise ValueError(f"duplicate condition: {key}")
        graph = graphs[g]
        channels = graph["num_features"] if graph["feature_mode"] == "vector_trace" else 1
        count = (
            graph["num_features"] if graph["feature_mode"] == "independent_scalar_columns" else 1
        )
        if _integer(row.get("count"), f"{label}.count") != count:
            raise ValueError(f"{label}: wrong input count")
        within = condition in ("sum_within", "sum_both")
        between = condition in ("sum_between", "sum_both")
        coordinates = (
            graph["tagged_coordinates"] if condition == "tagged" else graph["num_local_nodes"]
        ) * channels
        coordinates += graph["num_nodes"] if within else 0
        coordinates += 6 * graph["num_edges"] if between else 0
        if (
            _integer(row.get("observed_coordinates"), f"{label}.observed_coordinates")
            != coordinates
        ):
            raise ValueError(f"{label}: observed coordinates omit or add undeclared values")
        if (
            _integer(row.get("restricted_rank"), f"{label}.restricted_rank")
            != graph["restricted_rank_per_channel"] * channels
        ):
            raise ValueError(f"{label}: restricted rank must follow shared-field theorem")
        if _integer(row.get("additional_rank"), f"{label}.additional_rank") != 0:
            raise ValueError(
                f"{label}: no additional actual-field rank in the fixed known-C contract"
            )
        expected_q = reconstruction[g, w, t][
            "tagged_q_relative_error" if condition == "tagged" else "fused_q_relative_error"
        ]
        q_error = _number(row.get("q_relative_error"), f"{label}.q_relative_error", nullable=True)
        if expected_q is None:
            if q_error is not None:
                raise ValueError(f"{label}: zero flow relative error must be undefined")
        elif q_error is None:
            raise ValueError(f"{label}: measured q error missing")
        else:
            _close(q_error, expected_q, f"{label}.q_relative_error")
        for name, observed in (("energy", within), ("relation", between)):
            if (
                not isinstance(row.get(f"{name}_observed"), bool)
                or row[f"{name}_observed"] != observed
            ):
                raise ValueError(f"{label}: wrong {name} observation flag")
            if observed:
                _ratio(row, f"{name}_norm_sq", f"{name}_error_sq", f"{name}_relative_error", label)
            elif any(
                row.get(f"{name}_{suffix}") is not None
                for suffix in ("norm_sq", "error_sq", "relative_error")
            ):
                raise ValueError(f"{label}: absent {name} must have undefined metrics")
        conditions[key] = row
    _coverage(
        set(conditions), {(*key, c) for key in expected_states for c in CONDITIONS}, "condition"
    )
    for key in expected_states:
        for name, single in (("energy", "sum_within"), ("relation", "sum_between")):
            for suffix in ("norm_sq", "error_sq", "relative_error"):
                left = conditions[(*key, single)][f"{name}_{suffix}"]
                right = conditions[(*key, "sum_both")][f"{name}_{suffix}"]
                if left is None or right is None:
                    if left is not right:
                        raise ValueError(
                            "identical E/J conditional checks must agree across conditions"
                        )
                else:
                    _close(left, right, "E/J conditional checks")

    relations = {}
    for index, row in enumerate(relation_rows):
        label = f"relation[{index}]"
        g, w = _meta(row, graphs, label)
        a, b = _stage(row.get("stage_from")), _stage(row.get("stage_to"))
        relation = row.get("relation")
        key = g, w, a, b, relation
        if key in relations:
            raise ValueError(f"duplicate relation: {key}")
        if relation not in RELATIONS or (a, b) not in STAGE_PAIRS:
            raise ValueError(f"{label}: undeclared relation or stage pair")
        count = 2 * graphs[g]["num_edges"]
        if graphs[g]["feature_mode"] == "independent_scalar_columns":
            count *= graphs[g]["num_features"]
        if _integer(row.get("count"), f"{label}.count") != count:
            raise ValueError(f"{label}: all directed pairs and fields must be included")
        for field in ("original_sum", "reconstructed_sum"):
            _number(row.get(field), f"{label}.{field}")
        _ratio(row, "norm_sq", "error_sq", "relative_error", label)
        negatives = _integer(row.get("negative_count"), f"{label}.negative_count")
        if negatives > count:
            raise ValueError(f"{label}: negative count exceeds count")
        if count == 0 and (
            row["original_sum"] != 0
            or row["reconstructed_sum"] != 0
            or row["norm_sq"] != 0
            or row["error_sq"] != 0
        ):
            raise ValueError(f"{label}: empty relation must have zero absolute statistics")
        relations[key] = row
    _coverage(
        set(relations),
        {
            (g, w, a, b, r)
            for g in graphs
            for w in WEIGHTS
            for a, b in STAGE_PAIRS
            for r in RELATIONS
        },
        "relation",
    )
    for g in graphs:
        for w in WEIGHTS:
            for a, b in STAGE_PAIRS:
                for field in ("original_sum", "reconstructed_sum"):
                    node = relations[g, w, a, b, "shared_nodes"][field]
                    shared = relations[g, w, a, b, "shared_edges"][field]
                    distinct = relations[g, w, a, b, "distinct_edges"][field]
                    _close(node, 2 * shared + distinct, "relation decomposition")
    return graphs


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value):
    if value is None:
        return "undefined"
    if isinstance(value, (float, np.floating)):
        return f"{value:.7g}"
    return str(value).replace("|", "\\|")


def _table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend(
        "| " + " | ".join(_format(row.get(field)) for field in fields) + " |" for row in rows
    )
    return "\n".join(lines)


def _figures(output, reconstruction, conditions):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"unit": "#0072B2", "local_degree": "#D55E00"}
    markers = {0: "o", 1: "s", 2: "^"}
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    omitted = 0
    for weight in WEIGHTS:
        for state in STATES:
            selected = [
                r for r in reconstruction if r["weight"] == weight and _stage(r["state"]) == state
            ]
            valid = [
                r
                for r in selected
                if r["tag_contrast_fraction_sq"] is not None
                and r["fused_q_relative_error"] is not None
            ]
            omitted += len(selected) - len(valid)
            ax.scatter(
                [r["tag_contrast_fraction_sq"] for r in valid],
                [r["fused_q_relative_error"] for r in valid],
                s=20,
                alpha=0.7,
                color=colors[weight],
                marker=markers[state],
                label=f"{weight}, H{state}",
            )
    ax.set(
        xlabel="Squared receipt contrast / squared tagged receipts",
        ylabel="Actual q relative reconstruction error",
        title="Hidden source labels and restricted actual-flow recovery",
    )
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.98,
        f"Undefined ratios: {omitted} rows",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"{FIGURES[0]}.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, name, condition in zip(
        axes, ("energy", "relation"), ("sum_within", "sum_between"), strict=True
    ):
        omitted = 0
        for weight in WEIGHTS:
            for state in STATES:
                selected = [
                    r
                    for r in conditions
                    if r["condition"] == condition
                    and r["weight"] == weight
                    and _stage(r["state"]) == state
                ]
                valid = [r for r in selected if r[f"{name}_relative_error"] is not None]
                omitted += len(selected) - len(valid)
                ax.scatter(
                    [r[f"{name}_norm_sq"] for r in valid],
                    [r[f"{name}_relative_error"] for r in valid],
                    s=20,
                    alpha=0.7,
                    color=colors[weight],
                    marker=markers[state],
                    label=f"{weight}, H{state}",
                )
        ax.set(
            xlabel=f"Squared observed {name} norm",
            ylabel="Conditional reconstruction relative error",
            title=f"{name.capitalize()} reproduced from Y-only inverse",
        )
        ax.set_xscale("symlog", linthresh=1e-12)
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.grid(alpha=0.2)
        ax.text(
            0.02,
            0.98,
            f"Undefined ratios: {omitted} rows",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"{FIGURES[1]}.{suffix}", dpi=180)
    plt.close(fig)


def write_report(
    output_dir,
    config,
    graph_rows,
    reconstruction_rows,
    condition_rows,
    relation_rows,
    resource_rows,
    completion,
):
    """Write every macro row, all citation diagnostics, and standalone figures."""
    output = Path(output_dir)
    if not output.is_dir():
        raise ValueError("report output directory must exist")
    occupied = [name for name in ARTIFACTS if (output / name).exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite report artifacts: {occupied}")
    graphs = validate_rows(config, graph_rows, reconstruction_rows, condition_rows, relation_rows)
    for field in ("trainable_parameters", "optimizer_updates"):
        if completion.get(field, 0) != 0:
            raise ValueError("fixed diagnostic must not claim trained parameters or updates")
    if completion.get("classifier_training_run", False):
        raise ValueError("fixed diagnostic must not claim classifier training")
    resources = _json_value(resource_rows)
    json.dumps(
        {
            "config": _json_value(config),
            "completion": _json_value(completion),
            "resources": resources,
        },
        allow_nan=False,
    )
    (output / "figures").mkdir(exist_ok=True)
    for name, rows in zip(
        TABLES, (graph_rows, reconstruction_rows, condition_rows, relation_rows), strict=True
    ):
        _write_csv(output / f"{name}.csv", rows)
    with (output / "resources.json").open("x", encoding="utf-8") as stream:
        json.dump(resources, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    _figures(output, reconstruction_rows, condition_rows)

    actual = [row for row in graph_rows if row["actual_data"]]
    display_graphs = (
        actual or [r for r in graph_rows if r["feature_mode"] == "vector_trace"] or graph_rows
    )
    ids = {r["graph_id"] for r in display_graphs}
    display_reconstruction = [r for r in reconstruction_rows if r["graph_id"] in ids]
    conditions_by_key = {
        (r["graph_id"], r["weight"], _stage(r["state"]), r["condition"]): r for r in condition_rows
    }
    augmentation = []
    for row in display_reconstruction:
        key = row["graph_id"], row["weight"], _stage(row["state"])
        e = conditions_by_key[(*key, "sum_within")]
        j = conditions_by_key[(*key, "sum_between")]
        augmentation.append(
            {
                "graph_id": key[0],
                "weight": key[1],
                "state": row["state"],
                "E_norm_sq": e["energy_norm_sq"],
                "E_relative_error": e["energy_relative_error"],
                "J_norm_sq": j["relation_norm_sq"],
                "J_relative_error": j["relation_relative_error"],
                "additional_rank": 0,
            }
        )
    relations_by_key = {
        (
            r["graph_id"],
            r["weight"],
            _stage(r["stage_from"]),
            _stage(r["stage_to"]),
            r["relation"],
        ): r
        for r in relation_rows
    }
    relation_display = []
    for graph in display_graphs:
        for weight in WEIGHTS:
            for a, b in STAGE_PAIRS:
                row = {"graph_id": graph["graph_id"], "weight": weight, "stage": f"H{a}→H{b}"}
                for name, short in (
                    ("shared_edges", "shared"),
                    ("shared_nodes", "node"),
                    ("distinct_edges", "distinct"),
                ):
                    r = relations_by_key[graph["graph_id"], weight, a, b, name]
                    row[f"{short}_mean"] = r["original_sum"] / r["count"] if r["count"] else None
                    row[f"{short}_relative_error"] = r["relative_error"]
                    if name == "distinct_edges":
                        row["distinct_negative_percent"] = (
                            100 * r["negative_count"] / r["count"] if r["count"] else None
                        )
                relation_display.append(row)
    profile_note = (
        "DEBUG fixture 검증이며 실제 citation의 서버 FULL 결과가 아니다."
        if config["profile"] == "debug"
        else "검증된 이전 FULL 입력을 모두 사용한 고정 진단이다."
    )
    graph_table = _table(
        display_graphs,
        (
            "graph_id",
            "num_nodes",
            "num_edges",
            "num_features",
            "components",
            "tagged_coordinates",
            "fused_active_coordinates",
            "ambient_tag_kernel_per_channel",
            "restricted_rank_per_channel",
        ),
    )
    reconstruction_table = _table(
        display_reconstruction,
        (
            "graph_id",
            "weight",
            "state",
            "tag_contrast_fraction_sq",
            "tagged_q_relative_error",
            "fused_q_relative_error",
            "centered_h_relative_error",
            "tagged_reconstruction_relative_error",
            "fused_relative_residual",
            "solver_iterations_max",
        ),
    )
    augmentation_table = _table(
        augmentation,
        (
            "graph_id",
            "weight",
            "state",
            "E_norm_sq",
            "E_relative_error",
            "J_norm_sq",
            "J_relative_error",
            "additional_rank",
        ),
    )
    relation_table = _table(
        relation_display,
        (
            "graph_id",
            "weight",
            "stage",
            "shared_mean",
            "node_mean",
            "distinct_mean",
            "distinct_negative_percent",
            "shared_relative_error",
            "node_relative_error",
            "distinct_relative_error",
        ),
    )
    report = f"""# 수신 측 집계와 복원 진단

Profile: **{config["profile"]}**.
전체 그래프 **{len(graphs)}개**, 실제 citation **{len(actual)}개**.
Trainable parameters **0**, optimizer updates **0**, epoch **해당 없음**.
{profile_note}

## 이번 관측과 판정 범위

모든 원래 노드·물리 엣지·특징 채널, induced 1홉 로컬, 양방향 인접 중심 전달을 사용했다.
`tagged`는 송신별 receipt R, `sum`은 수신 로컬·원래 노드별 합 Y다.
`sum_within`, `sum_between`, `sum_both`는 실제 내부 이차 에너지 E와
incidence 대응 쌍선형 J를 관측에 붙였다.
H0/H1/H2는 기존 전체 그래프의 고정 reference states다.
이번 수신 합으로 업데이트한 학습 층이 아니다.

**receipt 합의 임의 입력 공간과 현재 실제 메시지 제약을 구분한다.**
합에 직접 보이지 않는 송신별 성분은 `R−Sᵀdiag(receipt_count)†Y`다.
그러나 실제 R은 알려진 양의 C 아래 한 전역 H에서 생성된다.
수신 중심 행이 원래 연결의 positive directed Laplacian을 포함한다.
따라서 실제 rank는 채널당 N−components다. 성분별 상수는 q/E/J에 영향을 주지 않는다.
실제 q의 추가 kernel 차원과 E/J의 추가 rank는 **0**이다.
아래 tag contrast 비율을 영구히 잃은 정보량으로 해석하면 안 된다.

## 모든 citation의 차원

{graph_table}

tagged/fused 좌표는 채널당 receipt/수신 행 수다. 실제 citation 관측은 전체 채널을 포함한다.
DEBUG이면 위 표의 입력은 DEBUG vector fixtures다.

## 송신별 상쇄와 실제 복원

{reconstruction_table}

tag contrast는 **제곱 norm 비율**이다. 다른 복원 지표는 norm 상대 오차다.
복원 target H는 연결성분 평균을 뺀 H다.
R의 복원은 source labels가 있는 전체 실제 receipt와 비교했다.
normal CG는 실제 Y 상대 잔차를 확인한다.
잔차와 H/q/R 오차는 conditioning 때문에 같은 수치일 필요가 없다.
0인 분모의 상대 지표는 `undefined`이며 절대 norm/error는 CSV에 있다.

## E/J 관측의 조건부 재현

{augmentation_table}

추가 관측은 실제 forward 출력에 연결됐다. **decoder는 동일한 Y만 사용한다.**
복원한 H에서 E/J를 다시 계산해 원래 E/J와 비교했다.
E/J를 inverse 입력에 사용해 개선한 복원 실험이 아니다.
이는 E/J가 현재 조건에서 독립 정보가 아니라 Y로 결정되는 비선형 좌표임을 검사한다.
`sum_both`의 E/J 검사는 각각 단독 추가 조건과 같으며, condition CSV에 다섯 조건 전체를 기록했다.

## 같은 단계와 연속 단계의 쌍선형 관계

{relation_table}

연속 단계의 J는 각 단계에서 복원한 두 H를 사용했다.
한 단계의 관측만으로 다른 단계를 독립적으로 관측한 것처럼 사용하지 않았다.
`J_node=2J_shared+J_distinct`이며 세 J는 독립 항 세 개가 아니다.
부호를 유지하며 전체 합의 PSD를 주장하지 않는다.
서로 다른 그래프의 raw trace를 분류 점수나 독립 seed 수로 비교하지 않는다.

## 전체 기록과 그림

- [모든 그래프·연결성분·rank](graph_summary.csv)
- [모든 그래프 × 두 C × 세 상태의 실제 복원](reconstruction_summary.csv)
- [다섯 관측 조건의 실제 E/J와 오차](condition_summary.csv)
- [모든 그래프의 같은 단계·연속 단계 관계와 재현 오차](relation_summary.csv)
- [실제 calibration·execution 자원](resources.json)

Macro rows: graph={len(graph_rows)}, reconstruction={len(reconstruction_rows)},
condition={len(condition_rows)}, relation={len(relation_rows)}.
그림은 전체 macro rows의 정의된 지표를 사용한다. 임의의 그래프/채널 subset을 만들지 않았다.
정의되지 않은 비율은 그림에서 제외한 수를 표시하고 0으로 채우지 않았다. PNG와 PDF를 각각 저장했다.

![tag contrast and actual q recovery](figures/tag_contrast_and_recovery.png)

![E/J reproduced from the Y-only inverse](figures/energy_relation_reconstruction.png)

## 다음 연구의 범위

이번 조건에서는 수신 합이 실제 q/E/J를 결정한다. E/J의 예측 효과, 학습된 C/W, source별 독립 상태,
일부 수신만 관측하는 모델이나 학습된 비선형 압축에는 이 결론을 자동으로 적용하지 않는다.
에너지·쌍선형 관계가 비선형 표현으로 학습에 도움을 주는지는 별도 실험에서 확인한다.
그 실험에는 실제 forward·loss·gradient·optimizer 경로와 예측 평가가 있어야 한다.
이번 완료를 분류 학습 완료나 예측 개선으로 보고하지 않는다.
"""
    with (output / "RECEIVER_AGGREGATION_SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    return {
        "artifacts": list(ARTIFACTS),
        "graph_count": len(graphs),
        "actual_citation_graphs": len(actual),
        "macro_rows": {
            name: len(rows)
            for name, rows in zip(
                TABLES,
                (graph_rows, reconstruction_rows, condition_rows, relation_rows),
                strict=True,
            )
        },
    }
