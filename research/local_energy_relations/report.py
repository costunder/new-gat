"""Measured fixed local-energy audits: summaries and exportable scientific figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

WEIGHTS = ("unit", "local_degree")
STATES = (0, 1, 2)
STAGE_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2))
RELATIONS = ("shared_edges", "shared_nodes", "distinct_edges")
TABLES = (
    "graph_summary",
    "local_summary",
    "relation_summary",
    "transfer_summary",
    "assembly_summary",
)
FIGURES = ("energy_vs_flow", "cycle_and_reconstruction", "signed_relations")
ARTIFACTS = (
    "LOCAL_ENERGY_RELATIONS_SUMMARY.md",
    *(f"{name}.csv" for name in TABLES),
    "resources.json",
    *(f"figures/{name}.{suffix}" for name in FIGURES for suffix in ("png", "pdf")),
)


def _number(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    numeric = _number(value, label)
    if numeric is None or numeric < int(positive) or not numeric.is_integer():
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(numeric)


def _nonnegative(row: dict, field: str, label: str) -> float:
    value = _number(row.get(field), f"{label}.{field}")
    if value is None or value < 0:
        raise ValueError(f"{label}.{field} must be nonnegative")
    return value


def _close(a: float, b: float, label: str) -> None:
    if not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(f"{label} identity mismatch: {a!r} versus {b!r}")


def _metadata(row: dict, graphs: dict[str, dict], label: str) -> tuple[str, str]:
    graph_id = row.get("graph_id")
    if graph_id not in graphs:
        raise ValueError(f"{label}: unknown graph {graph_id!r}")
    weight = row.get("weight")
    if weight not in WEIGHTS:
        raise ValueError(f"{label}: unknown weight {weight!r}")
    for field in ("family", "dataset"):
        if field in row and row[field] != graphs[graph_id][field]:
            raise ValueError(f"{label}: mismatched {field}")
    return str(graph_id), str(weight)


def _stage(value: Any, label: str) -> int:
    if isinstance(value, str) and value in ("H0", "H1", "H2"):
        return int(value[1])
    return _integer(value, label)


def _state(row: dict, label: str) -> int:
    value = _stage(row.get("state"), f"{label}.state")
    if value not in STATES:
        raise ValueError(f"{label}: state must be 0, 1 or 2")
    return value


def _ensure_coverage(actual: set[tuple], expected: set[tuple], label: str) -> None:
    if actual != expected:
        missing, extra = expected - actual, actual - expected
        raise ValueError(
            f"{label} coverage mismatch: missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )


def validate_rows(
    config: dict,
    graph_rows: list[dict],
    local_rows: list[dict],
    relation_rows: list[dict],
    transfer_rows: list[dict],
    assembly_rows: list[dict],
) -> dict[str, dict]:
    """Check measured row semantics and all graph/weight/state coverage before writing."""
    profile = config.get("profile")
    if profile not in ("full", "debug"):
        raise ValueError("report profile must explicitly be full or debug")
    expected_graphs = 201 if profile == "full" else 21
    if len(graph_rows) != expected_graphs:
        raise ValueError(
            f"{profile} requires {expected_graphs} measured graphs, received {len(graph_rows)}"
        )
    graphs: dict[str, dict] = {}
    actual_count = 0
    for index, row in enumerate(graph_rows):
        label = f"graphs[{index}]"
        graph_id = row.get("graph_id")
        if not isinstance(graph_id, str) or not graph_id:
            raise ValueError(f"{label}: graph_id must be nonempty")
        if graph_id in graphs:
            raise ValueError(f"duplicate graph_id: {graph_id}")
        for field in ("family", "dataset"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{label}: {field} must be nonempty")
        for field in ("num_nodes", "num_features"):
            _integer(row.get(field), f"{label}.{field}", positive=True)
        _integer(row.get("num_edges"), f"{label}.num_edges")
        if not isinstance(row.get("actual_data"), bool):
            raise ValueError(f"{label}: actual_data must be explicit")
        actual_count += int(row["actual_data"])
        if row.get("feature_mode") not in ("independent_scalar_columns", "vector_trace"):
            raise ValueError(f"{label}: unknown feature_mode")
        if row["actual_data"] and row["feature_mode"] != "vector_trace":
            raise ValueError(f"{label}: actual citation channels must use vector_trace")
        if "sampling_ratio" in row and row["sampling_ratio"] != 1:
            raise ValueError(f"{label}: sampling is forbidden in this audit")
        graphs[graph_id] = row
    if profile == "full" and actual_count != 3:
        raise ValueError("full requires three actual citation graphs")
    if profile == "debug" and actual_count:
        raise ValueError("DEBUG fixtures must not be reported as actual citation data")

    all_states = {(g, w, t) for g in graphs for w in WEIGHTS for t in STATES}
    seen_local: set[tuple] = set()
    local_by_key: dict[tuple, dict] = {}
    for index, row in enumerate(local_rows):
        label = f"local[{index}]"
        graph, weight = _metadata(row, graphs, label)
        state = _state(row, label)
        key = (graph, weight, state)
        if key in seen_local:
            raise ValueError(f"duplicate local summary: {key}")
        seen_local.add(key)
        meta = graphs[graph]
        units = meta["num_nodes"] * (
            meta["num_features"] if meta["feature_mode"] == "independent_scalar_columns" else 1
        )
        if _integer(row.get("count"), f"{label}.count") != units:
            raise ValueError(
                f"{label}: count must include all centers and the declared feature mode"
            )
        energy = _nonnegative(row, "energy_sum", label)
        flow = _nonnegative(row, "flow_norm_sq_sum", label)
        _nonnegative(row, "divergence_norm_sq_sum", label)
        cycle = _nonnegative(row, "cycle_norm_sq_sum", label)
        _nonnegative(row, "reconstruction_error_sq_sum", label)
        fraction = _number(
            row.get("cycle_fraction_sq"), f"{label}.cycle_fraction_sq", nullable=True
        )
        recon = _number(
            row.get("reconstruction_relative_error_max"),
            f"{label}.reconstruction_relative_error_max",
            nullable=True,
        )
        if flow == 0:
            if fraction is not None or recon is not None:
                raise ValueError(f"{label}: zero-flow relative metrics must be undefined")
            _close(cycle, 0.0, f"{label}.zero_flow_cycle")
        else:
            if fraction is None or recon is None or fraction < 0 or recon < 0:
                raise ValueError(
                    f"{label}: positive-flow relative metrics must be defined and nonnegative"
                )
            _close(fraction, cycle / flow, f"{label}.cycle_fraction_sq")
            if fraction > 1 + 1e-8:
                raise ValueError(f"{label}: Euclidean cycle fraction exceeds one")
        if weight == "unit":
            _close(energy, flow, f"{label}.unit_energy_flow")
        local_by_key[key] = row
    _ensure_coverage(seen_local, all_states, "local")

    expected_relations = {
        (g, w, a, b, r) for g in graphs for w in WEIGHTS for a, b in STAGE_PAIRS for r in RELATIONS
    }
    expected_relations |= {
        (g, w, a, b, "same_local") for g in graphs for w in WEIGHTS for a, b in ((0, 1), (1, 2))
    }
    seen_relations: set[tuple] = set()
    relation_by_key: dict[tuple, dict] = {}
    for index, row in enumerate(relation_rows):
        label = f"relations[{index}]"
        graph, weight = _metadata(row, graphs, label)
        a = _stage(row.get("stage_from"), f"{label}.stage_from")
        b = _stage(row.get("stage_to"), f"{label}.stage_to")
        relation = row.get("relation")
        key = (graph, weight, a, b, relation)
        if key in seen_relations:
            raise ValueError(f"duplicate relation summary: {key}")
        seen_relations.add(key)
        if relation not in (*RELATIONS, "same_local"):
            raise ValueError(f"{label}: unknown relation")
        meta = graphs[graph]
        units = meta["num_nodes"] if relation == "same_local" else 2 * meta["num_edges"]
        units *= meta["num_features"] if meta["feature_mode"] == "independent_scalar_columns" else 1
        count = _integer(row.get("count"), f"{label}.count")
        if count != units:
            raise ValueError(
                f"{label}: count must cover every directed center pair and feature unit"
            )
        total = _number(row.get("sum"), f"{label}.sum")
        stats = [
            _number(row.get(field), f"{label}.{field}", nullable=True)
            for field in ("mean", "min", "max")
        ]
        negatives = _integer(row.get("negative_count"), f"{label}.negative_count")
        if negatives > count:
            raise ValueError(f"{label}: negative_count exceeds count")
        if count == 0:
            if stats != [None, None, None] or total != 0 or negatives:
                raise ValueError(f"{label}: empty relation statistics must be explicit")
        else:
            if any(v is None for v in stats):
                raise ValueError(f"{label}: measured relation statistics must be defined")
            mean, minimum, maximum = stats
            _close(float(mean), float(total) / count, f"{label}.mean")
            tolerance = 1e-12 + 1e-8 * max(abs(float(minimum)), abs(float(maximum)))
            if float(minimum) - tolerance > float(mean) or float(mean) > float(maximum) + tolerance:
                raise ValueError(f"{label}: mean lies outside min/max")
        relation_by_key[key] = row
    _ensure_coverage(seen_relations, expected_relations, "relation")
    for graph in graphs:
        for weight in WEIGHTS:
            for a, b in STAGE_PAIRS:
                shared = relation_by_key[graph, weight, a, b, "shared_edges"]["sum"]
                node = relation_by_key[graph, weight, a, b, "shared_nodes"]["sum"]
                distinct = relation_by_key[graph, weight, a, b, "distinct_edges"]["sum"]
                _close(
                    float(node),
                    2 * float(shared) + float(distinct),
                    f"{graph}/{weight}/{a}/{b}.relation_decomposition",
                )

    seen_transfer: set[tuple] = set()
    for index, row in enumerate(transfer_rows):
        label = f"transfer[{index}]"
        graph, weight = _metadata(row, graphs, label)
        state = _state(row, label)
        key = (graph, weight, state)
        if key in seen_transfer:
            raise ValueError(f"duplicate transfer summary: {key}")
        seen_transfer.add(key)
        meta = graphs[graph]
        units = (
            2
            * meta["num_edges"]
            * (meta["num_features"] if meta["feature_mode"] == "independent_scalar_columns" else 1)
        )
        if _integer(row.get("count"), f"{label}.count") != units:
            raise ValueError(f"{label}: count must cover all directed center pairs")
        values = {
            field: _nonnegative(row, field, label)
            for field in (
                "sender_divergence_norm_sq_sum",
                "retained_divergence_norm_sq_sum",
                "omitted_divergence_norm_sq_sum",
                "sender_flow_norm_sq_sum",
                "retained_flow_norm_sq_sum",
                "boundary_flow_norm_sq_sum",
                "omitted_flow_norm_sq_sum",
            )
        }
        _close(
            values["sender_divergence_norm_sq_sum"],
            values["retained_divergence_norm_sq_sum"] + values["omitted_divergence_norm_sq_sum"],
            f"{label}.divergence_partition",
        )
        _close(
            values["sender_flow_norm_sq_sum"],
            values["retained_flow_norm_sq_sum"]
            + values["boundary_flow_norm_sq_sum"]
            + values["omitted_flow_norm_sq_sum"],
            f"{label}.flow_partition",
        )
    _ensure_coverage(seen_transfer, all_states, "transfer")

    seen_assembly: set[tuple] = set()
    for index, row in enumerate(assembly_rows):
        label = f"assembly[{index}]"
        graph, weight = _metadata(row, graphs, label)
        state = _state(row, label)
        key = (graph, weight, state)
        if key in seen_assembly:
            raise ValueError(f"duplicate assembly summary: {key}")
        seen_assembly.add(key)
        values = {
            field: _nonnegative(row, field, label)
            for field in (
                "energy_raw_local_sum",
                "energy_overlap_corrected_sum",
                "energy_physical_average_sum",
                "energy_physical_unit_sum",
                "identity_abs_error_max",
            )
        }
        _close(
            values["energy_raw_local_sum"], local_by_key[key]["energy_sum"], f"{label}.local_sum"
        )
        _close(
            values["energy_overlap_corrected_sum"],
            values["energy_physical_average_sum"],
            f"{label}.overlap_identity",
        )
        if weight == "unit":
            _close(
                values["energy_physical_average_sum"],
                values["energy_physical_unit_sum"],
                f"{label}.unit_assembly",
            )
    _ensure_coverage(seen_assembly, all_states, "assembly")
    return graphs


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: Any) -> Any:
    """Keep measured NumPy scalar values numeric without hiding unsupported types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _format(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, (float, np.floating)):
        return f"{value:.8g}"
    return str(value).replace("|", "\\|")


def _table(rows: list[dict], fields: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend(
        "| " + " | ".join(_format(row.get(field)) for field in fields) + " |" for row in rows
    )
    return "\n".join(lines)


def _figures(
    output: Path, graphs: dict[str, dict], local_rows: list[dict], relation_rows: list[dict]
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = {"unit": "#0072B2", "local_degree": "#D55E00"}
    markers = {0: "o", 1: "s", 2: "^"}
    families = sorted({row["family"] for row in graphs.values()})

    def panels(title: str) -> tuple[Any, list[Any]]:
        columns = min(3, len(families))
        figure, axes = plt.subplots(
            math.ceil(len(families) / columns),
            columns,
            figsize=(5 * columns, 3.7 * math.ceil(len(families) / columns)),
            squeeze=False,
        )
        visible = list(axes.flat)[: len(families)]
        for axis, family in zip(visible, families, strict=True):
            axis.set_title(family)
            axis.grid(alpha=0.2)
        for axis in list(axes.flat)[len(families) :]:
            axis.set_visible(False)
        figure.suptitle(title, fontsize=13)
        return figure, visible

    def scale(axis: Any, values: list[float], coordinate: str) -> None:
        nonzero = [abs(float(value)) for value in values if value]
        maximum = max(nonzero, default=0.0)
        setter = axis.set_xscale if coordinate == "x" else axis.set_yscale
        target = axis.xaxis if coordinate == "x" else axis.yaxis
        if maximum and maximum / min(nonzero) > 1000:
            setter("symlog", linthresh=maximum / 1000)
        else:
            setter("linear")
            target.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(axis=coordinate, labelsize=8)

    def save(figure: Any, axes: list[Any], name: str) -> None:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        figure.tight_layout(rect=(0, 0.08, 1, 0.95))
        for suffix in ("png", "pdf"):
            with (output / "figures" / f"{name}.{suffix}").open("xb") as stream:
                figure.savefig(stream, format=suffix, dpi=180, bbox_inches="tight")
        plt.close(figure)

    fig, axes = panels("All graph/state summaries; energy and flow norm squared differ")
    for ax, family in zip(axes, families, strict=True):
        family_rows = [row for row in local_rows if graphs[row["graph_id"]]["family"] == family]
        for weight in WEIGHTS:
            for state in STATES:
                rows = [
                    row
                    for row in family_rows
                    if row["weight"] == weight and _state(row, "figure") == state
                ]
                ax.scatter(
                    [r["energy_sum"] for r in rows],
                    [r["flow_norm_sq_sum"] for r in rows],
                    s=[55 if graphs[r["graph_id"]]["actual_data"] else 18 for r in rows],
                    alpha=0.65,
                    color=colors[weight],
                    marker=markers[state],
                    label=f"{weight}, H{state}",
                )
        values = [
            float(row[field]) for row in family_rows for field in ("energy_sum", "flow_norm_sq_sum")
        ]
        maximum = max(values, default=0.0)
        ax.plot(
            [0, maximum],
            [0, maximum],
            color="0.4",
            linewidth=0.8,
            linestyle="--",
            label="flow squared = energy",
        )
        scale(ax, [row["energy_sum"] for row in family_rows], "x")
        scale(ax, [row["flow_norm_sq_sum"] for row in family_rows], "y")
        ax.set_xlabel("Local energy sum: C")
        ax.set_ylabel("Flow norm squared sum: C squared")
    save(fig, axes, "energy_vs_flow")

    undefined = sum(row["cycle_fraction_sq"] is None for row in local_rows)
    fig, axes = panels(
        f"Euclidean cycle and known-C recovery; {undefined} zero-flow rows undefined"
    )
    for ax, family in zip(axes, families, strict=True):
        for weight in WEIGHTS:
            for state in STATES:
                rows = [
                    r
                    for r in local_rows
                    if graphs[r["graph_id"]]["family"] == family
                    and r["weight"] == weight
                    and _state(r, "figure") == state
                    and r["cycle_fraction_sq"] is not None
                ]
                ax.scatter(
                    [r["cycle_fraction_sq"] for r in rows],
                    [r["reconstruction_relative_error_max"] for r in rows],
                    s=[55 if graphs[r["graph_id"]]["actual_data"] else 18 for r in rows],
                    alpha=0.65,
                    color=colors[weight],
                    marker=markers[state],
                    label=f"{weight}, H{state}",
                )
        errors = [
            row["reconstruction_relative_error_max"]
            for row in local_rows
            if graphs[row["graph_id"]]["family"] == family
            and row["reconstruction_relative_error_max"] is not None
        ]
        scale(ax, errors, "y")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Cycle squared / flow squared")
        ax.set_ylabel("Maximum known-C relative error")
    save(fig, axes, "cycle_and_reconstruction")

    fig, axes = panels("Graph-specific signed means: same and cross reference stages")
    indexed = {
        (r["graph_id"], r["weight"], r["stage_from"], r["stage_to"], r["relation"]): r
        for r in relation_rows
    }
    for ax, family in zip(axes, families, strict=True):
        for weight in WEIGHTS:
            for cross in (False, True):
                pairs = [
                    (
                        r,
                        indexed[
                            r["graph_id"], weight, r["stage_from"], r["stage_to"], "distinct_edges"
                        ],
                    )
                    for r in relation_rows
                    if graphs[r["graph_id"]]["family"] == family
                    and r["weight"] == weight
                    and r["relation"] == "shared_edges"
                    and (r["stage_from"] != r["stage_to"]) == cross
                    and r["count"]
                ]
                ax.scatter(
                    [a["mean"] for a, _ in pairs],
                    [b["mean"] for _, b in pairs],
                    s=[55 if graphs[a["graph_id"]]["actual_data"] else 18 for a, _ in pairs],
                    alpha=0.65,
                    marker="^" if cross else "o",
                    color=colors[weight],
                    label=f"{weight}, {'cross stage' if cross else 'same stage'}",
                )
        family_shared = [
            row["mean"]
            for row in relation_rows
            if graphs[row["graph_id"]]["family"] == family
            and row["relation"] == "shared_edges"
            and row["mean"] is not None
        ]
        family_distinct = [
            row["mean"]
            for row in relation_rows
            if graphs[row["graph_id"]]["family"] == family
            and row["relation"] == "distinct_edges"
            and row["mean"] is not None
        ]
        scale(ax, family_shared, "x")
        scale(ax, family_distinct, "y")
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.axvline(0, color="0.4", linewidth=0.8)
        ax.set_xlabel("Mean shared-edge relation")
        ax.set_ylabel("Mean distinct-edge relation")
    save(fig, axes, "signed_relations")


def write_report(
    output_dir: str | Path,
    config: dict,
    graph_rows: list[dict],
    local_rows: list[dict],
    relation_rows: list[dict],
    transfer_rows: list[dict],
    assembly_rows: list[dict],
    resource_rows: list[dict],
    completion: dict,
) -> dict:
    """Write all macro rows and scientific figures; never rewrite raw runner CSVs."""
    output = Path(output_dir)
    if not output.is_dir():
        raise ValueError("report output directory must already exist")
    occupied = [name for name in ARTIFACTS if (output / name).exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite report artifacts: {occupied}")
    graphs = validate_rows(
        config, graph_rows, local_rows, relation_rows, transfer_rows, assembly_rows
    )
    for field in ("trainable_parameters", "optimizer_updates"):
        if field in completion and completion[field] != 0:
            raise ValueError(f"fixed audit completion requires {field}=0")
    if completion.get("classifier_training_run", False):
        raise ValueError("fixed audit must not claim classifier training")
    resources = _json_value(resource_rows)
    json.dumps(
        {
            "resources": resources,
            "completion": _json_value(completion),
            "config": _json_value(config),
        },
        allow_nan=False,
    )
    (output / "figures").mkdir(exist_ok=True)
    for name, rows in zip(
        TABLES, (graph_rows, local_rows, relation_rows, transfer_rows, assembly_rows), strict=True
    ):
        _write_csv(output / f"{name}.csv", rows)
    with (output / "resources.json").open("x", encoding="utf-8") as stream:
        json.dump(resources, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    _figures(output, graphs, local_rows, relation_rows)
    actual = [r for r in graph_rows if r["actual_data"]]
    citation_ids = {r["graph_id"] for r in actual}
    display_local = [r for r in local_rows if r["graph_id"] in citation_ids]
    if not display_local:
        display_local = local_rows
    display_assembly = [r for r in assembly_rows if r["graph_id"] in citation_ids]
    if not display_assembly:
        display_assembly = assembly_rows
    local_fields = (
        "graph_id",
        "weight",
        "state",
        "count",
        "energy_sum",
        "flow_norm_sq_sum",
        "cycle_fraction_sq",
        "reconstruction_relative_error_max",
    )
    assembly_fields = (
        "graph_id",
        "weight",
        "state",
        "energy_raw_local_sum",
        "energy_overlap_corrected_sum",
        "energy_physical_average_sum",
        "energy_physical_unit_sum",
        "identity_abs_error_max",
    )
    profile_note = (
        "DEBUG fixture 결과이며 서버 FULL 실제 데이터 완료를 뜻하지 않는다."
        if config["profile"] == "debug"
        else (
            "FULL 원래 입력의 측정 결과다. 실행 성공 여부와 CG/coverage 검사는 "
            "completion 기록을 함께 확인한다."
        )
    )
    report = f"""# 로컬 에너지·집합 사이 관계 감사

Profile: **{config["profile"]}**. 측정 그래프 {len(graphs)}개, 실제 citation {len(actual)}개.
Trainable parameters: **0**. Epoch/optimizer updates: **해당 없음**.
모든 원래 노드·엣지·특징을 사용한 고정 연산 감사다. 분류 성능 결과가 아니다.
{profile_note}

## 계산과 해석

각 중심의 1홉 **induced graph**에는 이웃끼리의 원래 엣지도 들어 있다.
`q=C BH`, `d=Bᵀq`이며 내부 에너지는 `Σc||BH||²`, flow 제곱 norm은 `Σc²||BH||²`다.
서로 다른 로컬 그래프의 관계는 공통 엣지 내적과 `qvᵀ(Bv_global Bu_globalᵀ)qu`다.
`Jnode=2Jshared+Jdistinct`이며 distinct 항의 부호를 유지한다.
H0/H1/H2는 공통 global Laplacian 좌표의 두 고정 확산 단계다. 학습 층이 아니다.

Euclidean cycle 성분은 실제 q에서 계산했다. 알려진 양의 C와 모든 로컬 노드 집계를
관측한 `q=CBH`의 복원은 별도 검사다.
cycle 성분이 있는 것을 복구 불가능한 손실과 같게 해석하지 않는다.
전달 표는 공통 노드로 복사한 d와 남지 않은 d를 기록한다. 부분 관측 역복원을 수행하지 않았다.
이 복사는 bookkeeping·진단이다. 복사한 d를 H_next에 사용하지 않았으며,
H1/H2는 전체 물리 라플라시안의 고정 reference 확산에서 얻었다.

## 로컬 에너지와 복원

{_table(display_local, local_fields)}

실제 citation은 모든 특징 채널을 합한 vector trace이며 각 채널을 독립 그래프로 세지 않는다.
합성 데이터의 count는 모든 중심 × 독립 scalar columns다. 서로 다른 입력의 raw 에너지를
같은 분류 점수로 비교하거나 occurrence 수를 독립 학습 seed 수로 사용하지 않는다.

## 중복 집계

{_table(display_assembly, assembly_fields)}

raw 합은 여러 ego에서 같은 물리 엣지를 반복해서 센다.
inverse-occurrence 보정은 물리 엣지별 평균 C를 사용한 에너지와 일치한다.
unit 조건에서만 이 보정 에너지가 원래 unit 물리 에너지와 같다.

## 모든 측정과 자원

- [그래프 입력·feature mode](graph_summary.csv)
- [모든 로컬 에너지·cycle·known-C 복원](local_summary.csv)
- [모든 방향·같은 단계/교차 단계 관계와 부호](relation_summary.csv)
- [노드 전달 및 retained/boundary/omitted 엣지 norm](transfer_summary.csv)
- [물리 에너지와 중복 보정](assembly_summary.csv)
- [실제 calibration·execution 자원 기록](resources.json)

과학 그림은 모든 graph macro rows를 사용한다. 큰 점은 실제 citation이다.
zero-flow에서 정의되지 않는 상대 비율은 0으로 채우지 않고 undefined로 기록한다.

![energy versus flow](figures/energy_vs_flow.png)

![cycle versus restricted recovery](figures/cycle_and_reconstruction.png)

![signed local relations](figures/signed_relations.png)

PNG와 같은 이름의 PDF는 독립적으로 저장했다.
CSV 수: graph={len(graph_rows)}, local={len(local_rows)}, relation={len(relation_rows)},
transfer={len(transfer_rows)}, assembly={len(assembly_rows)}.

## 남은 연구 질문

C/W는 학습하지 않았다. encoder·분류기·CE·예측에 연결한 관계 항도 없다.
항등식 검사를 통과한 것을 학습된 새 메시지 패싱의 성능, GATv2 대체, 전체 cycle 정보 복원의
성공이나 선행 연구 대비 신규성으로 주장하지 않는다. 다음 학습 범위는 이 감사 결과를 읽은 뒤 정한다.
"""
    with (output / "LOCAL_ENERGY_RELATIONS_SUMMARY.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    return {
        "artifacts": list(ARTIFACTS),
        "graph_count": len(graphs),
        "actual_citation_graphs": len(actual),
        "macro_rows": {
            name: len(rows)
            for name, rows in zip(
                TABLES,
                (graph_rows, local_rows, relation_rows, transfer_rows, assembly_rows),
                strict=True,
            )
        },
    }
