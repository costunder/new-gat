"""Explicit DEBUG arithmetic fixtures from dense incidence formulas, never training."""

import copy
import csv
import json

import numpy as np
import pytest

from research.local_energy_relations.report import ARTIFACTS, validate_rows, write_report


def _summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "sum": 0.0, "mean": None, "min": None, "max": None, "negative_count": 0}
    return {
        "count": len(values),
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "negative_count": int((values < 0).sum()),
    }


def dense_rows(*, zero=False):
    """21 named DEBUG arithmetic graphs; dense formulas independently measure every row."""
    graphs, local, relations, transfers, assembly = [], [], [], [], []
    edges = np.array([(0, 1), (0, 2), (1, 2), (2, 3)], dtype=np.int64)
    n, channels = 4, 2
    b = np.zeros((len(edges), n))
    for e, (a, z) in enumerate(edges):
        b[e, a], b[e, z] = -1.0, 1.0
    physical_degree = np.abs(b).sum(axis=0)
    base = np.array([[1.0, 3.0], [-2.0, 0.0], [3.0, -1.0], [5.0, -2.0]])
    h0 = np.zeros_like(base) if zero else base
    states = [h0]
    for _ in range(2):
        states.append(states[-1] - b.T @ b @ states[-1] / (2 * physical_degree.max()))
    nodes, ids, local_b = [], [], []
    for v in range(n):
        members = {v}
        members.update(edges[(edges == v).any(axis=1)].reshape(-1).tolist())
        nodes.append(sorted(members))
        occurrence = [e for e, (a, z) in enumerate(edges) if a in members and z in members]
        ids.append(occurrence)
        local_b.append(b[occurrence][:, sorted(members)])
    multiplicity = np.bincount(np.concatenate(ids), minlength=len(edges))
    directed = [tuple(pair) for a, z in edges for pair in ((a, z), (z, a))]
    for graph_index in range(21):
        mode = "independent_scalar_columns" if graph_index < 18 else "vector_trace"
        graph_id = f"DEBUG-arithmetic-{graph_index}"
        meta = {
            "graph_id": graph_id,
            "family": "triangle_tail",
            "dataset": "DEBUG-arithmetic",
            "actual_data": False,
            "feature_mode": mode,
            "num_nodes": n,
            "num_edges": len(edges),
            "num_features": channels,
            "sampling_ratio": 1.0,
        }
        graphs.append(meta)
        for weight in ("unit", "local_degree"):
            weights = []
            for bv in local_b:
                degree = np.abs(bv).sum(axis=0)
                local_ends = [np.flatnonzero(row) for row in bv]
                weights.append(
                    np.ones(len(bv))
                    if weight == "unit"
                    else np.array([2.0 / degree[endpoints].sum() for endpoints in local_ends])
                )
            qs, ds = [], []
            for t, h in enumerate(states):
                q_stage, d_stage, energy, flow, divergence, cycle, error, ratios = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                physical_average = np.zeros(len(edges))
                corrected = 0.0
                for v, (bv, cv) in enumerate(zip(local_b, weights, strict=True)):
                    g = bv @ h[nodes[v]]
                    q = cv[:, None] * g
                    d = bv.T @ q
                    q_cycle = q - bv @ np.linalg.pinv(bv.T @ bv) @ d
                    q_recon = cv[:, None] * (bv @ np.linalg.pinv(bv.T @ (cv[:, None] * bv)) @ d)
                    dg = np.zeros((n, channels))
                    dg[nodes[v]] = d
                    q_stage.append(q)
                    d_stage.append(dg)
                    energy.append(float((cv[:, None] * g**2).sum()))
                    flow.append(float((q**2).sum()))
                    divergence.append(float((d**2).sum()))
                    cycle.append(float((q_cycle**2).sum()))
                    error.append(float(((q_recon - q) ** 2).sum()))
                    if mode == "vector_trace":
                        denominator = np.linalg.norm(q)
                        if denominator:
                            ratios.append(float(np.linalg.norm(q_recon - q) / denominator))
                    else:
                        denominator = np.linalg.norm(q, axis=0)
                        ratios.extend(
                            (
                                np.linalg.norm(q_recon - q, axis=0)[denominator > 0]
                                / denominator[denominator > 0]
                            ).tolist()
                        )
                    physical_average[ids[v]] += cv / multiplicity[ids[v]]
                    corrected += float(((cv / multiplicity[ids[v]])[:, None] * g**2).sum())
                qs.append(q_stage)
                ds.append(d_stage)
                total_flow = sum(flow)
                common = {
                    "graph_id": graph_id,
                    "family": meta["family"],
                    "dataset": meta["dataset"],
                    "weight": weight,
                    "state": t,
                }
                local.append(
                    {
                        **common,
                        "count": n * (channels if mode == "independent_scalar_columns" else 1),
                        "energy_sum": sum(energy),
                        "flow_norm_sq_sum": total_flow,
                        "divergence_norm_sq_sum": sum(divergence),
                        "cycle_norm_sq_sum": sum(cycle),
                        "cycle_fraction_sq": sum(cycle) / total_flow if total_flow else None,
                        "reconstruction_error_sq_sum": sum(error),
                        "reconstruction_relative_error_max": max(ratios) if ratios else None,
                    }
                )
                physical_difference = b @ h
                average_energy = float((physical_average[:, None] * physical_difference**2).sum())
                physical_unit = float((physical_difference**2).sum())
                assembly.append(
                    {
                        **common,
                        "energy_raw_local_sum": sum(energy),
                        "energy_overlap_corrected_sum": corrected,
                        "energy_physical_average_sum": average_energy,
                        "energy_physical_unit_sum": physical_unit,
                        "identity_abs_error_max": abs(corrected - average_energy),
                    }
                )
                transfer = {
                    field: 0.0
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
                for v, u in directed:
                    common_nodes = set(nodes[v]) & set(nodes[u])
                    missing_nodes = set(nodes[v]) - common_nodes
                    dv = d_stage[v]
                    transfer["sender_divergence_norm_sq_sum"] += float((dv**2).sum())
                    transfer["retained_divergence_norm_sq_sum"] += float(
                        (dv[sorted(common_nodes)] ** 2).sum()
                    )
                    transfer["omitted_divergence_norm_sq_sum"] += float(
                        (dv[sorted(missing_nodes)] ** 2).sum()
                    )
                    transfer["sender_flow_norm_sq_sum"] += float((q_stage[v] ** 2).sum())
                    for index, e in enumerate(ids[v]):
                        shared_ends = sum(int(end in common_nodes) for end in edges[e])
                        partition = (
                            "retained"
                            if shared_ends == 2
                            else "boundary"
                            if shared_ends == 1
                            else "omitted"
                        )
                        transfer[f"{partition}_flow_norm_sq_sum"] += float(
                            (q_stage[v][index] ** 2).sum()
                        )
                transfers.append(
                    {
                        **common,
                        "count": len(directed)
                        * (channels if mode == "independent_scalar_columns" else 1),
                        **transfer,
                    }
                )
            for a, z in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2)):
                records = {name: [] for name in ("shared_edges", "shared_nodes", "distinct_edges")}
                for v, u in directed:
                    shared = np.zeros(channels)
                    for e in set(ids[v]) & set(ids[u]):
                        shared += qs[a][v][ids[v].index(e)] * qs[z][u][ids[u].index(e)]
                    node = (ds[a][v] * ds[z][u]).sum(axis=0)
                    for name, value in (
                        ("shared_edges", shared),
                        ("shared_nodes", node),
                        ("distinct_edges", node - 2 * shared),
                    ):
                        records[name].extend(
                            value.tolist()
                            if mode == "independent_scalar_columns"
                            else [float(value.sum())]
                        )
                for name, values in records.items():
                    relations.append(
                        {
                            "graph_id": graph_id,
                            "family": meta["family"],
                            "dataset": meta["dataset"],
                            "weight": weight,
                            "stage_from": a,
                            "stage_to": z,
                            "relation": name,
                            **_summary(values),
                        }
                    )
            for a, z in ((0, 1), (1, 2)):
                records = []
                for v in range(n):
                    value = (qs[a][v] * qs[z][v]).sum(axis=0)
                    records.extend(
                        value.tolist()
                        if mode == "independent_scalar_columns"
                        else [float(value.sum())]
                    )
                relations.append(
                    {
                        "graph_id": graph_id,
                        "family": meta["family"],
                        "dataset": meta["dataset"],
                        "weight": weight,
                        "stage_from": a,
                        "stage_to": z,
                        "relation": "same_local",
                        **_summary(records),
                    }
                )
    return {"profile": "debug"}, graphs, local, relations, transfers, assembly


def test_dense_formula_contract_and_signed_distinct_relations():
    config, graphs, local, relations, transfers, assembly = dense_rows()
    assert len(validate_rows(config, graphs, local, relations, transfers, assembly)) == 21
    assert any(row["relation"] == "distinct_edges" and row["negative_count"] for row in relations)
    weighted = [row for row in local if row["weight"] == "local_degree"]
    assert any(row["energy_sum"] > row["flow_norm_sq_sum"] for row in weighted)
    assert any(row["cycle_fraction_sq"] > 1e-5 for row in weighted)
    assert max(row["reconstruction_relative_error_max"] for row in weighted) < 1e-12
    scalar, vector = local[0], local[-6]
    assert scalar["count"] == 8 and vector["count"] == 4


def _state_strings(rows):
    for index in (2, 4, 5):
        for row in rows[index]:
            row["state"] = f"H{row['state']}"
    for row in rows[3]:
        for field in ("stage_from", "stage_to"):
            row[field] = f"H{row[field]}"
    return rows


def test_runner_H_state_strings_preserve_all_incidence_identities():
    validate_rows(*_state_strings(dense_rows()))


def test_zero_flows_keep_relative_metrics_undefined():
    rows = dense_rows(zero=True)
    validate_rows(*rows)
    assert all(row["cycle_fraction_sq"] is None for row in rows[2])
    assert all(row["reconstruction_relative_error_max"] is None for row in rows[2])
    broken = copy.deepcopy(rows)
    broken[2][0]["cycle_fraction_sq"] = 0.0
    with pytest.raises(ValueError, match="undefined"):
        validate_rows(*broken)


@pytest.mark.parametrize("table_index", [1, 2, 3, 4, 5])
def test_every_macro_table_requires_complete_coverage(table_index):
    rows = dense_rows()
    rows[table_index].pop()
    with pytest.raises(ValueError, match="coverage|requires"):
        validate_rows(*rows)


@pytest.mark.parametrize("table_index", [1, 2, 3, 4, 5])
def test_duplicate_rows_are_rejected(table_index):
    rows = dense_rows()
    rows[table_index].append(copy.deepcopy(rows[table_index][0]))
    with pytest.raises(ValueError, match="duplicate|requires"):
        validate_rows(*rows)


@pytest.mark.parametrize(
    "table_index,field",
    [
        (2, "energy_sum"),
        (3, "sum"),
        (4, "sender_flow_norm_sq_sum"),
        (5, "energy_physical_average_sum"),
    ],
)
def test_nonfinite_measurements_do_not_become_zero(table_index, field):
    rows = dense_rows()
    rows[table_index][0][field] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_rows(*rows)


@pytest.mark.parametrize("table_index,field", [(2, "count"), (3, "count"), (4, "count")])
def test_feature_and_directed_pair_count_contract(table_index, field):
    rows = dense_rows()
    rows[table_index][0][field] -= 1
    with pytest.raises(ValueError, match="count"):
        validate_rows(*rows)


@pytest.mark.parametrize(
    "table_index,field",
    [
        (3, "sum"),
        (4, "retained_divergence_norm_sq_sum"),
        (4, "retained_flow_norm_sq_sum"),
        (5, "energy_overlap_corrected_sum"),
    ],
)
def test_decomposition_and_assembly_failures_are_rejected(table_index, field):
    rows = dense_rows()
    rows[table_index][0][field] += 1.0
    with pytest.raises(ValueError, match="identity|mean"):
        validate_rows(*rows)


def test_debug_cannot_claim_actual_citation_or_full_training():
    rows = dense_rows()
    rows[1][-1]["actual_data"] = True
    with pytest.raises(ValueError, match="DEBUG fixtures"):
        validate_rows(*rows)
    rows = dense_rows()
    rows[0]["profile"] = "full"
    with pytest.raises(ValueError, match="201"):
        validate_rows(*rows)


def test_report_writes_all_macro_rows_and_real_scientific_exports_without_raw_overwrite(tmp_path):
    rows = _state_strings(dense_rows())
    raw = tmp_path / "local_states.csv"
    raw.write_text("raw sentinel\n", encoding="utf-8")
    result = write_report(
        tmp_path,
        *rows,
        [
            {"type": "calibration", "peak_bytes": np.int64(123)},
            {"type": "execution", "seconds": np.float64(0.5)},
        ],
        {"debug": True, "optimizer_updates": 0},
    )
    assert result["graph_count"] == 21 and result["actual_citation_graphs"] == 0
    assert raw.read_text(encoding="utf-8") == "raw sentinel\n"
    for filename in ARTIFACTS:
        assert (tmp_path / filename).is_file()
    assert (tmp_path / "figures/energy_vs_flow.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "figures/energy_vs_flow.pdf").read_bytes().startswith(b"%PDF")
    with (tmp_path / "relation_summary.csv").open(encoding="utf-8", newline="") as stream:
        measured = list(csv.DictReader(stream))
    assert len(measured) == len(rows[3])
    assert any(float(row["min"]) < 0 for row in measured)
    summary = (tmp_path / "LOCAL_ENERGY_RELATIONS_SUMMARY.md").read_text(encoding="utf-8")
    assert "DEBUG fixture" in summary and "분류 성능 결과가 아니다" in summary
    assert "복사한 d를 H_next에 사용하지 않았으며" in summary
    resources = json.loads((tmp_path / "resources.json").read_text(encoding="utf-8"))
    assert resources[0]["peak_bytes"] == 123 and isinstance(resources[0]["peak_bytes"], int)
    assert resources[1]["seconds"] == 0.5
    assert "partial" not in summary
    before = {name: (tmp_path / name).read_bytes() for name in ARTIFACTS}
    with pytest.raises(FileExistsError, match="overwrite"):
        write_report(tmp_path, *rows, [], {})
    assert before == {name: (tmp_path / name).read_bytes() for name in ARTIFACTS}


def test_invalid_coverage_rejected_before_any_report_artifact(tmp_path):
    rows = dense_rows()
    rows[2].pop()
    with pytest.raises(ValueError, match="coverage"):
        write_report(tmp_path, *rows, [], {})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "completion",
    [
        {"optimizer_updates": 1},
        {"trainable_parameters": 1},
        {"classifier_training_run": True},
    ],
)
def test_fixed_audit_cannot_claim_trained_model(tmp_path, completion):
    with pytest.raises(ValueError, match="fixed audit"):
        write_report(tmp_path, *dense_rows(), [], completion)
    assert not list(tmp_path.iterdir())


def test_nonfinite_numpy_resource_does_not_silently_serialize(tmp_path):
    with pytest.raises(ValueError, match="JSON compliant"):
        write_report(tmp_path, *dense_rows(), [{"seconds": np.float64(float("nan"))}], {})
    assert not list(tmp_path.iterdir())
