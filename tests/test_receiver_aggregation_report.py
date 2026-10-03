"""Independent dense DEBUG arithmetic fixtures, never final experiment results."""

import copy
import csv
import json

import numpy as np
import pytest

from research.local_energy_relations.receiver_aggregation.report import (
    ARTIFACTS,
    CONDITIONS,
    RELATIONS,
    STAGE_PAIRS,
    validate_rows,
    write_report,
)


def _ratio(error, norm):
    return float(np.sqrt(error / norm)) if norm else None


def dense_rows(*, zero=False):
    """21 explicitly DEBUG named copies measured from small dense operators."""
    n, f = 4, 2
    edges = [(0, 1), (1, 2), (2, 3)]
    nodes = [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3]]
    edge_ids = [[0], [0, 1], [1, 2], [2]]
    offsets = np.r_[0, np.cumsum(list(map(len, nodes)))]
    physical_b = np.zeros((3, n))
    for e, (a, b) in enumerate(edges):
        physical_b[e, a], physical_b[e, b] = -1, 1
    local_b = [physical_b[ids][:, members] for ids, members in zip(edge_ids, nodes, strict=True)]
    directed = [p for a, b in edges for p in ((a, b), (b, a))]
    left, right = [], []
    for v, u in directed:
        for a in sorted(set(nodes[v]) & set(nodes[u])):
            left.append(offsets[v] + nodes[v].index(a))
            right.append(offsets[u] + nodes[u].index(a))
    left, right = np.asarray(left), np.asarray(right)
    nl, k = int(offsets[-1]), len(left)
    counts = np.bincount(right, minlength=nl)
    h0 = np.zeros((n, f)) if zero else np.array([[1.0, 3.0], [-2.0, 0.0], [3.0, -1.0], [5.0, -2.0]])
    states = [h0]
    for _ in range(2):
        states.append(states[-1] - 0.25 * physical_b.T @ physical_b @ states[-1])
    graphs, reconstruction, conditions, relations = [], [], [], []
    for graph_index in range(21):
        vector = graph_index >= 18
        graph_id = f"DEBUG-dense-{graph_index}"
        metadata = {
            "graph_id": graph_id,
            "family": "path",
            "dataset": "DEBUG-arithmetic",
            "actual_data": False,
            "feature_mode": "vector_trace" if vector else "independent_scalar_columns",
        }
        graphs.append(
            {
                **metadata,
                "num_nodes": n,
                "num_edges": len(edges),
                "num_features": f,
                "num_local_nodes": nl,
                "components": 1,
                "tagged_coordinates": k,
                "fused_active_coordinates": nl,
                "ambient_tag_kernel_per_channel": k - nl,
                "restricted_rank_per_channel": n - 1,
                "sparse_nnz_unit": 28,
                "sparse_nnz_local_degree": 28,
            }
        )
        for weight in ("unit", "local_degree"):
            cv = []
            for b in local_b:
                degrees = np.abs(b).sum(0)
                cv.append(np.ones(len(b)) if weight == "unit" else 2 / (np.abs(b) @ degrees))
            dm = np.concatenate(
                [
                    b.T @ (c[:, None] * b) @ np.eye(n)[members]
                    for b, c, members in zip(local_b, cv, nodes, strict=True)
                ]
            )
            a = np.zeros((nl, n))
            np.add.at(a, right, dm[left])
            assert np.linalg.matrix_rank(a) == n - 1
            original_q, recovered_q, original_d, recovered_d, energies, energy_hats = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for state, h in enumerate(states):
                ds = dm @ h
                tagged = ds[left]
                y = np.zeros((nl, f))
                np.add.at(y, right, tagged)
                h_hat = np.linalg.pinv(a) @ y
                h_hat -= h_hat.mean(0)
                q = [
                    c[:, None] * (b @ h[members])
                    for b, c, members in zip(local_b, cv, nodes, strict=True)
                ]
                q_hat = [
                    c[:, None] * (b @ h_hat[members])
                    for b, c, members in zip(local_b, cv, nodes, strict=True)
                ]
                q_tagged = [
                    c[:, None] * (b @ np.linalg.pinv(b.T @ (c[:, None] * b)) @ dv)
                    for b, c, dv in zip(local_b, cv, np.split(ds, offsets[1:-1]), strict=True)
                ]
                e = np.stack(
                    [
                        (c[:, None] * (b @ h[members]) ** 2).sum(0)
                        for b, c, members in zip(local_b, cv, nodes, strict=True)
                    ]
                )
                e_hat = np.stack(
                    [
                        (c[:, None] * (b @ h_hat[members]) ** 2).sum(0)
                        for b, c, members in zip(local_b, cv, nodes, strict=True)
                    ]
                )
                d_hat = dm @ h_hat
                original_q.append(q)
                recovered_q.append(q_hat)
                original_d.append(np.split(ds, offsets[1:-1]))
                recovered_d.append(np.split(d_hat, offsets[1:-1]))
                energies.append(e)
                energy_hats.append(e_hat)
                q_flat, q_hat_flat, q_tagged_flat = (
                    np.concatenate(q),
                    np.concatenate(q_hat),
                    np.concatenate(q_tagged),
                )
                contrast = tagged - y[right] / counts[right, None]
                null = np.zeros_like(y)
                np.add.at(null, right, contrast)
                tagged_norm = float((tagged**2).sum())
                contrast_norm = float((contrast**2).sum())
                q_norm = float((q_flat**2).sum())
                tagged_q_error = float(((q_tagged_flat - q_flat) ** 2).sum())
                fused_q_error = float(((q_hat_flat - q_flat) ** 2).sum())
                centered = h - h.mean(0)
                h_norm, h_error = float((centered**2).sum()), float(((h_hat - centered) ** 2).sum())
                receipt_error = float(((d_hat[left] - tagged) ** 2).sum())
                y_norm, y_error = float((y**2).sum()), float(((a @ h_hat - y) ** 2).sum())
                reconstruction.append(
                    {
                        **metadata,
                        "weight": weight,
                        "state": f"H{state}",
                        "count": 1 if vector else f,
                        "tagged_norm_sq": tagged_norm,
                        "tag_contrast_norm_sq": contrast_norm,
                        "tag_contrast_fraction_sq": contrast_norm / tagged_norm
                        if tagged_norm
                        else None,
                        "null_projection_residual_sq": float((null**2).sum()),
                        "q_norm_sq": q_norm,
                        "tagged_q_error_sq": tagged_q_error,
                        "tagged_q_relative_error": _ratio(tagged_q_error, q_norm),
                        "fused_q_error_sq": fused_q_error,
                        "fused_q_relative_error": _ratio(fused_q_error, q_norm),
                        "centered_h_norm_sq": h_norm,
                        "centered_h_error_sq": h_error,
                        "centered_h_relative_error": _ratio(h_error, h_norm),
                        "tagged_reconstruction_error_sq": receipt_error,
                        "tagged_reconstruction_relative_error": _ratio(receipt_error, tagged_norm),
                        "fused_norm_sq": y_norm,
                        "fused_residual_sq": y_error,
                        "fused_relative_residual": _ratio(y_error, y_norm),
                        "solver_iterations_max": 0,
                        "solver_residual_max": _ratio(y_error, y_norm) or 0.0,
                        "solver_normal_residual_max": 0.0,
                    }
                )

            def j_values(qs, ds, stage_a, stage_b, vector=vector):
                rows = {name: [] for name in RELATIONS}
                for v, u in directed:
                    shared = sum(
                        (
                            qs[stage_a][v][edge_ids[v].index(e)]
                            * qs[stage_b][u][edge_ids[u].index(e)]
                            for e in set(edge_ids[v]) & set(edge_ids[u])
                        ),
                        start=np.zeros(f),
                    )
                    node = sum(
                        (
                            ds[stage_a][v][nodes[v].index(a)] * ds[stage_b][u][nodes[u].index(a)]
                            for a in set(nodes[v]) & set(nodes[u])
                        ),
                        start=np.zeros(f),
                    )
                    for name, value in (
                        ("shared_edges", shared),
                        ("shared_nodes", node),
                        ("distinct_edges", node - 2 * shared),
                    ):
                        rows[name].append(value.sum() if vector else value)
                return {name: np.asarray(value) for name, value in rows.items()}

            same_relations = {}
            for stage_a, stage_b in STAGE_PAIRS:
                original = j_values(original_q, original_d, stage_a, stage_b)
                recovered = j_values(recovered_q, recovered_d, stage_a, stage_b)
                if stage_a == stage_b:
                    same_relations[stage_a] = original, recovered
                for name in RELATIONS:
                    value, hat = original[name], recovered[name]
                    norm, error = float((value**2).sum()), float(((hat - value) ** 2).sum())
                    relations.append(
                        {
                            **metadata,
                            "weight": weight,
                            "stage_from": f"H{stage_a}",
                            "stage_to": f"H{stage_b}",
                            "relation": name,
                            "count": value.size,
                            "original_sum": float(value.sum()),
                            "reconstructed_sum": float(hat.sum()),
                            "norm_sq": norm,
                            "error_sq": error,
                            "relative_error": _ratio(error, norm),
                            "negative_count": int((value < 0).sum()),
                        }
                    )
            for state in range(3):
                original, recovered = same_relations[state]
                e, eh = energies[state], energy_hats[state]
                if vector:
                    e, eh = e.sum(1), eh.sum(1)
                en, ee = float((e**2).sum()), float(((eh - e) ** 2).sum())
                jn = sum(float((original[name] ** 2).sum()) for name in RELATIONS)
                je = sum(
                    float(((recovered[name] - original[name]) ** 2).sum()) for name in RELATIONS
                )
                rec = reconstruction[-3 + state]
                for condition in CONDITIONS:
                    within = condition in ("sum_within", "sum_both")
                    between = condition in ("sum_between", "sum_both")
                    dimensions = (k if condition == "tagged" else nl) * (f if vector else 1)
                    dimensions += n if within else 0
                    dimensions += 6 * len(edges) if between else 0
                    conditions.append(
                        {
                            **metadata,
                            "weight": weight,
                            "state": f"H{state}",
                            "count": 1 if vector else f,
                            "condition": condition,
                            "observed_coordinates": dimensions,
                            "restricted_rank": (n - 1) * (f if vector else 1),
                            "additional_rank": 0,
                            "q_relative_error": rec[
                                "tagged_q_relative_error"
                                if condition == "tagged"
                                else "fused_q_relative_error"
                            ],
                            "energy_observed": within,
                            "relation_observed": between,
                            "energy_norm_sq": en if within else None,
                            "energy_error_sq": ee if within else None,
                            "energy_relative_error": _ratio(ee, en) if within else None,
                            "relation_norm_sq": jn if between else None,
                            "relation_error_sq": je if between else None,
                            "relation_relative_error": _ratio(je, jn) if between else None,
                        }
                    )
    return {"profile": "debug"}, graphs, reconstruction, conditions, relations


def test_dense_measured_coverage_projection_and_recoverability():
    rows = dense_rows()
    assert len(validate_rows(*rows)) == 21
    reconstruction = rows[2]
    assert max(r["tag_contrast_fraction_sq"] for r in reconstruction) > 0
    assert max(r["fused_q_relative_error"] for r in reconstruction) < 1e-12
    assert max(r["null_projection_residual_sq"] for r in reconstruction) < 1e-25


def test_zero_denominators_remain_undefined():
    rows = dense_rows(zero=True)
    validate_rows(*rows)
    assert all(r["fused_q_relative_error"] is None for r in rows[2])
    rows[2][0]["fused_q_relative_error"] = 0.0
    with pytest.raises(ValueError, match="zero denominator"):
        validate_rows(*rows)


@pytest.mark.parametrize(
    "table,field,value,match",
    [
        (1, "restricted_rank_per_channel", 1, "restricted rank"),
        (1, "ambient_tag_kernel_per_channel", 999, "ambient receipt"),
        (2, "fused_q_relative_error", 0.9, "mismatch"),
        (2, "q_norm_sq", float("nan"), "finite"),
        (3, "additional_rank", 1, "additional actual-field"),
        (3, "observed_coordinates", 1, "observed coordinates"),
        (3, "energy_observed", True, "energy observation"),
        (4, "negative_count", 999, "negative count"),
    ],
)
def test_report_rejects_corrupted_measurement_contract(table, field, value, match):
    rows = dense_rows()
    rows[table][0][field] = value
    with pytest.raises(ValueError, match=match):
        validate_rows(*rows)


@pytest.mark.parametrize("table", [2, 3, 4])
def test_no_missing_or_duplicate_coverage(table):
    rows = dense_rows()
    rows[table].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_rows(*rows)
    rows = dense_rows()
    rows[table][-1] = copy.deepcopy(rows[table][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(*rows)


def test_full_cannot_report_debug_input_as_completed():
    rows = dense_rows()
    rows[0]["profile"] = "full"
    with pytest.raises(ValueError, match="201"):
        validate_rows(*rows)


def test_all_artifacts_include_relation_summary_and_decoder_scope(tmp_path):
    rows = dense_rows()
    result = write_report(
        tmp_path,
        *rows,
        [{"kind": "DEBUG_arithmetic", "seconds": 0.1}],
        {"trainable_parameters": 0, "optimizer_updates": 0},
    )
    assert result["graph_count"] == 21
    assert all(
        (tmp_path / path).is_file() and (tmp_path / path).stat().st_size for path in ARTIFACTS
    )
    text = (tmp_path / "RECEIVER_AGGREGATION_SUMMARY.md").read_text(encoding="utf-8")
    for phrase in (
        "decoder는 동일한 Y만",
        "송신별 상쇄",
        "연속 단계",
        "DEBUG",
        "독립 정보가 아니라",
    ):
        assert phrase in text
    for graph in rows[1][18:]:
        assert graph["graph_id"] in text
    with (tmp_path / "relation_summary.csv").open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(rows[4])
    assert json.loads((tmp_path / "resources.json").read_text())[0]["kind"] == "DEBUG_arithmetic"
    with pytest.raises(FileExistsError, match="overwrite"):
        write_report(tmp_path, *rows, [], {})


def test_no_classifier_or_nan_resource_claim(tmp_path):
    rows = dense_rows()
    with pytest.raises(ValueError, match="classifier"):
        write_report(tmp_path, *rows, [], {"classifier_training_run": True})
    with pytest.raises(ValueError, match="JSON"):
        write_report(tmp_path, *rows, [{"seconds": float("nan")}], {})
    assert not any(tmp_path.iterdir())
