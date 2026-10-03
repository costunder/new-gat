"""Receiver study DEBUG integration with independent complete dense references."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from research.local_energy_relations.data import AuditCase
from research.local_energy_relations.receiver_aggregation import study
from research.local_energy_relations.receiver_aggregation.contract import read_config
from research.local_energy_relations.receiver_aggregation.data import prepare_cases
from research.local_energy_relations.receiver_aggregation.operators import prepare_receiver_operator
from research.local_energy_relations.topology import build_topology


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_case(graph_id, pairs, n, features=5, vector=False):
    """Explicit DEBUG arithmetic fields, never a replacement FULL input."""
    edges = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    generator = torch.Generator().manual_seed(20261004 + n)
    x = torch.randn((n, features), generator=generator, dtype=torch.float64)
    x[:, 0] = 0  # Includes exactly zero RHS among independently active columns.
    return AuditCase(
        graph_id,
        "citation" if vector else "DEBUG-arithmetic",
        edges,
        x,
        False,
        "vector_trace" if vector else "independent_scalar_columns",
        {"debug": True},
    )


def operators(cases):
    return [
        {
            mode: prepare_receiver_operator(build_topology(case.num_nodes, case.edges), mode)
            for mode in ("unit", "local_degree")
        }
        for case in cases
    ]


def dense_reference(case, mode):
    """Direct physical/local dense matrices, no receiver-study kernel reuse."""
    n = case.num_nodes
    edges = case.edges.numpy()
    b = np.zeros((edges.shape[1], n))
    b[np.arange(edges.shape[1]), edges[0]] = -1
    b[np.arange(edges.shape[1]), edges[1]] = 1
    degree = (b != 0).sum(0)
    h = [case.x.numpy()]
    tau = 0.5 / degree.max() if degree.max() else 0.0
    for _ in range(2):
        h.append(h[-1] - tau * b.T @ b @ h[-1])
    members, ids, cs = [], [], []
    for center in range(n):
        local = set(np.flatnonzero(b[:, center] != 0))
        nodes = sorted({center} | set(edges[:, sorted(local)].ravel()))
        selected = [i for i, (a, z) in enumerate(edges.T) if a in nodes and z in nodes]
        sub_b = b[selected][:, nodes]
        deg = (sub_b != 0).sum(0)
        weights = (
            np.ones(len(selected))
            if mode == "unit"
            else np.array(
                [
                    2.0 / (deg[nodes.index(int(edges[0, i]))] + deg[nodes.index(int(edges[1, i]))])
                    for i in selected
                ]
            )
        )
        members.append(nodes)
        ids.append(selected)
        cs.append(weights)
    directed = list(edges.T) + list(edges[::-1].T)
    flows, energies, divergences, receipts = [], [], [], []
    for state in h:
        q = [c[:, None] * (b[index] @ state) for c, index in zip(cs, ids, strict=True)]
        d = [
            b[index][:, nodes].T @ value
            for index, nodes, value in zip(ids, members, q, strict=True)
        ]
        e = [
            np.sum(c[:, None] * (b[index] @ state) ** 2, axis=0)
            for c, index in zip(cs, ids, strict=True)
        ]
        tagged = np.asarray(
            [d[v][members[v].index(a)] for v, u in directed for a in members[v] if a in members[u]]
        )
        flows.append(q)
        divergences.append(d)
        energies.append(np.asarray(e))
        receipts.append(tagged)
    relations = {}
    for left, right in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2)):
        shared, nodes = [], []
        for v, u in directed:
            common_edges = set(ids[v]) & set(ids[u])
            common_nodes = set(members[v]) & set(members[u])
            shared.append(
                sum(
                    (
                        flows[left][v][ids[v].index(e)] * flows[right][u][ids[u].index(e)]
                        for e in common_edges
                    ),
                    np.zeros(case.num_features),
                )
            )
            nodes.append(
                sum(
                    (
                        divergences[left][v][members[v].index(a)]
                        * divergences[right][u][members[u].index(a)]
                        for a in common_nodes
                    ),
                    np.zeros(case.num_features),
                )
            )
        shared, nodes = np.asarray(shared), np.asarray(nodes)
        relations[left, right] = {
            "shared_edges": shared,
            "shared_nodes": nodes,
            "distinct_edges": nodes - 2 * shared,
        }
    return {"flows": flows, "energies": energies, "receipts": receipts, "relations": relations}


def collect_rows(cases, individual_ops, computed, path):
    combined = study.batch_operators(individual_ops, ("unit", "local_degree"))
    info = {row["graph_id"]: row for row in study.graph_rows(cases, individual_ops)}
    rows = {key: [] for key in ("reconstruction", "condition", "relation")}
    table = study.Table(path)
    try:
        study.collect(cases, combined, computed, info, rows, table)
    finally:
        table.close()
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    return rows, records, info


def test_actual_compute_collect_scalar_batch_matches_dense(tmp_path):
    cases = [
        make_case("DEBUG-branch", [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)], 6),
        make_case("DEBUG-components", [(0, 1), (1, 2), (3, 4)], 7),
    ]
    individual = operators(cases)
    combined = study.batch_operators(individual, ("unit", "local_degree"))
    computed = study.compute(
        combined, torch.cat([case.x for case in cases]), read_config(profile="debug"), 1
    )
    rows, records, info = collect_rows(cases, individual, computed, tmp_path / "scalar.csv")
    assert len(records) == 6 * sum(case.num_features for case in cases)
    assert len(rows["condition"]) == len(cases) * 2 * 3 * 5
    assert len(rows["relation"]) == len(cases) * 2 * 5 * 3
    for mode, fields in computed.items():
        references = [dense_reference(case, mode) for case in cases]
        for stage in range(3):
            energy = np.concatenate([ref["energies"][stage] for ref in references])
            np.testing.assert_allclose(fields["energy"][stage], energy, rtol=1e-12, atol=1e-12)
            for graph, reference in enumerate(references):
                expected_q = sum(np.square(part).sum(0) for part in reference["flows"][stage])
                np.testing.assert_allclose(
                    fields["metrics"][stage]["q_norm_sq"][graph], expected_q, rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(
                    fields["metrics"][stage]["tagged_norm_sq"][graph],
                    np.square(reference["receipts"][stage]).sum(0),
                    rtol=1e-12,
                    atol=1e-12,
                )
        for stages, relations in fields["relations"].items():
            for name, arrays in relations.items():
                expected = np.concatenate([ref["relations"][stages][name] for ref in references])
                np.testing.assert_allclose(arrays["original"], expected, rtol=1e-12, atol=1e-12)
    for row in rows["condition"]:
        graph = info[row["graph_id"]]
        within = row["condition"] in ("sum_within", "sum_both")
        between = row["condition"] in ("sum_between", "sum_both")
        coordinates = (
            graph["tagged_coordinates"]
            if row["condition"] == "tagged"
            else graph["num_local_nodes"]
        )
        coordinates += graph["num_nodes"] * within + 6 * graph["num_edges"] * between
        assert row["observed_coordinates"] == coordinates
        assert row["restricted_rank"] == graph["restricted_rank_per_channel"]
        assert row["additional_rank"] == 0
        assert row["energy_observed"] == within and row["relation_observed"] == between
        assert (row["energy_norm_sq"] is not None) == within
        assert (row["relation_norm_sq"] is not None) == between
    for row in rows["reconstruction"]:
        assert row["count"] == 5
        assert row["fused_relative_residual"] <= 1e-8
        assert row["fused_q_relative_error"] < 1e-6


def test_vector_trace_chunk_merge_collect_sums_channels_before_error_square(tmp_path):
    case = make_case(
        "DEBUG-vector", [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)], 6, features=7, vector=True
    )
    individual = operators([case])
    combined = study.batch_operators(individual, ("unit", "local_degree"))
    config = read_config(profile="debug")
    full = study.merge_trace({}, study.compute(combined, case.x, config, 3))
    chunked = {}
    for start in range(0, case.num_features, 2):
        study.merge_trace(chunked, study.compute(combined, case.x[:, start : start + 2], config, 1))
    full_rows, records, info = collect_rows([case], individual, full, tmp_path / "full.csv")
    chunk_rows, _, _ = collect_rows([case], individual, chunked, tmp_path / "chunks.csv")
    assert len(records) == 6 and {r["realization"] for r in records} == {"vector_trace"}
    for mode in config["weights"]:
        reference = dense_reference(case, mode)
        for stage in range(3):
            expected = reference["energies"][stage].sum(1, keepdims=True)
            np.testing.assert_allclose(
                full[mode]["energy"][stage], expected, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                chunked[mode]["energy"][stage], expected, rtol=1e-12, atol=1e-12
            )
            row = next(
                r
                for r in full_rows["condition"]
                if r["weight"] == mode
                and r["state"] == f"H{stage}"
                and r["condition"] == "sum_both"
            )
            assert row["energy_norm_sq"] == pytest.approx(np.square(expected).sum())
            assert row["energy_norm_sq"] != pytest.approx(
                np.square(reference["energies"][stage]).sum()
            )
            assert (
                row["observed_coordinates"]
                == info[case.graph_id]["num_local_nodes"] * 7
                + case.num_nodes
                + 6 * case.edges.shape[1]
            )
            assert row["restricted_rank"] == info[case.graph_id]["restricted_rank_per_channel"] * 7
        for stages in reference["relations"]:
            for relation in config["relations"]:
                expected = reference["relations"][stages][relation].sum(1, keepdims=True)
                np.testing.assert_allclose(
                    chunked[mode]["relations"][stages][relation]["original"],
                    expected,
                    rtol=1e-12,
                    atol=1e-12,
                )
    for group in full_rows:
        for left, right in zip(full_rows[group], chunk_rows[group], strict=True):
            for key, value in left.items():
                if isinstance(value, float):
                    np.testing.assert_allclose(value, right[key], atol=1e-10, rtol=1e-6)
                elif not key.startswith("solver_iterations"):
                    assert value == right[key]


def test_full_device_dtype_and_runtime_dimension_guards(tmp_path):
    with pytest.raises(ValueError, match="server CUDA"):
        study.validate_device(read_config(profile="full"), torch.device("cpu"))
    case = make_case("DEBUG-guard", [(0, 1), (1, 2)], 3)
    individual = operators([case])
    combined = study.batch_operators(individual, ("unit", "local_degree"))
    with pytest.raises(ValueError, match="dtype"):
        study.compute(combined, case.x.float(), read_config(profile="debug"), 1)
    config = read_config(profile="debug")
    config["runtime"]["channel_chunk"] = case.num_features + 1
    with pytest.raises(ValueError, match="exceeds complete"):
        study.calibrate([case], individual, config, torch.device("cpu"), [], "channels")
    assert study._relative(0, 0) is None
    assert study._relative(4, 16) == 0.5


def test_five_actual_observation_bundles_preserve_tensor_coordinates_and_aliases():
    receipts = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    receiver = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    energy = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    relations = {
        name: torch.full((2, 3), index, dtype=torch.float64)
        for index, name in enumerate(("shared_edges", "shared_nodes", "distinct_edges"))
    }
    bundles = study.make_observations(receipts, receiver, energy, relations)
    assert set(bundles) == set(study.config_conditions())
    assert bundles["tagged"] == {"receipts": receipts}
    assert bundles["sum"] == {"sum": receiver}
    for condition in ("sum", "sum_within", "sum_between", "sum_both"):
        assert bundles[condition]["sum"] is receiver
        assert bundles[condition]["sum"].shape == receiver.shape
    for condition in ("sum_within", "sum_both"):
        assert bundles[condition]["energy"] is energy
    for condition in ("sum_between", "sum_both"):
        assert bundles[condition]["relations"] is relations
        for name, value in relations.items():
            assert bundles[condition]["relations"][name] is value
    assert "energy" not in bundles["sum_between"]
    assert "relations" not in bundles["sum_within"]


def test_all_saved_debug_source_graphs_compute_and_collect(tmp_path):
    source = Path("results/local-energy-DEBUG-20261004-02")
    if not source.is_dir():
        pytest.skip("explicit prior DEBUG source snapshots are unavailable")
    config = read_config(profile="debug")
    cases, topologies, _ = prepare_cases(config, source, tmp_path / "source", workers=1)
    assert len(cases) == 21 and sum(case.actual_data for case in cases) == 0
    individual = [
        {mode: prepare_receiver_operator(top, mode) for mode in config["weights"]}
        for top in topologies
    ]
    table = study.Table(tmp_path / "all-debug.csv")
    rows = {key: [] for key in ("reconstruction", "condition", "relation")}
    graphs = study.graph_rows(cases, individual)
    info = {row["graph_id"]: row for row in graphs}
    try:
        for start, stop in ((0, 18), (18, 21)):
            batch_cases, batch_ops = cases[start:stop], individual[start:stop]
            combined = study.batch_operators(batch_ops, config["weights"])
            calculated = study.compute(
                combined, torch.cat([case.x for case in batch_cases]), config, 4
            )
            if start == 18:
                calculated = study.merge_trace({}, calculated)
            study.collect(batch_cases, combined, calculated, info, rows, table)
    finally:
        table.close()
    assert table.count == 6 * (72 + 3)
    assert len(rows["condition"]) == 21 * 30
    assert len(rows["relation"]) == 21 * 30
    assert len(rows["reconstruction"]) == 21 * 6
    assert {r["graph_id"] for r in rows["reconstruction"]} == {case.graph_id for case in cases}
    assert all(
        r["fused_relative_residual"] is None or r["fused_relative_residual"] <= 1e-8
        for r in rows["reconstruction"]
    )
