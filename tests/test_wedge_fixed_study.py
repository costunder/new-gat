"""Independent numerical and data-contract tests; never production runs."""

from pathlib import Path

import numpy as np
import pytest
import torch

from research.wedge_propagation.algebra import run_algebra
from research.wedge_propagation.data import GraphCase, make_specs
from research.wedge_propagation.operators import dense_incidence, dense_wedge
from research.wedge_propagation.study import (
    Prepared,
    available_cpus,
    collect_batch,
    compute_batch,
    prepare_case,
    save_dataset,
    upload_batch,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _prepared(family: str, n: int = 8) -> Prepared:
    spec = next(
        spec
        for spec in make_specs(profile="debug")
        if spec.family == family and spec.num_nodes == n
    )
    return prepare_case(spec)


def test_independent_dense_preparation():
    item = _prepared("tree_chord")
    b = dense_incidence(item.case.edges, item.case.num_nodes)
    a = dense_wedge(item.wedges, item.case.num_nodes)
    torch.testing.assert_close(item.lap, b.T @ b)
    torch.testing.assert_close(item.q, a.T @ a)
    torch.testing.assert_close(item.q, item.lap @ item.lap + item.correction)


def test_batched_polynomial_reductions_and_normalized_actions():
    batch = upload_batch([_prepared("cycle"), _prepared("star")], torch.device("cpu"))
    results = compute_batch(batch)
    rows, actions, spectra, values = collect_batch(batch, results)
    assert len(rows) == 2 and len(actions) == 8 and len(spectra) == 6
    for row in rows:
        assert row["polynomial_residual"] < 1e-10
        assert row["nullity_Q"] == 1
    np.testing.assert_allclose(results["coeff"].numpy(), [[0, 1], [4, 1]], atol=1e-10)
    cycle_id = batch.prepared[0].case.graph_id
    np.testing.assert_allclose(values[f"{cycle_id}__L2X"], values[f"{cycle_id}__QX"])
    for row in actions[:4]:
        assert row["normalized_action_Q_L2_relative_difference"] < 1e-12
        assert row["normalized_Q_energy"] == pytest.approx(row["normalized_L2_energy"])


def test_polynomial_residual_matches_independent_least_squares():
    item = _prepared("tree_chord")
    results = compute_batch(upload_batch([item], torch.device("cpu")))
    lap = item.lap.numpy()
    q = item.q.numpy()
    basis = np.stack([lap.reshape(-1), (lap @ lap).reshape(-1)], axis=1)
    coeff, *_ = np.linalg.lstsq(basis, q.reshape(-1), rcond=None)
    expected = np.linalg.norm(q.reshape(-1) - basis @ coeff) / np.linalg.norm(q)
    np.testing.assert_allclose(results["poly"].numpy(), [expected], atol=1e-12)
    np.testing.assert_allclose(results["coeff"].numpy()[0], coeff, atol=1e-10)
    x = item.case.features.numpy()
    expected_q_energy = np.sum(x * (q @ x), axis=0)
    np.testing.assert_allclose(results["energy"].numpy()[0, :, 2], expected_q_energy)


def test_zero_operator_relative_metrics_remain_undefined():
    case = GraphCase(
        "empty",
        "er",
        5,
        torch.empty((2, 0), dtype=torch.long),
        torch.ones((5, 4), dtype=torch.float64),
        1,
        2,
    )
    zero = torch.zeros((5, 5), dtype=torch.float64)
    item = Prepared(case, torch.empty((3, 0), dtype=torch.long), zero, zero, zero, None)
    batch = upload_batch([item], torch.device("cpu"))
    rows, actions, spectra, values = collect_batch(batch, compute_batch(batch))
    assert rows[0]["polynomial_residual"] is None
    assert rows[0]["polynomial_absolute_residual"] == 0
    assert rows[0]["num_components"] == rows[0]["nullity_Q"] == 5
    assert rows[0]["degree_sum_std"] is None
    assert actions[0]["normalized_Q_norm"] is None
    assert actions[0]["normalized_Q_energy"] is None
    assert actions[0]["normalized_action_Q_L2_relative_difference"] is None
    assert len(spectra) == 3 and len(values) == 3


def test_algebra_fixtures():
    audit = run_algebra()
    assert len(audit) == 7
    assert all(row["status"] == "passed" for row in audit)


def test_saved_dataset_is_complete_and_protected(tmp_path: Path):
    item = _prepared("grid")
    target = tmp_path / "dataset.npz"
    manifest = save_dataset(target, [item])
    graph_id = item.case.graph_id
    with np.load(target, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored[f"{graph_id}__edges"], item.case.edges.numpy())
        np.testing.assert_array_equal(stored[f"{graph_id}__features"], item.case.features.numpy())
        np.testing.assert_array_equal(stored[f"{graph_id}__wedges"], item.wedges.numpy())
        assert stored[f"{graph_id}__edge_degree_sum"].shape == (item.case.edges.shape[1],)
    assert manifest[0]["num_features"] == 4
    with pytest.raises(FileExistsError):
        save_dataset(target, [item])


def test_bad_operator_cannot_produce_success_rows():
    batch = upload_batch([_prepared("cycle")], torch.device("cpu"))
    batch.q[0, 0, 0] += 1
    with pytest.raises(ArithmeticError, match="disagree"):
        collect_batch(batch, compute_batch(batch))


def test_nullspace_input_uses_absolute_roundoff_tolerance():
    item = _prepared("star")
    item.case.features.fill_(0.3)
    batch = upload_batch([item], torch.device("cpu"))
    rows, actions, _, _ = collect_batch(batch, compute_batch(batch))
    assert rows[0]["dense_explicit_absolute_error"] < 1e-10
    assert actions[0]["normalized_action_Q_L2_relative_difference"] is None


def test_batch_and_allocation_validation():
    with pytest.raises(ValueError, match="empty"):
        upload_batch([], torch.device("cpu"))
    with pytest.raises(ValueError, match="common node count"):
        upload_batch([_prepared("cycle", 5), _prepared("cycle", 8)], torch.device("cpu"))
    assert available_cpus({"cpu_affinity_count": 16, "cpu_quota": 3.5}) == 3
    assert available_cpus({"cpu_affinity_count": 8, "cpu_quota": None}) == 8
