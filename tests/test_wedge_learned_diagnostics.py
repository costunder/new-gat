"""Debug-only dense references for the teacher-operator audit; no training."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.wedge_propagation.learned.diagnostics import operator_audit
from research.wedge_propagation.learned.model import teacher_weights
from research.wedge_propagation.operators import build_wedges


def debug_case(graph_id, pairs, n, realizations=4, constant=False):
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()
    wedges = build_wedges(edges, n)
    x = torch.randn(n, realizations, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(112 + n))
    p = wedges.shape[1]
    if constant:
        c = torch.ones(p, realizations, dtype=torch.float64)
    else:
        g1 = x[wedges[1]] - x[wedges[0]]
        g2 = x[wedges[2]] - x[wedges[1]]
        c = teacher_weights(g1, g2, torch.zeros(p, dtype=torch.long), 1)
    # Independently construct dense rows for debug labels/reference matrices.
    b = torch.zeros(edges.shape[1], n, dtype=torch.float64)
    for row, (u, v) in enumerate(edges.t().tolist()):
        b[row, u], b[row, v] = -1, 1
    a = torch.zeros(p, n, dtype=torch.float64)
    for row, (i, j, k) in enumerate(wedges.t().tolist()):
        a[row, i], a[row, j], a[row, k] = 1, -2, 1
    laplacian, fixed = b.t() @ b, a.t() @ a
    return SimpleNamespace(
        graph_id=graph_id, family="debug", split="debug", num_nodes=n,
        edges=edges, wedges=wedges, features=x, teacher_c=c,
        pair_edges=torch.empty(2, 0, dtype=torch.long),
        pair_coefficients=torch.empty(2, 0, dtype=torch.float64),
        lx=laplacian @ x, l2x=laplacian @ laplacian @ x, qx=fixed @ x,
        target_path=a.t() @ (c * (a @ x)),
        debug_b=b, debug_a=a,
    )


def dense_expected(case):
    b, a, c = case.debug_b.numpy(), case.debug_a.numpy(), case.teacher_c.numpy()
    laplacian, fixed = b.T @ b, a.T @ a
    basis = np.stack((laplacian.ravel(), (laplacian @ laplacian).ravel(), fixed.ravel()), axis=1)
    teacher = np.stack([a.T @ (c[:, r, None] * a) for r in range(c.shape[1])], axis=-1)
    teacher = teacher.reshape(case.num_nodes**2, -1)
    norms = np.linalg.norm(teacher, axis=0)
    coefficients = np.linalg.lstsq(basis, teacher, rcond=1e-12)[0]
    residual_norms = np.linalg.norm(teacher - basis @ coefficients, axis=0)
    defined = norms > 0
    relative = float(np.mean(residual_norms[defined] / norms[defined])) if defined.any() else None
    variation = None
    if c.shape[1] > 1 and norms[0] > 0:
        variation = float(np.mean(np.linalg.norm(teacher[:, 1:] - teacher[:, :1], axis=0)
                                  / (norms[0] + 1e-8)))
    return {
        "teacher_span_L_L2_Q_residual": relative,
        "teacher_span_absolute_residual_mean": float(residual_norms.mean()),
        "teacher_span_defined_features": int(defined.sum()),
        "teacher_span_undefined_features": int((~defined).sum()),
        "teacher_operator_variation": variation,
        "teacher_operator_reference_norm": float(norms[0]),
        "teacher_operator_variation_defined_pairs": c.shape[1] - 1 if variation is not None else 0,
        "teacher_c_std": float(c.std(axis=0).mean()) if len(c) else None,
    }


def assert_metrics(actual, expected, tolerance=1e-11):
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None
        else:
            assert actual[key] == pytest.approx(value, rel=tolerance, abs=tolerance)


@pytest.mark.parametrize("pairs,n", [
    ([(0, 1), (1, 2), (0, 2)], 3),
    ([(0, 1), (1, 2), (1, 3), (3, 4)], 5),
])
def test_debug_teacher_operator_scatter_matches_independent_dense(pairs, n):
    case = debug_case("dense", pairs, n)
    row = operator_audit([case], "cpu", 2)[0]
    assert_metrics(row, dense_expected(case))
    assert row["teacher_operator_variation"] > 0.01
    assert row["teacher_span_L_L2_Q_residual"] > 0.01
    torch.testing.assert_close(case.teacher_c.mean(0), torch.ones(4).double())


@pytest.mark.parametrize("pairs,n", [
    ([(0, 1), (1, 2), (0, 2)], 3),
    ([(0, 1), (1, 2), (1, 3), (3, 4)], 5),
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5),
    ([(0, 1), (0, 2), (0, 3), (0, 4)], 5),
])
def test_debug_constant_weights_have_zero_span_residual_and_variation(pairs, n):
    case = debug_case("constant", pairs, n, constant=True)
    row = operator_audit([case], "cpu", 2)[0]
    assert_metrics(row, dense_expected(case))
    assert row["teacher_span_L_L2_Q_residual"] < 1e-12
    assert row["teacher_operator_variation"] == 0.0
    assert row["teacher_c_std"] == 0.0


def test_debug_size_bucketing_preserves_all_cases_and_input_order():
    cases = [
        debug_case("cycle5", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5),
        debug_case("triangle3", [(0, 1), (1, 2), (0, 2)], 3),
        debug_case("tree5", [(0, 1), (1, 2), (1, 3), (3, 4)], 5),
        debug_case("triangle_r1", [(0, 1), (1, 2), (0, 2)], 3, realizations=1),
    ]
    reference = operator_audit(cases, "cpu", 1)
    for batch_size in (2, 100):
        actual = operator_audit(cases, "cpu", batch_size)
        assert [row["graph_id"] for row in actual] == [case.graph_id for case in cases]
        for row, expected in zip(actual, reference, strict=True):
            assert_metrics(row, {key: value for key, value in expected.items()
                                 if key.startswith("teacher_")})


@pytest.mark.parametrize("pairs", [[], [(0, 1)]])
def test_debug_zero_wedge_has_undefined_ratios_instead_of_fake_zero(pairs):
    case = debug_case("zero", pairs, 3)
    row = operator_audit([case], "cpu", 2)[0]
    assert_metrics(row, dense_expected(case))
    assert row["teacher_span_L_L2_Q_residual"] is None
    assert row["teacher_operator_variation"] is None
    assert row["teacher_c_std"] is None
    assert row["teacher_span_defined_features"] == 0
    assert row["teacher_span_undefined_features"] == 4


def test_debug_one_realization_has_undefined_variation():
    case = debug_case("one", [(0, 1), (1, 2), (0, 2)], 3, realizations=1)
    row = operator_audit([case], "cpu", 3)[0]
    assert_metrics(row, dense_expected(case))
    assert row["teacher_operator_variation"] is None
    assert row["teacher_operator_variation_defined_pairs"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_debug_cuda_batched_float64_matches_cpu_dense():
    cases = [debug_case("triangle", [(0, 1), (1, 2), (0, 2)], 3),
             debug_case("path3", [(0, 1), (1, 2)], 3)]
    actual = operator_audit(cases, "cuda", 2)
    for row, case in zip(actual, cases, strict=True):
        assert_metrics(row, dense_expected(case), tolerance=1e-10)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_debug_invalid_batch_size_raises(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        operator_audit([], "cpu", batch_size)


def test_debug_invalid_cases_raise_clear_errors():
    with pytest.raises(ValueError, match="at least one case"):
        operator_audit([], "cpu", 2)
    case = debug_case("one", [(0, 1), (1, 2)], 3)
    with pytest.raises(ValueError, match="duplicate graph ID"):
        operator_audit([case, case], "cpu", 2)
    case.teacher_c = torch.ones(1, 3).double()
    with pytest.raises(ValueError, match="teacher_c must have shape"):
        operator_audit([case], "cpu", 2)
    case.teacher_c = torch.full((1, 4), float("nan"), dtype=torch.float64)
    with pytest.raises(ValueError, match="finite weights"):
        operator_audit([case], "cpu", 2)
