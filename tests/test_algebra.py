import numpy as np
import pytest

from chartgat.algebra import (
    chart_transition,
    decode_edge_state,
    encode_edge_state,
    flip_cycle_basis,
    flip_edge_quantity,
    flip_incidence,
    fundamental_cycle_basis,
    incidence_matrix,
    orthonormal_cycle_basis,
    validate_spanning_tree,
)

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
    (0, 2),
    (1, 3),
]
TREES = (
    [0, 1, 2, 3],
    [4, 0, 1, 2],
    [5, 1, 2, 3],
)


@pytest.fixture
def B():
    return incidence_matrix(5, EDGES)


def test_incidence_convention_and_tree_validation(B):
    np.testing.assert_array_equal(B[0], [-1, 1, 0, 0, 0])
    np.testing.assert_array_equal(validate_spanning_tree(B, TREES[0]), TREES[0])
    with pytest.raises(ValueError, match="do not form"):
        validate_spanning_tree(B, [0, 1, 5, 4])
    with pytest.raises(ValueError, match="unique"):
        validate_spanning_tree(B, [0, 0, 1, 2])


def test_fundamental_basis_is_cycle_space_with_chord_identity(B):
    F, chords = fundamental_cycle_basis(B, TREES[0], return_chords=True)
    beta = B.shape[0] - B.shape[1] + 1

    assert F.shape == (B.shape[0], beta)
    np.testing.assert_allclose(B.T @ F, 0.0, atol=1e-12)
    np.testing.assert_array_equal(F[chords], np.eye(beta))
    assert np.linalg.matrix_rank(F) == beta


def test_gradient_cycle_encoding_is_lossless(B):
    rng = np.random.default_rng(2026)
    F = fundamental_cycle_basis(B, TREES[0])
    edge_state = rng.normal(size=(B.shape[0], 4))

    p, a = encode_edge_state(B, F, edge_state)
    reconstructed = decode_edge_state(B, F, p, a)

    np.testing.assert_allclose(reconstructed, edge_state, atol=1e-12)
    np.testing.assert_allclose(p.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(B.T @ (F @ a), 0.0, atol=1e-12)


def test_orthonormal_cycle_basis_preserves_projector(B):
    F = fundamental_cycle_basis(B, TREES[0])
    U = orthonormal_cycle_basis(F)

    np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-12)
    projector_from_F = F @ np.linalg.solve(F.T @ F, F.T)
    np.testing.assert_allclose(U @ U.T, projector_from_F, atol=1e-12)
    np.testing.assert_allclose(B.T @ U, 0.0, atol=1e-12)


def test_multi_tree_transition_and_cocycle(B):
    F0, F1, F2 = [fundamental_cycle_basis(B, tree) for tree in TREES]
    M10 = chart_transition(F0, F1)
    M21 = chart_transition(F1, F2)
    M20 = chart_transition(F0, F2)

    np.testing.assert_array_equal(F1 @ M10, F0)
    np.testing.assert_array_equal(F2 @ M21, F1)
    np.testing.assert_array_equal(M21 @ M10, M20)
    assert abs(round(np.linalg.det(M10))) == 1
    assert abs(round(np.linalg.det(M21))) == 1

    a0 = np.asarray([0.25, -1.0, 2.5])
    np.testing.assert_allclose(F0 @ a0, F2 @ (M20 @ a0), atol=1e-12)


def test_orientation_flips_preserve_physical_relations(B):
    F = fundamental_cycle_basis(B, TREES[0])
    signs = np.asarray([-1, 1, -1, -1, 1, 1, -1])
    B_flipped = flip_incidence(B, signs)
    F_flipped = flip_cycle_basis(F, signs)
    p = np.arange(B.shape[1], dtype=float)
    a = np.asarray([1.0, -2.0, 0.5])

    np.testing.assert_allclose(B_flipped.T @ F_flipped, 0.0, atol=1e-12)
    np.testing.assert_allclose(B_flipped @ p, flip_edge_quantity(B @ p, signs))
    np.testing.assert_allclose(F_flipped @ a, flip_edge_quantity(F @ a, signs))
