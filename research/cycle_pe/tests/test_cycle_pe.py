from __future__ import annotations

import numpy as np

from chartgat.algebra import incidence_matrix
from research.cycle_pe.features import (
    SET_STAT_NAMES,
    cycle_projector,
    cycle_set_statistics,
    projector_leverage_pe,
    raw_padded_basis_pe,
    static_cycle_feature_bundle,
    static_fundamental_basis,
)


def _two_cycle_graph() -> tuple[np.ndarray, np.ndarray]:
    # First three rows form the spanning path 0--1--2--3.  The two chords
    # produce beta=2 and make sign/permutation tests nontrivial.
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]
    incidence = incidence_matrix(4, edges)
    tree = np.asarray([0, 1, 2], dtype=np.int64)
    return incidence, tree


def test_static_bundle_is_deterministic_and_has_expected_shapes() -> None:
    incidence, tree = _two_cycle_graph()
    first = static_cycle_feature_bundle(incidence, tree, max_cycles=5)
    second = static_cycle_feature_bundle(incidence, tree, max_cycles=5)

    assert first["basis"].shape == (5, 2)
    assert first["raw_padded"].shape == (5, 5)
    assert first["cycle_set"].shape == (5, len(SET_STAT_NAMES))
    assert first["projector_leverage"].shape == (5, 1)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])

    # A caller mutating its returned tensor cannot change a later extraction;
    # there is no persistent/sample-dependent state inside the PE constructor.
    first["raw_padded"][0, 0] = 123.0
    third = static_cycle_feature_bundle(incidence, tree, max_cycles=5)
    assert third["raw_padded"][0, 0] != 123.0


def test_raw_padding_preserves_basis_but_makes_no_invariance_claim() -> None:
    incidence, tree = _two_cycle_graph()
    basis = static_fundamental_basis(incidence, tree)
    padded = raw_padded_basis_pe(basis, max_cycles=4)
    np.testing.assert_array_equal(padded[:, :2], basis)
    np.testing.assert_array_equal(padded[:, 2:], 0.0)

    permuted_signed = basis[:, [1, 0]] * np.asarray([-1.0, 1.0])
    changed = raw_padded_basis_pe(permuted_signed, max_cycles=4)
    assert not np.array_equal(changed, padded)


def test_cycle_set_statistics_are_column_sign_and_permutation_invariant() -> None:
    incidence, tree = _two_cycle_graph()
    basis = static_fundamental_basis(incidence, tree)
    transformed = basis[:, [1, 0]] * np.asarray([-1.0, 1.0])
    np.testing.assert_allclose(
        cycle_set_statistics(transformed),
        cycle_set_statistics(basis),
        atol=1e-12,
    )


def test_projector_and_leverage_are_invariant_to_any_basis_change() -> None:
    incidence, tree = _two_cycle_graph()
    basis = static_fundamental_basis(incidence, tree)
    coordinate_change = np.asarray([[1.0, 2.0], [-1.0, 1.0]])
    changed_basis = basis @ coordinate_change

    np.testing.assert_allclose(cycle_projector(changed_basis), cycle_projector(basis), atol=1e-12)
    np.testing.assert_allclose(
        projector_leverage_pe(changed_basis), projector_leverage_pe(basis), atol=1e-12
    )

    projector = cycle_projector(basis)
    np.testing.assert_allclose(projector, projector.T, atol=1e-12)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)


def test_structural_scalar_features_ignore_incidence_orientation() -> None:
    incidence, tree = _two_cycle_graph()
    basis = static_fundamental_basis(incidence, tree)
    signs = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])
    flipped_basis = signs[:, None] * basis
    np.testing.assert_allclose(
        cycle_set_statistics(flipped_basis), cycle_set_statistics(basis), atol=1e-12
    )
    np.testing.assert_allclose(
        projector_leverage_pe(flipped_basis), projector_leverage_pe(basis), atol=1e-12
    )


def test_tree_graph_has_zero_width_cycle_space_and_zero_static_pe() -> None:
    edges = [(0, 1), (1, 2), (2, 3)]
    incidence = incidence_matrix(4, edges)
    basis = static_fundamental_basis(incidence, np.asarray([0, 1, 2]))
    assert basis.shape == (3, 0)
    np.testing.assert_array_equal(raw_padded_basis_pe(basis, 2), np.zeros((3, 2)))
    np.testing.assert_array_equal(cycle_set_statistics(basis), np.zeros((3, len(SET_STAT_NAMES))))
    np.testing.assert_array_equal(projector_leverage_pe(basis), np.zeros((3, 1)))
