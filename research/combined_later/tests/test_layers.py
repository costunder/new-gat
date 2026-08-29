"""Algebraic invariance tests for the reference PyTorch layers."""

from __future__ import annotations

import torch

from research.combined_later.layers import (
    PersistentTransportBlock,
    PositiveInvariantConductance,
    hard_observation_coordinate_projector,
    incidence_node_message,
    physical_cycle_increment,
)

DTYPE = torch.float64


def _incidence(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Oriented incidence with -1 at the tail and +1 at the head."""

    matrix = torch.zeros((len(edges), num_nodes), dtype=DTYPE)
    for edge_index, (tail, head) in enumerate(edges):
        matrix[edge_index, tail] = -1.0
        matrix[edge_index, head] = 1.0
    return matrix


def _fundamental_basis(
    incidence: torch.Tensor, tree_edges: list[int]
) -> tuple[torch.Tensor, list[int]]:
    """Construct an integer fundamental-cycle basis for a given tree."""

    num_edges, num_nodes = incidence.shape
    assert len(tree_edges) == num_nodes - 1
    chords = [edge for edge in range(num_edges) if edge not in tree_edges]
    basis = incidence.new_zeros((num_edges, len(chords)))

    # Delete one redundant conservation equation.  A reduced incidence matrix
    # of a spanning tree is square and nonsingular.
    reduced_tree = incidence[tree_edges].transpose(0, 1)[:-1]
    for column, chord in enumerate(chords):
        rhs = -incidence[chord][:-1]
        tree_coefficients = torch.linalg.solve(reduced_tree, rhs)
        basis[tree_edges, column] = tree_coefficients.round()
        basis[chord, column] = 1.0

    torch.testing.assert_close(
        incidence.transpose(0, 1) @ basis,
        torch.zeros((num_nodes, len(chords)), dtype=DTYPE),
        atol=0.0,
        rtol=0.0,
    )
    return basis, chords


def _two_tree_charts() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
    ]
    incidence = _incidence(5, edges)
    basis_1, _ = _fundamental_basis(incidence, [0, 1, 2, 3])
    basis_2, chords_2 = _fundamental_basis(incidence, [4, 5, 6, 2])

    # Chord rows of a fundamental basis form the identity.  Reading F_1 on
    # the second chart's chords therefore gives a_2 = M a_1 and F_2 M = F_1.
    transition = basis_1[chords_2]
    torch.testing.assert_close(basis_2 @ transition, basis_1, atol=0.0, rtol=0.0)
    assert abs(torch.linalg.det(transition).item()) == 1.0
    # Make sure this test covers more than a signed permutation transition.
    assert torch.count_nonzero(transition).item() > transition.shape[0]
    return incidence, basis_1, basis_2, transition


def test_positive_conductance_and_orientation_gauge_invariance() -> None:
    torch.manual_seed(7)
    incidence, basis, _, _ = _two_tree_charts()
    channels = 3
    potential = torch.randn((incidence.shape[1], channels), dtype=DTYPE)
    coordinates = torch.randn((basis.shape[1], channels), dtype=DTYPE)
    edge_features = torch.randn((incidence.shape[0], 2), dtype=DTYPE)
    gradient = incidence @ potential
    circulation = basis @ coordinates

    attention = PositiveInvariantConductance(
        channels,
        hidden_channels=11,
        edge_feature_channels=2,
        minimum=2.5e-3,
    ).to(dtype=DTYPE)
    conductance = attention(gradient, circulation, edge_features)

    signs = torch.tensor([-1, 1, -1, -1, 1, 1, -1], dtype=DTYPE)
    flipped_incidence = signs[:, None] * incidence
    flipped_basis = signs[:, None] * basis
    flipped_gradient = flipped_incidence @ potential
    flipped_circulation = flipped_basis @ coordinates
    flipped_conductance = attention(flipped_gradient, flipped_circulation, edge_features)

    assert torch.all(conductance > 2.5e-3)
    torch.testing.assert_close(flipped_conductance, conductance, atol=1.0e-12, rtol=1.0e-12)

    node_message = incidence_node_message(incidence, potential, conductance)
    flipped_node_message = incidence_node_message(flipped_incidence, potential, flipped_conductance)
    torch.testing.assert_close(flipped_node_message, node_message, atol=1.0e-12, rtol=1.0e-12)


def test_physical_cycle_increment_is_chart_covariant_and_hard_preserving() -> None:
    torch.manual_seed(11)
    _, basis_1, basis_2, transition = _two_tree_charts()
    residual = torch.randn((basis_1.shape[0], 2), dtype=DTYPE)
    observed_edges = torch.tensor([0], dtype=torch.long)

    increment_1 = physical_cycle_increment(basis_1, residual, observation=observed_edges)
    increment_2 = physical_cycle_increment(basis_2, residual, observation=observed_edges)
    torch.testing.assert_close(
        increment_2,
        transition @ increment_1,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    torch.testing.assert_close(
        basis_1[observed_edges] @ increment_1,
        torch.zeros((1, 2), dtype=DTYPE),
        atol=2.0e-11,
        rtol=0.0,
    )

    projector_1 = hard_observation_coordinate_projector(basis_1, observed_edges)
    projector_2 = hard_observation_coordinate_projector(basis_2, observed_edges)
    transition_inverse = torch.linalg.inv(transition)
    torch.testing.assert_close(
        projector_2,
        transition @ projector_1 @ transition_inverse,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    torch.testing.assert_close(
        projector_1 @ projector_1,
        projector_1,
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_nonlinear_multilayer_rollout_is_tree_chart_equivariant() -> None:
    torch.manual_seed(19)
    incidence, basis_1, basis_2, transition = _two_tree_charts()
    channels = 2
    potential_1 = torch.randn((incidence.shape[1], channels), dtype=DTYPE)
    potential_2 = potential_1.clone()
    coordinates_1 = torch.randn((basis_1.shape[1], channels), dtype=DTYPE)
    coordinates_2 = transition @ coordinates_1
    edge_features = torch.randn((incidence.shape[0], 3), dtype=DTYPE)
    observed_edges = torch.tensor([0], dtype=torch.long)

    blocks = torch.nn.ModuleList(
        [
            PersistentTransportBlock(
                channels,
                hidden_channels=13,
                edge_feature_channels=3,
                potential_step=0.07,
                cycle_step=0.09,
            ).to(dtype=DTYPE)
            for _ in range(4)
        ]
    )

    for block in blocks:
        potential_1, coordinates_1, diagnostics_1 = block(
            incidence,
            basis_1,
            potential_1,
            coordinates_1,
            edge_features,
            observation=observed_edges,
            return_diagnostics=True,
        )
        potential_2, coordinates_2, diagnostics_2 = block(
            incidence,
            basis_2,
            potential_2,
            coordinates_2,
            edge_features,
            observation=observed_edges,
            return_diagnostics=True,
        )

        torch.testing.assert_close(potential_2, potential_1, atol=3.0e-10, rtol=3.0e-10)
        torch.testing.assert_close(
            coordinates_2,
            transition @ coordinates_1,
            atol=3.0e-10,
            rtol=3.0e-10,
        )
        torch.testing.assert_close(
            basis_2 @ coordinates_2,
            basis_1 @ coordinates_1,
            atol=3.0e-10,
            rtol=3.0e-10,
        )
        torch.testing.assert_close(
            diagnostics_2["conductance"],
            diagnostics_1["conductance"],
            atol=3.0e-10,
            rtol=3.0e-10,
        )
        torch.testing.assert_close(
            basis_1[observed_edges] @ diagnostics_1["cycle_increment"],
            torch.zeros((1, channels), dtype=DTYPE),
            atol=3.0e-10,
            rtol=0.0,
        )


def test_full_block_is_orientation_invariant_and_centers_potential() -> None:
    torch.manual_seed(23)
    incidence, basis, _, _ = _two_tree_charts()
    channels = 2
    potential = 4.0 + torch.randn((incidence.shape[1], channels), dtype=DTYPE)
    coordinates = torch.randn((basis.shape[1], channels), dtype=DTYPE)
    block = PersistentTransportBlock(
        channels, hidden_channels=9, potential_step=0.05, cycle_step=0.08
    ).to(dtype=DTYPE)

    signs = torch.tensor([1, -1, -1, 1, -1, 1, -1], dtype=DTYPE)
    next_p, next_a, diagnostics = block(
        incidence,
        basis,
        potential,
        coordinates,
        return_diagnostics=True,
    )
    flipped_p, flipped_a, flipped_diagnostics = block(
        signs[:, None] * incidence,
        signs[:, None] * basis,
        potential,
        coordinates,
        return_diagnostics=True,
    )

    torch.testing.assert_close(next_p, flipped_p, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(next_a, flipped_a, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        diagnostics["conductance"],
        flipped_diagnostics["conductance"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        diagnostics["node_message"],
        flipped_diagnostics["node_message"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        next_p.mean(dim=0),
        torch.zeros(channels, dtype=DTYPE),
        atol=1.0e-12,
        rtol=0.0,
    )
