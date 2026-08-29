import numpy as np

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix
from research.combined_later.completion import (
    analytic_cycle_completion,
    cycle_observation_spectrum,
    hard_observation_affine,
    metric_minimum_anchor,
    project_hard_observation_update,
    weighted_particular_flow,
)
from research.combined_later.synthetic import structured_cycle_flows


def _graph():
    # A square with a diagonal: m=5, n=4, beta=2.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    incidence = incidence_matrix(4, edges)
    first = fundamental_cycle_basis(incidence, [0, 1, 2])
    second = fundamental_cycle_basis(incidence, [0, 2, 3])
    return incidence, first, second


def test_weighted_particular_flow_conserves_and_minimizes_energy():
    incidence, cycle_basis, _ = _graph()
    conductance = np.array([0.5, 1.2, 2.0, 0.8, 1.7])
    seed_flow = np.array([0.3, -0.9, 1.1, 0.6, -0.2])
    divergence = incidence.T @ seed_flow

    particular = weighted_particular_flow(incidence, divergence, conductance)

    np.testing.assert_allclose(incidence.T @ particular, divergence, atol=1e-10)
    inverse_conductance = np.diag(1.0 / conductance)
    base_energy = particular @ inverse_conductance @ particular
    for coefficients in (np.array([0.2, -0.5]), np.array([1.0, 0.3])):
        competitor = particular + cycle_basis @ coefficients
        competitor_energy = competitor @ inverse_conductance @ competitor
        assert competitor_energy >= base_energy - 1e-11


def test_full_rank_analytic_completion_is_exact_and_conservative():
    incidence, cycle_basis, _ = _graph()
    divergence = np.array([-1.0, 0.4, 0.2, 0.4])
    particular = weighted_particular_flow(incidence, divergence)
    true_coefficients = np.array([0.8, -1.1])
    true_flow = particular + cycle_basis @ true_coefficients
    observed = np.array([3, 4])  # chord rows of the first chart

    result = analytic_cycle_completion(
        particular,
        cycle_basis,
        observed,
        true_flow[observed],
        require_exact=True,
    )

    np.testing.assert_allclose(result.flow, true_flow, atol=1e-10)
    np.testing.assert_allclose(incidence.T @ result.flow, divergence, atol=1e-10)
    np.testing.assert_allclose(result.predicted_observations, true_flow[observed])


def test_rank_deficient_metric_anchor_is_chart_independent():
    _, first, second = _graph()
    physical_cycle = first @ np.array([0.7, -1.3])
    observed = np.array([0])
    residual = physical_cycle[observed]

    first_anchor = metric_minimum_anchor(first, observed, residual)
    second_anchor = metric_minimum_anchor(second, observed, residual)

    np.testing.assert_allclose(first @ first_anchor, second @ second_anchor, atol=1e-10)
    np.testing.assert_allclose((first @ first_anchor)[observed], residual, atol=1e-10)


def test_physical_ridge_completion_is_chart_independent():
    _, first, second = _graph()
    observed = np.array([0, 1, 4])
    values = np.array([0.3, -0.6, 0.8])
    ridge = 0.25

    first_result = analytic_cycle_completion(
        np.zeros(first.shape[0]), first, observed, values, ridge=ridge
    )
    second_result = analytic_cycle_completion(
        np.zeros(second.shape[0]), second, observed, values, ridge=ridge
    )

    np.testing.assert_allclose(first_result.flow, second_result.flow, atol=1e-10)
    np.testing.assert_allclose(
        first_result.predicted_observations,
        second_result.predicted_observations,
        atol=1e-10,
    )


def test_hard_observation_affine_preserves_every_observed_edge():
    _, cycle_basis, _ = _graph()
    particular = np.linspace(-0.4, 0.6, cycle_basis.shape[0])
    true_flow = particular + cycle_basis @ np.array([0.5, -0.2])
    observed = np.array([0])
    affine = hard_observation_affine(particular, cycle_basis, observed, true_flow[observed])
    proposed_update = np.array([3.2, -1.7])
    safe_update = project_hard_observation_update(proposed_update, affine)
    coefficients = affine.anchor + safe_update
    completed = particular + cycle_basis @ coefficients

    np.testing.assert_allclose(completed[observed], true_flow[observed], atol=1e-10)
    np.testing.assert_allclose(
        affine.observation_matrix @ affine.projector,
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(affine.projector @ affine.projector, affine.projector)


def test_observation_spectrum_is_chart_independent_and_predicts_noise_gain():
    _, first, second = _graph()
    observed = np.array([3, 4])
    first_spectrum = cycle_observation_spectrum(first, observed)
    second_spectrum = cycle_observation_spectrum(second, observed)

    np.testing.assert_allclose(
        first_spectrum.singular_values,
        second_spectrum.singular_values,
        atol=1e-10,
    )
    assert first_spectrum.rank == first.shape[1]
    assert np.isclose(
        first_spectrum.noise_amplification,
        1.0 / first_spectrum.sigma_min,
    )

    gram = first.T @ first
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    orthonormal = first @ inverse_sqrt
    design = orthonormal[observed]
    left, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    epsilon = 1e-5
    observation_noise = epsilon * left[:, -1]
    result = analytic_cycle_completion(np.zeros(first.shape[0]), first, observed, observation_noise)
    np.testing.assert_allclose(
        np.linalg.norm(result.flow),
        epsilon / singular_values[-1],
        rtol=1e-9,
    )

    deficient = cycle_observation_spectrum(first, [0])
    assert deficient.rank == 1
    assert np.isinf(deficient.noise_amplification)


def test_structured_cycle_generator_is_conservative_and_orientation_covariant():
    incidence, _, _ = _graph()
    rng = np.random.default_rng(4)
    node_features = rng.normal(size=(4, 3))
    edge_features = rng.normal(size=(5, 2))
    cycles = structured_cycle_flows(
        incidence,
        num_samples=3,
        node_features=node_features,
        edge_features=edge_features,
        scale=2.0,
    )
    np.testing.assert_allclose(cycles @ incidence, 0.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(cycles, axis=1), 2.0, atol=1e-10)

    signs = np.array([-1.0, 1.0, -1.0, 1.0, 1.0])
    reoriented = signs[:, None] * incidence
    flipped_cycles = structured_cycle_flows(
        reoriented,
        num_samples=3,
        node_features=node_features,
        edge_features=edge_features,
        scale=2.0,
    )
    np.testing.assert_allclose(flipped_cycles, cycles * signs, atol=1e-10)
