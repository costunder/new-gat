from __future__ import annotations

import torch

from research.conductance_gat.model import (
    IncidenceConductanceAttention,
    IsotropicConductanceAttention,
    PositiveInvariantScalarConductance,
    frozen_operator_spectral_radius,
)
from research.conductance_gat.paper import resolve_device, runtime_metadata
from research.conductance_gat.tests.dense_model_inputs import (
    evaluate_model,
    make_conductance_dataset,
)


def _small_dataset(dtype: torch.dtype = torch.float64):
    return make_conductance_dataset(
        num_nodes=9,
        extra_edges=6,
        num_excitations=20,
        channels=2,
        edge_feature_channels=3,
        requested_step=0.025,
        seed=12,
        dtype=dtype,
    )


def test_device_resolution_is_host_aware(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_runtime_metadata_records_resolved_cpu_host() -> None:
    metadata = runtime_metadata(torch.device("cpu"), amp=False, pin_memory=False, batch_size=2)
    assert metadata["device"] == "cpu"
    assert metadata["device_name"] == "cpu"
    assert metadata["python"]
    assert metadata["torch"]


def test_dataset_obeys_diagonal_conductance_equations() -> None:
    dataset = _small_dataset()
    expected_gradient = torch.einsum("mn,bnd->bmd", dataset.incidence, dataset.potentials)
    expected_flux = dataset.true_conductance[None, :, None] * expected_gradient
    expected_message = torch.einsum("mn,bmd->bnd", dataset.incidence, expected_flux)

    assert dataset.num_excitations == 20
    assert torch.all(dataset.true_conductance > 0.0)
    assert torch.allclose(dataset.true_gradient, expected_gradient)
    assert torch.allclose(dataset.true_flux, expected_flux)
    assert torch.allclose(dataset.true_node_message, expected_message)
    assert dataset.excited_edges().all()


def test_scalar_conductance_is_positive_and_orientation_invariant() -> None:
    torch.manual_seed(3)
    estimator = PositiveInvariantScalarConductance(2, 3, hidden_channels=8).double()
    gradient = torch.randn(4, 13, 2, dtype=torch.float64)
    edge_features = torch.randn(13, 3, dtype=torch.float64)
    signs = torch.where(torch.arange(13) % 2 == 0, 1.0, -1.0).double()

    first = estimator(gradient, edge_features)
    second = estimator(signs[None, :, None] * gradient, edge_features)
    assert torch.all(first > 0.0)
    assert first.shape == (4, 13, 1)
    assert torch.allclose(first, second, atol=1.0e-12, rtol=1.0e-12)


def test_full_layer_is_orientation_invariant_and_flux_is_equivariant() -> None:
    torch.manual_seed(4)
    dataset = _small_dataset()
    model = IncidenceConductanceAttention(
        channels=2,
        edge_feature_channels=3,
        hidden_channels=12,
        step_size=dataset.step_size,
    ).double()
    signs = torch.where(torch.arange(dataset.num_edges) % 3 == 0, -1.0, 1.0).double()
    flipped_incidence = signs[:, None] * dataset.incidence

    first, first_diagnostics = model(
        dataset.incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    second, second_diagnostics = model(
        flipped_incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    assert torch.allclose(first, second, atol=1.0e-11, rtol=1.0e-11)
    assert torch.allclose(
        first_diagnostics["conductance"],
        second_diagnostics["conductance"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.allclose(
        second_diagnostics["edge_flux"],
        signs[None, :, None] * first_diagnostics["edge_flux"],
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_residual_preserves_node_mean_and_is_one_step_stable() -> None:
    torch.manual_seed(5)
    dataset = _small_dataset()
    model = IncidenceConductanceAttention(
        channels=2,
        edge_feature_channels=3,
        hidden_channels=10,
        step_size=10.0,
        adaptive_stability=True,
    ).double()
    output, diagnostics = model(
        dataset.incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    assert torch.allclose(
        output.mean(dim=1), dataset.potentials.mean(dim=1), atol=1.0e-12, rtol=1.0e-12
    )
    input_norm = torch.linalg.vector_norm(dataset.potentials, dim=(1, 2))
    output_norm = torch.linalg.vector_norm(output, dim=(1, 2))
    assert torch.all(output_norm <= input_norm + 1.0e-10)

    radius = frozen_operator_spectral_radius(
        dataset.incidence,
        diagnostics["conductance"][0],
        diagnostics["effective_step"][0],
    )
    assert radius <= 1.0 + 1.0e-10


def test_learned_model_has_gradient_and_isotropic_baseline_is_scalar() -> None:
    torch.manual_seed(7)
    dataset = _small_dataset(dtype=torch.float32)
    model = IncidenceConductanceAttention(
        channels=2,
        edge_feature_channels=3,
        hidden_channels=10,
        step_size=dataset.step_size,
    )
    prediction, diagnostics = model(
        dataset.incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    loss = (prediction - dataset.true_next_state).square().mean()
    loss = loss + (diagnostics["edge_flux"] - dataset.true_flux).square().mean()
    loss.backward()
    gradient_norm = sum(
        parameter.grad.abs().sum() for parameter in model.parameters() if parameter.grad is not None
    )
    assert gradient_norm > 0.0

    baseline = IsotropicConductanceAttention(
        channels=2,
        edge_feature_channels=3,
        step_size=dataset.step_size,
    )
    _, diagnostics = baseline(
        dataset.incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    predicted = diagnostics["conductance"]
    assert torch.allclose(predicted, predicted[:, :1].expand_as(predicted))
    metrics = evaluate_model(baseline, dataset)
    assert metrics["conductance_correlation_excited"] is None
    assert metrics["conductance_correlation_defined"] is False
