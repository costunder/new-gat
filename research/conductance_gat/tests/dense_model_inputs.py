"""Test-only tensor inputs and diagnostics for dense conductance algebra.

This module is not a benchmark adapter, experiment entrypoint, or paper dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as nnf

from chartgat.algebra import incidence_matrix
from chartgat.graphs import make_connected_graph

if TYPE_CHECKING:
    from research.conductance_gat.model import IncidenceConductanceAttention


@dataclass(frozen=True)
class ConductanceDataset:
    """All tensors needed to identify diagonal conductance from excitations."""

    incidence: Tensor
    edge_features: Tensor
    potentials: Tensor
    true_conductance: Tensor
    true_gradient: Tensor
    true_flux: Tensor
    true_node_message: Tensor
    true_next_state: Tensor
    step_size: float

    @property
    def num_excitations(self) -> int:
        return int(self.potentials.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.incidence.shape[0])

    def subset(self, indices: Tensor | np.ndarray | list[int]) -> ConductanceDataset:
        index = torch.as_tensor(indices, dtype=torch.long, device=self.potentials.device)
        return replace(
            self,
            potentials=self.potentials.index_select(0, index),
            true_gradient=self.true_gradient.index_select(0, index),
            true_flux=self.true_flux.index_select(0, index),
            true_node_message=self.true_node_message.index_select(0, index),
            true_next_state=self.true_next_state.index_select(0, index),
        )

    def to(self, device: torch.device | str) -> ConductanceDataset:
        return replace(
            self,
            incidence=self.incidence.to(device),
            edge_features=self.edge_features.to(device),
            potentials=self.potentials.to(device),
            true_conductance=self.true_conductance.to(device),
            true_gradient=self.true_gradient.to(device),
            true_flux=self.true_flux.to(device),
            true_node_message=self.true_node_message.to(device),
            true_next_state=self.true_next_state.to(device),
        )

    def excited_edges(self, threshold: float = 1.0e-6) -> Tensor:
        return self.true_gradient.abs().amax(dim=(0, 2)) > threshold


def make_conductance_dataset(
    *,
    num_nodes: int = 24,
    extra_edges: int = 18,
    num_excitations: int = 128,
    channels: int = 3,
    edge_feature_channels: int = 3,
    potential_scale: float = 1.0,
    requested_step: float = 0.03,
    stability_margin: float = 0.8,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> ConductanceDataset:
    """Generate a fixed graph/conductivity with many independent potentials.

    Ground-truth conductance is a positive nonlinear function of static,
    orientation-invariant edge attributes.  Potentials are observed and are
    independently excited, so edges with a near-zero gradient in one sample
    remain identifiable from other samples.
    """

    if num_excitations < 2:
        raise ValueError("multiple-excitation data requires at least two excitations")
    if channels < 1 or edge_feature_channels < 1:
        raise ValueError("channels and edge_feature_channels must be positive")
    if potential_scale <= 0.0 or requested_step <= 0.0:
        raise ValueError("potential_scale and requested_step must be positive")
    if not 0.0 < stability_margin < 1.0:
        raise ValueError("stability_margin must lie strictly between zero and one")

    edges = make_connected_graph(num_nodes, extra_edges, seed=seed)
    incidence_np = incidence_matrix(num_nodes, edges)
    incidence = torch.as_tensor(incidence_np, dtype=dtype)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    edge_features = torch.randn(len(edges), edge_feature_channels, generator=generator, dtype=dtype)
    weights = torch.linspace(0.85, -0.35, edge_feature_channels, dtype=dtype)
    logits = edge_features @ weights
    logits = logits + 0.25 * edge_features[:, 0].square()
    if edge_feature_channels > 1:
        logits = logits + 0.20 * torch.sin(edge_features[:, 1])
    true_conductance = 0.15 + nnf.softplus(logits)

    potentials = potential_scale * torch.randn(
        num_excitations, num_nodes, channels, generator=generator, dtype=dtype
    )
    potentials = potentials - potentials.mean(dim=1, keepdim=True)
    true_gradient = torch.einsum("mn,bnd->bmd", incidence, potentials)
    true_flux = true_conductance[None, :, None] * true_gradient
    true_node_message = torch.einsum("mn,bmd->bnd", incidence, true_flux)

    weighted_degree = incidence.abs().transpose(0, 1) @ true_conductance
    safe_step = stability_margin / weighted_degree.max().item()
    step_size = min(float(requested_step), float(safe_step))
    true_next_state = potentials - step_size * true_node_message

    return ConductanceDataset(
        incidence=incidence,
        edge_features=edge_features,
        potentials=potentials,
        true_conductance=true_conductance,
        true_gradient=true_gradient,
        true_flux=true_flux,
        true_node_message=true_node_message,
        true_next_state=true_next_state,
        step_size=step_size,
    )


def split_excitations(
    dataset: ConductanceDataset, train_fraction: float = 0.75, seed: int = 0
) -> tuple[ConductanceDataset, ConductanceDataset]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    generator = torch.Generator(device=dataset.potentials.device)
    generator.manual_seed(seed)
    order = torch.randperm(dataset.num_excitations, generator=generator)
    train_count = int(round(train_fraction * dataset.num_excitations))
    train_count = min(max(train_count, 1), dataset.num_excitations - 1)
    return dataset.subset(order[:train_count]), dataset.subset(order[train_count:])


@torch.no_grad()
def evaluate_model(
    model: IncidenceConductanceAttention,
    dataset: ConductanceDataset,
    *,
    excitation_threshold: float = 1.0e-6,
) -> dict[str, float | bool | None]:
    model.eval()
    predicted_next, diagnostics = model(
        dataset.incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    predicted_flux = diagnostics["edge_flux"]
    predicted_conductance = diagnostics["conductance"].mean(dim=0).squeeze(-1)
    excited = dataset.excited_edges(excitation_threshold)

    node_rmse = torch.sqrt(torch.mean((predicted_next - dataset.true_next_state).square()))
    flux_rmse = torch.sqrt(torch.mean((predicted_flux - dataset.true_flux).square()))
    conductance_rmse = torch.sqrt(
        torch.mean((predicted_conductance[excited] - dataset.true_conductance[excited]).square())
    )
    correlation = _pearson_correlation(
        predicted_conductance[excited], dataset.true_conductance[excited]
    )
    correlation_value = None if correlation is None else float(correlation.cpu())
    return {
        "node_update_rmse": float(node_rmse.cpu()),
        "flux_rmse": float(flux_rmse.cpu()),
        "conductance_rmse": float(conductance_rmse.cpu()),
        "conductance_correlation_excited": correlation_value,
        "conductance_correlation_defined": correlation is not None,
        "excited_edge_fraction": float(excited.float().mean().cpu()),
        "mean_effective_step": float(diagnostics["effective_step"].mean().cpu()),
    }


def _pearson_correlation(first: Tensor, second: Tensor) -> Tensor | None:
    if first.numel() < 2:
        return None
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = torch.linalg.vector_norm(first_centered) * torch.linalg.vector_norm(
        second_centered
    )
    if denominator <= torch.finfo(first.dtype).eps:
        return None
    return torch.dot(first_centered, second_centered) / denominator


__all__ = [
    "ConductanceDataset",
    "evaluate_model",
    "make_conductance_dataset",
    "split_excitations",
]
