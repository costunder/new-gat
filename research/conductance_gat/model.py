"""Incidence conductance attention with no auxiliary graph coordinates.

The only graph computation in this module is

    H -> B H -> C_theta(B H, x_E) B H -> B.T C_theta B H.

``B`` always has shape ``(num_edges, num_nodes)``.  The edge orientation is
only a gauge: changing ``B`` to ``Q B`` must leave scalar conductance and the
node update unchanged while changing signed edge flux to ``Q q``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf


def _node_batch(node_state: Tensor) -> tuple[Tensor, bool]:
    if node_state.ndim == 2:
        return node_state.unsqueeze(0), True
    if node_state.ndim == 3:
        return node_state, False
    raise ValueError("node_state must have shape (nodes, channels) or (batch, nodes, channels)")


def _edge_batch(edge_state: Tensor) -> tuple[Tensor, bool]:
    if edge_state.ndim == 2:
        return edge_state.unsqueeze(0), True
    if edge_state.ndim == 3:
        return edge_state, False
    raise ValueError("edge_state must have shape (edges, channels) or (batch, edges, channels)")


def _restore_batch(value: Tensor, was_unbatched: bool) -> Tensor:
    return value.squeeze(0) if was_unbatched else value


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value), dtype=torch.float64)
    return torch.log(torch.expm1(value_tensor)).item()


class PositiveInvariantScalarConductance(nn.Module):
    """Predict one positive, orientation-invariant scalar per edge.

    Signed edge gradients are represented only through ``abs(BH)`` and
    ``(BH)^2``.  Optional edge attributes are assumed to be orientation
    invariant.  Consequently the prediction is identical for ``BH`` and
    ``QBH`` for every diagonal sign matrix ``Q``.
    """

    def __init__(
        self,
        channels: int,
        edge_feature_channels: int,
        hidden_channels: int = 32,
        minimum: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if edge_feature_channels < 0:
            raise ValueError("edge_feature_channels cannot be negative")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if minimum <= 0.0:
            raise ValueError("minimum must be strictly positive")

        self.channels = int(channels)
        self.edge_feature_channels = int(edge_feature_channels)
        self.minimum = float(minimum)
        input_channels = 2 * channels + edge_feature_channels
        self.network = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, edge_gradient: Tensor, edge_features: Tensor | None = None) -> Tensor:
        gradient, was_unbatched = _edge_batch(edge_gradient)
        if gradient.shape[-1] != self.channels:
            raise ValueError("edge-gradient width differs from configured channels")

        pieces = [gradient.abs(), gradient.square()]
        if self.edge_feature_channels:
            if edge_features is None:
                raise ValueError("edge_features are required by this conductance model")
            if edge_features.ndim == 2:
                features = edge_features.unsqueeze(0).expand(gradient.shape[0], -1, -1)
            elif edge_features.ndim == 3:
                features = edge_features
                if features.shape[0] == 1 and gradient.shape[0] != 1:
                    features = features.expand(gradient.shape[0], -1, -1)
            else:
                raise ValueError("edge_features must have shape (edges, features) or batched shape")
            if features.shape[:2] != gradient.shape[:2]:
                raise ValueError("edge_features and edge_gradient disagree on batch or edge count")
            if features.shape[-1] != self.edge_feature_channels:
                raise ValueError("edge-feature width differs from configured width")
            pieces.append(features.to(dtype=gradient.dtype, device=gradient.device))
        elif edge_features is not None and edge_features.shape[-1] != 0:
            raise ValueError("this conductance model was configured without edge features")

        invariant_features = torch.cat(pieces, dim=-1)
        conductance = nnf.softplus(self.network(invariant_features)) + self.minimum
        return _restore_batch(conductance, was_unbatched)


class IncidenceConductanceAttention(nn.Module):
    r"""Residual node layer ``H' = H - eta B.T C_theta B H``.

    The requested step is capped per sample by ``margin / max_weighted_degree``.
    Since ``lambda_max(B.T C B) <= 2 max_weighted_degree``, a margin below one
    makes the frozen-operator residual map non-expansive.  The cap is not an
    assertion that a state-dependent energy decreases across multiple layers;
    it is a one-step numerical stability guard.
    """

    def __init__(
        self,
        channels: int,
        edge_feature_channels: int,
        hidden_channels: int = 32,
        minimum_conductance: float = 1.0e-4,
        step_size: float = 0.03,
        stability_margin: float = 0.95,
        adaptive_stability: bool = True,
    ) -> None:
        super().__init__()
        if step_size <= 0.0:
            raise ValueError("step_size must be strictly positive")
        if not 0.0 < stability_margin < 1.0:
            raise ValueError("stability_margin must lie strictly between zero and one")
        self.channels = int(channels)
        self.requested_step = float(step_size)
        self.stability_margin = float(stability_margin)
        self.adaptive_stability = bool(adaptive_stability)
        self.conductance = PositiveInvariantScalarConductance(
            channels=channels,
            edge_feature_channels=edge_feature_channels,
            hidden_channels=hidden_channels,
            minimum=minimum_conductance,
        )

    def _effective_step(self, incidence: Tensor, conductance: Tensor) -> Tensor:
        batch_conductance, _ = _edge_batch(conductance)
        batch_size = batch_conductance.shape[0]
        requested = batch_conductance.new_full((batch_size,), self.requested_step)
        if not self.adaptive_stability:
            return requested

        weighted_degree = torch.einsum("mn,bm->bn", incidence.abs(), batch_conductance.squeeze(-1))
        largest_degree = weighted_degree.amax(dim=1).clamp_min(torch.finfo(incidence.dtype).eps)
        safe_step = self.stability_margin / largest_degree
        return torch.minimum(requested, safe_step)

    def forward(
        self,
        incidence: Tensor,
        node_state: Tensor,
        edge_features: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        state, was_unbatched = _node_batch(node_state)
        if incidence.ndim != 2 or incidence.shape[1] != state.shape[1]:
            raise ValueError("incidence must have shape (edges, nodes)")
        if state.shape[-1] != self.channels:
            raise ValueError("node-state width differs from configured channels")
        incidence = incidence.to(dtype=state.dtype, device=state.device)

        edge_gradient = torch.einsum("mn,bnd->bmd", incidence, state)
        conductance = self.conductance(edge_gradient, edge_features)
        conductance_batch, _ = _edge_batch(conductance)
        edge_flux = conductance_batch * edge_gradient
        node_message = torch.einsum("mn,bmd->bnd", incidence, edge_flux)
        effective_step = self._effective_step(incidence, conductance_batch)
        next_state = state - effective_step[:, None, None] * node_message

        output = _restore_batch(next_state, was_unbatched)
        if not return_diagnostics:
            return output
        diagnostics = {
            "edge_gradient": _restore_batch(edge_gradient, was_unbatched),
            "conductance": _restore_batch(conductance_batch, was_unbatched),
            "edge_flux": _restore_batch(edge_flux, was_unbatched),
            "node_message": _restore_batch(node_message, was_unbatched),
            "effective_step": effective_step.squeeze(0) if was_unbatched else effective_step,
        }
        return output, diagnostics


class IsotropicConductanceAttention(IncidenceConductanceAttention):
    """A fair one-scalar ``C = c I`` baseline with the same residual update."""

    def __init__(
        self,
        channels: int,
        edge_feature_channels: int = 0,
        initial_conductance: float = 1.0,
        minimum_conductance: float = 1.0e-4,
        step_size: float = 0.03,
        stability_margin: float = 0.95,
        adaptive_stability: bool = True,
    ) -> None:
        if initial_conductance <= minimum_conductance:
            raise ValueError("initial_conductance must exceed minimum_conductance")
        super().__init__(
            channels=channels,
            edge_feature_channels=edge_feature_channels,
            hidden_channels=1,
            minimum_conductance=minimum_conductance,
            step_size=step_size,
            stability_margin=stability_margin,
            adaptive_stability=adaptive_stability,
        )
        self.minimum_conductance = float(minimum_conductance)
        raw_value = _inverse_softplus(initial_conductance - minimum_conductance)
        self.raw_isotropic_conductance = nn.Parameter(torch.tensor(raw_value))
        del self.conductance

    @property
    def scalar_conductance(self) -> Tensor:
        return nnf.softplus(self.raw_isotropic_conductance) + self.minimum_conductance

    def forward(
        self,
        incidence: Tensor,
        node_state: Tensor,
        edge_features: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        del edge_features
        state, was_unbatched = _node_batch(node_state)
        if incidence.ndim != 2 or incidence.shape[1] != state.shape[1]:
            raise ValueError("incidence must have shape (edges, nodes)")
        if state.shape[-1] != self.channels:
            raise ValueError("node-state width differs from configured channels")
        incidence = incidence.to(dtype=state.dtype, device=state.device)

        edge_gradient = torch.einsum("mn,bnd->bmd", incidence, state)
        conductance = self.scalar_conductance.to(dtype=state.dtype).expand(
            state.shape[0], incidence.shape[0], 1
        )
        edge_flux = conductance * edge_gradient
        node_message = torch.einsum("mn,bmd->bnd", incidence, edge_flux)
        effective_step = self._effective_step(incidence, conductance)
        next_state = state - effective_step[:, None, None] * node_message

        output = _restore_batch(next_state, was_unbatched)
        if not return_diagnostics:
            return output
        diagnostics = {
            "edge_gradient": _restore_batch(edge_gradient, was_unbatched),
            "conductance": _restore_batch(conductance, was_unbatched),
            "edge_flux": _restore_batch(edge_flux, was_unbatched),
            "node_message": _restore_batch(node_message, was_unbatched),
            "effective_step": effective_step.squeeze(0) if was_unbatched else effective_step,
        }
        return output, diagnostics


def frozen_operator_spectral_radius(
    incidence: Tensor, conductance: Tensor, step: Tensor | float
) -> Tensor:
    """Return ``max |eig(I - eta B.T C B)|`` for diagnostics and tests."""

    if incidence.ndim != 2:
        raise ValueError("incidence must be two-dimensional")
    if conductance.ndim == 2 and conductance.shape[-1] == 1:
        conductance = conductance.squeeze(-1)
    if conductance.ndim != 1 or conductance.shape[0] != incidence.shape[0]:
        raise ValueError("conductance must contain one scalar per edge")
    weighted_laplacian = incidence.transpose(0, 1) @ (conductance[:, None] * incidence)
    eigenvalues = torch.linalg.eigvalsh(weighted_laplacian)
    eta = torch.as_tensor(step, dtype=incidence.dtype, device=incidence.device)
    return torch.max(torch.abs(1.0 - eta * eigenvalues))


__all__ = [
    "IncidenceConductanceAttention",
    "IsotropicConductanceAttention",
    "PositiveInvariantScalarConductance",
    "frozen_operator_spectral_radius",
]
