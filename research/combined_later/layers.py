"""Reference PyTorch layers for chart-equivariant graph transport.

The convention used throughout this module is

``B: (num_edges, num_nodes)`` and ``F: (num_edges, cycle_rank)``.

Thus ``B @ p`` is a gradient edge signal and ``F @ a`` is a physical
circulation.  A change of spanning-tree chart has the form

``F_new = F_old @ inv(M)`` and ``a_new = M @ a_old``.

The implementation deliberately performs every nonlinear operation on
physical edge signals.  Cycle coordinates are used only to store a state and
to encode/decode it through ``F``.  This is what makes a nonlinear rollout
equivariant to a change of fundamental-cycle chart.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf

Observation = Tensor | None


def _edge_matrix(value: Tensor, name: str) -> tuple[Tensor, bool]:
    """Return an ``(edges, channels)`` view and whether input was a vector."""

    if value.ndim == 1:
        return value.unsqueeze(-1), True
    if value.ndim == 2:
        return value, False
    raise ValueError(f"{name} must have shape (items,) or (items, channels)")


def _restore_vector(value: Tensor, was_vector: bool) -> Tensor:
    return value.squeeze(-1) if was_vector else value


def center_potential(potential: Tensor) -> Tensor:
    """Fix the additive potential gauge by zero-centering every channel."""

    matrix, was_vector = _edge_matrix(potential, "potential")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    return _restore_vector(centered, was_vector)


def invariant_edge_features(
    gradient_edge: Tensor,
    cycle_edge: Tensor,
    edge_features: Tensor | None = None,
) -> Tensor:
    """Build features invariant to an arbitrary incidence-orientation flip.

    If an edge orientation is changed, both physical signed signals transform
    as ``x -> Q x`` and ``z -> Q z`` for a diagonal sign matrix ``Q``.  The
    returned features are unchanged because they contain only absolute values,
    squares, and the sign-even product ``x * z``.

    ``edge_features``, when supplied, must already be orientation invariant.
    """

    gradient, _ = _edge_matrix(gradient_edge, "gradient_edge")
    cycle, _ = _edge_matrix(cycle_edge, "cycle_edge")
    if gradient.shape != cycle.shape:
        raise ValueError("gradient_edge and cycle_edge must have the same shape")

    pieces = [
        gradient.abs(),
        gradient.square(),
        cycle.abs(),
        cycle.square(),
        gradient * cycle,
    ]
    if edge_features is not None:
        invariant, _ = _edge_matrix(edge_features, "edge_features")
        if invariant.shape[0] != gradient.shape[0]:
            raise ValueError("edge_features has the wrong number of edges")
        pieces.append(invariant)
    return torch.cat(pieces, dim=-1)


class PositiveInvariantConductance(nn.Module):
    """Positive scalar edge conductance from orientation-invariant inputs.

    The same MLP is applied independently to every edge.  A strictly positive
    floor preserves the weighted-Laplacian interpretation and avoids a
    singular conductance caused solely by the parametrization.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 32,
        edge_feature_channels: int = 0,
        minimum: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if edge_feature_channels < 0:
            raise ValueError("edge_feature_channels cannot be negative")
        if minimum <= 0.0:
            raise ValueError("minimum conductance must be strictly positive")

        self.channels = channels
        self.edge_feature_channels = edge_feature_channels
        self.minimum = float(minimum)
        input_channels = 5 * channels + edge_feature_channels
        self.network = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        gradient_edge: Tensor,
        cycle_edge: Tensor,
        edge_features: Tensor | None = None,
    ) -> Tensor:
        features = invariant_edge_features(gradient_edge, cycle_edge, edge_features=edge_features)
        if features.shape[-1] != self.network[0].in_features:
            raise ValueError(
                "input feature width does not match channels and "
                "edge_feature_channels from construction"
            )
        return nnf.softplus(self.network(features)) + self.minimum


def incidence_node_message(
    incidence: Tensor,
    potential: Tensor,
    conductance: Tensor,
) -> Tensor:
    """Compute ``B.T @ C @ B @ p`` without materializing ``C``."""

    p, was_vector = _edge_matrix(potential, "potential")
    if incidence.ndim != 2 or incidence.shape[1] != p.shape[0]:
        raise ValueError("incidence must have shape (edges, nodes)")
    c, _ = _edge_matrix(conductance, "conductance")
    if c.shape != (incidence.shape[0], 1):
        raise ValueError("conductance must contain one scalar per edge")

    gradient = incidence @ p
    message = incidence.transpose(0, 1) @ (c * gradient)
    return _restore_vector(message, was_vector)


def _observation_times_basis(observation: Tensor, basis: Tensor) -> Tensor:
    """Return ``S @ F`` for a selection or a general observation matrix."""

    if observation.ndim == 1:
        if observation.dtype == torch.bool:
            if observation.numel() != basis.shape[0]:
                raise ValueError("boolean observation mask has the wrong length")
            mask = observation.to(device=basis.device)
            return basis[mask]
        indices = observation.to(device=basis.device, dtype=torch.long)
        return basis.index_select(0, indices)

    if observation.ndim == 2:
        if observation.shape[1] != basis.shape[0]:
            raise ValueError("observation matrix must have one column per edge")
        return observation.to(device=basis.device, dtype=basis.dtype) @ basis

    raise ValueError("observation must be edge indices, a mask, or a matrix")


def hard_observation_coordinate_projector(
    basis: Tensor,
    observation: Observation,
    *,
    rtol: float | None = None,
) -> Tensor:
    r"""Project cycle increments onto ``ker(S F)`` in the physical metric.

    With ``G = F.T F`` and ``A = S F``, this returns

    ``P = I - G^-1 A.T (A G^-1 A.T)^dagger A``.

    Consequently ``A @ P = 0`` (up to numerical tolerance).  Unlike the
    Euclidean coordinate projector ``I - A^dagger A``, this projector obeys
    ``P_new = M P_old inv(M)`` under a spanning-tree chart transition.
    """

    if basis.ndim != 2:
        raise ValueError("basis must have shape (edges, cycle_rank)")
    cycle_rank = basis.shape[1]
    identity = torch.eye(cycle_rank, dtype=basis.dtype, device=basis.device)
    if observation is None or cycle_rank == 0:
        return identity

    observed_basis = _observation_times_basis(observation, basis)
    if observed_basis.shape[0] == 0:
        return identity

    gram = basis.transpose(0, 1) @ basis
    gram_inv_at = torch.linalg.solve(gram, observed_basis.transpose(0, 1))
    schur = observed_basis @ gram_inv_at
    if rtol is None:
        schur_pinv = torch.linalg.pinv(schur)
    else:
        schur_pinv = torch.linalg.pinv(schur, rtol=rtol)
    return identity - gram_inv_at @ schur_pinv @ observed_basis


def physical_cycle_increment(
    basis: Tensor,
    physical_residual: Tensor,
    observation: Observation = None,
    *,
    rtol: float | None = None,
) -> Tensor:
    r"""Encode a physical edge residual as a chart-equivariant cycle update.

    The unconstrained update is

    ``delta_a = solve(F.T F, F.T r)``.

    If ``observation`` is supplied, it is additionally projected in the
    physical ``F.T F`` metric so that ``S F delta_a = 0``.  Therefore a hard
    observed edge state is unchanged by this cycle update.
    """

    if basis.ndim != 2:
        raise ValueError("basis must have shape (edges, cycle_rank)")
    residual, was_vector = _edge_matrix(physical_residual, "physical_residual")
    if residual.shape[0] != basis.shape[0]:
        raise ValueError("basis and physical_residual disagree on edge count")

    cycle_rank = basis.shape[1]
    if cycle_rank == 0:
        empty = residual.new_zeros((0, residual.shape[1]))
        return _restore_vector(empty, was_vector)

    gram = basis.transpose(0, 1) @ basis
    rhs = basis.transpose(0, 1) @ residual
    increment = torch.linalg.solve(gram, rhs)
    if observation is not None:
        projector = hard_observation_coordinate_projector(basis, observation, rtol=rtol)
        increment = projector @ increment
    return _restore_vector(increment, was_vector)


class OrientationEquivariantEdgeResidual(nn.Module):
    """Nonlinear signed edge residual assembled from invariant gates.

    The network predicts invariant gates and applies them to signed physical
    signals.  Hence ``r(Qx, Qz) = Q r(x, z)`` for every diagonal sign matrix
    ``Q``.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 32,
        edge_feature_channels: int = 0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.edge_feature_channels = edge_feature_channels
        input_channels = 5 * channels + edge_feature_channels
        self.gate_network = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * channels),
        )

    def forward(
        self,
        gradient_edge: Tensor,
        cycle_edge: Tensor,
        edge_features: Tensor | None = None,
    ) -> Tensor:
        gradient, gradient_was_vector = _edge_matrix(gradient_edge, "gradient_edge")
        cycle, cycle_was_vector = _edge_matrix(cycle_edge, "cycle_edge")
        if gradient_was_vector != cycle_was_vector:
            raise ValueError("gradient_edge and cycle_edge ranks must agree")
        features = invariant_edge_features(gradient, cycle, edge_features=edge_features)
        if features.shape[-1] != self.gate_network[0].in_features:
            raise ValueError(
                "input feature width does not match channels and "
                "edge_feature_channels from construction"
            )
        gates = torch.tanh(self.gate_network(features))
        gradient_gate, cycle_gate = gates.chunk(2, dim=-1)
        residual = gradient_gate * gradient + cycle_gate * cycle
        return _restore_vector(residual, gradient_was_vector)


class PersistentTransportBlock(nn.Module):
    r"""One chart-equivariant persistent potential/circulation block.

    The persistent physical edge state is ``e = B p + F a``.  Conductance is
    an operator used by the node update; it is not part of this state codec.
    The cycle update is first computed as a signed physical residual and only
    then encoded in the current chart.

    Args:
        channels: Number of feature channels in both ``p`` and ``a``.
        hidden_channels: MLP width for conductance and edge residual gates.
        edge_feature_channels: Width of optional orientation-invariant edge
            attributes.
        potential_step: Initial positive residual step for ``p``.
        cycle_step: Initial positive residual step for ``a``.
        minimum_conductance: Strict lower bound on every learned conductance.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 32,
        edge_feature_channels: int = 0,
        potential_step: float = 0.1,
        cycle_step: float = 0.1,
        minimum_conductance: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if potential_step <= 0.0 or cycle_step <= 0.0:
            raise ValueError("step sizes must be strictly positive")
        self.channels = channels
        self.edge_feature_channels = edge_feature_channels
        self.conductance = PositiveInvariantConductance(
            channels,
            hidden_channels=hidden_channels,
            edge_feature_channels=edge_feature_channels,
            minimum=minimum_conductance,
        )
        # Conductance is itself invariant, so it is a valid extra scalar input
        # to the orientation-equivariant physical edge residual.
        self.edge_residual = OrientationEquivariantEdgeResidual(
            channels,
            hidden_channels=hidden_channels,
            edge_feature_channels=edge_feature_channels + 1,
        )
        self.node_mixer = nn.Linear(channels, channels, bias=False)
        with torch.no_grad():
            self.node_mixer.weight.copy_(torch.eye(channels))

        self.raw_potential_step = nn.Parameter(
            torch.tensor(_inverse_softplus(potential_step), dtype=torch.get_default_dtype())
        )
        self.raw_cycle_step = nn.Parameter(
            torch.tensor(_inverse_softplus(cycle_step), dtype=torch.get_default_dtype())
        )

    @property
    def potential_step(self) -> Tensor:
        return nnf.softplus(self.raw_potential_step)

    @property
    def cycle_step(self) -> Tensor:
        return nnf.softplus(self.raw_cycle_step)

    def forward(
        self,
        incidence: Tensor,
        basis: Tensor,
        potential: Tensor,
        cycle_coordinates: Tensor,
        edge_features: Tensor | None = None,
        observation: Observation = None,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        p, p_was_vector = _edge_matrix(potential, "potential")
        a, a_was_vector = _edge_matrix(cycle_coordinates, "cycle_coordinates")
        if p_was_vector != a_was_vector:
            raise ValueError("potential and cycle_coordinates ranks must agree")
        if p.shape[1] != self.channels or a.shape[1] != self.channels:
            raise ValueError("state channel width differs from block channels")
        if incidence.ndim != 2 or incidence.shape[1] != p.shape[0]:
            raise ValueError("incidence must have shape (edges, nodes)")
        if basis.ndim != 2 or basis.shape[0] != incidence.shape[0]:
            raise ValueError("basis must have shape (edges, cycle_rank)")
        if basis.shape[1] != a.shape[0]:
            raise ValueError("basis and cycle_coordinates disagree on cycle rank")

        gradient_edge = incidence @ p
        cycle_edge = basis @ a
        conductance = self.conductance(gradient_edge, cycle_edge, edge_features=edge_features)
        node_message = incidence.transpose(0, 1) @ (conductance * gradient_edge)

        # The centering is explicit rather than relying on B.T to happen to
        # have zero sum.  It therefore fixes the p -> p + constant gauge even
        # in the presence of learned channel mixing and numerical drift.
        potential_delta = torch.tanh(self.node_mixer(node_message))
        next_p = center_potential(p - self.potential_step * potential_delta)

        if edge_features is None:
            residual_features = conductance
        else:
            invariant_attributes, _ = _edge_matrix(edge_features, "edge_features")
            residual_features = torch.cat([invariant_attributes, conductance], dim=-1)
        physical_residual = self.edge_residual(
            gradient_edge,
            cycle_edge,
            edge_features=residual_features,
        )
        increment = physical_cycle_increment(basis, physical_residual, observation=observation)
        increment_matrix, _ = _edge_matrix(increment, "cycle_increment")
        next_a = a + self.cycle_step * increment_matrix

        next_p_out = _restore_vector(next_p, p_was_vector)
        next_a_out = _restore_vector(next_a, a_was_vector)
        if not return_diagnostics:
            return next_p_out, next_a_out

        diagnostics = {
            "gradient_edge": _restore_vector(gradient_edge, p_was_vector),
            "cycle_edge": _restore_vector(cycle_edge, a_was_vector),
            "conductance": conductance,
            "node_message": _restore_vector(node_message, p_was_vector),
            "physical_cycle_residual": physical_residual,
            "cycle_increment": increment,
        }
        return next_p_out, next_a_out, diagnostics


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value), dtype=torch.float64)
    return torch.log(torch.expm1(value_tensor)).item()


__all__ = [
    "PositiveInvariantConductance",
    "OrientationEquivariantEdgeResidual",
    "PersistentTransportBlock",
    "center_potential",
    "hard_observation_coordinate_projector",
    "incidence_node_message",
    "invariant_edge_features",
    "physical_cycle_increment",
]
