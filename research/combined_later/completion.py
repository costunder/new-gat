"""Analytic flow-completion and identifiability utilities.

The incidence convention used throughout this module is ``B.shape == (m, n)``:
rows are oriented edges and ``B.T @ q`` is the node divergence of an edge flow.
Cycle coordinates are columns of a full-rank matrix ``F`` satisfying
``B.T @ F == 0``.

The routines deliberately regularize the *physical* cycle flow ``F @ a``, not
the raw coordinate vector ``a``.  Consequently their results are invariant to
any nonsingular change of cycle chart.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _matrix(name: str, value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _selection_matrix(observed: ArrayLike | Iterable[int], size: int) -> FloatArray:
    """Convert edge indices, a boolean mask, or a linear observation map to S."""

    raw = np.asarray(observed)
    if raw.ndim == 2:
        selection = np.asarray(raw, dtype=float)
        if selection.shape[1] != size:
            raise ValueError(
                f"observation matrix has {selection.shape[1]} columns; expected {size}"
            )
        if not np.all(np.isfinite(selection)):
            raise ValueError("observation matrix contains non-finite values")
        return selection

    if raw.ndim != 1:
        raise ValueError("observed must be edge indices, a boolean mask, or a matrix")
    if raw.dtype == bool:
        if raw.size != size:
            raise ValueError(f"boolean observation mask must have length {size}")
        indices = np.flatnonzero(raw)
    else:
        if not np.issubdtype(raw.dtype, np.integer):
            rounded = np.rint(raw)
            if not np.allclose(raw, rounded):
                raise ValueError("edge observations must contain integer indices")
            raw = rounded
        indices = raw.astype(int, copy=False)

    if np.any(indices < 0) or np.any(indices >= size):
        raise IndexError("observed edge index is out of range")
    selection = np.zeros((indices.size, size), dtype=float)
    selection[np.arange(indices.size), indices] = 1.0
    return selection


def _rank_tolerance(singular_values: FloatArray, shape: tuple[int, int], rcond: float) -> float:
    if singular_values.size == 0:
        return 0.0
    return float(rcond * max(shape) * singular_values[0])


def _cycle_metric(cycle_basis: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``F``, ``G^{-1/2}``, and an orthonormal physical cycle basis."""

    basis = _matrix("cycle_basis", cycle_basis)
    gram = basis.T @ basis
    if gram.size == 0:
        return basis, np.zeros((0, 0)), basis.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(float(eigenvalues[-1]), 1.0)
    if eigenvalues[0] <= np.finfo(float).eps * max(gram.shape) * scale:
        raise ValueError("cycle_basis must have linearly independent columns")
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    orthonormal = basis @ inverse_sqrt
    return basis, inverse_sqrt, orthonormal


def orthonormal_cycle_basis(cycle_basis: ArrayLike) -> FloatArray:
    """Return ``U = F (F.T F)^(-1/2)`` with ``U.T U = I``."""

    return _cycle_metric(cycle_basis)[2]


@dataclass(frozen=True)
class ObservationSpectrum:
    """Chart-independent observability diagnostics for a cycle subspace.

    ``singular_values`` always has one entry per cycle dimension.  It therefore
    contains explicit zeros when fewer than ``beta`` scalar observations exist.
    ``noise_amplification`` is infinite unless the observation map is injective
    on the entire cycle space.
    """

    singular_values: FloatArray
    rank: int
    sigma_min: float
    sigma_min_nonzero: float
    condition_number: float
    noise_amplification: float


def cycle_observation_spectrum(
    cycle_basis: ArrayLike,
    observed: ArrayLike | Iterable[int],
    *,
    rcond: float = 1e-12,
) -> ObservationSpectrum:
    """Compute the singular spectrum of ``S U`` for an orthonormal cycle basis.

    Unlike the spectrum of ``S F``, this spectrum depends only on the physical
    cycle subspace and observation operator, not on the spanning-tree chart.
    """

    basis, _, orthonormal = _cycle_metric(cycle_basis)
    beta = basis.shape[1]
    selection = _selection_matrix(observed, basis.shape[0])
    raw = np.linalg.svd(selection @ orthonormal, compute_uv=False)
    values = np.zeros(beta, dtype=float)
    values[: raw.size] = raw
    tolerance = _rank_tolerance(raw, (selection.shape[0], beta), rcond)
    rank = int(np.count_nonzero(raw > tolerance))
    sigma_min = float(values[-1]) if beta else float("inf")
    sigma_min_nonzero = float(raw[rank - 1]) if rank else 0.0
    if beta == 0:
        condition = 1.0
        amplification = 0.0
    elif rank < beta or sigma_min <= tolerance:
        condition = float("inf")
        amplification = float("inf")
    else:
        condition = float(values[0] / sigma_min)
        amplification = float(1.0 / sigma_min)
    return ObservationSpectrum(
        singular_values=values,
        rank=rank,
        sigma_min=sigma_min,
        sigma_min_nonzero=sigma_min_nonzero,
        condition_number=condition,
        noise_amplification=amplification,
    )


def _conductance_matrix(conductance: ArrayLike | None, edge_count: int) -> FloatArray:
    if conductance is None:
        return np.eye(edge_count, dtype=float)
    value = np.asarray(conductance, dtype=float)
    if value.ndim == 1:
        if value.shape != (edge_count,):
            raise ValueError(f"conductance vector must have shape ({edge_count},)")
        if np.any(value <= 0.0) or not np.all(np.isfinite(value)):
            raise ValueError("all conductances must be finite and strictly positive")
        return np.diag(value)
    if value.shape != (edge_count, edge_count):
        raise ValueError(f"conductance matrix must have shape ({edge_count}, {edge_count})")
    if not np.allclose(value, value.T, atol=1e-12, rtol=1e-12):
        raise ValueError("conductance matrix must be symmetric")
    if np.linalg.eigvalsh(value)[0] <= 0.0:
        raise ValueError("conductance matrix must be positive definite")
    return value


def weighted_particular_flow(
    incidence: ArrayLike,
    divergence: ArrayLike,
    conductance: ArrayLike | None = None,
    *,
    rcond: float = 1e-12,
    check_conservation: bool = True,
) -> FloatArray:
    r"""Return the minimum-:math:`C^{-1}`-energy flow with given divergence.

    The returned value is

    ``q = C B (B.T C B)^dagger b``.

    ``divergence`` may be an ``(n,)`` vector or an ``(n, d)`` feature matrix.
    If it is incompatible with the connected components of the graph, the
    pseudoinverse can only satisfy its projection; by default this is reported
    as an error instead of silently changing the requested divergence.
    """

    incidence_matrix = _matrix("incidence", incidence)
    edge_count, node_count = incidence_matrix.shape
    target = np.asarray(divergence, dtype=float)
    if target.ndim not in (1, 2) or target.shape[0] != node_count:
        raise ValueError(f"divergence must have shape ({node_count},) or ({node_count}, d)")
    if not np.all(np.isfinite(target)):
        raise ValueError("divergence contains non-finite values")
    conductivity = _conductance_matrix(conductance, edge_count)
    laplacian = incidence_matrix.T @ conductivity @ incidence_matrix
    potential = np.linalg.pinv(laplacian, rcond=rcond, hermitian=True) @ target
    flow = conductivity @ incidence_matrix @ potential
    if check_conservation:
        error = incidence_matrix.T @ flow - target
        tolerance = 100.0 * rcond * max(1.0, float(np.linalg.norm(target)))
        if np.linalg.norm(error) > tolerance:
            raise ValueError(
                "divergence is not compatible with the incidence matrix "
                "(each connected component must have zero net divergence)"
            )
    return flow


def metric_minimum_anchor(
    cycle_basis: ArrayLike,
    observed: ArrayLike | Iterable[int],
    residual: ArrayLike,
    *,
    rcond: float = 1e-12,
    require_consistent: bool = True,
) -> FloatArray:
    r"""Find the minimum-physical-norm cycle coordinates matching observations.

    For ``A = S F`` and ``G = F.T F`` this evaluates

    ``a0 = G^-1 A.T (A G^-1 A.T)^dagger residual``.

    Thus ``F @ a0`` is independent of the chosen full-rank cycle chart.  With
    ``require_consistent=False`` the result is the minimum-physical-norm
    least-squares solution when the observations cannot be matched exactly.
    """

    basis = _matrix("cycle_basis", cycle_basis)
    selection = _selection_matrix(observed, basis.shape[0])
    target = np.asarray(residual, dtype=float)
    if target.ndim not in (1, 2) or target.shape[0] != selection.shape[0]:
        raise ValueError("residual must have one row/entry per scalar observation")
    observation = selection @ basis
    gram = basis.T @ basis
    if basis.shape[1] == 0:
        anchor = np.zeros((0,) + target.shape[1:], dtype=float)
    else:
        gram_inverse_at = np.linalg.solve(gram, observation.T)
        observation_kernel = observation @ gram_inverse_at
        anchor = (
            gram_inverse_at
            @ np.linalg.pinv(observation_kernel, rcond=rcond, hermitian=True)
            @ target
        )
    if require_consistent:
        error = observation @ anchor - target
        tolerance = 100.0 * rcond * max(1.0, float(np.linalg.norm(target)))
        if np.linalg.norm(error) > tolerance:
            raise ValueError("observations are inconsistent with the cycle subspace")
    return anchor


@dataclass(frozen=True)
class CompletionResult:
    """Output of analytic physical-metric cycle completion."""

    flow: FloatArray
    coefficients: FloatArray
    predicted_observations: FloatArray
    observation_residual: FloatArray


def analytic_cycle_completion(
    particular_flow: ArrayLike,
    cycle_basis: ArrayLike,
    observed: ArrayLike | Iterable[int],
    observed_values: ArrayLike,
    *,
    ridge: float = 0.0,
    rcond: float = 1e-12,
    require_exact: bool = False,
) -> CompletionResult:
    r"""Complete a flow using analytic, chart-independent cycle regression.

    This minimizes

    ``||S(q_part + F a) - y||^2 + ridge * ||F a||^2``.

    At ``ridge=0`` the least-squares minimizer with minimum physical cycle norm
    is selected.  A positive ridge value is also chart independent because it
    penalizes ``F a`` rather than the coordinate norm ``a``.
    """

    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    basis, inverse_sqrt, orthonormal = _cycle_metric(cycle_basis)
    particular = np.asarray(particular_flow, dtype=float)
    if particular.ndim not in (1, 2) or particular.shape[0] != basis.shape[0]:
        raise ValueError("particular_flow must have one row/entry per edge")
    selection = _selection_matrix(observed, basis.shape[0])
    values = np.asarray(observed_values, dtype=float)
    predicted_particular = selection @ particular
    if values.shape != predicted_particular.shape:
        raise ValueError(
            f"observed_values has shape {values.shape}; expected {predicted_particular.shape}"
        )
    target = values - predicted_particular
    design = selection @ orthonormal
    if ridge == 0.0:
        orthonormal_coordinates = np.linalg.pinv(design, rcond=rcond) @ target
    else:
        normal = design.T @ design + ridge * np.eye(design.shape[1])
        orthonormal_coordinates = np.linalg.solve(normal, design.T @ target)
    coefficients = inverse_sqrt @ orthonormal_coordinates
    flow = particular + basis @ coefficients
    predicted = selection @ flow
    residual = predicted - values
    if require_exact:
        tolerance = 100.0 * rcond * max(1.0, float(np.linalg.norm(values)))
        if np.linalg.norm(residual) > tolerance:
            raise ValueError("observations cannot be matched exactly")
    return CompletionResult(flow, coefficients, predicted, residual)


def ridge_cycle_completion(
    particular_flow: ArrayLike,
    cycle_basis: ArrayLike,
    observed: ArrayLike | Iterable[int],
    observed_values: ArrayLike,
    ridge: float,
    *,
    rcond: float = 1e-12,
) -> CompletionResult:
    """Named wrapper for positive-ridge analytic cycle completion."""

    if ridge <= 0.0:
        raise ValueError("ridge_cycle_completion requires ridge > 0")
    return analytic_cycle_completion(
        particular_flow,
        cycle_basis,
        observed,
        observed_values,
        ridge=ridge,
        rcond=rcond,
    )


@dataclass(frozen=True)
class HardObservationAffine:
    r"""Affine cycle-coordinate set ``anchor + projector @ free``.

    The projector maps arbitrary coordinate increments onto ``ker(S F)`` and is
    orthogonal in the physical metric ``G = F.T F``.
    """

    anchor: FloatArray
    projector: FloatArray
    observation_matrix: FloatArray
    rank: int

    def coordinates(self, free: ArrayLike) -> FloatArray:
        """Map arbitrary free coordinates into the hard-observation affine set."""

        value = np.asarray(free, dtype=float)
        if value.ndim not in (1, 2) or value.shape[0] != self.projector.shape[1]:
            raise ValueError("free coordinates have incompatible shape")
        return self.anchor + self.projector @ value


def hard_observation_affine(
    particular_flow: ArrayLike,
    cycle_basis: ArrayLike,
    observed: ArrayLike | Iterable[int],
    observed_values: ArrayLike,
    *,
    rcond: float = 1e-12,
) -> HardObservationAffine:
    """Construct an exact observation-preserving affine cycle parameterization."""

    basis = _matrix("cycle_basis", cycle_basis)
    particular = np.asarray(particular_flow, dtype=float)
    if particular.ndim not in (1, 2) or particular.shape[0] != basis.shape[0]:
        raise ValueError("particular_flow must have one row/entry per edge")
    selection = _selection_matrix(observed, basis.shape[0])
    values = np.asarray(observed_values, dtype=float)
    predicted_particular = selection @ particular
    if values.shape != predicted_particular.shape:
        raise ValueError("observed_values has incompatible shape")
    target = values - predicted_particular
    anchor = metric_minimum_anchor(basis, selection, target, rcond=rcond, require_consistent=True)
    observation = selection @ basis
    gram = basis.T @ basis
    beta = basis.shape[1]
    if beta == 0:
        projector = np.zeros((0, 0), dtype=float)
        rank = 0
    else:
        gram_inverse_at = np.linalg.solve(gram, observation.T)
        kernel = observation @ gram_inverse_at
        projector = (
            np.eye(beta)
            - gram_inverse_at @ np.linalg.pinv(kernel, rcond=rcond, hermitian=True) @ observation
        )
        singular_values = np.linalg.svd(observation, compute_uv=False)
        tolerance = _rank_tolerance(singular_values, observation.shape, rcond)
        rank = int(np.count_nonzero(singular_values > tolerance))
    return HardObservationAffine(anchor, projector, observation, rank)


def project_hard_observation_update(
    update: ArrayLike,
    affine: HardObservationAffine,
) -> FloatArray:
    """Project a proposed coordinate update so observed edges remain fixed."""

    value = np.asarray(update, dtype=float)
    if value.ndim not in (1, 2) or value.shape[0] != affine.projector.shape[1]:
        raise ValueError("update has incompatible shape")
    return affine.projector @ value
