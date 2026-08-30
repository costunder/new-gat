# .gitignore

````text
.venv/
.venv-gpu/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.matplotlib/
.gpu-environment.json
.gpu-environment.freeze.txt
.coverage
htmlcov/
build/
dist/
*.egg-info/
runs/
research/*/results/*
results/*
!results/.gitkeep
data/*
!data/.gitkeep
````

# constraints-cu126.txt

````text
# CUDA_WHEEL_TAG=cu126
# Official wheel channel: https://download.pytorch.org/whl/cu126
# Exact top-level stack; resolved transitive packages are snapshotted after setup.
matplotlib==3.11.1
networkx==3.6.1
numpy==2.4.6
ogb==1.3.6
pandas==3.0.5
pytest==9.1.1
pytest-cov==7.1.0
PyYAML==6.0.3
ruff==0.16.5
scikit-learn==1.9.0
scipy==1.17.1
torch==2.13.0
torch-geometric==2.8.0.post1
tqdm==4.70.0
````

# constraints-cu130.txt

````text
# CUDA_WHEEL_TAG=cu130
# Official wheel channel: https://download.pytorch.org/whl/cu130
# Exact top-level stack; resolved transitive packages are snapshotted after setup.
matplotlib==3.11.1
networkx==3.6.1
numpy==2.4.6
ogb==1.3.6
pandas==3.0.5
pytest==9.1.1
pytest-cov==7.1.0
PyYAML==6.0.3
ruff==0.16.5
scikit-learn==1.9.0
scipy==1.17.1
torch==2.13.0
torch-geometric==2.8.0.post1
tqdm==4.70.0
````

# constraints-cu132.txt

````text
# CUDA_WHEEL_TAG=cu132
# Official wheel channel: https://download.pytorch.org/whl/cu132
# Exact top-level stack; resolved transitive packages are snapshotted after setup.
matplotlib==3.11.1
networkx==3.6.1
numpy==2.4.6
ogb==1.3.6
pandas==3.0.5
pytest==9.1.1
pytest-cov==7.1.0
PyYAML==6.0.3
ruff==0.16.5
scikit-learn==1.9.0
scipy==1.17.1
torch==2.13.0
torch-geometric==2.8.0.post1
tqdm==4.70.0
````

# pyproject.toml

````toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chartgat"
version = "0.1.0"
description = "Independent incidence-conductance, cycle-PE, and tree-augmentation experiments"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "networkx>=3.2",
  "torch>=2.2",
  "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.6",
]
paper = [
  "ogb>=1.3.6",
  "pandas>=2.1",
  "scikit-learn>=1.4",
  "scipy>=1.11",
  "torch-geometric>=2.5",
  "tqdm>=4.66",
]
combined = [
  "pandas>=2.1",
  "matplotlib>=3.8",
]

[tool.setuptools.packages.find]
where = ["src", "."]
include = ["chartgat*", "research*"]
exclude = ["research.combined_later*", "research.*.tests*"]
namespaces = false

[tool.setuptools.package-data]
"research.conductance_gat" = ["config.yaml", "datasets.yaml"]
"research.cycle_pe" = ["config.yaml", "datasets.yaml"]
"research.tree_augmentation" = ["config.yaml", "datasets.yaml"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = [
  "tests",
  "research/conductance_gat/tests",
  "research/cycle_pe/tests",
  "research/tree_augmentation/tests",
]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
````

# requirements-lock.txt

````text
# Exact top-level research stack.  These versions all support Python 3.11.
# CUDA wheel selection is intentionally kept in constraints-cu*.txt and
# scripts/setup_gpu.sh.  The resolved transitive environment is recorded by
# setup_gpu.sh after installation.
matplotlib==3.11.1
networkx==3.6.1
numpy==2.4.6
ogb==1.3.6
pandas==3.0.5
pytest==9.1.1
pytest-cov==7.1.0
PyYAML==6.0.3
ruff==0.16.5
scikit-learn==1.9.0
scipy==1.17.1
torch==2.13.0
torch-geometric==2.8.0.post1
tqdm==4.70.0
````

# requirements-paper.txt

````text
# This file constrains every direct paper dependency to requirements-lock.txt.
# For a Linux GPU host, use scripts/setup_gpu.sh instead: it additionally picks
# and verifies the matching official CUDA wheel channel.
-c requirements-lock.txt
-e .[dev,paper]
````

# requirements.txt

````text
# Runtime and experiment dependencies. Install with:
#   bash scripts/setup.sh        # Linux/MobaXterm
#   .\scripts\setup.ps1         # Windows PowerShell
-e .[dev]
````

# research/__init__.py

````python
"""Physically separated research tracks."""
````

# research/combined_later/__init__.py

````python
"""Quarantined integration prototype; excluded from the active research pipeline."""
````

# research/combined_later/completion.py

````python
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
````

# research/combined_later/layers.py

````python
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
````

# research/combined_later/run_certify.py

````python
"""E0: algebraic, orientation-gauge, and chart-equivariance certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chartgat.algebra import (
    chart_transition,
    decode_edge_state,
    encode_edge_state,
    flip_cycle_basis,
    flip_incidence,
    fundamental_cycle_basis,
    incidence_matrix,
)
from chartgat.graphs import make_connected_graph, spanning_tree_indices
from research.combined_later.layers import PersistentTransportBlock


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def certify_graph(seed: int, num_nodes: int, extra_edges: int, depth: int) -> dict[str, float]:
    edges = make_connected_graph(num_nodes, extra_edges, seed=seed)
    B = incidence_matrix(num_nodes, edges)
    trees = [
        spanning_tree_indices(num_nodes, edges, mode="bfs"),
        spanning_tree_indices(num_nodes, edges, mode="dfs"),
        spanning_tree_indices(num_nodes, edges, mode="random", seed=seed + 17),
    ]
    bases_and_chords = [fundamental_cycle_basis(B, tree, return_chords=True) for tree in trees]
    bases = [item[0] for item in bases_and_chords]

    cycle_null_error = max(_max_abs(B.T @ basis) for basis in bases)
    chord_identity_error = max(
        _max_abs(basis[chords] - np.eye(basis.shape[1])) for basis, chords in bases_and_chords
    )
    M10 = chart_transition(bases[0], bases[1])
    M21 = chart_transition(bases[1], bases[2])
    M20 = chart_transition(bases[0], bases[2])
    transition_error = max(
        _max_abs(bases[1] @ M10 - bases[0]),
        _max_abs(bases[2] @ M21 - bases[1]),
    )
    cocycle_error = _max_abs(M21 @ M10 - M20)

    rng = np.random.default_rng(seed + 101)
    edge_state = rng.normal(size=(B.shape[0], 3))
    p, a = encode_edge_state(B, bases[0], edge_state)
    reconstruction_error = _max_abs(decode_edge_state(B, bases[0], p, a) - edge_state)

    signs = rng.choice(np.asarray([-1.0, 1.0]), size=B.shape[0])
    B_flipped = flip_incidence(B, signs)
    F_flipped = flip_cycle_basis(bases[0], signs)
    orientation_null_error = _max_abs(B_flipped.T @ F_flipped)

    torch.manual_seed(seed)
    dtype = torch.float64
    B_t = torch.as_tensor(B, dtype=dtype)
    F0_t = torch.as_tensor(bases[0], dtype=dtype)
    F1_t = torch.as_tensor(bases[1], dtype=dtype)
    M10_t = torch.as_tensor(M10, dtype=dtype)
    potential = torch.randn(num_nodes, 2, dtype=dtype)
    potential = potential - potential.mean(dim=0, keepdim=True)
    a0 = torch.randn(bases[0].shape[1], 2, dtype=dtype)
    a1 = M10_t @ a0
    edge_features = torch.randn(B.shape[0], 2, dtype=dtype)
    block = PersistentTransportBlock(channels=2, hidden_channels=16, edge_feature_channels=2).to(
        dtype=dtype
    )

    p0, p1 = potential.clone(), potential.clone()
    for _ in range(depth):
        p0, a0 = block(B_t, F0_t, p0, a0, edge_features=edge_features)
        p1, a1 = block(B_t, F1_t, p1, a1, edge_features=edge_features)
    multilayer_p_error = float(torch.max(torch.abs(p0 - p1)).item())
    multilayer_cycle_error = float(torch.max(torch.abs(F0_t @ a0 - F1_t @ a1)).item())

    Q_signs = torch.as_tensor(signs, dtype=dtype)
    Bq_t = Q_signs[:, None] * B_t
    Fq_t = Q_signs[:, None] * F0_t
    p_base, a_base = block(
        B_t, F0_t, potential, torch.as_tensor(a, dtype=dtype)[:, :2], edge_features
    )
    p_flip, a_flip = block(
        Bq_t, Fq_t, potential, torch.as_tensor(a, dtype=dtype)[:, :2], edge_features
    )
    orientation_layer_error = max(
        float(torch.max(torch.abs(p_base - p_flip)).item()),
        float(torch.max(torch.abs(Fq_t @ a_flip - Q_signs[:, None] * (F0_t @ a_base))).item()),
    )

    return {
        "cycle_null_error": cycle_null_error,
        "chord_identity_error": chord_identity_error,
        "transition_error": transition_error,
        "cocycle_error": cocycle_error,
        "reconstruction_error": reconstruction_error,
        "orientation_null_error": orientation_null_error,
        "multilayer_p_error": multilayer_p_error,
        "multilayer_cycle_error": multilayer_cycle_error,
        "orientation_layer_error": orientation_layer_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/combined_later/certification.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--graphs", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=9)
    parser.add_argument("--extra-edges", type=int, default=7)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()

    per_graph = [
        certify_graph(args.seed + index, args.nodes, args.extra_edges, args.depth)
        for index in range(args.graphs)
    ]
    maxima = {key: max(result[key] for result in per_graph) for key in per_graph[0]}
    passed = all(value <= args.tolerance for value in maxima.values())
    payload = {
        "experiment": "E0_algebraic_symmetry_certification",
        "passed": passed,
        "tolerance": args.tolerance,
        "configuration": vars(args) | {"output": str(args.output)},
        "max_errors": maxima,
        "per_graph": per_graph,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "max_errors": maxima}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
````

# research/combined_later/run_fixed_c.py

````python
"""MVP fixed-conductance flow completion with hard observation preservation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix
from chartgat.graphs import make_connected_graph, spanning_tree_indices
from research.combined_later.completion import (
    analytic_cycle_completion,
    hard_observation_affine,
)
from research.combined_later.layers import (
    OrientationEquivariantEdgeResidual,
    hard_observation_coordinate_projector,
)
from research.combined_later.synthetic import structured_cycle_flows


@dataclass
class Dataset:
    q_part: np.ndarray
    target: np.ndarray
    edge_features: np.ndarray
    anchor_train: np.ndarray
    anchor_unseen: np.ndarray
    observed: np.ndarray
    missing: np.ndarray


class PhysicalCompletionModel(nn.Module):
    """Nonlinear physical-edge refinement encoded in the active cycle chart."""

    def __init__(self, edge_feature_channels: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                OrientationEquivariantEdgeResidual(
                    channels=1,
                    hidden_channels=hidden,
                    edge_feature_channels=edge_feature_channels,
                )
                for _ in range(depth)
            ]
        )
        self.raw_steps = nn.Parameter(torch.full((depth,), -1.5))

    @staticmethod
    def _encoder(basis: Tensor, observed: Tensor) -> Tensor:
        gram = basis.T @ basis
        unconstrained = torch.linalg.solve(gram, basis.T)
        projector = hard_observation_coordinate_projector(basis, observed)
        return projector @ unconstrained

    def forward(
        self,
        basis: Tensor,
        observed: Tensor,
        q_part: Tensor,
        anchor: Tensor,
        edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # q_part: (samples, edges), anchor: (samples, beta)
        encoder = self._encoder(basis, observed)
        coordinates = anchor
        samples, edges = q_part.shape
        for layer, raw_step in zip(self.layers, self.raw_steps, strict=True):
            cycle = coordinates @ basis.T
            residual = layer(
                q_part.reshape(samples * edges, 1),
                cycle.reshape(samples * edges, 1),
                edge_features.reshape(samples * edges, edge_features.shape[-1]),
            ).reshape(samples, edges)
            coordinates = coordinates + torch.nn.functional.softplus(raw_step) * (
                residual @ encoder.T
            )
        return q_part + coordinates @ basis.T, coordinates


class RawCoordinateBaseline(nn.Module):
    """Negative baseline whose learned outputs are tied to one raw tree chart."""

    def __init__(self, edges: int, beta: int, feature_channels: int, hidden: int) -> None:
        super().__init__()
        width = edges * (feature_channels + 2)
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, beta),
        )

    def forward(
        self,
        basis: Tensor,
        observed: Tensor,
        q_part: Tensor,
        anchor: Tensor,
        edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cycle_anchor = anchor @ basis.T
        inputs = torch.cat(
            [q_part[..., None], cycle_anchor[..., None], edge_features], dim=-1
        ).flatten(start_dim=1)
        proposed = self.network(inputs)
        projector = hard_observation_coordinate_projector(basis, observed)
        coordinates = anchor + proposed @ projector.T
        return q_part + coordinates @ basis.T, coordinates


def _build_dataset(args: argparse.Namespace) -> tuple[Dataset, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    edges = make_connected_graph(args.nodes, args.extra_edges, seed=args.seed)
    incidence = incidence_matrix(args.nodes, edges)
    tree_train = spanning_tree_indices(args.nodes, edges, mode="bfs")
    basis_train = fundamental_cycle_basis(incidence, tree_train)
    for offset in range(1, 100):
        tree_unseen = spanning_tree_indices(
            args.nodes, edges, mode="random", seed=args.seed + offset
        )
        basis_unseen = fundamental_cycle_basis(incidence, tree_unseen)
        if not np.array_equal(basis_train, basis_unseen):
            break
    else:
        raise RuntimeError("failed to construct a distinct unseen spanning-tree chart")

    latent = structured_cycle_flows(
        incidence,
        num_samples=args.samples,
        seed=args.seed + 101,
        scale=args.cycle_scale,
        return_latents=True,
    )
    potentials = latent.node_features[..., 0]
    potentials -= potentials.mean(axis=1, keepdims=True)
    gradients = np.einsum("mn,sn->sm", incidence, potentials)
    static_conductance = 0.35 + np.logaddexp(0.0, rng.normal(size=incidence.shape[0]))
    q_part = gradients * static_conductance[None, :]
    target = q_part + latent.cycle_flows

    observed_count = max(1, int(round(args.observed_fraction * incidence.shape[0])))
    observed_count = min(observed_count, max(1, basis_train.shape[1] - 1))
    observed = np.sort(rng.choice(incidence.shape[0], observed_count, replace=False))
    missing = np.asarray(
        [edge for edge in range(incidence.shape[0]) if edge not in set(observed)],
        dtype=np.int64,
    )
    observation_mask = np.zeros(incidence.shape[0])
    observation_mask[observed] = 1.0
    observed_filled = np.zeros_like(target)
    observed_filled[:, observed] = target[:, observed]
    endpoint_magnitude = np.einsum(
        "mn,sn->sm", np.abs(incidence), np.abs(latent.node_features[..., 1])
    )
    edge_features = np.concatenate(
        [
            latent.edge_features,
            endpoint_magnitude[..., None],
            np.broadcast_to(static_conductance, target.shape)[..., None],
            np.broadcast_to(observation_mask, target.shape)[..., None],
            np.abs(observed_filled)[..., None],
        ],
        axis=-1,
    )

    anchor_train = np.stack(
        [
            hard_observation_affine(
                q_part[index], basis_train, observed, target[index, observed]
            ).anchor
            for index in range(args.samples)
        ]
    )
    anchor_unseen = np.stack(
        [
            hard_observation_affine(
                q_part[index], basis_unseen, observed, target[index, observed]
            ).anchor
            for index in range(args.samples)
        ]
    )
    return (
        Dataset(
            q_part=q_part,
            target=target,
            edge_features=edge_features,
            anchor_train=anchor_train,
            anchor_unseen=anchor_unseen,
            observed=observed,
            missing=missing,
        ),
        incidence,
        basis_train,
        basis_unseen,
    )


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _rmse(prediction: Tensor, target: Tensor, indices: Tensor | None = None) -> float:
    difference = prediction - target
    if indices is not None:
        difference = difference.index_select(1, indices)
    return float(torch.sqrt(torch.mean(difference.square())).item())


def _evaluate(
    model: nn.Module,
    basis: Tensor,
    observed: Tensor,
    q_part: Tensor,
    anchor: Tensor,
    features: Tensor,
    target: Tensor,
    missing: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        prediction, _ = model(basis, observed, q_part, anchor, features)
    metrics = {
        "full_rmse": _rmse(prediction, target),
        "missing_rmse": _rmse(prediction, target, missing),
        "observed_max_abs_error": float(
            torch.max(
                torch.abs(prediction.index_select(1, observed) - target.index_select(1, observed))
            ).item()
        ),
    }
    return prediction, metrics


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dataset, incidence_np, basis_train_np, basis_unseen_np = _build_dataset(args)
    samples = dataset.target.shape[0]
    train_end = int(0.7 * samples)
    valid_end = int(0.85 * samples)
    train_index = torch.arange(0, train_end, device=device)
    test_index = torch.arange(valid_end, samples, device=device)

    q_part = _tensor(dataset.q_part, device)
    target = _tensor(dataset.target, device)
    features = _tensor(dataset.edge_features, device)
    anchor_train = _tensor(dataset.anchor_train, device)
    anchor_unseen = _tensor(dataset.anchor_unseen, device)
    basis_train = _tensor(basis_train_np, device)
    basis_unseen = _tensor(basis_unseen_np, device)
    observed = torch.as_tensor(dataset.observed, dtype=torch.long, device=device)
    missing = torch.as_tensor(dataset.missing, dtype=torch.long, device=device)

    physical = PhysicalCompletionModel(features.shape[-1], args.hidden, args.depth).to(
        device=device, dtype=torch.float64
    )
    raw = RawCoordinateBaseline(
        edges=target.shape[1],
        beta=basis_train.shape[1],
        feature_channels=features.shape[-1],
        hidden=args.hidden,
    ).to(device=device, dtype=torch.float64)
    optimizers = {
        "physical": torch.optim.Adam(physical.parameters(), lr=args.learning_rate),
        "raw_coordinate": torch.optim.Adam(raw.parameters(), lr=args.learning_rate),
    }
    models: dict[str, nn.Module] = {"physical": physical, "raw_coordinate": raw}
    history: list[dict[str, float | int | str]] = []
    for epoch in range(args.epochs):
        for name, model in models.items():
            model.train()
            prediction, _ = model(
                basis_train,
                observed,
                q_part.index_select(0, train_index),
                anchor_train.index_select(0, train_index),
                features.index_select(0, train_index),
            )
            error = prediction.index_select(1, missing) - target.index_select(
                0, train_index
            ).index_select(1, missing)
            loss = error.square().mean()
            optimizers[name].zero_grad()
            loss.backward()
            optimizers[name].step()
            history.append({"epoch": epoch, "model": name, "train_mse": float(loss.item())})

    test_q = q_part.index_select(0, test_index)
    test_target = target.index_select(0, test_index)
    test_features = features.index_select(0, test_index)
    test_anchor_train = anchor_train.index_select(0, test_index)
    test_anchor_unseen = anchor_unseen.index_select(0, test_index)

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, Tensor] = {}
    for name, model in models.items():
        same_prediction, same_metrics = _evaluate(
            model,
            basis_train,
            observed,
            test_q,
            test_anchor_train,
            test_features,
            test_target,
            missing,
        )
        unseen_prediction, unseen_metrics = _evaluate(
            model,
            basis_unseen,
            observed,
            test_q,
            test_anchor_unseen,
            test_features,
            test_target,
            missing,
        )
        same_metrics["unseen_chart_missing_rmse"] = unseen_metrics["missing_rmse"]
        same_metrics["chart_variation_max_abs"] = float(
            torch.max(torch.abs(same_prediction - unseen_prediction)).item()
        )
        same_metrics["unseen_observed_max_abs_error"] = unseen_metrics["observed_max_abs_error"]
        metrics[name] = same_metrics
        predictions[name] = same_prediction

    particular_prediction = test_q
    anchor_prediction = test_q + test_anchor_train @ basis_train.T
    ridge_predictions = []
    for index in range(valid_end, samples):
        ridge_predictions.append(
            analytic_cycle_completion(
                dataset.q_part[index],
                basis_train_np,
                dataset.observed,
                dataset.target[index, dataset.observed],
                ridge=args.ridge,
            ).flow
        )
    ridge_prediction = _tensor(np.stack(ridge_predictions), device)
    for name, prediction in {
        "particular_only": particular_prediction,
        "analytic_anchor": anchor_prediction,
        "analytic_ridge": ridge_prediction,
    }.items():
        metrics[name] = {
            "full_rmse": _rmse(prediction, test_target),
            "missing_rmse": _rmse(prediction, test_target, missing),
            "observed_max_abs_error": float(
                torch.max(
                    torch.abs(
                        prediction.index_select(1, observed) - test_target.index_select(1, observed)
                    )
                ).item()
            ),
        }

    physical_prediction = predictions["physical"]
    conservation_error = physical_prediction @ _tensor(incidence_np, device) - test_q @ _tensor(
        incidence_np, device
    )
    metrics["physical"]["conservation_max_abs_error"] = float(
        torch.max(torch.abs(conservation_error)).item()
    )
    summary: dict[str, object] = {
        "experiment": "fixed_C_hard_observation_completion",
        "device": str(device),
        "nodes": args.nodes,
        "edges": int(incidence_np.shape[0]),
        "cycle_rank": int(basis_train_np.shape[1]),
        "samples": samples,
        "observed_edges": dataset.observed.tolist(),
        "observation_rank": int(np.linalg.matrix_rank(basis_train_np[dataset.observed])),
        "epochs": args.epochs,
        "metrics": metrics,
    }
    return pd.DataFrame.from_records(history), summary


def _plot_history(history: pd.DataFrame, output: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for name, group in history.groupby("model"):
        axis.plot(group["epoch"], group["train_mse"], label=name)
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("missing-edge train MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/combined_later/fixed_c"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nodes", type=int, default=14)
    parser.add_argument("--extra-edges", type=int, default=9)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--observed-fraction", type=float, default=0.25)
    parser.add_argument("--cycle-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    history, summary = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "training.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_history(history, args.output_dir / "training.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
````

# research/combined_later/run_identifiability.py

````python
"""E1: cycle-nullspace identifiability and observation-conditioning sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix
from chartgat.graphs import make_connected_graph, spanning_tree_indices
from research.combined_later.completion import (
    analytic_cycle_completion,
    cycle_observation_spectrum,
    weighted_particular_flow,
)
from research.combined_later.synthetic import structured_cycle_flow


def _greedy_observation_order(cycle_basis: np.ndarray) -> np.ndarray:
    remaining = list(range(cycle_basis.shape[0]))
    selected: list[int] = []
    while remaining:
        best_edge = remaining[0]
        best_score = (-1, -1.0, -1.0)
        for edge in remaining:
            spectrum = cycle_observation_spectrum(cycle_basis, selected + [edge])
            finite_sigma = spectrum.sigma_min if np.isfinite(spectrum.sigma_min) else 0.0
            score = (spectrum.rank, spectrum.sigma_min_nonzero, finite_sigma)
            if score > best_score:
                best_score = score
                best_edge = edge
        selected.append(best_edge)
        remaining.remove(best_edge)
    return np.asarray(selected, dtype=np.int64)


def _relative_error(prediction: np.ndarray, target: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(prediction - target) / max(scale, 1.0e-12))


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    edges = make_connected_graph(args.nodes, args.extra_edges, seed=args.seed)
    incidence = incidence_matrix(args.nodes, edges)
    tree = spanning_tree_indices(args.nodes, edges, mode="bfs")
    cycle_basis, chords = fundamental_cycle_basis(incidence, tree, return_chords=True)
    beta = cycle_basis.shape[1]
    rng = np.random.default_rng(args.seed + 1)

    p = rng.normal(size=args.nodes)
    p -= p.mean()
    seed_flow = incidence @ p
    divergence = incidence.T @ seed_flow
    particular = weighted_particular_flow(incidence, divergence)
    cycle = structured_cycle_flow(
        incidence,
        node_features=rng.normal(size=(args.nodes, 3)),
        edge_features=rng.normal(size=(incidence.shape[0], 2)),
        scale=args.cycle_scale,
    )
    target = particular + cycle
    pair_divergence_error = float(np.max(np.abs(incidence.T @ target - incidence.T @ particular)))
    cycle_scale = float(np.linalg.norm(cycle))

    all_edges = np.arange(incidence.shape[0], dtype=np.int64)
    tree_set = set(tree.tolist())
    tree_order = np.asarray(list(tree) + list(chords), dtype=np.int64)
    chord_order = np.asarray(list(chords) + [e for e in all_edges if e in tree_set])
    greedy_order = _greedy_observation_order(cycle_basis)
    fixed_orders = {
        "tree_first": tree_order,
        "chord_first": chord_order,
        "rank_greedy": greedy_order,
    }

    records: list[dict[str, object]] = []
    for observed_count in range(incidence.shape[0] + 1):
        for repeat in range(args.repeats):
            orders = fixed_orders | {"random": rng.permutation(all_edges)}
            for strategy, order in orders.items():
                observed = np.asarray(order[:observed_count], dtype=np.int64)
                spectrum = cycle_observation_spectrum(cycle_basis, observed)
                observed_values = target[observed]
                noiseless = analytic_cycle_completion(
                    particular, cycle_basis, observed, observed_values
                )
                noise = args.noise_std * rng.normal(size=observed_count)
                noisy_values = observed_values + noise
                noisy_ls = analytic_cycle_completion(
                    particular, cycle_basis, observed, noisy_values
                )
                noisy_ridge = analytic_cycle_completion(
                    particular,
                    cycle_basis,
                    observed,
                    noisy_values,
                    ridge=args.ridge,
                )
                records.append(
                    {
                        "strategy": strategy,
                        "observed_count": observed_count,
                        "observed_fraction": observed_count / incidence.shape[0],
                        "repeat": repeat,
                        "rank": spectrum.rank,
                        "beta": beta,
                        "sigma_min": spectrum.sigma_min,
                        "sigma_min_nonzero": spectrum.sigma_min_nonzero,
                        "condition_number": spectrum.condition_number,
                        "noise_amplification": spectrum.noise_amplification,
                        "noiseless_relative_error": _relative_error(
                            noiseless.flow, target, cycle_scale
                        ),
                        "noisy_ls_relative_error": _relative_error(
                            noisy_ls.flow, target, cycle_scale
                        ),
                        "noisy_ridge_relative_error": _relative_error(
                            noisy_ridge.flow, target, cycle_scale
                        ),
                    }
                )

    frame = pd.DataFrame.from_records(records)
    full_rank = frame[frame["rank"] == beta]
    first_full_rank: dict[str, int | None] = {}
    for strategy, group in frame.groupby("strategy"):
        strategy_full_rank = group[group["rank"] == beta]
        first_full_rank[str(strategy)] = (
            int(strategy_full_rank["observed_count"].min())
            if not strategy_full_rank.empty
            else None
        )
    summary: dict[str, object] = {
        "experiment": "E1_nullspace_identifiability",
        "nodes": args.nodes,
        "edges": int(incidence.shape[0]),
        "cycle_rank": beta,
        "pair_divergence_error": pair_divergence_error,
        "first_full_rank_observation_count": first_full_rank,
        "max_full_rank_noiseless_relative_error": (
            float(full_rank["noiseless_relative_error"].max()) if not full_rank.empty else None
        ),
        "noise_std": args.noise_std,
        "ridge": args.ridge,
    }
    return frame, summary


def _write_plot(frame: pd.DataFrame, output: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = frame.groupby(["strategy", "observed_count"], as_index=False).agg(
        rank=("rank", "mean"),
        noiseless=("noiseless_relative_error", "mean"),
        noisy_ls=("noisy_ls_relative_error", "mean"),
        noisy_ridge=("noisy_ridge_relative_error", "mean"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy, group in grouped.groupby("strategy"):
        axes[0].plot(group["observed_count"], group["noiseless"], label=strategy)
        axes[1].plot(group["observed_count"], group["rank"], label=strategy)
    axes[0].set_yscale("symlog", linthresh=1.0e-12)
    axes[0].set_xlabel("observed edge count")
    axes[0].set_ylabel("noiseless relative reconstruction error")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("observed edge count")
    axes[1].set_ylabel("rank(S U_c)")
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/combined_later/identifiability"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--extra-edges", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--cycle-scale", type=float, default=1.0)
    args = parser.parse_args()

    frame, summary = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "sweep.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_plot(frame, args.output_dir / "identifiability.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
````

# research/combined_later/synthetic.py

````python
"""Small graph constructors and chart-independent synthetic cycle signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticCycleBatch:
    """Latents returned by :func:`structured_cycle_flows`."""

    cycle_flows: FloatArray
    raw_edge_signals: FloatArray
    node_features: FloatArray
    edge_features: FloatArray


def _feature_batch(
    value: ArrayLike | None,
    samples: int,
    elements: int,
    channels: int,
    rng: np.random.Generator,
    name: str,
) -> FloatArray:
    if value is None:
        return rng.normal(size=(samples, elements, channels))
    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and array.shape[0] == elements:
        return np.broadcast_to(array, (samples,) + array.shape).copy()
    if array.ndim == 3 and array.shape[:2] == (samples, elements):
        return array.copy()
    raise ValueError(f"{name} must have shape ({elements}, d) or ({samples}, {elements}, d)")


def structured_cycle_flows(
    incidence: ArrayLike,
    num_samples: int = 1,
    *,
    node_features: ArrayLike | None = None,
    edge_features: ArrayLike | None = None,
    seed: int | None = None,
    scale: float = 1.0,
    return_latents: bool = False,
) -> FloatArray | SyntheticCycleBatch:
    r"""Generate structured circulation without referring to a cycle chart.

    A feature-dependent physical edge signal is first constructed from ``B``;
    it is then projected with

    ``P_cyc = I - B (B.T B)^dagger B.T``.

    The generator consequently depends on the physical graph and features, not
    on a selected spanning tree or fundamental-cycle coordinates.  Reorienting
    edges by ``B -> Q B`` reorients its output by the same ``Q``.
    """

    matrix = np.asarray(incidence, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("incidence must be a two-dimensional array")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    edge_count, node_count = matrix.shape
    rng = np.random.default_rng(seed)
    nodes = _feature_batch(node_features, num_samples, node_count, 3, rng, "node_features")
    edges = _feature_batch(edge_features, num_samples, edge_count, 2, rng, "edge_features")
    if nodes.shape[2] == 0 or edges.shape[2] == 0:
        raise ValueError("node_features and edge_features need at least one channel")

    # The oriented drop changes sign under a row flip of B.  All factors in the
    # weight are orientation invariant, so the raw signal transforms covariantly.
    node_signal = nodes[..., 0]
    oriented_drop = np.einsum("mn,sn->sm", matrix, node_signal)
    endpoint_magnitude = np.einsum(
        "mn,sn->sm", np.abs(matrix), np.abs(nodes[..., min(1, nodes.shape[2] - 1)])
    )
    edge_channel = edges[..., 0]
    logits = 0.65 * edge_channel + 0.25 * endpoint_magnitude
    weights = np.logaddexp(0.0, logits) + 0.1
    raw = weights * np.tanh(oriented_drop)

    laplacian = matrix.T @ matrix
    gradient_projector = matrix @ np.linalg.pinv(laplacian, hermitian=True) @ matrix.T
    cycle_projector = np.eye(edge_count) - gradient_projector
    cycles = raw @ cycle_projector.T
    norms = np.linalg.norm(cycles, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 100.0 * np.finfo(float).eps
    cycles[nonzero] *= scale / norms[nonzero]
    cycles[~nonzero] = 0.0

    if return_latents:
        return SyntheticCycleBatch(cycles, raw, nodes, edges)
    return cycles


def structured_cycle_flow(
    incidence: ArrayLike,
    **kwargs: object,
) -> FloatArray:
    """Convenience wrapper returning one structured cycle-flow vector."""

    result = structured_cycle_flows(incidence, num_samples=1, **kwargs)
    if isinstance(result, SyntheticCycleBatch):
        return result.cycle_flows[0]
    return result[0]
````

# research/combined_later/tests/__init__.py

````python
"""Tests retained with the postponed integration prototype."""
````

# research/combined_later/tests/test_completion.py

````python
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
````

# research/combined_later/tests/test_layers.py

````python
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
````

# research/conductance_gat/__init__.py

````python
"""Independent incidence-conductance-attention research track."""

from .model import (
    IncidenceConductanceAttention,
    IsotropicConductanceAttention,
    PositiveInvariantScalarConductance,
)
from .sparse import (
    PackedGraphBatch,
    SparseIncidenceConductanceLayer,
    SparsePositiveConductance,
    edge_divergence,
    edge_gradient,
    pack_graph_examples,
)
from .synthetic import ConductanceDataset, make_conductance_dataset

__all__ = [
    "ConductanceDataset",
    "IncidenceConductanceAttention",
    "IsotropicConductanceAttention",
    "PositiveInvariantScalarConductance",
    "PackedGraphBatch",
    "SparseIncidenceConductanceLayer",
    "SparsePositiveConductance",
    "edge_divergence",
    "edge_gradient",
    "make_conductance_dataset",
    "pack_graph_examples",
]
````

# research/conductance_gat/cache_validation.py

````python
"""Read-only cache validators used by the repository-level dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paper_data import validate_core_cache
from .public_data import validate_public_cache

CORE_DATASETS = {
    "static_multigraph_identification",
    "topology_size_ood",
    "nonlinear_rollout",
    "identifiability_robustness",
}
PUBLIC_DATASETS = {"pascalvoc_sp", "ogbg_molhiv"}


def validate_dataset_cache(
    dataset_id: str,
    data_root: Path,
    *,
    data_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    tiny: bool,
) -> dict[str, Any]:
    """Validate every requested cache for one conductance registry entry."""

    del split_seeds
    paths: list[str] = []
    if dataset_id in CORE_DATASETS:
        for seed in data_seeds:
            _, manifest_path, _ = validate_core_cache(data_root, seed=seed, tiny=tiny)
            paths.append(str(manifest_path))
    elif dataset_id in PUBLIC_DATASETS:
        validation_seeds = data_seeds if tiny else data_seeds[:1]
        for seed in validation_seeds:
            marker, _ = validate_public_cache(data_root, seed=seed, tiny=tiny)
            paths.append(str(marker))
    else:
        raise ValueError(f"unsupported conductance cache dataset {dataset_id!r}")
    return {"paths": sorted(set(paths)), "requested_data_seeds": list(data_seeds)}


__all__ = ["validate_dataset_cache"]
````

# research/conductance_gat/config.yaml

````yaml
seed: 17
# Select CUDA when the server's PyTorch build exposes it; otherwise use CPU.
device: auto
output_dir: results

data:
  num_nodes: 28
  extra_edges: 22
  num_excitations: 192
  channels: 3
  edge_feature_channels: 3
  potential_scale: 1.0
  requested_step: 0.03
  stability_margin: 0.80

model:
  hidden_channels: 48
  minimum_conductance: 0.0001
  stability_margin: 0.95
  adaptive_stability: true

training:
  train_fraction: 0.75
  epochs: 500
  learning_rate: 0.008
  weight_decay: 0.00001
  node_weight: 1.0
  flux_weight: 1.0
  conductance_weight: 0.0
  log_every: 10
````

# research/conductance_gat/datasets.yaml

````yaml
registry_version: 2
track: conductance_gat
paper_suite_complete: true
claim: Positive incidence conductance learns heterogeneous edge transport independently of cycle PE.
objective_protocol:
  headline: Train the full conductance model from observed node messages only.
  supervised_ceiling: Report full_flux_supervised separately because it reads per-edge flux labels.
  ablation: Report full_joint separately and never merge it into the headline result.
  capacity_limit: Core ablations share hidden width and optimization but not parameter budgets; core per-baseline parameter counts are not currently emitted.

datasets:
  - id: single_graph_excitation_smoke
    name: Current single-graph excitation sanity check
    tier: smoke
    status: implemented
    data_policy: generated
    source_url: generated://research.conductance_gat.synthetic
    task: Recover flux and node update on one fixed graph with known potentials.
    split: 144/48 excitation split on the same 28-node graph; not graph OOD.
    metrics: [node_update_rmse, flux_rmse, conductance_rmse, conductance_correlation]
    claim: Algebra, optimization, and artifact-pipeline certification only.
    adapter: research.conductance_gat.synthetic.make_conductance_dataset
    leakage_guard: Do not report as unseen-graph or conductance-law generalization.

  - id: static_multigraph_identification
    name: Static heterogeneous multi-graph synthetic
    tier: paper_core
    status: implemented
    data_policy: generated
    cache_glob: conductance_gat/core-*/manifest.json
    source_url: generated://research.conductance_gat.paper_data/s1-v5
    task: Learn a shared positive conductance law across independently generated graphs and excitations.
    split: Graph-ID 70/15/15 with separate seen-graph and unseen-graph excitation tests.
    metrics: [graph_macro_flux_relative_l2, graph_macro_node_message_relative_l2, graph_macro_log_conductance_rmse, graph_macro_conductance_spearman, node_message_nnls_gap]
    claim: Evaluate the shared law on independently seeded held graph IDs; canonical topology/feature/conductance uniqueness is not certified.
    adapter: research.conductance_gat.paper_data.generate_s1
    validator: research.conductance_gat.cache_validation.validate_dataset_cache
    leakage_guard: The validator checks seed-derived graph-ID separation, cardinality, tensor validity, and cache checksums, but not canonical topology or feature/conductance-content hashes; label same-evaluation node-message NNLS and per-edge flux supervision as ceilings.

  - id: topology_size_ood
    name: Conductance topology and size OOD synthetic
    tier: paper_core
    status: implemented
    data_policy: generated
    cache_glob: conductance_gat/core-*/manifest.json
    source_url: generated://research.conductance_gat.paper_data/s2-v5
    task: Apply the same conductance law to unseen graph families and larger graphs.
    split: Train n=16..32 ER-like/RGG-like generators; test n=48..96 grid/barbell/family-OOD.
    metrics: [graph_macro_flux_relative_l2, graph_macro_log_conductance_rmse]
    claim: Generalization is not edge or topology memorization.
    adapter: research.conductance_gat.paper_data.generate_s2
    validator: research.conductance_gat.cache_validation.validate_dataset_cache
    leakage_guard: Freeze generator seeds and graph IDs in the split manifest; exact cross-split isomorphism deduplication is not implemented.

  - id: nonlinear_rollout
    name: Positive state-dependent nonlinear diffusion rollout
    tier: paper_core
    status: implemented
    data_policy: generated
    cache_glob: conductance_gat/core-*/manifest.json
    source_url: generated://research.conductance_gat.paper_data/s3-v5
    task: Learn c_e=f(x_e,abs(BH)) and predict trajectories on held-out graphs.
    split: One trajectory is generated per graph; the graph-ID split therefore makes trajectories disjoint, but it does not create an independent unseen-initial-condition axis; evaluate horizons 1/5/10/50.
    metrics: [horizon_relative_l2, flux_error, dissipation_violations, norm_growth, cap_activation]
    claim: Tests held-graph state-dependent rollout for one initial condition per graph, without separating unseen-graph and unseen-initial-condition effects.
    adapter: research.conductance_gat.paper_data.generate_s3
    validator: research.conductance_gat.cache_validation.validate_dataset_cache
    leakage_guard: Never split time steps from the same trajectory across train and test; do not describe trajectory identity as an independently randomized split axis.

  - id: identifiability_robustness
    name: Conductance contrast, excitation coverage, and noise factorial
    tier: paper_core
    status: implemented
    data_policy: generated
    cache_glob: conductance_gat/core-*/manifest.json
    source_url: generated://research.conductance_gat.paper_data/s4-v5
    task: Sweep known contrast 1/10/100, sparse excitation, and noise infinity/40/20 dB; contrast is supplied as an edge feature.
    split: Every split contains all 18 factor cells with independent graph and excitation seeds, so evaluation is factor-grid ID and held-graph-ID rather than unseen-contrast OOD.
    metrics: [error_vs_snr, error_vs_coverage, flux_ls_gap, node_message_nnls_gap]
    claim: Measures conditional empirical recovery and NNLS rank diagnostics on held graph IDs; it is not blind contrast identification or a formal identifiability result.
    adapter: research.conductance_gat.paper_data.generate_s4
    validator: research.conductance_gat.cache_validation.validate_dataset_cache
    leakage_guard: Label per-edge flux LS and node-message NNLS as transductive same-evaluation ceilings; only node_only full is headline; report that graph-global min/max normalization defines the true contrast law while the learned estimator is edge-local, so recovery error includes function-class mismatch.

  - id: pascalvoc_sp
    name: LRGB PascalVOC-SP
    tier: paper_core
    status: implemented
    data_policy: download
    cache_glob: conductance_gat/public/official-ready.json
    source_url: https://github.com/vijaydwivedi75/lrgb
    task: Superpixel node classification using graph edge weights.
    split: Official LRGB train/validation/test split.
    metrics: [macro_f1]
    baselines: [no_message_mlp, gcn, gat, gine, conductance_model]
    claim: Predictive utility on a standard edge-weighted node task.
    adapter: research.conductance_gat.public_data.prepare_public_data
    validator: research.conductance_gat.cache_validation.validate_dataset_cache
    leakage_guard: Use the official split and shared active node encoder/head/depth/optimizer; no-message and GCN intentionally freeze/skip the edge encoder; report active parameter counts and disclose that the custom one-layer backbone budgets are not matched or reference-tuned.

  - id: ogbg_molhiv
    name: OGB ogbg-molhiv
    tier: paper_core
    status: implemented
    data_policy: download
    cache_glob: conductance_gat/public/official-ready.json
    source_url: https://ogb.stanford.edu/docs/graphprop/
    task: Molecular graph classification with categorical atom and bond features.
    split: Official scaffold split.
    metrics: [roc_auc]
    baselines: [no_message_mlp, gcn, gat, gine, conductance_model]
    claim: Graph-level utility on unseen molecular scaffolds.
    adapter: research.conductance_gat.public_data.prepare_public_data
    validator: research.conductance_gat.cache_validation.validate_dataset_cache
    leakage_guard: Use the official split and AtomEncoder/BondEncoder where active, deduplicate bidirectional physical edges, report active parameter counts, and disclose that the custom one-layer backbone budgets are not matched or reference-tuned.

  - id: pglib_dc
    name: PGLib/MATPOWER DC transport proxy
    tier: conditional
    status: blocked
    data_policy: manual
    source_url: https://github.com/power-grid-lib/pglib-opf
    task: Recover branch transport on held-out grid topologies and operating points.
    split: Split by case/topology; scenario-only random split cannot support topology generalization.
    metrics: [line_flow_rmse, node_injection_rmse, balance_residual]
    claim: Conditional physical proxy for B^T C B transport, not real sensor validation.
    adapter: Requires MATPOWER case parser, DC solver, sparse incidence, and phase-shift handling.
    leakage_guard: Preserve case version/license and compare with the analytic DC oracle.

  - id: roman_empire_boundary
    name: Roman-empire positive-diffusion boundary test
    tier: optional
    status: planned
    data_policy: download
    source_url: https://github.com/yandex-research/heterophilous-graphs
    task: Heterophilous node classification.
    split: All ten official splits.
    metrics: [accuracy_mean, accuracy_std]
    claim: Negative-control boundary for strictly positive diffusion.
    adapter: planned sparse node-classification adapter
    leakage_guard: Treat a failure as a model limitation, not a dataset to tune away.
````

# research/conductance_gat/model.py

````python
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
````

# research/conductance_gat/paper.py

````python
"""Linux/CUDA paper runner for the independent conductance-GAT track.

Examples
--------
python -m research.conductance_gat.paper --suite core --data-root ./data \
    --output-dir ./results/conductance --device cuda --seed 17
python -m research.conductance_gat.paper --suite all --tiny --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import platform
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader

from chartgat.seeds import SeedAxes, resolve_seed_axes

from .paper_data import nonlinear_conductance, prepare_core_cache
from .public_data import prepare_public_data
from .sparse import (
    PackedGraphBatch,
    SparseIncidenceConductanceLayer,
    edge_gradient,
    pack_graph_examples,
)

CORE_CLAIMS = {
    "s1": "Static shared conductance law generalizes to held-out graph identities.",
    "s2": "The law transfers from ER/RGG n=16..32 to larger grid/barbell graphs.",
    "s3": "State-dependent positive conductance supports stable held-graph rollout.",
    "s4": "Identification limits are mapped across contrast, excitation coverage, and SNR.",
}
TRAINING_OBJECTIVES = {"node_only", "flux_only", "joint"}


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {normalized!r} was requested but this PyTorch build cannot use CUDA"
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def runtime_metadata(
    device: torch.device, *, amp: bool, pin_memory: bool, batch_size: int
) -> dict[str, Any]:
    cuda = device.type == "cuda"
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "amp": bool(amp),
        "pin_memory": bool(pin_memory),
        "batch_size": int(batch_size),
        "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
    }
    if cuda:
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_total_memory_bytes": int(properties.total_memory),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    else:
        metadata.update({"cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0})
    return metadata


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(
    examples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=pack_graph_examples,
    )


def _normalized_loss(
    model: SparseIncidenceConductanceLayer,
    batch: PackedGraphBatch,
    *,
    objective: str,
) -> tuple[Tensor, dict[str, float | None]]:
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"unknown training objective {objective!r}")
    _, diagnostics = model(batch, return_diagnostics=True)
    flux_target = None
    if objective in {"flux_only", "joint"}:
        flux_target = batch.observed_flux if batch.observed_flux is not None else batch.true_flux
    node_target = None
    if objective in {"node_only", "joint"}:
        node_target = (
            batch.observed_node_message
            if batch.observed_node_message is not None
            else batch.true_node_message
        )
    epsilon = torch.finfo(diagnostics["edge_flux"].dtype).eps
    flux_relative = None
    if flux_target is not None:
        flux_mse = (diagnostics["edge_flux"] - flux_target).square().mean()
        flux_scale = flux_target.square().mean().clamp_min(epsilon)
        flux_relative = flux_mse / flux_scale
    node_relative = None
    if node_target is not None:
        node_mse = (diagnostics["node_message"] - node_target).square().mean()
        node_scale = node_target.square().mean().clamp_min(epsilon)
        node_relative = node_mse / node_scale
    if objective == "node_only":
        if node_relative is None:
            raise ValueError("node_only training requires a node-message target")
        loss = node_relative
    elif objective == "flux_only":
        if flux_relative is None:
            raise ValueError("flux_only training requires an edge-flux target")
        loss = flux_relative
    else:
        if flux_relative is None or node_relative is None:
            raise ValueError("joint training requires edge-flux and node-message targets")
        loss = flux_relative + node_relative
    return loss, {
        "loss": float(loss.detach().float().cpu()),
        "flux_relative_mse": (
            None if flux_relative is None else float(flux_relative.detach().float().cpu())
        ),
        "node_relative_mse": (
            None if node_relative is None else float(node_relative.detach().float().cpu())
        ),
    }


@torch.no_grad()
def _validation_loss(
    model: SparseIncidenceConductanceLayer,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    objective: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    loader = _loader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            loss, _ = _normalized_loss(model, batch, objective=objective)
        total += float(loss.float().cpu()) * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


def train_sparse_model(
    model: SparseIncidenceConductanceLayer,
    train_examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
    objective: str,
) -> list[dict[str, Any]]:
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"unknown training objective {objective!r}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    scaler = _grad_scaler(amp)
    best_validation = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        loader = _loader(
            train_examples,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        for batch in loader:
            batch = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, _ = _normalized_loss(model, batch, objective=objective)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().float().cpu()) * batch.num_graphs
            count += batch.num_graphs
        validation = _validation_loss(
            model,
            validation_examples,
            device=device,
            amp=amp,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            objective=objective,
        )
        train_loss = total / max(count, 1)
        history.append(
            {
                "epoch": epoch,
                "training_objective": objective,
                "train_loss": train_loss,
                "validation_loss": validation,
            }
        )
        if validation < best_validation:
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _pearson(first: Tensor, second: Tensor) -> float | None:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    if first.numel() < 2:
        return None
    first = first - first.mean()
    second = second - second.mean()
    first_tolerance = 1.0e-7 * max(float(first.abs().max()), 1.0)
    second_tolerance = 1.0e-7 * max(float(second.abs().max()), 1.0)
    if float(first.norm()) <= first_tolerance or float(second.norm()) <= second_tolerance:
        return None
    denominator = first.norm() * second.norm()
    if float(denominator) <= torch.finfo(torch.float32).eps:
        return None
    return float(torch.dot(first, second) / denominator)


def _rank(values: Tensor) -> Tensor:
    # Synthetic conductances are continuous; ties are vanishingly rare.  The
    # deterministic stable ordering is sufficient for this diagnostic.
    order = torch.argsort(values.reshape(-1), stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
    return ranks


def _mean(values: Iterable[float | None]) -> float | None:
    selected = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    return sum(selected) / len(selected) if selected else None


@torch.no_grad()
def evaluate_sparse_model(
    model: SparseIncidenceConductanceLayer,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    oracle: bool = False,
) -> dict[str, Any]:
    model.eval()
    flux_rel: list[float] = []
    node_rel: list[float] = []
    next_rel: list[float] = []
    log_c_rmse: list[float] = []
    correlations: list[float | None] = []
    rank_correlations: list[float | None] = []
    coverage: list[float] = []
    cap_active = 0
    cap_total = 0
    predictions_by_graph: dict[str, list[Tensor]] = {}
    loader = _loader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=pin_memory)
        override = batch.true_conductance if oracle else None
        with _autocast(device, amp):
            predicted_next, diagnostics = model(
                batch, conductance_override=override, return_diagnostics=True
            )
        for graph_number, graph_id in enumerate(batch.graph_ids):
            edge_mask = batch.edge_graph == graph_number
            node_mask = batch.node_graph == graph_number
            predicted_flux = diagnostics["edge_flux"][edge_mask].float()
            predicted_c = diagnostics["conductance"][edge_mask].float()
            true_flux = batch.true_flux[edge_mask].float()
            true_c = batch.true_conductance[edge_mask].float()
            true_message = batch.true_node_message[node_mask].float()
            predicted_message = diagnostics["node_message"][node_mask].float()
            true_next = batch.true_next_state[node_mask].float()
            current_next = predicted_next[node_mask].float()
            epsilon = torch.finfo(torch.float32).eps
            flux_rel.append(
                float((predicted_flux - true_flux).norm() / true_flux.norm().clamp_min(epsilon))
            )
            node_rel.append(
                float(
                    (predicted_message - true_message).norm()
                    / true_message.norm().clamp_min(epsilon)
                )
            )
            next_rel.append(
                float((current_next - true_next).norm() / true_next.norm().clamp_min(epsilon))
            )
            log_c_rmse.append(
                float(
                    torch.mean(
                        (predicted_c.clamp_min(1e-8).log() - true_c.clamp_min(1e-8).log()).square()
                    ).sqrt()
                )
            )
            correlation = _pearson(predicted_c.cpu(), true_c.cpu())
            correlations.append(correlation)
            rank_correlations.append(
                None
                if correlation is None
                else _pearson(_rank(predicted_c.cpu()), _rank(true_c.cpu()))
            )
            gradient = batch.true_gradient[edge_mask]
            coverage.append(float((gradient.abs().amax(dim=1) > 1.0e-6).float().mean()))
            predictions_by_graph.setdefault(graph_id, []).append(predicted_c.detach().cpu())
        cap_active += int(diagnostics["cap_active"].sum())
        cap_total += int(diagnostics["cap_active"].numel())
    state_variation = []
    for values in predictions_by_graph.values():
        if len(values) > 1 and all(value.shape == values[0].shape for value in values):
            state_variation.append(float(torch.stack(values).std(dim=0, unbiased=False).mean()))
    return {
        "graph_macro_flux_relative_l2": _mean(flux_rel),
        "graph_macro_node_message_relative_l2": _mean(node_rel),
        "graph_macro_next_state_relative_l2": _mean(next_rel),
        "graph_macro_log_conductance_rmse": _mean(log_c_rmse),
        "graph_macro_conductance_pearson": _mean(correlations),
        "conductance_pearson_defined_fraction": sum(value is not None for value in correlations)
        / max(len(correlations), 1),
        "graph_macro_conductance_spearman": _mean(rank_correlations),
        "excited_edge_fraction": _mean(coverage),
        "mean_conductance_state_variation": _mean(state_variation),
        "stability_cap_activation_fraction": cap_active / max(cap_total, 1),
        "num_examples": len(examples),
        "num_graph_ids": len({str(example["graph_id"]) for example in examples}),
    }


def least_squares_metrics(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-graph flux LS using the evaluated excitations (diagnostic ceiling)."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example["graph_id"]), []).append(example)
    flux_errors: list[float] = []
    log_errors: list[float] = []
    correlations: list[float | None] = []
    identifiable: list[float] = []
    for group in groups.values():
        numerator = None
        denominator = None
        for example in group:
            gradient = example["true_gradient"].float()
            observed = example.get("observed_flux", example["true_flux"]).float()
            current_numerator = (gradient * observed).sum(dim=1)
            current_denominator = gradient.square().sum(dim=1)
            numerator = current_numerator if numerator is None else numerator + current_numerator
            denominator = (
                current_denominator if denominator is None else denominator + current_denominator
            )
        assert numerator is not None and denominator is not None
        estimated = (numerator / denominator.clamp_min(1.0e-12)).clamp_min(1.0e-6)
        truth = group[0]["true_conductance"].float()
        excited = denominator > 1.0e-10
        identifiable.append(float(excited.float().mean()))
        if excited.any():
            log_errors.append(
                float(((estimated[excited].log() - truth[excited].log()).square().mean()).sqrt())
            )
            correlations.append(_pearson(estimated[excited], truth[excited]))
        for example in group:
            gradient = example["true_gradient"].float()
            truth_flux = example["true_flux"].float()
            predicted_flux = estimated[:, None] * gradient
            flux_errors.append(
                float((predicted_flux - truth_flux).norm() / truth_flux.norm().clamp_min(1.0e-12))
            )
    return {
        "protocol": "transductive_same-evaluation-excitations_identification_ceiling",
        "graph_macro_flux_relative_l2": _mean(flux_errors),
        "graph_macro_log_conductance_rmse": _mean(log_errors),
        "graph_macro_conductance_pearson": _mean(correlations),
        "identifiable_edge_fraction": _mean(identifiable),
        "num_graph_ids": len(groups),
    }


def _node_message_design(example: Mapping[str, Any]) -> Tensor:
    """Dense diagnostic design only; the learned layer remains gather/scatter sparse."""

    edge_index = example["edge_index"].long().cpu()
    gradient = example["true_gradient"].double().cpu()
    num_nodes = int(example["node_state"].shape[0])
    channels = int(gradient.shape[1])
    num_edges = int(edge_index.shape[1])
    design = gradient.new_zeros((num_nodes * channels, num_edges))
    edge_ids = torch.arange(num_edges).view(-1, 1).expand(-1, channels)
    channel_ids = torch.arange(channels).view(1, -1)
    tail_rows = edge_index[0].view(-1, 1) * channels + channel_ids
    head_rows = edge_index[1].view(-1, 1) * channels + channel_ids
    design[tail_rows.reshape(-1), edge_ids.reshape(-1)] = -gradient.reshape(-1)
    design[head_rows.reshape(-1), edge_ids.reshape(-1)] = gradient.reshape(-1)
    return design


def _projected_nnls(
    design: Tensor,
    target: Tensor,
    *,
    max_iterations: int = 1_000,
    tolerance: float = 1.0e-10,
) -> tuple[Tensor, int]:
    """Solve nonnegative least squares with deterministic projected FISTA."""

    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise ValueError("NNLS design and target shapes are inconsistent")
    if design.shape[1] == 0:
        return design.new_empty(0), 0
    spectral = torch.linalg.svdvals(design)
    lipschitz = spectral[0].square() if spectral.numel() else design.new_tensor(0.0)
    if float(lipschitz) <= torch.finfo(design.dtype).eps:
        return design.new_zeros(design.shape[1]), 0
    # The unconstrained solution is already the exact NNLS solution when it is
    # nonnegative.  This makes the noiseless, full-rank ceiling numerically sharp.
    unconstrained = torch.linalg.lstsq(design, target).solution
    if bool(torch.all(unconstrained >= 0)):
        return unconstrained, 0
    estimate = unconstrained.clamp_min(0)
    accelerated = estimate.clone()
    momentum = 1.0
    scale = max(float(estimate.norm()), 1.0)
    for iteration in range(1, max_iterations + 1):
        gradient = design.mT @ (design @ accelerated - target)
        updated = (accelerated - gradient / lipschitz).clamp_min(0)
        if float((updated - estimate).norm()) <= tolerance * scale:
            return updated, iteration
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - estimate)
        estimate = updated
        momentum = next_momentum
        scale = max(float(estimate.norm()), 1.0)
    return estimate, max_iterations


def node_message_nnls_metrics(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Same-evaluation node-output NNLS ceiling for one conductance per edge.

    Unlike :func:`least_squares_metrics`, this diagnostic never reads observed
    per-edge flux.  It estimates nonnegative edge conductances only from the
    observed node messages and the known excitation gradients.  It is still a
    transductive ceiling, not a held-graph predictive baseline.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example["graph_id"]), []).append(example)
    clean_errors: list[float] = []
    observed_fit_errors: list[float] = []
    log_errors: list[float] = []
    correlations: list[float | None] = []
    excited_fractions: list[float] = []
    rank_fractions: list[float] = []
    iterations: list[float] = []
    for group in groups.values():
        reference_edges = group[0]["edge_index"]
        reference_truth = group[0]["true_conductance"].double().cpu()
        designs: list[Tensor] = []
        observed_targets: list[Tensor] = []
        for example in group:
            if not torch.equal(example["edge_index"], reference_edges):
                raise ValueError("examples sharing graph_id must share edge_index")
            truth = example["true_conductance"].double().cpu()
            if not torch.allclose(truth, reference_truth):
                raise ValueError("node-message NNLS requires static conductance per graph_id")
            design = _node_message_design(example)
            observed = example.get("observed_node_message")
            if observed is None:
                observed = example["true_node_message"]
            designs.append(design)
            observed_targets.append(observed.double().cpu().reshape(-1))
        stacked_design = torch.cat(designs, dim=0)
        stacked_target = torch.cat(observed_targets, dim=0)
        estimated, used_iterations = _projected_nnls(stacked_design, stacked_target)
        iterations.append(float(used_iterations))
        column_energy = stacked_design.square().sum(dim=0)
        excited = column_energy > 1.0e-12
        excited_fractions.append(float(excited.float().mean()))
        rank = int(torch.linalg.matrix_rank(stacked_design))
        rank_fractions.append(rank / max(stacked_design.shape[1], 1))
        if excited.any():
            log_errors.append(
                float(
                    (
                        estimated[excited].clamp_min(1.0e-12).log()
                        - reference_truth[excited].clamp_min(1.0e-12).log()
                    )
                    .square()
                    .mean()
                    .sqrt()
                )
            )
            correlations.append(_pearson(estimated[excited], reference_truth[excited]))
        for example, design, observed_target in zip(group, designs, observed_targets, strict=True):
            predicted = design @ estimated
            clean_target = example["true_node_message"].double().cpu().reshape(-1)
            clean_errors.append(
                float((predicted - clean_target).norm() / clean_target.norm().clamp_min(1.0e-12))
            )
            observed_fit_errors.append(
                float(
                    (predicted - observed_target).norm() / observed_target.norm().clamp_min(1.0e-12)
                )
            )
    return {
        "protocol": "transductive_same-evaluation-node-messages_nnls_ceiling",
        "graph_macro_node_message_relative_l2": _mean(clean_errors),
        "graph_macro_observed_fit_relative_l2": _mean(observed_fit_errors),
        "graph_macro_log_conductance_rmse": _mean(log_errors),
        "graph_macro_conductance_pearson": _mean(correlations),
        "excited_edge_fraction": _mean(excited_fractions),
        "design_rank_fraction": _mean(rank_fractions),
        "mean_solver_iterations": _mean(iterations),
        "num_graph_ids": len(groups),
    }


@torch.no_grad()
def evaluate_rollout(
    model: SparseIncidenceConductanceLayer,
    trajectories: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    device: torch.device,
    amp: bool,
    oracle: bool,
) -> dict[str, Any]:
    errors: dict[int, list[float]] = {int(horizon): [] for horizon in horizons}
    growth: list[float] = []
    dissipation_violations = 0
    steps_total = 0
    cap_active = 0
    for trajectory in trajectories:
        state = trajectory["states"][0].to(device)
        initial_norm = float(state.norm())
        previous_norm = initial_norm
        edge_index = trajectory["edge_index"].to(device)
        edge_features = trajectory["edge_features"].to(device)
        for time_index in range(max(horizons)):
            record = {
                "graph_id": trajectory["graph_id"],
                "node_state": state,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "step_size": float(trajectory["steps"][time_index]),
            }
            batch = pack_graph_examples([record]).to(device)
            override = None
            if oracle:
                override = nonlinear_conductance(edge_features, edge_gradient(edge_index, state))
            with _autocast(device, amp):
                state, diagnostics = model(
                    batch,
                    node_state=state,
                    conductance_override=override,
                    return_diagnostics=True,
                )
            current_norm = float(state.float().norm())
            dissipation_violations += int(current_norm > previous_norm + 1.0e-6)
            previous_norm = current_norm
            steps_total += 1
            cap_active += int(diagnostics["cap_active"].sum())
            horizon = time_index + 1
            if horizon in errors:
                truth = trajectory["states"][horizon].to(device)
                errors[horizon].append(
                    float((state.float() - truth).norm() / truth.norm().clamp_min(1e-12))
                )
        growth.append(previous_norm / max(initial_norm, 1.0e-12))
    result = {f"horizon_{horizon}_relative_l2": _mean(values) for horizon, values in errors.items()}
    result.update(
        {
            "final_norm_over_initial": _mean(growth),
            "dissipation_violation_fraction": dissipation_violations / max(steps_total, 1),
            "stability_cap_activation_fraction": cap_active / max(steps_total, 1),
        }
    )
    return result


def _model_for_examples(
    examples: Sequence[Mapping[str, Any]], mode: str, *, hidden_channels: int
) -> SparseIncidenceConductanceLayer:
    first = examples[0]
    return SparseIncidenceConductanceLayer(
        channels=int(first["node_state"].shape[1]),
        edge_feature_channels=int(first["edge_features"].shape[1]),
        hidden_channels=hidden_channels,
        requested_step=0.025,
        stability_margin=0.95,
        adaptive_stability=True,
        mode=mode,
    )


def _factorial_key(example: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    metadata = example["metadata"]
    return metadata["contrast"], metadata["active_node_fraction"], metadata["snr_db"]


def run_core(
    core: dict[str, Any],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Tensor]]]:
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    states: dict[str, dict[str, Tensor]] = {}
    baseline_specs = (
        ("isotropic", "isotropic", "node_only", "constant-conductance ablation"),
        ("edge_only", "edge_only", "node_only", "static edge-feature ablation"),
        (
            "gradient_only",
            "gradient_only",
            "node_only",
            "state-gradient-only ablation C=f(abs(BH))",
        ),
        (
            "full",
            "full",
            "node_only",
            "headline node-output-only predictive model",
        ),
        (
            "full_flux_supervised",
            "full",
            "flux_only",
            "per-edge-flux-supervised neural ceiling",
        ),
        ("full_joint", "full", "joint", "joint-supervision objective ablation"),
    )
    mode_seed_offset = {"isotropic": 0, "edge_only": 1, "gradient_only": 2, "full": 3}
    for suite_number, suite_name in enumerate(("s1", "s2", "s3", "s4")):
        suite = core[suite_name]
        train_examples = suite["train"]
        validation_examples = suite["validation"]
        test_examples = suite["test"]
        hidden_channels = 16 if core["tiny"] else 64
        suite_result: dict[str, Any] = {
            "claim": CORE_CLAIMS[suite_name],
            "description": suite["description"],
            "split_graph_counts": {
                split: len({item["graph_id"] for item in suite.get(split, [])})
                for split in ("train", "validation", "test", "seen_test")
                if split in suite
            },
            "headline_baseline": "full",
            "objective_protocol": {
                "headline": "node_only",
                "flux_supervised_ceiling": "full_flux_supervised",
                "joint_objective_ablation": "full_joint",
            },
            "baselines": {},
        }
        trained: dict[str, tuple[SparseIncidenceConductanceLayer, str]] = {}
        for baseline_name, mode, objective, role in baseline_specs:
            initialization_offset = mode_seed_offset[mode]
            seed_everything(seed + suite_number * 100 + initialization_offset)
            model = _model_for_examples(train_examples, mode, hidden_channels=hidden_channels).to(
                device
            )
            history = train_sparse_model(
                model,
                train_examples,
                validation_examples,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=num_workers,
                seed=seed + suite_number * 1000 + initialization_offset * 100,
                objective=objective,
            )
            for row in history:
                histories.append({"suite": suite_name, "baseline": baseline_name, **row})
            trained[baseline_name] = (model, objective)
            states[f"{suite_name}_{baseline_name}"] = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            metric = evaluate_sparse_model(
                model,
                test_examples,
                device=device,
                amp=amp,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            suite_result["baselines"][baseline_name] = {
                "training_objective": objective,
                "role": role,
                "unseen_graph_test": metric,
            }
            if suite_name == "s1":
                suite_result["baselines"][baseline_name]["seen_graph_new_excitation_test"] = (
                    evaluate_sparse_model(
                        model,
                        suite["seen_test"],
                        device=device,
                        amp=amp,
                        batch_size=batch_size,
                        pin_memory=pin_memory,
                        num_workers=num_workers,
                    )
                )
            if suite_name == "s3":
                suite_result["baselines"][baseline_name]["rollout"] = evaluate_rollout(
                    model,
                    suite["rollout_test"],
                    suite["horizons"],
                    device=device,
                    amp=amp,
                    oracle=False,
                )
        oracle_model = _model_for_examples(
            train_examples, "full", hidden_channels=hidden_channels
        ).to(device)
        suite_result["baselines"]["oracle"] = {
            "training_objective": "analytic_oracle",
            "role": "ground-truth conductance oracle",
            "unseen_graph_test": evaluate_sparse_model(
                oracle_model,
                test_examples,
                device=device,
                amp=amp,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                oracle=True,
            ),
        }
        if suite_name == "s1":
            suite_result["baselines"]["oracle"]["seen_graph_new_excitation_test"] = (
                evaluate_sparse_model(
                    oracle_model,
                    suite["seen_test"],
                    device=device,
                    amp=amp,
                    batch_size=batch_size,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    oracle=True,
                )
            )
        if suite_name == "s3":
            suite_result["baselines"]["oracle"]["rollout"] = evaluate_rollout(
                oracle_model,
                suite["rollout_test"],
                suite["horizons"],
                device=device,
                amp=amp,
                oracle=True,
            )
        if suite_name in {"s1", "s4"}:
            suite_result["baselines"]["flux_ls"] = {
                "training_objective": "same-evaluation observed edge flux",
                "role": "transductive per-edge-flux least-squares ceiling",
                "unseen_graph_test": least_squares_metrics(test_examples),
            }
            suite_result["baselines"]["node_message_nnls"] = {
                "training_objective": "same-evaluation observed node message",
                "role": "transductive node-output nonnegative least-squares ceiling",
                "unseen_graph_test": node_message_nnls_metrics(test_examples),
            }
            if suite_name == "s1":
                suite_result["baselines"]["flux_ls"]["seen_graph_new_excitation_test"] = (
                    least_squares_metrics(suite["seen_test"])
                )
                suite_result["baselines"]["node_message_nnls"]["seen_graph_new_excitation_test"] = (
                    node_message_nnls_metrics(suite["seen_test"])
                )
        if suite_name == "s4":
            factorial: list[dict[str, Any]] = []
            keys = sorted({_factorial_key(example) for example in test_examples}, key=str)
            for key in keys:
                subset = [example for example in test_examples if _factorial_key(example) == key]
                for baseline_name, (model, objective) in trained.items():
                    factorial.append(
                        {
                            "contrast": key[0],
                            "active_node_fraction": key[1],
                            "snr_db": key[2],
                            "baseline": baseline_name,
                            "training_objective": objective,
                            **evaluate_sparse_model(
                                model,
                                subset,
                                device=device,
                                amp=amp,
                                batch_size=batch_size,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                            ),
                        }
                    )
                factorial.append(
                    {
                        "contrast": key[0],
                        "active_node_fraction": key[1],
                        "snr_db": key[2],
                        "baseline": "flux_ls",
                        "training_objective": "same-evaluation observed edge flux",
                        **least_squares_metrics(subset),
                    }
                )
                factorial.append(
                    {
                        "contrast": key[0],
                        "active_node_fraction": key[1],
                        "snr_db": key[2],
                        "baseline": "node_message_nnls",
                        "training_objective": "same-evaluation observed node message",
                        **node_message_nnls_metrics(subset),
                    }
                )
            suite_result["factorial"] = factorial
        results[suite_name] = suite_result
    return results, histories, states


@dataclass
class PublicPacked:
    x: Tensor
    edge_index: Tensor
    edge_features: Tensor
    node_graph: Tensor
    y: Tensor
    graph_ids: list[str]
    task: str
    categorical: bool

    @property
    def num_graphs(self) -> int:
        return len(self.graph_ids)

    def to(self, device: torch.device, *, non_blocking: bool) -> PublicPacked:
        return PublicPacked(
            x=self.x.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            edge_features=self.edge_features.to(device, non_blocking=non_blocking),
            node_graph=self.node_graph.to(device, non_blocking=non_blocking),
            y=self.y.to(device, non_blocking=non_blocking),
            graph_ids=self.graph_ids,
            task=self.task,
            categorical=self.categorical,
        )

    def pin_memory(self) -> PublicPacked:
        return PublicPacked(
            x=self.x.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            edge_features=self.edge_features.pin_memory(),
            node_graph=self.node_graph.pin_memory(),
            y=self.y.pin_memory(),
            graph_ids=self.graph_ids,
            task=self.task,
            categorical=self.categorical,
        )


def pack_public(records: Sequence[Mapping[str, Any]]) -> PublicPacked:
    if not records:
        raise ValueError("empty public batch")
    task = str(records[0]["task"])
    categorical = bool(records[0]["categorical"])
    nodes: list[Tensor] = []
    edges: list[Tensor] = []
    edge_features: list[Tensor] = []
    node_graph: list[Tensor] = []
    labels: list[Tensor] = []
    graph_ids: list[str] = []
    offset = 0
    for graph_number, record in enumerate(records):
        if record["task"] != task or bool(record["categorical"]) != categorical:
            raise ValueError("public batch mixes tasks or feature types")
        x = record["x"]
        nodes.append(x)
        edges.append(record["edge_index"] + offset)
        edge_features.append(record["edge_features"])
        node_graph.append(torch.full((x.shape[0],), graph_number, dtype=torch.long))
        labels.append(record["y"])
        graph_ids.append(str(record["graph_id"]))
        offset += int(x.shape[0])
    y = (
        torch.cat(labels)
        if task == "node"
        else torch.stack([label.reshape(-1) for label in labels])
    )
    return PublicPacked(
        x=torch.cat(nodes),
        edge_index=torch.cat(edges, dim=1),
        edge_features=torch.cat(edge_features),
        node_graph=torch.cat(node_graph),
        y=y,
        graph_ids=graph_ids,
        task=task,
        categorical=categorical,
    )


class SumCategoricalEncoder(nn.Module):
    def __init__(self, columns: int, hidden: int, categories: int = 256) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, hidden) for _ in range(columns)])

    def forward(self, values: Tensor) -> Tensor:
        result = self.embeddings[0](values[:, 0].long())
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            result = result + embedding(values[:, column].long())
        return result


def _bidirectional_edges(
    edge_index: Tensor, edge_features: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    tail, head = edge_index
    source = torch.cat((tail, head))
    target = torch.cat((head, tail))
    return source, target, torch.cat((edge_features, edge_features), dim=0)


class NoMessageMLPLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, node_state: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        del edge_index, edge_features
        return self.network(node_state)


class SparseGCNLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, node_state: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        del edge_features
        source, target, _ = _bidirectional_edges(
            edge_index, node_state.new_empty((edge_index.shape[1], 0))
        )
        nodes = torch.arange(node_state.shape[0], device=node_state.device)
        source = torch.cat((source, nodes))
        target = torch.cat((target, nodes))
        degree = torch.bincount(target, minlength=node_state.shape[0]).to(node_state)
        weight = degree.index_select(0, source).rsqrt() * degree.index_select(0, target).rsqrt()
        aggregate = torch.zeros_like(node_state)
        aggregate.index_add_(0, target, weight[:, None] * node_state.index_select(0, source))
        return self.linear(aggregate)


class SparseGATLayer(nn.Module):
    """Single-head edge-aware GAT with a torch-only segment softmax."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(hidden, hidden, bias=False)
        self.edge_projection = nn.Linear(hidden, hidden, bias=False)
        self.attention = nn.Linear(3 * hidden, 1, bias=False)
        self.self_projection = nn.Linear(hidden, hidden)

    def forward(self, node_state: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        source, target, directed_features = _bidirectional_edges(edge_index, edge_features)
        projected = self.node_projection(node_state)
        source_state = projected.index_select(0, source)
        target_state = projected.index_select(0, target)
        edge_state = self.edge_projection(directed_features)
        logits = nnf.leaky_relu(
            self.attention(torch.cat((source_state, target_state, edge_state), dim=1)).squeeze(-1),
            negative_slope=0.2,
        )
        maximum = logits.new_full((node_state.shape[0],), -torch.inf)
        maximum.scatter_reduce_(0, target, logits, reduce="amax", include_self=True)
        unnormalized = torch.exp(logits - maximum.index_select(0, target))
        denominator = logits.new_zeros(node_state.shape[0])
        denominator.index_add_(0, target, unnormalized)
        attention = unnormalized / denominator.index_select(0, target).clamp_min(1.0e-12)
        message = attention[:, None] * (source_state + edge_state)
        aggregate = torch.zeros_like(projected)
        aggregate.index_add_(0, target, message)
        return self.self_projection(node_state) + aggregate


class SparseGINELayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.edge_projection = nn.Linear(hidden, hidden)
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.network = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, node_state: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        source, target, directed_features = _bidirectional_edges(edge_index, edge_features)
        message = nnf.relu(
            node_state.index_select(0, source) + self.edge_projection(directed_features)
        )
        aggregate = torch.zeros_like(node_state)
        aggregate.index_add_(0, target, message)
        return self.network((1.0 + self.epsilon) * node_state + aggregate)


class PublicConductanceModel(nn.Module):
    def __init__(
        self,
        sample: Mapping[str, Any],
        *,
        hidden: int,
        num_classes: int,
        official_molecule: bool,
        backbone: str = "conductance_model",
    ) -> None:
        super().__init__()
        if backbone not in {"no_message_mlp", "gcn", "gat", "gine", "conductance_model"}:
            raise ValueError(f"unknown public backbone {backbone!r}")
        node_width = int(sample["x"].shape[1])
        edge_width = int(sample["edge_features"].shape[1])
        self.task = str(sample["task"])
        self.backbone_name = backbone
        if bool(sample["categorical"]) and official_molecule:
            try:
                from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
            except (ImportError, OSError) as error:  # pragma: no cover - optional path
                raise RuntimeError(
                    "official MolHIV requires OGB AtomEncoder/BondEncoder"
                ) from error
            self.node_encoder = AtomEncoder(hidden)
            self.edge_encoder = BondEncoder(hidden)
        elif bool(sample["categorical"]):
            self.node_encoder = SumCategoricalEncoder(node_width, hidden)
            self.edge_encoder = SumCategoricalEncoder(edge_width, hidden)
        else:
            self.node_encoder = nn.Linear(node_width, hidden)
            self.edge_encoder = nn.Linear(edge_width, hidden)
        self.uses_edge_features = backbone not in {"no_message_mlp", "gcn"}
        if not self.uses_edge_features:
            self.edge_encoder.requires_grad_(False)
        # Construct shared components before the backbone so resetting the same
        # seed gives every comparison identical encoder/head initialization.
        self.normalization = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes if self.task == "node" else 1)
        if backbone == "conductance_model":
            self.layer: nn.Module = SparseIncidenceConductanceLayer(
                channels=hidden,
                edge_feature_channels=hidden,
                hidden_channels=hidden,
                requested_step=0.02,
                mode="full",
            )
        elif backbone == "no_message_mlp":
            self.layer = NoMessageMLPLayer(hidden)
        elif backbone == "gcn":
            self.layer = SparseGCNLayer(hidden)
        elif backbone == "gat":
            self.layer = SparseGATLayer(hidden)
        else:
            self.layer = SparseGINELayer(hidden)

    def forward(self, batch: PublicPacked) -> Tensor:
        node_state = self.node_encoder(batch.x)
        edge_features = (
            self.edge_encoder(batch.edge_features)
            if self.uses_edge_features
            else node_state.new_empty((batch.edge_index.shape[1], 0))
        )
        if self.backbone_name == "conductance_model":
            edge_graph = batch.node_graph.index_select(0, batch.edge_index[0])
            sparse_batch = PackedGraphBatch(
                node_state=node_state,
                edge_index=batch.edge_index,
                edge_features=edge_features,
                node_graph=batch.node_graph,
                edge_graph=edge_graph,
                graph_ids=batch.graph_ids,
                requested_step=node_state.new_full((batch.num_graphs,), 0.02),
            )
            node_state = self.layer(sparse_batch)
        else:
            node_state = self.layer(node_state, batch.edge_index, edge_features)
        node_state = nnf.silu(self.normalization(node_state))
        if self.task == "node":
            return self.head(node_state)
        pooled = node_state.new_zeros((batch.num_graphs, node_state.shape[1]))
        pooled.index_add_(0, batch.node_graph, node_state)
        counts = torch.bincount(batch.node_graph, minlength=batch.num_graphs).to(node_state)
        return self.head(pooled / counts[:, None].clamp_min(1)).squeeze(-1)


def _public_loader(
    dataset: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=pack_public,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _public_loss(logits: Tensor, labels: Tensor, task: str) -> Tensor:
    if task == "node":
        return nnf.cross_entropy(logits, labels.long())
    valid = torch.isfinite(labels.reshape(-1))
    return nnf.binary_cross_entropy_with_logits(logits[valid], labels.reshape(-1)[valid].float())


def _public_loss_weight(labels: Tensor, task: str) -> int:
    """Return the number of labels represented by a mean-reduced task loss."""

    if task == "node":
        return int(labels.numel())
    return int(torch.isfinite(labels.reshape(-1)).sum())


def _macro_f1(predictions: Tensor, labels: Tensor) -> float:
    scores = []
    for label in torch.unique(labels):
        true_positive = ((predictions == label) & (labels == label)).sum().float()
        false_positive = ((predictions == label) & (labels != label)).sum().float()
        false_negative = ((predictions != label) & (labels == label)).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator > 0:
            scores.append(float(2 * true_positive / denominator))
    return sum(scores) / max(len(scores), 1)


def _binary_auc(scores: Tensor, labels: Tensor) -> float | None:
    valid = torch.isfinite(labels)
    scores = scores[valid].float().cpu()
    labels = labels[valid].long().cpu()
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return None
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, order.numel() + 1, dtype=torch.float32)
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


@torch.no_grad()
def evaluate_public(
    model: PublicConductanceModel,
    dataset: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    official_molecule: bool = False,
) -> dict[str, Any]:
    model.eval()
    outputs: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in _public_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        num_workers=num_workers,
    ):
        batch = batch.to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            outputs.append(model(batch).float().cpu())
        labels.append(batch.y.float().cpu())
    output = torch.cat(outputs)
    label = torch.cat(labels)
    if model.task == "node":
        return {
            "macro_f1": _macro_f1(output.argmax(dim=1), label.long()),
            "num_labels": label.numel(),
        }
    if official_molecule:
        try:
            from ogb.graphproppred import Evaluator
        except (ImportError, OSError) as error:  # pragma: no cover - optional path
            raise RuntimeError("official MolHIV evaluation requires the OGB evaluator") from error
        evaluator = Evaluator(name="ogbg-molhiv")
        score = evaluator.eval({"y_true": label.reshape(-1, 1), "y_pred": output.reshape(-1, 1)})[
            "rocauc"
        ]
        return {
            "roc_auc": float(score),
            "num_graphs": label.numel(),
            "evaluator": "ogb.graphproppred.Evaluator",
        }
    return {
        "roc_auc": _binary_auc(output.reshape(-1), label.reshape(-1)),
        "num_graphs": label.numel(),
        "evaluator": "offline_fixture_rank_auc",
    }


def run_public(
    datasets: dict[str, Any],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    amp: bool,
    pin_memory: bool,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Tensor]]]:
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    states: dict[str, dict[str, Tensor]] = {}
    backbone_names = ("no_message_mlp", "gcn", "gat", "gine", "conductance_model")
    for dataset_number, dataset_name in enumerate(("pascalvoc_sp", "ogbg_molhiv")):
        splits = datasets[dataset_name]
        sample = splits["train"][0]
        num_classes = 21 if dataset_name == "pascalvoc_sp" and not datasets["fixture"] else 3
        hidden = 24 if datasets["fixture"] else 96
        results[dataset_name] = {
            "fixture": bool(datasets["fixture"]),
            "official_result": not bool(datasets["fixture"]),
            "comparison_protocol": {
                "hidden_channels": hidden,
                "backbone_depth": 1,
                "shared_encoder_initialization": True,
                "shared_head_initialization": True,
                "shared_optimizer_and_split": True,
            },
            "baselines": {},
        }
        model_seed = seed + dataset_number * 101
        for backbone in backbone_names:
            # Shared modules are constructed before the variable backbone, so
            # this reset makes their initial tensors identical across models.
            seed_everything(model_seed)
            model = PublicConductanceModel(
                sample,
                hidden=hidden,
                num_classes=num_classes,
                official_molecule=(dataset_name == "ogbg_molhiv" and not datasets["fixture"]),
                backbone=backbone,
            ).to(device)
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scaler = _grad_scaler(amp)
            best_validation = math.inf
            best_state = None
            for epoch in range(1, epochs + 1):
                model.train()
                total = 0.0
                count = 0
                for batch in _public_loader(
                    splits["train"],
                    batch_size=batch_size,
                    shuffle=True,
                    seed=seed + epoch,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                ):
                    batch = batch.to(device, non_blocking=pin_memory)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device, amp):
                        loss = _public_loss(model(batch), batch.y, model.task)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    loss_weight = _public_loss_weight(batch.y, model.task)
                    total += float(loss.detach().float().cpu()) * loss_weight
                    count += loss_weight
                model.eval()
                validation_total = 0.0
                validation_count = 0
                with torch.no_grad():
                    for batch in _public_loader(
                        splits["validation"],
                        batch_size=batch_size,
                        shuffle=False,
                        seed=0,
                        pin_memory=pin_memory,
                        num_workers=num_workers,
                    ):
                        batch = batch.to(device, non_blocking=pin_memory)
                        with _autocast(device, amp):
                            loss = _public_loss(model(batch), batch.y, model.task)
                        loss_weight = _public_loss_weight(batch.y, model.task)
                        validation_total += float(loss.float().cpu()) * loss_weight
                        validation_count += loss_weight
                validation_loss = validation_total / max(validation_count, 1)
                histories.append(
                    {
                        "suite": dataset_name,
                        "baseline": backbone,
                        "epoch": epoch,
                        "train_loss": total / max(count, 1),
                        "validation_loss": validation_loss,
                    }
                )
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            if best_state is not None:
                model.load_state_dict(best_state)
            state_key = f"{dataset_name}_{backbone}"
            states[state_key] = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            results[dataset_name]["baselines"][backbone] = {
                "parameter_count": parameter_count,
                "parameter_count_policy": "trainable_active_parameters_only",
                "uses_edge_features": model.uses_edge_features,
                "best_validation_loss": best_validation,
                "test": evaluate_public(
                    model,
                    splits["test"],
                    device=device,
                    batch_size=batch_size,
                    amp=amp,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    official_molecule=(dataset_name == "ogbg_molhiv" and not datasets["fixture"]),
                ),
            }
    return results, histories, states


def _metric_rows(value: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_metric_rows(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_metric_rows(child, (*path, str(index))))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append({"path": "/".join(path), "value": value})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output_dir(path: Path) -> Path:
    """Claim an empty run directory before data preparation or artifact writes."""

    resolved = path.expanduser().resolve()
    if resolved.parent == resolved:
        raise ValueError("--output-dir cannot be a filesystem root")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"--output-dir is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(
                f"--output-dir already contains artifacts; choose a new empty path: {resolved}"
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def _seed_axis_applicability(
    suite: str,
    *,
    public_fixture: bool | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Describe which resolved seed axes actually affect each requested protocol."""

    applicability: dict[str, dict[str, dict[str, Any]]] = {}
    if suite in {"core", "all"}:
        applicability["core"] = {
            "data": {
                "applicable": True,
                "use": "generated graphs, excitations, trajectories, labels, and cache key",
            },
            "split": {
                "applicable": False,
                "use": "not_applicable: generated split assignment is part of data_seed",
            },
            "chart": {
                "applicable": False,
                "use": "not_applicable: conductance track has no spanning-tree chart sampling",
            },
            "model": {
                "applicable": True,
                "use": "model initialization and training DataLoader shuffle",
            },
        }
    if suite in {"public", "all"}:
        fixture = bool(public_fixture)
        applicability["public"] = {
            "data": {
                "applicable": fixture,
                "use": (
                    "offline fixture graph generation"
                    if fixture
                    else "not_applicable: official dataset content is fixed by its source"
                ),
            },
            "split": {
                "applicable": False,
                "use": (
                    "not_applicable: fixture split membership is preassigned by data_seed"
                    if fixture
                    else "not_applicable: official PascalVOC-SP/MolHIV splits are fixed"
                ),
            },
            "chart": {
                "applicable": False,
                "use": "not_applicable: public conductance baselines do not sample tree charts",
            },
            "model": {
                "applicable": True,
                "use": "model initialization and training DataLoader shuffle",
            },
        }
    return applicability


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "public", "all"), default="core")
    parser.add_argument("--data-root", type=Path, default=default_root / "data")
    parser.add_argument("--output-dir", type=Path, default=default_root / "results" / "paper")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="legacy fallback for any seed axis not supplied explicitly",
    )
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--chart-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="offline CPU CI/fixture profile")
    parser.add_argument(
        "--allow-download", action="store_true", help="allow official PyG/OGB downloads"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    arguments = build_parser().parse_args(argv)
    seed_axes: SeedAxes = resolve_seed_axes(
        arguments.seed,
        data_seed=arguments.data_seed,
        split_seed=arguments.split_seed,
        chart_seed=arguments.chart_seed,
        model_seed=arguments.model_seed,
    )
    if arguments.batch_size < 1 or arguments.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative")
    device = resolve_device(arguments.device)
    amp = device.type == "cuda" if arguments.amp is None else bool(arguments.amp)
    if device.type != "cuda" and amp:
        raise ValueError("--amp is a CUDA float16 path; use --no-amp on CPU")
    if arguments.tiny and device.type == "cpu":
        amp = False
    pin_memory = (
        device.type == "cuda" if arguments.pin_memory is None else bool(arguments.pin_memory)
    )
    if device.type != "cuda":
        pin_memory = False
    epochs = arguments.epochs if arguments.epochs is not None else (8 if arguments.tiny else 100)
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    # Dataset preparation receives only the data axis.  Reset the global RNG to
    # the model axis immediately before optimization below.
    seed_everything(seed_axes.data)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = _prepare_output_dir(arguments.output_dir)
    started = time.perf_counter()
    prepared: dict[str, Any] = {}
    core = None
    public = None
    if arguments.suite in {"core", "all"}:
        core, manifest_path, manifest = prepare_core_cache(
            arguments.data_root, seed=seed_axes.data, tiny=arguments.tiny
        )
        prepared["core"] = {
            "manifest": str(manifest_path),
            "cache_key": manifest["cache_key"],
            "data_seed": seed_axes.data,
        }
    if arguments.suite in {"public", "all"}:
        public, marker_path, manifest = prepare_public_data(
            arguments.data_root,
            seed=seed_axes.data,
            tiny=arguments.tiny,
            allow_download=arguments.allow_download,
        )
        prepared["public"] = {
            "manifest": str(marker_path),
            "fixture": manifest["fixture"],
            "data_seed": seed_axes.data if manifest["fixture"] else "not_applicable",
            "split_seed": "not_applicable",
            "chart_seed": "not_applicable",
        }
    public_fixture = None if public is None else bool(public["fixture"])
    seed_applicability = _seed_axis_applicability(arguments.suite, public_fixture=public_fixture)
    if arguments.prepare_only:
        summary = {
            "status": "prepared",
            "suite": arguments.suite,
            "seed_axes": seed_axes.to_manifest(),
            "seed_axis_applicability": seed_applicability,
            "tiny": arguments.tiny,
            "prepared": prepared,
        }
        (output_dir / "prepare_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    seed_everything(seed_axes.model)
    results: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    model_states: dict[str, Any] = {}
    try:
        if core is not None:
            core_results, core_history, core_states = run_core(
                core,
                device=device,
                epochs=epochs,
                learning_rate=arguments.learning_rate,
                batch_size=arguments.batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=arguments.num_workers,
                seed=seed_axes.model,
            )
            results["core"] = core_results
            histories.extend(core_history)
            model_states["core"] = core_states
        if public is not None:
            public_epochs = min(epochs, 50)
            public_results, public_history, public_states = run_public(
                public,
                device=device,
                epochs=public_epochs,
                learning_rate=arguments.learning_rate,
                batch_size=arguments.batch_size,
                amp=amp,
                pin_memory=pin_memory,
                num_workers=arguments.num_workers,
                seed=seed_axes.model,
            )
            results["public"] = public_results
            histories.extend(public_history)
            model_states["public"] = public_states
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA out of memory in the paper runner. Re-run with a smaller --batch-size "
            "(and optionally --no-amp only for numerical diagnosis; AMP normally saves memory)."
        ) from error
    elapsed = time.perf_counter() - started
    summary = {
        "scope": "independent_sparse_incidence_conductance_attention",
        "suite": arguments.suite,
        "seed_axes": seed_axes.to_manifest(),
        "seed_axis_applicability": seed_applicability,
        "tiny": arguments.tiny,
        "prepared": prepared,
        "configuration": {
            "epochs": epochs,
            "learning_rate": arguments.learning_rate,
            "batch_size": arguments.batch_size,
            "num_workers": arguments.num_workers,
        },
        "runtime": {
            **runtime_metadata(
                device, amp=amp, pin_memory=pin_memory, batch_size=arguments.batch_size
            ),
            "elapsed_seconds": elapsed,
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metric_rows = _metric_rows(results)
    _write_csv(output_dir / "metrics.csv", metric_rows, ["path", "value"])
    _write_csv(
        output_dir / "history.csv",
        histories,
        [
            "suite",
            "baseline",
            "training_objective",
            "epoch",
            "train_loss",
            "validation_loss",
        ],
    )
    torch.save(model_states, output_dir / "models.pt")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
````

# research/conductance_gat/paper_data.py

````python
"""Deterministic S1--S4 synthetic paper datasets for conductance GAT.

All generated examples use sparse ``edge_index`` tensors.  The cache key is a
canonical hash of the generation request and each manifest contains both a
content fingerprint (independent of ``torch.save`` metadata) and the serialized
artifact checksum.  There are no network or PyG dependencies in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as nnf

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_publish,
    atomic_write_json,
)

from .sparse import edge_divergence, edge_gradient, weighted_degree

SCHEMA_VERSION = 2
GENERATOR_VERSION = "conductance-s1-s4-edge-index-v5"


def _generator(seed: int) -> torch.Generator:
    result = torch.Generator(device="cpu")
    result.manual_seed(int(seed))
    return result


def _canonical_edges(pairs: list[tuple[int, int]], num_nodes: int, seed: int) -> Tensor:
    unique = sorted({(min(a, b), max(a, b)) for a, b in pairs if a != b})
    if not unique:
        raise ValueError("a graph needs at least one non-self edge")
    generator = _generator(seed)
    oriented: list[tuple[int, int]] = []
    signs = torch.randint(0, 2, (len(unique),), generator=generator)
    for index, (first, second) in enumerate(unique):
        oriented.append((second, first) if int(signs[index]) else (first, second))
    result = torch.tensor(oriented, dtype=torch.long).t().contiguous()
    if int(result.max()) >= num_nodes:
        raise ValueError("edge endpoint outside graph")
    return result


def make_graph(num_nodes: int, family: str, seed: int) -> Tensor:
    """Generate a connected simple undirected graph with arbitrary orientation."""

    if num_nodes < 4:
        raise ValueError("num_nodes must be at least four")
    generator = _generator(seed)
    pairs: list[tuple[int, int]] = []
    if family == "er":
        # A random recursive tree guarantees connectedness before extra edges.
        for node in range(1, num_nodes):
            parent = int(torch.randint(0, node, (1,), generator=generator))
            pairs.append((parent, node))
        target_edges = min(num_nodes * (num_nodes - 1) // 2, 2 * num_nodes)
        candidates = torch.randperm(num_nodes * num_nodes, generator=generator).tolist()
        for flat in candidates:
            a, b = divmod(flat, num_nodes)
            if a < b:
                pairs.append((a, b))
            if len(set((min(x, y), max(x, y)) for x, y in pairs)) >= target_edges:
                break
    elif family == "rgg":
        coordinates = torch.rand((num_nodes, 2), generator=generator)
        distances = torch.cdist(coordinates, coordinates)
        candidates = sorted(
            (float(distances[a, b]), a, b)
            for a in range(num_nodes)
            for b in range(a + 1, num_nodes)
        )
        # A Euclidean minimum spanning tree makes the random geometric graph
        # connected; the shortest remaining pairs define its radius-like edge
        # set. No arbitrary long random-tree edges are injected.
        parents = list(range(num_nodes))

        def find(node: int) -> int:
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node

        for _, first, second in candidates:
            root_first, root_second = find(first), find(second)
            if root_first != root_second:
                parents[root_second] = root_first
                pairs.append((first, second))
            if len(pairs) == num_nodes - 1:
                break
        target_edges = min(num_nodes * (num_nodes - 1) // 2, 2 * num_nodes)
        for _, first, second in candidates:
            pairs.append((first, second))
            if len(set((min(x, y), max(x, y)) for x, y in pairs)) >= target_edges:
                break
    elif family == "grid":
        columns = max(2, int(math.ceil(math.sqrt(num_nodes))))
        for node in range(num_nodes):
            row, column = divmod(node, columns)
            if column and node - 1 >= 0:
                pairs.append((node - 1, node))
            above = node - columns
            if row and above >= 0:
                pairs.append((above, node))
    elif family == "barbell":
        left = max(2, num_nodes // 3)
        right_start = num_nodes - left
        for start, stop in ((0, left), (right_start, num_nodes)):
            for a in range(start, stop):
                for b in range(a + 1, stop):
                    pairs.append((a, b))
        for node in range(left - 1, right_start):
            pairs.append((node, node + 1))
    else:
        raise ValueError(f"unknown graph family {family!r}")
    return _canonical_edges(pairs, num_nodes, seed + 991)


def make_edge_features(edge_index: Tensor, num_nodes: int, seed: int, width: int = 3) -> Tensor:
    if width < 3:
        raise ValueError("synthetic edge features require at least three channels")
    generator = _generator(seed)
    random_features = torch.randn((edge_index.shape[1], width), generator=generator)
    degree = torch.bincount(edge_index.reshape(-1), minlength=num_nodes).float()
    tail, head = edge_index
    random_features[:, 2] = (degree[tail] + degree[head]) / degree.max().clamp_min(1.0)
    return random_features


def static_conductance(edge_features: Tensor, contrast: float | None = None) -> Tensor:
    base = (
        0.85 * edge_features[:, 0]
        - 0.35 * edge_features[:, 1]
        + 0.25 * edge_features[:, 0].square()
        + 0.20 * torch.sin(edge_features[:, 1])
    )
    if contrast is None:
        return 0.15 + nnf.softplus(base)
    if contrast < 1:
        raise ValueError("contrast must be at least one")
    if contrast == 1:
        return torch.ones_like(base)
    centered = base - base.mean()
    span = centered.max() - centered.min()
    normalized = (centered - centered.min()) / span.clamp_min(1.0e-8)
    return torch.exp((normalized - 0.5) * math.log(float(contrast)))


def nonlinear_conductance(edge_features: Tensor, gradient: Tensor) -> Tensor:
    base = static_conductance(edge_features)
    state_factor = 0.65 + 0.70 * torch.sigmoid(1.5 * (gradient.abs().mean(dim=1) - 0.7))
    return base * state_factor


def _sample_potential(
    num_nodes: int,
    channels: int,
    seed: int,
    active_fraction: float = 1.0,
) -> Tensor:
    generator = _generator(seed)
    state = torch.randn((num_nodes, channels), generator=generator)
    if active_fraction < 1.0:
        active_count = max(2, int(round(active_fraction * num_nodes)))
        active = torch.randperm(num_nodes, generator=generator)[:active_count]
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[active] = True
        state[~mask] = 0.0
    return state - state.mean(dim=0, keepdim=True)


def _safe_step(edge_index: Tensor, conductance: Tensor, num_nodes: int, requested: float) -> float:
    maximum = float(weighted_degree(edge_index, conductance, num_nodes).max())
    return min(float(requested), 0.80 / max(maximum, 1.0e-8))


def make_example(
    *,
    graph_id: str,
    num_nodes: int,
    family: str,
    graph_seed: int,
    excitation_seed: int,
    channels: int = 2,
    active_fraction: float = 1.0,
    snr_db: float | None = None,
    contrast: float | None = None,
    nonlinear: bool = False,
    requested_step: float = 0.025,
) -> dict[str, Any]:
    edges = make_graph(num_nodes, family, graph_seed)
    feature_width = 4 if contrast is not None else 3
    features = make_edge_features(edges, num_nodes, graph_seed + 31, feature_width)
    if contrast is not None:
        features[:, 3] = math.log10(float(contrast)) / 2.0
    state = _sample_potential(num_nodes, channels, excitation_seed, active_fraction)
    gradient = edge_gradient(edges, state)
    conductance = (
        nonlinear_conductance(features, gradient)
        if nonlinear
        else static_conductance(features, contrast)
    )
    flux = conductance[:, None] * gradient
    message = edge_divergence(edges, flux, num_nodes)
    step = _safe_step(edges, conductance, num_nodes, requested_step)
    next_state = state - step * message
    observed_flux = flux.clone()
    if snr_db is not None:
        generator = _generator(excitation_seed + 701)
        signal_rms = flux.square().mean().sqrt()
        noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
        observed_flux = flux + noise_rms * torch.randn(flux.shape, generator=generator)
    observed_node_message = edge_divergence(edges, observed_flux, num_nodes)
    excited = gradient.abs().amax(dim=1) > 1.0e-6
    return {
        "graph_id": graph_id,
        "edge_index": edges,
        "edge_features": features,
        "node_state": state,
        "true_conductance": conductance,
        "true_gradient": gradient,
        "true_flux": flux,
        "true_node_message": message,
        "true_next_state": next_state,
        "observed_flux": observed_flux,
        "observed_node_message": observed_node_message,
        "step_size": step,
        "metadata": {
            "family": family,
            "num_nodes": num_nodes,
            "contrast": contrast,
            "snr_db": "infinity" if snr_db is None else float(snr_db),
            "active_node_fraction": float(active_fraction),
            "excited_edge_fraction": float(excited.float().mean()),
        },
    }


def _vary_nodes(low: int, high: int, seed: int) -> int:
    return int(torch.randint(low, high + 1, (1,), generator=_generator(seed)))


def generate_s1(seed: int, tiny: bool) -> dict[str, Any]:
    counts = (4, 1, 1) if tiny else (42, 9, 9)
    excitation_counts = (3, 2, 2) if tiny else (6, 3, 3)
    result: dict[str, Any] = {name: [] for name in ("train", "validation", "test", "seen_test")}
    offset = 0
    for split, graph_count, excitation_count in zip(
        ("train", "validation", "test"), counts, excitation_counts, strict=True
    ):
        for graph_number in range(graph_count):
            graph_seed = seed * 100_000 + offset * 101 + 11
            graph_id = f"s1-{split}-{graph_number:03d}"
            nodes = _vary_nodes(8 if tiny else 16, 14 if tiny else 32, graph_seed)
            family = "er" if graph_number % 2 == 0 else "rgg"
            for excitation in range(excitation_count):
                result[split].append(
                    make_example(
                        graph_id=graph_id,
                        num_nodes=nodes,
                        family=family,
                        graph_seed=graph_seed,
                        excitation_seed=graph_seed + 10_000 + excitation,
                    )
                )
            if split == "train":
                seen_count = 1 if tiny else 2
                for excitation in range(seen_count):
                    result["seen_test"].append(
                        make_example(
                            graph_id=graph_id,
                            num_nodes=nodes,
                            family=family,
                            graph_seed=graph_seed,
                            excitation_seed=graph_seed + 20_000 + excitation,
                        )
                    )
            offset += 1
    result["description"] = "S1 static shared-law identification; graph-ID split 70/15/15"
    return result


def _s2_protocol_counts(tiny: bool) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return graph and per-graph excitation counts for train/validation/test."""

    if tiny:
        return (4, 2, 4), (2, 2, 2)
    return (28, 8, 16), (4, 3, 3)


def generate_s2(seed: int, tiny: bool) -> dict[str, Any]:
    counts, excitation_counts = _s2_protocol_counts(tiny)
    result: dict[str, Any] = {name: [] for name in ("train", "validation", "test")}
    for split_number, (split, count, excitations) in enumerate(
        zip(("train", "validation", "test"), counts, excitation_counts, strict=True)
    ):
        for graph_number in range(count):
            graph_seed = seed * 200_000 + split_number * 20_000 + graph_number * 131 + 29
            if split == "test":
                low, high = (12, 20) if tiny else (48, 96)
                family = "grid" if graph_number % 2 == 0 else "barbell"
            else:
                low, high = (6, 10) if tiny else (16, 32)
                family = "er" if graph_number % 2 == 0 else "rgg"
            nodes = _vary_nodes(low, high, graph_seed)
            graph_id = f"s2-{split}-{family}-{graph_number:03d}"
            for excitation in range(excitations):
                result[split].append(
                    make_example(
                        graph_id=graph_id,
                        num_nodes=nodes,
                        family=family,
                        graph_seed=graph_seed,
                        excitation_seed=graph_seed + 30_000 + excitation,
                    )
                )
    result["description"] = "S2 ER/RGG n=16..32 to grid/barbell n=48..96 topology/size OOD"
    return result


def _make_trajectory(
    *,
    graph_id: str,
    num_nodes: int,
    family: str,
    graph_seed: int,
    trajectory_seed: int,
    horizon: int,
) -> dict[str, Any]:
    edges = make_graph(num_nodes, family, graph_seed)
    features = make_edge_features(edges, num_nodes, graph_seed + 31, 3)
    state = _sample_potential(num_nodes, 2, trajectory_seed)
    states = [state]
    conductances: list[Tensor] = []
    fluxes: list[Tensor] = []
    steps: list[float] = []
    for _ in range(horizon):
        gradient = edge_gradient(edges, state)
        conductance = nonlinear_conductance(features, gradient)
        flux = conductance[:, None] * gradient
        message = edge_divergence(edges, flux, num_nodes)
        step = _safe_step(edges, conductance, num_nodes, 0.025)
        state = state - step * message
        conductances.append(conductance)
        fluxes.append(flux)
        steps.append(step)
        states.append(state)
    return {
        "graph_id": graph_id,
        "edge_index": edges,
        "edge_features": features,
        "states": torch.stack(states),
        "conductances": torch.stack(conductances),
        "fluxes": torch.stack(fluxes),
        "steps": torch.tensor(steps),
        "metadata": {"family": family, "num_nodes": num_nodes, "horizon": horizon},
    }


def _trajectory_examples(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []
    edges = trajectory["edge_index"]
    for time in range(trajectory["conductances"].shape[0]):
        state = trajectory["states"][time]
        next_state = trajectory["states"][time + 1]
        conductance = trajectory["conductances"][time]
        gradient = edge_gradient(edges, state)
        flux = trajectory["fluxes"][time]
        examples.append(
            {
                "graph_id": trajectory["graph_id"],
                "edge_index": edges,
                "edge_features": trajectory["edge_features"],
                "node_state": state,
                "true_conductance": conductance,
                "true_gradient": gradient,
                "true_flux": flux,
                "true_node_message": edge_divergence(edges, flux, state.shape[0]),
                "true_next_state": next_state,
                "observed_flux": flux,
                "observed_node_message": edge_divergence(edges, flux, state.shape[0]),
                "step_size": float(trajectory["steps"][time]),
                "metadata": {**trajectory["metadata"], "time": time},
            }
        )
    return examples


def generate_s3(seed: int, tiny: bool) -> dict[str, Any]:
    counts = (3, 1, 2) if tiny else (12, 3, 5)
    horizon = 10 if tiny else 50
    result: dict[str, Any] = {
        "train": [],
        "validation": [],
        "test": [],
        "rollout_test": [],
        "horizons": [1, 5, 10] if tiny else [1, 5, 10, 50],
    }
    for split_number, (split, count) in enumerate(
        zip(("train", "validation", "test"), counts, strict=True)
    ):
        for number in range(count):
            graph_seed = seed * 300_000 + split_number * 30_000 + number * 151 + 37
            nodes = _vary_nodes(8 if tiny else 18, 14 if tiny else 36, graph_seed)
            family = "er" if number % 2 == 0 else "rgg"
            trajectory = _make_trajectory(
                graph_id=f"s3-{split}-{number:03d}",
                num_nodes=nodes,
                family=family,
                graph_seed=graph_seed,
                trajectory_seed=graph_seed + 50_000,
                horizon=horizon,
            )
            result[split].extend(_trajectory_examples(trajectory))
            if split == "test":
                result["rollout_test"].append(trajectory)
    result["description"] = "S3 state-dependent positive nonlinear held-graph rollout"
    return result


def generate_s4(seed: int, tiny: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"train": [], "validation": [], "test": []}
    contrasts = (1.0, 10.0, 100.0)
    active_fractions = (1.0, 0.25)
    snrs: tuple[float | None, ...] = (None, 40.0, 20.0)
    graph_counts = (1, 1, 1) if tiny else (3, 1, 2)
    excitation_counts = (2, 1, 2) if tiny else (6, 2, 4)
    cell = 0
    for contrast in contrasts:
        for active_fraction in active_fractions:
            for snr in snrs:
                for split_number, (split, graph_count, excitation_count) in enumerate(
                    zip(
                        ("train", "validation", "test"),
                        graph_counts,
                        excitation_counts,
                        strict=True,
                    )
                ):
                    for graph_number in range(graph_count):
                        graph_seed = (
                            seed * 400_000
                            + cell * 10_000
                            + split_number * 2_000
                            + graph_number * 173
                            + 41
                        )
                        nodes = _vary_nodes(8 if tiny else 18, 14 if tiny else 32, graph_seed)
                        graph_id = (
                            f"s4-{split}-c{contrast:g}-a{active_fraction:g}-"
                            f"s{snr}-{graph_number:02d}"
                        )
                        for excitation in range(excitation_count):
                            result[split].append(
                                make_example(
                                    graph_id=graph_id,
                                    num_nodes=nodes,
                                    family="er" if cell % 2 == 0 else "rgg",
                                    graph_seed=graph_seed,
                                    excitation_seed=graph_seed + 70_000 + excitation,
                                    active_fraction=active_fraction,
                                    snr_db=snr,
                                    contrast=contrast,
                                    requested_step=0.01,
                                )
                            )
                cell += 1
    result["description"] = "S4 contrast x coverage x SNR identifiability robustness factorial"
    return result


def generate_core(seed: int, tiny: bool = False) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "tiny": bool(tiny),
        "s1": generate_s1(seed + 101, tiny),
        "s2": generate_s2(seed + 202, tiny),
        "s3": generate_s3(seed + 303, tiny),
        "s4": generate_s4(seed + 404, tiny),
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _content_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(_canonical_json(list(tensor.shape)))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for child in item:
                update(child)
        else:
            digest.update(_canonical_json(item))

    update(value)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_ids(core: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for suite_name in ("s1", "s2", "s3", "s4"):
        result[suite_name] = {}
        for split in ("train", "validation", "test", "seen_test"):
            examples = core[suite_name].get(split, [])
            result[suite_name][split] = sorted({str(example["graph_id"]) for example in examples})
    return result


def _split_counts(core: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        suite_name: {
            split: len(core[suite_name].get(split, []))
            for split in ("train", "validation", "test", "seen_test")
            if split in core[suite_name]
        }
        for suite_name in ("s1", "s2", "s3", "s4")
    }


def _expected_split_counts(tiny: bool) -> dict[str, dict[str, int]]:
    s2_graph_counts, s2_excitation_counts = _s2_protocol_counts(tiny)
    s2_counts = {
        split: graph_count * excitation_count
        for split, graph_count, excitation_count in zip(
            ("train", "validation", "test"),
            s2_graph_counts,
            s2_excitation_counts,
            strict=True,
        )
    }
    return (
        {
            "s1": {"train": 12, "validation": 2, "test": 2, "seen_test": 4},
            "s2": s2_counts,
            "s3": {"train": 30, "validation": 10, "test": 20},
            "s4": {"train": 36, "validation": 18, "test": 36},
        }
        if tiny
        else {
            "s1": {"train": 252, "validation": 27, "test": 27, "seen_test": 84},
            "s2": s2_counts,
            "s3": {"train": 600, "validation": 150, "test": 250},
            "s4": {"train": 324, "validation": 36, "test": 144},
        }
    )


def _core_request(seed: int, tiny: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "tiny": bool(tiny),
    }


def _core_cache_paths(data_root: Path | str, request: dict[str, Any]) -> tuple[Path, Path]:
    cache_key = hashlib.sha256(_canonical_json(request)).hexdigest()[:16]
    cache_dir = Path(data_root).expanduser().resolve() / "conductance_gat" / f"core-{cache_key}"
    return cache_dir / "core.pt", cache_dir / "manifest.json"


def _validate_example(example: dict[str, Any]) -> None:
    required = {
        "edge_index",
        "edge_features",
        "node_state",
        "true_conductance",
        "true_gradient",
        "true_flux",
        "true_node_message",
        "true_next_state",
        "observed_flux",
        "observed_node_message",
    }
    if not required.issubset(example):
        raise CacheCorruptError("conductance example is missing required tensors")
    tensors = {name: example[name] for name in required}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise CacheCorruptError("conductance example contains a non-tensor payload")
    edge_index = tensors["edge_index"]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise CacheCorruptError("conductance edge_index must have shape [2, m] and dtype long")
    edge_count = int(edge_index.shape[1])
    node_state = tensors["node_state"]
    if node_state.ndim != 2:
        raise CacheCorruptError("conductance node_state must have shape [n, channels]")
    node_count, channels = map(int, node_state.shape)
    expected_shapes = {
        "edge_features": (edge_count, tensors["edge_features"].shape[-1]),
        "true_conductance": (edge_count,),
        "true_gradient": (edge_count, channels),
        "true_flux": (edge_count, channels),
        "true_node_message": (node_count, channels),
        "true_next_state": (node_count, channels),
        "observed_flux": (edge_count, channels),
        "observed_node_message": (node_count, channels),
    }
    for name, expected in expected_shapes.items():
        if tuple(tensors[name].shape) != tuple(expected):
            raise CacheCorruptError(f"conductance tensor {name!r} has an invalid shape")
    if (
        edge_count < 1
        or node_count < 2
        or int(edge_index.min()) < 0
        or int(edge_index.max()) >= node_count
    ):
        raise CacheCorruptError("conductance graph topology is invalid")
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise CacheCorruptError("conductance cache contains a non-finite tensor")
    if not torch.all(tensors["true_conductance"] > 0):
        raise CacheCorruptError("conductance cache contains a non-positive conductance")


def _validate_core_content(core: Any, request: dict[str, Any], manifest: dict[str, Any]) -> None:
    if not isinstance(core, dict):
        raise CacheCorruptError("conductance core artifact must be a mapping")
    for key, value in request.items():
        if core.get(key) != value:
            raise CacheWrongRequestError(f"conductance core field {key!r} does not match request")
    for suite_name in ("s1", "s2", "s3", "s4"):
        suite = core.get(suite_name)
        if not isinstance(suite, dict):
            raise CacheCorruptError(f"conductance cache is missing suite {suite_name!r}")
        for split in ("train", "validation", "test"):
            examples = suite.get(split)
            if not isinstance(examples, list) or not examples:
                raise CacheCorruptError(f"conductance {suite_name}.{split} is empty or invalid")
            for example in examples:
                if not isinstance(example, dict):
                    raise CacheCorruptError("conductance split contains a non-mapping example")
                _validate_example(example)
    graph_ids = _graph_ids(core)
    if manifest.get("graph_ids") != graph_ids:
        raise CacheCorruptError("conductance graph-ID manifest does not match the artifact")
    split_counts = _split_counts(core)
    if manifest.get("split_counts") != split_counts:
        raise CacheCorruptError("conductance split-count manifest does not match the artifact")
    if split_counts != _expected_split_counts(bool(request["tiny"])):
        raise CacheCorruptError("conductance split cardinalities do not match the paper protocol")
    for suite_splits in graph_ids.values():
        named_sets = [set(values) for name, values in suite_splits.items() if name != "seen_test"]
        for index, left in enumerate(named_sets):
            if any(left.intersection(right) for right in named_sets[index + 1 :]):
                raise CacheCorruptError("conductance graph IDs cross physical graph splits")


def validate_core_cache(
    data_root: Path | str, *, seed: int, tiny: bool = False
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Read and fully validate one requested generated core cache without writing."""

    request = _core_request(seed, tiny)
    artifact_path, manifest_path = _core_cache_paths(data_root, request)
    present = (artifact_path.is_file(), manifest_path.is_file())
    if not any(present):
        raise FileNotFoundError(
            f"conductance core cache is missing for seed={seed}: {artifact_path}"
        )
    if not all(present):
        raise CacheIncompleteError(
            f"conductance core.pt and manifest.json must both exist: {artifact_path.parent}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"invalid conductance cache manifest: {manifest_path}") from error
    if manifest.get("request") != request:
        raise CacheWrongRequestError(f"cache manifest request mismatch: {manifest_path}")
    if _file_sha256(artifact_path) != manifest.get("artifact_sha256"):
        raise CacheCorruptError(f"cache artifact checksum mismatch: {artifact_path}")
    try:
        try:
            core = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            core = torch.load(artifact_path, map_location="cpu")
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise CacheCorruptError(
            f"failed to deserialize conductance cache: {artifact_path}"
        ) from error
    if _content_fingerprint(core) != manifest.get("content_sha256"):
        raise CacheCorruptError(f"cache tensor-content checksum mismatch: {artifact_path}")
    _validate_core_content(core, request, manifest)
    return core, manifest_path, manifest


def prepare_core_cache(
    data_root: Path | str, *, seed: int, tiny: bool = False, force: bool = False
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    request = _core_request(seed, tiny)
    artifact_path, manifest_path = _core_cache_paths(data_root, request)
    cache_key = artifact_path.parent.name.removeprefix("core-")
    if (artifact_path.exists() or manifest_path.exists()) and not force:
        return validate_core_cache(data_root, seed=seed, tiny=tiny)

    core = generate_core(seed, tiny)
    expected_content_sha256 = _content_fingerprint(core)

    def validate_artifact(temporary: Path) -> None:
        try:
            loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            loaded = torch.load(temporary, map_location="cpu")
        if _content_fingerprint(loaded) != expected_content_sha256:
            raise CacheCorruptError("new conductance artifact failed temporary validation")

    def write_artifact(temporary: Path) -> None:
        # Saving through a stream prevents the unique temporary basename from
        # entering PyTorch's ZIP metadata, preserving byte determinism.
        with temporary.open("wb") as stream:
            torch.save(core, stream)

    atomic_publish(artifact_path, write_artifact, validator=validate_artifact)
    manifest = {
        "request": request,
        "cache_key": cache_key,
        "artifact": artifact_path.name,
        "artifact_sha256": _file_sha256(artifact_path),
        "content_sha256": expected_content_sha256,
        "graph_ids": _graph_ids(core),
        "split_counts": _split_counts(core),
    }
    atomic_write_json(
        manifest_path,
        manifest,
        validator=lambda temporary: json.loads(temporary.read_text(encoding="utf-8")),
    )
    return validate_core_cache(data_root, seed=seed, tiny=tiny)


__all__ = [
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "generate_core",
    "generate_s1",
    "generate_s2",
    "generate_s3",
    "generate_s4",
    "make_example",
    "make_graph",
    "nonlinear_conductance",
    "prepare_core_cache",
    "static_conductance",
    "validate_core_cache",
]
````

# research/conductance_gat/public_data.py

````python
"""Optional PyG/OGB public adapters plus deterministic offline fixtures.

The synthetic paper core has no optional dependencies.  Official
PascalVOC-SP and ogbg-molhiv data are touched only through this module and only
when the caller explicitly allows the dataset classes to download.  ``tiny``
fixtures exercise the exact adapter/training path without network access and
are always labelled as fixtures in result files.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_write_json,
)

PUBLIC_SCHEMA_VERSION = 2
SOURCE_URLS = {
    "pascalvoc_sp": "https://github.com/vijaydwivedi75/lrgb",
    "ogbg_molhiv": "https://ogb.stanford.edu/docs/graphprop/",
}
OFFICIAL_SPLIT_SIZES = {
    "pascalvoc_sp": {"train": 8_498, "validation": 1_227, "test": 1_449},
    "ogbg_molhiv": {"train": 32_901, "validation": 4_113, "test": 4_113},
}


class OptionalDatasetDependencyError(RuntimeError):
    pass


class IndexedCollection(Sequence[dict[str, Any]]):
    """Lazy adapter over a PyG dataset and an official index split."""

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int] | Tensor,
        adapter: Callable[[Any, str], dict[str, Any]],
        prefix: str,
    ) -> None:
        self.dataset = dataset
        if isinstance(indices, Tensor):
            self.indices = [int(value) for value in indices.reshape(-1)]
        else:
            self.indices = [int(value) for value in indices]
        self.adapter = adapter
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[index]
        return self.adapter(self.dataset[source_index], f"{self.prefix}-{source_index}")


def deduplicate_undirected_edges(
    edge_index: Tensor, edge_features: Tensor | None, num_nodes: int
) -> tuple[Tensor, Tensor]:
    """Collapse reciprocal PyG arcs and remove incidence-zero self loops.

    Continuous directional attributes are averaged into one orientation-free
    physical-edge feature. Integer/categorical reciprocal attributes must
    agree exactly; silently selecting one category would corrupt chemistry.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if edge_features is None:
        edge_features = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)
    if edge_features.ndim == 1:
        edge_features = edge_features[:, None]
    if edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError("edge attributes and edge_index disagree")
    selected: dict[tuple[int, int], list[int]] = {}
    for column in range(edge_index.shape[1]):
        first = int(edge_index[0, column])
        second = int(edge_index[1, column])
        if first == second:
            continue
        if first < 0 or second < 0 or first >= num_nodes or second >= num_nodes:
            raise ValueError("edge endpoint outside graph")
        key = (min(first, second), max(first, second))
        selected.setdefault(key, []).append(column)
    if not selected:
        raise ValueError("graph has no non-self edges after incidence conversion")
    keys = sorted(selected)
    indices = torch.tensor(keys, dtype=torch.long).t().contiguous()
    attributes: list[Tensor] = []
    for key in keys:
        values = edge_features[selected[key]]
        if values.is_floating_point():
            attributes.append(values.mean(dim=0))
        else:
            reference = values[0]
            if not torch.equal(values, reference.expand_as(values)):
                raise ValueError(f"conflicting categorical reciprocal edge attributes for {key}")
            attributes.append(reference)
    return indices, torch.stack(attributes)


def adapt_pyg_graph(data: Any, graph_id: str, *, task: str) -> dict[str, Any]:
    x = data.x
    if x is None:
        raise ValueError(f"{graph_id} has no node features")
    x = x.detach().cpu()
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.detach().cpu()
    edges, attributes = deduplicate_undirected_edges(
        data.edge_index.detach().cpu(), edge_attr, int(x.shape[0])
    )
    y = data.y.detach().cpu()
    if task == "node":
        y = y.reshape(-1).long()
        if y.numel() != x.shape[0]:
            raise ValueError("PascalVOC-SP node labels do not match the node count")
    elif task == "graph":
        y = y.reshape(-1).float()
    else:
        raise ValueError("task must be node or graph")
    return {
        "graph_id": graph_id,
        "x": x,
        "edge_index": edges,
        "edge_features": attributes,
        "y": y,
        "task": task,
        "categorical": task == "graph",
    }


def _fixture_graph(seed: int, graph_id: str, *, task: str, categorical: bool) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_nodes = 6 + seed % 4
    pairs = [(node, node + 1) for node in range(num_nodes - 1)]
    pairs.extend((node, node + 2) for node in range(num_nodes - 2) if node % 2 == 0)
    # Feed reciprocal arcs through the public deduplicator, as official PyG
    # datasets do, so fixture tests cover this otherwise easy-to-miss step.
    directed = pairs + [(b, a) for a, b in pairs]
    directed_edges = torch.tensor(directed, dtype=torch.long).t().contiguous()
    if categorical:
        x = torch.randint(0, 4, (num_nodes, 3), generator=generator)
        base_edge_attr = torch.randint(0, 3, (len(pairs), 2), generator=generator)
    else:
        x = torch.randn((num_nodes, 5), generator=generator)
        base_edge_attr = torch.randn((len(pairs), 2), generator=generator)
    edge_attr = torch.cat((base_edge_attr, base_edge_attr), dim=0)
    edge_index, edge_features = deduplicate_undirected_edges(directed_edges, edge_attr, num_nodes)
    if task == "node":
        y = ((x[:, 0] > 0).long() + (x[:, 1] > 0).long()).clamp_max(2)
    else:
        # Alternate labels deterministically so every fixture split exercises
        # the ROC-AUC evaluator; this is pipeline validation, not a benchmark.
        y = torch.tensor([float(seed % 2)])
    return {
        "graph_id": graph_id,
        "x": x,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "y": y,
        "task": task,
        "categorical": categorical,
    }


def make_public_fixtures(seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"fixture": True}
    for dataset_name, task, categorical in (
        ("pascalvoc_sp", "node", False),
        ("ogbg_molhiv", "graph", True),
    ):
        result[dataset_name] = {}
        offset = 0 if dataset_name == "pascalvoc_sp" else 10_000
        for split, count in (("train", 8), ("validation", 3), ("test", 4)):
            result[dataset_name][split] = [
                _fixture_graph(
                    seed + offset + number * 19,
                    f"fixture-{dataset_name}-{split}-{number:03d}",
                    task=task,
                    categorical=categorical,
                )
                for number in range(count)
            ]
    return result


def _dependency_error() -> OptionalDatasetDependencyError:
    return OptionalDatasetDependencyError(
        "Official public suites require optional packages 'torch-geometric' and 'ogb'. "
        "Install a PyTorch build matching the server CUDA runtime first, then run "
        "`python -m pip install torch-geometric ogb`. See "
        "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html "
        "and https://ogb.stanford.edu/docs/home/. The core S1-S4 suite does not need them."
    )


def _load_official(data_root: Path) -> dict[str, Any]:
    try:
        import torch_geometric  # noqa: F401
        from ogb.graphproppred import PygGraphPropPredDataset
        from torch_geometric.datasets import LRGBDataset
    except (ImportError, OSError) as error:
        raise _dependency_error() from error

    pyg_root = data_root / "pyg"
    pascal: dict[str, Any] = {}
    for split, official_split in (("train", "train"), ("validation", "val"), ("test", "test")):
        dataset = LRGBDataset(root=str(pyg_root), name="PascalVOC-SP", split=official_split)
        pascal[split] = IndexedCollection(
            dataset,
            range(len(dataset)),
            partial(adapt_pyg_graph, task="node"),
            f"pascal-{split}",
        )

    mol_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(data_root / "ogb"))
    split_indices = mol_dataset.get_idx_split()
    mol = {
        split: IndexedCollection(
            mol_dataset,
            split_indices[official],
            partial(adapt_pyg_graph, task="graph"),
            f"molhiv-{split}",
        )
        for split, official in (("train", "train"), ("validation", "valid"), ("test", "test"))
    }
    return {"fixture": False, "pascalvoc_sp": pascal, "ogbg_molhiv": mol}


def _fixture_hash(data: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for dataset_name in ("pascalvoc_sp", "ogbg_molhiv"):
        for split in ("train", "validation", "test"):
            for graph in data[dataset_name][split]:
                digest.update(graph["graph_id"].encode())
                for key in ("x", "edge_index", "edge_features", "y"):
                    tensor = graph[key].contiguous()
                    digest.update(str(tensor.dtype).encode())
                    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _processed_paths(datasets: dict[str, Any], root: Path) -> list[str]:
    paths: set[str] = set()
    for dataset_name in ("pascalvoc_sp", "ogbg_molhiv"):
        for split in ("train", "validation", "test"):
            collection = datasets[dataset_name][split]
            path_values = list(getattr(collection.dataset, "processed_paths", []))
            if not path_values and getattr(collection.dataset, "processed_dir", None):
                path_values = [collection.dataset.processed_dir]
            for path_value in path_values:
                path = Path(path_value).resolve()
                try:
                    paths.add(str(path.relative_to(root.resolve())))
                except ValueError:
                    paths.add(str(path))
    return sorted(paths)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processed_hashes(root: Path, paths: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path_value in paths:
        candidate = Path(path_value)
        resolved = candidate if candidate.is_absolute() else root / candidate
        files = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for path in files:
            if not path.is_file():
                continue
            try:
                key = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                key = str(path.resolve())
            hashes[key] = _file_sha256(path)
    return hashes


def validate_public_cache(
    data_root: Path | str, *, seed: int, tiny: bool
) -> tuple[Path, dict[str, Any]]:
    """Validate the public marker and all recorded processed files without downloading."""

    public_root = Path(data_root).expanduser().resolve() / "conductance_gat" / "public"
    marker = public_root / (f"fixture-seed{seed}.json" if tiny else "official-ready.json")
    if not marker.is_file():
        raise FileNotFoundError(f"conductance public cache marker is missing: {marker}")
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"invalid conductance public marker: {marker}") from error
    if manifest.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise CacheWrongRequestError(f"unsupported conductance public marker schema: {marker}")
    if bool(manifest.get("fixture")) != bool(tiny):
        raise CacheWrongRequestError(f"conductance public marker profile mismatch: {marker}")
    if tiny and manifest.get("seed") != int(seed):
        raise CacheWrongRequestError(f"conductance public fixture seed mismatch: {marker}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(SOURCE_URLS):
        raise CacheCorruptError("conductance public marker has an invalid dataset set")
    expected_sizes = (
        {name: {"train": 8, "validation": 3, "test": 4} for name in SOURCE_URLS}
        if tiny
        else OFFICIAL_SPLIT_SIZES
    )
    for name, split_sizes in expected_sizes.items():
        if datasets[name].get("source_url") != SOURCE_URLS[name]:
            raise CacheWrongRequestError(f"conductance public source mismatch for {name}")
        if datasets[name].get("splits") != split_sizes:
            raise CacheCorruptError(
                f"conductance public split cardinalities are invalid for {name}"
            )
    if tiny:
        expected_hash = _fixture_hash(make_public_fixtures(seed))
        if manifest.get("content_sha256") != expected_hash:
            raise CacheCorruptError("conductance public fixture checksum mismatch")
        return marker, manifest
    required_paths = manifest.get("required_processed_paths")
    stored_hashes = manifest.get("processed_sha256")
    if not isinstance(required_paths, list) or not required_paths:
        raise CacheIncompleteError("conductance public marker has no processed-file inventory")
    if not isinstance(stored_hashes, dict) or not stored_hashes:
        raise CacheIncompleteError("conductance public marker has no processed-file checksums")
    missing = []
    for path_value in required_paths:
        path = Path(path_value)
        resolved = path if path.is_absolute() else public_root / path
        if not resolved.exists():
            missing.append(str(resolved))
    if missing:
        raise CacheIncompleteError("conductance public processed files are missing: " + missing[0])
    actual_hashes = _processed_hashes(public_root, required_paths)
    if actual_hashes != stored_hashes:
        raise CacheCorruptError("conductance public processed-file checksum mismatch")
    return marker, manifest


def prepare_public_data(
    data_root: Path | str,
    *,
    seed: int,
    tiny: bool,
    allow_download: bool = False,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    public_root = Path(data_root).expanduser().resolve() / "conductance_gat" / "public"
    public_root.mkdir(parents=True, exist_ok=True)
    marker = public_root / (f"fixture-seed{seed}.json" if tiny else "official-ready.json")
    if tiny:
        datasets = make_public_fixtures(seed)
        manifest = {
            "schema_version": PUBLIC_SCHEMA_VERSION,
            "fixture": True,
            "seed": int(seed),
            "content_sha256": _fixture_hash(datasets),
            "datasets": {
                name: {
                    "source_url": SOURCE_URLS[name],
                    "splits": {split: len(datasets[name][split]) for split in datasets[name]},
                }
                for name in SOURCE_URLS
            },
        }
    else:
        if not allow_download and not marker.exists():
            raise RuntimeError(
                "Official public data is not marked prepared. Run once with "
                "`--suite public --prepare-only --allow-download` to let the official "
                "PyG/OGB dataset classes download into --data-root. No download is "
                "attempted without that explicit flag. Use --tiny for offline fixtures."
            )
        if not allow_download:
            validate_public_cache(data_root, seed=seed, tiny=False)
        datasets = _load_official(public_root)
        split_sizes = {
            name: {split: len(datasets[name][split]) for split in datasets[name]}
            for name in SOURCE_URLS
        }
        if split_sizes != OFFICIAL_SPLIT_SIZES:
            raise RuntimeError(
                "Official public split cardinalities do not match the pinned protocol: "
                f"{split_sizes}"
            )
        required_paths = _processed_paths(datasets, public_root)
        manifest = {
            "schema_version": PUBLIC_SCHEMA_VERSION,
            "fixture": False,
            "datasets": {
                name: {
                    "source_url": SOURCE_URLS[name],
                    "splits": split_sizes[name],
                }
                for name in SOURCE_URLS
            },
            "required_processed_paths": required_paths,
            "processed_sha256": _processed_hashes(public_root, required_paths),
        }
    atomic_write_json(
        marker,
        manifest,
        validator=lambda temporary: json.loads(temporary.read_text(encoding="utf-8")),
    )
    validate_public_cache(data_root, seed=seed, tiny=tiny)
    return datasets, marker, manifest


__all__ = [
    "IndexedCollection",
    "OptionalDatasetDependencyError",
    "SOURCE_URLS",
    "adapt_pyg_graph",
    "deduplicate_undirected_edges",
    "make_public_fixtures",
    "prepare_public_data",
    "validate_public_cache",
]
````

# research/conductance_gat/run.py

````python
"""Train and compare learned edge conductance against an isotropic baseline."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .model import IncidenceConductanceAttention, IsotropicConductanceAttention
from .synthetic import (
    ConductanceDataset,
    evaluate_model,
    make_conductance_dataset,
    split_excitations,
)


def _training_loss(
    model: nn.Module,
    dataset: ConductanceDataset,
    *,
    node_weight: float,
    flux_weight: float,
    conductance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_next, diagnostics = model(
        dataset.incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    node_loss = torch.mean((predicted_next - dataset.true_next_state).square())
    flux_loss = torch.mean((diagnostics["edge_flux"] - dataset.true_flux).square())
    predicted_c = diagnostics["conductance"].mean(dim=0).squeeze(-1)
    conductance_loss = torch.mean((predicted_c - dataset.true_conductance).square())
    loss = node_weight * node_loss + flux_weight * flux_loss + conductance_weight * conductance_loss
    terms = {
        "loss": float(loss.detach().cpu()),
        "node_mse": float(node_loss.detach().cpu()),
        "flux_mse": float(flux_loss.detach().cpu()),
        "conductance_mse": float(conductance_loss.detach().cpu()),
    }
    return loss, terms


def train_model(
    model: nn.Module,
    dataset: ConductanceDataset,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    node_weight: float,
    flux_weight: float,
    conductance_weight: float,
    log_every: int = 10,
) -> list[dict[str, float]]:
    if epochs < 1 or log_every < 1:
        raise ValueError("epochs and log_every must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = _training_loss(
            model,
            dataset,
            node_weight=node_weight,
            flux_weight=flux_weight,
            conductance_weight=conductance_weight,
        )
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            history.append({"epoch": float(epoch), **terms})
    return history


def resolve_device(requested: str | None) -> torch.device:
    """Resolve a portable device request without assuming the launch host."""
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {normalized!r} was requested, but this PyTorch build cannot use CUDA"
        )
    return device


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    """Record enough host information to distinguish CPU/CUDA experiment runs."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(config.get("device"))
    data_config = dict(config["data"])
    dataset = make_conductance_dataset(seed=seed, **data_config)
    train_data, test_data = split_excitations(
        dataset,
        train_fraction=float(config["training"]["train_fraction"]),
        seed=seed + 1,
    )
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    model_config = dict(config["model"])
    common = {
        "channels": int(data_config["channels"]),
        "edge_feature_channels": int(data_config["edge_feature_channels"]),
        "step_size": dataset.step_size,
        "stability_margin": float(model_config.pop("stability_margin", 0.95)),
        "adaptive_stability": bool(model_config.pop("adaptive_stability", True)),
    }
    learned = IncidenceConductanceAttention(**common, **model_config).to(device)
    isotropic = IsotropicConductanceAttention(**common).to(device)

    training = config["training"]
    train_kwargs = {
        "epochs": int(training["epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "node_weight": float(training["node_weight"]),
        "flux_weight": float(training["flux_weight"]),
        "conductance_weight": float(training["conductance_weight"]),
        "log_every": int(training["log_every"]),
    }
    learned_history = train_model(learned, train_data, **train_kwargs)
    isotropic_history = train_model(isotropic, train_data, **train_kwargs)

    summary: dict[str, Any] = {
        "scope": "incidence_conductance_attention_only",
        "seed": seed,
        "device": str(device),
        "runtime": runtime_metadata(device),
        "num_nodes": int(dataset.incidence.shape[1]),
        "num_edges": int(dataset.incidence.shape[0]),
        "train_excitations": train_data.num_excitations,
        "test_excitations": test_data.num_excitations,
        "step_size": dataset.step_size,
        "learned": evaluate_model(learned, test_data),
        "isotropic": evaluate_model(isotropic, test_data),
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _write_history(output_dir / "learned_history.csv", learned_history)
    _write_history(output_dir / "isotropic_history.csv", isotropic_history)
    portable_state = {
        name: parameter.detach().cpu() for name, parameter in learned.state_dict().items()
    }
    torch.save(portable_state, output_dir / "learned_model.pt")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML experiment configuration",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device override: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory override (defaults to the config-relative path)",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if arguments.output_dir is not None:
        output_dir = arguments.output_dir.expanduser()
    else:
        output_dir = Path(config.get("output_dir", "results")).expanduser()
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir
    config["output_dir"] = str(output_dir)
    if arguments.device is not None:
        config["device"] = arguments.device
    run_experiment(config)


if __name__ == "__main__":
    main()
````

# research/conductance_gat/sparse.py

````python
"""Sparse, variable-graph incidence conductance operators.

This module deliberately never materializes an incidence matrix.  For an
oriented edge ``tail -> head`` it evaluates ``g = H[head] - H[tail]`` and
implements ``B.T q`` with two ``index_add_`` calls.  Concatenating graphs is
therefore just concatenating nodes/edges and offsetting ``edge_index``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as nnf


def _inverse_softplus(value: float) -> float:
    x = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(x)))


def edge_gradient(edge_index: Tensor, node_state: Tensor) -> Tensor:
    """Return oriented edge differences without constructing ``B``."""

    _validate_edge_index(edge_index, node_state.shape[0])
    tail, head = edge_index
    return node_state.index_select(0, head) - node_state.index_select(0, tail)


def edge_divergence(edge_index: Tensor, edge_flux: Tensor, num_nodes: int) -> Tensor:
    """Return ``B.T @ edge_flux`` using CUDA-safe indexed accumulation."""

    _validate_edge_index(edge_index, num_nodes)
    if edge_flux.ndim != 2 or edge_flux.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_flux must have shape (num_edges, channels)")
    tail, head = edge_index
    result = edge_flux.new_zeros((num_nodes, edge_flux.shape[1]))
    result.index_add_(0, head, edge_flux)
    result.index_add_(0, tail, -edge_flux)
    return result


def weighted_degree(edge_index: Tensor, conductance: Tensor, num_nodes: int) -> Tensor:
    """Weighted undirected degree for one scalar conductance per edge."""

    _validate_edge_index(edge_index, num_nodes)
    values = conductance.reshape(-1)
    if values.shape[0] != edge_index.shape[1]:
        raise ValueError("conductance must contain one value per edge")
    result = values.new_zeros(num_nodes)
    result.index_add_(0, edge_index[0], values)
    result.index_add_(0, edge_index[1], values)
    return result


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a long tensor with shape (2, num_edges)")
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError("edge_index contains a node outside node_state")
        if torch.any(edge_index[0] == edge_index[1]):
            raise ValueError("incidence conductance edges cannot be self-loops")


def _scatter_graph_max(values: Tensor, graph_index: Tensor, num_graphs: int) -> Tensor:
    """Per-graph max with a torch-only CUDA implementation."""

    output = values.new_full((num_graphs,), -torch.inf)
    if hasattr(output, "scatter_reduce_"):
        output.scatter_reduce_(0, graph_index, values, reduce="amax", include_self=True)
    else:  # pragma: no cover - only reached on obsolete PyTorch versions
        for graph_id in range(num_graphs):
            selected = values[graph_index == graph_id]
            if selected.numel():
                output[graph_id] = selected.max()
    return output.masked_fill(torch.isinf(output), 0.0)


@dataclass
class PackedGraphBatch:
    """A dependency-free variable-graph mini-batch.

    Targets are optional so the same container can be used for inference and
    public benchmark adapters.  Every tensor is flat over all nodes or all
    edges; ``node_graph`` and ``edge_graph`` identify the owning graph.
    """

    node_state: Tensor
    edge_index: Tensor
    edge_features: Tensor
    node_graph: Tensor
    edge_graph: Tensor
    graph_ids: list[str]
    requested_step: Tensor
    true_conductance: Tensor | None = None
    true_gradient: Tensor | None = None
    true_flux: Tensor | None = None
    true_node_message: Tensor | None = None
    true_next_state: Tensor | None = None
    observed_flux: Tensor | None = None
    observed_node_message: Tensor | None = None
    metadata: list[dict[str, Any]] | None = None

    @property
    def num_graphs(self) -> int:
        return len(self.graph_ids)

    @property
    def num_nodes(self) -> int:
        return int(self.node_state.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PackedGraphBatch:
        values: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            values[name] = (
                value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
            )
        return PackedGraphBatch(**values)

    def pin_memory(self) -> PackedGraphBatch:
        values: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            values[name] = value.pin_memory() if isinstance(value, Tensor) else value
        return PackedGraphBatch(**values)


def pack_graph_examples(examples: Iterable[Mapping[str, Any]]) -> PackedGraphBatch:
    """Pack graph dictionaries while offsetting edges exactly once."""

    records = list(examples)
    if not records:
        raise ValueError("cannot pack an empty example list")
    node_states: list[Tensor] = []
    edge_indices: list[Tensor] = []
    edge_features: list[Tensor] = []
    node_graph: list[Tensor] = []
    edge_graph: list[Tensor] = []
    graph_ids: list[str] = []
    steps: list[float] = []
    metadata: list[dict[str, Any]] = []
    optional_names = (
        "true_conductance",
        "true_gradient",
        "true_flux",
        "true_node_message",
        "true_next_state",
        "observed_flux",
        "observed_node_message",
    )
    optional: dict[str, list[Tensor]] = {name: [] for name in optional_names}
    node_offset = 0
    feature_width: int | None = None
    channels: int | None = None
    for graph_number, record in enumerate(records):
        state = record["node_state"]
        edges = record["edge_index"]
        features = record["edge_features"]
        if state.ndim != 2 or features.ndim != 2:
            raise ValueError("node_state and edge_features must be matrices")
        _validate_edge_index(edges, state.shape[0])
        if edges.shape[1] != features.shape[0]:
            raise ValueError("edge_index and edge_features disagree on edge count")
        if channels is None:
            channels = int(state.shape[1])
            feature_width = int(features.shape[1])
        if state.shape[1] != channels or features.shape[1] != feature_width:
            raise ValueError("all examples in a batch need equal feature widths")
        node_states.append(state)
        edge_indices.append(edges + node_offset)
        edge_features.append(features)
        node_graph.append(torch.full((state.shape[0],), graph_number, dtype=torch.long))
        edge_graph.append(torch.full((edges.shape[1],), graph_number, dtype=torch.long))
        graph_ids.append(str(record.get("graph_id", graph_number)))
        steps.append(float(record.get("step_size", 0.02)))
        metadata.append(dict(record.get("metadata", {})))
        for name in optional_names:
            value = record.get(name)
            if value is not None:
                optional[name].append(value)
        node_offset += int(state.shape[0])
    for name, values in optional.items():
        if values and len(values) != len(records):
            raise ValueError(f"optional target {name!r} must be present for every example")
    packed_optional = {
        name: torch.cat(values, dim=0) if values else None for name, values in optional.items()
    }
    return PackedGraphBatch(
        node_state=torch.cat(node_states, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat(edge_features, dim=0),
        node_graph=torch.cat(node_graph, dim=0),
        edge_graph=torch.cat(edge_graph, dim=0),
        graph_ids=graph_ids,
        requested_step=torch.tensor(steps, dtype=node_states[0].dtype),
        metadata=metadata,
        **packed_optional,
    )


class SparsePositiveConductance(nn.Module):
    """Positive orientation-invariant full, static, or gradient-only edge law."""

    def __init__(
        self,
        channels: int,
        edge_feature_channels: int,
        hidden_channels: int = 48,
        minimum: float = 1.0e-5,
        mode: str = "full",
    ) -> None:
        super().__init__()
        if mode not in {"full", "edge_only", "gradient_only"}:
            raise ValueError("mode must be full, edge_only, or gradient_only")
        if channels < 1 or edge_feature_channels < 0 or hidden_channels < 1 or minimum <= 0:
            raise ValueError("invalid conductance dimensions or minimum")
        self.channels = int(channels)
        self.edge_feature_channels = int(edge_feature_channels)
        self.minimum = float(minimum)
        self.mode = mode
        if mode == "full":
            width = edge_feature_channels + 2 * channels
        elif mode == "gradient_only":
            width = channels
        else:
            width = edge_feature_channels
        if width == 0:
            raise ValueError("edge_only conductance requires edge features")
        self.network = nn.Sequential(
            nn.Linear(width, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, gradient: Tensor, edge_features: Tensor) -> Tensor:
        if gradient.ndim != 2 or gradient.shape[1] != self.channels:
            raise ValueError("gradient width differs from configured channels")
        if edge_features.shape != (gradient.shape[0], self.edge_feature_channels):
            raise ValueError("edge feature shape differs from configured shape")
        pieces = [edge_features]
        if self.mode == "full":
            pieces = [gradient.abs(), gradient.square(), edge_features]
        elif self.mode == "gradient_only":
            pieces = [gradient.abs()]
        raw = self.network(torch.cat(pieces, dim=-1))
        return nnf.softplus(raw).squeeze(-1) + self.minimum


class SparseIncidenceConductanceLayer(nn.Module):
    """Dense-``B``-free ``H - eta B.T C B H`` on packed variable graphs."""

    def __init__(
        self,
        channels: int,
        edge_feature_channels: int,
        hidden_channels: int = 48,
        minimum_conductance: float = 1.0e-5,
        requested_step: float = 0.02,
        stability_margin: float = 0.95,
        adaptive_stability: bool = True,
        mode: str = "full",
        initial_isotropic: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"full", "edge_only", "gradient_only", "isotropic"}:
            raise ValueError("mode must be full, edge_only, gradient_only, or isotropic")
        if requested_step <= 0 or not 0 < stability_margin < 1:
            raise ValueError("requested_step and stability_margin are invalid")
        self.channels = int(channels)
        self.edge_feature_channels = int(edge_feature_channels)
        self.requested_step = float(requested_step)
        self.stability_margin = float(stability_margin)
        self.adaptive_stability = bool(adaptive_stability)
        self.mode = mode
        self.minimum_conductance = float(minimum_conductance)
        if mode == "isotropic":
            if initial_isotropic <= minimum_conductance:
                raise ValueError("initial isotropic value must exceed the minimum")
            raw = _inverse_softplus(initial_isotropic - minimum_conductance)
            self.raw_isotropic = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
            self.estimator = None
        else:
            self.estimator = SparsePositiveConductance(
                channels,
                edge_feature_channels,
                hidden_channels,
                minimum_conductance,
                mode,
            )

    @property
    def isotropic_conductance(self) -> Tensor:
        if self.mode != "isotropic":
            raise AttributeError("only the isotropic baseline has a scalar conductance")
        return nnf.softplus(self.raw_isotropic) + self.minimum_conductance

    def forward(
        self,
        batch: PackedGraphBatch,
        *,
        node_state: Tensor | None = None,
        conductance_override: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        state = batch.node_state if node_state is None else node_state
        if state.ndim != 2 or state.shape != batch.node_state.shape:
            raise ValueError("node_state must match the packed batch shape")
        if state.shape[1] != self.channels:
            raise ValueError("node-state width differs from configured channels")
        gradient = edge_gradient(batch.edge_index, state)
        if conductance_override is not None:
            conductance = conductance_override.to(device=state.device, dtype=state.dtype).reshape(
                -1
            )
            if conductance.shape[0] != batch.num_edges:
                raise ValueError("conductance_override needs one scalar per edge")
        elif self.mode == "isotropic":
            conductance = self.isotropic_conductance.to(state).expand(batch.num_edges)
        else:
            assert self.estimator is not None
            conductance = self.estimator(gradient, batch.edge_features.to(state))
        flux = conductance[:, None] * gradient
        message = edge_divergence(batch.edge_index, flux, batch.num_nodes)
        degree = weighted_degree(batch.edge_index, conductance, batch.num_nodes)
        max_degree = _scatter_graph_max(degree, batch.node_graph, batch.num_graphs)
        requested = batch.requested_step.to(state)
        if requested.numel() != batch.num_graphs:
            requested = state.new_full((batch.num_graphs,), self.requested_step)
        if self.adaptive_stability:
            safe = self.stability_margin / max_degree.clamp_min(torch.finfo(state.dtype).eps)
            step = torch.minimum(requested, safe)
        else:
            step = requested
        next_state = state - step.index_select(0, batch.node_graph)[:, None] * message
        if not return_diagnostics:
            return next_state
        return next_state, {
            "edge_gradient": gradient,
            "conductance": conductance,
            "edge_flux": flux,
            "node_message": message,
            "effective_step": step,
            "cap_active": step < requested,
            "max_weighted_degree": max_degree,
        }


__all__ = [
    "PackedGraphBatch",
    "SparseIncidenceConductanceLayer",
    "SparsePositiveConductance",
    "edge_divergence",
    "edge_gradient",
    "pack_graph_examples",
    "weighted_degree",
]
````

# research/conductance_gat/synthetic.py

````python
"""Known-potential, multiple-excitation conductance benchmark."""

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
    from .model import IncidenceConductanceAttention


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
````

# research/conductance_gat/tests/test_conductance_gat.py

````python
from __future__ import annotations

import torch

from research.conductance_gat.model import (
    IncidenceConductanceAttention,
    IsotropicConductanceAttention,
    PositiveInvariantScalarConductance,
    frozen_operator_spectral_radius,
)
from research.conductance_gat.run import _training_loss, resolve_device, runtime_metadata
from research.conductance_gat.synthetic import evaluate_model, make_conductance_dataset


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
    metadata = runtime_metadata(torch.device("cpu"))
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
    loss, _ = _training_loss(
        model,
        dataset,
        node_weight=1.0,
        flux_weight=1.0,
        conductance_weight=0.0,
    )
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
````

# research/conductance_gat/tests/test_paper_pipeline.py

````python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import research.conductance_gat.paper as paper_module
from research.conductance_gat.paper import (
    _normalized_loss,
    _seed_axis_applicability,
    node_message_nnls_metrics,
)
from research.conductance_gat.paper import main as paper_main
from research.conductance_gat.paper_data import (
    _expected_split_counts,
    generate_core,
    make_example,
    prepare_core_cache,
)
from research.conductance_gat.public_data import (
    deduplicate_undirected_edges,
    make_public_fixtures,
    prepare_public_data,
)
from research.conductance_gat.sparse import (
    SparseIncidenceConductanceLayer,
    edge_divergence,
    edge_gradient,
    pack_graph_examples,
)


def _explicit_incidence(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    incidence = torch.zeros(edge_index.shape[1], num_nodes, dtype=torch.float64)
    incidence[torch.arange(edge_index.shape[1]), edge_index[0]] = -1.0
    incidence[torch.arange(edge_index.shape[1]), edge_index[1]] = 1.0
    return incidence


def test_sparse_gather_scatter_matches_dense_algebra_only_in_reference_test() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    state = torch.randn(4, 3, dtype=torch.float64)
    flux = torch.randn(4, 3, dtype=torch.float64)
    incidence = _explicit_incidence(edge_index, 4)

    assert torch.allclose(edge_gradient(edge_index, state), incidence @ state)
    assert torch.allclose(edge_divergence(edge_index, flux, 4), incidence.t() @ flux)


def test_variable_graph_sparse_layer_is_positive_and_orientation_invariant() -> None:
    first = make_example(
        graph_id="first",
        num_nodes=7,
        family="er",
        graph_seed=11,
        excitation_seed=12,
    )
    second = make_example(
        graph_id="second",
        num_nodes=9,
        family="rgg",
        graph_seed=21,
        excitation_seed=22,
    )
    batch = pack_graph_examples([first, second])
    assert torch.allclose(
        first["observed_node_message"],
        edge_divergence(first["edge_index"], first["observed_flux"], 7),
    )
    torch.manual_seed(4)
    model = SparseIncidenceConductanceLayer(2, 3, hidden_channels=12).double()
    batch = batch.to(torch.device("cpu"))
    # Keep the generated float input aligned with the double precision model.
    for name, value in list(batch.__dict__.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(batch, name, value.double())
    output, diagnostics = model(batch, return_diagnostics=True)
    assert torch.all(diagnostics["conductance"] > 0)
    for graph_number in range(batch.num_graphs):
        assert torch.allclose(
            diagnostics["node_message"][batch.node_graph == graph_number].sum(dim=0),
            torch.zeros(2, dtype=torch.float64),
            atol=1e-12,
        )

    flipped = dict(first)
    flipped["edge_index"] = first["edge_index"].flip(0)
    original_batch = pack_graph_examples([first])
    flipped_batch = pack_graph_examples([flipped])
    for packed in (original_batch, flipped_batch):
        for name, value in list(packed.__dict__.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                setattr(packed, name, value.double())
    original, original_diagnostics = model(original_batch, return_diagnostics=True)
    reoriented, flipped_diagnostics = model(flipped_batch, return_diagnostics=True)
    assert torch.allclose(original, reoriented, atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        original_diagnostics["edge_flux"],
        -flipped_diagnostics["edge_flux"],
        atol=1e-11,
        rtol=1e-11,
    )

    gradient_only = SparseIncidenceConductanceLayer(
        2, 3, hidden_channels=12, mode="gradient_only"
    ).double()
    _, gradient_diagnostics = gradient_only(original_batch, return_diagnostics=True)
    _, flipped_gradient_diagnostics = gradient_only(flipped_batch, return_diagnostics=True)
    assert torch.all(gradient_diagnostics["conductance"] > 0)
    assert torch.allclose(
        gradient_diagnostics["conductance"],
        flipped_gradient_diagnostics["conductance"],
        atol=1e-11,
        rtol=1e-11,
    )


def test_training_objectives_keep_headline_independent_of_flux_labels() -> None:
    example = make_example(
        graph_id="objective",
        num_nodes=9,
        family="er",
        graph_seed=31,
        excitation_seed=32,
    )
    batch = pack_graph_examples([example])
    model = SparseIncidenceConductanceLayer(2, 3, hidden_channels=8)
    node_before, node_diagnostics = _normalized_loss(model, batch, objective="node_only")
    assert node_diagnostics["flux_relative_mse"] is None
    flux_before, _ = _normalized_loss(model, batch, objective="flux_only")
    assert batch.observed_flux is not None
    batch.observed_flux = batch.observed_flux + 10.0
    node_after, _ = _normalized_loss(model, batch, objective="node_only")
    flux_after, _ = _normalized_loss(model, batch, objective="flux_only")
    assert torch.equal(node_before, node_after)
    assert not torch.isclose(flux_before, flux_after)

    batch.observed_flux = None
    batch.true_flux = None
    node_without_flux, diagnostics = _normalized_loss(model, batch, objective="node_only")
    assert torch.isfinite(node_without_flux)
    assert diagnostics["flux_relative_mse"] is None
    with pytest.raises(ValueError, match="edge-flux target"):
        _normalized_loss(model, batch, objective="flux_only")


def test_node_message_nnls_recovers_static_conductance_without_flux_labels() -> None:
    suite = generate_core(seed=19, tiny=True)["s1"]
    examples = suite["test"]
    for example in examples:
        example["observed_flux"] = torch.full_like(example["observed_flux"], float("nan"))
    metrics = node_message_nnls_metrics(examples)
    assert metrics["protocol"] == ("transductive_same-evaluation-node-messages_nnls_ceiling")
    assert metrics["graph_macro_node_message_relative_l2"] < 1.0e-5
    assert metrics["graph_macro_log_conductance_rmse"] < 1.0e-5
    assert metrics["graph_macro_conductance_pearson"] == pytest.approx(1.0, abs=1.0e-5)


def test_s1_s4_splits_and_factorial_are_leakage_safe() -> None:
    core = generate_core(seed=7, tiny=True)
    s1 = core["s1"]
    train_ids = {example["graph_id"] for example in s1["train"]}
    validation_ids = {example["graph_id"] for example in s1["validation"]}
    test_ids = {example["graph_id"] for example in s1["test"]}
    assert train_ids.isdisjoint(validation_ids | test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert {example["graph_id"] for example in s1["seen_test"]} == train_ids

    s2 = core["s2"]
    assert {example["metadata"]["family"] for example in s2["train"]} == {"er", "rgg"}
    assert {example["metadata"]["family"] for example in s2["test"]} == {"grid", "barbell"}
    assert min(example["metadata"]["num_nodes"] for example in s2["test"]) > max(
        example["metadata"]["num_nodes"] for example in s2["train"]
    )

    s3 = core["s3"]
    assert s3["horizons"] == [1, 5, 10]
    assert all(trajectory["states"].shape[0] == 11 for trajectory in s3["rollout_test"])

    s4 = core["s4"]
    ids = [
        {example["graph_id"] for example in s4[split]} for split in ("train", "validation", "test")
    ]
    assert ids[0].isdisjoint(ids[1] | ids[2]) and ids[1].isdisjoint(ids[2])
    cells = {
        (
            example["metadata"]["contrast"],
            example["metadata"]["active_node_fraction"],
            example["metadata"]["snr_db"],
        )
        for example in s4["test"]
    }
    assert len(cells) == 18


def test_full_s2_cache_cardinality_matches_graph_and_excitation_protocol() -> None:
    # Check the full cache contract without materializing the 52 full-size
    # graphs and their 184 excitation examples.
    expected = _expected_split_counts(tiny=False)["s2"]
    graph_counts = {"train": 28, "validation": 8, "test": 16}
    excitations_per_graph = {"train": 4, "validation": 3, "test": 3}
    assert expected == {
        split: graph_counts[split] * excitations_per_graph[split] for split in graph_counts
    }
    assert expected == {"train": 112, "validation": 24, "test": 48}


def test_cache_manifest_is_deterministic_and_checksum_verified(tmp_path) -> None:
    first, first_path, first_manifest = prepare_core_cache(tmp_path, seed=9, tiny=True)
    second, second_path, second_manifest = prepare_core_cache(tmp_path, seed=9, tiny=True)
    _, _, independent_manifest = prepare_core_cache(
        tmp_path / "independent-root", seed=9, tiny=True
    )
    assert first_path == second_path
    assert first_manifest == second_manifest
    assert first_manifest["content_sha256"] == second_manifest["content_sha256"]
    assert first_manifest == independent_manifest
    assert first["s1"]["train"][0]["graph_id"] == second["s1"]["train"][0]["graph_id"]


def test_public_fixture_uses_same_reciprocal_edge_adapter_without_network(tmp_path) -> None:
    directed = torch.tensor([[0, 1, 1, 0, 2], [1, 0, 1, 2, 0]], dtype=torch.long)
    attributes = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    edges, features = deduplicate_undirected_edges(directed, attributes, 3)
    assert edges.shape == (2, 2)
    assert features.shape == (2, 2)

    reciprocal = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    continuous = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    _, averaged = deduplicate_undirected_edges(reciprocal, continuous, 2)
    torch.testing.assert_close(averaged, torch.tensor([[2.0, 4.0]]))
    categorical = torch.tensor([[1, 2], [1, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="conflicting categorical reciprocal"):
        deduplicate_undirected_edges(reciprocal, categorical, 2)

    fixture = make_public_fixtures(5)
    assert fixture["fixture"] is True
    labels = [int(graph["y"].item()) for graph in fixture["ogbg_molhiv"]["test"]]
    assert set(labels) == {0, 1}
    _, marker, manifest = prepare_public_data(tmp_path, seed=5, tiny=True)
    assert marker.exists() and manifest["fixture"] is True


def test_public_loss_weight_and_inactive_edge_encoders_match_active_computation() -> None:
    fixture = make_public_fixtures(7)
    node_sample = fixture["pascalvoc_sp"]["train"][0]
    assert paper_module._public_loss_weight(node_sample["y"], "node") == node_sample["y"].numel()
    graph_sample = fixture["ogbg_molhiv"]["train"][0]
    assert paper_module._public_loss_weight(graph_sample["y"], "graph") == 1

    gcn = paper_module.PublicConductanceModel(
        node_sample,
        hidden=8,
        num_classes=3,
        official_molecule=False,
        backbone="gcn",
    )
    gat = paper_module.PublicConductanceModel(
        node_sample,
        hidden=8,
        num_classes=3,
        official_molecule=False,
        backbone="gat",
    )
    assert not gcn.uses_edge_features
    assert not any(parameter.requires_grad for parameter in gcn.edge_encoder.parameters())
    assert gat.uses_edge_features
    assert all(parameter.requires_grad for parameter in gat.edge_encoder.parameters())


def test_cli_refuses_nonempty_output_without_touching_existing_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text('{"status":"previous"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        paper_main(
            [
                "--suite",
                "core",
                "--prepare-only",
                "--tiny",
                "--device",
                "cpu",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output),
            ]
        )
    assert sentinel.read_text(encoding="utf-8") == '{"status":"previous"}\n'
    assert list(output.iterdir()) == [sentinel]
    assert not (tmp_path / "data").exists()


def test_tiny_all_cli_writes_machine_readable_results(tmp_path) -> None:
    output = tmp_path / "output"
    summary = paper_main(
        [
            "--suite",
            "all",
            "--tiny",
            "--device",
            "cpu",
            "--no-amp",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--workers",
            "0",
            "--seed",
            "13",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
        ]
    )
    assert summary["runtime"]["amp"] is False
    assert summary["seed_axes"] == {"data": 13, "split": 13, "chart": 13, "model": 13}
    assert summary["results"]["public"]["pascalvoc_sp"]["fixture"] is True
    expected_public_baselines = {
        "no_message_mlp",
        "gcn",
        "gat",
        "gine",
        "conductance_model",
    }
    for dataset_name in ("pascalvoc_sp", "ogbg_molhiv"):
        public_result = summary["results"]["public"][dataset_name]
        assert set(public_result["baselines"]) == expected_public_baselines
        assert all(result["parameter_count"] > 0 for result in public_result["baselines"].values())
        assert public_result["comparison_protocol"]["backbone_depth"] == 1
    assert summary["results"]["core"]["s3"]["baselines"]["oracle"]["rollout"][
        "horizon_10_relative_l2"
    ] == pytest.approx(0.0)
    for suite_name in ("s1", "s2", "s3", "s4"):
        core_result = summary["results"]["core"][suite_name]
        assert core_result["headline_baseline"] == "full"
        assert core_result["baselines"]["full"]["training_objective"] == "node_only"
        assert core_result["baselines"]["full_flux_supervised"]["training_objective"] == "flux_only"
        assert core_result["baselines"]["full_joint"]["training_objective"] == "joint"
        assert core_result["baselines"]["gradient_only"]["training_objective"] == "node_only"
    assert "node_message_nnls" in summary["results"]["core"]["s1"]["baselines"]
    assert "node_message_nnls" in summary["results"]["core"]["s4"]["baselines"]
    assert (output / "summary.json").exists()
    assert (output / "metrics.csv").exists()
    assert (output / "history.csv").exists()
    assert (output / "models.pt").exists()
    history_header = (output / "history.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "training_objective" in history_header
    parsed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert parsed["scope"] == "independent_sparse_incidence_conductance_attention"


def test_explicit_seed_axes_route_data_and_model_randomness_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, int] = {}

    def fake_run_core(core, **kwargs):
        captured["model_seed"] = kwargs["seed"]
        return {}, [], {}

    monkeypatch.setattr(paper_module, "run_core", fake_run_core)
    summary = paper_module.main(
        [
            "--suite",
            "core",
            "--tiny",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--seed",
            "99",
            "--data-seed",
            "3",
            "--split-seed",
            "4",
            "--chart-seed",
            "5",
            "--model-seed",
            "6",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert "seed" not in summary
    assert summary["seed_axes"] == {"data": 3, "split": 4, "chart": 5, "model": 6}
    assert captured["model_seed"] == 6
    core_manifest = json.loads(
        Path(summary["prepared"]["core"]["manifest"]).read_text(encoding="utf-8")
    )
    assert core_manifest["request"]["seed"] == 3
    assert summary["prepared"]["core"]["data_seed"] == 3
    assert summary["seed_axis_applicability"]["core"]["split"]["applicable"] is False
    assert summary["seed_axis_applicability"]["core"]["chart"]["applicable"] is False


def test_official_public_split_and_chart_seed_axes_are_not_applicable() -> None:
    applicability = _seed_axis_applicability("public", public_fixture=False)["public"]
    assert applicability["data"]["applicable"] is False
    assert applicability["split"]["applicable"] is False
    assert "official" in applicability["split"]["use"]
    assert applicability["chart"]["applicable"] is False
    assert applicability["model"]["applicable"] is True
````

# research/cycle_pe/__init__.py

````python
"""Independent research track for topology-only static graph cycle PE."""

from .features import (
    SET_STAT_NAMES,
    cycle_projector,
    cycle_set_statistics,
    degree_only_edge_features,
    projector_leverage_pe,
    raw_padded_basis_pe,
    static_cycle_feature_bundle,
    static_fundamental_basis,
)

__all__ = [
    "SET_STAT_NAMES",
    "cycle_projector",
    "cycle_set_statistics",
    "degree_only_edge_features",
    "projector_leverage_pe",
    "raw_padded_basis_pe",
    "static_cycle_feature_bundle",
    "static_fundamental_basis",
]
````

# research/cycle_pe/cache_validation.py

````python
"""Read-only cache validators used by the repository-level dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from chartgat.cache import CacheCorruptError, CacheIncompleteError

from .paper_adapters import BRECAdapter, find_brec_v3
from .paper_data import sha256_file, validate_cycle_count_ood_cache


def _load_torch_cache(path: Path) -> Any:
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            return torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise CacheCorruptError(f"failed to parse PyG processed cache: {path}") from error


def _pyg_processed_count(path: Path) -> int:
    payload = _load_torch_cache(path)
    if not isinstance(payload, tuple) or len(payload) < 2 or not isinstance(payload[1], dict):
        raise CacheCorruptError(f"unsupported PyG processed-cache layout: {path}")
    slices = payload[1]
    counts: set[int] = set()
    for value in slices.values():
        try:
            count = int(len(value)) - 1
        except TypeError as error:
            raise CacheCorruptError(f"invalid PyG slice table: {path}") from error
        if count >= 0:
            counts.add(count)
    if len(counts) != 1:
        raise CacheCorruptError(f"inconsistent PyG split cardinality: {path}")
    return counts.pop()


def _validate_zinc(data_root: Path, *, tiny: bool) -> dict[str, Any]:
    processed = data_root.expanduser().resolve() / "ZINC12K" / "subset" / "processed"
    paths = {split: processed / f"{split}.pt" for split in ("train", "val", "test")}
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        raise FileNotFoundError(f"Cycle PE ZINC processed cache is missing: {processed}")
    if not all(present.values()):
        missing = [name for name, exists in present.items() if not exists]
        raise CacheIncompleteError(f"Cycle PE ZINC processed splits are missing: {missing}")
    counts = {name: _pyg_processed_count(path) for name, path in paths.items()}
    expected = {"train": 10_000, "val": 1_000, "test": 1_000}
    if not tiny and counts != expected:
        raise CacheCorruptError(f"Cycle PE ZINC official split cardinalities are invalid: {counts}")
    minimum = {"train": 32, "val": 8, "test": 8}
    if tiny and any(counts[name] < minimum[name] for name in counts):
        raise CacheCorruptError(f"Cycle PE ZINC cache is too small for tiny validation: {counts}")
    return {
        "paths": [str(path) for path in paths.values()],
        "split_sizes": counts,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def _validate_brec(data_root: Path, *, tiny: bool) -> dict[str, Any]:
    if tiny:
        path = data_root.expanduser().resolve() / "cycle_pe_fixtures" / "brec_v3_q32.npy"
        if not path.is_file():
            raise FileNotFoundError(f"tiny BREC fixture is missing: {path}")
        expected_pairs = 2
    else:
        path = find_brec_v3(data_root)
        expected_pairs = 400
    try:
        adapter = BRECAdapter(
            path,
            num_relabel=32,
            fixture=tiny,
            protocol="custom" if tiny else "official",
        )
    except RuntimeError as error:
        raise CacheCorruptError(f"invalid BREC cache: {path}") from error
    if adapter.pair_count != expected_pairs:
        raise CacheCorruptError(
            f"BREC pair cardinality must be {expected_pairs}, got {adapter.pair_count}"
        )
    for pair_index in range(adapter.pair_count):
        try:
            adapter.load_pair(pair_index)
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise CacheCorruptError(f"BREC graph6 decode failed at pair {pair_index}") from error
    return {
        "paths": [str(path)],
        "pair_count": adapter.pair_count,
        "records": int(adapter.metadata["records"]),
        "sha256": adapter.metadata["sha256"],
    }


def validate_dataset_cache(
    dataset_id: str,
    data_root: Path,
    *,
    data_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    tiny: bool,
) -> dict[str, Any]:
    """Validate every requested cycle cache without generating or downloading data."""

    del split_seeds
    if dataset_id == "cyclecount_ood":
        paths = []
        for seed in data_seeds:
            bundle = validate_cycle_count_ood_cache(data_root, seed=seed, tiny=tiny)
            if bundle.cache_path is not None:
                paths.append(str(bundle.cache_path))
        return {"paths": paths, "requested_data_seeds": list(data_seeds)}
    if dataset_id == "brec_v3":
        return _validate_brec(data_root, tiny=tiny)
    if dataset_id == "zinc12k":
        return _validate_zinc(data_root, tiny=tiny)
    raise ValueError(f"unsupported cycle cache dataset {dataset_id!r}")


__all__ = ["validate_dataset_cache"]
````

# research/cycle_pe/config.yaml

````yaml
# Independent static cycle-PE probe.  All PE tensors are computed once from
# topology before the linear probe is fitted.
seed: 7
samples_per_family: 8
max_cycles: 12

probe:
  steps: 1200
  learning_rate: 0.15
  l2: 0.001

output: results/summary.json
````

# research/cycle_pe/datasets.yaml

````yaml
registry_version: 2
track: cycle_pe
paper_suite_complete: true
claim: Static cycle-space PE is evaluated on independent cycle-count targets, official BREC RPC, and ZINC-12K.

datasets:
  - id: cycle_membership_smoke
    name: Bridge versus any-cycle smoke probe
    tier: smoke
    status: implemented
    data_policy: generated
    source_url: generated://research.cycle_pe.synthetic
    task: Classify whether an edge belongs to any cycle across small graph families.
    split: 16 training and 16 test graphs from disjoint generator families.
    metrics: [balanced_accuracy, f1, auroc, graph_macro_accuracy]
    claim: Feature, batching, and invariance pipeline certification only.
    adapter: research.cycle_pe.synthetic.make_graph_family_split
    leakage_guard: Projector leverage directly reveals this target, so it is excluded from headline evidence.

  - id: cyclecount_ood
    name: CycleCount-OOD
    tier: paper_core
    status: implemented
    data_policy: generated
    cache_glob: cycle_count_ood/*.json.gz
    source_url: generated://research.cycle_pe.paper_data/cycle-count-ood-v4
    task: Independently predict edge, node, or graph C3-C6 counts plus edge shortest-cycle length and congestion.
    split: 10000 train, 2000 validation, 2000 ID, 3000 size-OOD, and 3000 family-OOD graphs.
    metrics: [mae, rmse, normalized_mae, graph_macro_mae, rounded_exact_accuracy]
    claim: Nontrivial cycle composition with size and graph-family extrapolation.
    adapter: research.cycle_pe.paper_data.load_or_generate_cycle_count_ood
    validator: research.cycle_pe.cache_validation.validate_dataset_cache
    leakage_guard: Edge, node, and graph targets use separate models; raw width is fit on train only and never test-fit or truncated.
    protocol_coverage: Size-OOD and family-OOD are implemented; degree-sequence-matched counterfactuals are not implemented.

  - id: brec_v3
    name: BREC v3
    tier: paper_core
    status: implemented
    data_policy: download
    cache_glob: BREC/Data/raw/brec_v3.npy
    source_url: https://github.com/GraphPKU/BREC
    task: Distinguish 400 non-isomorphic graph pairs under the official RPC protocol.
    split: Official Basic, Regular, Extension, CFI, 4-Vertex-Condition, and Distance-Regular categories with ten official search seeds.
    metrics: [per_seed_correct, per_seed_fail, per_seed_real_correct, global_valid, hotelling_t2]
    claim: Expressivity beyond ordinary message passing with reliability-gated results.
    adapter: python -m research.cycle_pe.paper --suite brec
    validator: research.cycle_pe.cache_validation.validate_dataset_cache
    leakage_guard: Official mode reports ten independent complete-seed runs and is globally valid only with zero reliability failures; any-seed union is custom-only.

  - id: zinc12k
    name: ZINC-12K
    tier: paper_core
    status: implemented
    data_policy: download
    cache_glob: ZINC12K/subset/processed/*.pt
    source_url: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html
    task: Molecular constrained-solubility graph regression with official PyG subset partitions.
    split: Official 10000/1000/1000 train, validation, and test split.
    metrics: [mae, rmse, normalized_mae, graph_macro_mae]
    claim: Real molecular graph utility under a common static-cycle PE backbone.
    adapter: python -m research.cycle_pe.paper --suite zinc
    validator: research.cycle_pe.cache_validation.validate_dataset_cache
    leakage_guard: Chemistry features and topology PE are fixed before training; raw PE width is fit on train only.

  - id: aqsol
    name: AQSOL scaffold OOD
    tier: conditional
    status: blocked
    data_policy: download
    source_url: https://www.jmlr.org/papers/v24/22-0567.html
    task: Measured aqueous-solubility graph regression.
    split: Official scaffold 7831/996/996 split.
    metrics: [mae, scaffold_ood_gap]
    claim: Required only for a separate scaffold-OOD headline claim.
    adapter: No AQSOL adapter is implemented in this track.
    leakage_guard: Do not replace the official scaffold split with a random split.

  - id: peptides
    name: LRGB Peptides-func and Peptides-struct
    tier: optional
    status: planned
    data_policy: download
    source_url: https://github.com/vijaydwivedi75/lrgb
    task: Long-range molecular graph classification and regression.
    split: Official 70/15/15 splits.
    metrics: [average_precision, mae]
    claim: Optional larger-graph scaling and long-range utility.
    adapter: A future LRGB/PyG molecular adapter is planned.
    leakage_guard: Use the official evaluator and report PE preprocessing memory and time.
````

# research/cycle_pe/features.py

````python
"""Static edge positional encodings derived from the graph cycle space.

This module deliberately contains no sample-dependent state and no trainable
operator.  Every feature is a deterministic function of graph topology, an
incidence orientation, and (for chart-dependent variants) a spanning tree.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chartgat.algebra import fundamental_cycle_basis

FloatArray = NDArray[np.float64]

SET_STAT_NAMES = (
    "participation_fraction",
    "magnitude_rms",
    "max_magnitude",
    "mean_cycle_length_fraction",
    "min_cycle_length_fraction",
    "max_cycle_length_fraction",
)


def _as_cycle_basis(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    basis = np.asarray(cycle_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("cycle_basis must have shape (num_edges, cycle_rank)")
    if not np.all(np.isfinite(basis)):
        raise ValueError("cycle_basis must contain only finite values")
    if basis.shape[1] and np.linalg.matrix_rank(basis, tol=atol) != basis.shape[1]:
        raise ValueError("cycle_basis must have full column rank")
    return basis


def static_fundamental_basis(
    incidence: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
) -> FloatArray:
    """Return the fixed fundamental cycle basis for one graph and tree.

    The output is topology-only.  It is intentionally exposed because raw
    fundamental-basis columns are a useful diagnostic PE, even though their
    ordering and signs are chart conventions rather than intrinsic structure.
    """

    return np.asarray(fundamental_cycle_basis(incidence, tree_edge_indices), dtype=np.float64)


def raw_padded_basis_pe(cycle_basis: ArrayLike, max_cycles: int) -> FloatArray:
    """Pad raw fundamental-cycle columns to a common feature width.

    This representation is *not* invariant to cycle-column permutation, sign,
    or spanning-tree choice.  It is included as a transparent diagnostic, not
    as an intrinsic graph PE.
    """

    basis = _as_cycle_basis(cycle_basis)
    if not isinstance(max_cycles, (int, np.integer)) or max_cycles < 0:
        raise ValueError("max_cycles must be a non-negative integer")
    if basis.shape[1] > max_cycles:
        raise ValueError(f"cycle rank {basis.shape[1]} exceeds padded width {max_cycles}")
    padded = np.zeros((basis.shape[0], int(max_cycles)), dtype=np.float64)
    padded[:, : basis.shape[1]] = basis
    return padded


def cycle_set_statistics(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    """Compute per-edge statistics invariant to column signs and permutations.

    The six output columns are listed in :data:`SET_STAT_NAMES`.  These
    statistics summarize a *chosen fundamental cycle set*.  They are invariant
    to reordering or independently flipping that set's columns, but they are
    not claimed to be invariant to replacing the spanning tree.
    """

    basis = _as_cycle_basis(cycle_basis, atol=atol)
    edge_count, cycle_rank = basis.shape
    if cycle_rank == 0:
        return np.zeros((edge_count, len(SET_STAT_NAMES)), dtype=np.float64)

    magnitude = np.abs(basis)
    membership = magnitude > atol
    cycle_lengths = membership.sum(axis=0).astype(np.float64)
    participation = membership.sum(axis=1)
    nonempty = participation > 0

    features = np.zeros((edge_count, len(SET_STAT_NAMES)), dtype=np.float64)
    features[:, 0] = participation / cycle_rank
    features[:, 1] = np.sqrt(np.mean(np.square(magnitude), axis=1))
    features[:, 2] = magnitude.max(axis=1)

    weighted_lengths = membership @ cycle_lengths
    features[nonempty, 3] = (
        weighted_lengths[nonempty] / participation[nonempty] / max(1, edge_count)
    )
    for edge_index in np.flatnonzero(nonempty):
        lengths = cycle_lengths[membership[edge_index]] / max(1, edge_count)
        features[edge_index, 4] = float(lengths.min())
        features[edge_index, 5] = float(lengths.max())
    return features


def cycle_projector(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    r"""Return the orthogonal projector onto :math:`\ker(B^\top)`.

    For any full-rank cycle basis ``F``, this is

    ``P_cycle = F (F.T F)^(-1) F.T``.

    Unlike raw fundamental columns, the result is invariant to every invertible
    change of cycle basis.  Projector-based cycle PE is established prior-style
    methodology and is treated only as a baseline in this research track.
    """

    basis = _as_cycle_basis(cycle_basis, atol=atol)
    edge_count, cycle_rank = basis.shape
    if cycle_rank == 0:
        return np.zeros((edge_count, edge_count), dtype=np.float64)
    gram = basis.T @ basis
    projector = basis @ np.linalg.solve(gram, basis.T)
    # Suppress harmless asymmetry from floating-point linear solves.
    return 0.5 * (projector + projector.T)


def projector_leverage_pe(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    """Return the projector diagonal as one static feature per edge.

    The leverage is zero exactly on bridges (up to numerical precision) and is
    invariant to cycle-basis coordinates and incidence orientation.
    """

    leverage = np.diag(cycle_projector(cycle_basis, atol=atol)).copy()
    leverage[np.abs(leverage) < atol] = 0.0
    return leverage[:, None]


def degree_only_edge_features(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> FloatArray:
    """Build a small topology baseline from unordered endpoint degrees."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    degrees = np.zeros(num_nodes, dtype=np.float64)
    for u, v in edges:
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("edge endpoint out of range")
        if u == v:
            degrees[u] += 2.0
        else:
            degrees[u] += 1.0
            degrees[v] += 1.0

    rows: list[list[float]] = []
    scale = max(1.0, float(num_nodes - 1))
    for u, v in edges:
        low, high = sorted((degrees[u], degrees[v]))
        rows.append(
            [
                low / scale,
                high / scale,
                (low + high) / (2.0 * scale),
                abs(high - low) / scale,
            ]
        )
    return np.asarray(rows, dtype=np.float64).reshape(len(edges), 4)


def static_cycle_feature_bundle(
    incidence: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
    *,
    max_cycles: int,
) -> dict[str, FloatArray]:
    """Construct all static cycle-PE variants for one graph."""

    basis = static_fundamental_basis(incidence, tree_edge_indices)
    return {
        "basis": basis,
        "raw_padded": raw_padded_basis_pe(basis, max_cycles),
        "cycle_set": cycle_set_statistics(basis),
        "projector_leverage": projector_leverage_pe(basis),
    }


__all__ = [
    "SET_STAT_NAMES",
    "cycle_projector",
    "cycle_set_statistics",
    "degree_only_edge_features",
    "projector_leverage_pe",
    "raw_padded_basis_pe",
    "static_cycle_feature_bundle",
    "static_fundamental_basis",
]
````

# research/cycle_pe/paper.py

````python
"""Linux/CUDA paper entry point for the static cycle-PE track.

Examples
--------
python -m research.cycle_pe.paper --suite core --data-root data --output-dir runs/cycle \
    --device cuda --seed 2025
python -m research.cycle_pe.paper --suite core --data-root data --output-dir /tmp/cycle \
    --device cpu --seed 7 --tiny
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from chartgat.seeds import SeedAxes, resolve_seed_axes
from research.cycle_pe.paper_adapters import (
    BREC_CATEGORIES,
    BREC_OFFICIAL_NUM_RELABEL,
    BREC_OFFICIAL_PAIR_COUNT,
    BREC_OFFICIAL_RECORD_COUNT,
    BREC_SOURCE_URL,
    BRECAdapter,
    load_brec_v3,
    load_zinc12k,
)
from research.cycle_pe.paper_data import (
    GENERATOR_VERSION,
    DatasetBundle,
    load_or_generate_cycle_count_ood,
    sha256_file,
)
from research.cycle_pe.paper_model import (
    PE_VARIANTS,
    PaperCycleModel,
    PreparedGraph,
    prepare_splits,
)
from research.cycle_pe.paper_train import (
    TrainSettings,
    clone_cpu_state,
    cuda_autocast,
    evaluate_supervised,
    make_grad_scaler,
    resolve_device,
    runtime_environment,
    seed_everything,
    train_supervised,
)

PAPER_SCHEMA_VERSION = 1
BREC_OFFICIAL_SEEDS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
BREC_PROTOCOLS = ("official", "custom")
BREC_OFFICIAL_BATCH_SIZE = 16
BREC_OFFICIAL_EPOCHS = 20
BREC_OFFICIAL_LEARNING_RATE = 1e-4
BREC_OFFICIAL_WEIGHT_DECAY = 1e-4
BREC_OFFICIAL_THRESHOLD = 72.34
BREC_OUTPUT_DIM = 16
COMMAND_CONTRACT = (
    "python -m research.cycle_pe.paper --suite core|brec|zinc|all "
    "--data-root PATH --output-dir PATH --device cuda --seed N "
    "[--data-seed N --split-seed N --chart-seed N --model-seed N] [--workers N] "
    "[--prepare-only] [--tiny] [--allow-download] [--brec-protocol official|custom] "
    "[--brec-seeds 100,...,1000]"
)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _claim_empty_output(path: Path) -> None:
    """Refuse to collide with an existing run before creating any artifact."""

    if path.parent == path:
        raise ValueError("--output-dir cannot be a filesystem root")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"--output-dir is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"--output-dir already contains artifacts; choose a new empty path: {path}"
            )
    else:
        path.mkdir(parents=True)


def _clean_failed_suite_output(path: Path, suite: str | None) -> None:
    """Remove an incomplete suite while preserving every completed suite."""

    if suite is None:
        return
    root = path.resolve()
    target = (root / suite).resolve()
    if target.parent != root or target.name != suite:
        raise RuntimeError(f"refusing to clean unsafe suite output target: {target}")
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    elif target.exists() or target.is_symlink():
        target.unlink(missing_ok=True)


def _artifact_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _argument_manifest(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            result[name] = str(value)
        elif isinstance(value, tuple):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _implementation_hashes() -> dict[str, str]:
    module_root = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(module_root.glob("paper*.py"))
        if path.is_file()
    }


def _split_statistics(bundle: DatasetBundle) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, graphs in bundle.splits.items():
        betas = [graph.beta for graph in graphs]
        result[split] = {
            "graphs": len(graphs),
            "nodes": sum(graph.num_nodes for graph in graphs),
            "edges": sum(len(graph.edges) for graph in graphs),
            "cycle_rank_min": min(betas) if betas else None,
            "cycle_rank_max": max(betas) if betas else None,
            "families": sorted({graph.family for graph in graphs}),
        }
    return result


def _resolve_seed_axes(args: argparse.Namespace) -> SeedAxes:
    return resolve_seed_axes(
        args.seed,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        chart_seed=args.chart_seed,
        model_seed=args.model_seed,
    )


def _seed_axis_policy(
    suite: str,
    axes: SeedAxes,
    *,
    brec_protocol: str | None = None,
    brec_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    if suite == "core":
        return {
            "data": {
                "value": axes.data,
                "used": True,
                "role": "CycleCount generation and content-addressed cache identity",
            },
            "split": {
                "value": axes.split,
                "used": False,
                "status": "not_applicable",
                "reason": (
                    "split families and size regimes are generator-defined; "
                    "data_seed generates each split"
                ),
            },
            "chart": {
                "value": axes.chart,
                "used": False,
                "status": "not_applicable",
                "reason": (
                    "static PE uses a deterministic BFS fundamental basis, not sampled charts"
                ),
            },
            "model": {
                "value": axes.model,
                "used": True,
                "role": "model initialization and supervised minibatch shuffling",
            },
        }
    if suite == "zinc":
        return {
            "data": {
                "value": axes.data,
                "used": False,
                "status": "not_applicable",
                "reason": "ZINC-12K is a fixed public dataset",
            },
            "split": {
                "value": axes.split,
                "used": False,
                "status": "not_applicable",
                "reason": "PyG official train/validation/test partitions are used unchanged",
            },
            "chart": {
                "value": axes.chart,
                "used": False,
                "status": "not_applicable",
                "reason": (
                    "static PE uses a deterministic BFS fundamental basis, not sampled charts"
                ),
            },
            "model": {
                "value": axes.model,
                "used": True,
                "role": "model initialization and supervised minibatch shuffling",
            },
        }
    if suite != "brec":
        raise ValueError(f"unknown seed-axis suite: {suite}")
    return {
        "data": {
            "value": axes.data,
            "used": False,
            "status": "not_applicable",
            "reason": "BREC v3 is a fixed official artifact",
        },
        "split": {
            "value": axes.split,
            "used": False,
            "status": "not_applicable",
            "reason": "BREC uses fixed paired RPC blocks rather than a randomized split",
        },
        "chart": {
            "value": axes.chart,
            "used": False,
            "status": "not_applicable",
            "reason": "static PE uses a deterministic BFS fundamental basis, not sampled charts",
        },
        "model": {
            "value": axes.model,
            "used": False,
            "status": "not_applicable",
            "reason": "outer model_seed is not mixed into BREC protocol seeds",
        },
        "protocol": {
            "name": f"brec_{brec_protocol or 'unspecified'}_search_seed",
            "used": True,
            "values": list(brec_seeds),
            "role": "BREC model initialization/search axis internal to the RPC protocol",
        },
    }


def _settings(args: argparse.Namespace, device: torch.device, suite: str) -> TrainSettings:
    default_epochs = {"core": 60, "zinc": 100, "brec": 20}
    default_lr = {"core": 1e-3, "zinc": 1e-3, "brec": 1e-4}
    default_weight_decay = {"core": 1e-5, "zinc": 1e-5, "brec": 1e-4}
    epochs = args.epochs if args.epochs is not None else (2 if args.tiny else default_epochs[suite])
    return TrainSettings(
        device=device,
        seed=_resolve_seed_axes(args).model,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=(args.learning_rate if args.learning_rate is not None else default_lr[suite]),
        weight_decay=(
            args.weight_decay if args.weight_decay is not None else default_weight_decay[suite]
        ),
        workers=args.workers,
        amp_requested=args.amp,
        pin_memory_requested=args.pin_memory,
        non_blocking_requested=args.non_blocking,
    )


def _effective_brec_protocol(args: argparse.Namespace) -> str:
    requested = getattr(args, "brec_protocol", None)
    if requested is None:
        return "custom" if args.tiny else "official"
    return str(requested)


def _brec_reference_compatibility(protocol: str) -> dict[str, Any]:
    """Describe static upstream compatibility without claiming numerical parity."""

    return {
        "static_constants_and_control_flow_compatible": protocol == "official",
        "compatibility_scope": (
            "q=32, 400 ordered pairs, ten independent search seeds, batch size 16, "
            "20 epochs, Adam lr/weight_decay 1e-4, float32/no-AMP, no pair shuffle, "
            "no gradient clipping, and the upstream T2/reliability predicates"
        ),
        "differential_parity_verified": False,
        "parity_note": (
            "No golden-output or differential run against GraphPKU/BREC has been completed. "
            "Static protocol compatibility must not be interpreted as bytewise or numerical "
            "identity with the upstream runner."
        ),
    }


def _brec_settings(args: argparse.Namespace, device: torch.device, protocol: str) -> TrainSettings:
    if protocol == "custom":
        return _settings(args, device, "brec")
    return TrainSettings(
        device=device,
        seed=0,
        epochs=BREC_OFFICIAL_EPOCHS,
        batch_size=BREC_OFFICIAL_BATCH_SIZE,
        learning_rate=BREC_OFFICIAL_LEARNING_RATE,
        weight_decay=BREC_OFFICIAL_WEIGHT_DECAY,
        workers=0,
        amp_requested=False,
        pin_memory_requested=False,
        non_blocking_requested=False,
    )


def _model_dimensions(args: argparse.Namespace) -> tuple[int, int, int]:
    hidden = args.hidden_dim if args.hidden_dim is not None else (24 if args.tiny else 64)
    pe = args.pe_dim if args.pe_dim is not None else (12 if args.tiny else 32)
    layers = args.layers if args.layers is not None else (2 if args.tiny else 3)
    return hidden, pe, layers


def _normalizer_json(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        level: {"mean": value.mean.tolist(), "std": value.std.tolist()}
        for level, value in stats.items()
    }


def _save_checkpoint(
    path: Path,
    model: PaperCycleModel,
    stats: dict[str, Any],
    *,
    variant: str,
    raw_width: int,
    model_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": PAPER_SCHEMA_VERSION,
            "variant": variant,
            "raw_width": raw_width,
            "model_seed": model_seed,
            "state_dict": clone_cpu_state(model),
            "target_normalization": _normalizer_json(stats),
        },
        path,
    )


def _run_supervised_bundle(
    bundle: DatasetBundle,
    *,
    suite: str,
    suite_root: Path,
    args: argparse.Namespace,
    device: torch.device,
    train_split: str,
    validation_split: str,
    integer_targets: bool,
    target_tasks: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    preparation_started = time.perf_counter()
    prepared, raw_width = prepare_splits(
        bundle.splits,
        fit_split=train_split,
        required_variants=args.variants,
    )
    preparation_seconds = time.perf_counter() - preparation_started
    train_graphs = prepared[train_split]
    first = train_graphs[0]
    hidden_dim, pe_dim, layers = _model_dimensions(args)
    settings = _settings(args, device, suite)
    target_names = {
        "edge": bundle.edge_target_names,
        "node": bundle.node_target_names,
        "graph": bundle.graph_target_names,
    }
    manifest: dict[str, Any] = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "suite": suite,
        "dataset": bundle.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed_axes": seed_axes.to_manifest(),
        "seed_axis_policy": _seed_axis_policy(suite, seed_axes),
        "tiny": args.tiny,
        "prepare_only": args.prepare_only,
        "command_contract": COMMAND_CONTRACT,
        "cli_arguments": _argument_manifest(args),
        "implementation_sha256": _implementation_hashes(),
        "dataset_metadata": bundle.metadata or {},
        "split_statistics": _split_statistics(bundle),
        "split_sizes": {name: len(graphs) for name, graphs in bundle.splits.items()},
        "total_graphs": sum(len(graphs) for graphs in bundle.splits.values()),
        "target_names": target_names,
        "target_tasks": {name: list(levels) for name, levels in target_tasks.items()},
        "target_independence_policy": (
            "Each target level is trained in an independent model; edge/node/graph labels "
            "are never optimized jointly."
        ),
        "raw_width": raw_width,
        "raw_width_policy": (
            f"maximum cycle rank from {train_split!r} only; OOD overflow is reported "
            "and never truncated"
        ),
        "raw_overflow_by_split": {
            split: {
                "graphs": sum(graph.cycle_rank > raw_width for graph in graphs),
                "max_cycle_rank": max((graph.cycle_rank for graph in graphs), default=None),
            }
            for split, graphs in prepared.items()
        },
        "preparation_seconds": preparation_seconds,
        "runtime_environment": runtime_environment(settings),
        "model": {
            "hidden_dim": hidden_dim,
            "pe_dim": pe_dim,
            "layers": layers,
            "node_input_dim": int(first.node_features.shape[1]),
            "edge_input_dim": int(first.edge_features.shape[1]),
        },
        "training": {
            "epochs": settings.epochs,
            "batch_size": settings.batch_size,
            "workers": settings.workers,
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "amp_requested": settings.amp_requested,
            "amp_effective": settings.amp,
            "pin_memory_effective": settings.pin_memory,
            "non_blocking_effective": settings.non_blocking,
        },
    }
    if bundle.cache_path is not None:
        manifest["cache"] = {
            "path": str(bundle.cache_path),
            "sha256": bundle.cache_sha256,
        }

    if args.prepare_only:
        manifest["variants"] = list(args.variants)
        manifest["experiments"] = {}
        manifest["artifacts"] = _artifact_checksums(suite_root)
        _write_json(suite_root / "manifest.json", manifest)
        return manifest

    experiment_summaries: dict[str, Any] = {}
    peak_gpu = 0
    training_wall = 0.0
    for task_name, target_levels in target_tasks.items():
        task_summary: dict[str, Any] = {}
        for variant in args.variants:
            print(f"[{suite}] task={task_name} training variant={variant}", flush=True)
            overflow = {
                split: [graph for graph in graphs if graph.cycle_rank > raw_width]
                for split, graphs in prepared.items()
            }
            validation_graphs = prepared[validation_split]
            validation_fit_note: dict[str, Any] | None = None
            if variant == "raw" and overflow[validation_split]:
                validation_graphs = [
                    graph for graph in validation_graphs if graph.cycle_rank <= raw_width
                ]
                validation_fit_note = {
                    "policy": "compatible_validation_subset_for_early_stopping",
                    "full_split_reported_as": ("not_applicable_train_fitted_width_overflow"),
                    "compatible_graphs": len(validation_graphs),
                    "overflow_graphs": len(overflow[validation_split]),
                }
                if not validation_graphs:
                    summary = {
                        "status": "not_applicable_train_fitted_width_overflow",
                        "reason": "no compatible validation graph for early stopping",
                        "fitted_raw_width": raw_width,
                        "overflow_graphs": len(overflow[validation_split]),
                        "truncated": False,
                    }
                    task_summary[variant] = summary
                    _write_json(suite_root / task_name / variant / "metrics.json", summary)
                    continue
            seed_everything(seed_axes.model)
            enabled = set(target_levels)
            model = PaperCycleModel(
                variant=variant,
                raw_width=raw_width,
                node_input_dim=int(first.node_features.shape[1]),
                edge_input_dim=int(first.edge_features.shape[1]),
                edge_output_dim=(len(bundle.edge_target_names) if "edge" in enabled else 0),
                node_output_dim=(len(bundle.node_target_names) if "node" in enabled else 0),
                graph_output_dim=(len(bundle.graph_target_names) if "graph" in enabled else 0),
                hidden_dim=hidden_dim,
                pe_dim=pe_dim,
                layers=layers,
            )
            model, stats, history, runtime = train_supervised(
                model,
                train_graphs,
                validation_graphs,
                settings,
                target_levels=target_levels,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            evaluation_started = time.perf_counter()
            metrics: dict[str, Any] = {}
            for split, graphs in prepared.items():
                if variant == "raw" and overflow[split]:
                    metrics[split] = {
                        "status": "not_applicable_train_fitted_width_overflow",
                        "fitted_raw_width": raw_width,
                        "overflow_graphs": len(overflow[split]),
                        "max_cycle_rank": max(graph.cycle_rank for graph in overflow[split]),
                        "truncated": False,
                    }
                else:
                    metrics[split] = evaluate_supervised(
                        model,
                        graphs,
                        stats,
                        settings,
                        target_names,
                        integer_targets=integer_targets,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                runtime["peak_gpu_memory_bytes"] = max(
                    int(runtime["peak_gpu_memory_bytes"]),
                    int(torch.cuda.max_memory_allocated(device)),
                )
            runtime["evaluation_wall_seconds"] = time.perf_counter() - evaluation_started
            runtime["total_train_evaluation_wall_seconds"] = float(runtime["wall_seconds"]) + float(
                runtime["evaluation_wall_seconds"]
            )
            if validation_fit_note is not None:
                runtime["raw_validation_fit"] = validation_fit_note
            variant_root = suite_root / task_name / variant
            _write_json(variant_root / "metrics.json", metrics)
            _write_json(variant_root / "history.json", history)
            _write_json(variant_root / "runtime.json", runtime)
            _save_checkpoint(
                variant_root / "model.pt",
                model,
                stats,
                variant=variant,
                raw_width=raw_width,
                model_seed=seed_axes.model,
            )
            peak_gpu = max(peak_gpu, int(runtime["peak_gpu_memory_bytes"]))
            training_wall += float(runtime["total_train_evaluation_wall_seconds"])
            reported_split = "test" if "test" in metrics else "id_test"
            reported = metrics[reported_split]
            task_summary[variant] = {
                "status": reported.get("status", "complete"),
                "reported_split": reported_split,
                "macro_normalized_mae": reported.get("macro_normalized_mae"),
                "best_validation_loss": runtime["best_validation_loss"],
                "total_train_evaluation_wall_seconds": runtime[
                    "total_train_evaluation_wall_seconds"
                ],
                "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        experiment_summaries[task_name] = task_summary

    manifest["variants"] = list(args.variants)
    manifest["experiments"] = experiment_summaries
    manifest["runtime_summary"] = {
        "train_evaluation_wall_seconds_sum": training_wall,
        "peak_gpu_memory_bytes_max": peak_gpu,
    }
    manifest["artifacts"] = _artifact_checksums(suite_root)
    _write_json(suite_root / "manifest.json", manifest)
    return manifest


def run_core(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    bundle = load_or_generate_cycle_count_ood(args.data_root, seed=seed_axes.data, tiny=args.tiny)
    if bundle.metadata is None:
        bundle.metadata = {}
    bundle.metadata.update(
        {
            "source": "built-in deterministic generator",
            "generator_version": GENERATOR_VERSION,
            "cache_identity_seed_axis": "data",
            "data_seed": seed_axes.data,
            "seed_axis_policy": _seed_axis_policy("core", seed_axes),
            "split_protocol": {
                "id_test": "held-out seeds, training graph families and size range",
                "size_ood": "held-out larger node-count range",
                "family_ood": "held-out small-world and local-chord families",
            },
            "protocol_coverage": {
                "size_and_family_ood": True,
                "degree_sequence_matched_counterfactuals": False,
                "note": (
                    "The implemented generator controls size/family but does not claim "
                    "degree-sequence-matched counterfactual coverage."
                ),
            },
        }
    )
    return _run_supervised_bundle(
        bundle,
        suite="core",
        suite_root=args.output_dir / "core",
        args=args,
        device=device,
        train_split="train",
        validation_split="validation",
        integer_targets=True,
        target_tasks={level: (level,) for level in args.core_targets},
    )


def run_zinc(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    bundle = load_zinc12k(args.data_root, tiny=args.tiny, allow_download=args.allow_download)
    if bundle.metadata is None:
        bundle.metadata = {}
    bundle.metadata["seed_axis_policy"] = _seed_axis_policy("zinc", seed_axes)
    return _run_supervised_bundle(
        bundle,
        suite="zinc",
        suite_root=args.output_dir / "zinc",
        args=args,
        device=device,
        train_split="train",
        validation_split="validation",
        integer_targets=False,
        target_tasks={"graph": ("graph",)},
    )


def _brec_batches(
    graphs: list[PreparedGraph],
    order: np.ndarray,
    *,
    batch_size: int,
) -> list[list[PreparedGraph]]:
    pairs_per_batch = max(1, batch_size // 2)
    result: list[list[PreparedGraph]] = []
    for start in range(0, len(order), pairs_per_batch):
        batch: list[PreparedGraph] = []
        for pair_index in order[start : start + pairs_per_batch]:
            index = 2 * int(pair_index)
            batch.extend((graphs[index], graphs[index + 1]))
        result.append(batch)
    return result


def _move_brec_batch(graphs: list[PreparedGraph], settings: TrainSettings) -> list[PreparedGraph]:
    return [graph.to(settings.device, non_blocking=settings.non_blocking) for graph in graphs]


@torch.no_grad()
def brec_hotelling_t2(embeddings: Tensor) -> Tensor:
    """Match GraphPKU/BREC Release/base/test_BREC.py ``T2_calculation``."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 4 or embeddings.shape[0] % 2:
        raise ValueError("BREC embeddings must contain at least two interleaved pairs")
    # The official implementation operates in float32 without an extra q
    # multiplier: D_mean.T @ pinv(cov(D)) @ D_mean.
    matrix = embeddings.float()
    left = matrix[0::2].T
    right = matrix[1::2].T
    difference = left - right
    difference_mean = torch.mean(difference, dim=1).reshape(-1, 1)
    covariance = torch.cov(difference)
    inverse = torch.linalg.pinv(covariance)
    return torch.mm(torch.mm(difference_mean.T, inverse), difference_mean).reshape(())


def brec_rpc_decision(
    train_t2: Tensor | float,
    reliability_t2: Tensor | float,
    *,
    threshold: float,
) -> dict[str, bool]:
    """Apply the official distinguishability and reliability predicates."""

    train = torch.as_tensor(train_t2, dtype=torch.float32).reshape(())
    reliability = torch.as_tensor(reliability_t2, dtype=torch.float32).reshape(())
    distinguished = bool(
        (train > threshold).item() and not torch.isclose(train, reliability, atol=1e-6).item()
    )
    reliable = bool((reliability < threshold).item())
    return {
        "distinguished": distinguished,
        "reliable": reliable,
        "successful": distinguished and reliable,
    }


@torch.no_grad()
def _brec_t2(
    model: PaperCycleModel,
    graphs: list[PreparedGraph],
    settings: TrainSettings,
) -> float:
    model.eval()
    embeddings: list[Tensor] = []
    order = np.arange(len(graphs) // 2)
    for cpu_batch in _brec_batches(graphs, order, batch_size=settings.batch_size):
        batch = _move_brec_batch(cpu_batch, settings)
        with cuda_autocast(settings.amp):
            embeddings.extend(output.embedding for output in model(batch))
    return float(brec_hotelling_t2(torch.stack(embeddings)).cpu())


def _train_brec_pair(
    model: PaperCycleModel,
    train_test: list[PreparedGraph],
    reliability: list[PreparedGraph],
    settings: TrainSettings,
    *,
    threshold: float,
    shuffle_pairs: bool,
    gradient_clip_norm: float | None,
) -> tuple[dict[str, Any], int]:
    model = model.to(settings.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = make_grad_scaler(settings.amp)
    cosine = nn.CosineEmbeddingLoss(margin=0.0)
    if settings.pin_memory:
        train_test = [graph.pin_memory() for graph in train_test]
        reliability = [graph.pin_memory() for graph in reliability]
    if settings.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(settings.device)
        torch.cuda.synchronize(settings.device)
    started = time.perf_counter()
    final_loss = math.inf
    epochs_completed = 0
    for epoch in range(settings.epochs):
        model.train()
        if shuffle_pairs:
            order = np.random.default_rng(settings.seed + epoch).permutation(len(train_test) // 2)
        else:
            order = np.arange(len(train_test) // 2)
        total = 0.0
        pairs = 0
        for cpu_batch in _brec_batches(train_test, order, batch_size=settings.batch_size):
            batch = _move_brec_batch(cpu_batch, settings)
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(settings.amp):
                embedding = torch.stack([output.embedding for output in model(batch)])
                target = -torch.ones(
                    embedding.shape[0] // 2,
                    device=embedding.device,
                    dtype=embedding.dtype,
                )
                loss = cosine(embedding[0::2], embedding[1::2], target)
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            pair_count = embedding.shape[0] // 2
            total += float(loss.detach().cpu()) * pair_count
            pairs += pair_count
        final_loss = total / max(1, pairs)
        epochs_completed = epoch + 1
        scheduler.step(final_loss)
        if final_loss < 0.2:
            break
    train_t2 = _brec_t2(model, train_test, settings)
    reliability_t2 = _brec_t2(model, reliability, settings)
    decision = brec_rpc_decision(train_t2, reliability_t2, threshold=threshold)
    if settings.device.type == "cuda":
        torch.cuda.synchronize(settings.device)
        peak = int(torch.cuda.max_memory_allocated(settings.device))
    else:
        peak = 0
    result = {
        **decision,
        "train_test_t2": train_t2,
        "reliability_t2": reliability_t2,
        "threshold": threshold,
        "final_cosine_loss": final_loss,
        "epochs_completed": epochs_completed,
        "pair_shuffle": shuffle_pairs,
        "gradient_clip_norm": gradient_clip_norm,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak,
    }
    return result, peak


def _aggregate_custom_brec_results(
    results: list[dict[str, Any]],
    *,
    pair_indices: list[int],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Compute the repository's explicitly custom reliable any-seed union."""

    by_pair: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    by_seed: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_pair[int(result["pair_index"])].append(result)
        by_seed[int(result["search_seed"])].append(result)
    pair_summary: list[dict[str, Any]] = []
    for pair_index in pair_indices:
        values = by_pair[pair_index]
        completed = [value for value in values if value.get("status") == "complete"]
        reliability_failures = sum(not bool(value["reliable"]) for value in completed)
        protocol_complete = len(completed) == len(seeds)
        final_success = bool(
            protocol_complete
            and reliability_failures == 0
            and any(bool(value["successful"]) for value in completed)
        )
        pair_summary.append(
            {
                "pair_index": pair_index,
                "category": _category_for_result(values),
                "attempts": len(values),
                "completed_attempts": len(completed),
                "not_applicable_attempts": len(values) - len(completed),
                "distinguished_seeds": sum(bool(value["distinguished"]) for value in completed),
                "successful_seeds": sum(bool(value["successful"]) for value in completed),
                "reliability_failures": reliability_failures,
                "protocol_complete": protocol_complete,
                "successful_pair": final_success,
            }
        )
    category_summary: dict[str, Any] = {}
    grouped_pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in pair_summary:
        grouped_pairs[str(value["category"])].append(value)
    for category, values in grouped_pairs.items():
        category_summary[category] = {
            "pairs": len(values),
            "successful_pairs": sum(bool(value["successful_pair"]) for value in values),
            "success_rate": float(np.mean([bool(value["successful_pair"]) for value in values])),
            "reliability_failures": sum(int(value["reliability_failures"]) for value in values),
        }
    seed_summary = {
        str(seed): {
            "pairs": len(by_seed[seed]),
            "completed_attempts": sum(value.get("status") == "complete" for value in by_seed[seed]),
            "distinguished_attempts": sum(
                bool(value["distinguished"])
                for value in by_seed[seed]
                if value.get("status") == "complete"
            ),
            "successful_attempts": sum(
                bool(value["successful"])
                for value in by_seed[seed]
                if value.get("status") == "complete"
            ),
            "reliability_failures": sum(
                not bool(value["reliable"])
                for value in by_seed[seed]
                if value.get("status") == "complete"
            ),
        }
        for seed in seeds
    }
    successful_pairs = sum(bool(value["successful_pair"]) for value in pair_summary)
    return {
        "protocol": "custom",
        "metric_name": "custom_pairwise_union",
        "pairs": len(pair_indices),
        "seeds": list(seeds),
        "attempts": len(results),
        "successful_pairs": successful_pairs,
        "success_rate": successful_pairs / max(1, len(pair_indices)),
        "reliability_failures": sum(int(value["reliability_failures"]) for value in pair_summary),
        "not_applicable_attempts": sum(
            int(value["not_applicable_attempts"]) for value in pair_summary
        ),
        "final_pair_rule": (
            "at least one seed distinguishes, every configured seed is evaluable, and no "
            "seed fails reliability"
        ),
        "per_pair": pair_summary,
        "per_seed": seed_summary,
        "categories": category_summary,
    }


def _aggregate_official_brec_results(
    results: list[dict[str, Any]],
    *,
    pair_indices: list[int],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Reproduce BREC's per-seed Correct/Fail/Real_correct reporting.

    The official search script launches ten independent complete runs and does
    not define an any-seed union score.  We therefore retain the per-seed
    results and expose only the README's global reliability-validity gate.
    """

    by_seed: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_seed[int(result["search_seed"])].append(result)

    per_seed: dict[str, Any] = {}
    for seed in seeds:
        values = by_seed[seed]
        completed = [value for value in values if value.get("status") == "complete"]
        correct = sum(bool(value["distinguished"]) for value in completed)
        failures = sum(not bool(value["reliable"]) for value in completed)
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in completed:
            grouped[str(value["category"])].append(value)
        categories: dict[str, Any] = {}
        for category, category_values in grouped.items():
            category_correct = sum(bool(value["distinguished"]) for value in category_values)
            category_failures = sum(not bool(value["reliable"]) for value in category_values)
            categories[category] = {
                "pairs": len(category_values),
                "Correct": category_correct,
                "Fail": category_failures,
                "Real_correct": category_correct - category_failures,
            }
        protocol_complete = len(completed) == len(pair_indices)
        per_seed[str(seed)] = {
            "pairs_expected": len(pair_indices),
            "attempts": len(values),
            "completed_attempts": len(completed),
            "not_applicable_attempts": len(values) - len(completed),
            "protocol_complete": protocol_complete,
            "Correct": correct,
            "Fail": failures,
            "Real_correct": correct - failures,
            "categories": categories,
        }

    all_seeds_complete = all(per_seed[str(seed)]["protocol_complete"] for seed in seeds)
    global_valid = bool(
        all_seeds_complete and all(per_seed[str(seed)]["Fail"] == 0 for seed in seeds)
    )
    return {
        "protocol": "official",
        "pairs": len(pair_indices),
        "seeds": list(seeds),
        "attempts": len(results),
        "all_seeds_complete": all_seeds_complete,
        "global_valid": global_valid,
        "global_valid_definition": (
            "repository-defined conservative gate: every configured seed is complete and "
            "has Fail == 0; this is not an upstream BREC metric"
        ),
        "per_seed": per_seed,
        "merged_score": None,
        "score_note": (
            "GraphPKU/BREC test_BREC_search.py emits independent complete runs per seed; "
            "no any-seed pair union is labeled as an official score."
        ),
    }


def _aggregate_brec_results(
    results: list[dict[str, Any]],
    *,
    pair_indices: list[int],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Compatibility alias for the explicitly custom pairwise-union metric."""

    return _aggregate_custom_brec_results(results, pair_indices=pair_indices, seeds=seeds)


def _category_for_result(values: list[dict[str, Any]]) -> str:
    return str(values[0]["category"]) if values else "unknown"


def _brec_model_seed(search_seed: int, pair_index: int) -> int:
    return int((search_seed * 1_000_003 + pair_index) % (2**31 - 1))


def _validate_official_brec_arguments(args: argparse.Namespace) -> None:
    if args.tiny:
        raise ValueError("official BREC mode requires the full 400-pair artifact, not --tiny")
    if args.brec_num_relabel != BREC_OFFICIAL_NUM_RELABEL:
        raise ValueError("official BREC mode requires --brec-num-relabel 32")
    if args.brec_threshold is not None and not math.isclose(
        float(args.brec_threshold), BREC_OFFICIAL_THRESHOLD, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("official BREC mode requires --brec-threshold 72.34")
    if tuple(args.brec_seeds) != BREC_OFFICIAL_SEEDS:
        raise ValueError("official BREC mode requires search seeds 100,200,...,1000 in that order")


def _prepare_brec_pair(
    adapter: BRECAdapter,
    pair_index: int,
    *,
    required_variants: tuple[str, ...],
) -> tuple[str, dict[str, list[PreparedGraph]], int, list[int], list[PreparedGraph]]:
    pair = adapter.load_pair(pair_index)
    prepared, raw_width = prepare_splits(
        {
            "train_test": list(pair.train_test),
            "reliability": list(pair.reliability),
        },
        fit_split="train_test",
        required_variants=required_variants,
    )
    betas = [graph.cycle_rank for graph in prepared["train_test"] + prepared["reliability"]]
    reliability_overflow = [
        graph for graph in prepared["reliability"] if graph.cycle_rank > raw_width
    ]
    return pair.category, prepared, raw_width, betas, reliability_overflow


def _brec_seeded_settings(settings: TrainSettings, seed: int) -> TrainSettings:
    return TrainSettings(
        device=settings.device,
        seed=seed,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        workers=settings.workers,
        amp_requested=settings.amp_requested,
        pin_memory_requested=settings.pin_memory_requested,
        non_blocking_requested=settings.non_blocking_requested,
    )


def _execute_brec_attempt(
    *,
    variant: str,
    pair_index: int,
    category: str,
    prepared: dict[str, list[PreparedGraph]],
    raw_width: int,
    betas: list[int],
    reliability_overflow: list[PreparedGraph],
    search_seed: int,
    rng_seed: int,
    rng_scope: str,
    settings: TrainSettings,
    hidden_dim: int,
    pe_dim: int,
    layers: int,
    threshold: float,
    protocol: str,
) -> tuple[dict[str, Any], int]:
    common = {
        "pair_index": pair_index,
        "category": category,
        "search_seed": search_seed,
        "rng_seed": rng_seed,
        "rng_scope": rng_scope,
        "raw_width": raw_width,
        "cycle_rank_min": min(betas),
        "cycle_rank_max": max(betas),
    }
    if variant == "raw" and reliability_overflow:
        return (
            {
                **common,
                "status": "not_applicable_train_fitted_width_overflow",
                "overflow_graphs": len(reliability_overflow),
                "max_overflow_cycle_rank": max(graph.cycle_rank for graph in reliability_overflow),
                "truncated": False,
                "distinguished": False,
                "reliable": False,
                "successful": False,
            },
            0,
        )

    first = prepared["train_test"][0]
    model = PaperCycleModel(
        variant=variant,
        raw_width=raw_width,
        node_input_dim=int(first.node_features.shape[1]),
        edge_input_dim=int(first.edge_features.shape[1]),
        edge_output_dim=0,
        node_output_dim=0,
        graph_output_dim=0,
        hidden_dim=hidden_dim,
        pe_dim=pe_dim,
        layers=layers,
        embedding_dim=BREC_OUTPUT_DIM,
    )
    result, peak = _train_brec_pair(
        model,
        prepared["train_test"],
        prepared["reliability"],
        _brec_seeded_settings(settings, rng_seed),
        threshold=threshold,
        shuffle_pairs=protocol == "custom",
        gradient_clip_norm=5.0 if protocol == "custom" else None,
    )
    result.update(common)
    result["status"] = "complete"
    del model
    if settings.device.type == "cuda":
        torch.cuda.empty_cache()
    return result, peak


def run_brec(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_axes = _resolve_seed_axes(args)
    protocol = _effective_brec_protocol(args)
    if protocol == "official":
        _validate_official_brec_arguments(args)
    adapter: BRECAdapter = load_brec_v3(
        args.data_root,
        num_relabel=args.brec_num_relabel,
        allow_download=args.allow_download,
        tiny=args.tiny,
        protocol=protocol,
    )
    if protocol == "official":
        threshold = BREC_OFFICIAL_THRESHOLD
    elif args.brec_threshold is None:
        if adapter.num_relabel != 32:
            raise ValueError(
                "The official 72.34 RPC threshold is calibrated for q=32. Pass "
                "--brec-threshold explicitly for a customized --brec-num-relabel."
            )
        threshold = BREC_OFFICIAL_THRESHOLD
    else:
        threshold = float(args.brec_threshold)
    suite_root = args.output_dir / "brec"
    settings = _brec_settings(args, device, protocol)
    hidden_dim, pe_dim, layers = _model_dimensions(args)
    pair_indices = list(range(min(adapter.pair_count, 2 if args.tiny else adapter.pair_count)))
    manifest: dict[str, Any] = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "suite": "brec",
        "dataset": "BREC v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed_axes": seed_axes.to_manifest(),
        "seed_axis_policy": _seed_axis_policy(
            "brec",
            seed_axes,
            brec_protocol=protocol,
            brec_seeds=args.brec_seeds,
        ),
        "tiny": args.tiny,
        "prepare_only": args.prepare_only,
        "command_contract": COMMAND_CONTRACT,
        "cli_arguments": _argument_manifest(args),
        "implementation_sha256": _implementation_hashes(),
        "dataset_metadata": adapter.metadata,
        "brec_protocol": {
            "effective": protocol,
            "default_policy": "official for full runs; custom for --tiny fixtures",
            "official_reference_compatibility": _brec_reference_compatibility(protocol),
            "custom_metric": "custom_pairwise_union" if protocol == "custom" else None,
            "outer_model_seed_used": False,
            "protocol_seed_axis": list(args.brec_seeds),
        },
        "rpc_reference": {
            "num_relabel": adapter.num_relabel,
            "embedding_dim": BREC_OUTPUT_DIM,
            "threshold": threshold,
            "search_seeds": list(args.brec_seeds),
            "t2_formula": "D_mean.T @ torch.linalg.pinv(torch.cov(D)) @ D_mean",
            "q_multiplier": False,
            "distinction_rule": (
                "train_t2 > threshold and not torch.isclose(train_t2, reliability_t2, atol=1e-6)"
            ),
            "reliability_rule": "reliability_t2 < threshold",
            "categories": BREC_CATEGORIES,
            "source_url": BREC_SOURCE_URL,
            "reference_implementation": (
                "https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC.py"
            ),
            "seed_reference": (
                "https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC_search.py"
            ),
        },
        "official_artifact_contract": {
            "required_only_in_official_mode": True,
            "num_relabel": BREC_OFFICIAL_NUM_RELABEL,
            "pair_count": BREC_OFFICIAL_PAIR_COUNT,
            "record_count": BREC_OFFICIAL_RECORD_COUNT,
            "sha256_pinned_by_upstream": False,
            "sha256_policy": "record provenance hash; do not claim an upstream published pin",
        },
        "pairs_selected": pair_indices,
        "runtime_environment": runtime_environment(settings),
        "training": {
            "protocol": protocol,
            "epochs": settings.epochs,
            "batch_size_graphs": settings.batch_size,
            "workers": settings.workers,
            "workers_note": "BREC RPC preserves explicit pairs and does not use DataLoader",
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "amp_effective": settings.amp,
            "pin_memory_effective": settings.pin_memory,
            "non_blocking_effective": settings.non_blocking,
            "pair_shuffle": protocol == "custom",
            "gradient_clip_norm": 5.0 if protocol == "custom" else None,
            "rng_policy": (
                "seed once per variant and official search seed, then traverse all 400 pairs "
                "in order without reseeding"
                if protocol == "official"
                else "derive and reset one model seed per variant, search seed, and pair"
            ),
            "requested_global_overrides": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "amp": args.amp,
                "pin_memory": args.pin_memory,
                "non_blocking": args.non_blocking,
            },
            "official_overrides_global_training_options": protocol == "official",
        },
        "raw_width_policy": (
            "fit on train_test graphs for each RPC pair only; reliability overflow is "
            "not applicable and is never truncated or fitted"
        ),
    }
    # Parse representative complete RPC pairs even in prepare-only mode. This
    # catches malformed graph6, disconnected graphs, and PE extraction failures.
    if args.prepare_only:
        check_indices = pair_indices if args.tiny else [pair_indices[0], pair_indices[-1]]
        checks: list[dict[str, Any]] = []
        for pair_index in check_indices:
            category, prepared, raw_width, _, _ = _prepare_brec_pair(
                adapter,
                pair_index,
                required_variants=args.variants,
            )
            checks.append(
                {
                    "pair": pair_index,
                    "category": category,
                    "graphs": sum(len(graphs) for graphs in prepared.values()),
                    "raw_width": raw_width,
                    "reliability_raw_overflow_graphs": sum(
                        graph.cycle_rank > raw_width for graph in prepared["reliability"]
                    ),
                }
            )
        manifest["preparation_checks"] = checks
        manifest["preparation_check_policy"] = (
            "all selected pairs in tiny mode; first and last pair in full mode"
        )
        manifest["variants"] = list(args.variants)
        manifest["artifacts"] = _artifact_checksums(suite_root)
        _write_json(suite_root / "manifest.json", manifest)
        return manifest

    pair_results: dict[str, list[dict[str, Any]]] = {variant: [] for variant in args.variants}
    peak_gpu = 0
    run_started = time.perf_counter()
    if protocol == "official":
        for variant in args.variants:
            for search_seed in args.brec_seeds:
                seed_everything(search_seed)
                for position, pair_index in enumerate(pair_indices, start=1):
                    print(
                        f"[brec:official] variant={variant} seed={search_seed} "
                        f"pair={pair_index} ({position}/{len(pair_indices)})",
                        flush=True,
                    )
                    category, prepared, raw_width, betas, reliability_overflow = _prepare_brec_pair(
                        adapter,
                        pair_index,
                        required_variants=(variant,),
                    )
                    result, pair_peak = _execute_brec_attempt(
                        variant=variant,
                        pair_index=pair_index,
                        category=category,
                        prepared=prepared,
                        raw_width=raw_width,
                        betas=betas,
                        reliability_overflow=reliability_overflow,
                        search_seed=search_seed,
                        rng_seed=search_seed,
                        rng_scope="variant_search_seed_full_pair_sequence",
                        settings=settings,
                        hidden_dim=hidden_dim,
                        pe_dim=pe_dim,
                        layers=layers,
                        threshold=float(threshold),
                        protocol=protocol,
                    )
                    pair_results[variant].append(result)
                    peak_gpu = max(peak_gpu, pair_peak)
    else:
        for position, pair_index in enumerate(pair_indices, start=1):
            print(f"[brec:custom] pair={pair_index} ({position}/{len(pair_indices)})", flush=True)
            category, prepared, raw_width, betas, reliability_overflow = _prepare_brec_pair(
                adapter,
                pair_index,
                required_variants=args.variants,
            )
            for variant in args.variants:
                for search_seed in args.brec_seeds:
                    model_seed = _brec_model_seed(search_seed, pair_index)
                    seed_everything(model_seed)
                    result, pair_peak = _execute_brec_attempt(
                        variant=variant,
                        pair_index=pair_index,
                        category=category,
                        prepared=prepared,
                        raw_width=raw_width,
                        betas=betas,
                        reliability_overflow=reliability_overflow,
                        search_seed=search_seed,
                        rng_seed=model_seed,
                        rng_scope="derived_per_pair_variant_search_seed",
                        settings=settings,
                        hidden_dim=hidden_dim,
                        pe_dim=pe_dim,
                        layers=layers,
                        threshold=float(threshold),
                        protocol=protocol,
                    )
                    pair_results[variant].append(result)
                    peak_gpu = max(peak_gpu, pair_peak)

    summaries: dict[str, Any] = {}
    for variant, results in pair_results.items():
        if protocol == "official":
            summaries[variant] = _aggregate_official_brec_results(
                results, pair_indices=pair_indices, seeds=args.brec_seeds
            )
        else:
            summaries[variant] = _aggregate_custom_brec_results(
                results, pair_indices=pair_indices, seeds=args.brec_seeds
            )
        _write_json(suite_root / variant / "pairs.json", results)
        _write_json(suite_root / variant / "metrics.json", summaries[variant])
    manifest["variants"] = summaries
    manifest["runtime_summary"] = {
        "wall_seconds": time.perf_counter() - run_started,
        "peak_gpu_memory_bytes_max": peak_gpu,
    }
    manifest["artifacts"] = _artifact_checksums(suite_root)
    _write_json(suite_root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static cycle-PE paper runner (CycleCount-OOD, BREC v3, ZINC-12K)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--suite", choices=("core", "brec", "zinc", "all"), default="core")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_runs/cycle_pe"))
    parser.add_argument("--device", default="cuda", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="CycleCount generation/cache axis; defaults to --seed",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="split axis; recorded as not applicable for current Cycle PE suites",
    )
    parser.add_argument(
        "--chart-seed",
        type=int,
        default=None,
        help="chart axis; static Cycle PE records this as not applicable",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="supervised initialization/minibatch axis; defaults to --seed",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--variants",
        default=",".join(PE_VARIANTS),
        help="comma-separated subset of no_pe,raw,set,projector",
    )
    parser.add_argument(
        "--core-targets",
        default="edge,node,graph",
        help="independent CycleCount target levels selected from edge,node,graph",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers for core/ZINC")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--pe-dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use CUDA autocast and GradScaler (always disabled on CPU)",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--non-blocking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--brec-protocol",
        choices=BREC_PROTOCOLS,
        default=None,
        help="official for full paper runs; omitted --tiny runs resolve to custom",
    )
    parser.add_argument("--brec-num-relabel", type=int, default=32)
    parser.add_argument("--brec-threshold", type=float)
    parser.add_argument(
        "--brec-seeds",
        default=",".join(str(seed) for seed in BREC_OFFICIAL_SEEDS),
        help="comma-separated BREC model-search seeds",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow official BREC/PyG ZINC download when a local cache is absent",
    )
    return parser


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in value.split(",") if part.strip())
    if not variants:
        raise ValueError("--variants cannot be empty")
    unknown = [variant for variant in variants if variant not in PE_VARIANTS]
    if unknown:
        raise ValueError(f"unknown PE variant(s): {', '.join(unknown)}; choose from {PE_VARIANTS}")
    if len(set(variants)) != len(variants):
        raise ValueError("--variants must not contain duplicates")
    return variants


def _parse_core_targets(value: str) -> tuple[str, ...]:
    targets = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = ("edge", "node", "graph")
    if not targets or any(target not in allowed for target in targets):
        raise ValueError(f"--core-targets must be a comma-separated subset of {allowed}")
    if len(set(targets)) != len(targets):
        raise ValueError("--core-targets must not contain duplicates")
    return targets


def _parse_brec_seeds(value: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError("--brec-seeds cannot be empty")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--brec-seeds must contain comma-separated integers") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError("--brec-seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--brec-seeds must not contain duplicates")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_owned = False
    completed: list[str] = []
    selected: tuple[str, ...] = ()
    try:
        args.variants = _parse_variants(args.variants)
        args.core_targets = _parse_core_targets(args.core_targets)
        args.brec_seeds = _parse_brec_seeds(args.brec_seeds)
        args.brec_protocol = _effective_brec_protocol(args)
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        seed_axes = _resolve_seed_axes(args)
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.workers < 0:
            raise ValueError("--workers must be non-negative")
        if args.epochs is not None and args.epochs < 1:
            raise ValueError("--epochs must be positive")
        if args.hidden_dim is not None and args.hidden_dim < 4:
            raise ValueError("--hidden-dim must be at least 4")
        if args.pe_dim is not None and args.pe_dim < 1:
            raise ValueError("--pe-dim must be positive")
        if args.layers is not None and args.layers < 1:
            raise ValueError("--layers must be positive")
        if args.learning_rate is not None and args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        if args.weight_decay is not None and args.weight_decay < 0:
            raise ValueError("--weight-decay must be non-negative")
        if args.brec_num_relabel < 2:
            raise ValueError("--brec-num-relabel must be at least 2")
        if args.brec_threshold is not None and args.brec_threshold <= 0:
            raise ValueError("--brec-threshold must be positive")
        args.data_root = args.data_root.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().resolve()
        device = resolve_device(args.device)
        runners = {
            "core": run_core,
            "brec": run_brec,
            "zinc": run_zinc,
        }
        selected = tuple(runners) if args.suite == "all" else (args.suite,)
        _claim_empty_output(args.output_dir)
        output_owned = True
        started_utc = datetime.now(UTC).isoformat()
        _write_json(
            args.output_dir / "run_manifest.json",
            {
                "schema_version": PAPER_SCHEMA_VERSION,
                "status": "running",
                "started_utc": started_utc,
                "selected_suites": list(selected),
                "completed_suites": [],
                "seed_axes": seed_axes.to_manifest(),
                "cli_arguments": _argument_manifest(args),
            },
        )
        manifests: dict[str, dict[str, Any]] = {}
        for suite in selected:
            manifests[suite] = runners[suite](args, device)
            completed.append(suite)
            _write_json(
                args.output_dir / "run_manifest.json",
                {
                    "schema_version": PAPER_SCHEMA_VERSION,
                    "status": "running",
                    "started_utc": started_utc,
                    "selected_suites": list(selected),
                    "completed_suites": completed,
                    "seed_axes": seed_axes.to_manifest(),
                    "cli_arguments": _argument_manifest(args),
                },
            )
        suite_manifests = {
            suite: {
                "path": str(args.output_dir / suite / "manifest.json"),
                "sha256": sha256_file(args.output_dir / suite / "manifest.json"),
            }
            for suite in completed
            if (args.output_dir / suite / "manifest.json").is_file()
        }
        _write_json(
            args.output_dir / "run_manifest.json",
            {
                "schema_version": PAPER_SCHEMA_VERSION,
                "status": "complete",
                "started_utc": started_utc,
                "completed_utc": datetime.now(UTC).isoformat(),
                "selected_suites": list(selected),
                "completed_suites": completed,
                "suite_manifests": suite_manifests,
                "seed_axes": seed_axes.to_manifest(),
                "cli_arguments": _argument_manifest(args),
            },
        )
    except Exception as exc:
        if output_owned:
            failed_suite = selected[len(completed)] if len(completed) < len(selected) else None
            _clean_failed_suite_output(args.output_dir, failed_suite)
            completed_manifests = {
                suite: {
                    "path": str(args.output_dir / suite / "manifest.json"),
                    "sha256": sha256_file(args.output_dir / suite / "manifest.json"),
                }
                for suite in completed
                if (args.output_dir / suite / "manifest.json").is_file()
            }
            _write_json(
                args.output_dir / "run_manifest.json",
                {
                    "schema_version": PAPER_SCHEMA_VERSION,
                    "status": "failed",
                    "failed_utc": datetime.now(UTC).isoformat(),
                    "selected_suites": list(selected),
                    "completed_suites": completed,
                    "suite_manifests": completed_manifests,
                    "failed_suite": failed_suite,
                    "seed_axes": seed_axes.to_manifest(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        parser.error(str(exc))
    summary = {
        suite: {
            "manifest": str(args.output_dir / suite / "manifest.json"),
            "variants": list(manifest.get("variants", {})),
        }
        for suite, manifest in manifests.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BREC_OFFICIAL_BATCH_SIZE",
    "BREC_OFFICIAL_SEEDS",
    "BREC_PROTOCOLS",
    "COMMAND_CONTRACT",
    "brec_hotelling_t2",
    "brec_rpc_decision",
    "build_parser",
    "main",
    "run_brec",
    "run_core",
    "run_zinc",
]
````

# research/cycle_pe/paper_adapters.py

````python
"""Lazy adapters for the public BREC v3 and ZINC-12K benchmarks.

Neither public dataset is downloaded by the built-in CycleCount-OOD tests.
ZINC is loaded through its official PyTorch Geometric split implementation;
BREC uses the official graph6 ``brec_v3.npy`` artifact and keeps RPC pairs
lazy so a 51,200-graph file is never tensorized all at once.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import networkx as nx
import numpy as np

from chartgat.cache import atomic_publish, atomic_write_json
from research.cycle_pe.paper_data import (
    DatasetBundle,
    PaperGraph,
    canonical_edges,
    sha256_file,
)

BREC_SOURCE_URL = "https://github.com/GraphPKU/BREC"
BREC_RAW_URL = "https://raw.githubusercontent.com/GraphPKU/BREC/Release/BREC_data_all.zip"
ZINC_SOURCE_URL = (
    "https://pytorch-geometric.readthedocs.io/en/latest/generated/"
    "torch_geometric.datasets.ZINC.html"
)
PYG_INSTALL_URL = "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"

BREC_CATEGORIES = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}

BREC_OFFICIAL_NUM_RELABEL = 32
BREC_OFFICIAL_PAIR_COUNT = 400
BREC_OFFICIAL_RECORD_COUNT = 4 * BREC_OFFICIAL_NUM_RELABEL * BREC_OFFICIAL_PAIR_COUNT

_BREC_DOWNLOAD_LIMIT = 512 * 1024 * 1024
_BREC_EXTRACT_LIMIT = 512 * 1024 * 1024
_BREC_ARCHIVE_MEMBER_LIMIT = 10_000
_BREC_ARCHIVE_TOTAL_LIMIT = 1024 * 1024 * 1024
_BREC_DOWNLOAD_HOSTS = {
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "github.com",
}


def _load_brec_records(path: Path) -> np.ndarray:
    try:
        records = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to load BREC graph6 records from {path}") from exc
    if records.ndim != 1 or len(records) < 1:
        raise RuntimeError("BREC artifact must be a non-empty one-dimensional NumPy array")
    return records


def _atomic_save_npy(path: Path, records: np.ndarray) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as stream:
            np.save(stream, records, allow_pickle=True)
            stream.flush()

    atomic_publish(path, write, validator=lambda temporary: _load_brec_records(temporary))


def _require_pyg_zinc() -> type:
    try:
        from torch_geometric.datasets import ZINC
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ZINC-12K is optional and requires PyTorch Geometric. Install a "
            "PyTorch build matching the target CUDA runtime, then run "
            "`python -m pip install torch-geometric`; use the wheel matrix at "
            f"{PYG_INSTALL_URL}. The `core` suite does not require PyG."
        ) from exc
    return ZINC


def _one_hot(values: np.ndarray, width: int, *, name: str) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= width):
        raise RuntimeError(f"unexpected {name} category outside [0, {width - 1}]")
    return np.eye(width, dtype=np.float64)[flat]


def _pyg_zinc_graph(data: Any, *, graph_id: str, split: str) -> PaperGraph:
    num_nodes = int(data.num_nodes)
    node_features = _one_hot(data.x.detach().cpu().numpy(), 28, name="ZINC atom")
    edge_index = data.edge_index.detach().cpu().numpy()
    raw_edge_attr = data.edge_attr.detach().cpu().numpy()
    attributes: dict[tuple[int, int], int] = {}
    for column in range(edge_index.shape[1]):
        u, v = int(edge_index[0, column]), int(edge_index[1, column])
        if u == v:
            continue
        edge = (min(u, v), max(u, v))
        category = int(np.asarray(raw_edge_attr[column]).reshape(-1)[0])
        previous = attributes.setdefault(edge, category)
        if previous != category:
            raise RuntimeError("directed copies of a ZINC bond disagree on bond type")
    edges = tuple(sorted(attributes))
    edge_features = _one_hot(np.asarray([attributes[edge] for edge in edges]), 4, name="ZINC bond")
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    if not nx.is_connected(graph):
        raise RuntimeError(f"ZINC molecule {graph_id} is unexpectedly disconnected")
    target = np.asarray(data.y.detach().cpu().numpy(), dtype=np.float64).reshape(1)
    return PaperGraph(
        graph_id=graph_id,
        split=split,
        family="zinc_molecule",
        num_nodes=num_nodes,
        edges=edges,
        node_features=node_features,
        edge_features=edge_features,
        graph_targets=target,
    )


def _zinc_cache_ready(root: Path) -> bool:
    processed = root / "subset" / "processed"
    processed_ready = all(
        (processed / f"{split}.pt").is_file() for split in ("train", "val", "test")
    )
    raw = root / "raw"
    raw_names = (
        "train.pickle",
        "val.pickle",
        "test.pickle",
        "train.index",
        "val.index",
        "test.index",
    )
    return processed_ready or all((raw / name).is_file() for name in raw_names)


def _zinc_cache_hashes(root: Path) -> dict[str, str]:
    candidates = [
        *(root / "subset" / "processed" / f"{split}.pt" for split in ("train", "val", "test")),
        *(
            root / "raw" / name
            for name in (
                "train.pickle",
                "val.pickle",
                "test.pickle",
                "train.index",
                "val.index",
                "test.index",
            )
        ),
    ]
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }


def load_zinc12k(
    data_root: Path, *, tiny: bool = False, allow_download: bool = False
) -> DatasetBundle:
    """Load PyG's official 10k/1k/1k ZINC subset partitions."""

    zinc_class = _require_pyg_zinc()
    root = data_root.expanduser().resolve() / "ZINC12K"
    if not allow_download and not _zinc_cache_ready(root):
        raise FileNotFoundError(
            f"No complete PyG ZINC-12K cache was found at {root}. Copy an existing "
            "official cache there, or explicitly permit the PyG download with "
            f"`--allow-download`. Loader documentation: {ZINC_SOURCE_URL}"
        )
    requested = {"train": "train", "validation": "val", "test": "test"}
    tiny_limits = {"train": 32, "validation": 8, "test": 8}
    splits: dict[str, list[PaperGraph]] = {}
    official_sizes: dict[str, int] = {}
    try:
        for split, pyg_split in requested.items():
            dataset = zinc_class(root=str(root), subset=True, split=pyg_split)
            official_sizes[split] = len(dataset)
            limit = min(len(dataset), tiny_limits[split]) if tiny else len(dataset)
            splits[split] = [
                _pyg_zinc_graph(
                    dataset[index], graph_id=f"zinc12k:{split}:{index:05d}", split=split
                )
                for index in range(limit)
            ]
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to prepare ZINC-12K at {root}. Ensure outbound access for the "
            "first PyG download or copy an existing PyG ZINC cache there. Official "
            f"loader documentation: {ZINC_SOURCE_URL}. Original error: {exc}"
        ) from exc
    return DatasetBundle(
        name="ZINC-12K",
        splits=splits,
        graph_target_names=("constrained_solubility",),
        metadata={
            "adapter": "torch_geometric.datasets.ZINC(subset=True)",
            "source_url": ZINC_SOURCE_URL,
            "official_split_names": requested,
            "official_split_sizes": official_sizes,
            "loaded_split_sizes": {name: len(graphs) for name, graphs in splits.items()},
            "tiny": bool(tiny),
            "download_allowed": bool(allow_download),
            "cache_sha256": _zinc_cache_hashes(root),
        },
    )


def _category(pair_index: int) -> str:
    for name, (start, stop) in BREC_CATEGORIES.items():
        if start <= pair_index < stop:
            return name
    return "custom"


def _decode_graph6(record: Any) -> nx.Graph:
    value = record.item() if isinstance(record, np.ndarray) and record.ndim == 0 else record
    if isinstance(value, str):
        payload = value.encode("ascii")
    elif isinstance(value, (bytes, np.bytes_)):
        payload = bytes(value)
    else:
        raise RuntimeError(f"unsupported BREC graph6 record type: {type(value).__name__}")
    try:
        graph = nx.from_graph6_bytes(payload.strip())
    except (nx.NetworkXError, ValueError) as exc:
        raise RuntimeError("invalid graph6 record in brec_v3.npy") from exc
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
        raise RuntimeError("the static paper model currently requires connected BREC graphs")
    return graph


def _brec_graph(record: Any, *, graph_id: str, family: str) -> PaperGraph:
    graph = _decode_graph6(record)
    return PaperGraph(
        graph_id=graph_id,
        split="brec_rpc",
        family=family,
        num_nodes=graph.number_of_nodes(),
        edges=canonical_edges(graph.edges()),
    )


@dataclass(frozen=True)
class BRECPair:
    pair_index: int
    category: str
    train_test: tuple[PaperGraph, ...]
    reliability: tuple[PaperGraph, ...]


class BRECAdapter:
    """Lazy view of the official RPC layout (G/H and G/G permutation pairs)."""

    def __init__(
        self,
        path: Path,
        *,
        num_relabel: int = BREC_OFFICIAL_NUM_RELABEL,
        fixture: bool = False,
        protocol: str = "custom",
    ) -> None:
        if num_relabel < 2:
            raise ValueError("BREC RPC needs at least two relabelings")
        if protocol not in {"official", "custom"}:
            raise ValueError("BREC protocol must be 'official' or 'custom'")
        self.path = path.expanduser().resolve()
        self.num_relabel = int(num_relabel)
        self.fixture = bool(fixture)
        self.protocol = protocol
        self._records = _load_brec_records(self.path)
        block = 4 * self.num_relabel
        if len(self._records) % block:
            raise RuntimeError(
                f"BREC artifact must contain 4*q records per pair (q={self.num_relabel})"
            )
        self.pair_count = len(self._records) // block
        if self.pair_count < 1:
            raise RuntimeError("BREC artifact contains no RPC pairs")
        if self.protocol == "official":
            if self.fixture:
                raise RuntimeError("offline BREC fixtures cannot be used in official mode")
            if self.num_relabel != BREC_OFFICIAL_NUM_RELABEL:
                raise RuntimeError(
                    "official BREC requires q=32; use --brec-protocol custom for other q values"
                )
            if len(self._records) != BREC_OFFICIAL_RECORD_COUNT:
                raise RuntimeError(
                    "official BREC v3 requires exactly "
                    f"{BREC_OFFICIAL_RECORD_COUNT:,} records, found {len(self._records):,}"
                )
            if self.pair_count != BREC_OFFICIAL_PAIR_COUNT:
                raise RuntimeError(
                    "official BREC v3 requires exactly "
                    f"{BREC_OFFICIAL_PAIR_COUNT} pairs, found {self.pair_count}"
                )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "BREC v3 graph6/RPC",
            "source_url": BREC_SOURCE_URL,
            "raw_artifact_url": BREC_RAW_URL,
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "records": len(self._records),
            "pair_count": self.pair_count,
            "num_relabel": self.num_relabel,
            "offline_fixture": self.fixture,
            "protocol": self.protocol,
            "rpc_threshold": 72.34 if self.num_relabel == 32 else None,
            "categories": BREC_CATEGORIES,
            "official_shape_validated": self.protocol == "official",
            "official_source_hash_pinned": False,
            "hash_note": (
                "SHA-256 is recorded for provenance; GraphPKU/BREC does not publish a "
                "canonical SHA-256 in the Release runner or README."
            ),
        }

    def load_pair(self, pair_index: int) -> BRECPair:
        if not 0 <= pair_index < self.pair_count:
            raise IndexError("BREC pair index out of range")
        category = _category(pair_index)
        span = 2 * self.num_relabel
        train_start = pair_index * span
        reliability_start = (self.pair_count + pair_index) * span

        def decode_block(start: int, phase: str) -> tuple[PaperGraph, ...]:
            return tuple(
                _brec_graph(
                    self._records[start + offset],
                    graph_id=f"brec:{pair_index:03d}:{phase}:{offset:02d}",
                    family=category,
                )
                for offset in range(span)
            )

        return BRECPair(
            pair_index=pair_index,
            category=category,
            train_test=decode_block(train_start, "train_test"),
            reliability=decode_block(reliability_start, "reliability"),
        )


def validate_brec_v3(
    path: Path,
    *,
    protocol: str = "official",
    num_relabel: int = BREC_OFFICIAL_NUM_RELABEL,
) -> dict[str, Any]:
    """Parse and structurally validate a BREC artifact, returning provenance metadata."""

    return BRECAdapter(path, num_relabel=num_relabel, protocol=protocol).metadata


def _brec_candidates(data_root: Path) -> tuple[Path, ...]:
    root = data_root.expanduser().resolve()
    return (
        root / "BREC" / "Data" / "raw" / "brec_v3.npy",
        root / "Data" / "raw" / "brec_v3.npy",
        root / "brec_v3.npy",
    )


def find_brec_v3(data_root: Path) -> Path:
    candidates = _brec_candidates(data_root)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    locations = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "BREC v3 is absent and network access is fail-closed. Extract the official "
        f"BREC_data_all.zip from {BREC_SOURCE_URL}, place brec_v3.npy at one of "
        f"the paths below, or explicitly pass --allow-download:\n  - {locations}"
    )


def _validated_brec_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    infos = archive.infolist()
    if len(infos) > _BREC_ARCHIVE_MEMBER_LIMIT:
        raise RuntimeError("BREC archive has an unsafe number of members")
    if sum(info.file_size for info in infos) > _BREC_ARCHIVE_TOTAL_LIMIT:
        raise RuntimeError("BREC archive exceeds the uncompressed-size safety limit")
    matches: list[zipfile.ZipInfo] = []
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or (path.parts and ":" in path.parts[0])
        ):
            raise RuntimeError(f"unsafe path in BREC archive: {name!r}")
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise RuntimeError(f"symbolic link rejected in BREC archive: {name!r}")
        if info.file_size > _BREC_EXTRACT_LIMIT:
            raise RuntimeError(f"oversized member rejected in BREC archive: {name!r}")
        if info.compress_size and info.file_size > 1_000 * info.compress_size:
            raise RuntimeError(f"suspicious compression ratio in BREC archive: {name!r}")
        if not info.is_dir() and path.name == "brec_v3.npy":
            matches.append(info)
    if len(matches) != 1:
        raise RuntimeError("official BREC archive must contain exactly one brec_v3.npy member")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_brec_v3(data_root: Path) -> Path:
    """Explicitly download and safely extract the official BREC v3 artifact."""

    target = _brec_candidates(data_root)[0]
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_path: Path | None = None
    extracted_path: Path | None = None
    try:
        request = urllib.request.Request(
            BREC_RAW_URL,
            headers={"User-Agent": "cycle-pe-paper/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            final_url = response.geturl()
            parsed = urllib.parse.urlparse(final_url)
            if parsed.scheme != "https" or parsed.hostname not in _BREC_DOWNLOAD_HOSTS:
                raise RuntimeError(f"unsafe redirect while downloading BREC: {final_url}")
            length_header = response.headers.get("Content-Length")
            if length_header is not None and int(length_header) > _BREC_DOWNLOAD_LIMIT:
                raise RuntimeError("BREC download exceeds the compressed-size safety limit")
            with tempfile.NamedTemporaryFile(
                prefix="brec-v3-", suffix=".zip", dir=target.parent, delete=False
            ) as archive_stream:
                archive_path = Path(archive_stream.name)
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > _BREC_DOWNLOAD_LIMIT:
                        raise RuntimeError("BREC download exceeds the compressed-size safety limit")
                    archive_stream.write(chunk)
                archive_stream.flush()
                os.fsync(archive_stream.fileno())
        if archive_path is None:
            raise RuntimeError("BREC download did not create an archive")
        archive_sha256 = _sha256(archive_path)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                member = _validated_brec_member(archive)
                with (
                    tempfile.NamedTemporaryFile(
                        prefix="brec-v3-", suffix=".npy", dir=target.parent, delete=False
                    ) as destination,
                    archive.open(member) as source,
                ):
                    extracted_path = Path(destination.name)
                    extracted = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        extracted += len(chunk)
                        if extracted > _BREC_EXTRACT_LIMIT:
                            raise RuntimeError(
                                "BREC brec_v3.npy exceeds the extraction safety limit"
                            )
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
        except zipfile.BadZipFile as exc:
            raise RuntimeError("downloaded BREC artifact is not a valid ZIP file") from exc
        if extracted_path is None:
            raise RuntimeError("BREC extraction did not produce brec_v3.npy")
        with extracted_path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                raise RuntimeError("extracted brec_v3.npy has an invalid NumPy header")
        _load_brec_records(extracted_path)
        npy_sha256 = _sha256(extracted_path)
        if target.exists():
            # A concurrent successful preparation wins; never overwrite it.
            return target
        os.replace(extracted_path, target)
        extracted_path = None
        metadata = {
            "source_url": BREC_RAW_URL,
            "archive_sha256": archive_sha256,
            "brec_v3_sha256": npy_sha256,
            "bytes": target.stat().st_size,
        }
        metadata_path = target.with_name("brec_v3.download.json")
        atomic_write_json(metadata_path, metadata)
        return target
    finally:
        for temporary in (archive_path, extracted_path):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def load_brec_v3(
    data_root: Path,
    *,
    num_relabel: int = BREC_OFFICIAL_NUM_RELABEL,
    allow_download: bool = False,
    tiny: bool = False,
    protocol: str = "custom",
) -> BRECAdapter:
    root = data_root.expanduser().resolve()
    if tiny:
        fixture_path = root / "cycle_pe_fixtures" / f"brec_v3_q{num_relabel}.npy"
        if not fixture_path.is_file():
            write_tiny_brec_fixture(fixture_path, num_relabel=num_relabel)
        return BRECAdapter(
            fixture_path,
            num_relabel=num_relabel,
            fixture=True,
            protocol=protocol,
        )
    try:
        path = find_brec_v3(root)
    except FileNotFoundError:
        if not allow_download:
            raise
        path = download_brec_v3(root)
    return BRECAdapter(path, num_relabel=num_relabel, protocol=protocol)


def write_tiny_brec_fixture(path: Path, *, num_relabel: int = 2) -> Path:
    """Write a two-pair graph6 fixture matching the official RPC block layout."""

    if num_relabel < 2:
        raise ValueError("num_relabel must be at least two")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = (
        (nx.cycle_graph(5), nx.complete_bipartite_graph(2, 3)),
        (nx.cycle_graph(6), nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))),
    )
    # The second H must be connected for the paper model; add one joining edge.
    pairs[1][1].add_edge(2, 3)
    train_records: list[bytes] = []
    reliability_records: list[bytes] = []
    for left, right in pairs:
        for relabel in range(num_relabel):
            permutation = np.random.default_rng(relabel).permutation(left.number_of_nodes())
            mapping = {node: int(permutation[node]) for node in left.nodes()}
            train_records.append(
                nx.to_graph6_bytes(nx.relabel_nodes(left, mapping), header=False).strip()
            )
            permutation = np.random.default_rng(100 + relabel).permutation(right.number_of_nodes())
            mapping = {node: int(permutation[node]) for node in right.nodes()}
            train_records.append(
                nx.to_graph6_bytes(nx.relabel_nodes(right, mapping), header=False).strip()
            )
        for relabel in range(num_relabel):
            for offset in (200, 300):
                permutation = np.random.default_rng(offset + relabel).permutation(
                    left.number_of_nodes()
                )
                mapping = {node: int(permutation[node]) for node in left.nodes()}
                reliability_records.append(
                    nx.to_graph6_bytes(nx.relabel_nodes(left, mapping), header=False).strip()
                )
    _atomic_save_npy(path, np.asarray(train_records + reliability_records, dtype=object))
    return path


__all__ = [
    "BRECAdapter",
    "BRECPair",
    "BREC_CATEGORIES",
    "BREC_OFFICIAL_NUM_RELABEL",
    "BREC_OFFICIAL_PAIR_COUNT",
    "BREC_OFFICIAL_RECORD_COUNT",
    "BREC_SOURCE_URL",
    "PYG_INSTALL_URL",
    "ZINC_SOURCE_URL",
    "download_brec_v3",
    "find_brec_v3",
    "load_brec_v3",
    "validate_brec_v3",
    "load_zinc12k",
    "write_tiny_brec_fixture",
]
````

# research/cycle_pe/paper_data.py

````python
"""Deterministic datasets and exact short-cycle labels for the paper path.

The built-in ``CycleCount-OOD`` suite is intentionally self contained: it does
not download public data and uses only NetworkX/NumPy.  Public benchmarks are
adapted lazily in :mod:`research.cycle_pe.paper_adapters`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from chartgat.cache import CacheCorruptError, CacheWrongRequestError, atomic_publish

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CYCLE_LENGTHS = (3, 4, 5, 6)
EDGE_TARGET_NAMES = tuple(f"edge_c{length}" for length in CYCLE_LENGTHS) + (
    "edge_shortest_cycle",
    "edge_short_cycle_congestion",
)
NODE_TARGET_NAMES = tuple(f"node_c{length}" for length in CYCLE_LENGTHS)
GRAPH_TARGET_NAMES = tuple(f"graph_c{length}" for length in CYCLE_LENGTHS)
CORE_SPLITS = ("train", "validation", "id_test", "size_ood", "family_ood")
GENERATOR_VERSION = "cycle-count-ood-v4"


@dataclass
class PaperGraph:
    """One connected undirected graph and optional supervision arrays."""

    graph_id: str
    split: str
    family: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    node_features: FloatArray | None = None
    edge_features: FloatArray | None = None
    edge_targets: FloatArray | None = None
    node_targets: FloatArray | None = None
    graph_targets: FloatArray | None = None

    @property
    def beta(self) -> int:
        return len(self.edges) - self.num_nodes + 1


@dataclass
class DatasetBundle:
    """Named graph splits and target metadata used by the common trainer."""

    name: str
    splits: dict[str, list[PaperGraph]]
    edge_target_names: tuple[str, ...] = ()
    node_target_names: tuple[str, ...] = ()
    graph_target_names: tuple[str, ...] = ()
    cache_path: Path | None = None
    cache_sha256: str | None = None
    metadata: dict[str, Any] | None = None


def canonical_edges(edges: Any) -> tuple[tuple[int, int], ...]:
    """Return a deterministic simple undirected edge tuple."""

    values = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if int(u) != int(v)}
    return tuple(sorted(values))


def enumerate_short_cycles(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
    *,
    max_length: int = 6,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate each undirected simple cycle of length at most ``max_length``.

    The smallest vertex is fixed as the start and the two orientations are
    broken by comparing the first and last vertices.  This avoids relying on
    the ordering conventions of a library cycle-basis implementation.
    """

    if max_length < 3:
        return ()
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    cycles: list[tuple[int, ...]] = []
    for start in range(num_nodes):
        for first in adjacency[start]:
            if first <= start:
                continue
            stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [
                (first, (start, first), frozenset((start, first)))
            ]
            while stack:
                current, path, seen = stack.pop()
                for neighbor in reversed(adjacency[current]):
                    if neighbor == start:
                        if len(path) >= 3 and path[1] < path[-1]:
                            cycles.append(path)
                        continue
                    if len(path) >= max_length or neighbor <= start or neighbor in seen:
                        continue
                    stack.append((neighbor, (*path, neighbor), seen | {neighbor}))
    return tuple(sorted(cycles, key=lambda item: (len(item), item)))


def exact_cycle_targets(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build edge/node/graph C3--C6 counts plus edge length/congestion labels."""

    edge_index = {edge: index for index, edge in enumerate(edges)}
    edge_counts = np.zeros((len(edges), len(CYCLE_LENGTHS)), dtype=np.float64)
    node_counts = np.zeros((num_nodes, len(CYCLE_LENGTHS)), dtype=np.float64)
    graph_counts = np.zeros(len(CYCLE_LENGTHS), dtype=np.float64)
    shortest = _shortest_cycle_lengths(num_nodes, edges)

    for cycle in enumerate_short_cycles(num_nodes, edges, max_length=max(CYCLE_LENGTHS)):
        length = len(cycle)
        if length not in CYCLE_LENGTHS:
            continue
        target_index = CYCLE_LENGTHS.index(length)
        graph_counts[target_index] += 1.0
        node_counts[np.asarray(cycle, dtype=np.int64), target_index] += 1.0
        cycle_edges = [
            (min(cycle[i], cycle[(i + 1) % length]), max(cycle[i], cycle[(i + 1) % length]))
            for i in range(length)
        ]
        for edge in cycle_edges:
            index = edge_index[edge]
            edge_counts[index, target_index] += 1.0

    congestion = edge_counts.sum(axis=1)
    edge_targets = np.concatenate((edge_counts, shortest[:, None], congestion[:, None]), axis=1)
    return edge_targets, node_counts, graph_counts


def _shortest_cycle_lengths(num_nodes: int, edges: tuple[tuple[int, int], ...]) -> FloatArray:
    """Return the exact girth-through-edge, with zero reserved for bridges.

    Removing edge ``(u, v)`` turns its shortest containing cycle into the
    shortest remaining ``u``--``v`` path plus that removed edge.  The sparse
    BFS implementation avoids enumerating long cycles.
    """

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    result = np.zeros(len(edges), dtype=np.float64)
    for edge_index, (source, target) in enumerate(edges):
        distance = [-1] * num_nodes
        distance[source] = 0
        frontier: deque[int] = deque((source,))
        while frontier and distance[target] < 0:
            node = frontier.popleft()
            for neighbor in adjacency[node]:
                if (node == source and neighbor == target) or (
                    node == target and neighbor == source
                ):
                    continue
                if distance[neighbor] >= 0:
                    continue
                distance[neighbor] = distance[node] + 1
                frontier.append(neighbor)
        if distance[target] >= 0:
            result[edge_index] = float(distance[target] + 1)
    return result


def _tree_plus_chords(num_nodes: int, beta: int, seed: int) -> tuple[tuple[int, int], ...]:
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    for node in range(1, num_nodes):
        parent = int(rng.integers(0, node))
        edges.add((parent, node))
    candidates = [
        (u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes) if (u, v) not in edges
    ]
    chosen = rng.choice(len(candidates), size=beta, replace=False)
    edges.update(candidates[int(index)] for index in np.atleast_1d(chosen))
    return tuple(sorted(edges))


def _random_regular(num_nodes: int, seed: int) -> tuple[tuple[int, int], ...]:
    # Cubic graphs give useful short-cycle variation while remaining sparse.
    if num_nodes % 2:
        num_nodes += 1
    for attempt in range(128):
        graph = nx.random_regular_graph(3, num_nodes, seed=seed + attempt)
        if nx.is_connected(graph):
            return canonical_edges(graph.edges())
    raise RuntimeError("failed to generate a connected random-regular graph")


def _small_world(num_nodes: int, seed: int) -> tuple[tuple[int, int], ...]:
    graph = nx.connected_watts_strogatz_graph(
        num_nodes,
        k=4,
        p=0.35,
        tries=256,
        seed=seed,
    )
    return canonical_edges(graph.edges())


def _local_chords(num_nodes: int, beta: int, seed: int) -> tuple[tuple[int, int], ...]:
    """Generate a path with local chords, producing overlapping cycle blocks."""

    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = {(node, node + 1) for node in range(num_nodes - 1)}
    candidates: list[tuple[int, int]] = []
    for span in (2, 3, 4, 5):
        candidates.extend((start, start + span) for start in range(num_nodes - span))
    rng.shuffle(candidates)
    for edge in candidates:
        if edge not in edges:
            edges.add(edge)
        if len(edges) == num_nodes - 1 + beta:
            break
    if len(edges) != num_nodes - 1 + beta:
        raise RuntimeError("not enough distinct local chord candidates")
    return tuple(sorted(edges))


def _graph_seed(seed: int, split_index: int, sample_index: int) -> int:
    sequence = np.random.SeedSequence([int(seed), int(split_index), int(sample_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _generate_graph(split: str, index: int, seed: int) -> PaperGraph:
    split_index = CORE_SPLITS.index(split)
    graph_seed = _graph_seed(seed, split_index, index)
    rng = np.random.default_rng(graph_seed)

    if split == "size_ood":
        num_nodes = int(rng.integers(28, 39))
        if index % 2:
            num_nodes += num_nodes % 2
            family = "random_regular"
            edges = _random_regular(num_nodes, graph_seed)
        else:
            family = "tree_plus_chords"
            beta = int(rng.integers(8, 15))
            edges = _tree_plus_chords(num_nodes, beta, graph_seed)
    elif split == "family_ood":
        num_nodes = int(rng.integers(14, 23))
        if index % 2:
            family = "small_world"
            edges = _small_world(num_nodes, graph_seed)
        else:
            family = "local_chords"
            beta = int(rng.integers(4, min(9, num_nodes - 2)))
            edges = _local_chords(num_nodes, beta, graph_seed)
    else:
        num_nodes = int(rng.integers(14, 23))
        if index % 2:
            num_nodes += num_nodes % 2
            family = "random_regular"
            edges = _random_regular(num_nodes, graph_seed)
        else:
            family = "tree_plus_chords"
            beta = int(rng.integers(3, min(9, num_nodes - 2)))
            edges = _tree_plus_chords(num_nodes, beta, graph_seed)

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edges)
    if not nx.is_connected(nx_graph):
        raise RuntimeError("CycleCount-OOD generator produced a disconnected graph")
    edge_targets, node_targets, graph_targets = exact_cycle_targets(num_nodes, edges)
    return PaperGraph(
        graph_id=f"{split}:{family}:{index:06d}:{graph_seed}",
        split=split,
        family=family,
        num_nodes=num_nodes,
        edges=edges,
        edge_targets=edge_targets,
        node_targets=node_targets,
        graph_targets=graph_targets,
    )


def cycle_count_split_sizes(tiny: bool) -> dict[str, int]:
    if tiny:
        return {
            "train": 10,
            "validation": 4,
            "id_test": 4,
            "size_ood": 4,
            "family_ood": 4,
        }
    return {
        "train": 10_000,
        "validation": 2_000,
        "id_test": 2_000,
        "size_ood": 3_000,
        "family_ood": 3_000,
    }


def _graph_to_json(graph: PaperGraph) -> dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "split": graph.split,
        "family": graph.family,
        "num_nodes": graph.num_nodes,
        "edges": [list(edge) for edge in graph.edges],
        "edge_targets": graph.edge_targets.tolist() if graph.edge_targets is not None else None,
        "node_targets": graph.node_targets.tolist() if graph.node_targets is not None else None,
        "graph_targets": (
            graph.graph_targets.tolist() if graph.graph_targets is not None else None
        ),
    }


def _graph_from_json(record: dict[str, Any]) -> PaperGraph:
    def array_or_none(name: str) -> FloatArray | None:
        value = record.get(name)
        return None if value is None else np.asarray(value, dtype=np.float64)

    return PaperGraph(
        graph_id=str(record["graph_id"]),
        split=str(record["split"]),
        family=str(record["family"]),
        num_nodes=int(record["num_nodes"]),
        edges=canonical_edges(record["edges"]),
        edge_targets=array_or_none("edge_targets"),
        node_targets=array_or_none("node_targets"),
        graph_targets=array_or_none("graph_targets"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cycle_count_specification(
    *, seed: int, tiny: bool, split_sizes: dict[str, int] | None
) -> tuple[dict[str, int], dict[str, Any], Path]:
    sizes = dict(cycle_count_split_sizes(tiny) if split_sizes is None else split_sizes)
    if set(sizes) != set(CORE_SPLITS) or any(int(value) < 1 for value in sizes.values()):
        raise ValueError(f"split_sizes must provide positive counts for {CORE_SPLITS}")
    specification = {
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "split_sizes": {name: int(sizes[name]) for name in CORE_SPLITS},
        "cycle_lengths": list(CYCLE_LENGTHS),
    }
    key = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return sizes, specification, Path(f"{GENERATOR_VERSION}-{key}.json.gz")


def _validate_cycle_count_payload(
    payload: Any, specification: dict[str, Any]
) -> dict[str, list[PaperGraph]]:
    if not isinstance(payload, dict):
        raise CacheCorruptError("CycleCount cache root must be a mapping")
    if payload.get("schema_version") != 1:
        raise CacheWrongRequestError("unsupported CycleCount cache schema")
    if payload.get("specification") != specification:
        raise CacheWrongRequestError("CycleCount cache specification mismatch")
    expected_targets = {
        "edge": list(EDGE_TARGET_NAMES),
        "node": list(NODE_TARGET_NAMES),
        "graph": list(GRAPH_TARGET_NAMES),
    }
    if payload.get("target_names") != expected_targets:
        raise CacheCorruptError("CycleCount target schema mismatch")
    records = payload.get("graphs")
    if not isinstance(records, list):
        raise CacheCorruptError("CycleCount graphs must be a list")
    expected_total = sum(int(value) for value in specification["split_sizes"].values())
    if len(records) != expected_total:
        raise CacheCorruptError("CycleCount graph count does not match the requested splits")
    splits = {name: [] for name in CORE_SPLITS}
    graph_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise CacheCorruptError("CycleCount graph record must be a mapping")
        try:
            graph = _graph_from_json(record)
        except (KeyError, TypeError, ValueError) as error:
            raise CacheCorruptError("invalid CycleCount graph record") from error
        if graph.split not in splits:
            raise CacheCorruptError(f"unknown CycleCount split {graph.split!r}")
        if not graph.graph_id or graph.graph_id in graph_ids:
            raise CacheCorruptError("CycleCount graph IDs must be non-empty and unique")
        graph_ids.add(graph.graph_id)
        edge_count = len(graph.edges)
        raw_edges = record.get("edges")
        if not isinstance(raw_edges, list) or len(raw_edges) != edge_count:
            raise CacheCorruptError("CycleCount edges are duplicated, malformed, or missing")
        if graph.num_nodes < 2 or edge_count < 1:
            raise CacheCorruptError("CycleCount graph dimensions are invalid")
        if any(u < 0 or v >= graph.num_nodes for u, v in graph.edges):
            raise CacheCorruptError("CycleCount edge endpoint lies outside the graph")
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(graph.num_nodes))
        nx_graph.add_edges_from(graph.edges)
        if not nx.is_connected(nx_graph):
            raise CacheCorruptError("CycleCount graph must be connected")
        arrays = (
            (graph.edge_targets, (edge_count, len(EDGE_TARGET_NAMES))),
            (graph.node_targets, (graph.num_nodes, len(NODE_TARGET_NAMES))),
            (graph.graph_targets, (len(GRAPH_TARGET_NAMES),)),
        )
        for array, shape in arrays:
            if array is None or array.shape != shape or not np.all(np.isfinite(array)):
                raise CacheCorruptError("CycleCount target tensor has invalid shape or values")
        splits[graph.split].append(graph)
    actual_sizes = {name: len(graphs) for name, graphs in splits.items()}
    if actual_sizes != specification["split_sizes"]:
        raise CacheCorruptError("CycleCount split cardinalities do not match the specification")
    for graphs in splits.values():
        graphs.sort(key=lambda graph: graph.graph_id)
    return splits


def validate_cycle_count_ood_cache(
    data_root: Path,
    *,
    seed: int,
    tiny: bool = False,
    split_sizes: dict[str, int] | None = None,
) -> DatasetBundle:
    """Read and fully validate a requested CycleCount-OOD cache without writing."""

    _, specification, filename = _cycle_count_specification(
        seed=seed, tiny=tiny, split_sizes=split_sizes
    )
    cache_path = data_root.expanduser().resolve() / "cycle_count_ood" / filename
    if not cache_path.is_file():
        raise FileNotFoundError(f"CycleCount cache is missing for seed={seed}: {cache_path}")
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"failed to parse CycleCount cache: {cache_path}") from error
    splits = _validate_cycle_count_payload(payload, specification)
    return DatasetBundle(
        name="CycleCount-OOD",
        splits=splits,
        edge_target_names=EDGE_TARGET_NAMES,
        node_target_names=NODE_TARGET_NAMES,
        graph_target_names=GRAPH_TARGET_NAMES,
        cache_path=cache_path,
        cache_sha256=sha256_file(cache_path),
        metadata=specification,
    )


def load_or_generate_cycle_count_ood(
    data_root: Path,
    *,
    seed: int,
    tiny: bool = False,
    split_sizes: dict[str, int] | None = None,
) -> DatasetBundle:
    """Load a content-addressed cache or deterministically build CycleCount-OOD."""

    sizes, specification, filename = _cycle_count_specification(
        seed=seed, tiny=tiny, split_sizes=split_sizes
    )
    cache_dir = data_root.expanduser().resolve() / "cycle_count_ood"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / filename

    if cache_path.exists():
        return validate_cycle_count_ood_cache(
            data_root, seed=seed, tiny=tiny, split_sizes=split_sizes
        )
    records = []
    for split in CORE_SPLITS:
        for index in range(int(sizes[split])):
            records.append(_graph_to_json(_generate_graph(split, index, seed)))
    payload = {
        "schema_version": 1,
        "specification": specification,
        "target_names": {
            "edge": list(EDGE_TARGET_NAMES),
            "node": list(NODE_TARGET_NAMES),
            "graph": list(GRAPH_TARGET_NAMES),
        },
        "graphs": records,
    }

    def write(temporary: Path) -> None:
        # mtime=0 makes the compressed cache byte-deterministic as well.
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(filename="", fileobj=raw_stream, mode="wb", mtime=0) as zipped:
                zipped.write(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )

    def validate_temporary(temporary: Path) -> None:
        with gzip.open(temporary, "rt", encoding="utf-8") as stream:
            _validate_cycle_count_payload(json.load(stream), specification)

    atomic_publish(cache_path, write, validator=validate_temporary)
    return validate_cycle_count_ood_cache(data_root, seed=seed, tiny=tiny, split_sizes=split_sizes)


def structural_input_features(graph: PaperGraph) -> tuple[FloatArray, FloatArray]:
    """Return node/edge attributes without leaking any exact cycle label."""

    if graph.node_features is not None:
        node_features = np.asarray(graph.node_features, dtype=np.float64)
    else:
        degrees = np.zeros(graph.num_nodes, dtype=np.float64)
        for u, v in graph.edges:
            degrees[u] += 1.0
            degrees[v] += 1.0
        scale = max(1.0, float(graph.num_nodes - 1))
        node_features = np.column_stack((np.ones(graph.num_nodes), degrees / scale))

    if graph.edge_features is not None:
        edge_features = np.asarray(graph.edge_features, dtype=np.float64)
    else:
        degrees = np.zeros(graph.num_nodes, dtype=np.float64)
        for u, v in graph.edges:
            degrees[u] += 1.0
            degrees[v] += 1.0
        scale = max(1.0, float(graph.num_nodes - 1))
        rows = []
        for u, v in graph.edges:
            low, high = sorted((degrees[u], degrees[v]))
            rows.append((1.0, low / scale, high / scale, abs(high - low) / scale))
        edge_features = np.asarray(rows, dtype=np.float64).reshape(len(graph.edges), 4)
    return node_features, edge_features


__all__ = [
    "CORE_SPLITS",
    "CYCLE_LENGTHS",
    "DatasetBundle",
    "EDGE_TARGET_NAMES",
    "GENERATOR_VERSION",
    "GRAPH_TARGET_NAMES",
    "NODE_TARGET_NAMES",
    "PaperGraph",
    "canonical_edges",
    "cycle_count_split_sizes",
    "enumerate_short_cycles",
    "exact_cycle_targets",
    "load_or_generate_cycle_count_ood",
    "sha256_file",
    "structural_input_features",
    "validate_cycle_count_ood_cache",
]
````

# research/cycle_pe/paper_model.py

````python
"""Batch-safe neural models for the static cycle-PE paper experiments.

The paper path deliberately keeps graph batches ragged.  Every graph may have
a different edge count and cycle rank; raw bases are padded to the maximum
rank fitted on the training split only, never to a fixed constant or a width
selected from validation/test graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chartgat.algebra import incidence_matrix
from chartgat.graphs import spanning_tree_indices
from research.cycle_pe.features import (
    SET_STAT_NAMES,
    cycle_projector,
    cycle_set_statistics,
    static_fundamental_basis,
)
from research.cycle_pe.paper_data import PaperGraph, structural_input_features

PE_VARIANTS = ("no_pe", "raw", "set", "projector")


class RawCycleRankOverflow(RuntimeError):
    """Raised rather than silently truncating an OOD raw cycle basis."""

    def __init__(self, actual_rank: int, fitted_width: int) -> None:
        self.actual_rank = int(actual_rank)
        self.fitted_width = int(fitted_width)
        super().__init__(f"cycle rank {actual_rank} exceeds train-fitted raw width {fitted_width}")


@dataclass
class PreparedGraph:
    """One tensorized graph with only the requested expensive PE representations."""

    graph_id: str
    split: str
    family: str
    num_nodes: int
    cycle_rank: int
    edges: Tensor
    node_features: Tensor
    edge_features: Tensor
    raw_basis: Tensor
    cycle_set: Tensor | None
    projector: Tensor | None
    edge_targets: Tensor | None
    node_targets: Tensor | None
    graph_targets: Tensor | None

    def pin_memory(self) -> PreparedGraph:
        """Support recursive DataLoader pinning for ragged graph objects."""

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                setattr(self, field.name, value.pin_memory())
        return self

    def to(self, device: torch.device, *, non_blocking: bool = False) -> PreparedGraph:
        values: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device=device, non_blocking=non_blocking)
                if isinstance(value, Tensor)
                else value
            )
        return PreparedGraph(**values)  # type: ignore[arg-type]


def infer_raw_width(graphs: list[PaperGraph]) -> int:
    """Infer a lossless raw-basis width from arbitrary cycle ranks."""

    if not graphs:
        raise ValueError("cannot infer a raw width from an empty graph list")
    ranks = [graph.beta for graph in graphs]
    if any(rank < 0 for rank in ranks):
        raise ValueError("all paper graphs must be connected")
    return max(ranks)


def prepare_graph(
    graph: PaperGraph,
    *,
    required_variants: tuple[str, ...] = PE_VARIANTS,
) -> PreparedGraph:
    """Extract only requested topology PEs and convert one graph to CPU tensors."""

    unknown = set(required_variants) - set(PE_VARIANTS)
    if unknown:
        raise ValueError(f"unknown PE variants: {sorted(unknown)}")

    edge_list = list(graph.edges)
    incidence = incidence_matrix(graph.num_nodes, edge_list)
    tree = spanning_tree_indices(graph.num_nodes, edge_list, mode="bfs")
    basis = static_fundamental_basis(incidence, tree)
    if basis.shape[1] != graph.beta:
        raise RuntimeError("cycle-rank mismatch while preparing graph")
    node_features, edge_features = structural_input_features(graph)

    def float_tensor(value: np.ndarray | None) -> Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(np.asarray(value), dtype=torch.float32).contiguous()

    return PreparedGraph(
        graph_id=graph.graph_id,
        split=graph.split,
        family=graph.family,
        num_nodes=graph.num_nodes,
        cycle_rank=graph.beta,
        edges=torch.as_tensor(edge_list, dtype=torch.long).reshape(-1, 2).contiguous(),
        node_features=torch.as_tensor(node_features, dtype=torch.float32).contiguous(),
        edge_features=torch.as_tensor(edge_features, dtype=torch.float32).contiguous(),
        # Keep every coordinate. The raw encoder pads only ranks that fit the
        # train-derived width and raises explicitly on OOD overflow.
        raw_basis=torch.as_tensor(basis, dtype=torch.float32).contiguous(),
        cycle_set=(
            torch.as_tensor(cycle_set_statistics(basis), dtype=torch.float32).contiguous()
            if "set" in required_variants
            else None
        ),
        projector=(
            torch.as_tensor(cycle_projector(basis), dtype=torch.float32).contiguous()
            if "projector" in required_variants
            else None
        ),
        edge_targets=float_tensor(graph.edge_targets),
        node_targets=float_tensor(graph.node_targets),
        graph_targets=float_tensor(graph.graph_targets),
    )


def prepare_splits(
    splits: dict[str, list[PaperGraph]],
    *,
    fit_split: str | None = None,
    required_variants: tuple[str, ...] = PE_VARIANTS,
) -> tuple[dict[str, list[PreparedGraph]], int]:
    """Prepare all splits with a raw width fitted on training data only."""

    if not splits:
        raise ValueError("splits cannot be empty")
    selected = fit_split or ("train" if "train" in splits else next(iter(splits)))
    if selected not in splits:
        raise ValueError(f"fit_split {selected!r} is not present")
    raw_width = infer_raw_width(splits[selected])
    prepared = {
        split: [prepare_graph(graph, required_variants=required_variants) for graph in graphs]
        for split, graphs in splits.items()
    }
    return prepared, raw_width


class StaticPEEncoder(nn.Module):
    """Map any supported static PE to a common per-edge representation.

    The projector path is a row-wise DeepSets encoder over the full intrinsic
    cycle-space projector.  It is independent of cycle rank and invariant to
    every invertible change of cycle basis.  Absolute pair entries additionally
    remove arbitrary incidence-orientation signs.
    """

    def __init__(
        self,
        variant: str,
        *,
        raw_width: int,
        pe_dim: int,
    ) -> None:
        super().__init__()
        if variant not in PE_VARIANTS:
            raise ValueError(f"variant must be one of {PE_VARIANTS}")
        if raw_width < 0 or pe_dim < 1:
            raise ValueError("raw_width must be non-negative and pe_dim positive")
        self.variant = variant
        self.raw_width = raw_width
        self.pe_dim = pe_dim
        self.raw_encoder = (
            nn.Sequential(nn.Linear(raw_width, pe_dim), nn.GELU()) if raw_width else None
        )
        self.empty_raw = nn.Parameter(torch.zeros(pe_dim))
        self.set_encoder = nn.Sequential(
            nn.Linear(len(SET_STAT_NAMES), pe_dim),
            nn.GELU(),
            nn.Linear(pe_dim, pe_dim),
        )
        projector_hidden = max(8, pe_dim)
        self.projector_pair = nn.Sequential(
            nn.Linear(3, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, projector_hidden),
            nn.GELU(),
        )
        self.projector_row = nn.Sequential(
            nn.Linear(2 * projector_hidden + 2, pe_dim),
            nn.GELU(),
            nn.Linear(pe_dim, pe_dim),
        )

    def forward(
        self,
        raw_basis: Tensor,
        cycle_set: Tensor | None,
        projector: Tensor | None,
    ) -> Tensor:
        edge_count = raw_basis.shape[0]
        if self.variant == "no_pe":
            return raw_basis.new_zeros((edge_count, self.pe_dim))
        if self.variant == "raw":
            actual_rank = raw_basis.shape[1]
            if actual_rank > self.raw_width:
                raise RawCycleRankOverflow(actual_rank, self.raw_width)
            if actual_rank < self.raw_width:
                raw_basis = F.pad(raw_basis, (0, self.raw_width - actual_rank))
            if self.raw_encoder is None:
                return self.empty_raw.unsqueeze(0).expand(edge_count, -1)
            return self.raw_encoder(raw_basis)
        if self.variant == "set":
            if cycle_set is None:
                raise ValueError("set PE was not prepared for this graph")
            return self.set_encoder(cycle_set)

        if projector is None:
            raise ValueError("projector PE was not prepared for this graph")
        if projector.shape != (edge_count, edge_count):
            raise ValueError("projector must have shape (num_edges, num_edges)")
        if edge_count == 0:
            return raw_basis.new_zeros((0, self.pe_dim))
        absolute = projector.abs()
        diagonal = projector.diagonal().abs()
        pair_features = torch.stack(
            (
                absolute,
                absolute.square(),
                diagonal.unsqueeze(0).expand(edge_count, -1),
            ),
            dim=-1,
        )
        encoded = self.projector_pair(pair_features)
        mean = encoded.mean(dim=1)
        maximum = encoded.amax(dim=1)
        row_features = torch.cat(
            (mean, maximum, diagonal[:, None], absolute.mean(dim=1, keepdim=True)),
            dim=1,
        )
        return self.projector_row(row_features)


class GraphOutput(NamedTuple):
    edge: Tensor | None
    node: Tensor | None
    graph: Tensor | None
    embedding: Tensor


class _MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        u, v = edge_index[:, 0], edge_index[:, 1]
        symmetric = torch.cat((node[u] + node[v], (node[u] - node[v]).abs(), edge), dim=1)
        updated_edge = self.edge_norm(edge + self.edge_update(symmetric))

        source = torch.cat((u, v), dim=0)
        target = torch.cat((v, u), dim=0)
        directed_edge = torch.cat((updated_edge, updated_edge), dim=0)
        messages = self.message(torch.cat((node[source], node[target], directed_edge), dim=1))
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(node.shape[0], device=node.device, dtype=node.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=node.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)[:, None]
        updated_node = self.node_norm(node + self.node_update(torch.cat((node, aggregate), dim=1)))
        return updated_node, updated_edge


class PaperCycleModel(nn.Module):
    """Small edge-aware GNN shared by CycleCount-OOD, ZINC, and BREC."""

    def __init__(
        self,
        *,
        variant: str,
        raw_width: int,
        node_input_dim: int,
        edge_input_dim: int,
        edge_output_dim: int,
        node_output_dim: int,
        graph_output_dim: int,
        hidden_dim: int = 64,
        pe_dim: int = 32,
        layers: int = 3,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim < 4 or layers < 1:
            raise ValueError("hidden_dim must be >=4 and layers positive")
        self.pe_encoder = StaticPEEncoder(variant, raw_width=raw_width, pe_dim=pe_dim)
        self.node_encoder = nn.Sequential(nn.Linear(node_input_dim, hidden_dim), nn.GELU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_input_dim + pe_dim, hidden_dim), nn.GELU())
        self.layers = nn.ModuleList(_MessageLayer(hidden_dim) for _ in range(layers))
        self.edge_head = nn.Linear(hidden_dim, edge_output_dim) if edge_output_dim else None
        self.node_head = nn.Linear(hidden_dim, node_output_dim) if node_output_dim else None
        pooled_dim = 4 * hidden_dim
        self.graph_trunk = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.graph_head = nn.Linear(hidden_dim, graph_output_dim) if graph_output_dim else None
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)

    def forward_graph(self, graph: PreparedGraph) -> GraphOutput:
        return self.forward([graph])[0]

    def forward(self, graphs: list[PreparedGraph]) -> list[GraphOutput]:
        if not graphs:
            return []
        node_counts = [graph.node_features.shape[0] for graph in graphs]
        edge_counts = [graph.edge_features.shape[0] for graph in graphs]
        positional = [
            self.pe_encoder(graph.raw_basis, graph.cycle_set, graph.projector) for graph in graphs
        ]
        node = self.node_encoder(torch.cat([graph.node_features for graph in graphs], dim=0))
        edge = self.edge_encoder(
            torch.cat(
                [
                    torch.cat((graph.edge_features, pe), dim=1)
                    for graph, pe in zip(graphs, positional, strict=True)
                ],
                dim=0,
            )
        )
        offsets: list[int] = []
        running = 0
        for count in node_counts:
            offsets.append(running)
            running += count
        edge_index = torch.cat(
            [graph.edges + offset for graph, offset in zip(graphs, offsets, strict=True)],
            dim=0,
        )
        for layer in self.layers:
            node, edge = layer(node, edge, edge_index)
        node_parts = list(torch.split(node, node_counts, dim=0))
        edge_parts = list(torch.split(edge, edge_counts, dim=0))
        pooled_rows: list[Tensor] = []
        for node_part, edge_part in zip(node_parts, edge_parts, strict=True):
            if edge_part.shape[0]:
                edge_mean = edge_part.mean(dim=0)
                edge_maximum = edge_part.amax(dim=0)
            else:
                edge_mean = node_part.new_zeros(node_part.shape[1])
                edge_maximum = node_part.new_zeros(node_part.shape[1])
            pooled_rows.append(
                torch.cat(
                    (
                        node_part.mean(dim=0),
                        node_part.amax(dim=0),
                        edge_mean,
                        edge_maximum,
                    ),
                    dim=0,
                )
            )
        graph_state = self.graph_trunk(torch.stack(pooled_rows))
        edge_prediction = None if self.edge_head is None else self.edge_head(edge)
        node_prediction = None if self.node_head is None else self.node_head(node)
        graph_prediction = None if self.graph_head is None else self.graph_head(graph_state)
        embedding = self.embedding_head(graph_state)
        edge_outputs = (
            [None] * len(graphs)
            if edge_prediction is None
            else list(torch.split(edge_prediction, edge_counts, dim=0))
        )
        node_outputs = (
            [None] * len(graphs)
            if node_prediction is None
            else list(torch.split(node_prediction, node_counts, dim=0))
        )
        return [
            GraphOutput(
                edge=edge_outputs[index],
                node=node_outputs[index],
                graph=None if graph_prediction is None else graph_prediction[index],
                embedding=embedding[index],
            )
            for index in range(len(graphs))
        ]


__all__ = [
    "GraphOutput",
    "PE_VARIANTS",
    "PaperCycleModel",
    "PreparedGraph",
    "RawCycleRankOverflow",
    "StaticPEEncoder",
    "infer_raw_width",
    "prepare_graph",
    "prepare_splits",
]
````

# research/cycle_pe/paper_train.py

````python
"""Training, evaluation, and runtime accounting for the paper CLI."""

from __future__ import annotations

import copy
import math
import platform
import random
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from research.cycle_pe.paper_model import GraphOutput, PaperCycleModel, PreparedGraph


@dataclass(frozen=True)
class TrainSettings:
    device: torch.device
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    workers: int
    amp_requested: bool
    pin_memory_requested: bool
    non_blocking_requested: bool

    @property
    def amp(self) -> bool:
        return self.amp_requested and self.device.type == "cuda"

    @property
    def pin_memory(self) -> bool:
        return self.pin_memory_requested and self.device.type == "cuda"

    @property
    def non_blocking(self) -> bool:
        return self.non_blocking_requested and self.pin_memory


@dataclass(frozen=True)
class TargetStats:
    mean: Tensor
    std: Tensor


def cuda_autocast(enabled: bool):
    """Use the public autocast API available across supported PyTorch releases."""

    return torch.autocast(device_type="cuda", enabled=enabled)


def make_grad_scaler(enabled: bool):
    """Construct a CUDA GradScaler on both PyTorch 2.2 and newer releases."""

    unified_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if unified_scaler is not None:
        try:
            return unified_scaler("cuda", enabled=enabled)
        except TypeError:
            return unified_scaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device specification: {requested!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this PyTorch build cannot access CUDA. On the "
                "Linux/MobaXterm host, verify `nvidia-smi`, then install the matching "
                "CUDA-enabled PyTorch wheel. Use `--device cpu --tiny` only for smoke tests."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; detected "
                f"{torch.cuda.device_count()} device(s)"
            )
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise ValueError("paper CLI supports only cpu, cuda, cuda:N, or auto")
    return device


def runtime_environment(settings: TrainSettings) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(settings.device),
        "amp_requested": settings.amp_requested,
        "amp_effective": settings.amp,
        "pin_memory_requested": settings.pin_memory_requested,
        "pin_memory_effective": settings.pin_memory,
        "non_blocking_requested": settings.non_blocking_requested,
        "non_blocking_effective": settings.non_blocking,
        "batch_size": settings.batch_size,
        "workers": settings.workers,
    }
    if settings.device.type == "cuda":
        index = settings.device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_capability": [properties.major, properties.minor],
                "cuda_total_memory_bytes": properties.total_memory,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return result


def _target(graph: PreparedGraph, level: str) -> Tensor | None:
    return getattr(graph, f"{level}_targets")


def fit_target_stats(
    graphs: list[PreparedGraph],
    *,
    levels: tuple[str, ...] = ("edge", "node", "graph"),
) -> dict[str, TargetStats]:
    if not graphs:
        raise ValueError("training split cannot be empty")
    result: dict[str, TargetStats] = {}
    unknown = set(levels) - {"edge", "node", "graph"}
    if unknown:
        raise ValueError(f"unknown target levels: {sorted(unknown)}")
    for level in levels:
        values = [_target(graph, level) for graph in graphs]
        present = [value for value in values if value is not None]
        if not present:
            continue
        if len(present) != len(graphs):
            raise ValueError(f"{level} targets are missing on part of the training split")
        matrix = torch.cat([value.reshape(-1, value.shape[-1]) for value in present], dim=0)
        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0, unbiased=False).clamp_min(1e-6)
        result[level] = TargetStats(mean=mean, std=std)
    if not result:
        raise ValueError("at least one supervised target level is required")
    return result


def _collate(graphs: list[PreparedGraph]) -> list[PreparedGraph]:
    return graphs


def _loader(
    graphs: list[PreparedGraph],
    settings: TrainSettings,
    *,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(settings.seed + seed_offset)
    return DataLoader(
        graphs,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.workers,
        pin_memory=settings.pin_memory,
        collate_fn=_collate,
        generator=generator,
        drop_last=False,
        persistent_workers=settings.workers > 0,
    )


def _move_batch(graphs: list[PreparedGraph], settings: TrainSettings) -> list[PreparedGraph]:
    return [graph.to(settings.device, non_blocking=settings.non_blocking) for graph in graphs]


def _output(outputs: list[GraphOutput], level: str) -> list[Tensor]:
    values = [getattr(output, level) for output in outputs]
    if any(value is None for value in values):
        raise RuntimeError(f"model did not produce the configured {level} head")
    return [value for value in values if value is not None]


def normalized_multitask_loss(
    outputs: list[GraphOutput],
    graphs: list[PreparedGraph],
    stats: dict[str, TargetStats],
) -> Tensor:
    losses: list[Tensor] = []
    for level, level_stats in stats.items():
        predictions = _output(outputs, level)
        targets = [_target(graph, level) for graph in graphs]
        if any(target is None for target in targets):
            raise RuntimeError(f"batch has missing {level} targets")
        prediction = torch.cat([value.reshape(-1, value.shape[-1]) for value in predictions], dim=0)
        target = torch.cat(
            [value.reshape(-1, value.shape[-1]) for value in targets if value is not None],
            dim=0,
        )
        mean = level_stats.mean.to(target.device)
        std = level_stats.std.to(target.device)
        losses.append(torch.mean((prediction - (target - mean) / std) ** 2))
    return torch.stack(losses).mean()


@torch.no_grad()
def validation_loss(
    model: PaperCycleModel,
    graphs: list[PreparedGraph],
    stats: dict[str, TargetStats],
    settings: TrainSettings,
) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    for cpu_graphs in _loader(graphs, settings, shuffle=False):
        batch = _move_batch(cpu_graphs, settings)
        with cuda_autocast(settings.amp):
            loss = normalized_multitask_loss(model(batch), batch, stats)
        weighted += float(loss.detach().cpu()) * len(batch)
        count += len(batch)
    return weighted / max(1, count)


def _peak_memory(settings: TrainSettings) -> int:
    if settings.device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(settings.device))


def train_supervised(
    model: PaperCycleModel,
    train_graphs: list[PreparedGraph],
    validation_graphs: list[PreparedGraph],
    settings: TrainSettings,
    *,
    target_levels: tuple[str, ...] = ("edge", "node", "graph"),
) -> tuple[PaperCycleModel, dict[str, TargetStats], list[dict[str, float]], dict[str, Any]]:
    """Train normalized edge/node/graph heads and restore best validation state."""

    if settings.epochs < 1 or settings.batch_size < 1 or settings.workers < 0:
        raise ValueError("epochs/batch_size must be positive and workers non-negative")
    seed_everything(settings.seed)
    model = model.to(settings.device)
    stats = fit_target_stats(train_graphs, levels=target_levels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scaler = make_grad_scaler(settings.amp)
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    if settings.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(settings.device)
        torch.cuda.synchronize(settings.device)
    started = time.perf_counter()
    for epoch in range(settings.epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for cpu_graphs in _loader(train_graphs, settings, shuffle=True, seed_offset=epoch):
            graphs = _move_batch(cpu_graphs, settings)
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(settings.amp):
                loss = normalized_multitask_loss(model(graphs), graphs, stats)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(graphs)
            seen += len(graphs)
        current_validation = validation_loss(model, validation_graphs, stats, settings)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": loss_sum / max(1, seen),
                "validation_loss": current_validation,
            }
        )
        if current_validation < best_loss:
            best_loss = current_validation
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    if settings.device.type == "cuda":
        torch.cuda.synchronize(settings.device)
    wall_seconds = time.perf_counter() - started
    runtime = runtime_environment(settings)
    runtime.update(
        {
            "wall_seconds": wall_seconds,
            "peak_gpu_memory_bytes": _peak_memory(settings),
            "best_validation_loss": best_loss,
            "epochs_completed": settings.epochs,
        }
    )
    return model, stats, history, runtime


@torch.no_grad()
def evaluate_supervised(
    model: PaperCycleModel,
    graphs: list[PreparedGraph],
    stats: dict[str, TargetStats],
    settings: TrainSettings,
    target_names: dict[str, tuple[str, ...]],
    *,
    integer_targets: bool,
) -> dict[str, Any]:
    """Return per-target MAE/RMSE, normalized MAE, and graph-macro MAE."""

    model.eval()
    predictions: dict[str, list[np.ndarray]] = {level: [] for level in stats}
    targets: dict[str, list[np.ndarray]] = {level: [] for level in stats}
    for cpu_graphs in _loader(graphs, settings, shuffle=False):
        batch = _move_batch(cpu_graphs, settings)
        with cuda_autocast(settings.amp):
            outputs = model(batch)
        for graph, output in zip(batch, outputs, strict=True):
            for level, level_stats in stats.items():
                raw_prediction = getattr(output, level)
                raw_target = _target(graph, level)
                if raw_prediction is None or raw_target is None:
                    raise RuntimeError(f"missing {level} output during evaluation")
                mean = level_stats.mean.to(raw_prediction.device)
                std = level_stats.std.to(raw_prediction.device)
                prediction = raw_prediction * std + mean
                predictions[level].append(prediction.detach().float().cpu().numpy())
                targets[level].append(raw_target.detach().float().cpu().numpy())

    result: dict[str, Any] = {
        "graphs": len(graphs),
        "nodes": sum(graph.num_nodes for graph in graphs),
        "edges": sum(graph.edges.shape[0] for graph in graphs),
        "levels": {},
    }
    macro_normalized: list[float] = []
    for level, level_stats in stats.items():
        names = target_names[level]
        if len(names) != level_stats.mean.numel():
            raise ValueError(f"target-name count mismatch for {level}")
        flat_prediction = np.concatenate(
            [value.reshape(-1, value.shape[-1]) for value in predictions[level]], axis=0
        )
        flat_target = np.concatenate(
            [value.reshape(-1, value.shape[-1]) for value in targets[level]], axis=0
        )
        per_target: dict[str, Any] = {}
        std = level_stats.std.numpy()
        for index, name in enumerate(names):
            error = flat_prediction[:, index] - flat_target[:, index]
            graph_mae = float(
                np.mean(
                    [
                        np.mean(np.abs(pred[..., index] - target[..., index]))
                        for pred, target in zip(predictions[level], targets[level], strict=True)
                    ]
                )
            )
            metrics: dict[str, Any] = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "normalized_mae": float(np.mean(np.abs(error)) / std[index]),
                "graph_macro_mae": graph_mae,
                "values": int(error.size),
            }
            if integer_targets:
                metrics["rounded_exact_accuracy"] = float(
                    np.mean(np.rint(flat_prediction[:, index]) == flat_target[:, index])
                )
            per_target[name] = metrics
            macro_normalized.append(metrics["normalized_mae"])
        result["levels"][level] = {
            "targets": per_target,
            "macro_mae": float(np.mean([value["mae"] for value in per_target.values()])),
            "macro_normalized_mae": float(
                np.mean([value["normalized_mae"] for value in per_target.values()])
            ),
        }
    result["macro_normalized_mae"] = float(np.mean(macro_normalized))
    return result


def clone_cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})


__all__ = [
    "TargetStats",
    "TrainSettings",
    "clone_cpu_state",
    "cuda_autocast",
    "evaluate_supervised",
    "fit_target_stats",
    "make_grad_scaler",
    "normalized_multitask_loss",
    "resolve_device",
    "runtime_environment",
    "seed_everything",
    "train_supervised",
    "validation_loss",
]
````

# research/cycle_pe/run.py

````python
"""Run the graph-family structural probe for static cycle PE."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from .synthetic import PROBE_VARIANTS, make_graph_family_split, stack_probe_graphs

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class LogisticProbe:
    """A deterministic linear probe fitted on frozen static PE features."""

    mean: FloatArray
    scale: FloatArray
    weights: FloatArray
    bias: float

    def decision_function(self, features: FloatArray) -> FloatArray:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.weights + self.bias

    def predict_proba(self, features: FloatArray) -> FloatArray:
        logits = np.clip(self.decision_function(features), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic_probe(
    features: FloatArray,
    targets: IntArray,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
) -> LogisticProbe:
    """Fit a full-batch logistic probe without modifying the PE extractor."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("features/targets have incompatible shapes")
    if steps < 1 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("invalid probe optimizer configuration")
    if np.unique(y).size != 2:
        raise ValueError("the training split must contain both edge classes")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float64)
    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    bias = float(np.log(prevalence / (1.0 - prevalence)))

    for step in range(steps):
        logits = np.clip(standardized @ weights + bias, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = probabilities - y
        gradient_weights = standardized.T @ residual / len(y) + l2 * weights
        gradient_bias = float(residual.mean())
        # Mild decay makes the small deterministic optimizer insensitive to the
        # exact requested step count while preserving a fast smoke configuration.
        rate = learning_rate / np.sqrt(1.0 + step / 250.0)
        weights -= rate * gradient_weights
        bias -= rate * gradient_bias
    return LogisticProbe(mean, scale, weights, bias)


def _binary_metrics(
    targets: IntArray,
    probabilities: FloatArray,
    graph_ids: tuple[str, ...],
) -> dict[str, float]:
    y = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predictions = (scores >= 0.5).astype(np.int64)
    positives = y == 1
    negatives = ~positives
    true_positive = int(np.count_nonzero(predictions[positives] == 1))
    true_negative = int(np.count_nonzero(predictions[negatives] == 0))
    false_positive = int(np.count_nonzero(predictions[negatives] == 1))
    false_negative = int(np.count_nonzero(predictions[positives] == 0))
    recall_positive = true_positive / max(1, true_positive + false_negative)
    recall_negative = true_negative / max(1, true_negative + false_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    f1 = 2.0 * precision * recall_positive / max(1e-12, precision + recall_positive)

    positive_scores = scores[positives]
    negative_scores = scores[negatives]
    if len(positive_scores) and len(negative_scores):
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        auroc = float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))
    else:
        auroc = float("nan")

    graph_accuracy = []
    ids = np.asarray(graph_ids, dtype=object)
    for graph_id in dict.fromkeys(graph_ids):
        mask = ids == graph_id
        graph_accuracy.append(float(np.mean(predictions[mask] == y[mask])))
    return {
        "accuracy": float(np.mean(predictions == y)),
        "balanced_accuracy": float(0.5 * (recall_positive + recall_negative)),
        "f1": float(f1),
        "auroc": auroc,
        "macro_graph_accuracy": float(np.mean(graph_accuracy)),
    }


def run_structural_probe(
    *,
    samples_per_family: int = 8,
    seed: int = 7,
    max_cycles: int = 12,
    steps: int = 1200,
    learning_rate: float = 0.15,
    l2: float = 1e-3,
) -> dict[str, Any]:
    """Compare degree-only features with three frozen static cycle PEs."""

    train_graphs, test_graphs = make_graph_family_split(
        samples_per_family=samples_per_family,
        seed=seed,
    )
    results: dict[str, Any] = {
        "scope": "static_graph_cycle_pe_only",
        "task": "edge_is_in_any_cycle",
        "split": {
            "train_families": sorted({graph.family for graph in train_graphs}),
            "test_families": sorted({graph.family for graph in test_graphs}),
            "train_graphs": len(train_graphs),
            "test_graphs": len(test_graphs),
        },
        "variants": {},
        "notes": {
            "raw": "diagnostic; column order/sign and spanning-tree dependent",
            "cycle_set": "invariant only to sign/permutation of a fixed cycle set",
            "projector_leverage": "basis-invariant prior-style baseline, not novelty",
        },
    }

    for variant in PROBE_VARIANTS:
        train = stack_probe_graphs(train_graphs, variant, max_cycles=max_cycles)
        test = stack_probe_graphs(test_graphs, variant, max_cycles=max_cycles)
        probe = fit_logistic_probe(
            train.features,
            train.targets,
            steps=steps,
            learning_rate=learning_rate,
            l2=l2,
        )
        results["variants"][variant] = {
            "feature_dim": int(train.features.shape[1]),
            "train_edges": int(train.features.shape[0]),
            "test_edges": int(test.features.shape[0]),
            "train": _binary_metrics(
                train.targets, probe.predict_proba(train.features), train.graph_ids
            ),
            "test": _binary_metrics(
                test.targets, probe.predict_proba(test.features), test.graph_ids
            ),
        }
    return results


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _read_config(config_path)
    probe_config = config.get("probe", {})
    results = run_structural_probe(
        samples_per_family=int(config.get("samples_per_family", 8)),
        seed=int(config.get("seed", 7)),
        max_cycles=int(config.get("max_cycles", 12)),
        steps=int(probe_config.get("steps", 1200)),
        learning_rate=float(probe_config.get("learning_rate", 0.15)),
        l2=float(probe_config.get("l2", 1e-3)),
    )
    if args.output is not None:
        output = args.output.expanduser()
    else:
        output = Path(config.get("output", "results/summary.json")).expanduser()
        if not output.is_absolute():
            output = config_path.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
````

# research/cycle_pe/synthetic.py

````python
"""Graph-family splits for static cycle-PE structural probes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from chartgat.algebra import incidence_matrix
from chartgat.graphs import make_connected_graph, spanning_tree_indices

from .features import degree_only_edge_features, static_cycle_feature_bundle

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PROBE_VARIANTS = (
    "degree_only",
    "degree_plus_raw",
    "degree_plus_cycle_set",
    "degree_plus_projector_leverage",
)


@dataclass(frozen=True)
class ProbeGraph:
    """One topology-only graph example in a graph-family split."""

    family: str
    split: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ProbeMatrix:
    """Stacked edge examples for one split and one PE variant."""

    features: FloatArray
    targets: IntArray
    graph_ids: tuple[str, ...]


def _canonical_edges(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    canonical = {tuple(sorted((int(u), int(v)))) for u, v in edges if u != v}
    return tuple(sorted(canonical))


def _cycle_edges(offset: int, size: int) -> list[tuple[int, int]]:
    if size < 3:
        raise ValueError("cycle size must be at least three")
    nodes = list(range(offset, offset + size))
    return [(nodes[index], nodes[(index + 1) % size]) for index in range(size)]


def cycle_with_tail(cycle_size: int, tail_length: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    """A single cycle with an attached bridge path."""

    if tail_length < 1:
        raise ValueError("tail_length must be positive")
    edges = _cycle_edges(0, cycle_size)
    previous = 0
    for node in range(cycle_size, cycle_size + tail_length):
        edges.append((previous, node))
        previous = node
    return cycle_size + tail_length, _canonical_edges(edges)


def double_cycle_bridge(
    left_size: int,
    right_size: int,
    bridge_length: int,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Two disjoint cycles connected by a bridge path."""

    if bridge_length < 1:
        raise ValueError("bridge_length must be positive")
    intermediate_count = bridge_length - 1
    right_offset = left_size + intermediate_count
    edges = _cycle_edges(0, left_size)
    edges.extend(_cycle_edges(right_offset, right_size))
    path = [0, *range(left_size, right_offset), right_offset]
    edges.extend(zip(path[:-1], path[1:], strict=True))
    return right_offset + right_size, _canonical_edges(edges)


def cactus_cycle_chain(
    cycle_sizes: Sequence[int],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """A chain of vertex-disjoint cycles connected by single bridge edges."""

    if len(cycle_sizes) < 2:
        raise ValueError("a cactus chain needs at least two cycles")
    edges: list[tuple[int, int]] = []
    offsets: list[int] = []
    offset = 0
    for size in cycle_sizes:
        offsets.append(offset)
        edges.extend(_cycle_edges(offset, int(size)))
        offset += int(size)
    edges.extend((offsets[index], offsets[index + 1]) for index in range(len(offsets) - 1))
    return offset, _canonical_edges(edges)


def bridge_cycle_labels(graph: ProbeGraph) -> IntArray:
    """Label an edge one iff it belongs to at least one undirected cycle."""

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))
    nx_graph.add_edges_from(graph.edges)
    if not nx.is_connected(nx_graph):
        raise ValueError("probe graphs must be connected")
    bridges = {tuple(sorted(edge)) for edge in nx.bridges(nx_graph)}
    return np.asarray([int(edge not in bridges) for edge in graph.edges], dtype=np.int64)


def make_graph_family_split(
    *,
    samples_per_family: int = 8,
    seed: int = 0,
) -> tuple[list[ProbeGraph], list[ProbeGraph]]:
    """Build train/test sets with disjoint graph-generator families.

    Training uses ``cycle_tail`` and ``random_tree_plus_chords``.  Testing uses
    only the unseen ``double_cycle_bridge`` and ``cactus_cycle_chain`` families.
    The split therefore probes family transfer rather than random edge leakage.
    """

    if samples_per_family < 1:
        raise ValueError("samples_per_family must be positive")
    rng = np.random.default_rng(seed)
    train: list[ProbeGraph] = []
    test: list[ProbeGraph] = []

    for sample in range(samples_per_family):
        cycle_size = int(rng.integers(4, 9))
        tail_length = int(rng.integers(3, 8))
        node_count, edges = cycle_with_tail(cycle_size, tail_length)
        train.append(ProbeGraph("cycle_tail", "train", node_count, edges))

        node_count = int(rng.integers(9, 16))
        extra_edges = int(rng.integers(1, min(4, node_count - 2)))
        edges = tuple(make_connected_graph(node_count, extra_edges, seed=seed * 1009 + sample))
        train.append(ProbeGraph("random_tree_plus_chords", "train", node_count, edges))

        left_size = int(rng.integers(3, 8))
        right_size = int(rng.integers(3, 8))
        bridge_length = int(rng.integers(2, 7))
        node_count, edges = double_cycle_bridge(left_size, right_size, bridge_length)
        test.append(ProbeGraph("double_cycle_bridge", "test", node_count, edges))

        sizes = tuple(int(value) for value in rng.integers(3, 8, size=3))
        node_count, edges = cactus_cycle_chain(sizes)
        test.append(ProbeGraph("cactus_cycle_chain", "test", node_count, edges))

    return train, test


def graph_feature_matrix(
    graph: ProbeGraph,
    variant: str,
    *,
    max_cycles: int,
) -> FloatArray:
    """Create one static edge-feature matrix for the requested PE variant."""

    if variant not in PROBE_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {PROBE_VARIANTS}")
    incidence = incidence_matrix(graph.num_nodes, graph.edges)
    tree = spanning_tree_indices(graph.num_nodes, graph.edges, mode="bfs")
    cycle = static_cycle_feature_bundle(incidence, tree, max_cycles=max_cycles)
    degree = degree_only_edge_features(graph.num_nodes, graph.edges)
    if variant == "degree_only":
        return degree
    if variant == "degree_plus_raw":
        return np.concatenate((degree, cycle["raw_padded"]), axis=1)
    if variant == "degree_plus_cycle_set":
        return np.concatenate((degree, cycle["cycle_set"]), axis=1)
    return np.concatenate((degree, cycle["projector_leverage"]), axis=1)


def stack_probe_graphs(
    graphs: Sequence[ProbeGraph],
    variant: str,
    *,
    max_cycles: int,
) -> ProbeMatrix:
    """Stack edges across graphs while retaining graph identifiers."""

    if not graphs:
        raise ValueError("graphs must not be empty")
    feature_parts: list[FloatArray] = []
    target_parts: list[IntArray] = []
    graph_ids: list[str] = []
    for index, graph in enumerate(graphs):
        features = graph_feature_matrix(graph, variant, max_cycles=max_cycles)
        targets = bridge_cycle_labels(graph)
        feature_parts.append(features)
        target_parts.append(targets)
        graph_ids.extend([f"{graph.split}:{graph.family}:{index}"] * len(graph.edges))
    return ProbeMatrix(
        np.concatenate(feature_parts, axis=0),
        np.concatenate(target_parts, axis=0),
        tuple(graph_ids),
    )


__all__ = [
    "PROBE_VARIANTS",
    "ProbeGraph",
    "ProbeMatrix",
    "bridge_cycle_labels",
    "cactus_cycle_chain",
    "cycle_with_tail",
    "double_cycle_bridge",
    "graph_feature_matrix",
    "make_graph_family_split",
    "stack_probe_graphs",
]
````

# research/cycle_pe/tests/test_brec_protocol.py

````python
from __future__ import annotations

import torch

from research.cycle_pe.paper import (
    BREC_OFFICIAL_BATCH_SIZE,
    BREC_OFFICIAL_SEEDS,
    _aggregate_custom_brec_results,
    _aggregate_official_brec_results,
    _brec_reference_compatibility,
    _brec_settings,
    _effective_brec_protocol,
    brec_hotelling_t2,
    brec_rpc_decision,
    build_parser,
)


def test_hotelling_t2_matches_official_torch_reference_without_q_multiplier() -> None:
    generator = torch.Generator().manual_seed(919)
    embeddings = torch.randn((64, 16), generator=generator)
    difference = embeddings[0::2].T - embeddings[1::2].T
    mean = torch.mean(difference, dim=1).reshape(-1, 1)
    expected = (mean.T @ torch.linalg.pinv(torch.cov(difference)) @ mean).reshape(())

    actual = brec_hotelling_t2(embeddings)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert not torch.isclose(actual, expected * 32)


def test_rpc_decision_uses_official_isclose_and_reliability_gate() -> None:
    # Default torch rtol is intentionally retained by the official code.
    close = brec_rpc_decision(100.0, 100.0009, threshold=72.34)
    assert close == {"distinguished": False, "reliable": False, "successful": False}

    unreliable = brec_rpc_decision(100.0, 80.0, threshold=72.34)
    assert unreliable == {"distinguished": True, "reliable": False, "successful": False}

    success = brec_rpc_decision(100.0, 1.0, threshold=72.34)
    assert success == {"distinguished": True, "reliable": True, "successful": True}


def test_custom_pairwise_union_excludes_any_pair_with_reliability_failure() -> None:
    seeds = (100, 200)
    results = [
        {
            "pair_index": 0,
            "category": "Basic",
            "search_seed": 100,
            "status": "complete",
            "distinguished": True,
            "reliable": True,
            "successful": True,
        },
        {
            "pair_index": 0,
            "category": "Basic",
            "search_seed": 200,
            "status": "complete",
            "distinguished": False,
            "reliable": True,
            "successful": False,
        },
        {
            "pair_index": 1,
            "category": "Basic",
            "search_seed": 100,
            "status": "complete",
            "distinguished": True,
            "reliable": True,
            "successful": True,
        },
        {
            "pair_index": 1,
            "category": "Basic",
            "search_seed": 200,
            "status": "complete",
            "distinguished": True,
            "reliable": False,
            "successful": False,
        },
    ]
    summary = _aggregate_custom_brec_results(results, pair_indices=[0, 1], seeds=seeds)
    assert summary["protocol"] == "custom"
    assert summary["metric_name"] == "custom_pairwise_union"
    assert summary["successful_pairs"] == 1
    assert summary["reliability_failures"] == 1
    assert summary["per_pair"][0]["successful_pair"] is True
    assert summary["per_pair"][1]["successful_pair"] is False


def test_cli_defaults_to_the_official_ten_search_seeds() -> None:
    args = build_parser().parse_args([])
    assert tuple(int(value) for value in args.brec_seeds.split(",")) == BREC_OFFICIAL_SEEDS


def test_official_aggregation_reports_each_seed_without_union() -> None:
    seeds = (100, 200)
    results = [
        {
            "pair_index": pair_index,
            "category": "Basic",
            "search_seed": seed,
            "status": "complete",
            "distinguished": distinguished,
            "reliable": reliable,
            "successful": distinguished and reliable,
        }
        for seed, decisions in (
            (100, ((True, True), (False, True))),
            (200, ((True, True), (True, False))),
        )
        for pair_index, (distinguished, reliable) in enumerate(decisions)
    ]
    summary = _aggregate_official_brec_results(results, pair_indices=[0, 1], seeds=seeds)
    assert summary["protocol"] == "official"
    assert summary["merged_score"] is None
    assert summary["global_valid"] is False
    assert "repository-defined" in summary["global_valid_definition"]
    assert "not an upstream BREC metric" in summary["global_valid_definition"]
    assert summary["per_seed"]["100"]["Correct"] == 1
    assert summary["per_seed"]["100"]["Fail"] == 0
    assert summary["per_seed"]["100"]["Real_correct"] == 1
    assert summary["per_seed"]["200"]["Correct"] == 2
    assert summary["per_seed"]["200"]["Fail"] == 1
    assert summary["per_seed"]["200"]["Real_correct"] == 1


def test_official_mode_resolves_for_full_runs_and_forces_reference_settings() -> None:
    full = build_parser().parse_args(["--suite", "brec", "--batch-size", "99", "--amp"])
    full.brec_protocol = _effective_brec_protocol(full)
    assert full.brec_protocol == "official"
    settings = _brec_settings(full, torch.device("cpu"), full.brec_protocol)
    assert settings.batch_size == BREC_OFFICIAL_BATCH_SIZE == 16
    assert settings.epochs == 20
    assert settings.learning_rate == 1e-4
    assert settings.weight_decay == 1e-4
    assert settings.amp_requested is False
    assert settings.pin_memory_requested is False

    tiny = build_parser().parse_args(["--suite", "brec", "--tiny"])
    assert _effective_brec_protocol(tiny) == "custom"


def test_official_reference_compatibility_does_not_claim_differential_parity() -> None:
    compatibility = _brec_reference_compatibility("official")
    assert compatibility["static_constants_and_control_flow_compatible"] is True
    assert compatibility["differential_parity_verified"] is False
    assert "must not be interpreted" in compatibility["parity_note"]

    custom = _brec_reference_compatibility("custom")
    assert custom["static_constants_and_control_flow_compatible"] is False
    assert custom["differential_parity_verified"] is False
````

# research/cycle_pe/tests/test_cycle_pe.py

````python
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
from research.cycle_pe.run import run_structural_probe
from research.cycle_pe.synthetic import (
    PROBE_VARIANTS,
    ProbeGraph,
    bridge_cycle_labels,
    cycle_with_tail,
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


def test_bridge_labels_match_cycle_with_tail_construction() -> None:
    node_count, edges = cycle_with_tail(cycle_size=5, tail_length=4)
    graph = ProbeGraph("cycle_tail", "test", node_count, edges)
    labels = bridge_cycle_labels(graph)
    assert int(labels.sum()) == 5
    assert int((labels == 0).sum()) == 4


def test_graph_family_probe_smoke() -> None:
    results = run_structural_probe(
        samples_per_family=2,
        seed=3,
        max_cycles=8,
        steps=350,
        learning_rate=0.15,
        l2=1e-3,
    )
    assert results["scope"] == "static_graph_cycle_pe_only"
    assert set(results["variants"]) == set(PROBE_VARIANTS)
    assert set(results["split"]["train_families"]).isdisjoint(results["split"]["test_families"])
    for variant in PROBE_VARIANTS:
        metrics = results["variants"][variant]["test"]
        assert all(np.isfinite(value) for value in metrics.values())
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

    leverage = results["variants"]["degree_plus_projector_leverage"]["test"]
    degree = results["variants"]["degree_only"]["test"]
    assert leverage["balanced_accuracy"] >= 0.95
    assert leverage["balanced_accuracy"] >= degree["balanced_accuracy"]
````

# research/cycle_pe/tests/test_paper_adapters.py

````python
from __future__ import annotations

import builtins
import io
import zipfile

import networkx as nx
import numpy as np
import pytest
import torch

from research.cycle_pe import paper_adapters
from research.cycle_pe.paper_adapters import (
    BREC_OFFICIAL_RECORD_COUNT,
    BRECAdapter,
    download_brec_v3,
    find_brec_v3,
    load_brec_v3,
    validate_brec_v3,
    write_tiny_brec_fixture,
)


def test_tiny_brec_fixture_matches_lazy_rpc_layout(tmp_path) -> None:
    path = write_tiny_brec_fixture(
        tmp_path / "BREC" / "Data" / "raw" / "brec_v3.npy", num_relabel=2
    )
    assert find_brec_v3(tmp_path) == path
    adapter = BRECAdapter(path, num_relabel=2)
    assert adapter.pair_count == 2
    pair = adapter.load_pair(0)
    assert len(pair.train_test) == 4
    assert len(pair.reliability) == 4
    left = nx.Graph(pair.train_test[0].edges)
    right = nx.Graph(pair.train_test[1].edges)
    assert not nx.is_isomorphic(left, right)
    assert nx.is_isomorphic(
        nx.Graph(pair.reliability[0].edges), nx.Graph(pair.reliability[1].edges)
    )
    assert adapter.metadata["sha256"]
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_official_brec_validation_enforces_400_pair_record_layout(tmp_path) -> None:
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.asarray([b"A_"] * 128, dtype=object), allow_pickle=True)
    with pytest.raises(RuntimeError, match="51,200 records"):
        validate_brec_v3(wrong, protocol="official")

    official_shape = tmp_path / "official-shape.npy"
    np.save(
        official_shape,
        np.asarray([b"A_"] * BREC_OFFICIAL_RECORD_COUNT, dtype=object),
        allow_pickle=True,
    )
    metadata = validate_brec_v3(official_shape, protocol="official")
    assert metadata["records"] == 51_200
    assert metadata["pair_count"] == 400
    assert metadata["official_shape_validated"] is True
    assert metadata["official_source_hash_pinned"] is False


def test_missing_brec_artifact_error_is_actionable(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="GraphPKU/BREC"):
        find_brec_v3(tmp_path)


def test_brec_full_load_is_fail_closed_without_opt_in(monkeypatch, tmp_path) -> None:
    def unexpected_network(*args, **kwargs):
        raise AssertionError("network must not be touched without --allow-download")

    monkeypatch.setattr(paper_adapters.urllib.request, "urlopen", unexpected_network)
    with pytest.raises(FileNotFoundError, match="--allow-download"):
        load_brec_v3(tmp_path, allow_download=False)


def test_brec_tiny_fixture_is_always_offline(monkeypatch, tmp_path) -> None:
    def unexpected_network(*args, **kwargs):
        raise AssertionError("tiny BREC fixture must never use the network")

    monkeypatch.setattr(paper_adapters.urllib.request, "urlopen", unexpected_network)
    adapter = load_brec_v3(
        tmp_path,
        num_relabel=2,
        allow_download=True,
        tiny=True,
    )
    assert adapter.pair_count == 2
    assert adapter.metadata["offline_fixture"] is True


class _FakeHTTPResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def geturl(self) -> str:
        return paper_adapters.BREC_RAW_URL

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _zip_payload(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def test_explicit_brec_download_extracts_only_valid_npy(monkeypatch, tmp_path) -> None:
    fixture = write_tiny_brec_fixture(tmp_path / "source.npy", num_relabel=2)
    payload = _zip_payload({"BREC/Data/raw/brec_v3.npy": fixture.read_bytes()})
    monkeypatch.setattr(
        paper_adapters.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )

    target = download_brec_v3(tmp_path / "data")
    assert target == (tmp_path / "data" / "BREC" / "Data" / "raw" / "brec_v3.npy")
    assert target.read_bytes().startswith(b"\x93NUMPY")
    metadata = target.with_name("brec_v3.download.json").read_text(encoding="utf-8")
    assert "archive_sha256" in metadata and "brec_v3_sha256" in metadata
    records = np.load(target, allow_pickle=True)
    assert records.shape == (16,)


def test_brec_download_rejects_archive_path_traversal(monkeypatch, tmp_path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray([b"A_"], dtype=object), allow_pickle=True)
    payload = _zip_payload(
        {
            "BREC/Data/raw/brec_v3.npy": buffer.getvalue(),
            "../escaped.txt": b"unsafe",
        }
    )
    monkeypatch.setattr(
        paper_adapters.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(payload),
    )
    data_root = tmp_path / "data"
    with pytest.raises(RuntimeError, match="unsafe path"):
        download_brec_v3(data_root)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (data_root / "BREC" / "Data" / "raw" / "brec_v3.npy").exists()


def test_missing_pyg_error_has_cuda_install_guidance(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("torch_geometric"):
            raise ImportError("fixture blocks optional PyG")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="CUDA runtime") as error:
        paper_adapters._require_pyg_zinc()
    assert "torch-geometric" in str(error.value)
    assert "installation.html" in str(error.value)


def test_zinc_download_requires_explicit_opt_in(monkeypatch, tmp_path) -> None:
    class UnexpectedZincConstruction:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("adapter must reject before PyG starts a download")

    monkeypatch.setattr(paper_adapters, "_require_pyg_zinc", lambda: UnexpectedZincConstruction)
    with pytest.raises(FileNotFoundError, match="--allow-download"):
        paper_adapters.load_zinc12k(tmp_path, tiny=True, allow_download=False)


def test_zinc_adapter_uses_official_split_names_without_network(monkeypatch, tmp_path) -> None:
    processed = tmp_path / "ZINC12K" / "subset" / "processed"
    processed.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (processed / f"{split}.pt").touch()

    class FakeData:
        num_nodes = 3
        x = torch.tensor([[0], [2], [4]])
        edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
        edge_attr = torch.tensor([1, 1, 2, 2, 3, 3])
        y = torch.tensor([0.75])

    calls: list[tuple[bool, str]] = []

    class FakeZinc:
        def __init__(self, *, root, subset, split) -> None:
            calls.append((subset, split))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index):
            assert index == 0
            return FakeData()

    monkeypatch.setattr(paper_adapters, "_require_pyg_zinc", lambda: FakeZinc)
    bundle = paper_adapters.load_zinc12k(tmp_path, tiny=True, allow_download=False)
    assert calls == [(True, "train"), (True, "val"), (True, "test")]
    assert {name: len(graphs) for name, graphs in bundle.splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    graph = bundle.splits["train"][0]
    assert graph.edges == ((0, 1), (0, 2), (1, 2))
    assert graph.node_features is not None and graph.node_features.shape == (3, 28)
    assert graph.edge_features is not None and graph.edge_features.shape == (3, 4)
    assert graph.graph_targets is not None and graph.graph_targets.tolist() == [0.75]
    assert set(bundle.metadata["cache_sha256"]) == {
        "subset/processed/train.pt",
        "subset/processed/val.pt",
        "subset/processed/test.pt",
    }
````

# research/cycle_pe/tests/test_paper_cli.py

````python
from __future__ import annotations

import json

import pytest
import torch

from research.cycle_pe import paper

main = paper.main


def test_core_tiny_cli_trains_without_download_and_writes_manifest(tmp_path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "runs"
    exit_code = main(
        [
            "--suite",
            "core",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_root),
            "--device",
            "cpu",
            "--seed",
            "11",
            "--data-seed",
            "19",
            "--model-seed",
            "37",
            "--tiny",
            "--epochs",
            "1",
            "--batch-size",
            "5",
            "--variants",
            "no_pe",
            "--core-targets",
            "edge",
        ]
    )
    assert exit_code == 0
    manifest_path = output_root / "core" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["raw_width"] == manifest["split_statistics"]["train"]["cycle_rank_max"]
    assert manifest["seed_axes"] == {"data": 19, "split": 19, "chart": 19, "model": 37}
    assert manifest["dataset_metadata"]["data_seed"] == 19
    assert manifest["seed_axis_policy"]["chart"]["status"] == "not_applicable"
    assert "'train' only" in manifest["raw_width_policy"]
    assert "never truncated" in manifest["raw_width_policy"]
    assert manifest["training"]["amp_effective"] is False
    assert manifest["training"]["workers"] == 0
    assert manifest["training"]["pin_memory_effective"] is False
    assert manifest["training"]["non_blocking_effective"] is False
    assert manifest["experiments"]["edge"]["no_pe"]["reported_split"] == "id_test"
    artifacts = manifest["artifacts"]
    assert "edge/no_pe/model.pt" in artifacts
    assert "edge/no_pe/metrics.json" in artifacts
    metrics = json.loads(
        (output_root / "core" / "edge" / "no_pe" / "metrics.json").read_text(encoding="utf-8")
    )
    assert set(metrics) == {
        "train",
        "validation",
        "id_test",
        "size_ood",
        "family_ood",
    }
    assert "edge_shortest_cycle" in metrics["size_ood"]["levels"]["edge"]["targets"]
    checkpoint = torch.load(
        output_root / "core" / "edge" / "no_pe" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["model_seed"] == 37


def test_core_tiny_prepare_only_stops_before_training(tmp_path) -> None:
    output_root = tmp_path / "runs"
    assert (
        main(
            [
                "--suite",
                "core",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--tiny",
                "--prepare-only",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "core" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_only"] is True
    assert manifest["variants"] == ["no_pe", "raw", "set", "projector"]
    assert manifest["experiments"] == {}
    assert manifest["runtime_environment"]["workers"] == 1
    assert not list((output_root / "core").glob("*/model.pt"))


def test_brec_tiny_prepare_only_uses_offline_rpc_fixture(tmp_path) -> None:
    output_root = tmp_path / "runs"
    assert (
        main(
            [
                "--suite",
                "brec",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--tiny",
                "--prepare-only",
                "--brec-num-relabel",
                "2",
                "--brec-threshold",
                "1",
                "--brec-seeds",
                "100,200",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "brec" / "manifest.json").read_text("utf-8"))
    assert manifest["dataset_metadata"]["offline_fixture"] is True
    assert manifest["brec_protocol"]["effective"] == "custom"
    assert "official_training_reference_matched" not in manifest["brec_protocol"]
    compatibility = manifest["brec_protocol"]["official_reference_compatibility"]
    assert compatibility["static_constants_and_control_flow_compatible"] is False
    assert compatibility["differential_parity_verified"] is False
    assert manifest["rpc_reference"]["search_seeds"] == [100, 200]
    assert manifest["seed_axis_policy"]["model"]["used"] is False
    assert manifest["seed_axis_policy"]["protocol"]["values"] == [100, 200]
    assert len(manifest["preparation_checks"]) == 2


def test_brec_tiny_custom_training_is_labeled_separately(tmp_path) -> None:
    output_root = tmp_path / "runs"
    assert (
        main(
            [
                "--suite",
                "brec",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--tiny",
                "--brec-protocol",
                "custom",
                "--brec-num-relabel",
                "2",
                "--brec-threshold",
                "1",
                "--brec-seeds",
                "100",
                "--variants",
                "no_pe",
                "--epochs",
                "1",
                "--batch-size",
                "4",
            ]
        )
        == 0
    )
    metrics = json.loads(
        (output_root / "brec" / "no_pe" / "metrics.json").read_text(encoding="utf-8")
    )
    pairs = json.loads((output_root / "brec" / "no_pe" / "pairs.json").read_text(encoding="utf-8"))
    assert metrics["protocol"] == "custom"
    assert metrics["metric_name"] == "custom_pairwise_union"
    assert pairs[0]["rng_scope"] == "derived_per_pair_variant_search_seed"
    assert pairs[0]["pair_shuffle"] is True
    assert pairs[0]["gradient_clip_norm"] == 5.0


def test_all_suite_forwards_prepare_and_download_policy(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, bool, bool]] = []

    def runner(name):
        def run(args, device):
            calls.append((name, args.prepare_only, args.allow_download))
            return {"variants": []}

        return run

    monkeypatch.setattr(paper, "run_core", runner("core"))
    monkeypatch.setattr(paper, "run_brec", runner("brec"))
    monkeypatch.setattr(paper, "run_zinc", runner("zinc"))
    assert (
        main(
            [
                "--suite",
                "all",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / "runs"),
                "--device",
                "cpu",
                "--prepare-only",
                "--allow-download",
            ]
        )
        == 0
    )
    assert calls == [
        ("core", True, True),
        ("brec", True, True),
        ("zinc", True, True),
    ]


def test_existing_output_collision_is_rejected_without_modification(tmp_path) -> None:
    output_root = tmp_path / "existing"
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("user artifact", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--suite",
                "core",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--tiny",
                "--prepare-only",
            ]
        )
    assert marker.read_text(encoding="utf-8") == "user artifact"
    assert list(output_root.iterdir()) == [marker]


def test_all_suite_failure_preserves_completed_and_removes_failed_suite_artifacts(
    monkeypatch, tmp_path
) -> None:
    def successful_core(args, device):
        partial = args.output_dir / "core" / "partial.txt"
        partial.parent.mkdir(parents=True)
        partial.write_text("partial", encoding="utf-8")
        return {"variants": []}

    def failing_brec(args, device):
        partial = args.output_dir / "brec" / "partial.txt"
        partial.parent.mkdir(parents=True)
        partial.write_text("partial", encoding="utf-8")
        raise RuntimeError("fixture BREC failure")

    def unexpected_zinc(args, device):
        raise AssertionError("ZINC must not run after BREC fails")

    monkeypatch.setattr(paper, "run_core", successful_core)
    monkeypatch.setattr(paper, "run_brec", failing_brec)
    monkeypatch.setattr(paper, "run_zinc", unexpected_zinc)
    output_root = tmp_path / "runs"
    with pytest.raises(SystemExit):
        main(
            [
                "--suite",
                "all",
                "--data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(output_root),
                "--device",
                "cpu",
                "--prepare-only",
            ]
        )
    assert sorted(path.name for path in output_root.iterdir()) == ["core", "run_manifest.json"]
    assert (output_root / "core" / "partial.txt").read_text(encoding="utf-8") == "partial"
    assert not (output_root / "brec").exists()
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_suite"] == "brec"
    assert manifest["completed_suites"] == ["core"]
````

# research/cycle_pe/tests/test_paper_data.py

````python
from __future__ import annotations

import numpy as np

from research.cycle_pe.paper_data import (
    CORE_SPLITS,
    canonical_edges,
    cycle_count_split_sizes,
    enumerate_short_cycles,
    exact_cycle_targets,
    load_or_generate_cycle_count_ood,
)


def test_full_cycle_count_protocol_contains_exactly_twenty_thousand_graphs() -> None:
    sizes = cycle_count_split_sizes(tiny=False)
    assert sizes == {
        "train": 10_000,
        "validation": 2_000,
        "id_test": 2_000,
        "size_ood": 3_000,
        "family_ood": 3_000,
    }
    assert sum(sizes.values()) == 20_000


def test_exact_short_cycle_targets_cover_edge_node_and_graph_levels() -> None:
    # A triangle and square share vertex 2; edge (5, 6) is a bridge.
    edges = canonical_edges(((0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (2, 5), (5, 6)))
    cycles = enumerate_short_cycles(7, edges)
    assert [(len(cycle), cycle) for cycle in cycles] == [
        (3, (0, 1, 2)),
        (4, (2, 3, 4, 5)),
    ]

    edge, node, graph = exact_cycle_targets(7, edges)
    np.testing.assert_array_equal(graph, np.asarray([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(node[2], np.asarray([1.0, 1.0, 0.0, 0.0]))
    bridge_index = edges.index((5, 6))
    np.testing.assert_array_equal(edge[bridge_index], np.zeros(6))
    triangle_index = edges.index((0, 1))
    np.testing.assert_array_equal(edge[triangle_index], np.asarray([1.0, 0.0, 0.0, 0.0, 3.0, 1.0]))

    seven_cycle = canonical_edges((node, (node + 1) % 7) for node in range(7))
    long_edge, _, long_graph = exact_cycle_targets(7, seven_cycle)
    np.testing.assert_array_equal(long_graph, np.zeros(4))
    np.testing.assert_array_equal(long_edge[:, 4], np.full(7, 7.0))
    np.testing.assert_array_equal(long_edge[:, 5], np.zeros(7))


def test_cycle_count_ood_cache_and_splits_are_deterministic(tmp_path) -> None:
    first = load_or_generate_cycle_count_ood(tmp_path, seed=19, tiny=True)
    second = load_or_generate_cycle_count_ood(tmp_path, seed=19, tiny=True)
    independent = load_or_generate_cycle_count_ood(
        tmp_path / "independent-root", seed=19, tiny=True
    )

    assert first.cache_path == second.cache_path
    assert first.cache_sha256 == second.cache_sha256
    assert first.cache_sha256 == independent.cache_sha256
    assert tuple(first.splits) == CORE_SPLITS
    assert {name: len(graphs) for name, graphs in first.splits.items()} == {
        "train": 10,
        "validation": 4,
        "id_test": 4,
        "size_ood": 4,
        "family_ood": 4,
    }
    assert [graph.graph_id for graph in first.splits["train"]] == [
        graph.graph_id for graph in second.splits["train"]
    ]
    assert min(graph.num_nodes for graph in first.splits["size_ood"]) > max(
        graph.num_nodes for graph in first.splits["train"]
    )
    training_families = {graph.family for graph in first.splits["train"]}
    family_ood = {graph.family for graph in first.splits["family_ood"]}
    assert training_families.isdisjoint(family_ood)
    for split in CORE_SPLITS:
        for left, right in zip(first.splits[split], second.splits[split], strict=True):
            np.testing.assert_array_equal(left.edge_targets, right.edge_targets)
            np.testing.assert_array_equal(left.node_targets, right.node_targets)
            np.testing.assert_array_equal(left.graph_targets, right.graph_targets)
````

# research/cycle_pe/tests/test_paper_model.py

````python
from __future__ import annotations

import numpy as np
import pytest
import torch

import research.cycle_pe.paper_model as paper_model_module
from chartgat.algebra import incidence_matrix
from chartgat.graphs import spanning_tree_indices
from research.cycle_pe.features import cycle_projector, static_fundamental_basis
from research.cycle_pe.paper_data import PaperGraph, canonical_edges
from research.cycle_pe.paper_model import (
    PE_VARIANTS,
    PaperCycleModel,
    RawCycleRankOverflow,
    StaticPEEncoder,
    prepare_splits,
)


def _graph(name: str, num_nodes: int, edges: tuple[tuple[int, int], ...]) -> PaperGraph:
    return PaperGraph(
        graph_id=name,
        split="test",
        family="fixture",
        num_nodes=num_nodes,
        edges=canonical_edges(edges),
        edge_targets=np.zeros((len(edges), 1), dtype=np.float64),
        node_targets=np.zeros((num_nodes, 1), dtype=np.float64),
        graph_targets=np.zeros(1, dtype=np.float64),
    )


def test_paper_preparation_removes_fixed_max_cycles_and_batches_variable_beta() -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))
    dense_edges = tuple((u, v) for u in range(8) for v in range(u + 1, 8))
    dense = _graph("dense", 8, dense_edges)
    assert dense.beta == 21

    prepared, raw_width = prepare_splits({"train": [triangle, dense]}, fit_split="train")
    assert raw_width == 21
    assert raw_width > 12
    assert prepared["train"][0].raw_basis.shape == (3, 1)
    assert prepared["train"][1].raw_basis.shape == (28, 21)

    for variant in PE_VARIANTS:
        torch.manual_seed(3)
        model = PaperCycleModel(
            variant=variant,
            raw_width=raw_width,
            node_input_dim=2,
            edge_input_dim=4,
            edge_output_dim=1,
            node_output_dim=1,
            graph_output_dim=1,
            hidden_dim=16,
            pe_dim=8,
            layers=1,
        )
        outputs = model(prepared["train"])
        separate = [model.forward_graph(graph) for graph in prepared["train"]]
        assert outputs[0].edge is not None and outputs[0].edge.shape == (3, 1)
        assert outputs[1].edge is not None and outputs[1].edge.shape == (28, 1)
        assert outputs[0].node is not None and outputs[0].node.shape == (3, 1)
        assert outputs[1].graph is not None and outputs[1].graph.shape == (1,)
        assert outputs[0].embedding.shape == (16,)
        for batched, single in zip(outputs, separate, strict=True):
            assert batched.edge is not None and single.edge is not None
            assert batched.node is not None and single.node is not None
            assert batched.graph is not None and single.graph is not None
            torch.testing.assert_close(batched.edge, single.edge)
            torch.testing.assert_close(batched.node, single.node)
            torch.testing.assert_close(batched.graph, single.graph)
            torch.testing.assert_close(batched.embedding, single.embedding)


def test_non_projector_variants_do_not_materialize_dense_projector(monkeypatch) -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))

    def forbidden_projector(_basis):
        raise AssertionError("dense projector should be lazy")

    monkeypatch.setattr(paper_model_module, "cycle_projector", forbidden_projector)
    prepared, _ = prepare_splits(
        {"train": [triangle]},
        fit_split="train",
        required_variants=("no_pe", "set"),
    )
    graph = prepared["train"][0]
    assert graph.cycle_set is not None
    assert graph.projector is None


def test_raw_width_is_fit_on_train_only_and_ood_is_never_truncated() -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))
    dense_edges = tuple((u, v) for u in range(8) for v in range(u + 1, 8))
    dense = _graph("dense", 8, dense_edges)
    prepared, raw_width = prepare_splits(
        {"train": [triangle], "size_ood": [dense]},
        fit_split="train",
    )
    assert raw_width == 1
    assert prepared["size_ood"][0].raw_basis.shape[1] == 21

    raw_encoder = StaticPEEncoder("raw", raw_width=raw_width, pe_dim=4)
    graph = prepared["size_ood"][0]
    with pytest.raises(RawCycleRankOverflow, match="train-fitted raw width 1"):
        raw_encoder(graph.raw_basis, graph.cycle_set, graph.projector)

    projector_encoder = StaticPEEncoder("projector", raw_width=raw_width, pe_dim=4)
    output = projector_encoder(graph.raw_basis, graph.cycle_set, graph.projector)
    assert output.shape == (28, 4)


def test_projector_encoder_is_basis_change_and_orientation_invariant() -> None:
    edges = ((0, 1), (1, 2), (2, 3), (0, 3), (0, 2))
    incidence = incidence_matrix(4, edges)
    tree = spanning_tree_indices(4, edges, mode="bfs")
    basis = static_fundamental_basis(incidence, tree)
    changed = basis @ np.asarray([[1.0, 2.0], [-1.0, 1.0]])
    projector = cycle_projector(basis)
    changed_projector = cycle_projector(changed)

    torch.manual_seed(7)
    encoder = StaticPEEncoder("projector", raw_width=2, pe_dim=9)
    raw = torch.zeros((5, 2))
    cycle_set = torch.zeros((5, 6))
    original = encoder(raw, cycle_set, torch.as_tensor(projector, dtype=torch.float32))
    transformed = encoder(raw, cycle_set, torch.as_tensor(changed_projector, dtype=torch.float32))
    torch.testing.assert_close(original, transformed, atol=2e-6, rtol=2e-6)

    signs = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])
    oriented = signs[:, None] * projector * signs[None, :]
    flipped = encoder(raw, cycle_set, torch.as_tensor(oriented, dtype=torch.float32))
    torch.testing.assert_close(original, flipped, atol=2e-6, rtol=2e-6)

    permutation = np.asarray([3, 0, 4, 1, 2])
    permuted_projector = projector[np.ix_(permutation, permutation)]
    permuted = encoder(
        raw[permutation],
        cycle_set[permutation],
        torch.as_tensor(permuted_projector, dtype=torch.float32),
    )
    torch.testing.assert_close(original[permutation], permuted, atol=2e-6, rtol=2e-6)


def test_projector_model_handles_connected_singleton_without_edges() -> None:
    singleton = PaperGraph(
        graph_id="singleton",
        split="test",
        family="fixture",
        num_nodes=1,
        edges=(),
        graph_targets=np.asarray([0.0]),
    )
    prepared, raw_width = prepare_splits({"test": [singleton]})
    model = PaperCycleModel(
        variant="projector",
        raw_width=raw_width,
        node_input_dim=2,
        edge_input_dim=4,
        edge_output_dim=0,
        node_output_dim=0,
        graph_output_dim=1,
        hidden_dim=12,
        pe_dim=6,
        layers=1,
    )
    output = model(prepared["test"])[0]
    assert output.graph is not None and torch.isfinite(output.graph).all()
````

# research/cycle_pe/tests/test_seed_axes.py

````python
from __future__ import annotations

import json

import torch

from chartgat.seeds import SeedAxes
from research.cycle_pe.paper import _seed_axis_policy, _settings, build_parser, main


def test_cycle_settings_use_model_axis_and_record_not_applicable_axes() -> None:
    args = build_parser().parse_args(
        [
            "--seed",
            "7",
            "--data-seed",
            "11",
            "--split-seed",
            "13",
            "--chart-seed",
            "17",
            "--model-seed",
            "19",
        ]
    )
    settings = _settings(args, torch.device("cpu"), "core")
    assert settings.seed == 19

    axes = SeedAxes(data=11, split=13, chart=17, model=19)
    core = _seed_axis_policy("core", axes)
    assert core["data"]["used"] is True
    assert core["split"]["status"] == "not_applicable"
    assert core["chart"]["status"] == "not_applicable"
    assert core["model"]["used"] is True

    zinc = _seed_axis_policy("zinc", axes)
    assert zinc["split"]["status"] == "not_applicable"
    assert "official" in zinc["split"]["reason"]
    assert zinc["chart"]["status"] == "not_applicable"


def test_cyclecount_cache_identity_uses_data_seed_not_model_seed(tmp_path) -> None:
    data_root = tmp_path / "data"
    manifests = []
    for model_seed in (31, 37):
        output_root = tmp_path / f"run-{model_seed}"
        assert (
            main(
                [
                    "--suite",
                    "core",
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(output_root),
                    "--device",
                    "cpu",
                    "--tiny",
                    "--prepare-only",
                    "--seed",
                    "5",
                    "--data-seed",
                    "23",
                    "--split-seed",
                    "29",
                    "--chart-seed",
                    "30",
                    "--model-seed",
                    str(model_seed),
                ]
            )
            == 0
        )
        manifests.append(
            json.loads((output_root / "core" / "manifest.json").read_text(encoding="utf-8"))
        )

    first, second = manifests
    assert first["seed_axes"] == {"data": 23, "split": 29, "chart": 30, "model": 31}
    assert second["seed_axes"]["model"] == 37
    assert first["cache"] == second["cache"]
    assert first["dataset_metadata"]["seed"] == 23
    assert first["seed_axis_policy"]["split"]["used"] is False
    assert first["seed_axis_policy"]["chart"]["used"] is False


def test_brec_policy_uses_internal_protocol_seed_axis_only() -> None:
    axes = SeedAxes(data=1, split=2, chart=3, model=4)
    policy = _seed_axis_policy("brec", axes, brec_protocol="official", brec_seeds=(100, 200))
    assert policy["model"]["used"] is False
    assert policy["protocol"]["used"] is True
    assert policy["protocol"]["values"] == [100, 200]
````

# research/tree_augmentation/__init__.py

````python
"""Lossless spanning-tree chart augmentation for static Cycle PE.

This package intentionally depends only on shared graph/algebra utilities.  It
does not expose conductance, attention, potential, or flow-completion models.
"""

from .augmentation import (
    TreeChart,
    build_tree_chart,
    chart_probe_features,
    cycle_projector,
    cycle_projector_diagonal,
    ensure_full_cycle_budget,
    evaluate_probe,
    find_unseen_chart,
    lossless_transition_error,
    run_static_cycle_pe_probe,
    sample_tree_charts,
    train_probe,
    transition_cocycle_error,
    transport_coordinates,
)
from .paper_data import build_paper_chart, wilson_ust_indices
from .paper_model import VariableBetaCycleEncoder

__all__ = [
    "TreeChart",
    "VariableBetaCycleEncoder",
    "build_paper_chart",
    "build_tree_chart",
    "chart_probe_features",
    "cycle_projector",
    "cycle_projector_diagonal",
    "ensure_full_cycle_budget",
    "evaluate_probe",
    "find_unseen_chart",
    "lossless_transition_error",
    "run_static_cycle_pe_probe",
    "sample_tree_charts",
    "train_probe",
    "transition_cocycle_error",
    "transport_coordinates",
    "wilson_ust_indices",
]
````

# research/tree_augmentation/augmentation.py

````python
"""Tree-chart resampling and a small static Cycle-PE augmentation probe.

No conductance, GAT, node-potential, or flow-completion object is imported or
used in this module.  Every enabled chart keeps the full cycle rank ``beta``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from chartgat.algebra import chart_transition, fundamental_cycle_basis, incidence_matrix
from chartgat.graphs import spanning_tree_indices

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TreeChart:
    """A full fundamental-cycle coordinate chart induced by one spanning tree."""

    name: str
    tree_edge_indices: IntArray
    chord_edge_indices: IntArray
    basis: FloatArray

    @property
    def beta(self) -> int:
        """Return the cycle rank represented by this chart."""

        return int(self.basis.shape[1])

    @property
    def num_edges(self) -> int:
        """Return the physical edge dimension."""

        return int(self.basis.shape[0])


def ensure_full_cycle_budget(beta: int, k: int | None = None) -> int:
    """Accept only the full-``beta`` lossless mode.

    ``k < beta`` is a distinct lossy extension.  It is intentionally disabled
    so that truncation cannot silently enter a lossless augmentation result.
    """

    if beta < 0:
        raise ValueError("beta must be non-negative")
    if k is None:
        return beta
    if k < 0 or k > beta:
        raise ValueError("k must lie in [0, beta]")
    if k < beta:
        raise NotImplementedError(
            "k < beta is a disabled lossy extension, not lossless tree augmentation"
        )
    return k


def build_tree_chart(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    method: str,
    seed: int = 0,
    name: str | None = None,
) -> TreeChart:
    """Sample a BFS, DFS, or random tree and construct its full cycle chart."""

    tree = spanning_tree_indices(num_nodes, edges, mode=method, seed=seed)
    B = incidence_matrix(num_nodes, edges)
    basis, chords = fundamental_cycle_basis(B, tree, return_chords=True)
    ensure_full_cycle_budget(int(basis.shape[1]))
    chart_name = name if name is not None else f"{method}:{seed}"
    return TreeChart(chart_name, tree, chords, basis)


def sample_tree_charts(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    include_bfs: bool = True,
    include_dfs: bool = True,
    random_count: int = 0,
    random_seed_start: int = 0,
) -> list[TreeChart]:
    """Return unique full-``beta`` BFS/DFS/random charts.

    Different samplers can occasionally return the same edge set.  Such charts
    are deduplicated because repeated coordinates are not useful augmentation.
    """

    if random_count < 0:
        raise ValueError("random_count must be non-negative")
    requests: list[tuple[str, int]] = []
    if include_bfs:
        requests.append(("bfs", 0))
    if include_dfs:
        requests.append(("dfs", 0))
    requests.extend(("random", random_seed_start + offset) for offset in range(random_count))

    charts: list[TreeChart] = []
    seen: set[tuple[int, ...]] = set()
    for method, seed in requests:
        chart = build_tree_chart(num_nodes, edges, method=method, seed=seed)
        key = tuple(sorted(int(index) for index in chart.tree_edge_indices))
        if key not in seen:
            charts.append(chart)
            seen.add(key)
    if not charts:
        raise ValueError("at least one tree sampler must be enabled")
    return charts


def find_unseen_chart(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    seen_charts: Sequence[TreeChart],
    *,
    seed_start: int = 10_000,
    max_attempts: int = 10_000,
) -> TreeChart:
    """Sample a random chart whose spanning tree was not used for training."""

    seen = {tuple(sorted(int(index) for index in chart.tree_edge_indices)) for chart in seen_charts}
    for offset in range(max_attempts):
        chart = build_tree_chart(
            num_nodes,
            edges,
            method="random",
            seed=seed_start + offset,
            name=f"unseen-random:{seed_start + offset}",
        )
        key = tuple(sorted(int(index) for index in chart.tree_edge_indices))
        if key not in seen:
            return chart
    raise RuntimeError("failed to sample an unseen spanning-tree chart")


def transport_coordinates(
    source: TreeChart,
    target: TreeChart,
    coordinates: ArrayLike,
) -> FloatArray:
    """Transport full cycle coordinates without changing the physical cycle state."""

    if source.basis.shape != target.basis.shape:
        raise ValueError("source and target charts must describe the same edge/cycle dimensions")
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] != source.beta:
        raise ValueError("coordinates must have shape (beta,) or (beta, channels)")
    transition = chart_transition(source.basis, target.basis)
    return np.asarray(transition @ values, dtype=np.float64)


def lossless_transition_error(
    charts: Sequence[TreeChart],
    coordinates: ArrayLike,
) -> float:
    """Return the largest physical reconstruction error over all chart pairs."""

    if not charts:
        raise ValueError("charts must not be empty")
    source = charts[0]
    source_coordinates = np.asarray(coordinates, dtype=np.float64)
    physical = source.basis @ source_coordinates
    maximum = 0.0
    for target in charts:
        target_coordinates = transport_coordinates(source, target, source_coordinates)
        maximum = max(maximum, float(np.linalg.norm(target.basis @ target_coordinates - physical)))
    return maximum


def transition_cocycle_error(charts: Sequence[TreeChart]) -> float:
    """Return the largest spanning-tree chart-transition cocycle residual."""

    if not charts:
        raise ValueError("charts must not be empty")
    maximum = 0.0
    for source in charts:
        for middle in charts:
            source_to_middle = chart_transition(source.basis, middle.basis)
            for target in charts:
                middle_to_target = chart_transition(middle.basis, target.basis)
                source_to_target = chart_transition(source.basis, target.basis)
                residual = middle_to_target @ source_to_middle - source_to_target
                maximum = max(maximum, float(np.linalg.norm(residual)))
    return maximum


def cycle_projector(cycle_basis: ArrayLike) -> FloatArray:
    """Return the physical orthogonal projector onto the represented cycle space."""

    basis = np.asarray(cycle_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("cycle_basis must be two-dimensional")
    beta = basis.shape[1]
    if beta == 0:
        return np.zeros((basis.shape[0], basis.shape[0]), dtype=np.float64)
    if np.linalg.matrix_rank(basis) != beta:
        raise ValueError("cycle_basis must have full column rank")
    gram_inverse = np.linalg.inv(basis.T @ basis)
    projector = basis @ gram_inverse @ basis.T
    return np.asarray((projector + projector.T) / 2.0, dtype=np.float64)


def cycle_projector_diagonal(cycle_basis: ArrayLike) -> FloatArray:
    """Return chart-independent static edge cycle leverage scores."""

    return np.diag(cycle_projector(cycle_basis)).copy()


def chart_probe_features(cycle_basis: ArrayLike) -> FloatArray:
    """Build deliberately chart-dependent raw features for the diagnostic probe.

    Each edge receives its raw fundamental-cycle row, the squared row, and the
    flattened inverse chart Gram matrix.  All are static topology quantities;
    no node/flow state is involved.
    """

    basis = np.asarray(cycle_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("cycle_basis must be two-dimensional")
    m, beta = basis.shape
    if beta == 0:
        return np.ones((m, 1), dtype=np.float64)
    gram_inverse = np.linalg.inv(basis.T @ basis)
    global_features = np.broadcast_to(gram_inverse.reshape(1, -1), (m, beta * beta))
    return np.concatenate((basis, basis**2, global_features), axis=1)


class _StaticCycleProbe(nn.Module):
    """Small raw-coordinate MLP used only for the augmentation comparison."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _probe_arrays(charts: Iterable[TreeChart], target: FloatArray) -> tuple[Tensor, Tensor]:
    feature_blocks: list[FloatArray] = []
    target_blocks: list[FloatArray] = []
    for chart in charts:
        feature_blocks.append(chart_probe_features(chart.basis))
        target_blocks.append(target)
    if not feature_blocks:
        raise ValueError("at least one training chart is required")
    features = torch.as_tensor(np.concatenate(feature_blocks), dtype=torch.float64)
    targets = torch.as_tensor(np.concatenate(target_blocks), dtype=torch.float64)
    return features, targets


def train_probe(
    charts: Sequence[TreeChart],
    target: ArrayLike,
    *,
    hidden_dim: int = 48,
    epochs: int = 800,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-5,
    seed: int = 0,
) -> nn.Module:
    """Fit the raw-coordinate static Cycle-PE probe on one or many tree charts."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    values = np.asarray(target, dtype=np.float64)
    if values.shape != (charts[0].num_edges,):
        raise ValueError("target must contain one scalar per physical edge")
    torch.manual_seed(seed)
    features, targets = _probe_arrays(charts, values)
    model = _StaticCycleProbe(features.shape[1], hidden_dim).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(features) - targets) ** 2)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def evaluate_probe(model: nn.Module, chart: TreeChart, target: ArrayLike) -> float:
    """Return mean squared error for one chart of the same physical graph."""

    values = np.asarray(target, dtype=np.float64)
    if values.shape != (chart.num_edges,):
        raise ValueError("target must contain one scalar per physical edge")
    features = torch.as_tensor(chart_probe_features(chart.basis), dtype=torch.float64)
    targets = torch.as_tensor(values, dtype=torch.float64)
    return float(torch.mean((model(features) - targets) ** 2))


def run_static_cycle_pe_probe(
    training_charts: Sequence[TreeChart],
    unseen_chart: TreeChart,
    *,
    hidden_dim: int = 48,
    epochs: int = 800,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-5,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Compare fixed-tree training, multi-tree augmentation, and unseen-tree tests."""

    if not training_charts:
        raise ValueError("training_charts must not be empty")
    reference = training_charts[0]
    if any(chart.basis.shape != reference.basis.shape for chart in training_charts):
        raise ValueError("all charts must describe the same physical graph")
    if unseen_chart.basis.shape != reference.basis.shape:
        raise ValueError("unseen_chart must describe the same physical graph")

    target = cycle_projector_diagonal(reference.basis)
    if not np.allclose(cycle_projector_diagonal(unseen_chart.basis), target, atol=1e-10, rtol=0.0):
        raise ValueError("charts do not represent the same physical cycle space")

    settings = {
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    fixed_model = train_probe([reference], target, **settings)
    multi_model = train_probe(training_charts, target, **settings)
    fixed_train_mse = evaluate_probe(fixed_model, reference, target)
    multi_train_mse = float(
        np.mean([evaluate_probe(multi_model, chart, target) for chart in training_charts])
    )
    fixed_unseen_mse = evaluate_probe(fixed_model, unseen_chart, target)
    multi_unseen_mse = evaluate_probe(multi_model, unseen_chart, target)
    oracle_target = cycle_projector_diagonal(unseen_chart.basis)
    oracle_unseen_mse = float(np.mean((oracle_target - target) ** 2))

    return {
        "cycle_rank_beta": reference.beta,
        "num_training_charts": len(training_charts),
        "fixed_chart": reference.name,
        "unseen_chart": unseen_chart.name,
        "fixed_train_mse": fixed_train_mse,
        "multi_train_mse": multi_train_mse,
        "fixed_unseen_mse": fixed_unseen_mse,
        "multi_unseen_mse": multi_unseen_mse,
        "unseen_mse_ratio_multi_over_fixed": multi_unseen_mse
        / max(fixed_unseen_mse, np.finfo(np.float64).tiny),
        "projector_oracle_unseen_mse": oracle_unseen_mse,
    }
````

# research/tree_augmentation/cache_validation.py

````python
"""Read-only cache validators used by the repository-level dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paper_data import validate_prepared_cache

SUITES = {
    "cyclecount_ood_multichart": "core",
    "csl_chart_sanity": "csl",
    "zinc12k_multichart": "zinc",
}


def validate_dataset_cache(
    dataset_id: str,
    data_root: Path,
    *,
    data_seeds: tuple[int, ...],
    split_seeds: tuple[int, ...],
    tiny: bool,
) -> dict[str, Any]:
    """Validate every requested processed tree cache without writing."""

    try:
        suite = SUITES[dataset_id]
    except KeyError as error:
        raise ValueError(f"unsupported tree cache dataset {dataset_id!r}") from error
    seeds = split_seeds if suite == "csl" else data_seeds
    paths: list[str] = []
    for seed in seeds:
        prepared = validate_prepared_cache(suite, data_root, seed=seed, tiny=tiny)
        paths.extend((str(prepared.data_path), str(prepared.manifest_path)))
    return {
        "paths": sorted(set(paths)),
        "requested_axis": "split" if suite == "csl" else "data",
        "requested_seeds": list(seeds),
    }


__all__ = ["validate_dataset_cache"]
````

# research/tree_augmentation/config.yaml

````yaml
seed: 17

graph:
  num_nodes: 12
  extra_edges: 7
  seed: 29

augmentation:
  include_bfs: true
  include_dfs: true
  random_count: 14
  random_seed_start: 100
  unseen_seed_start: 10000
  # null means k = beta. Any integer k < beta is deliberately rejected.
  k: null
  lossy_extension_enabled: false

probe:
  hidden_dim: 48
  epochs: 800
  learning_rate: 0.01
  weight_decay: 0.00001

output: results/summary.json

# Independent downstream paper protocol. The legacy projector smoke above
# remains unchanged and is never reported as the paper headline.
paper:
  learning_rate: 0.002
  weight_decay: 0.00001
  amp: true
  pin_memory: true
  non_blocking: true
  full:
    hidden_dim: 64
    optimizer_updates: 800
    batch_size: 16
    train_charts_per_graph: 8
    eval_charts_per_graph: 8
  tiny:
    hidden_dim: 16
    optimizer_updates: 24
    batch_size: 4
    train_charts_per_graph: 3
    eval_charts_per_graph: 2
````

# research/tree_augmentation/datasets.yaml

````yaml
registry_version: 2
track: tree_augmentation
paper_suite_complete: true
claim: Spanning-tree chart augmentation improves robustness without changing downstream labels or graph splits.

datasets:
  - id: single_graph_chart_smoke
    name: Current single-graph chart transition sanity check
    tier: smoke
    status: implemented
    data_policy: generated
    source_url: generated://research.tree_augmentation.run
    task: Certify lossless chart transitions and compare fixed versus multiple charts on one graph.
    split: One 12-node graph, 16 training trees, one unseen tree.
    metrics: [transition_error, cocycle_error, unseen_chart_mse]
    claim: Algebra and augmentation-pipeline certification only.
    adapter: research.tree_augmentation.augmentation
    leakage_guard: Target is projector-derived and cannot support a downstream novelty claim.

  - id: cyclecount_ood_multichart
    name: CycleCount-OOD multi-chart protocol
    tier: paper_core
    status: implemented
    data_policy: generated
    cache_glob: cyclecount_ood_v2/*.manifest.json
    source_url: generated://research.tree_augmentation.paper/core-v2
    task: Predict independent cycle-count labels while varying charts per graph.
    split: Graph split first, then fresh BFS seen-family and held-out Wilson sampler-family charts within every ID/OOD quadrant.
    metrics: [task_metric_mean, worst_chart, chart_std, prediction_flip_rate]
    claim: Separates chart robustness from topology OOD.
    adapter: python -m research.tree_augmentation.paper --suite core
    validator: research.tree_augmentation.cache_validation.validate_dataset_cache
    leakage_guard: Different charts of one graph must never cross graph train/test splits.

  - id: csl_chart_sanity
    name: CSL fixed-beta chart sanity benchmark
    tier: paper_core
    status: implemented
    data_policy: download
    cache_glob: csl_pyg_v2/*.manifest.json
    source_url: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GNNBenchmarkDataset.html
    task: Ten-class circular-skip-link graph classification under fresh BFS and held-out Wilson sampler-family charts.
    split: Deterministic label-stratified five-fold 3/1/1 protocol cached by graph index.
    metrics: [accuracy, worst_chart_accuracy, chart_std]
    claim: Controlled fixed-beta chart robustness, not broad real-world generalization.
    adapter: python -m research.tree_augmentation.paper --suite csl (optional PyG)
    validator: research.tree_augmentation.cache_validation.validate_dataset_cache
    leakage_guard: Fold membership and source graph indices are frozen before chart sampling.

  - id: zinc12k_multichart
    name: ZINC-12K multi-chart downstream evaluation
    tier: paper_core
    status: implemented
    data_policy: download
    cache_glob: zinc_pyg_v2/*.manifest.json
    source_url: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html
    task: Molecular regression combining chart/topology coordinates with preserved categorical atom and bond features under fixed-BFS versus mixed-chart training.
    split: Official graph split plus config-controlled fresh BFS and held-out Wilson chart samples per test graph.
    metrics: [mae, worst_chart_mae, prediction_spread, preprocessing_time, peak_memory]
    claim: Real downstream utility of chart augmentation on variable-beta molecular graphs with atom and bond chemistry retained; this is not a topology-only ZINC model.
    adapter: python -m research.tree_augmentation.paper --suite zinc (optional PyG)
    validator: research.tree_augmentation.cache_validation.validate_dataset_cache
    leakage_guard: Match optimizer updates across K; preserve official ZINC splits and targets; atom/bond tensors stay fixed across charts; projector targets are excluded from headline labels.

  - id: brec_chart_stress
    name: BREC relabeling and chart stress test
    tier: optional
    status: planned
    data_policy: manual
    source_url: https://github.com/GraphPKU/BREC
    task: Measure whether chart selection changes pair distinguishability and reliability.
    split: Official RPC groups with charts nested inside each graph instance.
    metrics: [successful_pairs, chart_failure_rate, reliability_failures]
    claim: Optional interaction between expressivity and chart robustness.
    adapter: planned BREC chart wrapper
    leakage_guard: Fix cycle-sensitive subset before observing model results.

chart_protocol:
  training_conditions: [single_root0_bfs, mixed_random_root_bfs_dfs]
  evaluation_conditions: [fresh_chart_seen_bfs_family, fresh_chart_unseen_wilson_family, id_graphs, topology_ood_graphs]
  unseen_family_excluded_from_training: true
  exact_tree_overlap_between_families_allowed: true
  wilson_draws_conditioned_on_bfs_outputs: false
  current_random_sampler_name: random_priority_kruskal
  uniform_sampler: Wilson loop-erased random walk implemented in paper_data.py
  variable_beta_encoder: sign-even orientation-gauge-safe masked coordinate-set and edge-set pooling implemented in paper_model.py
  permutation_scope: Exact for orientation/order/relabeling of the same physical tree; label-dependent selection of another BFS/DFS tree remains chart shift.
````

# research/tree_augmentation/paper.py

````python
"""Run the independent multi-chart paper protocol.

Examples
--------
python -m research.tree_augmentation.paper --suite core --tiny --device cpu
python -m research.tree_augmentation.paper --suite core --device cuda --amp
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from chartgat.seeds import SeedAxes, resolve_seed_axes

from .paper_data import (
    OptionalDatasetError,
    PreparedDataset,
    prepare_cyclecount_dataset,
    prepare_optional_pyg_dataset,
)
from .paper_model import build_chart_views, run_fixed_vs_multichart

SUITES = ("core", "csl", "zinc")


def _sampler_protocol() -> dict[str, Any]:
    """Return the declared sampler exposure for the fixed/multi comparison."""

    return {
        "train_fixed": ["bfs_root_0"],
        "train_multi": ["bfs_random_root", "dfs_random_root"],
        "fresh_chart_seen_family": ["bfs_random_root"],
        "fresh_chart_unseen_family": ["wilson_ust"],
        "unseen_family_is_disjoint_from_all_training_families": True,
        "exact_tree_overlap_between_families_allowed": True,
        "wilson_draws_conditioned_on_bfs_outputs": False,
    }


def _protocol_name(suite: str) -> str:
    return (
        "cyclecount_graph_x_fresh_chart_family_2x2_v2"
        if suite == "core"
        else "public_pyg_fresh_chart_family_benchmark_v2"
    )


def _dataset_seed_policy(suite: str) -> dict[str, Any]:
    if suite == "core":
        return {
            "cache_identity_axis": "data",
            "record_generation_axis": "data",
            "split_assignment_axis": "data",
            "split_seed_used": False,
        }
    if suite == "csl":
        return {
            "cache_identity_axis": "split",
            "record_source": "fixed_public_dataset",
            "split_assignment_axis": "split",
            "data_seed_used": False,
        }
    return {
        "cache_identity_axis": "data",
        "record_source": "fixed_public_dataset",
        "split_assignment_axis": "official",
        "data_seed_changes_records": False,
        "split_seed_changes_records": False,
    }


def resolve_device(requested: str) -> torch.device:
    """Resolve CPU/CUDA requests and fail before a long experiment starts."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {normalized!r} was requested, but this PyTorch build cannot use CUDA"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        device = torch.device("cuda", index)
    return device


def _seed_runtime(seed: int, device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"output directory is not empty; refusing to overwrite: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_settings(*, tiny: bool) -> tuple[dict[str, Any], Path]:
    config_path = Path(__file__).with_name("config.yaml").resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("paper"), dict):
        raise ValueError("config.yaml must contain a paper mapping")
    paper = dict(config["paper"])
    profile_name = "tiny" if tiny else "full"
    profile = paper.get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"paper.{profile_name} must be a mapping")
    merged = {key: value for key, value in paper.items() if key not in {"tiny", "full"}}
    merged.update(profile)
    return merged, config_path


def _prepare_dataset(
    suite: str,
    data_root: Path,
    *,
    seed_axes: SeedAxes,
    tiny: bool,
    allow_download: bool,
) -> PreparedDataset:
    if suite == "core":
        return prepare_cyclecount_dataset(data_root, seed=seed_axes.data, tiny=tiny)
    cache_seed = seed_axes.split if suite == "csl" else seed_axes.data
    return prepare_optional_pyg_dataset(
        suite,
        data_root,
        seed=cache_seed,
        tiny=tiny,
        allow_download=allow_download,
    )


def _runtime_metadata(
    *,
    device: torch.device,
    amp_requested: bool,
    pin_memory: bool,
    non_blocking: bool,
    batch_size: int,
    workers: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    cuda = device.type == "cuda"
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "amp_requested": amp_requested,
        "amp_effective": bool(amp_requested and cuda),
        "pin_memory": bool(pin_memory and cuda),
        "non_blocking": bool(non_blocking and cuda),
        "batch_size": batch_size,
        "workers": workers,
        "elapsed_seconds": elapsed_seconds,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "peak_gpu_allocated_bytes": 0,
        "peak_gpu_reserved_bytes": 0,
    }
    if cuda:
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return metadata


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _save_models(
    output_dir: Path,
    models: dict[str, Any],
    *,
    settings: dict[str, Any],
    task_type: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, fitted in models.items():
        path = output_dir / f"{name}_model.pt"
        torch.save(
            {
                "state_dict": _cpu_state_dict(fitted.model),
                "target_mean": torch.as_tensor(fitted.target_mean),
                "target_scale": torch.as_tensor(fitted.target_scale),
                "settings": settings,
                "task_type": task_type,
            },
            path,
        )
        paths[name] = str(path)
    return paths


def _split(records: tuple[Any, ...], name: str) -> list[Any]:
    return [record for record in records if record.split == name]


def _chart_keys_by_graph(views: list[Any]) -> dict[str, set[tuple[int, ...]]]:
    grouped: dict[str, set[tuple[int, ...]]] = {}
    for view in views:
        grouped.setdefault(view.graph_id, set()).add(view.tree_key)
    return grouped


def _chart_overlap_stats(left: list[Any], right: list[Any]) -> dict[str, int]:
    left_keys = _chart_keys_by_graph(left)
    right_keys = _chart_keys_by_graph(right)
    intersections = [
        left_keys[graph_id] & right_keys.get(graph_id, set()) for graph_id in left_keys
    ]
    return {
        "graphs_with_exact_tree_overlap": sum(bool(overlap) for overlap in intersections),
        "unique_graph_tree_overlaps": sum(len(overlap) for overlap in intersections),
    }


def _fresh_axis_overlap_stats(evaluation: dict[str, list[Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    marker = "_fresh_chart_seen_family"
    for seen_name, seen_views in evaluation.items():
        if not seen_name.endswith(marker):
            continue
        unseen_name = seen_name.replace(marker, "_fresh_chart_unseen_family")
        result[seen_name.removesuffix(marker)] = _chart_overlap_stats(
            seen_views, evaluation[unseen_name]
        )
    return result


def _protocol_views(
    dataset: PreparedDataset,
    *,
    settings: dict[str, Any],
    chart_seed: int,
) -> tuple[list[Any], list[Any], dict[str, list[Any]]]:
    train_records = _split(dataset.records, "train")
    if not train_records:
        raise ValueError(f"suite {dataset.suite} contains no training graphs")
    train_charts = int(settings["train_charts_per_graph"])
    eval_charts = int(settings["eval_charts_per_graph"])
    fixed_train = build_chart_views(
        train_records,
        chart_status="train_fixed_bfs_family",
        count=1,
        methods=("bfs",),
        roots=(0,),
        seed=chart_seed + 1_000,
    )
    multi_train = build_chart_views(
        train_records,
        chart_status="train_multi_bfs_dfs_families",
        count=train_charts,
        methods=("bfs", "dfs"),
        seed=chart_seed + 2_000,
    )
    if dataset.suite == "core":
        id_records = _split(dataset.records, "id_test")
        ood_records = _split(dataset.records, "ood_test")
        if not id_records or not ood_records:
            raise ValueError("core suite requires non-empty id_test and ood_test graph splits")
        id_seen = build_chart_views(
            id_records,
            chart_status="fresh_chart_seen_family",
            count=eval_charts,
            methods=("bfs",),
            seed=chart_seed + 10_000,
        )
        id_unseen = build_chart_views(
            id_records,
            chart_status="fresh_chart_unseen_family",
            count=eval_charts,
            methods=("wilson_ust",),
            seed=chart_seed + 20_000,
        )
        ood_seen = build_chart_views(
            ood_records,
            chart_status="fresh_chart_seen_family",
            count=eval_charts,
            methods=("bfs",),
            seed=chart_seed + 30_000,
        )
        ood_unseen = build_chart_views(
            ood_records,
            chart_status="fresh_chart_unseen_family",
            count=eval_charts,
            methods=("wilson_ust",),
            seed=chart_seed + 40_000,
        )
        evaluation = {
            "id_graph_fresh_chart_seen_family": id_seen,
            "id_graph_fresh_chart_unseen_family": id_unseen,
            "ood_graph_fresh_chart_seen_family": ood_seen,
            "ood_graph_fresh_chart_unseen_family": ood_unseen,
        }
    else:
        test_records = _split(dataset.records, "test")
        if not test_records:
            raise ValueError(f"suite {dataset.suite} contains no test graphs")
        test_seen = build_chart_views(
            test_records,
            chart_status="fresh_chart_seen_family",
            count=eval_charts,
            methods=("bfs",),
            seed=chart_seed + 10_000,
        )
        test_unseen = build_chart_views(
            test_records,
            chart_status="fresh_chart_unseen_family",
            count=eval_charts,
            methods=("wilson_ust",),
            seed=chart_seed + 20_000,
        )
        evaluation = {
            "test_graph_fresh_chart_seen_family": test_seen,
            "test_graph_fresh_chart_unseen_family": test_unseen,
        }
    return fixed_train, multi_train, evaluation


def _output_dim(dataset: PreparedDataset) -> int:
    if dataset.task_type == "classification":
        labels = [int(record.target[0]) for record in dataset.records]
        return max(labels) + 1
    return len(dataset.records[0].target)


def _headline_comparison(metrics: dict[str, Any], *, suite: str, tiny: bool) -> dict[str, Any]:
    eligible = not tiny
    if tiny:
        eligibility_reason = "tiny fixture; pipeline validation only"
    elif suite == "core":
        eligibility_reason = "full independent CycleCount-style core protocol"
    elif suite == "zinc":
        eligibility_reason = "official ZINC-12K split with atom/bond chemistry and held-out charts"
    else:
        eligibility_reason = "full CSL controlled chart-robustness protocol"
    comparison: dict[str, Any] = {
        "paper_headline_eligible": eligible,
        "paper_headline_eligibility_reason": eligibility_reason,
        "projector_target_used": False,
        "fixed_and_multi_optimizer_updates_matched": True,
    }
    if suite == "core":
        improvements = {}
        for quadrant, fixed in metrics["fixed_bfs"]["quadrants"].items():
            multi = metrics["multi_chart"]["quadrants"][quadrant]
            improvements[quadrant] = {
                "mae_improvement_fixed_minus_multi": fixed["mae"] - multi["mae"],
                "worst_chart_mae_improvement_fixed_minus_multi": (
                    fixed["worst_chart_mae"] - multi["worst_chart_mae"]
                ),
                "chart_std_improvement_fixed_minus_multi": (
                    fixed["chart_prediction_std"] - multi["chart_prediction_std"]
                ),
            }
        comparison["quadrant_improvements"] = improvements
    return comparison


def _view_stats(views: list[Any]) -> dict[str, Any]:
    sampler_counts: dict[str, int] = {}
    chart_status_counts: dict[str, int] = {}
    for view in views:
        method = str(view.chart_name).split(":", 1)[0]
        sampler_counts[method] = sampler_counts.get(method, 0) + 1
        status = str(view.chart_status)
        chart_status_counts[status] = chart_status_counts.get(status, 0) + 1
    return {
        "views": len(views),
        "graphs": len({view.graph_id for view in views}),
        "unique_graph_tree_pairs": len({(view.graph_id, view.tree_key) for view in views}),
        "sampler_counts": sampler_counts,
        "chart_status_counts": chart_status_counts,
    }


def run_suite(
    suite: str,
    *,
    data_root: Path,
    output_dir: Path,
    requested_device: str,
    seed: int,
    data_seed: int | None = None,
    split_seed: int | None = None,
    chart_seed: int | None = None,
    model_seed: int | None = None,
    tiny: bool,
    prepare_only: bool,
    amp_override: bool | None,
    batch_size_override: int | None,
    pin_memory_override: bool | None,
    non_blocking_override: bool | None,
    workers: int = 0,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Prepare and optionally train exactly one independent suite."""

    seed_axes = resolve_seed_axes(
        seed,
        data_seed=data_seed,
        split_seed=split_seed,
        chart_seed=chart_seed,
        model_seed=model_seed,
    )
    settings, config_path = _load_settings(tiny=tiny)
    output = _prepare_output_dir(output_dir)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "preparing",
        "suite": suite,
        "protocol": _protocol_name(suite),
        "seed_axes": seed_axes.to_manifest(),
        "dataset_seed_policy": _dataset_seed_policy(suite),
        "tiny": tiny,
        "prepare_only": prepare_only,
        "allow_download": allow_download,
        "workers": workers,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_files": {
            path.name: _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("paper_data.py").resolve(),
                Path(__file__).with_name("paper_model.py").resolve(),
                Path(__file__).with_name("augmentation.py").resolve(),
                Path(__file__).with_name("datasets.yaml").resolve(),
            )
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "networkx", "torch", "PyYAML")
        },
        "output_dir": str(output),
        "sampler_protocol": _sampler_protocol(),
        "orientation_gauge_policy": (
            "sign-even fundamental-cycle coordinates; exact only for the same physical tree"
        ),
    }
    _write_json(manifest_path, manifest)
    try:
        if workers < 0:
            raise ValueError("workers must be non-negative")
        dataset = _prepare_dataset(
            suite,
            data_root,
            seed_axes=seed_axes,
            tiny=tiny,
            allow_download=allow_download,
        )
        manifest["dataset"] = {
            "data_path": str(dataset.data_path),
            "manifest_path": str(dataset.manifest_path),
            "manifest_sha256": _sha256(dataset.manifest_path),
            "data_sha256": dataset.data_sha256,
            "num_graphs": len(dataset.records),
            "task_type": dataset.task_type,
            "target_names": list(dataset.target_names),
        }
        if prepare_only:
            manifest["status"] = "prepared"
            manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
            _write_json(manifest_path, manifest)
            return manifest

        device = resolve_device(requested_device)
        _seed_runtime(seed_axes.model, device)
        amp = bool(settings.get("amp", True)) if amp_override is None else amp_override
        batch_size = (
            int(settings["batch_size"]) if batch_size_override is None else batch_size_override
        )
        pin_memory = (
            bool(settings.get("pin_memory", True))
            if pin_memory_override is None
            else pin_memory_override
        )
        non_blocking = (
            bool(settings.get("non_blocking", True))
            if non_blocking_override is None
            else non_blocking_override
        )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fixed_train, multi_train, evaluation = _protocol_views(
            dataset, settings=settings, chart_seed=seed_axes.chart
        )
        started = time.perf_counter()
        metrics, models = run_fixed_vs_multichart(
            fixed_train_views=fixed_train,
            multi_train_views=multi_train,
            evaluation_views=evaluation,
            task_type=dataset.task_type,
            output_dim=_output_dim(dataset),
            hidden_dim=int(settings["hidden_dim"]),
            updates=int(settings["optimizer_updates"]),
            batch_size=batch_size,
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            device=device,
            seed=seed_axes.model,
            amp=amp,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            workers=workers,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        runtime = _runtime_metadata(
            device=device,
            amp_requested=amp,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            batch_size=batch_size,
            workers=workers,
            elapsed_seconds=elapsed,
        )
        model_paths = _save_models(
            output,
            models,
            settings=settings
            | {
                "batch_size": batch_size,
                "amp": amp,
                "pin_memory": pin_memory,
                "non_blocking": non_blocking,
                "workers": workers,
                "seed_axes": seed_axes.to_manifest(),
            },
            task_type=dataset.task_type,
        )
        split_counts: dict[str, int] = {}
        for record in dataset.records:
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
        summary = {
            "track": "tree_augmentation_only",
            "suite": suite,
            "seed_axes": seed_axes.to_manifest(),
            "dataset_seed_policy": _dataset_seed_policy(suite),
            "protocol": _protocol_name(suite),
            "downstream_target": list(dataset.target_names),
            "target_is_independent_of_chart": True,
            "legacy_projector_smoke_preserved_separately": True,
            "samplers": {
                "uniform": "wilson_ust",
                "traversal": ["bfs_random_root", "dfs_random_root"],
                "legacy_nonuniform_baseline": "random_priority_kruskal",
            },
            "sampler_protocol": _sampler_protocol(),
            "orientation_gauge_policy": {
                "coordinate_features": ["abs_f", "square_f", "normalized_cycle_support"],
                "same_physical_tree_invariant": True,
                "different_tree_chart_invariant": False,
                "label_dependent_tree_selection_is_chart_shift": True,
            },
            "graph_split_before_chart_sampling": True,
            "fresh_axis_exact_tree_overlap": _fresh_axis_overlap_stats(evaluation),
            "split_counts": split_counts,
            "view_counts": {
                "fixed_train": _view_stats(fixed_train),
                "multi_train": _view_stats(multi_train),
                "evaluation": {name: _view_stats(views) for name, views in evaluation.items()},
            },
            "settings": settings
            | {
                "batch_size": batch_size,
                "amp": amp,
                "pin_memory": pin_memory,
                "non_blocking": non_blocking,
                "workers": workers,
                "seed_axes": seed_axes.to_manifest(),
            },
            "runtime": runtime,
            "models": metrics,
            "comparison": _headline_comparison(metrics, suite=suite, tiny=tiny),
            "checkpoints": model_paths,
        }
        summary_path = output / "summary.json"
        _write_json(summary_path, summary)
        artifacts = [summary_path, *(Path(path) for path in model_paths.values())]
        manifest.update(
            {
                "status": "passed",
                "device": str(device),
                "runtime": runtime,
                "artifacts": {
                    path.name: {"path": str(path), "sha256": _sha256(path)} for path in artifacts
                },
                "finished_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(manifest_path, manifest)
        return summary
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(manifest_path, manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="core")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).with_name("data"),
        help="deterministic processed-cache root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "paper",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="dataset-generation/cache axis; defaults to --seed",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="split-assignment axis; defaults to the resolved data seed",
    )
    parser.add_argument(
        "--chart-seed",
        type=int,
        default=None,
        help="spanning-tree chart sampling axis; defaults to the resolved data seed",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="model initialization/minibatch axis; defaults to --seed",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="small profile (core is offline; CSL/ZINC still require a cache or download)",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker processes (0 loads in the main process)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow optional CSL/ZINC adapters to access public download endpoints",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--non-blocking", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _run_from_args(args: argparse.Namespace, suite: str, output_dir: Path) -> dict[str, Any]:
    return run_suite(
        suite,
        data_root=args.data_root,
        output_dir=output_dir,
        requested_device=args.device,
        seed=args.seed,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        chart_seed=args.chart_seed,
        model_seed=args.model_seed,
        tiny=args.tiny,
        prepare_only=args.prepare_only,
        amp_override=args.amp,
        batch_size_override=args.batch_size,
        pin_memory_override=args.pin_memory,
        non_blocking_override=args.non_blocking,
        workers=args.workers,
        allow_download=args.allow_download,
    )


def _run_all(args: argparse.Namespace, output_root: Path) -> int:
    """Run every independent suite and leave an aggregate manifest on partial failure."""

    seed_axes = resolve_seed_axes(
        args.seed,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        chart_seed=args.chart_seed,
        model_seed=args.model_seed,
    )
    _prepare_output_dir(output_root)
    aggregate_path = output_root / "manifest.json"
    aggregate: dict[str, Any] = {
        "status": "preparing",
        "suite": "all",
        "seed_axes": seed_axes.to_manifest(),
        "tiny": args.tiny,
        "prepare_only": args.prepare_only,
        "allow_download": args.allow_download,
        "workers": args.workers,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "suites": {},
    }
    _write_json(aggregate_path, aggregate)
    results: dict[str, Any] = {}
    optional_failure = False
    protocol_failure = False
    for suite in SUITES:
        suite_output = output_root / suite
        try:
            result = _run_from_args(args, suite, suite_output)
            child_status = "prepared" if args.prepare_only else "passed"
            results[suite] = result
            aggregate["suites"][suite] = {
                "status": child_status,
                "manifest_path": str(suite_output / "manifest.json"),
                "manifest_sha256": _sha256(suite_output / "manifest.json"),
            }
        except OptionalDatasetError as error:
            optional_failure = True
            failure = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "manifest_path": str(suite_output / "manifest.json"),
            }
            results[suite] = failure
            if (suite_output / "manifest.json").is_file():
                failure["manifest_sha256"] = _sha256(suite_output / "manifest.json")
            aggregate["suites"][suite] = failure
        except Exception as error:  # keep independent suites observable on partial failure
            protocol_failure = True
            failure = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "manifest_path": str(suite_output / "manifest.json"),
            }
            results[suite] = failure
            if (suite_output / "manifest.json").is_file():
                failure["manifest_sha256"] = _sha256(suite_output / "manifest.json")
            aggregate["suites"][suite] = failure
    if optional_failure or protocol_failure:
        aggregate["status"] = "failed"
    else:
        aggregate["status"] = "prepared" if args.prepare_only else "passed"
    aggregate["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(aggregate_path, aggregate)
    print(json.dumps(results, indent=2, sort_keys=True))
    if protocol_failure:
        print(f"one or more paper suites failed; see {aggregate_path}", file=sys.stderr)
        return 1
    if optional_failure:
        print(
            f"one or more optional datasets are unavailable; see {aggregate_path}",
            file=sys.stderr,
        )
        return 2
    return 0


def main() -> int:
    args = _parser().parse_args()
    suites = SUITES if args.suite == "all" else (args.suite,)
    output_root = args.output_dir.expanduser().resolve()
    try:
        if len(suites) > 1:
            return _run_all(args, output_root)
        results = {suites[0]: _run_from_args(args, suites[0], output_root)}
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    except OptionalDatasetError as error:
        print(f"optional dataset unavailable: {error}", file=sys.stderr)
        return 2
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"paper protocol failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
````

# research/tree_augmentation/paper_data.py

````python
"""Datasets and spanning-tree charts for the independent paper protocol.

The existing :mod:`research.tree_augmentation.run` smoke experiment is kept
separate.  This module adds graph-level downstream labels, a true uniform
spanning-tree sampler, deterministic caches, and optional PyG dataset adapters.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix, validate_spanning_tree
from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_write_bytes,
)
from chartgat.graphs import make_connected_graph, spanning_tree_indices

from .augmentation import TreeChart

IntArray = NDArray[np.int64]
DATASET_VERSION = 2
TARGET_CYCLE_LENGTHS = (3, 4, 5, 6)
ZINC_NUM_ATOM_TYPES = 28
ZINC_NUM_BOND_TYPES = 4


class OptionalDatasetError(RuntimeError):
    """A requested optional dataset cannot be imported or downloaded."""


@dataclass(frozen=True)
class GraphRecord:
    """One physical graph and a downstream label that is independent of its chart."""

    graph_id: str
    family: str
    split: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    target: tuple[float, ...]
    task_type: str = "regression"
    x: tuple[int, ...] | None = None
    edge_attr: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.x is not None:
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in self.x
            ):
                raise ValueError("categorical node x values must be non-negative integers")
            normalized_x = tuple(int(value) for value in self.x)
            if len(normalized_x) != self.num_nodes:
                raise ValueError("categorical node x must have one value per node")
            if any(value < 0 for value in normalized_x):
                raise ValueError("categorical node x values must be non-negative integers")
            object.__setattr__(self, "x", normalized_x)
        if self.edge_attr is not None:
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in self.edge_attr
            ):
                raise ValueError("categorical edge_attr values must be non-negative integers")
            normalized_edge_attr = tuple(int(value) for value in self.edge_attr)
            if len(normalized_edge_attr) != len(self.edges):
                raise ValueError(
                    "categorical edge_attr must align one-to-one with undirected edges"
                )
            if any(value < 0 for value in normalized_edge_attr):
                raise ValueError("categorical edge_attr values must be non-negative integers")
            object.__setattr__(self, "edge_attr", normalized_edge_attr)

    @property
    def beta(self) -> int:
        return len(self.edges) - self.num_nodes + 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "family": self.family,
            "split": self.split,
            "num_nodes": self.num_nodes,
            "edges": [list(edge) for edge in self.edges],
            "target": list(self.target),
            "task_type": self.task_type,
            "x": None if self.x is None else list(self.x),
            "edge_attr": None if self.edge_attr is None else list(self.edge_attr),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> GraphRecord:
        num_nodes = int(payload["num_nodes"])
        raw_edges = tuple((int(edge[0]), int(edge[1])) for edge in payload["edges"])
        edges = _canonical_edges(num_nodes, raw_edges)
        if len(edges) != len(raw_edges):
            raise ValueError(
                "cached undirected edges contain a self-loop, duplicate, or parallel edge"
            )
        raw_edge_attr = payload.get("edge_attr")
        if raw_edge_attr is None:
            edge_attr = None
        else:
            values = tuple(raw_edge_attr)
            if len(values) != len(raw_edges):
                raise ValueError(
                    "cached categorical edge_attr does not align with undirected edges"
                )
            by_edge = {
                (min(u, v), max(u, v)): value
                for (u, v), value in zip(raw_edges, values, strict=True)
            }
            edge_attr = tuple(by_edge[edge] for edge in edges)
        return cls(
            graph_id=str(payload["graph_id"]),
            family=str(payload["family"]),
            split=str(payload["split"]),
            num_nodes=num_nodes,
            edges=edges,
            target=tuple(float(value) for value in payload["target"]),
            task_type=str(payload.get("task_type", "regression")),
            x=(None if payload.get("x") is None else tuple(payload["x"])),
            edge_attr=edge_attr,
        )


@dataclass(frozen=True)
class PreparedDataset:
    """Validated records plus cache provenance."""

    suite: str
    records: tuple[GraphRecord, ...]
    data_path: Path
    manifest_path: Path
    data_sha256: str
    target_names: tuple[str, ...]
    task_type: str


def _canonical_edges(
    num_nodes: int, edges: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least two")
    canonical: set[tuple[int, int]] = set()
    for raw_u, raw_v in edges:
        u, v = int(raw_u), int(raw_v)
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError("edge endpoint lies outside [0, num_nodes)")
        if u == v:
            continue
        canonical.add((min(u, v), max(u, v)))
    result = tuple(sorted(canonical))
    if not result:
        raise ValueError("graph must contain at least one edge")
    _adjacency(num_nodes, result, require_connected=True)
    return result


def _adjacency(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    require_connected: bool,
) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    seen_edges: set[tuple[int, int]] = set()
    for edge_index, (raw_u, raw_v) in enumerate(edges):
        u, v = int(raw_u), int(raw_v)
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError("edge endpoint lies outside [0, num_nodes)")
        if u == v:
            raise ValueError("self-loops are not supported by the chart protocol")
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            raise ValueError("parallel or duplicate undirected edges are not supported")
        seen_edges.add(key)
        adjacency[u].append((v, edge_index))
        adjacency[v].append((u, edge_index))
    for neighbors in adjacency:
        neighbors.sort()
    if require_connected:
        reached = {0}
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adjacency[node]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if len(reached) != num_nodes:
            raise ValueError("graph is disconnected")
    return adjacency


def traversal_tree_indices(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    method: str,
    root: int,
) -> IntArray:
    """Return a deterministic BFS/DFS tree from an explicit root."""

    if method not in {"bfs", "dfs"}:
        raise ValueError("method must be bfs or dfs")
    if not 0 <= root < num_nodes:
        raise ValueError("root lies outside [0, num_nodes)")
    adjacency = _adjacency(num_nodes, edges, require_connected=True)
    selected: list[int] = []
    seen = {root}
    frontier: deque[int] | list[int]
    if method == "bfs":
        frontier = deque([root])
        while frontier:
            node = frontier.popleft()
            for neighbor, edge_index in adjacency[node]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                selected.append(edge_index)
                frontier.append(neighbor)
    else:
        stack = [root]
        next_neighbor = [0]
        while stack:
            node = stack[-1]
            position = next_neighbor[-1]
            if position == len(adjacency[node]):
                stack.pop()
                next_neighbor.pop()
                continue
            next_neighbor[-1] += 1
            neighbor, edge_index = adjacency[node][position]
            if neighbor in seen:
                continue
            seen.add(neighbor)
            selected.append(edge_index)
            stack.append(neighbor)
            next_neighbor.append(0)
    tree = np.asarray(sorted(selected), dtype=np.int64)
    validate_spanning_tree(incidence_matrix(num_nodes, edges), tree)
    return tree


def wilson_ust_indices(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    seed: int,
    root: int | None = None,
) -> IntArray:
    """Sample an unweighted uniform spanning tree with Wilson's algorithm.

    Loop-erased random walks generate every spanning tree with equal
    probability on a finite connected unweighted graph.  ``root`` changes the
    construction order but not the UST distribution.
    """

    adjacency = _adjacency(num_nodes, edges, require_connected=True)
    rng = np.random.default_rng(seed)
    resolved_root = int(rng.integers(num_nodes)) if root is None else int(root)
    if not 0 <= resolved_root < num_nodes:
        raise ValueError("root lies outside [0, num_nodes)")

    in_tree = np.zeros(num_nodes, dtype=np.bool_)
    in_tree[resolved_root] = True
    selected: list[int] = []
    starts = [int(node) for node in rng.permutation(num_nodes) if node != resolved_root]
    for start in starts:
        if in_tree[start]:
            continue
        path_nodes = [start]
        path_edges: list[int] = []
        positions = {start: 0}
        while not in_tree[path_nodes[-1]]:
            node = path_nodes[-1]
            choices = adjacency[node]
            neighbor, edge_index = choices[int(rng.integers(len(choices)))]
            if neighbor in positions:
                keep = positions[neighbor]
                for removed in path_nodes[keep + 1 :]:
                    positions.pop(removed)
                path_nodes = path_nodes[: keep + 1]
                path_edges = path_edges[:keep]
                continue
            path_edges.append(edge_index)
            path_nodes.append(neighbor)
            positions[neighbor] = len(path_nodes) - 1
        selected.extend(path_edges)
        in_tree[np.asarray(path_nodes, dtype=np.int64)] = True

    tree = np.asarray(sorted(selected), dtype=np.int64)
    validate_spanning_tree(incidence_matrix(num_nodes, edges), tree)
    return tree


def build_paper_chart(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    method: str,
    seed: int,
    root: int | None = None,
    name: str | None = None,
) -> TreeChart:
    """Build a full-beta chart from an explicit paper-protocol sampler."""

    normalized = method.strip().lower()
    rng = np.random.default_rng(seed ^ 0x5EED5EED)
    resolved_root = int(rng.integers(num_nodes)) if root is None else int(root)
    if normalized in {"bfs", "dfs"}:
        tree = traversal_tree_indices(num_nodes, edges, method=normalized, root=resolved_root)
    elif normalized in {"wilson", "wilson_ust", "ust"}:
        tree = wilson_ust_indices(num_nodes, edges, seed=seed, root=resolved_root)
        normalized = "wilson_ust"
    elif normalized in {"random", "random_priority", "random_priority_kruskal"}:
        tree = spanning_tree_indices(num_nodes, edges, mode="random", seed=seed)
        normalized = "random_priority_kruskal"
    else:
        raise ValueError(
            "unknown chart method; use bfs, dfs, wilson_ust, or random_priority_kruskal"
        )
    incidence = incidence_matrix(num_nodes, edges)
    basis, chords = fundamental_cycle_basis(incidence, tree, return_chords=True)
    chart_name = name or f"{normalized}:root={resolved_root}:seed={seed}"
    return TreeChart(chart_name, tree, chords, basis)


def chart_key(chart: TreeChart) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in chart.tree_edge_indices))


def sample_paper_charts(
    record: GraphRecord,
    *,
    count: int,
    methods: Sequence[str],
    seed: int,
    roots: Sequence[int] | None = None,
    exclude: Iterable[tuple[int, ...]] = (),
    require_distinct: bool = False,
) -> list[TreeChart]:
    """Sample deterministic mixed-method charts with random-root coverage."""

    if count < 1:
        raise ValueError("count must be positive")
    if not methods:
        raise ValueError("at least one chart method is required")
    if record.beta == 0:
        only_chart = build_paper_chart(
            record.num_nodes, record.edges, method="bfs", seed=seed, root=0
        )
        return [only_chart] * count
    forbidden = set(exclude)
    seen = set(forbidden)
    charts: list[TreeChart] = []
    max_attempts = max(128, count * 64)
    for attempt in range(max_attempts):
        method = methods[attempt % len(methods)]
        chart_seed = seed + attempt * 104_729
        if roots:
            root_index = (attempt // len(methods)) % len(roots)
            root = int(roots[root_index]) % record.num_nodes
        else:
            root = int(np.random.default_rng(chart_seed).integers(record.num_nodes))
        chart = build_paper_chart(
            record.num_nodes,
            record.edges,
            method=method,
            seed=chart_seed,
            root=root,
        )
        key = chart_key(chart)
        if key in seen:
            continue
        charts.append(chart)
        seen.add(key)
        if len(charts) == count:
            return charts
    if require_distinct and len(charts) < count:
        raise RuntimeError(
            f"only {len(charts)} distinct charts were available for {record.graph_id}; "
            f"requested {count}"
        )
    if not charts:
        fallback = build_paper_chart(
            record.num_nodes, record.edges, method="bfs", seed=seed, root=0
        )
        charts.append(fallback)
    while len(charts) < count:
        charts.append(charts[len(charts) % len(charts)])
    return charts


def simple_cycle_counts(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    lengths: Sequence[int] = TARGET_CYCLE_LENGTHS,
) -> tuple[int, ...]:
    """Count undirected simple cycles exactly for the requested small lengths."""

    requested = tuple(int(length) for length in lengths)
    if not requested or min(requested) < 3:
        raise ValueError("cycle lengths must all be at least three")
    adjacency = _adjacency(num_nodes, edges, require_connected=True)
    maximum = max(requested)
    cycles: set[tuple[int, ...]] = set()

    def canonical_cycle(path: Sequence[int]) -> tuple[int, ...]:
        values = tuple(path)
        rotations = []
        for orientation in (values, tuple(reversed(values))):
            rotations.extend(
                orientation[offset:] + orientation[:offset] for offset in range(len(orientation))
            )
        return min(rotations)

    for start in range(num_nodes):
        stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [
            (start, (start,), frozenset({start}))
        ]
        while stack:
            node, path, visited = stack.pop()
            for neighbor, _ in adjacency[node]:
                if neighbor == start and len(path) >= 3:
                    cycles.add(canonical_cycle(path))
                    continue
                if neighbor < start or neighbor in visited or len(path) >= maximum:
                    continue
                stack.append((neighbor, (*path, neighbor), visited | {neighbor}))
    counts = {length: 0 for length in requested}
    for cycle in cycles:
        if len(cycle) in counts:
            counts[len(cycle)] += 1
    return tuple(counts[length] for length in requested)


def _cycle_chain_graph(cycle_sizes: Sequence[int]) -> tuple[int, tuple[tuple[int, int], ...]]:
    edges: list[tuple[int, int]] = []
    anchors: list[int] = []
    offset = 0
    for raw_size in cycle_sizes:
        size = int(raw_size)
        if size < 3:
            raise ValueError("cycle size must be at least three")
        nodes = list(range(offset, offset + size))
        anchors.append(nodes[0])
        edges.extend((nodes[index], nodes[(index + 1) % size]) for index in range(size))
        offset += size
    edges.extend((anchors[index], anchors[index + 1]) for index in range(len(anchors) - 1))
    return offset, _canonical_edges(offset, edges)


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _register_unique_graph(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    buckets: dict[tuple[int, int], list[nx.Graph]],
) -> bool:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    bucket = buckets.setdefault((num_nodes, len(edges)), [])
    if any(nx.is_isomorphic(graph, previous) for previous in bucket):
        return False
    bucket.append(graph)
    return True


def build_cyclecount_records(*, seed: int, tiny: bool) -> tuple[GraphRecord, ...]:
    """Create graph-first ID/OOD splits with chart-independent cycle-count labels."""

    counts = (
        {"train": 8, "validation": 3, "id_test": 4, "ood_test": 4}
        if tiny
        else {"train": 128, "validation": 24, "id_test": 40, "ood_test": 40}
    )
    records: list[GraphRecord] = []
    graph_buckets: dict[tuple[int, int], list[nx.Graph]] = {}
    for split in ("train", "validation", "id_test"):
        for index in range(counts[split]):
            for attempt in range(1_000):
                graph_seed = _stable_seed(f"{split}:{index}:{attempt}", seed)
                rng = np.random.default_rng(graph_seed)
                num_nodes = int(rng.integers(8, 13))
                extra_edges = int(rng.integers(2, min(6, num_nodes - 2)))
                edges = _canonical_edges(
                    num_nodes,
                    make_connected_graph(num_nodes, extra_edges, seed=graph_seed),
                )
                if _register_unique_graph(num_nodes, edges, graph_buckets):
                    break
            else:
                raise RuntimeError("failed to generate a unique ID graph split")
            target = simple_cycle_counts(num_nodes, edges)
            records.append(
                GraphRecord(
                    graph_id=f"id-{split}-{index:05d}",
                    family="recursive_tree_plus_chords",
                    split=split,
                    num_nodes=num_nodes,
                    edges=edges,
                    target=tuple(float(value) for value in target),
                )
            )
    for index in range(counts["ood_test"]):
        for attempt in range(1_000):
            graph_seed = _stable_seed(f"ood_test:{index}:{attempt}", seed)
            rng = np.random.default_rng(graph_seed)
            cycle_count = int(rng.integers(2, 5))
            cycle_sizes = tuple(int(value) for value in rng.integers(3, 7, size=cycle_count))
            num_nodes, edges = _cycle_chain_graph(cycle_sizes)
            if _register_unique_graph(num_nodes, edges, graph_buckets):
                break
        else:
            raise RuntimeError("failed to generate a unique OOD graph split")
        target = simple_cycle_counts(num_nodes, edges)
        records.append(
            GraphRecord(
                graph_id=f"ood-cycle-chain-{index:05d}",
                family="cactus_cycle_chain_family_ood",
                split="ood_test",
                num_nodes=num_nodes,
                edges=edges,
                target=tuple(float(value) for value in target),
            )
        )
    return tuple(records)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    def validate_json(temporary: Path) -> None:
        json.loads(temporary.read_text(encoding="utf-8"))

    atomic_write_bytes(path, content, validator=validate_json)


def _load_cached_dataset(
    *,
    suite: str,
    data_path: Path,
    manifest_path: Path,
) -> PreparedDataset:
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "dataset cache and manifest must either both exist or both be absent"
        )
    data_content = data_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256_bytes(data_content)
    if manifest.get("data_sha256") != digest:
        raise ValueError(f"dataset cache checksum mismatch: {data_path}")
    payload = json.loads(data_content)
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"unsupported dataset manifest version: {manifest_path}")
    if payload.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"unsupported dataset cache version: {data_path}")
    if manifest.get("suite") != suite or payload.get("suite") != suite:
        raise ValueError(f"dataset cache suite mismatch for {suite!r}: {data_path}")
    records = tuple(GraphRecord.from_payload(record) for record in payload["records"])
    if int(manifest.get("num_graphs", -1)) != len(records):
        raise ValueError(f"dataset cache graph count mismatch: {data_path}")
    graph_ids = [record.graph_id for record in records]
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError(f"dataset cache contains duplicate graph IDs: {data_path}")
    expected_split_ids: dict[str, list[str]] = {}
    for record in records:
        expected_split_ids.setdefault(record.split, []).append(record.graph_id)
    if manifest.get("split_graph_ids") != expected_split_ids:
        raise ValueError(f"dataset split manifest mismatch: {manifest_path}")
    task_type = str(manifest["task_type"])
    if any(record.task_type != task_type for record in records):
        raise ValueError(f"dataset cache contains conflicting task types: {data_path}")
    if not all(np.all(np.isfinite(record.target)) for record in records):
        raise ValueError(f"dataset cache contains a non-finite target: {data_path}")
    return PreparedDataset(
        suite=suite,
        records=records,
        data_path=data_path,
        manifest_path=manifest_path,
        data_sha256=digest,
        target_names=tuple(str(name) for name in manifest["target_names"]),
        task_type=task_type,
    )


def _cache_records(
    *,
    suite: str,
    records: Sequence[GraphRecord],
    data_path: Path,
    manifest_path: Path,
    target_names: Sequence[str],
    task_type: str,
    source: str,
    seed: int,
    tiny: bool,
) -> PreparedDataset:
    payload = {
        "dataset_version": DATASET_VERSION,
        "suite": suite,
        "records": [record.to_payload() for record in records],
    }
    content = _json_bytes(payload)
    digest = _sha256_bytes(content)
    if data_path.exists() or manifest_path.exists():
        cached = _load_cached_dataset(suite=suite, data_path=data_path, manifest_path=manifest_path)
        if cached.data_sha256 != digest:
            raise ValueError(
                f"existing deterministic cache does not match requested seed/options: {data_path}"
            )
        return cached
    split_ids: dict[str, list[str]] = {}
    for record in records:
        split_ids.setdefault(record.split, []).append(record.graph_id)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "suite": suite,
        "source": source,
        "seed": seed,
        "tiny": tiny,
        "task_type": task_type,
        "target_names": list(target_names),
        "data_path": str(data_path),
        "data_sha256": digest,
        "num_graphs": len(records),
        "split_graph_ids": split_ids,
        "graph_split_before_chart_sampling": True,
        "categorical_feature_schema": {
            "x": "optional non-negative integer atom/node category per node",
            "edge_attr": (
                "optional non-negative integer bond category aligned with each "
                "canonical undirected edge"
            ),
            "missing_value": None,
        },
    }
    _atomic_write(data_path, content)
    _atomic_write(manifest_path, _json_bytes(manifest))
    return _load_cached_dataset(suite=suite, data_path=data_path, manifest_path=manifest_path)


def prepare_cyclecount_dataset(data_root: Path, *, seed: int, tiny: bool) -> PreparedDataset:
    """Create or verify the offline CycleCount-style deterministic cache."""

    variant = "tiny" if tiny else "full"
    cache_dir = data_root.expanduser().resolve() / "cyclecount_ood_v2"
    stem = f"seed-{seed}-{variant}"
    records = build_cyclecount_records(seed=seed, tiny=tiny)
    return _cache_records(
        suite="core",
        records=records,
        data_path=cache_dir / f"{stem}.json",
        manifest_path=cache_dir / f"{stem}.manifest.json",
        target_names=tuple(f"cycles_len_{length}" for length in TARGET_CYCLE_LENGTHS),
        task_type="regression",
        source="generated://tree_augmentation/cyclecount_ood_v2",
        seed=seed,
        tiny=tiny,
    )


def _require_pyg(suite: str) -> tuple[Any, Any]:
    try:
        from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalDatasetError(
            f"suite {suite!r} requires the optional 'torch-geometric' package and its "
            "matching PyTorch wheels. Install it in the active Linux/CUDA environment; "
            "the core offline suite does not require PyG."
        ) from error
    return GNNBenchmarkDataset, ZINC


def _pyg_edges(data: Any) -> tuple[int, tuple[tuple[int, int], ...]]:
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().numpy()
    edges = ((int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1]))
    return num_nodes, _canonical_edges(num_nodes, edges)


def _pyg_categorical_vector(value: Any, *, expected: int, name: str) -> tuple[int, ...]:
    """Read a scalar categorical PyG feature without casting away information."""

    if value is None:
        raise ValueError(f"ZINC record is missing required categorical {name}")
    raw = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.asarray(raw)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) != expected:
        raise ValueError(f"categorical {name} must have shape [{expected}] or [{expected}, 1]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"categorical {name} must use an integer dtype")
    result = tuple(int(item) for item in array.tolist())
    if any(item < 0 for item in result):
        raise ValueError(f"categorical {name} values must be non-negative")
    return result


def zinc_record_from_pyg(
    data: Any,
    *,
    graph_id: str,
    split: str,
) -> GraphRecord:
    """Convert one PyG ZINC molecule while preserving atom and bond categories.

    PyG stores each undirected bond as directed arcs.  The cache stores each
    physical bond once, in canonical edge order, and rejects conflicting
    categories rather than silently choosing one direction.
    """

    num_nodes = int(data.num_nodes)
    x = _pyg_categorical_vector(data.x, expected=num_nodes, name="node x")
    if any(value >= ZINC_NUM_ATOM_TYPES for value in x):
        raise ValueError(f"ZINC node x category exceeds supported range [0, {ZINC_NUM_ATOM_TYPES})")
    edge_index_raw = (
        data.edge_index.detach().cpu().numpy()
        if hasattr(data.edge_index, "detach")
        else np.asarray(data.edge_index)
    )
    edge_index = np.asarray(edge_index_raw)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("PyG edge_index must have shape [2, num_directed_edges]")
    directed_attr = _pyg_categorical_vector(
        data.edge_attr,
        expected=edge_index.shape[1],
        name="bond edge_attr",
    )
    attributes_by_edge: dict[tuple[int, int], int] = {}
    directed_arcs: set[tuple[int, int]] = set()
    for index in range(edge_index.shape[1]):
        u, v = int(edge_index[0, index]), int(edge_index[1, index])
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError("ZINC edge endpoint lies outside [0, num_nodes)")
        if u == v:
            raise ValueError("ZINC self-loops are not supported by the chart protocol")
        if (u, v) in directed_arcs:
            raise ValueError(
                "parallel or duplicate directed ZINC bonds cannot be represented losslessly"
            )
        directed_arcs.add((u, v))
        category = directed_attr[index]
        if category >= ZINC_NUM_BOND_TYPES:
            raise ValueError(
                f"ZINC bond edge_attr category exceeds supported range [0, {ZINC_NUM_BOND_TYPES})"
            )
        edge = (min(u, v), max(u, v))
        previous = attributes_by_edge.setdefault(edge, category)
        if previous != category:
            raise ValueError(f"directed copies of ZINC bond {edge} have conflicting edge_attr")
    edges = _canonical_edges(num_nodes, attributes_by_edge)
    edge_attr = tuple(attributes_by_edge[edge] for edge in edges)
    target_raw = data.y.detach().cpu().numpy() if hasattr(data.y, "detach") else np.asarray(data.y)
    target_array = np.asarray(target_raw).reshape(-1)
    if target_array.size != 1 or not np.isfinite(target_array[0]):
        raise ValueError("ZINC target y must contain exactly one finite scalar")
    return GraphRecord(
        graph_id=graph_id,
        family="ZINC-12K",
        split=split,
        num_nodes=num_nodes,
        edges=edges,
        target=(float(target_array[0]),),
        x=x,
        edge_attr=edge_attr,
    )


def _prepare_csl_records(data_root: Path, *, seed: int, tiny: bool) -> tuple[GraphRecord, ...]:
    GNNBenchmarkDataset, _ = _require_pyg("csl")
    raw_root = data_root / "pyg" / "CSL"
    try:
        dataset = GNNBenchmarkDataset(root=str(raw_root), name="CSL")
    except Exception as error:
        raise OptionalDatasetError(
            f"failed to prepare CSL under {raw_root}. Check network access, write permission, "
            "and the PyG dataset download; original error: {error}"
        ) from error
    labels = [int(dataset[index].y.reshape(-1)[0]) for index in range(len(dataset))]
    folds: dict[int, int] = {}
    for label in sorted(set(labels)):
        members = [index for index, value in enumerate(labels) if value == label]
        rng = np.random.default_rng(_stable_seed(f"csl:{label}", seed))
        for position, index in enumerate(rng.permutation(members)):
            folds[int(index)] = position % 5
    records: list[GraphRecord] = []
    if tiny:
        selected = []
        for label in sorted(set(labels)):
            members = [index for index, value in enumerate(labels) if value == label]
            for fold in range(5):
                candidates = [index for index in members if folds[index] == fold]
                if candidates:
                    selected.append(candidates[0])
        selected.sort()
    else:
        selected = list(range(len(dataset)))
    for index in selected:
        data = dataset[index]
        num_nodes, edges = _pyg_edges(data)
        fold = folds[index]
        split = "train" if fold < 3 else "validation" if fold == 3 else "test"
        records.append(
            GraphRecord(
                graph_id=f"csl-{index:05d}",
                family="CSL",
                split=split,
                num_nodes=num_nodes,
                edges=edges,
                target=(float(labels[index]),),
                task_type="classification",
            )
        )
    return tuple(records)


def _prepare_zinc_records(data_root: Path, *, tiny: bool) -> tuple[GraphRecord, ...]:
    _, ZINC = _require_pyg("zinc")
    raw_root = data_root / "pyg" / "ZINC"
    records: list[GraphRecord] = []
    for split in ("train", "val", "test"):
        try:
            dataset = ZINC(root=str(raw_root), subset=True, split=split)
        except Exception as error:
            raise OptionalDatasetError(
                f"failed to prepare ZINC-12K split {split!r} under {raw_root}. Check network "
                f"access, write permission, and the PyG dataset download; original error: {error}"
            ) from error
        limit = min(len(dataset), 32 if split == "train" else 12) if tiny else len(dataset)
        normalized_split = "validation" if split == "val" else split
        for index in range(limit):
            data = dataset[index]
            records.append(
                zinc_record_from_pyg(
                    data,
                    graph_id=f"zinc-{normalized_split}-{index:05d}",
                    split=normalized_split,
                )
            )
    return tuple(records)


def prepare_optional_pyg_dataset(
    suite: str,
    data_root: Path,
    *,
    seed: int,
    tiny: bool,
    allow_download: bool = False,
) -> PreparedDataset:
    """Prepare CSL or ZINC through optional PyG adapters with verified caches."""

    normalized = suite.lower()
    if normalized not in {"csl", "zinc"}:
        raise ValueError("optional PyG suite must be csl or zinc")
    cache_dir = data_root.expanduser().resolve() / f"{normalized}_pyg_v2"
    variant = "tiny" if tiny else "full"
    stem = f"seed-{seed}-{variant}"
    data_path = cache_dir / f"{stem}.json"
    manifest_path = cache_dir / f"{stem}.manifest.json"
    if data_path.exists() or manifest_path.exists():
        return _load_cached_dataset(
            suite=normalized, data_path=data_path, manifest_path=manifest_path
        )
    if not allow_download:
        raise OptionalDatasetError(
            f"suite {normalized!r} has no verified processed cache under {cache_dir}. "
            "Re-run the CLI with --allow-download to let the PyG adapter access its "
            "public dataset endpoint, or copy a complete cache plus manifest here."
        )
    if normalized == "csl":
        records = _prepare_csl_records(data_root.expanduser().resolve(), seed=seed, tiny=tiny)
        target_names = tuple(f"class_{index}" for index in range(10))
        task_type = "classification"
        source = "PyG:GNNBenchmarkDataset/CSL"
    else:
        records = _prepare_zinc_records(data_root.expanduser().resolve(), tiny=tiny)
        target_names = ("constrained_logP",)
        task_type = "regression"
        source = "PyG:ZINC(subset=True)"
    return _cache_records(
        suite=normalized,
        records=records,
        data_path=data_path,
        manifest_path=manifest_path,
        target_names=target_names,
        task_type=task_type,
        source=source,
        seed=seed,
        tiny=tiny,
    )


def validate_prepared_cache(
    suite: str,
    data_root: Path,
    *,
    seed: int,
    tiny: bool,
) -> PreparedDataset:
    """Validate one requested processed cache without generating or downloading data."""

    normalized = suite.lower()
    if normalized not in {"core", "csl", "zinc"}:
        raise ValueError("suite must be core, csl, or zinc")
    variant = "tiny" if tiny else "full"
    cache_name = "cyclecount_ood_v2" if normalized == "core" else f"{normalized}_pyg_v2"
    cache_dir = data_root.expanduser().resolve() / cache_name
    stem = f"seed-{seed}-{variant}"
    data_path = cache_dir / f"{stem}.json"
    manifest_path = cache_dir / f"{stem}.manifest.json"
    present = (data_path.is_file(), manifest_path.is_file())
    if not any(present):
        raise FileNotFoundError(f"tree {normalized} cache is missing for seed={seed}: {data_path}")
    if not all(present):
        raise CacheIncompleteError(
            f"tree {normalized} data and manifest must both exist: {cache_dir}"
        )
    try:
        prepared = _load_cached_dataset(
            suite=normalized, data_path=data_path, manifest_path=manifest_path
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise CacheCorruptError(f"invalid tree {normalized} processed cache") from error
    if manifest.get("seed") != int(seed) or manifest.get("tiny") is not bool(tiny):
        raise CacheWrongRequestError(f"tree {normalized} cache seed/profile mismatch")
    expected_source = {
        "core": "generated://tree_augmentation/cyclecount_ood_v2",
        "csl": "PyG:GNNBenchmarkDataset/CSL",
        "zinc": "PyG:ZINC(subset=True)",
    }[normalized]
    if manifest.get("source") != expected_source:
        raise CacheWrongRequestError(f"tree {normalized} cache source mismatch")
    expected_counts = {
        "core": (
            {"train": 8, "validation": 3, "id_test": 4, "ood_test": 4}
            if tiny
            else {"train": 128, "validation": 24, "id_test": 40, "ood_test": 40}
        ),
        "csl": (
            {"train": 30, "validation": 10, "test": 10}
            if tiny
            else {"train": 90, "validation": 30, "test": 30}
        ),
        "zinc": (
            {"train": 32, "validation": 12, "test": 12}
            if tiny
            else {"train": 10_000, "validation": 1_000, "test": 1_000}
        ),
    }[normalized]
    actual_counts = {name: len(ids) for name, ids in manifest["split_graph_ids"].items()}
    if actual_counts != expected_counts:
        raise CacheCorruptError(f"tree {normalized} split cardinalities are invalid")
    expected_target_width = 4 if normalized == "core" else 1
    for record in prepared.records:
        if len(record.target) != expected_target_width or not np.all(np.isfinite(record.target)):
            raise CacheCorruptError(f"tree {normalized} target shape or value is invalid")
        if normalized == "zinc":
            if record.x is None or len(record.x) != record.num_nodes:
                raise CacheCorruptError("tree ZINC atom features are missing or misaligned")
            if record.edge_attr is None or len(record.edge_attr) != len(record.edges):
                raise CacheCorruptError("tree ZINC bond features are missing or misaligned")
            if any(not 0 <= int(value) < ZINC_NUM_ATOM_TYPES for value in record.x):
                raise CacheCorruptError("tree ZINC atom category is outside the supported range")
            if any(not 0 <= int(value) < ZINC_NUM_BOND_TYPES for value in record.edge_attr):
                raise CacheCorruptError("tree ZINC bond category is outside the supported range")
    return prepared


__all__ = [
    "DATASET_VERSION",
    "TARGET_CYCLE_LENGTHS",
    "ZINC_NUM_ATOM_TYPES",
    "ZINC_NUM_BOND_TYPES",
    "GraphRecord",
    "OptionalDatasetError",
    "PreparedDataset",
    "build_cyclecount_records",
    "build_paper_chart",
    "chart_key",
    "prepare_cyclecount_dataset",
    "prepare_optional_pyg_dataset",
    "sample_paper_charts",
    "simple_cycle_counts",
    "traversal_tree_indices",
    "validate_prepared_cache",
    "wilson_ust_indices",
    "zinc_record_from_pyg",
]
````

# research/tree_augmentation/paper_model.py

````python
"""Variable-beta encoder and fair multi-chart downstream training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .paper_data import (
    ZINC_NUM_ATOM_TYPES,
    ZINC_NUM_BOND_TYPES,
    GraphRecord,
    chart_key,
    sample_paper_charts,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GraphChartView:
    """One chart view of one physical graph."""

    graph_id: str
    graph_family: str
    graph_status: str
    chart_status: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    basis: FloatArray
    target: tuple[float, ...]
    chart_name: str
    tree_key: tuple[int, ...]
    x: tuple[int, ...] | None = None
    edge_attr: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PaddedChartBatch:
    """Dense padded batch with independent masks for edges and cycle columns."""

    basis: Tensor
    edge_features: Tensor
    edge_mask: Tensor
    cycle_mask: Tensor
    edge_index: Tensor
    node_categories: Tensor
    edge_categories: Tensor
    node_mask: Tensor
    targets: Tensor
    graph_ids: tuple[str, ...]

    @property
    def x(self) -> Tensor:
        """Categorical node input, including the explicit missing-feature sentinel."""

        return self.node_categories

    @property
    def edge_attr(self) -> Tensor:
        """Undirected-edge-aligned categorical bond input."""

        return self.edge_categories

    def pin_memory(self) -> PaddedChartBatch:
        """Pin tensor fields so ``DataLoader(pin_memory=True)`` can handle the batch."""

        return PaddedChartBatch(
            basis=self.basis.pin_memory(),
            edge_features=self.edge_features.pin_memory(),
            edge_mask=self.edge_mask.pin_memory(),
            cycle_mask=self.cycle_mask.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            node_categories=self.node_categories.pin_memory(),
            edge_categories=self.edge_categories.pin_memory(),
            node_mask=self.node_mask.pin_memory(),
            targets=self.targets.pin_memory(),
            graph_ids=self.graph_ids,
        )

    def to(
        self,
        device: torch.device,
        *,
        pin_memory: bool,
        non_blocking: bool,
    ) -> PaddedChartBatch:
        def move(tensor: Tensor) -> Tensor:
            value = tensor.pin_memory() if pin_memory and not tensor.is_pinned() else tensor
            return value.to(device, non_blocking=non_blocking)

        return PaddedChartBatch(
            basis=move(self.basis),
            edge_features=move(self.edge_features),
            edge_mask=move(self.edge_mask),
            cycle_mask=move(self.cycle_mask),
            edge_index=move(self.edge_index),
            node_categories=move(self.node_categories),
            edge_categories=move(self.edge_categories),
            node_mask=move(self.node_mask),
            targets=move(self.targets),
            graph_ids=self.graph_ids,
        )


@dataclass(frozen=True)
class FitResult:
    model: nn.Module
    target_mean: FloatArray
    target_scale: FloatArray
    history: tuple[dict[str, float], ...]


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _edge_features(record: GraphChartView) -> FloatArray:
    degrees = np.zeros(record.num_nodes, dtype=np.float64)
    for u, v in record.edges:
        degrees[u] += 1.0
        degrees[v] += 1.0
    max_degree = max(1.0, float(degrees.max()))
    num_edges = max(1, len(record.edges))
    result = np.empty((len(record.edges), 4), dtype=np.float64)
    for edge_index, (u, v) in enumerate(record.edges):
        low, high = sorted((degrees[u], degrees[v]))
        result[edge_index] = (
            low / max_degree,
            high / max_degree,
            1.0 / record.num_nodes,
            1.0 / num_edges,
        )
    return result


def collate_chart_views(views: Sequence[GraphChartView]) -> PaddedChartBatch:
    """Pad variable edge/cycle dimensions without exposing padded values."""

    if not views:
        raise ValueError("views must not be empty")
    target_dim = len(views[0].target)
    if target_dim < 1 or any(len(view.target) != target_dim for view in views):
        raise ValueError("all views must have the same positive target dimension")
    max_edges = max(len(view.edges) for view in views)
    max_nodes = max(view.num_nodes for view in views)
    max_beta = max(view.basis.shape[1] for view in views)
    batch_size = len(views)
    basis = torch.zeros((batch_size, max_edges, max_beta), dtype=torch.float32)
    edge_features = torch.zeros((batch_size, max_edges, 4), dtype=torch.float32)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool)
    cycle_mask = torch.zeros((batch_size, max_beta), dtype=torch.bool)
    edge_index = torch.zeros((batch_size, max_edges, 2), dtype=torch.long)
    node_categories = torch.full((batch_size, max_nodes), ZINC_NUM_ATOM_TYPES, dtype=torch.long)
    edge_categories = torch.full((batch_size, max_edges), ZINC_NUM_BOND_TYPES, dtype=torch.long)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    targets = torch.empty((batch_size, target_dim), dtype=torch.float32)
    for batch_index, view in enumerate(views):
        num_edges, beta = view.basis.shape
        if num_edges != len(view.edges):
            raise ValueError("basis edge dimension does not match the physical graph")
        if view.x is not None:
            if len(view.x) != view.num_nodes:
                raise ValueError("categorical node x must have one value per node")
            if any(value < 0 or value >= ZINC_NUM_ATOM_TYPES for value in view.x):
                raise ValueError("categorical node x is outside the supported ZINC range")
            node_categories[batch_index, : view.num_nodes] = torch.as_tensor(
                view.x, dtype=torch.long
            )
        if view.edge_attr is not None:
            if len(view.edge_attr) != num_edges:
                raise ValueError("categorical edge_attr must align with undirected edges")
            if any(value < 0 or value >= ZINC_NUM_BOND_TYPES for value in view.edge_attr):
                raise ValueError("categorical edge_attr is outside the supported ZINC range")
            edge_categories[batch_index, :num_edges] = torch.as_tensor(
                view.edge_attr, dtype=torch.long
            )
        if beta:
            basis[batch_index, :num_edges, :beta] = torch.as_tensor(
                np.array(view.basis, copy=True), dtype=torch.float32
            )
            cycle_mask[batch_index, :beta] = True
        edge_features[batch_index, :num_edges] = torch.as_tensor(
            _edge_features(view), dtype=torch.float32
        )
        edge_index[batch_index, :num_edges] = torch.as_tensor(view.edges, dtype=torch.long)
        edge_mask[batch_index, :num_edges] = True
        node_mask[batch_index, : view.num_nodes] = True
        targets[batch_index] = torch.as_tensor(view.target, dtype=torch.float32)
    return PaddedChartBatch(
        basis=basis,
        edge_features=edge_features,
        edge_mask=edge_mask,
        cycle_mask=cycle_mask,
        edge_index=edge_index,
        node_categories=node_categories,
        edge_categories=edge_categories,
        node_mask=node_mask,
        targets=targets,
        graph_ids=tuple(view.graph_id for view in views),
    )


class VariableBetaCycleEncoder(nn.Module):
    """Orientation-gauge-safe full-beta chart encoder with masked graph readout.

    Each edge sees a set of sign-even cycle-column memberships.  A shared
    coordinate MLP is pooled over valid columns, then a second MLP is pooled
    over valid edges.  The sign-even inputs remove arbitrary edge-orientation
    and fundamental-cycle direction gauges, while the set pooling removes edge
    and cycle-column ordering.  Neither ``max_edges`` nor ``max_beta`` is a
    learned architectural constant.

    This guarantees invariance when the same physical tree is represented with
    another orientation, ordering, or node labeling.  It does not make two
    *different* spanning-tree charts identical: label-dependent BFS/DFS
    preprocessing may still select another tree, which is the chart-shift axis
    measured by this track.
    """

    def __init__(self, *, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if hidden_dim < 4 or output_dim < 1:
            raise ValueError("hidden_dim >= 4 and output_dim >= 1 are required")
        self.coordinate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        chemistry_dim = max(4, hidden_dim // 4)
        self.atom_embedding = nn.Embedding(ZINC_NUM_ATOM_TYPES + 1, chemistry_dim)
        self.bond_embedding = nn.Embedding(ZINC_NUM_BOND_TYPES + 1, chemistry_dim)
        self.edge = nn.Sequential(
            nn.Linear(3 * hidden_dim + 4 + 3 * chemistry_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _masked_max(values: Tensor, mask: Tensor, *, dimension: int) -> Tensor:
        masked = values.masked_fill(~mask, -torch.inf)
        maximum = masked.amax(dim=dimension)
        return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    def forward(self, batch: PaddedChartBatch) -> Tensor:
        basis = batch.basis
        batch_size, max_edges, max_beta = basis.shape
        hidden_dim = self.coordinate[0].out_features
        if max_beta:
            edge_counts = batch.edge_mask.sum(dim=1).clamp_min(1)[:, None]
            normalized_cycle_support = basis.abs().sum(dim=1) / edge_counts
            normalized_cycle_support = normalized_cycle_support[:, None, :].expand(
                batch_size, max_edges, max_beta
            )
            coordinate_input = torch.stack(
                (basis.abs(), basis.square(), normalized_cycle_support), dim=-1
            )
            coordinate_hidden = self.coordinate(coordinate_input)
            coordinate_mask = (batch.edge_mask[:, :, None] & batch.cycle_mask[:, None, :])[
                :, :, :, None
            ]
            coordinate_hidden = coordinate_hidden * coordinate_mask
            count = coordinate_mask.sum(dim=2).clamp_min(1)
            coordinate_sum = coordinate_hidden.sum(dim=2)
            coordinate_mean = coordinate_sum / count
            coordinate_max = self._masked_max(
                coordinate_hidden,
                coordinate_mask,
                dimension=2,
            )
        else:
            zeros = basis.new_zeros((batch_size, max_edges, hidden_dim))
            coordinate_sum = zeros
            coordinate_mean = zeros
            coordinate_max = zeros
        atom_hidden = self.atom_embedding(batch.node_categories)
        atom_hidden = atom_hidden * batch.node_mask[:, :, None]
        batch_indices = torch.arange(batch_size, device=basis.device)[:, None]
        start = batch.edge_index[:, :, 0]
        end = batch.edge_index[:, :, 1]
        start_atom = atom_hidden[batch_indices, start]
        end_atom = atom_hidden[batch_indices, end]
        bond_hidden = self.bond_embedding(batch.edge_categories)
        chemistry = torch.cat(
            (start_atom + end_atom, (start_atom - end_atom).abs(), bond_hidden), dim=-1
        )
        edge_input = torch.cat(
            (
                coordinate_sum,
                coordinate_mean,
                coordinate_max,
                batch.edge_features,
                chemistry,
            ),
            dim=-1,
        )
        edge_hidden = self.edge(edge_input)
        edge_mask = batch.edge_mask[:, :, None]
        edge_hidden = edge_hidden * edge_mask
        edge_count = edge_mask.sum(dim=1).clamp_min(1)
        edge_sum = edge_hidden.sum(dim=1)
        edge_mean = edge_sum / edge_count
        edge_max = self._masked_max(edge_hidden, edge_mask, dimension=1)
        return self.head(torch.cat((edge_sum, edge_mean, edge_max), dim=-1))


def build_chart_views(
    records: Sequence[GraphRecord],
    *,
    chart_status: str,
    count: int,
    methods: Sequence[str],
    seed: int,
    roots: Sequence[int] | None = None,
    exclude_by_graph: Mapping[str, set[tuple[int, ...]]] | None = None,
    require_distinct: bool = False,
) -> list[GraphChartView]:
    """Generate chart views only after the physical graph split is fixed."""

    views: list[GraphChartView] = []
    for record in records:
        graph_seed = _stable_seed(f"{chart_status}:{record.graph_id}", seed)
        charts = sample_paper_charts(
            record,
            count=count,
            methods=methods,
            seed=graph_seed,
            roots=roots,
            exclude=(exclude_by_graph or {}).get(record.graph_id, set()),
            require_distinct=require_distinct,
        )
        graph_status = "ood" if record.split == "ood_test" else "id"
        for chart in charts:
            views.append(
                GraphChartView(
                    graph_id=record.graph_id,
                    graph_family=record.family,
                    graph_status=graph_status,
                    chart_status=chart_status,
                    num_nodes=record.num_nodes,
                    edges=record.edges,
                    basis=chart.basis,
                    target=record.target,
                    chart_name=chart.name,
                    tree_key=chart_key(chart),
                    x=record.x,
                    edge_attr=record.edge_attr,
                )
            )
    return views


def _unique_graph_targets(views: Sequence[GraphChartView]) -> FloatArray:
    targets: dict[str, tuple[float, ...]] = {}
    for view in views:
        previous = targets.setdefault(view.graph_id, view.target)
        if previous != view.target:
            raise ValueError("one graph_id was assigned conflicting downstream targets")
    return np.asarray(list(targets.values()), dtype=np.float64)


def fit_downstream_model(
    views: Sequence[GraphChartView],
    *,
    task_type: str,
    output_dim: int,
    hidden_dim: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> FitResult:
    """Fit with a fixed number of optimizer updates for fair chart comparisons."""

    if not views:
        raise ValueError("training views must not be empty")
    if updates < 1 or batch_size < 1 or workers < 0:
        raise ValueError("updates/batch_size must be positive and workers non-negative")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")
    use_amp = bool(amp and device.type == "cuda")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = VariableBetaCycleEncoder(hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_grad_scaler = getattr(torch.amp, "GradScaler", None)
    if amp_grad_scaler is not None:
        scaler = amp_grad_scaler("cuda", enabled=use_amp)
    else:  # pragma: no cover - compatibility with the minimum supported torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    graph_targets = _unique_graph_targets(views)
    if task_type == "regression":
        target_mean = graph_targets.mean(axis=0)
        target_scale = graph_targets.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
    elif task_type == "classification":
        target_mean = np.zeros(1, dtype=np.float64)
        target_scale = np.ones(1, dtype=np.float64)
    else:
        raise ValueError("task_type must be regression or classification")
    mean_tensor = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(target_scale, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 101)
    sampled_indices = torch.randint(len(views), (updates, batch_size), generator=generator).tolist()
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 202)
    loader = DataLoader(
        list(views),
        batch_sampler=sampled_indices,
        collate_fn=collate_chart_views,
        num_workers=workers,
        pin_memory=pin_memory and device.type == "cuda",
        generator=loader_generator,
    )
    history: list[dict[str, float]] = []
    model.train()
    for update, cpu_batch in enumerate(loader, start=1):
        batch = cpu_batch.to(
            device,
            pin_memory=pin_memory and device.type == "cuda",
            non_blocking=non_blocking and device.type == "cuda",
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(batch)
            if task_type == "classification":
                loss = nn.functional.cross_entropy(prediction, batch.targets[:, 0].to(torch.long))
            else:
                normalized = (batch.targets - mean_tensor) / scale_tensor
                loss = nn.functional.mse_loss(prediction, normalized)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if update == 1 or update == updates or update % max(1, updates // 10) == 0:
            history.append({"update": float(update), "loss": float(loss.detach().cpu())})
    return FitResult(
        model=model,
        target_mean=np.asarray(target_mean, dtype=np.float64),
        target_scale=np.asarray(target_scale, dtype=np.float64),
        history=tuple(history),
    )


@torch.no_grad()
def _predict(
    fitted: FitResult,
    views: Sequence[GraphChartView],
    *,
    task_type: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> FloatArray:
    fitted.model.eval()
    use_amp = bool(amp and device.type == "cuda")
    predictions: list[FloatArray] = []
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(0)
    loader = DataLoader(
        list(views),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_chart_views,
        num_workers=workers,
        pin_memory=pin_memory and device.type == "cuda",
        generator=loader_generator,
    )
    for cpu_batch in loader:
        batch = cpu_batch.to(
            device,
            pin_memory=pin_memory and device.type == "cuda",
            non_blocking=non_blocking and device.type == "cuda",
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = fitted.model(batch)
        values = output.float().cpu().numpy().astype(np.float64, copy=False)
        if task_type == "regression":
            values = values * fitted.target_scale + fitted.target_mean
        predictions.append(values)
    return np.concatenate(predictions, axis=0)


def _group_indices(views: Sequence[GraphChartView]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, view in enumerate(views):
        groups.setdefault(view.graph_id, []).append(index)
    return groups


def _regression_metrics(
    views: Sequence[GraphChartView], predictions: FloatArray, target_scale: FloatArray
) -> dict[str, float]:
    targets = np.asarray([view.target for view in views], dtype=np.float64)
    errors = np.abs(predictions - targets)
    view_mae = errors.mean(axis=1)
    graph_macro = []
    graph_worst = []
    chart_std = []
    flip_rates = []
    for indices in _group_indices(views).values():
        selected = np.asarray(indices, dtype=np.int64)
        graph_macro.append(float(view_mae[selected].mean()))
        graph_worst.append(float(view_mae[selected].max()))
        chart_std.append(float(predictions[selected].std(axis=0).mean()))
        rounded = np.rint(predictions[selected]).astype(np.int64)
        flip_rates.append(float(np.mean(np.any(rounded != rounded[:1], axis=1))))
    safe_scale = np.where(target_scale < 1e-6, 1.0, target_scale)
    return {
        "mae": float(errors.mean()),
        "normalized_mae": float((errors / safe_scale).mean()),
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "graph_macro_mae": float(np.mean(graph_macro)),
        "worst_chart_mae": float(np.mean(graph_worst)),
        "chart_prediction_std": float(np.mean(chart_std)),
        "prediction_flip_rate": float(np.mean(flip_rates)),
        "rounded_exact_vector_accuracy": float(
            np.mean(np.all(np.rint(predictions) == targets, axis=1))
        ),
    }


def _classification_metrics(
    views: Sequence[GraphChartView], logits: FloatArray
) -> dict[str, float]:
    targets = np.asarray([int(view.target[0]) for view in views], dtype=np.int64)
    predictions = logits.argmax(axis=1)
    correct = predictions == targets
    graph_accuracy = []
    graph_worst = []
    flip_rates = []
    probability_std = []
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    for indices in _group_indices(views).values():
        selected = np.asarray(indices, dtype=np.int64)
        graph_accuracy.append(float(correct[selected].mean()))
        graph_worst.append(float(correct[selected].min()))
        flip_rates.append(float(np.mean(predictions[selected] != predictions[selected[0]])))
        probability_std.append(float(probabilities[selected].std(axis=0).mean()))
    return {
        "accuracy": float(correct.mean()),
        "graph_macro_accuracy": float(np.mean(graph_accuracy)),
        "worst_chart_accuracy": float(np.mean(graph_worst)),
        "chart_probability_std": float(np.mean(probability_std)),
        "prediction_flip_rate": float(np.mean(flip_rates)),
    }


def evaluate_downstream_model(
    fitted: FitResult,
    views: Sequence[GraphChartView],
    *,
    task_type: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> dict[str, float]:
    if not views:
        raise ValueError("evaluation views must not be empty")
    predictions = _predict(
        fitted,
        views,
        task_type=task_type,
        batch_size=batch_size,
        device=device,
        amp=amp,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        workers=workers,
    )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("model produced non-finite predictions")
    if task_type == "classification":
        return _classification_metrics(views, predictions)
    return _regression_metrics(views, predictions, fitted.target_scale)


def run_fixed_vs_multichart(
    *,
    fixed_train_views: Sequence[GraphChartView],
    multi_train_views: Sequence[GraphChartView],
    evaluation_views: Mapping[str, Sequence[GraphChartView]],
    task_type: str,
    output_dim: int,
    hidden_dim: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    amp: bool,
    pin_memory: bool,
    non_blocking: bool,
    workers: int,
) -> tuple[dict[str, Any], dict[str, FitResult]]:
    """Train fair fixed/multi models and evaluate every requested quadrant."""

    common = {
        "task_type": task_type,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": device,
        "seed": seed,
        "amp": amp,
        "pin_memory": pin_memory,
        "non_blocking": non_blocking,
        "workers": workers,
    }
    fixed = fit_downstream_model(fixed_train_views, **common)
    multi = fit_downstream_model(multi_train_views, **common)
    models = {"fixed_bfs": fixed, "multi_chart": multi}
    metrics: dict[str, Any] = {}
    for model_name, fitted in models.items():
        metrics[model_name] = {
            "optimizer_updates": updates,
            "num_training_views": (
                len(fixed_train_views) if model_name == "fixed_bfs" else len(multi_train_views)
            ),
            "history": list(fitted.history),
            "quadrants": {},
        }
        for quadrant, views in evaluation_views.items():
            values = evaluate_downstream_model(
                fitted,
                views,
                task_type=task_type,
                batch_size=batch_size,
                device=device,
                amp=amp,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                workers=workers,
            )
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"non-finite metric in {model_name}/{quadrant}")
            metrics[model_name]["quadrants"][quadrant] = values
    return metrics, models


__all__ = [
    "FitResult",
    "GraphChartView",
    "PaddedChartBatch",
    "VariableBetaCycleEncoder",
    "build_chart_views",
    "collate_chart_views",
    "evaluate_downstream_model",
    "fit_downstream_model",
    "run_fixed_vs_multichart",
]
````

# research/tree_augmentation/run.py

````python
"""Run the standalone spanning-tree augmentation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chartgat.graphs import make_connected_graph
from research.tree_augmentation.augmentation import (
    ensure_full_cycle_budget,
    find_unseen_chart,
    lossless_transition_error,
    run_static_cycle_pe_probe,
    sample_tree_charts,
    transition_cocycle_error,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute chart certification and the fixed/multi/unseen static probe."""

    graph_config = config["graph"]
    augmentation_config = config["augmentation"]
    probe_config = config["probe"]
    seed = int(config.get("seed", 0))

    if augmentation_config.get("lossy_extension_enabled", False):
        raise NotImplementedError("the k < beta lossy extension is intentionally disabled")

    num_nodes = int(graph_config["num_nodes"])
    edges = make_connected_graph(
        num_nodes,
        int(graph_config["extra_edges"]),
        seed=int(graph_config["seed"]),
    )
    charts = sample_tree_charts(
        num_nodes,
        edges,
        include_bfs=bool(augmentation_config.get("include_bfs", True)),
        include_dfs=bool(augmentation_config.get("include_dfs", True)),
        random_count=int(augmentation_config.get("random_count", 0)),
        random_seed_start=int(augmentation_config.get("random_seed_start", 0)),
    )
    beta = charts[0].beta
    ensure_full_cycle_budget(beta, augmentation_config.get("k"))
    unseen = find_unseen_chart(
        num_nodes,
        edges,
        charts,
        seed_start=int(augmentation_config.get("unseen_seed_start", 10_000)),
    )

    rng = np.random.default_rng(seed)
    coordinates = rng.normal(size=beta)
    certification_charts = [*charts, unseen]
    certification = {
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "cycle_rank_beta": beta,
        "full_beta_enabled": True,
        "lossy_k_lt_beta_enabled": False,
        "unique_training_trees": len(charts),
        "lossless_transition_error": lossless_transition_error(certification_charts, coordinates),
        "transition_cocycle_error": transition_cocycle_error(certification_charts),
    }
    probe = run_static_cycle_pe_probe(
        charts,
        unseen,
        hidden_dim=int(probe_config.get("hidden_dim", 48)),
        epochs=int(probe_config.get("epochs", 800)),
        learning_rate=float(probe_config.get("learning_rate", 0.01)),
        weight_decay=float(probe_config.get("weight_decay", 1e-5)),
        seed=seed,
    )
    return {
        "track": "static_cycle_pe_tree_augmentation",
        "certification": certification,
        "probe": probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).with_name("config.yaml")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    result = run(config)
    if args.output is not None:
        output = args.output.expanduser()
    else:
        output = Path(config.get("output", "results/summary.json")).expanduser()
        if not output.is_absolute():
            output = config_path.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
````

# research/tree_augmentation/tests/test_paper.py

````python
"""Offline fixtures for the independent tree-augmentation paper path."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix, validate_spanning_tree
from chartgat.seeds import SeedAxes
from research.tree_augmentation import paper as tree_paper
from research.tree_augmentation.paper import main, run_suite
from research.tree_augmentation.paper_data import (
    GraphRecord,
    OptionalDatasetError,
    build_paper_chart,
    prepare_cyclecount_dataset,
    prepare_optional_pyg_dataset,
    simple_cycle_counts,
    traversal_tree_indices,
    wilson_ust_indices,
    zinc_record_from_pyg,
)
from research.tree_augmentation.paper_model import (
    GraphChartView,
    VariableBetaCycleEncoder,
    build_chart_views,
    collate_chart_views,
)


def test_wilson_is_deterministic_valid_and_uniform_on_triangle() -> None:
    edges = ((0, 1), (0, 2), (1, 2))
    first = wilson_ust_indices(3, edges, seed=19, root=2)
    second = wilson_ust_indices(3, edges, seed=19, root=2)
    np.testing.assert_array_equal(first, second)
    validate_spanning_tree(incidence_matrix(3, edges), first)

    counts = Counter(
        tuple(int(index) for index in wilson_ust_indices(3, edges, seed=seed))
        for seed in range(1_500)
    )
    assert set(counts) == {(0, 1), (0, 2), (1, 2)}
    frequencies = np.asarray(list(counts.values()), dtype=np.float64) / 1_500
    assert np.max(np.abs(frequencies - 1.0 / 3.0)) < 0.04


def test_random_root_traversals_and_legacy_sampler_stay_separate() -> None:
    edges = ((0, 1), (0, 3), (1, 2), (2, 3), (0, 2))
    root_zero = traversal_tree_indices(4, edges, method="bfs", root=0)
    root_two = traversal_tree_indices(4, edges, method="bfs", root=2)
    assert tuple(root_zero) != tuple(root_two)
    for method in ("bfs", "dfs", "wilson_ust", "random_priority_kruskal"):
        chart = build_paper_chart(4, edges, method=method, seed=11, root=1)
        validate_spanning_tree(incidence_matrix(4, edges), chart.tree_edge_indices)
        assert chart.beta == 2
    assert "wilson_ust" in build_paper_chart(4, edges, method="wilson_ust", seed=11, root=1).name
    assert (
        "random_priority_kruskal"
        in build_paper_chart(4, edges, method="random_priority_kruskal", seed=11, root=1).name
    )


def test_cyclecount_target_is_chart_independent() -> None:
    triangle_and_square = (
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 5),
    )
    assert simple_cycle_counts(6, triangle_and_square) == (1, 1, 0, 0)
    bfs = build_paper_chart(6, triangle_and_square, method="bfs", seed=1, root=0)
    ust = build_paper_chart(6, triangle_and_square, method="wilson_ust", seed=9, root=4)
    assert bfs.beta == ust.beta == 2
    assert simple_cycle_counts(6, triangle_and_square) == (1, 1, 0, 0)


def _view(record: GraphRecord) -> GraphChartView:
    chart = build_paper_chart(record.num_nodes, record.edges, method="bfs", seed=3, root=0)
    return GraphChartView(
        graph_id=record.graph_id,
        graph_family=record.family,
        graph_status="id",
        chart_status="seen",
        num_nodes=record.num_nodes,
        edges=record.edges,
        basis=chart.basis,
        target=record.target,
        chart_name=chart.name,
        tree_key=tuple(int(index) for index in chart.tree_edge_indices),
        x=record.x,
        edge_attr=record.edge_attr,
    )


def test_variable_beta_batch_masks_tree_cycle_and_multicycle() -> None:
    records = (
        GraphRecord("tree", "fixture", "train", 4, ((0, 1), (1, 2), (2, 3)), (0.0,)),
        GraphRecord(
            "cycle",
            "fixture",
            "train",
            4,
            ((0, 1), (0, 3), (1, 2), (2, 3)),
            (1.0,),
        ),
        GraphRecord(
            "multi",
            "fixture",
            "train",
            5,
            ((0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)),
            (2.0,),
        ),
    )
    batch = collate_chart_views([_view(record) for record in records])
    assert batch.basis.shape == (3, 6, 2)
    assert batch.cycle_mask.sum(dim=1).tolist() == [0, 1, 2]
    assert batch.edge_mask.sum(dim=1).tolist() == [3, 4, 6]
    assert torch.all(batch.node_categories[batch.node_mask] == 28)
    assert torch.all(batch.edge_categories[batch.edge_mask] == 4)
    output = VariableBetaCycleEncoder(hidden_dim=8, output_dim=2)(batch)
    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()


def _gauge_fixture_view() -> GraphChartView:
    record = GraphRecord(
        "gauge-fixture",
        "fixture",
        "train",
        5,
        ((0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)),
        (2.0,),
        x=(1, 2, 3, 4, 5),
        edge_attr=(0, 1, 2, 3, 1, 0),
    )
    return _view(record)


def _gauge_predictions(views: list[GraphChartView]) -> torch.Tensor:
    torch.manual_seed(109)
    model = VariableBetaCycleEncoder(hidden_dim=12, output_dim=2).eval()
    with torch.no_grad():
        return model(collate_chart_views(views))


def test_encoder_ignores_legal_edge_orientation_and_cycle_column_gauges() -> None:
    original = _gauge_fixture_view()
    orientation_signs = np.asarray((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    reoriented_edges = tuple(
        (v, u) if orientation_signs[index] < 0 else (u, v)
        for index, (u, v) in enumerate(original.edges)
    )
    reoriented_basis, chords = fundamental_cycle_basis(
        incidence_matrix(original.num_nodes, reoriented_edges),
        original.tree_key,
        return_chords=True,
    )
    expected_basis = (
        orientation_signs[:, None] * original.basis * orientation_signs[chords][None, :]
    )
    np.testing.assert_allclose(reoriented_basis, expected_basis, atol=1e-12)
    reoriented = replace(original, edges=reoriented_edges, basis=reoriented_basis)

    column_order = np.asarray((1, 0))
    column_signs = np.asarray((-1.0, 1.0))
    signed_column_permutation = replace(
        original,
        basis=original.basis[:, column_order] * column_signs[None, :],
    )
    predictions = _gauge_predictions([original, reoriented, signed_column_permutation])
    torch.testing.assert_close(predictions[1], predictions[0], atol=1e-7, rtol=0.0)
    torch.testing.assert_close(predictions[2], predictions[0], atol=1e-7, rtol=0.0)


def test_encoder_ignores_aligned_edge_order_permutations() -> None:
    original = _gauge_fixture_view()
    edge_order = np.asarray((4, 2, 0, 5, 1, 3))
    old_to_new = {int(old): new for new, old in enumerate(edge_order)}
    reordered = replace(
        original,
        edges=tuple(original.edges[index] for index in edge_order),
        basis=original.basis[edge_order],
        edge_attr=tuple(original.edge_attr[index] for index in edge_order),
        tree_key=tuple(sorted(old_to_new[index] for index in original.tree_key)),
    )
    predictions = _gauge_predictions([original, reordered])
    torch.testing.assert_close(predictions[1], predictions[0], atol=1e-7, rtol=0.0)


def test_encoder_ignores_same_tree_node_relabeling_with_mapped_chemistry() -> None:
    original = _gauge_fixture_view()
    old_to_new_node = (4, 1, 3, 0, 2)
    mapped_edges = []
    for old_edge_index, (u, v) in enumerate(original.edges):
        mapped_u, mapped_v = old_to_new_node[u], old_to_new_node[v]
        mapped_edges.append(((min(mapped_u, mapped_v), max(mapped_u, mapped_v)), old_edge_index))
    mapped_edges.sort()
    relabeled_edges = tuple(edge for edge, _ in mapped_edges)
    old_to_new_edge = {
        old_edge_index: new_edge_index
        for new_edge_index, (_, old_edge_index) in enumerate(mapped_edges)
    }
    relabeled_tree = tuple(
        sorted(old_to_new_edge[old_edge_index] for old_edge_index in original.tree_key)
    )
    relabeled_basis = fundamental_cycle_basis(
        incidence_matrix(original.num_nodes, relabeled_edges), relabeled_tree
    )
    relabeled_x = [0] * original.num_nodes
    for old_node, new_node in enumerate(old_to_new_node):
        relabeled_x[new_node] = original.x[old_node]
    relabeled = replace(
        original,
        edges=relabeled_edges,
        basis=relabeled_basis,
        tree_key=relabeled_tree,
        x=tuple(relabeled_x),
        edge_attr=tuple(original.edge_attr[old_index] for _, old_index in mapped_edges),
    )
    predictions = _gauge_predictions([original, relabeled])
    torch.testing.assert_close(predictions[1], predictions[0], atol=1e-7, rtol=0.0)


def test_core_cache_is_deterministic_and_graph_splits_are_disjoint(tmp_path: Path) -> None:
    first = prepare_cyclecount_dataset(tmp_path, seed=31, tiny=True)
    second = prepare_cyclecount_dataset(tmp_path, seed=31, tiny=True)
    assert first.data_sha256 == second.data_sha256
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    split_sets = [set(ids) for ids in manifest["split_graph_ids"].values()]
    for index, left in enumerate(split_sets):
        for right in split_sets[index + 1 :]:
            assert left.isdisjoint(right)
    assert manifest["graph_split_before_chart_sampling"] is True


def test_optional_pyg_adapter_has_actionable_dependency_error(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch_geometric") is not None:
        pytest.skip("PyG is installed; download behavior is environment-specific")
    with pytest.raises(OptionalDatasetError, match="torch-geometric"):
        prepare_optional_pyg_dataset("csl", tmp_path, seed=1, tiny=True, allow_download=True)


def test_optional_pyg_adapter_requires_explicit_download_permission(
    tmp_path: Path,
) -> None:
    with pytest.raises(OptionalDatasetError, match="--allow-download"):
        prepare_optional_pyg_dataset("zinc", tmp_path, seed=1, tiny=True)


def test_dataset_seed_axes_route_to_their_declared_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    sentinel = object()

    def prepare_core(data_root: Path, *, seed: int, tiny: bool) -> object:
        calls.append(("core", seed))
        return sentinel

    def prepare_public(
        suite: str,
        data_root: Path,
        *,
        seed: int,
        tiny: bool,
        allow_download: bool,
    ) -> object:
        calls.append((suite, seed))
        return sentinel

    monkeypatch.setattr(tree_paper, "prepare_cyclecount_dataset", prepare_core)
    monkeypatch.setattr(tree_paper, "prepare_optional_pyg_dataset", prepare_public)
    axes = SeedAxes(data=11, split=13, chart=17, model=19)
    for suite in ("core", "csl", "zinc"):
        assert (
            tree_paper._prepare_dataset(
                suite,
                tmp_path,
                seed_axes=axes,
                tiny=True,
                allow_download=False,
            )
            is sentinel
        )
    assert calls == [("core", 11), ("csl", 13), ("zinc", 11)]


def _pyg_like_zinc_fixture() -> SimpleNamespace:
    # Directed arcs are deliberately not in canonical undirected order.
    arcs = (
        (2, 3, 3),
        (1, 3, 2),
        (0, 2, 1),
        (0, 1, 0),
        (3, 2, 3),
        (3, 1, 2),
        (2, 0, 1),
        (1, 0, 0),
        (1, 2, 1),
        (2, 1, 1),
    )
    return SimpleNamespace(
        num_nodes=4,
        x=torch.tensor([[3], [7], [2], [11]], dtype=torch.long),
        edge_index=torch.tensor(
            [[u for u, _, _ in arcs], [v for _, v, _ in arcs]], dtype=torch.long
        ),
        edge_attr=torch.tensor([[kind] for _, _, kind in arcs], dtype=torch.long),
        y=torch.tensor([0.375], dtype=torch.float32),
    )


def test_zinc_pyg_fixture_chemistry_is_lossless_and_cache_roundtrips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = zinc_record_from_pyg(
        _pyg_like_zinc_fixture(), graph_id="zinc-test-00000", split="test"
    )
    assert record.x == (3, 7, 2, 11)
    assert record.edges == ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
    assert record.edge_attr == (0, 1, 1, 2, 3)

    monkeypatch.setattr(
        "research.tree_augmentation.paper_data._prepare_zinc_records",
        lambda data_root, *, tiny: (record,),
    )
    prepared = prepare_optional_pyg_dataset(
        "zinc", tmp_path, seed=9, tiny=True, allow_download=True
    )
    loaded = prepare_optional_pyg_dataset("zinc", tmp_path, seed=9, tiny=True, allow_download=False)
    assert loaded.records == prepared.records == (record,)
    payload = json.loads(prepared.data_path.read_text(encoding="utf-8"))
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == 2
    assert payload["records"][0]["x"] == [3, 7, 2, 11]
    assert payload["records"][0]["edge_attr"] == [0, 1, 1, 2, 3]
    assert "canonical undirected edge" in manifest["categorical_feature_schema"]["edge_attr"]


def test_chemistry_is_chart_invariant_and_changes_model_input_and_prediction() -> None:
    record = zinc_record_from_pyg(
        _pyg_like_zinc_fixture(), graph_id="zinc-test-00000", split="train"
    )
    chart_views = build_chart_views(
        [record],
        chart_status="seen",
        count=2,
        methods=("bfs", "dfs"),
        roots=(0, 3),
        seed=13,
        require_distinct=True,
    )
    batch = collate_chart_views(chart_views)
    assert chart_views[0].tree_key != chart_views[1].tree_key
    assert not torch.equal(batch.basis[0], batch.basis[1])
    assert torch.equal(batch.edge_index[0], batch.edge_index[1])
    assert torch.equal(batch.node_categories[0], batch.node_categories[1])
    assert torch.equal(batch.edge_categories[0], batch.edge_categories[1])

    changed = replace(
        record,
        x=(4, *record.x[1:]),
        edge_attr=(*record.edge_attr[:-1], 0),
    )
    original_view = build_chart_views(
        [record],
        chart_status="seen",
        count=1,
        methods=("bfs",),
        roots=(0,),
        seed=17,
    )[0]
    changed_view = build_chart_views(
        [changed],
        chart_status="seen",
        count=1,
        methods=("bfs",),
        roots=(0,),
        seed=17,
    )[0]
    chemistry_batch = collate_chart_views([original_view, changed_view])
    assert torch.equal(chemistry_batch.basis[0], chemistry_batch.basis[1])
    assert not torch.equal(chemistry_batch.node_categories[0], chemistry_batch.node_categories[1])
    assert not torch.equal(chemistry_batch.edge_categories[0], chemistry_batch.edge_categories[1])
    torch.manual_seed(101)
    model = VariableBetaCycleEncoder(hidden_dim=12, output_dim=1).eval()
    prediction = model(chemistry_batch)
    assert not torch.allclose(prediction[0], prediction[1])


def test_tiny_cpu_core_cli_path_writes_2x2_results(tmp_path: Path) -> None:
    summary = run_suite(
        "core",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        requested_device="cpu",
        seed=43,
        tiny=True,
        prepare_only=False,
        amp_override=False,
        batch_size_override=4,
        pin_memory_override=False,
        non_blocking_override=False,
        workers=0,
        allow_download=False,
    )
    assert summary["runtime"]["workers"] == 0
    assert summary["seed_axes"] == {"data": 43, "split": 43, "chart": 43, "model": 43}
    assert "seed" not in summary
    assert summary["protocol"] == "cyclecount_graph_x_fresh_chart_family_2x2_v2"
    assert summary["comparison"]["paper_headline_eligible"] is False
    assert summary["comparison"]["projector_target_used"] is False
    expected = {
        "id_graph_fresh_chart_seen_family",
        "id_graph_fresh_chart_unseen_family",
        "ood_graph_fresh_chart_seen_family",
        "ood_graph_fresh_chart_unseen_family",
    }
    for model_name in ("fixed_bfs", "multi_chart"):
        quadrants = summary["models"][model_name]["quadrants"]
        assert set(quadrants) == expected
        assert all(
            np.isfinite(value) for metrics in quadrants.values() for value in metrics.values()
        )
    manifest = json.loads((tmp_path / "results" / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["seed_axes"] == summary["seed_axes"]
    assert "seed" not in manifest
    assert manifest["protocol"] == summary["protocol"]
    assert manifest["runtime"]["device"] == "cpu"
    assert manifest["runtime"]["amp_effective"] is False
    assert manifest["sampler_protocol"]["train_multi"] == [
        "bfs_random_root",
        "dfs_random_root",
    ]
    assert manifest["sampler_protocol"]["fresh_chart_unseen_family"] == ["wilson_ust"]
    assert manifest["sampler_protocol"]["exact_tree_overlap_between_families_allowed"] is True
    assert manifest["sampler_protocol"]["wilson_draws_conditioned_on_bfs_outputs"] is False
    assert summary["sampler_protocol"] == manifest["sampler_protocol"]
    assert set(summary["view_counts"]["fixed_train"]["sampler_counts"]) == {"bfs"}
    assert set(summary["view_counts"]["fixed_train"]["chart_status_counts"]) == {
        "train_fixed_bfs_family"
    }
    assert set(summary["view_counts"]["multi_train"]["sampler_counts"]) == {"bfs", "dfs"}
    assert set(summary["view_counts"]["multi_train"]["chart_status_counts"]) == {
        "train_multi_bfs_dfs_families"
    }
    for axis, stats in summary["view_counts"]["evaluation"].items():
        expected_sampler = "wilson_ust" if "unseen_family" in axis else "bfs"
        expected_status = (
            "fresh_chart_unseen_family" if "unseen_family" in axis else "fresh_chart_seen_family"
        )
        assert set(stats["sampler_counts"]) == {expected_sampler}
        assert set(stats["chart_status_counts"]) == {expected_status}
    assert set(summary["fresh_axis_exact_tree_overlap"]) == {"id_graph", "ood_graph"}

    repeated = run_suite(
        "core",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "repeated-results",
        requested_device="cpu",
        seed=999,
        data_seed=43,
        split_seed=43,
        chart_seed=43,
        model_seed=43,
        tiny=True,
        prepare_only=False,
        amp_override=False,
        batch_size_override=4,
        pin_memory_override=False,
        non_blocking_override=False,
        workers=0,
        allow_download=False,
    )
    assert repeated["models"] == summary["models"]
    assert repeated["comparison"] == summary["comparison"]


def test_prepare_only_all_attempts_every_suite_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "all"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper",
            "--suite",
            "all",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
            "--seed",
            "999",
            "--data-seed",
            "5",
            "--split-seed",
            "7",
            "--chart-seed",
            "11",
            "--model-seed",
            "13",
            "--tiny",
            "--prepare-only",
            "--workers",
            "0",
        ],
    )
    assert main() == 2
    aggregate = json.loads((output / "manifest.json").read_text("utf-8"))
    assert aggregate["status"] == "failed"
    assert aggregate["seed_axes"] == {"data": 5, "split": 7, "chart": 11, "model": 13}
    assert "seed" not in aggregate
    assert aggregate["suites"]["core"]["status"] == "prepared"
    assert aggregate["suites"]["csl"]["status"] == "failed"
    assert aggregate["suites"]["zinc"]["status"] == "failed"
    assert "--allow-download" in aggregate["suites"]["csl"]["error"]
    assert (output / "zinc" / "manifest.json").is_file()
````

# research/tree_augmentation/tests/test_tree_augmentation.py

````python
"""Tests for the standalone static Cycle-PE tree-augmentation track."""

from __future__ import annotations

import numpy as np
import pytest

from chartgat.algebra import chart_transition, incidence_matrix, validate_spanning_tree
from chartgat.graphs import make_connected_graph
from research.tree_augmentation.augmentation import (
    build_tree_chart,
    cycle_projector,
    cycle_projector_diagonal,
    ensure_full_cycle_budget,
    find_unseen_chart,
    lossless_transition_error,
    run_static_cycle_pe_probe,
    sample_tree_charts,
    transition_cocycle_error,
    transport_coordinates,
)


@pytest.fixture
def graph() -> tuple[int, list[tuple[int, int]]]:
    num_nodes = 9
    return num_nodes, make_connected_graph(num_nodes, extra_edges=5, seed=23)


def test_bfs_dfs_and_random_tree_sampling(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    B = incidence_matrix(num_nodes, edges)
    for method in ("bfs", "dfs", "random"):
        chart = build_tree_chart(num_nodes, edges, method=method, seed=31)
        validate_spanning_tree(B, chart.tree_edge_indices)
        assert chart.beta == len(edges) - num_nodes + 1
        assert np.allclose(B.T @ chart.basis, 0.0)


def test_full_beta_chart_transitions_are_lossless_and_unimodular(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    charts = sample_tree_charts(num_nodes, edges, random_count=4, random_seed_start=40)
    rng = np.random.default_rng(5)
    coordinates = rng.normal(size=(charts[0].beta, 3))
    physical = charts[0].basis @ coordinates

    for target in charts[1:]:
        target_coordinates = transport_coordinates(charts[0], target, coordinates)
        transition = chart_transition(charts[0].basis, target.basis)
        assert np.allclose(target.basis @ target_coordinates, physical, atol=1e-10)
        assert np.allclose(transition, np.rint(transition), atol=1e-10)
        assert abs(round(float(np.linalg.det(transition)))) == 1

    assert lossless_transition_error(charts, coordinates) < 1e-10


def test_chart_transition_cocycle_law(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    charts = sample_tree_charts(num_nodes, edges, random_count=3, random_seed_start=70)
    assert len(charts) >= 3
    assert transition_cocycle_error(charts) < 1e-10


def test_cycle_projector_is_chart_invariant(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    charts = sample_tree_charts(num_nodes, edges, random_count=5, random_seed_start=90)
    reference = cycle_projector(charts[0].basis)
    for chart in charts[1:]:
        assert np.allclose(cycle_projector(chart.basis), reference, atol=1e-10)
        assert np.allclose(cycle_projector_diagonal(chart.basis), np.diag(reference), atol=1e-10)


def test_lossy_cycle_budget_is_explicitly_disabled() -> None:
    assert ensure_full_cycle_budget(6) == 6
    assert ensure_full_cycle_budget(6, 6) == 6
    with pytest.raises(NotImplementedError, match="lossy extension"):
        ensure_full_cycle_budget(6, 5)


def test_static_probe_reports_fixed_multi_and_unseen_evaluation(
    graph: tuple[int, list[tuple[int, int]]],
) -> None:
    num_nodes, edges = graph
    training = sample_tree_charts(
        num_nodes,
        edges,
        random_count=5,
        random_seed_start=120,
    )
    unseen = find_unseen_chart(num_nodes, edges, training, seed_start=900)
    result = run_static_cycle_pe_probe(
        training,
        unseen,
        hidden_dim=16,
        epochs=12,
        learning_rate=0.01,
        seed=3,
    )

    assert result["num_training_charts"] == len(training)
    assert result["cycle_rank_beta"] == len(edges) - num_nodes + 1
    for key in (
        "fixed_train_mse",
        "multi_train_mse",
        "fixed_unseen_mse",
        "multi_unseen_mse",
        "projector_oracle_unseen_mse",
    ):
        assert np.isfinite(result[key])
        assert result[key] >= 0.0
    assert result["projector_oracle_unseen_mse"] < 1e-20
````

# scripts/aggregate_paper.py

````python
#!/usr/bin/env python3
"""Aggregate seed-aligned paper artifacts without mixing experimental axes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AGGREGATE_FILENAMES = {"metrics.json", "runtime.json", "summary.json"}
CONDITIONS = {
    "no_pe",
    "raw",
    "set",
    "projector",
    "isotropic",
    "edge_only",
    "gradient_only",
    "full",
    "full_flux_supervised",
    "full_joint",
    "flux_ls",
    "node_message_nnls",
    "oracle",
    "no_message_mlp",
    "gcn",
    "gat",
    "gine",
    "conductance_model",
    "fixed_bfs",
    "multi_chart",
}


@dataclass(frozen=True)
class AggregateMetricRule:
    """An explicit contract for one family of aggregate result fields."""

    name: str
    track: str
    artifact_pattern: re.Pattern[str]
    metric_pattern: re.Pattern[str]
    pairable: bool


def _metric_rule(
    name: str,
    track: str,
    artifact_pattern: str,
    metric_pattern: str,
    *,
    pairable: bool,
) -> AggregateMetricRule:
    return AggregateMetricRule(
        name=name,
        track=track,
        artifact_pattern=re.compile(artifact_pattern),
        metric_pattern=re.compile(metric_pattern),
        pairable=pairable,
    )


_CONDUCTANCE_PREDICTION_METRICS = (
    r"(?:graph_macro_flux_relative_l2|graph_macro_node_message_relative_l2|"
    r"graph_macro_next_state_relative_l2|graph_macro_log_conductance_rmse|"
    r"graph_macro_conductance_pearson|graph_macro_conductance_spearman|"
    r"graph_macro_observed_fit_relative_l2)"
)
_CONDUCTANCE_BASELINES = (
    r"(?:isotropic|edge_only|gradient_only|full|full_flux_supervised|full_joint|"
    r"flux_ls|node_message_nnls|oracle)"
)
_CYCLE_TEST_SPLITS = r"(?:id_test|size_ood|family_ood|test)"
_CYCLE_SUPERVISED_METRICS = (
    rf"{_CYCLE_TEST_SPLITS}\.(?:"
    r"macro_(?:normalized_)?mae|"
    r"levels\.(?:edge|node|graph)\.macro_(?:normalized_)?mae|"
    r"levels\.(?:edge|node|graph)\.targets\.[^.]+\."
    r"(?:mae|rmse|normalized_mae|graph_macro_mae|rounded_exact_accuracy))"
)
_TREE_EVALUATION_METRICS = (
    r"(?:mae|normalized_mae|rmse|graph_macro_mae|worst_chart_mae|"
    r"chart_prediction_std|rounded_exact_vector_accuracy|accuracy|graph_macro_accuracy|"
    r"worst_chart_accuracy|chart_probability_std|prediction_flip_rate)"
)

# This registry is intentionally closed.  Adding a numeric field to a result JSON
# does not make it a paper metric until a reviewer deliberately extends this
# schema.  Runtime, memory, parameter counts, configuration, sample counts, seed
# axes, and optimization histories therefore cannot leak into hypothesis tests.
PAPER_METRIC_SCHEMA_VERSION = 2
PAPER_METRIC_SCHEMA: tuple[AggregateMetricRule, ...] = (
    _metric_rule(
        "conductance.core.prediction",
        "conductance_gat",
        r"summary\.json",
        rf"results\.core\.s[1-4]\.baselines\.{_CONDUCTANCE_BASELINES}\."
        rf"(?:unseen_graph_test|seen_graph_new_excitation_test)\."
        rf"{_CONDUCTANCE_PREDICTION_METRICS}",
        pairable=True,
    ),
    _metric_rule(
        "conductance.core.rollout",
        "conductance_gat",
        r"summary\.json",
        rf"results\.core\.s3\.baselines\.{_CONDUCTANCE_BASELINES}\.rollout\."
        r"(?:horizon_[1-9][0-9]*_relative_l2|final_norm_over_initial|"
        r"dissipation_violation_fraction)",
        pairable=True,
    ),
    _metric_rule(
        "conductance.core.factorial",
        "conductance_gat",
        r"summary\.json",
        rf"results\.core\.s4\.factorial\.[0-9]+\.{_CONDUCTANCE_PREDICTION_METRICS}",
        # The current factorial JSON stores the baseline as a sibling string,
        # not in the numeric path.  Keep the measurements but do not manufacture
        # a paired comparison from opaque list indices.
        pairable=False,
    ),
    _metric_rule(
        "conductance.public.test",
        "conductance_gat",
        r"summary\.json",
        r"results\.public\.[^.]+\.baselines\."
        r"(?:no_message_mlp|gcn|gat|gine|conductance_model)\.test\.(?:macro_f1|roc_auc)",
        pairable=True,
    ),
    _metric_rule(
        "cycle.supervised.test",
        "cycle_pe",
        r"(?:core|zinc)/[^/]+/(?:no_pe|raw|set|projector)/metrics\.json",
        _CYCLE_SUPERVISED_METRICS,
        pairable=True,
    ),
    _metric_rule(
        "cycle.brec.official",
        "cycle_pe",
        r"brec/(?:no_pe|raw|set|projector)/metrics\.json",
        r"per_seed\.[0-9]+\.(?:Correct|Fail|Real_correct)",
        # BREC owns its internal ten-seed protocol; the outer model-seed axis is
        # explicitly disabled and no cross-variant paired test is defined here.
        pairable=False,
    ),
    _metric_rule(
        "cycle.brec.custom",
        "cycle_pe",
        r"brec/(?:no_pe|raw|set|projector)/metrics\.json",
        r"(?:success_rate|categories\.[^.]+\.success_rate)",
        pairable=True,
    ),
    _metric_rule(
        "tree.downstream.test",
        "tree_augmentation",
        r"summary\.json",
        rf"models\.(?:fixed_bfs|multi_chart)\.quadrants\.[^.]+\."
        rf"{_TREE_EVALUATION_METRICS}",
        pairable=True,
    ),
    _metric_rule(
        "tree.precomputed_improvement",
        "tree_augmentation",
        r"summary\.json",
        r"comparison\.quadrant_improvements\.[^.]+\."
        r"(?:mae_improvement_fixed_minus_multi|worst_chart_mae_improvement_fixed_minus_multi|"
        r"chart_std_improvement_fixed_minus_multi)",
        pairable=False,
    ),
)

# Efficiency observations are emitted as raw, seed-addressable rows in a
# separate table.  They are never bootstrapped or paired.  The allowlist is
# deliberately limited to elapsed time, peak accelerator memory, and active
# trainable parameter counts; epochs, batch size, workers, and seeds are not
# efficiency outcomes.
EFFICIENCY_METRIC_SCHEMA_VERSION = 1
EFFICIENCY_METRIC_SCHEMA: tuple[AggregateMetricRule, ...] = (
    _metric_rule(
        "conductance.runtime",
        "conductance_gat",
        r"summary\.json",
        r"runtime\.(?:elapsed_seconds|cuda_peak_allocated_bytes|cuda_peak_reserved_bytes)",
        pairable=False,
    ),
    _metric_rule(
        "conductance.active_parameters",
        "conductance_gat",
        r"summary\.json",
        r"results\.public\.[^.]+\.baselines\."
        r"(?:no_message_mlp|gcn|gat|gine|conductance_model)\.parameter_count",
        pairable=False,
    ),
    _metric_rule(
        "cycle.runtime",
        "cycle_pe",
        r"(?:core|zinc)/[^/]+/(?:no_pe|raw|set|projector)/runtime\.json",
        r"(?:total_train_evaluation_wall_seconds|peak_gpu_memory_bytes)",
        pairable=False,
    ),
    _metric_rule(
        "tree.runtime",
        "tree_augmentation",
        r"summary\.json",
        r"runtime\.(?:elapsed_seconds|peak_gpu_allocated_bytes|peak_gpu_reserved_bytes)",
        pairable=False,
    ),
)


def _flag(command: list[str], name: str) -> str | None:
    value: str | None = None
    for index, token in enumerate(command[:-1]):
        if token == name:
            value = command[index + 1]
    return value


def _integer_flag(command: list[str], *names: str, default: int | None = None) -> int | None:
    for name in names:
        value = _flag(command, name)
        if value is not None:
            return int(value)
    return default


def _factors(entry: dict[str, Any]) -> dict[str, int | None]:
    command = [str(value) for value in entry.get("command", [])]
    legacy = _integer_flag(command, "--seed")
    data_seed = _integer_flag(command, "--data-seed", default=legacy)
    model_seed = _integer_flag(command, "--model-seed", default=legacy)
    # Official BREC owns a separate ten-seed search protocol.  The outer
    # runner's placeholder model seed is deliberately not a BREC sample axis.
    if _flag(command, "--suite") == "brec" and _flag(command, "--brec-protocol") == "official":
        model_seed = None
    return {
        "model_seed": model_seed,
        "data_seed": data_seed,
        "split_seed": _integer_flag(command, "--split-seed", default=data_seed),
        "chart_seed": _integer_flag(command, "--chart-seed", default=data_seed),
    }


def _flatten_numeric(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, bool) or value is None:
        return rows
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            rows.append((".".join(prefix) or "value", numeric))
        return rows
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_flatten_numeric(item, (*prefix, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_flatten_numeric(item, (*prefix, str(index))))
    return rows


def _select_metric_rule(
    schema: tuple[AggregateMetricRule, ...],
    track: str,
    artifact: str,
    metric: str,
    *,
    table: str,
) -> AggregateMetricRule | None:
    matches = [
        rule
        for rule in schema
        if rule.track == track
        and rule.artifact_pattern.fullmatch(artifact)
        and rule.metric_pattern.fullmatch(metric)
    ]
    if len(matches) > 1:
        names = ", ".join(rule.name for rule in matches)
        raise RuntimeError(
            f"ambiguous {table} metric schema for {track}:{artifact}:{metric}: {names}"
        )
    return matches[0] if matches else None


def _nested_json_value(payload: Any, dotted_path: str) -> Any:
    value = payload
    for token in dotted_path.split("."):
        if not isinstance(value, dict) or token not in value:
            return None
        value = value[token]
    return value


def _efficiency_rule_is_applicable(rule: AggregateMetricRule, payload: Any, metric: str) -> bool:
    if rule.name != "conductance.active_parameters":
        return True
    parent = metric.rsplit(".", 1)[0]
    return (
        _nested_json_value(payload, f"{parent}.parameter_count_policy")
        == "trainable_active_parameters_only"
    )


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summary(values: list[float], *, key: str, bootstrap_samples: int) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    if len(values) == 1 or bootstrap_samples == 0:
        low = high = mean
    else:
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        generator = random.Random(seed)
        means = sorted(
            statistics.fmean(generator.choice(values) for _ in values)
            for _ in range(bootstrap_samples)
        )
        low = _quantile(means, 0.025)
        high = _quantile(means, 0.975)
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def _condition_template(artifact: str, metric: str) -> tuple[str, str, str] | None:
    artifact_parts = artifact.replace("\\", "/").split("/")
    metric_parts = metric.split(".")
    occurrences: list[tuple[str, int, str]] = []
    for index, token in enumerate(artifact_parts):
        if token in CONDITIONS:
            occurrences.append(("artifact", index, token))
    for index, token in enumerate(metric_parts):
        if token in CONDITIONS:
            occurrences.append(("metric", index, token))
    if len(occurrences) != 1:
        return None
    location, index, condition = occurrences[0]
    if location == "artifact":
        artifact_parts[index] = "{condition}"
    else:
        metric_parts[index] = "{condition}"
    return "/".join(artifact_parts), ".".join(metric_parts), condition


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    fieldnames = sorted({key for row in rows for key in row})
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def aggregate_manifest(
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    """Aggregate explicitly registered paper metrics from completed artifacts.

    Data, split, and chart seeds are grouping keys.  Only model seeds are averaged,
    so changing a dataset or chart does not silently inflate a model-seed standard
    deviation.  Legacy children that expose only ``--seed`` remain readable, but
    all four axes then intentionally resolve to that same value in the audit output.

    Numeric fields that are not in :data:`PAPER_METRIC_SCHEMA` or
    :data:`EFFICIENCY_METRIC_SCHEMA` are counted for the audit trail and otherwise
    ignored.  Explicit runtime, peak-memory, and active-parameter observations go
    to a raw efficiency table; they never enter bootstrap summaries or paired tests.
    Configuration, sample-count, seed, and optimizer-history numbers enter neither
    table.
    """

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else manifest_path.parent / "aggregate"
    )
    samples: list[dict[str, Any]] = []
    efficiency_samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    numeric_fields_seen = 0
    ignored_numeric_fields = 0
    for entry in manifest.get("commands", []):
        name = str(entry.get("name", "unknown"))
        if name == "gpu_preflight":
            continue
        command = [str(value) for value in entry.get("command", [])]
        track = name.split(":", 1)[0]
        suite = _flag(command, "--suite") or "unknown"
        factors = _factors(entry)
        return_code = int(entry.get("returncode", 1))
        artifact_errors = list(entry.get("artifact_errors") or [])
        output_value = entry.get("output")
        output_path = Path(output_value).expanduser().resolve() if output_value else None
        if return_code != 0 or artifact_errors or output_path is None or not output_path.exists():
            log_text = ""
            log_value = entry.get("log")
            if log_value:
                try:
                    log_text = Path(str(log_value)).read_text(encoding="utf-8", errors="replace")[
                        -100_000:
                    ]
                except OSError:
                    pass
            error_text = " | ".join(str(error) for error in artifact_errors)
            searchable_error = f"{error_text}\n{log_text}".casefold()
            failures.append(
                {
                    "command": name,
                    "track": track,
                    "suite": suite,
                    "returncode": return_code,
                    "artifact_errors": error_text,
                    "oom": any(
                        marker in searchable_error
                        for marker in (
                            "out of memory",
                            "outofmemoryerror",
                            "cublas_status_alloc_failed",
                            "cuda error: memory allocation",
                        )
                    ),
                    **factors,
                }
            )
            continue
        candidates = (
            [output_path]
            if output_path.is_file() and output_path.name in AGGREGATE_FILENAMES
            else sorted(
                path for path in output_path.rglob("*.json") if path.name in AGGREGATE_FILENAMES
            )
        )
        for artifact_path in candidates:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact = (
                artifact_path.name
                if output_path.is_file()
                else str(artifact_path.relative_to(output_path)).replace("\\", "/")
            )
            numeric_fields = _flatten_numeric(payload)
            numeric_fields_seen += len(numeric_fields)
            for metric, value in numeric_fields:
                paper_rule = _select_metric_rule(
                    PAPER_METRIC_SCHEMA, track, artifact, metric, table="paper"
                )
                efficiency_rule = _select_metric_rule(
                    EFFICIENCY_METRIC_SCHEMA, track, artifact, metric, table="efficiency"
                )
                if efficiency_rule is not None and not _efficiency_rule_is_applicable(
                    efficiency_rule, payload, metric
                ):
                    efficiency_rule = None
                if paper_rule is not None and efficiency_rule is not None:
                    raise RuntimeError(
                        f"metric belongs to paper and efficiency schemas: "
                        f"{track}:{artifact}:{metric}"
                    )
                if paper_rule is None and efficiency_rule is None:
                    ignored_numeric_fields += 1
                    continue
                common = {
                    "command": name,
                    "track": track,
                    "suite": suite,
                    "artifact": artifact,
                    "artifact_path": str(artifact_path),
                    "metric": metric,
                    "value": value,
                    **factors,
                }
                if paper_rule is not None:
                    samples.append(
                        {
                            **common,
                            "metric_rule": paper_rule.name,
                            "pairable": paper_rule.pairable,
                        }
                    )
                else:
                    assert efficiency_rule is not None
                    efficiency_samples.append(
                        {
                            **common,
                            "metric_rule": efficiency_rule.name,
                        }
                    )

    grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        key = (
            row["track"],
            row["suite"],
            row["artifact"],
            row["metric"],
            row["metric_rule"],
            row["data_seed"],
            row["split_seed"],
            row["chart_seed"],
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        track, suite, artifact, metric, metric_rule, data_seed, split_seed, chart_seed = key
        values = [float(row["value"]) for row in rows]
        model_seeds = sorted(
            {int(row["model_seed"]) for row in rows if row["model_seed"] is not None}
        )
        identity = "|".join(str(value) for value in key)
        summaries.append(
            {
                "track": track,
                "suite": suite,
                "artifact": artifact,
                "metric": metric,
                "metric_rule": metric_rule,
                "data_seed": data_seed,
                "split_seed": split_seed,
                "chart_seed": chart_seed,
                "model_seeds": ",".join(str(seed) for seed in model_seeds),
                **_summary(values, key=identity, bootstrap_samples=bootstrap_samples),
            }
        )

    paired_samples: defaultdict[tuple[Any, ...], dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in samples:
        if not row["pairable"]:
            continue
        template = _condition_template(str(row["artifact"]), str(row["metric"]))
        if template is None or row["model_seed"] is None:
            continue
        artifact_template, metric_template, condition = template
        key = (
            row["track"],
            row["suite"],
            artifact_template,
            metric_template,
            row["metric_rule"],
            row["data_seed"],
            row["split_seed"],
            row["chart_seed"],
        )
        paired_samples[key][condition][int(row["model_seed"])] = float(row["value"])
    paired: list[dict[str, Any]] = []
    for key, by_condition in sorted(paired_samples.items(), key=lambda item: str(item[0])):
        conditions = sorted(by_condition)
        for left_index, left in enumerate(conditions):
            for right in conditions[left_index + 1 :]:
                common = sorted(set(by_condition[left]) & set(by_condition[right]))
                if not common:
                    continue
                differences = [
                    by_condition[right][seed] - by_condition[left][seed] for seed in common
                ]
                identity = "|".join(str(value) for value in (*key, left, right))
                (
                    track,
                    suite,
                    artifact_template,
                    metric_template,
                    metric_rule,
                    data_seed,
                    split_seed,
                    chart_seed,
                ) = key
                difference_summary = _summary(
                    differences,
                    key=identity,
                    bootstrap_samples=bootstrap_samples,
                )
                paired.append(
                    {
                        "track": track,
                        "suite": suite,
                        "artifact_template": artifact_template,
                        "metric_template": metric_template,
                        "metric_rule": metric_rule,
                        "condition_left": left,
                        "condition_right": right,
                        "difference_definition": "right_minus_left",
                        "data_seed": data_seed,
                        "split_seed": split_seed,
                        "chart_seed": chart_seed,
                        "model_seeds": ",".join(str(seed) for seed in common),
                        **difference_summary,
                        "effect_size_name": "paired_cohens_dz",
                        "effect_size": (
                            difference_summary["mean"] / difference_summary["sample_std"]
                            if len(differences) > 1
                            and difference_summary["sample_std"]
                            > max(1e-12, abs(difference_summary["mean"]) * 1e-12)
                            else None
                        ),
                    }
                )

    payload = {
        "schema_version": 2,
        "paper_metric_schema_version": PAPER_METRIC_SCHEMA_VERSION,
        "efficiency_metric_schema_version": EFFICIENCY_METRIC_SCHEMA_VERSION,
        "paper_metric_rules": [rule.name for rule in PAPER_METRIC_SCHEMA],
        "efficiency_metric_rules": [rule.name for rule in EFFICIENCY_METRIC_SCHEMA],
        "source_manifest": str(manifest_path),
        "source_run_id": manifest.get("run_id"),
        "source_status": manifest.get("status"),
        "seed_policy": (
            "group by data/split/chart seed; summarize and pair only aligned model seeds"
        ),
        "bootstrap_samples": bootstrap_samples,
        "numeric_fields_seen": numeric_fields_seen,
        "ignored_numeric_fields": ignored_numeric_fields,
        "sample_rows": len(samples),
        "efficiency_rows": len(efficiency_samples),
        "metric_groups": len(summaries),
        "paired_groups": len(paired),
        "failed_commands": len(failures),
        "oom_failures": sum(bool(row["oom"]) for row in failures),
        "files": {
            "samples": "samples.csv",
            "metrics": "metrics.csv",
            "paired": "paired.csv",
            "efficiency": "efficiency.csv",
            "failures": "failures.csv",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "samples.csv", samples)
    _write_csv(output_dir / "metrics.csv", summaries)
    _write_csv(output_dir / "paired.csv", paired)
    _write_csv(output_dir / "efficiency.csv", efficiency_samples)
    _write_csv(output_dir / "failures.csv", failures)
    _atomic_text(
        output_dir / "aggregate.json",
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative")
    payload = aggregate_manifest(
        args.manifest,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

# scripts/check_datasets.py

````python
#!/usr/bin/env python3
"""Validate code readiness and separately report dataset-cache availability."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheValidationError,
    CacheWrongRequestError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS = ("conductance_gat", "cycle_pe", "tree_augmentation")
REGISTRY_VERSION = 2
REQUIRED_ENTRY_FIELDS = {
    "id",
    "name",
    "tier",
    "status",
    "data_policy",
    "source_url",
    "task",
    "split",
    "metrics",
    "claim",
    "adapter",
    "leakage_guard",
}
ALLOWED_TIERS = {"smoke", "paper_core", "conditional", "optional"}
# ``status`` is code readiness. Dataset optionality belongs in ``tier``.
ALLOWED_STATUSES = {"implemented", "planned", "blocked"}
ALLOWED_DATA_POLICIES = {"generated", "download", "manual", "none"}


def load_registry(track: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "research" / track / "datasets.yaml"
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict):
        raise ValueError(f"{path}: registry root must be a mapping")
    registry["_path"] = str(path)
    return registry


def _load_python_reference(reference: str) -> Any:
    """Resolve and return a dotted module attribute."""

    pieces = reference.split(".")
    for stop in range(len(pieces), 0, -1):
        module_name = ".".join(pieces[:stop])
        try:
            specification = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            specification = None
        if specification is None:
            continue
        attributes = pieces[stop:]
        if not attributes:
            return importlib.import_module(module_name)
        try:
            value: Any = importlib.import_module(module_name)
            for attribute in attributes:
                value = getattr(value, attribute)
        except (ImportError, AttributeError, ModuleNotFoundError) as error:
            raise ImportError(f"cannot resolve Python reference {reference!r}: {error}") from error
        return value
    raise ImportError(f"cannot find Python module for reference {reference!r}")


def _resolve_python_reference(reference: str) -> str | None:
    """Return an error when a dotted module/callable cannot be resolved."""

    try:
        _load_python_reference(reference)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as error:
        return str(error)
    return None


def _validate_adapter(adapter: Any) -> str | None:
    if not isinstance(adapter, str) or not adapter.strip():
        return "adapter must be a non-empty string"
    normalized = adapter.strip()
    if normalized.lower().startswith(("planned", "requires")):
        return f"implemented adapter cannot be prose: {adapter!r}"
    if normalized.startswith("python -m "):
        try:
            tokens = shlex.split(normalized)
        except ValueError as error:
            return f"invalid adapter command: {error}"
        if len(tokens) < 3 or tokens[:2] != ["python", "-m"]:
            return f"invalid Python module command {adapter!r}"
        error = _resolve_python_reference(tokens[2])
        if error is not None:
            return error
        if "--suite" in tokens:
            suite_index = tokens.index("--suite")
            if suite_index + 1 >= len(tokens) or tokens[suite_index + 1].startswith("-"):
                return f"adapter command has no --suite value: {adapter!r}"
        return None
    if any(character.isspace() for character in normalized):
        return f"adapter is neither a dotted reference nor python -m command: {adapter!r}"
    return _resolve_python_reference(normalized)


def _validate_source(source: Any) -> str | None:
    if not isinstance(source, str) or not source.strip():
        return "source_url must be a non-empty string"
    if source.startswith("generated://"):
        module_name = source.removeprefix("generated://").split("/", 1)[0]
        error = _resolve_python_reference(module_name)
        return None if error is None else f"invalid generated source: {error}"
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return None
    return f"source_url must be generated:// or an HTTP(S) URL: {source!r}"


def _validate_cache_glob(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "cache_glob must be a non-empty relative string"
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return "cache_glob must remain under --data-root"
    return None


def validate_registry(track: str, registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("registry_version") != REGISTRY_VERSION:
        errors.append(f"{track}: registry_version must be {REGISTRY_VERSION}")
    if registry.get("track") != track:
        errors.append(f"{track}: registry track field does not match")
    if not isinstance(registry.get("paper_suite_complete"), bool):
        errors.append(f"{track}: paper_suite_complete must be boolean")
    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return [*errors, f"{track}: datasets must be a non-empty list"]

    identifiers: set[str] = set()
    for index, entry in enumerate(datasets):
        label = f"{track}.datasets[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry must be a mapping")
            continue
        missing = sorted(REQUIRED_ENTRY_FIELDS - entry.keys())
        if missing:
            errors.append(f"{label}: missing fields {missing}")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"{label}: id must be a non-empty string")
        elif identifier in identifiers:
            errors.append(f"{label}: duplicate id {identifier!r}")
        else:
            identifiers.add(identifier)
        if entry.get("tier") not in ALLOWED_TIERS:
            errors.append(f"{label}: invalid tier {entry.get('tier')!r}")
        status = entry.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{label}: invalid status {status!r}")
        data_policy = entry.get("data_policy")
        if data_policy not in ALLOWED_DATA_POLICIES:
            errors.append(f"{label}: invalid data_policy {data_policy!r}")
        if status == "implemented" and data_policy == "none":
            errors.append(f"{label}: implemented code cannot use data_policy 'none'")
        metrics = entry.get("metrics")
        if (
            not isinstance(metrics, list)
            or not metrics
            or not all(isinstance(metric, str) and metric for metric in metrics)
        ):
            errors.append(f"{label}: metrics must be a non-empty string list")
        if "cache_glob" in entry:
            cache_error = _validate_cache_glob(entry["cache_glob"])
            if cache_error is not None:
                errors.append(f"{label}: {cache_error}")
        source_error = _validate_source(entry.get("source_url"))
        if source_error is not None:
            errors.append(f"{label}: {source_error}")
        if status == "implemented":
            adapter_error = _validate_adapter(entry.get("adapter"))
            if adapter_error is not None:
                errors.append(f"{label}: {adapter_error}")
        if entry.get("tier") == "paper_core":
            validator = entry.get("validator")
            if not isinstance(validator, str) or not validator.strip():
                errors.append(f"{label}: paper_core entry requires a validator")
            else:
                validator_error = _resolve_python_reference(validator)
                if validator_error is not None:
                    errors.append(f"{label}: invalid cache validator: {validator_error}")
                else:
                    resolved = _load_python_reference(validator)
                    if not callable(resolved):
                        errors.append(f"{label}: cache validator must be callable")

    smoke = [entry for entry in datasets if entry.get("tier") == "smoke"]
    if not smoke or any(entry.get("status") != "implemented" for entry in smoke):
        errors.append(f"{track}: every smoke entry must be implemented")
    paper_core = [entry for entry in datasets if entry.get("tier") == "paper_core"]
    if not paper_core:
        errors.append(f"{track}: at least one paper_core dataset is required")
    else:
        computed_complete = all(entry.get("status") == "implemented" for entry in paper_core)
        if registry.get("paper_suite_complete") is not computed_complete:
            errors.append(
                f"{track}: paper_suite_complete must be {str(computed_complete).lower()} "
                "because it is derived only from paper_core code status"
            )
    return errors


def _validate_cache(
    entry: dict[str, Any],
    data_root: Path | None,
    *,
    seeds: tuple[int, ...] | None = None,
    data_seeds: tuple[int, ...] | None = None,
    split_seeds: tuple[int, ...] | None = None,
    tiny: bool,
) -> dict[str, Any]:
    resolved_data_seeds = data_seeds if data_seeds is not None else seeds or (0,)
    resolved_split_seeds = split_seeds if split_seeds is not None else resolved_data_seeds
    if entry.get("cache_glob") is None:
        return {"cache_status": "not_applicable", "cache_detail": None}
    if data_root is None:
        return {"cache_status": "not_checked", "cache_detail": None}
    validator_reference = entry.get("validator")
    if not isinstance(validator_reference, str):
        return {
            "cache_status": "incomplete",
            "cache_detail": "registry entry has no read-only cache validator",
        }
    try:
        validator = _load_python_reference(validator_reference)
        metadata = validator(
            entry["id"],
            data_root,
            data_seeds=resolved_data_seeds,
            split_seeds=resolved_split_seeds,
            tiny=tiny,
        )
    except FileNotFoundError as error:
        return {"cache_status": "missing", "cache_detail": str(error)}
    except CacheIncompleteError as error:
        return {"cache_status": "incomplete", "cache_detail": str(error)}
    except CacheWrongRequestError as error:
        return {"cache_status": "wrong_request", "cache_detail": str(error)}
    except CacheCorruptError as error:
        return {"cache_status": "corrupt", "cache_detail": str(error)}
    except CacheValidationError as error:
        return {"cache_status": "corrupt", "cache_detail": str(error)}
    except (ImportError, ModuleNotFoundError) as error:
        return {
            "cache_status": "incomplete",
            "cache_detail": f"validation dependency unavailable: {error}",
        }
    except Exception as error:  # fail closed on an unexpected parser/validator error
        return {
            "cache_status": "corrupt",
            "cache_detail": f"{type(error).__name__}: {error}",
        }
    return {"cache_status": "valid", "cache_detail": metadata}


def readiness(
    registries: dict[str, dict[str, Any]],
    profile: str,
    *,
    data_root: Path | None = None,
    data_seeds: tuple[int, ...] = (0,),
    split_seeds: tuple[int, ...] | None = None,
    tiny: bool = False,
) -> list[dict[str, Any]]:
    tier = "smoke" if profile == "smoke" else "paper_core"
    rows: list[dict[str, Any]] = []
    validation_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for track, registry in registries.items():
        for entry in registry["datasets"]:
            if entry["tier"] == tier:
                validation_key = (
                    str(entry.get("validator", "")),
                    str(entry.get("cache_glob", entry["id"])),
                )
                cache_result = validation_cache.get(validation_key)
                if cache_result is None:
                    cache_result = _validate_cache(
                        entry,
                        data_root,
                        data_seeds=data_seeds,
                        split_seeds=split_seeds,
                        tiny=tiny,
                    )
                    validation_cache[validation_key] = cache_result
                rows.append(
                    {
                        "track": track,
                        "id": entry["id"],
                        "tier": entry["tier"],
                        "status": entry["status"],
                        "code_ready": entry["status"] == "implemented",
                        "data_policy": entry["data_policy"],
                        **cache_result,
                    }
                )
    return rows


def _parse_seeds(parser: argparse.ArgumentParser, value: str, option: str) -> tuple[int, ...]:
    try:
        seeds = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError:
        parser.error(f"{option} must be a comma-separated list of integers")
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error(f"{option} must contain at least one non-negative integer")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--data-seeds",
        "--seeds",
        dest="data_seeds",
        default="0",
        help="comma-separated generated-data/cache seeds; --seeds is a compatibility alias",
    )
    parser.add_argument(
        "--split-seeds",
        help="comma-separated split/cache seeds; defaults to --data-seeds",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="validate tiny/fixture caches instead of full paper caches",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="require every selected cache to pass its read-only validator (requires --data-root)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.require_cache and args.data_root is None:
        parser.error("--require-cache requires --data-root")
    data_seeds = _parse_seeds(parser, args.data_seeds, "--data-seeds")
    split_seeds = (
        data_seeds
        if args.split_seeds is None
        else _parse_seeds(parser, args.split_seeds, "--split-seeds")
    )

    registries = {track: load_registry(track) for track in TRACKS}
    errors = [
        error
        for track, registry in registries.items()
        for error in validate_registry(track, registry)
    ]
    data_root = args.data_root.expanduser().resolve() if args.data_root is not None else None
    rows = readiness(
        registries,
        args.profile,
        data_root=data_root,
        data_seeds=data_seeds,
        split_seeds=split_seeds,
        tiny=bool(args.tiny),
    )
    code_ready = not errors and all(row["code_ready"] for row in rows)
    cache_ready: bool | None = None
    if data_root is not None:
        cache_ready = all(row["cache_status"] in {"valid", "not_applicable"} for row in rows)
    ready = code_ready and (cache_ready is True if args.require_cache else True)
    payload = {
        "profile": args.profile,
        "ready": ready,
        "code_ready": code_ready,
        "require_cache": bool(args.require_cache),
        "cache_checked": data_root is not None,
        "cached_data_ready": cache_ready,
        "cache_validation": "content-and-request" if data_root is not None else "not_checked",
        "requested_seeds": list(data_seeds),
        "requested_seed_axes": {
            "data": list(data_seeds),
            "split": list(split_seeds),
        },
        "tiny": bool(args.tiny),
        "paper_benchmark_suite_complete": not errors
        and all(registry["paper_suite_complete"] for registry in registries.values()),
        "rows": rows,
        "errors": errors,
    }

    if args.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"dataset profile: {args.profile}")
        for row in rows:
            print(
                f"  {row['track']:18} {row['id']:36} "
                f"code={row['status']:11} cache={row['cache_status']}"
            )
        for error in errors:
            print(f"ERROR: {error}")
        if data_root is not None:
            print("CACHED DATA READY" if cache_ready else "CACHED DATA INCOMPLETE")
        print("READY" if ready else "NOT READY")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
````

# scripts/conda_env.sh

````bash
#!/usr/bin/env bash
# Sourced by the Bash entrypoints after they set project_root.
# Never create an environment or fall back to a PATH/system Python here.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active Conda environment. Create and activate a dedicated environment first:" >&2
    echo "  conda create -n new-gat python=3.11 pip -y" >&2
    echo "  conda activate new-gat" >&2
    exit 2
fi

environment_python="${CONDA_PREFIX%/}/bin/python"
if [[ ! -x "${environment_python}" ]]; then
    echo "The active Conda environment has no executable Python: ${environment_python}" >&2
    echo "Activate the dedicated new-gat environment before running this script." >&2
    exit 2
fi

if ! "${environment_python}" "${project_root}/scripts/verify_conda_env.py"; then
    exit 2
fi
````

# scripts/generate_code_summary.py

````python
#!/usr/bin/env python3
"""Generate the reviewable ``# path`` + exact source ``code_summary.md`` snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "code_summary.md"

SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".sh", ".ps1"}
EXCLUDED_PARTS = {
    ".git",
    ".agents",
    ".codex",
    ".matplotlib",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "results",
    "runs",
}
LANGUAGES = {
    ".py": "python",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".ps1": "powershell",
    ".txt": "text",
}


def _excluded(path: Path, *, root: Path) -> bool:
    relative = path.relative_to(root)
    return any(
        part in EXCLUDED_PARTS or part.startswith(".venv") or part.endswith(".egg-info")
        for part in relative.parts
    )


def _is_source(path: Path, *, root: Path) -> bool:
    if not path.is_file() or _excluded(path, root=root):
        return False
    if path.name == ".gitignore":
        return True
    if path.suffix in SOURCE_SUFFIXES:
        return True
    return path.suffix == ".txt" and path.name.startswith(("requirements", "constraints"))


def discover_sources(root: Path = PROJECT_ROOT) -> list[Path]:
    """Return the deterministic set of human-authored code/configuration sources."""

    return sorted(
        (path for path in root.rglob("*") if _is_source(path, root=root)),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def render_summary(root: Path = PROJECT_ROOT) -> tuple[str, list[str]]:
    """Render sources with normalized LF separators and no content omissions."""

    sections: list[str] = []
    relative_paths: list[str] = []
    for path in discover_sources(root):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if source.endswith("\n"):
            source = source[:-1]
        language = LANGUAGES.get(path.suffix, "text")
        sections.append(f"# {relative}\n\n````{language}\n{source}\n````")
        relative_paths.append(relative)
    return "\n\n".join(sections) + "\n", relative_paths


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _report(content: str, sources: list[str], *, status: str) -> dict[str, object]:
    encoded = content.encode("utf-8")
    return {
        "status": status,
        "output": str(OUTPUT_PATH),
        "source_files": len(sources),
        "bytes": len(encoded),
        "lines": len(content.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest().upper(),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if code_summary.md does not exactly match the current selected sources",
    )
    parser.add_argument("--json", action="store_true", help="include the selected source list")
    args = parser.parse_args()

    content, sources = render_summary()
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else None
        matches = current == content
        report = _report(content, sources, status="current" if matches else "stale")
        if not args.json:
            report.pop("sources")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if matches else 1

    _atomic_write(OUTPUT_PATH, content)
    report = _report(content, sources, status="written")
    if not args.json:
        report.pop("sources")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

# scripts/gpu_preflight.py

````python
#!/usr/bin/env python3
"""Fail-fast CUDA and paper-dependency validation for a Linux experiment host."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as nnf


class PreflightError(RuntimeError):
    """A server cannot safely run the requested experiment profile."""


PAPER_IMPORTS = {
    "networkx": "networkx",
    "numpy": "numpy",
    "ogb": "ogb",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "torch-geometric": "torch_geometric",
    "PyYAML": "yaml",
}

PROFILE_NAMES = (
    "conductance",
    "cycle-projector",
    "tree-chart",
    "brec",
    "public-pyg",
)
CYCLE_VARIANTS = ("no_pe", "raw", "set", "projector")
BREC_PROTOCOLS = ("official", "custom")


@dataclass(frozen=True)
class ProfileConfig:
    """Synthetic envelope used to exercise one production training path."""

    batch_size: int = 32
    brec_batch_size: int = 16
    nodes_per_graph: int = 64
    edges_per_graph: int = 128
    cycle_rank: int = 64
    amp: bool = True
    cycle_variants: tuple[str, ...] = ("projector",)
    brec_protocol: str = "official"
    brec_amp: bool = False

    def validate(self) -> None:
        values = {
            "batch_size": self.batch_size,
            "brec_batch_size": self.brec_batch_size,
            "nodes_per_graph": self.nodes_per_graph,
            "edges_per_graph": self.edges_per_graph,
            "cycle_rank": self.cycle_rank,
        }
        invalid = [name for name, value in values.items() if value < 1]
        if invalid:
            raise PreflightError("profile dimensions must be positive: " + ", ".join(invalid))
        if self.nodes_per_graph < 2:
            raise PreflightError("--nodes-per-graph must be at least 2")
        if self.cycle_rank > self.edges_per_graph:
            raise PreflightError("--cycle-rank cannot exceed --edges-per-graph")
        if not self.cycle_variants or len(set(self.cycle_variants)) != len(self.cycle_variants):
            raise PreflightError("--cycle-variants must be non-empty and unique")
        unknown_variants = sorted(set(self.cycle_variants) - set(CYCLE_VARIANTS))
        if unknown_variants:
            raise PreflightError(f"unknown cycle variants: {unknown_variants}")
        if self.brec_protocol not in BREC_PROTOCOLS:
            raise PreflightError(f"--brec-protocol must be one of {BREC_PROTOCOLS}")
        if self.brec_protocol == "official":
            if self.brec_batch_size != 16:
                raise PreflightError("official BREC profile requires --brec-batch-size 16")
            if self.brec_amp:
                raise PreflightError("official BREC profile requires --no-brec-amp")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"torch": torch.__version__}
    for distribution in PAPER_IMPORTS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _missing_paper_dependencies() -> list[str]:
    return sorted(
        distribution
        for distribution, module in PAPER_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    )


def _paper_dependency_import_errors() -> dict[str, str]:
    """Import paper packages so an ABI-broken environment fails before training."""

    errors: dict[str, str] = {}
    for distribution, module in PAPER_IMPORTS.items():
        if importlib.util.find_spec(module) is None:
            errors[distribution] = "module not installed"
            continue
        try:
            importlib.import_module(module)
        except Exception as error:  # dependency imports can raise ABI/runtime errors
            errors[distribution] = f"{type(error).__name__}: {error}"
    return errors


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "rows": [], "error": "nvidia-smi not found"}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "available": completed.returncode == 0,
        "rows": rows,
        "error": completed.stderr.strip() or None,
    }


def _resolve_device(requested: str, *, allow_cpu: bool) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cpu":
        if not allow_cpu:
            raise PreflightError(
                "paper execution requires CUDA; pass --allow-cpu only for tiny local tests"
            )
        return device
    if device.type != "cuda":
        raise PreflightError(f"unsupported accelerator request: {requested!r}")
    if not torch.cuda.is_available():
        raise PreflightError(
            "torch.cuda.is_available() is false; install a CUDA PyTorch wheel and expose the GPU"
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        visible_count = torch.cuda.device_count()
        raise PreflightError(
            f"CUDA device index {index} is invalid; visible device count is {visible_count}"
        )
    return torch.device("cuda", index)


def _effective_profile_config(config: ProfileConfig, device: torch.device) -> ProfileConfig:
    """Keep the CPU path a code smoke while preserving exact requested CUDA sizes."""

    if device.type == "cuda":
        return config
    edges = min(config.edges_per_graph, 24)
    return ProfileConfig(
        batch_size=min(config.batch_size, 2),
        brec_batch_size=(
            4 if config.brec_protocol == "official" else min(config.brec_batch_size, 4)
        ),
        nodes_per_graph=min(config.nodes_per_graph, 12),
        edges_per_graph=edges,
        cycle_rank=min(config.cycle_rank, edges, 6),
        amp=config.amp,
        cycle_variants=config.cycle_variants,
        brec_protocol=config.brec_protocol,
        brec_amp=config.brec_amp,
    )


def _base_edges(num_nodes: int, num_edges: int) -> Tensor:
    columns = torch.arange(num_edges, dtype=torch.long)
    tail = columns.remainder(num_nodes)
    hop = 1 + torch.div(columns, num_nodes, rounding_mode="floor").remainder(num_nodes - 1)
    head = (tail + hop).remainder(num_nodes)
    return torch.stack((tail, head))


def _packed_edges(batch_size: int, num_nodes: int, num_edges: int) -> Tensor:
    base = _base_edges(num_nodes, num_edges)
    offsets = torch.arange(batch_size, dtype=torch.long) * num_nodes
    return (base.unsqueeze(0) + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count the logical tensor payload requested by a synthetic host-side batch."""

    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, visited) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, visited) for item in value)
    if hasattr(value, "__dict__"):
        return _tensor_bytes(vars(value), visited)
    return 0


def _current_rss_bytes() -> int | None:
    """Return Linux current RSS without adding a preflight dependency."""

    status = Path("/proc/self/statm")
    if not status.is_file():
        return None
    try:
        resident_pages = int(status.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows developer host
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value if platform.system() == "Darwin" else value * 1024


def _assert_finite(name: str, *values: Tensor | None) -> None:
    if any(value is None or not bool(torch.isfinite(value).all()) for value in values):
        raise PreflightError(f"{name} produced a missing or non-finite tensor")


def _seed_profile(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_parameter_gradient(module: torch.nn.Module) -> Tensor:
    gradient = next(
        (parameter.grad for parameter in module.parameters() if parameter.grad is not None),
        None,
    )
    if gradient is None:
        raise PreflightError("backward produced no parameter gradient")
    return gradient


ProfileWorkload = Callable[[], tuple[dict[str, Any], Any]]


def _measure_profile(
    name: str,
    device: torch.device,
    spec: dict[str, Any],
    workload: ProfileWorkload,
) -> dict[str, Any]:
    """Measure one isolated forward/backward path, including saved-tensor peak."""

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_allocated = 0
        baseline_reserved = 0
    rss_before = _current_rss_bytes()
    started = time.perf_counter()
    keepalive: Any = None
    try:
        details, keepalive = workload()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            allocated = reserved = peak_allocated = peak_reserved = 0
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower():
            raise PreflightError(
                f"profile {name!r} exhausted accelerator memory for envelope {spec}"
            ) from error
        raise
    wall_time = time.perf_counter() - started
    rss_after = _current_rss_bytes()
    result = {
        "status": "passed",
        "profile": name,
        "spec": spec,
        "memory_unit": "bytes",
        "wall_time_unit": "seconds",
        "allocated": allocated,
        "reserved": reserved,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "peak_allocated_above_baseline": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_above_baseline": max(0, peak_reserved - baseline_reserved),
        "wall_time": wall_time,
        "cpu_rss_before": rss_before,
        "cpu_rss_after": rss_after,
        "cpu_rss_above_before": (
            None if rss_before is None or rss_after is None else max(0, rss_after - rss_before)
        ),
        "cpu_peak_rss": _peak_rss_bytes(),
        **details,
    }
    del keepalive
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _conductance_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.conductance_gat.sparse import (
        PackedGraphBatch,
        SparseIncidenceConductanceLayer,
    )

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    channels = 64 if device.type == "cuda" else 8
    edge_width = 3
    use_amp = config.amp and device.type == "cuda"
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "channels": channels,
        "edge_feature_channels": edge_width,
        "amp": use_amp,
        "production_path": "SparseIncidenceConductanceLayer",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260829)
        generator = torch.Generator(device="cpu").manual_seed(20260829)
        host_batch = PackedGraphBatch(
            node_state=torch.randn(batch_size * nodes, channels, generator=generator),
            edge_index=_packed_edges(batch_size, nodes, edges),
            edge_features=torch.randn(batch_size * edges, edge_width, generator=generator),
            node_graph=torch.arange(batch_size).repeat_interleave(nodes),
            edge_graph=torch.arange(batch_size).repeat_interleave(edges),
            graph_ids=[f"preflight-{index}" for index in range(batch_size)],
            requested_step=torch.full((batch_size,), 0.02),
        )
        host_bytes = _tensor_bytes(host_batch)
        batch = host_batch.to(device)
        batch.node_state.requires_grad_(True)
        model = SparseIncidenceConductanceLayer(
            channels=channels,
            edge_feature_channels=edge_width,
            hidden_channels=channels,
            requested_step=0.02,
            mode="full",
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            output, diagnostics = model(batch, return_diagnostics=True)
            loss = output.square().mean()
        loss.backward()
        first_gradient = _first_parameter_gradient(model)
        _assert_finite("conductance profile", output, loss, batch.node_state.grad, first_gradient)
        return {
            "loss": float(loss.detach().cpu()),
            "message_sum_abs": float(
                diagnostics["node_message"].sum(dim=0).abs().max().detach().cpu()
            ),
            "host_input_tensor_bytes": host_bytes,
        }, (host_batch, batch, model, output, diagnostics, loss)

    return _measure_profile("conductance", device, spec, workload)


def _prepared_cycle_graphs(
    *,
    required_variants: Sequence[str],
    batch_size: int,
    nodes: int,
    edges: int,
    cycle_rank: int,
    generator: torch.Generator,
) -> list[Any]:
    from research.cycle_pe.features import cycle_set_statistics
    from research.cycle_pe.paper_model import PreparedGraph

    raw = torch.randn(edges, cycle_rank, generator=generator)
    orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
    cycle_set = (
        torch.as_tensor(cycle_set_statistics(orthogonal.numpy()), dtype=torch.float32)
        if "set" in required_variants
        else None
    )
    projector = orthogonal @ orthogonal.T if "projector" in required_variants else None
    edge_pairs = _base_edges(nodes, edges).T.contiguous()
    result = []
    for index in range(batch_size):
        result.append(
            PreparedGraph(
                graph_id=f"preflight-{index}",
                split="preflight",
                family="synthetic-envelope",
                num_nodes=nodes,
                cycle_rank=cycle_rank,
                edges=edge_pairs.clone(),
                node_features=torch.randn(nodes, 5, generator=generator),
                edge_features=torch.randn(edges, 3, generator=generator),
                raw_basis=orthogonal.clone(),
                cycle_set=None if cycle_set is None else cycle_set.clone(),
                projector=None if projector is None else projector.clone(),
                edge_targets=None,
                node_targets=None,
                graph_targets=torch.zeros(1),
            )
        )
    return result


def _cycle_projector_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.cycle_pe.paper_model import PaperCycleModel

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    cycle_rank = config.cycle_rank
    hidden = 64 if device.type == "cuda" else 8
    pe_dim = 32 if device.type == "cuda" else 8
    layers = 3 if device.type == "cuda" else 1
    use_amp = config.amp and device.type == "cuda"
    variants = config.cycle_variants
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "cycle_rank": cycle_rank,
        "selected_variants": list(variants),
        "projector_shape_per_graph": [edges, edges] if "projector" in variants else None,
        "hidden_dim": hidden,
        "pe_dim": pe_dim,
        "layers": layers,
        "amp": use_amp,
        "production_path": "PaperCycleModel(selected variants)",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260830)
        generator = torch.Generator(device="cpu").manual_seed(20260830)
        host_graphs = _prepared_cycle_graphs(
            required_variants=variants,
            batch_size=batch_size,
            nodes=nodes,
            edges=edges,
            cycle_rank=cycle_rank,
            generator=generator,
        )
        variant_losses: dict[str, float] = {}
        host_bytes = _tensor_bytes(host_graphs)
        for variant in variants:
            _seed_profile(20260830)
            graphs = [graph.to(device) for graph in host_graphs]
            model = PaperCycleModel(
                variant=variant,
                raw_width=cycle_rank,
                node_input_dim=5,
                edge_input_dim=3,
                edge_output_dim=0,
                node_output_dim=0,
                graph_output_dim=1,
                hidden_dim=hidden,
                pe_dim=pe_dim,
                layers=layers,
                embedding_dim=16,
            ).to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(graphs)
                predictions = torch.cat(
                    [output.graph for output in outputs if output.graph is not None]
                )
                embeddings = torch.stack([output.embedding for output in outputs])
                loss = predictions.square().mean() + embeddings.square().mean()
            loss.backward()
            first_gradient = _first_parameter_gradient(model)
            _assert_finite(
                f"cycle profile ({variant})",
                predictions,
                embeddings,
                loss,
                first_gradient,
            )
            variant_losses[variant] = float(loss.detach().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del first_gradient, predictions, embeddings, loss, outputs, model, graphs
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        return {
            "loss": max(variant_losses.values()),
            "variant_losses": variant_losses,
            "host_input_tensor_bytes": host_bytes,
        }, None

    return _measure_profile("cycle-projector", device, spec, workload)


def _tree_chart_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.tree_augmentation.paper_model import PaddedChartBatch, VariableBetaCycleEncoder

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    cycle_rank = config.cycle_rank
    hidden = 64 if device.type == "cuda" else 8
    use_amp = config.amp and device.type == "cuda"
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "cycle_rank": cycle_rank,
        "dense_chart_shape": [batch_size, edges, cycle_rank],
        "hidden_dim": hidden,
        "amp": use_amp,
        "production_path": "VariableBetaCycleEncoder(PaddedChartBatch)",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260831)
        generator = torch.Generator(device="cpu").manual_seed(20260831)
        basis = torch.randn(batch_size, edges, cycle_rank, generator=generator)
        base_pairs = _base_edges(nodes, edges).T
        host_batch = PaddedChartBatch(
            basis=basis,
            edge_features=torch.randn(batch_size, edges, 4, generator=generator),
            edge_mask=torch.ones(batch_size, edges, dtype=torch.bool),
            cycle_mask=torch.ones(batch_size, cycle_rank, dtype=torch.bool),
            edge_index=base_pairs.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            node_categories=torch.zeros(batch_size, nodes, dtype=torch.long),
            edge_categories=torch.zeros(batch_size, edges, dtype=torch.long),
            node_mask=torch.ones(batch_size, nodes, dtype=torch.bool),
            targets=torch.randn(batch_size, 1, generator=generator),
            graph_ids=tuple(f"preflight-{index}" for index in range(batch_size)),
        )
        host_bytes = _tensor_bytes(host_batch)
        batch = host_batch.to(device, pin_memory=False, non_blocking=False)
        model = VariableBetaCycleEncoder(hidden_dim=hidden, output_dim=1).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            predictions = model(batch)
            loss = nnf.mse_loss(predictions, batch.targets)
        loss.backward()
        first_gradient = _first_parameter_gradient(model)
        _assert_finite("tree-chart profile", predictions, loss, first_gradient)
        return {
            "loss": float(loss.detach().cpu()),
            "host_input_tensor_bytes": host_bytes,
        }, (host_batch, batch, model, predictions, loss)

    return _measure_profile("tree-chart", device, spec, workload)


def _brec_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.cycle_pe.paper import brec_hotelling_t2
    from research.cycle_pe.paper_model import PaperCycleModel

    protocol = config.brec_protocol
    requested_batch_size = config.brec_batch_size
    if device.type == "cuda" and protocol == "official" and requested_batch_size != 16:
        raise PreflightError("official BREC CUDA profile requires --brec-batch-size 16")
    # ``research.cycle_pe.paper._brec_batches`` consumes whole graph pairs.
    # Preserve that exact behavior for custom/tiny odd batch-size requests.
    pairs_per_batch = max(1, requested_batch_size // 2)
    batch_size = 2 * pairs_per_batch
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    cycle_rank = config.cycle_rank
    hidden = 64 if device.type == "cuda" else 8
    pe_dim = 32 if device.type == "cuda" else 8
    layers = 3 if device.type == "cuda" else 1
    num_relabel = 32
    embedding_dim = 16
    variants = config.cycle_variants
    use_amp = protocol == "custom" and config.brec_amp and device.type == "cuda"
    spec = {
        "protocol": protocol,
        "requested_batch_size": requested_batch_size,
        "batch_size_graphs": batch_size,
        "official_batch_size": 16,
        "official_batch_size_match": batch_size == 16 if protocol == "official" else None,
        "selected_variants": list(variants),
        "dtype": "mixed_float32_autocast" if use_amp else "float32",
        "model_parameter_dtype": "float32",
        "amp": use_amp,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "cycle_rank": cycle_rank,
        "projector_shape_per_graph": [edges, edges] if "projector" in variants else None,
        "num_relabel_pairs_for_t2": num_relabel,
        "covariance_shape": [embedding_dim, embedding_dim],
        "production_path": "PaperCycleModel(selected variants) + brec_hotelling_t2(cov/pinv)",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260901)
        generator = torch.Generator(device="cpu").manual_seed(20260901)
        host_graphs = _prepared_cycle_graphs(
            required_variants=variants,
            batch_size=batch_size,
            nodes=nodes,
            edges=edges,
            cycle_rank=cycle_rank,
            generator=generator,
        )
        variant_losses: dict[str, float] = {}
        variant_t2: dict[str, float] = {}
        host_bytes = _tensor_bytes(host_graphs)
        for variant in variants:
            _seed_profile(20260901)
            graphs = [graph.to(device) for graph in host_graphs]
            model = PaperCycleModel(
                variant=variant,
                raw_width=cycle_rank,
                node_input_dim=5,
                edge_input_dim=3,
                edge_output_dim=0,
                node_output_dim=0,
                graph_output_dim=0,
                hidden_dim=hidden,
                pe_dim=pe_dim,
                layers=layers,
                embedding_dim=embedding_dim,
            ).to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(graphs)
                embeddings = torch.stack([output.embedding for output in outputs])
                targets = -torch.ones(
                    batch_size // 2,
                    device=device,
                    dtype=embeddings.dtype,
                )
                loss = nnf.cosine_embedding_loss(embeddings[0::2], embeddings[1::2], targets)
            loss.backward()
            repeat_count = math.ceil((2 * num_relabel) / embeddings.shape[0])
            t2_embeddings = embeddings.detach().float().repeat(repeat_count, 1)[: 2 * num_relabel]
            jitter = torch.linspace(
                -1.0e-3,
                1.0e-3,
                t2_embeddings.numel(),
                device=device,
                dtype=torch.float32,
            ).reshape_as(t2_embeddings)
            t2 = brec_hotelling_t2(t2_embeddings + jitter)
            first_gradient = _first_parameter_gradient(model)
            _assert_finite(f"BREC profile ({variant})", embeddings, loss, t2, first_gradient)
            variant_losses[variant] = float(loss.detach().cpu())
            variant_t2[variant] = float(t2.detach().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del (
                first_gradient,
                t2,
                jitter,
                t2_embeddings,
                repeat_count,
                loss,
                targets,
                embeddings,
                outputs,
                model,
                graphs,
            )
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        return {
            "loss": max(variant_losses.values()),
            "hotelling_t2": max(variant_t2.values()),
            "variant_losses": variant_losses,
            "variant_hotelling_t2": variant_t2,
            "host_input_tensor_bytes": host_bytes,
        }, None

    return _measure_profile("brec", device, spec, workload)


def _reciprocal_pyg_data(
    *,
    nodes: int,
    edges: int,
    task: str,
    categorical: bool,
    seed: int,
) -> Any:
    try:
        from torch_geometric.data import Data
    except (ImportError, OSError) as error:
        raise PreflightError(
            "public-pyg profile requires torch-geometric; run scripts/setup_gpu.sh"
        ) from error

    generator = torch.Generator(device="cpu").manual_seed(seed)
    undirected = _base_edges(nodes, edges)
    directed = torch.cat((undirected, undirected.flip(0)), dim=1)
    if categorical:
        x = torch.randint(0, 2, (nodes, 9), generator=generator)
        base_attributes = torch.randint(0, 2, (edges, 3), generator=generator)
        y = torch.tensor([float(seed % 2)])
    else:
        x = torch.randn(nodes, 14, generator=generator)
        base_attributes = torch.randn(edges, 2, generator=generator)
        y = torch.arange(nodes).remainder(21)
    edge_attr = torch.cat((base_attributes, base_attributes), dim=0)
    return Data(x=x, edge_index=directed, edge_attr=edge_attr, y=y), task


def _public_pyg_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    try:
        import ogb  # noqa: F401
        import torch_geometric  # noqa: F401
    except (ImportError, OSError) as error:
        raise PreflightError(
            "public-pyg profile requires importable torch-geometric and ogb; "
            "run scripts/setup_gpu.sh"
        ) from error

    from research.conductance_gat.paper import (
        PublicConductanceModel,
        _public_loader,
        _public_loss,
    )
    from research.conductance_gat.public_data import adapt_pyg_graph

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    hidden = 64 if device.type == "cuda" else 8
    use_amp = config.amp and device.type == "cuda"
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "physical_edges_per_graph": edges,
        "hidden_dim": hidden,
        "amp": use_amp,
        "tasks": ["pascalvoc_sp/node", "ogbg_molhiv/graph"],
        "production_path": "PyG Data -> adapt_pyg_graph -> DataLoader -> PublicConductanceModel",
        "downloads": False,
    }

    def records(task: str, categorical: bool, seed_offset: int) -> list[dict[str, Any]]:
        result = []
        for index in range(batch_size):
            data, selected_task = _reciprocal_pyg_data(
                nodes=nodes,
                edges=edges,
                task=task,
                categorical=categorical,
                seed=20260902 + seed_offset + index,
            )
            result.append(adapt_pyg_graph(data, f"preflight-{task}-{index}", task=selected_task))
        return result

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260902)
        pascal_records = records("node", False, 0)
        pascal_host_bytes = _tensor_bytes(pascal_records)
        pascal_loader = _public_loader(
            pascal_records,
            batch_size=batch_size,
            shuffle=False,
            seed=20260902,
            pin_memory=False,
            num_workers=0,
        )
        pascal_batch = next(iter(pascal_loader)).to(device, non_blocking=False)
        pascal_model = PublicConductanceModel(
            pascal_records[0],
            hidden=hidden,
            num_classes=21,
            official_molecule=False,
            backbone="conductance_model",
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pascal_logits = pascal_model(pascal_batch)
            pascal_loss = _public_loss(pascal_logits, pascal_batch.y, "node")
        pascal_loss.backward()
        _assert_finite("public PascalVOC-SP profile", pascal_logits, pascal_loss)
        pascal_loss_value = float(pascal_loss.detach().cpu())
        del pascal_loader, pascal_batch, pascal_model, pascal_logits, pascal_loss
        if device.type == "cuda":
            gc.collect()

        molecule_records = records("graph", True, 10_000)
        molecule_host_bytes = _tensor_bytes(molecule_records)
        molecule_loader = _public_loader(
            molecule_records,
            batch_size=batch_size,
            shuffle=False,
            seed=20260903,
            pin_memory=False,
            num_workers=0,
        )
        molecule_batch = next(iter(molecule_loader)).to(device, non_blocking=False)
        molecule_model = PublicConductanceModel(
            molecule_records[0],
            hidden=hidden,
            num_classes=2,
            official_molecule=True,
            backbone="conductance_model",
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            molecule_logits = molecule_model(molecule_batch)
            molecule_loss = _public_loss(molecule_logits, molecule_batch.y, "graph")
        molecule_loss.backward()
        first_gradient = _first_parameter_gradient(molecule_model)
        _assert_finite(
            "public ogbg-molhiv profile",
            molecule_logits,
            molecule_loss,
            first_gradient,
        )
        return {
            "pascalvoc_node_loss": pascal_loss_value,
            "molhiv_graph_loss": float(molecule_loss.detach().cpu()),
            "host_input_tensor_bytes": pascal_host_bytes + molecule_host_bytes,
        }, (
            pascal_records,
            molecule_records,
            molecule_loader,
            molecule_batch,
            molecule_model,
            molecule_logits,
            molecule_loss,
        )

    return _measure_profile("public-pyg", device, spec, workload)


PROFILE_RUNNERS: dict[str, Callable[[torch.device, ProfileConfig], dict[str, Any]]] = {
    "conductance": _conductance_profile,
    "cycle-projector": _cycle_projector_profile,
    "tree-chart": _tree_chart_profile,
    "brec": _brec_profile,
    "public-pyg": _public_pyg_profile,
}


def _normalize_profiles(profiles: Sequence[str] | None) -> tuple[str, ...]:
    selected = ("conductance",) if profiles is None else tuple(profiles)
    if not selected:
        raise PreflightError("at least one --profile is required")
    if "all" in selected:
        if len(selected) != 1:
            raise PreflightError("--profile all cannot be combined with another profile")
        return PROFILE_NAMES
    unknown = sorted(set(selected) - set(PROFILE_NAMES))
    if unknown:
        raise PreflightError(f"unknown preflight profiles: {unknown}")
    return tuple(dict.fromkeys(selected))


def _cycle_variants(value: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError("--cycle-variants must be non-empty")
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("--cycle-variants must not contain duplicates")
    unknown = sorted(set(selected) - set(CYCLE_VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--cycle-variants contains unsupported values {unknown}; "
            f"choose from {list(CYCLE_VARIANTS)}"
        )
    return selected


def build_report(
    requested_device: str,
    *,
    allow_cpu: bool,
    require_paper_dependencies: bool,
    min_free_gb: float,
    profiles: Sequence[str] | None = None,
    profile_config: ProfileConfig | None = None,
) -> dict[str, Any]:
    if min_free_gb < 0.0 or not math.isfinite(min_free_gb):
        raise PreflightError("--min-free-gb must be a finite non-negative number")
    dependency_errors = _paper_dependency_import_errors()
    missing = _missing_paper_dependencies()
    if require_paper_dependencies and dependency_errors:
        rendered_errors = "; ".join(
            f"{name}: {message}" for name, message in sorted(dependency_errors.items())
        )
        raise PreflightError(
            "paper dependency import check failed: "
            + rendered_errors
            + "; run scripts/setup_gpu.sh"
        )
    selected_profiles = _normalize_profiles(profiles)
    requested_config = ProfileConfig() if profile_config is None else profile_config
    requested_config.validate()
    device = _resolve_device(requested_device, allow_cpu=allow_cpu)
    effective_config = _effective_profile_config(requested_config, device)
    report: dict[str, Any] = {
        "status": "passed",
        "requested_device": requested_device,
        "resolved_device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "paper_dependencies": _package_versions(),
        "missing_paper_dependencies": missing,
        "paper_dependency_import_errors": dependency_errors,
        "nvidia_smi": _nvidia_smi(),
        "profile_kind": "synthetic_shape_stress_not_dataset_e2e",
        "selected_profiles": list(selected_profiles),
        "requested_profile_config": vars(requested_config),
        "effective_profile_config": vars(effective_config),
    }
    if device.type == "cuda":
        index = device.index
        assert index is not None
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        free_gb = free_bytes / 1024**3
        if free_gb < min_free_gb:
            raise PreflightError(
                f"CUDA device {index} has {free_gb:.2f} GiB free, below {min_free_gb:.2f} GiB"
            )
        properties = torch.cuda.get_device_properties(index)
        report["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_gb": total_bytes / 1024**3,
            "free_memory_gb": free_gb,
            "device_count": torch.cuda.device_count(),
        }
    else:
        report["gpu"] = None
    profile_results: dict[str, Any] = {}
    for name in selected_profiles:
        try:
            profile_results[name] = PROFILE_RUNNERS[name](device, effective_config)
        except PreflightError:
            raise
        except Exception as error:
            raise PreflightError(
                f"profile {name!r} failed: {type(error).__name__}: {error}"
            ) from error
    report["profiles"] = profile_results
    # Compatibility for callers of the original single-path preflight report.
    if "conductance" in profile_results:
        report["incidence_forward_backward"] = profile_results["conductance"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--require-paper-deps", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        choices=("all", *PROFILE_NAMES),
        help="repeat for every experiment path to shape-stress; default: conductance",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--brec-batch-size", type=int, default=16)
    parser.add_argument("--nodes-per-graph", type=int, default=64)
    parser.add_argument("--edges-per-graph", type=int, default=128)
    parser.add_argument("--cycle-rank", type=int, default=64)
    parser.add_argument(
        "--cycle-variants",
        type=_cycle_variants,
        default=("projector",),
        help="comma-separated PaperCycleModel variants to exercise",
    )
    parser.add_argument(
        "--brec-protocol",
        choices=BREC_PROTOCOLS,
        default="official",
        help="match the selected paper runner's official or tiny/custom BREC path",
    )
    parser.add_argument(
        "--brec-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="match AMP in custom BREC; the official protocol rejects AMP",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match AMP-enabled suites; BREC official always remains float32/no-AMP",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        report = build_report(
            args.device,
            allow_cpu=args.allow_cpu,
            require_paper_dependencies=args.require_paper_deps,
            min_free_gb=args.min_free_gb,
            profiles=args.profiles,
            profile_config=ProfileConfig(
                batch_size=args.batch_size,
                brec_batch_size=args.brec_batch_size,
                nodes_per_graph=args.nodes_per_graph,
                edges_per_graph=args.edges_per_graph,
                cycle_rank=args.cycle_rank,
                amp=args.amp,
                cycle_variants=args.cycle_variants,
                brec_protocol=args.brec_protocol,
                brec_amp=args.brec_amp,
            ),
        )
    except (PreflightError, RuntimeError) as error:
        print(f"GPU PREFLIGHT FAILED: {error}")
        return 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

# scripts/paper.sh

````bash
#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/conda_env.sh"

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"
"${environment_python}" scripts/run_paper.py "$@"
````

# scripts/run_all.py

````python
#!/usr/bin/env python3
"""Run every currently implemented research track without combining models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCE_DIRS = (
    "src",
    "tests",
    "research/conductance_gat",
    "research/cycle_pe",
    "research/tree_augmentation",
    "scripts",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("smoke-%Y%m%dT%H%M%S%fZ")


def _validate_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _commands(
    *,
    python: str,
    device: str,
    run_id: str,
    skip_tests: bool,
    skip_lint: bool,
) -> list[tuple[str, list[str]]]:
    conductance_output = PROJECT_ROOT / "research" / "conductance_gat" / "results" / run_id
    cycle_output = PROJECT_ROOT / "research" / "cycle_pe" / "results" / run_id / "summary.json"
    tree_output = (
        PROJECT_ROOT / "research" / "tree_augmentation" / "results" / run_id / "summary.json"
    )
    commands: list[tuple[str, list[str]]] = []
    commands.append(("dependencies", [python, "-m", "pip", "check"]))
    commands.append(
        (
            "dataset_plan",
            [python, str(PROJECT_ROOT / "scripts" / "check_datasets.py"), "--profile", "smoke"],
        )
    )
    if not skip_tests:
        commands.append(("tests", [python, "-m", "pytest", "-q"]))
    if not skip_lint:
        commands.append(("lint", [python, "-m", "ruff", "check", *ACTIVE_SOURCE_DIRS]))
    commands.extend(
        [
            (
                "conductance_gat",
                [
                    python,
                    "-m",
                    "research.conductance_gat.run",
                    "--config",
                    str(PROJECT_ROOT / "research" / "conductance_gat" / "config.yaml"),
                    "--device",
                    device,
                    "--output-dir",
                    str(conductance_output),
                ],
            ),
            (
                "cycle_pe",
                [
                    python,
                    "-m",
                    "research.cycle_pe.run",
                    "--config",
                    str(PROJECT_ROOT / "research" / "cycle_pe" / "config.yaml"),
                    "--output",
                    str(cycle_output),
                ],
            ),
            (
                "tree_augmentation",
                [
                    python,
                    "-m",
                    "research.tree_augmentation.run",
                    "--config",
                    str(PROJECT_ROOT / "research" / "tree_augmentation" / "config.yaml"),
                    "--output",
                    str(tree_output),
                ],
            ),
        ]
    )
    return commands


def _output_paths(run_id: str) -> dict[str, Path]:
    return {
        "conductance_gat": PROJECT_ROOT / "research" / "conductance_gat" / "results" / run_id,
        "cycle_pe": PROJECT_ROOT / "research" / "cycle_pe" / "results" / run_id / "summary.json",
        "tree_augmentation": PROJECT_ROOT
        / "research"
        / "tree_augmentation"
        / "results"
        / run_id
        / "summary.json",
    }


def _expected_artifacts(outputs: dict[str, Path]) -> tuple[Path, ...]:
    conductance = outputs["conductance_gat"]
    return (
        conductance / "summary.json",
        conductance / "learned_history.csv",
        conductance / "isotropic_history.csv",
        conductance / "learned_model.pt",
        outputs["cycle_pe"],
        outputs["tree_augmentation"],
    )


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_run(run_dir: Path) -> dict[str, Any]:
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    snapshots: dict[str, Any] = {}
    for track in ("conductance_gat", "cycle_pe", "tree_augmentation"):
        for source_name in ("config.yaml", "datasets.yaml"):
            source = PROJECT_ROOT / "research" / track / source_name
            target = config_dir / f"{track}-{source_name}"
            shutil.copy2(source, target)
            snapshots[f"{track}-{source_name}"] = {
                "path": str(target),
                "sha256": _sha256(target),
            }

    distributions = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    environment_path = run_dir / "environment.txt"
    environment_path.write_text("\n".join(distributions) + "\n", encoding="utf-8")
    snapshots["environment"] = {
        "path": str(environment_path),
        "sha256": _sha256(environment_path),
    }
    return snapshots


def _source_revision() -> dict[str, Any]:
    if not (PROJECT_ROOT / ".git").exists():
        return {"git_available": False, "revision": None, "dirty": None}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_available": revision.returncode == 0,
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--run-id", type=_validate_run_id, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or _default_run_id()
    commands = _commands(
        python=sys.executable,
        device=args.device,
        run_id=run_id,
        skip_tests=args.skip_tests,
        skip_lint=args.skip_lint,
    )
    run_dir = PROJECT_ROOT / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    outputs = _output_paths(run_id)
    payload: dict[str, Any] = {
        "scope": "all_currently_implemented_independent_smoke_tracks",
        "paper_benchmark_suite_complete": False,
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source": _source_revision(),
        "device_request": args.device,
        "commands": [],
    }

    environment = os.environ.copy()
    python_path = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["PYTHONUTF8"] = "1"

    if args.dry_run:
        for name, command in commands:
            print(f"[{name}] {_command_text(command)}")
        print(f"[manifest] {manifest_path}")
        return 0

    collisions = [
        path
        for path in (
            run_dir,
            outputs["conductance_gat"],
            outputs["cycle_pe"].parent,
            outputs["tree_augmentation"].parent,
        )
        if path.exists()
    ]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        print(f"run id {run_id!r} already exists; refusing to overwrite: {joined}", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=False)
    payload["snapshots"] = _snapshot_run(run_dir)
    payload["status"] = "running"
    _write_manifest(manifest_path, payload)

    try:
        for name, command in commands:
            print(f"\n== {name}: {_command_text(command)} ==", flush=True)
            started = datetime.now(UTC)
            log_path = run_dir / "logs" / f"{name}.log"
            returncode = _run_logged(
                command,
                cwd=PROJECT_ROOT,
                environment=environment,
                log_path=log_path,
            )
            record = {
                "name": name,
                "command": command,
                "returncode": returncode,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                "log": str(log_path),
            }
            payload["commands"].append(record)
            _write_manifest(manifest_path, payload)
            if returncode != 0:
                payload["status"] = "failed"
                payload["failed_step"] = name
                payload["finished_at_utc"] = datetime.now(UTC).isoformat()
                _write_manifest(manifest_path, payload)
                print(f"failed at {name}; manifest: {manifest_path}", file=sys.stderr)
                return returncode
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, payload)
        print(f"interrupted; manifest: {manifest_path}", file=sys.stderr)
        return 130

    missing = [str(path) for path in _expected_artifacts(outputs) if not path.is_file()]
    if missing:
        payload["status"] = "failed"
        payload["failed_step"] = "artifact_validation"
        payload["missing_artifacts"] = missing
        payload["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, payload)
        print(f"missing expected artifacts: {missing}", file=sys.stderr)
        return 1

    try:
        for summary in (
            outputs["conductance_gat"] / "summary.json",
            outputs["cycle_pe"],
            outputs["tree_augmentation"],
        ):
            content = json.loads(summary.read_text(encoding="utf-8"))
            if not _all_finite(content):
                raise ValueError(f"non-finite value in {summary}")
    except (json.JSONDecodeError, ValueError) as error:
        payload["status"] = "failed"
        payload["failed_step"] = "artifact_validation"
        payload["artifact_error"] = str(error)
        payload["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_manifest(manifest_path, payload)
        print(f"invalid result artifact: {error}", file=sys.stderr)
        return 1

    payload["status"] = "passed"
    payload["finished_at_utc"] = datetime.now(UTC).isoformat()
    payload["outputs"] = {name: str(path) for name, path in outputs.items()}
    _write_manifest(manifest_path, payload)
    print(f"\nall implemented tracks passed; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

# scripts/run_paper.py

````python
#!/usr/bin/env python3
"""Run the independent paper experiment tracks on a CUDA host."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chartgat.cache import atomic_write_bytes, atomic_write_json

try:
    from scripts.aggregate_paper import aggregate_manifest
except ModuleNotFoundError:  # Direct ``python scripts/run_paper.py`` execution.
    from aggregate_paper import aggregate_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_MODULES = {
    "conductance_gat": "research.conductance_gat.paper",
    "cycle_pe": "research.cycle_pe.paper",
    "tree_augmentation": "research.tree_augmentation.paper",
}
CYCLE_BREC_OFFICIAL_SEEDS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
CYCLE_VARIANTS = ("no_pe", "raw", "set", "projector")
CYCLE_CORE_TARGETS = ("edge", "node", "graph")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("paper-%Y%m%dT%H%M%S%fZ")


def _run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    return seeds


def _comma_subset(value: str, *, choices: tuple[str, ...], option: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError(f"{option} must be non-empty")
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"{option} must not contain duplicates")
    unknown = sorted(set(selected) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"{option} contains unsupported values {unknown}; choose from {list(choices)}"
        )
    return selected


def _cycle_variants(value: str) -> tuple[str, ...]:
    return _comma_subset(value, choices=CYCLE_VARIANTS, option="--cycle-variants")


def _cycle_core_targets(value: str) -> tuple[str, ...]:
    return _comma_subset(value, choices=CYCLE_CORE_TARGETS, option="--cycle-core-targets")


def _selected_tracks(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return tuple(TRACK_MODULES)
    return tuple(dict.fromkeys(values))


def _track_run_root(track: str, run_id: str, results_root: Path | None = None) -> Path:
    if results_root is None:
        base = PROJECT_ROOT / "research" / track / "results" / "paper"
    else:
        base = results_root.expanduser().resolve() / track
    return base / run_id


def _output_dir(
    track: str,
    run_id: str,
    model_seed: int,
    results_root: Path | None = None,
) -> Path:
    return _track_run_root(track, run_id, results_root) / f"model-seed-{model_seed}"


def _commands(args: argparse.Namespace, run_id: str) -> list[tuple[str, list[str], Path | None]]:
    commands: list[tuple[str, list[str], Path | None]] = []
    selected_tracks = _selected_tracks(args.tracks)
    preflight_output = PROJECT_ROOT / "runs" / "paper" / run_id / "gpu-preflight.json"
    preflight_device = "cpu" if args.prepare_only else args.device
    brec_protocol = "custom" if args.tiny else "official"
    preflight_brec_batch_size = args.batch_size if args.tiny else 16
    preflight_brec_amp = (
        brec_protocol == "custom" and args.amp and preflight_device.lower().startswith("cuda")
    )
    preflight = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "gpu_preflight.py"),
        "--device",
        preflight_device,
        "--min-free-gb",
        str(0.0 if args.prepare_only else args.min_free_gb),
        "--batch-size",
        str(args.batch_size),
        "--brec-batch-size",
        str(preflight_brec_batch_size),
        "--nodes-per-graph",
        str(args.preflight_nodes_per_graph),
        "--edges-per-graph",
        str(args.preflight_edges_per_graph),
        "--cycle-rank",
        str(args.preflight_cycle_rank),
        "--cycle-variants",
        ",".join(args.cycle_variants),
        "--brec-protocol",
        brec_protocol,
        "--json-out",
        str(preflight_output),
    ]
    if args.amp and preflight_device.lower().startswith("cuda"):
        preflight.append("--amp")
    else:
        preflight.append("--no-amp")
    preflight.append("--brec-amp" if preflight_brec_amp else "--no-brec-amp")
    if args.prepare_only or args.allow_cpu:
        preflight.append("--allow-cpu")
    if args.suite == "all":
        preflight.append("--require-paper-deps")
    profiles: list[str] = []
    if args.prepare_only:
        # Data preparation needs an inexpensive CPU code smoke, not a claim
        # about accelerator capacity.  The CUDA training invocation below runs
        # the selected high-memory envelopes at their exact requested sizes.
        profiles.append("conductance")
    else:
        if "conductance_gat" in selected_tracks:
            profiles.append("conductance")
        if "cycle_pe" in selected_tracks:
            profiles.append("cycle-projector")
        if "tree_augmentation" in selected_tracks:
            profiles.append("tree-chart")
        if args.suite == "all" and "cycle_pe" in selected_tracks:
            profiles.append("brec")
        if args.suite == "all" and "conductance_gat" in selected_tracks:
            profiles.append("public-pyg")
        if not profiles:
            profiles.append("conductance")
    for profile in dict.fromkeys(profiles):
        preflight.extend(("--profile", profile))
    commands.append(("gpu_preflight", preflight, preflight_output))

    data_root = args.data_root.expanduser().resolve()

    def cycle_arguments(suite: str) -> tuple[str, ...]:
        values = ["--variants", ",".join(args.cycle_variants)]
        if suite == "core":
            values.extend(("--core-targets", ",".join(args.cycle_core_targets)))
        # Official BREC owns its fixed 20-epoch/1e-4 optimization protocol.
        # Master tuning knobs apply to CycleCount/ZINC and to tiny custom BREC only.
        if suite != "brec" or args.tiny:
            if args.cycle_epochs is not None:
                values.extend(("--epochs", str(args.cycle_epochs)))
            if args.cycle_learning_rate is not None:
                values.extend(("--learning-rate", str(args.cycle_learning_rate)))
        return tuple(values)

    def add_child(
        *,
        track: str,
        suite: str,
        model_seed: int,
        name: str,
        output_dir: Path,
        extra_arguments: tuple[str, ...] = (),
        batch_size: int | None = None,
        workers: int | None = None,
        amp: bool | None = None,
    ) -> None:
        effective_batch_size = args.batch_size if batch_size is None else batch_size
        effective_workers = args.workers if workers is None else workers
        effective_amp = args.amp if amp is None else amp
        command = [
            sys.executable,
            "-m",
            TRACK_MODULES[track],
            "--suite",
            suite,
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu" if args.prepare_only else args.device,
            "--data-seed",
            str(args.data_seed),
            "--split-seed",
            str(args.split_seed),
            "--chart-seed",
            str(args.chart_seed),
            "--model-seed",
            str(model_seed),
            "--batch-size",
            str(effective_batch_size),
            "--workers",
            str(effective_workers),
        ]
        if args.prepare_only:
            command.append("--prepare-only")
        if args.allow_download:
            command.append("--allow-download")
        if args.tiny:
            command.append("--tiny")
        if not args.prepare_only:
            if effective_amp and args.device.lower().startswith("cuda"):
                command.append("--amp")
            elif not effective_amp or args.device.lower().startswith("cpu"):
                command.append("--no-amp")
        command.extend(extra_arguments)
        commands.append((name, command, output_dir))

    executed_model_seeds = args.model_seeds[:1] if args.prepare_only else args.model_seeds
    for track in selected_tracks:
        # BREC already performs its official ten model-search seeds internally.
        # Under suite=all, run CycleCount and ZINC for every outer experiment
        # seed, but dispatch BREC exactly once rather than multiplying it by the
        # five default outer seeds.
        if track == "cycle_pe" and args.suite == "all":
            cycle_root = _track_run_root(track, run_id, args.results_root)
            for model_seed in executed_model_seeds:
                for suite in ("core", "zinc"):
                    add_child(
                        track=track,
                        suite=suite,
                        model_seed=model_seed,
                        name=f"{track}:{suite}:model-seed-{model_seed}",
                        output_dir=cycle_root / f"model-seed-{model_seed}" / suite,
                        extra_arguments=cycle_arguments(suite),
                    )
            brec_label = "custom-tiny" if args.tiny else "official-10-seed"
            brec_run_name = f"brec-{brec_label}"
            add_child(
                track=track,
                suite="brec",
                model_seed=args.model_seeds[0],
                name=f"{track}:brec:{brec_label}",
                output_dir=cycle_root / brec_run_name,
                extra_arguments=(
                    *cycle_arguments("brec"),
                    "--brec-protocol",
                    brec_protocol,
                    "--brec-seeds",
                    ",".join(str(seed) for seed in CYCLE_BREC_OFFICIAL_SEEDS),
                ),
                batch_size=(args.batch_size if args.tiny else 16),
                workers=(args.workers if args.tiny else 0),
                amp=(args.amp if args.tiny else False),
            )
            continue

        # Dataset, split, and chart axes are fixed for a model-seed sweep.  A
        # prepare-only run therefore materializes each requested suite once.
        for model_seed in executed_model_seeds:
            extra_arguments = cycle_arguments(args.suite) if track == "cycle_pe" else ()
            add_child(
                track=track,
                suite=args.suite,
                model_seed=model_seed,
                name=f"{track}:model-seed-{model_seed}",
                output_dir=_output_dir(track, run_id, model_seed, args.results_root),
                extra_arguments=extra_arguments,
            )
    return commands


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> dict[str, Any]:
    if not (PROJECT_ROOT / ".git").exists():
        return {"git_available": False, "revision": None, "dirty": None}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_available": revision.returncode == 0,
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _environment_snapshot(path: Path) -> dict[str, Any]:
    distributions = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    atomic_write_bytes(path, ("\n".join(distributions) + "\n").encode("utf-8"))
    return {"path": str(path), "sha256": _sha256(path)}


def _snapshot_registries(run_dir: Path, tracks: tuple[str, ...]) -> dict[str, Any]:
    directory = run_dir / "dataset-registries"
    directory.mkdir(parents=True, exist_ok=False)
    snapshots: dict[str, Any] = {}
    for track in tracks:
        source = PROJECT_ROOT / "research" / track / "datasets.yaml"
        target = directory / f"{track}.yaml"
        shutil.copy2(source, target)
        snapshots[track] = {"path": str(target), "sha256": _sha256(target)}
    return snapshots


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _validate_json_outputs(path: Path) -> list[str]:
    if path.is_file() and path.suffix == ".json":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("*.json"))
    else:
        return [f"missing output: {path}"]
    if not candidates:
        return [f"no JSON artifact found under {path}"]
    errors: list[str] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid JSON {candidate}: {error}")
            continue
        if not _all_finite(payload):
            errors.append(f"non-finite numeric value in {candidate}")
    return errors


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload, sort_keys=False)


def _run_logged(command: list[str], *, log_path: Path) -> int:
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    child_environment["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            try:
                print(line, end="", flush=True)
            except UnicodeEncodeError:
                console_encoding = sys.stdout.encoding or "utf-8"
                safe_line = line.encode(console_encoding, errors="backslashreplace").decode(
                    console_encoding
                )
                print(safe_line, end="", flush=True)
            log.write(line)
        return process.wait()


def _stop_after_failure(name: str, *, fail_fast: bool) -> bool:
    """A shared preflight failure is fatal even when track failures are independent."""

    return name == "gpu_preflight" or fail_fast


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("all", *TRACK_MODULES),
        default=["all"],
    )
    parser.add_argument("--suite", choices=("core", "all"), default="core")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", type=_run_id)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "paper")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="optional shared result root (useful for scratch storage on a GPU cluster)",
    )
    parser.add_argument(
        "--model-seeds",
        "--seeds",
        dest="model_seeds",
        type=_seeds,
        default=(0, 1, 2, 3, 4),
        help="model/minibatch seeds; --seeds is a compatibility alias",
    )
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--chart-seed", type=int, default=0)
    parser.add_argument(
        "--cycle-variants",
        type=_cycle_variants,
        default=CYCLE_VARIANTS,
        help="comma-separated Cycle PE candidates forwarded only to the cycle track",
    )
    parser.add_argument(
        "--cycle-core-targets",
        type=_cycle_core_targets,
        default=CYCLE_CORE_TARGETS,
        help="comma-separated CycleCount target levels forwarded to cycle core runs",
    )
    parser.add_argument("--cycle-epochs", type=int)
    parser.add_argument("--cycle-learning-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument(
        "--preflight-nodes-per-graph",
        type=int,
        default=64,
        help="synthetic shape-stress envelope; this is not inferred from a dataset cache",
    )
    parser.add_argument(
        "--preflight-edges-per-graph",
        type=int,
        default=128,
        help="synthetic shape-stress envelope; increase for a larger intended graph regime",
    )
    parser.add_argument(
        "--preflight-cycle-rank",
        type=int,
        default=64,
        help="synthetic projector/chart rank envelope; must not exceed preflight edges",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow official public datasets to be downloaded into --data-root",
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    failure = parser.add_mutually_exclusive_group()
    failure.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed track/seed (the default audits every independent run)",
    )
    failure.add_argument(
        "--continue-on-error",
        dest="fail_fast",
        action="store_false",
        help="deprecated compatibility alias for the default independent-run behavior",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.workers < 0:
        print("batch size must be positive and workers must be non-negative", file=sys.stderr)
        return 2
    if min(args.data_seed, args.split_seed, args.chart_seed) < 0:
        print("data, split, and chart seeds must be non-negative", file=sys.stderr)
        return 2
    if args.cycle_epochs is not None and args.cycle_epochs < 1:
        print("--cycle-epochs must be positive", file=sys.stderr)
        return 2
    if args.cycle_learning_rate is not None and args.cycle_learning_rate <= 0:
        print("--cycle-learning-rate must be positive", file=sys.stderr)
        return 2
    if (
        min(
            args.preflight_nodes_per_graph,
            args.preflight_edges_per_graph,
            args.preflight_cycle_rank,
        )
        < 1
    ):
        print("preflight graph dimensions must be positive", file=sys.stderr)
        return 2
    if args.preflight_nodes_per_graph < 2:
        print("--preflight-nodes-per-graph must be at least 2", file=sys.stderr)
        return 2
    if args.preflight_cycle_rank > args.preflight_edges_per_graph:
        print("--preflight-cycle-rank cannot exceed --preflight-edges-per-graph", file=sys.stderr)
        return 2
    if (
        args.device.startswith("cpu")
        and not args.prepare_only
        and not (args.allow_cpu and args.tiny)
    ):
        print(
            "CPU is allowed only with --tiny --allow-cpu; paper training requires CUDA",
            file=sys.stderr,
        )
        return 2
    if args.allow_cpu and not args.tiny and not args.prepare_only:
        print("--allow-cpu is restricted to --tiny validation", file=sys.stderr)
        return 2

    run_id = args.run_id or _default_run_id()
    tracks = _selected_tracks(args.tracks)
    commands = _commands(args, run_id)
    if args.dry_run:
        for name, command, output in commands:
            print(f"[{name}] {_command_text(command)}")
            if output is not None:
                print(f"  output: {output}")
        return 0

    run_dir = PROJECT_ROOT / "runs" / "paper" / run_id
    if run_dir.exists() or any(
        _track_run_root(track, run_id, args.results_root).exists() for track in tracks
    ):
        print(f"run id already exists: {run_id}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "scope": "independent_paper_tracks",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source": _source_revision(),
        "device_request": args.device,
        "suite": args.suite,
        "tracks": list(tracks),
        "data_root": str(args.data_root.expanduser().resolve()),
        "results_root": (
            str(args.results_root.expanduser().resolve()) if args.results_root is not None else None
        ),
        "seed_axes": {
            "data": args.data_seed,
            "split": args.split_seed,
            "chart": args.chart_seed,
            "model": list(args.model_seeds),
        },
        "requested_model_seeds": list(args.model_seeds),
        "executed_model_seeds": ([] if args.prepare_only else list(args.model_seeds)),
        "execution_protocol": {
            "outer_model_seeds": list(args.model_seeds),
            "prepare_once_for_fixed_non_model_axes": args.prepare_only,
            "cycle_selection": (
                {
                    "variants": list(args.cycle_variants),
                    "core_targets": list(args.cycle_core_targets),
                    "epochs_override": args.cycle_epochs,
                    "learning_rate_override": args.cycle_learning_rate,
                    "official_brec_optimization_overrides_ignored": not args.tiny,
                }
                if "cycle_pe" in tracks
                else None
            ),
            "gpu_preflight": {
                "kind": "synthetic_shape_stress_not_dataset_e2e",
                "batch_size": args.batch_size,
                "brec_batch_size": args.batch_size if args.tiny else 16,
                "brec_protocol": "custom" if args.tiny else "official",
                "brec_amp": bool(
                    args.tiny
                    and args.amp
                    and not args.prepare_only
                    and args.device.lower().startswith("cuda")
                ),
                "cycle_variants": list(args.cycle_variants),
                "nodes_per_graph": args.preflight_nodes_per_graph,
                "edges_per_graph": args.preflight_edges_per_graph,
                "cycle_rank": args.preflight_cycle_rank,
            },
            "cycle_brec_internal_seeds": (
                list(CYCLE_BREC_OFFICIAL_SEEDS)
                if args.suite == "all" and "cycle_pe" in tracks
                else None
            ),
            "cycle_brec_dispatch_count": (1 if args.suite == "all" and "cycle_pe" in tracks else 0),
            "cycle_brec_protocol": (
                ("custom" if args.tiny else "official")
                if args.suite == "all" and "cycle_pe" in tracks
                else None
            ),
            "cycle_brec_training": (
                {
                    "batch_size": args.batch_size if args.tiny else 16,
                    "workers": args.workers if args.tiny else 0,
                    "amp": bool(
                        args.tiny
                        and args.amp
                        and not args.prepare_only
                        and args.device.lower().startswith("cuda")
                    ),
                }
                if args.suite == "all" and "cycle_pe" in tracks
                else None
            ),
        },
        "tiny": args.tiny,
        "prepare_only": args.prepare_only,
        "environment": _environment_snapshot(run_dir / "environment.txt"),
        "dataset_registries": _snapshot_registries(run_dir, tracks),
        "commands": [],
    }
    _write_manifest(manifest_path, manifest)

    failed = False
    for index, (name, command, output) in enumerate(commands):
        print(f"\n== {name}: {_command_text(command)} ==", flush=True)
        safe_name = name.replace(":", "-")
        log_path = logs_dir / f"{index:02d}-{safe_name}.log"
        started = datetime.now(UTC)
        return_code = _run_logged(command, log_path=log_path)
        finished = datetime.now(UTC)
        errors: list[str] = []
        if return_code == 0 and output is not None:
            errors = _validate_json_outputs(output)
        entry = {
            "name": name,
            "command": command,
            "returncode": return_code,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "log": str(log_path),
            "output": str(output) if output is not None else None,
            "artifact_errors": errors,
        }
        manifest["commands"].append(entry)
        _write_manifest(manifest_path, manifest)
        if return_code != 0 or errors:
            failed = True
            # Every child relies on the same accelerator/dependency preflight.
            # Independent-track continuation only makes sense after that
            # shared prerequisite has passed.
            if _stop_after_failure(name, fail_fast=args.fail_fast):
                break

    manifest["status"] = "failed" if failed else "passed"
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_manifest(manifest_path, manifest)
    if args.prepare_only:
        manifest["aggregation"] = {
            "status": "skipped",
            "reason": "prepare-only runs contain no model-seed metrics",
        }
    else:
        try:
            aggregate = aggregate_manifest(manifest_path)
            manifest["aggregation"] = {
                "status": "passed",
                "path": str(manifest_path.parent / "aggregate" / "aggregate.json"),
                **aggregate,
            }
        except Exception as error:  # Preserve the completed child-run audit trail.
            failed = True
            manifest["status"] = "failed"
            manifest["aggregation"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    _write_manifest(manifest_path, manifest)
    if failed:
        print(f"paper run failed; inspect {manifest_path}", file=sys.stderr)
        return 1
    print(f"all requested independent paper tracks passed; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

# scripts/setup.ps1

````powershell
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CodexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    if (Test-Path -LiteralPath $CodexPython) {
        $Python = $CodexPython
    } else {
        throw "Python was not found. Pass its path with -Python."
    }
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv (Join-Path $ProjectRoot ".venv")
}

& $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 or newer is required: $VenvPython"
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-lock.txt")
& $VenvPython -m pip install --no-deps --no-build-isolation -e $ProjectRoot
& $VenvPython -m pytest
````

# scripts/setup.sh

````bash
#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/conda_env.sh"

if [[ "${SKIP_DEPS:-0}" == "1" ]]; then
    # Use an already-provisioned cluster environment without changing its packages.
    "${environment_python}" -m pip install --no-deps --no-build-isolation -e "${project_root}"
elif [[ "${USE_LOCK:-0}" == "1" ]]; then
    "${environment_python}" -m pip install --upgrade pip
    "${environment_python}" -m pip install "setuptools>=75" wheel
    "${environment_python}" -m pip install -r "${project_root}/requirements-lock.txt"
    "${environment_python}" -m pip install --no-deps --no-build-isolation -e "${project_root}"
else
    "${environment_python}" -m pip install --upgrade pip
    "${environment_python}" -m pip install "setuptools>=75" wheel
    # Flexible minimum versions from pyproject.toml are the portable default.
    # A preinstalled CUDA-specific PyTorch satisfying torch>=2.2 is preserved.
    "${environment_python}" -m pip install --no-build-isolation -e "${project_root}[dev]"
fi

cd "${project_root}"
"${environment_python}" -m pytest -q
````

# scripts/setup_gpu.sh

````bash
#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "scripts/setup_gpu.sh targets the Linux host reached through MobaXterm." >&2
    exit 2
fi
source "${project_root}/scripts/conda_env.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; request a GPU node before running setup_gpu.sh." >&2
    exit 2
fi
nvidia-smi -L

cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
if [[ -z "${cuda_version}" ]]; then
    echo "Could not read the driver CUDA compatibility from nvidia-smi." >&2
    exit 2
fi
cuda_major="${cuda_version%%.*}"
cuda_minor="${cuda_version#*.}"
cuda_minor="${cuda_minor%%.*}"
driver_cuda_code=$((10#${cuda_major} * 100 + 10#${cuda_minor}))

wheel_tag="${CUDA_WHEEL_TAG:-}"
if [[ -z "${wheel_tag}" ]]; then
    if (( driver_cuda_code >= 1302 )); then
        wheel_tag="cu132"
    elif (( driver_cuda_code >= 1300 )); then
        wheel_tag="cu130"
    elif (( driver_cuda_code >= 1206 )); then
        wheel_tag="cu126"
    else
        echo "The locked torch 2.13 stack requires a driver supporting CUDA 12.6+." >&2
        echo "Driver compatibility reported by nvidia-smi: CUDA ${cuda_version}." >&2
        echo "This script will not silently install an older cu118 PyTorch release." >&2
        exit 2
    fi
fi

case "${wheel_tag}" in
    cu126) required_cuda_code=1206; expected_cuda_runtime="12.6" ;;
    cu130) required_cuda_code=1300; expected_cuda_runtime="13.0" ;;
    cu132) required_cuda_code=1302; expected_cuda_runtime="13.2" ;;
    *)
        echo "Unsupported CUDA_WHEEL_TAG=${wheel_tag}; choose cu126, cu130, or cu132." >&2
        exit 2
        ;;
esac
if (( driver_cuda_code < required_cuda_code )); then
    echo "CUDA_WHEEL_TAG=${wheel_tag} needs CUDA ${expected_cuda_runtime}+ driver compatibility." >&2
    echo "nvidia-smi reports CUDA ${cuda_version}." >&2
    exit 2
fi

constraints_file="${project_root}/constraints-${wheel_tag}.txt"
lock_file="${project_root}/requirements-lock.txt"
torch_index_url="https://download.pytorch.org/whl/${wheel_tag}"
if [[ ! -f "${constraints_file}" || ! -f "${lock_file}" ]]; then
    echo "GPU lock files are missing: ${constraints_file} or ${lock_file}" >&2
    exit 2
fi

if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
    "${environment_python}" -m pip install --upgrade pip
    "${environment_python}" -m pip install "setuptools>=75" wheel
    torch_version="$(sed -n 's/^torch==//p' "${constraints_file}")"
    if [[ -z "${torch_version}" || "${torch_version}" == *$'\n'* ]]; then
        echo "${constraints_file} must contain exactly one torch==version pin." >&2
        exit 2
    fi
    echo "Installing torch==${torch_version} from ${torch_index_url}"
    "${environment_python}" -m pip install --upgrade \
        --constraint "${constraints_file}" \
        "torch==${torch_version}" \
        --index-url "${torch_index_url}"
    "${environment_python}" -m pip install \
        --constraint "${constraints_file}" \
        --requirement "${lock_file}"
    "${environment_python}" -m pip install \
        --no-deps --no-build-isolation -e "${project_root}"
else
    "${environment_python}" -m pip install --no-deps --no-build-isolation -e "${project_root}"
fi

cd "${project_root}"
"${environment_python}" -m pip check
snapshot_dir="${ENVIRONMENT_SNAPSHOT_DIR:-${project_root}}"
mkdir -p "${snapshot_dir}"
lock_report="${snapshot_dir}/.gpu-environment.json"
freeze_report="${snapshot_dir}/.gpu-environment.freeze.txt"
"${environment_python}" scripts/verify_gpu_lock.py \
    --lock "${lock_file}" \
    --constraints "${constraints_file}" \
    --cuda-tag "${wheel_tag}" \
    --json-out "${lock_report}"
freeze_temporary="$(mktemp "${freeze_report}.tmp.XXXXXX")"
"${environment_python}" -m pip freeze --all > "${freeze_temporary}"
mv -f "${freeze_temporary}" "${freeze_report}"
"${environment_python}" scripts/gpu_preflight.py \
    --device "${DEVICE:-cuda}" \
    --require-paper-deps \
    --min-free-gb "${MIN_FREE_GB:-2}"

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    "${environment_python}" -m pytest -q
fi

echo "Exact environment report: ${lock_report}"
echo "Resolved transitive snapshot: ${freeze_report}"
echo "GPU environment ready. Run: bash scripts/paper.sh --help"
````

# scripts/smoke.ps1

````powershell
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:MPLCONFIGDIR = Join-Path $ProjectRoot ".matplotlib"
$env:PYTHONPATH = "$(Join-Path $ProjectRoot 'src');$ProjectRoot"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}

& $Python (Join-Path $ProjectRoot "scripts\run_all.py") @args
````

# scripts/smoke.sh

````bash
#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/conda_env.sh"

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"

"${environment_python}" scripts/run_all.py --device "${DEVICE:-auto}" "$@"
````

# scripts/verify_conda_env.py

````python
#!/usr/bin/env python3
"""Reject base, nested venvs, and mismatched Python before any package installation."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


class CondaEnvironmentError(RuntimeError):
    """A Bash entrypoint is not using a dedicated active Conda environment."""


def _conda_base(environ: Mapping[str, str]) -> Path:
    command = environ.get("CONDA_EXE") or "conda"
    try:
        result = subprocess.run(
            [command, "info", "--base"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CondaEnvironmentError(
            "Cannot query 'conda info --base'; initialize Conda and activate new-gat again."
        ) from error
    if not result.stdout.strip():
        raise CondaEnvironmentError("'conda info --base' returned an empty path.")
    return Path(result.stdout.strip()).resolve()


def verify_conda_environment(environ: Mapping[str, str] | None = None) -> Path:
    """Validate the current interpreter without installing or creating anything."""

    environ = os.environ if environ is None else environ
    raw_prefix = environ.get("CONDA_PREFIX", "")
    if not raw_prefix:
        raise CondaEnvironmentError("No active Conda environment; run 'conda activate new-gat'.")
    if environ.get("CONDA_DEFAULT_ENV") == "base":
        raise CondaEnvironmentError("Do not use Conda base; create and activate new-gat first.")
    if environ.get("VIRTUAL_ENV"):
        raise CondaEnvironmentError(
            "A venv is still active; deactivate it before activating the Conda environment."
        )

    expected = Path(raw_prefix).resolve()
    if not (expected / "conda-meta").is_dir():
        raise CondaEnvironmentError(f"Not a Conda environment (conda-meta is missing): {expected}")
    if sys.version_info < (3, 11):  # noqa: UP036 - this runs before package installation
        raise CondaEnvironmentError(
            "Python 3.11 or newer is required; create new-gat with Python 3.11."
        )
    if Path(sys.prefix).resolve() != expected:
        raise CondaEnvironmentError("Python sys.prefix does not match the active CONDA_PREFIX.")
    if Path(sys.base_prefix).resolve() != expected:
        raise CondaEnvironmentError(
            "Nested venv Python is not supported; use Conda's Python directly."
        )
    expected_python = (expected / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise CondaEnvironmentError("The interpreter is not CONDA_PREFIX/bin/python.")
    if expected == _conda_base(environ):
        raise CondaEnvironmentError(
            "Do not install into Conda base; activate a dedicated environment."
        )
    return expected


def main() -> int:
    try:
        prefix = verify_conda_environment()
    except (CondaEnvironmentError, OSError) as error:
        print(f"CONDA ENVIRONMENT CHECK FAILED: {error}", file=sys.stderr)
        return 2
    print(f"Conda environment: {prefix}")
    print(f"Python: {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

# scripts/verify_gpu_lock.py

````python
#!/usr/bin/env python3
"""Verify the exact GPU research stack selected by ``setup_gpu.sh``."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


class LockVerificationError(RuntimeError):
    """The installed environment does not satisfy the selected GPU lock."""


CUDA_RUNTIMES = {"cu126": "12.6", "cu130": "13.0", "cu132": "13.2"}
REQUIRED_RESEARCH_PACKAGES = {
    "networkx",
    "numpy",
    "ogb",
    "pandas",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "torch",
    "torch-geometric",
}
IMPORT_NAMES = {
    "networkx": "networkx",
    "numpy": "numpy",
    "ogb": "ogb",
    "pandas": "pandas",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "torch-geometric": "torch_geometric",
}


def canonical_name(name: str) -> str:
    """Return the distribution-name normalization used for lock comparison."""

    return re.sub(r"[-_.]+", "-", name).lower()


def read_exact_pins(path: Path) -> dict[str, str]:
    """Read a deliberately simple constraints file containing only ``name==version``."""

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise LockVerificationError(
                f"{path}:{line_number} is not an exact name==version pin: {line!r}"
            )
        raw_name, version = (part.strip() for part in line.split("==", 1))
        name = canonical_name(raw_name)
        if not name or not version or any(character.isspace() for character in version):
            raise LockVerificationError(f"{path}:{line_number} has an invalid exact pin")
        if name in pins:
            raise LockVerificationError(f"{path}:{line_number} duplicates {name}")
        pins[name] = version
    missing = sorted(REQUIRED_RESEARCH_PACKAGES - pins.keys())
    if missing:
        raise LockVerificationError(f"{path} is missing required pins: {', '.join(missing)}")
    return pins


def assert_same_pins(lock_path: Path, constraints_path: Path) -> dict[str, str]:
    """Reject drift between the portable lock and a CUDA-specific constraints file."""

    lock_pins = read_exact_pins(lock_path)
    constraint_pins = read_exact_pins(constraints_path)
    if lock_pins != constraint_pins:
        missing = sorted(lock_pins.keys() - constraint_pins.keys())
        extra = sorted(constraint_pins.keys() - lock_pins.keys())
        changed = sorted(
            name
            for name in lock_pins.keys() & constraint_pins.keys()
            if lock_pins[name] != constraint_pins[name]
        )
        raise LockVerificationError(
            "CUDA constraints drift from requirements-lock.txt: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return constraint_pins


def version_matches(name: str, expected: str, actual: str) -> bool:
    """Allow only the official CUDA local suffix on the pinned torch version."""

    if name == "torch":
        return actual.split("+", 1)[0] == expected
    return actual == expected


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def verify_environment(*, lock_path: Path, constraints_path: Path, cuda_tag: str) -> dict[str, Any]:
    """Check package versions, import-time ABI health, and the CUDA runtime."""

    if cuda_tag not in CUDA_RUNTIMES:
        raise LockVerificationError(f"unsupported CUDA wheel tag: {cuda_tag}")
    expected_pins = assert_same_pins(lock_path, constraints_path)

    installed: dict[str, str] = {}
    mismatches: list[str] = []
    for name, expected in sorted(expected_pins.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name}: missing (expected {expected})")
            continue
        installed[name] = actual
        if not version_matches(name, expected, actual):
            mismatches.append(f"{name}: installed {actual}, expected {expected}")
    if mismatches:
        raise LockVerificationError("exact package assertion failed: " + "; ".join(mismatches))

    import_errors: list[str] = []
    for distribution, module in IMPORT_NAMES.items():
        try:
            importlib.import_module(module)
        except Exception as error:  # binary dependencies can fail at import time
            import_errors.append(f"{distribution}: {type(error).__name__}: {error}")
    if import_errors:
        raise LockVerificationError("paper dependency import failed: " + "; ".join(import_errors))

    torch = importlib.import_module("torch")
    expected_runtime = CUDA_RUNTIMES[cuda_tag]
    actual_runtime = str(torch.version.cuda)
    if actual_runtime != expected_runtime:
        raise LockVerificationError(
            f"torch CUDA runtime is {actual_runtime}, expected {expected_runtime} for {cuda_tag}"
        )
    if not torch.cuda.is_available():
        raise LockVerificationError("torch.cuda.is_available() is false")

    return {
        "status": "passed",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_wheel_tag": cuda_tag,
        "torch_cuda_runtime": actual_runtime,
        "constraints_path": str(constraints_path.resolve()),
        "constraints_sha256": hashlib.sha256(constraints_path.read_bytes()).hexdigest(),
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "installed_top_level_versions": installed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--cuda-tag", choices=sorted(CUDA_RUNTIMES), required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = verify_environment(
            lock_path=args.lock,
            constraints_path=args.constraints,
            cuda_tag=args.cuda_tag,
        )
    except (LockVerificationError, OSError) as error:
        print(f"GPU LOCK VERIFICATION FAILED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out is not None:
        _write_json_atomic(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

# src/chartgat/__init__.py

````python
"""Shared incidence and graph-algebra primitives for independent tracks."""

from .algebra import (
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

__all__ = [
    "chart_transition",
    "decode_edge_state",
    "encode_edge_state",
    "flip_cycle_basis",
    "flip_edge_quantity",
    "flip_incidence",
    "fundamental_cycle_basis",
    "incidence_matrix",
    "orthonormal_cycle_basis",
    "validate_spanning_tree",
]
````

# src/chartgat/algebra.py

````python
"""Linear-algebra utilities for gradient--cycle graph coordinates.

The incidence convention throughout this module is ``B.shape == (m, n)``:
each edge is a row, with ``-1`` at its tail and ``+1`` at its head.  Thus
``B @ p`` is an edge gradient and ``ker(B.T)`` is the circulation space.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def incidence_matrix(
    num_nodes: int,
    edges: Sequence[tuple[int, int]] | Iterable[tuple[int, int]],
    *,
    dtype: np.dtype | type = np.float64,
) -> NDArray:
    """Construct an oriented edge-by-node incidence matrix.

    Parameters
    ----------
    num_nodes:
        Number of graph vertices. Vertices are indexed from ``0``.
    edges:
        Directed representatives ``(tail, head)``. For an undirected graph the
        direction is merely an orientation gauge. Parallel edges and self-loops
        are accepted; a self-loop produces a zero incidence row.
    """

    if not isinstance(num_nodes, (int, np.integer)) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")

    edge_list = list(edges)
    B = np.zeros((len(edge_list), int(num_nodes)), dtype=dtype)
    for edge_index, edge in enumerate(edge_list):
        if len(edge) != 2:
            raise ValueError(f"edge {edge_index} must contain (tail, head)")
        tail, head = edge
        if not isinstance(tail, (int, np.integer)) or not isinstance(head, (int, np.integer)):
            raise TypeError("edge endpoints must be integer node indices")
        if not (0 <= tail < num_nodes and 0 <= head < num_nodes):
            raise ValueError(f"edge {edge_index} has an endpoint out of range")
        B[edge_index, int(tail)] -= 1
        B[edge_index, int(head)] += 1
    return B


def _as_incidence(B: ArrayLike, *, atol: float) -> FloatArray:
    matrix = np.asarray(B, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("B must be a two-dimensional edge-by-node matrix")
    if matrix.shape[1] < 1:
        raise ValueError("B must have at least one node column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("B must contain only finite values")
    if not np.allclose(matrix.sum(axis=1), 0.0, atol=atol, rtol=0.0):
        raise ValueError("each incidence row must sum to zero")
    return matrix


def validate_spanning_tree(
    B: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
    *,
    atol: float = 1e-10,
) -> IntArray:
    """Validate and return the edge indices of a spanning tree.

    For an incidence matrix, selecting ``n - 1`` rows of rank ``n - 1`` is
    equivalent to selecting a spanning tree. This rank formulation also handles
    parallel edges without reconstructing an adjacency list.
    """

    matrix = _as_incidence(B, atol=atol)
    m, n = matrix.shape
    raw = np.asarray(tree_edge_indices)
    if raw.ndim != 1:
        raise ValueError("tree_edge_indices must be one-dimensional")
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.equal(raw, np.floor(raw))):
            raise TypeError("tree edge indices must be integers")
    tree = raw.astype(np.int64, copy=False)
    if tree.size != n - 1:
        raise ValueError(f"a spanning tree on {n} nodes must contain {n - 1} edges")
    if np.unique(tree).size != tree.size:
        raise ValueError("tree edge indices must be unique")
    if np.any(tree < 0) or np.any(tree >= m):
        raise ValueError("tree edge index out of range")

    if n > 1:
        singular_values = np.linalg.svd(matrix[tree], compute_uv=False)
        rank = int(np.count_nonzero(singular_values > atol * max(matrix.shape)))
        if rank != n - 1:
            raise ValueError("selected edges do not form a spanning tree")
    return tree.copy()


def fundamental_cycle_basis(
    B: ArrayLike,
    tree_edge_indices: Sequence[int] | ArrayLike,
    *,
    return_chords: bool = False,
    atol: float = 1e-10,
) -> FloatArray | tuple[FloatArray, IntArray]:
    """Build the fundamental cycle basis associated with a spanning tree.

    Chords are ordered by their original edge index. The returned ``F`` obeys

    ``B.T @ F == 0`` and ``F[chord_edge_indices, :] == I``.

    Consequently, the physical circulation ``z = F @ a`` is encoded simply by
    reading its values on the chord edges: ``a = z[chord_edge_indices]``.
    """

    matrix = _as_incidence(B, atol=atol)
    m, n = matrix.shape
    tree = validate_spanning_tree(matrix, tree_edge_indices, atol=atol)
    tree_set = set(tree.tolist())
    chords = np.asarray([i for i in range(m) if i not in tree_set], dtype=np.int64)
    beta = chords.size
    F = np.zeros((m, beta), dtype=np.float64)

    if beta:
        tree_incidence_transpose = matrix[tree].T
        for column, chord in enumerate(chords):
            tree_values, *_ = np.linalg.lstsq(tree_incidence_transpose, -matrix[chord], rcond=None)
            rounded = np.rint(tree_values)
            if np.allclose(tree_values, rounded, atol=atol, rtol=0.0):
                tree_values = rounded
            F[tree, column] = tree_values
            F[chord, column] = 1.0

    residual = matrix.T @ F
    scale = max(1.0, float(np.linalg.norm(matrix) * np.linalg.norm(F)))
    if not np.allclose(residual, 0.0, atol=atol * scale, rtol=0.0):
        raise RuntimeError("failed to construct a circulation basis")
    if beta and np.linalg.matrix_rank(F, tol=atol) != beta:
        raise RuntimeError("constructed fundamental cycles are rank deficient")

    if return_chords:
        return F, chords
    return F


def chart_transition(
    source_basis: ArrayLike,
    target_basis: ArrayLike,
    *,
    atol: float = 1e-10,
) -> FloatArray:
    """Return coordinates mapping a source cycle chart into a target chart.

    If ``z = F_source @ a_source = F_target @ a_target``, this function returns
    ``M`` such that ``a_target = M @ a_source`` and
    ``F_target @ M == F_source``. For fundamental cycle charts, ``M`` is a
    unimodular integer matrix (up to floating-point representation).
    """

    source = np.asarray(source_basis, dtype=np.float64)
    target = np.asarray(target_basis, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("cycle bases must be two-dimensional")
    if source.shape != target.shape:
        raise ValueError("source and target bases must have identical shape")
    beta = source.shape[1]
    if beta == 0:
        return np.empty((0, 0), dtype=np.float64)
    if (
        np.linalg.matrix_rank(source, tol=atol) != beta
        or np.linalg.matrix_rank(target, tol=atol) != beta
    ):
        raise ValueError("cycle bases must have full column rank")

    M, *_ = np.linalg.lstsq(target, source, rcond=None)
    scale = max(1.0, float(np.linalg.norm(source)))
    if not np.allclose(target @ M, source, atol=atol * scale, rtol=0.0):
        raise ValueError("source and target do not span the same cycle space")
    rounded = np.rint(M)
    if np.allclose(M, rounded, atol=atol * max(1, beta), rtol=0.0):
        M = rounded
    return M


def encode_edge_state(
    B: ArrayLike,
    cycle_basis: ArrayLike,
    edge_state: ArrayLike,
    *,
    atol: float = 1e-10,
) -> tuple[FloatArray, FloatArray]:
    """Encode ``e`` as ``e = B @ p + F @ a`` using the mean-zero potential.

    Vector-valued edge features are supported: ``edge_state`` may have shape
    ``(m,)`` or ``(m, d)``. Since the two subspaces are orthogonal in the
    unweighted Euclidean metric, the minimum-norm least-squares potential fixes
    the additive potential gauge automatically.
    """

    matrix = _as_incidence(B, atol=atol)
    F = np.asarray(cycle_basis, dtype=np.float64)
    edge = np.asarray(edge_state, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != matrix.shape[0]:
        raise ValueError("cycle_basis must have shape (num_edges, beta)")
    if edge.ndim not in (1, 2) or edge.shape[0] != matrix.shape[0]:
        raise ValueError("edge_state must have shape (num_edges,) or (num_edges, d)")
    if not np.allclose(matrix.T @ F, 0.0, atol=atol, rtol=0.0):
        raise ValueError("cycle_basis columns must lie in ker(B.T)")

    p, *_ = np.linalg.lstsq(matrix, edge, rcond=None)
    cycle_state = edge - matrix @ p
    if F.shape[1]:
        a, *_ = np.linalg.lstsq(F, cycle_state, rcond=None)
    else:
        trailing_shape = edge.shape[1:] if edge.ndim == 2 else ()
        a = np.empty((0, *trailing_shape), dtype=np.float64)

    reconstruction = decode_edge_state(matrix, F, p, a)
    scale = max(1.0, float(np.linalg.norm(edge)))
    if not np.allclose(reconstruction, edge, atol=atol * scale, rtol=0.0):
        raise ValueError("B and cycle_basis do not span the supplied edge state")
    return np.asarray(p), np.asarray(a)


def decode_edge_state(
    B: ArrayLike, cycle_basis: ArrayLike, potential: ArrayLike, cycle_coordinates: ArrayLike
) -> FloatArray:
    """Decode gradient--cycle coordinates as ``B @ p + F @ a``."""

    matrix = np.asarray(B, dtype=np.float64)
    F = np.asarray(cycle_basis, dtype=np.float64)
    p = np.asarray(potential, dtype=np.float64)
    a = np.asarray(cycle_coordinates, dtype=np.float64)
    if matrix.ndim != 2 or F.ndim != 2 or matrix.shape[0] != F.shape[0]:
        raise ValueError("B and cycle_basis must share their edge dimension")
    try:
        return np.asarray(matrix @ p + F @ a)
    except ValueError as exc:
        raise ValueError("incompatible coordinate shapes") from exc


def orthonormal_cycle_basis(cycle_basis: ArrayLike, *, atol: float = 1e-10) -> FloatArray:
    """Orthonormalize a full-rank cycle basis without changing its span."""

    F = np.asarray(cycle_basis, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("cycle_basis must be two-dimensional")
    m, beta = F.shape
    if beta == 0:
        return np.empty((m, 0), dtype=np.float64)
    if np.linalg.matrix_rank(F, tol=atol) != beta:
        raise ValueError("cycle_basis must have full column rank")
    U, _ = np.linalg.qr(F, mode="reduced")
    # Remove QR's arbitrary sign choice for deterministic tests and artifacts.
    for column in range(beta):
        pivot = int(np.argmax(np.abs(U[:, column])))
        if U[pivot, column] < 0:
            U[:, column] *= -1
    return U


def _orientation_signs(signs: ArrayLike, num_edges: int) -> FloatArray:
    values = np.asarray(signs, dtype=np.float64)
    if values.shape != (num_edges,):
        raise ValueError("orientation signs must have shape (num_edges,)")
    if not np.all(np.isin(values, (-1.0, 1.0))):
        raise ValueError("orientation signs must all equal -1 or +1")
    return values


def flip_incidence(B: ArrayLike, signs: ArrayLike) -> FloatArray:
    """Apply independent edge-orientation flips, equivalently ``Q @ B``."""

    matrix = np.asarray(B, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("B must be two-dimensional")
    values = _orientation_signs(signs, matrix.shape[0])
    return values[:, None] * matrix


def flip_edge_quantity(edge_quantity: ArrayLike, signs: ArrayLike) -> FloatArray:
    """Transform an orientation-covariant edge vector or feature matrix."""

    edge = np.asarray(edge_quantity, dtype=np.float64)
    if edge.ndim not in (1, 2):
        raise ValueError("edge_quantity must be a vector or matrix")
    values = _orientation_signs(signs, edge.shape[0])
    if edge.ndim == 1:
        return values * edge
    return values[:, None] * edge


def flip_cycle_basis(cycle_basis: ArrayLike, signs: ArrayLike) -> FloatArray:
    """Transform physical cycle columns under an edge-orientation flip."""

    F = np.asarray(cycle_basis, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("cycle_basis must be two-dimensional")
    values = _orientation_signs(signs, F.shape[0])
    return values[:, None] * F
````

# src/chartgat/cache.py

````python
"""Crash-safe helpers for publishing immutable dataset-cache files."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


class CacheValidationError(RuntimeError):
    """Base class for a cache that exists but is not usable."""


class CacheIncompleteError(CacheValidationError):
    """A multi-file cache is only partially present."""


class CacheWrongRequestError(CacheValidationError):
    """A cache belongs to a different seed, profile, schema, or source."""


class CacheCorruptError(CacheValidationError):
    """A cache cannot be parsed or fails its integrity contract."""


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after a rename.

    Windows does not expose a portable directory ``fsync``.  The file itself is
    always synced; on platforms supporting directory descriptors, the rename is
    synced as well.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_publish(
    path: Path,
    writer: Callable[[Path], None],
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Write, sync, optionally validate, and atomically publish one file.

    The temporary file is unique and located beside the destination, ensuring
    that ``os.replace`` cannot cross filesystems.  A failed writer or validator
    leaves the previous destination untouched.
    """

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        # Windows requires a writable descriptor for ``FlushFileBuffers``
        # (which backs ``os.fsync``); the writer has already finished here.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if validator is not None:
            validator(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Atomically publish bytes after an optional read-only validation."""

    def write(temporary: Path) -> None:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()

    atomic_publish(path, write, validator=validator)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = True,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Serialize JSON deterministically and atomically publish it."""

    content = (
        json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, content, validator=validator)


__all__ = [
    "CacheCorruptError",
    "CacheIncompleteError",
    "CacheValidationError",
    "CacheWrongRequestError",
    "atomic_publish",
    "atomic_write_bytes",
    "atomic_write_json",
]
````

# src/chartgat/graphs.py

````python
"""Small deterministic graph generators used by the experiment CLIs."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

EdgeList = list[tuple[int, int]]


def make_connected_graph(
    num_nodes: int,
    extra_edges: int,
    *,
    seed: int = 0,
) -> EdgeList:
    """Generate a simple connected undirected graph with fixed orientations.

    A random recursive tree guarantees connectivity; additional edges are then
    sampled without replacement. Every undirected edge is oriented from the
    smaller to the larger node, leaving orientation randomization to explicit
    gauge tests.
    """

    if num_nodes < 2:
        raise ValueError("num_nodes must be at least two")
    max_extra = num_nodes * (num_nodes - 1) // 2 - (num_nodes - 1)
    if not 0 <= extra_edges <= max_extra:
        raise ValueError(f"extra_edges must lie in [0, {max_extra}]")

    rng = np.random.default_rng(seed)
    edge_set: set[tuple[int, int]] = set()
    for node in range(1, num_nodes):
        parent = int(rng.integers(0, node))
        edge_set.add((parent, node))

    candidates = [
        (u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes) if (u, v) not in edge_set
    ]
    if extra_edges:
        chosen = rng.choice(len(candidates), size=extra_edges, replace=False)
        edge_set.update(candidates[int(index)] for index in np.atleast_1d(chosen))
    return sorted(edge_set)


def spanning_tree_indices(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    mode: str = "bfs",
    seed: int = 0,
) -> NDArray[np.int64]:
    """Return edge indices for a BFS, DFS, or random-weight spanning tree."""

    if mode not in {"bfs", "dfs", "random"}:
        raise ValueError("mode must be one of: bfs, dfs, random")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for index, (u, v) in enumerate(edges):
        if u == v:
            continue
        adjacency[u].append((v, index))
        adjacency[v].append((u, index))

    if mode == "random":
        return _random_weight_tree(num_nodes, edges, seed=seed)

    selected: list[int] = []
    seen = {0}
    if mode == "bfs":
        frontier: deque[int] | list[int] = deque([0])
        pop = frontier.popleft
        push = frontier.append
    else:
        frontier = [0]
        pop = frontier.pop
        push = frontier.append

    while frontier:
        node = pop()
        neighbors = sorted(adjacency[node], reverse=(mode == "dfs"))
        for neighbor, edge_index in neighbors:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            selected.append(edge_index)
            push(neighbor)
    if len(seen) != num_nodes:
        raise ValueError("graph is disconnected")
    return np.asarray(selected, dtype=np.int64)


def _random_weight_tree(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    seed: int,
) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random(len(edges)))
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    selected: list[int] = []
    for edge_index in order:
        u, v = edges[int(edge_index)]
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue
        if rank[root_u] < rank[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        if rank[root_u] == rank[root_v]:
            rank[root_u] += 1
        selected.append(int(edge_index))
        if len(selected) == num_nodes - 1:
            break
    if len(selected) != num_nodes - 1:
        raise ValueError("graph is disconnected")
    return np.asarray(selected, dtype=np.int64)


__all__ = ["make_connected_graph", "spanning_tree_indices"]
````

# src/chartgat/seeds.py

````python
"""Explicit random-seed axes for reproducible graph experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SeedAxes:
    """Keep benchmark construction and estimator randomness independently auditable."""

    data: int
    split: int
    chart: int
    model: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} seed must be a non-negative integer")

    def to_manifest(self) -> dict[str, int]:
        return asdict(self)


def resolve_seed_axes(
    legacy_seed: int,
    *,
    data_seed: int | None = None,
    split_seed: int | None = None,
    chart_seed: int | None = None,
    model_seed: int | None = None,
) -> SeedAxes:
    """Resolve new independent axes while preserving standalone ``--seed`` compatibility.

    A missing data seed falls back to the legacy seed.  Split and chart seeds then
    default to that data seed, while the model seed defaults directly to the legacy
    seed.  The master paper runner passes every axis explicitly and therefore never
    relies on these compatibility fallbacks.
    """

    data = legacy_seed if data_seed is None else data_seed
    return SeedAxes(
        data=data,
        split=data if split_seed is None else split_seed,
        chart=data if chart_seed is None else chart_seed,
        model=legacy_seed if model_seed is None else model_seed,
    )


__all__ = ["SeedAxes", "resolve_seed_axes"]
````

# tests/test_aggregate_paper.py

````python
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.aggregate_paper import aggregate_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_keeps_data_axes_fixed_and_pairs_model_seeds(tmp_path: Path) -> None:
    commands = []
    for model_seed, full, edge in ((1, 0.2, 0.5), (2, 0.4, 0.7)):
        output = tmp_path / f"seed-{model_seed}"
        _write_json(
            output / "summary.json",
            {
                "configuration": {"epochs": 100, "batch_size": 16},
                "runtime": {"elapsed_seconds": 3.0},
                "seed_axes": {"data": 11, "split": 13, "chart": 17, "model": model_seed},
                "results": {
                    "core": {
                        "s1": {
                            "baselines": {
                                "full": {
                                    "unseen_graph_test": {
                                        "graph_macro_flux_relative_l2": full,
                                        "num_examples": 20,
                                    }
                                },
                                "edge_only": {
                                    "unseen_graph_test": {
                                        "graph_macro_flux_relative_l2": edge,
                                        "num_examples": 20,
                                    }
                                },
                            }
                        }
                    },
                    "public": {
                        "pascalvoc_sp": {
                            "baselines": {
                                "conductance_model": {
                                    "parameter_count": 1_234,
                                    "parameter_count_policy": "trainable_active_parameters_only",
                                },
                                "gcn": {
                                    "parameter_count": 9_999,
                                    "parameter_count_policy": "all_constructed_parameters",
                                },
                            }
                        }
                    },
                },
            },
        )
        commands.append(
            {
                "name": f"conductance_gat:core:model-seed-{model_seed}",
                "command": [
                    "python",
                    "-m",
                    "research.conductance_gat.paper",
                    "--suite",
                    "core",
                    "--model-seed",
                    str(model_seed),
                    "--data-seed",
                    "11",
                    "--split-seed",
                    "13",
                    "--chart-seed",
                    "17",
                ],
                "returncode": 0,
                "artifact_errors": [],
                "output": str(output),
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"run_id": "fixture", "status": "passed", "commands": commands})

    payload = aggregate_manifest(manifest, bootstrap_samples=100)

    assert payload["failed_commands"] == 0
    assert payload["metric_groups"] == 2
    assert payload["sample_rows"] == 4
    assert payload["efficiency_rows"] == 4
    assert payload["ignored_numeric_fields"] > 0
    with (tmp_path / "aggregate" / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    full_row = next(
        row
        for row in rows
        if row["metric"]
        == "results.core.s1.baselines.full.unseen_graph_test.graph_macro_flux_relative_l2"
    )
    assert float(full_row["mean"]) == pytest.approx(0.3)
    assert full_row["data_seed"] == "11"
    assert full_row["model_seeds"] == "1,2"
    with (tmp_path / "aggregate" / "paired.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["condition_left"] == "edge_only"
    assert pair["condition_right"] == "full"
    assert float(pair["mean"]) == pytest.approx(-0.3)
    assert pair["difference_definition"] == "right_minus_left"
    assert pair["effect_size_name"] == "paired_cohens_dz"
    assert pair["effect_size"] == ""
    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        sample = next(csv.DictReader(stream))
    assert Path(sample["artifact_path"]).is_absolute()
    with (tmp_path / "aggregate" / "efficiency.csv").open(encoding="utf-8", newline="") as stream:
        efficiency = list(csv.DictReader(stream))
    assert {row["metric"] for row in efficiency} == {
        "runtime.elapsed_seconds",
        "results.public.pascalvoc_sp.baselines.conductance_model.parameter_count",
    }
    assert all("batch_size" not in row["metric"] for row in efficiency)


def test_aggregate_preserves_failures_and_legacy_seed_axes(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    _write_json(
        output / "core" / "edge" / "no_pe" / "metrics.json",
        {
            "id_test": {"macro_normalized_mae": 1.25, "graphs": 20},
            "train": {"macro_normalized_mae": 0.25},
        },
    )
    _write_json(
        output / "core" / "edge" / "no_pe" / "runtime.json",
        {
            "total_train_evaluation_wall_seconds": 5.0,
            "peak_gpu_memory_bytes": 2_048,
            "batch_size": 16,
            "epochs_completed": 20,
        },
    )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "run_id": "legacy",
            "status": "failed",
            "commands": [
                {
                    "name": "cycle_pe:core:seed-7",
                    "command": ["python", "--suite", "core", "--seed", "7"],
                    "returncode": 0,
                    "artifact_errors": [],
                    "output": str(output),
                },
                {
                    "name": "tree_augmentation:seed-8",
                    "command": ["python", "--suite", "core", "--seed", "8"],
                    "returncode": 1,
                    "artifact_errors": ["CUDA out of memory"],
                    "output": str(tmp_path / "missing"),
                },
            ],
        },
    )

    payload = aggregate_manifest(manifest, bootstrap_samples=0)

    assert payload["failed_commands"] == 1
    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        sample = next(csv.DictReader(stream))
    assert sample["model_seed"] == sample["data_seed"] == sample["split_seed"] == "7"
    assert sample["metric"] == "id_test.macro_normalized_mae"
    assert payload["efficiency_rows"] == 2
    with (tmp_path / "aggregate" / "failures.csv").open(encoding="utf-8", newline="") as stream:
        failure = next(csv.DictReader(stream))
    assert failure["oom"] == "True"


def test_aggregate_reads_oom_logs_and_ignores_outer_seed_for_official_brec(
    tmp_path: Path,
) -> None:
    brec_output = tmp_path / "brec"
    _write_json(
        brec_output / "brec" / "no_pe" / "metrics.json",
        {
            "protocol": "official",
            "global_valid": True,
            "per_seed": {
                "100": {
                    "pairs_expected": 400,
                    "Correct": 12,
                    "Fail": 0,
                    "Real_correct": 12,
                }
            },
        },
    )
    oom_log = tmp_path / "oom.log"
    oom_log.write_text("torch.OutOfMemoryError: CUDA out of memory", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "run_id": "brec-and-oom",
            "status": "failed",
            "commands": [
                {
                    "name": "cycle_pe:brec:official-10-seed",
                    "command": [
                        "python",
                        "--suite",
                        "brec",
                        "--brec-protocol",
                        "official",
                        "--model-seed",
                        "0",
                        "--data-seed",
                        "3",
                    ],
                    "returncode": 0,
                    "artifact_errors": [],
                    "output": str(brec_output),
                },
                {
                    "name": "conductance_gat:model-seed-4",
                    "command": ["python", "--suite", "core", "--model-seed", "4"],
                    "returncode": 1,
                    "artifact_errors": [],
                    "log": str(oom_log),
                    "output": str(tmp_path / "missing"),
                },
            ],
        },
    )

    aggregate_manifest(manifest, bootstrap_samples=0)

    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 3
    assert {row["metric"] for row in samples} == {
        "per_seed.100.Correct",
        "per_seed.100.Fail",
        "per_seed.100.Real_correct",
    }
    assert all(row["model_seed"] == "" for row in samples)
    assert all(row["pairable"] == "False" for row in samples)
    with (tmp_path / "aggregate" / "failures.csv").open(encoding="utf-8", newline="") as stream:
        failure = next(csv.DictReader(stream))
    assert failure["oom"] == "True"


def test_aggregate_tree_schema_pairs_only_registered_downstream_metrics(tmp_path: Path) -> None:
    commands = []
    for model_seed, fixed, multi in ((1, 0.8, 0.5), (2, 0.6, 0.4)):
        output = tmp_path / f"tree-{model_seed}"
        _write_json(
            output / "summary.json",
            {
                "settings": {
                    "optimizer_updates": 100,
                    "batch_size": 32,
                    "seed_axes": {"model": model_seed},
                },
                "runtime": {"elapsed_seconds": 4.0, "peak_gpu_allocated_bytes": 1024},
                "models": {
                    "fixed_bfs": {
                        "optimizer_updates": 100,
                        "history": [{"update": 1, "loss": 2.0}],
                        "quadrants": {"ood": {"mae": fixed, "num_examples": 10}},
                    },
                    "multi_chart": {
                        "optimizer_updates": 100,
                        "history": [{"update": 1, "loss": 1.0}],
                        "quadrants": {"ood": {"mae": multi, "num_examples": 10}},
                    },
                },
                "comparison": {
                    "quadrant_improvements": {
                        "ood": {"mae_improvement_fixed_minus_multi": fixed - multi}
                    }
                },
                "diagnostics": {"mae": 999.0},
            },
        )
        commands.append(
            {
                "name": f"tree_augmentation:core:model-seed-{model_seed}",
                "command": [
                    "python",
                    "--suite",
                    "core",
                    "--model-seed",
                    str(model_seed),
                    "--data-seed",
                    "11",
                    "--split-seed",
                    "13",
                    "--chart-seed",
                    "17",
                ],
                "returncode": 0,
                "artifact_errors": [],
                "output": str(output),
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"run_id": "tree-schema", "status": "passed", "commands": commands})

    payload = aggregate_manifest(manifest, bootstrap_samples=0)

    assert payload["sample_rows"] == 6
    assert payload["efficiency_rows"] == 4
    assert payload["metric_groups"] == 3
    assert payload["paired_groups"] == 1
    assert payload["ignored_numeric_fields"] == payload["numeric_fields_seen"] - 10
    with (tmp_path / "aggregate" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert all("history" not in row["metric"] for row in samples)
    assert all("runtime" not in row["metric"] for row in samples)
    assert all("settings" not in row["metric"] for row in samples)
    improvements = [row for row in samples if row["metric_rule"] == "tree.precomputed_improvement"]
    assert len(improvements) == 2
    assert all(row["pairable"] == "False" for row in improvements)
    with (tmp_path / "aggregate" / "efficiency.csv").open(encoding="utf-8", newline="") as stream:
        efficiency = list(csv.DictReader(stream))
    assert {row["metric"] for row in efficiency} == {
        "runtime.elapsed_seconds",
        "runtime.peak_gpu_allocated_bytes",
    }
    assert all("batch_size" not in row["metric"] for row in efficiency)
````

# tests/test_algebra.py

````python
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
````

# tests/test_cache_io.py

````python
from __future__ import annotations

from pathlib import Path

import pytest

from chartgat.cache import atomic_write_bytes


def test_atomic_write_preserves_previous_file_when_validation_fails(tmp_path: Path) -> None:
    destination = tmp_path / "cache.bin"
    destination.write_bytes(b"previous-valid-cache")

    def reject(temporary: Path) -> None:
        assert temporary.read_bytes() == b"new-invalid-cache"
        raise ValueError("invalid temporary cache")

    with pytest.raises(ValueError, match="invalid temporary cache"):
        atomic_write_bytes(destination, b"new-invalid-cache", validator=reject)

    assert destination.read_bytes() == b"previous-valid-cache"
    assert list(tmp_path.glob(".cache.bin.*.tmp")) == []


def test_atomic_write_validates_then_replaces(tmp_path: Path) -> None:
    destination = tmp_path / "cache.bin"
    destination.write_bytes(b"old")
    atomic_write_bytes(
        destination,
        b"new",
        validator=lambda temporary: temporary.read_bytes() == b"new" or None,
    )
    assert destination.read_bytes() == b"new"
````

# tests/test_conda_env.py

````python
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_conda_env

ROOT = Path(__file__).resolve().parents[1]
BASH_ENTRYPOINTS = ("setup.sh", "setup_gpu.sh", "paper.sh", "smoke.sh")
BASH = shutil.which("bash")
LINUX_BASH_ONLY = pytest.mark.skipif(
    sys.platform != "linux" or BASH is None,
    reason="Dynamic shell contracts require Linux and Bash; unavailable on this local host.",
)


@pytest.fixture
def active_conda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    prefix = tmp_path / "Conda environments" / "new gat"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    python = prefix / "bin" / "python"
    python.touch()
    monkeypatch.setattr(
        verify_conda_env,
        "sys",
        SimpleNamespace(
            version_info=(3, 11, 9),
            prefix=str(prefix),
            base_prefix=str(prefix),
            executable=str(python),
        ),
    )
    monkeypatch.setattr(verify_conda_env, "_conda_base", lambda _environ: tmp_path / "base")
    return prefix, {"CONDA_PREFIX": str(prefix), "CONDA_DEFAULT_ENV": "new-gat"}


@pytest.mark.parametrize("environ", [{}, {"CONDA_PREFIX": ""}])
def test_missing_conda_prefix_is_rejected(environ: dict[str, str]) -> None:
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="No active Conda"):
        verify_conda_env.verify_conda_environment(environ)


def test_named_base_environment_is_rejected(active_conda: tuple[Path, dict[str, str]]) -> None:
    _, environ = active_conda
    environ["CONDA_DEFAULT_ENV"] = "base"
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Do not use Conda base"):
        verify_conda_env.verify_conda_environment(environ)


@pytest.mark.parametrize("default_env", [None, "not-named-base"])
def test_base_prefix_is_rejected_even_without_base_name(
    active_conda: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    default_env: str | None,
) -> None:
    prefix, environ = active_conda
    if default_env is None:
        environ.pop("CONDA_DEFAULT_ENV")
    else:
        environ["CONDA_DEFAULT_ENV"] = default_env
    monkeypatch.setattr(verify_conda_env, "_conda_base", lambda _environ: prefix.resolve())
    with pytest.raises(
        verify_conda_env.CondaEnvironmentError, match="Do not install into Conda base"
    ):
        verify_conda_env.verify_conda_environment(environ)


def test_virtual_env_is_rejected(active_conda: tuple[Path, dict[str, str]]) -> None:
    prefix, environ = active_conda
    environ["VIRTUAL_ENV"] = str(prefix / "nested")
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="venv is still active"):
        verify_conda_env.verify_conda_environment(environ)


def test_missing_conda_metadata_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="conda-meta is missing"):
        verify_conda_env.verify_conda_environment({"CONDA_PREFIX": str(tmp_path)})


@pytest.mark.parametrize("version", [(3, 9, 20), (3, 10, 15)])
def test_old_python_is_rejected(
    active_conda: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int, int],
) -> None:
    _, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "version_info", version)
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Python 3.11 or newer"):
        verify_conda_env.verify_conda_environment(environ)


def test_other_interpreter_prefix_is_rejected(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "prefix", str(prefix / "other"))
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="sys.prefix does not match"):
        verify_conda_env.verify_conda_environment(environ)


def test_nested_venv_is_rejected_without_virtual_env_variable(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "base_prefix", str(prefix / "other-base"))
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Nested venv Python"):
        verify_conda_env.verify_conda_environment(environ)


def test_other_executable_is_rejected(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.sys, "executable", str(prefix / "other" / "python"))
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="not CONDA_PREFIX/bin/python"):
        verify_conda_env.verify_conda_environment(environ)


def test_valid_conda_prefix_with_spaces_and_normalized_suffix(
    active_conda: tuple[Path, dict[str, str]],
) -> None:
    prefix, environ = active_conda
    environ["CONDA_PREFIX"] = f"{prefix}/."
    assert verify_conda_env.verify_conda_environment(environ) == prefix.resolve()


def test_verifier_defaults_to_process_environment(
    active_conda: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, environ = active_conda
    monkeypatch.setattr(verify_conda_env.os, "environ", environ)
    assert verify_conda_env.verify_conda_environment() == prefix.resolve()


@pytest.mark.parametrize("conda_exe", [None, "/Conda installation/bin/conda"])
def test_conda_base_queries_configured_executable_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conda_exe: str | None
) -> None:
    environ = {} if conda_exe is None else {"CONDA_EXE": conda_exe}
    base = tmp_path / "Conda installation"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"  {base}\n")

    monkeypatch.setattr(verify_conda_env.subprocess, "run", fake_run)
    assert verify_conda_env._conda_base(environ) == base.resolve()
    assert calls == [
        (
            [conda_exe or "conda", "info", "--base"],
            {"check": True, "capture_output": True, "text": True, "timeout": 30},
        )
    ]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("conda missing"),
        PermissionError("conda not executable"),
        subprocess.CalledProcessError(1, ["conda", "info", "--base"]),
        subprocess.TimeoutExpired(["conda", "info", "--base"], 30),
    ],
)
def test_conda_base_query_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(verify_conda_env.subprocess, "run", fail_run)
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="Cannot query") as caught:
        verify_conda_env._conda_base({})
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("stdout", ["", " \n\t"])
def test_conda_base_rejects_empty_result(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    monkeypatch.setattr(
        verify_conda_env.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout)
    )
    with pytest.raises(verify_conda_env.CondaEnvironmentError, match="empty path"):
        verify_conda_env._conda_base({})


@pytest.mark.parametrize("script_name", BASH_ENTRYPOINTS)
def test_bash_entrypoints_validate_conda_before_installation_or_dispatch(script_name: str) -> None:
    source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
    guard = 'source "${project_root}/scripts/conda_env.sh"'
    assert source.count(guard) == 1
    assert source.index("project_root=") < source.index(guard)
    assert source.index(guard) < source.index('"${environment_python}"')
    for forbidden in (".venv", "VENV_DIR", "USE_ACTIVE_ENV", "-m venv", "environment_python="):
        assert forbidden not in source
    assert not re.search(r"\$\{?PYTHON(?=[:}\s\"/]|$)", source)
    assert not re.search(r"^\s*conda\s+(?:create|install)\b", source, flags=re.MULTILINE)
    if script_name == "setup_gpu.sh":
        assert source.index(guard) < source.index("command -v nvidia-smi")
        assert source.index(guard) < source.index('mkdir -p "${snapshot_dir}"')


def test_shared_bash_guard_uses_only_conda_python_and_runs_verification() -> None:
    source = (ROOT / "scripts" / "conda_env.sh").read_text(encoding="utf-8")
    assert 'environment_python="${CONDA_PREFIX%/}/bin/python"' in source
    assert '[[ ! -x "${environment_python}" ]]' in source
    assert '"${environment_python}" "${project_root}/scripts/verify_conda_env.py"' in source
    for forbidden in (".venv", "VENV_DIR", "USE_ACTIVE_ENV", "-m pip", "-m venv"):
        assert forbidden not in source
    assert not re.search(r"^\s*conda\s+(?:create|install)\b", source, flags=re.MULTILINE)


def _shell_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Stub only dispatch; Python unit tests above validate real guard decisions."""

    prefix = tmp_path / "Conda environments" / "new gat"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  */verify_conda_env.py)\n"
        '    printf "verify\\n" >> "$TEST_CALL_LOG"\n'
        '    exit "${TEST_VERIFY_EXIT:-0}" ;;\n'
        "  scripts/run_paper.py)\n"
        '    printf "paper\\n" >> "$TEST_CALL_LOG"\n'
        '    printf "%s\\0" "$@" > "$TEST_DISPATCH_ARGS"\n'
        '    exit "${TEST_RUN_EXIT:-0}" ;;\n'
        '  *) printf "unexpected\\n" >> "$TEST_CALL_LOG"; exit 97 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    call_log = tmp_path / "calls.log"
    dispatch_args = tmp_path / "dispatch.args"
    environ = os.environ.copy()
    environ.pop("VIRTUAL_ENV", None)
    environ.update(
        {
            "CONDA_PREFIX": f"{prefix}/",
            "CONDA_DEFAULT_ENV": "new-gat",
            "TEST_CALL_LOG": str(call_log),
            "TEST_DISPATCH_ARGS": str(dispatch_args),
            "TEST_VERIFY_EXIT": "0",
            "TEST_RUN_EXIT": "0",
            "PYTHON": str(tmp_path / "wrong-python"),
            "VENV_DIR": str(tmp_path / "must-not-be-created"),
            "USE_ACTIVE_ENV": "0",
            "ENVIRONMENT_SNAPSHOT_DIR": str(tmp_path / "must-not-have-snapshots"),
        }
    )
    return environ, call_log, dispatch_args


@LINUX_BASH_ONLY
@pytest.mark.parametrize("script_name", BASH_ENTRYPOINTS)
@pytest.mark.parametrize("skip_deps", ["0", "1"])
def test_bash_guard_failure_stops_before_pip_or_dispatch(
    tmp_path: Path, script_name: str, skip_deps: str
) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    environ.update({"TEST_VERIFY_EXIT": "23", "SKIP_DEPS": skip_deps})
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["verify"]
    assert not dispatch_args.exists()
    assert not Path(environ["VENV_DIR"]).exists()
    assert not Path(environ["ENVIRONMENT_SNAPSHOT_DIR"]).exists()


@LINUX_BASH_ONLY
@pytest.mark.parametrize("script_name", BASH_ENTRYPOINTS)
def test_bash_without_active_conda_stops_before_invoking_python(
    tmp_path: Path, script_name: str
) -> None:
    environ, call_log, _ = _shell_environment(tmp_path)
    environ.pop("CONDA_PREFIX")
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / script_name), "--help"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2
    assert "No active Conda environment" in result.stderr
    assert not call_log.exists()


@LINUX_BASH_ONLY
def test_paper_bash_preserves_arguments_exit_code_and_conda_selection(tmp_path: Path) -> None:
    environ, call_log, dispatch_args = _shell_environment(tmp_path)
    environ["TEST_RUN_EXIT"] = "37"
    arguments = ["--run-id", "space value", "", "literal;$HOME", "--seeds", "1,2"]
    result = subprocess.run(
        [BASH, str(ROOT / "scripts" / "paper.sh"), *arguments],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 37, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["verify", "paper"]
    forwarded = dispatch_args.read_bytes().split(b"\0")[:-1]
    assert [value.decode("utf-8") for value in forwarded] == ["scripts/run_paper.py", *arguments]
    assert not Path(environ["VENV_DIR"]).exists()
````

# tests/test_dataset_plans.py

````python
from __future__ import annotations

import copy
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import check_datasets

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_datasets.py"


def _check(
    profile: str,
    cwd: Path,
    *,
    as_json: bool = False,
    data_root: Path | None = None,
    require_cache: bool = False,
    seeds: tuple[int, ...] = (0,),
    split_seeds: tuple[int, ...] | None = None,
    tiny: bool = False,
) -> SimpleNamespace:
    command = [str(CHECKER), "--profile", profile]
    if as_json:
        command.append("--json")
    if data_root is not None:
        command.extend(("--data-root", str(data_root)))
    if require_cache:
        command.append("--require-cache")
    command.extend(("--seeds", ",".join(str(seed) for seed in seeds)))
    if split_seeds is not None:
        command.extend(("--split-seeds", ",".join(str(seed) for seed in split_seeds)))
    if tiny:
        command.append("--tiny")
    stdout = StringIO()
    stderr = StringIO()
    with (
        patch.object(sys, "argv", command),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            return_code = check_datasets.main()
        except SystemExit as error:
            return_code = int(error.code)
    # Preserve the old subprocess-like test interface without starting a second
    # interpreter after PyTorch has initialized Windows worker threads.
    assert cwd.is_dir()
    return SimpleNamespace(
        returncode=return_code,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def test_smoke_dataset_profile_is_implemented(tmp_path: Path) -> None:
    result = _check("smoke", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "READY" in result.stdout
    assert "code=implemented" in result.stdout


def test_paper_dataset_profile_matches_complete_core_code(tmp_path: Path) -> None:
    result = _check("paper", tmp_path, as_json=True)
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["ready"] is True
    assert payload["code_ready"] is True
    assert payload["paper_benchmark_suite_complete"] is True
    assert all(row["tier"] == "paper_core" for row in payload["rows"])
    assert all(row["status"] == "implemented" for row in payload["rows"])
    assert all(row["cache_status"] == "not_checked" for row in payload["rows"])


def test_complete_flag_is_derived_only_from_required_core_status() -> None:
    registry = check_datasets.load_registry("conductance_gat")
    inconsistent = copy.deepcopy(registry)
    inconsistent["paper_suite_complete"] = False
    errors = check_datasets.validate_registry("conductance_gat", inconsistent)
    assert any("paper_suite_complete must be true" in error for error in errors)

    optional_change = copy.deepcopy(registry)
    optional_entry = next(
        entry for entry in optional_change["datasets"] if entry["tier"] == "optional"
    )
    optional_entry["status"] = "blocked"
    errors = check_datasets.validate_registry("conductance_gat", optional_change)
    assert not any("paper_suite_complete" in error for error in errors)


def test_code_readiness_does_not_claim_cache_presence(tmp_path: Path) -> None:
    empty_data_root = tmp_path / "empty-data"
    empty_data_root.mkdir()
    result = _check("paper", tmp_path, as_json=True, data_root=empty_data_root)
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["code_ready"] is True
    assert payload["cache_checked"] is True
    assert payload["cached_data_ready"] is False
    assert any(row["cache_status"] == "missing" for row in payload["rows"])


def test_require_cache_controls_ready_and_exit_status(tmp_path: Path) -> None:
    without_root = _check("paper", tmp_path, as_json=True, require_cache=True)
    assert without_root.returncode == 2
    assert "--require-cache requires --data-root" in without_root.stderr

    empty_data_root = tmp_path / "empty-data"
    empty_data_root.mkdir()
    missing = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=empty_data_root,
        require_cache=True,
    )
    missing_payload = json.loads(missing.stdout)
    assert missing.returncode == 2
    assert missing_payload["code_ready"] is True
    assert missing_payload["cached_data_ready"] is False
    assert missing_payload["ready"] is False

    registries = {track: check_datasets.load_registry(track) for track in check_datasets.TRACKS}
    for registry in registries.values():
        for entry in registry["datasets"]:
            if entry["tier"] != "paper_core":
                continue
            pattern = entry.get("cache_glob")
            if pattern is None:
                continue
            fixture_path = empty_data_root / pattern.replace("*", "fixture")
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text("fixture\n", encoding="utf-8")
    present = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=empty_data_root,
        require_cache=True,
    )
    present_payload = json.loads(present.stdout)
    assert present.returncode == 2
    assert present_payload["cached_data_ready"] is False
    assert present_payload["ready"] is False
    assert any(
        row["cache_status"] in {"missing", "incomplete", "corrupt", "wrong_request"}
        for row in present_payload["rows"]
    )


def _prepare_valid_tiny_paper_caches(data_root: Path, seed: int) -> None:
    import torch

    from research.conductance_gat.paper_data import prepare_core_cache
    from research.conductance_gat.public_data import prepare_public_data
    from research.cycle_pe.paper_adapters import write_tiny_brec_fixture
    from research.cycle_pe.paper_data import load_or_generate_cycle_count_ood
    from research.tree_augmentation.paper_data import (
        GraphRecord,
        _cache_records,
        prepare_cyclecount_dataset,
    )

    prepare_core_cache(data_root, seed=seed, tiny=True)
    prepare_public_data(data_root, seed=seed, tiny=True)
    load_or_generate_cycle_count_ood(data_root, seed=seed, tiny=True)
    write_tiny_brec_fixture(data_root / "cycle_pe_fixtures" / "brec_v3_q32.npy", num_relabel=32)
    processed = data_root / "ZINC12K" / "subset" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for split, count in {"train": 32, "val": 8, "test": 8}.items():
        torch.save(({}, {"x": torch.arange(count + 1)}), processed / f"{split}.pt")

    prepare_cyclecount_dataset(data_root, seed=seed, tiny=True)

    def records(counts: dict[str, int], *, zinc: bool) -> tuple[GraphRecord, ...]:
        result = []
        for split, count in counts.items():
            for index in range(count):
                result.append(
                    GraphRecord(
                        graph_id=f"fixture-{'zinc' if zinc else 'csl'}-{split}-{index}",
                        family="fixture",
                        split=split,
                        num_nodes=3,
                        edges=((0, 1), (0, 2), (1, 2)),
                        target=(float(index % 10) if not zinc else float(index) / 10.0,),
                        task_type="regression" if zinc else "classification",
                        x=(0, 1, 2) if zinc else None,
                        edge_attr=(0, 0, 0) if zinc else None,
                    )
                )
        return tuple(result)

    tree_specs = {
        "csl": (
            {"train": 30, "validation": 10, "test": 10},
            tuple(f"class_{index}" for index in range(10)),
            "classification",
            "PyG:GNNBenchmarkDataset/CSL",
        ),
        "zinc": (
            {"train": 32, "validation": 12, "test": 12},
            ("constrained_logP",),
            "regression",
            "PyG:ZINC(subset=True)",
        ),
    }
    for suite, (counts, target_names, task_type, source) in tree_specs.items():
        cache_dir = data_root / f"{suite}_pyg_v2"
        _cache_records(
            suite=suite,
            records=records(counts, zinc=suite == "zinc"),
            data_path=cache_dir / f"seed-{seed}-tiny.json",
            manifest_path=cache_dir / f"seed-{seed}-tiny.manifest.json",
            target_names=target_names,
            task_type=task_type,
            source=source,
            seed=seed,
            tiny=True,
        )


def test_require_cache_runs_content_validators_for_requested_tiny_seed(tmp_path: Path) -> None:
    data_root = tmp_path / "valid-data"
    seed = 7
    _prepare_valid_tiny_paper_caches(data_root, seed)
    result = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=data_root,
        require_cache=True,
        seeds=(seed,),
        tiny=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["cached_data_ready"] is True
    assert payload["requested_seeds"] == [seed]
    assert payload["requested_seed_axes"] == {"data": [seed], "split": [seed]}
    assert payload["tiny"] is True
    assert all(row["cache_status"] == "valid" for row in payload["rows"])


def test_require_cache_rejects_corrupt_requested_cache(tmp_path: Path) -> None:
    data_root = tmp_path / "corrupt-data"
    seed = 11
    _prepare_valid_tiny_paper_caches(data_root, seed)
    cycle_cache = next((data_root / "cycle_count_ood").glob("*.json.gz"))
    cycle_cache.write_bytes(b"truncated")
    result = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=data_root,
        require_cache=True,
        seeds=(seed,),
        tiny=True,
    )
    payload = json.loads(result.stdout)
    cycle_row = next(row for row in payload["rows"] if row["id"] == "cyclecount_ood")
    assert result.returncode == 2
    assert cycle_row["cache_status"] == "corrupt"


def test_validator_distinguishes_wrong_request_and_incomplete_cache(tmp_path: Path) -> None:
    from research.conductance_gat.paper_data import prepare_core_cache

    entry = next(
        item
        for item in check_datasets.load_registry("conductance_gat")["datasets"]
        if item["id"] == "static_multigraph_identification"
    )

    wrong_root = tmp_path / "wrong"
    _, manifest_path, manifest = prepare_core_cache(wrong_root, seed=3, tiny=True)
    manifest["request"]["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong = check_datasets._validate_cache(entry, wrong_root, seeds=(3,), tiny=True)
    assert wrong["cache_status"] == "wrong_request"

    incomplete_root = tmp_path / "incomplete"
    _, incomplete_manifest, _ = prepare_core_cache(incomplete_root, seed=3, tiny=True)
    incomplete_manifest.with_name("core.pt").unlink()
    incomplete = check_datasets._validate_cache(entry, incomplete_root, seeds=(3,), tiny=True)
    assert incomplete["cache_status"] == "incomplete"


def test_checker_routes_tree_csl_cache_to_split_seed_axis(tmp_path: Path) -> None:
    data_root = tmp_path / "independent-seed-axes"
    _prepare_valid_tiny_paper_caches(data_root, 11)
    _prepare_valid_tiny_paper_caches(data_root, 13)

    result = _check(
        "paper",
        tmp_path,
        as_json=True,
        data_root=data_root,
        require_cache=True,
        seeds=(11,),
        split_seeds=(13,),
        tiny=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["requested_seed_axes"] == {"data": [11], "split": [13]}
    csl = next(row for row in payload["rows"] if row["id"] == "csl_chart_sanity")
    tree_core = next(row for row in payload["rows"] if row["id"] == "cyclecount_ood_multichart")
    assert csl["cache_detail"]["requested_axis"] == "split"
    assert csl["cache_detail"]["requested_seeds"] == [13]
    assert tree_core["cache_detail"]["requested_axis"] == "data"
    assert tree_core["cache_detail"]["requested_seeds"] == [11]


def test_implemented_adapters_and_generated_sources_resolve() -> None:
    for track in check_datasets.TRACKS:
        registry = check_datasets.load_registry(track)
        errors = check_datasets.validate_registry(track, registry)
        assert not errors, "\n".join(errors)


def test_optional_is_a_tier_not_a_code_status() -> None:
    for track in check_datasets.TRACKS:
        registry = check_datasets.load_registry(track)
        for entry in registry["datasets"]:
            assert entry["tier"] in check_datasets.ALLOWED_TIERS
            assert entry["status"] in check_datasets.ALLOWED_STATUSES
            assert entry["data_policy"] in check_datasets.ALLOWED_DATA_POLICIES
            assert entry["status"] not in {"optional", "implemented_optional"}
````

# tests/test_gpu_preflight.py

````python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.gpu_preflight import (
    PROFILE_NAMES,
    PreflightError,
    ProfileConfig,
    _normalize_profiles,
    _paper_dependency_import_errors,
    build_report,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "gpu_preflight.py"


def test_cpu_preflight_exercises_incidence_path() -> None:
    report = build_report(
        "cpu",
        allow_cpu=True,
        require_paper_dependencies=False,
        min_free_gb=0.0,
    )
    assert report["status"] == "passed"
    assert report["resolved_device"] == "cpu"
    assert report["gpu"] is None
    assert report["profile_kind"] == "synthetic_shape_stress_not_dataset_e2e"
    assert report["selected_profiles"] == ["conductance"]
    assert set(report["profiles"]) == {"conductance"}
    assert report["incidence_forward_backward"]["loss"] > 0.0
    assert report["incidence_forward_backward"]["message_sum_abs"] < 1.0e-3
    assert report["profiles"]["conductance"]["peak_allocated"] == 0


def test_cpu_suite_profiles_cover_dense_forward_backward_and_brec_pinv() -> None:
    selected = ("cycle-projector", "tree-chart", "brec")
    report = build_report(
        "cpu",
        allow_cpu=True,
        require_paper_dependencies=False,
        min_free_gb=0.0,
        profiles=selected,
        profile_config=ProfileConfig(
            batch_size=3,
            brec_batch_size=3,
            nodes_per_graph=10,
            edges_per_graph=16,
            cycle_rank=5,
            cycle_variants=("no_pe", "raw", "set", "projector"),
            brec_protocol="custom",
            brec_amp=True,
        ),
    )
    assert tuple(report["profiles"]) == selected
    for profile in report["profiles"].values():
        assert profile["status"] == "passed"
        assert profile["loss"] >= 0.0
        assert profile["memory_unit"] == "bytes"
        assert profile["wall_time_unit"] == "seconds"
        assert profile["wall_time"] > 0.0
        assert {
            "allocated",
            "reserved",
            "peak_allocated",
            "peak_reserved",
        } <= set(profile)
    assert report["profiles"]["cycle-projector"]["spec"]["projector_shape_per_graph"] == [
        16,
        16,
    ]
    assert report["profiles"]["cycle-projector"]["spec"]["selected_variants"] == [
        "no_pe",
        "raw",
        "set",
        "projector",
    ]
    assert set(report["profiles"]["cycle-projector"]["variant_losses"]) == {
        "no_pe",
        "raw",
        "set",
        "projector",
    }
    assert report["profiles"]["tree-chart"]["spec"]["dense_chart_shape"] == [2, 16, 5]
    assert report["profiles"]["brec"]["spec"]["dtype"] == "float32"
    assert report["profiles"]["brec"]["spec"]["amp"] is False
    assert report["profiles"]["brec"]["spec"]["protocol"] == "custom"
    assert report["profiles"]["brec"]["spec"]["requested_batch_size"] == 3
    assert report["profiles"]["brec"]["spec"]["batch_size_graphs"] == 2
    assert set(report["profiles"]["brec"]["variant_losses"]) == {
        "no_pe",
        "raw",
        "set",
        "projector",
    }
    assert set(report["profiles"]["brec"]["variant_hotelling_t2"]) == {
        "no_pe",
        "raw",
        "set",
        "projector",
    }
    assert report["profiles"]["brec"]["spec"]["num_relabel_pairs_for_t2"] == 32


def test_profile_selection_and_shape_validation_are_fail_closed() -> None:
    assert _normalize_profiles(("all",)) == PROFILE_NAMES
    assert _normalize_profiles(("conductance", "conductance")) == ("conductance",)
    with pytest.raises(PreflightError, match="cannot be combined"):
        _normalize_profiles(("all", "conductance"))
    with pytest.raises(PreflightError, match="cycle-rank"):
        ProfileConfig(edges_per_graph=8, cycle_rank=9).validate()
    with pytest.raises(PreflightError, match="requires --brec-batch-size 16"):
        ProfileConfig(brec_batch_size=15).validate()
    with pytest.raises(PreflightError, match="requires --no-brec-amp"):
        ProfileConfig(brec_amp=True).validate()
    ProfileConfig(
        brec_batch_size=15,
        brec_protocol="custom",
        brec_amp=True,
    ).validate()


def test_gpu_preflight_refuses_cpu_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(
        sys,
        "argv",
        [str(PREFLIGHT), "--device", "cpu", "--min-free-gb", "0"],
    ):
        assert main() == 2
    assert "requires CUDA" in capsys.readouterr().out


def test_paper_dependency_check_detects_import_time_abi_failure() -> None:
    import scripts.gpu_preflight as preflight

    real_import = preflight.importlib.import_module

    def broken_scipy(name: str):
        if name == "scipy":
            raise OSError("undefined symbol from binary extension")
        return real_import(name)

    with (
        patch.dict(preflight.PAPER_IMPORTS, {"scipy": "scipy"}, clear=True),
        patch.object(preflight.importlib.util, "find_spec", return_value=object()),
        patch.object(preflight.importlib, "import_module", side_effect=broken_scipy),
    ):
        errors = _paper_dependency_import_errors()
    assert errors["scipy"].startswith("OSError: undefined symbol")
````

# tests/test_gpu_setup_lock.py

````python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.verify_gpu_lock import (
    REQUIRED_RESEARCH_PACKAGES,
    LockVerificationError,
    assert_same_pins,
    read_exact_pins,
    verify_environment,
    version_matches,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock.txt"
CUDA_TAGS = ("cu126", "cu130", "cu132")


def test_all_cuda_constraints_are_exact_and_match_portable_lock() -> None:
    expected = read_exact_pins(LOCK)
    assert REQUIRED_RESEARCH_PACKAGES <= expected.keys()
    for tag in CUDA_TAGS:
        path = ROOT / f"constraints-{tag}.txt"
        assert path.read_text(encoding="utf-8").splitlines()[0] == f"# CUDA_WHEEL_TAG={tag}"
        assert assert_same_pins(LOCK, path) == expected


def test_lock_contains_python_311_compatible_numeric_stack() -> None:
    pins = read_exact_pins(LOCK)
    assert pins["numpy"] == "2.4.6"
    assert pins["scipy"] == "1.17.1"
    assert pins["torch"] == "2.13.0"
    assert pins["torch-geometric"] == "2.8.0.post1"
    assert pins["ogb"] == "1.3.6"
    assert pins["scikit-learn"] == "1.9.0"


def test_exact_pin_parser_rejects_ranges(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("torch>=2.2\n", encoding="utf-8")
    with pytest.raises(LockVerificationError, match="exact name==version"):
        read_exact_pins(invalid)


def test_only_torch_may_have_a_cuda_local_version_suffix() -> None:
    assert version_matches("torch", "2.13.0", "2.13.0+cu126")
    assert not version_matches("torch", "2.13.0", "2.12.1+cu126")
    assert not version_matches("numpy", "2.4.6", "2.4.6+local")


def test_environment_verifier_checks_exact_versions_and_cuda_runtime() -> None:
    pins = read_exact_pins(LOCK)
    installed = {**pins, "torch": f"{pins['torch']}+cu126"}
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    def import_module(name: str) -> object:
        return fake_torch if name == "torch" else object()

    with (
        patch(
            "scripts.verify_gpu_lock.importlib.metadata.version",
            side_effect=installed.__getitem__,
        ),
        patch("scripts.verify_gpu_lock.importlib.import_module", side_effect=import_module),
    ):
        report = verify_environment(
            lock_path=LOCK,
            constraints_path=ROOT / "constraints-cu126.txt",
            cuda_tag="cu126",
        )

    assert report["status"] == "passed"
    assert report["torch_cuda_runtime"] == "12.6"
    assert report["installed_top_level_versions"]["torch"].endswith("+cu126")


def test_gpu_setup_uses_lock_and_has_no_cu118_install_branch() -> None:
    source = (ROOT / "scripts" / "setup_gpu.sh").read_text(encoding="utf-8")
    assert 'constraints_file="${project_root}/constraints-${wheel_tag}.txt"' in source
    assert '--requirement "${lock_file}"' in source
    assert "scripts/verify_gpu_lock.py" in source
    assert 'wheel_tag="cu118"' not in source
    assert "TORCH_SPEC" not in source
    assert "TORCH_INDEX_URL" not in source
    assert "requires a driver supporting CUDA 12.6+" in source
````

# tests/test_research_boundaries.py

````python
"""Prevent the independent research tracks from silently recombining."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(folder: str) -> set[str]:
    imported: set[str] = set()
    for path in (ROOT / "research" / folder).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    return imported


def _assert_no_prefix(imports: set[str], forbidden: tuple[str, ...]) -> None:
    violations = sorted(
        module for module in imports if any(module.startswith(prefix) for prefix in forbidden)
    )
    assert not violations, f"cross-track imports found: {violations}"


def test_conductance_gat_does_not_import_cycle_or_combined_tracks() -> None:
    _assert_no_prefix(
        _imports("conductance_gat"),
        (
            "research.cycle_pe",
            "research.tree_augmentation",
            "research.combined_later",
            "chartgat.completion",
            "chartgat.layers",
        ),
    )


def test_cycle_pe_does_not_import_conductance_or_combined_tracks() -> None:
    _assert_no_prefix(
        _imports("cycle_pe"),
        (
            "research.conductance_gat",
            "research.tree_augmentation",
            "research.combined_later",
            "chartgat.completion",
            "chartgat.layers",
        ),
    )


def test_tree_augmentation_depends_on_neither_conductance_nor_combined_track() -> None:
    _assert_no_prefix(
        _imports("tree_augmentation"),
        (
            "research.conductance_gat",
            "research.combined_later",
            "chartgat.completion",
            "chartgat.layers",
        ),
    )
````

# tests/test_run_all.py

````python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_all.py"


def test_master_runner_dry_run_lists_every_independent_track(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--run-id",
            "portable-dry-run",
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "research.conductance_gat.run" in result.stdout
    assert "research.cycle_pe.run" in result.stdout
    assert "research.tree_augmentation.run" in result.stdout
    assert "combined_later" not in result.stdout
    assert "paper_benchmark" not in result.stdout


def test_master_runner_rejects_unsafe_run_id(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", "--run-id", "../escape"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "run id" in result.stderr.lower()
````

# tests/test_run_paper.py

````python
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.run_paper import (
    CYCLE_BREC_OFFICIAL_SEEDS,
    _run_logged,
    _stop_after_failure,
)
from scripts.run_paper import main as run_paper_main

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_paper.py"
CYCLE_RUNNER = ROOT / "research" / "cycle_pe" / "paper.py"


def _dry_run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> SimpleNamespace:
    with patch.object(sys, "argv", [str(RUNNER), *arguments]):
        return_code = run_paper_main()
    captured = capsys.readouterr()
    return SimpleNamespace(returncode=return_code, stdout=captured.out, stderr=captured.err)


def _literal_assignment(path: Path, name: str) -> object:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"assignment {name!r} was not found in {path}")


def test_shared_preflight_failure_is_always_fatal() -> None:
    assert _stop_after_failure("gpu_preflight", fail_fast=False)
    assert not _stop_after_failure("cycle_pe:seed-0", fail_fast=False)
    assert _stop_after_failure("cycle_pe:seed-0", fail_fast=True)


def test_root_brec_protocol_matches_cycle_runner() -> None:
    assert CYCLE_BREC_OFFICIAL_SEEDS == _literal_assignment(CYCLE_RUNNER, "BREC_OFFICIAL_SEEDS")


def test_logged_child_uses_utf8_for_non_ascii_artifacts(tmp_path: Path) -> None:
    log_path = tmp_path / "child.log"
    return_code = _run_logged(
        [sys.executable, "-c", "print('프로젝트/결과/β')"],
        log_path=log_path,
    )
    assert return_code == 0
    assert log_path.read_text(encoding="utf-8").strip() == "프로젝트/결과/β"


def test_paper_runner_defaults_to_cuda_and_every_independent_track(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--suite",
            "all",
            "--run-id",
            "paper-dry-run",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gpu_preflight.py" in completed.stdout
    assert "--device cuda" in completed.stdout
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    for profile in ("conductance", "cycle-projector", "tree-chart", "brec", "public-pyg"):
        assert f"--profile {profile}" in preflight_line
    assert "--batch-size 32" in preflight_line
    assert "--brec-batch-size 16" in preflight_line
    assert "--cycle-variants no_pe,raw,set,projector" in preflight_line
    assert "--brec-protocol official" in preflight_line
    assert "--no-brec-amp" in preflight_line
    for module in (
        "research.conductance_gat.paper",
        "research.cycle_pe.paper",
        "research.tree_augmentation.paper",
    ):
        assert module in completed.stdout
    assert "combined_later" not in completed.stdout
    assert completed.stdout.count("[cycle_pe:brec:official-10-seed]") == 1
    assert completed.stdout.count("--suite brec") == 1
    brec_line = next(
        line for line in completed.stdout.splitlines() if "[cycle_pe:brec:official-10-seed]" in line
    )
    assert "--batch-size 16" in brec_line
    assert "--workers 0" in brec_line
    assert "--no-amp" in brec_line
    assert "--brec-protocol official" in brec_line
    assert "--batch-size 32" not in brec_line
    assert "--amp" not in brec_line
    assert "--brec-seeds 100,200,300,400,500,600,700,800,900,1000" in completed.stdout


def test_paper_runner_refuses_full_cpu_execution(capsys: pytest.CaptureFixture[str]) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "bad-cpu",
            "--device",
            "cpu",
        ],
        capsys,
    )
    assert completed.returncode == 2
    assert "requires CUDA" in completed.stderr


def test_paper_runner_allows_only_explicit_tiny_cpu_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result_root = tmp_path / "scratch results"
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "tiny-cpu",
            "--device",
            "cpu",
            "--tiny",
            "--allow-cpu",
            "--seeds",
            "7",
            "--results-root",
            str(result_root),
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--tiny" in completed.stdout
    assert "--no-amp" in completed.stdout
    assert "model-seed-7" in completed.stdout
    assert "--model-seed 7" in completed.stdout
    assert "--data-seed 0" in completed.stdout
    assert "--split-seed 0" in completed.stdout
    assert "--chart-seed 0" in completed.stdout
    assert str(result_root.resolve()) in completed.stdout


def test_paper_runner_allows_cpu_data_preparation_without_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "prepare-cpu",
            "--device",
            "cpu",
            "--prepare-only",
            "--suite",
            "all",
            "--allow-download",
            "--seeds",
            "11,12",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--prepare-only" in completed.stdout
    assert "--allow-download" in completed.stdout
    assert completed.stdout.count("--model-seed 11") == 5
    assert "--model-seed 12" not in completed.stdout
    assert "--seed" not in completed.stdout
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--device cpu" in preflight_line
    assert "--profile conductance" in preflight_line
    for expensive_profile in ("cycle-projector", "tree-chart", "brec", "public-pyg"):
        assert f"--profile {expensive_profile}" not in preflight_line


def test_paper_runner_routes_independent_seed_axes(capsys: pytest.CaptureFixture[str]) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "seed-axes",
            "--tracks",
            "tree_augmentation",
            "--model-seeds",
            "5,7",
            "--data-seed",
            "11",
            "--split-seed",
            "13",
            "--chart-seed",
            "17",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("--data-seed 11") == 2
    assert completed.stdout.count("--split-seed 13") == 2
    assert completed.stdout.count("--chart-seed 17") == 2
    assert completed.stdout.count("--model-seed 5") == 1
    assert completed.stdout.count("--model-seed 7") == 1


def test_paper_runner_exposes_cycle_candidate_reduction_without_overriding_official_brec(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "cycle-candidates",
            "--tracks",
            "cycle_pe",
            "--suite",
            "all",
            "--model-seeds",
            "3",
            "--cycle-variants",
            "no_pe,projector",
            "--cycle-core-targets",
            "graph",
            "--cycle-epochs",
            "7",
            "--cycle-learning-rate",
            "0.002",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    cycle_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("[cycle_pe:") and "research.cycle_pe.paper" in line
    ]
    assert len(cycle_lines) == 3
    assert all("--variants no_pe,projector" in line for line in cycle_lines)
    core_line = next(line for line in cycle_lines if "--suite core" in line)
    zinc_line = next(line for line in cycle_lines if "--suite zinc" in line)
    brec_line = next(line for line in cycle_lines if "--suite brec" in line)
    assert "--core-targets graph" in core_line
    assert "--core-targets" not in zinc_line
    assert "--core-targets" not in brec_line
    assert "--epochs 7" in core_line and "--epochs 7" in zinc_line
    assert "--learning-rate 0.002" in core_line
    assert "--learning-rate 0.002" in zinc_line
    assert "--epochs" not in brec_line
    assert "--learning-rate" not in brec_line
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--profile cycle-projector" in preflight_line
    assert "--profile brec" in preflight_line
    assert "--profile public-pyg" not in preflight_line
    assert "--profile conductance" not in preflight_line
    assert "--profile tree-chart" not in preflight_line
    assert "--cycle-variants no_pe,projector" in preflight_line
    assert "--brec-protocol official" in preflight_line
    assert "--brec-batch-size 16" in preflight_line
    assert "--no-brec-amp" in preflight_line


def test_cycle_preflight_runs_selected_non_projector_variants(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "cycle-no-projector",
            "--tracks",
            "cycle_pe",
            "--suite",
            "core",
            "--model-seeds",
            "3",
            "--cycle-variants",
            "no_pe,raw,set",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--profile cycle-projector" in preflight_line
    assert "--profile conductance" not in preflight_line
    assert "--cycle-variants no_pe,raw,set" in preflight_line


def test_tiny_custom_brec_preflight_matches_batch_amp_and_variants(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = _dry_run(
        [
            "--dry-run",
            "--run-id",
            "cycle-tiny-custom",
            "--tracks",
            "cycle_pe",
            "--suite",
            "all",
            "--tiny",
            "--batch-size",
            "7",
            "--model-seeds",
            "3",
            "--cycle-variants",
            "no_pe,raw",
        ],
        capsys,
    )
    assert completed.returncode == 0, completed.stderr
    preflight_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("[gpu_preflight]")
    )
    assert "--profile cycle-projector" in preflight_line
    assert "--profile brec" in preflight_line
    assert "--profile public-pyg" not in preflight_line
    assert "--cycle-variants no_pe,raw" in preflight_line
    assert "--brec-protocol custom" in preflight_line
    assert "--brec-batch-size 7" in preflight_line
    assert "--brec-amp" in preflight_line
    brec_line = next(
        line for line in completed.stdout.splitlines() if "[cycle_pe:brec:custom-tiny]" in line
    )
    assert "--batch-size 7" in brec_line
    assert "--amp" in brec_line
    assert "--brec-protocol custom" in brec_line
    assert "--variants no_pe,raw" in brec_line
````

# tests/test_seed_protocol.py

````python
from __future__ import annotations

import pytest

from chartgat.seeds import SeedAxes, resolve_seed_axes


def test_legacy_seed_fallback_is_explicit() -> None:
    assert resolve_seed_axes(7) == SeedAxes(data=7, split=7, chart=7, model=7)


def test_seed_axes_can_be_varied_independently() -> None:
    axes = resolve_seed_axes(
        99,
        data_seed=1,
        split_seed=2,
        chart_seed=3,
        model_seed=4,
    )
    assert axes.to_manifest() == {"data": 1, "split": 2, "chart": 3, "model": 4}


@pytest.mark.parametrize("field", ("data", "split", "chart", "model"))
def test_negative_seed_axis_is_rejected(field: str) -> None:
    values = {"data": 1, "split": 2, "chart": 3, "model": 4}
    values[field] = -1
    with pytest.raises(ValueError, match=f"{field} seed"):
        SeedAxes(**values)
````
