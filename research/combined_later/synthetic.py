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
