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
