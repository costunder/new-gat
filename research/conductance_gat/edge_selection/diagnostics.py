"""Zero-safe graph/weight diagnostics; CPU work belongs to post-training audit only."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def distribution(values):
    values = torch.as_tensor(values).detach().float().cpu().flatten()
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite values in edge-selection audit")
    if values.numel() == 0:
        return {"count": 0, "mean": None, "quantiles": None, "zero_fraction": None}
    quantiles = torch.quantile(
        values, torch.tensor([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    )
    boundaries = torch.linspace(0, 1, 11)
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        "std_population": float(values.std(unbiased=False)),
        "quantile_probabilities": [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
        "quantiles": quantiles.tolist(),
        "zero_fraction": float((values == 0).float().mean()),
        "below_zero_count": int((values < 0).sum()),
        "above_one_count": int((values > 1).sum()),
        "unit_interval_histogram_boundaries": boundaries.tolist(),
        "unit_interval_histogram_counts": torch.histogram(values, bins=boundaries)
        .hist.long()
        .tolist(),
    }


def adjacency(num_nodes, edges, active=None):
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape[0] != 2:
        raise ValueError("incidence endpoint array must be 2 x E")
    if active is not None:
        edges = edges[:, np.asarray(active, dtype=bool)]
    source, target = edges
    return coo_matrix(
        (np.ones(2 * source.size), (np.r_[source, target], np.r_[target, source])),
        shape=(num_nodes, num_nodes),
    ).tocsr()


def topology_statistics(matrix):
    count = matrix.shape[0]
    components = (
        int(connected_components(matrix, directed=False, return_labels=False)) if count else 0
    )
    edges = matrix.nnz // 2
    degree = np.diff(matrix.indptr)
    return {
        "nodes": count,
        "edges": edges,
        "components": components,
        "isolated_nodes": int((degree == 0).sum()),
        "cycle_rank": edges - count + components,
        "degree": distribution(degree),
    }


def path_reference(matrix, *, sources=32, seed=0):
    if sources < 1:
        raise ValueError("path diagnostic must request a positive number of landmark sources")
    count = matrix.shape[0]
    chosen = np.sort(np.random.default_rng(seed).choice(count, min(sources, count), replace=False))
    distances = shortest_path(matrix, directed=False, unweighted=True, indices=chosen)
    return chosen, distances


def path_change(matrix, reference):
    chosen, before = reference
    after = shortest_path(matrix, directed=False, unweighted=True, indices=chosen)
    positive = np.isfinite(before) & (before > 0)
    retained = positive & np.isfinite(after)
    lost = positive & ~np.isfinite(after)
    return {
        "scope": "fixed seeded landmark sources to every node; not all-pairs distances",
        "source_nodes": chosen.tolist(),
        "source_count": chosen.size,
        "source_coverage": chosen.size / matrix.shape[0] if matrix.shape[0] else None,
        "originally_reachable_ordered_pairs": int(positive.sum()),
        "lost_reachable_ordered_pairs": int(lost.sum()),
        "distance_increase_on_retained_pairs": distribution(after[retained] - before[retained]),
        "stretch_on_retained_pairs": distribution(after[retained] / before[retained]),
    }


def gate_origins(gate, targets):
    gate, targets = torch.as_tensor(gate), torch.as_tensor(targets)
    if gate.shape != targets.shape or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("origin diagnostic target alignment is invalid")
    result = {}
    for name, mask in (("original", targets == 1), ("added", targets == 0)):
        result[name] = {
            "candidate_count": int(mask.sum()),
            "active_count": int((gate[mask] > 0).sum()),
            "active_rate": float((gate[mask] > 0).float().mean()) if mask.any() else None,
            "gate": distribution(gate[mask]),
        }
    return result


def coefficient_statistics(tail_coeff, head_coeff, tail, head, num_nodes):
    # Directed coefficients are normalized across incoming neighbors, not across heads.
    coefficients = torch.cat((tail_coeff, head_coeff)).detach().float()
    receivers = torch.cat((tail, head))
    degree = torch.bincount(receivers[coefficients.max(dim=1).values > 0], minlength=num_nodes)
    mass = coefficients.new_zeros((num_nodes, coefficients.shape[1])).index_add_(
        0, receivers, coefficients
    )
    entropy = mass.new_zeros(mass.shape).index_add_(
        0, receivers, -torch.special.xlogy(coefficients, coefficients)
    )
    squares = mass.new_zeros(mass.shape).index_add_(0, receivers, coefficients.square())
    active = squares > 0
    effective = torch.where(active, squares.reciprocal(), 0.0)
    maximum = mass.new_zeros(mass.shape).scatter_reduce_(
        0,
        receivers[:, None].expand_as(coefficients),
        coefficients,
        reduce="amax",
        include_self=True,
    )
    rows = {}
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 11), (11, 33), (33, None)):
        selected = degree >= lo
        if hi is not None:
            selected &= degree < hi
        rows[f"{lo}:{hi}"] = {
            "nodes": int(selected.sum()),
            "mass": distribution(mass[selected]),
            "entropy": distribution(entropy[selected]),
            "effective_neighbors": distribution(effective[selected]),
            "max_alpha": distribution(maximum[selected]),
        }
    return {
        "alpha": distribution(coefficients),
        "incoming_mass": distribution(mass),
        "active_degree_bins": rows,
        "isolates_have_zero_neighbor_mass": True,
    }
