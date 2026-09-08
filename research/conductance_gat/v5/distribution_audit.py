"""Exact selected-checkpoint C/propagation distributions; no training hot-path work.

All edges, nodes and heads participate. Graph loops only format disjoint scopes;
edge/head arithmetic, degree aggregation and pair comparisons stay vectorized.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import conductance_propagation_coefficients

QUANTILES = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
HISTOGRAM_BOUNDARIES = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.0)


def _summary(values: Tensor) -> dict:
    """Columns are heads (or head pairs), never pooled together."""
    count, _ = values.shape
    if not count:
        return {
            "count_per_column": 0,
            "mean": None,
            "std": None,
            "cv": None,
            "quantiles": None,
            "reason": "no observations in this scope",
        }
    value = values.detach().float()
    mean, std = value.mean(0), value.std(0, correction=0)
    # Sorting has no torch.quantile total-element limit and samples no subset.
    ordered = value.sort(dim=0).values
    positions = value.new_tensor(QUANTILES) * (count - 1)
    lower, upper = positions.floor().long(), positions.ceil().long()
    fraction = (positions - lower).unsqueeze(1)
    quantiles = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
    return {
        "count_per_column": count,
        "mean": mean,
        "std": std,
        "cv": std / mean.abs().clamp_min(torch.finfo(value.dtype).tiny),
        "cv_undefined_zero_mean": mean == 0,
        "quantiles": {str(q): quantiles[index] for index, q in enumerate(QUANTILES)},
    }


def _raw_summary(value: Tensor) -> dict:
    result = _summary(value)
    if not value.shape[0]:
        return {**result, "fractions": None, "histogram": None}
    boundaries = value.new_tensor(HISTOGRAM_BOUNDARIES)
    bins = torch.bucketize(value.contiguous(), boundaries, right=True)
    offsets = torch.arange(value.shape[1], device=value.device) * (len(boundaries) + 1)
    counts = torch.bincount(
        (bins + offsets).flatten(), minlength=value.shape[1] * (len(boundaries) + 1)
    )
    return {
        **result,
        "fractions": {
            "c_ge_0_7": (value >= 0.7).float().mean(0),
            "abs_c_minus_1_le_0_1": ((value >= 0.9) & (value <= 1.1)).float().mean(0),
            "c_gt_1": (value > 1).float().mean(0),
        },
        "histogram": {
            "finite_boundaries": list(HISTOGRAM_BOUNDARIES),
            "interval_rule": "[-inf,b0), [b0,b1), ..., [blast,+inf)",
            "counts_by_head": counts.reshape(value.shape[1], -1),
        },
    }


def _sum_nodes(value: Tensor, destination: Tensor, nodes: int) -> Tensor:
    return value.new_zeros((nodes, value.shape[1])).index_add_(0, destination, value)


def _node_statistics(alpha: Tensor, destination: Tensor, nodes: int, degree: Tensor) -> dict:
    top1 = alpha.new_zeros((nodes, alpha.shape[1]))
    top1.scatter_reduce_(
        0, destination[:, None].expand_as(alpha), alpha, reduce="amax", include_self=True
    )
    entropy = _sum_nodes(-torch.special.xlogy(alpha, alpha), destination, nodes)
    squares = _sum_nodes(alpha.square(), destination, nodes)
    active = degree[:, None] > 0
    return {
        "top1": top1,
        "entropy_nats": entropy,
        "normalized_entropy": torch.where(
            degree[:, None] > 1,
            entropy / degree[:, None].clamp_min(2).float().log(),
            torch.zeros_like(entropy),
        ),
        "effective_neighbors": torch.where(
            active, squares.clamp_min(1e-30).reciprocal(), torch.zeros_like(squares)
        ),
    }


def _head_pairs(alpha: Tensor, destination: Tensor, nodes: int) -> tuple[Tensor, dict]:
    pair = torch.triu_indices(alpha.shape[1], alpha.shape[1], offset=1, device=alpha.device)
    if not pair.shape[1]:
        return pair, {}
    left, right = alpha[:, pair[0]], alpha[:, pair[1]]
    midpoint = (left + right) * 0.5
    tv = _sum_nodes((left - right).abs() * 0.5, destination, nodes)
    # Positive conductances imply positive probabilities on every retained edge.
    js = _sum_nodes(
        0.5
        * (
            torch.special.xlogy(left, left / midpoint)
            + torch.special.xlogy(right, right / midpoint)
        ),
        destination,
        nodes,
    )
    return pair, {"total_variation": tv, "jensen_shannon_nats": js}


@torch.no_grad()
def audit_conductance_distribution(
    conductance: Tensor,
    incidence: Tensor,
    node_graph: Tensor,
    num_graphs: int,
    *,
    heads: int,
    beta: Tensor,
    sampling_correction: Tensor | None = None,
    normalization: str = "symmetric",
    propagation_filter: str = "linear",
) -> dict:
    """Summarize this exact forward, not historical epochs or a whole dataset.

    ``weighted_c_row_alpha`` is a/d_destination. ``actual_kernel_row_relative``
    is P/sum(P) and is explicitly only a diagnostic for symmetric propagation.
    P excludes beta and polynomial higher-order terms, reported separately.
    """
    if normalization not in {"symmetric", "row"}:
        raise ValueError("unknown propagation normalization")
    nodes, edges = node_graph.numel(), incidence.shape[1]
    if conductance.shape not in {(edges,), (edges, heads)}:
        raise ValueError("C must be physical-edge shared E or per-head E x H")
    c = conductance.detach().float()
    c = c[:, None].expand(-1, heads) if c.ndim == 1 else c
    if sampling_correction is None:
        correction = c.new_ones(edges)
    else:
        correction = sampling_correction.detach().float()
    if correction.shape != (edges,):
        raise ValueError("sampling correction must be one value per physical edge")
    if beta.shape != (num_graphs, heads):
        raise ValueError("beta must be graph x head")
    if not bool(
        torch.isfinite(c).all()
        & (c > 0).all()
        & torch.isfinite(correction).all()
        & (correction > 0).all()
    ):
        raise ValueError("C and correction must be finite and positive")
    tail, head = incidence
    destination = torch.cat((head, tail))
    degree = torch.bincount(destination, minlength=nodes)
    edge_graph = node_graph[tail]
    effective = c * correction[:, None]

    def coefficients(edge_weight):
        directed = torch.cat((edge_weight, edge_weight), dim=0)
        weighted_degree = _sum_nodes(directed, destination, nodes)
        row_alpha = directed / weighted_degree[destination]
        to_tail, to_head, _ = conductance_propagation_coefficients(
            edge_weight, incidence, nodes, normalization=normalization
        )
        coefficient = torch.cat((to_head, to_tail), dim=0)
        rowsum = _sum_nodes(coefficient, destination, nodes)
        relative = coefficient / rowsum[destination].clamp_min(1e-30)
        return coefficient, row_alpha, relative, rowsum

    coefficient, alpha, relative, rowsum = coefficients(effective)
    c1_coefficient, c1_alpha, c1_relative, c1_rowsum = coefficients(
        correction[:, None].expand_as(c)
    )
    families = {}
    for name, probability, reference in (
        ("weighted_c_row_alpha", alpha, c1_alpha),
        ("actual_kernel_row_relative", relative, c1_relative),
    ):
        pair, diversity = _head_pairs(probability, destination, nodes)
        families[name] = {
            "statistics": _node_statistics(probability, destination, nodes, degree),
            "c_one_statistics": _node_statistics(reference, destination, nodes, degree),
            "head_pairs": pair.T,
            "head_diversity": diversity,
            "c_one_total_variation": _sum_nodes(
                (probability - reference).abs() * 0.5, destination, nodes
            ),
        }
    graphs = []
    for graph_id in range(num_graphs):
        node_mask = node_graph == graph_id
        edge_mask = edge_graph == graph_id
        directed_mask = node_graph[destination] == graph_id
        active = node_mask & (degree > 0)
        buckets = (
            ("degree_1", 1, 1),
            ("degree_2_4", 2, 4),
            ("degree_5_9", 5, 9),
            ("degree_10_19", 10, 19),
            ("degree_ge_20", 20, None),
        )
        graph = {
            "graph_in_batch": graph_id,
            "heads": list(range(heads)),
            "nodes": node_mask.sum(),
            "physical_edges": edge_mask.sum(),
            "isolates": (node_mask & (degree == 0)).sum(),
            "degree_one_nodes": (node_mask & (degree == 1)).sum(),
            "raw_c": _raw_summary(c[edge_mask]),
            "beta_by_head": beta[graph_id],
            "actual_one_hop_coefficient": _summary(coefficient[directed_mask]),
            "actual_one_hop_row_sum": _summary(rowsum[node_mask]),
            "c_one_same_correction_coefficient": _summary(c1_coefficient[directed_mask]),
            "c_one_same_correction_row_sum": _summary(c1_rowsum[node_mask]),
            "probabilities": {},
        }
        for name, family in families.items():
            masks = [("all_nonisolates", active)] + [
                (
                    label,
                    node_mask
                    & (degree >= lower)
                    & ((degree <= upper) if upper is not None else True),
                )
                for label, lower, upper in buckets
            ]
            graph["probabilities"][name] = {
                "node_scopes": {
                    label: {
                        "nodes": mask.sum(),
                        **{
                            key: _summary(value[mask])
                            for key, value in family["statistics"].items()
                        },
                        "c_one_same_correction": {
                            key: _summary(value[mask])
                            for key, value in family["c_one_statistics"].items()
                        },
                        "c_one_total_variation": _summary(family["c_one_total_variation"][mask]),
                    }
                    for label, mask in masks
                },
                "head_pairs": family["head_pairs"],
                "head_diversity_nonisolates": {
                    key: _summary(value[active]) for key, value in family["head_diversity"].items()
                },
            }
        graphs.append(graph)
    return {
        "scope": "this layer forward and physical validation batch; graphs kept separate",
        "raw_c_units": "positive conductance, not an attention probability; may exceed 1",
        "conductance_layout": "shared_E" if conductance.ndim == 1 else "per_head_E_H",
        "propagation_normalization": normalization,
        "propagation_filter": propagation_filter,
        "coefficient_scope": "one-hop P before beta; excludes polynomial P^2/P^3 terms",
        "weighted_c_row_alpha_definition": "omega*c / incident sum(omega*c)",
        "actual_kernel_row_relative_definition": "P / row_sum(P); diagnostic only",
        "c_one_reference": "same topology, sampling correction, normalization and heads; C=1",
        "isolates_policy": "counted; excluded from undefined neighbor probabilities",
        "degree_one_policy": "top1=1, effective_neighbors=1, normalized_entropy=0",
        "head_diversity_scope": "normalized neighbor distributions; scale-only C differs by 0",
        "graphs": graphs,
    }
