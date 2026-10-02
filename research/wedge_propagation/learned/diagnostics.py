"""Dense teacher-operator diagnostics, separate from trained model performance.

The span test compares only ``L``, ``L @ L`` and the fixed wedge operator ``Q``.
A positive residual rules out this three-matrix span, not arbitrary spectral
functions. Each scalar input realization has its own frozen teacher matrix.
All cases are retained and equal-size cases share batched float64 kernels.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import isfinite
from numbers import Integral
from typing import Any

import torch

from .data import RuleCase, pack_cases


def _batch_matrices(batch: Any, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build L, Q and T_r by four/nine-entry scatter, without a dense A."""

    graph_count, realizations = batch.num_graphs, batch.x.shape[1]
    edge_graph = batch.node_graph[batch.edges[0]]
    u, v = batch.edges - edge_graph[None] * n
    rows = torch.stack((u, v, u, v))
    cols = torch.stack((u, v, v, u))
    edge_index = (edge_graph[None] * n * n + rows * n + cols).reshape(-1)
    edge_values = batch.x.new_tensor((1.0, 1.0, -1.0, -1.0))[:, None]
    laplacian = batch.x.new_zeros(graph_count * n * n).index_add(
        0, edge_index, edge_values.expand(-1, batch.edges.shape[1]).reshape(-1)
    ).reshape(graph_count, n, n)

    local_nodes = batch.wedges - batch.path_graph[None] * n
    path_index = (batch.path_graph[None, None] * n * n
                  + local_nodes[:, None] * n + local_nodes[None, :]).reshape(-1)
    a = batch.x.new_tensor((1.0, -2.0, 1.0))
    coefficients = (a[:, None] * a[None, :]).reshape(9)
    path_count = batch.wedges.shape[1]
    fixed = batch.x.new_zeros(graph_count * n * n).index_add(
        0, path_index, coefficients[:, None].expand(-1, path_count).reshape(-1)
    ).reshape(graph_count, n, n)
    teacher_values = coefficients[:, None, None] * batch.teacher_c[None]
    teacher = batch.x.new_zeros(graph_count * n * n, realizations).index_add(
        0, path_index, teacher_values.reshape(9 * path_count, realizations)
    ).reshape(graph_count, n * n, realizations)
    return laplacian, fixed, teacher


def _batch_metrics(batch: Any, n: int, path_counts: torch.Tensor) -> torch.Tensor:
    laplacian, fixed, teacher = _batch_matrices(batch, n)
    graph_count, _, realizations = teacher.shape
    basis = torch.stack((laplacian, laplacian @ laplacian, fixed), dim=-1)
    basis = basis.reshape(graph_count, n * n, 3)
    column_norms = torch.linalg.vector_norm(basis, dim=1, keepdim=True)
    basis = basis / torch.where(column_norms > 0, column_norms, 1.0)
    gram = basis.transpose(1, 2) @ basis
    rhs = basis.transpose(1, 2) @ teacher
    fitted = basis @ (torch.linalg.pinv(gram, rtol=1e-12, hermitian=True) @ rhs)
    residual_norms = torch.linalg.vector_norm(teacher - fitted, dim=1)
    target_norms = torch.linalg.vector_norm(teacher, dim=1)
    defined = target_norms > 0
    defined_count = defined.sum(dim=1)
    relative = residual_norms / torch.where(defined, target_norms, 1.0)
    residual_mean = (relative * defined).sum(dim=1) / defined_count.clamp_min(1)
    nan = teacher.new_full((graph_count,), float("nan"))
    residual_mean = torch.where(defined_count > 0, residual_mean, nan)

    reference_norm = target_norms[:, 0]
    if realizations > 1:
        changes = torch.linalg.vector_norm(teacher[:, :, 1:] - teacher[:, :, :1], dim=1)
        variation = (changes / (reference_norm[:, None] + 1e-8)).mean(dim=1)
        variation = torch.where(reference_norm > 0, variation, nan)
        variation_count = (reference_norm > 0).long() * (realizations - 1)
    else:
        variation, variation_count = nan, torch.zeros_like(defined_count)

    # Population std across paths, then mean across scalar input realizations.
    c_sum = teacher.new_zeros(graph_count, realizations).index_add(
        0, batch.path_graph, batch.teacher_c
    )
    counts = path_counts[:, None].clamp_min(1)
    centered_c = batch.teacher_c - (c_sum / counts).index_select(0, batch.path_graph)
    c_var = teacher.new_zeros(graph_count, realizations).index_add(
        0, batch.path_graph, centered_c.square()
    ) / counts
    c_std = torch.where(path_counts > 0, c_var.sqrt().mean(dim=1), nan)
    return torch.stack((residual_mean, residual_norms.mean(dim=1), defined_count,
                        realizations - defined_count, variation, reference_norm,
                        variation_count, c_std), dim=1)


@torch.no_grad()
def operator_audit(
    cases: Sequence[RuleCase], device: torch.device | str, batch_size: int
) -> list[dict[str, Any]]:
    """Audit every teacher matrix and return one row per graph in input order.

    ``teacher_span_L_L2_Q_residual`` is the mean Frobenius residual ratio across
    nonzero feature-realization matrices. ``teacher_operator_variation`` compares
    realizations 1..R-1 with realization 0 using denominator ``||T_0||F + 1e-8``.
    Undefined ratios (zero teacher matrix/reference or R=1) are returned as None,
    with counts included in each row. No teacher labels enter a trained model.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    cases = list(cases)
    if not cases:
        raise ValueError("operator audit requires at least one case")
    if len({case.graph_id for case in cases}) != len(cases):
        raise ValueError("duplicate graph ID in operator audit")
    buckets: dict[tuple[int, int], list[RuleCase]] = defaultdict(list)
    for case in cases:
        if (case.num_nodes <= 0 or case.features.ndim != 2
                or case.features.shape[0] != case.num_nodes):
            raise ValueError("case features must be [num_nodes, realizations] with nodes > 0")
        realizations = case.features.shape[1]
        if realizations < 1:
            raise ValueError("operator audit requires at least one scalar realization")
        if case.teacher_c.shape != (case.wedges.shape[1], realizations):
            raise ValueError("teacher_c must have shape [num_paths, realizations]")
        if not torch.isfinite(case.teacher_c).all():
            raise ValueError("teacher_c must contain finite weights")
        buckets[(case.num_nodes, realizations)].append(case)

    rows_by_id: dict[str, dict[str, Any]] = {}
    for (n, realizations), bucket in buckets.items():
        for offset in range(0, len(bucket), batch_size):
            group = bucket[offset:offset + batch_size]
            batch = pack_cases(group, device=device, dtype=torch.float64)
            path_counts = torch.tensor([case.wedges.shape[1] for case in group],
                                       dtype=torch.long, device=batch.x.device)
            # One result transfer per graph batch; the matrix kernels stay batched.
            metrics = _batch_metrics(batch, n, path_counts).cpu()
            if torch.isinf(metrics).any():
                raise RuntimeError("nonfinite teacher-operator diagnostic")
            for case, values in zip(group, metrics.tolist(), strict=True):
                residual, absolute, defined, undefined, variation, norm, pairs, c_std = values
                required = [absolute, defined, undefined, norm, pairs]
                if defined > 0:
                    required.append(residual)
                if pairs > 0:
                    required.append(variation)
                if case.wedges.shape[1] > 0:
                    required.append(c_std)
                if not all(isfinite(value) for value in required):
                    raise RuntimeError("nonfinite teacher-operator diagnostic")
                rows_by_id[case.graph_id] = {
                    "graph_id": case.graph_id, "family": case.family, "split": case.split,
                    "num_nodes": n, "num_edges": case.edges.shape[1],
                    "num_paths": case.wedges.shape[1], "num_features": realizations,
                    "teacher_span_L_L2_Q_residual": None if residual != residual else residual,
                    "teacher_span_absolute_residual_mean": absolute,
                    "teacher_span_defined_features": int(defined),
                    "teacher_span_undefined_features": int(undefined),
                    "teacher_operator_variation": None if variation != variation else variation,
                    "teacher_operator_reference_norm": norm,
                    "teacher_operator_variation_defined_pairs": int(pairs),
                    "teacher_c_std": None if c_std != c_std else c_std,
                }
    return [rows_by_id[case.graph_id] for case in cases]
