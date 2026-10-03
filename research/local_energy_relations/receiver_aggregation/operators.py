"""Complete receiver receipt operators and restricted physical reconstruction.

The sum map is lossy on arbitrary labelled receipt arrays. Its restriction to
positive local Laplacians of one shared physical field determines that field up
to physical-component constants. These are distinct admissible domains.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

from ..operators import incidence, incidence_transpose, local_weight
from ..topology import CORRESPONDENCE_KINDS, LocalTopology, batch_topologies


@dataclass(frozen=True)
class ReceiverOperator:
    topology: LocalTopology
    weight_mode: str
    weights: Tensor
    component_index: Tensor
    component_sizes: Tensor
    node_graph: Tensor
    receipt_counts: Tensor
    receiver_counts: Tensor
    matrix: Tensor
    matrix_transpose: Tensor
    normal_diagonal: Tensor
    maximum_graph_nodes: int
    metadata: dict

    @property
    def num_graphs(self):
        return self.topology.num_graphs

    @property
    def num_components(self):
        return self.component_sizes.numel()

    def to(self, device, dtype=torch.float64):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("receiver operator requires float32 or float64")
        return ReceiverOperator(
            self.topology.to(device),
            self.weight_mode,
            self.weights.to(device=device, dtype=dtype),
            self.component_index.to(device),
            self.component_sizes.to(device),
            self.node_graph.to(device),
            self.receipt_counts.to(device),
            self.receiver_counts.to(device),
            self.matrix.to(device=device, dtype=dtype),
            self.matrix_transpose.to(device=device, dtype=dtype),
            self.normal_diagonal.to(device=device, dtype=dtype),
            self.maximum_graph_nodes,
            self.metadata,
        )


def _tensor_csr(matrix):
    matrix = matrix.tocsr()
    matrix.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)),
        torch.from_numpy(matrix.indices.astype(np.int64)),
        torch.from_numpy(matrix.data.astype(np.float64)),
        size=matrix.shape,
        dtype=torch.float64,
        check_invariants=True,
    )


def _scipy_csr(matrix: Tensor):
    if matrix.device.type != "cpu" or matrix.layout != torch.sparse_csr:
        raise ValueError("sparse graph batching requires CPU CSR operators")
    return sp.csr_matrix(
        (matrix.values().numpy(), matrix.col_indices().numpy(), matrix.crow_indices().numpy()),
        shape=matrix.shape,
    )


def batch_receiver_operators(operators: list[ReceiverOperator]) -> ReceiverOperator:
    """Block-diagonal batch of cached complete graphs, with no local CSR rebuild."""
    if not operators:
        raise ValueError("at least one complete receiver operator is required")
    if len({op.weight_mode for op in operators}) != 1:
        raise ValueError("one receiver batch requires a common weight mode")
    if any(op.weights.device.type != "cpu" for op in operators):
        raise ValueError("receiver batching requires CPU operators")
    started = time.monotonic()
    top = batch_topologies([op.topology for op in operators]).to("cpu")
    matrix = sp.block_diag([_scipy_csr(op.matrix) for op in operators], format="csr")
    transpose = matrix.T.tocsr()
    component_offset = graph_offset = 0
    components, graph_ids = [], []
    for op in operators:
        components.append(op.component_index + component_offset)
        graph_ids.append(op.node_graph + graph_offset)
        component_offset += op.num_components
        graph_offset += op.num_graphs
    metadata = {
        "sparse_shape": list(matrix.shape),
        "sparse_nnz": int(matrix.nnz),
        "cpu_csr_bytes": int(
            8 * (matrix.nnz * 2 + matrix.shape[0] + 1)
            + 8 * (transpose.nnz * 2 + transpose.shape[0] + 1)
        ),
        "prepare_seconds": time.monotonic() - started,
        "tagged_coordinates": int(sum(op.metadata["tagged_coordinates"] for op in operators)),
        "fused_active_coordinates": int(
            sum(op.metadata["fused_active_coordinates"] for op in operators)
        ),
        "ambient_receipt_kernel_dimension": int(
            sum(op.metadata["ambient_receipt_kernel_dimension"] for op in operators)
        ),
        "restricted_field_rank": int(sum(op.metadata["restricted_field_rank"] for op in operators)),
        "restricted_q_kernel_dimension": 0,
        "normal_preconditioner": "exact_diagonal_of_A_transpose_A",
        "complete_graphs_batched": graph_offset,
        "cached_input_operators": len(operators),
    }
    return ReceiverOperator(
        top,
        operators[0].weight_mode,
        torch.cat([op.weights for op in operators]),
        torch.cat(components),
        torch.cat([op.component_sizes for op in operators]),
        torch.cat(graph_ids),
        torch.cat([op.receipt_counts for op in operators]),
        torch.cat([op.receiver_counts for op in operators]),
        _tensor_csr(matrix),
        _tensor_csr(transpose),
        torch.cat([op.normal_diagonal for op in operators]),
        max(op.maximum_graph_nodes for op in operators),
        metadata,
    )


def prepare_receiver_operator(topology: LocalTopology, weight_mode: str) -> ReceiverOperator:
    """Prepare all components and exact sparse A=T blockdiag(L_C) M once."""
    started = time.monotonic()
    if isinstance(topology.edges, Tensor) and topology.edges.device.type != "cpu":
        raise ValueError("receiver preparation needs a CPU topology")
    top = topology.to("cpu")
    parent = np.arange(top.n)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in top.edges.numpy().T:
        left, right = find(int(a)), find(int(b))
        if left != right:
            parent[right] = left
    _, component = np.unique([find(i) for i in range(top.n)], return_inverse=True)
    component_index = torch.from_numpy(component)
    sizes = torch.bincount(component_index)
    graph_sizes = top.graph_node_offsets[1:] - top.graph_node_offsets[:-1]
    node_graph = torch.repeat_interleave(torch.arange(top.num_graphs), graph_sizes)
    counts = torch.bincount(top.shared_node_left, minlength=top.num_local_nodes)
    receiver_counts = torch.bincount(top.shared_node_right, minlength=top.num_local_nodes)
    if not bool(((counts > 0) | (top.local_degree == 0)).all()):
        raise ValueError("complete adjacent receivers fail to cover a nonisolated local occurrence")
    weights = local_weight(top, weight_mode)
    ml, nl = top.num_local_edges, top.num_local_nodes
    incidence_rows = np.tile(np.arange(ml), 2)
    b = sp.csr_matrix(
        (
            np.r_[-np.ones(ml), np.ones(ml)],
            (incidence_rows, top.local_edge_nodes.numpy().reshape(-1)),
        ),
        shape=(ml, nl),
    )
    local_l = b.T @ sp.diags(weights.numpy()) @ b
    local_copy = sp.csr_matrix(
        (np.ones(nl), (np.arange(nl), top.local_node_global.numpy())), shape=(nl, top.n)
    )
    t = sp.csr_matrix(
        (
            np.ones(top.shared_node_left.numel()),
            (top.shared_node_right.numpy(), top.shared_node_left.numpy()),
        ),
        shape=(nl, nl),
    )
    matrix = (t @ local_l @ local_copy).tocsr()
    matrix.eliminate_zeros()
    transpose = matrix.T.tocsr()
    diagonal = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
    scipy_bytes = sum(
        a.nbytes
        for a in (
            matrix.data,
            matrix.indices,
            matrix.indptr,
            transpose.data,
            transpose.indices,
            transpose.indptr,
        )
    )
    metadata = {
        "sparse_shape": list(matrix.shape),
        "sparse_nnz": int(matrix.nnz),
        "cpu_csr_bytes": int(
            8 * (matrix.nnz * 2 + matrix.shape[0] + 1)
            + 8 * (transpose.nnz * 2 + transpose.shape[0] + 1)
        ),
        "scipy_csr_bytes": int(scipy_bytes),
        "prepare_seconds": time.monotonic() - started,
        "tagged_coordinates": int(top.shared_node_left.numel()),
        "fused_active_coordinates": int(np.count_nonzero(receiver_counts.numpy())),
        "ambient_receipt_kernel_dimension": int(
            top.shared_node_left.numel() - np.count_nonzero(receiver_counts.numpy())
        ),
        "restricted_field_rank": int(top.n - len(sizes)),
        "restricted_q_kernel_dimension": 0,
        "normal_preconditioner": "exact_diagonal_of_A_transpose_A",
    }
    return ReceiverOperator(
        top,
        weight_mode,
        weights,
        component_index,
        sizes,
        node_graph,
        counts,
        receiver_counts,
        _tensor_csr(matrix),
        _tensor_csr(transpose),
        torch.from_numpy(diagonal),
        int(graph_sizes.max()),
        metadata,
    )


def _validate(op, values, rows, name):
    if (
        not isinstance(values, Tensor)
        or values.ndim != 2
        or values.shape[0] != rows
        or values.shape[1] < 1
        or values.dtype != op.weights.dtype
        or values.device != op.weights.device
    ):
        raise ValueError(f"{name} must be [{rows},F] on the operator device and dtype")


def _chunks(top, relation_batch):
    if relation_batch is not None and (
        isinstance(relation_batch, bool)
        or not isinstance(relation_batch, Integral)
        or relation_batch < 1
    ):
        raise ValueError("relation_batch must be a positive pair count or None")
    batch = max(1, top.num_pairs) if relation_batch is None else int(relation_batch)
    offsets = top.correspondence_offsets[CORRESPONDENCE_KINDS.index("shared_node")]
    for start in range(0, top.num_pairs, batch):
        yield offsets[start], offsets[min(start + batch, top.num_pairs)]


def shared_flow(op, physical):
    _validate(op, physical, op.topology.n, "physical field")
    q = op.weights[:, None] * incidence(op.topology, physical[op.topology.local_node_global])
    return q, incidence_transpose(op.topology, q)


def tagged_receipts(op, divergence):
    _validate(op, divergence, op.topology.num_local_nodes, "local divergence")
    return divergence.index_select(0, op.topology.shared_node_left)


def fuse_receipts(op, tagged):
    _validate(op, tagged, op.topology.shared_node_left.numel(), "tagged receipts")
    return tagged.new_zeros((op.topology.num_local_nodes, tagged.shape[1])).index_add(
        0, op.topology.shared_node_right, tagged
    )


def receipt_reconstruct(op, tagged):
    """Recover consistent sender copies, with structural zero for isolates."""
    _validate(op, tagged, op.topology.shared_node_left.numel(), "tagged receipts")
    recovered = (
        tagged.new_zeros((op.topology.num_local_nodes, tagged.shape[1])).index_add(
            0, op.topology.shared_node_left, tagged
        )
        / op.receipt_counts.clamp_min(1)[:, None]
    )
    if not torch.allclose(recovered[op.topology.shared_node_left], tagged, rtol=1e-12, atol=1e-14):
        raise ValueError("tagged receipts contain inconsistent sender copies")
    return recovered


def ambient_receipt_projection(op, tagged):
    """r_min=S.T(SS.T)^dagger Y and z=r-r_min in the unrestricted receipt domain."""
    y = fuse_receipts(op, tagged)
    mean = y / op.receiver_counts.clamp_min(1)[:, None]
    minimum = mean[op.topology.shared_node_right]
    return {"fused": y, "minimum_norm_receipts": minimum, "contrast": tagged - minimum}


def fuse_divergence(op, divergence, *, relation_batch=None):
    top = op.topology
    _validate(op, divergence, top.num_local_nodes, "local divergence")
    result = divergence.new_zeros((top.num_local_nodes, divergence.shape[1]))
    for begin, end in _chunks(top, relation_batch):
        result.index_add_(
            0, top.shared_node_right[begin:end], divergence[top.shared_node_left[begin:end]]
        )
    return result


def fuse_transpose(op, receiver_values, *, relation_batch=None):
    top = op.topology
    _validate(op, receiver_values, top.num_local_nodes, "receiver field")
    result = receiver_values.new_zeros(receiver_values.shape)
    for begin, end in _chunks(top, relation_batch):
        result.index_add_(
            0, top.shared_node_left[begin:end], receiver_values[top.shared_node_right[begin:end]]
        )
    return result


def fused_apply(op, physical, *, relation_batch=None):
    """Exact cached sparse A H. relation_batch remains accepted for API symmetry."""
    _validate(op, physical, op.topology.n, "physical field")
    return torch.sparse.mm(op.matrix, physical)


def fused_transpose(op, receiver_values, *, relation_batch=None):
    _validate(op, receiver_values, op.topology.num_local_nodes, "receiver field")
    return torch.sparse.mm(op.matrix_transpose, receiver_values)


def project_components(op, physical):
    _validate(op, physical, op.topology.n, "physical field")
    means = (
        physical.new_zeros((op.num_components, physical.shape[1])).index_add(
            0, op.component_index, physical
        )
        / op.component_sizes[:, None]
    )
    return physical - means[op.component_index]


def _graph_sum(op, values):
    return values.new_zeros((op.num_graphs, values.shape[1])).index_add(0, op.node_graph, values)


def _receiver_sum(op, values):
    return values.new_zeros((op.num_graphs, values.shape[1])).index_add(
        0, op.node_graph[op.topology.local_node_center], values
    )


def graph_norm_sq(op: ReceiverOperator, values: Tensor, *, space: str) -> Tensor:
    """Per-complete-graph, per-channel squared norms, preserving original batches."""
    top = op.topology
    domains = {
        "physical": (top.n, op.node_graph),
        "local_node": (top.num_local_nodes, op.node_graph[top.local_node_center]),
        "local_edge": (top.num_local_edges, op.node_graph[top.local_edge_center]),
        "tagged": (
            top.shared_node_left.numel(),
            op.node_graph[top.pair_centers[1, top.shared_node_pair]],
        ),
    }
    if space not in domains:
        raise ValueError("norm space must be physical, local_node, local_edge, or tagged")
    rows, graph = domains[space]
    _validate(op, values, rows, "norm values")
    return values.new_zeros((op.num_graphs, values.shape[1])).index_add(0, graph, values.square())


@torch.no_grad()
def recover_fused(
    op,
    observation,
    *,
    tolerance=1e-8,
    max_iterations_factor=10,
    relation_batch=None,
    progress: Callable[[int, int, float], None] | None = None,
):
    """Projected normal CG, validated using the actual receiver residual.

    A nonconvergent or inconsistent observation raises an explicit error. The
    numerical iteration ceiling never removes graphs, nodes, or features.
    """
    top = op.topology
    _validate(op, observation, top.num_local_nodes, "fused observation")
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be finite and in (0,1)")
    if (
        isinstance(max_iterations_factor, bool)
        or not isinstance(max_iterations_factor, Integral)
        or max_iterations_factor < 1
    ):
        raise ValueError("max_iterations_factor must be a positive integer")
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("observation must be finite")
    rhs = project_components(op, fused_transpose(op, observation))
    rhs_norm = _graph_sum(op, rhs.square()).sqrt()
    y_norm = _receiver_sum(op, observation.square()).sqrt()
    zero = _receiver_sum(op, observation.abs()) == 0
    if not bool(((y_norm > 0) | zero).all()):
        raise FloatingPointError("observation norm underflow")
    safe_y = torch.where(y_norm > 0, y_norm, torch.ones_like(y_norm))
    safe_rhs = torch.where(rhs_norm > 0, rhs_norm, torch.ones_like(rhs_norm))
    inverse = torch.where(
        op.normal_diagonal > 0, op.normal_diagonal, torch.ones_like(op.normal_diagonal)
    ).reciprocal()
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    z = project_components(op, r * inverse[:, None])
    p = z.clone()
    rz = _graph_sum(op, r * z)
    active = ~zero
    iterations = torch.zeros_like(active, dtype=torch.long)
    maximum = int(max_iterations_factor) * op.maximum_graph_nodes
    started = last_progress = time.monotonic()
    performed = 0
    for iteration in range(1, maximum + 1):
        if not bool(active.any()):
            break
        now = time.monotonic()
        if progress is not None and now - last_progress >= 30:
            progress(iteration - 1, maximum, now - started)
            last_progress = now
        ap = fused_apply(op, p)
        normal_p = project_components(op, fused_transpose(op, ap))
        denominator = _receiver_sum(op, ap.square())
        if bool((active & (denominator <= 0)).any()):
            raise RuntimeError("normal CG breakdown before verified observation convergence")
        alpha = torch.where(
            active,
            rz / torch.where(denominator > 0, denominator, torch.ones_like(denominator)),
            0.0,
        )
        x += alpha[op.node_graph] * p
        r = project_components(op, r - alpha[op.node_graph] * normal_p)
        recurrence = _graph_sum(op, r.square()).sqrt() / safe_rhs
        inspect = iteration % 10 == 0 or bool((active & (recurrence <= tolerance)).any())
        if inspect:
            predicted = fused_apply(op, x)
            actual_r = rhs - fused_transpose(op, predicted)
            relative = _receiver_sum(op, (observation - predicted).square()).sqrt() / safe_y
            new_active = active & (relative > tolerance)
            iterations = torch.where(
                active & ~new_active, torch.full_like(iterations, iteration), iterations
            )
            r = project_components(op, actual_r)
        else:
            new_active = active
        z = project_components(op, r * inverse[:, None])
        next_rz = _graph_sum(op, r * z)
        beta = torch.where(new_active, next_rz / torch.where(rz != 0, rz, torch.ones_like(rz)), 0.0)
        p = (z + beta[op.node_graph] * p) * new_active[op.node_graph]
        rz, active, performed = next_rz, new_active, iteration
    x = project_components(op, x)
    predicted = fused_apply(op, x)
    relative = _receiver_sum(op, (observation - predicted).square()).sqrt() / safe_y
    normal_relative = (
        _graph_sum(op, (rhs - fused_transpose(op, predicted)).square()).sqrt() / safe_rhs
    )
    converged = relative <= tolerance
    if (
        not bool(torch.isfinite(x).all())
        or not bool(torch.isfinite(relative).all())
        or not bool(converged.all())
    ):
        raise RuntimeError(
            f"normal CG not converged: iterations={performed}/{maximum}, "
            f"maximum actual relative observation residual={float(relative.max()):.6g}, "
            f"tolerance={tolerance}"
        )
    return x, {
        "iterations": iterations,
        "relative_residual": relative,
        "normal_relative_residual": normal_relative,
        "converged": converged,
        "global_iterations": performed,
        "max_iterations": maximum,
        "minimum_norm_constraint": "zero_mean_per_physical_connected_component",
    }


def observation_jvp(op, local_field, local_direction, *, relation_batch=None):
    """Exact E/J derivative at local fields, with no nonlinear injectivity claim."""
    top = op.topology
    _validate(op, local_field, top.num_local_nodes, "local field")
    _validate(op, local_direction, top.num_local_nodes, "local direction")
    if local_field.shape != local_direction.shape:
        raise ValueError("field and direction need the same feature coordinates")
    list(_chunks(top, relation_batch))  # validates the pair batching contract
    g, delta_g = incidence(top, local_field), incidence(top, local_direction)
    q, delta_q = op.weights[:, None] * g, op.weights[:, None] * delta_g
    energy = q.new_zeros((top.n, q.shape[1])).index_add(0, top.local_edge_center, 2 * q * delta_g)
    d, delta_d = incidence_transpose(top, q), incidence_transpose(top, delta_q)

    def derivative(left, delta, kind):
        offsets = top.correspondence_offsets[CORRESPONDENCE_KINDS.index(kind)]
        li, ri, pairs = (
            getattr(top, kind + "_left"),
            getattr(top, kind + "_right"),
            getattr(top, kind + "_pair"),
        )
        result = left.new_zeros((top.num_pairs, left.shape[1]))
        batch = max(1, top.num_pairs) if relation_batch is None else int(relation_batch)
        for start in range(0, top.num_pairs, batch):
            begin, end = offsets[start], offsets[min(start + batch, top.num_pairs)]
            value = (
                delta[li[begin:end]] * left[ri[begin:end]]
                + left[li[begin:end]] * delta[ri[begin:end]]
            )
            result.index_add_(0, pairs[begin:end], value)
        return result

    shared = derivative(q, delta_q, "shared_edge")
    nodes = derivative(d, delta_d, "shared_node")
    return {
        "energy": energy,
        "shared_edges": shared,
        "shared_nodes": nodes,
        "distinct_edges": nodes - 2 * shared,
    }


__all__ = [
    "ReceiverOperator",
    "prepare_receiver_operator",
    "batch_receiver_operators",
    "shared_flow",
    "tagged_receipts",
    "fuse_receipts",
    "receipt_reconstruct",
    "ambient_receipt_projection",
    "fuse_divergence",
    "fuse_transpose",
    "fused_apply",
    "fused_transpose",
    "project_components",
    "graph_norm_sq",
    "recover_fused",
    "observation_jvp",
]
