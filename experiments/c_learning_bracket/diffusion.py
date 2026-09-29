"""The existing row diffusion, with checks on the actual degree and coefficients."""

import torch

from research.conductance_gat.v5.operator import (
    _ChunkedHeadPropagation,
    conductance_propagation_coefficients,
    graph_broadcast,
)


def row_coefficients(c, edges, nodes, *, correction=None, edge_chunk_size):
    tail, head, degree = conductance_propagation_coefficients(
        c,
        edges,
        nodes,
        sampling_correction=correction,
        normalization="row",
        edge_chunk_size=edge_chunk_size,
    )
    torch._assert_async(
        (torch.isfinite(degree) & (degree >= 0)).all(),
        "nonfinite/negative weighted degree; stop without changing weights",
    )
    torch._assert_async(
        (torch.isfinite(tail) & torch.isfinite(head) & (tail >= 0) & (head >= 0)).all(),
        "nonfinite/negative row coefficients; stop without changing weights",
    )
    return tail, head, degree


def row_diffusion(value, weight, edges, groups, beta, *, edge_chunk_size):
    dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
    message = value.to(dtype)
    tail, head, degree = row_coefficients(
        weight.to(dtype),
        edges,
        value.shape[0],
        edge_chunk_size=edge_chunk_size,
    )
    propagated = _ChunkedHeadPropagation.apply(message, tail, head, edges, edge_chunk_size)
    active = (degree > 0).unsqueeze(-1)
    node_beta = graph_broadcast(beta.to(dtype), groups, beta.shape[0]).unsqueeze(-1)
    output = message + node_beta * (propagated - active * message)
    return output.to(value.dtype), (tail, head)
