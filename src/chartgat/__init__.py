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
