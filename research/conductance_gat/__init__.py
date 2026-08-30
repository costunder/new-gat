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

__all__ = [
    "IncidenceConductanceAttention",
    "IsotropicConductanceAttention",
    "PositiveInvariantScalarConductance",
    "PackedGraphBatch",
    "SparseIncidenceConductanceLayer",
    "SparsePositiveConductance",
    "edge_divergence",
    "edge_gradient",
    "pack_graph_examples",
]
