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
