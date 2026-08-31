"""Direct, graph-bound edge conductances; separate from the shared-MLP experiment."""

from ..ablation.protocol import COMMON

SUITE = "conductance_direct_c_v2"
PARAMETERIZATION = "direct_log_edge_conductance"
DATASETS = ("cora", "citeseer", "pubmed", "ogbn-arxiv")
DEFAULT_DATASETS = ("ogbn-arxiv",)
DEFAULT_EDGE_CHUNK_SIZE = 65_536
CONDITIONS = {
    "direct_c": {
        "normalization": "node_degree",
        "gate_mode": "direct",
        "gate_weight_decay": 0.0,
    },
    "fixed_c": {
        "normalization": "node_degree",
        "gate_mode": "fixed_one",
        "gate_weight_decay": 0.0,
    },
}

PROTOCOL_NOTE = (
    "Transductive, graph-bound per-edge parameters, not an inductive edge law. "
    "Each layer starts at C=1; direct log C has no weight decay. "
    "Official train labels supply cross-entropy gradients; validation selects checkpoints. "
    "No test evaluation, old checkpoint reuse, graph sampling, or PPI transfer. "
    "Exact edge chunking preserves the full-graph weighted degree. "
    "Common positive C scale cancels under row normalization."
)

__all__ = [
    "COMMON",
    "CONDITIONS",
    "DATASETS",
    "DEFAULT_DATASETS",
    "DEFAULT_EDGE_CHUNK_SIZE",
    "PARAMETERIZATION",
    "PROTOCOL_NOTE",
    "SUITE",
]
