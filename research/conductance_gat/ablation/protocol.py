"""Dependency-free specification of the single-seed, two-factor experiment."""

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
DEFAULT_DATASETS = ("ppi", "ogbn-arxiv")
CONDITIONS = {
    "baseline": {"normalization": "global_max", "gate_weight_decay": 0.0005},
    "gate_no_wd": {"normalization": "global_max", "gate_weight_decay": 0.0},
    "node_degree": {"normalization": "node_degree", "gate_weight_decay": 0.0005},
    "node_degree_gate_no_wd": {"normalization": "node_degree", "gate_weight_decay": 0.0},
}
COMMON = {
    "hidden_channels": 64,
    "layers": 2,
    "dropout": 0.5,
    "lr": 0.005,
    "weight_decay": 0.0005,
    "amp": False,
    "compile": False,
}
