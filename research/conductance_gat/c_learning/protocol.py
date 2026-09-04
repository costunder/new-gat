"""Dependency-free, single-seed learned-C contribution experiment specification."""

from ..ablation.protocol import COMMON, DATASETS, DEFAULT_DATASETS

SUITE = "conductance_c_learning"
CONDITIONS = {
    "learned_c": {
        "normalization": "node_degree",
        "gate_mode": "learned",
        "gate_weight_decay": 0.0005,
    },
    "fixed_c": {
        "normalization": "node_degree",
        "gate_mode": "fixed_one",
        # There is no gate parameter in this arm and therefore no gate WD.
        "gate_weight_decay": 0.0,
    },
}

__all__ = ["COMMON", "CONDITIONS", "DATASETS", "DEFAULT_DATASETS", "SUITE"]
