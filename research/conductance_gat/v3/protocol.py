"""Dependency-free protocol for the first relative-conductance v3 experiment."""

SUITE = "conductance_relative_c_v3"
PARAMETERIZATION = "shared_relative_log_conductance"
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
DEFAULT_DATASETS = DATASETS
DEFAULT_EDGE_CHUNK_SIZE = 65536
COMMON = {
    "hidden_channels": 64,
    "layers": 2,
    "dropout": 0.5,
    "lr": 0.005,
    "weight_decay": 0.0005,
    "amp": False,
    "compile": False,
    "optimizer": "AdamW",
    "gate_lr_multiplier": 2.0,
    "scalar_weight_decay": 0.0,
}
CONDITIONS = {
    "relative_c": {
        "normalization": "symmetric",
        "gate_mode": "relative",
        "gate_weight_decay": 0.0,
    },
    "fixed_c": {
        "normalization": "symmetric",
        "gate_mode": "fixed_one",
        "gate_weight_decay": 0.0,
    },
}
PROTOCOL_NOTE = (
    "Shared graph-centered relative conductance, symmetric normalization and a separate "
    "learnable alpha; both arms initialize C=1 and alpha=.5. Fixed C freezes the entire "
    "estimator but keeps alpha trainable. AdamW separates backbone, gate and scalar controls. "
    "Optimization uses official train labels only; validation labels select checkpoints, "
    "with no test evaluation. "
    "PPI uses the official 20/2/2 split: train 20 and validation 2 run in whole-graph "
    "minibatches of 2 with BCEWithLogits and global logit>0 node-label micro-F1; test 2 is "
    "not scored. "
    "V1/V2 are unchanged and their historical scores are not a matched one-factor contrast. "
    "Graph means and degrees are full graph, not chunk-local; first-order gradients only. "
    "The kernel is sparse/chunked; citation/arxiv training remains full graph while PPI uses "
    "whole-graph minibatches without neighbor sampling."
)
