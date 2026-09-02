"""Dependency-free protocol for the matched conductance-by-spatial v4 experiment."""

SUITE = "conductance_hybrid_c_spatial_v4"
PARAMETERIZATION = "shared_relative_log_conductance_x_spatial_message_transform"
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
DEFAULT_DATASETS = DATASETS
BATCH_SIZE_BY_DATASET = {dataset: 2 if dataset == "ppi" else 1 for dataset in DATASETS}
METRIC_BY_DATASET = {
    dataset: "micro_f1" if dataset == "ppi" else "accuracy" for dataset in DATASETS
}
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
    "fixed_c_identity_w": {
        "normalization": "symmetric",
        "gate_mode": "fixed_one",
        "spatial_mode": "fixed_identity",
        "gate_weight_decay": 0.0,
    },
    "relative_c_identity_w": {
        "normalization": "symmetric",
        "gate_mode": "relative",
        "spatial_mode": "fixed_identity",
        "gate_weight_decay": 0.0,
    },
    "fixed_c_spatial_w": {
        "normalization": "symmetric",
        "gate_mode": "fixed_one",
        "spatial_mode": "learned",
        "gate_weight_decay": 0.0,
    },
    "relative_c_spatial_w": {
        "normalization": "symmetric",
        "gate_mode": "relative",
        "spatial_mode": "learned",
        "gate_weight_decay": 0.0,
    },
}
PROTOCOL_NOTE = (
    "A matched 2x2 factorial separates graph-operator adaptation by relative C from a "
    "bias-free spatial feature transform W. Every arm allocates the same estimator and W "
    "state, starts at C=1, W=I and alpha=.5, and trains alpha. Inactive estimator or W "
    "groups are frozen and excluded from AdamW; active W uses the ordinary backbone learning "
    "rate and weight decay. C is computed from the pre-W state, while symmetric propagation "
    "aggregates H W. Graph means and C-dependent weighted degrees are exact full-graph "
    "quantities; edge computation is chunked and first-order only. Cora, CiteSeer, PubMed and "
    "ogbn-arxiv use their original transductive full graph and official node masks. PPI uses the "
    "official 20/2/2 inductive graph split, batch size 2, binary cross entropy and global "
    "node-label micro-F1. Optimization uses official train labels only; validation labels "
    "select checkpoints, with no test evaluation. The V3 model definition is unchanged and "
    "cross-version score differences are not a matched single-factor causal contrast. The "
    "report releases five within-V4 factorial contrasts only after all four fresh arms and "
    "source integrity pass; selected-checkpoint C/W interventions are read-only diagnostics."
)
