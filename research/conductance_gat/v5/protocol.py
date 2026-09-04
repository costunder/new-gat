"""Protocol constants for graph-conditioned shared-conductance V5."""

from __future__ import annotations

import math

SUITE = "conductance_graph_conditioned_v5"
PARAMETERIZATION = "shared_dynamic_relative_c_with_multihead_w_and_graph_beta"
BETA_PARAMETERIZATIONS = ("sigmoid", "margin_sigmoid")
DEFAULT_BETA_PARAMETERIZATION = "sigmoid"
DEFAULT_BETA_INITIAL = 0.1
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
DEFAULT_DATASETS = DATASETS
METRIC_BY_DATASET = {
    dataset: "micro_f1" if dataset == "ppi" else "accuracy" for dataset in DATASETS
}
BATCH_SIZE_BY_DATASET = {dataset: 2 if dataset == "ppi" else 1 for dataset in DATASETS}
DEFAULT_EDGE_CHUNK_SIZE = 65536


def beta_configuration(
    beta_parameterization: str = DEFAULT_BETA_PARAMETERIZATION,
    beta_initial: float = DEFAULT_BETA_INITIAL,
    beta_min: float | None = None,
    beta_max: float | None = None,
) -> dict[str, float | str]:
    """Validate and canonicalize the graph-beta parameterization contract."""

    if beta_parameterization not in BETA_PARAMETERIZATIONS:
        raise ValueError(f"unsupported beta parameterization: {beta_parameterization}")
    if (
        isinstance(beta_initial, bool)
        or not isinstance(beta_initial, (int, float))
        or not math.isfinite(beta_initial)
    ):
        raise ValueError("beta_initial must be finite")
    initial = float(beta_initial)
    if beta_parameterization == "sigmoid":
        if beta_min is not None or beta_max is not None:
            raise ValueError("beta_min/beta_max are only valid for margin_sigmoid")
        if not 0 < initial < 1:
            raise ValueError("sigmoid beta_initial must be strictly inside (0, 1)")
        return {
            "beta_parameterization": beta_parameterization,
            "beta_initial": initial,
        }
    if beta_min is None or beta_max is None:
        raise ValueError("margin_sigmoid requires explicit beta_min and beta_max")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in (beta_min, beta_max)
    ):
        raise ValueError("beta_min and beta_max must be finite")
    lower, upper = float(beta_min), float(beta_max)
    if not 0 <= lower < initial < upper <= 1:
        raise ValueError("margin_sigmoid beta bounds must satisfy 0 <= min < initial < max <= 1")
    return {
        "beta_parameterization": beta_parameterization,
        "beta_initial": initial,
        "beta_min": lower,
        "beta_max": upper,
    }


# Hardware profiles leave the V5 architecture unchanged but alter the numeric
# execution and optimization recipe (including precision and real batch/sample
# sizes), so metrics must not be compared across profiles.  The A6000 profile
# is opt-in and is checked against the *visible* device at child start so it
# cannot accidentally be used on a 10 GiB MIG slice.
HARDWARE_PROFILES = {
    "portable": {
        "precision": "fp32",
        "tf32": False,
        "activation_checkpoint": True,
        "edge_chunk_size": DEFAULT_EDGE_CHUNK_SIZE,
        "sample_seed_batch_size": 1024,
        "ppi_batch_size": 2,
        "sample_prefetch": False,
        "pin_memory": True,
        "minimum_total_memory_gib": 0.0,
        "minimum_free_memory_gib": 0.0,
        "minimum_compute_capability_major": 0,
    },
    "a6000-48gb": {
        "precision": "bf16",
        "tf32": True,
        "activation_checkpoint": False,
        "edge_chunk_size": 131072,
        "sample_seed_batch_size": 2048,
        "ppi_batch_size": 8,
        "sample_prefetch": True,
        "pin_memory": True,
        "minimum_total_memory_gib": 40.0,
        "minimum_free_memory_gib": 32.0,
        "minimum_compute_capability_major": 8,
    },
}

# Research-scale defaults; V1--V4's 64-wide/two-layer settings remain mechanism probes.
SCALE_PROFILES = {
    "reference": {
        "hidden_channels": 256,
        "layers": 8,
        "heads": 8,
        "ffn_multiplier": 4,
        "dropout": 0.2,
    },
    "large": {
        "hidden_channels": 384,
        "layers": 12,
        "heads": 8,
        "ffn_multiplier": 4,
        "dropout": 0.2,
    },
}
COMMON = {
    **SCALE_PROFILES["reference"],
    "lr": 0.0005,
    "conductance_lr_multiplier": 1.0,
    "beta_lr_multiplier": 1.0,
    "weight_decay": 0.01,
    "conductance_weight_decay": 0.0,
    "scalar_weight_decay": 0.0,
    "optimizer": "AdamW",
    "max_log_conductance": 2.0,
    "beta_parameterization": DEFAULT_BETA_PARAMETERIZATION,
    "beta_initial": DEFAULT_BETA_INITIAL,
    "amp": False,
    "compile": False,
    "gradient_clip_norm": 5.0,
}

# W_h, graph-conditioned beta_h, FFNs and the classifier are learned in both arms.
CONDITIONS = {
    "fixed_c": {"conductance_mode": "fixed_one"},
    "shared_dynamic_c": {"conductance_mode": "dynamic"},
}
SAMPLING_MODES = ("full", "neighbor", "cluster")
TRAINING_PHASES = ("spatial_warmup", "conductance_calibration", "alternating", "joint")

# This is deliberately an end-to-end recipe comparison.  The fixed arm spends
# dynamic-C calibration/alternation turns updating its spatial model, whereas
# the dynamic arm spends those turns on C.  Effective group step counts are
# therefore mandatory output metadata and the contrast is not a one-variable
# causal estimate of merely replacing C=1.
COMPARISON_DESIGN = {
    "estimand": "fixed-C training recipe versus shared-dynamic-C training recipe",
    "single_factor_causal_effect_of_c": False,
    "unequal_parameter_group_update_allocation": True,
    "required_audit_field": "effective_optimizer_steps_by_group",
    "checkpoint_selection": {
        "primary": (
            "fixed_c selects its all-epoch validation best; shared_dynamic_c selects its "
            "C-active validation best from calibration, alternating, or joint phases"
        ),
        "auxiliary_prediction": "all-epoch validation best is reported for both arms",
        "early_stopping": (
            "fixed_c monitors its all-epoch best; shared_dynamic_c monitors a separate "
            "joint-phase best so warmup cannot terminate C training"
        ),
    },
    "hardware_profile_comparability": (
        "compare fixed_c versus shared_dynamic_c only under the same hardware profile; "
        "portable and a6000-48gb are distinct optimization recipes"
    ),
    "resume_semantics": (
        "epoch-boundary deterministic resume with exact stored RNG/optimizer/model state; "
        "CUDA kernels are not claimed bitwise deterministic"
    ),
}

PROTOCOL_NOTE = (
    "V5 compares fixed C=1 against one shared graph-conditioned dynamic conductance field "
    "per layer. Both arms learn identical multi-head W_h and graph-conditioned beta_h. "
    "The default beta is an un-margined sigmoid with nominal beta_0=0.1; the historical "
    "bounded-margin sigmoid remains an explicit ablation. "
    "Dynamic C is symmetric, positive, bounded and relative; beta carries identifiable "
    "diffusion magnitude. Transductive datasets may train on dependency-free samples that "
    "retain original degrees and apply explicit boundary correction; validation remains the "
    "complete official graph. PPI retains its official 20/2/2 split. No test labels are used."
    " The two arms use intentionally different phase-wise parameter-group update allocations, "
    "so their reported contrast is a recipe comparison, not a single-C causal effect."
    " The a6000-48gb profile changes real batch/sample size and numeric execution, so it must "
    "not be pooled with or directly contrasted against portable-profile metrics."
)
