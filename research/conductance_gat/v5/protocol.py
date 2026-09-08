"""Protocol constants for graph-conditioned shared-conductance V5."""

from __future__ import annotations

import math
import os
from typing import Any

SUITE = "conductance_graph_conditioned_v5"
PARAMETERIZATION = "shared_unrolled_optimized_relative_c_with_multihead_w_and_graph_beta_v1"
CONDUCTANCE_BACKENDS = ("optimization", "mlp")
DEFAULT_CONDUCTANCE_BACKEND = "optimization"
DEFAULT_SOLVER_STEPS = 8
DEFAULT_SOLVER_STEP_SIZE = 0.25
DEFAULT_SOLVER_ENTROPY = 1.0
DEFAULT_SOLVER_DEGREE_BARRIER = 0.1
SOLVER_COST_SCALINGS = ("legacy_unit", "width_scaled")
CONDUCTANCE_HEAD_MODES = ("shared", "per_head")
PROPAGATION_NORMALIZATIONS = ("symmetric", "row")
CONDUCTANCE_GENERATORS = ("optimized", "degree_only", "entropy_exact")
PROPAGATION_FILTERS = ("linear", "polynomial3")
LEARNING_BUDGET_POLICIES = ("epochs", "reference_updates")
TRAINING_SCHEDULES = ("joint", "staged")
DEFAULT_TRAINING_SCHEDULE = "joint"
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


def conductance_configuration(
    conductance_backend: str = DEFAULT_CONDUCTANCE_BACKEND,
    solver_steps: int = DEFAULT_SOLVER_STEPS,
    solver_step_size: float = DEFAULT_SOLVER_STEP_SIZE,
    solver_entropy: float = DEFAULT_SOLVER_ENTROPY,
    solver_degree_barrier: float = DEFAULT_SOLVER_DEGREE_BARRIER,
    solver_cost_scaling: str = "legacy_unit",
    *,
    conductance_heads: str = "shared",
    propagation_normalization: str = "symmetric",
    conductance_generator: str = "optimized",
    num_relations: int = 0,
    edge_direction: str = "undirected",
    propagation_filter: str = "linear",
) -> dict[str, int | float | str]:
    """Canonical architecture identity; solver fields are inactive for the MLP ablation."""

    if conductance_backend not in CONDUCTANCE_BACKENDS:
        raise ValueError(f"unsupported conductance backend: {conductance_backend}")
    if solver_cost_scaling not in SOLVER_COST_SCALINGS:
        raise ValueError(f"unsupported solver cost scaling: {solver_cost_scaling}")
    if conductance_backend == "mlp" and solver_cost_scaling != "legacy_unit":
        raise ValueError("width_scaled costs require the optimization conductance backend")
    for name, value, choices in (
        ("conductance_heads", conductance_heads, CONDUCTANCE_HEAD_MODES),
        ("propagation_normalization", propagation_normalization, PROPAGATION_NORMALIZATIONS),
        ("conductance_generator", conductance_generator, CONDUCTANCE_GENERATORS),
        ("propagation_filter", propagation_filter, PROPAGATION_FILTERS),
    ):
        if value not in choices:
            raise ValueError(f"unsupported {name}: {value}")
    if isinstance(num_relations, bool) or not isinstance(num_relations, int) or num_relations < 0:
        raise ValueError("num_relations must be a nonnegative integer")
    if edge_direction != "undirected":
        raise ValueError(
            "B^T C B requires undirected physical relations; directed edges need a separate model"
        )
    if conductance_backend == "mlp" and (
        conductance_generator != "optimized" or conductance_heads != "shared" or num_relations
    ):
        raise ValueError("the explicit MLP control supports shared untyped C only")
    if conductance_generator == "entropy_exact" and solver_degree_barrier != 0:
        raise ValueError("entropy_exact requires solver_degree_barrier=0")
    if conductance_generator == "degree_only" and num_relations:
        raise ValueError("degree_only has no learned relation metric; use untyped topology control")
    if isinstance(solver_steps, bool) or not isinstance(solver_steps, int) or solver_steps < 1:
        raise ValueError("solver_steps must be a positive integer")
    values = {
        "solver_step_size": solver_step_size,
        "solver_entropy": solver_entropy,
        "solver_degree_barrier": solver_degree_barrier,
    }
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if name != "solver_degree_barrier" else value < 0)
        ):
            raise ValueError(
                f"{name} must be finite and "
                + ("nonnegative" if name == "solver_degree_barrier" else "positive")
            )
    return {
        "conductance_backend": conductance_backend,
        "solver_steps": solver_steps,
        **{name: float(value) for name, value in values.items()},
        # Omit inactive defaults to preserve historical configuration identities.
        **(
            {"solver_cost_scaling": solver_cost_scaling}
            if solver_cost_scaling != "legacy_unit"
            else {}
        ),
        **{
            name: value
            for name, value, default in (
                ("conductance_heads", conductance_heads, "shared"),
                ("propagation_normalization", propagation_normalization, "symmetric"),
                ("conductance_generator", conductance_generator, "optimized"),
                ("num_relations", num_relations, 0),
                ("edge_direction", edge_direction, "undirected"),
                ("propagation_filter", propagation_filter, "linear"),
            )
            if value != default
        },
    }


def add_conductance_arguments(parser, *, prefix: str = "") -> None:
    """Share explicit architecture CLI options across standalone and nested runners."""

    for name, choices, default, help_text in (
        (
            "conductance-heads",
            CONDUCTANCE_HEAD_MODES,
            "shared",
            "per_head learns independent C and degrees for every feature head",
        ),
        (
            "propagation-normalization",
            PROPAGATION_NORMALIZATIONS,
            "symmetric",
            "row uses receiver-normalized neighbor attention with row sum one",
        ),
        (
            "conductance-generator",
            CONDUCTANCE_GENERATORS,
            "optimized",
            "degree_only has no learned edge cost; entropy_exact requires zero degree barrier",
        ),
        (
            "propagation-filter",
            PROPAGATION_FILTERS,
            "linear",
            "polynomial3 adds learned degree-three propagation without shrinking the backbone",
        ),
    ):
        parser.add_argument(f"--{prefix}{name}", choices=choices, default=default, help=help_text)
    parser.add_argument(
        f"--{prefix}num-relations",
        type=int,
        default=0,
        help="actual physical-edge relation types; requires explicit edge_relation_id metadata",
    )
    parser.add_argument(f"--{prefix}edge-direction", choices=("undirected",), default="undirected")
    parser.add_argument(
        f"--{prefix}conductance-backend",
        choices=CONDUCTANCE_BACKENDS,
        default=DEFAULT_CONDUCTANCE_BACKEND,
        help="optimization unrolls C updates; mlp explicitly selects the legacy ablation",
    )
    parser.add_argument(f"--{prefix}solver-steps", type=int, default=DEFAULT_SOLVER_STEPS)
    parser.add_argument(
        f"--{prefix}solver-cost-scaling",
        choices=SOLVER_COST_SCALINGS,
        default="legacy_unit",
        help=(
            "width_scaled centers sqrt(width)-scaled quadratic costs before bounding; "
            "changes the training recipe"
        ),
    )
    parser.add_argument(
        f"--{prefix}learning-budget-policy",
        choices=LEARNING_BUDGET_POLICIES,
        default="epochs",
        help=(
            "reference_updates explicitly extends the epoch ceiling/patience "
            "when a larger batch reduces updates"
        ),
    )
    parser.add_argument(
        f"--{prefix}budget-reference-batch-size",
        type=int,
        help="reference physical batch; otherwise use the original hardware-profile batch",
    )
    parser.add_argument(f"--{prefix}solver-step-size", type=float, default=DEFAULT_SOLVER_STEP_SIZE)
    parser.add_argument(f"--{prefix}solver-entropy", type=float, default=DEFAULT_SOLVER_ENTROPY)
    parser.add_argument(
        f"--{prefix}solver-degree-barrier", type=float, default=DEFAULT_SOLVER_DEGREE_BARRIER
    )
    parser.add_argument(
        f"--{prefix}training-schedule",
        choices=TRAINING_SCHEDULES,
        default=DEFAULT_TRAINING_SCHEDULE,
        help="joint learns C/W from epoch one; staged is the explicit historical ablation",
    )


def conductance_arguments_configuration(args, *, prefix: str = "") -> dict[str, Any]:
    """Validate both solver architecture and schedule from a parsed namespace."""

    configuration = conductance_configuration(
        **{name: getattr(args, prefix + name) for name in conductance_configuration()},
        solver_cost_scaling=getattr(args, prefix + "solver_cost_scaling", "legacy_unit"),
        **{
            name: getattr(args, prefix + name, default)
            for name, default in (
                ("conductance_heads", "shared"),
                ("propagation_normalization", "symmetric"),
                ("conductance_generator", "optimized"),
                ("num_relations", 0),
                ("edge_direction", "undirected"),
                ("propagation_filter", "linear"),
            )
        },
    )
    schedule = getattr(args, prefix + "training_schedule")
    if schedule not in TRAINING_SCHEDULES:
        raise ValueError(f"unsupported training schedule: {schedule}")
    return {**configuration, "training_schedule": schedule}


def learning_budget_arguments_configuration(args, *, prefix: str = "") -> dict[str, Any]:
    """Keep budget policy separate from model architecture and legacy identities."""
    policy = getattr(args, prefix + "learning_budget_policy", "epochs")
    reference = getattr(args, prefix + "budget_reference_batch_size", None)
    if policy not in LEARNING_BUDGET_POLICIES:
        raise ValueError(f"unsupported learning budget policy: {policy}")
    if reference is not None and (
        isinstance(reference, bool) or not isinstance(reference, int) or reference < 1
    ):
        raise ValueError("budget reference batch size must be a positive integer")
    if policy == "epochs":
        if reference is not None:
            raise ValueError("budget reference batch size requires reference_updates")
        return {}
    if getattr(args, prefix + "training_schedule", "joint") != "joint":
        raise ValueError("reference_updates currently requires the explicit joint schedule")
    return {"learning_budget_policy": policy, "budget_reference_batch_size": reference}


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
    **conductance_configuration(),
    "training_schedule": DEFAULT_TRAINING_SCHEDULE,
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
SAMPLING_MODES = ("full", "neighbor", "cluster", "cluster_disjoint")
SAMPLING_CHOICES = ("auto", "auto_disjoint", *SAMPLING_MODES)


def resolve_sampling(dataset: str, requested: str) -> str:
    """Keep legacy auto stable; the new structural sampling recipe is explicit."""
    if requested not in SAMPLING_CHOICES:
        raise ValueError(f"unsupported sampling mode: {requested}")
    if requested in {"auto", "auto_disjoint"}:
        if dataset != "ogbn-arxiv":
            return "full"
        return "cluster_disjoint" if requested == "auto_disjoint" else "cluster"
    return requested


def add_sampling_context_arguments(parser, *, prefix: str = "") -> None:
    parser.add_argument(
        f"--{prefix}sample-context-seed-batch-size",
        type=int,
        help=(
            "required for cluster_disjoint/auto_disjoint: supervised seeds per independent "
            "B context, separate from the total physical seed batch; changes sampling recipe"
        ),
    )
    parser.add_argument(
        f"--{prefix}sample-context-workers",
        type=int,
        help=(
            "CPU context-construction threads, separate from DataLoader workers; omission "
            "starts from up to four allocated CPUs and rich calibration measures alternatives"
        ),
    )


def sampling_context_configuration(args, *, prefix: str = "", sampling=None) -> dict[str, int]:
    requested = getattr(args, prefix + "sampling", "full")
    mode = requested if sampling is None else sampling
    size = getattr(args, prefix + "sample_context_seed_batch_size", None)
    workers = getattr(args, prefix + "sample_context_workers", None)
    if mode not in {"auto_disjoint", "cluster_disjoint"}:
        if requested not in {"auto_disjoint", "cluster_disjoint"} and (
            size is not None or workers is not None
        ):
            raise ValueError("context options require explicit cluster_disjoint/auto_disjoint")
        return {}
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("disjoint sampling requires a positive explicit context seed batch size")
    if workers is None:
        affinity = getattr(os, "sched_getaffinity", None)
        cpus = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
        workers = min(4, cpus)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("sample context workers must be a positive integer")
    return {"sample_context_seed_batch_size": size, "sample_context_workers": workers}


TRAINING_PHASES = ("spatial_warmup", "conductance_calibration", "alternating", "joint")

# The default joint schedule updates every active group from epoch one. The
# historical staged ablation allocates different C/backbone steps across arms.
# Capacity and actual optimizer-step counts remain mandatory audit metadata.
COMPARISON_DESIGN = {
    "estimand": "fixed-C training recipe versus shared-dynamic-C training recipe",
    "single_factor_causal_effect_of_c": False,
    "unequal_parameter_group_update_allocation": {"joint": False, "staged": True},
    "default_training_schedule": DEFAULT_TRAINING_SCHEDULE,
    "required_audit_field": "effective_optimizer_steps_by_group",
    "parameterization": (
        "fixed C=1 is parameter-free; dynamic C adds the active optimizer-cost parameters "
        "or the explicitly selected legacy MLP scorer. C is optimized by a finite unrolled "
        "inner solver in the default backend, not stored as a per-edge parameter table. "
        "Shared backbone/W/beta initialization is paired and hash-verified, while total "
        "parameter capacity is reported separately rather than padded with unused weights"
    ),
    "checkpoint_selection": {
        "primary": (
            "joint: both arms select their all-epoch validation best because C is active "
            "from epoch one; staged: fixed_c selects all-epoch best and shared_dynamic_c "
            "selects C-active best from calibration, alternating, or joint phases"
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
    "Default dynamic C is a symmetric positive mean-one field computed by finite unrolled "
    "optimization; legacy MLP-C remains an explicit ablation. Solver steps do not certify "
    "convergence, so residuals are recorded. beta carries identifiable "
    "diffusion magnitude. Transductive datasets may train on dependency-free samples that "
    "retain original degrees and apply explicit boundary correction; validation remains the "
    "complete official graph. PPI retains its official 20/2/2 split. No test labels are used."
    " Default training jointly learns all active groups from epoch one; the explicit staged "
    "ablation retains historical phase allocations. Parameter capacity and actual group "
    "updates are reported rather than claiming a pure single-C causal effect."
    " The a6000-48gb profile changes real batch/sample size and numeric execution, so it must "
    "not be pooled with or directly contrasted against portable-profile metrics."
)
