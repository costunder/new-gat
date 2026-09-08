"""Separate scientific identity for topology selection, not a V5 resume migration."""

from __future__ import annotations

import math

SUITE = "conductance_edge_selection_v1"
MODES = ("full", "forest_only", "forest_random", "forest_learned", "forest_cycle", "hard_concrete")
BUDGET_MODES = {"forest_random", "forest_learned", "forest_cycle"}


def add_arguments(parser):
    parser.add_argument("--selection-mode", choices=MODES, required=True)
    parser.add_argument("--chord-fraction", type=float)
    parser.add_argument("--forest-seed", type=int, default=0)
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--gate-temperature", type=float, default=2 / 3)
    parser.add_argument("--corruption-ratio", type=float, default=0.0)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--negative-loss-weight", type=float, default=0.0)
    parser.add_argument("--l0-weight", type=float, default=0.0)


def validate(args):
    if args.selection_mode not in MODES:
        raise ValueError("unknown edge selection condition")
    fraction = args.chord_fraction
    if args.selection_mode in BUDGET_MODES:
        if fraction is None or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError(
                "chord comparisons require an explicit fraction strictly between 0 and 1"
            )
    elif fraction is not None:
        raise ValueError("chord-fraction is inactive outside budgeted chord comparisons")
    for name in ("selection_temperature", "gate_temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("corruption_ratio", "negative_loss_weight", "l0_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.forest_seed < 0 or args.corruption_seed < 0:
        raise ValueError("forest and corruption seeds must be nonnegative")
    if (
        args.selection_mode not in {"forest_learned", "forest_cycle"}
        and args.selection_temperature != 1.0
    ):
        raise ValueError("selection-temperature is inactive outside learned exact-budget gates")
    if args.selection_mode != "hard_concrete" and args.gate_temperature != 2 / 3:
        raise ValueError("gate-temperature is inactive outside hard-concrete gates")
    if args.selection_mode != "hard_concrete" and args.corruption_seed != 0:
        raise ValueError("corruption-seed is inactive in clean structure comparisons")
    if args.selection_mode == "hard_concrete":
        if args.corruption_ratio <= 0:
            raise ValueError(
                "the corruption experiment requires a positive explicit corruption ratio"
            )
        if args.forest_seed != 0:
            raise ValueError("hard-concrete corruption does not protect or select a forest")
    elif args.corruption_ratio or args.negative_loss_weight or args.l0_weight:
        raise ValueError(
            "corruption and auxiliary losses belong only to the separate hard-concrete experiment"
        )
    if args.conductance_heads != "per_head" or args.propagation_normalization != "row":
        raise ValueError(
            "edge-selection comparisons fix all amplitude heads to per_head and propagation to row"
        )
    if args.conductance_backend != "optimization" or args.conductance_generator != "optimized":
        raise ValueError(
            "edge-selection comparisons retain the optimized positive amplitude generator"
        )
    if args.propagation_filter != "linear" or args.num_relations:
        raise ValueError(
            "this comparison fixes linear untyped propagation; no silent relation conversion"
        )
    if args.condition != "shared_dynamic_c" or args.training_schedule != "joint":
        raise ValueError(
            "all edge-selection arms train the positive amplitude and backbone jointly"
        )
    if args.transition_from_checkpoint is not None:
        raise ValueError("edge-selection runs cannot import or relabel historical V5 training")


def model_configuration(args):
    result = {
        "condition": args.selection_mode,
        "selection_temperature": args.selection_temperature,
        "hard_concrete_temperature": args.gate_temperature,
        "hard_concrete_lower": -0.1,
        "hard_concrete_upper": 1.1,
        "selection_seed": args.forest_seed,
    }
    if args.selection_mode in BUDGET_MODES:
        result["chord_fraction"] = args.chord_fraction
    return result


def configuration(args):
    return {
        "model": model_configuration(args),
        "forest_seed": args.forest_seed,
        "corruption_ratio": args.corruption_ratio,
        "corruption_seed": args.corruption_seed,
        "negative_loss_weight": args.negative_loss_weight,
        "l0_weight": args.l0_weight,
        "gate_axis": (
            "one physical topology shared across heads; positive amplitude remains per-head"
        ),
        "connectivity_scope": (
            "each supplied graph or sampled B_s; not unsampled full-graph reachability"
        ),
        "negative_provenance_model_input": False,
        "negative_definition": "synthetically added nonedge, not a negative signed conductance",
        "budget_rule": "floor(chord_fraction * number_of_candidate_chords) per disjoint graph",
        "sampling_axis": "same V5 sampling law and seed across topology comparison arms",
    }
