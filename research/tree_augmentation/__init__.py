"""Lossless spanning-tree chart augmentation for static Cycle PE.

This package intentionally depends only on shared graph/algebra utilities.  It
does not expose conductance, attention, potential, or flow-completion models.
"""

from .augmentation import (
    TreeChart,
    build_tree_chart,
    chart_probe_features,
    cycle_projector,
    cycle_projector_diagonal,
    ensure_full_cycle_budget,
    evaluate_probe,
    find_unseen_chart,
    lossless_transition_error,
    run_static_cycle_pe_probe,
    sample_tree_charts,
    train_probe,
    transition_cocycle_error,
    transport_coordinates,
)
from .paper_data import build_paper_chart, wilson_ust_indices
from .paper_model import VariableBetaCycleEncoder

__all__ = [
    "TreeChart",
    "VariableBetaCycleEncoder",
    "build_paper_chart",
    "build_tree_chart",
    "chart_probe_features",
    "cycle_projector",
    "cycle_projector_diagonal",
    "ensure_full_cycle_budget",
    "evaluate_probe",
    "find_unseen_chart",
    "lossless_transition_error",
    "run_static_cycle_pe_probe",
    "sample_tree_charts",
    "train_probe",
    "transition_cocycle_error",
    "transport_coordinates",
    "wilson_ust_indices",
]
