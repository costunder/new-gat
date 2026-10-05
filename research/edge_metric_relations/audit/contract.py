"""Complete fixed audit contract; runtime chunks do not alter data coverage."""
from __future__ import annotations

import json
from pathlib import Path

from ..common import digest, file_sha256, read_json, source_manifest, assert_source_unchanged, write_json

FOLDER = Path(__file__).resolve().parent
OPERATOR_PLAN = {"L0": (1,), "Ld": (1, 2), "smooth_L0": (1, 2, 4, 8, 16),
                 "smooth_Ld": (1, 2, 4, 8, 16), "copy_off": (1,), "copy_on": (1,),
                 "Q_diag": (1,), "P_diag": (1,), "Q_F0": (1,), "P_F0": (1,),
                 "Q_reference": (1,), "P_reference": (1,)}


def profile_config(profile="full"):
    if profile not in ("full", "debug"):
        raise ValueError("profile must be full or debug")
    return {"schema": "edge-metric-observability-audit-v2", "experiment": "edge_metric_relations_audit",
            "profile": profile, "precision": "float64", "recipes": ["unit", "local_degree"],
            "operator_plan": {key: list(value) for key, value in OPERATOR_PLAN.items()},
            "observations": ["full", "target", "one_hop"], "targets": ["q", "E", "J_shared", "J_node", "J_distinct"],
            "source_graphs": 201 if profile == "full" else 21, "sampling_ratio": 1.0,
            "all_original_nodes_edges_channels": True, "synthetic_scalar_channels_independent": True,
            "citation_channels_one_vector_trace": True, "fixed_reference_coefficients": True,
            "epsilon": 1e-4, "rank_relative_tolerance": 1e-10,
            "noise_levels": [0., 1e-6, 1e-3],
            "noise": "deterministic_sine_probe_scaled_to_observed_RHS_norm_not_natural_data",
            "solver": {"tolerance": 1e-8, "maximum_iterations_factor": 10,
                       "iteration_rule": "10*max_observed_nodes_in_work_chunk",
                       "regularizer": 0., "linear_system": "primal_AT_A_x_AT_y_zero_start_minimum_norm_least_squares",
                       "convergence": "true_normal_stationarity_residual_and_no_breakdown",
                       "observation_residual": "separate_irreducible_noise_or_model_mismatch_diagnostic",
                       "successful_summary": "whole_original_vector_all_constituent_channels_converged",
                       "unconverged": "raw_iterates_explicit_excluded_from_successful_recovery_summary"},
            "training": {"trainable_parameters": 0, "optimizer_updates": 0, "epochs": "N/A"},
            "runtime": {"gpu_memory_safety_fraction": .75, "cpu_workers": "auto", "cpu_threads": "auto",
                        "target_chunk": "auto", "channel_chunk": "auto", "pair_chunk": "auto",
                        "physical_graph_batch": "auto", "allocated_gpus": "all_visible", "exact_chunks": True},
            "trained_model_audit": "not_available_without_validated_selected_model_checkpoints",
            "full_server_only": True}


def read_config(profile="full", path=None):
    expected = profile_config(profile)
    candidate = expected if path is None else read_json(path)
    if candidate != expected:
        raise ValueError("changed fixed audit science/runtime contract; full coverage is mandatory")
    return candidate


def expected_target_rows(topology, realizations):
    return (2*topology.n+3*topology.num_pairs)*realizations


def validate_completion(value, config):
    required = {"completed": True, "status": "complete", "experiment": config["experiment"],
                "profile": config["profile"], "graphs": config["source_graphs"],
                "math_checks_passed": True, "all_original_nodes_edges_channels": True,
                "optimizer_updates": 0, "classifier_training_run": False,
                "trained_model_audit_run": False,
                "least_squares_solver": "primal_CG_zero_start_or_direct_SVD",
                "convergence_criterion": "true_relative_normal_stationarity_residual_and_no_breakdown",
                "successful_summary_scope": "all_constituent_channels_solver_converged_only"}
    if any(value.get(key) != item for key, item in required.items()):
        raise RuntimeError("incomplete/mismatched fixed audit completion")
    if not isinstance(value.get("source_digest"), str) or len(value["source_digest"]) != 64:
        raise RuntimeError("missing scientific source digest")
    for key in ("stationarity_unresolved_channel_solves", "observation_residual_outside_tolerance_channel_solves", "unresolved_target_rows"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise RuntimeError(f"missing/invalid least-squares diagnostic: {key}")
    return value
