"""E0: algebraic, orientation-gauge, and chart-equivariance certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chartgat.algebra import (
    chart_transition,
    decode_edge_state,
    encode_edge_state,
    flip_cycle_basis,
    flip_incidence,
    fundamental_cycle_basis,
    incidence_matrix,
)
from chartgat.graphs import make_connected_graph, spanning_tree_indices
from research.combined_later.layers import PersistentTransportBlock


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def certify_graph(seed: int, num_nodes: int, extra_edges: int, depth: int) -> dict[str, float]:
    edges = make_connected_graph(num_nodes, extra_edges, seed=seed)
    B = incidence_matrix(num_nodes, edges)
    trees = [
        spanning_tree_indices(num_nodes, edges, mode="bfs"),
        spanning_tree_indices(num_nodes, edges, mode="dfs"),
        spanning_tree_indices(num_nodes, edges, mode="random", seed=seed + 17),
    ]
    bases_and_chords = [fundamental_cycle_basis(B, tree, return_chords=True) for tree in trees]
    bases = [item[0] for item in bases_and_chords]

    cycle_null_error = max(_max_abs(B.T @ basis) for basis in bases)
    chord_identity_error = max(
        _max_abs(basis[chords] - np.eye(basis.shape[1])) for basis, chords in bases_and_chords
    )
    M10 = chart_transition(bases[0], bases[1])
    M21 = chart_transition(bases[1], bases[2])
    M20 = chart_transition(bases[0], bases[2])
    transition_error = max(
        _max_abs(bases[1] @ M10 - bases[0]),
        _max_abs(bases[2] @ M21 - bases[1]),
    )
    cocycle_error = _max_abs(M21 @ M10 - M20)

    rng = np.random.default_rng(seed + 101)
    edge_state = rng.normal(size=(B.shape[0], 3))
    p, a = encode_edge_state(B, bases[0], edge_state)
    reconstruction_error = _max_abs(decode_edge_state(B, bases[0], p, a) - edge_state)

    signs = rng.choice(np.asarray([-1.0, 1.0]), size=B.shape[0])
    B_flipped = flip_incidence(B, signs)
    F_flipped = flip_cycle_basis(bases[0], signs)
    orientation_null_error = _max_abs(B_flipped.T @ F_flipped)

    torch.manual_seed(seed)
    dtype = torch.float64
    B_t = torch.as_tensor(B, dtype=dtype)
    F0_t = torch.as_tensor(bases[0], dtype=dtype)
    F1_t = torch.as_tensor(bases[1], dtype=dtype)
    M10_t = torch.as_tensor(M10, dtype=dtype)
    potential = torch.randn(num_nodes, 2, dtype=dtype)
    potential = potential - potential.mean(dim=0, keepdim=True)
    a0 = torch.randn(bases[0].shape[1], 2, dtype=dtype)
    a1 = M10_t @ a0
    edge_features = torch.randn(B.shape[0], 2, dtype=dtype)
    block = PersistentTransportBlock(channels=2, hidden_channels=16, edge_feature_channels=2).to(
        dtype=dtype
    )

    p0, p1 = potential.clone(), potential.clone()
    for _ in range(depth):
        p0, a0 = block(B_t, F0_t, p0, a0, edge_features=edge_features)
        p1, a1 = block(B_t, F1_t, p1, a1, edge_features=edge_features)
    multilayer_p_error = float(torch.max(torch.abs(p0 - p1)).item())
    multilayer_cycle_error = float(torch.max(torch.abs(F0_t @ a0 - F1_t @ a1)).item())

    Q_signs = torch.as_tensor(signs, dtype=dtype)
    Bq_t = Q_signs[:, None] * B_t
    Fq_t = Q_signs[:, None] * F0_t
    p_base, a_base = block(
        B_t, F0_t, potential, torch.as_tensor(a, dtype=dtype)[:, :2], edge_features
    )
    p_flip, a_flip = block(
        Bq_t, Fq_t, potential, torch.as_tensor(a, dtype=dtype)[:, :2], edge_features
    )
    orientation_layer_error = max(
        float(torch.max(torch.abs(p_base - p_flip)).item()),
        float(torch.max(torch.abs(Fq_t @ a_flip - Q_signs[:, None] * (F0_t @ a_base))).item()),
    )

    return {
        "cycle_null_error": cycle_null_error,
        "chord_identity_error": chord_identity_error,
        "transition_error": transition_error,
        "cocycle_error": cocycle_error,
        "reconstruction_error": reconstruction_error,
        "orientation_null_error": orientation_null_error,
        "multilayer_p_error": multilayer_p_error,
        "multilayer_cycle_error": multilayer_cycle_error,
        "orientation_layer_error": orientation_layer_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/combined_later/certification.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--graphs", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=9)
    parser.add_argument("--extra-edges", type=int, default=7)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Certification output already exists: {args.output}")
    per_graph = [
        certify_graph(args.seed + index, args.nodes, args.extra_edges, args.depth)
        for index in range(args.graphs)
    ]
    maxima = {key: max(result[key] for result in per_graph) for key in per_graph[0]}
    passed = all(value <= args.tolerance for value in maxima.values())
    payload = {
        "experiment": "E0_algebraic_symmetry_certification",
        "passed": passed,
        "tolerance": args.tolerance,
        "configuration": vars(args) | {"output": str(args.output)},
        "max_errors": maxima,
        "per_graph": per_graph,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "max_errors": maxima}, indent=2))
    if not passed:
        raise RuntimeError(
            "algebraic symmetry certification exceeded the configured tolerance; "
            f"details were preserved in {args.output}"
        )


if __name__ == "__main__":
    main()
