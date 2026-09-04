"""E1: cycle-nullspace identifiability and observation-conditioning sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix
from chartgat.graphs import make_connected_graph, spanning_tree_indices
from research.combined_later.completion import (
    analytic_cycle_completion,
    cycle_observation_spectrum,
    weighted_particular_flow,
)
from research.combined_later.synthetic import structured_cycle_flow


def _greedy_observation_order(cycle_basis: np.ndarray) -> np.ndarray:
    remaining = list(range(cycle_basis.shape[0]))
    selected: list[int] = []
    while remaining:
        best_edge = remaining[0]
        best_score = (-1, -1.0, -1.0)
        for edge in remaining:
            spectrum = cycle_observation_spectrum(cycle_basis, selected + [edge])
            finite_sigma = spectrum.sigma_min if np.isfinite(spectrum.sigma_min) else 0.0
            score = (spectrum.rank, spectrum.sigma_min_nonzero, finite_sigma)
            if score > best_score:
                best_score = score
                best_edge = edge
        selected.append(best_edge)
        remaining.remove(best_edge)
    return np.asarray(selected, dtype=np.int64)


def _relative_error(prediction: np.ndarray, target: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(prediction - target) / max(scale, 1.0e-12))


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    edges = make_connected_graph(args.nodes, args.extra_edges, seed=args.seed)
    incidence = incidence_matrix(args.nodes, edges)
    tree = spanning_tree_indices(args.nodes, edges, mode="bfs")
    cycle_basis, chords = fundamental_cycle_basis(incidence, tree, return_chords=True)
    beta = cycle_basis.shape[1]
    rng = np.random.default_rng(args.seed + 1)

    p = rng.normal(size=args.nodes)
    p -= p.mean()
    seed_flow = incidence @ p
    divergence = incidence.T @ seed_flow
    particular = weighted_particular_flow(incidence, divergence)
    cycle = structured_cycle_flow(
        incidence,
        node_features=rng.normal(size=(args.nodes, 3)),
        edge_features=rng.normal(size=(incidence.shape[0], 2)),
        scale=args.cycle_scale,
    )
    target = particular + cycle
    pair_divergence_error = float(np.max(np.abs(incidence.T @ target - incidence.T @ particular)))
    cycle_scale = float(np.linalg.norm(cycle))

    all_edges = np.arange(incidence.shape[0], dtype=np.int64)
    tree_set = set(tree.tolist())
    tree_order = np.asarray(list(tree) + list(chords), dtype=np.int64)
    chord_order = np.asarray(list(chords) + [e for e in all_edges if e in tree_set])
    greedy_order = _greedy_observation_order(cycle_basis)
    fixed_orders = {
        "tree_first": tree_order,
        "chord_first": chord_order,
        "rank_greedy": greedy_order,
    }

    records: list[dict[str, object]] = []
    for observed_count in range(incidence.shape[0] + 1):
        for repeat in range(args.repeats):
            orders = fixed_orders | {"random": rng.permutation(all_edges)}
            for strategy, order in orders.items():
                observed = np.asarray(order[:observed_count], dtype=np.int64)
                spectrum = cycle_observation_spectrum(cycle_basis, observed)
                observed_values = target[observed]
                noiseless = analytic_cycle_completion(
                    particular, cycle_basis, observed, observed_values
                )
                noise = args.noise_std * rng.normal(size=observed_count)
                noisy_values = observed_values + noise
                noisy_ls = analytic_cycle_completion(
                    particular, cycle_basis, observed, noisy_values
                )
                noisy_ridge = analytic_cycle_completion(
                    particular,
                    cycle_basis,
                    observed,
                    noisy_values,
                    ridge=args.ridge,
                )
                records.append(
                    {
                        "strategy": strategy,
                        "observed_count": observed_count,
                        "observed_fraction": observed_count / incidence.shape[0],
                        "repeat": repeat,
                        "rank": spectrum.rank,
                        "beta": beta,
                        "sigma_min": spectrum.sigma_min,
                        "sigma_min_nonzero": spectrum.sigma_min_nonzero,
                        "condition_number": spectrum.condition_number,
                        "noise_amplification": spectrum.noise_amplification,
                        "noiseless_relative_error": _relative_error(
                            noiseless.flow, target, cycle_scale
                        ),
                        "noisy_ls_relative_error": _relative_error(
                            noisy_ls.flow, target, cycle_scale
                        ),
                        "noisy_ridge_relative_error": _relative_error(
                            noisy_ridge.flow, target, cycle_scale
                        ),
                    }
                )

    frame = pd.DataFrame.from_records(records)
    full_rank = frame[frame["rank"] == beta]
    first_full_rank: dict[str, int | None] = {}
    for strategy, group in frame.groupby("strategy"):
        strategy_full_rank = group[group["rank"] == beta]
        first_full_rank[str(strategy)] = (
            int(strategy_full_rank["observed_count"].min())
            if not strategy_full_rank.empty
            else None
        )
    summary: dict[str, object] = {
        "experiment": "E1_nullspace_identifiability",
        "nodes": args.nodes,
        "edges": int(incidence.shape[0]),
        "cycle_rank": beta,
        "pair_divergence_error": pair_divergence_error,
        "first_full_rank_observation_count": first_full_rank,
        "max_full_rank_noiseless_relative_error": (
            float(full_rank["noiseless_relative_error"].max()) if not full_rank.empty else None
        ),
        "noise_std": args.noise_std,
        "ridge": args.ridge,
    }
    return frame, summary


def _write_plot(frame: pd.DataFrame, output: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = frame.groupby(["strategy", "observed_count"], as_index=False).agg(
        rank=("rank", "mean"),
        noiseless=("noiseless_relative_error", "mean"),
        noisy_ls=("noisy_ls_relative_error", "mean"),
        noisy_ridge=("noisy_ridge_relative_error", "mean"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy, group in grouped.groupby("strategy"):
        axes[0].plot(group["observed_count"], group["noiseless"], label=strategy)
        axes[1].plot(group["observed_count"], group["rank"], label=strategy)
    axes[0].set_yscale("symlog", linthresh=1.0e-12)
    axes[0].set_xlabel("observed edge count")
    axes[0].set_ylabel("noiseless relative reconstruction error")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("observed edge count")
    axes[1].set_ylabel("rank(S U_c)")
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/combined_later/identifiability"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--extra-edges", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--cycle-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output path already exists: {args.output_dir}; choose a new path"
        )
    frame, summary = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "sweep.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_plot(frame, args.output_dir / "identifiability.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
