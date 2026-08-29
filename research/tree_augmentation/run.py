"""Run the standalone spanning-tree augmentation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from chartgat.graphs import make_connected_graph
from research.tree_augmentation.augmentation import (
    ensure_full_cycle_budget,
    find_unseen_chart,
    lossless_transition_error,
    run_static_cycle_pe_probe,
    sample_tree_charts,
    transition_cocycle_error,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute chart certification and the fixed/multi/unseen static probe."""

    graph_config = config["graph"]
    augmentation_config = config["augmentation"]
    probe_config = config["probe"]
    seed = int(config.get("seed", 0))

    if augmentation_config.get("lossy_extension_enabled", False):
        raise NotImplementedError("the k < beta lossy extension is intentionally disabled")

    num_nodes = int(graph_config["num_nodes"])
    edges = make_connected_graph(
        num_nodes,
        int(graph_config["extra_edges"]),
        seed=int(graph_config["seed"]),
    )
    charts = sample_tree_charts(
        num_nodes,
        edges,
        include_bfs=bool(augmentation_config.get("include_bfs", True)),
        include_dfs=bool(augmentation_config.get("include_dfs", True)),
        random_count=int(augmentation_config.get("random_count", 0)),
        random_seed_start=int(augmentation_config.get("random_seed_start", 0)),
    )
    beta = charts[0].beta
    ensure_full_cycle_budget(beta, augmentation_config.get("k"))
    unseen = find_unseen_chart(
        num_nodes,
        edges,
        charts,
        seed_start=int(augmentation_config.get("unseen_seed_start", 10_000)),
    )

    rng = np.random.default_rng(seed)
    coordinates = rng.normal(size=beta)
    certification_charts = [*charts, unseen]
    certification = {
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "cycle_rank_beta": beta,
        "full_beta_enabled": True,
        "lossy_k_lt_beta_enabled": False,
        "unique_training_trees": len(charts),
        "lossless_transition_error": lossless_transition_error(certification_charts, coordinates),
        "transition_cocycle_error": transition_cocycle_error(certification_charts),
    }
    probe = run_static_cycle_pe_probe(
        charts,
        unseen,
        hidden_dim=int(probe_config.get("hidden_dim", 48)),
        epochs=int(probe_config.get("epochs", 800)),
        learning_rate=float(probe_config.get("learning_rate", 0.01)),
        weight_decay=float(probe_config.get("weight_decay", 1e-5)),
        seed=seed,
    )
    return {
        "track": "static_cycle_pe_tree_augmentation",
        "certification": certification,
        "probe": probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).with_name("config.yaml")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    result = run(config)
    if args.output is not None:
        output = args.output.expanduser()
    else:
        output = Path(config.get("output", "results/summary.json")).expanduser()
        if not output.is_absolute():
            output = config_path.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
