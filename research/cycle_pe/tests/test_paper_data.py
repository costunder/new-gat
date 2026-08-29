from __future__ import annotations

import numpy as np

from research.cycle_pe.paper_data import (
    CORE_SPLITS,
    canonical_edges,
    cycle_count_split_sizes,
    enumerate_short_cycles,
    exact_cycle_targets,
    load_or_generate_cycle_count_ood,
)


def test_full_cycle_count_protocol_contains_exactly_twenty_thousand_graphs() -> None:
    sizes = cycle_count_split_sizes(tiny=False)
    assert sizes == {
        "train": 10_000,
        "validation": 2_000,
        "id_test": 2_000,
        "size_ood": 3_000,
        "family_ood": 3_000,
    }
    assert sum(sizes.values()) == 20_000


def test_exact_short_cycle_targets_cover_edge_node_and_graph_levels() -> None:
    # A triangle and square share vertex 2; edge (5, 6) is a bridge.
    edges = canonical_edges(((0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (2, 5), (5, 6)))
    cycles = enumerate_short_cycles(7, edges)
    assert [(len(cycle), cycle) for cycle in cycles] == [
        (3, (0, 1, 2)),
        (4, (2, 3, 4, 5)),
    ]

    edge, node, graph = exact_cycle_targets(7, edges)
    np.testing.assert_array_equal(graph, np.asarray([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(node[2], np.asarray([1.0, 1.0, 0.0, 0.0]))
    bridge_index = edges.index((5, 6))
    np.testing.assert_array_equal(edge[bridge_index], np.zeros(6))
    triangle_index = edges.index((0, 1))
    np.testing.assert_array_equal(edge[triangle_index], np.asarray([1.0, 0.0, 0.0, 0.0, 3.0, 1.0]))

    seven_cycle = canonical_edges((node, (node + 1) % 7) for node in range(7))
    long_edge, _, long_graph = exact_cycle_targets(7, seven_cycle)
    np.testing.assert_array_equal(long_graph, np.zeros(4))
    np.testing.assert_array_equal(long_edge[:, 4], np.full(7, 7.0))
    np.testing.assert_array_equal(long_edge[:, 5], np.zeros(7))


def test_cycle_count_ood_cache_and_splits_are_deterministic(tmp_path) -> None:
    first = load_or_generate_cycle_count_ood(tmp_path, seed=19, tiny=True)
    second = load_or_generate_cycle_count_ood(tmp_path, seed=19, tiny=True)
    independent = load_or_generate_cycle_count_ood(
        tmp_path / "independent-root", seed=19, tiny=True
    )

    assert first.cache_path == second.cache_path
    assert first.cache_sha256 == second.cache_sha256
    assert first.cache_sha256 == independent.cache_sha256
    assert tuple(first.splits) == CORE_SPLITS
    assert {name: len(graphs) for name, graphs in first.splits.items()} == {
        "train": 10,
        "validation": 4,
        "id_test": 4,
        "size_ood": 4,
        "family_ood": 4,
    }
    assert [graph.graph_id for graph in first.splits["train"]] == [
        graph.graph_id for graph in second.splits["train"]
    ]
    assert min(graph.num_nodes for graph in first.splits["size_ood"]) > max(
        graph.num_nodes for graph in first.splits["train"]
    )
    training_families = {graph.family for graph in first.splits["train"]}
    family_ood = {graph.family for graph in first.splits["family_ood"]}
    assert training_families.isdisjoint(family_ood)
    for split in CORE_SPLITS:
        for left, right in zip(first.splits[split], second.splits[split], strict=True):
            np.testing.assert_array_equal(left.edge_targets, right.edge_targets)
            np.testing.assert_array_equal(left.node_targets, right.node_targets)
            np.testing.assert_array_equal(left.graph_targets, right.graph_targets)
