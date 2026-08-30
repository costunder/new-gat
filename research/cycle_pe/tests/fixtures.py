"""Small in-memory/file fixtures for unit tests, never production datasets."""

from pathlib import Path

import networkx as nx
import numpy as np

from research.cycle_pe.paper_data import load_or_generate_cycle_count_ood

CORE_TEST_SPLIT_SIZES = {
    "train": 10,
    "validation": 4,
    "id_test": 4,
    "size_ood": 4,
    "family_ood": 4,
}


def small_cyclecount_loader(data_root: Path, *, seed: int):
    return load_or_generate_cycle_count_ood(data_root, seed=seed, split_sizes=CORE_TEST_SPLIT_SIZES)


def write_brec_fixture(path: Path, *, num_relabel: int = 2) -> Path:
    """Create two RPC-layout pairs exclusively for unit tests."""

    if num_relabel < 2:
        raise ValueError("num_relabel must be at least two")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = (
        (nx.cycle_graph(5), nx.complete_bipartite_graph(2, 3)),
        (nx.cycle_graph(6), nx.path_graph(6)),
    )
    train_records: list[bytes] = []
    reliability_records: list[bytes] = []
    for left, right in pairs:
        for relabel in range(num_relabel):
            for graph, offset in ((left, 0), (right, 100)):
                permutation = np.random.default_rng(offset + relabel).permutation(
                    graph.number_of_nodes()
                )
                mapping = {node: int(permutation[node]) for node in graph.nodes()}
                train_records.append(
                    nx.to_graph6_bytes(nx.relabel_nodes(graph, mapping), header=False).strip()
                )
        for relabel in range(num_relabel):
            for offset in (200, 300):
                permutation = np.random.default_rng(offset + relabel).permutation(
                    left.number_of_nodes()
                )
                mapping = {node: int(permutation[node]) for node in left.nodes()}
                reliability_records.append(
                    nx.to_graph6_bytes(nx.relabel_nodes(left, mapping), header=False).strip()
                )
    np.save(path, np.asarray(train_records + reliability_records, dtype=object), allow_pickle=True)
    return path
