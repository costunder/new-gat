import time

import torch

from research.conductance_gat.v5.sampling import (
    csr_neighbors,
    csr_values,
    induced_physical_edge_ids,
    physical_edge_id_csr,
)


def naive_neighbors(arcs, rowptr, nodes):
    return torch.cat([arcs[1, rowptr[node] : rowptr[node + 1]] for node in nodes])


def test_vectorized_csr_gather_preserves_row_order_and_candidate_multiset():
    arcs = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 3]], dtype=torch.long)
    rowptr = torch.tensor([0, 2, 3, 5, 5], dtype=torch.long)
    # Include unsorted, duplicate, and empty rows to pin down the exact contract.
    nodes = torch.tensor([2, 0, 2, 3, 1], dtype=torch.long)
    expected = naive_neighbors(arcs, rowptr, nodes)
    actual = csr_neighbors(arcs, rowptr, nodes)
    assert torch.equal(actual, expected)
    assert torch.equal(actual.sort().values, expected.sort().values)


def test_vectorized_csr_gather_large_frontier_smoke():
    nodes, degree = 100_000, 8
    sources = torch.arange(nodes).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1).repeat(nodes)
    destinations = (sources + offsets) % nodes
    arcs = torch.stack((sources, destinations))
    rowptr = torch.arange(0, (nodes + 1) * degree, degree)
    frontier = torch.arange(0, nodes, 2)
    started = time.perf_counter()
    result = csr_neighbors(arcs, rowptr, frontier)
    elapsed = time.perf_counter() - started
    assert result.shape == (frontier.numel() * degree,)
    # A generous smoke guard catches accidental reintroduction of per-node Python
    # work without pretending to be a stable microbenchmark across CI hardware.
    assert elapsed < 10.0


def test_incident_edge_csr_matches_full_scan_induced_edges():
    incidence = torch.tensor([[0, 0, 1, 2, 2, 4, 5], [1, 2, 2, 3, 4, 5, 6]], dtype=torch.long)
    edge_ids, rowptr = physical_edge_id_csr(incidence, 7)
    nodes = torch.tensor([4, 2, 0, 1], dtype=torch.long)
    actual, candidate_count = induced_physical_edge_ids(incidence, edge_ids, rowptr, nodes, 7)
    membership = torch.zeros(7, dtype=torch.bool)
    membership[nodes] = True
    expected = (membership[incidence[0]] & membership[incidence[1]]).nonzero().flatten()
    assert torch.equal(actual, expected)
    assert candidate_count == csr_values(edge_ids, rowptr, nodes.unique()).numel()


def test_large_induced_lookup_scales_with_sample_incident_edges():
    num_nodes, degree = 100_000, 8
    tail = torch.arange(num_nodes).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1).repeat(num_nodes)
    incidence = torch.stack((tail, (tail + offsets) % num_nodes))
    edge_ids, rowptr = physical_edge_id_csr(incidence, num_nodes)
    selected = torch.arange(128, dtype=torch.long)
    started = time.perf_counter()
    result, candidate_count = induced_physical_edge_ids(
        incidence, edge_ids, rowptr, selected, num_nodes
    )
    elapsed = time.perf_counter() - started
    # The lookup touches only the selected CSR rows (two endpoint directions),
    # rather than scanning all 800k physical edges for every training batch.
    assert candidate_count <= selected.numel() * 2 * degree
    assert result.numel() > 0
    assert elapsed < 10.0
