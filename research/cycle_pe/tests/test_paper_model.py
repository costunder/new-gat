from __future__ import annotations

import numpy as np
import pytest
import torch

import research.cycle_pe.paper_model as paper_model_module
from chartgat.algebra import incidence_matrix
from chartgat.graphs import spanning_tree_indices
from research.cycle_pe.features import cycle_projector, static_fundamental_basis
from research.cycle_pe.paper_data import PaperGraph, canonical_edges
from research.cycle_pe.paper_model import (
    PE_VARIANTS,
    BatchOutput,
    PaperCycleModel,
    RawCycleRankOverflow,
    StaticPEEncoder,
    pack_prepared_graphs,
    prepare_splits,
)


def _graph(name: str, num_nodes: int, edges: tuple[tuple[int, int], ...]) -> PaperGraph:
    return PaperGraph(
        graph_id=name,
        split="test",
        family="fixture",
        num_nodes=num_nodes,
        edges=canonical_edges(edges),
        edge_targets=np.zeros((len(edges), 1), dtype=np.float64),
        node_targets=np.zeros((num_nodes, 1), dtype=np.float64),
        graph_targets=np.zeros(1, dtype=np.float64),
    )


def test_paper_preparation_removes_fixed_max_cycles_and_batches_variable_beta() -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))
    dense_edges = tuple((u, v) for u in range(8) for v in range(u + 1, 8))
    dense = _graph("dense", 8, dense_edges)
    assert dense.beta == 21

    prepared, raw_width = prepare_splits({"train": [triangle, dense]}, fit_split="train")
    assert raw_width == 21
    assert raw_width > 12
    assert prepared["train"][0].raw_basis.shape == (3, 1)
    assert prepared["train"][1].raw_basis.shape == (28, 21)

    for variant in PE_VARIANTS:
        torch.manual_seed(3)
        model = PaperCycleModel(
            variant=variant,
            raw_width=raw_width,
            node_input_dim=2,
            edge_input_dim=4,
            edge_output_dim=1,
            node_output_dim=1,
            graph_output_dim=1,
            hidden_dim=16,
            pe_dim=8,
            layers=1,
        )
        outputs = model(prepared["train"])
        separate = [model.forward_graph(graph) for graph in prepared["train"]]
        assert outputs[0].edge is not None and outputs[0].edge.shape == (3, 1)
        assert outputs[1].edge is not None and outputs[1].edge.shape == (28, 1)
        assert outputs[0].node is not None and outputs[0].node.shape == (3, 1)
        assert outputs[1].graph is not None and outputs[1].graph.shape == (1,)
        assert outputs[0].embedding.shape == (16,)
        for batched, single in zip(outputs, separate, strict=True):
            assert batched.edge is not None and single.edge is not None
            assert batched.node is not None and single.node is not None
            assert batched.graph is not None and single.graph is not None
            torch.testing.assert_close(batched.edge, single.edge)
            torch.testing.assert_close(batched.node, single.node)
            torch.testing.assert_close(batched.graph, single.graph)
            torch.testing.assert_close(batched.embedding, single.embedding)


def test_non_projector_variants_do_not_materialize_dense_projector(monkeypatch) -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))

    def forbidden_projector(_basis):
        raise AssertionError("dense projector should be lazy")

    monkeypatch.setattr(paper_model_module, "cycle_projector", forbidden_projector)
    prepared, _ = prepare_splits(
        {"train": [triangle]},
        fit_split="train",
        required_variants=("no_pe", "set"),
    )
    graph = prepared["train"][0]
    assert graph.cycle_set is not None
    assert graph.projector is None


def test_raw_width_is_fit_on_train_only_and_ood_is_never_truncated() -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))
    dense_edges = tuple((u, v) for u in range(8) for v in range(u + 1, 8))
    dense = _graph("dense", 8, dense_edges)
    prepared, raw_width = prepare_splits(
        {"train": [triangle], "size_ood": [dense]},
        fit_split="train",
    )
    assert raw_width == 1
    assert prepared["size_ood"][0].raw_basis.shape[1] == 21

    raw_encoder = StaticPEEncoder("raw", raw_width=raw_width, pe_dim=4)
    graph = prepared["size_ood"][0]
    with pytest.raises(RawCycleRankOverflow, match="train-fitted raw width 1"):
        raw_encoder(graph.raw_basis, graph.cycle_set, graph.projector)

    projector_encoder = StaticPEEncoder("projector", raw_width=raw_width, pe_dim=4)
    output = projector_encoder(graph.raw_basis, graph.cycle_set, graph.projector)
    assert output.shape == (28, 4)


def test_projector_encoder_is_basis_change_and_orientation_invariant() -> None:
    edges = ((0, 1), (1, 2), (2, 3), (0, 3), (0, 2))
    incidence = incidence_matrix(4, edges)
    tree = spanning_tree_indices(4, edges, mode="bfs")
    basis = static_fundamental_basis(incidence, tree)
    changed = basis @ np.asarray([[1.0, 2.0], [-1.0, 1.0]])
    projector = cycle_projector(basis)
    changed_projector = cycle_projector(changed)

    torch.manual_seed(7)
    encoder = StaticPEEncoder("projector", raw_width=2, pe_dim=9)
    raw = torch.zeros((5, 2))
    cycle_set = torch.zeros((5, 6))
    original = encoder(raw, cycle_set, torch.as_tensor(projector, dtype=torch.float32))
    transformed = encoder(raw, cycle_set, torch.as_tensor(changed_projector, dtype=torch.float32))
    torch.testing.assert_close(original, transformed, atol=2e-6, rtol=2e-6)

    signs = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])
    oriented = signs[:, None] * projector * signs[None, :]
    flipped = encoder(raw, cycle_set, torch.as_tensor(oriented, dtype=torch.float32))
    torch.testing.assert_close(original, flipped, atol=2e-6, rtol=2e-6)

    permutation = np.asarray([3, 0, 4, 1, 2])
    permuted_projector = projector[np.ix_(permutation, permutation)]
    permuted = encoder(
        raw[permutation],
        cycle_set[permutation],
        torch.as_tensor(permuted_projector, dtype=torch.float32),
    )
    torch.testing.assert_close(original[permutation], permuted, atol=2e-6, rtol=2e-6)


def test_projector_model_handles_connected_singleton_without_edges() -> None:
    singleton = PaperGraph(
        graph_id="singleton",
        split="test",
        family="fixture",
        num_nodes=1,
        edges=(),
        graph_targets=np.asarray([0.0]),
    )
    prepared, raw_width = prepare_splits({"test": [singleton]})
    model = PaperCycleModel(
        variant="projector",
        raw_width=raw_width,
        node_input_dim=2,
        edge_input_dim=4,
        edge_output_dim=0,
        node_output_dim=0,
        graph_output_dim=1,
        hidden_dim=12,
        pe_dim=6,
        layers=1,
    )
    output = model(prepared["test"])[0]
    assert output.graph is not None and torch.isfinite(output.graph).all()


@pytest.mark.parametrize("variant", PE_VARIANTS)
@pytest.mark.parametrize("target_level", ("edge", "node", "graph"))
def test_packed_physical_batch_connects_every_active_parameter(
    variant: str, target_level: str
) -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))
    square = _graph("square", 4, ((0, 1), (1, 2), (2, 3), (0, 3), (0, 2)))
    prepared, raw_width = prepare_splits(
        {"train": [triangle, square]},
        fit_split="train",
        required_variants=(variant,),
    )
    batch = pack_prepared_graphs(prepared["train"])
    torch.manual_seed(41)
    model = PaperCycleModel(
        variant=variant,
        raw_width=raw_width,
        node_input_dim=2,
        edge_input_dim=4,
        edge_output_dim=1 if target_level == "edge" else 0,
        node_output_dim=1 if target_level == "node" else 0,
        graph_output_dim=1 if target_level == "graph" else 0,
        hidden_dim=12,
        pe_dim=6,
        layers=2,
        embedding_dim=0,
    )
    output = model(batch)
    assert isinstance(output, BatchOutput)
    selected = getattr(output, target_level)
    assert selected is not None
    selected.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if variant == "no_pe":
        assert sum(parameter.numel() for parameter in model.pe_encoder.parameters()) == 0


def test_active_variant_pack_does_not_transfer_unused_dense_representations() -> None:
    triangle = _graph("triangle", 3, ((0, 1), (1, 2), (0, 2)))
    prepared, _ = prepare_splits({"train": [triangle]}, fit_split="train")
    graph = prepared["train"][0]
    assert graph.cycle_set is not None and graph.projector is not None

    no_pe = pack_prepared_graphs([graph], variant="no_pe", target_levels=("edge",))
    assert no_pe.raw_basis.shape == (3, 0)
    assert no_pe.cycle_set is None and no_pe.projector_values is None
    assert no_pe.edge_targets is not None
    assert no_pe.node_targets is None and no_pe.graph_targets is None

    projector = pack_prepared_graphs([graph], variant="projector", target_levels=())
    assert projector.raw_basis.shape == (3, 0)
    assert projector.cycle_set is None
    assert projector.projector_values is not None
    assert projector.projector_values.numel() == 9
