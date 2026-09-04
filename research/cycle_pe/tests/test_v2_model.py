"""Synthetic CPU/CUDA unit fixtures, never benchmark training or datasets."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from research.cycle_pe.v2.data import Graph, collate, prepare_graph
from research.cycle_pe.v2.model import (
    MODEL_NAME,
    MODEL_NAMES,
    CycleBasisPEModel,
    LeftNullBasisEncoder,
    architecture_protocol,
)


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _graph(
    n: int = 4,
    *,
    complete: bool = False,
    forest: bool = False,
    basis_backend: str = "dfs_fundamental",
) -> Graph:
    if complete:
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    elif forest:
        edges = [(i, i + 1) for i in range(n - 1)]
    else:
        edges = sorted({tuple(sorted((i, (i + 1) % n))) for i in range(n)})
    return prepare_graph(
        SimpleNamespace(
            num_nodes=n,
            x=torch.arange(n).reshape(-1, 1),
            edge_index=torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long)
            .reshape(-1, 2)
            .T.contiguous(),
            edge_attr=torch.ones((2 * len(edges), 1), dtype=torch.long),
            y=torch.tensor([0.7]),
        ),
        basis_backend=basis_backend,
    )


def _disconnected_graph(*, basis_backend: str = "dfs_fundamental") -> Graph:
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (5, 6)]
    return prepare_graph(
        SimpleNamespace(
            num_nodes=8,
            x=torch.arange(8).reshape(-1, 1),
            edge_index=torch.tensor(edges + [(v, u) for u, v in edges]).T.contiguous(),
            edge_attr=torch.ones(2 * len(edges), 1, dtype=torch.long),
            y=torch.tensor([0.4]),
        ),
        basis_backend=basis_backend,
    )


def _encode(encoder, bond, batch):
    return encoder(
        bond, batch.cycle_membership, batch.cycle_lengths,
        batch.edge_cycle_counts, batch.edge_cycle_features,
        batch.cycle_position_values,
    )


def _sparse(indices, values, shape):
    return torch.sparse_coo_tensor(indices, values, shape, check_invariants=True).coalesce()


def _assert_parameter_gradients_match(first, second):
    actual, expected = dict(first.named_parameters()), dict(second.named_parameters())
    assert actual.keys() == expected.keys()
    for name, parameter in actual.items():
        assert parameter.grad is not None, name
        assert expected[name].grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad, expected[name].grad, atol=4e-6, rtol=4e-5, msg=name
        )


def test_sparse_encoder_matches_explicit_edge_cycle_edge_reductions():
    torch.manual_seed(11)
    batch = collate([_graph(5, complete=True), _disconnected_graph(), _graph(3, forest=True)])
    encoder = LeftNullBasisEncoder(5, 9)
    bond = torch.randn(len(batch.edge_attr), 5)
    actual = _encode(encoder, bond, batch)
    edge_ids, cycle_ids = batch.cycle_membership.indices()
    values = encoder.column_phi(bond)
    cycle_sum = values.new_zeros((len(batch.cycle_lengths), encoder.pe_dim))
    cycle_sum.index_add_(0, cycle_ids, values[edge_ids])
    log_lengths = batch.cycle_lengths.log1p()[:, None]
    cycle_hidden = encoder.cycle_mlp(
        torch.cat((cycle_sum / batch.cycle_lengths[:, None], log_lengths), dim=1)
    )
    edge_sum = values.new_zeros(values.shape)
    edge_sum.index_add_(0, edge_ids, cycle_hidden[cycle_ids])
    structure = torch.cat((log_lengths, batch.cycle_lengths.reciprocal()[:, None]), dim=1)
    structural_sum = values.new_zeros((len(values), 2))
    structural_sum.index_add_(0, edge_ids, structure[cycle_ids])
    counts = batch.edge_cycle_counts[:, None]
    active = (counts > 0).float()
    features = torch.cat(
        (
            values * active,
            edge_sum / counts.clamp_min(1),
            counts.log1p(),
            structural_sum / counts.clamp_min(1),
        ),
        dim=1,
    )
    expected = encoder.output(encoder.edge_psi(features)) * active
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_no_factorization_dense_cycle_matrix_or_graphwise_sparse_loop(monkeypatch, encoding):
    batch = collate([_graph(4), _graph(5, complete=True), _disconnected_graph()])
    model = CycleBasisPEModel(
        dataset="zinc12k", encoding=encoding, hidden=16, pe_dim=8, layers=3
    )
    calls = []
    original_mm = torch.sparse.mm

    def forbidden(*args, **kwargs):
        raise AssertionError("factorization or sparse densification reached the model")

    def observed(matrix, values, *args, **kwargs):
        assert matrix.layout == torch.sparse_coo
        assert matrix.dtype == values.dtype == torch.float32
        calls.append((tuple(matrix.shape), tuple(values.shape)))
        return original_mm(matrix, values, *args, **kwargs)

    for name in ("qr", "svd", "eigh", "eig", "inv", "pinv", "cholesky"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    monkeypatch.setattr(torch.sparse, "mm", observed)
    prediction = model(batch)
    (prediction - batch.y).abs().mean().backward()
    edges, cycles = batch.cycle_membership.shape
    expected_calls = [
        ((cycles, edges), (edges, 8)),
        ((edges, cycles), (cycles, 8)),
    ]
    if encoding == "pe":
        expected_calls += [
            ((cycles, edges), (edges, 8)),
            ((cycles, edges), (edges, 8)),
            ((edges, cycles), (cycles, 8)),
            ((edges, cycles), (cycles, 8)),
        ]
    assert calls == expected_calls
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_every_selected_cycle_receives_task_gradient_without_rank_dependent_parameters():
    torch.manual_seed(23)
    encoder = LeftNullBasisEncoder(7, 11)
    parameter_count = sum(p.numel() for p in encoder.parameters())
    for graph in (_graph(3), _graph(7, complete=True), _disconnected_graph()):
        batch = collate([graph])
        cycle_outputs = []

        def capture(_module, _args, output, captured=cycle_outputs):
            output.retain_grad()
            captured.append(output)

        hook = encoder.cycle_mlp.register_forward_hook(capture)
        output = _encode(encoder, torch.randn(len(graph.edge_attr), 7), batch)
        output.square().sum().backward()
        hook.remove()
        assert len(cycle_outputs) == 1
        hidden = cycle_outputs[0]
        assert hidden.shape[0] == graph.cycle_basis.shape[1]
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert (hidden.grad.abs().sum(dim=1) > 0).all()
        assert parameter_count == sum(p.numel() for p in encoder.parameters())


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_cycle_sign_and_column_order_do_not_change_selected_membership_pe(encoding):
    torch.manual_seed(19)
    graph = _graph(6, complete=True)
    rank = graph.cycle_basis.shape[1]
    order = torch.randperm(rank)
    inverse_order = torch.argsort(order)
    indices = graph.cycle_basis.indices().clone()
    signs = torch.where(torch.arange(rank) % 2 == 0, -1.0, 1.0)
    values = graph.cycle_basis.values() * signs[indices[1]]
    indices[1] = inverse_order[indices[1]]
    transported = replace(
        graph,
        cycle_basis=_sparse(indices, values, graph.cycle_basis.shape),
        cycle_lengths=graph.cycle_lengths[order],
        cycle_position_indices=_sparse(
            indices, graph.cycle_position_indices, graph.cycle_basis.shape
        ).values(),
        cycle_position_values=torch.stack([
            _sparse(indices, row, graph.cycle_basis.shape).values()
            for row in graph.cycle_position_values
        ]),
    )
    encoder = LeftNullBasisEncoder(5, 8, encoding=encoding)
    bond = torch.randn(len(graph.edge_attr), 5)
    torch.testing.assert_close(
        _encode(encoder, bond, collate([graph])),
        _encode(encoder, bond, collate([transported])),
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_edge_order_equivariance_with_transported_sparse_membership(encoding):
    torch.manual_seed(29)
    graph = _graph(5, complete=True)
    batch = collate([graph])
    encoder = LeftNullBasisEncoder(5, 7, encoding=encoding)
    bond = torch.randn(len(graph.edge_attr), 5)
    expected = _encode(encoder, bond, batch)
    order = torch.randperm(len(bond))
    inverse_order = torch.argsort(order)
    indices = batch.cycle_membership.indices().clone()
    indices[0] = inverse_order[indices[0]]
    membership = _sparse(
        indices, batch.cycle_membership.values(), batch.cycle_membership.shape
    )
    actual = encoder(
        bond[order], membership, batch.cycle_lengths,
        batch.edge_cycle_counts[order], batch.edge_cycle_features[order],
        torch.stack([
            _sparse(indices, row, membership.shape).values()
            for row in batch.cycle_position_values
        ]),
    )
    torch.testing.assert_close(actual, expected[order], atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_node_permutation_preserves_prediction_when_selected_basis_is_transported(encoding):
    torch.manual_seed(31)
    graph = _disconnected_graph()
    order = torch.randperm(len(graph.x))
    inverse_order = torch.argsort(order)
    transported = replace(
        graph, x=graph.x[order], edge_index=inverse_order[graph.edge_index]
    )
    model = CycleBasisPEModel(dataset="zinc12k", encoding=encoding, hidden=16, pe_dim=8, layers=3
    ).eval()
    with torch.no_grad():
        expected = model(collate([graph]))
        actual = model(collate([transported]))
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-5)


@pytest.mark.parametrize("encoding", ["se", "pe"])
@pytest.mark.parametrize("edges", [0, 4])
def test_empty_and_forest_encoder_pe_and_all_parameter_gradients_are_exactly_zero(edges, encoding):
    encoder = LeftNullBasisEncoder(5, 7, encoding=encoding)
    for parameter in encoder.parameters():
        torch.nn.init.constant_(parameter, 0.3)
    membership = _sparse(
        torch.empty((2, 0), dtype=torch.long), torch.empty(0), (edges, 0)
    )
    bond = torch.randn(edges, 5, requires_grad=True)
    actual = encoder(
        bond, membership, torch.empty(0), torch.zeros(edges),
        torch.zeros(edges, 2), torch.empty(2, 0),
    )
    assert actual.shape == (edges, 7)
    assert torch.equal(actual, torch.zeros_like(actual))
    actual.square().sum().backward()
    assert bond.grad is not None and torch.count_nonzero(bond.grad) == 0
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) == 0, name


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_bridges_have_zero_pe_with_nonzero_mlp_biases(encoding):
    batch = collate([_disconnected_graph(), _graph(3, forest=True)])
    encoder = LeftNullBasisEncoder(4, 8, encoding=encoding)
    for name, parameter in encoder.named_parameters():
        if name.endswith("bias"):
            torch.nn.init.constant_(parameter, 0.3)
    actual = _encode(encoder, torch.randn(len(batch.edge_attr), 4), batch)
    inactive = batch.edge_cycle_counts == 0
    assert inactive.any() and (~inactive).any()
    assert torch.equal(actual[inactive], torch.zeros_like(actual[inactive]))


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_sparse_physical_batch_matches_graphwise_outputs_and_every_parameter_gradient(encoding):
    torch.manual_seed(37)
    graphs = [_graph(4), _graph(5, complete=True), _graph(4, forest=True), _disconnected_graph()]
    batched = CycleBasisPEModel(
        dataset="zinc12k", encoding=encoding, hidden=16, pe_dim=8, layers=3
    )
    reference = copy.deepcopy(batched)
    batch = collate(graphs)
    actual = batched(batch)
    expected = torch.cat([reference(collate([graph])) for graph in graphs])
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-5)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    _assert_parameter_gradients_match(batched, reference)


@pytest.mark.parametrize("encoding", ["se", "pe"])
def test_full_default_architecture_pe_affects_loss_and_all_pe_parameters_update(encoding):
    torch.manual_seed(713)
    batch = collate([_graph(5, complete=True), _graph(4, forest=True), _disconnected_graph()])
    model = CycleBasisPEModel(dataset="zinc12k", encoding=encoding)
    assert len(model.layers) == 10
    assert model.pe_encoder.pe_dim == 64
    assert model.graph_head.in_features == 128
    assert sum(p.numel() for p in model.parameters()) == 7_262_785
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert {id(p) for p in model.parameters()} == {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    predicted = model(batch)
    loss = (predicted - batch.y).abs().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = {name: p.detach().clone() for name, p in model.pe_encoder.named_parameters()}
    assert all(torch.count_nonzero(p.grad) > 0 for p in model.pe_encoder.parameters())
    optimizer.step()
    assert all(
        not torch.equal(before[name], parameter)
        for name, parameter in model.pe_encoder.named_parameters()
    )
    model.eval()
    with torch.no_grad():
        actual = model(batch)
        hook = model.pe_encoder.register_forward_hook(
            lambda _module, _args, output: torch.zeros_like(output)
        )
        ablated = model(batch)
        hook.remove()
    assert not torch.allclose(actual, ablated, atol=1e-7, rtol=1e-7)
    assert torch.isfinite((actual - batch.y).abs().mean())


@pytest.mark.parametrize("encoding", ["se", "pe"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_fp32_islands_under_bfloat16_autocast(device, encoding, monkeypatch):
    if device == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA with BF16 support is required; CPU results do not validate server CUDA")
    batch = collate([_graph(5, complete=True), _graph(4, forest=True)]).to(device)
    model = CycleBasisPEModel(
        dataset="zinc12k", encoding=encoding, hidden=16, pe_dim=8, layers=3
    ).to(device)
    original = torch.sparse.mm
    sparse_dtypes = []

    def observed(matrix, values, *args, **kwargs):
        sparse_dtypes.append((matrix.dtype, values.dtype))
        return original(matrix, values, *args, **kwargs)

    monkeypatch.setattr(torch.sparse, "mm", observed)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        predicted = model(batch)
        loss = (predicted.float() - batch.y).abs().mean()
    loss.backward()
    assert sparse_dtypes and all(
        left == right == torch.float32 for left, right in sparse_dtypes
    )
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("dataset,targets", [("zinc12k", 1), ("peptides_struct", 11)])
def test_official_target_width_is_preserved(dataset, targets):
    model = CycleBasisPEModel(dataset=dataset)
    assert model.graph_head.out_features == targets


@pytest.mark.parametrize("kwargs", [{"bond_dim": 0, "pe_dim": 3}, {"bond_dim": 3, "pe_dim": 0}])
def test_encoder_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError, match="positive"):
        LeftNullBasisEncoder(**kwargs)


@pytest.mark.parametrize("option", ["column_chunk_size", "basis_pair_budget", "basis_execution"])
def test_obsolete_projector_options_fail_loudly(option):
    with pytest.raises(TypeError, match=option):
        CycleBasisPEModel(dataset="zinc12k", **{option: 2})


def test_encoder_rejects_dense_membership_and_inconsistent_shapes():
    encoder = LeftNullBasisEncoder(5, 8)
    with pytest.raises(ValueError, match="sparse COO"):
        encoder(
            torch.randn(4, 5), torch.zeros(4, 1), torch.ones(1),
            torch.ones(4), torch.zeros(4, 2),
        )
    batch = collate([_graph(4)])
    with pytest.raises(ValueError, match="lengths"):
        encoder(
            torch.randn(4, 5), batch.cycle_membership, torch.ones(2),
            torch.ones(4), batch.edge_cycle_features,
        )


def test_protocol_is_explicit_about_selected_dfs_dependence_and_no_projector():
    protocol = architecture_protocol()
    assert protocol["model"] == MODEL_NAME == MODEL_NAMES["se"] == "cycle_dfs_se_v2"
    assert "a different DFS tree" in protocol["symmetry"]
    assert "may change" in protocol["symmetry"]
    assert "no QR, SVD" in protocol["execution"]
    assert "one sparse block-diagonal physical batch" in protocol["execution"]


def _captured_edge_features(encoder, bond, batch):
    captured = []
    hook = encoder.edge_psi.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0])
    )
    output = _encode(encoder, bond, batch)
    hook.remove()
    assert len(captured) == 1
    return output, captured[0]


def test_pe_residual_matches_dense_cyclic_distance_kernel_on_tiny_fixture():
    """The independent dense pair kernel exists only in this tiny unit reference."""
    torch.manual_seed(83)
    graph = _graph(5, complete=True)
    batch = collate([graph])
    pe = LeftNullBasisEncoder(5, 7, encoding="pe")
    se = LeftNullBasisEncoder(5, 7, encoding="se")
    se.load_state_dict(pe.state_dict(), strict=True)
    bond = torch.randn(len(graph.edge_attr), 5)
    _, pe_features = _captured_edge_features(pe, bond, batch)
    _, se_features = _captured_edge_features(se, bond, batch)
    actual = pe_features[:, 7:14] - se_features[:, 7:14]
    values = pe.column_phi(bond)
    expected = torch.zeros_like(values)
    edge_ids, cycle_ids = graph.cycle_basis.indices()
    for cycle in range(graph.cycle_basis.shape[1]):
        selected = cycle_ids == cycle
        edges = edge_ids[selected]
        position = graph.cycle_position_indices[selected].float()
        length = graph.cycle_lengths[cycle]
        pair_angle = 2 * torch.pi * (position[:, None] - position[None, :]) / length
        expected.index_add_(0, edges, pair_angle.cos() @ values[edges] / length)
    expected = expected / batch.edge_cycle_counts.clamp_min(1)[:, None]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_pe_is_invariant_to_independent_cycle_origin_and_reversal_with_all_gradients():
    torch.manual_seed(89)
    batch = collate([_graph(6, complete=True), _disconnected_graph()])
    cycle_ids = batch.cycle_membership.indices()[1]
    origin = torch.linspace(-2.3, 1.7, len(batch.cycle_lengths))[cycle_ids]
    direction = torch.where(cycle_ids % 2 == 0, -1.0, 1.0)
    cosine, sine = batch.cycle_position_values
    transported = torch.stack((
        cosine * origin.cos() - direction * sine * origin.sin(),
        cosine * origin.sin() + direction * sine * origin.cos(),
    ))
    changed = replace(batch, cycle_position_values=transported)
    original = LeftNullBasisEncoder(5, 7, encoding="pe")
    modified = copy.deepcopy(original)
    bond = torch.randn(len(batch.edge_attr), 5)
    expected = _encode(original, bond, batch)
    actual = _encode(modified, bond, changed)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.randn_like(actual)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    _assert_parameter_gradients_match(modified, original)


def test_pe_special_bond_response_depends_on_undirected_cyclic_distance():
    torch.manual_seed(112)
    graph = _graph(6)
    batch = collate([graph])
    bond = torch.ones(6, 5)
    bond[0] = 3.0
    se = LeftNullBasisEncoder(5, 9, encoding="se")
    pe = LeftNullBasisEncoder(5, 9, encoding="pe")
    pe.load_state_dict(se.state_dict(), strict=True)
    se_output, se_features = _captured_edge_features(se, bond, batch)
    pe_output, pe_features = _captured_edge_features(pe, bond, batch)
    # All ordinary bonds receive the same SE even at different cyclic distances.
    torch.testing.assert_close(
        se_output[1:], se_output[1].expand_as(se_output[1:]), atol=2e-7, rtol=2e-6
    )
    position = graph.cycle_position_indices
    distance = (position - position[0]).remainder(6)
    distance = torch.minimum(distance, 6 - distance)
    values = pe.column_phi(bond)
    contrast = (values[0] - values[1]) / 6
    expected = (2 * torch.pi * distance.float() / 6).cos()[:, None] * contrast
    actual = pe_features[:, 9:18] - se_features[:, 9:18]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    near = torch.nonzero(distance == 1).flatten()[0]
    far = torch.nonzero(distance == 3).flatten()[0]
    assert not torch.allclose(pe_output[near], pe_output[far], atol=1e-7, rtol=1e-7)


def test_uniform_single_cycle_does_not_invent_distinct_positions():
    torch.manual_seed(97)
    batch = collate([_graph(6)])
    pe = LeftNullBasisEncoder(5, 8, encoding="pe")
    se = LeftNullBasisEncoder(5, 8, encoding="se")
    se.load_state_dict(pe.state_dict(), strict=True)
    bond = torch.ones(6, 5)
    actual, expected = _encode(pe, bond, batch), _encode(se, bond, batch)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual, actual[0].expand_as(actual), atol=3e-6, rtol=3e-5)


def test_se_and_pe_have_identical_parameters_and_zero_relative_residual_recovers_se():
    torch.manual_seed(101)
    batch = collate([_graph(5, complete=True), _graph(4, forest=True), _disconnected_graph()])
    se = CycleBasisPEModel(dataset="zinc12k", encoding="se", hidden=16, pe_dim=8, layers=3)
    pe = CycleBasisPEModel(dataset="zinc12k", encoding="pe", hidden=16, pe_dim=8, layers=3)
    pe.load_state_dict(se.state_dict(), strict=True)
    assert list(se.state_dict()) == list(pe.state_dict())
    assert sum(p.numel() for p in se.parameters()) == sum(p.numel() for p in pe.parameters())
    # Explicit unit ablation of the relative term, never a valid prepared/cache
    # phase payload or a runtime fallback: deleting R must recover the fixed SE.
    ablated = replace(batch, cycle_position_values=torch.zeros_like(batch.cycle_position_values))
    actual, expected = pe(ablated), se(batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (actual - batch.y).square().mean().backward()
    (expected - batch.y).square().mean().backward()
    _assert_parameter_gradients_match(pe, se)


@pytest.mark.parametrize("bad", [None, torch.empty(1, 4), torch.ones(2, 4, dtype=torch.long)])
def test_pe_requires_aligned_floating_cycle_positions(bad):
    batch = collate([_graph(4)])
    encoder = LeftNullBasisEncoder(5, 8, encoding="pe")
    with pytest.raises(ValueError, match="cycle_position_values"):
        encoder(
            torch.ones(4, 5), batch.cycle_membership, batch.cycle_lengths,
            batch.edge_cycle_counts, batch.edge_cycle_features, bad,
        )


def test_encoding_names_and_protocol_identify_relative_residual_and_matched_parameters():
    assert MODEL_NAMES == {"se": "cycle_dfs_se_v2", "pe": "cycle_dfs_relative_pe_v2"}
    protocol = architecture_protocol("pe")
    assert protocol["model"] == MODEL_NAMES["pe"]
    assert "K_[1+cos] minus K_mean" in protocol["positional_encoding"]
    assert "not general graph shortest-path distance" in protocol["relative_position"]
    assert "identical learned modules" in protocol["parameter_matching"]
    with pytest.raises(ValueError, match="encoding"):
        CycleBasisPEModel(dataset="zinc12k", encoding="unknown")
    with pytest.raises(ValueError, match="encoding"):
        architecture_protocol("unknown")
