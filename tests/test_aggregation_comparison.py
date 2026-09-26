"""Synthetic CPU tests only; no benchmark score or CUDA-fit claim."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from experiments.aggregation_comparison import engine, provenance, runner
from experiments.aggregation_comparison.model import (
    ARMS,
    AggregationClassifier,
    DualAttention,
    local_gram,
    normalized_graph_step,
)
from tests.test_incidence_ablation import graph


def model(arm, checkpoint=False):
    return AggregationClassifier(
        5,
        3,
        arm=arm,
        selection_config={"condition": "full"},
        hidden_channels=16,
        layers=3,
        heads=4,
        dropout=0,
        activation_checkpoint=checkpoint,
        edge_chunk_size=3,
    )


@pytest.mark.parametrize("arm", ARMS)
def test_trainable_parameters_reach_loss_and_optimizer(arm):
    torch.manual_seed(91)
    net, data = model(arm), graph()
    optimizer = engine.make_optimizer(net)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    F.cross_entropy(net(data), data.y).backward()
    engine.validate_gradients(net)
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if "energy_readouts" in name:
            assert parameter.grad.norm() > 0, name
    optimizer.step()
    for name, parameter in net.named_parameters():
        if "energy_readouts" in name or "lift_projection" in name:
            assert not torch.equal(parameter, before[name]), name


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_recomputation_matches_forward_and_gradients(arm):
    torch.manual_seed(22)
    a, data = model(arm), graph()
    with torch.no_grad():
        for p in a.energy_readouts:
            p.normal_(std=0.03)
    b = model(arm, checkpoint=True)
    b.load_state_dict(a.state_dict())
    ya, yb = a(data), b(data)
    torch.testing.assert_close(ya, yb)
    F.cross_entropy(ya, data.y).backward()
    F.cross_entropy(yb, data.y).backward()
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, msg=name)


def test_no_hidden_residual_or_ffn_in_incidence_pipeline():
    net, data = model("incidence"), graph()
    assert not any("ffn" in name or "operator_norm" in name for name, _ in net.named_parameters())
    for operator in net.layers:
        with torch.no_grad():
            operator.output_projection.weight.zero_()
            if operator.output_projection.bias is not None:
                operator.output_projection.bias.zero_()
    # A hidden residual would carry the encoded input to the decoder here.
    torch.testing.assert_close(net(data), net.decoder.bias.expand(data.x.shape[0], -1))


def test_local_gram_detects_cancellation_and_matches_global_bilinear_form():
    edges = torch.tensor([[0, 0], [1, 2]])
    h = torch.tensor([0.0, 1.0, -1.0]).reshape(1, 3, 1, 1)
    weight = torch.ones(2, 1)
    result = local_gram(h, edges, weight, 1)
    assert result[0, 0, 0] == 1
    assert h[0, 1:].sum() == 0
    torch.testing.assert_close(local_gram(h * 0, edges, weight, 2), result * 0)
    torch.manual_seed(2)
    h = torch.randn(3, 3, 2, 4, requires_grad=True)
    weight = torch.rand(2, 2, requires_grad=True)
    result = local_gram(h, edges, weight, 1)
    differences = h[:, edges[1]] - h[:, edges[0]]
    pairs = torch.triu_indices(3, 3)
    reference = torch.einsum(
        "pehd,pehd,eh->hp", differences[pairs[0]], differences[pairs[1]], weight
    )
    torch.testing.assert_close(result.sum(0), reference)
    torch.testing.assert_close(result, local_gram(h, edges.flip(0), weight, 5))
    result.sum().backward()
    assert weight.grad.abs().sum() > 0


def test_dual_attention_matches_official_formula_and_disjoint_batching():
    torch.manual_seed(71)
    layer = DualAttention(4, 2)
    x = torch.randn(7, 4)
    q = layer.query(x).reshape(7, 4, 2)
    k = layer.key(x).reshape(7, 4, 2)
    v = layer.value(x).reshape(7, 4, 2)
    metric = torch.einsum("lmh,ldh->mdh", k / 7**0.5, v).softmax(0)
    expected = torch.einsum("lmh,mdh->ldh", q, metric).mean(-1)
    torch.testing.assert_close(layer(x, torch.zeros(7, dtype=torch.long), 1), expected)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    result = layer(x, batch, 2)
    for index in range(2):
        subset = x[batch == index]
        torch.testing.assert_close(
            result[batch == index], layer(subset, torch.zeros(len(subset), dtype=torch.long), 1)
        )


def test_normalized_graph_propagation_matches_dense_reference():
    data = graph()
    edges, x = data.incidence_edge_index, data.x
    adjacency = torch.eye(x.shape[0])
    adjacency[edges[0], edges[1]] = 1
    adjacency[edges[1], edges[0]] = 1
    inverse = adjacency.sum(1).rsqrt()
    expected = inverse[:, None] * adjacency * inverse[None, :]
    torch.testing.assert_close(normalized_graph_step(x, edges, 2), expected @ x)


def test_dual_normalization_cache_reuses_static_graph_and_invalidates_edge_changes():
    net, data = model("dualformer"), graph()
    net(data)
    first = data._comparison_sgc_normalization
    net(data)
    assert data._comparison_sgc_normalization is first
    data.incidence_edge_index[1, 0] = 2
    net(data)
    assert data._comparison_sgc_normalization is not first


@pytest.mark.parametrize("graphs", [1, 2])
def test_streamed_attention_matches_dense_forward_and_gradients(graphs):
    torch.manual_seed(14)
    dense, streamed = DualAttention(4, 2), DualAttention(4, 2, chunk_size=2)
    streamed.load_state_dict(dense.state_dict())
    x = torch.randn(7, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    batch = torch.arange(7) % graphs
    a, b = dense(x, batch, graphs), streamed(y, batch, graphs)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for pa, pb in zip(dense.parameters(), streamed.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad)


def test_shared_endpoints_seed_pairing_and_provenance_isolation():
    hashes = []
    for arm in ARMS:
        torch.manual_seed(40)
        hashes.append(engine.shared_initial_state_sha256(model(arm)))
    assert len(set(hashes)) == 1
    from experiments.incidence_ablation.provenance import source_snapshot as old_sources

    assert not any("aggregation_comparison" in p for p in old_sources())
    assert "experiments/aggregation_comparison/model.py" in provenance.source_snapshot()


@pytest.mark.parametrize("arm", ARMS)
def test_synthetic_training_epoch_checkpoint_optimizer_continuation(arm, tmp_path):
    """Real trainer/model/optimizer path on synthetic CPU data, never a GPU run."""
    torch.manual_seed(83)
    net, data = model(arm, checkpoint=True), graph()
    indices = torch.arange(data.x.shape[0])
    batch = SimpleNamespace(graph=data, selected_indices=indices)
    inputs = SimpleNamespace(
        indices=indices,
        training_batches=lambda epoch, device: iter([batch]),
        validation_batches=lambda device: iter([batch]),
    )
    args = SimpleNamespace(precision="fp32", l0_weight=0.0, negative_loss_weight=0.0)
    optimizer = engine.make_optimizer(net)
    device = torch.device("cpu")
    report = engine.run_training_epoch(net, optimizer, inputs, args, device, 1, validate=True)
    assert report["processed_units"] == len(indices)
    assert report["optimizer_steps"] == 1
    assert engine.evaluate(net, inputs, args, device)["label_count"] == len(indices)
    path = tmp_path / "synthetic-checkpoint.pt"
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = model(arm, checkpoint=True)
    saved = torch.load(path, weights_only=True)
    restored.load_state_dict(saved["model"])
    restored_optimizer = engine.make_optimizer(restored)
    restored_optimizer.load_state_dict(saved["optimizer"])
    engine.run_training_epoch(net, optimizer, inputs, args, device, 2, validate=True)
    engine.run_training_epoch(restored, restored_optimizer, inputs, args, device, 2, validate=True)
    for a, b in zip(net.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_real_cli_plans_recent_comparator_and_full_reference_architecture():
    args = runner.parser().parse_args(
        [
            "--run-id",
            "debug-plan",
            "--datasets",
            "ogbn-arxiv",
            "--profiles",
            "reference",
            "--hardware-profile",
            "portable",
            "--sampling",
            "full",
        ]
    )
    jobs = runner.make_jobs(args, Path("results/debug-plan"))
    assert {j["variant_id"] for j in jobs} == set(ARMS)
    assert "gat" not in ARMS
    for job in jobs:
        child = engine.build_parser().parse_args(job["command"][job["command"].index("-m") + 2 :])
        engine.validate_args(child)
        assert (child.layers, child.hidden_channels, child.heads) == (8, 256, 8)
        assert child.sampling == "full"
        assert child.epochs == 200
