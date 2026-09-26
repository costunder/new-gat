"""CUDA regression and scaling checks for the external review; synthetic only."""

import json

import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint

from experiments.aggregation_comparison.model import (
    ARMS,
    AggregationClassifier,
    local_gram,
    normalized_graph_step,
)
from experiments.aggregation_comparison.validation import require_reproduction
from tests.test_aggregation_comparison_cuda import (  # noqa: F401
    cuda_required,
    synthetic_disjoint_batch,
)


def old_gram(history, edges, c, chunk):
    depth, nodes, heads, _ = history.shape
    pairs = torch.triu_indices(depth, depth, device=history.device)
    out = history.new_zeros(nodes, heads, pairs.shape[1])
    chunk = max(1, chunk // depth)
    for start in range(0, edges.shape[1], chunk):
        ends = edges[:, start : start + chunk]
        delta = history[:, ends[1]] - history[:, ends[0]]
        value = (delta[pairs[0]] * delta[pairs[1]]).sum(-1).permute(1, 2, 0)
        value = value * c[start : start + chunk, :, None] / 2
        out = out.index_add(0, ends[0], value).index_add(0, ends[1], value)
    return out


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_scatter_output_history_and_conductance_gradient_under_nested_checkpoint(dtype):
    torch.manual_seed(812)
    h = torch.randn(4, 73, 3, 8, device="cuda", dtype=dtype, requires_grad=True)
    edges = torch.randint(73, (2, 201), device="cuda")
    c = torch.rand(201, 3, device="cuda", dtype=dtype, requires_grad=True)
    new_h, new_c = h.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
    reference = old_gram(h, edges, c, 64)
    actual = checkpoint(lambda v, w: local_gram(v, edges, w, 64), new_h, new_c, use_reentrant=False)
    # Fixed random cotangent detects local errors hidden by a global sum.
    cotangent = torch.randn_like(reference)
    old_grads = torch.autograd.grad(reference, (h, c), cotangent)
    new_grads = torch.autograd.grad(actual, (new_h, new_c), cotangent)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for new, old in zip(new_grads, old_grads, strict=True):
        torch.testing.assert_close(new, old, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sgc_inplace_accumulation_preserves_output_and_gradient(dtype):
    torch.manual_seed(32)
    x = torch.randn(57, 16, device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    edges = torch.randint(57, (2, 171), device="cuda")
    from experiments.aggregation_comparison.model import graph_normalization

    degree, weights = graph_normalization(x, edges)
    out = x / degree[:, None]
    for start in range(0, edges.shape[1], 19):
        ends, w = edges[:, start : start + 19], weights[start : start + 19]
        out = out.index_add(0, ends[0], x[ends[1]] * w[:, None])
        out = out.index_add(0, ends[1], x[ends[0]] * w[:, None])
    actual = checkpoint(lambda z: normalized_graph_step(z, edges, 19), y, use_reentrant=False)
    torch.testing.assert_close(actual, out, rtol=0, atol=0)
    cotangent = torch.randn_like(out)
    torch.testing.assert_close(
        torch.autograd.grad(actual, y, cotangent)[0],
        torch.autograd.grad(out, x, cotangent)[0],
        rtol=0,
        atol=0,
    )


def test_diagonal_is_exact_subset_of_full_gram():
    h = torch.randn(4, 19, 2, 8, device="cuda", requires_grad=True)
    c = torch.rand(31, 2, device="cuda", requires_grad=True)
    edges = torch.randint(19, (2, 31), device="cuda")
    pairs = torch.triu_indices(4, 4, device="cuda")
    full = local_gram(h, edges, c, 16)
    diagonal = local_gram(h, edges, c, 16, diagonal_only=True)
    torch.testing.assert_close(diagonal, full[..., pairs[0] == pairs[1]])


def test_all_twelve_incidence_controls_have_paired_initial_function():
    _, graph, _ = synthetic_disjoint_batch("cuda")
    outputs = []
    for arm in ARMS:
        if not arm.startswith("incidence"):
            continue
        torch.manual_seed(62)
        net = (
            AggregationClassifier(
                50,
                7,
                arm=arm,
                selection_config={"condition": "full"},
                hidden_channels=256,
                layers=8,
                heads=8,
                dropout=0,
                edge_chunk_size=128,
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            outputs.append(net(graph))
    assert len(outputs) == 12
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


def test_energy_branch_changes_conductance_gradient(monkeypatch):
    from experiments.aggregation_comparison import model as module

    _, graph, _ = synthetic_disjoint_batch("cuda")
    torch.manual_seed(28)
    net = AggregationClassifier(
        50,
        7,
        arm="incidence_energy",
        selection_config={"condition": "full"},
        hidden_channels=256,
        layers=8,
        heads=8,
        dropout=0,
        edge_chunk_size=128,
    ).cuda()
    with torch.no_grad():
        for weight in net.energy_readouts:
            weight.normal_(std=0.001)
    detached = AggregationClassifier(
        50,
        7,
        arm="incidence_energy",
        selection_config={"condition": "full"},
        hidden_channels=256,
        layers=8,
        heads=8,
        dropout=0,
        edge_chunk_size=128,
    ).cuda()
    detached.load_state_dict(net.state_dict())
    out = net(graph)
    out.square().mean().backward()
    original = module.local_gram
    monkeypatch.setattr(
        module, "local_gram", lambda h, e, c, *a, **k: original(h, e, c.detach(), *a, **k)
    )
    control = detached(graph)
    control.square().mean().backward()
    torch.testing.assert_close(control, out)
    differences = [
        torch.linalg.vector_norm(a.grad - b.grad)
        for (name, a), (_, b) in zip(
            net.named_parameters(), detached.named_parameters(), strict=True
        )
        if ".estimator." in name
    ]
    assert torch.stack(differences).sum() > 0


def test_gpu_profile_removes_full_node_copy_per_edge_chunk(record_property):
    # Same N/E/depth/heads as review, measured on CUDA, not a dataset benchmark.
    torch.manual_seed(37)
    h = torch.randn(4, 4000, 4, 8, device="cuda")
    c = torch.rand(16000, 4, device="cuda")
    edges = torch.randint(4000, (2, 16000), device="cuda")
    reports = {}
    with torch.no_grad():
        for name, fn in (("before", old_gram), ("after", local_gram)):
            fn(h, edges, c, 256)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True
            ) as measurement:
                start.record()
                result = fn(h, edges, c, 256)
                end.record()
                end.synchronize()
            copies = [
                event for event in measurement.key_averages() if event.key == "aten::index_add"
            ]
            reports[name] = {
                "out_of_place_calls": sum(e.count for e in copies),
                "out_of_place_device_bytes": sum(e.device_memory_usage for e in copies),
                "cuda_ms_with_profiler": start.elapsed_time(end),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "result_bytes": result.numel() * result.element_size(),
            }
    assert reports["before"]["out_of_place_calls"] == 500
    assert reports["after"]["out_of_place_calls"] == 0
    record_property("cuda_profile", json.dumps(reports, sort_keys=True))


def test_ppi_validation_records_integer_confusion_counts():
    from types import SimpleNamespace

    from experiments.aggregation_comparison import engine

    logits = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device="cuda")
    labels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
    graph = SimpleNamespace(y=labels)
    batch = SimpleNamespace(graph=graph, selected_indices=None)
    inputs = SimpleNamespace(indices=None, validation_batches=lambda device: iter([batch]))

    class ConstantDebugLogits(torch.nn.Module):
        def forward(self, graph):
            return logits

        def clear_auxiliary_cache(self):
            pass

    result = engine.evaluate(
        ConstantDebugLogits(), inputs, SimpleNamespace(precision="fp32"), torch.device("cuda")
    )
    assert result["counts"] == {"tp": 2, "fp": 1, "fn": 1, "total": 6}
    require_reproduction(result, result, label="synthetic PPI")


def test_gatv2_cached_support_matches_native_pyg_and_reuses_edges():
    from torch_geometric.nn import GATv2Conv

    _, graph, _ = synthetic_disjoint_batch("cuda")
    net = (
        AggregationClassifier(
            50,
            7,
            arm="gatv2",
            selection_config={"condition": "full"},
            hidden_channels=256,
            layers=8,
            heads=8,
            dropout=0,
        )
        .cuda()
        .eval()
    )
    net(graph)
    cached = graph._comparison_gatv2_edges
    net(graph)
    assert graph._comparison_gatv2_edges is cached
    layer = net.layers[0]
    assert all(block.res is None for block in net.layers)
    native = GATv2Conv(
        256, 32, heads=8, dropout=0, add_self_loops=True, share_weights=False, residual=False
    ).cuda()
    native.load_state_dict(layer.state_dict())
    x = net.encoder(graph.x).detach().requires_grad_()
    y = x.detach().clone().requires_grad_()
    edges = graph.incidence_edge_index
    expected = native(x, torch.cat((edges, edges.flip(0)), dim=1))
    actual = layer(y, cached[2])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
    for a, b in zip(layer.parameters(), native.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    graph.incidence_edge_index = edges.clone()
    net(graph)
    assert graph._comparison_gatv2_edges is not cached


@pytest.mark.parametrize("corrupt", [False, True])
def test_training_completion_and_corrupt_reload(monkeypatch, tmp_path, corrupt):
    """Actual CUDA training with an explicitly injected final-evaluation fault."""
    from experiments.aggregation_comparison import engine
    from tests.test_aggregation_comparison_cuda import reference_arguments

    inputs, graph, payload = synthetic_disjoint_batch("cuda")
    args = reference_arguments("gatv2", "fp32")
    args.epochs, args.patience = 4, 4  # Dedicated synthetic test budget only.
    payload["dataset"] = args.dataset
    # Mirror the real CPU dataset cache; PreparedInputs below supplies CUDA batches.
    payload["graphs"] = [
        {"x": graph.x.cpu(), "incidence_edge_index": graph.incidence_edge_index.cpu()}
    ]
    inputs.data, inputs.sampler = graph, None
    inputs.plan_preparation_seconds = 0
    inputs.provenance = [{"explicit_synthetic_cuda_negative_test": True}]
    inputs.metadata = lambda: {"train_count": 256, "provenance": inputs.provenance}
    monkeypatch.setattr(engine, "PreparedInputs", lambda *a: inputs)
    monkeypatch.setattr(
        engine.base,
        "_v5_data_observability",
        lambda *a: {"explicit_synthetic_cuda_negative_test": True},
    )
    evaluate = engine.evaluate
    evaluations = []

    def inject_failure(*values):
        result = evaluate(*values)
        if corrupt and len(evaluations) == 4:
            # A coherent but incorrect count/score; chosen opposite the best.
            best = max(row["metric"] for row in evaluations)
            correct = 0 if best > 0 else result["counts"]["total"]
            result["counts"]["correct"] = correct
            result["metric"] = correct / result["counts"]["total"]
        evaluations.append(result)
        return result

    monkeypatch.setattr(engine, "evaluate", inject_failure)
    output = tmp_path / "explicit-synthetic-completion-test"
    protocol = {
        "explicit_synthetic_cuda_negative_test": True,
        "data_sha256": "a" * 64,
        "split_sha256": {"train": "b" * 64, "validation": "c" * 64},
    }
    if corrupt:
        with pytest.raises(ValueError, match="best checkpoint reload.*reproduction failed"):
            engine.train_model(payload, protocol, args, torch.device("cuda:0"), output)
        assert not (output / "metrics.json").exists()
    else:
        result = engine.train_model(payload, protocol, args, torch.device("cuda:0"), output)
        assert result["status"] == "passed"
        assert engine.inspect_completed(output) == result
        from experiments.aggregation_comparison import audit

        monkeypatch.setattr(engine.base, "load_dataset", lambda *a, **k: (payload, protocol))
        report = audit.audit(output, args.data_root, torch.device("cuda:0"), 5)
        assert report["status"] == "passed"
        assert len(report["repeated_validation"]["evaluations"]) == 5
    assert len(evaluations) == (5 if corrupt else 10)
    assert (output / "best.pt").is_file() and (output / "last.pt").is_file()
