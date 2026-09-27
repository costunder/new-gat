"""CUDA smoke tests with reference-size models and explicitly synthetic graphs.

These tests are not a benchmark, a full-data fit measurement, or an A100 MIG test.
Invoke this file explicitly after a CUDA preflight; CPU execution is never used.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.aggregation_comparison import engine, runner
from experiments.aggregation_comparison.model import ARMS, DualAttention, local_gram
from research.conductance_gat.edge_selection.topology import build_topology


@pytest.fixture(scope="module", autouse=True)
def cuda_required():
    # Isolate checkpoint restoration from CUDA atomic-reduction ordering.
    # This applies only to these verification tests, never the training recipe.
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        pytest.fail("CUDA smoke tests require a CUDA-enabled PyTorch environment; no CPU fallback")
    previous = torch.backends.cuda.matmul.allow_tf32
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        from experiments.aggregation_comparison.evidence import synthetic_verification

        with synthetic_verification():
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
        torch.use_deterministic_algorithms(previous_determinism)
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def synthetic_disjoint_batch(device):
    # Four independent 64-node graphs, processed together in one CUDA batch.
    nodes, graphs, features, classes = 64, 4, 50, 7
    ids = torch.arange(nodes)
    pairs = torch.stack((ids, (ids + 1) % nodes))
    pairs = torch.cat((pairs, torch.stack((ids, (ids + 5) % nodes))), dim=1)
    edges = (pairs[:, None, :] + nodes * torch.arange(graphs)[None, :, None]).flatten(1)
    batch = torch.arange(graphs).repeat_interleave(nodes)
    plan = build_topology(nodes * graphs, edges, batch, forest_seed=0)
    graph = SimpleNamespace(
        x=torch.randn(nodes * graphs, features, device=device),
        y=torch.arange(nodes * graphs, device=device) % classes,
        incidence_edge_index=edges.to(device),
        batch=batch.to(device),
        _v5_num_graphs=graphs,
        edge_selection_topology=plan.to(device),
    )
    indices = torch.arange(nodes * graphs, device=device)
    batch = SimpleNamespace(graph=graph, selected_indices=indices)
    inputs = SimpleNamespace(
        indices=indices,
        training_batches=lambda epoch, device: iter([batch]),
        validation_batches=lambda device: iter([batch]),
    )
    return inputs, graph, {"graphs": [{"x": graph.x}], "classes": classes}


def reference_arguments(arm, precision):
    options = runner.parser().parse_args(
        [
            "--run-id",
            "debug-cuda-smoke",
            "--datasets",
            "ogbn-arxiv",
            "--profiles",
            "reference",
            "--arms",
            arm,
            "--hardware-profile",
            "portable",
            "--edge-chunk-size",
            "128",
        ]
    )
    job = runner.make_jobs(options, Path("results/debug-cuda-smoke"))[0]
    args = engine.build_parser().parse_args(job["command"][job["command"].index("-m") + 2 :])
    engine.validate_args(args)
    # Validation resolves portable hardware to FP32. This explicit synthetic
    # precision matrix overrides it AFTER validation; it is not a production
    # hardware profile or evidence that the A6000/MIG allocation fits.
    args.precision = precision
    with engine.autocast(args, torch.device("cuda:0")):
        assert torch.is_autocast_enabled("cuda") == (precision == "bf16")
        assert torch.get_autocast_dtype("cuda") == torch.bfloat16
    assert (args.layers, args.hidden_channels, args.heads) == (8, 256, 8)
    return args


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cuda_reference_model_training_and_optimizer_resume(
    arm, precision, tmp_path, record_property
):
    device = torch.device("cuda:0")
    if precision == "bf16":
        assert torch.cuda.is_bf16_supported(), "requested BF16 path is unsupported"
    engine.base._seed(71)
    inputs, graph, payload = synthetic_disjoint_batch(device)
    args = reference_arguments(arm, precision)
    model = engine.make_model(payload, args, device)
    assert all(p.is_cuda for p in model.parameters()) and graph.x.is_cuda
    optimizer = engine.make_optimizer(model, args.learning_rate)
    torch.cuda.reset_peak_memory_stats(device)
    started, ended = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started.record()
    report = engine.run_training_epoch(model, optimizer, inputs, args, device, 1, validate=True)
    assert all(p.grad is not None and p.grad.is_cuda for p in model.parameters())
    ended.record()
    ended.synchronize()
    assert report["processed_units"] == 256 and report["largest_measured_graph_batch"] == 4
    assert report["optimizer_steps"] == 1
    assert engine.evaluate(model, inputs, args, device)["label_count"] == 256
    peak = torch.cuda.max_memory_allocated(device)
    record_property("gpu", torch.cuda.get_device_name(device))
    record_property("precision", precision)
    record_property("deterministic_algorithms", torch.are_deterministic_algorithms_enabled())
    record_property("synthetic_batch", "4 graphs / 256 nodes / 512 undirected edges")
    record_property("parameters", sum(p.numel() for p in model.parameters()))
    record_property("peak_allocated_bytes", peak)
    record_property("first_training_step_cuda_ms", started.elapsed_time(ended))
    path = tmp_path / "synthetic-cuda-checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        path,
    )
    restored = engine.make_model(payload, args, device)
    saved = torch.load(path, map_location=device, weights_only=True)
    restored.load_state_dict(saved["model"])
    restored_optimizer = engine.make_optimizer(restored, args.learning_rate)
    restored_optimizer.load_state_dict(saved["optimizer"])
    for net, opt in ((model, optimizer), (restored, restored_optimizer)):
        torch.set_rng_state(saved["cpu_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        engine.run_training_epoch(net, opt, inputs, args, device, 2, validate=True)
    # Both tolerances are below one AdamW update; a missing optimizer restore
    # must not be hidden by the much looser BF16 forward-output tolerance.
    tolerance = 2e-6 if precision == "fp32" else 2e-5
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("graphs", [1, 4])
def test_cuda_streamed_attention_matches_dense_gradients(graphs):
    torch.manual_seed(18)
    dense, streamed = DualAttention(32, 4).cuda(), DualAttention(32, 4, chunk_size=17).cuda()
    streamed.load_state_dict(dense.state_dict())
    x = torch.randn(128, 32, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    batch = torch.arange(128, device="cuda") % graphs
    a, b = dense(x, batch, graphs), streamed(y, batch, graphs)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=1e-5)
    for pa, pb in zip(dense.parameters(), streamed.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-4, atol=3e-5)


def test_cuda_local_energy_matches_bilinear_form():
    torch.manual_seed(3)
    h = torch.randn(3, 64, 4, 8, device="cuda", requires_grad=True)
    edges = torch.stack((torch.arange(63), torch.arange(1, 64))).cuda()
    c = torch.rand(63, 4, device="cuda", requires_grad=True)
    result = local_gram(h, edges, c, 32)
    delta = h[:, edges[1]] - h[:, edges[0]]
    pairs = torch.triu_indices(3, 3, device="cuda")
    reference = torch.einsum("pehd,pehd,eh->hp", delta[pairs[0]], delta[pairs[1]], c)
    torch.testing.assert_close(result.sum(0), reference)
    result.square().sum().backward()
    assert torch.isfinite(h.grad).all() and c.grad.abs().sum() > 0
