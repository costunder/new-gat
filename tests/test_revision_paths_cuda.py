"""CUDA-only math/visibility verification using explicit synthetic inputs."""

import copy

import pytest
import torch

from experiments.aggregation_comparison import engine, temporal
from experiments.aggregation_comparison.gram import GramReadout, LocalGram
from tests.test_aggregation_comparison_cuda import cuda_required, reference_arguments  # noqa: F401
from tests.test_aggregation_sampling_path_cuda import transductive_payload


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("chunk", [1, 7])
def test_fused_gram_first_derivatives(dtype, shared, diagonal, chunk):
    torch.manual_seed(77)
    history = (
        torch.randn(4, 13, 3, 5, device="cuda", dtype=dtype)
        .to(torch.float64 if dtype == torch.float64 else torch.float32)
        .requires_grad_()
    )
    c = torch.rand(14, 1 if shared else 3, device="cuda", dtype=history.dtype).requires_grad_()
    readout = torch.randn(
        3, 4 if diagonal else 10, 5, device="cuda", dtype=history.dtype
    ).requires_grad_()
    edges = torch.randint(0, 13, (2, 14), device="cuda")
    expected = torch.einsum(
        "nhp,hpd->nhd", LocalGram.apply(history, c, edges, chunk, diagonal), readout
    )
    actual = GramReadout.apply(history, c, readout, edges, chunk, diagonal)
    upstream = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (history, c, readout), upstream)
    gb = torch.autograd.grad(expected, (history, c, readout), upstream)
    tolerance = 1e-11 if dtype == torch.float64 else 3e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("arm", ["incidence", "gcn", "graphsage"])
def test_temporal_unseen_future_isolation_and_sampled_backprop(arm):
    engine.base._seed(123)
    payload = transductive_payload()
    payload["node_year"] = torch.tensor([2017] * 96 + [2018] * 32 + [2019] * 16 + [2020] * 16)
    args = reference_arguments(arm, "fp32")
    args.visibility_protocol = temporal.NAME
    args.sampling = "cluster_disjoint"
    args.sampled_local_baselines = True
    args.complete_supervised_passes = True
    args.learning_budget_policy = "epochs"
    args.sample_seed_batch_size = 32
    args.sample_context_seed_batch_size = 8
    args.sample_context_workers = 2
    args.num_neighbors = [2]
    engine.validate_args(args)
    changed = copy.deepcopy(payload)
    changed["graphs"][0]["x"][128:] += 10000
    changed["graphs"][0]["y"][128:] = 0
    # Alter only edges involving future test nodes, keeping past support unchanged.
    edge = changed["graphs"][0]["incidence_edge_index"]
    changed["graphs"][0]["incidence_edge_index"] = edge[:, (edge < 128).all(0)]
    inputs, other = engine.PreparedInputs(payload, args), engine.PreparedInputs(changed, args)
    assert inputs.provenance == other.provenance
    assert inputs.training_record.graph.x.shape[0] == 96
    assert inputs.validation_record.graph.x.shape[0] == 128
    device = torch.device("cuda:0")
    model = engine.make_model(payload, args, device)
    model.eval()
    with torch.no_grad():
        a = model(next(inputs.validation_batches(device)).graph)
        b = model(next(other.validation_batches(device)).graph)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    optimizer = engine.make_optimizer(model, args.learning_rate)
    before = engine.base.state_sha256(model)
    result = engine.run_training_epoch(model, optimizer, inputs, args, device, 1, validate=True)
    assert result["supervised_pass_evidence"]["supervised_nodes"] == 96
    assert result["optimizer_steps"] == 3
    assert before != engine.base.state_sha256(model)
    assert all(p.grad is not None and p.grad.is_cuda for p in model.parameters())
    test = temporal.TemporalTestInputs(payload, args)
    observations = []

    def observe(_model, batch, logits, index):
        observations.extend(
            batch.graph.temporal_original_ids[batch.selected_indices].cpu().tolist()
        )

    before = engine.base.state_sha256(model)
    result = engine.evaluate(model, test, args, device, observer=observe)
    assert result["counts"]["total"] == 32
    assert sorted(observations) == list(range(128, 160))
    assert [v["nodes"] for v in test.view_evidence] == [144, 160]
    assert before == engine.base.state_sha256(model)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fused_full_reference_energy_model_gradient_and_checkpoint(precision):
    from tests.test_aggregation_comparison_cuda import synthetic_disjoint_batch

    engine.base._seed(33)
    device = torch.device("cuda:0")
    _, graph, payload = synthetic_disjoint_batch(device)
    args = reference_arguments("incidence_energy_pre_lift", precision)
    original = engine.make_model(payload, args, device)
    for p in original.energy_readouts:
        torch.nn.init.normal_(p, std=0.001)
    fused = copy.deepcopy(original)
    fused.gram_implementation = "fused"
    original.dropout = fused.dropout = 0
    with engine.autocast(args, device):
        expected, actual = original(graph), fused(graph)
    ga = torch.autograd.grad(actual.square().sum(), tuple(fused.parameters()))
    gb = torch.autograd.grad(expected.square().sum(), tuple(original.parameters()))
    tolerance = 0.05 if precision == "bf16" else 2e-4
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
