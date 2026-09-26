"""Synthetic CUDA audit of the actual transductive sampling path; no benchmark claim."""

import pytest
import torch

from experiments.aggregation_comparison import engine
from tests.test_aggregation_comparison_cuda import cuda_required, reference_arguments  # noqa: F401


def transductive_payload():
    n = 160
    ids = torch.arange(n)
    edges = (
        torch.cat([torch.stack((ids, (ids + 1) % n)), torch.stack((ids, (ids + 7) % n))], 1)
        .sort(0)
        .values
    )
    edges = torch.unique(edges, dim=1)
    masks = {name: torch.zeros(n, dtype=torch.bool) for name in ("train", "validation", "test")}
    masks["train"][:96] = True
    masks["validation"][96:128] = True
    masks["test"][128:] = True
    return {
        "dataset": "ogbn-arxiv",
        "classes": 7,
        "graphs": [{"x": torch.randn(n, 50), "y": ids % 7, "incidence_edge_index": edges}],
        "splits": masks,
    }


@pytest.mark.parametrize("mode", ["full", "cluster", "cluster_disjoint"])
@pytest.mark.parametrize(
    "arm", ["incidence_fixed", "incidence_shared", "incidence_energy_pre_lift"]
)
def test_actual_sampler_cuda_loss_optimizer_and_label_isolation(mode, arm, record_property):
    engine.base._seed(49)
    payload = transductive_payload()
    args = reference_arguments(arm, "fp32")
    args.dataset = "ogbn-arxiv"
    args.sampling = mode
    args.sample_seed_batch_size = 32  # explicitly synthetic debug, production profile unchanged
    args.num_neighbors = [2]
    if mode == "cluster_disjoint":
        args.sample_context_seed_batch_size = 8
        args.sample_context_workers = 2
    engine.validate_args(args)
    inputs = engine.PreparedInputs(payload, args)
    seed_ids = []
    for batch in inputs._cpu_training(1):
        local = batch.selected_indices
        seed_ids.append(batch.graph.global_node_id[local] if mode != "full" else local)
        if mode == "cluster_disjoint":
            assert batch.topology.num_graphs == 4
        if mode != "full":
            assert batch.graph.x.shape[0] < 160
    assert torch.equal(torch.cat(seed_ids).sort().values, torch.arange(96))
    device = torch.device("cuda:0")
    model = engine.make_model(payload, args, device)
    optimizer = engine.make_optimizer(model, args.learning_rate)
    before = engine.base.state_sha256(model)
    result = engine.run_training_epoch(model, optimizer, inputs, args, device, 1, validate=True)
    assert result["processed_units"] == 96
    assert result["optimizer_steps"] == (1 if mode == "full" else 3)
    assert before != engine.base.state_sha256(model)
    assert all(p.is_cuda and p.grad is not None and p.grad.is_cuda for p in model.parameters())
    evaluation = engine.evaluate(model, inputs, args, device)
    assert evaluation["counts"]["total"] == 32
    graph = next(inputs.validation_batches(device)).graph
    model.eval()
    with torch.no_grad():
        expected = model(graph)
        c = [op.last_effective_c.clone() for op in model.layers]
        graph.y = (graph.y + 1) % 7
        actual = model(graph)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for op, prior in zip(model.layers, c, strict=True):
            torch.testing.assert_close(op.last_effective_c, prior, rtol=0, atol=0)
    budget = engine.resolve_budget(inputs, args)
    record_property("synthetic_learning_budget", str(budget))
    record_property("scope", "actual sampler/preparation/CUDA training path, synthetic graph only")
