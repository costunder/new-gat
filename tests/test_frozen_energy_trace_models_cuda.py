"""DEBUG-only frozen checkpoint/stage/replay verification on the actual CUDA device."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from research.frozen_energy_trace.models import (
    assert_source_unchanged,
    discover,
    gcn_propagate,
    load_frozen,
    model_state_hash,
    operator_replays,
    trace_stages,
)
from research.wedge_propagation.classification.data import load_graph
from research.wedge_propagation.classification.evaluation import classification_metrics


@pytest.fixture(autouse=True)
def cuda_debug_execution():
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA is required for this explicitly DEBUG verification")
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


@pytest.fixture(scope="module")
def selected_debug_models(wedge_completed_classification_debug_run):
    config, descriptors = discover(wedge_completed_classification_debug_run, allow_debug=True)
    assert config["profile"] == "debug"
    graphs = {
        descriptor.dataset: load_graph(descriptor.graph_path, descriptor.graph_sha256).to("cuda")
        for descriptor in descriptors
    }
    return config, descriptors, graphs


def test_discovery_identifies_only_nine_selected_seed11_packs_without_loading_weights(
    wedge_completed_classification_debug_run,
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("discovery must not load unrelated seed states or create an optimizer")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    config, descriptors = discover(wedge_completed_classification_debug_run, allow_debug=True)
    assert len(descriptors) == 9
    assert {descriptor.condition for descriptor in descriptors} == {
        "mlp",
        "standard_gcn",
        "polynomial_2",
    }
    assert {descriptor.seed for descriptor in descriptors} == {11}
    assert {descriptor.dataset for descriptor in descriptors} == set(config["data"]["datasets"])
    assert_source_unchanged(descriptors)
    with pytest.raises(ValueError, match="configuration changed"):
        discover(wedge_completed_classification_debug_run)
    with pytest.raises(ValueError, match="seed 11"):
        discover(wedge_completed_classification_debug_run, seed=23, allow_debug=True)


@pytest.mark.parametrize("index", range(9))
def test_selected_best_state_frozen_stages_and_common_operator_replay(
    selected_debug_models,
    index,
    monkeypatch,
):
    config, descriptors, graphs = selected_debug_models
    descriptor = descriptors[index]
    graph = graphs[descriptor.dataset]

    def forbidden(*args, **kwargs):
        pytest.fail("frozen analysis must never construct an optimizer")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    model = load_frozen(graph, descriptor, config)
    payload = torch.load(descriptor.checkpoint, map_location="cpu", weights_only=False)
    selected_index = payload["metadata"]["seeds"].index(11)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value.cpu(),
            payload["best_state"][key][selected_index : selected_index + 1],
            atol=0,
            rtol=0,
        )
    assert not model.training and all(
        not parameter.requires_grad for parameter in model.parameters()
    )
    before = model_state_hash(model)
    stages = trace_stages(model, graph)
    assert [stage.name for stage in stages] == [
        "input",
        "layer_0_projected",
        "layer_0_aggregated",
        "layer_0_activated",
        "layer_1_projected",
        "layer_1_aggregated",
        "logits",
    ]
    assert torch.equal(stages[3].values, stages[2].values.relu())
    assert stages[-1].values is stages[-2].values
    assert model.trace_replay_evidence["within_tolerance"] is True
    assert all(stage.values.device.type == "cuda" for stage in stages)
    with torch.inference_mode():
        for row in descriptor.original_metrics:
            mask = getattr(
                graph,
                {"train": "train_mask", "validation": "val_mask", "test": "test_mask"}[
                    row["split"]
                ],
            )
            ce, accuracy = classification_metrics(stages[-1].values, graph.y, mask)
            assert float(ce[0]) == pytest.approx(row["ce"], abs=2e-6, rel=2e-6)
            assert float(accuracy[0]) == pytest.approx(row["accuracy"], abs=1e-7, rel=0)
    replay = operator_replays(model, graph, stages)
    assert [stage.name for stage in replay] == [
        "layer_0_replay_P",
        "layer_0_replay_P2",
        "layer_1_replay_P",
        "layer_1_replay_P2",
    ]
    # The tiny complete DEBUG graph permits an independent dense operator check.
    p = graph.x.new_zeros((graph.num_nodes, graph.num_nodes))
    p.index_put_((graph.gcn_edges[1], graph.gcn_edges[0]), graph.gcn_weight, accumulate=True)
    lookup = {stage.name: stage for stage in stages + replay}
    for layer in range(2):
        z = lookup[f"layer_{layer}_projected"].values
        expected = p @ z[0]
        torch.testing.assert_close(
            lookup[f"layer_{layer}_replay_P"].values[0], expected, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            lookup[f"layer_{layer}_replay_P2"].values[0], p @ expected, atol=2e-6, rtol=2e-6
        )
        if descriptor.condition == "mlp":
            assert torch.equal(z, lookup[f"layer_{layer}_aggregated"].values)
        if descriptor.condition == "standard_gcn":
            assert model.operator_replay_evidence[f"layer_{layer}_P_matches_actual_aggregation"][
                "within_tolerance"
            ]
    assert model_state_hash(model) == before
    assert_source_unchanged(descriptors)


def test_missing_checkpoint_or_changed_graph_never_falls_back(selected_debug_models, tmp_path):
    config, descriptors, graphs = selected_debug_models
    descriptor = descriptors[0]
    graph = graphs[descriptor.dataset]
    changed = replace(graph, x=graph.x + 0.125)
    with pytest.raises(ValueError, match="fingerprint differs"):
        load_frozen(changed, descriptor, config)
    with pytest.raises(FileNotFoundError):
        load_frozen(graph, replace(descriptor, checkpoint=tmp_path / "missing.pt"), config)
    with pytest.raises(ValueError, match="requires CUDA"):
        load_frozen(graph.to("cpu"), descriptor, config)


def test_output_layer_preserves_negative_logits_without_fake_relu(selected_debug_models):
    config, descriptors, graphs = selected_debug_models
    descriptor = next(item for item in descriptors if item.condition == "mlp")
    graph = graphs[descriptor.dataset]
    model = load_frozen(graph, descriptor, config)
    with torch.no_grad():
        model.projections[0].abs_()
        model.projections[1].fill_(-1)
    stages = trace_stages(model, graph)
    assert bool((stages[-1].values < 0).any())
    assert stages[-1].values is stages[-2].values
    assert all(not (stage.layer == 1 and stage.stage == "activated") for stage in stages)


def test_replay_requires_whole_single_seed_cuda_signal(selected_debug_models):
    _, descriptors, graphs = selected_debug_models
    graph = graphs[descriptors[0].dataset]
    with pytest.raises(ValueError, match="full CUDA seed signal"):
        gcn_propagate(graph, graph.x[None, :3])
