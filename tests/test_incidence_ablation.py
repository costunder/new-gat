"""Synthetic CPU/debug verification; never a full-data or GPU training claim."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from experiments.incidence_ablation import calibration, engine, provenance, runner
from experiments.incidence_ablation.audit import Observer
from experiments.incidence_ablation.diagnostics import (
    component_mean,
    components,
    cross_hop_gram,
    laplacian,
    reconstruction_probe,
    solve_centered,
)
from experiments.incidence_ablation.model import ARMS, IncidenceClassifier, cross_hop_score
from research.conductance_gat.edge_selection.model import EdgeSelectionClassifier
from research.conductance_gat.edge_selection.topology import build_topology


def graph():
    incidence = torch.tensor([[0, 0, 1, 1, 3, 3, 4], [1, 2, 2, 3, 4, 5, 5]])
    batch = torch.zeros(7, dtype=torch.long)
    return SimpleNamespace(
        x=torch.randn(7, 5, generator=torch.Generator().manual_seed(18)),
        y=torch.arange(7) % 3,
        incidence_edge_index=incidence,
        batch=batch,
        _v5_num_graphs=1,
        edge_selection_topology=build_topology(7, incidence, batch, forest_seed=0),
    )


def model(arm, checkpoint=True):
    return IncidenceClassifier(
        5,
        3,
        arm=arm,
        hidden_channels=16,
        heads=4,
        layers=3,
        dropout=0,
        edge_chunk_size=3,
        activation_checkpoint=checkpoint,
        selection_config={"condition": "full"},
    )


@pytest.mark.parametrize("arm", ARMS)
def test_every_trainable_parameter_has_finite_gradient_and_new_terms_update(arm):
    torch.manual_seed(13)
    net, data = model(arm), graph()
    optimizer = engine.make_optimizer(net)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    loss = F.cross_entropy(net(data), data.y)
    loss.backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if "lift_projection" in name or "hop_coefficients" in name:
            assert parameter.grad.norm() > 0, name
    optimizer.step()
    for name, parameter in net.named_parameters():
        if "lift_projection" in name or "hop_coefficients" in name:
            assert not torch.equal(parameter, before[name]), name
    assert torch.isfinite(net(data)).all()


def test_all_arms_have_same_seed_paired_backbone_and_neutral_initial_function():
    outputs, hashes = [], []
    for arm in ARMS:
        torch.manual_seed(23)
        net = model(arm).eval()
        hashes.append(engine.shared_initial_state_sha256(net))
        outputs.append(net(graph()))
    assert len(set(hashes)) == 1
    for value in outputs[1:]:
        torch.testing.assert_close(value, outputs[0])
    torch.manual_seed(23)
    old = EdgeSelectionClassifier(
        5,
        3,
        hidden_channels=16,
        heads=4,
        layers=3,
        dropout=0,
        edge_chunk_size=3,
        selection_config={"condition": "full"},
    )
    torch.testing.assert_close(old(graph()), outputs[0])


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_forward_gradient_and_state_restore(arm):
    torch.manual_seed(33)
    first, data = model(arm, checkpoint=False), graph()
    for op in first.operators:
        if op.hop_coefficients is not None:
            op.hop_coefficients.data.fill_(0.1)
        if op.lift_projection is not None:
            op.lift_projection.data[:, op.head_width :] = 0.03
    second = copy.deepcopy(first)
    second.activation_checkpoint = True
    a, b = first(data), second(data)
    torch.testing.assert_close(a, b)
    F.cross_entropy(a, data.y).backward()
    F.cross_entropy(b, data.y).backward()
    for (name, x), (_, y) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        torch.testing.assert_close(x.grad, y.grad, msg=name)
    third = model(arm)
    third.load_state_dict(second.state_dict(), strict=True)
    torch.testing.assert_close(third(data), b)


def test_bilinear_matches_dense_local_form_and_orientation_invariance():
    torch.manual_seed(5)
    history = torch.randn(4, 7, 2, 3, requires_grad=True)
    edges = graph().incidence_edge_index
    weight = torch.rand(7, 2) + 0.1
    coefficients = torch.randn(2, 6, requires_grad=True)
    actual = cross_hop_score(history, edges, weight, coefficients, 2)
    expected = torch.zeros(7, 2)
    degree = torch.zeros(7, 2)
    pairs = torch.triu_indices(4, 4, 1)
    for e, (tail, head) in enumerate(edges.T):
        delta = history[:, head] - history[:, tail]
        energy = ((delta[pairs[0]] * delta[pairs[1]]).sum(-1).T * coefficients).sum(-1) / 3**0.5
        expected[tail] += weight[e] * energy
        expected[head] += weight[e] * energy
        degree[tail] += weight[e]
        degree[head] += weight[e]
    torch.testing.assert_close(actual, expected / degree.clamp_min(1e-30))
    torch.testing.assert_close(
        actual, cross_hop_score(history, edges.flip(0), weight, coefficients, 4)
    )
    actual.square().sum().backward()
    assert history.grad.norm() > 0 and coefficients.grad.norm() > 0


def test_quadratic_recovery_cg_dense_pseudoinverse_and_unidentifiable_components():
    torch.manual_seed(13)
    edges = torch.tensor([[0, 1, 0, 3], [1, 2, 2, 4]])
    x = torch.randn(6, 4, dtype=torch.float64) + 3
    x[3:5, 2] = 2
    weight = torch.arange(1, 5, dtype=torch.float64)
    count, labels = components(edges, len(x))
    delta = x[edges[1]] - x[edges[0]]
    recovered, _ = solve_centered(edges, weight, delta, labels, count)
    matrix = laplacian(edges, weight, len(x)).to_dense()
    torch.testing.assert_close(recovered, torch.linalg.pinv(matrix) @ matrix @ x)
    torch.testing.assert_close(recovered, x - component_mean(x, labels, count)[labels])
    report = reconstruction_probe(x, edges, weight)
    assert report["linear_minimum_norm_relative_l2"] > 0.5
    assert report["pre_quadratic_identifiable_relative_l2"] < 1e-6
    assert report["linear_with_oracle_means_relative_l2"] < 1e-6
    assert report["identifiable_component_features"] == 7
    assert report["total_component_features"] == 12
    constant = reconstruction_probe(torch.ones_like(x), edges, weight)
    assert constant["identifiable_component_features"] == 0
    assert constant["pre_quadratic_identifiable_relative_l2"] is None
    noisy = reconstruction_probe(x, edges, weight, noise_relative=1e-3)
    assert (
        noisy["pre_quadratic_identifiable_relative_l2"]
        > report["pre_quadratic_identifiable_relative_l2"]
    )


def test_cross_hop_gram_equals_trace_form():
    torch.manual_seed(20)
    h = torch.randn(4, 7, 5, dtype=torch.double)
    edges = graph().incidence_edge_index
    weight = torch.rand(edges.shape[1], dtype=torch.double) + 1
    result = cross_hop_gram(h, edges, weight, 3)
    expected = torch.einsum("knf,nm,lmf->kl", h, laplacian(edges, weight, 7).to_dense(), h)
    torch.testing.assert_close(torch.tensor(result["energy_gram"], dtype=torch.double), expected)


def test_fresh_matrix_retains_full_recipe_and_independent_sources():
    args = runner.parser().parse_args(
        ["--run-id", "debug-contract", "--datasets", "ppi", "--profiles", "reference", "large"]
    )
    runner.validate_args(args)
    jobs = runner.make_jobs(args, Path("results/incidence_ablation/debug-contract"))
    assert len(jobs) == 16
    for job in jobs:
        child = calibration.parse_job(job)
        assert child.layers == (8 if job["profile"] == "reference" else 12)
        assert child.hidden_channels == (256 if job["profile"] == "reference" else 384)
        assert child.heads == 8 and child.epochs == 200 and child.sampling == "full"
        assert child.ablation_arm == job["variant_id"]
    snapshot = provenance.source_snapshot()
    assert "experiments/incidence_ablation/model.py" in snapshot
    assert not any(name.startswith("experiments/") for name in provenance.core_snapshot())
    assert provenance.require_source_compatibility(snapshot, snapshot, scope="debug") is None
    with pytest.raises(ValueError, match="source mismatch"):
        provenance.require_source_compatibility({}, snapshot, scope="debug")


def test_pre_post_lift_difference_and_linear_capacity_control():
    models = []
    for arm in ("linear_lift", "pre_lift", "post_lift"):
        torch.manual_seed(51)
        net = model(arm)
        for op in net.operators:
            op.lift_projection.data[:, op.head_width :] = torch.eye(op.head_width) * 0.5
        models.append(net)
    assert len({sum(p.numel() for p in net.parameters()) for net in models}) == 1
    assert not torch.allclose(models[1](graph()), models[2](graph()))


def test_mig_portable_recipe_preserves_model_data_and_all_eight_arms():
    args = runner.parser().parse_args(
        [
            "--run-id",
            "debug-mig-contract",
            "--datasets",
            "ppi",
            "--profiles",
            "reference",
            "--hardware-profile",
            "portable",
            "--edge-chunk-size",
            "4096",
            "--activation-checkpoint",
            "--min-free-gb",
            "8",
        ]
    )
    runner.validate_args(args)
    jobs = runner.make_jobs(args, Path("results/incidence_ablation/debug-mig-contract"))
    assert len(jobs) == 8
    for job in jobs:
        child = calibration.parse_job(job)
        assert (child.layers, child.hidden_channels, child.heads) == (8, 256, 8)
        assert child.batch_size == 2 and child.epochs == 200 and child.sampling == "full"
        assert child.edge_chunk_size == 4096 and child.activation_checkpoint
        assert child.precision == "fp32" and not child.tf32
        assert child.hardware_profile == "portable"


class SyntheticDebugInputs:
    def __init__(self, multilabel):
        self.graph = graph()
        if multilabel:
            self.graph.y = F.one_hot(self.graph.y, 3).float()
        self.indices = None if multilabel else {"train": torch.arange(7)}

    def training_batches(self, epoch, device):
        for _ in range(2):
            yield SimpleNamespace(
                graph=self.graph,
                selected_indices=None if self.indices is None else torch.arange(7),
                origin_targets=torch.ones(7),
                topology=self.graph.edge_selection_topology,
            )


def debug_args(multilabel=False):
    args = engine.build_parser().parse_args(
        [
            "--dataset",
            "ppi" if multilabel else "cora",
            "--condition",
            "shared_dynamic_c",
            "--selection-mode",
            "full",
            "--ablation-arm",
            "bilinear_pre_lift",
            "--output-dir",
            "not-created-debug-path",
            "--device",
            "cpu",
            "--hidden-channels",
            "16",
            "--heads",
            "4",
            "--layers",
            "3",
            "--edge-chunk-size",
            "3",
            "--workers",
            "0",
            "--dropout",
            "0.2",
        ]
    )
    engine.validate_args(args)
    return args


@pytest.mark.parametrize("multilabel", [False, True])
def test_real_engine_epoch_resume_matches_uninterrupted_optimizer_and_rng(multilabel):
    torch.manual_seed(63)
    args, inputs = debug_args(multilabel), SyntheticDebugInputs(multilabel)
    payload = {"graphs": [{"x": inputs.graph.x}], "classes": 3}
    net = engine.make_model(payload, args, torch.device("cpu"))
    optimizer = engine.make_optimizer(net)
    first = engine.run_training_epoch(
        net, optimizer, inputs, args, torch.device("cpu"), 1, validate=True
    )
    assert first["optimizer_steps"] == 2
    assert set(first["first_step_gradient_norms"]) == {
        "backbone",
        "spatial_w",
        "beta",
        "conductance",
    }
    saved_model, saved_optimizer = (
        copy.deepcopy(net.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    saved_rng = torch.get_rng_state().clone()
    expected = engine.run_training_epoch(net, optimizer, inputs, args, torch.device("cpu"), 2)
    restored = engine.make_model(payload, args, torch.device("cpu"))
    restored.load_state_dict(saved_model)
    restored_optimizer = engine.make_optimizer(restored)
    restored_optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(saved_rng)
    actual = engine.run_training_epoch(
        restored, restored_optimizer, inputs, args, torch.device("cpu"), 2
    )
    assert actual == expected
    for name, value in net.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_full_observer_consumes_all_layers_and_preserves_model_parameters():
    torch.manual_seed(82)
    args, inputs = debug_args(), SyntheticDebugInputs(False)
    net = model("bilinear_pre_lift").eval()
    net.capture = True
    batch = next(inputs.training_batches(0, torch.device("cpu")))
    before = engine.base.state_sha256(net)
    observer = Observer(args)
    with torch.no_grad():
        observer(net, batch, net(batch.graph), 0)
    result = observer.report(0.5)
    assert len(result["layers_and_batches"][0]["layers"]) == 3
    assert set(result["interventions"]) == {"disable_bilinear", "remove_second_lift_channel"}
    assert engine.base.state_sha256(net) == before
    assert net.last_history is None
    assert all(op.last_probe is None for op in net.operators)


def test_disconnected_isolates_and_edge_streaming_do_not_hide_coordinates():
    torch.manual_seed(35)
    x = torch.randn(7, 6, dtype=torch.double) + 4
    edges = graph().incidence_edge_index
    weight = torch.ones(edges.shape[1], dtype=torch.double)
    a = reconstruction_probe(x, edges, weight, edge_chunk_size=2)
    b = reconstruction_probe(x, edges, weight, edge_chunk_size=20)
    assert a["nodes"] == b["nodes"] == 7
    assert a["edges"] == b["edges"] == 7
    assert a["identifiable_component_features"] == b["identifiable_component_features"] == 6
    assert a["pre_quadratic_identifiable_relative_l2"] < 1e-6
    assert b["pre_quadratic_identifiable_relative_l2"] < 1e-6
