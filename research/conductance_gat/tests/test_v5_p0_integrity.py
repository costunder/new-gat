import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.conductance_gat.v5.diagnostics import require_finite_tensor
from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.train import (
    _canonical_sha256,
    _payload_graph_observability,
    _training_batches,
    _v5_batch_observability,
    _v5_data_observability,
    build_parser,
    build_resume_identity,
    configure_phase,
    make_optimizer,
    parameter_group,
    phase_schedule,
    recover_best_checkpoint,
    validate_active_gradient_connectivity,
    validate_args,
    validate_optimizer_parameter_ownership,
    validate_resume_identity,
    validate_selected_checkpoint,
)


def _graph(structure=None):
    return SimpleNamespace(
        x=torch.randn(9, 6),
        y=torch.arange(9) % 3,
        incidence_edge_index=torch.tensor(
            [[0, 0, 1, 2, 2, 3, 4, 5, 6, 7], [1, 2, 2, 3, 4, 4, 5, 6, 7, 8]],
            dtype=torch.long,
        ),
        graph_structure=structure,
    )


def test_v5_observability_reports_real_payload_usage_and_full_graph_batching():
    graph = _graph()
    payload = {
        "graphs": [
            {
                "x": graph.x,
                "y": graph.y,
                "incidence_edge_index": graph.incidence_edge_index,
            }
        ]
    }
    indices = {
        "train": torch.arange(4),
        "validation": torch.arange(4, 6),
        "test": torch.arange(6, 9),
    }
    args = SimpleNamespace(
        sampling="full",
        epochs=200,
        batch_size=1024,
        workers=0,
        pin_memory=False,
        sample_prefetch=False,
    )

    graph_report = _payload_graph_observability(payload)
    assert graph_report["nodes_per_graph"]["total"] == 9
    assert graph_report["stored_edge_columns_per_graph"]["total"] == 10
    data_report = _v5_data_observability(payload, graph, indices, args)
    assert data_report["full_dataset_count"] == 9
    assert data_report["actual_used_count"] == 6
    assert data_report["actual_used_fraction_of_full_dataset"]["value"] == pytest.approx(2 / 3)
    batch_report = _v5_batch_observability(graph, indices, None, args)
    assert batch_report["configured_physical_batch_size"] == 1
    assert batch_report["effective_batch_size"] == 1
    assert batch_report["planned_maximum_training_batches"]["value"] == 200


def test_ppi_epoch_seed_preserves_minibatch_order_across_resume_with_workers():
    class Batch(SimpleNamespace):
        def to(self, _device, non_blocking=False):
            assert non_blocking is True
            return self

    class Loader:
        num_workers = 4
        persistent_workers = True
        prefetch_factor = 2

        def __init__(self):
            self.generator = torch.Generator()

        def __iter__(self):
            order = torch.randperm(8, generator=self.generator).tolist()
            return iter([Batch(graph_id=value, num_graphs=1) for value in order])

    loader = Loader()
    args = SimpleNamespace(pin_memory=True)

    def order(epoch):
        return [
            graph.graph_id
            for graph, indices in _training_batches(
                {"train": loader}, None, None, epoch, torch.device("cpu"), 17, args
            )
            if indices is None
        ]

    epoch_seven = order(7)
    assert order(8) != epoch_seven
    assert order(7) == epoch_seven


def test_v5_optimizer_ownership_and_all_group_gradients_are_connected():
    model = _model()
    optimizer = make_optimizer(model)
    validate_optimizer_parameter_ownership(model, optimizer)
    graph = _graph()
    phase = configure_phase(model, "joint", 0)
    optimizer.zero_grad(set_to_none=True)
    model(graph).sum().backward()
    validate_active_gradient_connectivity(model, phase["active_parameter_groups"])


def _model():
    return GraphConditionedConductanceNodeClassifier(
        6,
        3,
        hidden_channels=32,
        layers=1,
        heads=4,
        ffn_multiplier=2,
        dropout=0.0,
        conductance_mode="dynamic",
        activation_checkpoint=False,
    )


def _identity():
    args = build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "shared_dynamic_c",
            "--output-dir",
            "out",
            "--epochs",
            "4",
            "--hidden-channels",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--no-activation-checkpoint",
        ]
    )
    validate_args(args)
    return build_resume_identity(
        args,
        {"data_sha256": "d" * 64, "dataset": "cora", "split": "official"},
        phase_schedule(args.epochs, list(args.phase_fractions)),
        initial_state_sha256="i" * 64,
        source_sha256={"implementation.py": "s" * 64},
        runtime_versions={"torch": "fixture"},
    )


def test_nonfinite_logits_and_loss_fail_fast():
    require_finite_tensor(torch.tensor([0.0, 1.0]), "training logits")
    with pytest.raises(FloatingPointError, match="training logits"):
        require_finite_tensor(torch.tensor([float("nan")]), "training logits")
    with pytest.raises(FloatingPointError, match="training loss"):
        require_finite_tensor(torch.tensor(float("inf")), "training loss")


def test_stale_primary_best_recovers_exact_bound_previous_slot(tmp_path):
    primary, previous = tmp_path / "best.pt", tmp_path / "best.previous.pt"
    primary.write_bytes(b"new-uncommitted-best")
    previous.write_bytes(b"old-last-bound-best")
    expected = hashlib.sha256(previous.read_bytes()).hexdigest()
    assert recover_best_checkpoint(primary, previous, expected) == "previous"
    assert primary.read_bytes() == b"old-last-bound-best"
    assert hashlib.sha256(primary.read_bytes()).hexdigest() == expected


def test_resume_and_selected_best_metadata_reject_mismatches():
    identity = _identity()
    digest = _canonical_sha256(identity)
    validate_resume_identity(identity, identity, digest)
    stale = copy.deepcopy(identity)
    stale["cache_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="resume identity mismatch"):
        validate_resume_identity(stale, identity, _canonical_sha256(stale))
    stale_workers = copy.deepcopy(identity)
    stale_workers["configuration"]["workers"] = 4
    with pytest.raises(ValueError, match="resume identity mismatch"):
        validate_resume_identity(
            stale_workers, identity, _canonical_sha256(stale_workers)
        )
    selected = {
        "resume_identity": identity,
        "resume_identity_sha256": digest,
        "epoch": 4,
        "validation": 0.75,
        "selection_role": "primary",
    }
    validate_selected_checkpoint(
        selected,
        expected_identity=identity,
        expected_identity_sha256=digest,
        expected_epoch=4,
        expected_metric=0.75,
    )
    selected["validation"] = 0.7
    with pytest.raises(ValueError, match="best_metric"):
        validate_selected_checkpoint(
            selected,
            expected_identity=identity,
            expected_identity_sha256=digest,
            expected_epoch=4,
            expected_metric=0.75,
        )
    selected["validation"] = 0.75
    selected["selection_role"] = "global_prediction_auxiliary"
    with pytest.raises(ValueError, match="selection role"):
        validate_selected_checkpoint(
            selected,
            expected_identity=identity,
            expected_identity_sha256=digest,
            expected_epoch=4,
            expected_metric=0.75,
        )


def test_graph_form_context_changes_dynamic_c_and_head_beta():
    torch.manual_seed(19)
    network = _model().eval()
    first = _graph(torch.zeros(1, 6))
    network(first)
    first_c = network.operators[0].estimator.last_c.clone()
    first_beta = network.operators[0].last_beta.clone()
    second = SimpleNamespace(**vars(first))
    second.graph_structure = torch.tensor([[8.0, 11.0, 2.0, 1.5, 5.0, 0.2]])
    network(second)
    assert not torch.equal(network.operators[0].estimator.last_c, first_c)
    assert not torch.equal(network.operators[0].last_beta, first_beta)


def _one_step(network, optimizer, data, phase, phase_epoch):
    state = configure_phase(network, phase, phase_epoch)
    before = {name: value.detach().clone() for name, value in network.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(network(data), data.y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (value for value in network.parameters() if value.requires_grad),
        5.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    delta = {name: 0.0 for name in ("backbone", "spatial_w", "beta", "conductance")}
    for name, value in network.named_parameters():
        delta[parameter_group(name)] += float((value.detach() - before[name]).abs().sum())
    return state, delta


def test_phase_steps_update_only_the_declared_parameter_coordinate():
    torch.manual_seed(23)
    network, data = _model(), _graph()
    optimizer = make_optimizer(network)
    calibration, calibration_delta = _one_step(
        network, optimizer, data, "conductance_calibration", 0
    )
    assert calibration["active_parameter_groups"] == ["conductance"]
    assert calibration_delta["conductance"] > 0
    assert all(calibration_delta[name] == 0 for name in ("backbone", "spatial_w", "beta"))
    spatial, spatial_delta = _one_step(network, optimizer, data, "alternating", 1)
    assert spatial["active_parameter_groups"] == ["backbone", "beta", "spatial_w"]
    assert spatial_delta["conductance"] == 0
    assert sum(spatial_delta[name] for name in ("backbone", "spatial_w", "beta")) > 0
