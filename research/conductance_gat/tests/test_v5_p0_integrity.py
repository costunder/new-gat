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
    build_parser,
    build_resume_identity,
    configure_phase,
    make_optimizer,
    parameter_group,
    phase_schedule,
    recover_best_checkpoint,
    validate_args,
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
    selected = {
        "resume_identity": identity,
        "resume_identity_sha256": digest,
        "epoch": 4,
        "validation": 0.75,
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
