"""CPU debug contracts for new mechanisms; no real GPU/benchmark evidence."""

from __future__ import annotations

import copy

import pytest
import torch

from research.conductance_gat.v5 import train
from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.protocol import SCALE_PROFILES, conductance_configuration
from scripts import run_conductance_v5 as standalone


def _args(*options):
    args = train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "shared_dynamic_c",
            "--output-dir",
            "debug-unused",
            "--hidden-channels",
            "16",
            "--heads",
            "4",
            "--layers",
            "2",
            "--epochs",
            "12",
            "--no-activation-checkpoint",
            *options,
        ]
    )
    train.validate_args(args)
    return args


def _model(args):
    return GraphConditionedConductanceNodeClassifier(
        5,
        3,
        **train.architecture_configuration(args),
        conductance_mode=train.CONDITIONS[args.condition]["conductance_mode"],
    )


@pytest.mark.parametrize(
    "name",
    [
        "conductance_heads",
        "propagation_normalization",
        "conductance_generator",
        "num_relations",
        "edge_direction",
        "propagation_filter",
    ],
)
def test_legacy_configuration_omits_new_inactive_fields(name):
    assert name not in conductance_configuration()
    assert name not in train.architecture_configuration(_args())


@pytest.mark.parametrize(
    "options,key,value",
    [
        (["--conductance-heads", "per_head"], "conductance_heads", "per_head"),
        (["--propagation-normalization", "row"], "propagation_normalization", "row"),
        (["--conductance-generator", "degree_only"], "conductance_generator", "degree_only"),
        (
            ["--conductance-generator", "entropy_exact", "--solver-degree-barrier", "0"],
            "conductance_generator",
            "entropy_exact",
        ),
        (["--propagation-filter", "polynomial3"], "propagation_filter", "polynomial3"),
        (["--num-relations", "3"], "num_relations", 3),
    ],
)
def test_new_configuration_round_trips_to_child_and_model(tmp_path, options, key, value):
    runner_args = standalone.parser().parse_args(["--datasets", "cora", *options])
    standalone._validate(runner_args)
    architecture = standalone._architecture(runner_args)
    assert architecture[key] == value
    jobs = standalone.make_jobs(runner_args, tmp_path / "debug-plan", architecture)
    assert len(jobs) == 2
    for job in jobs:
        offset = job["command"].index("research.conductance_gat.v5.train") + 1
        child_args = train.build_parser().parse_args(job["command"][offset:])
        train.validate_args(child_args)
        assert train.architecture_configuration(child_args)[key] == value
    model = _model(_args(*options))
    assert getattr(model, key) == value


def test_degree_only_is_not_misreported_as_trainable_c():
    model = _model(_args("--conductance-generator", "degree_only"))
    phase = train.configure_phase(model, "joint", 0)
    assert "conductance" not in phase["active_parameter_groups"]
    optimizer = train.make_optimizer(model)
    train.validate_optimizer_parameter_ownership(model, optimizer)
    assert "conductance" not in {group["name"] for group in optimizer.param_groups}
    assert all(not list(operator.estimator.parameters()) for operator in model.operators)


def test_mechanism_ablations_preserve_common_initial_backbone():
    options = [
        [],
        ["--conductance-heads", "per_head"],
        ["--propagation-normalization", "row"],
        ["--conductance-generator", "degree_only"],
        ["--propagation-filter", "polynomial3"],
        ["--conductance-backend", "mlp"],
    ]
    hashes = []
    with torch.random.fork_rng(devices=[]):
        for option in options:
            torch.manual_seed(3847)
            hashes.append(train.common_backbone_initial_state_sha256(_model(_args(*option))))
    assert len(set(hashes)) == 1


def test_filter_coefficients_are_optimizer_owned_without_scalar_weight_decay():
    model = _model(_args("--propagation-filter", "polynomial3"))
    optimizer = train.make_optimizer(model)
    train.validate_optimizer_parameter_ownership(model, optimizer)
    groups = [
        g
        for g in optimizer.param_groups
        if any("polynomial_delta" in n for n in g["parameter_names"])
    ]
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"conductance_heads": "bad"},
        {"propagation_normalization": "bad"},
        {"conductance_generator": "bad"},
        {"num_relations": -1},
        {"num_relations": True},
        {"edge_direction": "directed"},
        {"propagation_filter": "bad"},
        {"conductance_generator": "entropy_exact"},
        {"conductance_backend": "mlp", "conductance_heads": "per_head"},
    ],
)
def test_invalid_mechanisms_are_rejected(kwargs):
    with pytest.raises(ValueError):
        conductance_configuration(**kwargs)


def test_old_resume_identity_cannot_silently_become_multic():
    args = _args()
    common = dict(
        initial_state_sha256="a" * 64,
        source_sha256={"debug.py": "b" * 64},
        runtime_versions={"torch": "debug"},
    )
    protocol = {"data_sha256": "c" * 64}
    schedule = train.phase_schedule(args.epochs, list(args.phase_fractions), args.training_schedule)
    old = train.build_resume_identity(args, protocol, schedule, **common)
    new_args = copy.deepcopy(args)
    new_args.conductance_heads = "per_head"
    new = train.build_resume_identity(new_args, protocol, schedule, **common)
    assert old != new


def test_production_scale_is_not_shrunk():
    assert (
        SCALE_PROFILES["reference"]["hidden_channels"],
        SCALE_PROFILES["reference"]["layers"],
        SCALE_PROFILES["reference"]["heads"],
    ) == (256, 8, 8)
    assert (
        SCALE_PROFILES["large"]["hidden_channels"],
        SCALE_PROFILES["large"]["layers"],
        SCALE_PROFILES["large"]["heads"],
    ) == (384, 12, 8)


def _typed_payload():
    return {
        "graphs": [
            {
                "incidence_edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "edge_relation_id": torch.tensor([0, 1, 0]),
            }
        ]
    }


def test_real_relation_metadata_validated_without_label_derived_types():
    payload = _typed_payload()
    train.validate_relation_metadata(payload, 2)
    with pytest.raises(ValueError, match="cannot be ignored"):
        train.validate_relation_metadata(payload, 0)
    with pytest.raises(ValueError, match="vocabulary"):
        train.validate_relation_metadata(payload, 1)
    del payload["graphs"][0]["edge_relation_id"]
    with pytest.raises(ValueError, match="requires"):
        train.validate_relation_metadata(payload, 2)


def test_sampled_relation_ids_follow_original_physical_edges():
    pyg = pytest.importorskip(
        "torch_geometric.data", reason="real PyG sampling integration unavailable"
    )
    from research.conductance_gat.v5.sampling import TransductiveGraphSampler

    incidence = torch.tensor([[0, 0, 1, 2, 3], [1, 3, 2, 3, 4]])
    relation = torch.tensor([0, 1, 2, 1, 0])
    graph = pyg.Data(
        x=torch.randn(5, 3),
        y=torch.arange(5) % 2,
        incidence_edge_index=incidence,
        edge_index=torch.cat((incidence, incidence.flip(0)), dim=1),
        edge_relation_id=relation,
    )
    sampler = TransductiveGraphSampler(
        graph, torch.arange(5), mode="cluster", seed_batch_size=32, fanouts=[15, 10], model_seed=0
    )
    sample, edge_ids, _, _ = sampler._induced_result(torch.tensor([0, 1, 3]), torch.tensor([0]))
    assert torch.equal(sample.edge_relation_id, relation[edge_ids])
