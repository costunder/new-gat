"""Synthetic CPU integration tests, not production training or GPU measurements."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.conductance_gat.edge_selection import protocol, train
from research.conductance_gat.edge_selection.topology import build_topology


@pytest.fixture
def synthetic_path():
    """A path value only; these tests neither create nor read result files."""
    return Path(__file__).resolve().parent / "not-created-synthetic-unit-path"


def _args(synthetic_path, condition="forest_learned", *, dataset="cora", **changes):
    argv = [
        "--dataset",
        dataset,
        "--condition",
        "shared_dynamic_c",
        "--selection-mode",
        condition,
        "--output-dir",
        str(synthetic_path / "synthetic-output"),
        "--data-root",
        str(synthetic_path / "synthetic-cache"),
        "--device",
        "cpu",
        "--hidden-channels",
        "16",
        "--layers",
        "2",
        "--heads",
        "4",
        "--edge-chunk-size",
        "3",
        "--dropout",
        "0.2",
        "--workers",
        "0",
    ]
    if condition in protocol.BUDGET_MODES:
        argv += ["--chord-fraction", "0.5"]
    if condition == "hard_concrete":
        argv += [
            "--corruption-ratio",
            "0.2",
            "--l0-weight",
            "0.0001",
            "--negative-loss-weight",
            "0.1",
        ]
    args = train.build_parser().parse_args(argv)
    for key, value in changes.items():
        setattr(args, key, value)
    train.validate_args(args)
    return args


def _graph(*, multilabel=False):
    generator = torch.Generator().manual_seed(19)
    incidence = torch.tensor([[0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6], [1, 2, 3, 2, 3, 3, 5, 7, 6, 7, 7]])
    batch = torch.tensor([0] * 4 + [1] * 4 + [2])
    target = torch.arange(9) % 3
    if multilabel:
        target = torch.nn.functional.one_hot(target, 3).float()
    return SimpleNamespace(
        x=torch.randn(9, 5, generator=generator),
        y=target,
        incidence_edge_index=incidence,
        batch=batch,
        _v5_num_graphs=3,
        edge_selection_topology=build_topology(9, incidence, batch, forest_seed=0),
    )


class SyntheticDebugInputs:
    """Two complete synthetic graph-batches; never used by a final-data runner."""

    def __init__(self, *, multilabel=False):
        self.graph = _graph(multilabel=multilabel)
        self.selected = None if multilabel else torch.arange(9)
        self.indices = None if multilabel else {"train": self.selected}
        self.origin_targets = (torch.arange(11) % 2).float()

    def training_batches(self, epoch, device):
        assert device == torch.device("cpu")
        for _ in range(2):
            yield SimpleNamespace(
                graph=self.graph,
                selected_indices=self.selected,
                origin_targets=self.origin_targets,
            )


def _model(args, inputs):
    return train.make_model(
        {"graphs": [{"x": inputs.graph.x}], "classes": 3}, args, torch.device("cpu")
    )


@pytest.mark.parametrize("condition", protocol.MODES)
def test_optimizer_owns_every_actual_parameter_exactly_once(synthetic_path, condition):
    args, inputs = _args(synthetic_path, condition), SyntheticDebugInputs()
    model = _model(args, inputs)
    optimizer = train.make_optimizer(model)
    mapping = dict(model.named_parameters())
    actual = []
    for group in optimizer.param_groups:
        assert len(group["parameter_names"]) == len(group["params"])
        for name, value in zip(group["parameter_names"], group["params"], strict=True):
            assert value is mapping[name]
            assert train.parameter_group(name) == group["name"]
            actual.append(id(value))
    assert len(actual) == len(set(actual)) == len(mapping)
    assert set(actual) == {id(value) for value in mapping.values()}
    gate_groups = [group for group in optimizer.param_groups if group["name"] == "gate"]
    assert bool(gate_groups) == (condition in {"forest_learned", "forest_cycle", "hard_concrete"})
    if gate_groups:
        assert (
            gate_groups[0]["lr"]
            == train.base.COMMON["lr"] * train.base.COMMON["conductance_lr_multiplier"]
        )


@pytest.mark.parametrize("condition", ["forest_learned", "forest_cycle", "hard_concrete"])
@pytest.mark.parametrize("multilabel", [False, True])
def test_real_training_epoch_updates_task_and_auxiliary_parameters_and_clears_live_cache(
    synthetic_path, condition, multilabel
):
    torch.manual_seed(12)
    args = _args(synthetic_path, condition, dataset="ppi" if multilabel else "cora")
    inputs = SyntheticDebugInputs(multilabel=multilabel)
    model = _model(args, inputs)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = train.make_optimizer(model)
    result = train.run_training_epoch(
        model, optimizer, inputs, args, torch.device("cpu"), 1, validate=True
    )
    assert result["optimizer_steps"] == result["train_batches"] == 2
    assert result["train_labels"] == (54 if multilabel else 18)
    assert result["processed_units"] == (6 if multilabel else 18)
    assert result["largest_measured_graph_batch"] == 3
    assert result["largest_measured_physical_edges"] == 11
    assert set(result["first_step_gradient_norms"]) == {
        "backbone",
        "spatial_w",
        "beta",
        "conductance",
        "gate",
    }
    assert all(value > 0 for value in result["first_step_gradient_norms"].values())
    changed_groups = {
        train.parameter_group(name)
        for name, value in model.named_parameters()
        if not torch.equal(value.detach(), before[name])
    }
    assert changed_groups == {"backbone", "spatial_w", "beta", "conductance", "gate"}
    if condition == "hard_concrete":
        assert result["train_l0"] > 0 and result["train_negative_loss"] > 0
    else:
        assert result["train_l0"] == result["train_negative_loss"] == 0
    assert all(
        operator.selector.live_logits is None and operator.selector.live_probability is None
        for operator in model.operators
    )


def test_same_rng_epoch_resume_matches_uninterrupted_cpu_training(synthetic_path, monkeypatch):
    """Only CUDA RNG transport is mocked; model/loss/optimizer and CPU RNG are real."""
    transported = []
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda device: torch.tensor([7, 9], dtype=torch.uint8)
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda value, device: transported.append(value.clone())
    )
    args, inputs = _args(synthetic_path, "hard_concrete"), SyntheticDebugInputs()
    random.seed(55)
    np.random.seed(55)
    torch.manual_seed(55)
    model = _model(args, inputs)
    optimizer = train.make_optimizer(model)
    train.run_training_epoch(model, optimizer, inputs, args, torch.device("cpu"), 1)
    saved = {
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        **train._checkpoint_rng(torch.device("cpu")),
    }
    expected_random = (random.random(), float(np.random.random()))
    expected = train.run_training_epoch(model, optimizer, inputs, args, torch.device("cpu"), 2)
    restored = _model(args, inputs)
    restored_optimizer = train.make_optimizer(restored)
    restored.load_state_dict(saved["model_state"], strict=True)
    restored_optimizer.load_state_dict(saved["optimizer_state"])
    train._restore_rng(saved, torch.device("cpu"))
    assert (random.random(), float(np.random.random())) == expected_random
    actual = train.run_training_epoch(
        restored, restored_optimizer, inputs, args, torch.device("cpu"), 2
    )
    assert actual == expected
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for first, second in zip(
        optimizer.state.values(), restored_optimizer.state.values(), strict=True
    ):
        assert first.keys() == second.keys()
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    torch.testing.assert_close(transported[0], saved["cuda_rng_state"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"chord_fraction": 0.5},
        {"corruption_ratio": 0.2},
        {"negative_loss_weight": 0.1},
        {"l0_weight": 0.001},
        {"selection_temperature": 0.5},
        {"gate_temperature": 0.5},
        {"corruption_seed": 7},
        {"conductance_heads": "shared"},
        {"propagation_normalization": "symmetric"},
        {"conductance_generator": "degree_only"},
        {"propagation_filter": "polynomial3"},
        {"condition": "fixed_c"},
        {"training_schedule": "staged"},
        {"transition_from_checkpoint": Path("old-v5.pt")},
    ],
)
def test_protocol_rejects_inactive_or_changed_scientific_axes(synthetic_path, changes):
    args = _args(synthetic_path, "full")
    for name, value in changes.items():
        setattr(args, name, value)
    with pytest.raises(ValueError):
        protocol.validate(args)


def test_restore_arguments_roundtrips_exact_trained_configuration_without_reusing_output(
    synthetic_path,
):
    args = _args(synthetic_path, "forest_cycle")
    saved = {
        "resume_identity": {
            "training_arguments": train.serializable_arguments(args),
            "configuration": train.configuration(args),
        }
    }
    restored = train.restore_arguments(
        saved, synthetic_path / "audit-only", synthetic_path / "cache-new-location", "cpu"
    )
    assert restored.output_dir == synthetic_path / "audit-only"
    assert restored.data_root == synthetic_path / "cache-new-location"
    assert restored.device == "cpu"
    assert protocol.configuration(restored) == protocol.configuration(args)
    altered = copy.deepcopy(saved)
    altered["resume_identity"]["training_arguments"]["chord_fraction"] = 0.25
    with pytest.raises(ValueError, match="trained configuration"):
        train.restore_arguments(
            altered, synthetic_path / "audit-only", synthetic_path / "cache-new-location", "cpu"
        )


@pytest.mark.parametrize("change", ["source_sha256", "configuration", "research_suite"])
def test_resume_identity_refuses_changed_sources_recipes_and_legacy_evidence(change):
    identity = {
        "research_suite": protocol.SUITE,
        "source_sha256": {"model.py": "a" * 64},
        "configuration": {"edge_selection": {"condition": "forest_cycle", "chord_fraction": 0.5}},
    }
    saved = {
        "resume_identity": identity,
        "resume_identity_sha256": train.base._canonical_sha256(identity),
    }
    train.validate_identity(saved, copy.deepcopy(identity))
    expected = copy.deepcopy(identity)
    expected[change] = "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        train.validate_identity(saved, expected)
    corrupt = copy.deepcopy(saved)
    corrupt["resume_identity"][change] = "altered without updating hash"
    with pytest.raises(ValueError, match="corrupt"):
        train.validate_identity(corrupt, expected)


def test_first_launch_and_resume_share_identity_but_recipe_change_still_fails(
    synthetic_path, monkeypatch
):
    """Operational resume permission must not mutate scientific training identity."""
    monkeypatch.setattr(
        train, "implementation_source_hashes", lambda: {"synthetic-debug-source": "a" * 64}
    )
    monkeypatch.setattr(train.base, "_versions", lambda: {"scope": "synthetic-CPU-unit-test"})
    initial = _args(synthetic_path, "forest_cycle", resume=False)
    resumed = copy.deepcopy(initial)
    resumed.resume = True
    assert train.serializable_arguments(initial) == train.serializable_arguments(resumed)
    assert train.serializable_arguments(resumed)["resume"] is False
    assert initial.resume is False and resumed.resume is True
    inputs = SimpleNamespace(provenance={"scope": "synthetic fixture; no official cache read"})
    data_protocol = {"dataset": "synthetic-debug", "split": "synthetic-debug"}
    budget = {"planned_epochs": 200, "actual_batches_per_epoch": 2}
    before = train.build_identity(initial, data_protocol, budget, "b" * 64, inputs)
    after = train.build_identity(resumed, data_protocol, budget, "b" * 64, inputs)
    assert before == after
    assert train.base._canonical_sha256(before) == train.base._canonical_sha256(after)
    saved = {
        "resume_identity": before,
        "resume_identity_sha256": train.base._canonical_sha256(before),
    }
    train.validate_identity(saved, after)
    changed = copy.deepcopy(resumed)
    changed.chord_fraction = 0.25
    train.validate_args(changed)
    changed_identity = train.build_identity(changed, data_protocol, budget, "b" * 64, inputs)
    assert changed_identity["configuration"] != before["configuration"]
    with pytest.raises(ValueError, match="identity mismatch"):
        train.validate_identity(saved, changed_identity)
