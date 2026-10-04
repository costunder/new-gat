"""DEBUG training: actual CE, independent packed replicas and exact resume."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research.local_energy_relations.placement.data import PredictionGraph
from research.local_energy_relations.prediction.operators import prepare_geometry
from research.local_energy_relations.placement.model import active_layers, parse_condition
from research.local_energy_relations.placement.training import (
    _epoch,
    _resume_payload,
    choose_packing,
    make_model,
    make_optimizer,
    train_pack,
)
from research.local_energy_relations.topology import build_topology

_ROOT = Path(__file__).resolve().parents[1]


def debug_config():
    return json.loads(
        (_ROOT / "research/local_energy_relations/placement/config_debug.json").read_text(
            encoding="utf-8"
        )
    )


def debug_graph(dtype=torch.float32):
    generator = torch.Generator().manual_seed(4761)
    x = torch.rand(7, 4, generator=generator)
    x /= x.sum(1, keepdim=True)
    edges = torch.tensor([[0, 0, 1, 1, 2, 3, 4], [1, 2, 2, 3, 4, 4, 5]])
    masks = [torch.zeros(7, dtype=torch.bool) for _ in range(3)]
    masks[0][:3] = True
    masks[1][3:5] = True
    masks[2][5:] = True
    geometry = prepare_geometry(build_topology(7, edges)).to("cpu", dtype)
    return PredictionGraph(
        "DEBUG-Cora",
        x.to(dtype),
        torch.tensor([0, 1, 2, 0, 1, 2, 0]),
        *masks,
        edges,
        geometry,
        {"classes": 3},
    )


@pytest.mark.parametrize("condition", debug_config()["conditions"])
def test_all_active_parameters_connect_to_ce_and_enabled_lifts_update(condition):
    graph, config = debug_graph(), debug_config()
    model = make_model(graph, condition, [11, 23], config, 2)
    optimizer = make_optimizer(model, config, 0.01)
    rows = _epoch(model, graph, optimizer, 0)
    assert len(rows) == 2
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for label, enabled in [("within", model.has_energy), ("between", model.has_relation)]:
        if enabled:
            assert all(row[f"{label}_ce_gradient_norm"] > 0 for row in rows)
            assert all(row[f"{label}_parameter_update_norm"] > 0 for row in rows)
        else:
            assert all(row[f"{label}_ce_gradient_norm"] == 0 for row in rows)
    # The second update differentiates through E/J using the updated lifts.
    _epoch(model, graph, optimizer, 1)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_only_projection_matrices_receive_weight_decay():
    model = make_model(debug_graph(), "unit__both__all", [11, 23], debug_config(), 2)
    optimizer = make_optimizer(model, debug_config(), 0.01)
    decayed = {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"]
        for parameter in group["params"]
    }
    assert decayed == {id(parameter) for parameter in model.projections}
    assert all(id(parameter) not in decayed for parameter in model.energy_lifts.values())
    assert all(id(parameter) not in decayed for parameter in model.relation_lifts.values())


@pytest.mark.parametrize("placement", ["hidden", "output", "all"])
def test_packed_replicas_match_independent_training_updates(placement):
    graph, config = debug_graph(), debug_config()
    condition = f"local_degree__both__{placement}"
    packed = make_model(graph, condition, [11, 23], config, 2)
    singles = [make_model(graph, condition, [seed], config, 2) for seed in [11, 23]]
    packed_opt = make_optimizer(packed, config, 0.003)
    single_opt = [make_optimizer(model, config, 0.003) for model in singles]
    for epoch in range(2):
        _epoch(packed, graph, packed_opt, epoch)
        for model, optimizer in zip(singles, single_opt, strict=True):
            _epoch(model, graph, optimizer, epoch)
    for key, value in packed.state_dict().items():
        for index, model in enumerate(singles):
            torch.testing.assert_close(
                value[index : index + 1], model.state_dict()[key], atol=2e-6, rtol=2e-5
            )


def test_debug_pack_complete_resume_has_zero_new_updates_and_preserves_source(tmp_path):
    config, graph = debug_config(), debug_graph()
    calibration = {"packed_runs": 2, "path_chunk": 2, "seconds_per_epoch": 0.01, "trials": []}
    source = {"code_digest": "explicit-DEBUG-unit-test"}
    original = tmp_path / "original"
    result = train_pack(
        graph,
        "unit__both__output",
        [11, 23],
        0.003,
        config,
        original,
        source,
        "DEBUG-input",
        "final",
        calibration=calibration,
    )
    before = {path.name: path.read_bytes() for path in original.iterdir() if path.is_file()}
    resumed = train_pack(
        graph,
        "unit__both__output",
        [11, 23],
        0.003,
        config,
        tmp_path / "resumed",
        source,
        "DEBUG-input",
        "final",
        resume_dir=original,
        calibration=calibration,
    )
    assert all(row["new_optimizer_updates"] == 0 for row in resumed["rows"])
    assert all(row["placement"] == "output" for row in resumed["rows"])
    saved = json.loads(result["selected_path"].with_suffix(".json").read_text())
    assert saved["metadata"]["placement"] == "output"
    assert saved["metadata"]["injection_layers"] == [1]
    assert all(row["training_epochs"] == 3 for row in result["rows"])
    assert {path.name: path.read_bytes() for path in original.iterdir() if path.is_file()} == before
    assert torch.equal(result["model"].projections[0], resumed["model"].projections[0])


def test_resume_rejects_changed_metadata_or_checkpoint_bytes(tmp_path):
    config, graph = debug_config(), debug_graph()
    result = train_pack(
        graph,
        "unit__base",
        [11],
        0.003,
        config,
        tmp_path / "run",
        {"code_digest": "DEBUG"},
        "DEBUG-input",
        "tuning",
        calibration={"path_chunk": 2},
    )
    path = result["selected_path"]
    sidecar = json.loads(path.with_suffix(".json").read_text())
    changed = copy.deepcopy(sidecar["metadata"])
    changed["seeds"] = [23]
    with pytest.raises(ValueError, match="identity"):
        _resume_payload(path.parent, changed)
    changed = copy.deepcopy(sidecar["metadata"])
    changed["placement"] = "output"
    changed["injection_layers"] = [1]
    with pytest.raises(ValueError, match="identity"):
        _resume_payload(path.parent, changed)
    path.write_bytes(path.read_bytes() + b"DEBUG-corruption")
    with pytest.raises(ValueError, match="hash"):
        _resume_payload(path.parent, sidecar["metadata"])


def test_packing_selects_measured_throughput_without_reducing_seed_contract(monkeypatch):
    import research.local_energy_relations.placement.training as training

    calls = []

    def measured(graph, condition, seeds, lr, config, chunk):
        calls.append((tuple(seeds), chunk))
        return 0.02, None, 1

    monkeypatch.setattr(training, "benchmark_trial", measured)
    config = debug_config()
    selected = choose_packing(debug_graph(), "unit__both__hidden", [11, 23], 0.003, config, "final")
    assert selected["packed_runs"] == 2
    assert {len(seeds) for seeds, _ in calls} == {1, 2}
    assert config["training"]["final_seeds"] == [11, 23]


@pytest.mark.parametrize("condition", debug_config()["conditions"])
def test_factory_counts_only_active_layer_branch_parameters(condition):
    graph, config = debug_graph(), debug_config()
    model = make_model(graph, condition, [11], config, 2)
    _mode, variant, placement = parse_condition(condition)
    widths = (config["backbone"]["hidden_dim"], graph.num_classes)
    branches = int(variant in ("within", "both")) + int(variant in ("between", "both"))
    expected = graph.num_features * widths[0] + widths[0] * widths[1] + 2
    expected += branches * sum(widths[layer] for layer in active_layers(condition))
    assert model.parameters_per_seed == expected
    assert model.placement == placement
    for container, enabled in ((model.energy_lifts, model.has_energy), (model.relation_lifts, model.has_relation)):
        expected_keys = {str(layer) for layer in active_layers(condition)} if enabled else set()
        assert set(container) == expected_keys
