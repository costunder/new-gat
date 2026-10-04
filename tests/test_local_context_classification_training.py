"""DEBUG CE, independent packed Adam states, strict resume and telemetry checks."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.local_context_coupling.classification.common import digest, read_config, write_json
from research.local_context_coupling.classification import training
from research.local_context_coupling.operators import prepare_geometry
from research.local_energy_relations.topology import build_topology


@pytest.fixture(autouse=True)
def debug_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def graph(device="cpu"):
    generator = torch.Generator().manual_seed(4761)
    x = torch.rand((7, 4), generator=generator)
    x /= x.sum(1, keepdim=True)
    edges = torch.tensor([[0, 0, 1, 1, 2, 3, 4], [1, 2, 2, 3, 4, 4, 5]])
    top = build_topology(7, edges)
    geometries = {mode: prepare_geometry(top, mode).to(device, x.dtype) for mode in ("unit", "local_degree")}
    masks = [torch.zeros(7, dtype=torch.bool, device=device) for _ in range(3)]
    masks[0][:3], masks[1][3:5], masks[2][5:] = True, True, True
    value = SimpleNamespace(
        name="DEBUG-Cora", x=x.to(device), y=torch.tensor([0, 1, 2, 0, 1, 2, 0], device=device),
        train_mask=masks[0], val_mask=masks[1], test_mask=masks[2],
        edges=edges.to(device), geometries=geometries, topology=geometries["unit"].topology,
        num_nodes=7, num_features=4, num_classes=3,
    )
    value.geometry_for = lambda mode: value.geometries[mode]
    return value


def config():
    return read_config(profile="debug")


@pytest.mark.parametrize("condition", [f"{mode}__{variant}" for mode in ("unit", "local_degree") for variant in ("off", "fixed", "learned")])
def test_all_active_parameters_connect_to_ce_and_every_epoch_gate_telemetry(condition):
    data, cfg = graph(), config()
    model = training.make_model(data, condition, [11, 23], cfg, 2)
    optimizer = training.make_optimizer(model, cfg, .003)
    rows = training._epoch(model, data, optimizer, 0, diagnostics=True)
    assert len(rows) == 2
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(row["all_ce_gradient_norm"] > 0 for row in rows)
    assert all(row["diagnostics_recorded"] for row in rows)
    for row in rows:
        assert row["layer_0_projected_norm"] > 0 and row["layer_1_output_norm"] > 0
        if condition.endswith("__learned"):
            assert row["theta_available"]
            assert row["theta_before"] == 0
            assert row["cross_ce_gradient_norm"] > 0
            assert row["cross_parameter_update_norm"] > 0
            assert row["theta_after"] - row["theta_before"] == pytest.approx(row["cross_parameter_update_signed"])
            assert 0 < row["cross_gain_after"] < 1
        else:
            assert not row["theta_available"]
            assert row["theta_before"] is None and row["cross_ce_gradient_signed"] is None
            assert row["cross_parameter_update_signed"] is None
            assert row["cross_gain_after"] == int(condition.endswith("__fixed"))
    later = training._epoch(model, data, optimizer, 1, diagnostics=False)
    assert all(not row["diagnostics_recorded"] for row in later)
    assert all(row["layer_0_output_norm"] is None for row in later)
    assert set(rows[0]) == set(later[0])


def test_only_projections_receive_weight_decay_and_disabled_gates_are_absent():
    for variant in ("off", "fixed", "learned"):
        model = training.make_model(graph(), "unit__"+variant, [11, 23], config(), 2)
        optimizer = training.make_optimizer(model, config(), .001)
        decayed = {id(p) for group in optimizer.param_groups if group["weight_decay"] for p in group["params"]}
        assert decayed == {id(p) for p in model.weights}
        assert (model.theta_cross is not None) == (variant == "learned")
        if model.theta_cross is not None:
            assert id(model.theta_cross) not in decayed
        width = config()["backbone"]["hidden_dim"]
        assert model.parameters_per_seed == 4*width+width*3+int(variant == "learned")


@pytest.mark.parametrize("variant", ["off", "fixed", "learned"])
def test_packed_updates_and_adam_moments_match_independent_seed_runs(variant):
    data, cfg, seeds = graph(), config(), [11, 23]
    condition = "local_degree__"+variant
    packed = training.make_model(data, condition, seeds, cfg, 2)
    singles = [training.make_model(data, condition, [seed], cfg, 2) for seed in seeds]
    packed_opt = training.make_optimizer(packed, cfg, .003)
    optimizers = [training.make_optimizer(model, cfg, .003) for model in singles]
    for epoch in range(2):
        training._epoch(packed, data, packed_opt, epoch)
        for model, optimizer in zip(singles, optimizers, strict=True):
            training._epoch(model, data, optimizer, epoch)
    for name, value in packed.state_dict().items():
        for index, model in enumerate(singles):
            torch.testing.assert_close(value[index:index+1], model.state_dict()[name], atol=2e-6, rtol=2e-5)
    for parameter, moments in packed_opt.state.items():
        named = next(name for name, p in packed.named_parameters() if p is parameter)
        for index, (model, optimizer) in enumerate(zip(singles, optimizers, strict=True)):
            single = dict(model.named_parameters())[named]
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(moments[key][index:index+1], optimizer.state[single][key], atol=2e-7, rtol=2e-5)


def test_calibration_disposable_updates_restore_same_named_initialization():
    data, cfg = graph(), config()
    baseline = training.make_model(data, "unit__learned", [11, 23], cfg, 2)
    state = {name: p.clone() for name, p in baseline.state_dict().items()}
    seconds, peak, count = training.benchmark_trial(data, "unit__learned", [11, 23], .003, cfg, 2)
    fresh = training.make_model(data, "unit__learned", [11, 23], cfg, 2)
    assert seconds > 0 and peak is None and count == baseline.parameters_per_seed
    for name, value in fresh.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)


def test_packing_benchmarks_every_candidate_and_keeps_complete_seed_contract(monkeypatch):
    calls = []
    def measured(data, condition, seeds, lr, cfg, chunk):
        calls.append((tuple(seeds), chunk))
        return .02, None, 1
    monkeypatch.setattr(training, "benchmark_trial", measured)
    data, cfg = graph(), config()
    chosen = training.choose_packing(data, "unit__learned", [11, 23], .003, cfg, "final")
    assert chosen["packed_runs"] == 2
    assert {len(seeds) for seeds, _ in calls} == {1, 2}
    maximum = max(data.topology.num_local_edges, data.geometry_for("unit").num_cross_edges)
    assert maximum in {chunk for _, chunk in calls}
    assert cfg["training"]["final_seeds"] == [11, 23]
    assert all(row["includes_diagnostic_memory_probe"] for row in chosen["trials"])


def run(data, cfg, folder, resume=None):
    return training.train_pack(
        data, "unit__learned", [11, 23], .003, cfg, folder,
        {"code_digest": "explicit-DEBUG-training-test"}, "DEBUG-input", "final",
        resume_dir=resume, calibration={"packed_runs": 2, "path_chunk": 2},
    )


def test_full_budget_checkpoint_has_per_seed_best_and_complete_resume_zero_updates(tmp_path):
    data, cfg = graph(), config()
    original = tmp_path/"original"
    result = run(data, cfg, original)
    before = {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}
    resumed = run(data, cfg, tmp_path/"resumed", original)
    assert all(row["new_optimizer_updates"] == 0 for row in resumed["rows"])
    assert all(row["optimizer_updates"] == row["training_epochs"] == 3 for row in result["rows"])
    assert {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()} == before
    for name, value in result["model"].state_dict().items():
        torch.testing.assert_close(value, resumed["model"].state_dict()[name], atol=0, rtol=0)
    payload = torch.load(result["selected_path"], weights_only=False)
    assert len(payload["best_ce"]) == len(payload["best_epoch"]) == 2
    assert len(payload["history"]) == 6
    assert all(row["diagnostics_recorded"] for row in payload["history"])
    assert all(row["theta_available"] for row in payload["history"])
    with (original/"epoch_history.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 6
    assert json.loads((original/"completion.json").read_text())["test_evaluated"] is False


def test_partial_resume_restores_optimizer_dropout_and_per_seed_best_exactly(tmp_path):
    data, cfg = graph(), config()
    whole = run(data, cfg, tmp_path/"whole")
    partial = tmp_path/"partial"
    partial.mkdir()
    for suffix in ("pt", "json"):
        (partial/f"resume_epoch_0001.{suffix}").write_bytes((tmp_path/"whole"/f"resume_epoch_0001.{suffix}").read_bytes())
    continued = run(data, cfg, tmp_path/"continued", partial)
    assert all(row["new_optimizer_updates"] == 2 for row in continued["rows"])
    saved = torch.load(whole["selected_path"], weights_only=False)
    other = torch.load(continued["selected_path"], weights_only=False)
    for section in ("current_state", "best_state"):
        for name, value in saved[section].items():
            torch.testing.assert_close(value, other[section][name], atol=0, rtol=0)
    assert saved["best_epoch"] == other["best_epoch"]
    assert saved["best_ce"] == other["best_ce"]


def test_resume_rejects_metadata_hash_and_missing_history_identity(tmp_path):
    result = run(graph(), config(), tmp_path/"run")
    path = result["selected_path"]
    sidecar = json.loads(path.with_suffix(".json").read_text())
    for key, value in (("seeds", [23, 11]), ("variant", "fixed"), ("path_chunk", 3), ("graph_digest", "changed"), ("code_digest", "changed")):
        metadata = copy.deepcopy(sidecar["metadata"])
        metadata[key] = value
        with pytest.raises(ValueError, match="identity"):
            training._resume_payload(path.parent, metadata)
    payload = torch.load(path, weights_only=False)
    payload["history"] = payload["history"][:-1]
    with pytest.raises(ValueError, match="history"):
        training._validate_resume_state(payload, [11, 23], 3)
    path.write_bytes(path.read_bytes()+b"DEBUG-corrupt")
    with pytest.raises(ValueError, match="hash"):
        training._resume_payload(path.parent, sidecar["metadata"])


def test_preserved_resume_calibration_requires_identical_grouping_identity(tmp_path):
    cfg = config()
    original, copied = tmp_path/"source.json", tmp_path/"copy.json"
    selection = {"packed_runs": 2, "path_chunk": 2}
    write_json(original, {"config_digest": digest(cfg), "phase": "final", "dataset": "DEBUG-Cora", "condition": "unit__learned", "weight_mode": "unit", "variant": "learned", "selection": selection})
    assert training.preserve_resume_calibration(original, copied, cfg, "final", "DEBUG-Cora", "unit__learned") == selection
    assert original.read_bytes() == copied.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        training.preserve_resume_calibration(original, tmp_path/"other.json", cfg, "final", "DEBUG-Cora", "unit__fixed")
    with pytest.raises(FileExistsError):
        training.preserve_resume_calibration(original, copied, cfg, "final", "DEBUG-Cora", "unit__learned")


def test_epoch_telemetry_keeps_undefined_diagnostic_ratios_and_does_not_read_test_labels():
    data, cfg = graph(), config()
    data.x.zero_()
    # An invalid class ID on locked test nodes must never reach train/validation CE.
    data.y[data.test_mask] = 999
    model = training.make_model(data, "unit__learned", [11, 23], cfg, 2)
    rows = training._epoch(model, data, training.make_optimizer(model, cfg, .003), 0, diagnostics=True)
    assert all(row["layer_0_off_nonzero"] == 0 and row["layer_0_matched_delta_relative"] is None for row in rows)
    assert all(row["train_ce"] == pytest.approx(float(torch.log(torch.tensor(3.)))) for row in rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_epoch_ce_gate_gradient_and_update_use_all_nodes():
    data, cfg = graph("cuda"), config()
    model = training.make_model(data, "local_degree__learned", [11, 23], cfg, 2)
    rows = training._epoch(model, data, training.make_optimizer(model, cfg, .003), 0, diagnostics=True)
    assert len(rows) == 2
    assert all(row["cross_ce_gradient_norm"] > 0 and row["cross_parameter_update_norm"] > 0 for row in rows)
    assert model.weights[0].is_cuda and model.theta_cross.is_cuda
