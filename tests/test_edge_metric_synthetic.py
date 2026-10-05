"""Experiment B contracts, independent scalar axes, exact oracles and updates."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.edge_metric_relations.synthetic.data import (data_manifest, make_specs, pack_cases, prepare_cases)
from research.edge_metric_relations.synthetic.model import (BASELINES, CONDITIONS, TRAINABLE, RawStudent,
    baseline_predict, fit_baseline, graph_draw_errors, normalized_mse, target_message)
from research.edge_metric_relations.synthetic.study import (FOLDER, budget, checkpoint_record, load_resume,
    paired_comparisons, read_config, train_epoch, train_job, validation_scores,
    all_student_jobs, allocate_student_jobs, _shared_payload, _load_shared, worker_run)
from research.edge_metric_relations.synthetic.verify import dense_diagonal, dense_analytic_teacher, run_math_checks
from research.wedge_propagation.learned.data import make_specs as original_specs


@pytest.fixture(scope="module")
def config():
    return read_config(FOLDER / "config_debug.json", "debug")


@pytest.fixture(scope="module")
def cases(config):
    return prepare_cases(config, 2)


def test_full_contract_budget_and_membership():
    config = read_config(FOLDER / "config_full.json", "full")
    specs = make_specs(config)
    from collections import Counter
    assert len(specs) == 531
    assert Counter(spec.split for spec in specs) == {"train": 240, "validation": 60, "id": 120,
        "size_ood": 90, "family_ood": 12, "family_size_ood": 9}
    value = budget(config, len(specs))
    assert value["learned_runs"] == 240
    assert value["seed_optimizer_updates"] == 120000
    assert value["packed_optimizer_calls"] == 24000
    assert value["per_graph_rows"] == 152928
    assert value["per_realization_rows"] == 2446848
    assert config["features"] == 16 and config["hidden"] == 64
    assert config["epochs"] == 500 and config["model_seeds"] == [11, 23, 37, 53, 71]


def test_debug_contract_all_condition_axes(config, cases):
    value = budget(config, len(cases))
    assert len(cases) == 30
    assert value["learned_runs"] == 96 and value["seed_optimizer_updates"] == 288
    assert value["per_graph_rows"] == 4320 and value["per_realization_rows"] == 17280
    assert config["hidden"] == 64 and len(config["conditions"]) == 6


@pytest.mark.parametrize("profile,device,platform,mask,should_pass", [
    ("full", "cuda", "win32", "0", False),
    ("full", "cuda", "linux", "", False),
    ("full", "cpu", "linux", "0", False),
    ("full", "cuda", "linux", "0,2", True),
    ("debug", "cpu", "win32", "", True),
])
def test_full_server_allocation_guard(profile, device, platform, mask, should_pass, monkeypatch):
    import research.edge_metric_relations.synthetic.study as module
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    if should_pass:
        module.require_full_server(profile, torch.device(device))
    else:
        with pytest.raises(ValueError):
            module.require_full_server(profile, torch.device(device))


@pytest.mark.parametrize("key,value", [("epochs", 2), ("hidden", 16), ("features", 1), ("conditions", ["F2"])])
def test_no_silent_contract_reduction(config, tmp_path, key, value):
    changed = copy.deepcopy(config); changed[key] = value
    path = tmp_path / "changed.json"; path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="declared profile"):
        read_config(path, "debug")


def test_new_features_keep_original_graph_stream(config):
    original = original_specs(config)
    current = make_specs(config)
    assert [spec.graph_spec.graph_seed for spec in original] == [spec.graph_spec.graph_seed for spec in current]
    assert [spec.graph_id for spec in original] == [spec.graph_id for spec in current]
    assert all(a.graph_spec.feature_seed != b.graph_spec.feature_seed for a, b in zip(original, current))


def test_manifest_complete_no_graphs_or_draws_dropped(config, cases):
    manifest = data_manifest(cases, config)
    assert manifest["graph_count"] == 30 and manifest["scalar_inputs"] == 120
    assert manifest["independent_scalar_realizations"] == 4 and manifest["feature_channels"] == 1
    assert len({row["graph_sha256"] for row in manifest["rows"]}) == len(cases)
    assert len({row["feature_sha256"] for row in manifest["rows"]}) == len(cases)
    repeated = prepare_cases(config, 1)
    assert data_manifest(repeated, config) == manifest


@pytest.mark.parametrize("recipe", ["unit", "local_degree"])
@pytest.mark.parametrize("target", ["diagonal", "diagonal_squared", "analytic_pair"])
def test_raw_teacher_exact_dense_and_batching(cases, recipe, target):
    batch = pack_cases(cases[:3], recipe, "cpu", torch.float64)
    actual = target_message(batch.geometry, batch.x, target, pair_chunk=7)
    ld = dense_diagonal(batch.geometry)
    expected = ld @ batch.x if target == "diagonal" else ld @ (ld @ batch.x)
    if target == "analytic_pair":
        expected = dense_analytic_teacher(batch.geometry, batch.x)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    individually = [target_message(pack_cases([case], recipe, "cpu", torch.float64).geometry,
                                   case.graph.features.T[None, ..., None], target, 3) for case in cases[:3]]
    torch.testing.assert_close(actual, torch.cat(individually, -2), atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_parameter_and_scalar_seed_axis_contract(cases, condition):
    seeds = (11, 23) if condition in TRAINABLE else (-1,)
    model = RawStudent(condition, seeds).double()
    batch = pack_cases(cases[:2], "local_degree", "cpu", torch.float64)
    predicted, _ = model(batch, pair_chunk=7)
    assert predicted.shape == (len(seeds), 4, batch.geometry.n, 1)
    assert bool(sum(value.numel() for value in model.parameters())) == (condition in TRAINABLE)
    separate_draws = []
    for draw in range(4):
        from dataclasses import replace
        scalar = replace(batch, x=batch.x[:, draw:draw + 1])
        separate_draws.append(model(scalar, pair_chunk=3)[0])
    torch.testing.assert_close(predicted, torch.cat(separate_draws, 1), atol=1e-11, rtol=1e-11)


def test_seed_pack_equals_independent_models(cases):
    batch = pack_cases(cases[:2], "unit", "cpu", torch.float64)
    packed = RawStudent("F2", (11, 23)).double()
    with torch.no_grad():
        packed.gate.diagonal_gate.w2[:] = .03
        packed.gate.pair_gate.w2[:] = .02
    outputs = []
    for i, seed in enumerate((11, 23)):
        single = RawStudent("F2", (seed,)).double()
        single.load_state_dict({name: value[i:i + 1] for name, value in packed.state_dict().items()})
        outputs.append(single(batch, pair_chunk=5)[0])
    torch.testing.assert_close(packed(batch, pair_chunk=7)[0], torch.cat(outputs), atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("condition", TRAINABLE)
def test_exact_full_epoch_gradient_accumulation_and_adam(cases, condition):
    graphs = cases[:4]
    whole = [pack_cases(graphs, "local_degree", "cpu", torch.float64)]
    chunked = [pack_cases(graphs[i:i + 2], "local_degree", "cpu", torch.float64) for i in (0, 2)]
    targets_a = [target_message(batch.geometry, batch.x, "analytic_pair", 7) for batch in whole]
    targets_b = [target_message(batch.geometry, batch.x, "analytic_pair", 7) for batch in chunked]
    a, b = RawStudent(condition, (11, 23)).double(), RawStudent(condition, (11, 23)).double()
    optim_a, optim_b = torch.optim.Adam(a.parameters(), lr=.003), torch.optim.Adam(b.parameters(), lr=.003)
    for _ in range(2):
        stats_a = train_epoch(a, whole, targets_a, optim_a, 4, 1e-8, 7)
        stats_b = train_epoch(b, chunked, targets_b, optim_b, 4, 1e-8, 5)
        np.testing.assert_allclose(stats_a, stats_b, atol=1e-10, rtol=1e-10)
        for va, vb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(va, vb, atol=1e-10, rtol=1e-10)
    assert all(row[1] > 0 and row[3] > 0 for row in stats_a)


@pytest.mark.parametrize("recipe", ["unit", "local_degree"])
@pytest.mark.parametrize("target,condition", [("diagonal", "same_teacher_linear"),
    ("diagonal_squared", "same_teacher_squared"), ("diagonal", "same_teacher_polynomial"),
    ("diagonal_squared", "same_teacher_polynomial")])
def test_same_teacher_raw_ls_exact_positive_control(cases, recipe, target, condition):
    train = [pack_cases(cases[:4], recipe, "cpu", torch.float64)]
    targets = [target_message(batch.geometry, batch.x, target, 13) for batch in train]
    coefficients, record = fit_baseline(train, targets, condition)
    assert record["fit_split"] == "train" and record["optimizer_updates"] == 0
    held = pack_cases(cases[8:11], recipe, "cpu", torch.float64)
    prediction = baseline_predict(held, condition, coefficients)
    expected = target_message(held.geometry, held.x, target, 13)
    assert float(normalized_mse(prediction, expected, held.geometry)) < 1e-18


def test_loss_equal_graph_not_node_or_seed_mean(cases):
    batch = pack_cases([cases[0], cases[3]], "unit", "cpu", torch.float64)
    target = torch.ones_like(batch.x)
    index = batch.geometry.node_graph
    errors = torch.tensor([1., 3.], dtype=torch.float64)[index]
    prediction = target.expand(2, -1, -1, -1) + errors[None, None, :, None]
    loss = normalized_mse(prediction, target, batch.geometry)
    torch.testing.assert_close(loss, torch.tensor([5., 5.], dtype=torch.float64))
    zero = graph_draw_errors(torch.zeros_like(target), torch.zeros_like(target), batch.geometry)
    assert bool(zero["zero_target"].all()) and not bool(zero["nmse"].any())


def test_resume_one_job_identical_no_new_updates(config, cases, tmp_path):
    train = [pack_cases(cases[:3], "unit", "cpu", torch.float64)]
    validation = [pack_cases(cases[6:9], "unit", "cpu", torch.float64)]
    tr = [target_message(batch.geometry, batch.x, "analytic_pair", 7) for batch in train]
    va = [target_message(batch.geometry, batch.x, "analytic_pair", 7) for batch in validation]
    source = {"code_digest": "fixture-source"}
    job = "unit--analytic_pair--unit--F2"
    first = tmp_path / "first"; first.mkdir()
    model = RawStudent("F2", config["model_seeds"]).double()
    history, epochs, updates, _ = train_job(model, train, tr, validation, va, config, source, "fixture-data",
                                         job, first, 3, 7, None)
    assert updates == 6 and len(history) == 6
    original = copy.deepcopy(model.state_dict())
    resume = load_resume(first, job, config, source, "fixture-data", "cpu")
    second = tmp_path / "second"; second.mkdir()
    other = RawStudent("F2", config["model_seeds"]).double()
    h2, e2, u2, _ = train_job(other, train, tr, validation, va, config, source, "fixture-data",
                             job, second, 3, 7, resume)
    assert h2 == history and e2 == epochs and u2 == 0
    for name, value in original.items():
        torch.testing.assert_close(value, other.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="data_manifest_digest"):
        load_resume(first, job, config, source, "wrong-data", "cpu")
    with pytest.raises(ValueError, match="resume packing differs"):
        train_job(other, train, tr, validation, va, config, source, "fixture-data",
                  job, second, 2, 5, resume)


def test_preflight_math_and_real_debug_update():
    record = run_math_checks("cpu")
    assert record["passed"] and record["dense_teacher_checks"] == 30
    assert record["gradient_abs_max"] > 0 and record["optimizer_change_abs_max"] > 0
    assert record["debug_seed_optimizer_updates"] == 2 and not record["full_scientific_training"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_preflight_and_scalar_recovery(cases):
    record = run_math_checks("cuda")
    assert record["passed"]
    batch = pack_cases(cases[:3], "local_degree", "cuda", torch.float64)
    model = RawStudent("F2", (11, 23)).to(device="cuda", dtype=torch.float64)
    target = target_message(batch.geometry, batch.x, "analytic_pair", 7)
    loss = normalized_mse(model(batch, pair_chunk=7)[0], target, batch.geometry)
    loss.sum().backward()
    assert all(value.grad is not None for value in model.parameters())


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
def test_multigpu_job_partition_exact_and_balanced(count):
    shards = allocate_student_jobs(count)
    assert len(shards) == count and all(shards)
    flattened = [job for shard in shards for job in shard]
    assert len(flattened) == 72 and len(set(flattened)) == 72
    assert set(flattened) == set(all_student_jobs())
    learned = [sum(job[3] in TRAINABLE for job in shard) for shard in shards]
    assert max(learned) - min(learned) <= 1


def test_shared_cache_weights_only_complete_and_hashed(config, cases, tmp_path):
    import hashlib
    from research.edge_metric_relations.common import digest, save_checkpoint
    refs = {case.graph_id: {("unit", "diagonal"): case.graph.features.T[None, ..., None]} for case in cases}
    path = tmp_path / "worker_inputs.pt"
    save_checkpoint(path, _shared_payload(cases, refs))
    manifest = {"source_digest": "source", "config_digest": digest(config),
        "data_manifest_digest": digest(data_manifest(cases, config)),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "worker_input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded, actual_refs, actual_digest = _load_shared(tmp_path, config, {"code_digest": "source"})
    assert data_manifest(loaded, config) == data_manifest(cases, config)
    assert actual_digest == manifest["data_manifest_digest"]
    assert set(actual_refs) == set(refs)
    for key in refs:
        torch.testing.assert_close(actual_refs[key]["unit", "diagonal"], refs[key]["unit", "diagonal"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="source/config"):
        _load_shared(tmp_path, config, {"code_digest": "changed"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_actual_single_cuda_worker_declared_debug_job(config, cases, tmp_path, monkeypatch):
    import hashlib
    from argparse import Namespace
    from research.edge_metric_relations.common import digest, save_checkpoint
    import research.edge_metric_relations.synthetic.study as module
    source = {"code_digest": "explicit-DEBUG-worker-fixture", "sha256": {}}
    monkeypatch.setattr(module, "source_manifest", lambda: source)
    monkeypatch.setattr(module, "assert_source_unchanged", lambda record: None)
    shared = tmp_path / "shared"; shared.mkdir()
    refs = {}
    for case in cases:
        refs[case.graph_id] = {}
        for recipe in ("unit", "local_degree"):
            batch = pack_cases([case], recipe, "cpu", torch.float64)
            for target in ("diagonal", "diagonal_squared", "analytic_pair"):
                refs[case.graph_id][recipe, target] = target_message(batch.geometry, batch.x, target, 128)
    cache = shared / "worker_inputs.pt"; save_checkpoint(cache, _shared_payload(cases, refs))
    manifest = {"source_digest": source["code_digest"], "config_digest": digest(config),
        "data_manifest_digest": digest(data_manifest(cases, config)),
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}
    (shared / "worker_input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    plan = shared / "plan.json"
    jobs = [["unit", "unit", "analytic_pair", "F2"]]
    plan.write_text(json.dumps({"ordinal": 0, "jobs": jobs, "source_digest": source["code_digest"]}), encoding="utf-8")
    output = tmp_path / "worker"; output.mkdir()
    args = Namespace(config=None, profile="debug", device="cuda", worker_plan=plan,
        shared_prepared_dir=shared, batch_size="6", pair_chunk="128", resume_from=None)
    worker_run(args, output)
    completion = json.loads((output / "worker_completion.json").read_text())
    results = json.loads((output / "worker_results.json").read_text())
    assert completion["completed"] and completion["jobs"] == jobs
    assert completion["graph_rows"] == 60 and completion["draw_rows"] == 240
    assert results["new_updates"] == 6 and len(results["history"]) == 6


def test_mig_packing_measures_small_candidates_and_excludes_headroom(config, cases, monkeypatch):
    import research.edge_metric_relations.synthetic.study as module
    seen = []
    monkeypatch.setattr(module, "_calibration_window", lambda *args: None)
    monkeypatch.setattr(module, "_resident_batches", lambda *args: {})
    def measured(cases, refs, config, recipe, size, pair_chunk, device, dtype, resident_cases,
                 hardware_profile, prepared):
        seen.append((size, pair_chunk, recipe))
        return {"physical_graph_batch": size, "pair_chunk": pair_chunk, "recipe": recipe,
                "seconds_per_epoch": 1/size, "peak_vram_bytes": 10,
                "status": "measured" if size <= 4 else "measured_headroom_exceeded"}
    monkeypatch.setattr(module, "_measure_packing", measured)
    size, chunk, trials = module.choose_packing(cases[:6], {}, config, torch.device("cpu"), torch.float64,
                                               "auto", "auto", cases, "a100-mig-10gb")
    assert size == 4
    assert {2, 4, 6} <= {row[0] for row in seen}
    assert {128, 512, 1024, 4096} <= {row[1] for row in seen}
    assert any(row["status"] == "measured_headroom_exceeded" for row in trials)


def test_mig_actual_measurement_includes_all_static_graphs_and_six_targets(config, cases):
    import research.edge_metric_relations.synthetic.study as module
    refs = {}
    for case in cases:
        refs[case.graph_id] = {}
        for recipe in ("unit", "local_degree"):
            batch = pack_cases([case], recipe, "cpu", torch.float64)
            for target in ("diagonal", "diagonal_squared", "analytic_pair"):
                refs[case.graph_id][recipe, target] = target_message(batch.geometry, batch.x, target, 128)
    train = [case for case in cases if case.split == "train"]
    row = module._measure_packing(train, refs, config, "unit", 4, 128, torch.device("cpu"),
                                  torch.float64, cases, "a100-mig-10gb")
    assert row["status"] == "measured"
    assert row["static_graphs_resident"] == len(cases)
    assert row["target_cache_recipe_target_pairs"] == 6
    assert row["graphs"] == len(train) and row["realizations"] == config["features"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_mig_static_device_cache_retains_every_graph_draw_and_recipe(config, cases):
    import research.edge_metric_relations.synthetic.study as module
    for recipe in ("unit", "local_degree"):
        resident = module._resident_batches(cases, recipe, 4, torch.device("cuda"), torch.float64,
                                            "a100-mig-10gb")
        assert sum(batch.num_graphs for batches in resident.values() for batch in batches) == len(cases)
        for split, batches in resident.items():
            expected = module.cache_batches([case for case in cases if case.split == split], recipe, 4,
                                              torch.device("cpu"), torch.float64)
            for actual, cpu in zip(batches, expected, strict=True):
                assert actual.x.shape[1] == config["features"]
                torch.testing.assert_close(actual.x.cpu(), cpu.x, atol=0, rtol=0)
                torch.testing.assert_close(actual.geometry.c0.cpu(), cpu.geometry.c0, atol=0, rtol=0)
                assert actual.geometry.num_pairs == cpu.geometry.num_pairs
        del resident
