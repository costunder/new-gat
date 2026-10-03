"""Experiment 4.2 complete fresh training, strict selection and immutable artifacts."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

from research.wedge_propagation.node_normalization import common

_FOLDER = Path(__file__).resolve().parents[1] / "research/wedge_propagation/node_normalization"


def _config(profile="debug"):
    return json.loads((_FOLDER / f"config_{profile}.json").read_text(encoding="utf-8"))


def _tuning_rows(config):
    training = config["training"]
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "lr": lr,
            "seed": seed,
            "phase": "tuning",
            "best_epoch": 1,
            "validation_ce": 1.0 + lr,
            "validation_accuracy": 0.5,
            "training_epochs": training["epochs_per_run"],
            "optimizer_updates": training["epochs_per_run"],
            "new_optimizer_updates": training["epochs_per_run"],
        }
        for dataset in config["data"]["datasets"]
        for condition in config["conditions"]
        for lr in training["learning_rate_candidates"]
        for seed in training["tuning_seeds"]
    ]


def _final_rows(config):
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "split": split,
            "ce": 0.9,
            "accuracy": 0.5,
            "num_nodes": config["data"]["expected_shapes"][dataset]["nodes"],
            "num_labeled_nodes": config["data"]["expected_shapes"][dataset][split],
        }
        for dataset in config["data"]["datasets"]
        for condition in config["conditions"]
        for seed in config["training"]["final_seeds"]
        for split in ("train", "validation", "test")
    ]


@pytest.mark.parametrize("profile", ["full", "debug"])
def test_canonical_profile_preserves_declared_complete_run_counts(profile):
    from research.wedge_propagation.node_normalization.study import read_config

    config = read_config(_FOLDER / f"config_{profile}.json", profile)
    tuning = _tuning_rows(config)
    final = _final_rows(config)
    assert len(tuning) == config["training"]["tuning_runs"]
    assert len(final) == 3 * config["training"]["final_runs"]
    if profile == "full":
        assert config["backbone"]["layers"] == 2 and config["backbone"]["hidden_dim"] == 64
        assert config["gate"]["hidden_dim"] == 64
        assert len(tuning) == 135 and len(final) == 225
        assert config["training"]["total_updates"] == 105000
        assert config["data_source"] == "planetoid_public"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("backbone", "layers", 1),
        ("backbone", "hidden_dim", 8),
        ("gate", "hidden_dim", 8),
        ("data", "sampling_ratio", 0.5),
        ("training", "epochs_per_run", 5),
        ("training", "learning_rate_candidates", [0.003]),
        ("training", "final_seeds", [11]),
        ("training", "test_evaluation_during_tuning", True),
    ],
)
def test_full_config_changes_cannot_silently_shrink_or_leak_test(tmp_path, section, key, value):
    from research.wedge_propagation.node_normalization.study import read_config

    config = _config("full")
    config[section][key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(path, "full")


def test_debug_config_cannot_be_claimed_as_full():
    from research.wedge_propagation.node_normalization.study import read_config

    with pytest.raises(ValueError):
        read_config(_FOLDER / "config_debug.json", "full")


def test_all_jobs_keep_independent_seed_axes_and_unique_paths():
    from research.wedge_propagation.node_normalization.study import (
        build_jobs,
        select_learning_rates,
    )

    config = _config()
    tuning = build_jobs(config, "tuning")
    selections = select_learning_rates(config, _tuning_rows(config))
    final = build_jobs(config, "final", selections)
    assert len(tuning) == 30 and len(final) == 15
    assert len({job["job_key"] for job in tuning}) == len(tuning)
    assert len({job["job_key"] for job in final}) == len(final)
    assert all(job["seeds"] == config["training"]["tuning_seeds"] for job in tuning)
    assert all(job["seeds"] == config["training"]["final_seeds"] for job in final)
    assert all(job["phase"] == "tuning" for job in tuning)
    assert all(job["phase"] == "final" for job in final)
    assert all(job["lr"] == min(config["training"]["learning_rate_candidates"]) for job in final)


def test_learning_rate_uses_all_tuning_seeds_and_smaller_lr_for_exact_ties():
    from research.wedge_propagation.node_normalization.study import select_learning_rates

    config = _config()
    rows = _tuning_rows(config)
    target = (config["data"]["datasets"][0], config["conditions"][0])
    for row in rows:
        if (row["dataset"], row["condition"]) == target:
            # First seed favors 0.01. The two-seed mean ties exactly; use smaller LR.
            row["validation_ce"] = (
                0.25
                if row["seed"] == config["training"]["tuning_seeds"][0] and row["lr"] == 0.01
                else 0.75
                if row["lr"] == 0.01
                else 0.5
            )
    selected = select_learning_rates(config, list(reversed(rows)))
    chosen = next(row for row in selected if (row["dataset"], row["condition"]) == target)
    assert chosen["selected_lr"] == 0.003 and chosen["mean_tuning_validation_ce"] == 0.5


@pytest.mark.parametrize(
    "alteration",
    ["missing", "duplicate", "wrong_seed", "nonfinite", "short_training", "short_updates"],
)
def test_learning_rate_selection_rejects_incomplete_or_corrupt_tuning(alteration):
    from research.wedge_propagation.node_normalization.study import select_learning_rates

    config = _config()
    rows = _tuning_rows(config)
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif alteration == "wrong_seed":
        rows[0]["seed"] = config["training"]["final_seeds"][0]
    elif alteration == "nonfinite":
        rows[0]["validation_ce"] = float("nan")
    elif alteration == "short_training":
        rows[0]["training_epochs"] -= 1
    else:
        rows[0]["optimizer_updates"] -= 1
    with pytest.raises(ValueError):
        select_learning_rates(config, rows)


def test_complete_tuning_and_final_graph_condition_seed_split_coverage():
    from research.wedge_propagation.node_normalization.study import verify_coverage

    config = _config()
    result = verify_coverage(config, _tuning_rows(config), _final_rows(config))
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    "alteration",
    [
        "missing_tuning",
        "duplicate_tuning",
        "missing_final",
        "duplicate_final",
        "wrong_seed",
        "wrong_split",
        "wrong_nodes",
        "wrong_labeled_nodes",
        "missing_labeled_count",
    ],
)
def test_coverage_cannot_accept_missing_duplicates_or_wrong_seed_split(alteration):
    from research.wedge_propagation.node_normalization.study import verify_coverage

    config = _config()
    tuning, final = _tuning_rows(config), _final_rows(config)
    if alteration == "missing_tuning":
        tuning.pop()
    elif alteration == "duplicate_tuning":
        tuning.append(copy.deepcopy(tuning[0]))
    elif alteration == "missing_final":
        final.pop()
    elif alteration == "duplicate_final":
        final.append(copy.deepcopy(final[0]))
    elif alteration == "wrong_seed":
        final[0]["seed"] = config["training"]["tuning_seeds"][0]
    elif alteration == "wrong_split":
        final[0]["split"] = "hidden_subset"
    elif alteration == "wrong_nodes":
        final[0]["num_nodes"] -= 1
    elif alteration == "wrong_labeled_nodes":
        final[0]["num_labeled_nodes"] -= 1
    else:
        final[0].pop("num_labeled_nodes")
    with pytest.raises(ValueError):
        verify_coverage(config, tuning, final)


@pytest.mark.parametrize("kind", ["json", "csv", "checkpoint"])
def test_artifacts_refuse_overwriting_existing_files(tmp_path, kind):
    path = tmp_path / f"existing.{kind}"
    before = b"existing-user-artifact"
    path.write_bytes(before)
    with pytest.raises(FileExistsError):
        if kind == "json":
            common.write_json(path, {"value": 1})
        elif kind == "csv":
            common.write_csv(path, [{"value": 1}])
        else:
            common.save_checkpoint(path, {"value": torch.ones(2)})
    assert path.read_bytes() == before


def test_csv_empty_and_json_nonfinite_are_explicit_failures(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        common.write_csv(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError):
        common.write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_cpu_checkpoint_state_does_not_keep_aliases_or_autograd():
    value = torch.arange(4.0, requires_grad=True)
    packed = common.cpu_state({"model": value, "optimizer": [value]})
    assert packed["model"].device.type == "cpu" and not packed["model"].requires_grad
    assert packed["model"].data_ptr() != value.data_ptr()
    assert packed["optimizer"][0].data_ptr() != packed["model"].data_ptr()


def _snapshot(folder, epoch, metadata):
    from research.wedge_propagation.node_normalization.training import _save_snapshot

    path = folder / f"resume_epoch_{epoch:06d}.pt"
    _save_snapshot(path, {"epoch": epoch, "metadata": metadata, "value": torch.tensor([epoch])})
    return path


@pytest.mark.parametrize("partial_name", ["resume_epoch_000002.pt", "selected.pt"])
def test_resume_skips_uncommitted_partial_snapshot_and_preserves_it(tmp_path, capsys, partial_name):
    from research.wedge_propagation.node_normalization.training import _resume_payload

    metadata = {"config_digest": "a" * 64, "graph_digest": "b" * 64, "seeds": [11, 23]}
    older = _snapshot(tmp_path, 1, metadata)
    partial = tmp_path / partial_name
    partial.write_bytes(b"incomplete-uncommitted-payload")
    before = common.file_sha256(partial)
    payload, selected = _resume_payload(tmp_path, metadata)
    assert selected == older and payload["epoch"] == 1
    assert "incomplete snapshot" in capsys.readouterr().out
    assert common.file_sha256(partial) == before


def test_resume_rejects_committed_checkpoint_corruption_and_config_mismatch(tmp_path):
    from research.wedge_propagation.node_normalization.training import _resume_payload

    metadata = {"config_digest": "a" * 64, "graph_digest": "b" * 64, "seeds": [11, 23]}
    path = _snapshot(tmp_path, 1, metadata)
    changed = dict(metadata, config_digest="c" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        _resume_payload(tmp_path, changed)
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="content hash mismatch"):
        _resume_payload(tmp_path, metadata)


def test_debug_training_step_reads_train_and_validation_masks_only(tmp_path, monkeypatch):
    from research.wedge_propagation.classification.data import prepare_datasets
    from research.wedge_propagation.node_normalization import training

    config = _config()
    graphs, _ = prepare_datasets(config, tmp_path / "data", tmp_path / "report", workers="1")
    graph = graphs[config["data"]["datasets"][0]]
    model = training.make_model(graph, "learned_wedge_local_raw", [101, 202], config, path_chunk=16)
    optimizer = training.make_optimizer(model, config, 0.003)
    original = training.classification_metrics
    observed = []

    def tracked(logits, y, mask, **kwargs):
        observed.append(mask)
        return original(logits, y, mask, **kwargs)

    monkeypatch.setattr(training, "classification_metrics", tracked)
    rows = training._epoch(model, graph, optimizer, epoch=0)
    assert len(rows) == 2
    assert len(observed) == 2
    assert observed[0] is graph.train_mask and observed[1] is graph.val_mask
    assert all(mask is not graph.test_mask for mask in observed)
    assert all(row["validation_ce"] > 0 for row in rows)


@pytest.mark.parametrize(
    "reason",
    [
        "existing_results",
        "full_cpu",
        "full_cpu_index",
        "full_non_cuda",
        "full_no_allocation",
        "cuda_unavailable",
    ],
)
def test_cli_preflight_rejects_unsafe_or_unallocated_full_run_before_creating_output(
    tmp_path, monkeypatch, reason
):
    from argparse import Namespace

    from research.wedge_propagation.node_normalization import study

    output = tmp_path / "result"
    args = Namespace(
        config=_FOLDER / "config_full.json",
        profile="full",
        output_dir=str(output),
        device="cuda",
        resume_from=None,
    )
    if reason == "existing_results":
        output.mkdir()
        (output / "existing-user-result.json").write_text("preserve")
    elif reason in ("full_cpu", "full_cpu_index", "full_non_cuda"):
        args.device = {"full_cpu": "cpu", "full_cpu_index": "cpu:0", "full_non_cuda": "meta"}[
            reason
        ]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif reason == "cuda_unavailable":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setattr(study.torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    expected = (
        FileExistsError
        if reason == "existing_results"
        else RuntimeError
        if reason == "cuda_unavailable"
        else ValueError
    )
    with pytest.raises(expected):
        study.run(args)
    if reason == "existing_results":
        assert (output / "existing-user-result.json").read_text() == "preserve"
    else:
        assert not output.exists()


@pytest.fixture(scope="module")
def prepared_intervention_data(tmp_path_factory):
    from research.wedge_propagation.classification.data import prepare_datasets, save_graph
    from research.wedge_propagation.node_normalization import study

    config = _config()
    output = tmp_path_factory.mktemp("node-normalization-fixed-interventions-debug")
    graphs, _ = prepare_datasets(config, output / "data", output, workers="1")
    (output / "graphs").mkdir()
    (output / "manifests").mkdir()
    records = {}
    for name, graph in graphs.items():
        path = output / "graphs" / f"{name}.npz"
        saved = save_graph(graph, path)
        records[name] = {
            "path": str(path),
            "sha256": saved["sha256"],
            "content_digest": study._graph_digest(graph),
            "num_paths": graph.paths.shape[1],
        }
    study._prepare_manifests(output, config, records, workers=2)
    return output, config, graphs, records


def test_fixed_interventions_are_created_once_with_all_rows_and_shared_hashes(
    prepared_intervention_data, monkeypatch
):
    from research.wedge_propagation.node_normalization import study

    output, config, graphs, graph_records = prepared_intervention_data
    count = config["evaluation"]["shuffle_manifests_per_dataset"]
    files = sorted((output / "manifests").rglob("*"))
    before = {str(path): common.file_sha256(path) for path in files if path.is_file()}
    for name, graph in graphs.items():
        first = study._load_manifests(output, name, count)
        second = study._load_manifests(output, name, count)
        assert len(first) == count and [record["index"] for record in first] == list(range(count))
        assert [record["sha256"] for record in first] == [record["sha256"] for record in second]
        for record in first:
            paths = graph.paths.shape[1]
            assert torch.equal(record["permutation"].sort().values, torch.arange(paths))
    with pytest.raises(FileExistsError):
        study._prepare_manifests(output, config, graph_records, workers=2)
    assert before == {str(path): common.file_sha256(path) for path in files if path.is_file()}


@pytest.mark.parametrize(
    "alteration", ["file_bytes", "missing_manifest", "path_escape", "payload_identity"]
)
def test_fixed_intervention_loader_rejects_changed_files_counts_and_identity(
    prepared_intervention_data, tmp_path, alteration
):
    from research.wedge_propagation.node_normalization import study

    base, config, graphs, _ = prepared_intervention_data
    name = next(iter(graphs))
    output = tmp_path / "copied-debug-manifests"
    folder = output / "manifests" / name
    shutil.copytree(base / "manifests" / name, folder)
    index_path = folder / "manifest.json"
    index = common.read_json(index_path)
    file = folder / index[0]["file"]
    if alteration == "file_bytes":
        with file.open("ab") as stream:
            stream.write(b"corruption")
    elif alteration == "missing_manifest":
        index.pop()
    elif alteration == "path_escape":
        index[0]["file"] = "../../../outside.pt"
    else:
        payload = torch.load(file, weights_only=False)
        payload["index"] += 10
        torch.save(payload, file)
        index[0]["file_sha256"] = common.file_sha256(file)
    if alteration != "file_bytes":
        index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|intervention"):
        study._load_manifests(output, name, config["evaluation"]["shuffle_manifests_per_dataset"])


def test_two_cpu_workers_complete_all_debug_tuning_jobs_with_visible_progress(
    prepared_intervention_data, tmp_path, capsys
):
    from research.wedge_propagation.classification.data import save_graph
    from research.wedge_propagation.node_normalization import study

    _, config, graphs, _ = prepared_intervention_data
    output = tmp_path / "actual-two-worker-debug"
    output.mkdir()
    for name in ("plans", "graphs", "calibration", "workers", "jobs", "manifests"):
        (output / name).mkdir()
    common.write_json(output / "config.json", config)
    source = common.source_manifest()
    records = {}
    for name, graph in graphs.items():
        path = output / "graphs" / f"{name}.npz"
        saved = save_graph(graph, path)
        records[name] = {
            "path": str(path),
            "sha256": saved["sha256"],
            "content_digest": study._graph_digest(graph),
            "num_paths": graph.paths.shape[1],
        }
    result = study._dispatch(
        output,
        config,
        source,
        records,
        study.build_jobs(config, "tuning"),
        ["cpu", "cpu"],
        "tuning",
    )
    rows = result["selection_rows"]
    assert len(rows) == config["training"]["tuning_runs"]
    assert len(study.select_learning_rates(config, rows)) == len(graphs) * len(config["conditions"])
    assert (
        sum(row["new_optimizer_updates"] for row in rows)
        == len(rows) * config["training"]["epochs_per_run"]
    )
    assert not result["metric_rows"] and not result["intervention_rows"]
    assert not (output / "test_evaluation_unlocked.json").exists()
    for index in range(2):
        completion = common.read_json(output / "workers" / f"tuning-{index}" / "completion.json")
        assert completion["completed"] and completion["jobs"] > 0
    progress = capsys.readouterr().out
    assert "[epoch]" in progress and "worker=0" in progress and "worker=1" in progress
    assert common.source_manifest()["code_digest"] == source["code_digest"]
    assert all(
        common.file_sha256(record["path"]) == record["sha256"] for record in records.values()
    )


def _complete_frozen_row_keys(config):
    """Protocol identities only; the real DEBUG integration supplies actual metrics."""
    evaluated = {"intervention_rows": [], "scale_rows": [], "gate_rows": []}
    count = config["evaluation"]["shuffle_manifests_per_dataset"]
    variants = [
        (treatment, target, index)
        for target in config["evaluation"]["intervention_scopes"]
        for treatment in (
            "c_identity",
            "c_identity_norm_matched",
            "c_position_shuffle",
            "c_position_shuffle_norm_matched",
            "second_branch_remove",
        )
        for index in (range(count) if "shuffle" in treatment else [-1])
    ]
    for dataset in config["data"]["datasets"]:
        for condition in config["conditions"]:
            for seed in config["training"]["final_seeds"]:
                prefix = {"dataset": dataset, "condition": condition, "seed": seed}
                evaluated["gate_rows"].extend(
                    dict(
                        prefix,
                        layer=layer,
                        intervention="original",
                        target="none",
                        manifest_index=-1,
                    )
                    for layer in (0, 1)
                )
                if condition.startswith("learned_wedge"):
                    for intervention, target, index in variants:
                        identity = dict(
                            prefix, intervention=intervention, target=target, manifest_index=index
                        )
                        evaluated["intervention_rows"].extend(
                            dict(identity, split=split) for split in ("train", "validation", "test")
                        )
                        evaluated["gate_rows"].extend(
                            dict(identity, layer=layer) for layer in (0, 1)
                        )
    return evaluated


def test_complete_frozen_protocol_requires_every_amplitude_layer_manifest_and_split():
    from research.wedge_propagation.node_normalization.study import verify_frozen_coverage

    config = _config()
    result = verify_frozen_coverage(config, _complete_frozen_row_keys(config))
    assert result == {"intervention_rows": 1512, "scale_rows": 0, "gate_rows": 1068}


@pytest.mark.parametrize("table", ["intervention_rows", "gate_rows"])
@pytest.mark.parametrize("alteration", ["missing", "duplicate", "nonfinite", "wrong_seed"])
def test_frozen_protocol_rejects_incomplete_duplicated_or_invalid_table(table, alteration):
    from research.wedge_propagation.node_normalization.study import verify_frozen_coverage

    config = _config()
    evaluated = _complete_frozen_row_keys(config)
    rows = evaluated[table]
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif alteration == "nonfinite":
        rows[0]["additional_audit_metric"] = float("nan")
    else:
        rows[0]["seed"] = config["training"]["tuning_seeds"][0]
    with pytest.raises(ValueError, match="frozen"):
        verify_frozen_coverage(config, evaluated)


def test_frozen_protocol_rejects_uncontracted_scale_table():
    from research.wedge_propagation.node_normalization.study import verify_frozen_coverage

    config = _config()
    evaluated = _complete_frozen_row_keys(config)
    evaluated["scale_rows"] = [{"scope": "not_part_of_this_experiment"}]
    with pytest.raises(ValueError, match="scale rows"):
        verify_frozen_coverage(config, evaluated)


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "short_budget", "wrong_lr"])
def test_test_lock_requires_every_final_checkpoint_full_training_budget(alteration):
    from research.wedge_propagation.node_normalization.study import (
        _validate_final_training,
        build_jobs,
        select_learning_rates,
    )

    config = _config()
    selections = select_learning_rates(config, _tuning_rows(config))
    jobs = build_jobs(config, "final", selections)
    rows = [
        {
            "dataset": job["dataset"],
            "condition": job["condition"],
            "seed": seed,
            "lr": job["lr"],
            "phase": "final",
            "training_epochs": config["training"]["epochs_per_run"],
            "optimizer_updates": config["training"]["epochs_per_run"],
        }
        for job in jobs
        for seed in job["seeds"]
    ]
    _validate_final_training(config, rows, jobs)
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif alteration == "short_budget":
        rows[0]["optimizer_updates"] -= 1
    else:
        rows[0]["lr"] = 0.5
    with pytest.raises(ValueError, match="final"):
        _validate_final_training(config, rows, jobs)


def test_new_local_model_trains_full_debug_budget_and_resumes_without_updates(
    prepared_intervention_data,
    tmp_path,
):
    from research.wedge_propagation.node_normalization import training

    _, config, graphs, records = prepared_intervention_data
    graph = graphs[config["data"]["datasets"][0]]
    seeds = config["training"]["tuning_seeds"]
    condition = "learned_wedge_local_rms"
    rate = config["training"]["learning_rate_candidates"][0]
    source = common.source_manifest()
    calibration = training.choose_packing(graph, condition, seeds, rate, config, "tuning")
    # A pack uses the physically measured simultaneous seed count, without changing budgets.
    seeds = seeds[: calibration["packed_runs"]]
    first_dir = tmp_path / "fresh-local-debug-training"
    before = {record["path"]: record["sha256"] for record in records.values()}
    first = training.train_pack(
        graph,
        condition,
        seeds,
        rate,
        config,
        first_dir,
        source,
        records[graph.name]["content_digest"],
        "tuning",
        calibration=calibration,
    )
    epochs = config["training"]["epochs_per_run"]
    assert all(row["training_epochs"] == epochs for row in first["rows"])
    assert all(row["new_optimizer_updates"] == epochs for row in first["rows"])
    assert all(row["condition"] == condition for row in first["rows"])
    original_files = {
        path.relative_to(first_dir).as_posix(): common.file_sha256(path)
        for path in first_dir.rglob("*")
        if path.is_file()
    }
    second = training.train_pack(
        graph,
        condition,
        seeds,
        rate,
        config,
        tmp_path / "resumed-local-debug-training",
        source,
        records[graph.name]["content_digest"],
        "tuning",
        resume_dir=first_dir,
        calibration=calibration,
    )
    assert all(row["new_optimizer_updates"] == 0 for row in second["rows"])
    assert first["rows"] == [dict(row, new_optimizer_updates=epochs) for row in second["rows"]]
    for name, value in first["model"].state_dict().items():
        assert torch.equal(value, second["model"].state_dict()[name])
    assert original_files == {
        path.relative_to(first_dir).as_posix(): common.file_sha256(path)
        for path in first_dir.rglob("*")
        if path.is_file()
    }
    assert all(common.file_sha256(path) == sha for path, sha in before.items())


@pytest.mark.parametrize("case", ["cpu", "cpu_index", "no_allocation", "cuda_unavailable"])
def test_direct_worker_plan_cannot_bypass_full_cuda_contract(tmp_path, monkeypatch, case):
    from research.wedge_propagation.node_normalization import study

    output = tmp_path / "direct-worker-plan"
    output.mkdir()
    common.write_json(output / "config.json", _config("full"))
    device = "cpu:0" if case == "cpu_index" else "cpu" if case == "cpu" else "cuda:0"
    if case == "no_allocation":
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    if case == "cuda_unavailable":
        monkeypatch.setattr(study.torch.cuda, "is_available", lambda: False)
    plan = tmp_path / "unsafe-full-worker.json"
    common.write_json(
        plan,
        {
            "output_dir": str(output),
            "profile": "full",
            "device": device,
            "source": common.source_manifest(),
        },
    )
    with pytest.raises(RuntimeError if case == "cuda_unavailable" else ValueError, match="CUDA"):
        study._worker(plan)
    assert list(output.iterdir()) == [output / "config.json"]
