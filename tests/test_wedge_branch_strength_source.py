"""Frozen Experiment 4 source guards using a genuinely completed DEBUG run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from research.wedge_propagation.branch_strength import source
from research.wedge_propagation.classification.common import file_sha256, source_manifest


@pytest.mark.parametrize(
    "recorded",
    [
        "graphs/Cora.npz",
        "/home/research/results/old/graphs/Cora.npz",
        r"D:\research\old\graphs\Cora.npz",
    ],
)
def test_graph_cache_rebases_only_exact_canonical_destination(recorded):
    assert source._graph_path(recorded, "Cora") == Path("graphs/Cora.npz")


@pytest.mark.parametrize(
    "recorded",
    [
        "../graphs/Cora.npz",
        "copies/graphs/Cora.npz",
        "/home/research/graphs/../graphs/Cora.npz",
        r"C:\research\graphs\..\graphs\Cora.npz",
        "graphs/CiteSeer.npz",
        "/home/research/outside/Cora.npz",
    ],
)
def test_graph_cache_path_traversal_or_wrong_dataset_is_rejected(recorded):
    with pytest.raises(ValueError):
        source._graph_path(recorded, "Cora")


def test_reader_detects_file_change_during_loading_and_preserves_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    reader = source._Reader(tmp_path)
    reader.path("artifact.json")
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed while reading"):
        reader.path("artifact.json")
    assert path.read_text() == '{"changed": true}'


def test_reader_rejects_outside_link_before_reading(tmp_path, monkeypatch):
    declared = tmp_path / "declared"
    declared.mkdir()
    reader = source._Reader(declared)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    link = declared / "linked.json"
    resolve = Path.resolve

    def linked_resolve(path, *args, **kwargs):
        return outside if path == link else resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", linked_resolve)
    with pytest.raises(ValueError, match="path escape"):
        reader.path("linked.json")
    assert outside.read_text() == "{}"


@pytest.fixture(scope="module")
def completed_source(wedge_completed_classification_debug_run):
    return source.load_source(wedge_completed_classification_debug_run, "debug")


def _clone(original, path):
    path.mkdir()
    for relative in original.hashes:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original.directory / relative, target)
    return path


def _change_json(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_all_original_conditions_and_final_states_load_without_optimizer(
    completed_source, monkeypatch
):
    original = completed_source
    before = dict(original.hashes)

    def forbidden_optimizer(*args, **kwargs):
        pytest.fail("frozen source must never construct an optimizer")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden_optimizer)
    loaded = source.load_source(original.directory, "debug")
    assert sum(len(pack.seeds) for pack in loaded.packs) == 48
    assert sum(len(pack.seeds) for pack in loaded.learned_packs) == 12
    assert len(loaded.original_rows) == 144
    assert {pack.condition for pack in loaded.packs} == set(loaded.config["conditions"])
    assert set(loaded.graphs) == set(loaded.config["data"]["datasets"])
    for pack in loaded.packs:
        assert len(pack.best_epoch) == len(pack.best_ce) == len(pack.best_acc) == len(pack.seeds)
        assert all(
            value.device.type == "cpu" and value.shape[0] == len(pack.seeds)
            for value in pack.best_state.values()
        )
    source.assert_unchanged(loaded)
    assert loaded.hashes == before


def test_copied_source_rebases_server_paths_inside_copy_and_keeps_hashes(
    completed_source, tmp_path
):
    original = completed_source
    copied = _clone(original, tmp_path / "copied-complete-debug")
    loaded = source.load_source(copied, "debug")
    assert loaded.hashes == original.hashes
    assert loaded.directory == copied.resolve()
    assert all(pack.checkpoint.is_relative_to(copied) for pack in loaded.packs)
    for name, graph in loaded.graphs.items():
        assert Path(loaded.graph_records[name]["path"]) == copied / "graphs" / f"{name}.npz"
        assert torch.equal(graph.paths, original.graphs[name].paths)
        assert torch.equal(graph.x, original.graphs[name].x)
        assert torch.equal(graph.train_mask, original.graphs[name].train_mask)


def test_debug_source_cannot_be_used_as_full_experiment(completed_source):
    with pytest.raises(ValueError):
        source.load_source(completed_source.directory)


@pytest.mark.parametrize(
    "alteration",
    [
        "failure_marker",
        "missing_completion",
        "incomplete",
        "changed_config",
        "changed_code",
        "changed_coverage",
        "changed_lr_selection",
        "missing_final_seed",
        "graph_path_escape",
        "graph_corruption",
        "selected_corruption",
        "selected_metadata",
        "manifest_corruption",
        "missing_manifest",
    ],
)
def test_source_rejects_incomplete_changed_or_corrupt_artifacts(
    completed_source, tmp_path, alteration
):
    folder = _clone(completed_source, tmp_path / "mutated-debug")
    pack = completed_source.packs[0]
    relative = pack.checkpoint.relative_to(completed_source.directory)
    name = next(iter(completed_source.graphs))
    if alteration == "failure_marker":
        (folder / "failure.json").write_text('{"failed": true}')
    elif alteration == "missing_completion":
        (folder / "completion.json").unlink()
    elif alteration == "incomplete":
        _change_json(folder / "completion.json", lambda value: value.update(completed=False))
    elif alteration == "changed_config":
        _change_json(folder / "config.json", lambda value: value["backbone"].update(hidden_dim=1))
    elif alteration == "changed_code":
        _change_json(folder / "source.json", lambda value: value.update(code_digest="0" * 64))
    elif alteration == "changed_coverage":
        _change_json(folder / "coverage.json", lambda value: value.update(final_runs=1))
    elif alteration == "changed_lr_selection":
        _change_json(
            folder / "learning_rate_selection.json", lambda value: value[0].update(selected_lr=0.02)
        )
    elif alteration == "missing_final_seed":
        path = folder / "final_validation_selection.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
    elif alteration == "graph_path_escape":
        _change_json(
            folder / "graph_cache.json", lambda value: value[name].update(path="../outside.npz")
        )
    elif alteration == "graph_corruption":
        with (folder / "graphs" / f"{name}.npz").open("ab") as stream:
            stream.write(b"corrupted")
    elif alteration == "selected_corruption":
        with (folder / relative).open("ab") as stream:
            stream.write(b"corrupted")
    elif alteration == "selected_metadata":
        _change_json(
            (folder / relative).with_suffix(".json"),
            lambda value: value["metadata"].update(seeds=[999]),
        )
    elif alteration == "manifest_corruption":
        with (folder / "manifests" / name / "manifest-00.pt").open("ab") as stream:
            stream.write(b"corrupted")
    else:
        _change_json(folder / "manifests" / name / "manifest.json", lambda value: value.pop())
    before = source_manifest()["code_digest"]
    with pytest.raises(ValueError):
        source.load_source(folder, "debug")
    assert source_manifest()["code_digest"] == before
    source.assert_unchanged(completed_source)


@pytest.mark.parametrize(
    "alteration",
    [
        "missing_key",
        "wrong_shape",
        "wrong_dtype",
        "nonfinite",
        "wrong_best_epoch",
        "missing_history",
    ],
)
def test_checkpoint_internal_state_cannot_hide_behind_updated_sidecar_hash(
    completed_source, tmp_path, alteration
):
    folder = _clone(completed_source, tmp_path / "mutated-model-state")
    pack = completed_source.learned_packs[0]
    relative = pack.checkpoint.relative_to(completed_source.directory)
    checkpoint = folder / relative
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if alteration == "missing_key":
        payload["best_state"].pop(next(iter(payload["best_state"])))
    elif alteration == "wrong_shape":
        key = next(
            key
            for key, value in payload["best_state"].items()
            if value.is_floating_point() and value.ndim > 1
        )
        payload["best_state"][key] = payload["best_state"][key][..., :1]
    elif alteration == "wrong_dtype":
        key = next(key for key, value in payload["best_state"].items() if value.is_floating_point())
        payload["best_state"][key] = payload["best_state"][key].double()
    elif alteration == "nonfinite":
        key = next(key for key, value in payload["best_state"].items() if value.is_floating_point())
        payload["best_state"][key].reshape(-1)[0] = float("nan")
    elif alteration == "wrong_best_epoch":
        payload["best_epoch"][0] = 999
    else:
        payload["history"].pop()
    torch.save(payload, checkpoint)
    changed_hash = file_sha256(checkpoint)
    _change_json(checkpoint.with_suffix(".json"), lambda value: value.update(sha256=changed_hash))
    _change_json(
        checkpoint.parent / "completion.json",
        lambda value: value.update(selected_sha256=changed_hash),
    )
    with pytest.raises(ValueError):
        source.load_source(folder, "debug")


def test_post_diagnostic_hash_guard_rejects_changed_source_bytes(completed_source, tmp_path):
    folder = _clone(completed_source, tmp_path / "post-diagnostic-change")
    loaded = source.load_source(folder, "debug")
    target = folder / "metrics.csv"
    with target.open("ab") as stream:
        stream.write(b"changed-after-loading")
    with pytest.raises(ValueError, match="source artifact changed"):
        source.assert_unchanged(loaded)


def test_post_diagnostic_guard_rejects_extra_final_checkpoint(completed_source, tmp_path):
    folder = _clone(completed_source, tmp_path / "post-diagnostic-added-checkpoint")
    loaded = source.load_source(folder, "debug")
    extra = folder / "jobs/final/extra-unregistered/selected.pt"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"unregistered model")
    with pytest.raises(ValueError, match="checkpoint file coverage changed"):
        source.assert_unchanged(loaded)
