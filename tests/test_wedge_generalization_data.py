"""Experiment 3 data guards, using a genuinely completed DEBUG source run.

Full-profile tests generate CPU synthetic data only. No full training is run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from research.wedge_propagation.data import feature_content_hash, graph_content_hash
from research.wedge_propagation.generalization import data
from research.wedge_propagation.learned.data import prepare_cases, save_dataset
from research.wedge_propagation.learned.model import teacher_weights
from research.wedge_propagation.operators import dense_incidence, dense_wedge

_LEARNED = Path(__file__).resolve().parents[1] / "research/wedge_propagation/learned"
_FLOAT_FIELDS = ("features", "pair_coefficients", "teacher_c", "lx", "l2x", "qx", "target_path")
_INDEX_FIELDS = ("edges", "wedges", "pair_edges")


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def source(wedge_completed_debug_run):
    return data.load_source_run(wedge_completed_debug_run, "debug")


@pytest.fixture
def source_copy(wedge_completed_debug_run, tmp_path):
    directory = tmp_path / "source"
    shutil.copytree(wedge_completed_debug_run, directory)
    return directory


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _document(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _mutate_archive(directory, mutate):
    path = directory / "dataset.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutate(arrays)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = _document(directory, "data_manifest.json")
    manifest["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(directory / "data_manifest.json", manifest)


def test_complete_debug_reload_is_exact_read_only_and_hashes_every_input(source):
    assert len(source.cases) == 36
    assert sum(case.features.shape[1] for case in source.cases) == 144
    assert Counter(case.split for case in source.cases) == {
        "train": 12, "validation": 6, "id": 6, "size_ood": 3,
        "family_ood": 6, "family_size_ood": 3,
    }
    required = {"config.json", "contract.json", "completion.json", "data_manifest.json",
                "dataset.npz", "metrics.csv", "interventions.csv", "teacher_operator_audit.csv"}
    required.update(f"{target}-{condition}-fit.json" for target in ("L", "L2", "path")
                    for condition in ("first", "polynomial", "fixed"))
    required.update(f"checkpoints/{target}-{condition}-selected.pt"
                    for target in ("L", "L2", "path") for condition in ("learned", "random_pair"))
    assert required <= source.hashes.keys()
    assert all("\\" not in name for name in source.hashes)
    with np.load(source.run_dir / "dataset.npz", allow_pickle=False) as archive:
        for case in source.cases:
            for name in (*_INDEX_FIELDS, *_FLOAT_FIELDS):
                tensor = getattr(case, name)
                assert np.array_equal(tensor.numpy(), archive[f"{case.graph_id}__{name}"])
                assert tensor.device.type == "cpu" and not tensor.requires_grad
                assert tensor.dtype == (torch.long if name in _INDEX_FIELDS else torch.float64)
    again = data.load_source_run(source.run_dir, "debug")
    assert source.hashes == again.hashes


def test_loader_does_not_regenerate_source_topology(source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source graph generation is forbidden")
    monkeypatch.setattr("research.wedge_propagation.learned.data.make_case", forbidden)
    again = data.load_source_run(source.run_dir, "debug")
    assert [graph_content_hash(case) for case in again.cases] == [
        graph_content_hash(case) for case in source.cases]


@pytest.mark.parametrize("filename", ["completion.json", "dataset.npz",
                                     "checkpoints/path-learned-selected.pt"])
def test_missing_required_source_input_fails(source_copy, filename):
    (source_copy / filename).unlink()
    with pytest.raises(ValueError, match="missing"):
        data.load_source_run(source_copy, "debug")


def test_completion_and_profile_guards(source_copy):
    with pytest.raises(ValueError, match="canonical full"):
        data.load_source_run(source_copy, "full")
    completion = _document(source_copy, "completion.json")
    completion["status"] = "running"
    _write_json(source_copy / "completion.json", completion)
    with pytest.raises(ValueError, match="incomplete"):
        data.load_source_run(source_copy, "debug")


def test_failure_marker_cannot_be_ignored(source_copy):
    _write_json(source_copy / "failure.json", {"status": "failed"})
    with pytest.raises(ValueError, match="failure marker"):
        data.load_source_run(source_copy, "debug")


def test_config_hash_and_canonical_scale_guards(source_copy):
    contract = _document(source_copy, "contract.json")
    contract["config_hash"] = "0" * 64
    _write_json(source_copy / "contract.json", contract)
    with pytest.raises(ValueError, match="hash disagreement"):
        data.load_source_run(source_copy, "debug")
    config = _document(source_copy, "config.json")
    config["features"] = 1
    _write_json(source_copy / "config.json", config)
    with pytest.raises(ValueError, match="canonical debug"):
        data.load_source_run(source_copy, "debug")


def test_source_math_hash_paths_are_cross_platform_and_changes_are_rejected(source_copy):
    contract = _document(source_copy, "contract.json")
    hashes = contract["source"]["sha256"]
    contract["source"]["sha256"] = {name.replace("\\", "/"): value
                                        for name, value in hashes.items()}
    _write_json(source_copy / "contract.json", contract)
    assert len(data.load_source_run(source_copy, "debug").cases) == 36
    contract["source"]["sha256"]["learned/model.py"] = "0" * 64
    _write_json(source_copy / "contract.json", contract)
    with pytest.raises(ValueError, match="mathematical implementation differs"):
        data.load_source_run(source_copy, "debug")


def test_dataset_path_cannot_escape_source(source_copy):
    manifest = _document(source_copy, "data_manifest.json")
    manifest["dataset_file"] = "../dataset.npz"
    _write_json(source_copy / "data_manifest.json", manifest)
    with pytest.raises(ValueError, match="unsafe source input path"):
        data.load_source_run(source_copy, "debug")


def test_dataset_bytes_are_checked_before_array_loading(source_copy):
    with (source_copy / "dataset.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="dataset SHA256"):
        data.load_source_run(source_copy, "debug")


@pytest.mark.parametrize("field", ["target_path", "teacher_c", "l2x"])
def test_changed_labels_are_rejected_even_with_updated_dataset_sha(source_copy, field):
    graph_id = _document(source_copy, "data_manifest.json")["cases"][0]["graph_id"]
    def mutate(arrays):
        arrays[f"{graph_id}__{field}"].flat[0] += 0.1
    _mutate_archive(source_copy, mutate)
    with pytest.raises(ValueError, match="target inconsistent with equations"):
        data.load_source_run(source_copy, "debug")


def test_feature_hash_is_checked_after_npz_sha(source_copy):
    graph_id = _document(source_copy, "data_manifest.json")["cases"][0]["graph_id"]
    def mutate(arrays):
        arrays[f"{graph_id}__features"].flat[0] += 1
    _mutate_archive(source_copy, mutate)
    with pytest.raises(ValueError, match="feature content hash mismatch"):
        data.load_source_run(source_copy, "debug")


def test_dtype_and_nonfinite_values_cannot_pass_source_loader(source_copy):
    graph_id = _document(source_copy, "data_manifest.json")["cases"][0]["graph_id"]
    name = f"{graph_id}__features"
    def mutate(arrays):
        arrays[name] = arrays[name].astype(np.float32)
    _mutate_archive(source_copy, mutate)
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        data.load_source_run(source_copy, "debug")
    def mutate_again(arrays):
        arrays[name] = arrays[name].astype(np.float64)
        arrays[name].flat[0] = np.nan
    _mutate_archive(source_copy, mutate_again)
    with pytest.raises(ValueError, match="nonfinite source array"):
        data.load_source_run(source_copy, "debug")


def test_all_actual_unordered_wedges_must_be_retained(source_copy):
    manifest = _document(source_copy, "data_manifest.json")
    row = manifest["cases"][0]
    row["num_paths"] -= 1
    _write_json(source_copy / "data_manifest.json", manifest)
    def mutate(arrays):
        for field in ("wedges", "pair_edges", "pair_coefficients"):
            arrays[f"{row['graph_id']}__{field}"] = arrays[f"{row['graph_id']}__{field}"][:, :-1]
        for field in ("teacher_c", "pair_donor_path"):
            arrays[f"{row['graph_id']}__{field}"] = arrays[f"{row['graph_id']}__{field}"][:-1]
    _mutate_archive(source_copy, mutate)
    with pytest.raises(ValueError, match="unordered wedges"):
        data.load_source_run(source_copy, "debug")


def test_signed_random_pair_norm_is_checked(source_copy):
    graph_id = _document(source_copy, "data_manifest.json")["cases"][0]["graph_id"]
    def mutate(arrays):
        arrays[f"{graph_id}__pair_coefficients"][:, 0] *= 1.1
    _mutate_archive(source_copy, mutate)
    with pytest.raises(ValueError, match="row normalization"):
        data.load_source_run(source_copy, "debug")


def test_random_pair_donor_correspondence_cannot_be_reordered(source_copy):
    graph_id = _document(source_copy, "data_manifest.json")["cases"][0]["graph_id"]
    def mutate(arrays):
        name = f"{graph_id}__pair_donor_path"
        arrays[name] = arrays[name][::-1].copy()
    _mutate_archive(source_copy, mutate)
    with pytest.raises(ValueError, match="donor correspondence"):
        data.load_source_run(source_copy, "debug")


def test_manifest_order_and_metric_coverage_are_checked(source_copy):
    manifest = _document(source_copy, "data_manifest.json")
    manifest["cases"][0], manifest["cases"][1] = manifest["cases"][1], manifest["cases"][0]
    _write_json(source_copy / "data_manifest.json", manifest)
    with pytest.raises(ValueError, match="IDs/order mismatch"):
        data.load_source_run(source_copy, "debug")
    manifest["cases"][0], manifest["cases"][1] = manifest["cases"][1], manifest["cases"][0]
    _write_json(source_copy / "data_manifest.json", manifest)
    metrics = source_copy / "metrics.csv"
    metrics.write_text("\n".join(metrics.read_text(encoding="utf-8").splitlines()[:-1]),
                       encoding="utf-8")
    with pytest.raises(ValueError, match="CSV coverage incomplete"):
        data.load_source_run(source_copy, "debug")


def test_fresh_features_are_paired_across_every_amplitude_and_leave_source_unchanged(source):
    original_hashes = [feature_content_hash(case) for case in source.cases]
    original_metadata = copy.deepcopy([case.metadata for case in source.cases])
    reference = data.feature_cases(source.cases, source.config["teacher"], 20261003, 1, workers=2)
    for amplitude in (0.25, 0.5, 1, 2, 4):
        scaled = data.feature_cases(source.cases, source.config["teacher"], 20261003,
                                    amplitude, workers=2)
        assert len(scaled) == len(source.cases)
        for old, fresh, case in zip(source.cases, reference, scaled, strict=True):
            assert (case.graph_id, case.split, case.family, case.num_nodes) == (
                old.graph_id, old.split, old.family, old.num_nodes)
            assert case.feature_seed == fresh.feature_seed != old.feature_seed
            assert case.metadata["fresh_feature_seed"] == case.feature_seed
            assert (case.metadata["fresh_base_feature_hash"]
                    == fresh.metadata["fresh_base_feature_hash"])
            assert case.metadata["source_feature_hash"] == feature_content_hash(old)
            assert case.metadata["feature_status"] == "fresh"
            assert case.metadata["amplitude"] == amplitude
            assert case.metadata["resample_attempt"] == old.metadata["resample_attempt"]
            assert case.features.shape == old.features.shape
            assert torch.equal(case.features, fresh.features * amplitude)
            assert not torch.equal(case.features, old.features)
            for name in (*_INDEX_FIELDS, "pair_coefficients"):
                assert getattr(case, name) is getattr(old, name)
            for name in ("lx", "l2x", "qx"):
                torch.testing.assert_close(getattr(case, name), getattr(fresh, name) * amplitude,
                                           rtol=1e-12, atol=1e-12)
    assert original_hashes == [feature_content_hash(case) for case in source.cases]
    assert original_metadata == [case.metadata for case in source.cases]


def test_every_fresh_label_is_the_actual_scaled_equation(source):
    cases = data.feature_cases(source.cases, source.config["teacher"], 20261003, 4, workers=2)
    for case in cases:
        x, (i, j, k) = case.features, case.wedges
        b = dense_incidence(case.edges, case.num_nodes)
        a = dense_wedge(case.wedges, case.num_nodes)
        laplacian = b.T @ b
        c = teacher_weights(x[j] - x[i], x[k] - x[j],
                            torch.zeros(len(i), dtype=torch.long), 1, **source.config["teacher"])
        torch.testing.assert_close(case.teacher_c, c, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(case.lx, laplacian @ x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(case.l2x, laplacian @ laplacian @ x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(case.qx, a.T @ a @ x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(case.target_path, a.T @ (c * (a @ x)),
                                   rtol=1e-12, atol=1e-12)


def test_amplitude_recomputes_teacher_epsilon_instead_of_reusing_old_c(source):
    case = source.cases[0]
    # An explicit near-epsilon synthetic field makes the epsilon dependence measurable.
    tiny = data.feature_cases([case], source.config["teacher"], 20261003, 1e-9, workers=1)[0]
    unit = data.feature_cases([case], source.config["teacher"], 20261003, 1, workers=1)[0]
    assert not torch.allclose(tiny.teacher_c, unit.teacher_c, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(tiny.target_path, unit.target_path * 1e-9, rtol=1e-3, atol=1e-15)


def test_worker_count_and_call_order_do_not_change_new_features_or_labels(source):
    sequential = data.feature_cases(source.cases, source.config["teacher"], 20261003, 0.5, 1)
    parallel = data.feature_cases(source.cases, source.config["teacher"], 20261003, 0.5, 4)
    reverse = data.feature_cases(list(reversed(source.cases)), source.config["teacher"],
                                 20261003, 0.5, 2)
    assert [case.graph_id for case in sequential] == [case.graph_id for case in parallel]
    by_id = {case.graph_id: case for case in reverse}
    for left, right in zip(sequential, parallel, strict=True):
        for field in _FLOAT_FIELDS:
            assert torch.equal(getattr(left, field), getattr(right, field))
            assert torch.equal(getattr(left, field), getattr(by_id[left.graph_id], field))
        assert left.metadata == right.metadata == by_id[left.graph_id].metadata
    other_seed = data.feature_cases(source.cases, source.config["teacher"], 20261004, 0.5, 2)
    assert all(not torch.equal(left.features, right.features)
               for left, right in zip(sequential, other_seed, strict=True))


def test_full_fresh_data_preserves_531_graphs_and_8496_scalar_fields():
    config = json.loads((_LEARNED / "config_full.json").read_text(encoding="utf-8"))
    originals = prepare_cases(config, workers=2)  # CPU data construction; no training.
    fresh = data.feature_cases(originals, config["teacher"], 20261003, 1, workers=2)
    assert len(fresh) == 531
    assert sum(case.features.shape[1] for case in fresh) == 8496
    assert Counter(case.split for case in fresh) == {
        "train": 240, "validation": 60, "id": 120, "size_ood": 90,
        "family_ood": 12, "family_size_ood": 9,
    }
    assert [graph_content_hash(case) for case in originals] == [graph_content_hash(case)
                                                              for case in fresh]
    assert not ({feature_content_hash(case) for case in originals}
                & {feature_content_hash(case) for case in fresh})


def test_fresh_datasets_use_exclusive_reproducible_npz_manifest(source, tmp_path):
    fresh = data.feature_cases(source.cases, source.config["teacher"], 20261003, 0.25, 2)
    output = tmp_path / "fresh-a0.25"
    manifest = save_dataset(output, fresh)
    assert manifest["graph_count"] == 36 and manifest["feature_realization_count"] == 144
    assert manifest["dataset_sha256"] == data.file_hash(output / "dataset.npz")
    assert all(row["metadata"]["amplitude"] == 0.25 for row in manifest["cases"])
    with pytest.raises(FileExistsError, match="overwrite"):
        save_dataset(output, fresh)


@pytest.mark.parametrize("amplitude", [0, -1, float("nan"), float("inf")])
def test_invalid_amplitudes_fail_explicitly(source, amplitude):
    with pytest.raises(ValueError, match="amplitude"):
        data.feature_cases(source.cases, source.config["teacher"], 20261003, amplitude, 1)


def test_collision_and_mismatched_teacher_are_rejected(source, monkeypatch):
    monkeypatch.setattr(data, "_fresh_seed",
                        lambda master_seed, graph_id: source.cases[0].feature_seed)
    with pytest.raises(ValueError, match="seed collision"):
        data.feature_cases([source.cases[0]], source.config["teacher"], 20261003, 1, 1)
    monkeypatch.undo()
    wrong_teacher = {**source.config["teacher"], "tau": 2}
    with pytest.raises(ValueError, match="teacher differs"):
        data.feature_cases(source.cases, wrong_teacher, 20261003, 1, 1)
    case = source.cases[0]
    seed = data._fresh_seed(20261003, case.graph_id)
    x = torch.from_numpy(np.random.default_rng(seed).standard_normal(case.features.shape))
    colliding = replace(case, features=x)
    with pytest.raises(ValueError, match="base feature collision"):
        data.feature_cases([colliding], source.config["teacher"], 20261003, 1, 1)
