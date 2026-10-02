"""Read-only reuse guards against genuine completed Experiment 3 DEBUG inputs."""

from __future__ import annotations

import copy
import csv
import json
import shutil
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from research.wedge_propagation.generalization.data import file_hash, load_source_run
from research.wedge_propagation.scale_normalization import data


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source(wedge_completed_debug_run):
    return load_source_run(wedge_completed_debug_run, "debug")


@pytest.fixture
def feature_copy(wedge_completed_feature_debug_run, tmp_path):
    output = tmp_path / "feature-source"
    shutil.copytree(wedge_completed_feature_debug_run, output)
    return output


def _document(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")


def _mutate_archive(directory, mutate, amplitude=1.0):
    prefix = f"fresh-a{amplitude:g}"
    path = directory / prefix / "dataset.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutate(arrays)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    digest = file_hash(path)
    manifest = _document(directory, f"{prefix}/data_manifest.json")
    manifest["dataset_sha256"] = digest
    _write_json(directory / prefix / "data_manifest.json", manifest)
    contract = _document(directory, "contract.json")
    contract["fresh_data_hashes"][f"a{amplitude:g}"] = digest
    for row in contract["scenarios"]:
        if row["scenario"] == "fresh" and row["amplitude"] == amplitude:
            row["dataset_sha256"] = digest
    _write_json(directory / "contract.json", contract)


def test_actual_complete_feature_arrays_are_reloaded_and_sources_unchanged(
    source, wedge_completed_feature_debug_run
):
    loaded = data.load_feature_source(wedge_completed_feature_debug_run, source, "debug")
    assert loaded.directory == wedge_completed_feature_debug_run.resolve()
    assert set(loaded.cases_by_amplitude) == set(data.AMPLITUDES)
    assert len(loaded.hashes) == 15
    for amplitude, cases in loaded.cases_by_amplitude.items():
        assert len(cases) == 36
        assert Counter(case.split for case in cases) == Counter(case.split for case in source.cases)
        archive_path = loaded.directory / f"fresh-a{amplitude:g}" / "dataset.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            for case, original in zip(cases, source.cases, strict=True):
                assert case.graph_id == original.graph_id
                assert case.features.shape == (case.num_nodes, 4)
                for name in (*data._GEOMETRY, "features", *data._LABELS):
                    actual = getattr(case, name)
                    assert np.array_equal(actual.numpy(), archive[f"{case.graph_id}__{name}"])
                    assert actual.device.type == "cpu" and not actual.requires_grad
                for name in data._GEOMETRY:
                    assert torch.equal(getattr(case, name), getattr(original, name))
    for cases in loaded.cases_by_amplitude.values():
        for scaled, reference in zip(cases, loaded.cases_by_amplitude[1.0], strict=True):
            amplitude = scaled.metadata["amplitude"]
            assert torch.equal(scaled.features, reference.features * amplitude)
    assert all(
        file_hash(loaded.directory / name) == digest for name, digest in loaded.hashes.items()
    )
    assert all(file_hash(source.run_dir / name) == digest for name, digest in source.hashes.items())


def test_returns_saved_labels_instead_of_replacing_them_with_regeneration(feature_copy, source):
    graph_id = source.cases[0].graph_id
    before = np.load(feature_copy / "fresh-a1/dataset.npz", allow_pickle=False)
    original = before[f"{graph_id}__teacher_c"].copy()
    before.close()

    def mutate(arrays):
        arrays[f"{graph_id}__teacher_c"].flat[0] += 1e-13

    _mutate_archive(feature_copy, mutate)
    loaded = data.load_feature_source(feature_copy, source, "debug")
    with np.load(feature_copy / "fresh-a1/dataset.npz", allow_pickle=False) as archive:
        actual = archive[f"{graph_id}__teacher_c"]
    returned = loaded.cases_by_amplitude[1.0][0].teacher_c.numpy()
    assert np.array_equal(returned, actual)
    assert not np.array_equal(returned, original)


@pytest.mark.parametrize(
    "filename",
    [
        "completion.json",
        "fresh-a0.25/dataset.npz",
        "fresh-a4/data_manifest.json",
        "scale_checks.csv",
    ],
)
def test_missing_required_input_fails(feature_copy, source, filename):
    (feature_copy / filename).unlink()
    with pytest.raises(ValueError, match="missing"):
        data.load_feature_source(feature_copy, source, "debug")


def test_incomplete_profile_and_failure_marker_are_rejected(feature_copy, source):
    with pytest.raises(ValueError, match="canonical full"):
        data.load_feature_source(feature_copy, source, "full")
    completion = _document(feature_copy, "completion.json")
    completion["status"] = "failed"
    _write_json(feature_copy / "completion.json", completion)
    with pytest.raises(ValueError, match="incomplete"):
        data.load_feature_source(feature_copy, source, "debug")
    _write_json(feature_copy / "failure.json", {"status": "failed"})
    with pytest.raises(ValueError, match="failure marker"):
        data.load_feature_source(feature_copy, source, "debug")


@pytest.mark.parametrize(
    "field",
    [
        "source_data_hash",
        "source_artifact_hashes",
        "source_model_hashes",
        "models_unchanged",
        "source_artifacts_unchanged",
        "source_code_unchanged",
        "original_metric_reproduction",
        "amplitudes",
        "metric_rows",
        "scale_rows",
        "source_graph_count",
        "optimizer_updates",
    ],
)
def test_frozen_and_full_coverage_contract_guards(feature_copy, source, field):
    contract = _document(feature_copy, "contract.json")
    contract[field] = {
        "source_data_hash": "0" * 64,
        "source_artifact_hashes": {},
        "source_model_hashes": {},
        "models_unchanged": False,
        "source_artifacts_unchanged": False,
        "source_code_unchanged": False,
        "original_metric_reproduction": {"verified": False},
        "amplitudes": [1.0],
        "metric_rows": 1,
        "scale_rows": 1,
        "source_graph_count": 1,
        "optimizer_updates": 1,
    }[field]
    _write_json(feature_copy / "contract.json", contract)
    with pytest.raises(ValueError):
        data.load_feature_source(feature_copy, source, "debug")


def test_source_hash_mismatch_is_checked_against_actual_files(feature_copy, source):
    hashes = copy.deepcopy(source.hashes)
    hashes["dataset.npz"] = "0" * 64
    forged_source = replace(source, hashes=hashes)
    contract = _document(feature_copy, "contract.json")
    contract["source_artifact_hashes"] = hashes
    _write_json(feature_copy / "contract.json", contract)
    with pytest.raises(ValueError, match="original source artifact changed"):
        data.load_feature_source(feature_copy, forged_source, "debug")


@pytest.mark.parametrize(
    "field",
    [
        "features",
        "edges",
        "wedges",
        "pair_edges",
        "pair_coefficients",
        "teacher_c",
        "target_path",
        "l2x",
    ],
)
def test_altered_inputs_are_rejected_even_when_all_recorded_npz_hashes_are_updated(
    feature_copy, source, field
):
    graph_id = source.cases[0].graph_id

    def mutate(arrays):
        arrays[f"{graph_id}__{field}"].flat[0] += 1

    _mutate_archive(feature_copy, mutate)
    with pytest.raises(ValueError, match="differs from recorded source|label inconsistent"):
        data.load_feature_source(feature_copy, source, "debug")


def test_corrupt_bytes_are_rejected_before_loading(feature_copy, source):
    with (feature_copy / "fresh-a0.25/dataset.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="dataset SHA256"):
        data.load_feature_source(feature_copy, source, "debug")


def test_npz_array_coverage_is_exhaustive(feature_copy, source):
    graph_id = source.cases[0].graph_id
    _mutate_archive(feature_copy, lambda arrays: arrays.pop(f"{graph_id}__target_path"))
    with pytest.raises(ValueError, match="NPZ array coverage"):
        data.load_feature_source(feature_copy, source, "debug")


def test_manifest_metadata_and_path_escape_are_rejected(feature_copy, source):
    manifest = _document(feature_copy, "fresh-a0.25/data_manifest.json")
    manifest["dataset_file"] = "../../outside.npz"
    _write_json(feature_copy / "fresh-a0.25/data_manifest.json", manifest)
    with pytest.raises(ValueError, match="unsafe source input path"):
        data.load_feature_source(feature_copy, source, "debug")
    manifest["dataset_file"] = "dataset.npz"
    manifest["cases"][0]["metadata"]["amplitude"] = 10.0
    _write_json(feature_copy / "fresh-a0.25/data_manifest.json", manifest)
    with pytest.raises(ValueError, match="metadata manifest mismatch"):
        data.load_feature_source(feature_copy, source, "debug")


@pytest.mark.parametrize("filename", ["metrics.csv", "scale_checks.csv"])
def test_metric_csv_cannot_hide_missing_or_duplicate_rows(feature_copy, source, filename):
    path = feature_copy / filename
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV coverage incomplete"):
        data.load_feature_source(feature_copy, source, "debug")
    path.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV duplicate"):
        data.load_feature_source(feature_copy, source, "debug")


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("metrics.csv", "message_abs_rmse", "nan"),
        ("scale_checks.csv", "teacher_message_scale_equivariance_relerr", "nan"),
        ("scale_checks.csv", "student_weight_scale_relerr", "0"),
    ],
)
def test_nonfinite_and_filled_undefined_csv_values_are_rejected(
    feature_copy, source, filename, field, value
):
    path = feature_copy / filename
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV invalid|CSV undefined"):
        data.load_feature_source(feature_copy, source, "debug")
