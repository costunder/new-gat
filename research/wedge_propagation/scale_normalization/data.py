"""Read the exact completed Experiment 3 fields for Experiment 3.1.

All returned arrays come from the five saved NPZ files. Repeating the recorded
feature generator is a validation check only; it never replaces an input or
samples a new topology. The original teacher, including its epsilon, remains
the reference at every amplitude.
"""

from __future__ import annotations

import copy
import csv
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import feature_content_hash, graph_content_hash
from ..generalization.data import (
    _ARRAY_FIELDS,
    SourceRun,
    _integer,
    _json,
    _require,
    _source_path,
    _tensor,
    feature_cases,
    file_hash,
)
from ..learned.data import RuleCase, make_specs

AMPLITUDES = (0.25, 0.5, 1.0, 2.0, 4.0)
_TARGETS = ("L", "L2", "path")
_CONDITIONS = ("first", "polynomial", "fixed", "learned", "random_pair")
_SPLITS = ("train", "validation", "id", "size_ood", "family_ood", "family_size_ood")
_GEOMETRY = ("edges", "wedges", "pair_edges", "pair_coefficients")
_LABELS = ("teacher_c", "lx", "l2x", "qx", "target_path")


@dataclass(frozen=True)
class FeatureSource:
    directory: Path
    config: dict[str, Any]
    contract: dict[str, Any]
    hashes: dict[str, str]
    cases_by_amplitude: dict[float, list[RuleCase]]


def _hashes_valid(values: Any) -> bool:
    return (
        isinstance(values, dict)
        and bool(values)
        and all(
            isinstance(key, str)
            and isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for key, value in values.items()
        )
    )


def _validate_contract(source, config, contract, completion, profile):
    _require(profile in ("full", "debug"), "profile must be full or debug")
    root = Path(__file__).parents[1]
    canonical_source = _json(root / "learned" / f"config_{profile}.json")
    canonical_feature = _json(root / "generalization" / f"config_{profile}.json")
    _require(
        source.config == canonical_source,
        f"Experiment 2 source is not the canonical {profile} configuration",
    )
    _require(
        config == canonical_feature and contract.get("config") == config,
        f"Experiment 3 feature config is not the canonical {profile} contract",
    )
    _require(
        completion.get("status") == "complete"
        and completion.get("experiment") == contract.get("experiment") == 3,
        "Experiment 3 feature source is incomplete",
    )
    _require(
        completion.get("profile") == contract.get("profile") == profile
        and contract.get("debug") is (profile == "debug"),
        "Experiment 3 feature profile mismatch",
    )
    _require(
        contract.get("source_config") == source.config,
        "Experiment 3 refers to a different Experiment 2 configuration",
    )
    _require(
        contract.get("source_data_hash") == source.manifest["dataset_sha256"],
        "Experiment 3 original dataset hash mismatch",
    )
    _require(
        contract.get("source_artifact_hashes") == source.hashes,
        "Experiment 3 original artifact hashes mismatch",
    )
    for document in (contract, completion):
        _require(
            document.get("models_unchanged") is True
            and document.get("source_artifacts_unchanged") is True,
            "Experiment 3 did not preserve its models/source artifacts",
        )
    _require(
        contract.get("source_code_unchanged") is True
        and _hashes_valid(contract.get("source_code_sha256")),
        "Experiment 3 code preservation guard is missing",
    )
    expected_models = {f"{target}/{condition}" for target in _TARGETS for condition in _CONDITIONS}
    model_hashes = contract.get("source_model_hashes")
    _require(
        _hashes_valid(model_hashes)
        and set(model_hashes) == expected_models
        and model_hashes == contract.get("model_hashes_after"),
        "Experiment 3 frozen model state guard mismatch",
    )
    _require(
        contract.get("original_metric_reproduction", {}).get("verified") is True,
        "Experiment 3 original metric replay was not verified",
    )
    for field in (
        "new_training_epochs",
        "optimizer_updates",
        "checkpoint_selection_updates",
        "scalar_refits",
    ):
        _require(contract.get(field) == 0, f"Experiment 3 performed forbidden {field}")
    specs = make_specs(source.config)  # Declared identities only; no graph resampling.
    _require(
        [case.graph_id for case in source.cases] == [spec.graph_id for spec in specs],
        "Experiment 2 source graph identities/order are incomplete",
    )
    count, fields = len(specs), source.config["features"]
    metric_count = count * len(_TARGETS) * (3 + 2 * len(source.config["model_seeds"]))
    _require(
        contract.get("source_graph_count") == completion.get("graphs") == count
        and contract.get("source_input_count") == count * fields
        and contract.get("feature_realizations") == fields,
        "Experiment 3 omitted source graphs or feature realizations",
    )
    _require(
        contract.get("scenario_count") == 6
        and contract.get("graph_treatments") == 6 * count
        and contract.get("input_treatments") == 6 * count * fields,
        "Experiment 3 scenario/input treatment coverage mismatch",
    )
    _require(
        contract.get("amplitudes") == list(AMPLITUDES)
        and contract.get("new_feature_seed") == 20261003
        and contract.get("feature_seed_stream") == "wedge-generalization-v1"
        and contract.get("feature_amplitudes_are_paired_not_independent_graph_draws") is True,
        "Experiment 3 paired feature amplitude contract mismatch",
    )
    _require(
        contract.get("data_fraction") == contract.get("sampling_ratio") == 1
        and contract.get("path_sampling") is False
        and contract.get("model_seeds") == source.config["model_seeds"],
        "Experiment 3 graph/path/seed coverage mismatch",
    )
    _require(
        contract.get("dtype") == source.config["dtype"]
        and contract.get("teacher_reference_dtype") == "float64"
        and contract.get("real_dataset_classification") is False,
        "Experiment 3 precision or research scope mismatch",
    )
    _require(
        contract.get("metric_rows") == completion.get("metric_rows") == 6 * metric_count
        and contract.get("scale_rows") == completion.get("scale_rows") == 5 * metric_count,
        "Experiment 3 metric/scale coverage mismatch",
    )
    expected_splits = Counter(case.split for case in source.cases)
    statistics = contract.get("graph_statistics", {})
    _require(
        set(statistics) == set(_SPLITS)
        and all(statistics[split].get("graphs") == expected_splits[split] for split in _SPLITS),
        "Experiment 3 split coverage mismatch",
    )
    scenarios = contract.get("scenarios")
    _require(
        isinstance(scenarios, list) and len(scenarios) == 6,
        "Experiment 3 scenario manifest missing",
    )
    by_key = {(row.get("scenario"), row.get("amplitude")): row for row in scenarios}
    expected_keys = {("original", 1.0), *(("fresh", amplitude) for amplitude in AMPLITUDES)}
    _require(
        set(by_key) == expected_keys and len(by_key) == len(scenarios),
        "Experiment 3 scenario manifest duplicates/omits an amplitude",
    )
    for row in scenarios:
        _require(
            row.get("graphs") == count and row.get("inputs") == count * fields,
            "Experiment 3 scenario omits graphs or scalar fields",
        )
    _require(
        by_key[("original", 1.0)].get("dataset_sha256") == source.manifest["dataset_sha256"],
        "Experiment 3 original scenario dataset hash mismatch",
    )
    fresh_hashes = contract.get("fresh_data_hashes")
    _require(
        _hashes_valid(fresh_hashes)
        and set(fresh_hashes) == {f"a{amplitude:g}" for amplitude in AMPLITUDES},
        "Experiment 3 fresh dataset hash manifest mismatch",
    )
    return by_key


def _case_row(case):
    return {
        "graph_id": case.graph_id,
        "family": case.family,
        "split": case.split,
        "num_nodes": case.num_nodes,
        "num_edges": case.edges.shape[1],
        "num_paths": case.wedges.shape[1],
        "num_features": case.features.shape[1],
        "graph_seed": case.graph_seed,
        "feature_seed": case.feature_seed,
        "pair_seed": case.pair_seed,
        "graph_content_hash": graph_content_hash(case),
        "feature_content_hash": feature_content_hash(case),
        "metadata": case.metadata,
    }


def _load_amplitude(directory, source, amplitude, scenario, contract, hashes):
    prefix = f"fresh-a{amplitude:g}"
    manifest_name = f"{prefix}/data_manifest.json"
    manifest_path = _source_path(directory, manifest_name)
    hashes[manifest_name] = file_hash(manifest_path)
    manifest = _json(manifest_path)
    relative = f"{prefix}/{manifest.get('dataset_file', '')}"
    dataset_path = _source_path(directory, relative)
    _require(
        relative == scenario.get("dataset_file") == f"{prefix}/dataset.npz",
        f"Experiment 3 dataset file mismatch: amplitude={amplitude:g}",
    )
    digest = file_hash(dataset_path)
    hashes[relative] = digest
    _require(
        digest
        == manifest.get("dataset_sha256")
        == scenario.get("dataset_sha256")
        == contract["fresh_data_hashes"][f"a{amplitude:g}"],
        f"Experiment 3 dataset SHA256 mismatch: amplitude={amplitude:g}",
    )
    expected = feature_cases(
        source.cases, source.config["teacher"], 20261003, amplitude, source.validation_workers
    )
    counts = dict(Counter(case.split for case in expected))
    _require(
        manifest.get("graph_count") == len(expected)
        and manifest.get("feature_realization_count") == len(expected) * source.config["features"]
        and manifest.get("split_graph_counts") == counts
        and manifest.get("split_feature_counts")
        == {split: count * source.config["features"] for split, count in counts.items()}
        and manifest.get("graph_content_unique") is True
        and manifest.get("feature_content_unique") is True,
        f"Experiment 3 dataset coverage mismatch: amplitude={amplitude:g}",
    )
    rows = manifest.get("cases")
    _require(
        isinstance(rows, list) and len(rows) == len(expected),
        f"Experiment 3 case manifest incomplete: amplitude={amplitude:g}",
    )
    array_keys = {f"{case.graph_id}__{name}" for case in expected for name in _ARRAY_FIELDS}
    cases = []
    with np.load(dataset_path, allow_pickle=False) as archive:
        _require(
            len(archive.files) == len(array_keys) and set(archive.files) == array_keys,
            f"Experiment 3 NPZ array coverage mismatch: amplitude={amplitude:g}",
        )
        for row, correct in zip(rows, expected, strict=True):
            _require(
                row == _case_row(correct),
                f"Experiment 3 graph/feature/metadata manifest mismatch: {correct.graph_id}",
            )
            p = correct.wedges.shape[1]
            values = {
                name: _tensor(
                    archive,
                    f"{correct.graph_id}__{name}",
                    tuple(getattr(correct, name).shape),
                    name not in ("edges", "wedges", "pair_edges"),
                )
                for name in _ARRAY_FIELDS
                if name != "pair_donor_path"
            }
            donor = _tensor(archive, f"{correct.graph_id}__pair_donor_path", (p,), False)
            _require(
                torch.equal(donor, torch.arange(p)),
                f"Experiment 3 pair donor correspondence mismatch: {correct.graph_id}",
            )
            for name in (*_GEOMETRY, "features"):
                _require(
                    torch.equal(values[name], getattr(correct, name)),
                    f"Experiment 3 actual {name} differs from recorded source: {correct.graph_id}",
                )
            for name in _LABELS:
                _require(
                    torch.allclose(values[name], getattr(correct, name), rtol=1e-11, atol=1e-11),
                    f"Experiment 3 teacher/operator label inconsistent: {correct.graph_id}/{name}",
                )
            if p:
                _require(
                    torch.allclose(
                        values["teacher_c"].mean(0),
                        torch.ones(source.config["features"], dtype=torch.float64),
                        rtol=1e-11,
                        atol=1e-11,
                    ),
                    f"Experiment 3 teacher mean gauge mismatch: {correct.graph_id}",
                )
            cases.append(replace(correct, metadata=copy.deepcopy(row["metadata"]), **values))
    return cases


def _validate_metric_csv(path, source, kind):
    by_id = {case.graph_id: case for case in source.cases}
    expected = {
        (scenario, amplitude, target, condition, seed, case.split, case.graph_id)
        for scenario, amplitude in (
            (("fresh", amplitude) for amplitude in AMPLITUDES)
            if kind == "scale"
            else (("original", 1.0), *(("fresh", amplitude) for amplitude in AMPLITUDES))
        )
        for target in _TARGETS
        for condition in _CONDITIONS
        for seed in (
            [-1] if condition in ("first", "polynomial", "fixed") else source.config["model_seeds"]
        )
        for case in source.cases
    }
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["scenario"],
                float(row["amplitude"]),
                row["target"],
                row["condition"],
                int(row["seed"]),
                row["split"],
                row["graph_id"],
            )
            _require(
                key in expected and key not in seen,
                f"Experiment 3 CSV duplicate/unexpected key: {path.name}/{key}",
            )
            case = by_id[row["graph_id"]]
            _require(
                row["family"] == case.family, f"Experiment 3 CSV family mismatch: {case.graph_id}"
            )
            if kind == "metrics":
                _require(
                    row["source_split"] == case.split
                    and row["feature_status"]
                    == ("source_features" if row["scenario"] == "original" else "unseen_features"),
                    "Experiment 3 CSV source/feature status mismatch",
                )
                for field, value in (
                    ("num_nodes", case.num_nodes),
                    ("num_edges", case.edges.shape[1]),
                    ("num_paths", case.wedges.shape[1]),
                    ("num_realizations", source.config["features"]),
                ):
                    _require(int(row[field]) == value, f"Experiment 3 CSV {field} mismatch")
                for field in ("message_relerr", "message_abs_rmse"):
                    _require(
                        math.isfinite(float(row[field])) and float(row[field]) >= 0,
                        f"Experiment 3 CSV invalid metric: {field}",
                    )
            else:
                for field, defined in (
                    (
                        "student_weight_scale_relerr",
                        row["condition"] in ("learned", "random_pair") and case.wedges.shape[1] > 0,
                    ),
                    (
                        "teacher_weight_scale_relerr",
                        row["target"] == "path" and case.wedges.shape[1] > 0,
                    ),
                    ("message_scale_equivariance_relerr", True),
                    ("teacher_message_scale_equivariance_relerr", True),
                ):
                    if not defined:
                        _require(
                            row[field] == "", f"Experiment 3 CSV undefined metric filled: {field}"
                        )
                    else:
                        _require(
                            row[field] != ""
                            and math.isfinite(float(row[field]))
                            and float(row[field]) >= 0,
                            f"Experiment 3 CSV invalid scale metric: {field}",
                        )
            seen.add(key)
    _require(seen == expected, f"Experiment 3 CSV coverage incomplete: {path.name}")


def load_feature_source(run_dir: Path | str, source: SourceRun, profile: str) -> FeatureSource:
    """Validate all recorded Experiment 3 inputs and return their actual arrays."""
    directory = Path(run_dir).resolve()
    _require(directory.is_dir(), f"Experiment 3 feature directory missing: {directory}")
    _require(not (directory / "failure.json").exists(), "feature source contains a failure marker")
    _require(
        not (source.run_dir / "failure.json").exists(), "original source contains a failure marker"
    )
    _require(_hashes_valid(source.hashes), "original source artifact hash manifest is invalid")
    for name, digest in source.hashes.items():
        _require(
            file_hash(_source_path(source.run_dir.resolve(), name)) == digest,
            f"original source artifact changed: {name}",
        )
    names = ("config.json", "contract.json", "completion.json")
    paths = {name: _source_path(directory, name) for name in names}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    config, contract, completion = (_json(paths[name]) for name in names)
    scenarios = _validate_contract(source, config, contract, completion, profile)
    _integer(source.validation_workers, "source validation workers", 1)
    print(
        f"[feature source] validating all {len(source.cases)} graphs and five saved amplitudes; "
        f"workers={source.validation_workers}",
        flush=True,
    )
    loaded = {}
    for amplitude in AMPLITUDES:
        started = time.perf_counter()
        loaded[amplitude] = _load_amplitude(
            directory, source, amplitude, scenarios[("fresh", amplitude)], contract, hashes
        )
        print(
            f"[feature source] amplitude={amplitude:g} graphs={len(loaded[amplitude])} "
            f"verified seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )
    for name, kind in (("metrics.csv", "metrics"), ("scale_checks.csv", "scale")):
        path = _source_path(directory, name)
        hashes[name] = file_hash(path)
        _validate_metric_csv(path, source, kind)
    _require(
        all(file_hash(_source_path(directory, name)) == digest for name, digest in hashes.items()),
        "feature source changed while being validated",
    )
    _require(
        all(
            file_hash(_source_path(source.run_dir.resolve(), name)) == digest
            for name, digest in source.hashes.items()
        ),
        "original source changed while validating feature inputs",
    )
    return FeatureSource(directory, config, contract, hashes, loaded)
