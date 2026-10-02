"""Read-only Experiment 2 evidence and paired, independent Experiment 3 fields.

Source topology, unordered wedges, signed random pairs and original float64
labels are loaded from the recorded NPZ, never generated again. A source must
be a completed, complete canonical full/debug Experiment 2 run. SHA256 checks
cover every input artifact; mathematical source paths are normalized across
Windows and Linux before their recorded byte hashes are compared.

Each fresh feature column is an independent scalar Gaussian field. Its seed
does not include amplitude, so every amplitude uses exactly the same fresh
base field. Teacher weights and all operator targets are recomputed at each
amplitude, including the teacher's epsilon. This module never trains a model.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import psutil
import torch

from ..data import feature_content_hash, graph_content_hash
from ..learned.data import RuleCase, make_specs
from ..learned.model import teacher_weights
from ..operators import build_wedges, fixed_wedge_apply, laplacian_apply, wedge_transpose

_TARGETS = ("L", "L2", "path")
_CONDITIONS = ("first", "polynomial", "fixed", "learned", "random_pair")
_BASELINES = ("first", "polynomial", "fixed")
_GATES = ("learned", "random_pair")
_INTERVENTIONS = (
    "identity", "mean", "weight_shuffle", "other_graph_pattern",
    "correspondence_randomization",
)
_ARRAY_FIELDS = (
    "edges", "wedges", "pair_edges", "pair_coefficients", "features",
    "teacher_c", "lx", "l2x", "qx", "target_path", "pair_donor_path",
)
_MATH_FILES = (
    "data.py", "operators.py", "learned/data.py", "learned/model.py",
    "learned/evaluation.py",
)


@dataclass(frozen=True)
class SourceRun:
    run_dir: Path
    config: dict[str, Any]
    contract: dict[str, Any]
    completion: dict[str, Any]
    manifest: dict[str, Any]
    cases: list[RuleCase]
    hashes: dict[str, str]
    validation_workers: int = 1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"nonfinite JSON number: {value}")
    return result


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=_reject_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _require(isinstance(value, Integral) and not isinstance(value, bool)
             and value >= minimum, f"{label} must be an integer >= {minimum}")
    return int(value)


def _source_path(directory: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), "empty source input path")
    normalized = relative.replace("\\", "/")
    parts = PurePosixPath(normalized)
    _require(not parts.is_absolute() and ":" not in normalized
             and ".." not in parts.parts, f"unsafe source input path: {relative}")
    resolved = (directory / normalized).resolve()
    _require(resolved.is_relative_to(directory), f"source input escapes run: {relative}")
    _require(resolved.is_file(), f"required source input missing: {relative}")
    return resolved


def _validate_contract(config: dict, contract: dict, completion: dict,
                       manifest: dict, profile: str) -> list:
    _require(profile in ("full", "debug"), "profile must be full or debug")
    canonical = _json(Path(__file__).parents[1] / "learned" / f"config_{profile}.json")
    _require(config == canonical, f"source config is not the canonical {profile} contract")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    _require(contract.get("config") == config and contract.get("config_hash") == digest,
             "source contract/config hash disagreement")
    _require(completion.get("status") == "complete"
             and completion.get("experiment") == contract.get("experiment") == 2,
             "source Experiment 2 is incomplete")
    _require(completion.get("profile") == contract.get("profile") == profile
             and contract.get("debug") is (profile == "debug"), "source profile mismatch")
    _require(contract.get("source_unchanged_during_run") is True,
             "source code changed during Experiment 2")
    recorded = contract.get("source", {}).get("sha256", {})
    _require(isinstance(recorded, dict), "missing source math SHA256 manifest")
    normalized = {key.replace("\\", "/"): value for key, value in recorded.items()}
    _require(len(normalized) == len(recorded), "duplicate normalized source hash paths")
    math_root = Path(__file__).parents[1]
    for name in _MATH_FILES:
        _require(normalized.get(name) == file_hash(math_root / name),
                 f"source mathematical implementation differs: {name}")
    specs = make_specs(config)  # Metadata only; does not sample a graph or field.
    count, inputs = len(specs), len(specs) * config["features"]
    split_counts = dict(Counter(spec.split for spec in specs))
    expected_metrics = (count - split_counts["train"]) * (
        len(_TARGETS) * len(_BASELINES) + len(_TARGETS) * len(_GATES)
        * len(config["model_seeds"]))
    expected_controls = ((count - split_counts["train"]) * len(_TARGETS)
                         * len(config["model_seeds"]) * len(_INTERVENTIONS))
    for field in ("graph_count_total", "graph_count_used", "teacher_operator_audit_graphs"):
        _require(contract.get(field) == count, f"source {field} omits graphs")
    _require(completion.get("graphs") == manifest.get("graph_count") == count,
             "source total graph count mismatch")
    _require(contract.get("input_count_total") == manifest.get("feature_realization_count")
             == inputs, "source scalar realization count mismatch")
    _require(manifest.get("split_graph_counts") == contract.get("data_manifest_counts")
             == split_counts, "source split graph counts mismatch")
    _require(manifest.get("split_feature_counts") == {
        name: number * config["features"] for name, number in split_counts.items()
    }, "source split realization counts mismatch")
    _require(contract.get("data_fraction") == contract.get("sampling_ratio") == 1
             and contract.get("path_sampling") is False, "source used a graph/path subset")
    _require(contract.get("feature_realizations_parallel") == config["features"]
             and contract.get("seed_replicas_parallel") == len(config["model_seeds"]),
             "source field or seed axis contract mismatch")
    _require(contract.get("feature_channels") == 1 and contract.get("gnn_backbone") is None
             and contract.get("hidden_layers") == 1,
             "source is not the independent scalar operator experiment")
    _require(contract.get("precision") == config["dtype"]
             and contract.get("teacher_reference_precision") == "float64",
             "source precision mismatch")
    _require(contract.get("test_updates") == 0
             and contract.get("C2_labels_in_training_loss") is False
             and contract.get("mean_C2") == 1
             and contract.get("real_dataset_classification") is False,
             "source training/normalization contract mismatch")
    for field, expected in (("metric_rows", expected_metrics),
                            ("intervention_rows", expected_controls)):
        _require(contract.get(field) == completion.get(field) == expected,
                 f"source {field} is incomplete")
    _require(manifest.get("graph_content_unique") is True
             and manifest.get("feature_content_unique") is True,
             "source claims duplicate graph or feature content")
    return specs


def _tensor(archive: Any, name: str, shape: tuple[int, ...], floating: bool) -> torch.Tensor:
    array = archive[name]
    expected = np.dtype(np.float64 if floating else np.int64)
    _require(array.dtype == expected and array.shape == shape,
             f"source array shape/dtype mismatch: {name}")
    if floating:
        _require(bool(np.isfinite(array).all()), f"nonfinite source array: {name}")
    return torch.from_numpy(np.array(array, dtype=expected, order="C", copy=True))


def _targets(case: RuleCase, x: torch.Tensor, teacher: dict) -> dict[str, torch.Tensor]:
    i, j, k = case.wedges
    g1, g2 = x[j] - x[i], x[k] - x[j]
    with torch.no_grad():
        c = teacher_weights(g1, g2, torch.zeros(len(i), dtype=torch.long), 1, **teacher)
        lx = laplacian_apply(case.edges, x)
        result = {
            "teacher_c": c,
            "lx": lx,
            "l2x": laplacian_apply(case.edges, lx),
            "qx": fixed_wedge_apply(case.wedges, x),
            "target_path": wedge_transpose(case.wedges, c * (g2 - g1), case.num_nodes),
        }
    for name, tensor in result.items():
        _require(bool(torch.isfinite(tensor).all()), f"nonfinite {name}: {case.graph_id}")
    return result


def _validate_case(case: RuleCase, row: dict, spec: Any, teacher: dict) -> None:
    label, n = case.graph_id, case.num_nodes
    _require((label, case.family, case.split, n, case.feature_seed, case.pair_seed) == (
        spec.graph_id, spec.family, spec.split, spec.num_nodes,
        spec.graph_spec.feature_seed, spec.pair_seed,
    ), f"source case identity/seed/order mismatch: {label}")
    edge = case.edges
    _require(bool(((edge >= 0) & (edge < n)).all()) and bool((edge[0] < edge[1]).all()),
             f"source physical edges are invalid: {label}")
    keys = edge[0] * n + edge[1]
    _require(bool((keys[1:] > keys[:-1]).all()), f"edges are not canonical/unique: {label}")
    expected_wedges = build_wedges(edge, n)
    _require(torch.equal(expected_wedges, case.wedges),
             f"source omitted/reordered/invalid unordered wedges: {label}")
    p, e = case.wedges.shape[1], edge.shape[1]
    pairs, coeff = case.pair_edges, case.pair_coefficients
    _require(bool(((pairs >= 0) & (pairs < e)).all()) and bool((pairs[0] < pairs[1]).all()),
             f"source physical-edge pairs are invalid: {label}")
    _require(len(torch.unique(pairs[0] * e + pairs[1])) == p,
             f"source physical-edge pairs are duplicated: {label}")
    if p:
        first, second = edge[:, pairs[0]], edge[:, pairs[1]]
        dot = ((first[0] == second[0]).long() + (first[1] == second[1]).long()
               - (first[0] == second[1]).long() - (first[1] == second[0]).long())
        norm2 = 2 * coeff.square().sum(0) - 2 * coeff[0] * coeff[1] * dot
        _require(torch.allclose(coeff[0].abs(), coeff[1].abs(), rtol=1e-12, atol=1e-12)
                 and torch.allclose(norm2, torch.full_like(norm2, 6), rtol=1e-12, atol=1e-12),
                 f"source random-pair row normalization is invalid: {label}")
    _require(graph_content_hash(case) == row.get("graph_content_hash"),
             f"source graph content hash mismatch: {label}")
    _require(feature_content_hash(case) == row.get("feature_content_hash"),
             f"source feature content hash mismatch: {label}")
    metadata = case.metadata
    _require(metadata.get("teacher") == teacher and metadata.get("pair_count") == p
             and metadata.get("original_graph_seed") == spec.original_graph_seed,
             f"source teacher/geometry/seed metadata mismatch: {label}")
    attempt = _integer(metadata.get("resample_attempt"), "source resample_attempt")
    _require(attempt <= metadata.get("duplicate_max_attempts", -1) == 256,
             f"source topology retry metadata is invalid: {label}")
    if attempt == 0:
        _require(case.graph_seed == spec.original_graph_seed,
                 f"source graph seed mismatch: {label}")
    else:
        payload = (f"wedge-rule-data-v1\0{spec.original_graph_seed}\0topology-retry\0"
                   f"{label}:{attempt}").encode()
        _require(case.graph_seed == int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"),
                 f"source topology retry seed mismatch: {label}")
    for name, expected in _targets(case, case.features, teacher).items():
        _require(torch.allclose(getattr(case, name), expected, rtol=1e-11, atol=1e-11),
                 f"source teacher/operator target inconsistent with equations: {label}/{name}")


def _validate_csv(path: Path, cases: list[RuleCase], config: dict, kind: str) -> None:
    by_id = {case.graph_id: case for case in cases}
    evaluation = [case for case in cases if case.split != "train"]
    if kind == "metrics":
        expected = {
            (case.graph_id, target, condition, seed)
            for case in evaluation for target in _TARGETS for condition in _CONDITIONS
            for seed in ([-1] if condition in _BASELINES else config["model_seeds"])
        }
    elif kind == "interventions":
        expected = {
            (case.graph_id, target, "learned", seed, intervention)
            for case in evaluation for target in _TARGETS for seed in config["model_seeds"]
            for intervention in _INTERVENTIONS
        }
    else:
        expected = {(case.graph_id,) for case in cases}
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            graph_id = row.get("graph_id")
            _require(graph_id in by_id, f"unknown source CSV graph: {path.name}/{graph_id}")
            case = by_id[graph_id]
            _require((row.get("family"), row.get("split")) == (case.family, case.split),
                     f"source CSV family/split mismatch: {path.name}/{graph_id}")
            for field, value in (("num_nodes", case.num_nodes),
                                 ("num_edges", case.edges.shape[1]),
                                 ("num_paths", case.wedges.shape[1])):
                _require(int(row[field]) == value, f"source CSV {field} mismatch: {graph_id}")
            if kind == "audit":
                _require(int(row["num_features"]) == config["features"],
                         "source audit realization count mismatch")
                key = (graph_id,)
            else:
                _require(int(row["num_realizations"]) == config["features"],
                         "source CSV realization count mismatch")
                key = (graph_id, row["target"], row["condition"], int(row["seed"]))
                for field in ("message_relerr", "message_abs_rmse"):
                    _require(math.isfinite(float(row[field])) and float(row[field]) >= 0,
                             f"nonfinite/negative source metric: {graph_id}/{field}")
                if kind == "interventions":
                    key += (row["intervention"],)
            _require(key in expected and key not in seen,
                     f"source CSV has duplicate/unexpected key: {path.name}/{key}")
            seen.add(key)
    _require(seen == expected, f"source CSV coverage incomplete: {path.name}")


def load_source_run(run_dir: Path | str, profile: str,
                    workers: int | None = None) -> SourceRun:
    """Validate and reconstruct the entire completed source, without resampling.

    This reads checkpoints only as bytes for hashes. The frozen-model loader
    separately checks checkpoint payloads/state dictionaries before evaluation.
    """
    directory = Path(run_dir).resolve()
    _require(directory.is_dir(), f"source run directory missing: {directory}")
    _require(not (directory / "failure.json").exists(), "source contains a failure marker")
    json_names = ("config.json", "contract.json", "completion.json", "data_manifest.json")
    documents = {name: _json(_source_path(directory, name)) for name in json_names}
    config, contract, completion, manifest = (documents[name] for name in json_names)
    specs = _validate_contract(config, contract, completion, manifest, profile)
    affinity = len(psutil.Process().cpu_affinity())
    _require(affinity > 0, "current process has no usable CPU affinity")
    if workers is None:
        # Reuse the source's measured preprocessing choice within this host's
        # allowed affinity. It is recorded separately from the immutable source.
        workers = min(_integer(contract.get("cpu_workers"), "source cpu_workers", 1), affinity)
    else:
        workers = _integer(workers, "source validation workers", 1)
        _require(workers <= affinity, "source validation workers exceed current CPU affinity")
    dataset_path = _source_path(directory, manifest.get("dataset_file"))
    _require(file_hash(dataset_path) == manifest.get("dataset_sha256"),
             "source dataset SHA256 mismatch")
    rows = manifest.get("cases")
    _require(isinstance(rows, list) and len(rows) == len(specs),
             "source case manifest count mismatch")
    cases = []
    expected_arrays = {f"{spec.graph_id}__{field}" for spec in specs for field in _ARRAY_FIELDS}
    with np.load(dataset_path, allow_pickle=False) as archive:
        _require(len(archive.files) == len(expected_arrays)
                 and set(archive.files) == expected_arrays,
                 "source NPZ array coverage mismatch")
        for row, spec in zip(rows, specs, strict=True):
            graph_id = row.get("graph_id")
            _require(graph_id == spec.graph_id, "source graph IDs/order mismatch")
            n = _integer(row.get("num_nodes"), "num_nodes", 1)
            e = _integer(row.get("num_edges"), "num_edges")
            p = _integer(row.get("num_paths"), "num_paths")
            r = _integer(row.get("num_features"), "num_features", 1)
            _require(r == config["features"], "source realization axis mismatch")
            shapes = {"edges": (2, e), "wedges": (3, p), "pair_edges": (2, p),
                      "pair_coefficients": (2, p), "teacher_c": (p, r),
                      "pair_donor_path": (p,)}
            integer_fields = ("edges", "wedges", "pair_edges", "pair_donor_path")
            values = {name: _tensor(archive, f"{graph_id}__{name}", shapes.get(name, (n, r)),
                                    name not in integer_fields)
                      for name in _ARRAY_FIELDS}
            _require(torch.equal(values.pop("pair_donor_path"), torch.arange(p)),
                     f"source random-pair donor correspondence mismatch: {graph_id}")
            case = RuleCase(
                graph_id=graph_id, family=row["family"], split=row["split"], num_nodes=n,
                graph_seed=_integer(row["graph_seed"], "graph_seed"),
                feature_seed=_integer(row["feature_seed"], "feature_seed"),
                pair_seed=_integer(row["pair_seed"], "pair_seed"),
                metadata=copy.deepcopy(row["metadata"]), **values,
            )
            cases.append(case)
    validation_args = list(zip(cases, rows, specs, strict=True))

    def validate(arguments: tuple) -> None:
        case, row, spec = arguments
        _validate_case(case, row, spec, config["teacher"])

    if workers == 1:
        for arguments in validation_args:
            validate(arguments)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(validate, validation_args))
    _require(len({graph_content_hash(case) for case in cases}) == len(cases),
             "source exact topology content leaks across graph IDs/splits")
    _require(len({feature_content_hash(case) for case in cases}) == len(cases),
             "source exact feature content leaks across graph IDs/splits")
    by_id = {case.graph_id: case for case in cases}
    for case in cases:
        if case.family == "tree_chord":
            tree_id = case.graph_id.replace("-tree_chord-", "-tree-")
            tree = by_id[tree_id]
            _require(case.metadata.get("base_tree_seed") == tree.graph_seed
                     and case.metadata.get("base_tree_graph_id") == tree_id,
                     f"source paired tree seed mismatch: {case.graph_id}")
            tree_edges, chord_edges = set(map(tuple, tree.edges.T.tolist())), \
                set(map(tuple, case.edges.T.tolist()))
            _require(tree_edges < chord_edges
                     and len(chord_edges - tree_edges) == case.num_nodes // 4,
                     f"source tree_chord does not extend its matching tree: {case.graph_id}")
    _require(manifest.get("duplicate_resample_total") == sum(
        case.metadata["resample_attempt"] for case in cases), "source retry total mismatch")
    required = [*json_names, dataset_path.relative_to(directory).as_posix(),
                "metrics.csv", "interventions.csv", "teacher_operator_audit.csv"]
    for target in _TARGETS:
        for condition in _BASELINES:
            name = f"{target}-{condition}-fit.json"
            fit = _json(_source_path(directory, name))
            fields = {"u", "v"} if condition == "polynomial" else \
                {"u"} if condition == "first" else {"beta"}
            _require(set(fit) == fields and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in fit.values()), f"invalid scalar fit: {name}")
            required.append(name)
        required.extend(f"checkpoints/{target}-{condition}-selected.pt" for condition in _GATES)
    for name, kind in (("metrics.csv", "metrics"), ("interventions.csv", "interventions"),
                       ("teacher_operator_audit.csv", "audit")):
        _validate_csv(_source_path(directory, name), cases, config, kind)
    for optional in ("training.csv", "intervention_manifest.json"):
        if (directory / optional).is_file():
            required.append(optional)
    hashes = {name: file_hash(_source_path(directory, name)) for name in required}
    return SourceRun(directory, config, contract, completion, manifest, cases, hashes, workers)


def _fresh_seed(master_seed: int, graph_id: str) -> int:
    payload = f"wedge-generalization-v1\0{master_seed}\0feature\0{graph_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def feature_cases(source_cases: list[RuleCase], teacher: dict[str, Any], master_seed: int,
                  amplitude: float, workers: int) -> list[RuleCase]:
    """Reuse every original graph and replace only its independent scalar fields.

    Work is parallel across CPU cases; ordering and random streams do not depend
    on worker count. Geometry tensors are shared read-only, and metadata is
    copied. The full source list is returned; no node, path or graph is capped.
    """
    _integer(master_seed, "master_seed")
    count = _integer(workers, "workers", 1)
    amplitude = float(amplitude)
    _require(math.isfinite(amplitude) and amplitude > 0, "amplitude must be finite and positive")
    _require(bool(source_cases), "source_cases must be nonempty")
    _require(len({case.graph_id for case in source_cases}) == len(source_cases),
             "source_cases contains duplicate graph IDs")
    source_hashes = {feature_content_hash(case) for case in source_cases}
    source_seeds = {case.feature_seed for case in source_cases}

    def prepare(case: RuleCase) -> RuleCase:
        seed = _fresh_seed(master_seed, case.graph_id)
        _require(seed not in source_seeds, f"fresh/source feature seed collision: {case.graph_id}")
        _require(case.metadata.get("teacher") == teacher,
                 f"fresh teacher differs from frozen source teacher: {case.graph_id}")
        base = torch.from_numpy(np.random.default_rng(seed).standard_normal(case.features.shape))
        base_hash = feature_content_hash(replace(case, features=base))
        _require(base_hash not in source_hashes,
                 f"fresh/source base feature collision: {case.graph_id}")
        x = base * amplitude
        updated = _targets(case, x, teacher)
        metadata = copy.deepcopy(case.metadata)
        metadata.update({
            "feature_status": "fresh", "amplitude": amplitude,
            "source_split": case.split, "source_feature_seed": case.feature_seed,
            "source_feature_hash": feature_content_hash(case),
            "fresh_feature_seed": seed, "fresh_base_feature_hash": base_hash,
            "generalization_master_seed": int(master_seed),
            "feature_seed_salt": "wedge-generalization-v1",
            "feature_distribution": "standard_normal",
        })
        result = replace(case, features=x, feature_seed=seed, metadata=metadata, **updated)
        _require(feature_content_hash(result) not in source_hashes,
                 f"fresh/source scaled feature collision: {case.graph_id}")
        return result

    if count == 1:
        result = [prepare(case) for case in source_cases]
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            result = list(executor.map(prepare, source_cases))
    _require(len({feature_content_hash(case) for case in result}) == len(result),
             "fresh exact feature content collision between graph IDs")
    return result
