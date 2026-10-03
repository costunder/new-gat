"""Read complete, portable Experiment 4.1 scalar artifacts without model execution."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from ..branch_strength.contract import read_config, variants, verify_coverage
from ..branch_strength.report import _validate

TABLE_FILES = {
    "baseline_rows": "baseline_metrics.csv",
    "treatment_rows": "treatments.csv",
    "diagnostic_rows": "layer_diagnostics.csv",
    "fixed_z_rows": "fixed_z.csv",
    "resource_rows": "resources.csv",
    "replay_rows": "source_replay.csv",
}
JSON_FILES = (
    "config.json",
    "contract.json",
    "completion.json",
    "source.json",
    "coverage.json",
    "frozen_model_provenance.json",
    "source_run.json",
)
REQUIRED_FILES = (*JSON_FILES, *TABLE_FILES.values())
_TEXT_FIELDS = {
    "dataset",
    "condition",
    "split",
    "source",
    "treatment",
    "target",
    "scope",
    "device",
    "status",
}
_INTEGER_FIELDS = {
    "seed",
    "layer",
    "manifest_index",
    "num_nodes",
    "num_labeled_nodes",
    "packed_runs",
    "path_chunk",
    "all_paths",
    "parameters_per_seed",
    "completed_seed_cases",
    "peak_vram_bytes",
    "process_rss_bytes",
    "ram_available_bytes",
    "kappa_max_node",
    "kappa_active_node_count",
}
_BOOLEAN_FIELDS = {"measured", "within_declared_tolerance"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_INTEGER = re.compile(r"[+-]?\d+\Z")


@dataclass(frozen=True)
class SourceRun:
    directory: Path
    config: dict
    contract: dict
    completion: dict
    tables: dict[str, list[dict]]
    hashes: dict[str, str]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(items):
    values = {}
    for key, value in items:
        if key in values:
            raise ValueError(f"duplicate JSON key: {key}")
        values[key] = value
    return values


def _constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
    )


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _zero(record, key):
    if type(record.get(key)) is not int or record[key] != 0:
        raise ValueError(f"frozen scalar analysis requires integer {key}=0")


def _cell(field, value):
    if value == "":
        return None
    if field in _TEXT_FIELDS:
        return value
    if field in _BOOLEAN_FIELDS or field.endswith("_defined"):
        if value not in ("True", "False"):
            raise ValueError(f"{field} must be an explicit boolean")
        return value == "True"
    if field in _INTEGER_FIELDS:
        if not _INTEGER.fullmatch(value):
            raise ValueError(f"{field} must be an integer")
        return int(value)
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    _number(parsed, field)
    return parsed


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or any(not field for field in header) or len(header) != len(set(header)):
            raise ValueError(f"empty or duplicate CSV header: {path}")
        rows = []
        for line, cells in enumerate(reader, 2):
            if len(cells) != len(header):
                raise ValueError(f"CSV column count differs: {path}:{line}")
            rows.append(
                {field: _cell(field, value) for field, value in zip(header, cells, strict=True)}
            )
    if not rows:
        raise ValueError(f"empty CSV table: {path}")
    return rows


def _verify_code(record):
    folder = Path(__file__).parents[1] / "branch_strength"
    current = {
        path.name: _sha(path) for path in sorted([*folder.glob("*.py"), *folder.glob("*.json")])
    }
    if record.get("sha256") != current or record.get("code_digest") != _digest(current):
        raise ValueError("recorded Experiment 4.1 code/config hashes differ from preserved source")


def _verify_training(config, contract, source_run):
    profile = config["profile"]
    canonical = _json(Path(__file__).parents[1] / "classification" / f"config_{profile}.json")
    if contract.get("source_config") != canonical:
        raise ValueError(
            "recorded source training configuration differs from canonical Experiment 4"
        )
    completion = source_run["completion"]
    expected_scope = (
        "full_public_split_classification" if profile == "full" else "DEBUG_pipeline_only"
    )
    if (
        completion.get("completed") is not True
        or completion.get("profile") != profile
        or completion.get("scope") != expected_scope
        or completion.get("actual_data") is not (profile == "full")
        or completion.get("code_and_graphs_preserved") is not True
        or contract.get("source_training_completion") != completion
        or contract.get("source_run") != source_run.get("directory")
    ):
        raise ValueError("source training completion/provenance is inconsistent")
    _zero(completion, "frozen_model_updates")
    coverage, training = completion["coverage"], canonical["training"]
    for key in ("tuning_runs", "final_runs", "total_runs"):
        if coverage.get(key) != training[key]:
            raise ValueError(f"source training coverage differs: {key}")
    if (
        coverage.get("primary_metric_rows") != config["expected_counts"]["baseline_rows"]
        or coverage.get("contract_optimizer_updates") != training["total_updates"]
        or coverage.get("all_datasets_conditions_seeds_splits") is not True
    ):
        raise ValueError("source training coverage is incomplete")
    actual_updates = completion.get("new_optimizer_updates")
    if type(actual_updates) is not int or not 0 <= actual_updates <= training["total_updates"]:
        raise ValueError("invalid source training execution update count")
    hashes = source_run["artifact_hashes"]
    if not hashes or any(not _SHA.fullmatch(value) for value in hashes.values()):
        raise ValueError("source training artifact hashes are malformed")
    if contract.get("source_hash_digest") != _digest(hashes):
        raise ValueError("source training artifact digest differs")
    record = source_run["source"]
    if record.get("code_digest") != _digest(record["sha256"]):
        raise ValueError("source training code digest is inconsistent")


def _verify_nodes(config, contract, tables):
    shapes = contract["source_config"]["data"]["expected_shapes"]
    for row in tables["baseline_rows"] + tables["treatment_rows"]:
        expected = shapes[row["dataset"]]
        if (
            row.get("num_nodes") != expected["nodes"]
            or row.get("num_labeled_nodes") != expected[row["split"]]
        ):
            raise ValueError("metric graph/labeled-node counts differ from whole public split")
    if config["sampling_ratio"] != 1.0:
        raise ValueError("whole source graph contract required")


def _verify_diagnostics(rows):
    required = (
        "alpha",
        "beta",
        "kappa_reference",
        "kappa_used",
        "z_norm",
        "l_norm",
        "branch_norm",
        "reference_branch_norm",
        "alpha_l_norm",
        "beta_t_norm",
        "reference_beta_t_norm",
        "alpha_l_to_input",
        "beta_t_to_input",
        "beta_t_to_alpha_l",
        "normalization_gain",
    )
    for row in rows:
        if any(name not in row for name in required):
            raise ValueError("missing branch diagnostic needed for scalar decomposition")
        for field, value in row.items():
            if field.endswith("_defined") and field != "kappa_ratio_defined":
                measurement = field.removesuffix("_defined")
                if value is not (row.get(measurement) is not None):
                    raise ValueError(f"undefined diagnostic flag disagrees: {field}")
            if (
                value is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                if "cosine" in field or field in (
                    "first_second_cosine",
                    "second_input_cosine",
                    "message_reference_cosine",
                ):
                    if abs(value) > 1.00001:
                        raise ValueError(f"cosine outside numerical [-1,1] range: {field}")
                elif field.endswith("_norm") or field in required or field.startswith("c_"):
                    _number(value, field, nonnegative=True)
        for name in ("alpha", "beta"):
            _number(row[name], name, nonnegative=True)
            if row[name] > 1.00001:
                raise ValueError("branch coefficient outside [0,1]")
        for name in ("kappa_reference", "kappa_used"):
            if row[name] is None or row[name] <= 0:
                raise ValueError("positive reference/used kappa required")
        for name in (
            "z_norm",
            "l_norm",
            "branch_norm",
            "reference_branch_norm",
            "alpha_l_norm",
            "beta_t_norm",
        ):
            _number(row[name], name, nonnegative=True)
        flag = row.get("kappa_ratio_defined")
        values = [
            row.get("kappa_ratio_" + suffix) for suffix in ("mean", "p50", "p90", "p99", "max")
        ]
        if flag is True and any(value is None for value in values):
            raise ValueError("defined kappa distribution contains missing values")
        if flag is not True and any(value is not None for value in values):
            raise ValueError("undefined kappa distribution contains values")


def _verify_replay(config, tables):
    keys = ("dataset", "condition", "seed", "split")
    baseline = {tuple(row[key] for key in keys): row for row in tables["baseline_rows"]}
    seen = set()
    tolerance = config["source_replay"]
    for row in tables["replay_rows"]:
        key = tuple(row[name] for name in keys)
        if key in seen or key not in baseline or row.get("within_declared_tolerance") is not True:
            raise ValueError("duplicate, missing-pair or failed baseline replay")
        seen.add(key)
        for field in ("ce_abs_error", "accuracy_abs_error"):
            _number(row.get(field), field, nonnegative=True)
        # The replay table records abs error rather than original CE. This is the
        # largest compatible tolerance; original CE lies within replayed CE +/- error.
        allowed = tolerance["ce_atol"] + tolerance["ce_rtol"] * (
            baseline[key]["ce"] + row["ce_abs_error"]
        )
        if row["ce_abs_error"] > allowed or row["accuracy_abs_error"] > tolerance["accuracy_atol"]:
            raise ValueError("baseline replay exceeds declared tolerance")
    if seen != set(baseline):
        raise ValueError("baseline replay coverage incomplete")


def _verify_provenance(config, provenance, source_run):
    expected = set(
        itertools.product(config["data"]["datasets"], config["conditions"], config["final_seeds"])
    )
    seen, cases = set(), 0
    source_prefix = source_run["directory"].replace("\\", "/").rstrip("/") + "/"
    hashes = source_run["artifact_hashes"]
    for row in provenance:
        before, after = row.get("model_hash_before"), row.get("model_hash_after")
        if not isinstance(before, str) or not _SHA.fullmatch(before) or before != after:
            raise ValueError("frozen model hash changed or is malformed")
        _zero(row, "optimizer_updates")
        _zero(row, "trainable_parameters")
        seeds = row.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError("frozen model provenance requires explicit seed pack")
        for seed in seeds:
            key = row["dataset"], row["condition"], seed
            if key in seen or key not in expected:
                raise ValueError(
                    "frozen model provenance duplicates or changes selected seed identity"
                )
            seen.add(key)
        count = len(variants(config)) if row["condition"].startswith("learned_wedge") else 1
        if row.get("completed_seed_cases") != count * len(seeds):
            raise ValueError(
                "frozen model provenance case count differs from complete treatment scope"
            )
        cases += row["completed_seed_cases"]
        checkpoints = row.get("source_checkpoint_hashes")
        if not isinstance(checkpoints, dict) or not checkpoints:
            raise ValueError("frozen model source checkpoint hashes required")
        for path, sha in checkpoints.items():
            normal = path.replace("\\", "/")
            relative = normal.removeprefix(source_prefix)
            if normal == relative or hashes.get(relative) != sha:
                raise ValueError(
                    "recorded checkpoint hash is not in preserved source artifact manifest"
                )
    if seen != expected or cases != config["expected_counts"]["end_to_end_seed_cases"]:
        raise ValueError("frozen model provenance coverage is incomplete")


def load_run(directory):
    """Validate every saved scalar case; never read checkpoints or initialize CUDA."""
    directory = Path(directory).expanduser().resolve()
    hashes = {name: _sha(directory / name) for name in REQUIRED_FILES}
    records = {name: _json(directory / name) for name in JSON_FILES}
    profile = records["config.json"].get("profile")
    config = read_config(directory / "config.json", profile)
    contract, completion = records["contract.json"], records["completion.json"]
    expected = {**config["expected_counts"], "complete": True}
    if (
        completion.get("completed") is not True
        or completion.get("profile") != profile
        or completion.get("scope") != "frozen_branch_strength_diagnostic"
        or completion.get("actual_data") is not (profile == "full")
        or contract.get("actual_data") is not (profile == "full")
        or completion.get("coverage") != expected
        or contract.get("coverage") != expected
        or records["coverage.json"] != expected
    ):
        raise ValueError("completed full/DEBUG frozen diagnostic coverage required")
    for record in (contract, completion):
        _zero(record, "optimizer_updates")
        if record.get("source_files_and_models_preserved") is not True:
            raise ValueError("preserved source files/models guard required")
    if (
        contract.get("interpretation")
        != "posthoc_fixed_checkpoint_diagnostic_no_treatment_selection"
    ):
        raise ValueError("frozen post-hoc diagnostic interpretation required")
    _number(completion.get("seconds"), "completion seconds", nonnegative=True)
    _verify_code(records["source.json"])
    if contract.get("diagnostic_source") != records["source.json"]:
        raise ValueError("recorded diagnostic source provenance differs")
    source_run = records["source_run.json"]
    _verify_training(config, contract, source_run)
    tables = {key: _read_csv(directory / name) for key, name in TABLE_FILES.items()}
    four = [
        tables[name]
        for name in ("baseline_rows", "treatment_rows", "diagnostic_rows", "fixed_z_rows")
    ]
    if verify_coverage(config, *four) != expected:
        raise ValueError("saved scalar coverage differs")
    _validate(config, *four, tables["resource_rows"], contract)
    _verify_nodes(config, contract, tables)
    _verify_diagnostics(tables["diagnostic_rows"] + tables["fixed_z_rows"])
    _verify_replay(config, tables)
    _verify_provenance(config, records["frozen_model_provenance.json"], source_run)
    source = SourceRun(directory, config, contract, completion, tables, hashes)
    assert_unchanged(source)
    return source


def assert_unchanged(source):
    """Detect source artifact edits, replacement or deletion during analysis."""
    for name, recorded in source.hashes.items():
        if _sha(source.directory / name) != recorded:
            raise ValueError(f"saved Experiment 4.1 artifact changed during analysis: {name}")
