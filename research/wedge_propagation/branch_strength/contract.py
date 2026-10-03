"""Complete frozen treatment coverage, separate from training model sources."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

from ..classification.common import digest, file_sha256, read_json
from ..classification.common import source_manifest as classification_source_manifest
from ..classification.model import CONDITIONS

TREATMENTS = (
    "baseline",
    "true_c_unit_denominator",
    "identity_hold_reference",
    "identity_unit_denominator",
    "identity_norm_matched",
    "branch_off",
    "shuffle_hold_reference",
    "shuffle_norm_matched",
)
TARGETS = ("layer_0", "layer_1", "both")


def variants(config, include_baseline=True):
    rows = [{"treatment": "baseline", "target": "none", "manifest_index": -1}]
    for target in TARGETS:
        for treatment in TREATMENTS[1:]:
            indices = (
                range(config["manifests_per_dataset"]) if treatment.startswith("shuffle") else (-1,)
            )
            rows += [
                {"treatment": treatment, "target": target, "manifest_index": index}
                for index in indices
            ]
    return rows if include_baseline else rows[1:]


def fixed_variants(config):
    return [row for row in variants(config) if row["target"] in ("none", "both")]


def expected_counts(config):
    all_models = (
        len(config["data"]["datasets"]) * len(config["conditions"]) * len(config["final_seeds"])
    )
    learned = len(config["data"]["datasets"]) * 2 * len(config["final_seeds"])
    modified = learned * (len(variants(config)) - 1)
    return {
        "baseline_rows": 3 * all_models,
        "treatment_rows": 3 * modified,
        "diagnostic_rows": 2 * (all_models + modified),
        "fixed_z_rows": 2 * learned * len(fixed_variants(config)),
        "source_final_models": all_models,
        "learned_final_models": learned,
        "end_to_end_seed_cases": all_models + modified,
        "optimizer_updates": 0,
    }


def read_config(path, profile):
    config = read_json(path)
    if (
        config["profile"] != profile
        or config["conditions"] != list(CONDITIONS)
        or config["treatments"] != list(TREATMENTS)
        or config["targets"] != list(TARGETS)
    ):
        raise ValueError("explicit complete branch-strength contract required")
    expected_datasets = (
        ["Cora", "CiteSeer", "PubMed"]
        if profile == "full"
        else ["DEBUG-Cora", "DEBUG-CiteSeer", "DEBUG-PubMed"]
    )
    seeds = [11, 23, 37, 53, 71] if profile == "full" else [11, 23]
    manifests = 10 if profile == "full" else 2
    if profile not in ("full", "debug") or (
        config["data"]["datasets"] != expected_datasets
        or config["final_seeds"] != seeds
        or config["manifests_per_dataset"] != manifests
        or config["optimization_updates"] != 0
        or config["sampling_ratio"] != 1.0
        or config["precision"] != "float32"
    ):
        raise ValueError("whole source graphs/seeds/manifests and zero-update scope required")
    if config["expected_counts"] != expected_counts(config):
        raise ValueError("declared diagnostic coverage changed")
    reference = read_json(Path(__file__).with_name(f"config_{profile}.json"))
    if config != reference:
        raise ValueError("canonical frozen scope/runtime/replay/normalization contract changed")
    return config


def source_manifest():
    folder = Path(__file__).parent
    files = [*folder.glob("*.py"), *folder.glob("*.json")]
    hashes = {path.name: file_sha256(path) for path in sorted(files)}
    return {
        "sha256": hashes,
        "code_digest": digest(hashes),
        "git_commit": classification_source_manifest()["git_commit"],
    }


def verify_coverage(config, baseline_rows, treatment_rows, diagnostic_rows, fixed_z_rows):
    """Check every model, layer, treatment, scope, manifest and split exactly once."""
    base = list(
        itertools.product(config["data"]["datasets"], config["conditions"], config["final_seeds"])
    )
    learned = [key for key in base if key[1].startswith("learned_wedge")]
    changes, fixed = variants(config, False), fixed_variants(config)
    expected = {
        "baseline_rows": {
            (*key, split) for key in base for split in ("train", "validation", "test")
        },
        "treatment_rows": {
            (*key, v["treatment"], v["target"], v["manifest_index"], split)
            for key in learned
            for v in changes
            for split in ("train", "validation", "test")
        },
        "diagnostic_rows": {
            (*key, "baseline", "none", -1, layer) for key in base for layer in range(2)
        },
        "fixed_z_rows": {
            (*key, v["treatment"], f"layer_{layer}", v["manifest_index"], layer)
            for key in learned
            for v in fixed
            for layer in range(2)
        },
    }
    expected["diagnostic_rows"].update(
        (*key, v["treatment"], v["target"], v["manifest_index"], layer)
        for key in learned
        for v in changes
        for layer in range(2)
    )
    tables = {
        "baseline_rows": baseline_rows,
        "treatment_rows": treatment_rows,
        "diagnostic_rows": diagnostic_rows,
        "fixed_z_rows": fixed_z_rows,
    }
    for table, rows in tables.items():
        suffix = (
            ("split",)
            if table == "baseline_rows"
            else ("treatment", "target", "manifest_index", "split")
            if table == "treatment_rows"
            else ("treatment", "target", "manifest_index", "layer")
        )
        seen = set()
        for row in rows:
            key = tuple(row[field] for field in ("dataset", "condition", "seed", *suffix))
            if key in seen or key not in expected[table]:
                raise ValueError(f"duplicate/out-of-contract {table} row")
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError(f"nonfinite serialized {table} metric")
            seen.add(key)
        if seen != expected[table]:
            raise ValueError(f"incomplete {table} coverage: missing={len(expected[table] - seen)}")
    return {**expected_counts(config), "complete": True}
