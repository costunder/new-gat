"""Locked data, model, learning and comparison contracts; measured allocations vary."""
from __future__ import annotations

import math
from pathlib import Path

from ..common import (
    cpu_state, digest, file_sha256, read_json, save_checkpoint, write_csv, write_json,
    source_manifest, assert_source_unchanged,
)

FOLDER = Path(__file__).resolve().parent
RECIPES = ("unit", "local_degree")
VARIANTS = ("D0", "D1", "F0", "F1", "F2", "DA")
CONDITIONS = tuple(f"{r}__{v}" for r in RECIPES for v in VARIANTS) + ("G1", "G2", "P2")
SCOPES = ("layer_0", "layer_1", "both")


def parse_condition(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown edge metric classification condition: {condition}")
    if condition in ("G1", "G2", "P2"):
        return "unit", "baseline", "none", condition
    recipe, variant = condition.split("__")
    return recipe, "edge_metric", "pair" if variant in ("F0", "F1", "F2") else "none", variant


def condition_metadata(condition):
    recipe, family, cross, variant = parse_condition(condition)
    return {"recipe": recipe, "operator_family": family, "cross": cross,
            "variant": variant, "energy_operator": "occurrence_corrected_edge_metric"
            if family == "edge_metric" else variant,
            "learned_diagonal": variant in ("D1", "F2", "DA"),
            "learned_pair": variant in ("F1", "F2", "DA")}


def validate_config(config, profile=None):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    profile = config.get("profile") if profile is None else profile
    if profile not in ("full", "debug"):
        raise ValueError("Explicit full/debug profile is required")
    expected = read_json(FOLDER / f"config_{profile}.json")
    if set(config) != set(expected):
        raise ValueError("Unknown/missing scientific contract field")
    for key in expected:
        if key != "runtime" and digest(config[key]) != digest(expected[key]):
            raise ValueError(f"Changed locked contract: {key}")
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("All fifteen declared conditions are required")
    runtime = config["runtime"]
    if set(runtime) != set(expected["runtime"]):
        raise ValueError("Unknown/missing runtime allocation")
    mutable = {"cpu_threads", "cpu_workers", "gpu_memory_safety_fraction", "relation_chunk_candidates"}
    for key in expected["runtime"]:
        if key not in mutable and runtime[key] != expected["runtime"][key]:
            raise ValueError(f"Changed locked runtime setting: {key}")
    for key in ("cpu_threads", "cpu_workers"):
        value = runtime[key]
        if value != "auto" and (type(value) is not int or value < 1):
            raise ValueError(f"{key} requires auto or a positive integer")
    fraction = runtime["gpu_memory_safety_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Invalid GPU safety fraction")
    chunks = runtime["relation_chunk_candidates"]
    if not chunks or len(set(chunks)) != len(chunks) or any(type(v) is not int or v < 1 for v in chunks):
        raise ValueError("Exact chunk candidates must be distinct positive integers")
    t = config["training"]
    cells = len(config["conditions"]) * len(config["data"]["datasets"])
    tuning = cells * len(t["tuning_seeds"]) * len(t["learning_rate_candidates"])
    final = cells * len(t["final_seeds"])
    if (tuning, final, tuning + final, (tuning + final) * t["epochs_per_run"]) != (t["tuning_runs"], t["final_runs"], t["total_runs"], t["total_updates"]):
        raise ValueError("Complete run/update budget differs")
    if profile == "full" and digest(expected) != digest(read_json(FOLDER / "design_contract.json")):
        raise ValueError("FULL differs from the recorded design contract")
    return config


def read_config(path=None, profile="full"):
    return validate_config(read_json(path or FOLDER / f"config_{profile}.json"), profile)
