"""Reuse identical citation inputs and geometry through a checked contract adapter."""

from __future__ import annotations

from pathlib import Path

from ..prediction import common as original_common
from ..prediction import data as original_data
from ..prediction.data import (
    PredictionGraph,
    graph_content_digest,
    graph_from_arrays,
    load_graph,
    save_graph,
)
from .common import digest, file_sha256, read_json, validate_config, write_json

_BUDGET_FIELDS = {"tuning_runs", "final_runs", "total_runs", "total_updates"}
_CONSERVED_FIELDS = ("profile", "data_source", "data", "source", "backbone", "operators", "resources")


def compatible_prediction_config(config):
    """Prove that using the original loader preserves every data and per-run setting."""
    validate_config(config)
    original = original_common.read_config(profile=config["profile"])
    for key in _CONSERVED_FIELDS:
        if digest(config[key]) != digest(original[key]):
            raise ValueError(f"placement adapter would change original {key} contract")
    for key, value in original["training"].items():
        if key not in _BUDGET_FIELDS and digest(config["training"][key]) != digest(value):
            raise ValueError(f"placement adapter would change original per-run training {key}")
    if config["profile"] == "debug" and config["debug_fixture"] != original["debug_fixture"]:
        raise ValueError("placement adapter would change the original DEBUG fixture")
    original["runtime"] = dict(config["runtime"])
    original_common.validate_config(original)
    return original


def prepare_datasets(config, data_root, output_dir, source_dir, workers="auto", download=True):
    compatible = compatible_prediction_config(config)
    output = Path(output_dir).resolve()
    sidecar = output / "placement_data_contract.json"
    new_manifest = output / "placement_data_manifest.json"
    if sidecar.exists() or new_manifest.exists():
        raise FileExistsError("placement data provenance already exists; existing files preserved")
    graphs, original_manifest = original_data.prepare_datasets(
        compatible, data_root, output, source_dir, workers=workers, download=download
    )
    proof = {
        "schema": "local-energy-placement-data-contract-v1",
        "placement_config_digest": digest(config),
        "compatible_prediction_config_digest": digest(compatible),
        "original_data_manifest_digest": digest(original_manifest),
        "original_data_manifest_sha256": file_sha256(output / "data_manifest.json"),
        "conserved_fields": list(_CONSERVED_FIELDS),
        "per_run_training_unchanged": True,
        "original_prediction_code_unchanged": True,
        "only_model_branch_placement_and_factorial_budget_change": True,
        "classification_graph_content_digest": {
            name: graph_content_digest(graph) for name, graph in graphs.items()
        },
    }
    write_json(sidecar, proof)
    manifest = dict(original_manifest)
    manifest["placement_contract"] = {
        "path": str(sidecar), "sha256": file_sha256(sidecar), "digest": digest(proof),
        "original_data_manifest_path": str(output / "data_manifest.json"),
        "original_data_manifest_sha256": proof["original_data_manifest_sha256"],
        "placement_config_digest": proof["placement_config_digest"],
    }
    write_json(new_manifest, manifest)
    assert_data_unchanged(manifest)
    return graphs, manifest


def assert_data_unchanged(manifest):
    original_data.assert_data_unchanged(manifest)
    record = manifest.get("placement_contract")
    if not isinstance(record, dict):
        raise ValueError("placement data contract provenance is missing")
    if file_sha256(record["path"]) != record["sha256"]:
        raise ValueError("placement data contract checksum mismatch")
    proof = read_json(record["path"])
    if digest(proof) != record["digest"] or proof["placement_config_digest"] != record["placement_config_digest"]:
        raise ValueError("placement data contract identity changed")
    if file_sha256(record["original_data_manifest_path"]) != record["original_data_manifest_sha256"]:
        raise ValueError("original prediction data manifest changed")


__all__ = [
    "PredictionGraph", "prepare_datasets", "graph_from_arrays", "save_graph", "load_graph",
    "graph_content_digest", "assert_data_unchanged", "compatible_prediction_config",
]
