"""Reuse pinned complete citation data; replace only the model geometry."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

from ...local_energy_relations.prediction import common as original_common
from ...local_energy_relations.prediction import data as original_data
from ..operators import CouplingGeometry, prepare_geometry
from .common import digest, file_sha256, read_json, validate_config, write_json

_TENSOR_FIELDS = ("x", "y", "train_mask", "val_mask", "test_mask", "edges")
_MODE_FIELDS = {"intra_weights", "intra_degree", "eta", "max_intra_degree", "mode", "metadata"}


@dataclass(frozen=True)
class ContextGraph:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edges: torch.Tensor
    geometries: dict[str, CouplingGeometry]
    metadata: dict[str, Any]
    _original: original_data.PredictionGraph

    @property
    def topology(self):
        return self.geometries["unit"].topology

    @property
    def geometry(self):
        return self.geometries["unit"]

    def geometry_for(self, mode):
        if mode not in self.geometries:
            raise ValueError("unknown fixed C geometry")
        return self.geometries[mode]

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def num_classes(self):
        return self.metadata["classes"]

    def to(self, device, dtype=torch.float32):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("classification precision must be float32/64")
        values = {name: getattr(self, name).to(device=device, dtype=dtype if getattr(self, name).is_floating_point() else getattr(self, name).dtype) for name in _TENSOR_FIELDS}
        first = self.geometries["unit"].to(device, dtype)
        other = self.geometries["local_degree"]
        coefficients = {}
        for field in fields(other):
            value = getattr(other, field.name)
            if field.name not in _MODE_FIELDS:
                value = getattr(first, field.name)
            elif isinstance(value, torch.Tensor):
                value = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            coefficients[field.name] = value
        return replace(self, **values, geometries={"unit": first, "local_degree": CouplingGeometry(**coefficients)})


def _convert(original):
    geometry = {mode: prepare_geometry(original.topology, mode) for mode in ("unit", "local_degree")}
    return ContextGraph(original.name, **{name: getattr(original, name) for name in _TENSOR_FIELDS}, geometries=geometry, metadata=dict(original.metadata), _original=original)


def compatible_input_config(config):
    """The legacy loader's model contract never becomes this model's contract."""
    validate_config(config)
    original = original_common.read_config(profile=config["profile"])
    for key in ("profile", "data_source", "data", "source"):
        if digest(config[key]) != digest(original[key]):
            raise ValueError(f"input adapter would change original {key}")
    if config["profile"] == "debug" and config["debug_fixture"] != original["debug_fixture"]:
        raise ValueError("DEBUG input fixture differs from prior source")
    for key in ("cpu_workers", "cpu_threads", "gpu_memory_safety_fraction", "relation_chunk_candidates"):
        original["runtime"][key] = config["runtime"][key]
    original_common.validate_config(original)
    return original


def prepare_datasets(config, data_root, output_dir, source_dir, workers="auto", download=True):
    reader = compatible_input_config(config)
    output = Path(output_dir).resolve()
    sidecar = output / "context_data_contract.json"
    if sidecar.exists():
        raise FileExistsError("context input references already exist; preserved")
    raw_graphs, manifest = original_data.prepare_datasets(reader, data_root, output, source_dir, workers=workers, download=download)
    graphs = {name: _convert(graph) for name, graph in raw_graphs.items()}
    record = {
        "schema": "local-context-classification-input-adapter-v1",
        "config_digest": digest(config), "unchanged_reader_config_digest": digest(reader),
        "original_data_manifest_sha256": file_sha256(output / "data_manifest.json"),
        "conserved_fields": ["profile", "data_source", "data", "source"],
        "prior_features_edges_labels_masks_topology_preserved": True,
        "legacy_prediction_model_not_used": True,
        "coupling_coefficients_cached_per_dataset_worker_not_per_epoch": True,
        "geometry": {name: {mode: geometry.metadata for mode, geometry in graph.geometries.items()} for name, graph in graphs.items()},
    }
    write_json(sidecar, record)
    manifest = dict(manifest)
    manifest["context_contract"] = {"path": str(sidecar), "sha256": file_sha256(sidecar), "config_digest": digest(config), "original_manifest_path": str(output / "data_manifest.json"), "original_manifest_sha256": record["original_data_manifest_sha256"]}
    write_json(output / "context_data_manifest.json", manifest)
    assert_data_unchanged(manifest)
    return graphs, manifest


def graph_content_digest(graph):
    if not isinstance(graph, ContextGraph):
        raise ValueError("ContextGraph required")
    original = replace(graph._original, **{name: getattr(graph, name) for name in _TENSOR_FIELDS}, metadata=dict(graph.metadata))
    return digest({"original_data": original_data.graph_content_digest(original), "coupling_rule": "canonical_all_shared_copy_links; fixed_unit_or_local_degree; graphwise_degree_steps"})


def save_graph(graph, path):
    # This numeric cache includes the exact original topology and its checked
    # reader coefficients. Coupling geometry is cached once after worker load.
    original = replace(graph._original, **{name: getattr(graph, name) for name in _TENSOR_FIELDS}, metadata=dict(graph.metadata))
    return original_data.save_graph(original, path)


def load_graph(path, expected_sha256=None):
    return _convert(original_data.load_graph(path, expected_sha256))


def assert_data_unchanged(manifest):
    original_data.assert_data_unchanged(manifest)
    record = manifest.get("context_contract")
    if not isinstance(record, dict) or file_sha256(record["path"]) != record["sha256"]:
        raise ValueError("context input contract missing or changed")
    if read_json(record["path"])["config_digest"] != record["config_digest"]:
        raise ValueError("context input contract identity changed")
    if file_sha256(record["original_manifest_path"]) != record["original_manifest_sha256"]:
        raise ValueError("original input reference manifest changed")


__all__ = ["ContextGraph", "prepare_datasets", "graph_content_digest", "save_graph", "load_graph", "assert_data_unchanged", "compatible_input_config"]
