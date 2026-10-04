"""Conserve the complete pinned data and cache normalized sparse coefficients."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import torch

from ..classification import common as legacy_common, data as legacy_data
from .common import (CROSS_POLICIES, INTRA_POLICIES, WEIGHTS, digest, file_sha256,
                     read_json, validate_config, write_json)
from .operators import prepare_normalized_geometry


@dataclass(frozen=True)
class NormalizationGraph:
    _context: legacy_data.ContextGraph
    geometries: dict

    def __getattr__(self, name):
        return getattr(self._context, name)

    @property
    def geometry(self):
        return self.geometry_for("unit", "graph", "graph")

    def geometry_for(self, mode, intra, cross):
        if cross == "none":
            cross = "graph"
        key = mode, intra, cross
        if key not in self.geometries:
            raise ValueError("unknown normalized C/intra/cross geometry")
        return self.geometries[key]

    def to(self, device, dtype=torch.float32):
        context = self._context.to(device, dtype)
        geometries = {key: geometry.to(device, dtype, base=context.geometry_for(key[0]))
                      for key, geometry in self.geometries.items()}
        return replace(self, _context=context, geometries=geometries)


def _convert(context):
    geometries = {(weight, intra, cross): prepare_normalized_geometry(
        context.geometry_for(weight), intra, cross)
        for weight in WEIGHTS for intra in INTRA_POLICIES for cross in CROSS_POLICIES}
    return NormalizationGraph(context, geometries)


def compatible_input_config(config):
    validate_config(config)
    legacy = legacy_common.read_config(profile=config["profile"])
    for key in ("profile", "data_source", "data", "source"):
        if digest(config[key]) != digest(legacy[key]):
            raise ValueError(f"normalization input adapter would change original {key}")
    if config["profile"] == "debug" and config["debug_fixture"] != legacy["debug_fixture"]:
        raise ValueError("DEBUG input fixture differs from prior source")
    for key in ("cpu_workers", "cpu_threads", "gpu_memory_safety_fraction", "relation_chunk_candidates"):
        legacy["runtime"][key] = config["runtime"][key]
    legacy_common.validate_config(legacy)
    return legacy


def prepare_datasets(config, data_root, output_dir, source_dir, workers="auto", download=True):
    reader = compatible_input_config(config)
    output = Path(output_dir).resolve()
    sidecar = output / "normalization_data_contract.json"
    if sidecar.exists():
        raise FileExistsError("normalization input references already exist; preserved")
    contexts, manifest = legacy_data.prepare_datasets(reader, data_root, output, source_dir,
                                                     workers=workers, download=download)
    started = time.perf_counter()
    graphs = {name: _convert(graph) for name, graph in contexts.items()}
    normalization_seconds = time.perf_counter() - started
    print(f"[normalization cache] datasets={len(graphs)} policies_per_dataset=8 "
          f"seconds={normalization_seconds:.3f} all_nodes_edges_retained=True", flush=True)
    record = {
        "schema": "local-context-normalization-input-adapter-v1",
        "config_digest": digest(config), "unchanged_reader_config_digest": digest(reader),
        "context_manifest_sha256": file_sha256(output / "context_data_manifest.json"),
        "prior_features_edges_labels_masks_topology_preserved": True,
        "legacy_model_not_used": True, "all_eight_geometries_cached_once_per_worker": True,
        "normalization_cpu_seconds": normalization_seconds,
        "geometry": {name: {"__".join(key): geometry.metadata
                            for key, geometry in graph.geometries.items()}
                     for name, graph in graphs.items()},
    }
    write_json(sidecar, record)
    manifest = dict(manifest)
    manifest["normalization_contract"] = {
        "path": str(sidecar), "sha256": file_sha256(sidecar), "config_digest": digest(config),
        "context_manifest_path": str(output / "context_data_manifest.json"),
        "context_manifest_sha256": record["context_manifest_sha256"],
    }
    write_json(output / "normalization_data_manifest.json", manifest)
    assert_data_unchanged(manifest)
    return graphs, manifest


def graph_content_digest(graph):
    if not isinstance(graph, NormalizationGraph):
        raise ValueError("NormalizationGraph required")
    import hashlib
    weights = {}
    for key, geometry in graph.geometries.items():
        weights["__".join(key)] = {name: hashlib.sha256(
            getattr(geometry, name).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name in ("s_weights", "g_weights", "s_degree", "g_degree", "eta_by_local")}
    return digest({"original_context": legacy_data.graph_content_digest(graph._context),
                   "normalization_coefficients": weights})


def save_graph(graph, path):
    return legacy_data.save_graph(graph._context, path)


def load_graph(path, expected_sha256=None):
    return _convert(legacy_data.load_graph(path, expected_sha256))


def assert_data_unchanged(manifest):
    legacy_data.assert_data_unchanged(manifest)
    record = manifest.get("normalization_contract")
    if not isinstance(record, dict) or file_sha256(record["path"]) != record["sha256"]:
        raise ValueError("normalization input contract missing or changed")
    if read_json(record["path"])["config_digest"] != record["config_digest"]:
        raise ValueError("normalization input contract identity changed")
    if file_sha256(record["context_manifest_path"]) != record["context_manifest_sha256"]:
        raise ValueError("original context manifest changed")
