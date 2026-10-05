"""Preserve pinned full citation data and cache all eligible relations once."""
from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import torch

from ...local_context_coupling.classification import common as reader_common, data as reader
from ...wedge_propagation.study import available_cpus,runtime_resources
from ..geometry import prepare_geometries
from .common import digest, file_sha256, read_json, write_json, validate_config


@dataclass(frozen=True)
class MetricGraph:
    _context: reader.ContextGraph
    geometries: dict
    gcn_edges: torch.Tensor
    gcn_weights: torch.Tensor

    def __getattr__(self, name):
        return getattr(self._context, name)

    @property
    def geometry(self):
        return self.geometries["unit"]

    @property
    def topology(self):
        return self.geometry.topology

    def geometry_for(self, recipe):
        if recipe not in self.geometries:
            raise ValueError("Unknown metric recipe")
        return self.geometries[recipe]

    def to(self, device, dtype=torch.float32):
        unit=self.geometries["unit"].to(device,dtype)
        geometries={"unit":unit,"local_degree":self.geometries["local_degree"].to(device,dtype,shared=unit)}
        # The old reader's geometry is retained on CPU for provenance only.
        # Transfer model inputs, then use the new metric topology exclusively.
        tensors={name:getattr(self._context,name).to(device=device,dtype=dtype if getattr(self._context,name).is_floating_point() else getattr(self._context,name).dtype)
                 for name in ("x","y","train_mask","val_mask","test_mask","edges")}
        return replace(self, _context=replace(self._context,**tensors),
                       geometries=geometries,
                       gcn_edges=self.gcn_edges.to(device), gcn_weights=self.gcn_weights.to(device, dtype))


def _convert(context):
    geometries = prepare_geometries(context.topology)
    loops = torch.arange(context.num_nodes, dtype=torch.long).repeat(2, 1)
    edges = torch.cat((context.edges, context.edges.flip(0), loops), dim=1)
    degree = torch.bincount(edges[1], minlength=context.num_nodes).to(torch.float64)
    weights = (degree[edges[0]] * degree[edges[1]]).rsqrt()
    return MetricGraph(context, geometries, edges, weights)


def compatible_input_config(config):
    validate_config(config)
    previous = reader_common.read_config(profile=config["profile"])
    for key in ("profile", "data_source", "data", "source"):
        if digest(config[key]) != digest(previous[key]):
            raise ValueError(f"Input adapter would change {key}")
    for key in ("cpu_workers", "cpu_threads", "gpu_memory_safety_fraction", "relation_chunk_candidates"):
        previous["runtime"][key] = config["runtime"][key]
    reader_common.validate_config(previous)
    return previous


def prepare_datasets(config, data_root, output_dir, source_dir, workers="auto", download=True):
    output = Path(output_dir)
    context, manifest = reader.prepare_datasets(compatible_input_config(config), data_root, output,
                                              source_dir, workers, download)
    geometry_workers=min(len(context),available_cpus(runtime_resources(torch.device("cpu")))) if workers=="auto" else min(len(context),int(workers))
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=geometry_workers) as pool:
        graphs=dict(zip(context,pool.map(_convert,context.values()),strict=True))
    record = {"schema": "edge-metric-input-v1", "config_digest": digest(config),
              "original_context_manifest_sha256": file_sha256(output / "context_data_manifest.json"),
              "all_nodes_edges_features_masks_preserved": True,
              "occurrence_correction": "one_over_sqrt_physical_occurrence_count",
              "all_eligible_pairs_preserved": True,
              "geometry_cpu_workers":geometry_workers,"geometry_seconds":time.perf_counter()-started,
              "recipe_independent_topology_and_GPU_storage_shared":True,
              "geometry": {name: {r: g.metadata for r, g in graph.geometries.items()} for name, graph in graphs.items()}}
    path = output / "edge_metric_data_contract.json"
    write_json(path, record)
    manifest = {**manifest, "edge_metric_contract": {"path": str(path.resolve()), "sha256": file_sha256(path)}}
    write_json(output / "edge_metric_data_manifest.json", manifest)
    return graphs, manifest


def graph_content_digest(graph):
    return digest({"original_data": reader.graph_content_digest(graph._context),
                   "geometry": {r: g.metadata for r, g in graph.geometries.items()},
                   "gcn": "symmetric_self_loop_one_no_duplicate_physical_edges"})


def save_graph(graph, path):
    return reader.save_graph(graph._context, path)


def load_graph(path, expected_sha256=None):
    return _convert(reader.load_graph(path, expected_sha256))


def assert_data_unchanged(manifest):
    reader.assert_data_unchanged(manifest)
    record = manifest["edge_metric_contract"]
    if file_sha256(record["path"]) != record["sha256"]:
        raise RuntimeError("Metric input contract changed")
