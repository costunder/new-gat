"""Complete source-verified CPU inputs, with exclusive portable input snapshots.

Only generic graph generation and citation loading are reused. No wedge model,
teacher or learned checkpoint is imported into this independent energy audit.
Synthetic columns are independent scalar draws; citation columns form a vector.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from ..wedge_propagation.classification import data as citation_data
from ..wedge_propagation.data import GraphSpec, make_specs
from ..wedge_propagation.data import make_case as fixed_make_case
from .contract import digest, file_sha256, validate_config, write_json

_SCHEMA = "local-energy-relations-input-v1"
_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


@dataclass(frozen=True)
class AuditCase:
    graph_id: str
    family: str
    edges: torch.Tensor
    features: torch.Tensor
    actual_data: bool
    feature_mode: str
    metadata: dict[str, Any]

    @property
    def num_nodes(self):
        return self.features.shape[0]

    @property
    def num_features(self):
        return self.features.shape[1]

    @property
    def x(self):
        return self.features

    @property
    def is_real(self):
        return self.actual_data


def validate_case(case):
    if not isinstance(case, AuditCase) or not _ID.fullmatch(case.graph_id):
        raise ValueError("case requires a safe original graph identifier")
    x, edges = case.features, case.edges
    if (
        x.device.type != "cpu"
        or x.dtype != torch.float64
        or x.ndim != 2
        or min(x.shape) < 1
        or not bool(torch.isfinite(x).all())
    ):
        raise ValueError("complete CPU float64 finite features [N,F] required")
    if (
        edges.device.type != "cpu"
        or edges.dtype != torch.long
        or edges.ndim != 2
        or edges.shape[0] != 2
    ):
        raise ValueError("physical edges must be CPU Long[2,E]")
    value = edges.numpy()
    if value.size and (
        np.any(value < 0) or np.any(value >= case.num_nodes) or np.any(value[0] >= value[1])
    ):
        raise ValueError("physical edges must retain canonical distinct original endpoints")
    if value.shape[1] > 1:
        order = np.lexsort((value[1], value[0]))
        if not np.array_equal(order, np.arange(value.shape[1])) or np.any(
            np.all(value[:, 1:] == value[:, :-1], axis=0)
        ):
            raise ValueError("physical edges must be sorted unique original edges")
    if type(case.actual_data) is not bool or case.feature_mode not in (
        "independent_scalar_columns",
        "vector_trace",
    ):
        raise ValueError("actual-citation flag and scalar/vector feature interpretation required")
    if not isinstance(case.metadata, dict):
        raise ValueError("case metadata must be an object")
    if (
        "actual_citation_data" in case.metadata
        and case.metadata["actual_citation_data"] is not case.actual_data
    ):
        raise ValueError("case actual-citation metadata disagrees")
    if "feature_mode" in case.metadata and case.metadata["feature_mode"] != case.feature_mode:
        raise ValueError("case scalar/vector metadata disagrees")
    json.dumps(case.metadata, allow_nan=False)
    return case


def graph_content_hash(case):
    validate_case(case)
    payload = np.asarray([case.num_nodes], dtype="<i8").tobytes()
    payload += case.edges.numpy().astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def feature_content_hash(case):
    validate_case(case)
    payload = np.asarray(case.features.shape, dtype="<i8").tobytes()
    payload += case.features.numpy().astype("<f8", copy=False).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def make_synthetic_case(spec: GraphSpec):
    original = fixed_make_case(spec)
    return validate_case(
        AuditCase(
            original.graph_id,
            original.family,
            original.edges,
            original.features,
            False,
            "independent_scalar_columns",
            {
                "profile": spec.graph_id.split("-", 1)[0],
                "synthetic_graph": True,
                "actual_citation_data": False,
                "input_origin": "existing_fixed_graph_generator",
                "graph_seed": original.graph_seed,
                "feature_seed": original.feature_seed,
                "base_tree_seed": original.base_tree_seed,
                "feature_mode": "independent_scalar_columns",
                "sampling_ratio": 1.0,
                "encoder": None,
                "original_node_ids": "zero_based_original_graph_order",
            },
        )
    )


def _available_cpus():
    count = (
        len(psutil.Process().cpu_affinity())
        if hasattr(psutil.Process(), "cpu_affinity")
        else (os.cpu_count() or 1)
    )
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        limit, period = quota.read_text().split()
        if limit != "max":
            count = min(count, max(1, int(int(limit) / int(period))))
    return count


def _generate(specs, workers):
    if workers == 1:
        return [make_synthetic_case(spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(make_synthetic_case, specs))


def choose_workers(specs, requested="auto"):
    """Measure largest-size graph generation; final generation still uses every spec."""
    available = _available_cpus()
    if requested != "auto" and (type(requested) is not int or not 1 <= requested <= available):
        raise ValueError("CPU workers must be auto or fit actual affinity/quota")
    candidates = [n for n in (1, 2, 4, 8) if n <= available] if requested == "auto" else [requested]
    largest = max(spec.num_nodes for spec in specs)
    probes = [spec for spec in specs if spec.num_nodes == largest]
    trials, reference = [], None
    for workers in candidates:
        started = time.perf_counter()
        generated = _generate(probes, workers)
        elapsed = time.perf_counter() - started
        hashes = [(graph_content_hash(case), feature_content_hash(case)) for case in generated]
        if reference is None:
            reference = hashes
        elif hashes != reference:
            raise ValueError("parallel generation changed an input identity")
        trials.append(
            {
                "workers": workers,
                "graphs": len(probes),
                "nodes_per_graph": largest,
                "seconds": elapsed,
                "graphs_per_second": len(probes) / elapsed,
                "scope": "CPU_graph_and_feature_generation_calibration_not_final_subset",
            }
        )
        print(
            f"[CPU input calibration] workers={workers} graphs={len(probes)} seconds={elapsed:.4f}",
            flush=True,
        )
    return min(trials, key=lambda row: row["seconds"])["workers"], trials


def save_case(case, path):
    """Save every physical edge and channel, without labels, pickle or replacement."""
    validate_case(case)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": _SCHEMA,
        "graph_id": case.graph_id,
        "family": case.family,
        "actual_data": case.actual_data,
        "feature_mode": case.feature_mode,
        "metadata": case.metadata,
        "graph_sha256": graph_content_hash(case),
        "feature_sha256": feature_content_hash(case),
    }
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            edges=case.edges.numpy(),
            features=case.features.numpy(),
            original_node_ids=np.arange(case.num_nodes, dtype=np.int64),
            metadata=np.asarray(json.dumps(envelope, sort_keys=True, allow_nan=False)),
        )
    return {
        **{key: value for key, value in envelope.items() if key != "metadata"},
        "file": path.name,
        "sha256": file_sha256(path),
        "nodes": case.num_nodes,
        "physical_edges": case.edges.shape[1],
        "features": case.num_features,
        "input_tensor_shape": list(case.features.shape),
        "dtype": "float64",
        "all_nodes_edges_channels": True,
    }


def load_case(path, expected_sha256=None):
    """Validate a numeric input snapshot and its independent topology/feature hashes."""
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("input snapshot file checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if (
            set(archive.files) != {"edges", "features", "original_node_ids", "metadata"}
            or len(archive.files) != 4
        ):
            raise ValueError("input snapshot array coverage differs")
        record = json.loads(str(archive["metadata"].item()))
        if record.get("schema") != _SCHEMA:
            raise ValueError("input snapshot schema differs")
        if not np.array_equal(
            archive["original_node_ids"], np.arange(archive["features"].shape[0])
        ):
            raise ValueError("input snapshot original node correspondence differs")
        case = validate_case(
            AuditCase(
                record["graph_id"],
                record["family"],
                torch.from_numpy(np.array(archive["edges"], copy=True)),
                torch.from_numpy(np.array(archive["features"], copy=True)),
                record["actual_data"],
                record["feature_mode"],
                record["metadata"],
            )
        )
    if record["graph_sha256"] != graph_content_hash(case) or record[
        "feature_sha256"
    ] != feature_content_hash(case):
        raise ValueError("input snapshot content checksum mismatch")
    return case


def _citation_case(graph, config):
    expected = config["citation"]["expected_shapes"][graph.name]
    if graph.num_nodes != expected["nodes"] or graph.num_features != expected["features"]:
        raise ValueError("citation graph/feature scale differs from complete contract")
    if "physical_edges" in expected and graph.edges.shape[1] != expected["physical_edges"]:
        raise ValueError("citation physical-edge count differs from complete contract")
    actual = config["profile"] == "full"
    if graph.metadata.get("actual_citation_data") is not actual:
        raise ValueError("citation actual-data provenance differs")
    return validate_case(
        AuditCase(
            graph.name,
            "citation",
            graph.edges.detach().cpu().clone(),
            graph.x.detach().cpu().to(torch.float64),
            actual,
            "vector_trace",
            {
                **graph.metadata,
                "profile": config["profile"],
                "synthetic_graph": False,
                "input_origin": config["citation"]["feature_origin"],
                "feature_mode": "vector_trace",
                "source_loader_feature_dtype": str(graph.x.dtype),
                "audit_feature_dtype": "torch.float64",
                "encoder": None,
                "labels_and_masks_used": False,
                "original_node_ids": "zero_based_original_graph_order",
            },
        )
    )


def prepare_cases(config, data_root, output_dir, workers="auto", download=True):
    """Prepare all contracted inputs, preserve old caches, and write new snapshots."""
    validate_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs, loader_output = output / "inputs", output / "citation-loader"
    if inputs.exists() or loader_output.exists() or (output / "input_manifest.json").exists():
        raise FileExistsError(
            "input preparation artifacts already exist; all existing files preserved"
        )
    specs = make_specs(config["synthetic"]["master_seed"], config["profile"])
    if (
        len(specs) != config["synthetic"]["graphs"]
        or sum(s.num_features for s in specs) != config["synthetic"]["scalar_inputs"]
    ):
        raise ValueError("complete synthetic graph/realization count differs")
    requested = config["runtime"]["cpu_workers"] if workers == "auto" else workers
    selected, calibration = choose_workers(specs, requested)
    cases = _generate(specs, selected)
    source_config = Path(citation_data.__file__).parent / f"config_{config['profile']}.json"
    loader_config = json.loads(source_config.read_text(encoding="utf-8"))
    graphs, citation_manifest = citation_data.prepare_datasets(
        loader_config,
        data_root,
        loader_output,
        workers=selected,
        download=download,
    )
    cases.extend(_citation_case(graphs[name], config) for name in config["citation"]["datasets"])
    if len(cases) != config["data"]["total_graphs"] or len({c.graph_id for c in cases}) != len(
        cases
    ):
        raise ValueError("input case coverage is incomplete or duplicated")
    inputs.mkdir()
    with ThreadPoolExecutor(max_workers=selected) as pool:
        records = list(
            pool.map(lambda case: save_case(case, inputs / f"{case.graph_id}.npz"), cases)
        )
    manifest = {
        "schema": _SCHEMA,
        "profile": config["profile"],
        "graphs": len(cases),
        "synthetic_graphs": len(specs),
        "citation_graphs": len(graphs),
        "actual_citation_graphs": sum(case.actual_data for case in cases),
        "synthetic_scalar_inputs": config["synthetic"]["scalar_inputs"],
        "citation_channels_are_vector_not_independent_replicates": True,
        "precision": "float64",
        "sampling_ratio": 1.0,
        "encoder": None,
        "labels_and_masks_used": False,
        "selected_workers": selected,
        "cpu_input_calibration": calibration,
        "inputs": records,
        "input_content_digest": digest(records),
        "citation_loader_manifest": "citation-loader/data_manifest.json",
        "citation_loader_manifest_sha256": file_sha256(loader_output / "data_manifest.json"),
        "citation_raw_source": citation_manifest["raw"],
    }
    write_json(output / "input_manifest.json", manifest)
    return cases, manifest


def assert_inputs_unchanged(output_dir, manifest):
    for record in manifest["inputs"]:
        if file_sha256(Path(output_dir) / "inputs" / record["file"]) != record["sha256"]:
            raise ValueError(
                f"input snapshot changed: {record['graph_id']}; existing files preserved"
            )
    loader_path = Path(output_dir) / manifest["citation_loader_manifest"]
    if file_sha256(loader_path) != manifest["citation_loader_manifest_sha256"]:
        raise ValueError("citation loader manifest changed; existing files preserved")


__all__ = [
    "AuditCase",
    "validate_case",
    "make_synthetic_case",
    "prepare_cases",
    "choose_workers",
    "save_case",
    "load_case",
    "graph_content_hash",
    "feature_content_hash",
    "assert_inputs_unchanged",
]
