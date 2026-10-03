"""Pinned full public labels and masks, with exact prior local-topology snapshots.

The prior 201-graph audit supplies provenance and cached topology. Only its
three citation inputs are classification datasets. Synthetic audit graphs
never enter the classification loss. All caches are numeric, exclusive files.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...wedge_propagation.classification import data as public_data
from ..data import AuditCase, _available_cpus
from ..receiver_aggregation import contract as source_contract
from ..receiver_aggregation import data as source_data
from ..topology import CORRESPONDENCE_KINDS, LocalTopology
from .common import digest, file_sha256, read_json, validate_config, write_json
from .operators import (
    PredictionGeometry,
    geometry_arrays,
    prepare_geometry,
    restore_geometry,
)

SCHEMA = "local-energy-prediction-data-v1"
_TENSOR_FIELDS = ("x", "y", "train_mask", "val_mask", "test_mask", "edges")
_TOPOLOGY_FIELDS = {f.name for f in fields(LocalTopology)} - {"correspondence_offsets"}
_TOPOLOGY_OFFSET_FIELDS = {kind + "_offsets" for kind in CORRESPONDENCE_KINDS}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _numpy(value):
    if isinstance(value, torch.Tensor):
        _require(value.device.type == "cpu", "snapshot preparation requires CPU tensors")
        return value.detach().numpy()
    return np.asarray(value)


def _numpy_topology(top):
    return LocalTopology(
        **{
            f.name: getattr(top, f.name)
            if f.name in ("n", "correspondence_offsets")
            else _numpy(getattr(top, f.name))
            for f in fields(top)
        }
    )


def _topology_record(top):
    return {
        "nodes": top.n,
        "physical_edges": top.num_edges,
        "local_node_occurrences": top.num_local_nodes,
        "local_edge_occurrences": top.num_local_edges,
        "directed_center_pairs": top.num_pairs,
    }


@dataclass(frozen=True)
class PredictionGraph:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edges: torch.Tensor
    geometry: PredictionGeometry
    metadata: dict[str, Any]

    @property
    def topology(self):
        return self.geometry.topology

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
        _require(dtype in (torch.float32, torch.float64), "prediction precision must be float32/64")
        values = {
            name: getattr(self, name).to(
                device=device,
                dtype=dtype
                if getattr(self, name).is_floating_point()
                else getattr(self, name).dtype,
            )
            for name in _TENSOR_FIELDS
        }
        return replace(self, **values, geometry=self.geometry.to(device, dtype))


def _validate_graph(graph):
    _require(isinstance(graph, PredictionGraph), "complete PredictionGraph required")
    _require(
        isinstance(graph.name, str)
        and graph.name in (*public_data.EXPECTED, *("DEBUG-" + n for n in public_data.EXPECTED)),
        "only complete public or explicitly DEBUG citation graphs supported",
    )
    _require(
        isinstance(graph.metadata, dict)
        and type(graph.num_classes) is int
        and graph.num_classes > 1,
        "graph class metadata missing",
    )
    _require(
        all(getattr(graph, name).device.type == "cpu" for name in _TENSOR_FIELDS),
        "cache validation requires CPU data",
    )
    n, x = graph.num_nodes, graph.x
    _require(
        x.dtype == torch.float32
        and x.ndim == 2
        and min(x.shape) > 0
        and bool(torch.isfinite(x).all())
        and bool((x >= 0).all()),
        "finite nonnegative full float32 features required",
    )
    sums = x.sum(1)
    _require(
        torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-6, rtol=1e-6),
        "feature row normalization changed",
    )
    _require(graph.y.dtype == torch.long and graph.y.shape == (n,), "integer full labels required")
    masks = [graph.train_mask, graph.val_mask, graph.test_mask]
    _require(
        all(mask.dtype == torch.bool and mask.shape == (n,) and bool(mask.any()) for mask in masks)
        and not bool((masks[0] & masks[1] | masks[0] & masks[2] | masks[1] & masks[2]).any()),
        "train/validation/test masks must be nonempty and disjoint",
    )
    _require(
        bool(((graph.y >= -1) & (graph.y < graph.num_classes)).all())
        and bool((graph.y[masks[0] | masks[1] | masks[2]] >= 0).all()),
        "masked labels must identify a valid public class",
    )
    _require(
        graph.edges.dtype == torch.long
        and torch.equal(
            graph.edges, torch.from_numpy(public_data.canonical_edges(graph.edges.numpy(), n))
        ),
        "complete canonical physical edges required",
    )
    for key, expected in {
        "nodes": n,
        "features": graph.num_features,
        "physical_edges": graph.edges.shape[1],
        "train": int(graph.train_mask.sum()),
        "validation": int(graph.val_mask.sum()),
        "test": int(graph.test_mask.sum()),
        "sampling_ratio": 1.0,
    }.items():
        _require(graph.metadata.get(key) == expected, f"graph metadata {key} differs")
    actual = not graph.name.startswith("DEBUG-")
    _require(
        graph.metadata.get("actual_citation_data") is actual
        and graph.metadata.get("debug") is (not actual),
        "public/DEBUG provenance differs",
    )
    top = _numpy_topology(graph.topology)
    case = AuditCase(
        graph.name, "citation", graph.edges, graph.x.double(), actual, "vector_trace", {}
    )
    source_data.validate_topology(top, case, _topology_record(top))
    _require(
        graph.metadata.get("geometry") == graph.geometry.metadata,
        "cached topology normalization metadata differs",
    )
    # A restore validates shapes and finite coefficients without recomputing any
    # weighted Laplacian or local correspondence construction.
    restore_geometry(top, geometry_arrays(graph.geometry), graph.geometry.metadata)
    json.dumps(graph.metadata, allow_nan=False)
    return graph


def graph_from_arrays(name, values, topology, metadata=None, source_case=None):
    """Normalize all original channels once and prepare exact CPU geometry once."""
    x, y, edges, train, validation, test = values[:6]
    features = np.asarray(x, dtype=np.float64)
    _require(
        features.ndim == 2
        and min(features.shape) > 0
        and np.isfinite(features).all()
        and (features >= 0).all(),
        "raw full citation features must be finite and nonnegative",
    )
    row_sum = features.sum(1)
    normalized = np.divide(
        features, row_sum[:, None], out=np.zeros_like(features), where=row_sum[:, None] > 0
    ).astype(np.float32)
    n = normalized.shape[0]
    physical = public_data.canonical_edges(edges, n)
    if source_case is not None:
        _require(
            torch.equal(torch.from_numpy(normalized).double(), source_case.features)
            and torch.equal(torch.from_numpy(physical), source_case.edges),
            f"classification inputs differ from the complete original local audit: {name}",
        )
    labels = np.asarray(y)
    _require(np.issubdtype(labels.dtype, np.integer), "raw labels must be integers")
    masks = [np.asarray(mask) for mask in (train, validation, test)]
    _require(all(mask.dtype == np.bool_ for mask in masks), "raw public masks must be boolean")
    meta = dict(values[6] if len(values) > 6 else {})
    meta.update(metadata or {})
    meta.update(
        schema=SCHEMA,
        nodes=n,
        features=normalized.shape[1],
        physical_edges=physical.shape[1],
        train=int(masks[0].sum()),
        validation=int(masks[1].sum()),
        test=int(masks[2].sum()),
        zero_feature_rows=int((row_sum == 0).sum()),
        sampling_ratio=1.0,
        feature_normalization="row_sum_normalize_preserve_zero_rows",
        all_nodes_edges_features=True,
        all_induced_one_hop_egos=True,
        all_directed_adjacent_center_pairs=True,
        synthetic_graphs_used_for_training=0,
    )
    geometry = prepare_geometry(topology)
    meta["geometry"] = geometry.metadata
    return _validate_graph(
        PredictionGraph(
            name,
            torch.from_numpy(normalized),
            torch.from_numpy(labels.astype(np.int64)),
            *(torch.from_numpy(mask.copy()) for mask in masks),
            torch.from_numpy(physical),
            geometry,
            meta,
        )
    )


def _snapshot_arrays(graph):
    arrays = {name: _numpy(getattr(graph, name)) for name in _TENSOR_FIELDS}
    top = graph.topology
    arrays.update({"topology__" + name: _numpy(getattr(top, name)) for name in _TOPOLOGY_FIELDS})
    arrays.update(
        {
            "topology__" + kind + "_offsets": np.asarray(offset, dtype=np.int64)
            for kind, offset in zip(CORRESPONDENCE_KINDS, top.correspondence_offsets, strict=True)
        }
    )
    arrays.update(
        {"geometry__" + name: value for name, value in geometry_arrays(graph.geometry).items()}
    )
    arrays["name"] = np.asarray(graph.name)
    arrays["metadata"] = np.asarray(json.dumps(graph.metadata, sort_keys=True, allow_nan=False))
    return arrays


def graph_content_digest(graph):
    """Numeric input, public masks, topology and coefficients; no serialized objects."""
    hasher = hashlib.sha256()
    for name, value in sorted(_snapshot_arrays(graph).items()):
        array = np.ascontiguousarray(value)
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def save_graph(graph, path):
    _validate_graph(graph)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **_snapshot_arrays(graph))
    return {
        "schema": SCHEMA,
        "name": graph.name,
        "file": path.name,
        "sha256": file_sha256(path),
        "content_digest": graph_content_digest(graph),
        "statistics": graph.metadata,
        "cached_topology_and_geometry": True,
    }


def load_graph(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None:
        _require(file_sha256(path) == expected_sha256, "prediction graph checksum mismatch")
    required = set(_TENSOR_FIELDS) | {"name", "metadata"}
    required |= {"topology__" + name for name in _TOPOLOGY_FIELDS | _TOPOLOGY_OFFSET_FIELDS}
    with np.load(path, allow_pickle=False) as saved:
        keys = set(saved.files)
        _require(
            len(keys) == len(saved.files)
            and required <= keys
            and all(key.startswith("geometry__") for key in keys - required),
            "prediction graph snapshot field coverage differs",
        )
        values = {name: np.array(saved[name], copy=True) for name in keys}
    top_values = {
        name: values["topology__" + name] for name in _TOPOLOGY_FIELDS | _TOPOLOGY_OFFSET_FIELDS
    }
    _require(
        all(value.dtype == np.int64 for value in top_values.values())
        and top_values["n"].shape == (),
        "cached topology must use exact int64 numeric arrays",
    )
    top_values["n"] = int(top_values["n"])
    top_values["correspondence_offsets"] = tuple(
        tuple(int(value) for value in top_values.pop(kind + "_offsets"))
        for kind in CORRESPONDENCE_KINDS
    )
    top = LocalTopology(**top_values)
    metadata = json.loads(str(values["metadata"].item()))
    _require(metadata.get("schema") == SCHEMA, "snapshot prediction schema differs")
    coefficients = {
        key[len("geometry__") :]: value
        for key, value in values.items()
        if key.startswith("geometry__")
    }
    geometry = restore_geometry(top, coefficients, metadata["geometry"])
    graph = PredictionGraph(
        str(values["name"].item()),
        **{name: torch.from_numpy(values[name]) for name in _TENSOR_FIELDS},
        geometry=geometry,
        metadata=metadata,
    )
    return _validate_graph(graph)


def _decode(arguments):
    name, root = arguments
    return name, public_data.decode_planetoid(root, name, public_data.EXPECTED[name])


def _decode_all(arguments, workers):
    if workers == 1:
        return dict(_decode(item) for item in arguments)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_decode, arguments))


def _raw_array_digest(arrays):
    return digest(
        {
            name: [
                hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                for value in values[:6]
            ]
            for name, values in arrays.items()
        }
    )


def prepare_datasets(config, data_root, output_dir, source_dir, workers="auto", download=True):
    """Complete classification graphs; prior synthetic snapshots are provenance only."""
    validate_config(config)
    output, original = Path(output_dir).resolve(), Path(source_dir).resolve()
    _require(
        not output.is_relative_to(original) and not original.is_relative_to(output),
        "new prediction output must be separate from prior scientific results",
    )
    output.mkdir(parents=True, exist_ok=True)
    _require(
        not (output / "source-audit").exists() and not (output / "data_manifest.json").exists(),
        "prediction data artifacts already exist; existing files preserved",
    )
    source_config = source_contract.read_config(profile=config["profile"])
    _require(
        all(
            config["source"].get(key) == source_config["source"][key]
            for key in (
                "profile",
                "graphs",
                "synthetic_graphs",
                "synthetic_scalar_inputs",
                "citation_graphs",
                "actual_citation_graphs",
                "required_experiment",
            )
        ),
        "previous whole-audit source contract differs",
    )
    requested = config["runtime"]["cpu_workers"] if workers == "auto" else workers
    cases, topologies, source = source_data.prepare_cases(
        source_config, original, output / "source-audit", workers=requested
    )
    names = config["data"]["datasets"]
    selected = {
        case.graph_id: (case, top)
        for case, top in zip(cases, topologies, strict=True)
        if case.graph_id in names
    }
    _require(set(selected) == set(names), "prior audit omits a required complete citation graph")
    # Retain only complete citations in the classifier's resident data. All
    # prior synthetic hashes remain in the immutable provenance manifest.
    del cases, topologies
    raw, raw_closure, calibration = {}, {}, []
    if config["profile"] == "debug":
        loader_config = read_json(Path(public_data.__file__).with_name("config_debug.json"))
        _require(
            config["debug_fixture"] == loader_config["debug_fixture"],
            "DEBUG fixture must match its separate original audit",
        )
        arrays = public_data._debug_arrays(loader_config)
        selected_workers = source["selected_workers"]
    else:
        _require(config["data_source"] == "planetoid_public", "full requires pinned public data")
        _require(
            config["data"]["expected_shapes"] == public_data.EXPECTED,
            "complete public data scale or masks changed",
        )
        arguments = []
        for name in names:
            raw_root, provenance = public_data.ensure_raw(name, data_root, download)
            raw[name] = provenance
            _require(
                provenance == source["citation_raw_source"][name],
                "classification raw identity differs from the initial local audit",
            )
            for filename, expected in provenance["sha256"].items():
                raw_closure[str((raw_root / filename).resolve())] = expected
            manifest_path = raw_root / "raw_manifest.json"
            raw_closure[str(manifest_path.resolve())] = file_sha256(manifest_path)
            arguments.append((name, raw_root))
        available = _available_cpus()
        _require(
            requested == "auto" or (type(requested) is int and 1 <= requested <= available),
            "raw decode worker allocation exceeds CPU affinity/quota",
        )
        candidates = (
            [
                number
                for number in config["resources"]["preprocessing_worker_candidates"]
                if number <= available
            ]
            if requested == "auto"
            else [requested]
        )
        reference, arrays, best_seconds = None, None, float("inf")
        for number in candidates:
            started = time.perf_counter()
            decoded = _decode_all(arguments, number)
            seconds = time.perf_counter() - started
            identity = _raw_array_digest(decoded)
            if reference is None:
                reference = identity
            _require(identity == reference, "parallel public decoding changed graph or mask data")
            calibration.append(
                {
                    "workers": number,
                    "seconds": seconds,
                    "graphs": len(names),
                    "scope": "all_complete_public_raw_graphs",
                }
            )
            if seconds < best_seconds:
                best_seconds, arrays, selected_workers = seconds, decoded, number
            print(
                f"[CPU raw calibration] workers={number} graphs={len(names)} seconds={seconds:.3f}",
                flush=True,
            )
    graphs, records = {}, {}
    for name in names:
        prior, topology = selected[name]
        actual = config["profile"] == "full"
        metadata = {
            "data_source": config["data_source"],
            "actual_citation_data": actual,
            "debug": not actual,
            "raw_sha256": raw[name]["sha256"] if actual else {},
            "repository_commit": public_data.PLANETOID_COMMIT if actual else None,
            "source_graph_sha256": next(
                r["graph_sha256"] for r in source["inputs"] if r["graph_id"] == name
            ),
            "source_feature_sha256": next(
                r["feature_sha256"] for r in source["inputs"] if r["graph_id"] == name
            ),
            "source_topology_sha256": next(
                r["sha256"] for r in source["topology"] if r["graph_id"] == name
            ),
            "original_node_ids": "zero_based_original_graph_order",
            "public_masks_from_raw_not_used_in_prior_audit": actual,
        }
        values = arrays[name]
        metadata.setdefault(
            "classes",
            values[6]["classes"]
            if "classes" in values[6]
            else config["data"]["expected_shapes"][name]["classes"],
        )
        graph = graph_from_arrays(name, values, topology, metadata, source_case=prior)
        expected = config["data"]["expected_shapes"][name]
        _require(
            all(graph.metadata.get(key) == value for key, value in expected.items()),
            f"whole public/DEBUG graph or split size differs: {name}",
        )
        graphs[name] = graph
        records[name] = {
            "statistics": graph.metadata,
            "content_digest": graph_content_digest(graph),
        }
        print(
            f"[data] {name} x={list(graph.x.shape)} classes={graph.num_classes} "
            f"edges={graph.edges.shape[1]} local_edges={graph.topology.num_local_edges} "
            f"relations={graph.topology.num_pairs} train/val/test="
            f"{graph.metadata['train']}/{graph.metadata['validation']}/{graph.metadata['test']} "
            "sampling_ratio=1.0 synthetic_training_graphs=0",
            flush=True,
        )
    manifest = {
        "schema": SCHEMA,
        "profile": config["profile"],
        "data_source": config["data_source"],
        "datasets": records,
        "raw": raw,
        "raw_file_sha256": raw_closure,
        "raw_content_digest": digest(raw_closure),
        "selected_workers": selected_workers,
        "cpu_calibration": calibration,
        "source_audit": source,
        "source_dir": str(original),
        "classification_graphs": len(graphs),
        "actual_citation_data": config["profile"] == "full",
        "all_nodes_edges_features": True,
        "all_induced_one_hop_egos": True,
        "all_directed_adjacent_center_pairs": True,
        "sampling_ratio": 1.0,
        "synthetic_graphs_used_for_training": 0,
        "prior_synthetic_graphs_preserved": config["source"]["synthetic_graphs"],
        "cached_topology_and_geometry": True,
        "no_wedge_model_or_path_features": True,
        "official_source": "https://github.com/kimiyoung/planetoid",
        "public_loader_reference": "https://github.com/tkipf/gcn/blob/master/gcn/utils.py",
    }
    assert_data_unchanged(manifest)
    write_json(output / "data_manifest.json", manifest)
    return graphs, manifest


def assert_data_unchanged(manifest):
    source_data.assert_inputs_unchanged(manifest["source_dir"], manifest["source_audit"])
    _require(
        manifest["raw_content_digest"] == digest(manifest["raw_file_sha256"]),
        "pinned public raw reference manifest changed",
    )
    for path, expected in manifest["raw_file_sha256"].items():
        _require(file_sha256(path) == expected, f"original pinned public input changed: {path}")


__all__ = [
    "PredictionGraph",
    "prepare_datasets",
    "graph_from_arrays",
    "save_graph",
    "load_graph",
    "graph_content_digest",
    "assert_data_unchanged",
]
