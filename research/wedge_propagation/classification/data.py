"""Independent, complete citation data and cached classification topology.

The eight Planetoid files are pinned to the authors' repository commit and
verified before their historical pickle format is read. Public masks and the
CiteSeer test-index gaps follow the GCN authors' reference loader. No previous
Conductance experiment is imported.

Sources: https://github.com/kimiyoung/planetoid/tree/master/data
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import psutil
import scipy.sparse as sp
import torch

PLANETOID_COMMIT = "af6abf468a772cc055bec88c52e4d6f51b7c37e3"
RAW_BASE = f"https://raw.githubusercontent.com/kimiyoung/planetoid/{PLANETOID_COMMIT}/data"
RAW_PARTS = ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index")
SCHEMA = "wedge-citation-data-v1"
EXPECTED = {
    "Cora": {
        "nodes": 2708,
        "features": 1433,
        "classes": 7,
        "train": 140,
        "validation": 500,
        "test": 1000,
    },
    "CiteSeer": {
        "nodes": 3327,
        "features": 3703,
        "classes": 6,
        "train": 120,
        "validation": 500,
        "test": 1000,
    },
    "PubMed": {
        "nodes": 19717,
        "features": 500,
        "classes": 3,
        "train": 60,
        "validation": 500,
        "test": 1000,
    },
}
RAW_SHA256 = {
    "Cora": {
        "x": "23cfa55d91c6f624233f5eb7b6e3f141f1bd2b2ae39608a63cf5d084bb27baab",
        "y": "94465c14eb53e04ca262198dcbb2521ee8af60fb3f3d546cd6ca24a511b0e7d1",
        "tx": "b9afbaa4a400df991f6f02ef677e1e44da55ffc04801fd02f9e673987829226a",
        "ty": "41f5ac76596a1699cc33f53084a2419e92961c42bb75f36fc38e616d348532ad",
        "allx": "9419ba2f26f5c35243db64aba110e0f35a04851609e0dc5433450676ca6b8543",
        "ally": "2b998f5cc7fedc86e7f97ca2498f47a1ffc1462c29e2d578b23bf3f62b6e7d71",
        "graph": "58f13302f39dde8852dad6fe6d15b89b077b6d7f837626bce671ef80344b383d",
        "test.index": "297ce89af2b51a6a194181c7dbe1796c6a1cf9cd9349a88383b1b1e227867875",
    },
    "CiteSeer": {
        "x": "d20de19150741e555de0ed037195630fc194114ebc7fd72ced62810148aa22f6",
        "y": "8ba87e515d8e3ee3cf52f6d2c5b86aa73f11ecad825a808455209ec26cd54924",
        "tx": "539a8906a2da97628d212f2fdd668c109a8b3f0d36574b59e5d3fd5b5303da21",
        "ty": "55e5bd1ba1e733a04598753ad1a8d57957f4b8f89e74519f41f4f4938767e4ce",
        "allx": "2ac30345d95c9ec933a817ee0bdd6a5f077f8c184a20e696910192d534414668",
        "ally": "f704b2d986dde6c2669934de1f3ae5696a6cf9455c0f6b2646a2d135ea0a1c95",
        "graph": "d79a4ef9d3e7169aee8946145f6b7306e35bc79bffad26346cfb94e10e9912a7",
        "test.index": "2af990671580b6b2df5d158d1038f8292e30c9cd821b5453f399659b25b723d4",
    },
    "PubMed": {
        "x": "fbe9cb5c47200d1b7769a26e579be3b7130b556831f9d850e77c4c74f2599b33",
        "y": "e6c807633307a07ed659249006536a147c7999c797f11a2560752990181b41b3",
        "tx": "4eb5f5d0f30f26497eb0ac6ac9719d09c67e284f1ed0385c93f3ef27a671da3d",
        "ty": "ee2e0d4819d9fbcc403689c3bf2e49face401b605932f093bc58ab6461130fdd",
        "allx": "508e538e2abdccdd54c4d3e9f49d638b4af7107d85845f66ddea5a5152658b7e",
        "ally": "0c37bbe3b5014ec365e9d393cf4791925e5ed94fb194305d74f064a8d7b50f0e",
        "graph": "5b89f0036ca22909471f1e6558eb42fa06e0c730a577bde5b058b40638f81dc9",
        "test.index": "b101d421688bbaecbd82e8a18f1e282378d6830f3a37666f28ebc47f844341b5",
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_path(data_root, *parts):
    """Reject a linked cache/raw descendant that leaves the declared data root."""
    declared = Path(data_root).resolve()
    candidate = declared.joinpath(*parts)
    _require(
        candidate.resolve().is_relative_to(declared),
        f"citation data path escapes declared data root; preserved: {candidate}",
    )
    return candidate


@dataclass(frozen=True)
class GraphData:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edges: torch.Tensor
    paths: torch.Tensor
    degree: torch.Tensor
    qdiag: torch.Tensor
    sd: torch.Tensor
    sq: torch.Tensor
    gcn_edges: torch.Tensor
    gcn_weight: torch.Tensor
    metadata: dict[str, Any]

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
        _require(dtype in (torch.float32, torch.float64), "graph precision must be float32/float64")
        values = {}
        for field in fields(self):
            tensor = getattr(self, field.name)
            if isinstance(tensor, torch.Tensor):
                values[field.name] = tensor.to(
                    device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype
                )
        return replace(self, **values)


def canonical_edges(edges, n):
    supplied = np.asarray(edges)
    _require(np.issubdtype(supplied.dtype, np.integer), "edge indices must be integer")
    value = supplied.astype(np.int64, copy=False)
    _require(value.ndim == 2 and value.shape[0] == 2, "edges must have shape [2,E]")
    _require(bool(((value >= 0) & (value < n)).all()), "edge node outside full graph")
    pairs = np.sort(value.T, axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    return np.ascontiguousarray(pairs.T)


def _adjacency(edges, n):
    centers = np.concatenate((edges[0], edges[1]))
    neighbors = np.concatenate((edges[1], edges[0]))
    order = np.lexsort((neighbors, centers))
    degree = np.bincount(centers, minlength=n)
    return neighbors[order], degree, np.cumsum(degree) - degree


def _center_paths(arguments):
    begin, end, neighbors, degree, starts = arguments
    local_degree = degree[begin:end]
    counts = local_degree * (local_degree - 1) // 2
    total = int(counts.sum())
    if not total:
        return np.empty((3, 0), dtype=np.int64)
    centers = np.repeat(np.arange(begin, end, dtype=np.int64), counts)
    ordinal = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    d = degree[centers]
    first = np.floor(
        ((2 * d - 1) - np.sqrt((2 * d - 1).astype(np.float64) ** 2 - 8 * ordinal)) / 2
    ).astype(np.int64)
    first = np.maximum(first, 0)
    start = first * (2 * d - first - 1) // 2
    first -= start > ordinal
    next_start = (first + 1) * (2 * d - first - 2) // 2
    first += next_start <= ordinal
    start = first * (2 * d - first - 1) // 2
    second = first + 1 + ordinal - start
    return np.stack(
        (neighbors[starts[centers] + first], centers, neighbors[starts[centers] + second])
    )


def build_paths(edges, n, workers=1):
    _require(
        isinstance(workers, int) and not isinstance(workers, bool) and workers > 0,
        "preprocessing workers must be positive",
    )
    neighbors, degree, starts = _adjacency(edges, n)
    boundaries = np.linspace(0, n, min(workers, n) + 1, dtype=np.int64)
    arguments = [
        (int(a), int(b), neighbors, degree, starts)
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    if workers == 1:
        parts = [_center_paths(arguments[0])]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_center_paths, arguments))
    result = np.ascontiguousarray(np.concatenate(parts, axis=1))
    _require(
        result.shape[1] == int((degree * (degree - 1) // 2).sum()),
        "complete path construction omitted neighbor pairs",
    )
    return result


def _mask(indices, n):
    result = np.zeros(n, dtype=np.bool_)
    result[indices] = True
    return result


def graph_from_arrays(name, x, y, edges, train_mask, val_mask, test_mask, workers=1, metadata=None):
    """Build all physical edges, paths and topology-only normalizers once."""
    feature = np.asarray(x, dtype=np.float64)
    label = np.asarray(y)
    _require(
        feature.ndim == 2
        and feature.shape[0] > 0
        and feature.shape[1] > 0
        and bool(np.isfinite(feature).all())
        and bool((feature >= 0).all()),
        "features must be a finite nonnegative full [N,D] matrix",
    )
    n = feature.shape[0]
    _require(
        label.shape == (n,) and np.issubdtype(label.dtype, np.integer), "labels must be integer [N]"
    )
    masks = [np.asarray(mask) for mask in (train_mask, val_mask, test_mask)]
    _require(
        all(mask.shape == (n,) and mask.dtype == np.bool_ and bool(mask.any()) for mask in masks),
        "public masks must be nonempty bool [N]",
    )
    _require(
        not bool((masks[0] & masks[1] | masks[0] & masks[2] | masks[1] & masks[2]).any()),
        "train/validation/test masks overlap",
    )
    meta = dict(metadata or {})
    classes = meta.get("classes", int(label.max()) + 1)
    _require(
        classes > 1
        and bool(((label >= -1) & (label < classes)).all())
        and bool((label[np.logical_or.reduce(masks)] >= 0).all()),
        "masked labels must identify a valid class",
    )
    raw_sum = feature.sum(1)
    normalized = np.divide(
        feature, raw_sum[:, None], out=np.zeros_like(feature), where=raw_sum[:, None] > 0
    ).astype(np.float32)
    physical = canonical_edges(edges, n)
    paths = build_paths(physical, n, workers)
    _, degree, _ = _adjacency(physical, n)
    qdiag = (
        np.bincount(paths[0], minlength=n)
        + 4 * np.bincount(paths[1], minlength=n)
        + np.bincount(paths[2], minlength=n)
    ).astype(np.float32)
    sd = np.zeros(n, dtype=np.float32)
    sq = np.zeros(n, dtype=np.float32)
    np.divide(1, np.sqrt(degree), out=sd, where=degree > 0)
    np.divide(1, np.sqrt(qdiag), out=sq, where=qdiag > 0)
    node = np.arange(n, dtype=np.int64)
    gcn_edges = np.concatenate((physical, physical[::-1], np.stack((node, node))), axis=1)
    gcn_weight = ((degree[gcn_edges[0]] + 1) * (degree[gcn_edges[1]] + 1)) ** (-0.5)
    meta.update(
        classes=int(classes),
        nodes=n,
        features=feature.shape[1],
        physical_edges=physical.shape[1],
        paths=paths.shape[1],
        train=int(masks[0].sum()),
        validation=int(masks[1].sum()),
        test=int(masks[2].sum()),
        zero_feature_rows=int((raw_sum == 0).sum()),
        isolates=int((degree == 0).sum()),
        max_degree=int(degree.max()),
        sampling_ratio=1.0,
        all_paths=True,
        feature_normalization="row_sum_normalize_preserve_zero_rows",
    )
    return GraphData(
        name,
        torch.from_numpy(normalized),
        torch.from_numpy(label.astype(np.int64)),
        *(torch.from_numpy(mask.copy()) for mask in masks),
        torch.from_numpy(physical),
        torch.from_numpy(paths),
        torch.from_numpy(degree.astype(np.float32)),
        torch.from_numpy(qdiag),
        torch.from_numpy(sd),
        torch.from_numpy(sq),
        torch.from_numpy(gcn_edges),
        torch.from_numpy(gcn_weight.astype(np.float32)),
        meta,
    )


def decode_planetoid(raw_dir, name, expected):
    """Decode trusted hash-verified Planetoid bytes; tests use labeled raw fixtures."""
    root = Path(raw_dir)
    stem = f"ind.{name.lower()}."
    objects = []
    for part in RAW_PARTS[:-1]:
        with (root / f"{stem}{part}").open("rb") as stream:
            objects.append(pickle.load(stream, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects
    _require(
        all(sp.issparse(value) for value in (x, tx, allx)),
        "raw feature objects must be scipy sparse matrices",
    )
    _require(
        all(isinstance(value, np.ndarray) and value.ndim == 2 for value in (y, ty, ally)),
        "raw label objects must be two-dimensional numpy arrays",
    )
    _require(
        isinstance(graph, dict) and set(graph) == set(range(expected["nodes"])),
        "raw adjacency must retain every declared graph node",
    )
    text = (root / f"{stem}test.index").read_text(encoding="utf-8").splitlines()
    indices = np.asarray([int(line) for line in text], dtype=np.int64)
    ordered = np.sort(indices)
    _require(
        len(indices) == expected["test"]
        and len(np.unique(indices)) == len(indices)
        and bool(((indices >= 0) & (indices < expected["nodes"])).all()),
        "raw public test indices are invalid",
    )
    _require(
        x.shape == (expected["train"], expected["features"])
        and y.shape == (expected["train"], expected["classes"])
        and tx.shape == (expected["test"], expected["features"])
        and ty.shape == (expected["test"], expected["classes"])
        and allx.shape[1] == expected["features"]
        and ally.shape == (allx.shape[0], expected["classes"]),
        "raw feature/label source shape disagrees with public contract",
    )
    holes = []
    if name == "CiteSeer":
        span = np.arange(ordered.min(), ordered.max() + 1)
        holes = np.setdiff1d(span, ordered).tolist()
        extended = sp.lil_matrix((len(span), x.shape[1]), dtype=tx.dtype)
        extended[ordered - ordered.min()] = tx
        tx = extended
        labels = np.zeros((len(span), y.shape[1]), dtype=ty.dtype)
        labels[ordered - ordered.min()] = ty
        ty = labels
    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    _require(
        features.shape == (expected["nodes"], expected["features"])
        and labels.shape == (expected["nodes"], expected["classes"]),
        "decoded data omits full graph nodes",
    )
    features[indices] = features[ordered].copy()
    labels[indices] = labels[ordered].copy()
    _require(
        bool(np.isfinite(labels).all())
        and bool(((labels == 0) | (labels == 1)).all())
        and bool(np.isin(labels.sum(1), [0, 1]).all()),
        "raw one-hot labels are invalid",
    )
    label = np.where(labels.sum(1) > 0, labels.argmax(1), -1).astype(np.int64)
    masks = (
        _mask(np.arange(len(y)), len(label)),
        _mask(np.arange(len(y), len(y) + expected["validation"]), len(label)),
        _mask(ordered, len(label)),
    )
    pairs = [
        (int(node), int(neighbor)) for node, neighbors in graph.items() for neighbor in neighbors
    ]
    edges = np.asarray(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
    return (
        features.toarray(),
        label,
        edges,
        *masks,
        {
            "classes": expected["classes"],
            "citeseer_missing_test_indices": holes,
            "public_test_index_reorder": indices.tolist(),
            "unknown_label_nodes": int((label < 0).sum()),
        },
    )


def ensure_raw(name, data_root, download=True):
    _require(name in EXPECTED, "only the three contracted citation datasets are supported")
    root = _data_path(data_root, name, "raw")
    paths = {
        part: _data_path(data_root, name, "raw", f"ind.{name.lower()}.{part}") for part in RAW_PARTS
    }
    manifest_path = _data_path(data_root, name, "raw", "raw_manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for part in RAW_PARTS:
        filename = f"ind.{name.lower()}.{part}"
        path = paths[part]
        expected = RAW_SHA256[name][part]
        if not path.exists():
            _require(download, f"missing official raw file in offline mode: {path}")
            print(f"[download] {name} {filename}", flush=True)
            payload = urlopen(f"{RAW_BASE}/{filename}", timeout=60).read()
            _require(
                hashlib.sha256(payload).hexdigest() == expected,
                f"official download checksum mismatch: {filename}",
            )
            with path.open("xb") as stream:
                stream.write(payload)
        _require(
            path.is_file() and file_hash(path) == expected,
            f"existing raw checksum mismatch; file preserved: {path}",
        )
        hashes[filename] = expected
    manifest = {
        "schema": SCHEMA,
        "dataset": name,
        "repository_commit": PLANETOID_COMMIT,
        "official_raw_base": RAW_BASE,
        "sha256": hashes,
    }
    if manifest_path.exists():
        _require(
            json.loads(manifest_path.read_text(encoding="utf-8")) == manifest,
            "existing raw manifest mismatch; file preserved",
        )
    else:
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
    return root, manifest


def save_graph(graph: GraphData, path):
    """Write an exclusive numeric NPZ cache without Python-object deserialization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        field.name: getattr(graph, field.name).detach().cpu().numpy()
        for field in fields(graph)
        if isinstance(getattr(graph, field.name), torch.Tensor)
    }
    arrays["name"] = np.asarray(graph.name)
    arrays["metadata"] = np.asarray(json.dumps(graph.metadata, sort_keys=True, allow_nan=False))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "schema": SCHEMA,
        "name": graph.name,
        "sha256": file_hash(path),
        "statistics": graph.metadata,
    }


def load_graph(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None:
        _require(file_hash(path) == expected_sha256, "processed graph checksum mismatch")
    expected_fields = {field.name for field in fields(GraphData)}
    with np.load(path, allow_pickle=False) as archive:
        _require(
            set(archive.files) == expected_fields and len(archive.files) == len(expected_fields),
            "processed cache array coverage mismatch",
        )
        values = {
            name: torch.from_numpy(np.array(archive[name], copy=True))
            for name in expected_fields - {"name", "metadata"}
        }
        graph = GraphData(
            str(archive["name"].item()),
            **values,
            metadata=json.loads(str(archive["metadata"].item())),
        )
    _validate_graph(graph)
    return graph


def _validate_graph(graph):
    n = graph.num_nodes
    _require(
        graph.x.ndim == 2 and graph.x.dtype == torch.float32 and torch.isfinite(graph.x).all(),
        "processed feature shape/dtype/finiteness mismatch",
    )
    _require((graph.x >= 0).all(), "processed features contain negative entries")
    row_sum = graph.x.sum(1)
    _require(
        torch.allclose(
            row_sum[row_sum > 0], torch.ones_like(row_sum[row_sum > 0]), atol=1e-6, rtol=1e-6
        ),
        "processed feature row normalization mismatch",
    )
    _require(
        graph.y.shape == (n,) and graph.y.dtype == torch.long,
        "processed label shape/dtype mismatch",
    )
    masks = [graph.train_mask, graph.val_mask, graph.test_mask]
    _require(
        all(mask.shape == (n,) and mask.dtype == torch.bool and mask.any() for mask in masks)
        and not (masks[0] & masks[1] | masks[0] & masks[2] | masks[1] & masks[2]).any(),
        "processed masks are invalid",
    )
    _require(
        torch.equal(graph.edges, torch.from_numpy(canonical_edges(graph.edges.numpy(), n))),
        "processed physical edges are not canonical",
    )
    _require(
        graph.paths.ndim == 2 and graph.paths.shape[0] == 3 and graph.paths.dtype == torch.long,
        "processed paths have wrong shape/dtype",
    )
    _require(
        ((graph.paths >= 0) & (graph.paths < n)).all(), "processed path index outside full graph"
    )
    _require(
        torch.equal(graph.paths, torch.from_numpy(build_paths(graph.edges.numpy(), n))),
        "processed paths omit/reorder/change complete physical neighbor pairs",
    )
    degree = torch.bincount(graph.edges.flatten(), minlength=n).float()
    _require(
        torch.equal(graph.degree, degree)
        and graph.paths.shape[1] == int((degree.double() * (degree.double() - 1) / 2).sum()),
        "processed degree/full path count mismatch",
    )
    qdiag = (
        torch.bincount(graph.paths[0], minlength=n)
        + 4 * torch.bincount(graph.paths[1], minlength=n)
        + torch.bincount(graph.paths[2], minlength=n)
    ).float()
    for name, diagonal in (("sd", degree), ("sq", qdiag)):
        expected = torch.zeros_like(diagonal)
        positive = diagonal > 0
        expected[positive] = diagonal[positive].rsqrt()
        _require(
            torch.allclose(getattr(graph, name), expected, atol=1e-7, rtol=1e-6),
            f"processed {name} normalization mismatch",
        )
    _require(torch.equal(graph.qdiag, qdiag), "processed Q diagonal mismatch")
    _require(
        graph.gcn_edges.shape == (2, 2 * graph.edges.shape[1] + n)
        and graph.gcn_weight.shape == (graph.gcn_edges.shape[1],),
        "processed GCN cache mismatch",
    )
    node = torch.arange(n)
    expected_edges = torch.cat((graph.edges, graph.edges.flip(0), torch.stack((node, node))), 1)
    expected_weights = (
        ((degree[expected_edges[0]].double() + 1) * (degree[expected_edges[1]].double() + 1))
        .rsqrt()
        .float()
    )
    _require(
        graph.gcn_edges.dtype == torch.long
        and torch.equal(graph.gcn_edges, expected_edges)
        and torch.allclose(graph.gcn_weight, expected_weights, atol=1e-7, rtol=1e-6),
        "processed GCN edge/normalization mismatch",
    )
    _require(
        ((graph.y >= -1) & (graph.y < graph.num_classes)).all()
        and (graph.y[masks[0] | masks[1] | masks[2]] >= 0).all(),
        "processed masked labels are invalid",
    )
    _require(
        graph.metadata.get("nodes") == n
        and graph.metadata.get("features") == graph.num_features
        and graph.metadata.get("paths") == graph.paths.shape[1]
        and graph.metadata.get("physical_edges") == graph.edges.shape[1],
        "processed topology metadata mismatch",
    )


def _choose_workers(edge_sets, requested):
    available = len(psutil.Process().cpu_affinity())
    candidates = (
        [int(requested)]
        if requested != "auto"
        else [count for count in (1, 2, 4, 8) if count <= available]
    )
    _require(
        bool(candidates) and all(0 < count <= available for count in candidates),
        "preprocessing worker count exceeds available affinity",
    )
    trials, reference = [], None
    for count in candidates:
        started = time.perf_counter()
        built = [build_paths(edge, n, count) for edge, n in edge_sets]
        if reference is None:
            reference = built
        else:
            _require(
                all(np.array_equal(a, b) for a, b in zip(reference, built, strict=True)),
                "parallel center construction changed path identities/order",
            )
        elapsed = time.perf_counter() - started
        trials.append(
            {"workers": count, "seconds": elapsed, "paths": sum(paths.shape[1] for paths in built)}
        )
        print(
            f"[cpu calibration] workers={count} all_graphs={len(edge_sets)} "
            f"paths={trials[-1]['paths']} seconds={elapsed:.3f}",
            flush=True,
        )
    return min(trials, key=lambda row: row["seconds"])["workers"], trials


def _debug_arrays(config):
    fixture = config["debug_fixture"]
    _require(config.get("profile") == "debug", "debug fixtures require the separate DEBUG profile")
    nodes = fixture["nodes"]
    _require(
        len(nodes) == 3 and all(n >= 12 for n in nodes),
        "three complete DEBUG fixture graphs required",
    )
    rng = np.random.default_rng(fixture["master_seed"])
    result = {}
    for name, n in zip(config["data"]["datasets"], nodes, strict=True):
        x = rng.poisson(2.0, size=(n, fixture["features"])).astype(np.float64)
        y = np.arange(n, dtype=np.int64) % fixture["classes"]
        i, j = np.triu_indices(n, 1)
        keep = (j == i + 1) | (rng.random(len(i)) < 3 / n)
        edges = np.stack((i[keep], j[keep]))
        train_count = fixture["train_per_class"] * fixture["classes"]
        val_end = train_count + fixture["validation_nodes"]
        _require(0 < train_count < val_end < n, "DEBUG masks must retain nonempty train/val/test")
        result[name] = (
            x,
            y,
            edges,
            _mask(np.arange(train_count), n),
            _mask(np.arange(train_count, val_end), n),
            _mask(np.arange(val_end, n), n),
            {
                "classes": fixture["classes"],
                "data_source": "debug_fixture",
                "actual_citation_data": False,
                "debug": True,
                "master_seed": fixture["master_seed"],
            },
        )
    return result


def prepare_datasets(config, data_root, output_dir, workers="auto", download=True):
    """Load all three complete graphs, calibrate preprocessing, and retain caches."""
    source = config.get("data_source")
    names = [f"DEBUG-{name}" for name in EXPECTED] if source == "debug_fixture" else list(EXPECTED)
    _require(
        config.get("data", {}).get("datasets") == names,
        "all three citation datasets in the design contract are required",
    )
    arrays, raw_manifests = {}, {}
    if source == "debug_fixture":
        arrays = _debug_arrays(config)
    else:
        _require(
            source == "planetoid_public" and config.get("profile") == "full",
            "full classification requires actual planetoid_public data",
        )
        _require(
            config["data"].get("expected_shapes") == EXPECTED
            and config["data"].get("split") == "public"
            and config["data"].get("sampling_ratio") == 1.0,
            "full public split/data scale contract changed",
        )
        # Verify every processed destination before downloading or writing raw data.
        for name in EXPECTED:
            _data_path(data_root, name, "processed", f"{SCHEMA}.npz")
            _data_path(data_root, name, "processed", f"{SCHEMA}.json")
        for name in EXPECTED:
            root, raw = ensure_raw(name, data_root, download)
            values = decode_planetoid(root, name, EXPECTED[name])
            values[-1].update(
                data_source=source,
                actual_citation_data=True,
                debug=False,
                raw_sha256=raw["sha256"],
                repository_commit=PLANETOID_COMMIT,
            )
            arrays[name], raw_manifests[name] = values, raw
    edge_sets = [
        (canonical_edges(values[2], values[0].shape[0]), values[0].shape[0])
        for values in arrays.values()
    ]
    selected, trials = _choose_workers(edge_sets, workers)
    graphs, processed = {}, {}
    for name, values in arrays.items():
        cache = _data_path(data_root, name, "processed", f"{SCHEMA}.npz")
        manifest_path = _data_path(data_root, name, "processed", f"{SCHEMA}.json")
        if source == "planetoid_public" and (cache.exists() or manifest_path.exists()):
            _require(
                cache.is_file() and manifest_path.is_file(), "incomplete processed cache; preserved"
            )
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            _require(
                saved.get("schema") == SCHEMA
                and saved.get("statistics", {}).get("raw_sha256") == raw_manifests[name]["sha256"],
                "processed raw identity mismatch; preserved",
            )
            graph = load_graph(cache, saved["sha256"])
            reference = graph_from_arrays(name, *values[:-1], selected, values[-1])
            _require(
                all(
                    torch.equal(getattr(graph, field.name), getattr(reference, field.name))
                    for field in fields(graph)
                    if isinstance(getattr(graph, field.name), torch.Tensor)
                ),
                "processed cache differs from complete public data; preserved",
            )
            record = saved
        else:
            graph = graph_from_arrays(name, *values[:-1], selected, values[-1])
            if source == "planetoid_public":
                record = save_graph(graph, cache)
                with manifest_path.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, indent=2, allow_nan=False)
            else:
                record = {
                    "schema": SCHEMA,
                    "statistics": graph.metadata,
                    "name": graph.name,
                    "sha256": None,
                    "debug_fixture_not_public_data": True,
                }
        graphs[name], processed[name] = graph, record
        print(
            f"[data] {graph.name} nodes={graph.num_nodes} features={graph.num_features} "
            f"classes={graph.num_classes} edges={graph.edges.shape[1]} "
            f"paths={graph.paths.shape[1]} "
            f"train/val/test={graph.metadata['train']}/{graph.metadata['validation']}/"
            f"{graph.metadata['test']} "
            "sampling_ratio=1.0",
            flush=True,
        )
    manifest = {
        "schema": SCHEMA,
        "data_source": source,
        "profile": config["profile"],
        "created_utc": datetime.now(UTC).isoformat(),
        "datasets": processed,
        "raw": raw_manifests,
        "selected_workers": selected,
        "cpu_calibration": trials,
        "all_nodes_edges_paths": True,
        "actual_citation_data": source == "planetoid_public",
        "official_source": "https://github.com/kimiyoung/planetoid",
        "public_loader_reference": "https://github.com/tkipf/gcn/blob/master/gcn/utils.py",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "data_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return graphs, manifest
