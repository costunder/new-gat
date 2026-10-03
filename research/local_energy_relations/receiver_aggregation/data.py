"""Read every numeric source snapshot; never regenerate or mutate prior results."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np

from ...wedge_propagation.classification.data import PLANETOID_COMMIT, RAW_BASE, RAW_SHA256
from ...wedge_propagation.data import make_specs
from ..contract import validate_config as validate_parent_config
from ..data import (
    AuditCase,
    _available_cpus,
    feature_content_hash,
    graph_content_hash,
    load_case,
)
from ..topology import CORRESPONDENCE_KINDS, LocalTopology
from .contract import digest, file_sha256, read_json, validate_config, write_json

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_file(root, relative):
    _require(
        isinstance(relative, str) and bool(relative) and "\\" not in relative,
        "source reference must be a portable relative path",
    )
    path = Path(relative)
    _require(
        not path.is_absolute() and ".." not in path.parts,
        "source reference leaves declared source directory",
    )
    resolved = (root / path).resolve()
    _require(
        resolved.is_relative_to(root) and resolved.is_file(),
        f"source file missing or outside source directory: {relative}",
    )
    return resolved


def _sha(value):
    _require(isinstance(value, str) and _SHA.fullmatch(value), "invalid source SHA256")
    return value


def _checked(root, relative, expected):
    path = _safe_file(root, relative)
    _require(file_sha256(path) == _sha(expected), f"source checksum mismatch: {relative}")
    return path


def _offsets(value, count, total, name):
    _require(
        value.shape == (count + 1,)
        and value[0] == 0
        and value[-1] == total
        and np.all(np.diff(value) >= 0),
        f"topology {name} offsets differ",
    )


def _in_sorted(keys, query):
    positions = np.searchsorted(keys, query)
    return (positions < len(keys)) & (keys[np.minimum(positions, len(keys) - 1)] == query)


def validate_topology(top, case, record):
    """Validate exact saved local/overlap coverage without rebuilding a replacement."""
    n, e, nl, ml, pairs = (
        top.n,
        top.num_edges,
        top.num_local_nodes,
        top.num_local_edges,
        top.num_pairs,
    )
    _require(
        n == case.num_nodes and np.array_equal(top.edges, case.edges.numpy()),
        "saved topology disagrees with original physical graph",
    )
    _require(
        top.local_edge_nodes.shape == (2, ml) and top.pair_centers.shape == (2, 2 * e),
        "topology incidence or directed-pair shape differs",
    )
    _require(
        np.array_equal(top.pair_centers, np.concatenate((top.edges, top.edges[::-1]), axis=1)),
        "topology must retain both directions of every physical center pair",
    )
    _offsets(top.node_offsets, n, nl, "local node")
    _offsets(top.edge_offsets, n, ml, "local edge")
    _require(
        np.array_equal(top.graph_node_offsets, [0, n])
        and np.array_equal(top.graph_edge_offsets, [0, e]),
        "individual saved topology must contain one complete physical graph",
    )
    for values, limit, name in (
        (top.local_node_global, n, "local node global"),
        (top.local_edge_global, e, "local edge global"),
        (top.local_edge_nodes, nl, "incidence positions"),
        (top.center_positions, nl, "center positions"),
    ):
        _require(np.all(values >= 0) and np.all(values < limit), f"invalid topology {name}")
    _require(
        top.local_node_center.shape == (nl,)
        and top.local_edge_center.shape == (ml,)
        and top.local_degree.shape == (nl,)
        and top.center_positions.shape == (n,),
        "topology occurrence vector shape differs",
    )
    _require(
        np.array_equal(top.local_node_center, np.repeat(np.arange(n), np.diff(top.node_offsets)))
        and np.array_equal(
            top.local_edge_center, np.repeat(np.arange(n), np.diff(top.edge_offsets))
        ),
        "topology local occurrence ownership differs",
    )
    node_keys = top.local_node_center * n + top.local_node_global
    _require(np.all(np.diff(node_keys) > 0), "local node identities must be sorted unique")
    physical_degree = np.bincount(top.edges.ravel(), minlength=n)
    _require(
        np.array_equal(np.diff(top.node_offsets), physical_degree + 1),
        "induced one-hop local node set has missing or extra nodes",
    )
    _require(
        np.array_equal(top.local_node_global[top.center_positions], np.arange(n))
        and np.array_equal(top.local_node_center[top.center_positions], np.arange(n)),
        "saved centers must refer to their original node identities",
    )
    adjacency_keys = np.sort(
        np.concatenate(
            (
                top.edges[0] * n + top.edges[1],
                top.edges[1] * n + top.edges[0],
                np.arange(n) * (n + 1),
            )
        )
    )
    _require(
        np.all(_in_sorted(adjacency_keys, node_keys)), "local graph contains non-neighbor nodes"
    )
    _require(
        np.array_equal(
            top.local_node_global[top.local_edge_nodes], top.edges[:, top.local_edge_global]
        ),
        "saved local incidence orientation or physical edge identity differs",
    )
    _require(
        np.array_equal(
            top.local_node_center[top.local_edge_nodes],
            np.broadcast_to(top.local_edge_center, (2, ml)),
        ),
        "local edge endpoint is owned by another local graph",
    )
    if ml:
        _require(
            np.all(np.diff(top.local_edge_center * max(e, 1) + top.local_edge_global) > 0),
            "local physical edge identities must be sorted unique",
        )
    _require(
        np.array_equal(np.bincount(top.local_edge_nodes.ravel(), minlength=nl), top.local_degree),
        "saved local degree disagrees with incidence",
    )
    neighbors = [set() for _ in range(n)]
    for a, b in top.edges.T:
        neighbors[a].add(int(b))
        neighbors[b].add(int(a))
    expected_multiplicity = np.asarray(
        [2 + len(neighbors[a] & neighbors[b]) for a, b in top.edges.T]
    )
    _require(
        np.array_equal(np.bincount(top.local_edge_global, minlength=e), expected_multiplicity),
        "induced local edges omit or repeat an original physical edge occurrence",
    )
    counts = {}
    for index, kind in enumerate(CORRESPONDENCE_KINDS):
        pair = getattr(top, kind + "_pair")
        left = getattr(top, kind + "_left")
        nodes = kind.endswith("node")
        limit = nl if nodes else ml
        centers = top.local_node_center if nodes else top.local_edge_center
        _require(
            pair.ndim == left.ndim == 1
            and pair.shape == left.shape
            and np.all(pair >= 0)
            and np.all(pair < pairs)
            and np.all(left >= 0)
            and np.all(left < limit),
            f"invalid {kind} correspondence",
        )
        if len(pair):
            _require(
                np.all(np.diff(pair * max(limit, 1) + left) > 0),
                f"{kind} correspondences must be sorted and unique",
            )
        _require(
            np.array_equal(centers[left], top.pair_centers[0, pair]),
            f"{kind} source occurrence belongs to a different sender",
        )
        offsets = np.asarray(top.correspondence_offsets[index])
        _offsets(offsets, pairs, len(pair), kind)
        counts[kind] = np.bincount(pair, minlength=pairs)
        _require(np.array_equal(np.diff(offsets), counts[kind]), f"{kind} offsets are incomplete")
        receiver = top.pair_centers[1, pair]
        if kind.startswith("shared"):
            right = getattr(top, kind + "_right")
            _require(
                right.shape == left.shape
                and np.all(right >= 0)
                and np.all(right < limit)
                and np.array_equal(centers[right], receiver),
                f"{kind} target correspondence invalid",
            )
            ids = top.local_node_global if nodes else top.local_edge_global
            _require(np.array_equal(ids[left], ids[right]), f"{kind} original identity differs")
        elif nodes:
            _require(
                not np.any(_in_sorted(node_keys, receiver * n + top.local_node_global[left])),
                "omitted node is actually present in receiver",
            )
        else:
            endpoints = top.edges[:, top.local_edge_global[left]]
            inside = _in_sorted(node_keys, receiver * n + endpoints[0]).astype(int)
            inside += _in_sorted(node_keys, receiver * n + endpoints[1]).astype(int)
            _require(
                np.all(inside == (1 if kind == "boundary_edge" else 0)),
                f"{kind} edge endpoint membership differs",
            )
    sender = top.pair_centers[0]
    _require(
        np.array_equal(
            counts["shared_node"] + counts["omitted_node"], np.diff(top.node_offsets)[sender]
        ),
        "sender node transfer correspondence is incomplete",
    )
    _require(
        np.array_equal(
            counts["shared_edge"] + counts["boundary_edge"] + counts["omitted_edge"],
            np.diff(top.edge_offsets)[sender],
        ),
        "sender edge transfer correspondence is incomplete",
    )
    for name, value in {
        "nodes": n,
        "physical_edges": e,
        "local_node_occurrences": nl,
        "local_edge_occurrences": ml,
        "directed_center_pairs": pairs,
    }.items():
        _require(
            type(record.get(name)) is int and record[name] == value,
            f"topology manifest {name} differs",
        )
    return top


def load_topology(path, case, record):
    _require(file_sha256(path) == _sha(record["sha256"]), "topology snapshot checksum mismatch")
    names = {field.name for field in fields(LocalTopology)} - {"correspondence_offsets"}
    offset_names = {kind + "_offsets" for kind in CORRESPONDENCE_KINDS}
    with np.load(path, allow_pickle=False) as saved:
        _require(
            set(saved.files) == names | offset_names
            and len(saved.files) == len(names | offset_names),
            "saved topology arrays have missing or extra fields",
        )
        values = {name: np.array(saved[name], copy=True) for name in names | offset_names}
    _require(
        all(value.dtype == np.int64 for value in values.values()),
        "saved topology must use exact int64 numeric arrays",
    )
    _require(values["n"].shape == (), "saved topology node count must be scalar")
    values["n"] = int(values["n"])
    values["correspondence_offsets"] = tuple(
        tuple(int(v) for v in values.pop(kind + "_offsets")) for kind in CORRESPONDENCE_KINDS
    )
    return validate_topology(LocalTopology(**values), case, record)


def _load_one(arguments):
    root, input_record, topology_record, expected = arguments
    case = load_case(_checked(root, "inputs/" + input_record["file"], input_record["sha256"]))
    _require(
        case.graph_id == input_record["graph_id"] == topology_record["graph_id"],
        "source case original identity differs",
    )
    for key, value in {
        "family": case.family,
        "nodes": case.num_nodes,
        "physical_edges": case.edges.shape[1],
        "features": case.num_features,
        "actual_data": case.actual_data,
        "feature_mode": case.feature_mode,
        "graph_sha256": graph_content_hash(case),
        "feature_sha256": feature_content_hash(case),
        "dtype": "float64",
        "input_tensor_shape": list(case.features.shape),
        "all_nodes_edges_channels": True,
    }.items():
        _require(
            input_record.get(key) == value and type(input_record.get(key)) is type(value),
            f"source input manifest {key} differs: {case.graph_id}",
        )
    if hasattr(expected, "graph_seed"):
        _require(
            case.num_nodes == expected.num_nodes
            and case.num_features == expected.num_features
            and case.family == expected.family
            and not case.actual_data
            and case.feature_mode == "independent_scalar_columns",
            "source synthetic shape or feature interpretation differs",
        )
        for name in ("graph_seed", "feature_seed", "base_tree_seed"):
            _require(
                case.metadata.get(name) == getattr(expected, name), "source synthetic seed differs"
            )
    else:
        shape, actual = expected
        _require(
            case.num_nodes == shape["nodes"]
            and case.num_features == shape["features"]
            and ("physical_edges" not in shape or case.edges.shape[1] == shape["physical_edges"])
            and case.actual_data is actual
            and case.feature_mode == "vector_trace"
            and case.family == "citation",
            "source whole citation contract differs",
        )
        _require(
            case.metadata.get("encoder") is None
            and case.metadata.get("labels_and_masks_used") is False,
            "source numeric audit features must not use labels or an encoder",
        )
    topology = load_topology(_safe_file(root, topology_record["file"]), case, topology_record)
    return case, topology


def _parallel(arguments, workers):
    if workers == 1:
        return [_load_one(item) for item in arguments]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_load_one, arguments))


def _source_metadata(config, root):
    names = (
        "completion.json",
        "config.json",
        "source_manifest.json",
        "input_manifest.json",
        "topology_manifest.json",
    )
    metadata_hashes = {name: file_sha256(_safe_file(root, name)) for name in names}
    documents = {name: read_json(_safe_file(root, name)) for name in names}
    for name, expected_hash in metadata_hashes.items():
        _checked(root, name, expected_hash)
    original = validate_parent_config(documents["config.json"], config["profile"])
    _require(
        original["experiment"] == config["source"]["required_experiment"],
        "source experiment is not the complete initial local audit",
    )
    completed, inputs, topology, source = (
        documents[name]
        for name in (
            "completion.json",
            "input_manifest.json",
            "topology_manifest.json",
            "source_manifest.json",
        )
    )
    required = {
        key: config["source"][key]
        for key in (
            "profile",
            "graphs",
            "synthetic_graphs",
            "citation_graphs",
            "actual_citation_graphs",
            "synthetic_scalar_inputs",
        )
    }
    required.update(
        status="complete",
        debug=config["profile"] == "debug",
        sampling_ratio=1.0,
        all_nodes_edges_features=True,
        source_and_inputs_preserved=True,
        weights=config["weights"],
        states=config["states"],
        trainable_parameters=0,
        optimizer_updates=0,
        classifier_training_run=False,
    )
    for key, value in required.items():
        _require(
            completed.get(key) == value and type(completed.get(key)) is type(value),
            f"source completion {key} differs; complete matching profile required",
        )
    _require(
        source.get("code_digest") == digest(source.get("sha256"))
        and completed.get("source_digest") == source["code_digest"],
        "source completion implementation digest differs",
    )
    _require(
        isinstance(source.get("sha256"), dict) and source["sha256"],
        "source code hash closure missing",
    )
    for value in source["sha256"].values():
        _sha(value)
    _require(
        inputs.get("schema") == "local-energy-relations-input-v1"
        and topology.get("schema") == "induced-one-hop-local-correspondence-v1",
        "source snapshot schemas differ",
    )
    for key in (
        "profile",
        "graphs",
        "synthetic_graphs",
        "citation_graphs",
        "actual_citation_graphs",
        "synthetic_scalar_inputs",
    ):
        _require(inputs.get(key) == config["source"][key], f"source input {key} differs")
    _require(
        inputs.get("precision") == "float64"
        and inputs.get("sampling_ratio") == 1.0
        and inputs.get("encoder") is None
        and inputs.get("labels_and_masks_used") is False
        and inputs.get("citation_channels_are_vector_not_independent_replicates") is True,
        "source raw-input precision/feature interpretation differs",
    )
    _require(
        isinstance(inputs.get("inputs"), list)
        and inputs.get("input_content_digest") == digest(inputs["inputs"]),
        "source input content digest differs",
    )
    loader_path = _checked(
        root, inputs["citation_loader_manifest"], inputs["citation_loader_manifest_sha256"]
    )
    loader = read_json(loader_path)
    _require(
        loader.get("profile") == config["profile"] and loader.get("all_nodes_edges_paths") is True,
        "source citation loader scope differs",
    )
    _require(
        inputs.get("citation_raw_source") == loader.get("raw"),
        "source raw citation provenance differs",
    )
    citations = original["citation"]["datasets"]
    _require(
        set(loader.get("datasets", {})) == set(citations), "source citation loader coverage differs"
    )
    if config["profile"] == "full":
        _require(
            loader.get("actual_citation_data") is True and set(loader["raw"]) == set(citations),
            "source must contain actual complete citations",
        )
        for name in citations:
            raw = loader["raw"][name]
            expected = {
                f"ind.{name.lower()}.{part}": value for part, value in RAW_SHA256[name].items()
            }
            _require(
                raw.get("dataset") == name
                and raw.get("sha256") == expected
                and raw.get("repository_commit") == PLANETOID_COMMIT
                and raw.get("official_raw_base") == RAW_BASE,
                "source pinned public raw hashes differ",
            )
    else:
        _require(
            loader.get("actual_citation_data") is False and loader.get("raw") == {},
            "DEBUG fixtures must remain explicitly separate from actual citations",
        )
    metadata_hashes[loader_path.relative_to(root).as_posix()] = inputs[
        "citation_loader_manifest_sha256"
    ]
    return original, documents, metadata_hashes


def prepare_cases(config, source_dir, output_dir, workers="auto"):
    """Validate and cache all prior snapshots; output contains references and checksums."""
    validate_config(config)
    root, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    _require(root.is_dir(), "required completed source directory is missing")
    _require(
        not output.is_relative_to(root) and not root.is_relative_to(output),
        "new output must be separate from prior source results",
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "source_input_manifest.json"
    if destination.exists():
        raise FileExistsError("new source reference manifest already exists; files preserved")
    original, documents, metadata_hashes = _source_metadata(config, root)
    inputs = documents["input_manifest.json"]["inputs"]
    topology = documents["topology_manifest.json"]["graphs"]
    specs = make_specs(original["synthetic"]["master_seed"], config["profile"])
    citations = original["citation"]["datasets"]
    expected = {spec.graph_id: spec for spec in specs}
    expected.update(
        {
            name: (original["citation"]["expected_shapes"][name], config["profile"] == "full")
            for name in citations
        }
    )
    ids = list(expected)
    _require(
        [record.get("graph_id") for record in inputs] == ids
        and [record.get("graph_id") for record in topology] == ids,
        "source must retain every expected graph exactly once in original order",
    )
    _require(
        all(record.get("file") == record["graph_id"] + ".npz" for record in inputs)
        and all(
            record.get("file") == "topology/" + record["graph_id"] + ".npz" for record in topology
        ),
        "source numeric snapshot path mapping differs",
    )
    # Freeze input references before calibration/loading; detect concurrent source changes.
    closure = dict(metadata_hashes)
    closure.update({"inputs/" + r["file"]: r["sha256"] for r in inputs})
    closure.update({r["file"]: r["sha256"] for r in topology})
    available = _available_cpus()
    requested = config["runtime"]["cpu_workers"] if workers == "auto" else workers
    _require(
        requested == "auto" or (type(requested) is int and 1 <= requested <= available),
        "CPU workers must fit actual affinity/quota",
    )
    candidates = [w for w in (1, 2, 4, 8) if w <= available] if requested == "auto" else [requested]
    arguments = [
        (root, r, t, expected[r["graph_id"]]) for r, t in zip(inputs, topology, strict=True)
    ]
    # Include every whole citation snapshot: tiny synthetic files alone cannot
    # measure the feature decompression and semantic checks used by real data.
    # The final load still includes every contracted graph.
    largest = max(spec.num_nodes for spec in specs)
    probe = [
        item
        for item in arguments
        if not hasattr(item[3], "graph_seed") or item[3].num_nodes == largest
    ]
    calibration = []
    for number in candidates:
        started = time.perf_counter()
        _parallel(probe, number)
        elapsed = time.perf_counter() - started
        calibration.append(
            {
                "type": "source_cache_worker_calibration",
                "workers": number,
                "graphs": len(probe),
                "seconds": elapsed,
                "graphs_per_second": len(probe) / elapsed,
                "scope": "largest_synthetic_bucket_and_all_complete_citation_snapshots",
            }
        )
        print(
            f"[CPU source calibration] workers={number} graphs={len(probe)} seconds={elapsed:.4f}",
            flush=True,
        )
    selected = min(calibration, key=lambda row: row["seconds"])["workers"]
    started = time.perf_counter()
    loaded = _parallel(arguments, selected)
    cases, topologies = [item[0] for item in loaded], [item[1] for item in loaded]
    # The previous run must also have exported every configured center/pair feature unit.
    center_units = sum(
        top.n * (case.num_features if case.feature_mode == "independent_scalar_columns" else 1)
        for case, top in loaded
    )
    pair_units = sum(
        top.num_pairs
        * (case.num_features if case.feature_mode == "independent_scalar_columns" else 1)
        for case, top in loaded
    )
    _require(
        documents["completion.json"].get("raw_rows")
        == {
            "local": 6 * center_units,
            "transfer": 6 * pair_units,
            "relation": 30 * pair_units,
            "temporal": 4 * center_units,
        },
        "source completed raw record coverage differs",
    )
    manifest = {
        "schema": "receiver-aggregation-source-reference-v1",
        "profile": config["profile"],
        "source_dir": str(root),
        "graphs": len(cases),
        "sampling_ratio": 1.0,
        "all_nodes_edges_features": True,
        "no_regeneration_or_download": True,
        "source_completion": documents["completion.json"],
        "source_config": original,
        "source_code_digest": documents["source_manifest.json"]["code_digest"],
        "inputs": inputs,
        "topology": topology,
        "sha256": closure,
        "source_content_digest": digest(closure),
        "selected_workers": selected,
        "cpu_input_calibration": calibration,
        "load_seconds": time.perf_counter() - started,
        "citation_raw_source": documents["input_manifest.json"]["citation_raw_source"],
        "raw_files_redecoded": False,
    }
    assert_inputs_unchanged(root, manifest)
    write_json(destination, manifest)
    print(
        f"[source ready] graphs={len(cases)} workers={selected} all original channels retained",
        flush=True,
    )
    return cases, topologies, manifest


def assert_inputs_unchanged(source_dir, manifest):
    root = Path(source_dir).resolve()
    _require(
        str(root) == manifest["source_dir"]
        and manifest["source_content_digest"] == digest(manifest["sha256"]),
        "source reference manifest identity differs",
    )
    for relative, expected in manifest["sha256"].items():
        _checked(root, relative, expected)


validate_source_unchanged = assert_inputs_unchanged

__all__ = [
    "AuditCase",
    "prepare_cases",
    "load_topology",
    "validate_topology",
    "assert_inputs_unchanged",
    "validate_source_unchanged",
]
