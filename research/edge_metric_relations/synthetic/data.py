"""Generate all declared graphs with a new, named independent scalar stream.

Only the established graph specification/generator is reused. Old target,
path-gate, random-pair and feature arrays are not used as new model inputs.
The collision repair matches the original labeled-topology contract. It does
not establish isomorphism-level independence.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ...local_energy_relations.topology import LocalTopology, batch_topologies, build_topology
from ...wedge_propagation.data import GraphCase, feature_content_hash, graph_content_hash, make_case
from ...wedge_propagation.learned.data import make_specs as original_specs, _seed as original_seed

SPLITS = ("train", "validation", "id", "size_ood", "family_ood", "family_size_ood")


@dataclass(frozen=True)
class SyntheticCase:
    graph: GraphCase
    split: str
    topology: LocalTopology
    metadata: dict

    @property
    def graph_id(self):
        return self.graph.graph_id


@dataclass(frozen=True)
class SyntheticBatch:
    cases: tuple[SyntheticCase, ...]
    geometry: object
    x: torch.Tensor

    @property
    def num_graphs(self):
        return len(self.cases)


def scalar_seed(master_seed: int, stream: str, graph_id: str) -> int:
    value = f"edge-metric-synthetic-data-v1\0{master_seed}\0{stream}\0{graph_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def make_specs(config):
    specs = original_specs(config)
    return [replace(spec, graph_spec=replace(
        spec.graph_spec,
        feature_seed=scalar_seed(config["master_seed"], config["feature_stream"], spec.graph_id),
    )) for spec in specs]


def prepare_cases(config, workers: int):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    specs = make_specs(config)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        initial = list(executor.map(lambda spec: make_case(spec.graph_spec), specs))
    by_id = {case.graph_id: case for case in initial}
    accepted, owners, metadata = {}, {}, {}
    held = {"cycle", "star", "grid"}
    ordered = [spec for spec in specs if spec.family in held] + [spec for spec in specs if spec.family not in held]
    for original in ordered:
        current = original
        case = by_id[original.graph_id]
        if original.family == "tree_chord":
            tree_id = original.graph_id.replace("-tree_chord-", "-tree-")
            seed = accepted[tree_id].graph_seed
            if seed != current.graph_spec.base_tree_seed:
                current = replace(current, graph_spec=replace(current.graph_spec, base_tree_seed=seed))
                case = make_case(current.graph_spec)
        collisions = []
        while (digest := graph_content_hash(case)) in owners:
            if original.family in held:
                raise ValueError(f"deterministic graph collision: {case.graph_id}")
            collisions.append({"attempt": current.resample_attempt, "other_graph_id": owners[digest], "content_hash": digest})
            attempt = current.resample_attempt + 1
            if attempt > config["duplicate_max_attempts"]:
                raise RuntimeError(f"topology collision retries exhausted: {case.graph_id}; no graph was dropped")
            current = replace(current, resample_attempt=attempt, graph_spec=replace(
                current.graph_spec, graph_seed=original_seed(original.original_graph_seed, "topology-retry", f"{original.graph_id}:{attempt}")))
            case = make_case(current.graph_spec)
        owners[digest] = case.graph_id
        accepted[case.graph_id] = case
        metadata[case.graph_id] = {
            "original_graph_seed": original.original_graph_seed,
            "resample_attempt": current.resample_attempt,
            "duplicate_collisions": collisions,
            "duplicate_max_attempts": config["duplicate_max_attempts"],
        }
    graphs = [accepted[spec.graph_id] for spec in specs]
    feature_hashes = [feature_content_hash(graph) for graph in graphs]
    if len(feature_hashes) != len(set(feature_hashes)):
        raise ValueError("duplicate complete scalar feature arrays")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tops = list(executor.map(lambda graph: build_topology(graph.num_nodes, graph.edges), graphs))
    return [SyntheticCase(graph, spec.split, top, metadata[graph.graph_id])
            for graph, spec, top in zip(graphs, specs, tops, strict=True)]


def pack_cases(cases, recipe, device, dtype):
    from ..geometry import prepare_geometry

    if not cases:
        raise ValueError("empty physical graph batch")
    if len({case.graph.features.shape[1] for case in cases}) != 1:
        raise ValueError("independent scalar realization counts disagree")
    top = batch_topologies([case.topology for case in cases])
    geometry = prepare_geometry(top, recipe).to(device=device, dtype=dtype)
    # [1,R,N,1]: a realization is a separate scalar field, never an F=R vector.
    x = torch.cat([case.graph.features for case in cases], dim=0).T[None, ..., None].to(device=device, dtype=dtype)
    return SyntheticBatch(tuple(cases), geometry, x)


def data_manifest(cases, config):
    counts = dict(Counter(case.split for case in cases))
    return {
        "experiment": "B_raw_message_recovery",
        "profile": config["profile"], "debug": config["profile"] == "debug",
        "intended_synthetic_data": True, "actual_citation_data": False,
        "graph_count": len(cases), "scalar_inputs": len(cases) * config["features"],
        "split_graph_counts": counts, "master_seed": config["master_seed"],
        "feature_stream": config["feature_stream"], "feature_channels": 1,
        "independent_scalar_realizations": config["features"],
        "all_nodes_edges_ego_occurrences_eligible_pairs_used": True,
        "topology_contract": "original wedge learned labeled splits; no isomorphism disjointness claim",
        "rows": [dict(
            graph_id=case.graph_id, split=case.split, family=case.graph.family,
            nodes=case.graph.num_nodes, edges=case.graph.edges.shape[1],
            local_nodes=case.topology.num_local_nodes, occurrences=case.topology.num_local_edges,
            graph_seed=case.graph.graph_seed, feature_seed=case.graph.feature_seed,
            base_tree_seed=case.graph.base_tree_seed,
            graph_sha256=graph_content_hash(case.graph), feature_sha256=feature_content_hash(case.graph),
            **case.metadata,
        ) for case in cases],
    }


def save_dataset(folder: Path, cases):
    folder.mkdir(parents=True, exist_ok=False)
    for case in cases:
        np.savez_compressed(folder / f"{case.graph_id}.npz",
                            edges=case.graph.edges.numpy(), x=case.graph.features.numpy(),
                            local_node_global=np.asarray(case.topology.local_node_global),
                            local_node_center=np.asarray(case.topology.local_node_center),
                            local_edge_global=np.asarray(case.topology.local_edge_global),
                            local_edge_center=np.asarray(case.topology.local_edge_center),
                            local_edge_nodes=np.asarray(case.topology.local_edge_nodes),
                            center_positions=np.asarray(case.topology.center_positions))
