"""Independent graph splits and disjoint batching for path-rule recovery.

Each feature column is an independent scalar realization, not a learned hidden
channel. Teacher weights are diagnostic labels only; the model generates its
own weights from ``x``. Static topology/targets are generated once on CPU.
Exact labeled-topology collisions trigger recorded, bounded seed retries;
graphs are never dropped and node/path counts are never capped.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import (
    GraphSpec,
    feature_content_hash,
    graph_content_hash,
    make_case,
)
from ..operators import (
    build_wedges,
    fixed_wedge_apply,
    laplacian_apply,
    wedge_transpose,
)

_RANDOM_FAMILIES = ("er", "tree", "tree_chord")
_HELD_FAMILIES = ("cycle", "star", "grid")
_SPLITS = ("train", "validation", "id", "size_ood", "family_ood", "family_size_ood")


@dataclass(frozen=True)
class RuleSpec:
    graph_spec: GraphSpec
    split: str
    pair_seed: int
    original_graph_seed: int
    resample_attempt: int = 0

    @property
    def graph_id(self) -> str:
        return self.graph_spec.graph_id

    @property
    def family(self) -> str:
        return self.graph_spec.family

    @property
    def num_nodes(self) -> int:
        return self.graph_spec.num_nodes


@dataclass(frozen=True)
class RuleCase:
    graph_id: str
    family: str
    split: str
    num_nodes: int
    edges: torch.Tensor
    wedges: torch.Tensor
    pair_edges: torch.Tensor
    pair_coefficients: torch.Tensor
    features: torch.Tensor
    teacher_c: torch.Tensor
    lx: torch.Tensor
    l2x: torch.Tensor
    qx: torch.Tensor
    target_path: torch.Tensor
    graph_seed: int
    feature_seed: int
    pair_seed: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RuleBatch:
    x: torch.Tensor
    edges: torch.Tensor
    wedges: torch.Tensor
    path_graph: torch.Tensor
    node_graph: torch.Tensor
    pair_edges: torch.Tensor
    pair_coefficients: torch.Tensor
    num_graphs: int
    graph_ids: tuple[str, ...]
    num_nodes: torch.Tensor
    lx: torch.Tensor
    l2x: torch.Tensor
    qx: torch.Tensor
    teacher_c: torch.Tensor
    targets: dict[str, torch.Tensor]


def _seed(master_seed: int, stream: str, graph_id: str) -> int:
    payload = f"wedge-rule-data-v1\0{master_seed}\0{stream}\0{graph_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def make_specs(config: dict[str, Any]) -> list[RuleSpec]:
    """Derive independent streams and every declared split without defaults."""

    master_seed = config["master_seed"]
    if isinstance(master_seed, bool) or not isinstance(master_seed, Integral) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    profile = config["profile"]
    if profile not in ("full", "debug"):
        raise ValueError("profile must explicitly be 'full' or 'debug'")
    sizes = {
        label: tuple(_positive_integer(n, f"{label} node count") for n in config[label])
        for label in ("train_sizes", "ood_sizes")
    }
    if any(not values or len(values) != len(set(values)) for values in sizes.values()):
        raise ValueError("size lists must be nonempty and contain no duplicates")
    if min(sizes["train_sizes"] + sizes["ood_sizes"]) < 5:
        raise ValueError("ER with expected degree 4 needs at least five nodes")
    if min(sizes["ood_sizes"]) <= max(sizes["train_sizes"]):
        raise ValueError("size-OOD nodes must exceed every training size")
    if min(sizes["train_sizes"]) == 5:
        raise ValueError("n=5 makes ER deterministic and prevents independent topology splits")
    feature_count = _positive_integer(config["features"], "features")
    draws = {
        "train": _positive_integer(config["train_draws"], "train_draws"),
        "validation": _positive_integer(config["val_draws"], "val_draws"),
        "id": _positive_integer(config["id_draws"], "id_draws"),
        "size_ood": _positive_integer(config["size_ood_draws"], "size_ood_draws"),
        "family_ood": 1,
        "family_size_ood": 1,
    }
    _positive_integer(config["duplicate_max_attempts"], "duplicate_max_attempts")
    result = []
    for split in _SPLITS:
        node_sizes = sizes["ood_sizes"] if "size_ood" in split else sizes["train_sizes"]
        families = _HELD_FAMILIES if split.startswith("family_") else _RANDOM_FAMILIES
        for n in node_sizes:
            for family in families:
                columns = None
                if family == "grid":
                    columns = _positive_integer(config["grid_columns"][str(n)], "grid columns")
                    if n % columns:
                        raise ValueError("grid columns must divide the full node count")
                for draw in range(draws[split]):
                    graph_id = f"rule-{profile}-{split}-{family}-n{n}-draw{draw}"
                    graph_seed = _seed(int(master_seed), "graph", graph_id)
                    base_tree_seed = None
                    if family == "tree_chord":
                        tree_id = graph_id.replace("-tree_chord-", "-tree-")
                        base_tree_seed = _seed(int(master_seed), "graph", tree_id)
                    result.append(RuleSpec(
                        graph_spec=GraphSpec(
                            graph_id=graph_id,
                            family=family,
                            num_nodes=n,
                            num_features=feature_count,
                            graph_seed=graph_seed,
                            feature_seed=_seed(int(master_seed), "feature", graph_id),
                            grid_columns=columns,
                            base_tree_seed=base_tree_seed,
                        ),
                        split=split,
                        pair_seed=_seed(int(master_seed), "random-pair", graph_id),
                        original_graph_seed=graph_seed,
                    ))
    return result


def _random_pairs(edges: torch.Tensor, count: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Unique physical-edge pairs with traversal signs and sqrt(6) row norm.

    The resulting operator row is ``s2 * B[e2] - s1 * B[e1]``. Returned
    coefficients include the common row scale; they multiply the two gradients
    before both gate construction and the transpose operation.
    """

    edge_count = edges.shape[1]
    available = edge_count * (edge_count - 1) // 2
    if count > available:
        raise ValueError("requested random pair count exceeds all distinct physical-edge pairs")
    if count == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((2, 0), dtype=torch.float64)
    rng = np.random.default_rng(seed)
    left, right = np.triu_indices(edge_count, k=1)
    indices = rng.choice(available, size=count, replace=False)
    pair = np.vstack((left[indices], right[indices]))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(2, count))
    endpoint = edges.numpy()
    first, second = endpoint[:, pair[0]], endpoint[:, pair[1]]
    dot = ((first[0] == second[0]).astype(np.int64)
           + (first[1] == second[1]).astype(np.int64)
           - (first[0] == second[1]).astype(np.int64)
           - (first[1] == second[0]).astype(np.int64))
    norm_squared = 4 - 2 * signs[0] * signs[1] * dot
    coefficients = signs * np.sqrt(6 / norm_squared)[None, :]
    return torch.from_numpy(pair.copy()), torch.from_numpy(coefficients)


def _all_distances(edges: np.ndarray, n: int) -> np.ndarray:
    neighbors = [[] for _ in range(n)]
    for u, v in edges.T:
        neighbors[u].append(int(v))
        neighbors[v].append(int(u))
    distances = np.full((n, n), -1, dtype=np.int64)
    for source in range(n):
        distances[source, source] = 0
        frontier = deque([source])
        while frontier:
            current = frontier.popleft()
            for neighbor in neighbors[current]:
                if distances[source, neighbor] < 0:
                    distances[source, neighbor] = distances[source, current] + 1
                    frontier.append(neighbor)
    return distances


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("min", "mean", "max", "std")}
    return {"min": float(values.min()), "mean": float(values.mean()),
            "max": float(values.max()), "std": float(values.std())}


def _pair_geometry(edges: torch.Tensor, pairs: torch.Tensor,
                   coefficients: torch.Tensor, n: int) -> dict[str, Any]:
    physical = edges.numpy()
    pair = pairs.numpy()
    coefficients_np = coefficients.numpy()
    count = pair.shape[1]
    edge_usage = np.bincount(pair.reshape(-1), minlength=physical.shape[1])
    if count == 0:
        return {"pair_count": 0, "pair_shared_center_fraction": None,
                "pair_row_norm": _stats(np.empty(0)),
                "pair_node_support": _stats(np.empty(0)),
                "pair_span_hops": _stats(np.empty(0)),
                "pair_disconnected_support_fraction": None,
                "pair_edge_usage": _stats(edge_usage), "pair_trace": 0.0,
                "pair_operator_norm": 0.0}
    first, second = physical[:, pair[0]], physical[:, pair[1]]
    nodes = np.vstack((first, second))
    s1, s2 = coefficients_np
    # s2*B[e2] - s1*B[e1], with physical B rows (-1,+1).
    weights = np.vstack((s1, -s1, -s2, s2))
    equality = nodes[:, None, :] == nodes[None, :, :]
    combined = (equality * weights[None, :, :]).sum(axis=1)
    earlier = np.tril(np.ones((4, 4), dtype=bool), k=-1)
    first_occurrence = ~(equality & earlier[:, :, None]).any(axis=1)
    support = first_occurrence & (np.abs(combined) > 1e-12)
    row_norms = np.sqrt(((combined ** 2) * first_occurrence).sum(axis=0))
    shared = ((first[:, None, :] == second[None, :, :]).any(axis=(0, 1)))
    distances = _all_distances(physical, n)
    pair_span = np.zeros(count, dtype=np.int64)
    disconnected = np.zeros(count, dtype=bool)
    pair_matrix = np.zeros((n, n), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            np.add.at(pair_matrix, (nodes[a], nodes[b]), weights[a] * weights[b])
            active = support[a] & support[b]
            value = distances[nodes[a], nodes[b]]
            disconnected |= active & (value < 0)
            pair_span = np.maximum(pair_span, np.where(active, value, 0))
    return {
        "pair_count": count,
        "pair_shared_center_fraction": float(shared.mean()),
        "pair_row_norm": _stats(row_norms),
        "pair_node_support": _stats(support.sum(axis=0)),
        "pair_span_hops": _stats(pair_span[~disconnected]),
        "pair_disconnected_support_fraction": float(disconnected.mean()),
        "pair_edge_usage": _stats(edge_usage),
        "pair_trace": float(np.trace(pair_matrix)),
        "pair_operator_norm": float(np.linalg.eigvalsh(pair_matrix)[-1]),
    }


def prepare_case(spec: RuleSpec, teacher: dict[str, Any]) -> RuleCase:
    """Prepare one static float64 case; no optimizer or model input labels."""

    # Import lazily to avoid coupling spec generation to model initialization.
    from .model import teacher_weights

    case = make_case(spec.graph_spec)
    wedges = build_wedges(case.edges, case.num_nodes)
    x = case.features
    g1 = x[wedges[1]] - x[wedges[0]]
    g2 = x[wedges[2]] - x[wedges[1]]
    path_graph = torch.zeros(wedges.shape[1], dtype=torch.long)
    with torch.no_grad():
        c = teacher_weights(g1, g2, path_graph, num_graphs=1, **teacher).detach()
        lx = laplacian_apply(case.edges, x)
        l2x = laplacian_apply(case.edges, lx)
        qx = fixed_wedge_apply(wedges, x)
        target = wedge_transpose(wedges, c * (g2 - g1), case.num_nodes)
    pairs, coefficients = _random_pairs(case.edges, wedges.shape[1], spec.pair_seed)
    metadata = _pair_geometry(case.edges, pairs, coefficients, case.num_nodes)
    edge = case.edges.numpy()
    path = wedges.numpy()
    edge_keys = edge[0] * case.num_nodes + edge[1]
    true_keys = np.vstack((
        np.minimum(path[0], path[1]) * case.num_nodes + np.maximum(path[0], path[1]),
        np.minimum(path[1], path[2]) * case.num_nodes + np.maximum(path[1], path[2]),
    ))
    true_pairs = np.searchsorted(edge_keys, true_keys)
    true_usage = np.bincount(true_pairs.reshape(-1), minlength=case.edges.shape[1])
    random_usage = np.bincount(pairs.numpy().reshape(-1), minlength=case.edges.shape[1])
    metadata.update({
        "base_tree_seed": case.base_tree_seed,
        "base_tree_graph_id": (case.graph_id.replace("-tree_chord-", "-tree-")
                               if case.family == "tree_chord" else None),
        "original_graph_seed": spec.original_graph_seed,
        "resample_attempt": spec.resample_attempt,
        "true_wedge_trace": float(6 * wedges.shape[1]),
        "true_edge_usage": _stats(true_usage),
        "edge_usage_L1_difference": int(np.abs(true_usage - random_usage).sum()),
        "edge_usage_exact_match": bool(np.array_equal(true_usage, random_usage)),
        "pair_donor_mapping": "random-pair column p receives true-wedge column p",
        "teacher": dict(teacher),
    })
    for name, value in (("teacher_c", c), ("lx", lx), ("l2x", l2x),
                        ("qx", qx), ("target_path", target)):
        if not bool(torch.isfinite(value).all()):
            raise ArithmeticError(f"nonfinite {name} in {case.graph_id}")
    return RuleCase(
        graph_id=case.graph_id, family=case.family, split=spec.split,
        num_nodes=case.num_nodes, edges=case.edges, wedges=wedges,
        pair_edges=pairs, pair_coefficients=coefficients, features=x, teacher_c=c,
        lx=lx, l2x=l2x, qx=qx, target_path=target,
        graph_seed=case.graph_seed, feature_seed=case.feature_seed,
        pair_seed=spec.pair_seed, metadata=metadata,
    )


def prepare_cases(config: dict[str, Any], workers: int) -> list[RuleCase]:
    """Parallel initial generation, followed by deterministic collision repair.

    Duplicate prevention conditions topology draws on exclusion of already
    accepted exact labeled graphs. It does not establish isomorphism-level
    independence. Every collision and retry is preserved in case metadata.
    """

    worker_count = _positive_integer(workers, "workers")
    specs = make_specs(config)
    teacher = config["teacher"]
    if worker_count == 1:
        initial = [prepare_case(spec, teacher) for spec in specs]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            initial = list(executor.map(lambda spec: prepare_case(spec, teacher), specs))
    initial_by_id = {case.graph_id: case for case in initial}
    accepted = {}
    content_owner: dict[str, str] = {}
    # Reserve held-family deterministic graphs before any random collision repair.
    ordered = ([spec for spec in specs if spec.family in _HELD_FAMILIES]
               + [spec for spec in specs if spec.family in _RANDOM_FAMILIES])
    maximum = _positive_integer(config["duplicate_max_attempts"], "duplicate_max_attempts")
    for original in ordered:
        current = original
        case = initial_by_id[original.graph_id]
        if original.family == "tree_chord":
            tree_id = original.graph_id.replace("-tree_chord-", "-tree-")
            base_seed = accepted[tree_id].graph_seed
            if base_seed != current.graph_spec.base_tree_seed:
                current = replace(current, graph_spec=replace(
                    current.graph_spec, base_tree_seed=base_seed))
                case = prepare_case(current, teacher)
        collisions = []
        while (digest := graph_content_hash(case)) in content_owner:
            if original.family in _HELD_FAMILIES:
                raise ValueError(f"deterministic graph collision: {case.graph_id}")
            collisions.append({"attempt": current.resample_attempt,
                               "other_graph_id": content_owner[digest],
                               "content_hash": digest})
            attempt = current.resample_attempt + 1
            if attempt > maximum:
                raise RuntimeError(
                    f"cannot obtain unique topology for {case.graph_id} after "
                    f"{maximum} recorded retries; no graph was dropped"
                )
            current = replace(current, resample_attempt=attempt, graph_spec=replace(
                current.graph_spec, graph_seed=_seed(
                    original.original_graph_seed, "topology-retry",
                    f"{original.graph_id}:{attempt}")))
            case = prepare_case(current, teacher)
        case.metadata["duplicate_collisions"] = collisions
        case.metadata["duplicate_max_attempts"] = maximum
        content_owner[digest] = case.graph_id
        accepted[case.graph_id] = case
    result = [accepted[spec.graph_id] for spec in specs]
    feature_owner = {}
    for case in result:
        digest = feature_content_hash(case)
        if digest in feature_owner:
            raise ValueError(
                f"duplicate full feature content: {case.graph_id}, {feature_owner[digest]}"
            )
        feature_owner[digest] = case.graph_id
    return result


def pack_cases(cases: list[RuleCase], device: torch.device | str,
               dtype: torch.dtype) -> RuleBatch:
    """Disjoint union across graphs; all scalar realizations run together."""

    if not cases:
        raise ValueError("cannot pack an empty graph batch")
    if not dtype.is_floating_point:
        raise TypeError("batch dtype must be real floating point")
    if len({case.features.shape[1] for case in cases}) != 1:
        raise ValueError("all graphs in a batch must have the same realization count")
    if len({case.graph_id for case in cases}) != len(cases):
        raise ValueError("a batch must not duplicate graph IDs")
    node_offset = edge_offset = 0
    edges, wedges, pairs = [], [], []
    for case in cases:
        edges.append(case.edges + node_offset)
        wedges.append(case.wedges + node_offset)
        pairs.append(case.pair_edges + edge_offset)
        node_offset += case.num_nodes
        edge_offset += case.edges.shape[1]
    node_counts = torch.tensor([case.num_nodes for case in cases], dtype=torch.long)
    path_counts = torch.tensor([case.wedges.shape[1] for case in cases], dtype=torch.long)
    host = {
        "x": torch.cat([case.features for case in cases]),
        "edges": torch.cat(edges, dim=1),
        "wedges": torch.cat(wedges, dim=1),
        "pair_edges": torch.cat(pairs, dim=1),
        "pair_coefficients": torch.cat([case.pair_coefficients for case in cases], dim=1),
        "node_graph": torch.repeat_interleave(torch.arange(len(cases)), node_counts),
        "path_graph": torch.repeat_interleave(torch.arange(len(cases)), path_counts),
        "num_nodes": node_counts,
        **{name: torch.cat([getattr(case, name) for case in cases])
           for name in ("lx", "l2x", "qx", "teacher_c", "target_path")},
    }
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA batch requested but unavailable; no CPU fallback")
    moved = {}
    for name, value in host.items():
        if value.is_floating_point():
            value = value.to(dtype=dtype)
        if device.type == "cuda":
            value = value.pin_memory()
        moved[name] = value.to(device, non_blocking=device.type == "cuda")
    return RuleBatch(
        x=moved["x"], edges=moved["edges"], wedges=moved["wedges"],
        path_graph=moved["path_graph"], node_graph=moved["node_graph"],
        pair_edges=moved["pair_edges"], pair_coefficients=moved["pair_coefficients"],
        num_graphs=len(cases), graph_ids=tuple(case.graph_id for case in cases),
        num_nodes=moved["num_nodes"], lx=moved["lx"], l2x=moved["l2x"], qx=moved["qx"],
        teacher_c=moved["teacher_c"],
        targets={"L": moved["lx"], "L2": moved["l2x"], "path": moved["target_path"]},
    )


def save_dataset(output_dir: Path | str, cases: list[RuleCase]) -> dict[str, Any]:
    """Exclusive data/manifest writes containing topology, inputs and all labels."""

    if not cases:
        raise ValueError("refusing to save an empty dataset")
    if len({case.graph_id for case in cases}) != len(cases):
        raise ValueError("duplicate graph ID in dataset")
    if len({graph_content_hash(case) for case in cases}) != len(cases):
        raise ValueError("duplicate exact graph content in dataset")
    if len({feature_content_hash(case) for case in cases}) != len(cases):
        raise ValueError("duplicate full feature content in dataset")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    dataset_path = directory / "dataset.npz"
    manifest_path = directory / "data_manifest.json"
    if dataset_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite dataset or manifest")
    arrays = {}
    rows = []
    for case in cases:
        for name in ("edges", "wedges", "pair_edges", "pair_coefficients", "features",
                     "teacher_c", "lx", "l2x", "qx", "target_path"):
            arrays[f"{case.graph_id}__{name}"] = getattr(case, name).numpy()
        arrays[f"{case.graph_id}__pair_donor_path"] = np.arange(case.wedges.shape[1])
        rows.append({
            "graph_id": case.graph_id, "family": case.family, "split": case.split,
            "num_nodes": case.num_nodes, "num_edges": case.edges.shape[1],
            "num_paths": case.wedges.shape[1], "num_features": case.features.shape[1],
            "graph_seed": case.graph_seed, "feature_seed": case.feature_seed,
            "pair_seed": case.pair_seed, "graph_content_hash": graph_content_hash(case),
            "feature_content_hash": feature_content_hash(case), "metadata": case.metadata,
        })
    with dataset_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = {
        "kind": "synthetic path-rule recovery; teacher tensors are labels/diagnostics",
        "graph_count": len(cases),
        "feature_realization_count": sum(case.features.shape[1] for case in cases),
        "split_graph_counts": dict(Counter(case.split for case in cases)),
        "split_feature_counts": dict(Counter({
            split: sum(case.features.shape[1] for case in cases if case.split == split)
            for split in _SPLITS
        })),
        "graph_content_unique": len({row["graph_content_hash"] for row in rows}) == len(rows),
        "feature_content_unique": len({row["feature_content_hash"] for row in rows}) == len(rows),
        "uniqueness_scope": (
            "exact labeled topology and full feature content; not graph isomorphism"
        ),
        "topology_sampling": "conditioned on excluding recorded exact-content duplicates",
        "duplicate_resample_total": sum(case.metadata["resample_attempt"] for case in cases),
        "dataset_file": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "cases": rows,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return manifest
