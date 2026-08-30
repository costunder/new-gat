"""Deterministic S1--S4 synthetic paper datasets for conductance GAT.

All generated examples use sparse ``edge_index`` tensors.  The cache key is a
canonical hash of the generation request and each manifest contains both a
content fingerprint (independent of ``torch.save`` metadata) and the serialized
artifact checksum.  There are no network or PyG dependencies in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as nnf

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_publish,
    atomic_write_json,
)

from .sparse import edge_divergence, edge_gradient, weighted_degree

SCHEMA_VERSION = 2
GENERATOR_VERSION = "conductance-s1-s4-edge-index-v6-full-only"


def _generator(seed: int) -> torch.Generator:
    result = torch.Generator(device="cpu")
    result.manual_seed(int(seed))
    return result


def _canonical_edges(pairs: list[tuple[int, int]], num_nodes: int, seed: int) -> Tensor:
    unique = sorted({(min(a, b), max(a, b)) for a, b in pairs if a != b})
    if not unique:
        raise ValueError("a graph needs at least one non-self edge")
    generator = _generator(seed)
    oriented: list[tuple[int, int]] = []
    signs = torch.randint(0, 2, (len(unique),), generator=generator)
    for index, (first, second) in enumerate(unique):
        oriented.append((second, first) if int(signs[index]) else (first, second))
    result = torch.tensor(oriented, dtype=torch.long).t().contiguous()
    if int(result.max()) >= num_nodes:
        raise ValueError("edge endpoint outside graph")
    return result


def make_graph(num_nodes: int, family: str, seed: int) -> Tensor:
    """Generate a connected simple undirected graph with arbitrary orientation."""

    if num_nodes < 4:
        raise ValueError("num_nodes must be at least four")
    generator = _generator(seed)
    pairs: list[tuple[int, int]] = []
    if family == "er":
        # A random recursive tree guarantees connectedness before extra edges.
        for node in range(1, num_nodes):
            parent = int(torch.randint(0, node, (1,), generator=generator))
            pairs.append((parent, node))
        target_edges = min(num_nodes * (num_nodes - 1) // 2, 2 * num_nodes)
        candidates = torch.randperm(num_nodes * num_nodes, generator=generator).tolist()
        for flat in candidates:
            a, b = divmod(flat, num_nodes)
            if a < b:
                pairs.append((a, b))
            if len(set((min(x, y), max(x, y)) for x, y in pairs)) >= target_edges:
                break
    elif family == "rgg":
        coordinates = torch.rand((num_nodes, 2), generator=generator)
        distances = torch.cdist(coordinates, coordinates)
        candidates = sorted(
            (float(distances[a, b]), a, b)
            for a in range(num_nodes)
            for b in range(a + 1, num_nodes)
        )
        # A Euclidean minimum spanning tree makes the random geometric graph
        # connected; the shortest remaining pairs define its radius-like edge
        # set. No arbitrary long random-tree edges are injected.
        parents = list(range(num_nodes))

        def find(node: int) -> int:
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node

        for _, first, second in candidates:
            root_first, root_second = find(first), find(second)
            if root_first != root_second:
                parents[root_second] = root_first
                pairs.append((first, second))
            if len(pairs) == num_nodes - 1:
                break
        target_edges = min(num_nodes * (num_nodes - 1) // 2, 2 * num_nodes)
        for _, first, second in candidates:
            pairs.append((first, second))
            if len(set((min(x, y), max(x, y)) for x, y in pairs)) >= target_edges:
                break
    elif family == "grid":
        columns = max(2, int(math.ceil(math.sqrt(num_nodes))))
        for node in range(num_nodes):
            row, column = divmod(node, columns)
            if column and node - 1 >= 0:
                pairs.append((node - 1, node))
            above = node - columns
            if row and above >= 0:
                pairs.append((above, node))
    elif family == "barbell":
        left = max(2, num_nodes // 3)
        right_start = num_nodes - left
        for start, stop in ((0, left), (right_start, num_nodes)):
            for a in range(start, stop):
                for b in range(a + 1, stop):
                    pairs.append((a, b))
        for node in range(left - 1, right_start):
            pairs.append((node, node + 1))
    else:
        raise ValueError(f"unknown graph family {family!r}")
    return _canonical_edges(pairs, num_nodes, seed + 991)


def make_edge_features(edge_index: Tensor, num_nodes: int, seed: int, width: int = 3) -> Tensor:
    if width < 3:
        raise ValueError("synthetic edge features require at least three channels")
    generator = _generator(seed)
    random_features = torch.randn((edge_index.shape[1], width), generator=generator)
    degree = torch.bincount(edge_index.reshape(-1), minlength=num_nodes).float()
    tail, head = edge_index
    random_features[:, 2] = (degree[tail] + degree[head]) / degree.max().clamp_min(1.0)
    return random_features


def static_conductance(edge_features: Tensor, contrast: float | None = None) -> Tensor:
    base = (
        0.85 * edge_features[:, 0]
        - 0.35 * edge_features[:, 1]
        + 0.25 * edge_features[:, 0].square()
        + 0.20 * torch.sin(edge_features[:, 1])
    )
    if contrast is None:
        return 0.15 + nnf.softplus(base)
    if contrast < 1:
        raise ValueError("contrast must be at least one")
    if contrast == 1:
        return torch.ones_like(base)
    centered = base - base.mean()
    span = centered.max() - centered.min()
    normalized = (centered - centered.min()) / span.clamp_min(1.0e-8)
    return torch.exp((normalized - 0.5) * math.log(float(contrast)))


def nonlinear_conductance(edge_features: Tensor, gradient: Tensor) -> Tensor:
    base = static_conductance(edge_features)
    state_factor = 0.65 + 0.70 * torch.sigmoid(1.5 * (gradient.abs().mean(dim=1) - 0.7))
    return base * state_factor


def _sample_potential(
    num_nodes: int,
    channels: int,
    seed: int,
    active_fraction: float = 1.0,
) -> Tensor:
    generator = _generator(seed)
    state = torch.randn((num_nodes, channels), generator=generator)
    if active_fraction < 1.0:
        active_count = max(2, int(round(active_fraction * num_nodes)))
        active = torch.randperm(num_nodes, generator=generator)[:active_count]
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[active] = True
        state[~mask] = 0.0
    return state - state.mean(dim=0, keepdim=True)


def _safe_step(edge_index: Tensor, conductance: Tensor, num_nodes: int, requested: float) -> float:
    maximum = float(weighted_degree(edge_index, conductance, num_nodes).max())
    return min(float(requested), 0.80 / max(maximum, 1.0e-8))


def make_example(
    *,
    graph_id: str,
    num_nodes: int,
    family: str,
    graph_seed: int,
    excitation_seed: int,
    channels: int = 2,
    active_fraction: float = 1.0,
    snr_db: float | None = None,
    contrast: float | None = None,
    nonlinear: bool = False,
    requested_step: float = 0.025,
) -> dict[str, Any]:
    edges = make_graph(num_nodes, family, graph_seed)
    feature_width = 4 if contrast is not None else 3
    features = make_edge_features(edges, num_nodes, graph_seed + 31, feature_width)
    if contrast is not None:
        features[:, 3] = math.log10(float(contrast)) / 2.0
    state = _sample_potential(num_nodes, channels, excitation_seed, active_fraction)
    gradient = edge_gradient(edges, state)
    conductance = (
        nonlinear_conductance(features, gradient)
        if nonlinear
        else static_conductance(features, contrast)
    )
    flux = conductance[:, None] * gradient
    message = edge_divergence(edges, flux, num_nodes)
    step = _safe_step(edges, conductance, num_nodes, requested_step)
    next_state = state - step * message
    observed_flux = flux.clone()
    if snr_db is not None:
        generator = _generator(excitation_seed + 701)
        signal_rms = flux.square().mean().sqrt()
        noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
        observed_flux = flux + noise_rms * torch.randn(flux.shape, generator=generator)
    observed_node_message = edge_divergence(edges, observed_flux, num_nodes)
    excited = gradient.abs().amax(dim=1) > 1.0e-6
    return {
        "graph_id": graph_id,
        "edge_index": edges,
        "edge_features": features,
        "node_state": state,
        "true_conductance": conductance,
        "true_gradient": gradient,
        "true_flux": flux,
        "true_node_message": message,
        "true_next_state": next_state,
        "observed_flux": observed_flux,
        "observed_node_message": observed_node_message,
        "step_size": step,
        "metadata": {
            "family": family,
            "num_nodes": num_nodes,
            "contrast": contrast,
            "snr_db": "infinity" if snr_db is None else float(snr_db),
            "active_node_fraction": float(active_fraction),
            "excited_edge_fraction": float(excited.float().mean()),
        },
    }


def _vary_nodes(low: int, high: int, seed: int) -> int:
    return int(torch.randint(low, high + 1, (1,), generator=_generator(seed)))


def generate_s1(seed: int) -> dict[str, Any]:
    counts = (42, 9, 9)
    excitation_counts = (6, 3, 3)
    result: dict[str, Any] = {name: [] for name in ("train", "validation", "test", "seen_test")}
    offset = 0
    for split, graph_count, excitation_count in zip(
        ("train", "validation", "test"), counts, excitation_counts, strict=True
    ):
        for graph_number in range(graph_count):
            graph_seed = seed * 100_000 + offset * 101 + 11
            graph_id = f"s1-{split}-{graph_number:03d}"
            nodes = _vary_nodes(16, 32, graph_seed)
            family = "er" if graph_number % 2 == 0 else "rgg"
            for excitation in range(excitation_count):
                result[split].append(
                    make_example(
                        graph_id=graph_id,
                        num_nodes=nodes,
                        family=family,
                        graph_seed=graph_seed,
                        excitation_seed=graph_seed + 10_000 + excitation,
                    )
                )
            if split == "train":
                seen_count = 2
                for excitation in range(seen_count):
                    result["seen_test"].append(
                        make_example(
                            graph_id=graph_id,
                            num_nodes=nodes,
                            family=family,
                            graph_seed=graph_seed,
                            excitation_seed=graph_seed + 20_000 + excitation,
                        )
                    )
            offset += 1
    result["description"] = "S1 static shared-law identification; graph-ID split 70/15/15"
    return result


def _s2_protocol_counts() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return graph and per-graph excitation counts for train/validation/test."""

    return (28, 8, 16), (4, 3, 3)


def generate_s2(seed: int) -> dict[str, Any]:
    counts, excitation_counts = _s2_protocol_counts()
    result: dict[str, Any] = {name: [] for name in ("train", "validation", "test")}
    for split_number, (split, count, excitations) in enumerate(
        zip(("train", "validation", "test"), counts, excitation_counts, strict=True)
    ):
        for graph_number in range(count):
            graph_seed = seed * 200_000 + split_number * 20_000 + graph_number * 131 + 29
            if split == "test":
                low, high = (48, 96)
                family = "grid" if graph_number % 2 == 0 else "barbell"
            else:
                low, high = (16, 32)
                family = "er" if graph_number % 2 == 0 else "rgg"
            nodes = _vary_nodes(low, high, graph_seed)
            graph_id = f"s2-{split}-{family}-{graph_number:03d}"
            for excitation in range(excitations):
                result[split].append(
                    make_example(
                        graph_id=graph_id,
                        num_nodes=nodes,
                        family=family,
                        graph_seed=graph_seed,
                        excitation_seed=graph_seed + 30_000 + excitation,
                    )
                )
    result["description"] = "S2 ER/RGG n=16..32 to grid/barbell n=48..96 topology/size OOD"
    return result


def _make_trajectory(
    *,
    graph_id: str,
    num_nodes: int,
    family: str,
    graph_seed: int,
    trajectory_seed: int,
    horizon: int,
) -> dict[str, Any]:
    edges = make_graph(num_nodes, family, graph_seed)
    features = make_edge_features(edges, num_nodes, graph_seed + 31, 3)
    state = _sample_potential(num_nodes, 2, trajectory_seed)
    states = [state]
    conductances: list[Tensor] = []
    fluxes: list[Tensor] = []
    steps: list[float] = []
    for _ in range(horizon):
        gradient = edge_gradient(edges, state)
        conductance = nonlinear_conductance(features, gradient)
        flux = conductance[:, None] * gradient
        message = edge_divergence(edges, flux, num_nodes)
        step = _safe_step(edges, conductance, num_nodes, 0.025)
        state = state - step * message
        conductances.append(conductance)
        fluxes.append(flux)
        steps.append(step)
        states.append(state)
    return {
        "graph_id": graph_id,
        "edge_index": edges,
        "edge_features": features,
        "states": torch.stack(states),
        "conductances": torch.stack(conductances),
        "fluxes": torch.stack(fluxes),
        "steps": torch.tensor(steps),
        "metadata": {"family": family, "num_nodes": num_nodes, "horizon": horizon},
    }


def _trajectory_examples(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []
    edges = trajectory["edge_index"]
    for time in range(trajectory["conductances"].shape[0]):
        state = trajectory["states"][time]
        next_state = trajectory["states"][time + 1]
        conductance = trajectory["conductances"][time]
        gradient = edge_gradient(edges, state)
        flux = trajectory["fluxes"][time]
        examples.append(
            {
                "graph_id": trajectory["graph_id"],
                "edge_index": edges,
                "edge_features": trajectory["edge_features"],
                "node_state": state,
                "true_conductance": conductance,
                "true_gradient": gradient,
                "true_flux": flux,
                "true_node_message": edge_divergence(edges, flux, state.shape[0]),
                "true_next_state": next_state,
                "observed_flux": flux,
                "observed_node_message": edge_divergence(edges, flux, state.shape[0]),
                "step_size": float(trajectory["steps"][time]),
                "metadata": {**trajectory["metadata"], "time": time},
            }
        )
    return examples


def generate_s3(seed: int) -> dict[str, Any]:
    counts = (12, 3, 5)
    horizon = 50
    result: dict[str, Any] = {
        "train": [],
        "validation": [],
        "test": [],
        "rollout_test": [],
        "horizons": [1, 5, 10, 50],
    }
    for split_number, (split, count) in enumerate(
        zip(("train", "validation", "test"), counts, strict=True)
    ):
        for number in range(count):
            graph_seed = seed * 300_000 + split_number * 30_000 + number * 151 + 37
            nodes = _vary_nodes(18, 36, graph_seed)
            family = "er" if number % 2 == 0 else "rgg"
            trajectory = _make_trajectory(
                graph_id=f"s3-{split}-{number:03d}",
                num_nodes=nodes,
                family=family,
                graph_seed=graph_seed,
                trajectory_seed=graph_seed + 50_000,
                horizon=horizon,
            )
            result[split].extend(_trajectory_examples(trajectory))
            if split == "test":
                result["rollout_test"].append(trajectory)
    result["description"] = "S3 state-dependent positive nonlinear held-graph rollout"
    return result


def generate_s4(seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"train": [], "validation": [], "test": []}
    contrasts = (1.0, 10.0, 100.0)
    active_fractions = (1.0, 0.25)
    snrs: tuple[float | None, ...] = (None, 40.0, 20.0)
    graph_counts = (3, 1, 2)
    excitation_counts = (6, 2, 4)
    cell = 0
    for contrast in contrasts:
        for active_fraction in active_fractions:
            for snr in snrs:
                for split_number, (split, graph_count, excitation_count) in enumerate(
                    zip(
                        ("train", "validation", "test"),
                        graph_counts,
                        excitation_counts,
                        strict=True,
                    )
                ):
                    for graph_number in range(graph_count):
                        graph_seed = (
                            seed * 400_000
                            + cell * 10_000
                            + split_number * 2_000
                            + graph_number * 173
                            + 41
                        )
                        nodes = _vary_nodes(18, 32, graph_seed)
                        graph_id = (
                            f"s4-{split}-c{contrast:g}-a{active_fraction:g}-"
                            f"s{snr}-{graph_number:02d}"
                        )
                        for excitation in range(excitation_count):
                            result[split].append(
                                make_example(
                                    graph_id=graph_id,
                                    num_nodes=nodes,
                                    family="er" if cell % 2 == 0 else "rgg",
                                    graph_seed=graph_seed,
                                    excitation_seed=graph_seed + 70_000 + excitation,
                                    active_fraction=active_fraction,
                                    snr_db=snr,
                                    contrast=contrast,
                                    requested_step=0.01,
                                )
                            )
                cell += 1
    result["description"] = "S4 contrast x coverage x SNR identifiability robustness factorial"
    return result


def generate_core(seed: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "s1": generate_s1(seed + 101),
        "s2": generate_s2(seed + 202),
        "s3": generate_s3(seed + 303),
        "s4": generate_s4(seed + 404),
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _content_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(_canonical_json(list(tensor.shape)))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for child in item:
                update(child)
        else:
            digest.update(_canonical_json(item))

    update(value)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_ids(core: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for suite_name in ("s1", "s2", "s3", "s4"):
        result[suite_name] = {}
        for split in ("train", "validation", "test", "seen_test"):
            examples = core[suite_name].get(split, [])
            result[suite_name][split] = sorted({str(example["graph_id"]) for example in examples})
    return result


def _split_counts(core: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        suite_name: {
            split: len(core[suite_name].get(split, []))
            for split in ("train", "validation", "test", "seen_test")
            if split in core[suite_name]
        }
        for suite_name in ("s1", "s2", "s3", "s4")
    }


def _expected_split_counts() -> dict[str, dict[str, int]]:
    s2_graph_counts, s2_excitation_counts = _s2_protocol_counts()
    s2_counts = {
        split: graph_count * excitation_count
        for split, graph_count, excitation_count in zip(
            ("train", "validation", "test"),
            s2_graph_counts,
            s2_excitation_counts,
            strict=True,
        )
    }
    return {
        "s1": {"train": 252, "validation": 27, "test": 27, "seen_test": 84},
        "s2": s2_counts,
        "s3": {"train": 600, "validation": 150, "test": 250},
        "s4": {"train": 324, "validation": 36, "test": 144},
    }


def _core_request(seed: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
    }


def _core_cache_paths(data_root: Path | str, request: dict[str, Any]) -> tuple[Path, Path]:
    cache_key = hashlib.sha256(_canonical_json(request)).hexdigest()[:16]
    cache_dir = Path(data_root).expanduser().resolve() / "conductance_gat" / f"core-{cache_key}"
    return cache_dir / "core.pt", cache_dir / "manifest.json"


def _validate_example(example: dict[str, Any]) -> None:
    required = {
        "edge_index",
        "edge_features",
        "node_state",
        "true_conductance",
        "true_gradient",
        "true_flux",
        "true_node_message",
        "true_next_state",
        "observed_flux",
        "observed_node_message",
    }
    if not required.issubset(example):
        raise CacheCorruptError("conductance example is missing required tensors")
    tensors = {name: example[name] for name in required}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise CacheCorruptError("conductance example contains a non-tensor payload")
    edge_index = tensors["edge_index"]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise CacheCorruptError("conductance edge_index must have shape [2, m] and dtype long")
    edge_count = int(edge_index.shape[1])
    node_state = tensors["node_state"]
    if node_state.ndim != 2:
        raise CacheCorruptError("conductance node_state must have shape [n, channels]")
    node_count, channels = map(int, node_state.shape)
    expected_shapes = {
        "edge_features": (edge_count, tensors["edge_features"].shape[-1]),
        "true_conductance": (edge_count,),
        "true_gradient": (edge_count, channels),
        "true_flux": (edge_count, channels),
        "true_node_message": (node_count, channels),
        "true_next_state": (node_count, channels),
        "observed_flux": (edge_count, channels),
        "observed_node_message": (node_count, channels),
    }
    for name, expected in expected_shapes.items():
        if tuple(tensors[name].shape) != tuple(expected):
            raise CacheCorruptError(f"conductance tensor {name!r} has an invalid shape")
    if (
        edge_count < 1
        or node_count < 2
        or int(edge_index.min()) < 0
        or int(edge_index.max()) >= node_count
    ):
        raise CacheCorruptError("conductance graph topology is invalid")
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise CacheCorruptError("conductance cache contains a non-finite tensor")
    if not torch.all(tensors["true_conductance"] > 0):
        raise CacheCorruptError("conductance cache contains a non-positive conductance")


def _validate_core_content(core: Any, request: dict[str, Any], manifest: dict[str, Any]) -> None:
    if not isinstance(core, dict):
        raise CacheCorruptError("conductance core artifact must be a mapping")
    for key, value in request.items():
        if core.get(key) != value:
            raise CacheWrongRequestError(f"conductance core field {key!r} does not match request")
    for suite_name in ("s1", "s2", "s3", "s4"):
        suite = core.get(suite_name)
        if not isinstance(suite, dict):
            raise CacheCorruptError(f"conductance cache is missing suite {suite_name!r}")
        for split in ("train", "validation", "test"):
            examples = suite.get(split)
            if not isinstance(examples, list) or not examples:
                raise CacheCorruptError(f"conductance {suite_name}.{split} is empty or invalid")
            for example in examples:
                if not isinstance(example, dict):
                    raise CacheCorruptError("conductance split contains a non-mapping example")
                _validate_example(example)
    graph_ids = _graph_ids(core)
    if manifest.get("graph_ids") != graph_ids:
        raise CacheCorruptError("conductance graph-ID manifest does not match the artifact")
    split_counts = _split_counts(core)
    if manifest.get("split_counts") != split_counts:
        raise CacheCorruptError("conductance split-count manifest does not match the artifact")
    if split_counts != _expected_split_counts():
        raise CacheCorruptError("conductance split cardinalities do not match the paper protocol")
    for suite_splits in graph_ids.values():
        named_sets = [set(values) for name, values in suite_splits.items() if name != "seen_test"]
        for index, left in enumerate(named_sets):
            if any(left.intersection(right) for right in named_sets[index + 1 :]):
                raise CacheCorruptError("conductance graph IDs cross physical graph splits")


def validate_core_cache(
    data_root: Path | str, *, seed: int
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Read and fully validate one requested generated core cache without writing."""

    request = _core_request(seed)
    artifact_path, manifest_path = _core_cache_paths(data_root, request)
    present = (artifact_path.is_file(), manifest_path.is_file())
    if not any(present):
        raise FileNotFoundError(
            f"conductance core cache is missing for seed={seed}: {artifact_path}"
        )
    if not all(present):
        raise CacheIncompleteError(
            f"conductance core.pt and manifest.json must both exist: {artifact_path.parent}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"invalid conductance cache manifest: {manifest_path}") from error
    if manifest.get("request") != request:
        raise CacheWrongRequestError(f"cache manifest request mismatch: {manifest_path}")
    if _file_sha256(artifact_path) != manifest.get("artifact_sha256"):
        raise CacheCorruptError(f"cache artifact checksum mismatch: {artifact_path}")
    try:
        try:
            core = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            core = torch.load(artifact_path, map_location="cpu")
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise CacheCorruptError(
            f"failed to deserialize conductance cache: {artifact_path}"
        ) from error
    if _content_fingerprint(core) != manifest.get("content_sha256"):
        raise CacheCorruptError(f"cache tensor-content checksum mismatch: {artifact_path}")
    _validate_core_content(core, request, manifest)
    return core, manifest_path, manifest


def prepare_core_cache(
    data_root: Path | str, *, seed: int, force: bool = False
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    request = _core_request(seed)
    artifact_path, manifest_path = _core_cache_paths(data_root, request)
    cache_key = artifact_path.parent.name.removeprefix("core-")
    if (artifact_path.exists() or manifest_path.exists()) and not force:
        return validate_core_cache(data_root, seed=seed)

    core = generate_core(seed)
    expected_content_sha256 = _content_fingerprint(core)

    def validate_artifact(temporary: Path) -> None:
        try:
            loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            loaded = torch.load(temporary, map_location="cpu")
        if _content_fingerprint(loaded) != expected_content_sha256:
            raise CacheCorruptError("new conductance artifact failed temporary validation")

    def write_artifact(temporary: Path) -> None:
        # Saving through a stream prevents the unique temporary basename from
        # entering PyTorch's ZIP metadata, preserving byte determinism.
        with temporary.open("wb") as stream:
            torch.save(core, stream)

    atomic_publish(artifact_path, write_artifact, validator=validate_artifact)
    manifest = {
        "request": request,
        "cache_key": cache_key,
        "artifact": artifact_path.name,
        "artifact_sha256": _file_sha256(artifact_path),
        "content_sha256": expected_content_sha256,
        "graph_ids": _graph_ids(core),
        "split_counts": _split_counts(core),
    }
    atomic_write_json(
        manifest_path,
        manifest,
        validator=lambda temporary: json.loads(temporary.read_text(encoding="utf-8")),
    )
    return validate_core_cache(data_root, seed=seed)


__all__ = [
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "generate_core",
    "generate_s1",
    "generate_s2",
    "generate_s3",
    "generate_s4",
    "make_example",
    "make_graph",
    "nonlinear_conductance",
    "prepare_core_cache",
    "static_conductance",
    "validate_core_cache",
]
