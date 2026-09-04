"""Official molecular splits with ordered sparse DFS cycles shared by SE and PE.

Only the official split adapter and source fingerprint are shared with v1.
Neither v1 graph preparation nor cycle-set statistics are used here.  The
production backend caches signed cycles, support counts and actual circular edge positions.
No dense basis, QR/SVD or projector is created in preparation or minibatches.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch import Tensor

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheWrongRequestError,
    atomic_publish,
    atomic_write_json,
)
from research.cycle_pe.benchmark_data import graph_fingerprint, load_official_splits
from research.cycle_pe.v2.basis import (
    BASIS_BACKENDS,
    DEFAULT_BASIS_BACKEND,
    build_cycle_coordinates,
    cycle_position_factors,
    validate_cycle_basis,
    validate_cycle_positions,
)

DATASETS = ("zinc12k", "peptides_struct")
SPLITS = ("train", "validation", "test")
CACHE_VERSION = "ordered-dfs-cycle-coordinates-v2-1"
CACHE_NAMESPACE = "cycle_pe_v2_ordered_dfs_benchmark"
SCHEMAS = {
    "zinc12k": {"atoms": (28,), "bonds": (4,), "targets": 1},
    "peptides_struct": {
        "atoms": (119, 4, 12, 12, 10, 6, 6, 2, 2),
        "bonds": (5, 6, 2),
        "targets": 11,
    },
}
SOURCES = {
    "zinc12k": "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html",
    "peptides_struct": "https://github.com/vijaydwivedi75/lrgb",
}


@dataclass
class Graph:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Tensor
    cycle_basis: Tensor
    cycle_lengths: Tensor
    edge_cycle_counts: Tensor
    edge_cycle_features: Tensor
    cycle_position_indices: Tensor
    cycle_position_values: Tensor


@dataclass
class Batch:
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Tensor
    batch: Tensor
    ptr: Tensor
    cycle_membership: Tensor
    cycle_position_values: Tensor
    cycle_lengths: Tensor
    edge_cycle_counts: Tensor
    edge_cycle_features: Tensor
    cycle_basis_shapes: tuple[tuple[int, int], ...]
    cycle_graph_index: Tensor
    edge_graph_index: Tensor
    edge_ptr: Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            **{
                field.name: current.to(device, non_blocking=True)
                if isinstance(current, Tensor)
                else current
                for field in fields(self)
                for current in (getattr(self, field.name),)
            }
        )

    def pin_memory(self) -> Batch:
        return Batch(
            **{
                field.name: _pin_tensor(current) if isinstance(current, Tensor) else current
                for field in fields(self)
                for current in (getattr(self, field.name),)
            }
        )


def _pin_tensor(value: Tensor) -> Tensor:
    """Pin sparse storage explicitly; sparse Tensor.pin_memory is unsupported."""
    if value.layout == torch.sparse_coo:
        return torch.sparse_coo_tensor(
            value.indices().pin_memory(),
            value.values().pin_memory(),
            value.shape,
            is_coalesced=True,
            check_invariants=False,
        )
    return value.pin_memory()


def collate(graphs: list[Graph]) -> Batch:
    """Pack all graphs into one sparse block-diagonal edge/cycle membership.

    Only O(1)-per-field schema checks are repeated here.  Signed nullness,
    independence, finite values and cached support counts were certified at
    preparation/load; no graph's fixed cycle algebra is repeated per batch.
    """
    if not graphs:
        raise ValueError("cannot collate an empty graph list")
    for graph in graphs:
        validate_graph(graph, check_basis=False, check_values=False)
    widths = {(g.x.shape[1], g.edge_attr.shape[1], g.y.numel()) for g in graphs}
    if len(widths) != 1:
        raise ValueError("cannot collate graphs with different molecular schemas")
    counts = [len(graph.x) for graph in graphs]
    ptr = torch.tensor([0, *np.cumsum(counts).tolist()], dtype=torch.long)
    edge_ptr = torch.tensor(
        [0, *np.cumsum([graph.edge_index.shape[1] for graph in graphs]).tolist()],
        dtype=torch.long,
    )
    cycle_counts = [graph.cycle_basis.shape[1] for graph in graphs]
    edge_counts = [graph.edge_index.shape[1] for graph in graphs]
    sparse_indices = []
    edge_offset = cycle_offset = 0
    for graph in graphs:
        sparse_indices.append(
            graph.cycle_basis.indices() + torch.tensor([[edge_offset], [cycle_offset]])
        )
        edge_offset += graph.cycle_basis.shape[0]
        cycle_offset += graph.cycle_basis.shape[1]
    indices = torch.cat(sparse_indices, dim=1)
    membership = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.shape[1], dtype=torch.float32),
        (edge_offset, cycle_offset),
        is_coalesced=True,
        check_invariants=False,
    )
    return Batch(
        x=torch.cat([graph.x for graph in graphs]),
        edge_index=torch.cat(
            [graph.edge_index + ptr[index] for index, graph in enumerate(graphs)], dim=1
        ),
        edge_attr=torch.cat([graph.edge_attr for graph in graphs]),
        y=torch.stack([graph.y for graph in graphs]),
        batch=torch.repeat_interleave(torch.arange(len(graphs)), torch.tensor(counts)),
        ptr=ptr,
        cycle_membership=membership,
        cycle_position_values=torch.cat([graph.cycle_position_values for graph in graphs], dim=1),
        cycle_lengths=torch.cat([graph.cycle_lengths for graph in graphs]),
        edge_cycle_counts=torch.cat([graph.edge_cycle_counts for graph in graphs]),
        edge_cycle_features=torch.cat([graph.edge_cycle_features for graph in graphs]),
        cycle_basis_shapes=tuple(tuple(graph.cycle_basis.shape) for graph in graphs),
        cycle_graph_index=torch.repeat_interleave(
            torch.arange(len(graphs)), torch.tensor(cycle_counts)
        ),
        edge_graph_index=torch.repeat_interleave(
            torch.arange(len(graphs)), torch.tensor(edge_counts)
        ),
        edge_ptr=edge_ptr,
    )


def _integer_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f"{name} must be an integer tensor")
    value = value.detach().cpu()
    if not torch.isfinite(value).all() or (
        value.is_floating_point() and not torch.equal(value, value.round())
    ):
        raise ValueError(f"{name} must contain finite integer values")
    return value.long().contiguous()


def _canonical_inputs(data: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raw_nodes = data.num_nodes
    if isinstance(raw_nodes, bool) or not isinstance(raw_nodes, (int, np.integer)):
        raise ValueError("num_nodes must be a positive integer")
    num_nodes = int(raw_nodes)
    if num_nodes < 1:
        raise ValueError("official graph must contain at least one node")
    x = _integer_tensor(data.x, "atom features")
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2 or x.shape[0] != num_nodes or x.shape[1] < 1:
        raise ValueError("invalid official atom-feature shape")
    edge_index = _integer_tensor(data.edge_index, "edge_index")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if (edge_index < 0).any() or (edge_index >= num_nodes).any():
        raise ValueError("edge endpoint out of range")
    edge_attr = _integer_tensor(data.edge_attr, "bond features")
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(1)
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("invalid official bond-feature shape")
    if edge_attr.shape[1] < 1:
        raise ValueError("official bonds require categorical features")
    if not isinstance(data.y, Tensor) or data.y.is_complex():
        raise ValueError("official target must be a real tensor")
    y = data.y.detach().cpu().float().reshape(-1)
    if not len(y) or not torch.isfinite(y).all():
        raise ValueError("official targets must be finite and nonempty")
    pairs = list(map(tuple, edge_index.T.tolist()))
    if len(set(pairs)) != len(pairs) or any(u == v for u, v in pairs):
        raise ValueError("molecular benchmark requires simple loop-free edges")
    attributes = {edge: edge_attr[index] for index, edge in enumerate(pairs)}
    for u, v in pairs:
        if (v, u) not in attributes or not torch.equal(attributes[u, v], attributes[v, u]):
            raise ValueError("molecular bonds must have agreeing directed copies")
    canonical = sorted((u, v) for u, v in pairs if u < v)
    canonical_index = torch.tensor(canonical, dtype=torch.long).reshape(-1, 2).T.contiguous()
    canonical_attr = (
        torch.stack([attributes[pair] for pair in canonical])
        if canonical
        else edge_attr.new_empty((0, edge_attr.shape[1]))
    )
    return x, canonical_index, canonical_attr, y


def validate_graph(
    graph: Graph,
    *,
    dataset: str | None = None,
    check_basis: bool = True,
    check_values: bool = True,
) -> None:
    """Validate prepared/cache schema and, on preparation/load, basis identities."""
    for field in fields(Graph):
        value = getattr(graph, field.name)
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise ValueError("prepared graph fields must be CPU tensors")
        if field.name == "cycle_basis":
            if value.layout != torch.sparse_coo or not value.is_coalesced():
                raise ValueError(
                    "invalid prepared cycle-basis schema: sparse coalesced COO required"
                )
        elif value.layout != torch.strided:
            raise ValueError(f"prepared {field.name} must use strided storage")
        stored = (
            value.values() if value.layout == torch.sparse_coo and value.is_coalesced() else value
        )
        if check_values and not torch.isfinite(stored).all():
            raise ValueError(f"nonfinite prepared graph field: {field.name}")
    if graph.x.dtype != torch.long or graph.x.ndim != 2 or min(graph.x.shape) < 1:
        raise ValueError("invalid prepared atom-feature schema")
    if (
        graph.edge_index.dtype != torch.long
        or graph.edge_index.ndim != 2
        or graph.edge_index.shape[0] != 2
    ):
        raise ValueError("invalid prepared edge_index schema")
    edge_count = graph.edge_index.shape[1]
    if (
        graph.edge_attr.dtype != torch.long
        or graph.edge_attr.ndim != 2
        or graph.edge_attr.shape[0] != edge_count
        or graph.edge_attr.shape[1] < 1
    ):
        raise ValueError("invalid prepared bond-feature schema")
    if graph.y.dtype != torch.float32 or graph.y.ndim != 1 or not graph.y.numel():
        raise ValueError("invalid prepared target schema")
    if (
        graph.cycle_basis.dtype != torch.float32
        or graph.cycle_basis.layout != torch.sparse_coo
        or not graph.cycle_basis.is_coalesced()
        or graph.cycle_basis.ndim != 2
        or graph.cycle_basis.shape[0] != edge_count
    ):
        raise ValueError("invalid prepared cycle-basis schema")
    if (
        graph.cycle_lengths.dtype != torch.float32
        or graph.cycle_lengths.shape != (graph.cycle_basis.shape[1],)
        or graph.edge_cycle_counts.dtype != torch.float32
        or graph.edge_cycle_counts.shape != (edge_count,)
        or graph.edge_cycle_features.dtype != torch.float32
        or graph.edge_cycle_features.shape != (edge_count, 2)
    ):
        raise ValueError("invalid cycle-basis support-count schema")
    if (
        graph.cycle_position_indices.dtype != torch.long
        or graph.cycle_position_indices.shape != (graph.cycle_basis._nnz(),)
        or graph.cycle_position_values.dtype != torch.float32
        or graph.cycle_position_values.shape != (2, graph.cycle_basis._nnz())
    ):
        raise ValueError(
            "invalid cycle-position schema: integer order and aligned cos/sin required"
        )
    if check_values and ((graph.x < 0).any() or (graph.edge_attr < 0).any()):
        raise ValueError("categorical features must be nonnegative")
    if dataset is not None:
        if dataset not in DATASETS:
            raise ValueError(f"unknown cycle PE v2 dataset: {dataset}")
        schema = SCHEMAS[dataset]
        if graph.y.numel() != schema["targets"]:
            raise ValueError(f"{dataset}: unexpected target width")
        for values, name in ((graph.x, "atoms"), (graph.edge_attr, "bonds")):
            cardinalities = schema[name]
            if values.shape[1] != len(cardinalities):
                raise ValueError(f"{dataset}: unexpected {name} field count")
            if check_values and any(
                (values[:, i] >= size).any() for i, size in enumerate(cardinalities)
            ):
                raise ValueError(f"{dataset}: categorical {name} index out of range")
    if check_basis:
        indices, values = graph.cycle_basis.indices(), graph.cycle_basis.values()
        if not torch.equal(values.abs(), torch.ones_like(values)):
            raise ValueError("sparse DFS cycle_basis must contain signed unit entries")
        scipy_basis = sparse.coo_matrix(
            (values.numpy(), (indices[0].numpy(), indices[1].numpy())),
            shape=tuple(graph.cycle_basis.shape),
        ).tocsr()
        validate_cycle_basis(len(graph.x), graph.edge_index.numpy(), scipy_basis)
        positions = graph.cycle_position_indices.numpy()
        validate_cycle_positions(len(graph.x), graph.edge_index.numpy(), scipy_basis, positions)
        expected_positions = torch.from_numpy(cycle_position_factors(scipy_basis, positions))
        if not torch.allclose(
            graph.cycle_position_values, expected_positions, atol=2e-6, rtol=2e-6
        ):
            raise ValueError("cached cycle positions do not match actual ordered cycle cos/sin")
        if not torch.allclose(
            graph.cycle_position_values.square().sum(dim=0),
            torch.ones(graph.cycle_basis._nnz()),
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError("cached cycle position cos/sin must lie on the unit circle")
        lengths = torch.bincount(indices[1], minlength=graph.cycle_basis.shape[1]).float()
        counts = torch.bincount(indices[0], minlength=edge_count).float()
        if not torch.equal(lengths, graph.cycle_lengths) or not torch.equal(
            counts, graph.edge_cycle_counts
        ):
            raise ValueError("cached cycle lengths/counts disagree with complete DFS basis")
        if (lengths < 3).any():
            raise ValueError("simple-graph fundamental cycles must contain at least three edges")
        expected_features = _cycle_support_features(scipy_basis, lengths.numpy(), counts.numpy())
        if not torch.equal(expected_features, graph.edge_cycle_features):
            raise ValueError(
                "cached edge cycle structural features disagree with complete DFS basis"
            )


def _cycle_support_features(
    basis: sparse.spmatrix,
    lengths: np.ndarray,
    counts: np.ndarray,
) -> Tensor:
    """Cache graph-only mean log length and inverse length once, including trees."""
    descriptors = np.stack((np.log1p(lengths), np.reciprocal(lengths)), axis=1)
    values = (abs(basis) @ descriptors) / np.maximum(counts[:, None], 1.0)
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def prepare_graph(
    data: Any,
    *,
    dataset: str | None = None,
    basis_backend: str = DEFAULT_BASIS_BACKEND,
) -> Graph:
    """Preserve official data and attach the selected cycle-space representation."""
    if basis_backend not in BASIS_BACKENDS:
        raise ValueError(
            f"basis_backend must be one of {BASIS_BACKENDS}; old thin_q caches are retired"
        )
    x, edge_index, edge_attr, y = _canonical_inputs(data)
    sparse_basis, positions = build_cycle_coordinates(
        len(x), edge_index.numpy(), backend=basis_backend
    )
    position_values = cycle_position_factors(sparse_basis, positions)
    basis = sparse_basis.tocoo()
    indices = torch.from_numpy(np.vstack((basis.row, basis.col)).astype(np.int64))
    values = torch.from_numpy(basis.data)
    lengths = torch.bincount(indices[1], minlength=basis.shape[1]).float()
    counts = torch.bincount(indices[0], minlength=basis.shape[0]).float()
    graph = Graph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        cycle_basis=torch.sparse_coo_tensor(
            indices, values, basis.shape, is_coalesced=True, check_invariants=True
        ),
        cycle_lengths=lengths,
        edge_cycle_counts=counts,
        edge_cycle_features=_cycle_support_features(basis, lengths.numpy(), counts.numpy()),
        cycle_position_indices=torch.from_numpy(positions),
        cycle_position_values=torch.from_numpy(position_values),
    )
    validate_graph(graph, dataset=dataset)
    return graph


def preparation_signature(
    dataset: str, *, basis_backend: str = DEFAULT_BASIS_BACKEND
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown cycle PE v2 dataset: {dataset}")
    if basis_backend not in BASIS_BACKENDS:
        raise ValueError(f"basis_backend must be one of {BASIS_BACKENDS}")
    directory = Path(__file__).resolve().parent
    return {
        "version": CACHE_VERSION,
        "dataset": dataset,
        "basis_backend": basis_backend,
        "representation": "ordered_sparse_dfs_cycle_coordinates",
        "incidence": "B[m,n], canonical sorted u<v edges, tail -1 and head +1",
        "basis": (
            "iterative DFS forest + signed parent path per non-tree edge; all beta=m-n+c columns"
        ),
        "storage": (
            "float32 signed sparse COO Z [num_edges, cycle_rank], cycle lengths and edge "
            "membership counts; cached integer circular position and cos/sin values per "
            "nonzero; unsigned block-diagonal A and aligned cos/sin for minibatches"
        ),
        "construction_complexity": (
            "DFS forest O(V+E); explicit cycle support O(V+E+nnz(Z)); sparse storage O(nnz(Z)); "
            "no dense incidence/basis, Gram matrix, QR, SVD or projector"
        ),
        "basis_dependence": (
            "cycle order/sign invariant, but choice of DFS forest is observable; "
            "not arbitrary ZR invariant"
        ),
        "position_convention": (
            "each chord tail->head followed by ordered tree path head->tail; "
            "t=0..L-1, theta=2*pi*t/L on actual cycle edges; no CSR-row-order positions"
        ),
        "position_symmetry": (
            "PE's 1+cos(theta_e-theta_f) kernel is invariant to cycle origin/reversal; "
            "both SE and PE depend on the selected DFS forest"
        ),
        "numpy_version": np.__version__,
        "implementation_sha256": {
            "v2/basis.py": hashlib.sha256((directory / "basis.py").read_bytes()).hexdigest(),
            "v2/data.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "official_adapter": hashlib.sha256(
                (directory.parent / "benchmark_data.py").read_bytes()
            ).hexdigest(),
        },
    }


def _validate_cached_graph_task(payload: tuple[Any, Any, str]) -> Graph:
    row, source, dataset = payload
    names = {field.name for field in fields(Graph)}
    if not isinstance(row, dict) or set(row) != names:
        raise ValueError("cached graph field schema mismatch; legacy dense/Q cache is unsupported")
    graph = Graph(**row)
    validate_graph(graph, dataset=dataset)
    expected = _canonical_inputs(source)
    for name, source_value in zip(("x", "edge_index", "edge_attr", "y"), expected, strict=True):
        if not torch.equal(getattr(graph, name), source_value):
            raise ValueError(f"cached {name} disagrees with official graph content/order")
    return graph


def _apply_graph_chunk(function: Callable[[Any], Graph], chunk: list[Any]) -> list[Graph]:
    """Top-level spawn-picklable work item; no graph is omitted or truncated."""
    return [function(payload) for payload in chunk]


def _ordered_parallel_graphs(
    executor: Any,
    function: Callable[[Any], Graph],
    tasks: Iterable[Any],
    *,
    workers: int,
    chunksize: int,
) -> Iterator[Graph]:
    """Bound queued serialization while yielding every result in official order.

    Python 3.11 Executor.map eagerly submits its entire input.  Keep at most two
    chunks per worker in flight instead.  This is a buffer bound only: consumed
    chunks are replenished until the complete source iterator is exhausted.
    """
    if workers < 1 or chunksize < 1:
        raise ValueError("parallel graph buffering requires positive workers and chunksize")
    source = iter(tasks)
    pending = deque()

    def submit_next() -> bool:
        chunk = list(islice(source, chunksize))
        if not chunk:
            return False
        pending.append(executor.submit(_apply_graph_chunk, function, chunk))
        return True

    for _ in range(2 * workers):
        if not submit_next():
            break
    while pending:
        completed = pending.popleft().result()
        submit_next()
        yield from completed


def _validate_cached_graphs(
    rows: Any,
    official: Any,
    dataset: str,
    *,
    workers: int,
) -> list[Graph]:
    if not isinstance(rows, list) or len(rows) != len(official):
        raise ValueError("cached graph count/schema mismatch")
    tasks = ((row, source, dataset) for row, source in zip(rows, official, strict=True))
    if workers <= 1 or len(rows) <= 1:
        return list(map(_validate_cached_graph_task, tasks))
    with ProcessPoolExecutor(
        max_workers=min(workers, len(rows)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_preparation_worker,
    ) as executor:
        return list(
            _ordered_parallel_graphs(
                executor,
                _validate_cached_graph_task,
                tasks,
                workers=workers,
                chunksize=max(1, min(16, len(rows) // (4 * workers))),
            )
        )


def _initialize_preparation_worker() -> None:
    # Graph construction is Python/NumPy work; avoid one BLAS/PyTorch thread
    # pool per process oversubscribing the configured worker allocation.
    torch.set_num_threads(1)


def _prepare_task(payload: tuple[Any, str, str]) -> Graph:
    source, dataset, backend = payload
    return prepare_graph(source, dataset=dataset, basis_backend=backend)


def _prepare_split(
    official: Any,
    *,
    dataset: str,
    split: str,
    basis_backend: str,
    workers: int,
) -> list[Graph]:
    """Prepare every graph in stable official order, using allocated CPU workers."""
    tasks = ((source, dataset, basis_backend) for source in official)

    def collect(prepared: Any) -> list[Graph]:
        graphs = []
        for index, graph in enumerate(prepared, start=1):
            graphs.append(graph)
            if index % 1000 == 0:
                print(
                    f"{dataset}/{split}: sparse DFS cycle spaces {index}/{len(official)}",
                    flush=True,
                )
        return graphs

    if workers <= 1 or len(official) <= 1:
        return collect(map(_prepare_task, tasks))
    # spawn is safe even when the parent already initialized a CUDA context.
    with ProcessPoolExecutor(
        max_workers=min(workers, len(official)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_preparation_worker,
    ) as executor:
        return collect(
            _ordered_parallel_graphs(
                executor,
                _prepare_task,
                tasks,
                workers=workers,
                chunksize=max(1, min(16, len(official) // (4 * workers))),
            )
        )


def load_benchmark(
    data_root: Path,
    dataset: str,
    *,
    allow_download: bool,
    splits: tuple[str, ...] = SPLITS,
    basis_backend: str = DEFAULT_BASIS_BACKEND,
    workers: int = 4,
) -> tuple[dict[str, list[Graph]], dict[str, Any]]:
    """Load fixed official splits, validating immutable basis caches fail-closed."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError(
            "workers must be a nonnegative integer; zero is explicit serial preparation"
        )
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(split not in SPLITS for split in splits)
    ):
        raise ValueError("splits must be a nonempty unique subset of official splits")
    signature = preparation_signature(dataset, basis_backend=basis_backend)
    official = load_official_splits(
        data_root,
        dataset,
        allow_download=allow_download,
        splits=splits,
    )
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = data_root / CACHE_NAMESPACE / dataset / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Graph]] = {}
    split_hashes = {}
    for split in splits:
        digest = hashlib.sha256()
        for data in official[split]:
            graph_fingerprint(data, digest)
        source_hash = split_hashes[split] = digest.hexdigest()
        cache, meta = cache_dir / f"{split}.pt", cache_dir / f"{split}.json"
        if cache.exists() != meta.exists():
            raise CacheIncompleteError(
                f"Incomplete cycle PE v2 cache at {cache}; no silent rebuild"
            )
        if cache.exists():
            try:
                metadata = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CacheCorruptError(f"Unreadable cycle PE v2 metadata: {meta}") from exc
            if not isinstance(metadata, dict):
                raise CacheCorruptError(f"Invalid cycle PE v2 metadata schema: {meta}")
            if (
                metadata.get("signature") != signature
                or metadata.get("source_sha256") != source_hash
                or metadata.get("split") != split
            ):
                raise CacheWrongRequestError(f"Mismatched cycle PE v2 cache: {cache}; no rebuild")
            if metadata.get("cache_sha256") != hashlib.sha256(cache.read_bytes()).hexdigest():
                raise CacheCorruptError(f"Corrupt cycle PE v2 cache payload: {cache}; no rebuild")
            try:
                rows = torch.load(cache, map_location="cpu", weights_only=True)
                graphs = _validate_cached_graphs(rows, official[split], dataset, workers=workers)
            except Exception as exc:
                raise CacheCorruptError(
                    f"Invalid cycle PE v2 cache content: {cache}: {exc}"
                ) from exc
        else:
            graphs = _prepare_split(
                official[split],
                dataset=dataset,
                split=split,
                basis_backend=basis_backend,
                workers=workers,
            )
            rows = [
                {field.name: getattr(graph, field.name) for field in fields(Graph)}
                for graph in graphs
            ]
            atomic_publish(cache, lambda path, payload=rows: torch.save(payload, path))
            atomic_write_json(
                meta,
                {
                    "signature": signature,
                    "split": split,
                    "source_sha256": source_hash,
                    "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                },
            )
        result[split] = graphs
    protocol = {
        "comparison": "ours_only_on_official_benchmark_splits",
        "source_url": SOURCES[dataset],
        "official_splits": True,
        "loaded_splits": list(splits),
        "split_sizes": {split: len(result[split]) for split in splits},
        "split_content_sha256": split_hashes,
        "target_width": SCHEMAS[dataset]["targets"],
        "target_scaling": "official supplied labels, unchanged; no fitted target scaling",
        "input_features": "ZINC categorical atoms/bonds"
        if dataset == "zinc12k"
        else "OGB 9 atom / 3 bond categorical fields",
        "preparation": signature,
        "cache_directory": str(cache_dir),
        "basis_backend": basis_backend,
        "preparation_workers": workers,
        "basis_storage": (
            "all beta=m-n+c sparse signed DFS fundamental columns; no padding or truncation"
        ),
        "basis_coordinates": (
            "model uses cycle supports A=abs(Z) with cached cycle lengths/edge membership "
            "counts for SE, and actual circular cos/sin positions for PE; "
            "cycle order/sign invariant, selected DFS forest can affect the PE"
        ),
        "basis_runtime": (
            "SE and positional-kernel PE use sparse block-diagonal aggregation O(nnz(Z)*d), "
            "plus feature MLPs; no QR/SVD/projector or per-graph model forward"
        ),
    }
    return result, protocol


__all__ = [
    "Batch",
    "BASIS_BACKENDS",
    "CACHE_NAMESPACE",
    "CACHE_VERSION",
    "DATASETS",
    "DEFAULT_BASIS_BACKEND",
    "Graph",
    "collate",
    "load_benchmark",
    "preparation_signature",
    "prepare_graph",
    "validate_graph",
]
