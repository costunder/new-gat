#!/usr/bin/env python3
"""Fail-fast CUDA and paper-dependency validation for a Linux experiment host."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as nnf


class PreflightError(RuntimeError):
    """A server cannot safely run the requested experiment profile."""


PAPER_IMPORTS = {
    "networkx": "networkx",
    "numpy": "numpy",
    "ogb": "ogb",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "torch-geometric": "torch_geometric",
    "PyYAML": "yaml",
}

PROFILE_NAMES = (
    "conductance",
    "cycle-projector",
    "tree-chart",
    "brec",
    "public-pyg",
)
CYCLE_VARIANTS = ("no_pe", "raw", "set", "projector")
BREC_PROTOCOLS = ("official", "custom")


@dataclass(frozen=True)
class ProfileConfig:
    """Synthetic envelope used to exercise one production training path."""

    batch_size: int = 32
    brec_batch_size: int = 16
    nodes_per_graph: int = 64
    edges_per_graph: int = 128
    cycle_rank: int = 64
    amp: bool = True
    cycle_variants: tuple[str, ...] = ("projector",)
    brec_protocol: str = "official"
    brec_amp: bool = False

    def validate(self) -> None:
        values = {
            "batch_size": self.batch_size,
            "brec_batch_size": self.brec_batch_size,
            "nodes_per_graph": self.nodes_per_graph,
            "edges_per_graph": self.edges_per_graph,
            "cycle_rank": self.cycle_rank,
        }
        invalid = [name for name, value in values.items() if value < 1]
        if invalid:
            raise PreflightError("profile dimensions must be positive: " + ", ".join(invalid))
        if self.nodes_per_graph < 2:
            raise PreflightError("--nodes-per-graph must be at least 2")
        if self.cycle_rank > self.edges_per_graph:
            raise PreflightError("--cycle-rank cannot exceed --edges-per-graph")
        if not self.cycle_variants or len(set(self.cycle_variants)) != len(self.cycle_variants):
            raise PreflightError("--cycle-variants must be non-empty and unique")
        unknown_variants = sorted(set(self.cycle_variants) - set(CYCLE_VARIANTS))
        if unknown_variants:
            raise PreflightError(f"unknown cycle variants: {unknown_variants}")
        if self.brec_protocol not in BREC_PROTOCOLS:
            raise PreflightError(f"--brec-protocol must be one of {BREC_PROTOCOLS}")
        if self.brec_protocol == "official":
            if self.brec_batch_size != 16:
                raise PreflightError("official BREC profile requires --brec-batch-size 16")
            if self.brec_amp:
                raise PreflightError("official BREC profile requires --no-brec-amp")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"torch": torch.__version__}
    for distribution in PAPER_IMPORTS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _missing_paper_dependencies() -> list[str]:
    return sorted(
        distribution
        for distribution, module in PAPER_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    )


def _paper_dependency_import_errors() -> dict[str, str]:
    """Import paper packages so an ABI-broken environment fails before training."""

    errors: dict[str, str] = {}
    for distribution, module in PAPER_IMPORTS.items():
        if importlib.util.find_spec(module) is None:
            errors[distribution] = "module not installed"
            continue
        try:
            importlib.import_module(module)
        except Exception as error:  # dependency imports can raise ABI/runtime errors
            errors[distribution] = f"{type(error).__name__}: {error}"
    return errors


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "rows": [], "error": "nvidia-smi not found"}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "available": completed.returncode == 0,
        "rows": rows,
        "error": completed.stderr.strip() or None,
    }


def _resolve_device(requested: str, *, allow_cpu: bool) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cpu":
        if not allow_cpu:
            raise PreflightError(
                "paper execution requires CUDA; pass --allow-cpu only for tiny local tests"
            )
        return device
    if device.type != "cuda":
        raise PreflightError(f"unsupported accelerator request: {requested!r}")
    if not torch.cuda.is_available():
        raise PreflightError(
            "torch.cuda.is_available() is false; install a CUDA PyTorch wheel and expose the GPU"
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        visible_count = torch.cuda.device_count()
        raise PreflightError(
            f"CUDA device index {index} is invalid; visible device count is {visible_count}"
        )
    return torch.device("cuda", index)


def _effective_profile_config(config: ProfileConfig, device: torch.device) -> ProfileConfig:
    """Keep the CPU path a code smoke while preserving exact requested CUDA sizes."""

    if device.type == "cuda":
        return config
    edges = min(config.edges_per_graph, 24)
    return ProfileConfig(
        batch_size=min(config.batch_size, 2),
        brec_batch_size=(
            4 if config.brec_protocol == "official" else min(config.brec_batch_size, 4)
        ),
        nodes_per_graph=min(config.nodes_per_graph, 12),
        edges_per_graph=edges,
        cycle_rank=min(config.cycle_rank, edges, 6),
        amp=config.amp,
        cycle_variants=config.cycle_variants,
        brec_protocol=config.brec_protocol,
        brec_amp=config.brec_amp,
    )


def _base_edges(num_nodes: int, num_edges: int) -> Tensor:
    columns = torch.arange(num_edges, dtype=torch.long)
    tail = columns.remainder(num_nodes)
    hop = 1 + torch.div(columns, num_nodes, rounding_mode="floor").remainder(num_nodes - 1)
    head = (tail + hop).remainder(num_nodes)
    return torch.stack((tail, head))


def _packed_edges(batch_size: int, num_nodes: int, num_edges: int) -> Tensor:
    base = _base_edges(num_nodes, num_edges)
    offsets = torch.arange(batch_size, dtype=torch.long) * num_nodes
    return (base.unsqueeze(0) + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count the logical tensor payload requested by a synthetic host-side batch."""

    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, visited) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, visited) for item in value)
    if hasattr(value, "__dict__"):
        return _tensor_bytes(vars(value), visited)
    return 0


def _current_rss_bytes() -> int | None:
    """Return Linux current RSS without adding a preflight dependency."""

    status = Path("/proc/self/statm")
    if not status.is_file():
        return None
    try:
        resident_pages = int(status.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows developer host
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value if platform.system() == "Darwin" else value * 1024


def _assert_finite(name: str, *values: Tensor | None) -> None:
    if any(value is None or not bool(torch.isfinite(value).all()) for value in values):
        raise PreflightError(f"{name} produced a missing or non-finite tensor")


def _seed_profile(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_parameter_gradient(module: torch.nn.Module) -> Tensor:
    gradient = next(
        (parameter.grad for parameter in module.parameters() if parameter.grad is not None),
        None,
    )
    if gradient is None:
        raise PreflightError("backward produced no parameter gradient")
    return gradient


ProfileWorkload = Callable[[], tuple[dict[str, Any], Any]]


def _measure_profile(
    name: str,
    device: torch.device,
    spec: dict[str, Any],
    workload: ProfileWorkload,
) -> dict[str, Any]:
    """Measure one isolated forward/backward path, including saved-tensor peak."""

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_allocated = 0
        baseline_reserved = 0
    rss_before = _current_rss_bytes()
    started = time.perf_counter()
    keepalive: Any = None
    try:
        details, keepalive = workload()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            allocated = reserved = peak_allocated = peak_reserved = 0
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower():
            raise PreflightError(
                f"profile {name!r} exhausted accelerator memory for envelope {spec}"
            ) from error
        raise
    wall_time = time.perf_counter() - started
    rss_after = _current_rss_bytes()
    result = {
        "status": "passed",
        "profile": name,
        "spec": spec,
        "memory_unit": "bytes",
        "wall_time_unit": "seconds",
        "allocated": allocated,
        "reserved": reserved,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "peak_allocated_above_baseline": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_above_baseline": max(0, peak_reserved - baseline_reserved),
        "wall_time": wall_time,
        "cpu_rss_before": rss_before,
        "cpu_rss_after": rss_after,
        "cpu_rss_above_before": (
            None if rss_before is None or rss_after is None else max(0, rss_after - rss_before)
        ),
        "cpu_peak_rss": _peak_rss_bytes(),
        **details,
    }
    del keepalive
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _conductance_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.conductance_gat.sparse import (
        PackedGraphBatch,
        SparseIncidenceConductanceLayer,
    )

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    channels = 64 if device.type == "cuda" else 8
    edge_width = 3
    use_amp = config.amp and device.type == "cuda"
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "channels": channels,
        "edge_feature_channels": edge_width,
        "amp": use_amp,
        "production_path": "SparseIncidenceConductanceLayer",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260829)
        generator = torch.Generator(device="cpu").manual_seed(20260829)
        host_batch = PackedGraphBatch(
            node_state=torch.randn(batch_size * nodes, channels, generator=generator),
            edge_index=_packed_edges(batch_size, nodes, edges),
            edge_features=torch.randn(batch_size * edges, edge_width, generator=generator),
            node_graph=torch.arange(batch_size).repeat_interleave(nodes),
            edge_graph=torch.arange(batch_size).repeat_interleave(edges),
            graph_ids=[f"preflight-{index}" for index in range(batch_size)],
            requested_step=torch.full((batch_size,), 0.02),
        )
        host_bytes = _tensor_bytes(host_batch)
        batch = host_batch.to(device)
        batch.node_state.requires_grad_(True)
        model = SparseIncidenceConductanceLayer(
            channels=channels,
            edge_feature_channels=edge_width,
            hidden_channels=channels,
            requested_step=0.02,
            mode="full",
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            output, diagnostics = model(batch, return_diagnostics=True)
            loss = output.square().mean()
        loss.backward()
        first_gradient = _first_parameter_gradient(model)
        _assert_finite("conductance profile", output, loss, batch.node_state.grad, first_gradient)
        return {
            "loss": float(loss.detach().cpu()),
            "message_sum_abs": float(
                diagnostics["node_message"].sum(dim=0).abs().max().detach().cpu()
            ),
            "host_input_tensor_bytes": host_bytes,
        }, (host_batch, batch, model, output, diagnostics, loss)

    return _measure_profile("conductance", device, spec, workload)


def _prepared_cycle_graphs(
    *,
    required_variants: Sequence[str],
    batch_size: int,
    nodes: int,
    edges: int,
    cycle_rank: int,
    generator: torch.Generator,
) -> list[Any]:
    from research.cycle_pe.features import cycle_set_statistics
    from research.cycle_pe.paper_model import PreparedGraph

    raw = torch.randn(edges, cycle_rank, generator=generator)
    orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
    cycle_set = (
        torch.as_tensor(cycle_set_statistics(orthogonal.numpy()), dtype=torch.float32)
        if "set" in required_variants
        else None
    )
    projector = orthogonal @ orthogonal.T if "projector" in required_variants else None
    edge_pairs = _base_edges(nodes, edges).T.contiguous()
    result = []
    for index in range(batch_size):
        result.append(
            PreparedGraph(
                graph_id=f"preflight-{index}",
                split="preflight",
                family="synthetic-envelope",
                num_nodes=nodes,
                cycle_rank=cycle_rank,
                edges=edge_pairs.clone(),
                node_features=torch.randn(nodes, 5, generator=generator),
                edge_features=torch.randn(edges, 3, generator=generator),
                raw_basis=orthogonal.clone(),
                cycle_set=None if cycle_set is None else cycle_set.clone(),
                projector=None if projector is None else projector.clone(),
                edge_targets=None,
                node_targets=None,
                graph_targets=torch.zeros(1),
            )
        )
    return result


def _cycle_projector_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.cycle_pe.paper_model import PaperCycleModel

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    cycle_rank = config.cycle_rank
    hidden = 64 if device.type == "cuda" else 8
    pe_dim = 32 if device.type == "cuda" else 8
    layers = 3 if device.type == "cuda" else 1
    use_amp = config.amp and device.type == "cuda"
    variants = config.cycle_variants
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "cycle_rank": cycle_rank,
        "selected_variants": list(variants),
        "projector_shape_per_graph": [edges, edges] if "projector" in variants else None,
        "hidden_dim": hidden,
        "pe_dim": pe_dim,
        "layers": layers,
        "amp": use_amp,
        "production_path": "PaperCycleModel(selected variants)",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260830)
        generator = torch.Generator(device="cpu").manual_seed(20260830)
        host_graphs = _prepared_cycle_graphs(
            required_variants=variants,
            batch_size=batch_size,
            nodes=nodes,
            edges=edges,
            cycle_rank=cycle_rank,
            generator=generator,
        )
        variant_losses: dict[str, float] = {}
        host_bytes = _tensor_bytes(host_graphs)
        for variant in variants:
            _seed_profile(20260830)
            graphs = [graph.to(device) for graph in host_graphs]
            model = PaperCycleModel(
                variant=variant,
                raw_width=cycle_rank,
                node_input_dim=5,
                edge_input_dim=3,
                edge_output_dim=0,
                node_output_dim=0,
                graph_output_dim=1,
                hidden_dim=hidden,
                pe_dim=pe_dim,
                layers=layers,
                embedding_dim=16,
            ).to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(graphs)
                predictions = torch.cat(
                    [output.graph for output in outputs if output.graph is not None]
                )
                embeddings = torch.stack([output.embedding for output in outputs])
                loss = predictions.square().mean() + embeddings.square().mean()
            loss.backward()
            first_gradient = _first_parameter_gradient(model)
            _assert_finite(
                f"cycle profile ({variant})",
                predictions,
                embeddings,
                loss,
                first_gradient,
            )
            variant_losses[variant] = float(loss.detach().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del first_gradient, predictions, embeddings, loss, outputs, model, graphs
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        return {
            "loss": max(variant_losses.values()),
            "variant_losses": variant_losses,
            "host_input_tensor_bytes": host_bytes,
        }, None

    return _measure_profile("cycle-projector", device, spec, workload)


def _tree_chart_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.tree_augmentation.paper_model import PaddedChartBatch, VariableBetaCycleEncoder

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    cycle_rank = config.cycle_rank
    hidden = 64 if device.type == "cuda" else 8
    use_amp = config.amp and device.type == "cuda"
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "cycle_rank": cycle_rank,
        "dense_chart_shape": [batch_size, edges, cycle_rank],
        "hidden_dim": hidden,
        "amp": use_amp,
        "production_path": "VariableBetaCycleEncoder(PaddedChartBatch)",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260831)
        generator = torch.Generator(device="cpu").manual_seed(20260831)
        basis = torch.randn(batch_size, edges, cycle_rank, generator=generator)
        base_pairs = _base_edges(nodes, edges).T
        host_batch = PaddedChartBatch(
            basis=basis,
            edge_features=torch.randn(batch_size, edges, 4, generator=generator),
            edge_mask=torch.ones(batch_size, edges, dtype=torch.bool),
            cycle_mask=torch.ones(batch_size, cycle_rank, dtype=torch.bool),
            edge_index=base_pairs.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            node_categories=torch.zeros(batch_size, nodes, dtype=torch.long),
            edge_categories=torch.zeros(batch_size, edges, dtype=torch.long),
            node_mask=torch.ones(batch_size, nodes, dtype=torch.bool),
            targets=torch.randn(batch_size, 1, generator=generator),
            graph_ids=tuple(f"preflight-{index}" for index in range(batch_size)),
        )
        host_bytes = _tensor_bytes(host_batch)
        batch = host_batch.to(device, pin_memory=False, non_blocking=False)
        model = VariableBetaCycleEncoder(hidden_dim=hidden, output_dim=1).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            predictions = model(batch)
            loss = nnf.mse_loss(predictions, batch.targets)
        loss.backward()
        first_gradient = _first_parameter_gradient(model)
        _assert_finite("tree-chart profile", predictions, loss, first_gradient)
        return {
            "loss": float(loss.detach().cpu()),
            "host_input_tensor_bytes": host_bytes,
        }, (host_batch, batch, model, predictions, loss)

    return _measure_profile("tree-chart", device, spec, workload)


def _brec_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    from research.cycle_pe.paper import brec_hotelling_t2
    from research.cycle_pe.paper_model import PaperCycleModel

    protocol = config.brec_protocol
    requested_batch_size = config.brec_batch_size
    if device.type == "cuda" and protocol == "official" and requested_batch_size != 16:
        raise PreflightError("official BREC CUDA profile requires --brec-batch-size 16")
    # ``research.cycle_pe.paper._brec_batches`` consumes whole graph pairs.
    # Preserve that exact behavior for custom/tiny odd batch-size requests.
    pairs_per_batch = max(1, requested_batch_size // 2)
    batch_size = 2 * pairs_per_batch
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    cycle_rank = config.cycle_rank
    hidden = 64 if device.type == "cuda" else 8
    pe_dim = 32 if device.type == "cuda" else 8
    layers = 3 if device.type == "cuda" else 1
    num_relabel = 32
    embedding_dim = 16
    variants = config.cycle_variants
    use_amp = protocol == "custom" and config.brec_amp and device.type == "cuda"
    spec = {
        "protocol": protocol,
        "requested_batch_size": requested_batch_size,
        "batch_size_graphs": batch_size,
        "official_batch_size": 16,
        "official_batch_size_match": batch_size == 16 if protocol == "official" else None,
        "selected_variants": list(variants),
        "dtype": "mixed_float32_autocast" if use_amp else "float32",
        "model_parameter_dtype": "float32",
        "amp": use_amp,
        "nodes_per_graph": nodes,
        "edges_per_graph": edges,
        "cycle_rank": cycle_rank,
        "projector_shape_per_graph": [edges, edges] if "projector" in variants else None,
        "num_relabel_pairs_for_t2": num_relabel,
        "covariance_shape": [embedding_dim, embedding_dim],
        "production_path": "PaperCycleModel(selected variants) + brec_hotelling_t2(cov/pinv)",
    }

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260901)
        generator = torch.Generator(device="cpu").manual_seed(20260901)
        host_graphs = _prepared_cycle_graphs(
            required_variants=variants,
            batch_size=batch_size,
            nodes=nodes,
            edges=edges,
            cycle_rank=cycle_rank,
            generator=generator,
        )
        variant_losses: dict[str, float] = {}
        variant_t2: dict[str, float] = {}
        host_bytes = _tensor_bytes(host_graphs)
        for variant in variants:
            _seed_profile(20260901)
            graphs = [graph.to(device) for graph in host_graphs]
            model = PaperCycleModel(
                variant=variant,
                raw_width=cycle_rank,
                node_input_dim=5,
                edge_input_dim=3,
                edge_output_dim=0,
                node_output_dim=0,
                graph_output_dim=0,
                hidden_dim=hidden,
                pe_dim=pe_dim,
                layers=layers,
                embedding_dim=embedding_dim,
            ).to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(graphs)
                embeddings = torch.stack([output.embedding for output in outputs])
                targets = -torch.ones(
                    batch_size // 2,
                    device=device,
                    dtype=embeddings.dtype,
                )
                loss = nnf.cosine_embedding_loss(embeddings[0::2], embeddings[1::2], targets)
            loss.backward()
            repeat_count = math.ceil((2 * num_relabel) / embeddings.shape[0])
            t2_embeddings = embeddings.detach().float().repeat(repeat_count, 1)[: 2 * num_relabel]
            jitter = torch.linspace(
                -1.0e-3,
                1.0e-3,
                t2_embeddings.numel(),
                device=device,
                dtype=torch.float32,
            ).reshape_as(t2_embeddings)
            t2 = brec_hotelling_t2(t2_embeddings + jitter)
            first_gradient = _first_parameter_gradient(model)
            _assert_finite(f"BREC profile ({variant})", embeddings, loss, t2, first_gradient)
            variant_losses[variant] = float(loss.detach().cpu())
            variant_t2[variant] = float(t2.detach().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del (
                first_gradient,
                t2,
                jitter,
                t2_embeddings,
                repeat_count,
                loss,
                targets,
                embeddings,
                outputs,
                model,
                graphs,
            )
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        return {
            "loss": max(variant_losses.values()),
            "hotelling_t2": max(variant_t2.values()),
            "variant_losses": variant_losses,
            "variant_hotelling_t2": variant_t2,
            "host_input_tensor_bytes": host_bytes,
        }, None

    return _measure_profile("brec", device, spec, workload)


def _reciprocal_pyg_data(
    *,
    nodes: int,
    edges: int,
    task: str,
    categorical: bool,
    seed: int,
) -> Any:
    try:
        from torch_geometric.data import Data
    except (ImportError, OSError) as error:
        raise PreflightError(
            "public-pyg profile requires torch-geometric; run scripts/setup_gpu.sh"
        ) from error

    generator = torch.Generator(device="cpu").manual_seed(seed)
    undirected = _base_edges(nodes, edges)
    directed = torch.cat((undirected, undirected.flip(0)), dim=1)
    if categorical:
        x = torch.randint(0, 2, (nodes, 9), generator=generator)
        base_attributes = torch.randint(0, 2, (edges, 3), generator=generator)
        y = torch.tensor([float(seed % 2)])
    else:
        x = torch.randn(nodes, 14, generator=generator)
        base_attributes = torch.randn(edges, 2, generator=generator)
        y = torch.arange(nodes).remainder(21)
    edge_attr = torch.cat((base_attributes, base_attributes), dim=0)
    return Data(x=x, edge_index=directed, edge_attr=edge_attr, y=y), task


def _public_pyg_profile(device: torch.device, config: ProfileConfig) -> dict[str, Any]:
    try:
        import ogb  # noqa: F401
        import torch_geometric  # noqa: F401
    except (ImportError, OSError) as error:
        raise PreflightError(
            "public-pyg profile requires importable torch-geometric and ogb; "
            "run scripts/setup_gpu.sh"
        ) from error

    from research.conductance_gat.paper import (
        PublicConductanceModel,
        _public_loader,
        _public_loss,
    )
    from research.conductance_gat.public_data import adapt_pyg_graph

    batch_size = config.batch_size
    nodes = config.nodes_per_graph
    edges = config.edges_per_graph
    hidden = 64 if device.type == "cuda" else 8
    use_amp = config.amp and device.type == "cuda"
    spec = {
        "batch_size_graphs": batch_size,
        "nodes_per_graph": nodes,
        "physical_edges_per_graph": edges,
        "hidden_dim": hidden,
        "amp": use_amp,
        "tasks": ["pascalvoc_sp/node", "ogbg_molhiv/graph"],
        "production_path": "PyG Data -> adapt_pyg_graph -> DataLoader -> PublicConductanceModel",
        "downloads": False,
    }

    def records(task: str, categorical: bool, seed_offset: int) -> list[dict[str, Any]]:
        result = []
        for index in range(batch_size):
            data, selected_task = _reciprocal_pyg_data(
                nodes=nodes,
                edges=edges,
                task=task,
                categorical=categorical,
                seed=20260902 + seed_offset + index,
            )
            result.append(adapt_pyg_graph(data, f"preflight-{task}-{index}", task=selected_task))
        return result

    def workload() -> tuple[dict[str, Any], Any]:
        _seed_profile(20260902)
        pascal_records = records("node", False, 0)
        pascal_host_bytes = _tensor_bytes(pascal_records)
        pascal_loader = _public_loader(
            pascal_records,
            batch_size=batch_size,
            shuffle=False,
            seed=20260902,
            pin_memory=False,
            num_workers=0,
        )
        pascal_batch = next(iter(pascal_loader)).to(device, non_blocking=False)
        pascal_model = PublicConductanceModel(
            pascal_records[0],
            hidden=hidden,
            num_classes=21,
            official_molecule=False,
            backbone="conductance_model",
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pascal_logits = pascal_model(pascal_batch)
            pascal_loss = _public_loss(pascal_logits, pascal_batch.y, "node")
        pascal_loss.backward()
        _assert_finite("public PascalVOC-SP profile", pascal_logits, pascal_loss)
        pascal_loss_value = float(pascal_loss.detach().cpu())
        del pascal_loader, pascal_batch, pascal_model, pascal_logits, pascal_loss
        if device.type == "cuda":
            gc.collect()

        molecule_records = records("graph", True, 10_000)
        molecule_host_bytes = _tensor_bytes(molecule_records)
        molecule_loader = _public_loader(
            molecule_records,
            batch_size=batch_size,
            shuffle=False,
            seed=20260903,
            pin_memory=False,
            num_workers=0,
        )
        molecule_batch = next(iter(molecule_loader)).to(device, non_blocking=False)
        molecule_model = PublicConductanceModel(
            molecule_records[0],
            hidden=hidden,
            num_classes=2,
            official_molecule=True,
            backbone="conductance_model",
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            molecule_logits = molecule_model(molecule_batch)
            molecule_loss = _public_loss(molecule_logits, molecule_batch.y, "graph")
        molecule_loss.backward()
        first_gradient = _first_parameter_gradient(molecule_model)
        _assert_finite(
            "public ogbg-molhiv profile",
            molecule_logits,
            molecule_loss,
            first_gradient,
        )
        return {
            "pascalvoc_node_loss": pascal_loss_value,
            "molhiv_graph_loss": float(molecule_loss.detach().cpu()),
            "host_input_tensor_bytes": pascal_host_bytes + molecule_host_bytes,
        }, (
            pascal_records,
            molecule_records,
            molecule_loader,
            molecule_batch,
            molecule_model,
            molecule_logits,
            molecule_loss,
        )

    return _measure_profile("public-pyg", device, spec, workload)


PROFILE_RUNNERS: dict[str, Callable[[torch.device, ProfileConfig], dict[str, Any]]] = {
    "conductance": _conductance_profile,
    "cycle-projector": _cycle_projector_profile,
    "tree-chart": _tree_chart_profile,
    "brec": _brec_profile,
    "public-pyg": _public_pyg_profile,
}


def _normalize_profiles(profiles: Sequence[str] | None) -> tuple[str, ...]:
    selected = ("conductance",) if profiles is None else tuple(profiles)
    if not selected:
        raise PreflightError("at least one --profile is required")
    if "all" in selected:
        if len(selected) != 1:
            raise PreflightError("--profile all cannot be combined with another profile")
        return PROFILE_NAMES
    unknown = sorted(set(selected) - set(PROFILE_NAMES))
    if unknown:
        raise PreflightError(f"unknown preflight profiles: {unknown}")
    return tuple(dict.fromkeys(selected))


def _cycle_variants(value: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError("--cycle-variants must be non-empty")
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("--cycle-variants must not contain duplicates")
    unknown = sorted(set(selected) - set(CYCLE_VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--cycle-variants contains unsupported values {unknown}; "
            f"choose from {list(CYCLE_VARIANTS)}"
        )
    return selected


def build_report(
    requested_device: str,
    *,
    allow_cpu: bool,
    require_paper_dependencies: bool,
    min_free_gb: float,
    profiles: Sequence[str] | None = None,
    profile_config: ProfileConfig | None = None,
) -> dict[str, Any]:
    if min_free_gb < 0.0 or not math.isfinite(min_free_gb):
        raise PreflightError("--min-free-gb must be a finite non-negative number")
    dependency_errors = _paper_dependency_import_errors()
    missing = _missing_paper_dependencies()
    if require_paper_dependencies and dependency_errors:
        rendered_errors = "; ".join(
            f"{name}: {message}" for name, message in sorted(dependency_errors.items())
        )
        raise PreflightError(
            "paper dependency import check failed: "
            + rendered_errors
            + "; run scripts/setup_gpu.sh"
        )
    selected_profiles = _normalize_profiles(profiles)
    requested_config = ProfileConfig() if profile_config is None else profile_config
    requested_config.validate()
    device = _resolve_device(requested_device, allow_cpu=allow_cpu)
    effective_config = _effective_profile_config(requested_config, device)
    report: dict[str, Any] = {
        "status": "passed",
        "requested_device": requested_device,
        "resolved_device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "paper_dependencies": _package_versions(),
        "missing_paper_dependencies": missing,
        "paper_dependency_import_errors": dependency_errors,
        "nvidia_smi": _nvidia_smi(),
        "profile_kind": "synthetic_shape_stress_not_dataset_e2e",
        "selected_profiles": list(selected_profiles),
        "requested_profile_config": vars(requested_config),
        "effective_profile_config": vars(effective_config),
    }
    if device.type == "cuda":
        index = device.index
        assert index is not None
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        free_gb = free_bytes / 1024**3
        if free_gb < min_free_gb:
            raise PreflightError(
                f"CUDA device {index} has {free_gb:.2f} GiB free, below {min_free_gb:.2f} GiB"
            )
        properties = torch.cuda.get_device_properties(index)
        report["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_gb": total_bytes / 1024**3,
            "free_memory_gb": free_gb,
            "device_count": torch.cuda.device_count(),
        }
    else:
        report["gpu"] = None
    profile_results: dict[str, Any] = {}
    for name in selected_profiles:
        try:
            profile_results[name] = PROFILE_RUNNERS[name](device, effective_config)
        except PreflightError:
            raise
        except Exception as error:
            raise PreflightError(
                f"profile {name!r} failed: {type(error).__name__}: {error}"
            ) from error
    report["profiles"] = profile_results
    # Compatibility for callers of the original single-path preflight report.
    if "conductance" in profile_results:
        report["incidence_forward_backward"] = profile_results["conductance"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--require-paper-deps", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        choices=("all", *PROFILE_NAMES),
        help="repeat for every experiment path to shape-stress; default: conductance",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--brec-batch-size", type=int, default=16)
    parser.add_argument("--nodes-per-graph", type=int, default=64)
    parser.add_argument("--edges-per-graph", type=int, default=128)
    parser.add_argument("--cycle-rank", type=int, default=64)
    parser.add_argument(
        "--cycle-variants",
        type=_cycle_variants,
        default=("projector",),
        help="comma-separated PaperCycleModel variants to exercise",
    )
    parser.add_argument(
        "--brec-protocol",
        choices=BREC_PROTOCOLS,
        default="official",
        help="match the selected paper runner's official or tiny/custom BREC path",
    )
    parser.add_argument(
        "--brec-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="match AMP in custom BREC; the official protocol rejects AMP",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match AMP-enabled suites; BREC official always remains float32/no-AMP",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        report = build_report(
            args.device,
            allow_cpu=args.allow_cpu,
            require_paper_dependencies=args.require_paper_deps,
            min_free_gb=args.min_free_gb,
            profiles=args.profiles,
            profile_config=ProfileConfig(
                batch_size=args.batch_size,
                brec_batch_size=args.brec_batch_size,
                nodes_per_graph=args.nodes_per_graph,
                edges_per_graph=args.edges_per_graph,
                cycle_rank=args.cycle_rank,
                amp=args.amp,
                cycle_variants=args.cycle_variants,
                brec_protocol=args.brec_protocol,
                brec_amp=args.brec_amp,
            ),
        )
    except (PreflightError, RuntimeError) as error:
        print(f"GPU PREFLIGHT FAILED: {error}")
        return 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
