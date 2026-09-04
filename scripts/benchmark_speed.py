#!/usr/bin/env python3
"""Measure fixed official training-batch CUDA forward/backward, without an optimizer.

This is an execution microbenchmark, not a paper accuracy experiment. It never
downloads data, evaluates validation/test metrics, or updates model parameters.
The conductance_gat track targets the legacy V1 classifier; conductance_v5
reuses the current V5 joint-phase model, sampling, precision and training loss.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import math
import os
import random
import re
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

DATASETS = {
    "conductance_gat": ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"),
    "conductance_v5": ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"),
    "cycle_pe_v1": ("zinc12k", "peptides_struct"),
    "cycle_pe_v2": ("zinc12k", "peptides_struct"),
    "tree_augmentation": ("csl", "zinc"),
}
TRANSDUCTIVE_DATASETS = frozenset({"cora", "citeseer", "pubmed", "ogbn-arxiv"})
ATOL, RTOL = 2e-5, 2e-4


@dataclass
class SpeedCase:
    batch: Any
    make_model: Callable[[str], Any]
    objective: Callable[[Any], Any]
    protocol: dict[str, Any]
    description: dict[str, Any]
    comparison_scope: str
    compute_context: Callable[[], Any] = nullcontext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=tuple(DATASETS),
        required=True,
        help=(
            "conductance_gat means the legacy V1 classifier; cycle_pe_v1 and "
            "cycle_pe_v2 mean the integrated Cycle PE benchmark models; "
            "conductance_v5 reuses the current V5 sampled/full training path; "
            "tree_augmentation reuses the current padded chart-view training path"
        ),
    )
    parser.add_argument("--dataset")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    parser.add_argument(
        "--tree-data-root",
        type=Path,
        default=ROOT / "research/tree_augmentation/data",
        help="Verified Tree Augmentation processed-cache root; ignored by other tracks.",
    )
    parser.add_argument(
        "--tree-arm",
        choices=("fixed_bfs", "multi_chart"),
        default="multi_chart",
        help="Exact Tree Augmentation training-view arm; ignored by other tracks.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--minimum-measure-seconds",
        type=float,
        default=2.0,
        help=(
            "Extend (never reduce) measured steps until this wall duration is reached, "
            "so utilization sampling covers active GPU work."
        ),
    )
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch-size",
        type=int,
        help="Legacy single physical mini-batch size (ignored by full-graph datasets).",
    )
    batch_group.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help=(
            "Physical mini-batch candidates measured independently. Full-graph "
            "transductive datasets accept exactly one value because batching is inapplicable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument(
        "--v5-scale-profile",
        choices=("reference", "large"),
        default="reference",
        help="V5 architecture profile; ignored by other tracks.",
    )
    parser.add_argument(
        "--v5-hardware-profile",
        choices=("portable", "a6000-48gb"),
        default="portable",
        help="V5 numeric/execution profile; ignored by other tracks.",
    )
    parser.add_argument(
        "--v5-condition",
        choices=("fixed_c", "shared_dynamic_c"),
        default="shared_dynamic_c",
        help="V5 conductance condition; ignored by other tracks.",
    )
    parser.add_argument(
        "--v5-sampling",
        choices=("auto", "full", "neighbor", "cluster"),
        default="auto",
        help="V5 sampling; auto uses cluster for ogbn-arxiv and full otherwise.",
    )
    parser.add_argument(
        "--v5-num-neighbors",
        type=int,
        nargs="+",
        default=[15, 10],
        help="V5 neighbor-sampling fanouts; ignored by other tracks.",
    )
    parser.add_argument(
        "--resource-sample-interval-seconds",
        type=float,
        default=0.1,
        help="Background GPU/CPU/RAM sampling interval for each measured variant.",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    args.dataset = args.dataset or DATASETS[args.track][0]
    if args.dataset not in DATASETS[args.track]:
        raise ValueError(f"{args.track} datasets: {DATASETS[args.track]}")
    if args.steps < 1 or args.warmup < 1 or args.seed < 0:
        raise ValueError("steps/warmup must be positive and seed nonnegative")
    if args.minimum_measure_seconds <= 0 or not math.isfinite(
        args.minimum_measure_seconds
    ):
        raise ValueError("minimum measure seconds must be finite and positive")
    if args.track == "conductance_v5":
        from research.conductance_gat.v5.protocol import HARDWARE_PROFILES

        args.v5_sampling_resolved = (
            "cluster"
            if args.v5_sampling == "auto" and args.dataset == "ogbn-arxiv"
            else "full"
            if args.v5_sampling == "auto"
            else args.v5_sampling
        )
        if args.dataset == "ppi" and args.v5_sampling_resolved != "full":
            raise ValueError("V5 PPI already supplies graph minibatches; sampling is inapplicable")
        if not args.v5_num_neighbors or any(value < 1 for value in args.v5_num_neighbors):
            raise ValueError("every V5 neighbor fanout must be positive")
        hardware = HARDWARE_PROFILES[args.v5_hardware_profile]
        args.v5_precision = hardware["precision"]
        args.v5_tf32 = hardware["tf32"]
    else:
        args.v5_sampling_resolved = None
        args.v5_precision = None
        args.v5_tf32 = None
    if args.track == "tree_augmentation":
        from research.tree_augmentation.paper import _load_settings

        tree_settings, _ = _load_settings()
        args.tree_precision = (
            "float16_autocast" if bool(tree_settings["amp"]) else "float32"
        )
        if args.include_compile:
            raise ValueError(
                "the exact Tree Augmentation training path does not support torch.compile; "
                "no substitute execution path is permitted"
            )
    else:
        tree_settings = None
        args.tree_precision = None
    requested_candidates = getattr(args, "batch_sizes", None)
    if requested_candidates is None:
        if args.batch_size is None:
            if args.track == "conductance_gat":
                args.batch_size = 2
            elif args.track in {"cycle_pe_v1", "cycle_pe_v2"}:
                args.batch_size = 32
            elif args.track == "tree_augmentation":
                if tree_settings is None:
                    raise AssertionError("Tree settings were not resolved")
                args.batch_size = int(tree_settings["batch_size"])
            else:
                from research.conductance_gat.v5.protocol import HARDWARE_PROFILES

                hardware = HARDWARE_PROFILES[args.v5_hardware_profile]
                args.batch_size = (
                    hardware["ppi_batch_size"]
                    if args.dataset == "ppi"
                    else hardware["sample_seed_batch_size"]
                    if args.v5_sampling_resolved != "full"
                    else 1
                )
        requested_candidates = [args.batch_size]
    if not requested_candidates or any(value < 1 for value in requested_candidates):
        raise ValueError("every physical batch-size candidate must be positive")
    if len(set(requested_candidates)) != len(requested_candidates):
        raise ValueError("physical batch-size candidates must be unique")
    args.batch_sizes = list(requested_candidates)
    if (
        args.track == "conductance_v5"
        and args.v5_sampling_resolved in {"neighbor", "cluster"}
        and any(value < 32 for value in args.batch_sizes)
    ):
        raise ValueError(
            "V5 sampled seed-batch candidates below 32 are forbidden by the training contract"
        )
    if (
        args.track == "conductance_v5"
        and args.v5_sampling_resolved == "full"
        and args.dataset != "ppi"
        and args.batch_sizes != [1]
    ):
        raise ValueError(
            "V5 full transductive mode has exactly one full graph; its compatibility "
            "batch value must be 1"
        )
    if not _physical_batch_size_applicable(args) and len(args.batch_sizes) != 1:
        raise ValueError(
            f"{args.dataset} is one transductive full graph: physical batch-size "
            "sweeps are inapplicable, so provide at most one compatibility value"
        )
    if getattr(args, "resource_sample_interval_seconds", 0.1) <= 0:
        raise ValueError("resource sample interval must be positive")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("Performance measurements require CUDA; no CPU fallback")
    args.data_root = args.data_root.expanduser().resolve()
    args.tree_data_root = args.tree_data_root.expanduser().resolve()


def _physical_batch_size_applicable(args: argparse.Namespace) -> bool:
    if args.track in {"cycle_pe_v1", "cycle_pe_v2", "tree_augmentation"}:
        return True
    if args.dataset == "ppi":
        return True
    return (
        args.track == "conductance_v5"
        and getattr(args, "v5_sampling_resolved", None) in {"neighbor", "cluster"}
    )


def _seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _require_cuda(device_name: str):
    import torch

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no CPU performance/research fallback")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.cuda.get_device_properties(device)
    return device


def _build_conductance_case(
    args: argparse.Namespace,
    device,
    loaded: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> SpeedCase:
    import torch
    from torch.nn import functional as F

    from research.conductance_gat.benchmark import ConductanceNodeClassifier, _make_loaders
    from research.conductance_gat.benchmark_data import load_dataset

    class ReferenceClassifier(ConductanceNodeClassifier):
        def forward(self, graph):
            # The pre-optimization classifier: graph count is inferred on-device
            # inside each operator. Everything else is the SAME current model.
            hidden = F.dropout(F.elu(self.encoder(graph.x)), self.dropout, self.training)
            node_graph = getattr(graph, "batch", None)
            if node_graph is None:
                node_graph = torch.zeros(len(hidden), dtype=torch.long, device=hidden.device)
            for operator, norm in zip(self.operators, self.norms, strict=True):
                hidden = operator(hidden, graph.incidence_edge_index, node_graph)
                hidden = F.dropout(F.elu(norm(hidden)), self.dropout, self.training)
            return self.decoder(hidden)

    payload, protocol = (
        loaded
        if loaded is not None
        else load_dataset(args.dataset, args.data_root, allow_download=False)
    )
    loader_args = argparse.Namespace(
        model_seed=args.seed, batch_size=args.batch_size, workers=0, pin_memory=True
    )
    data, indices = _make_loaders(payload, loader_args, device)
    if indices is None:
        batch = next(iter(data["train"])).to(device)
        selected = None
        selection = "first seeded/shuffled official training graph minibatch"
        actual_batch_size = int(batch.num_graphs)
        if actual_batch_size != args.batch_size:
            raise ValueError(
                f"requested physical batch size {args.batch_size}, but the official "
                f"PPI training split produced {actual_batch_size}; candidate not measured"
            )
        batch_applicable = True
    else:
        batch = data
        selected = indices["train"]
        selection = "full official transductive graph; loss uses train indices only"
        actual_batch_size = None
        batch_applicable = False

    def make_model(kind):
        model_type = ReferenceClassifier if kind == "reference" else ConductanceNodeClassifier
        return model_type(
            batch.x.shape[1], payload["classes"], hidden_channels=64, layers=2, dropout=0.5
        )

    def objective(predicted):
        if selected is None:
            return F.binary_cross_entropy_with_logits(predicted, batch.y)
        return F.cross_entropy(
            predicted.index_select(0, selected), batch.y.index_select(0, selected)
        )

    return SpeedCase(
        batch,
        make_model,
        objective,
        protocol,
        {
            "selection": selection,
            "nodes": batch.x.shape[0],
            "physical_edges": batch.incidence_edge_index.shape[1],
            "graphs": int(batch.num_graphs) if indices is None else 1,
            "labels_in_loss": batch.y.numel() if selected is None else selected.numel(),
            "requested_physical_batch_size": args.batch_size,
            "actual_physical_batch_size": actual_batch_size,
            "physical_batch_size_unit": "graphs" if batch_applicable else "full_graph",
            "physical_batch_size_applicable": batch_applicable,
            "physical_batch_size_reason": (
                None
                if batch_applicable
                else (
                    "dataset is one transductive full graph; the compatibility "
                    "value is not applied"
                )
            ),
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size": actual_batch_size,
            "effective_batch_size_unit": "graphs" if batch_applicable else "full_graph",
            "data_loader_workers": 0,
            "pin_memory": True,
            "prefetch": "single fixed batch is materialized before timing",
            "cache": "verified official dataset cache loaded once for all candidates",
            "model_configuration": {
                "name": "legacy_conductance_gat_v1",
                "hidden_channels": 64,
                "layers": 2,
                "conductance_layers": 2,
                "attention_heads": 0,
                "dropout": 0.5,
            },
        },
        "Same current classifier; reference restores per-layer GPU graph-count max(). "
        "Both variants use the same indexed loss. Excludes epoch metric accumulation, "
        "loader/transfer, optimizer, checkpoint and validation overhead; NOT a whole-repo speedup.",
    )


def _build_v5_case(
    args: argparse.Namespace,
    device,
    loaded: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> SpeedCase:
    import torch

    from research.conductance_gat.ablation.train import training_loss
    from research.conductance_gat.benchmark_data import load_dataset
    from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
    from research.conductance_gat.v5.protocol import (
        COMMON,
        CONDITIONS,
        HARDWARE_PROFILES,
        SCALE_PROFILES,
        beta_configuration,
    )
    from research.conductance_gat.v5.train import (
        _prepare_data,
        _training_batches,
        configure_phase,
        validate_hardware_runtime,
    )

    payload, protocol = (
        loaded
        if loaded is not None
        else load_dataset(args.dataset, args.data_root, allow_download=False)
    )
    hardware = HARDWARE_PROFILES[args.v5_hardware_profile]
    sampled = args.v5_sampling_resolved in {"neighbor", "cluster"}
    execution_args = argparse.Namespace(
        dataset=args.dataset,
        sampling=args.v5_sampling_resolved,
        batch_size=args.batch_size if args.dataset == "ppi" else 1,
        workers=0,
        model_seed=args.seed,
        sample_seed_batch_size=(
            args.batch_size if sampled else hardware["sample_seed_batch_size"]
        ),
        num_neighbors=list(args.v5_num_neighbors),
        sample_prefetch=hardware["sample_prefetch"],
        pin_memory=hardware["pin_memory"],
        hardware_profile=args.v5_hardware_profile,
        precision=hardware["precision"],
        tf32=hardware["tf32"],
        activation_checkpoint=hardware["activation_checkpoint"],
        edge_chunk_size=hardware["edge_chunk_size"],
    )
    hardware_runtime = validate_hardware_runtime(execution_args, device)
    data, indices, sampler = _prepare_data(payload, execution_args, device)
    iterator = _training_batches(
        data,
        indices,
        sampler,
        1,
        device,
        args.seed,
        execution_args,
    )
    try:
        batch, selected = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("V5 official training source produced no batch") from exc
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if args.dataset == "ppi":
        actual_batch_size = int(batch.num_graphs)
        batch_unit = "graphs"
    elif sampled:
        if selected is None:
            raise RuntimeError("V5 sampled batch did not carry its supervised seed indices")
        actual_batch_size = int(selected.numel())
        batch_unit = "seed_nodes"
    else:
        actual_batch_size = None
        batch_unit = "full_graph"
    if actual_batch_size is not None and actual_batch_size != args.batch_size:
        raise ValueError(
            f"requested V5 physical batch size {args.batch_size} {batch_unit}, but "
            f"the official training path produced {actual_batch_size}; candidate not measured"
        )
    architecture = dict(SCALE_PROFILES[args.v5_scale_profile])
    architecture.update(
        beta_configuration(
            COMMON["beta_parameterization"],
            COMMON["beta_initial"],
        )
    )

    def make_model(_kind):
        model = GraphConditionedConductanceNodeClassifier(
            payload["graphs"][0]["x"].shape[1],
            payload["classes"],
            **architecture,
            conductance_mode=CONDITIONS[args.v5_condition]["conductance_mode"],
            max_log_conductance=COMMON["max_log_conductance"],
            edge_chunk_size=hardware["edge_chunk_size"],
            activation_checkpoint=hardware["activation_checkpoint"],
        )
        configure_phase(model, "joint", 0)
        return model

    def objective(predicted):
        return training_loss(predicted, batch, selected)[0]

    return SpeedCase(
        batch,
        make_model,
        objective,
        protocol,
        {
            "selection": (
                "first deterministic epoch-1 V5 sampled training seed/block"
                if sampled
                else "first deterministic epoch-1 official PPI graph minibatch"
                if args.dataset == "ppi"
                else "official transductive full graph with training-index loss"
            ),
            "nodes": int(batch.x.shape[0]),
            "physical_edges": int(batch.incidence_edge_index.shape[1]),
            "graphs": int(batch.num_graphs) if args.dataset == "ppi" else 1,
            "labels_in_loss": (
                int(selected.numel()) if selected is not None else int(batch.y.numel())
            ),
            "requested_physical_batch_size": args.batch_size,
            "actual_physical_batch_size": actual_batch_size,
            "physical_batch_size_unit": batch_unit,
            "physical_batch_size_applicable": actual_batch_size is not None,
            "physical_batch_size_reason": (
                None
                if actual_batch_size is not None
                else "V5 full mode processes the one transductive graph"
            ),
            "v5_scale_profile": args.v5_scale_profile,
            "v5_architecture": architecture,
            "v5_condition": args.v5_condition,
            "v5_sampling": args.v5_sampling_resolved,
            "v5_num_neighbors": list(args.v5_num_neighbors),
            "v5_hardware_profile": args.v5_hardware_profile,
            "precision": hardware["precision"],
            "tf32": hardware["tf32"],
            "activation_checkpoint": hardware["activation_checkpoint"],
            "edge_chunk_size": hardware["edge_chunk_size"],
            "hardware_validation": hardware_runtime,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size": actual_batch_size,
            "effective_batch_size_unit": batch_unit,
            "data_loader_workers": 0,
            "pin_memory": hardware["pin_memory"],
            "prefetch": (
                "one deterministic sampled batch prepared ahead"
                if hardware["sample_prefetch"] and sampled
                else "disabled for this fixed-batch measurement"
            ),
            "cache": "verified official dataset cache loaded once for all candidates",
            "production_path_identity": {
                "model": (
                    "research.conductance_gat.v5.model."
                    "GraphConditionedConductanceNodeClassifier"
                ),
                "training_batch": (
                    "research.conductance_gat.v5.train._prepare_data/"
                    "_training_batches"
                ),
                "loss": "research.conductance_gat.ablation.train.training_loss",
                "phase": "research.conductance_gat.v5.train.configure_phase(joint, epoch=0)",
            },
        },
        (
            "Current V5 joint-phase forward/train-label-loss/backward on its exact "
            "official full, PPI minibatch, or deterministic sampled seed/block path. "
            "No optimizer update, validation, checkpoint, or batch-size fallback."
        ),
        lambda: torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=hardware["precision"] == "bf16",
        ),
    )


def _build_cycle_case(
    args: argparse.Namespace,
    device,
    loaded: tuple[dict[str, list[Any]], dict[str, Any]] | None = None,
) -> SpeedCase:
    from research.cycle_pe.v2.data import collate, load_benchmark
    from research.cycle_pe.v2.model import CycleBasisPEModel

    splits, protocol = (
        loaded
        if loaded is not None
        else load_benchmark(args.data_root, args.dataset, allow_download=False)
    )
    if args.batch_size > len(splits["train"]):
        raise ValueError(
            f"requested physical batch size {args.batch_size}, but the official training "
            f"split contains {len(splits['train'])} graphs; candidate not measured"
        )
    selected = splits["train"][: args.batch_size]
    batch = collate(selected).to(device)

    def make_model(kind):
        return CycleBasisPEModel(
            dataset=args.dataset,
            basis_execution="reference" if kind == "reference" else "batched",
        )

    return SpeedCase(
        batch,
        make_model,
        lambda predicted: (predicted.float() - batch.y).abs().mean(),
        protocol,
        {
            "selection": "first official training graphs in source order",
            "nodes": batch.x.shape[0],
            "physical_edges": batch.edge_index.shape[1],
            "graphs": len(selected),
            "basis_ranks": [basis.shape[1] for basis in batch.cycle_bases],
            "basis_pairs": sum(basis.numel() for basis in batch.cycle_bases),
            "requested_physical_batch_size": args.batch_size,
            "actual_physical_batch_size": len(selected),
            "physical_batch_size_unit": "graphs",
            "physical_batch_size_applicable": True,
            "physical_batch_size_reason": None,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size": len(selected),
            "effective_batch_size_unit": "graphs",
            "data_loader_workers": 0,
            "pin_memory": False,
            "prefetch": "single fixed batch is collated and transferred before timing",
            "cache": "verified Cycle PE basis cache loaded once for all candidates",
            "model_configuration": {
                "name": "cycle_basis_pe_v2",
                "hidden_channels": 128,
                "pe_dimension": 64,
                "layers": 10,
                "attention_heads": 0,
                "ffn_multiplier": 4,
                "dropout": 0.0,
                "basis_pair_budget": 32768,
            },
        },
        "Same current Cycle PE v2 backbone/parameters; compares reference per-graph "
        "full-basis encoder with bounded batched full-basis encoder. Excludes data "
        "preparation/transfer, optimizer, checkpoint and validation; no basis truncation.",
    )


def _build_cycle_v1_case(
    args: argparse.Namespace,
    device,
    loaded: tuple[dict[str, list[Any]], dict[str, Any]] | None = None,
) -> SpeedCase:
    from research.cycle_pe.benchmark import _loader
    from research.cycle_pe.benchmark_data import load_benchmark
    from research.cycle_pe.benchmark_models import (
        CyclePEModel,
        architecture_protocol,
    )

    splits, protocol = (
        loaded
        if loaded is not None
        else load_benchmark(args.data_root, args.dataset, allow_download=False)
    )
    loader_args = argparse.Namespace(
        model_seed=args.seed,
        batch_size=args.batch_size,
        workers=0,
    )
    iterator = iter(_loader(splits["train"], loader_args, train=True))
    try:
        batch = next(iterator).to(device)
    except StopIteration as exc:
        raise RuntimeError("Cycle PE V1 official training split produced no batch") from exc
    actual_batch_size = int(len(batch.ptr) - 1)
    if actual_batch_size != args.batch_size:
        raise ValueError(
            f"requested physical batch size {args.batch_size}, but the official "
            f"Cycle PE V1 training loader produced {actual_batch_size}; "
            "candidate not measured"
        )

    def make_model(_kind):
        return CyclePEModel(
            dataset=args.dataset,
            hidden=64,
            pe_dim=32,
            layers=3,
        )

    return SpeedCase(
        batch,
        make_model,
        lambda predicted: (predicted.float() - batch.y).abs().mean(),
        protocol
        | {
            "architecture": architecture_protocol(),
            "microbenchmark_train_split_graphs": len(splits["train"]),
        },
        {
            "selection": (
                "first seeded/shuffled batch from the exact current Cycle PE V1 "
                "official training DataLoader"
            ),
            "nodes": int(batch.x.shape[0]),
            "physical_edges": int(batch.edge_index.shape[1]),
            "graphs": actual_batch_size,
            "labels_in_loss": int(batch.y.numel()),
            "requested_physical_batch_size": args.batch_size,
            "actual_physical_batch_size": actual_batch_size,
            "physical_batch_size_unit": "graphs",
            "physical_batch_size_applicable": True,
            "physical_batch_size_reason": None,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size": actual_batch_size,
            "effective_batch_size_unit": "graphs",
            "data_loader_workers": 0,
            "pin_memory": True,
            "prefetch": "one deterministic real training batch is transferred before timing",
            "cache": (
                "all verified official Cycle PE V1 splits loaded once for the "
                "candidate sweep"
            ),
            "production_path_identity": {
                "model": "research.cycle_pe.benchmark_models.CyclePEModel",
                "training_batch": "research.cycle_pe.benchmark._loader(train=True)",
                "loss": "research.cycle_pe.benchmark MAE training objective",
            },
            "model_configuration": {
                "name": "cycle_set_gnn",
                "hidden_channels": 64,
                "pe_dimension": 32,
                "layers": 3,
                "attention_heads": 0,
            },
        },
        (
            "Current integrated Cycle PE V1 model, seeded official training loader, "
            "MAE loss and backward. Excludes data preparation/transfer, optimizer "
            "and GradScaler update, validation, checkpoint and paper metrics. "
            "This is a non-paper fixed-batch microbenchmark with zero parameter updates."
        ),
    )


def _load_tree_inputs(args: argparse.Namespace) -> dict[str, Any]:
    from chartgat.seeds import resolve_seed_axes
    from research.tree_augmentation.paper import (
        _load_settings,
        _prepare_dataset,
        _training_views,
    )

    settings, config_path = _load_settings()
    seed_axes = resolve_seed_axes(args.seed)
    dataset = _prepare_dataset(
        args.dataset,
        args.tree_data_root,
        seed_axes=seed_axes,
        allow_download=False,
    )
    fixed_train, multi_train = _training_views(
        dataset,
        settings=settings,
        chart_seed=seed_axes.chart,
    )
    return {
        "dataset": dataset,
        "settings": settings,
        "config_path": config_path,
        "seed_axes": seed_axes,
        "views": {
            "fixed_bfs": fixed_train,
            "multi_chart": multi_train,
        },
    }


def _build_tree_case(
    args: argparse.Namespace,
    device,
    loaded: dict[str, Any] | None = None,
) -> SpeedCase:
    import numpy as np
    import torch
    from torch.nn import functional as F

    from research.tree_augmentation.paper import _output_dim, _protocol_name
    from research.tree_augmentation.paper_model import (
        VariableBetaCycleEncoder,
        _unique_graph_targets,
        collate_chart_views,
    )

    inputs = loaded if loaded is not None else _load_tree_inputs(args)
    dataset = inputs["dataset"]
    settings = inputs["settings"]
    seed_axes = inputs["seed_axes"]
    views = inputs["views"][args.tree_arm]
    if not views:
        raise RuntimeError(f"Tree Augmentation {args.tree_arm} training views are empty")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_axes.model + 101)
    sampled_indices = torch.randint(
        len(views),
        (1, args.batch_size),
        generator=generator,
    )[0].tolist()
    selected_views = [views[index] for index in sampled_indices]
    cpu_batch = collate_chart_views(selected_views)
    use_pin_memory = bool(settings["pin_memory"] and device.type == "cuda")
    use_non_blocking = bool(settings["non_blocking"] and device.type == "cuda")
    batch = cpu_batch.to(
        device,
        pin_memory=use_pin_memory,
        non_blocking=use_non_blocking,
    )
    actual_batch_size = len(selected_views)
    if actual_batch_size != args.batch_size:
        raise AssertionError(
            f"Tree Augmentation sampler returned {actual_batch_size} chart views for "
            f"requested batch {args.batch_size}; candidate not measured"
        )

    graph_targets = _unique_graph_targets(views)
    if dataset.task_type == "regression":
        target_mean = graph_targets.mean(axis=0)
        target_scale = graph_targets.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
    elif dataset.task_type == "classification":
        target_mean = np.zeros(1, dtype=np.float64)
        target_scale = np.ones(1, dtype=np.float64)
    else:
        raise ValueError(
            f"unsupported Tree Augmentation task type: {dataset.task_type}"
        )
    mean_tensor = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(target_scale, dtype=torch.float32, device=device)
    output_dim = _output_dim(dataset)

    def make_model(_kind):
        return VariableBetaCycleEncoder(
            hidden_dim=int(settings["hidden_dim"]),
            output_dim=output_dim,
            message_layers=int(settings["message_layers"]),
        )

    def objective(predicted):
        if dataset.task_type == "classification":
            return F.cross_entropy(predicted, batch.targets[:, 0].long())
        normalized = (batch.targets - mean_tensor) / scale_tensor
        return F.mse_loss(predicted, normalized)

    use_amp = bool(settings["amp"] and device.type == "cuda")
    unique_graphs = len({view.graph_id for view in selected_views})
    all_train_graphs = len({view.graph_id for view in views})
    return SpeedCase(
        batch,
        make_model,
        objective,
        {
            "track": "tree_augmentation_only",
            "suite": dataset.suite,
            "protocol": _protocol_name(dataset.suite),
            "task_type": dataset.task_type,
            "target_names": list(dataset.target_names),
            "dataset_data_sha256": dataset.data_sha256,
            "dataset_cache_integrity": {
                "full_cache_loaded": True,
                "all_declared_splits_validated": True,
                "loaded_and_validated_splits": sorted(
                    {record.split for record in dataset.records}
                ),
            },
            "seed_axes": seed_axes.to_manifest(),
            "tree_arm": args.tree_arm,
            "full_dataset_graphs": len(dataset.records),
            "official_training_graphs": all_train_graphs,
            "constructed_training_chart_views": len(views),
            "config_path": str(inputs["config_path"]),
        },
        {
            "selection": (
                "first seeded-with-replacement batch from the exact current Tree "
                "Augmentation training sampler over all constructed training views"
            ),
            "tree_arm": args.tree_arm,
            "sampled_chart_view_indices": sampled_indices,
            "nodes": sum(view.num_nodes for view in selected_views),
            "physical_edges": sum(len(view.edges) for view in selected_views),
            "graphs": unique_graphs,
            "unique_physical_graphs": unique_graphs,
            "chart_views": actual_batch_size,
            "labels_in_loss": int(batch.targets.numel()),
            "requested_physical_batch_size": args.batch_size,
            "actual_physical_batch_size": actual_batch_size,
            "physical_batch_size_unit": "chart_views",
            "physical_batch_size_applicable": True,
            "physical_batch_size_reason": None,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "effective_batch_size": actual_batch_size,
            "effective_batch_size_unit": "chart_views",
            "data_loader_workers": 0,
            "pin_memory": use_pin_memory,
            "non_blocking_transfer": use_non_blocking,
            "prefetch": "one deterministic real padded chart batch is transferred before timing",
            "cache": (
                "the full verified dataset cache and all official training chart views "
                "are loaded/constructed once for the candidate sweep"
            ),
            "production_path_identity": {
                "model": (
                    "research.tree_augmentation.paper_model."
                    "VariableBetaCycleEncoder"
                ),
                "training_views": (
                    "research.tree_augmentation.paper._training_views"
                ),
                "batch": (
                    "research.tree_augmentation.paper_model.collate_chart_views "
                    "with the fit_downstream_model seed+101 replacement sampler"
                ),
                "loss": (
                    "fit_downstream_model CSL cross_entropy or ZINC normalized MSE"
                ),
            },
            "precision": "float16_autocast" if use_amp else "float32",
            "padded_input_shapes": {
                "basis": list(batch.basis.shape),
                "edge_index": list(batch.edge_index.shape),
                "node_categories": list(batch.node_categories.shape),
                "targets": list(batch.targets.shape),
            },
            "target_normalization": {
                "mean": target_mean.tolist(),
                "scale": target_scale.tolist(),
                "source": "all unique physical graph targets in the selected full training arm",
            },
            "model_configuration": {
                "name": "variable_beta_cycle_encoder",
                "hidden_dimension": int(settings["hidden_dim"]),
                "message_layers": int(settings["message_layers"]),
                "output_dimension": output_dim,
                "attention_heads": 0,
            },
        },
        (
            "Current Tree Augmentation padded ChartBatch, VariableBetaCycleEncoder "
            "and exact CSL cross-entropy or ZINC train-target-normalized MSE under "
            "the configured autocast policy. Data loading/transfer, optimizer, "
            "GradScaler update, validation, checkpoints and paper metrics are excluded. "
            "The physical batch unit is chart views (which may repeat a physical graph); "
            "this is a non-paper microbenchmark with zero parameter updates."
        ),
        lambda: torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ),
    )


def _load_case_inputs(args: argparse.Namespace):
    if args.track in {"conductance_gat", "conductance_v5"}:
        from research.conductance_gat.benchmark_data import load_dataset

        return load_dataset(args.dataset, args.data_root, allow_download=False)
    if args.track == "cycle_pe_v1":
        from research.cycle_pe.benchmark_data import load_benchmark

        return load_benchmark(args.data_root, args.dataset, allow_download=False)
    if args.track == "tree_augmentation":
        return _load_tree_inputs(args)
    from research.cycle_pe.v2.data import load_benchmark

    return load_benchmark(args.data_root, args.dataset, allow_download=False)


def _build_case(args: argparse.Namespace, device, loaded=None) -> SpeedCase:
    if args.track == "conductance_gat":
        return _build_conductance_case(args, device, loaded)
    if args.track == "conductance_v5":
        return _build_v5_case(args, device, loaded)
    if args.track == "cycle_pe_v1":
        return _build_cycle_v1_case(args, device, loaded)
    if args.track == "tree_augmentation":
        return _build_tree_case(args, device, loaded)
    return _build_cycle_case(args, device, loaded)


def _implementation_hashes(track: str) -> dict[str, str]:
    files = [
        "scripts/benchmark_speed.py",
        "scripts/benchmark_speed.sh",
        "src/chartgat/execution.py",
        "src/chartgat/observability.py",
    ]
    if track == "conductance_gat":
        files.extend(
            f"research/conductance_gat/{name}"
            for name in ("benchmark.py", "benchmark_data.py", "sparse.py")
        )
        hashes = {}
    elif track == "conductance_v5":
        from research.conductance_gat.v5.train import implementation_source_hashes

        hashes = implementation_source_hashes()
    elif track == "cycle_pe_v1":
        from research.cycle_pe.benchmark import implementation_hashes

        hashes = implementation_hashes()
    elif track == "tree_augmentation":
        files.extend(
            (
                "research/tree_augmentation/paper.py",
                "research/tree_augmentation/paper_data.py",
                "research/tree_augmentation/paper_model.py",
                "research/tree_augmentation/augmentation.py",
                "research/tree_augmentation/datasets.yaml",
                "research/tree_augmentation/config.yaml",
                "src/chartgat/seeds.py",
            )
        )
        hashes = {}
    else:
        from research.cycle_pe.v2.benchmark import implementation_hashes

        hashes = implementation_hashes()
    hashes.update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files})
    return hashes


def _planned_variants(args: argparse.Namespace) -> list[str]:
    if args.track in {"conductance_v5", "cycle_pe_v1"}:
        return ["current"] + (["compiled"] if args.include_compile else [])
    if args.track == "tree_augmentation":
        return ["current"]
    return ["reference", "optimized"] + (
        ["compiled"] if args.include_compile else []
    )


def _probe(model, case: SpeedCase) -> dict[str, Any]:
    """One eval-mode train-label objective/gradient probe; never a test evaluation."""
    import torch

    model.eval()
    model.zero_grad(set_to_none=True)
    with case.compute_context():
        raw_prediction = model(case.batch)
        loss = case.objective(raw_prediction)
    predicted = raw_prediction.float()
    if not torch.isfinite(predicted).all() or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite correctness-probe prediction/loss")
    loss.backward()
    gradients = {}
    missing_trainable_gradients = []
    trainable_gradient_count = 0
    for name, parameter in model.named_parameters():
        gradient = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        if parameter.requires_grad and gradient is None:
            missing_trainable_gradients.append(name)
        elif parameter.requires_grad:
            trainable_gradient_count += 1
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Nonfinite correctness-probe gradient: {name}")
        gradients[name] = gradient
    if missing_trainable_gradients:
        raise AssertionError(
            "Trainable parameters are disconnected from the measured loss: "
            + ", ".join(missing_trainable_gradients)
        )
    result = {
        "prediction": predicted.detach().cpu(),
        "loss": loss.detach().cpu(),
        "gradients": gradients,
        "integrity": {
            "status": "passed",
            "finite_prediction": True,
            "finite_loss": True,
            "all_trainable_parameters_have_gradients": True,
            "all_trainable_parameter_gradients_finite": True,
            "trainable_parameter_gradient_tensors": trainable_gradient_count,
            "optimizer_steps": 0,
            "parameter_updates": 0,
        },
    }
    model.zero_grad(set_to_none=True)
    return result


def _compare_probes(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.testing.assert_close(
        candidate["prediction"], reference["prediction"], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(candidate["loss"], reference["loss"], atol=ATOL, rtol=RTOL)
    if candidate["gradients"].keys() != reference["gradients"].keys():
        raise AssertionError("Parameter names differ between execution variants")
    maximum_gradient_error = 0.0
    gradients_compared = 0
    for name, expected in reference["gradients"].items():
        actual = candidate["gradients"][name]
        if expected is None or actual is None:
            if expected is not actual:
                raise AssertionError(f"Gradient participation differs: {name}")
            continue
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL, msg=f"gradient: {name}")
        maximum_gradient_error = max(maximum_gradient_error, float((actual - expected).abs().max()))
        gradients_compared += 1
    return {
        "passed": True,
        "atol": ATOL,
        "rtol": RTOL,
        "prediction_max_abs_error": float(
            (candidate["prediction"] - reference["prediction"]).abs().max()
        ),
        "gradient_max_abs_error": maximum_gradient_error,
        "parameter_gradients_compared": gradients_compared,
        "mode": "eval-mode forward and train-label-loss backward; dropout disabled",
    }


def _equivalence_result(
    reference: dict[str, Any] | None,
    probe: dict[str, Any],
) -> dict[str, Any]:
    if reference is None:
        return {
            "status": "not_applicable",
            "passed": None,
            "reason": (
                "this is the first/current execution path and no independent "
                "execution oracle has been measured yet; self-comparison is not "
                "reported as numerical equivalence"
            ),
        }
    return _compare_probes(reference, probe) | {"status": "passed"}


def _measure_block(model, case: SpeedCase, steps: int, device) -> dict[str, float]:
    import torch

    finite = torch.ones((), dtype=torch.bool, device=device)
    start_event, end_event = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    stream = torch.cuda.current_stream(device)
    start_event.record(stream)
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        with case.compute_context():
            predicted = model(case.batch)
            loss = case.objective(predicted)
        loss.backward()
        # Catch failure in ANY step, with one host transfer after the measured block.
        finite.logical_and_(torch.isfinite(loss.detach()))
    end_event.record(stream)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if not finite:
        raise FloatingPointError("Nonfinite loss during performance block")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if gradients and not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
        raise FloatingPointError("Nonfinite final gradient during performance block")
    return {
        "wall_seconds": seconds,
        "cuda_event_seconds": start_event.elapsed_time(end_event) / 1000,
        "seconds_per_step": seconds / steps,
        "steps_per_second": steps / seconds,
    }


def _measure_for_minimum_duration(
    model,
    case: SpeedCase,
    *,
    requested_steps: int,
    minimum_seconds: float,
    device,
) -> dict[str, float | int | bool]:
    """Measure at least requested_steps and enough synchronized work for sampling."""

    completed_steps = 0
    wall_seconds = 0.0
    cuda_event_seconds = 0.0
    measurement_blocks = 0
    next_steps = requested_steps
    while completed_steps < requested_steps or wall_seconds < minimum_seconds:
        block = _measure_block(model, case, next_steps, device)
        completed_steps += next_steps
        wall_seconds += block["wall_seconds"]
        cuda_event_seconds += block["cuda_event_seconds"]
        measurement_blocks += 1
        remaining_seconds = minimum_seconds - wall_seconds
        if remaining_seconds <= 0:
            break
        if block["seconds_per_step"] <= 0:
            raise RuntimeError("measured nonpositive seconds per step")
        next_steps = max(1, math.ceil(remaining_seconds / block["seconds_per_step"]))
    return {
        "wall_seconds": wall_seconds,
        "cuda_event_seconds": cuda_event_seconds,
        "seconds_per_step": wall_seconds / completed_steps,
        "steps_per_second": completed_steps / wall_seconds,
        "measured_steps": completed_steps,
        "requested_minimum_steps": requested_steps,
        "minimum_measure_seconds": minimum_seconds,
        "minimum_measure_duration_met": wall_seconds >= minimum_seconds,
        "measurement_blocks": measurement_blocks,
    }


def _state_fingerprint(state: dict[str, Any]) -> str:
    fingerprint = hashlib.sha256()
    for name, value in state.items():
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
    return fingerprint.hexdigest()


def _assert_trainable_parameters_unchanged(model, state: dict[str, Any]) -> None:
    import torch

    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach().cpu(), state[name])
    ]
    if changed:
        raise AssertionError(
            "Microbenchmark changed trainable parameters without an optimizer: "
            + ", ".join(changed)
        )


def _resource_scalar(resource: dict[str, Any], *path: str) -> Any:
    current: Any = resource
    for key in path:
        current = current[key]
    return current["value"]


def _failure_metadata(exc: Exception) -> dict[str, Any]:
    import torch

    is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )
    return {
        "status": "failed",
        "failure_kind": "cuda_out_of_memory" if is_oom else "execution_error",
        "error": f"{type(exc).__name__}: {exc}",
        "fallback_or_automatic_batch_reduction_applied": False,
    }


def _add_resource_and_throughput_columns(
    row: dict[str, Any], resource: dict[str, Any], case: SpeedCase
) -> None:
    seconds = row["wall_seconds"]
    steps = row["measured_steps"]
    for key, output_name in (
        ("graphs", "graphs_per_second"),
        ("nodes", "nodes_per_second"),
        ("physical_edges", "physical_edges_per_second"),
        ("labels_in_loss", "loss_label_elements_per_second"),
    ):
        row[output_name] = float(case.description[key]) * steps / seconds
    physical_batch_size = case.description["actual_physical_batch_size"]
    physical_batch_unit = case.description["physical_batch_size_unit"]
    row["physical_batch_item_unit"] = physical_batch_unit
    if isinstance(physical_batch_size, int) and not isinstance(physical_batch_size, bool):
        row["physical_batch_items_per_second"] = (
            float(physical_batch_size) * steps / seconds
        )
        row["physical_batch_throughput_unavailable_reason"] = None
    else:
        row["physical_batch_items_per_second"] = None
        row["physical_batch_throughput_unavailable_reason"] = (
            case.description["physical_batch_size_reason"]
        )
    row.update(
        gpu_sm_utilization_mean_percent=_resource_scalar(
            resource, "interval_series", "gpu_sm_utilization_percent", "mean"
        ),
        gpu_sm_utilization_max_percent=_resource_scalar(
            resource, "interval_series", "gpu_sm_utilization_percent", "maximum"
        ),
        gpu_memory_controller_utilization_mean_percent=_resource_scalar(
            resource,
            "interval_series",
            "gpu_memory_controller_utilization_percent",
            "mean",
        ),
        gpu_memory_controller_utilization_max_percent=_resource_scalar(
            resource,
            "interval_series",
            "gpu_memory_controller_utilization_percent",
            "maximum",
        ),
        average_cpu_percent_of_allocated_capacity=_resource_scalar(
            resource, "summary", "average_cpu_percent_of_allocated_capacity"
        ),
        peak_process_resident_bytes=_resource_scalar(
            resource, "interval_series", "process_resident_bytes", "maximum"
        ),
        minimum_system_available_bytes=_resource_scalar(
            resource, "interval_series", "system_available_bytes", "minimum"
        ),
    )


def _finish_variant_monitor(
    monitor,
    device,
    row: dict[str, Any],
) -> tuple[int | None, int | None, list[tuple[str, BaseException]]]:
    """Attempt monitor termination once and retain every observable cleanup error."""

    import torch

    errors: list[tuple[str, BaseException]] = []
    peak_allocated_bytes = None
    peak_reserved_bytes = None
    try:
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(("cuda_peak_memory_query", exc))
    try:
        row["resource_observability"] = monitor.finish(
            peak_allocated_bytes=peak_allocated_bytes,
            peak_reserved_bytes=peak_reserved_bytes,
        )
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(("runtime_resource_monitor_finish", exc))
    row["resource_monitor_finish"] = {
        "status": "failed" if errors else "passed",
        "attempted": True,
        "errors": [
            {
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            for stage, error in errors
        ],
    }
    if errors:
        row["resource_observability_error"] = "; ".join(
            f"{stage}: {type(error).__name__}: {error}"
            for stage, error in errors
        )
    return peak_allocated_bytes, peak_reserved_bytes, errors


def _annotate_monitor_cleanup_errors(
    primary: BaseException,
    errors: list[tuple[str, BaseException]],
) -> None:
    for stage, cleanup_error in errors:
        primary.add_note(
            "resource monitor cleanup failed without replacing the primary error "
            f"({stage}): {type(cleanup_error).__name__}: {cleanup_error}"
        )


def _run_variant(args, device, case, state, variant, reference):
    import torch

    from chartgat.execution import configure_execution
    from chartgat.observability import RuntimeResourceMonitor

    row = {
        "variant": variant,
        "status": "running",
        "requested_physical_batch_size": case.description["requested_physical_batch_size"],
        "actual_physical_batch_size": case.description["actual_physical_batch_size"],
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }
    model = None
    probe = None
    monitor = None
    monitor_started = False
    monitor_finish_attempted = False
    interrupt_to_reraise = None
    failure_to_record = None
    try:
        _seed(args.seed)
        model = case.make_model("reference" if variant == "reference" else "optimized").to(device)
        model.load_state_dict(state)
        execution = configure_execution(
            model, argparse.Namespace(compile=variant == "compiled"), device
        )
        _seed(args.seed)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        probe = _probe(model, case)
        torch.cuda.synchronize(device)
        probe_seconds = time.perf_counter() - started
        equivalence = _equivalence_result(reference, probe)
        model.train()
        _seed(args.seed + 1)
        warmup = _measure_block(model, case, args.warmup, device)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
        baseline_reserved_bytes = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        monitor = RuntimeResourceMonitor(
            device,
            sample_interval_seconds=args.resource_sample_interval_seconds,
        )
        monitor.start()
        monitor_started = True
        _seed(args.seed + 2)
        measured = _measure_for_minimum_duration(
            model,
            case,
            requested_steps=args.steps,
            minimum_seconds=args.minimum_measure_seconds,
            device=device,
        )
        monitor_finish_attempted = True
        peak_allocated_bytes, peak_reserved_bytes, monitor_errors = (
            _finish_variant_monitor(monitor, device, row)
        )
        if monitor_errors:
            primary_monitor_error = monitor_errors[0][1]
            _annotate_monitor_cleanup_errors(
                primary_monitor_error,
                monitor_errors[1:],
            )
            raise primary_monitor_error
        if peak_allocated_bytes is None or peak_reserved_bytes is None:
            raise AssertionError("successful resource monitor finish omitted CUDA peaks")
        _assert_trainable_parameters_unchanged(model, state)
        row.update(
            status="passed",
            execution=execution,
            equivalence=equivalence,
            eval_probe_seconds_including_lazy_compile=probe_seconds,
            train_warmup_seconds_including_lazy_compile=warmup["wall_seconds"],
            warmup_steps_excluded=args.warmup,
            **measured,
            baseline_cuda_allocated_bytes=baseline_bytes,
            baseline_cuda_reserved_bytes=baseline_reserved_bytes,
            peak_cuda_allocated_bytes=peak_allocated_bytes,
            peak_cuda_reserved_bytes=peak_reserved_bytes,
            peak_cuda_incremental_bytes=peak_allocated_bytes - baseline_bytes,
            peak_cuda_reserved_incremental_bytes=peak_reserved_bytes
            - baseline_reserved_bytes,
            trainable_parameters_unchanged=True,
            production_path_integrity=probe["integrity"]
            | {
                "trainable_parameters_unchanged_after_measurement": True,
                "production_path_identity": case.description.get(
                    "production_path_identity"
                ),
            },
        )
        _add_resource_and_throughput_columns(row, row["resource_observability"], case)
    except KeyboardInterrupt as exc:
        if monitor_started and not monitor_finish_attempted:
            monitor_finish_attempted = True
            _, _, monitor_errors = _finish_variant_monitor(
                monitor,
                device,
                row,
            )
            _annotate_monitor_cleanup_errors(exc, monitor_errors)
        interrupt_to_reraise = exc
    except Exception as exc:
        failure_to_record = exc
        if monitor_started and not monitor_finish_attempted:
            monitor_finish_attempted = True
            _, _, monitor_errors = _finish_variant_monitor(
                monitor,
                device,
                row,
            )
            _annotate_monitor_cleanup_errors(exc, monitor_errors)
        row.update(_failure_metadata(exc))
    finally:
        post_variant_cleanup_errors = []
        if model is not None:
            try:
                model.zero_grad(set_to_none=True)
            except (Exception, KeyboardInterrupt) as cleanup_error:
                post_variant_cleanup_errors.append(
                    ("model_zero_grad", cleanup_error)
                )
            del model
        try:
            gc.collect()
        except (Exception, KeyboardInterrupt) as cleanup_error:
            post_variant_cleanup_errors.append(("python_gc", cleanup_error))
        try:
            torch.cuda.empty_cache()
        except (Exception, KeyboardInterrupt) as cleanup_error:
            post_variant_cleanup_errors.append(
                ("cuda_empty_cache", cleanup_error)
            )
        if post_variant_cleanup_errors:
            row["post_variant_cleanup_errors"] = [
                {
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                for stage, error in post_variant_cleanup_errors
            ]
            if interrupt_to_reraise is not None:
                for stage, cleanup_error in post_variant_cleanup_errors:
                    interrupt_to_reraise.add_note(
                        "post-interrupt cleanup failed without replacing the original "
                        f"KeyboardInterrupt ({stage}): "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            elif failure_to_record is not None:
                for stage, cleanup_error in post_variant_cleanup_errors:
                    failure_to_record.add_note(
                        "post-failure cleanup failed without replacing the original "
                        f"experiment error ({stage}): "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            else:
                primary_cleanup_error = post_variant_cleanup_errors[0][1]
                for stage, cleanup_error in post_variant_cleanup_errors[1:]:
                    primary_cleanup_error.add_note(
                        f"additional post-variant cleanup failure ({stage}): "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise primary_cleanup_error
    if interrupt_to_reraise is not None:
        raise interrupt_to_reraise
    return row, probe


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from chartgat.cache import atomic_write_bytes

    columns = (
        "requested_physical_batch_size",
        "actual_physical_batch_size",
        "variant",
        "status",
        "failure_kind",
        "error",
        "requested_minimum_steps",
        "measured_steps",
        "minimum_measure_seconds",
        "minimum_measure_duration_met",
        "measurement_blocks",
        "seconds_per_step",
        "steps_per_second",
        "graphs_per_second",
        "physical_batch_items_per_second",
        "physical_batch_item_unit",
        "physical_batch_throughput_unavailable_reason",
        "nodes_per_second",
        "physical_edges_per_second",
        "speedup_vs_reference",
        "cuda_event_seconds",
        "gpu_sm_utilization_mean_percent",
        "gpu_sm_utilization_max_percent",
        "gpu_memory_controller_utilization_mean_percent",
        "gpu_memory_controller_utilization_max_percent",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "peak_cuda_incremental_bytes",
        "average_cpu_percent_of_allocated_capacity",
        "peak_process_resident_bytes",
        "minimum_system_available_bytes",
        "eval_probe_seconds_including_lazy_compile",
        "train_warmup_seconds_including_lazy_compile",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _batch_candidate_analysis(
    candidates: list[dict[str, Any]], *, transductive: bool
) -> dict[str, Any]:
    if transductive:
        return {
            "status": "not_applicable",
            "selected_physical_batch_size": None,
            "training_batch_selection_performed": False,
            "reason": "the dataset is one transductive full graph, not a graph minibatch",
        }
    rankable = []
    evaluations = []
    for candidate in candidates:
        target = next(
            (
                row
                for row in candidate.get("variants", [])
                if row["variant"] in {"optimized", "current"}
                and row["status"] == "passed"
            ),
            None,
        )
        evaluation = {
            "requested_physical_batch_size": candidate["requested_physical_batch_size"],
            "candidate_status": candidate["status"],
            "rankable_for_microbenchmark": False,
        }
        if target is None:
            evaluation["reason"] = "current/optimized target variant did not complete"
            evaluations.append(evaluation)
            continue
        resource = target["resource_observability"]
        free_bytes = _resource_scalar(resource, "start", "gpu", "device_free_bytes")
        total_bytes = _resource_scalar(resource, "start", "gpu", "device_total_bytes")
        incremental = target["peak_cuda_reserved_incremental_bytes"]
        if not isinstance(free_bytes, (int, float)) or not isinstance(
            total_bytes, (int, float)
        ):
            evaluation["reason"] = "CUDA free/total memory observation was unavailable"
            evaluations.append(evaluation)
            continue
        projected_headroom = max(float(free_bytes) - float(incremental), 0.0)
        headroom_fraction = projected_headroom / float(total_bytes)
        physical_rate = target.get("physical_batch_items_per_second")
        physical_unit = target.get("physical_batch_item_unit")
        if not isinstance(physical_rate, (int, float)) or isinstance(physical_rate, bool):
            evaluation["reason"] = "physical-batch throughput observation was unavailable"
            evaluations.append(evaluation)
            continue
        if not isinstance(physical_unit, str) or not physical_unit:
            evaluation["reason"] = "physical-batch throughput unit was unavailable"
            evaluations.append(evaluation)
            continue
        evaluation.update(
            rankable_for_microbenchmark=headroom_fraction >= 0.10,
            projected_device_headroom_bytes=projected_headroom,
            projected_device_headroom_fraction=headroom_fraction,
            target_variant=target["variant"],
            target_physical_batch_items_per_second=physical_rate,
            physical_batch_item_unit=physical_unit,
            reason=(
                None
                if headroom_fraction >= 0.10
                else "less than 10% projected device-memory headroom in the microbenchmark"
            ),
        )
        evaluations.append(evaluation)
        if evaluation["rankable_for_microbenchmark"]:
            rankable.append((physical_rate, candidate, physical_unit))
    highest = max(rankable, key=lambda item: item[0])[1] if rankable else None
    ranked_unit = max(rankable, key=lambda item: item[0])[2] if rankable else None
    return {
        "status": (
            "informational_microbenchmark_ranking"
            if highest is not None
            else "no_rankable_microbenchmark_candidate"
        ),
        "selected_physical_batch_size": None,
        "training_batch_selection_performed": False,
        "training_batch_selection_unavailable_reason": (
            "this microbenchmark omits optimizer state, optimizer.step(), full-epoch data "
            "loading, validation, and the full training lifetime"
        ),
        "highest_observed_microbenchmark_physical_batch_size": (
            highest["requested_physical_batch_size"] if highest is not None else None
        ),
        "ranking_throughput_unit": ranked_unit,
        "ranking_rule": (
            "highest current/optimized physical batch items/second among completed "
            "candidates with the declared per-track graph, seed-node, or chart-view unit, "
            "retaining at least 10% projected device-memory headroom"
        ),
        "scope": (
            "informational forward/loss/backward microbenchmark ranking only; it is not a "
            "measured training-throughput optimum and cannot select or mutate a training default"
        ),
        "candidate_evaluations": evaluations,
    }


def _execute(args, report: dict[str, Any], output: Path) -> None:
    import torch

    from chartgat.cache import atomic_write_json
    from chartgat.observability import runtime_resource_snapshot

    device = _require_cuda(args.device)
    report["implementation_sha256"] = _implementation_hashes(args.track)
    # Precision is fixed across candidates/variants. V5 and Tree reuse their
    # current hardware/config profiles; legacy Conductance and Cycle retain FP32.
    use_tf32 = bool(args.v5_tf32) if args.track == "conductance_v5" else False
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.backends.cudnn.benchmark = False
    properties = torch.cuda.get_device_properties(device)
    gpu_name = torch.cuda.get_device_name(device)
    report["hardware"] = {
        "gpu": gpu_name,
        "visible_gpu_count": torch.cuda.device_count(),
        "selected_logical_device_index": device.index,
        "selected_total_memory_bytes": properties.total_memory,
        "selected_multiprocessor_count": properties.multi_processor_count,
        "selected_device_is_mig": "MIG" in gpu_name.upper(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "preflight_resource_snapshot": runtime_resource_snapshot(device),
    }
    loaded = _load_case_inputs(args)
    variants = _planned_variants(args)
    report["planned_variants"] = variants
    report["planned_physical_batch_sizes"] = args.batch_sizes
    report["physical_batch_size_applicable"] = _physical_batch_size_applicable(args)
    common_fingerprint = None
    for candidate_index, batch_size in enumerate(args.batch_sizes):
        candidate = {
            "candidate_index": candidate_index,
            "requested_physical_batch_size": batch_size,
            "status": "running",
            "variants": [],
        }
        report["batch_candidates"].append(candidate)
        report["active_batch_candidate"] = batch_size
        atomic_write_json(output / "report.json", report)
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.batch_size = batch_size
        candidate_interrupt = None
        try:
            _seed(args.seed)
            case = _build_case(candidate_args, device, loaded)
            candidate.update(
                batch=case.description,
                protocol=case.protocol,
                comparison_scope=case.comparison_scope,
            )
            if "protocol" not in report:
                report.update(
                    protocol=case.protocol,
                    comparison_scope=case.comparison_scope,
                )
            if len(args.batch_sizes) == 1:
                report["batch"] = case.description
            _seed(args.seed)
            initial = case.make_model("optimized")
            state = {
                name: value.detach().cpu().clone()
                for name, value in initial.state_dict().items()
            }
            fingerprint = _state_fingerprint(state)
            if common_fingerprint is None:
                common_fingerprint = fingerprint
            elif fingerprint != common_fingerprint:
                raise AssertionError(
                    "Model initialization changed between physical batch candidates"
                )
            candidate["initial_state_sha256"] = fingerprint
            candidate["trainable_parameters"] = sum(
                parameter.numel()
                for parameter in initial.parameters()
                if parameter.requires_grad
            )
            if len(args.batch_sizes) == 1:
                report["trainable_parameters"] = candidate["trainable_parameters"]
            del initial
            reference = None
            print(
                f"physical batch candidate {batch_size}: "
                f"actual={case.description['actual_physical_batch_size']} "
                f"unit={case.description['physical_batch_size_unit']}", flush=True
            )
            print(
                "variant       ms/step     steps/s   batch-items/s GPU SM mean/max    "
                "peak allocated MiB",
                flush=True,
            )
            for variant in variants:
                report["active_variant"] = variant
                atomic_write_json(output / "report.json", report)
                row, probe = _run_variant(
                    candidate_args, device, case, state, variant, reference
                )
                row["candidate_index"] = candidate_index
                candidate["variants"].append(row)
                report["variants"].append(row)
                if row["status"] != "passed":
                    candidate.update(
                        status="failed",
                        error=row["error"],
                        failure_kind=row["failure_kind"],
                    )
                    atomic_write_json(output / "report.json", report)
                    _write_csv(output / "summary.csv", report["variants"])
                    print(
                        f"{variant:<12} FAILED {row['failure_kind']}: {row['error']}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                if reference is None:
                    reference = probe
                reference_seconds = candidate["variants"][0]["seconds_per_step"]
                row["speedup_vs_reference"] = (
                    reference_seconds / row["seconds_per_step"]
                )
                atomic_write_json(output / "report.json", report)
                _write_csv(output / "summary.csv", report["variants"])
                sm_mean = row["gpu_sm_utilization_mean_percent"]
                sm_max = row["gpu_sm_utilization_max_percent"]
                sm_text = (
                    f"{sm_mean:5.1f}/{sm_max:5.1f}%"
                    if sm_mean is not None and sm_max is not None
                    else "unavailable"
                )
                batch_rate = row["physical_batch_items_per_second"]
                batch_rate_text = (
                    f"{batch_rate:15.2f}"
                    if isinstance(batch_rate, (int, float))
                    and not isinstance(batch_rate, bool)
                    else f"{'not-applicable':>15}"
                )
                print(
                    f"{variant:<12} {1000 * row['seconds_per_step']:>9.3f} "
                    f"{row['steps_per_second']:>10.2f} "
                    f"{batch_rate_text} {sm_text:>18} "
                    f"{row['peak_cuda_allocated_bytes'] / 2**20:>21.1f}",
                    flush=True,
                )
            if candidate["status"] == "running":
                candidate["status"] = "passed"
        except KeyboardInterrupt as exc:
            candidate_interrupt = exc
            candidate.update(
                status="interrupted",
                failure_kind="keyboard_interrupt",
                error=f"KeyboardInterrupt: {exc}",
                fallback_or_automatic_batch_reduction_applied=False,
            )
            report["status"] = "interrupted"
        except Exception as exc:
            candidate.update(_failure_metadata(exc))
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            persistence_errors = []
            try:
                atomic_write_json(output / "report.json", report)
            except (Exception, KeyboardInterrupt) as persistence_error:
                persistence_errors.append(("report.json", persistence_error))
            try:
                _write_csv(output / "summary.csv", report["variants"])
            except (Exception, KeyboardInterrupt) as persistence_error:
                persistence_errors.append(("summary.csv", persistence_error))
            if persistence_errors:
                if candidate_interrupt is not None:
                    for artifact, persistence_error in persistence_errors:
                        candidate_interrupt.add_note(
                            "interruption artifact persistence failed without replacing "
                            f"the original KeyboardInterrupt ({artifact}): "
                            f"{type(persistence_error).__name__}: {persistence_error}"
                        )
                else:
                    primary_persistence_error = persistence_errors[0][1]
                    for artifact, persistence_error in persistence_errors[1:]:
                        primary_persistence_error.add_note(
                            f"additional persistence failure ({artifact}): "
                            f"{type(persistence_error).__name__}: {persistence_error}"
                        )
                    raise primary_persistence_error
        if candidate_interrupt is not None:
            raise candidate_interrupt
    report["initial_state_sha256"] = common_fingerprint
    report["candidate_summary"] = {
        "planned": len(args.batch_sizes),
        "passed": sum(
            candidate["status"] == "passed" for candidate in report["batch_candidates"]
        ),
        "failed": sum(
            candidate["status"] == "failed" for candidate in report["batch_candidates"]
        ),
    }
    report["batch_candidate_analysis"] = _batch_candidate_analysis(
        report["batch_candidates"],
        transductive=not _physical_batch_size_applicable(args),
    )
    report.pop("active_batch_candidate", None)
    report.pop("active_variant", None)
    atomic_write_json(output / "report.json", report)
    if report["candidate_summary"]["failed"]:
        raise RuntimeError(
            f"{report['candidate_summary']['failed']} of "
            f"{report['candidate_summary']['planned']} physical batch candidates failed; "
            "no automatic batch reduction or fallback was applied"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir or ROOT / "runs/performance" / datetime.now(UTC).strftime(
        "speed-%Y%m%dT%H%M%S%fZ"
    )
    output = output.expanduser().resolve()
    # Even an existing empty directory is refused: every report has one owner.
    output.mkdir(parents=True, exist_ok=False)
    from chartgat.cache import atomic_write_json

    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "execution_microbenchmark_not_paper_training",
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "output_dir": str(output),
        "controls": {
            "official_training_batch_only": True,
            "allow_download": False,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "validation_or_test_metrics": False,
            "precision": (
                f"{args.v5_precision}; TF32={'enabled' if args.v5_tf32 else 'disabled'}"
                if args.track == "conductance_v5"
                else (
                    f"current Tree config: {args.tree_precision}; TF32 disabled"
                    if args.track == "tree_augmentation"
                    else "float32; AMP and TF32 disabled"
                )
            ),
            "equivalence": (
                "numerical comparison only when an independent execution variant exists; "
                "the first/current path is not_applicable rather than self-compared"
            ),
            "production_path_integrity": (
                "finite prediction/loss and every trainable gradient, exact imported "
                "model/batch/loss identity, unchanged parameters and zero optimizer updates"
            ),
            "timing": (
                "train-mode fixed-batch forward/loss/backward; "
                "CUDA-synchronized wall and event clocks"
            ),
            "warmup": (
                "excluded from steady-state steps; costs include lazy compilation, "
                "not compile-only time"
            ),
            "memory": "model, fixed batch, forward/backward; no optimizer state",
            "rng": "identical seeds per variant/phase; CUDA scatter is not bitwise deterministic",
            "throughput_unit": (
                "fixed-batch forward/backward steps, not epoch or dataset throughput"
            ),
            "batch_candidate_policy": (
                "every requested candidate is measured independently; OOM/error is "
                "recorded and no automatic reduction or fallback is applied"
            ),
            "resource_measurement": (
                "periodic process CPU/RAM, CUDA allocator and device-wide NVML-backed "
                "GPU SM/memory-controller observations per measured variant"
            ),
            "resource_sampling_scope": (
                "GPU utilization is device-wide and can include other work on a shared GPU"
            ),
            "minimum_measurement_duration": (
                f"{args.minimum_measure_seconds} seconds; measured steps are extended, "
                "never reduced, until synchronized work reaches this duration"
            ),
            "conductance_gat_scope": (
                "legacy V1 ConductanceNodeClassifier execution comparison; the V5 "
                "path is represented only by the separate conductance_v5 track"
            ),
            "conductance_v5_scope": (
                "current V5 reference/large model, selected hardware precision and "
                "deterministic full/PPI/sampled batch construction in joint phase"
            ),
            "cycle_pe_v1_scope": (
                "current integrated V1 model and seeded official training DataLoader; "
                "one fixed real batch per candidate, no optimizer update"
            ),
            "tree_augmentation_scope": (
                "current full/reference config, full official cache and training-view "
                "construction, exact seeded padded chart batch and task loss; one arm "
                "is selected explicitly and no optimizer update is performed"
            ),
        },
        "variants": [],
        "batch_candidates": [],
    }
    atomic_write_json(output / "report.json", report)
    try:
        _execute(args, report, output)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output / "report.json", report)
        _write_csv(output / "summary.csv", report["variants"])
        print(f"FAILED: {exc}\nReport: {output / 'report.json'}", file=sys.stderr, flush=True)
        return 1
    report["status"] = "passed"
    atomic_write_json(output / "report.json", report)
    _write_csv(output / "summary.csv", report["variants"])
    print(f"Performance report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
