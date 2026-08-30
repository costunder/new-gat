"""Train only our conductance model on official datasets used by GAT/GATv2.

Published competitor results are external references, not locally rerun models.
Dataset overlap does not imply identical architectures, tuning or table protocols.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chartgat.cache import atomic_publish, atomic_write_json

from .benchmark_data import DATASETS, load_dataset, sha256_file
from .sparse import SparsePositiveConductance

PROTOCOL_NOTE = (
    "Only our conductance model is trained, on official datasets/splits used by prior "
    "papers. Competitor table values must be compared externally with their complete "
    "protocols, not presented as local reproductions. Our ogbn-arxiv training is "
    "full-batch, unlike GATv2's GraphSAINT setup. No Cycle PE or tree augmentation."
)


class ConductanceConv(nn.Module):
    """Positive orientation-invariant C with stable sparse H - eta B.T C B H."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.estimator = SparsePositiveConductance(channels, 0, channels, mode="full")

    def forward(self, x: Tensor, incidence: Tensor, node_graph: Tensor) -> Tensor:
        # Computing the positive edge law and degree cap in fp32 avoids fp16 squares/overflow.
        with torch.autocast(device_type=x.device.type, enabled=False):
            state = x.float()
            tail, head = incidence
            gradient = state[head] - state[tail]
            c = self.estimator(gradient, state.new_empty((gradient.shape[0], 0)))
            flux = c[:, None] * gradient
            divergence = torch.zeros_like(state)
            divergence.index_add_(0, head, flux)
            divergence.index_add_(0, tail, -flux)
            degree = state.new_zeros(state.shape[0])
            degree.index_add_(0, head, c)
            degree.index_add_(0, tail, c)
            max_degree = state.new_zeros(int(node_graph.max()) + 1)
            max_degree.scatter_reduce_(0, node_graph, degree, reduce="amax", include_self=True)
            step = 0.95 / max_degree.clamp_min(1e-12)
            result = state - step[node_graph, None] * divergence
        return result.to(x.dtype)


class ConductanceNodeClassifier(nn.Module):
    """Our encoder/conductance-stack/prediction-head node classifier."""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        *,
        hidden_channels: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_channels < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden width/layers must be positive and dropout in [0, 1)")
        self.dropout = dropout
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)
        self.operators = nn.ModuleList(ConductanceConv(hidden_channels) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(layers))

    def forward(self, graph: Any) -> Tensor:
        h = F.dropout(F.elu(self.encoder(graph.x)), self.dropout, self.training)
        node_graph = getattr(graph, "batch", None)
        if node_graph is None:
            node_graph = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        for operator, norm in zip(self.operators, self.norms, strict=True):
            h = operator(h, graph.incidence_edge_index, node_graph)
            h = F.dropout(F.elu(norm(h)), self.dropout, self.training)
        return self.decoder(h)


def micro_f1(logits: Tensor, labels: Tensor) -> float:
    """Global node-label micro-F1, not per-graph averaging or multiclass argmax."""
    predicted, truth = logits > 0, labels > 0
    true_positive = (predicted & truth).sum().item()
    denominator = predicted.sum().item() + truth.sum().item()
    return float(2 * true_positive / denominator) if denominator else 0.0


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _device(name: str, *, prepare_only: bool) -> torch.device:
    device = torch.device(name)
    if not prepare_only and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError(
            "Matched benchmark training requires a CUDA GPU; "
            "no CPU training/fallback is implemented."
        )
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.get_device_properties(device)
    return device


def _versions() -> dict[str, str]:
    output = {"torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda)}
    for package in ("torch-geometric", "ogb", "numpy"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "not_installed"
    return output


def _selection(values: list[str], allowed: tuple[str, ...]) -> list[str]:
    selected = [
        item.strip().lower() for value in values for item in value.split(",") if item.strip()
    ]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(item not in allowed for item in selected)
    ):
        raise ValueError(f"Choose each supported value at most once from {allowed}")
    return selected


def _make_loaders(payload: dict[str, Any], args: argparse.Namespace, device: torch.device):
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    graphs = [Data(**graph) for graph in payload["graphs"]]
    if payload["dataset"] != "ppi":
        # Full graph/features are visible transductively; ONLY training-mask labels enter loss.
        return graphs[0].to(device), {
            name: mask.to(device) for name, mask in payload["splits"].items()
        }
    loaders = {}
    for split, indices in payload["splits"].items():
        generator = torch.Generator().manual_seed(args.model_seed)
        loaders[split] = DataLoader(
            [graphs[index] for index in indices],
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            generator=generator,
            pin_memory=args.pin_memory,
            persistent_workers=args.workers > 0,
        )
    return loaders, None


def train_model(
    payload: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Benchmark training requires CUDA (including direct train_model calls).")
    _seed(args.model_seed)
    data, masks = _make_loaders(payload, args, device)
    model = ConductanceNodeClassifier(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        hidden_channels=args.hidden_channels,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and amp_dtype == torch.float16)
    checkpoint = output / "best.pt"
    history: list[dict[str, Any]] = []
    best_validation, best_epoch = -float("inf"), 0
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    @torch.no_grad()
    def evaluate(split: str) -> float:
        model.eval()
        if masks is not None:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                logits = model(data)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite {split} logits: {payload['dataset']}/conductance")
            mask = masks[split]
            return float((logits[mask].argmax(dim=-1) == data.y[mask]).float().mean())
        true_positive = predicted_count = truth_count = 0
        for graph in data[split]:
            graph = graph.to(device, non_blocking=args.pin_memory)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                logits = model(graph)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite {split} logits: {payload['dataset']}/conductance")
            predicted = logits > 0
            truth = graph.y > 0
            true_positive += int((predicted & truth).sum())
            predicted_count += int(predicted.sum())
            truth_count += int(truth.sum())
        denominator = predicted_count + truth_count
        return float(2 * true_positive / denominator) if denominator else 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, label_count = 0.0, 0
        batches = [data] if masks is not None else data["train"]
        for graph in batches:
            if masks is None:
                graph = graph.to(device, non_blocking=args.pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                logits = model(graph)
                if masks is not None:
                    loss = F.cross_entropy(logits[masks["train"]], graph.y[masks["train"]])
                    count = int(masks["train"].sum())
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, graph.y)
                    count = graph.y.numel()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss: {payload['dataset']}/conductance, epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * count
            label_count += count
        validation = evaluate("validation")
        history.append(
            {"epoch": epoch, "train_loss": loss_sum / label_count, "validation": validation}
        )
        atomic_write_json(output / "history.json", history)
        if validation > best_validation:
            best_validation, best_epoch = validation, epoch
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            checkpoint_data = {
                "state_dict": state,
                "best_epoch": epoch,
                "validation": validation,
                "dataset": payload["dataset"],
                "model": "conductance",
                "architecture": {
                    "hidden_channels": args.hidden_channels,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
            }
            atomic_publish(checkpoint, lambda path, saved=checkpoint_data: torch.save(saved, path))
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{payload['dataset']}/conductance epoch={epoch} val={validation:.6f}", flush=True
            )
        if epoch - best_epoch >= args.patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    # Test is evaluated exactly once per method after validation-only model selection.
    test_metric = evaluate("test")
    result = {
        "validation": best_validation,
        "test": test_metric,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint.resolve()),
        "history": str((output / "history.json").resolve()),
        "elapsed_seconds": time.perf_counter() - start_time,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "amp_dtype": str(amp_dtype) if args.amp else "float32",
        "training": "full_batch" if masks is not None else "official_inductive_graph_minibatch",
        "model_seed": args.model_seed,
        "test_selection": "best_validation_checkpoint_only",
    }
    atomic_write_json(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("benchmark",), default="benchmark")
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/conductance_gat/benchmark")
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--chart-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", "--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.datasets = _selection(args.datasets, DATASETS)
    if min(args.batch_size, args.epochs, args.patience, args.layers) < 1 or args.workers < 0:
        raise ValueError(
            "batch size, epochs, patience, layers must be positive; workers nonnegative"
        )
    if args.hidden_channels < 1 or not 0 <= args.dropout < 1:
        raise ValueError("invalid hidden width/dropout")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay nonnegative")
    if min(args.data_seed, args.split_seed, args.chart_seed, args.model_seed) < 0:
        raise ValueError("seed values must be nonnegative")
    device = _device(args.device, prepare_only=args.prepare_only)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "track": "conductance_gat",
        "suite": "benchmark",
        "status": "running",
        "protocol_note": PROTOCOL_NOTE,
        "config": config,
        "versions": _versions(),
        "seed_axes": {
            "model_seed": args.model_seed,
            "data_seed": "not_applicable: fixed official source data",
            "split_seed": "not_applicable: official fixed masks/splits",
            "chart_seed": "not_applicable: no chart/PE/augmentation",
        },
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
        "completed": [],
        "expected": [f"{dataset}/conductance" for dataset in args.datasets],
        "sources": ["https://arxiv.org/abs/1710.10903", "https://arxiv.org/abs/2105.14491"],
        "implementation_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("benchmark.py", "benchmark_data.py", "sparse.py")
        },
        "reproducibility": (
            "Seeded runs; GPU scatter kernels can remain nondeterministic. No bitwise guarantee."
        ),
    }
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "track": "conductance_gat",
        "suite": "benchmark",
        "status": "running",
        "model_seed": args.model_seed,
        "datasets": {},
    }
    atomic_write_json(output / "manifest.json", manifest)
    try:
        for dataset in args.datasets:
            print(f"Loading official matched dataset: {dataset}", flush=True)
            payload, protocol = load_dataset(
                dataset, args.data_root, allow_download=args.allow_download
            )
            record: dict[str, Any] = {
                "metric": protocol["metric"],
                "protocol": protocol,
                "models": {},
            }
            metrics["datasets"][dataset] = record
            if args.prepare_only:
                continue
            record["models"]["conductance"] = train_model(
                payload, args, device, output / dataset / "conductance"
            )
            manifest["completed"].append(f"{dataset}/conductance")
            atomic_write_json(output / "metrics.json", metrics)
            atomic_write_json(output / "manifest.json", manifest)
            torch.cuda.empty_cache()
        if not args.prepare_only and manifest["completed"] != manifest["expected"]:
            raise RuntimeError("Incomplete matched benchmark; cannot mark passed")
        manifest["status"] = metrics["status"] = "prepared" if args.prepare_only else "passed"
    except Exception as exc:
        manifest["status"] = metrics["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(output / "manifest.json", manifest)
        atomic_write_json(output / "metrics.json", metrics)
        raise
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(output / "metrics.json", metrics)
    print(f"{manifest['status']}: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
