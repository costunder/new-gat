"""Train ONE fresh factorial arm on verified caches, with validation-only selection.

This is a causal investigation, not a benchmark test-set evaluation. Training
requires CUDA even when this module is called directly. CPU tensors are used only
by unit tests of the pure helpers. No downloads, test evaluation or AMP fallback.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chartgat.cache import atomic_publish, atomic_write_json

from ..benchmark import _binary_counts, _micro_f1_from_counts, _seed, _versions
from ..benchmark_data import load_dataset, sha256_file
from .model import FactorialNodeClassifier, is_gate_parameter, make_optimizer, state_sha256
from .protocol import COMMON, CONDITIONS, DATASETS


@dataclass(frozen=True)
class TrainingDefinition:
    """Explicit suite injection; existing factorial behavior remains the default.

    Related, separately reported experiments share this audited train/validation
    loop without replacing globals or silently changing the factorial protocol.
    """

    suite: str
    conditions: Mapping[str, Mapping[str, Any]]
    model_factory: Callable[..., nn.Module]
    optimizer_factory: Callable[[nn.Module, str], torch.optim.Optimizer]
    description: str | None = None


def _training_definition(definition: TrainingDefinition | None) -> TrainingDefinition:
    if definition is not None:
        return definition
    return TrainingDefinition(
        "conductance_factorial", CONDITIONS, FactorialNodeClassifier, make_optimizer
    )


OBSERVATION_POLICY = {
    "validation": "eval mode, no gradients, official validation labels only",
    "train_trajectory": (
        "Every epoch's FIRST actual training batch, dropout ON, before optimizer.step. "
        "PPI is a first minibatch observation, not a full training-split gradient. "
        "No extra backward or training-loader iteration is performed."
    ),
    "gradient": (
        "Raw task-loss .grad after backward and before coupled Adam L2 is applied; "
        "lambda*parameter norm is recorded separately, not an Adam update estimate. "
        "Zero decay norm yields null ratio; no epsilon denominator or clamp."
    ),
    "statistics": (
        "Conductance/rho scalar moments are exact over the observed edges/nodes. "
        "Quantiles use at most 4096 deterministic evenly spaced entries PER graph. "
        "For PPI use within-graph C CV; pooled CV also includes between-graph variation."
    ),
    "normalization": (
        "Node-degree is row preconditioning, generally nonsymmetric in Euclidean space. "
        "Both normalization choices cancel a common scale of C."
    ),
}


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **COMMON,
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "tf32": False,
        "pin_memory": True,
    }


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Conductance training requires a CUDA GPU; no CPU fallback is allowed.")
    torch.cuda.get_device_properties(device)


def _configure_fp32() -> None:
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _make_data(payload: dict[str, Any], args: argparse.Namespace, device: torch.device):
    """Do not construct a test loader or even read its split index."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if payload["dataset"] != "ppi":
        graph = Data(**payload["graphs"][0]).to(device)
        indices = {
            key: payload["splits"][key].nonzero(as_tuple=False).flatten().to(device)
            for key in ("train", "validation")
        }
        return graph, indices
    loaders = {}
    for split in ("train", "validation"):
        generator = torch.Generator().manual_seed(args.model_seed)
        loaders[split] = DataLoader(
            [Data(**payload["graphs"][i]) for i in payload["splits"][split]],
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            generator=generator,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
    return loaders, None


def training_loss(logits: Tensor, graph: Any, train_indices: Tensor | None) -> tuple[Tensor, int]:
    if train_indices is not None:
        return (
            F.cross_entropy(
                logits.index_select(0, train_indices), graph.y.index_select(0, train_indices)
            ),
            train_indices.numel(),
        )
    return F.binary_cross_entropy_with_logits(logits, graph.y), graph.y.numel()


def _moments(value: Tensor, *, rho: bool = False) -> dict[str, Any]:
    flat = value.detach().flatten().double()
    if not flat.numel():
        return {"count": 0, "mean": None, "std": None, "cv": None, "min": None, "max": None}
    if not bool(torch.isfinite(flat).all()):
        raise RuntimeError("Non-finite conductance/propagation observation")
    mean = float(flat.mean())
    std = float(flat.std(correction=0))
    count = flat.numel()
    selected = (
        flat
        if count <= 4096
        else flat.index_select(0, torch.linspace(0, count - 1, 4096, device=flat.device).long())
    )
    quantiles = torch.quantile(selected, selected.new_tensor([0.1, 0.5, 0.9])).tolist()
    result = {
        "count": count,
        "mean": mean,
        "std": std,
        "cv": std / abs(mean) if mean != 0 else None,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "quantiles": dict(zip(("p10", "p50", "p90"), quantiles, strict=True)),
        "quantile_sample_count": selected.numel(),
        "quantile_policy": "exact" if count <= 4096 else "deterministic_evenly_spaced_sample",
    }
    if rho:
        result["fraction_below_0_01"] = float((flat < 0.01).double().mean())
        result["isolated_node_count"] = int((flat == 0).sum())
    return result


def _pooled_moments(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Stable population-moment merging; do not invent pooled quantiles."""
    nonempty = [record for record in records if record["count"]]
    if not nonempty:
        return {"count": 0, "mean": None, "std": None, "cv": None}
    count = sum(record["count"] for record in nonempty)
    mean = sum(record["count"] * record["mean"] for record in nonempty) / count
    variance = (
        sum(
            record["count"] * (record["std"] ** 2 + (record["mean"] - mean) ** 2)
            for record in nonempty
        )
        / count
    )
    std = math.sqrt(max(variance, 0.0))
    result = {
        "count": count,
        "mean": mean,
        "std": std,
        "cv": std / abs(mean) if mean else None,
        "min": min(record["min"] for record in nonempty),
        "max": max(record["max"] for record in nonempty),
    }
    if "fraction_below_0_01" in nonempty[0]:
        result["fraction_below_0_01"] = (
            sum(record["count"] * record["fraction_below_0_01"] for record in nonempty) / count
        )
        result["isolated_node_count"] = sum(record["isolated_node_count"] for record in nonempty)
    return result


class ForwardObservation:
    """Read-only hooks on the actual forward: no second gate evaluation or RNG use."""

    def __init__(self, model: FactorialNodeClassifier) -> None:
        self.model = model
        self.records: dict[int, list[dict[str, Any]]] = {
            index: [] for index in range(len(model.operators))
        }
        self._captured: dict[int, Tensor] = {}
        self._handles: list[Any] = []

    def __enter__(self):
        try:
            for index, operator in enumerate(self.model.operators):
                self._handles.append(
                    operator.estimator.register_forward_hook(
                        lambda module, inputs, output, i=index: self._capture(i, output)
                    )
                )
                self._handles.append(
                    operator.register_forward_hook(
                        lambda module, inputs, output, i=index: self._observe(i, inputs, output)
                    )
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _capture(self, index: int, output: Tensor) -> None:
        self._captured[index] = output.detach()

    @torch.no_grad()
    def _observe(self, index: int, inputs: tuple, output: Tensor) -> None:
        state, incidence, node_graph = inputs[:3]
        state = state.detach().float()
        output = output.detach().float()
        c = self._captured.pop(index)
        tail, head = incidence
        degree = state.new_zeros(state.shape[0])
        degree.index_add_(0, tail, c)
        degree.index_add_(0, head, c)
        num_graphs = int(inputs[3]) if len(inputs) > 3 else int(node_graph.max()) + 1
        for graph_id in range(num_graphs):
            node_mask = node_graph == graph_id
            edge_mask = node_graph[tail] == graph_id
            graph_degree = degree[node_mask]
            if self.model.normalization == "global_max":
                rho = 0.95 * graph_degree / graph_degree.max().clamp_min(1e-12)
            else:
                rho = (graph_degree > 0).to(state.dtype) * 0.95
            before = state[node_mask]
            difference = output[node_mask] - before
            state_squared = float(before.double().square().sum())
            delta_squared = float(difference.double().square().sum())
            self.records[index].append(
                {
                    "graph_observation_index": len(self.records[index]),
                    "conductance": _moments(c[edge_mask]),
                    "rho": _moments(rho, rho=True),
                    "relative_conv_change": (
                        math.sqrt(delta_squared / state_squared) if state_squared else None
                    ),
                    "state_squared_norm": state_squared,
                    "change_squared_norm": delta_squared,
                }
            )

    def summary(self) -> list[dict[str, Any]]:
        output = []
        for index, records in self.records.items():
            state_squared = sum(record["state_squared_norm"] for record in records)
            change_squared = sum(record["change_squared_norm"] for record in records)
            output.append(
                {
                    "layer": index,
                    "conductance": _pooled_moments([r["conductance"] for r in records]),
                    "rho": _pooled_moments([r["rho"] for r in records]),
                    "relative_conv_change": (
                        math.sqrt(change_squared / state_squared) if state_squared else None
                    ),
                    "graphs": records,
                }
            )
        return output

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()


def _tensor_squared_norm(value: Tensor) -> float:
    # Avoid a second entire huge parameter tensor in float64.
    flat = value.detach().flatten()
    total = 0.0
    for chunk in flat.split(1_048_576):
        total += float(chunk.double().square().sum())
    return total


def parameter_norms(model: nn.Module) -> dict[str, float]:
    return {
        name: math.sqrt(_tensor_squared_norm(parameter))
        for name, parameter in model.named_parameters()
    }


def gradient_observation(
    model: nn.Module, condition: str, *, definition: TrainingDefinition | None = None
) -> dict[str, Any]:
    specification = _training_definition(definition).conditions[condition]
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {"non_gate": []}
    for name, parameter in model.named_parameters():
        group = name.split(".estimator.")[0] if is_gate_parameter(name) else "non_gate"
        groups.setdefault(group, []).append((name, parameter))
    output = {}
    for group, parameters in groups.items():
        wd = COMMON["weight_decay"] if group == "non_gate" else specification["gate_weight_decay"]
        trainable = [p for _, p in parameters if p.requires_grad]
        if not trainable:
            wd = 0.0
        parameter_norm = math.sqrt(sum(_tensor_squared_norm(p) for _, p in parameters))
        task_norm = math.sqrt(
            sum(_tensor_squared_norm(p.grad) for _, p in parameters if p.grad is not None)
        )
        decay_norm = wd * math.sqrt(sum(_tensor_squared_norm(p) for p in trainable))
        output[group] = {
            "parameter_norm": parameter_norm,
            "task_gradient_norm": task_norm,
            "weight_decay": wd,
            "decay_term_norm": decay_norm,
            "task_to_decay_ratio": task_norm / decay_norm if decay_norm > 0 else None,
            "ratio_policy": "exact" if decay_norm > 0 else "undefined_zero_decay_norm",
            "parameter_count": sum(p.numel() for _, p in parameters),
            "trainable_parameter_count": sum(p.numel() for p in trainable),
            "optimizer_included": bool(trainable),
            "missing_gradient_parameters": [name for name, p in parameters if p.grad is None],
        }
    return output


@contextmanager
def _evaluation_mode(model: nn.Module):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def evaluate_validation(
    model: FactorialNodeClassifier,
    data: Any,
    split_indices: dict[str, Tensor] | None,
    device: torch.device,
) -> dict[str, Any]:
    with _evaluation_mode(model), ForwardObservation(model) as observation:
        if split_indices is not None:
            logits = model(data)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Non-finite validation logits")
            indices = split_indices["validation"]
            value = float(
                (logits.index_select(0, indices).argmax(dim=-1) == data.y.index_select(0, indices))
                .float()
                .mean()
            )
        else:
            counts = torch.zeros(3, dtype=torch.int64, device=device)
            for graph in data["validation"]:
                graph = graph.to(device, non_blocking=True)
                logits = model(graph)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Non-finite validation logits")
                counts.add_(_binary_counts(logits, graph.y))
            value = _micro_f1_from_counts(counts)
    return {
        "metric": value,
        "layers": observation.summary(),
        "parameter_norms": parameter_norms(model),
        "mode": "eval",
        "split": "validation",
        "observation_scope": (
            "whole transductive graph states; validation labels only for metric"
            if split_indices is not None
            else "all official validation graphs"
        ),
    }


def checkpoint_payload(
    model: FactorialNodeClassifier,
    args: argparse.Namespace,
    protocol: dict[str, Any],
    initial_hash: str,
    epoch: int,
    validation: float,
    optimizer_steps: int = 0,
    *,
    definition: TrainingDefinition | None = None,
) -> dict[str, Any]:
    experiment = _training_definition(definition)
    spec = experiment.conditions[args.condition]
    gate_metadata = {"gate_mode": spec["gate_mode"]} if "gate_mode" in spec else {}
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "research_suite": experiment.suite,
        # Distinct model identity makes the existing baseline diagnostics reject it.
        "model": experiment.suite,
        "dataset": args.dataset,
        "condition": args.condition,
        "model_seed": args.model_seed,
        "architecture": {
            "hidden_channels": COMMON["hidden_channels"],
            "layers": COMMON["layers"],
            "dropout": COMMON["dropout"],
            "normalization": spec["normalization"],
            **gate_metadata,
        },
        **gate_metadata,
        "configuration": configuration(args),
        "gate_weight_decay": spec["gate_weight_decay"],
        "non_gate_weight_decay": COMMON["weight_decay"],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "cache_sha256": protocol["data_sha256"],
        "initial_state_sha256": initial_hash,
        "best_epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "validation": validation,
        "evaluation_split": "validation",
        "test_evaluated": False,
    }


def train_model(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
    *,
    definition: TrainingDefinition | None = None,
) -> dict[str, Any]:
    _require_cuda(device)
    experiment = _training_definition(definition)
    spec = experiment.conditions[args.condition]
    _configure_fp32()
    _seed(args.model_seed)
    data, split_indices = _make_data(payload, args, device)
    train_indices = None if split_indices is None else split_indices["train"]
    gate_kwargs = {"gate_mode": spec["gate_mode"]} if "gate_mode" in spec else {}
    model = experiment.model_factory(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        normalization=spec["normalization"],
        hidden_channels=COMMON["hidden_channels"],
        layers=COMMON["layers"],
        dropout=COMMON["dropout"],
        **gate_kwargs,
    ).to(device)
    initial_hash = state_sha256(model)
    optimizer = experiment.optimizer_factory(model, args.condition)
    checkpoint = output / "best.pt"
    history: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    best_validation, best_epoch = -math.inf, 0
    optimizer_steps, best_optimizer_steps = 0, 0
    best_observation: dict[str, Any] | None = None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    initial_observation = evaluate_validation(model, data, split_indices, device)
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        label_count = 0
        batches = [data] if split_indices is not None else data["train"]
        for batch_index, graph in enumerate(batches):
            if split_indices is None:
                graph = graph.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if batch_index == 0:
                with ForwardObservation(model) as training_observation:
                    logits = model(graph)
            else:
                logits = model(graph)
            loss, count = training_loss(logits, graph, train_indices)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite train loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            if batch_index == 0:
                trajectory.append(
                    {
                        "epoch": epoch,
                        "batch_index": 0,
                        "optimizer_steps_before_batch": optimizer_steps,
                        "scope": "full_graph_train_mask"
                        if train_indices is not None
                        else "first_actual_training_minibatch_only",
                        "mode": "train_dropout_on",
                        "stage": "after_task_backward_before_optimizer_step",
                        "label_count": count,
                        "train_loss": float(loss.detach()),
                        "layers": training_observation.summary(),
                        "parameter_groups": gradient_observation(
                            model, args.condition, definition=experiment
                        ),
                    }
                )
            optimizer.step()
            optimizer_steps += 1
            loss_sum.add_(loss.detach().double() * count)
            label_count += count
        if not label_count:
            raise RuntimeError("Training split produced no labels")
        validation_observation = evaluate_validation(model, data, split_indices, device)
        validation = validation_observation["metric"]
        train_loss = float(loss_sum / label_count)
        torch.cuda.synchronize(device)
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "train_loss": train_loss,
                "validation": validation,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "validation_observation": validation_observation,
                "training_first_batch": trajectory[-1],
            }
        )
        atomic_write_json(output / "history.json", history)
        if validation > best_validation:
            best_validation, best_epoch = validation, epoch
            best_optimizer_steps = optimizer_steps
            best_observation = validation_observation
            saved = checkpoint_payload(
                model,
                args,
                protocol,
                initial_hash,
                epoch,
                validation,
                optimizer_steps,
                definition=experiment,
            )
            atomic_publish(checkpoint, lambda path, state=saved: torch.save(state, path))
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{args.dataset}/{args.condition} epoch={epoch} train_loss={train_loss:.6f} "
                f"val={validation:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            break
    final_observation = validation_observation
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    selected_observation = evaluate_validation(model, data, split_indices, device)
    if abs(selected_observation["metric"] - best_validation) > 1e-4:
        raise RuntimeError("Best checkpoint validation recheck disagrees with model selection")
    torch.cuda.synchronize(device)
    return {
        "schema_version": 1,
        "status": "passed",
        "research_suite": experiment.suite,
        "dataset": args.dataset,
        "condition": args.condition,
        "model_seed": args.model_seed,
        **spec,
        "non_gate_weight_decay": COMMON["weight_decay"],
        "configuration": configuration(args),
        "cache_sha256": protocol["data_sha256"],
        "protocol": protocol,
        "initial_state_sha256": initial_hash,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "optimizer_steps": optimizer_steps,
        "best_checkpoint_optimizer_steps": best_optimizer_steps,
        "validation": best_validation,
        "metric_name": "micro_f1" if args.dataset == "ppi" else "accuracy",
        "train_loss": history[best_epoch - 1]["train_loss"],
        "train_loss_scope": "label-weighted epoch mean at selected best validation epoch",
        "final_train_loss": history[-1]["train_loss"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": str((output / "history.json").resolve()),
        "history_sha256": sha256_file(output / "history.json"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "evaluation_split": "validation",
        "test_evaluated": False,
        "versions": _versions(),
        "gpu": torch.cuda.get_device_name(device),
        "diagnostics": {
            "initial_validation": initial_observation,
            "best_validation": selected_observation,
            "best_validation_at_selection": best_observation,
            "final_validation": final_observation,
            "train_trajectory": trajectory,
            "observation_policy": OBSERVATION_POLICY,
        },
        "reproducibility": "Same seeds/initialization; GPU scatter may remain nondeterministic.",
        "timing_policy": "includes training, validation observations and checkpoint/history IO",
    }


def build_parser(*, definition: TrainingDefinition | None = None) -> argparse.ArgumentParser:
    experiment = _training_definition(definition)
    parser = argparse.ArgumentParser(description=experiment.description or __doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--condition", required=True, choices=tuple(experiment.conditions))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main(argv: list[str] | None = None, *, definition: TrainingDefinition | None = None) -> int:
    experiment = _training_definition(definition)
    args = build_parser(definition=experiment).parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    if min(args.workers, args.model_seed) < 0:
        raise ValueError("workers and model seed must be nonnegative")
    device = torch.device(args.device)
    _require_cuda(device)
    output = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if output == data_root or data_root in output.parents:
        raise ValueError("Experiment output must not be written inside the dataset cache root")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is not an empty new arm directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "research_suite": experiment.suite,
        "dataset": args.dataset,
        "condition": args.condition,
        "model_seed": args.model_seed,
        **experiment.conditions[args.condition],
        "non_gate_weight_decay": COMMON["weight_decay"],
        "configuration": configuration(args),
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    atomic_write_json(output / "metrics.json", record)
    try:
        payload, protocol = load_dataset(args.dataset, data_root, allow_download=False)
        record.update(cache_sha256=protocol["data_sha256"], protocol=protocol)
        result = train_model(payload, protocol, args, device, output, definition=experiment)
        record.update(result)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output / "metrics.json", record)
        raise
    atomic_write_json(output / "metrics.json", record)
    print(f"passed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
