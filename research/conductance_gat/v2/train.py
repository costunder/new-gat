"""Train one direct-edge-C v2 arm on its bound official transductive graph.

The audited train/validation loop is reused explicitly, without replacing its
globals or altering the historical MLP model. Only the new v2 checkpoint is
enriched with its topology, execution settings and source provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from chartgat.cache import atomic_publish, atomic_write_json

from ..ablation import train as shared
from ..ablation.model import is_gate_parameter
from ..benchmark_data import load_dataset, sha256_file, tensor_hash
from .model import DirectCNodeClassifier
from .protocol import (
    COMMON,
    CONDITIONS,
    DATASETS,
    DEFAULT_EDGE_CHUNK_SIZE,
    PARAMETERIZATION,
    PROTOCOL_NOTE,
    SUITE,
)


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {**shared.configuration(args), "edge_chunk_size": args.edge_chunk_size}


def topology_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("dataset") not in DATASETS or len(payload.get("graphs", [])) != 1:
        raise ValueError(
            "Direct edge C requires one fixed transductive graph; "
            "PPI held-out graph transfer is not supported"
        )
    graph = payload["graphs"][0]
    incidence = graph["incidence_edge_index"]
    return {
        "num_nodes": int(graph["x"].shape[0]),
        "num_edges": int(incidence.shape[1]),
        "incidence_sha256": tensor_hash(incidence),
    }


def make_optimizer(model: DirectCNodeClassifier, condition: str) -> torch.optim.Adam:
    spec = CONDITIONS[condition]
    if model.gate_mode != spec["gate_mode"] or model.normalization != "node_degree":
        raise ValueError("Model and direct-C v2 condition disagree")
    gate, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (gate if is_gate_parameter(name) else other).append(parameter)
    if not other or bool(gate) != (model.gate_mode == "direct"):
        raise ValueError("Unexpected trainable parameter groups for direct edge C")
    groups = [{"params": other, "weight_decay": COMMON["weight_decay"], "name": "non_gate"}]
    if gate:
        groups.append({"params": gate, "weight_decay": 0.0, "name": "gate"})
    return torch.optim.Adam(groups, lr=COMMON["lr"])


def edge_gradient_coverage(model: DirectCNodeClassifier) -> list[dict[str, Any]]:
    """Observe actual task gradients before Adam; do not claim all edges learn.

    A graph can have edges outside the training labels' receptive fields. With
    independent edge parameters those entries can receive no learning signal.
    Exact nonzero coverage is a descriptive count, not a significance threshold.
    """
    records = []
    for index, operator in enumerate(model.operators):
        parameter = operator.estimator.log_c
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient.detach()).all()):
            raise FloatingPointError("Nonfinite direct edge-C gradient before optimizer update")
        count = parameter.numel()
        nonzero = int(torch.count_nonzero(gradient.detach())) if gradient is not None else 0
        records.append(
            {
                "layer": index,
                "edge_parameters": count,
                "trainable": parameter.requires_grad,
                "gradient_present": gradient is not None,
                "nonzero_task_gradient_edges": nonzero,
                "nonzero_fraction": nonzero / count if count else None,
            }
        )
    return records


def _source_hashes() -> dict[str, str]:
    # This is a dependency-free source inventory, not a runner invocation.
    from scripts.run_conductance_v2 import _source_snapshot

    return _source_snapshot()["sha256"]


def _validate_args(args: argparse.Namespace) -> None:
    if args.dataset not in DATASETS:
        raise ValueError("Direct C is graph-bound; PPI's unseen graphs are not supported")
    if min(args.epochs, args.patience, args.batch_size, args.edge_chunk_size) < 1:
        raise ValueError("epochs, patience, batch size and edge chunk size must be positive")
    if min(args.workers, args.model_seed) < 0:
        raise ValueError("workers and model seed must be nonnegative")
    if args.batch_size != 1 or args.workers != 0:
        raise ValueError("v2 uses one full transductive graph: batch-size=1 and workers=0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="Full graph only: must be 1")
    parser.add_argument("--workers", type=int, default=0, help="Full graph only: must be 0")
    parser.add_argument("--edge-chunk-size", type=int, default=DEFAULT_EDGE_CHUNK_SIZE)
    return parser


def train_model(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    shared._require_cuda(device)
    _validate_args(args)
    if payload.get("dataset") != args.dataset:
        raise ValueError("Requested dataset differs from the bound graph")
    topology = topology_metadata(payload)
    sources = _source_hashes()
    graph = payload["graphs"][0]
    gradient_history: list[dict[str, Any]] = []

    def bound_model(in_channels: int, classes: int, **kwargs) -> DirectCNodeClassifier:
        return DirectCNodeClassifier(
            in_channels,
            classes,
            incidence=graph["incidence_edge_index"],
            num_nodes=topology["num_nodes"],
            edge_chunk_size=args.edge_chunk_size,
            **kwargs,
        )

    def monitored_optimizer(model: DirectCNodeClassifier, condition: str) -> torch.optim.Adam:
        optimizer = make_optimizer(model, condition)

        def observe(optimizer, positional, keywords):
            gradient_history.append(
                {
                    "optimizer_step": len(gradient_history) + 1,
                    "scope": "full_graph_train_mask",
                    "stage": "after_task_backward_before_optimizer_step",
                    "layers": edge_gradient_coverage(model),
                }
            )

        optimizer.register_step_pre_hook(observe)
        return optimizer

    definition = shared.TrainingDefinition(
        SUITE, CONDITIONS, bound_model, monitored_optimizer, description=__doc__
    )
    result = shared.train_model(payload, protocol, args, device, output, definition=definition)
    if _source_hashes() != sources:
        raise RuntimeError("Direct-C v2 source changed during training; refusing mixed sources")
    checkpoint = Path(result["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # A graph-specific parameter vector must never be read as a transferable
    # MLP checkpoint. The bound incidence is also part of its state_dict/hash.
    saved.update(
        topology=topology,
        parameterization=PARAMETERIZATION,
        configuration=configuration(args),
        source_sha256=sources,
        protocol_note=PROTOCOL_NOTE,
    )
    saved["architecture"]["edge_chunk_size"] = args.edge_chunk_size
    atomic_publish(checkpoint, lambda path: torch.save(saved, path))
    result.update(
        topology=topology,
        parameterization=PARAMETERIZATION,
        configuration=configuration(args),
        source_sha256=sources,
        protocol_note=PROTOCOL_NOTE,
        checkpoint_sha256=sha256_file(checkpoint),
        graph_parameter_count=topology["num_edges"] * COMMON["layers"],
        execution={
            "training": "full_graph_transductive",
            "propagation": "exact_edge_chunked_autograd",
            "edge_chunk_size": args.edge_chunk_size,
            "neighbor_sampling": False,
            "dense_incidence": False,
            "eigendecomposition": False,
            "operator_work": "O(m*d+n*d) per layer; no per-edge MLP",
            "operator_memory": "O(n*d+m+edge_chunk_size*d); excludes backbone/Adam/diagnostics",
            "parameter_storage": "O(m) per layer plus gradient and Adam state",
        },
    )
    result["diagnostics"]["edge_gradient_coverage"] = gradient_history
    result["diagnostics"]["edge_gradient_coverage_policy"] = (
        "Actual train-loss gradient immediately before each Adam step; exact nonzero count, "
        "no epsilon threshold. Full-graph computation does not imply every edge receives "
        "a nonzero gradient. Fixed C is excluded from the optimizer."
    )
    if _source_hashes() != sources:
        raise RuntimeError("Direct-C v2 source changed while publishing its checkpoint")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device = torch.device(args.device)
    shared._require_cuda(device)
    output = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if output == data_root or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("Experiment output and dataset cache must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is not an empty new arm directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "research_suite": SUITE,
        "dataset": args.dataset,
        "condition": args.condition,
        "model_seed": args.model_seed,
        **CONDITIONS[args.condition],
        "non_gate_weight_decay": COMMON["weight_decay"],
        "configuration": configuration(args),
        "parameterization": PARAMETERIZATION,
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    atomic_write_json(output / "metrics.json", record)
    try:
        starting_sources = _source_hashes()
        record["source_sha256"] = starting_sources
        payload, protocol = load_dataset(args.dataset, data_root, allow_download=False)
        if _source_hashes() != starting_sources:
            raise RuntimeError("Direct-C v2 source changed while verifying the dataset cache")
        record.update(cache_sha256=protocol["data_sha256"], protocol=protocol)
        result = train_model(payload, protocol, args, device, output)
        if result["source_sha256"] != starting_sources:
            raise RuntimeError("Dataset preparation and training used different v2 sources")
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
