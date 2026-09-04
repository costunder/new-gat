"""Read-only mean-C validation audits of completed learned-conductance checkpoints.

The source manifest selects either the factorial suite's node_degree arm or the
C-learning suite's learned_c arm. Fixed-C checkpoints are never substituted for a
learned checkpoint, and both suites retain their own strict artifact validation.

This does not retrain, evaluate a test split, or modify source artifacts. Replacing
C by its arithmetic mean within each graph/layer tests reliance of this selected
checkpoint on relative edge weights, not the benefit of learning C from scratch.
For row normalization, positive constant C has the same operator as C=1 (up to
floating-point rounding); the original operator recomputes its weighted degree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from chartgat.cache import atomic_write_bytes, atomic_write_json
from chartgat.observability import RuntimeResourceMonitor, observed

from ..ablation.model import FactorialNodeClassifier, is_gate_parameter, state_sha256
from ..ablation.protocol import COMMON, CONDITIONS, DATASETS, DEFAULT_DATASETS
from ..ablation.report import (
    _build_comparison,
    _contained,
    _integer,
    _load_child,
    _reject_nonfinite_json,
    _same,
)
from ..benchmark import _binary_counts, _micro_f1_from_counts, _versions
from ..benchmark_data import load_dataset, sha256_file
from .model import CLearningNodeClassifier
from .protocol import CONDITIONS as C_LEARNING_CONDITIONS
from .protocol import SUITE as C_LEARNING_SUITE
from .report import _load as _load_c_learning_child
from .report import build_comparison as _build_c_learning_comparison

ROOT = Path(__file__).resolve().parents[3]
BASELINE_TOLERANCE = 1e-4
MODEL_SOURCES = (
    "research/conductance_gat/ablation/model.py",
    "research/conductance_gat/ablation/protocol.py",
    "research/conductance_gat/benchmark.py",
    "research/conductance_gat/sparse.py",
    "research/conductance_gat/benchmark_data.py",
)
C_LEARNING_MODEL_SOURCES = MODEL_SOURCES + (
    "research/conductance_gat/c_learning/model.py",
    "research/conductance_gat/c_learning/protocol.py",
)
AUDIT_SOURCES = C_LEARNING_MODEL_SOURCES + (
    "research/conductance_gat/c_learning/intervene.py",
    "research/conductance_gat/ablation/report.py",
    "research/conductance_gat/c_learning/report.py",
    "src/chartgat/cache.py",
    "src/chartgat/observability.py",
)


def source_spec(suite: str) -> dict[str, Any]:
    """Exact suite dispatch; never infer identity from an arm name or path."""
    if suite == "conductance_factorial":
        return {
            "condition": "node_degree",
            "conditions": CONDITIONS,
            "model_sources": MODEL_SOURCES,
            "build_comparison": _build_comparison,
            "load_child": _load_child,
            "model_factory": FactorialNodeClassifier,
        }
    if suite == C_LEARNING_SUITE:
        return {
            "condition": "learned_c",
            "conditions": C_LEARNING_CONDITIONS,
            "model_sources": C_LEARNING_MODEL_SOURCES,
            "build_comparison": _build_c_learning_comparison,
            "load_child": _load_c_learning_child,
            "model_factory": CLearningNodeClassifier,
        }
    raise ValueError(f"Unsupported checkpoint source suite: {suite!r}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def _assert_unchanged(snapshot: dict[str, str], label: str) -> None:
    if _hashes(Path(path) for path in snapshot) != snapshot:
        raise ValueError(f"{label} changed during the audit; contrasts are invalid")


def validate_source(root: Path, datasets: list[str]) -> tuple[dict, dict, dict]:
    """Validate a complete source matrix without writing or regenerating old reports."""
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Source run must be a directory containing manifest.json")
    if (
        not datasets
        or any(not isinstance(dataset, str) or dataset not in DATASETS for dataset in datasets)
        or len(set(datasets)) != len(datasets)
    ):
        raise ValueError("Requested datasets must be a nonempty unique supported list")
    manifest_path = _contained("manifest.json", root, "source manifest")
    manifest = _read_json(manifest_path)
    specification = source_spec(manifest.get("suite"))
    if manifest.get("source_integrity_valid") is not True:
        raise ValueError("Source manifest must explicitly certify source_integrity_valid=true")
    if not _same(manifest.get("conditions"), specification["conditions"]):
        raise ValueError("Source manifest condition definitions differ from its declared suite")
    comparison = specification["build_comparison"](root, manifest)
    if comparison["status"] != "passed":
        raise ValueError("Source run is not complete and valid: " + str(comparison["errors"]))
    if any(dataset not in manifest["config"]["datasets"] for dataset in datasets):
        raise ValueError("Requested dataset is absent from the source run")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get("sha256"), dict):
        raise ValueError("Source manifest must contain executed source SHA-256 fingerprints")
    historical = sources["sha256"]
    for name in specification["model_sources"]:
        if historical.get(name) != sha256_file(ROOT / name):
            raise ValueError(f"Executed model/cache source differs from historical run: {name}")
    selected = {}
    paths = [manifest_path]
    for job in manifest["jobs"]:
        metrics_path = _contained(job["metrics_path"], root, "source metrics")
        metrics = specification["load_child"](root, job, manifest["config"])
        output = _contained(job["output_dir"], root, "source job")
        paths += [metrics_path]
        paths += [_contained(metrics[key], output, key) for key in ("checkpoint", "history")]
        if job["condition"] == specification["condition"] and job["dataset"] in datasets:
            selected[job["dataset"]] = metrics
    if set(selected) != set(datasets):
        raise ValueError("Source run lacks a unique learned-conductance arm for every dataset")
    return manifest, selected, _hashes(paths)


def validate_checkpoint(saved: dict, metrics: dict) -> None:
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must contain metadata and a state_dict")
    if not isinstance(metrics, dict):
        raise ValueError("Selected metrics must be an object")
    suite = metrics.get("research_suite")
    specification = source_spec(suite)
    required_metrics = {
        "condition": specification["condition"],
        "normalization": "node_degree",
        "gate_weight_decay": COMMON["weight_decay"],
        "non_gate_weight_decay": COMMON["weight_decay"],
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    if suite == C_LEARNING_SUITE:
        required_metrics.update(gate_mode="learned", frozen_parameters=0)
    elif "gate_mode" in metrics or "gate_mode" in saved:
        raise ValueError("Factorial checkpoint must not contain C-learning gate_mode metadata")
    for key, value in required_metrics.items():
        if not _same(metrics.get(key), value):
            raise ValueError(f"Selected checkpoint metrics {key} is not the learned source arm")
    gate_metadata = {"gate_mode": "learned"} if suite == C_LEARNING_SUITE else {}
    expected = {
        "research_suite": suite,
        "model": suite,
        "condition": specification["condition"],
        **gate_metadata,
        "architecture": {
            "hidden_channels": COMMON["hidden_channels"],
            "layers": COMMON["layers"],
            "dropout": COMMON["dropout"],
            "normalization": "node_degree",
            **gate_metadata,
        },
        **{
            key: metrics[key]
            for key in (
                "dataset",
                "model_seed",
                "configuration",
                "gate_weight_decay",
                "non_gate_weight_decay",
                "cache_sha256",
                "initial_state_sha256",
                "best_epoch",
                "validation",
                "evaluation_split",
                "test_evaluated",
            )
        },
        "optimizer_steps": metrics["best_checkpoint_optimizer_steps"],
    }
    if suite == C_LEARNING_SUITE:
        expected["shared_backbone_initial_state_sha256"] = metrics[
            "shared_backbone_initial_state_sha256"
        ]
        for key in (
            "total_parameters",
            "estimator_parameters",
            "non_estimator_parameters",
            "trainable_parameters",
            "frozen_parameters",
        ):
            expected[key] = _integer(
                metrics.get(key), key, minimum=0 if key == "frozen_parameters" else 1
            )
        if expected["total_parameters"] != expected["trainable_parameters"]:
            raise ValueError("Learned-C checkpoint must have zero frozen parameters")
        if expected["total_parameters"] != (
            expected["estimator_parameters"] + expected["non_estimator_parameters"]
        ):
            raise ValueError("Learned-C estimator/non-estimator counts do not sum to total")
    for key, value in expected.items():
        if not _same(saved.get(key), value):
            raise ValueError(f"Checkpoint {key} disagrees with selected source metrics")
    if not isinstance(saved.get("state_dict"), dict) or not saved["state_dict"]:
        raise ValueError("Checkpoint state_dict is missing")


def reconstruct_model(
    saved: dict, metrics: dict, payload: dict, device: torch.device
) -> FactorialNodeClassifier:
    """Restore the declared model exactly, rejecting cross-suite/frozen-gate artifacts."""
    validate_checkpoint(saved, metrics)
    specification = source_spec(metrics["research_suite"])
    model = specification["model_factory"](
        payload["graphs"][0]["x"].shape[1], payload["classes"], **saved["architecture"]
    ).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    if metrics["research_suite"] == C_LEARNING_SUITE:
        counts = {
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "estimator_parameters": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if is_gate_parameter(name)
            ),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "frozen_parameters": sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            ),
        }
        counts["non_estimator_parameters"] = (
            counts["total_parameters"] - counts["estimator_parameters"]
        )
        for key, value in counts.items():
            if not _same(saved[key], value):
                raise ValueError(f"Reconstructed learned-C model {key} differs from checkpoint")
    return model


@contextmanager
def preserved_runtime():
    """Restore RNG and FP32 backend settings even when validation fails."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    dtype, precision = torch.get_default_dtype(), torch.get_float32_matmul_precision()
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        torch.set_default_dtype(dtype)
        torch.set_float32_matmul_precision(precision)
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
        ) = flags


def graphwise_mean(c: Tensor, edge_graph: Tensor, num_graphs: int) -> Tensor:
    """Arithmetic mean PER graph, never pooled across a PPI minibatch."""
    if c.ndim != 1 or c.shape != edge_graph.shape:
        raise ValueError("Expected one conductance and graph id per undirected edge")
    if not bool(torch.isfinite(c).all()) or bool((c <= 0).any()):
        raise ValueError("Conductance must be finite and positive")
    sums = torch.zeros(num_graphs, dtype=torch.float64, device=c.device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, edge_graph, c.double())
    counts.index_add_(0, edge_graph, torch.ones_like(c, dtype=torch.float64))
    means = sums / counts.clamp_min(1)
    return means[edge_graph].to(c.dtype)


class MeanConductance:
    """Temporary estimator-output hooks; use the actual operator's denominator."""

    def __init__(self, model: FactorialNodeClassifier, layers: tuple[int, ...]) -> None:
        if model.normalization != "node_degree":
            raise ValueError("Mean-C audit only supports node_degree normalization")
        if len(set(layers)) != len(layers) or any(
            i not in range(len(model.operators)) for i in layers
        ):
            raise ValueError("Intervention layers must be unique valid layer indices")
        self.model, self.layers = model, layers
        self.handles: list[Any] = []
        self.graphs: dict[int, tuple[Tensor, int]] = {}

    def _before(self, index: int, inputs: tuple) -> None:
        _, incidence, node_graph, *rest = inputs
        tail, head = incidence
        if not torch.equal(node_graph[tail], node_graph[head]):
            raise ValueError("An incidence edge crosses graph boundaries")
        count = int(rest[0]) if rest else int(node_graph.max()) + 1
        self.graphs[index] = (node_graph[tail], count)

    def _replace(self, index: int, c: Tensor) -> Tensor:
        edge_graph, count = self.graphs.pop(index)
        return graphwise_mean(c, edge_graph, count)

    def __enter__(self):
        try:
            for index in self.layers:
                operator = self.model.operators[index]
                self.handles.append(
                    operator.register_forward_pre_hook(
                        lambda module, inputs, i=index: self._before(i, inputs)
                    )
                )
                self.handles.append(
                    operator.estimator.register_forward_hook(
                        lambda module, inputs, output, i=index: self._replace(i, output)
                    )
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.graphs.clear()


def validation_data(
    payload: dict,
    metrics: dict,
    device: torch.device,
    workers: int = 4,
):
    """Construct validation ONLY; full-graph tasks still use all graph features."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if payload["dataset"] != "ppi":
        graph = Data(**payload["graphs"][0]).to(device)
        indices = payload["splits"]["validation"].nonzero(as_tuple=False).flatten().to(device)
        return [graph], indices
    generator = torch.Generator().manual_seed(metrics["model_seed"])
    loader = DataLoader(
        [Data(**payload["graphs"][index]) for index in payload["splits"]["validation"]],
        batch_size=metrics["configuration"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=generator,
    )
    return loader, None


def evaluate(
    model: FactorialNodeClassifier,
    batches: Any,
    indices: Tensor | None,
    device: torch.device,
    layers: tuple[int, ...] = (),
    reference: list[Tensor] | None = None,
) -> tuple[dict, list[Tensor]]:
    """Pure evaluation helper; CPU is used only by bounded unit fixtures."""
    modes = [(module, module.training) for module in model.modules()]
    generator = getattr(batches, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    outputs = []
    counts = torch.zeros(3, dtype=torch.int64, device=device)
    correct = total = changed = logit_count = 0
    abs_sum = square_sum = reference_square = max_delta = 0.0
    try:
        model.eval()
        with preserved_runtime(), torch.no_grad(), MeanConductance(model, layers):
            for batch_index, graph in enumerate(batches):
                graph = graph.to(device, non_blocking=True)
                logits = model(graph)
                # Read only the validation labels for transductive graphs.
                labels = graph.y if indices is None else graph.y.index_select(0, indices)
                if indices is not None:
                    logits = logits.index_select(0, indices)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Non-finite validation logits")
                if indices is None:
                    counts.add_(_binary_counts(logits, labels))
                    total += logits.numel()
                else:
                    correct += int((logits.argmax(-1) == labels).sum())
                    total += labels.numel()
                current = logits.detach().cpu()
                outputs.append(current)
                if reference is not None:
                    if (
                        batch_index >= len(reference)
                        or reference[batch_index].shape != current.shape
                    ):
                        raise ValueError("Validation batch alignment changed between interventions")
                    before = reference[batch_index]
                    old_prediction = before > 0 if indices is None else before.argmax(-1)
                    prediction = current > 0 if indices is None else current.argmax(-1)
                    changed += int((prediction != old_prediction).sum())
                    difference = current.double() - before.double()
                    abs_sum += float(difference.abs().sum())
                    square_sum += float(difference.square().sum())
                    reference_square += float(before.double().square().sum())
                    max_delta = max(max_delta, float(difference.abs().max()))
                    logit_count += current.numel()
            if reference is not None and len(reference) != len(outputs):
                raise ValueError("Validation batch count changed between interventions")
            if not total:
                raise ValueError("Validation contains no labels")
            metric = _micro_f1_from_counts(counts) if indices is None else correct / total
    finally:
        for module, mode in modes:
            module.training = mode
        if generator is not None:
            generator.set_state(loader_rng)
    return {
        "validation": metric,
        "metric_name": "micro_f1" if indices is None else "accuracy",
        "intervened_layers": list(layers),
        "prediction_count": total,
        "prediction_unit": "node-label decision" if indices is None else "node class",
        "graph_batches": len(outputs),
        "changed_predictions": changed if reference is not None else None,
        "changed_prediction_fraction": changed / total if reference is not None else None,
        "logit_mean_absolute_delta": abs_sum / logit_count if logit_count else None,
        "logit_max_absolute_delta": max_delta if reference is not None else None,
        "logit_relative_l2_delta": math.sqrt(square_sum / reference_square)
        if reference_square
        else None,
    }, outputs


def audit_model(model, batches, indices, device, saved_validation: float) -> dict:
    """No contrasts are produced if the original selected score cannot be reproduced."""
    before = state_sha256(model)
    original, reference = evaluate(model, batches, indices, device)
    difference = abs(original["validation"] - saved_validation)
    if not math.isfinite(saved_validation) or difference > BASELINE_TOLERANCE:
        raise ValueError(
            f"Original checkpoint validation mismatch: saved={saved_validation:.9f}, "
            f"rerun={original['validation']:.9f}, tolerance={BASELINE_TOLERANCE}"
        )
    records = []
    cases = [("mean_c_all_layers", tuple(range(len(model.operators))))]
    cases += [(f"mean_c_layer_{i}", (i,)) for i in range(len(model.operators))]
    for name, layers in cases:
        result, _ = evaluate(model, batches, indices, device, layers, reference)
        delta = result["validation"] - original["validation"]
        records.append(
            {"intervention": name, **result, "score_delta": delta, "percentage_points": 100 * delta}
        )
    if state_sha256(model) != before:
        raise ValueError("Model parameters/buffers changed during read-only audit")
    return {
        "original": original,
        "saved_validation": saved_validation,
        "baseline_absolute_error": difference,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "interventions": records,
        "model_state_sha256": before,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Selected-checkpoint mean-C validation audit",
        "",
        f"Status: **{report['status']}**. No training; validation only; test not evaluated.",
        "",
        "Positive graph/layer-constant C cancels under row node-degree normalization, so "
        "this intervention is mathematically equivalent to C=1 at that layer (rounding aside). "
        "It measures reliance of this selected checkpoint, not whether learning C improves "
        "training. All-layer and individual-layer interventions are separate forwards.",
        "",
    ]
    if "source_suite" in report and "source_condition" in report:
        lines += [
            f"Source: `{report['source_suite']}` / `{report['source_condition']}`; "
            "the selected learned-conductance checkpoint from that source run.",
            "",
        ]
    if report.get("error"):
        lines += [f"Audit invalid: {report['error']}", "Contrasts withheld.", ""]
    for item in report["datasets"]:
        lines += [
            f"## {item['dataset']} ({item['metric_name']})",
            "",
            f"Saved validation: {100 * item['saved_validation']:.6f}%; "
            f"original rerun: {100 * item['original']['validation']:.6f}%.",
            "",
            "| Intervention | Validation (%) | Delta original (pp) | Changed predictions (%) "
            "| Mean absolute logit delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in item["interventions"]:
            lines.append(
                f"| {row['intervention']} | {100 * row['validation']:.6f} | "
                f"{row['percentage_points']:+.6f} | "
                f"{100 * row['changed_prediction_fraction']:.6f} | "
                f"{row['logit_mean_absolute_delta']:.6g} |"
            )
        lines.append("")
    lines += [
        "One model seed; no test scores, confidence interval, or significance estimate. "
        "A small metric change does not imply identical logits, and a checkpoint intervention "
        "does not replace the separate learned-C versus fixed-C retraining experiment.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Completed factorial or C-learning run; suite and learned arm come from its manifest",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="PPI validation DataLoader workers; transductive full-graph datasets use no loader",
    )
    return parser


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Checkpoint audit requires a CUDA GPU; no CPU fallback")
    torch.cuda.get_device_properties(device)


def _finish_resource_monitor(
    monitor: RuntimeResourceMonitor,
    device: torch.device,
    primary_error: BaseException | None,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Finish sampling once and retain an earlier audit failure when cleanup fails."""

    collection_notes: list[str] = []
    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if device.type == "cuda":
        try:
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        except BaseException as cleanup_error:
            note = (
                "CUDA allocator peak collection failed during audit cleanup: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
            collection_notes.append(note)
            if primary_error is not None:
                primary_error.add_note(note)
    try:
        resources = monitor.finish(
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except BaseException as cleanup_error:
        reason = (
            "resource monitor cleanup failed"
            + (
                " without replacing the audit error"
                if primary_error is not None
                else ""
            )
            + f": {type(cleanup_error).__name__}: {cleanup_error}"
        )
        if primary_error is not None:
            primary_error.add_note(reason)
        return None, reason, collection_notes
    if collection_notes:
        resources["collection_notes"] = list(collection_notes)
    return resources, None, collection_notes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    _require_cuda(device)
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate datasets are not allowed")
    source = args.source_run.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        (
            args.output_dir
            or ROOT / "results/conductance_gat/c_learning_audits" / f"{source.name}-{stamp}"
        )
        .expanduser()
        .resolve()
    )
    for protected in (source, data_root):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError("Audit output must be separate from source results and dataset cache")
    if output.exists():
        raise FileExistsError(f"Audit output already exists; choose a fresh directory: {output}")
    manifest, selected, source_hashes = validate_source(source, args.datasets)
    specification = source_spec(manifest["suite"])
    audit_hashes = _hashes(ROOT / name for name in AUDIT_SOURCES)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "suite": "conductance_mean_c_audit",
        "status": "running",
        "source_run": str(source),
        "source_suite": manifest["suite"],
        "source_condition": specification["condition"],
        "source_git_revision": manifest["sources"].get("git_revision"),
        "source_manifest_sha256": source_hashes[str(source / "manifest.json")],
        "source_artifact_sha256": source_hashes,
        "executed_audit_source_sha256": audit_hashes,
        "historical_model_source_sha256": {
            name: manifest["sources"]["sha256"][name] for name in specification["model_sources"]
        },
        "n_model_seeds": 1,
        "model_seed": manifest["config"]["model_seed"],
        "training_performed": False,
        "test_evaluated": False,
        "evaluation_split": "validation",
        "interpretation": "selected-checkpoint reliance, NOT benefit of learning C from scratch",
        "dataset_artifact_sha256": {},
        "datasets": [],
        "versions": _versions(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "baseline_tolerance": BASELINE_TOLERANCE,
        "execution_plan": {
            "requested_datasets": list(args.datasets),
            "evaluation_split": "validation",
            "training": False,
            "optimizer_steps": 0,
            "precision": "float32",
            "ppi_physical_batch_size": "validated source-run configuration",
            "ppi_dataloader_workers": args.workers,
            "ppi_persistent_workers": args.workers > 0,
            "ppi_prefetch_factor": 2 if args.workers > 0 else None,
            "transductive_batching": "one complete official graph; no DataLoader",
            "subset_or_fast_mode": False,
        },
    }
    atomic_write_json(output / "audit.json", report)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resource_monitor = RuntimeResourceMonitor(device)
    resource_monitor.start()
    completed_forward_batches = 0
    completed_prediction_decisions = 0
    primary_error: BaseException | None = None
    try:
        with preserved_runtime():
            for dataset in args.datasets:
                print(
                    f"Read-only mean-C audit: {dataset} / {manifest['suite']} / "
                    f"{specification['condition']}",
                    flush=True,
                )
                metrics = selected[dataset]
                payload, protocol = load_dataset(dataset, data_root, allow_download=False)
                if (
                    not _same(protocol, metrics["protocol"])
                    or protocol["data_sha256"] != metrics["cache_sha256"]
                ):
                    raise ValueError(f"{dataset}: dataset cache/protocol differs from source run")
                cache = data_root / "conductance_gat/matched_benchmark_v1" / dataset
                report["dataset_artifact_sha256"].update(
                    _hashes(cache / filename for filename in ("data.pt", "manifest.json"))
                )
                saved = torch.load(metrics["checkpoint"], map_location="cpu", weights_only=True)
                model = reconstruct_model(saved, metrics, payload, device)
                parameter_counts = {
                    "total_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "frozen_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if not parameter.requires_grad
                    ),
                }
                batches, indices = validation_data(
                    payload,
                    metrics,
                    device,
                    args.workers,
                )
                result = audit_model(model, batches, indices, device, metrics["validation"])
                dataset_forward_batches = int(result["original"]["graph_batches"]) + sum(
                    int(row["graph_batches"]) for row in result["interventions"]
                )
                dataset_prediction_decisions = int(
                    result["original"]["prediction_count"]
                ) + sum(int(row["prediction_count"]) for row in result["interventions"])
                completed_forward_batches += dataset_forward_batches
                completed_prediction_decisions += dataset_prediction_decisions
                report["datasets"].append(
                    {
                        "dataset": dataset,
                        "source_suite": manifest["suite"],
                        "source_condition": specification["condition"],
                        "metric_name": metrics["metric_name"],
                        "checkpoint_sha256": metrics["checkpoint_sha256"],
                        "cache_sha256": metrics["cache_sha256"],
                        "model_configuration": metrics["configuration"],
                        "parameter_counts": parameter_counts,
                        "evaluation_work": {
                            "forward_batches": dataset_forward_batches,
                            "prediction_decisions": dataset_prediction_decisions,
                            "physical_batch_size": (
                                1
                                if dataset != "ppi"
                                else metrics["configuration"]["batch_size"]
                            ),
                            "physical_batch_unit": (
                                "complete_transductive_graph"
                                if dataset != "ppi"
                                else "graphs"
                            ),
                            "dataloader_workers": 0 if dataset != "ppi" else args.workers,
                        },
                        **result,
                    }
                )
                _assert_unchanged(source_hashes, "Source artifacts")
                _assert_unchanged(audit_hashes, "Audit/model sources")
                _assert_unchanged(report["dataset_artifact_sha256"], "Dataset artifacts")
                del model, batches, indices, payload, saved
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        primary_error = exc
        report["status"] = "invalid"
        report["error"] = f"{type(exc).__name__}: {exc}"
        # One failed provenance/baseline check invalidates the whole derived report.
        report["datasets"] = []
    resources, unavailable_reason, collection_notes = _finish_resource_monitor(
        resource_monitor,
        device,
        primary_error,
    )
    report["resource_observability"] = resources
    report["resource_observability_unavailable_reason"] = unavailable_reason
    report["resource_observability_collection_notes"] = collection_notes
    if unavailable_reason is not None:
        report["resource_observability_cleanup_error"] = unavailable_reason
        if primary_error is None:
            report["status"] = "invalid"
            report["error"] = f"ResourceObservabilityError: {unavailable_reason}"
    elapsed = (
        resources["summary"]["observed_wall_seconds"]["value"]
        if isinstance(resources, dict)
        else None
    )
    rate_reason = (
        None
        if isinstance(elapsed, (int, float)) and elapsed > 0
        else unavailable_reason
        or "the monitored evaluation interval had no positive wall duration"
    )
    report["throughput"] = {
        "scope": (
            "end-to-end selected-checkpoint validation audit, including official cache reads, "
            "checkpoint reconstruction, baseline reproduction and every mean-C intervention"
        ),
        "completed_forward_batches": completed_forward_batches,
        "completed_prediction_decisions": completed_prediction_decisions,
        "forward_batches_per_second": observed(
            completed_forward_batches / elapsed if rate_reason is None else None,
            reason=rate_reason,
            unit="batches_per_second",
        ),
        "prediction_decisions_per_second": observed(
            completed_prediction_decisions / elapsed if rate_reason is None else None,
            reason=rate_reason,
            unit="decisions_per_second",
        ),
    }
    atomic_write_json(output / "audit.json", report)
    atomic_write_bytes(output / "report.md", _markdown(report).encode("utf-8"))
    print(_markdown(report), flush=True)
    print(f"Reports: {output / 'report.md'}; {output / 'audit.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
