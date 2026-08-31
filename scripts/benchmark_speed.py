#!/usr/bin/env python3
"""Measure fixed official training-batch CUDA forward/backward, without an optimizer.

This is an execution microbenchmark, not a paper accuracy experiment. It never
downloads data, evaluates validation/test metrics, or updates model parameters.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

DATASETS = {
    "conductance_gat": ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"),
    "cycle_pe_v2": ("zinc12k", "peptides_struct"),
}
ATOL, RTOL = 2e-5, 2e-4


@dataclass
class SpeedCase:
    batch: Any
    make_model: Callable[[str], Any]
    objective: Callable[[Any], Any]
    protocol: dict[str, Any]
    description: dict[str, Any]
    comparison_scope: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=tuple(DATASETS), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-compile", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    args.dataset = args.dataset or DATASETS[args.track][0]
    if args.dataset not in DATASETS[args.track]:
        raise ValueError(f"{args.track} datasets: {DATASETS[args.track]}")
    if args.steps < 1 or args.warmup < 1 or args.seed < 0:
        raise ValueError("steps/warmup must be positive and seed nonnegative")
    if args.batch_size is None:
        args.batch_size = 2 if args.track == "conductance_gat" else 32
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("Performance measurements require CUDA; no CPU fallback")
    args.data_root = args.data_root.expanduser().resolve()


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


def _build_conductance_case(args: argparse.Namespace, device) -> SpeedCase:
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

    payload, protocol = load_dataset(args.dataset, args.data_root, allow_download=False)
    loader_args = argparse.Namespace(
        model_seed=args.seed, batch_size=args.batch_size, workers=0, pin_memory=True
    )
    data, indices = _make_loaders(payload, loader_args, device)
    if indices is None:
        batch = next(iter(data["train"])).to(device)
        selected = None
        selection = "first seeded/shuffled official training graph minibatch"
    else:
        batch = data
        selected = indices["train"]
        selection = "full official transductive graph; loss uses train indices only"

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
        },
        "Same current classifier; reference restores per-layer GPU graph-count max(). "
        "Both variants use the same indexed loss. Excludes epoch metric accumulation, "
        "loader/transfer, optimizer, checkpoint and validation overhead; NOT a whole-repo speedup.",
    )


def _build_cycle_case(args: argparse.Namespace, device) -> SpeedCase:
    from research.cycle_pe.v2.data import collate, load_benchmark
    from research.cycle_pe.v2.model import CycleBasisPEModel

    splits, protocol = load_benchmark(args.data_root, args.dataset, allow_download=False)
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
        },
        "Same current Cycle PE v2 backbone/parameters; compares reference per-graph "
        "full-basis encoder with bounded batched full-basis encoder. Excludes data "
        "preparation/transfer, optimizer, checkpoint and validation; no basis truncation.",
    )


def _build_case(args: argparse.Namespace, device) -> SpeedCase:
    if args.track == "conductance_gat":
        return _build_conductance_case(args, device)
    return _build_cycle_case(args, device)


def _implementation_hashes(track: str) -> dict[str, str]:
    files = [
        "scripts/benchmark_speed.py",
        "scripts/benchmark_speed.sh",
        "src/chartgat/execution.py",
    ]
    if track == "conductance_gat":
        files.extend(
            f"research/conductance_gat/{name}"
            for name in ("benchmark.py", "benchmark_data.py", "sparse.py")
        )
        hashes = {}
    else:
        from research.cycle_pe.v2.benchmark import implementation_hashes

        hashes = implementation_hashes()
    hashes.update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files})
    return hashes


def _probe(model, case: SpeedCase) -> dict[str, Any]:
    """One eval-mode train-label objective/gradient probe; never a test evaluation."""
    import torch

    model.eval()
    model.zero_grad(set_to_none=True)
    predicted = model(case.batch).float()
    loss = case.objective(predicted)
    if not torch.isfinite(predicted).all() or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite correctness-probe prediction/loss")
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        gradient = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Nonfinite correctness-probe gradient: {name}")
        gradients[name] = gradient
    result = {
        "prediction": predicted.detach().cpu(),
        "loss": loss.detach().cpu(),
        "gradients": gradients,
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
        predicted = model(case.batch).float()
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


def _run_variant(args, device, case, state, variant, reference):
    import torch

    from chartgat.execution import configure_execution

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
    equivalence = _compare_probes(reference or probe, probe)
    model.train()
    _seed(args.seed + 1)
    warmup = _measure_block(model, case, args.warmup, device)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    _seed(args.seed + 2)
    measured = _measure_block(model, case, args.steps, device)
    row = {
        "variant": variant,
        "execution": execution,
        "equivalence": equivalence,
        "eval_probe_seconds_including_lazy_compile": probe_seconds,
        "train_warmup_seconds_including_lazy_compile": warmup["wall_seconds"],
        "warmup_steps_excluded": args.warmup,
        "measured_steps": args.steps,
        **measured,
        "baseline_cuda_allocated_bytes": baseline_bytes,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    row["peak_cuda_incremental_bytes"] = row["peak_cuda_allocated_bytes"] - baseline_bytes
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row, probe


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from chartgat.cache import atomic_write_bytes

    columns = (
        "variant",
        "measured_steps",
        "seconds_per_step",
        "steps_per_second",
        "speedup_vs_reference",
        "cuda_event_seconds",
        "peak_cuda_allocated_bytes",
        "peak_cuda_incremental_bytes",
        "eval_probe_seconds_including_lazy_compile",
        "train_warmup_seconds_including_lazy_compile",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _execute(args, report: dict[str, Any], output: Path) -> None:
    import torch

    from chartgat.cache import atomic_write_json

    device = _require_cuda(args.device)
    report["implementation_sha256"] = _implementation_hashes(args.track)
    # Explicit common FP32 correctness/performance policy; no AMP/TF32 comparison.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report["hardware"] = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    _seed(args.seed)
    case = _build_case(args, device)
    report.update(
        protocol=case.protocol, batch=case.description, comparison_scope=case.comparison_scope
    )
    initial = case.make_model("optimized")
    state = {name: value.detach().cpu().clone() for name, value in initial.state_dict().items()}
    fingerprint = hashlib.sha256()
    for name, value in state.items():
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(value.numpy().tobytes())
    report["initial_state_sha256"] = fingerprint.hexdigest()
    report["trainable_parameters"] = sum(p.numel() for p in initial.parameters() if p.requires_grad)
    del initial
    reference = None
    variants = ["reference", "optimized"] + (["compiled"] if args.include_compile else [])
    print(
        "variant       ms/step     steps/s    ratio vs reference    peak allocated MiB", flush=True
    )
    for variant in variants:
        report["active_variant"] = variant
        atomic_write_json(output / "report.json", report)
        row, probe = _run_variant(args, device, case, state, variant, reference)
        if reference is None:
            reference = probe
        reference_seconds = (
            report["variants"][0]["seconds_per_step"]
            if report["variants"]
            else row["seconds_per_step"]
        )
        row["speedup_vs_reference"] = reference_seconds / row["seconds_per_step"]
        report["variants"].append(row)
        atomic_write_json(output / "report.json", report)
        _write_csv(output / "summary.csv", report["variants"])
        print(
            f"{variant:<12} {1000 * row['seconds_per_step']:>9.3f} "
            f"{row['steps_per_second']:>10.2f} {row['speedup_vs_reference']:>16.3f}x "
            f"{row['peak_cuda_allocated_bytes'] / 2**20:>21.1f}",
            flush=True,
        )
    report.pop("active_variant", None)


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
        "schema_version": 1,
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
            "validation_or_test_metrics": False,
            "precision": "float32; AMP and TF32 disabled",
            "equivalence": "same initial state/batch; eval prediction and every parameter gradient",
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
        },
        "variants": [],
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
