"""Real-data CUDA calibration probe; allocator cap is not A100 MIG emulation.

No final training, test evaluation, model/data downsizing, or checkpoint export.
Use a new output path for every measurement. Dataset must already be prepared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import torch  # noqa: E402

from experiments.aggregation_comparison import core, engine, provenance, runner  # noqa: E402
from research.conductance_gat.v5.protocol import HARDWARE_PROFILES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=core.ARMS, required=True)
    parser.add_argument("--hardware-profile", choices=HARDWARE_PROFILES, default="portable")
    parser.add_argument("--sampling", choices=("full", "cluster_disjoint"), default="full")
    parser.add_argument("--context-seeds", type=int, default=2048)
    parser.add_argument("--physical-seeds", type=int, default=2048)
    parser.add_argument("--edge-chunk-size", type=int, default=4096)
    parser.add_argument("--allocator-limit-gib", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving previous evidence: {args.output}")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU model fallback")
    total = torch.cuda.get_device_properties(device).total_memory
    cap = int(args.allocator_limit_gib * 1024**3)
    if not 0 < cap <= total:
        raise ValueError("allocator cap must be positive and fit the actual visible device")
    options = [
        "--run-id",
        "debug-real-arxiv-memory-probe",
        "--profiles",
        "reference",
        "--arms",
        args.arm,
        "--datasets",
        "ogbn-arxiv",
        "--model-seeds",
        "0",
        "--device",
        "cuda:0",
        "--hardware-profile",
        args.hardware_profile,
        "--cuda-allocator-limit-gib",
        str(args.allocator_limit_gib),
        "--learning-budget-policy",
        "epochs",
        "--complete-supervised-passes",
        "--activation-checkpoint",
        "--edge-chunk-size",
        str(args.edge_chunk_size),
        "--sampling",
        args.sampling,
    ]
    if args.sampling != "full":
        options += [
            "--sample-context-seed-batch-size",
            str(args.context_seeds),
            "--sample-seed-batch-size",
            str(args.physical_seeds),
        ]
    control = runner.parser().parse_args(options)
    runner.validate_args(control)
    command = runner.make_jobs(control, args.output.parent)[0]["command"]
    training = engine.build_parser().parse_args(command[command.index("-m") + 2 :])
    engine.validate_args(training)
    result = {
        "scope": "real arxiv complete calibration; not final training or MIG speed/fit proof",
        "actual_gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "visible_memory_bytes": total,
        "allocator_limit_bytes": cap,
        "allocator_environment": {
            name: os.environ.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        },
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "source_sha256": provenance.source_snapshot(),
        "configuration": engine.configuration(training),
        "events": [],
    }
    started = time.perf_counter()

    def persist():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def mark(stage):
        result["last_stage"] = stage
        event = {
            "stage": stage,
            "seconds": time.perf_counter() - started,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        result["events"].append(event)
        persist()
        print(json.dumps(event), flush=True)

    original_train, original_evaluate = engine.run_training_epoch, engine.evaluate
    epoch_count = 0

    def train(*values, **kwargs):
        nonlocal epoch_count
        epoch_count += 1
        mark(f"train_epoch_{epoch_count}_start")
        report = original_train(*values, **kwargs)
        mark(f"train_epoch_{epoch_count}_end")
        return report

    def evaluate(*values, **kwargs):
        stage = "diagnostic" if values[0].diagnostic_collector is not None else "validation"
        mark(stage + "_start")
        report = original_evaluate(*values, **kwargs)
        mark(stage + "_end")
        return report

    engine.run_training_epoch, engine.evaluate = train, evaluate
    try:
        mark("official_data_load")
        payload, protocol = engine.load_dataset(training)
        result["dataset_protocol"] = protocol
        mark("calibration_start")
        result["measurement"] = engine.run_calibration_candidate(
            {"payload": payload, "protocol": protocol},
            training,
            device,
            physical_batch_size=1 if args.sampling == "full" else args.physical_seeds,
            workers=0 if args.sampling == "full" else training.sample_context_workers,
        )
        if result["source_sha256"] != provenance.source_snapshot():
            raise RuntimeError("source changed during calibration; do not use as final evidence")
        result["source_unchanged_during_measurement"] = True
        result["status"] = "passed"
        peak = result["measurement"]["peak_reserved_bytes"]
        result["cap_minus_peak_reserved_bytes"] = cap - peak
        result["minimum_free_bytes_for_two_gib_headroom"] = peak + 2 * 1024**3
        mark("calibration_complete")
    except Exception as error:
        result.update(
            status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        result["failure_resources"] = getattr(error, "calibration_resource_observability", None)
        mark("failed_after_" + result["last_stage"])
        print(result["error"], flush=True)
        return 1
    finally:
        engine.run_training_epoch, engine.evaluate = original_train, original_evaluate
        persist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
