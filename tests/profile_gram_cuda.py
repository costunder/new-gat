"""Explicit synthetic CUDA microbenchmark; not arxiv or end-to-end speed evidence."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.aggregation_comparison.gram import GramReadout, LocalGram  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing profiling evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.manual_seed(103)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Explicit synthetic workload; reference K/head/feature widths, not training data.
    history = torch.randn(8, 4096, 8, 32, device="cuda", requires_grad=True)
    c = torch.rand(16384, 8, device="cuda", requires_grad=True)
    readout = torch.randn(8, 36, 32, device="cuda", requires_grad=True)
    edges = torch.randint(0, 4096, (2, 16384), device="cuda")
    upstream = torch.randn(4096, 8, 32, device="cuda")
    records = {}
    for implementation in ("reference", "fused"):
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for repetition in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            if implementation == "reference":
                output = torch.einsum(
                    "nhp,hpd->nhd", LocalGram.apply(history, c, edges, 512, False), readout
                )
            else:
                output = GramReadout.apply(history, c, readout, edges, 512, False)
            gradient = torch.autograd.grad(output, (history, c, readout), upstream)
            end.record()
            end.synchronize()
            if repetition >= 2:
                durations.append(start.elapsed_time(end))
            del output, gradient
        records[implementation] = {
            "forward_backward_milliseconds": durations,
            "median_milliseconds": statistics.median(durations),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    record = {
        "explicit_synthetic_debug": True,
        "scope": __doc__,
        "GPU": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": {"history": list(history.shape), "edges": 16384, "chunk": 512},
        "results": records,
        "end_to_end_speedup_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
