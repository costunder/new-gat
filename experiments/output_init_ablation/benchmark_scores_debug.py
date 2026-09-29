"""Synthetic component benchmark; no dataset, optimizer, or model training."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from experiments.c_learning_bracket.conductance import BracketConductance

from .log_row import ChunkedSymmetricScores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    # Dimensions from local calibration timing-learned-2048-4-1.json.
    # Edges and Q/K are synthetic, not that real training batch.
    nodes, edge_count, heads, width, chunk = 53248, 379398, 8, 32, 16384
    torch.manual_seed(191)
    query = torch.randn(nodes, heads, width, device="cuda", requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    edges = torch.randint(nodes, (2, edge_count), device="cuda")
    probe = torch.randn(edge_count, heads, device="cuda")

    def reference():
        return torch.cat(
            [
                checkpoint(
                    BracketConductance.edge_score,
                    query,
                    key,
                    edges[:, start : start + chunk],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                for start in range(0, edge_count, chunk)
            ]
        )

    def optimized():
        return ChunkedSymmetricScores.apply(query, key, edges, chunk)

    expected, actual = reference(), optimized()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    old_grad = torch.autograd.grad(expected, (query, key), probe)
    new_grad = torch.autograd.grad(actual, (query, key), probe)
    errors = []
    for before, after in zip(old_grad, new_grad, strict=True):
        torch.testing.assert_close(after, before, rtol=1e-4, atol=2e-7)
        errors.append(float((after - before).abs().max()))
    del expected, actual, old_grad, new_grad

    def measure(function):
        query.grad = key.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = function()
        torch.cuda.synchronize()
        forward_done = time.perf_counter()
        output.backward(probe)
        torch.cuda.synchronize()
        end = time.perf_counter()
        return {
            "forward_seconds": forward_done - start,
            "backward_seconds": end - forward_done,
            "total_seconds": end - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }

    functions = {"checkpoint_reference": reference, "accumulate_once": optimized}
    for function in functions.values():
        measure(function)
    samples = {name: [] for name in functions}
    # Alternate order so a fixed ordering does not systematically favor one.
    for repeat in range(6):
        order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
        for name in order:
            samples[name].append(measure(functions[name]))
    result = {
        "scope": "synthetic Q/K edge-score forward/backward only; NOT full training",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": str(query.dtype),
        "nodes": nodes,
        "edges": edge_count,
        "heads": heads,
        "head_width": width,
        "chunk": chunk,
        "gradient_max_absolute_errors": errors,
        "samples": samples,
        "medians": {
            name: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
            for name, rows in samples.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
