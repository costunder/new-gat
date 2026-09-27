"""Explicit CUDA allocator budget, independent of scientific model/data sizes."""

import math

import torch


def add_argument(parser):
    parser.add_argument(
        "--cuda-allocator-limit-gib",
        type=float,
        help="cap PyTorch CUDA allocations/reservations; never resize model, graph or batch",
    )


def validate(args):
    value = getattr(args, "cuda_allocator_limit_gib", None)
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError("cuda-allocator-limit-gib must be finite and positive")
    return value


def configure(args, device):
    value = validate(args)
    if value is None:
        return None
    if torch.device(device).type != "cuda":
        raise ValueError("CUDA allocator budget requires CUDA")
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise ValueError("the measured CUDA allocator limit requires the native allocator")
    total = torch.cuda.get_device_properties(device).total_memory
    limit = int(value * 1024**3)
    if not 0 < limit <= total:
        raise ValueError("CUDA allocator limit exceeds actual visible device capacity")
    if torch.cuda.memory_allocated(device) > limit:
        raise RuntimeError(
            "existing live CUDA tensors already exceed the requested allocator limit"
        )
    torch.cuda.set_per_process_memory_fraction(limit / total, device)
    return limit
