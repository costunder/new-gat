"""Measured allocation policies; scientific profiles remain unchanged.

The MIG policy changes execution chunks and memory headroom only. CUDA reports
the assigned instance's capacity; the parent GPU name is never a memory budget.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass

import torch

HARDWARE_PROFILES = ("auto", "a100-mig-10gb")
_GIB = 1 << 30


def get_policy(name="auto"):
    if name not in HARDWARE_PROFILES:
        raise ValueError(f"Unknown hardware allocation profile: {name}")
    mig = name == "a100-mig-10gb"
    return {
        "name": name,
        "memory_safety_fraction": .70 if mig else .75,
        "reserve_bytes": _GIB if mig else 0,
        "extra_pair_chunks": [128, 512] if mig else [],
        "extra_channel_chunks": [1, 2, 4] if mig else [],
        "extra_graph_batches": [2, 4] if mig else [],
        "scientific_scale_changed": False,
    }


def configure_classification_runtime(config, name="auto"):
    """Modify only fields already declared mutable by C's contract validator."""
    updated = deepcopy(config)
    policy = get_policy(name)
    if name != "auto":
        runtime = updated["runtime"]
        runtime["gpu_memory_safety_fraction"] = policy["memory_safety_fraction"]
        runtime["relation_chunk_candidates"] = sorted(set(
            policy["extra_pair_chunks"] + runtime["relation_chunk_candidates"]
        ))
    return updated


def memory_window(device, policy=None):
    """Snapshot additional allocatable bytes after live inputs/model residency.

    Caller should release disposable tensors and empty its CUDA allocator cache
    before comparing independent candidates. Both allocator peaks are checked:
    reserved memory can exceed live tensor memory due to fragmentation.
    """
    policy = get_policy() if policy is None else policy
    device = torch.device(device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    else:
        import psutil
        ram = psutil.virtual_memory()
        free, total, allocated, reserved = ram.available, ram.total, 0, 0
    reserve = int(policy["reserve_bytes"])
    budget = max(0, int(min(free * policy["memory_safety_fraction"], free - reserve)))
    return {
        "device": str(device), "hardware_profile": policy["name"],
        "free_bytes": int(free), "total_bytes": int(total),
        "allocated_bytes": int(allocated), "reserved_bytes": int(reserved),
        "reserve_bytes": reserve, "allocation_budget_bytes": budget,
        "allocated_ceiling_bytes": int(allocated) + budget,
        "reserved_ceiling_bytes": int(reserved) + budget,
        "memory_safety_fraction": policy["memory_safety_fraction"],
    }


def headroom_safe(peak_allocated, peak_reserved, before):
    return (
        int(peak_allocated) <= before["allocated_ceiling_bytes"]
        and int(peak_reserved) <= before["reserved_ceiling_bytes"]
    )


def unique_tensor_bytes(value):
    """Count shared tensor storages once, including dataclass graph containers."""
    objects, storages = set(), set()

    def visit(item):
        if id(item) in objects:
            return 0
        objects.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in storages:
                return 0
            storages.add(key)
            return storage.nbytes()
        if is_dataclass(item) and not isinstance(item, type):
            return sum(visit(getattr(item, field.name)) for field in fields(item))
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(v) for v in item)
        return 0

    return visit(value)
