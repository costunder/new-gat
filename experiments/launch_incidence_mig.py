"""Select a MIG on an explicitly requested physical GPU before importing torch.

Read-only NVML APIs: https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlMultiInstanceGPU.html
This environment-only launcher is outside the immutable training source scope.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import math
import os
import re
import subprocess
import sys


class Memory(ct.Structure):
    _fields_ = [(name, ct.c_ulonglong) for name in ("total", "free", "used")]


def nvml_devices(physical_gpu):
    library = ct.CDLL("libnvidia-ml.so.1")
    signatures = {
        "nvmlInit_v2": [],
        "nvmlShutdown": [],
        "nvmlDeviceGetHandleByIndex_v2": [ct.c_uint, ct.POINTER(ct.c_void_p)],
        "nvmlDeviceGetMaxMigDeviceCount": [ct.c_void_p, ct.POINTER(ct.c_uint)],
        "nvmlDeviceGetMigDeviceHandleByIndex": [ct.c_void_p, ct.c_uint, ct.POINTER(ct.c_void_p)],
        "nvmlDeviceGetGpuInstanceId": [ct.c_void_p, ct.POINTER(ct.c_uint)],
        "nvmlDeviceGetComputeInstanceId": [ct.c_void_p, ct.POINTER(ct.c_uint)],
        "nvmlDeviceGetUUID": [ct.c_void_p, ct.POINTER(ct.c_char), ct.c_uint],
        "nvmlDeviceGetMemoryInfo": [ct.c_void_p, ct.POINTER(Memory)],
    }
    for name, signature in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = signature, ct.c_int

    def call(name, *args):
        code = getattr(library, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed with NVML status {code}")

    call("nvmlInit_v2")
    try:
        parent, count = ct.c_void_p(), ct.c_uint()
        call("nvmlDeviceGetHandleByIndex_v2", physical_gpu, ct.byref(parent))
        call("nvmlDeviceGetMaxMigDeviceCount", parent, ct.byref(count))
        records = []
        for index in range(count.value):
            handle = ct.c_void_p()
            code = library.nvmlDeviceGetMigDeviceHandleByIndex(parent, index, ct.byref(handle))
            if code == 6:  # NVML_ERROR_NOT_FOUND: an unpopulated MIG slot.
                continue
            if code:
                raise RuntimeError(f"MIG slot {index}: NVML status {code}")
            gi, ci, memory = ct.c_uint(), ct.c_uint(), Memory()
            uuid = ct.create_string_buffer(128)
            call("nvmlDeviceGetGpuInstanceId", handle, ct.byref(gi))
            call("nvmlDeviceGetComputeInstanceId", handle, ct.byref(ci))
            call("nvmlDeviceGetUUID", handle, uuid, len(uuid))
            call("nvmlDeviceGetMemoryInfo", handle, ct.byref(memory))
            records.append(
                {
                    "physical_gpu": physical_gpu,
                    "mig_index": index,
                    "gpu_instance": gi.value,
                    "compute_instance": ci.value,
                    "uuid": uuid.value.decode("ascii"),
                    "total_bytes": memory.total,
                    "free_bytes": memory.free,
                }
            )
        return records
    finally:
        call("nvmlShutdown")  # Release this process's NVML library handle only.


def select_device(records, *, physical_gpu, gpu_instance, compute_instance, min_free_gb):
    matching = [
        row
        for row in records
        if row["physical_gpu"] == physical_gpu
        and (gpu_instance is None or row["gpu_instance"] == gpu_instance)
        and (compute_instance is None or row["compute_instance"] == compute_instance)
        and 8 * 2**30 <= row["total_bytes"] <= 11 * 2**30
        and row["free_bytes"] >= min_free_gb * 2**30
    ]
    if not matching:
        raise RuntimeError(
            "No requested 10GB MIG has the required free memory; "
            f"observed={json.dumps(records, sort_keys=True)}"
        )
    chosen = max(matching, key=lambda row: (row["free_bytes"], -row["mig_index"]))
    if not re.fullmatch(r"MIG-[A-Za-z0-9/-]+", chosen["uuid"]):
        raise ValueError("NVML returned an invalid MIG UUID")
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-instance", type=int)
    parser.add_argument("--compute-instance", type=int)
    parser.add_argument("--min-free-gb", type=float, default=8)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.training_args
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide the unchanged training arguments after --")
    if (
        any(
            value is not None and value < 0
            for value in (args.physical_gpu, args.gpu_instance, args.compute_instance)
        )
        or not math.isfinite(args.min_free_gb)
        or args.min_free_gb < 0
    ):
        parser.error("device identifiers and minimum free memory must be nonnegative")
    if args.compute_instance is not None and args.gpu_instance is None:
        parser.error("compute-instance requires gpu-instance")
    try:
        chosen = select_device(
            nvml_devices(args.physical_gpu),
            physical_gpu=args.physical_gpu,
            gpu_instance=args.gpu_instance,
            compute_instance=args.compute_instance,
            min_free_gb=args.min_free_gb,
        )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = chosen["uuid"]
        environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
        print(json.dumps({"selected_mig": chosen, "logical_device": "cuda:0"}), flush=True)
        return subprocess.run(
            [sys.executable, "-B", "-m", "experiments.incidence_ablation", *command],
            env=environment,
            check=False,
        ).returncode
    except (OSError, RuntimeError, ValueError) as error:
        print(f"MIG selection stopped safely: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
