"""Execution-only options shared by independent research tracks.

Compilation is opt-in: it changes execution, not parameters or the optimizer.
No custom extension, system compiler installation, precision change or silent
fallback is performed here. Imports stay lazy so CLI help needs no Torch import.
"""

from __future__ import annotations

import argparse
from typing import Any


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile tensor MLP blocks with Inductor (CUDA); first calls include compile cost.",
    )


def configure_execution(model: Any, args: argparse.Namespace, device: Any) -> dict[str, Any]:
    """Compile tensor-only Sequential blocks, preserving ordinary checkpoint keys.

    Do not replace the model by an OptimizedModule wrapper: existing checkpoints
    must remain loadable by the eager implementation and vice versa. Keep ragged
    graph/column scheduling outside Dynamo: tracing those Python loops specializes
    on every graph's cycle rank and can exhaust the recompilation cache quickly.
    Compiler errors, including errors on the first lazy invocation, propagate.
    """
    enabled = bool(getattr(args, "compile", False))
    metadata = {
        "torch_compile": enabled,
        "backend": "inductor" if enabled else "eager",
        "dynamic_shapes": enabled,
        "scope": "tensor_mlp_blocks" if enabled else "eager",
        "compiled_modules": [],
        "precision_changed_by_execution_option": False,
        "checkpoint_format": "ordinary_module_state_dict",
    }
    if not enabled:
        return metadata
    import torch
    import torch._dynamo

    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("--compile requires CUDA; no CPU research fallback")
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("--compile requires Dynamo suppress_errors=False; no silent fallback")
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Sequential)
        and not any(isinstance(child, torch.nn.Sequential) for child in module.children())
    ]
    if not targets:
        raise RuntimeError("--compile found no tensor MLP blocks in this model")
    if not callable(getattr(torch, "compile", None)):
        raise RuntimeError("This PyTorch build does not support torch.compile")
    for name, module in targets:
        # Target the bound forward explicitly. Some Torch releases skip a
        # Sequential's generic _call_impl when Module.compile wraps it, leaving
        # no compiled frames. Keep Module.__call__ (and its hooks) unchanged.
        module.forward = torch.compile(module.forward, backend="inductor", dynamic=True)
        metadata["compiled_modules"].append(name or "<root>")
    return metadata
