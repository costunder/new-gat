"""Shared primitives; stdlib-only cache/CLI use must not import NumPy eagerly."""

from importlib import import_module
from typing import Any

__all__ = [
    "chart_transition",
    "decode_edge_state",
    "encode_edge_state",
    "flip_cycle_basis",
    "flip_edge_quantity",
    "flip_incidence",
    "fundamental_cycle_basis",
    "incidence_matrix",
    "orthonormal_cycle_basis",
    "validate_spanning_tree",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".algebra", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
