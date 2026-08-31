"""Independent incidence-conductance-attention research track."""

from importlib import import_module
from typing import Any

__all__ = [
    "IncidenceConductanceAttention",
    "IsotropicConductanceAttention",
    "PositiveInvariantScalarConductance",
    "PackedGraphBatch",
    "SparseIncidenceConductanceLayer",
    "SparsePositiveConductance",
    "edge_divergence",
    "edge_gradient",
    "pack_graph_examples",
]


def __getattr__(name: str) -> Any:
    # Planning/reporting CLIs need only stdlib; preserve the public model API lazily.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".model" if name in __all__[:3] else ".sparse"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
