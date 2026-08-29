"""Explicit random-seed axes for reproducible graph experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SeedAxes:
    """Keep benchmark construction and estimator randomness independently auditable."""

    data: int
    split: int
    chart: int
    model: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} seed must be a non-negative integer")

    def to_manifest(self) -> dict[str, int]:
        return asdict(self)


def resolve_seed_axes(
    legacy_seed: int,
    *,
    data_seed: int | None = None,
    split_seed: int | None = None,
    chart_seed: int | None = None,
    model_seed: int | None = None,
) -> SeedAxes:
    """Resolve new independent axes while preserving standalone ``--seed`` compatibility.

    A missing data seed falls back to the legacy seed.  Split and chart seeds then
    default to that data seed, while the model seed defaults directly to the legacy
    seed.  The master paper runner passes every axis explicitly and therefore never
    relies on these compatibility fallbacks.
    """

    data = legacy_seed if data_seed is None else data_seed
    return SeedAxes(
        data=data,
        split=data if split_seed is None else split_seed,
        chart=data if chart_seed is None else chart_seed,
        model=legacy_seed if model_seed is None else model_seed,
    )


__all__ = ["SeedAxes", "resolve_seed_axes"]
