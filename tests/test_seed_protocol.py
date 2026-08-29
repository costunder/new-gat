from __future__ import annotations

import pytest

from chartgat.seeds import SeedAxes, resolve_seed_axes


def test_legacy_seed_fallback_is_explicit() -> None:
    assert resolve_seed_axes(7) == SeedAxes(data=7, split=7, chart=7, model=7)


def test_seed_axes_can_be_varied_independently() -> None:
    axes = resolve_seed_axes(
        99,
        data_seed=1,
        split_seed=2,
        chart_seed=3,
        model_seed=4,
    )
    assert axes.to_manifest() == {"data": 1, "split": 2, "chart": 3, "model": 4}


@pytest.mark.parametrize("field", ("data", "split", "chart", "model"))
def test_negative_seed_axis_is_rejected(field: str) -> None:
    values = {"data": 1, "split": 2, "chart": 3, "model": 4}
    values[field] = -1
    with pytest.raises(ValueError, match=f"{field} seed"):
        SeedAxes(**values)
