"""Reporting checks; tensor model/energy arithmetic is covered by CUDA tests."""

import math

import torch

from research.frozen_energy_trace.report import (
    ENERGIES,
    RELATIONS,
    distribution,
    operation_fields,
    retention,
    transition,
)


def test_signed_relation_ratios_use_absolute_magnitude_and_zero_flags():
    before = {key: torch.tensor([0.0, 2.0, 4.0]) for key in ENERGIES}
    after = {key: torch.tensor([3.0, 1.0, 2.0]) for key in ENERGIES}
    for key in RELATIONS:
        before[key] = torch.tensor([-4.0, 0.0, 4.0])
        after[key] = -before[key] / 2
        before[key + "_normalized"] = before[key] / 4
        after[key + "_normalized"] = after[key] / 4
    before["global_E_per_node"] = torch.tensor(3.0)
    after["global_E_per_node"] = torch.tensor(1.5)
    row = transition(before, after, "aggregation")
    assert row["E_operation_ratio"]["count"] == 2
    assert row["E_operation_ratio"]["median"] == 0.5
    assert row["E_undefined_ratio_count"] == 1
    assert math.isclose(row["J_distinct_normalized_mean_abs_ratio"], 0.5, abs_tol=1e-11)
    assert math.isclose(row["J_distinct_retention"]["pattern_cosine"], -1.0, abs_tol=1e-14)
    assert row["global_E_per_node_operation_ratio"] == 0.5
    fields = operation_fields(before, after)
    assert fields["E_ratio_defined"].tolist() == [False, True, True]
    assert math.isnan(float(fields["E_ratio"][0]))
    assert fields["E_ratio"][1:].tolist() == [0.5, 0.5]


def test_empty_and_zero_reference_do_not_become_success_or_zero_metric():
    assert distribution(torch.empty(0))["median"] is None
    result = retention(torch.ones(3), torch.zeros(3))
    assert result["magnitude_ratio"] is None
    assert result["pattern_cosine"] is None
    result = retention(torch.zeros(3), torch.ones(3))
    assert result["magnitude_ratio"] == 0
    assert result["pattern_cosine"] is None
