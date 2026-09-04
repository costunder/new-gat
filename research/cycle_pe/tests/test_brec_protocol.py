from __future__ import annotations

import pytest
import torch

from research.cycle_pe.paper import (
    BREC_OFFICIAL_BATCH_SIZE,
    BREC_OFFICIAL_SEEDS,
    _aggregate_custom_brec_results,
    _aggregate_official_brec_results,
    _brec_batches,
    _brec_reference_compatibility,
    _brec_settings,
    _effective_brec_protocol,
    brec_hotelling_t2,
    brec_rpc_decision,
    build_parser,
)


def test_hotelling_t2_matches_official_torch_reference_without_q_multiplier() -> None:
    generator = torch.Generator().manual_seed(919)
    embeddings = torch.randn((64, 16), generator=generator)
    difference = embeddings[0::2].T - embeddings[1::2].T
    mean = torch.mean(difference, dim=1).reshape(-1, 1)
    expected = (mean.T @ torch.linalg.pinv(torch.cov(difference)) @ mean).reshape(())

    actual = brec_hotelling_t2(embeddings)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert not torch.isclose(actual, expected * 32)


def test_rpc_decision_uses_official_isclose_and_reliability_gate() -> None:
    # Default torch rtol is intentionally retained by the official code.
    close = brec_rpc_decision(100.0, 100.0009, threshold=72.34)
    assert close == {"distinguished": False, "reliable": False, "successful": False}

    unreliable = brec_rpc_decision(100.0, 80.0, threshold=72.34)
    assert unreliable == {"distinguished": True, "reliable": False, "successful": False}

    success = brec_rpc_decision(100.0, 1.0, threshold=72.34)
    assert success == {"distinguished": True, "reliable": True, "successful": True}


def test_custom_pairwise_union_excludes_any_pair_with_reliability_failure() -> None:
    seeds = (100, 200)
    results = [
        {
            "pair_index": 0,
            "category": "Basic",
            "search_seed": 100,
            "status": "complete",
            "distinguished": True,
            "reliable": True,
            "successful": True,
        },
        {
            "pair_index": 0,
            "category": "Basic",
            "search_seed": 200,
            "status": "complete",
            "distinguished": False,
            "reliable": True,
            "successful": False,
        },
        {
            "pair_index": 1,
            "category": "Basic",
            "search_seed": 100,
            "status": "complete",
            "distinguished": True,
            "reliable": True,
            "successful": True,
        },
        {
            "pair_index": 1,
            "category": "Basic",
            "search_seed": 200,
            "status": "complete",
            "distinguished": True,
            "reliable": False,
            "successful": False,
        },
    ]
    summary = _aggregate_custom_brec_results(results, pair_indices=[0, 1], seeds=seeds)
    assert summary["protocol"] == "custom"
    assert summary["metric_name"] == "custom_pairwise_union"
    assert summary["successful_pairs"] == 1
    assert summary["reliability_failures"] == 1
    assert summary["per_pair"][0]["successful_pair"] is True
    assert summary["per_pair"][1]["successful_pair"] is False


def test_cli_defaults_to_the_official_ten_search_seeds() -> None:
    args = build_parser().parse_args([])
    assert tuple(int(value) for value in args.brec_seeds.split(",")) == BREC_OFFICIAL_SEEDS


def test_official_aggregation_reports_each_seed_without_union() -> None:
    seeds = (100, 200)
    results = [
        {
            "pair_index": pair_index,
            "category": "Basic",
            "search_seed": seed,
            "status": "complete",
            "distinguished": distinguished,
            "reliable": reliable,
            "successful": distinguished and reliable,
        }
        for seed, decisions in (
            (100, ((True, True), (False, True))),
            (200, ((True, True), (True, False))),
        )
        for pair_index, (distinguished, reliable) in enumerate(decisions)
    ]
    summary = _aggregate_official_brec_results(results, pair_indices=[0, 1], seeds=seeds)
    assert summary["protocol"] == "official"
    assert summary["merged_score"] is None
    assert summary["global_valid"] is False
    assert "repository-defined" in summary["global_valid_definition"]
    assert "not an upstream BREC metric" in summary["global_valid_definition"]
    assert summary["per_seed"]["100"]["Correct"] == 1
    assert summary["per_seed"]["100"]["Fail"] == 0
    assert summary["per_seed"]["100"]["Real_correct"] == 1
    assert summary["per_seed"]["200"]["Correct"] == 2
    assert summary["per_seed"]["200"]["Fail"] == 1
    assert summary["per_seed"]["200"]["Real_correct"] == 1


def test_official_mode_resolves_for_full_runs_and_forces_reference_settings() -> None:
    full = build_parser().parse_args(["--suite", "brec", "--batch-size", "99", "--amp"])
    full.brec_protocol = _effective_brec_protocol(full)
    assert full.brec_protocol == "official"
    settings = _brec_settings(full, torch.device("cpu"), full.brec_protocol)
    assert settings.batch_size == BREC_OFFICIAL_BATCH_SIZE == 16
    assert settings.epochs == 20
    assert settings.learning_rate == 1e-4
    assert settings.weight_decay == 1e-4
    assert settings.amp_requested is False
    assert settings.pin_memory_requested is False

    custom = build_parser().parse_args(["--suite", "brec", "--brec-protocol", "custom"])
    assert _effective_brec_protocol(custom) == "custom"


def test_official_reference_compatibility_does_not_claim_differential_parity() -> None:
    compatibility = _brec_reference_compatibility("official")
    assert compatibility["static_constants_and_control_flow_compatible"] is True
    assert compatibility["differential_parity_verified"] is False
    assert "must not be interpreted" in compatibility["parity_note"]

    custom = _brec_reference_compatibility("custom")
    assert custom["static_constants_and_control_flow_compatible"] is False
    assert custom["differential_parity_verified"] is False


def test_brec_batch_size_must_preserve_complete_pairs_without_silent_rounding() -> None:
    with pytest.raises(ValueError, match="even integer"):
        _brec_batches([], torch.arange(0).numpy(), batch_size=1)
    with pytest.raises(ValueError, match="even integer"):
        _brec_batches([], torch.arange(0).numpy(), batch_size=3)
