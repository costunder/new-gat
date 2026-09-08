"""CPU-only unit tests of audit provenance/CLI contracts, not a GPU experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import audit_v5_stages as audit


def _write_json(path: Path, value) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return audit._sha(path)


@pytest.fixture
def evidence(tmp_path):
    folder = tmp_path / "v5/reference/model-seed-0/cora/shared_dynamic_c"
    folder.mkdir(parents=True)
    checkpoint = folder / "best.pt"
    checkpoint.write_bytes(b"explicit unit fixture; not a training checkpoint")
    config = {
        "conductance_backend": "optimization",
        "training_schedule": "joint",
        "solver_cost_scaling": "width_scaled",
        "beta_parameterization": "sigmoid",
        "beta_initial": 0.5,
        "hidden_channels": 256,
        "layers": 8,
        "heads": 8,
        "ffn_multiplier": 4,
        "solver_steps": 8,
        "batch_size": 1,
    }
    protocol = {
        "data_sha256": "a" * 64,
        "split_sha256": {"train": "b" * 64, "validation": "c" * 64, "test": "d" * 64},
    }
    identity = {
        "research_suite": audit.SUITE,
        "dataset": "cora",
        "condition": "shared_dynamic_c",
        "configuration": config,
        "dataset_protocol": protocol,
        "dataset_protocol_sha256": audit._canonical(protocol),
        "cache_sha256": protocol["data_sha256"],
        "source_sha256": audit.BASELINE_SOURCES,
    }
    rows = [{"epoch": 1, "validation": 0.6}, {"epoch": 2, "validation": 0.7}]
    history_sha = _write_json(folder / "history.json", rows)
    metrics = {
        "schema_version": 1,
        "research_suite": audit.SUITE,
        "status": "passed",
        "dataset": "cora",
        "condition": "shared_dynamic_c",
        "configuration": config,
        "protocol": protocol,
        "resume_identity": identity,
        "resume_identity_sha256": audit._canonical(identity),
        "cache_sha256": protocol["data_sha256"],
        "source_sha256": audit.BASELINE_SOURCES,
        "best_epoch": 2,
        "validation": 0.7,
        "checkpoint_sha256": audit._sha(checkpoint),
        "history_sha256": history_sha,
        "checkpoint_selection": {"primary_epoch": 2, "primary_validation": 0.7, "test_used": False},
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    path = folder / "metrics.json"
    _write_json(path, metrics)
    return path, metrics


def _args(root: Path):
    return ["--root", str(root), "--reference-steps", "64", "--reference-tolerance", "0.001"]


def test_reference_budget_and_tolerance_are_explicit(tmp_path):
    parser = audit.build_parser()
    for missing in (
        ["--root", str(tmp_path)],
        ["--root", str(tmp_path), "--reference-steps", "64"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(missing)
    args = parser.parse_args(_args(tmp_path))
    assert args.device == "cuda:0"
    assert args.reference_steps == 64
    assert args.reference_tolerance == 0.001
    assert args.json is False
    assert args.repeat_evaluations == 5
    assert args.head_gradient_conflict is False
    explicit = parser.parse_args(
        [*_args(tmp_path), "--repeat-evaluations", "6", "--head-gradient-conflict"]
    )
    assert explicit.repeat_evaluations == 6 and explicit.head_gradient_conflict is True


@pytest.mark.parametrize(
    "option,value",
    [
        ("--reference-steps", "0"),
        ("--reference-steps", "-1"),
        ("--reference-tolerance", "0"),
        ("--reference-tolerance", "nan"),
        ("--reference-tolerance", "inf"),
        ("--repeat-evaluations", "4"),
    ],
)
def test_invalid_reference_arguments(tmp_path, option, value):
    args = _args(tmp_path)
    if option in args:
        args[args.index(option) + 1] = value
    else:
        args.extend([option, value])
    with pytest.raises(SystemExit):
        audit.build_parser().parse_args(args)


def test_discovery_and_inspection_are_read_only(evidence, tmp_path):
    path, metrics = evidence
    before = {p: audit._sha(p) for p in path.parent.iterdir()}
    assert audit.discover_conditions(tmp_path) == [path.resolve()]
    inspected = audit.inspect_evidence(path)
    assert inspected["metrics"] == metrics
    assert inspected["history"][-1]["epoch"] == 2
    audit.verify_unchanged(inspected["artifacts"])
    assert {p: audit._sha(p) for p in path.parent.iterdir()} == before


def test_missing_root_and_missing_results(tmp_path):
    with pytest.raises(audit.AuditError, match="existing"):
        audit.discover_conditions(tmp_path / "missing")
    with pytest.raises(audit.AuditError, match="No V5"):
        audit.discover_conditions(tmp_path)


@pytest.mark.parametrize(
    "change",
    ["checkpoint", "history", "identity", "selection", "test", "config", "protocol", "source"],
)
def test_rejects_inconsistent_or_mutated_evidence(evidence, change):
    path, metrics = evidence
    if change == "checkpoint":
        (path.parent / "best.pt").write_bytes(b"changed")
    elif change == "history":
        _write_json(path.parent / "history.json", [{"epoch": 2, "validation": 0.1}])
    elif change == "identity":
        metrics["resume_identity_sha256"] = "f" * 64
    elif change == "selection":
        metrics["checkpoint_selection"]["primary_epoch"] = 1
    elif change == "test":
        metrics["test_evaluated"] = True
    elif change == "config":
        metrics["configuration"] = {**metrics["configuration"], "layers": 1}
    elif change == "protocol":
        metrics["protocol"] = {**metrics["protocol"], "data_sha256": "f" * 64}
    elif change == "source":
        metrics["source_sha256"] = {"bad": "f" * 64}
    _write_json(path, metrics)
    with pytest.raises(audit.AuditError):
        audit.inspect_evidence(path)


def test_rejects_uncorrected_recipe_even_with_valid_identity(evidence):
    path, metrics = evidence
    metrics["configuration"]["solver_cost_scaling"] = "legacy_unit"
    metrics["resume_identity_sha256"] = audit._canonical(metrics["resume_identity"])
    _write_json(path, metrics)
    with pytest.raises(audit.AuditError, match="explicitly corrected"):
        audit.inspect_evidence(path)


def test_rejects_history_selection_mismatch_even_with_valid_history_hash(evidence):
    path, metrics = evidence
    metrics["history_sha256"] = _write_json(
        path.parent / "history.json", [{"epoch": 2, "validation": 0.5}]
    )
    _write_json(path, metrics)
    with pytest.raises(audit.AuditError, match="Selected epoch"):
        audit.inspect_evidence(path)


def test_recorded_diagnostics_do_not_invent_missing_overlap_or_gradient(evidence):
    path, _ = evidence
    inspected = audit.inspect_evidence(path)
    result = audit.historical_diagnostics(inspected["metrics"], inspected["history"])
    assert result["sampling_observations"]["value"] is None
    assert "not recorded" in result["sampling_observations"]["reason"]
    assert result["first_active_conductance_gradient"] == "unavailable"


def test_explicit_mlp_control_is_auditable_but_not_mislabeled_optimized(evidence):
    path, metrics = evidence
    metrics["configuration"]["conductance_backend"] = "mlp"
    metrics["configuration"]["solver_cost_scaling"] = "legacy_unit"
    metrics["resume_identity_sha256"] = audit._canonical(metrics["resume_identity"])
    _write_json(path, metrics)
    assert audit.inspect_evidence(path)["metrics"]["configuration"]["conductance_backend"] == "mlp"
    metrics["configuration"]["conductance_heads"] = "per_head"
    metrics["resume_identity_sha256"] = audit._canonical(metrics["resume_identity"])
    _write_json(path, metrics)
    with pytest.raises(audit.AuditError, match="shared untyped MLP"):
        audit.inspect_evidence(path)


def test_beta_override_requires_exact_current_release_not_historical_relaxation(evidence):
    path, metrics = evidence
    metrics["configuration"]["beta_initial"] = 0.3
    metrics["resume_identity_sha256"] = audit._canonical(metrics["resume_identity"])
    _write_json(path, metrics)
    with pytest.raises(audit.AuditError, match="explicitly corrected"):
        audit.inspect_evidence(path)
    metrics["source_sha256"] = {**audit.BASELINE_SOURCES, **audit.AUDIT_SOURCE_OVERRIDES}
    metrics["resume_identity"]["source_sha256"] = metrics["source_sha256"]
    metrics["resume_identity_sha256"] = audit._canonical(metrics["resume_identity"])
    _write_json(path, metrics)
    assert audit.inspect_evidence(path)["metrics"]["configuration"]["beta_initial"] == 0.3


def test_human_distribution_distinguishes_raw_c_beta_and_row_relative_coefficients():
    torch = pytest.importorskip("torch")
    from research.conductance_gat.v5.distribution_audit import audit_conductance_distribution
    from research.conductance_gat.v5.stage_audit import _json

    distribution = _json(
        audit_conductance_distribution(
            torch.tensor([0.2, 0.8, 2.0]),
            torch.tensor([[0, 0, 0], [1, 2, 3]]),
            torch.zeros(5, dtype=torch.long),
            1,
            heads=2,
            beta=torch.tensor([[0.3, 0.7]]),
        )
    )
    output = "\n".join(audit.human_distribution(distribution))
    for text in (
        "raw C is not probability",
        "CV=std/mean",
        "beta=[",
        "raw C quantiles=",
        "c_ge_0_7",
        "histogram=",
        "degree1=3",
        "isolates=1",
        "diagnostic only",
        "effective neighbors=",
        "normalized entropy=",
        "normalized head TV means=",
        "JS means=",
        "C=1 same-correction",
    ):
        assert text in output


def test_recorded_layer_solver_and_sampling_observations_retained():
    layers = [
        {
            "layer": 0,
            "conductance": {"mean": 1.0, "cv": 0.2},
            "beta": {"mean": 0.4, "min": 0.3, "max": 0.5},
            "conductance_gradient_norm": 0.001,
            "c_optimization": {"projected_gradient_rms_final": [0.06]},
        }
    ]
    rows = [{"epoch": 1, "layers": layers, "sampling_observation": {"nodes": 11, "overlap": 0.5}}]
    result = audit.historical_diagnostics({}, rows)
    assert result["layers"][0]["c_cv"] == 0.2
    assert result["epoch_layers"][0]["layers"][0]["conductance_gradient_norm"] == 0.001
    assert result["sampling_observations"][0]["sampling_observation"]["overlap"] == 0.5


def test_source_compatibility_is_exact_and_readonly(monkeypatch):
    baseline = audit.BASELINE_SOURCES
    monkeypatch.setattr(audit, "AUDIT_SOURCE_OVERRIDES", {})
    result = audit.verify_sources(copy.deepcopy(baseline), copy.deepcopy(baseline))
    assert result["training_resume_authorized"] is False
    assert result["historical_repair"] is None
    modified = {**baseline, "research/conductance_gat/v5/model.py": "1" * 64}
    with pytest.raises(audit.AuditError, match="exact reviewed"):
        audit.verify_sources(baseline, modified)
    with pytest.raises(ValueError):
        audit.verify_sources(modified, baseline)


def test_packaged_audit_source_pins_match_live_implementation():
    pytest.importorskip("torch")
    from research.conductance_gat.v5.train import implementation_source_hashes

    current = implementation_source_hashes()
    verified = audit.verify_sources(audit.BASELINE_SOURCES, current)
    assert verified["training_resume_authorized"] is False
    assert verified["audit_sources"] == current


def test_historical_audit_release_is_readonly_and_exact():
    current = {**audit.BASELINE_SOURCES, **audit.AUDIT_SOURCE_OVERRIDES}
    historical = {**audit.BASELINE_SOURCES, **audit.HISTORICAL_AUDIT_SOURCE_OVERRIDES}
    verified = audit.verify_sources(historical, current)
    assert verified["training_resume_authorized"] is False
    assert verified["checkpoint_sources"] == historical
    assert verified["historical_repair"] is None
    tampered = {**historical, "research/conductance_gat/v5/model.py": "f" * 64}
    with pytest.raises(ValueError):
        audit.verify_sources(tampered, current)


def test_reviewed_added_source_requires_exact_digest(monkeypatch):
    added = {"research/conductance_gat/v5/stage_audit.py": "2" * 64}
    monkeypatch.setattr(audit, "AUDIT_SOURCE_OVERRIDES", added)
    current = {**audit.BASELINE_SOURCES, **added}
    assert (
        audit.verify_sources(audit.BASELINE_SOURCES, current)["training_resume_authorized"] is False
    )
    with pytest.raises(audit.AuditError):
        audit.verify_sources(audit.BASELINE_SOURCES, audit.BASELINE_SOURCES)


def test_immutable_guard_detects_mutation(evidence):
    path, _ = evidence
    original = audit.inspect_evidence(path)["artifacts"]
    (path.parent / "best.pt").write_bytes(b"mutation")
    with pytest.raises(audit.AuditError, match="changed during"):
        audit.verify_unchanged(original)


def test_cpu_device_rejected_without_running_a_model(tmp_path, capsys):
    result = audit.main([*_args(tmp_path), "--device", "cpu", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert result == 1
    assert report["files_written"] is False
    assert report["test_evaluated"] is False
    assert "CPU fallback" in report["error"]
    assert list(tmp_path.iterdir()) == []


def test_validation_source_never_selects_test_split():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    class ValidationOnly(dict):
        def __getitem__(self, key):
            assert key == "validation", "audit attempted to select a non-validation split"
            return super().__getitem__(key)

    payload = {
        "dataset": "cora",
        "graphs": [
            {
                "x": torch.ones(4, 2),
                "y": torch.zeros(4, dtype=torch.long),
                "incidence_edge_index": torch.tensor([[0, 1], [1, 2]]),
            }
        ],
        "splits": ValidationOnly(validation=torch.tensor([False, True, False, True])),
    }
    source, indices, metadata = audit.validation_source(payload, {})
    assert indices.tolist() == [1, 3]
    assert source.x.shape == (4, 2)
    assert metadata["used_fraction"] == 1.0


def test_pii_validation_preserves_disjoint_batch_and_all_graphs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    graph = {
        "x": torch.ones(3, 2),
        "y": torch.zeros(3, 2),
        "incidence_edge_index": torch.tensor([[0, 1], [1, 2]]),
    }
    payload = {"dataset": "ppi", "graphs": [graph, graph], "splits": {"validation": [0, 1]}}
    source, indices, metadata = audit.validation_source(
        payload, {"workers": 0, "pin_memory": False, "batch_size": 20}
    )
    batches = list(source)
    assert indices is None and len(batches) == 1
    assert batches[0].num_graphs == 2
    assert batches[0].incidence_edge_index.max().item() == 5
    assert metadata["saved_graph_batch_size"] == 20
    assert metadata["actual_physical_batch_size"] == 2


def test_json_duplicate_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    for data in ('{"x": 1, "x": 2}', '{"x": NaN}'):
        path.write_text(data, encoding="utf-8")
        with pytest.raises(audit.AuditError):
            audit._read(path)


def test_aggregate_verification_does_not_certify_unexecuted_conditions():
    complete = {"verification": {"source_verified": True, "cache_split_hashes_verified": True}}
    assert audit.aggregate_verification([complete])["cache_split_hashes_verified"] is True
    missing = {"execution_status": "failed", "verification": {"source_verified": False}}
    report = audit.aggregate_verification([complete, missing])
    assert report["cache_split_hashes_verified"] is None
    assert report["source_verified"] is False
    assert audit.aggregate_verification([])["cache_split_hashes_verified"] is None


def test_sampling_summary_retains_latest_batch_sizes_without_invented_overlap():
    result = audit.sampling_summary(
        {
            "batch_observations": [
                {"epoch": 3, "batches": [{"nodes": 10, "physical_edges": 20}]},
                {
                    "epoch": 4,
                    "batches": [
                        {"nodes": 13, "physical_edges": 21, "sampling_observation": None},
                        {"nodes": 11, "physical_edges": 19, "sampling_observation": None},
                    ],
                },
            ],
        }
    )
    assert result["epoch"] == 4
    assert result["nodes"] == {"min": 11, "max": 13}
    assert result["physical_edges"] == {"min": 19, "max": 21}
    assert result["previous_batch_node_jaccard"] is None
    assert result["overlap_available"] is False
    assert result["saturated_batch_count"] is None


def test_sampling_summary_reads_actual_overlap():
    result = audit.sampling_summary(
        {
            "batch_observations": [
                {
                    "epoch": 2,
                    "batches": [
                        {
                            "nodes": 14,
                            "physical_edges": 20,
                            "sampling_observation": {
                                "cluster_budget_saturated": True,
                                "previous_physical_batch_overlap": {
                                    "nodes": {"jaccard": 0.7},
                                    "physical_edges": {"jaccard": 0.6},
                                },
                                "within_batch_context_overlap": [{"nodes": {"jaccard": 0.3}}],
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert result["previous_batch_node_jaccard"] == {"min": 0.7, "max": 0.7}
    assert result["previous_batch_edge_jaccard"] == {"min": 0.6, "max": 0.6}
    assert result["within_batch_node_jaccard"] == {"min": 0.3, "max": 0.3}
    assert result["saturated_batch_count"] == 1


def test_human_report_contains_c_intervention_and_frozen_input_reference_details():
    row = {
        "dataset": "cora",
        "condition": "shared_dynamic_c",
        "metrics_path": "/unit/metrics.json",
        "selected_epoch": 2,
        "selected_validation": 0.7,
        "historical_diagnostics": {"layers": "unavailable", "batch_observations": "unavailable"},
        "resources": {"peak_allocated_bytes": 100, "elapsed_seconds": 3.0},
        "audit": {
            "execution_status": "passed",
            "contribution_status": "observed",
            "interventions": {
                "c_one": {
                    "metric": 0.68,
                    "delta_from_learned": -0.02,
                    "logit_relative_l2": 0.12,
                    "logit_max_abs": 0.8,
                    "prediction_changed_fraction": 0.06,
                }
            },
            "reference_contract": {"reference_steps": 64, "exact_optimum_claimed": False},
            "local_layer_comparisons": [
                {
                    "batch": 0,
                    "layer": 1,
                    "operator_deployed_vs_reference": {"relative_l2": 0.03},
                    "operator_c_one_vs_deployed": {"relative_l2": 0.04},
                    "c_deployed_vs_reference": {"relative_l2": 0.05},
                    "baseline_solver": {"projected_gradient_rms_final": [0.07]},
                    "reference_solver": {"projected_gradient_rms_final": [0.002]},
                    "reference_residual_tolerance": 0.001,
                    "reference_tolerance_reached_by_graph": [False],
                }
            ],
        },
    }
    output = audit.human_condition(row)
    for expected in (
        "68.0000%",
        "-2.0000 pp",
        "logit relative-L2=0.12",
        "prediction changed=6.0000%",
        "frozen H/B/W/beta",
        "C relative-L2=0.05",
        "residual deployed/reference=[0.07]/[0.002]",
        "tolerance=0.001",
        "exact optimum=False",
        "not recorded",
    ):
        assert expected in output
    assert "not a usefulness certificate" in output


def _mock_monitor(monkeypatch, *, forward_error=None, finish_error=None):
    torch = pytest.importorskip("torch")
    from chartgat import observability
    from research.conductance_gat.v5 import stage_audit

    events = []
    missing_gpu = {"value": None, "reason": "synthetic CPU test has no GPU measurement"}
    report = {
        "summary": {
            "run_average_gpu_sm_utilization_percent": missing_gpu,
            "average_cpu_percent_of_one_core": {"value": 125.0, "reason": None},
        },
        "interval_series": {
            "process_resident_bytes": {"maximum": {"value": 2**30, "reason": None}},
            "system_available_bytes": {"minimum": {"value": 4 * 2**30, "reason": None}},
        },
        "background_sample_count": 2,
        "sampler_errors": [],
    }

    class Monitor:
        active = False

        def __init__(self, device):
            assert device.type == "cpu"

        def start(self):
            events.append("start")
            self.active = True

        def finish(self, **peaks):
            events.append("finish")
            assert peaks == {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
            self.active = False
            if finish_error is not None:
                raise finish_error
            return report

    monitor = Monitor(torch.device("cpu"))
    monkeypatch.setattr(observability, "RuntimeResourceMonitor", lambda device: monitor)

    def forward(*args, **kwargs):
        events.append("forward")
        assert monitor.active
        assert kwargs["reference_steps"] == 64
        if forward_error is not None:
            raise forward_error
        return {"execution_status": "passed", "scope": "synthetic mocked CPU forward"}

    monkeypatch.setattr(stage_audit, "audit_stage_roles", forward)
    return monitor, events, report


def _run_mocked_monitor():
    import torch

    return audit._monitored_stage_audit(
        object(),
        object(),
        None,
        device=torch.device("cpu"),
        precision="fp32",
        reference_steps=64,
        reference_tolerance=0.001,
    )


def test_interval_monitor_actually_wraps_forward_and_preserves_full_report(monkeypatch):
    monitor, events, expected = _mock_monitor(monkeypatch)
    result, resources = _run_mocked_monitor()
    assert events == ["start", "forward", "finish"]
    assert not monitor.active
    assert result["execution_status"] == "passed"
    assert resources["runtime_observability"] is expected
    assert resources["gpu_utilization"]["value"] is None
    assert resources["peak_allocated_bytes"] is None
    json.dumps(resources, allow_nan=False)
    text = "\n".join(audit.human_resources(resources))
    assert "synthetic CPU test has no GPU measurement" in text
    assert "CPU process=125.000% of one core" in text
    assert "RSS max=1.000 GiB" in text
    assert "host RAM min available=4.000 GiB" in text


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_forward_failure_still_finishes_monitor_and_keeps_primary_error(monkeypatch, error_type):
    primary = error_type("synthetic forward failure")
    monitor, events, expected = _mock_monitor(monkeypatch, forward_error=primary)
    with pytest.raises(error_type) as caught:
        _run_mocked_monitor()
    assert caught.value is primary
    assert events == ["start", "forward", "finish"]
    assert not monitor.active
    assert primary.audit_resources["runtime_observability"] is expected


def test_monitor_cleanup_failure_is_visible_without_masking_forward_failure(monkeypatch):
    primary = RuntimeError("synthetic primary")
    cleanup = ValueError("synthetic finish failure")
    monitor, events, _ = _mock_monitor(monkeypatch, forward_error=primary, finish_error=cleanup)
    with pytest.raises(RuntimeError) as caught:
        _run_mocked_monitor()
    assert caught.value is primary
    assert events == ["start", "forward", "finish"] and not monitor.active
    assert any("synthetic finish failure" in note for note in primary.__notes__)
    resources = primary.audit_resources
    assert resources["runtime_observability"] is None
    assert resources["resource_monitor"]["cleanup_errors"][0]["stage"] == (
        "runtime_resource_monitor_finish"
    )


def test_monitor_failure_after_successful_forward_is_not_hidden(monkeypatch):
    cleanup = RuntimeError("synthetic finish failure")
    monitor, events, _ = _mock_monitor(monkeypatch, finish_error=cleanup)
    with pytest.raises(RuntimeError) as caught:
        _run_mocked_monitor()
    assert caught.value is cleanup
    assert events == ["start", "forward", "finish"] and not monitor.active
    assert any("numerical forwards completed" in note for note in cleanup.__notes__)
