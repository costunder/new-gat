"""Synthetic CPU artifact-contract fixtures, not actual training or GPU results."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from chartgat.observability import finalize_resource_observability, runtime_resource_snapshot
from research.conductance_gat.v5 import train
from research.conductance_gat.v5 import transition_report as report
from research.conductance_gat.v5.protocol import SUITE
from scripts import run_conductance_scaling as scaling


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path, *, condition="fixed_c", source_epoch=0, transitioned=False):
    """Serialize clearly synthetic records with production configuration writers."""
    options = scaling.parser().parse_args(
        ["--versions", "v5", "--profiles", "reference", "--datasets", "cora"]
    )
    job = next(
        item
        for item in scaling.make_jobs(options, tmp_path / "debug-artifacts")
        if item["condition"] == condition
    )
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    args = train.build_parser().parse_args(job["command"][5:])
    train.validate_args(args)
    config = train.configuration(args)
    if not transitioned:
        # A historical record genuinely lacks the new options; do not normalize
        # it using the current optimization/joint defaults.
        for key in (
            "conductance_backend",
            "solver_steps",
            "solver_step_size",
            "solver_entropy",
            "solver_degree_barrier",
            "training_schedule",
        ):
            config.pop(key, None)
            job["architecture"].pop(key, None)
    epochs = config["epochs"]
    history = [
        {"epoch": epoch, "validation": 0.75, "train_batches": 3, "train_label_count": 120}
        for epoch in range(source_epoch + 1, epochs + 1)
    ]
    sources = {"explicit-debug-fixture/v5.py": "a" * 64}
    protocol = {"data_sha256": "b" * 64, "split": "explicit synthetic unit fixture"}
    schedule = [{"phase": "joint" if transitioned else "staged-fixture", "start": 1, "end": epochs}]
    identity = {
        "schema_version": 1,
        "research_suite": SUITE,
        "dataset": "cora",
        "condition": condition,
        "configuration": config,
        "schedule": schedule,
        "dataset_protocol": protocol,
        "dataset_protocol_sha256": report._canonical(protocol),
        "cache_sha256": "b" * 64,
        "initial_state_sha256": "c" * 64,
        "source_sha256": sources,
        "runtime_versions": {"torch": "explicit CPU fixture, no experiment"},
    }
    provenance = None
    if transitioned:
        source_path = tmp_path / "explicit-debug-source.pt"
        torch.save({"explicit_cpu_fixture_only": torch.ones(1)}, source_path)
        source_identity = copy.deepcopy(identity)
        original_history = [
            {"epoch": value, "validation": 0.95} for value in range(1, source_epoch + 1)
        ]
        _write_json(output / "source-history.json", original_history)
        provenance = {
            "source_checkpoint_sha256": _sha(source_path),
            "source_path": str(source_path.resolve()),
            "source_identity": source_identity,
            "source_identity_sha256": report._canonical(source_identity),
            "source_history_sha256": report._canonical(original_history),
            "source_epoch": source_epoch,
            "source_epochs_requested": epochs,
            "target_total_epochs": epochs,
            "epoch_offset": source_epoch,
            "mode": "replace_c",
            "additional_epochs": 0,
            "source_complete": False,
            "historical_metrics_are_new_c_metrics": False,
            "test_labels_used": False,
            "common_optimizer_state": "reused",
            "conductance_optimizer_state": "reset",
        }
        identity["transition_provenance_sha256"] = report._canonical(provenance)
        job.update(
            source_checkpoint=str(source_path.resolve()), source_checkpoint_sha256=_sha(source_path)
        )
    common = {
        "resume_identity": identity,
        "resume_identity_sha256": report._canonical(identity),
        "model_state": {"explicit_cpu_fixture_only": torch.ones(1)},
    }
    if provenance is not None:
        common["transition_provenance"] = provenance
    best = {
        **common,
        "epoch": epochs,
        "validation": 0.75,
        "selection_role": "primary",
        "configuration": config,
        "schedule": schedule,
        "condition": condition,
    }
    torch.save(best, output / "best.pt")
    last = {
        **common,
        "schema_version": 4 if transitioned else 3,
        "epoch_offset": source_epoch,
        "complete": True,
        "epoch": epochs,
        "history": history,
        "best_metric": 0.75,
        "best_epoch": epochs,
        "global_best_metric": 0.75,
        "global_best_epoch": epochs,
        "joint_best_metric": 0.75,
        "joint_best_epoch": epochs,
        "best_checkpoint_sha256": _sha(output / "best.pt"),
    }
    torch.save(last, output / "last.pt")
    _write_json(output / "history.json", history)
    hardware = {
        "profile": config["hardware_profile"],
        "precision": config["precision"],
        "tf32": config["tf32"],
        "activation_checkpoint": config["activation_checkpoint"],
        "edge_chunk_size": config["edge_chunk_size"],
        "sample_seed_batch_size": config["sample_seed_batch_size"],
        "graph_batch_size": config["batch_size"],
        "sample_prefetch": config["sample_prefetch"],
        "pin_memory": config["pin_memory"],
    }
    device = torch.device("cpu")
    metrics = {
        "status": "passed",
        "research_suite": SUITE,
        "dataset": "cora",
        "condition": condition,
        "model_seed": 0,
        "configuration": config,
        "schedule": schedule,
        "protocol": protocol,
        "cache_sha256": "b" * 64,
        "source_sha256": sources,
        "initial_state_sha256": "c" * 64,
        "shared_initial_state_sha256": "d" * 64,
        "versions": identity["runtime_versions"],
        "resume_identity": identity,
        "resume_identity_sha256": report._canonical(identity),
        "comparison_design": {
            "single_factor_causal_effect_of_c": False,
            "historical_recipe": "explicit fixture",
        },
        "validation": 0.75,
        "best_epoch": epochs,
        "epochs_run": len(history),
        "global_best_validation": 0.75,
        "global_best_epoch": epochs,
        "joint_best_validation": None if condition == "fixed_c" else 0.75,
        "joint_best_epoch": None if condition == "fixed_c" else epochs,
        "checkpoint_selection": {
            "test_used": False,
            "primary_validation": 0.75,
            "primary_epoch": epochs,
            "primary_role": "all_epoch_prediction_best"
            if condition == "fixed_c"
            else "c_active_mechanism_best",
            "global_prediction_validation": 0.75,
            "global_prediction_epoch": epochs,
        },
        "metric_name": "accuracy",
        "evaluation_split": "validation",
        "test_evaluated": False,
        "trainable_parameters": 1234,
        "total_parameters": 1300,
        "elapsed_seconds": 1.5,
        "peak_cuda_allocated_bytes": 0,
        "hardware_execution": hardware,
        "resource_observability": finalize_resource_observability(
            runtime_resource_snapshot(device),
            device,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            sample_interval_seconds=1.0,
        ),
        "throughput": train.training_throughput(history, 1.5),
    }
    if provenance is not None:
        metrics["transition_provenance"] = provenance
        metrics["source_history"] = str((output / "source-history.json").resolve())
        metrics["source_history_sha256"] = _sha(output / "source-history.json")
    for key, filename in (
        ("checkpoint", "best.pt"),
        ("last_checkpoint", "last.pt"),
        ("history", "history.json"),
    ):
        metrics[key] = str((output / filename).resolve())
        metrics[f"{key}_sha256"] = _sha(output / filename)
    _write_json(output / "metrics.json", metrics)
    job.update(status="passed", metrics_sha256=_sha(output / "metrics.json"))
    job["result"] = scaling._load_child(job)
    manifest = {"status": "passed", "source_sha256": sources, "jobs": [job]}
    manifest_path = tmp_path / "source-manifest.json"
    _write_json(manifest_path, manifest)
    return job, manifest, manifest_path


def _snapshot(root):
    return {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_historical_fixed_is_verified_without_retraining_or_rewriting(tmp_path):
    job, manifest, path = _fixtures(tmp_path)
    before = _snapshot(tmp_path)
    reference = report.validate_historical_reference(job, manifest, source_manifest_path=path)
    assert reference["role"] == "historical_reference"
    assert reference["newly_trained_epochs"] == 0
    assert reference["counts_as_new_solver_completion"] is False
    assert "conductance_backend" not in reference["configuration"]
    assert reference["source_manifest_sha256"] == _sha(path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("artifact", ["metrics.json", "best.pt", "last.pt", "history.json"])
def test_historical_tampered_artifact_is_rejected_without_writes(tmp_path, artifact):
    job, manifest, path = _fixtures(tmp_path)
    target = Path(job["output_dir"]) / artifact
    target.write_bytes(target.read_bytes() + b" ")
    before = _snapshot(tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="hash|result"):
        report.validate_historical_reference(job, manifest, source_manifest_path=path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_incomplete_fixed_is_not_a_completed_historical_reference(tmp_path, status):
    job, manifest, path = _fixtures(tmp_path)
    job["status"] = status
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="completed passed"):
        report.validate_historical_reference(job, manifest, source_manifest_path=path)


@pytest.mark.parametrize(
    "field", ["source_sha256", "runtime_versions", "dataset_protocol", "configuration"]
)
def test_self_consistent_identity_rehash_does_not_allow_metric_contract_mismatch(tmp_path, field):
    job, manifest, path = _fixtures(tmp_path)
    metrics_path = Path(job["metrics_path"])
    metrics = json.loads(metrics_path.read_bytes())
    metrics["resume_identity"][field]["unexpected"] = "changed"
    metrics["resume_identity_sha256"] = report._canonical(metrics["resume_identity"])
    _write_json(metrics_path, metrics)
    job["metrics_sha256"] = _sha(metrics_path)
    job["result"] = scaling._load_child(job)
    _write_json(path, manifest)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        report.validate_historical_reference(job, manifest, source_manifest_path=path)
    assert _snapshot(tmp_path) == before


def test_changed_source_manifest_and_saved_summary_are_rejected(tmp_path):
    job, manifest, path = _fixtures(tmp_path)
    altered = copy.deepcopy(manifest)
    altered["status"] = "changed"
    with pytest.raises(ValueError, match="source manifest changed"):
        report.validate_historical_reference(job, altered, source_manifest_path=path)
    job["result"]["validation"] = 0.99
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="saved result"):
        report.validate_historical_reference(job, manifest, source_manifest_path=path)


def test_source_manifest_must_bind_recorded_child_sources(tmp_path):
    job, manifest, path = _fixtures(tmp_path)
    manifest["source_sha256"] = {"different-source.py": "f" * 64}
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="child/source manifest"):
        report.validate_historical_reference(job, manifest, source_manifest_path=path)


def test_completed_legacy_dynamic_is_preserved_but_never_counted_as_new_c(tmp_path):
    job, source, path = _fixtures(tmp_path, condition="shared_dynamic_c")
    reference = report.validate_historical_reference(job, source, source_manifest_path=path)
    manifest = {
        "jobs": [
            {
                "action": "preserve_legacy_dynamic",
                "status": "blocked",
                "source_job": job,
                "historical_reference": reference,
            }
        ]
    }
    before = _snapshot(tmp_path)
    result = report.build_transition_report(manifest)
    assert result["status"] == "partial"
    assert result["new_solver_completions"] == 0
    assert result["rows"][0]["validation"] == 0.75
    assert result["comparison_design"]["fresh_same_initialization_paired_comparison"] is False
    assert result["comparison_design"]["sota_claim"] is False
    assert _snapshot(tmp_path) == before


def test_report_revalidates_historical_reference_instead_of_trusting_cached_verified(tmp_path):
    job, source, path = _fixtures(tmp_path)
    reference = report.validate_historical_reference(job, source, source_manifest_path=path)
    reference["validation"] = 0.99
    manifest = {"jobs": [{"action": "reuse_completed_fixed", "historical_reference": reference}]}
    with pytest.raises(ValueError, match="reference changed"):
        report.build_transition_report(manifest)


def test_new_solver_only_selects_post_transition_history(tmp_path):
    job, _, _ = _fixtures(
        tmp_path, condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    job.update(action="transition_dynamic", source_epoch=40)
    result = report.validate_transition_child(job)
    assert result["role"] == "transitioned_training"
    assert result["pre_transition_epochs"] == 40
    assert result["post_transition_epochs"] == result["configuration"]["epochs"] - 40
    assert result["counts_as_new_solver_completion"] is True
    assert result["fresh_same_initialization_comparison"] is False


@pytest.mark.parametrize(
    "field,value", [("source_epoch", 39), ("source_checkpoint_sha256", "f" * 64)]
)
def test_transition_lineage_mismatch_is_rejected(tmp_path, field, value):
    job, _, _ = _fixtures(
        tmp_path, condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    job.update(action="transition_dynamic", source_epoch=40)
    job[field] = value
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="epoch|hash"):
        report.validate_transition_child(job)
    assert _snapshot(tmp_path) == before


def test_resolved_training_command_is_checked_not_only_baseline(tmp_path):
    job, _, _ = _fixtures(
        tmp_path, condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    job.update(action="transition_dynamic", source_epoch=40)
    job["resolved_command"] = list(job["command"])
    index = job["resolved_command"].index("--hidden-channels") + 1
    job["resolved_command"][index] = "512"
    with pytest.raises(ValueError, match="resolved command/configuration"):
        report.validate_transition_child(job)


def test_ordinary_v5_report_contract_is_not_replaced(tmp_path):
    from research.conductance_gat.v5 import report as ordinary

    job, _, _ = _fixtures(tmp_path)
    metrics = json.loads(Path(job["metrics_path"]).read_bytes())
    assert metrics["comparison_design"] != ordinary.COMPARISON_DESIGN
    assert "sota_claim" not in ordinary.COMPARISON_DESIGN


def _rebind_transition_fixture(job, mutation):
    """Rehash edited synthetic fixtures to test semantic, not only hash, guards."""
    output = Path(job["output_dir"])
    metrics = json.loads((output / "metrics.json").read_bytes())
    mutation(metrics)
    provenance = metrics["transition_provenance"]
    metrics["resume_identity"]["transition_provenance_sha256"] = report._canonical(provenance)
    metrics["resume_identity_sha256"] = report._canonical(metrics["resume_identity"])
    for filename in ("best.pt", "last.pt"):
        value = torch.load(output / filename, map_location="cpu", weights_only=True)
        value.update(
            transition_provenance=provenance,
            resume_identity=metrics["resume_identity"],
            resume_identity_sha256=metrics["resume_identity_sha256"],
        )
        if filename == "last.pt":
            value["best_checkpoint_sha256"] = _sha(output / "best.pt")
        torch.save(value, output / filename)
    metrics["checkpoint_sha256"] = _sha(output / "best.pt")
    metrics["last_checkpoint_sha256"] = _sha(output / "last.pt")
    _write_json(output / "metrics.json", metrics)
    job["metrics_sha256"] = _sha(output / "metrics.json")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("mode", "continue_fixed", "mode mismatch"),
        ("additional_epochs", 1, "epoch budget mismatch"),
        ("historical_metrics_are_new_c_metrics", True, "metric provenance claims"),
        ("test_labels_used", True, "metric provenance claims"),
    ],
)
def test_rehashed_provenance_still_cannot_change_transition_policy(tmp_path, field, value, message):
    job, _, _ = _fixtures(
        tmp_path, condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    job.update(action="transition_dynamic", source_epoch=40)
    _rebind_transition_fixture(
        job, lambda child: child["transition_provenance"].update({field: value})
    )
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match=message):
        report.validate_transition_child(job)
    assert _snapshot(tmp_path) == before


def test_new_c_cannot_select_the_better_pre_transition_score(tmp_path):
    job, _, _ = _fixtures(
        tmp_path, condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    job.update(action="transition_dynamic", source_epoch=40)
    _rebind_transition_fixture(job, lambda child: child.update(best_epoch=20, validation=0.95))
    with pytest.raises(ValueError, match="selected metric/epoch"):
        report.validate_transition_child(job)


def test_new_source_history_is_required_and_hash_verified(tmp_path):
    job, _, _ = _fixtures(
        tmp_path, condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    job.update(action="transition_dynamic", source_epoch=40)
    path = Path(job["output_dir"]) / "source-history.json"
    path.write_bytes(path.read_bytes() + b" ")
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="source_history hash"):
        report.validate_transition_child(job)
    assert _snapshot(tmp_path) == before


def test_source_checkpoint_change_after_training_is_rejected(tmp_path):
    job, _, _ = _fixtures(
        tmp_path, condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    job.update(action="transition_dynamic", source_epoch=40)
    path = Path(job["source_checkpoint"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source checkpoint path/hash changed"):
        report.validate_transition_child(job)


def test_historical_and_new_dynamic_scores_are_never_reported_as_a_fresh_pair(tmp_path):
    old_job, source, path = _fixtures(tmp_path / "old")
    reference = report.validate_historical_reference(old_job, source, source_manifest_path=path)
    new_job, _, _ = _fixtures(
        tmp_path / "new", condition="shared_dynamic_c", source_epoch=40, transitioned=True
    )
    new_job.update(action="transition_dynamic", source_epoch=40)
    manifest = {
        "jobs": [
            {
                "action": "reuse_completed_fixed",
                "status": "passed",
                "historical_reference": reference,
            },
            new_job,
        ]
    }
    result = report.build_transition_report(manifest)
    assert result["status"] == "passed"
    assert {row["role"] for row in result["rows"]} == {
        "historical_reference",
        "transitioned_training",
    }
    assert result["new_solver_completions"] == 1
    assert "contrasts" not in result
    assert result["comparison_design"]["fresh_same_initialization_paired_comparison"] is False


def _reviewed_legacy_sources():
    """Immutable archival hashes, not today's changed V5 implementation."""
    from chartgat import resume_compat
    from research.conductance_gat.v5.transition import LEGACY_SOURCE_SNAPSHOTS
    from tests.test_v5_transition_state import LEGACY_SOURCE

    before = copy.deepcopy(LEGACY_SOURCE)
    registry = json.loads(resume_compat.REGISTRY_PATH.read_bytes())
    after = {
        name: registry["changes"].get(name, {}).get("after", value)
        for name, value in before.items()
    }
    after[resume_compat.HELPER_SOURCE] = registry["changes"][resume_compat.HELPER_SOURCE]["after"]
    after[resume_compat.REGISTRY_SOURCE] = registry.get("performance_repair", {}).get(
        "registry_before_sha256", _sha(resume_compat.REGISTRY_PATH)
    )
    assert LEGACY_SOURCE_SNAPSHOTS[report._canonical(before)].startswith("76e514a")
    assert LEGACY_SOURCE_SNAPSHOTS[report._canonical(after)].startswith("8963821")
    evidence = resume_compat.require_source_compatibility(before, after)
    assert evidence["patch_id"] == "v5-rng-cycle-workers-v1"
    return before, after, evidence


def _historical_journal_fixture(tmp_path, *, resumed_after_repair):
    job, manifest, path = _fixtures(tmp_path)
    before, after, evidence = _reviewed_legacy_sources()
    output = Path(job["output_dir"])
    metrics = json.loads((output / "metrics.json").read_bytes())
    metric_sources = after if resumed_after_repair else before
    metrics["source_sha256"] = metric_sources
    metrics["resume_identity"]["source_sha256"] = metric_sources
    metrics["resume_identity_sha256"] = report._canonical(metrics["resume_identity"])
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    best["resume_identity"]["source_sha256"] = before
    best["resume_identity_sha256"] = report._canonical(best["resume_identity"])
    torch.save(best, output / "best.pt")
    # The numerical repair may leave an original journal slot on disk. Its
    # hashes and identity stay archival instead of being rewritten to new C.
    torch.save(best, output / "best.previous.pt")
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    last["resume_identity"]["source_sha256"] = metric_sources
    last["resume_identity_sha256"] = report._canonical(last["resume_identity"])
    last["best_checkpoint_sha256"] = _sha(output / "best.pt")
    torch.save(last, output / "last.pt")
    metrics["checkpoint_sha256"] = _sha(output / "best.pt")
    metrics["last_checkpoint_sha256"] = _sha(output / "last.pt")
    _write_json(output / "metrics.json", metrics)
    job["metrics_sha256"] = _sha(output / "metrics.json")
    job["result"] = scaling._load_child(job)
    manifest["source_sha256"] = after
    manifest["source_compatibility"] = [evidence]
    _write_json(path, manifest)
    return job, manifest, path


@pytest.mark.parametrize("resumed_after_repair", [False, True])
def test_reviewed_76_best_and_896_manifest_or_last_remain_historical_read_only(
    tmp_path, resumed_after_repair
):
    job, manifest, path = _historical_journal_fixture(
        tmp_path, resumed_after_repair=resumed_after_repair
    )
    before = _snapshot(tmp_path)
    reference = report.validate_historical_reference(job, manifest, source_manifest_path=path)
    assert reference["role"] == "historical_reference"
    assert reference["counts_as_new_solver_completion"] is False
    assert "best_previous_checkpoint" in reference["artifacts"]
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("change", ["unreviewed_source", "changed_recipe"])
def test_archived_mixed_journal_never_waives_unreviewed_code_or_recipe(tmp_path, change):
    job, manifest, path = _historical_journal_fixture(tmp_path, resumed_after_repair=True)
    output = Path(job["output_dir"])
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if change == "unreviewed_source":
        best["resume_identity"]["source_sha256"]["research/conductance_gat/v5/train.py"] = "f" * 64
    else:
        best["resume_identity"]["configuration"]["dropout"] = 0.99
    best["resume_identity_sha256"] = report._canonical(best["resume_identity"])
    torch.save(best, output / "best.pt")
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    last["best_checkpoint_sha256"] = _sha(output / "best.pt")
    torch.save(last, output / "last.pt")
    metrics = json.loads((output / "metrics.json").read_bytes())
    metrics["checkpoint_sha256"] = _sha(output / "best.pt")
    metrics["last_checkpoint_sha256"] = _sha(output / "last.pt")
    _write_json(output / "metrics.json", metrics)
    job["metrics_sha256"] = _sha(output / "metrics.json")
    job["result"] = scaling._load_child(job)
    _write_json(path, manifest)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="checkpoint/metrics identity mismatch"):
        report.validate_historical_reference(job, manifest, source_manifest_path=path)
    assert _snapshot(tmp_path) == before
