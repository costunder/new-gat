"""Read-only artifact checks and provenance-separated selective-transition reports.

Historical scores are archival references, never newly completed solver runs or
fresh same-initialization controls. This module neither rewrites source artifacts
nor relaxes the ordinary V5 comparison/resume contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from chartgat.resume_compat import snapshots_match

from .protocol import CONDITIONS, METRIC_BY_DATASET, SUITE


class TransitionReportIntegrityError(ValueError):
    """An artifact or its declared transition lineage cannot be verified."""


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransitionReportIntegrityError(message)


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _metric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _source_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(name, str) and bool(name) and _sha(digest) for name, digest in value.items()
        )
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _identity(record: dict[str, Any], expected: dict[str, Any] | None = None) -> dict[str, Any]:
    identity = record.get("resume_identity")
    _require(isinstance(identity, dict), "missing resume identity")
    _require(record.get("resume_identity_sha256") == _canonical(identity), "identity hash mismatch")
    if expected is not None and identity != expected:
        changed = {
            key
            for key in identity.keys() | expected.keys()
            if identity.get(key) != expected.get(key)
        }
        _require(
            changed == {"source_sha256"}
            and snapshots_match(identity["source_sha256"], expected["source_sha256"]),
            "checkpoint/metrics identity mismatch",
        )
    return identity


def _artifact(child: dict[str, Any], output: Path, key: str, filename: str) -> dict[str, str]:
    path_value, expected = child.get(key), child.get(f"{key}_sha256")
    _require(isinstance(path_value, str) and _sha(expected), f"missing {key} path/hash")
    path = Path(path_value).expanduser().resolve()
    _require(path == output / filename and path.is_file(), f"{key} path mismatch")
    _require(_digest(path) == expected, f"{key} hash mismatch")
    return {"path": str(path), "sha256": expected}


def _verified_child(
    job: dict[str, Any], *, epoch_offset: int = 0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify file bindings and internally recorded contracts, not today's recipe."""
    from scripts.telemetry_validation import (
        validate_resource_observability,
        validate_throughput_observability,
    )

    from .train import load_checkpoint_on_cpu

    output = Path(job["output_dir"]).expanduser().resolve()
    path = Path(job.get("metrics_path", output / "metrics.json")).expanduser().resolve()
    _require(path == output / "metrics.json" and path.is_file(), "metrics path mismatch")
    digest = _digest(path)
    declared_digest = job.get("metrics_sha256", job.get("result", {}).get("metrics_sha256"))
    if declared_digest is not None:
        _require(_sha(declared_digest) and declared_digest == digest, "metrics hash mismatch")
    child = _json(path)
    for key, value in (
        ("status", "passed"),
        ("research_suite", SUITE),
        ("dataset", job["dataset"]),
        ("condition", job["condition"]),
        ("model_seed", job["model_seed"]),
        ("evaluation_split", "validation"),
        ("test_evaluated", False),
    ):
        _require(child.get(key) == value, f"job/metrics {key} mismatch")
    _require(
        child["condition"] in CONDITIONS and "test" not in child,
        "invalid validation-only condition",
    )
    _require(
        child.get("metric_name") == METRIC_BY_DATASET.get(child["dataset"]), "metric name mismatch"
    )
    identity = _identity(child)
    _require(identity.get("schema_version") == 1, "unsupported identity schema")
    for key in (
        "research_suite",
        "dataset",
        "condition",
        "configuration",
        "schedule",
        "cache_sha256",
        "source_sha256",
        "initial_state_sha256",
    ):
        _require(identity.get(key) == child.get(key), f"identity/metrics {key} mismatch")
    config, protocol = child.get("configuration"), child.get("protocol")
    _require(
        isinstance(config, dict) and config.get("model_seed") == child["model_seed"],
        "configuration seed mismatch",
    )
    _require(
        isinstance(protocol, dict) and protocol == identity.get("dataset_protocol"),
        "dataset/split protocol mismatch",
    )
    _require(
        identity.get("dataset_protocol_sha256") == _canonical(protocol),
        "dataset/split protocol hash mismatch",
    )
    _require(
        _sha(child.get("cache_sha256")) and protocol.get("data_sha256") == child["cache_sha256"],
        "dataset cache hash mismatch",
    )
    _require(_source_map(child.get("source_sha256")), "invalid source hash inventory")
    _require(
        isinstance(child.get("versions"), dict)
        and bool(child["versions"])
        and child["versions"] == identity.get("runtime_versions"),
        "runtime identity mismatch",
    )
    _require(
        _sha(child.get("initial_state_sha256")) and _sha(child.get("shared_initial_state_sha256")),
        "missing initialization hashes",
    )
    for key, value in job.get("architecture", {}).items():
        _require(config.get(key) == value, f"job/metrics architecture mismatch: {key}")
    for key in ("workers", "batch_size", "sampling"):
        if key in job:
            _require(config.get(key) == job[key], f"job/metrics configuration mismatch: {key}")
    design = child.get("comparison_design")
    _require(isinstance(design, dict) and bool(design), "missing historical comparison design")
    _require(
        design.get("single_factor_causal_effect_of_c") is False, "invalid causal comparison claim"
    )
    resource = validate_resource_observability(
        child.get("resource_observability"), "transition.resource_observability"
    )
    throughput = validate_throughput_observability(child.get("throughput"), "transition.throughput")
    hardware = child.get("hardware_execution")
    _require(isinstance(hardware, dict), "missing hardware execution")
    for hardware_key, config_key in (
        ("profile", "hardware_profile"),
        ("precision", "precision"),
        ("tf32", "tf32"),
        ("activation_checkpoint", "activation_checkpoint"),
        ("edge_chunk_size", "edge_chunk_size"),
        ("sample_seed_batch_size", "sample_seed_batch_size"),
        ("graph_batch_size", "batch_size"),
        ("sample_prefetch", "sample_prefetch"),
        ("pin_memory", "pin_memory"),
    ):
        _require(
            hardware.get(hardware_key) == config.get(config_key), "hardware/configuration mismatch"
        )
    artifacts = {
        key: _artifact(child, output, key, filename)
        for key, filename in (
            ("checkpoint", "best.pt"),
            ("last_checkpoint", "last.pt"),
            ("history", "history.json"),
        )
    }
    last = load_checkpoint_on_cpu(Path(artifacts["last_checkpoint"]["path"]))
    best = load_checkpoint_on_cpu(Path(artifacts["checkpoint"]["path"]))
    _identity(last, identity)
    _identity(best, identity)
    _require(
        last.get("schema_version") == (4 if child.get("transition_provenance") is not None else 3)
        and last.get("complete") is True,
        "completed result has incomplete/unsupported last.pt",
    )
    if child.get("transition_provenance") is not None:
        _require(last.get("epoch_offset") == epoch_offset, "last checkpoint epoch offset mismatch")
        _require(
            identity.get("transition_provenance_sha256")
            == _canonical(child["transition_provenance"]),
            "transition provenance identity hash mismatch",
        )
        artifacts["source_history"] = _artifact(
            child, output, "source_history", "source-history.json"
        )
        original_history = json.loads(Path(artifacts["source_history"]["path"]).read_bytes())
        _require(
            isinstance(original_history, list)
            and _canonical(original_history)
            == child["transition_provenance"].get("source_history_sha256")
            and len(original_history) == child["transition_provenance"].get("source_epoch"),
            "original source history provenance mismatch",
        )
    history = json.loads(Path(artifacts["history"]["path"]).read_bytes())
    _require(
        isinstance(history, list) and bool(history) and history == last.get("history"),
        "checkpoint/history mismatch",
    )
    _require(
        _integer(epoch_offset)
        and [row.get("epoch") for row in history if isinstance(row, dict)]
        == list(range(epoch_offset + 1, epoch_offset + len(history) + 1)),
        "history epoch sequence mismatch",
    )
    _require(
        child.get("epochs_run") == len(history)
        and last.get("epoch") == epoch_offset + len(history),
        "completed epoch count mismatch",
    )
    _require(
        _integer(config.get("epochs"), 1) and last["epoch"] <= config["epochs"],
        "epoch budget exceeded",
    )
    for row in history:
        _require(_metric(row.get("validation")), "invalid history validation")
    value, epoch = child.get("validation"), child.get("best_epoch")
    _require(
        _metric(value) and _integer(epoch, epoch_offset + 1) and epoch <= last["epoch"],
        "invalid selected metric/epoch",
    )
    _require(
        last.get("best_metric") == value
        and last.get("best_epoch") == epoch
        and last.get("best_checkpoint_sha256") == artifacts["checkpoint"]["sha256"],
        "last/best selection mismatch",
    )
    _require(
        best.get("validation") == value
        and best.get("epoch") == epoch
        and best.get("selection_role") == "primary",
        "best checkpoint selection mismatch",
    )
    _require(
        best.get("configuration") == config
        and best.get("schedule") == child["schedule"]
        and best.get("condition") == child["condition"],
        "best checkpoint recipe mismatch",
    )
    _require(
        history[epoch - epoch_offset - 1]["validation"] == value,
        "selected metric is absent from history",
    )
    selection = child.get("checkpoint_selection")
    _require(
        isinstance(selection, dict)
        and selection.get("test_used") is False
        and selection.get("primary_validation") == value
        and selection.get("primary_epoch") == epoch,
        "checkpoint selection metadata mismatch",
    )
    fixed = child["condition"] == "fixed_c"
    _require(
        selection.get("primary_role")
        == ("all_epoch_prediction_best" if fixed else "c_active_mechanism_best"),
        "primary selection role mismatch",
    )
    global_metric, global_epoch = (
        child.get("global_best_validation"),
        child.get("global_best_epoch"),
    )
    _require(
        _metric(global_metric)
        and global_metric >= value
        and _integer(global_epoch, epoch_offset + 1)
        and global_epoch <= last["epoch"],
        "global selection mismatch",
    )
    _require(
        last.get("global_best_metric") == global_metric
        and last.get("global_best_epoch") == global_epoch
        and history[global_epoch - epoch_offset - 1]["validation"] == global_metric,
        "global checkpoint/history mismatch",
    )
    _require(
        selection.get("global_prediction_validation") == global_metric
        and selection.get("global_prediction_epoch") == global_epoch,
        "global selection metadata mismatch",
    )
    if fixed:
        _require(
            global_metric == value
            and global_epoch == epoch
            and child.get("joint_best_validation") is None
            and child.get("joint_best_epoch") is None,
            "fixed-C selection mismatch",
        )
    else:
        joint_metric, joint_epoch = (
            child.get("joint_best_validation"),
            child.get("joint_best_epoch"),
        )
        _require(
            _metric(joint_metric)
            and _integer(joint_epoch, epoch_offset + 1)
            and joint_epoch <= last["epoch"],
            "joint selection mismatch",
        )
        _require(
            last.get("joint_best_metric") == joint_metric
            and last.get("joint_best_epoch") == joint_epoch
            and history[joint_epoch - epoch_offset - 1]["validation"] == joint_metric,
            "joint checkpoint/history mismatch",
        )
    for record in (last, best):
        _require(
            record.get("transition_provenance") == child.get("transition_provenance"),
            "checkpoint transition provenance mismatch",
        )
    previous = output / "best.previous.pt"
    if previous.exists():
        _require(previous.is_file(), "invalid previous checkpoint path")
        artifacts["best_previous_checkpoint"] = {"path": str(previous), "sha256": _digest(previous)}
    _require(_digest(path) == digest, "metrics changed during verification")
    for artifact in artifacts.values():
        _require(
            _digest(Path(artifact["path"])) == artifact["sha256"],
            "artifact changed during verification",
        )
    summary = {
        "status": "verified",
        "dataset": child["dataset"],
        "condition": child["condition"],
        "model_seed": child["model_seed"],
        "metrics_path": str(path),
        "metrics_sha256": digest,
        "artifacts": artifacts,
        "validation": value,
        "metric_name": child["metric_name"],
        "best_epoch": epoch,
        "epochs_completed": last["epoch"],
        "epochs_in_history": len(history),
        "configuration": config,
        "schedule": child["schedule"],
        "protocol": protocol,
        "cache_sha256": child["cache_sha256"],
        "source_sha256": child["source_sha256"],
        "runtime_versions": child["versions"],
        "comparison_design": design,
        "initial_state_sha256": child["initial_state_sha256"],
        "shared_initial_state_sha256": child["shared_initial_state_sha256"],
        "resume_identity_sha256": child["resume_identity_sha256"],
        "hardware_execution": hardware,
        "resource_observability": resource,
        "throughput": throughput,
        "fresh_same_initialization_comparison": False,
        "test_evaluated": False,
    }
    return copy.deepcopy(child), copy.deepcopy(summary)


def validate_historical_reference(
    source_job: dict[str, Any], source_manifest: dict[str, Any], *, source_manifest_path: Path
) -> dict[str, Any]:
    """Verify a completed original result without treating it as a new training."""
    from scripts.run_conductance_scaling import _load_child

    manifest_path = Path(source_manifest_path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    _require(
        json.loads(raw) == source_manifest,
        "source manifest changed or differs from supplied manifest",
    )
    _require(
        source_job in source_manifest.get("jobs", []),
        "historical job is absent from source manifest",
    )
    _require(
        source_job.get("status") == "passed", "historical reference requires a completed passed job"
    )
    _require(_sha(source_job.get("metrics_sha256")), "historical metrics hash is missing")
    # This validates the *stored* architecture/execution contract and telemetry;
    # it does not apply today's comparison-design or default-backend constants.
    actual_result = _load_child(source_job)
    _require(source_job.get("result") == actual_result, "historical saved result mismatch")
    child, reference = _verified_child(source_job)
    _require(
        child.get("transition_provenance") is None, "historical source is already a transition"
    )
    snapshot = source_manifest.get("source_sha256")
    _require(_source_map(snapshot), "missing source manifest hash inventory")
    candidates = [snapshot]
    evidence = source_manifest.get("source_compatibility", [])
    _require(isinstance(evidence, list), "invalid historical source compatibility evidence")
    for item in evidence:
        _require(isinstance(item, dict), "invalid historical source compatibility evidence")
        previous, current = item.get("previous_source_sha256"), item.get("current_source_sha256")
        _require(
            _source_map(previous) and _source_map(current) and snapshots_match(previous, current),
            "unverified historical source compatibility evidence",
        )
        if all(snapshot.get(name) == value for name, value in current.items()):
            candidates.append(previous)
    _require(
        any(
            all(candidate.get(name) == value for name, value in child["source_sha256"].items())
            for candidate in candidates
        ),
        "historical child/source manifest mismatch",
    )
    reference.update(
        role="historical_reference",
        original_condition=child["condition"],
        source_job_id=source_job.get("job_id"),
        source_manifest_path=str(manifest_path),
        source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        newly_trained_epochs=0,
        counts_as_new_solver_completion=False,
        verification_scope=(
            "original artifact hashes and recorded data/split/recipe/source/runtime identity; "
            "no rerun"
        ),
    )
    _require(manifest_path.read_bytes() == raw, "source manifest changed during verification")
    return reference


def validate_transition_child(job: dict[str, Any]) -> dict[str, Any]:
    """Validate a completed new-output child without inventing a paired contrast."""
    from .train import build_parser, configuration, validate_args

    action = job.get("action")
    _require(
        action in {"transition_dynamic", "resume_incomplete_fixed", "fresh_dynamic", "fresh_fixed"},
        "invalid training action",
    )
    command = job.get("resolved_command", job.get("command"))
    _require(
        isinstance(command, list)
        and "-m" in command
        and command[command.index("-m") + 1] == "research.conductance_gat.v5.train",
        "missing resolved V5 training command",
    )
    args = build_parser().parse_args(command[command.index("-m") + 2 :])
    validate_args(args)
    epoch_offset = job.get("source_epoch", 0) if action == "transition_dynamic" else 0
    child, result = _verified_child(job, epoch_offset=epoch_offset)
    _require(
        child["configuration"] == configuration(args), "resolved command/configuration mismatch"
    )
    _require(
        Path(args.output_dir).expanduser().resolve()
        == Path(job["output_dir"]).expanduser().resolve(),
        "resolved command output mismatch",
    )
    provenance = child.get("transition_provenance")
    if action in {"transition_dynamic", "resume_incomplete_fixed"}:
        _require(isinstance(provenance, dict), "missing transition provenance")
        mode = "replace_c" if action == "transition_dynamic" else "continue_fixed"
        _require(provenance.get("mode") == mode, "transition mode mismatch")
        _require(
            provenance.get("source_checkpoint_sha256") == job.get("source_checkpoint_sha256")
            and _sha(job.get("source_checkpoint_sha256")),
            "transition source checkpoint hash mismatch",
        )
        _require(
            provenance.get("source_epoch") == job.get("source_epoch"),
            "transition source epoch mismatch",
        )
        _require(provenance.get("epoch_offset") == epoch_offset, "transition epoch offset mismatch")
        _require(
            provenance.get("target_total_epochs") == child["configuration"]["epochs"],
            "transition epoch budget mismatch",
        )
        source_identity = provenance.get("source_identity")
        _require(
            isinstance(source_identity, dict)
            and provenance.get("source_identity_sha256") == _canonical(source_identity),
            "transition source identity hash mismatch",
        )
        for source_key, child_key in (
            ("dataset", "dataset"),
            ("condition", "condition"),
            ("dataset_protocol", "protocol"),
            ("cache_sha256", "cache_sha256"),
            ("runtime_versions", "versions"),
        ):
            _require(
                source_identity.get(source_key) == child[child_key],
                "transition data/split/runtime mismatch",
            )
        source_config = source_identity.get("configuration", {})
        original_budget, extra_budget = (
            provenance.get("source_epochs_requested"),
            provenance.get("additional_epochs"),
        )
        _require(
            _integer(original_budget, 1)
            and _integer(extra_budget)
            and original_budget == source_config.get("epochs")
            and original_budget + extra_budget == child["configuration"]["epochs"],
            "transition original/additional epoch budget mismatch",
        )
        _require(
            provenance.get("historical_metrics_are_new_c_metrics") is False
            and provenance.get("test_labels_used") is False,
            "invalid transition metric provenance claims",
        )
        source_path_value = job.get("source_checkpoint")
        _require(isinstance(source_path_value, str), "missing transition source checkpoint path")
        source_path = Path(source_path_value).expanduser().resolve()
        _require(
            source_path.is_file()
            and _digest(source_path) == provenance["source_checkpoint_sha256"]
            and str(source_path) == provenance.get("source_path"),
            "transition source checkpoint path/hash changed",
        )
        if action == "resume_incomplete_fixed":
            _require(
                provenance.get("source_complete") is False,
                "completed fixed-C must not be retrained",
            )
            _require(
                child["condition"] == "fixed_c"
                and child["configuration"].get("conductance_backend") == "mlp"
                and child["configuration"].get("training_schedule") == "staged",
                "fixed continuation must preserve the historical recipe",
            )
        _require(
            result["epochs_completed"] > job["source_epoch"],
            "transition has no newly completed epoch",
        )
        if action == "transition_dynamic":
            _require(
                child["condition"] == "shared_dynamic_c"
                and child["configuration"].get("conductance_backend") == "optimization"
                and child["configuration"].get("training_schedule") == "joint",
                "transition must train new joint optimization C",
            )
    else:
        _require(provenance is None, "fresh training has transition provenance")
    result.update(
        role="transitioned_training" if provenance is not None else "fresh_training",
        action=action,
        transition_provenance=copy.deepcopy(provenance),
        pre_transition_epochs=job.get("source_epoch", 0),
        post_transition_epochs=result["epochs_completed"] - job.get("source_epoch", 0),
        counts_as_new_solver_completion=child["condition"] == "shared_dynamic_c"
        and child["configuration"].get("conductance_backend") == "optimization",
    )
    return result


def build_transition_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize distinct histories; deliberately provide no causal paired delta."""
    rows, pending = [], []
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list), "transition manifest jobs are missing")
    for job in jobs:
        action = job.get("action")
        if action in {"reuse_completed_fixed", "preserve_legacy_dynamic"}:
            reference = job.get("historical_reference")
            _require(
                isinstance(reference, dict)
                and reference.get("role") == "historical_reference"
                and reference.get("status") == "verified",
                "missing verified historical reference",
            )
            source_path = Path(reference["source_manifest_path"])
            _require(
                _digest(source_path) == reference.get("source_manifest_sha256"),
                "source manifest hash changed",
            )
            source = _json(source_path)
            source_job = job.get("source_job")
            if not isinstance(source_job, dict):
                matches = [
                    item
                    for item in source.get("jobs", [])
                    if item.get("job_id") == reference.get("source_job_id")
                ]
                _require(len(matches) == 1, "historical source job is ambiguous")
                source_job = matches[0]
            actual = validate_historical_reference(
                source_job, source, source_manifest_path=source_path
            )
            _require(actual == reference, "historical reference changed")
            expected_condition = (
                "fixed_c" if action == "reuse_completed_fixed" else "shared_dynamic_c"
            )
            _require(
                actual["condition"] == expected_condition, "historical action/condition mismatch"
            )
            rows.append({**actual, "action": action, "profile": job.get("profile")})
            if action == "preserve_legacy_dynamic":
                pending.append(
                    {
                        "job_id": job.get("job_id"),
                        "reason": (
                            "new C has not trained; explicit additional epoch budget required"
                        ),
                    }
                )
        elif job.get("status") == "passed":
            rows.append({**validate_transition_child(job), "profile": job.get("profile")})
        else:
            pending.append(
                {
                    "job_id": job.get("job_id"),
                    "action": action,
                    "status": job.get("status"),
                    "reason": job.get("error"),
                }
            )
    return {
        "schema_version": 1,
        "research_suite": SUITE,
        "report_kind": "selective_transition_provenance",
        "status": "partial" if pending else "passed",
        "rows": rows,
        "incomplete_new_training": pending,
        "comparison_design": {
            "historical_references_are_new_solver_results": False,
            "fresh_same_initialization_paired_comparison": False,
            "single_factor_causal_effect_of_c": False,
            "sota_claim": False,
            "test_evaluated": False,
            "interpretation": (
                "historical references and continued/warm-start training have different "
                "optimization histories and budgets; validation scores are descriptive, "
                "not a fresh paired causal comparison"
            ),
        },
        "new_solver_completions": sum(
            bool(row.get("counts_as_new_solver_completion")) for row in rows
        ),
    }
