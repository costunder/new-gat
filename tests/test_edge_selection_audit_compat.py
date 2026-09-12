"""CPU-only source/driver controls, not GPU training or measured calibration."""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.edge_selection import audit, audit_compat, train
from scripts import run_v5_edge_selection as driver


def sources(scope, base_commit=audit_compat.BASE_COMMIT):
    current = (
        driver.resources.source_snapshot()
        if scope == "manifest"
        else train.implementation_source_hashes()
    )
    registry, _ = audit_compat._registry(base_commit)
    previous = copy.deepcopy(current)
    for name, change in registry["changes"].items():
        if change["before"] is None:
            previous.pop(name)
        else:
            previous[name] = change["before"]
    return previous, current


@pytest.mark.parametrize("scope", ["manifest", "training"])
@pytest.mark.parametrize("base_commit", audit_compat.BASE_COMMITS)
def test_exact_pinned_release_passes_without_relabeling(scope, base_commit):
    previous, current = sources(scope, base_commit)
    before = copy.deepcopy(previous)
    proof = audit_compat.require_source_compatibility(previous, current, scope=scope)
    assert proof["patch_id"] == audit_compat.PATCH_ID
    assert proof["base_commit"] == base_commit
    assert proof["training_artifacts_rewritten"] is False
    assert proof["previous_source_map_sha256"] != proof["current_source_map_sha256"]
    assert previous == before
    assert audit_compat.require_source_compatibility(current, current, scope=scope) is None


@pytest.mark.parametrize(
    "damage", ["model", "missing", "added", "partial_patch", "helper", "old_release", "scope"]
)
def test_no_unregistered_source_change_is_accepted(damage):
    previous, current = sources("training")
    if damage == "model":
        current["research/conductance_gat/edge_selection/model.py"] = "0" * 64
    elif damage == "missing":
        current.pop("research/conductance_gat/edge_selection/model.py")
    elif damage == "added":
        current["research/conductance_gat/edge_selection/extra.py"] = "0" * 64
    elif damage == "partial_patch":
        name = "research/conductance_gat/edge_selection/diagnostics.py"
        current[name] = previous[name]
    elif damage == "helper":
        current[audit_compat.HELPER_SOURCE] = "0" * 64
    elif damage == "old_release":
        previous["research/conductance_gat/edge_selection/model.py"] = "0" * 64
    with pytest.raises(ValueError):
        audit_compat.require_source_compatibility(
            previous, current, scope="manifest" if damage == "scope" else "training"
        )


def test_live_pins_are_linux_lf_bytes_not_windows_normalized_claims():
    registry, _ = audit_compat._registry()
    previous, current = sources("training")
    for name, pins in registry["changes"].items():
        raw = (audit_compat.ROOT / name).read_bytes()
        assert b"\r\n" not in raw
        assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == pins["after"]
    raw = (audit_compat.ROOT / audit_compat.HELPER_SOURCE).read_bytes()
    current[audit_compat.HELPER_SOURCE] = hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()
    with pytest.raises(ValueError, match="pinned live"):
        audit_compat.require_source_compatibility(previous, current, scope="training")


def test_registry_is_exact_and_rejects_duplicate_keys(tmp_path, monkeypatch):
    registry = json.loads(audit_compat.REGISTRY_PATH.read_text())
    target = tmp_path / "debug-registry.json"
    target.write_text(
        json.dumps(registry).replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
    )
    monkeypatch.setattr(audit_compat, "REGISTRY_PATH", target)
    with pytest.raises(ValueError, match="duplicate"):
        audit_compat._registry()
    registry["releases"][0]["changes"]["research/conductance_gat/edge_selection/model.py"] = {
        "before": "0" * 64,
        "after": "1" * 64,
    }
    target.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="exact change set"):
        audit_compat._registry()


@pytest.mark.parametrize("base_commit", audit_compat.BASE_COMMITS)
def test_partial_checkpoint_accepts_only_exact_source_repair(base_commit):
    previous, current = sources("training", base_commit)
    identity = {
        "research_suite": train.SUITE,
        "configuration": {"synthetic": "debug-identity-only"},
        "source_sha256": previous,
        "training_arguments": {"device": "cuda:0", "seed": 0},
        "learning_budget": {"epochs": 200},
    }
    saved = {
        "resume_identity": identity,
        "resume_identity_sha256": train.base._canonical_sha256(identity),
    }
    snapshot = copy.deepcopy(saved)
    expected = {**identity, "source_sha256": current}
    original, proof = train.resolve_training_resume(saved, expected)
    assert original == identity and original is not identity
    assert proof["base_commit"] == base_commit
    assert saved == snapshot
    for key in ("research_suite", "configuration", "training_arguments", "learning_budget"):
        damaged = {**expected, key: "changed"}
        with pytest.raises(ValueError, match="identity mismatch"):
            train.resolve_training_resume(saved, damaged)
    damaged_sources = {**current, "unexpected.py": "a" * 64}
    with pytest.raises(ValueError, match="unreviewed"):
        train.resolve_training_resume(saved, {**expected, "source_sha256": damaged_sources})


def fake_result():
    return {
        "validation": 0.721098,
        "best_epoch": 6,
        "epochs_run": 56,
        "shared_initial_state_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "learning_budget": {"explicit_debug_fixture": True},
        "checkpoint_sha256": "d" * 64,
    }


def test_repaired_resume_skips_training_and_measurement_retries_only_failed_audit(
    tmp_path, monkeypatch
):
    previous, current = sources("manifest")
    options = driver.parser().parse_args(
        [
            "--run-id",
            "debug-source-repair",
            "--datasets",
            "cora",
            "--profiles",
            "reference",
            "--suites",
            "structure",
            "--chord-fractions",
            "0.5",
        ]
    )
    planned = driver.make_jobs(options, tmp_path)
    jobs = copy.deepcopy(planned)
    result = fake_result()
    failed_bytes = b"original explicit synthetic failed audit"
    old_passed = None
    for index in (0, 1):
        jobs[index].update(status="passed", result=copy.deepcopy(result))
        log = tmp_path / f"old-audit-{index}.log"
        log.write_bytes(failed_bytes if index else b"old passed audit source retained")
        # Use the actual command the controller would launch, without executing it.
        monkeypatch.setattr(driver.standalone.shared, "run_logged", lambda *_: 0)
        monkeypatch.setattr(driver.standalone, "_next_log", lambda _, target=log: target)
        driver._audit(options, jobs[index], {}, lambda: None)
        jobs[index]["audit"].pop("evaluator_source_sha256")
        if index:
            jobs[index]["audit"]["status"] = "failed"
        else:
            old_passed = copy.deepcopy(jobs[index]["audit"])
    monkeypatch.undo()
    entries = [{"status": "passed", "explicit_debug_fixture": "immutable measurement"}]
    manifest = {
        "schema_version": 1,
        "suite": driver.SUITE,
        "run_id": options.run_id,
        "status": "failed",
        "config": driver._config(options),
        "source_sha256": previous,
        "dependencies": {},
        "planned_jobs": planned,
        "jobs": jobs,
        "calibration_entries": copy.deepcopy(entries),
        "resources_applied": True,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    checked_calibration, training, audited = [], [], []

    def calibration_check(_args, saved, persist):
        assert saved["calibration_entries"] == entries
        checked_calibration.append(True)

    def dispatch(command, log, _environment):
        if "--root" in command:
            audited.append(command[command.index("--root") + 1])
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("new synthetic audit with explicit evaluator provenance")
        else:
            training.append(command[command.index("--output-dir") + 1])
        return 0

    monkeypatch.setattr(driver.resources, "source_snapshot", lambda: current)
    monkeypatch.setattr(driver, "_ensure_calibration", calibration_check)
    monkeypatch.setattr(driver, "_read_result", lambda _: copy.deepcopy(result))
    monkeypatch.setattr(driver.standalone.shared, "run_logged", dispatch)
    assert driver._run(options, tmp_path, planned, current, {}) == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["source_sha256"] == previous
    assert len(saved["source_transitions"]) == 1
    assert saved["calibration_entries"] == entries
    assert saved["jobs"][0]["audit"] == old_passed
    assert Path(jobs[1]["audit"]["log_path"]).read_bytes() == failed_bytes
    assert saved["jobs"][1]["audit_attempts"][0] == jobs[1]["audit"]
    assert saved["jobs"][1]["result"] == result
    assert jobs[0]["output_dir"] not in training + audited
    assert jobs[1]["output_dir"] not in training
    assert len(training) == len(planned) - 2 and len(audited) == len(planned) - 1
    assert checked_calibration == [True]
    assert driver._run(options, tmp_path, planned, current, {}) == 0
    assert len(training) == len(planned) - 2 and len(audited) == len(planned) - 1


def test_audit_reports_actual_evaluator_separately_from_training(tmp_path, monkeypatch):
    previous, current = sources("training")
    identity = {"source_sha256": previous, "dataset_protocol": {}, "input_provenance": ["debug"]}
    metrics = {"resume_identity": identity, "checkpoint_sha256": "a" * 64}
    model = torch.nn.Linear(2, 2)
    model.operators = []
    model.clear_auxiliary_cache = lambda: None
    args = SimpleNamespace(
        model_seed=0, dataset="cora", data_root=tmp_path, corruption_ratio=0, selection_mode="full"
    )
    checkpoint = {
        "model_state": copy.deepcopy(model.state_dict()),
        "resume_identity": identity,
        "resume_identity_sha256": train.base._canonical_sha256(identity),
    }
    monkeypatch.setattr(train, "inspect_completed", lambda _: copy.deepcopy(metrics))
    monkeypatch.setattr(train, "restore_arguments", lambda *_: args)
    monkeypatch.setattr(train.base, "_require_cuda", lambda _: None)
    monkeypatch.setattr(train.base, "configure_compute", lambda _: None)
    monkeypatch.setattr(train.base, "load_dataset", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(train, "PreparedInputs", lambda *_: SimpleNamespace(provenance=["debug"]))
    monkeypatch.setattr(train, "make_model", lambda *_: model)
    monkeypatch.setattr(train.base, "load_checkpoint_on_cpu", lambda _: checkpoint)
    monkeypatch.setattr(audit, "_isolated_execution_state", lambda _: nullcontext())
    monkeypatch.setattr(audit, "release_amplitude_diagnostics", lambda _: None)
    monkeypatch.setattr(train, "evaluate", lambda *_args, **_kwargs: {"metric": 0.5})
    monkeypatch.setattr(audit, "Observer", lambda *_: SimpleNamespace(report=lambda _: {}))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 0)
    monkeypatch.setattr(
        audit,
        "RuntimeResourceMonitor",
        lambda _: SimpleNamespace(
            start=lambda: None, finish=lambda **_: {"explicit_debug_fixture": True}
        ),
    )
    model.selection_metadata = lambda: {}
    report = audit.audit(tmp_path, tmp_path, torch.device("cpu"), 5, 32)
    assert report["training_source_sha256"] == previous
    assert report["source_sha256"] == previous
    assert report["evaluator_source_sha256"] == current
    assert report["source_compatibility"]["patch_id"] == audit_compat.PATCH_ID
    assert report["repeated_validation"]["count"] == 5
    assert list(tmp_path.iterdir()) == []


def test_result_reader_accepts_pinned_old_source_without_changing_result_schema(
    tmp_path, monkeypatch
):
    previous, current = sources("training")
    options = driver.parser().parse_args(
        [
            "--run-id",
            "debug-read",
            "--datasets",
            "cora",
            "--profiles",
            "reference",
            "--suites",
            "structure",
            "--chord-fractions",
            "0.5",
        ]
    )
    job = driver.make_jobs(options, tmp_path)[0]
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    history = [
        {"epoch": i, "validation": 0.721098 if i == 6 else 0.7, "optimizer_steps": i}
        for i in range(1, 57)
    ]
    (output / "history.json").write_text(json.dumps(history))
    (output / "last.pt").write_bytes(
        b"explicit CPU controller fixture: integrity separately tested"
    )
    (output / "best.pt").write_bytes(b"explicit CPU controller selected checkpoint fixture")
    original = {
        name: (output / name).read_bytes() for name in ("history.json", "last.pt", "best.pt")
    }
    protocol = {"data_sha256": "b" * 64, "split_sha256": "c" * 64}
    identity = {
        "research_suite": train.SUITE,
        "dataset": "cora",
        "condition": "full",
        "configuration": train.configuration(driver.calibration.parse_job(job)),
        "source_sha256": previous,
        "dataset_protocol": protocol,
        "dataset_protocol_sha256": train.base._canonical_sha256(protocol),
    }
    payload = {
        **identity,
        "status": "passed",
        "resume_identity": identity,
        "resume_identity_sha256": train.base._canonical_sha256(identity),
        "protocol": protocol,
        "test_evaluated": False,
        "epochs_run": 56,
        "best_epoch": 6,
        "best_validation": 0.721098,
        "optimizer_steps": 56,
        "shared_initial_state_sha256": "a" * 64,
        "learning_budget": {"explicit_debug_fixture": True},
        "checkpoint_sha256": driver.common._file_sha(output / "best.pt"),
        "last_checkpoint_sha256": driver.common._file_sha(output / "last.pt"),
        "history_sha256": driver.common._file_sha(output / "history.json"),
    }
    monkeypatch.setattr(train, "inspect_completed", lambda _: copy.deepcopy(payload))
    result = driver._read_result(job)
    assert result == {
        "validation": 0.721098,
        "best_epoch": 6,
        "epochs_run": 56,
        "shared_initial_state_sha256": "a" * 64,
        **protocol,
        "learning_budget": payload["learning_budget"],
        **{
            key: payload[key]
            for key in ("checkpoint_sha256", "last_checkpoint_sha256", "history_sha256")
        },
    }
    assert {name: (output / name).read_bytes() for name in original} == original
    current["research/conductance_gat/edge_selection/model.py"] = "0" * 64
    monkeypatch.setattr(train, "implementation_source_hashes", lambda: current)
    with pytest.raises(ValueError, match="unreviewed"):
        driver._read_result(job)
