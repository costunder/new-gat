"""CPU-only synthetic certificate tests; no GPU measurement or training is claimed."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.conductance_gat.v5 import train
from research.conductance_gat.v5 import transition_training as transition
from scripts import calibrate_training_resources as calibration
from scripts import training_resource_plan as plans

GIB = 1024**3


def _measurement(batch, *, workers=0, condition="shared_dynamic_c", rate=100):
    return {
        "status": "passed",
        "condition": condition,
        "model_seed": 0,
        "batch_size": batch,
        "workers": workers,
        "unit": "supervised_seed_nodes",
        "elapsed_seconds": 3.0,
        "processed_units": rate * 3,
        "samples_per_second": rate,
        "optimizer_steps": 5,
        "optimizer_state_bytes": 1024,
        "measurement_steps_requested": 5,
        "warmup_steps_requested": 2,
        "minimum_measure_seconds_requested": 3.0,
        "peak_allocated_bytes": 8 * GIB,
        "peak_reserved_bytes": 10 * GIB,
        "total_memory_bytes": 48 * GIB,
        "free_bytes_before": 46 * GIB,
    }


def _candidate(batch, *, condition="shared_dynamic_c", rate=100):
    return {
        "status": "passed",
        "batch_size": batch,
        "workers": 0,
        "measurements": [_measurement(batch, condition=condition, rate=rate)],
    }


def _fixture(tmp_path, monkeypatch, *, mode="replace_c"):
    """Real parsing/score/selection, explicitly mocked hardware/runtime and evidence."""
    source = tmp_path / "original" / "last.pt"
    source.parent.mkdir()
    source.write_bytes(b"explicit CPU certificate fixture, not a trained checkpoint")
    path = tmp_path / "resource-certificate.json"
    fixed = mode == "continue_fixed"
    args = train.build_parser().parse_args(
        [
            "--dataset",
            "ogbn-arxiv",
            "--condition",
            "fixed_c" if fixed else "shared_dynamic_c",
            "--output-dir",
            str(tmp_path / "new-output"),
            "--device",
            "cuda:0",
            "--conductance-backend",
            "mlp" if fixed else "optimization",
            "--training-schedule",
            "staged" if fixed else "joint",
            "--sampling",
            "neighbor",
            "--sample-seed-batch-size",
            "2048" if fixed else "4096",
            "--workers",
            "0",
            "--transition-from-checkpoint",
            str(source),
            "--transition-source-sha256",
            "a" * 64,
            "--transition-mode",
            mode,
            "--transition-resource-certificate",
            str(path),
            "--transition-resource-sha256",
            "b" * 64,
        ]
    )
    train.validate_args(args)
    target = train.configuration(args)
    original = copy.deepcopy(target)
    original["sample_seed_batch_size"] = 2048
    protocol = {
        "data_sha256": "c" * 64,
        "split": "explicit synthetic CPU fixture",
        "split_counts": {"train": 4096, "validation": 1024, "test": 1024},
    }
    runtime = {"torch": "synthetic CPU contract fixture", "cuda": "not measured"}
    sources = {"explicit-unit-fixture.py": "d" * 64}
    hardware = {
        "device": "cuda:0",
        "name": "explicit CPU mock, NOT actual GPU measurement",
        "uuid": "unit-fixture",
        "total_memory_bytes": 48 * GIB,
        "allocated_cpu_count": 8,
    }
    monkeypatch.setattr(train, "_versions", lambda: copy.deepcopy(runtime))
    monkeypatch.setattr(plans, "source_snapshot", lambda: copy.deepcopy(sources))
    monkeypatch.setattr(calibration, "_hardware", lambda device: copy.deepcopy(hardware))
    inspected = {
        "sha256": "a" * 64,
        "source_epoch": 40,
        "identity": {"configuration": original},
    }
    certificate = {
        "schema_version": 1,
        "kind": "v5_transition_resource_certificate",
        "status": "passed",
        "classification": "resource_calibration_not_final_training",
        "source_checkpoint_sha256": inspected["sha256"],
        "transition_mode": mode,
        "source_epoch": inspected["source_epoch"],
        "runtime_versions": runtime,
        "dataset_protocol_sha256": train._canonical_sha256(protocol),
        "cache_sha256": protocol["data_sha256"],
        "source_sha256": sources,
        "hardware": hardware,
        "selected_configuration": copy.deepcopy(target),
        "baseline_execution": {
            key: original[key] for key in ("batch_size", "workers", "sample_seed_batch_size")
        },
        "selected_execution": {
            key: target[key] for key in ("batch_size", "workers", "sample_seed_batch_size")
        },
        "selection": {
            "policy": "preserve_fixed_execution" if fixed else "measured_transition_candidates",
            "no_downscale": True,
            "global_optimum_claimed": False,
        },
        "batch_axis": "sampled_seed_nodes",
        "natural_training_split_size": 4096,
        "selected_candidate": {"batch_size": target["sample_seed_batch_size"], "workers": 0},
        "candidates": [_candidate(2048, condition=args.condition, rate=100)],
    }
    if not fixed:
        certificate["candidates"].append(_candidate(4096, condition=args.condition, rate=150))
    _publish(args, certificate)
    return args, inspected, protocol, target, certificate


def _publish(args, certificate):
    args.transition_resource_certificate.write_text(json.dumps(certificate), encoding="utf-8")
    args.transition_resource_sha256 = hashlib.sha256(
        args.transition_resource_certificate.read_bytes()
    ).hexdigest()


def _read(fixture):
    args, inspected, protocol, target, _ = fixture
    return transition._read_certificate(args, inspected, protocol, target)


def _snapshot(root):
    return {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_valid_cpu_contract_fixture_uses_real_safety_score_and_selection(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args, _, _, _, certificate = fixture
    before = _snapshot(tmp_path)
    validated, changes = _read(fixture)
    assert validated == certificate
    assert changes == {"sample_seed_batch_size": {"before": 2048, "after": 4096}}
    assert plans.choose_candidate(certificate["candidates"], 2048)["batch_size"] == 4096
    assert (
        transition.request_from_args(args)["resource_certificate_sha256"]
        == args.transition_resource_sha256
    )
    assert _snapshot(tmp_path) == before


def test_bad_certificate_file_hash_rejected_without_writes(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[0].transition_resource_sha256 = "f" * 64
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="path/SHA"):
        _read(fixture)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_checkpoint_sha256", "f" * 64),
        ("source_epoch", 39),
        ("runtime_versions", {"torch": "different"}),
        ("source_sha256", {"changed.py": "f" * 64}),
        ("hardware", {"device": "cuda:1"}),
        ("dataset_protocol_sha256", "f" * 64),
        ("cache_sha256", "f" * 64),
        ("transition_mode", "continue_fixed"),
        ("status", "running"),
        ("classification", "final_training"),
    ],
)
def test_certificate_is_bound_to_source_split_runtime_code_and_hardware(
    tmp_path, monkeypatch, field, value
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[4][field] = value
    _publish(fixture[0], fixture[4])
    with pytest.raises(ValueError, match="contract mismatch"):
        _read(fixture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("condition", "fixed_c"),
        ("model_seed", 1),
        ("batch_size", 17),
        ("workers", 3),
        ("samples_per_second", float("nan")),
        ("optimizer_state_bytes", 0),
        ("measurement_steps_requested", 6),
        ("minimum_measure_seconds_requested", 4.0),
    ],
)
def test_each_measurement_has_exact_condition_seed_batch_workers_and_real_window(
    tmp_path, monkeypatch, field, value
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[4]["candidates"][1]["measurements"][0][field] = value
    _publish(fixture[0], fixture[4])
    with pytest.raises(ValueError):
        _read(fixture)


@pytest.mark.parametrize("replacement", [[], None])
def test_missing_measurement_is_never_accepted_as_calibration(tmp_path, monkeypatch, replacement):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[4]["candidates"][1]["measurements"] = replacement
    _publish(fixture[0], fixture[4])
    with pytest.raises(ValueError, match="no measurements"):
        _read(fixture)


def test_unsafe_memory_headroom_is_rejected_even_when_throughput_is_best(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[4]["candidates"][1]["measurements"][0]["peak_reserved_bytes"] = 45 * GIB
    _publish(fixture[0], fixture[4])
    with pytest.raises(ValueError, match="safe throughput"):
        _read(fixture)


def test_picking_slower_candidate_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args, _, _, target, certificate = fixture
    args.sample_seed_batch_size = target["sample_seed_batch_size"] = 2048
    certificate["selected_execution"]["sample_seed_batch_size"] = 2048
    certificate["selected_configuration"] = copy.deepcopy(target)
    certificate["selected_candidate"]["batch_size"] = 2048
    _publish(args, certificate)
    with pytest.raises(ValueError, match="safe throughput"):
        _read(fixture)


def test_batch_downscale_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[4]["candidates"].insert(0, _candidate(1024))
    _publish(fixture[0], fixture[4])
    with pytest.raises(ValueError, match="out-of-contract"):
        _read(fixture)


def test_only_one_batch_candidate_rejected_when_graph_can_grow(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args, _, _, target, certificate = fixture
    args.sample_seed_batch_size = target["sample_seed_batch_size"] = 2048
    certificate["selected_configuration"] = copy.deepcopy(target)
    certificate["selected_execution"]["sample_seed_batch_size"] = 2048
    certificate["selected_candidate"]["batch_size"] = 2048
    certificate["candidates"] = certificate["candidates"][:1]
    _publish(args, certificate)
    with pytest.raises(ValueError, match="multiple measured"):
        _read(fixture)


@pytest.mark.parametrize("axis", ["graphs", "full_graph", None])
def test_wrong_physical_batch_axis_rejected(tmp_path, monkeypatch, axis):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[4]["batch_axis"] = axis
    _publish(fixture[0], fixture[4])
    with pytest.raises(ValueError, match="physical batch axis"):
        _read(fixture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("solver_steps", 80),
        ("solver_entropy", 2.0),
        ("hidden_channels", 512),
        ("precision", "fp32"),
        ("num_neighbors", [25, 20]),
    ],
)
def test_certificate_cannot_be_reused_for_unmeasured_scientific_configuration(
    tmp_path, monkeypatch, field, value
):
    fixture = _fixture(tmp_path, monkeypatch)
    target = fixture[3]
    if target.get(field) == value:
        value = "bf16" if field == "precision" else value
    target[field] = value
    with pytest.raises(ValueError, match="contract mismatch"):
        _read(fixture)


def test_fixed_continuation_preserves_execution_with_one_measured_batch(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, mode="continue_fixed")
    _, changes = _read(fixture)
    assert changes == {}


def test_fixed_continuation_cannot_adopt_changed_workers(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch, mode="continue_fixed")
    args, _, _, target, certificate = fixture
    args.workers = target["workers"] = target["loader_workers"] = 2
    certificate["selected_configuration"] = copy.deepcopy(target)
    certificate["selected_execution"]["workers"] = 2
    certificate["selected_candidate"]["workers"] = 2
    baseline = copy.deepcopy(certificate["candidates"][0])
    candidate = certificate["candidates"][0]
    candidate["workers"] = candidate["measurements"][0]["workers"] = 2
    candidate["measurements"][0].update(samples_per_second=200, processed_units=600)
    certificate["candidates"].append(baseline)
    _publish(args, certificate)
    with pytest.raises(ValueError, match="fixed continuation cannot change"):
        _read(fixture)


def test_certificate_change_during_validation_detected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    original = train.sha256_file
    count = 0

    def changed_second_read(path):
        nonlocal count
        count += 1
        return original(path) if count == 1 else "f" * 64

    monkeypatch.setattr(train, "sha256_file", changed_second_read)
    with pytest.raises(ValueError, match="changed during validation"):
        _read(fixture)


def test_no_resume_and_incomplete_transition_options_rejected(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)[0]
    args.resume = False
    with pytest.raises(ValueError, match="no-resume"):
        transition.validate_arguments(args)
    args.resume = True
    args.transition_resource_certificate = None
    with pytest.raises(ValueError, match="requires source SHA"):
        transition.validate_arguments(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("transition_extra_epochs", -1),
        ("transition_extra_epochs", True),
        ("transition_source_sha256", "A" * 64),
        ("transition_mode", "unknown"),
        ("condition", "fixed_c"),
        ("conductance_backend", "mlp"),
        ("training_schedule", "staged"),
    ],
)
def test_invalid_transition_arguments_fail_before_training(tmp_path, monkeypatch, field, value):
    args = _fixture(tmp_path, monkeypatch)[0]
    setattr(args, field, value)
    with pytest.raises(ValueError):
        transition.validate_arguments(args)


def test_transition_flags_without_source_rejected(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)[0]
    args.transition_from_checkpoint = None
    with pytest.raises(ValueError, match="require --transition-from"):
        transition.validate_arguments(args)


@pytest.mark.parametrize("location", ["same", "descendant", "ancestor"])
def test_output_overlap_with_source_rejected_without_artifact_writes(
    tmp_path, monkeypatch, location
):
    args = _fixture(tmp_path, monkeypatch)[0]
    source = args.transition_from_checkpoint
    output = {"same": source.parent, "descendant": source.parent / "nested", "ancestor": tmp_path}[
        location
    ]
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="separate from the source"):
        transition.validate_output_boundary(args, output)
    assert _snapshot(tmp_path) == before


def test_separate_new_output_is_allowed_without_creating_it(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)[0]
    transition.validate_output_boundary(args, args.output_dir)
    assert not args.output_dir.exists()


def test_output_symlink_rejected_without_creating_real_windows_symlinks(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)[0]
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == args.output_dir or original(path))
    with pytest.raises(ValueError, match="symlink"):
        transition.validate_output_boundary(args, args.output_dir)


def test_measurement_cannot_claim_a_larger_gpu_than_the_bound_hardware(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    measurement = fixture[4]["candidates"][1]["measurements"][0]
    measurement.update(
        total_memory_bytes=96 * GIB, free_bytes_before=90 * GIB, peak_reserved_bytes=80 * GIB
    )
    assert plans.measurement_is_safe(measurement)  # Safe only for its *incorrect* 96 GiB claim.
    _publish(fixture[0], fixture[4])
    with pytest.raises(ValueError):
        _read(fixture)


def test_natural_split_size_cannot_be_shrunk_to_skip_multiple_candidate_measurement(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    args, _, _, target, certificate = fixture
    args.sample_seed_batch_size = target["sample_seed_batch_size"] = 2048
    certificate["selected_configuration"] = copy.deepcopy(target)
    certificate["selected_execution"]["sample_seed_batch_size"] = 2048
    certificate["selected_candidate"]["batch_size"] = 2048
    certificate["natural_training_split_size"] = 2048
    certificate["candidates"] = certificate["candidates"][:1]
    _publish(args, certificate)
    with pytest.raises(ValueError):
        _read(fixture)


def test_original_baseline_must_have_an_actual_measurement(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args, _, protocol, target, certificate = fixture
    protocol["split_counts"]["train"] = 8192
    certificate["dataset_protocol_sha256"] = train._canonical_sha256(protocol)
    certificate["natural_training_split_size"] = 8192
    args.sample_seed_batch_size = target["sample_seed_batch_size"] = 8192
    certificate["selected_configuration"] = copy.deepcopy(target)
    certificate["selected_execution"]["sample_seed_batch_size"] = 8192
    certificate["selected_candidate"]["batch_size"] = 8192
    certificate["candidates"] = [_candidate(4096, rate=100), _candidate(8192, rate=150)]
    _publish(args, certificate)
    with pytest.raises(ValueError):
        _read(fixture)


def test_real_oom_boundary_preserved_and_safe_baseline_selected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    args, _, _, target, certificate = fixture
    args.sample_seed_batch_size = target["sample_seed_batch_size"] = 2048
    certificate["selected_configuration"] = copy.deepcopy(target)
    certificate["selected_execution"]["sample_seed_batch_size"] = 2048
    certificate["selected_candidate"]["batch_size"] = 2048
    certificate["candidates"][1] = {
        "status": "oom",
        "batch_size": 4096,
        "workers": 0,
        "measurements": [
            {
                "status": "oom",
                "error": "explicit CPU fixture of a CUDA OOM exception",
                "condition": "shared_dynamic_c",
                "model_seed": 0,
                "batch_size": 4096,
                "workers": 0,
            }
        ],
    }
    _publish(args, certificate)
    validated, changes = _read(fixture)
    assert validated["candidates"][1]["measurements"][0]["error"]
    assert changes == {}


def test_output_symlink_is_rejected_even_after_main_resolves_the_target(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)[0]
    resolved_target = tmp_path / "resolved-new-output"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == args.output_dir or original(path))
    with pytest.raises(ValueError, match="symlink"):
        transition.validate_output_boundary(args, resolved_target)


def test_configured_batch_larger_than_natural_split_is_preserved_without_fake_split_size(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    args, _, protocol, target, certificate = fixture
    protocol["split_counts"]["train"] = 1024
    certificate["dataset_protocol_sha256"] = train._canonical_sha256(protocol)
    certificate["natural_training_split_size"] = 1024
    args.sample_seed_batch_size = target["sample_seed_batch_size"] = 2048
    certificate["selected_configuration"] = copy.deepcopy(target)
    certificate["selected_execution"]["sample_seed_batch_size"] = 2048
    certificate["selected_candidate"]["batch_size"] = 2048
    certificate["candidates"] = certificate["candidates"][:1]
    _publish(args, certificate)
    validated, changes = _read(fixture)
    assert validated["natural_training_split_size"] == 1024
    assert validated["selected_candidate"]["batch_size"] == 2048
    assert changes == {}
