"""Training integration for explicit checkpoint transitions, never implicit resume.

Old artifacts remain read-only. A new, provenance-bound epoch-boundary checkpoint
is published before the first update, so interruption does not restart retained W.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import torch

from chartgat.cache import atomic_write_json

from .transition import inspect_transition_source, prepare_transition_state
from .transition_initialization import reuse_initialization_artifacts


def request_from_args(args) -> dict[str, Any] | None:
    path = getattr(args, "transition_from_checkpoint", None)
    if path is None:
        return None
    return {
        "source_path": str(path.expanduser().resolve()),
        "source_checkpoint_sha256": args.transition_source_sha256,
        "mode": args.transition_mode,
        "additional_epochs": args.transition_extra_epochs,
        "resource_certificate_path": str(
            args.transition_resource_certificate.expanduser().resolve()
        ),
        "resource_certificate_sha256": args.transition_resource_sha256,
    }


def validate_arguments(args) -> None:
    source = getattr(args, "transition_from_checkpoint", None)
    other = (
        getattr(args, "transition_source_sha256", None),
        getattr(args, "transition_mode", None),
        getattr(args, "transition_resource_certificate", None),
        getattr(args, "transition_resource_sha256", None),
    )
    extra = getattr(args, "transition_extra_epochs", 0)
    if type(extra) is not int or extra < 0:
        raise ValueError("transition-extra-epochs must be an explicit nonnegative integer")
    if source is None:
        if any(value is not None for value in other) or extra:
            raise ValueError("transition options require --transition-from-checkpoint")
        return
    if any(value is None for value in other):
        raise ValueError(
            "transition requires source SHA, mode, and a measured resource certificate/SHA"
        )
    if args.resume is not True:
        raise ValueError(
            "transition requires epoch-boundary resume; --no-resume would discard progress"
        )
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (other[0], other[3])
    ):
        raise ValueError("transition SHA values must be lowercase SHA-256 digests")
    if args.transition_mode == "replace_c":
        if (args.condition, args.conductance_backend, args.training_schedule) != (
            "shared_dynamic_c",
            "optimization",
            "joint",
        ):
            raise ValueError("replace_c requires shared_dynamic_c, optimization, and joint")
    elif args.transition_mode == "continue_fixed":
        if (args.condition, args.conductance_backend, args.training_schedule) != (
            "fixed_c",
            "mlp",
            "staged",
        ):
            raise ValueError("continue_fixed must retain fixed_c, mlp, and staged")
    else:
        raise ValueError("unsupported transition mode")


def validate_output_boundary(args, output: Path) -> None:
    request = request_from_args(args)
    if request is None:
        return
    source = Path(request["source_path"])
    if (
        output == source.parent
        or output.is_relative_to(source.parent)
        or source.is_relative_to(output)
    ):
        raise ValueError("transition output must be separate from the source checkpoint directory")
    original_output = args.output_dir.expanduser().absolute()
    if any(path.is_symlink() for path in (original_output, *original_output.parents, output)):
        raise ValueError("transition output may not be a symlink")


def _read_certificate(args, inspected, protocol, target_configuration) -> tuple[dict, dict]:
    from scripts.calibrate_training_resources import _hardware
    from scripts.training_resource_plan import candidate_score, choose_candidate, source_snapshot

    from . import train

    path = args.transition_resource_certificate
    if (
        path.is_symlink()
        or not path.is_file()
        or train.sha256_file(path) != args.transition_resource_sha256
    ):
        raise ValueError("transition resource certificate path/SHA mismatch")
    certificate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "kind": "v5_transition_resource_certificate",
        "status": "passed",
        "classification": "resource_calibration_not_final_training",
        "source_checkpoint_sha256": inspected["sha256"],
        "transition_mode": args.transition_mode,
        "source_epoch": inspected["source_epoch"],
        "runtime_versions": train._versions(),
        "dataset_protocol_sha256": train._canonical_sha256(protocol),
        "cache_sha256": protocol["data_sha256"],
        "source_sha256": source_snapshot(),
        "hardware": _hardware(args.device),
        "selected_configuration": target_configuration,
    }
    if not isinstance(certificate, dict) or any(
        certificate.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("transition resource certificate source/data/runtime contract mismatch")
    original = inspected["identity"]["configuration"]
    for key, configuration in (
        ("baseline_execution", original),
        ("selected_execution", target_configuration),
    ):
        expected_execution = {
            name: configuration[name]
            for name in ("batch_size", "workers", "sample_seed_batch_size")
        }
        if certificate.get(key) != expected_execution:
            raise ValueError(f"transition resource certificate {key} mismatch")
    policy = certificate.get("selection", {})
    expected_policy = (
        "preserve_fixed_execution"
        if args.transition_mode == "continue_fixed"
        else "measured_transition_candidates"
    )
    if policy.get("no_downscale") is not True or policy.get("policy") != expected_policy:
        raise ValueError("transition resource certificate selection policy mismatch")
    selected = certificate.get("selected_candidate")
    candidates = certificate.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list) or not candidates:
        raise ValueError("transition certificate is missing actual candidate measurements")
    expected_axis = (
        "graphs"
        if args.dataset == "ppi"
        else "full_graph"
        if args.sampling == "full"
        else "sampled_seed_nodes"
    )
    if certificate.get("batch_axis") != expected_axis:
        raise ValueError("transition certificate physical batch axis differs from the dataset")
    natural = certificate.get("natural_training_split_size")
    baseline_physical = (
        original["batch_size"]
        if expected_axis == "graphs"
        else original["sample_seed_batch_size"]
        if expected_axis == "sampled_seed_nodes"
        else 1
    )
    selected_physical = (
        target_configuration["batch_size"]
        if expected_axis == "graphs"
        else target_configuration["sample_seed_batch_size"]
        if expected_axis == "sampled_seed_nodes"
        else 1
    )
    expected_natural = 1 if expected_axis == "full_graph" else protocol["split_counts"]["train"]
    if type(natural) is not int or natural < 1 or natural != expected_natural:
        raise ValueError("transition certificate natural split size is invalid")
    if selected != {"batch_size": selected_physical, "workers": target_configuration["workers"]}:
        raise ValueError("transition certificate selected physical resources mismatch")
    seen = set()
    for candidate in candidates:
        candidate_score(candidate)
        key = candidate["batch_size"], candidate["workers"]
        if (
            key in seen
            or key[0] < baseline_physical
            or key[0] > max(natural, baseline_physical)
            or key[1] < original["workers"]
        ):
            raise ValueError("transition certificate has duplicate/out-of-contract candidates")
        seen.add(key)
        for measurement in candidate["measurements"]:
            if (
                measurement.get("status") == "passed"
                and measurement.get("total_memory_bytes")
                != certificate["hardware"]["total_memory_bytes"]
            ):
                raise ValueError("transition measurement GPU capacity contradicts bound hardware")
            if measurement.get("batch_size") != key[0] or measurement.get("workers") != key[1]:
                raise ValueError("transition measurement batch/workers contradict its candidate")
            if (
                measurement.get("condition") != args.condition
                or measurement.get("model_seed") != args.model_seed
            ):
                raise ValueError(
                    "transition certificate measured a different condition or model seed"
                )
    if (baseline_physical, original["workers"]) not in seen:
        raise ValueError("transition certificate is missing an actual baseline measurement")
    if (
        args.transition_mode == "replace_c"
        and baseline_physical < natural
        and len({key[0] for key in seen}) < 2
    ):
        raise ValueError("new C resource selection requires multiple measured physical batches")
    best = choose_candidate(candidates, baseline_physical)
    if selected != {"batch_size": best["batch_size"], "workers": best["workers"]}:
        raise ValueError("transition resources differ from the measured safe throughput selection")
    change_names = {
        "batch_size",
        "workers",
        "loader_workers",
        "persistent_workers",
        "prefetch_factor",
        "worker_configuration_source",
        "sample_seed_batch_size",
    }
    changes = {
        name: {"before": original.get(name), "after": target_configuration.get(name)}
        for name in change_names
        if original.get(name) != target_configuration.get(name)
    }
    if args.transition_mode == "continue_fixed" and changes:
        raise ValueError("fixed continuation cannot change its measured execution configuration")
    if train.sha256_file(path) != args.transition_resource_sha256:
        raise ValueError("transition resource certificate changed during validation")
    return certificate, changes


def prepare_training_origin(args, model, optimizer, protocol, output: Path) -> dict[str, Any]:
    """Construct reproducible transferred initialization and bind its provenance."""
    from . import train

    validate_output_boundary(args, output)
    inspected = inspect_transition_source(
        args.transition_from_checkpoint, expected_sha256=args.transition_source_sha256
    )
    certificate, changes = _read_certificate(args, inspected, protocol, train.configuration(args))
    if args.transition_mode == "continue_fixed":
        schedule = copy.deepcopy(inspected["identity"]["schedule"])
        schedule[-1]["end_epoch"] += args.transition_extra_epochs
        schedule[-1]["length"] += args.transition_extra_epochs
    else:
        schedule = train.phase_schedule(
            args.epochs, list(args.phase_fractions), args.training_schedule
        )
    identity = train.build_resume_identity(
        args, protocol, schedule, initial_state_sha256=train.state_sha256(model)
    )
    request = request_from_args(args)
    request["declared_execution_changes"] = changes
    identity["transition_request"] = request
    prepared = prepare_transition_state(
        args.transition_from_checkpoint,
        expected_sha256=args.transition_source_sha256,
        target_model=model,
        target_optimizer=optimizer,
        target_identity=identity,
        mode=args.transition_mode,
        additional_epochs=args.transition_extra_epochs,
        declared_execution_changes=changes,
    )
    model.load_state_dict(prepared["model_state"], strict=True)
    optimizer.load_state_dict(prepared["optimizer_state"])
    train.validate_optimizer_parameter_ownership(model, optimizer)
    identity["initial_state_sha256"] = train.state_sha256(model)
    shared_initial = train.shared_initial_state_sha256(model)
    provenance = prepared["provenance"]
    provenance.update(
        source_checkpoint_sha256=inspected["sha256"],
        source_epochs_requested=inspected["identity"]["configuration"]["epochs"],
        epoch_offset=prepared["epoch_offset"],
        source_history_sha256=train._canonical_sha256(prepared["source_history"]),
        resource_certificate_sha256=args.transition_resource_sha256,
        execution_changes=changes,
        same_experiment_resume=False,
        initialization="retained_shared_weights_and_adamw_state_with_new_c"
        if args.transition_mode == "replace_c"
        else "retained_fixed_c_training_state",
        source_elapsed_seconds=prepared["counters"]["elapsed_seconds"],
    )
    identity["transition_provenance_sha256"] = train._canonical_sha256(provenance)
    prepared.update(
        resume_identity=identity,
        resume_identity_sha256=train._canonical_sha256(identity),
        initial_state_sha256=identity["initial_state_sha256"],
        shared_initial_state_sha256=shared_initial,
        schedule=schedule,
        resource_certificate=certificate,
    )
    if args.transition_mode == "replace_c":
        # New-stage throughput must not divide new batches by old-model wall time.
        prepared["counters"].update(
            elapsed_seconds=0.0, peak_cuda_allocated_bytes=0, peak_cuda_reserved_bytes=0
        )
    return prepared


def publish_transition_boundary(prepared, args, output: Path, architecture) -> None:
    """Publish an explicit source-epoch boundary in the new output directory only."""
    from . import train

    if (output / "last.pt").exists():
        return
    identity, identity_hash = prepared["resume_identity"], prepared["resume_identity_sha256"]
    selection = copy.deepcopy(prepared["selection_state"])
    provenance = prepared["provenance"]
    existing = reuse_initialization_artifacts(prepared, args, output, architecture)
    if existing["best_checkpoint_sha256"] is not None:
        selection["best_checkpoint_sha256"] = existing["best_checkpoint_sha256"]
    if args.transition_mode == "continue_fixed" and existing["best_checkpoint_sha256"] is None:
        expected_hash = selection["best_checkpoint_sha256"]
        source_dir = args.transition_from_checkpoint.parent
        source_best = next(
            (
                path
                for path in (source_dir / "best.pt", source_dir / "best.previous.pt")
                if not path.is_symlink()
                and path.is_file()
                and train.sha256_file(path) == expected_hash
            ),
            None,
        )
        if source_best is None:
            raise ValueError(
                "fixed continuation source best checkpoint has no valid read-only recovery slot"
            )
        with source_best.open("rb") as stream:
            selected = torch.load(stream, map_location="cpu", weights_only=True)
        train.validate_selected_checkpoint(
            selected,
            expected_identity=provenance["source_identity"],
            expected_identity_sha256=provenance["source_identity_sha256"],
            expected_epoch=selection["best_epoch"],
            expected_metric=selection["best_metric"],
            expected_selection_role="primary",
        )
        if train.sha256_file(source_best) != expected_hash:
            raise ValueError("source best checkpoint changed during transition validation")
        selected.update(
            resume_identity=identity,
            resume_identity_sha256=identity_hash,
            architecture=architecture,
            configuration=train.configuration(args),
            schedule=prepared["schedule"],
            transition_provenance=provenance,
            source_selected_checkpoint_sha256=expected_hash,
        )
        selection["best_checkpoint_sha256"] = train.publish_best_checkpoint(
            output / "best.pt",
            output / "best.previous.pt",
            selected,
        )
    if not existing["source_history_exists"]:
        atomic_write_json(output / "source-history.json", prepared["source_history"])
    train._save(
        output / "last.pt",
        {
            "schema_version": 4,
            "complete": False,
            "model_state": prepared["model_state"],
            "optimizer_state": prepared["optimizer_state"],
            "resume_identity": identity,
            "resume_identity_sha256": identity_hash,
            "epoch": provenance["source_epoch"],
            "epoch_offset": prepared["epoch_offset"],
            "phase": {"coordinate": "explicit_transition_boundary"},
            "history": prepared["history"],
            "transition_provenance": provenance,
            "resume_source_compatibility": [],
            **selection,
            **prepared["counters"],
            **prepared["rng_state"],
        },
    )


def validate_transition_resume(saved, prepared, output: Path) -> None:
    from . import train

    if (
        saved.get("schema_version") != 4
        or saved.get("transition_provenance") != prepared["provenance"]
    ):
        raise ValueError("transition last.pt provenance/schema mismatch")
    if saved.get("epoch_offset") != prepared["epoch_offset"]:
        raise ValueError("transition last.pt epoch offset mismatch")
    source_history = output / "source-history.json"
    if source_history.is_symlink() or not source_history.is_file():
        raise ValueError("transition archived source history is missing or unsafe")
    history_payload = json.loads(source_history.read_text(encoding="utf-8"))
    if train._canonical_sha256(history_payload) != prepared["provenance"]["source_history_sha256"]:
        raise ValueError("transition archived source history was changed")
    if train._canonical_sha256(saved["transition_provenance"]) != saved["resume_identity"].get(
        "transition_provenance_sha256"
    ):
        raise ValueError("transition last.pt provenance hash mismatch")
    history, offset = saved.get("history"), prepared["epoch_offset"]
    if not isinstance(history, list) or any(
        not isinstance(row, dict) or row.get("epoch") != offset + index
        for index, row in enumerate(history, 1)
    ):
        raise ValueError("transition last.pt history does not match its cumulative epoch range")
    if prepared["provenance"]["mode"] == "replace_c":
        for key in ("best_epoch", "global_best_epoch", "joint_best_epoch"):
            value = saved.get(key)
            if type(value) is not int or (
                value != 0 and not offset < value <= saved.get("epoch", -1)
            ):
                raise ValueError("transition last.pt reused an old-C selection epoch")
