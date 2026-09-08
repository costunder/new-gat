"""Read-only semantic validation of completed edge-selection training evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _score(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label} must be a finite validation score in [0, 1]")
    return value


def _fingerprint(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA256 fingerprint")
    return value


def _budget(args, metrics):
    from ..v5.learning_budget import deterministic_batches_per_epoch, plan_learning_budget
    from ..v5.protocol import HARDWARE_PROFILES, learning_budget_arguments_configuration

    topology = metrics.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("completed evidence has no full-training topology/count metadata")
    count = _positive_integer(topology.get("train_count"), "official training unit count")
    selected = learning_budget_arguments_configuration(args)
    reference = selected.get("budget_reference_batch_size")
    if args.dataset != "ppi" and args.sampling == "full":
        if reference not in {None, 1}:
            raise ValueError("full-graph training cannot declare a replicated reference batch")
        actual_batches = reference_batches = 1
    else:
        physical = args.batch_size if args.dataset == "ppi" else args.sample_seed_batch_size
        field = "ppi_batch_size" if args.dataset == "ppi" else "sample_seed_batch_size"
        reference = reference or HARDWARE_PROFILES[args.hardware_profile][field]
        actual_batches = deterministic_batches_per_epoch(count, physical)
        reference_batches = deterministic_batches_per_epoch(count, reference)
    return plan_learning_budget(
        args.epochs,
        args.patience,
        reference_batches,
        actual_batches,
        policy=selected.get("learning_budget_policy", "epochs"),
    ), count


def _history(metrics, identity, rows, args):
    from ..v5.learning_budget import should_stop_learning_budget

    epochs = _positive_integer(metrics.get("epochs_run"), "completed epoch count")
    if (
        not isinstance(rows, list)
        or len(rows) != epochs
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("completed history must contain one record per completed epoch")
    if [row.get("epoch") for row in rows] != list(range(1, epochs + 1)):
        raise ValueError("completed history does not contain contiguous full epochs")
    budget, count = _budget(args, metrics)
    if metrics.get("learning_budget") != budget or identity.get("learning_budget") != budget:
        raise ValueError(
            "completed learning budget differs from the actual CLI and full-training count"
        )
    if epochs > budget["planned_epochs"]:
        raise ValueError("completed epochs exceed the declared full learning budget")
    for row in rows:
        _positive_integer(row.get("epoch"), "history epoch")
        _score(row.get("validation"), "history validation")
        if (
            row.get("train_batches") != budget["actual_batches_per_epoch"]
            or row.get("optimizer_steps") != row["epoch"] * budget["actual_batches_per_epoch"]
            or row.get("processed_units") != count
        ):
            raise ValueError("history does not prove complete supervised epoch/update coverage")
        for key in ("train_batches", "optimizer_steps", "processed_units"):
            _positive_integer(row[key], key)
        if row.get("phase", {}).get("phase") != "joint":
            raise ValueError("edge-selection history contains a foreign training phase")
    steps = _positive_integer(metrics.get("optimizer_steps"), "completed optimizer update count")
    if steps != rows[-1]["optimizer_steps"]:
        raise ValueError("completed optimizer update count disagrees with history")
    best = _positive_integer(metrics.get("best_epoch"), "selected epoch")
    score = _score(metrics.get("best_validation"), "selected validation score")
    first_maximum = max(range(epochs), key=lambda index: rows[index]["validation"])
    if best != first_maximum + 1 or score != rows[first_maximum]["validation"]:
        raise ValueError("selected checkpoint is not the first strict maximum validation epoch")
    _score(metrics.get("validation"), "selected-checkpoint validation recheck")
    if epochs < budget["planned_epochs"] and not should_stop_learning_budget(
        budget,
        epochs_since_best=epochs - best,
        optimizer_steps_since_best=steps - rows[best - 1]["optimizer_steps"],
        eligible=True,
    ):
        raise ValueError("completed training stopped before its declared budget and patience")


def _checkpoint(checkpoint, identity, *, role):
    from . import train

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{role} checkpoint must be an object")
    train.validate_identity(checkpoint, identity)
    if not isinstance(checkpoint.get("model_state"), dict) or not checkpoint["model_state"]:
        raise ValueError(f"{role} checkpoint has no model state")


def inspect_completed(output):
    """Validate metadata, budget, full epochs, and both CPU checkpoint interiors.

    The larger last checkpoint is released before loading best.pt. No model is
    instantiated, CUDA tensor allocated, dataset fetched, or artifact written.
    """
    from . import train

    output = Path(output)
    paths = {name: output / name for name in ("metrics.json", "history.json", "last.pt", "best.pt")}
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"completed edge-selection artifact is missing or indirect: {path}")
    fingerprints = {name: train.base.sha256_file(path) for name, path in paths.items()}
    metrics = json.loads(paths["metrics.json"].read_text(encoding="utf-8"))
    if (
        not isinstance(metrics, dict)
        or metrics.get("status") != "passed"
        or metrics.get("research_suite") != train.SUITE
    ):
        raise ValueError("not completed edge-selection evidence")
    identity = metrics.get("resume_identity")
    if not isinstance(identity, dict) or metrics.get(
        "resume_identity_sha256"
    ) != train.base._canonical_sha256(identity):
        raise ValueError("completed edge-selection identity is corrupt")
    for name, field in (
        ("best.pt", "checkpoint_sha256"),
        ("last.pt", "last_checkpoint_sha256"),
        ("history.json", "history_sha256"),
    ):
        if fingerprints[name] != metrics.get(field):
            raise ValueError(f"completed edge-selection artifact mismatch: {name}")
    for key in (
        "research_suite",
        "dataset",
        "condition",
        "configuration",
        "source_sha256",
        "initial_state_sha256",
        "learning_budget",
    ):
        if metrics.get(key) != identity.get(key):
            raise ValueError(f"completed metrics and immutable identity disagree on {key}")
    if (
        metrics.get("test_evaluated") is not False
        or metrics.get("debug") is not False
        or metrics.get("subset") is not False
    ):
        raise ValueError(
            "completed edge-selection evidence is not full validation-only research training"
        )
    sources = identity.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("completed identity has no source provenance")
    for name, value in sources.items():
        _fingerprint(value, f"source {name}")
    protocol = metrics.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol != identity.get("dataset_protocol")
        or train.base._canonical_sha256(protocol) != identity.get("dataset_protocol_sha256")
    ):
        raise ValueError("completed data/split protocol identity mismatch")
    _fingerprint(protocol.get("data_sha256"), "official data cache")
    if not isinstance(identity.get("input_provenance"), list) or not identity["input_provenance"]:
        raise ValueError("completed identity has no topology/corruption provenance")
    if metrics.get("topology", {}).get("provenance") != identity["input_provenance"]:
        raise ValueError("completed topology/corruption provenance differs from training")
    saved_args = identity.get("training_arguments")
    if (
        not isinstance(saved_args, dict)
        or "data_root" not in saved_args
        or "device" not in saved_args
    ):
        raise ValueError("completed identity has no restorable training arguments")
    if "training_arguments" in metrics and metrics["training_arguments"] != saved_args:
        raise ValueError("completed training arguments disagree with the immutable identity")
    args = train.restore_arguments(metrics, output, saved_args["data_root"], saved_args["device"])
    if args.dataset != metrics["dataset"] or args.selection_mode != metrics["condition"]:
        raise ValueError("saved dataset/selection mode disagrees with the actual trained arguments")
    initial = _fingerprint(metrics.get("initial_state_sha256"), "initial model")
    if initial != identity.get("initial_state_sha256"):
        raise ValueError("initial state differs from the immutable identity")
    shared = _fingerprint(metrics.get("shared_initial_state_sha256"), "shared initial model")
    if metrics.get("common_backbone_initial_state_sha256", shared) != shared:
        raise ValueError("common backbone initialization differs from shared initialization")
    rows = json.loads(paths["history.json"].read_text(encoding="utf-8"))
    _history(metrics, identity, rows, args)
    last = train.base.load_checkpoint_on_cpu(paths["last.pt"])
    _checkpoint(last, identity, role="last")
    expected = {
        "epoch": metrics["epochs_run"],
        "optimizer_steps": metrics["optimizer_steps"],
        "best_epoch": metrics["best_epoch"],
        "best_validation": metrics["best_validation"],
        "best_checkpoint_sha256": metrics["checkpoint_sha256"],
        "shared_initial_state_sha256": shared,
        "history": rows,
    }
    if any(last.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "last checkpoint history/best/update metadata disagrees with completed metrics"
        )
    optimizer = last.get("optimizer_state")
    if (
        not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("state"), dict)
        or not optimizer["state"]
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer["param_groups"]
    ):
        raise ValueError("last checkpoint lacks actual optimizer state")
    del optimizer, last
    best = train.base.load_checkpoint_on_cpu(paths["best.pt"])
    _checkpoint(best, identity, role="best")
    if (
        best.get("selection_role") != "primary"
        or best.get("epoch") != metrics["best_epoch"]
        or best.get("validation") != metrics["best_validation"]
    ):
        raise ValueError("best checkpoint selection metadata disagrees with completed metrics")
    del best
    if fingerprints != {name: train.base.sha256_file(path) for name, path in paths.items()}:
        raise ValueError("completed evidence changed while being inspected")
    return metrics
