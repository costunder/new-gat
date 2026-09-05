"""Immutable ownership journal for retrying V5 transition initialization only.

An interrupted initialization is not an ordinary resume. This marker permits
retrying only the exact request in the directory that this invocation claimed
while it was empty. It never authorizes overwriting unrelated/trained results.
Once last.pt exists, normal strict transition-resume validation takes over.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from chartgat.cache import atomic_write_json

from .transition import canonical_sha256

MARKER_FILENAME = "transition-initialization.json"
_INITIALIZATION_FILES = {
    MARKER_FILENAME,
    "metrics.json",
    "source-history.json",
    "best.pt",
    "failure-resource-observability.json",
}


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"transition initialization artifact must be a regular file: {path.name}")


def _sha256(path: Path) -> str:
    _regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    _regular_file(path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(result, allow_nan=False)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise ValueError(f"transition initialization JSON is invalid: {path.name}") from error
    return result


def _identity(args, output, configuration, source_sha256, runtime_versions):
    from .transition_training import request_from_args

    return {
        "schema_version": 1,
        "kind": "v5_transition_initialization_ownership",
        "dataset": args.dataset,
        "condition": args.condition,
        "request": request_from_args(args),
        "configuration": configuration,
        "source_sha256": source_sha256,
        "runtime_versions": runtime_versions,
        "output_directory": str(output.resolve()),
        "data_root": str(args.data_root.expanduser().resolve()),
    }


def _validate_marker(path: Path, identity: dict) -> None:
    marker = _json(path)
    if (
        not isinstance(marker, dict)
        or set(marker) != {"identity", "identity_sha256"}
        or marker["identity"] != identity
        or marker["identity_sha256"] != canonical_sha256(identity)
    ):
        raise ValueError("transition initialization marker request/config/source identity mismatch")


def _validate_partial_files(args, output: Path, configuration: dict) -> None:
    for artifact in output.iterdir():
        if artifact.name not in _INITIALIZATION_FILES:
            raise FileExistsError(
                "transition initialization cannot adopt unrelated or trained artifact: "
                f"{artifact.name}"
            )
        _regular_file(artifact)
    metrics = output / "metrics.json"
    if metrics.exists():
        record = _json(metrics)
        expected = {
            "schema_version": 1,
            "research_suite": "conductance_graph_conditioned_v5",
            "dataset": args.dataset,
            "condition": args.condition,
            "configuration": configuration,
            "test_evaluated": False,
        }
        allowed = set(expected) | {
            "status",
            "error",
            "failure_resource_observability",
            "failure_resource_observability_sha256",
        }
        if (
            not isinstance(record, dict)
            or record.get("status") not in {"running", "failed"}
            or set(record) - allowed
            or any(record.get(key) != value for key, value in expected.items())
        ):
            raise ValueError(
                "transition initialization metrics are completed, trained or mismatched"
            )
    failure = output / "failure-resource-observability.json"
    if failure.exists():
        record = _json(failure)
        if (
            not isinstance(record, dict)
            or record.get("status") != "failed"
            or record.get("dataset") != args.dataset
            or record.get("condition") != args.condition
            or record.get("research_suite") != "conductance_graph_conditioned_v5"
        ):
            raise ValueError("transition initialization failure telemetry belongs to another run")
    if (output / "source-history.json").exists():
        if not isinstance(_json(output / "source-history.json"), list):
            raise ValueError("transition initialization source history is malformed")
    if (output / "best.pt").exists() and args.transition_mode != "continue_fixed":
        raise ValueError("new-C initialization cannot contain a pre-training best checkpoint")


def ensure_transition_initialization(
    args, output: Path, *, configuration: dict, source_sha256: dict, runtime_versions: dict
) -> bool:
    """Claim an empty transition output or validate its exact initialization retry.

    Call before main's nonempty-output gate and metrics.running publication.
    True permits ONLY this journaled initialization to pass that gate. False
    means no transition, or last.pt already exists and ordinary resume applies.
    """
    if getattr(args, "transition_from_checkpoint", None) is None:
        return False
    raw_output = args.output_dir.expanduser()
    for path in (raw_output, *raw_output.parents):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("transition initialization output may not traverse symlinks/junctions")
    if not args.resume:
        raise ValueError("transition initialization requires resume enabled")
    output = Path(output)
    if output.is_symlink():
        raise ValueError("transition initialization output may not be a symlink")
    identity = _identity(args, output, configuration, source_sha256, runtime_versions)
    json.dumps(identity, allow_nan=False)
    marker_path = output / MARKER_FILENAME
    last_path = output / "last.pt"
    if last_path.exists() or last_path.is_symlink():
        _regular_file(last_path)
        if marker_path.exists() or marker_path.is_symlink():
            _validate_marker(marker_path, identity)
        return False
    if output.exists() and not output.is_dir():
        raise ValueError("transition initialization output must be a directory")
    if marker_path.exists() or marker_path.is_symlink():
        _validate_marker(marker_path, identity)
        _validate_partial_files(args, output, configuration)
        return True
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("nonempty transition output has no matching initialization marker")
    # Validate the immutable source bytes before creating ownership metadata.
    if _sha256(args.transition_from_checkpoint) != args.transition_source_sha256:
        raise ValueError("transition initialization source checkpoint SHA mismatch")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        marker_path,
        {
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
        },
    )
    return True


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and torch.equal(actual, expected)
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, type(expected))
            and len(actual) == len(expected)
            and all(_equal(a, b) for a, b in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and actual == expected


def reuse_initialization_artifacts(
    prepared: dict, args, output: Path, architecture: dict
) -> dict[str, Any]:
    """Validate owned pre-last history/fixed-best artifacts; never overwrite them.

    Direct internal train tests may start with an empty directory. Reusing any
    partial publication requires the durable marker from the public main path.
    """
    output = Path(output)
    history_path, best_path = output / "source-history.json", output / "best.pt"
    has_partial = any(path.exists() or path.is_symlink() for path in (history_path, best_path))
    marker_path = output / MARKER_FILENAME
    if marker_path.exists() or marker_path.is_symlink():
        identity = prepared["resume_identity"]
        _validate_marker(
            marker_path,
            _identity(
                args,
                output,
                identity["configuration"],
                identity["source_sha256"],
                identity["runtime_versions"],
            ),
        )
        _validate_partial_files(args, output, identity["configuration"])
    elif has_partial:
        raise ValueError(
            "transition partial boundary artifacts have no initialization ownership marker"
        )
    result = {"best_checkpoint_sha256": None, "source_history_exists": False}
    if history_path.exists() or history_path.is_symlink():
        if _json(history_path) != prepared["source_history"]:
            raise ValueError(
                "transition initialization source history differs from verified source"
            )
        result["source_history_exists"] = True
    if best_path.exists() or best_path.is_symlink():
        if args.transition_mode != "continue_fixed":
            raise ValueError("new C may not reuse an initialization best checkpoint")
        _regular_file(best_path)
        provenance = prepared["provenance"]
        expected_source_hash = prepared["selection_state"]["best_checkpoint_sha256"]
        source_dir = args.transition_from_checkpoint.parent
        source_best = None
        for path in (source_dir / "best.pt", source_dir / "best.previous.pt"):
            if path.is_file() and not path.is_symlink() and _sha256(path) == expected_source_hash:
                source_best = path
                break
        if source_best is None:
            raise ValueError("transition initialization source best has no verified recovery slot")
        with source_best.open("rb") as stream:
            expected = torch.load(stream, map_location="cpu", weights_only=True)
        if _sha256(source_best) != expected_source_hash:
            raise ValueError("source best changed while checking initialization reuse")
        from . import train

        train.validate_selected_checkpoint(
            expected,
            expected_identity=provenance["source_identity"],
            expected_identity_sha256=provenance["source_identity_sha256"],
            expected_epoch=prepared["selection_state"]["best_epoch"],
            expected_metric=prepared["selection_state"]["best_metric"],
            expected_selection_role="primary",
        )
        expected.update(
            resume_identity=prepared["resume_identity"],
            resume_identity_sha256=prepared["resume_identity_sha256"],
            architecture=architecture,
            configuration=train.configuration(args),
            schedule=prepared["schedule"],
            transition_provenance=provenance,
            source_selected_checkpoint_sha256=expected_source_hash,
        )
        before = _sha256(best_path)
        with best_path.open("rb") as stream:
            actual = torch.load(stream, map_location="cpu", weights_only=True)
        if _sha256(best_path) != before or not _equal(actual, expected):
            raise ValueError(
                "transition initialization best checkpoint does not match retained source"
            )
        result["best_checkpoint_sha256"] = before
    return result
