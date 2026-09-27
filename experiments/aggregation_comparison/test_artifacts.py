"""Bound original test predictions, separate from mutable summary rows."""

import math
from pathlib import Path

import torch

from . import engine
from .evidence import origin
from .validation import validate_evaluation


def binding(job, protocol, sources, test_hash):
    return {
        "job_id": job["job_id"],
        "checkpoint_sha256": job["result"]["checkpoint_sha256"],
        "data_protocol": protocol,
        "source_sha256": sources,
        "test_mask_sha256": test_hash,
        "visibility_protocol": protocol.get("visibility_protocol", "official_transductive"),
    }


def validate_record(record, job, expected_count):
    validate_evaluation(record["evaluation"], label="saved official test")
    loss = record["evaluation"].get("loss")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
        or loss < 0
    ):
        raise ValueError("saved test loss must be finite and nonnegative")
    elapsed = record.get("evaluation_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("saved official test runtime must be finite and nonnegative")
    if (
        record["checkpoint_sha256"] != job["result"]["checkpoint_sha256"]
        or record["evaluation"]["metric_kind"] != "accuracy"
        or record["evaluation"]["counts"]["total"] != expected_count
        or record["arm"] != job["variant_id"]
        or record["profile"] != job["profile"]
        or record["seed"] != job["model_seed"]
        or record["parameters"] != job["result"]["total_parameters"]
    ):
        raise ValueError("saved official test identity/count/checkpoint changed")


def artifact_path(job):
    return Path(job["output_dir"]) / "test_result.json"


def verify(artifact, expected_binding, job, payload, test_mask):
    if artifact.get("schema_version") != 1 or artifact.get("binding") != expected_binding:
        raise ValueError("saved official test artifact identity differs")
    selected = test_mask.nonzero(as_tuple=False).flatten().tolist()
    predictions = artifact["predictions"]
    if predictions["node_ids"] != selected or any(
        type(v) is not int for v in predictions["node_ids"]
    ):
        raise ValueError("saved test node order/coverage changed")
    classes = predictions["class_ids"]
    if len(classes) != len(selected) or any(
        type(value) is not int or not 0 <= value < payload["classes"] for value in classes
    ):
        raise ValueError("saved test predictions have invalid classes/counts")
    target = payload["graphs"][0]["y"].reshape(-1)[selected].cpu()
    correct = int((torch.tensor(classes, dtype=torch.long) == target).sum())
    record = artifact["record"]
    if record.get("evidence_origin") != origin(expected_binding["data_protocol"]):
        raise ValueError("saved test evidence origin changed")
    if expected_binding["visibility_protocol"] != "official_transductive":
        from .temporal import view

        years = payload["node_year"]
        expected_views = []
        for year in years[test_mask].unique(sorted=True).tolist():
            _, _, identity = view(payload, years <= year, test_mask & (years == year))
            expected_views.append({"year": year, **identity})
        if record.get("visibility_views") != expected_views:
            raise ValueError("saved temporal test views changed")
    validate_record(record, job, len(selected))
    if record["evaluation"]["counts"]["correct"] != correct:
        raise ValueError("saved test count differs from original predictions")
    return record


def read(job, expected_binding, payload, test_mask, stored=None):
    import json

    path = artifact_path(job)
    if path.is_symlink() or not path.is_file():
        raise ValueError("saved official test artifact missing; inference will not be repeated")
    fingerprint = engine.base.sha256_file(path)
    if stored is not None and stored.get("artifact_sha256") != fingerprint:
        raise ValueError("saved official test artifact hash changed")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    record = verify(artifact, expected_binding, job, payload, test_mask)
    row = {**record, "artifact_sha256": fingerprint}
    if stored is not None and stored != row:
        raise ValueError("saved official test summary differs from original artifact")
    return row


def publish(job, expected_binding, payload, test_mask, record, predictions):
    from chartgat.cache import atomic_write_json

    path = artifact_path(job)
    if path.exists():
        raise ValueError("original test artifact already exists; refuse to overwrite")
    artifact = {
        "schema_version": 1,
        "binding": expected_binding,
        "record": record,
        "predictions": predictions,
        "loss_evidence": "original sealed evaluation; class IDs only rederive accuracy",
    }
    verify(artifact, expected_binding, job, payload, test_mask)
    atomic_write_json(path, artifact)
    return read(job, expected_binding, payload, test_mask)
