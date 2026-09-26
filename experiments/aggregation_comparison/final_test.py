"""Official arxiv test after the entire validation-selected matrix is frozen."""

from __future__ import annotations

import copy
import gc
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from . import engine, runner
from .benchmark_policy import require_benchmark_datasets
from .validation import require_reproduction, validate_evaluation


class HeldoutInputs:
    """Reuse full graph geometry, replace only the supervised evaluation indices."""

    def __init__(self, validation_inputs, test_mask):
        if test_mask.dtype != torch.bool or test_mask.ndim != 1 or not test_mask.any():
            raise ValueError("official test requires a nonempty boolean node mask")
        self.base = validation_inputs
        self.selected = test_mask.nonzero(as_tuple=False).flatten().long()
        self.indices = {"test": self.selected}

    def validation_batches(self, device):
        selected = self.selected.to(device)
        for batch in self.base.validation_batches(device):
            if batch.graph.x.shape[0] != self.base.validation_record.graph.x.shape[0]:
                raise ValueError("official test must use the complete graph")
            yield replace(batch, selected_indices=selected)


def freeze_matrix(args, manifest, persist):
    """Validate every cell before loading the dataset or entering test inference."""
    require_benchmark_datasets(args.datasets)
    expected = {
        f"{profile}/{dataset}/model-seed-{seed}/{arm}"
        for profile in args.profiles
        for dataset in args.datasets
        for seed in args.model_seeds
        for arm in args.arms
    }
    jobs = manifest["jobs"]
    if len(jobs) != len(expected) or {j["job_id"] for j in jobs} != expected:
        raise ValueError("official test requires the exact complete planned matrix")
    for job in jobs:
        audit = job.get("audit", {})
        if job.get("status") != "passed" or audit.get("status") != "passed":
            raise ValueError("every cell must pass training and validation audit before test")
        result = runner._read_result(job)
        if result != job.get("result"):
            raise ValueError("on-disk validation result differs from the test matrix")
        if (
            audit.get("checkpoint_sha256") != result["checkpoint_sha256"]
            or audit.get("evaluator_source_sha256") != engine.implementation_source_hashes()
            or engine.base.sha256_file(Path(audit["log_path"])) != audit.get("log_sha256")
        ):
            raise ValueError("validation audit evidence changed before official test")
    runner._compare(jobs)
    lock = {j["job_id"]: j["result"]["checkpoint_sha256"] for j in jobs}
    if manifest.get("official_test_checkpoint_lock") not in (None, lock):
        raise ValueError("official test checkpoint matrix cannot be changed")
    manifest["official_test_checkpoint_lock"] = lock
    persist()
    return lock


def evaluate_matrix(args, manifest, persist):
    lock = freeze_matrix(args, manifest, persist)
    device = torch.device(args.device)
    engine.base._require_cuda(device)
    payload, protocol = engine.base.load_dataset(
        args.datasets[0], args.data_root, allow_download=False
    )
    engine.base.validate_cached_graphs_once(payload)
    test_mask = payload["splits"]["test"]
    for split in ("train", "validation"):
        if (test_mask & payload["splits"][split]).any():
            raise ValueError("official test nodes overlap training or validation")
    test_hash = engine.base.tensor_hash(test_mask)
    sources = engine.implementation_source_hashes()
    report = manifest.setdefault(
        "official_test",
        {
            "dataset": args.datasets[0],
            "split_sha256": test_hash,
            "source_sha256": sources,
            "selection": "validation only; all checkpoint hashes frozen before test",
            "expected_cells": len(lock),
            "status": "running",
            "results": {},
        },
    )
    if (
        report["split_sha256"] != test_hash
        or report["source_sha256"] != sources
        or report["dataset"] != args.datasets[0]
        or report["expected_cells"] != len(lock)
        or not set(report["results"]) <= set(lock)
    ):
        raise ValueError("official test identity or cell set changed")
    prepared = {}
    for job in manifest["jobs"]:
        key, root = job["job_id"], Path(job["output_dir"])
        if (
            job["result"]["data_sha256"] != protocol["data_sha256"]
            or job["result"]["split_sha256"] != protocol["split_sha256"]
        ):
            raise ValueError("official test current cache differs from the frozen training data")
        if key in report["results"]:
            stored = report["results"][key]
            validate_evaluation(stored["evaluation"], label="saved official test")
            if (
                stored["checkpoint_sha256"] != lock[key]
                or stored["evaluation"]["metric_kind"] != "accuracy"
                or stored["evaluation"]["counts"]["total"] != int(test_mask.sum())
                or stored["arm"] != job["variant_id"]
                or stored["profile"] != job["profile"]
                or stored["seed"] != job["model_seed"]
                or stored["parameters"] != job["result"]["total_parameters"]
            ):
                raise ValueError("saved official test count/checkpoint changed")
            continue
        metrics = engine.inspect_completed(root)
        if metrics["resume_identity"]["dataset_protocol"] != protocol:
            raise ValueError("official test dataset/split provenance differs from training")
        child = engine.restore_arguments(metrics, root, args.data_root, device)
        engine.base.configure_compute(child)
        engine.base._seed(child.model_seed)
        cache_key = (child.dataset, child.forest_seed)
        if cache_key not in prepared:
            full_args = copy.deepcopy(child)
            full_args.sampling = "full"
            prepared[cache_key] = engine.PreparedInputs(payload, full_args)
        inputs = prepared[cache_key]
        if inputs.provenance != metrics["resume_identity"]["input_provenance"]:
            raise ValueError("official test full-graph topology differs from training")
        best = root / "best.pt"
        if engine.base.sha256_file(best) != lock[key]:
            raise ValueError("checkpoint changed after matrix freeze")
        saved = engine.base.load_checkpoint_on_cpu(best)
        engine.validate_identity(saved, metrics["resume_identity"])
        model = engine.make_model(payload, child, device)
        model.load_state_dict(saved["model_state"], strict=True)
        del saved
        state_hash = engine.base.state_sha256(model)
        require_reproduction(
            metrics["selected_validation_evidence"],
            engine.evaluate(model, inputs, child, device),
            label="official test selected disk checkpoint validation",
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        evaluation = engine.evaluate(model, HeldoutInputs(inputs, test_mask), child, device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        validate_evaluation(evaluation, label="official test")
        if evaluation["metric_kind"] != "accuracy" or evaluation["counts"]["total"] != int(
            test_mask.sum()
        ):
            raise ValueError("official test must cover every test node exactly once")
        if (
            engine.base.state_sha256(model) != state_hash
            or engine.base.sha256_file(best) != lock[key]
            or engine.implementation_source_hashes() != sources
        ):
            raise ValueError("official test modified checkpoint/model/source")
        report["results"][key] = {
            "arm": job["variant_id"],
            "profile": job["profile"],
            "seed": job["model_seed"],
            "checkpoint_sha256": lock[key],
            "evaluation": evaluation,
            "evaluation_seconds": elapsed,
            "parameters": metrics["model_contract"]["total_parameters"],
        }
        persist()
        del model
        gc.collect()
    report["status"] = "passed"
    manifest["test_evaluated"] = True
    persist()


def markdown(report):
    lines = [
        "",
        "## Official test (validation-selected checkpoints)",
        "",
        f"Status: {report['status']}; completed cells: "
        f"{len(report['results'])}/{report['expected_cells']}.",
        "",
        "Accuracy mean and sample standard deviation across the recorded model seeds.",
        "One seed has no standard-deviation estimate. Parameters are not matched.",
        "",
        "| Profile | Model | Seeds | Mean accuracy | Std | Parameters |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    groups = {}
    for row in report["results"].values():
        groups.setdefault((row["profile"], row["arm"]), []).append(row)
    for (profile, arm), rows in sorted(groups.items()):
        values = [row["evaluation"]["metric"] for row in rows]
        deviation = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "unavailable"
        parameters = sorted({row["parameters"] for row in rows})
        lines.append(
            f"| {profile} | {arm} | {len(values)} | {statistics.mean(values):.6f} | "
            f"{deviation} | {','.join(map(str, parameters))} |"
        )
    return lines
