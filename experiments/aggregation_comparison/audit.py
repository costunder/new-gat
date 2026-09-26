"""Read-only full-validation audit of fresh aggregation comparison checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from chartgat.observability import RuntimeResourceMonitor
from research.conductance_gat.edge_selection.audit import (
    _synchronize,
)
from research.conductance_gat.v5.batch_calibration import _isolated_execution_state

from . import engine as train
from .provenance import require_source_compatibility
from .validation import POLICY, require_reproduction


def audit(root, data_root, device, repeats):
    if repeats < 5:
        raise ValueError("at least five repeated full validations are required")
    metrics = train.inspect_completed(root)
    identity = metrics["resume_identity"]
    sources = train.implementation_source_hashes()
    require_source_compatibility(identity["source_sha256"], sources, scope="training")
    args = train.restore_arguments(metrics, root, data_root, device)
    train.base._require_cuda(device)
    train.base.configure_compute(args)
    payload, protocol = train.base.load_dataset(args.dataset, args.data_root, allow_download=False)
    if protocol != identity["dataset_protocol"]:
        raise ValueError("audit cache/split provenance differs from training")
    monitor = RuntimeResourceMonitor(device)
    monitor.start()
    try:
        with _isolated_execution_state(device), torch.no_grad():
            train.base._seed(args.model_seed)
            inputs = train.PreparedInputs(payload, args)
            if inputs.provenance != identity["input_provenance"]:
                raise ValueError("audit topology provenance differs from training")
            model = train.make_model(payload, args, device)
            selected = train.base.load_checkpoint_on_cpu(Path(root) / "best.pt")
            train.validate_identity(selected, identity)
            model.load_state_dict(selected["model_state"], strict=True)
            del selected
            before = train.base.state_sha256(model)
            torch.cuda.reset_peak_memory_stats(device)
            evaluations, timings = [], []
            for _ in range(repeats):
                _synchronize(device)
                started = time.perf_counter()
                evaluation = train.evaluate(model, inputs, args, device)
                require_reproduction(
                    metrics["selected_validation_evidence"],
                    evaluation,
                    label=f"audit repeat {len(evaluations) + 1}",
                )
                require_reproduction(
                    metrics["validation_evidence"],
                    evaluation,
                    label=f"final recheck versus audit repeat {len(evaluations) + 1}",
                )
                evaluations.append(evaluation)
                model.clear_auxiliary_cache()
                _synchronize(device)
                timings.append(time.perf_counter() - started)
            if before != train.base.state_sha256(model) or train.inspect_completed(root) != metrics:
                raise ValueError("read-only audit unexpectedly changed training evidence")
            require_source_compatibility(
                sources, train.implementation_source_hashes(), scope="audit"
            )
            scores = [row["metric"] for row in evaluations]
            return {
                "status": "passed",
                "research_suite": train.SUITE,
                "ablation_arm": args.ablation_arm,
                "dataset": args.dataset,
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "source_sha256": sources,
                "test_evaluated": False,
                "validation": evaluations[0],
                "repeated_validation": {
                    "count": repeats,
                    "scores": scores,
                    "evaluations": evaluations,
                    "reproduction_policy": POLICY,
                    "range_pp": 100 * (max(scores) - min(scores)),
                    "evaluation_seconds": timings,
                },
                "model_contract": model.contract(),
                "published_score_reproduction_claim": False,
            }
    finally:
        resources = monitor.finish(
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
        print(json.dumps({"audit_resources": resources}, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat-evaluations", type=int, default=5)
    args = parser.parse_args(argv)
    result = audit(
        args.root,
        args.data_root,
        torch.device(args.device),
        args.repeat_evaluations,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
