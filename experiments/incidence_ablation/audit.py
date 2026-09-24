"""Read-only full-validation audit of fresh incidence ablation checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from chartgat.observability import RuntimeResourceMonitor
from research.conductance_gat.edge_selection.audit import (
    PredictionSummary,
    _synchronize,
    release_amplitude_diagnostics,
)
from research.conductance_gat.v5.batch_calibration import _isolated_execution_state

from . import engine as train
from .diagnostics import components, cross_hop_gram, reconstruction_probe
from .provenance import require_source_compatibility


class Observer:
    def __init__(self, args, *, tolerance=1e-7, iterations=2000):
        self.args = args
        self.tolerance, self.iterations = tolerance, iterations
        self.records = []
        self.predictions = {}

    @torch.no_grad()
    def __call__(self, model, batch, logits, batch_index):
        incidence = batch.graph.incidence_edge_index
        topology = components(incidence, batch.graph.x.shape[0])
        records = []
        for layer, operator in enumerate(model.operators):
            weight = operator.last_effective_c.mean(dim=-1)
            if operator.last_sampling_correction is not None:
                weight = weight * operator.last_sampling_correction
            probe = operator.last_probe.flatten(1)
            records.append(
                {
                    "layer": layer,
                    "reconstruction": [
                        reconstruction_probe(
                            probe,
                            incidence,
                            weight,
                            noise_relative=noise,
                            seed=self.args.model_seed + layer,
                            topology=topology,
                            edge_chunk_size=self.args.edge_chunk_size,
                            tolerance=self.tolerance,
                            iterations=self.iterations,
                        )
                        for noise in (0.0, 1e-3)
                    ],
                }
            )
        self.records.append(
            {
                "batch": batch_index,
                "layers": records,
                "all_hop_cross_energy": cross_hop_gram(
                    model.last_history,
                    incidence,
                    weight,
                    self.args.edge_chunk_size,
                ),
            }
        )
        active = []
        if any(op.hop_coefficients is not None for op in model.operators):
            active.append(("disable_bilinear", "disable_bilinear"))
        if any(op.lift_projection is not None for op in model.operators):
            active.append(("remove_second_lift_channel", "disable_lift_channel"))
        model.capture = False
        try:
            for name, flag in active:
                try:
                    for op in model.operators:
                        setattr(op, flag, True)
                    with train.autocast(self.args, logits.device):
                        changed = model(batch.graph)
                    train.base.require_finite_tensor(changed, f"incidence intervention {name}")
                    self.predictions.setdefault(
                        name, PredictionSummary(self.args.dataset == "ppi")
                    ).add(changed, logits, batch)
                finally:
                    for op in model.operators:
                        setattr(op, flag, False)
        finally:
            model.capture = True
            model.last_history = None
            for op in model.operators:
                op.last_probe = None
            release_amplitude_diagnostics(model)

    def report(self, baseline):
        interventions = {name: value.report() for name, value in self.predictions.items()}
        for value in interventions.values():
            value["delta_validation_pp"] = 100 * (value["validation"] - baseline)
        return {
            "layers_and_batches": self.records,
            "interventions": interventions,
            "metric_scope": (
                "frozen arithmetic mean of effective head conductances, "
                "including sampling correction"
            ),
            "reconstruction_scope": (
                "all layer projected input features and full validation graph support; "
                "B[x,x²] constrained inverse, not inversion of P, the encoder or complete network"
            ),
            "cross_hop_scope": (
                "all encoder/block states under the final layer frozen mean-head metric; "
                "depth states are not exact-distance shells"
            ),
            "post_lift_scope": (
                "phi(Bx) cannot identify component means; the actual trained post arm is phi(Px), "
                "a different operator retaining constants"
            ),
            "intervention_scope": (
                "same fresh selected checkpoint; full validation; no retraining; "
                "primary causal comparison is the independent retrained eight-arm matrix"
            ),
        }


def audit(root, data_root, device, repeats, *, tolerance=1e-7, iterations=2000):
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
                evaluations.append(train.evaluate(model, inputs, args, device))
                release_amplitude_diagnostics(model)
                _synchronize(device)
                timings.append(time.perf_counter() - started)
            model.capture = True
            observer = Observer(args, tolerance=tolerance, iterations=iterations)
            detailed = train.evaluate(model, inputs, args, device, observer=observer)
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
                "validation": detailed,
                "repeated_validation": {
                    "count": repeats,
                    "scores": scores,
                    "range_pp": 100 * (max(scores) - min(scores)),
                    "evaluation_seconds": timings,
                },
                "diagnostics": observer.report(detailed["metric"]),
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
    parser.add_argument("--cg-tolerance", type=float, default=1e-7)
    parser.add_argument("--cg-iterations", type=int, default=2000)
    args = parser.parse_args(argv)
    if not 0 < args.cg_tolerance < 1 or args.cg_iterations < 1:
        raise ValueError("CG tolerance must be in (0,1) and iterations positive")
    result = audit(
        args.root,
        args.data_root,
        torch.device(args.device),
        args.repeat_evaluations,
        tolerance=args.cg_tolerance,
        iterations=args.cg_iterations,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
