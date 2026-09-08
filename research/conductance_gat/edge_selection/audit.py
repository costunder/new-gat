"""Read-only full-validation gate, amplitude, topology and intervention audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from chartgat.observability import RuntimeResourceMonitor

from ..v5.batch_calibration import _isolated_execution_state
from ..v5.operator import conductance_propagation_coefficients
from . import diagnostics as diag
from . import train


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_amplitude_diagnostics(model):
    """Post-audit-only cache release; gates and learned state remain unchanged."""
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("diagnostic cache release is eval/no-grad only")
    model.clear_auxiliary_cache()
    for operator in model.operators:
        operator.last_r = operator.last_effective_c = None
        operator.estimator.last_c = operator.estimator.last_log_c = None
        operator.estimator.last_scores = None
        operator.estimator.last_solver_diagnostics = {}


class PredictionSummary:
    def __init__(self, multilabel):
        self.multilabel = multilabel
        self.rows = []

    def add(self, logits, baseline, batch):
        selected = batch.selected_indices
        target = batch.graph.y if selected is None else batch.graph.y[selected]
        current = logits if selected is None else logits[selected]
        original = baseline if selected is None else baseline[selected]
        if self.multilabel:
            predicted, truth = current > 0, target.bool()
            counts = (
                (predicted & truth).sum(),
                (predicted & ~truth).sum(),
                (~predicted & truth).sum(),
            )
            flips = ((current > 0) != (original > 0)).sum()
            decisions = current.numel()
        else:
            counts = (
                (current.argmax(-1) == target).sum(),
                target.new_zeros(()),
                target.new_zeros(()),
            )
            flips = (current.argmax(-1) != original.argmax(-1)).sum()
            decisions = target.numel()
        values = (
            *counts,
            flips,
            torch.tensor(decisions, device=current.device, dtype=torch.float64),
            (current.float() - original.float()).square().sum(),
            original.float().square().sum(),
        )
        self.rows.append(torch.stack([value.detach().double() for value in values]))

    def report(self):
        a, fp, fn, flips, decisions, difference, baseline = (
            torch.stack(self.rows).sum(0).cpu().tolist()
        )
        denominator = 2 * a + fp + fn
        metric = (2 * a / denominator if denominator else 0.0) if self.multilabel else a / decisions
        return {
            "validation": metric,
            "prediction_flip_fraction": flips / decisions,
            "logit_relative_l2": (difference / baseline) ** 0.5 if baseline > 0 else None,
        }


class Observer:
    def __init__(self, args, path_sources):
        self.args, self.path_sources = args, path_sources
        self.records = []
        names = ["all_gates_open", "frozen_gates_amplitude_one"]
        if args.selection_mode in train.selection_protocol.BUDGET_MODES:
            names.append("random_same_chord_budget")
        self.predictions = {name: PredictionSummary(args.dataset == "ppi") for name in names}
        self.intervention_seconds = {name: 0.0 for name in names}

    @torch.no_grad()
    def __call__(self, model, batch, logits, batch_index):
        plan = batch.topology
        incidence = batch.graph.incidence_edge_index
        edges = incidence.cpu().numpy()
        graph_nodes = plan.node_graph.cpu().numpy()
        graph_edges = plan.edge_graph.cpu().numpy()
        origin = batch.origin_targets.cpu()
        references = {}
        for group in range(plan.num_graphs):
            nodes = np.flatnonzero(graph_nodes == group)
            chosen = np.flatnonzero(graph_edges == group)
            local = np.searchsorted(nodes, edges[:, chosen])
            matrix = diag.adjacency(nodes.size, local)
            references[group] = (
                nodes,
                chosen,
                local,
                diag.topology_statistics(matrix),
                diag.path_reference(matrix, sources=self.path_sources, seed=self.args.forest_seed),
            )
        frozen = [operator.last_gate.detach().clone() for operator in model.operators]
        for layer, operator in enumerate(model.operators):
            gate = operator.last_gate.cpu()
            amplitude, effective = operator.last_r, operator.last_effective_c
            tail_alpha, head_alpha, _ = conductance_propagation_coefficients(
                effective,
                incidence,
                batch.graph.x.shape[0],
                sampling_correction=operator.last_sampling_correction,
                normalization="row",
                edge_chunk_size=self.args.edge_chunk_size,
            )
            row = {
                "batch": batch_index,
                "layer": layer,
                "gate": diag.distribution(gate),
                "positive_amplitude_r": diag.distribution(amplitude),
                "effective_c_zr": diag.distribution(effective),
                "alpha": diag.coefficient_statistics(
                    tail_alpha, head_alpha, *incidence, batch.graph.x.shape[0]
                ),
                "beta": diag.distribution(operator.last_beta),
                "graphs": [],
            }
            for group, (nodes, chosen, local, before, reference) in references.items():
                active = diag.adjacency(nodes.size, local, gate[chosen].numpy() > 0)
                row["graphs"].append(
                    {
                        "graph_in_batch": group,
                        "before": before,
                        "after": diag.topology_statistics(active),
                        "origins": diag.gate_origins(gate[chosen], origin[chosen]),
                        "path_change": diag.path_change(active, reference),
                    }
                )
            self.records.append(row)
            del tail_alpha, head_alpha, amplitude, effective
        release_amplitude_diagnostics(model)
        variants = {"all_gates_open": ("all", False), "frozen_gates_amplitude_one": (frozen, True)}
        if "random_same_chord_budget" in self.predictions:
            variants["random_same_chord_budget"] = ("random_budget", False)
        for name, (mode, ones) in variants.items():
            _synchronize(logits.device)
            started = time.perf_counter()
            with (
                model.gate_intervention(mode, amplitude_ones=ones),
                train.autocast(self.args, logits.device),
            ):
                changed = model(batch.graph)
            train.base.require_finite_tensor(changed, f"audit intervention {name}")
            _synchronize(logits.device)
            self.intervention_seconds[name] += time.perf_counter() - started
            self.predictions[name].add(changed, logits, batch)

    def report(self, baseline):
        interventions = {}
        for name, values in self.predictions.items():
            result = values.report()
            interventions[name] = {
                **result,
                "delta_validation_pp": 100 * (result["validation"] - baseline),
                "model_forward_seconds": self.intervention_seconds[name],
            }
        return {
            "layers_and_graphs": self.records,
            "interventions": interventions,
            "intervention_scope": (
                "same selected checkpoint and complete validation labels; "
                "no retraining; r=1 keeps every baseline gate fixed"
            ),
            "amplitude_scope": (
                "r is positive candidate-support solver output; z learns via "
                "task/auxiliary loss, not an inner gated-solver objective"
            ),
        }


def audit(root, data_root, device, repeats, path_sources):
    if repeats < 5:
        raise ValueError(
            "at least five repeated full validations are required for numerical-noise measurement"
        )
    metrics = train.inspect_completed(root)
    identity = metrics["resume_identity"]
    if identity["source_sha256"] != train.implementation_source_hashes():
        raise ValueError(
            "audit source differs from the trained implementation; evidence not relabeled"
        )
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
                raise ValueError(
                    "audit edge corruption or topology provenance differs from training"
                )
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
            scores = [row["metric"] for row in evaluations]
            observer = Observer(args, path_sources)
            detailed = train.evaluate(model, inputs, args, device, observer=observer)
            clean = (
                train.evaluate(model, inputs.clean_validation(payload), args, device)
                if args.corruption_ratio
                else detailed
            )
            if before != train.base.state_sha256(model):
                raise ValueError("read-only audit unexpectedly changed model state")
            if train.inspect_completed(root) != metrics:
                raise ValueError("training evidence changed while auditing")
            return {
                "status": "passed",
                "research_suite": train.SUITE,
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "source_sha256": identity["source_sha256"],
                "dataset": args.dataset,
                "condition": args.selection_mode,
                "test_evaluated": False,
                "validation": detailed,
                "clean_validation": clean,
                "repeated_validation": {
                    "count": repeats,
                    "scores": scores,
                    "min": min(scores),
                    "max": max(scores),
                    "range_pp": 100 * (max(scores) - min(scores)),
                    "evaluation_seconds": timings,
                },
                "selection_semantics": model.selection_metadata(),
                "diagnostics": observer.report(detailed["metric"]),
                "automatic_sparse_speedup_claimed": False,
                "physical_compute_scope": (
                    "candidate edges remain materialized so zero gates retain "
                    "gradient paths; exact zeros do not imply sparse-kernel speedup"
                ),
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
    parser.add_argument(
        "--path-probe-sources",
        type=int,
        default=32,
        help="diagnostic landmarks only; evaluation still uses all validation labels",
    )
    args = parser.parse_args(argv)
    result = audit(
        args.root,
        args.data_root,
        torch.device(args.device),
        args.repeat_evaluations,
        args.path_probe_sources,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    print(
        f"Validation={result['validation']['metric']:.6f}; "
        f"clean={result['clean_validation']['metric']:.6f}; no test evaluated",
        flush=True,
    )
    for name, row in result["diagnostics"]["interventions"].items():
        print(
            f"{name}: validation={row['validation']:.6f}, "
            f"delta={row['delta_validation_pp']:+.4f} pp",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
