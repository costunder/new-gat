"""Full-size, epoch-resumable edge selection on official V1 caches; no legacy migration."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from chartgat.cache import atomic_write_json
from chartgat.observability import RuntimeResourceMonitor

from ..v5 import train as base
from ..v5.batch_calibration import (
    _candidate_args,
    _isolated_execution_state,
    _optimizer_state_bytes,
)
from ..v5.learning_budget import should_stop_learning_budget
from ..v5.timing import StageTimer
from . import protocol as selection_protocol
from .audit_compat import require_source_compatibility
from .data import PreparedInputs
from .model import EdgeSelectionClassifier

ROOT = Path(__file__).resolve().parents[3]
SUITE = selection_protocol.SUITE


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        conductance_heads="per_head",
        propagation_normalization="row",
        solver_cost_scaling="width_scaled",
        beta_initial=0.5,
        training_schedule="joint",
    )
    selection_protocol.add_arguments(parser)
    return parser


def validate_args(args):
    base.validate_args(args)
    selection_protocol.validate(args)


def configuration(args):
    return {**base.configuration(args), "edge_selection": selection_protocol.configuration(args)}


def implementation_source_hashes():
    paths = list((ROOT / "research/conductance_gat/edge_selection").glob("*.py"))
    paths += [ROOT / "scripts/run_v5_edge_selection.py"]
    return {
        **base.implementation_source_hashes(),
        **{path.relative_to(ROOT).as_posix(): base.sha256_file(path) for path in sorted(paths)},
    }


def make_model(payload, args, device):
    return EdgeSelectionClassifier(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        **base.architecture_configuration(args),
        conductance_mode="dynamic",
        max_log_conductance=base.COMMON["max_log_conductance"],
        edge_chunk_size=args.edge_chunk_size,
        selection_config=selection_protocol.model_configuration(args),
    ).to(device)


def parameter_group(name):
    if ".selector." in name:
        return "gate"
    return base.parameter_group(name)


def make_optimizer(model):
    options = {
        "backbone": (base.COMMON["lr"], base.COMMON["weight_decay"]),
        "spatial_w": (base.COMMON["lr"], base.COMMON["weight_decay"]),
        "beta": (
            base.COMMON["lr"] * base.COMMON["beta_lr_multiplier"],
            base.COMMON["scalar_weight_decay"],
        ),
        "conductance": (
            base.COMMON["lr"] * base.COMMON["conductance_lr_multiplier"],
            base.COMMON["conductance_weight_decay"],
        ),
        "gate": (
            base.COMMON["lr"] * base.COMMON["conductance_lr_multiplier"],
            base.COMMON["conductance_weight_decay"],
        ),
    }
    grouped = {name: [] for name in options}
    names = {name: [] for name in options}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = parameter_group(name)
            grouped[group].append(parameter)
            names[group].append(name)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": name,
                "params": values,
                "parameter_names": names[name],
                "lr": options[name][0],
                "weight_decay": options[name][1],
            }
            for name, values in grouped.items()
            if values
        ],
        lr=base.COMMON["lr"],
    )
    expected = {id(value) for value in model.parameters() if value.requires_grad}
    actual = [id(value) for group in optimizer.param_groups for value in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError("edge-selection optimizer ownership is not exactly once per parameter")
    return optimizer


def shared_initial_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if ".selector." not in name:
            digest.update(name.encode())
            digest.update(base.tensor_hash(value).encode())
    return digest.hexdigest()


def resolve_budget(inputs, args):
    if inputs.indices is not None:
        return base.resolve_learning_budget(inputs.data, inputs.indices, inputs.sampler, args)
    return base.resolve_learning_budget(inputs.data, None, None, args)


def build_identity(args, protocol, budget, initial_hash, inputs):
    return {
        "schema_version": 1,
        "research_suite": SUITE,
        "dataset": args.dataset,
        "condition": args.selection_mode,
        "configuration": configuration(args),
        "training_arguments": serializable_arguments(args),
        "dataset_protocol": protocol,
        "dataset_protocol_sha256": base._canonical_sha256(protocol),
        "source_sha256": implementation_source_hashes(),
        "runtime_versions": base._versions(),
        "initial_state_sha256": initial_hash,
        "learning_budget": budget,
        "input_provenance": inputs.provenance,
        "resume_semantics": (
            "epoch-boundary model/optimizer/Python/NumPy/CPU/CUDA RNG restore; "
            "no bitwise CUDA claim"
        ),
    }


def serializable_arguments(args):
    return {
        key: False if key == "resume" else str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def restore_arguments(metrics, output, data_root, device):
    identity = metrics["resume_identity"]
    saved = identity.get("training_arguments")
    if not isinstance(saved, dict):
        raise ValueError("audit requires immutable training arguments")
    args = argparse.Namespace(**saved)
    for name in ("output_dir", "data_root"):
        setattr(args, name, Path(getattr(args, name)))
    validate_args(args)
    if configuration(args) != identity["configuration"]:
        raise ValueError("restored CLI arguments do not reproduce the trained configuration")
    args.output_dir, args.data_root, args.device = Path(output), Path(data_root), str(device)
    return args


def validate_identity(saved, expected):
    identity = saved.get("resume_identity")
    if not isinstance(identity, dict) or base._canonical_sha256(identity) != saved.get(
        "resume_identity_sha256"
    ):
        raise ValueError("edge-selection checkpoint identity is absent or corrupt")
    if identity != expected:
        changed = sorted(
            key for key in set(identity) | set(expected) if identity.get(key) != expected.get(key)
        )
        raise ValueError(
            f"edge-selection resume identity mismatch: {changed}; old evidence preserved"
        )


def resolve_training_resume(saved, expected):
    """Preserve the original identity; admit only a pinned infrastructure repair.

    Model, optimizer, data, sampling, arguments, budget and runtime must still
    agree exactly. Actual resumed execution sources are recorded separately.
    """
    identity = saved.get("resume_identity")
    validate_identity(saved, identity)
    changed = sorted(
        key
        for key in set(identity) | set(expected)
        if key != "source_sha256" and identity.get(key) != expected.get(key)
    )
    if changed:
        raise ValueError(
            f"edge-selection resume identity mismatch: {changed}; old evidence preserved"
        )
    proof = require_source_compatibility(
        identity.get("source_sha256"), expected.get("source_sha256"), scope="training"
    )
    return copy.deepcopy(identity), proof


def autocast(args, device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"
    )


def loss_components(model, batch, logits, args):
    task, count = base.training_loss(logits, batch.graph, batch.selected_indices)
    targets = batch.origin_targets if args.negative_loss_weight else None
    auxiliary = model.auxiliary_loss(targets)
    loss = (
        task + args.l0_weight * auxiliary["l0"] + args.negative_loss_weight * auxiliary["negative"]
    )
    return loss, task, auxiliary, int(count)


def validate_gradients(model):
    missing = [
        name
        for name, value in model.named_parameters()
        if value.requires_grad and value.grad is None
    ]
    if missing:
        raise RuntimeError(f"trainable parameters disconnected from task/auxiliary loss: {missing}")
    by_group = {}
    for name, value in model.named_parameters():
        if value.grad is not None:
            by_group.setdefault(parameter_group(name), []).append(
                value.grad.detach().float().square().sum()
            )
    result = {name: torch.stack(values).sum().sqrt() for name, values in by_group.items()}
    for name, norm in result.items():
        torch._assert_async(torch.isfinite(norm), f"nonfinite {name} gradient")
    return result


def run_training_epoch(
    model, optimizer, inputs, args, device, epoch, *, timing=None, validate=False
):
    model.train()
    timing = timing or StageTimer(device)
    sums = torch.zeros(4, device=device, dtype=torch.float64)
    labels = steps = units = 0
    largest_nodes = largest_edges = largest_graphs = 0
    observations = []
    gradient_rows = {}
    iterator = iter(inputs.training_batches(epoch, device))
    while True:
        with timing.stage("sampling_forest_loader_transfer"):
            batch = next(iterator, None)
        if batch is None:
            break
        with timing.stage("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with timing.stage("forward_and_loss"):
            with autocast(args, device):
                logits = model(batch.graph)
                loss, task, auxiliary, count = loss_components(model, batch, logits, args)
        with timing.stage("backward"):
            loss.backward()
        if validate and steps == 0:
            gradient_rows = validate_gradients(model)
        with timing.stage("gradient_clipping"):
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                base.COMMON["gradient_clip_norm"],
                error_if_nonfinite=False,
                foreach=True,
            )
            base.require_finite_gradient_norm_async(norm)
        with timing.stage("optimizer"):
            optimizer.step()
        sums += (
            torch.stack(
                (
                    loss.detach(),
                    task.detach(),
                    auxiliary["l0"].detach(),
                    auxiliary["negative"].detach(),
                )
            ).double()
            * count
        )
        labels += count
        steps += 1
        graph_count = int(batch.graph._v5_num_graphs)
        units += graph_count if inputs.indices is None else count
        largest_nodes = max(largest_nodes, batch.graph.x.shape[0])
        largest_edges = max(largest_edges, batch.graph.incidence_edge_index.shape[1])
        largest_graphs = max(largest_graphs, graph_count)
        observations.append(
            {
                "nodes": batch.graph.x.shape[0],
                "candidate_edges": batch.graph.incidence_edge_index.shape[1],
                "disjoint_graphs": graph_count,
                "supervised_labels": count,
                "sampling": getattr(batch.graph, "sampling_observation", None),
            }
        )
        model.clear_auxiliary_cache()
        del loss, task, auxiliary, logits, batch
    if labels < 1 or steps < 1:
        raise RuntimeError("official training split produced no supervised updates")
    values = (sums / labels).cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("nonfinite training/auxiliary epoch loss")
    return {
        "train_loss": values[0],
        "train_task_loss": values[1],
        "train_l0": values[2],
        "train_negative_loss": values[3],
        "train_labels": labels,
        "train_batches": steps,
        "optimizer_steps": steps,
        "processed_units": units,
        "largest_measured_nodes": largest_nodes,
        "largest_measured_physical_edges": largest_edges,
        "largest_measured_graph_batch": largest_graphs,
        "batch_observations": observations,
        "first_step_gradient_norms": {
            name: float(value.cpu()) for name, value in gradient_rows.items()
        },
    }


@torch.no_grad()
def evaluate(model, inputs, args, device, *, observer=None):
    model.eval()
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    batches = 0
    for batch in inputs.validation_batches(device):
        with autocast(args, device):
            logits = model(batch.graph)
            task, count = base.training_loss(logits, batch.graph, batch.selected_indices)
        base.require_finite_tensor(logits, "edge-selection validation logits")
        if batch.selected_indices is None:
            pred, target = logits > 0, batch.graph.y.bool()
            numbers = ((pred & target).sum(), (pred & ~target).sum(), (~pred & target).sum())
        else:
            chosen = logits[batch.selected_indices]
            target = batch.graph.y[batch.selected_indices]
            numbers = (
                (chosen.argmax(-1) == target).sum(),
                target.new_zeros(()),
                target.new_zeros(()),
            )
        totals[:3] += torch.stack(numbers)
        totals[3] += count
        totals[4] += task.double() * count
        totals[5] += torch.isfinite(task).double()
        if observer is not None:
            observer(model, batch, logits, batches)
        model.clear_auxiliary_cache()
        batches += 1
    first, fp, fn, count, loss, finite = totals.cpu().tolist()
    if count <= 0 or finite != batches:
        raise ValueError("validation is empty or contains a nonfinite loss")
    metric = (
        (2 * first / (2 * first + fp + fn) if 2 * first + fp + fn else 0.0)
        if inputs.indices is None
        else first / count
    )
    return {"metric": metric, "loss": loss / count, "label_count": int(count), "batches": batches}


def _checkpoint_rng(device):
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
    }


def _restore_rng(saved, device):
    random.setstate(saved["python_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    base.restore_checkpoint_rng(saved, device)


def inspect_completed(output):
    from .integrity import inspect_completed as inspect_evidence

    return inspect_evidence(output)


def train_model(payload, protocol, args, device, output):
    base._require_cuda(device)
    validate_args(args)
    base.validate_cached_graphs_once(payload)
    if payload["dataset"] != args.dataset:
        raise ValueError("dataset request and verified cache disagree")
    hardware = base.validate_hardware_runtime(args, device)
    base.configure_compute(args)
    base._seed(args.model_seed)
    monitor = RuntimeResourceMonitor(device)
    monitor.start()
    finished = False
    try:
        inputs = PreparedInputs(payload, args)
        budget = resolve_budget(inputs, args)
        model = make_model(payload, args, device)
        initial_hash = base.state_sha256(model)
        shared_hash = shared_initial_state_sha256(model)
        optimizer = make_optimizer(model)
        identity = build_identity(args, protocol, budget, initial_hash, inputs)
        identity_hash = base._canonical_sha256(identity)
        execution_sources = copy.deepcopy(identity["source_sha256"])
        source_transitions = []
        last_path, best_path, previous_path = (
            output / "last.pt",
            output / "best.pt",
            output / "best.previous.pt",
        )
        history = []
        best_metric, best_epoch, best_hash, steps = -math.inf, 0, None, 0
        if last_path.exists():
            if not args.resume or last_path.is_symlink():
                raise ValueError("existing checkpoint cannot be replaced without a valid resume")
            saved = base.load_checkpoint_on_cpu(last_path)
            identity, source_proof = resolve_training_resume(saved, identity)
            identity_hash = base._canonical_sha256(identity)
            source_transitions = copy.deepcopy(saved.get("source_transitions", []))
            if not isinstance(source_transitions, list):
                raise ValueError("checkpoint source transition evidence must be a list")
            source_transitions.append(
                {
                    "after_epoch": saved["epoch"],
                    "optimizer_steps": saved["optimizer_steps"],
                    "source_sha256": execution_sources,
                    "source_compatibility": source_proof,
                    "restored_checkpoint_sha256": base.sha256_file(last_path),
                    "hardware": hardware,
                    "scope": "epoch-boundary continuation; original training identity retained",
                }
            )
            history = saved["history"]
            if [row.get("epoch") for row in history] != list(range(1, saved["epoch"] + 1)):
                raise ValueError("resume history has missing or repeated epochs")
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            best_metric, best_epoch = saved["best_validation"], saved["best_epoch"]
            best_hash, steps = saved["best_checkpoint_sha256"], saved["optimizer_steps"]
            base.recover_best_checkpoint(best_path, previous_path, best_hash)
            _restore_rng(saved, device)
            del saved
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError(
                "nonempty selection output has no valid last.pt; no files overwritten"
            )
        output.mkdir(parents=True, exist_ok=True)
        pre_run = {
            "research_suite": SUITE,
            "configuration": configuration(args),
            "hardware": hardware,
            "parameters": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            "optimizer_groups": base.optimizer_metadata(optimizer),
            "learning_budget": budget,
            "data": base._v5_data_observability(payload, inputs.data, inputs.indices, args),
            "topology": inputs.metadata(),
            "debug": False,
            "subset": False,
            "test_evaluated": False,
            "batching": {
                "physical_batch_size": args.batch_size
                if inputs.indices is None
                else (args.sample_seed_batch_size if inputs.sampler is not None else 1),
                "batch_axis": "graphs"
                if inputs.indices is None
                else ("supervised_seed_nodes" if inputs.sampler is not None else "complete_graph"),
                "gradient_accumulation_steps": 1,
                "data_parallel_workers": 1,
                "full_graph_exception": (
                    "one complete transductive graph, not serial independent samples"
                )
                if inputs.indices is not None and inputs.sampler is None
                else None,
            },
        }
        if not (output / "configuration.json").exists():
            atomic_write_json(output / "configuration.json", pre_run)
        print(json.dumps(pre_run, sort_keys=True), flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(len(history) + 1, budget["planned_epochs"] + 1):
            if history and should_stop_learning_budget(
                budget,
                epochs_since_best=history[-1]["epoch"] - best_epoch,
                optimizer_steps_since_best=steps - history[best_epoch - 1]["optimizer_steps"],
                eligible=True,
            ):
                break
            started = time.perf_counter()
            timing = StageTimer(device)
            values = run_training_epoch(
                model, optimizer, inputs, args, device, epoch, timing=timing, validate=True
            )
            if values["train_batches"] != budget["actual_batches_per_epoch"]:
                raise RuntimeError(
                    "measured supervised batches differ from the immutable update budget"
                )
            steps += values.pop("optimizer_steps")
            with timing.stage("validation"):
                validation = evaluate(model, inputs, args, device)
            row = {
                **values,
                "epoch": epoch,
                "phase": {"phase": "joint"},
                "optimizer_steps": steps,
                "validation": validation["metric"],
                "validation_loss": validation["loss"],
                "stage_seconds": timing.report(synchronize=True),
                "elapsed_wall_seconds": time.perf_counter() - started,
                "topology_preparation_seconds_cumulative": inputs.plan_preparation_seconds,
            }
            history.append(row)
            if validation["metric"] > best_metric:
                best_metric, best_epoch = validation["metric"], epoch
                best_hash = base.publish_best_checkpoint(
                    best_path,
                    previous_path,
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "validation": best_metric,
                        "selection_role": "primary",
                        "resume_identity": identity,
                        "resume_identity_sha256": identity_hash,
                        "execution_source_sha256": execution_sources,
                        "source_transitions": source_transitions,
                    },
                )
            base._save(
                last_path,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "history": history,
                    "optimizer_steps": steps,
                    "best_validation": best_metric,
                    "best_epoch": best_epoch,
                    "best_checkpoint_sha256": best_hash,
                    "resume_identity": identity,
                    "resume_identity_sha256": identity_hash,
                    "execution_source_sha256": execution_sources,
                    "source_transitions": source_transitions,
                    "shared_initial_state_sha256": shared_hash,
                    **_checkpoint_rng(device),
                },
            )
            atomic_write_json(output / "history.json", history)
            print(
                f"{args.dataset}/{args.selection_mode} epoch={epoch} "
                f"loss={row['train_loss']:.6f} task={row['train_task_loss']:.6f} "
                f"val={row['validation']:.6f} best={best_metric:.6f} "
                f"seconds={row['elapsed_wall_seconds']:.2f}",
                flush=True,
            )
        if not history or best_epoch < 1:
            raise RuntimeError("training completed without valid epoch/selection evidence")
        atomic_write_json(output / "history.json", history)
        selected = base.load_checkpoint_on_cpu(best_path)
        validate_identity(selected, identity)
        if selected["epoch"] != best_epoch or selected["validation"] != best_metric:
            raise ValueError("best checkpoint selection disagrees with last.pt")
        model.load_state_dict(selected["model_state"], strict=True)
        del selected
        final_validation = evaluate(model, inputs, args, device)
        resources = monitor.finish(
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
        finished = True
        result = {
            "schema_version": 1,
            "status": "passed",
            "research_suite": SUITE,
            "dataset": args.dataset,
            "condition": args.selection_mode,
            "configuration": configuration(args),
            "protocol": protocol,
            "source_sha256": identity["source_sha256"],
            "execution_source_sha256": execution_sources,
            "source_transitions": source_transitions,
            "resume_identity": identity,
            "resume_identity_sha256": identity_hash,
            "learning_budget": budget,
            "initial_state_sha256": initial_hash,
            "shared_initial_state_sha256": shared_hash,
            "common_backbone_initial_state_sha256": shared_hash,
            "epochs_run": len(history),
            "optimizer_steps": steps,
            "best_epoch": best_epoch,
            "best_validation": best_metric,
            "validation": final_validation["metric"],
            "validation_loss": final_validation["loss"],
            "checkpoint_sha256": base.sha256_file(best_path),
            "last_checkpoint_sha256": base.sha256_file(last_path),
            "history_sha256": base.sha256_file(output / "history.json"),
            "resource_observability": resources,
            "topology": inputs.metadata(),
            "test_evaluated": False,
            "debug": False,
            "subset": False,
        }
        atomic_write_json(output / "metrics.json", result)
        return result
    except BaseException as error:
        if not finished:
            try:
                resources = monitor.finish(
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                )
                finished = True
                if output.is_dir():
                    atomic_write_json(
                        output / "failure-resources.json",
                        {"error": f"{type(error).__name__}: {error}", "resources": resources},
                    )
            except BaseException as reporting_error:
                error.add_note(f"failure telemetry also failed: {reporting_error}")
        raise


def load_calibration_payload(args):
    payload, protocol = base.load_dataset(args.dataset, args.data_root, allow_download=False)
    maximum = (
        len(payload["splits"]["train"])
        if args.dataset == "ppi"
        else (int(payload["splits"]["train"].count_nonzero()) if args.sampling != "full" else 1)
    )
    axis = (
        "graphs"
        if args.dataset == "ppi"
        else ("sampled_seed_nodes" if args.sampling != "full" else "full_graph")
    )
    identity = {
        "dataset": args.dataset,
        "data_sha256": protocol["data_sha256"],
        "split_sha256": protocol["split_sha256"],
        "protocol": protocol,
        "corruption_ratio": args.corruption_ratio,
        "corruption_seed": args.corruption_seed,
    }
    return {"payload": payload, "protocol": protocol}, identity, maximum, axis


def run_calibration_candidate(
    loaded,
    args,
    device,
    *,
    physical_batch_size,
    workers,
    warmup_steps=2,
    measurement_steps=5,
    minimum_measure_seconds=3.0,
):
    base._require_cuda(device)
    if warmup_steps < 2 or measurement_steps < 5 or minimum_measure_seconds < 3:
        raise ValueError(
            "resource probes require complete-epoch windows meeting the calibration minima"
        )
    candidate = _candidate_args(args, physical_batch_size, workers)
    validate_args(candidate)
    model = optimizer = inputs = None
    monitor = None
    report = None
    with _isolated_execution_state(device):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            hardware = base.validate_hardware_runtime(candidate, device)
            free_before, total = torch.cuda.mem_get_info(device)
            torch.cuda.reset_peak_memory_stats(device)
            monitor = RuntimeResourceMonitor(device)
            monitor.start()
            base.configure_compute(candidate)
            base._seed(candidate.model_seed)
            started = time.perf_counter()
            inputs = PreparedInputs(loaded["payload"], candidate)
            model = make_model(loaded["payload"], candidate, device)
            optimizer = make_optimizer(model)
            initial_hash = base.state_sha256(model)
            # Match the persistent full-validation cache used during production training.
            if inputs.indices is not None:
                next(inputs.validation_batches(device))
            torch.cuda.synchronize(device)
            setup_seconds = time.perf_counter() - started
            stress = None
            if inputs.indices is None:
                stress_started = time.perf_counter()
                stress_batch = inputs.stress_batch(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(candidate, device):
                    stress_logits = model(stress_batch.graph)
                    stress_loss, stress_task, _auxiliary, _count = loss_components(
                        model, stress_batch, stress_logits, candidate
                    )
                stress_loss.backward()
                validate_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), base.COMMON["gradient_clip_norm"], foreach=True
                )
                base.require_finite_gradient_norm_async(norm)
                optimizer.step()
                torch.cuda.synchronize(device)
                stress = {
                    "graphs": int(stress_batch.graph._v5_num_graphs),
                    "nodes": stress_batch.graph.x.shape[0],
                    "physical_edges": stress_batch.graph.incidence_edge_index.shape[1],
                    "seconds": time.perf_counter() - stress_started,
                    "optimizer_steps": 1,
                    "scope": (
                        "additional largest-graph joint-batch stress; outside throughput window"
                    ),
                }
                model.clear_auxiliary_cache()
                del stress_batch, stress_logits, stress_loss, stress_task, _auxiliary
            warmup_epochs = warmup_updates = 0
            while warmup_updates < warmup_steps:
                warmup_epochs += 1
                values = run_training_epoch(
                    model, optimizer, inputs, candidate, device, warmup_epochs, validate=True
                )
                warmup_updates += values["optimizer_steps"]
            timing = StageTimer(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            measured_epochs = measured_steps = units = labels = 0
            largest_nodes = largest_edges = largest_graphs = 0
            elapsed = 0.0
            while measured_steps < measurement_steps or elapsed < minimum_measure_seconds:
                measured_epochs += 1
                values = run_training_epoch(
                    model,
                    optimizer,
                    inputs,
                    candidate,
                    device,
                    warmup_epochs + measured_epochs,
                    timing=timing,
                )
                measured_steps += values["optimizer_steps"]
                units += values["processed_units"]
                labels += values["train_labels"]
                largest_nodes = max(largest_nodes, values["largest_measured_nodes"])
                largest_edges = max(largest_edges, values["largest_measured_physical_edges"])
                largest_graphs = max(largest_graphs, values["largest_measured_graph_batch"])
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
            started = time.perf_counter()
            evaluate(model, inputs, candidate, device)
            torch.cuda.synchronize(device)
            validation_seconds = time.perf_counter() - started
            free_after, _ = torch.cuda.mem_get_info(device)
            report = {
                "status": "passed",
                "calibration_not_final": True,
                "elapsed_seconds": elapsed,
                "processed_units": units,
                "samples_per_second": units / elapsed,
                "unit": "graphs" if inputs.indices is None else "supervised_seed_nodes",
                "optimizer_steps": measured_steps,
                "complete_measurement_epochs": measured_epochs,
                "complete_warmup_epochs": warmup_epochs,
                "warmup_optimizer_steps": warmup_updates,
                "warmup_steps_requested": warmup_steps,
                "measurement_steps_requested": measurement_steps,
                "minimum_measure_seconds_requested": minimum_measure_seconds,
                "stage_seconds": timing.report(),
                "large_graph_batch_stress": stress,
                "setup_seconds": setup_seconds,
                "topology_preparation_seconds": inputs.plan_preparation_seconds,
                "validation_seconds": validation_seconds,
                "validation_completed": True,
                "auxiliary_path_measured": True,
                "cycle_preparation_measured": candidate.selection_mode == "forest_cycle",
                "batch_size": physical_batch_size,
                "workers": workers,
                "configuration": configuration(candidate),
                "hardware": hardware,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "free_bytes_before": int(free_before),
                "free_bytes_after": int(free_after),
                "total_memory_bytes": int(total),
                "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
                "model_parameter_count": sum(p.numel() for p in model.parameters()),
                "initial_model_sha256": initial_hash,
                "parameter_update_verified": initial_hash != base.state_sha256(model),
                "supervised_labels": labels,
                "largest_measured_nodes": largest_nodes,
                "largest_measured_physical_edges": largest_edges,
                "largest_measured_graph_batch": largest_graphs,
                "gradient_accumulation_steps": 1,
                "data_parallel_workers": 1,
                "effective_batch_size": physical_batch_size,
                "scope": (
                    "disposable complete official training epochs + full validation + "
                    "topology preparation; no test or checkpoints"
                ),
            }
            if not report["parameter_update_verified"] or report["optimizer_state_bytes"] <= 0:
                raise RuntimeError(
                    "calibration failed to verify actual optimizer state and parameter update"
                )
        except BaseException as error:
            if monitor is not None:
                failed_monitor, monitor = monitor, None
                try:
                    error.calibration_resource_observability = failed_monitor.finish(
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    )
                except BaseException as report_error:
                    error.add_note(f"probe telemetry failed: {report_error}")
            raise
        finally:
            try:
                if monitor is not None:
                    resources = monitor.finish(
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    )
                    if report is not None:
                        report["resource_observability"] = resources
            finally:
                model = optimizer = inputs = None
                gc.collect()
                torch.cuda.empty_cache()
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    output, data_root = (
        args.output_dir.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
    )
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("selection output must not overlap the immutable official data cache")
    if args.output_dir.is_symlink() or any(path.is_symlink() for path in args.output_dir.parents):
        raise ValueError("selection output must not be indirect")
    payload, protocol = base.load_dataset(args.dataset, data_root, allow_download=False)
    train_model(payload, protocol, args, torch.device(args.device), output)
    print(f"passed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
