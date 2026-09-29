"""Output initialization ablation; inherited full training and evidence contract."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import psutil
import torch
from torch.nn import functional as F

from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.evidence import require_approval
from experiments.c_learning_bracket.evaluate import evaluate
from experiments.c_learning_bracket.inputs import BracketInputs as StudyInputs
from experiments.c_learning_bracket.inspect_c import UpdateInspection
from research.conductance_gat.v5.timing import StageTimer

from . import SUITE
from .model import INITIALIZATIONS, make_model, non_output_digest

CONDITIONS = ("learned", "fixed")


class RecipeParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        requested = super().parse_args(args, namespace)
        inherited = engine.build_parser().parse_args(
            [
                "--dataset",
                "ogbn-arxiv",
                "--condition",
                "shared_dynamic_c",
                "--ablation-arm",
                "incidence",
                "--selection-mode",
                "full",
                "--output-dir",
                str(requested.output_dir),
            ]
        )
        inherited.complete_supervised_passes = True
        inherited.resume = False
        inherited.sampling = "cluster_disjoint"
        vars(inherited).update(vars(requested))
        for name in list(vars(inherited)):
            if name.startswith("solver_"):
                delattr(inherited, name)
        inherited.conductance_backend = "bracket"
        inherited.conductance_generator = "symmetric_dot_exp"
        inherited.ablation_arm = "bracket"
        return inherited


def parser():
    result = RecipeParser(description=__doc__)
    result.add_argument("--output-initialization", choices=INITIALIZATIONS, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--data-root", type=Path, default=Path("data/paper"))
    result.add_argument("--device", default="cuda")
    result.add_argument("--model-seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--sample-seed-batch-size", type=int, required=True)
    result.add_argument("--sample-context-seed-batch-size", type=int, required=True)
    result.add_argument("--sample-context-workers", type=int, required=True)
    result.add_argument("--edge-chunk-size", type=int, required=True)
    result.add_argument("--precision", choices=("fp32", "bf16"), default=argparse.SUPPRESS)
    result.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    result.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    result.add_argument(
        "--sample-prefetch", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    result.add_argument(
        "--activation-checkpoint", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    result.add_argument("--action", choices=("calibrate", "train"), required=True)
    result.add_argument("--physical-seed-candidates", nargs="+", type=int)
    result.add_argument("--context-worker-candidates", nargs="+", type=int)
    result.add_argument("--calibration-repeats", type=int, default=4)
    result.add_argument("--calibration-report", type=Path)
    result.add_argument("--inspection-example-nodes", type=int, default=4)
    result.add_argument("--eval-context-seeds", nargs="+", type=int, required=True)
    result.add_argument("--checkpoint-edges", action=argparse.BooleanOptionalAction, default=True)
    return result


def validate_shared_data_recipe(args):
    # The legacy validator also validates its own generator enum. Supply those
    # inactive fields only to its temporary data/hardware-validation adapter.
    # They are never model arguments or the recorded scientific configuration.
    proxy = copy.deepcopy(args)
    proxy.ablation_arm = "incidence"
    proxy.conductance_backend, proxy.conductance_generator = "optimization", "optimized"
    proxy.solver_steps, proxy.solver_step_size = 8, 0.25
    proxy.solver_entropy, proxy.solver_degree_barrier = 1.0, 0.1
    proxy.solver_cost_scaling = "width_scaled"
    engine.validate_args(proxy)
    for name, value in vars(proxy).items():
        if not name.startswith("solver_") and name not in {
            "ablation_arm",
            "conductance_backend",
            "conductance_generator",
        }:
            setattr(args, name, value)


def validate(args):
    validate_shared_data_recipe(args)
    required = {
        "dataset": "ogbn-arxiv",
        "ablation_arm": "bracket",
        "condition": "shared_dynamic_c",
        "visibility_protocol": "official_transductive",
        "sampling": "cluster_disjoint",
        "layers": 8,
        "hidden_channels": 256,
        "heads": 8,
        "conductance_backend": "bracket",
        "conductance_generator": "symmetric_dot_exp",
        "conductance_heads": "per_head",
        "propagation_normalization": "row",
        "propagation_filter": "linear",
        "training_schedule": "joint",
        "learning_budget_policy": "epochs",
        "complete_supervised_passes": True,
        "resume": False,
        "num_relations": 0,
        "learning_rate": engine.base.COMMON["lr"],
        "dropout": engine.base.COMMON["dropout"],
        "beta_parameterization": "sigmoid",
        "beta_initial": 0.5,
    }
    differences = {k: (getattr(args, k), v) for k, v in required.items() if getattr(args, k) != v}
    if differences:
        raise ValueError(
            f"bracket preserves the specified comparison recipe (actual, required): {differences}"
        )
    if (
        args.transition_from_checkpoint is not None
        or args.beta_min is not None
        or args.beta_max is not None
    ):
        raise ValueError("bracket starts fresh with unchanged beta and backbone initializers")
    if args.epochs < 200 or not str(args.device).startswith("cuda"):
        raise ValueError("production requires CUDA and >=200 complete epochs; tests are separate")
    if args.inspection_example_nodes < 1 or args.calibration_repeats < 4:
        raise ValueError(
            "positive display count; calibration needs warmup, two plain and one observed step"
        )
    if not args.eval_context_seeds or min(args.eval_context_seeds) < 1:
        raise ValueError("declare positive frozen evaluation context sizes")
    if args.action == "calibrate":
        for name in ("physical_seed_candidates", "context_worker_candidates"):
            values = getattr(args, name)
            if not values or len(set(values)) < 2 or min(values) < 1:
                raise ValueError(f"measure at least two distinct positive {name}")
        if min(args.context_worker_candidates) < 2:
            raise ValueError("measure parallel context construction")
    elif args.calibration_report is None:
        raise ValueError("training requires this suite's measured calibration report")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def save(path, value):
    with Path(path).open("xb") as handle:
        torch.save(value, handle)


def resources(device):
    memory, process = psutil.virtual_memory(), psutil.Process()
    return {
        "gpu": torch.cuda.get_device_name(device),
        "visible_gpu_count": torch.cuda.device_count(),
        "vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "steady_allocated": torch.cuda.memory_allocated(device),
        "cpu_logical": os.cpu_count(),
        "cpu_affinity": process.cpu_affinity(),
        "cpu_percent_snapshot": psutil.cpu_percent(),
        "ram_total": memory.total,
        "ram_available": memory.available,
        "process_ram": process.memory_info().rss,
        "cpu_times": process.cpu_times()._asdict(),
    }


def research_contract(args, protocol):
    hashes = engine.implementation_source_hashes()
    for path in (engine.ROOT / "experiments/c_learning_only").glob("*.py"):
        hashes[path.relative_to(engine.ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for path in (engine.ROOT / "experiments/c_learning_bracket").glob("*.py"):
        hashes[path.relative_to(engine.ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for path in Path(__file__).parent.glob("*.py"):
        hashes[path.relative_to(engine.ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    fields = (
        "dataset",
        "model_seed",
        "output_initialization",
        "epochs",
        "layers",
        "hidden_channels",
        "heads",
        "dropout",
        "learning_rate",
        "precision",
        "tf32",
        "device",
        "hardware_profile",
        "activation_checkpoint",
        "checkpoint_edges",
        "edge_chunk_size",
        "pin_memory",
        "sample_prefetch",
        "sampling",
        "sample_context_seed_batch_size",
        "num_neighbors",
        "beta_initial",
        "beta_parameterization",
        "forest_seed",
        "selection_mode",
        "visibility_protocol",
        "learning_budget_policy",
        "complete_supervised_passes",
    )
    configuration = {name: getattr(args, name) for name in fields}
    configuration.update(
        {
            "generator": "symmetric_dot_exp",
            "head_aggregation": "none; per-head C",
            "score_scale": "1/head_width",
            "query_key_initialization": "Xavier uniform; zero bias",
            "weight_decay": engine.base.COMMON["weight_decay"],
            "gradient_clip_norm": engine.base.COMMON["gradient_clip_norm"],
            "data_parallel_workers": 1,
            "gradient_accumulation_steps": 1,
        }
    )
    # Execution choices are separately matched against measured candidate rows.
    for name in ("sample_seed_batch_size", "sample_context_workers"):
        configuration.pop(name, None)
    return {
        "suite": SUITE,
        "implementation_revision": "output_init_ablation_1",
        "input_pipeline": "bracket tensors without forest/cycle plans; verified CPU pinning",
        "dataset_protocol": protocol,
        "configuration": configuration,
        "source_sha256": hashes,
        "conditions": CONDITIONS,
        "fixed_layer_support": True,
        "inspection_example_nodes": args.inspection_example_nodes,
        "eval_context_seeds": args.eval_context_seeds,
    }


def common_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if ".estimator." not in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def step(model, batch, optimizer, example_nodes=None, timer=None):
    def phase(name):
        return nullcontext() if timer is None else timer.stage(name)

    model.train()
    with phase("zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with phase("inspection_setup") if example_nodes is not None else nullcontext():
        observation = (
            None
            if example_nodes is None
            else UpdateInspection(
                model,
                example_nodes,
                optimizer=optimizer,
            )
        )
    label = "_with_observation" if observation is not None else ""
    with phase("forward_and_ce" + label):
        logits = model(batch.graph, observer=observation)
        ids = batch.selected_indices
        target = batch.graph.y[ids].reshape(-1)
        loss = F.cross_entropy(logits[ids], target)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite classification loss")
    with phase("backward" + label):
        loss.backward()
    with phase("gradient_validation_and_clipping"):
        disconnected = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        if disconnected:
            raise RuntimeError(f"parameters disconnected from CE: {disconnected}")
        if observation is not None:
            observation.record_parameter_gradients(model, "before_clipping")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), engine.base.COMMON["gradient_clip_norm"], error_if_nonfinite=True
        )
        if observation is not None:
            observation.record_parameter_gradients(model, "after_clipping")
    with phase("optimizer"):
        optimizer.step()
    if observation is not None:
        with phase("inspection_same_input_replay"):
            report = observation.finish(model, batch.graph)
    else:
        report = None
    return loss.detach(), target.numel(), report


def batch_shape(batch):
    graph = batch.graph
    groups = graph.batch
    nodes = torch.bincount(groups).cpu().tolist()
    edges = (
        torch.bincount(groups[graph.incidence_edge_index[0]], minlength=len(nodes)).cpu().tolist()
    )
    return {
        "input_shape": list(graph.x.shape),
        "context_nodes": nodes,
        "context_edges": edges,
        "physical_contexts": len(nodes),
        "supervised_seeds": batch.selected_indices.numel(),
        "transfer_evidence": getattr(batch, "transfer_evidence", None),
    }


def context_inputs(payload, args, context):
    view = dict(payload)
    view["splits"] = dict(payload["splits"])
    view["splits"]["train"] = view["splits"]["validation"]
    config = copy.deepcopy(args)
    config.sample_context_seed_batch_size = context
    config.sample_seed_batch_size = math.ceil(args.sample_seed_batch_size / context) * context
    return StudyInputs(view, config)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def timed_step(model, batch, optimizer, args, device, observed):
    timer = StageTimer(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with engine.autocast(args, device):
        loss, count, inspection = step(
            model,
            batch,
            optimizer,
            args.inspection_example_nodes if observed else None,
            timer=timer,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "observed": observed,
        "step_wall_seconds": elapsed,
        "timing": timer.report(),
        "loss": float(loss),
        "supervised_seeds": count,
    }, inspection


def paired_timing(model, batch, optimizer, args, device):
    # Snapshot/restore time is excluded. Both paths receive identical weights,
    # optimizer moments, dropout RNG and physical sampled graph.
    state = cpu_copy(model.state_dict())
    optimizer_state = cpu_copy(optimizer.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    plain, _ = timed_step(model, batch, optimizer, args, device, False)
    model.load_state_dict(state)
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)
    observed, inspection = timed_step(model, batch, optimizer, args, device, True)
    # Small parallel-reduction differences are permitted, not a changed formula.
    if not math.isclose(plain["loss"], observed["loss"], rel_tol=1e-5, abs_tol=1e-7):
        raise RuntimeError("observation timing replay did not reproduce the plain loss")
    return {
        "plain": plain,
        "observed": observed,
        "observation_added_wall_seconds": observed["step_wall_seconds"]
        - plain["step_wall_seconds"],
        "same_batch_state_optimizer_rng": True,
    }, inspection


def calibration_evaluations(model, inputs, payload, args, device):
    """Preserve frozen metrics and verified global validation-ID coverage."""
    expected = inputs.indices["validation"]
    full = evaluate(
        model, inputs.validation_batches(device), intervention=True, expected_seed_ids=expected
    )
    contexts = {}
    for context in args.eval_context_seeds:
        frozen_inputs = context_inputs(payload, args, context)
        result = evaluate(
            model,
            frozen_inputs.training_batches(args.epochs + 1, device),
            intervention=True,
            expected_seed_ids=expected,
        )
        if result["parameter_sha256"] != full["parameter_sha256"]:
            raise RuntimeError("parameters changed between frozen evaluation contexts")
        result["sampling_evidence"] = frozen_inputs.last_pass_evidence
        result["context_seed_batch_size"] = context
        result["physical_seed_batch_size"] = frozen_inputs.args.sample_seed_batch_size
        result["sampling_pass_role"] = "validation seeds; no optimizer update"
        contexts[str(context)] = result
        del frozen_inputs
    return {"full_validation": full, "new_sampling_contexts": contexts}


def calibrate(payload, args, protocol, output, device):
    from .progress import announce

    contract = research_contract(args, protocol)
    rows = []
    for physical in args.physical_seed_candidates:
        for workers in args.context_worker_candidates:
            selected = copy.deepcopy(args)
            selected.sample_seed_batch_size, selected.sample_context_workers = physical, workers
            validate_shared_data_recipe(selected)
            for condition in CONDITIONS:
                announce(f"calibration {condition} batch={physical} workers={workers} | preparing")
                inputs = StudyInputs(payload, selected)
                model = make_model(payload, selected, device, condition)
                optimizer = engine.make_optimizer(model, selected.learning_rate)
                iterator = inputs.training_batches(0, device)
                torch.cuda.reset_peak_memory_stats(device)
                pairs, sampling_seconds = [], []
                shape = None
                for index in range(args.calibration_repeats):
                    announce(
                        f"calibration {condition} batch={physical} workers={workers} | "
                        f"measurement {index + 1}/{args.calibration_repeats}"
                    )
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = inputs.training_batches(index + 1, device)
                        batch = next(iterator)
                    torch.cuda.synchronize(device)
                    load_seconds = time.perf_counter() - start
                    if index == 0:
                        warmup, _ = timed_step(model, batch, optimizer, selected, device, False)
                    else:
                        pair, inspection = paired_timing(model, batch, optimizer, selected, device)
                        pair["sampling_and_transfer_wall_seconds"] = load_seconds
                        pair["transfer_evidence"] = getattr(batch, "transfer_evidence", None)
                        pairs.append(pair)
                        sampling_seconds.append(load_seconds)
                        write_json(
                            output / f"timing-{condition}-{physical}-{workers}-{index}.json", pair
                        )
                        if index == 1:
                            write_json(
                                output / f"inspection-{condition}-{physical}-{workers}.json",
                                {**inspection, "scope": "real-data calibration only"},
                            )
                    shape = batch_shape(batch)
                # Full validation/intervention also has to fit; it is never a training subset.
                announce(f"calibration {condition} | full validation and new-context evaluations")
                evaluation = calibration_evaluations(model, inputs, payload, selected, device)
                row = {
                    "condition": condition,
                    "physical_seed_batch": physical,
                    "context_workers": workers,
                    "supervised_nodes_per_second": sum(
                        pair["plain"]["supervised_seeds"] for pair in pairs
                    )
                    / (
                        sum(sampling_seconds)
                        + sum(pair["plain"]["step_wall_seconds"] for pair in pairs)
                    ),
                    "warmup": warmup,
                    "paired_timings": pairs,
                    "throughput_scope": "ordinary step plus exposed loader/transfer wait",
                    "observation_timing": "paired replay; excluded from ordinary throughput",
                    "full_validation_measured": True,
                    "frozen_context_sizes_measured": args.eval_context_seeds,
                    "validation_nodes": evaluation["full_validation"]["total"],
                    "evaluation": evaluation,
                    "resources": resources(device),
                    "last_batch": shape,
                    "model": model.contract(),
                    "inputs": inputs.metadata(),
                }
                rows.append(row)
                write_json(output / f"calibration-{condition}-{physical}-{workers}.json", row)
                print(json.dumps(row), flush=True)
                del iterator, optimizer, model, inputs, batch, inspection
                gc.collect()
                torch.cuda.empty_cache()
    if contract != research_contract(args, protocol):
        raise RuntimeError("source or scientific configuration changed during calibration")
    report = {
        "scope": "real-data calibration only; no trained checkpoint or performance conclusion",
        "contract": contract,
        "measurements": rows,
    }
    write_json(output / "calibration.json", report)
    return report


def validate_calibration(args, protocol, device):
    report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(research_contract(args, protocol), default=str))
    if report["contract"] != expected:
        raise ValueError("calibration source/science contract differs from this run")
    rows = [
        row
        for row in report["measurements"]
        if row["physical_seed_batch"] == args.sample_seed_batch_size
        and row["context_workers"] == args.sample_context_workers
        and row["resources"]["gpu"] == torch.cuda.get_device_name(device)
    ]
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("physical batch/worker choice must be measured for both conditions")
    memory = torch.cuda.get_device_properties(device).total_memory
    if any(
        row["resources"]["peak_reserved"] > memory * 0.9
        or not row["full_validation_measured"]
        or row["frozen_context_sizes_measured"] != args.eval_context_seeds
        for row in rows
    ):
        raise ValueError(
            "calibration needs 10% VRAM headroom including inspection and full validation"
        )


def train_one(payload, args, condition, output, device, paired=None):
    from .progress import announce

    announce(f"TRAIN {condition} | {args.epochs} epochs | preparing model and initial validation")
    debug = bool(payload.get("explicit_synthetic_debug", False))
    inputs = StudyInputs(payload, args)
    model = make_model(payload, args, device, condition)
    initial = common_digest(model)
    if paired is not None and paired["initial_common_sha256"] != initial:
        raise RuntimeError("common initialization differs between learned and fixed C")
    optimizer = engine.make_optimizer(model, args.learning_rate)
    initial_validation = evaluate(
        model,
        inputs.validation_batches(device),
        intervention=True,
        expected_seed_ids=inputs.indices["validation"],
    )
    write_json(
        output / f"{condition}-initial.json",
        {
            "debug": debug,
            "model": model.contract(),
            "common_sha256": initial,
            "initial_non_output_sha256": non_output_digest(model),
            "validation": initial_validation,
            "inputs": inputs.metadata(),
            "full_input_shape": list(inputs.data.x.shape),
            "full_physical_edges": inputs.data.incidence_edge_index.shape[1],
            "train_nodes_used_per_epoch": inputs.train_count,
            "train_coverage": 1.0,
            "epochs": args.epochs,
            "optimization_steps": args.epochs * len(inputs.sampler),
        },
    )
    history, best = [], None
    for epoch in range(args.epochs):
        # Each condition sees identical epoch-specific dropout and sample streams.
        engine.base._seed(args.model_seed + 1_000_003 * (epoch + 1))
        start = time.perf_counter()
        last_progress = start
        announce(f"{condition} epoch {epoch + 1}/{args.epochs} | starting training")
        total_loss = torch.zeros((), device=device)
        total = steps = 0
        torch.cuda.reset_peak_memory_stats(device)
        for batch in inputs.training_batches(epoch, device):
            with engine.autocast(args, device):
                loss, count, observation = step(
                    model, batch, optimizer, args.inspection_example_nodes if steps == 0 else None
                )
            total_loss += loss * count
            total += count
            if observation is not None:
                write_json(
                    output / f"{condition}-inspection-{epoch + 1:04d}.json",
                    {
                        "debug": debug,
                        "epoch": epoch + 1,
                        "batch": batch_shape(batch),
                        **observation,
                    },
                )
            steps += 1
            now = time.perf_counter()
            if steps == 1 or steps == len(inputs.sampler) or now - last_progress >= 30:
                announce(
                    f"{condition} epoch {epoch + 1}/{args.epochs} | "
                    f"batch {steps}/{len(inputs.sampler)} | "
                    f"supervised seeds {total}/{inputs.train_count} | elapsed {now - start:.0f}s"
                )
                last_progress = now
        if total != inputs.train_count:
            raise RuntimeError("incomplete supervised epoch")
        evidence = inputs.last_pass_evidence
        if paired is not None and evidence != paired["history"][epoch]["sampling_evidence"]:
            raise RuntimeError(
                "conditions did not receive identical sampled contexts and train seeds"
            )
        announce(f"{condition} epoch {epoch + 1}/{args.epochs} | validating")
        validation = evaluate(
            model,
            inputs.validation_batches(device),
            expected_seed_ids=inputs.indices["validation"],
        )
        torch.cuda.synchronize(device)
        row = {
            "debug": debug,
            "epoch": epoch + 1,
            "loss": float(total_loss / total),
            "supervised_nodes": total,
            "optimization_steps": steps,
            "validation": validation,
            "sampling_evidence": evidence,
            "seconds_including_inspection_validation": time.perf_counter() - start,
            "resources": resources(device),
        }
        history.append(row)
        write_json(output / f"{condition}-epoch-{epoch + 1:04d}.json", row)
        print(json.dumps({"condition": condition, **row}), flush=True)
        if best is None or validation["correct"] > best["validation"]["correct"]:
            path = output / f"{condition}-best-{epoch + 1:04d}.pt"
            save(
                path,
                {
                    "suite": SUITE,
                    "debug": debug,
                    "condition": condition,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "model_contract": model.contract(),
                },
            )
            best = {
                "path": str(path),
                "validation": validation,
                "epoch": epoch + 1,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    result = {"debug": debug, "initial_common_sha256": initial, "best": best, "history": history}
    write_json(output / f"{condition}-trained.json", result)
    return result


def evaluate_selected(payload, args, selected, output, device):
    write_json(output / "frozen-checkpoints.json", {k: v["best"] for k, v in selected.items()})
    results = {}
    for condition, record in selected.items():
        best = record["best"]
        path = Path(best["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != best["file_sha256"]:
            raise ValueError("selected checkpoint changed")
        model = make_model(payload, args, device, condition)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)["state_dict"], strict=True
        )
        inputs = StudyInputs(payload, args)
        expected = inputs.indices["validation"].clone()
        full = evaluate(
            model,
            inputs.validation_batches(device),
            intervention=True,
            expected_seed_ids=expected,
        )
        if (
            full["correct"] != best["validation"]["correct"]
            or full["parameter_sha256"] != best["validation"]["parameter_sha256"]
        ):
            raise RuntimeError("selected checkpoint failed to reproduce validation")
        results[condition] = {"full_validation": full, "new_sampling_contexts": {}}
        del inputs
        for context in args.eval_context_seeds:
            inputs = context_inputs(payload, args, context)
            result = evaluate(
                model,
                inputs.training_batches(args.epochs + 1, device),
                intervention=True,
                expected_seed_ids=expected,
            )
            if result["parameter_sha256"] != full["parameter_sha256"]:
                raise RuntimeError("parameters changed between frozen evaluation contexts")
            result["sampling_evidence"] = inputs.last_pass_evidence
            results[condition]["new_sampling_contexts"][str(context)] = result
            del inputs
        del model
        gc.collect()
    for context in map(str, args.eval_context_seeds):
        if (
            results["learned"]["new_sampling_contexts"][context]["sampling_evidence"]
            != results["fixed"]["new_sampling_contexts"][context]["sampling_evidence"]
        ):
            raise RuntimeError("frozen context evaluation is not paired")
    write_json(
        output / "evaluation.json",
        {
            "debug": bool(payload.get("explicit_synthetic_debug", False)),
            "results": results,
            "test_set_evaluated": False,
            "scope": "same-graph validation and new sampling contexts; no independent-graph claim",
            "validation_accuracy_difference": results["learned"]["full_validation"]["accuracy"]
            - results["fixed"]["full_validation"]["accuracy"],
            "run_completed": True,
            "c_benefit_confirmed": None,
            "interpretation": "review C updates, intervention and validation separately",
        },
    )
    return results


def main(argv=None):
    from .progress import announce

    args = parser().parse_args(argv)
    announce(f"{args.output_initialization} {args.action} | checking CUDA and data")
    validate(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback for production")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("single-GPU execution only; configure multi-GPU scheduling explicitly")
    device = torch.device(args.device)
    engine.base.configure_compute(args)
    engine.base.validate_hardware_runtime(args, device)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("use a fresh run directory; existing results are preserved")
    payload, protocol = engine.load_dataset(args)
    require_approval(protocol)
    if args.action == "train":
        validate_calibration(args, protocol, device)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "contract.json",
        {
            **research_contract(args, protocol),
            "arguments": vars(args),
            "resources": resources(device),
            "physical_batch_unit": "supervised seed nodes in disjoint-union contexts",
            "physical_batch_size": args.sample_seed_batch_size,
            "effective_batch_size": args.sample_seed_batch_size,
            "gradient_accumulation_steps": 1,
            "data_parallel_workers": 1,
            "debug": False,
            "subset": False,
            "epoch_policy": "complete passes; no early stopping",
            "loader": "cached graph adjacency, context thread pool, verified CPU pinning",
            "not_applicable": ["image resolution", "time window", "DataLoader worker pool"],
            "scope_change": "only output initialization differs; accepted bracket source preserved",
        },
    )
    try:
        frozen_contract = research_contract(args, protocol)
        if args.action == "calibrate":
            return calibrate(payload, args, protocol, output, device)
        selected = {}
        for condition in CONDITIONS:
            selected[condition] = train_one(
                payload, args, condition, output, device, selected.get("learned")
            )
            gc.collect()
            if research_contract(args, protocol) != frozen_contract:
                raise RuntimeError("source or scientific configuration changed during training")
        return evaluate_selected(payload, args, selected, output, device)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "exception": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "action": args.action,
                "formula_changed": False,
                "fallback_used": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
