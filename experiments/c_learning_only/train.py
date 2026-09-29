"""Independent v2.0 calibration and paired full-epoch training entry point."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import psutil
import torch
from torch.nn import functional as F

from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.evidence import require_approval
from experiments.aggregation_comparison.study_inputs import StudyInputs

from . import SUITE
from .evaluate import evaluate
from .inspect_c import UpdateInspection
from .model import make_model

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
        return inherited


def parser():
    result = RecipeParser(description=__doc__)
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
    return result


def validate(args):
    engine.validate_args(args)
    required = {
        "dataset": "ogbn-arxiv",
        "ablation_arm": "incidence",
        "condition": "shared_dynamic_c",
        "visibility_protocol": "official_transductive",
        "sampling": "cluster_disjoint",
        "layers": 8,
        "hidden_channels": 256,
        "heads": 8,
        "conductance_backend": "optimization",
        "conductance_generator": "optimized",
        "solver_steps": 8,
        "solver_step_size": 0.25,
        "solver_entropy": 1.0,
        "solver_degree_barrier": 0.1,
        "solver_cost_scaling": "width_scaled",
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
            f"v2.0 preserves the first comparison recipe (actual, required): {differences}"
        )
    if (
        args.transition_from_checkpoint is not None
        or args.beta_min is not None
        or args.beta_max is not None
    ):
        raise ValueError("v2.0 starts fresh with unchanged beta and initialization")
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
    for path in Path(__file__).parent.glob("*.py"):
        hashes[path.relative_to(engine.ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    configuration = engine.configuration(args)
    # Execution choices are separately matched against measured candidate rows.
    for name in ("sample_seed_batch_size", "sample_context_workers"):
        configuration.pop(name, None)
    return {
        "suite": SUITE,
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


def step(model, batch, optimizer, example_nodes=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observation = None if example_nodes is None else UpdateInspection(model, example_nodes)
    logits = model(batch.graph, observer=observation)
    ids = batch.selected_indices
    target = batch.graph.y[ids].reshape(-1)
    loss = F.cross_entropy(logits[ids], target)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite classification loss")
    loss.backward()
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
    optimizer.step()
    report = None if observation is None else observation.finish(model, batch.graph)
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
    }


def context_inputs(payload, args, context):
    view = dict(payload)
    view["splits"] = dict(payload["splits"])
    view["splits"]["train"] = view["splits"]["validation"]
    config = copy.deepcopy(args)
    config.sample_context_seed_batch_size = context
    config.sample_seed_batch_size = math.ceil(args.sample_seed_batch_size / context) * context
    return StudyInputs(view, config)


def calibrate(payload, args, protocol, output, device):
    contract = research_contract(args, protocol)
    rows = []
    for physical in args.physical_seed_candidates:
        for workers in args.context_worker_candidates:
            selected = copy.deepcopy(args)
            selected.sample_seed_batch_size, selected.sample_context_workers = physical, workers
            engine.validate_args(selected)
            for condition in CONDITIONS:
                inputs = StudyInputs(payload, selected)
                model = make_model(payload, selected, device, condition)
                optimizer = engine.make_optimizer(model, selected.learning_rate)
                iterator = inputs.training_batches(0, device)
                torch.cuda.reset_peak_memory_stats(device)
                times, counts = [], []
                shape = None
                for index in range(args.calibration_repeats):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = inputs.training_batches(index + 1, device)
                        batch = next(iterator)
                    with engine.autocast(selected, device):
                        _, count, inspection = step(
                            model,
                            batch,
                            optimizer,
                            args.inspection_example_nodes
                            if index == args.calibration_repeats - 1
                            else None,
                        )
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - start)
                    counts.append(count)
                    shape = batch_shape(batch)
                # Full validation/intervention also has to fit; it is never a training subset.
                validation = evaluate(model, inputs.validation_batches(device), intervention=True)
                for context in args.eval_context_seeds:
                    frozen_inputs = context_inputs(payload, selected, context)
                    evaluate(
                        model,
                        frozen_inputs.training_batches(args.epochs + 1, device),
                        intervention=True,
                    )
                    del frozen_inputs
                row = {
                    "condition": condition,
                    "physical_seed_batch": physical,
                    "context_workers": workers,
                    "supervised_nodes_per_second": sum(counts[1:-1]) / sum(times[1:-1]),
                    "step_seconds": times,
                    "last_step_includes_inspection": True,
                    "full_validation_measured": True,
                    "frozen_context_sizes_measured": args.eval_context_seeds,
                    "validation_nodes": validation["total"],
                    "resources": resources(device),
                    "last_batch": shape,
                    "model": model.contract(),
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
    debug = bool(payload.get("explicit_synthetic_debug", False))
    inputs = StudyInputs(payload, args)
    model = make_model(payload, args, device, condition)
    initial = common_digest(model)
    if paired is not None and paired["initial_common_sha256"] != initial:
        raise RuntimeError("common initialization differs between learned and fixed C")
    optimizer = engine.make_optimizer(model, args.learning_rate)
    initial_validation = evaluate(model, inputs.validation_batches(device), intervention=True)
    write_json(
        output / f"{condition}-initial.json",
        {
            "debug": debug,
            "model": model.contract(),
            "common_sha256": initial,
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
        if total != inputs.train_count:
            raise RuntimeError("incomplete supervised epoch")
        evidence = inputs.last_pass_evidence
        if paired is not None and evidence != paired["history"][epoch]["sampling_evidence"]:
            raise RuntimeError(
                "conditions did not receive identical sampled contexts and train seeds"
            )
        validation = evaluate(model, inputs.validation_batches(device))
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
        full = evaluate(model, inputs.validation_batches(device), intervention=True)
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
                model, inputs.training_batches(args.epochs + 1, device), intervention=True
            )
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
    args = parser().parse_args(argv)
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
            "loader": "cached topology, context thread pool, configured prefetch and pinning",
            "not_applicable": ["image resolution", "time window", "DataLoader worker pool"],
            "scope_change": "C-only, fixed layer support, two conditions; old candidate preserved",
        },
    )
    if args.action == "calibrate":
        return calibrate(payload, args, protocol, output, device)
    selected = {}
    for condition in CONDITIONS:
        selected[condition] = train_one(
            payload, args, condition, output, device, selected.get("learned")
        )
        gc.collect()
    return evaluate_selected(payload, args, selected, output, device)


if __name__ == "__main__":
    main()
