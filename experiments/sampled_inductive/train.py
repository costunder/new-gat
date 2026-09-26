"""CUDA-only training with graph-held-out selection and complete seed coverage."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from chartgat.cache import atomic_write_json
from chartgat.observability import RuntimeResourceMonitor
from experiments.aggregation_comparison import engine
from experiments.aggregation_comparison.validation import require_reproduction, validate_evaluation
from research.conductance_gat.v5.timing import StageTimer

from .data import Coverage, Inputs

SUITE = "sampled_inductive_v1"


def sources():
    result = engine.implementation_source_hashes()
    root = Path(__file__).resolve().parents[2]
    for path in Path(__file__).parent.glob("*.py"):
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def make_inputs(payload, args, mode, batch, workers, *, split="validation"):
    return Inputs(
        payload,
        mode,
        physical_batch=batch,
        workers=workers,
        context_seeds=args.context_seeds,
        fanouts=args.num_neighbors,
        seed=args.model_seed,
        evaluation_split=split,
    )


@contextmanager
def solver_timing(model, timer):
    handles, active = [], {}

    def before(module, values):
        context = timer.stage("c_solver_including_checkpoint_recompute")
        context.__enter__()
        active.setdefault(module, []).append(context)

    def after(module, values, output):
        active[module].pop().__exit__(None, None, None)

    for operator in model.layers:
        handles.extend(
            (
                operator.estimator.register_forward_pre_hook(before),
                operator.estimator.register_forward_hook(after, always_call=True),
            )
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def train_epoch(model, optimizer, inputs, args, device, epoch):
    model.train()
    inputs.batchers["train"].epoch = epoch
    coverage = Coverage(inputs.datasets["train"])
    timer, numerator, decisions, steps = StageTimer(device), torch.zeros((), device=device), 0, 0
    gradients = {}
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    iterator = iter(inputs.loaders["train"])
    with solver_timing(model, timer):
        while True:
            with timer.stage("sampling_and_loader_wait"):
                batch = next(iterator, None)
            if batch is None:
                break
            coverage.add(batch.observations)
            with timer.stage("host_to_device"):
                batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with timer.stage("forward_and_loss"), engine.autocast(args, device):
                output = model(batch.graph)
                chosen = output[batch.seed_mask]
                target = batch.graph.y[batch.seed_mask]
                loss = F.binary_cross_entropy_with_logits(chosen, target)
            with timer.stage("backward"):
                loss.backward()
            if steps == 0:
                gradients = engine.validate_gradients(model)
            with timer.stage("optimizer"):
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), engine.base.COMMON["gradient_clip_norm"], foreach=True
                )
                engine.base.require_finite_gradient_norm_async(norm)
                optimizer.step()
            numerator += loss.detach() * target.numel()
            decisions += target.numel()
            steps += 1
            model.clear_auxiliary_cache()
            del output, chosen, target, loss, batch
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    observed = coverage.finish()
    if steps != len(inputs.batchers["train"]) or not decisions:
        raise ValueError("incomplete training epoch")
    value = float(numerator / decisions)
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite training loss")
    return {
        "epoch": epoch,
        "loss": value,
        "optimizer_steps": steps,
        "training_seconds": elapsed,
        "supervised_nodes_per_second": observed["supervised_nodes"] / elapsed,
        "coverage": observed,
        "stage_seconds": timer.report(),
        "gradient_norms": {k: float(v) for k, v in gradients.items()},
    }


@torch.no_grad()
def evaluate(model, inputs, args, device):
    model.eval()
    totals = torch.zeros(4, dtype=torch.int64, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch in inputs.loaders[inputs.evaluation_split]:
        batch = batch.to(device)
        with engine.autocast(args, device):
            logits = model(batch.graph)
        engine.base.require_finite_tensor(logits, "held-out graph logits")
        pred, labels = logits > 0, batch.graph.y.bool()
        totals += torch.stack(
            (
                (pred & labels).sum(),
                (pred & ~labels).sum(),
                (~pred & labels).sum(),
                torch.tensor(labels.numel(), device=device),
            )
        )
        model.clear_auxiliary_cache()
    tp, fp, fn, total = totals.cpu().tolist()
    result = {
        "metric_kind": "micro_f1",
        "metric": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "total": total},
        "seconds": time.perf_counter() - started,
        "graph_ids": inputs.datasets[inputs.evaluation_split].ids,
    }
    validate_evaluation(result, label="independent held-out graphs")
    return result


def configuration(args, mode):
    config = engine.configuration(args)
    config["sampling"] = "inductive_cluster_contexts" if mode == "sampled" else "full"
    # This runner performs persisted-best count reproduction, not the separate
    # aggregation runner's layer/head diagnostic and intervention audit.
    config["comparison_contract"]["mechanism_audit"] = None
    config["comparison_contract"]["validation_audit"] = (
        "fresh CUDA model loaded from last.pt/best_state; five full count reproductions"
    )
    return config


def identity(args, mode, batch, workers, protocol, inputs):
    return {
        "suite": SUITE,
        "training_support": mode,
        "model": args.ablation_arm,
        "configuration": configuration(args, mode),
        "inputs": inputs.metadata(),
        "protocol": protocol,
        "sources": sources(),
        "versions": engine.base._versions(),
        "budget": {
            "epochs": args.epochs,
            "early_stopping": False,
            "unit": "complete epochs; each training node supervised once",
            "planned_optimizer_steps": args.epochs * len(inputs.batchers["train"]),
            "update_matched_between_support_modes": False,
        },
        "sampling_recipe": {"context_seeds": args.context_seeds, "fanouts": args.num_neighbors},
        "physical_batch": batch,
        "workers": workers,
    }


def completed(folder, expected=None):
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    saved = engine.base.load_checkpoint_on_cpu(folder / "last.pt")
    if metrics.get("status") != "passed" or metrics.get("history") != saved["history"]:
        raise ValueError("passing metrics must match the committed training history")
    if metrics["shared_initial_sha256"] != saved["shared_initial_sha256"]:
        raise ValueError("initialization provenance differs from checkpoint")
    if expected is not None and saved["identity"] != expected:
        raise ValueError("inductive identity changed; use a fresh run ID")
    if saved["identity"]["sources"] != sources() or metrics["identity"] != saved["identity"]:
        raise ValueError("inductive evidence/source mismatch")
    if metrics["last_sha256"] != engine.base.sha256_file(folder / "last.pt"):
        raise ValueError("inductive checkpoint was changed")
    from research.conductance_gat.ablation.model import _named_state_sha256

    expected_audit = {
        "schema_version": 1,
        "checkpoint_sha256": metrics["last_sha256"],
        "best_state_sha256": _named_state_sha256(saved["best_state"].items()),
        "best_epoch": saved["best_epoch"],
        "origin": "last.pt/best_state loaded into a fresh CUDA model",
        "repeats": 5,
    }
    if metrics.get("persisted_best_audit") != expected_audit:
        raise ValueError("missing or mismatched persisted-best CUDA audit evidence")
    history = saved["history"]
    budget = saved["identity"]["budget"]
    if len(history) != budget["epochs"] or [r["epoch"] for r in history] != list(
        range(1, len(history) + 1)
    ):
        raise ValueError("training history did not complete the full declared budget")
    if sum(r["optimizer_steps"] for r in history) != budget["planned_optimizer_steps"]:
        raise ValueError("optimizer update budget mismatch")
    best = max(history, key=lambda r: r["validation"]["metric"])
    if best["epoch"] != saved["best_epoch"] or best["epoch"] != metrics["best_epoch"]:
        raise ValueError("best epoch differs from validation-only selection")
    for row in history:
        validate_evaluation(row["validation"], label="history")
        if row["validation"]["graph_ids"] != saved["identity"]["inputs"]["graph_ids"]["validation"]:
            raise ValueError("history validation graph IDs changed")
        if row["coverage"]["seed_coverage"] != 1.0:
            raise ValueError("incomplete supervised coverage")
        if row["coverage"]["supervised_nodes"] != saved["identity"]["inputs"]["training_nodes"]:
            raise ValueError("supervision count differs from the complete training split")
    require_reproduction(best["validation"], saved["best_validation"], label="best checkpoint")
    if metrics["validation"]["graph_ids"] != saved["identity"]["inputs"]["graph_ids"]["validation"]:
        raise ValueError("evaluation used different held-out graphs")
    require_reproduction(best["validation"], metrics["validation"], label="selected/reloaded")
    for repeat in metrics["audit"]:
        require_reproduction(best["validation"], repeat, label="repeat audit")
        if repeat["graph_ids"] != metrics["validation"]["graph_ids"]:
            raise ValueError("repeat audit used different held-out graphs")
    if len(metrics["audit"]) < 5:
        raise ValueError("missing repeated validation audit")
    return metrics


def train_cell(payload, protocol, args, mode, batch, workers, device, folder):
    engine.base._require_cuda(device)
    engine.validate_args(args)
    engine.base.validate_hardware_runtime(args, device)
    engine.base.configure_compute(args)
    engine.base._seed(args.model_seed)
    if folder.is_symlink() or any(p.is_symlink() for p in folder.parents):
        raise ValueError("indirect output path forbidden")
    monitor = RuntimeResourceMonitor(device)
    monitor.start()
    inputs = None
    started = time.perf_counter()
    try:
        inputs = make_inputs(payload, args, mode, batch, workers)
        contract = identity(args, mode, batch, workers, protocol, inputs)
        if (folder / "metrics.json").exists():
            return completed(folder, contract)
        model = engine.make_model(payload, args, device)
        optimizer = engine.make_optimizer(model, args.learning_rate)
        shared_hash = engine.shared_initial_state_sha256(model)
        history, best_state, best_epoch, best_value, prior_seconds = [], None, 0, -1.0, 0.0
        checkpoint = folder / "last.pt"
        resumed = checkpoint.exists()
        if resumed:
            saved = engine.base.load_checkpoint_on_cpu(checkpoint)
            if saved["identity"] != contract:
                raise ValueError("run identity changed; old checkpoint preserved")
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            engine._restore_rng(saved, device)
            history, best_state, best_epoch = (
                saved["history"],
                saved["best_state"],
                saved["best_epoch"],
            )
            best_value, prior_seconds = saved["best_validation"]["metric"], saved["wall_seconds"]
            del saved
        elif folder.exists() and any(folder.iterdir()):
            raise FileExistsError("nonempty cell without checkpoint; preserving all files")
        folder.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "identity": contract,
                    "model_contract": model.contract(),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "debug": bool(protocol.get("explicit_synthetic_debug", False)),
                    "subset": False,
                    "test_evaluated": False,
                }
            ),
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(len(history) + 1, args.epochs + 1):
            row = train_epoch(model, optimizer, inputs, args, device, epoch)
            row["validation"] = evaluate(model, inputs, args, device)
            history.append(row)
            if row["validation"]["metric"] > best_value:
                best_value, best_epoch = row["validation"]["metric"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Best and last states share a single atomic epoch commit, including all RNG.
            engine.base._save(
                checkpoint,
                {
                    "identity": contract,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_validation": history[best_epoch - 1]["validation"],
                    "wall_seconds": prior_seconds + time.perf_counter() - started,
                    "shared_initial_sha256": shared_hash,
                    **engine._checkpoint_rng(device),
                },
            )
            print(
                f"{mode}/{args.ablation_arm} epoch={epoch} "
                f"val={row['validation']['metric']:.6f} seconds={row['training_seconds']:.2f}",
                flush=True,
            )
        selected = history[best_epoch - 1]["validation"]
        # Discard the in-memory best and training model. A correct in-memory
        # prediction cannot attest to the weights actually committed to disk.
        del optimizer, model, best_state
        gc.collect()
        torch.cuda.empty_cache()
        committed_sha = engine.base.sha256_file(checkpoint)
        saved = engine.base.load_checkpoint_on_cpu(checkpoint)
        if (
            saved["identity"] != contract
            or saved["history"] != history
            or saved["best_epoch"] != best_epoch
        ):
            raise ValueError("persisted checkpoint identity/history/selection changed")
        require_reproduction(selected, saved["best_validation"], label="persisted selection")
        model = engine.make_model(payload, args, device)
        model.load_state_dict(saved["best_state"], strict=True)
        del saved
        repetitions = []
        state_hash = engine.base.state_sha256(model)
        for repeat in range(5):
            validation = evaluate(model, inputs, args, device)
            require_reproduction(selected, validation, label=f"persisted-best audit {repeat}")
            repetitions.append(validation)
        if state_hash != engine.base.state_sha256(model) or contract["sources"] != sources():
            raise ValueError("read-only validation or source changed")
        if engine.base.sha256_file(checkpoint) != committed_sha:
            raise ValueError("persisted checkpoint changed during its CUDA audit")
        result = {
            "status": "passed",
            "identity": contract,
            "model_contract": model.contract(),
            "validation": repetitions[0],
            "audit": repetitions,
            "persisted_best_audit": {
                "schema_version": 1,
                "checkpoint_sha256": committed_sha,
                "best_state_sha256": state_hash,
                "best_epoch": best_epoch,
                "origin": "last.pt/best_state loaded into a fresh CUDA model",
                "repeats": len(repetitions),
            },
            "best_epoch": best_epoch,
            "shared_initial_sha256": shared_hash,
            "test_evaluated": False,
            "last_sha256": committed_sha,
            "history": history,
            "resumed": resumed,
            "cost_comparable": not resumed,
            "wall_seconds_including_setup_train_validation_checkpoints_audit": prior_seconds
            + time.perf_counter()
            - started,
            "resources": monitor.finish(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            ),
        }
        monitor = None
        atomic_write_json(folder / "metrics.json", result)
        return completed(folder, contract)
    finally:
        if inputs is not None:
            inputs.close()
        if monitor is not None:
            monitor.finish(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )


def probe(payload, args, mode, batch, workers, device):
    from research.conductance_gat.v5.batch_calibration import _isolated_execution_state

    engine.base._require_cuda(device)
    engine.base.validate_hardware_runtime(args, device)
    with _isolated_execution_state(device):
        gc.collect()
        torch.cuda.empty_cache()
        engine.base.configure_compute(args)
        engine.base._seed(args.model_seed)
        free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        inputs = model = optimizer = None
        try:
            inputs = make_inputs(payload, args, mode, batch, workers)
            model = engine.make_model(payload, args, device)
            optimizer = engine.make_optimizer(model, args.learning_rate)
            initial = engine.base.state_sha256(model)
            warmup, epoch = 0, 0
            while warmup < 2:
                epoch += 1
                warmup += train_epoch(model, optimizer, inputs, args, device, epoch)[
                    "optimizer_steps"
                ]
            measured, elapsed, rows = 0, 0.0, []
            while measured < 5 or elapsed < 3:
                epoch += 1
                row = train_epoch(model, optimizer, inputs, args, device, epoch)
                measured += row["optimizer_steps"]
                elapsed += row["training_seconds"]
                rows.append(row)
            validation = evaluate(model, inputs, args, device)
            allocated, reserved = (
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            )
            if initial == engine.base.state_sha256(model) or not optimizer.state:
                raise ValueError("calibration did not update a real model/optimizer")
            return {
                "status": "passed",
                "batch": batch,
                "workers": workers,
                "training_support": mode,
                "arm": args.ablation_arm,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "free_before_bytes": free,
                "total_bytes": total,
                "safe": reserved <= 0.9 * free,
                "full_epochs": len(rows),
                "optimizer_steps": measured,
                "epoch_seconds": elapsed / len(rows),
                "validation_seconds": validation["seconds"],
                "epoch_evidence": rows,
                "inputs": inputs.metadata(),
            }
        finally:
            if inputs is not None:
                inputs.close()
            inputs = model = optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
