"""Fresh independently packed training for local E/J placement retraining.

Every reported run processes the complete graph and all local correspondences for its full
contracted budget. Resume accepts only this experiment's identical source,
configuration, graph and seed packing; all previous experiments remain untouched.
"""

from __future__ import annotations

import contextlib
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from ...wedge_propagation.classification.evaluation import classification_metrics, split_indices
from ...wedge_propagation.study import Tee, synchronize
from .common import (
    cpu_state,
    digest,
    file_sha256,
    read_json,
    save_checkpoint,
    write_json,
)
from .model import PackedClassifier, active_layers, parse_condition


def make_model(graph, condition, seeds, config, path_chunk):
    model = PackedClassifier(
        condition,
        graph.num_features,
        graph.num_classes,
        seeds,
        hidden=config["backbone"]["hidden_dim"],
        dropout=config["backbone"]["dropout_probability"],
        path_chunk=path_chunk,
        checkpoint_paths=True,
        dataset_name=graph.name,
    ).to(device=graph.x.device, dtype=graph.x.dtype)
    _mode, variant, _placement = parse_condition(condition)
    widths = (config["backbone"]["hidden_dim"], graph.num_classes)
    channels = sum(widths[layer] for layer in active_layers(condition))
    branches = int(variant in ("within", "both")) + int(variant in ("between", "both"))
    expected = graph.num_features * config["backbone"]["hidden_dim"]
    expected += config["backbone"]["hidden_dim"] * graph.num_classes + 2
    expected += branches * channels
    if model.parameters_per_seed != expected:
        raise ValueError(f"active parameter count mismatch: {graph.name}/{condition}")
    return model


def make_optimizer(model, config, lr):
    train = config["training"]
    return torch.optim.Adam(
        model.weight_decay_groups(train["weight_decay"]),
        lr=lr,
        betas=tuple(train["optimizer_betas"]),
        eps=train["optimizer_epsilon"],
        foreach=False,
    )


def _epoch(model, graph, optimizer, epoch, telemetry=True):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(graph, epoch=epoch)
    indices = split_indices(graph)
    ce, acc = classification_metrics(
        logits,
        graph.y,
        graph.train_mask,
        validate=False,
        indices=indices["train"],
    )
    ce.sum().backward()
    named = list(model.named_parameters())
    for name, parameter in named:
        if parameter.grad is None:
            raise RuntimeError(f"trainable parameter disconnected from train CE: {name}")
    grad = {}
    before = {}
    if telemetry:
        for label, token in (
            ("within", "energy_lifts."),
            ("between", "relation_lifts."),
            ("all", ""),
        ):
            values = [
                p.grad.flatten(1).square().sum(1) if p.ndim > 1 else p.grad.square()
                for name, p in named
                if token in name
            ]
            grad[label] = torch.stack(values).sum(0).sqrt() if values else ce.new_zeros(ce.shape)
        before = {
            name: p.detach().clone()
            for name, p in named
            if name.startswith(("energy_lifts.", "relation_lifts."))
        }
    optimizer.step()
    update = {}
    if telemetry:
        for label, token in (("within", "energy_lifts."), ("between", "relation_lifts.")):
            values = [
                (p.detach() - before[name]).flatten(1).square().sum(1)
                if p.ndim > 1
                else (p.detach() - before[name]).square()
                for name, p in named
                if name.startswith(token)
            ]
            update[label] = torch.stack(values).sum(0).sqrt() if values else ce.new_zeros(ce.shape)
    model.eval()
    with torch.no_grad():
        validation, details = model(graph)
        val_ce, val_acc = classification_metrics(
            validation,
            graph.y,
            graph.val_mask,
            validate=False,
            indices=indices["validation"],
        )
    # One packed transfer for logging, finite guards, gradient and selection data.
    columns = {
        "train_ce": ce.detach(),
        "train_accuracy": acc.detach(),
        "validation_ce": val_ce,
        "validation_accuracy": val_acc,
        "finite_logits": torch.isfinite(logits.detach()).flatten(1).all(1).to(ce.dtype),
        "finite_validation_logits": torch.isfinite(validation).flatten(1).all(1).to(ce.dtype),
    }
    if telemetry:
        columns.update({f"{key}_ce_gradient_norm": value for key, value in grad.items()})
        columns.update({f"{key}_parameter_update_norm": value for key, value in update.items()})
    for layer, detail in enumerate(details):
        for key, value in detail.items():
            if isinstance(value, torch.Tensor) and value.shape == ce.shape:
                columns[f"layer_{layer}_{key}"] = value.detach()
    values = torch.stack(list(columns.values()), 1).detach().cpu().numpy()
    if not np.isfinite(values).all() or not np.all(values[:, 4:6] == 1):
        raise FloatingPointError(
            "nonfinite logits/loss/gradient/update; preserving failure artifacts"
        )
    return [{key: float(value) for key, value in zip(columns, row, strict=True)} for row in values]


def benchmark_trial(graph, condition, seeds, lr, config, path_chunk):
    model = make_model(graph, condition, seeds, config, path_chunk)
    optimizer = make_optimizer(model, config, lr)
    runtime = config["runtime"]
    for epoch in range(runtime["calibration_warmups"]):
        _epoch(model, graph, optimizer, epoch, telemetry=False)
    synchronize(graph.x.device)
    if graph.x.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(graph.x.device)
    start = time.perf_counter()
    for epoch in range(runtime["calibration_repeats"]):
        _epoch(model, graph, optimizer, epoch, telemetry=False)
    synchronize(graph.x.device)
    seconds = (time.perf_counter() - start) / runtime["calibration_repeats"]
    peak = (
        torch.cuda.max_memory_allocated(graph.x.device) if graph.x.device.type == "cuda" else None
    )
    return seconds, peak, model.parameters_per_seed


def choose_packing(graph, condition, seeds, lr, config, phase):
    candidates = config["resources"][f"parallel_{phase}_run_candidates"]
    candidates = sorted({min(value, len(seeds)) for value in candidates})
    paths = graph.topology.num_pairs
    # Chunking is exact; each candidate still processes every path and feature.
    chunks = sorted(
        {
            max(1, min(paths, value))
            for value in [*config["runtime"]["relation_chunk_candidates"], max(1, paths)]
        }
    )
    if condition.endswith("__base"):
        chunks = [max(1, paths)]
    trials = []
    device = graph.x.device
    for packed in candidates:
        for chunk in chunks:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(device)
                baseline = torch.cuda.memory_allocated(device)
            else:
                free = baseline = None
            row = {
                "dataset": graph.name,
                "condition": condition,
                "placement": parse_condition(condition)[2],
                "phase": phase,
                "scope": "calibration",
                "device": str(device),
                "packed_runs": packed,
                "path_chunk": chunk,
                "all_relations": paths,
                "measured": False,
            }
            try:
                seconds, peak, parameters = benchmark_trial(
                    graph,
                    condition,
                    seeds[:packed],
                    lr,
                    config,
                    chunk,
                )
                safe = (
                    device.type != "cuda"
                    or peak - baseline < free * config["runtime"]["gpu_memory_safety_fraction"]
                )
                row.update(
                    seconds_per_epoch=seconds,
                    peak_vram_bytes=peak,
                    parameters_per_seed=parameters,
                    model_updates_per_second=packed / seconds,
                    measured=True,
                    status="measured" if safe else "memory_safety_rejected",
                )
                print(
                    f"[calibration] {phase} {graph.name}/{condition} packed={packed} "
                    f"chunk={chunk} seconds/epoch={seconds:.4f} peak={peak} "
                    f"status={row['status']}",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as error:
                row.update(
                    status="OOM",
                    error=str(error),
                    seconds_per_epoch=None,
                    peak_vram_bytes=None,
                    model_updates_per_second=None,
                )
                print(
                    f"[calibration OOM] {graph.name}/{condition} packed={packed} "
                    f"chunk={chunk}; trying exact alternative chunks",
                    flush=True,
                )
            trials.append(row)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    feasible = [row for row in trials if row["status"] == "measured"]
    if not feasible:
        raise RuntimeError(
            f"no measured safe packing/chunk for {graph.name}/{condition}; "
            "graph/model scope preserved; inspect calibration rows"
        )
    chosen = max(feasible, key=lambda row: (row["model_updates_per_second"], row["packed_runs"]))
    return {
        "packed_runs": chosen["packed_runs"],
        "path_chunk": chosen["path_chunk"],
        "seconds_per_epoch": chosen["seconds_per_epoch"],
        "trials": trials,
        "temporary_calibration_updates_not_training_budget": sum(
            row["packed_runs"]
            * (config["runtime"]["calibration_warmups"] + config["runtime"]["calibration_repeats"])
            for row in trials
            if row["measured"]
        ),
    }


def _resume_payload(folder, metadata):
    if folder is None or not Path(folder).is_dir():
        return None, None
    folder = Path(folder)
    selected = folder / "selected.pt"
    snapshots = ([selected] if selected.is_file() else []) + sorted(
        folder.glob("resume_epoch_*.pt"),
        reverse=True,
    )
    for path in snapshots:
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            print(
                f"[resume] incomplete snapshot without hash manifest preserved: {path}", flush=True
            )
            continue
        saved = read_json(sidecar)
        if file_sha256(path) != saved["sha256"]:
            raise ValueError(f"resume checkpoint content hash mismatch: {path}")
        if saved["metadata"] != metadata:
            raise ValueError(f"resume graph/config/source/packing identity mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["metadata"] != metadata or payload["epoch"] != saved["epoch"]:
            raise ValueError(f"resume checkpoint payload identity mismatch: {path}")
        return payload, path
    return None, None


def _save_snapshot(path, payload):
    save_checkpoint(path, cpu_state(payload))
    write_json(
        path.with_suffix(".json"),
        {
            "sha256": file_sha256(path),
            "metadata": payload["metadata"],
            "epoch": payload["epoch"],
        },
    )


def _hardware_sample(device, resources, previous=None):
    process = psutil.Process()
    cpu_time = sum(process.cpu_times()[:2])
    now = time.monotonic()
    row = {
        "process_rss_bytes": process.memory_info().rss,
        "cpu_percent": None,
        "cpu_time_seconds": cpu_time,
        "monotonic_seconds": now,
        "ram_available_bytes": psutil.virtual_memory().available,
        "gpu_utilization_percent": None,
    }
    if previous is not None:
        row["cpu_percent"] = (
            100 * (cpu_time - previous["cpu_time_seconds"]) / (now - previous["monotonic_seconds"])
        )
    uuid = (resources.get("gpu") or {}).get("uuid")
    if device.type == "cuda" and uuid and uuid != "unavailable":
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={uuid if uuid.startswith(('GPU-', 'MIG-')) else 'GPU-' + uuid}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            ).stdout.strip()
            row["gpu_utilization_percent"] = float(output)
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
        ) as error:
            row["gpu_utilization_unavailable_reason"] = str(error)
    else:
        row["gpu_utilization_unavailable_reason"] = "CPU or runtime UUID unavailable"
    return row


def train_pack(
    graph,
    condition,
    seeds,
    lr,
    config,
    output_dir,
    source,
    graph_digest,
    phase,
    resume_dir=None,
    calibration=None,
    resources=None,
):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    if calibration is None:
        raise ValueError("measured packing/chunk calibration is required")
    with (folder / "terminal.log").open("x", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
            return _train_pack(
                graph,
                condition,
                list(seeds),
                lr,
                config,
                folder,
                source,
                graph_digest,
                phase,
                resume_dir,
                calibration,
                resources or {},
            )


def _train_pack(
    graph,
    condition,
    seeds,
    lr,
    config,
    folder,
    source,
    graph_digest,
    phase,
    resume_dir,
    calibration,
    resources,
):
    device = graph.x.device
    model = make_model(graph, condition, seeds, config, calibration["path_chunk"])
    optimizer = make_optimizer(model, config, lr)
    metadata = {
        "dataset": graph.name,
        "condition": condition,
        "placement": model.placement,
        "injection_layers": list(model.injection_layers),
        "seeds": seeds,
        "lr": lr,
        "phase": phase,
        "config_digest": digest(config),
        "code_digest": source["code_digest"],
        "graph_digest": graph_digest,
        "packed_runs": len(seeds),
        "path_chunk": calibration["path_chunk"],
    }
    payload, restored = _resume_payload(resume_dir, metadata)
    epochs = config["training"]["epochs_per_run"]
    history, hardware = [], []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_ce, best_acc = np.full(len(seeds), np.inf), np.full(len(seeds), -np.inf)
    best_epoch = np.zeros(len(seeds), dtype=int)
    start_epoch = 0
    if payload is not None:
        model.load_state_dict(payload["current_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        best_state = {key: value.to(device) for key, value in payload["best_state"].items()}
        best_ce, best_acc = np.array(payload["best_ce"]), np.array(payload["best_acc"])
        best_epoch = np.array(payload["best_epoch"])
        history, hardware = payload["history"], payload["hardware"]
        start_epoch = int(payload["epoch"])
        if not 0 <= start_epoch <= epochs:
            raise ValueError("resume epoch outside exact training budget")
        print(
            f"[resume] {phase} {graph.name}/{condition} seeds={seeds} "
            f"epoch={start_epoch}/{epochs} source={restored}",
            flush=True,
        )
    print(
        f"[train start] phase={phase} dataset={graph.name} condition={condition} "
        f"placement={model.placement} injection_layers={model.injection_layers} "
        f"seeds={seeds} lr={lr} layers=2 hidden={model.hidden} "
        f"parameters/seed={model.parameters_per_seed} "
        f"trainable/seed={model.trainable_parameters_per_seed} "
        f"N={graph.num_nodes} E={graph.edges.shape[1]} "
        f"localE={graph.topology.num_local_edges} pairs={graph.topology.num_pairs} "
        f"input={tuple(graph.x.shape)} full_graph_batch=1 effective_graph_batch=1 "
        f"packed_runs={len(seeds)} epochs={epochs} debug={config['profile'] == 'debug'}",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    durations = []
    execution_hardware = []
    with (folder / "epoch_history.csv").open("x", encoding="utf-8", newline="") as stream:
        import csv

        writer = None
        if history:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        for epoch in range(start_epoch, epochs):
            start = time.perf_counter()
            values = _epoch(model, graph, optimizer, epoch)
            seconds = time.perf_counter() - start
            durations.append(seconds)
            better_indices = []
            for index, row in enumerate(values):
                candidate_ce, candidate_acc = row["validation_ce"], row["validation_accuracy"]
                better = candidate_ce < best_ce[index] or (
                    candidate_ce == best_ce[index] and candidate_acc > best_acc[index]
                )
                if better:
                    best_ce[index], best_acc[index], best_epoch[index] = (
                        candidate_ce,
                        candidate_acc,
                        epoch + 1,
                    )
                    better_indices.append(index)
                row.update(
                    epoch=epoch + 1,
                    seed=seeds[index],
                    dataset=graph.name,
                    condition=condition,
                    placement=model.placement,
                    phase=phase,
                    lr=lr,
                    seconds_per_epoch=seconds,
                    packed_runs=len(seeds),
                    best_epoch=int(best_epoch[index]),
                )
            if better_indices:
                mask = torch.zeros(len(seeds), dtype=torch.bool, device=device)
                mask[better_indices] = True
                for name, value in model.state_dict().items():
                    best_state[name] = torch.where(
                        mask.reshape(len(seeds), *([1] * (value.ndim - 1))),
                        value.detach(),
                        best_state[name],
                    )
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
            writer.writerows(values)
            stream.flush()
            history.extend(values)
            avg = float(np.mean(durations[-10:]))
            peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else "N/A"
            print(
                f"[epoch] {phase} {graph.name}/{condition} lr={lr:g} seeds={seeds} "
                f"{epoch + 1}/{epochs} trainCE={np.mean([v['train_ce'] for v in values]):.5f} "
                f"valCE={np.mean([v['validation_ce'] for v in values]):.5f} "
                f"valAcc={np.mean([v['validation_accuracy'] for v in values]):.4f} "
                f"seconds={seconds:.3f} current_pack_ETA={(epochs - epoch - 1) * avg:.0f}s "
                f"peakVRAM={peak_vram}",
                flush=True,
            )
            if (epoch + 1) % config["runtime"]["resume_every_epochs"] == 0 or epoch + 1 == epochs:
                sample = _hardware_sample(
                    device,
                    resources,
                    execution_hardware[-1] if execution_hardware else None,
                )
                sample.update(
                    epoch=epoch + 1,
                    dataset=graph.name,
                    condition=condition,
                    placement=model.placement,
                    phase=phase,
                )
                hardware.append(sample)
                execution_hardware.append(sample)
                snapshot = {
                    "metadata": metadata,
                    "epoch": epoch + 1,
                    "current_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_state": best_state,
                    "best_ce": best_ce.tolist(),
                    "best_acc": best_acc.tolist(),
                    "best_epoch": best_epoch.tolist(),
                    "history": history,
                    "hardware": hardware,
                }
                _save_snapshot(folder / f"resume_epoch_{epoch + 1:04d}.pt", snapshot)
    if np.any(best_epoch == 0) or len(history) != epochs * len(seeds):
        raise RuntimeError("incomplete epoch/seed coverage or no validation-selected checkpoint")
    last_state = cpu_state(model.state_dict())
    model.load_state_dict(best_state)
    rows = [
        {
            "dataset": graph.name,
            "condition": condition,
            "placement": model.placement,
            "phase": phase,
            "lr": lr,
            "seed": seed,
            "best_epoch": int(best_epoch[index]),
            "validation_ce": float(best_ce[index]),
            "validation_accuracy": float(best_acc[index]),
            "training_epochs": epochs,
            "optimizer_updates": epochs,
            "new_optimizer_updates": epochs - start_epoch,
        }
        for index, seed in enumerate(seeds)
    ]
    selected = {
        "metadata": metadata,
        "epoch": epochs,
        "current_state": last_state,
        "optimizer_state": optimizer.state_dict(),
        "best_state": best_state,
        "best_ce": best_ce.tolist(),
        "best_acc": best_acc.tolist(),
        "best_epoch": best_epoch.tolist(),
        "history": history,
        "hardware": hardware,
    }
    _save_snapshot(folder / "selected.pt", selected)
    write_json(folder / "selected_validation.json", {"rows": rows})
    write_json(folder / "hardware_samples.json", hardware)
    resource_row = {
        "dataset": graph.name,
        "condition": condition,
        "placement": model.placement,
        "phase": phase,
        "scope": "training",
        "device": str(device),
        "packed_runs": len(seeds),
        "path_chunk": calibration["path_chunk"],
        "seconds_per_epoch": float(np.mean(durations)) if durations else None,
        "seconds_this_execution": time.perf_counter() - begin,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "parameters_per_seed": model.parameters_per_seed,
        "status": "measured" if durations else "reused_complete",
        "measured": bool(durations),
        "resumed_from": str(restored) if restored else None,
    }
    write_json(folder / "training_resource.json", resource_row)
    write_json(
        folder / "completion.json",
        {
            "completed": True,
            "rows": len(rows),
            "epochs_per_seed": epochs,
            "new_updates": (epochs - start_epoch) * len(seeds),
            "selected_sha256": file_sha256(folder / "selected.pt"),
            "metadata": metadata,
        },
    )
    return {
        "model": model,
        "rows": rows,
        "resource": resource_row,
        "selected_path": folder / "selected.pt",
    }


def preserve_resume_calibration(source_path, output_path, config, phase, dataset, condition):
    """Preserve original packed/chunk grouping so saved states retain exact identity."""
    source_path, output_path = Path(source_path), Path(output_path)
    record = read_json(source_path)
    if (
        record["config_digest"] != digest(config)
        or record["phase"] != phase
        or record["dataset"] != dataset
        or record["condition"] != condition
        or record.get("placement") != parse_condition(condition)[2]
    ):
        raise ValueError("resume calibration identity mismatch")
    if output_path.exists():
        raise ValueError("calibration output already exists")
    shutil.copyfile(source_path, output_path)
    return record["selection"]
