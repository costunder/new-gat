"""Independent packed training of the adaptive edge metric classifier.

Every seed receives its own projections, active generators and Adam moments. The
loss is the sum of per-seed mean train CE, so packing does not divide a seed's
gradient. Calibration uses disposable fresh models; training starts from the
same named initialization and dropout stream, or an exact checked resume.
Only train and validation metrics enter this module. Test scores remain locked.
"""

from __future__ import annotations

import contextlib
import gc
import math
import shutil
import subprocess
import sys
import time
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import psutil
import torch

from ...wedge_propagation.classification.evaluation import classification_metrics, split_indices
from ...wedge_propagation.study import Tee, synchronize
from .common import condition_metadata, cpu_state, digest, file_sha256, parse_condition, read_json, save_checkpoint, write_json
from .model import PackedClassifier

_DIAGNOSTIC_KEYS = (
    "projected_norm", "output_norm", "off_norm", "matched_delta_norm",
    "matched_delta_relative", "off_nonzero", "energy_intra", "energy_cross", "energy_total",
    "message_diag_norm", "message_cross_norm", "diagonal_mean", "diagonal_std", "pair_mean", "pair_std",
)


def _condition_fields(condition):
    return condition_metadata(condition)


def make_model(graph, condition, seeds, config, path_chunk):
    recipe, family, cross, variant = parse_condition(condition)
    if config["profile"] == "full" and graph.x.dtype != torch.float32:
        raise ValueError("FULL requires contracted float32 precision")
    if config["training"]["tf32"]:
        raise ValueError("TF32 is disabled by contract")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = PackedClassifier(condition, graph.num_features, graph.num_classes, seeds,
        hidden=config["backbone"]["hidden_dim"], dropout=config["backbone"]["dropout_probability"],
        path_chunk=path_chunk, checkpoint_paths=config["runtime"]["activation_checkpointing"],
        dataset_name=graph.name).to(device=graph.x.device,dtype=graph.x.dtype)
    gates = 2*(385*int(variant in ("D1","F2","DA"))+641*int(variant in ("F1","F2","DA")))
    expected = graph.num_features*model.hidden+model.hidden*graph.num_classes+gates+6*int(variant=="P2")
    if model.parameters_per_seed != expected or model.trainable_parameters_per_seed != expected:
        raise ValueError(f"Active parameter count differs: {condition} expected={expected} actual={model.parameters_per_seed}")
    return model


def make_optimizer(model, config, lr):
    if isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    train = config["training"]
    return torch.optim.Adam(
        model.weight_decay_groups(train["weight_decay"]), lr=lr,
        betas=tuple(train["optimizer_betas"]), eps=train["optimizer_epsilon"], foreach=False,
    )


def _seed_rows(value, seeds):
    if value.shape[0] != seeds:
        raise ValueError("all classifier parameters must retain an independent leading seed axis")
    return value.reshape(seeds, -1)


def _epoch(model, graph, optimizer, epoch, telemetry=True, diagnostics=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(graph, epoch=epoch)
    indices = split_indices(graph)
    ce, acc = classification_metrics(logits, graph.y, graph.train_mask, validate=False,indices=indices["train"])
    ce.sum().backward()
    named = list(model.named_parameters())
    count = len(model.seeds)
    for name,p in named:
        if p.grad is None:
            raise RuntimeError(f"Parameter disconnected from CE: {name}")
        _seed_rows(p,count)
    finite_grad=torch.stack([torch.isfinite(_seed_rows(p.grad,count)).all(1) for _,p in named]).all(0)
    before={name:p.detach().clone() for name,p in named} if telemetry else {}
    groups={"all":[(n,p) for n,p in named],"projection":[(n,p) for n,p in named if n.startswith("weights.")],
            "diagonal":[(n,p) for n,p in named if "diagonal_gate" in n],
            "pair":[(n,p) for n,p in named if "pair_gate" in n],
            "polynomial":[(n,p) for n,p in named if n.startswith("polynomial.")]}
    gradients={key:torch.stack([_seed_rows(p.grad,count).square().sum(1) for _,p in members]).sum(0).sqrt()
               for key,members in groups.items() if members} if telemetry else {}
    nonzero={key:torch.stack([(_seed_rows(p.grad,count)!=0).sum(1) for _,p in members]).sum(0)
             for key,members in groups.items() if members} if telemetry else {}
    optimizer.step()
    finite_parameter=torch.stack([torch.isfinite(_seed_rows(p,count)).all(1) for _,p in named]).all(0)
    model.eval()
    with torch.no_grad():
        validation,details=model(graph,diagnostics=diagnostics)
        val_ce,val_acc=classification_metrics(validation,graph.y,graph.val_mask,validate=False,indices=indices["validation"])
    columns={"train_ce":ce.detach(),"train_accuracy":acc.detach(),"validation_ce":val_ce,"validation_accuracy":val_acc,
             "finite_logits":torch.isfinite(logits).flatten(1).all(1).to(ce.dtype),
             "finite_validation_logits":torch.isfinite(validation).flatten(1).all(1).to(ce.dtype),
             "finite_ce_gradients":finite_grad.to(ce.dtype),"finite_parameters":finite_parameter.to(ce.dtype)}
    if telemetry:
        for key,members in groups.items():
            if not members: continue
            columns[key+"_ce_gradient_norm"]=gradients[key]
            columns[key+"_gradient_nonzero_count"]=nonzero[key].to(ce.dtype)
            columns[key+"_parameter_update_norm"]=torch.stack([_seed_rows(p.detach()-before[n],count).square().sum(1) for n,p in members]).sum(0).sqrt()
    for layer,detail in enumerate(details):
        for key in _DIAGNOSTIC_KEYS:
            value=detail.get(key)
            if value is not None:
                if value.shape != ce.shape: raise ValueError(f"Diagnostic seed axis differs: {key}")
                columns[f"layer_{layer}_{key}"]=value.detach().to(ce.dtype)
    values=torch.stack(list(columns.values()),1).detach().cpu().numpy()
    finite_keys=[k for k in columns if k.startswith("finite_")]
    if not np.isfinite(values).all() or any(not np.all(values[:,list(columns).index(k)]==1) for k in finite_keys):
        raise FloatingPointError("Nonfinite CE/logit/gradient/update; preserve failure artifacts")
    rows=[{k:float(v) for k,v in zip(columns,row,strict=True)} for row in values]
    for row in rows:
        row.update(_condition_fields(model.condition),diagnostics_recorded=bool(diagnostics))
        for key in groups:
            for suffix in ("ce_gradient_norm","gradient_nonzero_count","parameter_update_norm"):
                row.setdefault(key+"_"+suffix,None)
        for layer in range(2):
            for key in _DIAGNOSTIC_KEYS: row.setdefault(f"layer_{layer}_{key}",None)
            if diagnostics and row[f"layer_{layer}_off_nonzero"]==0:
                row[f"layer_{layer}_matched_delta_relative"]=None
    return rows


def benchmark_trial(graph, condition, seeds, lr, config, path_chunk):
    """Disposable updates plus a validation diagnostic memory probe; no run reuse."""
    model = make_model(graph, condition, seeds, config, path_chunk)
    optimizer = make_optimizer(model, config, lr)
    runtime = config["runtime"]
    try:
        for epoch in range(runtime["calibration_warmups"]):
            _epoch(model, graph, optimizer, epoch, telemetry=False)
        synchronize(graph.x.device)
        if graph.x.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(graph.x.device)
        start = time.perf_counter()
        for epoch in range(runtime["calibration_repeats"]):
            _epoch(model, graph, optimizer, epoch, telemetry=True,diagnostics=True)
        synchronize(graph.x.device)
        seconds = (time.perf_counter()-start) / runtime["calibration_repeats"]
        # First/final and scheduled epochs request this exact post-update path.
        # It contributes to the memory measurement, without another optimizer update.
        model.eval()
        with torch.no_grad():
            probe, details = model(graph, diagnostics=True)
            if len(details) != 2:
                raise ValueError("diagnostic memory probe requires both classifier layers")
            if not bool(torch.isfinite(probe).all()):
                raise FloatingPointError("nonfinite diagnostic calibration logits")
        synchronize(graph.x.device)
        peak = torch.cuda.max_memory_allocated(graph.x.device) if graph.x.device.type == "cuda" else None
        return seconds, peak, model.parameters_per_seed
    finally:
        del model, optimizer


def choose_packing(graph, condition, seeds, lr, config, phase):
    mode, intra, cross, variant = parse_condition(condition)
    candidates = sorted({min(value, len(seeds)) for value in config["resources"][f"parallel_{phase}_run_candidates"]})
    if not candidates or min(candidates) < 1:
        raise ValueError("calibration requires a nonempty independent seed group")
    geometry = graph.geometry_for(mode)
    paths = max(graph.topology.num_local_edges, geometry.num_cross_edges)
    chunks = sorted({max(1, min(paths, value)) for value in [*config["runtime"]["relation_chunk_candidates"], max(1, paths)]})
    trials, device = [], graph.x.device
    for packed in candidates:
        for chunk in chunks:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(device)
                baseline = torch.cuda.memory_allocated(device)
            else:
                free = baseline = None
            row = {
                "dataset": graph.name, "condition": condition, **_condition_fields(condition), "phase": phase, "scope": "calibration", "device": str(device),
                "packed_runs": packed, "path_chunk": chunk, "all_intra_edges": graph.topology.num_local_edges,
                "all_cross_edges": geometry.num_cross_edges, "measured": False,
                "includes_diagnostic_memory_probe": True,
                "calibration_state": "disposable_fresh_model_and_Adam; identical_named_initialization_restored_for_training",
            }
            try:
                seconds, peak, parameters = benchmark_trial(graph, condition, seeds[:packed], lr, config, chunk)
                safe = device.type != "cuda" or peak-baseline < free*config["runtime"]["gpu_memory_safety_fraction"]
                row.update(
                    seconds_per_epoch=seconds, peak_vram_bytes=peak, parameters_per_seed=parameters,
                    model_updates_per_second=packed/seconds, measured=True,
                    status="measured" if safe else "memory_safety_rejected",
                )
                print(f"[calibration] {phase} {graph.name}/{condition} packed={packed} chunk={chunk} seconds/epoch={seconds:.4f} peak={peak} status={row['status']}", flush=True)
            except torch.cuda.OutOfMemoryError as error:
                row.update(status="OOM", error=str(error), seconds_per_epoch=None, peak_vram_bytes=None, model_updates_per_second=None)
                print(f"[calibration OOM] {graph.name}/{condition} packed={packed} chunk={chunk}; exact alternative allocation", flush=True)
            trials.append(row)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    feasible = [row for row in trials if row["status"] == "measured"]
    if not feasible:
        raise RuntimeError(f"no measured safe packing/chunk for {graph.name}/{condition}; graph/model/seed scope preserved")
    chosen = max(feasible, key=lambda row: (row["model_updates_per_second"], row["packed_runs"]))
    return {
        "packed_runs": chosen["packed_runs"], "path_chunk": chosen["path_chunk"],
        "seconds_per_epoch": chosen["seconds_per_epoch"], "trials": trials,
        "temporary_calibration_updates_not_training_budget": sum(
            row["packed_runs"]*(config["runtime"]["calibration_warmups"]+config["runtime"]["calibration_repeats"])
            for row in trials if row["measured"]
        ),
    }


def _resume_payload(folder, metadata):
    if folder is None or not Path(folder).is_dir():
        return None, None
    folder = Path(folder)
    selected = folder/"selected.pt"
    snapshots = ([selected] if selected.is_file() else []) + sorted(folder.glob("resume_epoch_*.pt"), reverse=True)
    for path in snapshots:
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            print(f"[resume] incomplete snapshot without hash manifest preserved: {path}", flush=True)
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
    write_json(path.with_suffix(".json"), {"sha256": file_sha256(path), "metadata": payload["metadata"], "epoch": payload["epoch"]})


def _hardware_sample(device, resources, previous=None):
    process = psutil.Process()
    cpu_time, now = sum(process.cpu_times()[:2]), time.monotonic()
    row = {
        "process_rss_bytes": process.memory_info().rss, "cpu_percent": None,
        "cpu_time_seconds": cpu_time, "monotonic_seconds": now,
        "ram_available_bytes": psutil.virtual_memory().available, "gpu_utilization_percent": None,
    }
    if previous is not None:
        row["cpu_percent"] = 100*(cpu_time-previous["cpu_time_seconds"])/(now-previous["monotonic_seconds"])
    uuid = (resources.get("gpu") or {}).get("uuid")
    if device.type == "cuda" and uuid and uuid != "unavailable":
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={uuid if uuid.startswith(('GPU-', 'MIG-')) else 'GPU-'+uuid}", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            row["gpu_utilization_percent"] = float(result.stdout.strip())
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
            row["gpu_utilization_unavailable_reason"] = str(error)
    else:
        row["gpu_utilization_unavailable_reason"] = "CPU or runtime UUID unavailable"
    return row


def train_pack(
    graph, condition, seeds, lr, config, output_dir, source, graph_digest, phase,
    resume_dir=None, calibration=None, resources=None,
):
    if calibration is None:
        raise ValueError("measured seed packing/exact edge chunk calibration is required")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder/"terminal.log").open("x", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
            return _train_pack(graph, condition, list(seeds), lr, config, folder, source, graph_digest, phase, resume_dir, calibration, resources or {})


def _validate_resume_state(payload, seeds, epochs):
    epoch = payload["epoch"]
    if type(epoch) is not int or not 0 <= epoch <= epochs:
        raise ValueError("resume epoch outside exact training budget")
    for key in ("best_ce", "best_acc", "best_epoch"):
        if len(payload[key]) != len(seeds):
            raise ValueError("resume best-state seed coverage differs")
    if len(payload["history"]) != epoch*len(seeds):
        raise ValueError("resume history does not cover every finished epoch/seed")
    keys = {(row["epoch"], row["seed"]) for row in payload["history"]}
    if keys != {(e, seed) for e in range(1, epoch+1) for seed in seeds}:
        raise ValueError("resume history has duplicate/missing epoch/seed identities")
    if any(type(e) is not int or not 0 <= e <= epoch for e in payload["best_epoch"]):
        raise ValueError("resume best epoch outside finished optimizer updates")
    if epoch > 0 and (not np.isfinite(payload["best_ce"]).all() or not np.isfinite(payload["best_acc"]).all() or any(e == 0 for e in payload["best_epoch"])):
        raise ValueError("resume validation selection is incomplete or nonfinite")


def _train_pack(graph, condition, seeds, lr, config, folder, source, graph_digest, phase, resume_dir, calibration, resources):
    import csv

    device = graph.x.device
    mode, intra, cross, variant = parse_condition(condition)
    model = make_model(graph, condition, seeds, config, calibration["path_chunk"])
    optimizer = make_optimizer(model, config, lr)
    metadata = {
        "experiment": config["experiment"], "dataset": graph.name, "condition": condition,
        **_condition_fields(condition), "seeds": seeds, "lr": lr, "phase": phase,
        "config_digest": digest(config), "code_digest": source["code_digest"], "graph_digest": graph_digest,
        "packed_runs": len(seeds), "path_chunk": calibration["path_chunk"],
    }
    payload, restored = _resume_payload(resume_dir, metadata)
    epochs = config["training"]["epochs_per_run"]
    diagnostic_every = config["training"]["diagnostic_every_epochs"]
    if isinstance(diagnostic_every, bool) or not isinstance(diagnostic_every, Integral) or diagnostic_every < 1:
        raise ValueError("diagnostic_every_epochs must be a positive integer")
    history, hardware = [], []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_ce, best_acc = np.full(len(seeds), np.inf), np.full(len(seeds), -np.inf)
    best_epoch, start_epoch = np.zeros(len(seeds), dtype=int), 0
    if payload is not None:
        _validate_resume_state(payload, seeds, epochs)
        model.load_state_dict(payload["current_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        best_state = {key: value.to(device) for key, value in payload["best_state"].items()}
        best_ce, best_acc, best_epoch = (np.array(payload[key]) for key in ("best_ce", "best_acc", "best_epoch"))
        history, hardware, start_epoch = payload["history"], payload["hardware"], payload["epoch"]
        print(f"[resume] {phase} {graph.name}/{condition} seeds={seeds} epoch={start_epoch}/{epochs} source={restored}", flush=True)
    print(
        f"[train start] phase={phase} dataset={graph.name} condition={condition} mode={model.mode} variant={model.variant} "
        f"seeds={seeds} lr={lr} layers=2 hidden={model.hidden} parameters/seed={model.parameters_per_seed} "
        f"trainable/seed={model.trainable_parameters_per_seed} N={graph.num_nodes} E={graph.edges.shape[1]} "
        f"family={intra} cross={cross} energy_operator={_condition_fields(condition)['energy_operator']} "
        f"localE={graph.topology.num_local_edges} crossE={graph.geometry_for(mode).num_cross_edges} "
        f"input={tuple(graph.x.shape)} full_graph_batch=1 effective_graph_batch=1 packed_runs={len(seeds)} "
        f"epochs={epochs} diagnostics_every={diagnostic_every} test_locked=True debug={config['profile']=='debug'}",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begin, durations, execution_hardware = time.perf_counter(), [], []
    with (folder/"epoch_history.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = None
        if history:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        for epoch in range(start_epoch, epochs):
            start = time.perf_counter()
            observe = epoch == 0 or (epoch+1) % diagnostic_every == 0 or epoch+1 == epochs
            values = _epoch(model, graph, optimizer, epoch, diagnostics=observe)
            seconds = time.perf_counter()-start
            durations.append(seconds)
            better_indices = []
            for index, row in enumerate(values):
                candidate_ce, candidate_acc = row["validation_ce"], row["validation_accuracy"]
                if candidate_ce < best_ce[index] or (candidate_ce == best_ce[index] and candidate_acc > best_acc[index]):
                    best_ce[index], best_acc[index], best_epoch[index] = candidate_ce, candidate_acc, epoch+1
                    better_indices.append(index)
                row.update(
                    epoch=epoch+1, seed=seeds[index], dataset=graph.name, condition=condition,
                    **_condition_fields(condition), phase=phase, lr=lr,
                    seconds_per_epoch=seconds, packed_runs=len(seeds), best_epoch=int(best_epoch[index]),
                )
            if better_indices:
                mask = torch.zeros(len(seeds), dtype=torch.bool, device=device)
                mask[better_indices] = True
                for name, value in model.state_dict().items():
                    _seed_rows(value, len(seeds))
                    best_state[name] = torch.where(mask.reshape(len(seeds), *([1]*(value.ndim-1))), value.detach(), best_state[name])
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
            writer.writerows(values)
            stream.flush()
            history.extend(values)
            average = float(np.mean(durations[-10:]))
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else "N/A"
            print(
                f"[epoch] {phase} {graph.name}/{condition} lr={lr:g} seeds={seeds} {epoch+1}/{epochs} "
                f"trainCE={np.mean([v['train_ce'] for v in values]):.5f} valCE={np.mean([v['validation_ce'] for v in values]):.5f} "
                f"valAcc={np.mean([v['validation_accuracy'] for v in values]):.4f} allGrad={np.mean([v['all_ce_gradient_norm'] for v in values]):.5f} "
                f"seconds={seconds:.3f} current_pack_ETA={(epochs-epoch-1)*average:.0f}s peakVRAM={peak}", flush=True,
            )
            if (epoch+1) % config["runtime"]["resume_every_epochs"] == 0 or epoch+1 == epochs:
                sample = _hardware_sample(device, resources, execution_hardware[-1] if execution_hardware else None)
                sample.update(epoch=epoch+1, dataset=graph.name, condition=condition, **_condition_fields(condition), phase=phase)
                hardware.append(sample)
                execution_hardware.append(sample)
                snapshot = {
                    "metadata": metadata, "epoch": epoch+1, "current_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_state": best_state,
                    "best_ce": best_ce.tolist(), "best_acc": best_acc.tolist(), "best_epoch": best_epoch.tolist(),
                    "history": history, "hardware": hardware,
                }
                _save_snapshot(folder/f"resume_epoch_{epoch+1:04d}.pt", snapshot)
    if np.any(best_epoch == 0) or len(history) != epochs*len(seeds):
        raise RuntimeError("incomplete full epoch/seed coverage or no validation-selected checkpoint")
    last_state = cpu_state(model.state_dict())
    model.load_state_dict(best_state)
    rows = [{
        "dataset": graph.name, "condition": condition, **_condition_fields(condition),
        "phase": phase, "lr": lr, "seed": seed, "best_epoch": int(best_epoch[index]),
        "validation_ce": float(best_ce[index]), "validation_accuracy": float(best_acc[index]),
        "training_epochs": epochs, "optimizer_updates": epochs, "new_optimizer_updates": epochs-start_epoch,
    } for index, seed in enumerate(seeds)]
    selected = {
        "metadata": metadata, "epoch": epochs, "current_state": last_state,
        "optimizer_state": optimizer.state_dict(), "best_state": best_state,
        "best_ce": best_ce.tolist(), "best_acc": best_acc.tolist(), "best_epoch": best_epoch.tolist(),
        "history": history, "hardware": hardware,
    }
    _save_snapshot(folder/"selected.pt", selected)
    write_json(folder/"selected_validation.json", {"rows": rows})
    write_json(folder/"hardware_samples.json", hardware)
    resource = {
        "dataset": graph.name, "condition": condition, **_condition_fields(condition),
        "phase": phase, "scope": "training", "device": str(device), "packed_runs": len(seeds),
        "path_chunk": calibration["path_chunk"], "seconds_per_epoch": float(np.mean(durations)) if durations else None,
        "seconds_this_execution": time.perf_counter()-begin,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "parameters_per_seed": model.parameters_per_seed, "status": "measured" if durations else "reused_complete",
        "measured": bool(durations), "resumed_from": str(restored) if restored else None,
    }
    write_json(folder/"training_resource.json", resource)
    write_json(folder/"completion.json", {
        "completed": True, "rows": len(rows), "epochs_per_seed": epochs,
        "new_updates": (epochs-start_epoch)*len(seeds), "selected_sha256": file_sha256(folder/"selected.pt"),
        "metadata": metadata, "test_evaluated": False,
    })
    return {"model": model, "rows": rows, "resource": resource, "selected_path": folder/"selected.pt"}


def preserve_resume_calibration(source_path, output_path, config, phase, dataset, condition):
    """Copy checked allocation choices; resumed Adam states keep their seed packing."""
    source_path, output_path = Path(source_path), Path(output_path)
    record = read_json(source_path)
    fields = _condition_fields(condition)
    if (
        record["config_digest"] != digest(config) or record["phase"] != phase
        or record["dataset"] != dataset or record["condition"] != condition
        or any(record.get(key) != value for key, value in fields.items())
    ):
        raise ValueError("resume calibration identity mismatch")
    if output_path.exists():
        raise FileExistsError("calibration output already exists; preserved")
    with output_path.open("xb") as destination, source_path.open("rb") as original:
        shutil.copyfileobj(original, destination)
    return record["selection"]


