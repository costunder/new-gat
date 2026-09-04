#!/usr/bin/env python3
"""Read-only CUDA checkpoint diagnostics; no training, downloads, or new test queries."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional NEW directory; extended audits auto-save, basic diagnostics use stdout",
    )
    parser.add_argument("--edge-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--ablate-graph",
        action="store_true",
        help="Validation-only identity-convolution ablation; no retraining",
    )
    parser.add_argument(
        "--full-audit",
        action="store_true",
        help="single-checkpoint C interventions plus a train-label gate/gradient audit",
    )
    parser.add_argument(
        "--interventions",
        action="store_true",
        help="validation-only learned/mean/shuffled/off C interventions",
    )
    parser.add_argument(
        "--gate-audit",
        action="store_true",
        help="optimizer-free gate input/parameter/task-gradient audit on train labels",
    )
    parser.add_argument(
        "--layerwise-interventions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also intervene on one conductance layer at a time (default: enabled)",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--gradient-mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--gradient-batches", type=int, default=1)
    parser.add_argument("--gradient-sample-limit", type=int, default=4096)
    parser.add_argument("--near-zero-threshold", type=float, default=1e-8)
    return parser


def _extended_requested(args: argparse.Namespace) -> bool:
    return bool(args.full_audit or args.interventions or args.gate_audit)


def _automatic_output(args: argparse.Namespace) -> Path | None:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    if not _extended_requested(args):
        return None
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        ROOT
        / "runs"
        / "diagnostics"
        / f"conductance-{args.run_id}-model-seed-{args.model_seed}-{timestamp}"
    ).resolve()


def resolve_run(args: argparse.Namespace) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id) or args.model_seed < 0:
        raise ValueError("Invalid run ID or model seed")
    base = (
        (args.results_root / "conductance_gat")
        if args.results_root is not None
        else ROOT / "research/conductance_gat/results/paper"
    )
    return (
        (base / args.run_id / f"model-seed-{args.model_seed}" / "benchmark").expanduser().resolve()
    )


def validate_run(manifest: dict, metrics: dict, model_seed: int, datasets: list[str]) -> dict:
    for record in (manifest, metrics):
        if any(
            record.get(key) != value
            for key, value in {
                "schema_version": 2,
                "track": "conductance_gat",
                "suite": "benchmark",
                "status": "passed",
            }.items()
        ):
            raise ValueError("Require a completed/passed conductance benchmark run")
    config = manifest["config"]
    if config.get("model_seed") != model_seed or metrics.get("model_seed") != model_seed:
        raise ValueError("Run model seed mismatch")
    expected = [f"{name}/conductance" for name in config["datasets"]]
    if (
        not expected
        or len(expected) != len(set(expected))
        or manifest.get("expected") != expected
        or manifest.get("completed") != expected
    ):
        raise ValueError("Run expected/completed datasets disagree")
    if (
        not datasets
        or len(set(datasets)) != len(datasets)
        or not set(datasets).issubset(config["datasets"])
    ):
        raise ValueError("Selected datasets are missing or duplicated")
    for dataset in config["datasets"]:
        if "conductance" not in metrics["datasets"][dataset]["models"]:
            raise ValueError(f"Missing completed model metrics: {dataset}")
        saved = metrics["datasets"][dataset]["models"]["conductance"]
        if any(
            not math.isfinite(saved[key]) or not 0 <= saved[key] <= 1
            for key in ("validation", "test")
        ):
            raise ValueError(f"Invalid saved validation/test metric: {dataset}")
    return config


def summarize_history(history: list[dict], saved_metrics: dict) -> dict:
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Training history must contain contiguous epochs")
    for row in history:
        if (
            not all(math.isfinite(row[key]) for key in ("train_loss", "validation"))
            or row["train_loss"] < 0
            or not 0 <= row["validation"] <= 1
        ):
            raise ValueError("Invalid/nonfinite training history")
    best = max(history, key=lambda row: row["validation"])
    if (
        saved_metrics["best_epoch"] != best["epoch"]
        or saved_metrics["epochs_run"] != len(history)
        or not math.isclose(saved_metrics["validation"], best["validation"], abs_tol=1e-7)
    ):
        raise ValueError("Saved metrics disagree with validation-selected history checkpoint")
    return {
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "train_loss_first": history[0]["train_loss"],
        "train_loss_min": min(row["train_loss"] for row in history),
        "train_loss_last": history[-1]["train_loss"],
        "train_loss_at_selected_epoch": best["train_loss"],
        "validation_first": history[0]["validation"],
        "validation_best": best["validation"],
        "validation_last": history[-1]["validation"],
        "train_loss_note": (
            "Original train-mode loss; not directly comparable to eval-mode train loss"
        ),
    }


def validate_checkpoint(
    checkpoint: dict, dataset: str, config: dict, saved_metrics: dict, history: list[dict]
) -> dict:
    summarize_history(history, saved_metrics)
    if checkpoint.get("dataset") != dataset or checkpoint.get("model") != "conductance":
        raise ValueError("Checkpoint dataset/model mismatch")
    if checkpoint.get("best_epoch") != saved_metrics["best_epoch"] or not math.isclose(
        checkpoint.get("validation", float("nan")), saved_metrics["validation"], abs_tol=1e-7
    ):
        raise ValueError("Checkpoint selection metadata mismatch")
    architecture = checkpoint["architecture"]
    expected = {key: config[key] for key in ("hidden_channels", "layers", "dropout")}
    if architecture != expected or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Checkpoint architecture/state_dict mismatch")
    if (
        any(
            type(expected[key]) is not int or expected[key] < 1
            for key in ("hidden_channels", "layers")
        )
        or not 0 <= expected["dropout"] < 1
    ):
        raise ValueError("Invalid checkpoint architecture")
    return expected


def restore_model(checkpoint, payload, config, saved_metrics, history, device):
    from research.conductance_gat.benchmark import ConductanceNodeClassifier

    architecture = validate_checkpoint(
        checkpoint, payload["dataset"], config, saved_metrics, history
    )
    model = ConductanceNodeClassifier(
        payload["graphs"][0]["x"].shape[1], payload["classes"], **architecture
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def require_cuda(name: str):
    import torch

    device = torch.device(name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Checkpoint inference requires CUDA; no CPU fallback")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.cuda.get_device_properties(device)
    return device


def _distribution(values) -> dict:
    import torch

    values = values.double().cpu()
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite diagnostic distribution")
    if not values.numel():
        return {"count": 0, "mean": None, "quantiles": None}
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        "quantiles": dict(
            zip(
                ("min", "p10", "median", "p90", "p99", "max"),
                torch.quantile(
                    values, torch.tensor([0, 0.1, 0.5, 0.9, 0.99, 1], dtype=torch.float64)
                ).tolist(),
                strict=True,
            )
        ),
    }


def layer_diagnostics(module, inputs: tuple, output, edge_chunk_size: int = 16384) -> dict:
    import torch

    state, edges, node_graph = inputs[:3]
    if edge_chunk_size < 1 or torch.any(node_graph != 0):
        raise ValueError("Layer diagnostics require one graph and a positive chunk size")
    state, output = state.float(), output.float()
    if not torch.isfinite(state).all() or not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite layer input/output")
    degree = state.new_zeros(len(state))
    conductances = []
    with torch.inference_mode(), torch.autocast(device_type=state.device.type, enabled=False):
        for start in range(0, edges.shape[1], edge_chunk_size):
            tail, head = edges[:, start : start + edge_chunk_size]
            gradient = state[head] - state[tail]
            c = module.estimator(gradient, state.new_empty((len(tail), 0)))
            if not torch.isfinite(c).all() or torch.any(c <= 0):
                raise FloatingPointError("Nonfinite/nonpositive conductance")
            degree.index_add_(0, head, c)
            degree.index_add_(0, tail, c)
            conductances.append(c.cpu())
        if not torch.isfinite(degree).all() or not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite layer output/weighted degree")
        rho = 0.95 * degree / degree.max().clamp_min(1e-12)
        delta = output.double() - state.double()
        input_squared = float(state.double().square().sum())
        delta_squared = float(delta.square().sum())
        change = delta.norm(dim=1) / state.double().norm(dim=1).clamp_min(1e-12)
        c = (
            torch.cat(conductances).double()
            if conductances
            else torch.empty(0, dtype=torch.float64)
        )
    return {
        "_c": c,
        "_rho": rho.cpu(),
        "_degree": degree.cpu(),
        "_node_change": change.cpu(),
        "rho_mean": float(rho.mean()),
        "c_cv": float(c.std(unbiased=False) / c.mean()) if c.numel() else None,
        "c_count": c.numel(),
        "c_sum": float(c.sum()),
        "c_squared_sum": float(c.square().sum()),
        "input_squared_sum": input_squared,
        "delta_squared_sum": delta_squared,
        "global_update_ratio": math.sqrt(delta_squared) / max(math.sqrt(input_squared), 1e-12),
        "zero_input_nodes": int((state.norm(dim=1) == 0).sum()),
    }


def summarize_layers(records: list[dict]) -> dict:
    import torch

    if not records:
        raise ValueError("No layer records")
    pooled = {
        key: torch.cat([record[key] for record in records])
        for key in ("_c", "_rho", "_degree", "_node_change")
    }
    c = pooled["_c"]
    cvs = [record["c_cv"] for record in records if record["c_cv"] is not None]
    return {
        "graphs": len(records),
        "graph_macro": {
            "rho_mean": sum(r["rho_mean"] for r in records) / len(records),
            "update_ratio_mean": sum(r["global_update_ratio"] for r in records) / len(records),
            "c_cv_mean": sum(cvs) / len(cvs) if cvs else None,
        },
        "node_pooled": {
            "rho": _distribution(pooled["_rho"]),
            "rho_below": {
                str(t): float((pooled["_rho"] < t).double().mean()) for t in (0.01, 0.05, 0.1)
            },
            "weighted_degree": _distribution(pooled["_degree"]),
            "relative_conv_change": _distribution(pooled["_node_change"]),
            "zero_input_nodes": sum(r["zero_input_nodes"] for r in records),
        },
        "edge_pooled": {
            "conductance": _distribution(c),
            "c_cv": float(c.std(unbiased=False) / c.mean()) if c.numel() else None,
        },
        "global_update_ratio": math.sqrt(sum(r["delta_squared_sum"] for r in records))
        / max(math.sqrt(sum(r["input_squared_sum"] for r in records)), 1e-12),
    }


def prediction_statistics(logits, labels, multilabel: bool) -> dict:
    import torch
    from torch.nn import functional as F

    if not torch.isfinite(logits).all() or not labels.numel():
        raise FloatingPointError("Nonfinite logits or empty evaluation labels")
    if multilabel:
        prediction, truth = logits > 0, labels > 0
        loss_sum = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Nonfinite summed BCE loss")
        return {
            "count": labels.numel(),
            "nodes": len(labels),
            "loss_sum": float(loss_sum),
            "tp": int((prediction & truth).sum()),
            "predicted_positive": int(prediction.sum()),
            "true_positive_labels": int(truth.sum()),
        }
    loss_sum = F.cross_entropy(logits, labels, reduction="sum")
    if not torch.isfinite(loss_sum):
        raise FloatingPointError("Nonfinite summed CE loss")
    return {
        "count": labels.numel(),
        "nodes": len(labels),
        "loss_sum": float(loss_sum),
        "correct": int((logits.argmax(dim=-1) == labels).sum()),
    }


def merge_predictions(records: list[dict], multilabel: bool) -> dict:
    count = sum(row["count"] for row in records)
    if count <= 0:
        raise ValueError("Cannot evaluate an empty split")
    result = {
        "count": count,
        "nodes": sum(r["nodes"] for r in records),
        "loss": sum(row["loss_sum"] for row in records) / count,
        "metric_name": "micro_f1" if multilabel else "accuracy",
    }
    if multilabel:
        tp, predicted, truth = (
            sum(row[key] for row in records)
            for key in ("tp", "predicted_positive", "true_positive_labels")
        )
        result.update(
            metric=2 * tp / (predicted + truth) if predicted + truth else 0,
            predicted_positive_fraction=predicted / count,
            true_positive_fraction=truth / count,
        )
    else:
        result["metric"] = sum(row["correct"] for row in records) / count
    return result


def _forward(model, graph, records, chunk_size, ablate=False):
    hooks = []
    try:
        for index, operator in enumerate(model.operators):

            def hook(module, inputs, output, index=index):
                if ablate:
                    return inputs[0]
                records[index].append(layer_diagnostics(module, inputs, output, chunk_size))
                return None

            hooks.append(operator.register_forward_hook(hook))
        return model(graph)
    finally:
        for hook in hooks:
            hook.remove()


def evaluate_checkpoint(model, payload, device, chunk_size, *, ablate=False) -> dict:
    import torch

    results = {}
    splits = ("validation",) if ablate else ("train", "validation")
    shared_logits, shared_layers = None, None
    with torch.inference_mode():
        for split in splits:
            layers = [[] for _ in model.operators]
            statistics = []
            indices = payload["splits"][split] if payload["dataset"] == "ppi" else [0]
            for index in indices:
                raw = payload["graphs"][index]
                graph = SimpleNamespace(
                    **{key: raw[key].to(device) for key in ("x", "y", "incidence_edge_index")}
                )
                if shared_logits is None:
                    logits = _forward(model, graph, layers, chunk_size, ablate)
                else:
                    logits, layers = shared_logits, shared_layers
                labels = graph.y
                if payload["dataset"] != "ppi":
                    shared_logits, shared_layers = logits, layers
                    selected = payload["splits"][split].nonzero(as_tuple=False).flatten().to(device)
                    logits, labels = (
                        logits.index_select(0, selected),
                        labels.index_select(0, selected),
                    )
                statistics.append(
                    prediction_statistics(logits, labels, payload["dataset"] == "ppi")
                )
            results[split] = {
                "prediction": merge_predictions(statistics, payload["dataset"] == "ppi")
            }
            if not ablate:
                results[split]["layers"] = [summarize_layers(record) for record in layers]
    return results


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _quantile_text(distribution: dict) -> str:
    values = distribution["quantiles"]
    return (
        "empty"
        if values is None
        else " ".join(f"{key}={_number(value)}" for key, value in values.items())
    )


def _print_layer(index: int, layer: dict) -> None:
    nodes, edges, macro = layer["node_pooled"], layer["edge_pooled"], layer["graph_macro"]
    print(
        f"    layer {index}: graphs={layer['graphs']} nodes={nodes['rho']['count']} "
        f"edges={edges['conductance']['count']}"
    )
    print(
        f"      C(edge pooled): mean={_number(edges['conductance']['mean'])} "
        f"CV={_number(edges['c_cv'])}; {_quantile_text(edges['conductance'])}"
    )
    print("      weighted degree(node pooled):", _quantile_text(nodes["weighted_degree"]))
    print("      rho(node pooled):", _quantile_text(nodes["rho"]))
    print(
        "      fraction rho below:",
        " ".join(f"{key}={value:.2%}" for key, value in nodes["rho_below"].items()),
    )
    print(
        f"      Conv change: global ||out-in||/||in||={_number(layer['global_update_ratio'])}; "
        f"node ratios {_quantile_text(nodes['relative_conv_change'])}"
    )
    print(
        f"      graph-macro means: rho={_number(macro['rho_mean'])} "
        f"change={_number(macro['update_ratio_mean'])} C-CV={_number(macro['c_cv_mean'])}"
    )


def additional_audits(model, payload, device, args, config, item: dict) -> None:
    """Extend the current item incrementally so a failed audit keeps earlier evidence."""

    if args.full_audit or args.interventions:
        from scripts.conductance_interventions import evaluate_interventions

        item["stage"] = "validation_interventions"
        print("  Comparing C interventions on validation only...", flush=True)
        item["interventions"] = evaluate_interventions(
            model,
            payload,
            device,
            edge_chunk_size=args.edge_chunk_size,
            shuffle_seed=args.shuffle_seed,
            layerwise=args.layerwise_interventions,
            progress=lambda name: print(f"    validation intervention: {name}", flush=True),
        )
        reference = item["interventions"]["variants"][0]["prediction"]
        baseline = item["baseline"]["validation"]["prediction"]
        if any(abs(reference[key] - baseline[key]) > 1e-4 for key in ("metric", "loss")):
            raise RuntimeError("Intervention learned reference disagrees with baseline recheck")
        for variant in item["interventions"]["variants"]:
            prediction = variant["prediction"]
            delta = variant["delta_vs_learned"]
            print(
                f"    {variant['name']}: {prediction['metric_name']}={prediction['metric']:.6f} "
                f"delta={delta['metric']:+.6f} loss={prediction['loss']:.6f} "
                f"logit_change={_number(delta['logits_relative_l2'])} "
                f"prediction_flip={_number(delta['prediction_flip_fraction'])}",
                flush=True,
            )
        if args.ablate_graph:
            off = next(
                variant
                for variant in item["interventions"]["variants"]
                if variant["name"] == "graph_off_all"
            )
            item["identity_convolution_validation"] = {"prediction": off["prediction"]}
            item["identity_minus_original_validation"] = (
                off["prediction"]["metric"] - item["baseline"]["validation"]["prediction"]["metric"]
            )
    elif args.ablate_graph:
        item["identity_convolution_validation"] = evaluate_checkpoint(
            model, payload, device, args.edge_chunk_size, ablate=True
        )["validation"]
        item["identity_minus_original_validation"] = (
            item["identity_convolution_validation"]["prediction"]["metric"]
            - item["baseline"]["validation"]["prediction"]["metric"]
        )
    if args.full_audit or args.gate_audit:
        from scripts.conductance_gate_audit import audit_gate_gradients

        if "weight_decay" not in config:
            raise ValueError("Saved weight_decay is required for the gradient/decay audit")
        item["stage"] = "train_label_gradient_audit"
        print(
            f"  Gate audit: {args.gradient_mode} mode, autograd ON, train labels only; "
            "no optimizer step...",
            flush=True,
        )
        item["gate_audit"] = audit_gate_gradients(
            model,
            payload,
            device,
            weight_decay=config["weight_decay"],
            mode=args.gradient_mode,
            ppi_batches=args.gradient_batches,
            ppi_batch_size=config.get("batch_size", 2),
            rng_seed=args.model_seed,
            sample_limit=args.gradient_sample_limit,
            near_zero=args.near_zero_threshold,
        )
        audit = item["gate_audit"]
        print("    audited train loss:", json.dumps(audit["loss"]), flush=True)
        for layer in audit["layers"]:
            raw = layer["tensors"]["raw_logit"]["all_element_moments"] or {}
            raw_gradient = layer["tensors"]["raw_logit_gradient"]["all_element_moments"] or {}
            print(
                f"    layer {layer['layer']}: raw-logit mean={_number(raw.get('mean'))} "
                f"std={_number(raw.get('std_population'))} "
                f"raw-logit task-gradient norm={_number(raw_gradient.get('l2_norm'))}",
                flush=True,
            )
        for name, parameter in audit["parameters"].items():
            print(
                f"    {name}: norm={_number(parameter['parameter']['l2_norm'])} "
                f"task_grad={_number(parameter['task_gradient']['l2_norm'])} "
                f"decay={_number(parameter['weight_decay_term_norm'])} "
                f"ratio={_number(parameter['task_to_decay_norm_ratio'])} "
                f"cosine={_number(parameter['task_decay_cosine'])}",
                flush=True,
            )


def _diagnose_impl(args, run: Path, report: dict, device) -> None:
    import torch

    from research.conductance_gat.benchmark_data import load_dataset

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    # Python's JSON reader permits NaN/Infinity; reject them before copying source metadata.
    json.dumps(manifest, allow_nan=False)
    json.dumps(metrics, allow_nan=False)
    config = validate_run(manifest, metrics, args.model_seed, args.datasets)
    print(
        "Saved training configuration:",
        json.dumps(
            {
                key: config[key]
                for key in (
                    "hidden_channels",
                    "layers",
                    "dropout",
                    "lr",
                    "weight_decay",
                    "epochs",
                    "patience",
                    "batch_size",
                )
                if key in config
            }
        ),
    )
    data_root = args.data_root or Path(config.get("data_root", ROOT / "data/paper"))
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Recorded data root unavailable: {data_root}; supply --data-root explicitly"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    current_hashes = {
        name: hashlib.sha256((ROOT / "research/conductance_gat" / name).read_bytes()).hexdigest()
        for name in ("benchmark.py", "benchmark_data.py", "sparse.py")
    }
    report.update(
        config=config,
        data_root=str(data_root),
        software={
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        implementation_sha256=current_hashes,
        source_run_implementation_sha256=manifest.get("implementation_sha256"),
        source_hash_mismatches=[
            name
            for name, digest in current_hashes.items()
            if manifest.get("implementation_sha256", {}).get(name) != digest
        ],
    )
    if report["source_hash_mismatches"]:
        print(
            "WARNING: source hashes differ from the saved run; verify changes are execution-only:",
            report["source_hash_mismatches"],
        )
    for dataset_index, dataset in enumerate(args.datasets, start=1):
        report["active_dataset"] = dataset
        print(
            f"\n[{dataset_index}/{len(args.datasets)}] {dataset}, model seed {args.model_seed}: "
            "checking existing cache/checkpoint...",
            flush=True,
        )
        directory = run / dataset / "conductance"
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if saved != metrics["datasets"][dataset]["models"]["conductance"]:
            raise ValueError(f"Child/root model metrics disagree: {dataset}")
        cache = data_root / "conductance_gat/matched_benchmark_v1" / dataset
        if not all((cache / name).is_file() for name in ("data.pt", "manifest.json")):
            raise FileNotFoundError(f"Existing complete cache required: {cache}")
        payload, protocol = load_dataset(dataset, data_root, allow_download=False)
        for key in ("data_sha256", "split_sha256"):
            if protocol[key] != metrics["datasets"][dataset]["protocol"][key]:
                raise ValueError(f"Dataset differs from the saved run: {dataset}/{key}")
        checkpoint_bytes = (directory / "best.pt").read_bytes()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        model = restore_model(checkpoint, payload, config, saved, history, device)
        before = _state_hash(model)
        item = {
            "status": "running",
            "stage": "baseline_train_validation",
            "history": summarize_history(history, saved),
            "saved_test_historical_only": saved["test"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "checkpoint_path": str(directory / "best.pt"),
            "data_sha256": protocol["data_sha256"],
        }
        report["datasets"][dataset] = item
        print("  Rechecking train/validation and C/rho distributions...", flush=True)
        item["baseline"] = evaluate_checkpoint(model, payload, device, args.edge_chunk_size)
        item["validation_recheck_minus_saved"] = (
            item["baseline"]["validation"]["prediction"]["metric"] - saved["validation"]
        )
        item["validation_recheck_warning"] = abs(item["validation_recheck_minus_saved"]) > 1e-4
        if item["validation_recheck_warning"] and _extended_requested(args):
            raise RuntimeError(
                "Validation recheck differs from saved checkpoint by >1e-4; "
                "inspect source/software/precision before extended interventions"
            )
        additional_audits(model, payload, device, args, config, item)
        if (
            _state_hash(model) != before
            or hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()
            != item["checkpoint_sha256"]
        ):
            raise RuntimeError("Model/checkpoint changed during read-only diagnostics")
        item["model_state_unchanged"] = True
        item.update(status="passed", stage="complete")
        print(
            f"\n{dataset}: best epoch {saved['best_epoch']}/{saved['epochs_run']}; "
            f"saved test ONLY={saved['test']:.6f}"
        )
        history_summary = item["history"]
        print(
            f"  train-mode loss: first={history_summary['train_loss_first']:.6f} "
            f"last={history_summary['train_loss_last']:.6f} "
            f"min={history_summary['train_loss_min']:.6f} "
            f"selected={history_summary['train_loss_at_selected_epoch']:.6f}"
        )
        print(
            f"  historical validation: first={history_summary['validation_first']:.6f} "
            f"best={history_summary['validation_best']:.6f} "
            f"last={history_summary['validation_last']:.6f}"
        )
        print(f"  validation recheck minus saved: {item['validation_recheck_minus_saved']:+.8f}")
        if item["validation_recheck_warning"]:
            print(
                "  WARNING: validation recheck differs by >1e-4; inspect "
                "precision/software/source/checkpoint consistency before interpreting ablations."
            )
        for split, values in item["baseline"].items():
            print(f"  {split}:", json.dumps(values["prediction"]))
            for index, layer in enumerate(values["layers"]):
                _print_layer(index, layer)
        if args.ablate_graph:
            print(
                "  identity-convolution validation delta "
                "(distribution-shift ablation, NOT causal proof):",
                item["identity_minus_original_validation"],
            )
        del model, payload, checkpoint
        torch.cuda.empty_cache()
    report.pop("active_dataset", None)


def _diagnose(args, run: Path, report: dict) -> None:
    """Run the CUDA diagnostic inside a fail-visible resource observation boundary."""

    import torch

    from chartgat.observability import RuntimeResourceMonitor, observed

    for required in (run / "manifest.json", run / "metrics.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Required completed-run artifact is unavailable: {required}")
    device = require_cuda(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    monitor = RuntimeResourceMonitor(device)
    monitor.start()
    try:
        _diagnose_impl(args, run, report, device)
    except BaseException:
        try:
            report["resource_observability"] = monitor.finish(
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as monitor_error:
            report["resource_observability_failure"] = (
                f"{type(monitor_error).__name__}: {monitor_error}"
            )
        raise
    resources = monitor.finish(
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )
    report["resource_observability"] = resources
    elapsed = resources["summary"]["observed_wall_seconds"]["value"]
    completed_datasets = sum(
        item.get("status") == "passed" for item in report.get("datasets", {}).values()
    )
    missing_rate_reason = (
        None
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed > 0
        else "the monitored diagnostic interval had no positive wall duration"
    )
    report["throughput"] = {
        "scope": (
            "end-to-end read-only checkpoint diagnostic, including cache/checkpoint reads, "
            "baseline train/validation forwards and requested interventions/backward audits"
        ),
        "completed_dataset_audits": completed_datasets,
        "dataset_audits_per_second": observed(
            completed_datasets / elapsed if missing_rate_reason is None else None,
            reason=missing_rate_reason,
            unit="datasets_per_second",
        ),
    }


def render_report(report: dict) -> str:
    """A readable companion to the complete machine-readable diagnostic report."""

    lines = [
        "# Conductance checkpoint audit",
        "",
        f"Status: {report['status']}. Model seed: {report['model_seed']} "
        "(one checkpoint per dataset).",
        "",
        "No training, optimizer steps, downloads, original artifact writes or new test queries.",
        "C interventions use validation; gradient audits use train labels only.",
        "These are checkpoint-local observations, not proof of the cause of collapse.",
        "",
    ]
    if report.get("error"):
        lines.extend((f"Error: {report['error']}", ""))
    for dataset, item in report["datasets"].items():
        lines.extend((f"## {dataset}", "", f"Stage: {item.get('stage', 'unknown')}", ""))
        if item.get("error"):
            lines.extend((f"Error: {item['error']}", ""))
        if "baseline" in item:
            lines.extend(("| Split | Metric | Value | Loss |", "|---|---|---:|---:|"))
            for split, values in item["baseline"].items():
                prediction = values["prediction"]
                lines.append(
                    f"| {split} | {prediction['metric_name']} | {prediction['metric']:.8f} | "
                    f"{prediction['loss']:.8f} |"
                )
            lines.extend(("", "Historical test values were read, not re-evaluated.", ""))
            lines.extend(
                (
                    "| Split / layer | C mean | C CV | rho median (ratio) | Conv relative change |",
                    "|---|---:|---:|---:|---:|",
                )
            )
            for split, values in item["baseline"].items():
                for index, layer in enumerate(values["layers"]):
                    edge = layer["edge_pooled"]
                    rho = layer["node_pooled"]["rho"]["quantiles"] or {}
                    lines.append(
                        f"| {split} / {index} | {_number(edge['conductance']['mean'])} | "
                        f"{_number(edge['c_cv'])} | {_number(rho.get('median'))} | "
                        f"{_number(layer['global_update_ratio'])} |"
                    )
            lines.append("")
        if "interventions" in item:
            lines.extend(
                (
                    "### Validation C interventions",
                    "",
                    "| Variant | Metric | Delta | Loss | Logit relative L2 | Prediction flip |",
                    "|---|---:|---:|---:|---:|---:|",
                )
            )
            for variant in item["interventions"]["variants"]:
                p, d = variant["prediction"], variant["delta_vs_learned"]
                lines.append(
                    f"| {variant['name']} | {p['metric']:.8f} | {d['metric']:+.8f} | "
                    f"{p['loss']:.8f} | {_number(d['logits_relative_l2'])} | "
                    f"{_number(d['prediction_flip_fraction'])} |"
                )
            lines.extend(
                (
                    "",
                    "The degree cap is recomputed after C replacement. "
                    "Shuffle changes rho as well as edge alignment.",
                    "Prediction flip is nodewise for multiclass tasks and labelwise for PPI.",
                    "C/rho/update distributions per layer and intervention are in report.json.",
                    "",
                )
            )
        if "gate_audit" in item:
            audit = item["gate_audit"]
            lines.extend(
                (
                    "### Train-label gate/gradient audit",
                    "",
                    f"Mode: {audit['mode']}; loss: `{json.dumps(audit['loss'])}`.",
                    "",
                    "| Parameter | Norm | Task gradient norm | Decay term norm | "
                    "Task/decay | Cosine |",
                    "|---|---:|---:|---:|---:|---:|",
                )
            )
            for name, parameter in audit["parameters"].items():
                lines.append(
                    f"| {name} | {_number(parameter['parameter']['l2_norm'])} | "
                    f"{_number(parameter['task_gradient']['l2_norm'])} | "
                    f"{_number(parameter['weight_decay_term_norm'])} | "
                    f"{_number(parameter['task_to_decay_norm_ratio'])} | "
                    f"{_number(parameter['task_decay_cosine'])} |"
                )
            lines.extend(
                (
                    "",
                    "This compares raw task gradient with lambda*parameter, "
                    "not Adam's historical update.",
                    "",
                    "| Layer / tensor | Mean | Population std | L2 norm | Zero fraction |",
                    "|---|---:|---:|---:|---:|",
                )
            )
            for layer in audit["layers"]:
                for name in (
                    "input_abs_bh",
                    "input_squared_bh",
                    "raw_logit",
                    "conductance",
                    "raw_logit_gradient",
                ):
                    moments = layer["tensors"][name]["all_element_moments"] or {}
                    lines.append(
                        f"| {layer['layer']} / {name} | {_number(moments.get('mean'))} | "
                        f"{_number(moments.get('std_population'))} | "
                        f"{_number(moments.get('l2_norm'))} | "
                        f"{_number(moments.get('zero_fraction'))} |"
                    )
            lines.extend(
                (
                    "",
                    "Moments use all elements; quantiles use explicitly labelled bounded samples.",
                    "Activation/gradient distributions and sample metadata are in report.json.",
                    "",
                )
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = resolve_run(args)
    if args.edge_chunk_size < 1:
        raise ValueError("edge chunk size must be positive")
    if (
        args.shuffle_seed < 0
        or args.gradient_batches < 1
        or args.gradient_sample_limit < 1
        or not math.isfinite(args.near_zero_threshold)
        or args.near_zero_threshold < 0
    ):
        raise ValueError("Invalid audit seed, batch/sample limit, or near-zero threshold")
    output = _automatic_output(args)
    if output:
        source_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        recorded_data_root = Path(
            source_manifest.get("config", {}).get("data_root", ROOT / "data/paper")
        )
        recorded_data_root = recorded_data_root.expanduser().resolve()
        active_data_root = (
            args.data_root.expanduser().resolve() if args.data_root else recorded_data_root
        )
        if (
            output.is_relative_to(run.parents[1])
            or output.is_relative_to(recorded_data_root)
            or output.is_relative_to(active_data_root)
            or output.is_relative_to(ROOT / "data")
        ):
            raise ValueError("Diagnostic output must not be inside the source run/data")
        output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "run": str(run),
        "model_seed": args.model_seed,
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audit_helper_sha256": {
            name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in ("conductance_interventions.py", "conductance_gate_audit.py")
            if (ROOT / "scripts" / name).is_file()
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "output_directory": str(output) if output else None,
        "datasets": {},
        "policy": {
            "optimizer_steps": 0,
            "downloads": False,
            "cache_writes": False,
            "new_test_queries": False,
            "model_seed_count": 1,
            "gradient_audit": (
                f"{args.gradient_mode} mode, autograd ON, train labels only; "
                "checkpoint-local task gradient, not historical Adam updates"
                if args.full_audit or args.gate_audit
                else "disabled"
            ),
            "interventions": (
                "Validation only; graph/layer-local C replacement and recomputed degree cap; "
                "one fixed shuffle seed, no retraining or surrogate-gradient changes"
                if args.full_audit or args.interventions
                else "disabled"
            ),
            "precision": "FP32; AMP/TF32 disabled even if original run used AMP",
            "structural_stats": (
                "All nodes in each graph; transductive test-node features remain visible; "
                "no test-label loss/metric evaluation"
            ),
            "rho": (
                ".95 * weighted_degree / SAME_GRAPH_max_weighted_degree; "
                "gate recomputed in FP32 edge chunks"
            ),
            "c_scale": (
                "Absolute conductance scale is not identifiable; interpret CV and rho together"
            ),
            "ablation": (
                "Optional validation-only same-checkpoint Conv identity; "
                "no retraining, not causal proof"
            ),
        },
    }
    try:
        print(
            "Read-only diagnostic: FP32 inference, AMP/TF32 disabled; train+validation only. "
            "Test scores are saved historical values, not re-evaluated.",
            flush=True,
        )
        print(
            f"One model seed: {args.model_seed}; datasets: {', '.join(args.datasets)}. "
            f"Gradient audit PPI batches: {args.gradient_batches} (default one).",
            flush=True,
        )
        if output:
            print(f"Reports will be saved to: {output}", flush=True)
        _diagnose(args, run, report)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        active = report.get("active_dataset")
        if active in report["datasets"]:
            report["datasets"][active].update(status="failed", error=report["error"])
        if "out of memory" in str(exc).lower():
            report["recovery_note"] = (
                "No CPU fallback or smaller replacement graph was used. Earlier diagnostics "
                "from completed stages are preserved, not unfinished variants within a stage. "
                "Free GPU memory or select one dataset; --interventions skips "
                "backward, while --gate-audit requests backward only after baseline recheck. "
                "The edge chunk option does not bound the original model's backward memory."
            )
        print(report["error"], file=sys.stderr)
    if output:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (output / "report.md").write_text(render_report(report), encoding="utf-8")
        print(f"Diagnostic report: {output / 'report.json'}")
        print(f"Readable report: {output / 'report.md'}")
    resources = report.get("resource_observability")
    if isinstance(resources, dict):
        print(
            "Resource summary:",
            json.dumps(resources.get("summary", {}), allow_nan=False),
            flush=True,
        )
    print(f"Diagnostic status: {report['status']}" + (" (stdout only)" if output is None else ""))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
