#!/usr/bin/env python3
"""Read-only CUDA checkpoint diagnostics; no training, downloads, or new test queries."""

from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=["cora", "ppi", "ogbn-arxiv"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, help="Optional NEW directory; default is stdout only"
    )
    parser.add_argument("--edge-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--ablate-graph",
        action="store_true",
        help="Validation-only identity-convolution ablation; no retraining",
    )
    return parser


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


def _diagnose(args, run: Path, report: dict) -> None:
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
    device = require_cuda(args.device)
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
    for dataset in args.datasets:
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
            "history": summarize_history(history, saved),
            "saved_test_historical_only": saved["test"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "checkpoint_path": str(directory / "best.pt"),
            "data_sha256": protocol["data_sha256"],
            "baseline": evaluate_checkpoint(model, payload, device, args.edge_chunk_size),
        }
        item["validation_recheck_minus_saved"] = (
            item["baseline"]["validation"]["prediction"]["metric"] - saved["validation"]
        )
        item["validation_recheck_warning"] = abs(item["validation_recheck_minus_saved"]) > 1e-4
        if args.ablate_graph:
            item["identity_convolution_validation"] = evaluate_checkpoint(
                model, payload, device, args.edge_chunk_size, ablate=True
            )["validation"]
            item["identity_minus_original_validation"] = (
                item["identity_convolution_validation"]["prediction"]["metric"]
                - item["baseline"]["validation"]["prediction"]["metric"]
            )
        if (
            _state_hash(model) != before
            or hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()
            != item["checkpoint_sha256"]
        ):
            raise RuntimeError("Model/checkpoint changed during read-only diagnostics")
        item["model_state_unchanged"] = True
        report["datasets"][dataset] = item
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = resolve_run(args)
    if args.edge_chunk_size < 1:
        raise ValueError("edge chunk size must be positive")
    output = args.output_dir.expanduser().resolve() if args.output_dir else None
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
        "status": "running",
        "run": str(run),
        "model_seed": args.model_seed,
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "datasets": {},
        "policy": {
            "optimizer_steps": 0,
            "downloads": False,
            "cache_writes": False,
            "new_test_queries": False,
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
            "Test scores are saved historical values, not re-evaluated."
        )
        _diagnose(args, run, report)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr)
    if output:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Diagnostic report: {output / 'report.json'}")
    print(f"Diagnostic status: {report['status']}" + (" (stdout only)" if output is None else ""))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
