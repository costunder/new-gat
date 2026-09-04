"""Train one V1 architecture profile with validation-only checkpoint selection.

This child exists specifically for model-size exploration. Unlike the historical
V1 benchmark command, it never constructs a test loader or reports a test score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from chartgat.cache import atomic_write_json
from chartgat.execution import add_execution_arguments

from .benchmark import _device, _versions, train_model
from .benchmark_data import DATASETS, load_dataset, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults by dataset: 2 for PPI graph minibatches, 1 for full graphs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Default: 4 for PPI graph minibatches, 0 for transductive full graphs",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    add_execution_arguments(parser)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.workers is None:
        args.workers = 4 if args.dataset == "ppi" else 0
        args.worker_configuration_source = "dataset_default"
    elif not hasattr(args, "worker_configuration_source"):
        args.worker_configuration_source = "explicit_cli"
    expected_batch_size = 2 if args.dataset == "ppi" else 1
    if args.batch_size is None:
        args.batch_size = expected_batch_size
    if (
        min(
            args.batch_size,
            args.epochs,
            args.patience,
            args.hidden_channels,
            args.layers,
        )
        < 1
    ):
        raise ValueError(
            "batch size, epochs, patience, hidden channels and layers must be positive"
        )
    if args.workers < 0 or args.model_seed < 0:
        raise ValueError("workers and model seed must be nonnegative")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("dropout/LR/weight decay configuration is invalid")
    if args.dataset != "ppi" and args.batch_size != expected_batch_size:
        raise ValueError(
            f"V1 {args.dataset} is one full graph and requires batch-size={expected_batch_size}"
        )
    if args.dataset != "ppi" and args.workers != 0:
        raise ValueError("transductive V1 datasets use no DataLoader and require workers=0")
    if args.amp or args.compile:
        raise ValueError("V1 scaling fixes AMP and compilation off across architecture profiles")


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    loader_workers = args.workers if args.dataset == "ppi" else 0
    return {
        "hidden_channels": args.hidden_channels,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "workers": loader_workers,
        "device": args.device,
        "amp": args.amp,
        "compile": args.compile,
        "pin_memory": args.pin_memory,
        "persistent_workers": loader_workers > 0,
        "prefetch_factor": 2 if loader_workers > 0 else None,
        "worker_configuration_source": getattr(
            args, "worker_configuration_source", "explicit_cli"
        ),
        "validation_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate(args)
    # train_model and _make_loaders inspect this internal marker. It is not added
    # to the historical benchmark parser or its default output contract.
    args.validation_only = True
    device = _device(args.device, prepare_only=False)
    output = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if output == data_root or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("V1 scaling output and dataset cache must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is not a new empty child directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "research_suite": "conductance_scaling_v1",
        "dataset": args.dataset,
        "condition": "conductance",
        "model_seed": args.model_seed,
        "configuration": _configuration(args),
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    atomic_write_json(output / "metrics.json", record)
    try:
        payload, protocol = load_dataset(args.dataset, data_root, allow_download=False)
        result = train_model(payload, args, device, output)
        if "test" in result or result.get("test_evaluated") is not False:
            raise RuntimeError("V1 scaling child exposed a test metric")
        record.update(
            result,
            status="passed",
            metric_name=protocol["metric"],
            protocol=protocol,
            cache_sha256=protocol["data_sha256"],
            versions=_versions(),
            source_sha256={
                Path(__file__).name: sha256_file(Path(__file__)),
                "benchmark.py": sha256_file(Path(__file__).with_name("benchmark.py")),
                "src/chartgat/observability.py": sha256_file(
                    Path(__file__).resolve().parents[2] / "src/chartgat/observability.py"
                ),
            },
        )
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            atomic_write_json(output / "metrics.json", record)
        except BaseException as reporting_error:
            exc.add_note(
                "failed metrics could not be written without replacing this error: "
                f"{type(reporting_error).__name__}: {reporting_error}"
            )
        raise
    atomic_write_json(output / "metrics.json", record)
    print(f"passed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
