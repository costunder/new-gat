"""Read-only validation audit of completed corrected V5 checkpoints.

No training, optimizer, checkpoint recovery/publication, dataset download, or
test evaluation. All reports go to stdout. Source compatibility here authorizes
only this diagnostic, never a training resume or a changed model recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for _entry in (ROOT, ROOT / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

SUITE = "conductance_graph_conditioned_v5"
BASE_COMMIT = "8da06cacec59515d84c08d892315e4c8ecfd5b5b"
BASELINE_SOURCES = {
    "research/conductance_gat/ablation/train.py": (
        "522df8ee0c7d40acb2db6814bcf6065648f8674fb3669d299ffb9872c66dbff5"
    ),
    "research/conductance_gat/benchmark.py": (
        "c441a65e22f0431e0c362d4eeadb3036aaceabb5b89863bd8709e6b7680b5d13"
    ),
    "research/conductance_gat/benchmark_data.py": (
        "082e4a1a94bb7beffddfa2d5f59a922c806f15093d5ffc97fb3240b0bb844dc7"
    ),
    "research/conductance_gat/v5/__init__.py": (
        "8b313a477bbd51eedc0861fad06cffdf7f7a0aed5d9fcddc55fa89c333a3bd46"
    ),
    "research/conductance_gat/v5/batch_calibration.py": (
        "265ebacb059289f1b7445cf0db1fc9f8c2359de52fd2c83cf455f2ebfa264ddc"
    ),
    "research/conductance_gat/v5/diagnostics.py": (
        "b1d629f638319c95a19997b0525a31f1d7aa44dcfe0451a7e20d979f11845a68"
    ),
    "research/conductance_gat/v5/learning_budget.py": (
        "e0ae75c0ca5bd2423ea7903ec920c871acb528e05ca07c326672153951eca7a1"
    ),
    "research/conductance_gat/v5/model.py": (
        "963881743c2c7113df5cecaf05d91670b190d9fb7a3ca3a3f976d363d6925282"
    ),
    "research/conductance_gat/v5/operator.py": (
        "f79b59708b5787ff79ffc3c0a8976432bccaa8f4cb6f6edbf172d3f037bd03d3"
    ),
    "research/conductance_gat/v5/optimization.py": (
        "34fc5475cd95966d63f14b720d82c73062bf3586236ec8671258a1f3e5d4c49b"
    ),
    "research/conductance_gat/v5/protocol.py": (
        "3018d646b044ef8b2d1eba786d7ce97bba70149dbd9692f15da4ca1925e4991a"
    ),
    "research/conductance_gat/v5/report.py": (
        "8fdcbcf02a1658873d6c6102beb904a2ffb2f711e5a484dbda674d0ab1f79f1d"
    ),
    "research/conductance_gat/v5/sampling.py": (
        "98333613164281dbb599189fb91703d6b231413ffbbe6b0dafde5fa93d28a6a2"
    ),
    "research/conductance_gat/v5/timing.py": (
        "c4767edd02f5b7a6c0c857cc7f17b2f22c36b1404576481c60d66d967a76d033"
    ),
    "research/conductance_gat/v5/train.py": (
        "b3f4bf7503651c75cd3f3f1b030bff26f243a45eda5f0d645e9a842664b3faec"
    ),
    "research/conductance_gat/v5/transition.py": (
        "fccfe27baec519231ba1405c56f17375a460adba663b6ad3b98176b295f51700"
    ),
    "research/conductance_gat/v5/transition_initialization.py": (
        "574c9ce4f61aa7f64a071bdbb00bea95f3f229612647f3d5699961ef43004611"
    ),
    "research/conductance_gat/v5/transition_report.py": (
        "872528bf12290db06c28374f0a6160dbe808d0709e3a76ebb3a4df83ec63d51c"
    ),
    "research/conductance_gat/v5/transition_training.py": (
        "9a57550941505654eafeeeb90cd509e0e7e22eed0d2f07a8a4148f81264de4c3"
    ),
    "scripts/resume_compatibility_v1.json": (
        "aa496773d18f9ae7eee9fff83f45a88f8e6c4ffc773852a6ea2a2028f25605e9"
    ),
    "src/chartgat/cache.py": ("b9feeef3c3e033a064677a612790b036d78d09eecddce773a44273e007cd7cab"),
    "src/chartgat/observability.py": (
        "53b1b037f84371ad51abb9524c808b5d72cf27f3c66eca8fcc97a825e81755b1"
    ),
    "src/chartgat/resume_compat.py": (
        "c503e207ff10db490b2624473ae330c2cee7873bf3ebdd930ce1d9cc682076bd"
    ),
}
# Exact reviewed diagnostic/observability additions. No wildcard source bypass.
AUDIT_SOURCE_OVERRIDES: dict[str, str] = {
    "research/conductance_gat/v5/batch_calibration.py": (
        "df8e6a975e9a25cc2672ea64636afef43607a7da764ad73d44c5bdd75612d966"
    ),
    "research/conductance_gat/v5/protocol.py": (
        "c476c1c4e362968396031aa3fb03bb8c4807c6542741f17c46851695ebae473e"
    ),
    "research/conductance_gat/v5/sampling.py": (
        "05c24c85765bdc0b0c1bc9a58261f849fc4af8eb11fec9e4b101a2dcff8f3803"
    ),
    "research/conductance_gat/v5/stage_audit.py": (
        "44e99513aa00f080945a25ae802b991854128e36bf3752740b1f30f0e77191e4"
    ),
    "research/conductance_gat/v5/train.py": (
        "a64e46d0c340a5f65920a0de4ca2cb8707bd3900410753e15da78e543f2e04a3"
    ),
}


class AuditError(ValueError):
    """Evidence is incomplete, corrupt, or outside the reviewed audit contract."""


def _finite_positive(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Existing run or condition directory"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/paper")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference-steps",
        type=_positive_integer,
        required=True,
        help="Explicit diagnostic reference K; never changes training K",
    )
    parser.add_argument(
        "--reference-tolerance",
        type=_finite_positive,
        required=True,
        help="Explicit convergence residual tolerance for the reference",
    )
    parser.add_argument("--json", action="store_true", help="One full JSON report to stdout")
    return parser


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"Required regular file is missing or is a symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise AuditError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value):
    raise AuditError(f"Nonfinite JSON value: {value}")


def _read(path: Path) -> Any:
    _sha(path)
    return json.loads(
        path.read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs, parse_constant=_constant
    )


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def discover_conditions(root: Path) -> list[Path]:
    root = root.expanduser()
    if root.is_symlink() or not root.is_dir():
        raise AuditError(f"--root must be an existing regular directory: {root}")
    found = []
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if not (Path(folder) / name).is_symlink()
        )
        if "metrics.json" not in files:
            continue
        path = Path(folder) / "metrics.json"
        metrics = _read(path)
        if isinstance(metrics, dict) and metrics.get("research_suite") == SUITE:
            found.append(path.resolve())
    if not found:
        raise AuditError(f"No V5 metrics.json found under {root}")
    return sorted(found)


def verify_sources(previous: Any, current: dict[str, str]) -> dict[str, Any]:
    from chartgat.resume_compat import require_source_compatibility

    target = {**BASELINE_SOURCES, **AUDIT_SOURCE_OVERRIDES}
    if current != target:
        changed = sorted(
            key for key in set(current) | set(target) if current.get(key) != target.get(key)
        )
        raise AuditError(f"Live audit sources are not the exact reviewed release: {changed}")
    if previous == target:
        transition = None
    else:
        transition = require_source_compatibility(previous, BASELINE_SOURCES)
    return {
        "base_commit": BASE_COMMIT,
        "training_resume_authorized": False,
        "scope": "read-only validation diagnostic only",
        "historical_repair": transition,
        "checkpoint_sources": previous,
        "audit_sources": current,
    }


def inspect_evidence(path: Path) -> dict[str, Any]:
    metrics = _read(path)
    if not isinstance(metrics, dict) or metrics.get("research_suite") != SUITE:
        raise AuditError(f"Not a V5 metrics artifact: {path}")
    if metrics.get("status") != "passed":
        raise AuditError(f"Only completed passed checkpoints can be audited: {path}")
    if metrics.get("condition") not in {"fixed_c", "shared_dynamic_c"}:
        raise AuditError("Unsupported V5 condition")
    identity = metrics.get("resume_identity")
    if not isinstance(identity, dict) or _canonical(identity) != metrics.get(
        "resume_identity_sha256"
    ):
        raise AuditError("metrics resume identity SHA256 mismatch")
    for key in ("research_suite", "dataset", "condition", "configuration"):
        if identity.get(key) != metrics.get(key):
            raise AuditError(f"metrics and resume identity disagree on {key}")
    protocol = metrics.get("protocol")
    if not isinstance(protocol, dict) or protocol != identity.get("dataset_protocol"):
        raise AuditError("metrics and identity dataset protocol mismatch")
    if _canonical(protocol) != identity.get("dataset_protocol_sha256"):
        raise AuditError("dataset protocol SHA256 mismatch")
    if protocol.get("data_sha256") != metrics.get("cache_sha256") or metrics.get(
        "cache_sha256"
    ) != identity.get("cache_sha256"):
        raise AuditError("dataset cache SHA256 mismatch")
    if metrics.get("source_sha256") != identity.get("source_sha256"):
        raise AuditError("metrics and identity source fingerprints disagree")
    config = metrics.get("configuration")
    required = {
        "conductance_backend": "optimization",
        "training_schedule": "joint",
        "solver_cost_scaling": "width_scaled",
        "beta_parameterization": "sigmoid",
        "beta_initial": 0.5,
    }
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in required.items()):
        raise AuditError(
            "This audit requires the explicitly corrected optimization/joint/width_scaled V5 recipe"
        )
    for name in (
        "hidden_channels",
        "layers",
        "heads",
        "ffn_multiplier",
        "solver_steps",
        "batch_size",
    ):
        if type(config.get(name)) is not int or config[name] < 1:
            raise AuditError(f"Missing/invalid saved configuration: {name}")
    if (
        type(metrics.get("best_epoch")) is not int
        or metrics["best_epoch"] < 1
        or not _number(metrics.get("validation"))
    ):
        raise AuditError("Selected checkpoint epoch or validation score is invalid")
    selection = metrics.get("checkpoint_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("primary_epoch") != metrics["best_epoch"]
        or selection.get("primary_validation") != metrics["validation"]
    ):
        raise AuditError("Selected checkpoint role disagrees with metrics")
    if (
        metrics.get("evaluation_split") != "validation"
        or metrics.get("test_evaluated") is not False
        or selection.get("test_used") is not False
    ):
        raise AuditError("Training checkpoint selection must be validation-only")
    checkpoint, history_path = path.parent / "best.pt", path.parent / "history.json"
    for artifact, expected in (
        (checkpoint, metrics.get("checkpoint_sha256")),
        (history_path, metrics.get("history_sha256")),
    ):
        if _sha(artifact) != expected:
            raise AuditError(f"Artifact SHA256 does not match metrics: {artifact}")
    rows = _read(history_path)
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise AuditError("History must be a nonempty list of epoch records")
    epochs = [row.get("epoch") for row in rows]
    if any(type(epoch) is not int or epoch < 1 for epoch in epochs) or any(
        a >= b for a, b in zip(epochs, epochs[1:], strict=False)
    ):
        raise AuditError("History epochs are invalid or not strictly increasing")
    selected = [row for row in rows if row["epoch"] == metrics["best_epoch"]]
    if len(selected) != 1 or selected[0].get("validation") != metrics["validation"]:
        raise AuditError("Selected epoch validation is absent or inconsistent in history")
    artifacts = {str(item): _sha(item) for item in (path, checkpoint, history_path)}
    return {
        "metrics": metrics,
        "history": rows,
        "checkpoint": checkpoint,
        "artifacts": artifacts,
        "metrics_path": path,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def historical_diagnostics(metrics: dict, rows: list[dict]) -> dict:
    from scripts.analyze_v5_results import _diagnostics

    diagnostics = _diagnostics(metrics, rows)
    diagnostics["epoch_layers"] = [
        {"epoch": row["epoch"], "layers": row["layers"]}
        for row in rows
        if isinstance(row.get("layers"), list)
    ]
    diagnostics["sampling_metadata"] = metrics.get("sampling", _unavailable("not recorded"))
    observations = [
        {"epoch": row["epoch"], "sampling_observation": row["sampling_observation"]}
        for row in rows
        if "sampling_observation" in row
    ]
    batch_observations = [
        {"epoch": row["epoch"], "batches": row["batch_observations"]}
        for row in rows
        if isinstance(row.get("batch_observations"), list)
    ]
    diagnostics["batch_observations"] = batch_observations or _unavailable(
        "Per-batch node/edge/seed counts were not recorded in this historical history"
    )
    diagnostics["sampling_observations"] = (
        observations
        or batch_observations
        or _unavailable(
            "Historical batches/overlap were not recorded; no retrospective batch diversity claim"
        )
    )
    return diagnostics


def validation_source(payload: dict, config: dict):
    """Construct only the official validation source; never select test indices."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if payload["dataset"] != "ppi":
        indices = payload["splits"]["validation"].nonzero(as_tuple=False).flatten().long()
        graph = Data(**payload["graphs"][0])
        if not indices.numel():
            raise AuditError("Official validation mask is empty")
        return (
            graph,
            indices,
            {
                "validation_nodes": indices.numel(),
                "graph_nodes": graph.x.shape[0],
                "graph_edges": graph.incidence_edge_index.shape[1],
                "used_fraction": 1.0,
            },
        )
    selected = payload["splits"]["validation"]
    graphs = [Data(**payload["graphs"][int(index)]) for index in selected]
    if not graphs:
        raise AuditError("Official PPI validation graph split is empty")
    workers = config.get("workers")
    if type(workers) is not int or workers < 0 or type(config.get("pin_memory")) is not bool:
        raise AuditError("Missing saved PPI worker/pin-memory settings")
    loader = DataLoader(
        graphs,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=config["pin_memory"],
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    return (
        loader,
        None,
        {
            "validation_graphs": len(graphs),
            "validation_nodes": sum(g.x.shape[0] for g in graphs),
            "graph_nodes": [g.x.shape[0] for g in graphs],
            "graph_edges": [g.incidence_edge_index.shape[1] for g in graphs],
            "saved_graph_batch_size": config["batch_size"],
            "actual_physical_batch_size": min(config["batch_size"], len(graphs)),
            "used_fraction": 1.0,
            "workers": workers,
        },
    )


def verify_unchanged(artifacts: dict[str, str]) -> None:
    for name, expected in artifacts.items():
        if _sha(Path(name)) != expected:
            raise AuditError(f"Read-only audit input changed during execution: {name}")


def _monitored_stage_audit(model, source, indices, *, device, **kwargs) -> tuple[dict, dict]:
    """Sample the actual audit interval and always finish a started monitor.

    CUDA peaks were reset by the caller before model allocation. This wrapper
    never resets them and makes no numeric-ordinal NVML mapping of its own.
    """
    import torch

    from chartgat.observability import RuntimeResourceMonitor
    from research.conductance_gat.v5.stage_audit import audit_stage_roles

    monitor = RuntimeResourceMonitor(device)
    started = False
    primary_error = None
    cleanup_errors = []
    telemetry = None
    peaks = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    clock_started = time.perf_counter()
    try:
        monitor.start()
        started = True
        result = audit_stage_roles(model, source, indices, device=device, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if started:
            if device.type == "cuda":
                for key, counter in (
                    ("peak_allocated_bytes", torch.cuda.max_memory_allocated),
                    ("peak_reserved_bytes", torch.cuda.max_memory_reserved),
                ):
                    try:
                        peaks[key] = int(counter(device))
                    except BaseException as error:
                        cleanup_errors.append((key, error))
            # A failed counter read must not bypass stopping/joining the sampler.
            try:
                telemetry = monitor.finish(**peaks)
            except BaseException as error:
                cleanup_errors.append(("runtime_resource_monitor_finish", error))
        resources = {
            "scope": "this condition only; CUDA allocator peaks reset before model allocation",
            "telemetry_scope": (
                "validation audit interval including local recomputations and all interventions; "
                "GPU utilization is device-wide, not attributed solely to this process"
            ),
            "elapsed_seconds": time.perf_counter() - clock_started,
            **peaks,
            "runtime_observability": telemetry,
            "resource_monitor": {
                "started": started,
                "finish_attempted": started,
                "finished": telemetry is not None,
                "cleanup_errors": [
                    {"stage": stage, "error": f"{type(error).__name__}: {error}"}
                    for stage, error in cleanup_errors
                ],
            },
            "gpu_utilization": (
                telemetry["summary"]["run_average_gpu_sm_utilization_percent"]
                if telemetry is not None
                else _unavailable(
                    "runtime resource monitor did not produce a completed report; "
                    "see primary error and resource_monitor cleanup errors"
                )
            ),
        }
        if primary_error is not None:
            primary_error.audit_resources = resources
            for stage, error in cleanup_errors:
                primary_error.add_note(
                    f"Audit resource cleanup also failed at {stage}: "
                    f"{type(error).__name__}: {error}"
                )
        elif cleanup_errors:
            stage, error = cleanup_errors[0]
            error.audit_resources = resources
            error.add_note(
                f"Audit numerical forwards completed; resource cleanup failed at {stage}"
            )
            for additional_stage, additional in cleanup_errors[1:]:
                error.add_note(
                    f"Additional cleanup failure at {additional_stage}: "
                    f"{type(additional).__name__}: {additional}"
                )
            raise error
    return result, resources


def audit_condition(evidence: dict, args, device) -> dict:
    import torch

    from research.conductance_gat.v5 import train

    metrics, config = evidence["metrics"], evidence["metrics"]["configuration"]
    verification = evidence.setdefault(
        "audit_verification",
        {"source_verified": False, "cache_split_hashes_verified": None},
    )
    provenance = verify_sources(metrics["source_sha256"], train.implementation_source_hashes())
    verification["source_verified"] = True
    if args.reference_steps <= config["solver_steps"]:
        raise AuditError("--reference-steps must exceed the saved training solver_steps")
    runtime = SimpleNamespace(
        **{
            **config,
            "dataset": metrics["dataset"],
            "condition": metrics["condition"],
            "device": str(device),
            "beta_min": config.get("beta_min"),
            "beta_max": config.get("beta_max"),
        }
    )
    train._require_cuda(device)
    hardware = train.validate_hardware_runtime(runtime, device)
    train.configure_compute(runtime)
    selected = train.load_checkpoint_on_cpu(evidence["checkpoint"])
    train.validate_selected_checkpoint(
        selected,
        expected_identity=metrics["resume_identity"],
        expected_identity_sha256=metrics["resume_identity_sha256"],
        expected_epoch=metrics["best_epoch"],
        expected_metric=metrics["validation"],
        expected_selection_role="primary",
    )
    architecture = train.architecture_configuration(runtime)
    if (
        selected.get("architecture") != architecture
        or selected.get("configuration") != config
        or selected.get("condition") != metrics["condition"]
    ):
        raise AuditError(
            "Selected checkpoint architecture/recipe/condition disagrees with immutable metrics"
        )
    cache_folder = (
        args.data_root.expanduser().resolve()
        / "conductance_gat/matched_benchmark_v1"
        / metrics["dataset"]
    )
    cache_artifacts = {
        str(path): _sha(path) for path in (cache_folder / "data.pt", cache_folder / "manifest.json")
    }
    artifacts = {**evidence["artifacts"], **cache_artifacts}
    try:
        payload, protocol = train.load_dataset(
            metrics["dataset"], args.data_root, allow_download=False
        )
        for name in ("data_sha256", "split_sha256"):
            if protocol.get(name) != metrics["protocol"].get(name):
                raise AuditError(f"Official data fingerprint mismatch: {name}")
        verification["cache_split_hashes_verified"] = True
        source, indices, source_metadata = validation_source(payload, config)
        torch.cuda.reset_peak_memory_stats(device)
        model = train.GraphConditionedConductanceNodeClassifier(
            payload["graphs"][0]["x"].shape[1],
            payload["classes"],
            **architecture,
            conductance_mode=train.CONDITIONS[metrics["condition"]]["conductance_mode"],
            max_log_conductance=config["max_log_conductance"],
            edge_chunk_size=config["edge_chunk_size"],
        ).to(device)
        model.load_state_dict(selected["model_state"], strict=True)
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count != metrics.get("total_parameters"):
            raise AuditError("Restored model parameter count differs from recorded metrics")
        model.eval()
        result, resources = _monitored_stage_audit(
            model,
            source,
            indices,
            device=device,
            precision=config["precision"],
            reference_steps=args.reference_steps,
            reference_tolerance=args.reference_tolerance,
        )
        return {
            "dataset": metrics["dataset"],
            "condition": metrics["condition"],
            "metrics_path": str(evidence["metrics_path"]),
            "configuration": config,
            "selected_epoch": metrics["best_epoch"],
            "selected_validation": metrics["validation"],
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "source_provenance": provenance,
            "verification": dict(verification),
            "validation_source": source_metadata,
            "hardware": hardware,
            "total_parameters": parameter_count,
            "historical_diagnostics": historical_diagnostics(metrics, evidence["history"]),
            "audit": result,
            "resources": resources,
            "files_written": False,
            "test_evaluated": False,
            "optimizer_created": False,
        }
    finally:
        primary_error = sys.exception()
        try:
            verify_unchanged(artifacts)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "Immutable input verification also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


def _score(value) -> str:
    return f"{100 * value:.4f}%" if _number(value) else "unavailable"


def _resource_value(observation, *, divisor: float = 1.0, suffix: str = "") -> str:
    if not isinstance(observation, dict):
        return "unavailable (not recorded)"
    value = observation.get("value")
    if not _number(value):
        return f"unavailable ({observation.get('reason') or 'no finite observation'})"
    return f"{value / divisor:.3f}{suffix}"


def human_resources(resource: dict) -> list[str]:
    telemetry = resource.get("runtime_observability") or {}
    summary = telemetry.get("summary", {})
    series = telemetry.get("interval_series", {})
    return [
        "  measured GPU SM mean="
        + _resource_value(resource.get("gpu_utilization"), suffix="%")
        + " (device-wide); CPU process="
        + _resource_value(summary.get("average_cpu_percent_of_one_core"), suffix="% of one core"),
        "  sampled RSS max="
        + _resource_value(
            series.get("process_resident_bytes", {}).get("maximum"),
            divisor=2**30,
            suffix=" GiB",
        )
        + "; host RAM min available="
        + _resource_value(
            series.get("system_available_bytes", {}).get("minimum"),
            divisor=2**30,
            suffix=" GiB",
        )
        + f"; interval samples={telemetry.get('background_sample_count', 'unavailable')}",
        *(
            ["  resource sampler errors: " + "; ".join(telemetry["sampler_errors"])]
            if telemetry.get("sampler_errors")
            else []
        ),
    ]


def sampling_summary(historical: dict) -> dict:
    rows = historical.get("batch_observations")
    if not isinstance(rows, list) or not rows:
        return {"available": False, "reason": "Historical per-batch sizes/overlap not recorded"}
    latest = rows[-1]
    batches = latest["batches"]
    observations = [
        batch["sampling_observation"]
        for batch in batches
        if isinstance(batch.get("sampling_observation"), dict)
    ]
    pairs = [
        observation["previous_physical_batch_overlap"]
        for observation in observations
        if isinstance(observation.get("previous_physical_batch_overlap"), dict)
    ]
    context_pairs = [
        pair
        for observation in observations
        for pair in observation.get("within_batch_context_overlap", [])
    ]

    def span(values):
        finite = [value for value in values if _number(value)]
        return {"min": min(finite), "max": max(finite)} if finite else None

    return {
        "available": True,
        "scope": "last recorded epoch only",
        "epoch": latest["epoch"],
        "batches": len(batches),
        "nodes": span(batch.get("nodes") for batch in batches),
        "physical_edges": span(batch.get("physical_edges") for batch in batches),
        "previous_batch_node_jaccard": span(pair.get("nodes", {}).get("jaccard") for pair in pairs),
        "previous_batch_edge_jaccard": span(
            pair.get("physical_edges", {}).get("jaccard") for pair in pairs
        ),
        "within_batch_node_jaccard": span(
            pair.get("nodes", {}).get("jaccard") for pair in context_pairs
        ),
        "overlap_available": bool(pairs or context_pairs),
        "saturated_batch_count": sum(
            observation.get("cluster_budget_saturated") is True for observation in observations
        )
        if observations
        else None,
    }


def human_condition(row: dict) -> str:
    title = (
        f"[{row.get('dataset', '?')} | {row.get('condition', '?')}] {row.get('metrics_path', '')}"
    )
    if row.get("execution_status") == "failed":
        lines = [title, "  FAILED: " + row["error"]]
        lines.extend("  " + note for note in row.get("error_notes", []))
        if "resources" in row:
            lines.extend(human_resources(row["resources"]))
        return "\n".join(lines)
    audit = row["audit"]
    lines = [
        title,
        f"  selected epoch={row['selected_epoch']}; "
        f"saved validation={_score(row['selected_validation'])}",
        f"  execution={audit['execution_status']}; contribution={audit['contribution_status']} "
        "(not a usefulness certificate)",
    ]
    for name, observation in audit["interventions"].items():
        delta = observation.get("delta_from_learned")
        lines.append(
            f"  {name}: validation={_score(observation.get('metric'))}; delta={100 * delta:+.4f} pp"
            if _number(delta)
            else f"  {name}: validation={_score(observation.get('metric'))}"
        )
        lines.append(
            f"    logit relative-L2={observation.get('logit_relative_l2')}; "
            f"max-abs={observation.get('logit_max_abs')}; "
            f"prediction changed={_score(observation.get('prediction_changed_fraction'))}"
        )
    lines.append(
        "  reference contract: " + json.dumps(audit.get("reference_contract"), ensure_ascii=False)
    )
    historical = row["historical_diagnostics"]
    if isinstance(historical.get("layers"), list):
        for layer in historical["layers"]:
            lines.append(
                f"  recorded layer={layer['layer']}: C CV={layer['c_cv']}; beta mean/min/max="
                f"{layer['beta_mean']}/{layer['beta_min']}/{layer['beta_max']}"
            )
    for local in audit.get("local_layer_comparisons", []):
        reference = local["operator_deployed_vs_reference"]
        one = local.get("operator_c_one_vs_deployed")
        one_difference = one.get("relative_l2") if isinstance(one, dict) else None
        lines.append(
            f"  local batch/layer={local['batch']}/{local['layer']}: "
            f"C=1 operator relative-L2={one_difference}; "
            f"K-reference relative-L2={reference.get('relative_l2')}; "
            f"residual tolerance reached={local.get('reference_tolerance_reached_by_graph')}"
        )
        baseline_residual = local.get("baseline_solver", {}).get("projected_gradient_rms_final")
        reference_residual = local.get("reference_solver", {}).get("projected_gradient_rms_final")
        lines.append(
            f"    frozen H/B/W/beta: C relative-L2="
            f"{local.get('c_deployed_vs_reference', {}).get('relative_l2')}; "
            f"residual deployed/reference={baseline_residual}/{reference_residual}; "
            f"tolerance={local.get('reference_residual_tolerance')}; exact optimum=False"
        )
    samples = sampling_summary(historical)
    lines.append("  historical sampling: " + json.dumps(samples, ensure_ascii=False))
    resource = row["resources"]
    lines.append(
        f"  condition peak allocated={resource['peak_allocated_bytes'] / 2**30:.3f} GiB; "
        f"elapsed={resource['elapsed_seconds']:.2f}s"
    )
    lines.extend(human_resources(resource))
    return "\n".join(lines)


def aggregate_verification(rows: list[dict]) -> dict:
    keys = ("source_verified", "cache_split_hashes_verified")
    result = {}
    for key in keys:
        values = [row.get("verification", {}).get(key) for row in rows]
        result[key] = all(values) if values and all(isinstance(v, bool) for v in values) else None
    result["scope"] = (
        "all requested conditions; null means at least one condition failed before verification"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict] = []
    try:
        if not (args.device == "cuda" or args.device.startswith("cuda:")):
            raise AuditError("CUDA is required; CPU fallback is forbidden")
        paths = discover_conditions(args.root)
        evidences = [inspect_evidence(path) for path in paths]
        import torch

        from research.conductance_gat.v5 import train

        device = torch.device(args.device)
        train._require_cuda(device)
        for index, evidence in enumerate(evidences, start=1):
            if not args.json:
                print(
                    f"[{index}/{len(evidences)}] validation-only audit: "
                    f"{evidence['metrics_path'].parent}",
                    flush=True,
                )
            try:
                row = audit_condition(evidence, args, device)
            except (AuditError, RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
                row = {
                    "execution_status": "failed",
                    "dataset": evidence["metrics"]["dataset"],
                    "condition": evidence["metrics"]["condition"],
                    "metrics_path": str(evidence["metrics_path"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_notes": list(getattr(exc, "__notes__", [])),
                    "verification": dict(evidence.get("audit_verification", {})),
                    "files_written": False,
                    "test_evaluated": False,
                }
                if hasattr(exc, "audit_resources"):
                    row["resources"] = exc.audit_resources
            rows.append(row)
            if not args.json:
                print(human_condition(row), flush=True)
        failures = sum(row.get("execution_status") == "failed" for row in rows)
        peaks = [
            row["resources"]["peak_allocated_bytes"]
            for row in rows
            if _number(row.get("resources", {}).get("peak_allocated_bytes"))
        ]
        report = {
            "execution_status": "failed" if failures else "passed",
            "conditions": rows,
            "failed_conditions": failures,
            "files_written": False,
            "test_evaluated": False,
            "verification": aggregate_verification(rows),
            "scope": (
                "full official validation only; cache integrity checks split metadata, "
                "never test predictions"
            ),
            "aggregate_resources": {
                "max_condition_peak_allocated_bytes": max(peaks) if peaks else None,
                "scope": (
                    "maximum of individually reset condition peaks; not the last condition peak"
                ),
                "missing_conditions": failures,
            },
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        else:
            print(
                f"Audit: {report['execution_status']}; {len(rows)} conditions; {failures} failed. "
                "No files written, no training, no test evaluation."
            )
        return 1 if failures else 0
    except (AuditError, RuntimeError, ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        error = {
            "execution_status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "files_written": False,
            "test_evaluated": False,
            "conditions": rows,
        }
        print(
            json.dumps(error, ensure_ascii=False)
            if args.json
            else f"Audit failed: {error['error']}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
