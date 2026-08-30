#!/usr/bin/env python3
"""Run the independent paper experiment tracks on a CUDA host."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chartgat.cache import atomic_write_bytes, atomic_write_json

try:
    from scripts.aggregate_paper import aggregate_manifest
    from scripts.check_dependencies import DependencyCheckError, check_dependencies, error_message
except ModuleNotFoundError:  # Direct ``python scripts/run_paper.py`` execution.
    from aggregate_paper import aggregate_manifest
    from check_dependencies import DependencyCheckError, check_dependencies, error_message

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_MODULES = {
    "conductance_gat": "research.conductance_gat.paper",
    "cycle_pe": "research.cycle_pe.paper",
    "tree_augmentation": "research.tree_augmentation.paper",
}
BENCHMARK_MODULES = {
    "conductance_gat": "research.conductance_gat.benchmark",
    "cycle_pe": "research.cycle_pe.benchmark",
}
CYCLE_BREC_OFFICIAL_SEEDS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
CYCLE_VARIANTS = ("no_pe", "raw", "set", "projector")
DEFAULT_CYCLE_VARIANTS = ("raw", "set", "projector")
CYCLE_CORE_TARGETS = ("edge", "node", "graph")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("paper-%Y%m%dT%H%M%S%fZ")


def _run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    return seeds


def _comma_subset(value: str, *, choices: tuple[str, ...], option: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError(f"{option} must be non-empty")
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"{option} must not contain duplicates")
    unknown = sorted(set(selected) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"{option} contains unsupported values {unknown}; choose from {list(choices)}"
        )
    return selected


def _cycle_variants(value: str) -> tuple[str, ...]:
    return _comma_subset(value, choices=CYCLE_VARIANTS, option="--cycle-variants")


def _cycle_core_targets(value: str) -> tuple[str, ...]:
    return _comma_subset(value, choices=CYCLE_CORE_TARGETS, option="--cycle-core-targets")


def _selected_tracks(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return tuple(TRACK_MODULES)
    return tuple(dict.fromkeys(values))


def _track_run_root(track: str, run_id: str, results_root: Path | None = None) -> Path:
    if results_root is None:
        base = PROJECT_ROOT / "research" / track / "results" / "paper"
    else:
        base = results_root.expanduser().resolve() / track
    return base / run_id


def _output_dir(
    track: str,
    run_id: str,
    model_seed: int,
    results_root: Path | None = None,
) -> Path:
    return _track_run_root(track, run_id, results_root) / f"model-seed-{model_seed}"


def _commands(args: argparse.Namespace, run_id: str) -> list[tuple[str, list[str], Path | None]]:
    commands: list[tuple[str, list[str], Path | None]] = []
    selected_tracks = _selected_tracks(args.tracks)
    if not args.prepare_only:
        preflight_output = PROJECT_ROOT / "runs" / "paper" / run_id / "gpu-preflight.json"
        preflight = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "gpu_preflight.py"),
            "--device",
            args.device,
            "--min-free-gb",
            str(args.min_free_gb),
            "--json-out",
            str(preflight_output),
        ]
        if args.suite in {"all", "benchmark"}:
            preflight.append("--require-paper-deps")
        commands.append(("gpu_preflight", preflight, preflight_output))
    brec_protocol = "official"

    data_root = args.data_root.expanduser().resolve()

    def cycle_arguments(suite: str) -> tuple[str, ...]:
        values = ["--variants", ",".join(args.cycle_variants)]
        if suite == "core":
            values.extend(("--core-targets", ",".join(args.cycle_core_targets)))
        # Official BREC owns its fixed 20-epoch/1e-4 optimization protocol.
        # Master tuning knobs apply only to CycleCount/ZINC.
        if suite != "brec":
            if args.cycle_epochs is not None:
                values.extend(("--epochs", str(args.cycle_epochs)))
            if args.cycle_learning_rate is not None:
                values.extend(("--learning-rate", str(args.cycle_learning_rate)))
        return tuple(values)

    def add_child(
        *,
        track: str,
        suite: str,
        model_seed: int,
        name: str,
        output_dir: Path,
        extra_arguments: tuple[str, ...] = (),
        batch_size: int | None = None,
        workers: int | None = None,
        amp: bool | None = None,
    ) -> None:
        default_batch_size = 2 if track == "conductance_gat" and suite == "benchmark" else 32
        requested_batch_size = (
            args.batch_size if args.batch_size is not None else default_batch_size
        )
        effective_batch_size = requested_batch_size if batch_size is None else batch_size
        effective_workers = args.workers if workers is None else workers
        requested_amp = args.amp if args.amp is not None else args.suite != "benchmark"
        effective_amp = requested_amp if amp is None else amp
        command = [
            sys.executable,
            "-m",
            BENCHMARK_MODULES[track] if suite == "benchmark" else TRACK_MODULES[track],
            "--suite",
            suite,
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu" if args.prepare_only else args.device,
            "--data-seed",
            str(args.data_seed),
            "--split-seed",
            str(args.split_seed),
            "--chart-seed",
            str(args.chart_seed),
            "--model-seed",
            str(model_seed),
            "--batch-size",
            str(effective_batch_size),
            "--workers",
            str(effective_workers),
        ]
        if args.prepare_only:
            command.append("--prepare-only")
        if args.allow_download:
            command.append("--allow-download")
        if not args.prepare_only:
            if effective_amp and args.device.lower().startswith("cuda"):
                command.append("--amp")
            elif not effective_amp or args.device.lower().startswith("cpu"):
                command.append("--no-amp")
        command.extend(extra_arguments)
        commands.append((name, command, output_dir))

    executed_model_seeds = args.model_seeds[:1] if args.prepare_only else args.model_seeds
    for track in selected_tracks:
        if args.suite == "benchmark":
            # Original-paper public datasets with our models only.  Tree
            # augmentation keeps its own fixed-vs-multi-chart comparison on
            # public CSL/ZINC; it remains an ablation of our own model.
            suites = ("csl", "zinc") if track == "tree_augmentation" else ("benchmark",)
            for model_seed in executed_model_seeds:
                for suite in suites:
                    add_child(
                        track=track,
                        suite=suite,
                        model_seed=model_seed,
                        name=f"{track}:{suite}:model-seed-{model_seed}",
                        output_dir=(
                            _output_dir(track, run_id, model_seed, args.results_root) / suite
                        ),
                    )
            continue

        # BREC already performs its official ten model-search seeds internally.
        # Under suite=all, run CycleCount and ZINC for every outer experiment
        # seed, but dispatch BREC exactly once rather than multiplying it by the
        # five default outer seeds.
        if track == "cycle_pe" and args.suite == "all":
            cycle_root = _track_run_root(track, run_id, args.results_root)
            for model_seed in executed_model_seeds:
                for suite in ("core", "zinc"):
                    add_child(
                        track=track,
                        suite=suite,
                        model_seed=model_seed,
                        name=f"{track}:{suite}:model-seed-{model_seed}",
                        output_dir=cycle_root / f"model-seed-{model_seed}" / suite,
                        extra_arguments=cycle_arguments(suite),
                    )
            brec_label = "official-10-seed"
            brec_run_name = f"brec-{brec_label}"
            add_child(
                track=track,
                suite="brec",
                model_seed=args.model_seeds[0],
                name=f"{track}:brec:{brec_label}",
                output_dir=cycle_root / brec_run_name,
                extra_arguments=(
                    *cycle_arguments("brec"),
                    "--brec-protocol",
                    brec_protocol,
                    "--brec-seeds",
                    ",".join(str(seed) for seed in CYCLE_BREC_OFFICIAL_SEEDS),
                ),
                batch_size=16,
                workers=0,
                amp=False,
            )
            continue

        # Dataset, split, and chart axes are fixed for a model-seed sweep.  A
        # prepare-only run therefore materializes each requested suite once.
        for model_seed in executed_model_seeds:
            extra_arguments = cycle_arguments(args.suite) if track == "cycle_pe" else ()
            add_child(
                track=track,
                suite=args.suite,
                model_seed=model_seed,
                name=f"{track}:model-seed-{model_seed}",
                output_dir=_output_dir(track, run_id, model_seed, args.results_root),
                extra_arguments=extra_arguments,
            )
    return commands


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> dict[str, Any]:
    if not (PROJECT_ROOT / ".git").exists():
        return {"git_available": False, "revision": None, "dirty": None}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_available": revision.returncode == 0,
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _environment_snapshot(path: Path) -> dict[str, Any]:
    distributions = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    atomic_write_bytes(path, ("\n".join(distributions) + "\n").encode("utf-8"))
    return {"path": str(path), "sha256": _sha256(path)}


def _snapshot_registries(run_dir: Path, tracks: tuple[str, ...]) -> dict[str, Any]:
    directory = run_dir / "dataset-registries"
    directory.mkdir(parents=True, exist_ok=False)
    snapshots: dict[str, Any] = {}
    for track in tracks:
        source = PROJECT_ROOT / "research" / track / "datasets.yaml"
        target = directory / f"{track}.yaml"
        shutil.copy2(source, target)
        snapshots[track] = {"path": str(target), "sha256": _sha256(target)}
    return snapshots


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _validate_json_outputs(path: Path) -> list[str]:
    if path.is_file() and path.suffix == ".json":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("*.json"))
    else:
        return [f"missing output: {path}"]
    if not candidates:
        return [f"no JSON artifact found under {path}"]
    errors: list[str] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid JSON {candidate}: {error}")
            continue
        if not _all_finite(payload):
            errors.append(f"non-finite numeric value in {candidate}")
    return errors


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload, sort_keys=False)


def _run_logged(command: list[str], *, log_path: Path) -> int:
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    child_environment["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            try:
                print(line, end="", flush=True)
            except UnicodeEncodeError:
                console_encoding = sys.stdout.encoding or "utf-8"
                safe_line = line.encode(console_encoding, errors="backslashreplace").decode(
                    console_encoding
                )
                print(safe_line, end="", flush=True)
            log.write(line)
        return process.wait()


def _stop_after_failure(name: str, *, fail_fast: bool) -> bool:
    """A shared preflight failure is fatal even when track failures are independent."""

    return name == "gpu_preflight" or fail_fast


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("all", *TRACK_MODULES),
        default=["all"],
    )
    parser.add_argument(
        "--suite",
        choices=("benchmark", "core", "all"),
        default="benchmark",
        help=(
            "benchmark: our models on track-specific public datasets (default); "
            "core/all: supplementary own-method studies"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", type=_run_id)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "paper")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="optional shared result root (useful for scratch storage on a GPU cluster)",
    )
    parser.add_argument(
        "--model-seeds",
        "--seeds",
        dest="model_seeds",
        type=_seeds,
        default=(0, 1, 2, 3, 4),
        help="model/minibatch seeds; --seeds is a compatibility alias",
    )
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--chart-seed", type=int, default=0)
    parser.add_argument(
        "--cycle-variants",
        type=_cycle_variants,
        default=DEFAULT_CYCLE_VARIANTS,
        help="supplementary Cycle PE variants; no_pe is an explicit optional ablation",
    )
    parser.add_argument(
        "--cycle-core-targets",
        type=_cycle_core_targets,
        default=CYCLE_CORE_TARGETS,
        help="comma-separated CycleCount target levels forwarded to cycle core runs",
    )
    parser.add_argument("--cycle-epochs", type=int)
    parser.add_argument("--cycle-learning-rate", type=float)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="override track batch size (default: PPI 2, molecular/tree graphs 32)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow official public datasets to be downloaded into --data-root",
    )
    failure = parser.add_mutually_exclusive_group()
    failure.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed track/seed (the default audits every independent run)",
    )
    failure.add_argument(
        "--continue-on-error",
        dest="fail_fast",
        action="store_false",
        help="deprecated compatibility alias for the default independent-run behavior",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override precision (benchmark defaults to float32; supplementary suites use AMP)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (args.batch_size is not None and args.batch_size < 1) or args.workers < 0:
        print("batch size must be positive and workers must be non-negative", file=sys.stderr)
        return 2
    if min(args.data_seed, args.split_seed, args.chart_seed) < 0:
        print("data, split, and chart seeds must be non-negative", file=sys.stderr)
        return 2
    if args.cycle_epochs is not None and args.cycle_epochs < 1:
        print("--cycle-epochs must be positive", file=sys.stderr)
        return 2
    if args.cycle_learning_rate is not None and args.cycle_learning_rate <= 0:
        print("--cycle-learning-rate must be positive", file=sys.stderr)
        return 2
    if not args.device.lower().startswith("cuda") and not args.prepare_only:
        print(
            "paper training requires CUDA; CPU is supported only for --prepare-only",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or _default_run_id()
    tracks = _selected_tracks(args.tracks)
    commands = _commands(args, run_id)
    if args.dry_run:
        for name, command, output in commands:
            print(f"[{name}] {_command_text(command)}")
            if output is not None:
                print(f"  output: {output}")
        return 0

    try:
        check_dependencies()
    except DependencyCheckError as error:
        print(error_message(error), file=sys.stderr)
        return 2

    run_dir = PROJECT_ROOT / "runs" / "paper" / run_id
    if run_dir.exists() or any(
        _track_run_root(track, run_id, args.results_root).exists() for track in tracks
    ):
        print(f"run id already exists: {run_id}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "scope": "independent_paper_tracks",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source": _source_revision(),
        "device_request": args.device,
        "suite": args.suite,
        "tracks": list(tracks),
        "data_root": str(args.data_root.expanduser().resolve()),
        "results_root": (
            str(args.results_root.expanduser().resolve()) if args.results_root is not None else None
        ),
        "seed_axes": {
            "data": args.data_seed,
            "split": args.split_seed,
            "chart": args.chart_seed,
            "model": list(args.model_seeds),
        },
        "requested_model_seeds": list(args.model_seeds),
        "executed_model_seeds": ([] if args.prepare_only else list(args.model_seeds)),
        "execution_protocol": {
            "outer_model_seeds": list(args.model_seeds),
            "prepare_once_for_fixed_non_model_axes": args.prepare_only,
            "cycle_selection": (
                {
                    "variants": list(args.cycle_variants),
                    "core_targets": list(args.cycle_core_targets),
                    "epochs_override": args.cycle_epochs,
                    "learning_rate_override": args.cycle_learning_rate,
                    "official_brec_optimization_overrides_ignored": True,
                }
                if "cycle_pe" in tracks and args.suite != "benchmark"
                else None
            ),
            "comparison_protocol": (
                "our_models_only_on_track_specific_public_datasets"
                if args.suite == "benchmark"
                else "supplementary_research_suites"
            ),
            "gpu_preflight": None
            if args.prepare_only
            else {
                "kind": "hardware_and_dependency_check",
                "min_free_gb": args.min_free_gb,
                "dataset_loaded": False,
                "model_executed": False,
            },
            "cycle_brec_internal_seeds": (
                list(CYCLE_BREC_OFFICIAL_SEEDS)
                if args.suite == "all" and "cycle_pe" in tracks
                else None
            ),
            "cycle_brec_dispatch_count": (1 if args.suite == "all" and "cycle_pe" in tracks else 0),
            "cycle_brec_protocol": (
                "official" if args.suite == "all" and "cycle_pe" in tracks else None
            ),
            "cycle_brec_training": (
                {
                    "batch_size": 16,
                    "workers": 0,
                    "amp": False,
                }
                if args.suite == "all" and "cycle_pe" in tracks
                else None
            ),
        },
        "prepare_only": args.prepare_only,
        "environment": _environment_snapshot(run_dir / "environment.txt"),
        "dataset_registries": _snapshot_registries(run_dir, tracks),
        "commands": [],
    }
    _write_manifest(manifest_path, manifest)

    failed = False
    for index, (name, command, output) in enumerate(commands):
        print(f"\n== {name}: {_command_text(command)} ==", flush=True)
        safe_name = name.replace(":", "-")
        log_path = logs_dir / f"{index:02d}-{safe_name}.log"
        started = datetime.now(UTC)
        return_code = _run_logged(command, log_path=log_path)
        finished = datetime.now(UTC)
        errors: list[str] = []
        if return_code == 0 and output is not None:
            errors = _validate_json_outputs(output)
        entry = {
            "name": name,
            "command": command,
            "returncode": return_code,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "log": str(log_path),
            "output": str(output) if output is not None else None,
            "artifact_errors": errors,
        }
        manifest["commands"].append(entry)
        _write_manifest(manifest_path, manifest)
        if return_code != 0 or errors:
            failed = True
            # Every child relies on the same accelerator/dependency preflight.
            # Independent-track continuation only makes sense after that
            # shared prerequisite has passed.
            if _stop_after_failure(name, fail_fast=args.fail_fast):
                break

    manifest["status"] = "failed" if failed else "passed"
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_manifest(manifest_path, manifest)
    if args.prepare_only:
        manifest["aggregation"] = {
            "status": "skipped",
            "reason": "prepare-only runs contain no model-seed metrics",
        }
    else:
        try:
            aggregate = aggregate_manifest(manifest_path)
            manifest["aggregation"] = {
                "status": "passed",
                "path": str(manifest_path.parent / "aggregate" / "aggregate.json"),
                **aggregate,
            }
        except Exception as error:  # Preserve the completed child-run audit trail.
            failed = True
            manifest["status"] = "failed"
            manifest["aggregation"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    _write_manifest(manifest_path, manifest)
    if failed:
        print(f"paper run failed; inspect {manifest_path}", file=sys.stderr)
        return 1
    print(f"all requested independent paper tracks passed; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
