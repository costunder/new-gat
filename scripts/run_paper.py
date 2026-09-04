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
from chartgat.execution import add_execution_arguments

try:
    from scripts.aggregate_paper import aggregate_manifest
    from scripts.check_dependencies import DependencyCheckError, check_dependencies, error_message
    from scripts.process_safety import (
        close_owned_child_stdout,
        terminate_owned_child_after_error,
    )
    from scripts.telemetry_validation import (
        validate_resource_observability,
        validate_throughput_observability,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_paper.py`` execution.
    from aggregate_paper import aggregate_manifest
    from check_dependencies import DependencyCheckError, check_dependencies, error_message
    from process_safety import close_owned_child_stdout, terminate_owned_child_after_error
    from telemetry_validation import (
        validate_resource_observability,
        validate_throughput_observability,
    )

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
CYCLE_BASIS_BACKENDS = ("thin_q", "dfs_fundamental")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".sh", ".ps1"}
SOURCE_ROOTS = ("scripts", "src", "research")


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


def _track_run_root(
    track: str,
    run_id: str,
    results_root: Path | None = None,
    *,
    cycle_pe_version: str = "v1",
) -> Path:
    is_cycle_v2 = track == "cycle_pe" and cycle_pe_version == "v2"
    if results_root is None:
        track_path = PROJECT_ROOT / "research" / track
        if is_cycle_v2:
            track_path = track_path / "v2"
        base = track_path / "results" / "paper"
    else:
        base = results_root.expanduser().resolve() / ("cycle_pe_v2" if is_cycle_v2 else track)
    return base / run_id


def _output_dir(
    track: str,
    run_id: str,
    model_seed: int,
    results_root: Path | None = None,
    *,
    cycle_pe_version: str = "v1",
) -> Path:
    return (
        _track_run_root(track, run_id, results_root, cycle_pe_version=cycle_pe_version)
        / f"model-seed-{model_seed}"
    )


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
        module = BENCHMARK_MODULES[track] if suite == "benchmark" else TRACK_MODULES[track]
        if track == "cycle_pe" and args.cycle_pe_version == "v2":
            module = "research.cycle_pe.v2.benchmark"
        command = [
            sys.executable,
            "-m",
            module,
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
            if args.compile and suite == "benchmark":
                command.append("--compile")
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
                    cycle_v2 = track == "cycle_pe" and args.cycle_pe_version == "v2"
                    label = "benchmark-v2" if cycle_v2 else suite
                    overrides: list[str] = []
                    if cycle_v2:
                        overrides.extend(
                            (
                                "--basis-backend",
                                args.basis_backend,
                                "--basis-execution",
                                args.basis_execution,
                                "--basis-pair-budget",
                                str(args.basis_pair_budget),
                            )
                        )
                        if args.cycle_epochs is not None:
                            overrides.extend(("--epochs", str(args.cycle_epochs)))
                        if args.cycle_learning_rate is not None:
                            overrides.extend(("--lr", str(args.cycle_learning_rate)))
                    add_child(
                        track=track,
                        suite=suite,
                        model_seed=model_seed,
                        name=f"{track}:{label}:model-seed-{model_seed}",
                        output_dir=(
                            _output_dir(
                                track,
                                run_id,
                                model_seed,
                                args.results_root,
                                cycle_pe_version=args.cycle_pe_version,
                            )
                            / suite
                        ),
                        extra_arguments=tuple(overrides),
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
    ignored_parts = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "data",
        "results",
        "tests",
    }
    source_paths = [
        path
        for root_name in SOURCE_ROOTS
        for path in (PROJECT_ROOT / root_name).rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and not any(part in ignored_parts for part in path.relative_to(PROJECT_ROOT).parts)
    ]
    for name in ("pyproject.toml", "environment.yml"):
        path = PROJECT_ROOT / name
        if path.is_file():
            source_paths.append(path)
    source_sha256 = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in sorted(set(source_paths))
    }
    if not (PROJECT_ROOT / ".git").exists():
        return {
            "git_available": False,
            "revision": None,
            "dirty": None,
            "source_sha256": source_sha256,
        }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "git_available": revision.returncode == 0,
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "source_sha256": source_sha256,
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


def _snapshot_registries(
    run_dir: Path, tracks: tuple[str, ...], *, cycle_pe_version: str = "v1"
) -> dict[str, Any]:
    directory = run_dir / "dataset-registries"
    directory.mkdir(parents=True, exist_ok=False)
    snapshots: dict[str, Any] = {}
    for track in tracks:
        source_root = PROJECT_ROOT / "research" / track
        if track == "cycle_pe" and cycle_pe_version == "v2":
            source_root = source_root / "v2"
        source = source_root / "datasets.yaml"
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


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _canonical_payloads(path: Path) -> dict[Path, dict[str, Any]]:
    names = {
        "manifest.json",
        "run_manifest.json",
        "metrics.json",
        "summary.json",
        "prepare_summary.json",
        "runtime.json",
    }
    candidates = (
        [path]
        if path.is_file()
        else sorted(candidate for candidate in path.rglob("*.json") if candidate.name in names)
    )
    return {candidate: _json_object(candidate) for candidate in candidates}


def _walk_named(value: Any, names: set[str], label: str = "root") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if key in names:
                found.append((child_label, child))
            found.extend(_walk_named(child, names, child_label))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_named(child, names, f"{label}[{index}]"))
    return found


def _module_name(command: list[str]) -> str | None:
    try:
        module_index = command.index("-m")
    except ValueError:
        return None
    return command[module_index + 1] if module_index + 1 < len(command) else None


def _flag(command: list[str], name: str) -> str | None:
    try:
        index = command.index(name)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _validate_child_status(
    name: str,
    command: list[str],
    output: Path,
    payloads: dict[Path, dict[str, Any]],
    *,
    prepare_only: bool,
) -> list[str]:
    expected = "prepared" if prepare_only else "passed"
    module = _module_name(command)
    errors: list[str] = []
    if name == "gpu_preflight":
        payload = payloads.get(output.resolve()) or next(iter(payloads.values()), {})
        if payload.get("status") != "passed":
            errors.append("GPU preflight artifact must have status=passed")
        if payload.get("kind") != "hardware_and_dependency_check":
            errors.append("GPU preflight kind/provenance marker is missing")
        if payload.get("requested_device") != _flag(command, "--device"):
            errors.append("GPU preflight requested-device binding differs from the command")
        return errors

    if module == "research.conductance_gat.paper":
        summary_name = "prepare_summary.json" if prepare_only else "summary.json"
        summary = payloads.get((output / summary_name).resolve(), {})
        if summary.get("status") != expected:
            errors.append(f"conductance paper summary must have status={expected}")
        if summary.get("suite") != _flag(command, "--suite"):
            errors.append("conductance paper suite differs from the command")
    elif module == "research.cycle_pe.paper":
        run_manifest = payloads.get((output / "run_manifest.json").resolve(), {})
        if run_manifest.get("status") != "complete":
            errors.append("cycle paper run_manifest must have status=complete")
        selected_suite = _flag(command, "--suite")
        expected_suites = [selected_suite] if selected_suite != "all" else ["core", "brec", "zinc"]
        if run_manifest.get("selected_suites") != expected_suites:
            errors.append("cycle paper selected suites differ from the command")
        if run_manifest.get("completed_suites") != expected_suites:
            errors.append("cycle paper did not complete every requested suite")
    else:
        manifest = payloads.get((output / "manifest.json").resolve(), {})
        if manifest.get("status") != expected:
            errors.append(f"child manifest must have status={expected}")
        metrics_path = (output / "metrics.json").resolve()
        if metrics_path in payloads and payloads[metrics_path].get("status") != expected:
            errors.append(f"child metrics must have status={expected}")
    return errors


def _source_path_candidates(key: str, module: str | None) -> tuple[Path, ...]:
    normalized = Path(key)
    module_dir = (
        PROJECT_ROOT.joinpath(*module.split(".")).parent if module is not None else PROJECT_ROOT
    )
    candidates = [PROJECT_ROOT / normalized, module_dir / normalized]
    if len(normalized.parts) == 1:
        candidates.extend(
            (
                PROJECT_ROOT / "src" / "chartgat" / normalized,
                PROJECT_ROOT / "scripts" / normalized,
            )
        )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _validate_child_provenance(
    command: list[str], payloads: dict[Path, dict[str, Any]]
) -> list[str]:
    mappings: list[tuple[str, Any]] = []
    for path, payload in payloads.items():
        mappings.extend(
            _walk_named(
                payload,
                {"implementation_sha256", "source_files"},
                path.as_posix(),
            )
        )
        for label, integrity in _walk_named(payload, {"implementation_integrity"}, path.as_posix()):
            if isinstance(integrity, dict) and isinstance(integrity.get("sha256"), dict):
                mappings.append((f"{label}.sha256", integrity["sha256"]))
    if not mappings:
        return ["child has no implementation SHA-256 provenance"]
    errors: list[str] = []
    resolved_sources = 0
    module = _module_name(command)
    for label, mapping in mappings:
        if not isinstance(mapping, dict) or not mapping:
            errors.append(f"{label} must be a nonempty object")
            continue
        for key, digest in mapping.items():
            if (
                not isinstance(key, str)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                errors.append(f"{label} contains invalid source hash metadata")
                continue
            existing = next(
                (path for path in _source_path_candidates(key, module) if path.is_file()), None
            )
            if existing is None:
                continue
            resolved_sources += 1
            if _sha256(existing) != digest:
                errors.append(f"{label}.{key} does not match the current source")
    if resolved_sources == 0:
        errors.append("child source provenance does not resolve to any current source file")
    return errors


def _finite_rates(value: Any, label: str) -> list[float]:
    rates: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if (
                "_per_second" in key
                and isinstance(child, (int, float))
                and not isinstance(child, bool)
            ):
                number = float(child)
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"{child_label} must be finite and non-negative")
                rates.append(number)
            else:
                rates.extend(_finite_rates(child, child_label))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rates.extend(_finite_rates(child, f"{label}[{index}]"))
    return rates


def _validate_child_telemetry(payloads: dict[Path, dict[str, Any]]) -> list[str]:
    resources: list[tuple[str, Any]] = []
    throughput: list[tuple[str, Any]] = []
    for path, payload in payloads.items():
        resources.extend(
            _walk_named(
                payload,
                {"resource_observability", "evaluation_resource_observability"},
                path.as_posix(),
            )
        )
        throughput.extend(
            _walk_named(payload, {"throughput", "training_throughput"}, path.as_posix())
        )
    errors: list[str] = []
    if not resources:
        errors.append("executed child has no periodic GPU/CPU/RAM resource_observability")
    for label, value in resources:
        try:
            validate_resource_observability(value, label)
        except ValueError as error:
            errors.append(str(error))
    valid_throughput = 0
    for label, value in throughput:
        try:
            if isinstance(value, dict) and "scope" in value:
                validate_throughput_observability(value, label)
                valid_throughput += 1
            elif _finite_rates(value, label):
                valid_throughput += 1
        except ValueError as error:
            errors.append(str(error))
    if valid_throughput == 0:
        errors.append("executed child has no measured *_per_second throughput telemetry")
    return errors


def _validate_completed_output(
    name: str,
    command: list[str],
    output: Path | None,
    *,
    prepare_only: bool,
) -> list[str]:
    if output is None:
        return [f"{name}: command has no declared output"]
    output = output.resolve()
    errors = _validate_json_outputs(output)
    if errors:
        return errors
    try:
        payloads = _canonical_payloads(output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"invalid canonical child artifact: {type(error).__name__}: {error}"]
    errors.extend(
        _validate_child_status(name, command, output, payloads, prepare_only=prepare_only)
    )
    if name != "gpu_preflight":
        errors.extend(_validate_child_provenance(command, payloads))
        if not prepare_only:
            errors.extend(_validate_child_telemetry(payloads))
    return errors


def _output_sha256(path: Path) -> str:
    path = path.resolve()
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    )
    if not files:
        raise ValueError(f"cannot hash missing/empty output: {path}")
    for candidate in files:
        relative = candidate.name if path.is_file() else candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(candidate)))
    return digest.hexdigest()


def _assert_source_hashes_unchanged(expected: dict[str, str]) -> None:
    current = _source_revision().get("source_sha256")
    if not isinstance(current, dict) or current != expected:
        changed = sorted(
            name
            for name in set(expected) | set(current or {})
            if expected.get(name) != (current or {}).get(name)
        )
        raise RuntimeError(
            "paper runtime source changed after the immutable run snapshot: " + ", ".join(changed)
        )


def _aggregation_inputs(manifest: dict[str, Any]) -> dict[str, str | None]:
    return {
        str(entry.get("name")): (
            str(entry["accepted_output_sha256"])
            if isinstance(entry, dict) and isinstance(entry.get("accepted_output_sha256"), str)
            else None
        )
        for entry in manifest.get("commands", [])
        if isinstance(entry, dict)
    }


def _validate_accepted_aggregation(
    aggregation: dict[str, Any], aggregate_dir: Path, inputs: dict[str, str | None]
) -> list[str]:
    errors: list[str] = []
    required = {
        "aggregate.json",
        "samples.csv",
        "metrics.csv",
        "paired.csv",
        "efficiency.csv",
        "failures.csv",
    }
    if aggregation.get("path") != str((aggregate_dir / "aggregate.json").resolve()):
        errors.append("accepted aggregation path differs from the exact run output")
    missing = sorted(name for name in required if not (aggregate_dir / name).is_file())
    if missing:
        errors.append(f"accepted aggregation is missing required artifacts: {missing}")
    errors.extend(_validate_json_outputs(aggregate_dir))
    if not errors:
        actual = _output_sha256(aggregate_dir)
        if aggregation.get("accepted_output_sha256") != actual:
            errors.append("accepted aggregate output SHA-256 changed after completion")
    if aggregation.get("input_child_sha256") != inputs:
        errors.append("accepted aggregation input-child binding differs from current outputs")
    return errors


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _run_configuration(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    excluded = {"dry_run", "resume", "run_id"}
    return {
        "run_id": run_id,
        **{key: _json_safe(value) for key, value in vars(args).items() if key not in excluded},
    }


def _command_plan(commands: list[tuple[str, list[str], Path | None]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "command": list(command),
            "output": str(output.resolve()) if output is not None else None,
        }
        for name, command, output in commands
    ]


def _persist_manifest_after_error(
    path: Path, manifest: dict[str, Any], original_error: BaseException
) -> None:
    try:
        _write_manifest(path, manifest)
    except BaseException as reporting_error:
        original_error.add_note(
            "paper failure manifest could not be written without replacing the original error: "
            f"{type(reporting_error).__name__}: {reporting_error}"
        )


def _quarantine_output(path: Path, *, attempt: int) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return resolved
    suffix = max(1, attempt)
    while True:
        target = resolved.with_name(f"{resolved.name}.incomplete-attempt-{suffix}")
        if not target.exists():
            resolved.replace(target)
            return target
        suffix += 1


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload, sort_keys=False)


def _run_logged(command: list[str], *, log_path: Path) -> int:
    child_environment = os.environ.copy()
    child_environment.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
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
        primary_error: BaseException | None = None
        try:
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
                log.flush()
            return process.wait()
        except BaseException as error:
            primary_error = error
            terminate_owned_child_after_error(
                process,
                command,
                original_error=error,
                log_target=log,
            )
            raise
        finally:
            close_owned_child_stdout(process, original_error=primary_error)


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
        default=(0,),
        help=(
            "model/minibatch seeds (default: 0); pass a comma-separated list for a sweep; "
            "--seeds is a compatibility alias"
        ),
    )
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--chart-seed", type=int, default=0)
    parser.add_argument(
        "--cycle-pe-version",
        choices=("v1", "v2"),
        default="v1",
        help="v2: full left-nullspace basis; select --tracks cycle_pe --suite benchmark",
    )
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
    parser.add_argument("--basis-backend", choices=CYCLE_BASIS_BACKENDS, default="thin_q")
    parser.add_argument("--basis-execution", choices=("batched", "reference"), default="batched")
    parser.add_argument("--basis-pair-budget", type=int, default=32768)
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
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "resume the same run id after exact configuration/source/command-plan validation; "
            "completed child outputs are hash-verified and never rerun"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override precision (benchmark defaults to float32; supplementary suites use AMP)",
    )
    add_execution_arguments(parser)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.compile and (
        args.suite != "benchmark" or "tree_augmentation" in _selected_tracks(args.tracks)
    ):
        print("--compile supports conductance_gat/cycle_pe benchmark tracks only", file=sys.stderr)
        return 2
    if args.basis_pair_budget < 1:
        print("--basis-pair-budget must be positive", file=sys.stderr)
        return 2
    if args.basis_backend != "thin_q" and args.cycle_pe_version != "v2":
        print("nondefault --basis-backend requires --cycle-pe-version v2", file=sys.stderr)
        return 2
    if args.cycle_pe_version == "v2" and (
        args.suite != "benchmark" or _selected_tracks(args.tracks) != ("cycle_pe",)
    ):
        print(
            "Cycle PE v2 is independent: use --tracks cycle_pe --suite benchmark",
            file=sys.stderr,
        )
        return 2
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
        dependency_report = check_dependencies()
    except DependencyCheckError as error:
        print(error_message(error), file=sys.stderr)
        return error.exit_code

    run_dir = PROJECT_ROOT / "runs" / "paper" / run_id
    manifest_path = run_dir / "manifest.json"
    source = _source_revision()
    command_plan = _command_plan(commands)
    runtime_environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "research_environment": {
            key: value
            for key, value in dependency_report.items()
            if key != "pytorch_nvml_based_cuda_check_removed"
        },
    }
    resume_identity = {
        "schema": "paper-parent-resume-v1",
        "run_configuration": _run_configuration(args, run_id),
        "source": source,
        "runtime_environment": runtime_environment,
        "command_plan": command_plan,
    }
    track_roots = [
        _track_run_root(track, run_id, args.results_root, cycle_pe_version=args.cycle_pe_version)
        for track in tracks
    ]
    if run_dir.exists():
        if not args.resume:
            print(f"run id already exists and --no-resume was requested: {run_id}", file=sys.stderr)
            return 2
        try:
            manifest = _json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"existing run has no valid resumable manifest: {error}", file=sys.stderr)
            return 2
        if manifest.get("resume_identity") != resume_identity:
            print(
                "existing run cannot be resumed: immutable configuration, source, or command "
                "plan differs",
                file=sys.stderr,
            )
            return 2
        if manifest.get("status") not in {"running", "failed", "passed"}:
            print("existing run cannot be resumed: invalid manifest status", file=sys.stderr)
            return 2
        stored_commands = manifest.get("commands")
        if not isinstance(stored_commands, list) or len(stored_commands) != len(command_plan):
            print("existing run cannot be resumed: command records differ", file=sys.stderr)
            return 2
        for planned, stored in zip(command_plan, stored_commands, strict=True):
            if not isinstance(stored, dict) or any(
                stored.get(key) != planned[key] for key in ("name", "command", "output")
            ):
                print("existing run cannot be resumed: command records differ", file=sys.stderr)
                return 2
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["resumed_at_utc"] = datetime.now(UTC).isoformat()
    else:
        orphaned = [str(path) for path in track_roots if path.exists()]
        if orphaned:
            print(
                "run id has child outputs but no parent resume manifest: " + ", ".join(orphaned),
                file=sys.stderr,
            )
            return 2
        run_dir.mkdir(parents=True)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir()
        manifest = {
            "schema_version": 2,
            "scope": "independent_paper_tracks",
            "run_id": run_id,
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "python": runtime_environment["python"],
            "platform": runtime_environment["platform"],
            "source": source,
            "device_request": args.device,
            "suite": args.suite,
            "tracks": list(tracks),
            "data_root": str(args.data_root.expanduser().resolve()),
            "results_root": (
                str(args.results_root.expanduser().resolve())
                if args.results_root is not None
                else None
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
                "torch_compile": args.compile and not args.prepare_only,
                "basis_backend": args.basis_backend if args.cycle_pe_version == "v2" else None,
                "basis_execution": args.basis_execution if args.cycle_pe_version == "v2" else None,
                "basis_pair_budget": (
                    args.basis_pair_budget if args.cycle_pe_version == "v2" else None
                ),
                "cycle_pe_version": args.cycle_pe_version if "cycle_pe" in tracks else None,
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
                "cycle_brec_dispatch_count": (
                    1 if args.suite == "all" and "cycle_pe" in tracks else 0
                ),
                "cycle_brec_protocol": (
                    "official" if args.suite == "all" and "cycle_pe" in tracks else None
                ),
                "cycle_brec_training": (
                    {"batch_size": 16, "workers": 0, "amp": False}
                    if args.suite == "all" and "cycle_pe" in tracks
                    else None
                ),
            },
            "prepare_only": args.prepare_only,
            "environment": _environment_snapshot(run_dir / "environment.txt"),
            "research_environment": dependency_report,
            "dataset_registries": _snapshot_registries(
                run_dir, tracks, cycle_pe_version=args.cycle_pe_version
            ),
            "resume_identity": resume_identity,
            "resume_count": 0,
            "commands": [
                {
                    **planned,
                    "status": "pending",
                    "returncode": None,
                    "artifact_errors": [],
                    "attempts": [],
                }
                for planned in command_plan
            ],
        }
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest["status"] = "running"
    manifest.pop("finished_at_utc", None)
    manifest.pop("error", None)
    _write_manifest(manifest_path, manifest)

    failed = False
    for index, (name, command, output) in enumerate(commands):
        try:
            _assert_source_hashes_unchanged(source["source_sha256"])
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
            _persist_manifest_after_error(manifest_path, manifest, error)
            raise
        entry = manifest["commands"][index]
        assert entry["name"] == name and entry["command"] == command
        if entry.get("status") == "passed":
            errors = _validate_completed_output(
                name, command, output, prepare_only=args.prepare_only
            )
            if output is not None and not errors:
                digest = _output_sha256(output)
                if entry.get("accepted_output_sha256") != digest:
                    errors.append("accepted child output SHA-256 changed after completion")
            if errors:
                entry["resume_artifact_errors"] = errors
                manifest["status"] = "failed"
                manifest["error"] = "previously accepted child output failed resume validation"
                _write_manifest(manifest_path, manifest)
                print(
                    f"refusing to overwrite previously accepted child {name}: {errors}",
                    file=sys.stderr,
                )
                return 2
            entry["resume_validation"] = "passed_and_skipped"
            continue

        can_recover = (
            entry.get("returncode") in (None, 0) and output is not None and output.exists()
        )
        if can_recover:
            recovery_errors = _validate_completed_output(
                name, command, output, prepare_only=args.prepare_only
            )
            if not recovery_errors:
                entry.update(
                    {
                        "status": "passed",
                        "returncode": 0,
                        "artifact_errors": [],
                        "accepted_output_sha256": _output_sha256(output),
                        "recovered_at_utc": datetime.now(UTC).isoformat(),
                    }
                )
                _write_manifest(manifest_path, manifest)
                print(
                    f"\n== {name}: recovered complete child output; skipping rerun ==", flush=True
                )
                continue
            entry.setdefault("resume_artifact_errors", []).extend(recovery_errors)

        if output is not None and output.exists():
            quarantined = _quarantine_output(output, attempt=len(entry.get("attempts", [])) + 1)
            entry.setdefault("preserved_incomplete_outputs", []).append(str(quarantined))

        print(f"\n== {name}: {_command_text(command)} ==", flush=True)
        safe_name = name.replace(":", "-")
        attempt_number = len(entry.get("attempts", [])) + 1
        log_path = logs_dir / f"{index:02d}-{safe_name}-attempt-{attempt_number}.log"
        started = datetime.now(UTC)
        attempt = {
            "attempt": attempt_number,
            "status": "running",
            "started_at_utc": started.isoformat(),
            "log": str(log_path),
        }
        entry.setdefault("attempts", []).append(attempt)
        entry.update(
            {"status": "running", "started_at_utc": started.isoformat(), "log": str(log_path)}
        )
        _write_manifest(manifest_path, manifest)
        try:
            return_code = _run_logged(command, log_path=log_path)
        except BaseException as error:
            finished = datetime.now(UTC)
            attempt.update(
                {
                    "status": "failed",
                    "finished_at_utc": finished.isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            entry.update(
                {
                    "status": "failed",
                    "returncode": None,
                    "finished_at_utc": finished.isoformat(),
                    "error": attempt["error"],
                }
            )
            manifest["status"] = "failed"
            manifest["error"] = attempt["error"]
            manifest["finished_at_utc"] = finished.isoformat()
            _persist_manifest_after_error(manifest_path, manifest, error)
            raise
        finished = datetime.now(UTC)
        errors: list[str] = []
        if return_code == 0 and output is not None:
            errors = _validate_completed_output(
                name, command, output, prepare_only=args.prepare_only
            )
        status = "passed" if return_code == 0 and not errors else "failed"
        attempt.update(
            {
                "status": status,
                "returncode": return_code,
                "finished_at_utc": finished.isoformat(),
                "elapsed_seconds": (finished - started).total_seconds(),
                "artifact_errors": errors,
            }
        )
        entry.update(attempt)
        entry["artifact_errors"] = errors
        if status == "passed" and output is not None:
            entry["accepted_output_sha256"] = _output_sha256(output)
        _write_manifest(manifest_path, manifest)
        if status == "failed":
            failed = True
            # Every child relies on the same accelerator/dependency preflight.
            # Independent-track continuation only makes sense after that
            # shared prerequisite has passed.
            if _stop_after_failure(name, fail_fast=args.fail_fast):
                break

    failed = failed or any(entry.get("status") != "passed" for entry in manifest["commands"])
    manifest["status"] = "failed" if failed else "passed"
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_manifest(manifest_path, manifest)
    try:
        _assert_source_hashes_unchanged(source["source_sha256"])
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        _persist_manifest_after_error(manifest_path, manifest, error)
        raise
    if args.prepare_only:
        manifest["aggregation"] = {
            "status": "skipped",
            "reason": "prepare-only runs contain no model-seed metrics",
        }
    else:
        aggregate_dir = manifest_path.parent / "aggregate"
        inputs = _aggregation_inputs(manifest)
        previous_aggregation = manifest.get("aggregation")
        skip_aggregation = False
        if (
            isinstance(previous_aggregation, dict)
            and previous_aggregation.get("status") == "passed"
        ):
            aggregation_errors = _validate_accepted_aggregation(
                previous_aggregation, aggregate_dir, inputs
            )
            only_input_changed = aggregation_errors == [
                "accepted aggregation input-child binding differs from current outputs"
            ]
            if not aggregation_errors:
                previous_aggregation["resume_validation"] = "passed_and_skipped"
                skip_aggregation = True
            elif not only_input_changed:
                manifest["status"] = "failed"
                manifest["aggregation_resume_errors"] = aggregation_errors
                _write_manifest(manifest_path, manifest)
                print(
                    "refusing to overwrite a previously accepted aggregation: "
                    + "; ".join(aggregation_errors),
                    file=sys.stderr,
                )
                return 2
        if not skip_aggregation:
            if aggregate_dir.exists():
                preserved = _quarantine_output(
                    aggregate_dir, attempt=int(manifest.get("resume_count", 0)) + 1
                )
                manifest.setdefault("preserved_aggregate_outputs", []).append(str(preserved))
                _write_manifest(manifest_path, manifest)
            try:
                aggregate = aggregate_manifest(manifest_path)
                manifest["aggregation"] = {
                    "status": "passed",
                    "path": str((aggregate_dir / "aggregate.json").resolve()),
                    "input_child_sha256": inputs,
                    "accepted_output_sha256": _output_sha256(aggregate_dir),
                    **aggregate,
                }
            except BaseException as error:  # Preserve the completed child-run audit trail.
                failed = True
                manifest["status"] = "failed"
                manifest["aggregation"] = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
                _persist_manifest_after_error(manifest_path, manifest, error)
                raise
    _write_manifest(manifest_path, manifest)
    if failed:
        print(f"paper run failed; inspect {manifest_path}", file=sys.stderr)
        return 1
    print(f"all requested independent paper tracks passed; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
