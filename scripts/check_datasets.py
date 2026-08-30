#!/usr/bin/env python3
"""Validate code readiness and separately report dataset-cache availability."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from chartgat.cache import (
    CacheCorruptError,
    CacheIncompleteError,
    CacheValidationError,
    CacheWrongRequestError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS = ("conductance_gat", "cycle_pe", "tree_augmentation")
REGISTRY_VERSION = 2
REQUIRED_ENTRY_FIELDS = {
    "id",
    "name",
    "tier",
    "status",
    "data_policy",
    "source_url",
    "task",
    "split",
    "metrics",
    "claim",
    "adapter",
    "leakage_guard",
}
ALLOWED_TIERS = {"paper_core", "conditional", "optional"}
# ``status`` is code readiness. Dataset optionality belongs in ``tier``.
ALLOWED_STATUSES = {"implemented", "planned", "blocked"}
ALLOWED_DATA_POLICIES = {"generated", "download", "manual", "none"}


def load_registry(track: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "research" / track / "datasets.yaml"
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict):
        raise ValueError(f"{path}: registry root must be a mapping")
    registry["_path"] = str(path)
    return registry


def _load_python_reference(reference: str) -> Any:
    """Resolve and return a dotted module attribute."""

    pieces = reference.split(".")
    for stop in range(len(pieces), 0, -1):
        module_name = ".".join(pieces[:stop])
        try:
            specification = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            specification = None
        if specification is None:
            continue
        attributes = pieces[stop:]
        if not attributes:
            return importlib.import_module(module_name)
        try:
            value: Any = importlib.import_module(module_name)
            for attribute in attributes:
                value = getattr(value, attribute)
        except (ImportError, AttributeError, ModuleNotFoundError) as error:
            raise ImportError(f"cannot resolve Python reference {reference!r}: {error}") from error
        return value
    raise ImportError(f"cannot find Python module for reference {reference!r}")


def _resolve_python_reference(reference: str) -> str | None:
    """Return an error when a dotted module/callable cannot be resolved."""

    try:
        _load_python_reference(reference)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as error:
        return str(error)
    return None


def _validate_adapter(adapter: Any) -> str | None:
    if not isinstance(adapter, str) or not adapter.strip():
        return "adapter must be a non-empty string"
    normalized = adapter.strip()
    if normalized.lower().startswith(("planned", "requires")):
        return f"implemented adapter cannot be prose: {adapter!r}"
    if normalized.startswith("python -m "):
        try:
            tokens = shlex.split(normalized)
        except ValueError as error:
            return f"invalid adapter command: {error}"
        if len(tokens) < 3 or tokens[:2] != ["python", "-m"]:
            return f"invalid Python module command {adapter!r}"
        error = _resolve_python_reference(tokens[2])
        if error is not None:
            return error
        if "--suite" in tokens:
            suite_index = tokens.index("--suite")
            if suite_index + 1 >= len(tokens) or tokens[suite_index + 1].startswith("-"):
                return f"adapter command has no --suite value: {adapter!r}"
        return None
    if any(character.isspace() for character in normalized):
        return f"adapter is neither a dotted reference nor python -m command: {adapter!r}"
    return _resolve_python_reference(normalized)


def _validate_source(source: Any) -> str | None:
    if not isinstance(source, str) or not source.strip():
        return "source_url must be a non-empty string"
    if source.startswith("generated://"):
        module_name = source.removeprefix("generated://").split("/", 1)[0]
        error = _resolve_python_reference(module_name)
        return None if error is None else f"invalid generated source: {error}"
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return None
    return f"source_url must be generated:// or an HTTP(S) URL: {source!r}"


def _validate_cache_glob(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "cache_glob must be a non-empty relative string"
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return "cache_glob must remain under --data-root"
    return None


def validate_registry(track: str, registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("registry_version") != REGISTRY_VERSION:
        errors.append(f"{track}: registry_version must be {REGISTRY_VERSION}")
    if registry.get("track") != track:
        errors.append(f"{track}: registry track field does not match")
    if not isinstance(registry.get("paper_suite_complete"), bool):
        errors.append(f"{track}: paper_suite_complete must be boolean")
    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return [*errors, f"{track}: datasets must be a non-empty list"]

    identifiers: set[str] = set()
    for index, entry in enumerate(datasets):
        label = f"{track}.datasets[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry must be a mapping")
            continue
        missing = sorted(REQUIRED_ENTRY_FIELDS - entry.keys())
        if missing:
            errors.append(f"{label}: missing fields {missing}")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"{label}: id must be a non-empty string")
        elif identifier in identifiers:
            errors.append(f"{label}: duplicate id {identifier!r}")
        else:
            identifiers.add(identifier)
        if entry.get("tier") not in ALLOWED_TIERS:
            errors.append(f"{label}: invalid tier {entry.get('tier')!r}")
        status = entry.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{label}: invalid status {status!r}")
        data_policy = entry.get("data_policy")
        if data_policy not in ALLOWED_DATA_POLICIES:
            errors.append(f"{label}: invalid data_policy {data_policy!r}")
        if status == "implemented" and data_policy == "none":
            errors.append(f"{label}: implemented code cannot use data_policy 'none'")
        metrics = entry.get("metrics")
        if (
            not isinstance(metrics, list)
            or not metrics
            or not all(isinstance(metric, str) and metric for metric in metrics)
        ):
            errors.append(f"{label}: metrics must be a non-empty string list")
        if "cache_glob" in entry:
            cache_error = _validate_cache_glob(entry["cache_glob"])
            if cache_error is not None:
                errors.append(f"{label}: {cache_error}")
        source_error = _validate_source(entry.get("source_url"))
        if source_error is not None:
            errors.append(f"{label}: {source_error}")
        if status == "implemented":
            adapter_error = _validate_adapter(entry.get("adapter"))
            if adapter_error is not None:
                errors.append(f"{label}: {adapter_error}")
        if entry.get("tier") == "paper_core":
            validator = entry.get("validator")
            if not isinstance(validator, str) or not validator.strip():
                errors.append(f"{label}: paper_core entry requires a validator")
            else:
                validator_error = _resolve_python_reference(validator)
                if validator_error is not None:
                    errors.append(f"{label}: invalid cache validator: {validator_error}")
                else:
                    resolved = _load_python_reference(validator)
                    if not callable(resolved):
                        errors.append(f"{label}: cache validator must be callable")

    paper_core = [entry for entry in datasets if entry.get("tier") == "paper_core"]
    if not paper_core:
        errors.append(f"{track}: at least one paper_core dataset is required")
    else:
        computed_complete = all(entry.get("status") == "implemented" for entry in paper_core)
        if registry.get("paper_suite_complete") is not computed_complete:
            errors.append(
                f"{track}: paper_suite_complete must be {str(computed_complete).lower()} "
                "because it is derived only from paper_core code status"
            )
    return errors


def _validate_cache(
    entry: dict[str, Any],
    data_root: Path | None,
    *,
    seeds: tuple[int, ...] | None = None,
    data_seeds: tuple[int, ...] | None = None,
    split_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    resolved_data_seeds = data_seeds if data_seeds is not None else seeds or (0,)
    resolved_split_seeds = split_seeds if split_seeds is not None else resolved_data_seeds
    if entry.get("cache_glob") is None:
        return {"cache_status": "not_applicable", "cache_detail": None}
    if data_root is None:
        return {"cache_status": "not_checked", "cache_detail": None}
    validator_reference = entry.get("validator")
    if not isinstance(validator_reference, str):
        return {
            "cache_status": "incomplete",
            "cache_detail": "registry entry has no read-only cache validator",
        }
    try:
        validator = _load_python_reference(validator_reference)
        metadata = validator(
            entry["id"],
            data_root,
            data_seeds=resolved_data_seeds,
            split_seeds=resolved_split_seeds,
        )
    except FileNotFoundError as error:
        return {"cache_status": "missing", "cache_detail": str(error)}
    except CacheIncompleteError as error:
        return {"cache_status": "incomplete", "cache_detail": str(error)}
    except CacheWrongRequestError as error:
        return {"cache_status": "wrong_request", "cache_detail": str(error)}
    except CacheCorruptError as error:
        return {"cache_status": "corrupt", "cache_detail": str(error)}
    except CacheValidationError as error:
        return {"cache_status": "corrupt", "cache_detail": str(error)}
    except (ImportError, ModuleNotFoundError) as error:
        return {
            "cache_status": "incomplete",
            "cache_detail": f"validation dependency unavailable: {error}",
        }
    except Exception as error:  # fail closed on an unexpected parser/validator error
        return {
            "cache_status": "corrupt",
            "cache_detail": f"{type(error).__name__}: {error}",
        }
    return {"cache_status": "valid", "cache_detail": metadata}


def readiness(
    registries: dict[str, dict[str, Any]],
    profile: str,
    *,
    data_root: Path | None = None,
    data_seeds: tuple[int, ...] = (0,),
    split_seeds: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    if profile != "paper":
        raise ValueError("only the full paper dataset profile is supported")
    tier = "paper_core"
    rows: list[dict[str, Any]] = []
    validation_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for track, registry in registries.items():
        for entry in registry["datasets"]:
            if entry["tier"] == tier:
                validation_key = (
                    str(entry.get("validator", "")),
                    str(entry.get("cache_glob", entry["id"])),
                )
                cache_result = validation_cache.get(validation_key)
                if cache_result is None:
                    cache_result = _validate_cache(
                        entry,
                        data_root,
                        data_seeds=data_seeds,
                        split_seeds=split_seeds,
                    )
                    validation_cache[validation_key] = cache_result
                rows.append(
                    {
                        "track": track,
                        "id": entry["id"],
                        "tier": entry["tier"],
                        "status": entry["status"],
                        "code_ready": entry["status"] == "implemented",
                        "data_policy": entry["data_policy"],
                        **cache_result,
                    }
                )
    return rows


def _parse_seeds(parser: argparse.ArgumentParser, value: str, option: str) -> tuple[int, ...]:
    try:
        seeds = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError:
        parser.error(f"{option} must be a comma-separated list of integers")
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error(f"{option} must contain at least one non-negative integer")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("paper",), default="paper")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--data-seeds",
        "--seeds",
        dest="data_seeds",
        default="0",
        help="comma-separated generated-data/cache seeds; --seeds is a compatibility alias",
    )
    parser.add_argument(
        "--split-seeds",
        help="comma-separated split/cache seeds; defaults to --data-seeds",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="require every selected cache to pass its read-only validator (requires --data-root)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.require_cache and args.data_root is None:
        parser.error("--require-cache requires --data-root")
    data_seeds = _parse_seeds(parser, args.data_seeds, "--data-seeds")
    split_seeds = (
        data_seeds
        if args.split_seeds is None
        else _parse_seeds(parser, args.split_seeds, "--split-seeds")
    )

    registries = {track: load_registry(track) for track in TRACKS}
    errors = [
        error
        for track, registry in registries.items()
        for error in validate_registry(track, registry)
    ]
    data_root = args.data_root.expanduser().resolve() if args.data_root is not None else None
    rows = readiness(
        registries,
        args.profile,
        data_root=data_root,
        data_seeds=data_seeds,
        split_seeds=split_seeds,
    )
    code_ready = not errors and all(row["code_ready"] for row in rows)
    cache_ready: bool | None = None
    if data_root is not None:
        cache_ready = all(row["cache_status"] in {"valid", "not_applicable"} for row in rows)
    ready = code_ready and (cache_ready is True if args.require_cache else True)
    payload = {
        "profile": args.profile,
        "ready": ready,
        "code_ready": code_ready,
        "require_cache": bool(args.require_cache),
        "cache_checked": data_root is not None,
        "cached_data_ready": cache_ready,
        "cache_validation": "content-and-request" if data_root is not None else "not_checked",
        "requested_seeds": list(data_seeds),
        "requested_seed_axes": {
            "data": list(data_seeds),
            "split": list(split_seeds),
        },
        "paper_benchmark_suite_complete": not errors
        and all(registry["paper_suite_complete"] for registry in registries.values()),
        "rows": rows,
        "errors": errors,
    }

    if args.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"dataset profile: {args.profile}")
        for row in rows:
            print(
                f"  {row['track']:18} {row['id']:36} "
                f"code={row['status']:11} cache={row['cache_status']}"
            )
        for error in errors:
            print(f"ERROR: {error}")
        if data_root is not None:
            print("CACHED DATA READY" if cache_ready else "CACHED DATA INCOMPLETE")
        print("READY" if ready else "NOT READY")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
