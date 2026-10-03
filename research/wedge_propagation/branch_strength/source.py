"""Read a completed Experiment 4 without training or changing its artifacts.

Every final seed and every original condition is required. Checkpoints are read
on CPU, validated against the public run contract, and only their selected model
states are retained. Recorded server paths are rebased to the identical, hashed
files inside the supplied run directory; external paths are never read.
"""

from __future__ import annotations

import csv
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch

from ..classification.common import digest, file_sha256, read_json, source_manifest
from ..classification.data import PLANETOID_COMMIT, RAW_BASE, RAW_SHA256, GraphData, load_graph
from ..classification.evaluation import _tensor_hash
from ..classification.study import (
    _graph_digest,
    _load_manifests,
    build_jobs,
    read_config,
    select_learning_rates,
    verify_coverage,
    verify_frozen_coverage,
)
from ..classification.training import make_model


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class FrozenPack:
    dataset: str
    condition: str
    seeds: tuple[int, ...]
    lr: float
    path_chunk: int
    checkpoint: Path
    metadata: dict[str, Any]
    best_state: dict[str, torch.Tensor]
    best_epoch: tuple[int, ...]
    best_ce: tuple[float, ...]
    best_acc: tuple[float, ...]


@dataclass(frozen=True)
class SourceRun:
    directory: Path
    config: dict[str, Any]
    source: dict[str, Any]
    completion: dict[str, Any]
    contract: dict[str, Any]
    graphs: dict[str, GraphData]
    graph_records: dict[str, Any]
    manifests: dict[str, list[dict[str, Any]]]
    packs: list[FrozenPack]
    selections: list[dict[str, Any]]
    selected_rows: list[dict[str, Any]]
    original_rows: list[dict[str, Any]]
    hashes: dict[str, str]

    @property
    def learned_packs(self):
        return [pack for pack in self.packs if pack.condition.startswith("learned_wedge")]


class _Reader:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.hashes = {}

    def path(self, relative):
        relative = Path(relative)
        _require(not relative.is_absolute() and ".." not in relative.parts, "source path escape")
        path = self.directory / relative
        _require(path.resolve().is_relative_to(self.directory), f"source path escape: {relative}")
        _require(path.is_file(), f"required Experiment 4 source file is missing: {relative}")
        key = relative.as_posix()
        current = file_sha256(path)
        _require(self.hashes.get(key, current) == current, f"source changed while reading: {key}")
        self.hashes[key] = current
        return path

    def json(self, relative):
        return read_json(self.path(relative))

    def csv(self, relative):
        with self.path(relative).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(
                reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                f"invalid or duplicate CSV headers: {relative}",
            )
            rows = []
            for row in reader:
                _require(
                    None not in row and all(v is not None for v in row.values()), "invalid CSV row"
                )
                rows.append({key: _csv_value(value) for key, value in row.items()})
        _require(rows, f"empty source CSV: {relative}")
        return rows


def _csv_value(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _graph_path(recorded, name):
    """Permit copied server paths only for the exact canonical local graph file."""
    _require(isinstance(recorded, str), "graph cache path must be a string")
    windows, posix = PureWindowsPath(recorded), PurePosixPath(recorded)
    path = windows if windows.is_absolute() or "\\" in recorded else posix
    _require(".." not in path.parts, "recorded graph path escape")
    expected = ("graphs", f"{name}.npz")
    _require(path.parts[-2:] == expected, "recorded graph cache has noncanonical destination")
    _require(path.is_absolute() or path.parts == expected, "relative graph cache path escape")
    return Path(*expected)


def _validate_history(payload, metadata, selected_rows, epochs):
    seeds = metadata["seeds"]
    history = payload["history"]
    expected = set(itertools.product(seeds, range(1, epochs + 1)))
    seen, by_seed = set(), {seed: [] for seed in seeds}
    for row in history:
        key = row["seed"], row["epoch"]
        _require(key in expected and key not in seen, "checkpoint history coverage mismatch")
        _require(
            all(row[field] == metadata[field] for field in ("dataset", "condition", "phase", "lr")),
            "checkpoint history identity mismatch",
        )
        ce, acc = row["validation_ce"], row["validation_accuracy"]
        _require(
            math.isfinite(ce) and ce >= 0 and 0 <= acc <= 1, "invalid checkpoint validation history"
        )
        seen.add(key)
        by_seed[row["seed"]].append(row)
    _require(seen == expected, "checkpoint history omitted training epochs/seeds")
    for index, seed in enumerate(seeds):
        chosen = min(
            by_seed[seed],
            key=lambda row: (row["validation_ce"], -row["validation_accuracy"], row["epoch"]),
        )
        row = selected_rows[(metadata["dataset"], metadata["condition"], seed)]
        _require(
            payload["best_epoch"][index] == chosen["epoch"] == row["best_epoch"]
            and payload["best_ce"][index] == chosen["validation_ce"] == row["validation_ce"]
            and payload["best_acc"][index]
            == chosen["validation_accuracy"]
            == row["validation_accuracy"],
            "checkpoint selected validation state disagrees with source selection/history",
        )


def _load_pack(reader, relative, metadata, graph, config, selected_rows):
    checkpoint = reader.path(relative / "selected.pt")
    sidecar = reader.json(relative / "selected.json")
    completion = reader.json(relative / "completion.json")
    rows = reader.json(relative / "selected_validation.json")
    epochs, seeds = config["training"]["epochs_per_run"], metadata["seeds"]
    _require(
        sidecar["sha256"] == file_sha256(checkpoint), "selected checkpoint content hash mismatch"
    )
    _require(
        sidecar["metadata"] == metadata and sidecar["epoch"] == epochs,
        "selected checkpoint metadata/epoch mismatch",
    )
    _require(
        completion.get("completed") is True
        and completion.get("metadata") == metadata
        and completion.get("selected_sha256") == sidecar["sha256"]
        and completion.get("epochs_per_seed") == epochs
        and completion.get("rows") == len(seeds),
        "selected pack completion mismatch",
    )
    expected_rows = [
        selected_rows[(metadata["dataset"], metadata["condition"], seed)] for seed in seeds
    ]
    _require(rows == {"rows": expected_rows}, "selected pack validation rows mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(
        payload["metadata"] == metadata and payload["epoch"] == epochs,
        "selected payload identity/epoch mismatch",
    )
    _require(
        all(len(payload[field]) == len(seeds) for field in ("best_epoch", "best_ce", "best_acc")),
        "selected checkpoint seed-state coverage mismatch",
    )
    _validate_history(payload, metadata, selected_rows, epochs)
    state = payload["best_state"]
    _require(isinstance(state, dict) and state, "selected model state is empty")
    for name, value in state.items():
        _require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.ndim > 0
            and value.shape[0] == len(seeds),
            f"selected seed axis/CPU tensor mismatch: {name}",
        )
        _require(
            not value.is_floating_point() or torch.isfinite(value).all(),
            f"nonfinite selected parameter: {name}",
        )
    # Constructing the model verifies every key/shape. No optimizer is constructed.
    model = make_model(graph, metadata["condition"], seeds, config, metadata["path_chunk"])
    expected_state = model.state_dict()
    _require(
        set(state) == set(expected_state)
        and all(value.dtype == expected_state[name].dtype for name, value in state.items()),
        "selected model key/precision mismatch",
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"selected model shape/key mismatch: {checkpoint}") from error
    return FrozenPack(
        graph.name,
        metadata["condition"],
        tuple(seeds),
        metadata["lr"],
        metadata["path_chunk"],
        checkpoint,
        metadata,
        state,
        tuple(payload["best_epoch"]),
        tuple(payload["best_ce"]),
        tuple(payload["best_acc"]),
    )


def load_source(run_dir, profile="full"):
    """Validate and read every final model from a complete, immutable source run."""
    started = time.perf_counter()
    reader = _Reader(run_dir)
    _require(reader.directory.is_dir(), "Experiment 4 source directory does not exist")
    _require(
        not (reader.directory / "failure.json").exists(),
        "failed Experiment 4 source is not eligible",
    )
    config = read_config(reader.path("config.json"), profile)
    canonical = read_json(Path(__file__).parents[1] / "classification" / f"config_{profile}.json")
    _require(config == canonical, "source config differs from the canonical complete profile")
    source = reader.json("source.json")
    current = source_manifest()
    _require(
        source.get("sha256") == current["sha256"]
        and source.get("code_digest") == digest(source["sha256"]) == current["code_digest"],
        "Experiment 4 implementation source code/hash changed",
    )
    completion, contract = reader.json("completion.json"), reader.json("contract.json")
    _require(
        completion.get("completed") is True
        and completion.get("profile") == profile
        and completion.get("scope")
        == ("full_public_split_classification" if profile == "full" else "DEBUG_pipeline_only")
        and completion.get("actual_data") is (profile == "full")
        and completion.get("frozen_model_updates") == 0
        and completion.get("code_and_graphs_preserved") is True,
        "source must be a completed full Experiment 4 or explicitly completed DEBUG run",
    )
    data_manifest = reader.json("data_manifest.json")
    _require(
        contract.get("config_digest") == digest(config)
        and contract.get("source") == source
        and contract.get("data_manifest_digest") == digest(data_manifest)
        and contract.get("actual_data") is (profile == "full")
        and data_manifest.get("profile") == profile
        and data_manifest.get("data_source") == config["data_source"]
        and data_manifest.get("actual_citation_data") is (profile == "full")
        and data_manifest.get("all_nodes_edges_paths") is True,
        "source provenance/data contract mismatch",
    )
    original_rows = reader.csv("metrics.csv")
    tuning_rows = reader.csv("tuning_validation.csv")
    selections = reader.json("learning_rate_selection.json")
    _require(
        select_learning_rates(config, tuning_rows) == selections,
        "locked learning-rate selections changed",
    )
    selected_rows = reader.csv("final_validation_selection.csv")
    jobs = build_jobs(config, "final", selections)
    expected = {
        (job["dataset"], job["condition"], seed): job["lr"] for job in jobs for seed in job["seeds"]
    }
    selected_by_key = {}
    epochs = config["training"]["epochs_per_run"]
    for row in selected_rows:
        key = row["dataset"], row["condition"], row["seed"]
        _require(
            key in expected and key not in selected_by_key,
            "final selection model/seed coverage mismatch",
        )
        _require(
            row["lr"] == expected[key]
            and row["phase"] == "final"
            and row["training_epochs"] == epochs
            and row["optimizer_updates"] == epochs,
            "final selection learning-rate/training budget mismatch",
        )
        selected_by_key[key] = row
    _require(set(selected_by_key) == set(expected), "source omitted final model/seed selections")
    unlock = reader.json("test_evaluation_unlocked.json")
    _require(
        unlock.get("all_final_checkpoints_locked") is True
        and unlock.get("final_runs") == len(expected)
        and unlock.get("selection_digest") == digest(selected_rows)
        and unlock.get("learning_rate_selection_digest") == digest(selections),
        "source final checkpoint/test unlock changed",
    )
    coverage = verify_coverage(config, tuning_rows, original_rows)
    evaluated = {
        name: reader.csv(filename)
        for name, filename in (
            ("intervention_rows", "interventions.csv"),
            ("scale_rows", "scale.csv"),
            ("gate_rows", "gate_diagnostics.csv"),
        )
    }
    coverage.update(verify_frozen_coverage(config, evaluated))
    _require(
        coverage == reader.json("coverage.json") == completion["coverage"] == contract["coverage"],
        "source completed result coverage changed",
    )
    _require(
        contract.get("new_optimizer_updates")
        == completion.get("new_optimizer_updates")
        == sum(row["new_optimizer_updates"] for row in tuning_rows + selected_rows),
        "source optimizer update accounting changed",
    )
    records = reader.json("graph_cache.json")
    names = config["data"]["datasets"]
    _require(
        set(records) == set(names) == set(data_manifest["datasets"]),
        "source graph dataset coverage mismatch",
    )
    graphs, manifests, rebased_records = {}, {}, {}
    for name in names:
        record = records[name]
        path = reader.path(_graph_path(record["path"], name))
        _require(file_sha256(path) == record["sha256"], "source graph NPZ hash mismatch")
        graph = load_graph(path, record["sha256"])
        shape = config["data"]["expected_shapes"][name]
        _require(
            graph.name == name
            and all(graph.metadata[key] == value for key, value in shape.items())
            and graph.metadata == data_manifest["datasets"][name]["statistics"]
            and record["content_digest"] == _graph_digest(graph)
            and record["num_paths"] == graph.paths.shape[1],
            "source graph geometry/features/masks/metadata changed",
        )
        if profile == "full":
            raw = data_manifest["raw"][name]
            expected_raw = {
                f"ind.{name.lower()}.{part}": value for part, value in RAW_SHA256[name].items()
            }
            _require(
                raw["sha256"] == expected_raw == graph.metadata.get("raw_sha256")
                and raw["repository_commit"]
                == graph.metadata.get("repository_commit")
                == PLANETOID_COMMIT
                and raw["official_raw_base"] == RAW_BASE
                and graph.metadata.get("actual_citation_data") is True
                and graph.metadata.get("debug") is False,
                "full source lacks the pinned official citation raw-data identity",
            )
        graphs[name] = graph
        rebased_records[name] = {**record, "path": str(path)}
        folder = Path("manifests") / name
        index = reader.json(folder / "manifest.json")
        count = config["evaluation"]["shuffle_and_random_manifests_per_dataset"]
        for row in index:
            _require(
                row["file"] == f"manifest-{row['index']:02d}.pt",
                "source manifest noncanonical file path",
            )
            reader.path(folder / row["file"])
        loaded = _load_manifests(reader.directory, name, count)
        for item in loaded:
            permutation, random = item["permutation"], item["random_rows"]
            paths = graph.paths.shape[1]
            _require(
                item["seed"] == 20261003
                and permutation.dtype == torch.long
                and torch.equal(permutation.sort().values, torch.arange(paths))
                and random["indices"].shape == random["coefficients"].shape == (4, paths)
                and random["indices"].dtype == torch.long
                and ((random["indices"] >= 0) & (random["indices"] < graph.num_nodes)).all()
                and torch.isfinite(random["coefficients"]).all()
                and item["sha256"]
                == _tensor_hash(permutation, random["indices"], random["coefficients"]),
                "source intervention manifest tensors/hash changed",
            )
        manifests[name] = loaded
    packs, expected_paths = [], set()
    for job in jobs:
        dataset, condition = job["dataset"], job["condition"]
        calibration = reader.json(f"calibration/final-{dataset}-{condition}.json")
        choice = calibration["selection"]
        _require(
            calibration["dataset"] == dataset
            and calibration["condition"] == condition
            and calibration["phase"] == "final"
            and calibration["config_digest"] == digest(config)
            and isinstance(choice["packed_runs"], int)
            and not isinstance(choice["packed_runs"], bool)
            and 0 < choice["packed_runs"] <= len(job["seeds"])
            and isinstance(choice["path_chunk"], int)
            and choice["path_chunk"] > 0,
            "source final calibration/packing changed",
        )
        packed = choice["packed_runs"]
        for start in range(0, len(job["seeds"]), packed):
            seeds = job["seeds"][start : start + packed]
            relative = Path("jobs") / job["job_key"] / f"pack-{start:02d}"
            metadata = {
                "dataset": dataset,
                "condition": condition,
                "seeds": seeds,
                "lr": job["lr"],
                "phase": "final",
                "config_digest": digest(config),
                "code_digest": source["code_digest"],
                "graph_digest": records[dataset]["content_digest"],
                "packed_runs": len(seeds),
                "path_chunk": choice["path_chunk"],
            }
            packs.append(
                _load_pack(reader, relative, metadata, graphs[dataset], config, selected_by_key)
            )
            expected_paths.add(relative / "selected.pt")
    actual_paths = {
        path.relative_to(reader.directory)
        for path in (reader.directory / "jobs/final").rglob("selected.pt")
    }
    _require(actual_paths == expected_paths, "source has missing/extra selected final packs")
    _require(
        sum(len(pack.seeds) for pack in packs) == config["training"]["final_runs"],
        "source final model states incomplete",
    )
    source_run = SourceRun(
        reader.directory,
        config,
        source,
        completion,
        contract,
        graphs,
        rebased_records,
        manifests,
        packs,
        selections,
        selected_rows,
        original_rows,
        dict(reader.hashes),
    )
    assert_unchanged(source_run)
    print(
        f"[source] Experiment4 profile={profile} all_final_models={len(expected)} "
        f"learned_models={sum(len(pack.seeds) for pack in source_run.learned_packs)} "
        f"graphs={len(graphs)} selected_packs={len(packs)} hashed_artifacts={len(reader.hashes)} "
        f"seconds={time.perf_counter() - started:.3f} optimizer_updates=0",
        flush=True,
    )
    return source_run


def assert_unchanged(source_run):
    """Check source artifact bytes and frozen implementation after diagnostics."""
    reader = _Reader(source_run.directory)
    _require(not (reader.directory / "failure.json").exists(), "source failure marker appeared")
    for relative, expected in source_run.hashes.items():
        _require(
            file_sha256(reader.path(relative)) == expected, f"source artifact changed: {relative}"
        )
    actual_checkpoints = {
        path.relative_to(reader.directory)
        for path in (reader.directory / "jobs/final").rglob("selected.pt")
    }
    expected_checkpoints = {
        pack.checkpoint.relative_to(reader.directory) for pack in source_run.packs
    }
    _require(
        actual_checkpoints == expected_checkpoints,
        "source selected final checkpoint file coverage changed",
    )
    _require(
        source_manifest()["code_digest"] == source_run.source["code_digest"],
        "Experiment 4 implementation source code changed",
    )
