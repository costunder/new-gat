"""Read nine existing selected models and expose their exact frozen stages.

Only one already-trained seed is extracted from each selected seed pack. This
module never imports a training entry point, creates an optimizer, selects a new
learning rate, or modifies a source artifact. CPU transfers are provenance checks;
all model projections and propagation execute on the supplied CUDA graph.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch

from research.wedge_propagation.classification.common import (
    digest,
    file_sha256,
    read_json,
    source_manifest,
)
from research.wedge_propagation.classification.evaluation import model_state_hash
from research.wedge_propagation.classification.model import PackedClassifier, _lbar

CONDITIONS = ("mlp", "standard_gcn", "polynomial_2")
REPLAY_ATOL = 2e-6
REPLAY_RTOL = 2e-6
_GRAPH_TENSORS = (
    "x",
    "y",
    "train_mask",
    "val_mask",
    "test_mask",
    "edges",
    "paths",
    "degree",
    "qdiag",
    "sd",
    "sq",
    "gcn_edges",
    "gcn_weight",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def graph_digest(graph):
    """Same byte fingerprint as the original classification run, without training imports."""
    hasher = hashlib.sha256()
    for name in _GRAPH_TENSORS:
        value = getattr(graph, name).detach().cpu().contiguous()
        hasher.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class SelectedCheckpoint:
    dataset: str
    condition: str
    seed: int
    checkpoint: Path
    checkpoint_sha256: str
    graph_path: Path
    graph_sha256: str
    graph_digest: str
    path_chunk: int
    selected_epoch: int
    selected_validation_ce: float
    selected_validation_accuracy: float
    metadata: dict
    epochs: int
    original_metrics: tuple[dict, ...]
    artifact_hashes: dict[str, str]
    source_code_digest: str

    def record(self):
        return {
            "dataset": self.dataset,
            "condition": self.condition,
            "seed": self.seed,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "graph_path": str(self.graph_path),
            "graph_sha256": self.graph_sha256,
            "graph_digest": self.graph_digest,
            "path_chunk": self.path_chunk,
            "selected_epoch": self.selected_epoch,
            "selected_validation_ce": self.selected_validation_ce,
            "selected_validation_accuracy": self.selected_validation_accuracy,
            "original_metrics": list(self.original_metrics),
            "source_code_digest": self.source_code_digest,
        }


@dataclass(frozen=True)
class Stage:
    layer: int
    stage: str
    values: torch.Tensor

    @property
    def name(self):
        if self.layer < 0 or self.stage == "logits":
            return self.stage
        return f"layer_{self.layer}_{self.stage}"


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(
            reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
            f"invalid CSV header: {path}",
        )
        rows = []
        for supplied in reader:
            _require(
                None not in supplied and all(value is not None for value in supplied.values()),
                f"invalid CSV row: {path}",
            )
            row = {}
            for key, value in supplied.items():
                if value == "":
                    row[key] = None
                else:
                    try:
                        row[key] = int(value)
                    except ValueError:
                        try:
                            row[key] = float(value)
                        except ValueError:
                            row[key] = value
            rows.append(row)
    _require(rows, f"empty source table: {path}")
    return rows


def _verify_locked_lrs(config, tuning, selections):
    train = config["training"]
    expected = set(
        itertools.product(
            config["data"]["datasets"],
            config["conditions"],
            train["learning_rate_candidates"],
            train["tuning_seeds"],
        )
    )
    seen, buckets = set(), {}
    for row in tuning:
        key = row["dataset"], row["condition"], row["lr"], row["seed"]
        _require(
            key in expected and key not in seen and row.get("phase") == "tuning",
            "source tuning identity/coverage differs",
        )
        _require(
            not any("test" in name.lower() for name in row),
            "test metrics entered source learning-rate selection",
        )
        _require(
            math.isfinite(row["validation_ce"])
            and row["validation_ce"] >= 0
            and 0 <= row["validation_accuracy"] <= 1
            and 1 <= row["best_epoch"] <= train["epochs_per_run"]
            and row["training_epochs"] == row["optimizer_updates"] == train["epochs_per_run"],
            "source tuning validation/budget differs",
        )
        seen.add(key)
        buckets.setdefault(key[:3], []).append(row["validation_ce"])
    _require(seen == expected, "source tuning coverage is incomplete")
    locked = {}
    for row in selections:
        key = row["dataset"], row["condition"]
        _require(key not in locked, "duplicate locked learning-rate row")
        options = [
            (sum(buckets[(*key, lr)]) / len(buckets[(*key, lr)]), lr)
            for lr in train["learning_rate_candidates"]
        ]
        mean_ce, lr = min(options)
        _require(
            row["selected_lr"] == lr
            and row["mean_tuning_validation_ce"] == mean_ce
            and row["selection_scope"] == "validation_only_independent_tuning_seeds",
            "source locked learning-rate selection differs from saved validation",
        )
        locked[key] = lr
    _require(
        set(locked) == set(itertools.product(config["data"]["datasets"], config["conditions"])),
        "source locked learning-rate coverage is incomplete",
    )
    return locked


def discover(source_root, seed=11, profile="full", *, allow_debug=False):
    """Validate original scalar provenance, then identify ONLY the nine selected packs.

    ``profile='debug'`` is solely for the separate complete GPU fixture contract.
    Full uses all Cora/CiteSeer/PubMed nodes and saved seed 11; no fitting occurs.
    Recorded absolute server paths are rebased to hashed files within source_root.
    """
    if allow_debug:
        profile = "debug"
    _require(type(seed) is int and seed == 11, "this single-seed diagnostic is fixed to seed 11")
    _require(profile in ("full", "debug"), "explicit full/debug source profile required")
    root = Path(source_root).resolve()
    _require(
        root.is_dir() and not (root / "failure.json").exists(),
        "a completed classification source directory without failure is required",
    )
    hashes = {}

    def path(relative):
        relative = Path(relative)
        supplied = root / relative
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and supplied.resolve().is_relative_to(root)
            and supplied.is_file(),
            f"source file missing or escapes run directory: {relative}",
        )
        current = file_sha256(supplied)
        _require(
            hashes.get(str(supplied), current) == current,
            f"source artifact changed while reading: {relative}",
        )
        hashes[str(supplied)] = current
        return supplied

    def read(relative):
        return read_json(path(relative))

    config = read("config.json")
    from research.wedge_propagation.classification import model as source_model

    canonical = read_json(Path(source_model.__file__).with_name(f"config_{profile}.json"))
    _require(config == canonical, "saved complete classification configuration changed")
    source = read("source.json")
    actual_source = source_manifest()
    _require(
        source.get("sha256") == actual_source["sha256"]
        and source.get("code_digest") == actual_source["code_digest"] == digest(source["sha256"]),
        "classification implementation differs from the saved training source",
    )
    completion, contract, data_manifest = (
        read("completion.json"),
        read("contract.json"),
        read("data_manifest.json"),
    )
    scope = "full_public_split_classification" if profile == "full" else "DEBUG_pipeline_only"
    _require(
        completion.get("completed") is True
        and completion.get("profile") == profile
        and completion.get("scope") == scope
        and completion.get("actual_data") is (profile == "full")
        and completion.get("code_and_graphs_preserved") is True
        and completion.get("frozen_model_updates") == 0,
        "source study did not complete the original classification contract",
    )
    _require(
        contract.get("config_digest") == digest(config)
        and contract.get("source") == source
        and contract.get("data_manifest_digest") == digest(data_manifest)
        and contract.get("actual_data") is (profile == "full")
        and data_manifest.get("profile") == profile
        and data_manifest.get("all_nodes_edges_paths") is True
        and data_manifest.get("actual_citation_data") is (profile == "full"),
        "source configuration/data provenance differs",
    )
    coverage = read("coverage.json")
    _require(
        coverage == completion["coverage"] == contract["coverage"]
        and coverage.get("all_datasets_conditions_seeds_splits") is True,
        "source completed coverage differs",
    )
    for key in ("tuning_runs", "final_runs", "total_runs"):
        _require(coverage.get(key) == config["training"][key], f"source coverage differs: {key}")
    _require(
        coverage.get("contract_optimizer_updates") == config["training"]["total_updates"],
        "source original optimization budget differs",
    )
    selections = read("learning_rate_selection.json")
    locked = _verify_locked_lrs(config, _csv(path("tuning_validation.csv")), selections)
    selected_rows = _csv(path("final_validation_selection.csv"))
    selected = {}
    expected = set(
        itertools.product(
            config["data"]["datasets"], config["conditions"], config["training"]["final_seeds"]
        )
    )
    for row in selected_rows:
        key = row["dataset"], row["condition"], row["seed"]
        _require(
            key in expected
            and key not in selected
            and row.get("phase") == "final"
            and row["lr"] == locked[key[:2]]
            and row["training_epochs"]
            == row["optimizer_updates"]
            == config["training"]["epochs_per_run"],
            "source final validation selection identity/budget differs",
        )
        selected[key] = row
    _require(set(selected) == expected, "source final validation selection coverage is incomplete")
    unlock = read("test_evaluation_unlocked.json")
    _require(
        unlock.get("all_final_checkpoints_locked") is True
        and unlock["final_runs"] == len(expected)
        and unlock["selection_digest"] == digest(selected_rows)
        and unlock["learning_rate_selection_digest"] == digest(selections),
        "source selected-checkpoint lock differs",
    )
    original_rows = _csv(path("metrics.csv"))
    original = {}
    expected_metrics = {
        (*key, split) for key in expected for split in ("train", "validation", "test")
    }
    for row in original_rows:
        key = row["dataset"], row["condition"], row["seed"], row["split"]
        _require(
            key in expected_metrics
            and key not in original
            and math.isfinite(row["ce"])
            and row["ce"] >= 0
            and 0 <= row["accuracy"] <= 1,
            "source metrics identity/finiteness differs",
        )
        original[key] = row
    _require(set(original) == expected_metrics, "source metric coverage is incomplete")
    graph_records = read("graph_cache.json")
    descriptors = []
    for dataset, condition in itertools.product(config["data"]["datasets"], CONDITIONS):
        record = graph_records[dataset]
        recorded = str(record["path"])
        spelling = PureWindowsPath(recorded) if "\\" in recorded else PurePosixPath(recorded)
        _require(
            ".." not in spelling.parts
            and spelling.parts[-2:] == ("graphs", f"{dataset}.npz")
            and (spelling.is_absolute() or spelling.parts == ("graphs", f"{dataset}.npz")),
            "source recorded graph path is not the canonical local cache",
        )
        graph_path = path(Path("graphs") / f"{dataset}.npz")
        _require(file_sha256(graph_path) == record["sha256"], "source graph cache checksum differs")
        calibration = read(f"calibration/final-{dataset}-{condition}.json")
        choice = calibration["selection"]
        packed, chunk = choice["packed_runs"], choice["path_chunk"]
        _require(
            calibration["dataset"] == dataset
            and calibration["condition"] == condition
            and calibration["phase"] == "final"
            and calibration["config_digest"] == digest(config)
            and type(packed) is int
            and 0 < packed <= len(config["training"]["final_seeds"])
            and type(chunk) is int
            and chunk > 0,
            "source selected packing/calibration differs",
        )
        index = config["training"]["final_seeds"].index(seed)
        start = (index // packed) * packed
        seeds = config["training"]["final_seeds"][start : start + packed]
        lr = locked[(dataset, condition)]
        relative = Path("jobs") / "final" / dataset / condition / f"lr-{lr:g}" / f"pack-{start:02d}"
        metadata = {
            "dataset": dataset,
            "condition": condition,
            "seeds": seeds,
            "lr": lr,
            "phase": "final",
            "config_digest": digest(config),
            "code_digest": source["code_digest"],
            "graph_digest": record["content_digest"],
            "packed_runs": len(seeds),
            "path_chunk": chunk,
        }
        checkpoint = path(relative / "selected.pt")
        sidecar, done, pack_rows = (
            read(relative / "selected.json"),
            read(relative / "completion.json"),
            read(relative / "selected_validation.json"),
        )
        epochs = config["training"]["epochs_per_run"]
        _require(
            sidecar["metadata"] == metadata
            and sidecar["epoch"] == epochs
            and sidecar["sha256"] == file_sha256(checkpoint)
            and done.get("completed") is True
            and done["metadata"] == metadata
            and done["epochs_per_seed"] == epochs
            and done["rows"] == len(seeds)
            and done["selected_sha256"] == sidecar["sha256"]
            and pack_rows == {"rows": [selected[(dataset, condition, s)] for s in seeds]},
            "source selected checkpoint/manifest/completion differs",
        )
        row = selected[(dataset, condition, seed)]
        descriptors.append(
            SelectedCheckpoint(
                dataset,
                condition,
                seed,
                checkpoint,
                sidecar["sha256"],
                graph_path,
                record["sha256"],
                record["content_digest"],
                chunk,
                row["best_epoch"],
                row["validation_ce"],
                row["validation_accuracy"],
                metadata,
                epochs,
                tuple(
                    original[(dataset, condition, seed, split)]
                    for split in ("train", "validation", "test")
                ),
                {},
                source["code_digest"],
            )
        )
    for descriptor in descriptors:
        descriptor.artifact_hashes.update(hashes)
    assert_source_unchanged(descriptors)
    return config, descriptors


def assert_source_unchanged(descriptors):
    """Recheck only the identified artifacts and saved classification implementation."""
    hashes = {
        path: sha for descriptor in descriptors for path, sha in descriptor.artifact_hashes.items()
    }
    for path, expected in hashes.items():
        _require(file_sha256(path) == expected, f"source artifact changed: {path}")
    expected_codes = {descriptor.source_code_digest for descriptor in descriptors}
    _require(
        expected_codes == {source_manifest()["code_digest"]},
        "classification implementation changed during frozen trace",
    )


def load_frozen(graph, descriptor, config, seed=11):
    """Load best_state for one saved seed; never use last-epoch current_state."""
    _require(type(seed) is int and seed == descriptor.seed == 11, "only saved seed 11 is allowed")
    _require(graph.x.device.type == "cuda", "frozen model execution requires CUDA; no CPU fallback")
    _require(
        graph.name == descriptor.dataset and graph_digest(graph) == descriptor.graph_digest,
        "full source graph/features/masks fingerprint differs",
    )
    shape = config["data"]["expected_shapes"][graph.name]
    _require(
        all(graph.metadata.get(key) == value for key, value in shape.items()),
        "source whole-graph shape/public split metadata differs",
    )
    if config["profile"] == "full":
        from research.wedge_propagation.classification.data import PLANETOID_COMMIT, RAW_SHA256

        raw = {
            f"ind.{graph.name.lower()}.{part}": value
            for part, value in RAW_SHA256[graph.name].items()
        }
        _require(
            graph.metadata.get("actual_citation_data") is True
            and graph.metadata.get("debug") is False
            and graph.metadata.get("repository_commit") == PLANETOID_COMMIT
            and graph.metadata.get("raw_sha256") == raw,
            "whole source graph lacks pinned official raw-data provenance",
        )
    _require(
        digest(config) == descriptor.metadata["config_digest"], "frozen model configuration differs"
    )
    _require(
        file_sha256(descriptor.checkpoint) == descriptor.checkpoint_sha256,
        "selected checkpoint bytes changed",
    )
    payload = torch.load(descriptor.checkpoint, map_location="cpu", weights_only=False)
    _require(
        payload["metadata"] == descriptor.metadata and payload["epoch"] == descriptor.epochs,
        "selected payload identity/complete budget differs",
    )
    seeds = descriptor.metadata["seeds"]
    expected = set(itertools.product(seeds, range(1, descriptor.epochs + 1)))
    seen, by_seed = set(), {value: [] for value in seeds}
    for row in payload["history"]:
        key = row["seed"], row["epoch"]
        _require(
            key in expected
            and key not in seen
            and all(
                row[name] == descriptor.metadata[name]
                for name in ("dataset", "condition", "phase", "lr")
            )
            and math.isfinite(row["validation_ce"])
            and row["validation_ce"] >= 0
            and 0 <= row["validation_accuracy"] <= 1,
            "selected checkpoint history identity/coverage differs",
        )
        seen.add(key)
        by_seed[row["seed"]].append(row)
    _require(seen == expected, "selected checkpoint omitted an original training epoch")
    for index, value in enumerate(seeds):
        best = min(
            by_seed[value],
            key=lambda row: (row["validation_ce"], -row["validation_accuracy"], row["epoch"]),
        )
        _require(
            (payload["best_epoch"][index], payload["best_ce"][index], payload["best_acc"][index])
            == (best["epoch"], best["validation_ce"], best["validation_accuracy"]),
            "selected state metadata differs from original validation selection",
        )
    index = seeds.index(seed)
    _require(
        (payload["best_epoch"][index], payload["best_ce"][index], payload["best_acc"][index])
        == (
            descriptor.selected_epoch,
            descriptor.selected_validation_ce,
            descriptor.selected_validation_accuracy,
        ),
        "requested seed selected state disagrees with root source table",
    )
    model = PackedClassifier(
        descriptor.condition,
        graph.num_features,
        graph.num_classes,
        [seed],
        hidden=config["backbone"]["hidden_dim"],
        gate_hidden=config["gate"]["hidden_dim"],
        dropout=config["backbone"]["dropout_probability"],
        path_chunk=descriptor.path_chunk,
        checkpoint_paths=True,
        dataset_name=graph.name,
    ).to(device=graph.x.device, dtype=graph.x.dtype)
    state = payload["best_state"]
    _require(set(state) == set(model.state_dict()), "selected state model keys differ")
    selected_state = {}
    for name, value in state.items():
        _require(
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == len(seeds)
            and value.dtype == model.state_dict()[name].dtype
            and torch.isfinite(value).all(),
            f"selected seed state shape/precision/finiteness differs: {name}",
        )
        selected_state[name] = value[index : index + 1]
    model.load_state_dict(selected_state, strict=True)
    model.requires_grad_(False).eval()
    return model


def gcn_propagate(graph, z):
    """Apply the original GCN self-loop normalized P to Z, without W or ReLU."""
    _require(
        z.ndim == 3
        and z.shape[0] == 1
        and z.shape[1] == graph.num_nodes
        and z.device == graph.x.device
        and z.device.type == "cuda"
        and z.dtype == graph.x.dtype,
        "GCN replay requires one full CUDA seed signal [1,N,F]",
    )
    source = z.index_select(1, graph.gcn_edges[0]) * graph.gcn_weight[None, :, None]
    return z.new_zeros(z.shape).index_add(1, graph.gcn_edges[1], source)


def _replay_evidence(actual, expected, description):
    error = (actual - expected).abs().amax()
    valid = torch.allclose(actual, expected, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
    _require(valid, f"{description} differs beyond declared FP32 replay tolerance")
    return {
        "within_tolerance": valid,
        "exact_equal": torch.equal(actual, expected),
        "max_abs_error": float(error),
        "atol": REPLAY_ATOL,
        "rtol": REPLAY_RTOL,
        "note": "same model arithmetic; repeated CUDA scatter sums need not be bitwise identical",
    }


def trace_stages(model, graph):
    """Return exact eval stages and verify against the original model forward.

    Seven identifiers include an honest aggregated/logits alias on layer 1.
    The before-projection stages are input (layer 0) and activated (layer 0 for
    the next layer). Only layer 0 has ReLU, as in the selected source model.
    """
    _require(
        model.condition in CONDITIONS and tuple(model.seeds) == (11,),
        "trace accepts only the three selected single-seed conditions",
    )
    _require(
        graph.x.device.type == "cuda"
        and not model.training
        and not any(parameter.requires_grad for parameter in model.parameters()),
        "trace requires a frozen eval CUDA model",
    )
    before = model_state_hash(model)
    with torch.inference_mode():
        original, _ = model(graph)
        h = graph.x[None]
        stages = [Stage(-1, "input", h)]
        for layer in range(2):
            z = torch.bmm(h, model.projections[layer])
            stages.append(Stage(layer, "projected", z))
            if model.condition == "standard_gcn":
                aggregated = gcn_propagate(graph, z)
            elif model.condition == "polynomial_2":
                first = _lbar(graph, z)
                second = _lbar(graph, first)
                alpha, beta = model._coefficients(layer, z)
                aggregated = z - alpha[:, None, None] * first - beta[:, None, None] * second
            else:
                aggregated = z
            stages.append(Stage(layer, "aggregated", aggregated))
            h = aggregated.relu() if layer == 0 else aggregated
            stages.append(Stage(layer, "activated" if layer == 0 else "logits", h))
        model.trace_replay_evidence = _replay_evidence(h, original, "staged selected eval logits")
        _require(
            all(torch.isfinite(stage.values).all() for stage in stages), "nonfinite frozen stage"
        )
    _require(
        model_state_hash(model) == before, "selected frozen model parameters or buffers changed"
    )
    return stages


def operator_replays(model, graph, stages):
    """Replay common GCN P and P² on each model's fixed projected Z.

    The identity I·Z is the already-returned projected stage. No projection,
    nonlinearity, fitting or new checkpoint selection is included in either
    replay. For Polynomial-2 this common P replay is distinct from its trained
    I-alpha*L_bar-beta*L_bar² aggregation, which trace_stages retains separately.
    """
    _require(
        model.condition in CONDITIONS
        and tuple(model.seeds) == (11,)
        and not model.training
        and not any(p.requires_grad for p in model.parameters()),
        "operator replay requires the selected frozen single-seed model",
    )
    lookup = {(stage.layer, stage.stage): stage for stage in stages}
    _require(len(lookup) == len(stages), "duplicate frozen stage identities")
    before = model_state_hash(model)
    replay, evidence = [], {}
    with torch.inference_mode():
        for layer in range(2):
            z = lookup[(layer, "projected")].values
            first = gcn_propagate(graph, z)
            second = gcn_propagate(graph, first)
            replay.extend((Stage(layer, "replay_P", first), Stage(layer, "replay_P2", second)))
            if model.condition == "standard_gcn":
                evidence[f"layer_{layer}_P_matches_actual_aggregation"] = _replay_evidence(
                    first, lookup[(layer, "aggregated")].values, "GCN P replay"
                )
            if model.condition == "mlp":
                _require(
                    torch.equal(z, lookup[(layer, "aggregated")].values),
                    "MLP aggregation must be exact identity",
                )
            _require(
                torch.isfinite(first).all() and torch.isfinite(second).all(),
                "nonfinite common GCN operator replay",
            )
    _require(model_state_hash(model) == before, "frozen model changed during operator replay")
    model.operator_replay_evidence = evidence
    return replay


__all__ = [
    "CONDITIONS",
    "SelectedCheckpoint",
    "Stage",
    "discover",
    "graph_digest",
    "load_frozen",
    "trace_stages",
    "operator_replays",
    "gcn_propagate",
    "assert_source_unchanged",
    "model_state_hash",
]
