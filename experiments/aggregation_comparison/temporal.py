"""Separate arxiv node-year visibility protocol; never an official OGB score.

Uses provided features as fixed covariates. Node years do not prove edge-time or
feature-time causality, and views of one graph are not independent graphs.
"""

import copy
import json
import time
from pathlib import Path

import torch

from chartgat.cache import atomic_write_json
from research.conductance_gat.benchmark_data import canonical_edges, sha256_file, tensor_hash
from research.conductance_gat.edge_selection.data import PreparedInputs as BaseInputs

NAME = "arxiv_node_year_views_v1"


def validate_years(payload, years):
    nodes = payload["graphs"][0]["x"].shape[0]
    if years.device.type != "cpu" or years.dtype != torch.long or years.shape != (nodes,):
        raise ValueError("node-year sidecar requires one CPU int64 year per original node")
    if not ((years >= 1800) & (years <= 2200)).all():
        raise ValueError("invalid publication year")
    splits = payload["splits"]
    if any(not splits[s].any() for s in ("train", "validation", "test")):
        raise ValueError("temporal protocol requires all three complete splits")
    if not torch.equal(sum(v.long() for v in splits.values()), torch.ones(nodes, dtype=torch.long)):
        raise ValueError("temporal protocol requires disjoint exhaustive official splits")
    if not (
        years[splits["train"]].max() < years[splits["validation"]].min()
        and years[splits["validation"]].max() < years[splits["test"]].min()
    ):
        raise ValueError("official split/year chronology is inconsistent")


def sidecar_path(data_root):
    return Path(data_root) / "ogbn-arxiv" / "node_year_v1.json"


def prepare_years(data_root, raw_dir):
    """Import existing OGB raw files after exact node/edge/label alignment checks."""
    import numpy as np

    from . import engine

    payload, protocol = engine.base.load_dataset("ogbn-arxiv", data_root, allow_download=False)
    path = sidecar_path(data_root)
    if path.exists():
        raise ValueError("node-year sidecar already exists; preserving immutable input")
    names = ("node-feat.csv.gz", "node-label.csv.gz", "edge.csv.gz", "node-year.csv.gz")
    paths = {name: Path(raw_dir) / name for name in names}
    if any(p.is_symlink() or not p.is_file() for p in paths.values()):
        raise ValueError("provide the existing complete OGB raw directory; no download fallback")
    before = {name: sha256_file(p) for name, p in paths.items()}
    x = torch.from_numpy(np.loadtxt(paths[names[0]], delimiter=",", dtype=np.float32))
    y = torch.from_numpy(np.loadtxt(paths[names[1]], delimiter=",", dtype=np.int64)).reshape(-1)
    edge = torch.from_numpy(
        np.loadtxt(paths[names[2]], delimiter=",", dtype=np.int64)
    ).T.contiguous()
    years = torch.from_numpy(np.loadtxt(paths[names[3]], delimiter=",", dtype=np.int64)).reshape(-1)
    graph = payload["graphs"][0]
    if not torch.equal(x, graph["x"]) or not torch.equal(y, graph["y"].reshape(-1)):
        raise ValueError("raw features/labels do not exactly match cached original node order")
    if not torch.equal(canonical_edges(edge, x.shape[0])[1], graph["incidence_edge_index"]):
        raise ValueError("raw edges differ from cached canonical graph")
    validate_years(payload, years)
    if before != {name: sha256_file(p) for name, p in paths.items()}:
        raise ValueError("raw files changed during sidecar verification")
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "dataset_sha256": protocol["data_sha256"],
            "split_sha256": protocol["split_sha256"],
            "raw_sha256": before,
            "alignment": "exact raw features, labels and canonical edges",
            "years_sha256": tensor_hash(years),
            "years": years.tolist(),
        },
    )
    return path


def attach(payload, protocol, data_root):
    path = sidecar_path(data_root)
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "temporal protocol needs verified node_year_v1.json; old cache has no years"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 1
        or document.get("dataset_sha256") != protocol["data_sha256"]
        or document.get("split_sha256") != protocol["split_sha256"]
        or document.get("alignment") != "exact raw features, labels and canonical edges"
        or set(document.get("raw_sha256", {}))
        != {"node-feat.csv.gz", "node-label.csv.gz", "edge.csv.gz", "node-year.csv.gz"}
    ):
        raise ValueError("node-year sidecar identity/alignment mismatch")
    if any(type(year) is not int for year in document["years"]):
        raise ValueError("node years must be integers")
    years = torch.tensor(document["years"], dtype=torch.long)
    validate_years(payload, years)
    if tensor_hash(years) != document["years_sha256"]:
        raise ValueError("node-year sidecar tensor hash mismatch")
    return {**payload, "node_year": years}, {
        **protocol,
        "visibility_protocol": NAME,
        "node_year_sidecar_sha256": sha256_file(path),
        "temporal_contract": {
            "train": "train-induced graph only",
            "validation": "train+validation-induced graph",
            "test": "cumulative node-year views; score each test node in its publication year",
            "parameters": "frozen after validation selection",
            "features": "provided fixed covariates; feature-time causality not established",
            "edge_time": "not observed; induced support uses original edges among visible nodes",
            "official_ogb_score": False,
            "independent_graph_generalization": False,
        },
    }


def view(payload, visible, selected):
    """Recompute all geometry on an induced graph; never carry full-graph degrees."""
    graph = payload["graphs"][0]
    ids = visible.nonzero(as_tuple=False).flatten()
    inverse = torch.full((visible.numel(),), -1, dtype=torch.long)
    inverse[ids] = torch.arange(ids.numel())
    edge = graph["incidence_edge_index"]
    local_edge = inverse[edge[:, visible[edge[0]] & visible[edge[1]]]]
    row = {
        "x": graph["x"][ids],
        "y": graph["y"][ids],
        "incidence_edge_index": local_edge,
        "edge_index": torch.cat((local_edge, local_edge.flip(0)), 1),
    }
    masks = {name: mask[ids] for name, mask in payload["splits"].items()}
    masks["validation"] = selected[ids]
    result = {
        "dataset": payload["dataset"],
        "classes": payload["classes"],
        "graphs": [row],
        "splits": masks,
    }
    identity = {
        "original_ids": tensor_hash(ids),
        "features": tensor_hash(row["x"]),
        "edges": tensor_hash(local_edge),
        "selected": tensor_hash(selected[ids]),
        "nodes": ids.numel(),
        "edge_count": local_edge.shape[1],
    }
    return result, ids, identity


class TemporalInputs:
    def __init__(self, payload, args):
        from .study_inputs import StudyInputs

        started = time.perf_counter()
        validate_years(payload, payload["node_year"])
        train_mask, validation_mask = (payload["splits"][s] for s in ("train", "validation"))
        train, self.train_ids, train_identity = view(
            payload, train_mask, torch.zeros_like(train_mask)
        )
        validation, self.val_ids, val_identity = view(
            payload, train_mask | validation_mask, validation_mask
        )
        factory = StudyInputs if args.complete_supervised_passes else BaseInputs
        self.train = factory(train, args)
        full = copy.deepcopy(args)
        full.sampling = "full"
        self.validation = BaseInputs(validation, full)
        self.view_identity = {"protocol": NAME, "train": train_identity, "validation": val_identity}
        self.plan_preparation_seconds = time.perf_counter() - started

    def __getattr__(self, name):
        return getattr(self.train, name)

    @property
    def validation_count(self):
        return self.validation.validation_count

    @property
    def validation_record(self):
        return self.validation.validation_record

    @property
    def provenance(self):
        return [
            {
                "views": self.view_identity,
                "training": self.train.provenance,
                "validation": self.validation.provenance,
            }
        ]

    @property
    def indices(self):
        return {
            "train": self.train.indices["train"],
            "validation": self.validation.indices["validation"],
        }

    def metadata(self):
        return {
            **self.train.metadata(),
            "provenance": self.provenance,
            "validation_count": self.validation_count,
            "visibility": self.view_identity,
            "topology_preparation_seconds": self.plan_preparation_seconds,
        }

    def validation_batches(self, device):
        yield from self.validation.validation_batches(device)


class TemporalTestInputs:
    """One cumulative graph per year; these views overlap and must not be merged."""

    def __init__(self, payload, args):
        self.payload, self.args = payload, copy.deepcopy(args)
        self.args.sampling = "full"
        self.indices = {"test": payload["splits"]["test"].nonzero().flatten()}
        self.view_evidence = []

    def validation_batches(self, device):
        years, test = self.payload["node_year"], self.payload["splits"]["test"]
        self.view_evidence = []
        for year in years[test].unique(sorted=True).tolist():
            selected = test & (years == year)
            data, ids, identity = view(self.payload, years <= year, selected)
            inputs = BaseInputs(data, self.args)
            self.view_evidence.append({"year": year, **identity})
            for batch in inputs.validation_batches(device):
                batch.graph.temporal_original_ids = ids.to(device)
                yield batch


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify existing raw arxiv node-year data; no downloads"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_years(args.data_root, args.raw_dir))


if __name__ == "__main__":
    main()
